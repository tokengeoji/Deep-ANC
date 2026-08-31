"""모델 검증 — 파라미터 규모, 인과성, 스트리밍/오프라인 등가성, GLSTM 이중 경로."""

import numpy as np
import subprocess
import sys

import pytest
import torch
import yaml

from deep_anc.config import REPO_ROOT
from deep_anc.models import build_model
from deep_anc.models.glstm import GLSTM
from deep_anc.models.hybrid_anc import parameter_count
from deep_anc.models.streaming import ExportWrapper, flatten_states, state_names
from deep_anc.model_input import (
    canonical_stage1_model_input_contract,
    validate_stage1_ref_only_tensor,
)


def _load(name):
    with open(REPO_ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize(
    "name,lo,hi,expected",
    [
        ("model_tiny", 0.9e6, 1.5e6, 1_164_809),
        ("model_tiny_attn", 0.9e6, 1.5e6, 1_231_369),
        ("model_tiny_long", 0.9e6, 1.5e6, 1_301_771),
        ("model_tiny_long_attn", 0.9e6, 1.5e6, 1_368_331),
        ("model_base", 5.0e6, 7.0e6, 5_994_512),
    ],
)
def test_parameter_budget(name, lo, hi, expected):
    model = build_model(_load(name))
    n = parameter_count(model)
    assert lo < n < hi, f"{name}: {n/1e6:.2f}M — 설계 예산 밖"
    assert n == expected, f"{name}: 문서화된 파라미터 수 {expected:,}와 불일치"


@pytest.mark.parametrize(
    "name",
    [
        "model_tiny",
        "model_tiny_attn",
        "model_tiny_long",
        "model_tiny_long_attn",
        "model_base",
    ],
)
def test_causality(name):
    torch.manual_seed(0)
    model = build_model(_load(name)).eval()
    x = torch.randn(1, 2, 128 * 12) * 0.02
    x2 = x.clone()
    change = 128 * 10
    x2[..., change:] += 1.0
    with torch.no_grad():
        y1, y2 = model(x), model(x2)
    # 미래 입력 변경은 그 이전 출력에 영향을 줄 수 없다
    assert torch.equal(y1[..., :change], y2[..., :change])


@pytest.mark.parametrize(
    "name",
    [
        "model_tiny",
        "model_tiny_attn",
        "model_tiny_long",
        "model_tiny_long_attn",
        "model_base",
    ],
)
def test_streaming_equivalence(name):
    torch.manual_seed(1)
    model = build_model(_load(name)).eval()
    x = torch.randn(1, 2, 128 * 16) * 0.02
    with torch.no_grad():
        y_off = model(x)
        states = model.init_states(1, "cpu")
        outs = []
        for i in range(0, x.shape[-1], 256):
            yb, states = model.streaming_step(x[..., i : i + 256], states)
            outs.append(yb)
        y_str = torch.cat(outs, dim=-1)
    assert (y_off - y_str).abs().max().item() < 1e-5


def test_tiny_ref_only_input_is_offline_streaming_equivalent_and_fail_closed():
    torch.manual_seed(17)
    model = build_model(_load("model_tiny")).eval()
    reference = torch.randn(1, 1, 256 * 4) * 0.02
    x = torch.cat([reference, torch.zeros_like(reference)], dim=1)
    contract = canonical_stage1_model_input_contract()
    validate_stage1_ref_only_tensor(x, contract, label="ref-only fixture")
    with torch.no_grad():
        offline = model(x)
        states = model.init_states(1, "cpu")
        streamed = []
        for start in range(0, x.shape[-1], 256):
            block, states = model.streaming_step(x[..., start : start + 256], states)
            streamed.append(block)
    assert (offline - torch.cat(streamed, dim=-1)).abs().max().item() < 1e-5

    leaked = x.clone()
    leaked[:, 1, 100] = 1.0e-12
    with pytest.raises(ValueError, match="ERR feature"):
        validate_stage1_ref_only_tensor(leaked, contract, label="leaked fixture")

    bad_ref = x.clone()
    bad_ref[:, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="x_ref.*NaN/Inf"):
        validate_stage1_ref_only_tensor(bad_ref, contract, label="bad ref fixture")

    bad_err = x.clone()
    bad_err[:, 1, 0] = float("inf")
    with pytest.raises(ValueError, match="ERR feature"):
        validate_stage1_ref_only_tensor(bad_err, contract, label="bad err fixture")


def test_model_input_contract_import_does_not_eager_load_torch():
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            (
                "import sys; import deep_anc.model_input; "
                "raise SystemExit(1 if 'torch' in sys.modules else 0)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_glstm_offline_vs_manual():
    """nn.LSTM(학습)과 수동 셀(export)의 등가성 [설계 교차검증 H1]."""
    torch.manual_seed(2)
    g = GLSTM(channels=64, groups=2, hidden_per_group=48).eval()
    x = torch.randn(2, 64, 20)
    with torch.no_grad():
        y_off = g(x)
        h, c = g.init_state(2, torch.device("cpu"))
        y_str, _, _ = g.streaming_forward(x, h, c)
    assert (y_off - y_str).abs().max().item() < 1e-5


def test_export_wrapper_roundtrip():
    torch.manual_seed(3)
    model = build_model(_load("model_base")).eval()
    wrapper = ExportWrapper(model, block_samples=256)
    states = flatten_states(model.init_states(1, "cpu"))
    names = state_names(model)
    assert len(states) == len(names)
    x = torch.randn(1, 2, 256) * 0.02
    with torch.no_grad():
        out = wrapper(x, *states)
    y, new_states = out[0], out[1:]
    assert y.shape == (1, 1, 256)
    assert len(new_states) == len(states)
    for old, new in zip(states, new_states):
        assert old.shape == new.shape          # 상태 shape 불변 (정적 그래프 요건)


def test_limiter_bound():
    model = build_model(_load("model_tiny")).eval()
    x = torch.randn(1, 2, 128 * 8) * 5.0        # 과대 입력
    with torch.no_grad():
        y = model(x)
    assert y.abs().max().item() <= model.limit + 1e-6
