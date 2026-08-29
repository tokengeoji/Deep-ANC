"""λ 가 항끼리의 그래디언트 예산 밖으로 **조용히 흘러가지 못하게** 한다.

왜 이 파일이 있는가
------------------
2026-08-06 감사: "λ_dnh 가 새 대역 구성에서 재교정되지 않았다 — 실측 그래디언트 비
**1333%**(설계 목표 20~40%)". 그런데 그 1333% 를 만든 계산은 저장소 어디에도 없었다.
숫자만 문서에 남고 재현 수단이 없으면, 다음 사람은 그 숫자를 믿거나 무시하는 것 말고
할 수 있는 일이 없다.

이제 측정은 :meth:`ANCLoss.gradient_budget` 하나이고(:meth:`forward` 와 **같은 항
dict** 를 쓴다), 이 파일이 출하 설정을 그 측정에 걸어 둔다. λ 가 예산을 벗어나면
학습을 돌리기 전에 여기서 실패한다.

중요한 범위
-----------
이 smoke는 **현재 Trainer가 쓰는 strict S(z), 유도된 정착 절단, 대역**을 그대로
쓴다. 고정 합성 파형의 ``∂L/∂y`` 비율은 정확성 회귀를 잡는 용도일 뿐, canonical
승인값 자체는 아니다. canonical 전에는 실제 A100 batch/model 출력 ``y``에 대한
output-gradient evidence와 ``loss_start_sample``을 campaign ledger에 함께 결속해야 한다.
이 비율은 model parameter-gradient라고 부르면 안 된다.

현재 ``frame`` 항은 별도 통제 실험에서 영출력 붕괴와 연관된 것으로 확인 중이다.
이 파일은 그 항을 승인하지 않으며, 손실 정의에서 누락되지 않았다는 사실만 점검한다.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from deep_anc.config import REPO_ROOT, load_yaml
from deep_anc.dsp.secondary_path import DifferentiableSecondaryPath, load_secondary_path
from deep_anc.dsp.timing import BandPlan, PlantSettle, handoff_samples_from_config
from deep_anc.losses.anc_loss import ANCLoss

SAMPLES = 1 << 14
BATCH = 4

# 고정 파형 smoke의 상한일 뿐 canonical 승인 범위는 campaign ledger가 실제 A100
# batch/model 출력으로 별도 검증한다.
BUDGET_HI = 0.4


def _active_contract() -> tuple[dict, object, int, int, PlantSettle, tuple[float, float]]:
    """실제 Trainer와 같은 strict S·정착·최적화 대역을 한 곳에서 유도한다."""

    train = load_yaml(REPO_ROOT / "configs" / "train_pretrain.yaml")
    duct = load_yaml(REPO_ROOT / train["duct_config"])
    data = load_yaml(REPO_ROOT / train["data_config"])
    sp = load_secondary_path(REPO_ROOT / duct["secondary_path"]["npz"])
    fs = int(data["sample_rate"])
    handoff = handoff_samples_from_config(duct)
    settle = PlantSettle.derive(
        secondary_delay_samples=int(sp.delay_samples),
        handoff_samples=handoff,
        fir_taps=int(sp.fir.size),
        sample_rate=fs,
    )
    trusted = BandPlan.resolve(
        plant_trusted_band_hz=sp.trusted_band_hz(), duct_cfg=duct, sample_rate=fs
    ).optimize.as_tuple()
    return duct, sp, fs, handoff, settle, trusted


def _plant() -> DifferentiableSecondaryPath:
    """현재 학습이 실제로 쓰는 strict 실측 S(z)를 쓴다."""

    _, sp, _, handoff, _, _ = _active_contract()
    return DifferentiableSecondaryPath(sp, handoff_extra_samples=handoff)


def _band_signal(lo: float, hi: float, seed: int, *, fs: int) -> torch.Tensor:
    freqs = np.fft.rfftfreq(SAMPLES, 1.0 / fs)
    rng = np.random.default_rng(seed)
    spec = np.zeros((BATCH, freqs.size), dtype=complex)
    mask = (freqs >= lo) & (freqs <= hi)
    spec[:, mask] = rng.normal(size=(BATCH, int(mask.sum()))) + 1j * rng.normal(
        size=(BATCH, int(mask.sum()))
    )
    wave = np.fft.irfft(spec, n=SAMPLES)
    wave = 0.05 * wave / (np.abs(wave).max(axis=-1, keepdims=True) + 1e-12)
    return torch.from_numpy(wave).float().unsqueeze(1)


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    """학습 중간을 닮은 (y, d). ``y=0`` 이면 dnh 가 0 이라 예산이 정의되지 않는다."""

    *_, fs, _handoff, _settle, trusted = _active_contract()
    d = _band_signal(20.0, 8000.0, seed=1, fs=fs)
    y = -0.7 * _band_signal(trusted[0], trusted[1], seed=2, fs=fs)
    return y, d


def _criterion(lambda_dnh: float) -> ANCLoss:
    train = load_yaml(REPO_ROOT / "configs" / "train_pretrain.yaml")
    *_, fs, _handoff, _settle, trusted = _active_contract()
    loss_cfg = dict(train["loss"])
    loss_cfg["lambda_dnh"] = float(lambda_dnh)
    return ANCLoss(_plant(), loss_cfg, fs, trusted_band_hz=trusted).eval()


def _budget(
    lambda_dnh: float, *, loss_start_sample: int | None = None
) -> dict[str, float]:
    *_, _fs, _handoff, settle, _trusted = _active_contract()
    criterion = _criterion(lambda_dnh)
    y, d = _fixture()
    return criterion.gradient_budget(
        y,
        d,
        perturb={"jitter": 0},
        loss_start_sample=(settle.samples if loss_start_sample is None else loss_start_sample),
    )


# --------------------------------------------------------------------------- 양성 대조
def test_active_loss_contract_is_strict_and_derives_the_actual_settle_window() -> None:
    duct, sp, fs, handoff, settle, trusted = _active_contract()

    assert duct["secondary_path"]["npz"].endswith("secondary_path_il_strict_5dc06fdd.npz")
    # 측정 artifact가 바뀌면 이 수치는 함께 재승인해야 한다. 손으로 쓰는 학습 숫자는
    # 아니며, 위의 PlantSettle 유도가 본체다.
    assert int(sp.delay_samples) == 1245
    assert handoff == 256
    assert int(sp.fir.size) == 2048
    assert settle.samples == 3549
    assert fs == 48_000
    assert trusted == pytest.approx((150.0, 1600.0))


def test_gradient_budget_forwards_the_actual_training_context() -> None:
    """예산 측정이 정착 구간을 빼먹으면 동일한 loss가 아니다."""

    *_, settle, _trusted = _active_contract()
    criterion = _criterion(0.00075)
    y, d = _fixture()
    with patch.object(criterion, "forward", wraps=criterion.forward) as forwarded:
        budget = criterion.gradient_budget(
            y,
            d,
            {"jitter": 0},  # 기존 positional perturb 호출 호환성도 유지한다.
            loss_start_sample=settle.samples,
            nl_params=None,
        )
    assert budget["nmse"] == pytest.approx(1.0)
    assert forwarded.call_count == 1
    kwargs = forwarded.call_args.kwargs
    assert kwargs["loss_start_sample"] == settle.samples
    assert kwargs["perturb"] == {"jitter": 0}
    assert kwargs["nl_params"] is None


def test_strict_training_trim_changes_the_gradient_budget() -> None:
    """legacy/skip=0 수치를 실제 학습 수치로 오인하는 회귀를 막는다."""

    *_, settle, _trusted = _active_contract()
    trimmed = _budget(0.00075)
    untrimmed = _budget(0.00075, loss_start_sample=0)
    assert trimmed["nmse"] == pytest.approx(1.0)
    assert trimmed["dnh"] > 0.0
    assert trimmed["dnh"] < untrimmed["dnh"] * 0.75
    assert settle.samples == 3549


def test_shipped_lambda_dnh_is_a_nonzero_non_swamping_smoke_value() -> None:
    pretrain = load_yaml(REPO_ROOT / "configs" / "train_pretrain.yaml")
    finetune = load_yaml(REPO_ROOT / "configs" / "train_finetune.yaml")
    shipped = float(pretrain["loss"]["lambda_dnh"])
    # 이 fixture는 canonical 승인값을 결정하지 않는다. 실제 모델·batch의 0.2~0.4
    # 증거는 loss_start_sample과 함께 campaign ledger에서만 통과시킨다.
    assert shipped == float(finetune["loss"]["lambda_dnh"])
    assert shipped == pytest.approx(0.00075)
    share = _budget(shipped)["dnh"]
    assert 0.0 < share < BUDGET_HI


def test_the_nmse_term_is_the_denominator_and_is_alive() -> None:
    budget = _budget(0.001)
    assert budget["nmse"] == pytest.approx(1.0)
    # frame은 metric-only지만 MR-STFT와 DNH는 여전히 목적함수에 살아 있어야 한다.
    assert budget["mrstft"] > 0.0 and budget["dnh"] > 0.0
    assert budget["frame"] == 0.0


def test_other_terms_are_recorded_so_a_silent_drift_is_visible() -> None:
    """dnh 말고 다른 항이 목적함수를 덮는 것도 잡는다.

    frame은 현재 승인된 항이 아니다. 여기서는 strict+trim 경로에서 값이 기록되는지만
    확인하고, 실제 model-output ``y`` gradient 방향/재가중은 A100 control evidence로
    결정한다. parameter-gradient의 증거는 아니다.
    """

    budget = _budget(0.001)
    for name in ("mrstft", "sat"):
        assert budget[name] < 3.0, (
            f"{name} 항의 몫이 {budget[name]:.2f} 입니다 — 목적함수를 덮습니다"
        )
    assert budget["frame"] == 0.0


# --------------------------------------------------------------------------- 음성 대조
def test_the_old_lambda_would_swamp_the_objective() -> None:
    """마진을 조인 뒤 옛 λ=0.12 가 목적함수를 몇 배로 덮는지 **숫자로** 고정한다.

    이 테스트가 없으면 위 테스트는 "λ 가 목표 안에 있다" 만 말하고, **왜 바꿔야 했는지**
    를 잃는다. 그리고 누가 λ 를 0.12 로 되돌리면 조용히 지나간다.
    """

    share = _budget(0.12)["dnh"]
    assert share > 10.0, (
        f"옛 λ=0.12 의 몫이 {share:.2f} 입니다 — 10 을 넘지 않으면 이 픽스처가 힌지를 "
        "제대로 활성화하지 못한 것이고, 그러면 위 테스트도 의미가 없습니다"
    )


def test_zero_lambda_kills_the_term_entirely() -> None:
    assert _budget(0.0)["dnh"] == 0.0


def test_budget_scales_linearly_with_lambda() -> None:
    """예산비가 λ 에 선형이어야 한다 — 아니면 측정이 항을 잘못 잡고 있는 것이다."""

    a, b = _budget(0.001)["dnh"], _budget(0.002)["dnh"]
    assert b == pytest.approx(2.0 * a, rel=1e-3)
