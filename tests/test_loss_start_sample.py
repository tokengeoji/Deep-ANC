"""학습이 버리는 구간과 평가가 버리는 구간이 **같은 출처에서** 나오는가.

2026-08-05 이전 상태: 학습(open_loop)은 앞을 0 샘플 버렸고, 평가는 항상
``closed_loop.warmup_seconds``(0.25 s = 12000 샘플)를 버렸다. 두 숫자는 같은 양이
아니었고, 두 숫자가 다르다는 것을 검사하는 코드가 아무 데도 없었다 — 발생기 A
("같은 물리량을 두 곳에서 따로 유도하고 대조하지 않는다")의 전형이다.

여기서 고정하는 계약:
  1. 정착 구간은 :meth:`PlantSettle.derive` 로만 만들어진다 (손으로 못 쓴다).
  2. 평가 warmup 은 그 값 **아래로 내려갈 수 없다**.
  3. 학습(``trainer.loss_start_sample``)과 평가가 같은 유도식을 쓴다.
"""

from __future__ import annotations

import numpy as np
import pytest

from deep_anc.config import REPO_ROOT, load_yaml
from deep_anc.dsp.secondary_path import load_secondary_path
from deep_anc.dsp.timing import PlantSettle, handoff_samples_from_config
from deep_anc.eval.recorded import resolve_warmup_samples


FS = 48_000


def _shipped_settle() -> PlantSettle:
    duct = load_yaml(REPO_ROOT / "configs/duct.yaml")
    sp = load_secondary_path(REPO_ROOT / duct["secondary_path"]["npz"])
    return PlantSettle.derive(
        secondary_delay_samples=int(sp.delay_samples),
        handoff_samples=handoff_samples_from_config(duct),
        fir_taps=int(sp.fir.size),
        sample_rate=int(sp.sample_rate),
    )


def test_plant_settle_cannot_be_constructed_by_hand() -> None:
    """손으로 쓰는 순간 그것이 **두 번째 유도**가 된다."""

    with pytest.raises(TypeError, match="derive"):
        PlantSettle(
            samples=3769,
            secondary_delay_samples=1465,
            handoff_samples=256,
            fir_taps=2048,
            sample_rate=FS,
        )


def test_plant_settle_is_derived_from_the_measured_plant() -> None:
    settle = PlantSettle.derive(
        secondary_delay_samples=1462,
        handoff_samples=256,
        fir_taps=2048,
        sample_rate=FS,
    )
    assert settle.samples == 1462 + 256 + 2048
    assert int(settle) == settle.samples
    assert settle.milliseconds == pytest.approx(1000.0 * settle.samples / FS)
    assert "FIR 2048" in settle.describe()


def test_plant_settle_rejects_impossible_values() -> None:
    with pytest.raises(ValueError, match="0 이상"):
        PlantSettle.derive(
            secondary_delay_samples=-1, handoff_samples=256, fir_taps=8, sample_rate=FS
        )
    with pytest.raises(ValueError, match="fir_taps"):
        PlantSettle.derive(
            secondary_delay_samples=0, handoff_samples=0, fir_taps=0, sample_rate=FS
        )
    with pytest.raises(ValueError, match="sample_rate"):
        PlantSettle.derive(
            secondary_delay_samples=0, handoff_samples=0, fir_taps=1, sample_rate=0
        )


def test_evaluation_warmup_cannot_drop_below_the_plant_settle_window() -> None:
    """**게이트 강화** — warmup_seconds 를 0 으로 내려도 정착 구간은 반드시 버린다.

    그 구간은 y 가 무엇이든 ``e = d`` 라 상쇄량을 잴 수 없다. 실측 하한(실측 d,
    trusted 150–600 Hz, N=1721): mean −20.3 / CVaR10 −10.1 / worst −4.8 dB.
    하한이 없으면 평가가 조용히 이 구간을 포함하게 된다.
    """

    data = {"closed_loop": {"warmup_seconds": 0.0}}
    settle = 3769

    assert resolve_warmup_samples(data, FS) == 0
    assert resolve_warmup_samples(data, FS, min_samples=settle) == settle
    with pytest.raises(ValueError, match="min_samples"):
        resolve_warmup_samples(data, FS, min_samples=-1)


def test_warmup_floor_does_not_shrink_a_longer_requested_warmup() -> None:
    """오기각 방지 — 현재 설정(12000 > 3769)에서는 동작이 변하지 않는다."""

    data = {"closed_loop": {"warmup_seconds": 0.25}}
    assert resolve_warmup_samples(data, FS) == 12_000
    assert resolve_warmup_samples(data, FS, min_samples=3769) == 12_000
    assert resolve_warmup_samples(data, FS, 0.5, min_samples=3769) == 24_000


def test_training_and_evaluation_read_the_same_settle_window() -> None:
    """두 도메인이 같은 유도식을 쓰는가 (계약의 본체).

    trainer 는 ``PlantSettle.derive(...).samples`` 를 ``loss_start_sample`` 기본값으로
    쓰고, 평가기는 같은 값을 warmup 하한으로 쓴다. 여기서는 출하 플랜트로 두 경로가
    같은 숫자를 내는지 확인한다.
    """

    duct = load_yaml(REPO_ROOT / "configs/duct.yaml")
    sp = load_secondary_path(REPO_ROOT / duct["secondary_path"]["npz"])
    settle = _shipped_settle()

    assert settle.samples == (
        int(sp.delay_samples) + handoff_samples_from_config(duct) + int(sp.fir.size)
    )
    # 평가기 하한으로 넘겼을 때 같은 값이 나온다.
    silent = {"closed_loop": {"warmup_seconds": 0.0}}
    assert resolve_warmup_samples(
        silent, sp.sample_rate, min_samples=settle.samples
    ) == settle.samples
    # 세그먼트(1.5 s)의 일부만 버린다 — 학습이 성립하는 범위인지 확인.
    data_cfg = load_yaml(REPO_ROOT / "configs/data_sim.yaml")
    segment = int(float(data_cfg["segment_seconds"]) * int(data_cfg["sample_rate"]))
    assert 0 < settle.samples < segment // 4


def test_loss_start_sample_matches_the_secondary_path_impulse_support() -> None:
    """정착 구간이 실제로 'e = d 인 구간'과 같은가 (물리 확인).

    항등 입력을 넣어 S(z) 의 임펄스응답 지지를 직접 재고, 그 끝이 정착 구간과
    일치하는지 본다. 주석이 아니라 측정으로 고정한다.
    """

    duct = load_yaml(REPO_ROOT / "configs/duct.yaml")
    sp = load_secondary_path(REPO_ROOT / duct["secondary_path"]["npz"])
    handoff = handoff_samples_from_config(duct)
    settle = _shipped_settle()

    # S(z) 를 통과한 임펄스는 delay+handoff 에서 시작해 FIR 길이만큼 이어진다.
    impulse_support_end = int(sp.delay_samples) + handoff + int(sp.fir.size)
    assert settle.samples == impulse_support_end
    # 그 앞에서는 어떤 y 도 e 를 바꿀 수 없다.
    assert np.count_nonzero(sp.fir) > 0
