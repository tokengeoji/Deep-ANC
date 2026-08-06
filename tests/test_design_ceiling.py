"""설계 상한을 **아티팩트에서 다시 푼다** — 설정에 적힌 숫자를 믿지 않는다.

왜 이 파일이 있는가
------------------
2026-08-06 감사가 재현하고 내가 직접 확인한 fail-open:
``--set readiness.measured_design_ceiling_db=30.0`` 으로 날조해도 게이트가 **PASS** 했다
(100.0 도 마찬가지). 구속 상한이 "플랜트 일관성 27.73 dB" 로 폴백하면서 통과한다.

즉 **물리적으로 도달할 수 없는 목표를 세워도 진입 게이트가 막지 못했다.** 그리고 배선을
고쳐 P/S 를 다시 재면 아티팩트 sha 는 바뀌지만 설정의 숫자는 그대로 남아 계속 통과한다 —
게이트가 자기 자신을 증명하는 구조였다.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import signal

from deep_anc.dsp.design_ceiling import design_ceiling_db

FS = 48_000
BAND = (150.0, 1600.0)


def _paths(delay_p: int = 200, delay_s: int = 120):
    """장난감 P/S — 실측 아티팩트 없이도 성질을 검사할 수 있게."""

    rng = np.random.default_rng(20260806)
    lowpass = signal.firwin(129, 2000.0, fs=FS)
    p = np.convolve(np.concatenate([np.zeros(delay_p), rng.normal(size=64)]), lowpass)
    s = np.convolve(np.concatenate([np.zeros(delay_s), rng.normal(size=64)]), lowpass)
    return p.astype(np.float64), s.astype(np.float64)


def test_more_lead_never_hurts_the_ceiling() -> None:
    """lead 는 미래를 보는 양이다 — 늘리면 상한이 나빠질 수 없다.

    부호 규약이나 shift 방향이 뒤집히면 이 단조성이 먼저 깨진다.
    """

    p, s = _paths()
    values = [
        design_ceiling_db(p, s, lead_samples=k, band_hz=BAND, sample_rate=FS, taps=256).ceiling_db
        for k in (0, 40, 120)
    ]
    assert values[1] >= values[0] - 0.05, values
    assert values[2] >= values[1] - 0.05, values


def test_a_secondary_path_that_arrives_too_late_cannot_cancel() -> None:
    """S 가 P 보다 훨씬 늦게 도착하면 인과적으로 상쇄할 수 없다.

    ⚠ "S 를 작게 만들면 된다" 가 아니다. S 를 상수배 줄이면 해가 그만큼 커져 똑같이
    상쇄한다 — 실제로 1e-9 배 입력에서 이 함수는 ``|w|max = 1.4e9`` 로 31.6 dB 를
    "달성" 했다. 물리적 무능력은 **이득이 아니라 인과성**에서 온다.
    """

    p, s = _paths(delay_p=40, delay_s=1200)
    result = design_ceiling_db(
        p, s, lead_samples=0, band_hz=BAND, sample_rate=FS, taps=256
    )
    assert result.ceiling_db < 1.0, result


def test_a_numerically_exploding_solution_is_not_called_stable() -> None:
    """탭이 폭발한 해를 "안정" 이라고 말하지 않는다 (위 사례의 회귀 방어)."""

    p, s = _paths()
    result = design_ceiling_db(
        p, 1.0e-9 * s, lead_samples=100, band_hz=BAND, sample_rate=FS, taps=256
    )
    assert result.max_tap > 1.0e3, result
    assert not result.stable_over_regularisation, result


def test_the_solution_is_not_numerical_garbage() -> None:
    """정규화가 실제로 작동하는지 — 탭이 폭발하면 그 값은 상한이 아니다.

    브릭월 대역으로 자르면 유효 랭크가 2048 중 약 124 이고 정규화 없이 풀면
    ``|w|max ≈ 6.7e+141`` 이 나온다. 그 해는 수치 쓰레기다.
    """

    p, s = _paths()
    result = design_ceiling_db(p, s, lead_samples=100, band_hz=BAND, sample_rate=FS, taps=256)
    assert result.max_tap < 1.0e3, result
    assert math.isfinite(result.condition_number)
    assert result.stable_over_regularisation, result


def test_the_band_matters_and_is_carried_with_the_number() -> None:
    """대역이 다르면 상한이 다르다 — 그래서 숫자만 들고 다니면 안 된다.

    실제 오판정: 설정이 150-600Hz 에서 푼 6.53 을 선언했는데 요구 대역은 150-1600Hz
    였다. 같은 플랜트를 요구 대역에서 다시 풀면 값이 달라진다.

    ⚠ "좁은 대역이 항상 쉽다" 는 성립하지 않는다. 실측 P/S 에서는 150-600 이 5.20,
    150-1600 이 4.83 으로 좁은 쪽이 높지만, 장난감 입력에서는 반대로 나온다 — 상쇄
    난이도는 대역 폭이 아니라 그 대역에서 S 가 P 를 얼마나 흉내낼 수 있는가로 정해진다.
    그래서 여기서는 **값이 대역에 의존한다는 사실**만 강제하고 방향은 주장하지 않는다.
    """

    p, s = _paths()
    narrow = design_ceiling_db(
        p, s, lead_samples=100, band_hz=(150.0, 600.0), sample_rate=FS, taps=256
    )
    wide = design_ceiling_db(
        p, s, lead_samples=100, band_hz=BAND, sample_rate=FS, taps=256
    )
    assert abs(narrow.ceiling_db - wide.ceiling_db) > 0.5, (narrow, wide)
    assert narrow.band_hz != wide.band_hz


# ------------------------------------------------------- 실측 아티팩트에서의 재현
def test_shipped_declaration_matches_the_recomputation() -> None:
    """출하 설정의 선언값이 실측 아티팩트 재계산과 맞는지 — 이것이 게이트의 본체다."""

    import yaml

    from deep_anc.config import REPO_ROOT
    from deep_anc.dsp.design_ceiling import cached_design_ceiling_db
    from deep_anc.train.finetune_readiness import DESIGN_CEILING_TOLERANCE_DB

    primary = REPO_ROOT / "assets" / "measured" / "primary_path_il.npz"
    secondary = REPO_ROOT / "assets" / "measured" / "secondary_path_il.npz"
    if not primary.is_file() or not secondary.is_file():
        pytest.skip("실측 P/S 아티팩트가 없는 환경")

    cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "train_finetune.yaml").read_text(encoding="utf-8")
    )
    readiness = cfg["readiness"]
    declared = float(readiness["measured_design_ceiling_db"])
    band = tuple(float(v) for v in readiness["measured_design_ceiling_band_hz"][:2])
    data_sim = yaml.safe_load(
        (REPO_ROOT / "configs" / "data_sim.yaml").read_text(encoding="utf-8")
    )
    lead = int(data_sim["digital_reference_lead_samples"])

    solved = cached_design_ceiling_db(
        primary, secondary, lead_samples=lead, band_hz=band, sample_rate=float(FS)
    )
    assert declared <= solved.ceiling_db + DESIGN_CEILING_TOLERANCE_DB, (
        f"출하 선언 {declared:.2f} dB 가 재계산 {solved.ceiling_db:.2f} dB 보다 낙관적입니다"
    )
    # 재계산이 "그럴듯한 숫자" 인지도 함께 본다 — 0 이나 30 이 나오면 계산이 깨진 것이다.
    assert 3.0 < solved.ceiling_db < 8.0, solved
    # 경계를 명시한다: 2026-08-06 실측 재계산 4.83 dB, 출하 선언 4.58 dB.
    # 선언이 4.83 + 0.5 = 5.33 을 넘으면 게이트가 거부한다 (5.5 로 직접 확인).
    assert solved.ceiling_db == pytest.approx(4.83, abs=0.30), solved
    assert declared == pytest.approx(4.58, abs=0.01)
