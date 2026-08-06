"""손실 힌지와 G4 게이트가 **같은 물리를 말하는지** 직접 실행으로 강제한다.

왜 이 파일이 있는가
------------------
2026-08-06 이전, 손실은 ``dnh_margin_db: 6.0`` 을, 게이트는
``MAX_OUT_OF_BAND_AMPLIFICATION_DB = 1.0`` 을 각자 들고 있었다. 두 값이 같은 것을
말하는지 확인하는 코드도 테스트도 없었다. 실측하니 **손실을 정확히 만족한 모델이
게이트를 8.5 dB 차이로 FAIL** 했다.

그래서 이 파일의 핵심 테스트는 하나다 —

    "힌지를 **정확히** 만족하는 최악의 신호를 만들어 실제 게이트에 넣는다."

주석이나 부등식 대조가 아니라 **양쪽 코드를 실제로 실행**해서 잇는다. 그리고 같은
구성으로 옛 마진(+6.0)이 **FAIL 하는 것을 함께 강제**한다 — 음성 대조가 없으면
게이트가 그냥 꺼져 있어도 이 파일은 초록이기 때문이다.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml

from deep_anc.dsp.do_no_harm import (
    MAX_OUT_OF_BAND_AMPLIFICATION_DB,
    OCTAVE_BAND_CENTERS_HZ,
    gate_consistent_margin_db,
    octave_band_edges_hz,
    worst_case_amplification_db,
)
from deep_anc.dsp.timing import FrequencyBand
from deep_anc.eval.metrics import octave_band_attenuation
from deep_anc.losses.config import DEFAULT_DNH_MARGIN_DB, DoNoHarmBand, DoNoHarmPlan, LossConfig

SAMPLE_RATE = 48_000
SAMPLES = 1 << 16  # 1.37 s — 옥타브 125 (88 Hz) 를 재기에 충분하다
PROTECTED = (150.0, 1600.0)
"""현행 플랜트의 신뢰대역. S npz 가 신고하는 값과 같다(HANDOFF §플랜트 복구 결과)."""

# 게이트 판정에 허용하는 수치 여유. 게이트는 4차 Butterworth 로 옥타브를 자르고
# 힌지는 FFT 빈으로 자르므로 경계에서 미세한 누설이 있다. 실측 차이는 0.01 dB 미만이다.
_TOLERANCE_DB = 0.05


def _shipped_plan(margin_db: float | None = None) -> DoNoHarmPlan:
    """출하 경로 그대로 do-no-harm 계획을 만든다."""

    kwargs = {} if margin_db is None else {"margin_db": margin_db}
    return DoNoHarmPlan.derive(
        protected=FrequencyBand(lo_hz=PROTECTED[0], hi_hz=PROTECTED[1]),
        nyquist_hz=SAMPLE_RATE / 2.0,
        **kwargs,
    )


def _disturbance() -> np.ndarray:
    """20 Hz–20 kHz 백색 교란. 모든 옥타브에 에너지가 있어야 판정이 정의된다."""

    rng = np.random.default_rng(20260806)
    spectrum = np.zeros(SAMPLES // 2 + 1, dtype=complex)
    freqs = np.fft.rfftfreq(SAMPLES, 1.0 / SAMPLE_RATE)
    mask = (freqs >= 20.0) & (freqs <= 20_000.0)
    spectrum[mask] = rng.normal(size=int(mask.sum())) + 1j * rng.normal(size=int(mask.sum()))
    return spectrum


def _anti_noise_at_hinge(
    spectrum: np.ndarray, plan: DoNoHarmPlan, *, concentrate_in: tuple[float, float] | None = None
) -> np.ndarray:
    """힌지를 **정확히** 만족하면서 게이트에 가장 나쁜 ``S·y`` 를 만든다.

    최악은 두 가지가 겹칠 때다.

    1. **동상** — ``e = d + S·y`` 에서 위상이 맞으면 진폭이 그대로 더해진다
       (대역별 Cauchy–Schwarz 등호).
    2. **집중** — 힌지 대역 안 어디에 에너지를 두든 대역 비율은 같다. 옥타브를
       가로지르는 대역이 있으면 한 옥타브에 몰아넣어 그 옥타브만 나쁘게 만들 수 있다.
    """

    freqs = np.fft.rfftfreq(SAMPLES, 1.0 / SAMPLE_RATE)
    out = np.zeros_like(spectrum)
    for item in plan.bands:
        lo, hi = item.band.as_tuple()
        mask = (freqs >= lo) & (freqs < hi)
        available = float((np.abs(spectrum[mask]) ** 2).sum())
        if available <= 0.0:
            continue  # d 에 에너지가 없는 대역 — 비율이 정의되지 않는다
        budget = (10.0 ** (item.margin_db / 10.0)) * available
        target = mask
        if concentrate_in is not None:
            sub = mask & (freqs >= concentrate_in[0]) & (freqs < concentrate_in[1])
            if sub.any() and float((np.abs(spectrum[sub]) ** 2).sum()) > 0.0:
                target = sub
        have = float((np.abs(spectrum[target]) ** 2).sum())
        out[target] = math.sqrt(budget / have) * spectrum[target]
    return out


def _octave_attenuations(d: np.ndarray, e: np.ndarray) -> dict[float, float]:
    rows = octave_band_attenuation(
        d, e, SAMPLE_RATE, list(OCTAVE_BAND_CENTERS_HZ), trusted_band_hz=PROTECTED
    )
    return {float(row["center_hz"]): float(row["attenuation_db"]) for row in rows}


def _run(plan: DoNoHarmPlan, *, concentrate_in: tuple[float, float] | None = None):
    spectrum = _disturbance()
    d = np.fft.irfft(spectrum, n=SAMPLES)
    s_y = np.fft.irfft(_anti_noise_at_hinge(spectrum, plan, concentrate_in=concentrate_in), n=SAMPLES)
    return _octave_attenuations(d, d + s_y)


# --------------------------------------------------------------------------- 양성 대조
def test_a_model_that_exactly_satisfies_the_hinge_passes_the_g4_gate() -> None:
    """**이 저장소에서 가장 중요한 교차 검증.**

    출하 마진을 정확히 만족하는 최악 신호가 게이트를 통과해야 한다. 통과하지 못하면
    손실이 요구하는 것과 게이트가 요구하는 것이 다르다는 뜻이고, 그 상태로 학습하면
    38.5분짜리 재녹음과 GPU 시간을 **확정적으로 버린다**.
    """

    attenuations = _run(_shipped_plan())
    worst_center = min(attenuations, key=lambda fc: attenuations[fc])
    assert attenuations[worst_center] >= -(MAX_OUT_OF_BAND_AMPLIFICATION_DB + _TOLERANCE_DB), (
        f"힌지를 정확히 만족한 신호가 {worst_center:.0f} Hz 옥타브에서 "
        f"{attenuations[worst_center]:+.2f} dB — 게이트 한계 "
        f"{-MAX_OUT_OF_BAND_AMPLIFICATION_DB:+.1f} dB 를 넘었습니다. "
        "손실 마진과 게이트 임계가 다시 갈라졌습니다."
    )


def test_energy_concentrated_in_one_octave_still_passes() -> None:
    """대역 안에서 에너지를 몰아넣어도 게이트를 못 넘어야 한다.

    옛 구성에서는 ``[1633, 6000]`` 이 옥타브 2000·4000 을 함께 덮어서, 대역 비율을
    정확히 만족한 채로 2000 옥타브만 −12.63 dB 로 만들 수 있었다. 옥타브 정렬이
    이 자유도를 없앤다.
    """

    attenuations = _run(_shipped_plan(), concentrate_in=(1633.0, 2828.4))
    worst_center = min(attenuations, key=lambda fc: attenuations[fc])
    assert attenuations[worst_center] >= -(MAX_OUT_OF_BAND_AMPLIFICATION_DB + _TOLERANCE_DB), (
        f"에너지 집중 공격이 {worst_center:.0f} Hz 옥타브를 "
        f"{attenuations[worst_center]:+.2f} dB 로 만들었습니다"
    )


# --------------------------------------------------------------------------- 음성 대조
def test_the_old_margin_would_have_failed_the_gate() -> None:
    """옛 마진 +6.0 dB 가 **실제로 FAIL** 하는 것을 강제한다.

    이 테스트가 없으면 위의 두 테스트는 "게이트가 꺼져 있어서" 통과하는 것과 구별되지
    않는다. 2026-08-06 실측값: 4000 Hz 옥타브 −9.53 dB.
    """

    plan = DoNoHarmPlan.model_construct(  # 검증을 우회해 옛 상태를 재현한다
        protected=FrequencyBand(lo_hz=PROTECTED[0], hi_hz=PROTECTED[1]),
        bands=tuple(
            DoNoHarmBand.model_construct(band=item.band, margin_db=6.0, weight=item.weight)
            for item in _shipped_plan().bands
        ),
    )
    attenuations = _run(plan)
    worst = min(attenuations.values())
    assert worst <= -(MAX_OUT_OF_BAND_AMPLIFICATION_DB + 1.0), (
        f"옛 마진 +6.0 dB 가 게이트를 통과했습니다(최악 {worst:+.2f} dB) — 이 테스트는 "
        "실패할 수 있어야 의미가 있는데 실패하지 못했습니다"
    )
    assert worst == pytest.approx(-9.53, abs=0.15), (
        f"옛 마진의 최악값이 {worst:+.2f} dB 로, 2026-08-06 실측 −9.53 dB 와 다릅니다"
    )


# ------------------------------------------------------------------- 구성이 막는 것
def test_config_rejects_a_margin_looser_than_the_gate_allows() -> None:
    with pytest.raises(ValueError, match="G4 임계"):
        LossConfig(nmse_objective="trusted_band", lambda_dnh=0.1, dnh_margin_db=6.0).resolve_do_no_harm(
            protected=FrequencyBand(lo_hz=PROTECTED[0], hi_hz=PROTECTED[1]),
            nyquist_hz=SAMPLE_RATE / 2.0,
        )


def test_config_rejects_bands_that_cross_an_octave_edge() -> None:
    """직접 대역을 적는 경로(``do_no_harm_bands``)로도 우회할 수 없어야 한다."""

    cfg = LossConfig(
        nmse_objective="trusted_band",
        lambda_dnh=0.1,
        do_no_harm_bands=[[1633.0, 6000.0, DEFAULT_DNH_MARGIN_DB, 2.0]],
    )
    with pytest.raises(ValueError, match="옥타브 경계"):
        cfg.resolve_do_no_harm(
            protected=FrequencyBand(lo_hz=PROTECTED[0], hi_hz=PROTECTED[1]),
            nyquist_hz=SAMPLE_RATE / 2.0,
        )


def test_derived_plan_never_crosses_an_octave_edge() -> None:
    plan = _shipped_plan()
    edges = [edge for fc in OCTAVE_BAND_CENTERS_HZ for edge in octave_band_edges_hz(fc)]
    for item in plan.bands:
        lo, hi = item.band.as_tuple()
        inside = [edge for edge in edges if lo + 1e-6 < edge < hi - 1e-6]
        assert not inside, f"대역 [{lo:g}, {hi:g}] 가 옥타브 경계 {inside} 를 가로지릅니다"


# ------------------------------------------------------- 단일 출처가 실제로 하나인가
def test_eval_configs_agree_with_the_octave_center_single_source() -> None:
    """``configs/eval*.yaml`` 에 같은 목록이 세 벌 있다. 갈라지면 즉시 실패시킨다."""

    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "configs"
    checked = 0
    for path in sorted(root.glob("eval*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        centers = data.get("octave_bands_hz")
        if centers is None:
            continue
        checked += 1
        assert tuple(float(v) for v in centers) == OCTAVE_BAND_CENTERS_HZ, (
            f"{path.name} 의 octave_bands_hz 가 코드의 단일 출처와 다릅니다: {centers}"
        )
    assert checked >= 3, f"eval 설정을 {checked}개만 검사했습니다 — 스캔이 좁아졌습니다"


def test_gate_constant_has_exactly_one_definition() -> None:
    """``MAX_OUT_OF_BAND_AMPLIFICATION_DB`` 가 다시 두 곳에서 정의되지 않게 한다."""

    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "deep_anc"
    definitions = [
        path
        for path in root.rglob("*.py")
        if any(
            line.startswith("MAX_OUT_OF_BAND_AMPLIFICATION_DB =")
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    ]
    assert [p.name for p in definitions] == ["do_no_harm.py"], (
        f"게이트 임계가 {[str(p) for p in definitions]} 에서 정의됩니다 — 단일 출처여야 합니다"
    )


def test_eval_recorded_still_exposes_the_constant() -> None:
    """기존 참조(스크립트·테스트)가 깨지지 않았는지 확인한다."""

    from deep_anc.eval.recorded import MAX_OUT_OF_BAND_AMPLIFICATION_DB as reexported

    assert reexported == MAX_OUT_OF_BAND_AMPLIFICATION_DB


# --------------------------------------------------------------------- 유도식 자체
def test_derived_margin_saturates_the_gate_exactly() -> None:
    """유도식이 게이트 임계를 **정확히** 포화시키는지 — 닫힌 형태끼리 대조한다."""

    assert worst_case_amplification_db(gate_consistent_margin_db()) == pytest.approx(
        MAX_OUT_OF_BAND_AMPLIFICATION_DB, abs=1e-9
    )
    assert DEFAULT_DNH_MARGIN_DB == pytest.approx(-18.2715, abs=1e-3)


def test_plant_uncertainty_only_tightens_the_margin() -> None:
    """|S| 불확실성이 마진을 **좁히는** 방향으로만 작용해야 한다 (부호 오류 회귀 방어)."""

    assert gate_consistent_margin_db(plant_uncertainty_db=3.0) < gate_consistent_margin_db()


def test_zero_gate_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="유한한 양수"):
        gate_consistent_margin_db(0.0)
