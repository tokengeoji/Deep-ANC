"""교차 도메인 불변식 검사기 — 정상 입력과 **오염 입력** 양쪽을 넣는다 (발생기 A'/B).

각 검사에 대해 반드시 두 개가 있다:
  · 오염 입력에서 FAIL 하는가 — 이게 없으면 그 검사가 작동한다는 증거가 없다.
  · 정상 입력에서 PASS 하는가 — 이게 없으면 무조건 거부하는 검사와 구별되지 않는다.

오염 입력은 만들어낸 것이 아니라 **2026-08-04 에 실제로 일어난 값**이다.
"""

from __future__ import annotations

import numpy as np
import pytest

from deep_anc.dsp.invariants import (
    MAX_STREAM_DELAY_P95_P5_SAMPLES,
    MAX_STREAM_DELAY_ROBUST_STD_SAMPLES,
    MIN_STREAM_DELAY_VALID_WINDOW_RATIO,
    InvariantViolation,
    check_corpus_disjoint,
    check_lead_agreement,
    check_measured_delay_agreement,
    check_plant_fingerprint_match,
    check_relative_tau_constancy,
    check_stream_coherence,
    check_stream_delay_stability,
    derive_playback_to_error_delay_samples,
)
from deep_anc.dsp.timing import BandPlan, PlantDelays, PlantFingerprint


FS = 48_000

# 2026-08-04 출하 아티팩트의 실측 P−S 상대 τ. 반복 11 에서 1.4 → 32 샘플 점프.
SHIPPED_RELATIVE_TAU = [
    0.0, 1.20, 1.13, 1.09, 1.09, 1.29, 1.41, 1.47, 1.14, 1.48, 1.36,
    32.11, 32.18, 31.75, 30.26, 29.06,
]


# ----------------------------------------------------------------------------------
# P−S 상대 τ 상수성
# ----------------------------------------------------------------------------------
def test_relative_tau_constancy_fails_on_the_measured_frame_slip():
    rel = np.asarray(SHIPPED_RELATIVE_TAU)
    result = check_relative_tau_constancy(rel, np.zeros_like(rel), tolerance_samples=3.0)

    assert not result.ok
    assert result.measured["outlier_indices"] == [11, 12, 13, 14, 15]
    # 게이트가 실제로 본 값은 range 32.18 하나였고, 아티팩트가 스스로 써넣은 허용치
    # 48 과 비교해 통과시켰다. 계단 자체를 보면 30.75 샘플이다.
    assert result.measured["max_consecutive_step_samples"] == pytest.approx(30.75, abs=0.01)
    with pytest.raises(InvariantViolation, match="프레임 슬립"):
        result.raise_if_failed()


def test_relative_tau_constancy_passes_on_the_clean_subset():
    """같은 캡처의 오염 전 11 반복은 통과해야 한다 — 무조건 거부가 아님을 증명."""

    rel = np.asarray(SHIPPED_RELATIVE_TAU[:11])
    result = check_relative_tau_constancy(rel, np.zeros_like(rel), tolerance_samples=3.0)

    assert result.ok
    assert result.measured["max_deviation_samples"] < 3.0
    result.raise_if_failed()


def test_relative_tau_constancy_survives_a_contamination_majority():
    """오염이 과반이어도 잡아야 한다 — MAD 스케일 임계였다면 여기서 무력화된다."""

    rel = np.asarray([0.0, 1.2, 1.1, 1.3] + [32.0] * 12)
    result = check_relative_tau_constancy(rel, np.zeros_like(rel), tolerance_samples=3.0)

    assert not result.ok
    assert result.measured["outlier_indices"]


# ----------------------------------------------------------------------------------
# 재생→캡처 결맞음
# ----------------------------------------------------------------------------------
def _band_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    from scipy import signal

    sos = signal.butter(6, [120.0, 900.0], btype="bandpass", fs=FS, output="sos")
    return signal.sosfilt(sos, rng.standard_normal(n)).astype(np.float64)


def test_stream_coherence_passes_on_a_delayed_copy():
    """정상 세션 = 캡처가 재생의 지연·감쇠 사본 + 잡음. 실측 coh² 0.96~0.99."""

    rng = np.random.default_rng(20260805)
    n = FS * 4
    source = _band_noise(rng, n)
    capture = np.roll(source, 120) * 0.7 + 0.05 * rng.standard_normal(n)

    result = check_stream_coherence(
        source, capture, sample_rate=FS, band_hz=(150.0, 600.0), min_coherence=0.60
    )
    assert result.ok
    assert result.measured["coherence"] > 0.9
    result.raise_if_failed()


def test_stream_coherence_fails_on_a_collapsed_timebase():
    """실측 결함 2 재현: 음향(REF→ERR)은 멀쩡한데 source→ERR 만 붕괴한 세션.

    붕괴본 실측 coh²(source→ERR) 0.021~0.126 / coh²(REF→ERR) 0.959~0.991.
    파일별 RMS·clip·길이는 전부 정상이라 기존 QA 는 80/80 을 PASS 시켰다.
    """

    rng = np.random.default_rng(4)
    n = FS * 4
    source = _band_noise(rng, n)
    unrelated = _band_noise(np.random.default_rng(5), n)
    capture = np.roll(unrelated, 120) * 0.7 + 0.05 * rng.standard_normal(n)
    reference = np.roll(unrelated, 60) * 0.8 + 0.05 * rng.standard_normal(n)

    result = check_stream_coherence(
        source,
        capture,
        sample_rate=FS,
        band_hz=(150.0, 600.0),
        min_coherence=0.60,
        control=reference,
    )

    assert not result.ok
    assert result.measured["coherence"] < 0.3
    # 대조군이 살아 있으면 진단까지 나와야 한다.
    assert result.measured["control_coherence"] > 0.6
    assert "타임베이스" in result.detail
    with pytest.raises(InvariantViolation):
        result.raise_if_failed()


# ----------------------------------------------------------------------------------
# 플랜트 지문
# ----------------------------------------------------------------------------------
def _fingerprint(*, primary: int, secondary: int, physics_status: str) -> PlantFingerprint:
    delays = PlantDelays(
        primary_delay_samples=primary,
        secondary_delay_samples=secondary,
        handoff_samples=256,
        sample_rate=FS,
    )
    bands = BandPlan.resolve(
        plant_trusted_band_hz=(150.0, 1600.0),
        duct_cfg={"acoustics": {"realistic_target_band_hz": [80.0, 800.0]}},
        sample_rate=FS,
    )
    return PlantFingerprint.build(
        delays=delays,
        lead=delays.lead(),
        physics_status=physics_status,
        bands=bands,
    )


def test_plant_fingerprint_match_fails_across_the_20260804_plants():
    before = _fingerprint(
        primary=1489,
        secondary=1342,
        physics_status="secondary_surrogate_representation_pretrain",
    )
    after = _fingerprint(
        primary=1608, secondary=1465, physics_status="measured_primary_path"
    )

    result = check_plant_fingerprint_match(before, after)
    assert not result.ok
    assert any("secondary_delay_samples" in item for item in result.measured["differences"])
    with pytest.raises(InvariantViolation, match="서로 다른 플랜트"):
        result.raise_if_failed()


def test_plant_fingerprint_match_passes_for_the_same_plant():
    one = _fingerprint(primary=1602, secondary=1462, physics_status="measured_primary_path")
    two = _fingerprint(primary=1602, secondary=1462, physics_status="measured_primary_path")

    result = check_plant_fingerprint_match(one, two)
    assert result.ok
    result.raise_if_failed()


# ----------------------------------------------------------------------------------
# lead 유도값 일치
# ----------------------------------------------------------------------------------
def _measured_delays() -> PlantDelays:
    return PlantDelays(
        primary_delay_samples=1602,
        secondary_delay_samples=1462,
        handoff_samples=256,
        sample_rate=FS,
    )


def test_lead_agreement_passes_on_the_measured_plant():
    result = check_lead_agreement(116, _measured_delays())
    assert result.ok
    assert result.measured["derived_lead_samples"] == 116
    assert result.measured["relative_delay_samples"] == 140
    result.raise_if_failed()


def test_lead_agreement_fails_on_the_aaeef41_mismatch():
    """설정이 109 인데 실측 지연이 요구하는 값은 116 이었다 — 양쪽 다 "통과" 였다."""

    result = check_lead_agreement(109, _measured_delays())

    assert not result.ok
    assert result.measured["mismatch_samples"] == 7
    with pytest.raises(InvariantViolation, match="유도되는 값과 다릅니다"):
        result.raise_if_failed()


def test_lead_agreement_tolerance_is_explicit_and_only_widens_when_asked():
    delays = _measured_delays()
    assert not check_lead_agreement(109, delays).ok
    assert check_lead_agreement(109, delays, tolerance_samples=16).ok


# ======================================================================================
# 재생→캡처 지연 안정성 (결함 2 의 두 번째 축)
# ======================================================================================
def _seeded_band_noise(frames: int, seed: int) -> np.ndarray:
    from scipy.signal import butter, lfilter

    rng = np.random.default_rng(seed)
    b, a = butter(4, [100.0 / (FS / 2), 2_000.0 / (FS / 2)], btype="band")
    return lfilter(b, a, rng.standard_normal(frames))


def _wideband_argmax_trajectory(
    playback: np.ndarray, capture: np.ndarray, *, window: int = FS, max_lag: int = 8_000
) -> np.ndarray:
    """**옛 추정기의 재현** — 대역제한도 PHAT 도 없는 1초창 광대역 상관의 |argmax|.

    운영 코드에서는 지웠다. 여기 테스트 안에만 남긴 이유는, 이 추정기가 왜 좋은
    데이터를 떨어뜨렸는지를 숫자로 붙잡아 두기 위해서다. 이것이 프로덕션으로
    되돌아오면 아래 회귀 테스트가 그 사실을 바로 드러낸다.
    """

    from scipy.signal import correlate

    n = int(min(playback.size, capture.size))
    lags = np.arange(-window + 1, window)
    mask = np.abs(lags) <= min(max_lag, window - 1)
    taus = []
    for start in range(0, n - window + 1, window):
        left = playback[start : start + window]
        right = capture[start : start + window]
        if float(left.std()) < 1e-9 or float(right.std()) < 1e-9:
            continue
        corr = correlate(right - right.mean(), left - left.mean(), mode="full")
        taus.append(int(lags[mask][int(np.argmax(np.abs(corr[mask])))]))
    return np.asarray(taus, dtype=np.float64)


def test_stream_delay_stability_passes_on_a_constant_delay():
    """지연이 상수면 통과한다 — 무조건 거부하는 검사가 아님을 증명."""

    frames = FS * 5
    source = _seeded_band_noise(frames, 1)
    capture = np.roll(source, 120)
    result = check_stream_delay_stability(source, capture, sample_rate=FS)
    assert result.ok
    assert abs(result.measured["median_samples"] - 120.0) <= 2.0
    assert result.measured["robust_std_samples"] <= 8.0
    assert result.measured["p95_p5_samples"] <= 48.0


def test_stream_delay_stability_fails_on_a_drifting_timebase():
    """지연이 창마다 크게 움직이면 FAIL 한다.

    실측(2026-08-04, 1초창): 붕괴 세션 τ std 1019~2216 / range 8869~13532 인데
    음향 대조군은 std 17.7~20.1 / range 106~215 였다. 두 무리가 50배 벌어져 있다.

    이 크기(창당 1200 샘플)에서는 추정기가 추적을 아예 놓친다 — 91 창 중 17 창만
    유효해진다. 살아남은 창끼리는 붙어 있어 **로버스트 통계만 보면 PASS 한다**(실측
    robust-std 0.56 / p95−p5 2.39). 유효창 비율을 함께 보지 않으면 그것이 곧
    fail-open 이므로, 이 테스트가 그 구멍을 지킨다.
    """

    frames = FS * 6
    source = _seeded_band_noise(frames, 2)
    capture = np.zeros(frames)
    for index, start in enumerate(range(0, frames, FS)):
        stop = min(frames, start + FS)
        capture[start:stop] = np.roll(source, 120 + index * 1_200)[start:stop]
    result = check_stream_delay_stability(source, capture, sample_rate=FS)
    assert not result.ok
    assert "떠다닙니다" in result.detail
    assert result.measured["valid_window_ratio"] < 0.50
    with pytest.raises(InvariantViolation):
        result.raise_if_failed()


def test_stream_delay_stability_fails_on_a_slow_drift_within_the_search_range():
    """탐색 범위 **안에서** 천천히 떠다니는 경우도 잡아야 한다.

    이쪽이 실측 원본 세션(``source.wav``)의 모습이다: 유효창 비율은 0.70~1.00 으로
    멀쩡한데 robust-std 25~121 / p95−p5 228~371 이다. 즉 "추적은 되는데 값이
    움직인다". 유효창 비율 축만 있으면 이 무리가 통과하므로 두 축이 다 필요하다.
    """

    frames = FS * 8
    source = _seeded_band_noise(frames, 7)
    times = np.arange(frames, dtype=np.float64)
    lag = 140.0 + 150.0 * np.sin(2.0 * np.pi * times / FS / 4.5)  # 실측 헌팅과 같은 규모
    positions = np.clip(times - lag, 0.0, frames - 2.0)
    base = positions.astype(np.int64)
    frac = positions - base
    capture = (1.0 - frac) * source[base] + frac * source[base + 1]

    result = check_stream_delay_stability(source, capture, sample_rate=FS)
    assert not result.ok
    assert result.measured["valid_window_ratio"] >= 0.50  # 추적은 됐다
    assert result.measured["robust_std_samples"] > 8.0  # 그런데 값이 움직인다


def test_stream_delay_stability_does_not_reject_a_session_over_a_few_outlier_windows():
    """**오기각 회귀 방어** — 반증 #14/#18 의 정면 대응.

    실측 세션 101813 의 ``source_aligned → ERR`` 창별 τ 는 옛 추정기로 재면
    ``[142, 479, 487, 129, ..., 6311, ...]`` 이다: 30 창 중 27 창이 125~150 인데
    이상치 3 개가 std 를 1106.55 / range 를 6186 으로 만든다. 같은 신호를
    ``timeline.estimate_lag_track`` 로 재면 robust-std **1.84** 다 — **600배** 차이.
    그 결과 제대로 재정렬된 47 세션 중 **22 개(47%)** 가 오기각됐다.

    여기서는 그 구조를 합성으로 만든다: 상수 지연 142 의 직접파에, 세 구간에서만
    직접파보다 강한 늦은 반사(6311 샘플)를 얹는다. 광대역 argmax 는 그 구간에서
    반사로 튀지만, 대역제한 PHAT + 로버스트 통계는 흔들리지 않아야 한다.
    """

    frames = FS * 30
    source = _seeded_band_noise(frames, 11)
    capture = 0.8 * np.roll(source, 142)
    late = np.roll(source, 6_311)
    for index in (1, 2, 14):  # 1초창 기준 3개 창에만 강한 늦은 반사
        capture[index * FS : (index + 1) * FS] += 1.6 * late[index * FS : (index + 1) * FS]

    legacy = _wideband_argmax_trajectory(source, capture)
    # (1) 옛 판정량이 실제로 폭발하는 신호인지부터 확인한다 — 이걸 안 보면 이
    #     테스트는 "아무 신호나 통과한다"와 구별되지 않는다.
    assert float(np.std(legacy)) > 500.0
    assert float(legacy.max() - legacy.min()) > 3_000.0
    assert float(np.median(legacy)) == pytest.approx(142.0, abs=4.0)

    # (2) 단일 출처 추정기 + 로버스트 통계는 같은 신호를 안정하다고 판정해야 한다.
    result = check_stream_delay_stability(source, capture, sample_rate=FS)
    assert result.ok, result.detail
    assert result.measured["robust_std_samples"] < 8.0
    assert result.measured["p95_p5_samples"] < 48.0
    assert abs(result.measured["median_samples"] - 142.0) <= 2.0


def test_stream_delay_trajectory_has_exactly_one_derivation():
    """지연 궤적을 유도하는 코드가 **한 곳뿐**인지 구조적으로 확인한다.

    두 벌이 공존하면 언젠가 서로 다른 답을 내고 아무도 대조하지 않는다 — 그것이
    반증 #14/#18 의 발생기 A 다. 여기서는 두 가지로 못박는다:
      1. ``invariants`` 가 ``timeline`` 의 함수를 실제로 호출한다(monkeypatch 로 증명).
      2. ``invariants.py`` 소스에 상관/argmax 기반 자체 추정 코드가 없다.
    """

    import deep_anc.data.timeline as timeline_module
    import deep_anc.dsp.invariants as invariants_module

    calls: list[int] = []
    original = timeline_module.measure_delay_trajectory

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    timeline_module.measure_delay_trajectory = spy
    try:
        source = _seeded_band_noise(FS * 5, 3)
        check_stream_delay_stability(source, np.roll(source, 100), sample_rate=FS)
    finally:
        timeline_module.measure_delay_trajectory = original
    assert calls, "invariants 가 timeline 단일 출처를 거치지 않고 지연을 유도했습니다"

    def _code_only(function) -> str:
        """docstring 을 뺀 실행 코드만 돌려준다 — 설명문에 적힌 단어를 잡지 않도록."""

        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        node = tree.body[0]
        body = node.body[1:] if ast.get_docstring(node) is not None else node.body
        return "\n".join(ast.unparse(item) for item in body)

    trajectory_code = _code_only(invariants_module.measure_stream_delay_trajectory)
    for banned in ("correlate", "argmax", "std", "percentile", "for "):
        assert banned not in trajectory_code, (
            f"measure_stream_delay_trajectory 가 지연을 자체 유도하려 합니다: "
            f"{banned!r} — 추정도 통계도 deep_anc.data.timeline 한 곳에서만 한다\n"
            f"{trajectory_code}"
        )
    # 검사기도 통계를 다시 계산하면 안 된다. DelayTrajectory 가 준 값만 읽어야 한다.
    check_code = _code_only(invariants_module.check_stream_delay_stability)
    for banned in ("correlate", "argmax", "np.std", "np.median", "percentile"):
        assert banned not in check_code, (
            f"check_stream_delay_stability 가 통계를 재계산합니다: {banned!r}"
        )


def test_stream_delay_stability_fails_on_a_partial_frame_slip():
    """세션 **일부**만 어긋난 경우 — 로버스트 통계가 구조적으로 못 보는 결함.

    2026-08-06 통합 검증이 잡은 실제 fail-open 이다. 어긋난 구간의 창은 지연이
    ``coarse_search_samples`` 밖으로 튀어 **무효 처리되고 통계에서 빠진다.** 그래서
    살아남은 창만 보는 robust-std / p95−p5 는 어느 슬립 크기에서도 0.7 / 2.1 로
    깨끗하다 — 아래 단언이 그 사실 자체를 못 박는다. 목격자는 유효창 비율 하나뿐이고,
    임계가 0.50 이던 동안 30 초 중 9 초가 42 ms 어긋난 실측 세션이 QA 를 전부 통과했다.

    이것이 ``configs/train_finetune.yaml`` 이 실측 관측 실패모드로 적어 둔
    **출력 버퍼 프레임 슬립** 바로 그것이다.
    """

    frames = FS * 12
    source = _seeded_band_noise(frames, 11)
    corrupted_frames = int(0.25 * frames)

    for slip in (700, 1_200, 2_000):
        capture = np.roll(source, 142)
        capture[:corrupted_frames] = np.roll(source, 142 + slip)[:corrupted_frames]
        result = check_stream_delay_stability(source, capture, sample_rate=FS)

        assert not result.ok, f"슬립 {slip} 샘플이 통과했습니다: {result.measured}"
        # 판정을 실제로 한 축이 유효창 비율임을 못 박는다 — 다른 두 축은 눈이 멀었다.
        assert result.measured["robust_std_samples"] < 1.0
        assert result.measured["p95_p5_samples"] < 4.0
        assert result.measured["valid_window_ratio"] < MIN_STREAM_DELAY_VALID_WINDOW_RATIO

    # 짝: 슬립이 없으면 같은 신호가 통과한다 (임계가 꺼져서 잡은 것이 아니다).
    clean = check_stream_delay_stability(source, np.roll(source, 142), sample_rate=FS)
    assert clean.ok and clean.measured["valid_window_ratio"] > 0.95


def test_stream_delay_thresholds_sit_in_the_measured_valley():
    """임계가 **정상군 최대와 오염군 최소 사이**에 있는지 못박는다.

    2026-08-06 전수 실측(``data/recorded_broken`` 앞 30 초, 단일 추정기):

    ==================  =====================  ====================
    통계                정상군 47 세션          오염군 80 세션
    ==================  =====================  ====================
    robust-std          1.221 ~   2.992        25.159 ~ 120.620
    p95−p5              4.834 ~  11.980       228.136 ~ 370.770
    유효창 비율         0.857 ~   1.000         0.704 ~   1.000
    ==================  =====================  ====================

    임계를 만지면 이 테스트가 먼저 깨진다 — "통과시키려고 임계를 올리는" 경로를
    막는 것이 목적이다.

    ⚠ 2026-08-06 통합 검증 — **유효창 비율 축의 대조군을 바꿨다.**
    이전에는 이 축을 "오염군(원본 source.wav)도 통과하는 판정가능성 축" 으로 두고
    하한을 오염군 최소(0.704) 아래인 0.50 에 놓았다. 그런데 이 축의 진짜 상대는
    오염군이 아니라 **부분 프레임 슬립**이다: 세션 일부만 어긋나면 어긋난 창이
    통계에서 통째로 빠져 robust-std·p95−p5 가 둘 다 깨끗해지고, 목격자가 이 비율
    하나만 남는다. 실제 QA 코드경로로 주입해 재현한 값이 아래 slip_* 이며, 그때
    오류는 0 건이었다. 그래서 골짜기를 [슬립 주입 최대, 정상군 최소] 로 다시 잡는다.
    """

    normal_max_robust_std, contaminated_min_robust_std = 2.992, 25.159
    normal_max_p95_p5, contaminated_min_p95_p5 = 11.980, 228.136
    # 유효창 비율 — 정상군 최소 vs 부분 프레임 슬립 주입(앞 15~30% 를 700~2000 샘플 이동)
    normal_min_valid_ratio = 0.857
    slip_max_valid_ratio = 0.695
    lost_track_valid_ratio = 0.187

    assert normal_max_robust_std < MAX_STREAM_DELAY_ROBUST_STD_SAMPLES
    assert MAX_STREAM_DELAY_ROBUST_STD_SAMPLES < contaminated_min_robust_std
    assert normal_max_p95_p5 < MAX_STREAM_DELAY_P95_P5_SAMPLES
    assert MAX_STREAM_DELAY_P95_P5_SAMPLES < contaminated_min_p95_p5
    # 임계는 슬립 주입 위·정상군 아래에 있어야 한다. 추적 실패(0.187)는 훨씬 아래다.
    assert lost_track_valid_ratio < slip_max_valid_ratio
    assert slip_max_valid_ratio < MIN_STREAM_DELAY_VALID_WINDOW_RATIO
    assert MIN_STREAM_DELAY_VALID_WINDOW_RATIO < normal_min_valid_ratio


def test_realign_script_gate_is_never_looser_than_the_qa_floor():
    """같은 물리량(유효창 비율)에 임계가 두 곳에 선언돼 있다 — 순서를 못 박는다.

    ``scripts/data/realign_recorded_sessions.py`` 는 재정렬본을 **만들 때** 0.90 을
    요구하고, :data:`MIN_STREAM_DELAY_VALID_WINDOW_RATIO` 는 만들어진 세션을 학습에
    **들일 때** 요구하는 바닥이다. 둘이 다른 것은 의도된 것이지만 순서가 뒤집히면
    느슨한 쪽이 학습 데이터를 지키게 된다 — 2026-08-06 이전이 정확히 그 상태였다
    (재정렬 0.90 vs 바닥 0.50, 대조 코드 없음).
    """

    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts/data/realign_recorded_sessions.py"
    spec = importlib.util.spec_from_file_location("_realign_gate_probe", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # import 시점에 순서를 단언한다

    assert module.DEFAULT_MIN_VALID_WINDOW_RATIO >= MIN_STREAM_DELAY_VALID_WINDOW_RATIO


# ======================================================================================
# D2 — 같은 지연을 두 방법으로 잰 값의 대조
# ======================================================================================
def test_measured_delay_agreement_passes_within_tolerance():
    result = check_measured_delay_agreement(1_672.0, 1_690.0, observation_count=80)
    assert result.ok
    assert result.measured["mismatch_samples"] == 18.0


def test_measured_delay_agreement_fails_on_the_d2_gap():
    """2026-08-05 감사가 찾은 실제 불일치를 그대로 넣는다.

    관측(독립 3방법 일치): 포락선 상관 80세션 중앙값 1672 / 반송파 상관 1663±73 /
    제어기 시작지연 스윕 ~1670.
    유도: P 벌크지연 1602 + argmax tap 247 = 1849.
    """

    result = check_measured_delay_agreement(1_672.0, 1_849.0, observation_count=80)
    assert not result.ok
    assert result.measured["mismatch_samples"] == 177.0
    assert "두 방법으로 잰 값이 다릅니다" in result.detail


def test_derived_delay_uses_the_same_estimator_as_the_observation():
    """유도 쪽은 무게중심이 아니라 **최대점**을 써야 한다.

    상호상관 최대점은 임펄스 응답의 최대점에 대응한다. 무게중심을 쓰면 잔향 꼬리가
    값을 뒤로 끌어당겨 가짜 불일치가 만들어진다 — 실측 P(z) 에서 argmax 247 vs
    무게중심 363.7 로 117 샘플 차이다.
    """

    fir = np.zeros(512)
    fir[247] = 1.0
    fir[400:] = 0.2  # 잔향 꼬리: 무게중심을 뒤로 끌어당긴다
    assert derive_playback_to_error_delay_samples(1_602, fir) == 1_849.0

    with pytest.raises(ValueError):
        derive_playback_to_error_delay_samples(-1, fir)
    with pytest.raises(ValueError):
        derive_playback_to_error_delay_samples(0, np.asarray([np.nan, 1.0]))


# ======================================================================================
# D1 — 코퍼스 누수
# ======================================================================================
def test_corpus_disjoint_passes_on_separate_corpora():
    result = check_corpus_disjoint(
        {"music": ["a.mp3", "b.mp3"], "speech": ["s1.flac"]},
        {"music": ["data/raw/music/000/c.mp3"], "speech": ["data/raw/speech/x.flac"]},
    )
    assert result.ok
    assert result.measured["total_overlap_clips"] == 0


def test_corpus_disjoint_fails_on_the_music_leak():
    """2026-08-05 감사가 찾은 상태 — music 만 100% 겹친다.

    실측: music 60/60 겹침, 그중 55개가 합성 train. 나머지 계열은 0.
    같은 곡에서 두 브랜치가 반대 방향 gradient 를 주고, **music 만 개선되지 않았다**.
    """

    recorded = {
        "music": [f"{index:06d}.mp3" for index in range(60)],
        "speech": ["spk-1.flac", "spk-2.flac"],
    }
    synthetic = {
        # 합성 풀은 저장소 경로를 들고 실측 csv 는 basename 만 든다 —
        # 정규화가 없으면 교집합이 영원히 비어 보인다.
        "music": [f"data/raw/music/fma_small/000/{index:06d}.mp3" for index in range(60)],
        "speech": ["data/raw/speech/LibriSpeech/other.flac"],
    }
    splits = {path: ("train" if index < 55 else "val")
              for index, path in enumerate(synthetic["music"])}

    result = check_corpus_disjoint(recorded, synthetic, synthetic_splits=splits)

    assert not result.ok
    assert "music" in result.detail
    families = result.measured["families"]
    assert families["music"]["overlap_clips"] == 60
    assert families["music"]["overlap_ratio"] == 1.0
    assert families["music"]["overlap_by_synthetic_split"] == {"train": 55, "val": 5}
    assert families["speech"]["overlap_clips"] == 0


def test_corpus_disjoint_normalises_paths_before_comparing():
    """경로 형태가 달라도 같은 원본이면 잡아야 한다.

    이 정규화가 없으면 게이트가 **있는데도 아무것도 잡지 못한다** — 누수 게이트의
    가장 흔한 실패 방식이다.
    """

    result = check_corpus_disjoint(
        {"music": ["Track01.MP3"]},
        {"music": ["data\\raw\\music\\000\\track01.mp3"]},
    )
    assert not result.ok
    assert result.measured["families"]["music"]["overlap_clips"] == 1
