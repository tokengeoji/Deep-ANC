"""재생↔녹음 시간축 부기(결함 2) 테스트.

각 게이트는 **그것을 FAIL 시키는 테스트**와 짝을 이룬다. "좋아 보이는 데이터에서 한 번
통과시켜 보고 끝" 이 결함 군집 B 의 발생기였다 — recorded QA 는 80/80 PASS 였는데
학습 데이터의 시간축이 붕괴해 있었다.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, lfilter

from deep_anc.data.timeline import (
    TIMELINE_METHOD,
    LagTrack,
    TimelineReport,
    TimelineSettings,
    align_source_to_adc,
    estimate_lag_track,
    median_coherence,
    warp_by_lag_track,
)

FS = 48000


def _wideband(n: int, seed: int = 20260805) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b, a = butter(4, [100.0 / (FS / 2), 1600.0 / (FS / 2)], btype="band")
    return lfilter(b, a, rng.standard_normal(n)).astype(np.float64)


def _make_session(
    seconds: float = 6.0,
    *,
    lag_bias: float = 1500.0,
    lag_amplitude: float = 134.0,
    lag_period_s: float = 4.5,
    acoustic_delay: int = 142,
    seed: int = 20260805,
):
    """알려진 L(t) 를 주입한 (source, witness=REF, holdout=ERR) 3인조를 만든다."""

    n = int(seconds * FS)
    pad = 4000
    wide = _wideband(n + 2 * pad, seed=seed)
    t = np.arange(n, dtype=np.float64)
    lag = lag_bias + lag_amplitude * np.sin(2.0 * np.pi * t / FS / lag_period_s)
    pos = t + pad - lag
    base = np.floor(pos).astype(np.int64)
    frac = pos - base
    witness = (1.0 - frac) * wide[base] + frac * wide[base + 1]
    holdout = np.concatenate([np.zeros(acoustic_delay), witness[:-acoustic_delay]])
    source = wide[pad : pad + n]
    return source, witness, holdout, lag


# ---------------------------------------------------------------- 왕복 정확도
def test_known_time_warp_is_recovered_and_coherence_is_restored():
    """정답을 아는 L(t) 를 넣고 되찾는다. 이것이 유일하게 정직한 검증이다.

    실측 세션에서 coh² 가 올라갔다는 것만으로는 "L(t) 를 맞게 추정했다"를 증명하지
    못한다 — 정답을 아는 신호로 재야 한다.
    """

    source, witness, holdout, lag_true = _make_session()
    settings = TimelineSettings(sample_rate=FS)

    track = estimate_lag_track(source, witness, settings)
    centres = (np.asarray(track.times_s) * FS).astype(np.int64)
    valid = np.asarray(track.valid, dtype=bool)
    assert valid.mean() > 0.9
    error = np.abs(np.asarray(track.lag_samples)[valid] - lag_true[centres[valid]])
    # 창 0.25s 안에서 L 이 최대 47 샘플 움직이는 극단 조건이라 창 평균 오차가 남는다.
    # 실측 세션의 변조 진폭은 이보다 3~5배 작다.
    assert float(np.max(error)) < 4.0

    aligned, report = align_source_to_adc(source, witness, holdout, FS, settings=settings)
    assert report.method == TIMELINE_METHOD
    assert report.coh2_150_600_before < 0.2
    assert report.coh2_150_600_after > 0.95
    # 검증은 추정에 쓰지 않은 채널로만 했다 — 잔여 지연은 주입한 음향 지연이어야 한다.
    assert abs(report.aligned_lag_median_samples - 142.0) < 2.0
    assert report.aligned_lag_robust_std_samples < 2.0
    assert aligned.shape == source.shape


def test_alignment_does_not_touch_the_holdout_channel():
    """ERR(홀드아웃)을 완전히 다른 신호로 바꿔도 추정된 워프는 같아야 한다.

    같은 채널로 추정하고 검증하면 그것은 게이트가 아니라 자기증명이다. 이 테스트는
    홀드아웃 규약이 코드로 지켜지는지를 본다.
    """

    source, witness, holdout, _ = _make_session(seconds=4.0)
    settings = TimelineSettings(sample_rate=FS)
    aligned_a, _ = align_source_to_adc(source, witness, holdout, FS, settings=settings)
    other = _wideband(holdout.size, seed=7)
    aligned_b, report_b = align_source_to_adc(source, witness, other, FS, settings=settings)
    assert np.allclose(aligned_a, aligned_b)
    # 그리고 무관한 홀드아웃에서는 검증이 **떨어져야** 한다.
    assert report_b.coh2_150_600_after < 0.2


# ---------------------------------------------------------------- 실패 증명
def test_collapsed_timebase_is_not_silently_repaired():
    """복구 불가능한 입력에서 통과값이 나오면 안 된다 (negative fixture).

    증인 채널이 소스와 아무 관계가 없으면(=REF 마이크가 죽었거나 배선이 틀렸으면)
    워프는 의미가 없고, coh² 가 낮게 나와야 한다. 여기서 높은 값이 나오면 워프가
    **소스를 홀드아웃에 맞춰 과적합**하고 있다는 뜻이다.
    """

    n = int(4.0 * FS)
    source = _wideband(n, seed=1)
    witness = _wideband(n, seed=2)
    holdout = _wideband(n, seed=3)
    _, report = align_source_to_adc(source, witness, holdout, FS)
    assert report.coh2_150_600_after < 0.3
    assert report.coh2_ref_err_150_600 < 0.3


def test_silent_source_is_detected_and_its_windows_are_rejected():
    """소스에 내용이 없는 구간은 유효창에서 빠져야 한다.

    L(t) 추정은 소스가 150–700Hz 에 에너지를 가질 때만 가능하다. "추정할 수 없었다"를
    "지연이 0이었다" 로 적으면 워프가 조용히 틀린다.
    """

    source, witness, holdout, _ = _make_session(seconds=8.0)
    quiet = source.copy()
    quiet[int(3.0 * FS) :] = 0.0
    settings = TimelineSettings(sample_rate=FS)
    track_full = estimate_lag_track(source, witness, settings)
    track_quiet = estimate_lag_track(quiet, witness, settings)
    assert track_full.valid_window_ratio > 0.9
    assert track_quiet.valid_window_ratio < 0.6
    del holdout


# ---------------------------------------------------------------- 타입 강제
@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"sample_rate": FS, "track_band_hz": (700.0, 150.0)},
        {"sample_rate": FS, "track_band_hz": (150.0, 40000.0)},
        {"sample_rate": FS, "hop_seconds": 1.0, "window_seconds": 0.25},
        {"sample_rate": FS, "refine_search_samples": 900},
        {"sample_rate": FS, "coarse_search_samples": 0},
    ],
)
def test_impossible_timeline_settings_cannot_be_constructed(kwargs):
    """물리적으로 불가능한 값은 **생성 시점에** 막힌다. 주석으로는 이미 실패해 봤다."""

    with pytest.raises(ValueError):
        TimelineSettings(**kwargs)


def test_timeline_report_rejects_out_of_range_coherence():
    with pytest.raises(ValueError):
        TimelineReport(
            track_band_hz=(150.0, 700.0),
            track_window_s=0.25,
            track_hop_s=0.0625,
            valid_window_ratio=0.95,
            raw_lag_median_samples=1500.0,
            raw_lag_ptp_samples=200.0,
            raw_lag_std_samples=60.0,
            aligned_lag_median_samples=142.0,
            aligned_lag_std_samples=2.0,
            aligned_lag_robust_std_samples=1.0,
            aligned_lag_p95_p5_samples=7.0,
            aligned_valid_window_ratio=0.95,
            coh2_150_600_before=0.05,
            coh2_150_600_after=1.4,          # ← 불가능
            coh2_600_1600_before=0.01,
            coh2_600_1600_after=0.8,
            coh2_ref_err_150_600=0.99,
        )


def test_robust_std_is_not_fooled_by_a_few_lobe_jumps():
    """원시 std 를 게이트에 쓰면 소수의 로브 점프가 판정을 뒤집는다.

    실측: 세션 121917 의 잔여 궤적은 창 1071개 중 16개(1.5%)가 로브 한 칸 건너뛰어
    std 가 2.2 → 7.8 로 부풀었다. p95−p5 는 7.16 으로 멀쩡했다.
    """

    lags = np.full(1000, 142.0)
    lags += np.random.default_rng(0).normal(0.0, 1.0, size=1000)
    lags[:15] = 62.0                     # 로브 점프 1.5%
    track = LagTrack(
        times_s=np.arange(1000) * 0.0625,
        lag_samples=lags,
        quality=np.full(1000, 0.5),
        valid=np.ones(1000, dtype=bool),
        sample_rate=FS,
    )
    assert track.std_samples > 8.0        # 원시 std 는 속는다
    assert track.robust_std_samples < 2.0  # robust 는 안 속는다
    assert track.p95_p5_samples < 5.0


# ---------------------------------------------------------------- 워프 수치
def test_warp_by_lag_track_reproduces_a_pure_integer_delay():
    """상수 L 이면 워프는 그냥 시프트여야 한다 — 보간 자체의 정확도 확인."""

    n = 4 * FS
    source = _wideband(n, seed=11)
    track = LagTrack(
        times_s=np.linspace(0.5, (n / FS) - 0.5, 20),
        lag_samples=np.full(20, 100.0),
        quality=np.full(20, 0.9),
        valid=np.ones(20, dtype=bool),
        sample_rate=FS,
    )
    warped = warp_by_lag_track(source, track, length=n)
    core = slice(2000, n - 2000)
    assert np.max(np.abs(warped[core] - source[core.start - 100 : core.stop - 100])) < 1e-3


def test_median_coherence_matches_the_shared_invariant_implementation():
    """코히런스는 불변식 모듈의 구현을 그대로 쓴다 (두 번째 정의 금지)."""

    from deep_anc.dsp.invariants import check_stream_coherence

    a = _wideband(FS * 3, seed=5)
    b = np.concatenate([np.zeros(64), a[:-64]])
    mine = median_coherence(a, b, sample_rate=FS, band_hz=(150.0, 600.0))
    theirs = check_stream_coherence(
        a, b, sample_rate=FS, band_hz=(150.0, 600.0), min_coherence=0.0
    ).measured["coherence"]
    assert mine == pytest.approx(theirs)
    assert mine > 0.95
