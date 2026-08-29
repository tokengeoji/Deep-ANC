"""평가 지표의 해석적 검증."""

import numpy as np
import pytest

from deep_anc.eval.metrics import (
    attenuation_db,
    band_power,
    band_nmse_db,
    intersect_frequency_bands,
    nmse_db,
    octave_band_attenuation,
    segment_stats,
)
from deep_anc.eval.trusted_subbands import (
    STRICT_TRUSTED_SUBBANDS_HZ,
    strict_subband_includes_upper_edge,
)

FS = 48000


def test_nmse_half_amplitude():
    rng = np.random.default_rng(0)
    d = rng.standard_normal(FS)
    e = 0.5 * d                                   # 진폭 절반 → -6.02 dB
    assert abs(nmse_db(d, e) + 6.0206) < 0.01
    assert abs(attenuation_db(d, e) - 6.0206) < 0.01


def test_nmse_uses_mean_power_for_different_durations():
    """OFF/ON 길이가 달라도 같은 진폭 비율은 같은 NMSE여야 한다."""
    d = np.ones(FS)
    e = 0.5 * np.ones(FS * 2)
    assert nmse_db(d, e) == pytest.approx(-6.0206, abs=0.01)


def test_band_nmse_separates_trusted_and_fullband():
    t = np.arange(FS * 2) / FS
    low = np.sin(2 * np.pi * 300 * t)
    high = np.sin(2 * np.pi * 1200 * t)
    d = low + high
    e = 0.1 * low + high

    trusted = band_nmse_db(d, e, FS, (150, 600))
    fullband = nmse_db(d, e)
    assert trusted == pytest.approx(-20.0, abs=0.05)
    assert fullband == pytest.approx(-2.967, abs=0.05)


def test_band_nmse_handles_different_durations():
    td = np.arange(FS) / FS
    te = np.arange(FS * 2) / FS
    d = np.sin(2 * np.pi * 300 * td)
    e = 0.5 * np.sin(2 * np.pi * 300 * te)
    assert band_nmse_db(d, e, FS, (150, 600)) == pytest.approx(-6.0206, abs=0.01)


@pytest.mark.parametrize(
    ("first", "second", "match"),
    [
        ((600, 150), (80, 800), "잘못된"),
        ((-1, 600), (80, 800), "잘못된"),
        ((150, 600), (800, 1600), "교집이 비어"),
        ((150, float("nan")), (80, 800), "유한한"),
    ],
)
def test_intersect_frequency_bands_fails_fast(first, second, match):
    with pytest.raises(ValueError, match=match):
        intersect_frequency_bands(first, second, FS / 2)


def test_intersect_frequency_bands_uses_measured_and_target_overlap():
    assert intersect_frequency_bands((150, 600), (80, 800), FS / 2) == (150, 600)


def test_strict_subband_partition_does_not_double_count_1000hz_boundary():
    """1000 Hz target은 1000–1600만 채우며 두 부대역 coverage가 될 수 없다."""

    samples = 1_536
    time = np.arange(samples, dtype=np.float64) / FS
    target = np.sin(2.0 * np.pi * 1000.0 * time)
    lower = STRICT_TRUSTED_SUBBANDS_HZ[2]
    upper = STRICT_TRUSTED_SUBBANDS_HZ[3]
    lower_power = band_power(
        target,
        FS,
        lower,
        include_upper=strict_subband_includes_upper_edge(lower),
    )
    upper_power = band_power(
        target,
        FS,
        upper,
        include_upper=strict_subband_includes_upper_edge(upper),
    )

    assert upper_power > 0.1
    assert lower_power < upper_power * 1.0e-12


def test_band_nmse_fails_when_fft_has_no_band_bin():
    with pytest.raises(ValueError, match="FFT.*빈이 없"):
        band_nmse_db(np.ones(8), np.ones(8), FS, (150, 600))


def test_octave_band_selective():
    """300Hz 성분만 제거된 경우 250Hz 밴드(177~354Hz)에서만 큰 감쇠."""
    t = np.arange(FS * 2) / FS
    tone300 = np.sin(2 * np.pi * 300 * t)
    tone1200 = np.sin(2 * np.pi * 1200 * t)
    d = tone300 + tone1200
    e = 0.01 * tone300 + tone1200                 # 300Hz 만 40dB 감쇠

    bands = octave_band_attenuation(d, e, FS, [125, 250, 500, 1000, 2000], (150, 600))
    by_center = {b["center_hz"]: b for b in bands}
    assert by_center[250]["attenuation_db"] > 30
    assert abs(by_center[1000]["attenuation_db"]) < 3
    assert by_center[250]["trusted"] is True
    assert by_center[2000]["trusted"] is False    # S(z) 유효대역(150~600) 밖


def test_segment_stats():
    rng = np.random.default_rng(1)
    d = rng.standard_normal(FS * 3)
    e = 0.1 * d
    stats = segment_stats(d, e, FS, seg_seconds=1.0)
    assert stats["n_segments"] == 3
    assert abs(stats["median_db"] - 20.0) < 0.5
