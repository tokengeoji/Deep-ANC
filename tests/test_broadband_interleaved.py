from __future__ import annotations

import numpy as np
import pytest

from deep_anc.dsp.broadband_interleaved import (
    BROADBAND_CANONICAL_PANEL_BANDS_HZ,
    build_clock_piloted_panel_probe,
    build_nonperiodic_timing_markers,
    fixed_clock_pilot_complex_spectrum,
)


@pytest.mark.parametrize(
    "panel",
    [(100.0, 1800.0), (1400.0, 3200.0), (7800.0, 11400.0)],
)
def test_clock_piloted_panel_has_disjoint_dac_bins_and_lowband_witness(panel):
    probe = build_clock_piloted_panel_probe(
        sample_rate=48_000,
        period_seconds=0.125,
        panel_band_hz=panel,
        amplitude=0.003,
    )
    resolution = probe.sample_rate / probe.period_samples
    noise = probe.noise_bins * resolution
    cancel = probe.cancel_bins * resolution

    assert not set(probe.noise_bins).intersection(set(probe.cancel_bins))
    assert np.all(probe.noise_bins % 2 == 0)
    assert np.all(probe.cancel_bins % 2 == 1)
    assert probe.guard_bins() == 1
    assert np.sum((noise >= 150.0) & (noise <= 600.0)) >= 8
    assert np.sum((cancel >= 150.0) & (cancel <= 600.0)) >= 8
    assert np.sum((noise >= panel[0]) & (noise <= panel[1])) >= 8
    assert np.sum((cancel >= panel[0]) & (cancel <= panel[1])) >= 8
    assert np.max(np.abs(probe.noise_signal)) <= 0.003 + 1e-9
    assert np.max(np.abs(probe.cancel_signal)) <= 0.003 + 1e-9


def test_clock_piloted_probe_is_deterministic():
    kwargs = dict(
        sample_rate=48_000,
        period_seconds=0.125,
        panel_band_hz=(5400.0, 8500.0),
        amplitude=0.003,
    )
    left = build_clock_piloted_panel_probe(**kwargs)
    right = build_clock_piloted_panel_probe(**kwargs)
    np.testing.assert_array_equal(left.noise_signal, right.noise_signal)
    np.testing.assert_array_equal(left.cancel_signal, right.cancel_signal)


def test_fixed_pilot_complex_spectrum_is_panel_independent():
    probes = [
        build_clock_piloted_panel_probe(
            sample_rate=48_000,
            period_seconds=0.125,
            panel_band_hz=band,
            amplitude=0.003,
        )
        for band in BROADBAND_CANONICAL_PANEL_BANDS_HZ
    ]
    authority = fixed_clock_pilot_complex_spectrum(
        sample_rate=48_000, period_seconds=0.125
    )
    for drive in ("noise", "cancel"):
        bins, _ = authority[drive]
        spectra = [np.fft.rfft(getattr(probe, f"{drive}_signal")) for probe in probes]
        for observed in spectra[1:]:
            np.testing.assert_allclose(observed[bins], spectra[0][bins], atol=2e-9, rtol=0.0)


def test_nonperiodic_markers_have_four_hz_grid_and_unique_4800_range():
    markers, metadata = build_nonperiodic_timing_markers(
        sample_rate=48_000, amplitude=0.003
    )
    assert markers.shape == (12_000, 2)
    assert metadata["frequency_gcd_hz"] == 4.0
    assert metadata["delay_alias_period_samples"] == 12_000
    assert metadata["delay_alias_period_samples"] > 4_800
    assert np.max(np.abs(markers)) <= 0.003 + 1e-9
