from __future__ import annotations

import numpy as np
import pytest

from deep_anc.dsp.broadband_interleaved import build_clock_piloted_panel_probe
from deep_anc.dsp.broadband_plant_analysis import (
    apply_shared_panel_delay,
    band_roundtrip_metrics,
    compact_fir_identifiability_receipt,
    estimate_bulk_delay_samples,
    fit_real_compact_fir,
    fit_shared_panel_delay,
    merge_stitched_panels,
)
from deep_anc.dsp.control_band_contract import ControlBandContract


FS = 48_000


def _plant(frequency: np.ndarray, *, delay: float, taps: np.ndarray) -> np.ndarray:
    index = delay + np.arange(taps.size, dtype=np.float64)
    return np.exp(-2j * np.pi * np.outer(frequency, index) / FS) @ taps


def _panel(panel_band, *, time_offset: float):
    probe = build_clock_piloted_panel_probe(
        sample_rate=FS,
        period_seconds=0.125,
        panel_band_hz=panel_band,
        amplitude=0.003,
    )
    frequencies = {
        drive: probe.bins_for(drive).astype(np.float64) * FS / probe.period_samples
        for drive in ("noise", "cancel")
    }
    taps = {
        "noise": np.asarray([0.8, -0.15, 0.04, 0.01]),
        "cancel": np.asarray([0.6, 0.12, -0.03, 0.02]),
    }
    transfers = {
        "noise": _plant(frequencies["noise"], delay=41.0, taps=taps["noise"]),
        "cancel": _plant(frequencies["cancel"], delay=35.0, taps=taps["cancel"]),
    }
    for drive in transfers:
        transfers[drive] *= np.exp(
            -2j * np.pi * frequencies[drive] * time_offset / FS
        )
    return {
        "panel_band_hz": panel_band,
        "frequencies": frequencies,
        "transfers": transfers,
    }


def test_shared_panel_delay_uses_one_correction_for_p_and_s_exact_overlap():
    reference = _panel((1400.0, 3200.0), time_offset=0.0)
    current = _panel((2800.0, 6000.0), time_offset=2.375)

    report = fit_shared_panel_delay(
        reference_frequencies=reference["frequencies"],
        reference_transfers=reference["transfers"],
        current_frequencies=current["frequencies"],
        current_transfers=current["transfers"],
        overlap_band_hz=(2800.0, 3200.0),
    )

    assert report["passed"] is True
    assert report["shared_delay_samples"] == pytest.approx(2.375, abs=2e-5)
    assert report["drives"]["noise"]["tone_count"] >= 8
    assert report["drives"]["cancel"]["tone_count"] >= 8


def test_shared_stitch_rejects_drive_specific_phase_repair():
    reference = _panel((1400.0, 3200.0), time_offset=0.0)
    current = _panel((2800.0, 6000.0), time_offset=1.0)
    current["transfers"]["cancel"] *= np.exp(
        -2j * np.pi * current["frequencies"]["cancel"] * 0.8 / FS
    )

    report = fit_shared_panel_delay(
        reference_frequencies=reference["frequencies"],
        reference_transfers=reference["transfers"],
        current_frequencies=current["frequencies"],
        current_transfers=current["transfers"],
        overlap_band_hz=(2800.0, 3200.0),
    )
    assert report["passed"] is False


def test_merge_then_single_real_compact_fir_roundtrips_all_control_bands():
    contract = ControlBandContract.broadband_point_control()
    offsets = (0.0, 1.25, -0.75, 2.5, 3.125)
    panels = [
        _panel(band, time_offset=offset)
        for band, offset in zip(contract.measurement_panels_hz, offsets, strict=True)
    ]

    # 첫 panel을 authority로 두고, synthetic fixture의 알려진 offset을 적용한다. delay
    # 추정 자체는 위 두 테스트가 검증하며 여기서는 merge/FIR 단일 적합을 검증한다.
    for panel, offset in zip(panels, offsets, strict=True):
        panel["transfers"] = apply_shared_panel_delay(
            panel["frequencies"], panel["transfers"], delay_samples=offset
        )
    frequency, measured, counts = merge_stitched_panels(panels, drive="noise")
    assert np.max(counts) > 1
    fitted = fit_real_compact_fir(
        frequency,
        measured,
        effective_delay_samples=41,
        fir_length=64,
        ridge_relative=1e-10,
    )
    rows = band_roundtrip_metrics(
        frequency,
        measured,
        fitted["reconstructed_transfer"],
        subbands_hz=contract.point_control_subbands_hz,
    )
    assert fitted["passed"] is True
    assert fitted["complex_agreement"] > 0.999999
    assert fitted["relative_error"] < 1e-5
    assert all(row["passed"] for row in rows)


def test_compact_fir_refuses_more_real_taps_than_complex_observations_support():
    frequency = np.arange(100.0, 900.0, 100.0)
    measured = np.ones_like(frequency, dtype=np.complex128)
    with pytest.raises(ValueError, match="tone이 부족"):
        fit_real_compact_fir(
            frequency,
            measured,
            effective_delay_samples=0,
            fir_length=17,
        )


def test_current_16hz_measured_grid_cannot_identify_1024_tap_training_plant():
    # broadband v4/v5 per-drive canonical grid: 104..11400 Hz, 16 Hz spacing.
    # 2*N >= taps라는 개수 검사는 통과하지만 band-limited Fourier design의 실제
    # numeric rank/condition은 통과하지 못한다.
    frequency = np.arange(104.0, 11_400.0 + 0.1, 16.0, dtype=np.float64)
    receipt = compact_fir_identifiability_receipt(
        frequency,
        effective_delay_samples=1245,
        fir_length=1024,
        ridge_relative=1.0e-8,
    )
    assert receipt["tone_count"] in {706, 707}
    assert receipt["real_equation_count"] >= 1024
    assert receipt["numeric_rank"] < 1024
    assert receipt["condition_number"] is not None
    assert receipt["condition_number"] > receipt["maximum_condition_number"]
    assert receipt["compact_role"] == "diagnostic_only"
    assert receipt["compact_training_eligible"] is False
    assert receipt["ridge_does_not_restore_identifiability"] is True


def test_bulk_delay_estimator_resolves_fractional_wideband_slope():
    frequency = np.arange(150.0, 11_314.0, 16.0)
    taps = np.asarray([0.8, -0.1, 0.03])
    measured = _plant(frequency, delay=1386.375, taps=taps)
    estimated = estimate_bulk_delay_samples(
        frequency,
        measured,
        minimum_delay_samples=1200.0,
        maximum_delay_samples=1500.0,
    )
    # 반환은 FIR의 group delay까지 포함한 matched phase delay다. 같은 fixture를 반복해
    # 재현되는지와 1-sample 이내인지를 요구하고, exact tap onset으로 과잉해석하지 않는다.
    assert estimated == pytest.approx(1386.3, abs=1.0)
