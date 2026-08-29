from __future__ import annotations

import numpy as np
import pytest

from deep_anc.audio_io import pcm_int32_to_float32
from deep_anc.dsp.broadband_interleaved import (
    build_clock_piloted_panel_probe,
    build_nonperiodic_timing_markers,
    validate_submitted_pilot_cross_channel_null,
)
from deep_anc.dsp.control_band_contract import ControlBandContract
from scripts.data import measure_paths_interleaved as mpi
from scripts.data.analyse_broadband_interleaved import (
    analyse_panel_capture,
    derive_global_clock_map,
    estimate_nonperiodic_marker_delay,
    stitch_broadband_panels,
    validate_submitted_pilot_global_map,
)


FS = 48_000


def _frequency_response(frequency, *, delay, taps):
    index = float(delay) + np.arange(len(taps), dtype=np.float64)
    return np.exp(-2j * np.pi * np.outer(frequency, index) / FS) @ np.asarray(taps)


def test_one_panel_fractional_joint_path_keeps_shared_p_s_timing():
    panel = (1400.0, 3200.0)
    probe = build_clock_piloted_panel_probe(
        sample_rate=FS,
        period_seconds=0.125,
        panel_band_hz=panel,
        amplitude=0.003,
    )
    warmup = 4
    repeats = 10
    noise = np.tile(probe.noise_signal, warmup + repeats)
    cancel = np.tile(probe.cancel_signal, warmup + repeats)
    output = np.stack((noise, cancel), axis=1).astype(np.float32)
    p = np.r_[np.zeros(41), [0.8, -0.12, 0.03]]
    s = np.r_[np.zeros(35), [0.6, 0.10, -0.02]]
    err = np.convolve(noise, p)[: noise.size] + np.convolve(cancel, s)[: noise.size]
    ref = 0.5 * err
    starts = [
        (warmup + index) * probe.period_samples for index in range(repeats)
    ]

    period_authority_mask = np.ones(repeats, dtype=np.bool_)
    period_authority_mask[[0, -1]] = False
    result = analyse_panel_capture(
        err=err,
        ref=ref,
        output_pcm_int16=mpi.cw.float32_to_pcm_int16(output),
        probe=probe,
        period_starts=starts,
        panel_band_hz=panel,
        period_authority_mask=period_authority_mask,
    )

    assert result["panel_consistency"]["noise"] > 0.999
    assert result["panel_consistency"]["cancel"] > 0.999
    assert result["relative_tau_max_abs_samples"] < result["selection"][
        "phase_budget_samples"
    ]
    assert result["selection"]["keep"][[0, -1]].tolist() == [False, False]
    assert int(result["selection"]["keep"].sum()) == repeats - 2


def _synthetic_panel(panel_band, *, offset):
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
    transfers = {
        "noise": _frequency_response(
            frequencies["noise"], delay=41, taps=(0.8, -0.12, 0.03)
        ),
        "cancel": _frequency_response(
            frequencies["cancel"], delay=35, taps=(0.6, 0.10, -0.02)
        ),
    }
    for drive in transfers:
        transfers[drive] *= np.exp(
            -2j * np.pi * frequencies[drive] * float(offset) / FS
        )
    stacks = {
        drive: np.tile(value[None, :], (9, 1)) for drive, value in transfers.items()
    }
    return {
        "panel_band_hz": panel_band,
        "frequencies": frequencies,
        "transfers": transfers,
        "crosscheck_transfers": {
            drive: value.copy() for drive, value in transfers.items()
        },
        "aligned_stacks": stacks,
        "aligned_crosscheck_stacks": {
            drive: value.copy() for drive, value in stacks.items()
        },
        "panel_consistency": {"noise": 1.0, "cancel": 1.0},
        "relative_tau_max_abs_samples": 0.0,
        "separation": {},
        "selection": {},
    }


def test_five_panel_stitch_publishes_measured_response_with_diagnostic_compact():
    contract = ControlBandContract.broadband_point_control()
    # global clock map 적용 뒤에는 highband transfer로 추가 offset을 맞추지 않는다.
    offsets = (0.0,) * 5
    panels = [
        _synthetic_panel(band, offset=offset)
        for band, offset in zip(contract.measurement_panels_hz, offsets, strict=True)
    ]

    result = stitch_broadband_panels(
        panels,
        contract=contract,
        fir_length=64,
        pre_roll_samples=8,
        maximum_delay_samples=100,
    )

    assert result["status"] == "PASS"
    assert len(result["panel_stitch"]) == 4
    assert result["applied_per_drive_phase_repair_samples"] == (0.0,) * 10
    assert all(
        row["applied_phase_repair_samples"] == 0.0
        for row in result["panel_stitch"]
    )
    assert all(value > 0.999 for value in result["primary_consistency"])
    assert all(value > 0.999 for value in result["secondary_consistency"])
    for drive in ("noise", "cancel"):
        assert result["drives"][drive]["compact_role"] == "diagnostic_only"
        assert result["drives"][drive]["compact_training_eligible"] is False
        assert result["drives"][drive]["measured_interpolation_holdout"][
            "passed"
        ] is True


@pytest.mark.parametrize("drift_ppm", (-413.931, 0.0, 413.931))
def test_global_clock_map_accumulates_signed_strict_drift_across_panels(
    drift_ppm: float,
):
    drift = drift_ppm * 6000.0 / 1.0e6
    starts = np.arange(293, dtype=np.int64) * 6000
    delays = np.full(starts.size, drift, dtype=np.float64)
    valid = np.ones(starts.size, dtype=np.bool_)
    valid[-1] = False
    result = derive_global_clock_map(
        period_starts=starts,
        common_delay_samples=delays,
        valid=valid,
        period_samples=6000,
    )
    offsets = result["period_offsets_samples"]
    indices = np.asarray([0, 73, 146, 219, 292], dtype=np.float64)
    np.testing.assert_allclose(offsets[[0, 73, 146, 219, 292]], indices * drift)
    assert 1.0e6 * result["slope_samples_per_sample"] == pytest.approx(
        drift_ppm, abs=1e-9
    )
    assert result["maximum_residual_samples"] < 1e-9


def test_global_clock_map_accepts_small_variable_drift_but_rejects_sample_slip():
    starts = np.arange(128, dtype=np.int64) * 6000
    delays = 2.48 + 1e-5 * np.sin(np.linspace(0.0, 2.0 * np.pi, starts.size))
    valid = np.ones(starts.size, dtype=np.bool_)
    valid[-1] = False
    assert derive_global_clock_map(
        period_starts=starts,
        common_delay_samples=delays,
        valid=valid,
        period_samples=6000,
    )["maximum_residual_samples"] < 0.06755189029558945
    nonlinear = 2.48 + 0.005 * np.sin(
        np.linspace(0.0, 2.0 * np.pi, starts.size)
    )
    with pytest.raises(ValueError, match="affine residual"):
        derive_global_clock_map(
            period_starts=starts,
            common_delay_samples=nonlinear,
            valid=valid,
            period_samples=6000,
        )
    slipped = delays.copy()
    slipped[61] += 1.0
    with pytest.raises(ValueError, match="slip/change"):
        derive_global_clock_map(
            period_starts=starts,
            common_delay_samples=slipped,
            valid=valid,
            period_samples=6000,
        )

    piecewise = np.full(starts.size, 2.48, dtype=np.float64)
    piecewise[64:] += 0.10
    with pytest.raises(ValueError, match="affine residual"):
        derive_global_clock_map(
            period_starts=starts,
            common_delay_samples=piecewise,
            valid=valid,
            period_samples=6000,
        )


def test_nonperiodic_marker_selects_tau_not_tau_plus_3000():
    marker = np.zeros((12_000, 2), dtype=np.int16)
    rng = np.random.default_rng(17)
    marker[:, 0] = rng.integers(-90, 91, size=12_000, dtype=np.int16)
    delay = 1386
    raw = np.zeros(12_000 + 4_800, dtype=np.float64)
    raw[delay : delay + 12_000] = marker[:, 0]
    report = estimate_nonperiodic_marker_delay(
        output_pcm=marker,
        err=raw,
        start_frame=0,
        stop_frame=12_000,
        output_channel=0,
    )
    assert report["coarse_delay_samples"] == pytest.approx(delay, abs=1e-6)
    assert report["alias_candidate_count"] == 1
    assert report["search_width_samples"] < 3000.0
    assert not (
        report["search_lower_samples"]
        <= delay + 3000
        <= report["search_upper_samples"]
    )


def test_actual_int16_nonperiodic_marker_has_one_branch_over_zero_to_4800():
    marker, metadata = build_nonperiodic_timing_markers(
        sample_rate=FS,
        amplitude=0.003,
    )
    marker_pcm = mpi.cw.float32_to_pcm_int16(marker)
    delay = 4_311
    raw = np.zeros(marker_pcm.shape[0] + 4_800, dtype=np.float64)
    raw[delay : delay + marker_pcm.shape[0]] = marker_pcm[:, 1]
    report = estimate_nonperiodic_marker_delay(
        output_pcm=marker_pcm,
        err=raw,
        start_frame=0,
        stop_frame=marker_pcm.shape[0],
        output_channel=1,
    )
    assert metadata["delay_alias_period_samples"] == 12_000
    assert report["coarse_delay_samples"] == pytest.approx(delay, abs=1e-6)
    assert report["alias_candidate_count"] == 1
    assert 0.0 <= report["search_lower_samples"] < report[
        "search_upper_samples"
    ] <= 4_800.0
    assert report["search_width_samples"] < 2_999.0


def test_nonperiodic_marker_rejects_zero_or_two_delay_branches():
    marker = np.zeros((12_000, 2), dtype=np.int16)
    rng = np.random.default_rng(23)
    marker[:, 0] = rng.integers(-90, 91, size=12_000, dtype=np.int16)
    silent = np.zeros(12_000 + 4_800, dtype=np.float64)
    with pytest.raises(ValueError, match="candidate_count=0"):
        estimate_nonperiodic_marker_delay(
            output_pcm=marker,
            err=silent,
            start_frame=0,
            stop_frame=12_000,
            output_channel=0,
        )
    ambiguous = np.zeros_like(silent)
    for delay in (800, 3_800):
        ambiguous[delay : delay + 12_000] += marker[:, 0]
    with pytest.raises(ValueError, match="candidate_count=2"):
        estimate_nonperiodic_marker_delay(
            output_pcm=marker,
            err=ambiguous,
            start_frame=0,
            stop_frame=12_000,
            output_channel=0,
        )


def _submitted_pilot_drift_receipt(
    *,
    mutate_high_phase: bool,
    drift_ppm: float = 413.931,
    sample_slip_at_period: int | None = None,
):
    contract = ControlBandContract.broadband_point_control()
    periods = []
    first_probe = None
    for panel_index, band in enumerate(contract.measurement_panels_hz):
        probe = build_clock_piloted_panel_probe(
            sample_rate=FS,
            period_seconds=0.125,
            panel_band_hz=band,
            amplitude=0.003,
        )
        first_probe = probe if first_probe is None else first_probe
        channels = []
        for signal in (probe.noise_signal, probe.cancel_signal):
            spectrum = np.fft.rfft(np.asarray(signal, dtype=np.float64))
            if mutate_high_phase:
                frequency = np.fft.rfftfreq(probe.period_samples, 1.0 / FS)
                mask = frequency > 600.0
                spectrum[mask] *= np.exp(
                    1j * (0.173 * (panel_index + 1) + 0.001 * frequency[mask])
                )
            channels.append(np.fft.irfft(spectrum, n=probe.period_samples))
        period = np.stack(channels, axis=1)
        periods.extend([period] * 12)
    assert first_probe is not None
    # 음의 drift(q>1)에서도 마지막 분석 period 뒤 interpolation source가
    # 충분하도록 마지막 actual-int16 period를 두 번 더 둔다. 분석 starts는
    # 앞 60개만 사용하므로 gate 수나 panel 경계는 바뀌지 않는다.
    periods.extend([periods[-1], periods[-1]])
    output_pcm = mpi.cw.float32_to_pcm_int16(np.concatenate(periods, axis=0))
    output = output_pcm.astype(np.float64) / np.iinfo(np.int16).max
    drift = float(drift_ppm) * 6000.0 / 1.0e6
    q = 1.0 / (1.0 + drift / 6000.0)
    coordinate = q * np.arange(output.shape[0], dtype=np.float64)
    if sample_slip_at_period is not None:
        slip_start = int(sample_slip_at_period) * 6000
        coordinate[slip_start:] += 1.0
    playback_index = np.arange(output.shape[0], dtype=np.float64)
    err = np.interp(
        coordinate,
        playback_index,
        0.8 * output[:, 0] + 0.6 * output[:, 1],
    )
    ref = np.interp(
        coordinate,
        playback_index,
        0.4 * output[:, 0] + 0.3 * output[:, 1],
    )
    captured_int32 = np.rint(
        np.clip(np.stack((err, ref), axis=1), -1.0, 1.0)
        * float(np.iinfo(np.int32).max)
    ).astype(np.int32)
    captured = pcm_int32_to_float32(captured_int32).astype(np.float64)
    err, ref = captured[:, 0], captured[:, 1]
    starts = np.arange(60, dtype=np.int64) * 6000
    observation = mpi.observe_period_clock_ratios(
        err=err,
        ref=ref,
        probe=first_probe,
        period_starts=starts.tolist(),
        max_drift_deviation_samples=1.0,
        min_valid_periods=8,
        clock_band_hz=(150.0, 600.0),
    )
    valid = np.asarray(observation["valid"], dtype=np.bool_).copy()
    boundaries = np.asarray([12, 24, 36, 48], dtype=np.int64)
    valid[np.r_[boundaries - 1, boundaries]] = False
    observation["valid"] = valid
    observation["q"] = np.where(valid, observation["q"], np.nan)
    clock_map = derive_global_clock_map(
        period_starts=starts,
        common_delay_samples=observation["common_delay_samples"],
        valid=valid,
        period_samples=6000,
    )
    receipt = validate_submitted_pilot_global_map(
        err=err,
        ref=ref,
        output_pcm_int16=output_pcm,
        probe=first_probe,
        period_starts=starts,
        clock_observation=observation,
        global_clock_map=clock_map,
    )
    return clock_map, receipt


@pytest.mark.parametrize("drift_ppm", (-413.931, 0.0, 413.931))
def test_actual_int16_err_ref_by_p_s_trajectories_follow_signed_global_clock(
    drift_ppm: float,
):
    clock_map, receipt = _submitted_pilot_drift_receipt(
        mutate_high_phase=False,
        drift_ppm=drift_ppm,
    )
    assert receipt["valid_period_count"] >= 8
    assert receipt["maximum_residual_samples"] <= 0.06755189029558945
    assert receipt["pairwise_trajectory_agreement_samples"] <= 0.06755189029558945
    assert set(receipt["trajectories"]) == {"err", "ref"}
    assert all(
        set(receipt["trajectories"][mic]) == {"noise", "cancel"}
        for mic in ("err", "ref")
    )
    observed_ppm = 1.0e6 * float(clock_map["slope_samples_per_sample"])
    assert observed_ppm == pytest.approx(drift_ppm, abs=0.1)


def test_actual_int16_global_trajectory_rejects_one_sample_slip_at_masked_boundary():
    # row 경계의 local adjacent witness를 보수적으로 버려도, 이후 모든
    # ERR/REF×P/S transfer trajectory에는 영구 1-sample step이 남아야 한다.
    with pytest.raises(ValueError, match="global trajectory"):
        _submitted_pilot_drift_receipt(
            mutate_high_phase=False,
            drift_ppm=413.931,
            sample_slip_at_period=24,
        )


def test_actual_submitted_pcm_division_keeps_clock_map_inside_phase_budget_after_high_mutation():
    baseline_map, baseline = _submitted_pilot_drift_receipt(mutate_high_phase=False)
    mutated_map, mutated = _submitted_pilot_drift_receipt(mutate_high_phase=True)

    assert baseline["highband_phase_used_for_map"] is False
    assert mutated["highband_phase_used_for_map"] is False
    assert baseline["maximum_residual_samples"] < 0.06755189029558945
    assert mutated["maximum_residual_samples"] < 0.06755189029558945
    assert baseline["submitted_pilot_spectra_sha256"] != mutated[
        "submitted_pilot_spectra_sha256"
    ]
    offset_delta = np.asarray(
        baseline_map["period_offsets_samples"], dtype=np.float64
    ) - np.asarray(mutated_map["period_offsets_samples"], dtype=np.float64)
    assert float(np.max(np.abs(offset_delta))) < 0.06755189029558945


def test_actual_submitted_pilot_requires_exact_opposite_channel_null():
    probe = build_clock_piloted_panel_probe(
        sample_rate=FS,
        period_seconds=0.125,
        panel_band_hz=(7800.0, 11400.0),
        amplitude=0.003,
    )
    period = np.rint(
        np.clip(
            np.stack((probe.noise_signal, probe.cancel_signal), axis=1),
            -1.0,
            1.0,
        )
        * 32767.0
    ).astype(np.int16)
    receipt = validate_submitted_pilot_cross_channel_null(
        period,
        sample_rate=FS,
        period_seconds=0.125,
    )
    assert receipt["passed"] is True
    assert max(
        float(receipt["drives"][drive]["cross_to_main_max_ratio"])
        for drive in ("noise", "cancel")
    ) <= 1.0e-12

    mutated = period.copy()
    mutated[17, 1] = np.int16(int(mutated[17, 1]) + 1)
    with pytest.raises(ValueError, match="반대 channel null"):
        validate_submitted_pilot_cross_channel_null(
            mutated,
            sample_rate=FS,
            period_seconds=0.125,
        )
