from __future__ import annotations

import builtins
import hashlib

import numpy as np
from scipy.signal import fftconvolve

from deep_anc.dsp.control_band_contract import BROADBAND_POINT_CONTROL_SUBBANDS_HZ
from deep_anc.dsp.fullband_causal_probe import (
    FIT_PERIOD_SAMPLES,
    HOLDOUT_PERIOD_SAMPLES,
    build_signal_plan,
    off_grid_holdout_bins,
)
from scripts.data import analyse_fullband_causal as analysis
from scripts.data import measure_paths_fullband_causal as measure


def test_signal_plan_is_exact_low_crest_channel_separated_and_block_aligned() -> None:
    plan, pcm = build_signal_plan()
    assert pcm.dtype == np.int16
    assert pcm.shape == (1_787_136, 2)
    assert pcm.shape[0] % 256 == 0
    assert plan["output"]["duration_seconds"] == 37.232
    assert plan["output"]["active_slot_duration_seconds"] == 34.730666666666664
    assert plan["output"]["peak_pcm"] == 98
    assert plan["output"]["idle_channel_nonzero_count"] == 0
    for role in ("fit", "holdout"):
        receipt = plan[role]
        assert 0.00140 <= receipt["rms"] <= 0.00143
        assert 6.3 <= receipt["crest_db"] <= 6.7
        assert receipt["all_rfft_bins_nonzero"] is True
        assert receipt["design_condition_number"] <= 1.10
    assert FIT_PERIOD_SAMPLES == 32_768
    assert HOLDOUT_PERIOD_SAMPLES == 30_720
    assert off_grid_holdout_bins().size == 6_646


def test_live_is_locked_before_sounddevice_import(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise AssertionError("live lock 전에 sounddevice를 import했습니다")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert measure.FULLBAND_CAUSAL_LIVE_AUTHORITY is None
    assert measure.main(["--execute-live"]) == 2


def test_raw_first_publisher_is_immutable_and_keeps_callback_witness(tmp_path) -> None:
    plan, pcm = build_signal_plan()
    starts = np.arange(0, pcm.shape[0], 256, dtype=np.int64)
    counts = np.full(starts.size, 256, dtype=np.int64)
    times = np.arange(starts.size, dtype=np.float64) * (256.0 / 48_000.0) + 1.0
    callback = {
        "callback_start_frames": starts,
        "callback_frame_counts": counts,
        "input_buffer_adc_time": times,
        "output_buffer_dac_time": times + 0.001,
        "callback_current_time": times + 0.002,
    }
    level = tmp_path / "level.json"
    meter = tmp_path / "meter.npz"
    level.write_bytes(b"level-evidence")
    meter.write_bytes(b"meter-raw")
    metadata = {
        "operator_confirmations": {
            "speaker_output": True,
            "user_present": True,
            "volume_minimum": True,
            "routing_and_geometry": True,
            "same_amplifier_setting": True,
        },
        "level_evidence": {
            "path": str(level),
            "sha256": hashlib.sha256(level.read_bytes()).hexdigest(),
        },
        "meter": {
            "path": str(meter),
            "sha256": hashlib.sha256(meter.read_bytes()).hexdigest(),
        },
        "xrun_count": 0,
    }
    target = tmp_path / "raw"
    published = measure.publish_raw_capture(
        session_dir=target,
        plan=plan,
        planned_pcm=pcm,
        submitted_pcm=pcm.copy(),
        input_raw_int32=np.zeros((pcm.shape[0], 2), dtype=np.int32),
        preflight_raw_int32=np.zeros((32, 2), dtype=np.int32),
        callback_time_info=callback,
        metadata=metadata,
    )
    assert published["valid"] is True
    assert published["metadata"]["analysis_status"] == "NOT_RUN_RAW_FIRST"
    assert published["metadata"]["callback_timing"]["sample_slip_count"] == 0
    assert (target / "raw_measurement.npz").is_file()
    assert (target / "metadata.json").is_file()
    try:
        measure.publish_raw_capture(
            session_dir=target,
            plan=plan,
            planned_pcm=pcm,
            submitted_pcm=pcm,
            input_raw_int32=np.zeros((pcm.shape[0], 2), dtype=np.int32),
            preflight_raw_int32=np.zeros((32, 2), dtype=np.int32),
            callback_time_info=callback,
            metadata=metadata,
        )
    except FileExistsError:
        pass
    else:  # pragma: no cover
        raise AssertionError("기존 raw session을 덮어썼습니다")


def _known_kernel(delay: int, scale: float) -> np.ndarray:
    result = np.zeros(delay + 97, dtype=np.float64)
    result[delay] = scale
    result[delay + 13] = -0.16 * scale
    result[delay + 47] = 0.09 * scale
    result[delay + 96] = -0.03 * scale
    return result


def _panel_binding(kernel: np.ndarray) -> dict[str, np.ndarray]:
    frequencies = np.concatenate(
        [
            np.linspace(low + 0.1, high - 0.1, 8, dtype=np.float64)
            for low, high in BROADBAND_POINT_CONTROL_SUBBANDS_HZ
        ]
    )
    index = np.arange(kernel.size, dtype=np.float64)
    transfer = np.exp(
        -2j * np.pi * np.outer(frequencies, index) / 48_000.0
    ) @ kernel
    return {"frequencies_hz": frequencies, "transfer": transfer}


def test_known_causal_fir_selects_smallest_support_and_passes_off_grid() -> None:
    plan, pcm = build_signal_plan()
    primary = _known_kernel(173, 0.8)
    secondary = _known_kernel(311, -0.55)
    responses = {
        "primary": fftconvolve(pcm[:, 0].astype(np.float64) / 32_767.0, primary)[
            : pcm.shape[0]
        ],
        "secondary": fftconvolve(
            pcm[:, 1].astype(np.float64) / 32_767.0, secondary
        )[: pcm.shape[0]],
    }
    clock = {
        "schema": "absolute_dac_q_timewarp_v1",
        "valid_fit_repeats": 16,
        "valid_holdout_repeats": 8,
        "minimum_adjacent_score": 1.0,
        "maximum_residual_samples": 0.0,
        "sample_slip_count": 0,
    }
    receipt = analysis.analyse_resampled_capture(
        plan=plan,
        submitted_pcm=pcm,
        responses_by_path=responses,
        clock_receipt=clock,
        panel_bindings={
            "primary": _panel_binding(primary),
            "secondary": _panel_binding(secondary),
        },
        final_noise_floor_receipts={
            path: {
                "schema": "final_tail_input_noise_floor_v1",
                "valid": True,
                "last_tail_vs_input_noise_db": 0.0,
            }
            for path in ("primary", "secondary")
        },
        synthetic_fixture=True,
    )
    assert receipt["status"] == "PASS"
    assert receipt["canonical_training_eligible"] is False
    assert receipt["synthetic_fixture_only"] is True
    for path, expected_delay in (("primary", 173), ("secondary", 311)):
        result = receipt["paths"][path]
        assert result["delay_samples"] == expected_delay
        assert result["selected_support_samples"] == 1024
        candidate = result["candidates"][0]
        assert candidate["passed"] is True
        assert candidate["off_grid_holdout"]["tone_count"] == 6646
        assert candidate["off_grid_holdout"]["relative_error"] < 1.0e-10
        assert candidate["tail"]["tail_to_retained_l1_ratio_upper"] < 1.0e-9
        assert candidate["tail"]["heldout_induced_output_ratio_upper"] < 1.0e-9
        assert all(row["passed"] for row in candidate["panel_cross_binding"])


def test_clock_gate_and_missing_panel_fail_closed() -> None:
    plan, pcm = build_signal_plan()
    response = np.zeros(pcm.shape[0], dtype=np.float64)
    bad_clock = {
        "schema": "absolute_dac_q_timewarp_v1",
        "valid_fit_repeats": 16,
        "valid_holdout_repeats": 8,
        "minimum_adjacent_score": 1.0,
        "maximum_residual_samples": 0.06755189029558946 + 1.0e-6,
        "sample_slip_count": 0,
    }
    try:
        analysis.analyse_resampled_capture(
            plan=plan,
            submitted_pcm=pcm,
            responses_by_path={"primary": response, "secondary": response},
            clock_receipt=bad_clock,
            panel_bindings={},
            final_noise_floor_receipts={},
        )
    except ValueError as exc:
        assert "clock" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("나쁜 clock receipt가 통과했습니다")
