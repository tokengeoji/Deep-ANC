"""단일 스트림 덕트 전달맵의 신호 규약·안전 게이트 테스트."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from scripts.bench import measure_duct_transfer_map as transfer_map


def _channels() -> dict[str, int]:
    return {
        "error_mic": 0,
        "reference_mic": 1,
        "noise_out": 0,
        "cancel_out": 1,
    }


def _valid_probe_report() -> dict:
    return {
        "channels": [
            {"channel": 0, "valid": True, "clip_ratio": 0.0},
            {"channel": 1, "valid": True, "clip_ratio": 0.0},
        ]
    }


def _full_valid_probe_report(frames: int = 64) -> tuple[np.ndarray, dict]:
    raw = np.arange(frames * 2, dtype=np.int32).reshape(frames, 2)
    channels = []
    for channel in range(2):
        channels.append(
            {
                "channel": channel,
                "rms_dbfs": -46.0,
                "peak": 0.01,
                "clip_ratio": 0.0,
                "unique_codes": frames,
                "raw_min": int(raw[:, channel].min()),
                "raw_max": int(raw[:, channel].max()),
                "stuck": False,
                "valid": True,
            }
        )
    return raw, {"channels": channels, "device": 1, "sample_rate": 48_000}


def test_defaults_are_low_level_single_stream_and_banded():
    args = transfer_map.build_parser().parse_args([])
    assert args.excitation == "ess"
    assert args.band == [80.0, 1600.0]
    assert args.amplitude == 0.005
    assert args.amplitude <= transfer_map.MAX_AMPLITUDE
    assert args.repeats == 3
    assert args.latency == "high"
    assert args.confirm_volume_minimum is False


def test_confirmation_precedes_yaml_and_hardware_access(monkeypatch, capsys):
    monkeypatch.setattr(
        transfer_map,
        "load_yaml",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("YAML must not be read")
        ),
    )
    assert transfer_map.main([]) == 2
    message = capsys.readouterr().err
    assert "--confirm-volume-minimum" in message or "--confirm-speaker" in message


def test_amplitude_and_channel_map_are_hard_gated():
    args = transfer_map.build_parser().parse_args(["--confirm-volume-minimum"])
    args.amplitude = 0.02001
    with pytest.raises(ValueError, match="0.02"):
        transfer_map.validate_options(
            args, sample_rate=48_000, hardware_channels=_channels()
        )

    args.amplitude = 0.005
    bad = {**_channels(), "cancel_out": 0}
    with pytest.raises(ValueError, match="noise_out/cancel_out"):
        transfer_map.validate_options(args, sample_rate=48_000, hardware_channels=bad)

    args.band = [100.0, 1000.0]
    with pytest.raises(ValueError, match="80 1600"):
        transfer_map.validate_options(
            args, sample_rate=48_000, hardware_channels=_channels()
        )


def test_time_division_program_never_drives_both_speakers_together():
    excitation = np.linspace(-0.005, 0.005, 800, dtype=np.float32)
    output, bounds, gap = transfer_map.build_time_division_program(
        excitation,
        sample_rate=8_000,
        gap_seconds=0.25,
        repeats=3,
        noise_channel=0,
        cancel_channel=1,
    )

    ns_start, ns_stop = bounds["noise_out"]
    cs_start, cs_stop = bounds["cancel_out"]
    separator_start, separator_stop = bounds["separator"]
    assert gap == 2_000
    assert np.any(output[ns_start:ns_stop, 0] != 0.0)
    assert np.all(output[ns_start:ns_stop, 1] == 0.0)
    assert np.all(output[separator_start:separator_stop] == 0.0)
    assert np.all(output[cs_start:cs_stop, 0] == 0.0)
    assert np.any(output[cs_start:cs_stop, 1] != 0.0)
    assert np.all(np.count_nonzero(output, axis=1) <= 1)
    assert float(np.max(np.abs(output))) <= 0.005


def test_gcc_phat_sign_is_err_minus_ref():
    fs = 8_000
    rng = np.random.default_rng(20260803)
    source = rng.normal(size=4096)
    ref = source.copy()
    err = np.concatenate([np.zeros(135), source[:-135]])
    lag, _confidence = transfer_map.gcc_phat_lag(
        err,
        ref,
        sample_rate=fs,
        band_hz=(80.0, 1600.0),
        max_lag_samples=300,
    )
    assert lag == 135

    err = source.copy()
    ref = np.concatenate([np.zeros(142), source[:-142]])
    lag, _confidence = transfer_map.gcc_phat_lag(
        err,
        ref,
        sample_rate=fs,
        band_hz=(80.0, 1600.0),
        max_lag_samples=300,
    )
    assert lag == -142


def _repeat_epoch_with_tdoa(*, repeats: int, shot_size: int, lag: int) -> np.ndarray:
    rng = np.random.default_rng(9102 + lag)
    epoch = np.zeros((repeats * shot_size, 2), dtype=np.float64)
    for repeat in range(repeats):
        source = rng.normal(size=shot_size)
        start = repeat * shot_size
        if lag >= 0:
            epoch[start : start + shot_size, 1] = source
            epoch[start : start + shot_size, 0] = np.concatenate(
                [np.zeros(lag), source[: shot_size - lag]]
            )
        else:
            delay = -lag
            epoch[start : start + shot_size, 0] = source
            epoch[start : start + shot_size, 1] = np.concatenate(
                [np.zeros(delay), source[: shot_size - delay]]
            )
    return epoch


def test_same_clock_tdoa_repeats_match_observed_sign_convention():
    for expected in (135, -142):
        epoch = _repeat_epoch_with_tdoa(repeats=5, shot_size=4096, lag=expected)
        report = transfer_map.relative_tdoa_by_repeat(
            epoch,
            error_channel=0,
            reference_channel=1,
            repeats=5,
            shot_size=4096,
            sample_rate=48_000,
            band_hz=(80.0, 1600.0),
            max_lag_samples=500,
            max_jitter_samples=12,
        )
        assert report["repeat_lag_err_minus_ref_samples"] == [expected] * 5
        assert report["spread_samples"] == 0
        assert report["stable"] is True
        assert "same I2S" not in report["buffer_common_property"]  # 설명은 한국어로 보존


def test_silence_cannot_be_declared_a_stable_tdoa():
    report = transfer_map.relative_tdoa_by_repeat(
        np.zeros((3 * 4096, 2), dtype=np.float64),
        error_channel=0,
        reference_channel=1,
        repeats=3,
        shot_size=4096,
        sample_rate=48_000,
        band_hz=(80.0, 1600.0),
        max_lag_samples=500,
        max_jitter_samples=12,
    )
    assert report["stable"] is False
    assert report["signal_valid"] is False
    assert report["confidence_valid"] is False


def test_periodic_ambient_without_speaker_response_fails_driven_excess_gate():
    fs = 8_000
    repeats = 3
    gap = 2_000
    excitation_samples = 800
    shot = 2 * gap + excitation_samples
    time_axis = np.arange(repeats * shot, dtype=np.float64) / fs
    # 출력과 무관하게 계속 존재하는 위상 고정 300Hz 주변음/전기 crosstalk.
    response = 0.01 * np.sin(2.0 * np.pi * 300.0 * time_axis)
    irs = np.zeros((repeats, 1024), dtype=np.float64)
    irs[:, 100] = 1.0  # 다른 gate가 완벽해도 driven gap 대비 상승이 없어야 한다.
    model = {"repeat_onset_samples": [100] * repeats}
    report = transfer_map.driven_response_snr_report(
        response,
        irs,
        model,
        repeats=repeats,
        gap_samples=gap,
        excitation_samples=excitation_samples,
        max_delay_samples=400,
        fir_length=512,
        sample_rate=fs,
        band_hz=(80.0, 1600.0),
    )
    assert report["driven_excess_valid"] is False
    assert max(report["repeat_driven_excess_db"]) < 1.0
    assert report["valid"] is False


def test_tdoa_peak_at_search_boundary_is_invalid():
    epoch = _repeat_epoch_with_tdoa(repeats=3, shot_size=4096, lag=100)
    report = transfer_map.relative_tdoa_by_repeat(
        epoch,
        error_channel=0,
        reference_channel=1,
        repeats=3,
        shot_size=4096,
        sample_rate=48_000,
        band_hz=(80.0, 1600.0),
        max_lag_samples=100,
        max_jitter_samples=12,
    )
    assert report["stable"] is False
    assert report["search_boundary_valid"] is False


def test_repeat_frequency_response_contains_phase_coherence_and_group_delay():
    fs = 8_000
    delay = 40
    irs = np.zeros((3, 1024), dtype=np.float64)
    irs[:, delay] = 0.5
    response = transfer_map.frequency_response_from_repeat_irs(
        irs,
        sample_rate=fs,
        band_hz=(80.0, 1600.0),
    )

    assert response["repeat_transfer_complex"].shape[0] == 3
    mask = (response["frequencies_hz"] >= 100.0) & (
        response["frequencies_hz"] <= 1500.0
    )
    assert np.median(response["coherence"][mask]) == pytest.approx(1.0)
    assert np.median(response["magnitude_db"][mask]) == pytest.approx(
        20 * np.log10(0.5), abs=0.02
    )
    assert np.median(response["group_delay_seconds"][mask]) == pytest.approx(
        delay / fs, abs=1e-8
    )
    assert all(row["valid"] for row in response["band_rows"])


def test_multitone_frequency_table_uses_only_excited_bins():
    fs = 8_000
    irs = np.zeros((3, 1024), dtype=np.float64)
    irs[:, 30] = 1.0
    tones = np.asarray([100.0, 180.0, 400.0, 700.0, 1300.0])
    response = transfer_map.frequency_response_from_repeat_irs(
        irs,
        sample_rate=fs,
        band_hz=(80.0, 1600.0),
        excited_frequencies_hz=tones,
    )
    assert response["frequency_support"] == "multitone_bins_only"
    assert response["supported_frequency_bin_count"] == tones.size
    assert all(row["valid"] for row in response["band_rows"])


def test_callback_adc_dac_timestamps_are_separate_and_jitter_gated():
    rows = [
        {
            "frame_start": index * 256,
            "frames": 256,
            "input_buffer_adc_time": 10.0 + index * 256 / 48_000,
            "current_time": 10.02 + index * 256 / 48_000,
            "output_buffer_dac_time": 10.03 + index * 256 / 48_000,
        }
        for index in range(5)
    ]
    report = transfer_map.summarize_callback_times(
        rows,
        sample_rate=48_000,
        max_jitter_seconds=0.001,
    )
    assert report["stable"] is True
    assert report["dac_minus_adc_seconds"]["median"] == pytest.approx(0.03)
    assert report["dac_minus_adc_seconds"]["median_samples"] == pytest.approx(1440)

    rows[-1]["output_buffer_dac_time"] += 0.003
    report = transfer_map.summarize_callback_times(
        rows,
        sample_rate=48_000,
        max_jitter_seconds=0.001,
    )
    assert report["stable"] is False


@pytest.mark.parametrize("timestamp", [0.0, 17.0])
def test_constant_or_zero_callback_timestamps_are_invalid(timestamp):
    rows = [
        {
            "frame_start": index * 256,
            "frames": 256,
            "input_buffer_adc_time": timestamp,
            "current_time": timestamp,
            "output_buffer_dac_time": timestamp,
        }
        for index in range(4)
    ]
    report = transfer_map.summarize_callback_times(
        rows,
        sample_rate=48_000,
        max_jitter_seconds=0.001,
    )
    assert report["stable"] is False
    assert report["strictly_progressing"] is False


def test_cross_epoch_portaudio_offset_change_is_invalid():
    common = {
        "stable": True,
        "dac_minus_adc_seconds": {"median": 0.030},
    }
    result = transfer_map.cross_epoch_timestamp_consistency(
        {
            "noise_out": common,
            "cancel_out": {
                "stable": True,
                "dac_minus_adc_seconds": {"median": 0.033},
            },
        },
        sample_rate=48_000,
        max_difference_seconds=0.001,
    )
    assert result["stable"] is False
    assert result["noise_minus_cancel_offset_samples"] == pytest.approx(-144.0)


def _absolute(onset: float, *, stable: bool = True) -> dict:
    return {
        "stable": stable,
        "callback_frame_onset_median_samples": onset,
        "repeat_callback_frame_onset_samples": [onset, onset, onset],
        "timestamp_corrected_median_samples": onset,
        "timestamp_corrected_dac_to_adc_path_samples": [onset, onset, onset],
    }


def test_causal_budget_reproduces_109_sample_digital_lead_and_negative_deadline():
    budget = transfer_map.calculate_causal_budget(
        sample_rate=48_000,
        noise_tdoa={
            "stable": True,
            "median_lag_err_minus_ref_samples": 140.0,
        },
        ns_to_err_absolute=_absolute(1489.0),
        cs_to_err_absolute=_absolute(1342.0),
        cross_epoch_timestamps={"stable": True},
        processing_samples=256,
    )

    assert budget["valid"] is True
    assert budget["acoustic_reference"]["processing_deadline_samples"] == -1202.0
    assert budget["acoustic_reference"]["causal_without_prediction"] is False
    assert budget["digital_reference"]["required_source_lead_samples"] == 109.0
    assert budget["acoustic_reference"]["cancel_arrival_alignment_error_samples"] > 0


def test_causal_budget_is_invalid_when_absolute_output_delay_is_unstable():
    budget = transfer_map.calculate_causal_budget(
        sample_rate=48_000,
        noise_tdoa={
            "stable": True,
            "median_lag_err_minus_ref_samples": 135.0,
        },
        ns_to_err_absolute=_absolute(1800.0),
        cs_to_err_absolute=_absolute(1700.0, stable=False),
        cross_epoch_timestamps={"stable": True},
        processing_samples=256,
    )
    assert budget["valid"] is False
    assert budget["acoustic_reference"]["valid"] is False
    assert budget["digital_reference"]["valid"] is False


def test_routing_topology_matches_duct_signs_and_rejects_swapped_outputs():
    positions = {
        "noise_speaker": 0.0,
        "reference_mic": 0.1,
        "cancel_speaker": 1.05,
        "error_mic": 1.1,
    }
    paths = {
        "ns_to_ref": {"absolute_delay": _absolute(1000.0)},
        "ns_to_err": {"absolute_delay": _absolute(1140.0)},
        "cs_to_ref": {"absolute_delay": _absolute(1100.0)},
        "cs_to_err": {"absolute_delay": _absolute(974.0)},
    }
    tdoa = {
        "noise_out": {
            "stable": True,
            "median_lag_err_minus_ref_samples": 140.0,
        },
        "cancel_out": {
            "stable": True,
            "median_lag_err_minus_ref_samples": -126.0,
        },
    }
    report = transfer_map.routing_topology_gate(
        positions_m=positions,
        speed_of_sound_mps=343.0,
        sample_rate=48_000,
        tdoa_results=tdoa,
        path_results=paths,
    )
    assert report["valid"] is True
    assert report["drives"]["noise_out"]["expected_sign"] == 1
    assert report["drives"]["cancel_out"]["expected_sign"] == -1

    swapped = {
        "noise_out": tdoa["cancel_out"],
        "cancel_out": tdoa["noise_out"],
    }
    report = transfer_map.routing_topology_gate(
        positions_m=positions,
        speed_of_sound_mps=343.0,
        sample_rate=48_000,
        tdoa_results=swapped,
        path_results=paths,
    )
    assert report["valid"] is False


def test_single_stream_differential_delay_is_invalid_if_either_path_unstable():
    valid = transfer_map.differential_delay_report(
        _absolute(1500.0),
        _absolute(1400.0),
        sample_rate=48_000,
        max_jitter_samples=48,
        sign_convention="positive",
    )
    assert valid["stable"] is True
    assert valid["median_difference_samples"] == 100.0
    assert valid["single_stream_time_division"] is True

    invalid = transfer_map.differential_delay_report(
        _absolute(1500.0),
        _absolute(1400.0, stable=False),
        sample_rate=48_000,
        max_jitter_samples=48,
        sign_convention="positive",
    )
    assert invalid["stable"] is False


def test_xrun_clip_or_unverified_final_mute_blocks_complete_map():
    valid_path = {"valid": True}
    paths = {key: valid_path for key in transfer_map.PATH_ORDER}
    stable = {"stable": True}
    output = np.zeros((32, 2), dtype=np.float32)
    output[:, 0] = 0.005
    common = {
        "preflight_report": _valid_probe_report(),
        "measurement_report": _valid_probe_report(),
        "output_float": output,
        "output_pcm": transfer_map.float32_to_pcm_int16(output),
        "path_results": paths,
        "tdoa_results": {"noise_out": stable, "cancel_out": stable},
        "timestamp_results": {"noise_out": stable, "cancel_out": stable},
        "cross_epoch_timestamps": stable,
        "differential_results": {"err": stable, "ref": stable},
        "routing_topology": {"valid": True},
        "causal_budget": {"valid": True},
    }
    passed, reasons, _ = transfer_map.overall_quality_gate(
        **common,
        telemetry={
            "xrun_count": 1,
            "unexpected_status_count": 0,
            "completed": True,
            "normal_drain_completed": True,
        },
        final_mute={
            "attempted": True,
            "both_channels_zero": True,
            "stream_closed": True,
            "underflow_blocks": 0,
        },
    )
    assert passed is False
    assert "xrun_detected" in reasons

    passed, reasons, _ = transfer_map.overall_quality_gate(
        **common,
        telemetry={
            "xrun_count": 0,
            "unexpected_status_count": 0,
            "completed": True,
            "normal_drain_completed": True,
        },
        final_mute={"attempted": True, "both_channels_zero": False},
    )
    assert passed is False
    assert "final_mute_unverified" in reasons

    invalid_measurement = {
        "channels": [
            {"channel": 0, "valid": False, "clip_ratio": 0.0},
            {"channel": 1, "valid": True, "clip_ratio": 0.0},
        ]
    }
    passed, reasons, _ = transfer_map.overall_quality_gate(
        **{**common, "measurement_report": invalid_measurement},
        telemetry={
            "xrun_count": 0,
            "unexpected_status_count": 0,
            "completed": True,
            "normal_drain_completed": True,
        },
        final_mute={
            "attempted": True,
            "both_channels_zero": True,
            "stream_closed": True,
            "underflow_blocks": 0,
        },
    )
    assert passed is False
    assert "measurement_both_mics_invalid" in reasons


def test_capture_failure_still_calls_final_zero_flush(monkeypatch):
    calls: list[str] = []

    def fail_capture(*_args, **_kwargs):
        calls.append("capture")
        raise RuntimeError("synthetic capture failure")

    def mute(*_args, **_kwargs):
        calls.append("mute")
        return {
            "attempted": True,
            "both_channels_zero": True,
            "stream_closed": True,
            "underflow_blocks": 0,
        }

    monkeypatch.setattr(transfer_map, "capture_time_division", fail_capture)
    monkeypatch.setattr(transfer_map, "_safe_final_mute", mute)
    report: dict = {}
    with pytest.raises(RuntimeError, match="synthetic"):
        transfer_map.capture_and_mute(
            object(),
            final_mute_report=report,
            sample_rate=48_000,
            block_size=512,
            latency="high",
            input_device=1,
            output_device=2,
            output_float=np.zeros((32, 2), dtype=np.float32),
        )
    assert calls == ["capture", "mute"]
    assert report["both_channels_zero"] is True


def test_normal_callback_completion_uses_stop_not_abort():
    calls: list[str] = []

    class CallbackStop(Exception):
        pass

    class CallbackAbort(Exception):
        pass

    class FakeStream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]
            self.active = False

        def start(self):
            calls.append("start")
            self.active = True
            indata = np.zeros((32, 2), dtype=np.int32)
            outdata = np.zeros((32, 2), dtype=np.int16)
            time_info = {
                "inputBufferAdcTime": 10.0,
                "currentTime": 10.01,
                "outputBufferDacTime": 10.02,
            }
            try:
                self.callback(indata, outdata, 32, time_info, None)
            except CallbackStop:
                self.active = False

        def stop(self):
            calls.append("stop")
            self.active = False

        def abort(self):
            calls.append("abort")
            self.active = False

        def close(self):
            calls.append("close")

    class FakeSD:
        Stream = FakeStream

    FakeSD.CallbackStop = CallbackStop
    FakeSD.CallbackAbort = CallbackAbort

    _raw, _pcm, telemetry = transfer_map.capture_time_division(
        FakeSD,
        sample_rate=48_000,
        block_size=32,
        latency="high",
        input_device=1,
        output_device=2,
        output_float=np.zeros((32, 2), dtype=np.float32),
    )
    assert calls == ["start", "stop", "close"]
    assert telemetry["normal_drain_completed"] is True


def test_artifacts_include_npz_json_markdown_and_refuse_overwrite(tmp_path):
    paths = {
        "npz": tmp_path / "map.npz",
        "json": tmp_path / "map.json",
        "markdown": tmp_path / "map.md",
    }
    metadata = {"schema_version": 1, "result": {"duct_identification_complete": False}}
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata)),
        "ns_to_err_repeat_irs": np.ones((3, 16)),
    }
    transfer_map.save_artifacts(
        paths,
        arrays=arrays,
        metadata=metadata,
        markdown="# synthetic\n",
    )
    assert all(path.exists() for path in paths.values())
    with np.load(paths["npz"], allow_pickle=False) as data:
        assert data["ns_to_err_repeat_irs"].shape == (3, 16)
    assert Path(paths["markdown"]).read_text(encoding="utf-8") == "# synthetic\n"

    with pytest.raises(FileExistsError, match="덮어쓰지"):
        transfer_map.save_artifacts(
            paths,
            arrays=arrays,
            metadata=metadata,
            markdown="# overwrite\n",
        )


def test_artifact_publish_failure_rolls_back_partial_outputs(tmp_path, monkeypatch):
    paths = {
        "npz": tmp_path / "map.npz",
        "json": tmp_path / "map.json",
        "markdown": tmp_path / "map.md",
    }
    real_link = transfer_map.os.link
    calls = 0

    def fail_second_link(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic publish failure")
        return real_link(source, target)

    monkeypatch.setattr(transfer_map.os, "link", fail_second_link)
    with pytest.raises(OSError, match="synthetic publish"):
        transfer_map.save_artifacts(
            paths,
            arrays={"x": np.ones(4)},
            metadata={"schema_version": 1},
            markdown="# test\n",
        )
    assert not any(path.exists() for path in paths.values())
    assert not list(tmp_path.glob(".*.tmp"))


def test_invalid_preflight_report_can_still_render_diagnostics_markdown():
    metadata = {
        "created_at_utc": "2026-08-03T00:00:00+00:00",
        "configuration": {
            "excitation": "ess",
            "band_hz": [80.0, 1600.0],
            "amplitude_peak": 0.005,
        },
        "paths": {},
        "relative_tdoa": {},
        "causal_budget": {"valid": False},
        "result": {
            "duct_identification_complete": False,
            "measurement_claim_allowed": False,
            "invalid_reasons": ["preflight_both_mics_invalid"],
        },
    }
    markdown = transfer_map.render_markdown(metadata)
    assert "INVALID" in markdown
    assert "preflight_both_mics_invalid" in markdown
    assert "ANC/FxLMS 성능 주장 가능: **False**" in markdown


def test_full_cli_pipeline_with_synthetic_audio_only(tmp_path, monkeypatch):
    """장치를 열지 않는 합성 2×2 덕트로 main→3종 산출물까지 검증한다."""
    outputs = {
        "npz": tmp_path / "map.npz",
        "json": tmp_path / "map.json",
        "markdown": tmp_path / "map.md",
    }
    monkeypatch.setattr(
        transfer_map,
        "output_paths",
        lambda _prefix, *, include_plot: {
            **outputs,
            **({"plot": tmp_path / "map.png"} if include_plot else {}),
        },
    )
    preflight_raw, preflight = _full_valid_probe_report()
    monkeypatch.setattr(
        transfer_map.calibration,
        "_capture_preflight",
        lambda *_args, **_kwargs: (preflight_raw, preflight),
    )
    monkeypatch.setattr(
        transfer_map,
        "resolve_alsa_portaudio_device",
        lambda *_args, **_kwargs: 2,
    )
    monkeypatch.setattr(transfer_map, "assert_live_pcm_clock_preconditions", lambda *_args: None)
    monkeypatch.setattr(transfer_map, "assert_measurement_preconditions", lambda *_args: None)
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace())

    def synthetic_capture(_sd, **kwargs):
        output = np.asarray(kwargs["output_float"], dtype=np.float64)
        fs = int(kwargs["sample_rate"])
        schedule = int(round(0.030 * fs))

        def path(delay, gain):
            impulse = np.zeros(schedule + delay + 96, dtype=np.float64)
            impulse[schedule + delay] = gain
            impulse[schedule + delay + 23] = 0.1 * gain
            return impulse

        err = signal.fftconvolve(output[:, 0], path(154, 0.50))[: output.shape[0]]
        err += signal.fftconvolve(output[:, 1], path(7, 0.45))[: output.shape[0]]
        ref = signal.fftconvolve(output[:, 0], path(14, 0.40))[: output.shape[0]]
        ref += signal.fftconvolve(output[:, 1], path(133, 0.35))[: output.shape[0]]
        floating = np.stack([err, ref], axis=1)
        raw = np.rint(
            np.clip(floating, -1.0, 1.0) * float(2**31 - 1)
        ).astype(np.int32)
        rows = []
        for start in range(0, output.shape[0], int(kwargs["block_size"])):
            timestamp = 100.0 + start / fs
            rows.append(
                {
                    "frame_start": start,
                    "frames": min(int(kwargs["block_size"]), output.shape[0] - start),
                    "input_buffer_adc_time": timestamp,
                    "current_time": timestamp + 0.015,
                    "output_buffer_dac_time": timestamp + 0.030,
                }
            )
        kwargs["final_mute_report"].update(
            {
                "attempted": True,
                "zero_blocks": 2,
                "underflow_blocks": 0,
                "both_channels_zero": True,
                "stream_closed": True,
            }
        )
        return (
            raw,
            transfer_map.float32_to_pcm_int16(output),
            {
                "callback_count": len(rows),
                "callback_status_count": 0,
                "xrun_count": 0,
                "unexpected_status_count": 0,
                "callback_error": None,
                "completed": True,
                "normal_drain_completed": True,
                "callback_time_info": rows,
            },
        )

    monkeypatch.setattr(transfer_map, "capture_and_mute", synthetic_capture)
    exit_code = transfer_map.main(
            [
                "--confirm-volume-minimum",
                "--confirm-speaker",
                "--confirm-user-present",
            "--excitation-seconds",
            "0.2",
            "--gap-seconds",
            "0.25",
            "--fir-length",
            "512",
            "--max-delay-ms",
            "100",
            "--out-prefix",
            "results/ignored_by_test_patch",
        ]
    )
    assert exit_code == 0
    assert all(path.exists() for path in outputs.values())
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["result"]["duct_identification_complete"] is True
    assert payload["result"]["anc_performance_claim_allowed"] is False
    assert payload["relative_tdoa"]["noise_out"][
        "median_lag_err_minus_ref_samples"
    ] == pytest.approx(140.0)
    assert payload["relative_tdoa"]["cancel_out"][
        "median_lag_err_minus_ref_samples"
    ] == pytest.approx(-126.0)
    assert payload["routing_topology"]["valid"] is True
    assert all(
        path["driven_response_snr"]["valid"]
        for path in payload["paths"].values()
    )
