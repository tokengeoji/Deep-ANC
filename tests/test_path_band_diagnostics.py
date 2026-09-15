"""ESS raw_measurement 포맷의 합성 배열만 사용한다. 장치·legacy 코드를 import하지 않는다."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from deep_anc.eval.path_band_diagnostics import analyze_path_bands
from scripts.eval import analyze_path_bands as cli


@pytest.fixture
def raw_fixture(tmp_path):
    def make(*, delays=(40, 40, 40), gains=(1.0, 1.0, 1.0), raw=True, change=None):
        fs, fir_length = 8000, 256
        rng = np.random.default_rng(42)
        response = rng.normal(size=fir_length) * np.exp(-np.arange(fir_length) / 50) * 0.1
        irs = np.zeros((len(delays), 768))
        for i, (delay, gain) in enumerate(zip(delays, gains)):
            irs[i, delay:delay + fir_length] = gain * response
        metadata = {
            "schema_version": 1, "measurement_kind": "wideband_path_raw_diagnostic",
            "configuration": {"sample_rate": fs, "repeats": len(delays), "fir_length": fir_length,
                              "band_hz": [100.0, 2000.0], "output_channel": "cancel", "output_channel_index": 1,
                              "sweep_seconds": 0.05, "amplitude": 0.005, "block_size": 256,
                              "latency": "low", "pre_roll": 32, "max_delay_ms": 250.0,
                              "max_delay_jitter_samples": 8, "max_delay_jitter_ms": 1.0},
            "telemetry": {"xrun_count": 0, "unexpected_status_count": 0, "completed": True,
                          "callback_error": None},
            "preflight": {"passed_both": True, "channels": []},
            "result": {"repeat_delay_samples": list(delays), "valid_for_model": False,
                       "invalid_reasons": ["synthetic_fixture_not_measurement"]},
        }
        arrays = {"repeat_irs": irs}
        if raw:
            # calibrate_wideband.save_diagnostics의 키/dtype/채널 및 1초 gap 포맷을 그대로 사용.
            sweep = (0.005 * np.sin(2 * np.pi * 1200 * np.arange(400) / fs)).astype(np.float32)
            playback = np.tile(np.r_[np.zeros(fs, np.float32), sweep, np.zeros(fs, np.float32)], len(delays))
            output = np.stack([np.zeros_like(playback), playback], axis=1)
            pcm = np.rint(output * 32767).astype(np.int16)
            inputs = np.rint(rng.normal(scale=0.001, size=output.shape) * 2 ** 31).astype(np.int32)
            normalized = inputs.astype(np.float32) * np.float32(1 / 2 ** 31)
            arrays.update(output=output, output_pcm_int16=pcm, input_raw_int32=inputs,
                          preflight_raw_int32=inputs[:2000], err=normalized[:, 0], ref=normalized[:, 1])
            metadata["telemetry"]["captured_frames"] = len(inputs)
        if change:
            change(arrays, metadata)
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))
        path = tmp_path / "raw_measurement.npz"
        np.savez_compressed(path, **arrays)
        return path
    return make


def _target(report, mode="delay_aligned_compact"):
    return next(row for row in report["metrics"] if row["mode"] == mode and row["band"] == "target")


def test_existing_raw_measurement_fields_are_supported_without_promotion(raw_fixture):
    path = raw_fixture()
    before = path.read_bytes()
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {"output", "output_pcm_int16", "err", "ref", "input_raw_int32",
                                   "preflight_raw_int32", "repeat_irs", "metadata_json"}
    report = analyze_path_bands(path)
    assert report["diagnostic_only"]
    assert not report["performance_claim_allowed"] and not report["promote_secondary_allowed"]
    assert report["repeats_used"] == [0, 1, 2] and report["repeats_dropped"] == []
    assert _target(report)["coherence_min"] == pytest.approx(1.0)
    assert _target(report)["worst_pair_phase_sensitive_cosine"] == pytest.approx(1.0)
    assert report["quality"]["input_snr"] is None and report["quality"]["clock_stability"] is None
    assert not report["quality"]["original_result_claim"]["valid_for_model"]
    assert "consistency_band_hz" not in report
    json.dumps(report, allow_nan=False)
    assert path.read_bytes() == before


def test_alignment_only_changes_delay_phase_and_reports_spread(raw_fixture):
    report = analyze_path_bands(raw_fixture(delays=(40, 43, 49)))
    assert _target(report)["coherence_min"] == pytest.approx(1.0)
    assert _target(report, "fixed_origin")["coherence_min"] < 0.8
    assert report["delay_spread_samples"] == 9
    assert report["delay_spread_ms"] == pytest.approx(1.125)
    assert report["repeats_used"] == [0, 1, 2]  # 기존 1ms gate 초과여도 진단은 모든 반복을 보존


def test_both_modes_use_identical_compact_window_not_full_ir_noise(raw_fixture):
    baseline = analyze_path_bands(raw_fixture(delays=(40, 43, 49)))
    def outside_noise(arrays, _meta):
        arrays["repeat_irs"][:, 500:] = np.arange(3)[:, None] + 0.7
    changed = analyze_path_bands(raw_fixture(delays=(40, 43, 49), change=outside_noise))
    assert baseline["metrics"] == changed["metrics"]


def test_polarity_corruption_is_not_hidden_by_absolute_inner_product(raw_fixture):
    report = analyze_path_bands(raw_fixture(gains=(1.0, 1.0, -1.0)))
    assert _target(report)["coherence_median"] == pytest.approx(1 / 9)
    assert _target(report)["worst_pair_phase_sensitive_cosine"] == pytest.approx(-1)
    assert len(report["pairs"]) == 18  # 3 pairs × 3 bands × 2 modes


def test_magnitude_variation_is_separate_from_shape_similarity(raw_fixture):
    report = analyze_path_bands(raw_fixture(gains=(1.0, 2.0, 4.0)))
    row = _target(report)
    assert row["coherence_median"] == pytest.approx(49 / 63)
    assert row["worst_pair_phase_sensitive_cosine"] == pytest.approx(1)
    assert row["max_pair_magnitude_difference_db"] == pytest.approx(20 * np.log10(4))


def test_zero_power_bins_and_pairs_remain_null(raw_fixture):
    report = analyze_path_bands(raw_fixture(gains=(0.0, 0.0, 0.0)))
    assert _target(report)["coherence_median"] is None
    assert _target(report)["worst_pair_phase_sensitive_cosine"] is None
    assert all(value is None for value in report["spectra"]["fixed_origin"]["coherence"])
    json.dumps(report, allow_nan=False)


def test_exact_1khz_boundary_is_in_high_subband_only(raw_fixture):
    report = analyze_path_bands(raw_fixture())
    low = next(row for row in report["metrics"] if row["band"] == "target_below_1khz")
    high = next(row for row in report["metrics"] if row["band"] == "target_at_or_above_1khz")
    assert (low["low_hz"], low["high_hz"], low["high_edge_inclusive"]) == (800, 1000, False)
    assert (high["low_hz"], high["high_hz"], high["high_edge_inclusive"]) == (1000, 1600, True)
    assert low["fft_bin_count"] + high["fft_bin_count"] == _target(report)["fft_bin_count"]


def test_outside_excitation_and_empty_bin_bands_have_no_numeric_result(raw_fixture):
    def narrow(_arrays, meta):
        meta["configuration"]["band_hz"] = [100, 1000]
    report = analyze_path_bands(raw_fixture(change=narrow))
    assert _target(report)["support_status"] == "outside_excitation"
    assert _target(report)["coherence_median"] is None
    assert _target(report)["max_pair_magnitude_difference_db"] is None
    empty = analyze_path_bands(raw_fixture(), (801, 802))
    assert _target(empty)["support_status"] == "empty_fft_band"
    assert _target(empty)["coherence_median"] is None


def test_missing_raw_quality_is_unknown_not_pass(raw_fixture):
    def no_health(_arrays, meta):
        meta.pop("telemetry")
    report = analyze_path_bands(raw_fixture(raw=False, change=no_health))
    assert report["quality"]["input_raw_int32"] is None
    assert report["quality"]["telemetry"]["xrun_count"] is None
    assert report["quality"]["captured_frames_matches_input"] is None
    assert report["quality"]["inactive_output_channel_nonzero_samples"]["output"] is None
    assert "runtime_health_incomplete" in report["issues"]


@pytest.mark.parametrize("captured", [0, 1, 100_000])
def test_completed_capture_length_contradiction_is_explicit(raw_fixture, captured):
    def mismatch(_arrays, meta):
        meta["telemetry"].update(captured_frames=captured, completed=True)
    report = analyze_path_bands(raw_fixture(change=mismatch))
    assert report["quality"]["telemetry"]["completed"] is True
    assert report["quality"]["captured_frames_matches_input"] is False
    assert "captured_length_mismatch" in report["issues"]
    assert not report["promote_secondary_allowed"]


def test_matching_capture_length_is_not_flagged(raw_fixture):
    report = analyze_path_bands(raw_fixture())
    assert report["quality"]["captured_frames_matches_input"] is True
    assert "captured_length_mismatch" not in report["issues"]


def test_cancel_metadata_with_nonzero_ch0_is_reported_without_guessing_hardware(raw_fixture):
    def wrong_channel(arrays, _meta):
        arrays["output"][20:24, 0] = 0.001
        arrays["output_pcm_int16"][20:24, 0] = np.rint(np.float32(0.001) * 32767).astype(np.int16)
    report = analyze_path_bands(raw_fixture(change=wrong_channel))
    assert report["quality"]["inactive_output_channel_nonzero_samples"] == {
        "output": 4, "output_pcm_int16": 4,
    }
    assert "unexpected_output_channel_samples" in report["issues"]
    assert report["repeats_used"] == [0, 1, 2]
    assert not report["promote_secondary_allowed"]


def test_actual_raw_clips_and_telemetry_faults_are_reported_without_dropping_repeats(raw_fixture):
    def fault(arrays, meta):
        arrays["input_raw_int32"][0, 0] = np.iinfo(np.int32).max
        arrays["err"][0] = np.float32(np.iinfo(np.int32).max) * np.float32(1 / 2 ** 31)
        meta["telemetry"].update(xrun_count=2, completed=False)
    report = analyze_path_bands(raw_fixture(change=fault))
    assert report["quality"]["input_raw_int32"]["channels"][0]["clip_count"] == 1
    assert "input_raw_int32_clipping_observed" in report["issues"]
    assert "runtime_faults_reported" in report["issues"]
    assert report["repeats_used"] == [0, 1, 2]


@pytest.mark.parametrize("fault", ["missing_irs", "nan", "complex", "few_repeats", "count_mismatch",
                                  "no_delays", "delay_negative", "short_ir", "schema", "kind",
                                  "fs_bool", "noise_path", "raw_dtype", "raw_length", "err_mismatch",
                                  "pcm_mismatch", "bad_health"])
def test_malformed_arrays_and_metadata_fail_closed(raw_fixture, fault):
    def change(arrays, meta):
        if fault == "missing_irs": arrays.pop("repeat_irs")
        elif fault == "nan": arrays["repeat_irs"][0, 0] = np.nan
        elif fault == "complex": arrays["repeat_irs"] = arrays["repeat_irs"].astype(complex)
        elif fault == "few_repeats": meta["configuration"]["repeats"] = 2
        elif fault == "count_mismatch": meta["configuration"]["repeats"] = 4
        elif fault == "no_delays": meta["result"].pop("repeat_delay_samples")
        elif fault == "delay_negative": meta["result"]["repeat_delay_samples"][0] = -1
        elif fault == "short_ir": arrays["repeat_irs"] = arrays["repeat_irs"][:, :50]
        elif fault == "schema": meta["schema_version"] = True
        elif fault == "kind": meta["measurement_kind"] = "interleaved"
        elif fault == "fs_bool": meta["configuration"]["sample_rate"] = True
        elif fault == "noise_path": meta["configuration"]["output_channel"] = "noise"
        elif fault == "raw_dtype": arrays["input_raw_int32"] = arrays["input_raw_int32"].astype(np.float64)
        elif fault == "raw_length": arrays["output"] = arrays["output"][:-1]
        elif fault == "err_mismatch": arrays["err"] = np.ones_like(arrays["err"])
        elif fault == "pcm_mismatch": arrays["output_pcm_int16"][0, 0] = 1
        elif fault == "bad_health": meta["telemetry"]["xrun_count"] = True
    with pytest.raises(ValueError):
        analyze_path_bands(raw_fixture(change=change))


def test_cli_writes_three_diagnostic_files_and_refuses_overwrite(raw_fixture, tmp_path):
    raw = raw_fixture()
    out = tmp_path / "report"
    before = raw.read_bytes()
    args = ["--raw-npz", str(raw), "--out", str(out)]
    assert cli.main(args) == 0
    assert {path.name for path in out.iterdir()} == {"report.json", "metrics.csv", "summary.md"}
    report = json.loads((out / "report.json").read_text())
    assert report["promote_secondary_allowed"] is False
    assert "coherence_median" in (out / "metrics.csv").read_text()
    assert "S 모델을 승격하지 않는다" in (out / "summary.md").read_text()
    contents = {path.name: path.read_bytes() for path in out.iterdir()}
    assert cli.main(args) == 1
    assert contents == {path.name: path.read_bytes() for path in out.iterdir()}
    assert raw.read_bytes() == before


def test_cli_invalid_input_creates_no_output(raw_fixture, tmp_path):
    def change(_arrays, meta):
        meta["result"].pop("repeat_delay_samples")
    path = raw_fixture(change=change)
    out = tmp_path / "not_created"
    assert cli.main(["--raw-npz", str(path), "--out", str(out)]) == 1
    assert not out.exists()


def test_cli_rejects_symlink_parent(raw_fixture, tmp_path):
    path = raw_fixture()
    parent = tmp_path / "real"
    parent.mkdir()
    link = tmp_path / "link"
    link.symlink_to(parent, target_is_directory=True)
    assert cli.main(["--raw-npz", str(path), "--out", str(link / "report")]) == 1
    assert list(parent.iterdir()) == []


def test_cli_rejects_symlink_parent_even_before_dotdot(raw_fixture, tmp_path):
    path = raw_fixture()
    parent = tmp_path / "real"
    parent.mkdir()
    link = tmp_path / "link"
    link.symlink_to(parent, target_is_directory=True)
    out = link / ".." / "report"
    assert cli.main(["--raw-npz", str(path), "--out", str(out)]) == 1
    assert not (tmp_path / "report").exists()
    assert list(parent.iterdir()) == []


def test_production_import_and_help_do_not_load_audio_torch_or_legacy():
    code = (
        "import sys; from deep_anc.eval import path_band_diagnostics; "
        "from scripts.eval import analyze_path_bands; "
        "assert not any(k.startswith(('torch', 'sounddevice', 'deep_anc.audio_io')) for k in sys.modules); "
        "assert not any('calibrate_wideband' in k or 'anc_project' in k for k in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    help_result = subprocess.run([sys.executable, "scripts/eval/analyze_path_bands.py", "--help"],
                                 capture_output=True, text=True)
    assert help_result.returncode == 0 and "--raw-npz" in help_result.stdout
