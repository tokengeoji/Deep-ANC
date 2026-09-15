"""Acoustic 진단 CLI의 파일 보존·진단 범위·계산 불가 표기 계약."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "scripts" / "eval" / "analyze_acoustic_session.py"


@pytest.fixture
def session(tmp_path):
    fs = 8000
    samples = 16 * fs
    time = np.arange(samples) / fs
    reference = 0.04 * np.sin(2 * np.pi * 125 * time) + 0.02 * np.sin(2 * np.pi * 1500 * time)
    gain = np.zeros(samples)
    gain[5 * fs:11 * fs] = 1.0
    signals = {
        "fs": fs, "ref": reference, "err": reference * (1.0 - 0.5 * gain),
        "source": np.zeros(samples), "control": -0.5 * reference * gain, "anc_gain": gain,
    }
    recording = tmp_path / "recording.npz"
    np.savez(recording, **signals)
    secondary = tmp_path / "secondary.npz"
    np.savez(secondary, fir=np.array([1.0, 0.1]), delay_samples=8, sample_rate=fs,
             consistency_band_hz=[80.0, 800.0], excitation_band_hz=[80.0, 1600.0],
             coherence_median=0.95, calibration_block_size=64, calibration_latency="low",
             output_channel="cancel")
    hardware = tmp_path / "hardware.yaml"
    hardware.write_text(yaml.safe_dump({
        "audio": {"sample_rate": fs, "block_size": 64, "latency": "low"},
        "channels": {"error_mic": 0, "reference_mic": 1, "noise_out": 0, "cancel_out": 1},
        "dc_blocker_r": 0.995,
    }), encoding="utf-8")
    duct = tmp_path / "duct.yaml"
    duct.write_text(yaml.safe_dump({
        "secondary_path": {"npz": str(secondary), "handoff_extra_samples": 64},
        "positions_m": {"reference_mic": 0.1, "error_mic": 1.1},
        "duct": {"speed_of_sound_mps": 343.0},
    }), encoding="utf-8")
    config = tmp_path / "runtime.yaml"
    config.write_text(yaml.safe_dump({
        "hardware_config": str(hardware), "duct_config": str(duct),
        "reference": "mic", "controller": "fxlms", "digital_reference_lead_samples": 0,
        "hop": 64, "noise": {"enabled": False}, "safety": {"control_limit": 0.1},
    }), encoding="utf-8")
    return {"recording": recording, "secondary": secondary, "config": config,
            "signals": signals, "out": tmp_path / "analysis", "fs": fs}


def _args(session, *extra):
    return ["--npz", str(session["recording"]), "--config", str(session["config"]),
            "--source-family", "music", "--out", str(session["out"]), *extra]


def _run(session, *extra):
    return subprocess.run([sys.executable, str(CLI), *_args(session, *extra)],
                          cwd=REPO_ROOT, text=True, capture_output=True, timeout=30)


def _csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _report(session):
    return json.loads((session["out"] / "report.json").read_text(encoding="utf-8"))


def test_complete_legacy_session_keeps_untrusted_bands_and_one_numeric_source(session):
    before = {key: hashlib.sha256(session[key].read_bytes()).hexdigest()
              for key in ("recording", "secondary", "config")}
    result = _run(session, "--set", "safety.control_limit=0.05", "--set", "noise.enabled=false")
    assert result.returncode == 0, result.stderr
    report = _report(session)
    assert report["diagnostic_only"] and not report["performance_claim_allowed"]
    assert report["all_cycles_complete"] and report["comparison_available"]
    assert not report["provenance"]["recording_context_verified"]
    assert "legacy_recording_context_unverified" in report["issues"]
    assert report["analysis_config"]["overrides"] == ["safety.control_limit=0.05", "noise.enabled=false"]
    rows = _csv(session["out"] / "metrics.csv")
    full = next(row for row in rows if row["band"] == "full")
    expected = next(row for row in report["metrics"] if row["band"] == "full")
    assert float(full["observed_err_reduction_db"]) == expected["observed_err_reduction_db"]
    assert float(full["observed_err_reduction_db"]) == pytest.approx(20 * np.log10(2), abs=1e-8)
    assert next(row for row in rows if row["band"] == "high_1000_nyquist")["trusted"] == "false"
    for name in ("target_800_1600", "target_800_1000", "target_1000_1600"):
        exported = next(row for row in rows if row["band"] == name)
        reported = next(row for row in report["metrics"] if row["band"] == name)
        assert exported["trusted"] == "false"
        assert float(exported["off_pre_power"]) == reported["off_pre_power"]
        assert float(exported["on_power"]) == reported["on_power"]
    assert len(rows) == len(report["metrics"])
    summary = (session["out"] / "summary.md").read_text(encoding="utf-8")
    assert full["observed_err_reduction_db"] in summary
    assert "legacy_recording_context_unverified" in summary
    assert "performance_claim_allowed=false" in summary
    windows = _csv(session["out"] / "windows.csv")
    assert len(windows) == len(report["windows"])
    assert all(row["diagnostic_only"] == "true" and row["performance_claim_allowed"] == "false"
               for row in [*rows, *windows])
    assert {path.name for path in session["out"].iterdir()} == {
        "metrics.csv", "windows.csv", "report.json", "summary.md",
    }
    assert {key: hashlib.sha256(session[key].read_bytes()).hexdigest() for key in before} == before


@pytest.mark.parametrize("frequency,active_bands", [
    (800, {"target_800_1600", "target_800_1000"}),
    (1000, {"target_800_1600", "target_1000_1600"}),
    (1600, set()),
])
def test_cli_target_boundary_tones_keep_json_csv_and_windows_consistent(session, frequency, active_bands):
    signals = session["signals"]
    time = np.arange(signals["err"].size) / session["fs"]
    signals["ref"] = 0.04 * np.sin(2 * np.pi * frequency * time)
    signals["err"] = signals["ref"] * (1 - 0.5 * signals["anc_gain"])
    np.savez(session["recording"], **signals)
    result = _run(session)
    assert result.returncode == 0, result.stderr
    report = _report(session)
    metrics = _csv(session["out"] / "metrics.csv")
    windows = _csv(session["out"] / "windows.csv")
    summary = (session["out"] / "summary.md").read_text(encoding="utf-8")
    for name, expected_bins in (("target_800_1600", 800), ("target_800_1000", 200),
                                ("target_1000_1600", 600)):
        row = next(row for row in report["metrics"] if row["band"] == name)
        exported = next(row for row in metrics if row["band"] == name)
        assert name in summary
        assert row["trusted"] is False and exported["trusted"] == "false"
        if name in active_bands:
            assert row["observed_err_reduction_db"] == pytest.approx(20 * np.log10(2), abs=1e-8)
            assert float(exported["observed_err_reduction_db"]) == row["observed_err_reduction_db"]
        else:
            assert row["observed_err_reduction_db"] is None
            assert exported["observed_err_reduction_db"] == "null"
        selected = [window for window in windows if window["band"] == name]
        assert selected and all(int(window["fft_bins"]) == expected_bins for window in selected)
    assert report["diagnostic_only"] and not report["performance_claim_allowed"]


def test_cli_rejects_partial_target_frequency_range_without_output(session):
    fs = 3199
    session["signals"]["fs"] = fs
    np.savez(session["recording"], **session["signals"])
    with np.load(session["secondary"], allow_pickle=False) as archive:
        secondary = {key: archive[key] for key in archive.files}
    secondary.update(sample_rate=fs, excitation_band_hz=[80.0, fs / 2])
    np.savez(session["secondary"], **secondary)
    runtime = yaml.safe_load(session["config"].read_text(encoding="utf-8"))
    hardware_path = Path(runtime["hardware_config"])
    hardware = yaml.safe_load(hardware_path.read_text(encoding="utf-8"))
    hardware["audio"]["sample_rate"] = fs
    hardware_path.write_text(yaml.safe_dump(hardware), encoding="utf-8")
    before = session["recording"].read_bytes()
    result = _run(session)
    assert result.returncode == 1, result.stderr
    assert "sample_rate >= 3200" in result.stderr
    assert not session["out"].exists()
    assert session["recording"].read_bytes() == before


def test_silent_complete_cycle_is_null_not_zero_or_success(session):
    for name in ("err", "ref", "control"):
        session["signals"][name] = np.zeros_like(session["signals"][name])
    np.savez(session["recording"], **session["signals"])
    result = _run(session)
    assert result.returncode == 2, result.stderr
    report = _report(session)
    assert report["all_cycles_complete"] and not report["comparison_available"]
    assert all(row["observed_err_reduction_db"] is None for row in report["metrics"])
    rows = _csv(session["out"] / "metrics.csv")
    assert all(row["observed_err_reduction_db"] == "null" for row in rows)
    assert (session["out"] / "windows.csv").is_file()


@pytest.mark.parametrize("no_on", [False, True])
def test_unavailable_cycles_have_explicit_csv_row_without_windows(session, no_on):
    session["signals"]["anc_gain"][:] = 0
    if not no_on:
        session["signals"]["anc_gain"][5 * session["fs"]:] = 1
    np.savez(session["recording"], **session["signals"])
    result = _run(session)
    assert result.returncode == 2, result.stderr
    report = _report(session)
    assert not report["all_cycles_complete"] and not report["comparison_available"]
    assert report["metrics"] == []
    rows = _csv(session["out"] / "metrics.csv")
    assert len(rows) == 1 and rows[0]["status"] == "unavailable"
    assert rows[0]["observed_err_reduction_db"] == "null"
    assert not (session["out"] / "windows.csv").exists()
    assert report["artifacts"]["windows"] is None
    assert report["artifacts"]["windows_omission_reason"] == "no_valid_windows"
    assert "windows.csv: 생성하지 않음" in (session["out"] / "summary.md").read_text(encoding="utf-8")
    if not no_on:
        assert rows[0]["unavailable_reason"] == "missing_off_post"


def test_incomplete_cycle_is_not_hidden_by_another_complete_cycle(session):
    gain = session["signals"]["anc_gain"]
    gain[:] = 0
    gain[3 * session["fs"]:7 * session["fs"]] = 1
    gain[12 * session["fs"]:] = 1
    np.savez(session["recording"], **session["signals"])
    result = _run(session)
    assert result.returncode == 2, result.stderr
    report = _report(session)
    assert report["comparison_available"] and not report["all_cycles_complete"]
    assert len(report["cycles"]) == 2
    incomplete = [row for row in _csv(session["out"] / "metrics.csv") if row["cycle"] == "1"]
    assert len(incomplete) == 1 and incomplete[0]["unavailable_reason"] == "missing_off_post"


@pytest.mark.parametrize("fault", ["length", "nonfinite", "empty_family", "bad_window", "missing_input"])
def test_invalid_input_never_creates_output_directory(session, fault):
    extra = []
    if fault == "length":
        session["signals"]["err"] = session["signals"]["err"][:-1]
    elif fault == "nonfinite":
        session["signals"]["ref"][0] = np.nan
    elif fault == "empty_family":
        extra = ["--source-family", " "]
    elif fault == "bad_window":
        extra = ["--window-seconds", "0"]
    elif fault == "missing_input":
        extra = ["--npz", str(session["recording"].parent / "missing.npz")]
    np.savez(session["recording"], **session["signals"])
    before = session["recording"].read_bytes()
    result = _run(session, *extra)
    assert result.returncode == 1, result.stderr
    assert not session["out"].exists()
    assert session["recording"].read_bytes() == before


def test_existing_output_directory_is_preserved(session):
    session["out"].mkdir()
    sentinel = session["out"] / "metrics.csv"
    sentinel.write_text("existing report", encoding="utf-8")
    result = _run(session)
    assert result.returncode == 1
    assert sentinel.read_text(encoding="utf-8") == "existing report"
    assert list(session["out"].iterdir()) == [sentinel]


@pytest.mark.parametrize("kind", ["output", "dangling", "parent", "parent_dotdot"])
def test_symlink_output_routes_are_rejected_without_writes(session, kind):
    target = session["out"].parent / "target"
    if kind != "dangling":
        target.mkdir()
    if kind in {"output", "dangling"}:
        session["out"].symlink_to(target, target_is_directory=True)
    else:
        parent_link = session["out"].parent / "linked_parent"
        parent_link.symlink_to(target, target_is_directory=True)
        session["out"] = parent_link / "result" if kind == "parent" else parent_link / ".." / "result"
    result = _run(session)
    assert result.returncode == 1
    assert not (target / "result").exists()
    if target.exists():
        assert list(target.iterdir()) == []
    else:
        assert kind == "dangling"


def test_cli_does_not_import_torch_audio_or_inference_runtimes(session):
    bootstrap = '''
import importlib.abc
import runpy
import sys
class RejectHardwareImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'sounddevice', 'onnxruntime', 'tensorrt'} or fullname == 'deep_anc.audio_io':
            raise RuntimeError('금지된 import: ' + fullname)
sys.meta_path.insert(0, RejectHardwareImports())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-c", bootstrap, str(CLI), *_args(session)],
                            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_missing_required_cli_argument_is_input_error():
    result = subprocess.run([sys.executable, str(CLI)], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
