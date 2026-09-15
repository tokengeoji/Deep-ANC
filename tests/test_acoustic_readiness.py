import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from deep_anc.eval.acoustic_readiness import build_acoustic_readiness_report, check_acoustic_readiness


@pytest.fixture
def acoustic_config(tmp_path):
    path = tmp_path / "secondary.npz"
    np.savez(
        path, fir=np.array([1.0, 0.1]), delay_samples=1465, sample_rate=48000,
        consistency_band_hz=[150.0, 600.0], excitation_band_hz=[72.0, 1640.0],
        coherence_median=0.9556, calibration_block_size=256,
        calibration_latency="low", output_channel="cancel",
    )
    return {
        "reference": "mic", "digital_reference_lead_samples": 0, "hop": 256,
        "hardware": {"audio": {"sample_rate": 48000, "block_size": 256, "latency": "low"}},
        "duct": {
            "secondary_path": {"npz": str(path), "handoff_extra_samples": 256},
            "positions_m": {"reference_mic": 0.1, "error_mic": 1.1},
            "duct": {"speed_of_sound_mps": 343.0},
            # 잘못된 digital P를 열거나 도착 예산에 쓰면 실패해야 한다.
            "digital_reference": {"primary_path_npz": "/missing/primary.npz", "d_noise_delay_samples": -12345},
        },
    }


def _replace_npz(cfg, **changes):
    path = cfg["duct"]["secondary_path"]["npz"]
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    for key, value in changes.items():
        if value is None:
            values.pop(key, None)
        else:
            values[key] = value
    np.savez(path, **values)


def _config_file(cfg, tmp_path):
    runtime = copy.deepcopy(cfg)
    for name in ("hardware", "duct"):
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(runtime.pop(name)), encoding="utf-8")
        runtime[f"{name}_config"] = str(path)
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(runtime), encoding="utf-8")
    return path


def test_report_distinguishes_geometry_prediction_and_narrow_trusted_band(acoustic_config):
    report = build_acoustic_readiness_report(acoustic_config)
    assert report["passed"]  # 기본 보고는 주기음 실험 자체를 거부하지 않는다.
    assert report["causality"]["control_delay_samples"] == 1721
    assert report["causality"]["reference_preview_samples"] == pytest.approx(48000 / 343)
    assert report["causality"]["required_prediction_samples"] == pytest.approx(1721 - 48000 / 343)
    assert not report["causality"]["broadband_arrival_condition_met"]
    assert not report["band_validation"]["validated_band_extends_above_1khz"]
    assert report["secondary_path"]["consistency_band_hz"] == [150, 600]
    assert report["warnings"]
    json.dumps(report, allow_nan=False)


def test_optional_gates_are_independent(acoustic_config):
    assert build_acoustic_readiness_report(acoustic_config, required_band_hz=(150, 600))["passed"]
    report = build_acoustic_readiness_report(acoustic_config, require_broadband=True, required_band_hz=(80, 1600))
    assert report["failed_requirements"] == ["broadband_arrival_condition", "required_band"]


def test_causal_arrival_can_pass_without_promising_cancellation(acoustic_config):
    acoustic_config["hop"] = 64
    acoustic_config["hardware"]["audio"]["block_size"] = 64
    acoustic_config["duct"]["secondary_path"]["handoff_extra_samples"] = 64
    _replace_npz(acoustic_config, delay_samples=10, calibration_block_size=64, consistency_band_hz=[80, 1600])
    report = build_acoustic_readiness_report(acoustic_config, require_broadband=True, required_band_hz=(1000, 1600))
    assert report["passed"]
    assert report["causality"]["required_prediction_samples"] < 0
    assert "보장하지 않습니다" in report["causality"]["interpretation"]


def test_reversed_microphones_do_not_create_false_preview(acoustic_config):
    acoustic_config["duct"]["positions_m"] = {"reference_mic": 1.1, "error_mic": 0.1}
    report = build_acoustic_readiness_report(acoustic_config)
    assert report["causality"]["reference_preview_samples"] < 0
    assert report["causality"]["required_prediction_samples"] > 1721


@pytest.mark.parametrize("metadata", ["consistency_band_hz", "coherence_median"])
def test_missing_trust_metadata_never_falls_back_to_excitation(acoustic_config, metadata):
    _replace_npz(acoustic_config, **{metadata: None})
    report = build_acoustic_readiness_report(acoustic_config, required_band_hz=(1000, 1600))
    assert not report["passed"]
    assert not report["band_validation"]["validated_band_extends_above_1khz"]


@pytest.mark.parametrize("changes", [
    {"sample_rate": 44100}, {"sample_rate": None}, {"delay_samples": None},
    {"delay_samples": -1}, {"delay_samples": 1.5}, {"fir": [0.0]},
    {"fir": [float("nan")]}, {"calibration_block_size": 512},
    {"calibration_latency": "high"}, {"output_channel": "noise"},
    {"consistency_band_hz": [600, 150]},
])
def test_invalid_secondary_metadata_fails(acoustic_config, changes):
    _replace_npz(acoustic_config, **changes)
    with pytest.raises(ValueError):
        build_acoustic_readiness_report(acoustic_config)


@pytest.mark.parametrize("key,value", [("reference", "digital"), ("digital_reference_lead_samples", 113), ("hop", 128)])
def test_incompatible_runtime_fails(acoustic_config, key, value):
    acoustic_config[key] = value
    with pytest.raises(ValueError):
        build_acoustic_readiness_report(acoustic_config)


def test_low_consistency_is_not_a_validated_high_band(acoustic_config):
    _replace_npz(acoustic_config, consistency_band_hz=[80, 1600], coherence_median=0.4)
    assert not build_acoustic_readiness_report(acoustic_config, required_band_hz=(1000, 1600))["passed"]


def test_file_api_and_overrides(acoustic_config, tmp_path):
    path = _config_file(acoustic_config, tmp_path)
    assert check_acoustic_readiness(path)["passed"]
    with pytest.raises(ValueError, match="reference=mic"):
        check_acoustic_readiness(path, ["reference=digital"])


@pytest.mark.parametrize("options,code", [([], 0), (["--require-broadband"], 1), (["--require-band", "1000", "1600"], 1)])
def test_cli_json_and_no_audio_or_torch_import(acoustic_config, tmp_path, options, code):
    path = _config_file(acoustic_config, tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/bench/check_acoustic_readiness.py"
    # 새 프로세스에서 라이브러리가 이미 import되어 검증을 빠져나가는 것을 막는다.
    bootstrap = """
import importlib.abc, runpy, sys
class BlockDeviceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'sounddevice', 'onnxruntime', 'tensorrt'}:
            raise AssertionError('장치/추론 import 금지: ' + fullname)
sys.meta_path.insert(0, BlockDeviceImports())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", bootstrap, str(script), "--config", str(path), "--json", *options],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == code, result.stderr
    report = json.loads(result.stdout)
    assert report["config_valid"]
    assert report["passed"] == (code == 0)


def test_cli_missing_file_is_json_failure(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/bench/check_acoustic_readiness.py"
    result = subprocess.run(
        [sys.executable, str(script), "--config", str(tmp_path / "missing.yaml"), "--json"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["config_valid"] is False
