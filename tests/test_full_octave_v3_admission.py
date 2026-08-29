from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from deep_anc.dsp.control_band_contract import BroadbandFullOctaveContractV3
from deep_anc.train.full_octave_v3_admission import (
    ADMISSION_CONFIG_SCHEMA,
    audit_full_octave_v3_admission,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _default_config() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "configs/full_octave_v3_admission.yaml").read_text(encoding="utf-8")
    )


def _write(root: Path, relative: str, content: bytes) -> str:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return _sha(content)


def _complete_future_fixture(root: Path) -> dict:
    """future publisher schema의 cross-byte binding을 검사할 작은 fixture."""

    contract = BroadbandFullOctaveContractV3.canonical()
    raw_bytes = b"S32 fullband raw-first fixture\x00"
    raw_sha = _write(root, "artifacts/raw.npz", raw_bytes)
    operator = lambda role: {
        "role": role,
        "sample_rate_hz": 48_000,
        "causal": True,
        "physical_measurement": True,
        "operator_file_sha256": "a" * 64,
        "verified_lower_hz": 80.0,
        "verified_upper_hz": 11_400.0,
    }
    plant = {
        "schema": "full_octave_causal_plant_authority_v3",
        "control_band_contract_sha256": contract.digest(),
        "raw_capture_sha256": raw_sha,
        "analysis_sha256": "b" * 64,
        "training_timing_contract_sha256": "c" * 64,
        "sample_rate_hz": 48_000,
        "block_size": 256,
        "plant_identification_pass": True,
        "electrical_output_witness_pass": True,
        "shared_clock_authority_pass": True,
        "hardware_sample_identity_pass": True,
        "canonical_training_eligible": True,
        "timing_contract": {
            "schema": "training_timing_contract_v1",
            "plant_delays_lead_derived": True,
            "manual_lead_allowed": False,
            "lead_samples": 116,
        },
        "primary_operator": operator("primary"),
        "secondary_operator": operator("secondary"),
    }
    plant_sha = _write(root, "artifacts/plant.json", _json_bytes(plant))
    population = {
        "schema": "full_octave_population_authority_v3",
        "control_band_contract_sha256": contract.digest(),
        "population_audit_sha256": "d" * 64,
        "manifest_bundle_sha256": "e" * 64,
        "canonical_training_eligible": True,
        "external_manifest_authority_bound": True,
        "connected_component_authority_bound": True,
        "interval_alias_authority_bound": True,
        "actual_raw_manifest_authority_bound": True,
        "recorded_synthetic_lineage_intersections_zero": True,
        "required_source_families": ["speech", "music", "environment", "machine"],
        "required_splits": ["train", "val", "test"],
        "minimum_independent_components_per_split_family_band": 4,
    }
    population_sha = _write(root, "artifacts/population.json", _json_bytes(population))
    batch = {
        "schema": "full_octave_family_balanced_batch_receipt_v3",
        "control_band_contract_sha256": contract.digest(),
        "population_authority_sha256": population_sha,
        "canonical_training_eligible": True,
        "actual_family_balanced_batch_receipt_consumed": True,
        "global_sample_index_deterministic": True,
        "component_uniform_long_run_sampler_proven": True,
        "batch_size": 8,
        "family_counts": {"speech": 2, "music": 2, "environment": 2, "machine": 2},
    }
    batch_sha = _write(root, "artifacts/batch.json", _json_bytes(batch))
    calibration = {
        "schema": "full_octave_dnh_gradient_calibration_v3",
        "control_band_contract_sha256": contract.digest(),
        "batch_receipt_sha256": batch_sha,
        "canonical_training_eligible": True,
        "calibration_pass": True,
        "actual_causal_secondary_output": True,
        "output_y_gradient_share": 0.3,
    }
    calibration_sha = _write(root, "artifacts/calibration.json", _json_bytes(calibration))
    config = _default_config()
    config["artifacts"] = {
        "fullband_raw_capture": {"path": "artifacts/raw.npz", "sha256": raw_sha},
        "causal_plant_authority": {"path": "artifacts/plant.json", "sha256": plant_sha},
        "population_authority": {"path": "artifacts/population.json", "sha256": population_sha},
        "family_balanced_batch_receipt": {"path": "artifacts/batch.json", "sha256": batch_sha},
        "dnh_gradient_calibration": {"path": "artifacts/calibration.json", "sha256": calibration_sha},
    }
    return config


def test_default_admission_is_structured_blocked_without_audio_or_training() -> None:
    report = audit_full_octave_v3_admission(_default_config(), repo_root=REPO_ROOT)
    assert report["status"] == "BLOCKED"
    assert report["eligible"] is False
    assert report["admission_only"] is True
    assert report["audio_opened"] is False
    assert report["trainer_constructed"] is False
    assert report["gpu_initialized"] is False
    assert report["runs_directory_created"] is False
    assert report["control_band_contract_sha256"] == BroadbandFullOctaveContractV3.canonical().digest()
    assert {check["id"] for check in report["checks"] if check["status"] == "BLOCKED"} >= {
        "fullband_raw_capture",
        "causal_plant_authority",
        "population_authority",
        "family_balanced_batch_receipt",
        "dnh_gradient_calibration",
        "v3_trainer_evaluator_consumer_wiring",
    }


def test_future_artifacts_are_hash_and_cross_reference_bound_but_code_gate_stays_blocked(
    tmp_path: Path,
) -> None:
    report = audit_full_octave_v3_admission(
        _complete_future_fixture(tmp_path), repo_root=tmp_path
    )
    assert report["status"] == "BLOCKED"
    assert report["eligible"] is False
    blocked = {check["id"] for check in report["checks"] if check["status"] == "BLOCKED"}
    assert blocked == {"v3_trainer_evaluator_consumer_wiring"}


def test_admission_rejects_v2_or_forged_config_contract() -> None:
    config = _default_config()
    config["control_band_contract"]["id"] = "broadband_point_control_150_11314_v2"
    with pytest.raises(ValueError, match="canonical v3"):
        audit_full_octave_v3_admission(config, repo_root=REPO_ROOT)

    config = _default_config()
    config["schema"] = "legacy"
    with pytest.raises(ValueError, match="schema"):
        audit_full_octave_v3_admission(config, repo_root=REPO_ROOT)


def test_admission_rejects_sha_mismatch_and_symlink_artifact(tmp_path: Path) -> None:
    config = _complete_future_fixture(tmp_path)
    config["artifacts"]["fullband_raw_capture"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bytes SHA"):
        audit_full_octave_v3_admission(config, repo_root=tmp_path)

    config = _default_config()
    target = tmp_path / "outside.raw"
    target.write_bytes(b"outside")
    (tmp_path / "inside.raw").symlink_to(target)
    config["artifacts"]["fullband_raw_capture"] = {
        "path": "inside.raw",
        "sha256": _sha(b"outside"),
    }
    with pytest.raises(ValueError, match="symlink"):
        audit_full_octave_v3_admission(config, repo_root=tmp_path)


def test_malformed_future_metadata_cannot_open_admission(tmp_path: Path) -> None:
    config = _complete_future_fixture(tmp_path)
    path = tmp_path / "artifacts/calibration.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["output_y_gradient_share"] = 0.41
    content = _json_bytes(payload)
    path.write_bytes(content)
    config["artifacts"]["dnh_gradient_calibration"]["sha256"] = _sha(content)
    with pytest.raises(ValueError, match="0.2, 0.4"):
        audit_full_octave_v3_admission(config, repo_root=tmp_path)


def test_admission_source_has_no_audio_gpu_or_trainer_imports() -> None:
    sources = [
        REPO_ROOT / "src/deep_anc/train/full_octave_v3_admission.py",
        REPO_ROOT / "scripts/train/check_full_octave_v3_admission.py",
    ]
    tree = ast.parse("\n".join(path.read_text(encoding="utf-8") for path in sources))
    imported: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    assert not {"sounddevice", "torch", "subprocess"} & imported
    assert "Trainer" not in names


def _load_train_entry():
    spec = importlib.util.spec_from_file_location(
        "v3_admission_train_entry", REPO_ROOT / "scripts/train/train.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_entry_rejects_admission_only_before_train_config_or_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_entry = _load_train_entry()
    payload = _default_config()
    assert payload["schema"] == ADMISSION_CONFIG_SCHEMA
    called = {"load_train_config": False}

    monkeypatch.setattr(train_entry, "load_yaml", lambda _: deepcopy(payload))

    def unexpected_train_config(*_args, **_kwargs):
        called["load_train_config"] = True
        raise AssertionError("load_train_config must not be called")

    monkeypatch.setattr(train_entry, "load_train_config", unexpected_train_config)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "ignored.yaml"])
    assert train_entry.main() == 2
    assert called["load_train_config"] is False
