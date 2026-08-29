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
from deep_anc.train.full_octave_v3_execution import (
    FULL_OCTAVE_V3_EXECUTION_CONFIG_SCHEMA,
    FULL_OCTAVE_V3_EXECUTION_RECEIPT_SCHEMA,
    FULL_OCTAVE_V3_RAW_BOUND_BINDING_SCHEMA,
    FULL_OCTAVE_V3_TRAIN_CONFIG_BINDING_SCHEMA,
    UNATTESTED_EXECUTION_PROVENANCE_BLOCKERS,
    UNATTESTED_EXECUTION_PROVENANCE_STATUS,
    audit_full_octave_v3_execution,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _yaml_bytes(payload: dict) -> bytes:
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=True).encode("utf-8")


def _write(root: Path, relative: str, content: bytes) -> str:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return _sha(content)


def _default_config() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "configs/full_octave_v3_execution.yaml").read_text(encoding="utf-8")
    )


def _plant_payload(*, contract_sha: str, raw_sha: str, analysis_sha: str, timing_sha: str) -> dict:
    def operator(role: str) -> dict:
        return {
            "role": role,
            "sample_rate_hz": 48_000,
            "causal": True,
            "physical_measurement": True,
            "operator_file_sha256": "a" * 64,
            "verified_lower_hz": 80.0,
            "verified_upper_hz": 11_400.0,
        }

    return {
        "schema": "full_octave_causal_plant_authority_v3",
        "control_band_contract_sha256": contract_sha,
        "raw_capture_sha256": raw_sha,
        "analysis_sha256": analysis_sha,
        "training_timing_contract_sha256": timing_sha,
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


def _population_payload(contract_sha: str) -> dict:
    return {
        "schema": "full_octave_population_authority_v3",
        "control_band_contract_sha256": contract_sha,
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


def _complete_nonfixture_execution_fixture(root: Path) -> dict:
    """raw bytes와 receipt가 실제로 cross-bind되는 future-shape fixture.

    이 fixture는 physical authority가 아니며 temp directory에만 존재한다. 특히
    fixture_only=true이면 public execution preflight가 실패하는 회귀를 별도 검사한다.
    """

    contract = BroadbandFullOctaveContractV3.canonical()
    contract_sha = contract.digest()
    timing_sha = "c" * 64
    nonce_sha = "f" * 64
    raw_sha = _write(root, "artifacts/raw_capture.npz", b"future S32 raw capture\x00")
    analysis_sha = _write(root, "artifacts/analysis.json", b"future causal analysis\x00")
    witness_sha = _write(root, "artifacts/electrical_witness.json", b"future witness\x00")
    primary_operator_sha = _write(root, "artifacts/primary_operator.npz", b"P operator\x00")
    secondary_operator_sha = _write(root, "artifacts/secondary_operator.npz", b"S operator\x00")
    plant_sha = _write(
        root,
        "artifacts/plant_authority.json",
        _json_bytes(
            _plant_payload(
                contract_sha=contract_sha,
                raw_sha=raw_sha,
                analysis_sha=analysis_sha,
                timing_sha=timing_sha,
            )
        ),
    )
    population_sha = _write(
        root, "artifacts/population.json", _json_bytes(_population_payload(contract_sha))
    )
    batch = {
        "schema": "full_octave_family_balanced_batch_receipt_v3",
        "control_band_contract_sha256": contract_sha,
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
        "control_band_contract_sha256": contract_sha,
        "batch_receipt_sha256": batch_sha,
        "canonical_training_eligible": True,
        "calibration_pass": True,
        "actual_causal_secondary_output": True,
        "output_y_gradient_share": 0.3,
    }
    calibration_sha = _write(root, "artifacts/calibration.json", _json_bytes(calibration))
    binding = {
        "schema": FULL_OCTAVE_V3_RAW_BOUND_BINDING_SCHEMA,
        "binding_schema": "full_octave_causal_plant_binding_v4",
        "control_band_contract_sha256": contract_sha,
        "raw_capture_sha256": raw_sha,
        "analysis_sha256": analysis_sha,
        "causal_plant_authority_sha256": plant_sha,
        "training_timing_contract_sha256": timing_sha,
        "electrical_witness_receipt_sha256": witness_sha,
        "primary_operator_file_sha256": primary_operator_sha,
        "secondary_operator_file_sha256": secondary_operator_sha,
        "sample_rate_hz": 48_000,
        "block_size": 256,
        "verified_physical_subbands_hz": [
            list(band) for band in contract.physical_identification_subbands_hz
        ],
        "err_channel_index": 0,
        "reference_channel_index": 1,
        "fixture_only": False,
        "canonical_training_eligible": True,
        "publisher": "full_octave_v3_raw_bound_physical_publisher_v1",
    }
    binding_sha = _write(root, "artifacts/causal_binding.json", _json_bytes(binding))
    train_config = {
        "experiment_role": "canonical_pretrain",
        "full_octave_v3_execution": {
            "schema": FULL_OCTAVE_V3_TRAIN_CONFIG_BINDING_SCHEMA,
            "execution_stage": "canonical_pretrain",
            "control_band_contract_sha256": contract_sha,
            "execution_nonce_sha256": nonce_sha,
            "causal_plant_binding_sha256": binding_sha,
            "training_timing_contract_sha256": timing_sha,
            "fixture_only": False,
            "canonical_training_eligible": True,
            "requires_execution_preflight": True,
        },
    }
    train_config_sha = _write(root, "configs/future_train.yaml", _yaml_bytes(train_config))
    config = _default_config()
    config["execution_nonce_sha256"] = nonce_sha
    config["artifacts"] = {
        "fullband_raw_capture": {"path": "artifacts/raw_capture.npz", "sha256": raw_sha},
        "fullband_analysis": {"path": "artifacts/analysis.json", "sha256": analysis_sha},
        "causal_plant_authority": {
            "path": "artifacts/plant_authority.json",
            "sha256": plant_sha,
        },
        "electrical_witness_receipt": {
            "path": "artifacts/electrical_witness.json",
            "sha256": witness_sha,
        },
        "primary_causal_operator": {
            "path": "artifacts/primary_operator.npz",
            "sha256": primary_operator_sha,
        },
        "secondary_causal_operator": {
            "path": "artifacts/secondary_operator.npz",
            "sha256": secondary_operator_sha,
        },
        "population_authority": {"path": "artifacts/population.json", "sha256": population_sha},
        "family_balanced_batch_receipt": {"path": "artifacts/batch.json", "sha256": batch_sha},
        "dnh_gradient_calibration": {
            "path": "artifacts/calibration.json",
            "sha256": calibration_sha,
        },
        "causal_plant_binding": {
            "path": "artifacts/causal_binding.json",
            "sha256": binding_sha,
        },
        "canonical_training_config": {
            "path": "configs/future_train.yaml",
            "sha256": train_config_sha,
        },
        "execution_receipt": {"path": None, "sha256": None},
    }
    partial = audit_full_octave_v3_execution(config, repo_root=root)
    assert partial["execution_input_sha256"] is not None
    receipt = {
        "schema": FULL_OCTAVE_V3_EXECUTION_RECEIPT_SCHEMA,
        "preflight_role": "raw_bound_execution_receipt",
        "control_band_contract_sha256": contract_sha,
        "execution_input_sha256": partial["execution_input_sha256"],
        "execution_stage": "canonical_pretrain",
        "execution_nonce_sha256": nonce_sha,
        "fullband_raw_capture_sha256": raw_sha,
        "fullband_analysis_sha256": analysis_sha,
        "causal_plant_authority_sha256": plant_sha,
        "electrical_witness_receipt_sha256": witness_sha,
        "primary_causal_operator_sha256": primary_operator_sha,
        "secondary_causal_operator_sha256": secondary_operator_sha,
        "causal_plant_binding_sha256": binding_sha,
        "canonical_training_config_sha256": train_config_sha,
        "population_authority_sha256": population_sha,
        "family_balanced_batch_receipt_sha256": batch_sha,
        "dnh_gradient_calibration_sha256": calibration_sha,
        "training_timing_contract_sha256": timing_sha,
        "fixture_only": False,
        "canonical_training_eligible": True,
    }
    receipt_sha = _write(root, "artifacts/execution_receipt.json", _json_bytes(receipt))
    config["artifacts"]["execution_receipt"] = {
        "path": "artifacts/execution_receipt.json",
        "sha256": receipt_sha,
    }
    return config


def test_default_execution_preflight_is_structured_blocked_without_audio_or_training() -> None:
    report = audit_full_octave_v3_execution(_default_config(), repo_root=REPO_ROOT)
    assert report["status"] == "BLOCKED"
    assert report["preflight_ready"] is False
    assert report["trainer_release"] is False
    assert report["audio_opened"] is False
    assert report["trainer_constructed"] is False
    assert report["gpu_initialized"] is False
    assert report["runs_directory_created"] is False
    assert report["control_band_contract_sha256"] == BroadbandFullOctaveContractV3.canonical().digest()
    assert "v3_raw_bound_execution_config" in report["blockers"]


def test_nonfixture_self_attested_receipt_remains_unattested_and_never_ready(tmp_path: Path) -> None:
    report = audit_full_octave_v3_execution(
        _complete_nonfixture_execution_fixture(tmp_path), repo_root=tmp_path
    )
    assert report["status"] == UNATTESTED_EXECUTION_PROVENANCE_STATUS
    assert report["preflight_ready"] is False
    assert report["declared_sha_structure_valid"] is True
    assert report["typed_execution_provenance_attested"] is False
    assert report["self_attested_artifacts_only"] is True
    assert report["canonical_training_eligible"] is False
    assert report["deployment_eligible"] is False
    assert report["trainer_release"] is False
    assert report["audio_opened"] is False
    assert report["trainer_constructed"] is False
    assert report["gpu_initialized"] is False
    assert report["runs_directory_created"] is False
    assert next(
        check
        for check in report["checks"]
        if check["id"] == "declared_sha_structure"
    )["status"] == "PASS"
    assert next(
        check
        for check in report["checks"]
        if check["id"] == "typed_execution_provenance"
    )["status"] == "BLOCKED"
    assert "v3_raw_bound_execution_config" in report["blockers"]
    assert set(UNATTESTED_EXECUTION_PROVENANCE_BLOCKERS) <= set(report["blocking_requirements"])


def test_nonfixture_self_attested_execution_cli_never_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _complete_nonfixture_execution_fixture(tmp_path)
    config_path = tmp_path / "execution.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    script_path = REPO_ROOT / "scripts/train/check_full_octave_v3_execution.py"
    spec = importlib.util.spec_from_file_location("v3_execution_checker_test", script_path)
    assert spec is not None and spec.loader is not None
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    assert checker.main(["--config", "execution.yaml"]) == 1


def test_fixture_only_binding_cannot_enter_execution_preflight(tmp_path: Path) -> None:
    config = _complete_nonfixture_execution_fixture(tmp_path)
    binding_path = tmp_path / "artifacts/causal_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["fixture_only"] = True
    binding_content = _json_bytes(binding)
    binding_path.write_bytes(binding_content)
    config["artifacts"]["causal_plant_binding"]["sha256"] = _sha(binding_content)
    with pytest.raises(ValueError, match="fixture_only=false"):
        audit_full_octave_v3_execution(config, repo_root=tmp_path)


def test_execution_receipt_must_bind_the_current_input_digest(tmp_path: Path) -> None:
    config = _complete_nonfixture_execution_fixture(tmp_path)
    receipt_path = tmp_path / "artifacts/execution_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["execution_input_sha256"] = "0" * 64
    receipt_content = _json_bytes(receipt)
    receipt_path.write_bytes(receipt_content)
    config["artifacts"]["execution_receipt"]["sha256"] = _sha(receipt_content)
    with pytest.raises(ValueError, match="exact execution input"):
        audit_full_octave_v3_execution(config, repo_root=tmp_path)


def test_execution_schema_rejects_legacy_or_half_declared_artifacts() -> None:
    config = _default_config()
    config["schema"] = "legacy"
    with pytest.raises(ValueError, match="schema"):
        audit_full_octave_v3_execution(config, repo_root=REPO_ROOT)

    config = _default_config()
    config["artifacts"]["fullband_analysis"] = {"path": "x", "sha256": None}
    with pytest.raises(ValueError, match="함께 선언"):
        audit_full_octave_v3_execution(config, repo_root=REPO_ROOT)


def test_execution_preflight_source_has_no_audio_gpu_or_trainer_imports() -> None:
    sources = [
        REPO_ROOT / "src/deep_anc/train/full_octave_v3_execution.py",
        REPO_ROOT / "scripts/train/check_full_octave_v3_execution.py",
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
        "v3_execution_train_entry", REPO_ROOT / "scripts/train/train.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_entry_blocks_execution_envelope_before_train_config_or_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_entry = _load_train_entry()
    payload = _default_config()
    assert payload["schema"] == FULL_OCTAVE_V3_EXECUTION_CONFIG_SCHEMA
    called = {"load_train_config": False, "trainer": False}
    monkeypatch.setattr(train_entry, "load_yaml", lambda _: deepcopy(payload))

    def unexpected_train_config(*_args, **_kwargs):
        called["load_train_config"] = True
        raise AssertionError("load_train_config must not be called")

    class UnexpectedTrainer:
        def __init__(self, *_args, **_kwargs):
            called["trainer"] = True
            raise AssertionError("Trainer must not be constructed")

    monkeypatch.setattr(train_entry, "load_train_config", unexpected_train_config)
    monkeypatch.setattr(train_entry, "Trainer", UnexpectedTrainer)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "ignored.yaml"])
    assert train_entry.main() == 2
    assert called == {"load_train_config": False, "trainer": False}


def test_train_entry_still_refuses_generic_trainer_after_preflight_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_entry = _load_train_entry()
    payload = _default_config()
    called = {"load_train_config": False, "trainer": False}
    monkeypatch.setattr(train_entry, "load_yaml", lambda _: deepcopy(payload))
    monkeypatch.setattr(train_entry, "audit_full_octave_v3_execution", lambda *_args, **_kwargs: {"status": "READY"})

    def unexpected_train_config(*_args, **_kwargs):
        called["load_train_config"] = True
        raise AssertionError("load_train_config must not be called")

    class UnexpectedTrainer:
        def __init__(self, *_args, **_kwargs):
            called["trainer"] = True
            raise AssertionError("Trainer must not be constructed")

    monkeypatch.setattr(train_entry, "load_train_config", unexpected_train_config)
    monkeypatch.setattr(train_entry, "Trainer", UnexpectedTrainer)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "ignored.yaml"])
    assert train_entry.main() == 2
    assert called == {"load_train_config": False, "trainer": False}
