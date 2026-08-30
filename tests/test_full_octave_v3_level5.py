from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from deep_anc.data.full_octave_v3_physical_bundle import (
    INPUT_CHANNELS,
    SIDECAR_SCHEMA,
    UNATTESTED_STRUCTURAL_RAW_STATUS,
    build_full_octave_v3_physical_session_plan,
    load_full_octave_v3_physical_session_bundle,
)
from deep_anc.dsp.control_band_contract import BroadbandFullOctaveContractV3
from deep_anc.eval.full_octave_v3_level5 import (
    BLOCKED_UNATTESTED_INVALID_DECLARATION,
    BLOCKED_UNATTESTED_MISSING_AUTHORITY,
    BLOCKED_UNATTESTED_SELF_DECLARED_CHAIN,
    BLOCKED_UNATTESTED_TERMINAL_RECEIPT,
    DEFAULT_CONFIG_RELATIVE_PATH,
    FULL_OCTAVE_V3_LEVEL5_CAPABILITY_SCHEMA,
    FULL_OCTAVE_V3_LEVEL5_CONSUMED_SCHEMA,
    FULL_OCTAVE_V3_LEVEL5_MODEL_LOCK_SCHEMA,
    FULL_OCTAVE_V3_LEVEL5_PHYSICAL_BUNDLE_LOCK_SCHEMA,
    FULL_OCTAVE_V3_LEVEL5_RAW_MANIFEST_SCHEMA,
    FULL_OCTAVE_V3_LEVEL5_RECEIPT_SCHEMA,
    audit_full_octave_v3_level5_lifecycle,
    canonical_level5_ledger_paths,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("speech", "music", "environment", "machine")


def _bytes(payload: object) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sealed(payload: dict, key: str) -> dict:
    value = dict(payload)
    value[key] = _sha(_bytes(value))
    return value


def _physical_canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _physical_sealed(payload: dict, key: str) -> dict:
    value = dict(payload)
    value[key] = _sha(_physical_canonical_bytes(value))
    return value


def _write(root: Path, relative: str, content: bytes) -> dict[str, object]:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return {"path": relative, "size_bytes": len(content), "sha256": _sha(content)}


def _config() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / DEFAULT_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8")
    )


def _set_artifact(config: dict, role: str, reference: dict[str, object]) -> None:
    config["artifacts"][role] = {
        "path": reference["path"],
        "sha256": reference["sha256"],
    }


def _experiment_contract() -> dict:
    body = {
        "schema_version": 2,
        "config_sha256": "1" * 64,
        "source": {
            "git_commit": "0123456789abcdef0123456789abcdef01234567",
            "source_tree_sha256": "2" * 64,
            "verifiable": True,
            "clean_exact_commit": True,
            "dirty_paths": [],
            "replace_refs": [],
            "index_flags_clean": True,
        },
        "input_generation": {
            "bootstrap_receipt_sha256": None,
            "transfer_manifest_sha256": None,
            "recorded_transfer_aggregate_sha256": None,
        },
        "artifacts": {},
    }
    return {**body, "sha256": _sha(_bytes(body))}


def _raw_manifest(
    root: Path,
    *,
    partition: str,
    model_lock_sha256: str | None,
    prefix: str,
) -> dict[str, object]:
    records: list[dict] = []
    for family in FAMILIES:
        native = _write(
            root,
            f"sources/{prefix}_{family}.native",
            f"native:{prefix}:{family}".encode("utf-8"),
        )
        decoded = _write(
            root,
            f"sources/{prefix}_{family}.f32",
            f"decoded:{prefix}:{family}".encode("utf-8"),
        )
        records.append(
            {
                "record_id": f"{prefix}-{family}-record",
                "source_family": family,
                "source_ids": [f"{prefix}:{family}:source"],
                "lineage_component_id": f"{prefix}:{family}:component",
                "lineage_keys": [f"{prefix}:{family}:lineage"],
                "native_source": native,
                "decoded_pcm": decoded,
            }
        )
    contract = BroadbandFullOctaveContractV3.canonical()
    payload = _sealed(
        {
            "schema": FULL_OCTAVE_V3_LEVEL5_RAW_MANIFEST_SCHEMA,
            "role": "immutable_raw_source_identity_manifest",
            "partition": partition,
            "fixture_only": False,
            "control_band_contract_sha256": contract.digest(),
            "raw_manifest_authority": True,
            "artifact_bytes_verified": True,
            "source_identity_complete": True,
            "model_lock_sha256": model_lock_sha256,
            "records": records,
        },
        "manifest_evidence_sha256",
    )
    return _write(root, f"manifests/{partition}.json", _bytes(payload))


def _physical_bundle_report(
    root: Path,
    *,
    challenge_manifest_sha256: str,
    controller_artifact_sha256: str,
) -> dict[str, object]:
    """의도적으로 fake bytes인, 그러나 official bundle checker에는 구조상 유효한 8ch 묶음.

    이 fixture는 Level-5가 독립 authority 없이 fake non-fixture artifact를 READY로
    승격하지 않는지 증명하기 위한 것이다. 실제 음향 capture나 device open은 전혀 없다.
    """

    base = "results/future_level5_bundle"
    targets = {
        "native_raw": f"{base}/native.s32le",
        "canonical_raw": f"{base}/canonical.s32le",
        "session_sidecar": f"{base}/sidecar.json",
    }
    role_channels = {
        "REF": 0,
        "NOISE_TAP": 1,
        "CANCEL_TAP": 2,
        "ERR_0": 3,
        "ERR_1": 4,
        "ERR_2": 5,
        "ERR_3": 6,
        "ERR_4": 7,
    }
    identity = {
        "source": {
            "source_kind": "submitted_pcm",
            "submitted_pcm_sha256": "6" * 64,
            "source_manifest_sha256": challenge_manifest_sha256,
        },
        "controller": {
            "controller_mode": "deep_anc_open_loop",
            "controller_artifact_sha256": controller_artifact_sha256,
            "controller_config_sha256": "7" * 64,
        },
        "plant": {
            "plant_campaign_contract_sha256": "8" * 64,
            "hardware_fingerprint_sha256": "9" * 64,
            "routing_geometry_sha256": "a" * 64,
        },
        "timing": {
            "training_timing_contract_sha256": "b" * 64,
            "plant_delays_sha256": "c" * 64,
            "handoff_samples": 256,
            "lead_samples": 115,
            "lead_derivation": "PlantDelays.lead()",
        },
    }
    plan = build_full_octave_v3_physical_session_plan(
        artifact_targets=targets,
        role_channels=role_channels,
        topology="single_acquisition_clock_all_eight",
        identity=identity,
        planned_s32_callback_sha256="d" * 64,
    )
    plan_ref = _write(root, f"{base}/capture_plan.json", _bytes(plan))
    frames = 256
    raw_size = frames * INPUT_CHANNELS * 4
    native_content = bytes(range(256)) * (raw_size // 256)
    canonical_content = bytes(reversed(range(256))) * (raw_size // 256)
    native_ref = _write(root, targets["native_raw"], native_content)
    canonical_ref = _write(root, targets["canonical_raw"], canonical_content)
    sidecar = _physical_sealed(
        {
            "schema": SIDECAR_SCHEMA,
            "role": plan["role"],
            "fixture_only": False,
            "capture_plan_file_sha256": plan_ref["sha256"],
            "capture_plan_evidence_sha256": plan["plan_evidence_sha256"],
            "control_band_contract_sha256": plan["control_band_contract_sha256"],
            "native_raw": native_ref,
            "canonical_raw": canonical_ref,
            "capture": {
                "sample_rate_hz": 48_000,
                "block_size": 256,
                "input_channels": 8,
                "raw_dtype": "<i4",
                "raw_layout": "interleaved_s32le",
                "frames": frames,
                "role_channels": role_channels,
                "topology": "single_acquisition_clock_all_eight",
                "same_frame_witness": deepcopy(plan["capture"]["same_frame_witness"]),
                "frame_counter_start": 4096,
                "frame_counter_stop_exclusive": 4096 + frames,
                "planned_s32_callback_sha256": "d" * 64,
                "actual_s32_callback_sha256": "d" * 64,
                "xrun_count": 0,
                "drop_count": 0,
                "add_count": 0,
            },
            "identity": identity,
            "publication": {
                "raw_first": True,
                "publication_sequence": [
                    "capture_plan",
                    "native_raw",
                    "canonical_raw",
                    "session_sidecar",
                ],
                "publication_methods": {
                    "capture_plan": "O_EXCL",
                    "native_raw": "O_EXCL",
                    "canonical_raw": "O_EXCL",
                    "session_sidecar": "O_EXCL",
                },
                "no_replace": {
                    "capture_plan": True,
                    "native_raw": True,
                    "canonical_raw": True,
                    "session_sidecar": True,
                },
            },
            "canonical_training_eligible": False,
            "deployment_eligible": False,
        },
        "sidecar_evidence_sha256",
    )
    sidecar_ref = _write(root, targets["session_sidecar"], _bytes(sidecar))
    bundle_config = yaml.safe_load(
        (REPO_ROOT / "configs/full_octave_v3_physical_session_bundle.yaml").read_text(
            encoding="utf-8"
        )
    )
    bundle_config["artifacts"] = {
        "capture_plan": {"path": plan_ref["path"], "sha256": plan_ref["sha256"]},
        "native_raw": {"path": native_ref["path"], "sha256": native_ref["sha256"]},
        "canonical_raw": {"path": canonical_ref["path"], "sha256": canonical_ref["sha256"]},
        "session_sidecar": {"path": sidecar_ref["path"], "sha256": sidecar_ref["sha256"]},
    }
    bundle_config_path = "configs/future_level5_bundle.yaml"
    _write(
        root,
        bundle_config_path,
        yaml.safe_dump(bundle_config, allow_unicode=True, sort_keys=False).encode("utf-8"),
    )
    report = load_full_octave_v3_physical_session_bundle(
        root / bundle_config_path, repository_root=root
    )
    assert report["status"] == UNATTESTED_STRUCTURAL_RAW_STATUS
    assert report["declared_sha_structure_valid"] is True
    assert report["physical_raw_provenance_attested"] is False
    return _write(root, f"{base}/physical_bundle_report.json", _bytes(report))


def _build_primary_chain(
    root: Path, *, train_validation_overlap: bool = False
) -> tuple[dict, dict[str, dict[str, object]]]:
    """Non-fixture future-shape chain. It remains non-canonical by construction."""

    config = _config()
    base_manifests = {
        partition: _raw_manifest(root, partition=partition, model_lock_sha256=None, prefix=partition)
        for partition in ("training", "validation", "test")
    }
    if train_validation_overlap:
        validation_path = root / str(base_manifests["validation"]["path"])
        validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
        validation_payload["records"][0]["source_ids"] = ["training:speech:source"]
        validation_payload.pop("manifest_evidence_sha256")
        validation_payload = _sealed(validation_payload, "manifest_evidence_sha256")
        base_manifests["validation"] = _write(
            root,
            str(base_manifests["validation"]["path"]),
            _bytes(validation_payload),
        )
    # 현재 canonical practice: model selection은 별도 split이 아니라 validation의 exact alias.
    base_manifests["selection"] = base_manifests["validation"]
    contract_payload = _experiment_contract()
    contract_ref = _write(root, "locks/experiment_contract.json", _bytes(contract_payload))
    checkpoint_ref = _write(root, "runs/future/best.pt", b"future checkpoint bytes")
    controller_ref = _write(root, "runs/future/controller.onnx", b"future controller bytes")
    model_lock = _sealed(
        {
            "schema": FULL_OCTAVE_V3_LEVEL5_MODEL_LOCK_SCHEMA,
            "role": "frozen_canonical_model_after_val_selection",
            "fixture_only": False,
            "control_band_contract_sha256": BroadbandFullOctaveContractV3.canonical().digest(),
            "checkpoint": checkpoint_ref,
            "controller_artifact": controller_ref,
            "experiment_contract": contract_ref,
            "experiment_contract_sha256": contract_payload["sha256"],
            "selection_manifest_sha256": base_manifests["selection"]["sha256"],
            "canonical_model_frozen": True,
            "selection_finalized": True,
        },
        "model_lock_evidence_sha256",
    )
    model_lock_ref = _write(root, "locks/model_lock.json", _bytes(model_lock))
    challenge = _raw_manifest(
        root,
        partition="challenge",
        model_lock_sha256=str(model_lock_ref["sha256"]),
        prefix="challenge",
    )

    # 이 report는 official 8-input structural checker를 통과하지만 모든 raw/checkpoint가
    # test에서 만든 bytes다. 따라서 Level-5 independent authority가 될 수 없다.
    report_ref = _physical_bundle_report(
        root,
        challenge_manifest_sha256=str(challenge["sha256"]),
        controller_artifact_sha256=str(controller_ref["sha256"]),
    )
    physical_lock = _sealed(
        {
            "schema": FULL_OCTAVE_V3_LEVEL5_PHYSICAL_BUNDLE_LOCK_SCHEMA,
            "role": "frozen_eight_input_bundle_for_level5_one_shot",
            "fixture_only": False,
            "control_band_contract_sha256": BroadbandFullOctaveContractV3.canonical().digest(),
            "model_lock_sha256": model_lock_ref["sha256"],
            "physical_bundle_report": report_ref,
            "required_roles": [
                "REF",
                "NOISE_TAP",
                "CANCEL_TAP",
                "ERR_0",
                "ERR_1",
                "ERR_2",
                "ERR_3",
                "ERR_4",
            ],
        },
        "physical_bundle_lock_evidence_sha256",
    )
    physical_lock_ref = _write(root, "locks/physical_bundle_lock.json", _bytes(physical_lock))

    references = {
        "model_lock": model_lock_ref,
        "training_raw_manifest": base_manifests["training"],
        "validation_raw_manifest": base_manifests["validation"],
        "test_raw_manifest": base_manifests["test"],
        "selection_raw_manifest": base_manifests["selection"],
        "challenge_raw_manifest": challenge,
        "physical_bundle_lock": physical_lock_ref,
    }
    for role, reference in references.items():
        _set_artifact(config, role, reference)
    return config, references


def _add_capability(root: Path, config: dict, references: dict[str, dict[str, object]]) -> dict[str, object]:
    first = audit_full_octave_v3_level5_lifecycle(config, repo_root=root)
    assert first["status"] == BLOCKED_UNATTESTED_SELF_DECLARED_CHAIN
    input_sha = str(first["challenge_input_sha256"])
    paths = canonical_level5_ledger_paths(input_sha, repo_root=root)
    source_shas = {
        partition: references[f"{partition}_raw_manifest"]["sha256"]
        for partition in ("training", "validation", "test", "selection", "challenge")
    }
    capability = _sealed(
        {
            "schema": FULL_OCTAVE_V3_LEVEL5_CAPABILITY_SCHEMA,
            "role": "issued_level5_physical_evaluation_capability",
            "phase": "issued",
            "fixture_only": False,
            "control_band_contract_sha256": BroadbandFullOctaveContractV3.canonical().digest(),
            "challenge_input_sha256": input_sha,
            "model_lock_sha256": references["model_lock"]["sha256"],
            "physical_bundle_lock_sha256": references["physical_bundle_lock"]["sha256"],
            "source_manifest_sha256": source_shas,
            "token_sha256": "5" * 64,
            "no_replace_declared": True,
        },
        "capability_evidence_sha256",
    )
    relative = paths["issued"].relative_to(root).as_posix()
    capability_ref = _write(root, relative, _bytes(capability))
    _set_artifact(config, "one_shot_capability", capability_ref)
    return capability_ref


def _add_terminal_receipt(
    root: Path,
    config: dict,
    references: dict[str, dict[str, object]],
    capability_ref: dict[str, object],
) -> None:
    second = audit_full_octave_v3_level5_lifecycle(config, repo_root=root)
    assert second["status"] == BLOCKED_UNATTESTED_SELF_DECLARED_CHAIN
    input_sha = str(second["challenge_input_sha256"])
    paths = canonical_level5_ledger_paths(input_sha, repo_root=root)
    consumed = _sealed(
        {
            "schema": FULL_OCTAVE_V3_LEVEL5_CONSUMED_SCHEMA,
            "role": "consumed_level5_physical_evaluation_capability",
            "phase": "running",
            "fixture_only": False,
            "capability_sha256": capability_ref["sha256"],
            "challenge_input_sha256": input_sha,
            "model_lock_sha256": references["model_lock"]["sha256"],
            "physical_bundle_lock_sha256": references["physical_bundle_lock"]["sha256"],
            "no_replace_declared": True,
        },
        "consumed_evidence_sha256",
    )
    consumed_ref = _write(root, paths["running"].relative_to(root).as_posix(), _bytes(consumed))
    _set_artifact(config, "one_shot_consumed_marker", consumed_ref)
    raw_eval = _write(root, "results/future_level5/off_on_raw.s32le", b"future off on raw")
    metrics = _write(root, "results/future_level5/metrics.npz", b"future metrics")
    evaluator = _write(root, "results/future_level5/evaluator.json", b"future evaluator receipt")
    receipt = _sealed(
        {
            "schema": FULL_OCTAVE_V3_LEVEL5_RECEIPT_SCHEMA,
            "role": "terminal_level5_physical_evaluation_receipt",
            "phase": "completed",
            "fixture_only": False,
            "evaluation_domain": "physical_duct_level5_one_shot",
            "verdict": "PASS",
            "capability_sha256": capability_ref["sha256"],
            "consumed_marker_sha256": consumed_ref["sha256"],
            "challenge_input_sha256": input_sha,
            "model_lock_sha256": references["model_lock"]["sha256"],
            "physical_bundle_lock_sha256": references["physical_bundle_lock"]["sha256"],
            "raw_evaluation_bundle": raw_eval,
            "metrics": metrics,
            "evaluator_receipt": evaluator,
            "no_replace_declared": True,
        },
        "receipt_evidence_sha256",
    )
    receipt_ref = _write(root, paths["completed"].relative_to(root).as_posix(), _bytes(receipt))
    _set_artifact(config, "one_shot_receipt", receipt_ref)


def test_default_static_null_config_is_read_only_blocked() -> None:
    report = audit_full_octave_v3_level5_lifecycle(_config(), repo_root=REPO_ROOT)
    assert report["status"] == BLOCKED_UNATTESTED_MISSING_AUTHORITY
    assert report["challenge_input_sha256"] is None
    assert report["audio_opened"] is False
    assert report["alsa_opened"] is False
    assert report["gpu_initialized"] is False
    assert report["network_opened"] is False
    assert report["files_written"] is False
    assert report["canonical_generalization_pass"] is False
    assert report["physical_generalization_authority"] is False


def test_nonfixture_fake_chain_never_becomes_ready_and_reports_all_authority_blockers(
    tmp_path: Path,
) -> None:
    config, _references = _build_primary_chain(tmp_path)
    report = audit_full_octave_v3_level5_lifecycle(config, repo_root=tmp_path)
    assert report["status"] == BLOCKED_UNATTESTED_SELF_DECLARED_CHAIN
    assert report["challenge_input_sha256"] is not None
    assert report["level5_exclusion"] is not None
    for family in FAMILIES:
        for partition in ("training", "validation", "test"):
            assert report["level5_exclusion"][family][partition]["passed"] is True
            assert all(
                value == 0
                for key, value in report["level5_exclusion"][family][partition].items()
                if key != "passed"
            )
    assert report["base_train_val_test_pairwise_leakage"] is not None
    assert {
        "training__validation",
        "training__test",
        "validation__test",
    } <= set(report["base_train_val_test_pairwise_leakage"])
    assert report["selection_validation_manifest_alias"]["exact_sha_match"] is True
    assert (
        report["selection_validation_manifest_alias"]["selection_sha256"]
        == report["selection_validation_manifest_alias"]["validation_sha256"]
    )
    assert report["challenge_preregistration_limitations"]["independent_reservation_verified"] is False
    assert {
        "official_physical_bundle_revalidation_8ch_raw_sidecar",
        "immutable_lineage_inventory_population_reservation",
        "completed_canonical_checkpoint_contract_selection_export_provenance",
        "submitted_pcm_controller_ps_timing_lead_exact_binding",
        "all_four_family_matched_off_dl_fxlms_campaigns",
        "dirfd_o_excl_one_shot_issuer_terminal_mutual_exclusion",
        "independent_raw_evaluator_receipt",
    } <= set(report["blockers"])
    assert report["self_attested_chain_only"] is True
    assert report["independent_raw_evaluator_receipt_verified"] is False
    assert report["canonical_generalization_pass"] is False
    assert report["physical_generalization_authority"] is False


def test_self_declared_terminal_pass_remains_blocked_and_cannot_be_cli_success(
    tmp_path: Path,
) -> None:
    config, references = _build_primary_chain(tmp_path)
    capability_ref = _add_capability(tmp_path, config, references)
    issued = audit_full_octave_v3_level5_lifecycle(config, repo_root=tmp_path)
    assert issued["status"] == BLOCKED_UNATTESTED_SELF_DECLARED_CHAIN
    assert issued["canonical_generalization_pass"] is False
    _add_terminal_receipt(tmp_path, config, references, capability_ref)
    receipt = audit_full_octave_v3_level5_lifecycle(config, repo_root=tmp_path)
    assert receipt["status"] == BLOCKED_UNATTESTED_TERMINAL_RECEIPT
    assert receipt["self_declared_terminal_verdict"] == "PASS"
    assert receipt["canonical_generalization_pass"] is False
    assert receipt["physical_generalization_authority"] is False
    assert receipt["actual_capture_started"] is False
    assert receipt["actual_evaluation_started"] is False


def test_fixture_manifest_cannot_advance_to_a_lifecycle_status(tmp_path: Path) -> None:
    config, references = _build_primary_chain(tmp_path)
    challenge_path = tmp_path / str(references["challenge_raw_manifest"]["path"])
    payload = json.loads(challenge_path.read_text(encoding="utf-8"))
    payload["fixture_only"] = True
    payload.pop("manifest_evidence_sha256")
    payload = _sealed(payload, "manifest_evidence_sha256")
    changed = _write(tmp_path, str(references["challenge_raw_manifest"]["path"]), _bytes(payload))
    _set_artifact(config, "challenge_raw_manifest", changed)
    report = audit_full_octave_v3_level5_lifecycle(config, repo_root=tmp_path)
    assert report["status"] == BLOCKED_UNATTESTED_INVALID_DECLARATION
    assert "self_attested_structural_chain" in report["blockers"]


def test_any_source_id_lineage_or_bytes_overlap_with_training_is_rejected(tmp_path: Path) -> None:
    config, references = _build_primary_chain(tmp_path)
    challenge_path = tmp_path / str(references["challenge_raw_manifest"]["path"])
    payload = json.loads(challenge_path.read_text(encoding="utf-8"))
    payload["records"][0]["source_ids"] = ["training:speech:source"]
    payload.pop("manifest_evidence_sha256")
    payload = _sealed(payload, "manifest_evidence_sha256")
    changed = _write(tmp_path, str(references["challenge_raw_manifest"]["path"]), _bytes(payload))
    _set_artifact(config, "challenge_raw_manifest", changed)
    report = audit_full_octave_v3_level5_lifecycle(config, repo_root=tmp_path)
    assert report["status"] == BLOCKED_UNATTESTED_INVALID_DECLARATION
    structural = next(
        check for check in report["checks"] if check["id"] == "self_attested_structural_chain"
    )
    assert "source와 lineage/bytes/source ID가 겹칩니다" in structural["detail"]


def test_base_train_validation_leakage_is_explicitly_rejected(tmp_path: Path) -> None:
    config, _references = _build_primary_chain(tmp_path, train_validation_overlap=True)

    report = audit_full_octave_v3_level5_lifecycle(config, repo_root=tmp_path)
    assert report["status"] == BLOCKED_UNATTESTED_INVALID_DECLARATION
    structural = next(
        check for check in report["checks"] if check["id"] == "self_attested_structural_chain"
    )
    assert "base split pairwise leakage" in structural["detail"]


def test_selection_must_be_exact_validation_manifest_sha_alias(tmp_path: Path) -> None:
    config, references = _build_primary_chain(tmp_path)
    _set_artifact(config, "selection_raw_manifest", references["training_raw_manifest"])

    report = audit_full_octave_v3_level5_lifecycle(config, repo_root=tmp_path)
    assert report["status"] == BLOCKED_UNATTESTED_INVALID_DECLARATION
    structural = next(
        check for check in report["checks"] if check["id"] == "self_attested_structural_chain"
    )
    assert "exact same SHA" in structural["detail"]


def test_self_declared_capability_at_any_path_cannot_change_blocked_status(tmp_path: Path) -> None:
    config, references = _build_primary_chain(tmp_path)
    capability_ref = _add_capability(tmp_path, config, references)
    forged = _write(
        tmp_path,
        "results/not_the_level5_ledger/capability.json",
        (tmp_path / str(capability_ref["path"])).read_bytes(),
    )
    _set_artifact(config, "one_shot_capability", forged)
    report = audit_full_octave_v3_level5_lifecycle(config, repo_root=tmp_path)
    assert report["status"] == BLOCKED_UNATTESTED_SELF_DECLARED_CHAIN


def test_nonfixture_fake_terminal_config_returns_nonzero_from_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, references = _build_primary_chain(tmp_path)
    capability = _add_capability(tmp_path, config, references)
    _add_terminal_receipt(tmp_path, config, references, capability)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    script_path = REPO_ROOT / "scripts/eval/check_full_octave_v3_level5_lifecycle.py"
    spec = importlib.util.spec_from_file_location("level5_checker_test", script_path)
    assert spec is not None and spec.loader is not None
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    assert checker.main(["--config", "config.yaml"]) == 1


def test_lifecycle_exposes_no_public_success_predicate_or_zero_exit_path() -> None:
    lifecycle_source = (REPO_ROOT / "src/deep_anc/eval/full_octave_v3_level5.py").read_text(
        encoding="utf-8"
    )
    checker_source = (
        REPO_ROOT / "scripts/eval/check_full_octave_v3_level5_lifecycle.py"
    ).read_text(encoding="utf-8")
    assert "is_canonical_independent_evaluator_physical_pass" not in lifecycle_source
    assert "CANONICAL_INDEPENDENT_EVALUATOR_PHYSICAL_PASS" not in lifecycle_source
    assert "return 0" not in checker_source
    assert "return 1" in checker_source


def test_cli_has_no_audio_gpu_or_writer_imports() -> None:
    for path in (
        REPO_ROOT / "src/deep_anc/eval/full_octave_v3_level5.py",
        REPO_ROOT / "scripts/eval/check_full_octave_v3_level5_lifecycle.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        assert not imported_roots & {
            "sounddevice",
            "pyaudio",
            "torch",
            "onnxruntime",
            "subprocess",
        }
