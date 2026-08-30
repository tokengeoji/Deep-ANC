from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from deep_anc.data.full_octave_v3_matched_campaign import (
    CONDITIONS,
    CONDITION_CONTROLLER_MODE,
    DEFAULT_CONFIG_RELATIVE_PATH,
    RECEIPT_SCHEMA,
    SESSION_RUN_RECEIPT_SCHEMA,
    UNATTESTED_PHYSICAL_PROVENANCE_BLOCKERS,
    UNATTESTED_PHYSICAL_PROVENANCE_STATUS,
    audit_full_octave_v3_matched_campaign,
    build_full_octave_v3_matched_campaign_plan,
)
from deep_anc.data.full_octave_v3_physical_bundle import (
    INPUT_CHANNELS,
    SIDECAR_SCHEMA,
    build_full_octave_v3_physical_session_plan,
)
from deep_anc.dsp.control_band_contract import BroadbandFullOctaveContractV3


REPO_ROOT = Path(__file__).resolve().parents[1]


def _bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sealed(payload: dict, key: str) -> dict:
    value = dict(payload)
    value[key] = _sha(_bytes(value))
    return value


def _write(root: Path, relative: str, content: bytes) -> dict[str, str]:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return {"path": relative, "sha256": _sha(content)}


def _artifact(root: Path, name: str, content: bytes | None = None) -> dict[str, str]:
    return _write(root, f"evidence/{name}", content or name.encode("utf-8"))


def _default_config() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / DEFAULT_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8")
    )


def _identity(root: Path) -> dict:
    return {
        "level_gain": {
            "measurement_level_evidence": _artifact(root, "level_evidence.json"),
            "gain_contract": _artifact(root, "gain_contract.json"),
            "meter_target_dbfs": -50.1,
            "noise_playback_gain_db": -24.0,
            "cancel_playback_gain_db": -24.0,
        },
        "plant_timing": {
            "plant_campaign_contract": _artifact(root, "plant_campaign.json"),
            "primary_path_operator": _artifact(root, "primary_operator.npz"),
            "secondary_path_operator": _artifact(root, "secondary_operator.npz"),
            "training_timing_contract": _artifact(root, "timing_contract.json"),
            "plant_delays": _artifact(root, "plant_delays.json"),
            "handoff_samples": 256,
            "lead_samples": 115,
            "lead_derivation": "PlantDelays.lead()",
        },
        "geometry": {"routing_geometry": _artifact(root, "routing_geometry.json")},
        "window": {
            "window_contract": _artifact(root, "window_contract.json"),
            "warmup_samples": 256,
            "analysis_start_sample": 256,
            "analysis_stop_sample_exclusive": 512,
        },
        "limiter": {
            "limiter_contract": _artifact(root, "limiter_contract.json"),
            "limiter_limit": 0.2,
            "limiter_enabled": True,
        },
        "hardware": {
            "hardware_fingerprint": _artifact(root, "hardware_fingerprint.json"),
            "acquisition_topology": _artifact(root, "topology_evidence.json"),
            "expected_bundle_topology": "single_acquisition_clock_all_eight",
        },
    }


def _one_shot(root: Path) -> dict:
    return {
        "test_once_required": True,
        "allow_session_append": False,
        "allow_session_replacement": False,
        "model_selection_locked_before_test": True,
        "model_selection_receipt": _artifact(root, "model_selection_receipt.json"),
        "campaign_nonce_sha256": "f" * 64,
    }


def _controllers(root: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for condition in CONDITIONS:
        result[condition] = {
            "controller_mode": CONDITION_CONTROLLER_MODE[condition],
            "controller_artifact": _artifact(root, f"controllers/{condition}.artifact"),
            "controller_config": _artifact(root, f"controllers/{condition}.config"),
        }
    return result


def _source(root: Path, family: str, index: int) -> dict:
    return {
        "source_family": family,
        "submitted_pcm": _artifact(
            root,
            f"sources/{family}_{index}.s32le",
            f"submitted:{family}:{index}".encode("utf-8"),
        ),
        "source_manifest": _artifact(root, "sources/canonical_manifest.json"),
    }


def _role_channels() -> dict[str, int]:
    return {
        "REF": 0,
        "NOISE_TAP": 1,
        "CANCEL_TAP": 2,
        "ERR_0": 3,
        "ERR_1": 4,
        "ERR_2": 5,
        "ERR_3": 6,
        "ERR_4": 7,
    }


def _publication() -> dict:
    return {
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
    }


def _write_bundle(
    root: Path,
    *,
    identity: dict,
    source: dict,
    controller: dict,
    config_target: str,
    label: str,
    raw_content_seed: str | None = None,
    canonical_matches_native: bool = False,
    canonical_from_native_content_seed: str | None = None,
) -> dict:
    """테스트용 non-fixture 8-input artifact를 raw-first 순서로 만든다."""

    base = f"results/matched_fixture_raw/{label}"
    targets = {
        "native_raw": f"{base}/native.s32le",
        "canonical_raw": f"{base}/canonical.s32le",
        "session_sidecar": f"{base}/sidecar.json",
    }
    raw_identity = {
        "source": {
            "source_kind": "submitted_pcm",
            "submitted_pcm_sha256": source["submitted_pcm"]["sha256"],
            "source_manifest_sha256": source["source_manifest"]["sha256"],
        },
        "controller": {
            "controller_mode": controller["controller_mode"],
            "controller_artifact_sha256": controller["controller_artifact"]["sha256"],
            "controller_config_sha256": controller["controller_config"]["sha256"],
        },
        "plant": {
            "plant_campaign_contract_sha256": identity["plant_timing"]["plant_campaign_contract"]["sha256"],
            "hardware_fingerprint_sha256": identity["hardware"]["hardware_fingerprint"]["sha256"],
            "routing_geometry_sha256": identity["geometry"]["routing_geometry"]["sha256"],
        },
        "timing": {
            "training_timing_contract_sha256": identity["plant_timing"]["training_timing_contract"]["sha256"],
            "plant_delays_sha256": identity["plant_timing"]["plant_delays"]["sha256"],
            "handoff_samples": 256,
            "lead_samples": 115,
            "lead_derivation": "PlantDelays.lead()",
        },
    }
    plan = build_full_octave_v3_physical_session_plan(
        artifact_targets=targets,
        role_channels=_role_channels(),
        topology="single_acquisition_clock_all_eight",
        identity=raw_identity,
        planned_s32_callback_sha256="a" * 64,
    )
    plan_path = f"{base}/capture_plan.json"
    plan_ref = _write(root, plan_path, _bytes(plan))

    frames = 512
    raw_size = frames * INPUT_CHANNELS * 4
    # session path가 달라도 same bytes를 재사용하면 campaign-wide SHA guard가 잡아야
    # 한다. 기본값은 label digest를 써서 complete fixture의 모든 raw가 unique하도록 한다.
    content_seed = raw_content_seed or label
    native_digest = hashlib.sha256(f"native:{content_seed}".encode("utf-8")).digest()
    canonical_digest = hashlib.sha256(
        f"canonical:{content_seed}".encode("utf-8")
    ).digest()
    native = (native_digest * ((raw_size + len(native_digest) - 1) // len(native_digest)))[:raw_size]
    if canonical_matches_native:
        canonical = native
    else:
        if canonical_from_native_content_seed is not None:
            canonical_digest = hashlib.sha256(
                f"native:{canonical_from_native_content_seed}".encode("utf-8")
            ).digest()
        canonical = (
            canonical_digest * ((raw_size + len(canonical_digest) - 1) // len(canonical_digest))
        )[:raw_size]
    native_ref = _write(root, targets["native_raw"], native)
    canonical_ref = _write(root, targets["canonical_raw"], canonical)
    sidecar = _sealed(
        {
            "schema": SIDECAR_SCHEMA,
            "role": plan["role"],
            "fixture_only": False,
            "capture_plan_file_sha256": plan_ref["sha256"],
            "capture_plan_evidence_sha256": plan["plan_evidence_sha256"],
            "control_band_contract_sha256": plan["control_band_contract_sha256"],
            "native_raw": {
                "path": native_ref["path"],
                "size_bytes": len(native),
                "sha256": native_ref["sha256"],
            },
            "canonical_raw": {
                "path": canonical_ref["path"],
                "size_bytes": len(canonical),
                "sha256": canonical_ref["sha256"],
            },
            "capture": {
                "sample_rate_hz": 48_000,
                "block_size": 256,
                "input_channels": 8,
                "raw_dtype": "<i4",
                "raw_layout": "interleaved_s32le",
                "frames": frames,
                "role_channels": _role_channels(),
                "topology": "single_acquisition_clock_all_eight",
                "same_frame_witness": deepcopy(plan["capture"]["same_frame_witness"]),
                "frame_counter_start": 4096,
                "frame_counter_stop_exclusive": 4096 + frames,
                "planned_s32_callback_sha256": "a" * 64,
                "actual_s32_callback_sha256": "a" * 64,
                "xrun_count": 0,
                "drop_count": 0,
                "add_count": 0,
            },
            "identity": raw_identity,
            "publication": _publication(),
            "canonical_training_eligible": False,
            "deployment_eligible": False,
        },
        "sidecar_evidence_sha256",
    )
    sidecar_ref = _write(root, targets["session_sidecar"], _bytes(sidecar))
    bundle_config = {
        "schema": "full_octave_v3_physical_session_bundle_config_v1",
        "role": "raw_first_eight_input_synchronized_physical_campaign_no_audio",
        "control_band_contract": {
            "id": BroadbandFullOctaveContractV3.canonical().contract_id,
            "sha256": BroadbandFullOctaveContractV3.canonical().digest(),
            "sample_rate_hz": 48_000,
            "block_size": 256,
        },
        "artifacts": {
            "capture_plan": plan_ref,
            "native_raw": native_ref,
            "canonical_raw": canonical_ref,
            "session_sidecar": sidecar_ref,
        },
    }
    return {
        "bundle_config": _write(
            root, config_target, yaml.safe_dump(bundle_config, sort_keys=False).encode("utf-8")
        ),
        "capture_plan_evidence_sha256": plan["plan_evidence_sha256"],
        "session_sidecar_evidence_sha256": sidecar["sidecar_evidence_sha256"],
        "native_raw_sha256": native_ref["sha256"],
        "canonical_raw_sha256": canonical_ref["sha256"],
    }


def _write_run_receipt(
    root: Path,
    *,
    identity: dict,
    unit: dict,
    session: dict,
    bundle: dict,
) -> dict[str, str]:
    """실제 capture adapter 없이 receipt schema만 만드는 test fixture helper."""

    source = unit["source"]
    level_gain = identity["level_gain"]
    plant_timing = identity["plant_timing"]
    window = identity["window"]
    limiter = identity["limiter"]
    hardware = identity["hardware"]
    payload = _sealed(
        {
            "schema": SESSION_RUN_RECEIPT_SCHEMA,
            "role": "physical_matched_off_dl_fxlms_metadata_only_no_audio",
            "fixture_only": False,
            "comparison_unit_id": unit["comparison_unit_id"],
            "session_id": session["session_id"],
            "condition": session["condition"],
            "source": {
                "submitted_pcm_sha256": source["submitted_pcm"]["sha256"],
                "source_manifest_sha256": source["source_manifest"]["sha256"],
            },
            "level_gain": {
                "measurement_level_evidence_sha256": level_gain["measurement_level_evidence"]["sha256"],
                "gain_contract_sha256": level_gain["gain_contract"]["sha256"],
                "meter_target_dbfs": level_gain["meter_target_dbfs"],
                "noise_playback_gain_db": level_gain["noise_playback_gain_db"],
                "cancel_playback_gain_db": level_gain["cancel_playback_gain_db"],
            },
            "plant_timing": {
                "plant_campaign_contract_sha256": plant_timing["plant_campaign_contract"]["sha256"],
                "primary_path_operator_sha256": plant_timing["primary_path_operator"]["sha256"],
                "secondary_path_operator_sha256": plant_timing["secondary_path_operator"]["sha256"],
                "training_timing_contract_sha256": plant_timing["training_timing_contract"]["sha256"],
                "plant_delays_sha256": plant_timing["plant_delays"]["sha256"],
                "handoff_samples": plant_timing["handoff_samples"],
                "lead_samples": plant_timing["lead_samples"],
                "lead_derivation": plant_timing["lead_derivation"],
            },
            "geometry": {"routing_geometry_sha256": identity["geometry"]["routing_geometry"]["sha256"]},
            "window": {
                "window_contract_sha256": window["window_contract"]["sha256"],
                "warmup_samples": window["warmup_samples"],
                "analysis_start_sample": window["analysis_start_sample"],
                "analysis_stop_sample_exclusive": window["analysis_stop_sample_exclusive"],
            },
            "limiter": {
                "limiter_contract_sha256": limiter["limiter_contract"]["sha256"],
                "limiter_limit": limiter["limiter_limit"],
                "limiter_enabled": limiter["limiter_enabled"],
            },
            "hardware": {
                "hardware_fingerprint_sha256": hardware["hardware_fingerprint"]["sha256"],
                "acquisition_topology_sha256": hardware["acquisition_topology"]["sha256"],
                "expected_bundle_topology": hardware["expected_bundle_topology"],
            },
            "bundle": {
                "bundle_config_path": bundle["bundle_config"]["path"],
                "bundle_config_sha256": bundle["bundle_config"]["sha256"],
                "capture_plan_evidence_sha256": bundle["capture_plan_evidence_sha256"],
                "session_sidecar_evidence_sha256": bundle["session_sidecar_evidence_sha256"],
                "native_raw_sha256": bundle["native_raw_sha256"],
                "canonical_raw_sha256": bundle["canonical_raw_sha256"],
            },
        },
        "run_evidence_sha256",
    )
    return _write(root, session["comparison_run_receipt_target"], _bytes(payload))


def _complete_campaign(
    root: Path,
    *,
    cross_session_cross_kind_duplicate: bool = False,
    same_session_identity_raw: bool = False,
) -> tuple[dict, str, str]:
    identity = _identity(root)
    controllers = _controllers(root)
    units: list[dict] = []
    orders = (
        ["OFF", "DL", "FxLMS"],
        ["DL", "FxLMS", "OFF"],
        ["FxLMS", "OFF", "DL"],
    )
    for family in BroadbandFullOctaveContractV3.canonical().source_families:
        for index in range(4):
            unit_id = f"{family}-group-{index}"
            order = orders[index % len(orders)]
            sessions = []
            for position, condition in enumerate(order):
                session_id = f"{unit_id}-{condition.lower()}"
                sessions.append(
                    {
                        "session_id": session_id,
                        "condition": condition,
                        "order_index": position,
                        "controller": deepcopy(controllers[condition]),
                        "bundle_config_target": f"results/matched_campaign/bundles/{session_id}.yaml",
                        "comparison_run_receipt_target": (
                            f"results/matched_campaign/run_receipts/{session_id}.json"
                        ),
                    }
                )
            units.append(
                {
                    "comparison_unit_id": unit_id,
                    "independent_group_id": f"independent:{unit_id}",
                    "source": _source(root, family, index),
                    "counterbalance_order": order,
                    "sessions": sessions,
                }
            )
    plan = build_full_octave_v3_matched_campaign_plan(
        campaign_identity=identity, comparison_units=units, one_shot=_one_shot(root)
    )
    plan_path = "results/matched_campaign/campaign_plan.json"
    plan_ref = _write(root, plan_path, _bytes(plan))

    bundle_refs: dict[str, dict] = {}
    bundle_index = 0
    for unit in units:
        for session in unit["sessions"]:
            bundle = _write_bundle(
                root,
                identity=identity,
                source=unit["source"],
                controller=session["controller"],
                config_target=session["bundle_config_target"],
                label=session["session_id"],
                raw_content_seed=(
                    "intentional-cross-session-cross-kind-duplicate"
                    if cross_session_cross_kind_duplicate and bundle_index == 0
                    else None
                ),
                canonical_matches_native=same_session_identity_raw and bundle_index == 0,
                canonical_from_native_content_seed=(
                    "intentional-cross-session-cross-kind-duplicate"
                    if cross_session_cross_kind_duplicate and bundle_index == 1
                    else None
                ),
            )
            bundle_index += 1
            bundle_refs[session["session_id"]] = {
                "bundle_config": bundle["bundle_config"],
                "comparison_run_receipt": _write_run_receipt(
                    root, identity=identity, unit=unit, session=session, bundle=bundle
                ),
            }
    entries = []
    for unit in units:
        for session in unit["sessions"]:
            entries.append(
                {
                    "comparison_unit_id": unit["comparison_unit_id"],
                    "session_id": session["session_id"],
                    "condition": session["condition"],
                    "order_index": session["order_index"],
                    **bundle_refs[session["session_id"]],
                }
            )
    receipt = _sealed(
        {
            "schema": RECEIPT_SCHEMA,
            "role": "physical_matched_off_dl_fxlms_metadata_only_no_audio",
            "fixture_only": False,
            "campaign_plan_file_sha256": plan_ref["sha256"],
            "campaign_plan_evidence_sha256": plan["plan_evidence_sha256"],
            "control_band_contract_sha256": plan["control_band_contract_sha256"],
            "one_shot": {
                "test_once_completed": True,
                "no_session_append": True,
                "no_session_replacement": True,
                "model_selection_locked_before_test": True,
                "observed_capture_order": plan["campaign_capture_order"],
            },
            "session_receipts": entries,
        },
        "receipt_evidence_sha256",
    )
    receipt_path = "results/matched_campaign/campaign_receipt.json"
    receipt_ref = _write(root, receipt_path, _bytes(receipt))
    config = _default_config()
    config["artifacts"] = {"campaign_plan": plan_ref, "campaign_receipt": receipt_ref}
    return config, plan_path, receipt_path


def _rewrite_receipt(root: Path, config: dict, receipt_path: str, mutate) -> None:  # type: ignore[no-untyped-def]
    receipt = json.loads((root / receipt_path).read_text(encoding="utf-8"))
    mutate(receipt)
    receipt.pop("receipt_evidence_sha256")
    receipt = _sealed(receipt, "receipt_evidence_sha256")
    ref = _write(root, receipt_path, _bytes(receipt))
    config["artifacts"]["campaign_receipt"] = ref


def test_default_static_null_config_is_read_only_blocked() -> None:
    report = audit_full_octave_v3_matched_campaign(_default_config(), repository_root=REPO_ROOT)
    assert report["status"] == "BLOCKED"
    assert report["matched_campaign_structural_valid"] is False
    assert report["audio_opened"] is False
    assert report["alsa_opened"] is False
    assert report["gpu_initialized"] is False
    assert report["network_opened"] is False
    assert report["results_written"] is False
    assert report["canonical_matched_physical_pass"] is False
    assert report["physical_attenuation_math_performed"] is False


def test_nonfixture_counterbalanced_eight_input_campaign_is_unattested_and_blocked(
    tmp_path: Path,
) -> None:
    config, _plan_path, _receipt_path = _complete_campaign(tmp_path)
    report = audit_full_octave_v3_matched_campaign(config, repository_root=tmp_path)
    assert report["status"] == UNATTESTED_PHYSICAL_PROVENANCE_STATUS
    assert report["declared_sha_structure_valid"] is True
    assert report["matched_campaign_structural_valid"] is False
    assert report["physical_provenance_attested"] is False
    assert report["self_attested_artifacts_only"] is True
    assert report["session_count"] == 4 * 4 * 3
    assert report["family_independent_group_counts"] == {
        family: 4 for family in BroadbandFullOctaveContractV3.canonical().source_families
    }
    assert report["canonical_matched_physical_pass"] is False
    assert report["canonical_training_eligible"] is False
    assert report["deployment_eligible"] is False
    assert report["physical_attenuation_math_performed"] is False
    assert report["blocking_requirements"] == list(UNATTESTED_PHYSICAL_PROVENANCE_BLOCKERS)


def test_nonfixture_self_attested_campaign_never_returns_cli_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """완전한 fake non-fixture artifact도 CLI success가 될 수 없다."""

    config, _plan_path, _receipt_path = _complete_campaign(tmp_path)
    config_path = tmp_path / "configs" / "self_attested.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    cli_path = REPO_ROOT / "scripts/data/check_full_octave_v3_matched_campaign.py"
    spec = importlib.util.spec_from_file_location("matched_campaign_cli_for_test", cli_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)

    assert cli.main(["--config", "configs/self_attested.yaml", "--dry-run"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == UNATTESTED_PHYSICAL_PROVENANCE_STATUS
    assert report["canonical_matched_physical_pass"] is False


def test_same_session_identity_native_to_canonical_raw_is_allowed(tmp_path: Path) -> None:
    config, _plan_path, _receipt_path = _complete_campaign(tmp_path, same_session_identity_raw=True)
    report = audit_full_octave_v3_matched_campaign(config, repository_root=tmp_path)
    assert report["status"] == UNATTESTED_PHYSICAL_PROVENANCE_STATUS
    assert report["declared_sha_structure_valid"] is True


def test_cross_session_cross_kind_raw_sha_duplicate_is_rejected(tmp_path: Path) -> None:
    config, _plan_path, _receipt_path = _complete_campaign(
        tmp_path, cross_session_cross_kind_duplicate=True
    )
    with pytest.raises(ValueError, match="raw SHA는 서로 다른 session에서 campaign-wide unique"):
        audit_full_octave_v3_matched_campaign(config, repository_root=tmp_path)


def test_fixture_plan_is_blocked_never_promoted(tmp_path: Path) -> None:
    config, plan_path, _receipt_path = _complete_campaign(tmp_path)
    plan = json.loads((tmp_path / plan_path).read_text(encoding="utf-8"))
    plan["fixture_only"] = True
    plan.pop("plan_evidence_sha256")
    plan = _sealed(plan, "plan_evidence_sha256")
    config["artifacts"]["campaign_plan"] = _write(tmp_path, plan_path, _bytes(plan))

    report = audit_full_octave_v3_matched_campaign(config, repository_root=tmp_path)
    assert report["status"] == "BLOCKED"
    assert report["fixture_only_evidence"] is True
    assert report["canonical_matched_physical_pass"] is False


def test_one_shot_receipt_rejects_reordered_or_extra_session(tmp_path: Path) -> None:
    config, _plan_path, receipt_path = _complete_campaign(tmp_path)
    _rewrite_receipt(
        tmp_path,
        config,
        receipt_path,
        lambda receipt: receipt["one_shot"].__setitem__(
            "observed_capture_order", list(reversed(receipt["one_shot"]["observed_capture_order"]))
        ),
    )
    with pytest.raises(ValueError, match="observed capture order"):
        audit_full_octave_v3_matched_campaign(config, repository_root=tmp_path)


def test_counterbalance_or_independent_group_shortcut_is_rejected_before_receipt(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    controllers = _controllers(tmp_path)
    units = []
    for index in range(4):
        order = ["OFF", "DL", "FxLMS"]
        units.append(
            {
                "comparison_unit_id": f"speech-{index}",
                "independent_group_id": f"g-{index}",
                "source": _source(tmp_path, "speech", index),
                "counterbalance_order": order,
                "sessions": [
                    {
                        "session_id": f"speech-{index}-{condition}",
                        "condition": condition,
                        "order_index": position,
                        "controller": deepcopy(controllers[condition]),
                        "bundle_config_target": f"results/nope/{index}-{condition}.yaml",
                        "comparison_run_receipt_target": (
                            f"results/nope/{index}-{condition}.json"
                        ),
                    }
                    for position, condition in enumerate(order)
                ],
            }
        )
    with pytest.raises(ValueError, match="세 counterbalance order"):
        build_full_octave_v3_matched_campaign_plan(
            campaign_identity=identity, comparison_units=units, one_shot=_one_shot(tmp_path)
        )


def test_matched_campaign_source_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    config, plan_path, receipt_path = _complete_campaign(tmp_path)
    plan = json.loads((tmp_path / plan_path).read_text(encoding="utf-8"))
    plan["comparison_units"][0]["source"]["submitted_pcm"] = _artifact(
        tmp_path, "sources/replaced_submitted_pcm.s32le", b"different submitted source"
    )
    plan.pop("plan_evidence_sha256")
    plan = _sealed(plan, "plan_evidence_sha256")
    plan_ref = _write(tmp_path, plan_path, _bytes(plan))
    config["artifacts"]["campaign_plan"] = plan_ref

    def mutate(receipt_value: dict) -> None:
        receipt_value["campaign_plan_file_sha256"] = plan_ref["sha256"]
        receipt_value["campaign_plan_evidence_sha256"] = plan["plan_evidence_sha256"]

    _rewrite_receipt(tmp_path, config, receipt_path, mutate)
    with pytest.raises(ValueError, match="submitted PCM SHA"):
        audit_full_octave_v3_matched_campaign(config, repository_root=tmp_path)


def test_session_run_receipt_rejects_level_gain_mismatch(tmp_path: Path) -> None:
    config, _plan_path, receipt_path = _complete_campaign(tmp_path)
    receipt = json.loads((tmp_path / receipt_path).read_text(encoding="utf-8"))
    first_run = receipt["session_receipts"][0]["comparison_run_receipt"]
    run_path = tmp_path / first_run["path"]
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["level_gain"]["noise_playback_gain_db"] += 1.0
    run.pop("run_evidence_sha256")
    run = _sealed(run, "run_evidence_sha256")
    run_ref = _write(tmp_path, first_run["path"], _bytes(run))

    def mutate(receipt_value: dict) -> None:
        receipt_value["session_receipts"][0]["comparison_run_receipt"] = run_ref

    _rewrite_receipt(tmp_path, config, receipt_path, mutate)
    with pytest.raises(ValueError, match="run.level_gain"):
        audit_full_octave_v3_matched_campaign(config, repository_root=tmp_path)


def test_source_and_cli_have_no_audio_gpu_network_or_writer() -> None:
    paths = [
        REPO_ROOT / "src/deep_anc/data/full_octave_v3_matched_campaign.py",
        REPO_ROOT / "scripts/data/check_full_octave_v3_matched_campaign.py",
    ]
    tree = ast.parse("\n".join(path.read_text(encoding="utf-8") for path in paths))
    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    assert not {"sounddevice", "torch", "subprocess", "socket", "requests", "alsa"} & imported
    assert not {"save", "savez", "write", "write_text", "write_bytes"} & called
