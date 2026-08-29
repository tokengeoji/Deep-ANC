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
    DEFAULT_CONFIG_RELATIVE_PATH,
    INPUT_CHANNELS,
    PLAN_SCHEMA,
    REQUIRED_ROLES,
    SIDECAR_SCHEMA,
    UNATTESTED_STRUCTURAL_RAW_BLOCKERS,
    UNATTESTED_STRUCTURAL_RAW_STATUS,
    audit_full_octave_v3_physical_session_bundle,
    build_full_octave_v3_physical_session_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sealed(payload: dict, key: str) -> dict:
    value = dict(payload)
    value[key] = _sha(_bytes(value))
    return value


def _write(root: Path, relative: str, content: bytes) -> str:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return _sha(content)


def _default_config() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / DEFAULT_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8")
    )


def _identity() -> dict:
    return {
        "source": {
            "source_kind": "submitted_pcm",
            "submitted_pcm_sha256": "1" * 64,
            "source_manifest_sha256": "2" * 64,
        },
        "controller": {
            "controller_mode": "plant_identification",
            "controller_artifact_sha256": "3" * 64,
            "controller_config_sha256": "4" * 64,
        },
        "plant": {
            "plant_campaign_contract_sha256": "5" * 64,
            "hardware_fingerprint_sha256": "6" * 64,
            "routing_geometry_sha256": "7" * 64,
        },
        "timing": {
            "training_timing_contract_sha256": "8" * 64,
            "plant_delays_sha256": "9" * 64,
            "handoff_samples": 256,
            "lead_samples": 115,
            "lead_derivation": "PlantDelays.lead()",
        },
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


def _complete_future_bundle(root: Path) -> dict:
    base = "results/physical_campaign_fixture"
    targets = {
        "native_raw": f"{base}/native.s32le",
        "canonical_raw": f"{base}/canonical.s32le",
        "session_sidecar": f"{base}/sidecar.json",
    }
    plan = build_full_octave_v3_physical_session_plan(
        artifact_targets=targets,
        role_channels=_role_channels(),
        topology="single_acquisition_clock_all_eight",
        identity=_identity(),
        planned_s32_callback_sha256="a" * 64,
    )
    assert plan["schema"] == PLAN_SCHEMA
    plan_path = f"{base}/capture_plan.json"
    plan_sha = _write(root, plan_path, _bytes(plan))

    frames = 256
    raw_size = frames * INPUT_CHANNELS * 4
    native_content = bytes(range(256)) * (raw_size // 256)
    canonical_content = bytes(reversed(range(256))) * (raw_size // 256)
    native_sha = _write(root, targets["native_raw"], native_content)
    canonical_sha = _write(root, targets["canonical_raw"], canonical_content)
    same_frame = deepcopy(plan["capture"]["same_frame_witness"])
    sidecar = _sealed(
        {
            "schema": SIDECAR_SCHEMA,
            "role": plan["role"],
            "fixture_only": False,
            "capture_plan_file_sha256": plan_sha,
            "capture_plan_evidence_sha256": plan["plan_evidence_sha256"],
            "control_band_contract_sha256": plan["control_band_contract_sha256"],
            "native_raw": {
                "path": targets["native_raw"],
                "size_bytes": len(native_content),
                "sha256": native_sha,
            },
            "canonical_raw": {
                "path": targets["canonical_raw"],
                "size_bytes": len(canonical_content),
                "sha256": canonical_sha,
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
                "same_frame_witness": same_frame,
                "frame_counter_start": 4096,
                "frame_counter_stop_exclusive": 4096 + frames,
                "planned_s32_callback_sha256": "a" * 64,
                "actual_s32_callback_sha256": "a" * 64,
                "xrun_count": 0,
                "drop_count": 0,
                "add_count": 0,
            },
            "identity": _identity(),
            "publication": _publication(),
            "canonical_training_eligible": False,
            "deployment_eligible": False,
        },
        "sidecar_evidence_sha256",
    )
    sidecar_sha = _write(root, targets["session_sidecar"], _bytes(sidecar))
    config = _default_config()
    config["artifacts"] = {
        "capture_plan": {"path": plan_path, "sha256": plan_sha},
        "native_raw": {"path": targets["native_raw"], "sha256": native_sha},
        "canonical_raw": {"path": targets["canonical_raw"], "sha256": canonical_sha},
        "session_sidecar": {"path": targets["session_sidecar"], "sha256": sidecar_sha},
    }
    return config


def test_default_static_null_config_is_read_only_blocked() -> None:
    report = audit_full_octave_v3_physical_session_bundle(
        _default_config(), repository_root=REPO_ROOT
    )
    assert report["status"] == "BLOCKED"
    assert report["raw_bundle_structural_valid"] is False
    assert report["audio_opened"] is False
    assert report["alsa_opened"] is False
    assert report["gpu_initialized"] is False
    assert report["network_opened"] is False
    assert report["results_written"] is False
    assert report["canonical_training_eligible"] is False
    assert report["deployment_eligible"] is False
    assert report["required_roles"] == list(REQUIRED_ROLES)


def test_nonfixture_eight_input_raw_bundle_is_unattested_structure_not_authority(
    tmp_path: Path,
) -> None:
    report = audit_full_octave_v3_physical_session_bundle(
        _complete_future_bundle(tmp_path), repository_root=tmp_path
    )
    assert report["status"] == UNATTESTED_STRUCTURAL_RAW_STATUS
    assert report["declared_sha_structure_valid"] is True
    assert report["raw_bundle_structural_valid"] is False
    assert report["physical_raw_provenance_attested"] is False
    assert report["self_attested_artifacts_only"] is True
    assert report["frames"] == 256
    assert report["canonical_training_eligible"] is False
    assert report["deployment_eligible"] is False
    assert report["physical_plant_identification_pass"] is False
    assert report["quiet_zone_performance_pass"] is False
    assert set(UNATTESTED_STRUCTURAL_RAW_BLOCKERS) <= set(report["blocking_requirements"])


def test_nonfixture_self_attested_bundle_cli_never_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _complete_future_bundle(tmp_path)
    config_path = tmp_path / "bundle.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    script_path = REPO_ROOT / "scripts/data/check_full_octave_v3_physical_session_bundle.py"
    spec = importlib.util.spec_from_file_location("v3_bundle_checker_test", script_path)
    assert spec is not None and spec.loader is not None
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    assert checker.main(["--config", "bundle.yaml"]) == 1


def test_fixture_only_capture_plan_is_blocked_not_promoted(tmp_path: Path) -> None:
    config = _complete_future_bundle(tmp_path)
    plan_path = tmp_path / config["artifacts"]["capture_plan"]["path"]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["fixture_only"] = True
    plan.pop("plan_evidence_sha256")
    plan = _sealed(plan, "plan_evidence_sha256")
    plan_sha = _write(tmp_path, config["artifacts"]["capture_plan"]["path"], _bytes(plan))
    config["artifacts"]["capture_plan"]["sha256"] = plan_sha

    # The plan itself is fixture-only, so the checker must stop before a later
    # fixture sidecar can be confused with actual physical authority.
    report = audit_full_octave_v3_physical_session_bundle(config, repository_root=tmp_path)
    assert report["status"] == "BLOCKED"
    assert report["fixture_only_evidence"] is True
    assert report["canonical_training_eligible"] is False


def test_fixture_only_sidecar_is_blocked_not_promoted(tmp_path: Path) -> None:
    config = _complete_future_bundle(tmp_path)
    sidecar_path = tmp_path / config["artifacts"]["session_sidecar"]["path"]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["fixture_only"] = True
    sidecar.pop("sidecar_evidence_sha256")
    sidecar = _sealed(sidecar, "sidecar_evidence_sha256")
    sidecar_sha = _write(
        tmp_path, config["artifacts"]["session_sidecar"]["path"], _bytes(sidecar)
    )
    config["artifacts"]["session_sidecar"]["sha256"] = sidecar_sha

    report = audit_full_octave_v3_physical_session_bundle(config, repository_root=tmp_path)
    assert report["status"] == "BLOCKED"
    assert report["fixture_only_evidence"] is True
    assert report["canonical_training_eligible"] is False


@pytest.mark.parametrize(
    "mutate,match",
    (
        (
            lambda p: p["capture"]["role_channels"].__setitem__("ERR_4", 6),
            "일대일",
        ),
        (
            lambda p: p["capture"]["same_frame_witness"].__setitem__("ws_witness", False),
            "ws_witness",
        ),
        (
            lambda p: p["identity"]["timing"].__setitem__("lead_derivation", "manual"),
            "PlantDelays.lead",
        ),
    ),
)
def test_plan_builder_rejects_channel_witness_and_manual_lead_shortcuts(mutate, match: str) -> None:  # type: ignore[no-untyped-def]
    targets = {
        "native_raw": "results/future/native.s32le",
        "canonical_raw": "results/future/canonical.s32le",
        "session_sidecar": "results/future/sidecar.json",
    }
    role_channels = _role_channels()
    identity = _identity()
    payload = {
        "artifact_targets": targets,
        "role_channels": role_channels,
        "topology": "single_acquisition_clock_all_eight",
        "identity": identity,
        "planned_s32_callback_sha256": "a" * 64,
    }
    # Mutate the candidate exactly where the actual builder receives it.
    if "role_channels" in match or match == "일대일":
        mutate({"capture": {"role_channels": role_channels}, "identity": identity})
    elif match == "ws_witness":
        # Builder owns this witness; use an already-built plan for the exact schema path.
        plan = build_full_octave_v3_physical_session_plan(**payload)
        mutate(plan)
        plan.pop("plan_evidence_sha256")
        plan = _sealed(plan, "plan_evidence_sha256")
        with pytest.raises(ValueError, match=match):
            from deep_anc.data.full_octave_v3_physical_bundle import _validate_plan

            _validate_plan(plan)
        return
    else:
        mutate({"capture": {"role_channels": role_channels}, "identity": identity})
    with pytest.raises(ValueError, match=match):
        build_full_octave_v3_physical_session_plan(**payload)


def test_sidecar_rejects_raw_length_and_channel_map_mismatch(tmp_path: Path) -> None:
    config = _complete_future_bundle(tmp_path)
    sidecar_path = tmp_path / config["artifacts"]["session_sidecar"]["path"]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["capture"]["frames"] = 512
    sidecar["capture"]["frame_counter_stop_exclusive"] += 256
    sidecar.pop("sidecar_evidence_sha256")
    sidecar = _sealed(sidecar, "sidecar_evidence_sha256")
    sidecar_sha = _write(tmp_path, config["artifacts"]["session_sidecar"]["path"], _bytes(sidecar))
    config["artifacts"]["session_sidecar"]["sha256"] = sidecar_sha
    with pytest.raises(ValueError, match="byte length"):
        audit_full_octave_v3_physical_session_bundle(config, repository_root=tmp_path)


def test_sidecar_rejects_actual_callback_sha_mismatch(tmp_path: Path) -> None:
    config = _complete_future_bundle(tmp_path)
    sidecar_path = tmp_path / config["artifacts"]["session_sidecar"]["path"]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["capture"]["actual_s32_callback_sha256"] = "b" * 64
    sidecar.pop("sidecar_evidence_sha256")
    sidecar = _sealed(sidecar, "sidecar_evidence_sha256")
    sidecar_sha = _write(
        tmp_path, config["artifacts"]["session_sidecar"]["path"], _bytes(sidecar)
    )
    config["artifacts"]["session_sidecar"]["sha256"] = sidecar_sha

    with pytest.raises(ValueError, match="actual S32 callback SHA"):
        audit_full_octave_v3_physical_session_bundle(config, repository_root=tmp_path)


def test_bundle_source_and_cli_have_no_audio_gpu_network_or_writer() -> None:
    paths = [
        REPO_ROOT / "src/deep_anc/data/full_octave_v3_physical_bundle.py",
        REPO_ROOT / "scripts/data/check_full_octave_v3_physical_session_bundle.py",
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
    assert not {"sounddevice", "torch", "subprocess", "socket", "requests"} & imported
    assert not {"save", "savez", "write", "write_text", "write_bytes"} & called
