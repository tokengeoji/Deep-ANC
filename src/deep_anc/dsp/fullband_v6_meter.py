"""v6 meter/followup namespace; v5 immutable writer를 그대로 재사용한다."""

from __future__ import annotations

import datetime as dt
import hashlib
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

import yaml
from deep_anc.data.repository_fd import (
    RepositoryFileGuard,
    assert_repository_target_fresh_nofollow,
    repository_execution_identity,
    repository_root as _repository_root,
)
from deep_anc.dsp.fullband_v5_meter import (
    CONFIRMATION_KEYS,
    DEFAULT_LEVEL_EVIDENCE_PATH,
    TRACKED_V5_LEVEL_ATTESTATION_SCOPE,
    load_tracked_v5_level_attestation,
    payload_sha256,
    resolve_fullband_v5_devices,
    write_fullband_v5_meter_raw_atomic,
)
from deep_anc.dsp.measurement_level import (
    collect_alsa_physical_fingerprint,
    measurement_hardware_identity,
    meter_receipt_path,
    validate_bootstrap_meter_raw,
    validate_measurement_hardware_contract,
)
from deep_anc.dsp import fullband_live_authority_v6 as authority

FOLLOWUP_SCHEMA = "fullband_causal_v6_meter_followup_v1"
FULLBAND_V6_METER_IDENTITY_SCHEMA = "fullband_causal_v6_meter_identity_v1"
FOLLOWUP_SCOPE = "fresh_level_meter_binding_only_not_plant_delay_or_training_authority"
DEFAULT_PLAN_ENVELOPE_PATH = authority.SEALED_PLAN_ENVELOPE_RELATIVE_PATH
DEFAULT_LIVE_AUTHORITY_PATH = authority.SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH
DEFAULT_RAW_TARGET_PATH = authority.SEALED_RAW_RELATIVE_PATH
DEFAULT_HARDWARE_PATH = authority.SEALED_HARDWARE_RELATIVE_PATH
SET_AMP_REPOSITORY_PATH = "scripts/data/set_amp_level.py"


def _validate_repository_execution_binding_v6(
    metadata: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    """meter를 만든 clean exact ``set_amp_level.py`` checkout과 현재 실행을 결속한다."""

    saved_execution = metadata.get("repository_execution")
    current_execution = repository_execution_identity(
        repository_root, SET_AMP_REPOSITORY_PATH
    )
    if saved_execution != current_execution:
        raise ValueError(
            "v6 meter repository commit/branch/script path/SHA가 "
            "current clean checkout과 다릅니다"
        )
    return dict(current_execution)


def write_fullband_v6_meter_raw_atomic(*args, **kwargs):  # noqa: ANN002,ANN003
    """검증된 held-dirfd/no-replace/same-inode recovery writer를 공유한다."""
    if "_generation_label" in kwargs or "_recovery_tag" in kwargs:
        raise TypeError("v6 meter writer private generation override를 허용하지 않습니다")
    return write_fullband_v5_meter_raw_atomic(
        *args,
        **kwargs,
        _generation_label="v6",
        _recovery_tag="v6_raw",
    )


def build_fullband_v6_followup(
    contract: Mapping[str, Any], *, resolved_devices: Mapping[str, int],
    confirmations: Mapping[str, bool],
) -> dict[str, Any]:
    if set(resolved_devices) != {"input", "output"} or any(
        type(value) is not int or value < 0 for value in resolved_devices.values()
    ):
        raise ValueError("v6 resolved device 계약이 exact하지 않습니다")
    if set(confirmations) != CONFIRMATION_KEYS or any(value is not True for value in confirmations.values()):
        raise ValueError("v6 meter의 다섯 확인이 모두 필요합니다")
    core = {
        "schema": FOLLOWUP_SCHEMA,
        "scope": FOLLOWUP_SCOPE,
        "signal_plan": dict(contract["plan"]),
        "live_capture_authority": dict(contract["live_capture_authority"]),
        "hardware": dict(contract["hardware"]),
        "level_evidence": dict(contract["level_evidence"]),
        "sealed_raw": dict(contract["sealed_raw"]),
        "resolved_devices": dict(resolved_devices),
        "operator_confirmations": dict(confirmations),
    }
    return {**core, "followup_contract_sha256": payload_sha256(core)}


def validate_fullband_v6_followup(value: Mapping[str, Any], *, expected_contract: Mapping[str, Any], expected_devices: Mapping[str, int]) -> dict[str, Any]:
    expected = build_fullband_v6_followup(
        expected_contract,
        resolved_devices=expected_devices,
        confirmations={key: True for key in CONFIRMATION_KEYS},
    )
    if dict(value) != expected:
        raise ValueError("v6 meter followup가 current authority/profile과 다릅니다")
    return dict(expected)


def validate_fullband_v6_static_contract(
    *, repository_root: str | os.PathLike[str],
    plan_envelope_path: str | Path = DEFAULT_PLAN_ENVELOPE_PATH,
    live_authority_path: str | Path = DEFAULT_LIVE_AUTHORITY_PATH,
    level_evidence_path: str | Path = DEFAULT_LEVEL_EVIDENCE_PATH,
    hardware_path: str | Path = DEFAULT_HARDWARE_PATH,
    raw_target_path: str | Path = DEFAULT_RAW_TARGET_PATH,
    require_sealed_raw_fresh: bool,
) -> dict[str, Any]:
    """v6 exact assets와 현재 hardware/evidence를 backend import 전에 결속한다."""
    root = _repository_root(repository_root)
    exact = {
        "plan": (plan_envelope_path, DEFAULT_PLAN_ENVELOPE_PATH),
        "authority": (live_authority_path, DEFAULT_LIVE_AUTHORITY_PATH),
        "hardware": (hardware_path, DEFAULT_HARDWARE_PATH),
        "evidence": (level_evidence_path, DEFAULT_LEVEL_EVIDENCE_PATH),
        "raw": (raw_target_path, DEFAULT_RAW_TARGET_PATH),
    }
    for label, (supplied, relative) in exact.items():
        candidate = Path(os.path.abspath(os.fspath(supplied if Path(supplied).is_absolute() else root / supplied)))
        if candidate != root / relative:
            raise ValueError(f"v6 {label} path가 exact sealed path와 다릅니다")
    plan = authority.load_exact_saved_plan_v6(root / DEFAULT_PLAN_ENVELOPE_PATH, repository_root=root, expected_file_sha256=authority.EXPECTED_PLAN_ENVELOPE_FILE_SHA256)
    live = authority.load_exact_saved_live_capture_authority_v6(root / DEFAULT_LIVE_AUTHORITY_PATH, repository_root=root, expected_file_sha256=authority.EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256, expected_payload_sha256=authority.EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256)
    with RepositoryFileGuard(root, DEFAULT_HARDWARE_PATH, label="v6 hardware") as hardware_guard:
        if hardware_guard.sha256 != authority.EXPECTED_HARDWARE_FILE_SHA256:
            raise ValueError("v6 hardware SHA가 pinned 값과 다릅니다")
        hardware_config = yaml.safe_load(hardware_guard.bytes.decode("utf-8"))
        audio, channel_map = validate_measurement_hardware_contract(hardware_config)
        hardware_guard.verify()
    attestation = load_tracked_v5_level_attestation(root / DEFAULT_LEVEL_EVIDENCE_PATH, repository_root=root)
    fingerprint = collect_alsa_physical_fingerprint(hardware_config)
    identity = measurement_hardware_identity(hardware_config, physical_fingerprint=fingerprint)
    if attestation["hardware_identity"] != identity:
        raise ValueError("v6 level attestation과 current hardware가 다릅니다")
    if require_sealed_raw_fresh:
        assert_repository_target_fresh_nofollow(root, DEFAULT_RAW_TARGET_PATH, create_parents=False)
    else:
        with RepositoryFileGuard(root, DEFAULT_RAW_TARGET_PATH, label="v6 sealed raw") as guard:
            guard.verify()
    return {
        "plan": {key: plan[key] for key in ("path", "file_sha256", "payload_sha256", "pcm_sha256")},
        "live_capture_authority": {key: live[key] for key in ("path", "file_sha256", "payload_sha256")},
        "hardware": {"path": DEFAULT_HARDWARE_PATH, "file_sha256": authority.EXPECTED_HARDWARE_FILE_SHA256, "identity_sha256": payload_sha256(identity), "physical_fingerprint_sha256": fingerprint["sha256"]},
        "level_evidence": {"path": DEFAULT_LEVEL_EVIDENCE_PATH, "file_sha256": attestation["file_sha256"], "identity_sha256": payload_sha256(identity), "scope": TRACKED_V5_LEVEL_ATTESTATION_SCOPE, "preserved_raw_revalidated": False},
        "sealed_raw": {"path": DEFAULT_RAW_TARGET_PATH, "must_not_exist_before_capture": True},
        "hardware_config": hardware_config, "hardware_audio": audio, "channel_map": channel_map,
        "hardware_identity": identity, "physical_fingerprint": fingerprint,
    }


def resolve_fullband_v6_devices(contract: Mapping[str, Any], *, sd_module=None) -> dict[str, int]:  # noqa: ANN001
    return resolve_fullband_v5_devices(contract, sd_module=sd_module)


def validate_fullband_v6_meter_raw_static(
    raw_path: str | Path, *, repository_root: str | os.PathLike[str],
    now_utc: dt.datetime | None = None, require_fresh: bool = True,
    require_sealed_raw_fresh: bool = True,
) -> dict[str, Any]:
    root = _repository_root(repository_root)
    raw = Path(os.path.abspath(os.fspath(raw_path if Path(raw_path).is_absolute() else root / raw_path)))
    relative = raw.relative_to(root).as_posix()
    receipt = meter_receipt_path(raw)
    with ExitStack() as stack:
        raw_guard = stack.enter_context(RepositoryFileGuard(root, relative, label="v6 meter raw"))
        receipt_guard = stack.enter_context(RepositoryFileGuard(root, receipt.relative_to(root).as_posix(), label="v6 meter receipt"))
        contract = validate_fullband_v6_static_contract(repository_root=root, require_sealed_raw_fresh=require_sealed_raw_fresh)
        verified = validate_bootstrap_meter_raw(raw, repository_root=root, expected_hardware_identity=contract["hardware_identity"], now_utc=now_utc, require_fresh=require_fresh)
        _validate_repository_execution_binding_v6(
            verified["metadata"], repository_root=root
        )
        followup = verified["metadata"].get("fullband_v6_followup")
        if not isinstance(followup, Mapping):
            raise ValueError("v6 meter raw에 v6 followup이 없습니다")
        devices = followup.get("resolved_devices")
        validate_fullband_v6_followup(followup, expected_contract=contract, expected_devices=devices)
        if verified["metadata"].get("fullband_v6_post_capture_revalidation") != {"passed": True, "error": None}:
            raise ValueError("v6 meter post-capture binding이 PASS가 아닙니다")
        if raw_guard.sha256 != verified["sha256"]:
            raise ValueError("v6 meter raw held SHA가 generic validator와 다릅니다")
        raw_guard.verify(); receipt_guard.verify()
    completed = verified["completed_at_utc"].isoformat()
    identity_payload = {
        "schema": FULLBAND_V6_METER_IDENTITY_SCHEMA,
        "path": relative,
        "receipt_path": receipt.relative_to(root).as_posix(),
        "raw_sha256": raw_guard.sha256,
        "receipt_sha256": receipt_guard.sha256,
        "completed_at_utc": completed,
        "followup_contract_sha256": followup["followup_contract_sha256"],
    }
    return {"path": raw, "receipt_path": receipt, "raw_sha256": raw_guard.sha256,
            "receipt_sha256": receipt_guard.sha256, "metadata": verified["metadata"],
            "identity_sha256": payload_sha256(identity_payload),
            "completed_at_utc": verified["completed_at_utc"],
            "followup_contract_sha256": followup["followup_contract_sha256"],
            "plan": contract["plan"],
            "live_capture_authority": contract["live_capture_authority"],
            "level_evidence": contract["level_evidence"],
            "hardware": {**contract["hardware"], "resolved_devices": dict(devices)}}


def validate_fullband_v6_meter_raw(
    raw_path: str | Path, *, repository_root: str | os.PathLike[str],
    now_utc: dt.datetime | None = None, require_fresh: bool = True,
    require_sealed_raw_fresh: bool = True, sd_module=None,
) -> dict[str, Any]:  # noqa: ANN001
    result = validate_fullband_v6_meter_raw_static(
        raw_path, repository_root=repository_root, now_utc=now_utc,
        require_fresh=require_fresh,
        require_sealed_raw_fresh=require_sealed_raw_fresh,
    )
    contract = validate_fullband_v6_static_contract(
        repository_root=repository_root,
        require_sealed_raw_fresh=require_sealed_raw_fresh,
    )
    current = resolve_fullband_v6_devices(contract, sd_module=sd_module)
    if result["hardware"]["resolved_devices"] != current:
        raise ValueError("v6 meter embedded device가 current PortAudio device와 다릅니다")
    return result


__all__ = [
    "DEFAULT_LIVE_AUTHORITY_PATH", "DEFAULT_PLAN_ENVELOPE_PATH",
    "DEFAULT_RAW_TARGET_PATH", "DEFAULT_HARDWARE_PATH", "FOLLOWUP_SCHEMA", "FOLLOWUP_SCOPE",
    "FULLBAND_V6_METER_IDENTITY_SCHEMA",
    "build_fullband_v6_followup", "validate_fullband_v6_followup",
    "validate_fullband_v6_static_contract", "validate_fullband_v6_meter_raw_static",
    "validate_fullband_v6_meter_raw",
    "resolve_fullband_v6_devices", "write_fullband_v6_meter_raw_atomic",
]
