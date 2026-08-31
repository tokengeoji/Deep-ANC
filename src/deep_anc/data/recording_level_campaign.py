"""Fresh measurement-level meter를 추가 녹음과 결속하는 독립 authority.

이 모듈은 오디오 장치를 열지 않는다. 기존
``measurement_level_meter_raw_v1`` NPZ와 sibling receipt를 재검증하고,
하나의 fresh level campaign receipt로 묶는다. 여러 session은 큰 meter raw를
복제하지 않고 이 campaign path/SHA만 참조한다.

아날로그 amplifier knob은 ALSA fingerprint로 읽을 수 없다. 따라서 세션
binding은 fresh meter 시각과 ``same_amplifier_setting=True``를 둘 다 강제한다.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from ..dsp.measurement_level import (
    BOOTSTRAP_METER_MAX_AGE_SECONDS,
    BOOTSTRAP_METER_RECEIPT_SCHEMA,
    OFFICIAL_MEASUREMENT_LEVEL,
    atomic_publish_noreplace,
    band_rms_dbfs,
    measurement_hardware_identity,
    meter_receipt_path,
    validate_bootstrap_meter_raw,
)
from .holdout_contract import (
    FileSnapshot,
    HoldoutContractError,
    read_regular_file_snapshot,
    reject_symlink_components,
)
from .recording_source_gain import (
    PHYSICAL_SELECTOR_MAX_AMPLITUDE_MILLIONTHS,
    RECORDING_SOURCE_GAIN_SESSION_BINDING_SCHEMA,
    validate_recording_source_gain_session_binding,
)


RECORDING_LEVEL_CAMPAIGN_SCHEMA = "recording_level_campaign_v1"
RECORDING_LEVEL_SESSION_BINDING_SCHEMA = "recording_level_session_binding_v1"
RECORDING_LEVEL_RENDERED_SOURCE_SCHEMA = "recording_rendered_source_level_v1"
RECORDING_LEVEL_SOURCE_GAIN_SESSION_BINDING_SCHEMA = (
    "recording_level_source_gain_session_binding/v3_dynamic_gainprobe006"
)
RECORDING_LEVEL_RENDERED_SOURCE_V2_SCHEMA = (
    "recording_rendered_source_level/v3_dynamic_gainprobe006"
)
RECORDING_LEVEL_CAMPAIGN_ROOT = "results/recording_level_campaigns"
RECORDING_LEVEL_CAMPAIGN_FILENAME = "campaign.json"
RECORDING_LEVEL_MAX_AGE_SECONDS = BOOTSTRAP_METER_MAX_AGE_SECONDS
RECORDING_LEVEL_PLAYBACK_AMPLITUDE = 0.06
RECORDING_LEVEL_SAMPLE_RATE = 48_000
RECORDING_LEVEL_SESSION_FRAMES = 15 * RECORDING_LEVEL_SAMPLE_RATE

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CAPTURE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_CAMPAIGN_ID_RE = re.compile(r"^recording-level-[0-9a-f]{64}$")


class RecordingLevelCampaignError(ValueError):
    """Level campaign authority 계약 위반."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RecordingLevelCampaignError(
            "level campaign에 JSON으로 봉인할 수 없는 값이 있습니다"
        ) from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _without_seal(payload: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != field}


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecordingLevelCampaignError(
                f"level campaign JSON 중복 키는 허용되지 않습니다: {key}"
            )
        result[key] = value
    return result


def _load_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_object_without_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordingLevelCampaignError(f"{label} JSON을 읽을 수 없습니다") from exc
    if not isinstance(value, dict):
        raise RecordingLevelCampaignError(f"{label} 최상위는 mapping이어야 합니다")
    return value


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):  # noqa: ANN001
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise RecordingLevelCampaignError(
                f"hardware YAML 중복 키는 허용되지 않습니다: {key!r}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_hardware_config(raw: bytes) -> dict[str, Any]:
    try:
        value = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeySafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RecordingLevelCampaignError("hardware config YAML을 읽을 수 없습니다") from exc
    if not isinstance(value, dict):
        raise RecordingLevelCampaignError("hardware config 최상위는 mapping이어야 합니다")
    return value


def _root(path: str | Path) -> Path:
    root = Path(os.path.abspath(os.fspath(path)))
    try:
        info = root.lstat()
    except FileNotFoundError as exc:
        raise RecordingLevelCampaignError(f"저장소 root가 없습니다: {root}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RecordingLevelCampaignError(f"저장소 root가 안전한 directory가 아닙니다: {root}")
    return root


def _lexical_repo_path(root: Path, value: str | Path, *, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RecordingLevelCampaignError(
            f"{label} 경로는 저장소 안이어야 합니다: {candidate}"
        ) from exc
    return candidate


def _snapshot(root: Path, value: str | Path, *, label: str) -> FileSnapshot:
    candidate = _lexical_repo_path(root, value, label=label)
    try:
        return read_regular_file_snapshot(
            candidate,
            root=root,
            label=label,
            capture_bytes=True,
        )
    except HoldoutContractError as exc:
        raise RecordingLevelCampaignError(str(exc)) from exc


def _relative(root: Path, snapshot: FileSnapshot) -> str:
    return snapshot.path.relative_to(root).as_posix()


def _file_ref(root: Path, snapshot: FileSnapshot) -> dict[str, Any]:
    return {
        "path": _relative(root, snapshot),
        "size": snapshot.size,
        "sha256": snapshot.sha256,
    }


def _snapshot_identity(snapshot: FileSnapshot) -> tuple[Any, ...]:
    return (
        snapshot.path,
        snapshot.device,
        snapshot.inode,
        snapshot.size,
        snapshot.mtime_ns,
        snapshot.ctime_ns,
        snapshot.sha256,
    )


def _require_unchanged(before: FileSnapshot, after: FileSnapshot, *, label: str) -> None:
    if _snapshot_identity(before) != _snapshot_identity(after):
        raise RecordingLevelCampaignError(f"{label}이 검증 도중 변경/retarget됐습니다")


def _parse_utc(value: Any, *, label: str) -> dt.datetime:
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, str) and value.strip() == value and value:
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RecordingLevelCampaignError(f"{label} UTC timestamp가 잘못됐습니다") from exc
    else:
        raise RecordingLevelCampaignError(f"{label} UTC timestamp가 필요합니다")
    if parsed.tzinfo is None:
        raise RecordingLevelCampaignError(f"{label}는 timezone-aware timestamp여야 합니다")
    return parsed.astimezone(dt.timezone.utc)


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordingLevelCampaignError(f"{label}는 finite 수치여야 합니다")
    number = float(value)
    if not math.isfinite(number):
        raise RecordingLevelCampaignError(f"{label}는 finite 수치여야 합니다")
    return number


def _validate_ref(
    value: Any,
    *,
    expected: Mapping[str, Any],
    label: str,
) -> None:
    if not isinstance(value, Mapping) or dict(value) != dict(expected):
        raise RecordingLevelCampaignError(f"{label} path/size/SHA 결속이 다릅니다")


def _campaign_anchor(
    *,
    capture_id: str,
    completed_at_utc: str,
    raw_sha256: str,
    receipt_sha256: str,
    hardware_config_sha256: str,
) -> dict[str, str]:
    return {
        "capture_id": capture_id,
        "completed_at_utc": completed_at_utc,
        "raw_sha256": raw_sha256,
        "receipt_sha256": receipt_sha256,
        "hardware_config_sha256": hardware_config_sha256,
    }


def _campaign_id(anchor: Mapping[str, Any]) -> str:
    return "recording-level-" + _canonical_sha256(dict(anchor))


def campaign_receipt_relative_path(campaign_id: str) -> str:
    """Campaign ID의 고정 no-replace receipt 경로."""

    if not isinstance(campaign_id, str) or _CAMPAIGN_ID_RE.fullmatch(campaign_id) is None:
        raise RecordingLevelCampaignError("campaign_id 형식이 잘못됐습니다")
    return f"{RECORDING_LEVEL_CAMPAIGN_ROOT}/{campaign_id}/{RECORDING_LEVEL_CAMPAIGN_FILENAME}"


def _validate_meter_authority(
    *,
    repo_root: Path,
    meter_raw: str | Path,
    meter_receipt: str | Path,
    hardware_config: str | Path,
    now_utc: dt.datetime,
    require_fresh: bool,
) -> dict[str, Any]:
    """Safe snapshots 사이에 기존 full meter validator를 실행한다."""

    raw_before = _snapshot(repo_root, meter_raw, label="recording level meter raw")
    receipt_before = _snapshot(
        repo_root, meter_receipt, label="recording level meter receipt"
    )
    hardware_before = _snapshot(
        repo_root, hardware_config, label="recording level hardware config"
    )
    expected_receipt_path = meter_receipt_path(raw_before.path)
    if receipt_before.path != expected_receipt_path:
        raise RecordingLevelCampaignError(
            "meter receipt는 raw의 canonical sibling .receipt.json이어야 합니다"
        )
    assert receipt_before.data is not None
    receipt_payload = _load_json_object(
        receipt_before.data, label="recording level meter receipt"
    )
    expected_receipt_payload = {
        "schema": BOOTSTRAP_METER_RECEIPT_SCHEMA,
        "raw_path": _relative(repo_root, raw_before),
        "raw_sha256": raw_before.sha256,
    }
    if receipt_payload != expected_receipt_payload:
        raise RecordingLevelCampaignError(
            "meter receipt가 raw path/SHA/schema와 exact하게 일치하지 않습니다"
        )

    try:
        meter = validate_bootstrap_meter_raw(
            raw_before.path,
            repository_root=repo_root,
            now_utc=now_utc,
            require_fresh=require_fresh,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise RecordingLevelCampaignError(f"meter raw 재검증 실패: {exc}") from exc
    if meter.get("sha256") != raw_before.sha256:
        raise RecordingLevelCampaignError("meter validator가 safe snapshot과 다른 raw를 읽었습니다")
    metadata = meter.get("metadata")
    identity = metadata.get("hardware_identity") if isinstance(metadata, dict) else None
    if not isinstance(identity, dict):
        raise RecordingLevelCampaignError("meter hardware identity가 없습니다")
    capture_id = metadata.get("capture_id")
    if not isinstance(capture_id, str) or _CAPTURE_ID_RE.fullmatch(capture_id) is None:
        raise RecordingLevelCampaignError(
            "meter capture_id는 32자리 lowercase hex여야 합니다"
        )
    fingerprint = identity.get("physical_fingerprint")
    if not isinstance(fingerprint, dict) or _SHA256_RE.fullmatch(
        str(fingerprint.get("sha256") or "")
    ) is None:
        raise RecordingLevelCampaignError("meter physical fingerprint SHA가 없습니다")

    assert hardware_before.data is not None
    config = _load_hardware_config(hardware_before.data)
    try:
        expected_identity = measurement_hardware_identity(
            config,
            physical_fingerprint=fingerprint,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RecordingLevelCampaignError(
            f"hardware config/identity 계약 검증 실패: {exc}"
        ) from exc
    if identity != expected_identity:
        raise RecordingLevelCampaignError(
            "meter hardware identity가 결속할 hardware config에서 재유도되지 않습니다"
        )

    raw_after = _snapshot(repo_root, raw_before.path, label="recording level meter raw")
    receipt_after = _snapshot(
        repo_root, receipt_before.path, label="recording level meter receipt"
    )
    hardware_after = _snapshot(
        repo_root, hardware_before.path, label="recording level hardware config"
    )
    _require_unchanged(raw_before, raw_after, label="meter raw")
    _require_unchanged(receipt_before, receipt_after, label="meter receipt")
    _require_unchanged(hardware_before, hardware_after, label="hardware config")

    completed = meter.get("completed_at_utc")
    if not isinstance(completed, dt.datetime):
        raise RecordingLevelCampaignError("meter completed_at_utc 검증값이 없습니다")
    completed = completed.astimezone(dt.timezone.utc)
    level = _finite_number(meter.get("meter_ch0_dbfs"), label="meter_ch0_dbfs")
    return {
        "raw_snapshot": raw_before,
        "receipt_snapshot": receipt_before,
        "hardware_snapshot": hardware_before,
        "capture_id": capture_id,
        "completed_at_utc": completed,
        "meter_ch0_dbfs": level,
        "hardware_identity": json.loads(_canonical_json_bytes(identity)),
        "hardware_identity_sha256": _canonical_sha256(identity),
        "physical_fingerprint": json.loads(_canonical_json_bytes(fingerprint)),
        "physical_fingerprint_sha256": str(fingerprint["sha256"]),
    }


def build_recording_level_campaign_payload(
    *,
    repo_root: str | Path,
    meter_raw: str | Path,
    meter_receipt: str | Path,
    hardware_config: str | Path,
    now_utc: dt.datetime | None = None,
) -> dict[str, Any]:
    """Fresh meter authority를 self-sealed campaign payload로 만든다."""

    root = _root(repo_root)
    issued = _parse_utc(
        now_utc or dt.datetime.now(dt.timezone.utc), label="campaign issued_at_utc"
    )
    authority = _validate_meter_authority(
        repo_root=root,
        meter_raw=meter_raw,
        meter_receipt=meter_receipt,
        hardware_config=hardware_config,
        now_utc=issued,
        require_fresh=True,
    )
    completed = authority["completed_at_utc"]
    age = float((issued - completed).total_seconds())
    if age < 0.0 or age > RECORDING_LEVEL_MAX_AGE_SECONDS:
        raise RecordingLevelCampaignError(
            f"campaign 발행 시 meter age가 0..{RECORDING_LEVEL_MAX_AGE_SECONDS}초가 아닙니다: {age:.6f}"
        )
    completed_text = completed.isoformat()
    raw_ref = _file_ref(root, authority["raw_snapshot"])
    receipt_ref = _file_ref(root, authority["receipt_snapshot"])
    hardware_ref = _file_ref(root, authority["hardware_snapshot"])
    anchor = _campaign_anchor(
        capture_id=authority["capture_id"],
        completed_at_utc=completed_text,
        raw_sha256=raw_ref["sha256"],
        receipt_sha256=receipt_ref["sha256"],
        hardware_config_sha256=hardware_ref["sha256"],
    )
    payload: dict[str, Any] = {
        "schema": RECORDING_LEVEL_CAMPAIGN_SCHEMA,
        "campaign_id": _campaign_id(anchor),
        "issued_at_utc": issued.isoformat(),
        "meter": {
            "capture_id": authority["capture_id"],
            "completed_at_utc": completed_text,
            "age_seconds_at_issue": age,
            "raw": raw_ref,
            "receipt": receipt_ref,
            "probe_peak": OFFICIAL_MEASUREMENT_LEVEL.probe_amplitude,
            "ch0_dbfs": authority["meter_ch0_dbfs"],
            "target_dbfs": OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs,
            "tolerance_db": OFFICIAL_MEASUREMENT_LEVEL.meter_tolerance_db,
        },
        "hardware": {
            "config": hardware_ref,
            "identity_sha256": authority["hardware_identity_sha256"],
            "physical_fingerprint_sha256": authority[
                "physical_fingerprint_sha256"
            ],
        },
        "session_contract": {
            "max_meter_age_seconds": RECORDING_LEVEL_MAX_AGE_SECONDS,
            "same_amplifier_setting_required": True,
            "playback_amplitude": RECORDING_LEVEL_PLAYBACK_AMPLITUDE,
            "sample_rate": RECORDING_LEVEL_SAMPLE_RATE,
            "frames": RECORDING_LEVEL_SESSION_FRAMES,
            "trusted_band_hz": list(OFFICIAL_MEASUREMENT_LEVEL.meter_band_hz),
            "rendered_level_metric": "measurement_level.band_rms_dbfs_hann_v1",
        },
    }
    payload["evidence_sha256"] = _canonical_sha256(payload)
    return payload


def _validate_campaign_payload(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
    now_utc: dt.datetime,
    require_fresh: bool,
) -> dict[str, Any]:
    required = {
        "schema",
        "campaign_id",
        "issued_at_utc",
        "meter",
        "hardware",
        "session_contract",
        "evidence_sha256",
    }
    if set(payload) != required or payload.get("schema") != RECORDING_LEVEL_CAMPAIGN_SCHEMA:
        raise RecordingLevelCampaignError("recording level campaign schema가 다릅니다")
    seal = payload.get("evidence_sha256")
    if (
        not isinstance(seal, str)
        or _SHA256_RE.fullmatch(seal) is None
        or seal != _canonical_sha256(_without_seal(payload, field="evidence_sha256"))
    ):
        raise RecordingLevelCampaignError("recording level campaign self-seal이 다릅니다")
    meter = payload.get("meter")
    hardware = payload.get("hardware")
    session_contract = payload.get("session_contract")
    meter_keys = {
        "capture_id",
        "completed_at_utc",
        "age_seconds_at_issue",
        "raw",
        "receipt",
        "probe_peak",
        "ch0_dbfs",
        "target_dbfs",
        "tolerance_db",
    }
    hardware_keys = {"config", "identity_sha256", "physical_fingerprint_sha256"}
    contract_exact = {
        "max_meter_age_seconds": RECORDING_LEVEL_MAX_AGE_SECONDS,
        "same_amplifier_setting_required": True,
        "playback_amplitude": RECORDING_LEVEL_PLAYBACK_AMPLITUDE,
        "sample_rate": RECORDING_LEVEL_SAMPLE_RATE,
        "frames": RECORDING_LEVEL_SESSION_FRAMES,
        "trusted_band_hz": list(OFFICIAL_MEASUREMENT_LEVEL.meter_band_hz),
        "rendered_level_metric": "measurement_level.band_rms_dbfs_hann_v1",
    }
    if not isinstance(meter, Mapping) or set(meter) != meter_keys:
        raise RecordingLevelCampaignError("campaign meter 필드 집합이 다릅니다")
    if not isinstance(hardware, Mapping) or set(hardware) != hardware_keys:
        raise RecordingLevelCampaignError("campaign hardware 필드 집합이 다릅니다")
    if not isinstance(session_contract, Mapping) or dict(session_contract) != contract_exact:
        raise RecordingLevelCampaignError("campaign session_contract가 다릅니다")

    authority = _validate_meter_authority(
        repo_root=repo_root,
        meter_raw=str(meter.get("raw", {}).get("path", ""))
        if isinstance(meter.get("raw"), Mapping)
        else "",
        meter_receipt=str(meter.get("receipt", {}).get("path", ""))
        if isinstance(meter.get("receipt"), Mapping)
        else "",
        hardware_config=str(hardware.get("config", {}).get("path", ""))
        if isinstance(hardware.get("config"), Mapping)
        else "",
        now_utc=now_utc,
        require_fresh=require_fresh,
    )
    raw_ref = _file_ref(repo_root, authority["raw_snapshot"])
    receipt_ref = _file_ref(repo_root, authority["receipt_snapshot"])
    config_ref = _file_ref(repo_root, authority["hardware_snapshot"])
    _validate_ref(meter.get("raw"), expected=raw_ref, label="campaign meter raw")
    _validate_ref(
        meter.get("receipt"), expected=receipt_ref, label="campaign meter receipt"
    )
    _validate_ref(
        hardware.get("config"), expected=config_ref, label="campaign hardware config"
    )
    completed = _parse_utc(meter.get("completed_at_utc"), label="meter completed_at_utc")
    issued = _parse_utc(payload.get("issued_at_utc"), label="campaign issued_at_utc")
    age = float((issued - completed).total_seconds())
    stored_age = _finite_number(
        meter.get("age_seconds_at_issue"), label="meter age_seconds_at_issue"
    )
    if (
        completed != authority["completed_at_utc"]
        or meter.get("capture_id") != authority["capture_id"]
        or age < 0.0
        or age > RECORDING_LEVEL_MAX_AGE_SECONDS
        or not math.isclose(stored_age, age, rel_tol=0.0, abs_tol=1e-9)
        or not math.isclose(
            _finite_number(meter.get("probe_peak"), label="meter probe_peak"),
            OFFICIAL_MEASUREMENT_LEVEL.probe_amplitude,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite_number(meter.get("ch0_dbfs"), label="meter ch0_dbfs"),
            authority["meter_ch0_dbfs"],
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            _finite_number(meter.get("target_dbfs"), label="meter target_dbfs"),
            OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite_number(meter.get("tolerance_db"), label="meter tolerance_db"),
            OFFICIAL_MEASUREMENT_LEVEL.meter_tolerance_db,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RecordingLevelCampaignError("campaign meter level/time 결속이 다릅니다")
    if (
        hardware.get("identity_sha256") != authority["hardware_identity_sha256"]
        or hardware.get("physical_fingerprint_sha256")
        != authority["physical_fingerprint_sha256"]
    ):
        raise RecordingLevelCampaignError("campaign hardware identity/fingerprint SHA가 다릅니다")
    anchor = _campaign_anchor(
        capture_id=authority["capture_id"],
        completed_at_utc=completed.isoformat(),
        raw_sha256=raw_ref["sha256"],
        receipt_sha256=receipt_ref["sha256"],
        hardware_config_sha256=config_ref["sha256"],
    )
    campaign_id = payload.get("campaign_id")
    if campaign_id != _campaign_id(anchor):
        raise RecordingLevelCampaignError("campaign_id가 authority anchor에서 재유도되지 않습니다")
    return {
        "campaign_id": campaign_id,
        "payload": json.loads(_canonical_json_bytes(dict(payload))),
        "hardware_identity": authority["hardware_identity"],
        "physical_fingerprint": authority["physical_fingerprint"],
    }


def validate_recording_level_campaign(
    *,
    repo_root: str | Path,
    campaign_receipt: str | Path,
    expected_sha256: str | None = None,
    now_utc: dt.datetime | None = None,
    require_fresh: bool = True,
) -> dict[str, Any]:
    """No-replace campaign receipt와 참조 meter/config를 독립 재검증한다."""

    root = _root(repo_root)
    receipt_before = _snapshot(
        root, campaign_receipt, label="recording level campaign receipt"
    )
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or _SHA256_RE.fullmatch(expected_sha256) is None
        or receipt_before.sha256 != expected_sha256
    ):
        raise RecordingLevelCampaignError("campaign receipt 외부 SHA-256이 다릅니다")
    assert receipt_before.data is not None
    payload = _load_json_object(
        receipt_before.data, label="recording level campaign receipt"
    )
    expected_relative = campaign_receipt_relative_path(str(payload.get("campaign_id") or ""))
    if _relative(root, receipt_before) != expected_relative:
        raise RecordingLevelCampaignError("campaign receipt가 campaign_id 고정 경로에 없습니다")
    now = _parse_utc(
        now_utc or dt.datetime.now(dt.timezone.utc), label="campaign validation now_utc"
    )
    summary = _validate_campaign_payload(
        payload,
        repo_root=root,
        now_utc=now,
        require_fresh=require_fresh,
    )
    receipt_after = _snapshot(
        root, receipt_before.path, label="recording level campaign receipt"
    )
    _require_unchanged(receipt_before, receipt_after, label="campaign receipt")
    summary.update(
        {
            "receipt_path": _relative(root, receipt_before),
            "receipt_size": receipt_before.size,
            "receipt_sha256": receipt_before.sha256,
        }
    )
    return summary


def _safe_mkdir(path: Path, *, root: Path) -> None:
    parent = path.parent
    try:
        reject_symlink_components(parent, root=root)
    except HoldoutContractError as exc:
        raise RecordingLevelCampaignError(str(exc)) from exc
    try:
        os.mkdir(path, 0o755)
    except FileExistsError:
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise RecordingLevelCampaignError(f"directory publish race: {path}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RecordingLevelCampaignError(f"campaign 경로가 안전한 directory가 아닙니다: {path}")
    try:
        reject_symlink_components(path, root=root)
    except HoldoutContractError as exc:
        raise RecordingLevelCampaignError(str(exc)) from exc


def issue_recording_level_campaign(
    *,
    repo_root: str | Path,
    meter_raw: str | Path,
    meter_receipt: str | Path,
    hardware_config: str | Path,
    now_utc: dt.datetime | None = None,
) -> dict[str, Any]:
    """Campaign payload를 고정 경로에 race-safe no-replace로 발행한다."""

    root = _root(repo_root)
    now = _parse_utc(
        now_utc or dt.datetime.now(dt.timezone.utc), label="campaign issued_at_utc"
    )
    payload = build_recording_level_campaign_payload(
        repo_root=root,
        meter_raw=meter_raw,
        meter_receipt=meter_receipt,
        hardware_config=hardware_config,
        now_utc=now,
    )
    # build 직후에 source를 한 번 더 읽어 build→publish 사이 변경을
    # invalid campaign receipt로 고정하지 않는다.
    _validate_campaign_payload(
        payload,
        repo_root=root,
        now_utc=now,
        require_fresh=True,
    )
    relative = campaign_receipt_relative_path(payload["campaign_id"])
    target = _lexical_repo_path(root, relative, label="campaign output")
    results = root / "results"
    try:
        reject_symlink_components(results, root=root)
    except HoldoutContractError as exc:
        raise RecordingLevelCampaignError(
            "campaign 발행 전에 안전한 results/ directory가 필요합니다"
        ) from exc
    campaigns = root / RECORDING_LEVEL_CAMPAIGN_ROOT
    _safe_mkdir(campaigns, root=root)
    _safe_mkdir(target.parent, root=root)
    try:
        reject_symlink_components(target, root=root, allow_missing_leaf=True)
    except HoldoutContractError as exc:
        raise RecordingLevelCampaignError(str(exc)) from exc
    raw = _canonical_json_bytes(payload) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".campaign.", suffix=".partial", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            atomic_publish_noreplace(temporary, target)
        except FileExistsError as exc:
            raise RecordingLevelCampaignError(
                f"campaign receipt는 기존 파일을 덮어쓰지 않습니다: {relative}"
            ) from exc
        except (OSError, ValueError) as exc:
            raise RecordingLevelCampaignError(
                f"campaign no-replace publish 경로가 검증 도중 변경됐습니다: {relative}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    receipt_sha = hashlib.sha256(raw).hexdigest()
    return validate_recording_level_campaign(
        repo_root=root,
        campaign_receipt=relative,
        expected_sha256=receipt_sha,
        now_utc=now,
        require_fresh=True,
    )


def rendered_source_level_evidence(samples: np.ndarray) -> dict[str, Any]:
    """Actual rendered 15초 source의 peak/RMS/trusted-band 증거를 계산한다."""

    values = np.asarray(samples)
    if values.ndim != 1 or values.shape != (RECORDING_LEVEL_SESSION_FRAMES,):
        raise RecordingLevelCampaignError(
            f"rendered source는 mono exact {RECORDING_LEVEL_SESSION_FRAMES} frames여야 합니다"
        )
    # 실제 callback에 공급되는 source.wav의 표준 표현은 float32다. level scalar만
    # 저장하면 서로 다른 파형이 같은 RMS/peak를 만들 수 있으므로, endian에 무관한
    # little-endian float32 sample bytes도 함께 봉인한다.
    canonical_samples = np.ascontiguousarray(values, dtype="<f4")
    values = np.asarray(canonical_samples, dtype=np.float64)
    if not bool(np.isfinite(values).all()):
        raise RecordingLevelCampaignError("rendered source에 non-finite sample이 있습니다")
    peak = float(np.max(np.abs(values)))
    if peak <= 0.0:
        raise RecordingLevelCampaignError("rendered source가 무음입니다")
    floor = np.finfo(np.float64).tiny
    peak_dbfs = float(20.0 * math.log10(max(peak, floor)))
    rms_dbfs = float(
        20.0 * math.log10(math.sqrt(float(np.mean(np.square(values)))) + floor)
    )
    trusted = float(band_rms_dbfs(values))
    evidence = {
        "schema": RECORDING_LEVEL_RENDERED_SOURCE_SCHEMA,
        "metric_definition": "measurement_level.band_rms_dbfs_hann_v1",
        "sample_rate": RECORDING_LEVEL_SAMPLE_RATE,
        "frames": RECORDING_LEVEL_SESSION_FRAMES,
        "sample_encoding": "float32_le",
        "sample_sha256": hashlib.sha256(canonical_samples.tobytes()).hexdigest(),
        "playback_amplitude": RECORDING_LEVEL_PLAYBACK_AMPLITUDE,
        "peak_linear": peak,
        "peak_dbfs": peak_dbfs,
        "rms_dbfs": rms_dbfs,
        "trusted_band_hz": list(OFFICIAL_MEASUREMENT_LEVEL.meter_band_hz),
        "trusted_band_rms_dbfs": trusted,
    }
    return validate_rendered_source_level(evidence)


def validate_rendered_source_level(value: Any) -> dict[str, Any]:
    """Session이 보존할 rendered source level mapping을 fail-closed 검증한다."""

    required = {
        "schema",
        "metric_definition",
        "sample_rate",
        "frames",
        "sample_encoding",
        "sample_sha256",
        "playback_amplitude",
        "peak_linear",
        "peak_dbfs",
        "rms_dbfs",
        "trusted_band_hz",
        "trusted_band_rms_dbfs",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise RecordingLevelCampaignError("rendered source level 필드 집합이 다릅니다")
    peak = _finite_number(value.get("peak_linear"), label="rendered peak_linear")
    peak_dbfs = _finite_number(value.get("peak_dbfs"), label="rendered peak_dbfs")
    rms_dbfs = _finite_number(value.get("rms_dbfs"), label="rendered rms_dbfs")
    _finite_number(
        value.get("trusted_band_rms_dbfs"), label="rendered trusted_band_rms_dbfs"
    )
    amplitude = _finite_number(
        value.get("playback_amplitude"), label="rendered playback_amplitude"
    )
    sample_sha256 = value.get("sample_sha256")
    if (
        value.get("schema") != RECORDING_LEVEL_RENDERED_SOURCE_SCHEMA
        or value.get("metric_definition")
        != "measurement_level.band_rms_dbfs_hann_v1"
        or value.get("sample_rate") != RECORDING_LEVEL_SAMPLE_RATE
        or value.get("frames") != RECORDING_LEVEL_SESSION_FRAMES
        or value.get("sample_encoding") != "float32_le"
        or not isinstance(sample_sha256, str)
        or _SHA256_RE.fullmatch(sample_sha256) is None
        or value.get("trusted_band_hz")
        != list(OFFICIAL_MEASUREMENT_LEVEL.meter_band_hz)
        or not math.isclose(
            amplitude,
            RECORDING_LEVEL_PLAYBACK_AMPLITUDE,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or peak <= 0.0
        or peak > RECORDING_LEVEL_PLAYBACK_AMPLITUDE + 1e-6
        or not math.isclose(
            peak_dbfs,
            20.0 * math.log10(peak),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or rms_dbfs > peak_dbfs + 1e-9
    ):
        raise RecordingLevelCampaignError("rendered source level 절대/peak 계약 위반")
    return json.loads(_canonical_json_bytes(dict(value)))


def build_recording_level_session_binding(
    campaign: Mapping[str, Any],
    *,
    session_started_at_utc: str | dt.datetime,
    same_amplifier_setting: bool,
    rendered_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Validated campaign을 session.json에 넣을 작은 binding으로 만든다."""

    if not isinstance(campaign, Mapping) or not isinstance(campaign.get("payload"), Mapping):
        raise RecordingLevelCampaignError(
            "validate_recording_level_campaign의 summary가 필요합니다"
        )
    payload = campaign["payload"]
    if campaign.get("campaign_id") != payload.get("campaign_id"):
        raise RecordingLevelCampaignError("campaign summary/payload ID가 다릅니다")
    receipt_ref = {
        "path": campaign.get("receipt_path"),
        "size": campaign.get("receipt_size"),
        "sha256": campaign.get("receipt_sha256"),
    }
    if (
        not isinstance(receipt_ref["path"], str)
        or isinstance(receipt_ref["size"], bool)
        or not isinstance(receipt_ref["size"], int)
        or receipt_ref["size"] <= 0
        or not isinstance(receipt_ref["sha256"], str)
        or _SHA256_RE.fullmatch(receipt_ref["sha256"]) is None
    ):
        raise RecordingLevelCampaignError("campaign receipt summary가 불완전합니다")
    started = _parse_utc(session_started_at_utc, label="session_started_at_utc")
    completed = _parse_utc(
        payload["meter"].get("completed_at_utc"), label="meter completed_at_utc"
    )
    issued = _parse_utc(
        payload.get("issued_at_utc"), label="campaign issued_at_utc"
    )
    age = float((started - completed).total_seconds())
    if age < 0.0 or age > RECORDING_LEVEL_MAX_AGE_SECONDS:
        raise RecordingLevelCampaignError(
            f"session 시작 meter age가 0..{RECORDING_LEVEL_MAX_AGE_SECONDS}초가 아닙니다: {age:.6f}"
        )
    if started < issued:
        raise RecordingLevelCampaignError(
            "session 시작은 recording-level campaign 발행 시각보다 빠를 수 없습니다: "
            f"started={started.isoformat()}, issued={issued.isoformat()}"
        )
    if same_amplifier_setting is not True:
        raise RecordingLevelCampaignError(
            "same_amplifier_setting은 exact true여야 합니다"
        )
    rendered = validate_rendered_source_level(rendered_source)
    binding: dict[str, Any] = {
        "schema": RECORDING_LEVEL_SESSION_BINDING_SCHEMA,
        "campaign_id": payload["campaign_id"],
        "campaign_receipt": receipt_ref,
        "meter_capture_id": payload["meter"]["capture_id"],
        "meter_completed_at_utc": completed.isoformat(),
        "session_started_at_utc": started.isoformat(),
        "meter_age_seconds_at_session_start": age,
        "same_amplifier_setting": True,
        "rendered_source": rendered,
    }
    binding["binding_sha256"] = _canonical_sha256(binding)
    return binding


def validate_recording_level_session_binding(
    campaign: Mapping[str, Any], binding: Any
) -> dict[str, Any]:
    """Stored session binding을 campaign/time/rendered-level에서 exact 재유도한다."""

    required = {
        "schema",
        "campaign_id",
        "campaign_receipt",
        "meter_capture_id",
        "meter_completed_at_utc",
        "session_started_at_utc",
        "meter_age_seconds_at_session_start",
        "same_amplifier_setting",
        "rendered_source",
        "binding_sha256",
    }
    if not isinstance(binding, Mapping) or set(binding) != required:
        raise RecordingLevelCampaignError("recording level session binding schema가 다릅니다")
    seal = binding.get("binding_sha256")
    if (
        not isinstance(seal, str)
        or _SHA256_RE.fullmatch(seal) is None
        or seal != _canonical_sha256(_without_seal(binding, field="binding_sha256"))
    ):
        raise RecordingLevelCampaignError("recording level session binding self-seal이 다릅니다")
    rebuilt = build_recording_level_session_binding(
        campaign,
        session_started_at_utc=binding.get("session_started_at_utc"),
        same_amplifier_setting=binding.get("same_amplifier_setting") is True,
        rendered_source=binding.get("rendered_source"),
    )
    if dict(binding) != rebuilt:
        raise RecordingLevelCampaignError(
            "recording level session binding이 campaign/time/level에서 재유도되지 않습니다"
        )
    return json.loads(_canonical_json_bytes(dict(binding)))


def rendered_source_level_evidence_v2(
    samples: np.ndarray, *, amplitude_millionths: int
) -> dict[str, Any]:
    """Source-gain v2가 선택한 integer amplitude의 exact rendered bytes를 봉인한다."""

    if (
        isinstance(amplitude_millionths, bool)
        or not isinstance(amplitude_millionths, int)
        or not 1
        <= amplitude_millionths
        <= PHYSICAL_SELECTOR_MAX_AMPLITUDE_MILLIONTHS
    ):
        raise RecordingLevelCampaignError("v2 amplitude_millionths 범위 위반")
    values = np.asarray(samples)
    if values.ndim != 1 or values.shape != (RECORDING_LEVEL_SESSION_FRAMES,):
        raise RecordingLevelCampaignError(
            f"v2 rendered source는 mono exact {RECORDING_LEVEL_SESSION_FRAMES} frames여야 합니다"
        )
    canonical = np.ascontiguousarray(values, dtype="<f4")
    numeric = np.asarray(canonical, dtype=np.float64)
    if not bool(np.isfinite(numeric).all()):
        raise RecordingLevelCampaignError("v2 rendered source에 non-finite sample이 있습니다")
    peak = float(np.max(np.abs(numeric)))
    amplitude = amplitude_millionths / 1_000_000.0
    if peak <= 0.0 or peak > amplitude + 1.0e-6:
        raise RecordingLevelCampaignError("v2 rendered source peak/amplitude 계약 위반")
    floor = np.finfo(np.float64).tiny
    evidence = {
        "schema": RECORDING_LEVEL_RENDERED_SOURCE_V2_SCHEMA,
        "metric_definition": "measurement_level.band_rms_dbfs_hann_v1",
        "sample_rate": RECORDING_LEVEL_SAMPLE_RATE,
        "frames": RECORDING_LEVEL_SESSION_FRAMES,
        "sample_encoding": "float32_le",
        "sample_sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
        "playback_amplitude_millionths": amplitude_millionths,
        "playback_amplitude": amplitude,
        "peak_linear": peak,
        "peak_dbfs": float(20.0 * math.log10(max(peak, floor))),
        "rms_dbfs": float(
            20.0
            * math.log10(
                math.sqrt(float(np.mean(np.square(numeric)))) + floor
            )
        ),
        "trusted_band_hz": list(OFFICIAL_MEASUREMENT_LEVEL.meter_band_hz),
        "trusted_band_rms_dbfs": float(band_rms_dbfs(numeric)),
    }
    return validate_rendered_source_level_v2(evidence)


def validate_rendered_source_level_v2(value: Any) -> dict[str, Any]:
    required = {
        "schema",
        "metric_definition",
        "sample_rate",
        "frames",
        "sample_encoding",
        "sample_sha256",
        "playback_amplitude_millionths",
        "playback_amplitude",
        "peak_linear",
        "peak_dbfs",
        "rms_dbfs",
        "trusted_band_hz",
        "trusted_band_rms_dbfs",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise RecordingLevelCampaignError("v2 rendered source field 집합 불일치")
    millionths = value.get("playback_amplitude_millionths")
    if (
        isinstance(millionths, bool)
        or not isinstance(millionths, int)
        or not 1 <= millionths <= PHYSICAL_SELECTOR_MAX_AMPLITUDE_MILLIONTHS
    ):
        raise RecordingLevelCampaignError("v2 rendered source amplitude integer 위반")
    amplitude = _finite_number(value.get("playback_amplitude"), label="v2 amplitude")
    peak = _finite_number(value.get("peak_linear"), label="v2 peak")
    peak_dbfs = _finite_number(value.get("peak_dbfs"), label="v2 peak dBFS")
    rms_dbfs = _finite_number(value.get("rms_dbfs"), label="v2 RMS dBFS")
    _finite_number(value.get("trusted_band_rms_dbfs"), label="v2 trusted RMS")
    sample_sha = value.get("sample_sha256")
    if (
        value.get("schema") != RECORDING_LEVEL_RENDERED_SOURCE_V2_SCHEMA
        or value.get("metric_definition")
        != "measurement_level.band_rms_dbfs_hann_v1"
        or value.get("sample_rate") != RECORDING_LEVEL_SAMPLE_RATE
        or value.get("frames") != RECORDING_LEVEL_SESSION_FRAMES
        or value.get("sample_encoding") != "float32_le"
        or not isinstance(sample_sha, str)
        or _SHA256_RE.fullmatch(sample_sha) is None
        or value.get("trusted_band_hz")
        != list(OFFICIAL_MEASUREMENT_LEVEL.meter_band_hz)
        or not math.isclose(
            amplitude, millionths / 1_000_000.0, rel_tol=0.0, abs_tol=1e-15
        )
        or peak <= 0.0
        or peak > amplitude + 1.0e-6
        or not math.isclose(
            peak_dbfs, 20.0 * math.log10(peak), rel_tol=1e-12, abs_tol=1e-12
        )
        or rms_dbfs > peak_dbfs + 1.0e-9
    ):
        raise RecordingLevelCampaignError("v2 rendered source scalar/peak 계약 위반")
    return json.loads(_canonical_json_bytes(dict(value)))


def build_recording_level_source_gain_session_binding(
    campaign: Mapping[str, Any],
    source_gain_summary: Mapping[str, Any],
    *,
    session_started_at_utc: str | dt.datetime,
    same_amplifier_setting: bool,
    source_gain_binding: Mapping[str, Any],
    rendered_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Fresh meter knob authority와 v2 per-row gain authority를 합성한다."""

    if not isinstance(campaign, Mapping) or not isinstance(campaign.get("payload"), Mapping):
        raise RecordingLevelCampaignError("validated level campaign summary가 필요합니다")
    payload = campaign["payload"]
    if campaign.get("campaign_id") != payload.get("campaign_id"):
        raise RecordingLevelCampaignError("campaign summary/payload ID 불일치")
    try:
        gain_binding = validate_recording_source_gain_session_binding(
            source_gain_summary, source_gain_binding
        )
    except ValueError as exc:
        raise RecordingLevelCampaignError(f"source gain binding 검증 실패: {exc}") from exc
    rendered = validate_rendered_source_level_v2(rendered_source)
    if (
        rendered["playback_amplitude_millionths"]
        != gain_binding["amplitude_millionths"]
        or rendered["sample_sha256"] != gain_binding["render_sample_sha256"]
    ):
        raise RecordingLevelCampaignError(
            "rendered source가 source-gain row amplitude/sample SHA와 다릅니다"
        )
    started = _parse_utc(session_started_at_utc, label="session_started_at_utc")
    completed = _parse_utc(
        payload["meter"].get("completed_at_utc"), label="meter completed_at_utc"
    )
    issued = _parse_utc(payload.get("issued_at_utc"), label="campaign issued_at_utc")
    age = float((started - completed).total_seconds())
    if age < 0.0 or age > RECORDING_LEVEL_MAX_AGE_SECONDS:
        raise RecordingLevelCampaignError(
            f"v2 session meter age가 0..{RECORDING_LEVEL_MAX_AGE_SECONDS}초가 아닙니다: {age:.6f}"
        )
    if started < issued:
        raise RecordingLevelCampaignError("v2 session 시작이 campaign 발행보다 빠릅니다")
    if same_amplifier_setting is not True:
        raise RecordingLevelCampaignError("same_amplifier_setting은 exact true여야 합니다")
    receipt_ref = {
        "path": campaign.get("receipt_path"),
        "size": campaign.get("receipt_size"),
        "sha256": campaign.get("receipt_sha256"),
    }
    binding: dict[str, Any] = {
        "schema": RECORDING_LEVEL_SOURCE_GAIN_SESSION_BINDING_SCHEMA,
        "campaign_id": payload["campaign_id"],
        "campaign_receipt": receipt_ref,
        "meter_capture_id": payload["meter"]["capture_id"],
        "meter_completed_at_utc": completed.isoformat(),
        "session_started_at_utc": started.isoformat(),
        "meter_age_seconds_at_session_start": age,
        "same_amplifier_setting": True,
        "source_gain": gain_binding,
        "rendered_source": rendered,
    }
    binding["binding_sha256"] = _canonical_sha256(binding)
    return binding


def validate_recording_level_source_gain_session_binding(
    campaign: Mapping[str, Any],
    source_gain_summary: Mapping[str, Any],
    binding: Any,
) -> dict[str, Any]:
    if not isinstance(binding, Mapping):
        raise RecordingLevelCampaignError("v2 level/source-gain binding이 mapping이 아닙니다")
    seal = binding.get("binding_sha256")
    unsealed = dict(binding)
    unsealed.pop("binding_sha256", None)
    if (
        binding.get("schema") != RECORDING_LEVEL_SOURCE_GAIN_SESSION_BINDING_SCHEMA
        or not isinstance(seal, str)
        or _SHA256_RE.fullmatch(seal) is None
        or seal != _canonical_sha256(unsealed)
    ):
        raise RecordingLevelCampaignError("v2 level/source-gain binding seal/schema 불일치")
    rebuilt = build_recording_level_source_gain_session_binding(
        campaign,
        source_gain_summary,
        session_started_at_utc=binding.get("session_started_at_utc"),
        same_amplifier_setting=binding.get("same_amplifier_setting") is True,
        source_gain_binding=binding.get("source_gain"),
        rendered_source=binding.get("rendered_source"),
    )
    if dict(binding) != rebuilt:
        raise RecordingLevelCampaignError("v2 level/source-gain binding 독립 재유도 불일치")
    return json.loads(_canonical_json_bytes(dict(binding)))


__all__ = [
    "RECORDING_LEVEL_CAMPAIGN_FILENAME",
    "RECORDING_LEVEL_CAMPAIGN_ROOT",
    "RECORDING_LEVEL_CAMPAIGN_SCHEMA",
    "RECORDING_LEVEL_MAX_AGE_SECONDS",
    "RECORDING_LEVEL_PLAYBACK_AMPLITUDE",
    "RECORDING_LEVEL_RENDERED_SOURCE_SCHEMA",
    "RECORDING_LEVEL_RENDERED_SOURCE_V2_SCHEMA",
    "RECORDING_LEVEL_SESSION_BINDING_SCHEMA",
    "RECORDING_LEVEL_SOURCE_GAIN_SESSION_BINDING_SCHEMA",
    "RecordingLevelCampaignError",
    "build_recording_level_campaign_payload",
    "build_recording_level_session_binding",
    "build_recording_level_source_gain_session_binding",
    "campaign_receipt_relative_path",
    "issue_recording_level_campaign",
    "rendered_source_level_evidence",
    "rendered_source_level_evidence_v2",
    "validate_recording_level_campaign",
    "validate_recording_level_session_binding",
    "validate_recording_level_source_gain_session_binding",
    "validate_rendered_source_level",
    "validate_rendered_source_level_v2",
]
