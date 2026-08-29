"""full-octave v3 물리 OFF/DL/FxLMS matched campaign의 읽기 전용 경계.

이 모듈은 실제 capture, 출력, ANC 계산 또는 attenuation 계산을 하지 않는다.  미래의
8-input raw-first bundle이 만들어진 **뒤**에만 그 bundle들이 다음 조건을 같은 물리
비교 단위에서 공유했는지를 검증한다.

* submitted source PCM bytes와 source manifest
* level/SPL/gain evidence, P/S operator, timing/lead, geometry, limiter, hardware
* predeclared OFF / DL / FxLMS 순서와 독립 source group
* one-shot test lifecycle와 각 raw bundle의 immutable SHA

이 모듈이 읽을 수 있는 것은 선언된 JSON과 raw bytes의 SHA 구조뿐이다.  capture adapter의
O_EXCL event history, 실제 장치/stream, P/S·lead, level/window/limiter, lineage 또는
native→canonical 변환을 외부에서 증명하지 못하는 동안에는 **어떤 non-fixture JSON도
physical matched authority가 될 수 없다.** 따라서 self-attested plan/receipt/raw가 모두
있더라도 결과는 ``BLOCKED_UNATTESTED_PHYSICAL_PROVENANCE``이며, 실제 감쇠·FxLMS 우위·
quiet zone·학습·배포 PASS가 아니다.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from ..dsp.control_band_contract import BroadbandFullOctaveContractV3
from .full_octave_v3_physical_bundle import (
    ALLOWED_TOPOLOGIES,
    BLOCK_SIZE,
    SAMPLE_RATE_HZ,
    UNATTESTED_STRUCTURAL_RAW_STATUS,
    load_full_octave_v3_physical_session_bundle,
)


CONFIG_SCHEMA = "full_octave_v3_matched_campaign_config_v1"
PLAN_SCHEMA = "full_octave_v3_matched_campaign_plan_v1"
RECEIPT_SCHEMA = "full_octave_v3_matched_campaign_receipt_v1"
SESSION_RUN_RECEIPT_SCHEMA = "full_octave_v3_matched_session_run_receipt_v1"
REPORT_SCHEMA = "full_octave_v3_matched_campaign_report_v1"
ROLE = "physical_matched_off_dl_fxlms_metadata_only_no_audio"
DEFAULT_CONFIG_RELATIVE_PATH = "configs/full_octave_v3_matched_campaign.yaml"

CONDITIONS = ("OFF", "DL", "FxLMS")
CONDITION_CONTROLLER_MODE = {
    "OFF": "anc_off_reference",
    "DL": "deep_anc_open_loop",
    "FxLMS": "fxlms_reference",
}
COUNTERBALANCE_SCHEME = "cyclic_latin_square_3_condition_v1"
COUNTERBALANCED_ORDERS = (
    ("OFF", "DL", "FxLMS"),
    ("DL", "FxLMS", "OFF"),
    ("FxLMS", "OFF", "DL"),
)
MIN_INDEPENDENT_GROUPS_PER_FAMILY = 4

# 이 checker는 read-only filesystem snapshot만 볼 수 있다. 다음 다섯 항목은 sidecar에
# SHA를 적는 것만으로는 독립적으로 증명할 수 없으므로, 실제 capture adapter/lineage
# authority가 구현되기 전에는 non-fixture artifact도 반드시 이 상태로 남는다.
UNATTESTED_PHYSICAL_PROVENANCE_STATUS = "BLOCKED_UNATTESTED_PHYSICAL_PROVENANCE"
UNATTESTED_PHYSICAL_PROVENANCE_BLOCKERS = (
    "typed_primary_secondary_operator_raw_analysis_validators_and_exact_timing_crosslinks",
    "typed_raw_analysis_electrical_witness_validator",
    "actual_submitted_pcm_callback_telemetry_and_native_canonical_recipe_equality",
    "capture_adapter_o_excl_event_binding_plan_sha_nonce_adapter_build_device_session_monotonic",
    "canonical_lineage_derived_independent_groups",
    "stage_specific_training_schema_with_canonical_finetune_init_checkpoint_contract_and_recorded_selection",
    "independent_off_dl_fxlms_raw_metric_and_five_err_quiet_zone_evaluator",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONFIG_KEYS = frozenset({"schema", "role", "control_band_contract", "artifacts"})
_CONFIG_CONTRACT_KEYS = frozenset({"id", "sha256", "sample_rate_hz", "block_size"})
_CONFIG_ARTIFACT_KEYS = frozenset({"campaign_plan", "campaign_receipt"})
_REF_KEYS = frozenset({"path", "sha256"})
_PLAN_KEYS = frozenset(
    {
        "schema",
        "role",
        "fixture_only",
        "control_band_contract_sha256",
        "campaign_identity",
        "counterbalance_scheme",
        "comparison_units",
        "campaign_capture_order",
        "one_shot",
        "plan_evidence_sha256",
    }
)
_CAMPAIGN_IDENTITY_KEYS = frozenset(
    {"level_gain", "plant_timing", "geometry", "window", "limiter", "hardware"}
)
_LEVEL_GAIN_KEYS = frozenset(
    {
        "measurement_level_evidence",
        "gain_contract",
        "meter_target_dbfs",
        "noise_playback_gain_db",
        "cancel_playback_gain_db",
    }
)
_PLANT_TIMING_KEYS = frozenset(
    {
        "plant_campaign_contract",
        "primary_path_operator",
        "secondary_path_operator",
        "training_timing_contract",
        "plant_delays",
        "handoff_samples",
        "lead_samples",
        "lead_derivation",
    }
)
_GEOMETRY_KEYS = frozenset({"routing_geometry"})
_WINDOW_KEYS = frozenset(
    {
        "window_contract",
        "warmup_samples",
        "analysis_start_sample",
        "analysis_stop_sample_exclusive",
    }
)
_LIMITER_KEYS = frozenset({"limiter_contract", "limiter_limit", "limiter_enabled"})
_HARDWARE_KEYS = frozenset(
    {"hardware_fingerprint", "acquisition_topology", "expected_bundle_topology"}
)
_ONE_SHOT_PLAN_KEYS = frozenset(
    {
        "test_once_required",
        "allow_session_append",
        "allow_session_replacement",
        "model_selection_locked_before_test",
        "model_selection_receipt",
        "campaign_nonce_sha256",
    }
)
_SOURCE_KEYS = frozenset({"source_family", "submitted_pcm", "source_manifest"})
_CONTROLLER_KEYS = frozenset({"controller_mode", "controller_artifact", "controller_config"})
_SESSION_PLAN_KEYS = frozenset(
    {
        "session_id",
        "condition",
        "order_index",
        "controller",
        "bundle_config_target",
        "comparison_run_receipt_target",
    }
)
_UNIT_KEYS = frozenset(
    {
        "comparison_unit_id",
        "independent_group_id",
        "source",
        "counterbalance_order",
        "sessions",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "role",
        "fixture_only",
        "campaign_plan_file_sha256",
        "campaign_plan_evidence_sha256",
        "control_band_contract_sha256",
        "one_shot",
        "session_receipts",
        "receipt_evidence_sha256",
    }
)
_ONE_SHOT_RECEIPT_KEYS = frozenset(
    {
        "test_once_completed",
        "no_session_append",
        "no_session_replacement",
        "model_selection_locked_before_test",
        "observed_capture_order",
    }
)
_SESSION_RECEIPT_KEYS = frozenset(
    {
        "comparison_unit_id",
        "session_id",
        "condition",
        "order_index",
        "bundle_config",
        "comparison_run_receipt",
    }
)
_SESSION_RUN_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "role",
        "fixture_only",
        "comparison_unit_id",
        "session_id",
        "condition",
        "source",
        "level_gain",
        "plant_timing",
        "geometry",
        "window",
        "limiter",
        "hardware",
        "bundle",
        "run_evidence_sha256",
    }
)
_RUN_SOURCE_KEYS = frozenset({"submitted_pcm_sha256", "source_manifest_sha256"})
_RUN_LEVEL_GAIN_KEYS = frozenset(
    {
        "measurement_level_evidence_sha256",
        "gain_contract_sha256",
        "meter_target_dbfs",
        "noise_playback_gain_db",
        "cancel_playback_gain_db",
    }
)
_RUN_PLANT_TIMING_KEYS = frozenset(
    {
        "plant_campaign_contract_sha256",
        "primary_path_operator_sha256",
        "secondary_path_operator_sha256",
        "training_timing_contract_sha256",
        "plant_delays_sha256",
        "handoff_samples",
        "lead_samples",
        "lead_derivation",
    }
)
_RUN_GEOMETRY_KEYS = frozenset({"routing_geometry_sha256"})
_RUN_WINDOW_KEYS = frozenset(
    {"window_contract_sha256", "warmup_samples", "analysis_start_sample", "analysis_stop_sample_exclusive"}
)
_RUN_LIMITER_KEYS = frozenset({"limiter_contract_sha256", "limiter_limit", "limiter_enabled"})
_RUN_HARDWARE_KEYS = frozenset(
    {"hardware_fingerprint_sha256", "acquisition_topology_sha256", "expected_bundle_topology"}
)
_RUN_BUNDLE_KEYS = frozenset(
    {
        "bundle_config_path",
        "bundle_config_sha256",
        "capture_plan_evidence_sha256",
        "session_sidecar_evidence_sha256",
        "native_raw_sha256",
        "canonical_raw_sha256",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value or "")
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{label}는 lowercase SHA-256이어야 합니다")
    return text


def _exact_mapping(value: object, *, expected: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{label} key 집합이 정확하지 않습니다: {actual}")
    return dict(value)


def _require_exact_bool(value: object, *, expected: bool, label: str) -> None:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{label}는 exact {expected!r}여야 합니다")


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label}는 0 이상 bool 아닌 int여야 합니다")
    return int(value)


def _require_finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}는 finite 숫자여야 합니다")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}는 finite 숫자여야 합니다")
    return result


def _require_nonempty_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}는 비어 있지 않은 string이어야 합니다")
    return value


def _validate_ref(value: object, *, label: str) -> dict[str, str]:
    ref = _exact_mapping(value, expected=_REF_KEYS, label=label)
    path = _require_nonempty_text(ref["path"], label=f"{label}.path")
    return {"path": path, "sha256": _require_sha256(ref["sha256"], label=f"{label}.sha256")}


def _inside_repository(
    root: Path,
    raw_path: object,
    *,
    label: str,
    results_only: bool = False,
) -> tuple[str, Path]:
    text = _require_nonempty_text(raw_path, label=f"{label}.path")
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}.path는 repository 내부 상대경로여야 합니다")
    if results_only and (not relative.parts or relative.parts[0] != "results"):
        raise ValueError(f"{label}.path는 raw-first results/ 내부 상대경로여야 합니다")
    target = root / relative
    cursor = root
    for part in relative.parts:
        cursor /= part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label}.path에 symlink가 있습니다: {cursor}")
    return relative.as_posix(), target


def _snapshot_regular_file(path: Path, *, label: str) -> tuple[bytes, str, int]:
    """O_NOFOLLOW snapshot으로 read-only receipt 검사를 고정한다."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError(f"{label}를 열 수 없습니다: {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label}는 regular file이어야 합니다: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, item) != getattr(after, item) for item in stable):
        raise ValueError(f"{label} snapshot 중 파일이 바뀌었습니다: {path}")
    if stat.S_ISLNK(path.lstat().st_mode):
        raise ValueError(f"{label} symlink는 허용하지 않습니다: {path}")
    content = b"".join(chunks)
    if len(content) != int(after.st_size):
        raise ValueError(f"{label} byte 수와 file size가 다릅니다: {path}")
    return content, _sha256_bytes(content), int(after.st_size)


def _snapshot_ref(
    root: Path,
    value: object,
    *,
    label: str,
    results_only: bool = False,
) -> dict[str, Any]:
    ref = _validate_ref(value, label=label)
    relative, target = _inside_repository(
        root, ref["path"], label=label, results_only=results_only
    )
    content, actual_sha, size = _snapshot_regular_file(target, label=label)
    if actual_sha != ref["sha256"]:
        raise ValueError(f"{label} bytes SHA가 receipt와 다릅니다")
    return {"path": relative, "sha256": actual_sha, "size_bytes": size, "content": content}


def _load_json_no_duplicates(content: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} JSON duplicate key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(content.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}는 UTF-8 JSON object여야 합니다") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON root는 object여야 합니다")
    return payload


def _self_sealed_payload(
    payload: Mapping[str, Any], *, evidence_key: str, label: str
) -> dict[str, Any]:
    evidence = _require_sha256(payload.get(evidence_key), label=f"{label}.{evidence_key}")
    unhashed = {key: value for key, value in payload.items() if key != evidence_key}
    if _sha256_bytes(_canonical_json_bytes(unhashed)) != evidence:
        raise ValueError(f"{label}.{evidence_key}가 canonical payload와 다릅니다")
    return dict(payload)


def _validate_source(value: object, *, label: str) -> dict[str, Any]:
    source = _exact_mapping(value, expected=_SOURCE_KEYS, label=label)
    canonical = BroadbandFullOctaveContractV3.canonical()
    if source["source_family"] not in canonical.source_families:
        raise ValueError(f"{label}.source_family가 canonical v3 family가 아닙니다")
    return {
        "source_family": str(source["source_family"]),
        "submitted_pcm": _validate_ref(source["submitted_pcm"], label=f"{label}.submitted_pcm"),
        "source_manifest": _validate_ref(source["source_manifest"], label=f"{label}.source_manifest"),
    }


def _validate_controller(value: object, *, condition: str, label: str) -> dict[str, Any]:
    controller = _exact_mapping(value, expected=_CONTROLLER_KEYS, label=label)
    expected_mode = CONDITION_CONTROLLER_MODE[condition]
    if controller["controller_mode"] != expected_mode:
        raise ValueError(
            f"{label}.controller_mode는 {condition}의 exact mode {expected_mode!r}이어야 합니다"
        )
    return {
        "controller_mode": expected_mode,
        "controller_artifact": _validate_ref(
            controller["controller_artifact"], label=f"{label}.controller_artifact"
        ),
        "controller_config": _validate_ref(
            controller["controller_config"], label=f"{label}.controller_config"
        ),
    }


def _validate_campaign_identity(value: object) -> dict[str, Any]:
    identity = _exact_mapping(
        value, expected=_CAMPAIGN_IDENTITY_KEYS, label="campaign_identity"
    )
    level_gain = _exact_mapping(
        identity["level_gain"], expected=_LEVEL_GAIN_KEYS, label="campaign_identity.level_gain"
    )
    meter_target = _require_finite_float(
        level_gain["meter_target_dbfs"], label="campaign_identity.level_gain.meter_target_dbfs"
    )
    if meter_target >= 0.0:
        raise ValueError("campaign_identity.level_gain.meter_target_dbfs는 0 dBFS 미만이어야 합니다")

    plant_timing = _exact_mapping(
        identity["plant_timing"],
        expected=_PLANT_TIMING_KEYS,
        label="campaign_identity.plant_timing",
    )
    handoff = _require_nonnegative_int(
        plant_timing["handoff_samples"],
        label="campaign_identity.plant_timing.handoff_samples",
    )
    if handoff != BLOCK_SIZE:
        raise ValueError(
            f"campaign_identity.plant_timing.handoff_samples는 exact {BLOCK_SIZE}이어야 합니다"
        )
    lead = _require_nonnegative_int(
        plant_timing["lead_samples"],
        label="campaign_identity.plant_timing.lead_samples",
    )
    if plant_timing["lead_derivation"] != "PlantDelays.lead()":
        raise ValueError("campaign_identity.plant_timing.lead_derivation은 PlantDelays.lead()이어야 합니다")

    geometry = _exact_mapping(
        identity["geometry"], expected=_GEOMETRY_KEYS, label="campaign_identity.geometry"
    )
    window = _exact_mapping(
        identity["window"], expected=_WINDOW_KEYS, label="campaign_identity.window"
    )
    warmup = _require_nonnegative_int(
        window["warmup_samples"], label="campaign_identity.window.warmup_samples"
    )
    start = _require_nonnegative_int(
        window["analysis_start_sample"], label="campaign_identity.window.analysis_start_sample"
    )
    stop = _require_nonnegative_int(
        window["analysis_stop_sample_exclusive"],
        label="campaign_identity.window.analysis_stop_sample_exclusive",
    )
    if start < warmup or stop <= start:
        raise ValueError("campaign_identity.window의 warmup/start/stop 순서가 유효하지 않습니다")
    if start % BLOCK_SIZE or stop % BLOCK_SIZE:
        raise ValueError("campaign_identity.window analysis 구간은 256-sample 경계여야 합니다")

    limiter = _exact_mapping(
        identity["limiter"], expected=_LIMITER_KEYS, label="campaign_identity.limiter"
    )
    limit = _require_finite_float(
        limiter["limiter_limit"], label="campaign_identity.limiter.limiter_limit"
    )
    if limit <= 0.0:
        raise ValueError("campaign_identity.limiter.limiter_limit는 양수여야 합니다")
    if type(limiter["limiter_enabled"]) is not bool:
        raise ValueError("campaign_identity.limiter.limiter_enabled는 explicit bool이어야 합니다")

    hardware = _exact_mapping(
        identity["hardware"], expected=_HARDWARE_KEYS, label="campaign_identity.hardware"
    )
    if hardware["expected_bundle_topology"] not in ALLOWED_TOPOLOGIES:
        raise ValueError("campaign_identity.hardware.expected_bundle_topology가 8-input topology가 아닙니다")

    return {
        "level_gain": {
            "measurement_level_evidence": _validate_ref(
                level_gain["measurement_level_evidence"],
                label="campaign_identity.level_gain.measurement_level_evidence",
            ),
            "gain_contract": _validate_ref(
                level_gain["gain_contract"], label="campaign_identity.level_gain.gain_contract"
            ),
            "meter_target_dbfs": meter_target,
            "noise_playback_gain_db": _require_finite_float(
                level_gain["noise_playback_gain_db"],
                label="campaign_identity.level_gain.noise_playback_gain_db",
            ),
            "cancel_playback_gain_db": _require_finite_float(
                level_gain["cancel_playback_gain_db"],
                label="campaign_identity.level_gain.cancel_playback_gain_db",
            ),
        },
        "plant_timing": {
            "plant_campaign_contract": _validate_ref(
                plant_timing["plant_campaign_contract"],
                label="campaign_identity.plant_timing.plant_campaign_contract",
            ),
            "primary_path_operator": _validate_ref(
                plant_timing["primary_path_operator"],
                label="campaign_identity.plant_timing.primary_path_operator",
            ),
            "secondary_path_operator": _validate_ref(
                plant_timing["secondary_path_operator"],
                label="campaign_identity.plant_timing.secondary_path_operator",
            ),
            "training_timing_contract": _validate_ref(
                plant_timing["training_timing_contract"],
                label="campaign_identity.plant_timing.training_timing_contract",
            ),
            "plant_delays": _validate_ref(
                plant_timing["plant_delays"], label="campaign_identity.plant_timing.plant_delays"
            ),
            "handoff_samples": handoff,
            "lead_samples": lead,
            "lead_derivation": "PlantDelays.lead()",
        },
        "geometry": {
            "routing_geometry": _validate_ref(
                geometry["routing_geometry"], label="campaign_identity.geometry.routing_geometry"
            )
        },
        "window": {
            "window_contract": _validate_ref(
                window["window_contract"], label="campaign_identity.window.window_contract"
            ),
            "warmup_samples": warmup,
            "analysis_start_sample": start,
            "analysis_stop_sample_exclusive": stop,
        },
        "limiter": {
            "limiter_contract": _validate_ref(
                limiter["limiter_contract"], label="campaign_identity.limiter.limiter_contract"
            ),
            "limiter_limit": limit,
            "limiter_enabled": bool(limiter["limiter_enabled"]),
        },
        "hardware": {
            "hardware_fingerprint": _validate_ref(
                hardware["hardware_fingerprint"], label="campaign_identity.hardware.hardware_fingerprint"
            ),
            "acquisition_topology": _validate_ref(
                hardware["acquisition_topology"], label="campaign_identity.hardware.acquisition_topology"
            ),
            "expected_bundle_topology": str(hardware["expected_bundle_topology"]),
        },
    }


def _validate_one_shot_plan(value: object) -> dict[str, Any]:
    one_shot = _exact_mapping(value, expected=_ONE_SHOT_PLAN_KEYS, label="one_shot")
    _require_exact_bool(
        one_shot["test_once_required"], expected=True, label="one_shot.test_once_required"
    )
    _require_exact_bool(
        one_shot["allow_session_append"], expected=False, label="one_shot.allow_session_append"
    )
    _require_exact_bool(
        one_shot["allow_session_replacement"],
        expected=False,
        label="one_shot.allow_session_replacement",
    )
    _require_exact_bool(
        one_shot["model_selection_locked_before_test"],
        expected=True,
        label="one_shot.model_selection_locked_before_test",
    )
    return {
        "test_once_required": True,
        "allow_session_append": False,
        "allow_session_replacement": False,
        "model_selection_locked_before_test": True,
        "model_selection_receipt": _validate_ref(
            one_shot["model_selection_receipt"], label="one_shot.model_selection_receipt"
        ),
        "campaign_nonce_sha256": _require_sha256(
            one_shot["campaign_nonce_sha256"], label="one_shot.campaign_nonce_sha256"
        ),
    }


def _validate_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    plan = _self_sealed_payload(payload, evidence_key="plan_evidence_sha256", label="campaign plan")
    plan = _exact_mapping(plan, expected=_PLAN_KEYS, label="campaign plan")
    if plan["schema"] != PLAN_SCHEMA or plan["role"] != ROLE:
        raise ValueError("campaign plan schema/role이 full-octave physical matched campaign과 다릅니다")
    if type(plan["fixture_only"]) is not bool:
        raise ValueError("campaign plan.fixture_only는 explicit bool이어야 합니다")
    canonical = BroadbandFullOctaveContractV3.canonical()
    if plan["control_band_contract_sha256"] != canonical.digest():
        raise ValueError("campaign plan control-band contract SHA가 canonical v3와 다릅니다")
    if plan["counterbalance_scheme"] != COUNTERBALANCE_SCHEME:
        raise ValueError("campaign plan은 3-condition cyclic counterbalance를 사용해야 합니다")

    identity = _validate_campaign_identity(plan["campaign_identity"])
    one_shot = _validate_one_shot_plan(plan["one_shot"])
    units_raw = plan["comparison_units"]
    if not isinstance(units_raw, list) or not units_raw:
        raise ValueError("campaign plan.comparison_units는 비어 있지 않은 list여야 합니다")

    units: list[dict[str, Any]] = []
    seen_unit_ids: set[str] = set()
    seen_groups: set[str] = set()
    seen_session_ids: set[str] = set()
    family_groups: dict[str, set[str]] = {family: set() for family in canonical.source_families}
    family_sources: dict[str, set[str]] = {family: set() for family in canonical.source_families}
    family_order_counts: dict[str, Counter[tuple[str, str, str]]] = {
        family: Counter() for family in canonical.source_families
    }
    flattened_session_ids: list[str] = []

    for unit_index, raw_unit in enumerate(units_raw):
        unit = _exact_mapping(raw_unit, expected=_UNIT_KEYS, label=f"comparison_units[{unit_index}]")
        unit_id = _require_nonempty_text(
            unit["comparison_unit_id"], label=f"comparison_units[{unit_index}].comparison_unit_id"
        )
        group_id = _require_nonempty_text(
            unit["independent_group_id"], label=f"comparison_units[{unit_index}].independent_group_id"
        )
        if unit_id in seen_unit_ids or group_id in seen_groups:
            raise ValueError("comparison unit/group은 campaign에서 중복될 수 없습니다")
        seen_unit_ids.add(unit_id)
        seen_groups.add(group_id)
        source = _validate_source(unit["source"], label=f"comparison_units[{unit_index}].source")
        family = source["source_family"]
        source_sha = source["submitted_pcm"]["sha256"]
        if source_sha in family_sources[family]:
            raise ValueError(
                "같은 family에서 submitted PCM bytes를 다른 independent group으로 재사용할 수 없습니다"
            )
        family_sources[family].add(source_sha)
        family_groups[family].add(group_id)

        order_raw = unit["counterbalance_order"]
        if not isinstance(order_raw, list) or len(order_raw) != len(CONDITIONS):
            raise ValueError(f"comparison_units[{unit_index}].counterbalance_order가 3 condition list가 아닙니다")
        order = tuple(str(item) for item in order_raw)
        if order not in COUNTERBALANCED_ORDERS:
            raise ValueError("counterbalance_order는 predeclared cyclic Latin-square order여야 합니다")
        family_order_counts[family][order] += 1

        sessions_raw = unit["sessions"]
        if not isinstance(sessions_raw, list) or len(sessions_raw) != len(CONDITIONS):
            raise ValueError(f"comparison_units[{unit_index}].sessions는 exact OFF/DL/FxLMS 3개여야 합니다")
        sessions: list[dict[str, Any]] = []
        for session_index, raw_session in enumerate(sessions_raw):
            session = _exact_mapping(
                raw_session,
                expected=_SESSION_PLAN_KEYS,
                label=f"comparison_units[{unit_index}].sessions[{session_index}]",
            )
            session_id = _require_nonempty_text(
                session["session_id"],
                label=f"comparison_units[{unit_index}].sessions[{session_index}].session_id",
            )
            if session_id in seen_session_ids:
                raise ValueError("session_id는 campaign에서 중복될 수 없습니다")
            seen_session_ids.add(session_id)
            condition = session["condition"]
            if condition != order[session_index]:
                raise ValueError("각 session condition은 predeclared counterbalance order와 같아야 합니다")
            if session["order_index"] != session_index:
                raise ValueError("session.order_index는 unit 내부 predeclared index와 같아야 합니다")
            target = _require_nonempty_text(
                session["bundle_config_target"],
                label=f"comparison_units[{unit_index}].sessions[{session_index}].bundle_config_target",
            )
            target_path = Path(target)
            if (
                target_path.is_absolute()
                or ".." in target_path.parts
                or not target_path.parts
                or target_path.parts[0] != "results"
                or target_path.suffix not in {".yaml", ".yml"}
            ):
                raise ValueError("bundle_config_target은 results/ 내부 YAML 상대경로여야 합니다")
            run_target = _require_nonempty_text(
                session["comparison_run_receipt_target"],
                label=(
                    f"comparison_units[{unit_index}].sessions[{session_index}]"
                    ".comparison_run_receipt_target"
                ),
            )
            run_target_path = Path(run_target)
            if (
                run_target_path.is_absolute()
                or ".." in run_target_path.parts
                or not run_target_path.parts
                or run_target_path.parts[0] != "results"
                or run_target_path.suffix != ".json"
            ):
                raise ValueError(
                    "comparison_run_receipt_target은 results/ 내부 JSON 상대경로여야 합니다"
                )
            if run_target_path.as_posix() == target_path.as_posix():
                raise ValueError("bundle config와 comparison run receipt target은 달라야 합니다")
            sessions.append(
                {
                    "session_id": session_id,
                    "condition": condition,
                    "order_index": session_index,
                    "controller": _validate_controller(
                        session["controller"],
                        condition=condition,
                        label=f"comparison_units[{unit_index}].sessions[{session_index}].controller",
                    ),
                    "bundle_config_target": target_path.as_posix(),
                    "comparison_run_receipt_target": run_target_path.as_posix(),
                }
            )
            flattened_session_ids.append(session_id)
        units.append(
            {
                "comparison_unit_id": unit_id,
                "independent_group_id": group_id,
                "source": source,
                "counterbalance_order": list(order),
                "sessions": sessions,
            }
        )

    if not isinstance(plan["campaign_capture_order"], list):
        raise ValueError("campaign_capture_order는 explicit list여야 합니다")
    capture_order = [
        _require_nonempty_text(value, label="campaign_capture_order item")
        for value in plan["campaign_capture_order"]
    ]
    if capture_order != flattened_session_ids:
        raise ValueError("campaign_capture_order는 unit/session predeclared order와 exact하게 같아야 합니다")
    if len(set(capture_order)) != len(capture_order):
        raise ValueError("campaign_capture_order에 duplicate session_id가 있습니다")

    for family in canonical.source_families:
        groups = family_groups[family]
        if len(groups) < MIN_INDEPENDENT_GROUPS_PER_FAMILY:
            raise ValueError(
                f"{family} independent group {len(groups)} < {MIN_INDEPENDENT_GROUPS_PER_FAMILY}"
            )
        counts = family_order_counts[family]
        if set(counts) != set(COUNTERBALANCED_ORDERS):
            raise ValueError(f"{family}에 세 counterbalance order가 모두 필요합니다")
        values = [counts[order] for order in COUNTERBALANCED_ORDERS]
        if max(values) - min(values) > 1:
            raise ValueError(f"{family} counterbalance order가 1 session보다 더 불균형합니다")

    return {
        "payload": plan,
        "fixture_only": bool(plan["fixture_only"]),
        "identity": identity,
        "one_shot": one_shot,
        "units": units,
        "capture_order": capture_order,
        "family_group_counts": {family: len(groups) for family, groups in family_groups.items()},
        "family_order_counts": {
            family: {"/".join(order): counts[order] for order in COUNTERBALANCED_ORDERS}
            for family, counts in family_order_counts.items()
        },
    }


def _validate_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _self_sealed_payload(
        payload, evidence_key="receipt_evidence_sha256", label="campaign receipt"
    )
    receipt = _exact_mapping(receipt, expected=_RECEIPT_KEYS, label="campaign receipt")
    if receipt["schema"] != RECEIPT_SCHEMA or receipt["role"] != ROLE:
        raise ValueError("campaign receipt schema/role이 full-octave physical matched campaign과 다릅니다")
    if type(receipt["fixture_only"]) is not bool:
        raise ValueError("campaign receipt.fixture_only는 explicit bool이어야 합니다")
    canonical = BroadbandFullOctaveContractV3.canonical()
    if receipt["control_band_contract_sha256"] != canonical.digest():
        raise ValueError("campaign receipt control-band contract SHA가 canonical v3와 다릅니다")
    plan_file_sha = _require_sha256(
        receipt["campaign_plan_file_sha256"], label="campaign_receipt.campaign_plan_file_sha256"
    )
    plan_evidence_sha = _require_sha256(
        receipt["campaign_plan_evidence_sha256"],
        label="campaign_receipt.campaign_plan_evidence_sha256",
    )
    one_shot = _exact_mapping(
        receipt["one_shot"], expected=_ONE_SHOT_RECEIPT_KEYS, label="campaign_receipt.one_shot"
    )
    for key in (
        "test_once_completed",
        "no_session_append",
        "no_session_replacement",
        "model_selection_locked_before_test",
    ):
        _require_exact_bool(one_shot[key], expected=True, label=f"campaign_receipt.one_shot.{key}")
    observed_raw = one_shot["observed_capture_order"]
    if not isinstance(observed_raw, list) or not observed_raw:
        raise ValueError("campaign_receipt.one_shot.observed_capture_order는 비어 있지 않은 list여야 합니다")
    observed = [
        _require_nonempty_text(value, label="campaign_receipt.one_shot.observed_capture_order item")
        for value in observed_raw
    ]
    if len(set(observed)) != len(observed):
        raise ValueError("campaign_receipt.one_shot.observed_capture_order에 duplicate가 있습니다")

    receipts_raw = receipt["session_receipts"]
    if not isinstance(receipts_raw, list) or not receipts_raw:
        raise ValueError("campaign_receipt.session_receipts는 비어 있지 않은 list여야 합니다")
    session_receipts: list[dict[str, Any]] = []
    seen_session_ids: set[str] = set()
    for index, raw_entry in enumerate(receipts_raw):
        entry = _exact_mapping(
            raw_entry, expected=_SESSION_RECEIPT_KEYS, label=f"campaign_receipt.session_receipts[{index}]"
        )
        session_id = _require_nonempty_text(
            entry["session_id"], label=f"campaign_receipt.session_receipts[{index}].session_id"
        )
        if session_id in seen_session_ids:
            raise ValueError("campaign_receipt.session_id는 중복될 수 없습니다")
        seen_session_ids.add(session_id)
        condition = entry["condition"]
        if condition not in CONDITIONS:
            raise ValueError("campaign_receipt session condition이 OFF/DL/FxLMS가 아닙니다")
        if entry["order_index"] not in range(len(CONDITIONS)):
            raise ValueError("campaign_receipt session.order_index가 유효하지 않습니다")
        session_receipts.append(
            {
                "comparison_unit_id": _require_nonempty_text(
                    entry["comparison_unit_id"],
                    label=f"campaign_receipt.session_receipts[{index}].comparison_unit_id",
                ),
                "session_id": session_id,
                "condition": condition,
                "order_index": int(entry["order_index"]),
                "bundle_config": _validate_ref(
                    entry["bundle_config"],
                    label=f"campaign_receipt.session_receipts[{index}].bundle_config",
                ),
                "comparison_run_receipt": _validate_ref(
                    entry["comparison_run_receipt"],
                    label=(
                        f"campaign_receipt.session_receipts[{index}]"
                        ".comparison_run_receipt"
                    ),
                ),
            }
        )
    return {
        "payload": receipt,
        "fixture_only": bool(receipt["fixture_only"]),
        "campaign_plan_file_sha256": plan_file_sha,
        "campaign_plan_evidence_sha256": plan_evidence_sha,
        "observed_capture_order": observed,
        "session_receipts": session_receipts,
    }


def _validate_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    config = _exact_mapping(payload, expected=_CONFIG_KEYS, label="matched campaign config")
    if config["schema"] != CONFIG_SCHEMA or config["role"] != ROLE:
        raise ValueError("matched campaign config schema/role이 다릅니다")
    contract = _exact_mapping(
        config["control_band_contract"],
        expected=_CONFIG_CONTRACT_KEYS,
        label="matched campaign config.control_band_contract",
    )
    canonical = BroadbandFullOctaveContractV3.canonical()
    expected_contract = {
        "id": canonical.contract_id,
        "sha256": canonical.digest(),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "block_size": BLOCK_SIZE,
    }
    if contract != expected_contract:
        raise ValueError("matched campaign config가 canonical full-octave v3 contract와 다릅니다")
    artifacts = _exact_mapping(
        config["artifacts"], expected=_CONFIG_ARTIFACT_KEYS, label="matched campaign config.artifacts"
    )
    intents: dict[str, dict[str, str] | None] = {}
    for name in _CONFIG_ARTIFACT_KEYS:
        entry = _exact_mapping(artifacts[name], expected=_REF_KEYS, label=f"artifacts.{name}")
        path, digest = entry["path"], entry["sha256"]
        if (path is None) != (digest is None):
            raise ValueError(f"artifacts.{name}.path와 sha256은 함께 null이거나 함께 있어야 합니다")
        intents[name] = None if path is None else _validate_ref(entry, label=f"artifacts.{name}")
    return {"payload": config, "intents": intents}


def _snapshot_identity_references(root: Path, identity: Mapping[str, Any], one_shot: Mapping[str, Any]) -> None:
    """plan이 선언한 shared evidence bytes를 모두 실제로 재확인한다."""

    references = (
        ("level_gain.measurement_level_evidence", identity["level_gain"]["measurement_level_evidence"]),
        ("level_gain.gain_contract", identity["level_gain"]["gain_contract"]),
        ("plant_timing.plant_campaign_contract", identity["plant_timing"]["plant_campaign_contract"]),
        ("plant_timing.primary_path_operator", identity["plant_timing"]["primary_path_operator"]),
        ("plant_timing.secondary_path_operator", identity["plant_timing"]["secondary_path_operator"]),
        ("plant_timing.training_timing_contract", identity["plant_timing"]["training_timing_contract"]),
        ("plant_timing.plant_delays", identity["plant_timing"]["plant_delays"]),
        ("geometry.routing_geometry", identity["geometry"]["routing_geometry"]),
        ("window.window_contract", identity["window"]["window_contract"]),
        ("limiter.limiter_contract", identity["limiter"]["limiter_contract"]),
        ("hardware.hardware_fingerprint", identity["hardware"]["hardware_fingerprint"]),
        ("hardware.acquisition_topology", identity["hardware"]["acquisition_topology"]),
        ("one_shot.model_selection_receipt", one_shot["model_selection_receipt"]),
    )
    for label, reference in references:
        _snapshot_ref(root, reference, label=label)


def _verify_bundle_against_expected(
    *,
    root: Path,
    bundle_ref: Mapping[str, str],
    expected_unit: Mapping[str, Any],
    expected_session: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """8-input raw bundle의 immutable identity를 matched plan과 직접 대조한다."""

    bundle_snapshot = _snapshot_ref(
        root, bundle_ref, label=f"bundle_config:{expected_session['session_id']}", results_only=True
    )
    if bundle_snapshot["path"] != expected_session["bundle_config_target"]:
        raise ValueError("receipt bundle config path가 predeclared campaign target과 다릅니다")
    try:
        bundle_report = load_full_octave_v3_physical_session_bundle(
            root / bundle_snapshot["path"], repository_root=root
        )
    except ValueError as exc:
        raise ValueError(f"{expected_session['session_id']} 8-input raw bundle 검증 실패: {exc}") from exc
    if (
        bundle_report.get("status") != UNATTESTED_STRUCTURAL_RAW_STATUS
        or bundle_report.get("declared_sha_structure_valid") is not True
        or bundle_report.get("physical_raw_provenance_attested") is not False
    ):
        raise ValueError(
            f"{expected_session['session_id']} 8-input raw bundle의 declared SHA structure를 "
            "read-only로 재검증하지 못했습니다"
        )
    if bundle_report.get("canonical_training_eligible") is not False:
        raise ValueError("8-input raw bundle이 canonical training eligible을 주장할 수 없습니다")
    capture_ref = bundle_report.get("capture_plan")
    if not isinstance(capture_ref, Mapping):
        raise ValueError("8-input raw bundle에 capture plan reference가 없습니다")
    capture_file_sha = capture_ref.get("file_sha256")
    capture_evidence_sha = capture_ref.get("evidence_sha256")
    capture_snapshot = _snapshot_ref(
        root,
        {"path": capture_ref.get("path"), "sha256": capture_file_sha},
        label="bundle.capture_plan",
        results_only=True,
    )
    capture_plan = _load_json_no_duplicates(capture_snapshot["content"], label="bundle capture plan")
    if capture_plan.get("plan_evidence_sha256") != capture_evidence_sha:
        raise ValueError("8-input raw bundle capture plan evidence SHA가 report와 다릅니다")
    capture = capture_plan.get("capture")
    raw_identity = capture_plan.get("identity")
    if not isinstance(capture, Mapping) or not isinstance(raw_identity, Mapping):
        raise ValueError("8-input raw bundle capture plan의 capture/identity가 없습니다")
    source = raw_identity.get("source")
    controller = raw_identity.get("controller")
    plant = raw_identity.get("plant")
    timing = raw_identity.get("timing")
    if not all(isinstance(value, Mapping) for value in (source, controller, plant, timing)):
        raise ValueError("8-input raw bundle identity가 source/controller/plant/timing 전체를 갖지 않습니다")

    expected_source = expected_unit["source"]
    if source.get("submitted_pcm_sha256") != expected_source["submitted_pcm"]["sha256"]:
        raise ValueError("8-input raw bundle submitted PCM SHA가 matched unit과 다릅니다")
    if source.get("source_manifest_sha256") != expected_source["source_manifest"]["sha256"]:
        raise ValueError("8-input raw bundle source manifest SHA가 matched unit과 다릅니다")

    expected_controller = expected_session["controller"]
    if controller.get("controller_mode") != expected_controller["controller_mode"]:
        raise ValueError("8-input raw bundle controller mode가 matched condition과 다릅니다")
    if controller.get("controller_artifact_sha256") != expected_controller["controller_artifact"]["sha256"]:
        raise ValueError("8-input raw bundle controller artifact SHA가 matched plan과 다릅니다")
    if controller.get("controller_config_sha256") != expected_controller["controller_config"]["sha256"]:
        raise ValueError("8-input raw bundle controller config SHA가 matched plan과 다릅니다")

    plant_timing = identity["plant_timing"]
    if plant.get("plant_campaign_contract_sha256") != plant_timing["plant_campaign_contract"]["sha256"]:
        raise ValueError("8-input raw bundle P/S campaign SHA가 matched plan과 다릅니다")
    if plant.get("hardware_fingerprint_sha256") != identity["hardware"]["hardware_fingerprint"]["sha256"]:
        raise ValueError("8-input raw bundle hardware fingerprint SHA가 matched plan과 다릅니다")
    if plant.get("routing_geometry_sha256") != identity["geometry"]["routing_geometry"]["sha256"]:
        raise ValueError("8-input raw bundle routing/geometry SHA가 matched plan과 다릅니다")
    if timing.get("training_timing_contract_sha256") != plant_timing["training_timing_contract"]["sha256"]:
        raise ValueError("8-input raw bundle timing contract SHA가 matched plan과 다릅니다")
    if timing.get("plant_delays_sha256") != plant_timing["plant_delays"]["sha256"]:
        raise ValueError("8-input raw bundle PlantDelays SHA가 matched plan과 다릅니다")
    if timing.get("handoff_samples") != plant_timing["handoff_samples"]:
        raise ValueError("8-input raw bundle handoff samples가 matched plan과 다릅니다")
    if timing.get("lead_samples") != plant_timing["lead_samples"]:
        raise ValueError("8-input raw bundle lead samples가 matched plan과 다릅니다")
    if timing.get("lead_derivation") != plant_timing["lead_derivation"]:
        raise ValueError("8-input raw bundle lead derivation이 matched plan과 다릅니다")
    if capture.get("topology") != identity["hardware"]["expected_bundle_topology"]:
        raise ValueError("8-input raw bundle topology가 matched hardware plan과 다릅니다")
    frames = bundle_report.get("frames")
    if isinstance(frames, bool) or not isinstance(frames, int):
        raise ValueError("8-input raw bundle frame count가 없습니다")
    if frames < identity["window"]["analysis_stop_sample_exclusive"]:
        raise ValueError("8-input raw bundle이 predeclared analysis window 끝까지 보존하지 않았습니다")
    return {
        "bundle_config": {"path": bundle_snapshot["path"], "sha256": bundle_snapshot["sha256"]},
        "capture_plan": {
            "path": capture_snapshot["path"],
            "sha256": capture_snapshot["sha256"],
            "evidence_sha256": bundle_report["capture_plan"]["evidence_sha256"],
        },
        "session_sidecar": dict(bundle_report["session_sidecar"]),
        "native_raw": dict(bundle_report["native_raw"]),
        "canonical_raw": dict(bundle_report["canonical_raw"]),
        "frames": frames,
    }


def _expected_run_identity(
    *,
    expected_unit: Mapping[str, Any],
    expected_session: Mapping[str, Any],
    identity: Mapping[str, Any],
    bundle_record: Mapping[str, Any],
) -> dict[str, Any]:
    """plan/bundle에서 한 session의 exact run-receipt payload를 유도한다."""

    source = expected_unit["source"]
    level_gain = identity["level_gain"]
    plant_timing = identity["plant_timing"]
    window = identity["window"]
    limiter = identity["limiter"]
    hardware = identity["hardware"]
    return {
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
        "geometry": {
            "routing_geometry_sha256": identity["geometry"]["routing_geometry"]["sha256"]
        },
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
            "bundle_config_path": bundle_record["bundle_config"]["path"],
            "bundle_config_sha256": bundle_record["bundle_config"]["sha256"],
            "capture_plan_evidence_sha256": bundle_record["capture_plan"]["evidence_sha256"],
            "session_sidecar_evidence_sha256": bundle_record["session_sidecar"]["evidence_sha256"],
            "native_raw_sha256": bundle_record["native_raw"]["sha256"],
            "canonical_raw_sha256": bundle_record["canonical_raw"]["sha256"],
        },
    }


def _validate_run_block(
    value: object,
    *,
    expected: Mapping[str, Any],
    keys: frozenset[str],
    label: str,
) -> dict[str, Any]:
    actual = _exact_mapping(value, expected=keys, label=label)
    # SHA fields are validated independently before equality so a malformed value
    # cannot look like an ordinary mismatch in a forensic report.
    for key, item in actual.items():
        if key.endswith("_sha256"):
            _require_sha256(item, label=f"{label}.{key}")
    for key in ("meter_target_dbfs", "noise_playback_gain_db", "cancel_playback_gain_db", "limiter_limit"):
        if key in actual:
            _require_finite_float(actual[key], label=f"{label}.{key}")
    for key in (
        "handoff_samples",
        "lead_samples",
        "warmup_samples",
        "analysis_start_sample",
        "analysis_stop_sample_exclusive",
    ):
        if key in actual:
            _require_nonnegative_int(actual[key], label=f"{label}.{key}")
    if "limiter_enabled" in actual and type(actual["limiter_enabled"]) is not bool:
        raise ValueError(f"{label}.limiter_enabled는 explicit bool이어야 합니다")
    if actual != expected:
        raise ValueError(f"{label}가 matched campaign shared identity와 다릅니다")
    return actual


def _validate_session_run_receipt(
    payload: Mapping[str, Any],
    *,
    expected_unit: Mapping[str, Any],
    expected_session: Mapping[str, Any],
    identity: Mapping[str, Any],
    bundle_record: Mapping[str, Any],
) -> dict[str, Any]:
    run = _self_sealed_payload(
        payload, evidence_key="run_evidence_sha256", label="comparison run receipt"
    )
    run = _exact_mapping(run, expected=_SESSION_RUN_RECEIPT_KEYS, label="comparison run receipt")
    if run["schema"] != SESSION_RUN_RECEIPT_SCHEMA or run["role"] != ROLE:
        raise ValueError("comparison run receipt schema/role이 matched campaign과 다릅니다")
    if type(run["fixture_only"]) is not bool:
        raise ValueError("comparison run receipt.fixture_only는 explicit bool이어야 합니다")
    if run["comparison_unit_id"] != expected_unit["comparison_unit_id"]:
        raise ValueError("comparison run receipt comparison_unit_id가 plan과 다릅니다")
    if run["session_id"] != expected_session["session_id"]:
        raise ValueError("comparison run receipt session_id가 plan과 다릅니다")
    if run["condition"] != expected_session["condition"]:
        raise ValueError("comparison run receipt condition이 plan과 다릅니다")
    expected = _expected_run_identity(
        expected_unit=expected_unit,
        expected_session=expected_session,
        identity=identity,
        bundle_record=bundle_record,
    )
    _validate_run_block(run["source"], expected=expected["source"], keys=_RUN_SOURCE_KEYS, label="run.source")
    _validate_run_block(
        run["level_gain"], expected=expected["level_gain"], keys=_RUN_LEVEL_GAIN_KEYS, label="run.level_gain"
    )
    _validate_run_block(
        run["plant_timing"],
        expected=expected["plant_timing"],
        keys=_RUN_PLANT_TIMING_KEYS,
        label="run.plant_timing",
    )
    _validate_run_block(
        run["geometry"], expected=expected["geometry"], keys=_RUN_GEOMETRY_KEYS, label="run.geometry"
    )
    _validate_run_block(run["window"], expected=expected["window"], keys=_RUN_WINDOW_KEYS, label="run.window")
    _validate_run_block(
        run["limiter"], expected=expected["limiter"], keys=_RUN_LIMITER_KEYS, label="run.limiter"
    )
    _validate_run_block(
        run["hardware"], expected=expected["hardware"], keys=_RUN_HARDWARE_KEYS, label="run.hardware"
    )
    _validate_run_block(run["bundle"], expected=expected["bundle"], keys=_RUN_BUNDLE_KEYS, label="run.bundle")
    return {"payload": run, "fixture_only": bool(run["fixture_only"])}


def _verify_session_run_receipt(
    *,
    root: Path,
    run_ref: Mapping[str, str],
    expected_unit: Mapping[str, Any],
    expected_session: Mapping[str, Any],
    identity: Mapping[str, Any],
    bundle_record: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = _snapshot_ref(
        root,
        run_ref,
        label=f"comparison_run_receipt:{expected_session['session_id']}",
        results_only=True,
    )
    if snapshot["path"] != expected_session["comparison_run_receipt_target"]:
        raise ValueError("comparison run receipt path가 predeclared campaign target과 다릅니다")
    run = _validate_session_run_receipt(
        _load_json_no_duplicates(snapshot["content"], label="comparison run receipt"),
        expected_unit=expected_unit,
        expected_session=expected_session,
        identity=identity,
        bundle_record=bundle_record,
    )
    if run["fixture_only"]:
        raise ValueError("fixture-only comparison run receipt는 physical matched authority가 아닙니다")
    return {
        "path": snapshot["path"],
        "file_sha256": snapshot["sha256"],
        "evidence_sha256": run["payload"]["run_evidence_sha256"],
    }


def build_full_octave_v3_matched_campaign_plan(
    *,
    campaign_identity: Mapping[str, Any],
    comparison_units: Sequence[Mapping[str, Any]],
    one_shot: Mapping[str, Any],
) -> dict[str, Any]:
    """future campaign의 pure-data pre-capture plan을 만든다.

    파일을 쓰거나 장치를 열지 않는다. 호출자는 반환된 JSON을 ``O_EXCL``로 먼저
    발행한 뒤에만 실제 8-input capture를 시작해야 한다.
    """

    units = [dict(item) for item in comparison_units]
    capture_order: list[str] = []
    for unit in units:
        sessions = unit.get("sessions")
        if isinstance(sessions, Sequence) and not isinstance(sessions, (str, bytes)):
            for session in sessions:
                if isinstance(session, Mapping) and isinstance(session.get("session_id"), str):
                    capture_order.append(session["session_id"])
    payload: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "role": ROLE,
        "fixture_only": False,
        "control_band_contract_sha256": BroadbandFullOctaveContractV3.canonical().digest(),
        "campaign_identity": dict(campaign_identity),
        "counterbalance_scheme": COUNTERBALANCE_SCHEME,
        "comparison_units": units,
        "campaign_capture_order": capture_order,
        "one_shot": dict(one_shot),
    }
    payload["plan_evidence_sha256"] = _sha256_bytes(_canonical_json_bytes(payload))
    _validate_plan(payload)
    return payload


def audit_full_octave_v3_matched_campaign(
    payload: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    """physical matched campaign receipt를 읽기만 검증한다.

    어떤 분기에서도 canonical performance/deployment authority를 발행하지 않는다.
    특히 JSON의 self-attestation은 capture adapter의 외부 물리 provenance가 아니므로,
    형식상 완전한 non-fixture campaign도
    ``BLOCKED_UNATTESTED_PHYSICAL_PROVENANCE``로 유지한다. default null config의 정상
    결과는 ``BLOCKED``이다.
    """

    root = Path(repository_root).resolve(strict=True)
    config = _validate_config(payload)
    intents: dict[str, dict[str, str] | None] = config["intents"]
    base = {
        "schema": REPORT_SCHEMA,
        "role": ROLE,
        "audio_opened": False,
        "alsa_opened": False,
        "gpu_initialized": False,
        "network_opened": False,
        "results_written": False,
        "physical_attenuation_math_performed": False,
        "raw_capture_adapter_run": False,
        "declared_sha_structure_valid": False,
        "matched_campaign_structural_valid": False,
        "physical_provenance_attested": False,
        "control_band_contract_sha256": BroadbandFullOctaveContractV3.canonical().digest(),
        "canonical_matched_physical_pass": False,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
    }
    missing = sorted(name for name, value in intents.items() if value is None)
    if missing:
        return {
            **base,
            "status": "BLOCKED",
            "missing_artifacts": missing,
            "blocking_requirements": [
                "non_fixture_predeclared_matched_campaign_plan",
                "non_fixture_one_shot_campaign_receipt",
                "three-condition-counterbalanced-independent-groups",
                "per-session-eight-input-raw-bundle",
                "per-session-shared-identity-run-receipt",
                *UNATTESTED_PHYSICAL_PROVENANCE_BLOCKERS,
            ],
        }

    plan_snapshot = _snapshot_ref(root, intents["campaign_plan"], label="artifacts.campaign_plan")
    receipt_snapshot = _snapshot_ref(
        root, intents["campaign_receipt"], label="artifacts.campaign_receipt"
    )
    plan = _validate_plan(_load_json_no_duplicates(plan_snapshot["content"], label="campaign plan"))
    if plan["fixture_only"]:
        return {
            **base,
            "status": "BLOCKED",
            "fixture_only_evidence": True,
            "blocking_requirements": [
                "non_fixture_predeclared_matched_campaign_plan",
                *UNATTESTED_PHYSICAL_PROVENANCE_BLOCKERS,
            ],
        }
    receipt = _validate_receipt(
        _load_json_no_duplicates(receipt_snapshot["content"], label="campaign receipt")
    )
    if receipt["fixture_only"]:
        return {
            **base,
            "status": "BLOCKED",
            "fixture_only_evidence": True,
            "blocking_requirements": [
                "non_fixture_one_shot_campaign_receipt",
                *UNATTESTED_PHYSICAL_PROVENANCE_BLOCKERS,
            ],
        }
    if receipt["campaign_plan_file_sha256"] != plan_snapshot["sha256"]:
        raise ValueError("campaign receipt plan file SHA가 config artifact와 다릅니다")
    if receipt["campaign_plan_evidence_sha256"] != plan["payload"]["plan_evidence_sha256"]:
        raise ValueError("campaign receipt plan evidence SHA가 campaign plan과 다릅니다")
    if receipt["observed_capture_order"] != plan["capture_order"]:
        raise ValueError("one-shot receipt observed capture order가 predeclared plan과 다릅니다")

    expected_sessions = {
        session["session_id"]: (unit, session)
        for unit in plan["units"]
        for session in unit["sessions"]
    }
    receipt_ids = [entry["session_id"] for entry in receipt["session_receipts"]]
    if receipt_ids != plan["capture_order"] or set(receipt_ids) != set(expected_sessions):
        raise ValueError("one-shot receipt session 집합/순서가 predeclared campaign과 다릅니다")

    _snapshot_identity_references(root, plan["identity"], plan["one_shot"])
    bundle_config_paths: set[str] = set()
    raw_paths: set[str] = set()
    # Path를 바꾸는 것만으로 같은 capture bytes 재사용을 숨기지 못하게 한다. 한 capture의
    # native→canonical transform은 identity여서 같은 SHA일 수 있으므로 **같은 session의
    # native/canonical pair만** 허용한다. 그러나 다른 matched session에서 native/canonical
    # 어느 kind든 같은 bytes가 나타나면 condition별 독립 capture 전제가 깨진다.
    raw_sha256_origin_session: dict[str, str] = {}
    session_records: list[dict[str, Any]] = []
    for entry in receipt["session_receipts"]:
        unit, expected = expected_sessions[entry["session_id"]]
        if (
            entry["comparison_unit_id"] != unit["comparison_unit_id"]
            or entry["condition"] != expected["condition"]
            or entry["order_index"] != expected["order_index"]
        ):
            raise ValueError("campaign receipt session metadata가 predeclared plan과 다릅니다")
        _snapshot_ref(root, unit["source"]["submitted_pcm"], label="unit.submitted_pcm")
        _snapshot_ref(root, unit["source"]["source_manifest"], label="unit.source_manifest")
        _snapshot_ref(root, expected["controller"]["controller_artifact"], label="session.controller_artifact")
        _snapshot_ref(root, expected["controller"]["controller_config"], label="session.controller_config")
        record = _verify_bundle_against_expected(
            root=root,
            bundle_ref=entry["bundle_config"],
            expected_unit=unit,
            expected_session=expected,
            identity=plan["identity"],
        )
        run_receipt = _verify_session_run_receipt(
            root=root,
            run_ref=entry["comparison_run_receipt"],
            expected_unit=unit,
            expected_session=expected,
            identity=plan["identity"],
            bundle_record=record,
        )
        config_path = record["bundle_config"]["path"]
        if config_path in bundle_config_paths:
            raise ValueError("같은 8-input bundle config를 여러 matched session에 재사용할 수 없습니다")
        bundle_config_paths.add(config_path)
        for raw_kind in ("native_raw", "canonical_raw"):
            raw_path = record[raw_kind]["path"]
            if raw_path in raw_paths:
                raise ValueError("같은 raw file을 여러 matched session에 재사용할 수 없습니다")
            raw_paths.add(raw_path)
            raw_sha256 = _require_sha256(
                record[raw_kind].get("sha256"),
                label=f"{expected['session_id']}.{raw_kind}.sha256",
            )
            previous_session_id = raw_sha256_origin_session.get(raw_sha256)
            if previous_session_id is not None and previous_session_id != expected["session_id"]:
                raise ValueError(
                    "native/canonical raw SHA는 서로 다른 session에서 campaign-wide unique해야 합니다: "
                    f"{previous_session_id}와 {expected['session_id']}.{raw_kind}가 같은 bytes입니다"
                )
            raw_sha256_origin_session[raw_sha256] = expected["session_id"]
        session_records.append(
            {
                "comparison_unit_id": unit["comparison_unit_id"],
                "independent_group_id": unit["independent_group_id"],
                "source_family": unit["source"]["source_family"],
                "session_id": expected["session_id"],
                "condition": expected["condition"],
                "order_index": expected["order_index"],
                "comparison_run_receipt": run_receipt,
                **record,
            }
        )

    return {
        **base,
        # checksum/field equality만 self-attested artifact에서 확인했다. 이 명칭을
        # physical validity와 혼동하지 않도록 별도 field로 제한하고 status는 fail-closed
        # physical-provenance blocker로 고정한다.
        "status": UNATTESTED_PHYSICAL_PROVENANCE_STATUS,
        "declared_sha_structure_valid": True,
        "matched_campaign_structural_valid": False,
        "physical_provenance_attested": False,
        "self_attested_artifacts_only": True,
        "campaign_plan": {
            "path": plan_snapshot["path"],
            "file_sha256": plan_snapshot["sha256"],
            "evidence_sha256": plan["payload"]["plan_evidence_sha256"],
        },
        "campaign_receipt": {
            "path": receipt_snapshot["path"],
            "file_sha256": receipt_snapshot["sha256"],
            "evidence_sha256": receipt["payload"]["receipt_evidence_sha256"],
        },
        "one_shot_lifecycle_declaration_sha_structure_valid": True,
        "counterbalance_scheme": COUNTERBALANCE_SCHEME,
        "family_independent_group_counts": plan["family_group_counts"],
        "family_counterbalance_order_counts": plan["family_order_counts"],
        "session_count": len(session_records),
        "sessions": session_records,
        "blocking_requirements": list(UNATTESTED_PHYSICAL_PROVENANCE_BLOCKERS),
        "limitations": [
            "이 checker는 선언된 SHA/field 일치만 검사하며, self-attested JSON/raw가 실제 capture였다는 증명은 하지 않습니다",
            "filesystem snapshot은 capture adapter O_EXCL event history, plan SHA+nonce 결속, adapter build/device, session monotonic sequence를 사후 증명하지 못합니다",
            "filesystem snapshot은 실제 submitted stream, level/SPL, analysis window, limiter, topology 또는 acoustic stationarity를 증명하지 못합니다",
            "full-octave raw-bound P/S·lead, canonical lineage-derived independent group, native→canonical transform 및 physical ON/OFF/FxLMS metric은 별도 authority가 필요합니다",
        ],
        "next_required_authorities": [
            *UNATTESTED_PHYSICAL_PROVENANCE_BLOCKERS,
            "physical attenuation evaluation from immutable OFF/DL/FxLMS raw windows",
            "runtime latency/xrun/deadline receipt",
            "fullband causal P/S and source-population authority",
            "one-shot physical G4 and simultaneous five-ERR quiet-zone evaluation",
        ],
    }


def load_full_octave_v3_matched_campaign(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    """YAML config와 already-published artifacts를 읽기만 감사한다."""

    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"matched campaign config를 읽을 수 없습니다: {config_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("matched campaign config YAML root는 mapping이어야 합니다")
    report = audit_full_octave_v3_matched_campaign(payload, repository_root=repository_root)
    return {
        **report,
        "config": {"path": str(config_path), "file_sha256": _sha256_bytes(raw)},
    }


__all__ = [
    "CONDITIONS",
    "CONDITION_CONTROLLER_MODE",
    "CONFIG_SCHEMA",
    "COUNTERBALANCED_ORDERS",
    "COUNTERBALANCE_SCHEME",
    "DEFAULT_CONFIG_RELATIVE_PATH",
    "MIN_INDEPENDENT_GROUPS_PER_FAMILY",
    "PLAN_SCHEMA",
    "RECEIPT_SCHEMA",
    "REPORT_SCHEMA",
    "ROLE",
    "SESSION_RUN_RECEIPT_SCHEMA",
    "UNATTESTED_PHYSICAL_PROVENANCE_BLOCKERS",
    "UNATTESTED_PHYSICAL_PROVENANCE_STATUS",
    "audit_full_octave_v3_matched_campaign",
    "build_full_octave_v3_matched_campaign_plan",
    "load_full_octave_v3_matched_campaign",
]
