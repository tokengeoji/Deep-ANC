"""125 Hz--8 kHz physical campaign의 8-input raw bundle 경계.

이 모듈은 **capture adapter가 아니다.** ALSA, sounddevice, GPU, subprocess를 import하거나
열지 않고, 이미 immutable/no-replace 방식으로 발행된 future raw bundle만 읽어 검증한다.
따라서 이 파일의 가장 중요한 성질은 다음과 같다.

* REF, NOISE_TAP, CANCEL_TAP, ERR_0..ERR_4의 정확히 여덟 역할을 한 48 kHz/256
  sample frame에 묶는다.
* pre-capture plan -> native raw -> canonical raw -> sidecar 순서를 raw-first
  lifecycle로 고정한다.
* source/controller/plant/timing identity와 plan/raw/sidecar bytes SHA를 서로
  교차 결속한다.
* fixture, null static config, self-declared JSON만으로는 canonical training/deployment
  PASS를 절대로 발행하지 않는다.

``declared_sha_structure_valid``은 실제 raw bytes와 metadata가 이 *bundle schema*의
자기선언 SHA 관계에 맞는다는 뜻일 뿐 P/S 식별, nonlinearity, source lineage, training,
realtime ANC 또는 quiet-zone 성능을 뜻하지 않는다. trusted capture adapter/typed raw
validator가 아직 없으므로 complete-looking bundle도
``BLOCKED_UNATTESTED_STRUCTURAL_RAW``이며 모든 report의 canonical authority는
의도적으로 ``False``다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ..dsp.control_band_contract import BroadbandFullOctaveContractV3


CONFIG_SCHEMA = "full_octave_v3_physical_session_bundle_config_v1"
PLAN_SCHEMA = "full_octave_v3_physical_session_plan_v1"
SIDECAR_SCHEMA = "full_octave_v3_physical_session_sidecar_v1"
REPORT_SCHEMA = "full_octave_v3_physical_session_bundle_report_v1"
ROLE = "raw_first_eight_input_synchronized_physical_campaign_no_audio"

DEFAULT_CONFIG_RELATIVE_PATH = "configs/full_octave_v3_physical_session_bundle.yaml"
SAMPLE_RATE_HZ = 48_000
BLOCK_SIZE = 256
INPUT_CHANNELS = 8
RAW_DTYPE = "<i4"
RAW_LAYOUT = "interleaved_s32le"
REQUIRED_ROLES = (
    "REF",
    "NOISE_TAP",
    "CANCEL_TAP",
    "ERR_0",
    "ERR_1",
    "ERR_2",
    "ERR_3",
    "ERR_4",
)
ALLOWED_TOPOLOGIES = (
    "single_acquisition_clock_all_eight",
    "ape_external_hardware_frame_bridge_eight",
)
ALLOWED_CONTROLLER_MODES = (
    "anc_off_reference",
    "deep_anc_open_loop",
    "fxlms_reference",
    "plant_identification",
)
PUBLICATION_SEQUENCE = (
    "capture_plan",
    "native_raw",
    "canonical_raw",
    "session_sidecar",
)
PUBLICATION_METHOD = "O_EXCL"

# 이 checker는 local filesystem의 final bytes와 JSON 선언만 읽는다. O_EXCL syscall
# history, 실제 submitted PCM/telemetry, native→canonical transform recipe, P/S operator
# semantics 및 electrical witness는 사후 SHA equality로 증명할 수 없다.
UNATTESTED_STRUCTURAL_RAW_STATUS = "BLOCKED_UNATTESTED_STRUCTURAL_RAW"
UNATTESTED_STRUCTURAL_RAW_BLOCKERS = (
    "typed_primary_secondary_operator_raw_analysis_validators_and_exact_timing_crosslinks",
    "typed_raw_analysis_electrical_witness_validator",
    "actual_submitted_pcm_callback_telemetry_and_native_canonical_recipe_equality",
    "capture_adapter_o_excl_receipt_bound_to_plan_nonce_device_and_monotonic_session",
    "stage_specific_training_schema_with_canonical_finetune_init_checkpoint_contract_and_recorded_selection",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONFIG_KEYS = frozenset(
    {"schema", "role", "control_band_contract", "artifacts"}
)
_CONFIG_CONTRACT_KEYS = frozenset({"id", "sha256", "sample_rate_hz", "block_size"})
_CONFIG_ARTIFACT_KEYS = frozenset(
    {"capture_plan", "native_raw", "canonical_raw", "session_sidecar"}
)
_REF_KEYS = frozenset({"path", "sha256"})
_PLAN_KEYS = frozenset(
    {
        "schema",
        "role",
        "fixture_only",
        "control_band_contract_sha256",
        "capture",
        "artifact_targets",
        "identity",
        "publication",
        "plan_evidence_sha256",
    }
)
_PLAN_CAPTURE_KEYS = frozenset(
    {
        "sample_rate_hz",
        "block_size",
        "input_channels",
        "raw_dtype",
        "raw_layout",
        "role_channels",
        "topology",
        "same_frame_witness",
        "planned_s32_callback_sha256",
    }
)
_PLAN_TARGET_KEYS = frozenset(
    {"native_raw", "canonical_raw", "session_sidecar"}
)
_IDENTITY_KEYS = frozenset({"source", "controller", "plant", "timing"})
_SOURCE_IDENTITY_KEYS = frozenset(
    {"source_kind", "submitted_pcm_sha256", "source_manifest_sha256"}
)
_CONTROLLER_IDENTITY_KEYS = frozenset(
    {"controller_mode", "controller_artifact_sha256", "controller_config_sha256"}
)
_PLANT_IDENTITY_KEYS = frozenset(
    {
        "plant_campaign_contract_sha256",
        "hardware_fingerprint_sha256",
        "routing_geometry_sha256",
    }
)
_TIMING_IDENTITY_KEYS = frozenset(
    {
        "training_timing_contract_sha256",
        "plant_delays_sha256",
        "handoff_samples",
        "lead_samples",
        "lead_derivation",
    }
)
_SAME_FRAME_KEYS = frozenset(
    {
        "all_roles_same_frame",
        "shared_hardware_sample_clock",
        "continuous_frame_counter",
        "host_timestamp_only",
        "bclk_witness",
        "ws_witness",
        "absolute_frame_counter_witness",
        "software_timestamp_only",
    }
)
_PUBLICATION_KEYS = frozenset(
    {
        "raw_first_required",
        "no_replace_required",
        "publication_sequence",
        "publication_methods",
    }
)
_PUBLICATION_METHOD_KEYS = frozenset(PUBLICATION_SEQUENCE)
_SIDECAR_KEYS = frozenset(
    {
        "schema",
        "role",
        "fixture_only",
        "capture_plan_file_sha256",
        "capture_plan_evidence_sha256",
        "control_band_contract_sha256",
        "native_raw",
        "canonical_raw",
        "capture",
        "identity",
        "publication",
        "canonical_training_eligible",
        "deployment_eligible",
        "sidecar_evidence_sha256",
    }
)
_FILE_REFERENCE_KEYS = frozenset({"path", "size_bytes", "sha256"})
_SIDECAR_CAPTURE_KEYS = frozenset(
    {
        "sample_rate_hz",
        "block_size",
        "input_channels",
        "raw_dtype",
        "raw_layout",
        "frames",
        "role_channels",
        "topology",
        "same_frame_witness",
        "frame_counter_start",
        "frame_counter_stop_exclusive",
        "planned_s32_callback_sha256",
        "actual_s32_callback_sha256",
        "xrun_count",
        "drop_count",
        "add_count",
    }
)
_SIDECAR_PUBLICATION_KEYS = frozenset(
    {
        "raw_first",
        "publication_sequence",
        "publication_methods",
        "no_replace",
    }
)
_NO_REPLACE_KEYS = frozenset(PUBLICATION_SEQUENCE)


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


def _inside_repository(root: Path, raw_path: object, *, label: str) -> tuple[str, Path]:
    text = str(raw_path or "")
    relative = Path(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}.path는 저장소 내부 상대경로여야 합니다")
    if not relative.parts or relative.parts[0] != "results":
        raise ValueError(f"{label}.path는 raw-first results/ 아래여야 합니다")
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


def _snapshot_regular_file(path: Path, *, label: str) -> tuple[bytes, str, int, os.stat_result]:
    """``O_NOFOLLOW`` snapshot으로 TOCTOU/symlink artifact를 거부한다."""

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
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError(f"{label} snapshot 중 파일이 바뀌었습니다: {path}")
    if stat.S_ISLNK(path.lstat().st_mode):
        raise ValueError(f"{label} symlink는 허용하지 않습니다: {path}")
    content = b"".join(chunks)
    if len(content) != int(after.st_size):
        raise ValueError(f"{label} byte 수와 file size가 다릅니다: {path}")
    return content, _sha256_bytes(content), int(after.st_size), after


def _load_json_no_duplicates(content: bytes, *, label: str) -> dict[str, Any]:
    def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} JSON duplicate key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(content.decode("utf-8"), object_pairs_hook=_pairs)
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


def _file_ref_intent(value: object, *, label: str) -> tuple[str | None, str | None]:
    entry = _exact_mapping(value, expected=_REF_KEYS, label=label)
    path, digest = entry["path"], entry["sha256"]
    if (path is None) != (digest is None):
        raise ValueError(f"{label}.path와 sha256은 함께 null이거나 함께 있어야 합니다")
    if path is None:
        return None, None
    if not isinstance(path, str):
        raise ValueError(f"{label}.path는 string 또는 null이어야 합니다")
    return path, _require_sha256(digest, label=f"{label}.sha256")


def _snapshot_config_artifact(
    *, root: Path, path: str, expected_sha256: str, label: str
) -> dict[str, Any]:
    relative, target = _inside_repository(root, path, label=label)
    content, actual_sha, size, file_stat = _snapshot_regular_file(target, label=label)
    if actual_sha != expected_sha256:
        raise ValueError(f"{label} bytes SHA가 config와 다릅니다")
    return {
        "path": relative,
        "content": content,
        "sha256": actual_sha,
        "size_bytes": size,
        "stat": file_stat,
    }


def _validate_role_channels(value: object, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(REQUIRED_ROLES):
        raise ValueError(f"{label}는 {list(REQUIRED_ROLES)} exact role map이어야 합니다")
    mapping = {str(key): value[key] for key in REQUIRED_ROLES}
    indexes: list[int] = []
    for role in REQUIRED_ROLES:
        channel = mapping[role]
        if isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel < INPUT_CHANNELS:
            raise ValueError(f"{label}.{role}는 0..7 integer여야 합니다")
        indexes.append(int(channel))
    if sorted(indexes) != list(range(INPUT_CHANNELS)):
        raise ValueError(f"{label}은 여덟 역할을 0..7에 일대일 대응해야 합니다")
    return {role: int(mapping[role]) for role in REQUIRED_ROLES}


def _validate_same_frame_witness(value: object, *, label: str) -> dict[str, bool]:
    witness = _exact_mapping(value, expected=_SAME_FRAME_KEYS, label=label)
    expected = {
        "all_roles_same_frame": True,
        "shared_hardware_sample_clock": True,
        "continuous_frame_counter": True,
        "host_timestamp_only": False,
        "bclk_witness": True,
        "ws_witness": True,
        "absolute_frame_counter_witness": True,
        "software_timestamp_only": False,
    }
    for key, required in expected.items():
        _require_exact_bool(witness[key], expected=required, label=f"{label}.{key}")
    return expected


def _validate_identity(value: object, *, label: str) -> dict[str, Any]:
    identity = _exact_mapping(value, expected=_IDENTITY_KEYS, label=label)
    source = _exact_mapping(
        identity["source"], expected=_SOURCE_IDENTITY_KEYS, label=f"{label}.source"
    )
    if source["source_kind"] != "submitted_pcm":
        raise ValueError(f"{label}.source.source_kind는 submitted_pcm이어야 합니다")
    for key in ("submitted_pcm_sha256", "source_manifest_sha256"):
        _require_sha256(source[key], label=f"{label}.source.{key}")

    controller = _exact_mapping(
        identity["controller"], expected=_CONTROLLER_IDENTITY_KEYS, label=f"{label}.controller"
    )
    if controller["controller_mode"] not in ALLOWED_CONTROLLER_MODES:
        raise ValueError(f"{label}.controller.controller_mode가 허용된 physical mode가 아닙니다")
    for key in ("controller_artifact_sha256", "controller_config_sha256"):
        _require_sha256(controller[key], label=f"{label}.controller.{key}")

    plant = _exact_mapping(
        identity["plant"], expected=_PLANT_IDENTITY_KEYS, label=f"{label}.plant"
    )
    for key in _PLANT_IDENTITY_KEYS:
        _require_sha256(plant[key], label=f"{label}.plant.{key}")

    timing = _exact_mapping(
        identity["timing"], expected=_TIMING_IDENTITY_KEYS, label=f"{label}.timing"
    )
    for key in ("training_timing_contract_sha256", "plant_delays_sha256"):
        _require_sha256(timing[key], label=f"{label}.timing.{key}")
    if timing["handoff_samples"] != BLOCK_SIZE:
        raise ValueError(f"{label}.timing.handoff_samples는 exact {BLOCK_SIZE}이어야 합니다")
    _require_nonnegative_int(timing["lead_samples"], label=f"{label}.timing.lead_samples")
    if timing["lead_derivation"] != "PlantDelays.lead()":
        raise ValueError(f"{label}.timing.lead_derivation은 PlantDelays.lead()이어야 합니다")
    return {
        "source": dict(source),
        "controller": dict(controller),
        "plant": dict(plant),
        "timing": dict(timing),
    }


def _validate_publication_plan(value: object, *, label: str) -> dict[str, Any]:
    publication = _exact_mapping(value, expected=_PUBLICATION_KEYS, label=label)
    _require_exact_bool(
        publication["raw_first_required"], expected=True, label=f"{label}.raw_first_required"
    )
    _require_exact_bool(
        publication["no_replace_required"], expected=True, label=f"{label}.no_replace_required"
    )
    if tuple(publication["publication_sequence"]) != PUBLICATION_SEQUENCE:
        raise ValueError(f"{label}.publication_sequence가 raw-first exact sequence와 다릅니다")
    methods = _exact_mapping(
        publication["publication_methods"], expected=_PUBLICATION_METHOD_KEYS,
        label=f"{label}.publication_methods",
    )
    for key in PUBLICATION_SEQUENCE:
        if methods[key] != PUBLICATION_METHOD:
            raise ValueError(f"{label}.publication_methods.{key}는 {PUBLICATION_METHOD}이어야 합니다")
    return {
        "raw_first_required": True,
        "no_replace_required": True,
        "publication_sequence": list(PUBLICATION_SEQUENCE),
        "publication_methods": {key: PUBLICATION_METHOD for key in PUBLICATION_SEQUENCE},
    }


def _validate_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    plan = _self_sealed_payload(payload, evidence_key="plan_evidence_sha256", label="capture plan")
    plan = _exact_mapping(plan, expected=_PLAN_KEYS, label="capture plan")
    if plan["schema"] != PLAN_SCHEMA or plan["role"] != ROLE:
        raise ValueError("capture plan schema/role이 physical 8-input bundle과 다릅니다")
    if type(plan["fixture_only"]) is not bool:
        raise ValueError("capture plan.fixture_only는 명시적 bool이어야 합니다")
    contract = BroadbandFullOctaveContractV3.canonical()
    if plan["control_band_contract_sha256"] != contract.digest():
        raise ValueError("capture plan control-band contract SHA가 canonical v3와 다릅니다")

    capture = _exact_mapping(plan["capture"], expected=_PLAN_CAPTURE_KEYS, label="capture plan.capture")
    if (
        capture["sample_rate_hz"] != SAMPLE_RATE_HZ
        or capture["block_size"] != BLOCK_SIZE
        or capture["input_channels"] != INPUT_CHANNELS
        or capture["raw_dtype"] != RAW_DTYPE
        or capture["raw_layout"] != RAW_LAYOUT
    ):
        raise ValueError("capture plan은 exact 48 kHz/256/8ch interleaved S32이어야 합니다")
    role_channels = _validate_role_channels(capture["role_channels"], label="capture plan.capture.role_channels")
    if capture["topology"] not in ALLOWED_TOPOLOGIES:
        raise ValueError("capture plan.capture.topology가 8-input 동기 topology가 아닙니다")
    _require_sha256(
        capture["planned_s32_callback_sha256"],
        label="capture plan.capture.planned_s32_callback_sha256",
    )
    witness = _validate_same_frame_witness(
        capture["same_frame_witness"], label="capture plan.capture.same_frame_witness"
    )

    targets = _exact_mapping(plan["artifact_targets"], expected=_PLAN_TARGET_KEYS, label="capture plan.artifact_targets")
    target_paths: dict[str, str] = {}
    for key in _PLAN_TARGET_KEYS:
        relative = str(targets[key] or "")
        candidate = Path(relative)
        if not relative or candidate.is_absolute() or ".." in candidate.parts or not candidate.parts or candidate.parts[0] != "results":
            raise ValueError(f"capture plan.artifact_targets.{key}는 results/ 내부 상대경로여야 합니다")
        target_paths[key] = candidate.as_posix()
    if len(set(target_paths.values())) != len(target_paths):
        raise ValueError("capture plan raw/sidecar target은 서로 달라야 합니다")

    return {
        "payload": plan,
        "fixture_only": bool(plan["fixture_only"]),
        "role_channels": role_channels,
        "same_frame_witness": witness,
        "artifact_targets": target_paths,
        "identity": _validate_identity(plan["identity"], label="capture plan.identity"),
        "publication": _validate_publication_plan(plan["publication"], label="capture plan.publication"),
    }


def _validate_actual_file_reference(
    value: object,
    *,
    expected_path: str,
    snapshot: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    reference = _exact_mapping(value, expected=_FILE_REFERENCE_KEYS, label=label)
    if reference["path"] != expected_path:
        raise ValueError(f"{label}.path가 predeclared plan target과 다릅니다")
    size = _require_nonnegative_int(reference["size_bytes"], label=f"{label}.size_bytes")
    if size <= 0:
        raise ValueError(f"{label}.size_bytes는 양수여야 합니다")
    digest = _require_sha256(reference["sha256"], label=f"{label}.sha256")
    if size != snapshot["size_bytes"] or digest != snapshot["sha256"]:
        raise ValueError(f"{label} bytes가 sidecar/config SHA와 다릅니다")
    return {"path": expected_path, "size_bytes": size, "sha256": digest}


def _validate_sidecar(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    native_snapshot: Mapping[str, Any],
    canonical_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    sidecar = _self_sealed_payload(payload, evidence_key="sidecar_evidence_sha256", label="session sidecar")
    sidecar = _exact_mapping(sidecar, expected=_SIDECAR_KEYS, label="session sidecar")
    if sidecar["schema"] != SIDECAR_SCHEMA or sidecar["role"] != ROLE:
        raise ValueError("session sidecar schema/role이 physical 8-input bundle과 다릅니다")
    if type(sidecar["fixture_only"]) is not bool:
        raise ValueError("session sidecar.fixture_only는 명시적 bool이어야 합니다")
    _require_exact_bool(
        sidecar["canonical_training_eligible"], expected=False,
        label="session sidecar.canonical_training_eligible",
    )
    _require_exact_bool(
        sidecar["deployment_eligible"], expected=False, label="session sidecar.deployment_eligible"
    )
    if sidecar["capture_plan_file_sha256"] != plan_file_sha256:
        raise ValueError("session sidecar capture plan file SHA가 config bytes와 다릅니다")
    if sidecar["capture_plan_evidence_sha256"] != plan["payload"]["plan_evidence_sha256"]:
        raise ValueError("session sidecar capture plan evidence SHA가 다릅니다")
    if sidecar["control_band_contract_sha256"] != plan["payload"]["control_band_contract_sha256"]:
        raise ValueError("session sidecar control-band contract SHA가 capture plan과 다릅니다")

    native = _validate_actual_file_reference(
        sidecar["native_raw"],
        expected_path=plan["artifact_targets"]["native_raw"],
        snapshot=native_snapshot,
        label="session sidecar.native_raw",
    )
    canonical = _validate_actual_file_reference(
        sidecar["canonical_raw"],
        expected_path=plan["artifact_targets"]["canonical_raw"],
        snapshot=canonical_snapshot,
        label="session sidecar.canonical_raw",
    )

    capture = _exact_mapping(sidecar["capture"], expected=_SIDECAR_CAPTURE_KEYS, label="session sidecar.capture")
    expected_capture = plan["payload"]["capture"]
    for key in ("sample_rate_hz", "block_size", "input_channels", "raw_dtype", "raw_layout", "topology"):
        if capture[key] != expected_capture[key]:
            raise ValueError(f"session sidecar.capture.{key}가 predeclared capture plan과 다릅니다")
    if capture["sample_rate_hz"] != SAMPLE_RATE_HZ or capture["block_size"] != BLOCK_SIZE:
        raise ValueError("session sidecar는 exact 48 kHz/256이어야 합니다")
    if capture["input_channels"] != INPUT_CHANNELS:
        raise ValueError("session sidecar는 final quiet-zone 8 inputs여야 합니다")
    if capture["raw_dtype"] != RAW_DTYPE or capture["raw_layout"] != RAW_LAYOUT:
        raise ValueError("session sidecar raw format이 interleaved S32가 아닙니다")
    if _validate_role_channels(capture["role_channels"], label="session sidecar.capture.role_channels") != plan["role_channels"]:
        raise ValueError("session sidecar 8-input role map이 predeclared plan과 다릅니다")
    if _validate_same_frame_witness(
        capture["same_frame_witness"], label="session sidecar.capture.same_frame_witness"
    ) != plan["same_frame_witness"]:
        raise ValueError("session sidecar same-frame witness가 predeclared plan과 다릅니다")
    frames = _require_nonnegative_int(capture["frames"], label="session sidecar.capture.frames")
    if frames < BLOCK_SIZE or frames % BLOCK_SIZE:
        raise ValueError("session sidecar.capture.frames는 256 이상의 256 배수여야 합니다")
    first_frame = _require_nonnegative_int(
        capture["frame_counter_start"], label="session sidecar.capture.frame_counter_start"
    )
    stop_frame = _require_nonnegative_int(
        capture["frame_counter_stop_exclusive"],
        label="session sidecar.capture.frame_counter_stop_exclusive",
    )
    if stop_frame - first_frame != frames:
        raise ValueError("session sidecar continuous frame counter span이 raw frame 수와 다릅니다")
    expected_bytes = frames * INPUT_CHANNELS * 4
    if native["size_bytes"] != expected_bytes or canonical["size_bytes"] != expected_bytes:
        raise ValueError("8ch S32 raw byte length가 frame counter/format과 다릅니다")
    for key in ("planned_s32_callback_sha256", "actual_s32_callback_sha256"):
        _require_sha256(capture[key], label=f"session sidecar.capture.{key}")
    if capture["planned_s32_callback_sha256"] != expected_capture["planned_s32_callback_sha256"]:
        raise ValueError("session sidecar planned S32 callback SHA가 predeclared plan과 다릅니다")
    if capture["actual_s32_callback_sha256"] != expected_capture["planned_s32_callback_sha256"]:
        raise ValueError("session sidecar actual S32 callback SHA가 predeclared plan과 다릅니다")
    for key in ("xrun_count", "drop_count", "add_count"):
        if _require_nonnegative_int(capture[key], label=f"session sidecar.capture.{key}") != 0:
            raise ValueError(f"session sidecar.capture.{key}는 exact 0이어야 합니다")

    if _validate_identity(sidecar["identity"], label="session sidecar.identity") != plan["identity"]:
        raise ValueError("session sidecar source/controller/plant/timing identity가 plan과 다릅니다")

    publication = _exact_mapping(
        sidecar["publication"], expected=_SIDECAR_PUBLICATION_KEYS, label="session sidecar.publication"
    )
    _require_exact_bool(publication["raw_first"], expected=True, label="session sidecar.publication.raw_first")
    if tuple(publication["publication_sequence"]) != PUBLICATION_SEQUENCE:
        raise ValueError("session sidecar publication sequence가 raw-first plan과 다릅니다")
    methods = _exact_mapping(
        publication["publication_methods"], expected=_PUBLICATION_METHOD_KEYS,
        label="session sidecar.publication.publication_methods",
    )
    no_replace = _exact_mapping(
        publication["no_replace"], expected=_NO_REPLACE_KEYS,
        label="session sidecar.publication.no_replace",
    )
    for key in PUBLICATION_SEQUENCE:
        if methods[key] != PUBLICATION_METHOD:
            raise ValueError(f"session sidecar publication method {key}가 {PUBLICATION_METHOD}이 아닙니다")
        _require_exact_bool(no_replace[key], expected=True, label=f"session sidecar.no_replace.{key}")

    return {
        "payload": sidecar,
        "fixture_only": bool(sidecar["fixture_only"]),
        "frames": frames,
        "native_raw": native,
        "canonical_raw": canonical,
    }


def _validate_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    config = _exact_mapping(payload, expected=_CONFIG_KEYS, label="physical session bundle config")
    if config["schema"] != CONFIG_SCHEMA or config["role"] != ROLE:
        raise ValueError("physical session bundle config schema/role이 다릅니다")
    contract = _exact_mapping(
        config["control_band_contract"], expected=_CONFIG_CONTRACT_KEYS,
        label="physical session bundle config.control_band_contract",
    )
    canonical = BroadbandFullOctaveContractV3.canonical()
    expected_contract = {
        "id": canonical.contract_id,
        "sha256": canonical.digest(),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "block_size": BLOCK_SIZE,
    }
    if contract != expected_contract:
        raise ValueError("physical session bundle config가 canonical full-octave v3 contract와 다릅니다")
    artifacts = _exact_mapping(
        config["artifacts"], expected=_CONFIG_ARTIFACT_KEYS, label="physical session bundle config.artifacts"
    )
    return {
        "payload": config,
        "intents": {key: _file_ref_intent(artifacts[key], label=f"artifacts.{key}") for key in _CONFIG_ARTIFACT_KEYS},
    }


def build_full_octave_v3_physical_session_plan(
    *,
    artifact_targets: Mapping[str, str],
    role_channels: Mapping[str, int],
    topology: str,
    identity: Mapping[str, Any],
    planned_s32_callback_sha256: str,
) -> dict[str, Any]:
    """future recorder가 capture 전에 고정할 pure-data plan을 만든다.

    이 함수는 파일을 쓰거나 장치를 열지 않는다. 호출자는 반환값을 canonical JSON으로
    ``O_EXCL`` 발행한 뒤에만 raw capture를 시작해야 한다.
    """

    payload: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "role": ROLE,
        "fixture_only": False,
        "control_band_contract_sha256": BroadbandFullOctaveContractV3.canonical().digest(),
        "capture": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "block_size": BLOCK_SIZE,
            "input_channels": INPUT_CHANNELS,
            "raw_dtype": RAW_DTYPE,
            "raw_layout": RAW_LAYOUT,
            "role_channels": dict(role_channels),
            "topology": topology,
            "planned_s32_callback_sha256": planned_s32_callback_sha256,
            "same_frame_witness": {
                "all_roles_same_frame": True,
                "shared_hardware_sample_clock": True,
                "continuous_frame_counter": True,
                "host_timestamp_only": False,
                "bclk_witness": True,
                "ws_witness": True,
                "absolute_frame_counter_witness": True,
                "software_timestamp_only": False,
            },
        },
        "artifact_targets": dict(artifact_targets),
        "identity": dict(identity),
        "publication": {
            "raw_first_required": True,
            "no_replace_required": True,
            "publication_sequence": list(PUBLICATION_SEQUENCE),
            "publication_methods": {key: PUBLICATION_METHOD for key in PUBLICATION_SEQUENCE},
        },
    }
    # Build time validation catches unsafe target/channel/identity guesses before anyone
    # writes a plan. The returned payload is still not a physical authority.
    payload["plan_evidence_sha256"] = _sha256_bytes(_canonical_json_bytes(payload))
    _validate_plan(payload)
    return payload


def audit_full_octave_v3_physical_session_bundle(
    payload: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    """future 8-input raw bundle을 read-only/fail-closed로 감사한다.

    Default static config의 정상 결과는 ``BLOCKED``다. 이 함수는 어떤 입력에서도
    canonical training/deployment authority를 ``True``로 만들지 않는다. non-fixture
    bytes/SHA가 모두 맞아도 typed adapter provenance가 없으므로
    ``BLOCKED_UNATTESTED_STRUCTURAL_RAW``로 끝난다.
    """

    root = Path(repository_root).resolve(strict=True)
    config = _validate_config(payload)
    intents: dict[str, tuple[str | None, str | None]] = config["intents"]
    missing = [key for key in _CONFIG_ARTIFACT_KEYS if intents[key][0] is None]
    base = {
        "schema": REPORT_SCHEMA,
        "role": ROLE,
        "audio_opened": False,
        "alsa_opened": False,
        "gpu_initialized": False,
        "network_opened": False,
        "results_written": False,
        "control_band_contract_sha256": BroadbandFullOctaveContractV3.canonical().digest(),
        "required_roles": list(REQUIRED_ROLES),
        "required_input_channels": INPUT_CHANNELS,
        # `raw_bundle_structural_valid`은 과거 field와의 호환용으로 false를 유지한다.
        # SHA/field structure만 확인된 경우에는 아래 `declared_sha_structure_valid`만
        # true가 될 수 있으며, 그것도 physical provenance가 아니다.
        "declared_sha_structure_valid": False,
        "raw_bundle_structural_valid": False,
        "physical_raw_provenance_attested": False,
        "self_attested_artifacts_only": False,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
        "physical_plant_identification_pass": False,
        "quiet_zone_performance_pass": False,
    }
    if missing:
        return {
            **base,
            "status": "BLOCKED",
            "blocking_requirements": [
                "non_fixture_native_and_canonical_s32_raw",
                "predeclared_capture_plan",
                "immutable_session_sidecar",
                "same_frame_electrical_witness",
                "raw_first_no_replace_lifecycle",
                *UNATTESTED_STRUCTURAL_RAW_BLOCKERS,
            ],
            "missing_artifacts": sorted(missing),
        }

    snapshots = {
        key: _snapshot_config_artifact(
            root=root,
            path=intents[key][0] or "",
            expected_sha256=intents[key][1] or "",
            label=f"artifacts.{key}",
        )
        for key in _CONFIG_ARTIFACT_KEYS
    }
    plan_payload = _load_json_no_duplicates(snapshots["capture_plan"]["content"], label="capture plan")
    plan = _validate_plan(plan_payload)
    if plan["artifact_targets"]["native_raw"] != snapshots["native_raw"]["path"]:
        raise ValueError("capture plan native raw target이 config artifact와 다릅니다")
    if plan["artifact_targets"]["canonical_raw"] != snapshots["canonical_raw"]["path"]:
        raise ValueError("capture plan canonical raw target이 config artifact와 다릅니다")
    if plan["artifact_targets"]["session_sidecar"] != snapshots["session_sidecar"]["path"]:
        raise ValueError("capture plan sidecar target이 config artifact와 다릅니다")

    if plan["fixture_only"]:
        return {
            **base,
            "status": "BLOCKED",
            "blocking_requirements": [
                "non_fixture_physical_capture_plan_and_raw",
                *UNATTESTED_STRUCTURAL_RAW_BLOCKERS,
            ],
            "fixture_only_evidence": True,
        }

    sidecar_payload = _load_json_no_duplicates(snapshots["session_sidecar"]["content"], label="session sidecar")
    sidecar = _validate_sidecar(
        sidecar_payload,
        plan=plan,
        plan_file_sha256=snapshots["capture_plan"]["sha256"],
        native_snapshot=snapshots["native_raw"],
        canonical_snapshot=snapshots["canonical_raw"],
    )
    if sidecar["fixture_only"]:
        return {
            **base,
            "status": "BLOCKED",
            "blocking_requirements": [
                "non_fixture_physical_session_sidecar_and_raw",
                *UNATTESTED_STRUCTURAL_RAW_BLOCKERS,
            ],
            "fixture_only_evidence": True,
        }
    # Filesystem timestamps cannot prove O_EXCL after the fact. They do catch an
    # immediately impossible publication order; canonical authority remains false
    # even when these structural assertions hold.
    if not (
        snapshots["capture_plan"]["stat"].st_mtime_ns
        <= snapshots["native_raw"]["stat"].st_mtime_ns
        <= snapshots["canonical_raw"]["stat"].st_mtime_ns
        <= snapshots["session_sidecar"]["stat"].st_mtime_ns
    ):
        raise ValueError("filesystem mtime가 predeclared raw-first publication order와 다릅니다")
    # Filesystem snapshot/mtime와 self-sealed payload는 capture adapter가 실제로 stream을
    # 열었는지, planned PCM이 DAC로 제출됐는지, O_EXCL이 사용됐는지, canonical raw가
    # declared recipe로 native raw에서 만들어졌는지 증명하지 못한다. 따라서 structure가
    # complete해도 report는 authority가 아닌 blocked evidence inventory다.
    return {
        **base,
        "status": UNATTESTED_STRUCTURAL_RAW_STATUS,
        "declared_sha_structure_valid": True,
        "raw_bundle_structural_valid": False,
        "physical_raw_provenance_attested": False,
        "self_attested_artifacts_only": True,
        "frames": sidecar["frames"],
        "capture_plan": {
            "path": snapshots["capture_plan"]["path"],
            "file_sha256": snapshots["capture_plan"]["sha256"],
            "evidence_sha256": plan["payload"]["plan_evidence_sha256"],
        },
        "native_raw": sidecar["native_raw"],
        "canonical_raw": sidecar["canonical_raw"],
        "session_sidecar": {
            "path": snapshots["session_sidecar"]["path"],
            "file_sha256": snapshots["session_sidecar"]["sha256"],
            "evidence_sha256": sidecar["payload"]["sidecar_evidence_sha256"],
        },
        "lifecycle_scope": (
            "declared SHA/field structure plus weak plan/raw/sidecar mtime ordering only; "
            "post-hoc filesystem state cannot prove kernel O_EXCL history, capture adapter execution, "
            "submitted PCM/telemetry, or native↔canonical transform recipe/equality"
        ),
        "blocking_requirements": list(UNATTESTED_STRUCTURAL_RAW_BLOCKERS),
        "next_required_authorities": [
            *UNATTESTED_STRUCTURAL_RAW_BLOCKERS,
            "independent physical ANC ON/OFF and five-ERR quiet-zone evaluator",
        ],
    }


def load_full_octave_v3_physical_session_bundle(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    """YAML config와 future artifacts를 읽기만 한다. 파일을 생성하지 않는다."""

    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"physical session bundle config를 읽을 수 없습니다: {config_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("physical session bundle config YAML root는 mapping이어야 합니다")
    report = audit_full_octave_v3_physical_session_bundle(payload, repository_root=repository_root)
    return {
        **report,
        "config": {
            "path": str(config_path),
            "file_sha256": _sha256_bytes(config_path.read_bytes()),
        },
    }


__all__ = [
    "ALLOWED_CONTROLLER_MODES",
    "ALLOWED_TOPOLOGIES",
    "BLOCK_SIZE",
    "CONFIG_SCHEMA",
    "DEFAULT_CONFIG_RELATIVE_PATH",
    "INPUT_CHANNELS",
    "PLAN_SCHEMA",
    "PUBLICATION_METHOD",
    "PUBLICATION_SEQUENCE",
    "RAW_DTYPE",
    "RAW_LAYOUT",
    "REQUIRED_ROLES",
    "REPORT_SCHEMA",
    "ROLE",
    "SAMPLE_RATE_HZ",
    "SIDECAR_SCHEMA",
    "UNATTESTED_STRUCTURAL_RAW_BLOCKERS",
    "UNATTESTED_STRUCTURAL_RAW_STATUS",
    "audit_full_octave_v3_physical_session_bundle",
    "build_full_octave_v3_physical_session_plan",
    "load_full_octave_v3_physical_session_bundle",
]
