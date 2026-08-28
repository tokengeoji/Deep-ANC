"""v5 전이중 캡처를 immutable raw로 봉인하는 capture-only primitive.

오디오 장치를 열거나 plan/evidence 파일을 해석하지 않는다. 상위 live adapter가 이미
검증한 exact binding과 ``audio_duplex_v5`` 결과를 받아, 성공과 부분 실패를 서로 다른
상태로 보존한다. 현재 primitive post binding은 self-attestation이므로 외부 adapter의
deterministic post receipt가 별도로 결속되기 전에는 ``CAPTURE_PASS``도 분석 admission이 아니다.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import platform
import secrets
import stat
from typing import Any, Mapping

import numpy as np


LIVE_RAW_SCHEMA = "fullband_causal_live_raw_v5_v1"
PREFLIGHT_REPORT_SCHEMA = "fullband_input_preflight_report_v1"
PREFLIGHT_IDENTITY_SCHEMA = "fullband_input_preflight_identity_v1"
HARDWARE_BINDING_SCHEMA = "jetson_measurement_hardware_v1"
SESSION_SCHEMA = "fullband_causal_v5_live_session_v1"
POST_CAPTURE_BINDING_SCHEMA = "fullband_causal_v5_post_capture_binding_v1"
WRITER_CONTRACT_SCHEMA = "fullband_live_raw_container_writer_v1"
MAX_METER_AGE_SECONDS = 600.0
TIME_TOLERANCE_SECONDS = 1.0e-3
METADATA_MEMBER = "metadata_json_utf8"
TELEMETRY_ARRAY_FIELDS = (
    "callback_sequence",
    "callback_start_frames",
    "callback_frame_counts",
    "input_buffer_adc_time",
    "output_buffer_dac_time",
    "callback_current_time",
    "callback_status_bitmask",
)
RAW_ARRAY_FIELDS = (
    "planned_submitted_pcm",
    "actual_submitted_pcm",
    "captured_pcm",
    "submitted_valid_mask",
    "capture_valid_mask",
    "preflight_raw_int32",
    *TELEMETRY_ARRAY_FIELDS,
)
NPZ_MEMBERS = frozenset((*RAW_ARRAY_FIELDS, METADATA_MEMBER))

CONFIRMATION_KEYS = {
    "speaker_output",
    "user_present",
    "volume_minimum",
    "routing_and_geometry",
    "same_amplifier_setting",
}
BINDING_KEYS = {
    "signal_plan": {
        "schema",
        "path",
        "file_sha256",
        "payload_sha256",
        "pcm_sha256",
        "raw_session_relative_path",
    },
    "live_capture_authority": {
        "schema",
        "path",
        "file_sha256",
        "payload_sha256",
        "signal_plan_file_sha256",
        "signal_plan_payload_sha256",
        "signal_pcm_sha256",
        "hardware_file_sha256",
        "raw_session_relative_path",
    },
    "meter": {
        "schema",
        "path",
        "receipt_path",
        "raw_sha256",
        "receipt_sha256",
        "completed_at_utc",
        "identity_sha256",
        "followup_contract_sha256",
        "live_authority_file_sha256",
        "level_evidence_file_sha256",
        "hardware_file_sha256",
    },
    "level_evidence": {
        "schema",
        "path",
        "file_sha256",
        "identity_sha256",
        "scope",
        "preserved_raw_revalidated",
    },
    "hardware": {
        "schema",
        "path",
        "file_sha256",
        "identity_sha256",
        "physical_fingerprint_sha256",
        "resolved_devices",
    },
    "preflight": {
        "schema",
        "raw_sha256",
        "report_sha256",
        "identity_sha256",
        "passed",
    },
}
METADATA_KEYS = {
    "schema",
    "status",
    "valid",
    "analysis_admission_eligible",
    "canonical_training_eligible",
    "hardware_sample_slip_authority",
    "invalid_reasons",
    "capture_exception",
    "session",
    "bindings",
    "preflight_report",
    "container_writer_contract",
    "operator_confirmations",
    "post_capture_binding",
    "duplex_telemetry_schema",
    "duplex_telemetry_scalars",
    "array_sha256",
    "post_capture_binding_scope",
    "external_post_capture_receipt_bound",
}
SESSION_KEYS = {
    "schema",
    "capture_id",
    "started_at_utc",
    "completed_at_utc",
    "publisher_prepared_at_utc",
    "audio_lock_identity_sha256",
    "repository_commit",
    "repository_branch",
    "repository_dirty",
    "adapter_path",
    "adapter_file_sha256",
}
SESSION_INPUT_KEYS = SESSION_KEYS - {"publisher_prepared_at_utc"}
PREFLIGHT_REPORT_KEYS = {
    "schema",
    "passed",
    "identity_sha256",
    "resolved_input_device",
    "sample_rate_hz",
    "frames",
    "channels",
}
PREFLIGHT_CHANNEL_KEYS = {
    "channel",
    "valid",
    "rms_dbfs",
    "peak",
    "clip_ratio",
    "unique_codes",
    "raw_min",
    "raw_max",
    "stuck",
}
POST_CAPTURE_BINDING_KEYS = {
    "schema",
    "valid",
    "error",
    "refreshed_signal_plan_file_sha256",
    "refreshed_signal_plan_payload_sha256",
    "refreshed_signal_pcm_sha256",
    "refreshed_authority_file_sha256",
    "refreshed_authority_payload_sha256",
    "refreshed_meter_raw_sha256",
    "refreshed_meter_receipt_sha256",
    "refreshed_level_evidence_file_sha256",
    "refreshed_hardware_file_sha256",
    "refreshed_hardware_identity_sha256",
    "refreshed_physical_fingerprint_sha256",
    "refreshed_audio_lock_identity_sha256",
    "resolved_devices",
    "raw_target_fresh",
}


def _audio_module() -> Any:
    """공개 audio 계약을 런타임에 읽어 schema 문자열 복제를 피한다."""

    module = importlib.import_module("deep_anc.audio_duplex_v5")
    required = (
        "DUPLEX_TELEMETRY_SCHEMA",
        "DuplexCaptureFailure",
        "BLOCK_SIZE",
        "STATUS_XRUN_MASK",
        "STATUS_PRESENT",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"audio_duplex_v5 공개 계약이 부족합니다: {missing}")
    return module


def _authority_module() -> Any:
    """Raw admission의 trust root인 committed live authority 상수를 읽는다."""

    module = importlib.import_module("deep_anc.dsp.fullband_live_authority_v5")
    required = (
        "AUTHORITY_SCHEMA",
        "PLAN_ENVELOPE_SCHEMA",
        "SEALED_PLAN_ENVELOPE_RELATIVE_PATH",
        "SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH",
        "SEALED_RAW_RELATIVE_PATH",
        "SEALED_HARDWARE_RELATIVE_PATH",
        "EXPECTED_PLAN_PAYLOAD_SHA256",
        "EXPECTED_PCM_SHA256",
        "EXPECTED_PLAN_ENVELOPE_FILE_SHA256",
        "EXPECTED_HARDWARE_FILE_SHA256",
        "EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256",
        "EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"fullband live authority 공개 계약이 부족합니다: {missing}")
    return module


def _owned_array(value: Any) -> np.ndarray:
    return np.array(value, copy=True, order="C")


def _publisher_prepared_utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _writer_contract() -> dict[str, Any]:
    return {
        "schema": WRITER_CONTRACT_SCHEMA,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "container_writer": "numpy.savez_uncompressed",
        "canonical_json": "utf8_sort_keys_compact_allow_nan_false",
        "byte_reproducibility_scope": "same_python_numpy_runtime_only",
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_npz_bytes(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
) -> bytes:
    metadata_bytes = np.frombuffer(
        _canonical_json_bytes(metadata), dtype=np.uint8
    ).copy()
    stream = io.BytesIO()
    np.savez(
        stream,
        **{name: arrays[name] for name in RAW_ARRAY_FIELDS},
        metadata_json_utf8=metadata_bytes,
    )
    return stream.getvalue()


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.lower()
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label}는 64자리 lowercase SHA-256이어야 합니다")
    return value


def _relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}가 필요합니다")
    path = Path(value)
    if (
        "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != path.as_posix()
    ):
        raise ValueError(f"{label}는 정규화된 repository-relative POSIX 경로여야 합니다")
    return value


def _utc(value: Any, *, label: str) -> tuple[str, dt.datetime]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}가 필요합니다")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label}가 올바른 UTC 시각이 아닙니다") from error
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError(f"{label}는 timezone-aware UTC여야 합니다")
    return value, parsed


def _exact_mapping(value: Any, keys: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} key 집합이 exact하지 않습니다")
    return dict(value)


def _validate_bindings(value: Any) -> dict[str, Any]:
    bindings = _exact_mapping(value, set(BINDING_KEYS), label="live raw bindings")
    result: dict[str, Any] = {}
    for name, keys in BINDING_KEYS.items():
        result[name] = _exact_mapping(bindings[name], keys, label=f"binding.{name}")
        if not isinstance(result[name]["schema"], str) or not result[name]["schema"]:
            raise ValueError(f"binding.{name}.schema가 필요합니다")

    authority_contract = _authority_module()
    measurement_contract = importlib.import_module("deep_anc.dsp.measurement_level")
    expected_schemas = {
        "signal_plan": authority_contract.PLAN_ENVELOPE_SCHEMA,
        "live_capture_authority": authority_contract.AUTHORITY_SCHEMA,
        "meter": measurement_contract.BOOTSTRAP_METER_RAW_SCHEMA,
        "level_evidence": measurement_contract.MEASUREMENT_LEVEL_EVIDENCE_SCHEMA,
        "hardware": HARDWARE_BINDING_SCHEMA,
        "preflight": PREFLIGHT_REPORT_SCHEMA,
    }
    for name, expected in expected_schemas.items():
        if result[name]["schema"] != expected:
            raise ValueError(f"binding.{name}.schema가 canonical exact 값과 다릅니다")

    for name in ("signal_plan", "live_capture_authority", "meter", "level_evidence", "hardware"):
        _relative_path(result[name]["path"], label=f"binding.{name}.path")
    _relative_path(result["meter"]["receipt_path"], label="binding.meter.receipt_path")
    raw_path = _relative_path(
        result["signal_plan"]["raw_session_relative_path"],
        label="binding.signal_plan.raw_session_relative_path",
    )
    if _relative_path(
        result["live_capture_authority"]["raw_session_relative_path"],
        label="binding.live_capture_authority.raw_session_relative_path",
    ) != raw_path:
        raise ValueError("signal plan/authority raw path가 다릅니다")

    sha_fields = {
        "signal_plan": ("file_sha256", "payload_sha256", "pcm_sha256"),
        "live_capture_authority": (
            "file_sha256",
            "payload_sha256",
            "signal_plan_file_sha256",
            "signal_plan_payload_sha256",
            "signal_pcm_sha256",
            "hardware_file_sha256",
        ),
        "meter": (
            "raw_sha256",
            "receipt_sha256",
            "identity_sha256",
            "followup_contract_sha256",
            "live_authority_file_sha256",
            "level_evidence_file_sha256",
            "hardware_file_sha256",
        ),
        "level_evidence": ("file_sha256", "identity_sha256"),
        "hardware": (
            "file_sha256",
            "identity_sha256",
            "physical_fingerprint_sha256",
        ),
        "preflight": ("raw_sha256", "report_sha256", "identity_sha256"),
    }
    for name, fields in sha_fields.items():
        for field in fields:
            result[name][field] = _sha256(
                result[name][field], label=f"binding.{name}.{field}"
            )

    plan = result["signal_plan"]
    authority = result["live_capture_authority"]
    meter = result["meter"]
    evidence = result["level_evidence"]
    hardware = result["hardware"]
    if (
        evidence["scope"]
        != "tracked_historical_attestation_for_fresh_v5_meter_only"
        or evidence["preserved_raw_revalidated"] is not False
    ):
        raise ValueError(
            "level evidence는 tracked historical scope이고 preserved raw는 "
            "재검증되지 않은 상태여야 합니다"
        )
    cross_checks = (
        (authority["signal_plan_file_sha256"], plan["file_sha256"], "authority/plan file SHA"),
        (authority["signal_plan_payload_sha256"], plan["payload_sha256"], "authority/plan payload SHA"),
        (authority["signal_pcm_sha256"], plan["pcm_sha256"], "authority/plan PCM SHA"),
        (authority["hardware_file_sha256"], hardware["file_sha256"], "authority/hardware SHA"),
        (meter["live_authority_file_sha256"], authority["file_sha256"], "meter/authority SHA"),
        (meter["level_evidence_file_sha256"], evidence["file_sha256"], "meter/evidence SHA"),
        (meter["hardware_file_sha256"], hardware["file_sha256"], "meter/hardware SHA"),
    )
    for left, right, label in cross_checks:
        if left != right:
            raise ValueError(f"{label} 결속이 다릅니다")

    pinned_checks = (
        (
            plan["path"],
            authority_contract.SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
            "signal plan path",
        ),
        (
            plan["file_sha256"],
            authority_contract.EXPECTED_PLAN_ENVELOPE_FILE_SHA256,
            "signal plan file SHA",
        ),
        (
            plan["payload_sha256"],
            authority_contract.EXPECTED_PLAN_PAYLOAD_SHA256,
            "signal plan payload SHA",
        ),
        (
            plan["pcm_sha256"],
            authority_contract.EXPECTED_PCM_SHA256,
            "signal PCM SHA",
        ),
        (
            plan["raw_session_relative_path"],
            authority_contract.SEALED_RAW_RELATIVE_PATH,
            "signal plan raw path",
        ),
        (
            authority["path"],
            authority_contract.SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
            "live authority path",
        ),
        (
            authority["file_sha256"],
            authority_contract.EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256,
            "live authority file SHA",
        ),
        (
            authority["payload_sha256"],
            authority_contract.EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256,
            "live authority payload SHA",
        ),
        (
            hardware["path"],
            authority_contract.SEALED_HARDWARE_RELATIVE_PATH,
            "hardware path",
        ),
        (
            hardware["file_sha256"],
            authority_contract.EXPECTED_HARDWARE_FILE_SHA256,
            "hardware file SHA",
        ),
    )
    for observed, expected, label in pinned_checks:
        if observed != expected:
            raise ValueError(f"{label}가 committed authority constant와 다릅니다")

    devices = _exact_mapping(
        hardware["resolved_devices"], {"input", "output"},
        label="binding.hardware.resolved_devices",
    )
    if any(type(devices[name]) is not int or devices[name] < 0 for name in devices):
        raise ValueError("resolved device는 음이 아닌 exact int여야 합니다")
    hardware["resolved_devices"] = devices
    _utc(meter["completed_at_utc"], label="binding.meter.completed_at_utc")
    if type(result["preflight"]["passed"]) is not bool:
        raise ValueError("binding.preflight.passed는 exact bool이어야 합니다")
    return json.loads(_canonical_json_bytes(result).decode("utf-8"))


def _validate_confirmations(value: Any) -> dict[str, bool]:
    confirmations = _exact_mapping(value, CONFIRMATION_KEYS, label="operator confirmations")
    if any(confirmations[name] is not True for name in CONFIRMATION_KEYS):
        raise ValueError("다섯 operator confirmation이 모두 exact true여야 합니다")
    return {name: True for name in sorted(CONFIRMATION_KEYS)}


def _validate_post_binding(
    value: Any,
    *,
    bindings: Mapping[str, Any],
    session: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    binding = _exact_mapping(
        value, POST_CAPTURE_BINDING_KEYS, label="post_capture_binding"
    )
    if binding["schema"] != POST_CAPTURE_BINDING_SCHEMA:
        raise ValueError("post_capture_binding.schema가 exact하지 않습니다")
    if type(binding["valid"]) is not bool:
        raise ValueError("post_capture_binding.valid는 exact bool이어야 합니다")
    if type(binding["raw_target_fresh"]) is not bool:
        raise ValueError("post_capture_binding.raw_target_fresh는 exact bool이어야 합니다")
    error = binding["error"]
    if binding["valid"] is True:
        if error is not None:
            raise ValueError("valid post_capture_binding.error는 None이어야 합니다")
    elif not isinstance(error, str) or not error:
        raise ValueError("invalid post_capture_binding에는 error 문자열이 필요합니다")

    sha_fields = POST_CAPTURE_BINDING_KEYS - {
        "schema", "valid", "error", "resolved_devices", "raw_target_fresh"
    }
    for name in sha_fields:
        binding[name] = _sha256(binding[name], label=f"post_capture_binding.{name}")
    devices = _exact_mapping(
        binding["resolved_devices"], {"input", "output"},
        label="post_capture_binding.resolved_devices",
    )
    if any(type(devices[name]) is not int or devices[name] < 0 for name in devices):
        raise ValueError("post-capture resolved device는 음이 아닌 exact int여야 합니다")
    binding["resolved_devices"] = devices

    expected = {
        "refreshed_signal_plan_file_sha256": bindings["signal_plan"]["file_sha256"],
        "refreshed_signal_plan_payload_sha256": bindings["signal_plan"]["payload_sha256"],
        "refreshed_signal_pcm_sha256": bindings["signal_plan"]["pcm_sha256"],
        "refreshed_authority_file_sha256": bindings["live_capture_authority"]["file_sha256"],
        "refreshed_authority_payload_sha256": bindings["live_capture_authority"]["payload_sha256"],
        "refreshed_meter_raw_sha256": bindings["meter"]["raw_sha256"],
        "refreshed_meter_receipt_sha256": bindings["meter"]["receipt_sha256"],
        "refreshed_level_evidence_file_sha256": bindings["level_evidence"]["file_sha256"],
        "refreshed_hardware_file_sha256": bindings["hardware"]["file_sha256"],
        "refreshed_hardware_identity_sha256": bindings["hardware"]["identity_sha256"],
        "refreshed_physical_fingerprint_sha256": bindings["hardware"]["physical_fingerprint_sha256"],
        "refreshed_audio_lock_identity_sha256": session["audio_lock_identity_sha256"],
    }
    invalid: list[str] = []
    for name, expected_value in expected.items():
        if binding[name] != expected_value:
            invalid.append(f"post_capture_{name}_mismatch")
    if binding["resolved_devices"] != bindings["hardware"]["resolved_devices"]:
        invalid.append("post_capture_resolved_devices_mismatch")
    if binding["raw_target_fresh"] is False:
        invalid.append("post_capture_raw_target_not_fresh")
    if binding["valid"] is False:
        invalid.append("post_capture_binding_declared_invalid")
    checked = json.loads(_canonical_json_bytes(binding).decode("utf-8"))
    return checked, invalid


def _validate_session(value: Any) -> tuple[dict[str, Any], list[str]]:
    session = _exact_mapping(value, SESSION_KEYS, label="live raw session")
    if session["schema"] != SESSION_SCHEMA:
        raise ValueError("session.schema가 exact하지 않습니다")
    capture_id = session["capture_id"]
    if (
        not isinstance(capture_id, str)
        or len(capture_id) != 32
        or capture_id != capture_id.lower()
        or any(character not in "0123456789abcdef" for character in capture_id)
    ):
        raise ValueError("session.capture_id는 32자리 lowercase hex여야 합니다")
    _, started = _utc(session["started_at_utc"], label="session.started_at_utc")
    _, completed = _utc(
        session["completed_at_utc"], label="session.completed_at_utc"
    )
    _, prepared = _utc(
        session["publisher_prepared_at_utc"],
        label="session.publisher_prepared_at_utc",
    )
    utc_duration = (completed - started).total_seconds()
    preparation_delay = (prepared - completed).total_seconds()
    if not math.isfinite(utc_duration) or not math.isfinite(preparation_delay):
        raise ValueError("session chronology는 finite여야 합니다")
    session["audio_lock_identity_sha256"] = _sha256(
        session["audio_lock_identity_sha256"],
        label="session.audio_lock_identity_sha256",
    )
    commit = session["repository_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or commit != commit.lower()
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("session.repository_commit은 exact 40-char lowercase hex여야 합니다")
    branch = session["repository_branch"]
    if not isinstance(branch, str) or not branch or any(c.isspace() for c in branch):
        raise ValueError("session.repository_branch가 필요합니다")
    if session["repository_dirty"] is not False:
        raise ValueError("session.repository_dirty는 exact false여야 합니다")
    if session["adapter_path"] != "scripts/data/measure_paths_fullband_causal_v5.py":
        raise ValueError("session.adapter_path가 canonical adapter와 다릅니다")
    session["adapter_file_sha256"] = _sha256(
        session["adapter_file_sha256"], label="session.adapter_file_sha256"
    )
    invalid: list[str] = []
    if utc_duration <= 0.0:
        invalid.append("session_duration_zero_or_negative")
    if preparation_delay < 0.0:
        invalid.append("session_completed_after_publisher_preparation")
    checked = json.loads(_canonical_json_bytes(session).decode("utf-8"))
    return checked, invalid


def _preflight_identity_sha256(
    *,
    preflight_raw: np.ndarray,
    hardware_identity_sha256: str,
    resolved_input_device: int,
    sample_rate_hz: int,
    frames: int,
) -> str:
    payload = {
        "schema": PREFLIGHT_IDENTITY_SCHEMA,
        "raw_sha256": _array_sha256(preflight_raw),
        "hardware_identity_sha256": _sha256(
            hardware_identity_sha256, label="preflight hardware identity SHA"
        ),
        "resolved_input_device": resolved_input_device,
        "sample_rate_hz": sample_rate_hz,
        "frames": frames,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _validate_preflight_report(
    value: Any,
    *,
    preflight_raw: np.ndarray,
    preflight_binding: Mapping[str, Any],
    hardware_binding: Mapping[str, Any],
) -> dict[str, Any]:
    report = _exact_mapping(value, PREFLIGHT_REPORT_KEYS, label="preflight report")
    if report["schema"] != PREFLIGHT_REPORT_SCHEMA:
        raise ValueError("preflight report schema가 다릅니다")
    if preflight_binding["schema"] != PREFLIGHT_REPORT_SCHEMA:
        raise ValueError("preflight binding schema가 report schema와 다릅니다")
    if type(report["passed"]) is not bool:
        raise ValueError("preflight report passed는 exact bool이어야 합니다")
    report["identity_sha256"] = _sha256(
        report["identity_sha256"], label="preflight report identity SHA"
    )
    for name in ("resolved_input_device", "sample_rate_hz", "frames"):
        if type(report[name]) is not int or report[name] < 0:
            raise ValueError(
                f"preflight report {name}은 음이 아닌 exact int여야 합니다"
            )
    if (
        report["resolved_input_device"]
        != hardware_binding["resolved_devices"]["input"]
    ):
        raise ValueError("preflight input device가 hardware binding과 다릅니다")
    if report["sample_rate_hz"] != 48_000:
        raise ValueError("preflight sample rate는 exact 48000이어야 합니다")
    if report["frames"] != len(preflight_raw) or report["frames"] <= 0:
        raise ValueError("preflight report frame 수가 raw와 다릅니다")
    expected_identity = _preflight_identity_sha256(
        preflight_raw=preflight_raw,
        hardware_identity_sha256=hardware_binding["identity_sha256"],
        resolved_input_device=report["resolved_input_device"],
        sample_rate_hz=report["sample_rate_hz"],
        frames=report["frames"],
    )
    if report["identity_sha256"] != expected_identity:
        raise ValueError("preflight report identity SHA가 raw/hardware/device 계약과 다릅니다")
    if preflight_binding["identity_sha256"] != expected_identity:
        raise ValueError("preflight binding identity SHA가 결정적 identity와 다릅니다")

    analyzer = importlib.import_module("deep_anc.audio_io")
    recomputed = analyzer.analyze_int32_input_probe(preflight_raw)
    if not isinstance(recomputed, Mapping) or set(recomputed) != {"frames", "channels"}:
        raise RuntimeError("audio_io preflight analyzer 공용 결과 계약이 다릅니다")
    if recomputed["frames"] != report["frames"]:
        raise ValueError("preflight analyzer frame 수가 report와 다릅니다")
    channels = report["channels"]
    if not isinstance(channels, list) or len(channels) != 2:
        raise ValueError("preflight report는 ERR/REF 두 channel summary가 필요합니다")
    recomputed_channels = recomputed["channels"]
    if not isinstance(recomputed_channels, list) or len(recomputed_channels) != 2:
        raise RuntimeError("audio_io preflight analyzer channel 계약이 다릅니다")
    checked_channels: list[dict[str, Any]] = []
    for expected_channel, item in enumerate(channels):
        channel = _exact_mapping(
            item,
            PREFLIGHT_CHANNEL_KEYS,
            label=f"preflight channel {expected_channel}",
        )
        if (
            type(channel["channel"]) is not int
            or channel["channel"] != expected_channel
        ):
            raise ValueError("preflight channel index가 exact 0,1 순서가 아닙니다")
        for name in ("valid", "stuck"):
            if type(channel[name]) is not bool:
                raise ValueError(f"preflight channel {name}은 exact bool이어야 합니다")
        for name in ("rms_dbfs", "peak", "clip_ratio"):
            if type(channel[name]) not in (int, float) or not math.isfinite(
                float(channel[name])
            ):
                raise ValueError(f"preflight channel {name}은 finite number여야 합니다")
        if not 0.0 <= float(channel["peak"]) <= 1.0:
            raise ValueError("preflight channel peak 범위가 [0,1]이 아닙니다")
        if not 0.0 <= float(channel["clip_ratio"]) <= 1.0:
            raise ValueError("preflight channel clip_ratio 범위가 [0,1]이 아닙니다")
        for name in ("unique_codes", "raw_min", "raw_max"):
            if type(channel[name]) is not int:
                raise ValueError(f"preflight channel {name}은 exact int여야 합니다")
        if channel["unique_codes"] <= 0 or channel["raw_min"] > channel["raw_max"]:
            raise ValueError("preflight channel raw code summary가 잘못됐습니다")
        if not np.iinfo(np.int32).min <= channel["raw_min"] <= np.iinfo(np.int32).max:
            raise ValueError("preflight channel raw_min이 int32 범위 밖입니다")
        if not np.iinfo(np.int32).min <= channel["raw_max"] <= np.iinfo(np.int32).max:
            raise ValueError("preflight channel raw_max가 int32 범위 밖입니다")
        recomputed_channel = _exact_mapping(
            recomputed_channels[expected_channel], PREFLIGHT_CHANNEL_KEYS,
            label=f"audio_io preflight channel {expected_channel}",
        )
        if channel != recomputed_channel:
            raise ValueError(
                f"preflight channel {expected_channel} 전 필드가 audio_io 재계산과 다릅니다"
            )
        checked_channels.append(channel)
    if report["passed"] is not all(
        channel["valid"] for channel in checked_channels
    ):
        raise ValueError("preflight report passed가 channel valid 집계와 다릅니다")
    if report["passed"] is not preflight_binding["passed"]:
        raise ValueError("preflight report/binding passed가 다릅니다")
    report["channels"] = checked_channels
    checked = json.loads(_canonical_json_bytes(report).decode("utf-8"))
    report_sha = hashlib.sha256(_canonical_json_bytes(checked)).hexdigest()
    if report_sha != preflight_binding["report_sha256"]:
        raise ValueError("preflight report canonical payload SHA가 binding과 다릅니다")
    return checked


def _prefix_length(mask: np.ndarray, *, label: str) -> int:
    value = np.asarray(mask)
    if value.dtype != np.bool_ or value.ndim != 1:
        raise ValueError(f"{label}는 exact bool 1-D mask여야 합니다")
    false = np.flatnonzero(~value)
    prefix = len(value) if false.size == 0 else int(false[0])
    if np.any(value[prefix:]):
        raise ValueError(f"{label}가 contiguous prefix가 아닙니다")
    return prefix


def _telemetry_scalar_keys() -> set[str]:
    return {
        "schema",
        "callback_frame_semantics",
        "portaudio_xrun_status_witness",
        "hardware_sample_slip_authority",
        "watchdog_coverage",
        "sample_rate_hz",
        "block_size",
        "latency",
        "channels",
        "input_dtype",
        "output_dtype",
        "resolved_input_device",
        "resolved_output_device",
        "capture_monotonic_started",
        "capture_monotonic_completed",
        "capture_monotonic_elapsed_seconds",
        "watchdog_grace_seconds",
        "xrun_count",
        "status_present_count",
        "captured_frames",
        "submitted_frames",
        "completed",
        "callback_error",
        "canonical_invalid_reasons",
        "stream_stop_error",
        "stream_abort_error",
        "stream_close_error",
        "termination_signal",
        "normal_stop_completed",
        "output_stop_confirmed",
    }


def _normalize_capture(
    capture: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any], str | None]:
    audio = _audio_module()
    if isinstance(capture, audio.DuplexCaptureFailure):
        telemetry = dict(capture.telemetry)
        captured = _owned_array(capture.captured_pcm)
        actual = _owned_array(capture.submitted_pcm)
        cap_mask = _owned_array(capture.capture_valid_mask)
        out_mask = _owned_array(capture.submitted_valid_mask)
        capture_exception = str(capture)
    else:
        if not isinstance(capture, tuple) or len(capture) != 2:
            raise TypeError("capture는 capture_duplex_v5 성공 tuple 또는 DuplexCaptureFailure여야 합니다")
        captured = _owned_array(capture[0])
        telemetry = dict(capture[1]) if isinstance(capture[1], Mapping) else {}
        expected_keys = _telemetry_scalar_keys() | set(TELEMETRY_ARRAY_FIELDS) | {
            "actual_submitted_pcm", "capture_valid_mask", "submitted_valid_mask"
        }
        if set(telemetry) != expected_keys:
            raise ValueError("성공 duplex telemetry key 집합이 exact하지 않습니다")
        actual = _owned_array(telemetry.pop("actual_submitted_pcm"))
        cap_mask = _owned_array(telemetry.pop("capture_valid_mask"))
        out_mask = _owned_array(telemetry.pop("submitted_valid_mask"))
        capture_exception = None
    if set(telemetry) != (_telemetry_scalar_keys() | set(TELEMETRY_ARRAY_FIELDS)):
        raise ValueError("duplex telemetry key 집합이 exact하지 않습니다")
    arrays = {
        "actual_submitted_pcm": actual,
        "captured_pcm": captured,
        "submitted_valid_mask": out_mask,
        "capture_valid_mask": cap_mask,
        **{name: _owned_array(telemetry.pop(name)) for name in TELEMETRY_ARRAY_FIELDS},
    }
    return arrays, telemetry, capture_exception


def _validate_and_build_metadata(
    *,
    arrays: Mapping[str, np.ndarray],
    telemetry: Mapping[str, Any],
    capture_exception: str | None,
    session: Mapping[str, Any],
    bindings: Mapping[str, Any],
    preflight_report: Mapping[str, Any],
    operator_confirmations: Mapping[str, Any],
    post_capture_binding: Mapping[str, Any],
) -> dict[str, Any]:
    audio = _audio_module()
    if set(arrays) != set(RAW_ARRAY_FIELDS):
        raise ValueError("live raw array key 집합이 exact하지 않습니다")
    planned = np.asarray(arrays["planned_submitted_pcm"])
    actual = np.asarray(arrays["actual_submitted_pcm"])
    captured = np.asarray(arrays["captured_pcm"])
    preflight = np.asarray(arrays["preflight_raw_int32"])
    if planned.dtype != np.dtype("<i2") or planned.ndim != 2 or planned.shape[1] != 2:
        raise ValueError("planned submitted는 exact <i2 [frame,2]여야 합니다")
    if len(planned) <= 0 or len(planned) % int(audio.BLOCK_SIZE):
        raise ValueError("planned frame 수는 양수인 block 배수여야 합니다")
    if actual.dtype != np.dtype("<i2") or actual.shape != planned.shape:
        raise ValueError("actual submitted는 planned와 같은 exact <i2 shape여야 합니다")
    if captured.dtype != np.dtype("<i4") or captured.shape != planned.shape:
        raise ValueError("captured는 planned와 같은 exact <i4 shape여야 합니다")
    if preflight.dtype != np.dtype("<i4") or preflight.ndim != 2 or preflight.shape[0] <= 0 or preflight.shape[1] != 2:
        raise ValueError("preflight raw는 nonempty exact <i4 [frame,2]여야 합니다")

    submitted_prefix = _prefix_length(arrays["submitted_valid_mask"], label="submitted_valid_mask")
    captured_prefix = _prefix_length(arrays["capture_valid_mask"], label="capture_valid_mask")
    if len(arrays["submitted_valid_mask"]) != len(planned) or len(arrays["capture_valid_mask"]) != len(planned):
        raise ValueError("valid mask 길이가 planned frame 수와 다릅니다")
    if submitted_prefix != captured_prefix:
        raise ValueError("submitted/capture valid prefix가 다릅니다")
    prefix = submitted_prefix
    if prefix % int(audio.BLOCK_SIZE):
        raise ValueError("valid prefix는 exact callback block 배수여야 합니다")
    if not np.array_equal(actual[:prefix], planned[:prefix]):
        raise ValueError("actual submitted valid prefix가 planned와 다릅니다")
    if np.any(actual[prefix:] != 0):
        raise ValueError("actual submitted invalid tail은 exact zero여야 합니다")
    if np.any(captured[prefix:] != 0):
        raise ValueError("captured invalid tail은 exact zero여야 합니다")

    expected_dtypes = {
        "callback_sequence": np.dtype("<i8"),
        "callback_start_frames": np.dtype("<i8"),
        "callback_frame_counts": np.dtype("<i8"),
        "input_buffer_adc_time": np.dtype("<f8"),
        "output_buffer_dac_time": np.dtype("<f8"),
        "callback_current_time": np.dtype("<f8"),
        "callback_status_bitmask": np.dtype("<u4"),
    }
    callback_arrays = {name: np.asarray(arrays[name]) for name in TELEMETRY_ARRAY_FIELDS}
    lengths = {len(value) for value in callback_arrays.values() if value.ndim == 1}
    if any(value.ndim != 1 for value in callback_arrays.values()) or len(lengths) != 1:
        raise ValueError("callback telemetry arrays는 같은 길이의 1-D여야 합니다")
    count = next(iter(lengths))
    if count != prefix // int(audio.BLOCK_SIZE):
        raise ValueError("callback telemetry 길이가 valid prefix block 수와 다릅니다")
    for name, dtype in expected_dtypes.items():
        if callback_arrays[name].dtype != dtype:
            raise ValueError(f"{name} dtype이 exact {dtype.str}가 아닙니다")
    if not np.array_equal(callback_arrays["callback_sequence"], np.arange(count, dtype="<i8")):
        raise ValueError("callback sequence가 연속이 아닙니다")
    if not np.array_equal(
        callback_arrays["callback_start_frames"],
        np.arange(count, dtype="<i8") * int(audio.BLOCK_SIZE),
    ):
        raise ValueError("callback start frame accounting이 다릅니다")
    if np.any(callback_arrays["callback_frame_counts"] != int(audio.BLOCK_SIZE)):
        raise ValueError("callback frame count가 exact block size가 아닙니다")
    for name in ("input_buffer_adc_time", "output_buffer_dac_time", "callback_current_time"):
        values = callback_arrays[name]
        if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
            raise ValueError(f"{name}가 finite strict-monotonic이 아닙니다")

    scalar = _exact_mapping(telemetry, _telemetry_scalar_keys(), label="duplex telemetry scalar")
    if scalar["schema"] != audio.DUPLEX_TELEMETRY_SCHEMA:
        raise ValueError("duplex telemetry schema가 현재 audio public contract와 다릅니다")
    exact_contract = {
        "callback_frame_semantics": "software_accounting_only_not_hardware_slip_witness",
        "watchdog_coverage": "host_wait_until_planned_frames_plus_grace_not_hardware_deadline_witness",
        "sample_rate_hz": 48_000,
        "block_size": int(audio.BLOCK_SIZE),
        "latency": "low",
        "channels": [2, 2],
        "input_dtype": "<i4",
        "output_dtype": "<i2",
    }
    for name, expected in exact_contract.items():
        if scalar[name] != expected:
            raise ValueError(f"duplex telemetry {name} 계약이 다릅니다")
    if scalar["portaudio_xrun_status_witness"] is not True:
        raise ValueError("portaudio_xrun_status_witness는 exact true여야 합니다")
    if scalar["hardware_sample_slip_authority"] is not False:
        raise ValueError("hardware_sample_slip_authority는 exact false여야 합니다")
    for name in (
        "resolved_input_device",
        "resolved_output_device",
        "xrun_count",
        "status_present_count",
        "captured_frames",
        "submitted_frames",
    ):
        if type(scalar[name]) is not int or scalar[name] < 0:
            raise ValueError(f"duplex telemetry {name}은 음이 아닌 exact int여야 합니다")
    for name in (
        "capture_monotonic_started",
        "capture_monotonic_completed",
        "capture_monotonic_elapsed_seconds",
        "watchdog_grace_seconds",
    ):
        if type(scalar[name]) not in (int, float) or not math.isfinite(float(scalar[name])):
            raise ValueError(f"duplex telemetry {name}은 finite number여야 합니다")
    monotonic_started = float(scalar["capture_monotonic_started"])
    monotonic_completed = float(scalar["capture_monotonic_completed"])
    monotonic_elapsed = float(scalar["capture_monotonic_elapsed_seconds"])
    watchdog_grace = float(scalar["watchdog_grace_seconds"])
    if monotonic_started < 0.0 or monotonic_completed < monotonic_started:
        raise ValueError("duplex monotonic start/completion 순서가 잘못됐습니다")
    if monotonic_elapsed < 0.0 or watchdog_grace <= 0.0:
        raise ValueError("duplex elapsed/watchdog grace 범위가 잘못됐습니다")
    if not math.isclose(
        monotonic_completed - monotonic_started,
        monotonic_elapsed,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("duplex monotonic elapsed 재계산이 다릅니다")
    status = callback_arrays["callback_status_bitmask"]
    xrun_count = int(np.count_nonzero(status & int(audio.STATUS_XRUN_MASK)))
    status_count = int(np.count_nonzero(status & int(audio.STATUS_PRESENT)))
    if scalar["xrun_count"] != xrun_count or scalar["status_present_count"] != status_count:
        raise ValueError("duplex status scalar가 callback bitmask 재계산과 다릅니다")
    if scalar["captured_frames"] != prefix or scalar["submitted_frames"] != prefix:
        raise ValueError("duplex committed frame scalar가 valid mask와 다릅니다")
    reasons = scalar["canonical_invalid_reasons"]
    if not isinstance(reasons, list) or any(not isinstance(item, str) or not item for item in reasons):
        raise ValueError("canonical_invalid_reasons는 문자열 list여야 합니다")
    if len(reasons) != len(set(reasons)):
        raise ValueError("canonical_invalid_reasons가 중복됩니다")
    for name in ("completed", "normal_stop_completed", "output_stop_confirmed"):
        if type(scalar[name]) is not bool:
            raise ValueError(f"duplex telemetry {name}은 exact bool이어야 합니다")
    for name in ("callback_error", "stream_stop_error", "stream_abort_error", "stream_close_error"):
        if scalar[name] is not None and (not isinstance(scalar[name], str) or not scalar[name]):
            raise ValueError(f"duplex telemetry {name}은 None 또는 nonempty string이어야 합니다")
    termination = scalar["termination_signal"]
    if termination is not None and (
        type(termination) is not int or termination not in {1, 2, 15}
    ):
        raise ValueError("duplex telemetry termination_signal이 유효하지 않습니다")

    checked_bindings = _validate_bindings(bindings)
    if scalar["resolved_input_device"] != checked_bindings["hardware"]["resolved_devices"]["input"]:
        raise ValueError("duplex resolved input device가 hardware binding과 다릅니다")
    if scalar["resolved_output_device"] != checked_bindings["hardware"]["resolved_devices"]["output"]:
        raise ValueError("duplex resolved output device가 hardware binding과 다릅니다")
    if _array_sha256(planned) != checked_bindings["signal_plan"]["pcm_sha256"]:
        raise ValueError("planned PCM SHA가 signal plan binding과 다릅니다")
    if _array_sha256(preflight) != checked_bindings["preflight"]["raw_sha256"]:
        raise ValueError("preflight raw SHA가 binding과 다릅니다")
    checked_session, chronology_invalid = _validate_session(session)
    checked_preflight_report = _validate_preflight_report(
        preflight_report,
        preflight_raw=preflight,
        preflight_binding=checked_bindings["preflight"],
        hardware_binding=checked_bindings["hardware"],
    )
    confirmations = _validate_confirmations(operator_confirmations)
    post_binding, post_invalid = _validate_post_binding(
        post_capture_binding, bindings=checked_bindings, session=checked_session
    )
    if capture_exception is not None and (not isinstance(capture_exception, str) or not capture_exception):
        raise ValueError("capture_exception은 None 또는 nonempty string이어야 합니다")

    invalid: list[str] = [*chronology_invalid, *post_invalid]
    if post_invalid:
        invalid.append("post_capture_binding_invalid")
    _, meter_completed = _utc(
        checked_bindings["meter"]["completed_at_utc"],
        label="binding.meter.completed_at_utc",
    )
    _, session_started = _utc(
        checked_session["started_at_utc"], label="session.started_at_utc"
    )
    meter_age = (session_started - meter_completed).total_seconds()
    if not math.isfinite(meter_age) or not 0.0 <= meter_age <= MAX_METER_AGE_SECONDS:
        invalid.append("meter_session_age_invalid")
    nominal_duration = len(planned) / 48_000.0
    if not (
        nominal_duration - TIME_TOLERANCE_SECONDS
        <= monotonic_elapsed
        <= nominal_duration + watchdog_grace + TIME_TOLERANCE_SECONDS
    ):
        invalid.append("capture_monotonic_elapsed_outside_nominal_watchdog")
    if capture_exception is not None:
        invalid.append("duplex_capture_failure")
    if prefix != len(planned):
        invalid.append("capture_incomplete")
    if scalar["completed"] is False:
        invalid.append("telemetry_incomplete")
    if scalar["normal_stop_completed"] is False:
        invalid.append("normal_stop_not_completed")
    if scalar["output_stop_confirmed"] is False:
        invalid.append("output_stop_unconfirmed")
    if termination is not None:
        invalid.append("termination_signal_received")
    if np.any(status != 0) or xrun_count or status_count:
        invalid.append("callback_status_nonzero")
    for name in ("callback_error", "stream_stop_error", "stream_abort_error", "stream_close_error"):
        if scalar[name] is not None:
            invalid.append(name)
    invalid.extend(reasons)
    if post_binding["valid"] is False:
        invalid.append("post_capture_binding_invalid")
    if checked_bindings["preflight"]["passed"] is False:
        invalid.append("preflight_invalid")
    invalid = list(dict.fromkeys(invalid))
    success = not invalid
    if success and not np.array_equal(actual, planned):
        raise ValueError("full success actual submitted가 planned와 다릅니다")

    safe_scalar = json.loads(_canonical_json_bytes(scalar).decode("utf-8"))
    metadata = {
        "schema": LIVE_RAW_SCHEMA,
        "status": "CAPTURE_PASS" if success else "INVALID",
        "valid": success,
        "analysis_admission_eligible": False,
        "canonical_training_eligible": False,
        "hardware_sample_slip_authority": False,
        "invalid_reasons": invalid,
        "capture_exception": capture_exception,
        "session": checked_session,
        "bindings": checked_bindings,
        "preflight_report": checked_preflight_report,
        "container_writer_contract": _writer_contract(),
        "operator_confirmations": confirmations,
        "post_capture_binding": post_binding,
        "duplex_telemetry_schema": audio.DUPLEX_TELEMETRY_SCHEMA,
        "duplex_telemetry_scalars": safe_scalar,
        "array_sha256": {
            name: _array_sha256(np.asarray(arrays[name])) for name in RAW_ARRAY_FIELDS
        },
        "post_capture_binding_scope": "primitive_self_attestation_not_external_receipt",
        "external_post_capture_receipt_bound": False,
    }
    # allow_nan=False를 실제로 통과시켜 metadata가 canonical JSON 가능함을 보장한다.
    _canonical_json_bytes(metadata)
    return metadata


def _target_parts(
    target: Path, *, repository_root: Path
) -> tuple[Path, Path, str, tuple[str, ...], str]:
    root = Path(os.path.abspath(os.fspath(repository_root)))
    candidate = target if target.is_absolute() else root / target
    lexical = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative_path = lexical.relative_to(root)
    except ValueError as error:
        raise ValueError("live raw target은 repository 안에 있어야 합니다") from error
    if not relative_path.parts or relative_path.name in {"", ".", ".."}:
        raise ValueError("live raw target filename이 필요합니다")
    relative = relative_path.as_posix()
    return root, lexical, relative, relative_path.parts[:-1], relative_path.name


def _open_parent_dirfd_chain(
    target: Path,
    *,
    repository_root: Path,
    create: bool,
) -> tuple[Path, str, str, list[tuple[Path, int, int, int]]]:
    root, lexical, relative, parent_parts, filename = _target_parts(
        target, repository_root=repository_root
    )
    try:
        root_lstat = os.lstat(root)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"repository root가 존재하지 않습니다: {root}") from error
    if not stat.S_ISDIR(root_lstat.st_mode) or stat.S_ISLNK(root_lstat.st_mode):
        raise ValueError("repository root는 symlink가 아닌 directory여야 합니다")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    root_fd = os.open(root, directory_flags)
    opened: list[tuple[Path, int, int, int]] = []
    try:
        root_status = os.fstat(root_fd)
        if not stat.S_ISDIR(root_status.st_mode):
            raise NotADirectoryError(f"repository root fd가 directory가 아닙니다: {root}")
        opened.append((root, root_fd, root_status.st_dev, root_status.st_ino))
        cursor = root
        current_fd = root_fd
        for component in parent_parts:
            cursor = cursor / component
            try:
                component_status = os.stat(
                    component, dir_fd=current_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                component_status = None
            if component_status is not None and stat.S_ISLNK(component_status.st_mode):
                raise ValueError(f"live raw parent symlink를 거부합니다: {cursor}")
            try:
                child_fd = os.open(component, directory_flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o777, dir_fd=current_fd)
                    os.fsync(current_fd)
                except FileExistsError:
                    pass
                child_fd = os.open(component, directory_flags, dir_fd=current_fd)
            child_status = os.fstat(child_fd)
            if not stat.S_ISDIR(child_status.st_mode):
                os.close(child_fd)
                raise NotADirectoryError(f"live raw parent가 directory가 아닙니다: {cursor}")
            opened.append(
                (cursor, child_fd, child_status.st_dev, child_status.st_ino)
            )
            current_fd = child_fd
        _verify_dirfd_chain(opened)
        return lexical, relative, filename, opened
    except BaseException:
        for _, descriptor, _, _ in reversed(opened):
            os.close(descriptor)
        if not opened:
            os.close(root_fd)
        raise


def _verify_dirfd_chain(chain: list[tuple[Path, int, int, int]]) -> None:
    """열린 inode와 현재 lexical chain이 여전히 exact 동일한지 검사한다."""

    for path, descriptor, expected_dev, expected_ino in chain:
        fd_status = os.fstat(descriptor)
        try:
            lexical_status = os.lstat(path)
        except FileNotFoundError as error:
            raise RuntimeError(f"live raw parent chain이 publish 중 사라졌습니다: {path}") from error
        if (
            not stat.S_ISDIR(fd_status.st_mode)
            or not stat.S_ISDIR(lexical_status.st_mode)
            or stat.S_ISLNK(lexical_status.st_mode)
            or (fd_status.st_dev, fd_status.st_ino) != (expected_dev, expected_ino)
            or (lexical_status.st_dev, lexical_status.st_ino)
            != (expected_dev, expected_ino)
        ):
            raise RuntimeError(f"live raw parent inode/lexical chain이 변경됐습니다: {path}")


def _close_dirfd_chain(chain: list[tuple[Path, int, int, int]]) -> None:
    for _, descriptor, _, _ in reversed(chain):
        os.close(descriptor)


def _read_regular_file_at(parent_fd: int, filename: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(filename, flags, dir_fd=parent_fd)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("live raw target은 regular file이어야 합니다")
        named_status = os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(named_status.st_mode)
            or (named_status.st_dev, named_status.st_ino)
            != (status.st_dev, status.st_ino)
        ):
            raise RuntimeError("live raw target inode가 open 직후 변경됐습니다")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        final_fd_status = os.fstat(descriptor)
        final_named_status = os.stat(
            filename, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            (final_fd_status.st_dev, final_fd_status.st_ino)
            != (status.st_dev, status.st_ino)
            or (final_named_status.st_dev, final_named_status.st_ino)
            != (status.st_dev, status.st_ino)
        ):
            raise RuntimeError("live raw target inode가 fd read 중 변경됐습니다")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def publish_live_raw_v5(
    target: str | Path,
    *,
    repository_root: str | Path,
    planned_submitted_pcm: np.ndarray,
    capture: Any,
    preflight_raw_int32: np.ndarray,
    preflight_report: Mapping[str, Any],
    session: Mapping[str, Any],
    bindings: Mapping[str, Any],
    operator_confirmations: Mapping[str, Any],
    post_capture_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """성공 또는 ``DuplexCaptureFailure``를 single immutable NPZ로 발행한다."""

    owned_planned = _owned_array(planned_submitted_pcm)
    owned_preflight = _owned_array(preflight_raw_int32)
    session_input = _exact_mapping(session, SESSION_INPUT_KEYS, label="live raw session input")
    session_input["publisher_prepared_at_utc"] = _publisher_prepared_utc_now()
    normalized, telemetry, capture_exception = _normalize_capture(capture)
    arrays = {
        "planned_submitted_pcm": owned_planned,
        **normalized,
        "preflight_raw_int32": owned_preflight,
    }
    metadata = _validate_and_build_metadata(
        arrays=arrays,
        telemetry=telemetry,
        capture_exception=capture_exception,
        session=session_input,
        bindings=bindings,
        preflight_report=preflight_report,
        operator_confirmations=operator_confirmations,
        post_capture_binding=post_capture_binding,
    )
    checked_bindings = metadata["bindings"]
    raw_bytes = _canonical_npz_bytes(arrays, metadata)
    lexical, relative, filename, chain = _open_parent_dirfd_chain(
        Path(target), repository_root=Path(repository_root), create=True
    )
    parent_fd = chain[-1][1]
    if relative != checked_bindings["signal_plan"]["raw_session_relative_path"]:
        _close_dirfd_chain(chain)
        raise ValueError("live raw target이 signal plan/authority sealed path와 다릅니다")
    try:
        os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        _close_dirfd_chain(chain)
        raise FileExistsError(f"기존 live raw를 덮어쓸 수 없습니다: {lexical}")

    staging_name = ""
    staging_fd = -1
    linked = False
    warnings: list[str] = []
    try:
        staging_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        for _ in range(128):
            staging_name = f".{filename}.staging-{secrets.token_hex(16)}"
            try:
                staging_fd = os.open(
                    staging_name, staging_flags, 0o600, dir_fd=parent_fd
                )
                break
            except FileExistsError:
                continue
        if staging_fd < 0:
            raise RuntimeError("private staging 이름을 확보하지 못했습니다")
        with os.fdopen(staging_fd, "wb", closefd=False) as handle:
            handle.write(raw_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        staging_status = os.fstat(staging_fd)
        if not stat.S_ISREG(staging_status.st_mode) or staging_status.st_nlink != 1:
            raise RuntimeError("live raw staging이 private regular file이 아닙니다")
        _verify_dirfd_chain(chain)
        os.link(
            staging_name,
            filename,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        linked = True
        try:
            os.unlink(staging_name, dir_fd=parent_fd)
            staging_name = ""
        except OSError as error:
            warnings.append(
                "staging_unlink_failed_after_final_link: "
                f"{type(error).__name__}: {error}"
            )
        os.fsync(parent_fd)
        _verify_dirfd_chain(chain)
        final_status = os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(final_status.st_mode)
            or (final_status.st_dev, final_status.st_ino)
            != (staging_status.st_dev, staging_status.st_ino)
        ):
            raise RuntimeError("published live raw가 regular file이 아닙니다")
        _verify_dirfd_chain(chain)
    except BaseException:
        if staging_name and not linked:
            try:
                os.unlink(staging_name, dir_fd=parent_fd)
            except OSError:
                pass
        raise
    finally:
        if staging_fd >= 0:
            os.close(staging_fd)
        _close_dirfd_chain(chain)
    return {
        "path": lexical,
        "raw_file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "metadata": metadata,
        "publication_warnings": warnings,
    }


def load_live_raw_v5(
    path: str | Path,
    *,
    repository_root: str | Path,
    expected_bindings: Mapping[str, Any],
    expected_raw_file_sha256: str,
    require_analysis_admission: bool = False,
) -> dict[str, Any]:
    """canonical bytes와 authority binding을 재검증해 live raw를 읽는다."""

    checked_bindings = _validate_bindings(expected_bindings)
    lexical, relative, filename, chain = _open_parent_dirfd_chain(
        Path(path), repository_root=Path(repository_root), create=False
    )
    try:
        if relative != checked_bindings["signal_plan"]["raw_session_relative_path"]:
            raise ValueError("live raw path가 expected authority sealed path와 다릅니다")
        _verify_dirfd_chain(chain)
        raw_bytes = _read_regular_file_at(chain[-1][1], filename)
        _verify_dirfd_chain(chain)
    finally:
        _close_dirfd_chain(chain)
    actual_file_sha = hashlib.sha256(raw_bytes).hexdigest()
    if actual_file_sha != _sha256(expected_raw_file_sha256, label="expected raw file SHA"):
        raise ValueError("live raw file SHA가 expected receipt와 다릅니다")
    try:
        with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as archive:
            if set(archive.files) != NPZ_MEMBERS:
                raise ValueError("live raw NPZ member 집합이 exact하지 않습니다")
            arrays = {name: _owned_array(archive[name]) for name in RAW_ARRAY_FIELDS}
            metadata_member = _owned_array(archive[METADATA_MEMBER])
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and "member 집합" in str(error):
            raise
        raise ValueError(f"live raw NPZ를 안전하게 읽을 수 없습니다: {error}") from error
    if metadata_member.dtype != np.uint8 or metadata_member.ndim != 1:
        raise ValueError("metadata_json_utf8은 exact uint8 1-D여야 합니다")
    try:
        metadata = json.loads(metadata_member.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("live raw canonical metadata JSON이 잘못됐습니다") from error
    if not isinstance(metadata, dict) or set(metadata) != METADATA_KEYS:
        raise ValueError("live raw metadata key 집합이 exact하지 않습니다")
    if metadata.get("schema") != LIVE_RAW_SCHEMA:
        raise ValueError("live raw metadata schema가 다릅니다")
    if metadata.get("bindings") != checked_bindings:
        raise ValueError("live raw authority binding이 expected binding과 다릅니다")
    hashes = metadata.get("array_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(RAW_ARRAY_FIELDS):
        raise ValueError("live raw array SHA mapping이 exact하지 않습니다")
    for name in RAW_ARRAY_FIELDS:
        if hashes[name] != _array_sha256(arrays[name]):
            raise ValueError(f"live raw {name} SHA 재계산이 다릅니다")
    rebuilt = _validate_and_build_metadata(
        arrays=arrays,
        telemetry=metadata["duplex_telemetry_scalars"],
        capture_exception=metadata["capture_exception"],
        session=metadata["session"],
        bindings=metadata["bindings"],
        preflight_report=metadata["preflight_report"],
        operator_confirmations=metadata["operator_confirmations"],
        post_capture_binding=metadata["post_capture_binding"],
    )
    if rebuilt != metadata:
        raise ValueError("live raw metadata가 arrays/telemetry에서 재구성한 값과 다릅니다")
    if raw_bytes != _canonical_npz_bytes(arrays, metadata):
        raise ValueError("live raw NPZ가 canonical writer bytes와 다릅니다(repackage 거부)")
    if require_analysis_admission and metadata["analysis_admission_eligible"] is not True:
        raise ValueError("INVALID/partial live raw는 analysis admission이 아닙니다")
    return {
        "path": lexical,
        "raw_file_sha256": actual_file_sha,
        "metadata": metadata,
        "arrays": arrays,
    }


def admit_live_raw_v5_for_analysis(
    path: str | Path,
    *,
    repository_root: str | Path,
    expected_bindings: Mapping[str, Any],
    expected_raw_file_sha256: str,
) -> dict[str, Any]:
    """외부 post receipt 미결속 raw를 fail-closed로 거부하는 reserved entrypoint."""

    return load_live_raw_v5(
        path,
        repository_root=repository_root,
        expected_bindings=expected_bindings,
        expected_raw_file_sha256=expected_raw_file_sha256,
        require_analysis_admission=True,
    )


__all__ = [
    "LIVE_RAW_SCHEMA",
    "PREFLIGHT_REPORT_SCHEMA",
    "RAW_ARRAY_FIELDS",
    "TELEMETRY_ARRAY_FIELDS",
    "admit_live_raw_v5_for_analysis",
    "load_live_raw_v5",
    "publish_live_raw_v5",
]
