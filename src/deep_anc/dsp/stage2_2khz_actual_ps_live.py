"""Stage-2 RT5640/J511 actual P/S raw-first publication helpers.

이 모듈은 오디오 backend를 import하지 않는다. 같은 카드의 S32 disarmed primitive가
반환한 application-buffer capture와 pre/post-arm receipt를 immutable NPZ로 보존하는
순수 경계다. 이 파일이 발행하는 raw도 P/S 식별이나 학습 권한을 자동으로 만들지
않으며, 후속 clock/phase/holdout 분석이 PASS한 뒤에만 별도 binding이 허용된다.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from deep_anc.data.repository_fd import (
    canonical_relative_path,
    publish_repository_bytes_noreplace,
)

from .stage2_2khz_actual_ps_plan import (
    RAW_TARGET_RELATIVE_PATH,
    validate_stage2_actual_ps_excitation_plan,
)


RAW_SCHEMA = "stage2_2khz_actual_ps_s32_native_raw_v1"
PARTIAL_RAW_SCHEMA = "stage2_2khz_actual_ps_s32_partial_raw_v1"
RAW_TARGET = RAW_TARGET_RELATIVE_PATH

_ARRAY_FIELDS = (
    "prearm_callback_sequence",
    "prearm_callback_start_frames",
    "prearm_callback_frame_counts",
    "prearm_input_buffer_adc_time",
    "prearm_output_buffer_dac_time",
    "prearm_callback_current_time",
    "prearm_callback_status_bitmask",
    "planned_callback_sequence",
    "planned_callback_start_frames",
    "planned_callback_frame_counts",
    "planned_input_buffer_adc_time",
    "planned_output_buffer_dac_time",
    "planned_callback_current_time",
    "planned_callback_status_bitmask",
    "capture_valid_mask",
    "submitted_valid_mask",
    "actual_submitted_pcm",
)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        raise ValueError("PCM SHA input은 C-contiguous여야 합니다")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        raise TypeError("raw metadata에 ndarray가 남아 있습니다")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _require_capture_arrays(
    plan: Mapping[str, Any],
    planned_s32_pcm: np.ndarray,
    captured_pcm: np.ndarray,
    telemetry: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    canonical_plan, expected = validate_stage2_actual_ps_excitation_plan(
        plan, planned_s32_pcm
    )
    captured = np.asarray(captured_pcm)
    if captured.dtype != np.dtype("<i4") or captured.shape != expected.shape:
        raise ValueError("captured S32 PCM은 planned PCM과 같은 exact <i4 shape여야 합니다")
    if not isinstance(telemetry, Mapping):
        raise TypeError("S32 telemetry는 mapping이어야 합니다")
    actual = np.asarray(telemetry.get("actual_submitted_pcm"))
    if actual.dtype != np.dtype("<i4") or actual.shape != expected.shape:
        raise ValueError("telemetry.actual_submitted_pcm이 exact planned S32과 다릅니다")
    if not np.array_equal(actual, expected):
        raise ValueError("telemetry.actual_submitted_pcm이 sealed planned S32과 다릅니다")
    if telemetry.get("actual_submitted_pcm_sha256") != array_sha256(actual):
        raise ValueError("telemetry actual submitted PCM SHA가 다릅니다")
    if telemetry.get("planned_pcm_sha256") != canonical_plan["sealed_planned_s32_pcm"]["sha256"]:
        raise ValueError("telemetry planned S32 SHA가 sealed plan과 다릅니다")
    if telemetry.get("xrun_count") != 0:
        raise ValueError("xrun이 있는 capture는 정식 raw로 발행할 수 없습니다")
    for name in ("capture_valid_mask", "submitted_valid_mask"):
        mask = np.asarray(telemetry.get(name))
        if mask.dtype != np.dtype(np.bool_) or mask.shape != (len(expected),):
            raise ValueError(f"telemetry.{name} shape/dtype가 다릅니다")
    return canonical_plan, expected, captured


def serialize_actual_ps_raw(
    *,
    plan: Mapping[str, Any],
    planned_s32_pcm: np.ndarray,
    captured_pcm: np.ndarray,
    telemetry: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    user_live_gate: Mapping[str, Any],
    post_start_receipt: Mapping[str, Any] | None,
    session: Mapping[str, Any],
    partial: bool = False,
    failure_message: str | None = None,
) -> bytes:
    """raw를 memory에서 canonical NPZ bytes로 만든다. 파일 쓰기는 호출자가 담당한다."""

    canonical_plan, expected, captured = _require_capture_arrays(
        plan, planned_s32_pcm, captured_pcm, telemetry
    )
    if not isinstance(preflight_receipt, Mapping) or preflight_receipt.get("passed") is not True:
        raise ValueError("PASS preflight receipt가 필요합니다")
    if not isinstance(user_live_gate, Mapping) or user_live_gate.get("approved") is not True:
        raise ValueError("승인된 user live gate가 필요합니다")
    if not partial and not isinstance(post_start_receipt, Mapping):
        raise ValueError("성공 raw에는 post-start receipt가 필요합니다")
    status = PARTIAL_RAW_SCHEMA if partial else RAW_SCHEMA
    metadata: dict[str, Any] = {
        "schema": status,
        "status": "PARTIAL_CAPTURE_NOT_PROMOTABLE" if partial else "CAPTURED_RAW_UNANALYZED",
        "sample_rate_hz": 48_000,
        "block_size": 256,
        "native_format": "S32_LE",
        "actual_ps_plan_sha256": canonical_plan["canonical_payload_sha256"],
        "planned_s32_pcm_sha256": array_sha256(expected),
        "captured_s32_pcm_sha256": array_sha256(captured),
        "preflight_receipt": _jsonable(preflight_receipt),
        "user_live_gate": _jsonable(user_live_gate),
        "post_start_receipt": None if post_start_receipt is None else _jsonable(post_start_receipt),
        "session": _jsonable(session),
        "transport_authority": "application_buffer_only_until_clock_and_physical_ps_analysis",
        "canonical_training_eligible": False,
        "physical_ps_authority": False,
        "automatic_retry_allowed": False if partial else True,
    }
    if partial:
        metadata["failure_message"] = str(failure_message or "capture failed")
        metadata["partial_capture_never_promotable"] = True
    scalar = {
        key: _jsonable(value)
        for key, value in telemetry.items()
        if key not in _ARRAY_FIELDS
    }
    metadata["telemetry_scalar"] = scalar
    metadata["metadata_payload_sha256"] = payload_sha256(metadata)
    output = io.BytesIO()
    arrays = {
        "planned_pcm_s32": np.asarray(expected, dtype="<i4"),
        "captured_pcm_s32": np.asarray(captured, dtype="<i4"),
        **{
            name: np.asarray(telemetry[name])
            for name in _ARRAY_FIELDS
            if name in telemetry
        },
    }
    np.savez(output, metadata_json=np.asarray(_canonical_json_bytes(metadata).decode("utf-8")), **arrays)
    return output.getvalue()


def publish_actual_ps_raw_no_replace(
    repository_root: str | Path,
    *,
    plan: Mapping[str, Any],
    planned_s32_pcm: np.ndarray,
    captured_pcm: np.ndarray,
    telemetry: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    user_live_gate: Mapping[str, Any],
    post_start_receipt: Mapping[str, Any] | None,
    session: Mapping[str, Any],
    partial: bool = False,
    failure_message: str | None = None,
) -> dict[str, Any]:
    """success/partial raw를 고정 target에 O_EXCL로 한 번만 발행한다."""

    target = canonical_relative_path(RAW_TARGET, label="Stage-2 actual P/S raw target")
    payload = serialize_actual_ps_raw(
        plan=plan,
        planned_s32_pcm=planned_s32_pcm,
        captured_pcm=captured_pcm,
        telemetry=telemetry,
        preflight_receipt=preflight_receipt,
        user_live_gate=user_live_gate,
        post_start_receipt=post_start_receipt,
        session=session,
        partial=partial,
        failure_message=failure_message,
    )
    return publish_repository_bytes_noreplace(
        repository_root,
        target,
        payload,
        mode=0o600,
        preserve_recovery_link=True,
        recovery_tag="stage2_actual_ps_s32",
    )


def load_actual_ps_raw_bytes(payload: bytes) -> dict[str, Any]:
    """후속 offline analyzer가 사용할 수 있도록 raw bytes를 read-only로 읽는다."""

    if type(payload) is not bytes:
        raise TypeError("raw payload는 bytes여야 합니다")
    digest = hashlib.sha256(payload).hexdigest()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError("actual P/S raw metadata_json이 없습니다")
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {
            key: np.asarray(archive[key]).copy()
            for key in archive.files
            if key != "metadata_json"
        }
    if not isinstance(metadata, Mapping):
        raise ValueError("actual P/S raw metadata가 mapping이 아닙니다")
    return {"raw_npz_sha256": digest, "metadata": dict(metadata), "arrays": arrays}


__all__ = [
    "PARTIAL_RAW_SCHEMA",
    "RAW_SCHEMA",
    "RAW_TARGET",
    "array_sha256",
    "load_actual_ps_raw_bytes",
    "payload_sha256",
    "publish_actual_ps_raw_no_replace",
    "serialize_actual_ps_raw",
]
