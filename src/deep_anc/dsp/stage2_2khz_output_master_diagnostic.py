"""Stage-2 output-master diagnostic-only raw publication boundary.

이 모듈은 오디오 backend/device를 열지 않는다. 독립 ``InputStream``/``OutputStream``
transport가 반환한 서로 다른 clock-domain raw를 고유 session 아래 no-replace로 먼저
발행하고, 같은 bytes를 다시 읽은 뒤에만 global affine clock 진단을 실행한다.

이 경로의 PASS는 split transport A/B 진단일 뿐이다. P/S stream 시작, plant 식별,
canonical training 또는 ANC 성능 권한은 어떤 경우에도 부여하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping

import numpy as np

from deep_anc.audio_duplex_stage2 import (
    BLOCK_SIZE,
    OUTPUT_MASTER_TELEMETRY_SCHEMA,
    OutputMasterCaptureFailure,
)
from deep_anc.data.repository_fd import (
    RepositoryFileGuard,
    canonical_relative_path,
    publish_repository_bytes_noreplace,
)

from .stage2_2khz_diagnostic_clock import (
    DIAGNOSTIC_CLOCK_SCHEMA,
    estimate_stage2_diagnostic_global_clock,
)
from .stage2_2khz_measurement_v2 import (
    Stage2MeasurementV2Error,
    _array_sha256,
    _payload_sha256,
    validate_stage2_v2_live_safe_fallback_plan,
)


OUTPUT_MASTER_RAW_SCHEMA = "stage2_2khz_output_master_diagnostic_raw_v1"
OUTPUT_MASTER_PARTIAL_RAW_SCHEMA = (
    "stage2_2khz_output_master_diagnostic_partial_raw_v1"
)
OUTPUT_MASTER_CLOCK_RECEIPT_SCHEMA = (
    "stage2_2khz_output_master_diagnostic_clock_receipt_v1"
)
OUTPUT_MASTER_SESSION_ROOT = "results/stage2_2khz_output_master_diagnostic"
RAW_LEAF = "diagnostic_raw.npz"
PARTIAL_RAW_LEAF = "diagnostic_partial_raw.npz"
CLOCK_RECEIPT_LEAF = "clock_receipt.json"
PRE_ROLL_FRAMES = 4_096
POST_ROLL_FRAMES = 8_192

_TELEMETRY_ARRAY_NAMES = (
    "input_callback_sequence",
    "input_callback_start_frames",
    "input_callback_frame_counts",
    "input_buffer_adc_time",
    "input_callback_current_time",
    "input_callback_status_bitmask",
    "output_callback_sequence",
    "output_callback_start_frames",
    "output_callback_frame_counts",
    "output_buffer_dac_time",
    "output_callback_current_time",
    "output_callback_status_bitmask",
    "capture_valid_mask",
    "submitted_valid_mask",
)


@dataclass
class OutputMasterDiagnosticCaptureError(RuntimeError):
    """실패 raw가 durable 발행된 뒤 surface하는 capture 오류."""

    message: str
    partial_publication: dict[str, Any]

    def __str__(self) -> str:
        return self.message


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Stage2MeasurementV2Error(
            "output-master metadata가 canonical JSON이 아닙니다"
        ) from exc


def _diagnostic_slice(
    plan: Mapping[str, Any], full_submitted_pcm: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    canonical, full = validate_stage2_v2_live_safe_fallback_plan(
        plan, full_submitted_pcm
    )
    stop = int(
        canonical["live_phase_contract"]["diagnostic_phase_stop_frame"]
    )
    submitted = np.ascontiguousarray(full[:stop], dtype="<i2")
    if len(submitted) <= 0 or len(submitted) % BLOCK_SIZE:
        raise Stage2MeasurementV2Error(
            "output-master diagnostic slice가 256-frame 정렬이 아닙니다"
        )
    return canonical, submitted


def _session_path(session_relative_path: str, leaf: str) -> str:
    session = canonical_relative_path(
        session_relative_path, label="output-master diagnostic session"
    )
    path = PurePosixPath(session)
    root = PurePosixPath(OUTPUT_MASTER_SESSION_ROOT)
    if path.parent != root or path.name in {"", ".", ".."}:
        raise Stage2MeasurementV2Error(
            "output-master session은 고정 root 바로 아래 고유 directory여야 합니다"
        )
    return (path / leaf).as_posix()


def output_master_session_targets(session_relative_path: str) -> dict[str, str]:
    """고유 session의 성공/실패/분석 no-replace target을 반환한다."""

    return {
        "raw": _session_path(session_relative_path, RAW_LEAF),
        "partial_raw": _session_path(session_relative_path, PARTIAL_RAW_LEAF),
        "clock_receipt": _session_path(session_relative_path, CLOCK_RECEIPT_LEAF),
    }


def _require_array(
    telemetry: Mapping[str, Any], name: str, *, dtype: str | None = None
) -> np.ndarray:
    if name not in telemetry:
        raise Stage2MeasurementV2Error(
            f"output-master telemetry array가 없습니다: {name}"
        )
    value = np.asarray(telemetry[name])
    if dtype is not None and value.dtype != np.dtype(dtype):
        raise Stage2MeasurementV2Error(
            f"output-master telemetry {name} dtype이 {dtype}가 아닙니다"
        )
    return value


def _validate_callback_axis(
    telemetry: Mapping[str, Any],
    *,
    role: str,
    expected_frames: int,
) -> None:
    prefix = "input" if role == "input" else "output"
    sequence = _require_array(telemetry, f"{prefix}_callback_sequence", dtype="<i8")
    starts = _require_array(
        telemetry, f"{prefix}_callback_start_frames", dtype="<i8"
    )
    counts = _require_array(
        telemetry, f"{prefix}_callback_frame_counts", dtype="<i8"
    )
    status = _require_array(
        telemetry, f"{prefix}_callback_status_bitmask", dtype="<u4"
    )
    if role == "input":
        first_time = _require_array(telemetry, "input_buffer_adc_time", dtype="<f8")
        current_time = _require_array(
            telemetry, "input_callback_current_time", dtype="<f8"
        )
    else:
        first_time = _require_array(telemetry, "output_buffer_dac_time", dtype="<f8")
        current_time = _require_array(
            telemetry, "output_callback_current_time", dtype="<f8"
        )
    length = len(sequence)
    if any(
        value.ndim != 1 or len(value) != length
        for value in (sequence, starts, counts, status, first_time, current_time)
    ):
        raise Stage2MeasurementV2Error(
            f"output-master {role} callback telemetry 길이가 일치하지 않습니다"
        )
    expected_blocks = expected_frames // BLOCK_SIZE
    if length != expected_blocks:
        raise Stage2MeasurementV2Error(
            f"output-master {role} callback 수가 frame 수와 다릅니다"
        )
    expected_sequence = np.arange(expected_blocks, dtype="<i8")
    expected_starts = expected_sequence * BLOCK_SIZE
    if (
        not np.array_equal(sequence, expected_sequence)
        or not np.array_equal(starts, expected_starts)
        or not np.all(counts == BLOCK_SIZE)
        or np.any(status != 0)
        or not np.all(np.isfinite(first_time))
        or not np.all(np.isfinite(current_time))
        or (length > 1 and np.any(np.diff(first_time) <= 0.0))
        or (length > 1 and np.any(np.diff(current_time) <= 0.0))
    ):
        raise Stage2MeasurementV2Error(
            f"output-master {role} callback cursor/status/timestamp가 유효하지 않습니다"
        )


def validate_output_master_success_telemetry(
    telemetry: Mapping[str, Any],
    *,
    captured_pcm: np.ndarray,
    expected_submitted_pcm: np.ndarray,
) -> dict[str, Any]:
    """두 clock 축을 합치지 않고 성공 transport 자체만 검증한다."""

    captured = np.asarray(captured_pcm)
    expected = np.asarray(expected_submitted_pcm)
    actual = np.asarray(telemetry.get("actual_submitted_pcm"))
    capture_mask = _require_array(telemetry, "capture_valid_mask", dtype="bool")
    submitted_mask = _require_array(
        telemetry, "submitted_valid_mask", dtype="bool"
    )
    if (
        not isinstance(telemetry, Mapping)
        or telemetry.get("schema") != OUTPUT_MASTER_TELEMETRY_SCHEMA
        or telemetry.get("transport")
        != "independent_input_output_streams_output_clock_master"
        or telemetry.get("output_clock_owner") != "outputstream_callback_only"
        or telemetry.get("input_role") != "raw_witness_only_never_output_pacing"
        or telemetry.get("cross_clock_timestamp_alignment_used") is not False
        or telemetry.get("input_output_frame_identity_claimed") is not False
        or telemetry.get("hardware_sample_slip_authority") is not False
        or telemetry.get("legacy_combined_duplex_used") is not False
        or telemetry.get("completed") is not True
        or telemetry.get("normal_stop_completed") is not True
        or telemetry.get("failure_events") != []
    ):
        raise Stage2MeasurementV2Error(
            "output-master success telemetry scalar contract가 다릅니다"
        )
    if (
        expected.dtype != np.dtype("<i2")
        or expected.ndim != 2
        or expected.shape[1] != 2
        or actual.dtype != np.dtype("<i2")
        or actual.shape != expected.shape
        or not np.array_equal(actual, expected)
        or submitted_mask.shape != (len(expected),)
        or not np.all(submitted_mask)
        or int(telemetry.get("canonical_output_frames", -1)) != len(expected)
        or int(telemetry.get("submitted_output_frames", -1)) != len(expected)
    ):
        raise Stage2MeasurementV2Error(
            "output-master output-clock bytes/mask/frame contract가 다릅니다"
        )
    if (
        captured.dtype != np.dtype("<i4")
        or captured.ndim != 2
        or captured.shape[1] != 2
        or len(captured) < len(expected)
        or len(captured) % BLOCK_SIZE
        or capture_mask.shape != (len(captured),)
        or not np.all(capture_mask)
        or int(telemetry.get("captured_input_frames", -1)) != len(captured)
        or int(telemetry.get("pre_roll_requested_frames", -1))
        != PRE_ROLL_FRAMES
        or int(telemetry.get("post_roll_requested_frames", -1))
        != POST_ROLL_FRAMES
        or int(telemetry.get("pre_roll_observed_input_frames", -1))
        < PRE_ROLL_FRAMES
        or int(telemetry.get("post_roll_observed_input_frames", -1))
        < POST_ROLL_FRAMES
    ):
        raise Stage2MeasurementV2Error(
            "output-master input-clock raw/mask/pre-post-roll contract가 다릅니다"
        )
    _validate_callback_axis(telemetry, role="input", expected_frames=len(captured))
    _validate_callback_axis(telemetry, role="output", expected_frames=len(expected))
    elapsed = float(telemetry.get("capture_monotonic_elapsed_seconds", math.nan))
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise Stage2MeasurementV2Error(
            "output-master capture elapsed가 finite 양수가 아닙니다"
        )
    return {
        "schema": "stage2_2khz_output_master_transport_receipt_v1",
        "transport_completed": True,
        "input_clock_frames": len(captured),
        "output_clock_frames": len(expected),
        "input_output_frame_identity_claimed": False,
        "cross_clock_timestamp_alignment_used": False,
        "output_submitted_pcm_sha256": _array_sha256(expected),
        "captured_input_pcm_sha256": _array_sha256(captured),
        "input_callback_count": len(telemetry["input_callback_sequence"]),
        "output_callback_count": len(telemetry["output_callback_sequence"]),
        "callback_status_nonzero_count": 0,
        "xrun_count": 0,
    }


def _telemetry_scalar(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    scalar = {
        key: value
        for key, value in telemetry.items()
        if key not in _TELEMETRY_ARRAY_NAMES and key != "actual_submitted_pcm"
    }
    if any(isinstance(value, np.ndarray) for value in scalar.values()):
        raise Stage2MeasurementV2Error(
            "output-master telemetry scalar에 ndarray가 남았습니다"
        )
    _canonical_json_bytes(scalar)
    return scalar


def _serialize_success_raw(
    *,
    submitted: np.ndarray,
    captured: np.ndarray,
    telemetry: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> bytes:
    arrays = {
        f"telemetry_{name}": np.asarray(telemetry[name])
        for name in _TELEMETRY_ARRAY_NAMES
    }
    sealed = {**dict(metadata), "telemetry_scalar": _telemetry_scalar(telemetry)}
    output = io.BytesIO()
    np.savez(
        output,
        submitted_pcm=np.asarray(submitted, dtype="<i2"),
        captured_pcm=np.asarray(captured, dtype="<i4"),
        metadata_json=np.asarray(_canonical_json_bytes(sealed).decode("utf-8")),
        **arrays,
    )
    return output.getvalue()


def publish_output_master_raw_no_replace(
    repository_root: str,
    session_relative_path: str,
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    captured_pcm: np.ndarray,
    telemetry: Mapping[str, Any],
    capture_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    canonical, submitted = _diagnostic_slice(plan, full_submitted_pcm)
    captured = np.asarray(captured_pcm)
    transport = validate_output_master_success_telemetry(
        telemetry,
        captured_pcm=captured,
        expected_submitted_pcm=submitted,
    )
    clip_count = int(
        np.count_nonzero(np.abs(captured.astype(np.int64)) >= 2**31 - 1)
    )
    metadata = {
        **dict(capture_metadata),
        "schema": OUTPUT_MASTER_RAW_SCHEMA,
        "role": "diagnostic_only_output_master_transport_ab",
        "session_relative_path": session_relative_path,
        "signal_plan_sha256": canonical["canonical_payload_sha256"],
        "submitted_pcm_sha256": _array_sha256(submitted),
        "captured_pcm_sha256": _array_sha256(captured),
        "transport_receipt": transport,
        "adc_clip_count": clip_count,
        "input_output_frame_identity_claimed": False,
        "clock_authority_granted": False,
        "diagnostic_linearity_may_run": False,
        "ps_phase_may_start": False,
        "plant_identification_eligible": False,
        "canonical_training_eligible": False,
    }
    payload = _serialize_success_raw(
        submitted=submitted,
        captured=captured,
        telemetry=telemetry,
        metadata=metadata,
    )
    target = output_master_session_targets(session_relative_path)["raw"]
    published = publish_repository_bytes_noreplace(
        repository_root,
        target,
        payload,
        mode=0o600,
        preserve_recovery_link=True,
        recovery_tag="stage2_output_master_raw",
    )
    return {**published, "transport_receipt": transport}


def _load_success_raw_bytes(
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    session_relative_path: str,
    payload: bytes,
) -> dict[str, Any]:
    if type(payload) is not bytes:
        raise TypeError("output-master raw payload는 bytes여야 합니다")
    canonical, expected = _diagnostic_slice(plan, full_submitted_pcm)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        required = {
            "submitted_pcm",
            "captured_pcm",
            "metadata_json",
            *(f"telemetry_{name}" for name in _TELEMETRY_ARRAY_NAMES),
        }
        if set(archive.files) != required:
            raise Stage2MeasurementV2Error(
                "output-master raw NPZ key set이 exact하지 않습니다"
            )
        submitted = np.asarray(archive["submitted_pcm"]).copy()
        captured = np.asarray(archive["captured_pcm"]).copy()
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {
            name: np.asarray(archive[f"telemetry_{name}"]).copy()
            for name in _TELEMETRY_ARRAY_NAMES
        }
    if (
        submitted.dtype != np.dtype("<i2")
        or not np.array_equal(submitted, expected)
        or captured.dtype != np.dtype("<i4")
        or captured.ndim != 2
        or captured.shape[1] != 2
        or not isinstance(metadata, Mapping)
        or metadata.get("schema") != OUTPUT_MASTER_RAW_SCHEMA
        or metadata.get("role") != "diagnostic_only_output_master_transport_ab"
        or metadata.get("session_relative_path") != session_relative_path
        or metadata.get("signal_plan_sha256")
        != canonical["canonical_payload_sha256"]
        or metadata.get("submitted_pcm_sha256") != _array_sha256(submitted)
        or metadata.get("captured_pcm_sha256") != _array_sha256(captured)
        or metadata.get("clock_authority_granted") is not False
        or metadata.get("ps_phase_may_start") is not False
        or metadata.get("plant_identification_eligible") is not False
        or metadata.get("canonical_training_eligible") is not False
    ):
        raise Stage2MeasurementV2Error(
            "output-master raw PCM/metadata/authority binding이 다릅니다"
        )
    scalar = metadata.get("telemetry_scalar")
    if not isinstance(scalar, Mapping):
        raise Stage2MeasurementV2Error(
            "output-master raw telemetry scalar가 없습니다"
        )
    telemetry = {**dict(scalar), **arrays, "actual_submitted_pcm": submitted.copy()}
    transport = validate_output_master_success_telemetry(
        telemetry,
        captured_pcm=captured,
        expected_submitted_pcm=submitted,
    )
    if metadata.get("transport_receipt") != transport:
        raise Stage2MeasurementV2Error(
            "output-master stored/recomputed transport receipt가 다릅니다"
        )
    clip_count = int(
        np.count_nonzero(np.abs(captured.astype(np.int64)) >= 2**31 - 1)
    )
    if int(metadata.get("adc_clip_count", -1)) != clip_count:
        raise Stage2MeasurementV2Error(
            "output-master stored/recomputed ADC clip count가 다릅니다"
        )
    return {
        "raw_npz_sha256": hashlib.sha256(payload).hexdigest(),
        "submitted_pcm": submitted,
        "captured_pcm": captured,
        "metadata": dict(metadata),
        "telemetry": telemetry,
        "transport_receipt": transport,
    }


def snapshot_published_output_master_raw(
    repository_root: str,
    session_relative_path: str,
    publication: Mapping[str, Any],
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
) -> dict[str, Any]:
    target = output_master_session_targets(session_relative_path)["raw"]
    if (
        not isinstance(publication, Mapping)
        or publication.get("path") != target
        or type(publication.get("sha256")) is not str
    ):
        raise Stage2MeasurementV2Error(
            "output-master raw publication path/SHA가 다릅니다"
        )
    with RepositoryFileGuard(
        repository_root, target, label="Stage-2 output-master diagnostic raw"
    ) as guard:
        if guard.sha256 != publication["sha256"]:
            raise Stage2MeasurementV2Error(
                "output-master published raw SHA가 다릅니다"
            )
        loaded = _load_success_raw_bytes(
            plan,
            full_submitted_pcm,
            session_relative_path=session_relative_path,
            payload=guard.bytes,
        )
        guard.verify()
    loaded["artifact_ref"] = {
        "path": target,
        "sha256": str(publication["sha256"]),
    }
    return loaded


def _serialize_partial_raw(
    *,
    planned: np.ndarray,
    failure: OutputMasterCaptureFailure,
    metadata: Mapping[str, Any],
) -> bytes:
    telemetry = dict(failure.telemetry)
    telemetry.update(
        {
            "capture_valid_mask": np.asarray(failure.capture_valid_mask),
            "submitted_valid_mask": np.asarray(failure.submitted_valid_mask),
            "actual_submitted_pcm": np.asarray(failure.submitted_pcm),
        }
    )
    arrays: dict[str, np.ndarray] = {}
    for name in _TELEMETRY_ARRAY_NAMES:
        if name not in telemetry:
            raise Stage2MeasurementV2Error(
                f"output-master partial telemetry array가 없습니다: {name}"
            )
        arrays[f"telemetry_{name}"] = np.asarray(telemetry[name])
    sealed = {**dict(metadata), "telemetry_scalar": _telemetry_scalar(telemetry)}
    output = io.BytesIO()
    np.savez(
        output,
        planned_submitted_pcm=np.asarray(planned, dtype="<i2"),
        actual_submitted_pcm=np.asarray(failure.submitted_pcm, dtype="<i2"),
        captured_pcm=np.asarray(failure.captured_pcm, dtype="<i4"),
        metadata_json=np.asarray(_canonical_json_bytes(sealed).decode("utf-8")),
        **arrays,
    )
    return output.getvalue()


def publish_output_master_partial_raw_no_replace(
    repository_root: str,
    session_relative_path: str,
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    failure: OutputMasterCaptureFailure,
    capture_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    canonical, planned = _diagnostic_slice(plan, full_submitted_pcm)
    captured = np.asarray(failure.captured_pcm)
    actual = np.asarray(failure.submitted_pcm)
    capture_mask = np.asarray(failure.capture_valid_mask)
    submitted_mask = np.asarray(failure.submitted_valid_mask)
    if (
        captured.dtype != np.dtype("<i4")
        or captured.ndim != 2
        or captured.shape[1] != 2
        or actual.dtype != np.dtype("<i2")
        or actual.shape != planned.shape
        or capture_mask.shape != (len(captured),)
        or submitted_mask.shape != (len(planned),)
    ):
        raise Stage2MeasurementV2Error(
            "output-master partial raw shape/dtype/mask가 잘못됐습니다"
        )
    metadata = {
        **dict(capture_metadata),
        "schema": OUTPUT_MASTER_PARTIAL_RAW_SCHEMA,
        "role": "diagnostic_only_output_master_transport_failure",
        "session_relative_path": session_relative_path,
        "signal_plan_sha256": canonical["canonical_payload_sha256"],
        "planned_submitted_pcm_sha256": _array_sha256(planned),
        "actual_submitted_pcm_sha256": _array_sha256(actual),
        "captured_pcm_sha256": _array_sha256(captured),
        "capture_valid_frames": int(np.count_nonzero(capture_mask)),
        "submitted_valid_frames": int(np.count_nonzero(submitted_mask)),
        "capture_exception": str(failure),
        "partial_capture_never_promotable": True,
        "automatic_retry_allowed": False,
        "clock_authority_granted": False,
        "ps_phase_may_start": False,
        "plant_identification_eligible": False,
        "canonical_training_eligible": False,
    }
    payload = _serialize_partial_raw(
        planned=planned, failure=failure, metadata=metadata
    )
    return publish_repository_bytes_noreplace(
        repository_root,
        output_master_session_targets(session_relative_path)["partial_raw"],
        payload,
        mode=0o600,
        preserve_recovery_link=True,
        recovery_tag="stage2_output_master_partial",
    )


def _validate_clock_receipt(
    clock_receipt: Mapping[str, Any],
    *,
    canonical_plan: Mapping[str, Any],
    submitted: np.ndarray,
    captured: np.ndarray,
) -> dict[str, Any]:
    if not isinstance(clock_receipt, Mapping):
        raise Stage2MeasurementV2Error("global clock estimator 결과가 mapping이 아닙니다")
    value = dict(clock_receipt)
    supplied_sha = value.pop("canonical_payload_sha256", None)
    if (
        supplied_sha != _payload_sha256(value)
        or value.get("schema") != DIAGNOSTIC_CLOCK_SCHEMA
        or value.get("signal_plan_sha256")
        != canonical_plan["canonical_payload_sha256"]
        or int(value.get("submitted_phase_frames", -1)) != len(submitted)
        or int(value.get("captured_frames", -1)) != len(captured)
        or value.get("ps_phase_may_start") is not False
    ):
        raise Stage2MeasurementV2Error(
            "global clock receipt SHA/plan/frame/PS 금지 경계가 다릅니다"
        )
    return {**value, "canonical_payload_sha256": supplied_sha}


def analyse_and_publish_output_master_clock_receipt(
    repository_root: str,
    session_relative_path: str,
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    raw_publication: Mapping[str, Any],
    clock_estimator: Callable[
        [Mapping[str, Any], np.ndarray, np.ndarray], Mapping[str, Any]
    ] = estimate_stage2_diagnostic_global_clock,
) -> dict[str, Any]:
    """durable raw를 reload한 뒤 clock-only receipt를 no-replace 발행한다."""

    canonical, _planned = _diagnostic_slice(plan, full_submitted_pcm)
    loaded = snapshot_published_output_master_raw(
        repository_root,
        session_relative_path,
        raw_publication,
        canonical,
        full_submitted_pcm,
    )
    clock = _validate_clock_receipt(
        clock_estimator(
            canonical, loaded["submitted_pcm"], loaded["captured_pcm"]
        ),
        canonical_plan=canonical,
        submitted=loaded["submitted_pcm"],
        captured=loaded["captured_pcm"],
    )
    no_clip = int(loaded["metadata"]["adc_clip_count"]) == 0
    passed = bool(clock.get("passed") is True and no_clip)
    receipt: dict[str, Any] = {
        "schema": OUTPUT_MASTER_CLOCK_RECEIPT_SCHEMA,
        "status": (
            "PASS_OUTPUT_MASTER_CLOCK_DIAGNOSTIC_PS_STILL_FORBIDDEN"
            if passed
            else "FAIL_OUTPUT_MASTER_CLOCK_DIAGNOSTIC_RAW_PRESERVED"
        ),
        "session_relative_path": session_relative_path,
        "signal_plan_sha256": canonical["canonical_payload_sha256"],
        "raw_artifact": dict(loaded["artifact_ref"]),
        "submitted_pcm_sha256": _array_sha256(loaded["submitted_pcm"]),
        "captured_pcm_sha256": _array_sha256(loaded["captured_pcm"]),
        "transport_receipt": dict(loaded["transport_receipt"]),
        "adc_clip_count": int(loaded["metadata"]["adc_clip_count"]),
        "global_clock_receipt": clock,
        "global_clock_receipt_sha256": clock["canonical_payload_sha256"],
        "passed": passed,
        "diagnostic_linearity_may_run": passed,
        "clock_authority_granted": False,
        "ps_phase_may_start": False,
        "plant_identification_eligible": False,
        "canonical_training_eligible": False,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    payload = _canonical_json_bytes(receipt) + b"\n"
    publication = publish_repository_bytes_noreplace(
        repository_root,
        output_master_session_targets(session_relative_path)["clock_receipt"],
        payload,
        mode=0o600,
        preserve_recovery_link=False,
        recovery_tag="stage2_output_master_clock",
    )
    with RepositoryFileGuard(
        repository_root,
        str(publication["path"]),
        label="Stage-2 output-master clock receipt",
    ) as guard:
        if guard.sha256 != publication["sha256"] or guard.bytes != payload:
            raise Stage2MeasurementV2Error(
                "output-master clock receipt durable reload가 다릅니다"
            )
        guard.verify()
    return {**publication, "receipt": receipt, "raw": loaded}


def capture_publish_reload_analyse_output_master_diagnostic(
    repository_root: str,
    session_relative_path: str,
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    backend: Any,
    devices: Mapping[str, int],
    capture_metadata: Mapping[str, Any],
    capture_callable: Callable[..., tuple[np.ndarray, Mapping[str, Any]]],
    pre_open_check: Callable[[], None] | None,
    watchdog_grace_seconds: float,
    on_output_closed: Callable[[bool], None] | None,
    clock_estimator: Callable[
        [Mapping[str, Any], np.ndarray, np.ndarray], Mapping[str, Any]
    ] = estimate_stage2_diagnostic_global_clock,
) -> dict[str, Any]:
    """테스트 주입 가능한 output-master capture→raw→reload→clock 경계."""

    canonical, submitted = _diagnostic_slice(plan, full_submitted_pcm)
    try:
        captured, telemetry = capture_callable(
            backend,
            submitted_pcm=submitted,
            input_device=int(devices["input"]),
            output_device=int(devices["output"]),
            pre_roll_frames=PRE_ROLL_FRAMES,
            post_roll_frames=POST_ROLL_FRAMES,
            pre_open_check=pre_open_check,
            watchdog_grace_seconds=watchdog_grace_seconds,
            on_output_closed=on_output_closed,
        )
    except OutputMasterCaptureFailure as failure:
        partial = publish_output_master_partial_raw_no_replace(
            repository_root,
            session_relative_path,
            canonical,
            full_submitted_pcm,
            failure=failure,
            capture_metadata=capture_metadata,
        )
        raise OutputMasterDiagnosticCaptureError(
            (
                "output-master capture 실패 raw를 보존했습니다: "
                f"{partial['path']} SHA={partial['sha256']}"
            ),
            partial,
        ) from failure
    raw = publish_output_master_raw_no_replace(
        repository_root,
        session_relative_path,
        canonical,
        full_submitted_pcm,
        captured_pcm=captured,
        telemetry=telemetry,
        capture_metadata=capture_metadata,
    )
    clock = analyse_and_publish_output_master_clock_receipt(
        repository_root,
        session_relative_path,
        canonical,
        full_submitted_pcm,
        raw_publication=raw,
        clock_estimator=clock_estimator,
    )
    return {
        "status": clock["receipt"]["status"],
        "raw_publication": raw,
        "clock_publication": {
            key: clock[key] for key in ("path", "sha256", "size")
        },
        "clock_receipt": clock["receipt"],
        "ps_backend_calls_allowed": 0,
        "ps_phase_may_start": False,
        "plant_identification_eligible": False,
        "canonical_training_eligible": False,
    }


__all__ = [
    "CLOCK_RECEIPT_LEAF",
    "OUTPUT_MASTER_CLOCK_RECEIPT_SCHEMA",
    "OUTPUT_MASTER_PARTIAL_RAW_SCHEMA",
    "OUTPUT_MASTER_RAW_SCHEMA",
    "OUTPUT_MASTER_SESSION_ROOT",
    "OutputMasterDiagnosticCaptureError",
    "PARTIAL_RAW_LEAF",
    "POST_ROLL_FRAMES",
    "PRE_ROLL_FRAMES",
    "RAW_LEAF",
    "analyse_and_publish_output_master_clock_receipt",
    "capture_publish_reload_analyse_output_master_diagnostic",
    "output_master_session_targets",
    "publish_output_master_partial_raw_no_replace",
    "publish_output_master_raw_no_replace",
    "snapshot_published_output_master_raw",
    "validate_output_master_success_telemetry",
]
