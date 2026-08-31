"""Stage-2 2 kHz live transport evidence와 immutable raw publisher.

오디오 backend는 호출자가 주입한다. 이 모듈은 capture를 시작하지 않고, 검증된 duplex
telemetry/actual PCM/raw bytes를 별도 Stage-2 schema로 no-replace 발행한다.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from typing import Any, Mapping

import numpy as np

from deep_anc.audio_duplex_stage2 import DUPLEX_TELEMETRY_SCHEMA, DuplexCaptureFailure
from deep_anc.audio_duplex_v6 import DUPLEX_TELEMETRY_SCHEMA as V6_TELEMETRY_SCHEMA
from deep_anc.data.repository_fd import publish_repository_bytes_noreplace
from deep_anc.dsp.fullband_live_delay_core_v6 import validate_duplex_telemetry_v6
from deep_anc.dsp.measurement_level import (
    OFFICIAL_MEASUREMENT_LEVEL,
    meter_raw_level_dbfs,
)

from .stage2_2khz_measurement import (
    STAGE2_RAW_SCHEMA,
    Stage2MeasurementError,
    _array_sha256,
    _canonical_json_bytes,
    _safe_relative_path,
    validate_submitted_pcm,
)


STAGE2_METER_RAW_SCHEMA = "stage2_2khz_meter_raw_v1"
STAGE2_TELEMETRY_RECEIPT_SCHEMA = "stage2_2khz_duplex_telemetry_receipt_v1"
STAGE2_CAPTURE_ADAPTER_SCHEMA = "stage2_2khz_live_capture_adapter_v1"
STAGE2_FAILURE_RAW_SCHEMA = "stage2_2khz_partial_capture_raw_v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_stage2_duplex_telemetry(
    telemetry: Mapping[str, Any],
    *,
    captured_adc_pcm: np.ndarray,
    expected_submitted_pcm: np.ndarray,
) -> dict[str, Any]:
    """별도 Stage-2 schema를 v6/v5 검증 primitive에 exact adapter로 통과시킨다."""

    if telemetry.get("schema") != DUPLEX_TELEMETRY_SCHEMA:
        raise Stage2MeasurementError("Stage-2 duplex telemetry schema가 다릅니다")
    adapted = dict(telemetry)
    adapted["schema"] = V6_TELEMETRY_SCHEMA
    receipt = validate_duplex_telemetry_v6(
        adapted,
        captured_adc_pcm=np.asarray(captured_adc_pcm),
        expected_submitted_pcm=np.asarray(expected_submitted_pcm),
    )
    result = {
        key: value
        for key, value in receipt.items()
        if key not in {"schema", "source_schema", "sha256"}
    }
    result.update(
        {
            "schema": STAGE2_TELEMETRY_RECEIPT_SCHEMA,
            "source_schema": DUPLEX_TELEMETRY_SCHEMA,
            "vetted_v5_transport_primitive_reused": True,
            "absolute_hardware_sample_slip_authority": False,
            "software_callback_and_status_gate_pass": True,
        }
    )
    serializable = json.loads(
        json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )
    serializable["sha256"] = _sha256_bytes(_canonical_json_bytes(serializable))
    return serializable


def _telemetry_arrays(telemetry: Mapping[str, Any]) -> dict[str, np.ndarray]:
    names = (
        "callback_sequence",
        "callback_start_frames",
        "callback_frame_counts",
        "input_buffer_adc_time",
        "output_buffer_dac_time",
        "callback_current_time",
        "callback_status_bitmask",
        "capture_valid_mask",
        "submitted_valid_mask",
    )
    return {f"telemetry_{name}": np.asarray(telemetry[name]) for name in names}


def _raw_npz_bytes(
    *,
    submitted_pcm: np.ndarray,
    captured_pcm: np.ndarray,
    telemetry: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> bytes:
    array_names = {
        "actual_submitted_pcm",
        "callback_sequence",
        "callback_start_frames",
        "callback_frame_counts",
        "input_buffer_adc_time",
        "output_buffer_dac_time",
        "callback_current_time",
        "callback_status_bitmask",
        "capture_valid_mask",
        "submitted_valid_mask",
    }
    telemetry_scalar = {
        key: value for key, value in telemetry.items() if key not in array_names
    }
    if any(isinstance(value, np.ndarray) for value in telemetry_scalar.values()):
        raise Stage2MeasurementError("telemetry scalar metadata에 ndarray가 남았습니다")
    sealed_metadata = {**dict(metadata), "telemetry_scalar": telemetry_scalar}
    output = io.BytesIO()
    np.savez(
        output,
        submitted_pcm=np.asarray(submitted_pcm, dtype="<i2"),
        captured_pcm=np.asarray(captured_pcm, dtype="<i4"),
        metadata_json=np.asarray(
            _canonical_json_bytes(sealed_metadata).decode("utf-8").rstrip("\n")
        ),
        **_telemetry_arrays(telemetry),
    )
    return output.getvalue()


def validate_meter_capture(
    plan: Mapping[str, Any],
    *,
    submitted_pcm: np.ndarray,
    captured_pcm: np.ndarray,
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    expected = plan.get("meter_submitted_pcm")
    submitted = np.asarray(submitted_pcm)
    captured = np.asarray(captured_pcm)
    if not isinstance(expected, Mapping):
        raise Stage2MeasurementError("plan에 meter submitted lineage가 없습니다")
    if (
        submitted.dtype != np.dtype("<i2")
        or list(submitted.shape) != expected.get("shape")
        or _array_sha256(submitted) != expected.get("sha256")
    ):
        raise Stage2MeasurementError("meter actual submitted PCM이 plan lineage와 다릅니다")
    telemetry_receipt = validate_stage2_duplex_telemetry(
        telemetry,
        captured_adc_pcm=captured,
        expected_submitted_pcm=submitted,
    )
    level = meter_raw_level_dbfs(captured, error_channel=0)
    target = float(OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs)
    tolerance = float(OFFICIAL_MEASUREMENT_LEVEL.meter_tolerance_db)
    passed = bool(math.isfinite(level) and abs(level - target) <= tolerance)
    return {
        "schema": "stage2_2khz_meter_admission_v1",
        "passed": passed,
        "meter_level_dbfs": level,
        "target_dbfs": target,
        "tolerance_db": tolerance,
        "actual_submitted_pcm_sha256": expected["sha256"],
        "captured_pcm_sha256": _array_sha256(captured),
        "telemetry_receipt": telemetry_receipt,
    }


def publish_meter_raw_no_replace(
    repository_root: str,
    plan: Mapping[str, Any],
    *,
    submitted_pcm: np.ndarray,
    captured_pcm: np.ndarray,
    telemetry: Mapping[str, Any],
    capture_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    admission = validate_meter_capture(
        plan,
        submitted_pcm=submitted_pcm,
        captured_pcm=captured_pcm,
        telemetry=telemetry,
    )
    metadata = {
        **dict(capture_metadata),
        "schema": STAGE2_METER_RAW_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "admission": admission,
        "canonical_training_eligible": False,
        "role": "level_gate_only_before_stage2_signal",
    }
    payload = _raw_npz_bytes(
        submitted_pcm=submitted_pcm,
        captured_pcm=captured_pcm,
        telemetry=telemetry,
        metadata=metadata,
    )
    target = _safe_relative_path(
        plan.get("artifacts", {}).get("meter_raw"), suffix=".npz", label="meter raw path"
    )
    published = publish_repository_bytes_noreplace(
        repository_root,
        target,
        payload,
        mode=0o600,
        preserve_recovery_link=True,
        recovery_tag="stage2_meter",
    )
    return {**published, "admission": admission}


def publish_stage2_signal_raw_no_replace(
    repository_root: str,
    plan: Mapping[str, Any],
    *,
    submitted_pcm: np.ndarray,
    captured_pcm: np.ndarray,
    telemetry: Mapping[str, Any],
    capture_metadata: Mapping[str, Any],
    meter_raw_sha256: str,
) -> dict[str, Any]:
    validate_submitted_pcm(plan, submitted_pcm)
    captured = np.asarray(captured_pcm)
    telemetry_receipt = validate_stage2_duplex_telemetry(
        telemetry,
        captured_adc_pcm=captured,
        expected_submitted_pcm=np.asarray(submitted_pcm),
    )
    if type(meter_raw_sha256) is not str or len(meter_raw_sha256) != 64:
        raise Stage2MeasurementError("same-setting PASS meter raw SHA-256이 필요합니다")
    clip_count = int(
        np.count_nonzero(np.abs(captured.astype(np.int64)) >= 2**31 - 1)
    )
    metadata = {
        **dict(capture_metadata),
        "schema": STAGE2_RAW_SCHEMA,
        "capture_adapter_schema": STAGE2_CAPTURE_ADAPTER_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "actual_submitted_pcm_sha256": plan["actual_submitted_pcm"]["sha256"],
        "captured_pcm_sha256": _array_sha256(captured),
        "meter_raw_sha256": meter_raw_sha256,
        "telemetry_receipt": telemetry_receipt,
        "adc_clip_count": clip_count,
        "same_capture_ps": True,
        "physical_acoustic_capture": True,
        "absolute_hardware_frame_identity_claimed": False,
        "hardware_counter_slip_zero_claimed": False,
        "relative_ps_lead_scope_only": True,
        "canonical_training_eligible": False,
        "analysis_required_before_relative_candidate": True,
        "transport_and_clip_preanalysis_pass": clip_count == 0,
    }
    payload = _raw_npz_bytes(
        submitted_pcm=submitted_pcm,
        captured_pcm=captured,
        telemetry=telemetry,
        metadata=metadata,
    )
    target = _safe_relative_path(
        plan.get("artifacts", {}).get("native_raw_capture"),
        suffix=".npz",
        label="native raw path",
    )
    published = publish_repository_bytes_noreplace(
        repository_root,
        target,
        payload,
        mode=0o600,
        preserve_recovery_link=True,
        recovery_tag="stage2_raw",
    )
    return {**published, "telemetry_receipt": telemetry_receipt}


def publish_stage2_failure_raw_no_replace(
    repository_root: str,
    plan: Mapping[str, Any],
    *,
    phase: str,
    planned_pcm: np.ndarray,
    failure: DuplexCaptureFailure,
    capture_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """meter/signal callback 실패도 actual prefix와 masks를 같은 generation에 보존한다."""

    if phase not in {"meter", "same_capture_ps"}:
        raise Stage2MeasurementError("partial capture phase가 meter/same_capture_ps가 아닙니다")
    planned = np.asarray(planned_pcm)
    expected = (
        plan.get("meter_submitted_pcm")
        if phase == "meter"
        else plan.get("actual_submitted_pcm")
    )
    if (
        not isinstance(expected, Mapping)
        or planned.dtype != np.dtype("<i2")
        or list(planned.shape) != expected.get("shape")
        or _array_sha256(planned) != expected.get("sha256")
    ):
        raise Stage2MeasurementError("partial capture planned PCM이 plan lineage와 다릅니다")
    captured = np.asarray(failure.captured_pcm)
    actual = np.asarray(failure.submitted_pcm)
    capture_mask = np.asarray(failure.capture_valid_mask)
    submitted_mask = np.asarray(failure.submitted_valid_mask)
    if (
        captured.dtype != np.dtype("<i4")
        or captured.shape != planned.shape
        or actual.dtype != np.dtype("<i2")
        or actual.shape != planned.shape
        or capture_mask.dtype != np.dtype("bool")
        or submitted_mask.dtype != np.dtype("bool")
        or capture_mask.shape != (len(planned),)
        or submitted_mask.shape != (len(planned),)
    ):
        raise Stage2MeasurementError("partial capture PCM/mask dtype 또는 shape가 잘못됐습니다")
    telemetry = dict(failure.telemetry)
    telemetry.update(
        {
            "actual_submitted_pcm": actual,
            "capture_valid_mask": capture_mask,
            "submitted_valid_mask": submitted_mask,
        }
    )
    if telemetry.get("schema") != DUPLEX_TELEMETRY_SCHEMA:
        raise Stage2MeasurementError("partial capture telemetry schema가 다릅니다")
    metadata = {
        **dict(capture_metadata),
        "schema": STAGE2_FAILURE_RAW_SCHEMA,
        "capture_adapter_schema": STAGE2_CAPTURE_ADAPTER_SCHEMA,
        "phase": phase,
        "plan_sha256": plan["plan_sha256"],
        "planned_pcm_sha256": expected["sha256"],
        "actual_submitted_pcm_sha256": _array_sha256(actual),
        "captured_pcm_sha256": _array_sha256(captured),
        "capture_valid_frames": int(np.count_nonzero(capture_mask)),
        "submitted_valid_frames": int(np.count_nonzero(submitted_mask)),
        "capture_exception": str(failure),
        "canonical_training_eligible": False,
        "partial_capture_never_promotable": True,
    }
    payload = _raw_npz_bytes(
        submitted_pcm=actual,
        captured_pcm=captured,
        telemetry=telemetry,
        metadata=metadata,
    )
    artifact_key = "meter_raw" if phase == "meter" else "native_raw_capture"
    target = _safe_relative_path(
        plan.get("artifacts", {}).get(artifact_key),
        suffix=".npz",
        label=f"partial {phase} raw path",
    )
    return publish_repository_bytes_noreplace(
        repository_root,
        target,
        payload,
        mode=0o600,
        preserve_recovery_link=True,
        recovery_tag=f"stage2_partial_{phase}",
    )


def _load_raw_archive(payload: bytes) -> dict[str, Any]:
    if type(payload) is not bytes:
        raise TypeError("Stage-2 raw payload는 bytes여야 합니다")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        required = {
            "submitted_pcm",
            "captured_pcm",
            "metadata_json",
            "telemetry_callback_sequence",
            "telemetry_callback_start_frames",
            "telemetry_callback_frame_counts",
            "telemetry_input_buffer_adc_time",
            "telemetry_output_buffer_dac_time",
            "telemetry_callback_current_time",
            "telemetry_callback_status_bitmask",
            "telemetry_capture_valid_mask",
            "telemetry_submitted_valid_mask",
        }
        if set(archive.files) != required:
            raise Stage2MeasurementError("Stage-2 raw NPZ key set이 exact하지 않습니다")
        submitted = np.asarray(archive["submitted_pcm"])
        captured = np.asarray(archive["captured_pcm"])
        metadata = json.loads(str(archive["metadata_json"].item()))
        telemetry_arrays = {
            name.removeprefix("telemetry_"): np.asarray(archive[name]).copy()
            for name in required
            if name.startswith("telemetry_")
        }
    if submitted.dtype != np.dtype("<i2") or captured.dtype != np.dtype("<i4"):
        raise Stage2MeasurementError("Stage-2 raw submitted/captured dtype가 exact하지 않습니다")
    if submitted.shape != captured.shape or submitted.ndim != 2 or submitted.shape[1] != 2:
        raise Stage2MeasurementError("Stage-2 raw submitted/captured shape가 다릅니다")
    return {
        "raw_npz_sha256": _sha256_bytes(payload),
        "submitted_pcm": submitted.copy(),
        "captured_pcm": captured.copy(),
        "metadata": metadata,
        "telemetry_arrays": telemetry_arrays,
    }


def load_stage2_raw_bytes(payload: bytes) -> dict[str, Any]:
    """offline analyzer가 immutable signal raw bytes를 exact dtype/schema로 읽는다."""

    loaded = _load_raw_archive(payload)
    metadata = loaded["metadata"]
    if not isinstance(metadata, dict) or metadata.get("schema") != STAGE2_RAW_SCHEMA:
        raise Stage2MeasurementError("Stage-2 raw metadata schema가 다릅니다")
    if metadata.get("actual_submitted_pcm_sha256") != _array_sha256(
        loaded["submitted_pcm"]
    ):
        raise Stage2MeasurementError("Stage-2 raw actual submitted SHA가 다릅니다")
    if metadata.get("captured_pcm_sha256") != _array_sha256(loaded["captured_pcm"]):
        raise Stage2MeasurementError("Stage-2 raw captured SHA가 다릅니다")
    return loaded


def load_stage2_meter_raw_bytes(
    plan: Mapping[str, Any], payload: bytes
) -> dict[str, Any]:
    """meter raw arrays/telemetry를 다시 계산해 저장 admission을 신뢰하지 않는다."""

    loaded = _load_raw_archive(payload)
    metadata = loaded["metadata"]
    if not isinstance(metadata, dict) or metadata.get("schema") != STAGE2_METER_RAW_SCHEMA:
        raise Stage2MeasurementError("Stage-2 meter raw metadata schema가 다릅니다")
    scalar = metadata.get("telemetry_scalar")
    if not isinstance(scalar, Mapping):
        raise Stage2MeasurementError("Stage-2 meter raw native telemetry scalar가 없습니다")
    telemetry = dict(scalar)
    telemetry.update(loaded["telemetry_arrays"])
    telemetry["actual_submitted_pcm"] = loaded["submitted_pcm"].copy()
    admission = validate_meter_capture(
        plan,
        submitted_pcm=loaded["submitted_pcm"],
        captured_pcm=loaded["captured_pcm"],
        telemetry=telemetry,
    )
    if metadata.get("admission") != admission:
        raise Stage2MeasurementError("meter raw 저장 admission과 raw 재계산 결과가 다릅니다")
    return {**loaded, "admission": admission}


__all__ = [
    "STAGE2_CAPTURE_ADAPTER_SCHEMA",
    "STAGE2_FAILURE_RAW_SCHEMA",
    "STAGE2_METER_RAW_SCHEMA",
    "STAGE2_TELEMETRY_RECEIPT_SCHEMA",
    "load_stage2_raw_bytes",
    "load_stage2_meter_raw_bytes",
    "publish_meter_raw_no_replace",
    "publish_stage2_failure_raw_no_replace",
    "publish_stage2_signal_raw_no_replace",
    "validate_meter_capture",
    "validate_stage2_duplex_telemetry",
]
