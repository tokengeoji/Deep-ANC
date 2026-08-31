"""Stage-2 v2 two-stream raw의 no-replace publisher와 재검증 경계.

이 모듈은 audio backend/device를 열지 않는다. caller가 받은 actual submitted/captured
PCM과 callback telemetry를 exact canonical plan의 diagnostic/PS slice에 결속한다.
Diagnostic raw는 durable no-replace 발행된 뒤 다시 읽어 분석해야 하며, 그 raw SHA를
포함한 PASS receipt가 durable 발행되기 전에는 PS stream authorization을 내주지 않는다.
"""

from __future__ import annotations

import hashlib
import io
import json
from typing import Any, Mapping

import numpy as np

from deep_anc.audio_duplex_stage2 import DUPLEX_TELEMETRY_SCHEMA, DuplexCaptureFailure
from deep_anc.data.repository_fd import (
    RepositoryFileGuard,
    canonical_relative_path,
    publish_repository_bytes_noreplace,
)

from .stage2_2khz_live import validate_stage2_duplex_telemetry
from .stage2_2khz_measurement_v2 import (
    Stage2MeasurementV2Error,
    _array_sha256,
    _payload_sha256,
    validate_stage2_v2_live_safe_fallback_plan,
)
from .stage2_2khz_level_contract import (
    build_stage2_physical_operating_level_evidence,
    validate_stage2_physical_operating_level_evidence,
)


PHASE_RAW_SCHEMA = "stage2_2khz_two_stream_phase_raw_v2"
PARTIAL_RAW_SCHEMA = "stage2_2khz_two_stream_partial_raw_v2"
DIAGNOSTIC_AUTHORIZATION_SCHEMA = "stage2_2khz_diagnostic_ps_authorization_v2"
_PHASES = ("diagnostic", "ps")
_TELEMETRY_ARRAY_NAMES = (
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


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _phase_contract(plan: Mapping[str, Any], phase: str) -> tuple[dict[str, Any], int, int]:
    if phase not in _PHASES:
        raise Stage2MeasurementV2Error("Stage-2 v2 phase는 diagnostic/ps여야 합니다")
    key = "diagnostic_stream" if phase == "diagnostic" else "ps_stream"
    row = dict(plan["live_phase_contract"][key])
    start = int(row["logical_plan_start_frame"])
    stop = int(row["logical_plan_stop_frame"])
    if (
        int(row["local_start_frame"]) != 0
        or int(row["local_stop_frame"]) != stop - start
        or int(row["stream_index"]) != (1 if phase == "diagnostic" else 2)
    ):
        raise Stage2MeasurementV2Error("Stage-2 v2 phase stream boundary가 canonical이 아닙니다")
    return row, start, stop


def _phase_path(plan: Mapping[str, Any], phase: str) -> str:
    key = "diagnostic_phase_raw" if phase == "diagnostic" else "ps_phase_raw"
    value = canonical_relative_path(plan["artifacts"][key], label=f"Stage-2 {phase} raw")
    if not value.endswith(".npz"):
        raise Stage2MeasurementV2Error("Stage-2 phase raw target은 .npz여야 합니다")
    return value


def _serialize_phase_raw(
    *,
    submitted: np.ndarray,
    captured: np.ndarray,
    telemetry: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> bytes:
    scalar = {
        key: value
        for key, value in telemetry.items()
        if key not in _TELEMETRY_ARRAY_NAMES and key != "actual_submitted_pcm"
    }
    if any(isinstance(value, np.ndarray) for value in scalar.values()):
        raise Stage2MeasurementV2Error("Stage-2 telemetry scalar에 ndarray가 남았습니다")
    arrays = {
        f"telemetry_{name}": np.asarray(telemetry[name])
        for name in _TELEMETRY_ARRAY_NAMES
    }
    sealed = {**dict(metadata), "telemetry_scalar": scalar}
    output = io.BytesIO()
    np.savez(
        output,
        submitted_pcm=np.asarray(submitted, dtype="<i2"),
        captured_pcm=np.asarray(captured, dtype="<i4"),
        metadata_json=np.asarray(_canonical_json_bytes(sealed).decode("utf-8")),
        **arrays,
    )
    return output.getvalue()


def publish_stage2_v2_phase_raw_no_replace(
    repository_root: str,
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    phase: str,
    actual_submitted_pcm: np.ndarray,
    captured_pcm: np.ndarray,
    telemetry: Mapping[str, Any],
    capture_metadata: Mapping[str, Any],
    diagnostic_authorization_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    canonical, full = validate_stage2_v2_live_safe_fallback_plan(
        plan, full_submitted_pcm
    )
    stream, start, stop = _phase_contract(canonical, phase)
    expected = np.asarray(full[start:stop])
    submitted = np.asarray(actual_submitted_pcm)
    captured = np.asarray(captured_pcm)
    if (
        submitted.dtype != np.dtype("<i2")
        or submitted.shape != expected.shape
        or not np.array_equal(submitted, expected)
    ):
        raise Stage2MeasurementV2Error("Stage-2 phase actual submitted bytes가 canonical slice와 다릅니다")
    if captured.dtype != np.dtype("<i4") or captured.shape != expected.shape:
        raise Stage2MeasurementV2Error("Stage-2 phase captured PCM은 exact int32 slice여야 합니다")
    transport = validate_stage2_duplex_telemetry(
        telemetry,
        captured_adc_pcm=captured,
        expected_submitted_pcm=submitted,
    )
    if phase == "diagnostic":
        if diagnostic_authorization_ref is not None:
            raise Stage2MeasurementV2Error("diagnostic phase는 선행 authorization을 가질 수 없습니다")
    else:
        if not isinstance(diagnostic_authorization_ref, Mapping):
            raise Stage2MeasurementV2Error("PS phase raw에는 diagnostic authorization ref가 필요합니다")
        if set(diagnostic_authorization_ref) != {"path", "sha256"}:
            raise Stage2MeasurementV2Error("diagnostic authorization ref key가 exact하지 않습니다")
        # PS raw가 durable authorization의 모양만 복사해 authority를 위조하지 못하게
        # 실제 repository bytes와 그 안의 diagnostic raw/분석 receipt chain을 stream
        # publication 전에 다시 연다. 이 검증은 diagnostic raw만 snapshot하므로 재귀하지
        # 않는다.
        validate_published_diagnostic_authorization(
            repository_root,
            canonical,
            full,
            diagnostic_authorization_ref,
        )
    clip_count = int(np.count_nonzero(np.abs(captured.astype(np.int64)) >= 2**31 - 1))
    metadata = {
        **dict(capture_metadata),
        "schema": PHASE_RAW_SCHEMA,
        "phase": phase,
        "signal_plan_sha256": canonical["canonical_payload_sha256"],
        "operating_level_plan_sha256": canonical["operating_level_plan"][
            "canonical_payload_sha256"
        ],
        "stream_contract": stream,
        "logical_plan_frame_range": [start, stop],
        "actual_submitted_pcm_sha256": _array_sha256(submitted),
        "captured_pcm_sha256": _array_sha256(captured),
        "transport_receipt": transport,
        "adc_clip_count": clip_count,
        "diagnostic_authorization_ref": (
            None if diagnostic_authorization_ref is None else dict(diagnostic_authorization_ref)
        ),
        "single_stream_clock_claimed": False,
        "clock_scope": "none" if phase == "diagnostic" else "ps_stream_local_coordinates",
        "canonical_training_eligible": False,
    }
    payload = _serialize_phase_raw(
        submitted=submitted,
        captured=captured,
        telemetry=telemetry,
        metadata=metadata,
    )
    published = publish_repository_bytes_noreplace(
        repository_root,
        _phase_path(canonical, phase),
        payload,
        mode=0o600,
        preserve_recovery_link=True,
        recovery_tag=f"stage2_v2_{phase}",
    )
    return {**published, "phase": phase, "transport_receipt": transport}


def publish_stage2_v2_partial_raw_no_replace(
    repository_root: str,
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    phase: str,
    failure: DuplexCaptureFailure,
    capture_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """callback/stream 실패도 해당 phase target에 보존하고 재시도를 구조적으로 막는다."""

    canonical, full = validate_stage2_v2_live_safe_fallback_plan(
        plan, full_submitted_pcm
    )
    stream, start, stop = _phase_contract(canonical, phase)
    expected = np.asarray(full[start:stop])
    actual = np.asarray(failure.submitted_pcm)
    captured = np.asarray(failure.captured_pcm)
    if (
        actual.dtype != np.dtype("<i2")
        or captured.dtype != np.dtype("<i4")
        or actual.shape != expected.shape
        or captured.shape != expected.shape
        or np.asarray(failure.capture_valid_mask).shape != (len(expected),)
        or np.asarray(failure.submitted_valid_mask).shape != (len(expected),)
    ):
        raise Stage2MeasurementV2Error("Stage-2 partial phase raw shape/dtype가 잘못됐습니다")
    telemetry = dict(failure.telemetry)
    telemetry.update(
        {
            "actual_submitted_pcm": actual,
            "capture_valid_mask": np.asarray(failure.capture_valid_mask),
            "submitted_valid_mask": np.asarray(failure.submitted_valid_mask),
        }
    )
    metadata = {
        **dict(capture_metadata),
        "schema": PARTIAL_RAW_SCHEMA,
        "phase": phase,
        "signal_plan_sha256": canonical["canonical_payload_sha256"],
        "stream_contract": stream,
        "logical_plan_frame_range": [start, stop],
        "planned_submitted_pcm_sha256": _array_sha256(expected),
        "actual_submitted_pcm_sha256": _array_sha256(actual),
        "captured_pcm_sha256": _array_sha256(captured),
        "capture_valid_frames": int(np.count_nonzero(failure.capture_valid_mask)),
        "submitted_valid_frames": int(np.count_nonzero(failure.submitted_valid_mask)),
        "capture_exception": str(failure),
        "partial_capture_never_promotable": True,
        "automatic_retry_allowed": False,
        "canonical_training_eligible": False,
    }
    payload = _serialize_phase_raw(
        submitted=actual,
        captured=captured,
        telemetry=telemetry,
        metadata=metadata,
    )
    return publish_repository_bytes_noreplace(
        repository_root,
        _phase_path(canonical, phase),
        payload,
        mode=0o600,
        preserve_recovery_link=True,
        recovery_tag=f"stage2_v2_partial_{phase}",
    )


def _load_phase_payload(payload: bytes) -> dict[str, Any]:
    if type(payload) is not bytes:
        raise TypeError("Stage-2 phase raw payload는 bytes여야 합니다")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        required = {
            "submitted_pcm",
            "captured_pcm",
            "metadata_json",
            *(f"telemetry_{name}" for name in _TELEMETRY_ARRAY_NAMES),
        }
        if set(archive.files) != required:
            raise Stage2MeasurementV2Error("Stage-2 phase raw NPZ key set이 exact하지 않습니다")
        submitted = np.asarray(archive["submitted_pcm"]).copy()
        captured = np.asarray(archive["captured_pcm"]).copy()
        metadata = json.loads(str(archive["metadata_json"].item()))
        telemetry_arrays = {
            name: np.asarray(archive[f"telemetry_{name}"]).copy()
            for name in _TELEMETRY_ARRAY_NAMES
        }
    return {
        "raw_npz_sha256": hashlib.sha256(payload).hexdigest(),
        "submitted_pcm": submitted,
        "captured_pcm": captured,
        "metadata": metadata,
        "telemetry_arrays": telemetry_arrays,
    }


def load_stage2_v2_phase_raw_bytes(
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    phase: str,
    payload: bytes,
) -> dict[str, Any]:
    canonical, full = validate_stage2_v2_live_safe_fallback_plan(
        plan, full_submitted_pcm
    )
    loaded = _load_phase_payload(payload)
    metadata = loaded["metadata"]
    stream, start, stop = _phase_contract(canonical, phase)
    expected = np.asarray(full[start:stop])
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema") != PHASE_RAW_SCHEMA
        or metadata.get("phase") != phase
        or metadata.get("signal_plan_sha256") != canonical["canonical_payload_sha256"]
        or metadata.get("operating_level_plan_sha256")
        != canonical["operating_level_plan"]["canonical_payload_sha256"]
        or metadata.get("stream_contract") != stream
        or metadata.get("logical_plan_frame_range") != [start, stop]
    ):
        raise Stage2MeasurementV2Error("Stage-2 phase raw metadata/stream binding이 다릅니다")
    if (
        loaded["submitted_pcm"].dtype != np.dtype("<i2")
        or loaded["captured_pcm"].dtype != np.dtype("<i4")
        or not np.array_equal(loaded["submitted_pcm"], expected)
        or loaded["captured_pcm"].shape != expected.shape
        or metadata.get("actual_submitted_pcm_sha256")
        != _array_sha256(loaded["submitted_pcm"])
        or metadata.get("captured_pcm_sha256") != _array_sha256(loaded["captured_pcm"])
    ):
        raise Stage2MeasurementV2Error("Stage-2 phase raw PCM bytes/SHA가 다릅니다")
    scalar = metadata.get("telemetry_scalar")
    if not isinstance(scalar, Mapping) or scalar.get("schema") != DUPLEX_TELEMETRY_SCHEMA:
        raise Stage2MeasurementV2Error("Stage-2 phase raw telemetry scalar가 없습니다")
    telemetry = dict(scalar)
    telemetry.update(loaded["telemetry_arrays"])
    telemetry["actual_submitted_pcm"] = loaded["submitted_pcm"].copy()
    transport = validate_stage2_duplex_telemetry(
        telemetry,
        captured_adc_pcm=loaded["captured_pcm"],
        expected_submitted_pcm=loaded["submitted_pcm"],
    )
    if metadata.get("transport_receipt") != transport:
        raise Stage2MeasurementV2Error("Stage-2 phase stored/recomputed transport receipt가 다릅니다")
    if int(metadata.get("adc_clip_count", -1)) != 0:
        raise Stage2MeasurementV2Error("Stage-2 phase raw ADC clip count가 0이 아닙니다")
    authorization_ref = metadata.get("diagnostic_authorization_ref")
    if phase == "diagnostic":
        if authorization_ref is not None:
            raise Stage2MeasurementV2Error("diagnostic raw가 선행 authorization을 주장합니다")
    elif (
        not isinstance(authorization_ref, Mapping)
        or set(authorization_ref) != {"path", "sha256"}
        or not all(isinstance(authorization_ref[key], str) for key in ("path", "sha256"))
        or len(str(authorization_ref["sha256"])) != 64
    ):
        raise Stage2MeasurementV2Error("PS raw diagnostic authorization ref가 exact하지 않습니다")
    return {
        **loaded,
        "transport_receipt": transport,
        "diagnostic_authorization_ref": (
            None if authorization_ref is None else dict(authorization_ref)
        ),
    }


def snapshot_published_stage2_v2_phase(
    repository_root: str,
    publication: Mapping[str, Any],
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    phase: str,
) -> dict[str, Any]:
    if set(publication) < {"path", "sha256"}:
        raise Stage2MeasurementV2Error("Stage-2 publication path/SHA가 없습니다")
    with RepositoryFileGuard(
        repository_root, str(publication["path"]), label=f"Stage-2 {phase} raw"
    ) as guard:
        if guard.sha256 != publication["sha256"]:
            raise Stage2MeasurementV2Error("Stage-2 published phase raw SHA가 다릅니다")
        loaded = load_stage2_v2_phase_raw_bytes(
            plan,
            full_submitted_pcm,
            phase=phase,
            payload=guard.bytes,
        )
        guard.verify()
    if phase == "ps":
        embedded_ref = loaded.get("diagnostic_authorization_ref")
        if not isinstance(embedded_ref, Mapping):
            raise Stage2MeasurementV2Error("PS raw embedded authorization ref가 없습니다")
        authorization_chain = validate_published_diagnostic_authorization(
            repository_root,
            plan,
            full_submitted_pcm,
            embedded_ref,
        )
        loaded["diagnostic_authorization_chain"] = authorization_chain
    loaded["artifact_ref"] = {
        "path": str(publication["path"]),
        "sha256": str(publication["sha256"]),
    }
    return loaded


def seal_and_publish_diagnostic_authorization(
    repository_root: str,
    plan: Mapping[str, Any],
    *,
    diagnostic_analysis_receipt: Mapping[str, Any],
    diagnostic_raw: Mapping[str, Any],
) -> dict[str, Any]:
    raw_ref = diagnostic_raw.get("artifact_ref")
    if not isinstance(raw_ref, Mapping) or set(raw_ref) != {"path", "sha256"}:
        raise Stage2MeasurementV2Error("diagnostic raw artifact ref가 없습니다")
    receipt = dict(diagnostic_analysis_receipt)
    source_sha = receipt.pop("canonical_payload_sha256", None)
    if source_sha != _payload_sha256(receipt):
        raise Stage2MeasurementV2Error("diagnostic analysis receipt canonical SHA가 유효하지 않습니다")
    if (
        receipt.get("passed") is not True
        or receipt.get("ps_phase_may_start") is not True
        or receipt.get("diagnostic_captured_snapshot_sha256")
        != _array_sha256(diagnostic_raw["captured_pcm"])
    ):
        raise Stage2MeasurementV2Error("PASS diagnostic raw-derived receipt가 아닙니다")
    authorization: dict[str, Any] = {
        "schema": DIAGNOSTIC_AUTHORIZATION_SCHEMA,
        "status": "PASS_DIAGNOSTIC_RAW_DURABLY_PUBLISHED_PS_MAY_START",
        "signal_plan_sha256": plan["canonical_payload_sha256"],
        "operating_level_plan_sha256": plan["operating_level_plan"][
            "canonical_payload_sha256"
        ],
        "diagnostic_raw_artifact": dict(raw_ref),
        "diagnostic_submitted_pcm_sha256": _array_sha256(
            diagnostic_raw["submitted_pcm"]
        ),
        "diagnostic_captured_pcm_sha256": _array_sha256(
            diagnostic_raw["captured_pcm"]
        ),
        "diagnostic_analysis_receipt": {**receipt, "canonical_payload_sha256": source_sha},
        "diagnostic_analysis_receipt_sha256": source_sha,
        "phase_stream_contract": plan["live_phase_contract"],
        "physical_49_98_pcm_diagnostic_authority": True,
        "ps_phase_may_start": True,
        "canonical_training_eligible": False,
    }
    authorization["canonical_payload_sha256"] = _payload_sha256(authorization)
    payload = _canonical_json_bytes(authorization) + b"\n"
    target = canonical_relative_path(
        plan["artifacts"]["diagnostic_analysis_receipt"],
        label="Stage-2 diagnostic authorization",
    )
    published = publish_repository_bytes_noreplace(
        repository_root,
        target,
        payload,
        mode=0o600,
        preserve_recovery_link=True,
        recovery_tag="stage2_v2_diagnostic_authorization",
    )
    return {**published, "authorization": authorization}


def validate_published_diagnostic_authorization(
    repository_root: str,
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    publication: Mapping[str, Any],
) -> dict[str, Any]:
    with RepositoryFileGuard(
        repository_root,
        str(publication["path"]),
        label="Stage-2 diagnostic authorization",
    ) as guard:
        if guard.sha256 != publication["sha256"]:
            raise Stage2MeasurementV2Error("diagnostic authorization file SHA가 다릅니다")
        try:
            value = json.loads(guard.bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Stage2MeasurementV2Error("diagnostic authorization JSON이 유효하지 않습니다") from exc
        guard.verify()
    if not isinstance(value, Mapping):
        raise Stage2MeasurementV2Error("diagnostic authorization payload가 mapping이 아닙니다")
    digest = value.get("canonical_payload_sha256")
    if digest != _payload_sha256({key: item for key, item in value.items() if key != "canonical_payload_sha256"}):
        raise Stage2MeasurementV2Error("diagnostic authorization canonical SHA가 다릅니다")
    if (
        value.get("schema") != DIAGNOSTIC_AUTHORIZATION_SCHEMA
        or value.get("signal_plan_sha256") != plan["canonical_payload_sha256"]
        or value.get("phase_stream_contract") != plan["live_phase_contract"]
        or value.get("ps_phase_may_start") is not True
        or value.get("physical_49_98_pcm_diagnostic_authority") is not True
    ):
        raise Stage2MeasurementV2Error("diagnostic authorization 의미 gate가 실패했습니다")
    raw_ref = value.get("diagnostic_raw_artifact")
    if not isinstance(raw_ref, Mapping):
        raise Stage2MeasurementV2Error("diagnostic authorization raw ref가 없습니다")
    raw = snapshot_published_stage2_v2_phase(
        repository_root,
        raw_ref,
        plan,
        full_submitted_pcm,
        phase="diagnostic",
    )
    embedded = value.get("diagnostic_analysis_receipt")
    from .stage2_2khz_analysis_v2 import analyse_stage2_v2_diagnostic_preflight

    recomputed = analyse_stage2_v2_diagnostic_preflight(
        plan,
        full_submitted_pcm,
        raw["captured_pcm"],
        transport_counters={"xrun": 0, "clip": 0, "callback_status": 0},
    )
    if (
        not isinstance(embedded, Mapping)
        or dict(embedded) != recomputed
        or embedded.get("canonical_payload_sha256")
        != value.get("diagnostic_analysis_receipt_sha256")
        or embedded.get("diagnostic_captured_snapshot_sha256")
        != _array_sha256(raw["captured_pcm"])
        or embedded.get("passed") is not True
        or embedded.get("ps_phase_may_start") is not True
    ):
        raise Stage2MeasurementV2Error("diagnostic authorization embedded receipt/raw가 다릅니다")
    return {"authorization": dict(value), "diagnostic_raw": raw}


def _guard_exact_artifact_ref(
    repository_root: str, value: Mapping[str, Any], *, label: str
) -> bytes:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise Stage2MeasurementV2Error(f"{label} artifact ref key가 exact하지 않습니다")
    with RepositoryFileGuard(
        repository_root, str(value["path"]), label=label
    ) as guard:
        if guard.sha256 != value["sha256"]:
            raise Stage2MeasurementV2Error(f"{label} artifact SHA가 다릅니다")
        payload = guard.bytes
        guard.verify()
    return payload


def _fresh_meter_refs_from_phase_metadata(
    diagnostic_metadata: Mapping[str, Any], ps_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    diagnostic_meter = diagnostic_metadata.get("fresh_meter")
    ps_meter = ps_metadata.get("fresh_meter")
    if (
        not isinstance(diagnostic_meter, Mapping)
        or dict(diagnostic_meter) != dict(ps_meter or {})
        or set(diagnostic_meter)
        != {
            "path",
            "sha256",
            "receipt_path",
            "receipt_sha256",
            "capture_id",
            "completed_at_utc",
            "age_seconds",
            "meter_ch0_dbfs",
            "freshness_max_seconds",
            "resolved_devices",
            "physical_fingerprint",
            "hardware_identity",
            "calibration_evidence",
        }
    ):
        raise Stage2MeasurementV2Error("두 phase의 official fresh meter binding이 exact하지 않습니다")
    if int(diagnostic_meter["freshness_max_seconds"]) != 600:
        raise Stage2MeasurementV2Error("Stage-2 meter freshness policy가 official 600초가 아닙니다")
    return dict(diagnostic_meter)


def seal_and_publish_physical_operating_level_evidence(
    repository_root: str,
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    diagnostic_authorization_publication: Mapping[str, Any],
    ps_raw_publication: Mapping[str, Any],
) -> dict[str, Any]:
    """검증한 meter→diagnostic→PS immutable DAG에서만 49/98 evidence를 발행한다."""

    canonical, full = validate_stage2_v2_live_safe_fallback_plan(
        plan, full_submitted_pcm
    )
    authorization = validate_published_diagnostic_authorization(
        repository_root,
        canonical,
        full,
        diagnostic_authorization_publication,
    )
    ps_raw = snapshot_published_stage2_v2_phase(
        repository_root,
        ps_raw_publication,
        canonical,
        full,
        phase="ps",
    )
    embedded_ref = ps_raw["diagnostic_authorization_ref"]
    expected_auth_ref = {
        "path": str(diagnostic_authorization_publication["path"]),
        "sha256": str(diagnostic_authorization_publication["sha256"]),
    }
    if embedded_ref != expected_auth_ref:
        raise Stage2MeasurementV2Error("PS raw가 지정 diagnostic authorization을 소비하지 않았습니다")
    diagnostic_raw = authorization["diagnostic_raw"]
    diagnostic_metadata = diagnostic_raw["metadata"]
    ps_metadata = ps_raw["metadata"]
    capture_id = diagnostic_metadata.get("capture_id")
    if (
        not isinstance(capture_id, str)
        or not capture_id
        or ps_metadata.get("capture_id") != capture_id
        or diagnostic_metadata.get("measurement_git_authority")
        != ps_metadata.get("measurement_git_authority")
        or diagnostic_metadata.get("hardware_config_sha256")
        != ps_metadata.get("hardware_config_sha256")
    ):
        raise Stage2MeasurementV2Error("diagnostic/PS phase physical capture identity가 다릅니다")
    meter = _fresh_meter_refs_from_phase_metadata(diagnostic_metadata, ps_metadata)
    meter_raw_ref = {"path": meter["path"], "sha256": meter["sha256"]}
    meter_receipt_ref = {
        "path": meter["receipt_path"],
        "sha256": meter["receipt_sha256"],
    }
    calibration_ref = dict(meter["calibration_evidence"])
    _guard_exact_artifact_ref(repository_root, meter_raw_ref, label="Stage-2 meter raw")
    _guard_exact_artifact_ref(
        repository_root, meter_receipt_ref, label="Stage-2 meter receipt"
    )
    _guard_exact_artifact_ref(
        repository_root, calibration_ref, label="Stage-2 calibration evidence"
    )
    # Freshness는 capture open 직전에 CLI가 강제한다. durable evidence에서는 같은 raw와
    # receipt/hardware의 내용 유효성을 official validator로 재검산하되 시간이 지난 뒤에도
    # offline 분석이 가능하도록 freshness만 다시 요구하지 않는다.
    from .measurement_level import validate_bootstrap_meter_raw

    verified_meter = validate_bootstrap_meter_raw(
        meter_raw_ref["path"],
        repository_root=repository_root,
        expected_hardware_identity=dict(meter["hardware_identity"]),
        require_fresh=False,
    )
    if verified_meter["sha256"] != meter_raw_ref["sha256"]:
        raise Stage2MeasurementV2Error("official meter validator와 bound raw SHA가 다릅니다")
    evidence = build_stage2_physical_operating_level_evidence(
        signal_plan_sha256=canonical["canonical_payload_sha256"],
        capture_id=capture_id,
        hardware_identity=meter["hardware_identity"],
        meter_raw_artifact=meter_raw_ref,
        meter_receipt_artifact=meter_receipt_ref,
        calibration_evidence_artifact=calibration_ref,
        diagnostic_raw_artifact=authorization["authorization"]["diagnostic_raw_artifact"],
        diagnostic_authorization_artifact=expected_auth_ref,
        ps_raw_artifact={
            "path": str(ps_raw_publication["path"]),
            "sha256": str(ps_raw_publication["sha256"]),
        },
    )
    target = canonical_relative_path(
        canonical["artifacts"]["source_operating_level"],
        label="Stage-2 physical operating level evidence",
    )
    published = publish_repository_bytes_noreplace(
        repository_root,
        target,
        _canonical_json_bytes(evidence) + b"\n",
        mode=0o600,
        preserve_recovery_link=True,
        recovery_tag="stage2_v2_physical_operating_level",
    )
    return {**published, "evidence": evidence}


def validate_published_physical_operating_level_evidence(
    repository_root: str,
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    publication: Mapping[str, Any],
) -> dict[str, Any]:
    """재부팅 후에도 모든 lineage bytes를 다시 열어 physical level evidence를 복원한다."""

    canonical, full = validate_stage2_v2_live_safe_fallback_plan(
        plan, full_submitted_pcm
    )
    payload = _guard_exact_artifact_ref(
        repository_root, publication, label="Stage-2 physical operating level evidence"
    )
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2MeasurementV2Error("physical operating level JSON이 유효하지 않습니다") from exc
    try:
        evidence = validate_stage2_physical_operating_level_evidence(decoded)
    except (TypeError, ValueError) as exc:
        raise Stage2MeasurementV2Error(str(exc)) from exc
    if evidence["signal_plan_sha256"] != canonical["canonical_payload_sha256"]:
        raise Stage2MeasurementV2Error("physical operating level signal plan SHA가 다릅니다")
    lineage = evidence["artifact_lineage"]
    authorization = validate_published_diagnostic_authorization(
        repository_root, canonical, full, lineage["diagnostic_authorization"]
    )
    ps_raw = snapshot_published_stage2_v2_phase(
        repository_root,
        lineage["ps_phase_raw"],
        canonical,
        full,
        phase="ps",
    )
    if (
        authorization["diagnostic_raw"]["artifact_ref"]
        != lineage["diagnostic_phase_raw"]
        or ps_raw["diagnostic_authorization_ref"]
        != lineage["diagnostic_authorization"]
        or authorization["diagnostic_raw"]["metadata"].get("capture_id")
        != evidence["capture_id"]
        or ps_raw["metadata"].get("capture_id") != evidence["capture_id"]
    ):
        raise Stage2MeasurementV2Error("physical operating level raw/auth capture DAG가 다릅니다")
    for key, label in (
        ("meter_raw", "Stage-2 meter raw"),
        ("meter_receipt", "Stage-2 meter receipt"),
        ("calibration_evidence", "Stage-2 calibration evidence"),
    ):
        _guard_exact_artifact_ref(repository_root, lineage[key], label=label)
    from .measurement_level import validate_bootstrap_meter_raw

    verified_meter = validate_bootstrap_meter_raw(
        lineage["meter_raw"]["path"],
        repository_root=repository_root,
        expected_hardware_identity=evidence["hardware_identity"],
        require_fresh=False,
    )
    if verified_meter["sha256"] != lineage["meter_raw"]["sha256"]:
        raise Stage2MeasurementV2Error("physical operating level meter raw SHA가 다릅니다")
    return {
        "evidence": evidence,
        "diagnostic_authorization_chain": authorization,
        "ps_raw": ps_raw,
    }


__all__ = [
    "DIAGNOSTIC_AUTHORIZATION_SCHEMA",
    "PARTIAL_RAW_SCHEMA",
    "PHASE_RAW_SCHEMA",
    "load_stage2_v2_phase_raw_bytes",
    "publish_stage2_v2_partial_raw_no_replace",
    "publish_stage2_v2_phase_raw_no_replace",
    "seal_and_publish_physical_operating_level_evidence",
    "seal_and_publish_diagnostic_authorization",
    "snapshot_published_stage2_v2_phase",
    "validate_published_diagnostic_authorization",
    "validate_published_physical_operating_level_evidence",
]
