"""Stage-2 output-master P/S v3의 fail-closed production boundary.

USB DAC output과 APE input은 서로 다른 hardware clock domain이다. 따라서 input
raw의 frame ``n``을 output frame ``n``으로 취급한 legacy combined duplex raw는
P/S 또는 training authority가 될 수 없다.

현재 구현은 다음 두 가지를 의도적으로 분리한다.

* output-master transport가 보존해야 할 가변 길이 input/output raw 구조
* 그 raw를 plant로 승격하기 전에 필요한 clock/resampling/DPSS 분석 권한

진단용 output-master clock PASS만으로 P/S stream을 열지 않는다. 아래 admission은
durable diagnostic raw와 clock receipt를 독립적으로 다시 검증하더라도, 아직 구현·검증되지
않은 clock-corrected linearity 및 P/S-local resampling/fit adapter가 하나라도 남으면 항상
``ps_stream_may_open=False``로 닫힌다. 이 모듈은 audio backend/device를 열지 않는다.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping

import numpy as np
from scipy.interpolate import CubicSpline

from deep_anc.audio_duplex_stage2 import OutputMasterCaptureFailure
from deep_anc.data.repository_fd import (
    RepositoryFileGuard,
    canonical_relative_path,
    publish_repository_bytes_noreplace,
)

from .stage2_2khz_diagnostic_clock import DIAGNOSTIC_CLOCK_SCHEMA
from .stage2_2khz_analysis_v2 import (
    analyse_stage2_v2_capture,
    analyse_stage2_v2_diagnostic_preflight,
)
from .stage2_2khz_clock import CLOCK_SCHEMA, estimate_stage2_ps_local_clock
from .stage2_2khz_contract import Stage2TwoKilohertzContract
from .stage2_2khz_level_contract import (
    build_stage2_physical_operating_level_evidence,
    validate_stage2_physical_operating_level_evidence,
)
from .stage2_2khz_measurement_v2 import (
    SAMPLE_RATE,
    Stage2MeasurementV2Error,
    _array_sha256,
    _payload_sha256,
    validate_stage2_v2_live_safe_fallback_plan,
)
from .stage2_2khz_output_master_diagnostic import (
    OUTPUT_MASTER_CLOCK_RECEIPT_SCHEMA,
    POST_ROLL_FRAMES,
    PRE_ROLL_FRAMES,
    output_master_session_targets,
    snapshot_published_output_master_raw,
    validate_output_master_success_telemetry,
)


PS_V3_PLAN_SCHEMA = "stage2_2khz_output_master_ps_v3_plan_v1"
PS_V3_RAW_STRUCTURE_SCHEMA = "stage2_2khz_output_master_ps_v3_raw_structure_v1"
PS_V3_ADMISSION_SCHEMA = "stage2_2khz_output_master_ps_v3_admission_v1"
PS_V3_EXECUTION_SCHEMA = "stage2_2khz_output_master_ps_v3_execution_v1"
DIAGNOSTIC_LINEARITY_V3_SCHEMA = (
    "stage2_2khz_output_master_diagnostic_linearity_v3"
)
PS_DAC_GRID_TRANSFORM_SCHEMA = (
    "stage2_2khz_output_master_ps_dac_grid_transform_v1"
)
PS_CLOCK_ADAPTER_SCHEMA = "stage2_2khz_output_master_ps_clock_adapter_v1"
PS_PHYSICAL_ANALYSIS_V3_SCHEMA = (
    "stage2_2khz_output_master_ps_physical_analysis_v3"
)
DIAGNOSTIC_LINEARITY_PUBLICATION_SCHEMA = (
    "stage2_2khz_output_master_diagnostic_linearity_publication_v1"
)
PS_V3_RAW_SCHEMA = "stage2_2khz_output_master_ps_raw_v3"
PS_V3_PARTIAL_RAW_SCHEMA = "stage2_2khz_output_master_ps_partial_raw_v3"
PS_V3_PHYSICAL_LEVEL_PUBLICATION_SCHEMA = (
    "stage2_2khz_output_master_ps_physical_level_publication_v1"
)
PS_V3_SESSION_ROOT = "results/stage2_2khz_output_master_ps_v3"
PS_V3_RAW_LEAF = "ps_raw.npz"
PS_V3_PARTIAL_RAW_LEAF = "ps_partial_raw.npz"
PS_V3_PHYSICAL_LEVEL_LEAF = "source_operating_level.json"
DIAGNOSTIC_LINEARITY_LEAF = "linearity_v3.json"

# 이 항목들은 단순 status=true JSON으로 대체할 수 없다. 각 adapter가 durable raw bytes와
# 실제 계산 receipt를 결속한 뒤 이 tuple에서 제거하는 코드 변경/회귀가 있어야 한다.
_UNIMPLEMENTED_AUTHORITY_BLOCKERS: tuple[str, ...] = ()

_FRESH_METER_KEYS = {
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


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit_sha(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _artifact_ref(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise Stage2MeasurementV2Error(f"{label} artifact ref key가 exact하지 않습니다")
    try:
        path = canonical_relative_path(value.get("path"), label=label)
    except (TypeError, ValueError) as exc:
        raise Stage2MeasurementV2Error(f"{label} path가 canonical이 아닙니다") from exc
    digest = value.get("sha256")
    if not _is_sha256(digest):
        raise Stage2MeasurementV2Error(f"{label} SHA-256이 exact하지 않습니다")
    return {"path": path, "sha256": str(digest)}


def _validate_measurement_git_authority(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage2MeasurementV2Error("diagnostic measurement Git authority가 없습니다")
    authority = dict(value)
    hashes = authority.get("critical_file_sha256")
    head = authority.get("head")
    if (
        authority.get("schema")
        != "stage2_2khz_output_master_origin_dev_exact_bundle_v1"
        or authority.get("branch") != "dev"
        or not _is_commit_sha(head)
        or authority.get("origin_dev") != head
        or not isinstance(hashes, Mapping)
        or not hashes
        or any(
            type(path) is not str
            or not path
            or not _is_sha256(digest)
            for path, digest in hashes.items()
        )
    ):
        raise Stage2MeasurementV2Error(
            "diagnostic measurement Git authority가 attached exact origin/dev bundle이 아닙니다"
        )
    return authority


def _validate_physical_capture_metadata(metadata: Any) -> dict[str, Any]:
    """Raw에 durable하게 박힌 route/level/Git identity를 typed binding으로 복원한다."""

    if not isinstance(metadata, Mapping):
        raise Stage2MeasurementV2Error("output-master physical raw metadata가 없습니다")
    capture_id = metadata.get("capture_id")
    hardware_sha = metadata.get("hardware_config_sha256")
    devices = metadata.get("resolved_devices")
    meter = metadata.get("fresh_meter")
    confirmations = metadata.get("operator_confirmations")
    if (
        type(capture_id) is not str
        or not capture_id
        or not _is_sha256(hardware_sha)
        or not isinstance(devices, Mapping)
        or set(devices) != {"input", "output"}
        or any(type(devices[key]) is not int or devices[key] < 0 for key in devices)
        or not isinstance(meter, Mapping)
        or set(meter) != _FRESH_METER_KEYS
        or not isinstance(confirmations, Mapping)
        or set(confirmations)
        != {
            "speaker_output",
            "user_present",
            "volume_fixed_after_meter_adjustment",
            "routing_and_geometry",
            "same_amplifier_setting",
        }
        or any(value is not True for value in confirmations.values())
    ):
        raise Stage2MeasurementV2Error(
            "output-master physical raw capture/hardware/device/meter/confirmation binding이 다릅니다"
        )
    raw_ref = _artifact_ref(
        {"path": meter.get("path"), "sha256": meter.get("sha256")},
        label="fresh meter raw",
    )
    receipt_ref = _artifact_ref(
        {
            "path": meter.get("receipt_path"),
            "sha256": meter.get("receipt_sha256"),
        },
        label="fresh meter receipt",
    )
    calibration_ref = _artifact_ref(
        meter.get("calibration_evidence"), label="meter calibration evidence"
    )
    if (
        meter.get("resolved_devices") != dict(devices)
        or meter.get("hardware_identity") in (None, {})
        or meter.get("freshness_max_seconds") != 600
        or meter.get("physical_fingerprint")
        != meter["hardware_identity"].get("physical_fingerprint")
    ):
        raise Stage2MeasurementV2Error(
            "output-master fresh meter hardware/device/freshness binding이 다릅니다"
        )
    identity = metadata.get("repository_execution_identity")
    authority = _validate_measurement_git_authority(
        metadata.get("measurement_git_authority")
    )
    if (
        not isinstance(identity, Mapping)
        or identity.get("repository_commit") != authority["head"]
        or identity.get("repository_branch") != "dev"
        or identity.get("repository_dirty") is not False
    ):
        raise Stage2MeasurementV2Error(
            "output-master repository execution identity/Git authority가 다릅니다"
        )
    return {
        "capture_id": capture_id,
        "repository_execution_identity": dict(identity),
        "measurement_git_authority": authority,
        "hardware_config_sha256": hardware_sha,
        "resolved_devices": dict(devices),
        "hardware_identity": dict(meter["hardware_identity"]),
        "fresh_meter": dict(meter),
        "meter_raw_artifact": raw_ref,
        "meter_receipt_artifact": receipt_ref,
        "calibration_evidence_artifact": calibration_ref,
        "operator_confirmations": dict(confirmations),
    }


def _reopen_bound_meter(
    repository_root: str,
    binding: Mapping[str, Any],
    *,
    require_fresh: bool,
) -> dict[str, Any]:
    """Official meter validator와 세 lineage bytes를 같은 snapshot으로 닫는다."""

    payloads: dict[str, bytes] = {}
    for key, label in (
        ("meter_raw_artifact", "Stage-2 fresh meter raw"),
        ("meter_receipt_artifact", "Stage-2 fresh meter receipt"),
        ("calibration_evidence_artifact", "Stage-2 calibration evidence"),
    ):
        ref = _artifact_ref(binding.get(key), label=label)
        with RepositoryFileGuard(repository_root, ref["path"], label=label) as guard:
            if guard.sha256 != ref["sha256"]:
                raise Stage2MeasurementV2Error(f"{label} file SHA가 다릅니다")
            payloads[key] = guard.bytes
            guard.verify()
    from .measurement_level import validate_bootstrap_meter_raw

    verified = validate_bootstrap_meter_raw(
        binding["meter_raw_artifact"]["path"],
        repository_root=repository_root,
        expected_hardware_identity=dict(binding["hardware_identity"]),
        require_fresh=require_fresh,
    )
    metadata = verified.get("metadata")
    if (
        verified.get("sha256") != binding["meter_raw_artifact"]["sha256"]
        or not isinstance(metadata, Mapping)
        or metadata.get("resolved_devices") != binding["resolved_devices"]
        or metadata.get("hardware_identity") != binding["hardware_identity"]
    ):
        raise Stage2MeasurementV2Error(
            "official fresh meter validator/raw SHA/hardware/device binding이 다릅니다"
        )
    return {
        "verified_meter": verified,
        "artifact_payload_sha256": {
            key: hashlib.sha256(payload).hexdigest()
            for key, payload in payloads.items()
        },
    }


def _physical_continuity_binding(
    repository_root: str,
    diagnostic_binding: Mapping[str, Any],
    ps_capture_metadata: Mapping[str, Any],
    *,
    require_fresh_meter: bool,
) -> dict[str, Any]:
    expected = dict(diagnostic_binding)
    observed = _validate_physical_capture_metadata(ps_capture_metadata)
    for key in (
        "capture_id",
        "measurement_git_authority",
        "hardware_config_sha256",
        "resolved_devices",
        "hardware_identity",
        "meter_raw_artifact",
        "meter_receipt_artifact",
        "calibration_evidence_artifact",
        "operator_confirmations",
    ):
        if observed.get(key) != expected.get(key):
            raise Stage2MeasurementV2Error(
                f"diagnostic/P-S physical continuity가 다릅니다: {key}"
            )
    stable_meter_keys = _FRESH_METER_KEYS - {"age_seconds"}
    if any(
        observed["fresh_meter"].get(key) != expected["fresh_meter"].get(key)
        for key in stable_meter_keys
    ):
        raise Stage2MeasurementV2Error(
            "diagnostic/P-S fresh meter stable binding이 다릅니다"
        )
    reopened = _reopen_bound_meter(
        repository_root, observed, require_fresh=require_fresh_meter
    )
    receipt: dict[str, Any] = {
        "capture_id": observed["capture_id"],
        "measurement_git_authority": observed["measurement_git_authority"],
        "hardware_config_sha256": observed["hardware_config_sha256"],
        "resolved_devices": observed["resolved_devices"],
        "hardware_identity": observed["hardware_identity"],
        "fresh_meter_raw": observed["meter_raw_artifact"],
        "fresh_meter_receipt": observed["meter_receipt_artifact"],
        "calibration_evidence": observed["calibration_evidence_artifact"],
        "meter_reopened_with_official_validator": True,
        "artifact_payload_sha256": reopened["artifact_payload_sha256"],
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt


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
            "Stage-2 output-master P/S v3 JSON이 canonical finite가 아닙니다"
        ) from exc


def _ps_slice(
    plan: Mapping[str, Any], full_submitted_pcm: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, int]:
    canonical, full = validate_stage2_v2_live_safe_fallback_plan(
        plan, full_submitted_pcm
    )
    boundary = int(
        canonical["live_phase_contract"]["diagnostic_phase_stop_frame"]
    )
    submitted = np.ascontiguousarray(full[boundary:], dtype="<i2")
    stream = canonical["live_phase_contract"]["ps_stream"]
    if (
        boundary <= 0
        or boundary >= len(full)
        or int(stream["local_start_frame"]) != 0
        or int(stream["local_stop_frame"]) != len(submitted)
        or int(stream["logical_plan_start_frame"]) != boundary
        or int(stream["logical_plan_stop_frame"]) != len(full)
    ):
        raise Stage2MeasurementV2Error(
            "Stage-2 output-master P/S v3 canonical local frame contract가 다릅니다"
        )
    return canonical, submitted, boundary


def build_stage2_output_master_ps_v3_plan(
    plan: Mapping[str, Any], full_submitted_pcm: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    """가변 input clock 축을 명시한 P/S v3 signal-only 계획을 만든다."""

    canonical, submitted, boundary = _ps_slice(plan, full_submitted_pcm)
    receipt: dict[str, Any] = {
        "schema": PS_V3_PLAN_SCHEMA,
        "status": "SIGNAL_ONLY_PS_OUTPUT_FORBIDDEN_UNTIL_TYPED_ADMISSION",
        "source_signal_plan_sha256": canonical["canonical_payload_sha256"],
        "diagnostic_logical_output_frame_range": [0, boundary],
        "ps_logical_output_frame_range": [boundary, len(full_submitted_pcm)],
        "ps_local_output_frame_range": [0, len(submitted)],
        "ps_output_frames": len(submitted),
        "ps_output_seconds": len(submitted) / SAMPLE_RATE,
        "ps_submitted_pcm_dtype": submitted.dtype.str,
        "ps_submitted_pcm_shape": list(submitted.shape),
        "ps_submitted_pcm_sha256": _array_sha256(submitted),
        "transport": {
            "kind": "independent_input_output_streams_output_clock_master",
            "output_clock_owner": "outputstream_callback_only",
            "input_role": "raw_witness_only_never_output_pacing",
            "input_pre_roll_frames": PRE_ROLL_FRAMES,
            "input_post_roll_frames": POST_ROLL_FRAMES,
            "input_frame_count_is_independent_and_variable": True,
            "input_output_frame_identity_claimed": False,
            "cross_clock_timestamp_alignment_used": False,
            "legacy_combined_duplex_allowed": False,
        },
        "reserved_raw_schema": PS_V3_RAW_STRUCTURE_SCHEMA,
        "required_authority_adapters": list(_UNIMPLEMENTED_AUTHORITY_BLOCKERS),
        "diagnostic_clock_pass_alone_authorizes_ps": False,
        "ps_stream_may_open": False,
        "plant_identification_eligible": False,
        "canonical_training_eligible": False,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    submitted.setflags(write=False)
    return receipt, submitted


def validate_output_master_diagnostic_clock_publication(
    repository_root: str,
    session_relative_path: str,
    publication: Mapping[str, Any],
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
) -> dict[str, Any]:
    """clock JSON뿐 아니라 그 JSON이 결속한 diagnostic raw를 다시 연다.

    Caller가 만든 ``{"passed": true}``나 path/SHA가 없는 self-attestation은 여기서
    authority가 될 수 없다.
    """

    canonical, _submitted, _boundary = _ps_slice(plan, full_submitted_pcm)
    target = output_master_session_targets(session_relative_path)["clock_receipt"]
    if (
        not isinstance(publication, Mapping)
        or publication.get("path") != target
        or type(publication.get("sha256")) is not str
        or len(str(publication["sha256"])) != 64
    ):
        raise Stage2MeasurementV2Error(
            "output-master diagnostic clock publication path/SHA가 다릅니다"
        )
    with RepositoryFileGuard(
        repository_root, target, label="Stage-2 output-master diagnostic clock receipt"
    ) as guard:
        if guard.sha256 != publication["sha256"]:
            raise Stage2MeasurementV2Error(
                "output-master diagnostic clock receipt file SHA가 다릅니다"
            )
        try:
            receipt = json.loads(guard.bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Stage2MeasurementV2Error(
                "output-master diagnostic clock receipt JSON을 읽을 수 없습니다"
            ) from exc
        if not isinstance(receipt, dict):
            raise Stage2MeasurementV2Error(
                "output-master diagnostic clock receipt가 object가 아닙니다"
            )
        # Publisher의 canonical JSON + trailing newline까지 exact하게 닫는다.
        if guard.bytes != _canonical_json_bytes(receipt) + b"\n":
            raise Stage2MeasurementV2Error(
                "output-master diagnostic clock receipt bytes가 canonical이 아닙니다"
            )
        guard.verify()

    supplied_sha = receipt.get("canonical_payload_sha256")
    without_sha = dict(receipt)
    without_sha.pop("canonical_payload_sha256", None)
    raw_ref = receipt.get("raw_artifact")
    clock = receipt.get("global_clock_receipt")
    if (
        supplied_sha != _payload_sha256(without_sha)
        or receipt.get("schema") != OUTPUT_MASTER_CLOCK_RECEIPT_SCHEMA
        or receipt.get("session_relative_path") != session_relative_path
        or receipt.get("signal_plan_sha256")
        != canonical["canonical_payload_sha256"]
        or not isinstance(raw_ref, Mapping)
        or set(raw_ref) != {"path", "sha256"}
        or not isinstance(clock, Mapping)
        or clock.get("schema") != DIAGNOSTIC_CLOCK_SCHEMA
        or receipt.get("global_clock_receipt_sha256")
        != clock.get("canonical_payload_sha256")
        or receipt.get("clock_authority_granted") is not False
        or receipt.get("ps_phase_may_start") is not False
        or receipt.get("plant_identification_eligible") is not False
        or receipt.get("canonical_training_eligible") is not False
    ):
        raise Stage2MeasurementV2Error(
            "output-master diagnostic clock receipt schema/SHA/authority가 다릅니다"
        )
    clock_without_sha = dict(clock)
    clock_sha = clock_without_sha.pop("canonical_payload_sha256", None)
    if (
        clock_sha != _payload_sha256(clock_without_sha)
        or clock.get("signal_plan_sha256")
        != canonical["canonical_payload_sha256"]
        or clock.get("ps_phase_may_start") is not False
    ):
        raise Stage2MeasurementV2Error(
            "nested diagnostic global clock receipt SHA/plan/authority가 다릅니다"
        )
    raw = snapshot_published_output_master_raw(
        repository_root,
        session_relative_path,
        dict(raw_ref),
        canonical,
        full_submitted_pcm,
    )
    physical_binding = _validate_physical_capture_metadata(raw["metadata"])
    expected_pass = bool(clock.get("passed") is True and raw["metadata"]["adc_clip_count"] == 0)
    if (
        receipt.get("submitted_pcm_sha256") != _array_sha256(raw["submitted_pcm"])
        or receipt.get("captured_pcm_sha256") != _array_sha256(raw["captured_pcm"])
        or receipt.get("transport_receipt") != raw["transport_receipt"]
        or int(receipt.get("adc_clip_count", -1))
        != int(raw["metadata"]["adc_clip_count"])
        or receipt.get("passed") is not expected_pass
        or receipt.get("diagnostic_linearity_may_run") is not expected_pass
    ):
        raise Stage2MeasurementV2Error(
            "output-master diagnostic clock receipt와 실제 raw 재계산이 다릅니다"
        )
    return {
        "receipt": receipt,
        "raw": raw,
        "publication": dict(publication),
        "physical_capture_binding": physical_binding,
    }


def _normalised_int32_capture(value: np.ndarray, *, label: str) -> np.ndarray:
    source = np.asarray(value)
    if (
        source.dtype != np.dtype("<i4")
        or source.ndim != 2
        or source.shape[1] != 2
        or len(source) <= 0
    ):
        raise Stage2MeasurementV2Error(f"{label}는 exact <i4 [frames,2]여야 합니다")
    return np.ascontiguousarray(source, dtype=np.float64) / 2147483648.0


def _float_capture_to_int32(value: np.ndarray, *, label: str) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2 or not np.all(np.isfinite(source)):
        raise Stage2MeasurementV2Error(f"{label}가 finite [frames,2]가 아닙니다")
    scaled = np.rint(source * 2147483648.0)
    if float(np.max(scaled)) > 2147483647.0 or float(np.min(scaled)) < -2147483648.0:
        raise Stage2MeasurementV2Error(
            f"{label} cubic/linear interpolation이 int32 범위를 넘습니다"
        )
    return np.ascontiguousarray(scaled, dtype="<i4")


def _resample_capture_to_output_grid(
    captured_input_pcm: np.ndarray,
    *,
    output_origin_input_frame: int,
    rate_ratio: float,
    output_frames: int,
    interpolation_kind: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """ADC input 축을 DAC output local frame 축으로 bounded resampling한다."""

    captured = _normalised_int32_capture(
        captured_input_pcm, label="output-master captured input"
    )
    if (
        isinstance(output_origin_input_frame, (bool, np.bool_))
        or not isinstance(output_origin_input_frame, (int, np.integer))
        or int(output_origin_input_frame) < 0
        or not np.isfinite(rate_ratio)
        or not (1.0 - 1.0e-3 < float(rate_ratio) < 1.0 + 1.0e-3)
        or isinstance(output_frames, (bool, np.bool_))
        or not isinstance(output_frames, (int, np.integer))
        or int(output_frames) <= 0
        or interpolation_kind not in {"cubic", "linear"}
    ):
        raise Stage2MeasurementV2Error(
            "output-master resampling origin/q/frame/interpolation contract가 잘못됐습니다"
        )
    origin = int(output_origin_input_frame)
    frames = int(output_frames)
    q = float(rate_ratio)
    query = origin + np.arange(frames, dtype=np.float64) / q
    if query[0] < 0.0 or query[-1] > len(captured) - 1:
        raise Stage2MeasurementV2Error(
            "output-master input pre/post-roll이 전체 DAC grid 보간을 지원하지 않습니다"
        )
    axis = np.arange(len(captured), dtype=np.float64)
    if interpolation_kind == "cubic":
        corrected = np.column_stack(
            [
                CubicSpline(axis, captured[:, microphone], extrapolate=False)(query)
                for microphone in range(2)
            ]
        )
    else:
        corrected = np.column_stack(
            [
                np.interp(query, axis, captured[:, microphone])
                for microphone in range(2)
            ]
        )
    pcm = _float_capture_to_int32(
        corrected, label=f"output-master {interpolation_kind} DAC-grid capture"
    )
    receipt = {
        "interpolation_kind": interpolation_kind,
        "input_frames": len(captured),
        "output_frames": frames,
        "output_origin_input_frame": origin,
        "rate_ratio": q,
        "estimated_ppm": (q - 1.0) * 1.0e6,
        "query_input_frame_range": [float(query[0]), float(query[-1])],
        "left_support_margin_frames": float(query[0]),
        "right_support_margin_frames": float(len(captured) - 1 - query[-1]),
        "input_pcm_sha256": _array_sha256(np.asarray(captured_input_pcm)),
        "output_grid_pcm_sha256": _array_sha256(pcm),
        "input_output_frame_identity_claimed": False,
        "absolute_hardware_clock_authority_claimed": False,
    }
    return pcm, receipt


def _analyse_validated_stage2_output_master_diagnostic_linearity_v3(
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    durable_diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    """durable split raw를 q-corrected DAC grid로 옮겨 49/98 gate를 재계산한다."""

    canonical, _ps_submitted, boundary = _ps_slice(plan, full_submitted_pcm)
    if (
        not isinstance(durable_diagnostic, Mapping)
        or not isinstance(durable_diagnostic.get("receipt"), Mapping)
        or not isinstance(durable_diagnostic.get("raw"), Mapping)
    ):
        raise Stage2MeasurementV2Error(
            "validated durable diagnostic raw/clock bundle이 필요합니다"
        )
    publication_receipt = durable_diagnostic["receipt"]
    raw = durable_diagnostic["raw"]
    clock = publication_receipt.get("global_clock_receipt")
    captured = np.asarray(raw.get("captured_pcm"))
    if (
        publication_receipt.get("passed") is not True
        or not isinstance(clock, Mapping)
        or clock.get("schema") != DIAGNOSTIC_CLOCK_SCHEMA
        or clock.get("passed") is not True
        or publication_receipt.get("captured_pcm_sha256") != _array_sha256(captured)
    ):
        raise Stage2MeasurementV2Error(
            "diagnostic global affine clock/raw bundle이 PASS/bound 상태가 아닙니다"
        )
    alignment = clock.get("alignment")
    search = clock.get("global_search")
    if not isinstance(alignment, Mapping) or not isinstance(search, Mapping):
        raise Stage2MeasurementV2Error("diagnostic clock alignment/global search가 없습니다")
    origin = alignment.get("coarse_capture_offset_samples")
    q = search.get("selected_rate_ratio")
    corrected: dict[str, np.ndarray] = {}
    transforms: dict[str, Any] = {}
    analyses: dict[str, Any] = {}
    for kind in ("cubic", "linear"):
        corrected[kind], transforms[kind] = _resample_capture_to_output_grid(
            captured,
            output_origin_input_frame=origin,
            rate_ratio=q,
            output_frames=boundary,
            interpolation_kind=kind,
        )
        analyses[kind] = analyse_stage2_v2_diagnostic_preflight(
            canonical,
            full_submitted_pcm,
            corrected[kind],
            transport_counters={"xrun": 0, "clip": 0, "callback_status": 0},
            quiet_start_frame=8_192,
        )
    passed = bool(
        analyses["cubic"]["passed"] is True
        and analyses["linear"]["passed"] is True
    )
    receipt: dict[str, Any] = {
        "schema": DIAGNOSTIC_LINEARITY_V3_SCHEMA,
        "status": (
            "PASS_CLOCK_CORRECTED_LINEARITY_PS_RAW_CAPTURE_MAY_START"
            if passed
            else "FAIL_CLOCK_CORRECTED_LINEARITY_PS_RAW_CAPTURE_FORBIDDEN"
        ),
        "source_signal_plan_sha256": canonical["canonical_payload_sha256"],
        "diagnostic_clock_publication_receipt_sha256": publication_receipt[
            "canonical_payload_sha256"
        ],
        "diagnostic_raw_artifact": dict(publication_receipt["raw_artifact"]),
        "diagnostic_input_pcm_sha256": _array_sha256(captured),
        "clock_receipt_sha256": clock["canonical_payload_sha256"],
        "output_grid_transforms": transforms,
        "output_grid_analysis_receipts": analyses,
        "cubic_linear_both_must_pass": True,
        "passed": passed,
        "ps_phase_may_start": passed,
        "plant_identification_eligible": False,
        "canonical_training_eligible": False,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return {
        "receipt": receipt,
        "cubic_output_grid_pcm": corrected["cubic"],
        "linear_output_grid_pcm": corrected["linear"],
    }


def analyse_stage2_output_master_diagnostic_linearity_v3(
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    repository_root: str,
    diagnostic_session_relative_path: str,
    diagnostic_clock_publication: Mapping[str, Any],
) -> dict[str, Any]:
    """Public analyzer: caller mapping이 아닌 durable clock/raw bytes를 반드시 재검증한다."""

    durable = validate_output_master_diagnostic_clock_publication(
        repository_root,
        diagnostic_session_relative_path,
        diagnostic_clock_publication,
        plan,
        full_submitted_pcm,
    )
    return _analyse_validated_stage2_output_master_diagnostic_linearity_v3(
        plan, full_submitted_pcm, durable
    )


def _diagnostic_linearity_target(session_relative_path: str) -> str:
    # output_master_session_targets가 root 바로 아래의 canonical session을 검증한다.
    output_master_session_targets(session_relative_path)
    return (PurePosixPath(session_relative_path) / DIAGNOSTIC_LINEARITY_LEAF).as_posix()


def publish_stage2_output_master_diagnostic_linearity_v3_no_replace(
    repository_root: str,
    diagnostic_session_relative_path: str,
    diagnostic_clock_publication: Mapping[str, Any],
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
) -> dict[str, Any]:
    """Clock-corrected 49/98 receipt를 raw 재검증 후 no-replace 발행한다."""

    analysis = analyse_stage2_output_master_diagnostic_linearity_v3(
        plan,
        full_submitted_pcm,
        repository_root=repository_root,
        diagnostic_session_relative_path=diagnostic_session_relative_path,
        diagnostic_clock_publication=diagnostic_clock_publication,
    )
    receipt = analysis["receipt"]
    if receipt.get("passed") is not True or receipt.get("ps_phase_may_start") is not True:
        raise Stage2MeasurementV2Error(
            "clock-corrected diagnostic linearity FAIL은 durable PS authorization이 될 수 없습니다"
        )
    target = _diagnostic_linearity_target(diagnostic_session_relative_path)
    published = publish_repository_bytes_noreplace(
        repository_root,
        target,
        _canonical_json_bytes(receipt) + b"\n",
        mode=0o600,
        preserve_recovery_link=True,
        recovery_tag="stage2_output_master_linearity_v3",
    )
    return {**published, "receipt": receipt}


def validate_published_stage2_output_master_diagnostic_linearity_v3(
    repository_root: str,
    diagnostic_session_relative_path: str,
    diagnostic_clock_publication: Mapping[str, Any],
    linearity_publication: Mapping[str, Any],
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
) -> dict[str, Any]:
    """Linearity JSON을 재개방하고 clock/raw에서 독립 재계산한다."""

    target = _diagnostic_linearity_target(diagnostic_session_relative_path)
    expected_ref = _artifact_ref(linearity_publication, label="diagnostic linearity v3")
    if expected_ref["path"] != target:
        raise Stage2MeasurementV2Error("diagnostic linearity v3 publication path가 다릅니다")
    with RepositoryFileGuard(
        repository_root, target, label="Stage-2 output-master diagnostic linearity v3"
    ) as guard:
        if guard.sha256 != expected_ref["sha256"]:
            raise Stage2MeasurementV2Error("diagnostic linearity v3 file SHA가 다릅니다")
        try:
            stored = json.loads(guard.bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Stage2MeasurementV2Error("diagnostic linearity v3 JSON이 유효하지 않습니다") from exc
        if not isinstance(stored, Mapping) or guard.bytes != _canonical_json_bytes(stored) + b"\n":
            raise Stage2MeasurementV2Error("diagnostic linearity v3 bytes가 canonical이 아닙니다")
        guard.verify()
    recomputed = analyse_stage2_output_master_diagnostic_linearity_v3(
        plan,
        full_submitted_pcm,
        repository_root=repository_root,
        diagnostic_session_relative_path=diagnostic_session_relative_path,
        diagnostic_clock_publication=diagnostic_clock_publication,
    )
    if (
        dict(stored) != recomputed["receipt"]
        or stored.get("schema") != DIAGNOSTIC_LINEARITY_V3_SCHEMA
        or stored.get("passed") is not True
        or stored.get("ps_phase_may_start") is not True
        or stored.get("plant_identification_eligible") is not False
    ):
        raise Stage2MeasurementV2Error(
            "diagnostic linearity v3 stored/raw-recomputed receipt가 다릅니다"
        )
    return {
        **recomputed,
        "publication": expected_ref,
    }


def _ps_v3_session_target(session_relative_path: str, leaf: str) -> str:
    try:
        session = canonical_relative_path(
            session_relative_path, label="output-master P/S v3 session"
        )
    except (TypeError, ValueError) as exc:
        raise Stage2MeasurementV2Error("output-master P/S v3 session path가 canonical이 아닙니다") from exc
    path = PurePosixPath(session)
    if path.parent != PurePosixPath(PS_V3_SESSION_ROOT) or path.name in {"", ".", ".."}:
        raise Stage2MeasurementV2Error(
            "output-master P/S v3 session은 고정 root 바로 아래여야 합니다"
        )
    return (path / leaf).as_posix()


def output_master_ps_v3_session_targets(session_relative_path: str) -> dict[str, str]:
    return {
        "raw": _ps_v3_session_target(session_relative_path, PS_V3_RAW_LEAF),
        "partial_raw": _ps_v3_session_target(
            session_relative_path, PS_V3_PARTIAL_RAW_LEAF
        ),
        "physical_level": _ps_v3_session_target(
            session_relative_path, PS_V3_PHYSICAL_LEVEL_LEAF
        ),
    }


def _telemetry_scalar(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    scalar = {
        key: value
        for key, value in telemetry.items()
        if key not in _TELEMETRY_ARRAY_NAMES and key != "actual_submitted_pcm"
    }
    if any(isinstance(value, np.ndarray) for value in scalar.values()):
        raise Stage2MeasurementV2Error("P/S v3 telemetry scalar에 ndarray가 남았습니다")
    _canonical_json_bytes(scalar)
    return scalar


def _serialize_ps_v3_success_raw(
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


def publish_stage2_output_master_ps_v3_raw_no_replace(
    repository_root: str,
    ps_session_relative_path: str,
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    diagnostic_session_relative_path: str,
    diagnostic_clock_publication: Mapping[str, Any],
    diagnostic_linearity_publication: Mapping[str, Any],
    captured_ps_input_pcm: np.ndarray,
    ps_telemetry: Mapping[str, Any],
    capture_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """P/S output-master raw를 analysis 전에 no-replace로 발행한다."""

    canonical, submitted, _boundary = _ps_slice(plan, full_submitted_pcm)
    diagnostic = validate_published_stage2_output_master_diagnostic_linearity_v3(
        repository_root,
        diagnostic_session_relative_path,
        diagnostic_clock_publication,
        diagnostic_linearity_publication,
        canonical,
        full_submitted_pcm,
    )
    durable_clock = validate_output_master_diagnostic_clock_publication(
        repository_root,
        diagnostic_session_relative_path,
        diagnostic_clock_publication,
        canonical,
        full_submitted_pcm,
    )
    continuity = _physical_continuity_binding(
        repository_root,
        durable_clock["physical_capture_binding"],
        capture_metadata,
        require_fresh_meter=True,
    )
    captured = np.asarray(captured_ps_input_pcm)
    transport = validate_output_master_success_telemetry(
        ps_telemetry,
        captured_pcm=captured,
        expected_submitted_pcm=submitted,
    )
    clip_count = int(
        np.count_nonzero(np.abs(captured.astype(np.int64)) >= 2**31 - 1)
    )
    if clip_count:
        raise Stage2MeasurementV2Error("output-master P/S v3 raw ADC clip이 0이 아닙니다")
    clock_ref = _artifact_ref(
        diagnostic_clock_publication, label="diagnostic clock publication"
    )
    linearity_ref = _artifact_ref(
        diagnostic_linearity_publication, label="diagnostic linearity publication"
    )
    metadata: dict[str, Any] = {
        **dict(capture_metadata),
        "schema": PS_V3_RAW_SCHEMA,
        "role": "output_master_ps_raw_first_before_analysis",
        "session_relative_path": ps_session_relative_path,
        "signal_plan_sha256": canonical["canonical_payload_sha256"],
        "submitted_pcm_sha256": _array_sha256(submitted),
        "captured_pcm_sha256": _array_sha256(captured),
        "transport_receipt": transport,
        "adc_clip_count": clip_count,
        "diagnostic_session_relative_path": diagnostic_session_relative_path,
        "diagnostic_clock_artifact": clock_ref,
        "diagnostic_linearity_artifact": linearity_ref,
        "diagnostic_linearity_receipt_sha256": diagnostic["receipt"][
            "canonical_payload_sha256"
        ],
        "physical_continuity_receipt": continuity,
        "input_output_frame_identity_claimed": False,
        "raw_first_analysis_completed": False,
        "plant_identification_eligible": False,
        "canonical_training_eligible": False,
    }
    payload = _serialize_ps_v3_success_raw(
        submitted=submitted,
        captured=captured,
        telemetry=ps_telemetry,
        metadata=metadata,
    )
    published = publish_repository_bytes_noreplace(
        repository_root,
        output_master_ps_v3_session_targets(ps_session_relative_path)["raw"],
        payload,
        mode=0o600,
        preserve_recovery_link=True,
        recovery_tag="stage2_output_master_ps_v3_raw",
    )
    return {**published, "transport_receipt": transport}


def snapshot_published_stage2_output_master_ps_v3_raw(
    repository_root: str,
    ps_session_relative_path: str,
    publication: Mapping[str, Any],
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    diagnostic_session_relative_path: str,
    diagnostic_clock_publication: Mapping[str, Any],
    diagnostic_linearity_publication: Mapping[str, Any],
) -> dict[str, Any]:
    """P/S NPZ bytes/telemetry/diagnostic+meter continuity를 독립적으로 재검증한다."""

    canonical, expected, _boundary = _ps_slice(plan, full_submitted_pcm)
    target = output_master_ps_v3_session_targets(ps_session_relative_path)["raw"]
    ref = _artifact_ref(publication, label="output-master P/S v3 raw")
    if ref["path"] != target:
        raise Stage2MeasurementV2Error("output-master P/S v3 raw publication path가 다릅니다")
    with RepositoryFileGuard(repository_root, target, label="Stage-2 P/S v3 raw") as guard:
        if guard.sha256 != ref["sha256"]:
            raise Stage2MeasurementV2Error("output-master P/S v3 raw file SHA가 다릅니다")
        with np.load(io.BytesIO(guard.bytes), allow_pickle=False) as archive:
            required = {
                "submitted_pcm",
                "captured_pcm",
                "metadata_json",
                *(f"telemetry_{name}" for name in _TELEMETRY_ARRAY_NAMES),
            }
            if set(archive.files) != required:
                raise Stage2MeasurementV2Error("output-master P/S v3 raw NPZ key set이 exact하지 않습니다")
            submitted = np.asarray(archive["submitted_pcm"]).copy()
            captured = np.asarray(archive["captured_pcm"]).copy()
            metadata = json.loads(str(archive["metadata_json"].item()))
            arrays = {
                name: np.asarray(archive[f"telemetry_{name}"]).copy()
                for name in _TELEMETRY_ARRAY_NAMES
            }
        raw_file_sha = guard.sha256
        guard.verify()
    if (
        submitted.dtype != np.dtype("<i2")
        or not np.array_equal(submitted, expected)
        or captured.dtype != np.dtype("<i4")
        or captured.ndim != 2
        or captured.shape[1] != 2
        or not isinstance(metadata, Mapping)
        or metadata.get("schema") != PS_V3_RAW_SCHEMA
        or metadata.get("role") != "output_master_ps_raw_first_before_analysis"
        or metadata.get("session_relative_path") != ps_session_relative_path
        or metadata.get("signal_plan_sha256") != canonical["canonical_payload_sha256"]
        or metadata.get("submitted_pcm_sha256") != _array_sha256(submitted)
        or metadata.get("captured_pcm_sha256") != _array_sha256(captured)
        or metadata.get("adc_clip_count") != 0
        or metadata.get("raw_first_analysis_completed") is not False
        or metadata.get("plant_identification_eligible") is not False
        or metadata.get("canonical_training_eligible") is not False
    ):
        raise Stage2MeasurementV2Error("output-master P/S v3 raw PCM/metadata/authority binding이 다릅니다")
    scalar = metadata.get("telemetry_scalar")
    if not isinstance(scalar, Mapping):
        raise Stage2MeasurementV2Error("output-master P/S v3 raw telemetry scalar가 없습니다")
    telemetry = {**dict(scalar), **arrays, "actual_submitted_pcm": submitted.copy()}
    transport = validate_output_master_success_telemetry(
        telemetry, captured_pcm=captured, expected_submitted_pcm=submitted
    )
    if metadata.get("transport_receipt") != transport:
        raise Stage2MeasurementV2Error("output-master P/S v3 stored/recomputed transport가 다릅니다")
    diagnostic = validate_published_stage2_output_master_diagnostic_linearity_v3(
        repository_root,
        diagnostic_session_relative_path,
        diagnostic_clock_publication,
        diagnostic_linearity_publication,
        canonical,
        full_submitted_pcm,
    )
    durable_clock = validate_output_master_diagnostic_clock_publication(
        repository_root,
        diagnostic_session_relative_path,
        diagnostic_clock_publication,
        canonical,
        full_submitted_pcm,
    )
    continuity = _physical_continuity_binding(
        repository_root,
        durable_clock["physical_capture_binding"],
        metadata,
        require_fresh_meter=False,
    )
    if (
        metadata.get("diagnostic_session_relative_path")
        != diagnostic_session_relative_path
        or metadata.get("diagnostic_clock_artifact")
        != _artifact_ref(diagnostic_clock_publication, label="diagnostic clock")
        or metadata.get("diagnostic_linearity_artifact")
        != _artifact_ref(diagnostic_linearity_publication, label="diagnostic linearity")
        or metadata.get("diagnostic_linearity_receipt_sha256")
        != diagnostic["receipt"]["canonical_payload_sha256"]
        or metadata.get("physical_continuity_receipt") != continuity
    ):
        raise Stage2MeasurementV2Error("output-master P/S v3 diagnostic/meter continuity DAG가 다릅니다")
    return {
        "artifact_ref": ref,
        "raw_npz_sha256": raw_file_sha,
        "submitted_pcm": submitted,
        "captured_pcm": captured,
        "metadata": dict(metadata),
        "telemetry": telemetry,
        "transport_receipt": transport,
        "diagnostic": diagnostic,
        "physical_capture_binding": _validate_physical_capture_metadata(metadata),
    }


def publish_stage2_output_master_ps_v3_partial_raw_no_replace(
    repository_root: str,
    ps_session_relative_path: str,
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    failure: OutputMasterCaptureFailure,
    capture_metadata: Mapping[str, Any],
    diagnostic_session_relative_path: str,
    diagnostic_clock_publication: Mapping[str, Any],
    diagnostic_linearity_publication: Mapping[str, Any],
) -> dict[str, Any]:
    """P/S transport 실패 prefix를 보존하고 자동 재시도/승격을 금지한다."""

    canonical, planned, _boundary = _ps_slice(plan, full_submitted_pcm)
    actual = np.asarray(failure.submitted_pcm)
    captured = np.asarray(failure.captured_pcm)
    capture_mask = np.asarray(failure.capture_valid_mask)
    submitted_mask = np.asarray(failure.submitted_valid_mask)
    if (
        actual.dtype != np.dtype("<i2")
        or actual.shape != planned.shape
        or captured.dtype != np.dtype("<i4")
        or captured.ndim != 2
        or captured.shape[1] != 2
        or capture_mask.shape != (len(captured),)
        or submitted_mask.shape != (len(planned),)
    ):
        raise Stage2MeasurementV2Error("P/S v3 partial raw shape/dtype/mask가 다릅니다")
    telemetry = {
        **dict(failure.telemetry),
        "capture_valid_mask": capture_mask,
        "submitted_valid_mask": submitted_mask,
        "actual_submitted_pcm": actual,
    }
    arrays = {
        f"telemetry_{name}": np.asarray(telemetry[name])
        for name in _TELEMETRY_ARRAY_NAMES
    }
    metadata: dict[str, Any] = {
        **dict(capture_metadata),
        "schema": PS_V3_PARTIAL_RAW_SCHEMA,
        "role": "output_master_ps_failure_partial_raw_never_promotable",
        "session_relative_path": ps_session_relative_path,
        "signal_plan_sha256": canonical["canonical_payload_sha256"],
        "planned_submitted_pcm_sha256": _array_sha256(planned),
        "actual_submitted_pcm_sha256": _array_sha256(actual),
        "captured_pcm_sha256": _array_sha256(captured),
        "capture_valid_frames": int(np.count_nonzero(capture_mask)),
        "submitted_valid_frames": int(np.count_nonzero(submitted_mask)),
        "capture_exception": str(failure),
        "diagnostic_session_relative_path": diagnostic_session_relative_path,
        "diagnostic_clock_artifact": _artifact_ref(
            diagnostic_clock_publication, label="diagnostic clock"
        ),
        "diagnostic_linearity_artifact": _artifact_ref(
            diagnostic_linearity_publication, label="diagnostic linearity"
        ),
        "partial_capture_never_promotable": True,
        "automatic_retry_allowed": False,
        "plant_identification_eligible": False,
        "canonical_training_eligible": False,
        "telemetry_scalar": _telemetry_scalar(telemetry),
    }
    output = io.BytesIO()
    np.savez(
        output,
        planned_submitted_pcm=np.asarray(planned, dtype="<i2"),
        actual_submitted_pcm=actual,
        captured_pcm=captured,
        metadata_json=np.asarray(_canonical_json_bytes(metadata).decode("utf-8")),
        **arrays,
    )
    return publish_repository_bytes_noreplace(
        repository_root,
        output_master_ps_v3_session_targets(ps_session_relative_path)["partial_raw"],
        output.getvalue(),
        mode=0o600,
        preserve_recovery_link=True,
        recovery_tag="stage2_output_master_ps_v3_partial",
    )


def publish_stage2_output_master_ps_v3_physical_level_no_replace(
    repository_root: str,
    ps_session_relative_path: str,
    ps_raw_publication: Mapping[str, Any],
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    diagnostic_session_relative_path: str,
    diagnostic_clock_publication: Mapping[str, Any],
    diagnostic_linearity_publication: Mapping[str, Any],
) -> dict[str, Any]:
    """Meter→diagnostic raw/linearity→P/S raw DAG에서만 level evidence를 발행한다."""

    canonical, _submitted, _boundary = _ps_slice(plan, full_submitted_pcm)
    ps_raw = snapshot_published_stage2_output_master_ps_v3_raw(
        repository_root,
        ps_session_relative_path,
        ps_raw_publication,
        canonical,
        full_submitted_pcm,
        diagnostic_session_relative_path=diagnostic_session_relative_path,
        diagnostic_clock_publication=diagnostic_clock_publication,
        diagnostic_linearity_publication=diagnostic_linearity_publication,
    )
    binding = ps_raw["physical_capture_binding"]
    durable_clock = validate_output_master_diagnostic_clock_publication(
        repository_root,
        diagnostic_session_relative_path,
        diagnostic_clock_publication,
        canonical,
        full_submitted_pcm,
    )
    evidence = build_stage2_physical_operating_level_evidence(
        signal_plan_sha256=canonical["canonical_payload_sha256"],
        capture_id=binding["capture_id"],
        hardware_identity=binding["hardware_identity"],
        meter_raw_artifact=binding["meter_raw_artifact"],
        meter_receipt_artifact=binding["meter_receipt_artifact"],
        calibration_evidence_artifact=binding["calibration_evidence_artifact"],
        diagnostic_raw_artifact=durable_clock["receipt"]["raw_artifact"],
        diagnostic_authorization_artifact=_artifact_ref(
            diagnostic_linearity_publication, label="diagnostic linearity v3"
        ),
        ps_raw_artifact=_artifact_ref(ps_raw_publication, label="P/S v3 raw"),
    )
    target = output_master_ps_v3_session_targets(ps_session_relative_path)[
        "physical_level"
    ]
    published = publish_repository_bytes_noreplace(
        repository_root,
        target,
        _canonical_json_bytes(evidence) + b"\n",
        mode=0o600,
        preserve_recovery_link=True,
        recovery_tag="stage2_output_master_ps_v3_physical_level",
    )
    return {**published, "evidence": evidence}


def validate_published_stage2_output_master_ps_v3_physical_level(
    repository_root: str,
    ps_session_relative_path: str,
    physical_level_publication: Mapping[str, Any],
    ps_raw_publication: Mapping[str, Any],
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    diagnostic_session_relative_path: str,
    diagnostic_clock_publication: Mapping[str, Any],
    diagnostic_linearity_publication: Mapping[str, Any],
) -> dict[str, Any]:
    canonical, _submitted, _boundary = _ps_slice(plan, full_submitted_pcm)
    target = output_master_ps_v3_session_targets(ps_session_relative_path)[
        "physical_level"
    ]
    ref = _artifact_ref(physical_level_publication, label="P/S v3 physical level")
    if ref["path"] != target:
        raise Stage2MeasurementV2Error("P/S v3 physical level publication path가 다릅니다")
    with RepositoryFileGuard(repository_root, target, label="P/S v3 physical level") as guard:
        if guard.sha256 != ref["sha256"]:
            raise Stage2MeasurementV2Error("P/S v3 physical level file SHA가 다릅니다")
        try:
            decoded = json.loads(guard.bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Stage2MeasurementV2Error("P/S v3 physical level JSON이 유효하지 않습니다") from exc
        if not isinstance(decoded, Mapping) or guard.bytes != _canonical_json_bytes(decoded) + b"\n":
            raise Stage2MeasurementV2Error("P/S v3 physical level bytes가 canonical이 아닙니다")
        guard.verify()
    try:
        evidence = validate_stage2_physical_operating_level_evidence(decoded)
    except (TypeError, ValueError) as exc:
        raise Stage2MeasurementV2Error(str(exc)) from exc
    ps_raw = snapshot_published_stage2_output_master_ps_v3_raw(
        repository_root,
        ps_session_relative_path,
        ps_raw_publication,
        canonical,
        full_submitted_pcm,
        diagnostic_session_relative_path=diagnostic_session_relative_path,
        diagnostic_clock_publication=diagnostic_clock_publication,
        diagnostic_linearity_publication=diagnostic_linearity_publication,
    )
    durable_clock = validate_output_master_diagnostic_clock_publication(
        repository_root,
        diagnostic_session_relative_path,
        diagnostic_clock_publication,
        canonical,
        full_submitted_pcm,
    )
    binding = ps_raw["physical_capture_binding"]
    lineage = evidence["artifact_lineage"]
    if (
        evidence["signal_plan_sha256"] != canonical["canonical_payload_sha256"]
        or evidence["capture_id"] != binding["capture_id"]
        or evidence["hardware_identity"] != binding["hardware_identity"]
        or lineage["meter_raw"] != binding["meter_raw_artifact"]
        or lineage["meter_receipt"] != binding["meter_receipt_artifact"]
        or lineage["calibration_evidence"]
        != binding["calibration_evidence_artifact"]
        or lineage["diagnostic_phase_raw"]
        != durable_clock["receipt"]["raw_artifact"]
        or lineage["diagnostic_authorization"]
        != _artifact_ref(diagnostic_linearity_publication, label="diagnostic linearity")
        or lineage["ps_phase_raw"]
        != _artifact_ref(ps_raw_publication, label="P/S v3 raw")
    ):
        raise Stage2MeasurementV2Error("P/S v3 physical level lineage/raw DAG가 다릅니다")
    return {"evidence": evidence, "ps_raw": ps_raw, "publication": ref}


def estimate_stage2_output_master_ps_clock_and_resample(
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    captured_input_pcm: np.ndarray,
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    """P/S stream 자체의 pilot로 q를 추정하고 가변 input을 DAC grid로 옮긴다."""

    canonical, submitted, _boundary = _ps_slice(plan, full_submitted_pcm)
    structure = inspect_stage2_output_master_ps_v3_raw_structure(
        canonical, full_submitted_pcm, captured_input_pcm, telemetry
    )
    origin = telemetry.get("input_frame_cursor_at_output_start")
    if (
        isinstance(origin, (bool, np.bool_))
        or not isinstance(origin, (int, np.integer))
        or int(origin) < PRE_ROLL_FRAMES
        or int(origin) + len(submitted) > len(captured_input_pcm)
    ):
        raise Stage2MeasurementV2Error(
            "output-master P/S software output-start marker/crop support가 잘못됐습니다"
        )
    origin = int(origin)
    # Marker는 hardware timestamp authority가 아니라 q fit을 위한 local time gauge다.
    # 고정 callback/DAC/acoustic intercept는 fitted P와 S 모두에 남고 상대 P-S에서 상쇄된다.
    local = np.ascontiguousarray(
        np.asarray(captured_input_pcm)[origin : origin + len(submitted)], dtype="<i4"
    )
    clock = estimate_stage2_ps_local_clock(canonical, submitted, local)
    if clock.get("schema") != CLOCK_SCHEMA or clock.get("passed") is not True:
        raise Stage2MeasurementV2Error("output-master P/S local global clock가 PASS가 아닙니다")
    q = float(clock["estimated_rate_ratio"])
    output_grid: dict[str, np.ndarray] = {}
    transform_details: dict[str, Any] = {}
    transform_receipts: dict[str, Any] = {}
    for kind in ("cubic", "linear"):
        output_grid[kind], transform_details[kind] = _resample_capture_to_output_grid(
            captured_input_pcm,
            output_origin_input_frame=origin,
            rate_ratio=(
                q if kind == "cubic" else float(clock["linear_rate_ratio"])
            ),
            output_frames=len(submitted),
            interpolation_kind=kind,
        )
        transform: dict[str, Any] = {
            "schema": PS_DAC_GRID_TRANSFORM_SCHEMA,
            "source_signal_plan_sha256": canonical["canonical_payload_sha256"],
            "ps_submitted_pcm_sha256": _array_sha256(submitted),
            "captured_input_pcm_sha256": _array_sha256(
                np.asarray(captured_input_pcm)
            ),
            "ps_local_clock_receipt_sha256": clock["canonical_payload_sha256"],
            "interpolation": transform_details[kind],
            "interpolation_kind": kind,
            "output_grid_frames": len(output_grid[kind]),
            "output_grid_pcm_sha256": _array_sha256(output_grid[kind]),
            "input_output_frame_identity_claimed": False,
            "absolute_hardware_clock_authority_claimed": False,
            "relative_ps_time_gauge_only": True,
            "passed": True,
        }
        transform["canonical_payload_sha256"] = _payload_sha256(transform)
        transform_receipts[kind] = transform
    receipt: dict[str, Any] = {
        "schema": PS_CLOCK_ADAPTER_SCHEMA,
        "status": "PASS_RELATIVE_PS_LOCAL_CLOCK_AND_DAC_GRID_TRANSFORM",
        "source_signal_plan_sha256": canonical["canonical_payload_sha256"],
        "raw_structure_receipt_sha256": structure["canonical_payload_sha256"],
        "software_output_start_input_frame": origin,
        "software_marker_is_absolute_hardware_latency_authority": False,
        "constant_time_gauge_retained_in_both_plants": True,
        "relative_p_minus_s_time_gauge_cancels": True,
        "ps_local_clock_receipt": clock,
        "ps_local_clock_receipt_sha256": clock["canonical_payload_sha256"],
        "output_grid_transform_receipts": transform_receipts,
        "cubic_linear_clock_endpoint_disagreement_samples": clock[
            "cubic_linear_endpoint_disagreement_samples"
        ],
        "passed": True,
        "absolute_hardware_clock_authority_claimed": False,
        "plant_identification_eligible": False,
        "canonical_training_eligible": False,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return {
        "receipt": receipt,
        "cubic_output_grid_pcm": output_grid["cubic"],
        "linear_output_grid_pcm": output_grid["linear"],
        "cubic_transform_receipt": transform_receipts["cubic"],
        "linear_transform_receipt": transform_receipts["linear"],
    }


def _fir_crosscheck_rows(
    cubic_arrays: Mapping[str, np.ndarray], linear_arrays: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    frequency = np.fft.rfftfreq(65_536, 1.0 / SAMPLE_RATE)
    contract = Stage2TwoKilohertzContract.canonical()
    rows: list[dict[str, Any]] = []
    for path in ("primary", "secondary"):
        name = f"{path}_fir_by_mic"
        left = np.asarray(cubic_arrays[name])
        right = np.asarray(linear_arrays[name])
        if left.shape != right.shape or left.ndim != 2 or left.shape[0] != 2:
            raise Stage2MeasurementV2Error(
                "cubic/linear P/S FIR array shape가 다릅니다"
            )
        for microphone, microphone_name in enumerate(("ERR", "REF")):
            a = np.fft.rfft(left[microphone], n=65_536)
            b = np.fft.rfft(right[microphone], n=65_536)
            for lower, upper in contract.physical_identification_subbands_hz:
                mask = (frequency >= lower) & (frequency < upper)
                denominator = max(
                    float(np.linalg.norm(a[mask]) * np.linalg.norm(b[mask])),
                    np.finfo(np.float64).tiny,
                )
                agreement = float(abs(np.vdot(a[mask], b[mask])) / denominator)
                rows.append(
                    {
                        "path": path,
                        "microphone": microphone_name,
                        "band_hz": [float(lower), float(upper)],
                        "cubic_linear_complex_agreement": agreement,
                        "minimum_complex_agreement": 0.95,
                        "passed": bool(agreement >= 0.95),
                    }
                )
    return rows


def _analyse_validated_stage2_output_master_ps_v3_capture(
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    durable_diagnostic: Mapping[str, Any],
    captured_ps_input_pcm: np.ndarray,
    ps_telemetry: Mapping[str, Any],
    operating_level_evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """clock-corrected diagnostic + P/S raw를 DPSS fit/holdout까지 교차검증한다."""

    canonical, _submitted, _boundary = _ps_slice(plan, full_submitted_pcm)
    diagnostic = _analyse_validated_stage2_output_master_diagnostic_linearity_v3(
        canonical, full_submitted_pcm, durable_diagnostic
    )
    if diagnostic["receipt"]["passed"] is not True:
        raise Stage2MeasurementV2Error(
            "clock-corrected diagnostic linearity가 FAIL이라 P/S 분석할 수 없습니다"
        )
    ps = estimate_stage2_output_master_ps_clock_and_resample(
        canonical, full_submitted_pcm, captured_ps_input_pcm, ps_telemetry
    )
    analyses: dict[str, Any] = {}
    arrays: dict[str, Mapping[str, np.ndarray]] = {}
    for kind in ("cubic", "linear"):
        analyses[kind], arrays[kind] = analyse_stage2_v2_capture(
            canonical,
            full_submitted_pcm,
            diagnostic[f"{kind}_output_grid_pcm"],
            ps[f"{kind}_output_grid_pcm"],
            diagnostic_transport_counters={
                "xrun": 0,
                "clip": 0,
                "callback_status": 0,
            },
            ps_transport_counters={"xrun": 0, "clip": 0, "callback_status": 0},
            operating_level_evidence=operating_level_evidence,
            diagnostic_quiet_start_frame=8_192,
            ps_output_grid_transform_receipt=ps[
                f"{kind}_transform_receipt"
            ],
        )
    rows = _fir_crosscheck_rows(arrays["cubic"], arrays["linear"])
    if not rows or not all(row["passed"] for row in rows):
        raise Stage2MeasurementV2Error(
            "output-master cubic/linear DPSS P/S subband crosscheck가 실패했습니다"
        )
    receipt: dict[str, Any] = {
        "schema": PS_PHYSICAL_ANALYSIS_V3_SCHEMA,
        "status": "PASS_RELATIVE_PS_CANDIDATE_PLANT_BINDING_STILL_REQUIRED",
        "source_signal_plan_sha256": canonical["canonical_payload_sha256"],
        "diagnostic_linearity_receipt_sha256": diagnostic["receipt"][
            "canonical_payload_sha256"
        ],
        "ps_clock_adapter_receipt_sha256": ps["receipt"][
            "canonical_payload_sha256"
        ],
        "cubic_analysis_receipt_sha256": analyses["cubic"][
            "canonical_payload_sha256"
        ],
        "linear_analysis_receipt_sha256": analyses["linear"][
            "canonical_payload_sha256"
        ],
        "cubic_physical_subband_rows": analyses["cubic"]["subband_rows"],
        "linear_physical_subband_rows": analyses["linear"]["subband_rows"],
        "cubic_actuator_feasibility": analyses["cubic"]["actuator_feasibility"],
        "linear_actuator_feasibility": analyses["linear"]["actuator_feasibility"],
        "cubic_coarse_zeros_before_fir_samples": analyses["cubic"][
            "coarse_zeros_before_fir_samples"
        ],
        "linear_coarse_zeros_before_fir_samples": analyses["linear"][
            "coarse_zeros_before_fir_samples"
        ],
        "cubic_linear_fir_subband_rows": rows,
        "relative_ps_authority": True,
        "absolute_hardware_clock_authority_claimed": False,
        "plant_identification_eligible": True,
        "plant_binding_published": False,
        "canonical_training_eligible": False,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt, dict(arrays["cubic"])


def analyse_stage2_output_master_ps_v3_capture(
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    repository_root: str,
    diagnostic_session_relative_path: str,
    diagnostic_clock_publication: Mapping[str, Any],
    diagnostic_linearity_publication: Mapping[str, Any],
    ps_session_relative_path: str,
    ps_raw_publication: Mapping[str, Any],
    physical_level_publication: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Public P/S analyzer: 모든 raw/receipt/level bytes를 repository에서 재개방한다."""

    validated_level = validate_published_stage2_output_master_ps_v3_physical_level(
        repository_root,
        ps_session_relative_path,
        physical_level_publication,
        ps_raw_publication,
        plan,
        full_submitted_pcm,
        diagnostic_session_relative_path=diagnostic_session_relative_path,
        diagnostic_clock_publication=diagnostic_clock_publication,
        diagnostic_linearity_publication=diagnostic_linearity_publication,
    )
    durable = validate_output_master_diagnostic_clock_publication(
        repository_root,
        diagnostic_session_relative_path,
        diagnostic_clock_publication,
        plan,
        full_submitted_pcm,
    )
    ps_raw = validated_level["ps_raw"]
    receipt, arrays = _analyse_validated_stage2_output_master_ps_v3_capture(
        plan,
        full_submitted_pcm,
        durable_diagnostic=durable,
        captured_ps_input_pcm=ps_raw["captured_pcm"],
        ps_telemetry=ps_raw["telemetry"],
        operating_level_evidence=validated_level["evidence"],
    )
    receipt = {
        **receipt,
        "diagnostic_clock_artifact": _artifact_ref(
            diagnostic_clock_publication, label="diagnostic clock"
        ),
        "diagnostic_linearity_artifact": _artifact_ref(
            diagnostic_linearity_publication, label="diagnostic linearity"
        ),
        "ps_raw_artifact": _artifact_ref(ps_raw_publication, label="P/S v3 raw"),
        "physical_level_artifact": _artifact_ref(
            physical_level_publication, label="P/S v3 physical level"
        ),
        "physical_level_evidence_sha256": validated_level["evidence"][
            "canonical_payload_sha256"
        ],
    }
    receipt.pop("canonical_payload_sha256", None)
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt, arrays


def inspect_stage2_output_master_ps_v3_raw_structure(
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    captured_input_pcm: np.ndarray,
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    """가변 input/output 길이를 보존하되 아직 plant로 승격하지 않는 adapter.

    이 함수는 순수 offline structural validator다. 어떤 raw도 쓰지 않으며 clock correction,
    P/S fitting 또는 authority를 주장하지 않는다.
    """

    canonical, submitted, _boundary = _ps_slice(plan, full_submitted_pcm)
    captured = np.asarray(captured_input_pcm)
    transport = validate_output_master_success_telemetry(
        telemetry,
        captured_pcm=captured,
        expected_submitted_pcm=submitted,
    )
    clip_count = int(
        np.count_nonzero(np.abs(captured.astype(np.int64)) >= 2**31 - 1)
    )
    receipt: dict[str, Any] = {
        "schema": PS_V3_RAW_STRUCTURE_SCHEMA,
        "status": "RAW_STRUCTURE_VALID_ANALYSIS_AND_AUTHORITY_BLOCKED",
        "source_signal_plan_sha256": canonical["canonical_payload_sha256"],
        "ps_submitted_pcm_sha256": _array_sha256(submitted),
        "captured_input_pcm_sha256": _array_sha256(captured),
        "output_clock_frames": len(submitted),
        "input_clock_frames": len(captured),
        "input_output_length_difference_frames": len(captured) - len(submitted),
        "input_output_frame_identity_claimed": False,
        "input_clock_origin_or_rate_inferred_from_length": False,
        "transport_receipt": transport,
        "adc_clip_count": clip_count,
        "required_authority_adapters": list(_UNIMPLEMENTED_AUTHORITY_BLOCKERS),
        "raw_first_analysis_required": True,
        "ps_clock_authority_granted": False,
        "plant_identification_eligible": False,
        "canonical_training_eligible": False,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt


def _assess_stage2_output_master_ps_v3_admission_after_cli_authority(
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    repository_root: str | None = None,
    diagnostic_session_relative_path: str | None = None,
    diagnostic_clock_publication: Mapping[str, Any] | None = None,
    diagnostic_linearity_publication: Mapping[str, Any] | None = None,
    ps_capture_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """P/S backend open 직전의 typed admission을 계산한다."""

    v3_plan, _submitted = build_stage2_output_master_ps_v3_plan(
        plan, full_submitted_pcm
    )
    supplied = (
        repository_root is not None,
        diagnostic_session_relative_path is not None,
        diagnostic_clock_publication is not None,
        diagnostic_linearity_publication is not None,
        ps_capture_metadata is not None,
    )
    if any(supplied) and not all(supplied):
        raise Stage2MeasurementV2Error(
            "diagnostic repository/session/clock/linearity과 P/S capture metadata는 모두 함께 필요합니다"
        )

    blockers: list[str] = []
    diagnostic_binding: dict[str, Any] | None = None
    if all(supplied):
        assert repository_root is not None
        assert diagnostic_session_relative_path is not None
        assert diagnostic_clock_publication is not None
        assert diagnostic_linearity_publication is not None
        assert ps_capture_metadata is not None
        validated = validate_output_master_diagnostic_clock_publication(
            repository_root,
            diagnostic_session_relative_path,
            diagnostic_clock_publication,
            plan,
            full_submitted_pcm,
        )
        receipt = validated["receipt"]
        linearity = validate_published_stage2_output_master_diagnostic_linearity_v3(
            repository_root,
            diagnostic_session_relative_path,
            diagnostic_clock_publication,
            diagnostic_linearity_publication,
            plan,
            full_submitted_pcm,
        )
        continuity = _physical_continuity_binding(
            repository_root,
            validated["physical_capture_binding"],
            ps_capture_metadata,
            require_fresh_meter=True,
        )
        diagnostic_binding = {
            "session_relative_path": diagnostic_session_relative_path,
            "clock_artifact": {
                "path": diagnostic_clock_publication["path"],
                "sha256": diagnostic_clock_publication["sha256"],
            },
            "clock_receipt_sha256": receipt["canonical_payload_sha256"],
            "raw_artifact": dict(receipt["raw_artifact"]),
            "global_clock_passed": bool(receipt["passed"]),
            "linearity_artifact": _artifact_ref(
                diagnostic_linearity_publication, label="diagnostic linearity"
            ),
            "linearity_receipt_sha256": linearity["receipt"][
                "canonical_payload_sha256"
            ],
            "clock_corrected_linearity_passed": bool(
                linearity["receipt"]["passed"]
            ),
            "physical_continuity_receipt": continuity,
        }
        if receipt["passed"] is not True:
            blockers.append("DIAGNOSTIC_GLOBAL_AFFINE_CLOCK_NOT_PASS")
        if linearity["receipt"]["passed"] is not True:
            blockers.append("DIAGNOSTIC_CLOCK_CORRECTED_LINEARITY_NOT_PASS")
    else:
        blockers.append("DURABLE_DIAGNOSTIC_RAW_AND_CLOCK_RECEIPT_REQUIRED")
    blockers.extend(_UNIMPLEMENTED_AUTHORITY_BLOCKERS)

    admission: dict[str, Any] = {
        "schema": PS_V3_ADMISSION_SCHEMA,
        "status": (
            "PASS_OUTPUT_MASTER_PS_V3_BACKEND_MAY_OPEN_RAW_FIRST"
            if not blockers
            else "BLOCKED_OUTPUT_MASTER_PS_V3_NOT_YET_PRODUCTION_READY"
        ),
        "v3_plan_sha256": v3_plan["canonical_payload_sha256"],
        "source_signal_plan_sha256": v3_plan["source_signal_plan_sha256"],
        "diagnostic_evidence_binding": diagnostic_binding,
        "blockers": blockers,
        "diagnostic_clock_pass_alone_authorizes_ps": False,
        "legacy_combined_raw_may_satisfy_any_blocker": False,
        "ps_stream_may_open": not blockers,
        "ps_backend_calls_allowed": 1 if not blockers else 0,
        "plant_identification_eligible": False,
        "canonical_training_eligible": False,
    }
    admission["canonical_payload_sha256"] = _payload_sha256(admission)
    return admission


def _run_stage2_output_master_ps_v3_after_cli_authority(
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    capture_callable: Callable[..., Any],
    repository_root: str | None = None,
    diagnostic_session_relative_path: str | None = None,
    diagnostic_clock_publication: Mapping[str, Any] | None = None,
    diagnostic_linearity_publication: Mapping[str, Any] | None = None,
    ps_session_relative_path: str | None = None,
    ps_capture_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Typed admission 후 1회 capture를 raw-first no-replace로 보존한다."""

    admission = _assess_stage2_output_master_ps_v3_admission_after_cli_authority(
        plan,
        full_submitted_pcm,
        repository_root=repository_root,
        diagnostic_session_relative_path=diagnostic_session_relative_path,
        diagnostic_clock_publication=diagnostic_clock_publication,
        diagnostic_linearity_publication=diagnostic_linearity_publication,
        ps_capture_metadata=ps_capture_metadata,
    )
    if admission["ps_stream_may_open"] is not True:
        result: dict[str, Any] = {
            "schema": PS_V3_EXECUTION_SCHEMA,
            "status": "BLOCKED_BEFORE_AUDIO_BACKEND_OR_PS_OUTPUT",
            "admission": admission,
            "capture_callable_invoked": False,
            "ps_backend_calls": 0,
            "ps_raw_written": False,
            "plant_identification_eligible": False,
            "canonical_training_eligible": False,
        }
        result["canonical_payload_sha256"] = _payload_sha256(result)
        return result
    if (
        repository_root is None
        or diagnostic_session_relative_path is None
        or diagnostic_clock_publication is None
        or diagnostic_linearity_publication is None
        or ps_session_relative_path is None
        or ps_capture_metadata is None
    ):
        raise AssertionError("PASS P/S v3 admission에 required publication/session이 없습니다")
    _canonical, submitted, _boundary = _ps_slice(plan, full_submitted_pcm)
    try:
        captured, telemetry = capture_callable(submitted_pcm=submitted)
    except OutputMasterCaptureFailure as failure:
        partial = publish_stage2_output_master_ps_v3_partial_raw_no_replace(
            repository_root,
            ps_session_relative_path,
            plan,
            full_submitted_pcm,
            failure=failure,
            capture_metadata=ps_capture_metadata,
            diagnostic_session_relative_path=diagnostic_session_relative_path,
            diagnostic_clock_publication=diagnostic_clock_publication,
            diagnostic_linearity_publication=diagnostic_linearity_publication,
        )
        result = {
            "schema": PS_V3_EXECUTION_SCHEMA,
            "status": "FAILED_PS_TRANSPORT_PARTIAL_RAW_PRESERVED_NO_AUTOMATIC_RETRY",
            "admission": admission,
            "capture_callable_invoked": True,
            "ps_backend_calls": 1,
            "ps_raw_written": False,
            "partial_raw_publication": {
                "path": partial["path"],
                "sha256": partial["sha256"],
            },
            "automatic_retry_allowed": False,
            "plant_identification_eligible": False,
            "canonical_training_eligible": False,
        }
        result["canonical_payload_sha256"] = _payload_sha256(result)
        return result
    try:
        raw = publish_stage2_output_master_ps_v3_raw_no_replace(
            repository_root,
            ps_session_relative_path,
            plan,
            full_submitted_pcm,
            diagnostic_session_relative_path=diagnostic_session_relative_path,
            diagnostic_clock_publication=diagnostic_clock_publication,
            diagnostic_linearity_publication=diagnostic_linearity_publication,
            captured_ps_input_pcm=captured,
            ps_telemetry=telemetry,
            capture_metadata=ps_capture_metadata,
        )
    except Exception as publish_error:
        # Backend이 valid success arrays를 반환한 이후에는 meter expiry,
        # diagnostic re-open race, target collision 등 어떤 publisher 예외도 실물
        # raw를 유실시켜서는 안 된다. 성공으로 위장하지 않는
        # partial/recovery target에 actual 두 clock 축과 mask를 먼저 보존한다.
        actual = np.asarray(telemetry.get("actual_submitted_pcm", submitted))
        capture_mask = np.asarray(
            telemetry.get("capture_valid_mask", np.ones(len(captured), dtype="bool"))
        )
        submitted_mask = np.asarray(
            telemetry.get("submitted_valid_mask", np.ones(len(submitted), dtype="bool"))
        )
        recovery_failure = OutputMasterCaptureFailure(
            f"post-capture raw publisher failed: {publish_error}",
            np.asarray(captured),
            actual,
            capture_mask,
            submitted_mask,
            dict(telemetry),
        )
        try:
            partial = publish_stage2_output_master_ps_v3_partial_raw_no_replace(
                repository_root,
                ps_session_relative_path,
                plan,
                full_submitted_pcm,
                failure=recovery_failure,
                capture_metadata=ps_capture_metadata,
                diagnostic_session_relative_path=diagnostic_session_relative_path,
                diagnostic_clock_publication=diagnostic_clock_publication,
                diagnostic_linearity_publication=diagnostic_linearity_publication,
            )
        except Exception as recovery_error:
            raise RuntimeError(
                "P/S capture 후 success raw publication과 recovery raw publication이 모두 실패했습니다: "
                f"success={publish_error}; recovery={recovery_error}"
            ) from recovery_error
        result = {
            "schema": PS_V3_EXECUTION_SCHEMA,
            "status": "FAILED_POST_CAPTURE_VALIDATION_RECOVERY_RAW_PRESERVED",
            "admission": admission,
            "capture_callable_invoked": True,
            "ps_backend_calls": 1,
            "ps_raw_written": False,
            "partial_raw_publication": {
                "path": partial["path"],
                "sha256": partial["sha256"],
            },
            "post_capture_error": str(publish_error),
            "automatic_retry_allowed": False,
            "plant_identification_eligible": False,
            "canonical_training_eligible": False,
        }
        result["canonical_payload_sha256"] = _payload_sha256(result)
        return result
    try:
        reloaded = snapshot_published_stage2_output_master_ps_v3_raw(
            repository_root,
            ps_session_relative_path,
            raw,
            plan,
            full_submitted_pcm,
            diagnostic_session_relative_path=diagnostic_session_relative_path,
            diagnostic_clock_publication=diagnostic_clock_publication,
            diagnostic_linearity_publication=diagnostic_linearity_publication,
        )
    except Exception as reload_error:
        result = {
            "schema": PS_V3_EXECUTION_SCHEMA,
            "status": "FAILED_POST_RAW_RELOAD_SUCCESS_RAW_ALREADY_PRESERVED",
            "admission": admission,
            "capture_callable_invoked": True,
            "ps_backend_calls": 1,
            "ps_raw_written": True,
            "ps_raw_publication": {
                "path": raw["path"],
                "sha256": raw["sha256"],
            },
            "post_raw_reload_error": str(reload_error),
            "automatic_retry_allowed": False,
            "plant_identification_eligible": False,
            "canonical_training_eligible": False,
        }
        result["canonical_payload_sha256"] = _payload_sha256(result)
        return result
    result = {
        "schema": PS_V3_EXECUTION_SCHEMA,
        "status": "PASS_PS_RAW_DURABLY_PUBLISHED_RELOADED_ANALYSIS_STILL_REQUIRED",
        "admission": admission,
        "capture_callable_invoked": True,
        "ps_backend_calls": 1,
        "ps_raw_written": True,
        "ps_raw_publication": reloaded["artifact_ref"],
        "plant_identification_eligible": False,
        "canonical_training_eligible": False,
    }
    result["canonical_payload_sha256"] = _payload_sha256(result)
    return result


def assess_stage2_output_master_ps_v3_admission(
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    **_untrusted_live_arguments: Any,
) -> dict[str, Any]:
    """Public library boundary는 audio admission을 절대 발행하지 않는다.

    실제 ALSA/device/origin-dev 상태를 강제하는 tracked CLI만 private raw
    state machine을 호출한다. Caller가 만든 mapping/callback은 음향 권한이 아니다.
    """

    v3_plan, _submitted = build_stage2_output_master_ps_v3_plan(
        plan, full_submitted_pcm
    )
    receipt: dict[str, Any] = {
        "schema": PS_V3_ADMISSION_SCHEMA,
        "status": "BLOCKED_PUBLIC_LIBRARY_CANNOT_AUTHORIZE_LIVE_AUDIO",
        "v3_plan_sha256": v3_plan["canonical_payload_sha256"],
        "source_signal_plan_sha256": v3_plan["source_signal_plan_sha256"],
        "diagnostic_evidence_binding": None,
        "blockers": ["TRACKED_CLI_REPOSITORY_AND_PHYSICAL_PREOPEN_AUTHORITY_REQUIRED"],
        "diagnostic_clock_pass_alone_authorizes_ps": False,
        "legacy_combined_raw_may_satisfy_any_blocker": False,
        "ps_stream_may_open": False,
        "ps_backend_calls_allowed": 0,
        "plant_identification_eligible": False,
        "canonical_training_eligible": False,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt


def run_stage2_output_master_ps_v3_if_admitted(
    plan: Mapping[str, Any],
    full_submitted_pcm: np.ndarray,
    *,
    capture_callable: Callable[..., Any],
    **untrusted_live_arguments: Any,
) -> dict[str, Any]:
    """Public direct caller의 callback은 어떤 mapping으로도 호출하지 않는다."""

    del capture_callable
    admission = assess_stage2_output_master_ps_v3_admission(
        plan, full_submitted_pcm, **untrusted_live_arguments
    )
    result: dict[str, Any] = {
        "schema": PS_V3_EXECUTION_SCHEMA,
        "status": "BLOCKED_BEFORE_AUDIO_BACKEND_OR_PS_OUTPUT",
        "admission": admission,
        "capture_callable_invoked": False,
        "ps_backend_calls": 0,
        "ps_raw_written": False,
        "plant_identification_eligible": False,
        "canonical_training_eligible": False,
    }
    result["canonical_payload_sha256"] = _payload_sha256(result)
    return result


__all__ = [
    "DIAGNOSTIC_LINEARITY_V3_SCHEMA",
    "PS_CLOCK_ADAPTER_SCHEMA",
    "PS_DAC_GRID_TRANSFORM_SCHEMA",
    "PS_PHYSICAL_ANALYSIS_V3_SCHEMA",
    "PS_V3_ADMISSION_SCHEMA",
    "PS_V3_EXECUTION_SCHEMA",
    "PS_V3_PLAN_SCHEMA",
    "PS_V3_RAW_STRUCTURE_SCHEMA",
    "PS_V3_RAW_SCHEMA",
    "PS_V3_PARTIAL_RAW_SCHEMA",
    "analyse_stage2_output_master_diagnostic_linearity_v3",
    "analyse_stage2_output_master_ps_v3_capture",
    "assess_stage2_output_master_ps_v3_admission",
    "build_stage2_output_master_ps_v3_plan",
    "estimate_stage2_output_master_ps_clock_and_resample",
    "inspect_stage2_output_master_ps_v3_raw_structure",
    "output_master_ps_v3_session_targets",
    "publish_stage2_output_master_diagnostic_linearity_v3_no_replace",
    "publish_stage2_output_master_ps_v3_partial_raw_no_replace",
    "publish_stage2_output_master_ps_v3_physical_level_no_replace",
    "publish_stage2_output_master_ps_v3_raw_no_replace",
    "run_stage2_output_master_ps_v3_if_admitted",
    "snapshot_published_stage2_output_master_ps_v3_raw",
    "validate_published_stage2_output_master_diagnostic_linearity_v3",
    "validate_published_stage2_output_master_ps_v3_physical_level",
    "validate_output_master_diagnostic_clock_publication",
]
