"""Committed v5 raw에서 clock, 절대 지연, compact P/S를 복구하는 offline core.

외부 marker/window mapping은 받지 않는다. committed v5 plan과 actual submitted PCM의
SHA 및 layout을 검증한 뒤 plan 중앙 cyclic row만 유도해 사용한다. 오디오 장치, 파일
publisher, 네트워크는 이 모듈의 범위 밖이다.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import LinearOperator, lsmr

from ..audio_duplex_v5 import DUPLEX_TELEMETRY_SCHEMA as _DUPLEX_TELEMETRY_SCHEMA
from .fullband_causal_v5 import (
    BLOCK,
    CONDITION_AUDIT_SUPPORT,
    CYCLIC_PREFIX,
    CYCLIC_SUFFIX,
    FS,
    PERIOD,
    ROLES,
    SLOT_FRAMES,
    build_plan_v5,
    estimate_common_clock_from_waveforms_v5,
    exact_shifted_condition_audit_v5,
)
from .timing import PlantDelays


EXPECTED_PLAN_SHA256 = "32a79b3700b457dc40373dc4dd0969301287baea7100b1ec5edd86ea907ee127"
EXPECTED_PCM_SHA256 = "c18416e4066556479fd317659d908c215e6662d08f5bfa9d50e4ac63971c4aff"
ANALYSIS_SCHEMA = "fullband_committed_v5_live_delay_core_v3"

MAX_COARSE_DELAY_SAMPLES = 4_800
COMPACT_PRE_ROLL_SAMPLES = 256
COMPACT_SUPPORT_SAMPLES = CONDITION_AUDIT_SUPPORT
FULL_CAUSAL_SUPPORT_SAMPLES = MAX_COARSE_DELAY_SAMPLES + COMPACT_SUPPORT_SAMPLES

MAX_CLOCK_INTERPOLATION_ENDPOINT_DISAGREEMENT_SAMPLES = 0.06755189029558945
MAX_FIT_ROLE_PEAK_DISAGREEMENT_SAMPLES = 0.15
MAX_EARLY_ENERGY_RATIO = 1.0e-4
MAX_TAIL_ENERGY_RATIO = 1.0e-4
MAX_COMPACT_ROUNDTRIP_RELATIVE = 0.02
MAX_FRACTIONAL_ROUNDTRIP_ERROR_SAMPLES = 0.08
MAX_BAND_RELATIVE_RESIDUAL = 0.10
MIN_BAND_COMPLEX_VECTOR_AGREEMENT = 0.995
MIN_RESPONSE_TO_NOISE_DB = 20.0
MIN_TARGET_BIN_DENSITY = 0.95
LSMR_MAX_ITERATIONS = 800
LSMR_ACCEPTED_STOP_CODES = (1, 2)
MAX_SOLVER_RELATIVE_RESIDUAL = 0.15
MAX_SOLVER_NORMAL_EQUATION_RELATIVE_RESIDUAL = 1.0e-7

PATHS = ("primary", "secondary")
MICROPHONES = ("ERR", "REF")
FIT_ROLES = ("fit_a", "fit_b")
FINAL_FIXED_AVERAGE_WEIGHTS = (0.5, 0.5)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} key 집합이 exact하지 않습니다")


def validate_committed_plan_and_derive_windows(
    plan: Mapping[str, Any], submitted_pcm: np.ndarray
) -> tuple[dict[str, Any], dict[tuple[str, str], tuple[int, int]]]:
    """SHA-pinned committed plan에서만 중앙 window를 유도한다."""

    submitted = np.asarray(submitted_pcm)
    if submitted.dtype != np.int16 or submitted.ndim != 2 or submitted.shape[1] != 2:
        raise ValueError("actual submitted PCM은 exact int16 [frame,2]여야 합니다")
    if plan.get("canonical_payload_sha256") != EXPECTED_PLAN_SHA256:
        raise ValueError("committed v5 plan SHA가 아닙니다")
    payload = {key: value for key, value in plan.items() if key != "canonical_payload_sha256"}
    if _payload_sha256(payload) != EXPECTED_PLAN_SHA256:
        raise ValueError("v5 plan payload가 SHA와 일치하지 않습니다")
    if _array_sha256(submitted) != EXPECTED_PCM_SHA256:
        raise ValueError("committed v5 actual PCM SHA가 아닙니다")
    if plan.get("actual_submitted_pcm_sha256") != EXPECTED_PCM_SHA256:
        raise ValueError("plan의 actual PCM SHA가 pinned PCM과 다릅니다")

    expected_plan, expected_pcm = build_plan_v5()
    if dict(plan) != expected_plan or not np.array_equal(submitted, expected_pcm):
        raise ValueError("입력 plan/PCM이 committed builder의 exact 결과와 다릅니다")

    layout = list(plan["layout"])
    expected_order = ["pilot_only_lead"] + [
        f"{path}_{role}_slot" for role in ROLES for path in PATHS
    ] + ["pilot_only_tail"]
    if [row.get("kind") for row in layout] != expected_order:
        raise ValueError("v5 layout 순서가 committed order와 다릅니다")
    if len({int(row["start_frame"]) for row in layout}) != len(layout):
        raise ValueError("v5 layout start가 중복됩니다")
    if int(layout[0]["start_frame"]) != 0:
        raise ValueError("v5 layout은 frame 0에서 시작해야 합니다")
    for left, right in zip(layout, layout[1:]):
        if int(left["stop_frame"]) != int(right["start_frame"]):
            raise ValueError("v5 layout이 contiguous하지 않습니다")
    if int(layout[-1]["stop_frame"]) != len(submitted):
        raise ValueError("v5 layout 끝과 PCM frame 수가 다릅니다")

    windows: dict[tuple[str, str], tuple[int, int]] = {}
    for role in ROLES:
        for path in PATHS:
            matches = [
                row
                for row in layout
                if row.get("role") == role and row.get("path") == path
            ]
            if len(matches) != 1:
                raise ValueError(f"{role}/{path} slot이 exact 하나가 아닙니다")
            row = matches[0]
            start = int(row["start_frame"])
            stop = int(row["stop_frame"])
            central_start = int(row["central_start_frame"])
            central_stop = int(row["central_stop_frame"])
            if (
                stop - start != SLOT_FRAMES
                or central_start - start != CYCLIC_PREFIX
                or stop - central_stop != CYCLIC_SUFFIX
                or central_stop - central_start != PERIOD
                or int(row["pre_boundary_exclusion_samples"]) != CYCLIC_PREFIX
                or int(row["post_boundary_exclusion_samples"]) != CYCLIC_SUFFIX
            ):
                raise ValueError("v5 central cyclic layout 수치가 exact하지 않습니다")
            identity = (role, path)
            if identity in windows:
                raise ValueError("v5 central window identity가 중복됩니다")
            windows[identity] = (central_start, central_stop)

    receipt = {
        "schema": "fullband_committed_v5_plan_layout_v1",
        "signal_plan_payload_sha256": EXPECTED_PLAN_SHA256,
        "actual_submitted_pcm_sha256": EXPECTED_PCM_SHA256,
        "window_source": "committed_plan_layout_only_no_external_mapping",
        "role_order": list(ROLES),
        "path_order": list(PATHS),
        "central_period_samples": PERIOD,
        "pre_boundary_exclusion_samples": CYCLIC_PREFIX,
        "post_boundary_exclusion_samples": CYCLIC_SUFFIX,
        "windows": [
            {
                "role": role,
                "path": path,
                "start_frame": windows[(role, path)][0],
                "stop_frame": windows[(role, path)][1],
            }
            for role in ROLES
            for path in PATHS
        ],
    }
    receipt["sha256"] = _payload_sha256(receipt)
    return receipt, windows


def validate_duplex_telemetry_auxiliary(
    telemetry: Mapping[str, Any],
    *,
    captured_adc_pcm: np.ndarray,
    expected_submitted_pcm: np.ndarray,
) -> dict[str, Any]:
    """성공한 actual capture와 v3 telemetry를 exact 결속해 검증한다."""

    expected = {
        "schema",
        "callback_frame_semantics",
        "portaudio_xrun_status_witness",
        "hardware_sample_slip_authority",
        "watchdog_coverage",
        "sample_rate_hz",
        "block_size",
        "latency",
        "channels",
        "resolved_input_device",
        "resolved_output_device",
        "capture_monotonic_started",
        "capture_monotonic_completed",
        "capture_monotonic_elapsed_seconds",
        "watchdog_grace_seconds",
        "input_dtype",
        "output_dtype",
        "callback_sequence",
        "callback_start_frames",
        "callback_frame_counts",
        "input_buffer_adc_time",
        "output_buffer_dac_time",
        "callback_current_time",
        "callback_status_bitmask",
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
        "actual_submitted_pcm",
        "capture_valid_mask",
        "submitted_valid_mask",
    }
    _exact_keys(telemetry, expected, label="duplex telemetry")
    if telemetry["schema"] != _DUPLEX_TELEMETRY_SCHEMA:
        raise ValueError("duplex telemetry schema가 다릅니다")
    if (
        telemetry["callback_frame_semantics"]
        != "software_accounting_only_not_hardware_slip_witness"
        or telemetry["portaudio_xrun_status_witness"] is not True
        or telemetry["hardware_sample_slip_authority"] is not False
        or telemetry["watchdog_coverage"]
        != "host_wait_until_planned_frames_plus_grace_not_hardware_deadline_witness"
    ):
        raise ValueError("duplex telemetry 증거 경계 문구가 exact하지 않습니다")
    if (
        type(telemetry["sample_rate_hz"]) is not int
        or telemetry["sample_rate_hz"] != FS
        or type(telemetry["block_size"]) is not int
        or telemetry["block_size"] != BLOCK
        or telemetry["latency"] != "low"
        or type(telemetry["channels"]) is not list
        or telemetry["channels"] != [2, 2]
        or type(telemetry["resolved_input_device"]) is not int
        or telemetry["resolved_input_device"] < 0
        or type(telemetry["resolved_output_device"]) is not int
        or telemetry["resolved_output_device"] < 0
        or telemetry["input_dtype"] != "<i4"
        or telemetry["output_dtype"] != "<i2"
    ):
        raise ValueError("duplex telemetry stream 계약이 v5와 다릅니다")
    monotonic_values = [
        telemetry["capture_monotonic_started"], telemetry["capture_monotonic_completed"],
        telemetry["capture_monotonic_elapsed_seconds"], telemetry["watchdog_grace_seconds"],
    ]
    if not all(type(value) is float and np.isfinite(value) for value in monotonic_values):
        raise ValueError("duplex monotonic/watchdog telemetry가 finite float가 아닙니다")
    if (
        telemetry["capture_monotonic_completed"] < telemetry["capture_monotonic_started"]
        or telemetry["capture_monotonic_elapsed_seconds"] < 0.0
        or telemetry["watchdog_grace_seconds"] <= 0.0
    ):
        raise ValueError("duplex monotonic/watchdog telemetry 범위가 잘못됐습니다")

    captured = np.asarray(captured_adc_pcm)
    submitted = np.asarray(expected_submitted_pcm)
    if not isinstance(telemetry["actual_submitted_pcm"], np.ndarray):
        raise ValueError("actual submitted PCM은 exact ndarray여야 합니다")
    actual_submitted = np.asarray(telemetry["actual_submitted_pcm"])
    if (
        captured.dtype != np.dtype("<i4")
        or captured.ndim != 2
        or captured.shape[1] != 2
    ):
        raise ValueError("actual captured ADC는 exact <i4 [frame,2]여야 합니다")
    if (
        submitted.dtype != np.dtype("<i2")
        or submitted.ndim != 2
        or submitted.shape[1] != 2
        or submitted.shape != captured.shape
    ):
        raise ValueError("expected submitted PCM은 capture와 같은 exact <i2여야 합니다")
    if (
        actual_submitted.dtype != np.dtype("<i2")
        or actual_submitted.shape != submitted.shape
        or not np.array_equal(actual_submitted, submitted)
    ):
        raise ValueError("actual submitted PCM이 expected PCM과 exact 일치하지 않습니다")

    total = len(captured)
    if total <= 0 or total % BLOCK:
        raise ValueError("captured frame 수는 양수인 256 배수여야 합니다")
    count = total // BLOCK
    array_fields = {
        "sequence": np.asarray(telemetry["callback_sequence"]),
        "start": np.asarray(telemetry["callback_start_frames"]),
        "frames": np.asarray(telemetry["callback_frame_counts"]),
        "adc": np.asarray(telemetry["input_buffer_adc_time"]),
        "dac": np.asarray(telemetry["output_buffer_dac_time"]),
        "current": np.asarray(telemetry["callback_current_time"]),
        "status": np.asarray(telemetry["callback_status_bitmask"]),
    }
    mask_fields = {
        "capture_valid": np.asarray(telemetry["capture_valid_mask"]),
        "submitted_valid": np.asarray(telemetry["submitted_valid_mask"]),
    }
    if any(
        not isinstance(telemetry[key], np.ndarray)
        for key in (
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
    ):
        raise ValueError("duplex telemetry 배열은 exact ndarray여야 합니다")
    arrays = array_fields
    masks = mask_fields
    if any(value.ndim != 1 or len(value) != count for value in arrays.values()):
        raise ValueError("duplex callback 배열 길이가 exact capture block 수와 다릅니다")
    for name in ("sequence", "start", "frames"):
        if arrays[name].dtype != np.dtype("<i8"):
            raise ValueError(f"duplex {name} dtype은 exact <i8이어야 합니다")
    if arrays["status"].dtype != np.dtype("<u4"):
        raise ValueError("duplex status dtype은 exact <u4여야 합니다")
    for name in ("adc", "dac", "current"):
        if arrays[name].dtype != np.dtype("<f8"):
            raise ValueError(f"duplex {name} dtype은 exact <f8이어야 합니다")
    if any(
        value.dtype != np.dtype(np.bool_)
        or value.ndim != 1
        or value.shape != (total,)
        or not np.all(value)
        for value in masks.values()
    ):
        raise ValueError("duplex valid mask는 exact bool [frames] all-true여야 합니다")
    if not np.array_equal(arrays["sequence"], np.arange(count, dtype=np.int64)):
        raise ValueError("duplex callback sequence가 연속이 아닙니다")
    if not np.array_equal(arrays["start"], np.arange(count, dtype=np.int64) * BLOCK):
        raise ValueError("duplex software callback start accounting이 다릅니다")
    if np.any(arrays["frames"] != BLOCK):
        raise ValueError("duplex callback frame은 exact 256이어야 합니다")
    for name in ("adc", "dac", "current"):
        values = arrays[name].astype(np.float64, copy=False)
        if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
            raise ValueError(f"duplex {name} timestamp가 finite strict-monotonic이 아닙니다")
    if np.any(arrays["status"] != 0):
        raise ValueError("duplex callback status가 0이 아닙니다")
    if (
        type(telemetry["xrun_count"]) is not int
        or telemetry["xrun_count"] != 0
        or type(telemetry["status_present_count"]) is not int
        or telemetry["status_present_count"] != 0
        or type(telemetry["captured_frames"]) is not int
        or telemetry["captured_frames"] != total
        or type(telemetry["submitted_frames"]) is not int
        or telemetry["submitted_frames"] != total
        or telemetry["completed"] is not True
        or telemetry["callback_error"] is not None
        or telemetry["canonical_invalid_reasons"] != []
        or telemetry["stream_stop_error"] is not None
        or telemetry["stream_abort_error"] is not None
        or telemetry["stream_close_error"] is not None
        or telemetry["termination_signal"] is not None
        or telemetry["normal_stop_completed"] is not True
        or telemetry["output_stop_confirmed"] is not True
    ):
        raise ValueError("duplex capture completion/status/error gate가 실패했습니다")

    deltas = {
        name: {
            "minimum_seconds": float(np.min(np.diff(arrays[name]))),
            "maximum_seconds": float(np.max(np.diff(arrays[name]))),
        }
        for name in ("adc", "dac", "current")
    }
    receipt = {
        "schema": "fullband_duplex_telemetry_auxiliary_receipt_v1",
        "source_schema": _DUPLEX_TELEMETRY_SCHEMA,
        "callback_count": count,
        "exact_256_frame_callbacks": True,
        "captured_frames": total,
        "submitted_frames": total,
        "resolved_input_device": telemetry["resolved_input_device"],
        "resolved_output_device": telemetry["resolved_output_device"],
        "device_indices_are_auxiliary_capture_evidence": True,
        "device_identity_binding_authority": False,
        "device_identity_must_be_bound_by_raw_adapter": True,
        "timestamps_finite_strict_monotonic": True,
        "timestamp_delta_diagnostics": deltas,
        "statuses_all_zero": True,
        "portaudio_xrun_status_witness": True,
        "hardware_slip_authority": False,
        "timestamps_used_to_estimate_clock_q": False,
        "slip_samples_field_expected_or_fabricated": False,
        "actual_submitted_pcm_exact_match": True,
        "actual_submitted_pcm_sha256": _array_sha256(actual_submitted),
        "capture_valid_mask_all_true": True,
        "submitted_valid_mask_all_true": True,
        "valid_mask_sha256": {
            name: _array_sha256(value) for name, value in sorted(masks.items())
        },
        "array_sha256": {
            name: _array_sha256(value) for name, value in sorted(arrays.items())
        },
    }
    receipt["sha256"] = _payload_sha256(receipt)
    return receipt


def estimate_clock_cubic_linear_crosscheck(
    *, plan: Mapping[str, Any], submitted_pcm: np.ndarray, captured_adc_pcm: np.ndarray
) -> dict[str, Any]:
    """committed actual pilots만으로 cubic/linear 공통 q를 교차검증한다."""

    receipts = {
        kind: estimate_common_clock_from_waveforms_v5(
            plan=plan, submitted_pcm=submitted_pcm,
            captured_adc_pcm=captured_adc_pcm,
            interpolation_kind=kind,
            validation_policy="pilot_tail_only_pre_operator_holdout",
        )
        for kind in ("cubic", "linear")
    }
    if not all(receipt["passed"] for receipt in receipts.values()):
        raise ValueError("actual pilot common-clock estimator가 PASS하지 못했습니다")
    if not all(
        receipt["captured_adc_full_sha256_computed"] is False
        for receipt in receipts.values()
    ):
        raise ValueError("clock 단계에서 captured full SHA를 선소비했습니다")
    if not all(
        receipt["validation_policy"]
        == "pilot_tail_only_pre_operator_holdout"
        and receipt["operator_holdout_used_for_clock_validation"] is False
        and all(
            rows == ["tail"]
            for rows in receipt["clock_validation_rows_by_path"].values()
        )
        for receipt in receipts.values()
    ):
        raise ValueError("clock estimator가 operator holdout을 선소비했습니다")
    cubic_q = float(receipts["cubic"]["estimated_rate_ratio"])
    linear_q = float(receipts["linear"]["estimated_rate_ratio"])
    endpoint = abs(cubic_q - linear_q) * len(submitted_pcm)
    if endpoint > MAX_CLOCK_INTERPOLATION_ENDPOINT_DISAGREEMENT_SAMPLES:
        raise ValueError("cubic/linear clock q endpoint crosscheck가 실패했습니다")
    receipt = {
        "schema": "fullband_actual_pilot_clock_cubic_linear_v1",
        "method_authority": "fullband_causal_v5.estimate_common_clock_from_waveforms_v5",
        "selected_rate_ratio": cubic_q,
        "selected_interpolation": "cubic",
        "captured_adc_full_sha256_computed": False,
        "bounded_access_bundle_sha256_by_interpolation": {
            kind: receipts[kind]["accessed_waveform_bundle_sha256"]
            for kind in ("cubic", "linear")
        },
        "cubic": receipts["cubic"], "linear": receipts["linear"],
        "endpoint_disagreement_samples": endpoint,
        "maximum_endpoint_disagreement_samples": MAX_CLOCK_INTERPOLATION_ENDPOINT_DISAGREEMENT_SAMPLES,
        "callback_timestamp_q_used": False,
        "separate_marker_used": False,
        "holdout_used_for_clock_fit_or_selection": False,
        "clock_fit_rows": ["lead", "fit_a", "fit_b"],
        "clock_validation_rows": ["pilot_only_tail"],
        "operator_holdout_used_for_clock_validation": False,
        "operator_holdout_first_open_stage": "post_candidate_fit_cross_and_final_fix",
        "passed": True,
    }
    receipt["sha256"] = _payload_sha256(receipt)
    return receipt


def _resample_window(
    captured_adc_pcm: np.ndarray,
    *,
    start: int,
    stop: int,
    q: float,
    access_label: str,
    access_log: list[dict[str, Any]],
) -> np.ndarray:
    query = np.arange(int(start), int(stop), dtype=np.float64) / float(q)
    lower = max(0, int(math.floor(float(query[0]))) - 3)
    upper = min(
        len(captured_adc_pcm),
        int(math.ceil(float(query[-1]))) + 4,
    )
    if query[0] < lower or query[-1] > upper - 1 or upper - lower < 4:
        raise ValueError("clock-corrected central window가 captured support 밖입니다")
    local = np.array(
        captured_adc_pcm[lower:upper],
        dtype=np.float64,
        copy=True,
        order="C",
    )
    if local.ndim != 2 or local.shape[1] != 2 or not np.all(np.isfinite(local)):
        raise ValueError("bounded local captured window가 finite [frame,2]가 아닙니다")
    grid = np.arange(lower, upper, dtype=np.float64)
    result = np.column_stack(
        [
            CubicSpline(grid, local[:, mic], extrapolate=False)(query)
            for mic in range(2)
        ]
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("clock-corrected central window에 non-finite가 있습니다")
    access_log.append(
        {
            "access_label": access_label,
            "logical_dac_start_frame": int(start),
            "logical_dac_stop_frame": int(stop),
            "owned_adc_start_frame": lower,
            "owned_adc_stop_frame": upper,
            "owned_local_capture_sha256": _array_sha256(local),
        }
    )
    return result


def _role_rows(
    *, submitted: np.ndarray, captured_adc_pcm: np.ndarray,
    windows: Mapping[tuple[str, str], tuple[int, int]], role: str, q: float,
    access_log: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for path in PATHS:
        start, stop = windows[(role, path)]
        x_rows.append(submitted[start:stop].astype(np.float64))
        y_rows.append(
            _resample_window(
                captured_adc_pcm,
                start=start,
                stop=stop,
                q=q,
                access_label=f"role:{role}:path:{path}",
                access_log=access_log,
            )
        )
    return np.stack(x_rows), np.stack(y_rows)


def _joint_circular_operator(
    x_rows: np.ndarray, *, support: int, zeros_by_path: Sequence[int]
) -> tuple[LinearOperator, dict[str, Any]]:
    x = np.asarray(x_rows, dtype=np.float64)
    if x.shape != (2, PERIOD, 2):
        raise ValueError("joint operator input은 exact [2,PERIOD,2]여야 합니다")
    if not np.all(np.isfinite(x)):
        raise ValueError("joint operator input에 non-finite가 있습니다")
    taps = int(support)
    if taps <= 0 or taps > PERIOD or len(zeros_by_path) != 2:
        raise ValueError("joint operator support/zeros가 유효하지 않습니다")
    frequency_bin = np.arange(PERIOD // 2 + 1, dtype=np.float64)
    spectrum = np.fft.rfft(x, axis=1)
    shifted = np.empty_like(spectrum)
    for path, zeros in enumerate(zeros_by_path):
        phase = np.exp(-2j * np.pi * frequency_bin * int(zeros) / PERIOD)
        shifted[:, :, path] = spectrum[:, :, path] * phase[None, :]

    def matvec(value: np.ndarray) -> np.ndarray:
        fir = np.asarray(value, dtype=np.float64).reshape(2, taps)
        transfer = np.fft.rfft(fir, n=PERIOD, axis=1)
        predicted = np.fft.irfft(
            np.sum(shifted * transfer.T[None, :, :], axis=2), n=PERIOD, axis=1
        )
        return predicted.reshape(-1)

    def rmatvec(value: np.ndarray) -> np.ndarray:
        residual = np.asarray(value, dtype=np.float64).reshape(2, PERIOD)
        residual_spectrum = np.fft.rfft(residual, axis=1)
        return np.concatenate(
            [
                np.fft.irfft(
                    np.sum(np.conj(shifted[:, :, path]) * residual_spectrum, axis=0),
                    n=PERIOD,
                )[:taps]
                for path in range(2)
            ]
        )

    operator = LinearOperator(
        (2 * PERIOD, 2 * taps), matvec=matvec, rmatvec=rmatvec, dtype=np.float64
    )
    probe = np.sin(np.arange(2 * taps, dtype=np.float64) * 0.017)
    residual = np.cos(np.arange(2 * PERIOD, dtype=np.float64) * 0.013)
    left = float(np.dot(operator.matvec(probe), residual))
    right = float(np.dot(probe, operator.rmatvec(residual)))
    adjoint_error = abs(left - right) / max(abs(left), abs(right), 1.0)
    if adjoint_error > 1.0e-9:
        raise AssertionError("joint circular operator adjoint가 다릅니다")
    return operator, {
        "support_samples": taps,
        "zeros_before_fir_samples": [int(value) for value in zeros_by_path],
        "P_and_S_shifts_applied_before_joint_operator": True,
        "adjoint_relative_error": adjoint_error,
        "aggregate_2x2_path_gram_reported_as_exact_operator_condition": False,
        "exact_operator_condition_number": None,
    }


def _fit_candidate(
    *, x_rows: np.ndarray, y_rows: np.ndarray, role: str, support: int,
    zeros: Sequence[int],
) -> dict[str, Any]:
    observed = np.asarray(y_rows, dtype=np.float64)
    if observed.shape != (2, PERIOD, 2) or not np.all(np.isfinite(observed)):
        raise ValueError("joint solver target은 finite exact [2,PERIOD,2]여야 합니다")
    operator, contract = _joint_circular_operator(
        x_rows, support=support, zeros_by_path=zeros
    )
    fir = np.zeros((2, 2, int(support)), dtype=np.float64)
    fits = []
    for mic in range(2):
        target = observed[:, :, mic].reshape(-1)
        solved = lsmr(
            operator,
            target,
            atol=1.0e-10,
            btol=1.0e-10,
            maxiter=LSMR_MAX_ITERATIONS,
        )
        solution = np.asarray(solved[0], dtype=np.float64)
        raw_diagnostics = np.asarray(solved[1:8], dtype=np.float64)
        if (
            solution.shape != (2 * int(support),)
            or not np.all(np.isfinite(solution))
            or raw_diagnostics.shape != (7,)
            or not np.all(np.isfinite(raw_diagnostics))
        ):
            raise ValueError("LSMR x/diagnostics에 non-finite 또는 shape 오류가 있습니다")
        diagnostics = {
            "istop": int(raw_diagnostics[0]),
            "itn": int(raw_diagnostics[1]),
            "normr": float(raw_diagnostics[2]),
            "normar": float(raw_diagnostics[3]),
            "norma": float(raw_diagnostics[4]),
            "conda": float(raw_diagnostics[5]),
            "normx": float(raw_diagnostics[6]),
        }
        finite_diagnostics = True
        if diagnostics["itn"] >= LSMR_MAX_ITERATIONS:
            raise ValueError("LSMR가 predeclared maxiter를 소진했습니다")
        if diagnostics["istop"] not in LSMR_ACCEPTED_STOP_CODES:
            raise ValueError("LSMR istop이 predeclared accepted set {1,2} 밖입니다")

        estimate = operator.matvec(solution)
        residual = target - estimate
        if not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(residual)):
            raise ValueError("LSMR estimate/residual에 non-finite가 있습니다")
        residual_norm = float(np.linalg.norm(residual))
        target_norm = max(float(np.linalg.norm(target)), np.finfo(np.float64).tiny)
        relative_residual = residual_norm / target_norm
        normal_residual = operator.rmatvec(residual)
        normal_rhs = operator.rmatvec(target)
        if not np.all(np.isfinite(normal_residual)) or not np.all(
            np.isfinite(normal_rhs)
        ):
            raise ValueError("solver normal-equation residual에 non-finite가 있습니다")
        normal_equation_relative = float(
            np.linalg.norm(normal_residual)
            / max(
                np.linalg.norm(normal_rhs),
                np.finfo(np.float64).tiny,
            )
        )
        if not np.isfinite(relative_residual) or not np.isfinite(
            normal_equation_relative
        ):
            raise ValueError("solver 독립 residual gate 값이 non-finite입니다")
        if relative_residual > MAX_SOLVER_RELATIVE_RESIDUAL:
            raise ValueError("solver 전체 relative residual gate 실패")
        if (
            normal_equation_relative
            > MAX_SOLVER_NORMAL_EQUATION_RELATIVE_RESIDUAL
        ):
            raise ValueError("solver normal-equation relative residual gate 실패")

        fir[mic] = solution.reshape(2, int(support))
        fits.append(
            {
                "microphone": MICROPHONES[mic],
                "solution_x_shape": list(solution.shape),
                "solution_x_sha256": _array_sha256(solution),
                "solution_x_all_finite": True,
                "coefficient_fir_shape": [2, int(support)],
                "coefficient_fir_all_finite": True,
                "lsmr_diagnostics": diagnostics,
                "lsmr_diagnostics_all_finite": finite_diagnostics,
                "accepted_istop_codes": list(LSMR_ACCEPTED_STOP_CODES),
                "max_iterations": LSMR_MAX_ITERATIONS,
                "max_iterations_exhausted": False,
                "relative_residual": relative_residual,
                "maximum_relative_residual": MAX_SOLVER_RELATIVE_RESIDUAL,
                "normal_equation_relative_residual": normal_equation_relative,
                "normal_equation_relative_residual_definition": (
                    "norm(A_transpose_residual)/norm(A_transpose_target)"
                ),
                "maximum_normal_equation_relative_residual": (
                    MAX_SOLVER_NORMAL_EQUATION_RELATIVE_RESIDUAL
                ),
                "independent_residual_gates_passed": True,
            }
        )
    receipt = {
        "role": role, "support_samples": int(support),
        "fir_by_mic_path": fir, "fit_receipts": fits, "operator_contract": contract,
    }
    return receipt


def _continuous_peak(fir: np.ndarray) -> tuple[int, float]:
    values = np.asarray(fir, dtype=np.float64)
    peak = int(np.argmax(np.abs(values)))
    lower = max(0, peak - 4)
    upper = min(len(values) - 1, peak + 4)
    if upper - lower < 3:
        return peak, float(peak)
    grid = np.arange(lower, upper + 1, dtype=np.float64)
    spline = CubicSpline(grid, values[lower : upper + 1])
    fit = minimize_scalar(
        lambda point: -float(spline(point) ** 2),
        bounds=(max(lower, peak - 1.0), min(upper, peak + 1.0)),
        method="bounded", options={"xatol": 1.0e-10},
    )
    return peak, float(fit.x)


def _characterize_full_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    fir = np.asarray(candidate["fir_by_mic_path"], dtype=np.float64)
    if fir.shape != (2, 2, FULL_CAUSAL_SUPPORT_SAMPLES):
        raise ValueError("full causal candidate FIR shape이 다릅니다")
    paths: dict[str, dict[str, Any]] = {}
    for mic_index, microphone in enumerate(MICROPHONES):
        for path_index, path in enumerate(PATHS):
            values = fir[mic_index, path_index]
            peak_index, continuous = _continuous_peak(values)
            integer = int(math.floor(continuous + 0.5))
            fractional = continuous - integer
            if not COMPACT_PRE_ROLL_SAMPLES <= integer <= MAX_COARSE_DELAY_SAMPLES:
                raise ValueError("full causal bulk peak가 0..4800/pre-roll 범위 밖입니다")
            zeros = integer - COMPACT_PRE_ROLL_SAMPLES
            total_energy = max(float(np.dot(values, values)), np.finfo(np.float64).tiny)
            early = float(np.dot(values[:zeros], values[:zeros]) / total_energy)
            tail_start = zeros + COMPACT_SUPPORT_SAMPLES
            tail = float(np.dot(values[tail_start:], values[tail_start:]) / total_energy)
            paths[f"{path}_{microphone}"] = {
                "discrete_peak_index_samples": peak_index,
                "continuous_peak_samples": continuous,
                "bulk_integer_samples": integer,
                "fractional_residual_samples": fractional,
                "zeros_before_compact_fir_samples": zeros,
                "early_energy_ratio": early,
                "tail_energy_ratio": tail,
                "early_energy_diagnostic_threshold": MAX_EARLY_ENERGY_RATIO,
                "tail_energy_diagnostic_threshold": MAX_TAIL_ENERGY_RATIO,
                "early_energy_diagnostic_exceeded": bool(
                    early > MAX_EARLY_ENERGY_RATIO
                ),
                "tail_energy_diagnostic_exceeded": bool(
                    tail > MAX_TAIL_ENERGY_RATIO
                ),
                "energy_ratios_are_noise_sensitive_diagnostics_not_admission": True,
                "representation_admission_authority": False,
            }
    receipt = {
        "role": candidate["role"],
        "paths": paths,
        "early_tail_energy_policy": (
            "noise_sensitive_diagnostic_only_compact_roundtrip_is_representation_gate"
        ),
        "thresholds_predeclared_before_holdout": True,
    }
    return receipt


def derive_stationary_err_timing(
    characterized: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], PlantDelays]:
    """fit_a/b peak stationarity만으로 ERR timing을 고정한다."""

    if set(characterized) != set(FIT_ROLES):
        raise ValueError("timing에는 fit_a/fit_b candidate만 필요합니다")
    stationarity: dict[str, Any] = {}
    for path in PATHS:
        for microphone in MICROPHONES:
            key = f"{path}_{microphone}"
            values = [
                float(characterized[role]["paths"][key]["continuous_peak_samples"])
                for role in FIT_ROLES
            ]
            disagreement = abs(values[0] - values[1])
            if disagreement > MAX_FIT_ROLE_PEAK_DISAGREEMENT_SAMPLES:
                raise ValueError("fit_a/fit_b bulk/fraction peak stationarity gate 실패")
            stationarity[key] = {
                "fit_role_peak_samples": values,
                "disagreement_samples": disagreement, "passed": True,
            }
    path_timing: dict[str, Any] = {}
    for path in PATHS:
        key = f"{path}_ERR"
        peak = float(np.mean(stationarity[key]["fit_role_peak_samples"]))
        integer = int(math.floor(peak + 0.5))
        zeros = integer - COMPACT_PRE_ROLL_SAMPLES
        path_timing[path] = {
            "bulk_peak_samples": peak, "bulk_integer_samples": integer,
            "fractional_residual_samples": peak - integer,
            "zeros_before_compact_fir_samples": zeros,
            "pre_roll_samples": COMPACT_PRE_ROLL_SAMPLES,
            "compact_support_samples": COMPACT_SUPPORT_SAMPLES,
            "fractional_encoding": "inside_compact_refit_shape_exactly_once",
            "separate_fractional_phase_applications": 0,
        }
    plants = PlantDelays(
        primary_delay_samples=int(path_timing["primary"]["zeros_before_compact_fir_samples"]),
        secondary_delay_samples=int(path_timing["secondary"]["zeros_before_compact_fir_samples"]),
        handoff_samples=BLOCK, sample_rate=FS,
    )
    lead = plants.lead()
    receipt = {
        "schema": "fullband_fit_roles_stationary_err_timing_v1",
        "timing_authority_microphone": "ERR",
        "REF_role": "stationarity_only_not_PlantDelays",
        "fit_roles": list(FIT_ROLES),
        "holdout_used_for_threshold_support_peak_or_candidate_tuning": False,
        "stationarity": stationarity, "paths": path_timing,
        "plant_delays": plants.model_dump(mode="json"),
        "lead": lead.model_dump(mode="json", exclude={"token"}),
    }
    receipt["sha256"] = _payload_sha256(receipt)
    return receipt, plants


def _predict(x_rows: np.ndarray, fir: np.ndarray, *, zeros: Sequence[int]) -> np.ndarray:
    coefficients = np.asarray(fir, dtype=np.float64)
    operator, _ = _joint_circular_operator(
        x_rows, support=coefficients.shape[2], zeros_by_path=zeros
    )
    predicted = np.empty((2, PERIOD, 2), dtype=np.float64)
    for mic in range(2):
        predicted[:, :, mic] = operator.matvec(coefficients[mic].reshape(-1)).reshape(2, PERIOD)
    return predicted


def _compact_roundtrip(
    *, full_candidate: Mapping[str, Any], compact_candidate: Mapping[str, Any],
    x_rows: np.ndarray, zeros: Sequence[int], timing: Mapping[str, Any],
    y_rows: np.ndarray | None = None,
    bands: Sequence[Sequence[float]] | None = None,
    exact_zero_noise_bins: np.ndarray | None = None,
    noise_spectra: tuple[np.ndarray, np.ndarray] | None = None,
    evaluation_role: str = "unspecified",
) -> dict[str, Any]:
    full_prediction = _predict(
        x_rows, np.asarray(full_candidate["fir_by_mic_path"]), zeros=(0, 0)
    )
    compact_prediction = _predict(
        x_rows, np.asarray(compact_candidate["fir_by_mic_path"]), zeros=zeros
    )
    if y_rows is None or bands is None or exact_zero_noise_bins is None or noise_spectra is None:
        raise ValueError("representation gate에 cross-role observed/noise evidence가 필수입니다")
    frequency = np.fft.rfftfreq(PERIOD, 1.0 / FS)
    representation_rows: list[dict[str, Any]] = []
    for path_index, path in enumerate(PATHS):
        for mic_index, microphone in enumerate(MICROPHONES):
            observed = np.fft.rfft(y_rows[path_index, :, mic_index])
            full = np.fft.rfft(full_prediction[path_index, :, mic_index])
            compact_fft = np.fft.rfft(compact_prediction[path_index, :, mic_index])
            for band_index, (lower, upper) in enumerate(bands):
                band = (frequency >= lower) & (frequency <= upper)
                noise_band = band & exact_zero_noise_bins
                noise_power = max(float(np.mean([
                    np.mean(np.abs(noise[noise_band, mic_index]) ** 2)
                    for noise in noise_spectra
                ])), np.finfo(np.float64).tiny)
                observed_power = float(np.mean(np.abs(observed[band]) ** 2))
                full_residual = float(np.mean(np.abs(observed[band] - full[band]) ** 2))
                compact_residual = float(np.mean(np.abs(observed[band] - compact_fft[band]) ** 2))
                uncertainty = max(noise_power, min(full_residual, compact_residual))
                difference = float(np.mean(np.abs(full[band] - compact_fft[band]) ** 2))
                relative = math.sqrt(max(difference - uncertainty, 0.0) / max(
                    observed_power - uncertainty, np.finfo(np.float64).tiny
                ))
                representation_rows.append({
                    "evaluation_role": evaluation_role, "path": path,
                    "microphone": microphone, "band_index": band_index,
                    "band_hz": [float(lower), float(upper)],
                    "pilot_exact_null_noise_power": noise_power,
                    "full_observed_residual_power": full_residual,
                    "compact_observed_residual_power": compact_residual,
                    "cross_role_uncertainty_power": uncertainty,
                    "full_compact_difference_power": difference,
                    "noise_conditioned_difference_excess_relative": relative,
                    "passed": relative <= MAX_COMPACT_ROUNDTRIP_RELATIVE,
                })
    fractions = []
    compact = np.asarray(compact_candidate["fir_by_mic_path"])
    for path_index, path in enumerate(PATHS):
        _, compact_peak = _continuous_peak(compact[0, path_index])
        encoded_fraction = compact_peak - COMPACT_PRE_ROLL_SAMPLES
        expected_fraction = float(timing["paths"][path]["fractional_residual_samples"])
        error = abs(encoded_fraction - expected_fraction)
        fractions.append(
            {
                "path": path, "compact_peak_samples": compact_peak,
                "encoded_fractional_residual_samples": encoded_fraction,
                "expected_fractional_residual_samples": expected_fraction,
                "absolute_error_samples": error,
            }
        )
    if len(representation_rows) != 32 or not all(row["passed"] for row in representation_rows):
        raise ValueError("noise-conditioned cross-role full↔compact representation gate 실패")
    if max(row["absolute_error_samples"] for row in fractions) > MAX_FRACTIONAL_ROUNDTRIP_ERROR_SAMPLES:
        raise ValueError("fractional shape compact roundtrip gate 실패")
    receipt = {
        "schema": "noise_conditioned_cross_role_representation_gate_v1",
        "evaluation_role": evaluation_role,
        "rows": representation_rows,
        "expected_rows": 32,
        "uncertainty_definition": "max(pilot exact-null noise power,min(full observed residual power,compact observed residual power))",
        "metric_definition": "sqrt(max(full_compact_difference_power-uncertainty_power,0)/max(observed_power-uncertainty_power,tiny))",
        "maximum_allowed": MAX_COMPACT_ROUNDTRIP_RELATIVE,
        "fractional_shape_receipts": fractions,
        "fractional_residual_encoded_in_compact_fir_once": True,
        "separate_fractional_phase_applications": 0,
        "representation_admission_authority": True,
        "passed": True,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt


def _noise_windows(
    *, plan: Mapping[str, Any], submitted: np.ndarray,
    captured_adc_pcm: np.ndarray, q: float,
    access_log: list[dict[str, Any]],
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    lead_start = PERIOD
    tail = next(row for row in plan["layout"] if row["kind"] == "pilot_only_tail")
    tail_start = int(tail["start_frame"]) + PERIOD
    pilot = submitted[lead_start : lead_start + PERIOD].astype(np.float64)
    exact_zero = np.all(np.abs(np.fft.rfft(pilot, axis=0)) <= 1.0e-8, axis=1)
    captures = tuple(
        np.fft.rfft(
            _resample_window(
                captured_adc_pcm,
                start=start,
                stop=start + PERIOD,
                q=q,
                access_label=label,
                access_log=access_log,
            ),
            axis=0,
        )
        for start, label in (
            (lead_start, "noise:pilot_only_lead"),
            (tail_start, "noise:pilot_only_tail"),
        )
    )
    return exact_zero, captures  # type: ignore[return-value]


def _score_candidate_role(
    *, candidate_role: str, evaluation_role: str, compact_fir: np.ndarray,
    x_rows: np.ndarray, y_rows: np.ndarray, zeros: Sequence[int],
    bands: Sequence[Sequence[float]], exact_zero_noise_bins: np.ndarray,
    noise_spectra: tuple[np.ndarray, np.ndarray],
) -> list[dict[str, Any]]:
    prediction = _predict(x_rows, compact_fir, zeros=zeros)
    frequency = np.fft.rfftfreq(PERIOD, 1.0 / FS)
    rows = []
    if evaluation_role == "holdout":
        relation = "terminal_holdout"
    elif candidate_role == "fixed_average":
        relation = "preterminal_fit_role"
    elif evaluation_role == candidate_role:
        relation = "fit"
    else:
        relation = "cross"
    for path_index, path in enumerate(PATHS):
        for mic_index, microphone in enumerate(MICROPHONES):
            target_fft = np.fft.rfft(y_rows[path_index, :, mic_index])
            estimate_fft = np.fft.rfft(prediction[path_index, :, mic_index])
            for band_index, (lower, upper) in enumerate(bands):
                band = (frequency >= float(lower)) & (frequency <= float(upper))
                noise_band = band & exact_zero_noise_bins
                if int(np.sum(band)) < 8 or int(np.sum(noise_band)) < 8:
                    raise ValueError("8-band target/noise density에 필요한 bin이 부족합니다")
                target = target_fft[band]
                estimate = estimate_fft[band]
                target_power = float(np.mean(np.abs(target) ** 2))
                estimate_power = float(np.mean(np.abs(estimate) ** 2))
                residual_power = float(np.mean(np.abs(target - estimate) ** 2))
                noise_power = max(
                    float(np.mean([
                        np.mean(np.abs(noise[noise_band, mic_index]) ** 2)
                        for noise in noise_spectra
                    ])),
                    np.finfo(np.float64).tiny,
                )
                signal_power = max(target_power - noise_power, np.finfo(np.float64).tiny)
                residual_excess = max(residual_power - noise_power, 0.0)
                relative = math.sqrt(residual_excess / signal_power)
                agreement = float(
                    abs(np.vdot(target, estimate))
                    / max(len(target) * math.sqrt(target_power * estimate_power), np.finfo(np.float64).tiny)
                )
                snr_db = 10.0 * math.log10(max(target_power, np.finfo(np.float64).tiny) / noise_power)
                density = float(np.mean(np.abs(target) ** 2 >= noise_power * 100.0))
                passed = bool(
                    relative <= MAX_BAND_RELATIVE_RESIDUAL
                    and agreement >= MIN_BAND_COMPLEX_VECTOR_AGREEMENT
                    and snr_db >= MIN_RESPONSE_TO_NOISE_DB
                    and density >= MIN_TARGET_BIN_DENSITY
                )
                rows.append(
                    {
                        "candidate_role": candidate_role, "evaluation_role": evaluation_role,
                        "relation": relation, "path": path, "microphone": microphone,
                        "band_index": band_index, "band_hz": [float(lower), float(upper)],
                        "target_bins": int(np.sum(band)), "noise_bins": int(np.sum(noise_band)),
                        "target_bin_density_above_noise_20db": density,
                        "response_to_noise_db": snr_db,
                        "noise_conditioned_relative_residual": relative,
                        "complex_vector_agreement_not_coherence": agreement,
                        "independent_coherence_claimed": False, "passed": passed,
                    }
                )
    return rows


def _validate_payload_receipt(value: Mapping[str, Any], sha_key: str) -> str:
    payload = {key: item for key, item in value.items() if key != sha_key}
    computed = _payload_sha256(payload)
    if value.get(sha_key) != computed:
        raise ValueError(f"component {sha_key} stale payload SHA")
    return computed


def _validate_operator_component_bindings(
    analysis: Mapping[str, Any], operator: Mapping[str, Any]
) -> None:
    """operator receipt와 실제 component bundle의 splice를 fail-closed한다."""
    final = analysis["final_fixed_average"]
    receipt = final["operator_receipt"]
    telemetry_sha = _validate_payload_receipt(analysis["duplex_telemetry_auxiliary"], "sha256")
    clock_sha = _validate_payload_receipt(analysis["clock"], "sha256")
    timing_sha = _validate_payload_receipt(analysis["timing"], "sha256")
    formula_sha = _validate_payload_receipt(final["formula"], "canonical_payload_sha256")
    score_sha = _validate_payload_receipt(final["score"], "canonical_payload_sha256")
    shifted_sha = _validate_payload_receipt(
        analysis["compact_refit"]["shifted_support_1024_exact_condition_receipt"],
        "canonical_payload_sha256",
    )
    representation_sha = _validate_payload_receipt(
        final["representation_threshold_contract"], "canonical_payload_sha256"
    )
    score_threshold_sha = _validate_payload_receipt(
        final["score_threshold_contract"], "canonical_payload_sha256"
    )
    for receipt_group in (
        analysis["compact_refit"]["roundtrip"], final["roundtrip_on_fit_roles"]
    ):
        for item in receipt_group.values():
            _validate_payload_receipt(item, "canonical_payload_sha256")
    roundtrip_bundle_sha = _validate_payload_receipt(
        final["roundtrip_bundle"], "canonical_payload_sha256"
    )
    expected = {
        "duplex_telemetry_receipt_sha256": telemetry_sha,
        "clock_receipt_sha256": clock_sha,
        "timing_receipt_sha256": timing_sha,
        "fixed_average_formula_payload_sha256": formula_sha,
        "final_score_payload_sha256": score_sha,
        "shifted_condition_payload_sha256": shifted_sha,
        "final_representation_roundtrip_bundle_sha256": roundtrip_bundle_sha,
        "representation_threshold_contract_sha256": representation_sha,
        "score_threshold_contract_sha256": score_threshold_sha,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("operator component bundle SHA splice가 감지됐습니다")
    payload = {key: value for key, value in receipt.items() if key != "canonical_payload_sha256"}
    if receipt.get("canonical_payload_sha256") != _payload_sha256(payload):
        raise ValueError("operator receipt canonical payload SHA가 유효하지 않습니다")
    if receipt.get("score_thresholds") != final["score_threshold_contract"]:
        raise ValueError("embedded score thresholds가 contract와 exact 일치하지 않습니다")
    array_keys = set(receipt["operator_array_sha256"])
    if set(operator) != array_keys | {"receipt"} or operator.get("receipt") != receipt:
        raise ValueError("returned operator bundle key/receipt splice")
    actual_array_sha = {
        key: _array_sha256(np.asarray(operator[key])) for key in sorted(array_keys)
    }
    if actual_array_sha != receipt["operator_array_sha256"]:
        raise ValueError("returned operator array SHA splice")


def analyze_committed_v5_live_delay(
    *, plan: Mapping[str, Any], submitted_pcm: np.ndarray,
    captured_adc_pcm: np.ndarray, duplex_telemetry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Committed plan/raw만 받는 fail-closed offline 분석 entrypoint."""

    submitted = np.asarray(submitted_pcm)
    captured = np.asarray(captured_adc_pcm)
    if captured.dtype != np.dtype("<i4") or captured.shape != submitted.shape:
        raise ValueError("actual captured ADC는 submitted와 같은 exact <i4 [frame,2]여야 합니다")
    plan_receipt, windows = validate_committed_plan_and_derive_windows(plan, submitted)
    telemetry_receipt = validate_duplex_telemetry_auxiliary(
        duplex_telemetry,
        captured_adc_pcm=captured,
        expected_submitted_pcm=submitted,
    )
    clock = estimate_clock_cubic_linear_crosscheck(
        plan=plan, submitted_pcm=submitted, captured_adc_pcm=captured
    )
    execution_order = [
        "clock_fit_lead_fit_a_fit_b",
        "clock_validation_pilot_only_tail",
    ]
    q = float(clock["selected_rate_ratio"])
    preterminal_access_log: list[dict[str, Any]] = []
    fit_role_rows = {
        role: _role_rows(
            submitted=submitted, captured_adc_pcm=captured, windows=windows,
            role=role, q=q, access_log=preterminal_access_log,
        )
        for role in FIT_ROLES
    }

    full_candidates = {
        role: _fit_candidate(
            x_rows=fit_role_rows[role][0],
            y_rows=fit_role_rows[role][1],
            role=role,
            support=FULL_CAUSAL_SUPPORT_SAMPLES, zeros=(0, 0),
        )
        for role in FIT_ROLES
    }
    characterized = {
        role: _characterize_full_candidate(full_candidates[role]) for role in FIT_ROLES
    }
    timing, plants = derive_stationary_err_timing(characterized)
    zeros = (plants.primary_delay_samples, plants.secondary_delay_samples)
    compact_candidates = {
        role: _fit_candidate(
            x_rows=fit_role_rows[role][0],
            y_rows=fit_role_rows[role][1],
            role=role,
            support=COMPACT_SUPPORT_SAMPLES, zeros=zeros,
        )
        for role in FIT_ROLES
    }
    shifted_condition = exact_shifted_condition_audit_v5(
        plan,
        submitted,
        zeros_by_path=zeros,
        support=COMPACT_SUPPORT_SAMPLES,
    )
    if shifted_condition["passed"] is not True:
        raise ValueError("actual shifted compact operator exact condition gate 실패")
    execution_order.append("full_and_compact_candidate_fit")

    exact_zero, noise_spectra = _noise_windows(
        plan=plan, submitted=submitted, captured_adc_pcm=captured, q=q,
        access_log=preterminal_access_log,
    )
    holdout_adc_supports = []
    for path in PATHS:
        start, stop = windows[("holdout", path)]
        query_first = start / q
        query_last = (stop - 1) / q
        holdout_adc_supports.append({
            "path": path,
            "owned_adc_start_frame": max(0, int(math.floor(query_first)) - 3),
            "owned_adc_stop_frame": min(len(captured), int(math.ceil(query_last)) + 4),
        })
    clock_accesses = []
    for interpolation in ("cubic", "linear"):
        for path_rows in clock[interpolation]["accessed_waveform_receipts_by_path"].values():
            for rows in path_rows.values():
                clock_accesses.extend(rows)
    all_preterminal_accesses = clock_accesses + preterminal_access_log
    collisions = [
        (access, holdout)
        for access in all_preterminal_accesses
        for holdout in holdout_adc_supports
        if max(access["owned_adc_start_frame"], holdout["owned_adc_start_frame"])
        < min(access["owned_adc_stop_frame"], holdout["owned_adc_stop_frame"])
    ]
    if collisions:
        raise ValueError("preterminal bounded ADC access가 q-converted holdout support와 겹칩됩니다")
    holdout_access_separation = {
        "schema": "q_converted_holdout_adc_support_separation_v1",
        "selected_rate_ratio": q,
        "holdout_adc_supports": holdout_adc_supports,
        "preterminal_access_count": len(all_preterminal_accesses),
        "pairwise_intersections": 0,
        "all_preterminal_intervals_disjoint_from_holdout": True,
    }
    holdout_access_separation["canonical_payload_sha256"] = _payload_sha256(
        holdout_access_separation
    )
    bands = plan["control_band_contract"]["physical_identification_subbands_hz"]
    roundtrips = {
        role: _compact_roundtrip(
            full_candidate=full_candidates[role], compact_candidate=compact_candidates[role],
            x_rows=fit_role_rows["fit_b" if role == "fit_a" else "fit_a"][0],
            y_rows=fit_role_rows["fit_b" if role == "fit_a" else "fit_a"][1],
            zeros=zeros, timing=timing, bands=bands,
            exact_zero_noise_bins=exact_zero, noise_spectra=noise_spectra,
            evaluation_role="fit_b" if role == "fit_a" else "fit_a",
        )
        for role in FIT_ROLES
    }
    candidate_scores: dict[str, Any] = {}
    for candidate_role in FIT_ROLES:
        nonterminal_rows = []
        for evaluation_role in FIT_ROLES:
            nonterminal_rows.extend(
                _score_candidate_role(
                    candidate_role=candidate_role, evaluation_role=evaluation_role,
                    compact_fir=compact_candidates[candidate_role]["fir_by_mic_path"],
                    x_rows=fit_role_rows[evaluation_role][0],
                    y_rows=fit_role_rows[evaluation_role][1],
                    zeros=zeros, bands=bands, exact_zero_noise_bins=exact_zero,
                    noise_spectra=noise_spectra,
                )
            )
        if not all(row["passed"] for row in nonterminal_rows):
            raise ValueError("candidate fit/cross 2path×2mic×8band gate 실패")
        candidate_scores[candidate_role] = {
            "schema": "fullband_candidate_fit_cross_preterminal_v1",
            "candidate_role": candidate_role,
            "scored_compact_fir_array_sha256": _array_sha256(
                compact_candidates[candidate_role]["fir_by_mic_path"]
            ),
            "scored_zeros_before_fir_samples": list(zeros),
            "rows": nonterminal_rows,
            "expected_rows": 64,
            "evaluated_before_final_fix_and_terminal_holdout": True,
            "holdout_used_for_threshold_support_or_candidate_tuning": False,
            "independent_coherence_available": False,
            "all_rows_passed": len(nonterminal_rows) == 64
            and all(row["passed"] for row in nonterminal_rows),
        }
    execution_order.append("candidate_fit_cross_scoring")

    final_full = (
        FINAL_FIXED_AVERAGE_WEIGHTS[0]
        * full_candidates["fit_a"]["fir_by_mic_path"]
        + FINAL_FIXED_AVERAGE_WEIGHTS[1]
        * full_candidates["fit_b"]["fir_by_mic_path"]
    )
    final_compact = (
        FINAL_FIXED_AVERAGE_WEIGHTS[0]
        * compact_candidates["fit_a"]["fir_by_mic_path"]
        + FINAL_FIXED_AVERAGE_WEIGHTS[1]
        * compact_candidates["fit_b"]["fir_by_mic_path"]
    )
    if not np.all(np.isfinite(final_full)) or not np.all(np.isfinite(final_compact)):
        raise ValueError("fixed average full/compact FIR에 non-finite가 있습니다")
    final_formula = {
        "schema": "fullband_predeclared_fixed_average_formula_v1",
        "fixed_before_operator_holdout_open": True,
        "candidate_order": list(FIT_ROLES),
        "candidate_weights": list(FINAL_FIXED_AVERAGE_WEIGHTS),
        "selection_or_holdout_dependent_weighting": False,
        "full_candidate_array_sha256": {
            role: _array_sha256(full_candidates[role]["fir_by_mic_path"])
            for role in FIT_ROLES
        },
        "compact_candidate_array_sha256": {
            role: _array_sha256(compact_candidates[role]["fir_by_mic_path"])
            for role in FIT_ROLES
        },
        "fixed_average_full_array_sha256": _array_sha256(final_full),
        "fixed_average_compact_array_sha256": _array_sha256(final_compact),
    }
    final_formula["canonical_payload_sha256"] = _payload_sha256(final_formula)

    final_full_candidate = {
        "role": "fixed_average",
        "fir_by_mic_path": final_full,
    }
    final_compact_candidate = {
        "role": "fixed_average",
        "fir_by_mic_path": final_compact,
    }
    final_roundtrips = {
        role: _compact_roundtrip(
            full_candidate=final_full_candidate,
            compact_candidate=final_compact_candidate,
            x_rows=fit_role_rows[role][0],
            y_rows=fit_role_rows[role][1], zeros=zeros, timing=timing,
            bands=bands, exact_zero_noise_bins=exact_zero,
            noise_spectra=noise_spectra, evaluation_role=role,
        )
        for role in FIT_ROLES
    }
    execution_order.append("fixed_average_formula_and_roundtrip")
    final_fit_cross_rows: list[dict[str, Any]] = []
    for evaluation_role in FIT_ROLES:
        final_fit_cross_rows.extend(
            _score_candidate_role(
                candidate_role="fixed_average",
                evaluation_role=evaluation_role,
                compact_fir=final_compact,
                x_rows=fit_role_rows[evaluation_role][0],
                y_rows=fit_role_rows[evaluation_role][1],
                zeros=zeros,
                bands=bands,
                exact_zero_noise_bins=exact_zero,
                noise_spectra=noise_spectra,
            )
        )
    if len(final_fit_cross_rows) != 64 or not all(
        row["passed"] for row in final_fit_cross_rows
    ):
        raise ValueError("fixed average preterminal fit_a/fit_b 8-band gate 실패")
    execution_order.append("fixed_average_fit_a_fit_b_scoring")

    execution_order.append("operator_holdout_first_open")
    terminal_access_log: list[dict[str, Any]] = []
    holdout_rows = _role_rows(
        submitted=submitted,
        captured_adc_pcm=captured,
        windows=windows,
        role="holdout",
        q=q,
        access_log=terminal_access_log,
    )
    final_terminal_rows = _score_candidate_role(
        candidate_role="fixed_average",
        evaluation_role="holdout",
        compact_fir=final_compact,
        x_rows=holdout_rows[0],
        y_rows=holdout_rows[1],
        zeros=zeros,
        bands=bands,
        exact_zero_noise_bins=exact_zero,
        noise_spectra=noise_spectra,
    )
    if len(final_terminal_rows) != 32 or not all(
        row["passed"] for row in final_terminal_rows
    ):
        raise ValueError("fixed average terminal holdout 8-band admission gate 실패")
    execution_order.append("fixed_average_terminal_holdout_scoring")
    final_score_rows = final_fit_cross_rows + final_terminal_rows
    final_score = {
        "schema": "fullband_fixed_average_fit_fit_terminal_score_v1",
        "candidate_role": "fixed_average",
        "fixed_average_formula_payload_sha256": final_formula[
            "canonical_payload_sha256"
        ],
        "scored_compact_fir_array_sha256": _array_sha256(final_compact),
        "scored_zeros_before_fir_samples": list(zeros),
        "evaluation_order": ["fit_a", "fit_b", "terminal_holdout"],
        "rows": final_score_rows,
        "expected_rows": 96,
        "fit_roles_evaluated_before_terminal_holdout": True,
        "holdout_used_for_threshold_support_weight_or_candidate_tuning": False,
        "holdout_role": "terminal_admission_only",
        "independent_coherence_available": False,
        "all_rows_passed": len(final_score_rows) == 96
        and all(row["passed"] for row in final_score_rows),
    }
    final_score["canonical_payload_sha256"] = _payload_sha256(final_score)

    representation_threshold_contract = {
        "schema": "noise_conditioned_cross_role_representation_threshold_v1",
        "maximum_noise_conditioned_difference_excess_relative": MAX_COMPACT_ROUNDTRIP_RELATIVE,
        "uncertainty_definition": "max(pilot exact-null noise power,min(full observed residual power,compact observed residual power))",
        "candidate_evaluation_relation": "opposite_fit_role",
        "final_evaluation_roles": list(FIT_ROLES),
        "operator_holdout_excluded": True,
        "threshold_relaxed_for_noise": False,
    }
    representation_threshold_contract["canonical_payload_sha256"] = _payload_sha256(
        representation_threshold_contract
    )
    score_threshold_contract = {
        "schema": "fullband_live_score_threshold_contract_v1",
        "maximum_noise_conditioned_relative_residual": MAX_BAND_RELATIVE_RESIDUAL,
        "minimum_complex_vector_agreement": MIN_BAND_COMPLEX_VECTOR_AGREEMENT,
        "minimum_response_to_noise_db": MIN_RESPONSE_TO_NOISE_DB,
        "minimum_target_bin_density": MIN_TARGET_BIN_DENSITY,
        "expected_final_rows": 96,
    }
    score_threshold_contract["canonical_payload_sha256"] = _payload_sha256(score_threshold_contract)
    final_roundtrip_bundle = {
        "schema": "fullband_final_representation_roundtrip_bundle_v1",
        "fixed_average_compact_array_sha256": _array_sha256(final_compact),
        "rows_by_fit_role": final_roundtrips,
        "threshold_contract_sha256": representation_threshold_contract["canonical_payload_sha256"],
    }
    final_roundtrip_bundle["canonical_payload_sha256"] = _payload_sha256(final_roundtrip_bundle)
    captured_full_sha256 = _array_sha256(captured)  # terminal holdout scoring 완료 후 최초 계산

    operator_arrays = {
        "primary_compact_fir_by_mic": final_compact[:, 0].astype("<f8"),
        "secondary_compact_fir_by_mic": final_compact[:, 1].astype("<f8"),
        "primary_zeros_before_fir": np.asarray(zeros[0], dtype="<i8"),
        "secondary_zeros_before_fir": np.asarray(zeros[1], dtype="<i8"),
        "support_samples": np.asarray(COMPACT_SUPPORT_SAMPLES, dtype="<i8"),
        "separate_fractional_phase_applications": np.asarray(0, dtype="<i8"),
    }
    operator_receipt = {
        "schema": "fullband_fixed_average_compact_operator_receipt_v1",
        "signal_plan_payload_sha256": plan_receipt["signal_plan_payload_sha256"],
        "actual_submitted_pcm_sha256": _array_sha256(submitted),
        "captured_adc_pcm_sha256": captured_full_sha256,
        "duplex_telemetry_receipt_sha256": telemetry_receipt["sha256"],
        "clock_receipt_sha256": clock["sha256"],
        "timing_receipt_sha256": timing["sha256"],
        "fixed_average_formula_payload_sha256": final_formula[
            "canonical_payload_sha256"
        ],
        "final_score_payload_sha256": final_score["canonical_payload_sha256"],
        "shifted_condition_payload_sha256": shifted_condition[
            "canonical_payload_sha256"
        ],
        "final_representation_roundtrip_bundle_sha256": final_roundtrip_bundle["canonical_payload_sha256"],
        "representation_threshold_contract_sha256": representation_threshold_contract["canonical_payload_sha256"],
        "score_threshold_contract_sha256": score_threshold_contract["canonical_payload_sha256"],
        "score_thresholds": score_threshold_contract,
        "operator_array_sha256": {
            key: _array_sha256(value)
            for key, value in sorted(operator_arrays.items())
        },
        "raw_publisher_bound": False,
        "live_delay_authority_available": None,
        "canonical_training_eligible": False,
        "hardware_sample_slip_authority_available": False,
        "non_authoritative_reason": (
            "immutable_raw_publisher_and_hardware_slip_authority_unbound"
        ),
    }
    operator_receipt["canonical_payload_sha256"] = _payload_sha256(operator_receipt)
    operator: dict[str, Any] = {**operator_arrays, "receipt": operator_receipt}
    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "status": "OFFLINE_MATH_PASS_RAW_PUBLISHER_AUTHORITY_UNBOUND",
        "signal_plan": plan_receipt,
        "duplex_telemetry_auxiliary": telemetry_receipt,
        "clock": clock,
        "full_unshifted_causal_identification": {
            "support_samples": FULL_CAUSAL_SUPPORT_SAMPLES,
            "max_delay_scan_samples": MAX_COARSE_DELAY_SAMPLES,
            "zeros_assumed_during_fit": [0, 0],
            "candidates": {
                role: {key: value for key, value in full_candidates[role].items() if key != "fir_by_mic_path"}
                for role in FIT_ROLES
            },
            "characterized": characterized,
            "aggregate_2x2_gram_called_exact_operator_condition": False,
        },
        "timing": timing,
        "compact_refit": {
            "support_samples": COMPACT_SUPPORT_SAMPLES,
            "different_P_S_zeros": list(zeros),
            "fractional_shape_inside_FIR_once": True,
            "separate_fractional_phase_applications": 0,
            "candidate_receipts": {
                role: {key: value for key, value in compact_candidates[role].items() if key != "fir_by_mic_path"}
                for role in FIT_ROLES
            },
            "roundtrip": roundtrips,
            "early_tail_energy_is_diagnostic_only": True,
            "representation_admission_authority": "full_to_compact_prediction_roundtrip",
            "shifted_support_1024_exact_condition_receipt": shifted_condition,
            "unshifted_condition_receipt_reused_for_shifted_operator": False,
        },
        "candidate_fit_cross_preterminal_scores": candidate_scores,
        "final_fixed_average": {
            "formula": final_formula,
            "roundtrip_on_fit_roles": final_roundtrips,
            "roundtrip_bundle": final_roundtrip_bundle,
            "representation_threshold_contract": representation_threshold_contract,
            "score_threshold_contract": score_threshold_contract,
            "score": final_score,
            "operator_receipt": operator_receipt,
            "returned_operator_is_exact_scored_fixed_average": True,
        },
        "captured_raw_binding": {
            "captured_adc_pcm_sha256": captured_full_sha256,
            "signal_plan_payload_sha256": plan_receipt[
                "signal_plan_payload_sha256"
            ],
            "actual_submitted_pcm_sha256": _array_sha256(submitted),
            "raw_publisher_bound": False,
            "authoritative_live_raw_claimed": False,
        },
        "holdout_policy": {
            "predeclared_terminal_role": "holdout",
            "used_for_threshold_support_peak_or_candidate_tuning": False,
            "used_only_for_terminal_admission_after_fit_cross": True,
            "execution_order": execution_order,
            "operator_holdout_first_open_after_final_fixed": True,
            "preterminal_bounded_capture_access": preterminal_access_log,
            "terminal_bounded_capture_access": terminal_access_log,
            "captured_full_sha_computed_only_after_terminal_score": True,
            "q_converted_adc_support_separation": holdout_access_separation,
        },
        "raw_publisher_bound": False,
        "live_delay_authority_available": None,
        "canonical_training_eligible": False,
        "hardware_slip_authority_available": False,
        "canonical_blocker": "IMMUTABLE_RAW_PUBLISHER_AND_HARDWARE_SLIP_AUTHORITY_UNBOUND",
    }
    _validate_operator_component_bindings(analysis, operator)
    analysis["analysis_sha256"] = _payload_sha256(analysis)
    return analysis, operator


__all__ = [
    "ANALYSIS_SCHEMA", "COMPACT_PRE_ROLL_SAMPLES", "COMPACT_SUPPORT_SAMPLES",
    "EXPECTED_PCM_SHA256", "EXPECTED_PLAN_SHA256",
    "FULL_CAUSAL_SUPPORT_SAMPLES", "MAX_COARSE_DELAY_SAMPLES",
    "analyze_committed_v5_live_delay", "derive_stationary_err_timing",
    "estimate_clock_cubic_linear_crosscheck",
    "validate_committed_plan_and_derive_windows", "validate_duplex_telemetry_auxiliary",
]
