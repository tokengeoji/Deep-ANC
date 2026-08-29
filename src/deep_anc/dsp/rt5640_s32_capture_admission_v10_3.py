"""RT5640 S32 disarmed capture의 무출력 receipt validator.

``audio_duplex_s32_disarmed_v10_3``는 backend application buffer가 zero-only pre-arm을
지키는지를 보장한다. 이 모듈은 그 성공 반환값을 S32 fullband signal plan에 결속한다.
PCM/ALSA/sounddevice를 열거나 raw 파일을 쓰지 않으며, electrical tap·hardware sample
identity·P/S·학습 권한을 절대로 만들지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..audio_duplex_s32_disarmed_v10_3 import (
    BLOCK_SIZE,
    SAMPLE_RATE,
    S32DisarmedDuplexCaptureFailure,
    TELEMETRY_SCHEMA,
)
from .rt5640_fullband_s32_plan_v10 import (
    build_rt5640_fullband_s32_plan_v10,
    validate_rt5640_fullband_s32_plan_v10,
)


SCHEMA = "rt5640_s32_capture_admission_receipt_v10_3"
STATUS = "S32_CAPTURE_TRANSPORT_PASS_ELECTRICAL_WITNESS_UNBOUND"
BLOCKED_STATUS = "S32_CAPTURE_BLOCKED_PARTIAL_OR_INVALID"
POST_START_PRE_ARM_RECEIPT_SCHEMA = "rt5640_s32_post_start_pre_arm_receipt_v10_3"
PCM_DTYPE = np.dtype("<i4")
MASK_DTYPE = np.dtype(np.bool_)
_TIME_DTYPE = np.dtype("<f8")
_INDEX_DTYPE = np.dtype("<i8")
_STATUS_DTYPE = np.dtype("<u4")

_REQUIRED_SCALARS = frozenset(
    {
        "schema",
        "authority",
        "portaudio_application_buffer_only",
        "post_start_pre_arm_check_required",
        "post_start_pre_arm_check_passed",
        "nonzero_assignment_before_arm_allowed",
        "on_stream_started_is_nonzero_admission",
        "hardware_sample_slip_authority",
        "physical_output_authority",
        "electrical_output_authority",
        "acoustic_output_authority",
        "sample_rate_hz",
        "block_size",
        "latency",
        "channels",
        "input_dtype",
        "output_dtype",
        "dither_off",
        "planned_pcm_sha256",
        "planned_frames",
        "expected_planned_callbacks",
        "resolved_input_device",
        "resolved_output_device",
        "pre_open_monotonic_started",
        "pre_open_monotonic_completed",
        "capture_monotonic_started",
        "arm_check_monotonic_started",
        "arm_check_monotonic_completed",
        "arm_monotonic",
        "capture_monotonic_completed",
        "capture_monotonic_elapsed_seconds",
        "watchdog_grace_seconds",
        "prearm_callback_count",
        "prearm_output_zero_observed",
        "xrun_count",
        "status_present_count",
        "captured_frames",
        "submitted_frames",
        "callback_zero_attempt_count",
        "callback_zero_confirmed_count",
        "callback_planned_assignment_attempt_count",
        "callback_planned_assignment_confirmed_count",
        "possible_nonzero_output_after_failed_assignment",
        "planned_callback_sequence_contiguous",
        "planned_callback_start_frames_contiguous",
        "planned_callback_frame_counts_exact",
        "capture_valid_all_true",
        "submitted_valid_all_true",
        "full_frame_accounting_valid",
        "actual_matches_planned_application_buffer",
        "actual_submitted_pcm_hash_eligible",
        "actual_submitted_pcm_sha256",
        "completed",
        "callback_error",
        "canonical_invalid_reasons",
        "stream_constructor_error",
        "stream_start_error",
        "post_start_pre_arm_check_error",
        "watchdog_error",
        "stream_stop_error",
        "stream_abort_error",
        "stream_close_error",
        "on_output_closed_error",
        "termination_signals",
        "stream_start_returned_without_exception",
        "stream_stop_attempted",
        "stream_stop_returned_without_exception",
        "stream_abort_attempted",
        "stream_abort_returned_without_exception",
        "stream_close_attempted",
        "stream_close_returned_without_exception",
        "normal_stop_completed",
        "faults",
    }
)
_PLANNED_ARRAYS = {
    "planned_callback_sequence": _INDEX_DTYPE,
    "planned_callback_start_frames": _INDEX_DTYPE,
    "planned_callback_frame_counts": _INDEX_DTYPE,
    "planned_input_buffer_adc_time": _TIME_DTYPE,
    "planned_output_buffer_dac_time": _TIME_DTYPE,
    "planned_callback_current_time": _TIME_DTYPE,
    "planned_callback_status_bitmask": _STATUS_DTYPE,
}
_PREARM_ARRAYS = {
    "prearm_callback_sequence": _INDEX_DTYPE,
    "prearm_callback_start_frames": _INDEX_DTYPE,
    "prearm_callback_frame_counts": _INDEX_DTYPE,
    "prearm_input_buffer_adc_time": _TIME_DTYPE,
    "prearm_output_buffer_dac_time": _TIME_DTYPE,
    "prearm_callback_current_time": _TIME_DTYPE,
    "prearm_callback_status_bitmask": _STATUS_DTYPE,
}
_PAYLOAD_ARRAYS = {
    "actual_submitted_pcm": PCM_DTYPE,
    "capture_valid_mask": MASK_DTYPE,
    "submitted_valid_mask": MASK_DTYPE,
}
_POST_START_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "passed",
        "observed_after_stream_start",
        "resolved_input_device",
        "resolved_output_device",
        "input_hw_params",
        "output_hw_params",
        "routes",
        "j511",
        "occupancy",
        "pre_snapshot_sha256",
        "post_snapshot_sha256",
    }
)
_HW_PARAMS_KEYS = frozenset(
    {"card", "pcm", "format", "sample_rate_hz", "channels", "period_frames"}
)
_ROUTE_KEYS = frozenset({"input", "output", "observed_after_stream_start"})
_JACK_KEYS = frozenset({"state", "samples", "all_samples_equal"})
_OCCUPANCY_KEYS = frozenset(
    {"input_owned_by_this_capture", "output_owned_by_this_capture", "other_pcm_owners"}
)


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        raise ValueError("array SHA input은 C-contiguous여야 합니다")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}는 mapping이어야 합니다")
    return value


def _array(value: Any, *, label: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype or array.shape != shape or not array.flags.c_contiguous:
        raise ValueError(f"{label}는 exact {dtype.str} {list(shape)} C-contiguous여야 합니다")
    return array


def _bool(value: Any, *, label: str, expected: bool) -> None:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{label}는 exact {expected!r}여야 합니다")


def _integer(value: Any, *, label: str, expected: int | None = None, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{label}는 exact int여야 합니다")
    if expected is not None and value != expected:
        raise ValueError(f"{label}가 expected value와 다릅니다")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label}가 minimum보다 작습니다")
    return value


def _finite(value: Any, *, label: str, minimum: float | None = None) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label}는 finite number여야 합니다")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label}가 허용 범위를 벗어났습니다")
    return result


def _required(telemetry: Mapping[str, Any], *, label: str) -> None:
    missing = sorted(_REQUIRED_SCALARS - set(telemetry))
    arrays = set(_PLANNED_ARRAYS) | set(_PREARM_ARRAYS) | set(_PAYLOAD_ARRAYS)
    missing.extend(sorted(arrays - set(telemetry)))
    if missing:
        raise ValueError(f"{label}에 필수 field가 없습니다: {missing}")


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label}는 64자리 SHA-256이어야 합니다")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label}는 hexadecimal SHA-256이어야 합니다") from error
    return value


def _exact_mapping(value: Any, *, label: str, expected: frozenset[str]) -> dict[str, Any]:
    result = _mapping(value, label=label)
    if set(result) != set(expected):
        raise ValueError(
            f"{label} key가 exact하지 않습니다: "
            f"missing={sorted(expected - set(result))}, extra={sorted(set(result) - expected)}"
        )
    return dict(result)


def _validate_post_start_pre_arm_receipt(
    value: Mapping[str, Any], *, input_device: int, output_device: int
) -> dict[str, Any]:
    """stream open 뒤 adapter가 읽은 hw_params/route/jack/occupancy receipt를 검사한다.

    이 receipt는 amp 전원·실제 output voltage·electrical tap·P/S를 증명하지 않는다.
    오직 disarmed primitive가 arm하기 전에 negotiated S32 route를 점검했다는 input이다.
    """

    receipt = _exact_mapping(
        value,
        label="post_start_pre_arm_receipt",
        expected=_POST_START_RECEIPT_KEYS,
    )
    if receipt["schema"] != POST_START_PRE_ARM_RECEIPT_SCHEMA:
        raise ValueError("post-start receipt schema가 다릅니다")
    _bool(receipt["passed"], label="post_start_receipt.passed", expected=True)
    _bool(
        receipt["observed_after_stream_start"],
        label="post_start_receipt.observed_after_stream_start",
        expected=True,
    )
    _integer(receipt["resolved_input_device"], label="post_start_receipt.resolved_input_device", expected=input_device)
    _integer(receipt["resolved_output_device"], label="post_start_receipt.resolved_output_device", expected=output_device)
    for name, expected_port in (
        (
            "input_hw_params",
            {
                "card": "APE",
                "pcm": 1,
                "format": "S32_LE",
                "sample_rate_hz": SAMPLE_RATE,
                "channels": 2,
                "period_frames": BLOCK_SIZE,
            },
        ),
        (
            "output_hw_params",
            {
                "card": "APE",
                "pcm": 0,
                "format": "S32_LE",
                "sample_rate_hz": SAMPLE_RATE,
                "channels": 2,
                "period_frames": BLOCK_SIZE,
            },
        ),
    ):
        port = _exact_mapping(receipt[name], label=f"post_start_receipt.{name}", expected=_HW_PARAMS_KEYS)
        for key, expected in expected_port.items():
            if port[key] != expected or type(port[key]) is not type(expected):
                raise ValueError(f"post_start_receipt.{name}.{key}가 S32 route 계약과 다릅니다")
    routes = _exact_mapping(receipt["routes"], label="post_start_receipt.routes", expected=_ROUTE_KEYS)
    if routes["input"] != "I2S2_ADMAIF2_ERR_REF" or routes["output"] != "ADMAIF1_I2S1_RT5640_J511":
        raise ValueError("post_start_receipt route가 RT5640 S32 contract와 다릅니다")
    _bool(routes["observed_after_stream_start"], label="post_start_receipt.routes.observed_after_stream_start", expected=True)
    jack = _exact_mapping(receipt["j511"], label="post_start_receipt.j511", expected=_JACK_KEYS)
    if jack["state"] not in {"HP", "HS"}:
        raise ValueError("post_start_receipt J511 state는 HP 또는 HS여야 합니다")
    _integer(jack["samples"], label="post_start_receipt.j511.samples", minimum=1)
    _bool(jack["all_samples_equal"], label="post_start_receipt.j511.all_samples_equal", expected=True)
    occupancy = _exact_mapping(receipt["occupancy"], label="post_start_receipt.occupancy", expected=_OCCUPANCY_KEYS)
    _bool(occupancy["input_owned_by_this_capture"], label="post_start_receipt.occupancy.input_owned_by_this_capture", expected=True)
    _bool(occupancy["output_owned_by_this_capture"], label="post_start_receipt.occupancy.output_owned_by_this_capture", expected=True)
    if type(occupancy["other_pcm_owners"]) is not list or occupancy["other_pcm_owners"]:
        raise ValueError("post_start_receipt.occupancy.other_pcm_owners는 empty exact list여야 합니다")
    _sha256(receipt["pre_snapshot_sha256"], label="post_start_receipt.pre_snapshot_sha256")
    _sha256(receipt["post_snapshot_sha256"], label="post_start_receipt.post_snapshot_sha256")
    core = dict(receipt)
    return {**core, "payload_sha256": _payload_sha256(core)}


def _validate_time_arrays(telemetry: Mapping[str, Any], *, prefix: str, count: int) -> dict[str, np.ndarray]:
    definitions = _PLANNED_ARRAYS if prefix == "planned" else _PREARM_ARRAYS
    arrays = {
        name: _array(telemetry[name], label=f"telemetry.{name}", dtype=dtype, shape=(count,))
        for name, dtype in definitions.items()
    }
    sequence = arrays[f"{prefix}_callback_sequence"]
    starts = arrays[f"{prefix}_callback_start_frames"]
    frames = arrays[f"{prefix}_callback_frame_counts"]
    statuses = arrays[f"{prefix}_callback_status_bitmask"]
    if not np.array_equal(sequence, np.arange(count, dtype=_INDEX_DTYPE)):
        raise ValueError(f"telemetry.{prefix} callback sequence가 contiguous하지 않습니다")
    if not np.array_equal(frames, np.full(count, BLOCK_SIZE, dtype=_INDEX_DTYPE)):
        raise ValueError(f"telemetry.{prefix} callback frames가 exact 256이 아닙니다")
    if np.any(statuses):
        raise ValueError(f"telemetry.{prefix} callback status가 0이 아닙니다")
    for name in (
        f"{prefix}_input_buffer_adc_time",
        f"{prefix}_output_buffer_dac_time",
        f"{prefix}_callback_current_time",
    ):
        values = arrays[name]
        if not np.all(np.isfinite(values)) or (count > 1 and not np.all(np.diff(values) > 0.0)):
            raise ValueError(f"telemetry.{name}가 finite strict-monotonic이 아닙니다")
    if prefix == "planned":
        if not np.array_equal(starts, np.arange(count, dtype=_INDEX_DTYPE) * BLOCK_SIZE):
            raise ValueError("telemetry.planned callback start frame이 contiguous하지 않습니다")
    elif np.any(starts):
        raise ValueError("telemetry.prearm callback start frame은 모두 0이어야 합니다")
    return arrays


def validate_rt5640_s32_capture_admission_v10_3(
    *,
    plan: Mapping[str, Any],
    planned_pcm_s32: np.ndarray,
    captured_pcm: np.ndarray,
    telemetry: Mapping[str, Any],
    post_start_pre_arm_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """disarmed S32 success 반환을 capture-only receipt로 결속한다.

    이 함수가 반환하는 ``transport_capture_pass``는 callback application buffer와
    in-memory capture의 정합성만 의미한다. electrical output/tap sample identity나
    plant/training authority는 false로 고정한다.
    """

    checked_plan, expected_pcm = validate_rt5640_fullband_s32_plan_v10(plan, planned_pcm_s32)
    frames = int(expected_pcm.shape[0])
    callbacks = frames // BLOCK_SIZE
    captured = _array(captured_pcm, label="captured_pcm", dtype=PCM_DTYPE, shape=(frames, 2))
    raw = _mapping(telemetry, label="disarmed S32 telemetry")
    _required(raw, label="disarmed S32 telemetry")

    if raw["schema"] != TELEMETRY_SCHEMA:
        raise ValueError("telemetry.schema가 disarmed S32 schema와 다릅니다")
    if raw["authority"] != "application_buffer_disarmed_planned_s32_only_no_physical_sample_identity":
        raise ValueError("telemetry.authority가 application-buffer ceiling과 다릅니다")
    for name, expected in {
        "portaudio_application_buffer_only": True,
        "post_start_pre_arm_check_required": True,
        "post_start_pre_arm_check_passed": True,
        "nonzero_assignment_before_arm_allowed": False,
        "on_stream_started_is_nonzero_admission": False,
        "hardware_sample_slip_authority": False,
        "physical_output_authority": False,
        "electrical_output_authority": False,
        "acoustic_output_authority": False,
        "dither_off": True,
        "prearm_output_zero_observed": True,
        "possible_nonzero_output_after_failed_assignment": False,
        "planned_callback_sequence_contiguous": True,
        "planned_callback_start_frames_contiguous": True,
        "planned_callback_frame_counts_exact": True,
        "capture_valid_all_true": True,
        "submitted_valid_all_true": True,
        "full_frame_accounting_valid": True,
        "actual_matches_planned_application_buffer": True,
        "actual_submitted_pcm_hash_eligible": True,
        "completed": True,
        "stream_start_returned_without_exception": True,
        "stream_stop_attempted": True,
        "stream_stop_returned_without_exception": True,
        "stream_abort_attempted": False,
        "stream_abort_returned_without_exception": False,
        "stream_close_attempted": True,
        "stream_close_returned_without_exception": True,
        "normal_stop_completed": True,
    }.items():
        _bool(raw[name], label=f"telemetry.{name}", expected=expected)
    _integer(raw["sample_rate_hz"], label="telemetry.sample_rate_hz", expected=SAMPLE_RATE)
    _integer(raw["block_size"], label="telemetry.block_size", expected=BLOCK_SIZE)
    if raw["latency"] != "low" or raw["channels"] != [2, 2]:
        raise ValueError("telemetry latency/channels가 S32 contract와 다릅니다")
    if raw["input_dtype"] != PCM_DTYPE.str or raw["output_dtype"] != PCM_DTYPE.str:
        raise ValueError("telemetry S32 dtype가 다릅니다")
    for name in ("resolved_input_device", "resolved_output_device"):
        _integer(raw[name], label=f"telemetry.{name}", minimum=0)
    post_start = _validate_post_start_pre_arm_receipt(
        post_start_pre_arm_receipt,
        input_device=raw["resolved_input_device"],
        output_device=raw["resolved_output_device"],
    )
    if raw["planned_pcm_sha256"] != checked_plan["sealed_planned_s32_pcm_sha256"]:
        raise ValueError("telemetry planned S32 SHA가 sealed plan과 다릅니다")
    _integer(raw["planned_frames"], label="telemetry.planned_frames", expected=frames)
    _integer(raw["expected_planned_callbacks"], label="telemetry.expected_planned_callbacks", expected=callbacks)
    for name in ("captured_frames", "submitted_frames"):
        _integer(raw[name], label=f"telemetry.{name}", expected=frames)
    for name in ("xrun_count", "status_present_count"):
        _integer(raw[name], label=f"telemetry.{name}", expected=0)
    _integer(raw["callback_planned_assignment_attempt_count"], label="telemetry.callback_planned_assignment_attempt_count", expected=callbacks)
    _integer(raw["callback_planned_assignment_confirmed_count"], label="telemetry.callback_planned_assignment_confirmed_count", expected=callbacks)
    prearm_count = _integer(raw["prearm_callback_count"], label="telemetry.prearm_callback_count", minimum=0)
    zero_confirmed = _integer(raw["callback_zero_confirmed_count"], label="telemetry.callback_zero_confirmed_count", minimum=callbacks + prearm_count)
    _integer(raw["callback_zero_attempt_count"], label="telemetry.callback_zero_attempt_count", minimum=zero_confirmed)

    for name in (
        "callback_error",
        "stream_constructor_error",
        "stream_start_error",
        "post_start_pre_arm_check_error",
        "watchdog_error",
        "stream_stop_error",
        "stream_abort_error",
        "stream_close_error",
        "on_output_closed_error",
    ):
        if raw[name] is not None:
            raise ValueError(f"telemetry.{name}는 exact None이어야 합니다")
    for name in ("canonical_invalid_reasons", "termination_signals", "faults"):
        if type(raw[name]) is not list or raw[name]:
            raise ValueError(f"telemetry.{name}는 empty exact list여야 합니다")

    planned_arrays = _validate_time_arrays(raw, prefix="planned", count=callbacks)
    prearm_arrays = _validate_time_arrays(raw, prefix="prearm", count=prearm_count)
    if prearm_count and callbacks:
        if not (
            prearm_arrays["prearm_input_buffer_adc_time"][-1]
            < planned_arrays["planned_input_buffer_adc_time"][0]
            and prearm_arrays["prearm_output_buffer_dac_time"][-1]
            < planned_arrays["planned_output_buffer_dac_time"][0]
            and prearm_arrays["prearm_callback_current_time"][-1]
            < planned_arrays["planned_callback_current_time"][0]
        ):
            raise ValueError("prearm→planned callback timestamp 전체 순서가 strict-monotonic이 아닙니다")
    actual = _array(raw["actual_submitted_pcm"], label="telemetry.actual_submitted_pcm", dtype=PCM_DTYPE, shape=(frames, 2))
    capture_mask = _array(raw["capture_valid_mask"], label="telemetry.capture_valid_mask", dtype=MASK_DTYPE, shape=(frames,))
    submit_mask = _array(raw["submitted_valid_mask"], label="telemetry.submitted_valid_mask", dtype=MASK_DTYPE, shape=(frames,))
    if not np.all(capture_mask) or not np.all(submit_mask):
        raise ValueError("telemetry capture/submitted valid mask에 false가 있습니다")
    if not np.array_equal(actual, expected_pcm):
        raise ValueError("telemetry actual submitted S32 PCM이 sealed planned PCM과 다릅니다")
    if np.any(np.bitwise_and(actual.astype(np.int64), (1 << 16) - 1)):
        raise ValueError("telemetry actual submitted S32 PCM의 low 16 bits가 0이 아닙니다")
    actual_sha = _array_sha256(actual)
    if raw["actual_submitted_pcm_sha256"] != actual_sha or actual_sha != checked_plan["sealed_planned_s32_pcm_sha256"]:
        raise ValueError("telemetry actual submitted S32 PCM SHA가 sealed plan과 다릅니다")

    pre_open_started = _finite(raw["pre_open_monotonic_started"], label="telemetry.pre_open_monotonic_started", minimum=0.0)
    pre_open_completed = _finite(raw["pre_open_monotonic_completed"], label="telemetry.pre_open_monotonic_completed", minimum=pre_open_started)
    capture_started = _finite(raw["capture_monotonic_started"], label="telemetry.capture_monotonic_started", minimum=pre_open_completed)
    arm_started = _finite(raw["arm_check_monotonic_started"], label="telemetry.arm_check_monotonic_started", minimum=capture_started)
    arm_completed = _finite(raw["arm_check_monotonic_completed"], label="telemetry.arm_check_monotonic_completed", minimum=arm_started)
    _finite(raw["arm_monotonic"], label="telemetry.arm_monotonic", minimum=arm_completed)
    capture_completed = _finite(raw["capture_monotonic_completed"], label="telemetry.capture_monotonic_completed", minimum=capture_started)
    elapsed = _finite(raw["capture_monotonic_elapsed_seconds"], label="telemetry.capture_monotonic_elapsed_seconds", minimum=0.0)
    grace = _finite(raw["watchdog_grace_seconds"], label="telemetry.watchdog_grace_seconds", minimum=0.0)
    if not math.isclose(capture_completed - capture_started, elapsed, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("telemetry capture elapsed 재계산이 다릅니다")
    nominal = frames / SAMPLE_RATE
    if not nominal - 1.0e-3 <= elapsed <= nominal + grace + 1.0e-3:
        raise ValueError("telemetry capture elapsed가 nominal/watchdog 범위 밖입니다")

    callback_arrays = {
        **planned_arrays,
        **prearm_arrays,
        "capture_valid_mask": capture_mask,
        "submitted_valid_mask": submit_mask,
    }
    core: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "transport_capture_pass": True,
        "capture_only": True,
        "plan_payload_sha256": checked_plan["canonical_payload_sha256"],
        "planned_s32_pcm_sha256": checked_plan["sealed_planned_s32_pcm_sha256"],
        "actual_submitted_s32_pcm_sha256": actual_sha,
        "captured_pcm_sha256": _array_sha256(captured),
        "callback_array_sha256": {name: _array_sha256(array) for name, array in sorted(callback_arrays.items())},
        "resolved_devices": {
            "input": raw["resolved_input_device"],
            "output": raw["resolved_output_device"],
        },
        "post_start_pre_arm_receipt_payload_sha256": post_start["payload_sha256"],
        "prearm_callback_count": prearm_count,
        "callback_count": callbacks,
        "submitted_frames": frames,
        "captured_frames": frames,
        "timing": {
            "capture_monotonic_elapsed_seconds": elapsed,
            "watchdog_grace_seconds": grace,
            "callback_timestamp_semantics": "portaudio_application_observation_not_hardware_sample_identity",
        },
        "authority": {
            "s32_transport_capture_pass": True,
            "hardware_sample_slip_authority": False,
            "physical_output_authority": False,
            "electrical_witness_bound": False,
            "fullband_plant_identification_pass": False,
            "canonical_training_eligible": False,
            "deployment_eligible": False,
        },
        "next_required_gates": [
            "external_synchronous_electrical_witness_raw",
            "aperiodic_s32_to_tap_frame_identity",
            "raw_first_fullband_P_S_analysis",
            "v3_population_batch_DNH_and_trainer_admission",
        ],
    }
    return {**core, "canonical_payload_sha256": _payload_sha256(core)}


def assess_rt5640_s32_capture_admission_v10_3(
    *,
    plan: Mapping[str, Any],
    planned_pcm_s32: np.ndarray,
    capture: tuple[np.ndarray, Mapping[str, Any]] | S32DisarmedDuplexCaptureFailure,
    post_start_pre_arm_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """success는 strict validator로, partial failure는 명시적 BLOCKED receipt로 변환한다.

    failure raw를 지우거나 다시 측정하라는 판단을 하지 않는다. publication adapter는
    partial arrays를 immutable raw로 보존해야 하며, 이 pure boundary는 그 결과를 P/S에
    쓰지 못하게 하는 표지밖에 만들지 않는다.
    """

    checked_plan, _ = validate_rt5640_fullband_s32_plan_v10(plan, planned_pcm_s32)
    if isinstance(capture, S32DisarmedDuplexCaptureFailure):
        core: dict[str, Any] = {
            "schema": SCHEMA,
            "status": BLOCKED_STATUS,
            "transport_capture_pass": False,
            "capture_only": True,
            "plan_payload_sha256": checked_plan["canonical_payload_sha256"],
            "planned_s32_pcm_sha256": checked_plan["sealed_planned_s32_pcm_sha256"],
            "partial_capture_observed": True,
            "failure_type": type(capture).__name__,
            "failure_message": str(capture),
            "authority": {
                "s32_transport_capture_pass": False,
                "hardware_sample_slip_authority": False,
                "physical_output_authority": False,
                "electrical_witness_bound": False,
                "fullband_plant_identification_pass": False,
                "canonical_training_eligible": False,
                "deployment_eligible": False,
            },
        }
        return {**core, "canonical_payload_sha256": _payload_sha256(core)}
    if not isinstance(capture, tuple) or len(capture) != 2:
        raise TypeError("capture는 (captured_pcm, telemetry) 또는 S32DisarmedDuplexCaptureFailure여야 합니다")
    if post_start_pre_arm_receipt is None:
        raise ValueError("성공 S32 capture에는 post-start pre-arm receipt가 필수입니다")
    captured_pcm, telemetry = capture
    return validate_rt5640_s32_capture_admission_v10_3(
        plan=checked_plan,
        planned_pcm_s32=planned_pcm_s32,
        captured_pcm=captured_pcm,
        telemetry=telemetry,
        post_start_pre_arm_receipt=post_start_pre_arm_receipt,
    )


def build_expected_rt5640_s32_capture_admission_plan_v10_3() -> tuple[dict[str, Any], np.ndarray]:
    """테스트/후속 adapter가 사용할 sealed plan을 명시적으로 제공한다.

    이 helper도 device나 result 파일을 열지 않는다.
    """

    return build_rt5640_fullband_s32_plan_v10()


__all__ = [
    "PCM_DTYPE",
    "BLOCKED_STATUS",
    "POST_START_PRE_ARM_RECEIPT_SCHEMA",
    "SCHEMA",
    "STATUS",
    "assess_rt5640_s32_capture_admission_v10_3",
    "build_expected_rt5640_s32_capture_admission_plan_v10_3",
    "validate_rt5640_s32_capture_admission_v10_3",
]
