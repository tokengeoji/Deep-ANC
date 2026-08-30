"""RT5640 동시 무음 duplex를 위한 hardware-independent 순수 계약.

이 모듈은 오디오 backend, ALSA, ``sounddevice``를 import하거나 PCM 장치를 열지
않는다. :mod:`deep_anc.audio_zero_duplex`가 보존한 telemetry와 exact-zero 출력만
검증한다. all-zero payload는 물리 sample drop/add/reorder, callback deadline,
fallback, J511 route 또는 shared clock을 증명할 수 없으므로 성공 receipt의 권위
상한은 항상 ``ZERO_DUPLEX_TRANSPORT_SMOKE_PASS``다.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np


SAMPLE_RATE_HZ = 48_000
CHANNELS = 2
BLOCK_SIZE = 256
PCM_DTYPE = np.dtype("<i4")
TIME_TOLERANCE_SECONDS = 1.0e-3
WATCHDOG_GRACE_SECONDS = 2.0

PLAN_SCHEMA = "rt5640_zero_duplex_plan_v1"
TELEMETRY_SCHEMA = "rt5640_zero_duplex_telemetry_v1"
TELEMETRY_RECEIPT_SCHEMA = "rt5640_zero_duplex_telemetry_receipt_v1"
RECEIPT_SCHEMA = "rt5640_zero_duplex_transport_smoke_receipt_v1"
AUTHORITY_CEILING = "ZERO_DUPLEX_TRANSPORT_SMOKE_PASS"

_CALLBACK_ARRAY_FIELDS = frozenset(
    {
        "callback_sequence",
        "callback_start_frames",
        "callback_frame_counts",
        "callback_status_bitmask",
        "input_buffer_adc_time",
        "output_buffer_dac_time",
        "callback_current_time",
        "capture_valid_mask",
        "submitted_valid_mask",
    }
)

_TELEMETRY_FIELDS = frozenset(
    {
        "schema",
        "authority",
        "callback_frame_semantics",
        "output_zero_scope",
        "portaudio_application_buffer_only",
        "portaudio_timestamp_authority",
        "hardware_sample_slip_authority",
        "physical_output_zero_authority",
        "electrical_output_zero_authority",
        "acoustic_output_zero_authority",
        "sample_rate_hz",
        "block_size",
        "latency",
        "channels",
        "input_dtype",
        "output_dtype",
        "dither_off",
        "resolved_input_device",
        "resolved_output_device",
        "planned_frames",
        "expected_callbacks",
        "pre_open_monotonic_started",
        "pre_open_monotonic_completed",
        "capture_monotonic_started",
        "capture_monotonic_completed",
        "capture_monotonic_elapsed_seconds",
        "watchdog_grace_seconds",
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
        "actual_submitted_nonzero_count",
        "callback_zero_attempt_count",
        "callback_zero_confirmed_count",
        "callback_sequence_contiguous",
        "callback_start_frames_contiguous",
        "callback_frame_counts_exact",
        "capture_valid_all_true",
        "submitted_valid_all_true",
        "full_frame_accounting_valid",
        "actual_submitted_pcm_hash_eligible",
        "application_buffer_zero_submission_complete",
        "completed",
        "callback_error",
        "canonical_invalid_reasons",
        "stream_constructor_error",
        "stream_start_error",
        "watchdog_error",
        "stream_stop_error",
        "stream_abort_error",
        "stream_close_error",
        "on_output_closed_error",
        "termination_signals",
        "termination_signal",
        "termination_exit_code",
        "stream_start_returned_without_exception",
        "stream_stop_attempted",
        "stream_stop_returned_without_exception",
        "stream_abort_attempted",
        "stream_abort_returned_without_exception",
        "stream_close_attempted",
        "stream_close_returned_without_exception",
        "normal_stop_completed",
        "faults",
        "actual_submitted_pcm",
        "capture_valid_mask",
        "submitted_valid_mask",
    }
)

_PLAN_FIELDS = frozenset(
    {
        "schema",
        "sample_rate_hz",
        "channels",
        "block_size",
        "input_dtype",
        "output_dtype",
        "frame_count",
        "callback_count",
        "nominal_duration_seconds",
        "watchdog_grace_seconds",
        "watchdog_deadline_seconds",
        "output_semantics",
        "planned_pcm_sha256",
        "zero_payload_sha256",
        "zero_payload_bytes",
        "authority_ceiling",
        "canonical_payload_sha256",
    }
)

_NON_AUTHORITATIVE_OBSERVATION = {
    "callback_deadline_miss_authority": False,
    "callback_deadline_miss_observed": None,
    "fallback_block_authority": False,
    "fallback_block_observed": None,
    "hardware_drop_add_authority": False,
    "hardware_drop_sample_observed": None,
    "hardware_add_sample_observed": None,
}

_TELEMETRY_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "passed",
        "authority_ceiling",
        "plan_payload_sha256",
        "planned_pcm_sha256",
        "zero_payload_sha256",
        "actual_submitted_pcm_sha256",
        "actual_submitted_zero_payload_sha256",
        "all_submitted_buffers_bitwise_zero",
        "resolved_input_device",
        "resolved_output_device",
        "callback_count",
        "submitted_frames",
        "captured_frames",
        "callback_array_sha256",
        "pre_open_monotonic_started",
        "pre_open_monotonic_completed",
        "capture_monotonic_started",
        "capture_monotonic_completed",
        "capture_monotonic_elapsed_seconds",
        "watchdog_grace_seconds",
        "host_watchdog_completed_within_deadline",
        "portaudio_status_witness_pass",
        "portaudio_status_present_count",
        "portaudio_xrun_count",
        "non_authoritative_observation",
        "callback_frame_semantics",
        "output_zero_scope",
        "portaudio_application_buffer_only",
        "portaudio_timestamp_authority",
        "hardware_sample_slip_authority",
        "physical_output_zero_authority",
        "electrical_output_zero_authority",
        "acoustic_output_zero_authority",
        "dither_off",
        "application_buffer_zero_submission_complete",
        "stream_close_returned_without_exception",
        "canonical_payload_sha256",
    }
)

_AUTHORITY = {
    "zero_duplex_transport_smoke_pass": True,
    "common_clock_topology_pass": False,
    "shared_clock_authority_pass": False,
    "hardware_sample_slip_authority": False,
    "sample_identity_pass": False,
    "physical_output_route_pass": False,
    "electrical_witness_pass": False,
    "plant_identification_pass": False,
    "canonical_training_eligible": False,
    "attenuation_assessed": False,
}

_LIMITATIONS = [
    "software_callback_accounting_and_portaudio_status_only",
    "all_zero_cannot_detect_physical_sample_drop_add_or_reorder",
    "callback_deadline_and_fallback_are_not_observed_by_this_primitive",
    "all_zero_does_not_prove_j511_route_or_shared_clock",
]


def canonical_json_bytes(value: object) -> bytes:
    """정렬·compact·NaN 금지 canonical JSON bytes를 반환한다."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        raise ValueError("array SHA 입력은 C-contiguous여야 합니다")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _raw_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        raise ValueError("raw SHA 입력은 C-contiguous여야 합니다")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _exact_mapping(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}는 mapping이어야 합니다")
    observed = set(value)
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        extra = sorted(observed - set(expected))
        raise ValueError(f"{label} key 집합이 다릅니다: missing={missing}, extra={extra}")
    if any(type(key) is not str or not key for key in value):
        raise ValueError(f"{label} key는 nonempty exact str이어야 합니다")
    return dict(value)


def _exact_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}는 {minimum} 이상 exact int여야 합니다")
    return value


def _exact_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label}는 exact bool이어야 합니다")
    return value


def _exact_float(value: Any, *, label: str, minimum: float | None = None) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label}는 exact finite float여야 합니다")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label}는 {minimum} 이상이어야 합니다")
    return value


def _exact_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label}는 lowercase SHA-256 hex여야 합니다")
    return value


def _exact_array(
    value: Any,
    *,
    label: str,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise ValueError(f"{label}는 exact numpy.ndarray여야 합니다")
    array = value
    if array.dtype != dtype or array.shape != shape:
        raise ValueError(
            f"{label} dtype/shape이 exact {dtype.str}/{shape}가 아닙니다: "
            f"{array.dtype.str}/{array.shape}"
        )
    if not array.flags.c_contiguous:
        raise ValueError(f"{label}는 C-contiguous여야 합니다")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label}에 non-finite 값이 있습니다")
    return array


def _json_roundtrip(value: Mapping[str, Any]) -> dict[str, Any]:
    encoded = canonical_json_bytes(dict(value))
    decoded = json.loads(encoded.decode("utf-8"))
    if canonical_json_bytes(decoded) != encoded:
        raise ValueError("canonical JSON round-trip이 안정적이지 않습니다")
    return decoded


def build_zero_duplex_plan(*, frame_count: int) -> tuple[dict[str, Any], np.ndarray]:
    """48 kHz/2ch/block-256/S32_LE exact-zero 계획을 결정론적으로 만든다."""

    frames = _exact_int(frame_count, label="frame_count", minimum=1)
    if frames % BLOCK_SIZE:
        raise ValueError("frame_count는 exact block-256 배수여야 합니다")
    pcm = np.zeros((frames, CHANNELS), dtype=PCM_DTYPE, order="C")
    nominal = float(frames / SAMPLE_RATE_HZ)
    core: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "channels": CHANNELS,
        "block_size": BLOCK_SIZE,
        "input_dtype": PCM_DTYPE.str,
        "output_dtype": PCM_DTYPE.str,
        "frame_count": frames,
        "callback_count": frames // BLOCK_SIZE,
        "nominal_duration_seconds": nominal,
        "watchdog_grace_seconds": WATCHDOG_GRACE_SECONDS,
        "watchdog_deadline_seconds": float(nominal + WATCHDOG_GRACE_SECONDS),
        "output_semantics": "every_submitted_sample_bitwise_exact_zero",
        "planned_pcm_sha256": _array_sha256(pcm),
        "zero_payload_sha256": _raw_sha256(pcm),
        "zero_payload_bytes": int(pcm.nbytes),
        "authority_ceiling": AUTHORITY_CEILING,
    }
    plan = _json_roundtrip({**core, "canonical_payload_sha256": payload_sha256(core)})
    pcm.setflags(write=False)
    return plan, pcm


def validate_zero_duplex_plan(
    plan: Mapping[str, Any], planned_pcm: np.ndarray
) -> dict[str, Any]:
    checked = _exact_mapping(plan, _PLAN_FIELDS, label="zero duplex plan")
    if checked["schema"] != PLAN_SCHEMA:
        raise ValueError("zero duplex plan schema가 다릅니다")
    for name, expected in (
        ("sample_rate_hz", SAMPLE_RATE_HZ),
        ("channels", CHANNELS),
        ("block_size", BLOCK_SIZE),
    ):
        if _exact_int(checked[name], label=f"plan.{name}") != expected:
            raise ValueError(f"plan.{name} 계약이 다릅니다")
    if checked["input_dtype"] != PCM_DTYPE.str or checked["output_dtype"] != PCM_DTYPE.str:
        raise ValueError("plan PCM dtype 계약이 다릅니다")
    frames = _exact_int(checked["frame_count"], label="plan.frame_count", minimum=1)
    if frames % BLOCK_SIZE:
        raise ValueError("plan.frame_count는 block 배수여야 합니다")
    callbacks = _exact_int(
        checked["callback_count"], label="plan.callback_count", minimum=1
    )
    if callbacks != frames // BLOCK_SIZE:
        raise ValueError("plan callback count가 frame count와 다릅니다")
    nominal = _exact_float(
        checked["nominal_duration_seconds"],
        label="plan.nominal_duration_seconds",
        minimum=0.0,
    )
    if nominal != float(frames / SAMPLE_RATE_HZ):
        raise ValueError("plan nominal duration 재계산이 다릅니다")
    grace = _exact_float(
        checked["watchdog_grace_seconds"],
        label="plan.watchdog_grace_seconds",
        minimum=0.0,
    )
    if grace != WATCHDOG_GRACE_SECONDS:
        raise ValueError("plan watchdog grace가 고정 계약과 다릅니다")
    deadline = _exact_float(
        checked["watchdog_deadline_seconds"],
        label="plan.watchdog_deadline_seconds",
        minimum=nominal,
    )
    if deadline != float(nominal + WATCHDOG_GRACE_SECONDS):
        raise ValueError("plan watchdog deadline 재계산이 다릅니다")
    if checked["output_semantics"] != "every_submitted_sample_bitwise_exact_zero":
        raise ValueError("plan output semantics가 exact zero가 아닙니다")
    if checked["authority_ceiling"] != AUTHORITY_CEILING:
        raise ValueError("plan authority ceiling이 허용 범위를 넘습니다")
    zero_bytes = _exact_int(
        checked["zero_payload_bytes"], label="plan.zero_payload_bytes"
    )
    if zero_bytes != frames * CHANNELS * PCM_DTYPE.itemsize:
        raise ValueError("plan zero payload byte 수가 다릅니다")
    pcm = _exact_array(
        planned_pcm,
        label="planned_pcm",
        dtype=PCM_DTYPE,
        shape=(frames, CHANNELS),
    )
    if np.any(pcm.view(np.uint8) != 0):
        raise ValueError("planned_pcm은 모든 bit가 exact zero여야 합니다")
    _exact_sha256(checked["planned_pcm_sha256"], label="plan.planned_pcm_sha256")
    _exact_sha256(checked["zero_payload_sha256"], label="plan.zero_payload_sha256")
    if checked["planned_pcm_sha256"] != _array_sha256(pcm):
        raise ValueError("planned_pcm SHA가 plan과 다릅니다")
    if checked["zero_payload_sha256"] != _raw_sha256(pcm):
        raise ValueError("zero payload SHA가 plan과 다릅니다")
    declared = _exact_sha256(
        checked["canonical_payload_sha256"], label="plan.canonical_payload_sha256"
    )
    core = {
        key: value
        for key, value in checked.items()
        if key != "canonical_payload_sha256"
    }
    if declared != payload_sha256(core):
        raise ValueError("plan canonical payload SHA가 다릅니다")
    return _json_roundtrip(checked)


def validate_zero_duplex_telemetry(
    *,
    plan: Mapping[str, Any],
    planned_pcm: np.ndarray,
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    """현재 zero-duplex live primitive의 성공 telemetry를 fail-closed 검증한다."""

    checked_plan = validate_zero_duplex_plan(plan, planned_pcm)
    raw = _exact_mapping(telemetry, _TELEMETRY_FIELDS, label="zero duplex telemetry")
    if raw["schema"] != TELEMETRY_SCHEMA:
        raise ValueError("telemetry.schema 계약이 다릅니다")
    if raw["authority"] != "zero_duplex_transport_only_no_sample_identity":
        raise ValueError("telemetry.authority 계약이 다릅니다")
    if raw["callback_frame_semantics"] != (
        "software_accounting_only_not_hardware_slip_witness"
    ):
        raise ValueError("telemetry.callback_frame_semantics 계약이 다릅니다")
    if raw["output_zero_scope"] != "portaudio_application_callback_buffer_only":
        raise ValueError("telemetry.output_zero_scope 계약이 다릅니다")
    for name in (
        "portaudio_timestamp_authority",
        "hardware_sample_slip_authority",
        "physical_output_zero_authority",
        "electrical_output_zero_authority",
        "acoustic_output_zero_authority",
    ):
        if _exact_bool(raw[name], label=f"telemetry.{name}") is not False:
            raise ValueError(f"telemetry.{name}는 exact false여야 합니다")
    for name in ("portaudio_application_buffer_only", "dither_off"):
        if _exact_bool(raw[name], label=f"telemetry.{name}") is not True:
            raise ValueError(f"telemetry.{name}는 exact true여야 합니다")
    for name, expected in (
        ("sample_rate_hz", SAMPLE_RATE_HZ),
        ("block_size", BLOCK_SIZE),
    ):
        if _exact_int(raw[name], label=f"telemetry.{name}") != expected:
            raise ValueError(f"telemetry.{name} 계약이 다릅니다")
    if type(raw["channels"]) is not list or raw["channels"] != [CHANNELS, CHANNELS]:
        raise ValueError("telemetry.channels는 exact [2, 2] list여야 합니다")
    if any(type(value) is not int for value in raw["channels"]):
        raise ValueError("telemetry.channels 원소는 exact int여야 합니다")
    if raw["latency"] != "low":
        raise ValueError("telemetry.latency는 exact low여야 합니다")
    if raw["input_dtype"] != PCM_DTYPE.str or raw["output_dtype"] != PCM_DTYPE.str:
        raise ValueError("telemetry PCM dtype 계약이 다릅니다")
    input_device = _exact_int(
        raw["resolved_input_device"], label="telemetry.resolved_input_device"
    )
    output_device = _exact_int(
        raw["resolved_output_device"], label="telemetry.resolved_output_device"
    )

    frames = checked_plan["frame_count"]
    callbacks = checked_plan["callback_count"]
    if _exact_int(raw["planned_frames"], label="telemetry.planned_frames") != frames:
        raise ValueError("telemetry.planned_frames가 sealed plan과 다릅니다")
    if (
        _exact_int(raw["expected_callbacks"], label="telemetry.expected_callbacks")
        != callbacks
    ):
        raise ValueError("telemetry.expected_callbacks가 sealed plan과 다릅니다")

    actual = _exact_array(
        raw["actual_submitted_pcm"],
        label="telemetry.actual_submitted_pcm",
        dtype=PCM_DTYPE,
        shape=(frames, CHANNELS),
    )
    if np.any(actual.view(np.uint8) != 0):
        raise ValueError("모든 actual submitted buffer는 bitwise exact zero여야 합니다")
    if not np.array_equal(actual, planned_pcm):
        raise ValueError("actual submitted PCM이 sealed planned PCM과 다릅니다")

    callback_arrays = {
        "callback_sequence": _exact_array(
            raw["callback_sequence"],
            label="telemetry.callback_sequence",
            dtype=np.dtype("<i8"),
            shape=(callbacks,),
        ),
        "callback_start_frames": _exact_array(
            raw["callback_start_frames"],
            label="telemetry.callback_start_frames",
            dtype=np.dtype("<i8"),
            shape=(callbacks,),
        ),
        "callback_frame_counts": _exact_array(
            raw["callback_frame_counts"],
            label="telemetry.callback_frame_counts",
            dtype=np.dtype("<i8"),
            shape=(callbacks,),
        ),
        "callback_status_bitmask": _exact_array(
            raw["callback_status_bitmask"],
            label="telemetry.callback_status_bitmask",
            dtype=np.dtype("<u4"),
            shape=(callbacks,),
        ),
        "input_buffer_adc_time": _exact_array(
            raw["input_buffer_adc_time"],
            label="telemetry.input_buffer_adc_time",
            dtype=np.dtype("<f8"),
            shape=(callbacks,),
        ),
        "output_buffer_dac_time": _exact_array(
            raw["output_buffer_dac_time"],
            label="telemetry.output_buffer_dac_time",
            dtype=np.dtype("<f8"),
            shape=(callbacks,),
        ),
        "callback_current_time": _exact_array(
            raw["callback_current_time"],
            label="telemetry.callback_current_time",
            dtype=np.dtype("<f8"),
            shape=(callbacks,),
        ),
        "capture_valid_mask": _exact_array(
            raw["capture_valid_mask"],
            label="telemetry.capture_valid_mask",
            dtype=np.dtype(np.bool_),
            shape=(frames,),
        ),
        "submitted_valid_mask": _exact_array(
            raw["submitted_valid_mask"],
            label="telemetry.submitted_valid_mask",
            dtype=np.dtype(np.bool_),
            shape=(frames,),
        ),
    }
    if not np.array_equal(
        callback_arrays["callback_sequence"], np.arange(callbacks, dtype="<i8")
    ):
        raise ValueError("callback sequence가 exact 0..N-1이 아닙니다")
    if not np.array_equal(
        callback_arrays["callback_start_frames"],
        np.arange(callbacks, dtype="<i8") * BLOCK_SIZE,
    ):
        raise ValueError("callback start frame accounting이 연속이 아닙니다")
    if np.any(callback_arrays["callback_frame_counts"] != BLOCK_SIZE):
        raise ValueError("callback frame count가 exact block-256이 아닙니다")
    if np.any(callback_arrays["callback_status_bitmask"] != 0):
        raise ValueError("PortAudio callback status는 모두 exact 0이어야 합니다")
    if not np.all(callback_arrays["capture_valid_mask"]):
        raise ValueError("capture valid mask에 빈 frame이 있습니다")
    if not np.all(callback_arrays["submitted_valid_mask"]):
        raise ValueError("submitted valid mask에 빈 frame이 있습니다")
    for name in (
        "input_buffer_adc_time",
        "output_buffer_dac_time",
        "callback_current_time",
    ):
        if np.any(np.diff(callback_arrays[name]) <= 0.0):
            raise ValueError(f"{name}는 finite strict-monotonic이어야 합니다")

    for name in ("submitted_frames", "captured_frames"):
        if _exact_int(raw[name], label=f"telemetry.{name}") != frames:
            raise ValueError(f"telemetry.{name}가 planned frames와 다릅니다")
    for name in ("xrun_count", "status_present_count", "actual_submitted_nonzero_count"):
        if _exact_int(raw[name], label=f"telemetry.{name}") != 0:
            raise ValueError(f"telemetry.{name}는 exact 0이어야 합니다")
    for name in ("callback_zero_attempt_count", "callback_zero_confirmed_count"):
        if _exact_int(raw[name], label=f"telemetry.{name}") != callbacks:
            raise ValueError(f"telemetry.{name}가 expected callbacks와 다릅니다")
    for name in (
        "callback_sequence_contiguous",
        "callback_start_frames_contiguous",
        "callback_frame_counts_exact",
        "capture_valid_all_true",
        "submitted_valid_all_true",
        "full_frame_accounting_valid",
        "actual_submitted_pcm_hash_eligible",
        "application_buffer_zero_submission_complete",
        "completed",
        "stream_start_returned_without_exception",
        "stream_stop_attempted",
        "stream_stop_returned_without_exception",
        "stream_close_attempted",
        "stream_close_returned_without_exception",
        "normal_stop_completed",
    ):
        if _exact_bool(raw[name], label=f"telemetry.{name}") is not True:
            raise ValueError(f"telemetry.{name}는 exact true여야 합니다")
    for name in ("stream_abort_attempted", "stream_abort_returned_without_exception"):
        if _exact_bool(raw[name], label=f"telemetry.{name}") is not False:
            raise ValueError(f"telemetry.{name}는 exact false여야 합니다")
    for name in (
        "callback_error",
        "stream_constructor_error",
        "stream_start_error",
        "watchdog_error",
        "stream_stop_error",
        "stream_abort_error",
        "stream_close_error",
        "on_output_closed_error",
        "termination_signal",
        "termination_exit_code",
    ):
        if raw[name] is not None:
            raise ValueError(f"telemetry.{name}는 exact None이어야 합니다")
    if type(raw["canonical_invalid_reasons"]) is not list:
        raise ValueError("telemetry.canonical_invalid_reasons는 exact list여야 합니다")
    if raw["canonical_invalid_reasons"]:
        raise ValueError("telemetry.canonical_invalid_reasons는 비어 있어야 합니다")
    for name in ("termination_signals", "faults"):
        if type(raw[name]) is not list:
            raise ValueError(f"telemetry.{name}는 exact list여야 합니다")
        if raw[name]:
            raise ValueError(f"telemetry.{name}는 비어 있어야 합니다")

    pre_started = _exact_float(
        raw["pre_open_monotonic_started"],
        label="telemetry.pre_open_monotonic_started",
        minimum=0.0,
    )
    pre_completed = _exact_float(
        raw["pre_open_monotonic_completed"],
        label="telemetry.pre_open_monotonic_completed",
        minimum=pre_started,
    )
    started = _exact_float(
        raw["capture_monotonic_started"],
        label="telemetry.capture_monotonic_started",
        minimum=pre_completed,
    )
    completed = _exact_float(
        raw["capture_monotonic_completed"],
        label="telemetry.capture_monotonic_completed",
        minimum=started,
    )
    elapsed = _exact_float(
        raw["capture_monotonic_elapsed_seconds"],
        label="telemetry.capture_monotonic_elapsed_seconds",
        minimum=0.0,
    )
    grace = _exact_float(
        raw["watchdog_grace_seconds"],
        label="telemetry.watchdog_grace_seconds",
        minimum=0.0,
    )
    if grace != checked_plan["watchdog_grace_seconds"]:
        raise ValueError("watchdog grace가 sealed plan과 다릅니다")
    if not math.isclose(completed - started, elapsed, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("capture monotonic elapsed 재계산이 다릅니다")
    nominal = checked_plan["nominal_duration_seconds"]
    if not (
        nominal - TIME_TOLERANCE_SECONDS
        <= elapsed
        <= checked_plan["watchdog_deadline_seconds"] + TIME_TOLERANCE_SECONDS
    ):
        raise ValueError("capture elapsed가 planned-duration watchdog 범위 밖입니다")

    array_digests = {
        name: _array_sha256(callback_arrays[name]) for name in sorted(callback_arrays)
    }
    core: dict[str, Any] = {
        "schema": TELEMETRY_RECEIPT_SCHEMA,
        "passed": True,
        "authority_ceiling": AUTHORITY_CEILING,
        "plan_payload_sha256": checked_plan["canonical_payload_sha256"],
        "planned_pcm_sha256": checked_plan["planned_pcm_sha256"],
        "zero_payload_sha256": checked_plan["zero_payload_sha256"],
        "actual_submitted_pcm_sha256": _array_sha256(actual),
        "actual_submitted_zero_payload_sha256": _raw_sha256(actual),
        "all_submitted_buffers_bitwise_zero": True,
        "resolved_input_device": input_device,
        "resolved_output_device": output_device,
        "callback_count": callbacks,
        "submitted_frames": raw["submitted_frames"],
        "captured_frames": raw["captured_frames"],
        "callback_array_sha256": array_digests,
        "pre_open_monotonic_started": pre_started,
        "pre_open_monotonic_completed": pre_completed,
        "capture_monotonic_started": started,
        "capture_monotonic_completed": completed,
        "capture_monotonic_elapsed_seconds": elapsed,
        "watchdog_grace_seconds": grace,
        "host_watchdog_completed_within_deadline": True,
        "portaudio_status_witness_pass": True,
        "portaudio_status_present_count": raw["status_present_count"],
        "portaudio_xrun_count": raw["xrun_count"],
        "non_authoritative_observation": dict(_NON_AUTHORITATIVE_OBSERVATION),
        "callback_frame_semantics": raw["callback_frame_semantics"],
        "output_zero_scope": raw["output_zero_scope"],
        "portaudio_application_buffer_only": True,
        "portaudio_timestamp_authority": False,
        "hardware_sample_slip_authority": False,
        "physical_output_zero_authority": False,
        "electrical_output_zero_authority": False,
        "acoustic_output_zero_authority": False,
        "dither_off": True,
        "application_buffer_zero_submission_complete": True,
        "stream_close_returned_without_exception": True,
    }
    return _json_roundtrip({**core, "canonical_payload_sha256": payload_sha256(core)})


def capture_telemetry_to_contract(
    plan: Mapping[str, Any],
    capture_telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    """live primitive telemetry를 순수 계약 receipt로 변환한다.

    plan 자체가 exact-zero payload digest를 봉인하므로 동일 dtype/shape의 zero PCM을
    결정론적으로 재구성하고 plan을 먼저 재검증한다. 오디오 backend에는 접근하지 않는다.
    """

    preliminary = _exact_mapping(plan, _PLAN_FIELDS, label="zero duplex plan")
    frames = _exact_int(
        preliminary["frame_count"],
        label="plan.frame_count",
        minimum=1,
    )
    if frames % BLOCK_SIZE:
        raise ValueError("plan.frame_count는 block 배수여야 합니다")
    planned_pcm = np.zeros((frames, CHANNELS), dtype=PCM_DTYPE, order="C")
    planned_pcm.setflags(write=False)
    return validate_zero_duplex_telemetry(
        plan=preliminary,
        planned_pcm=planned_pcm,
        telemetry=capture_telemetry,
    )


def build_zero_duplex_receipt(
    *,
    plan: Mapping[str, Any],
    planned_pcm: np.ndarray,
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    """검증 성공을 transport smoke 이상으로 승격할 수 없는 receipt를 만든다."""

    checked_plan = validate_zero_duplex_plan(plan, planned_pcm)
    telemetry_receipt = validate_zero_duplex_telemetry(
        plan=checked_plan,
        planned_pcm=planned_pcm,
        telemetry=telemetry,
    )
    core: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": AUTHORITY_CEILING,
        "valid": True,
        "authority_ceiling": AUTHORITY_CEILING,
        "plan_payload_sha256": checked_plan["canonical_payload_sha256"],
        "planned_pcm_sha256": checked_plan["planned_pcm_sha256"],
        "zero_payload_sha256": checked_plan["zero_payload_sha256"],
        "telemetry_receipt": telemetry_receipt,
        "authority": dict(_AUTHORITY),
        "limitations": list(_LIMITATIONS),
    }
    receipt = _json_roundtrip({**core, "canonical_payload_sha256": payload_sha256(core)})
    return validate_zero_duplex_receipt(receipt)


def validate_zero_duplex_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """재서명된 권위 누수도 거부하도록 final receipt 의미를 재검증한다."""

    expected_keys = frozenset(
        {
            "schema",
            "status",
            "valid",
            "authority_ceiling",
            "plan_payload_sha256",
            "planned_pcm_sha256",
            "zero_payload_sha256",
            "telemetry_receipt",
            "authority",
            "limitations",
            "canonical_payload_sha256",
        }
    )
    checked = _exact_mapping(receipt, expected_keys, label="zero duplex receipt")
    if checked["schema"] != RECEIPT_SCHEMA:
        raise ValueError("zero duplex receipt schema가 다릅니다")
    if (
        checked["status"] != AUTHORITY_CEILING
        or checked["authority_ceiling"] != AUTHORITY_CEILING
    ):
        raise ValueError("zero duplex receipt가 transport smoke 이상으로 승격됐습니다")
    if _exact_bool(checked["valid"], label="receipt.valid") is not True:
        raise ValueError("zero duplex receipt valid는 exact true여야 합니다")
    for name in ("plan_payload_sha256", "planned_pcm_sha256", "zero_payload_sha256"):
        _exact_sha256(checked[name], label=f"receipt.{name}")
    authority = _exact_mapping(
        checked["authority"], frozenset(_AUTHORITY), label="receipt.authority"
    )
    for name, expected in _AUTHORITY.items():
        observed = _exact_bool(authority[name], label=f"receipt.authority.{name}")
        if observed is not expected:
            raise ValueError(f"receipt.authority.{name} 권위 누수입니다")
    if type(checked["limitations"]) is not list or checked["limitations"] != _LIMITATIONS:
        raise ValueError("zero duplex receipt limitations가 다릅니다")

    telemetry = _exact_mapping(
        checked["telemetry_receipt"],
        _TELEMETRY_RECEIPT_FIELDS,
        label="receipt.telemetry_receipt",
    )
    telemetry_digest = _exact_sha256(
        telemetry["canonical_payload_sha256"],
        label="receipt.telemetry_receipt.canonical_payload_sha256",
    )
    telemetry_core = {
        key: value
        for key, value in telemetry.items()
        if key != "canonical_payload_sha256"
    }
    if telemetry_digest != payload_sha256(telemetry_core):
        raise ValueError("telemetry receipt canonical SHA가 다릅니다")
    if telemetry["schema"] != TELEMETRY_RECEIPT_SCHEMA:
        raise ValueError("telemetry receipt schema가 다릅니다")
    for name, expected in (
        ("passed", True),
        ("all_submitted_buffers_bitwise_zero", True),
        ("host_watchdog_completed_within_deadline", True),
        ("portaudio_status_witness_pass", True),
        ("portaudio_application_buffer_only", True),
        ("dither_off", True),
        ("application_buffer_zero_submission_complete", True),
        ("stream_close_returned_without_exception", True),
        ("portaudio_timestamp_authority", False),
        ("hardware_sample_slip_authority", False),
        ("physical_output_zero_authority", False),
        ("electrical_output_zero_authority", False),
        ("acoustic_output_zero_authority", False),
    ):
        if _exact_bool(telemetry[name], label=f"receipt.telemetry_receipt.{name}") is not expected:
            raise ValueError(f"telemetry receipt {name} 의미가 다릅니다")
    if (
        telemetry["authority_ceiling"] != AUTHORITY_CEILING
        or telemetry["plan_payload_sha256"] != checked["plan_payload_sha256"]
        or telemetry["planned_pcm_sha256"] != checked["planned_pcm_sha256"]
        or telemetry["zero_payload_sha256"] != checked["zero_payload_sha256"]
    ):
        raise ValueError("telemetry receipt 의미 또는 parent 결속이 다릅니다")
    for name in (
        "plan_payload_sha256",
        "planned_pcm_sha256",
        "zero_payload_sha256",
        "actual_submitted_pcm_sha256",
        "actual_submitted_zero_payload_sha256",
    ):
        _exact_sha256(telemetry[name], label=f"receipt.telemetry_receipt.{name}")
    if telemetry["actual_submitted_pcm_sha256"] != telemetry["planned_pcm_sha256"]:
        raise ValueError("telemetry receipt actual/planned typed PCM SHA가 다릅니다")
    if (
        telemetry["actual_submitted_zero_payload_sha256"]
        != telemetry["zero_payload_sha256"]
    ):
        raise ValueError("telemetry receipt actual/planned zero payload SHA가 다릅니다")

    callback_digests = _exact_mapping(
        telemetry["callback_array_sha256"],
        _CALLBACK_ARRAY_FIELDS,
        label="receipt.telemetry_receipt.callback_array_sha256",
    )
    for name, digest in callback_digests.items():
        _exact_sha256(
            digest,
            label=f"receipt.telemetry_receipt.callback_array_sha256.{name}",
        )
    non_authoritative = _exact_mapping(
        telemetry["non_authoritative_observation"],
        frozenset(_NON_AUTHORITATIVE_OBSERVATION),
        label="receipt.telemetry_receipt.non_authoritative_observation",
    )
    for name, expected in _NON_AUTHORITATIVE_OBSERVATION.items():
        observed = non_authoritative[name]
        if type(expected) is bool:
            observed = _exact_bool(
                observed,
                label=f"receipt.telemetry_receipt.non_authoritative_observation.{name}",
            )
        if observed is not expected:
            raise ValueError(f"telemetry receipt {name} 비권위 계약이 다릅니다")
    for name in ("portaudio_status_present_count", "portaudio_xrun_count"):
        if _exact_int(telemetry[name], label=f"receipt.telemetry_receipt.{name}") != 0:
            raise ValueError(f"telemetry receipt {name}는 exact 0이어야 합니다")
    for name in ("resolved_input_device", "resolved_output_device"):
        _exact_int(telemetry[name], label=f"receipt.telemetry_receipt.{name}")
    callback_count = _exact_int(
        telemetry["callback_count"],
        label="receipt.telemetry_receipt.callback_count",
        minimum=1,
    )
    submitted_frames = _exact_int(
        telemetry["submitted_frames"],
        label="receipt.telemetry_receipt.submitted_frames",
        minimum=1,
    )
    captured_frames = _exact_int(
        telemetry["captured_frames"],
        label="receipt.telemetry_receipt.captured_frames",
        minimum=1,
    )
    if callback_count * BLOCK_SIZE != submitted_frames or captured_frames != submitted_frames:
        raise ValueError("telemetry receipt callback/frame accounting이 다릅니다")
    pre_started = _exact_float(
        telemetry["pre_open_monotonic_started"],
        label="receipt.telemetry_receipt.pre_open_monotonic_started",
        minimum=0.0,
    )
    pre_completed = _exact_float(
        telemetry["pre_open_monotonic_completed"],
        label="receipt.telemetry_receipt.pre_open_monotonic_completed",
        minimum=pre_started,
    )
    started = _exact_float(
        telemetry["capture_monotonic_started"],
        label="receipt.telemetry_receipt.capture_monotonic_started",
        minimum=pre_completed,
    )
    completed = _exact_float(
        telemetry["capture_monotonic_completed"],
        label="receipt.telemetry_receipt.capture_monotonic_completed",
        minimum=started,
    )
    elapsed = _exact_float(
        telemetry["capture_monotonic_elapsed_seconds"],
        label="receipt.telemetry_receipt.capture_monotonic_elapsed_seconds",
        minimum=0.0,
    )
    grace = _exact_float(
        telemetry["watchdog_grace_seconds"],
        label="receipt.telemetry_receipt.watchdog_grace_seconds",
        minimum=0.0,
    )
    if grace != WATCHDOG_GRACE_SECONDS or not math.isclose(
        completed - started, elapsed, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError("telemetry receipt watchdog timing 의미가 다릅니다")
    nominal = submitted_frames / SAMPLE_RATE_HZ
    if not (
        nominal - TIME_TOLERANCE_SECONDS
        <= elapsed
        <= nominal + WATCHDOG_GRACE_SECONDS + TIME_TOLERANCE_SECONDS
    ):
        raise ValueError("telemetry receipt watchdog deadline 의미가 다릅니다")
    if telemetry["callback_frame_semantics"] != (
        "software_accounting_only_not_hardware_slip_witness"
    ):
        raise ValueError("telemetry receipt callback 의미가 다릅니다")
    if telemetry["output_zero_scope"] != "portaudio_application_callback_buffer_only":
        raise ValueError("telemetry receipt output zero scope가 다릅니다")

    declared = _exact_sha256(
        checked["canonical_payload_sha256"], label="receipt.canonical_payload_sha256"
    )
    core = {
        key: value
        for key, value in checked.items()
        if key != "canonical_payload_sha256"
    }
    if declared != payload_sha256(core):
        raise ValueError("receipt canonical payload SHA가 다릅니다")
    return _json_roundtrip(checked)


__all__ = [
    "AUTHORITY_CEILING",
    "BLOCK_SIZE",
    "CHANNELS",
    "PCM_DTYPE",
    "PLAN_SCHEMA",
    "RECEIPT_SCHEMA",
    "SAMPLE_RATE_HZ",
    "TELEMETRY_RECEIPT_SCHEMA",
    "TELEMETRY_SCHEMA",
    "WATCHDOG_GRACE_SECONDS",
    "build_zero_duplex_plan",
    "build_zero_duplex_receipt",
    "capture_telemetry_to_contract",
    "canonical_json_bytes",
    "payload_sha256",
    "validate_zero_duplex_plan",
    "validate_zero_duplex_receipt",
    "validate_zero_duplex_telemetry",
]
