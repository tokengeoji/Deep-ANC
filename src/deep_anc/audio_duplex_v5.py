"""v5 측정용 전이중 callback primitive(장치 정책·publisher 제외)."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np


SAMPLE_RATE = 48_000
BLOCK_SIZE = 256
LATENCY = "low"
CHANNELS = (2, 2)
DUPLEX_TELEMETRY_SCHEMA = "fullband_causal_v5_duplex_telemetry_v3"
STATUS_INPUT_UNDERFLOW = 1 << 0
STATUS_INPUT_OVERFLOW = 1 << 1
STATUS_OUTPUT_UNDERFLOW = 1 << 2
STATUS_OUTPUT_OVERFLOW = 1 << 3
STATUS_PRIMING_OUTPUT = 1 << 4
STATUS_UNEXPECTED = 1 << 5
STATUS_PRESENT = 1 << 6
STATUS_XRUN_MASK = (
    STATUS_INPUT_UNDERFLOW
    | STATUS_INPUT_OVERFLOW
    | STATUS_OUTPUT_UNDERFLOW
    | STATUS_OUTPUT_OVERFLOW
)


def status_bitmask_v5(status: Any) -> np.uint32:
    """모든 callback status를 고정 bitmask로 만든다(무상태는 0)."""

    if not status:
        return np.uint32(0)
    mask = STATUS_PRESENT
    for name, bit in (
        ("input_underflow", STATUS_INPUT_UNDERFLOW),
        ("input_overflow", STATUS_INPUT_OVERFLOW),
        ("output_underflow", STATUS_OUTPUT_UNDERFLOW),
        ("output_overflow", STATUS_OUTPUT_OVERFLOW),
        ("priming_output", STATUS_PRIMING_OUTPUT),
    ):
        if bool(getattr(status, name, False)):
            mask |= bit
    if not mask & (STATUS_XRUN_MASK | STATUS_PRIMING_OUTPUT):
        mask |= STATUS_UNEXPECTED
    return np.uint32(mask)


@dataclass
class DuplexCaptureFailure(RuntimeError):
    message: str
    captured_pcm: np.ndarray
    submitted_pcm: np.ndarray
    capture_valid_mask: np.ndarray
    submitted_valid_mask: np.ndarray
    telemetry: dict[str, Any]

    def __str__(self) -> str:
        return self.message


def _tv(value: Any, name: str) -> float:
    raw = value.get(name, np.nan) if isinstance(value, dict) else getattr(
        value, name, np.nan
    )
    return float(raw)


def capture_duplex_v5(
    backend: Any,
    *,
    submitted_pcm: np.ndarray,
    input_device: int,
    output_device: int,
    pre_open_check: Callable[[], None] | None = None,
    watchdog_grace_seconds: float = 2.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[np.ndarray, dict[str, Any]]:
    planned = np.asarray(submitted_pcm)
    if (
        planned.dtype != np.dtype("<i2")
        or planned.ndim != 2
        or planned.shape[1] != 2
    ):
        raise ValueError("submitted_pcm은 exact <i2 [frames,2]여야 합니다")
    if len(planned) <= 0 or len(planned) % BLOCK_SIZE:
        raise ValueError("frame 수는 양수인 256의 배수여야 합니다")
    if type(input_device) is not int or input_device < 0:
        raise ValueError("input_device는 음이 아닌 exact int여야 합니다")
    if type(output_device) is not int or output_device < 0:
        raise ValueError("output_device는 음이 아닌 exact int여야 합니다")
    grace = float(watchdog_grace_seconds)
    if not np.isfinite(grace) or grace <= 0:
        raise ValueError("watchdog grace는 finite 양수여야 합니다")

    output = np.array(planned, dtype="<i2", copy=True, order="C")
    total = len(output)
    captured = np.zeros((total, 2), dtype="<i4")
    actual = np.zeros((total, 2), dtype="<i2")
    cap_valid = np.zeros(total, dtype=np.bool_)
    out_valid = np.zeros(total, dtype=np.bool_)
    cursor = 0
    completed = False
    callback_error = None
    invalid: list[str] = []
    rows: list[tuple[Any, ...]] = []
    last_times = None
    capture_monotonic_started = float(monotonic())
    if not np.isfinite(capture_monotonic_started):
        raise ValueError("capture monotonic start는 finite여야 합니다")

    def callback(indata, outdata, frames, time_info, status):  # noqa: ANN001
        nonlocal cursor, completed, callback_error, last_times
        sink_valid = False
        try:
            count = int(frames)
            source = np.asarray(indata)
            sink = np.asarray(outdata)
            if sink.dtype != np.dtype("<i2") or sink.ndim != 2 or sink.shape[1] != 2:
                raise ValueError("callback output은 exact <i2 [frames,2]여야 합니다")
            sink_valid = True
            if count != BLOCK_SIZE:
                raise ValueError("callback frames는 exact 256이어야 합니다")
            if sink.shape != (256, 2):
                raise ValueError("callback output은 exact <i2 [256,2]여야 합니다")
            if source.dtype != np.dtype("<i4") or source.shape != (256, 2):
                raise ValueError("callback input은 exact <i4 [256,2]여야 합니다")
            if cursor + 256 > total:
                raise ValueError("callback이 planned output 경계를 넘습니다")
            times = tuple(
                _tv(time_info, name)
                for name in (
                    "inputBufferAdcTime",
                    "outputBufferDacTime",
                    "currentTime",
                )
            )
            if not all(np.isfinite(value) for value in times):
                raise ValueError("callback timestamp가 finite가 아닙니다")
            if last_times is not None and any(
                current <= previous
                for current, previous in zip(times, last_times)
            ):
                raise ValueError("callback timestamp가 strict-monotonic이 아닙니다")
            mask = int(status_bitmask_v5(status))
            block = output[cursor : cursor + 256].copy()
            input_block = source.copy()
            # sink가 실패하면 evidence/cursor는 아직 갱신되지 않는다.
            outdata[...] = block
            captured[cursor : cursor + 256] = input_block
            actual[cursor : cursor + 256] = block
            cap_valid[cursor : cursor + 256] = True
            out_valid[cursor : cursor + 256] = True
            rows.append((len(rows), cursor, count, *times, mask))
            last_times = times
            cursor += 256
            if mask & STATUS_XRUN_MASK:
                invalid.append("portaudio_xrun_status")
            if mask & STATUS_UNEXPECTED:
                invalid.append("unexpected_portaudio_status")
            if mask & STATUS_PRIMING_OUTPUT:
                invalid.append("priming_callback_forbidden")
            if cursor == total:
                completed = True
                raise backend.CallbackStop
        except backend.CallbackStop:
            raise
        except Exception as exc:
            silence_error = None
            silence_confirmed = False
            if sink_valid:
                try:
                    outdata[...] = np.zeros_like(sink)
                    silence_confirmed = bool(np.all(np.asarray(outdata) == 0))
                except Exception as zero_exc:
                    silence_error = f"{type(zero_exc).__name__}: {zero_exc}"
            if not silence_confirmed:
                invalid.append("output_silence_not_confirmed_on_callback_failure")
            callback_error = f"{type(exc).__name__}: {exc}"
            if silence_error is not None:
                callback_error += f"; zero_fill={silence_error}"
            raise backend.CallbackAbort

    stream = None
    failure = None
    stop_error = None
    abort_error = None
    close_error = None
    try:
        if pre_open_check:
            pre_open_check()
        stream = backend.Stream(
            samplerate=48_000,
            blocksize=256,
            device=(input_device, output_device),
            channels=(2, 2),
            dtype=("int32", "int16"),
            latency=("low", "low"),
            callback=callback,
            prime_output_buffers_using_stream_callback=False,
        )
        stream.start()
        deadline = capture_monotonic_started + total / 48_000 + grace
        while not completed:
            if callback_error:
                raise RuntimeError(f"오디오 callback 실패: {callback_error}")
            if monotonic() >= deadline:
                raise TimeoutError("v5 duplex capture watchdog 초과")
            sleep(0.001)
        if invalid:
            failure = RuntimeError(
                "callback canonical-invalid: "
                + ",".join(dict.fromkeys(invalid))
            )
        else:
            try:
                stream.stop(ignore_errors=False)
            except BaseException as exc:
                stop_error = exc
    except BaseException as exc:
        failure = exc
    if stream is not None:
        if failure is not None or stop_error is not None:
            try:
                stream.abort(ignore_errors=False)
            except BaseException as exc:
                abort_error = exc
        try:
            stream.close(ignore_errors=False)
        except BaseException as exc:
            close_error = exc

    capture_monotonic_completed = float(monotonic())
    if not np.isfinite(capture_monotonic_completed):
        failure = failure or ValueError("capture monotonic completion은 finite여야 합니다")
    capture_monotonic_elapsed = capture_monotonic_completed - capture_monotonic_started
    if not np.isfinite(capture_monotonic_elapsed) or capture_monotonic_elapsed < 0.0:
        failure = failure or ValueError("capture monotonic elapsed는 finite nonnegative여야 합니다")

    def col(index: int, dtype: str) -> np.ndarray:
        return np.asarray([row[index] for row in rows], dtype=dtype)

    masks = col(6, "<u4")
    telemetry = {
        "schema": DUPLEX_TELEMETRY_SCHEMA,
        "callback_frame_semantics": (
            "software_accounting_only_not_hardware_slip_witness"
        ),
        "portaudio_xrun_status_witness": True,
        "hardware_sample_slip_authority": False,
        "watchdog_coverage": (
            "host_wait_until_planned_frames_plus_grace_not_hardware_deadline_witness"
        ),
        "sample_rate_hz": 48_000,
        "block_size": 256,
        "latency": "low",
        "channels": [2, 2],
        "resolved_input_device": input_device,
        "resolved_output_device": output_device,
        "capture_monotonic_started": capture_monotonic_started,
        "capture_monotonic_completed": capture_monotonic_completed,
        "capture_monotonic_elapsed_seconds": capture_monotonic_elapsed,
        "watchdog_grace_seconds": grace,
        "input_dtype": "<i4",
        "output_dtype": "<i2",
        "callback_sequence": col(0, "<i8"),
        "callback_start_frames": col(1, "<i8"),
        "callback_frame_counts": col(2, "<i8"),
        "input_buffer_adc_time": col(3, "<f8"),
        "output_buffer_dac_time": col(4, "<f8"),
        "callback_current_time": col(5, "<f8"),
        "callback_status_bitmask": masks,
        "xrun_count": int(np.count_nonzero(masks & STATUS_XRUN_MASK)),
        "status_present_count": int(np.count_nonzero(masks & STATUS_PRESENT)),
        "captured_frames": int(np.count_nonzero(cap_valid)),
        "submitted_frames": int(np.count_nonzero(out_valid)),
        "completed": completed,
        "callback_error": callback_error,
        "canonical_invalid_reasons": list(dict.fromkeys(invalid)),
        "stream_stop_error": (
            None if stop_error is None else f"{type(stop_error).__name__}: {stop_error}"
        ),
        "stream_abort_error": (
            None
            if abort_error is None
            else f"{type(abort_error).__name__}: {abort_error}"
        ),
        "stream_close_error": (
            None
            if close_error is None
            else f"{type(close_error).__name__}: {close_error}"
        ),
        "normal_stop_completed": bool(
            completed and failure is None and stop_error is None
        ),
        "output_stop_confirmed": bool(stream is None or close_error is None),
    }
    errors = [
        error
        for error in (failure, stop_error, abort_error, close_error)
        if error is not None
    ]
    if errors:
        raise DuplexCaptureFailure(
            "; ".join(f"{type(error).__name__}: {error}" for error in errors),
            captured.copy(),
            actual.copy(),
            cap_valid.copy(),
            out_valid.copy(),
            telemetry,
        )
    return captured, {
        **telemetry,
        "actual_submitted_pcm": actual,
        "capture_valid_mask": cap_valid,
        "submitted_valid_mask": out_valid,
    }
