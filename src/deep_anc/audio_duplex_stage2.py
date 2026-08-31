"""Stage-2 2 kHz 전용 duplex transport wrapper.

검증된 v5 callback primitive를 재사용하지만 telemetry schema는 분리한다. 이 모듈은
``sounddevice``를 import하거나 장치를 직접 해석하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Callable

import numpy as np

from .audio_duplex_v5 import (
    BLOCK_SIZE,
    CHANNELS,
    LATENCY,
    SAMPLE_RATE,
    STATUS_PRESENT,
    STATUS_XRUN_MASK,
    DuplexCaptureFailure,
    _capture_duplex,
    status_bitmask_v5,
)
from deep_anc.dsp.measurement_level import scoped_live_audio_signal_handlers


DUPLEX_TELEMETRY_SCHEMA = "stage2_2khz_duplex_telemetry_v1"
OUTPUT_MASTER_TELEMETRY_SCHEMA = "stage2_2khz_output_master_split_telemetry_v1"


@dataclass
class OutputMasterCaptureFailure(RuntimeError):
    """독립 input/output stream 실패의 두 clock-domain partial raw.

    ``captured_pcm``/``capture_valid_mask``는 input clock 축이고,
    ``submitted_pcm``/``submitted_valid_mask``는 output clock 축이다. 두 길이가 같다는
    주장은 금지한다. publisher는 이 객체를 그대로 보존한 뒤 별도 clock 분석 전에는
    두 배열을 sample-aligned data로 해석하면 안 된다.
    """

    message: str
    captured_pcm: np.ndarray
    submitted_pcm: np.ndarray
    capture_valid_mask: np.ndarray
    submitted_valid_mask: np.ndarray
    telemetry: dict[str, Any]

    def __str__(self) -> str:
        return self.message


def _time_value(value: Any, name: str) -> float:
    raw = value.get(name, np.nan) if isinstance(value, dict) else getattr(
        value, name, np.nan
    )
    return float(raw)


def _validate_output_master_arguments(
    submitted_pcm: np.ndarray,
    *,
    input_device: int,
    output_device: int,
    pre_roll_frames: int,
    post_roll_frames: int,
    watchdog_grace_seconds: float,
) -> tuple[np.ndarray, float]:
    planned = np.asarray(submitted_pcm)
    if (
        planned.dtype != np.dtype("<i2")
        or planned.ndim != 2
        or planned.shape[1] != 2
    ):
        raise ValueError("submitted_pcm은 exact <i2 [frames,2]여야 합니다")
    if len(planned) <= 0 or len(planned) % BLOCK_SIZE:
        raise ValueError("output frame 수는 양수인 256의 배수여야 합니다")
    if type(input_device) is not int or input_device < 0:
        raise ValueError("input_device는 음이 아닌 exact int여야 합니다")
    if type(output_device) is not int or output_device < 0:
        raise ValueError("output_device는 음이 아닌 exact int여야 합니다")
    for name, frames in (
        ("pre_roll_frames", pre_roll_frames),
        ("post_roll_frames", post_roll_frames),
    ):
        if type(frames) is not int or frames < BLOCK_SIZE or frames % BLOCK_SIZE:
            raise ValueError(f"{name}는 256 이상의 exact 256 배수여야 합니다")
    grace = float(watchdog_grace_seconds)
    if not math.isfinite(grace) or grace <= 0.0:
        raise ValueError("watchdog grace는 finite 양수여야 합니다")
    return np.array(planned, dtype="<i2", copy=True, order="C"), grace


def capture_output_master_stage2(
    backend: Any,
    *,
    submitted_pcm: np.ndarray,
    input_device: int,
    output_device: int,
    pre_roll_frames: int = 4_096,
    post_roll_frames: int = 4_096,
    pre_open_check: Callable[[], None] | None = None,
    watchdog_grace_seconds: float = 2.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    on_output_closed: Callable[[bool], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """APE input 선시작 + AB13X output-master split capture.

    InputStream callback은 raw witness만 append하며 output cursor를 전진시키지 않는다.
    OutputStream callback만 canonical submitted cursor를 소유한다. clean pre-roll 뒤에
    output을 열고, exact output 종료/close 뒤에도 input post-roll을 보존한다. 서로 다른
    ADC/DAC timestamp를 비교하거나 두 clock의 frame identity를 주장하지 않는다.

    이 함수는 backend를 주입받으므로 import 시 장치를 열지 않는다. callback/status/close
    실패는 :class:`OutputMasterCaptureFailure`에 두 clock-domain prefix와 telemetry를 함께
    담아 fail-closed한다.
    """

    output, grace = _validate_output_master_arguments(
        submitted_pcm,
        input_device=input_device,
        output_device=output_device,
        pre_roll_frames=pre_roll_frames,
        post_roll_frames=post_roll_frames,
        watchdog_grace_seconds=watchdog_grace_seconds,
    )
    total_output_frames = len(output)
    actual_output = np.zeros_like(output)
    submitted_valid = np.zeros(total_output_frames, dtype=np.bool_)
    input_blocks: list[np.ndarray] = []
    input_rows: list[tuple[Any, ...]] = []
    output_rows: list[tuple[Any, ...]] = []
    failure_events: list[dict[str, Any]] = []
    state_lock = threading.Lock()
    pre_roll_ready = threading.Event()
    output_complete = threading.Event()
    post_roll_ready = threading.Event()
    failure_ready = threading.Event()
    input_cursor = 0
    output_cursor = 0
    input_cursor_at_output_start: int | None = None
    input_cursor_at_output_complete: int | None = None
    last_input_times: tuple[float, float] | None = None
    last_output_times: tuple[float, float] | None = None

    def fail(role: str, message: str, **evidence: Any) -> None:
        with state_lock:
            if not failure_events:
                failure_events.append(
                    {"role": str(role), "message": str(message), **dict(evidence)}
                )
        failure_ready.set()

    def input_callback(indata, frames, time_info, status):  # noqa: ANN001
        nonlocal input_cursor, last_input_times
        try:
            count = int(frames)
            source = np.asarray(indata)
            if count != BLOCK_SIZE:
                raise ValueError("input callback frames는 exact 256이어야 합니다")
            if source.dtype != np.dtype("<i4") or source.shape != (BLOCK_SIZE, 2):
                raise ValueError("input callback은 exact <i4 [256,2]여야 합니다")
            adc_time = _time_value(time_info, "inputBufferAdcTime")
            current_time = _time_value(time_info, "currentTime")
            times = (adc_time, current_time)
            if not all(math.isfinite(value) for value in times):
                raise ValueError("input callback timestamp가 finite가 아닙니다")
            if last_input_times is not None and any(
                current <= previous
                for current, previous in zip(times, last_input_times)
            ):
                raise ValueError("input callback timestamp가 strict-monotonic이 아닙니다")
            mask = int(status_bitmask_v5(status))
            if mask != 0:
                raise ValueError(f"input callback status가 0이 아닙니다: {mask}")
            block = np.array(source, dtype="<i4", copy=True, order="C")
            stop_input = False
            with state_lock:
                if failure_events:
                    raise RuntimeError("선행 split transport failure가 있습니다")
                start = input_cursor
                input_blocks.append(block)
                input_rows.append(
                    (len(input_rows), start, count, adc_time, current_time, mask)
                )
                input_cursor += count
                last_input_times = times
                if input_cursor >= pre_roll_frames:
                    pre_roll_ready.set()
                if input_cursor_at_output_complete is not None and (
                    input_cursor
                    >= input_cursor_at_output_complete + post_roll_frames
                ):
                    post_roll_ready.set()
                    stop_input = True
            if stop_input:
                raise backend.CallbackStop
        except backend.CallbackStop:
            raise
        except BaseException as exc:
            fail("input", f"{type(exc).__name__}: {exc}")
            raise backend.CallbackAbort

    def output_callback(outdata, frames, time_info, status):  # noqa: ANN001
        nonlocal output_cursor, input_cursor_at_output_start
        nonlocal input_cursor_at_output_complete, last_output_times
        sink = np.asarray(outdata)
        sink_valid = bool(
            sink.dtype == np.dtype("<i2") and sink.shape == (BLOCK_SIZE, 2)
        )
        if sink_valid:
            sink.fill(0)
        try:
            count = int(frames)
            if not sink_valid:
                raise ValueError("output callback은 exact <i2 [256,2]여야 합니다")
            if count != BLOCK_SIZE:
                raise ValueError("output callback frames는 exact 256이어야 합니다")
            dac_time = _time_value(time_info, "outputBufferDacTime")
            current_time = _time_value(time_info, "currentTime")
            times = (dac_time, current_time)
            if not all(math.isfinite(value) for value in times):
                raise ValueError("output callback timestamp가 finite가 아닙니다")
            if last_output_times is not None and any(
                current <= previous
                for current, previous in zip(times, last_output_times)
            ):
                raise ValueError("output callback timestamp가 strict-monotonic이 아닙니다")
            mask = int(status_bitmask_v5(status))
            if mask != 0:
                raise ValueError(f"output callback status가 0이 아닙니다: {mask}")
            with state_lock:
                if failure_events:
                    raise RuntimeError("선행 split transport failure가 있습니다")
                start = output_cursor
                stop = start + BLOCK_SIZE
                if stop > total_output_frames:
                    raise ValueError("output callback이 submitted 경계를 넘습니다")
                if input_cursor_at_output_start is None:
                    if input_cursor < pre_roll_frames:
                        raise ValueError("clean input pre-roll 전에 output callback이 실행됐습니다")
                    input_cursor_at_output_start = input_cursor
                block = output[start:stop].copy()
                sink[:, :] = block
                actual_output[start:stop] = block
                submitted_valid[start:stop] = True
                output_rows.append(
                    (len(output_rows), start, count, dac_time, current_time, mask)
                )
                output_cursor = stop
                last_output_times = times
                if output_cursor == total_output_frames:
                    input_cursor_at_output_complete = input_cursor
                    output_complete.set()
                    if post_roll_frames == 0:
                        post_roll_ready.set()
                    raise backend.CallbackStop
        except backend.CallbackStop:
            raise
        except BaseException as exc:
            if sink_valid:
                sink.fill(0)
            fail("output", f"{type(exc).__name__}: {exc}")
            raise backend.CallbackAbort

    def wait_for(event: threading.Event, *, seconds: float, label: str) -> None:
        deadline = float(monotonic()) + float(seconds) + grace
        while not event.is_set():
            if failure_ready.is_set():
                return
            if float(monotonic()) >= deadline:
                fail("watchdog", f"{label} watchdog 초과")
                return
            sleep(0.001)

    def rows_array(rows: list[tuple[Any, ...]], index: int, dtype: str) -> np.ndarray:
        return np.asarray([row[index] for row in rows], dtype=dtype)

    pre_open_started = float(monotonic())
    pre_open_completed = pre_open_started
    capture_started = pre_open_started
    capture_completed = pre_open_started
    input_stream = None
    output_stream = None
    output_closed_notified = False
    output_stop_error: BaseException | None = None
    output_abort_error: BaseException | None = None
    output_close_error: BaseException | None = None
    output_closed = False
    input_stop_error: BaseException | None = None
    input_abort_error: BaseException | None = None
    input_close_error: BaseException | None = None
    outer_error: BaseException | None = None

    def notify_output_closed() -> None:
        nonlocal output_closed_notified, outer_error
        if output_closed_notified or on_output_closed is None:
            output_closed_notified = True
            return
        output_closed_notified = True
        try:
            on_output_closed(bool(output_stream is None or output_close_error is None))
        except BaseException as exc:
            outer_error = outer_error or exc
            fail("output_close_notification", f"{type(exc).__name__}: {exc}")

    signal_scope = scoped_live_audio_signal_handlers()
    signal_scope.__enter__()
    try:
        try:
            if pre_open_check is not None:
                pre_open_check()
            pre_open_completed = float(monotonic())
            if (
                not math.isfinite(pre_open_completed)
                or pre_open_completed < pre_open_started
            ):
                raise ValueError("pre-open completion monotonic이 유효하지 않습니다")
            input_stream = backend.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                device=input_device,
                channels=2,
                dtype="int32",
                latency=LATENCY,
                callback=input_callback,
            )
            capture_started = float(monotonic())
            input_stream.start()
            wait_for(
                pre_roll_ready,
                seconds=pre_roll_frames / SAMPLE_RATE,
                label="input pre-roll",
            )
            if failure_ready.is_set():
                raise RuntimeError(failure_events[0]["message"])
            output_stream = backend.OutputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                device=output_device,
                channels=2,
                dtype="int16",
                latency=LATENCY,
                callback=output_callback,
                prime_output_buffers_using_stream_callback=False,
            )
            output_stream.start()
            wait_for(
                output_complete,
                seconds=total_output_frames / SAMPLE_RATE,
                label="canonical output",
            )
            if failure_ready.is_set():
                raise RuntimeError(failure_events[0]["message"])
            try:
                output_stream.stop(ignore_errors=False)
            except BaseException as exc:
                output_stop_error = exc
                raise
            try:
                output_stream.close(ignore_errors=False)
                output_closed = True
            except BaseException as exc:
                output_close_error = exc
                raise
            finally:
                notify_output_closed()
            wait_for(
                post_roll_ready,
                seconds=post_roll_frames / SAMPLE_RATE,
                label="input post-roll",
            )
            if failure_ready.is_set():
                raise RuntimeError(failure_events[0]["message"])
            try:
                input_stream.stop(ignore_errors=False)
            except BaseException as exc:
                input_stop_error = exc
                raise
        except BaseException as exc:
            outer_error = exc
            fail("host", f"{type(exc).__name__}: {exc}")
    finally:
        if output_stream is not None and not output_closed and (
            outer_error is not None or output_stop_error is not None
        ):
            try:
                output_stream.abort(ignore_errors=False)
            except BaseException as exc:
                output_abort_error = exc
        if output_stream is not None and not output_closed and output_close_error is None:
            try:
                output_stream.close(ignore_errors=False)
                output_closed = True
            except BaseException as exc:
                output_close_error = exc
        notify_output_closed()
        if input_stream is not None and (
            outer_error is not None or input_stop_error is not None
        ):
            try:
                input_stream.abort(ignore_errors=False)
            except BaseException as exc:
                input_abort_error = exc
        if input_stream is not None:
            try:
                input_stream.close(ignore_errors=False)
            except BaseException as exc:
                input_close_error = exc
        capture_completed = float(monotonic())
        signal_scope.__exit__(None, None, None)

    captured = (
        np.ascontiguousarray(np.concatenate(input_blocks, axis=0), dtype="<i4")
        if input_blocks
        else np.zeros((0, 2), dtype="<i4")
    )
    capture_valid = np.ones(len(captured), dtype=np.bool_)
    with state_lock:
        start_marker = input_cursor_at_output_start
        complete_marker = input_cursor_at_output_complete
        failure_snapshot = [dict(value) for value in failure_events]
        input_frame_count = input_cursor
        output_frame_count = output_cursor
    if output_stop_error is not None:
        failure_snapshot.append(
            {"role": "output_stop", "message": str(output_stop_error)}
        )
    if output_abort_error is not None:
        failure_snapshot.append(
            {"role": "output_abort", "message": str(output_abort_error)}
        )
    if output_close_error is not None:
        failure_snapshot.append(
            {"role": "output_close", "message": str(output_close_error)}
        )
    if input_stop_error is not None:
        failure_snapshot.append(
            {"role": "input_stop", "message": str(input_stop_error)}
        )
    if input_abort_error is not None:
        failure_snapshot.append(
            {"role": "input_abort", "message": str(input_abort_error)}
        )
    if input_close_error is not None:
        failure_snapshot.append(
            {"role": "input_close", "message": str(input_close_error)}
        )
    pre_observed = 0 if start_marker is None else int(start_marker)
    post_observed = (
        0
        if complete_marker is None
        else max(0, int(input_frame_count - complete_marker))
    )
    completed = bool(
        not failure_snapshot
        and output_frame_count == total_output_frames
        and bool(np.all(submitted_valid))
        and pre_observed >= pre_roll_frames
        and post_observed >= post_roll_frames
        and output_complete.is_set()
        and post_roll_ready.is_set()
    )
    telemetry: dict[str, Any] = {
        "schema": OUTPUT_MASTER_TELEMETRY_SCHEMA,
        "transport": "independent_input_output_streams_output_clock_master",
        "sample_rate_hz": SAMPLE_RATE,
        "block_size": BLOCK_SIZE,
        "input_dtype": "<i4",
        "output_dtype": "<i2",
        "resolved_input_device": input_device,
        "resolved_output_device": output_device,
        "output_clock_owner": "outputstream_callback_only",
        "input_role": "raw_witness_only_never_output_pacing",
        "cross_clock_timestamp_alignment_used": False,
        "input_output_frame_identity_claimed": False,
        "hardware_sample_slip_authority": False,
        "input_stream_started_before_output_stream": bool(start_marker is not None),
        "pre_roll_requested_frames": pre_roll_frames,
        "pre_roll_observed_input_frames": pre_observed,
        "input_frame_cursor_at_output_start": start_marker,
        "input_frame_cursor_at_output_complete": complete_marker,
        "post_roll_requested_frames": post_roll_frames,
        "post_roll_observed_input_frames": post_observed,
        "captured_input_frames": len(captured),
        "canonical_output_frames": total_output_frames,
        "submitted_output_frames": int(np.count_nonzero(submitted_valid)),
        "completed": completed,
        "normal_stop_completed": completed,
        "output_stop_confirmed": bool(output_stream is None or output_close_error is None),
        "pre_open_monotonic_started": pre_open_started,
        "pre_open_monotonic_completed": pre_open_completed,
        "capture_monotonic_started": capture_started,
        "capture_monotonic_completed": capture_completed,
        "capture_monotonic_elapsed_seconds": capture_completed - capture_started,
        "watchdog_grace_seconds": grace,
        "input_callback_sequence": rows_array(input_rows, 0, "<i8"),
        "input_callback_start_frames": rows_array(input_rows, 1, "<i8"),
        "input_callback_frame_counts": rows_array(input_rows, 2, "<i8"),
        "input_buffer_adc_time": rows_array(input_rows, 3, "<f8"),
        "input_callback_current_time": rows_array(input_rows, 4, "<f8"),
        "input_callback_status_bitmask": rows_array(input_rows, 5, "<u4"),
        "output_callback_sequence": rows_array(output_rows, 0, "<i8"),
        "output_callback_start_frames": rows_array(output_rows, 1, "<i8"),
        "output_callback_frame_counts": rows_array(output_rows, 2, "<i8"),
        "output_buffer_dac_time": rows_array(output_rows, 3, "<f8"),
        "output_callback_current_time": rows_array(output_rows, 4, "<f8"),
        "output_callback_status_bitmask": rows_array(output_rows, 5, "<u4"),
        "actual_submitted_pcm": actual_output.copy(),
        "capture_valid_mask": capture_valid,
        "submitted_valid_mask": submitted_valid.copy(),
        "failure_events": failure_snapshot,
        "legacy_combined_duplex_used": False,
    }
    if not math.isfinite(telemetry["capture_monotonic_elapsed_seconds"]) or telemetry[
        "capture_monotonic_elapsed_seconds"
    ] < 0.0:
        failure_snapshot.append(
            {"role": "host_clock", "message": "capture monotonic elapsed가 유효하지 않습니다"}
        )
        telemetry["failure_events"] = failure_snapshot
        telemetry["completed"] = False
        telemetry["normal_stop_completed"] = False
    if telemetry["completed"] is not True:
        message = "; ".join(
            f"{row['role']}: {row['message']}" for row in failure_snapshot
        ) or "output-master split transport가 완결되지 않았습니다"
        raise OutputMasterCaptureFailure(
            message,
            captured.copy(),
            actual_output.copy(),
            capture_valid.copy(),
            submitted_valid.copy(),
            telemetry,
        )
    return captured, telemetry


def capture_duplex_stage2(
    backend: Any,
    *,
    submitted_pcm: np.ndarray,
    input_device: int,
    output_device: int,
    pre_open_check: Callable[[], None] | None = None,
    watchdog_grace_seconds: float = 2.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    on_output_closed: Callable[[bool], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """레거시 combined diagnostic 호환 경로.

    exact submitted bytes 보존에는 쓸 수 있지만 서로 다른 USB DAC/APE clock의 frame
    identity나 phase authority를 만들 수 없다. 신규 strict capture는
    :func:`capture_output_master_stage2`를 사용해야 한다.
    """

    return _capture_duplex(
        backend,
        submitted_pcm=submitted_pcm,
        input_device=input_device,
        output_device=output_device,
        telemetry_schema=DUPLEX_TELEMETRY_SCHEMA,
        include_pre_open_telemetry=True,
        pre_open_check=pre_open_check,
        watchdog_grace_seconds=watchdog_grace_seconds,
        monotonic=monotonic,
        sleep=sleep,
        on_output_closed=on_output_closed,
    )


__all__ = [
    "BLOCK_SIZE",
    "CHANNELS",
    "DUPLEX_TELEMETRY_SCHEMA",
    "DuplexCaptureFailure",
    "LATENCY",
    "OUTPUT_MASTER_TELEMETRY_SCHEMA",
    "OutputMasterCaptureFailure",
    "SAMPLE_RATE",
    "STATUS_PRESENT",
    "STATUS_XRUN_MASK",
    "capture_duplex_stage2",
    "capture_output_master_stage2",
    "status_bitmask_v5",
]
