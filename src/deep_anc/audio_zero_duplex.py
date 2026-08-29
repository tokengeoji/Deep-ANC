"""RT5640 shared-rate 후보의 exact-zero 전이중 transport primitive.

이 모듈은 sounddevice를 import하거나 장치를 해석하지 않는다. 호출자가 전달한 backend로
입력 S32_LE와 출력 S32_LE를 동시에 열되, callback은 어떤 신호·모델·입력 monitoring도
받지 않고 bitwise zero만 제출한다. 결과는 shared-clock, sample identity, P/S 또는 ANC
권한이 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass
import signal
import time
from typing import Any, Callable

import numpy as np

from deep_anc.audio_duplex_v5 import (
    STATUS_PRESENT,
    STATUS_PRIMING_OUTPUT,
    STATUS_UNEXPECTED,
    STATUS_XRUN_MASK,
    status_bitmask_v5,
)


SAMPLE_RATE = 48_000
BLOCK_SIZE = 256
CHANNELS = (2, 2)
INPUT_DTYPE = "int32"
OUTPUT_DTYPE = "int32"
LATENCY = "low"
ZERO_DUPLEX_TELEMETRY_SCHEMA = "rt5640_zero_duplex_telemetry_v1"


class _DeferredLiveAudioSignals:
    """INT/TERM/HUP를 stream 정리가 끝날 때까지 기록만 한다.

    Python signal handler 안에서 예외를 던지면 ``stop``/``abort``/``close`` 자체가
    중간에 끊길 수 있다. 이 scope는 handler에서는 signum만 보존하고 호출자가 모든
    cleanup을 끝낸 뒤 실패로 승격하게 한다.
    """

    def __init__(self) -> None:
        self.received: list[int] = []
        self.restore_errors: list[BaseException] = []
        self._previous: dict[signal.Signals, Any] = {}
        self._installed: list[signal.Signals] = []

    def __enter__(self) -> _DeferredLiveAudioSignals:
        watched = tuple(
            item
            for item in (
                getattr(signal, "SIGINT", None),
                getattr(signal, "SIGTERM", None),
                getattr(signal, "SIGHUP", None),
            )
            if item is not None
        )
        self._previous = {item: signal.getsignal(item) for item in watched}

        def defer(signum, _frame):  # noqa: ANN001
            self.received.append(int(signum))

        try:
            for item in watched:
                signal.signal(item, defer)
                self._installed.append(item)
        except BaseException:
            for item in reversed(self._installed):
                signal.signal(item, self._previous[item])
            raise
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> bool:  # noqa: ANN001
        for item in reversed(self._installed):
            try:
                signal.signal(item, self._previous[item])
            except BaseException as error:
                self.restore_errors.append(error)
        return False


class _DeferredLiveAudioTermination(RuntimeError):
    """정리 완료 뒤 receipt에 남기는 POSIX 종료 의미."""

    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        self.exit_code = 128 + self.signum
        super().__init__(
            f"deferred live audio signal {self.signum} (exit {self.exit_code})"
        )


@dataclass
class ZeroDuplexCaptureFailure(RuntimeError):
    """실패 시에도 callback이 관측한 partial raw와 zero 제출 증거를 보존한다."""

    message: str
    captured_pcm: np.ndarray
    actual_submitted_pcm: np.ndarray
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


def capture_zero_duplex(
    backend: Any,
    *,
    total_frames: int,
    input_device: int,
    output_device: int,
    pre_open_check: Callable[[], None] | None = None,
    on_stream_started: Callable[[], None] | None = None,
    on_output_closed: Callable[[bool], None] | None = None,
    watchdog_grace_seconds: float = 2.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[np.ndarray, dict[str, Any]]:
    """정확히 ``total_frames`` 동안 S32_LE zero만 출력하며 2채널 입력을 보존한다."""

    if type(total_frames) is not int or total_frames <= 0:
        raise ValueError("total_frames는 양의 exact int여야 합니다")
    if total_frames % BLOCK_SIZE:
        raise ValueError("total_frames는 256의 배수여야 합니다")
    if type(input_device) is not int or input_device < 0:
        raise ValueError("input_device는 음이 아닌 exact int여야 합니다")
    if type(output_device) is not int or output_device < 0:
        raise ValueError("output_device는 음이 아닌 exact int여야 합니다")
    grace = float(watchdog_grace_seconds)
    if not np.isfinite(grace) or grace <= 0.0:
        raise ValueError("watchdog grace는 finite 양수여야 합니다")
    for name, callback in (
        ("pre_open_check", pre_open_check),
        ("on_stream_started", on_stream_started),
        ("on_output_closed", on_output_closed),
    ):
        if callback is not None and not callable(callback):
            raise TypeError(f"{name}는 callable 또는 None이어야 합니다")

    captured = np.zeros((total_frames, 2), dtype="<i4")
    actual = np.zeros((total_frames, 2), dtype="<i4")
    capture_valid = np.zeros(total_frames, dtype=np.bool_)
    submitted_valid = np.zeros(total_frames, dtype=np.bool_)
    cursor = 0
    completed = False
    callback_error: str | None = None
    invalid: list[str] = []
    rows: list[tuple[Any, ...]] = []
    last_times: tuple[float, float, float] | None = None
    callback_zero_attempt_count = 0
    callback_zero_confirmed_count = 0

    pre_open_started = float(monotonic())
    if not np.isfinite(pre_open_started):
        raise ValueError("pre-open monotonic start는 finite여야 합니다")
    pre_open_completed = pre_open_started
    capture_started: float | None = None

    def callback(indata, outdata, frames, time_info, status):  # noqa: ANN001
        nonlocal cursor, completed, callback_error, last_times
        nonlocal callback_zero_attempt_count, callback_zero_confirmed_count
        zero_confirmed = False
        try:
            # callback의 첫 부작용은 무조건 전체 output buffer zero-fill이다. 입력, frame,
            # timestamp 또는 status가 잘못돼도 nonzero가 backend로 넘어갈 경로를 두지 않는다.
            try:
                outdata[...] = 0
            finally:
                callback_zero_attempt_count += 1
            sink = np.asarray(outdata)
            zero_confirmed = bool(np.count_nonzero(sink) == 0)
            if zero_confirmed:
                callback_zero_confirmed_count += 1
            else:
                raise RuntimeError("callback output zero-fill을 확인하지 못했습니다")

            count = int(frames)
            source = np.asarray(indata)
            if count != BLOCK_SIZE:
                raise ValueError("callback frames는 exact 256이어야 합니다")
            if sink.dtype != np.dtype("<i4") or sink.shape != (BLOCK_SIZE, 2):
                raise ValueError("callback output은 exact <i4 [256,2]여야 합니다")
            if source.dtype != np.dtype("<i4") or source.shape != (BLOCK_SIZE, 2):
                raise ValueError("callback input은 exact <i4 [256,2]여야 합니다")
            if cursor + BLOCK_SIZE > total_frames:
                raise ValueError("callback이 planned zero 경계를 넘습니다")

            times = tuple(
                _time_value(time_info, name)
                for name in (
                    "inputBufferAdcTime",
                    "outputBufferDacTime",
                    "currentTime",
                )
            )
            if not all(np.isfinite(item) for item in times):
                raise ValueError("callback timestamp가 finite가 아닙니다")
            if last_times is not None and any(
                current <= previous
                for current, previous in zip(times, last_times)
            ):
                raise ValueError("callback timestamp가 strict-monotonic이 아닙니다")

            mask = int(status_bitmask_v5(status))
            input_block = np.array(source, dtype="<i4", copy=True, order="C")
            submitted_block = np.array(sink, dtype="<i4", copy=True, order="C")
            if np.count_nonzero(submitted_block):
                raise RuntimeError("actual submitted block에 nonzero sample이 있습니다")

            start = cursor
            stop = cursor + BLOCK_SIZE
            captured[start:stop] = input_block
            actual[start:stop] = submitted_block
            capture_valid[start:stop] = True
            submitted_valid[start:stop] = True
            rows.append((len(rows), start, count, *times, mask))
            last_times = times
            cursor = stop

            if mask & STATUS_XRUN_MASK:
                invalid.append("portaudio_xrun_status")
            if mask & STATUS_UNEXPECTED:
                invalid.append("unexpected_portaudio_status")
            if mask & STATUS_PRIMING_OUTPUT:
                invalid.append("priming_callback_forbidden")
            if cursor == total_frames:
                completed = True
                raise backend.CallbackStop
        except backend.CallbackStop:
            raise
        except Exception as error:
            zero_error: str | None = None
            try:
                outdata[...] = 0
                zero_confirmed = bool(np.count_nonzero(np.asarray(outdata)) == 0)
            except Exception as nested:
                zero_error = f"{type(nested).__name__}: {nested}"
            if not zero_confirmed:
                invalid.append("output_zero_not_confirmed_on_callback_failure")
            callback_error = f"{type(error).__name__}: {error}"
            if zero_error is not None:
                callback_error += f"; zero_fill={zero_error}"
            raise backend.CallbackAbort

    stream = None
    faults: list[tuple[str, BaseException]] = []
    callback_fault_recorded = False
    stream_constructor_error: BaseException | None = None
    stream_start_error: BaseException | None = None
    watchdog_error: BaseException | None = None
    stop_error: BaseException | None = None
    abort_error: BaseException | None = None
    close_error: BaseException | None = None
    on_output_closed_error: BaseException | None = None
    stream_start_returned_without_exception = False
    stream_stop_attempted = False
    stream_stop_returned_without_exception = False
    stream_abort_attempted = False
    stream_abort_returned_without_exception = False
    stream_close_attempted = False
    stream_close_returned_without_exception = False

    def add_fault(stage: str, error: BaseException) -> None:
        faults.append((stage, error))

    signal_scope = _DeferredLiveAudioSignals()
    signal_scope.__enter__()
    try:
        pre_open_ready = True
        try:
            if pre_open_check is not None:
                pre_open_check()
        except BaseException as error:
            add_fault("pre_open_check", error)
            pre_open_ready = False
        try:
            candidate = float(monotonic())
            if not np.isfinite(candidate) or candidate < pre_open_started:
                raise ValueError(
                    "pre-open monotonic completion은 finite nondecreasing이어야 합니다"
                )
            pre_open_completed = candidate
        except BaseException as error:
            add_fault("pre_open_timing", error)
            pre_open_completed = pre_open_started
            pre_open_ready = False

        if pre_open_ready and not signal_scope.received:
            try:
                stream = backend.Stream(
                    samplerate=SAMPLE_RATE,
                    blocksize=BLOCK_SIZE,
                    device=(input_device, output_device),
                    channels=CHANNELS,
                    dtype=(INPUT_DTYPE, OUTPUT_DTYPE),
                    latency=(LATENCY, LATENCY),
                    callback=callback,
                    dither_off=True,
                    prime_output_buffers_using_stream_callback=False,
                )
            except BaseException as error:
                stream_constructor_error = error
                add_fault("stream_constructor", error)

        if stream is not None and not signal_scope.received:
            try:
                started_candidate = float(monotonic())
                if (
                    not np.isfinite(started_candidate)
                    or started_candidate < pre_open_completed
                ):
                    raise ValueError(
                        "capture monotonic start는 Stream.start 직전 finite 값이어야 합니다"
                    )
                capture_started = started_candidate
            except BaseException as error:
                stream_start_error = error
                add_fault("stream_start_timing", error)

        if (
            stream is not None
            and capture_started is not None
            and stream_start_error is None
            and not signal_scope.received
        ):
            try:
                stream.start()
                stream_start_returned_without_exception = True
            except BaseException as error:
                stream_start_error = error
                add_fault("stream_start", error)

        if callback_error is not None:
            add_fault(
                "callback",
                RuntimeError(f"zero duplex callback 실패: {callback_error}"),
            )
            callback_fault_recorded = True

        if (
            stream_start_returned_without_exception
            and callback_error is None
            and not signal_scope.received
            and on_stream_started is not None
        ):
            try:
                on_stream_started()
            except BaseException as error:
                add_fault("on_stream_started", error)

        if (
            stream_start_returned_without_exception
            and not faults
            and not signal_scope.received
        ):
            deadline = capture_started + total_frames / SAMPLE_RATE + grace
            while not completed and not signal_scope.received:
                if callback_error is not None:
                    if not callback_fault_recorded:
                        add_fault(
                            "callback",
                            RuntimeError(
                                f"zero duplex callback 실패: {callback_error}"
                            ),
                        )
                        callback_fault_recorded = True
                    break
                try:
                    now = float(monotonic())
                    if not np.isfinite(now):
                        raise ValueError("watchdog monotonic은 finite여야 합니다")
                    if now >= deadline:
                        raise TimeoutError("zero duplex watchdog 초과")
                except BaseException as error:
                    watchdog_error = error
                    add_fault("watchdog", error)
                    break
                try:
                    sleep(0.001)
                except BaseException as error:
                    watchdog_error = error
                    add_fault("watchdog_wait", error)
                    break

        if callback_error is not None and not callback_fault_recorded:
            add_fault(
                "callback",
                RuntimeError(f"zero duplex callback 실패: {callback_error}"),
            )
            callback_fault_recorded = True
        if invalid:
            add_fault(
                "callback_status",
                RuntimeError(
                    "callback canonical-invalid: "
                    + ",".join(dict.fromkeys(invalid))
                ),
            )

        can_stop_normally = bool(
            stream is not None
            and stream_start_returned_without_exception
            and completed
            and not faults
            and not signal_scope.received
        )
        if can_stop_normally:
            stream_stop_attempted = True
            try:
                stream.stop(ignore_errors=False)
                stream_stop_returned_without_exception = True
            except BaseException as error:
                stop_error = error
                add_fault("stream_stop", error)

        if stream is not None and not stream_stop_returned_without_exception:
            stream_abort_attempted = True
            try:
                stream.abort(ignore_errors=False)
                stream_abort_returned_without_exception = True
            except BaseException as error:
                abort_error = error
                add_fault("stream_abort", error)

        if stream is not None:
            stream_close_attempted = True
            try:
                stream.close(ignore_errors=False)
                stream_close_returned_without_exception = True
            except BaseException as error:
                close_error = error
                add_fault("stream_close", error)

        if on_output_closed is not None:
            try:
                on_output_closed(stream_close_returned_without_exception)
            except BaseException as error:
                on_output_closed_error = error
                add_fault("on_output_closed", error)

        if capture_started is None:
            capture_started = pre_open_completed
            capture_completed = capture_started
        else:
            try:
                capture_completed = float(monotonic())
            except BaseException as error:
                add_fault("capture_completion_clock", error)
                capture_completed = capture_started
        if not np.isfinite(capture_completed):
            add_fault(
                "capture_completion_timing",
                ValueError("capture monotonic completion은 finite여야 합니다"),
            )
            capture_completed = capture_started
        elapsed = capture_completed - capture_started
        if not np.isfinite(elapsed) or elapsed < 0.0:
            add_fault(
                "capture_completion_timing",
                ValueError(
                    "capture monotonic elapsed는 finite nonnegative여야 합니다"
                ),
            )
            elapsed = 0.0
    finally:
        signal_scope.__exit__(None, None, None)

    for error in signal_scope.restore_errors:
        add_fault("signal_handler_restore", error)
    termination_signal = (
        None if not signal_scope.received else int(signal_scope.received[0])
    )
    termination_exit_code = (
        None if termination_signal is None else 128 + termination_signal
    )
    if termination_signal is not None:
        add_fault(
            "termination_signal",
            _DeferredLiveAudioTermination(termination_signal),
        )

    def column(index: int, dtype: str) -> np.ndarray:
        return np.asarray([row[index] for row in rows], dtype=dtype)

    sequence = column(0, "<i8")
    start_frames = column(1, "<i8")
    frame_counts = column(2, "<i8")
    masks = column(6, "<u4")
    expected_callbacks = total_frames // BLOCK_SIZE
    callback_sequence_contiguous = bool(
        np.array_equal(sequence, np.arange(len(rows), dtype="<i8"))
    )
    callback_start_frames_contiguous = bool(
        np.array_equal(
            start_frames,
            np.arange(len(rows), dtype="<i8") * BLOCK_SIZE,
        )
    )
    callback_frame_counts_exact = bool(
        np.array_equal(
            frame_counts,
            np.full(len(rows), BLOCK_SIZE, dtype="<i8"),
        )
    )
    capture_valid_all_true = bool(np.all(capture_valid))
    submitted_valid_all_true = bool(np.all(submitted_valid))
    full_frame_accounting_valid = bool(
        completed
        and cursor == total_frames
        and len(rows) == expected_callbacks
        and callback_sequence_contiguous
        and callback_start_frames_contiguous
        and callback_frame_counts_exact
        and capture_valid_all_true
        and submitted_valid_all_true
    )
    application_buffer_zero_submission_complete = bool(
        full_frame_accounting_valid
        and np.count_nonzero(actual) == 0
        and callback_zero_attempt_count == expected_callbacks
        and callback_zero_confirmed_count == expected_callbacks
    )

    def error_text(error: BaseException | None) -> str | None:
        return None if error is None else f"{type(error).__name__}: {error}"

    def first_stage_error(*stages: str) -> str | None:
        accepted = set(stages)
        for stage, error in faults:
            if stage in accepted:
                return error_text(error)
        return None

    serialized_faults = [
        {
            "stage": stage,
            "exception_type": type(error).__name__,
            "message": str(error),
        }
        for stage, error in faults
    ]
    telemetry = {
        "schema": ZERO_DUPLEX_TELEMETRY_SCHEMA,
        "authority": "zero_duplex_transport_only_no_sample_identity",
        "callback_frame_semantics": "software_accounting_only_not_hardware_slip_witness",
        "output_zero_scope": "portaudio_application_callback_buffer_only",
        "portaudio_application_buffer_only": True,
        "portaudio_timestamp_authority": False,
        "hardware_sample_slip_authority": False,
        "physical_output_zero_authority": False,
        "electrical_output_zero_authority": False,
        "acoustic_output_zero_authority": False,
        "sample_rate_hz": SAMPLE_RATE,
        "block_size": BLOCK_SIZE,
        "latency": LATENCY,
        "channels": list(CHANNELS),
        "input_dtype": "<i4",
        "output_dtype": "<i4",
        "dither_off": True,
        "resolved_input_device": input_device,
        "resolved_output_device": output_device,
        "planned_frames": total_frames,
        "expected_callbacks": expected_callbacks,
        "pre_open_monotonic_started": pre_open_started,
        "pre_open_monotonic_completed": pre_open_completed,
        "capture_monotonic_started": capture_started,
        "capture_monotonic_completed": capture_completed,
        "capture_monotonic_elapsed_seconds": elapsed,
        "watchdog_grace_seconds": grace,
        "callback_sequence": sequence,
        "callback_start_frames": start_frames,
        "callback_frame_counts": frame_counts,
        "input_buffer_adc_time": column(3, "<f8"),
        "output_buffer_dac_time": column(4, "<f8"),
        "callback_current_time": column(5, "<f8"),
        "callback_status_bitmask": masks,
        "xrun_count": int(np.count_nonzero(masks & STATUS_XRUN_MASK)),
        "status_present_count": int(np.count_nonzero(masks & STATUS_PRESENT)),
        "captured_frames": int(np.count_nonzero(capture_valid)),
        "submitted_frames": int(np.count_nonzero(submitted_valid)),
        "actual_submitted_nonzero_count": int(np.count_nonzero(actual)),
        "callback_zero_attempt_count": int(callback_zero_attempt_count),
        "callback_zero_confirmed_count": int(callback_zero_confirmed_count),
        "callback_sequence_contiguous": callback_sequence_contiguous,
        "callback_start_frames_contiguous": callback_start_frames_contiguous,
        "callback_frame_counts_exact": callback_frame_counts_exact,
        "capture_valid_all_true": capture_valid_all_true,
        "submitted_valid_all_true": submitted_valid_all_true,
        "full_frame_accounting_valid": full_frame_accounting_valid,
        "actual_submitted_pcm_hash_eligible": full_frame_accounting_valid,
        "application_buffer_zero_submission_complete": (
            application_buffer_zero_submission_complete
        ),
        "completed": bool(completed),
        "callback_error": callback_error,
        "canonical_invalid_reasons": list(dict.fromkeys(invalid)),
        "stream_constructor_error": error_text(stream_constructor_error),
        "stream_start_error": error_text(stream_start_error),
        "watchdog_error": error_text(watchdog_error),
        "stream_stop_error": error_text(stop_error),
        "stream_abort_error": error_text(abort_error),
        "stream_close_error": error_text(close_error),
        "on_output_closed_error": error_text(on_output_closed_error),
        "termination_signals": [int(item) for item in signal_scope.received],
        "termination_signal": termination_signal,
        "termination_exit_code": termination_exit_code,
        "stream_start_returned_without_exception": (
            stream_start_returned_without_exception
        ),
        "stream_stop_attempted": stream_stop_attempted,
        "stream_stop_returned_without_exception": (
            stream_stop_returned_without_exception
        ),
        "stream_abort_attempted": stream_abort_attempted,
        "stream_abort_returned_without_exception": (
            stream_abort_returned_without_exception
        ),
        "stream_close_attempted": stream_close_attempted,
        "stream_close_returned_without_exception": (
            stream_close_returned_without_exception
        ),
        "normal_stop_completed": bool(
            completed
            and stream_stop_returned_without_exception
            and stream_close_returned_without_exception
            and not faults
        ),
        "faults": serialized_faults,
    }
    # 위 stage별 shortcut이 모든 fault 집합과 일치하는지도 telemetry 내부에서
    # 검사할 수 있게 한다. ``first_stage_error``는 callback 오류처럼 별도 필드가
    # 이미 있는 단계에는 사용하지 않는다.
    if telemetry["stream_constructor_error"] is None:
        telemetry["stream_constructor_error"] = first_stage_error(
            "stream_constructor"
        )
    if telemetry["stream_start_error"] is None:
        telemetry["stream_start_error"] = first_stage_error(
            "stream_start_timing", "stream_start"
        )
    if telemetry["watchdog_error"] is None:
        telemetry["watchdog_error"] = first_stage_error(
            "watchdog", "watchdog_wait"
        )
    if faults:
        raise ZeroDuplexCaptureFailure(
            "; ".join(
                f"{stage}={type(error).__name__}: {error}"
                for stage, error in faults
            ),
            captured.copy(),
            actual.copy(),
            capture_valid.copy(),
            submitted_valid.copy(),
            telemetry,
        )
    return captured, {
        **telemetry,
        "actual_submitted_pcm": actual,
        "capture_valid_mask": capture_valid,
        "submitted_valid_mask": submitted_valid,
    }


__all__ = [
    "BLOCK_SIZE",
    "CHANNELS",
    "INPUT_DTYPE",
    "LATENCY",
    "OUTPUT_DTYPE",
    "SAMPLE_RATE",
    "ZERO_DUPLEX_TELEMETRY_SCHEMA",
    "ZeroDuplexCaptureFailure",
    "capture_zero_duplex",
]
