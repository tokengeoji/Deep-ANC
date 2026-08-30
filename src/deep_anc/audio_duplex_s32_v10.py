"""RT5640 fullband v10의 planned-S32 전이중 callback primitive.

이 모듈은 audio backend를 import하거나 ALSA device를 해석하지 않는다. 호출자가 전달한
backend에서 only little-endian S32 stereo planned PCM을 입력 S32/output S32로 제출한다.
callback의 첫 output side effect는 항상 zero-fill이고, planned block assignment가 exact
equality로 확인된 뒤에만 capture/actual-submitted evidence를 기록한다. 오류에서는 다시
zero-fill 후 abort한다.

성공 telemetry는 PortAudio application buffer accounting일 뿐 physical DAC output,
hardware sample identity, shared clock, P/S, attenuation, training/deployment authority가
아니다. v8 exact-zero generation/receipt를 import하거나 확장하지 않는다.
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
TELEMETRY_SCHEMA = "rt5640_fullband_s32_duplex_telemetry_v10"


class _DeferredLiveAudioSignals:
    """cleanup 중 POSIX signal을 기록해 close 뒤 실패로 승격한다."""

    def __init__(self) -> None:
        self.received: list[int] = []
        self.restore_errors: list[BaseException] = []
        self._previous: dict[signal.Signals, Any] = {}
        self._installed: list[signal.Signals] = []

    def __enter__(self) -> "_DeferredLiveAudioSignals":
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

    def __exit__(self, _exc_type, _exc, _traceback) -> bool:
        for item in reversed(self._installed):
            try:
                signal.signal(item, self._previous[item])
            except BaseException as error:
                self.restore_errors.append(error)
        return False


@dataclass
class S32DuplexCaptureFailure(RuntimeError):
    """실패해도 callback이 관측한 partial input/submitted evidence를 보존한다."""

    message: str
    captured_pcm: np.ndarray
    actual_submitted_pcm: np.ndarray
    capture_valid_mask: np.ndarray
    submitted_valid_mask: np.ndarray
    telemetry: dict[str, Any]

    def __str__(self) -> str:
        return self.message


def _array_sha256(value: np.ndarray) -> str:
    import hashlib

    array = np.asarray(value)
    if not array.flags.c_contiguous:
        raise ValueError("PCM SHA input은 C-contiguous여야 합니다")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _time_value(value: Any, name: str) -> float:
    raw = value.get(name, np.nan) if isinstance(value, dict) else getattr(
        value, name, np.nan
    )
    return float(raw)


def _require_planned_s32(value: np.ndarray) -> np.ndarray:
    planned = np.asarray(value)
    if planned.dtype != np.dtype("<i4"):
        raise ValueError("planned_pcm은 exact little-endian int32여야 합니다")
    if planned.ndim != 2 or planned.shape[0] <= 0 or planned.shape[1] != 2:
        raise ValueError("planned_pcm은 nonempty exact <i4 [frames,2]여야 합니다")
    if len(planned) % BLOCK_SIZE:
        raise ValueError("planned_pcm frame 수는 256의 배수여야 합니다")
    if not planned.flags.c_contiguous:
        raise ValueError("planned_pcm은 C-contiguous여야 합니다")
    return planned


def capture_planned_s32_duplex(
    backend: Any,
    *,
    planned_pcm: np.ndarray,
    input_device: int,
    output_device: int,
    pre_open_check: Callable[[], None] | None = None,
    on_stream_started: Callable[[], None] | None = None,
    on_output_closed: Callable[[bool], None] | None = None,
    watchdog_grace_seconds: float = 2.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[np.ndarray, dict[str, Any]]:
    """planned S32 block을 exact 제출하며 항상 zero-first failure path를 쓴다."""

    source = _require_planned_s32(planned_pcm)
    if type(input_device) is not int or input_device < 0:
        raise ValueError("input_device는 음이 아닌 exact int여야 합니다")
    if type(output_device) is not int or output_device < 0:
        raise ValueError("output_device는 음이 아닌 exact int여야 합니다")
    grace = float(watchdog_grace_seconds)
    if not np.isfinite(grace) or grace <= 0.0:
        raise ValueError("watchdog_grace_seconds는 finite 양수여야 합니다")
    for label, callback in (
        ("pre_open_check", pre_open_check),
        ("on_stream_started", on_stream_started),
        ("on_output_closed", on_output_closed),
    ):
        if callback is not None and not callable(callback):
            raise TypeError(f"{label}는 callable 또는 None이어야 합니다")

    # caller가 capture 중 source array를 mutate해도 callback plan은 바뀌지 않는다.
    planned = np.array(source, dtype="<i4", copy=True, order="C")
    planned.setflags(write=False)
    total = len(planned)
    planned_sha256 = _array_sha256(planned)
    captured = np.zeros((total, 2), dtype="<i4")
    actual = np.zeros((total, 2), dtype="<i4")
    capture_valid = np.zeros(total, dtype=np.bool_)
    submitted_valid = np.zeros(total, dtype=np.bool_)
    cursor = 0
    completed = False
    callback_error: str | None = None
    invalid: list[str] = []
    rows: list[tuple[Any, ...]] = []
    last_times: tuple[float, float, float] | None = None
    zero_attempts = 0
    zero_confirmed = 0
    assignment_attempts = 0
    assignment_confirmed = 0
    possible_nonzero_after_failed_assignment = False
    pre_open_started = float(monotonic())
    if not np.isfinite(pre_open_started):
        raise ValueError("pre-open monotonic start는 finite여야 합니다")
    pre_open_completed = pre_open_started
    capture_started: float | None = None

    def zero_out(outdata: Any) -> bool:
        nonlocal zero_attempts, zero_confirmed
        try:
            outdata[...] = 0
        finally:
            zero_attempts += 1
        confirmed = bool(np.count_nonzero(np.asarray(outdata)) == 0)
        if confirmed:
            zero_confirmed += 1
        return confirmed

    def callback(indata, outdata, frames, time_info, status):  # noqa: ANN001
        nonlocal cursor, completed, callback_error, last_times
        nonlocal assignment_attempts, assignment_confirmed
        nonlocal possible_nonzero_after_failed_assignment
        zero_ok = False
        assigned_not_committed = False
        try:
            # No validation precedes this zero-fill: malformed inputs must never leave
            # backend-provided nonzero bytes in the output buffer.
            zero_ok = zero_out(outdata)
            if not zero_ok:
                raise RuntimeError("callback output zero-fill을 확인하지 못했습니다")
            count = int(frames)
            sink = np.asarray(outdata)
            source_block = np.asarray(indata)
            if count != BLOCK_SIZE:
                raise ValueError("callback frames는 exact 256이어야 합니다")
            if sink.dtype != np.dtype("<i4") or sink.shape != (BLOCK_SIZE, 2):
                raise ValueError("callback output은 exact <i4 [256,2]여야 합니다")
            if source_block.dtype != np.dtype("<i4") or source_block.shape != (BLOCK_SIZE, 2):
                raise ValueError("callback input은 exact <i4 [256,2]여야 합니다")
            if cursor + BLOCK_SIZE > total:
                raise ValueError("callback이 sealed planned PCM 경계를 넘습니다")
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

            # xrun/unknown/priming callback은 current block도 plan을 쓰지 않는다.
            # 실제 nonzero submission 전에 fail-closed 하려면 status 판단이 assignment보다
            # 앞서야 한다. event row는 남기되 cursor/mask는 전진시키지 않는다.
            mask = int(status_bitmask_v5(status))
            if mask & (STATUS_XRUN_MASK | STATUS_UNEXPECTED | STATUS_PRIMING_OUTPUT):
                rows.append((len(rows), cursor, count, *times, mask))
                last_times = times
                if mask & STATUS_XRUN_MASK:
                    invalid.append("portaudio_xrun_status")
                if mask & STATUS_UNEXPECTED:
                    invalid.append("unexpected_portaudio_status")
                if mask & STATUS_PRIMING_OUTPUT:
                    invalid.append("priming_callback_forbidden")
                raise RuntimeError("callback status가 planned nonzero assignment 전에 invalid입니다")

            block = planned[cursor : cursor + BLOCK_SIZE].copy()
            assignment_attempts += 1
            outdata[...] = block
            assigned_not_committed = True
            if not np.array_equal(np.asarray(outdata), block):
                raise RuntimeError("callback planned S32 block assignment을 확인하지 못했습니다")
            assignment_confirmed += 1
            start = cursor
            stop = cursor + BLOCK_SIZE
            captured[start:stop] = np.array(source_block, dtype="<i4", copy=True, order="C")
            actual[start:stop] = np.array(sink, dtype="<i4", copy=True, order="C")
            capture_valid[start:stop] = True
            submitted_valid[start:stop] = True
            rows.append((len(rows), start, count, *times, mask))
            last_times = times
            cursor = stop
            assigned_not_committed = False
            if cursor == total:
                completed = True
                raise backend.CallbackStop
        except backend.CallbackStop:
            raise
        except Exception as error:
            if assigned_not_committed:
                possible_nonzero_after_failed_assignment = True
            zero_error: str | None = None
            try:
                zero_ok = zero_out(outdata)
            except Exception as nested:
                zero_error = f"{type(nested).__name__}: {nested}"
                zero_ok = False
            if not zero_ok:
                invalid.append("output_zero_not_confirmed_on_callback_failure")
            callback_error = f"{type(error).__name__}: {error}"
            if zero_error is not None:
                callback_error += f"; zero_fill={zero_error}"
            raise backend.CallbackAbort

    stream = None
    faults: list[tuple[str, BaseException]] = []
    stream_constructor_error: BaseException | None = None
    stream_start_error: BaseException | None = None
    watchdog_error: BaseException | None = None
    stop_error: BaseException | None = None
    abort_error: BaseException | None = None
    close_error: BaseException | None = None
    close_notice_error: BaseException | None = None
    start_returned = False
    stop_returned = False
    abort_returned = False
    close_returned = False
    stop_attempted = False
    abort_attempted = False
    close_attempted = False

    def add_fault(stage: str, error: BaseException) -> None:
        faults.append((stage, error))

    signal_scope = _DeferredLiveAudioSignals()
    signal_scope.__enter__()
    try:
        ready = True
        try:
            if pre_open_check is not None:
                pre_open_check()
        except BaseException as error:
            add_fault("pre_open_check", error)
            ready = False
        try:
            candidate = float(monotonic())
            if not np.isfinite(candidate) or candidate < pre_open_started:
                raise ValueError("pre-open monotonic completion은 finite nondecreasing이어야 합니다")
            pre_open_completed = candidate
        except BaseException as error:
            add_fault("pre_open_timing", error)
            ready = False

        if ready and not signal_scope.received:
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
                candidate = float(monotonic())
                if not np.isfinite(candidate) or candidate < pre_open_completed:
                    raise ValueError("capture monotonic start는 finite nondecreasing이어야 합니다")
                capture_started = candidate
                stream.start()
                start_returned = True
            except BaseException as error:
                stream_start_error = error
                add_fault("stream_start", error)

        if callback_error is not None:
            add_fault("callback", RuntimeError(f"planned S32 callback 실패: {callback_error}"))
        if stream is not None and start_returned and not faults and not signal_scope.received:
            if on_stream_started is not None:
                try:
                    on_stream_started()
                except BaseException as error:
                    add_fault("on_stream_started", error)
        if stream is not None and start_returned and not faults and not signal_scope.received:
            assert capture_started is not None
            deadline = capture_started + total / SAMPLE_RATE + grace
            while not completed and not signal_scope.received:
                if callback_error is not None:
                    add_fault("callback", RuntimeError(f"planned S32 callback 실패: {callback_error}"))
                    break
                try:
                    now = float(monotonic())
                    if not np.isfinite(now):
                        raise ValueError("watchdog monotonic은 finite여야 합니다")
                    if now >= deadline:
                        raise TimeoutError("planned S32 duplex watchdog 초과")
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
        if callback_error is not None and not any(stage == "callback" for stage, _ in faults):
            add_fault("callback", RuntimeError(f"planned S32 callback 실패: {callback_error}"))
        if invalid:
            add_fault(
                "callback_status",
                RuntimeError("callback canonical-invalid: " + ",".join(dict.fromkeys(invalid))),
            )

        normal_stop = bool(stream is not None and start_returned and completed and not faults and not signal_scope.received)
        if normal_stop:
            stop_attempted = True
            try:
                stream.stop(ignore_errors=False)
                stop_returned = True
            except BaseException as error:
                stop_error = error
                add_fault("stream_stop", error)
        if stream is not None and not stop_returned:
            abort_attempted = True
            try:
                stream.abort(ignore_errors=False)
                abort_returned = True
            except BaseException as error:
                abort_error = error
                add_fault("stream_abort", error)
        if stream is not None:
            close_attempted = True
            try:
                stream.close(ignore_errors=False)
                close_returned = True
            except BaseException as error:
                close_error = error
                add_fault("stream_close", error)
        if on_output_closed is not None:
            try:
                on_output_closed(close_returned)
            except BaseException as error:
                close_notice_error = error
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
        if not np.isfinite(capture_completed) or capture_completed < capture_started:
            add_fault("capture_completion_timing", ValueError("capture completion timing이 유효하지 않습니다"))
            capture_completed = capture_started
    finally:
        signal_scope.__exit__(None, None, None)

    for error in signal_scope.restore_errors:
        add_fault("signal_handler_restore", error)
    if signal_scope.received:
        signum = int(signal_scope.received[0])
        add_fault("termination_signal", RuntimeError(f"deferred live audio signal {signum} (exit {128 + signum})"))

    def column(index: int, dtype: str) -> np.ndarray:
        return np.asarray([row[index] for row in rows], dtype=dtype)

    sequence = column(0, "<i8")
    starts = column(1, "<i8")
    counts = column(2, "<i8")
    masks = column(6, "<u4")
    expected_callbacks = total // BLOCK_SIZE
    sequence_contiguous = bool(np.array_equal(sequence, np.arange(len(rows), dtype="<i8")))
    starts_contiguous = bool(np.array_equal(starts, np.arange(len(rows), dtype="<i8") * BLOCK_SIZE))
    counts_exact = bool(np.array_equal(counts, np.full(len(rows), BLOCK_SIZE, dtype="<i8")))
    capture_all = bool(np.all(capture_valid))
    submitted_all = bool(np.all(submitted_valid))
    full_accounting = bool(
        completed
        and cursor == total
        and len(rows) == expected_callbacks
        and sequence_contiguous
        and starts_contiguous
        and counts_exact
        and capture_all
        and submitted_all
    )
    actual_matches = bool(full_accounting and np.array_equal(actual, planned))
    hash_eligible = bool(full_accounting and actual_matches and not invalid and callback_error is None)

    def error_text(error: BaseException | None) -> str | None:
        return None if error is None else f"{type(error).__name__}: {error}"

    telemetry = {
        "schema": TELEMETRY_SCHEMA,
        "authority": "application_buffer_planned_s32_only_no_physical_sample_identity",
        "portaudio_application_buffer_only": True,
        "portaudio_timestamp_authority": False,
        "hardware_sample_slip_authority": False,
        "physical_output_authority": False,
        "electrical_output_authority": False,
        "acoustic_output_authority": False,
        "sample_rate_hz": SAMPLE_RATE,
        "block_size": BLOCK_SIZE,
        "latency": LATENCY,
        "channels": list(CHANNELS),
        "input_dtype": "<i4",
        "output_dtype": "<i4",
        "dither_off": True,
        "planned_pcm_sha256": planned_sha256,
        "planned_frames": total,
        "expected_callbacks": expected_callbacks,
        "resolved_input_device": input_device,
        "resolved_output_device": output_device,
        "pre_open_monotonic_started": pre_open_started,
        "pre_open_monotonic_completed": pre_open_completed,
        "capture_monotonic_started": capture_started,
        "capture_monotonic_completed": capture_completed,
        "capture_monotonic_elapsed_seconds": capture_completed - capture_started,
        "watchdog_grace_seconds": grace,
        "callback_sequence": sequence,
        "callback_start_frames": starts,
        "callback_frame_counts": counts,
        "input_buffer_adc_time": column(3, "<f8"),
        "output_buffer_dac_time": column(4, "<f8"),
        "callback_current_time": column(5, "<f8"),
        "callback_status_bitmask": masks,
        "xrun_count": int(np.count_nonzero(masks & STATUS_XRUN_MASK)),
        "status_present_count": int(np.count_nonzero(masks & STATUS_PRESENT)),
        "captured_frames": int(np.count_nonzero(capture_valid)),
        "submitted_frames": int(np.count_nonzero(submitted_valid)),
        "callback_zero_attempt_count": zero_attempts,
        "callback_zero_confirmed_count": zero_confirmed,
        "callback_planned_assignment_attempt_count": assignment_attempts,
        "callback_planned_assignment_confirmed_count": assignment_confirmed,
        "possible_nonzero_output_after_failed_assignment": (
            possible_nonzero_after_failed_assignment
        ),
        "on_stream_started_is_observational_not_nonzero_admission": True,
        "callback_sequence_contiguous": sequence_contiguous,
        "callback_start_frames_contiguous": starts_contiguous,
        "callback_frame_counts_exact": counts_exact,
        "capture_valid_all_true": capture_all,
        "submitted_valid_all_true": submitted_all,
        "full_frame_accounting_valid": full_accounting,
        "actual_matches_planned_application_buffer": actual_matches,
        "actual_submitted_pcm_hash_eligible": hash_eligible,
        "actual_submitted_pcm_sha256": _array_sha256(actual) if hash_eligible else None,
        "completed": bool(completed),
        "callback_error": callback_error,
        "canonical_invalid_reasons": list(dict.fromkeys(invalid)),
        "stream_constructor_error": error_text(stream_constructor_error),
        "stream_start_error": error_text(stream_start_error),
        "watchdog_error": error_text(watchdog_error),
        "stream_stop_error": error_text(stop_error),
        "stream_abort_error": error_text(abort_error),
        "stream_close_error": error_text(close_error),
        "on_output_closed_error": error_text(close_notice_error),
        "termination_signals": [int(item) for item in signal_scope.received],
        "termination_signal": None if not signal_scope.received else int(signal_scope.received[0]),
        "termination_exit_code": None if not signal_scope.received else 128 + int(signal_scope.received[0]),
        "stream_stop_attempted": stop_attempted,
        "stream_stop_returned_without_exception": stop_returned,
        "stream_abort_attempted": abort_attempted,
        "stream_abort_returned_without_exception": abort_returned,
        "stream_close_attempted": close_attempted,
        "stream_close_returned_without_exception": close_returned,
        "normal_stop_completed": bool(completed and stop_returned and close_returned and not faults),
        "faults": [
            {"stage": stage, "exception_type": type(error).__name__, "message": str(error)}
            for stage, error in faults
        ],
    }
    if faults:
        raise S32DuplexCaptureFailure(
            "; ".join(f"{stage}={type(error).__name__}: {error}" for stage, error in faults),
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
    "S32DuplexCaptureFailure",
    "TELEMETRY_SCHEMA",
    "capture_planned_s32_duplex",
]
