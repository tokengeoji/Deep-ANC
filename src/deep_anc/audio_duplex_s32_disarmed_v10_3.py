"""RT5640 S32 live 측정용 disarmed planned-duplex primitive.

``stream.start()`` 뒤에 callback이 먼저 실행될 수 있으므로, 단순한
``on_stream_started`` callback은 nonzero PCM admission이 될 수 없다. 이 primitive는
stream을 열고 나서도 supervisor의 ``post_start_pre_arm_check``가 성공할 때까지 모든
callback output을 exact zero로 유지한다. check는 실제 hw_params/route/J511/occupancy를
읽는 외부 adapter가 제공해야 한다. 이 모듈은 backend를 import하거나 ALSA를 해석하지
않는다.

성공 telemetry는 application buffer accounting만 의미한다. physical DAC, electrical
tap, shared clock, P/S, attenuation 또는 training authority는 만들지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import signal
import threading
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
TELEMETRY_SCHEMA = "rt5640_fullband_s32_disarmed_duplex_telemetry_v10_3"


class _DeferredLiveAudioSignals:
    """cleanup 중 INT/TERM/HUP를 기록하고 close 뒤에 실패로 승격한다."""

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
class S32DisarmedDuplexCaptureFailure(RuntimeError):
    """partial raw/mask/telemetry를 버리지 않는 fail-closed exception."""

    message: str
    captured_pcm: np.ndarray
    actual_submitted_pcm: np.ndarray
    capture_valid_mask: np.ndarray
    submitted_valid_mask: np.ndarray
    telemetry: dict[str, Any]

    def __str__(self) -> str:
        return self.message


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        raise ValueError("PCM SHA input은 C-contiguous여야 합니다")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


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


def _time_value(value: Any, name: str) -> float:
    raw = value.get(name, np.nan) if isinstance(value, dict) else getattr(value, name, np.nan)
    return float(raw)


def _error_text(error: BaseException | None) -> str | None:
    return None if error is None else f"{type(error).__name__}: {error}"


def capture_disarmed_planned_s32_duplex(
    backend: Any,
    *,
    planned_pcm: np.ndarray,
    input_device: int,
    output_device: int,
    post_start_pre_arm_check: Callable[[], None],
    pre_open_check: Callable[[], None] | None = None,
    on_output_closed: Callable[[bool], None] | None = None,
    watchdog_grace_seconds: float = 2.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[np.ndarray, dict[str, Any]]:
    """hw_params admission까지 zero-only인 same-stream planned S32 capture.

    ``post_start_pre_arm_check``는 필수다. 호출자는 이 callback에서 stream open 뒤의
    negotiated hw_params, route, current J511/occupancy 및 planned-S32-only condition을
    읽기 전용으로 확인해야 한다. check가 성공하기 전에는 cursor/mask/capture가
    전혀 전진하지 않는다.
    """

    source = _require_planned_s32(planned_pcm)
    if type(input_device) is not int or input_device < 0:
        raise ValueError("input_device는 음이 아닌 exact int여야 합니다")
    if type(output_device) is not int or output_device < 0:
        raise ValueError("output_device는 음이 아닌 exact int여야 합니다")
    if not callable(post_start_pre_arm_check):
        raise TypeError("post_start_pre_arm_check는 mandatory callable이어야 합니다")
    for label, callback in (("pre_open_check", pre_open_check), ("on_output_closed", on_output_closed)):
        if callback is not None and not callable(callback):
            raise TypeError(f"{label}는 callable 또는 None이어야 합니다")
    grace = float(watchdog_grace_seconds)
    if not np.isfinite(grace) or grace <= 0.0:
        raise ValueError("watchdog_grace_seconds는 finite 양수여야 합니다")

    # capture 중 caller가 source bytes를 바꿔도 output plan은 바뀌지 않는다.
    planned = np.array(source, dtype="<i4", copy=True, order="C")
    planned.setflags(write=False)
    total = len(planned)
    planned_sha256 = _array_sha256(planned)
    captured = np.zeros((total, 2), dtype="<i4")
    actual = np.zeros((total, 2), dtype="<i4")
    capture_valid = np.zeros(total, dtype=np.bool_)
    submitted_valid = np.zeros(total, dtype=np.bool_)

    # state_lock은 main supervisor의 arm과 PortAudio callback 사이의 sole handoff다.
    # callback은 unarmed에서 cursor를 절대 바꾸지 않는다.
    state_lock = threading.Lock()
    armed = False
    arm_time: float | None = None
    cursor = 0
    completed = False
    callback_error: str | None = None
    invalid: list[str] = []
    last_times: tuple[float, float, float] | None = None
    prearm_rows: list[tuple[Any, ...]] = []
    planned_rows: list[tuple[Any, ...]] = []
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
    arm_check_started: float | None = None
    arm_check_completed: float | None = None
    arm_check_passed = False

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

    def record_callback_failure(error: BaseException) -> None:
        nonlocal callback_error
        with state_lock:
            if callback_error is None:
                callback_error = f"{type(error).__name__}: {error}"

    def callback(indata, outdata, frames, time_info, status):  # noqa: ANN001
        nonlocal cursor, completed, last_times, assignment_attempts, assignment_confirmed
        nonlocal possible_nonzero_after_failed_assignment
        zero_ok = False
        assigned_not_committed = False
        try:
            # 모든 validation보다 먼저 zero fill을 한다. callback start 때 backend buffer에
            # 남아 있을 수 있는 bytes가 planned nonzero로 해석될 경로를 없앤다.
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
            times = tuple(
                _time_value(time_info, name)
                for name in ("inputBufferAdcTime", "outputBufferDacTime", "currentTime")
            )
            if not all(np.isfinite(item) for item in times):
                raise ValueError("callback timestamp가 finite가 아닙니다")
            with state_lock:
                if last_times is not None and any(
                    current <= previous for current, previous in zip(times, last_times)
                ):
                    raise ValueError("callback timestamp가 strict-monotonic이 아닙니다")
                last_times = times
            mask = int(status_bitmask_v5(status))
            if mask & (STATUS_XRUN_MASK | STATUS_UNEXPECTED | STATUS_PRIMING_OUTPUT):
                with state_lock:
                    target_rows = planned_rows if armed else prearm_rows
                    target_rows.append((len(target_rows), cursor, count, *times, mask))
                    if mask & STATUS_XRUN_MASK:
                        invalid.append("portaudio_xrun_status")
                    if mask & STATUS_UNEXPECTED:
                        invalid.append("unexpected_portaudio_status")
                    if mask & STATUS_PRIMING_OUTPUT:
                        invalid.append("priming_callback_forbidden")
                raise RuntimeError("callback status가 planned nonzero assignment 전에 invalid입니다")

            with state_lock:
                current_armed = armed
                start = cursor
            if not current_armed:
                # Stream.start 내부 callback도 hardware checks 전에는 zero-only다.
                with state_lock:
                    prearm_rows.append((len(prearm_rows), start, count, *times, mask))
                return
            if start + BLOCK_SIZE > total:
                raise ValueError("callback이 sealed planned PCM 경계를 넘습니다")
            block = planned[start : start + BLOCK_SIZE].copy()
            with state_lock:
                assignment_attempts += 1
            outdata[...] = block
            assigned_not_committed = True
            if not np.array_equal(np.asarray(outdata), block):
                raise RuntimeError("callback planned S32 block assignment을 확인하지 못했습니다")
            with state_lock:
                assignment_confirmed += 1
            input_copy = np.array(source_block, dtype="<i4", copy=True, order="C")
            actual_copy = np.array(sink, dtype="<i4", copy=True, order="C")
            with state_lock:
                # PortAudio callback은 serial이어야 한다. 다른 callback이 cursor를
                # 움직였다면 zero-only abort를 선택한다.
                if cursor != start:
                    raise RuntimeError("planned S32 callback cursor가 concurrent하게 바뀌었습니다")
                stop = start + BLOCK_SIZE
                captured[start:stop] = input_copy
                actual[start:stop] = actual_copy
                capture_valid[start:stop] = True
                submitted_valid[start:stop] = True
                planned_rows.append((len(planned_rows), start, count, *times, mask))
                cursor = stop
                completed = cursor == total
            assigned_not_committed = False
            if completed:
                raise backend.CallbackStop
        except backend.CallbackStop:
            raise
        except Exception as error:
            if assigned_not_committed:
                with state_lock:
                    possible_nonzero_after_failed_assignment = True
            zero_error: str | None = None
            try:
                zero_ok = zero_out(outdata)
            except Exception as nested:
                zero_error = f"{type(nested).__name__}: {nested}"
                zero_ok = False
            with state_lock:
                if not zero_ok:
                    invalid.append("output_zero_not_confirmed_on_callback_failure")
            record_callback_failure(error)
            if zero_error is not None:
                with state_lock:
                    callback_error = f"{callback_error}; zero_fill={zero_error}"
            raise backend.CallbackAbort

    stream = None
    faults: list[tuple[str, BaseException]] = []
    stream_constructor_error: BaseException | None = None
    stream_start_error: BaseException | None = None
    arm_check_error: BaseException | None = None
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

        # callback은 start() 내부에서 이미 돌았을 수 있다. 그러나 arm check는 start
        # return 뒤에만 수행되고, error/status가 하나라도 있으면 armed를 바꾸지 않는다.
        if stream is not None and start_returned and not signal_scope.received:
            with state_lock:
                callback_failed_before_arm = callback_error is not None or bool(invalid)
            if callback_failed_before_arm:
                add_fault("prearm_callback", RuntimeError("pre-arm callback이 invalid했습니다"))
            else:
                try:
                    candidate = float(monotonic())
                    if not np.isfinite(candidate) or capture_started is None or candidate < capture_started:
                        raise ValueError("arm check start monotonic이 유효하지 않습니다")
                    arm_check_started = candidate
                    post_start_pre_arm_check()
                    candidate = float(monotonic())
                    if not np.isfinite(candidate) or candidate < arm_check_started:
                        raise ValueError("arm check completion monotonic이 유효하지 않습니다")
                    arm_check_completed = candidate
                    with state_lock:
                        if callback_error is not None or invalid:
                            raise RuntimeError("arm check 중 callback invalid가 발생했습니다")
                        armed = True
                        arm_time = candidate
                        arm_check_passed = True
                except BaseException as error:
                    arm_check_error = error
                    add_fault("post_start_pre_arm_check", error)

        if stream is not None and start_returned and arm_check_passed and not faults and not signal_scope.received:
            assert arm_time is not None
            deadline = arm_time + total / SAMPLE_RATE + grace
            while not completed and not signal_scope.received:
                with state_lock:
                    callback_failure = callback_error
                if callback_failure is not None:
                    add_fault("callback", RuntimeError(f"disarmed S32 callback 실패: {callback_failure}"))
                    break
                try:
                    now = float(monotonic())
                    if not np.isfinite(now):
                        raise ValueError("watchdog monotonic은 finite여야 합니다")
                    if now >= deadline:
                        raise TimeoutError("disarmed planned S32 duplex watchdog 초과")
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
        with state_lock:
            final_callback_error = callback_error
            final_invalid = list(dict.fromkeys(invalid))
        if final_callback_error is not None and not any(stage == "callback" for stage, _ in faults):
            add_fault("callback", RuntimeError(f"disarmed S32 callback 실패: {final_callback_error}"))
        if final_invalid:
            add_fault("callback_status", RuntimeError("callback canonical-invalid: " + ",".join(final_invalid)))

        normal_stop = bool(
            stream is not None and start_returned and completed and arm_check_passed
            and not faults and not signal_scope.received
        )
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

    def _column(rows: list[tuple[Any, ...]], index: int, dtype: str) -> np.ndarray:
        return np.asarray([row[index] for row in rows], dtype=dtype)

    with state_lock:
        final_cursor = cursor
        final_completed = completed
        final_prearm_rows = list(prearm_rows)
        final_planned_rows = list(planned_rows)
        final_invalid = list(dict.fromkeys(invalid))
        final_possible_nonzero = possible_nonzero_after_failed_assignment
        final_assignment_attempts = assignment_attempts
        final_assignment_confirmed = assignment_confirmed
    planned_sequence = _column(final_planned_rows, 0, "<i8")
    planned_starts = _column(final_planned_rows, 1, "<i8")
    planned_counts = _column(final_planned_rows, 2, "<i8")
    planned_masks = _column(final_planned_rows, 6, "<u4")
    prearm_masks = _column(final_prearm_rows, 6, "<u4")
    expected_callbacks = total // BLOCK_SIZE
    sequence_contiguous = bool(np.array_equal(planned_sequence, np.arange(len(final_planned_rows), dtype="<i8")))
    starts_contiguous = bool(np.array_equal(planned_starts, np.arange(len(final_planned_rows), dtype="<i8") * BLOCK_SIZE))
    counts_exact = bool(np.array_equal(planned_counts, np.full(len(final_planned_rows), BLOCK_SIZE, dtype="<i8")))
    capture_all = bool(np.all(capture_valid))
    submitted_all = bool(np.all(submitted_valid))
    full_accounting = bool(
        final_completed and final_cursor == total and len(final_planned_rows) == expected_callbacks
        and sequence_contiguous and starts_contiguous and counts_exact and capture_all and submitted_all
    )
    actual_matches = bool(full_accounting and np.array_equal(actual, planned))
    hash_eligible = bool(
        full_accounting and actual_matches and arm_check_passed and not final_invalid
        and final_callback_error is None
    )
    prearm_zero_observed = bool(zero_confirmed >= len(final_prearm_rows))
    telemetry = {
        "schema": TELEMETRY_SCHEMA,
        "authority": "application_buffer_disarmed_planned_s32_only_no_physical_sample_identity",
        "portaudio_application_buffer_only": True,
        "post_start_pre_arm_check_required": True,
        "post_start_pre_arm_check_passed": arm_check_passed,
        "nonzero_assignment_before_arm_allowed": False,
        "on_stream_started_is_nonzero_admission": False,
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
        "expected_planned_callbacks": expected_callbacks,
        "resolved_input_device": input_device,
        "resolved_output_device": output_device,
        "pre_open_monotonic_started": pre_open_started,
        "pre_open_monotonic_completed": pre_open_completed,
        "capture_monotonic_started": capture_started,
        "arm_check_monotonic_started": arm_check_started,
        "arm_check_monotonic_completed": arm_check_completed,
        "arm_monotonic": arm_time,
        "capture_monotonic_completed": capture_completed,
        "capture_monotonic_elapsed_seconds": capture_completed - capture_started,
        "watchdog_grace_seconds": grace,
        "prearm_callback_sequence": _column(final_prearm_rows, 0, "<i8"),
        "prearm_callback_start_frames": _column(final_prearm_rows, 1, "<i8"),
        "prearm_callback_frame_counts": _column(final_prearm_rows, 2, "<i8"),
        "prearm_input_buffer_adc_time": _column(final_prearm_rows, 3, "<f8"),
        "prearm_output_buffer_dac_time": _column(final_prearm_rows, 4, "<f8"),
        "prearm_callback_current_time": _column(final_prearm_rows, 5, "<f8"),
        "prearm_callback_status_bitmask": prearm_masks,
        "prearm_callback_count": len(final_prearm_rows),
        "prearm_output_zero_observed": prearm_zero_observed,
        "planned_callback_sequence": planned_sequence,
        "planned_callback_start_frames": planned_starts,
        "planned_callback_frame_counts": planned_counts,
        "planned_input_buffer_adc_time": _column(final_planned_rows, 3, "<f8"),
        "planned_output_buffer_dac_time": _column(final_planned_rows, 4, "<f8"),
        "planned_callback_current_time": _column(final_planned_rows, 5, "<f8"),
        "planned_callback_status_bitmask": planned_masks,
        "xrun_count": int(np.count_nonzero(planned_masks & STATUS_XRUN_MASK) + np.count_nonzero(prearm_masks & STATUS_XRUN_MASK)),
        "status_present_count": int(np.count_nonzero(planned_masks & STATUS_PRESENT) + np.count_nonzero(prearm_masks & STATUS_PRESENT)),
        "captured_frames": int(np.count_nonzero(capture_valid)),
        "submitted_frames": int(np.count_nonzero(submitted_valid)),
        "callback_zero_attempt_count": zero_attempts,
        "callback_zero_confirmed_count": zero_confirmed,
        "callback_planned_assignment_attempt_count": final_assignment_attempts,
        "callback_planned_assignment_confirmed_count": final_assignment_confirmed,
        "possible_nonzero_output_after_failed_assignment": final_possible_nonzero,
        "planned_callback_sequence_contiguous": sequence_contiguous,
        "planned_callback_start_frames_contiguous": starts_contiguous,
        "planned_callback_frame_counts_exact": counts_exact,
        "capture_valid_all_true": capture_all,
        "submitted_valid_all_true": submitted_all,
        "full_frame_accounting_valid": full_accounting,
        "actual_matches_planned_application_buffer": actual_matches,
        "actual_submitted_pcm_hash_eligible": hash_eligible,
        "actual_submitted_pcm_sha256": _array_sha256(actual) if hash_eligible else None,
        "completed": final_completed,
        "callback_error": final_callback_error,
        "canonical_invalid_reasons": final_invalid,
        "stream_constructor_error": _error_text(stream_constructor_error),
        "stream_start_error": _error_text(stream_start_error),
        "post_start_pre_arm_check_error": _error_text(arm_check_error),
        "watchdog_error": _error_text(watchdog_error),
        "stream_stop_error": _error_text(stop_error),
        "stream_abort_error": _error_text(abort_error),
        "stream_close_error": _error_text(close_error),
        "on_output_closed_error": _error_text(close_notice_error),
        "termination_signals": [int(item) for item in signal_scope.received],
        "stream_start_returned_without_exception": start_returned,
        "stream_stop_attempted": stop_attempted,
        "stream_stop_returned_without_exception": stop_returned,
        "stream_abort_attempted": abort_attempted,
        "stream_abort_returned_without_exception": abort_returned,
        "stream_close_attempted": close_attempted,
        "stream_close_returned_without_exception": close_returned,
        "normal_stop_completed": bool(final_completed and arm_check_passed and stop_returned and close_returned and not faults),
        "faults": [
            {"stage": stage, "exception_type": type(error).__name__, "message": str(error)}
            for stage, error in faults
        ],
    }
    if faults:
        raise S32DisarmedDuplexCaptureFailure(
            "; ".join(f"{stage}={type(error).__name__}: {error}" for stage, error in faults),
            captured.copy(), actual.copy(), capture_valid.copy(), submitted_valid.copy(), telemetry,
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
    "S32DisarmedDuplexCaptureFailure",
    "TELEMETRY_SCHEMA",
    "capture_disarmed_planned_s32_duplex",
]
