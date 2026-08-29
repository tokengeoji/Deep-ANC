from __future__ import annotations

import signal

import numpy as np
import pytest

from deep_anc.audio_duplex_v5 import STATUS_INPUT_OVERFLOW
from deep_anc.audio_zero_duplex import (
    ZERO_DUPLEX_TELEMETRY_SCHEMA,
    ZeroDuplexCaptureFailure,
    capture_zero_duplex,
)


class Stop(Exception):
    pass


class Abort(Exception):
    pass


class Overflow:
    input_overflow = True

    def __bool__(self) -> bool:
        return True


class Backend:
    CallbackStop = Stop
    CallbackAbort = Abort

    def __init__(self, **values):
        self.__dict__.update(values)
        self.calls: list[object] = []
        self.outputs: list[np.ndarray] = []
        self.kwargs = {}

    def Stream(self, **kwargs):  # noqa: ANN003, ANN202
        self.kwargs = kwargs
        self.calls.append("stream_constructor")
        if getattr(self, "constructor_error", False):
            raise RuntimeError("constructor failed")
        outer = self

        class Stream:
            def start(self):
                outer.calls.append("start")
                if getattr(outer, "start_error", False):
                    raise RuntimeError("start failed")
                for index in range(getattr(outer, "blocks", 4)):
                    frames = getattr(outer, "frames", 256)
                    source = np.full(
                        (frames, 2), index + 1, dtype=getattr(outer, "in_dtype", "<i4")
                    )
                    sink = np.full(
                        getattr(outer, "out_shape", (frames, 2)),
                        77,
                        dtype=getattr(outer, "out_dtype", "<i4"),
                    )
                    outer.outputs.append(sink)
                    base = float(index)
                    time_info = {
                        "inputBufferAdcTime": base + 0.1,
                        "outputBufferDacTime": base + 0.2,
                        "currentTime": base + 0.3,
                    }
                    status = Overflow() if getattr(outer, "xrun", False) and index == 1 else None
                    try:
                        kwargs["callback"](source, sink, frames, time_info, status)
                    except (Stop, Abort):
                        break

            def stop(self, *, ignore_errors):
                outer.calls.append(("stop", ignore_errors))
                if getattr(outer, "signal_during_stop", None) is not None:
                    signal.raise_signal(outer.signal_during_stop)
                if getattr(outer, "stop_error", False):
                    raise RuntimeError("stop failed")

            def abort(self, *, ignore_errors):
                outer.calls.append(("abort", ignore_errors))
                if getattr(outer, "signal_during_abort", None) is not None:
                    signal.raise_signal(outer.signal_during_abort)
                if getattr(outer, "abort_error", False):
                    raise RuntimeError("abort failed")

            def close(self, *, ignore_errors):
                outer.calls.append(("close", ignore_errors))
                if getattr(outer, "signal_during_close", None) is not None:
                    signal.raise_signal(outer.signal_during_close)
                if getattr(outer, "close_error", False):
                    raise RuntimeError("close failed")

        return Stream()


def run(backend: Backend, **kwargs):  # noqa: ANN003, ANN201
    return capture_zero_duplex(
        backend,
        total_frames=1024,
        input_device=5,
        output_device=4,
        **kwargs,
    )


def test_normal_callback_overwrites_nonzero_sink_with_exact_int32_zero() -> None:
    backend = Backend()
    captured, telemetry = run(backend)

    assert backend.kwargs["dtype"] == ("int32", "int32")
    assert backend.kwargs["device"] == (5, 4)
    assert backend.kwargs["dither_off"] is True
    assert backend.kwargs["prime_output_buffers_using_stream_callback"] is False
    assert all(np.count_nonzero(block) == 0 for block in backend.outputs)
    assert telemetry["schema"] == ZERO_DUPLEX_TELEMETRY_SCHEMA
    assert telemetry["authority"] == "zero_duplex_transport_only_no_sample_identity"
    assert telemetry["hardware_sample_slip_authority"] is False
    assert telemetry["portaudio_timestamp_authority"] is False
    assert telemetry["portaudio_application_buffer_only"] is True
    assert telemetry["output_zero_scope"] == "portaudio_application_callback_buffer_only"
    assert telemetry["physical_output_zero_authority"] is False
    assert telemetry["electrical_output_zero_authority"] is False
    assert telemetry["acoustic_output_zero_authority"] is False
    assert telemetry["actual_submitted_nonzero_count"] == 0
    assert telemetry["callback_zero_attempt_count"] == 4
    assert telemetry["callback_zero_confirmed_count"] == 4
    assert np.count_nonzero(telemetry["actual_submitted_pcm"]) == 0
    assert np.all(telemetry["capture_valid_mask"])
    assert np.all(telemetry["submitted_valid_mask"])
    assert telemetry["full_frame_accounting_valid"] is True
    assert telemetry["actual_submitted_pcm_hash_eligible"] is True
    assert telemetry["application_buffer_zero_submission_complete"] is True
    assert telemetry["normal_stop_completed"] is True
    assert telemetry["stream_close_returned_without_exception"] is True
    assert "output_stop_confirmed" not in telemetry
    assert telemetry["xrun_count"] == 0
    assert captured.dtype == np.dtype("<i4")


@pytest.mark.parametrize(
    "name,value",
    [
        ("total_frames", 0),
        ("total_frames", 257),
        ("total_frames", True),
        ("input_device", -1),
        ("input_device", "5"),
        ("output_device", True),
    ],
)
def test_invalid_contract_never_constructs_stream(name: str, value: object) -> None:
    backend = Backend()
    arguments = {"total_frames": 1024, "input_device": 5, "output_device": 4}
    arguments[name] = value
    with pytest.raises((TypeError, ValueError)):
        capture_zero_duplex(backend, **arguments)
    assert backend.calls == []
    assert backend.outputs == []


def test_pre_open_failure_never_constructs_stream() -> None:
    backend = Backend()
    ticks = iter([1.0, 2.0])
    with pytest.raises(ZeroDuplexCaptureFailure, match="physical barrier missing") as caught:
        run(
            backend,
            pre_open_check=lambda: (_ for _ in ()).throw(
                RuntimeError("physical barrier missing")
            ),
            monotonic=lambda: next(ticks),
        )
    assert backend.calls == []
    assert backend.outputs == []
    assert caught.value.telemetry["submitted_frames"] == 0
    assert caught.value.telemetry["actual_submitted_nonzero_count"] == 0


def test_bad_frame_count_is_zeroed_before_abort() -> None:
    backend = Backend(frames=128)
    with pytest.raises(ZeroDuplexCaptureFailure, match="exact 256") as caught:
        run(backend)
    assert np.count_nonzero(backend.outputs[0]) == 0
    assert caught.value.telemetry["submitted_frames"] == 0
    assert caught.value.telemetry["callback_zero_attempt_count"] == 1
    assert caught.value.telemetry["callback_zero_confirmed_count"] == 1
    assert ("abort", False) in backend.calls
    assert ("close", False) in backend.calls


def test_wrong_output_dtype_is_still_zeroed_then_rejected() -> None:
    backend = Backend(out_dtype="<i2")
    with pytest.raises(ZeroDuplexCaptureFailure, match="output은 exact <i4") as caught:
        run(backend)
    assert np.count_nonzero(backend.outputs[0]) == 0
    assert caught.value.telemetry["submitted_frames"] == 0


def test_xrun_preserves_only_zero_output_and_fails_closed() -> None:
    backend = Backend(xrun=True)
    with pytest.raises(ZeroDuplexCaptureFailure, match="canonical-invalid") as caught:
        run(backend)
    telemetry = caught.value.telemetry
    assert telemetry["completed"] is True
    assert telemetry["xrun_count"] == 1
    assert int(telemetry["callback_status_bitmask"][1]) & STATUS_INPUT_OVERFLOW
    assert np.count_nonzero(caught.value.actual_submitted_pcm) == 0
    assert np.all(caught.value.submitted_valid_mask)


def test_stream_hooks_are_ordered_after_start_and_after_close() -> None:
    backend = Backend()
    events: list[object] = []
    run(
        backend,
        on_stream_started=lambda: events.append("started"),
        on_output_closed=lambda closed: events.append(("closed", closed)),
    )
    assert events == ["started", ("closed", True)]
    assert backend.calls[-2:] == [("stop", False), ("close", False)]


def test_constructor_and_output_closed_faults_are_both_preserved() -> None:
    backend = Backend(constructor_error=True)

    def fail_notice(_closed: bool) -> None:
        raise RuntimeError("notice failed")

    with pytest.raises(ZeroDuplexCaptureFailure) as caught:
        run(backend, on_output_closed=fail_notice)

    assert "constructor failed" in str(caught.value)
    assert "notice failed" in str(caught.value)
    telemetry = caught.value.telemetry
    assert telemetry["stream_constructor_error"] == "RuntimeError: constructor failed"
    assert telemetry["on_output_closed_error"] == "RuntimeError: notice failed"
    assert telemetry["stream_close_attempted"] is False
    assert telemetry["stream_close_returned_without_exception"] is False
    assert [item["stage"] for item in telemetry["faults"]] == [
        "stream_constructor",
        "on_output_closed",
    ]


def test_start_fault_still_aborts_and_closes() -> None:
    backend = Backend(start_error=True)
    with pytest.raises(ZeroDuplexCaptureFailure, match="start failed") as caught:
        run(backend)

    telemetry = caught.value.telemetry
    assert telemetry["stream_start_error"] == "RuntimeError: start failed"
    assert telemetry["stream_abort_returned_without_exception"] is True
    assert telemetry["stream_close_returned_without_exception"] is True
    assert backend.calls[-2:] == [("abort", False), ("close", False)]


def test_partial_callback_masks_cannot_authorize_zero_pcm_hash() -> None:
    backend = Backend(blocks=1)
    ticks = iter([0.0, 0.0, 0.0, 10.0, 11.0])

    with pytest.raises(ZeroDuplexCaptureFailure, match="watchdog") as caught:
        run(
            backend,
            watchdog_grace_seconds=0.1,
            monotonic=lambda: next(ticks),
            sleep=lambda _seconds: None,
        )

    telemetry = caught.value.telemetry
    assert telemetry["watchdog_error"] == "TimeoutError: zero duplex watchdog 초과"
    assert telemetry["captured_frames"] == 256
    assert telemetry["submitted_frames"] == 256
    assert np.all(caught.value.capture_valid_mask[:256])
    assert not np.any(caught.value.capture_valid_mask[256:])
    assert np.all(caught.value.submitted_valid_mask[:256])
    assert not np.any(caught.value.submitted_valid_mask[256:])
    assert telemetry["capture_valid_all_true"] is False
    assert telemetry["submitted_valid_all_true"] is False
    assert telemetry["full_frame_accounting_valid"] is False
    assert telemetry["actual_submitted_pcm_hash_eligible"] is False
    assert telemetry["application_buffer_zero_submission_complete"] is False


def test_stop_abort_close_and_notice_faults_are_all_aggregated() -> None:
    backend = Backend(stop_error=True, abort_error=True, close_error=True)

    def fail_notice(_closed: bool) -> None:
        raise RuntimeError("notice failed")

    with pytest.raises(ZeroDuplexCaptureFailure) as caught:
        run(backend, on_output_closed=fail_notice)

    message = str(caught.value)
    assert all(
        item in message
        for item in ("stop failed", "abort failed", "close failed", "notice failed")
    )
    telemetry = caught.value.telemetry
    assert telemetry["stream_stop_error"] == "RuntimeError: stop failed"
    assert telemetry["stream_abort_error"] == "RuntimeError: abort failed"
    assert telemetry["stream_close_error"] == "RuntimeError: close failed"
    assert telemetry["on_output_closed_error"] == "RuntimeError: notice failed"
    assert [item["stage"] for item in telemetry["faults"]] == [
        "stream_stop",
        "stream_abort",
        "stream_close",
        "on_output_closed",
    ]


def test_callback_fault_is_not_hidden_by_abort_close_or_notice_faults() -> None:
    backend = Backend(frames=128, abort_error=True, close_error=True)

    def fail_notice(_closed: bool) -> None:
        raise RuntimeError("notice failed")

    with pytest.raises(ZeroDuplexCaptureFailure) as caught:
        run(backend, on_output_closed=fail_notice)

    message = str(caught.value)
    assert all(
        item in message
        for item in (
            "callback frames는 exact 256",
            "abort failed",
            "close failed",
            "notice failed",
        )
    )
    assert [item["stage"] for item in caught.value.telemetry["faults"]] == [
        "callback",
        "stream_abort",
        "stream_close",
        "on_output_closed",
    ]


@pytest.mark.parametrize("phase", ["stop", "close"])
def test_sigterm_during_cleanup_is_deferred_until_close_returns(phase: str) -> None:
    previous = signal.getsignal(signal.SIGTERM)
    backend = Backend(**{f"signal_during_{phase}": signal.SIGTERM})

    with pytest.raises(ZeroDuplexCaptureFailure) as caught:
        run(backend)

    telemetry = caught.value.telemetry
    assert signal.getsignal(signal.SIGTERM) == previous
    assert telemetry["termination_signals"] == [int(signal.SIGTERM)]
    assert telemetry["termination_signal"] == int(signal.SIGTERM)
    assert telemetry["termination_exit_code"] == 128 + int(signal.SIGTERM)
    assert telemetry["stream_close_returned_without_exception"] is True
    assert backend.calls[-1] == ("close", False)
    assert "exit 143" in str(caught.value)
