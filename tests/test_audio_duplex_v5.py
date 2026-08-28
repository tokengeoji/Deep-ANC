import numpy as np
import pytest

from deep_anc.audio_duplex_v5 import (
    DuplexCaptureFailure,
    STATUS_INPUT_OVERFLOW,
    capture_duplex_v5,
)


class Stop(Exception):
    pass


class Abort(Exception):
    pass


class Status:
    input_overflow = True

    def __bool__(self):
        return True


class Backend:
    CallbackStop = Stop
    CallbackAbort = Abort

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.calls = []
        self.kwargs = {}
        self.outputs = []

    def Stream(self, **kwargs):
        self.kwargs = kwargs
        outer = self

        class Stream:
            def start(self):
                outer.calls.append("start")
                for index in range(getattr(outer, "blocks", 4)):
                    frames = getattr(outer, "frames", 256)
                    input_data = np.full(
                        (frames, 2),
                        index + 1,
                        dtype=getattr(outer, "in_dtype", "<i4"),
                    )
                    output_data = np.full(
                        getattr(outer, "out_shape", (frames, 2)),
                        77,
                        dtype=getattr(outer, "out_dtype", "<i2"),
                    )
                    outer.outputs.append(output_data)
                    time_value = float(index)
                    time_info = {
                        "inputBufferAdcTime": time_value + 0.1,
                        "outputBufferDacTime": time_value + 0.2,
                        "currentTime": time_value + 0.3,
                    }
                    if getattr(outer, "bad_time", None) == "nan" and index == 1:
                        time_info["currentTime"] = np.nan
                    if getattr(outer, "bad_time", None) == "back" and index == 1:
                        time_info = {key: -1.0 for key in time_info}
                    status = (
                        Status()
                        if getattr(outer, "xrun", False) and index == 1
                        else None
                    )
                    try:
                        kwargs["callback"](
                            input_data,
                            output_data,
                            frames,
                            time_info,
                            status,
                        )
                    except (Stop, Abort):
                        break

            def stop(self, *, ignore_errors):
                outer.calls.append(("stop", ignore_errors))
                if getattr(outer, "stop_error", False):
                    raise RuntimeError("stop failed")

            def abort(self, *, ignore_errors):
                outer.calls.append(("abort", ignore_errors))
                if getattr(outer, "abort_error", False):
                    raise RuntimeError("abort failed")

            def close(self, *, ignore_errors):
                outer.calls.append(("close", ignore_errors))
                if getattr(outer, "close_error", False):
                    raise RuntimeError("close failed")

        return Stream()


def pcm():
    return np.arange(2048, dtype="<i2").reshape(1024, 2)


def run(backend, **kwargs):
    return capture_duplex_v5(
        backend,
        submitted_pcm=pcm(),
        input_device=1,
        output_device=2,
        **kwargs,
    )


def test_normal_exact_zero_status_and_stop():
    backend = Backend()
    raw, telemetry = run(backend)
    assert backend.kwargs["prime_output_buffers_using_stream_callback"] is False
    assert (
        backend.kwargs["samplerate"],
        backend.kwargs["blocksize"],
        backend.kwargs["latency"],
        backend.kwargs["channels"],
    ) == (48_000, 256, ("low", "low"), (2, 2))
    assert np.all(telemetry["callback_status_bitmask"] == 0)
    assert np.array_equal(telemetry["actual_submitted_pcm"], pcm())
    assert backend.calls[-2:] == [("stop", False), ("close", False)]
    assert telemetry["hardware_sample_slip_authority"] is False
    assert raw.dtype == np.dtype("<i4")


def test_xrun_completes_then_invalid_with_actual_prefix():
    backend = Backend(xrun=True)
    with pytest.raises(DuplexCaptureFailure) as caught:
        run(backend)
    failure = caught.value
    assert failure.telemetry["completed"]
    assert failure.telemetry["xrun_count"] == 1
    assert int(failure.telemetry["callback_status_bitmask"][1]) & STATUS_INPUT_OVERFLOW
    assert np.all(failure.submitted_valid_mask)
    assert np.array_equal(failure.submitted_pcm, pcm())
    assert ("abort", False) in backend.calls
    assert ("close", False) in backend.calls


@pytest.mark.parametrize(
    "args,match,expected",
    [
        (("frames", 128), "exact 256", 0),
        (("out_dtype", "<i4"), "output은 exact <i2", 0),
        (("out_shape", (256, 1)), "output은 exact <i2", 0),
        (("bad_time", "nan"), "finite", 256),
        (("bad_time", "back"), "strict-monotonic", 256),
    ],
)
def test_bad_callback_atomic_partial(args, match, expected):
    backend = Backend(**{args[0]: args[1]})
    with pytest.raises(DuplexCaptureFailure) as caught:
        run(backend)
    failure = caught.value
    assert match in str(failure)
    assert failure.telemetry["captured_frames"] == expected
    assert failure.telemetry["submitted_frames"] == expected
    assert np.count_nonzero(failure.capture_valid_mask) == expected
    assert ("abort", False) in backend.calls
    assert ("close", False) in backend.calls
    if args[0] not in {"out_dtype", "out_shape"}:
        assert np.all(backend.outputs[-1] == 0)


def test_external_plan_copy_and_watchdog_prefix():
    source = pcm()
    backend = Backend()
    _, telemetry = capture_duplex_v5(
        backend,
        submitted_pcm=source,
        input_device=1,
        output_device=2,
        pre_open_check=lambda: source.fill(0),
    )
    assert np.array_equal(telemetry["actual_submitted_pcm"], pcm())

    backend = Backend(blocks=2)
    ticks = iter([0.0, 99.0])
    with pytest.raises(DuplexCaptureFailure) as caught:
        run(backend, monotonic=lambda: next(ticks), sleep=lambda _: None)
    failure = caught.value
    assert failure.telemetry["captured_frames"] == 512
    assert np.all(failure.submitted_valid_mask[:512])
    assert not np.any(failure.submitted_valid_mask[512:])


def test_cleanup_errors_not_ignored():
    backend = Backend(stop_error=True, abort_error=True, close_error=True)
    with pytest.raises(DuplexCaptureFailure) as caught:
        run(backend)
    assert all(
        call in backend.calls
        for call in (("stop", False), ("abort", False), ("close", False))
    )
    assert all(
        message in str(caught.value)
        for message in ("stop failed", "abort failed", "close failed")
    )


def test_output_buffer_write_failure_is_atomic():
    backend = Backend()
    original_stream = backend.Stream

    def stream(**kwargs):
        instance = original_stream(**kwargs)

        class FailingOutput:
            def __init__(self):
                self.array = np.zeros((256, 2), dtype="<i2")

            def __array__(self, dtype=None, copy=None):
                return self.array

            def __setitem__(self, key, value):
                raise RuntimeError("sink write failed")

        def start():
            backend.calls.append("start")
            try:
                kwargs["callback"](
                    np.ones((256, 2), dtype="<i4"),
                    FailingOutput(),
                    256,
                    {
                        "inputBufferAdcTime": 0.1,
                        "outputBufferDacTime": 0.2,
                        "currentTime": 0.3,
                    },
                    None,
                )
            except Abort:
                pass

        instance.start = start
        return instance

    backend.Stream = stream
    with pytest.raises(DuplexCaptureFailure) as caught:
        run(backend)
    failure = caught.value
    assert "sink write failed" in str(failure)
    assert failure.telemetry["captured_frames"] == 0
    assert failure.telemetry["submitted_frames"] == 0
    assert not np.any(failure.capture_valid_mask)
    assert not np.any(failure.submitted_valid_mask)
    assert "output_silence_not_confirmed_on_callback_failure" in failure.telemetry[
        "canonical_invalid_reasons"
    ]
