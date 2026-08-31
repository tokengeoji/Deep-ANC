import numpy as np
import pytest

from deep_anc.audio_duplex_stage2 import (
    OUTPUT_MASTER_TELEMETRY_SCHEMA,
    OutputMasterCaptureFailure,
    capture_output_master_stage2,
)


class Stop(Exception):
    pass


class Abort(Exception):
    pass


class InputOverflow:
    input_overflow = True

    def __bool__(self):
        return True


class FakeBackend:
    CallbackStop = Stop
    CallbackAbort = Abort

    def __init__(
        self,
        *,
        output_blocks: int = 4,
        pre_roll_blocks: int = 2,
        input_blocks_per_output: int = 1,
        post_roll_blocks: int = 2,
        input_timestamp_step_frames: float = 256.0,
        input_status_at: int | None = None,
        output_bad_frames_at: int | None = None,
        output_close_error: bool = False,
    ) -> None:
        self.output_blocks = output_blocks
        self.pre_roll_blocks = pre_roll_blocks
        self.input_blocks_per_output = input_blocks_per_output
        self.post_roll_blocks = post_roll_blocks
        self.input_timestamp_step_frames = input_timestamp_step_frames
        self.input_status_at = input_status_at
        self.output_bad_frames_at = output_bad_frames_at
        self.output_close_error = output_close_error
        self.calls: list[object] = []
        self.input_kwargs: dict[str, object] = {}
        self.output_kwargs: dict[str, object] = {}
        self.output_buffers: list[np.ndarray] = []
        self.input_callback_index = 0
        self.output_callback_index = 0
        self.input_callback_stopped = False

    def _pump_input(self, blocks: int) -> None:
        callback = self.input_kwargs["callback"]
        for _ in range(blocks):
            if self.input_callback_stopped:
                return
            index = self.input_callback_index
            data = np.full((256, 2), index + 1, dtype="<i4")
            time_value = index * self.input_timestamp_step_frames / 48_000
            status = InputOverflow() if index == self.input_status_at else None
            self.input_callback_index += 1
            try:
                callback(
                    data,
                    256,
                    {
                        "inputBufferAdcTime": 10.0 + time_value,
                        "currentTime": 20.0 + time_value,
                    },
                    status,
                )
            except Stop:
                self.input_callback_stopped = True
                return
            except Abort:
                self.input_callback_stopped = True
                return

    def InputStream(self, **kwargs):
        self.calls.append("input_created")
        self.input_kwargs = kwargs
        outer = self

        class InputStream:
            def start(self):
                outer.calls.append("input_start")
                outer._pump_input(outer.pre_roll_blocks)

            def stop(self, *, ignore_errors):
                outer.calls.append(("input_stop", ignore_errors))

            def abort(self, *, ignore_errors):
                outer.calls.append(("input_abort", ignore_errors))

            def close(self, *, ignore_errors):
                outer.calls.append(("input_close", ignore_errors))

        return InputStream()

    def OutputStream(self, **kwargs):
        self.calls.append("output_created")
        self.output_kwargs = kwargs
        outer = self

        class OutputStream:
            def start(self):
                outer.calls.append("output_start")
                callback = kwargs["callback"]
                for _ in range(outer.output_blocks):
                    index = outer.output_callback_index
                    frames = 128 if index == outer.output_bad_frames_at else 256
                    out = np.full((256, 2), 77, dtype="<i2")
                    outer.output_buffers.append(out)
                    time_value = index * 256 / 48_000
                    outer.output_callback_index += 1
                    try:
                        callback(
                            out,
                            frames,
                            {
                                "outputBufferDacTime": 30.0 + time_value,
                                "currentTime": 40.0 + time_value,
                            },
                            None,
                        )
                    except Stop:
                        break
                    except Abort:
                        break
                    outer._pump_input(outer.input_blocks_per_output)

            def stop(self, *, ignore_errors):
                outer.calls.append(("output_stop", ignore_errors))
                outer._pump_input(outer.post_roll_blocks)

            def abort(self, *, ignore_errors):
                outer.calls.append(("output_abort", ignore_errors))

            def close(self, *, ignore_errors):
                outer.calls.append(("output_close", ignore_errors))
                if outer.output_close_error:
                    raise RuntimeError("output close failed")

        return OutputStream()


def planned() -> np.ndarray:
    return np.arange(2_048, dtype="<i2").reshape(1_024, 2)


def run(backend: FakeBackend, **kwargs):
    return capture_output_master_stage2(
        backend,
        submitted_pcm=planned(),
        input_device=5,
        output_device=24,
        pre_roll_frames=512,
        post_roll_frames=512,
        **kwargs,
    )


def test_output_callback_alone_owns_exact_submitted_cursor_with_pre_and_post_roll():
    backend = FakeBackend()
    closed: list[bool] = []
    captured, telemetry = run(backend, on_output_closed=closed.append)

    assert telemetry["schema"] == OUTPUT_MASTER_TELEMETRY_SCHEMA
    assert telemetry["transport"] == (
        "independent_input_output_streams_output_clock_master"
    )
    assert telemetry["output_clock_owner"] == "outputstream_callback_only"
    assert telemetry["input_role"] == "raw_witness_only_never_output_pacing"
    assert telemetry["cross_clock_timestamp_alignment_used"] is False
    assert telemetry["input_output_frame_identity_claimed"] is False
    assert telemetry["hardware_sample_slip_authority"] is False
    assert telemetry["pre_roll_observed_input_frames"] == 512
    assert telemetry["input_frame_cursor_at_output_start"] == 512
    assert telemetry["input_frame_cursor_at_output_complete"] == 1_280
    assert telemetry["post_roll_observed_input_frames"] == 512
    assert telemetry["captured_input_frames"] == 1_792
    assert telemetry["canonical_output_frames"] == 1_024
    assert telemetry["submitted_output_frames"] == 1_024
    assert telemetry["completed"] is True
    assert telemetry["failure_events"] == []
    assert telemetry["legacy_combined_duplex_used"] is False
    assert np.array_equal(telemetry["actual_submitted_pcm"], planned())
    assert np.all(telemetry["submitted_valid_mask"])
    assert np.all(telemetry["capture_valid_mask"])
    assert captured.shape == (1_792, 2)
    assert captured.dtype == np.dtype("<i4")
    assert np.array_equal(
        telemetry["output_callback_start_frames"], np.arange(4) * 256
    )
    assert np.array_equal(
        np.concatenate(backend.output_buffers, axis=0), planned()
    )
    assert backend.calls.index("input_start") < backend.calls.index("output_created")
    assert backend.output_kwargs["prime_output_buffers_using_stream_callback"] is False
    assert backend.input_kwargs["device"] == 5
    assert backend.output_kwargs["device"] == 24
    assert backend.calls.count(("output_close", False)) == 1
    assert closed == [True]


def test_input_callback_rate_never_changes_output_bytes_or_cursor():
    backend = FakeBackend(
        input_blocks_per_output=2,
        post_roll_blocks=2,
        input_timestamp_step_frames=257.25,
    )
    captured, telemetry = run(backend)

    assert captured.shape[0] != planned().shape[0]
    assert telemetry["captured_input_frames"] == 2_560
    assert telemetry["canonical_output_frames"] == 1_024
    assert np.array_equal(telemetry["actual_submitted_pcm"], planned())
    assert np.array_equal(
        telemetry["output_callback_start_frames"], [0, 256, 512, 768]
    )
    assert telemetry["input_output_frame_identity_claimed"] is False
    assert not np.isclose(
        np.diff(telemetry["input_buffer_adc_time"])[0],
        np.diff(telemetry["output_buffer_dac_time"])[0],
    )


def test_input_status_during_preroll_preserves_partial_raw_and_never_opens_output():
    backend = FakeBackend(input_status_at=1)
    with pytest.raises(OutputMasterCaptureFailure) as caught:
        run(backend)

    failure = caught.value
    assert failure.captured_pcm.shape == (256, 2)
    assert np.count_nonzero(failure.capture_valid_mask) == 256
    assert np.count_nonzero(failure.submitted_valid_mask) == 0
    assert failure.telemetry["completed"] is False
    assert failure.telemetry["failure_events"][0]["role"] == "input"
    assert "status" in failure.telemetry["failure_events"][0]["message"]
    assert "output_created" not in backend.calls
    assert ("input_abort", False) in backend.calls
    assert ("input_close", False) in backend.calls


def test_output_callback_failure_zero_fills_bad_block_and_preserves_both_prefixes():
    backend = FakeBackend(output_bad_frames_at=1)
    with pytest.raises(OutputMasterCaptureFailure) as caught:
        run(backend)

    failure = caught.value
    assert np.count_nonzero(failure.submitted_valid_mask) == 256
    assert np.array_equal(failure.submitted_pcm[:256], planned()[:256])
    assert np.count_nonzero(failure.submitted_pcm[256:]) == 0
    assert np.all(backend.output_buffers[1] == 0)
    assert failure.telemetry["output_callback_start_frames"].tolist() == [0]
    assert failure.telemetry["failure_events"][0]["role"] == "output"
    assert ("output_abort", False) in backend.calls
    assert ("input_abort", False) in backend.calls


class AdvancingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.1
        return self.value


def test_missing_post_roll_fails_after_exact_output_close_without_hanging():
    backend = FakeBackend(post_roll_blocks=0)
    closed: list[bool] = []
    clock = AdvancingClock()
    with pytest.raises(OutputMasterCaptureFailure) as caught:
        run(
            backend,
            monotonic=clock,
            sleep=lambda _seconds: None,
            watchdog_grace_seconds=0.01,
            on_output_closed=closed.append,
        )

    failure = caught.value
    assert np.all(failure.submitted_valid_mask)
    assert np.array_equal(failure.submitted_pcm, planned())
    assert failure.telemetry["post_roll_observed_input_frames"] == 0
    assert any(
        row["role"] == "watchdog" and "post-roll" in row["message"]
        for row in failure.telemetry["failure_events"]
    )
    assert backend.calls.index(("output_close", False)) < backend.calls.index(
        ("input_abort", False)
    )
    assert closed == [True]


@pytest.mark.parametrize(
    "name,value",
    [
        ("input_device", -1),
        ("output_device", True),
        ("pre_roll_frames", 257),
        ("post_roll_frames", 0),
    ],
)
def test_invalid_split_contract_fails_before_any_stream(name, value):
    backend = FakeBackend()
    arguments = {
        "submitted_pcm": planned(),
        "input_device": 5,
        "output_device": 24,
        "pre_roll_frames": 512,
        "post_roll_frames": 512,
        name: value,
    }
    with pytest.raises(ValueError):
        capture_output_master_stage2(backend, **arguments)
    assert backend.calls == []


def test_output_close_failure_is_partial_failure_not_success():
    backend = FakeBackend(output_close_error=True)
    with pytest.raises(OutputMasterCaptureFailure) as caught:
        run(backend)

    failure = caught.value
    assert np.all(failure.submitted_valid_mask)
    assert failure.telemetry["completed"] is False
    assert any(
        row["role"] == "output_close" for row in failure.telemetry["failure_events"]
    )
    assert ("input_abort", False) in backend.calls
