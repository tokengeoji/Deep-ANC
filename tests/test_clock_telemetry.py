from __future__ import annotations

import json
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import deep_anc.realtime.clock_telemetry as clock_module
import deep_anc.realtime.run_realtime as runtime_module
from deep_anc.realtime.clock_telemetry import (
    ClockTelemetryRecorder,
    RuntimeCounterSnapshot,
    bind_recording_to_clock_receipt,
    payload_sha256,
    write_clock_receipt_exclusive,
)
from deep_anc.realtime.ring_buffer import SPSCRing
from deep_anc.realtime.run_realtime import RealtimeANC


FS = 48_000
BLOCK = 256


def _recorder(**updates) -> ClockTelemetryRecorder:
    values = {
        "sample_rate": FS,
        "block_size": BLOCK,
        "input_device": "APE:1",
        "output_device": "Audio:0",
        "allowed_input_backlog_samples": BLOCK,
        "allowed_output_backlog_samples": BLOCK,
    }
    values.update(updates)
    return ClockTelemetryRecorder(**values)


def _time_info(index: int, *, offset_samples: float = 0.0) -> dict[str, float]:
    seconds = (index * BLOCK + offset_samples) / FS
    return {
        "inputBufferAdcTime": 10.0 + seconds,
        "outputBufferDacTime": 10.1 + seconds,
        "currentTime": 10.2 + seconds,
    }


def _finish(
    recorder: ClockTelemetryRecorder,
    index: int,
    *,
    frames: int = BLOCK,
    time_info=None,
    status=False,
    snapshot: RuntimeCounterSnapshot | None = None,
) -> None:
    token = recorder.begin_callback(
        frames=frames,
        time_info=_time_info(index) if time_info is None else time_info,
        status=status,
    )
    recorder.finish_callback(token, snapshot=snapshot or RuntimeCounterSnapshot())


def test_complete_exact_callbacks_are_inconclusive_never_false_pass():
    recorder = _recorder()
    for index in range(3):
        _finish(recorder, index)

    receipt = recorder.build_receipt()

    assert receipt["structural_status"] == "PASS"
    assert receipt["authority_status"] == "INCONCLUSIVE"
    assert "PASS" not in {receipt["authority_status"]}
    summary = receipt["callback_summary"]
    assert summary["callback_count"] == 3
    assert summary["completed_callback_count"] == 3
    assert summary["incomplete_callback_count"] == 0
    assert summary["pending_callback_count"] == 0
    assert summary["portaudio_status_callback_count"] == 0
    assert summary["callback_host_deadline_miss_count"] == 0
    assert summary["maximum_callback_host_duration_ns"] >= 0
    assert summary["application_observed_frames"] == 3 * BLOCK
    assert summary["application_observed_seconds"] == 3 * BLOCK / FS
    assert summary["stored_callback_record_count"] == 3
    assert summary["omitted_callback_record_count"] == 0
    assert [row["callback_start_frame"] for row in receipt["callbacks"]] == [
        0,
        BLOCK,
        2 * BLOCK,
    ]
    assert all(row["completed"] for row in receipt["callbacks"])
    assert receipt["clock_semantics"] == {
        "noise_and_cancel_outputs_share_one_output_stream_device_clock": True,
        "adc_dac_drift_is_not_noise_cancel_relative_output_phase": True,
        "callback_frame_counter_is_application_observed_not_physical_adc_proof": True,
    }
    assert "callback 호출 전에 버린 capture period" in receipt["authority_reasons"][-1]
    json.dumps(receipt, allow_nan=False)
    assert len(payload_sha256(receipt)) == 64


@pytest.mark.parametrize(
    ("second_time", "second_frames", "second_status", "issue_fragment"),
    [
        (
            {
                "inputBufferAdcTime": float("nan"),
                "outputBufferDacTime": 10.1 + BLOCK / FS,
                "currentTime": 10.2 + BLOCK / FS,
            },
            BLOCK,
            False,
            "inputBufferAdcTime_nonfinite",
        ),
        (_time_info(0), BLOCK, False, "not_strict_monotonic"),
        (_time_info(1, offset_samples=1.0), BLOCK, False, "unexpected_frame_step"),
        (_time_info(1), BLOCK - 1, False, "unexpected_callback_frame_count"),
        (_time_info(1), BLOCK, "input overflow", "portaudio_callback_status"),
    ],
)
def test_timestamp_frame_or_status_fault_blocks(
    second_time, second_frames, second_status, issue_fragment
):
    recorder = _recorder()
    _finish(recorder, 0)
    _finish(
        recorder,
        1,
        frames=second_frames,
        time_info=second_time,
        status=second_status,
    )

    receipt = recorder.build_receipt()

    assert receipt["authority_status"] == "BLOCKED"
    assert receipt["structural_status"] == "BLOCKED"
    assert any(issue_fragment in key for key in receipt["issue_counts"])


def test_incomplete_or_unfinished_callback_blocks_completion_authority():
    aborted = _recorder()
    token = aborted.begin_callback(frames=BLOCK, time_info=_time_info(0), status=False)
    aborted.abort_callback(token, error=RuntimeError("injected"))
    aborted_receipt = aborted.build_receipt()
    assert aborted_receipt["authority_status"] == "BLOCKED"
    assert aborted_receipt["callback_summary"]["incomplete_callback_count"] == 1
    assert aborted_receipt["callbacks"][0]["completed"] is False

    unfinished = _recorder()
    unfinished.begin_callback(frames=BLOCK, time_info=_time_info(0), status=False)
    unfinished_receipt = unfinished.build_receipt()
    assert unfinished_receipt["authority_status"] == "BLOCKED"
    assert unfinished_receipt["callback_summary"]["pending_callback_count"] == 1
    assert "callback_completion_missing" in unfinished_receipt["issue_counts"]
    assert unfinished.build_receipt() == unfinished_receipt

    empty = _recorder()
    first_empty = empty.build_receipt()
    assert empty.build_receipt() == first_empty


def test_ring_fallback_watchdog_and_excess_backlog_are_bound_and_blocked():
    recorder = _recorder()
    snapshot = RuntimeCounterSnapshot(
        xrun_count=1,
        deadline_miss_count=2,
        input_ring_drop_samples=3,
        output_ring_drop_samples=4,
        ring_add_samples=1,
        input_backlog_samples=BLOCK + 1,
        output_backlog_samples=BLOCK + 2,
        fallback_silence_blocks=1,
        watchdog_trip_counts={"runtime_handoff_backlog": 1},
    )
    _finish(recorder, 0, snapshot=snapshot)

    receipt = recorder.build_receipt(final_snapshot=snapshot)

    assert receipt["authority_status"] == "BLOCKED"
    assert receipt["runtime_counters_final"]["xrun_count"] == 1
    assert receipt["runtime_counters_final"]["deadline_miss_count"] == 2
    assert receipt["runtime_counters_final"]["fallback_silence_blocks"] == 1
    assert receipt["counter_semantics"][
        "deadline_and_fallback_are_distinct_not_duplicate_aliases"
    ] is True
    assert receipt["runtime_counters_final"]["ring_add_samples"] == 1
    assert receipt["maximum_input_backlog_samples"] == BLOCK + 1
    assert receipt["maximum_output_backlog_samples"] == BLOCK + 2
    reasons = "\n".join(receipt["authority_reasons"])
    assert "fallback_silence_blocks=1" in reasons
    assert "watchdog_trip_counts" in reasons
    assert "maximum_input_backlog_samples" in reasons


def test_callback_completion_past_block_deadline_is_blocked(monkeypatch):
    ticks = iter((1_000_000_000, 1_006_000_000))
    monkeypatch.setattr(clock_module.time, "monotonic_ns", lambda: next(ticks))
    recorder = _recorder()

    _finish(recorder, 0)
    receipt = recorder.build_receipt()

    assert receipt["authority_status"] == "BLOCKED"
    assert receipt["callback_summary"]["callback_host_deadline_miss_count"] == 1
    assert "callback_host_deadline_miss" in receipt["issue_counts"]


def test_inference_deadline_counter_is_not_the_fallback_counter(monkeypatch):
    ticks = iter((20.0, 20.006))
    monkeypatch.setattr(runtime_module.time, "perf_counter", lambda: next(ticks))
    anc = RealtimeANC.__new__(RealtimeANC)
    anc.cfg = {"engine": {}}
    anc.fs = FS
    anc.hop = BLOCK
    anc.reference = "digital"
    anc.in_ring = SPSCRing(4, BLOCK * 2)
    anc.out_ring = SPSCRing(1, BLOCK * 2)
    anc.in_ring.push(np.zeros((4, BLOCK), dtype=np.float32))
    anc.handoff_budget = SimpleNamespace(input_keep_backlog_samples=BLOCK)
    anc.step_times_ms = []
    anc._deadline_miss_blocks = 0
    anc._fallback_silence_blocks = 0
    anc.state = SimpleNamespace(
        quit_event=threading.Event(),
        reset_event=threading.Event(),
        messages=queue.Queue(),
    )

    def step(_ref, _err):
        anc.state.quit_event.set()
        return np.zeros(BLOCK, dtype=np.float32)

    anc.engine = SimpleNamespace(step=step)

    anc._inference_loop()

    assert anc._deadline_miss_blocks == 1
    assert anc._fallback_silence_blocks == 0


@pytest.mark.parametrize("occupied_suffix", [".npz", ".runtime_clock.json"])
def test_run_cli_rejects_existing_record_target_before_runtime_construction(
    tmp_path, monkeypatch, occupied_suffix
):
    base = tmp_path / "session"
    base.with_suffix(occupied_suffix).write_bytes(b"existing-authoritative-artifact")
    constructed = False

    def forbidden_constructor(*_args, **_kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("record target preflight 전에 runtime을 생성하면 안 됩니다")

    monkeypatch.setattr(runtime_module, "RealtimeANC", forbidden_constructor)

    with pytest.raises(FileExistsError, match="no-replace target"):
        runtime_module.run_cli({}, 1.0, str(base))

    assert constructed is False


def test_record_target_preflight_rejects_symlink_parent(tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        runtime_module._prepare_runtime_record_targets(linked_parent / "session")


def test_clock_receipt_is_no_replace_and_binds_recording(tmp_path):
    recorder = _recorder()
    _finish(recorder, 0)
    telemetry = recorder.build_receipt()
    recording = tmp_path / "session.npz"
    recording.write_bytes(b"immutable-runtime-npz")
    bundle = bind_recording_to_clock_receipt(
        telemetry,
        recording_path=recording,
        recording_sha256=("a" * 64),
    )
    target = tmp_path / "session.runtime_clock.json"

    path, first_sha = write_clock_receipt_exclusive(target, bundle)
    original = path.read_bytes()

    assert bundle["authority_status"] == "INCONCLUSIVE"
    assert bundle["runtime_clock_telemetry_sha256"] == payload_sha256(telemetry)
    assert len(first_sha) == 64
    with pytest.raises(FileExistsError):
        write_clock_receipt_exclusive(target, bundle)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="false PASS"):
        write_clock_receipt_exclusive(
            tmp_path / "false-pass.json", {"authority_status": "PASS"}
        )


class _PassThrough:
    def process(self, value):
        return value


class _Gate:
    def __init__(self, value: float):
        self.value = float(value)

    def process(self, frames):
        return np.full(int(frames), self.value, dtype=np.float32)

    def set_target(self, _value):
        return None


class _DigitalReferencePassThrough:
    def process(self, value):
        return value, value


class _Power:
    def update(self, _value):
        return 1.0e-4


def test_realtime_callback_binds_portaudio_time_info_without_opening_device():
    """RealtimeANC.__new__로 callback만 검증하며 sounddevice/PCM은 열지 않는다."""

    anc = RealtimeANC.__new__(RealtimeANC)
    anc.fs = FS
    anc.block = BLOCK
    anc.hop = BLOCK
    anc.ch_err = 0
    anc.ch_ref = 1
    anc.ch_noise = 0
    anc.ch_cancel = 1
    anc.reference = "digital"
    anc.clock_telemetry = _recorder()
    anc.xruns = 0
    anc._fallback_silence_blocks = 0
    anc._ring_add_samples = 0
    anc.err_dc = _PassThrough()
    anc.ref_dc = _PassThrough()
    anc.noise_gate = _Gate(1.0)
    anc.anc_gate = _Gate(0.0)
    anc.program = SimpleNamespace(generate=lambda frames: np.zeros(frames, np.float32))
    anc.digital_reference_buffer = _DigitalReferencePassThrough()
    anc.in_ring = SPSCRing(4, BLOCK * 4)
    anc.out_ring = SPSCRing(1, BLOCK * 4)
    anc.out_ring.push(np.zeros((1, BLOCK), dtype=np.float32))
    anc.handoff_budget = SimpleNamespace(
        input_keep_backlog_samples=BLOCK,
        output_keep_backlog_samples=BLOCK,
    )
    out_report = SimpleNamespace(
        signal=np.zeros(BLOCK, dtype=np.float32), clipped_fraction=0.0
    )
    verdict = SimpleNamespace(mute=False, messages=())
    anc.safety = SimpleNamespace(
        probe_active=False,
        limit_output=lambda _value: out_report,
        check_block=lambda _value: verdict,
        baseline=SimpleNamespace(initialized=False, power=0.0),
        trip_counts={},
    )
    anc.engine = SimpleNamespace(secondary_total_length=0)
    anc._last_anc = False
    anc._adaptation_hold_samples = 0
    anc._fade_samples = 0
    anc._last_input_drops = 0
    anc.err_meter = _Power()
    anc.ctrl_meter = _Power()
    anc.rec = None
    anc.rec_pos = 0
    anc.record_len = 0
    anc.step_times_ms = []
    anc.state = SimpleNamespace(
        noise_enabled=True,
        anc_enabled=False,
        latest_stats={},
        messages=queue.Queue(),
        fatal_error=None,
        quit_event=threading.Event(),
    )
    anc.sd = SimpleNamespace(CallbackAbort=RuntimeError)
    input_pcm = np.zeros((BLOCK, 2), dtype=np.int32)
    output_pcm = np.empty((BLOCK, 2), dtype=np.int16)

    anc._callback(input_pcm, output_pcm, BLOCK, _time_info(0), False)

    receipt = anc.clock_telemetry_receipt()
    callback = receipt["callbacks"][0]
    assert callback["input_buffer_adc_time"] == _time_info(0)["inputBufferAdcTime"]
    assert callback["output_buffer_dac_time"] == _time_info(0)["outputBufferDacTime"]
    assert callback["callback_current_time"] == _time_info(0)["currentTime"]
    assert callback["callback_start_frame"] == 0
    assert callback["callback_frame_count"] == BLOCK
    assert callback["completed"] is True
    assert callback["runtime_counters_at_callback_entry"] is not None
    assert callback["runtime_counters_after_callback"] is not None
    assert receipt["runtime_counters_final"]["deadline_miss_count"] == 0
    assert receipt["runtime_counters_final"]["fallback_silence_blocks"] == 0
    assert receipt["authority_status"] == "INCONCLUSIVE"
    assert anc.state.latest_stats["clock_telemetry_status"] == "INCONCLUSIVE"
