from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np
import pytest

from deep_anc.dsp.control_band_contract import BroadbandFullOctaveContractV3
from deep_anc.dsp.timing import TrainingTimingContract
from deep_anc.model_input import RefOnlyModelInputContract
from deep_anc.realtime.output_clock_master import (
    OutputClockMasterAdmission,
    OutputClockMasterScheduler,
)
from deep_anc.realtime.output_clock_runtime_adapter import (
    OutputClockMasterRuntimeAdapter,
    OutputClockRuntimeAbort,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _admission() -> OutputClockMasterAdmission:
    v3 = BroadbandFullOctaveContractV3.canonical()
    timing = TrainingTimingContract(
        primary_zeros_before_fir_samples=140,
        primary_fir_peak_offset_samples=4,
        primary_effective_delay_samples=144,
        secondary_delay_samples=0,
        handoff_samples=256,
        sample_rate=48_000,
        raw_digital_reference_lead_samples=116,
        digital_reference_lead_samples=116,
        synthetic_total_advance_samples=260,
    )
    mode = RefOnlyModelInputContract(error_dropout_probability=1.0)
    return OutputClockMasterAdmission(
        digital_reference_lead_samples=timing.digital_reference_lead_samples,
        control_band_contract=v3,
        control_band_contract_sha256=v3.digest(),
        training_timing_contract=timing,
        training_timing_contract_sha256=timing.digest(),
        experiment_contract_sha256=_sha("experiment"),
        checkpoint_sha256=_sha("checkpoint"),
        deployment_artifact_sha256=_sha("deployment"),
        model_input_contract=mode,
        model_input_mode_sha256=mode.digest(),
        ref_only_ablation_receipt_sha256=_sha("ablation"),
        ref_only_g0_receipt_sha256=_sha("g0"),
        ref_only_validation_receipt_sha256=_sha("validation"),
        offline_streaming_equivalence_receipt_sha256=_sha("equivalence"),
        offline_streaming_max_abs_error=1.0e-6,
    )


def _source(callback_index: int, frame_start: int, frames: int) -> np.ndarray:
    assert frame_start == callback_index * frames
    return np.linspace(
        -0.02 + callback_index * 0.001,
        0.02 + callback_index * 0.001,
        frames,
        dtype=np.float32,
    )


class _Engine:
    def __init__(self, *, fail: bool = False, hold: threading.Event | None = None) -> None:
        self.fail = fail
        self.hold = hold
        self.done = threading.Event()
        self.calls: list[tuple[int, np.ndarray, np.ndarray]] = []
        self.reset_count = 0

    def step(self, reference: np.ndarray, error_feature: np.ndarray) -> np.ndarray:
        self.calls.append(
            (
                threading.get_ident(),
                np.asarray(reference).copy(),
                np.asarray(error_feature).copy(),
            )
        )
        if self.hold is not None:
            self.hold.wait(timeout=1.0)
        if self.fail:
            raise RuntimeError("engine failed")
        self.done.set()
        return np.full(256, 0.01, dtype=np.float32)

    def reset(self) -> None:
        self.reset_count += 1


def _adapter(
    *,
    engine: _Engine | None = None,
    aborts: list[str] | None = None,
    capacity: int = 16,
    clock_ns=None,
) -> tuple[OutputClockMasterRuntimeAdapter, _Engine, list[str]]:
    actual_engine = _Engine() if engine is None else engine
    actual_aborts: list[str] = [] if aborts is None else aborts
    kwargs = {}
    if clock_ns is not None:
        kwargs["clock_ns"] = clock_ns
    adapter = OutputClockMasterRuntimeAdapter(
        scheduler=OutputClockMasterScheduler(_admission()),
        engine=actual_engine,
        future_source=_source,
        abort_stream=actual_aborts.append,
        input_witness_capacity_blocks=capacity,
        **kwargs,
    )
    return adapter, actual_engine, actual_aborts


def _out_time(index: int) -> dict[str, float]:
    return {"outputBufferDacTime": 100.0 + index * 256 / 48_000}


def _in_time(index: int) -> dict[str, float]:
    # DAC와 의도적으로 전혀 다른 origin이다. 두 clock을 서로 정렬하면 안 된다.
    return {"inputBufferAdcTime": -700.0 + index * 256 / 48_000}


def _wait_steps(adapter: OutputClockMasterRuntimeAdapter, count: int) -> None:
    event = threading.Event()
    for _ in range(1000):
        if adapter.counters_snapshot().inference_step_count >= count:
            return
        event.wait(0.001)
    raise AssertionError(f"inference_step_count가 {count}에 도달하지 못했습니다")


def test_output_clock_callback_worker_and_input_witness_are_strictly_separated() -> None:
    adapter, engine, aborts = _adapter()
    adapter.start_worker()
    adapter.request_anc_on()

    callback_thread = threading.get_ident()
    prime = np.empty((256, 2), dtype=np.int16)
    adapter.output_callback(prime, 256, _out_time(0), None)
    assert np.count_nonzero(prime[:, 1]) == 0
    assert engine.done.wait(timeout=1.0)
    _wait_steps(adapter, 1)

    # APE callback은 output과 다른 clock origin이어도 허용되며 engine을 깨우지 않는다.
    calls_before_input = len(engine.calls)
    raw = np.arange(512, dtype=np.int32).reshape(256, 2)
    adapter.input_witness_callback(raw, 256, _in_time(0), None)
    assert len(engine.calls) == calls_before_input

    # 중복 ON 명령은 새 prime을 삽입해 control을 한 블록 늦추지 않는다.
    adapter.request_anc_on()

    active = np.empty((256, 2), dtype=np.int16)
    engine.done.clear()
    adapter.output_callback(active, 256, _out_time(1), None)
    assert np.count_nonzero(active[:, 1]) == 256
    assert engine.done.wait(timeout=1.0)
    _wait_steps(adapter, 2)

    receipt = adapter.close_evidence_window()
    assert receipt.counters.violations() == {}
    assert receipt.counters.output_callback_count == 2
    assert receipt.counters.output_performance_callback_count == 1
    assert receipt.counters.input_witness_callback_count == 1
    assert receipt.counters.inference_step_count == 2
    assert receipt.counters.fallback_silence_block_count == 0
    assert receipt.scheduler_receipt.performance_output_block_count == 1
    assert receipt.cross_clock_timestamp_alignment_used is False
    assert receipt.sounddevice_imported_or_stream_opened is False
    assert receipt.physical_performance_pass is False
    assert receipt.deployment_eligible is False
    assert aborts == []

    assert all(thread_id != callback_thread for thread_id, _, _ in engine.calls)
    assert all(np.count_nonzero(err) == 0 for _, _, err in engine.calls)
    assert np.array_equal(engine.calls[0][1], _source(0, 0, 256))
    witness = adapter.drain_input_witness()
    assert len(witness) == 1
    assert witness[0].raw.flags.writeable is False
    assert np.array_equal(witness[0].raw, raw)


def test_missing_result_aborts_instead_of_emitting_silence_fallback() -> None:
    hold = threading.Event()
    engine = _Engine(hold=hold)
    adapter, _, aborts = _adapter(engine=engine)
    adapter.start_worker()
    adapter.request_anc_on()
    prime = np.empty((256, 2), dtype=np.int16)
    adapter.output_callback(prime, 256, _out_time(0), None)

    active = np.full((256, 2), 123, dtype=np.int16)
    with pytest.raises(OutputClockRuntimeAbort, match="underflow/late"):
        adapter.output_callback(active, 256, _out_time(1), None)
    hold.set()
    adapter.stop_worker()

    counters = adapter.counters_snapshot()
    assert counters.inference_queue_underflow_count == 1
    assert counters.fallback_silence_block_count == 0
    assert np.count_nonzero(active) == 0
    assert len(aborts) == 1


class _Status:
    output_underflow = True

    def __bool__(self) -> bool:
        return True

    def __str__(self) -> str:
        return "output underflow"


@pytest.mark.parametrize(
    ("frames", "time_info", "status", "counter", "value"),
    [
        (256, _out_time(0), _Status(), "output_xrun_count", 1),
        (255, _out_time(0), None, "dropped_sample_count", 1),
        (256, {}, None, "timestamp_missing_count", 1),
    ],
)
def test_output_status_frame_and_timestamp_failures_permanently_abort(
    frames: int,
    time_info: dict[str, float],
    status: object,
    counter: str,
    value: int,
) -> None:
    adapter, _, aborts = _adapter()
    out = np.full((256, 2), 321, dtype=np.int16)
    with pytest.raises(OutputClockRuntimeAbort):
        adapter.output_callback(out, frames, time_info, status)
    assert getattr(adapter.counters_snapshot(), counter) == value
    assert np.count_nonzero(out) == 0
    assert len(aborts) == 1
    with pytest.raises(OutputClockRuntimeAbort):
        adapter.output_callback(out, 256, _out_time(0), None)
    assert len(aborts) == 1


def test_engine_exception_is_not_converted_to_fallback() -> None:
    adapter, engine, aborts = _adapter(engine=_Engine(fail=True))
    adapter.start_worker()
    adapter.request_anc_on()
    adapter.output_callback(
        np.empty((256, 2), dtype=np.int16), 256, _out_time(0), None
    )
    deadline = threading.Event()
    # abort hook은 worker에서 즉시 호출된다.
    for _ in range(1000):
        if aborts:
            break
        deadline.wait(0.001)
    adapter.stop_worker()
    counters = adapter.counters_snapshot()
    assert len(engine.calls) == 1
    assert counters.inference_engine_exception_count == 1
    assert counters.fallback_silence_block_count == 0
    assert len(aborts) == 1


@pytest.mark.parametrize("callback_kind", ["output", "input"])
def test_callback_deadline_miss_has_a_distinct_exact_counter(
    callback_kind: str,
) -> None:
    values = iter((0, 256 * 1_000_000_000 // 48_000))
    adapter, _, aborts = _adapter(clock_ns=lambda: next(values))
    if callback_kind == "output":
        out = np.full((256, 2), 77, dtype=np.int16)
        with pytest.raises(OutputClockRuntimeAbort, match="output callback deadline"):
            adapter.output_callback(out, 256, _out_time(0), None)
        assert np.count_nonzero(out) == 0
        assert adapter.counters_snapshot().output_callback_deadline_miss_count == 1
        assert adapter.counters_snapshot().input_callback_deadline_miss_count == 0
    else:
        with pytest.raises(OutputClockRuntimeAbort, match="input callback deadline"):
            adapter.input_witness_callback(
                np.zeros((256, 2), dtype=np.int32), 256, _in_time(0), None
            )
        assert adapter.counters_snapshot().output_callback_deadline_miss_count == 0
        assert adapter.counters_snapshot().input_callback_deadline_miss_count == 1
    assert len(aborts) == 1


def test_inference_deadline_miss_is_not_aliased_to_callback_or_fallback() -> None:
    worker_calls = 0
    main_calls = 0

    def clock_ns() -> int:
        nonlocal worker_calls, main_calls
        if threading.current_thread().name == "anc-output-clock-inference":
            value = worker_calls * (256 * 1_000_000_000 // 48_000)
            worker_calls += 1
            return value
        value = main_calls * 1000
        main_calls += 1
        return value

    adapter, _, aborts = _adapter(clock_ns=clock_ns)
    adapter.start_worker()
    adapter.request_anc_on()
    adapter.output_callback(
        np.empty((256, 2), dtype=np.int16), 256, _out_time(0), None
    )
    event = threading.Event()
    for _ in range(1000):
        if aborts:
            break
        event.wait(0.001)
    adapter.stop_worker()
    counters = adapter.counters_snapshot()
    assert counters.inference_deadline_miss_count == 1
    assert counters.output_callback_deadline_miss_count == 0
    assert counters.input_callback_deadline_miss_count == 0
    assert counters.fallback_silence_block_count == 0
    assert len(aborts) == 1


def test_input_queue_overflow_blocks_but_never_paces_inference() -> None:
    adapter, engine, aborts = _adapter(capacity=1)
    raw = np.zeros((256, 2), dtype=np.int32)
    adapter.input_witness_callback(raw, 256, _in_time(0), None)
    assert engine.calls == []
    with pytest.raises(OutputClockRuntimeAbort, match="witness queue overflow"):
        adapter.input_witness_callback(raw, 256, _in_time(1), None)
    assert engine.calls == []
    assert adapter.counters_snapshot().input_witness_queue_overflow_count == 1
    assert len(aborts) == 1


def test_output_local_timestamp_slip_is_blocked_without_adc_dac_comparison() -> None:
    adapter, engine, _ = _adapter()
    adapter.start_worker()
    adapter.request_anc_on()
    adapter.output_callback(
        np.empty((256, 2), dtype=np.int16), 256, _out_time(0), None
    )
    assert engine.done.wait(timeout=1.0)
    _wait_steps(adapter, 1)
    slipped = {"outputBufferDacTime": 100.0 + 257 / 48_000}
    with pytest.raises(OutputClockRuntimeAbort, match="sample slip"):
        adapter.output_callback(
            np.empty((256, 2), dtype=np.int16), 256, slipped, None
        )
    adapter.stop_worker()
    counters = adapter.counters_snapshot()
    assert counters.sample_slip_count == 1
    assert counters.added_sample_count == 1


def test_module_has_no_sounddevice_or_device_open_path() -> None:
    source = Path(
        "src/deep_anc/realtime/output_clock_runtime_adapter.py"
    ).read_text(encoding="utf-8")
    executable = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert "import sounddevice" not in executable
    assert "OutputStream(" not in executable
    assert "InputStream(" not in executable
