"""오디오 장치를 열지 않고 acoustic REF 콜백·적응·기준선을 검증한다."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from deep_anc.realtime import run_realtime
from deep_anc.realtime.engines import FxLMSEngine, HybridEngine


class _FakeStream:
    """콜백과 추론 사이 링버퍼를 순차 구동하는 무음 스트림."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.control = np.zeros((1, runtime.block), dtype=np.float32)

    def pump(self, *, err=0.02, ref=0.05, status=False, have_output=True):
        runtime = self.runtime
        wave = np.sin(2 * np.pi * 1500 * np.arange(runtime.block) / runtime.fs)
        mics = np.stack([err * wave, ref * wave], axis=-1)
        indata = np.rint(mics * ((1 << 31) - 1)).astype(np.int32)
        outdata = np.empty((runtime.block, 2), dtype=np.int16)
        if have_output:
            runtime.out_ring.push(self.control)
        runtime._callback(indata, outdata, runtime.block, None, status)
        block, ok = runtime.in_ring.pop_latest(runtime.block, keep_backlog=runtime.block)
        assert ok
        self.control = runtime._step_input_block(block).reshape(1, -1)
        return outdata, block[3]


@pytest.fixture
def make_runtime(monkeypatch, tmp_path):
    class CallbackAbort(Exception):
        pass

    # RealtimeANC 초기화의 import/장치 조회만 대체한다. Stream은 제공하지 않는다.
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(CallbackAbort=CallbackAbort))
    monkeypatch.setattr(run_realtime, "resolve_alsa_portaudio_device", lambda *_args: 0)
    secondary = tmp_path / "secondary.npz"
    def make(
        reference="mic", noise_enabled=False, fade_ms=2.0,
        delay_samples=0, fir_size=1, neural_only=False, hybrid=False, measured=True,
        control_length=8, calibration_block_size=256, calibration_latency="low",
        record_seconds=0.0,
    ):
        fir = np.zeros(fir_size, dtype=np.float32)
        fir[0] = 1.0
        np.savez(
            secondary, fir=fir, delay_samples=np.array(delay_samples),
            sample_rate=np.array(48_000),
            calibration_block_size=calibration_block_size,
            calibration_latency=calibration_latency,
        )
        cfg = {
            "reference": reference,
            "controller": "hybrid" if hybrid else "dl" if neural_only else "fxlms",
            "hop": 256,
            "digital_reference_lead_samples": 0,
            "hardware": {
                "audio": {
                    "sample_rate": 48_000, "block_size": 256, "latency": "low",
                    "input": {"card": "fake", "pcm": 0},
                    "output": {"card": "fake", "pcm": 0},
                },
                "channels": {
                    "error_mic": 0, "reference_mic": 1,
                    "noise_out": 0, "cancel_out": 1,
                },
            },
            "noise": {"type": "tone", "amplitude": 0.03},
            "safety": {"fade_ms": fade_ms},
        }
        if measured:
            cfg["duct"] = {
                "secondary_path": {"npz": str(secondary), "handoff_extra_samples": 256}
            }
        if noise_enabled is not None:
            cfg["noise"]["enabled"] = noise_enabled
        engine = FxLMSEngine(
            str(secondary), {"control_length": control_length, "mu": 0.001}, hop=256,
        )
        if neural_only or hybrid:
            class Neural:
                hop = 256
                digital_reference_lead_samples = 0
                reference_mode = "acoustic"
                gain = 0.02

                def __init__(self):
                    self.reset_count = 0
                    self.seen_inputs = []

                def reset(self):
                    self.reset_count += 1

                def set_adapt_enabled(self, enabled):
                    pass

                def step(self, ref, err):
                    self.seen_inputs.append((ref.copy(), err.copy()))
                    return self.gain * ref

            engine = Neural() if neural_only else HybridEngine(Neural(), engine)
        monkeypatch.setattr(run_realtime, "build_engine", lambda _cfg: engine)
        runtime = run_realtime.RealtimeANC(cfg, record_seconds=record_seconds)
        return runtime, _FakeStream(runtime)

    return make


def test_external_mic_noise_establishes_baseline_and_adapts_without_playback(make_runtime):
    runtime, stream = make_runtime()
    assert not runtime.state.anc_enabled
    assert not runtime.state.noise_enabled
    assert runtime.noise_gate.target == 0.0
    outdata, gate = stream.pump()
    assert np.count_nonzero(outdata) == 0
    assert not np.any(gate)
    assert runtime.baseline_init and runtime.baseline_power > 0
    assert runtime.engine.controller.update_count == 0

    runtime.state.anc_enabled = True
    _, gate = stream.pump()
    assert not np.any(gate)  # 페이드와 이차 경로의 초기 워밍업을 보존한다.
    for _ in range(5):
        outdata, gate = stream.pump()
        assert np.count_nonzero(outdata[:, runtime.ch_noise]) == 0
    assert np.all(gate)
    assert runtime.engine.controller.update_count > 0


@pytest.mark.parametrize("fault", ["silence", "err_clip", "ref_clip", "err_nan", "ref_inf", "xrun"])
def test_mic_rejects_invalid_external_input_for_baseline_and_adaptation(
    make_runtime, monkeypatch, fault,
):
    runtime, stream = make_runtime()
    kwargs = {}
    if fault == "silence":
        kwargs["ref"] = 0.0
    elif fault == "err_clip":
        kwargs["err"] = 0.99
    elif fault == "ref_clip":
        kwargs["ref"] = 0.99
    elif fault == "xrun":
        kwargs["status"] = True
    else:
        convert = run_realtime.pcm_int32_to_float32

        def corrupted_input(samples):
            result = convert(samples)
            result[0, 0 if fault == "err_nan" else 1] = np.nan if fault == "err_nan" else np.inf
            return result

        monkeypatch.setattr(run_realtime, "pcm_int32_to_float32", corrupted_input)

    for _ in range(3):
        _, gate = stream.pump(**kwargs)
        assert not runtime.baseline_init
        assert not np.any(gate)
    runtime.state.anc_enabled = True
    for _ in range(5):
        _, gate = stream.pump(**kwargs)
        assert not np.any(gate)
    assert runtime.engine.controller.update_count == 0
    assert np.isfinite(runtime.err_meter.value)
    assert np.isfinite(runtime.ref_dc._zi).all()


def test_mic_underrun_stops_adaptation(make_runtime):
    runtime, stream = make_runtime()
    stream.pump()
    runtime.state.anc_enabled = True
    for _ in range(5):
        stream.pump()
    updates = runtime.engine.controller.update_count
    _, gate = stream.pump(have_output=False)
    assert not np.any(gate)
    assert runtime.engine.controller.update_count == updates


def test_damaged_mic_input_waits_for_secondary_and_control_filter_memory(make_runtime):
    runtime, stream = make_runtime(delay_samples=512, fir_size=3, control_length=513)
    stream.pump()
    runtime.state.anc_enabled = True
    for _ in range(8):
        stream.pump()
    controller = runtime.engine.controller
    updates = controller.update_count
    _, gate = stream.pump(ref=0.99)
    assert runtime.state.anc_enabled  # 선형 FxLMS는 유한 메모리가 비워진 뒤 재개한다.
    assert not np.any(gate)
    assert runtime.adaptive_input_memory_samples == runtime.secondary_total_length + 513
    assert runtime._adaptation_hold_samples == runtime.adaptive_input_memory_samples

    delayed_blocks = (runtime.adaptive_input_memory_samples + runtime.block - 1) // runtime.block
    for i in range(delayed_blocks):
        # OFF→ON 토글도 아직 남은 손상 입력 이력을 조기 해제하면 안 된다.
        if i == 0:
            runtime.state.anc_enabled = False
        elif i == 1:
            runtime.state.anc_enabled = True
        _, gate = stream.pump()
        assert not np.any(gate)
        assert controller.update_count == updates
    _, gate = stream.pump()
    assert np.all(gate)
    assert controller.update_count == updates + 1
    assert np.isfinite(controller._xf_history).all()


@pytest.mark.parametrize("hybrid", [False, True])
def test_neural_mic_damage_turns_anc_off_and_never_enters_recurrent_state(make_runtime, hybrid):
    runtime, stream = make_runtime(neural_only=not hybrid, hybrid=hybrid)
    neural = runtime.engine.neural if hybrid else runtime.engine
    stream.pump()
    runtime.state.anc_enabled = True
    for _ in range(5):
        stream.pump()
    seen = len(neural.seen_inputs)
    resets = neural.reset_count
    output, gate = stream.pump(ref=0.99)
    assert np.all(gate == -1.0)
    assert not runtime.state.anc_enabled
    assert len(neural.seen_inputs) == seen
    assert neural.reset_count == resets + 1
    assert runtime.anc_gate.target == 0.0
    assert np.count_nonzero(output[-1, runtime.ch_cancel]) == 0  # 기존 fade로 OFF 전환.
    assert "입력 손상" in runtime.state.messages.get_nowait()

    # reset 통지가 먼저 처리된 뒤 같은 손상 패킷이 소비되는 순서도 안전해야 한다.
    stale_packet = np.full((4, runtime.block), 0.99, dtype=np.float32)
    stale_packet[3] = -1.0
    runtime._neural_input_reset_pending.clear()
    np.testing.assert_array_equal(runtime._step_input_block(stale_packet), 0.0)
    assert len(neural.seen_inputs) == seen
    assert neural.reset_count == resets + 2
    for _ in range(6):
        stream.pump()
        assert not runtime.state.anc_enabled
    runtime.state.anc_enabled = True  # 입력 복구 후 현장 명시적 재활성만 허용한다.
    for _ in range(5):
        _, gate = stream.pump()
    assert runtime.state.anc_enabled
    assert np.all(gate == 1.0)


@pytest.mark.parametrize("channel", ["err", "ref"])
@pytest.mark.parametrize("hybrid", [False, True])
def test_clipped_measurements_remain_in_recording_when_engine_input_is_discarded(
    make_runtime, channel, hybrid,
):
    runtime, stream = make_runtime(hybrid=hybrid, record_seconds=0.1)
    _, gate = stream.pump(**{channel: 0.99})
    assert np.max(np.abs(runtime.rec[channel][:runtime.block])) > 0.98
    assert runtime.state.latest_stats["input_clip_fraction"] > 0.0
    assert not runtime.baseline_init
    assert not np.any(gate >= 0.5)
    adaptive = runtime.engine.adaptive if hybrid else runtime.engine
    np.testing.assert_array_equal(adaptive.controller._x_history, 0.0)
    if channel == "err":
        assert runtime.state.latest_stats["err_dbfs"] > -10.0


@pytest.mark.parametrize("fault", ["sum_clip", "xrun", "underrun", "nonfinite_output"])
def test_hybrid_waits_for_delayed_error_after_output_disruption(make_runtime, fault):
    runtime, stream = make_runtime(hybrid=True, delay_samples=768, fir_size=3)
    stream.pump()
    runtime.state.anc_enabled = True
    for _ in range(9):
        stream.pump()
    assert runtime.state.latest_stats["fxlms_adapt_allowed"]
    adaptive = runtime.engine.adaptive.controller
    if fault == "sum_clip":
        # 두 분기는 각각 limit 이하이지만 합산하면 포화되는 실제 hybrid 경로.
        runtime.engine.neural.gain = 3.0
        adaptive.w[:] = 0.0
        adaptive.w[0] = 3.0
        stream.pump()
        assert 0.2 < np.max(np.abs(stream.control)) < 0.4
        runtime.engine.neural.gain = 0.0
        adaptive.w[:] = 0.0
    elif fault == "nonfinite_output":
        stream.control[0, 0] = np.nan

    updates = adaptive.update_count
    output, gate = stream.pump(status=fault == "xrun", have_output=fault != "underrun")
    assert not np.any(gate)
    assert adaptive.update_count == updates
    assert np.max(np.abs(output[:, 1].astype(np.float64) / 32767)) <= 0.20002
    assert runtime._adaptation_hold_samples == runtime.secondary_total_length

    delayed_blocks = (runtime.secondary_total_length + runtime.block - 1) // runtime.block
    for _ in range(delayed_blocks):
        _, gate = stream.pump()
        assert not np.any(gate)
        assert adaptive.update_count == updates
    _, gate = stream.pump()
    assert np.all(gate)
    assert adaptive.update_count == updates + 1


@pytest.mark.parametrize("trigger", ["reset", "engine_error"])
def test_inference_rewarm_notification_reaches_callback(make_runtime, monkeypatch, trigger):
    runtime, stream = make_runtime()
    engine = runtime.engine
    engine.controller.w[:] = 0.1
    original_step = engine.step

    def supply_input(_frames, timeout):
        # 이전 callback의 허용 플래그가 남아 있어도 재워밍업 중 적응하지 않는다.
        block = np.full((4, runtime.block), 0.05, dtype=np.float32)
        block[3] = 1.0
        runtime.in_ring.push(block)
        return True

    def one_step(ref, err):
        runtime.state.quit_event.set()
        if trigger == "engine_error":
            raise RuntimeError("의도한 추론 실패")
        return original_step(ref, err)

    monkeypatch.setattr(runtime.in_ring, "wait_for", supply_input)
    monkeypatch.setattr(engine, "step", one_step)
    runtime.state.anc_enabled = True
    if trigger == "reset":
        runtime.state.reset_event.set()
    runtime._inference_loop()
    assert runtime._adaptation_rewarm_event.is_set()
    assert not runtime.state.reset_event.is_set()
    assert engine.controller.update_count == 0
    assert not engine.adapt
    np.testing.assert_array_equal(engine.controller.w, 0.0)

    monkeypatch.setattr(engine, "step", original_step)
    runtime.state.quit_event.clear()
    _, gate = stream.pump()
    assert not np.any(gate)
    assert not runtime._adaptation_rewarm_event.is_set()
    assert runtime._adaptation_hold_samples >= runtime.secondary_total_length


def test_mic_baseline_waits_for_control_tail_after_turning_off(make_runtime):
    runtime, stream = make_runtime(fade_ms=0.0)
    stream.pump()
    runtime.state.anc_enabled = True
    for _ in range(5):
        stream.pump()
    baseline = runtime.baseline_power
    runtime.state.anc_enabled = False
    # 이차 경로 257 samples 동안 남아 있는 상쇄음은 새 OFF 기준에 섞지 않는다.
    for _ in range(2):
        stream.pump(err=0.2)
        assert runtime.baseline_power == baseline
    stream.pump(err=0.04)
    assert runtime.baseline_power > baseline


def test_neural_only_mic_baseline_uses_measured_secondary_tail(make_runtime):
    runtime, stream = make_runtime(neural_only=True, delay_samples=512, fir_size=3, fade_ms=0.0)
    assert not hasattr(runtime.engine, "secondary_total_length")
    assert runtime.secondary_total_length == 512 + runtime.block + 3
    stream.pump()
    baseline = runtime.baseline_power
    runtime.state.anc_enabled = True
    stream.pump()
    runtime.state.anc_enabled = False
    tail_blocks = (runtime.secondary_total_length + runtime.block - 1) // runtime.block
    for _ in range(tail_blocks):
        stream.pump(err=0.2)
        assert runtime.baseline_power == baseline
    stream.pump(err=0.04)
    assert runtime.baseline_power > baseline


def test_neural_only_runtime_does_not_invent_missing_secondary_measurement(make_runtime):
    with pytest.raises(KeyError):
        make_runtime(neural_only=True, measured=False)


@pytest.mark.parametrize("calibration", [
    {"calibration_block_size": 512},
    {"calibration_latency": "high"},
    {"calibration_block_size": 0},
])
def test_neural_only_runtime_checks_its_measured_audio_conditions(make_runtime, calibration):
    with pytest.raises(ValueError, match="block_size|latency"):
        make_runtime(neural_only=True, **calibration)


def test_digital_still_requires_internal_noise_and_defaults_to_noise_on(make_runtime):
    runtime, stream = make_runtime(reference="digital", noise_enabled=False)
    for _ in range(3):
        stream.pump()
    assert not runtime.baseline_init
    runtime.state.anc_enabled = True
    for _ in range(5):
        _, gate = stream.pump()
    assert not np.any(gate)
    assert runtime.engine.controller.update_count == 0

    runtime, stream = make_runtime(reference="digital", noise_enabled=None)
    assert runtime.state.noise_enabled and not runtime.state.anc_enabled
    for _ in range(25):
        stream.pump()
    assert runtime.baseline_init
    runtime.state.anc_enabled = True
    for _ in range(5):
        _, gate = stream.pump(ref=0.99)  # digital 참조는 REF 마이크를 사용하지 않는다.
    assert np.all(gate)
    assert runtime.engine.controller.update_count > 0


@pytest.mark.parametrize("reference", ["acoustic", "unknown", "", None])
def test_unknown_runtime_reference_rejected_before_hardware(reference):
    with pytest.raises(ValueError, match="reference"):
        run_realtime.RealtimeANC({"reference": reference})
    with pytest.raises(ValueError, match="reference"):
        run_realtime.fxlms_adaptation_allowed(
            reference=reference, requested=True, full_anc_gain=True,
            full_noise_gain=True, hold_samples=0, output_clip_fraction=0.0,
            input_clip_fraction=0.0, reference_power=0.1, stream_ok=True,
        )


@pytest.mark.parametrize("power", [np.inf, np.nan, 0.0, -1.0])
def test_nonfinite_or_empty_reference_power_never_adapts(power):
    assert not run_realtime.fxlms_adaptation_allowed(
        reference="mic", requested=True, full_anc_gain=True, full_noise_gain=False,
        hold_samples=0, output_clip_fraction=0.0, input_clip_fraction=0.0,
        reference_power=power, stream_ok=True,
    )
