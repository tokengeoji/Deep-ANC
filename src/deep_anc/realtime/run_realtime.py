"""실시간 Deep ANC 런타임 — 3-스레드 구조 (콜백 / 추론 / 제어).

  python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml
  python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml --set controller=fxlms
  python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml --calibrate

구조 (docs/06):
  [콜백]   입력 변환/DC차단 → in_ring, 소음(ch0) 생성, out_ring→리미터/게이트→ch1 출력
  [추론]   in_ring 에서 hop 단위 소비 → engine.step → out_ring  (콜백은 절대 대기 안 함)
  [제어]   키보드, 1초 통계, 워치독 메시지
파이프라인 핸드오프 지연 = 1 hop — 학습 플랜트의 handoff_extra_samples 와 정합 [C1].
시작은 항상 ANC OFF. 시스템(전원모드/RT우선순위 등)은 건드리지 않는다 — 프로젝트 정책.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from ..audio_io import (
    capture_input_probe,
    float32_to_pcm_int16,
    format_sounddevice_devices,
    pcm_int32_to_float32,
    resolve_alsa_portaudio_device,
)
from ..config import DEFAULT_HANDOFF_SAMPLES, load_runtime_config
from ..dsp.filters import DCBlocker
from .engines import build_engine, secondary_path_npz, validate_secondary_calibration
from .noise_gen import DigitalReferenceBuffer, NoiseProgram
from .recording import (
    build_recording_payload,
    capture_recording_provenance,
    prepare_recording_path,
    save_recording,
)
from .ring_buffer import SPSCRing
from .safety import FadeGate, PowerEMA, SafetySupervisor
from .ui import KeyboardController, RuntimeState


def power_to_db(power: float, floor_db: float = -200.0) -> float:
    if not np.isfinite(power) or power <= 0.0:
        return floor_db
    return max(floor_db, 10.0 * float(np.log10(power)))


def _validate_reference_mode(reference: str) -> None:
    if reference not in {"digital", "mic"}:
        raise ValueError(f"reference는 digital 또는 mic이어야 합니다: {reference!r}")


def fxlms_adaptation_allowed(
    *,
    requested: bool,
    full_anc_gain: bool,
    full_noise_gain: bool,
    hold_samples: int,
    output_clip_fraction: float,
    input_clip_fraction: float,
    reference_power: float,
    stream_ok: bool,
    reference: str = "digital",
) -> bool:
    """FxLMS가 현재 ERR 블록으로 갱신해도 되는 안전 조건."""
    _validate_reference_mode(reference)
    return bool(
        requested
        and full_anc_gain
        and (reference == "mic" or full_noise_gain)
        and int(hold_samples) == 0
        and float(output_clip_fraction) == 0.0
        and float(input_clip_fraction) == 0.0
        and np.isfinite(reference_power)
        and float(reference_power) > 1.0e-12
        and stream_ok
    )


def input_preflight(cfg: dict, seconds: float = 2.0) -> bool:
    """스피커를 열기 전에 필수 I2S 입력 채널이 살아 있는지 확인한다."""
    _validate_reference_mode(str(cfg.get("reference", "digital")))
    report = capture_input_probe(cfg["hardware"]["audio"], seconds=seconds)
    names = ("ERR", "REF")
    for item in report["channels"][:2]:
        index = int(item["channel"])
        verdict = "PASS" if item["valid"] else "FAIL"
        print(
            f"[{verdict}] {names[index]} ch{index}: RMS {item['rms_dbfs']:.2f}dBFS, "
            f"peak {item['peak']:.6f}, clip {item['clip_ratio']:.3%}, "
            f"unique {item['unique_codes']}, raw [{item['raw_min']}, {item['raw_max']}]"
        )

    required = (0, 1) if cfg.get("reference") == "mic" else (0,)
    failed = [index for index in required if not report["channels"][index]["valid"]]
    if failed:
        labels = ", ".join(f"{names[index]} ch{index}" for index in failed)
        print(
            f"[중단] 필수 입력({labels})이 무효입니다. 오디오 출력을 시작하지 않습니다.",
            file=sys.stderr,
        )
        return False
    if not report["channels"][1]["valid"]:
        print(
            "[경고] REF ch1 무효 — digital-reference만 허용하며 mic-reference는 금지합니다.",
            file=sys.stderr,
        )
    return True


def validate_digital_reference_lead(
    reference: str,
    configured_lead: int,
    checkpoint_lead: int | None = None,
) -> int:
    """reference 모드와 학습/배포 lead 정합을 검증하고 정규화된 값을 반환한다."""
    _validate_reference_mode(reference)
    lead = int(configured_lead)
    if lead < 0:
        raise ValueError("digital_reference_lead_samples는 0 이상이어야 합니다")
    if reference != "digital" and lead:
        raise ValueError(
            "digital_reference_lead_samples는 reference=digital에서만 사용할 수 있습니다"
        )
    if checkpoint_lead is not None and lead != int(checkpoint_lead):
        raise ValueError(
            "digital-reference lead 불일치: "
            f"runtime={lead}, checkpoint={int(checkpoint_lead)}. "
            "학습과 배포의 digital_reference_lead_samples를 동일하게 맞추세요."
        )
    return lead


class RealtimeANC:
    """프로그래밍 API — evaluate_session 등에서 재사용. CLI 는 main() 참조."""

    def __init__(self, cfg: dict, record_seconds: float = 0.0) -> None:
        if bool(cfg.get("start_on", False)):
            raise ValueError(
                "안전 규약상 start_on=true는 허용되지 않습니다. "
                "ANC는 OFF로 시작한 뒤 현장에서 명시적으로 켜야 합니다."
            )

        reference = str(cfg.get("reference", "digital"))
        digital_reference_lead = validate_digital_reference_lead(
            reference, cfg.get("digital_reference_lead_samples", 0)
        )

        import sounddevice as sd

        self.sd = sd
        self.cfg = cfg
        hw = cfg["hardware"]["audio"]
        self.fs = int(hw["sample_rate"])
        self.block = int(hw["block_size"])
        self.latency = str(hw.get("latency", "low"))
        if self.latency not in {"low", "high"}:
            raise ValueError(f"hardware.audio.latency는 low/high여야 합니다: {self.latency!r}")
        self.hop = int(cfg.get("hop", self.block))
        if self.hop != self.block:
            raise ValueError("현재 구현은 hop == block_size 를 요구합니다")
        ch = cfg["hardware"]["channels"]
        self.ch_err, self.ch_ref = int(ch["error_mic"]), int(ch["reference_mic"])
        self.ch_noise, self.ch_cancel = int(ch["noise_out"]), int(ch["cancel_out"])
        self.reference = reference
        self.digital_reference_lead = digital_reference_lead

        self.in_dev = resolve_alsa_portaudio_device(hw["input"]["card"], hw["input"]["pcm"], "input", 2)
        self.out_dev = resolve_alsa_portaudio_device(hw["output"]["card"], hw["output"]["pcm"], "output", 2)

        self.engine = build_engine(cfg)
        checkpoint_lead = getattr(self.engine, "digital_reference_lead_samples", None)
        validate_digital_reference_lead(
            self.reference, self.digital_reference_lead, checkpoint_lead
        )
        # DL 단독도 같은 실제 S 지연·FIR 꼬리를 기다려야 한다. 엔진에 이 속성이
        # 없다는 이유로 지연을 0으로 가정하지 않고 공용 측정 자산에서 읽는다.
        from ..baselines.fxlms_core import load_secondary_path

        secondary = load_secondary_path(secondary_path_npz(cfg))
        validate_secondary_calibration(cfg, secondary)
        handoff = int(cfg["duct"]["secondary_path"].get("handoff_extra_samples", self.block))
        if secondary.sample_rate != self.fs:
            raise ValueError("S(z) sample_rate가 런타임 sample_rate와 다릅니다")
        if handoff != self.block:
            raise ValueError("런타임 handoff_extra_samples는 실제 1 block과 같아야 합니다")
        self.secondary_total_length = int(secondary.delay_samples + handoff + secondary.fir.size)
        adaptive = getattr(self.engine, "adaptive", self.engine)
        control_length = int(getattr(getattr(adaptive, "controller", None), "control_len", 0))
        self.adaptive_input_memory_samples = self.secondary_total_length + control_length
        self._has_neural_controller = cfg.get("controller", "dl") in {"dl", "hybrid"}
        self.program = NoiseProgram(cfg.get("noise", {}), self.fs)
        self.digital_reference_buffer = DigitalReferenceBuffer(self.digital_reference_lead)

        dc_r = float(cfg["hardware"].get("dc_blocker_r", 0.995))
        self.err_dc, self.ref_dc = DCBlocker(dc_r), DCBlocker(dc_r)

        safety_cfg = cfg.get("safety", {})
        self.safety = SafetySupervisor(safety_cfg, self.fs, self.block)
        fade = int(float(safety_cfg.get("fade_ms", 20.0)) * self.fs / 1000.0)
        self._fade_samples = fade
        self.state = RuntimeState(start_on=False)
        self.state.noise_enabled = bool(cfg.get("noise", {}).get("enabled", True))
        self.anc_gate = FadeGate(fade, initial=0.0)
        self.noise_gate = FadeGate(max(fade, int(0.1 * self.fs)), initial=0.0)
        self.noise_gate.set_target(1.0 if self.state.noise_enabled else 0.0)

        # 네 번째 채널: 1=적응 허용, 0=동결, -1=손상된 NN 입력(폐기+reset).
        # 가중치/NN 상태는 소유자인 추론 스레드만 변경한다.
        self.in_ring = SPSCRing(4, self.hop * 64)      # err, ref_mic, ref_digital, adapt
        self.out_ring = SPSCRing(1, self.hop * 64)

        self.err_meter = PowerEMA(self.fs, 0.4)
        self.ctrl_meter = PowerEMA(self.fs, 0.4)
        self.baseline_power = 0.0
        self.baseline_init = False
        self.step_times_ms: list[float] = []
        self.xruns = 0
        self._last_anc = False
        self._adaptation_hold_samples = 0
        self._baseline_hold_samples = 0
        # infer→callback 통지 전용. 사용자 reset_event는 추론 스레드만 소비한다.
        self._adaptation_rewarm_event = threading.Event()
        self._neural_input_reset_pending = threading.Event()
        self._neural_input_fault_active = False

        self.record_len = int(record_seconds * self.fs)
        self.rec_pos = 0
        self._recording_provenance = None
        if self.record_len > 0:
            self._recording_provenance = capture_recording_provenance(
                self, secondary, secondary_path_npz(cfg)
            )
            self.rec = {
                "err": np.zeros(self.record_len, dtype=np.float32),
                "ref": np.zeros(self.record_len, dtype=np.float32),
                "source": np.zeros(self.record_len, dtype=np.float32),
                "control": np.zeros(self.record_len, dtype=np.float32),
                "anc_gain": np.zeros(self.record_len, dtype=np.float32),
            }
        else:
            self.rec = None

        self._infer_thread: threading.Thread | None = None
        self._stream = None

    # ---------- 콜백 (PortAudio 스레드) ----------

    def _callback(self, indata, outdata, frames, _time_info, status) -> None:
        try:
            if status:
                self.xruns += 1

            mics = pcm_int32_to_float32(indata[:, :2])
            required_channels = [self.ch_err, self.ch_ref] if self.reference == "mic" else [self.ch_err]
            required_input = mics[:, required_channels]
            input_clip_fraction = float(np.mean(
                ~np.isfinite(required_input) | (np.abs(required_input) >= 0.98)
            ))
            input_disrupted = self.reference == "mic" and input_clip_fraction > 0.0
            neural_input_disrupted = input_disrupted and self._has_neural_controller
            if neural_input_disrupted:
                self._neural_input_reset_pending.set()
                if not self._neural_input_fault_active:
                    self.state.messages.put(
                        "REF/ERR 입력 손상: 신경망 상태를 reset하고 ANC를 OFF로 전환합니다. "
                        "입력 복구 후 현장에서 다시 켜세요."
                    )
            self._neural_input_fault_active = neural_input_disrupted
            if self._neural_input_reset_pending.is_set():
                self.state.anc_enabled = False
            # 손상된 블록을 적응/기준선에서 제외하면서 DC 필터 상태의 NaN 전파도 막는다.
            finite_mics = np.nan_to_num(mics, nan=0.0, posinf=0.0, neginf=0.0)
            if input_disrupted:
                self.err_dc.reset()
                self.ref_dc.reset()
                # 측정·녹음에는 실제 유한 PCM을 남긴다. 엔진으로 보낼 블록만
                # 아래에서 0/폐기 표식으로 바꾸어 가짜 감쇠를 기록하지 않는다.
                err = finite_mics[:, self.ch_err].copy()
                ref_mic = finite_mics[:, self.ch_ref].copy()
            else:
                err = self.err_dc.process(finite_mics[:, self.ch_err])
                ref_mic = self.ref_dc.process(finite_mics[:, self.ch_ref])

            noise_gain = self.noise_gate.process(frames)
            self.noise_gate.set_target(1.0 if self.state.noise_enabled else 0.0)
            # 자기생성 소스는 지금 만든 신호를 ref로 즉시 공급하고, 실제 ch0 재생만
            # lead만큼 늦춘다. 게이트도 같은 FIFO를 통과시켜 ON/OFF 전환 중에도
            # ref[t] == source[t+lead] 정렬을 보존한다 (digital-ref 전용).
            future_source = self.program.generate(frames) * noise_gain
            played, future = self.digital_reference_buffer.process(
                np.stack([future_source, noise_gain])
            )
            source, played_noise_gain = played
            ref_digital = future[0]

            # 백로그가 1 hop 을 넘으면 최신으로 재동기 — 언더런에 의한 지연 누적 방지 (#9)
            y_blk, had_data = self.out_ring.pop_latest(frames, keep_backlog=frames)
            y_lim, clip_frac = self.safety.limit_output(y_blk[0])

            if self.state.anc_enabled != self._last_anc:
                self.anc_gate.set_target(1.0 if self.state.anc_enabled else 0.0)
                if self.state.anc_enabled:
                    self._adaptation_hold_samples = max(
                        self._adaptation_hold_samples, self.secondary_total_length + self._fade_samples
                    )
                else:
                    if self.reference == "mic":
                        self._baseline_hold_samples = self.secondary_total_length + self._fade_samples
                self._last_anc = self.state.anc_enabled
            gain = self.anc_gate.process(frames)
            control = y_lim * gain

            rewarm = self._adaptation_rewarm_event.is_set()
            if rewarm:
                self._adaptation_rewarm_event.clear()
                if self.reference == "mic":
                    self._baseline_hold_samples = max(
                        self._baseline_hold_samples, self.secondary_total_length + frames
                    )
            # 합산 clipping/끊긴 출력은 S 지연 뒤의 ERR도 오염한다. 해당 출력
            # 블록 끝부터 FIR 꼬리까지 지난 뒤에만 선형 filtered-x 적응을 재개한다.
            output_disrupted = bool(
                clip_frac > 0.0 or status or not had_data or not np.all(np.isfinite(y_blk))
            )
            if rewarm or output_disrupted:
                self._adaptation_hold_samples = max(
                    self._adaptation_hold_samples, self.secondary_total_length + frames
                )
            if input_disrupted:
                # 손상 REF는 S FIR뿐 아니라 control filter의 filtered-x 탭 이력에도
                # 영향을 준다. 실제 S+W 메모리가 비워질 때까지 갱신을 중단한다.
                self._adaptation_hold_samples = max(
                    self._adaptation_hold_samples, self.adaptive_input_memory_samples + frames
                )

            out = np.zeros((frames, 2), dtype=np.float32)
            out[:, self.ch_noise] = source
            out[:, self.ch_cancel] = control
            outdata[:] = float32_to_pcm_int16(out)

            err_power = self.err_meter.update(err)
            ctrl_power = self.ctrl_meter.update(control)

            full_noise_gain = bool(
                played_noise_gain.size and float(np.min(played_noise_gain)) >= 0.999
            )
            selected_reference = ref_digital if self.reference == "digital" else ref_mic
            reference_power = float(np.mean(selected_reference.astype(np.float64) ** 2))
            finite_inputs = bool(np.all(np.isfinite(err)) and np.isfinite(reference_power))
            # mic 모드의 기준선은 외부 REF가 유효한 ANC OFF 블록에서만 수집한다.
            # OFF 직후에는 실제 상쇄음 꼬리를 기다리고 ON 중 EMA가 섞이지 않게 한다.
            baseline_allowed = full_noise_gain
            baseline_sample = err_power
            if self.reference == "mic":
                baseline_allowed = bool(
                    self._baseline_hold_samples == 0
                    and not status
                    and finite_inputs
                    and input_clip_fraction == 0.0
                    and reference_power > 1.0e-12
                )
                baseline_sample = float(np.mean(err.astype(np.float64) ** 2))
            if float(np.max(gain)) <= 0.001 and baseline_allowed:
                alpha = float(np.exp(-frames / (self.fs * 1.0)))
                if not self.baseline_init:
                    self.baseline_power = baseline_sample
                    self.baseline_init = True
                else:
                    self.baseline_power = alpha * self.baseline_power + (1 - alpha) * baseline_sample
            self._baseline_hold_samples = max(0, self._baseline_hold_samples - frames)

            mute = self.safety.check_block(
                self.state.anc_enabled, clip_frac, err_power, self.baseline_power,
                had_data or not self.state.anc_enabled,
            )
            if mute:
                self.state.anc_enabled = False
            for msg in self.safety.drain_messages():
                self.state.messages.put(msg)

            full_anc_gain = bool(gain.size and float(np.min(gain)) >= 0.999)
            adapt_allowed = fxlms_adaptation_allowed(
                requested=self.state.anc_enabled and not mute,
                full_anc_gain=full_anc_gain,
                full_noise_gain=full_noise_gain,
                hold_samples=self._adaptation_hold_samples,
                output_clip_fraction=clip_frac,
                input_clip_fraction=input_clip_fraction,
                reference_power=reference_power,
                stream_ok=not bool(status) and had_data and finite_inputs,
                reference=self.reference,
            )
            adapt_gate = np.full(
                frames, -1.0 if neural_input_disrupted else float(adapt_allowed), dtype=np.float32
            )
            engine_err = np.zeros_like(err) if input_disrupted else err
            engine_ref = np.zeros_like(ref_mic) if input_disrupted else ref_mic
            self.in_ring.push(np.stack([engine_err, engine_ref, ref_digital, adapt_gate]))
            self._adaptation_hold_samples = max(0, self._adaptation_hold_samples - frames)

            if self.rec is not None and self.rec_pos < self.record_len:
                n = min(frames, self.record_len - self.rec_pos)
                sl = slice(self.rec_pos, self.rec_pos + n)
                self.rec["err"][sl] = err[:n]
                self.rec["ref"][sl] = ref_mic[:n]
                self.rec["source"][sl] = source[:n]
                self.rec["control"][sl] = control[:n]
                self.rec["anc_gain"][sl] = gain[:n]
                self.rec_pos += n

            reduction = float("nan")
            if self.baseline_init and err_power > 0:
                reduction = 10.0 * np.log10((self.baseline_power + 1e-30) / (err_power + 1e-30))
            self.state.latest_stats = {
                "anc": self.state.anc_enabled,
                "err_dbfs": power_to_db(err_power),
                "ctrl_dbfs": power_to_db(ctrl_power),
                "reduction_db": reduction,
                "fxlms_adapt_allowed": adapt_allowed,
                "fxlms_adapt_hold_samples": self._adaptation_hold_samples,
                "input_clip_fraction": input_clip_fraction,
                "underruns": self.out_ring.underruns,
                "drops": self.out_ring.drops,
                "xruns": self.xruns,
                "step_ms": float(np.mean(self.step_times_ms[-50:])) if self.step_times_ms else 0.0,
            }
        except BaseException as exc:      # 콜백 예외 → 안전 정지
            outdata.fill(0)
            self.state.fatal_error = exc
            self.state.quit_event.set()
            raise self.sd.CallbackAbort from exc

    # ---------- 추론 스레드 ----------

    def _step_input_block(self, blk: np.ndarray) -> np.ndarray:
        """callback 블록을 소비한다. 손상 NN 입력은 reset 전후 어느 쪽에도 넣지 않는다."""
        err, ref_mic, ref_digital, adapt_gate = blk
        if np.any(adapt_gate < 0.0) or self._neural_input_reset_pending.is_set():
            self.engine.reset()
            self._adaptation_rewarm_event.set()
            self._neural_input_reset_pending.clear()
            return np.zeros(self.hop, dtype=np.float32)
        ref = ref_digital if self.reference == "digital" else ref_mic
        set_adapt = getattr(self.engine, "set_adapt_enabled", None)
        if set_adapt is not None:
            set_adapt(bool(
                np.all(adapt_gate >= 0.5) and not self._adaptation_rewarm_event.is_set()
            ))
        return self.engine.step(ref.copy(), err.copy())

    def _inference_loop(self) -> None:
        affinity = self.cfg.get("engine", {}).get("cpu_affinity")
        if affinity:
            try:
                os.sched_setaffinity(0, set(int(c) for c in affinity))
            except OSError:
                pass
        while not self.state.quit_event.is_set():
            if self.state.reset_event.is_set():
                self.engine.reset()
                self._adaptation_rewarm_event.set()
                # SPSC: 이 스레드는 in_ring 의 소비자만이다 — out_ring 의 read_pos 는
                # 콜백 소유이므로 건드리지 않는다 (콜백의 pop_latest 가 자연 배출).
                self.in_ring.consumer_reset()
                self.state.reset_event.clear()
            if not self.in_ring.wait_for(self.hop, timeout=0.1):
                continue
            # 추론이 뒤처지면 입력 백로그를 8 hop 까지만 허용 (지연 폭주 방지)
            blk, ok = self.in_ring.pop_latest(self.hop, keep_backlog=self.hop * 8)
            if not ok:
                continue
            t0 = time.perf_counter()
            try:
                y = self._step_input_block(blk)
            except Exception as exc:
                self.state.messages.put(f"엔진 오류: {exc!r} — 무음 출력")
                self.engine.reset()
                self._adaptation_rewarm_event.set()
                self.in_ring.consumer_reset()
                y = np.zeros(self.hop, dtype=np.float32)
            dt = (time.perf_counter() - t0) * 1000.0
            self.step_times_ms.append(dt)
            if len(self.step_times_ms) > 10000:
                del self.step_times_ms[:5000]
            self.out_ring.push(y.reshape(1, -1))

    # ---------- 실행 ----------

    def start(self) -> None:
        self._infer_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="anc-inference"
        )
        self._infer_thread.start()
        self._stream = self.sd.Stream(
            samplerate=self.fs,
            blocksize=self.block,
            device=(self.in_dev, self.out_dev),
            channels=(2, 2),
            dtype=("int32", "int16"),
            latency=(self.latency, self.latency),
            callback=self._callback,
            prime_output_buffers_using_stream_callback=True,
        )
        self._stream.start()

    def stop(self) -> None:
        # 종료 페이드 시퀀스 (안전장치 8)
        self.state.anc_enabled = False
        self.state.noise_enabled = False
        time.sleep(0.2)
        self.state.quit_event.set()
        if self._stream is not None:
            try:
                self._stream.abort()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=1.0)

    def session_data(self) -> dict[str, np.ndarray]:
        if self.rec is None:
            return {}
        n = self.rec_pos
        return {k: v[:n].copy() for k, v in self.rec.items()}


def run_cli(cfg: dict, run_seconds: float, record_path: str | Path | None) -> int:
    out = prepare_recording_path(record_path) if record_path else None
    if out is not None and (not np.isfinite(run_seconds) or run_seconds <= 0):
        raise ValueError("녹음에는 유한한 양수 run_seconds가 필요합니다")
    anc = RealtimeANC(cfg, record_seconds=run_seconds if record_path else 0.0)
    keyboard = KeyboardController(anc.state)

    engine_desc = cfg.get("controller", "dl")
    if engine_desc == "dl":
        engine_desc = f"dl/{cfg.get('engine', {}).get('type', 'torch')}"
    print("=" * 72)
    print(f"Deep ANC 실시간 런타임 | 컨트롤러: {engine_desc} | reference: {cfg.get('reference')}")
    print(f"블록 {anc.block} ({1000*anc.block/anc.fs:.2f}ms) @ {anc.fs}Hz | 시작: ANC OFF")
    print(KeyboardController.help_text())
    print("주의: TPA3116D2 볼륨을 낮춘 상태에서 시작하세요.")
    print("=" * 72)

    anc.start()
    keyboard.start()
    started = time.monotonic()
    next_report = started
    try:
        while not anc.state.quit_event.is_set():
            now = time.monotonic()
            if run_seconds > 0 and now - started >= run_seconds:
                break
            while True:
                try:
                    print(f"\n[명령] {anc.state.messages.get_nowait()}")
                except Exception:
                    break
            if now >= next_report:
                s = anc.state.latest_stats
                if s:
                    red = s.get("reduction_db", float("nan"))
                    red_txt = "  n/a" if not np.isfinite(red) else f"{red:6.2f}"
                    print(
                        f"[{'ON ' if s['anc'] else 'OFF'}] e={s['err_dbfs']:7.2f} dBFS | "
                        f"ctrl={s['ctrl_dbfs']:7.2f} | 저감={red_txt} dB | "
                        f"step={s['step_ms']:5.2f}ms | miss={s['underruns']} | xrun={s['xruns']}"
                    )
                next_report = now + 1.0
            if anc.state.fatal_error is not None:
                raise RuntimeError("오디오 콜백 실패") from anc.state.fatal_error
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        keyboard.stop()
        anc.stop()

    if out is not None:
        data = anc.session_data()
        if data:
            saved = save_recording(out, build_recording_payload(anc, data))
            print(f"세션 저장: {saved}")
    print("종료 — 양 채널 무음.")
    return 0


def run_calibrate(cfg: dict) -> int:
    """--calibrate: 3-스레드 경로 그대로의 실효 상쇄경로 지연 실측 [C1].

    추론 엔진 자리에 '처프 재생 엔진'을 넣어 out_ring→콜백→스피커→에러마이크
    왕복 지연을 상호상관으로 측정하고, 학습에 쓰는 지연(캘리브레이션+핸드오프)과
    비교해 어긋남을 리포트한다.
    """
    from scipy import signal as sp_signal

    from ..dsp.secondary_path import load_secondary_path

    fs = int(cfg["hardware"]["audio"]["sample_rate"])
    seconds = 6.0
    t = np.arange(int(seconds * fs)) / fs
    chirp = (
        0.05 * sp_signal.chirp(t, 100.0, seconds, 2000.0, method="logarithmic")
    ).astype(np.float32)
    fade = int(0.05 * fs)
    chirp[:fade] *= np.linspace(0, 1, fade)
    chirp[-fade:] *= np.linspace(1, 0, fade)

    class ChirpEngine:
        def __init__(self, hop: int) -> None:
            self.hop = hop
            self.pos = 0

        def reset(self) -> None:
            self.pos = 0

        def step(self, ref, err):
            out = np.zeros(self.hop, dtype=np.float32)
            n = min(self.hop, chirp.size - self.pos)
            if n > 0:
                out[:n] = chirp[self.pos : self.pos + n]
                self.pos += n
            return out

    cfg = dict(cfg)
    cfg["noise"] = {"type": "silence"}
    # 측정 모드: 워치독이 처프 출력을 mute 하지 않도록 임계값 무력화 (#13)
    cfg["safety"] = dict(cfg.get("safety", {}))
    cfg["safety"].update({"deadline_miss_mute": 10**9, "divergence_ratio": 1e12,
                          "clip_streak_mute": 10**9})
    anc = RealtimeANC(cfg, record_seconds=seconds + 2.0)
    anc.engine = ChirpEngine(anc.hop)
    anc.state.anc_enabled = True          # 게이트를 열어 처프를 내보낸다
    anc.anc_gate.set_target(1.0)
    print(f"실효 지연 측정: 처프 {seconds:.0f}s 재생 (상쇄 스피커 ch1) ...")
    anc.start()
    time.sleep(seconds + 1.5)
    anc.stop()

    data = anc.session_data()
    err = data["err"].astype(np.float64)
    ctrl = data["control"].astype(np.float64)
    if np.max(np.abs(ctrl)) < 1e-6:
        print("[실패] 출력이 재생되지 않았습니다", file=sys.stderr)
        return 1
    corr = sp_signal.fftconvolve(err, ctrl[::-1], mode="full")
    lag = int(np.argmax(np.abs(corr))) - (ctrl.size - 1)
    sp = load_secondary_path(secondary_path_npz(cfg))
    handoff = int(
        cfg["duct"]["secondary_path"].get("handoff_extra_samples", DEFAULT_HANDOFF_SAMPLES)
    )
    # rec['control'] 은 블록이 실제 "출력"되는 콜백 시점(핸드오프 이후)에 기록되므로,
    # 측정 lag 의 기대값은 캘리브레이션 지연(1342)뿐이다 — 핸드오프는 별도 합산 (#11)
    expected = sp.delay_samples
    total_training = sp.delay_samples + handoff
    print(f"측정 지연(출력→에러마이크): {lag}샘플 ({1000*lag/fs:.2f}ms)")
    print(f"캘리브레이션 기대값        : {expected}샘플 | 차이 {lag - expected:+d}샘플 "
          f"({1000*(lag-expected)/fs:+.2f}ms)")
    print(f"학습 플랜트 총지연         : 측정 {lag} + 핸드오프 {handoff} = {lag + handoff} "
          f"(설정값 {total_training})")
    if abs(lag - expected) > 512:
        print(
            "→ 차이가 지터 증강 범위(+512)를 벗어납니다. duct.yaml 의 "
            "handoff_extra_samples 조정 또는 재캘리브레이션 후 파인튜닝을 권장합니다."
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--run-seconds", type=float, default=None)
    parser.add_argument("--record", default=None, help="세션 npz 저장 경로")
    parser.add_argument("--calibrate", action="store_true", help="실효 지연 측정 모드")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument(
        "--input-probe-seconds",
        type=float,
        default=2.0,
        help="스피커 출력 전 무출력 마이크 사전점검 길이",
    )
    args = parser.parse_args()

    if args.list_devices:
        print(format_sounddevice_devices())
        return 0

    cfg = load_runtime_config(args.config, args.overrides)
    run_seconds = args.run_seconds if args.run_seconds is not None else float(cfg.get("run_seconds", 0.0))
    record = args.record or cfg.get("record")
    if not args.calibrate and record:
        if not np.isfinite(run_seconds) or run_seconds <= 0:
            parser.error("--record 는 유한한 양수 --run-seconds 가 필요합니다")
        try:
            record = prepare_recording_path(record)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    try:
        if not input_preflight(cfg, seconds=args.input_probe_seconds):
            return 2
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] 입력 사전점검 실패: {exc}", file=sys.stderr)
        return 2
    if args.calibrate:
        return run_calibrate(cfg)
    return run_cli(cfg, run_seconds, record)


if __name__ == "__main__":
    raise SystemExit(main())
