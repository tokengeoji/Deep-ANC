"""실시간 Deep ANC 런타임 — 3-스레드 구조 (콜백 / 추론 / 제어).

  python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml
  python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml --set controller=fxlms
  python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml --calibrate

구조 (docs/06):
  [콜백]   입력 변환/DC차단 → in_ring, 소음(ch0) 생성, out_ring→리미터/게이트→ch1 출력
  [추론]   in_ring 에서 hop 단위 소비 → engine.step → out_ring  (콜백은 절대 대기 안 함)
  [제어]   키보드, 1초 통계, 워치독 메시지
파이프라인 핸드오프 지연 = 1 hop — 학습 플랜트의 handoff_extra_samples 와 정합 [C1].
이 1 hop 은 이제 주석이 아니라 **강제**다: 입출력 링버퍼의 백로그 허용치를
safety.PipelineHandoffBudget 이 단 한 곳에서 유도하고, 비대칭이면 생성 자체가
거부된다 (예전에는 입력만 8 hop = 42.7 ms 여서 추론이 뒤처지면 상쇄가 증폭이 됐다).
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
from ..config import REPO_ROOT, load_runtime_config
from ..dsp.filters import DCBlocker
from .engines import build_engine, secondary_path_npz
from .noise_gen import DigitalReferenceBuffer, NoiseProgram
from .ring_buffer import SPSCRing
from .safety import (
    BlockObservation,
    FadeGate,
    PipelineHandoffBudget,
    PowerEMA,
    SafetySupervisor,
)
from .ui import KeyboardController, RuntimeState


def power_to_db(power: float, floor_db: float = -200.0) -> float:
    if not np.isfinite(power) or power <= 0.0:
        return floor_db
    return max(floor_db, 10.0 * float(np.log10(power)))


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
) -> bool:
    """FxLMS가 현재 ERR 블록으로 갱신해도 되는 안전 조건."""
    return bool(
        requested
        and full_anc_gain
        and full_noise_gain
        and int(hold_samples) == 0
        and float(output_clip_fraction) == 0.0
        and float(input_clip_fraction) == 0.0
        and float(reference_power) > 1.0e-12
        and stream_ok
    )


def input_preflight(cfg: dict, seconds: float = 2.0) -> bool:
    """스피커를 열기 전에 필수 I2S 입력 채널이 살아 있는지 확인한다."""
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


_ENGINE_ARTIFACT_KEYS = {
    "torch": ("ckpt",),
    "ort": ("onnx",),
    "trt": ("plan",),
}
"""엔진 종류별로 **실제 로드에 쓰이는** 키. 나머지 키는 읽히지 않는다."""


def engine_artifact_preflight(cfg: dict, *, require_all: bool = True) -> list[str]:
    """``engine`` 블록이 가리키는 파일이 실제로 존재하는지 오디오를 열기 전에 본다.

    왜 활성 엔진만 보면 안 되는가
    ----------------------------
    2026-08-05 실측: ``configs/runtime_tiny.yaml`` 의 ``plan:
    runs/export/tiny_fp16.plan`` 은 **존재하지 않는 파일**이었다. 실제 파일 이름은
    ``tiny_corrected_fp16.plan`` 이다. 그런데 ``engine.type`` 이 ``ort`` 라 이 키는
    한 번도 읽히지 않았고, 따라서 **조용히 썩어 있었다** — ``trt`` 로 바꾸는 순간
    터졌을 것이고, 그 시점은 대개 실기 앞이다.

    같은 종류의 부패가 ``configs/runtime.yaml`` 에도 있었다(``model.onnx`` /
    ``model_fp16.plan`` 둘 다 없음).

    그래서 기본값 ``require_all=True`` 는 **선언된 모든 아티팩트**를 검사한다.
    "지금 안 쓰니까 괜찮다"는 것이 바로 이 결함이 4개월 살아남은 이유다. 설정에
    적혀 있으면 존재해야 하고, 없으면 그 줄을 지워야 한다.

    반환은 사람이 읽을 문제 목록이다. 비어 있으면 통과다.
    """

    engine = (cfg or {}).get("engine", {}) or {}
    kind = str(engine.get("type", "torch"))
    if kind not in _ENGINE_ARTIFACT_KEYS:
        return [f"알 수 없는 engine.type={kind!r}; 허용={sorted(_ENGINE_ARTIFACT_KEYS)}"]

    active_keys = _ENGINE_ARTIFACT_KEYS[kind]
    checked = (
        tuple(key for key in ("ckpt", "onnx", "plan") if engine.get(key))
        if require_all
        else active_keys
    )
    problems: list[str] = []
    for key in active_keys:
        if not engine.get(key):
            problems.append(
                f"engine.type={kind} 인데 engine.{key} 가 비었습니다 — 로드할 것이 없습니다"
            )
    for key in checked:
        value = str(engine.get(key, ""))
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            role = "활성" if key in active_keys else "미사용(지금은 읽히지 않음)"
            problems.append(
                f"engine.{key} 아티팩트가 없습니다 [{role}]: {value} — "
                "설정에 적혀 있으면 존재해야 합니다. 쓰지 않는다면 그 줄을 지우세요"
            )
    return problems


def require_engine_artifacts(cfg: dict, *, require_all: bool = True) -> None:
    """:func:`engine_artifact_preflight` 를 실패 폐쇄로 감싼다."""

    problems = engine_artifact_preflight(cfg, require_all=require_all)
    if problems:
        raise FileNotFoundError(
            "런타임 엔진 아티팩트 preflight 실패:\n- " + "\n- ".join(problems)
        )


def validate_digital_reference_lead(
    reference: str,
    configured_lead: int,
    checkpoint_lead: int | None = None,
) -> int:
    """reference 모드와 학습/배포 lead 정합을 검증하고 정규화된 값을 반환한다."""
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
        self.program = NoiseProgram(cfg.get("noise", {}), self.fs)
        self.digital_reference_buffer = DigitalReferenceBuffer(self.digital_reference_lead)

        dc_r = float(cfg["hardware"].get("dc_blocker_r", 0.995))
        self.err_dc, self.ref_dc = DCBlocker(dc_r), DCBlocker(dc_r)

        self.safety = SafetySupervisor(cfg.get("safety", {}), self.fs, self.block)
        # 페이드 길이도 검증된 설정에서만 읽는다 (cfg.get 재유도 금지).
        fade = int(self.safety.limits.fade_ms * self.fs / 1000.0)
        self._fade_samples = fade
        # 링버퍼 백로그 예산의 단일 출처. 입력 8 hop / 출력 1 hop 의 비대칭은
        # 여기서 구조적으로 불가능하다 (safety.PipelineHandoffBudget 참조).
        self.handoff_budget = PipelineHandoffBudget.derive(
            duct_cfg=cfg.get("duct", {}), hop=self.hop
        )
        self.state = RuntimeState(start_on=False)
        self.anc_gate = FadeGate(fade, initial=0.0)
        self.noise_gate = FadeGate(max(fade, int(0.1 * self.fs)), initial=0.0)
        self.noise_gate.set_target(1.0)

        # 네 번째 채널은 callback이 판정한 FxLMS 적응 허용 플래그다. 가중치
        # 소유자인 추론 스레드만 이 플래그를 읽어 adapt 상태를 바꾼다.
        self.in_ring = SPSCRing(4, self.hop * 64)      # err, ref_mic, ref_digital, adapt
        self.out_ring = SPSCRing(1, self.hop * 64)

        self.err_meter = PowerEMA(self.fs, 0.4)
        self.ctrl_meter = PowerEMA(self.fs, 0.4)
        self.baseline_power = 0.0
        self.baseline_init = False
        self._last_input_drops = 0
        self.step_times_ms: list[float] = []
        self.xruns = 0
        self._last_anc = False
        self._adaptation_hold_samples = 0

        self.record_len = int(record_seconds * self.fs)
        self.rec_pos = 0
        if self.record_len > 0:
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
            err = self.err_dc.process(mics[:, self.ch_err])
            ref_mic = self.ref_dc.process(mics[:, self.ch_ref])

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

            # 백로그가 예산을 넘으면 최신으로 재동기 — 언더런에 의한 지연 누적 방지 (#9).
            # 허용치는 handoff_budget 이 유일하게 유도한다 (입출력 대칭 강제).
            y_blk, had_data = self.out_ring.pop_latest(
                frames, keep_backlog=self.handoff_budget.output_keep_backlog_samples
            )
            out_report = self.safety.limit_output(y_blk[0])
            y_lim, clip_frac = out_report.signal, out_report.clipped_fraction

            if self.state.anc_enabled != self._last_anc:
                self.anc_gate.set_target(1.0 if self.state.anc_enabled else 0.0)
                if self.state.anc_enabled:
                    secondary_total = int(
                        getattr(self.engine, "secondary_total_length", 0)
                    )
                    self._adaptation_hold_samples = secondary_total + self._fade_samples
                else:
                    self._adaptation_hold_samples = 0
                self._last_anc = self.state.anc_enabled
            gain = self.anc_gate.process(frames)
            control = y_lim * gain

            out = np.zeros((frames, 2), dtype=np.float32)
            out[:, self.ch_noise] = source
            out[:, self.ch_cancel] = control
            outdata[:] = float32_to_pcm_int16(out)

            err_power = self.err_meter.update(err)
            ctrl_power = self.ctrl_meter.update(control)

            # 베이스라인: ANC 게이트가 닫혀 있는 구간의 에러 파워.
            # 소음 게이트 조건을 빼는 이유 — 외부 소음원으로 운용할 때는
            # played_noise_gain 이 0 이라 베이스라인이 영원히 안 잡히고, 그러면
            # 발산 워치독이 fail-closed 로 ANC 를 끈다. "ANC 없이 마이크에 들어오는
            # 파워"가 발산 판정에 필요한 전부다. 유효성(무음 여부)은 baseline_floor_power
            # 하한으로 판정한다 — 판정 규칙의 단일 출처는 SafetySupervisor 다.
            if float(np.max(gain)) <= 0.001:
                alpha = float(np.exp(-frames / (self.fs * 1.0)))
                if not self.baseline_init:
                    self.baseline_power = err_power
                    self.baseline_init = True
                else:
                    self.baseline_power = alpha * self.baseline_power + (1 - alpha) * err_power

            input_drops = int(self.in_ring.drops)
            stale_input = max(0, input_drops - self._last_input_drops)
            self._last_input_drops = input_drops
            verdict = self.safety.check_block(
                BlockObservation(
                    anc_on=bool(self.state.anc_enabled),
                    output=out_report,
                    error_power=float(err_power),
                    baseline_power=float(self.baseline_power),
                    baseline_valid=self.safety.baseline_is_valid(
                        self.baseline_power, self.baseline_init
                    ),
                    had_output_data=bool(had_data),
                    stale_input_samples=stale_input,
                )
            )
            mute = verdict.mute
            if mute:
                self.state.anc_enabled = False
            for msg in verdict.messages:
                self.state.messages.put(msg)

            if self._adaptation_hold_samples > 0:
                self._adaptation_hold_samples = max(
                    0, self._adaptation_hold_samples - frames
                )
            full_anc_gain = bool(gain.size and float(np.min(gain)) >= 0.999)
            full_noise_gain = bool(
                played_noise_gain.size and float(np.min(played_noise_gain)) >= 0.999
            )
            input_clip_fraction = float(
                np.mean(np.abs(mics[:, self.ch_err]) >= 0.98)
            )
            selected_reference = ref_digital if self.reference == "digital" else ref_mic
            reference_power = float(
                np.mean(selected_reference.astype(np.float64) ** 2)
            )
            adapt_allowed = fxlms_adaptation_allowed(
                requested=self.state.anc_enabled and not mute,
                full_anc_gain=full_anc_gain,
                full_noise_gain=full_noise_gain,
                hold_samples=self._adaptation_hold_samples,
                output_clip_fraction=clip_frac,
                input_clip_fraction=input_clip_fraction,
                reference_power=reference_power,
                stream_ok=not bool(status) and had_data,
            )
            adapt_gate = np.full(frames, float(adapt_allowed), dtype=np.float32)
            self.in_ring.push(np.stack([err, ref_mic, ref_digital, adapt_gate]))

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
                "underruns": self.out_ring.underruns,
                "drops": self.out_ring.drops,
                "stale_input_drops": input_drops,
                "watchdog_trips": {
                    item.value: count
                    for item, count in self.safety.trip_counts.items()
                    if count
                },
                "xruns": self.xruns,
                "step_ms": float(np.mean(self.step_times_ms[-50:])) if self.step_times_ms else 0.0,
            }
        except BaseException as exc:      # 콜백 예외 → 안전 정지
            outdata.fill(0)
            self.state.fatal_error = exc
            self.state.quit_event.set()
            raise self.sd.CallbackAbort from exc

    # ---------- 추론 스레드 ----------

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
                # SPSC: 이 스레드는 in_ring 의 소비자만이다 — out_ring 의 read_pos 는
                # 콜백 소유이므로 건드리지 않는다 (콜백의 pop_latest 가 자연 배출).
                self.in_ring.consumer_reset()
                self.state.reset_event.clear()
            if not self.in_ring.wait_for(self.hop, timeout=0.1):
                continue
            # 입력 백로그 허용치는 출력과 **같아야 한다**. 예전에는 여기만 8 hop
            # (42.7 ms) 이어서 추론이 뒤처지면 실효 핸드오프가 학습 가정의 8배로
            # 조용히 늘었다 — 그 지연에서는 상쇄가 증폭이 된다.
            blk, ok = self.in_ring.pop_latest(
                self.hop, keep_backlog=self.handoff_budget.input_keep_backlog_samples
            )
            if not ok:
                continue
            err, ref_mic, ref_digital, adapt_gate = blk
            ref = ref_digital if self.reference == "digital" else ref_mic
            t0 = time.perf_counter()
            try:
                set_adapt = getattr(self.engine, "set_adapt_enabled", None)
                if set_adapt is not None:
                    set_adapt(bool(np.all(adapt_gate >= 0.5)))
                y = self.engine.step(ref.copy(), err.copy())
            except Exception as exc:
                self.state.messages.put(f"엔진 오류: {exc!r} — 무음 출력")
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


def run_cli(cfg: dict, run_seconds: float, record_path: str | None) -> int:
    anc = RealtimeANC(cfg, record_seconds=run_seconds if record_path else 0.0)
    keyboard = KeyboardController(anc.state)

    engine_desc = cfg.get("controller", "dl")
    if engine_desc == "dl":
        engine_desc = f"dl/{cfg.get('engine', {}).get('type', 'torch')}"
    print("=" * 72)
    print(f"Deep ANC 실시간 런타임 | 컨트롤러: {engine_desc} | reference: {cfg.get('reference')}")
    print(f"블록 {anc.block} ({1000*anc.block/anc.fs:.2f}ms) @ {anc.fs}Hz | 시작: ANC OFF")
    budget = anc.handoff_budget
    print(
        f"실효 핸드오프 {budget.effective_handoff_samples}샘플 "
        f"({1000*budget.effective_handoff_samples/anc.fs:.2f}ms) "
        f"| 백로그 허용 입력={budget.input_keep_backlog_samples} "
        f"출력={budget.output_keep_backlog_samples}"
    )
    for note in anc.safety.limits.legacy_notes:
        print(f"[설정 경고] {note}")
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

    if record_path:
        data = anc.session_data()
        if data:
            out = Path(record_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out.with_suffix(".npz"), fs=anc.fs, **data)
            print(f"세션 저장: {out.with_suffix('.npz')}")
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
    from ..dsp.timing import handoff_samples_from_config  # handoff 의 단일 출처

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
    # 측정 모드: 음향 성능 워치독(포화/RMS/데드라인/발산/백로그)은 자문으로 내린다.
    # 임계값을 1e12 로 무력화하던 예전 방식과 달리 **하드웨어 보호(DC·NaN)는 그대로
    # mute 한다** — 캘리브레이션 중이라고 보이스코일에 DC 를 흘려도 되는 것은 아니다.
    cfg["safety"] = dict(cfg.get("safety", {}))
    cfg["safety"]["measurement_mode"] = True
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
    handoff = handoff_samples_from_config(cfg["duct"])
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
    # 엔진 아티팩트 확인은 **오디오 장치를 열기 전에** 한다. 마이크 프로브보다도
    # 먼저인 이유는 이 검사가 하드웨어를 전혀 건드리지 않고 즉시 끝나기 때문이다 —
    # 없는 파일 때문에 실패할 실행에 스피커를 울릴 이유가 없다.
    try:
        require_engine_artifacts(cfg)
    except FileNotFoundError as exc:
        print(f"[중단] {exc}", file=sys.stderr)
        return 2
    try:
        if not input_preflight(cfg, seconds=args.input_probe_seconds):
            return 2
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] 입력 사전점검 실패: {exc}", file=sys.stderr)
        return 2
    if args.calibrate:
        return run_calibrate(cfg)
    run_seconds = args.run_seconds if args.run_seconds is not None else float(cfg.get("run_seconds", 0.0))
    record = args.record or cfg.get("record")
    if record and run_seconds <= 0:
        parser.error("--record 는 녹음 버퍼 크기 산정을 위해 --run-seconds 가 필요합니다")
    return run_cli(cfg, run_seconds, record)


if __name__ == "__main__":
    raise SystemExit(main())
