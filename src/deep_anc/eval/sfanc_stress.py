"""고정된 SFANC의 연속 처리·지연 민감도 진단. 하드웨어/최적 성능 증명이 아니다."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import signal
import torch

from deep_anc.baselines.filter_bank import new_output_directory
from deep_anc.baselines.sfanc_design import fit_control_fir
from deep_anc.baselines.sfanc_stream import StreamingFIRBank
from deep_anc.data.sfanc_sources import iter_librispeech_crops
from deep_anc.eval.sfanc_fxnlms import run_fxnlms
from deep_anc.train.sfanc_experiment import load_selector
from deep_anc.train.sfanc_selector import predict_selector


def _integer(value, name, lower=0, upper=1_048_576):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name}: {lower}~{upper} 정수가 필요합니다")
    return value


def _vector(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name}: 실수 수치 배열 필요")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 1 or not 1 <= array.size <= 1_048_576 or not np.isfinite(array).all():
        raise ValueError(f"{name}: 크기가 제한된 유한 1차원 배열이 필요합니다")
    return array


def _sha(array):
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


@dataclass(frozen=True)
class StressConfig:
    samples: int = 32768
    seed: int = 20260921
    primary_delays: tuple[int, ...] = (48, 128)
    additional_delays: tuple[int, ...] = (0, 8, 16, 32, 64, 128)
    block_samples: int = 32
    selector_delay_samples: int = 64
    transition_samples: int = 128
    control_limit: float = 0.2
    fxnlms_mu: float = 0.05
    fit_samples: int = 16384

    def __post_init__(self):
        for name in ("samples", "fit_samples"):
            _integer(getattr(self, name), name, 2048, 131072)
            if getattr(self, name) % 256:
                raise ValueError(f"{name}: 256의 배수 필요")
        _integer(self.seed, "seed", 0, 2**32-1)
        _integer(self.block_samples, "block_samples", 1, 4096)
        for name in ("selector_delay_samples", "transition_samples"):
            _integer(getattr(self, name), name, 0, 32768)
        for name in ("primary_delays", "additional_delays"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 16:
                raise ValueError(f"{name}: 1~16개 지연 필요")
            for value in values:
                _integer(value, name, 0, 2048)
            if len(set(values)) != len(values):
                raise ValueError(f"{name}: 중복 지연 금지")
        for name in ("control_limit", "fxnlms_mu"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not 0 < value <= (2 if name == "fxnlms_mu" else 1):
                raise ValueError(f"{name}: 범위 내 유한 양수 필요")


def control_residual(disturbance, control, secondary, additional_delay_samples):
    """S 원본을 그대로 적용하며 명시적 추가 지연을 정확히 한 번 적용."""
    d, u, s = (_vector(value, name) for value, name in
               ((disturbance, "disturbance"), (control, "control"), (secondary, "secondary")))
    delay = _integer(additional_delay_samples, "additional_delay_samples")
    if d.shape != u.shape:
        raise ValueError("d/u 길이 불일치")
    delayed = np.zeros_like(u)
    if delay < len(u):
        delayed[delay:] = u[:len(u)-delay]
    error = d + signal.lfilter(s, [1.0], delayed)
    if not np.isfinite(error).all():
        raise FloatingPointError("플랜트 계산 overflow")
    return error


def selected_stream(reference, coefficients, choose, *, window_samples,
                    block_samples=32, selector_delay_samples=0,
                    transition_samples=0, control_limit=0.2):
    """완료된 과거 창만 선택기에 전달. CNN 실제 wall-time과 다른 가상 scheduler.

    관측 [t-window,t) 결과는 t+selector_delay에서 적용한다. 첫 선택 전에는
    zero 후보로 시작한다. 연산은 순차 시뮬레이션이며 실제 비동기 스레드가 아니다.
    """
    x = _vector(reference, "reference")
    window = _integer(window_samples, "window_samples", 1, 32768)
    block = _integer(block_samples, "block_samples", 1, 4096)
    lag = _integer(selector_delay_samples, "selector_delay_samples", 0, 32768)
    raw_bank = np.asarray(coefficients)
    if raw_bank.dtype.kind not in "iuf":
        raise ValueError("bank: 실수 수치 배열 필요")
    bank = np.asarray(raw_bank, dtype=np.float64)
    if bank.ndim != 2 or min(bank.shape) < 1 or not np.all(bank[0] == 0):
        raise ValueError("0번 후보는 명시적 무제어 FIR이어야 합니다")
    controller = StreamingFIRBank(bank, initial_index=0,
                                 transition_samples=transition_samples, control_limit=control_limit)
    pending, decisions, chunks = [], [], []
    t, next_observation = 0, window
    while t < len(x):
        if t == next_observation:
            past = x[t-window:t].copy()
            index = 0 if float(np.mean(past**2)) <= 1e-12 else choose(past)
            index = _integer(index, "selector index", 0, len(bank)-1)
            event = {"observed_from_sample": t-window, "observed_until_sample": t,
                     "apply_at_sample": t+lag, "candidate": index}
            decisions.append(event)
            pending.append((t+lag, index))
            next_observation += window
        while pending and pending[0][0] == t:
            controller.request_filter(pending.pop(0)[1])
        stop = min(len(x), t+block, next_observation,
                   pending[0][0] if pending else len(x))
        chunks.append(controller.process(x[t:stop]))
        t = stop
    return {"control": np.concatenate(chunks), "decisions": decisions,
            "diagnostics": dict(controller.diagnostics)}


def band_metrics(disturbance, residual, *, sample_rate=16000):
    """FFT 직사각 창 대역 파워. 미약한 d/무신호는 dB null, 새 에너지는 별도 기록."""
    d, e = _vector(disturbance, "disturbance"), _vector(residual, "residual")
    if d.shape != e.shape:
        raise ValueError("d/e 길이 불일치")
    n = len(d)
    _integer(sample_rate, "sample_rate", 1, 192000)
    frequency = np.fft.rfftfreq(n, 1 / sample_rate)
    weights = np.full(len(frequency), 2.0)
    weights[0] = 1
    if n % 2 == 0:
        weights[-1] = 1
    dp = np.abs(np.fft.rfft(d))**2 * weights / n**2
    ep = np.abs(np.fft.rfft(e))**2 * weights / n**2
    if not np.isfinite(dp).all() or not np.isfinite(ep).all():
        raise FloatingPointError("대역 파워 계산 overflow")
    masks = {"fullband": np.ones(len(frequency), dtype=bool),
             "below_1000": frequency < 1000,
             "target_1000_1600": (frequency >= 1000) & (frequency < 1600),
             "above_1600": frequency >= 1600}
    rows = []
    for name, mask in masks.items():
        baseline, error = float(dp[mask].sum()), float(ep[mask].sum())
        rows.append({"band": name, "baseline_power": baseline, "residual_power": error,
                     "reduction_db": float(10*np.log10(baseline/error)) if min(baseline, error) > 1e-14 else None,
                     "emergent_error_energy": bool(baseline <= 1e-14 < error)})
    return rows


def diagnostic_sources(config, speech_manifest=None, librispeech_root=None):
    """새 난수 진단 + 기존 heldout 음성의 연속 결합. 실제 음악을 대신하지 않는다."""
    n, fs = config.samples, 16000
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 200]))
    white = rng.normal(0, 0.04, n)
    def band(low, high):
        values = signal.sosfilt(signal.butter(4, (low, high), btype="bandpass", fs=fs, output="sos"),
                               rng.normal(size=n+1024))[1024:]
        return values * (0.04 / np.sqrt(np.mean(values**2)))
    low, high = band(80, 800), band(1000, 1600)
    time = np.arange(n) / fs
    tone = .025 * (np.sin(2*np.pi*300*time) + np.sin(2*np.pi*1100*time) + np.sin(2*np.pi*1500*time))
    switch = np.concatenate([low[:n//2], high[n//2:]])
    for name, x in (("white_unpredictable", white), ("band_low", low), ("band_high", high),
                    ("abrupt_band_switch", switch), ("multitone_not_music", tone),
                    ("silence", np.zeros(n))):
        yield name, x, {"kind": "procedural", "new_seed_namespace": [config.seed, 200],
                        "music": False, "measured_ref_err": False, "sha256": _sha(x)}
    if speech_manifest is not None:
        chunks, metadata, remaining = [], [], n
        for audio, meta in iter_librispeech_crops(speech_manifest, "test", root=librispeech_root):
            take = min(remaining, len(audio))
            chunks.append(audio[:take].astype(np.float64))
            metadata.append({**meta, "used_samples": take})
            remaining -= take
            if not remaining:
                break
        if remaining:
            raise ValueError("요청 길이만큼의 heldout speech가 없습니다; 반복/패딩하지 않습니다")
        x = np.concatenate(chunks)
        yield "heldout_speech_concatenated", x, {
            "kind": "librispeech", "sources": metadata, "sha256": _sha(x),
            "reuses_previous_test_crops_for_fixed_diagnostic": True,
            "independent_new_test_claim": False, "concatenation_boundaries_are_synthetic": True,
            "normalization": False, "resampling": False, "measured_ref_err": False}


def run_stress(run_directory, out, config=StressConfig(), *, librispeech_root=None):
    """모델/임계값 재선택 없이 고정 설정으로 진단 보고서를 새 폴더에 기록."""
    destination = new_output_directory(out)
    source = Path(run_directory).resolve()
    model, bank, artifact = load_selector(source)
    prior = json.loads((source / "report.json").read_text())
    if artifact["sample_rate"] != 16000 or prior["config"]["secondary_kind"] != "omap_measured":
        raise ValueError("16 kHz OMAP 원본 S 사전학습 artifact 전용")
    from deepanc.calibration import load_secondary_path, DEFAULT_RIR
    secondary = load_secondary_path(sample_rate=16000)
    with np.load(source / "bank.npz", allow_pickle=False) as data:
        if not np.array_equal(data["secondary"], secondary):
            raise ValueError("bank S가 OMAP 원본 전체 탭과 다릅니다")
    window = artifact["window_samples"]
    path_memory = max(max(config.primary_delays)+73,
                      max(config.additional_delays)+len(secondary)+bank.shape[1]-2)
    warmup = window + config.selector_delay_samples + config.transition_samples + path_memory
    if config.samples <= warmup + 256 or config.fit_samples <= path_memory + 256:
        raise ValueError("평가/설계 길이가 경로보다 짧습니다")
    fixed_index = prior["training"]["fixed_candidate_from_train"]
    _integer(fixed_index, "fixed_candidate_from_train", 0, len(bank)-1)
    speech = None
    if librispeech_root is not None:
        speech = json.loads((source / "speech_manifest.json").read_text())
    destination.mkdir(parents=True, exist_ok=False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    rows, runs, source_records = [], [], {}
    try:
        choose = lambda x: int(predict_selector(model, x, n_fft=artifact["n_fft"], device="cpu")[0])
        sources = []
        for name, x, metadata in diagnostic_sources(config, speech, librispeech_root):
            selected = selected_stream(x, bank, choose, window_samples=window,
                          block_samples=config.block_samples, selector_delay_samples=config.selector_delay_samples,
                          transition_samples=config.transition_samples, control_limit=config.control_limit)
            fixed_raw = signal.lfilter(bank[fixed_index], [1.0], x)
            sources.append((name, x, selected, fixed_raw))
            source_records[name] = metadata
        # 같은 독립 fullband 설계 음원으로 각 알려진 합성 시나리오에 새 FIR을 계산한다.
        # test 최소잔차 oracle가 아니며, SFANC처럼 동결된 모델과 정보 조건이 다르다.
        design_x = np.random.default_rng(np.random.SeedSequence([config.seed, 300])).normal(0, .04, config.fit_samples)
        fitted_records = []
        for primary_delay, delay in itertools.product(config.primary_delays, config.additional_delays):
            primary = np.zeros(primary_delay+74)
            primary[[primary_delay, primary_delay+73]] = [.05, -.02]
            fit = fit_control_fir(design_x, signal.lfilter(primary, [1.0], design_x), secondary,
                     control_length=bank.shape[1], secondary_delay_samples=delay,
                     warmup_samples=max(len(primary)-1, len(secondary)+delay+bank.shape[1]-2),
                     regularization=prior["config"]["regularization"],
                     effort_penalty=prior["config"]["effort_penalty"])
            fitted_records.append({"primary_delay_samples": primary_delay, "additional_delay_samples": delay,
                                   "coefficient_sha256": _sha(fit.coefficients), "fit": fit.diagnostics})
            for name, x, selected, fixed_raw in sources:
                d = signal.lfilter(primary, [1.0], x)
                adaptive = run_fxnlms(x, d, secondary, additional_delay_samples=delay,
                             control_length=bank.shape[1], block_samples=config.block_samples,
                             mu=config.fxnlms_mu, control_limit=config.control_limit)
                fitted_raw = signal.lfilter(fit.coefficients, [1.0], x)
                methods = {
                    "zero": (np.zeros_like(x), {"limited_samples": 0}),
                    "frozen_train_best_fir": (np.clip(fixed_raw, -config.control_limit, config.control_limit),
                                            {"raw_peak": float(np.max(np.abs(fixed_raw))),
                                             "limited_samples": int(np.count_nonzero(np.abs(fixed_raw)>config.control_limit))}),
                    "frozen_sfanc": (selected["control"], {**selected["diagnostics"], "decisions": selected["decisions"]}),
                    "scenario_fitted_fir_not_frozen": (np.clip(fitted_raw, -config.control_limit, config.control_limit),
                                            {"raw_peak": float(np.max(np.abs(fitted_raw))),
                                             "limited_samples": int(np.count_nonzero(np.abs(fitted_raw)>config.control_limit))}),
                    "cold_block_fxnlms": (adaptive["control"], {key: value for key, value in adaptive.items()
                                              if key not in ("control", "raw_control", "residual", "final_weights")}),
                }
                for method, (u, diagnostics) in methods.items():
                    error = control_residual(d, u, secondary, delay)
                    if method == "cold_block_fxnlms" and not np.allclose(error, adaptive["residual"], atol=1e-10, rtol=1e-9):
                        raise ValueError("FxNLMS 플랜트와 독립 전체 컨볼루션 불일치")
                    keys = {"source": name, "method": method, "primary_delay_samples": primary_delay,
                            "additional_delay_samples": delay}
                    runs.append({**keys, "output_peak": float(np.max(np.abs(u))), "diagnostics": diagnostics})
                    periods = [("whole_stream", 0, len(x)), ("after_common_startup", warmup, len(x))]
                    periods += [(f"window_{i//window}", i, min(i+window, len(x)))
                                for i in range(0, len(x), window)]
                    for period, start, end in periods:
                        rows.extend({**keys, "period": period, "start_sample": start, "end_sample": end, **metric}
                                    for metric in band_metrics(d[start:end], error[start:end]))
            print(f"[지연 진단] 합성 P={primary_delay}/16000초, 추가 delay={delay}/16000초", flush=True)
        report = {
            "schema": "sfanc_continuous_delay_diagnostic.v1", "config": asdict(config),
            "sample_rate": 16000, "physical_performance_claim_allowed": False,
            "deployment_allowed": False, "real_time_claim_allowed": False,
            "audio_devices_opened": False, "continuous_simulation_evaluated": True,
            "live_connection_implemented": False, "primary_is_synthetic": True, "music_evaluated": False,
            "model_retrained": False, "test_used_for_tuning": False,
            "hybrid_ancnet_attenuation_comparison": {"status": "not_evaluated",
                 "reason": "같은 OMAP 16 kHz 경로/학습 조건의 검증된 HybridANCNet artifact 없음; legacy를 대체 투입하지 않음"},
            "input_artifacts": {**prior["artifacts"], "directory": str(source),
                 "source_report_sha256": hashlib.sha256((source/"report.json").read_bytes()).hexdigest(),
                 "secondary_source_sha256": hashlib.sha256(DEFAULT_RIR.read_bytes()).hexdigest()},
            "secondary": {"taps": len(secondary), "all_taps_gain_sign_delay_preserved": True,
                          "jetson_output_path_measured": False, "original_secondary_additional_delay_samples": 0},
            "protocol": {"polarity": "e=d+S*delay(u)", "output_limit": "hard clip BEFORE delay/secondary plant",
                "selector_scheduling": "completed past window, simulated ready time; actual asynchronous worker not implemented",
                "startup": "SFANC zero until first result; fixed FIR immediate; FxNLMS cold. Whole stream and common post-startup both retained",
                "primary_delay_48": "3 ms synthetic sensitivity point, NOT measured OMAP preview or Jetson geometry transfer",
                "bank_frozen_under_path_mismatch": True, "oracle_future_selection": False,
                "scenario_fitted_fir": "separate white-noise fit with known changed P/delay; not same information as frozen SFANC and not an optimum bound",
                "cold_block_fxnlms": "receives exact S and extra delay for filtered-x plus current ERR; fixed mu and finite cold-start duration, not optimized firmware reproduction",
                "delay": "extra causal integer samples beyond original S; no implicit block/handoff latency, no device latency claim",
                "jitter_evaluated": False, "filter_switch_policy": "diagnostic configuration, not deployed crossover/safety design",
                "band_metric": "rectangular FFT window, finite-window leakage; weak/zero powers give null dB",
                "speech": "existing test crops reused only for fixed diagnostic; no new independent test claim"},
            "fitted_reference": {"source_sha256": _sha(design_x), "seed_namespace": [config.seed, 300],
                                 "records": fitted_records},
            "sources": source_records, "runs": runs, "metrics": rows,
            "implementation_sha256": {str(path.relative_to(Path.cwd())): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in (Path(__file__).resolve(),
                                         Path(__file__).resolve().parents[1]/"baselines"/"sfanc_stream.py",
                                         Path(__file__).with_name("sfanc_fxnlms.py"))},
        }
        with (destination/"report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        return report
    finally:
        torch.set_num_threads(previous_threads)
