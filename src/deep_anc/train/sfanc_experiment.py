"""장치 연결 없는 SFANC 사전학습 실험. 합성 P는 실측 P가 아니다.

완료된 REF 창으로 다음 창의 고정 FIR을 선택한다. 라벨 계산에는 미래 평가
구간의 d를 쓸 수 있지만 추론 입력에는 넣지 않는다. 연속 교체/실기 배포는 별도다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import signal
import torch

from deep_anc.baselines.filter_bank import new_output_directory, ref_feature
from deep_anc.baselines.sfanc_design import fit_control_fir, score_control_filters
from deep_anc.train.sfanc_selector import RefSpectrumSelector, reference_features, train_selector


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()


@dataclass(frozen=True)
class ExperimentConfig:
    sample_rate: int = 16000
    control_length: int = 64
    bank_train_samples: int = 16384
    window_samples: int = 4096
    n_fft: int = 512
    secondary_kind: str = "synthetic"
    secondary_delay_samples: int = 8
    primary_delay_samples: int = 32
    primary_gain: float = 0.15
    regularization: float = 1e-7
    effort_penalty: float = 0.002
    target_band_weight: float = 3.0
    control_limit: float = 0.2
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 0.001
    temperature: float = 0.05
    risk_weight: float = 1.0
    seed: int = 20260920
    train_groups: int = 32
    validation_groups: int = 8
    test_groups: int = 8

    def __post_init__(self):
        for name in ("sample_rate", "control_length", "bank_train_samples", "window_samples",
                     "n_fft", "epochs", "batch_size", "train_groups", "validation_groups", "test_groups"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name}: 양의 정수 필요")
        for name in ("seed", "primary_delay_samples", "secondary_delay_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name}: 비음수 정수 필요")
        for name in ("regularization", "control_limit", "learning_rate", "temperature", "primary_gain"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name}: 유한 양수 필요")
        for name in ("effort_penalty", "target_band_weight", "risk_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name}는 유한 비음수")
        if self.secondary_kind not in ("synthetic", "omap_measured"):
            raise ValueError("secondary_kind는 synthetic 또는 omap_measured")
        if self.sample_rate != 16000:
            raise ValueError("현재 사전학습 소스/대역 규약은 16 kHz 전용; 암묵적 리샘플링 금지")
        if (self.window_samples % 256 or self.bank_train_samples % 256
                or self.n_fft > self.window_samples or self.n_fft < 32
                or self.control_length > 512 or self.window_samples > 32768
                or self.bank_train_samples > 131072 or self.epochs > 10000
                or max(self.train_groups, self.validation_groups, self.test_groups) > 1000):
            raise ValueError("세그먼트/FFT/실험 규모 제한 위반")
        secondary_taps = 500 if self.secondary_kind == "omap_measured" else 24
        if max(self.primary_delay_samples + 74, self.secondary_delay_samples + secondary_taps + self.control_length) >= self.window_samples:
            raise ValueError("경로 지연/settling이 관측 창보다 길거나 메모리 한도를 넘습니다")


def paths(config):
    """P는 언제나 명시적인 합성 예제. S 원본을 자르거나 정렬하지 않는다."""
    if config.secondary_kind == "omap_measured":
        from deepanc.calibration import DEFAULT_RIR, load_secondary_path
        secondary = load_secondary_path(sample_rate=config.sample_rate)
        provenance = {"kind": "omap_measured", "source": "rir.txt",
                      "source_sha256": digest(DEFAULT_RIR.read_bytes()),
                      "preserve_all_taps_gain_sign_delay": True,
                      "jetson_output_path_measured": False}
    else:
        secondary = np.zeros(24, dtype=np.float64)
        secondary[[0, 9, 23]] = [0.7, -0.5, 0.2]
        provenance = {"kind": "synthetic", "source": "sfanc_synthetic_secondary.v1"}
    primary = np.zeros(config.primary_delay_samples + 74, dtype=np.float64)
    primary[[config.primary_delay_samples, config.primary_delay_samples + 73]] = [config.primary_gain, -0.4 * config.primary_gain]
    provenance.update(coefficient_sha256=digest(np.asarray(secondary, dtype="<f8").tobytes()),
                      secondary_taps=len(secondary),
                      additional_delay_samples=config.secondary_delay_samples,
                      additional_delay_is_scenario_not_measurement=True,
                      primary_kind="synthetic_two_impulses",
                      primary_sha256=digest(primary.astype("<f8").tobytes()))
    return primary, secondary, provenance


def bands(config):
    return ((80, 400), (400, 800), (800, 1000), (1000, 1200),
            (1200, 1600), (1600, 3000), (3000, 7500), (80, 7500))


def make_band_reference(low, high, count, rng, sample_rate, rms=0.04):
    # 합성 음원 생성에만 적용한다. 실측·실제 음원은 정규화하지 않는다.
    sos = signal.butter(4, (low, high), btype="bandpass", fs=sample_rate, output="sos")
    value = signal.sosfilt(sos, rng.normal(size=count + 1024))[1024:]
    value *= rms / max(float(np.sqrt(np.mean(value ** 2))), 1e-12)
    return value.astype(np.float32)


def design_bank(config, primary, secondary):
    warmup = max(len(primary) - 1, len(secondary) - 1 + config.secondary_delay_samples
                 + config.control_length - 1)
    if warmup + config.control_length >= config.bank_train_samples:
        raise ValueError("필터 계산용 구간이 경로/제어 FIR보다 짧습니다")
    coefficients = [np.zeros(config.control_length)]
    prototypes = [np.zeros(6)]
    records = [{"id": "zero", "purpose": "명시적 무제어 대조군/무신호 fallback"}]
    for index, (low, high) in enumerate(bands(config)):
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0, index]))
        reference = make_band_reference(low, high, config.bank_train_samples, rng, config.sample_rate)
        disturbance = signal.lfilter(primary, [1.0], reference)
        fit = fit_control_fir(reference, disturbance, secondary,
                              control_length=config.control_length,
                              secondary_delay_samples=config.secondary_delay_samples,
                              warmup_samples=warmup, regularization=config.regularization,
                              effort_penalty=config.effort_penalty)
        scored = score_control_filters(reference, disturbance, secondary, fit.coefficients,
                                       secondary_delay_samples=config.secondary_delay_samples,
                                       warmup_samples=warmup, effort_penalty=config.effort_penalty)
        peak = float(np.max(np.abs(scored["control"])))
        if peak > config.control_limit:
            raise ValueError(f"필터 {index} 학습 출력 한도 초과: {peak}; 계수 임의 축소/재시도 없음")
        coefficients.append(fit.coefficients)
        prototypes.append(ref_feature(reference[:config.window_samples], config.sample_rate))
        records.append({"id": f"band_{low}_{high}", "source_seed_namespace": [config.seed, 0, index],
                        "reference_sha256": digest(reference.tobytes()), "band_hz": [low, high],
                        "fit": fit.diagnostics, "control_peak": peak})
        print(f"[FIR] {low}–{high} Hz 계산 완료", flush=True)
    return np.asarray(coefficients), np.asarray(prototypes), records


def procedural_sources(config, split):
    split_code = {"train": 1, "validation": 2, "test": 3}[split]
    count = getattr(config, {"train": "train_groups", "validation": "validation_groups", "test": "test_groups"}[split])
    length = 2 * config.window_samples
    for group in range(count):
        for family in range(10):
            rng = np.random.default_rng(np.random.SeedSequence([config.seed, split_code, group, family]))
            level = rng.uniform(0.015, 0.055)
            if family < 7:
                low, high = bands(config)[family]
                width = rng.uniform(0.2, 0.95) * (high - low)
                begin = rng.uniform(low, high - width)
                value = make_band_reference(begin, begin + width, length, rng, config.sample_rate, level)
            elif family == 7:
                value = make_band_reference(80, 7500, length, rng, config.sample_rate, level)
            elif family == 8:
                # 바뀐 대역은 과거 REF에서 예견할 수 없다. 실패도 평가에 남긴다.
                value = np.concatenate([make_band_reference(*bands(config)[i], config.window_samples,
                                        rng, config.sample_rate, level) for i in rng.choice(7, 2, replace=False)])
            else:
                value = np.zeros(length, dtype=np.float32)
            yield value, {"source_id": f"procedural:{split}:{group}:{family}",
                          "group_id": f"procedural:{split}:{group}", "split": split,
                          "kind": ("synthetic_switch" if family == 8 else
                                   "silence" if family == 9 else "synthetic_band_noise"),
                          "source_sha256": digest(value.tobytes())}


def _band_power(values, sample_rate):
    values = np.asarray(values, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(values, axis=-1)) ** 2 / values.shape[-1] ** 2
    weight = np.full(spectrum.shape[-1], 2.0)
    weight[0] = 1
    if values.shape[-1] % 2 == 0:
        weight[-1] = 1
    spectrum *= weight
    frequency = np.fft.rfftfreq(values.shape[-1], 1 / sample_rate)
    return {"fullband": np.mean(values ** 2, axis=-1),
            "below_1000": spectrum[..., frequency < 1000].sum(axis=-1),
            "target_1000_1600": spectrum[..., (frequency >= 1000) & (frequency < 1600)].sum(axis=-1),
            "above_1600": spectrum[..., frequency >= 1600].sum(axis=-1)}


def prepare_split(config, split, coefficients, primary, secondary, speech_manifest=None):
    sources = procedural_sources(config, split)
    if speech_manifest is not None:
        from deep_anc.data.sfanc_sources import iter_librispeech_crops
        sources = itertools.chain(sources, iter_librispeech_crops(speech_manifest, split))
    warmup = max(len(primary) - 1, len(secondary) - 1 + config.secondary_delay_samples
                 + config.control_length - 1)
    start = config.window_samples + warmup
    if start >= 2 * config.window_samples - 256:
        raise ValueError("관측 이후 독립 평가 창이 경로 settling보다 짧습니다")
    features, costs, rows, powers, simple_features = [], [], [], [], []
    for reference, meta in sources:
        reference = np.asarray(reference, dtype=np.float64)
        if reference.shape != (2 * config.window_samples,) or not np.isfinite(reference).all():
            raise ValueError("REF 원천 crop shape/유한값 불일치")
        disturbance = signal.lfilter(primary, [1.0], reference)
        scored = score_control_filters(reference, disturbance, secondary, coefficients,
                                       secondary_delay_samples=config.secondary_delay_samples,
                                       warmup_samples=start, effort_penalty=config.effort_penalty)
        baseline = float(scored["baseline_mse"])
        base_powers = _band_power(disturbance[start:], config.sample_rate)
        error_powers = _band_power(scored["residual"][:, start:], config.sample_rate)
        weighted_baseline = baseline + config.target_band_weight * base_powers["target_1000_1600"]
        # 고역을 추가 가중하되 전대역·출력 에너지 항을 유지한다.
        risk = (np.asarray(scored["objective"], dtype=np.float64)
                + config.target_band_weight * error_powers["target_1000_1600"]) / max(weighted_baseline, 1e-12)
        peak = np.max(np.abs(scored["control"][:, config.window_samples:]), axis=1)
        # 학습 라벨의 포화 후보는 벌점. 추론은 미래 peak를 보고 대체하지 않는다.
        risk = risk + 10 * (peak > config.control_limit)
        if not np.isfinite(risk).all():
            raise ValueError("비유한 후보 잔차")
        features.append(reference_features(reference[:config.window_samples], n_fft=config.n_fft)[0])
        simple = ref_feature(reference[:config.window_samples], config.sample_rate)
        simple_features.append(np.zeros(6) if simple is None else simple)
        costs.append(risk.astype(np.float32))
        rows.append({**meta, "crop_sha256": digest(reference.astype("<f8").tobytes()),
                     "observed_until_sample": config.window_samples,
                     "apply_at_sample": config.window_samples, "scored_from_sample": start,
                     "baseline_mse": baseline, "past_reference_mse": float(np.mean(reference[:config.window_samples] ** 2)),
                     "control_peak_by_candidate": peak.tolist()})
        powers.append({"baseline": {k: float(v) for k, v in base_powers.items()},
                       "residual": {k: v.tolist() for k, v in error_powers.items()}})
    print(f"[자료] {split}: {len(rows)}개 REF 창/후속 잔차 라벨", flush=True)
    return {"features": np.asarray(features), "costs": np.asarray(costs), "rows": rows,
            "powers": powers, "simple_features": np.asarray(simple_features)}


def selection_metrics(data, indices, control_limit):
    rows = []
    for index, selected in enumerate(indices):
        sample = data["rows"][index]
        for band, baseline in data["powers"][index]["baseline"].items():
            error = data["powers"][index]["residual"][band][selected]
            reduction = 10 * np.log10(baseline / error) if min(baseline, error) > 1e-14 else None
            rows.append({"source_id": sample["source_id"], "kind": sample.get("kind", "librispeech"),
                         "band": band, "baseline_power": baseline, "residual_power": error,
                         "reduction_db": float(reduction) if reduction is not None else None,
                         "emergent_error_energy": bool(baseline <= 1e-14 < error)})
    risk = data["costs"][np.arange(len(indices)), indices]
    return {"mean_cost": float(risk.mean()),
            "mean_regret": float(np.mean(risk - data["costs"].min(axis=1))),
            "selection_counts": np.bincount(indices, minlength=data["costs"].shape[1]).tolist(),
            "control_limit_exceeded_windows": sum(data["rows"][i]["control_peak_by_candidate"][c] > control_limit
                                                  for i, c in enumerate(indices)),
            "metrics": rows}


def save_selector(model, directory, bank_sha, config):
    artifact = {"schema": "sfanc_ref_spectrum_selector.v1", "bank_sha256": bank_sha,
                "candidate_count": model.candidate_count, "bins": config.n_fft // 2 + 1,
                "n_fft": config.n_fft, "window_samples": config.window_samples,
                "sample_rate": config.sample_rate, "deployment_allowed": False,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    path = directory / "selector.pt"
    with path.open("xb") as handle:
        torch.save(artifact, handle)
    return digest(path.read_bytes())


def load_selector(directory):
    """저장한 tensor-only artifact의 무결성 검사. SHA는 출처 인증이 아니다."""
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    for name in ("bank.npz", "selector.pt"):
        if digest((directory / name).read_bytes()) != report["artifacts"][name]:
            raise ValueError(f"{name} SHA 불일치")
    artifact = torch.load(directory / "selector.pt", map_location="cpu", weights_only=True)
    if (artifact["schema"] != "sfanc_ref_spectrum_selector.v1" or artifact["deployment_allowed"] is not False
            or artifact["bank_sha256"] != report["artifacts"]["bank.npz"]):
        raise ValueError("selector schema/bank 결합 불일치")
    model = RefSpectrumSelector(artifact["candidate_count"], artifact["bins"])
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("모델 비유한 계수")
    with np.load(directory / "bank.npz", allow_pickle=False) as bank:
        coefficients = bank["coefficients"].copy()
    if coefficients.ndim != 2 or coefficients.shape[0] != artifact["candidate_count"] or not np.isfinite(coefficients).all():
        raise ValueError("bank 후보 수/유한값 불일치")
    return model, coefficients, artifact


def run_experiment(config, out, *, device="cpu", librispeech_root=None):
    destination = new_output_directory(out)
    destination.mkdir(parents=True, exist_ok=False)  # 같은 output의 다른 writer를 학습 전에 거부.
    torch.set_num_threads(2)
    primary, secondary, provenance = paths(config)
    coefficients, prototypes, bank_records = design_bank(config, primary, secondary)
    bank_path = destination / "bank.npz"
    with bank_path.open("xb") as handle:
        np.savez_compressed(handle, coefficients=coefficients, secondary=secondary,
                            primary=primary, prototypes=prototypes)
    bank_sha = digest(bank_path.read_bytes())
    speech_manifest = None
    if librispeech_root is not None:
        from deep_anc.data.sfanc_sources import prepare_librispeech_manifest
        speech_manifest = prepare_librispeech_manifest(librispeech_root, seed=config.seed,
                            crop_samples=2 * config.window_samples,
                            max_files_per_split={"train": 60, "validation": 20, "test": 20})
        with (destination / "speech_manifest.json").open("x") as handle:
            json.dump(speech_manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
    train = prepare_split(config, "train", coefficients, primary, secondary, speech_manifest)
    validation = prepare_split(config, "validation", coefficients, primary, secondary, speech_manifest)
    print(f"[학습] device={device}, epochs={config.epochs}", flush=True)
    result = train_selector(train["features"], train["costs"], validation["features"], validation["costs"],
                            epochs=config.epochs, batch_size=config.batch_size, learning_rate=config.learning_rate,
                            seed=config.seed, device=device, temperature=config.temperature,
                            risk_weight=config.risk_weight)
    model = result.model.cpu().eval()
    selector_sha = save_selector(model, destination, bank_sha, config)
    # test는 모델/단일 고정 대조군 결정 후 처음 생성하며 학습 함수에 전달하지 않는다.
    fixed = int(np.argmin(train["costs"].mean(axis=0)))
    test = prepare_split(config, "test", coefficients, primary, secondary, speech_manifest)
    evaluation = {}
    for split, data in (("validation", validation), ("test", test)):
        with torch.inference_mode():
            logits = model(torch.from_numpy(data["features"]))
            if not torch.isfinite(logits).all():
                raise ValueError("최종 평가 selector의 비유한 출력")
            predicted = logits.argmax(dim=1).numpy()
        nearest = 1 + ((data["simple_features"][:, None, :] - prototypes[None, 1:, :]) ** 2).sum(axis=2).argmin(axis=1)
        silent_past = np.asarray([row["past_reference_mse"] <= 1e-12 for row in data["rows"]])
        predicted[silent_past] = 0
        nearest[silent_past] = 0
        evaluation[split] = {name: selection_metrics(data, indices, config.control_limit) for name, indices in (
            ("zero", np.zeros(len(predicted), dtype=int)),
            ("train_best_fixed", np.full(len(predicted), fixed, dtype=int)),
            ("nearest_psd", nearest), ("learned_selector", predicted),
            ("future_cost_oracle_not_deployable", data["costs"].argmin(axis=1)))}
    report = {"schema": "sfanc_offline_pretraining.v1", "config": asdict(config),
              "physical_performance_claim_allowed": False, "deployment_allowed": False,
              "real_time_claim_allowed": False, "live_connection_implemented": False,
              "primary_is_synthetic": True, "speech_sources_used": speech_manifest is not None,
              "music_evaluated": False, "path": provenance, "bank": bank_records,
              "artifacts": {"bank.npz": bank_sha, "selector.pt": selector_sha},
              "implementation_sha256": {str(path.relative_to(Path(__file__).resolve().parents[2])): digest(path.read_bytes())
                  for path in (Path(__file__).resolve(), Path(__file__).with_name("sfanc_selector.py").resolve(),
                               Path(__file__).resolve().parents[1] / "baselines" / "sfanc_design.py",
                               Path(__file__).resolve().parents[1] / "data" / "sfanc_sources.py")},
              "training": {"device": str(device), "torch_version": str(torch.__version__),
                           "cuda_device": torch.cuda.get_device_name() if str(device).startswith("cuda") else None,
                           "history": result.history, "best_epoch": result.best_epoch,
                           "initial_validation_loss": result.initial_validation_loss,
                           "best_validation_loss": result.best_validation_loss,
                           "fixed_candidate_from_train": fixed,
                           "train_oracle_class_counts": np.bincount(train["costs"].argmin(axis=1), minlength=len(coefficients)).tolist()},
              "datasets": {s: d["rows"] for s, d in (("train", train), ("validation", validation), ("test", test))},
              "evaluation": evaluation,
              "protocol": {"selector_input": "completed past REF window only, logPSD + logRMS",
                           "label": "next-window residual + effort cost, no source-family labels",
                           "training_objective": "soft cross entropy + risk_weight * expected candidate regret",
                           "cost": "(fullband residual + target_band_weight*1000–1600Hz residual + effort_penalty*control_mse)/(same weighted disturbance, floor1e-12) + 10 if output peak exceeds limit",
                           "coefficients": "FP64 causal regularized least squares, not the paper FxLMS reproduction",
                           "first_window": "zero output; next window cost excludes complete path settling",
                           "transition_evaluated": False, "continuous_stream_evaluated": False,
                           "test_used_for_checkpoint": False,
                           "target_hz": [1000, 1600], "target_high_endpoint_included": False,
                           "target_weighting": "fullband + target_band_weight additional 1000–1600 Hz; bank fit on each source band uses fullband ridge cost",
                           "silence_fallback": "past REF mean-square <=1e-12 selects zero; future is not inspected",
                           "hidden_path_inference_claim": False}}
    with (destination / "report.json").open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    restored, restored_filters, _ = load_selector(destination)
    with torch.inference_mode():
        inputs = torch.from_numpy(test["features"][:4])
        if not torch.equal(model(inputs), restored(inputs)) or not np.array_equal(coefficients, restored_filters):
            raise ValueError("checkpoint 복원 등가성 실패")
    print(f"[완료] {destination}, best_epoch={result.best_epoch}; 오프라인 사전학습, 실측 감쇠 아님", flush=True)
    return report
