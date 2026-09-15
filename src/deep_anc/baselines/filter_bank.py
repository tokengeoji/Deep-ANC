"""합성 전용 사전 FIR bank와 과거 REF만 쓰는 선택 기준선.

장치/신경망/실시간 스레드에 연결하지 않는다. train/validation/test의 source seed를
분리하며 실측 경로·음성·음악으로 승격하지 않는다. SHA는 재현/손상 검사이지 출처 인증이 아니다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from scipy import signal

from .fxlms_core import FxLMSController, SampleDelay
from .prepared_fir import FilterProposal, PreparedFxNLMSController


SCENARIOS = ("causal_toy", "long_delay_stress")
TRAIN_FAMILIES = ("low", "mid", "high", "mixed", "white")
EVAL_FAMILIES = (*TRAIN_FAMILIES, "switch", "silence")
VARIANTS = ("zero_control", "cold_fxnlms", "fixed_fir", "selected_fxnlms")
POWER_FLOOR = 1e-14
# 모든 합성 REF는 float32다. white는 최대 4*level, mixed는 최대 3*level이다.
MAX_EVALUATION_LEVEL = float(np.finfo(np.float32).max) / 4.0
MIN_SATURATION_LEVEL = float(np.finfo(np.float32).tiny)
MAX_SATURATION_LEVEL = float(np.finfo(np.float32).max)


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class BankConfig:
    sample_rate: int = 48000
    hop: int = 256
    secondary_delay_samples: int = 1465
    stress_preview_samples: int = 140
    control_length: int = 16
    train_blocks: int = 400
    eval_blocks: int = 128
    window_blocks: int = 8
    selection_interval_blocks: int = 16
    mu: float = 0.05
    train_seeds: tuple[int, ...] = (11, 23)
    validation_seeds: tuple[int, ...] = (101, 103)
    test_seeds: tuple[int, ...] = (211, 223)
    evaluation_levels: tuple[float, ...] = (0.025, 0.06)
    saturation_levels: tuple[float, ...] = (0.0, 0.03)
    secondary_fir: tuple[float, ...] = (-0.5, -0.08)

    def __post_init__(self):
        for name in ("sample_rate", "hop", "control_length", "train_blocks", "eval_blocks",
                     "window_blocks", "selection_interval_blocks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name}: 양의 정수가 필요합니다")
        for name in ("secondary_delay_samples", "stress_preview_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name}: 음수가 아닌 정수가 필요합니다")
        if self.sample_rate <= 3200 or self.stress_preview_samples >= self.control_delay:
            raise ValueError("sample_rate>3200, stress preview<S+handoff가 필요합니다")
        if self.eval_blocks * self.hop < 8 * (self.control_delay + self.hop):
            raise ValueError("평가 구간이 지연/변화 전후 창에 비해 짧습니다")
        if self.train_blocks <= self.window_blocks or self.train_blocks * self.hop <= self.control_delay:
            raise ValueError("학습 구간이 특징 창/지연보다 길어야 합니다")
        if self.eval_blocks <= self.window_blocks:
            raise ValueError("평가 구간이 특징 창보다 길어야 합니다")
        if self.window_blocks * self.hop < 2:
            raise ValueError("REF 특징 창은 최소 두 샘플이어야 합니다")
        if max(self.eval_blocks, self.train_blocks) * self.hop > 10_000_000 or self.control_length > 4096:
            raise ValueError("이 CPU 합성 도구의 입력 규모 제한을 넘었습니다")
        all_seeds = []
        for name in ("train_seeds", "validation_seeds", "test_seeds"):
            values = tuple(getattr(self, name))
            if not values or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
                raise ValueError(f"{name}: 비어 있지 않은 비음수 정수 seed가 필요합니다")
            object.__setattr__(self, name, values)
            all_seeds.extend(values)
        if len(set(all_seeds)) != len(all_seeds):
            raise ValueError("train/validation/test seed 그룹이 겹칩니다")
        for name, positive in (("evaluation_levels", True), ("saturation_levels", False)):
            values = tuple(getattr(self, name))
            if (not values or any(isinstance(v, bool) or not isinstance(v, (float, int))
                                  or not np.isfinite(v) or v < 0 or (positive and v == 0) for v in values)):
                raise ValueError(f"{name}: 유한한 {'양수' if positive else '비음수'}가 필요합니다")
            if len(set(values)) != len(values):
                raise ValueError(f"{name}: 중복 조건은 허용하지 않습니다")
            if name == "evaluation_levels" and any(v > MAX_EVALUATION_LEVEL for v in values):
                raise ValueError("evaluation_levels: 최대 합성 REF peak가 float32 범위를 벗어납니다")
            if name == "saturation_levels" and any(
                v != 0 and not MIN_SATURATION_LEVEL <= v <= MAX_SATURATION_LEVEL for v in values
            ):
                raise ValueError("saturation_levels: 0 또는 float32 normal 양수 범위가 필요합니다")
            object.__setattr__(self, name, tuple(float(v) for v in values))
        secondary = np.asarray(self.secondary_fir)
        if secondary.ndim != 1 or secondary.size == 0 or secondary.dtype.kind not in "fiu" or not np.isfinite(secondary).all():
            raise ValueError("secondary_fir는 유한한 실수 FIR이어야 합니다")
        object.__setattr__(self, "secondary_fir", tuple(float(v) for v in secondary))
        self.controller()  # 설정·S의 수치/물리 규약을 기존 소비자와 같은 기준으로 검사한다.
        object.__setattr__(self, "mu", float(self.mu))

    @property
    def control_delay(self):
        return self.secondary_delay_samples + self.hop

    @property
    def controller_configuration(self):
        return dict(sample_rate=self.sample_rate, hop=self.hop,
                    secondary_delay_samples=self.secondary_delay_samples, handoff_extra_samples=self.hop,
                    control_length=self.control_length, reference_peak_limit=0.25, control_limit=0.1,
                    prepared_l1_limit=2.0, residual_norm_limit=2.0, mu=self.mu, leakage=0.0,
                    transition_samples=self.hop, max_proposal_age_samples=self.hop)

    def controller(self):
        return PreparedFxNLMSController(np.asarray(self.secondary_fir, dtype=np.float32),
                                        **self.controller_configuration)

    def preview(self, scenario):
        if scenario not in SCENARIOS:
            raise ValueError(f"알 수 없는 합성 조건: {scenario}")
        return self.control_delay if scenario == "causal_toy" else self.stress_preview_samples


def synthetic_reference(config, family, seed, split, *, level=0.04):
    """PCG64/SeedSequence(seed,split,family) 합성 source. 원본 파일을 받지 않는다."""
    splits = {"train": 0, "validation": 1, "test": 2}
    if split not in splits or family not in EVAL_FAMILIES:
        raise ValueError("알 수 없는 source 그룹/종류")
    rng = np.random.default_rng(np.random.SeedSequence([seed, splits[split], EVAL_FAMILIES.index(family)]))
    count = (config.train_blocks if split == "train" else config.eval_blocks) * config.hop
    frequencies = {"train": (300.0, 900.0, 1300.0), "validation": (350.0, 1000.0, 1400.0),
                   "test": (250.0, 850.0, 1550.0)}[split]
    if family == "silence":
        return np.zeros(count, dtype=np.float32)
    if family == "white":
        return np.clip(rng.normal(0.0, level, count), -4 * level, 4 * level).astype(np.float32)
    time = np.arange(count, dtype=np.float64) / config.sample_rate
    components = [level * np.sin(2 * np.pi * frequency * time + rng.uniform(-np.pi, np.pi))
                  for frequency in frequencies]
    if family == "mixed":
        value = sum(components)
    elif family == "switch":
        value = np.empty(count, dtype=np.float64)
        for part, component in enumerate(components):
            begin, end = part * count // 3, (part + 1) * count // 3
            value[begin:end] = component[begin:end]
    else:
        value = components[("low", "mid", "high").index(family)]
    return np.asarray(value, dtype=np.float32)


def ref_feature(reference, sample_rate):
    """과거 REF Hann-PSD 비율의 제곱근 5개 + log RMS 1개. 무신호는 None."""
    values = np.asarray(reference, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("특징 입력은 유한한 REF 벡터여야 합니다")
    power = float(np.mean(values * values))
    if power <= POWER_FLOOR:
        return None
    spectrum = np.abs(np.fft.rfft(values * np.hanning(values.size))) ** 2
    frequencies = np.fft.rfftfreq(values.size, 1 / sample_rate)
    edges = (0.0, 400.0, 800.0, 1100.0, 1600.0, sample_rate / 2 + 1)
    bands = np.array([spectrum[(frequencies >= low) & (frequencies < high)].sum()
                      for low, high in zip(edges[:-1], edges[1:])])
    if bands.sum() <= POWER_FLOOR:
        return None
    return np.asarray([*np.sqrt(bands / bands.sum()), 0.05 * np.log10(power)], dtype=np.float32)


def _prototype(reference, config):
    window = config.window_blocks * config.hop
    features = [ref_feature(reference[i:i + window], config.sample_rate)
                for i in range(0, reference.size - window + 1, window)]
    usable = [feature for feature in features if feature is not None]
    if not usable:
        raise ValueError("학습 REF 특징에 유효 신호가 없습니다")
    return tuple(float(v) for v in np.mean(usable, axis=0).astype(np.float32))


@dataclass(frozen=True)
class BankCandidate:
    candidate_id: str
    source_family: str
    train_seed: int
    reference_sha256: str
    coefficients: tuple[float, ...]
    feature: tuple[float, ...]

    def __post_init__(self):
        if (not isinstance(self.candidate_id, str) or not self.candidate_id
                or self.source_family not in TRAIN_FAMILIES
                or isinstance(self.train_seed, bool) or not isinstance(self.train_seed, int) or self.train_seed < 0
                or not isinstance(self.reference_sha256, str) or len(self.reference_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.reference_sha256)):
            raise ValueError("후보의 source ID/seed/SHA 메타가 잘못되었습니다")
        for name in ("coefficients", "feature"):
            values = np.asarray(getattr(self, name))
            if (values.ndim != 1 or not values.size or values.dtype.kind not in "fiu"
                    or not np.isfinite(values).all() or np.max(np.abs(values.astype(np.float64))) > 1e6):
                raise ValueError("후보 계수/특징은 유한한 실수 벡터여야 합니다")
            object.__setattr__(self, name, tuple(float(v) for v in values.astype(np.float32)))
        if len(self.feature) != 6:
            raise ValueError("후보 REF 특징은 6차원이어야 합니다")


@dataclass(frozen=True)
class FilterBank:
    config: BankConfig
    scenario: str
    context_id: str
    bank_id: str
    candidates: tuple[BankCandidate, ...]

    def __post_init__(self):
        if not isinstance(self.config, BankConfig):
            raise ValueError("검증된 BankConfig가 필요합니다")
        self.config.preview(self.scenario)
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if (len(self.candidates) != len(TRAIN_FAMILIES) * len(self.config.train_seeds)
                or any(not isinstance(candidate, BankCandidate) for candidate in self.candidates)
                or any(len(candidate.coefficients) != self.config.control_length for candidate in self.candidates)
                or len({candidate.candidate_id for candidate in self.candidates}) != len(self.candidates)):
            raise ValueError("bank 후보 수/shape/ID가 잘못되었습니다")
        if self.context_id != self.config.controller().context_id:
            raise ValueError("bank context가 제어 설정과 다릅니다")
        if not isinstance(self.bank_id, str) or len(self.bank_id) != 64 or any(c not in "0123456789abcdef" for c in self.bank_id):
            raise ValueError("bank ID는 SHA-256이어야 합니다")


def _arrays(candidates):
    return (np.asarray([c.coefficients for c in candidates], dtype=np.float32),
            np.asarray([c.feature for c in candidates], dtype=np.float32))


def _feature_schema(config):
    return {"name": "hann_psd_rms.v1", "power_floor": POWER_FLOOR,
            "band_edges_hz": [0.0, 400.0, 800.0, 1100.0, 1600.0, config.sample_rate / 2 + 1],
            "spectral_features": "sqrt(band_power/sum_band_power), Hann window",
            "level_feature": "0.05*log10(mean(ref**2))",
            "prototype_pooling": "mean of nonoverlapping complete train windows"}


def _metadata(config, scenario, context_id, candidates):
    return {
        "schema": "synthetic_filter_bank.v1", "synthetic_only": True,
        "physical_claim_allowed": False, "performance_claim_allowed": False,
        "deployment_allowed": False, "reference_mode": "acoustic_simulation",
        "config": asdict(config), "controller_configuration": config.controller_configuration,
        "scenario": scenario, "primary_preview_samples": config.preview(scenario),
        "primary_gain": 0.15, "train_level": 0.04, "context_id": context_id,
        "rng": "PCG64/SeedSequence(seed, split_code, family_index)",
        "decoder": "synthetic_reference.v1/float32", "numpy_version": np.__version__,
        "source_split_codes": {"train": 0, "validation": 1, "test": 2},
        "source_family_order": list(EVAL_FAMILIES), "feature_schema": _feature_schema(config),
        "candidates": [{key: value for key, value in asdict(candidate).items()
                        if key not in ("coefficients", "feature")} for candidate in candidates],
    }


def _bank_id(metadata, coefficients, features):
    return _sha(_canonical(metadata) + coefficients.tobytes() + features.tobytes())


class BankTrainingError(ValueError):
    def __init__(self, diagnostics):
        super().__init__("학습 후보 검증 실패: 해당 조건 bank를 저장/승격하지 않습니다")
        self.diagnostics = diagnostics


class _Plant:
    def __init__(self, config, saturation=0.0):
        self.delay = SampleDelay(config.control_delay - config.hop)
        self.secondary = np.asarray(config.secondary_fir, dtype=np.float32)
        self.state = np.zeros(len(self.secondary) - 1, dtype=np.float32)
        self.saturation = saturation

    def step(self, previous_output):
        drive = previous_output
        if self.saturation:
            drive = self.saturation * np.tanh(drive / self.saturation)
        result, self.state = signal.lfilter(self.secondary, [1.0], self.delay.process(drive), zi=self.state)
        return np.asarray(result, dtype=np.float32)


def build_synthetic_bank(config: BankConfig, scenario: str) -> FilterBank:
    """train 그룹만 읽는다. clipping/실패 후보는 재시도·성공 seed 선별하지 않는다."""
    preview = config.preview(scenario)
    candidates, diagnostics = [], []
    context_id = config.controller().context_id
    for family in TRAIN_FAMILIES:
        for seed in config.train_seeds:
            candidate_id = f"{family}:seed={seed}"
            try:
                reference = synthetic_reference(config, family, seed, "train")
                disturbance = SampleDelay(preview).process(0.15 * reference)
                adaptive = FxLMSController(np.asarray(config.secondary_fir, dtype=np.float32),
                                          secondary_delay_samples=config.control_delay,
                                          control_len=config.control_length, mu=config.mu,
                                          leakage=0.0, weight_norm_limit=2.0)
                plant = _Plant(config)
                previous = np.zeros(config.hop, dtype=np.float32)
                for begin in range(0, reference.size, config.hop):
                    end = begin + config.hop
                    error = disturbance[begin:end] + plant.step(previous)
                    output = adaptive.generate_block(reference[begin:end])
                    if not np.isfinite(output).all() or np.any(np.abs(output) > 0.1):
                        raise ValueError("학습 출력 clipping/비유한 값: 후보 무효, 재시도 없음")
                    adaptive.adapt_block(error, enabled=True)
                    previous = np.clip(output, -0.1, 0.1)
                coefficients = tuple(float(v) for v in adaptive.w)
                proposal = FilterProposal(context_id, 0, 1, 0, coefficients, candidate_id)
                decision = config.controller().submit(proposal)
                if not decision.accepted:
                    raise ValueError(decision.reason)
                candidates.append(BankCandidate(candidate_id, family, seed, _sha(reference.tobytes()),
                                                coefficients, _prototype(reference, config)))
                diagnostics.append({"candidate_id": candidate_id, "status": "valid"})
            except (ValueError, FloatingPointError) as exc:
                diagnostics.append({"candidate_id": candidate_id, "status": "invalid", "reason": str(exc)})
    if any(row["status"] == "invalid" for row in diagnostics):
        raise BankTrainingError(diagnostics)
    coefficients, features = _arrays(candidates)
    metadata = _metadata(config, scenario, context_id, candidates)
    return FilterBank(config, scenario, context_id, _bank_id(metadata, coefficients, features), tuple(candidates))


def new_output_directory(path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path.cwd() / value
    for part in (value, *value.parents):
        if part.is_symlink():
            raise ValueError(f"심볼릭 링크 출력 경로 금지: {part}")
    if value.exists():
        raise FileExistsError(f"기존 결과 덮어쓰기 금지: {value}")
    return value


def save_bank(bank: FilterBank, out) -> None:
    """새 디렉터리에 논리 bank ID와 NPZ 파일 SHA가 연결된 JSON/NPZ를 기록한다."""
    destination = new_output_directory(out)
    coefficients, features = _arrays(bank.candidates)
    metadata = _metadata(bank.config, bank.scenario, bank.context_id, bank.candidates)
    if _bank_id(metadata, coefficients, features) != bank.bank_id:
        raise ValueError("메모리 bank의 논리 hash 불일치")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, coefficients=coefficients, features=features)
    raw = buffer.getvalue()
    metadata.update(bank_id=bank.bank_id, npz_sha256=_sha(raw))
    encoded = _canonical(metadata)
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "bank.npz").open("xb") as handle:
        handle.write(raw)
    with (destination / "bank.json").open("xb") as handle:
        handle.write(encoded)


def load_bank(path, *, expected_context_id: str) -> FilterBank:
    """pickle 없이 읽고 context·파일/논리/source hash·shape·유한값을 검증한다."""
    if not isinstance(expected_context_id, str) or len(expected_context_id) != 64:
        raise ValueError("소비자가 알고 있는 expected_context_id가 필요합니다")
    directory = Path(path)
    for item in (directory, directory / "bank.json", directory / "bank.npz"):
        if item.is_symlink():
            raise ValueError("bank 심볼릭 링크는 허용하지 않습니다")
    try:
        metadata = json.loads((directory / "bank.json").read_text(encoding="utf-8"))
        if (metadata["schema"] != "synthetic_filter_bank.v1" or metadata["synthetic_only"] is not True
                or metadata["deployment_allowed"] is not False or metadata["physical_claim_allowed"] is not False
                or metadata["performance_claim_allowed"] is not False):
            raise ValueError("알 수 없거나 합성 제한이 없는 bank schema")
        config = BankConfig(**metadata["config"])
        config.preview(metadata["scenario"])
        context = config.controller().context_id
        if metadata["context_id"] != expected_context_id or context != expected_context_id:
            raise ValueError("bank context 불일치")
        raw = (directory / "bank.npz").read_bytes()
        if _sha(raw) != metadata["npz_sha256"]:
            raise ValueError("NPZ SHA 불일치")
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            if set(arrays.files) != {"coefficients", "features"}:
                raise ValueError("알 수 없는 NPZ 배열")
            coefficients, features = arrays["coefficients"], arrays["features"]
        expected_count = len(TRAIN_FAMILIES) * len(config.train_seeds)
        if (coefficients.shape != (expected_count, config.control_length) or features.shape != (expected_count, 6)
                or coefficients.dtype != np.float32 or features.dtype != np.float32
                or not np.isfinite(coefficients).all() or not np.isfinite(features).all()
                or len(metadata["candidates"]) != expected_count):
            raise ValueError("bank 배열 shape/dtype/유한값 검증 실패")
        if np.any(np.abs(coefficients).sum(axis=1) > config.controller_configuration["prepared_l1_limit"]):
            raise ValueError("bank 계수 L1 제한 초과")
        logical = {key: value for key, value in metadata.items() if key not in ("bank_id", "npz_sha256")}
        if _bank_id(logical, coefficients, features) != metadata["bank_id"]:
            raise ValueError("논리 bank SHA 불일치")
        candidates = []
        expected_sources = [(family, seed) for family in TRAIN_FAMILIES for seed in config.train_seeds]
        for index, (family, seed) in enumerate(expected_sources):
            source = metadata["candidates"][index]
            reference = synthetic_reference(config, family, seed, "train")
            expected = {"candidate_id": f"{family}:seed={seed}", "source_family": family,
                        "train_seed": seed, "reference_sha256": _sha(reference.tobytes())}
            if source != expected or not np.allclose(features[index], _prototype(reference, config), atol=1e-6, rtol=0):
                raise ValueError("합성 decoder source SHA/특징 출처 불일치")
            candidates.append(BankCandidate(**source, coefficients=tuple(float(v) for v in coefficients[index]),
                                            feature=tuple(float(v) for v in features[index])))
        rebuilt = _metadata(config, metadata["scenario"], context, candidates)
        # 라이브러리 버전은 기록값을 유지하되 다른 decoder/전처리 메타의 위조는 거부한다.
        rebuilt["numpy_version"] = metadata["numpy_version"]
        if rebuilt != logical:
            # JSON은 tuple을 list로 바꾸므로 canonical 형식으로 비교한다.
            if _canonical(rebuilt) != _canonical(logical):
                raise ValueError("bank provenance/decoder 규약 불일치")
        return FilterBank(config, metadata["scenario"], context, metadata["bank_id"], tuple(candidates))
    except (KeyError, TypeError, BadZipFile, EOFError) as exc:
        raise ValueError(f"손상되거나 불완전한 bank: {exc}") from exc


@dataclass(frozen=True)
class SelectionDecision:
    proposal: FilterProposal | None
    candidate_id: str | None
    reason: str
    observed_until_sample: int


class PastRefSelector:
    """REF observe→완결 경계 choose→acknowledge. ERR/정답/전체 파일 인자가 없다."""

    def __init__(self, bank: FilterBank):
        self.bank = bank
        self.window = bank.config.window_blocks * bank.config.hop
        self.interval = bank.config.selection_interval_blocks * bank.config.hop
        self._features = np.asarray([candidate.feature for candidate in bank.candidates])
        self.reset(generation=0)

    def reset(self, *, generation):
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("generation은 비음수 정수여야 합니다")
        self.generation = generation
        self.sample_cursor = 0
        self._history = np.empty(0, dtype=np.float32)
        self._next = self.window
        self._revision = 0
        self._accepted_id = None
        self._pending_decision = None

    def observe(self, reference_block, *, end_sample_exclusive):
        values = np.asarray(reference_block)
        if (values.shape != (self.bank.config.hop,) or values.dtype.kind not in "fiu"
                or not np.isfinite(values).all() or np.max(np.abs(values)) > 0.25):
            raise ValueError("선택기 REF는 hop 길이의 유한 범위 내 벡터여야 합니다")
        if (isinstance(end_sample_exclusive, bool) or not isinstance(end_sample_exclusive, int)
                or end_sample_exclusive != self.sample_cursor + self.bank.config.hop):
            raise ValueError("관측 끝은 연속한 완료 블록의 exclusive 끝이어야 합니다")
        self._history = np.concatenate((self._history, values.astype(np.float32)))[-self.window:].copy()
        self.sample_cursor = end_sample_exclusive

    def choose(self, *, generation, consumed_until_sample) -> SelectionDecision:
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (generation, consumed_until_sample)):
            raise ValueError("소비자 generation/cursor는 정수여야 합니다")
        if generation != self.generation or consumed_until_sample != self.sample_cursor:
            raise ValueError("소비자 generation/cursor와 관측 이력이 다릅니다")
        if self._pending_decision is not None:
            raise RuntimeError("앞선 제안의 acknowledge가 필요합니다")
        if self.sample_cursor < self._next:
            return SelectionDecision(None, self._accepted_id, "window_or_interval_wait", self.sample_cursor)
        self._next = self.sample_cursor + self.interval
        feature = ref_feature(self._history, self.bank.config.sample_rate)
        if feature is None:
            return SelectionDecision(None, self._accepted_id, "unavailable_reference_power", self.sample_cursor)
        index = int(np.argmin(np.sum((self._features - feature) ** 2, axis=1)))
        candidate = self.bank.candidates[index]
        if candidate.candidate_id == self._accepted_id:
            return SelectionDecision(None, candidate.candidate_id, "same_candidate_hold", self.sample_cursor)
        self._revision += 1
        proposal = FilterProposal(self.bank.context_id, generation, self._revision, self.sample_cursor,
                                  candidate.coefficients, f"{self.bank.bank_id}:{candidate.candidate_id}")
        decision = SelectionDecision(proposal, candidate.candidate_id, "proposal", self.sample_cursor)
        self._pending_decision = decision
        return decision

    def acknowledge(self, decision: SelectionDecision, *, accepted: bool):
        if decision != self._pending_decision or not isinstance(accepted, bool):
            raise ValueError("현재 제안에 대한 bool acknowledge만 허용합니다")
        if accepted:
            self._accepted_id = decision.candidate_id
        self._pending_decision = None


def _rollout(config, bank, reference, disturbance, variant, saturation):
    controller = None if variant == "zero_control" else config.controller()
    selector = PastRefSelector(bank) if variant == "selected_fxnlms" else None
    if variant == "fixed_fir":
        # train에서 미리 정한 mixed/첫 seed. validation/test 점수로 고르지 않는다.
        candidate = next(c for c in bank.candidates if c.source_family == "mixed")
        accepted = controller.submit(FilterProposal(bank.context_id, controller.generation, 1, 0,
                                                     candidate.coefficients, candidate.candidate_id))
        if not accepted.accepted:
            raise ValueError(f"고정 후보 거부: {accepted.reason}")
    plant = _Plant(config, saturation)
    previous = np.zeros(config.hop, dtype=np.float32)
    errors, controls = np.zeros_like(reference), np.zeros_like(reference)
    events, updates = [], 0
    for begin in range(0, reference.size, config.hop):
        end = begin + config.hop
        if selector is not None:
            # 현재 블록을 observe하기 전에 선택한다. 선택 계산은 오프라인 dispatcher에만 있다.
            choice = selector.choose(generation=controller.generation, consumed_until_sample=controller.sample_cursor)
            if choice.proposal is not None:
                decision = controller.submit(choice.proposal)
                selector.acknowledge(choice, accepted=decision.accepted)
                events.append({"apply_at_sample": begin, "observed_until_sample": choice.observed_until_sample,
                               "candidate_id": choice.candidate_id, "revision": choice.proposal.revision,
                               "generation": choice.proposal.generation, "context_id": choice.proposal.context_id,
                               "accepted": decision.accepted, "reason": decision.reason})
            elif choice.reason != "window_or_interval_wait":
                events.append({"apply_at_sample": begin, "observed_until_sample": choice.observed_until_sample,
                               "candidate_id": choice.candidate_id, "accepted": False, "reason": choice.reason})
        error = disturbance[begin:end] + plant.step(previous)
        if controller is None:
            output = np.zeros(config.hop, dtype=np.float32)
        else:
            output = controller.generate_block(reference[begin:end])
            result = controller.adapt_block(error, enabled=variant in ("cold_fxnlms", "selected_fxnlms"))
            updates += int(result.adapted)
        errors[begin:end], controls[begin:end] = error, output
        previous = output
        if selector is not None:
            selector.observe(reference[begin:end], end_sample_exclusive=controller.sample_cursor)
    return {"error": errors, "control": controls, "events": events,
            "adapted_blocks": updates, "clip_samples": int(getattr(controller, "clip_samples", 0)),
            "generation": int(getattr(controller, "generation", 0))}


def _band_powers(values, sample_rate):
    values = np.asarray(values, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(values)) ** 2 / values.size ** 2
    weights = np.full(spectrum.shape, 2.0)
    weights[0] = 1.0
    if values.size % 2 == 0:
        weights[-1] = 1.0
    frequencies = np.fft.rfftfreq(values.size, 1 / sample_rate)
    spectrum *= weights
    return {"fullband": float(np.mean(values * values)),
            "below_800": float(spectrum[frequencies < 800].sum()),
            "target_800_1600": float(spectrum[(frequencies >= 800) & (frequencies < 1600)].sum()),
            "above_1600": float(spectrum[frequencies >= 1600].sum())}


def _metric_rows(config, description, disturbance, result, *, failure=None):
    count = disturbance.size
    span = count // 8
    change_at_err = (config.eval_blocks // 2) * config.hop + config.preview(description["scenario"])
    periods = {"early": (config.control_delay, config.control_delay + span),
               "after_gain_change": (change_at_err, change_at_err + span),
               "steady": (count - span, count)}
    rows = []
    for period, (begin, end) in periods.items():
        baseline = _band_powers(disturbance[begin:end], config.sample_rate)
        residual = _band_powers(result["error"][begin:end], config.sample_rate) if result else {}
        for band, power in baseline.items():
            error = residual.get(band)
            available = error is not None and power > POWER_FLOOR and error > POWER_FLOOR
            reduction = float(10 * (np.log10(power) - np.log10(error))) if available else None
            rows.append({**description, "period": period, "band": band,
                         "start_sample": begin, "end_sample_exclusive": end,
                         "baseline_power": power, "error_power": error, "reduction_db": reduction,
                         "comparison_available": available,
                         "run_failed": failure is not None,
                         "unavailable_reason": ((failure or "unspecified_run_failure") if failure is not None
                                                else (None if available else "insufficient_signal_power")),
                         "amplified": reduction < 0 if available else None,
                         "emergent_error_energy": error > POWER_FLOOR >= power if error is not None else None,
                         "control_peak": float(np.max(np.abs(result["control"][begin:end]))) if result else None,
                         "synthetic_only": True, "physical_claim_allowed": False,
                         "performance_claim_allowed": False})
    return rows


def _aggregate(rows):
    summaries = []
    groups = sorted({(r["scenario"], r["split"], r["variant"], r["band"]) for r in rows})
    for scenario, split, variant, band in groups:
        selected = [r for r in rows if (r["scenario"], r["split"], r["variant"], r["band"]) ==
                    (scenario, split, variant, band) and r["period"] == "steady"]
        values = [r["reduction_db"] for r in selected if r["comparison_available"]]
        summaries.append({"scenario": scenario, "split": split, "variant": variant, "band": band,
                          "total_rows": len(selected), "available_rows": len(values),
                          "unavailable_rows": len(selected) - len(values),
                          "amplified_rows": sum(v < 0 for v in values),
                          "emergent_rows": sum(r.get("emergent_error_energy") is True for r in selected),
                          "failed_rows": sum(r.get("run_failed") is True for r in selected),
                          "statistics_scope": "available_finite_db_rows_only",
                          "median_db": float(np.median(values)) if values else None,
                          "p10_db": float(np.percentile(values, 10)) if values else None,
                          "worst_db": min(values) if values else None,
                          "physical_claim_allowed": False})
    return summaries


def prepare_and_evaluate(config: BankConfig):
    """조건별 train bank를 만들고 고정된 validation/test 프로토콜을 전부 평가한다."""
    banks, training, rows, runs = {}, [], [], []
    for scenario in SCENARIOS:
        try:
            bank = build_synthetic_bank(config, scenario)
            banks[scenario] = bank
            training.append({"scenario": scenario, "status": "valid", "bank_id": bank.bank_id,
                             "candidate_count": len(bank.candidates), "context_id": bank.context_id})
        except BankTrainingError as exc:
            training.append({"scenario": scenario, "status": "invalid", "diagnostics": exc.diagnostics})
        bank = banks.get(scenario)
        for split, seeds in (("validation", config.validation_seeds), ("test", config.test_seeds)):
            for seed in seeds:
                for family in EVAL_FAMILIES:
                    for level in config.evaluation_levels:
                        reference = synthetic_reference(config, family, seed, split, level=level)
                        gain = np.full(reference.size, 0.15, dtype=np.float32)
                        gain[(config.eval_blocks // 2) * config.hop:] = 0.20
                        disturbance = SampleDelay(config.preview(scenario)).process(gain * reference)
                        for saturation in config.saturation_levels:
                            for variant in VARIANTS:
                                description = {"scenario": scenario, "split": split, "seed": seed,
                                               "source_family": family, "level": level,
                                               "saturation_level": saturation, "variant": variant}
                                result, failure = None, None
                                if bank is None and variant in ("fixed_fir", "selected_fxnlms"):
                                    failure = "missing_invalid_training_bank"
                                else:
                                    try:
                                        result = _rollout(config, bank, reference, disturbance, variant, saturation)
                                    except (ValueError, RuntimeError, FloatingPointError) as exc:
                                        failure = str(exc)
                                rows.extend(_metric_rows(config, description, disturbance, result, failure=failure))
                                runs.append({**description, "status": "complete" if result else "unavailable",
                                             "bank_id": bank.bank_id if bank else None,
                                             "failure": failure, "reference_sha256": _sha(reference.tobytes()),
                                             "events": result["events"] if result else [],
                                             "clip_samples": result["clip_samples"] if result else None,
                                             "adapted_blocks": result["adapted_blocks"] if result else None,
                                             "generation": result["generation"] if result else None})
    report = {
        "schema": "synthetic_filter_bank_evaluation.v1", "synthetic_only": True, "diagnostic_only": True,
        "physical_claim_allowed": False, "performance_claim_allowed": False, "real_time_claim_allowed": False,
        "speech_music_evaluated": False, "real_data_used": False, "config": asdict(config),
        "controller_configuration": config.controller_configuration, "context_id": config.controller().context_id,
        "split_protocol": {"train": list(config.train_seeds), "validation": list(config.validation_seeds),
                           "test": list(config.test_seeds), "validation_tuning_used": False,
                           "train_frequencies_hz": [300, 900, 1300],
                           "validation_frequencies_hz": [350, 1000, 1400], "test_frequencies_hz": [250, 850, 1550]},
        "selector": {"name": "past_ref_nearest_psd_rms", "uses_error": False, "uses_future": False,
                     "feature_schema": _feature_schema(config),
                     "window_samples": config.window_blocks * config.hop,
                     "interval_samples": config.selection_interval_blocks * config.hop,
                     "scheduling": "offline dispatcher before current block; no real-time timing claim",
                     "same_candidate": "hold_without_reinstall", "rejected_proposal": "reevaluate_at_next_interval",
                     "fixed_baseline": "first mixed candidate in train order, never score-selected"},
        "metric_definition": {"power_floor": POWER_FLOOR, "power": "rectangular-window FFT/mean-square",
                              "target_band_hz": [800, 1600], "target_high_endpoint_included": False,
                              "aggregate_pooling": "steady rows, by scenario/split/variant/band; pools seeds/families/levels/saturation; statistics use available dB only, emergent/failed counted separately"},
        "training": training, "runs": runs, "metrics": rows, "aggregate": _aggregate(rows),
        "warnings": [
            "합성 전용입니다. 실제 음성·음악·덕트·USB·Jetson 검증/실측 경로 사용은 하지 않았습니다.",
            "causal_toy와 long_delay_stress는 각각 해당 preview로 별도 train bank를 준비했습니다.",
            "긴 지연 숫자만 기존 기록을 참고하고 S 자체는 합성입니다. 인과성 한계가 제거되지 않습니다.",
            "train/validation/test의 seed·톤 주파수는 분리됩니다. 기본 평가 레벨은 train과 다르지만 validation/test끼리는 같으며, 같은 합성 발생기로 실데이터 일반화는 아닙니다.",
            "선택기는 REF PSD/RMS만 사용합니다. REF로 식별할 수 없는 경로 변화는 선택 근거가 없으며 ERR 적응과 구분합니다.",
            "선형 FIR 선택과 합성 tanh 스트레스는 비선형 제어의 해결/안정성 증명이 아닙니다.",
            "실패·무신호·대역 밖·증폭 행을 삭제하지 않습니다. FFT 유한 창 누설 때문에 미약한 대역 값에도 주의하세요.",
            "학습 clipping/계수 검증 실패는 해당 bank 저장을 거부하며 다른 seed 재시도/성공 후보 선별을 하지 않습니다.",
            "JSON/NPZ SHA는 재현/손상 검사이지 출처 인증이나 실기 배포 허가가 아닙니다.",
        ],
    }
    return banks, report
