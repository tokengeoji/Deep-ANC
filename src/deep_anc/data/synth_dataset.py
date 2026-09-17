"""온더플라이 합성 학습 데이터셋.

신호 모델 (Deep ANC 방식):
    x_ref(t) : 모델 입력 ch0 (레퍼런스)
    d(t)     : 에러 마이크 위치의 1차경로 소음 (타깃 아님 — 손실에서 e=d+S·y)
    err_in   : 모델 입력 ch1 (에러 피드백 근사; 캡처+블록 지연 랜덤화 [H3])

레퍼런스 모드별 지연 물리 [설계 교차검증 C2]:
  digital  : x_ref = n(t) (Jetson 이 소음을 직접 생성).
             d = P(z) · n(t − D_noise). P(z)는 noise→ERR 실측 FIR(권장),
             S(z) gain/FIR 대용, 기존 p_err RIR 중 하나를 명시 선택한다.
             D_noise = 실측(권장) 또는 기하 추정
             s_delay − t_ac(CS→ERR) + t_ac(NS→ERR).
             출력버퍼 지연이 소음·상쇄 경로에 공통 → 광대역 상쇄가 인과적으로 가능.
             digital_reference_lead_samples=K 이면 x_ref(t)=n(t+K). 런타임에서
             소음 재생을 K만큼 지연해 실제로 확보하는 미래이므로 오라클 누설이 아니다.
  acoustic : x_ref = P_ref · n(t), d = P_err · n(t).
             S(z) 실측 지연(≈28ms)이 그대로 예측 부담이 됨 → 주기성/협대역 한정.

S(z) 플랜트와 핸드오프(+256)는 손실 모듈(anc_loss)에서 적용된다 — 여기서는
d 경로만 만든다 (소음 ch0 은 콜백에서 직접 생성되므로 핸드오프가 없다 [C1]).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..config import DEFAULT_HANDOFF_SAMPLES, _resolve_path, default_d_noise_delay, duct_distance_samples
from ..dsp.duct_sim import build_rir_bank
from ..dsp.filters import fft_filter
from ..dsp.secondary_path import load_secondary_path
from .noise_pool import NoisePool
from .manifest import read_manifest, validate_group_splits, validate_source_family
from .primary_path import resolve_digital_primary_path
from .synthetic_signals import SyntheticNoise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_hash(path: Path, expected) -> str:
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError(f"SHA-256 메타가 없거나 잘못됐습니다: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 불일치: {path}")
    return actual


def _train_only_families(data_cfg: dict) -> tuple[str, ...]:
    values = data_cfg.get("train_only_source_families", [])
    if (not isinstance(values, list) or any(not isinstance(value, str) for value in values)
            or len(values) != len(set(values))):
        raise ValueError("train_only_source_families는 중복 없는 family 문자열 목록이어야 합니다")
    for value in values:
        validate_source_family(value)
        if value == "synthetic":
            raise ValueError("synthetic은 train_only_source_families로 제한하지 않습니다")
    # 현재 strict machine 원본은 MIMII다. 설정 누락/해제로 독립 평가에 섞지 않는다.
    # 구형 non-prepared 경로는 opt-in하지 않으면 기존 동작을 유지한다.
    if data_cfg.get("require_prepared_data") is True and values != ["machine"]:
        raise ValueError("prepared train_only_source_families는 ['machine']이어야 합니다")
    return tuple(values)


def source_mix_for_split(data_cfg: dict, split: str = "train") -> dict:
    """학습 전용 원본은 val/test에서 제외하고 나머지 상대 비율만 보존한다."""
    if split not in ("train", "val", "test"):
        raise ValueError("source mix split은 train/val/test여야 합니다")
    train_only = _train_only_families(data_cfg)
    if data_cfg.get("reference_mode") == "acoustic" and data_cfg.get("source_mix_ratio_acoustic"):
        mix = dict(data_cfg["source_mix_ratio_acoustic"])
    else:
        mix = dict(data_cfg.get("source_mix_ratio", {"synthetic": 1.0}))
    if split == "train" or not train_only:
        return mix
    if any(isinstance(value, (bool, str)) or not np.isfinite(value) or value < 0
           for value in mix.values()):
        raise ValueError("source mix 비율은 유한한 비음수여야 합니다")
    allowed = {tag: value for tag, value in mix.items() if tag not in train_only and value > 0}
    total = sum(allowed.values())
    if not allowed or not np.isfinite(total) or total <= 0:
        raise ValueError("학습 보조 전용 원본 제외 후 평가 가능한 source가 없습니다")
    return {tag: value / total for tag, value in allowed.items()}


def validate_prepared_handoff(duct_cfg: dict) -> int:
    """명시 prepared 경로의 handoff는 반올림/절단 없는 0 이상 정수여야 한다."""
    handoff = duct_cfg["secondary_path"].get("handoff_extra_samples", DEFAULT_HANDOFF_SAMPLES)
    if type(handoff) is not int or handoff < 0:
        raise ValueError("prepared handoff_extra_samples는 0 이상의 정수여야 합니다")
    return handoff


def validate_prepared_data(data_cfg: dict) -> tuple[dict, dict[str, np.ndarray]]:
    """명시 stage된 자료만 검사한다. 다운로드/Drive 조회/합성 fallback은 하지 않는다.

    qa.json은 준비 도구의 감사 기록이지 서명된 신뢰 루트나 음향 성능 인증은 아니다.
    원본/공식 metadata/manifest hash와 split을 다시 검사해 준비 후 변조를 거부한다.
    """
    import soundfile as sf

    if data_cfg.get("reference_mode") != "acoustic" or data_cfg.get("digital_reference_lead_samples") != 0:
        raise ValueError("prepared 데이터는 acoustic reference/lead=0만 지원합니다")
    mix = source_mix_for_split(data_cfg)
    if (not mix or any(isinstance(v, (bool, str)) or not np.isfinite(v) or v < 0 for v in mix.values())
            or not np.isclose(sum(mix.values()), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("prepared source_mix_ratio는 유한한 비음수이며 합이 1이어야 합니다")
    train_only = _train_only_families(data_cfg)
    split_mixes = {split: source_mix_for_split(data_cfg, split) for split in ("train", "val", "test")}
    families = sorted(tag for tag, ratio in mix.items() if tag != "synthetic" and ratio > 0)
    for family in families:
        validate_source_family(family)
    manifest_dir = _resolve_path(data_cfg["noise_manifest_dir"]).resolve()
    qa_path = manifest_dir / "qa.json"
    qa_bytes = qa_path.read_bytes()
    qa = json.loads(qa_bytes)
    qa_hash = hashlib.sha256(qa_bytes).hexdigest()
    # 해시는 실제 파싱한 QA 바이트에 고정한다. 모든 입력을 종료 시 다시 대조해
    # 긴 raw 검사 중 바뀐 metadata/manifest/PCM에 과거 PASS를 결합하지 않는다.
    input_hashes: dict[Path, str] = {qa_path: qa_hash}
    if (not isinstance(qa, dict) or type(qa.get("schema_version")) is not int or qa["schema_version"] != 1
            or qa.get("data_ready") is not True or qa.get("diagnostic_only") is not True
            or qa.get("performance_claim_allowed") is not False):
        raise ValueError("로컬 prepared QA data_ready/schema/진단 범위를 확인할 수 없습니다")
    if not set(families).issubset(qa.get("families", [])):
        raise ValueError("prepared QA에 요청한 모든 source family가 필요합니다")
    if qa.get("train_only_source_families") != list(train_only):
        raise ValueError("prepared QA train_only_source_families 정책 누락/불일치")
    if not isinstance(qa.get("manifest_sha256"), dict) or not isinstance(qa.get("split_counts"), dict):
        raise ValueError("prepared QA manifest_sha256/split_counts가 필요합니다")
    raw_root = Path(qa["raw_root"])
    if not raw_root.is_absolute() or not raw_root.is_dir():
        raise ValueError("prepared QA raw_root는 명시 stage된 절대 디렉터리여야 합니다")
    raw_root = raw_root.resolve()
    inventory_hash = _check_hash(manifest_dir / "inventory.jsonl", qa.get("inventory_sha256"))
    input_hashes[manifest_dir / "inventory.jsonl"] = inventory_hash
    inventory_rows = read_manifest(manifest_dir / "inventory.jsonl")
    metadata = qa.get("source_metadata_sha256")
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("공식 source metadata SHA-256이 필요합니다")
    for relative, digest in metadata.items():
        meta_path = (raw_root / relative).resolve()
        if Path(relative).is_absolute() or not meta_path.is_relative_to(raw_root):
            raise ValueError("source metadata 경로가 raw_root 밖입니다")
        input_hashes[meta_path] = _check_hash(meta_path, digest)
    entries, hashes, counts, seen_hashes = [], {}, {}, set()
    for family in families:
        manifest_path = manifest_dir / f"{family}.jsonl"
        hashes[manifest_path.name] = _check_hash(manifest_path, qa.get("manifest_sha256", {}).get(manifest_path.name))
        input_hashes[manifest_path] = hashes[manifest_path.name]
        rows = read_manifest(manifest_path)
        counts[family] = {split: 0 for split in ("train", "val", "test")}
        for row in rows:
            if (row.get("source_family") != family or row.get("tag") != family
                    or row.get("path_base") != "manifest" or not row.get("group_id")
                    or row.get("qa_valid") is not True
                    or row.get("split") not in counts[family]):
                raise ValueError(f"prepared manifest family/group/split/path 메타 오류: {manifest_path}")
            if family in train_only:
                role = row.get("metadata")
                if (row["split"] != "train" or row["group_id"] != "machine:mimii-dg-unresolved"
                        or not isinstance(role, dict)
                        or role.get("usage_policy") != "train_only_auxiliary"
                        or role.get("independent_evaluation_allowed") is not False):
                    raise ValueError("MIMII 학습 보조 전용: train split/보수적 그룹/평가 제외 metadata가 필요합니다")
            raw_path = Path(row["path"]).resolve()
            if not raw_path.is_relative_to(raw_root):
                raise ValueError("음원 경로가 staged raw_root 밖입니다")
            digest = _check_hash(raw_path, row.get("sha256"))
            input_hashes[raw_path] = digest
            if digest in seen_hashes:
                raise ValueError("prepared 원본 SHA-256 중복: split/group 누수 위험")
            seen_hashes.add(digest)
            info = sf.info(raw_path)
            if (info.frames <= 0 or any(row.get(key) != value for key, value in (
                    ("sample_rate", info.samplerate), ("channels", info.channels), ("frames", info.frames)))
                    or not np.isfinite(row.get("duration_s", np.nan))
                    or abs(row["duration_s"] - info.frames / info.samplerate) > 1 / info.samplerate):
                raise ValueError(f"prepared 원본 오디오 header 불일치: {raw_path}")
            counts[family][row["split"]] += 1
        if family in train_only:
            if counts[family]["train"] == 0 or counts[family]["val"] or counts[family]["test"]:
                raise ValueError(f"prepared {family}: 학습 전용 train>0, val=test=0이어야 합니다")
        elif any(count == 0 for count in counts[family].values()):
            raise ValueError(f"prepared {family}: 비어 있는 train/val/test split")
        if qa.get("split_counts", {}).get(family) != counts[family]:
            raise ValueError(f"prepared QA split_counts 불일치: {family}")
        entries.extend(rows)
    validate_group_splits(entries)
    inventory_subset = [row for row in inventory_rows if row.get("source_family") in families]
    key = lambda row: (row["source_family"], row["path"])
    if (len(inventory_subset) != len(entries)
            or {key(row): row for row in inventory_subset} != {key(row): row for row in entries}):
        raise ValueError("prepared inventory와 family manifest 행이 다릅니다")
    bank_path = _resolve_path(data_cfg["rir_bank"]).resolve()
    rir_hash = _sha256_file(bank_path)
    with np.load(bank_path, allow_pickle=False) as archive:
        fs = np.asarray(archive["sample_rate"])
        if fs.ndim != 0 or fs.dtype.kind not in "iu" or fs.item() != data_cfg["sample_rate"]:
            raise ValueError("prepared RIR sample_rate 불일치")
        rirs = {key: archive[key] for key in ("p_ref", "p_err", "f_fb")}
    if (len({value.shape for value in rirs.values()}) != 1
            or any(value.ndim != 2 or value.shape[0] < 3 or value.shape[1] == 0
                   or value.dtype.kind not in "fiu" or not np.isfinite(value).all()
                   or not np.all(np.any(value != 0, axis=1)) for value in rirs.values())):
        raise ValueError("prepared RIR는 동일 shape의 유한 nonzero 2D 경로/최소3변형이 필요합니다")
    with np.errstate(over="ignore", under="ignore"):
        rirs = {key: value.astype(np.float32) for key, value in rirs.items()}
    if any(not np.isfinite(value).all() or not np.all(np.any(value != 0, axis=1)) for value in rirs.values()):
        raise ValueError("prepared RIR가 float32 처리 범위를 벗어납니다")
    variant_hashes = [hashlib.sha256(b"".join(rirs[key][i].tobytes() for key in sorted(rirs))).hexdigest()
                      for i in range(rirs["p_ref"].shape[0])]
    if len(set(variant_hashes)) != len(variant_hashes):
        raise ValueError("prepared RIR 변형 중복: split 누수 위험")
    if _sha256_file(bank_path) != rir_hash:
        raise ValueError("검사 중 RIR 파일이 변경됐습니다")
    for input_path, digest in input_hashes.items():
        _check_hash(input_path, digest)
    return {
        "data_ready": True, "diagnostic_only": True, "performance_claim_allowed": False,
        "raw_root": str(raw_root), "qa_sha256": qa_hash,
        "inventory_sha256": inventory_hash, "manifest_sha256": hashes,
        "source_metadata_sha256": metadata, "split_counts": counts,
        "train_only_source_families": list(train_only), "effective_source_mix_by_split": split_mixes,
        "rir_sha256": rir_hash, "rir_path": str(bank_path), "rir_variants": len(variant_hashes),
    }, rirs


def _delay_np(x: np.ndarray, delay: int) -> np.ndarray:
    if delay <= 0:
        return x.copy()
    out = np.zeros_like(x)
    out[delay:] = x[: x.size - delay]
    return out


class SynthANCDataset(IterableDataset):
    """무한 IterableDataset — 매 아이템 (소음원, RIR 변형, 레벨, 지연)을 랜덤 추첨."""

    def __init__(
        self,
        data_cfg: dict,
        duct_cfg: dict,
        split: str = "train",
        seed: int = 20260802,
        rir_bank: dict[str, np.ndarray] | None = None,
    ) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.duct_cfg = duct_cfg
        self.split = split
        self.seed = int(seed)
        self.require_prepared_data = data_cfg.get("require_prepared_data", False)
        if type(self.require_prepared_data) is not bool:
            raise ValueError("require_prepared_data는 bool이어야 합니다")
        self.train_only_source_families = _train_only_families(data_cfg)
        self.prepared_data_metadata = None
        if self.require_prepared_data:
            validate_prepared_handoff(duct_cfg)
            if rir_bank is not None:
                raise ValueError("prepared 모드의 RIR 파일 검증을 인메모리 값으로 우회할 수 없습니다")
            self.prepared_data_metadata, rir_bank = validate_prepared_data(data_cfg)
        self.fs = int(data_cfg["sample_rate"])
        # 목표 분포 설정도 iterator 첫 호출까지 미루지 않고 검증한다.
        self.synthetic_generator(seed=0)
        # 세그먼트를 런타임 블록(256 = 모델 hop 128×2)의 배수로 내림 — 모델 입력 요건
        raw_segment = int(round(float(data_cfg["segment_seconds"]) * self.fs))
        self.segment = max(256, (raw_segment // 256) * 256)
        self.reference_mode = str(data_cfg.get("reference_mode", "digital"))
        if self.reference_mode not in ("digital", "acoustic"):
            raise ValueError(f"reference_mode: {self.reference_mode}")
        self.digital_reference_lead = int(
            data_cfg.get("digital_reference_lead_samples", 0)
        )
        if self.digital_reference_lead < 0:
            raise ValueError("digital_reference_lead_samples는 0 이상이어야 합니다")
        if self.reference_mode != "digital" and self.digital_reference_lead:
            raise ValueError(
                "digital_reference_lead_samples는 reference_mode=digital에서만 "
                "사용할 수 있습니다"
            )

        # RIR 뱅크: 파일이 있으면 로드, 없으면 소규모 즉석 생성 (스모크 테스트용)
        # 경로는 저장소 루트 기준으로 해석 — 실행 위치(CWD)에 의존하지 않는다 (감사 M2)
        if rir_bank is not None:
            self.rirs = rir_bank
        else:
            bank_path = data_cfg.get("rir_bank")
            try:
                with np.load(_resolve_path(bank_path)) as z:
                    self.rirs = {k: z[k] for k in ("p_ref", "p_err", "f_fb")}
            except (TypeError, FileNotFoundError, OSError):
                print(
                    "=" * 70 + f"\n[synth_dataset 경고] RIR 뱅크({bank_path})가 없어 즉석 32개로 "
                    "대체합니다.\n  본 학습에서는 도메인 랜덤화가 크게 약해집니다 — 반드시 "
                    "scripts/data/build_rir_bank.py 를 먼저 실행하세요.\n" + "=" * 70
                )
                self.rirs = build_rir_bank(duct_cfg, self.fs, n_variants=32, seed=self.seed)

        n_var = self.rirs["p_err"].shape[0]
        # RIR 변형도 split 단위 분할 (누수 방지)
        idx = np.arange(n_var)
        rng = np.random.default_rng(20260801)
        rng.shuffle(idx)
        n_val = max(1, int(n_var * 0.05))
        n_test = max(1, int(n_var * 0.05))
        self.rir_indices = {
            "val": idx[:n_val],
            "test": idx[n_val : n_val + n_test],
            "train": idx[n_val + n_test :],
        }[split]

        # 소스 풀 — source_mix_ratio 의 키가 곧 태그다 ('synthetic' 제외).
        # manifest(data/manifests/<tag>.jsonl)가 없는 태그는 합성원으로 자동 폴백하므로,
        # 데이터셋을 나중에 추가해도 설정 변경 없이 활성화된다 (speech/music 등).
        # acoustic-ref 는 전용 소스 구성을 사용 (주기성↑ + 예측불가 성분 무해화 학습) [로드맵 A2]
        self.mix_ratio = source_mix_for_split(data_cfg, split)
        manifest_dir = _resolve_path(data_cfg.get("noise_manifest_dir", "data/manifests"))
        self.pools: dict[str, list] = {
            tag: [str(manifest_dir / f"{tag}.jsonl")]
            for tag, ratio in self.mix_ratio.items()
            if tag != "synthetic" and float(ratio) > 0.0
        }
        self._pool_objs: dict[str, NoisePool] = {}
        self.dc_hum_prob = float(data_cfg.get("dc_hum_prob", 0.0))

        # digital-ref 1차경로 P(z). 실측/secondary-surrogate는 compact FIR과
        # D_noise 총지연을 resolver가 분리해 반환하므로 둘을 각각 정확히 한 번 적용한다.
        # legacy rir_surrogate만 p_err 안의 음향 onset을 고려해 추가지연을 계산한다.
        sp = load_secondary_path(_resolve_path(duct_cfg["secondary_path"]["npz"]))
        if self.require_prepared_data and sp.sample_rate != self.fs:
            raise ValueError("prepared S sample_rate와 데이터 sample_rate가 다릅니다")
        self.digital_primary_path = None
        self.digital_primary_path_mode = str(
            data_cfg.get("digital_primary_path_mode", "rir_surrogate")
        )
        if self.reference_mode == "digital":
            self.digital_primary_path, self.d_noise_total = resolve_digital_primary_path(
                data_cfg, duct_cfg, self.fs, sp
            )
        else:
            # acoustic-ref는 기존 P_ref/P_err RIR 경로만 사용한다. digital P(z) 모드가
            # measured여도 파일을 요구하지 않아 두 모드의 물리를 서로 침범하지 않는다.
            d_noise = duct_cfg.get("digital_reference", {}).get("d_noise_delay_samples")
            if d_noise is None:
                d_noise = default_d_noise_delay(duct_cfg, self.fs, sp.delay_samples)
            self.d_noise_total = int(d_noise)

        if self.digital_primary_path is None:
            # 규약: d_noise_total은 "디지털 출력→ERR 총 순수지연"이다. p_err
            # RIR에는 t_ac(NS→ERR)가 이미 포함되어 있으므로 전기/버퍼분만 더한다.
            t_ns_err = duct_distance_samples(
                duct_cfg, "noise_speaker", "error_mic", self.fs
            )
            self.d_noise_delay = max(0, self.d_noise_total - t_ns_err)
        else:
            self.d_noise_delay = int(self.digital_primary_path.delay_samples)

        self.level_range = tuple(data_cfg.get("level_dbfs", [-35, -10]))
        self.snr_range = tuple(data_cfg.get("snr_mic_noise_db", [5, 30]))
        fb = data_cfg.get("closed_loop", {}).get("feedback_delay_samples", [512, 1024])
        self.feedback_delay_range = (int(fb[0]), int(fb[1]))

    # ---------- 내부 ----------

    def _pool(self, tag: str, rng: np.random.Generator) -> NoisePool | None:
        if self.split != "train" and tag in self.train_only_source_families:
            raise ValueError(f"학습 보조 전용 원본은 평가/합성 대체에 사용할 수 없습니다: {tag}")
        if tag not in self.pools:
            return None
        if tag not in self._pool_objs:
            try:
                if self.require_prepared_data:
                    for manifest in self.pools[tag]:
                        path = Path(manifest)
                        _check_hash(path, self.prepared_data_metadata["manifest_sha256"][path.name])
                self._pool_objs[tag] = NoisePool(
                    self.pools[tag], self.split, self.fs, seed=int(rng.integers(1 << 31)),
                    strict=self.require_prepared_data,
                )
            except (FileNotFoundError, ValueError):
                if self.require_prepared_data:
                    raise
                print(f"[synth_dataset] {tag} manifest 없음 — 합성원으로 대체합니다")
                self.pools.pop(tag)
                return None
        return self._pool_objs[tag]

    def _sample_source(
        self,
        rng: np.random.Generator,
        synth: SyntheticNoise,
        n_samples: int | None = None,
    ) -> np.ndarray:
        n_samples = self.segment if n_samples is None else int(n_samples)
        tags = list(self.mix_ratio.keys())
        probs = np.array([self.mix_ratio[t] for t in tags], dtype=np.float64)
        probs = probs / probs.sum()
        tag = str(rng.choice(tags, p=probs))
        if tag != "synthetic":
            pool = self._pool(tag, rng)
            if pool is not None:
                seg = pool.sample_segment(n_samples)
                rms = float(np.sqrt(np.mean(seg**2)) + 1e-9)
                return seg / rms
        return synth.generate(n_samples)

    def _make_item(self, rng: np.random.Generator, synth: SyntheticNoise) -> dict:
        # digital lead가 켜졌을 때 tail을 0으로 채우면 세그먼트 끝에만 존재하는
        # 인공 패턴을 학습한다. 실제로 연속된 source를 K샘플 더 뽑아 미래 ref를 만든다.
        source_len = self.segment + self.digital_reference_lead
        n_full = self._sample_source(rng, synth, source_len)

        # 레벨 랜덤화
        level_db = float(rng.uniform(*self.level_range))
        n_full = n_full * (10.0 ** (level_db / 20.0))
        n = n_full[: self.segment]

        # RIR 변형 추첨
        ridx = int(rng.choice(self.rir_indices))
        p_ref = self.rirs["p_ref"][ridx]
        p_err = self.rirs["p_err"][ridx]

        if self.reference_mode == "digital":
            lead = self.digital_reference_lead
            x_ref = n_full[lead : lead + self.segment].copy()
            if self.digital_primary_path is None:
                # legacy rir_surrogate: p_err의 음향 onset + 전기/버퍼 추가지연.
                d = _delay_np(fft_filter(n, p_err), self.d_noise_delay)
            else:
                # measured/secondary_surrogate: compact FIR과 총 순수지연을 각 1회.
                d = _delay_np(
                    fft_filter(n, self.digital_primary_path.fir),
                    self.digital_primary_path.delay_samples,
                )
        else:
            x_ref = fft_filter(n, p_ref)
            d = fft_filter(n, p_err)

        # 에러 피드백 입력 (open-loop 근사: d 를 캡처+블록 지연 후 공급) [H3]
        fb_delay = int(rng.integers(*self.feedback_delay_range))
        err_in = _delay_np(d, fb_delay)

        # 마이크 자기잡음
        snr_db = float(rng.uniform(*self.snr_range))
        for sig in (x_ref, err_in):
            p_sig = float(np.mean(sig**2) + 1e-12)
            p_noise = p_sig / (10.0 ** (snr_db / 10.0))
            sig += rng.standard_normal(sig.size).astype(np.float32) * np.sqrt(p_noise)

        # 전원 험(50/60Hz + 2차 고조파) — 배포 환경의 DC/저역 험 모사 (런타임은 DCBlocker 보유)
        if self.dc_hum_prob > 0.0 and rng.random() < self.dc_hum_prob:
            f_hum = float(rng.choice([50.0, 60.0]))
            t = np.arange(self.segment) / self.fs
            rms_ref = float(np.sqrt(np.mean(x_ref**2)) + 1e-9)
            amp = rms_ref * (10.0 ** (float(rng.uniform(-35.0, -20.0)) / 20.0))
            hum = amp * (
                np.sin(2 * np.pi * f_hum * t + rng.uniform(0, 2 * np.pi))
                + 0.4 * np.sin(2 * np.pi * 2 * f_hum * t + rng.uniform(0, 2 * np.pi))
            ).astype(np.float32)
            x_ref += hum
            err_in += hum

        # 채널 dropout — ref-only / err-only 운용 대비 (동시 제거는 금지)
        u = rng.random()
        if u < 0.15:
            err_in = np.zeros_like(err_in)
        elif u < 0.30:
            x_ref = np.zeros_like(x_ref)

        x = np.stack([x_ref, err_in]).astype(np.float32)   # [2, T]
        if self.require_prepared_data:
            with np.errstate(over="ignore", invalid="ignore"):
                d = d.astype(np.float32)
        if self.require_prepared_data and (not np.isfinite(x).all() or not np.isfinite(d).all()):
            raise ValueError("prepared acoustic item의 수치 범위를 벗어났습니다")
        return {
            "x": torch.from_numpy(x),
            "d": torch.from_numpy(d.astype(np.float32)).unsqueeze(0),  # [1, T]
        }

    # ---------- IterableDataset ----------

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        split_offset = {"train": 0, "val": 7919, "test": 15859}[self.split]
        rng = np.random.default_rng(self.seed + split_offset + worker_id * 1009)
        synth = self.synthetic_generator(seed=int(rng.integers(1 << 31)))
        while True:
            yield self._make_item(rng, synth)

    def synthetic_generator(self, seed: int) -> SyntheticNoise:
        return SyntheticNoise(
            self.fs, seed=seed, target_band_hz=self.data_cfg.get("synthetic_target_band_hz"),
            target_probability=self.data_cfg.get("synthetic_target_probability", 0.0),
        )


def make_eval_batch(
    dataset: SynthANCDataset, n_items: int, seed: int = 12345
) -> dict[str, torch.Tensor]:
    """고정 시드 검증 배치 — 학습 중 val NMSE 추적용 (매번 동일 데이터)."""
    if dataset.train_only_source_families and dataset.split not in ("val", "test"):
        raise ValueError("학습 전용 source 정책의 평가 배치는 val/test dataset이 필요합니다")
    rng = np.random.default_rng(seed)
    synth = dataset.synthetic_generator(seed=seed)
    items = [dataset._make_item(rng, synth) for _ in range(n_items)]
    return {
        "x": torch.stack([it["x"] for it in items]),
        "d": torch.stack([it["d"] for it in items]),
    }
