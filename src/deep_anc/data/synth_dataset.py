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

import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..config import _resolve_path, default_d_noise_delay, duct_distance_samples
from ..dsp.duct_sim import build_rir_bank
from ..dsp.filters import fft_filter
from ..dsp.secondary_path import load_secondary_path
from ..dsp.timing import PlantDelays, TrainingTimingContract
from .manifest_contract import validate_manifest_generation
from .noise_pool import NoisePool
from .primary_path import resolve_digital_primary_path
from .resumable_stream import indexed_rng, worker_global_item_indices
from .synthetic_signals import SyntheticNoise


def assert_declared_sources_exist(
    pools: dict[str, list[str]],
    mix_ratio: dict[str, float],
    *,
    allow_missing: bool = False,
) -> None:
    """선언한 소스 태그의 manifest 가 **전부 있는지** 구성 시점에 확인한다.

    왜 구성 시점인가
    ----------------
    예전에는 ``_pool()`` 이 manifest 를 못 열면 한 줄 출력하고 태그를 버린 뒤
    합성원으로 대체했다. 그 코드는 **DataLoader 워커 안에서** 돈다 — 출력은 아무도 안
    보는 로그로 가고, 그때는 이미 학습이 시작된 뒤다. 2026-08-06 실측: 선언 비중
    **0.70**(speech 0.15 · music 0.10 · machine 0.07 · demand 0.08 · dns_fullband 0.30)
    이 통째로 사라진 채 100k step 을 돌 뻔했다. 그 학습은 **음성과 음악을 한 번도 보지
    못한 모델**을 만들고, 그것은 절대목표 2(모든 소리를 제거, 최악값 기준)를 정면으로
    위반한다. 60시간을 태운 뒤에야 알게 된다.

    ``allow_missing`` 은 진단 전용이다. 학습 설정에 넣지 마라 — 넣는 순간 이 게이트가
    사라지고, 사라진 것을 알려 주는 것은 이 문장뿐이다.
    """

    missing = [
        (tag, float(mix_ratio.get(tag, 0.0)), paths[0])
        for tag, paths in sorted(pools.items())
        if not Path(paths[0]).is_file()
    ]
    if not missing:
        return
    weight = sum(item[1] for item in missing)
    detail = "\n".join(f"    {tag} (선언 비중 {ratio:.2f}) — {path}" for tag, ratio, path in missing)
    message = (
        f"선언한 소스 태그 {len(missing)}개의 manifest 가 없습니다 "
        f"(합계 선언 비중 {weight:.2f}):\n{detail}\n"
        "  이 상태로 학습하면 그 태그가 조용히 합성원으로 대체되어, 선언한 "
        "source_mix_ratio 와 다른 분포로 돌게 됩니다.\n"
        "  둘 중 하나를 하세요: (1) scripts/data/prepare_noise_pool.py 로 manifest 를 "
        "만든다, (2) configs/data_sim.yaml 의 source_mix_ratio 에서 그 태그를 지운다.\n"
        "  ⚠ (2)는 절대목표 2(모든 소리를 제거)의 해당 소스를 포기하는 선언이므로 "
        "사람이 판단해야 합니다."
    )
    if allow_missing:
        warnings.warn(f"[진단 모드] {message}", RuntimeWarning, stacklevel=2)
        for tag, _, _ in missing:
            pools.pop(tag, None)
        return
    raise FileNotFoundError(message)


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
        training_batch_size: int = 1,
        resume_batch_index: int = 0,
    ) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.duct_cfg = duct_cfg
        self.split = split
        self.seed = int(seed)
        self.training_batch_size = int(training_batch_size)
        self.resume_batch_index = int(resume_batch_index)
        if self.training_batch_size < 1 or self.resume_batch_index < 0:
            raise ValueError("training_batch_size는 1 이상, resume_batch_index는 0 이상")
        self.fs = int(data_cfg["sample_rate"])
        # 세그먼트를 런타임 블록(256 = 모델 hop 128×2)의 배수로 내림 — 모델 입력 요건
        raw_segment = int(round(float(data_cfg["segment_seconds"]) * self.fs))
        self.segment = max(256, (raw_segment // 256) * 256)
        self.reference_mode = str(data_cfg.get("reference_mode", "digital"))
        if self.reference_mode not in ("digital", "acoustic"):
            raise ValueError(f"reference_mode: {self.reference_mode}")
        configured_lead = data_cfg.get("digital_reference_lead_samples")
        self.digital_reference_lead = (
            None if configured_lead is None else int(configured_lead)
        )
        if self.digital_reference_lead is not None and self.digital_reference_lead < 0:
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
        # acoustic-ref 는 전용 소스 구성을 사용 (주기성↑ + 예측불가 성분 무해화 학습) [로드맵 A2]
        if self.reference_mode == "acoustic" and data_cfg.get("source_mix_ratio_acoustic"):
            self.mix_ratio = dict(data_cfg["source_mix_ratio_acoustic"])
        else:
            self.mix_ratio = dict(data_cfg.get("source_mix_ratio", {"synthetic": 1.0}))
        manifest_dir = _resolve_path(data_cfg.get("noise_manifest_dir", "data/manifests"))
        self.pools: dict[str, list] = {
            tag: [str(manifest_dir / f"{tag}.jsonl")]
            for tag, ratio in self.mix_ratio.items()
            if tag != "synthetic" and float(ratio) > 0.0
        }
        allow_missing_sources = bool(data_cfg.get("allow_missing_source_manifests", False))
        assert_declared_sources_exist(
            self.pools,
            self.mix_ratio,
            allow_missing=allow_missing_sources,
        )
        self._validated_pool_entries: dict[str, list[dict]] = {}
        if self.pools and not allow_missing_sources:
            generation = validate_manifest_generation(
                manifest_dir,
                required_tags=self.pools,
            )
            self._validated_pool_entries = {
                tag: list(generation["_validated_entries"][tag]) for tag in self.pools
            }
            self.manifest_build_id = str(generation["build_id"])
        else:
            self.manifest_build_id = None
        self._pool_objs: dict[str, NoisePool] = {}
        self.dc_hum_prob = float(data_cfg.get("dc_hum_prob", 0.0))

        # digital-ref 1차경로 P(z). 실측/secondary-surrogate는 compact FIR과
        # D_noise 총지연을 resolver가 분리해 반환하므로 둘을 각각 정확히 한 번 적용한다.
        # legacy rir_surrogate만 p_err 안의 음향 onset을 고려해 추가지연을 계산한다.
        sp = load_secondary_path(_resolve_path(duct_cfg["secondary_path"]["npz"]))
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
            self.timing_contract = None
        else:
            self.d_noise_delay = int(self.digital_primary_path.delay_samples)
            plant_delays = PlantDelays.from_config(
                duct_cfg=duct_cfg,
                secondary_delay_samples=int(sp.delay_samples),
                primary_delay_samples=int(self.digital_primary_path.delay_samples),
                sample_rate=self.fs,
            )
            self.timing_contract = TrainingTimingContract.derive(
                primary_fir=self.digital_primary_path.fir,
                plant_delays=plant_delays,
            )
            derived_lead = int(self.timing_contract.digital_reference_lead_samples)
            if self.digital_reference_lead is None:
                # Compact P/S와 handoff가 유일한 출처다. config가 숫자를 생략한
                # canonical 형태에서는 여기서 계약값을 주입한다.
                self.digital_reference_lead = derived_lead
            elif derived_lead != self.digital_reference_lead:
                raise ValueError(
                    "digital_reference_lead_samples가 SynthANCDataset의 "
                    "PlantDelays.lead()와 다릅니다: "
                    f"configured={self.digital_reference_lead}, "
                    f"derived={self.timing_contract.digital_reference_lead_samples}"
                )
            declared = data_cfg.get("training_timing_contract")
            if declared is not None:
                configured = TrainingTimingContract.model_validate(declared)
                if configured != self.timing_contract:
                    raise ValueError(
                        "resolved training_timing_contract와 SynthANCDataset이 실제 P(z)에서 "
                        "유도한 계약이 다릅니다"
                    )
        if self.digital_reference_lead is None:
            self.digital_reference_lead = 0

        self.level_range = tuple(data_cfg.get("level_dbfs", [-35, -10]))
        self.snr_range = tuple(data_cfg.get("snr_mic_noise_db", [5, 30]))
        fb = data_cfg.get("closed_loop", {}).get("feedback_delay_samples", [512, 1024])
        self.feedback_delay_range = (int(fb[0]), int(fb[1]))

    # ---------- 내부 ----------

    def _pool(self, tag: str, rng: np.random.Generator) -> NoisePool | None:
        if tag not in self.pools:
            return None
        if tag not in self._pool_objs:
            try:
                self._pool_objs[tag] = NoisePool(
                    self.pools[tag],
                    self.split,
                    self.fs,
                    seed=int(rng.integers(1 << 31)),
                    validated_entries=self._validated_pool_entries.get(tag),
                )
            except (FileNotFoundError, ValueError) as exc:
                # 예전에는 여기서 한 줄 출력하고 태그를 버린 뒤 합성원으로 대체했다.
                # 이 코드는 **DataLoader 워커 안에서** 돈다 — 출력은 아무도 안 보는
                # 로그로 가고, 그때는 이미 학습이 시작된 뒤다. 실제로 선언 비중 0.70
                # (speech·music·machine·demand·dns_fullband)이 통째로 사라진 채
                # 100k step 을 돌 뻔했다. 구성 시점 검사(assert_declared_sources_exist)가
                # 정상 경로이고, 여기까지 왔다면 그 검사를 우회한 것이므로 멈춘다.
                raise RuntimeError(
                    f"소스 태그 '{tag}' 의 manifest 를 열 수 없습니다: "
                    f"{self.pools[tag]} — {type(exc).__name__}: {exc}\n"
                    "선언한 source_mix_ratio 와 다른 데이터로 학습하는 것을 막기 위해 "
                    "중단합니다. scripts/data/prepare_noise_pool.py 로 manifest 를 "
                    "만들거나 source_mix_ratio 에서 그 태그를 지우세요."
                ) from exc
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
                seg = pool.sample_segment(n_samples, rng=rng)
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
        return {
            "x": torch.from_numpy(x),
            "d": torch.from_numpy(d.astype(np.float32)).unsqueeze(0),  # [1, T]
        }

    # ---------- IterableDataset ----------

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        split_offset = {"train": 0, "val": 7919, "test": 15859}[self.split]
        indices = worker_global_item_indices(
            start_batch_index=self.resume_batch_index,
            batch_size=self.training_batch_size,
            worker_id=worker_id,
            num_workers=num_workers,
        )
        for global_index in indices:
            rng = indexed_rng(self.seed, split_offset, global_index)
            synth = SyntheticNoise(self.fs, seed=int(rng.integers(1 << 31)))
            yield self._make_item(rng, synth)


def make_eval_batch(
    dataset: SynthANCDataset, n_items: int, seed: int = 12345
) -> dict[str, torch.Tensor]:
    """고정 시드 검증 배치 — 학습 중 val NMSE 추적용 (매번 동일 데이터)."""
    rng = np.random.default_rng(seed)
    synth = SyntheticNoise(dataset.fs, seed=seed)
    items = [dataset._make_item(rng, synth) for _ in range(n_items)]
    return {
        "x": torch.stack([it["x"] for it in items]),
        "d": torch.stack([it["d"] for it in items]),
    }
