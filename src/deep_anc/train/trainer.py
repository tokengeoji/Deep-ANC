"""학습 루프 — Stage-1 open-loop / Stage-2 closed-loop, 단일 GPU·DDP(torchrun) 겸용.

실행 (Elice A100):
  단일:  python scripts/train/train.py --config configs/train_pretrain.yaml
  2-GPU: torchrun --nproc_per_node=2 scripts/train/train.py --config configs/train_pretrain.yaml
MIG(1g-10GB) 디버깅은 batch_size 를 4 로 낮추면 동일 코드로 동작한다.
"""

from __future__ import annotations

import math
import os
import time
import copy
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ..config import REPO_ROOT
from ..data.recorded_dataset import RecordedANCDataset
from ..data.synth_dataset import SynthANCDataset, make_eval_batch
from ..dsp.nonlinear import RandomNonlinear
from ..dsp.secondary_path import DifferentiableSecondaryPath, load_secondary_path
# 지연·대역 부기의 단일 출처. trainer 가 handoff/대역을 스스로 유도하면 게이트와
# 갈라진다 — 실제로 lead 가 109 와 113 으로 갈라졌다 (발생기 A).
from ..dsp.timing import BandPlan, PlantSettle, handoff_samples_from_config
from ..losses import ANCLoss
from ..models import build_model
from .checkpoint import load_checkpoint, save_checkpoint
from .reproducibility import set_seed, snapshot_run


def _ddp_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world, local_rank


def validate_training_physics(cfg: dict) -> str:
    """학습 단계가 요구하는 P(z) 물리 수준을 검사하고 상태 라벨을 반환."""
    data_cfg = cfg.get("data", {})
    reference = str(data_cfg.get("reference_mode", "digital"))
    primary_mode = str(data_cfg.get("digital_primary_path_mode", "rir_surrogate"))
    if (
        bool(cfg.get("require_measured_primary_path", False))
        and reference == "digital"
        and primary_mode != "measured"
    ):
        raise ValueError(
            "이 학습 설정은 실측 P(z)가 필수입니다. "
            "data.digital_primary_path_mode=measured와 "
            "duct.digital_reference.primary_path_npz를 지정하세요."
        )
    if reference != "digital":
        return "acoustic_rir_training"
    if primary_mode == "measured":
        return "measured_primary_path"
    return f"{primary_mode}_representation_pretrain"


BEST_METRIC_KEY = "nmse_trusted_cvar_db"
"""``best.pt`` 선택 기준이 **무엇을 잰 값인지**.

2026-08-05 이전에는 trusted **평균**이었다. 평가 게이트 G4 는 그때도 family 별
``worst10_mean_db < 0`` (= CVaR@10%)을 요구하고 있었다 — 평가는 최악값을 보는데
학습과 checkpoint 선택만 평균을 봤고, 그 불일치가 "평균은 좋은데 게이트는 FAIL"
(결함 4)을 만들었다. 이제 셋이 같은 종류의 양을 본다.

이 라벨을 checkpoint 에 박아 두지 않으면 서로 다른 지표끼리 ``min`` 비교하게 되고,
CVaR 은 평균보다 **항상 크므로** 옛 best 가 영원히 이겨 best.pt 가 갱신되지 않는다.
"""

_LEGACY_BEST_METRIC_KEY = "nmse_trusted_db"


def checkpoint_best_metric_key(state: dict) -> str:
    """checkpoint 의 best_metric 이 어떤 지표인지. 표식이 없으면 legacy(평균)."""

    saved_cfg = state.get("cfg", {}) or {}
    return str(saved_cfg.get("best_metric_key", _LEGACY_BEST_METRIC_KEY))


def checkpoint_best_metric_is_comparable(state: dict) -> bool:
    """지금 기준과 같은 양을 잰 best_metric 인가."""

    return checkpoint_best_metric_key(state) == BEST_METRIC_KEY


def checkpoint_training_lead(state: dict) -> int:
    """체크포인트 학습 lead를 읽는다. 메타 없는 legacy artifact는 0."""
    saved_cfg = state.get("cfg", {}) or {}
    if "digital_reference_lead_samples" in saved_cfg:
        return int(saved_cfg["digital_reference_lead_samples"])
    return int((saved_cfg.get("data", {}) or {}).get("digital_reference_lead_samples", 0))


def validate_resume_physics(state: dict, cfg: dict) -> None:
    """재개 체크포인트와 현재 학습의 지연/물리 모드가 같은지 검사한다.

    ``init_ckpt``는 surrogate 사전학습에서 measured 파인튜닝으로 가중치만
    옮기는 용도라 물리 모드가 달라도 된다. 반면 ``resume``은 옵티마이저와
    스케줄러 상태까지 이어받으므로 동일 실험이어야 한다.
    """
    expected_lead = int(cfg.get("data", {}).get("digital_reference_lead_samples", 0))
    saved_lead = checkpoint_training_lead(state)
    if saved_lead != expected_lead:
        raise ValueError(
            "resume checkpoint digital-reference lead 불일치: "
            f"checkpoint={saved_lead}, training={expected_lead}"
        )

    saved_cfg = state.get("cfg", {}) or {}
    expected_status = validate_training_physics(cfg)
    saved_status = saved_cfg.get("physics_status")
    if saved_status is None and "data" in saved_cfg:
        saved_status = validate_training_physics(saved_cfg)
    if saved_status is None:
        raise ValueError(
            "resume checkpoint에 physics_status/resolved data 설정이 없습니다. "
            "물리 정합을 검증할 수 없는 legacy artifact는 가중치 초기화에만 사용하세요."
        )
    if str(saved_status) != expected_status:
        raise ValueError(
            "resume checkpoint physics mode 불일치: "
            f"checkpoint={saved_status}, training={expected_status}"
        )


def resolve_run_until_step(cfg: dict, schedule_total_steps: int) -> int:
    """스케줄은 유지한 채 이번 프로세스가 멈출 step을 검증한다.

    구조 탐색은 100k cosine 스케줄의 동일한 초반 20k 곡선을 비교해야 한다.
    ``schedule.total_steps``를 20k로 줄이면 LR 궤적 자체가 달라지므로 별도
    ``run_until_step``에서 안전하게 구간 checkpoint를 만든다.
    """
    total = int(schedule_total_steps)
    stop = int(cfg.get("run_until_step", total))
    if total < 1:
        raise ValueError("schedule.total_steps는 1 이상이어야 합니다")
    if not 1 <= stop <= total:
        raise ValueError(
            f"run_until_step은 1 이상 schedule.total_steps({total}) 이하여야 합니다: "
            f"{stop}"
        )
    return stop


class MixedIterator:
    """합성/실측 데이터셋 혼합 (recorded_ratio 확률로 실측 배치 샘플)."""

    def __init__(self, synth_iter, recorded_iter, recorded_ratio: float, seed: int) -> None:
        import numpy as np

        self.synth = synth_iter
        self.recorded = recorded_iter
        self.ratio = float(recorded_ratio)
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self):
        if self.recorded is not None and self.rng.random() < self.ratio:
            return next(self.recorded)
        return next(self.synth)


class Trainer:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.physics_status = validate_training_physics(cfg)
        self.rank, self.world, self.local_rank = _ddp_env()
        self.is_main = self.rank == 0
        if self.world > 1 and not dist.is_initialized():
            dist.init_process_group("nccl")
            torch.cuda.set_device(self.local_rank)
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )

        seed = int(cfg.get("seed", 0)) + self.rank
        set_seed(seed)

        self.stage = str(cfg.get("stage", "open_loop"))
        if self.stage == "closed_loop" and self.world > 1:
            # 폐루프는 언랩된 모듈의 streaming_step 을 직접 호출하므로 DDP 그래디언트
            # 동기화가 동작하지 않는다 (리뷰 확정 결함 #3). 20~50k step 단기 학습이므로
            # 단일 GPU 로 실행할 것.
            raise RuntimeError(
                "closed_loop 스테이지는 단일 GPU 전용입니다 — torchrun 없이 실행하세요."
            )
        self.fs = int(cfg["data"]["sample_rate"])
        self.run_dir = Path(cfg["ckpt_dir"])
        if not self.run_dir.is_absolute():
            self.run_dir = REPO_ROOT / self.run_dir

        # ----- 모델 -----
        self.model = build_model(cfg["model"]).to(self.device)
        if self.world > 1:
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank])

        # ----- 플랜트 + 손실 -----
        duct = cfg["duct"]
        sp = load_secondary_path(REPO_ROOT / duct["secondary_path"]["npz"])
        if sp.sample_rate != self.fs:
            raise ValueError(
                f"S(z) npz 샘플레이트 {sp.sample_rate} ≠ 데이터 {self.fs} — "
                "duct.yaml secondary_path.npz 를 확인하세요 (감사 M7)"
            )
        pp = cfg["data"].get("plant_perturbation", {})
        plant = DifferentiableSecondaryPath(
            sp,
            handoff_extra_samples=handoff_samples_from_config(duct),
            delay_jitter_range=tuple(pp.get("delay_jitter_range", [0, 0])),
            gain_db_range=tuple(pp.get("gain_db", [0.0, 0.0])),
            tilt_db_per_octave_range=tuple(pp.get("gain_tilt_db_per_octave", [0.0, 0.0])),
            allpass_perturb=bool(pp.get("allpass_perturb", False)),
            seed=seed + 17,
        ).to(self.device)
        nl_cfg = cfg["data"].get("nonlinear", {})
        nonlinear = RandomNonlinear(
            nl_cfg.get("sef_eta_choices", [10.0]),
            tuple(nl_cfg.get("drive_range", [1.0, 1.0])),
            hardclip_prob=float(nl_cfg.get("hardclip_prob", 0.0)),
            seed=seed + 29,
        )
        acoustics = duct.get("acoustics", {})
        cutoff = float(acoustics.get("plane_wave_cutoff_hz", 1633.0))
        # 대역은 BandPlan 이 유일하게 유도한다. 예전에는 이 세 줄이 trainer /
        # eval.recorded / evaluate_offline / evaluate_session / render_anc_demo 에
        # **다섯 벌** 복붙돼 있었고, S npz 의 신뢰대역을 넓혀도 따라오지 않는 곳이 생겼다.
        self.band_plan = BandPlan.resolve(
            plant_trusted_band_hz=sp.trusted_band_hz(),
            duct_cfg=duct,
            sample_rate=self.fs,
        )
        target_band = self.band_plan.target.as_tuple()
        self.trusted_band_hz = self.band_plan.optimize.as_tuple()
        if self.is_main:
            print(
                "[trainer] NMSE trusted band: "
                f"{self.trusted_band_hz[0]:.0f}–{self.trusted_band_hz[1]:.0f}Hz "
                f"(S 신뢰 {sp.trusted_band_hz()[0]:.0f}–{sp.trusted_band_hz()[1]:.0f}Hz "
                f"∩ 목표 {target_band[0]:.0f}–{target_band[1]:.0f}Hz)"
            )
            print(f"[trainer] physics status: {self.physics_status}")
            if self.physics_status.endswith("_representation_pretrain"):
                print(
                    "[trainer 경고] surrogate P(z) 표현 사전학습입니다 — "
                    "실측 덕트 감쇠 성능으로 해석하지 마세요."
                )
        # 리미터 한계는 **모델이 단일 출처**다. 예전에는 loss.clip_margin 과
        # model.limiter.limit 이 서로 다른 두 곳에 적힌 같은 물리량이었고, trainer 가
        # 부등식 하나로 대조하는 것이 전부였다 (감사 L8). 이제 손실이 모델 값을
        # 그대로 받는다 — 두 번째 유도가 존재하지 않는다.
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.criterion = ANCLoss(
            plant, cfg["loss"], self.fs, nonlinear=nonlinear,
            cutoff_hz=cutoff, target_band_hz=target_band,
            trusted_band_hz=self.trusted_band_hz,
            limiter_limit=float(raw_model.limit),
        ).to(self.device)

        # S(z) 총지연 + FIR 정착 구간은 y 가 무엇이든 e = d 로 고정된다.
        # 합성 d 는 P(z) 지연 때문에 그 구간이 비어 있어 공짜지만(trusted 하한 −59.8 dB),
        # 실측 d 는 실신호가 들어 있어 노출된다(하한 mean −20.3 / CVaR10 −10.1 / worst
        # −4.8 dB). 평균 집계에서는 0.03 dB 로 작아 보였지만 CVaR 로 바꾸는 순간
        # 달성 불가능한 목표에 그래디언트가 집중된다 — 두 변경은 반드시 같이 간다.
        # 값의 단일 출처는 PlantSettle 이고, 평가기 warmup 도 같은 값을 하한으로 쓴다.
        self.plant_settle = PlantSettle.derive(
            secondary_delay_samples=int(sp.delay_samples),
            handoff_samples=handoff_samples_from_config(duct),
            fir_taps=int(sp.fir.size),
            sample_rate=self.fs,
        )
        self.loss_start_sample = int(
            cfg.get("loss_start_sample", self.plant_settle.samples)
        )
        if self.loss_start_sample < 0:
            raise ValueError("loss_start_sample 은 0 이상이어야 합니다")
        if self.is_main:
            print(f"[trainer] loss_start_sample {self.plant_settle.describe()}")
            if self.loss_start_sample != self.plant_settle.samples:
                print(
                    f"[trainer 경고] loss_start_sample 이 설정으로 "
                    f"{self.loss_start_sample} 로 덮였습니다 — 정착 구간 "
                    f"{self.plant_settle.samples} 와 다릅니다"
                )

        # ----- 데이터 -----
        # 비-synthetic 태그의 manifest 존재를 명시 검사 — 조용한 합성 폴백으로
        # 학습 분포가 바뀌는 사고 방지 (감사 M1). 부재 태그는 배너로 알린다.
        if self.is_main:
            data_cfg = cfg["data"]
            use_acoustic_mix = (
                data_cfg.get("reference_mode") == "acoustic"
                and data_cfg.get("source_mix_ratio_acoustic")
            )
            mix = data_cfg.get(
                "source_mix_ratio_acoustic" if use_acoustic_mix else "source_mix_ratio",
                {},
            )
            mdir = Path(cfg["data"].get("noise_manifest_dir", "data/manifests"))
            if not mdir.is_absolute():
                mdir = REPO_ROOT / mdir
            missing = [
                t for t, r in mix.items()
                if t != "synthetic" and float(r) > 0 and not (mdir / f"{t}.jsonl").exists()
            ]
            if missing:
                total_missing = sum(float(mix[t]) for t in missing)
                print("=" * 70)
                print(f"[trainer 경고] manifest 부재 태그 {missing} (비율 합 {total_missing:.0%})")
                print("  → 해당 비율은 합성원으로 대체됩니다. 의도가 아니면 학습을 중단하고")
                print("    scripts/data/prepare_noise_pool.py 를 실행하세요.")
                print("=" * 70)

        synth_train = SynthANCDataset(cfg["data"], duct, split="train", seed=seed)
        loader = DataLoader(
            synth_train,
            batch_size=int(cfg["batch_size"]),
            num_workers=int(cfg.get("num_workers", 4)),
            prefetch_factor=int(cfg.get("prefetch_factor", 2)) if cfg.get("num_workers", 4) else None,
            pin_memory=self.device.type == "cuda",
            persistent_workers=bool(cfg.get("num_workers", 4)),
        )
        self.train_iter = iter(loader)

        recorded_manifest = cfg.get("recorded_manifest")
        recorded_required = bool(cfg.get("require_recorded_manifest", False))
        if recorded_required and not recorded_manifest:
            raise ValueError(
                "이 학습 설정은 recorded_manifest가 필수입니다. "
                "실측 train/val/test 매니페스트를 지정하세요."
            )
        if recorded_manifest and not Path(recorded_manifest).is_absolute():
            recorded_manifest = str(REPO_ROOT / recorded_manifest)
        if recorded_required and not Path(recorded_manifest).exists():
            raise FileNotFoundError(
                "이 학습 설정은 유효한 recorded_manifest가 필수입니다: "
                f"{recorded_manifest}"
            )
        if recorded_manifest and Path(recorded_manifest).exists():
            rec = RecordedANCDataset(recorded_manifest, cfg["data"], split="train", seed=seed + 5)
            rec_loader = DataLoader(
                rec, batch_size=int(cfg["batch_size"]), num_workers=2,
                pin_memory=self.device.type == "cuda",
            )
            self.train_iter = MixedIterator(
                self.train_iter, iter(rec_loader), float(cfg.get("recorded_ratio", 0.5)), seed
            )
        elif recorded_manifest and self.is_main:
            print(f"[trainer] recorded_manifest({recorded_manifest}) 없음 — 합성 데이터만 사용")

        val_ds = SynthANCDataset(cfg["data"], duct, split="val", seed=1234)
        # CVaR 목적함수는 분위 추정이다. 16개면 q=0.25 가 top-4 뿐이라 추정이 흔들린다.
        # val 배치는 학습 batch_size 와 무관하게 키울 수 있다(그래디언트가 없다).
        self.val_items = int(cfg.get("val_items", 64))
        if self.val_items < 1:
            raise ValueError(f"val_items 는 1 이상이어야 합니다: {self.val_items}")
        self.val_batch = make_eval_batch(val_ds, n_items=self.val_items)

        # ----- 옵티마이저/스케줄 -----
        opt_cfg = cfg["optimizer"]
        opt_name = str(opt_cfg.get("name", "adamw")).lower()
        if opt_name != "adamw":
            raise ValueError(f"지원하지 않는 optimizer.name: {opt_name} (adamw 만 구현됨)")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(opt_cfg["lr"]),
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )
        sch = cfg["schedule"]
        warmup = int(sch.get("warmup_steps", 0))
        total = int(sch["total_steps"])
        min_ratio = float(sch.get("min_lr", 1e-5)) / float(opt_cfg["lr"])

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.total_steps = total
        self.run_until_step = resolve_run_until_step(cfg, total)
        self.grad_clip = float(cfg.get("grad_clip", 0.0))
        self.amp_dtype = torch.bfloat16 if cfg.get("amp") == "bf16" else None

        if bool(cfg.get("freeze_encoder", False)):
            raw = self.model.module if hasattr(self.model, "module") else self.model
            for p in raw.encoder.parameters():
                p.requires_grad_(False)

        # ----- 상태 -----
        self.step = 0
        self.best_metric = float("inf")
        self.writer = None
        if self.is_main:
            # resolved model/data/duct까지 전부 남겨야 surrogate/실측 P(z), lead,
            # plant curriculum을 나중에 정확히 구분할 수 있다. 비밀정보는 학습
            # config에 두지 않는 저장소 규약을 따른다.
            snapshot_run(self.run_dir, self._cfg_snapshot(cfg))
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(str(self.run_dir / "tb"))
            except ImportError:
                print("[trainer] tensorboard 미설치 — 파일 로그만 기록합니다")
        self.loss_log = open(self.run_dir / "loss_log.txt", "a", encoding="utf-8") if self.is_main else None

        # init_ckpt(파인튜닝) → resume(재개) 순서로 적용 (상대경로는 저장소 루트 기준)
        def _abs(p):
            if p is None:
                return None
            return str(p) if Path(p).is_absolute() else str(REPO_ROOT / p)

        init_ckpt = _abs(cfg.get("init_ckpt"))
        init_required = bool(cfg.get("require_init_checkpoint", False))
        if init_required and (not init_ckpt or not Path(init_ckpt).exists()):
            raise FileNotFoundError(
                f"이 학습 설정은 유효한 init_ckpt가 필수입니다: {init_ckpt}"
            )
        if cfg.get("init_ckpt") and not Path(init_ckpt).exists() and self.is_main:
            print(f"[trainer] 경고: init_ckpt({init_ckpt})가 없어 무시합니다")
        if init_ckpt and Path(init_ckpt).exists():
            state = load_checkpoint(init_ckpt, self.model, restore_rng=False, map_location="cpu")
            expected_lead = int(cfg["data"].get("digital_reference_lead_samples", 0))
            saved_lead = checkpoint_training_lead(state)
            # 허용 오차는 readiness 게이트와 **같은 설정값**을 읽는다. 같은 규칙을 두 곳에
            # 따로 구현하면 반드시 갈라진다 — 실제로 readiness 만 고쳤다가 여기서 막혔다.
            # 근거와 유계성은 finetune_readiness.audit_init_checkpoint 주석에 있다.
            tolerance = int(
                (cfg.get("readiness", {}) or {}).get(
                    "max_init_lead_mismatch_samples", 0
                )
            )
            if abs(saved_lead - expected_lead) > tolerance:
                raise ValueError(
                    "init checkpoint digital-reference lead 불일치: "
                    f"checkpoint={saved_lead}, training={expected_lead}, "
                    f"차이 {abs(saved_lead - expected_lead)} > 허용 {tolerance} samples"
                )
            if saved_lead != expected_lead and self.is_main:
                print(
                    f"[trainer] init lead {saved_lead} → 학습 {expected_lead} "
                    f"(차이 {abs(saved_lead - expected_lead)}, 허용 {tolerance}) — "
                    "surrogate 물리로 학습한 잠정값을 실측값으로 옮긴다"
                )
            if self.is_main:
                print(f"[trainer] init_ckpt 로드: {init_ckpt} (step {state.get('step')})")
        resume = _abs(cfg.get("resume"))
        if resume and Path(resume).exists():
            state = load_checkpoint(resume, self.model, self.optimizer, self.scheduler, map_location="cpu")
            validate_resume_physics(state, cfg)
            self.step = int(state.get("step", 0))
            self.best_metric = float(state.get("best_metric", float("inf")))
            if not checkpoint_best_metric_is_comparable(state):
                # 다른 지표끼리 min 비교하면 조용히 잘못된 best 가 고정된다. 2026-08-05
                # 이전 checkpoint 의 best 는 trusted **평균**(예: −19.54)이고 지금은
                # CVaR 이라 항상 평균이 이긴다 → best.pt 가 영원히 갱신되지 않는다.
                if self.is_main:
                    print(
                        "[trainer 경고] resume checkpoint 의 best_metric 이 "
                        f"'{checkpoint_best_metric_key(state)}' 기준입니다 — 현재 기준"
                        f" '{BEST_METRIC_KEY}' 과 비교할 수 없어 best_metric 을 "
                        "초기화합니다"
                    )
                self.best_metric = float("inf")
            else:
                # 구버전 last.pt는 eval에서 best 갱신 직전에 저장되어 best_metric이
                # 한 회 늦을 수 있다. 같은 run의 best.pt가 있으면 더 좋은 값을
                # authority로 사용해 재개 직후 나쁜 val이 best.pt를 덮어쓰지 못하게 한다.
                best_path = self.run_dir / "ckpt" / "best.pt"
                if best_path.exists():
                    try:
                        best_state = torch.load(
                            best_path, map_location="cpu", weights_only=False
                        )
                        if checkpoint_best_metric_is_comparable(best_state):
                            self.best_metric = min(
                                self.best_metric,
                                float(best_state.get("best_metric", float("inf"))),
                            )
                    except (OSError, RuntimeError, ValueError, TypeError):
                        pass
            if self.is_main:
                print(f"[trainer] 재개: step {self.step}, best {self.best_metric:.3f}")

    def _cfg_snapshot(self, cfg: dict) -> dict:
        """이 실행이 쓴 목적함수/절단 규약까지 포함한 스냅샷."""

        return cfg_snapshot(
            cfg,
            self.trusted_band_hz,
            best_metric_key=BEST_METRIC_KEY,
            loss_start_sample=self.loss_start_sample,
            cvar_scope="rank_local" if self.world > 1 else "global",
        )

    # ---------- 스텝 ----------

    def _forward_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        x = batch["x"].to(self.device, non_blocking=True)
        d = batch["d"].to(self.device, non_blocking=True)
        if self.stage == "closed_loop":
            return self._closed_loop_forward(x, d)
        y = self.model(x)
        # open_loop 도 정착 구간을 버린다. 예전에는 closed_loop 만 skip 을 넘겼고
        # 실측 배치(open_loop)는 e[0:1721] = d[0:1721] 을 그대로 손실에 넣었다.
        return self.criterion(y, d, loss_start_sample=self.loss_start_sample)

    def _closed_loop_forward(self, x: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Stage-2: 프레임 순차 unroll — 시뮬레이션 e 를 err 입력으로 되먹임 [H1/H3].

        피드백 지연(fb_delay ≥ 512)이 그룹 크기(hop×unroll_group)보다 크므로
        각 그룹 시작 시점까지 계산된 e 프리픽스만으로 인과적 되먹임이 가능하다.
        """
        cl = self.cfg["data"].get("closed_loop", {})
        group = int(cl.get("unroll_group_frames", 4))
        fb_lo, fb_hi = (int(v) for v in cl.get("feedback_delay_samples", [512, 1024]))
        warmup_s = float(cl.get("warmup_seconds", 0.25))

        raw = self.model.module if hasattr(self.model, "module") else self.model
        hop = raw.hop
        chunk = hop * group
        T = x.shape[-1] - (x.shape[-1] % chunk)
        x, d = x[..., :T], d[..., :T]

        import numpy as np

        fb_delay = int(np.random.default_rng(self.step).integers(fb_lo, fb_hi + 1))
        fb_delay = max(fb_delay, chunk)            # 인과성 보장

        states = raw.init_states(x.shape[0], self.device)
        y_parts: list[torch.Tensor] = []
        e_hist = torch.zeros_like(d)

        # 되먹임 경로와 최종 손실이 동일한 플랜트 섭동·비선형을 쓰도록 한 번만 샘플링
        # (리뷰 확정 결함 #6 — 되먹임만 공칭 선형이면 배포 분포와 어긋난다)
        plant = self.criterion.plant
        perturb = plant.sample_perturbation() if self.criterion.training else {"jitter": 0}
        nl = self.criterion.nonlinear
        nl_params = nl.sample(x.shape[0]) if (self.criterion.training and nl is not None) else None

        for start in range(0, T, chunk):
            sl = slice(start, start + chunk)
            err_in = e_hist[..., max(0, start - fb_delay) : max(0, start - fb_delay) + chunk]
            if err_in.shape[-1] < chunk:
                err_in = torch.nn.functional.pad(err_in, (chunk - err_in.shape[-1], 0))
            x_blk = torch.cat([x[:, :1, sl], err_in], dim=1)
            y_blk, states = raw.streaming_step(x_blk, states)
            y_parts.append(y_blk.float())          # 플랜트 FFT 는 FP32 필요 (bf16 미지원)
            # e 프리픽스 갱신 — 프리픽스 전체 재컨볼브 O(T²/chunk)는 알려진 성능 한계
            # (fb_delay ≥ chunk 라 인과성은 보장). 최적화 시 스트리밍 FIR 상태로 대체 가능.
            y_so_far = torch.cat(y_parts, dim=-1)
            y_nl = nl.apply_torch(y_so_far, nl_params) if nl_params is not None else y_so_far
            s_y = plant(y_nl, perturb)
            e_hist[..., : y_so_far.shape[-1]] = d[..., : y_so_far.shape[-1]] + s_y

        y = torch.cat(y_parts, dim=-1)
        # 폐루프 워밍업과 플랜트 정착 중 **더 긴 쪽**을 버린다.
        skip = max(int(warmup_s * self.fs), self.loss_start_sample)
        # 절단은 손실 내부에서 플랜트 적용 "후"에 수행 (결함 #2/#5)
        return self.criterion(y, d, loss_start_sample=skip, perturb=perturb, nl_params=nl_params)

    def _validate_metrics(self) -> dict[str, float]:
        self.model.eval()
        self.criterion.eval()
        with torch.no_grad():
            x = self.val_batch["x"].to(self.device)
            d = self.val_batch["d"].to(self.device)
            raw = self.model.module if hasattr(self.model, "module") else self.model
            y = raw(x)
            _, metrics = self.criterion(
                y, d, loss_start_sample=self.loss_start_sample
            )
        self.model.train()
        self.criterion.train()
        return metrics

    def _validate(self) -> float:
        """기존 호출 호환용 단일 값 검증 API — trusted-band NMSE를 반환."""
        return self._validate_metrics()["nmse_trusted_db"]

    # ---------- 메인 루프 ----------

    def train(self) -> None:
        cfg = self.cfg
        eval_every = int(cfg.get("eval_every", 2000))
        log_every = int(cfg.get("log_every", 100))
        patience = int(cfg.get("early_stop_patience", 0))
        bad_evals = 0
        self.model.train()
        self.criterion.train()
        t0 = time.time()

        while self.step < self.run_until_step:
            batch = next(self.train_iter)
            self.optimizer.zero_grad(set_to_none=True)

            if self.amp_dtype is not None and self.device.type == "cuda":
                with torch.autocast("cuda", dtype=self.amp_dtype):
                    loss, metrics = self._forward_loss(batch)
            else:
                loss, metrics = self._forward_loss(batch)

            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            if self.is_main and self.step % log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                sps = log_every / max(1e-9, time.time() - t0)
                t0 = time.time()
                line = (
                    f"step {self.step:7d} | loss {metrics['loss']:8.3f} | "
                    f"nmse_t {metrics['nmse_trusted_db']:7.2f} dB | "
                    f"nmse_f {metrics['nmse_fullband_db']:7.2f} dB | "
                    f"lr {lr:.2e} | {sps:5.2f} it/s"
                )
                print(line, flush=True)
                if self.loss_log:
                    self.loss_log.write(line + "\n")
                    self.loss_log.flush()
                if self.writer:
                    for k, v in metrics.items():
                        self.writer.add_scalar(f"train/{k}", v, self.step)
                    self.writer.add_scalar("train/lr", lr, self.step)

            if self.step % eval_every == 0:
                val_metrics = self._validate_metrics()
                # best 선택도 최악값 기준. 평가 G4 가 family 별 worst10(=CVaR@10%)을
                # 보는데 선택만 평균을 보면, 게이트에서 떨어질 checkpoint 를 best 로
                # 저장하게 된다 (결함 4 재현 경로).
                val_mean = float(val_metrics["nmse_trusted_db"])
                val_nmse = float(
                    val_metrics.get("nmse_trusted_cvar_db", val_metrics["nmse_trusted_db"])
                )
                val_worst = float(val_metrics.get("nmse_trusted_worst_db", float("nan")))
                val_fullband_nmse = val_metrics["nmse_fullband_db"]
                stop_flag = torch.zeros(1, device=self.device)
                if self.is_main:
                    print(
                        f"[eval] step {self.step}: val trusted CVaR {val_nmse:+.2f} dB "
                        f"(mean {val_mean:+.2f}, worst {val_worst:+.2f}) | "
                        f"fullband {val_fullband_nmse:+.2f} dB | "
                        f"대역밖 최악 {val_metrics.get('dnh_worst_db', float('nan')):+.2f} dB",
                        flush=True,
                    )
                    if self.writer:
                        # 기존 대시보드의 val/nmse_db는 목적함수 alias로 유지.
                        self.writer.add_scalar("val/nmse_db", val_nmse, self.step)
                        self.writer.add_scalar("val/nmse_trusted_db", val_mean, self.step)
                        for key in (
                            "nmse_trusted_cvar_db",
                            "nmse_trusted_worst_db",
                            "dnh",
                            "dnh_worst_db",
                            "frame_worst_db",
                            "sat",
                            "sat_u_over_limit_max",
                        ):
                            if key in val_metrics:
                                self.writer.add_scalar(
                                    f"val/{key}", val_metrics[key], self.step
                                )
                        self.writer.add_scalar(
                            "val/nmse_fullband_db", val_fullband_nmse, self.step
                        )
                    is_best = val_nmse < self.best_metric
                    if is_best:
                        self.best_metric = val_nmse
                        bad_evals = 0
                    else:
                        bad_evals += 1

                    # last.pt에도 이번 eval까지 반영된 best_metric을 기록한다.
                    save_checkpoint(
                        self.run_dir / "ckpt" / "last.pt",
                        self.model, self.optimizer, self.scheduler,
                        self.step, self.best_metric,
                        self._cfg_snapshot(cfg),
                    )
                    if is_best:
                        save_checkpoint(
                            self.run_dir / "ckpt" / "best.pt",
                            self.model, self.optimizer, self.scheduler,
                            self.step, self.best_metric,
                            self._cfg_snapshot(cfg),
                        )
                        print(f"[eval] best 갱신 → {val_nmse:.2f} dB", flush=True)
                    else:
                        if patience and bad_evals >= patience:
                            print(f"[eval] {patience}회 연속 미개선 — 조기 종료", flush=True)
                            stop_flag.fill_(1.0)
                # 조기종료 결정을 전 랭크에 전파 — rank0 만 break 하면 나머지가 행업된다 (#7)
                if self.world > 1:
                    dist.broadcast(stop_flag, src=0)
                if float(stop_flag.item()) > 0:
                    break

        if self.is_main:
            save_checkpoint(
                self.run_dir / "ckpt" / "last.pt",
                self.model, self.optimizer, self.scheduler,
                self.step, self.best_metric,
                self._cfg_snapshot(cfg),
            )
            print(
                f"학습 구간 종료: step {self.step}/{self.total_steps}, "
                f"best trusted val NMSE {self.best_metric:.2f} dB"
            )
        if self.world > 1:
            dist.destroy_process_group()


def cfg_snapshot(
    cfg: dict,
    trusted_band_hz: tuple[float, float] | None = None,
    *,
    best_metric_key: str | None = None,
    loss_start_sample: int | None = None,
    cvar_scope: str | None = None,
) -> dict:
    """체크포인트/실행 폴더에 남길 완전한 resolved 설정.

    과거에는 모델과 몇 개 필드만 저장해 P/S 경로·증강·소스 분포를 복원할 수
    없었다. 학습 입력 설정 전체를 보존하고, 배포가 즉시 검사할 lead alias와
    물리 유효성 라벨을 최상위에도 명시한다.

    ``best_metric_key`` 는 ``best_metric`` 이 **무엇을 잰 값인지**를 박아 둔다.
    2026-08-05 이전 checkpoint 의 best 는 trusted 평균이고 이후는 CVaR 이다. 두
    지표를 min 비교하면 조용히 잘못된 best 가 고정되므로, 이 키가 다르면 재개 시
    best 를 리셋한다.
    """
    out = copy.deepcopy(cfg)
    data_cfg = out.get("data", {})
    lead = int(data_cfg.get("digital_reference_lead_samples", 0))
    out["digital_reference_lead_samples"] = lead
    out["physics_status"] = validate_training_physics(out)
    if trusted_band_hz is not None:
        out["trusted_band_hz"] = [float(v) for v in trusted_band_hz]
    if best_metric_key is not None:
        out["best_metric_key"] = str(best_metric_key)
    if loss_start_sample is not None:
        out["loss_start_sample"] = int(loss_start_sample)
    if cvar_scope is not None:
        # DDP 에서 topk 는 랭크 로컬이다. world>1 이면 각 랭크의 상위 q 합집합이
        # 글로벌 상위 q 를 덮으므로 실효 분위가 넓어진다 — 산출물에 남긴다.
        out["nmse_cvar_scope"] = str(cvar_scope)
    return out
