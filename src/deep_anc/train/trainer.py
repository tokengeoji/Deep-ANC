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

from ..config import (
    CANONICAL_DETERMINISM_POLICY,
    CANONICAL_LOSS_PILOT_STEPS,
    PRETRAIN_DERIVATIVE_STRICT_ROLES,
    REPO_ROOT,
    loss_selection_sha256,
    validate_canonical_training_policy,
)
from ..data.recorded_dataset import RecordedANCDataset, make_recorded_eval_batch
from ..data.recorded_level_calibration import (
    require_recorded_level_calibration_config,
)
from ..data.transfer_contract import bind_recorded_transfer_config
from ..data.manifest_contract import validate_manifest_generation
from ..data.recorded_demand_selection import (
    DEMAND_SELECTION_RECEIPT,
    require_demand_selection_excluded_from_manifest_generation,
)
from ..data.resumable_stream import indexed_rng
from ..data.synth_dataset import (
    BROADBAND_SYNTH_PRIMARY_GENERATOR_SCHEMA,
    SynthANCDataset,
    make_eval_batch,
)
# 지연·대역 부기의 단일 출처. trainer 가 handoff/대역을 스스로 유도하면 게이트와
# 갈라진다 — 실제로 lead 가 109 와 113 으로 갈라졌다 (발생기 A).
from ..dsp.timing import PlantSettle, handoff_samples_from_config
from ..models import build_model
from ..losses.broadband_loss import CausalFIRPath
from ..model_input import (
    resolve_stage1_model_input_contract,
    validate_stage1_ref_only_tensor,
)
from .checkpoint import (
    load_checkpoint,
    read_checkpoint_snapshot,
    save_checkpoint,
    validate_resume_checkpoint_preview,
)
from .completion_receipt import write_completion_receipt
from .completion_receipt import validate_completion_receipt
from .campaign_prerequisite import validate_canonical_pretrain_prerequisites
from .a100_pretrain_smoke import (
    A100_PRETRAIN_SMOKE_ROLE,
    validate_a100_pretrain_smoke_config,
    write_a100_pretrain_smoke_phase_telemetry,
)
from .experiment_contract import (
    CANONICAL_ROLES,
    require_exact_source_trust,
    stamp_experiment_contract,
    validate_embedded_experiment_contract,
    validate_resume_experiment,
)
from .reproducibility import set_seed, snapshot_run
from .criterion_factory import (
    BROADBAND_CRITERION_ROLE,
    admit_criterion_config,
    build_criterion_from_config,
)


# pilot/probe는 canonical init 자체는 아니지만, winner를 결정하는 raw evidence다.
# non-strict snapshot으로 같은 contract directory를 재실행해 best/last를 바꾸거나,
# CUDA 결정론 설정 없이 seed 차이를 선택 결과로 오인하면 안 된다.
STRICT_RUN_ROLES = (
    CANONICAL_ROLES
    | frozenset({A100_PRETRAIN_SMOKE_ROLE})
    | PRETRAIN_DERIVATIVE_STRICT_ROLES
)


def _ddp_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world, local_rank


def validate_training_world_size(cfg: dict, world_size: int) -> None:
    """공식 단일-GPU 계약과 아직 미지원인 DDP exact-resume을 차단한다."""

    world = int(world_size)
    if world < 1:
        raise ValueError(f"WORLD_SIZE는 1 이상이어야 합니다: {world}")
    required = cfg.get("required_world_size")
    if required is not None and world != int(required):
        raise RuntimeError(
            "학습 world-size 계약 불일치: "
            f"required={int(required)}, actual={world}"
        )
    if world > 1 and cfg.get("resume"):
        raise RuntimeError(
            "DDP exact-resume은 rank별 torch/python/numpy 및 plant/nonlinear RNG를 "
            "checkpoint에 저장하기 전까지 지원하지 않습니다. 단일 GPU로 재개하거나 "
            "rank별 상태 계약을 먼저 구현하세요."
        )


def configure_canonical_determinism(cfg: dict, *, cuda_available: bool) -> None:
    """공식 실행의 결정론 backend 정책을 모델/RNG 생성 전에 고정한다.

    CPU smoke는 코드 경계만 증명한다. CUDA/bf16 exact-resume은 실제 A100의
    environment/completion receipt와 별도 중단→재개 검증 산출물이 있어야 한다.
    """

    if str(cfg.get("experiment_role", "")) not in STRICT_RUN_ROLES:
        return
    policy = cfg.get("determinism_policy")
    if policy != CANONICAL_DETERMINISM_POLICY:
        raise ValueError("canonical determinism_policy가 승인 정책과 다릅니다")
    allowed = tuple(policy["cublas_workspace_config_allowed"])
    if cuda_available and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in allowed:
        raise RuntimeError(
            "canonical CUDA 학습은 CUBLAS_WORKSPACE_CONFIG가 "
            f"{list(allowed)} 중 하나여야 합니다"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _resume_preview_components(cfg: dict, *, model=None):
    """live 학습 객체와 분리된 CPU model/optimizer/scheduler를 만든다."""

    if model is None:
        model = build_model(cfg["model"])
    opt_cfg = cfg["optimizer"]
    if str(opt_cfg.get("name", "adamw")).lower() != "adamw":
        raise ValueError("resume preview는 adamw optimizer만 지원합니다")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
    )
    schedule = cfg["schedule"]
    warmup = int(schedule.get("warmup_steps", 0))
    total = int(schedule["total_steps"])
    min_ratio = float(schedule.get("min_lr", 1e-5)) / float(opt_cfg["lr"])

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return model, optimizer, scheduler


def preflight_canonical_resume(cfg: dict):
    """ProcessLock/run 파일 생성 전 canonical resume 전체를 immutable preview한다."""

    if str(cfg.get("experiment_role", "")) not in STRICT_RUN_ROLES or not cfg.get(
        "resume"
    ):
        return None
    run_dir = Path(str(cfg["ckpt_dir"]))
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    validate_canonical_run_entry(cfg, run_dir, cfg["resume"])
    resume_path = Path(str(cfg["resume"]))
    if not resume_path.is_absolute():
        resume_path = REPO_ROOT / resume_path
    state, snapshot = read_checkpoint_snapshot(resume_path, map_location="cpu")
    validate_resume_physics(state, cfg)
    # 모델 생성의 CPU RNG 소비도 호출자 상태에 남기지 않는다. CUDA context나
    # backend flag를 건드리지 않고 checkpoint 구조만 검증한다.
    with torch.random.fork_rng(devices=[]):
        model, optimizer, scheduler = _resume_preview_components(cfg)
        validate_resume_checkpoint_preview(
            resume_path, state, model, optimizer, scheduler
        )
    return state, snapshot, resume_path.absolute()


def validate_canonical_run_entry(
    cfg: dict, run_dir: str | Path, resume: str | Path | None
) -> None:
    """canonical run의 신규/정확 재개/영구 완료 경계를 산출물 생성 전에 검사."""

    role = str(cfg.get("experiment_role", ""))
    if role not in STRICT_RUN_ROLES:
        return
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("canonical 학습 data config가 없습니다")
    if (
        data_cfg.get("bootstrap_receipt")
        != "data/manifests/elice_bootstrap_receipt.json"
        or not data_cfg.get("bootstrap_receipt_sha256")
    ):
        raise ValueError(
            "canonical 학습은 외부 SHA로 고정한 Elice bootstrap receipt가 필요합니다"
        )
    bind_recorded_transfer_config(data_cfg, repo_root=REPO_ROOT)
    # old82와 strict P의 약 20--25 dB 물리 단위 차이를 묵인한 채 70:30 stream을
    # 시작하면 run directory가 생긴 뒤에야 오염을 발견한다. 외부 SHA가 결속된
    # train-only receipt를 여기서 먼저 검증한다.
    require_recorded_level_calibration_config(cfg, repo_root=REPO_ROOT)
    # DKITCHEN이 recorded source로 선택된 뒤 exclusion sidecar/live manifest
    # 재발행 전의 짧은 중간 상태에서 GPU pretrain을 시작하지 못하게 한다.
    # 이 검사는 run directory 생성보다 먼저 실행된다.
    mix = data_cfg.get("source_mix_ratio")
    if (REPO_ROOT / DEMAND_SELECTION_RECEIPT).is_file() and isinstance(mix, dict):
        required_tags = {
            str(tag)
            for tag, ratio in mix.items()
            if str(tag) != "synthetic" and float(ratio) > 0.0
        }
        manifest_dir = Path(str(data_cfg.get("noise_manifest_dir", "data/manifests")))
        if not manifest_dir.is_absolute():
            manifest_dir = REPO_ROOT / manifest_dir
        manifest_generation = validate_manifest_generation(
            manifest_dir,
            required_tags=required_tags,
            repo_root=REPO_ROOT,
        )
        require_demand_selection_excluded_from_manifest_generation(
            manifest_generation,
            repo_root=REPO_ROOT,
        )
    # binder가 resolved 값을 주입했다면 이미 stamp된 계약과 정확히 같아야 한다.
    # stamp 뒤 조용히 cfg를 바꾸는 경로는 전부 거부한다.
    validate_embedded_experiment_contract(cfg)
    validate_canonical_training_policy(cfg)
    if role == A100_PRETRAIN_SMOKE_ROLE:
        validate_a100_pretrain_smoke_config(cfg, repo_root=REPO_ROOT)
    elif role in CANONICAL_ROLES:
        validate_canonical_pretrain_prerequisites(cfg, repo_root=REPO_ROOT)
    # loss_pilot/measured_probe는 canonical ledger를 만들기 전의 evidence라 ledger
    # 자체는 아직 요구하지 않는다. 위 공통 bootstrap/P/S/embedded-contract/source
    # 검증과 아래 no-overwrite/explicit-resume 경계는 canonical과 동일하게 적용된다.
    require_exact_source_trust(
        cfg, repo_root=REPO_ROOT, roles=STRICT_RUN_ROLES
    )
    directory = Path(run_dir).absolute()
    receipt = directory / "ckpt" / "completion.json"
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError(
            f"완료 receipt가 있는 canonical run은 재진입할 수 없습니다: {receipt}"
        )
    if resume is None:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                "canonical run은 resume 없이 기존 디렉터리를 덮어쓸 수 없습니다: "
                f"{directory}"
            )
        return
    requested = Path(resume)
    if not requested.is_absolute():
        requested = REPO_ROOT / requested
    requested = requested.absolute()
    # canonical/pretrain·finetune은 mutable last.pt만 exact resume input으로
    # 허용한다. 별도 init-ineligible smoke role만 runner가 hard-link로 보존한
    # immutable sibling stop.pt를 actual split input으로 쓴다.
    expected_name = "stop.pt" if role == A100_PRETRAIN_SMOKE_ROLE else "last.pt"
    expected = (directory / "ckpt" / expected_name).absolute()
    if requested != expected:
        rule = (
            "canonical resume은 해당 contract run의 exact last.pt만 허용합니다"
            if role != A100_PRETRAIN_SMOKE_ROLE
            else "A100 smoke resume은 해당 prerequisite run의 immutable stop.pt만 허용합니다"
        )
        raise ValueError(
            f"{rule}: "
            f"requested={requested}, expected={expected}"
        )


def resolve_validation_source(
    *, criterion_role: str, has_recorded_stream: bool, recorded_ratio: float
) -> str:
    """Stage-1 동작을 보존하면서 causal broadband fine-tune만 recorded val로 고정."""

    ratio = float(recorded_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("recorded_ratio는 [0,1]이어야 합니다")
    if criterion_role != BROADBAND_CRITERION_ROLE or ratio == 0.0:
        return "synthetic_val"
    if not bool(has_recorded_stream):
        raise FileNotFoundError(
            "causal broadband recorded_ratio>0 실행은 실제 recorded manifest와 "
            "recorded-val-only model selection이 필요합니다"
        )
    return "recorded_val_only"


def validate_init_checkpoint_role(state: dict, cfg: dict) -> None:
    """공식 fine-tune init은 완료 가능한 canonical pretrain만 허용한다."""

    if not bool(cfg.get("require_init_checkpoint", False)):
        return
    saved_cfg = state.get("cfg")
    if not isinstance(saved_cfg, dict):
        raise ValueError("init checkpoint에 resolved cfg가 없습니다")
    required_role = str(
        cfg.get("required_init_experiment_role", "canonical_pretrain")
    )
    saved_role = str(saved_cfg.get("experiment_role", ""))
    if saved_role != required_role:
        raise ValueError(
            "init checkpoint experiment_role이 승인된 canonical pretrain이 아닙니다: "
            f"checkpoint={saved_role!r}, required={required_role!r}"
        )
    if bool(cfg.get("require_init_eligible", True)) and saved_cfg.get(
        "init_eligible"
    ) is not True:
        raise ValueError(
            "init checkpoint가 init_eligible=true가 아닙니다 — loss pilot/measured "
            "probe checkpoint는 공식 fine-tune 초기값으로 사용할 수 없습니다"
        )
    current_role = str(cfg.get("experiment_role", ""))
    if current_role in {"canonical_finetune", "measured_probe"}:
        validate_canonical_training_policy(saved_cfg)
        saved_loss_sha = str(saved_cfg.get("loss_selection_sha256", ""))
        current_loss_sha = str(cfg.get("loss_selection_sha256", ""))
        if saved_loss_sha != loss_selection_sha256(saved_cfg.get("loss") or {}):
            raise ValueError("init checkpoint loss-selection digest가 embedded loss와 다릅니다")
        if current_loss_sha != loss_selection_sha256(cfg.get("loss") or {}):
            raise ValueError("fine-tune loss-selection digest가 resolved loss와 다릅니다")
        if saved_loss_sha != current_loss_sha:
            raise ValueError(
                "init checkpoint와 현재 학습의 loss selection이 다릅니다: "
                f"init={saved_loss_sha}, current={current_loss_sha}"
            )


def validate_measured_probe_init_chain(
    state: dict,
    cfg: dict,
    init_path: str | Path,
) -> None:
    """5k probe를 열기 전에 동일 alpha의 완료된 20k pilot을 확인한다.

    pilot은 100k schedule의 20k operational stop이라 completion receipt를 만들지
    않는다. 그렇다고 best.pt 하나만 보고 probe를 시작하면 아직 돌고 있는 pilot이나
    다른 alpha의 best에 A100 5k를 낭비할 수 있다. 같은 ckpt 디렉터리의 last.pt가
    정확히 20k에 도달했고 best/last 계약·loss가 같은지 read-only로 먼저 닫는다.
    """

    if str(cfg.get("experiment_role", "")) != "measured_probe":
        return
    checkpoint = Path(init_path)
    if checkpoint.name != "best.pt":
        raise ValueError("measured_probe init은 loss_pilot best.pt여야 합니다")
    saved_cfg = state.get("cfg")
    if not isinstance(saved_cfg, dict):
        raise ValueError("measured_probe init best.pt에 resolved cfg가 없습니다")
    # role과 같은 loss-selection 검사는 best checkpoint 자체에서 먼저 수행한다.
    validate_init_checkpoint_role(state, cfg)
    if int(saved_cfg.get("run_until_step", -1)) != CANONICAL_LOSS_PILOT_STEPS:
        raise ValueError("measured_probe init best.pt가 승인된 20k pilot 계약이 아닙니다")

    last_path = checkpoint.with_name("last.pt")
    if not last_path.is_file() or last_path.is_symlink():
        raise FileNotFoundError(
            "measured_probe 시작 전 loss_pilot의 완료된 sibling last.pt가 필요합니다: "
            f"{last_path}"
        )
    last_state, _ = read_checkpoint_snapshot(last_path, map_location="cpu")
    last_cfg = last_state.get("cfg")
    if not isinstance(last_cfg, dict):
        raise ValueError("loss_pilot last.pt에 resolved cfg가 없습니다")
    validate_canonical_training_policy(last_cfg)
    if str(last_cfg.get("experiment_role", "")) != "loss_pilot":
        raise ValueError("measured_probe sibling last.pt가 loss_pilot이 아닙니다")
    if (
        int(last_state.get("step", -1)) != CANONICAL_LOSS_PILOT_STEPS
        or int(last_cfg.get("run_until_step", -1))
        != CANONICAL_LOSS_PILOT_STEPS
    ):
        raise ValueError("measured_probe sibling loss_pilot last.pt가 정확히 20k 완료되지 않았습니다")
    if (
        saved_cfg.get("experiment_contract_sha256")
        != last_cfg.get("experiment_contract_sha256")
    ):
        raise ValueError("loss_pilot best.pt와 last.pt experiment contract가 다릅니다")
    best_loss_sha = str(saved_cfg.get("loss_selection_sha256", ""))
    last_loss_sha = str(last_cfg.get("loss_selection_sha256", ""))
    if last_loss_sha != loss_selection_sha256(last_cfg.get("loss") or {}):
        raise ValueError("loss_pilot last.pt loss-selection digest가 embedded loss와 다릅니다")
    if best_loss_sha != last_loss_sha:
        raise ValueError("loss_pilot best.pt와 last.pt loss selection이 다릅니다")


def validate_training_physics(cfg: dict) -> str:
    """학습 단계가 요구하는 P(z) 물리 수준을 검사하고 상태 라벨을 반환."""
    data_cfg = cfg.get("data", {})
    reference = str(data_cfg.get("reference_mode", "digital"))
    primary_mode = str(data_cfg.get("digital_primary_path_mode", "rir_surrogate"))
    causal_mode = (
        primary_mode == "causal_joint_v4"
        and isinstance(cfg.get("broadband_causal_training_authority"), dict)
    )
    if (
        bool(cfg.get("require_measured_primary_path", False))
        and reference == "digital"
        and primary_mode != "measured"
        and not causal_mode
    ):
        raise ValueError(
            "이 학습 설정은 실측 P(z)가 필수입니다. "
            "data.digital_primary_path_mode=measured와 "
            "duct.digital_reference.primary_path_npz를 지정하세요."
        )
    if reference != "digital":
        return "acoustic_rir_training"
    if causal_mode:
        return "fullband_causal_joint_fir_training"
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
BROADBAND_BEST_METRIC_KEY = "nmse_subband_guard_cvar_db"


def _require_finite_tensor(name: str, value: torch.Tensor) -> None:
    if value.numel() == 0 or not bool(torch.isfinite(value.detach()).all().item()):
        raise FloatingPointError(f"G0 finite 계약 위반: {name}에 NaN/Inf 또는 빈 tensor")


def _require_finite_named_tensors(
    named_values: list[tuple[str, torch.Tensor]],
) -> None:
    """정상 경로에서 device별 한 번만 host와 동기화해 finite를 검사한다.

    개별 tensor마다 ``.item()``을 호출하면 CUDA stream이 파라미터
    수만큼 매 step 동기화된다. 정상이면 device 하나당 집계 flag
    하나만 읽고, 집계가 실패했을 때만 기존 개별 검사를 다시 실행해
    정확한 tensor 이름을 보고한다. 빈 tensor도 같은 실패 경로로 보내
    기존 순서와 오류 메시지를 보존한다.
    """

    if not named_values:
        return

    by_device: dict[torch.device, list[torch.Tensor]] = {}
    has_empty = False
    for _, value in named_values:
        if value.numel() == 0:
            has_empty = True
            continue
        by_device.setdefault(value.device, []).append(value)

    aggregate_failed = has_empty
    for values in by_device.values():
        finite_flags = [torch.isfinite(value.detach()).all() for value in values]
        device_is_finite = torch.stack(finite_flags).all()
        if not bool(device_is_finite.item()):
            aggregate_failed = True

    if not aggregate_failed:
        return

    # 실패한 경우에만 이전과 같은 순서로 개별 검사해 이름을 특정한다.
    for name, value in named_values:
        _require_finite_tensor(name, value)
    raise FloatingPointError("G0 finite 계약 위반: 집계 finite 검사가 실패했습니다")


def validate_finite_batch(batch: dict, *, model_input_contract=None) -> None:
    """모든 tensor 학습 입력을 forward 전에 fail-closed 검사한다."""

    if not isinstance(batch, dict) or not batch:
        raise FloatingPointError("G0 finite 계약 위반: 학습 batch가 비었습니다")
    tensors = 0
    for name, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensors += 1
            if name == "x" and model_input_contract is not None:
                # REF-only 검사는 ERR exact-zero와 REF finite/nonzero를 이미 채널별
                # reduction으로 확인한다. x 전체 finite scan을 중복하면 canonical
                # A100 batch의 CPU 공급이 불필요하게 느려진다.
                validate_stage1_ref_only_tensor(
                    value,
                    model_input_contract,
                    label="canonical batch.x",
                )
                continue
            _require_finite_tensor(f"input.{name}", value)
    if tensors == 0:
        raise FloatingPointError("G0 finite 계약 위반: batch에 tensor 입력이 없습니다")


def validate_finite_output(output: torch.Tensor) -> None:
    _require_finite_tensor("model.output", output)


def validate_finite_loss_metrics(loss: torch.Tensor, metrics: dict) -> None:
    """loss와 모든 보고 지표(NMSE 포함)의 finite를 강제한다."""

    _require_finite_tensor("loss", loss)
    if not isinstance(metrics, dict) or not metrics:
        raise FloatingPointError("G0 finite 계약 위반: loss metrics가 비었습니다")
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor):
            _require_finite_tensor(f"metric.{name}", value)
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise FloatingPointError(
                f"G0 finite 계약 위반: metric.{name}을 수치로 검증할 수 없습니다"
            ) from exc
        if not math.isfinite(numeric):
            raise FloatingPointError(f"G0 finite 계약 위반: metric.{name}={numeric}")


def validate_finite_gradients(model: torch.nn.Module) -> None:
    """optimizer가 non-finite gradient를 상태에 반영하기 전에 차단한다."""

    _require_finite_named_tensors(
        [
            (f"gradient.{name}", parameter.grad)
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        ]
    )


def validate_finite_parameters(model: torch.nn.Module) -> None:
    """optimizer step이 생성한 non-finite 가중치도 즉시 차단한다."""

    _require_finite_named_tensors(
        [
            (f"parameter.{name}", parameter)
            for name, parameter in model.named_parameters()
        ]
    )


def validate_g0_nmse(
    metrics: dict,
    *,
    maximum_exclusive_db: float = -6.0,
) -> float:
    """G0 합격은 trusted NMSE가 -6 dB보다 **작을 때만**이다."""

    key = "nmse_trusted_db"
    if key not in metrics:
        raise FloatingPointError(f"G0 계약 위반: {key}가 없습니다")
    value = float(metrics[key])
    limit = float(maximum_exclusive_db)
    if not math.isfinite(value) or not math.isfinite(limit):
        raise FloatingPointError(f"G0 계약 위반: {key}/합격선이 non-finite입니다")
    if value >= limit:
        raise ValueError(
            f"G0 FAIL: {key}={value:.6f} dB >= 엄격 합격선 {limit:.6f} dB"
        )
    return value


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
    # lead/status 두 필드만 맞으면 구형 [150,600] loss가 현행 실험으로 재개되는 사고가
    # 난다. 전체 resolved config와 P/S/data artifact SHA를 먼저 대조한다.
    validate_resume_experiment(state, cfg, repo_root=REPO_ROOT)

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

    def __init__(
        self,
        synth_iter,
        recorded_iter,
        recorded_ratio: float,
        seed: int,
        *,
        start_batch_index: int = 0,
    ) -> None:
        self.synth = synth_iter
        self.recorded = recorded_iter
        self.ratio = float(recorded_ratio)
        self.seed = int(seed)
        self.batch_index = int(start_batch_index)
        if not 0.0 <= self.ratio <= 1.0 or self.batch_index < 0:
            raise ValueError("recorded_ratio/start_batch_index 계약 위반")

    def __iter__(self):
        return self

    def __next__(self):
        use_recorded = mixed_batch_is_recorded(
            self.seed, self.batch_index, self.ratio
        )
        self.batch_index += 1
        if self.recorded is not None and use_recorded:
            return next(self.recorded)
        return next(self.synth)


def mixed_batch_is_recorded(seed: int, batch_index: int, ratio: float) -> bool:
    """혼합 분기를 global batch index의 순수 함수로 만든다."""

    return bool(indexed_rng(seed, 0x4D4958, batch_index).random() < float(ratio))


def mixed_branch_counts(seed: int, batches: int, ratio: float) -> tuple[int, int]:
    """resume 지점 전 synth/recorded 소비 batch 수를 결정적으로 계산."""

    total = int(batches)
    if total < 0:
        raise ValueError("batches는 0 이상이어야 합니다")
    recorded = sum(
        mixed_batch_is_recorded(seed, index, ratio) for index in range(total)
    )
    return total - int(recorded), int(recorded)


class Trainer:
    def __init__(self, cfg: dict, *, resume_preflight=None) -> None:
        self.cfg = cfg
        # 광대역 schema/S-NPZ/contract SHA는 CUDA 조회, DDP 초기화, model/RNG 및
        # DataLoader worker보다 먼저 닫는다. strict-v1 NPZ에 광대역 이름만 붙인
        # 설정은 여기서 즉시 실패한다.
        self._criterion_admission = admit_criterion_config(
            cfg,
            repo_root=REPO_ROOT,
            require_bound=True,
        )
        expected_best_metric = (
            BROADBAND_BEST_METRIC_KEY
            if self._criterion_admission.role == BROADBAND_CRITERION_ROLE
            else BEST_METRIC_KEY
        )
        self.best_metric_key = str(cfg.get("best_metric_key", expected_best_metric))
        if self.best_metric_key != expected_best_metric:
            raise ValueError(
                "criterion role과 best_metric_key가 다릅니다: "
                f"role={self._criterion_admission.role}, "
                f"configured={self.best_metric_key}, required={expected_best_metric}"
            )
        self.physics_status = validate_training_physics(cfg)
        self.rank, self.world, self.local_rank = _ddp_env()
        validate_training_world_size(cfg, self.world)
        self.is_main = self.rank == 0
        if self.world > 1 and not dist.is_initialized():
            dist.init_process_group("nccl")
            torch.cuda.set_device(self.local_rank)
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )

        seed = int(cfg.get("seed", 0)) + self.rank

        self.stage = str(cfg.get("stage", "open_loop"))
        self.model_input_contract = resolve_stage1_model_input_contract(
            cfg.get("data") or {}
        )
        if self.model_input_contract is not None:
            if self.stage != "open_loop":
                raise RuntimeError(
                    "ref-only Stage-1 input contract는 open_loop 전용입니다 — "
                    "closed-loop는 ERR feedback을 다시 모델 feature로 만들기 때문입니다"
                )
            if (
                cfg.get("model_input_contract_sha256")
                != self.model_input_contract.digest()
            ):
                raise ValueError(
                    "Trainer model-input contract SHA가 resolved config와 다릅니다"
                )
        if (
            self._criterion_admission.causal_authority is not None
            and self.stage != "open_loop"
        ):
            raise RuntimeError(
                "v4 causal broadband authority는 현재 open_loop만 승인합니다; "
                "stateful closed-loop prefix 동치 검증 전입니다"
            )
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
        validate_canonical_run_entry(cfg, self.run_dir, cfg.get("resume"))

        def _abs_checkpoint(value):
            if value is None:
                return None
            return str(value) if Path(value).is_absolute() else str(REPO_ROOT / value)

        # 공식 init의 role/loss/receipt/bytes를 run 파일이나 RNG/model state를 만들기
        # 전에 모두 닫는다. 이후 weight load는 이 동일 snapshot state만 소비한다.
        self._init_state_preview: dict | None = None
        self._init_path_preview: Path | None = None
        init_ckpt_preview = _abs_checkpoint(cfg.get("init_ckpt"))
        init_required_preview = bool(cfg.get("require_init_checkpoint", False))
        if init_required_preview and (
            not init_ckpt_preview or not Path(init_ckpt_preview).is_file()
        ):
            raise FileNotFoundError(
                f"이 학습 설정은 유효한 init_ckpt가 필수입니다: {init_ckpt_preview}"
            )
        if init_ckpt_preview and Path(init_ckpt_preview).is_file():
            init_path = Path(init_ckpt_preview)
            receipt = None
            if init_required_preview:
                if init_path.name != "best.pt":
                    raise ValueError(
                        "필수 init checkpoint는 선택된 best.pt여야 합니다"
                    )
                if bool(cfg.get("require_init_completion_receipt", True)):
                    receipt = validate_completion_receipt(
                        init_path.parent,
                        expected_role=str(cfg.get("required_init_experiment_role")),
                        expected_init_eligible=bool(
                            cfg.get("require_init_eligible", True)
                        ),
                        repo_root=REPO_ROOT,
                    )
            init_state, init_snapshot = read_checkpoint_snapshot(
                init_path, map_location="cpu"
            )
            if receipt is not None and init_snapshot.sha256 != receipt.get(
                "best_checkpoint_sha256"
            ):
                raise ValueError("init checkpoint bytes가 completion receipt와 다릅니다")
            validate_init_checkpoint_role(init_state, cfg)
            if init_required_preview:
                validate_embedded_experiment_contract(init_state["cfg"])
                validate_canonical_training_policy(init_state["cfg"])
                validate_measured_probe_init_chain(init_state, cfg, init_path)
                if init_state["cfg"].get("model") != cfg.get("model"):
                    raise ValueError("init checkpoint model 설정이 fine-tune과 다릅니다")
                if init_state["cfg"].get("trusted_band_hz") != cfg.get(
                    "trusted_band_hz"
                ):
                    raise ValueError("init checkpoint trusted band가 fine-tune과 다릅니다")
            expected_lead = int(
                cfg["data"].get("digital_reference_lead_samples", 0)
            )
            saved_lead = checkpoint_training_lead(init_state)
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
            self._init_state_preview = init_state
            self._init_path_preview = init_path

        # DataLoader worker/prefetch RNG를 저장하는 대신 checkpoint의 절대
        # global batch index에서 모든 item을 다시 유도한다. 데이터셋을
        # 만들기 전에 resume step을 알아야 하므로 metadata만 먼저 읽는다.
        self.resume_batch_index = 0
        self.resume_checkpoint_sha256: str | None = None
        self._resume_state_preview: dict | None = None
        self._resume_path_preview: Path | None = None
        resume_value = cfg.get("resume")
        if resume_value:
            resume_path = Path(resume_value)
            if not resume_path.is_absolute():
                resume_path = REPO_ROOT / resume_path
            if not resume_path.is_file():
                raise FileNotFoundError(f"resume checkpoint가 없습니다: {resume_path}")
            if resume_preflight is not None:
                preview, resume_snapshot, preview_path = resume_preflight
                if Path(preview_path).absolute() != resume_path.absolute():
                    raise ValueError("resume preflight path가 resolved resume과 다릅니다")
            else:
                preview, resume_snapshot = read_checkpoint_snapshot(
                    resume_path, map_location="cpu"
                )
            data_stream = preview.get("data_stream")
            if not isinstance(data_stream, dict):
                raise ValueError(
                    "resume checkpoint에 data_stream global batch index가 없습니다"
                )
            saved_step = int(preview.get("step", -1))
            saved_batch = int(data_stream.get("global_batch_index", -1))
            if saved_step < 0 or saved_batch != saved_step:
                raise ValueError(
                    "resume checkpoint step/data_stream 불일치: "
                    f"step={saved_step}, global_batch_index={saved_batch}"
                )
            if not isinstance(preview.get("training_state"), dict):
                raise ValueError(
                    "resume checkpoint에 plant/nonlinear training_state가 없습니다"
                )
            self.resume_batch_index = saved_batch
            # 재개는 뒤에서 path를 다시 열지 않고 이 immutable preview state를 적용한다.
            # phase telemetry에 이 bytes SHA를 남겨 stop.pt와 직접 대조한다.
            self.resume_checkpoint_sha256 = str(resume_snapshot.sha256)
            self._resume_state_preview = preview
            self._resume_path_preview = resume_path
            # artifact/config 계약은 seed/model/DataLoader 등 live state를 만들기 전에 닫는다.
            validate_resume_physics(preview, cfg)

        # 모든 canonical artifact preview가 끝난 뒤에만 backend/global RNG를 바꾼다.
        configure_canonical_determinism(
            cfg, cuda_available=self.device.type == "cuda"
        )
        set_seed(seed)

        # ----- 모델 -----
        self.model = build_model(cfg["model"]).to(self.device)
        if self.world > 1:
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank])

        # 모델 shape와 optimizer/scheduler/RNG/training-state 전체 preview도 worker를
        # 띄우거나 run 산출물을 만들기 전에 끝낸다. 검사용 객체에는 상태를 적용하지 않는다.
        if self._resume_state_preview is not None:
            raw_preview_model = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            _, preview_optimizer, preview_scheduler = _resume_preview_components(
                cfg, model=raw_preview_model
            )
            assert self._resume_path_preview is not None
            validate_resume_checkpoint_preview(
                self._resume_path_preview,
                self._resume_state_preview,
                self.model,
                preview_optimizer,
                preview_scheduler,
            )
            del preview_scheduler, preview_optimizer

        # ----- 플랜트 + 손실 -----
        duct = cfg["duct"]
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        criterion_bundle = build_criterion_from_config(
            cfg,
            repo_root=REPO_ROOT,
            limiter_limit=float(raw_model.limit),
            device=self.device,
            admission=self._criterion_admission,
        )
        self.criterion = criterion_bundle.criterion
        sp = criterion_bundle.admission.secondary
        self.band_plan = criterion_bundle.admission.band_plan
        target_band = criterion_bundle.admission.target_band_hz
        self.trusted_band_hz = criterion_bundle.admission.trusted_band_hz
        if self.is_main:
            if criterion_bundle.admission.role == BROADBAND_CRITERION_ROLE:
                print(
                    "[trainer] broadband equal-subband NMSE: "
                    f"{self.trusted_band_hz[0]:.0f}–{self.trusted_band_hz[1]:.0f}Hz "
                    f"(contract {criterion_bundle.admission.control_band_contract_sha256})"
                )
            else:
                assert sp is not None
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
        # S(z) 총지연 + FIR 정착 구간은 y 가 무엇이든 e = d 로 고정된다.
        # 합성 d 는 P(z) 지연 때문에 그 구간이 비어 있어 공짜지만(trusted 하한 −59.8 dB),
        # 실측 d 는 실신호가 들어 있어 노출된다(하한 mean −20.3 / CVaR10 −10.1 / worst
        # −4.8 dB). 평균 집계에서는 0.03 dB 로 작아 보였지만 CVaR 로 바꾸는 순간
        # 달성 불가능한 목표에 그래디언트가 집중된다 — 두 변경은 반드시 같이 간다.
        # 값의 단일 출처는 PlantSettle 이고, 평가기 warmup 도 같은 값을 하한으로 쓴다.
        if criterion_bundle.admission.secondary_causal is not None:
            causal_secondary = criterion_bundle.admission.secondary_causal
            self.plant_settle = PlantSettle.derive(
                secondary_delay_samples=int(
                    causal_secondary.coarse_delay_samples
                ),
                handoff_samples=int(causal_secondary.handoff_extra_samples),
                fir_taps=int(causal_secondary.support_samples),
                sample_rate=self.fs,
            )
            required_prefix = criterion_bundle.admission.broadband_valid_prefix_samples
            if required_prefix is None:
                raise RuntimeError("causal criterion admission valid prefix가 없습니다")
            self.loss_start_sample = int(cfg.get("loss_start_sample", -1))
            if self.loss_start_sample != int(required_prefix):
                raise ValueError(
                    "causal loss_start_sample은 authority/model/P/S valid prefix와 "
                    f"같아야 합니다: {self.loss_start_sample} != {required_prefix}"
                )
        else:
            assert sp is not None
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
                print("  → 이 경로는 diagnostic 설정에서만 합성원으로 대체됩니다.")
                print("    canonical 역할은 config admission에서 이 상태를 이미 거부합니다.")
                print("    의도가 아니면 학습을 중단하고")
                print("    scripts/data/prepare_noise_pool.py 를 실행하세요.")
                print("=" * 70)

        batch_size = int(cfg["batch_size"])
        broadband_batch_qualified = (
            self._criterion_admission.role == BROADBAND_CRITERION_ROLE
        )
        causal_primary = (
            None
            if self._criterion_admission.primary_causal is None
            else CausalFIRPath(self._criterion_admission.primary_causal)
        )
        causal_prefix = self._criterion_admission.broadband_valid_prefix_samples
        causal_timing = (
            None
            if self._criterion_admission.causal_authority is None
            else self._criterion_admission.causal_authority.timing_contract
        )
        recorded_for_stream = cfg.get("recorded_manifest")
        if recorded_for_stream and not Path(recorded_for_stream).is_absolute():
            recorded_for_stream = str(REPO_ROOT / recorded_for_stream)
        has_recorded_stream = bool(
            recorded_for_stream and Path(recorded_for_stream).is_file()
        )
        validation_source = resolve_validation_source(
            criterion_role=self._criterion_admission.role,
            has_recorded_stream=has_recorded_stream,
            recorded_ratio=float(cfg.get("recorded_ratio", 0.0)),
        )
        if has_recorded_stream:
            synth_resume_batches, recorded_resume_batches = mixed_branch_counts(
                seed,
                self.resume_batch_index,
                float(cfg.get("recorded_ratio", 0.5)),
            )
        else:
            synth_resume_batches, recorded_resume_batches = (
                self.resume_batch_index,
                0,
            )

        synth_train = SynthANCDataset(
            cfg["data"],
            duct,
            split="train",
            seed=seed,
            training_batch_size=batch_size,
            resume_batch_index=synth_resume_batches,
            broadband_batch_qualified=broadband_batch_qualified,
            broadband_primary_operator=(
                None if causal_primary is None else causal_primary.filter_numpy
            ),
            broadband_primary_generator_schema=(
                None
                if causal_primary is None
                else BROADBAND_SYNTH_PRIMARY_GENERATOR_SCHEMA
            ),
            broadband_primary_history_samples=(
                None if causal_primary is None else causal_primary.history_samples
            ),
            broadband_valid_prefix_samples=causal_prefix,
            broadband_timing_contract=causal_timing,
        )
        if synth_train.timing_contract is not None:
            cfg["data"]["training_timing_contract"] = (
                synth_train.timing_contract.model_dump()
            )
        loader = DataLoader(
            synth_train,
            batch_size=batch_size,
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
            rec = RecordedANCDataset(
                recorded_manifest,
                cfg["data"],
                split="train",
                seed=seed + 5,
                timing_contract=synth_train.timing_contract,
                training_batch_size=batch_size,
                resume_batch_index=recorded_resume_batches,
                broadband_valid_prefix_samples=causal_prefix,
            )
            rec_loader = DataLoader(
                rec, batch_size=batch_size, num_workers=2,
                pin_memory=self.device.type == "cuda",
            )
            self.train_iter = MixedIterator(
                self.train_iter,
                iter(rec_loader),
                float(cfg.get("recorded_ratio", 0.5)),
                seed,
                start_batch_index=self.resume_batch_index,
            )
        elif recorded_manifest and self.is_main:
            print(f"[trainer] recorded_manifest({recorded_manifest}) 없음 — 합성 데이터만 사용")

        # CVaR 목적함수는 분위 추정이다. 16개면 q=0.25 가 top-4 뿐이라 추정이 흔들린다.
        # val 배치는 학습 batch_size 와 무관하게 키울 수 있다(그래디언트가 없다).
        self.val_items = int(cfg.get("val_items", 64))
        if self.val_items < 1:
            raise ValueError(f"val_items 는 1 이상이어야 합니다: {self.val_items}")
        if validation_source == "recorded_val_only":
            assert recorded_manifest is not None
            recorded_val_data = copy.deepcopy(cfg["data"])
            # 모델 선택은 실제 recorded val 자체를 본다. train 증강은 고정 RNG라도
            # 원본 val metric이 아니므로 validation branch에서 명시적으로 끈다.
            recorded_val_data["recorded_augment"] = {"enabled": False}
            recorded_val = RecordedANCDataset(
                recorded_manifest,
                recorded_val_data,
                split="val",
                seed=1234,
                timing_contract=synth_train.timing_contract,
                training_batch_size=self.val_items,
                resume_batch_index=0,
                broadband_valid_prefix_samples=causal_prefix,
            )
            self.val_batch = make_recorded_eval_batch(
                recorded_val, n_items=self.val_items
            )
            self.validation_source = validation_source
        else:
            val_ds = SynthANCDataset(
                cfg["data"],
                duct,
                split="val",
                seed=1234,
                training_batch_size=self.val_items,
                broadband_batch_qualified=broadband_batch_qualified,
                broadband_primary_operator=(
                    None if causal_primary is None else causal_primary.filter_numpy
                ),
                broadband_primary_generator_schema=(
                    None
                    if causal_primary is None
                    else BROADBAND_SYNTH_PRIMARY_GENERATOR_SCHEMA
                ),
                broadband_primary_history_samples=(
                    None if causal_primary is None else causal_primary.history_samples
                ),
                broadband_valid_prefix_samples=causal_prefix,
                broadband_timing_contract=causal_timing,
            )
            self.val_batch = make_eval_batch(val_ds, n_items=self.val_items)
            self.validation_source = validation_source

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
        resolved_snapshot = self._cfg_snapshot(cfg)
        cfg.clear()
        cfg.update(resolved_snapshot)
        # resume 계약은 snapshot/log/TensorBoard 등 run 산출물을 건드리기 전에
        # 검증한다. 불일치한 legacy last.pt가 같은 경로에 있다는 이유만으로 현재
        # config 스냅샷을 덮거나 loss_log를 append하면 fail-closed가 아니다.
        if self._resume_state_preview is not None:
            validate_resume_physics(self._resume_state_preview, cfg)
        if self.is_main:
            # resolved model/data/duct까지 전부 남겨야 surrogate/실측 P(z), lead,
            # plant curriculum을 나중에 정확히 구분할 수 있다. 비밀정보는 학습
            # config에 두지 않는 저장소 규약을 따른다.
            reproducibility_cfg = resolved_snapshot
            if str(cfg.get("experiment_role", "")) == A100_PRETRAIN_SMOKE_ROLE:
                # stop(예: 300) → resume(500)은 같은 target의 **운영** 차이일 뿐
                # config snapshot의 학습 의미가 달라진 것이 아니다. strict no-replace
                # snapshot에는 이 두 operational field를 남기지 않고, 실제 값은 각
                # immutable checkpoint cfg와 phase telemetry에 보존한다.
                reproducibility_cfg = copy.deepcopy(resolved_snapshot)
                reproducibility_cfg.pop("resume", None)
                reproducibility_cfg.pop("run_until_step", None)
            self.reproducibility_receipt = snapshot_run(
                self.run_dir,
                reproducibility_cfg,
                strict=str(cfg.get("experiment_role", "")) in STRICT_RUN_ROLES,
            )
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(str(self.run_dir / "tb"))
            except ImportError:
                print("[trainer] tensorboard 미설치 — 파일 로그만 기록합니다")
        self.loss_log = open(self.run_dir / "loss_log.txt", "a", encoding="utf-8") if self.is_main else None

        # init_ckpt(파인튜닝) → resume(재개) 순서로 적용. canonical metadata/bytes는
        # 위의 preflight에서 이미 한 번 snapshot했으며 pathname을 다시 열지 않는다.
        init_ckpt = _abs_checkpoint(cfg.get("init_ckpt"))
        init_required = bool(cfg.get("require_init_checkpoint", False))
        if cfg.get("init_ckpt") and self._init_state_preview is None and self.is_main:
            print(f"[trainer] 경고: init_ckpt({init_ckpt})가 없어 무시합니다")
        if init_ckpt and self._init_state_preview is not None:
            state = self._init_state_preview
            if state is None:
                # 비공식 optional init도 preflight에서 존재했다면 snapshot돼야 한다.
                raise RuntimeError("init checkpoint preflight snapshot이 없습니다")
            validate_init_checkpoint_role(state, cfg)
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
            state = load_checkpoint(
                init_ckpt,
                self.model,
                restore_rng=False,
                map_location="cpu",
                preloaded_state=state,
            )
            if self.is_main:
                print(f"[trainer] init_ckpt 로드: {init_ckpt} (step {state.get('step')})")
        resume = _abs_checkpoint(cfg.get("resume"))
        if resume and Path(resume).exists():
            # 모델/optimizer/scheduler를 변형하기 **전에** metadata와 artifact 계약을
            # 검사한다. 구형 checkpoint가 우연히 lead만 같아도 여기서 차단된다.
            state = self._resume_state_preview
            if state is None:
                raise RuntimeError("resume checkpoint preflight snapshot이 없습니다")
            validate_resume_physics(state, cfg)
            state = load_checkpoint(
                resume,
                self.model,
                self.optimizer,
                self.scheduler,
                map_location="cpu",
                preloaded_state=self._resume_state_preview,
            )
            try:
                stochastic = state["training_state"]
                if int(stochastic.get("schema_version", -1)) == 1:
                    self.criterion.plant.rng.bit_generator.state = stochastic["plant_rng"]
                elif (
                    int(stochastic.get("schema_version", -1)) == 2
                    and stochastic.get("plant_rng_kind")
                    == "not_applicable_frozen_causal_fir"
                    and stochastic.get("plant_rng") is None
                    and self._criterion_admission.causal_authority is not None
                ):
                    pass
                else:
                    raise ValueError("training_state plant RNG kind/schema가 다릅니다")
                if self.criterion.nonlinear is not None:
                    self.criterion.nonlinear.rng.bit_generator.state = stochastic[
                        "nonlinear_rng"
                    ]
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "resume checkpoint plant/nonlinear RNG를 복원할 수 없습니다"
                ) from exc
            self.step = int(state.get("step", 0))
            self.best_metric = float(state.get("best_metric", float("inf")))
            if self.is_main:
                print(f"[trainer] 재개: step {self.step}, best {self.best_metric:.3f}")

    def _cfg_snapshot(self, cfg: dict) -> dict:
        """이 실행이 쓴 목적함수/절단 규약까지 포함한 스냅샷."""

        return cfg_snapshot(
            cfg,
            self.trusted_band_hz,
            best_metric_key=self.best_metric_key,
            loss_start_sample=self.loss_start_sample,
            cvar_scope="rank_local" if self.world > 1 else "global",
        )

    def _training_state_snapshot(self) -> dict:
        """global RNG 밖 학습 구성요소의 순차 RNG 상태."""

        if self._criterion_admission.causal_authority is not None:
            return {
                "schema_version": 2,
                "plant_rng_kind": "not_applicable_frozen_causal_fir",
                "plant_rng": None,
                "nonlinear_rng": (
                    None
                    if self.criterion.nonlinear is None
                    else copy.deepcopy(
                        self.criterion.nonlinear.rng.bit_generator.state
                    )
                ),
            }
        return {
            "schema_version": 1,
            "plant_rng": copy.deepcopy(self.criterion.plant.rng.bit_generator.state),
            "nonlinear_rng": (
                None
                if self.criterion.nonlinear is None
                else copy.deepcopy(
                    self.criterion.nonlinear.rng.bit_generator.state
                )
            ),
        }

    # ---------- 스텝 ----------

    def _validate_causal_batch_prefix(self, batch: dict) -> None:
        required = self._criterion_admission.broadband_valid_prefix_samples
        if required is None:
            return
        value = batch.get("valid_start_sample")
        if not isinstance(value, torch.Tensor):
            raise ValueError("causal broadband batch에 valid_start_sample이 없습니다")
        flat = value.detach().cpu().reshape(-1)
        if flat.numel() < 1 or flat.dtype not in (torch.int32, torch.int64):
            raise ValueError("causal broadband valid_start_sample dtype/shape가 다릅니다")
        if not bool(torch.all(flat == int(required)).item()):
            raise ValueError(
                "causal broadband batch valid crop이 authority prefix와 다릅니다"
            )

    def _forward_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        self._validate_causal_batch_prefix(batch)
        validate_finite_batch(
            batch,
            model_input_contract=self.model_input_contract,
        )
        x = batch["x"].to(self.device, non_blocking=True)
        d = batch["d"].to(self.device, non_blocking=True)
        if self.stage == "closed_loop":
            loss, metrics = self._closed_loop_forward(x, d)
        else:
            y = self.model(x)
            validate_finite_output(y)
            # open_loop 도 정착 구간을 버린다. 예전에는 closed_loop 만 skip 을 넘겼고
            # 실측 배치(open_loop)는 e[0:1721] = d[0:1721] 을 그대로 손실에 넣었다.
            loss, metrics = self.criterion(
                y, d, loss_start_sample=self.loss_start_sample
            )
        validate_finite_loss_metrics(loss, metrics)
        return loss, metrics

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
        validate_finite_output(y)
        # 폐루프 워밍업과 플랜트 정착 중 **더 긴 쪽**을 버린다.
        skip = max(int(warmup_s * self.fs), self.loss_start_sample)
        # 절단은 손실 내부에서 플랜트 적용 "후"에 수행 (결함 #2/#5)
        return self.criterion(y, d, loss_start_sample=skip, perturb=perturb, nl_params=nl_params)

    def _validate_metrics(self) -> dict[str, float]:
        self.model.eval()
        self.criterion.eval()
        with torch.no_grad():
            self._validate_causal_batch_prefix(self.val_batch)
            validate_finite_batch(
                self.val_batch,
                model_input_contract=self.model_input_contract,
            )
            x = self.val_batch["x"].to(self.device)
            d = self.val_batch["d"].to(self.device)
            raw = self.model.module if hasattr(self.model, "module") else self.model
            y = raw(x)
            validate_finite_output(y)
            loss, metrics = self.criterion(
                y, d, loss_start_sample=self.loss_start_sample
            )
            validate_finite_loss_metrics(loss, metrics)
        self.model.train()
        self.criterion.train()
        return metrics

    def _validate(self) -> float:
        """기존 호출 호환용 단일 값 검증 API — role별 best metric을 반환."""
        return self._validate_metrics()[self.best_metric_key]

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
        run_started = time.monotonic()
        last_train_metrics: dict[str, float] = {}

        while self.step < self.run_until_step:
            batch = next(self.train_iter)
            self.optimizer.zero_grad(set_to_none=True)

            if self.amp_dtype is not None and self.device.type == "cuda":
                with torch.autocast("cuda", dtype=self.amp_dtype):
                    loss, metrics = self._forward_loss(batch)
            else:
                loss, metrics = self._forward_loss(batch)

            loss.backward()
            validate_finite_gradients(self.model)
            if self.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                _require_finite_tensor("gradient.global_norm", grad_norm.reshape(1))
                validate_finite_gradients(self.model)
            self.optimizer.step()
            validate_finite_parameters(self.model)
            self.scheduler.step()
            self.step += 1
            last_train_metrics = {
                str(key): float(value) for key, value in metrics.items()
            }

            if self.is_main and self.step % log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                sps = log_every / max(1e-9, time.time() - t0)
                t0 = time.time()
                line = (
                    f"step {self.step:7d} | loss {metrics['loss']:8.3f} | "
                    f"nmse_t {metrics[self.best_metric_key]:7.2f} dB | "
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
                if self._criterion_admission.role == BROADBAND_CRITERION_ROLE:
                    val_mean = float(val_metrics["nmse_subband_equal_db"])
                    val_worst = float(val_metrics["nmse_subband_worst_db"])
                else:
                    val_mean = float(val_metrics["nmse_trusted_db"])
                    val_worst = float(
                        val_metrics.get("nmse_trusted_worst_db", float("nan"))
                    )
                val_nmse = float(val_metrics[self.best_metric_key])
                val_fullband_nmse = val_metrics["nmse_fullband_db"]
                stop_flag = torch.zeros(1, device=self.device)
                if self.is_main:
                    metric_label = (
                        "broadband subband CVaR"
                        if self._criterion_admission.role == BROADBAND_CRITERION_ROLE
                        else "trusted CVaR"
                    )
                    print(
                        f"[eval] step {self.step}: val {metric_label} {val_nmse:+.2f} dB "
                        f"(mean {val_mean:+.2f}, worst {val_worst:+.2f}) | "
                        f"fullband {val_fullband_nmse:+.2f} dB | "
                        f"대역밖 최악 {val_metrics.get('dnh_worst_db', float('nan')):+.2f} dB",
                        flush=True,
                    )
                    if self.writer:
                        # 기존 대시보드의 val/nmse_db는 목적함수 alias로 유지.
                        self.writer.add_scalar("val/nmse_db", val_nmse, self.step)
                        if self._criterion_admission.role == BROADBAND_CRITERION_ROLE:
                            self.writer.add_scalar(
                                "val/nmse_subband_equal_db", val_mean, self.step
                            )
                            self.writer.add_scalar(
                                f"val/{self.best_metric_key}", val_nmse, self.step
                            )
                        else:
                            self.writer.add_scalar(
                                "val/nmse_trusted_db", val_mean, self.step
                            )
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
                        training_state=self._training_state_snapshot(),
                    )
                    if is_best:
                        save_checkpoint(
                            self.run_dir / "ckpt" / "best.pt",
                            self.model, self.optimizer, self.scheduler,
                            self.step, self.best_metric,
                            self._cfg_snapshot(cfg),
                            training_state=self._training_state_snapshot(),
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
                training_state=self._training_state_snapshot(),
            )
            if self.step == self.total_steps:
                write_completion_receipt(
                    self.run_dir / "ckpt", repo_root=REPO_ROOT
                )
            if str(cfg.get("experiment_role", "")) == A100_PRETRAIN_SMOKE_ROLE:
                if self.device.type == "cuda":
                    peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
                    peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
                else:
                    peak_allocated = 0
                    peak_reserved = 0
                telemetry_path = write_a100_pretrain_smoke_phase_telemetry(
                    self.run_dir,
                    cfg,
                    start_step=int(self.resume_batch_index),
                    completed_step=int(self.step),
                    elapsed_seconds=float(time.monotonic() - run_started),
                    device=str(self.device),
                    cuda_available=self.device.type == "cuda",
                    device_count=int(torch.cuda.device_count()),
                    max_memory_allocated_bytes=peak_allocated,
                    max_memory_reserved_bytes=peak_reserved,
                    resume_checkpoint_sha256=self.resume_checkpoint_sha256,
                    final_train_metrics=last_train_metrics,
                )
                print(f"[a100 smoke] telemetry → {telemetry_path}", flush=True)
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
    derived: dict[str, object] = {
        "digital_reference_lead_samples": lead,
        "physics_status": validate_training_physics(out),
    }
    if trusted_band_hz is not None:
        derived["trusted_band_hz"] = [float(v) for v in trusted_band_hz]
    if best_metric_key is not None:
        derived["best_metric_key"] = str(best_metric_key)
    if loss_start_sample is not None:
        derived["loss_start_sample"] = int(loss_start_sample)
    if cvar_scope is not None:
        # DDP 에서 topk 는 랭크 로컬이다. world>1 이면 각 랭크의 상위 q 합집합이
        # 글로벌 상위 q 를 덮으므로 실효 분위가 넓어진다 — 산출물에 남긴다.
        derived["nmse_cvar_scope"] = str(cvar_scope)
    already_stamped = "experiment_contract_sha256" in out
    for key, value in derived.items():
        if already_stamped and out.get(key) != value:
            raise ValueError(
                f"최종 contract stamp 뒤 파생 설정 {key}가 바뀌었습니다: "
                f"stamped={out.get(key)!r}, runtime={value!r}"
            )
        out[key] = value
    if not already_stamped:
        out = stamp_experiment_contract(out, repo_root=REPO_ROOT)
    else:
        validate_embedded_experiment_contract(out)
    return out
