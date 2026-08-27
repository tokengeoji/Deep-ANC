"""YAML 설정 로드/병합/검증.

모든 스크립트는 이 모듈을 통해 설정을 읽는다. 설정 파일 간 참조
(train_*.yaml 의 model_config / data_config / duct_config)는 여기서 해석한다.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .train.a100_pretrain_smoke import A100_PRETRAIN_SMOKE_ROLE

# 저장소 루트 (src/deep_anc/config.py 기준 두 단계 위)
REPO_ROOT = Path(__file__).resolve().parents[2]

# 3-스레드 런타임의 콜백↔추론 핸드오프(1 hop) — 학습 플랜트 지연에 가산되는 기본값.
# duct.yaml secondary_path.handoff_extra_samples 가 명시되면 그 값을 쓰고,
# 모든 소비처(.get 기본값)는 이 상수를 공유한다 (감사 L10 — 기본값 분기 금지).
DEFAULT_HANDOFF_SAMPLES = 256

CANONICAL_FINETUNE_POLICY_VERSION = "canonical_finetune_v1"
CANONICAL_PRETRAIN_POLICY_VERSION = "canonical_pretrain_v1"
A100_PRETRAIN_SMOKE_POLICY_VERSION = "a100_pretrain_smoke_v1"
CANONICAL_DETERMINISM_POLICY = {
    "schema_version": 1,
    "torch_use_deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    # CUDA에서는 PyTorch deterministic GEMM이 이 두 값 중 하나를 요구한다.
    # 실제 선택값은 environment.json에 기록하고 completion receipt에 결속한다.
    "cublas_workspace_config_allowed": [":4096:8", ":16:8"],
}
CANONICAL_LOSS_BASELINE = {
    "nmse_objective": "trusted_band",
    "nmse_cvar_q": 0.25,
    "nmse_cvar_min_k": 4,
    # 과거 0.264는 strict S라도 loss_start_sample=0으로 계산한 값이라 실제 Trainer
    # 목적함수의 증거가 아니다. 현행 strict S + 3549-sample 정착 절단 고정 fixture는
    # 0.130이지만, 이 값도 실제 A100 모델/배치를 대표하지 않는다. canonical 전에는
    # 같은 loss_start_sample을 결속한 campaign prerequisite ledger의 0.2–0.4 증거가
    # 있어야 하며, 그 전에는 이 baseline을 승인값으로 해석하지 않는다.
    "lambda_dnh": 0.00075,
    "dnh_weight_below": 1.0,
    "dnh_weight_above": 2.0,
    "nmse_frame_samples": 8192,
    "nmse_frame_silence_db": -40.0,
    "mrstft_ffts": [256, 512, 1024, 2048],
    "lambda_mrstft": 1.0,
    "band_weight": "trusted_only",
    "lambda_pow": 0.0,
    "lambda_sat": 1.0,
    "sat_margin": 2.0,
}
CANONICAL_LOSS_GRID = frozenset(
    {
        # signed frame-CVaR는 λ=0.5와 0.2 모두 fixed-batch control에서 y→0으로
        # 붕괴했다. v1 canonical은 frame을 metric-only로 두고, alpha만 비교한다.
        # frame gradient 재도입은 one-sided/item-wise v2 guard와 별도 evidence 뒤에만 한다.
        (0.7, 0.0),
        (1.0, 0.0),
        (0.85, 0.0),
    }
)
CANONICAL_OPTIMIZER_PRETRAIN = {
    "name": "adamw",
    "lr": 1.0e-3,
    "weight_decay": 1.0e-4,
    "betas": [0.9, 0.999],
}
CANONICAL_OPTIMIZER_FINETUNE = {
    "name": "adamw",
    "lr": 1.0e-4,
    "weight_decay": 1.0e-4,
    "betas": [0.9, 0.999],
}
CANONICAL_PRETRAIN_SCHEDULE = {
    "warmup_steps": 1250,
    "total_steps": 100000,
    "min_lr": 1.0e-5,
}
CANONICAL_FINETUNE_SCHEDULE = {
    "warmup_steps": 1000,
    "total_steps": 50000,
    "min_lr": 1.0e-6,
}
CANONICAL_SOURCE_MIX_RATIO = {
    "synthetic": 0.25,
    "dns_fullband": 0.30,
    "speech": 0.15,
    "music": 0.10,
    "demand": 0.08,
    "machine": 0.07,
    "esc50": 0.05,
}
CANONICAL_RECORDED_AUGMENT = {
    "enabled": True,
    "level_db_range": [-12.0, 6.0],
    "polarity_flip": True,
    "mic_noise_snr_db": [12.0, 40.0],
    "eq_tilt_db": 6.0,
    "eq_band_db": 4.0,
    "eq_bands_hz": [100.0, 300.0, 700.0, 1400.0],
    "mix_probability": 0.0,
    "mix_weight_range": [0.0, 0.7],
    "lead_jitter_samples": 0.0,
}
CANONICAL_FINETUNE_POLICY = {
    "experiment_role": "canonical_finetune",
    "init_eligible": False,
    "require_measured_primary_path": True,
    "require_init_checkpoint": True,
    "require_recorded_manifest": True,
    "required_init_experiment_role": "canonical_pretrain",
    "require_init_eligible": True,
    "contract_run_dir": True,
    "required_world_size": 1,
    "stage": "open_loop",
    "model_config": "configs/model_tiny.yaml",
    "batch_size": 16,
    "num_workers": 8,
    "prefetch_factor": 4,
    "optimizer": CANONICAL_OPTIMIZER_FINETUNE,
    "schedule": CANONICAL_FINETUNE_SCHEDULE,
    "amp": "bf16",
    "grad_clip": 5.0,
    "freeze_encoder": False,
    "val_items": 64,
    "eval_every": 1000,
    "log_every": 100,
    "early_stop_patience": 0,
    "recorded_manifest": "data/manifests/recorded_regrouped.jsonl",
    "recorded_ratio": 0.7,
}
CANONICAL_PRETRAIN_POLICY = {
    "experiment_role": "canonical_pretrain",
    "init_eligible": True,
    "stage": "open_loop",
    "model_config": "configs/model_tiny.yaml",
    "batch_size": 96,
    "num_workers": 14,
    "prefetch_factor": 4,
    "optimizer": CANONICAL_OPTIMIZER_PRETRAIN,
    "schedule": CANONICAL_PRETRAIN_SCHEDULE,
    "amp": "bf16",
    "grad_clip": 5.0,
    "val_items": 64,
    "eval_every": 500,
    "log_every": 100,
    "early_stop_patience": 0,
    "required_world_size": 1,
    "contract_run_dir": True,
}
CANONICAL_FINETUNE_READINESS_POLICY = {
    "required_path_band_hz": [150, 1600],
    "min_path_consistency": 0.9406,
    "required_recorded_ratio": 0.7,
    "min_recorded_sessions": 80,
    "min_recorded_duration_seconds": 5400,
    "required_source_families": ["speech", "music", "environment", "machine"],
    "target_cancellation_db": 1.0,
    "cancellation_ceiling_margin_db": 0.5,
    "measured_design_ceiling_db": 2.15,
    "measured_design_ceiling_band_hz": [150, 1600],
    "min_groups_per_family_per_split": 4,
    "min_source_err_coherence": 0.60,
    "min_ref_err_coherence": 0.60,
    "max_measured_delay_mismatch_samples": 64,
    "min_delay_crosscheck_sessions": 8,
    "max_init_lead_mismatch_samples": 16,
    "require_completed_init_checkpoint": True,
    "max_init_best_metric_db": 0.0,
    "allowed_init_physics_statuses": [
        "secondary_surrogate_representation_pretrain"
    ],
}


def _resolve_path(path: str | Path) -> Path:
    """상대 경로는 저장소 루트 기준으로 해석한다 (실행 위치와 무관하게 동작)."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        candidate = Path.cwd() / p
        p = candidate if candidate.exists() else REPO_ROOT / p
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: 최상위는 매핑(dict)이어야 합니다")
    return data


def deep_merge(base: dict, override: dict) -> dict:
    """중첩 dict 병합 — override 우선. 리스트는 통째로 교체."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """'a.b.c=value' 형태의 CLI 오버라이드 적용."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"오버라이드 형식 오류 (key=value): {item}")
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def load_train_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """학습 설정 로드 + 참조된 model/data/duct 설정을 함께 해석.

    오버라이드는 두 번 적용한다: 참조 경로 자체(model_config 등)를 바꿀 수 있도록
    로드 전에 한 번, 로드된 서브 설정의 내부 키(data.* 등)를 바꿀 수 있도록 후에 한 번.
    """
    cfg = load_yaml(path)
    resolved_config_path = _resolve_path(path).resolve()
    declared_canonical_finetune = (
        str(cfg.get("experiment_role", "")) == "canonical_finetune"
        or cfg.get("canonical_trust_policy")
        == CANONICAL_FINETUNE_POLICY_VERSION
        or resolved_config_path == (REPO_ROOT / "configs/train_finetune.yaml").resolve()
    )
    declared_canonical_pretrain = (
        str(cfg.get("experiment_role", "")) == "canonical_pretrain"
        or cfg.get("canonical_trust_policy")
        == CANONICAL_PRETRAIN_POLICY_VERSION
        or resolved_config_path
        == (REPO_ROOT / "configs/train_pretrain_tiny.yaml").resolve()
    )
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    cfg["model"] = load_yaml(cfg["model_config"])
    cfg["data"] = load_yaml(cfg["data_config"])
    cfg["duct"] = load_yaml(cfg["duct_config"])
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    if declared_canonical_finetune:
        _enforce_canonical_finetune_policy(cfg)
    if declared_canonical_pretrain:
        role = str(cfg.get("experiment_role", ""))
        if role == "canonical_pretrain":
            _enforce_canonical_pretrain_policy(cfg)
        elif role == A100_PRETRAIN_SMOKE_ROLE:
            _enforce_a100_pretrain_smoke_policy(cfg)
        else:
            _enforce_pretrain_derivative_policy(cfg)
    # canonical YAML 밖에서 이 역할 이름만 주입해 장기/완화 학습을 우회하는
    # 경로도 닫는다. smoke는 canonical pretrain의 semantic projection만 증명하는
    # 별도 role이지 임의 config를 A100 prerequisite로 승격하는 escape hatch가 아니다.
    elif str(cfg.get("experiment_role", "")) == A100_PRETRAIN_SMOKE_ROLE:
        _enforce_a100_pretrain_smoke_policy(cfg)
    transfer_anchor_keys = {
        "bootstrap_receipt_sha256",
        "transfer_manifest_sha256",
    }
    resolved_role = str(cfg.get("experiment_role", ""))
    if resolved_role in {
        "canonical_pretrain",
        "canonical_finetune",
        A100_PRETRAIN_SMOKE_ROLE,
    }:
        # 공식 계약은 검증된 Elice bootstrap generation을 config에 먼저
        # 결속한 뒤에만 timing/experiment SHA를 만들 수 있다. anchor 없는
        # config를 일단 stamp하고 Trainer에서 뒤늦게 거부하면 동일 YAML이
        # 서로 다른 입력 세대를 가리킬 수 있다.
        if (
            cfg["data"].get("bootstrap_receipt")
            != "data/manifests/elice_bootstrap_receipt.json"
            or not cfg["data"].get("bootstrap_receipt_sha256")
        ):
            raise ValueError(
                "canonical 학습 config를 stamp하려면 "
                "data.bootstrap_receipt와 외부 bootstrap_receipt_sha256이 필요합니다"
            )
        from .data.transfer_contract import bind_recorded_transfer_config

        bind_recorded_transfer_config(cfg["data"], repo_root=REPO_ROOT)
        if resolved_role == "canonical_pretrain":
            prerequisite_sha = str(cfg.get("campaign_prerequisite_sha256") or "")
            if (
                cfg.get("campaign_prerequisite")
                != "results/training_prerequisites/canonical_pretrain.json"
                or len(prerequisite_sha) != 64
                or any(character not in "0123456789abcdef" for character in prerequisite_sha)
            ):
                raise ValueError(
                    "canonical pretrain config를 stamp하려면 canonical campaign "
                    "prerequisite path와 외부 SHA-256이 필요합니다"
                )
    elif any(cfg["data"].get(key) for key in transfer_anchor_keys):
        from .data.transfer_contract import bind_recorded_transfer_config

        bind_recorded_transfer_config(cfg["data"], repo_root=REPO_ROOT)
    if str(cfg.get("experiment_role", "")) in {
        "canonical_pretrain",
        "canonical_finetune",
        A100_PRETRAIN_SMOKE_ROLE,
    }:
        declared_primary_delay = cfg["data"].get("require_primary_delay_artifact")
        if declared_primary_delay is False:
            raise ValueError(
                "canonical digital 학습의 data.require_primary_delay_artifact를 "
                "false로 약화할 수 없습니다"
            )
        cfg["data"]["require_primary_delay_artifact"] = True
    validate_duct(cfg["duct"])
    _propagate_training_timing_contract(cfg)
    _finalize_training_metadata(cfg)
    _validate_resolved_training_contract(cfg)
    if resolved_role == A100_PRETRAIN_SMOKE_ROLE:
        # smoke는 campaign ledger 없이 먼저 실행되지만, canonical과 같은 resolved
        # semantics를 target으로 결속하고 canonical ``runs/``와 분리한다. target은
        # output/resume/run_until을 제외하므로 uninterrupted와 K+resume arm이 같은
        # 학습 의미를 공유한다.
        from .train.a100_pretrain_smoke import (
            build_a100_pretrain_smoke_target,
            smoke_run_directory,
            validate_a100_pretrain_smoke_config,
        )
        from .train.experiment_contract import stamp_experiment_contract

        target = build_a100_pretrain_smoke_target(cfg, repo_root=REPO_ROOT)
        cfg["smoke_target_sha256"] = target["sha256"]
        run_dir = smoke_run_directory(cfg, repo_root=REPO_ROOT)
        cfg["ckpt_dir"] = str(run_dir.relative_to(REPO_ROOT))
        cfg["resolved_smoke_run_dir"] = {
            "schema": "results/training_prerequisites/a100_pretrain_smoke/<target>/<label>",
            "smoke_target_sha256": target["sha256"],
        }
        cfg = stamp_experiment_contract(cfg, repo_root=REPO_ROOT)
        validate_a100_pretrain_smoke_config(cfg, repo_root=REPO_ROOT)
    elif bool(cfg.get("contract_run_dir", False)):
        from .train.experiment_contract import contract_run_directory

        run_dir, contract_sha = contract_run_directory(cfg, repo_root=REPO_ROOT)
        cfg["ckpt_dir"] = str(run_dir.relative_to(REPO_ROOT))
        cfg["resolved_contract_run_dir"] = {
            "schema": "runs/<stage>_<experiment-contract-sha16>_<seed>",
            "experiment_contract_sha256": contract_sha,
        }
        from .train.experiment_contract import stamp_experiment_contract

        cfg = stamp_experiment_contract(cfg, repo_root=REPO_ROOT)
        if cfg["experiment_contract_sha256"] != contract_sha:
            raise RuntimeError(
                "resolved run directory와 최종 experiment contract SHA가 다릅니다"
            )
    return cfg


def _enforce_canonical_finetune_policy(cfg: dict) -> None:
    """canonical config의 CLI 약화/역할 세탁을 override 적용 뒤 차단한다."""

    mismatches: list[str] = []
    for key, required in CANONICAL_FINETUNE_POLICY.items():
        if cfg.get(key) != required:
            required_text = str(required).lower() if isinstance(required, bool) else repr(required)
            mismatches.append(
                f"{key}={cfg.get(key)!r} (required {key}={required_text})"
            )
    readiness = cfg.get("readiness")
    if not isinstance(readiness, dict):
        mismatches.append("readiness=<missing mapping>")
    else:
        for key, required in CANONICAL_FINETUNE_READINESS_POLICY.items():
            if readiness.get(key) != required:
                mismatches.append(
                    f"readiness.{key}={readiness.get(key)!r} (required {required!r})"
                )
    _collect_common_training_policy_mismatches(
        cfg, mismatches, role="canonical_finetune"
    )
    seed = cfg.get("seed")
    if seed not in {20260803, 20260903}:
        mismatches.append(
            f"seed={seed!r} (required one of [20260803, 20260903])"
        )
    if mismatches:
        raise ValueError(
            "canonical_finetune trust policy는 override로 약화할 수 없습니다: "
            + "; ".join(mismatches)
        )
    cfg["canonical_trust_policy"] = CANONICAL_FINETUNE_POLICY_VERSION


def _enforce_canonical_pretrain_policy(cfg: dict) -> None:
    """100k tiny canonical pretrain을 짧은 pilot로 역할 세탁하지 못하게 한다."""

    mismatches: list[str] = []
    for key, required in CANONICAL_PRETRAIN_POLICY.items():
        if cfg.get(key) != required:
            mismatches.append(f"{key}={cfg.get(key)!r} (required {required!r})")
    _collect_canonical_pretrain_semantic_mismatches(cfg, mismatches)
    if mismatches:
        raise ValueError(
            "canonical_pretrain trust policy는 override로 약화할 수 없습니다: "
            + "; ".join(mismatches)
        )
    cfg["canonical_trust_policy"] = CANONICAL_PRETRAIN_POLICY_VERSION


def _collect_canonical_pretrain_semantic_mismatches(
    cfg: dict, mismatches: list[str]
) -> None:
    """role/output/ledger 외 canonical-pretrain 학습 의미의 공용 검사."""

    if bool(cfg.get("freeze_encoder", False)):
        mismatches.append("freeze_encoder=true (required false)")
    _collect_common_training_policy_mismatches(
        cfg, mismatches, role="canonical_pretrain"
    )
    seed = cfg.get("seed")
    if seed not in {20260803, 20260903}:
        mismatches.append(
            f"seed={seed!r} (required one of [20260803, 20260903])"
        )
    data = cfg.get("data") or {}
    for key, required in {
        "sample_rate": 48000,
        "segment_seconds": 1.5,
        "reference_mode": "digital",
        "digital_primary_path_mode": "secondary_surrogate",
        "recorded_lead_mode": "timeline",
        "recorded_sampling": "family_lineage_session_balanced",
        "source_mix_ratio": CANONICAL_SOURCE_MIX_RATIO,
        "recorded_augment": CANONICAL_RECORDED_AUGMENT,
    }.items():
        if data.get(key) != required:
            mismatches.append(
                f"data.{key}={data.get(key)!r} (required {required!r})"
            )


def _enforce_a100_pretrain_smoke_policy(cfg: dict) -> None:
    """ledger 순환을 끊되 canonical pretrain 의미를 약화하지 않는 smoke 역할."""

    mismatches: list[str] = []
    for key, required in CANONICAL_PRETRAIN_POLICY.items():
        if key in {"experiment_role", "init_eligible", "contract_run_dir"}:
            continue
        if cfg.get(key) != required:
            mismatches.append(f"{key}={cfg.get(key)!r} (required {required!r})")
    if cfg.get("experiment_role") != A100_PRETRAIN_SMOKE_ROLE:
        mismatches.append(
            f"experiment_role={cfg.get('experiment_role')!r} "
            f"(required {A100_PRETRAIN_SMOKE_ROLE!r})"
        )
    if cfg.get("init_eligible") is not False:
        mismatches.append("init_eligible must be false")
    if cfg.get("contract_run_dir") is not False:
        mismatches.append("contract_run_dir must be false")
    if cfg.get("campaign_prerequisite") not in (None, "") or cfg.get(
        "campaign_prerequisite_sha256"
    ) not in (None, ""):
        mismatches.append("campaign prerequisite must be null for smoke")
    if cfg.get("init_ckpt") not in (None, ""):
        mismatches.append("init_ckpt must be null for smoke")
    label = str(cfg.get("a100_smoke_run_label", ""))
    if label not in {"uninterrupted", "resumed"}:
        mismatches.append("a100_smoke_run_label must be uninterrupted or resumed")
    # 이 role은 장기 학습의 우회 통로가 아니라 canonical 이전의 bounded
    # exact-resume 증거만 만드는 용도다. run_until_step은 smoke target에서
    # 제외되는 운영값이지만, role policy 자체는 200–500 step으로 닫는다.
    try:
        run_until_step = int(cfg.get("run_until_step", 0))
    except (TypeError, ValueError):
        run_until_step = 0
    if not 200 <= run_until_step <= 500:
        mismatches.append("run_until_step must be in [200, 500] for A100 smoke")
    _collect_canonical_pretrain_semantic_mismatches(cfg, mismatches)
    if mismatches:
        raise ValueError(
            "a100_pretrain_smoke trust policy는 canonical 학습 의미를 약화할 수 없습니다: "
            + "; ".join(mismatches)
        )
    cfg["canonical_trust_policy"] = A100_PRETRAIN_SMOKE_POLICY_VERSION


def _enforce_pretrain_derivative_policy(cfg: dict) -> None:
    """canonical 설정에서 파생된 pilot/probe/진단이 init 자격을 갖지 못하게 한다."""

    role = str(cfg.get("experiment_role", ""))
    allowed = {"loss_pilot", "measured_probe", "diagnostic_overfit"}
    if role not in allowed:
        raise ValueError(
            "canonical pretrain 설정의 역할 변경은 승인된 derivative만 허용합니다: "
            f"role={role!r}, allowed={sorted(allowed)}"
        )
    if cfg.get("init_eligible") is not False:
        raise ValueError(
            f"{role}는 init_eligible=false여야 합니다 — pilot/probe 승격을 금지합니다"
        )
    if role in {"loss_pilot", "measured_probe"} and not bool(
        cfg.get("contract_run_dir", False)
    ):
        raise ValueError(f"{role}는 contract_run_dir=true여야 합니다")
    cfg["canonical_trust_policy"] = f"{role}_derivative_v1"


def _collect_common_training_policy_mismatches(
    cfg: dict, mismatches: list[str], *, role: str
) -> None:
    expected_model = load_yaml("configs/model_tiny.yaml")
    if cfg.get("model") != expected_model:
        mismatches.append("model=<override> (required exact configs/model_tiny.yaml)")
    loss = cfg.get("loss")
    if not isinstance(loss, dict):
        mismatches.append("loss=<missing mapping>")
    else:
        expected_keys = set(CANONICAL_LOSS_BASELINE) | {
            "nmse_cvar_alpha",
            "lambda_frame",
        }
        if set(loss) != expected_keys:
            mismatches.append(
                "loss key set mismatch: "
                f"extra={sorted(set(loss) - expected_keys)}, "
                f"missing={sorted(expected_keys - set(loss))}"
            )
        for key, required in CANONICAL_LOSS_BASELINE.items():
            if loss.get(key) != required:
                mismatches.append(
                    f"loss.{key}={loss.get(key)!r} (required {required!r})"
                )
        pair = (loss.get("nmse_cvar_alpha"), loss.get("lambda_frame"))
        if pair not in CANONICAL_LOSS_GRID:
            mismatches.append(
                "loss alpha×frame이 승인 grid가 아닙니다: "
                f"{pair!r}, allowed={sorted(CANONICAL_LOSS_GRID)}"
            )
    data = cfg.get("data") or {}
    if role == "canonical_finetune":
        for key, required in {
            "sample_rate": 48000,
            "segment_seconds": 1.5,
            "reference_mode": "digital",
            "recorded_lead_mode": "timeline",
            "recorded_sampling": "family_lineage_session_balanced",
            "source_mix_ratio": CANONICAL_SOURCE_MIX_RATIO,
            "recorded_augment": CANONICAL_RECORDED_AUGMENT,
        }.items():
            if data.get(key) != required:
                mismatches.append(
                    f"data.{key}={data.get(key)!r} (required {required!r})"
                )


def loss_selection_sha256(loss_cfg: dict) -> str:
    """사전학습과 fine-tune이 공유해야 하는 전체 loss-selection 정체성."""

    if not isinstance(loss_cfg, dict) or not loss_cfg:
        raise ValueError("loss selection digest에는 비어 있지 않은 loss mapping이 필요합니다")
    encoded = json.dumps(
        loss_cfg,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_canonical_training_policy(cfg: dict) -> None:
    """checkpoint/receipt 경계에서도 canonical 이름만 붙인 설정을 거부한다."""

    role = str(cfg.get("experiment_role", ""))
    if role == "canonical_pretrain":
        if cfg.get("canonical_trust_policy") != CANONICAL_PRETRAIN_POLICY_VERSION:
            raise ValueError(
                "canonical_pretrain checkpoint에 canonical_pretrain_v1 trust policy가 없습니다"
            )
        _enforce_canonical_pretrain_policy(cfg)
    elif role == "canonical_finetune":
        if cfg.get("canonical_trust_policy") != CANONICAL_FINETUNE_POLICY_VERSION:
            raise ValueError(
                "canonical_finetune checkpoint에 canonical_finetune_v1 trust policy가 없습니다"
            )
        _enforce_canonical_finetune_policy(cfg)
    elif role == A100_PRETRAIN_SMOKE_ROLE:
        if cfg.get("canonical_trust_policy") != A100_PRETRAIN_SMOKE_POLICY_VERSION:
            raise ValueError(
                "a100_pretrain_smoke checkpoint에 a100_pretrain_smoke_v1 trust policy가 없습니다"
            )
        _enforce_a100_pretrain_smoke_policy(cfg)
    if role in {
        "canonical_pretrain",
        "canonical_finetune",
        A100_PRETRAIN_SMOKE_ROLE,
    }:
        if cfg.get("determinism_policy") != CANONICAL_DETERMINISM_POLICY:
            raise ValueError(
                "canonical checkpoint의 determinism_policy가 승인 정책과 다릅니다"
            )


def _finalize_training_metadata(cfg: dict) -> None:
    """계약 SHA 전에 결정 가능한 학습 파생값을 단 한 번 materialize한다."""

    from .dsp.secondary_path import load_secondary_path
    from .dsp.timing import BandPlan, PlantSettle, handoff_samples_from_config

    data = cfg.get("data") or {}
    duct = cfg.get("duct") or {}
    secondary_value = (duct.get("secondary_path") or {}).get("npz")
    if not secondary_value:
        raise ValueError("duct.secondary_path.npz가 없어 학습 metadata를 확정할 수 없습니다")
    secondary = load_secondary_path(_resolve_path(secondary_value))
    sample_rate = int(data["sample_rate"])
    band = BandPlan.resolve(
        plant_trusted_band_hz=secondary.trusted_band_hz(),
        duct_cfg=duct,
        sample_rate=sample_rate,
    ).optimize.as_tuple()
    derived = {
        "digital_reference_lead_samples": int(
            data.get("digital_reference_lead_samples", 0)
        ),
        "physics_status": (
            "acoustic_rir_training"
            if str(data.get("reference_mode", "digital")) != "digital"
            else (
                "measured_primary_path"
                if str(data.get("digital_primary_path_mode", "rir_surrogate"))
                == "measured"
                else f"{data.get('digital_primary_path_mode', 'rir_surrogate')}_representation_pretrain"
            )
        ),
        "trusted_band_hz": [float(value) for value in band],
        "best_metric_key": "nmse_trusted_cvar_db",
        "loss_start_sample": int(
            PlantSettle.derive(
                secondary_delay_samples=int(secondary.delay_samples),
                handoff_samples=handoff_samples_from_config(duct),
                fir_taps=int(secondary.fir.size),
                sample_rate=sample_rate,
            ).samples
        ),
        "loss_selection_sha256": loss_selection_sha256(cfg.get("loss") or {}),
    }
    if str(cfg.get("experiment_role", "")) in {
        "canonical_pretrain",
        "canonical_finetune",
        A100_PRETRAIN_SMOKE_ROLE,
    }:
        derived["determinism_policy"] = copy.deepcopy(
            CANONICAL_DETERMINISM_POLICY
        )
    if cfg.get("required_world_size") == 1:
        derived["nmse_cvar_scope"] = "global"
    for key, value in derived.items():
        declared = cfg.get(key)
        if declared is not None and declared != value:
            raise ValueError(
                f"파생 학습 metadata {key}를 override할 수 없습니다: "
                f"configured={declared!r}, derived={value!r}"
            )
        cfg[key] = value


def load_runtime_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = load_yaml(path)
    if overrides:
        # 참조 경로 자체(hardware_config/duct_config)를 바꿀 수 있도록 먼저 적용한다.
        cfg = apply_overrides(cfg, overrides)
    cfg["hardware"] = load_yaml(cfg["hardware_config"])
    cfg["duct"] = load_yaml(cfg["duct_config"])
    if overrides:
        # 로드된 하위 설정도 CLI에서 재현 가능하게 바꿀 수 있어야 한다.
        # 이 두 번째 적용이 없으면 ``--set hardware.audio.block_size=512`` 같은
        # 런타임 조정은 위의 참조 파일 로드에서 조용히 사라진다.
        cfg = apply_overrides(cfg, overrides)
    return cfg


def _propagate_training_timing_contract(cfg: dict) -> None:
    """실제 P(z) 연산에서 유도한 학습 시간축 계약을 resolved config에 기록한다.

    ``recorded_lead_mode=timeline`` 은 compact FIR의 최대 탭까지 포함한 합성 총
    선행량을 알아야 한다. YAML의 벌크 지연만 복사해서는 그 값을 만들 수 없으므로,
    설정 해석 경계에서 실제 P(z)를 읽고 :class:`TrainingTimingContract` 한 벌을 만든다.
    """

    data = cfg.get("data")
    duct = cfg.get("duct")
    if not isinstance(data, dict) or not isinstance(duct, dict):
        return
    if str(data.get("reference_mode", "digital")) != "digital":
        data.pop("training_timing_contract", None)
        return
    mode = str(data.get("digital_primary_path_mode", "rir_surrogate"))
    if mode == "rir_surrogate":
        # RIR variant마다 peak 위치가 달라 하나의 계약으로 표현할 수 없다. recorded
        # timeline과 함께 쓰면 RecordedANCDataset이 fail-closed 한다.
        data.pop("training_timing_contract", None)
        return

    from .data.primary_path import resolve_digital_primary_path
    from .dsp.secondary_path import load_secondary_path
    from .dsp.timing import PlantDelays, TrainingTimingContract

    secondary_cfg = duct.get("secondary_path") or {}
    secondary_path = secondary_cfg.get("npz")
    if not secondary_path:
        raise ValueError("duct.secondary_path.npz가 없어 training timing을 유도할 수 없습니다")
    secondary = load_secondary_path(_resolve_path(secondary_path))
    primary, _ = resolve_digital_primary_path(
        data, duct, int(data["sample_rate"]), secondary
    )
    if primary is None:  # pragma: no cover - mode 분기 방어
        raise ValueError("compact P(z)가 없어 training timing을 유도할 수 없습니다")
    delays = PlantDelays.from_config(
        duct_cfg=duct,
        secondary_delay_samples=int(secondary.delay_samples),
        primary_delay_samples=int(primary.delay_samples),
        sample_rate=int(data["sample_rate"]),
    )
    contract = TrainingTimingContract.derive(
        primary_fir=primary.fir,
        plant_delays=delays,
    )
    configured = data.get("digital_reference_lead_samples")
    derived_lead = int(contract.digital_reference_lead_samples)
    if configured is not None and int(configured) != derived_lead:
        raise ValueError(
            "digital_reference_lead_samples가 PlantDelays.lead()와 다릅니다: "
            f"configured={int(configured)}, derived={derived_lead}"
        )
    # digital compact-P 모드에서 lead의 유일한 생성자는 PlantDelays/TimingContract다.
    # YAML literal이 없을 때 0으로 간주하면 올바른 양의 lead도 조용히 거부한다.
    data["digital_reference_lead_samples"] = derived_lead
    declared = data.get("training_timing_contract")
    if declared is not None:
        declared_contract = TrainingTimingContract.model_validate(declared)
        if declared_contract != contract:
            raise ValueError(
                "data.training_timing_contract가 실제 P(z)와 다릅니다 — derived config를 "
                "손으로 덮어쓰지 마세요"
            )
    data["training_timing_contract"] = contract.model_dump()
    declared_delay = data.get("d_noise_delay_samples")
    if declared_delay is not None and int(declared_delay) != int(primary.delay_samples):
        raise ValueError(
            "data.d_noise_delay_samples가 TrainingTimingContract의 primary delay와 "
            "다릅니다"
        )
    # recorded timeline 소비처의 compatibility alias도 contract에서만 투영한다.
    data["d_noise_delay_samples"] = int(primary.delay_samples)


def _validate_resolved_training_contract(cfg: dict) -> None:
    """override 적용 **후** digital P/S/lead/시간축의 내부 모순을 차단한다.

    official provenance·반복 일관성·대역 품질은 readiness에서 게이트별로
    보고해야 한다. 여기서는 그 보고를 시작하기도 전에 확정할 수 있는
    **하나의 resolved config 안의 모순**만 예외로 막는다.
    """

    data = cfg.get("data") or {}
    duct = cfg.get("duct") or {}
    reference_mode = str(data.get("reference_mode", "digital"))
    primary_mode = str(data.get("digital_primary_path_mode", "rir_surrogate"))
    if (
        bool(cfg.get("require_measured_primary_path", False))
        and reference_mode == "digital"
        and primary_mode != "measured"
    ):
        raise ValueError(
            "require_measured_primary_path=true인 학습은 override 적용 후 "
            "data.digital_primary_path_mode=measured여야 합니다"
        )
    if bool(cfg.get("require_measured_primary_path", False)) and not bool(
        cfg.get("contract_run_dir", False)
    ):
        raise ValueError(
            "공식 fine-tune은 contract_run_dir=true여야 합니다 — 구형 고정 run "
            "경로의 last.pt와 섞을 수 없습니다"
        )
    if bool(cfg.get("require_recorded_manifest", False)) and str(
        data.get("recorded_sampling", "")
    ) != "family_lineage_session_balanced":
        raise ValueError(
            "공식 recorded fine-tune은 "
            "data.recorded_sampling=family_lineage_session_balanced여야 합니다"
        )
    if reference_mode != "digital" or primary_mode == "rir_surrogate":
        return

    from .data.primary_path import resolve_digital_primary_path
    from .dsp.invariants import ABSOLUTE_OBJECTIVE_BAND_HZ
    from .dsp.secondary_path import load_secondary_path
    from .dsp.timing import FrequencyBand, PlantDelays, TrainingTimingContract

    sample_rate = int(data["sample_rate"])
    secondary_value = (duct.get("secondary_path") or {}).get("npz")
    if not secondary_value:
        raise ValueError("duct.secondary_path.npz가 없어 P/S/lead를 검증할 수 없습니다")
    secondary = load_secondary_path(_resolve_path(secondary_value))
    if int(secondary.sample_rate) != sample_rate:
        raise ValueError(
            f"S(z) sample rate {secondary.sample_rate} != 학습 sample rate {sample_rate}"
        )
    primary, _ = resolve_digital_primary_path(data, duct, sample_rate, secondary)
    if primary is None:  # pragma: no cover - mode 분기 방어
        raise ValueError("compact P(z)가 없어 resolved timing을 검증할 수 없습니다")

    delays = PlantDelays.from_config(
        duct_cfg=duct,
        secondary_delay_samples=int(secondary.delay_samples),
        primary_delay_samples=int(primary.delay_samples),
        sample_rate=sample_rate,
    )
    derived_lead = int(delays.lead().samples)
    configured_lead = int(data.get("digital_reference_lead_samples", 0))
    if configured_lead != derived_lead:
        raise ValueError(
            "digital_reference_lead_samples가 PlantDelays.lead()와 다릅니다: "
            f"configured={configured_lead}, derived={derived_lead} "
            f"(P={primary.delay_samples}, S={secondary.delay_samples}, "
            f"handoff={delays.handoff_samples})"
        )

    actual_contract = TrainingTimingContract.derive(
        primary_fir=primary.fir,
        plant_delays=delays,
    )
    resolved_contract = TrainingTimingContract.from_data_config(data)
    if resolved_contract != actual_contract:
        raise ValueError(
            "data.training_timing_contract가 resolved P(z)/PlantDelays와 다릅니다"
        )

    # 절대 목표를 요구하는 fine-tune 설정은 CLI override로 대역을 줄일 수
    # 없다. P/S 아티팩트가 그 대역을 실제로 입증했는지는 readiness에서
    # official path gate로 따로 판정한다.
    if bool(cfg.get("require_measured_primary_path", False)):
        absolute = FrequencyBand.parse(
            ABSOLUTE_OBJECTIVE_BAND_HZ, name="absolute objective"
        )
        configured = FrequencyBand.parse(
            (cfg.get("readiness") or {}).get("required_path_band_hz"),
            name="readiness.required_path_band_hz",
        )
        if configured != absolute:
            raise ValueError(
                "readiness.required_path_band_hz는 절대 목표 대역과 정확히 "
                f"같아야 합니다: configured={configured.as_tuple()}, "
                f"required={absolute.as_tuple()}"
            )


def validate_duct(duct: dict) -> list[str]:
    """duct.yaml 의 미기입(null) 항목을 경고 목록으로 반환 (치명 오류는 예외)."""
    warnings: list[str] = []
    positions = duct.get("positions_m", {})
    for name in ("noise_speaker", "reference_mic", "cancel_speaker", "error_mic"):
        if positions.get(name) is None:
            warnings.append(f"duct.yaml positions_m.{name} 이 비어 있습니다 — 시뮬레이션 정확도에 영향")
    digital = duct.get("digital_reference", {})
    if digital.get("primary_path_npz") is None:
        warnings.append(
            "duct.yaml digital_reference.primary_path_npz 미실측 — canonical digital "
            "학습은 시작할 수 없습니다"
        )
    for w in warnings:
        print(f"[duct.yaml 경고] {w}")
    return warnings


def duct_distance_samples(duct: dict, a: str, b: str, sample_rate: int) -> int:
    """두 장비 위치 간 음향 전파 지연(샘플). a, b는 positions_m 키."""
    pos = duct["positions_m"]
    if pos.get(a) is None or pos.get(b) is None:
        raise ValueError(f"duct.yaml positions_m 에 {a}/{b} 값이 필요합니다")
    c = float(duct["duct"]["speed_of_sound_mps"])
    dist = abs(float(pos[a]) - float(pos[b]))
    return int(round(sample_rate * dist / c))


def default_d_noise_delay(duct: dict, sample_rate: int, s_path_delay: int) -> int:
    """digital-ref 1차경로 순수지연 기본값 (미실측 시).

    소음(ch0)과 상쇄(ch1)는 같은 USB 출력 장치를 쓰므로 전기/버퍼 지연이 공통이다.
    측정된 S(z) 지연 = 공통지연 + t_ac(CS→ERR) 이므로,
        D_noise ≈ s_path_delay − t_ac(CS→ERR) + t_ac(NS→ERR)
    (근거: docs/01_physics_limits.md, 교차검증 C2)
    """
    t_cs_err = duct_distance_samples(duct, "cancel_speaker", "error_mic", sample_rate)
    t_ns_err = duct_distance_samples(duct, "noise_speaker", "error_mic", sample_rate)
    return int(s_path_delay - t_cs_err + t_ns_err)
