"""YAML 설정 로드/병합/검증.

모든 스크립트는 이 모듈을 통해 설정을 읽는다. 설정 파일 간 참조
(train_*.yaml 의 model_config / data_config / duct_config)는 여기서 해석한다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from .model_input import (
    canonical_stage1_model_input_payload,
    resolve_stage1_model_input_contract,
)
from .train.a100_pretrain_smoke import A100_PRETRAIN_SMOKE_ROLE

# 저장소 루트 (src/deep_anc/config.py 기준 두 단계 위)
REPO_ROOT = Path(__file__).resolve().parents[2]

# 3-스레드 런타임의 콜백↔추론 핸드오프(1 hop) — 학습 플랜트 지연에 가산되는 기본값.
# duct.yaml secondary_path.handoff_extra_samples 가 명시되면 그 값을 쓰고,
# 모든 소비처(.get 기본값)는 이 상수를 공유한다 (감사 L10 — 기본값 분기 금지).
DEFAULT_HANDOFF_SAMPLES = 256

CANONICAL_FINETUNE_POLICY_VERSION = "canonical_finetune_v1"
CANONICAL_PRETRAIN_POLICY_VERSION = "canonical_pretrain_v1"
CANONICAL_STAGE1_MODEL_INPUT_CONFIG = "configs/stage1_ref_only_input.yaml"
CANONICAL_STAGE1_MODEL_INPUT = canonical_stage1_model_input_payload()
A100_PRETRAIN_SMOKE_POLICY_VERSION = "a100_pretrain_smoke_v1"
CANONICAL_PRIMARY_SEED = 20260803
CANONICAL_SECONDARY_SEED = 20260903
PRETRAIN_DERIVATIVE_STRICT_ROLES = frozenset({"loss_pilot", "measured_probe"})
"""canonical 선택 증거를 만드는 단일-GPU·결정론 derivative 역할."""
# 이 두 값은 campaign ledger가 다시 검증하는 선택 증거의 길이이면서, GPU를
# 잘못된 role/config에 쓰기 전에 차단하는 config admission 계약이다. campaign_evidence가
# config를 import하므로 여기서 한 번만 정의하고 그 모듈은 이를 재수출한다.
CANONICAL_LOSS_PILOT_STEPS = 20_000
CANONICAL_MEASURED_PROBE_STEPS = 5_000
CANONICAL_PRETRAIN_DERIVATIVE_STEPS = {
    "loss_pilot": CANONICAL_LOSS_PILOT_STEPS,
    "measured_probe": CANONICAL_MEASURED_PROBE_STEPS,
}
CANONICAL_RECORDED_MANIFEST = "data/manifests/recorded_regrouped.jsonl"
CANONICAL_RECORDED_RATIO = 0.7
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
# YAML의 시작값일 뿐 canonical 승인값이 아니다. alpha별 approved G0 model-output
# gradient calibration이 현재 λ의 share 0.2–0.4를 통과해야 그 alpha 후보의
# 20k/5k를 열 수 있다. 범위 밖이면 receipt의 추천값으로 config를 바꾸고 해당
# alpha의 G0부터 다시 실행한다.
CANONICAL_DNH_BOOTSTRAP_LAMBDA = 0.00075
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
CANONICAL_STAGE1_NONLINEAR = {
    "sef_eta_choices": [10.0],
    "drive_range": [1.0, 1.0],
    "hardclip_prob": 0.0,
}
CANONICAL_STAGE1_PLANT_PERTURBATION = {
    "delay_jitter_range": [0, 0],
    "gain_tilt_db_per_octave": [0, 0],
    "gain_db": [0, 0],
    "allpass_perturb": False,
}
CANONICAL_CLOSED_LOOP_DATA = {
    "feedback_delay_samples": [512, 1024],
    "warmup_seconds": 0.25,
    "unroll_group_frames": 4,
}
# 공개 corpus와 strict P/S를 통과한 Stage-1 분포를 한 곳에서 고정한다. experiment
# contract SHA만 다르면 된다는 해석은 충분하지 않다. canonical이라는 이름으로
# manifest/RIR/plant 증강을 바꾸면 pilot·ledger·100k가 서로 다른 문제를 풀게 된다.
CANONICAL_DATA_DISTRIBUTION = {
    "sample_rate": 48000,
    "segment_seconds": 1.5,
    "reference_mode": "digital",
    "model_input_contract": CANONICAL_STAGE1_MODEL_INPUT,
    "recorded_lead_mode": "timeline",
    "recorded_sampling": "family_plant_domain_component_session_balanced",
    "source_mix_ratio": CANONICAL_SOURCE_MIX_RATIO,
    "recorded_augment": CANONICAL_RECORDED_AUGMENT,
    "noise_manifest_dir": "data/manifests/canonical_v4",
    "rir_bank": "data/rir_bank/duct_rirs_v1.npz",
    "level_dbfs": [-45, -20],
    "snr_mic_noise_db": [5, 30],
    "dc_hum_prob": 0.2,
    "nonlinear": CANONICAL_STAGE1_NONLINEAR,
    "plant_perturbation": CANONICAL_STAGE1_PLANT_PERTURBATION,
    # stage=open_loop이어도 dataset은 이 feedback-delay 범위로 err input의
    # timeline을 만든다. 이름만 보고 override를 허용하면 다른 학습분포가 된다.
    "closed_loop": CANONICAL_CLOSED_LOOP_DATA,
}
# measured 70:30 stream을 소비하는 ``measured_probe``와
# ``canonical_finetune``은 init의 *완료 방식*만 다르다. P/S, recorded population,
# coverage, lineage와 달성 가능 상한은 같은 Stage-1 admission을 통과해야 한다.
# 이 21개 키를 role별로 복사해 두면 probe가 coverage/source-pool을 생략하거나 더 낮은
# consistency로 실행될 수 있으므로 여기 한 벌만 권위로 둔다.
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
    "recorded_subband_coverage_report_dir": (
        "results/data_audit/recorded_subband_coverage"
    ),
    "min_source_err_coherence": 0.60,
    "min_ref_err_coherence": 0.60,
    "recorded_source_pool_csv": [
        "data/source_pool_v2/sources.csv",
        "data/source_pool/sources.csv",
    ],
    "max_measured_delay_mismatch_samples": 64,
    "min_delay_crosscheck_sessions": 8,
    "max_init_lead_mismatch_samples": 16,
    "require_completed_init_checkpoint": True,
    "max_init_best_metric_db": 0.0,
    "allowed_init_physics_statuses": [
        "secondary_surrogate_representation_pretrain"
    ],
}
# ``measured_probe``는 surrogate pilot의 학습 곡선을 measured P에서 한 번 더
# 보는 synthetic-only 진단이 아니다. 선택된 20k pilot의 weight를 시작점으로
# canonical recorded train 70% + synthetic 30%를 정확히 5k step 학습하는 짧은
# measured fine-tune이다. 역할 이름만 바꿔 recorded stream/init을 빠뜨리는 실수를
# 막기 위해 role 해석 시 아래 필드를 resolved config에 물질화한다.
CANONICAL_MEASURED_PROBE_POLICY = {
    "require_measured_primary_path": True,
    "require_init_checkpoint": True,
    # 20k pilot은 init_eligible canonical run이 아니므로 completion receipt 발급
    # 대상이 아니다. 대신 campaign ledger가 selected pilot best.pt SHA와 probe의
    # init bytes를 직접 결속한다.
    "require_init_completion_receipt": False,
    "require_recorded_manifest": True,
    "required_init_experiment_role": "loss_pilot",
    "require_init_eligible": False,
    "recorded_manifest": CANONICAL_RECORDED_MANIFEST,
    "recorded_ratio": CANONICAL_RECORDED_RATIO,
    "readiness": CANONICAL_FINETUNE_READINESS_POLICY,
}
CANONICAL_FINETUNE_POLICY = {
    "experiment_role": "canonical_finetune",
    "init_eligible": False,
    "require_measured_primary_path": True,
    "require_init_checkpoint": True,
    "require_recorded_manifest": True,
    "required_init_experiment_role": "canonical_pretrain",
    "require_init_eligible": True,
    "require_init_completion_receipt": True,
    "contract_run_dir": True,
    "required_world_size": 1,
    "stage": "open_loop",
    "model_config": "configs/model_tiny.yaml",
    "data_model_input_contract_config": CANONICAL_STAGE1_MODEL_INPUT_CONFIG,
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
    "recorded_manifest": CANONICAL_RECORDED_MANIFEST,
    "recorded_ratio": CANONICAL_RECORDED_RATIO,
}


def canonical_recorded_manifest_for_data(data: dict) -> str:
    """data generation 선언에서 권위 recorded manifest를 fail-closed 유도한다.

    generation 선언이 전혀 없을 때만 immutable legacy parent82 manifest를 반환한다.
    generation path/SHA 중 하나라도 선언되면 둘 다 exact여야 하며, 같은 generation
    directory의 combined ``recorded.jsonl``만 선택한다.
    """

    if not isinstance(data, dict):
        raise ValueError("canonical recorded data config는 mapping이어야 합니다")
    generation = data.get("recorded_generation")
    generation_sha = data.get("recorded_generation_sha256")
    if generation in (None, "") and generation_sha in (None, ""):
        return CANONICAL_RECORDED_MANIFEST
    if not isinstance(generation, str) or not isinstance(generation_sha, str):
        raise ValueError(
            "recorded_generation path와 recorded_generation_sha256은 함께 선언해야 합니다"
        )
    generation_path = Path(generation)
    root = Path("data/manifests/recorded_generations")
    try:
        generation_relative = generation_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("recorded_generation canonical root가 아닙니다") from exc
    if (
        generation_path.is_absolute()
        or len(generation_relative.parts) != 2
        or generation_relative.name != "generation.json"
    ):
        raise ValueError("recorded_generation path가 exact generation.json 경로가 아닙니다")
    generation_id = generation_relative.parts[0]
    if (
        re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", generation_id) is None
        or len(generation_sha) != 64
        or any(character not in "0123456789abcdef" for character in generation_sha)
    ):
        raise ValueError("recorded_generation id/SHA-256 선언이 유효하지 않습니다")
    return (root / generation_id / "recorded.jsonl").as_posix()


def _canonical_recorded_manifest_binding(cfg: dict) -> bool:
    """Legacy 82 또는 transfer-결속된 generation 101 manifest 선언만 허용한다."""

    try:
        expected = canonical_recorded_manifest_for_data(cfg.get("data") or {})
    except ValueError:
        return False
    return cfg.get("recorded_manifest") == expected


CANONICAL_PRETRAIN_POLICY = {
    "experiment_role": "canonical_pretrain",
    "init_eligible": True,
    "stage": "open_loop",
    "model_config": "configs/model_tiny.yaml",
    "data_model_input_contract_config": CANONICAL_STAGE1_MODEL_INPUT_CONFIG,
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


def _bind_stage1_model_input_contract(cfg: dict) -> None:
    """선택적 Stage-1 입력 계약을 materialize하고 digest를 결속한다.

    별도 config에 두어 ``data_sim.yaml``의 legacy 사용자는 과거 ERR-context와
    channel-dropout 분포를 유지한다. Canonical policy만 exact 파일과 resolved
    payload를 요구한다.
    """

    data = cfg.get("data")
    if not isinstance(data, dict):
        return
    contract = resolve_stage1_model_input_contract(data)
    declared_sha = cfg.get("model_input_contract_sha256")
    if contract is None:
        if declared_sha is not None:
            raise ValueError(
                "model_input_contract_sha256가 있지만 data.model_input_contract가 없습니다"
            )
        return
    digest = contract.digest()
    if declared_sha is not None and declared_sha != digest:
        raise ValueError(
            "model_input_contract_sha256가 resolved data input contract와 다릅니다"
        )
    cfg["model_input_contract_sha256"] = digest


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
    model_input_config = cfg.get("data_model_input_contract_config")
    if model_input_config not in (None, ""):
        if not isinstance(model_input_config, str):
            raise ValueError("data_model_input_contract_config는 경로 문자열이어야 합니다")
        if "model_input_contract" in cfg["data"]:
            raise ValueError(
                "data config와 별도 input-contract config에 중복 선언이 있습니다"
            )
        cfg["data"]["model_input_contract"] = load_yaml(model_input_config)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    _bind_stage1_model_input_contract(cfg)
    if declared_canonical_finetune:
        _enforce_canonical_finetune_policy(cfg)
    if declared_canonical_pretrain:
        role = str(cfg.get("experiment_role", ""))
        if role == "canonical_pretrain":
            _enforce_canonical_pretrain_policy(cfg)
        elif role == A100_PRETRAIN_SMOKE_ROLE:
            _enforce_a100_pretrain_smoke_policy(cfg)
        else:
            if role == "measured_probe":
                _materialize_measured_probe_policy(cfg)
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
    } | PRETRAIN_DERIVATIVE_STRICT_ROLES:
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
                "canonical 학습/선택 evidence config를 stamp하려면 "
                "data.bootstrap_receipt와 외부 bootstrap_receipt_sha256이 필요합니다"
            )
        from .data.transfer_contract import bind_recorded_transfer_config

        bind_recorded_transfer_config(cfg["data"], repo_root=REPO_ROOT)
        _materialize_bound_recorded_manifest(cfg)
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
        _materialize_bound_recorded_manifest(cfg)
    if resolved_role in {
        "canonical_pretrain",
        "canonical_finetune",
        A100_PRETRAIN_SMOKE_ROLE,
    } | PRETRAIN_DERIVATIVE_STRICT_ROLES:
        _require_canonical_rir_bank(cfg)
    if str(cfg.get("experiment_role", "")) in {
        "canonical_pretrain",
        "canonical_finetune",
        A100_PRETRAIN_SMOKE_ROLE,
    } | PRETRAIN_DERIVATIVE_STRICT_ROLES:
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


def _materialize_bound_recorded_manifest(cfg: dict) -> None:
    """검증된 schema v1/v2 transfer에서 recorded 역할의 manifest를 투영한다."""

    role = str(cfg.get("experiment_role", ""))
    if role not in {"measured_probe", "canonical_finetune"}:
        return
    expected = canonical_recorded_manifest_for_data(cfg.get("data") or {})
    declared = cfg.get("recorded_manifest")
    if declared not in (None, "", CANONICAL_RECORDED_MANIFEST, expected):
        raise ValueError(
            "recorded_manifest override가 검증된 transfer generation과 다릅니다: "
            f"declared={declared!r}, expected={expected!r}"
        )
    cfg["recorded_manifest"] = expected
    # 초기 YAML policy 검사는 아직 schema를 모르는 legacy placeholder에서 실행된다.
    # transfer 결속 뒤 dynamic manifest를 물질화한 상태를 다시 검사해 checkpoint에
    # 저장되는 최종 semantics도 동일한 policy 경로를 통과시킨다.
    if role == "canonical_finetune":
        _enforce_canonical_finetune_policy(cfg)
    else:
        _enforce_pretrain_derivative_policy(cfg)


def _require_canonical_rir_bank(cfg: dict) -> None:
    """공식 역할에서 SynthANCDataset의 즉석 32-RIR fallback을 차단한다."""

    value = (cfg.get("data") or {}).get("rir_bank")
    if value != CANONICAL_DATA_DISTRIBUTION["rir_bank"]:
        raise ValueError(
            "canonical 학습은 data/rir_bank/duct_rirs_v1.npz만 사용할 수 있습니다"
        )
    target = _resolve_path(str(value)).absolute()
    try:
        relative = target.relative_to(REPO_ROOT.absolute())
    except ValueError as exc:
        raise ValueError("canonical RIR bank는 저장소 내부 regular file이어야 합니다") from exc
    cursor = REPO_ROOT.absolute()
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise ValueError(f"canonical RIR bank 경로에는 symlink를 허용하지 않습니다: {cursor}")
    from .train.evaluation_contract import snapshot_regular_file

    try:
        snapshot = snapshot_regular_file(target)
    except ValueError as exc:
        raise ValueError(f"canonical RIR bank regular file이 필요합니다: {target}") from exc
    if not snapshot.content:
        raise ValueError(f"canonical RIR bank가 비었습니다: {target}")
    # regular file이라는 이유만으로 손상된 NPZ를 허용하면 SynthANCDataset이 OSError를
    # 잡아 즉석 RIR로 되돌아간다. pathname을 다시 열지 않고 위 snapshot bytes 자체를
    # 검사해 세 경로 bank가 동일한 유한 2-D shape인지까지 닫는다.
    import io

    import numpy as np

    try:
        with np.load(io.BytesIO(snapshot.content), allow_pickle=False) as bank:
            arrays = [np.asarray(bank[key]) for key in ("p_ref", "p_err", "f_fb")]
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"canonical RIR bank NPZ가 손상됐습니다: {target}") from exc
    shapes = {array.shape for array in arrays}
    if (
        len(shapes) != 1
        or any(array.ndim != 2 or array.size < 1 for array in arrays)
        or any(not bool(np.isfinite(array).all()) for array in arrays)
    ):
        raise ValueError(
            "canonical RIR bank p_ref/p_err/f_fb는 같은 shape의 유한 2-D 배열이어야 합니다"
        )


def _enforce_canonical_finetune_policy(cfg: dict) -> None:
    """canonical config의 CLI 약화/역할 세탁을 override 적용 뒤 차단한다."""

    mismatches: list[str] = []
    for key, required in CANONICAL_FINETUNE_POLICY.items():
        if key == "recorded_manifest" and _canonical_recorded_manifest_binding(cfg):
            continue
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
    seed = cfg.get("seed")
    prerequisite = cfg.get("second_seed_prerequisite")
    prerequisite_sha = cfg.get("second_seed_prerequisite_sha256")
    if seed == CANONICAL_PRIMARY_SEED:
        if prerequisite not in (None, "") or prerequisite_sha not in (None, ""):
            mismatches.append(
                "seed=20260803 second_seed_prerequisite/path SHA must both be null"
            )
    elif seed == CANONICAL_SECONDARY_SEED:
        prefix = "results/training_prerequisites/second_seed/"
        suffix = "/seed_20260903.json"
        if (
            not isinstance(prerequisite, str)
            or not prerequisite.startswith(prefix)
            or not prerequisite.endswith(suffix)
            or prerequisite.count("/") != 4
        ):
            mismatches.append(
                "seed=20260903 second_seed_prerequisite must be the fixed "
                "results/training_prerequisites/second_seed/<digest>/seed_20260903.json path"
            )
        if (
            not isinstance(prerequisite_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", prerequisite_sha) is None
        ):
            mismatches.append(
                "seed=20260903 second_seed_prerequisite_sha256 must be lowercase SHA-256"
            )
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

    _collect_surrogate_only_stream_mismatches(
        cfg, mismatches, role="canonical_pretrain"
    )


def _collect_surrogate_only_stream_mismatches(
    cfg: dict, mismatches: list[str], *, role: str
) -> None:
    """surrogate pretrain/pilot에 recorded branch가 조용히 섞이지 않게 한다."""

    if cfg.get("recorded_manifest") not in (None, ""):
        mismatches.append(f"{role} recorded_manifest must be null (synthetic-only)")
    if cfg.get("recorded_ratio") not in (None, 0, 0.0):
        mismatches.append(f"{role} recorded_ratio must be 0 (synthetic-only)")
    for key in (
        "require_measured_primary_path",
        "require_init_checkpoint",
        "require_recorded_manifest",
    ):
        if bool(cfg.get(key, False)):
            mismatches.append(f"{role} {key} must be false")


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
    if cfg.get("second_seed_prerequisite") not in (None, "") or cfg.get(
        "second_seed_prerequisite_sha256"
    ) not in (None, ""):
        mismatches.append("second-seed prerequisite must be null for smoke")
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
    if role in PRETRAIN_DERIVATIVE_STRICT_ROLES and cfg.get("required_world_size") != 1:
        raise ValueError(
            f"{role}는 canonical 선택 증거이므로 required_world_size=1이어야 합니다"
        )
    if role in PRETRAIN_DERIVATIVE_STRICT_ROLES:
        mismatches: list[str] = []
        if cfg.get("seed") != CANONICAL_PRIMARY_SEED:
            mismatches.append(
                "loss-selection derivative seed must be exact 20260803"
            )
        if cfg.get("second_seed_prerequisite") not in (None, "") or cfg.get(
            "second_seed_prerequisite_sha256"
        ) not in (None, ""):
            mismatches.append(
                "loss-selection derivative cannot consume a second-seed prerequisite"
            )
        # derivative는 길이/role/primary/recorded stream만 다르고 optimizer,
        # schedule, batch, AMP, logging cadence 등 학습 의미는 canonical pretrain과
        # 정확히 같아야 한다. campaign validator가 checkpoint에서 이 정책을 다시
        # 실행하므로 이름만 derivative인 임의 실험은 ledger 후보가 될 수 없다.
        for key, required in CANONICAL_PRETRAIN_POLICY.items():
            if key in {"experiment_role", "init_eligible"}:
                continue
            if cfg.get(key) != required:
                mismatches.append(
                    f"{key}={cfg.get(key)!r} (required {required!r})"
                )
        expected_steps = CANONICAL_PRETRAIN_DERIVATIVE_STEPS[role]
        try:
            run_until_step = int(cfg.get("run_until_step", -1))
        except (TypeError, ValueError):
            run_until_step = -1
        if run_until_step != expected_steps:
            mismatches.append(
                f"run_until_step={cfg.get('run_until_step')!r} "
                f"(required {expected_steps} for {role})"
            )
        init_ckpt = cfg.get("init_ckpt")
        if role == "loss_pilot" and init_ckpt not in (None, ""):
            mismatches.append("loss_pilot init_ckpt must be null (from-scratch only)")
        if role == "measured_probe" and not isinstance(init_ckpt, str):
            mismatches.append(
                "measured_probe init_ckpt must name the selected 20k pilot best.pt"
            )
        elif role == "measured_probe" and not init_ckpt.strip():
            mismatches.append(
                "measured_probe init_ckpt must name the selected 20k pilot best.pt"
            )
        if role == "loss_pilot":
            _collect_surrogate_only_stream_mismatches(
                cfg, mismatches, role="loss_pilot"
            )
        else:
            for key, required in CANONICAL_MEASURED_PROBE_POLICY.items():
                if key == "recorded_manifest" and _canonical_recorded_manifest_binding(cfg):
                    continue
                if cfg.get(key) != required:
                    mismatches.append(
                        f"{key}={cfg.get(key)!r} "
                        f"(required {required!r} for measured_probe)"
                    )
        _collect_common_training_policy_mismatches(cfg, mismatches, role=role)
        if mismatches:
            raise ValueError(
                f"{role} trust policy는 override로 학습 분포를 약화할 수 없습니다: "
                + "; ".join(mismatches)
            )
    cfg["canonical_trust_policy"] = f"{role}_derivative_v1"


def _materialize_measured_probe_policy(cfg: dict) -> None:
    """role declaration을 정확한 measured 70:30 probe stream으로 해석한다.

    이 함수는 YAML/CLI를 처음 해석하는 경계에서만 호출한다. 저장된 checkpoint를
    다시 검증하는 :func:`validate_canonical_training_policy`는 값을 채우지 않고
    완성된 필드 전체를 요구하므로, 누락 필드가 있는 위조 cfg는 통과할 수 없다.
    """

    mismatches: list[str] = []
    for key, required in CANONICAL_MEASURED_PROBE_POLICY.items():
        if key == "recorded_manifest" and _canonical_recorded_manifest_binding(cfg):
            continue
        if key in cfg and cfg.get(key) != required:
            mismatches.append(
                f"{key}={cfg.get(key)!r} "
                f"(required {required!r} for measured_probe)"
            )
    if mismatches:
        raise ValueError(
            "measured_probe trust policy는 override로 학습 분포를 약화할 수 없습니다: "
            + "; ".join(mismatches)
        )
    for key, required in CANONICAL_MEASURED_PROBE_POLICY.items():
        if key == "recorded_manifest" and _canonical_recorded_manifest_binding(cfg):
            continue
        cfg[key] = copy.deepcopy(required)


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
            "lambda_dnh",
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
        lambda_dnh = loss.get("lambda_dnh")
        if (
            isinstance(lambda_dnh, bool)
            or not isinstance(lambda_dnh, (int, float))
            or not math.isfinite(float(lambda_dnh))
            or float(lambda_dnh) <= 0.0
        ):
            mismatches.append(
                "loss.lambda_dnh는 alpha별 output-gradient calibration을 위한 "
                f"finite 양수여야 합니다: {lambda_dnh!r}"
            )
        pair = (loss.get("nmse_cvar_alpha"), loss.get("lambda_frame"))
        if pair not in CANONICAL_LOSS_GRID:
            mismatches.append(
                "loss alpha×frame이 승인 grid가 아닙니다: "
                f"{pair!r}, allowed={sorted(CANONICAL_LOSS_GRID)}"
            )
    data = cfg.get("data")
    if not isinstance(data, dict):
        mismatches.append("data=<missing mapping>")
        return
    for key, required in CANONICAL_DATA_DISTRIBUTION.items():
        if data.get(key) != required:
            mismatches.append(
                f"data.{key}={data.get(key)!r} (required {required!r})"
            )
    try:
        model_input_contract = resolve_stage1_model_input_contract(data)
    except (TypeError, ValueError) as exc:
        mismatches.append(f"data.model_input_contract invalid: {exc}")
    else:
        if model_input_contract is None:
            mismatches.append("data.model_input_contract=<missing>")
        elif cfg.get("model_input_contract_sha256") != model_input_contract.digest():
            mismatches.append(
                "model_input_contract_sha256가 canonical REF-only payload와 다릅니다"
            )
    # 이 값은 SynthANCDataset에서 누락 tag를 조용히 synthetic으로 대체한다. canonical
    # pretrain/finetune, 선택 pilot/probe, A100 smoke 어디에서도 허용하면 안 된다.
    if bool(data.get("allow_missing_source_manifests", False)):
        mismatches.append(
            "data.allow_missing_source_manifests=true "
            "(canonical source family 누락 대체는 금지)"
        )
    expected_primary_mode = {
        "canonical_pretrain": "secondary_surrogate",
        A100_PRETRAIN_SMOKE_ROLE: "secondary_surrogate",
        "loss_pilot": "secondary_surrogate",
        "measured_probe": "measured",
        "canonical_finetune": "measured",
    }.get(role)
    if expected_primary_mode is not None and data.get(
        "digital_primary_path_mode"
    ) != expected_primary_mode:
        mismatches.append(
            "data.digital_primary_path_mode="
            f"{data.get('digital_primary_path_mode')!r} "
            f"(required {expected_primary_mode!r})"
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
    elif role in PRETRAIN_DERIVATIVE_STRICT_ROLES:
        expected_policy = f"{role}_derivative_v1"
        if cfg.get("canonical_trust_policy") != expected_policy:
            raise ValueError(
                f"{role} checkpoint에 {expected_policy} trust policy가 없습니다"
            )
        _enforce_pretrain_derivative_policy(cfg)
    if role in {
        "canonical_pretrain",
        "canonical_finetune",
        A100_PRETRAIN_SMOKE_ROLE,
    } | PRETRAIN_DERIVATIVE_STRICT_ROLES:
        if cfg.get("determinism_policy") != CANONICAL_DETERMINISM_POLICY:
            raise ValueError(
                "canonical/derivative checkpoint의 determinism_policy가 승인 정책과 다릅니다"
            )


def _finalize_training_metadata(
    cfg: dict, *, repo_root: str | Path = REPO_ROOT
) -> None:
    """계약 SHA 전에 결정 가능한 학습 파생값을 단 한 번 materialize한다."""

    from .dsp.timing import BandPlan, PlantSettle, handoff_samples_from_config
    from .train.criterion_factory import (
        BROADBAND_CRITERION_ROLE,
        bind_criterion_contract,
    )

    data = cfg.get("data") or {}
    duct = cfg.get("duct") or {}
    secondary_value = (duct.get("secondary_path") or {}).get("npz")
    if not secondary_value and cfg.get("broadband_causal_training_authority") is None:
        raise ValueError("duct.secondary_path.npz가 없어 학습 metadata를 확정할 수 없습니다")
    criterion_admission = bind_criterion_contract(cfg, repo_root=repo_root)
    secondary = criterion_admission.secondary
    sample_rate = int(data["sample_rate"])
    if criterion_admission.role == BROADBAND_CRITERION_ROLE:
        band = criterion_admission.trusted_band_hz
        best_metric_key = "nmse_subband_guard_cvar_db"
    else:
        assert secondary is not None
        band = BandPlan.resolve(
            plant_trusted_band_hz=secondary.trusted_band_hz(),
            duct_cfg=duct,
            sample_rate=sample_rate,
        ).optimize.as_tuple()
        best_metric_key = "nmse_trusted_cvar_db"
    derived = {
        "digital_reference_lead_samples": int(
            data.get("digital_reference_lead_samples", 0)
        ),
        "physics_status": (
            "fullband_causal_joint_fir_training"
            if criterion_admission.causal_authority is not None
            else (
                "acoustic_rir_training"
                if str(data.get("reference_mode", "digital")) != "digital"
                else (
                    "measured_primary_path"
                    if str(data.get("digital_primary_path_mode", "rir_surrogate"))
                    == "measured"
                    else f"{data.get('digital_primary_path_mode', 'rir_surrogate')}_representation_pretrain"
                )
            )
        ),
        "trusted_band_hz": [float(value) for value in band],
        "best_metric_key": best_metric_key,
        "loss_start_sample": int(
            criterion_admission.broadband_valid_prefix_samples
            if criterion_admission.causal_authority is not None
            else PlantSettle.derive(
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
    } | PRETRAIN_DERIVATIVE_STRICT_ROLES:
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
    if mode == "causal_joint_v4":
        # joint authority parser가 P/S/fractional delay/handoff/lead를 한 번에
        # 검증한다. 여기서 legacy compact P/S로 다시 유도하면 동일
        # 역할에 두 source가 생긴다.
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
    causal_mode = (
        primary_mode == "causal_joint_v4"
        and isinstance(cfg.get("broadband_causal_training_authority"), dict)
    )
    if (
        bool(cfg.get("require_measured_primary_path", False))
        and reference_mode == "digital"
        and primary_mode != "measured"
        and not causal_mode
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
    if bool(cfg.get("require_recorded_manifest", False)):
        broadband_role = str((cfg.get("loss") or {}).get("schema_version", "")).startswith(
            "broadband_equal_subband_loss_"
        )
        expected_sampling = (
            "family_lineage_session_subband_qualified"
            if broadband_role
            else "family_plant_domain_component_session_balanced"
        )
        if str(data.get("recorded_sampling", "")) != expected_sampling:
            raise ValueError(
                "공식 recorded fine-tune sampler가 criterion 역할과 다릅니다: "
                f"expected={expected_sampling}"
            )
        if broadband_role and not (
            data.get("recorded_broadband_batch_receipt")
            and data.get("recorded_broadband_batch_receipt_sha256")
            and data.get("recorded_broadband_val_batch_receipt")
            and data.get("recorded_broadband_val_batch_receipt_sha256")
        ):
            raise ValueError(
                "광대역 recorded fine-tune에는 train/val ERR subband batch receipt "
                "path/SHA가 모두 필수입니다"
            )
    if causal_mode:
        from .train.criterion_factory import admit_criterion_config

        admission = admit_criterion_config(
            cfg, repo_root=REPO_ROOT, require_bound=True
        )
        if admission.causal_authority is None:
            raise ValueError("causal_joint_v4 mode에 v4 authority admission이 없습니다")
        return
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
