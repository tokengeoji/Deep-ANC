"""Stage-2 2 kHz scratch pretrain/fine-tune의 fail-closed profile 감사기.

이 모듈은 기존 Stage-1 ``TrainConfig``나 generic criterion을 확장하지 않는다.
Stage-2 전용 duct/data/evaluation/training profile의 실제 bytes를 먼저 결속하고,
새 P/S·manifest·criterion receipt·외부 experiment contract·scratch checkpoint가
하나라도 없거나 다른 SHA를 가리키면 학습 진입을 차단한다.

전용 typed P/S/data/criterion adapter가 통과하면 recorded fine-tune artifact나 100k
checkpoint가 없어도 scratch pretrain smoke 권한을 별도로 발행한다. 이 감사기 자체는
오디오, GPU, Trainer, DataLoader, subprocess, run directory를 만들지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ..dsp.stage2_2khz_contract import Stage2TwoKilohertzContract
from .stage2_2khz_pretrain_admission import load_stage2_pretrain_typed_admission


STAGE2_CAMPAIGN_SCHEMA = "stage2_2khz_campaign_profile_v1"
STAGE2_CAMPAIGN_ROLE = "admission_preflight_no_trainer_no_gpu_no_audio"
STAGE2_CAMPAIGN_RESULT_SCHEMA = "stage2_2khz_campaign_preflight_v1"
STAGE2_UNATTESTED_STATUS = "BLOCKED_UNATTESTED_STAGE2_EXECUTION_PROVENANCE"

STAGE2_DUCT_PROFILE_SCHEMA = "stage2_2khz_duct_profile_v1"
STAGE2_DATA_PROFILE_SCHEMA = "stage2_2khz_data_profile_v1"
STAGE2_EVALUATION_POLICY_SCHEMA = "stage2_2khz_evaluation_policy_v1"
STAGE2_TRAINING_PROFILE_SCHEMA = "stage2_2khz_training_profile_v1"
STAGE2_EXTERNAL_CONTRACT_SCHEMA = "stage2_2khz_external_experiment_contract_v2"
STAGE2_CHECKPOINT_BINDING_SCHEMA = "stage2_2khz_checkpoint_binding_v2"
STAGE2_CRITERION_SCHEMA = "stage2_2khz_dedicated_criterion_v1"

_PROFILE_SCHEMAS = frozenset(
    {
        STAGE2_CAMPAIGN_SCHEMA,
        STAGE2_DUCT_PROFILE_SCHEMA,
        STAGE2_DATA_PROFILE_SCHEMA,
        STAGE2_EVALUATION_POLICY_SCHEMA,
        STAGE2_TRAINING_PROFILE_SCHEMA,
    }
)
_PROFILE_ROLES = (
    "duct",
    "data",
    "evaluation",
    "canonical_pretrain",
    "canonical_finetune",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_FAMILIES = ["speech", "music", "environment", "machine"]
_SPLITS = ["train", "val", "test"]
_PHYSICAL_SUBBANDS = [
    [88.3883476483, 150.0],
    [150.0, 300.0],
    [300.0, 600.0],
    [600.0, 1000.0],
    [1000.0, 1600.0],
    [1600.0, 2828.4271247462],
]
_OBJECTIVE_OCTAVES = [125.0, 250.0, 500.0, 1000.0, 2000.0]
_DNH_OCTAVES = [4000.0, 8000.0]
_CHECKPOINT_BINDINGS = [
    "stage2_contract_id",
    "stage2_contract_sha256",
    "primary_path_sha256",
    "secondary_path_sha256",
    "plant_binding_sha256",
    "manifest_bundle_sha256",
    "training_profile_sha256",
    "evaluation_policy_sha256",
    "external_experiment_contract_sha256",
    "a100_environment_sha256",
    "smoke_acceptance_sha256",
    "scratch_pretrain",
]
_PRETRAIN_ARTIFACT_REQUIREMENTS = (
    "new_stage2_primary_secondary_raw_analysis_relative_clock_binding",
    "public_manifest_lineage_frequency_coverage_transfer_receipts",
    "family_component_sampler_and_dnh_gradient_calibration_receipts",
    "clean_canonical_pretrain_external_contract",
)
_FINETUNE_ARTIFACT_REQUIREMENTS = (
    "completed_stage2_scratch_100k_checkpoint_with_embedded_binding",
    "recorded_additions_and_70_30_finetune_transfer_authority",
    "recorded_val_only_checkpoint_selection_and_test_once_receipt",
)

# 전체 campaign blocker 목록은 forensic 호환을 위해 유지하되, scratch pretrain을
# 여는 데 필요한 항목과 100k 이후 fine-tune 항목을 별도 표면에 노출한다. 운영자가
# checkpoint/fine-tune 미완료를 보고 "pretrain도 못 시작한다"고 오인하지 않게 한다.
_PRETRAIN_CHECK_IDS = frozenset(
    {
        "artifact_duct_primary_path",
        "artifact_duct_secondary_path",
        "artifact_duct_plant_binding",
        "artifact_data_manifest_bundle",
        "artifact_data_lineage_receipt",
        "artifact_data_frequency_coverage_receipt",
        "artifact_data_transfer_bootstrap_receipt",
        "artifact_canonical_pretrain_criterion_implementation_receipt",
        "artifact_canonical_pretrain_model_config",
        "external_contract_canonical_pretrain",
        "canonical_pretrain_external_contract_cross_binding",
        "typed_stage2_pretrain_execution",
        "typed_stage2_execution_provenance",
    }
)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} key 집합이 exact하지 않습니다: {actual}")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value or "")
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{label}는 lowercase SHA-256이어야 합니다")
    return text


def _require_exact(value: object, expected: object, *, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label}가 canonical Stage-2 값과 다릅니다: {value!r}")


def _check(condition: bool, *, check_id: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": "PASS" if condition else "BLOCKED",
        "detail": detail,
        **extra,
    }


def _inside_repository(root: Path, raw_path: object, *, label: str) -> Path:
    text = str(raw_path or "")
    if not text:
        raise ValueError(f"{label} path가 비었습니다")
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label}는 repository 내부 상대경로여야 합니다")
    cursor = root
    for part in candidate.parts:
        cursor /= part
        try:
            node = cursor.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(node.st_mode):
            raise ValueError(f"{label} 경로에 symlink가 있습니다: {cursor}")
    return root / candidate


def _snapshot_regular_file(path: Path) -> tuple[bytes, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError(f"artifact를 열 수 없습니다: {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"regular file만 허용합니다: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in stable):
        raise ValueError(f"artifact snapshot 도중 파일이 바뀌었습니다: {path}")
    if stat.S_ISLNK(path.lstat().st_mode):
        raise ValueError(f"artifact symlink는 허용하지 않습니다: {path}")
    content = b"".join(chunks)
    if len(content) != int(after.st_size):
        raise ValueError(f"artifact byte 수가 파일 크기와 다릅니다: {path}")
    return content, _sha256_bytes(content)


def _artifact_intent(value: object, *, label: str) -> tuple[str | None, str | None]:
    entry = _require_exact_keys(value, {"path", "sha256"}, label=label)
    path = entry["path"]
    digest = entry["sha256"]
    if (path is None) != (digest is None):
        raise ValueError(f"{label} path와 sha256은 함께 null이거나 함께 선언돼야 합니다")
    if path is None:
        return None, None
    if not isinstance(path, str):
        raise ValueError(f"{label}.path는 string 또는 null이어야 합니다")
    return path, _require_sha256(digest, label=f"{label}.sha256")


def _snapshot_intent(
    root: Path,
    intent: tuple[str | None, str | None],
    *,
    label: str,
) -> tuple[bytes, str] | None:
    relative, expected = intent
    if relative is None:
        return None
    target = _inside_repository(root, relative, label=label)
    try:
        content, actual = _snapshot_regular_file(target)
    except FileNotFoundError:
        return None
    if actual != expected:
        raise ValueError(f"{label} bytes SHA가 config와 다릅니다")
    return content, actual


def _read_yaml(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(content.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"{label}는 UTF-8 YAML mapping이어야 합니다") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 최상위는 mapping이어야 합니다")
    return payload


def _read_json(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}는 UTF-8 JSON mapping이어야 합니다") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 최상위는 mapping이어야 합니다")
    return payload


def _require_contract(value: object, *, label: str) -> None:
    entry = _require_exact_keys(value, {"id", "sha256"}, label=label)
    contract = Stage2TwoKilohertzContract.canonical()
    if entry["id"] != contract.contract_id or entry["sha256"] != contract.digest():
        raise ValueError(f"{label}가 exact Stage-2 2 kHz contract ID/SHA가 아닙니다")


def _validate_duct_profile(payload: Mapping[str, Any]) -> dict[str, tuple[str | None, str | None]]:
    root = _require_exact_keys(
        dict(payload),
        {
            "schema",
            "role",
            "control_band_contract",
            "sample_rate_hz",
            "block_size",
            "reference_mode",
            "measurement_policy",
            "timing_policy",
            "artifacts",
        },
        label="Stage-2 duct profile",
    )
    _require_exact(root["schema"], STAGE2_DUCT_PROFILE_SCHEMA, label="duct schema")
    _require_exact(root["role"], "stage2_physical_plant_only_no_audio", label="duct role")
    _require_contract(root["control_band_contract"], label="duct contract")
    _require_exact(root["sample_rate_hz"], 48_000, label="duct sample_rate_hz")
    _require_exact(root["block_size"], 256, label="duct block_size")
    _require_exact(root["reference_mode"], "digital", label="duct reference_mode")
    measurement = _require_exact_keys(
        root["measurement_policy"],
        {
            "required_excitation_band_hz",
            "physical_identification_subbands_hz",
            "minimum_subband_consistency",
            "maximum_timing_residual_samples",
            "minimum_independent_epochs_per_role",
            "periodic_repeat_indices_allowed",
            "require_same_capture",
            "require_actual_submitted_pcm",
            "require_raw_analysis_sha_binding",
            "require_xrun_clip_status_slip_zero",
        },
        label="duct measurement_policy",
    )
    _require_exact(
        measurement["required_excitation_band_hz"],
        [80.0, 2828.4271247462],
        label="Stage-2 excitation band",
    )
    _require_exact(
        measurement["physical_identification_subbands_hz"],
        _PHYSICAL_SUBBANDS,
        label="Stage-2 physical subbands",
    )
    _require_exact(measurement["minimum_subband_consistency"], 0.95, label="consistency")
    _require_exact(
        measurement["maximum_timing_residual_samples"], 0.270208, label="timing residual"
    )
    _require_exact(
        measurement["minimum_independent_epochs_per_role"],
        8,
        label="independent epochs per role",
    )
    _require_exact(
        measurement["periodic_repeat_indices_allowed"],
        False,
        label="periodic repeat indices",
    )
    for key in (
        "require_same_capture",
        "require_actual_submitted_pcm",
        "require_raw_analysis_sha_binding",
        "require_xrun_clip_status_slip_zero",
    ):
        _require_exact(measurement[key], True, label=f"measurement_policy.{key}")
    timing = _require_exact_keys(
        root["timing_policy"],
        {
            "handoff_extra_samples",
            "lead_source",
            "manual_lead_allowed",
            "stage1_lead_reuse_allowed",
        },
        label="duct timing_policy",
    )
    _require_exact(timing["handoff_extra_samples"], 256, label="handoff")
    _require_exact(timing["lead_source"], "PlantDelays.lead()", label="lead source")
    _require_exact(timing["manual_lead_allowed"], False, label="manual lead")
    _require_exact(timing["stage1_lead_reuse_allowed"], False, label="Stage-1 lead reuse")
    artifacts = _require_exact_keys(
        root["artifacts"], {"primary_path", "secondary_path", "plant_binding"}, label="duct artifacts"
    )
    return {
        role: _artifact_intent(artifacts[role], label=f"duct artifacts.{role}")
        for role in ("primary_path", "secondary_path", "plant_binding")
    }


def _validate_data_profile(payload: Mapping[str, Any]) -> dict[str, tuple[str | None, str | None]]:
    root = _require_exact_keys(
        dict(payload),
        {
            "schema",
            "role",
            "control_band_contract",
            "required_source_families",
            "required_splits",
            "minimum_independent_groups_per_family_octave",
            "minimum_source_density_ratio",
            "one_point_six_khz_sentinel_band_hz",
            "minimum_independent_groups_per_family_sentinel",
            "minimum_sentinel_target_density_ratio",
            "minimum_native_nyquist_hz",
            "recorded_synthetic_lineage_intersection_required",
            "component_split_allowed",
            "synthetic_fallback_for_missing_manifest_allowed",
            "pretrain_distribution",
            "finetune_distribution",
            "artifacts",
        },
        label="Stage-2 data profile",
    )
    _require_exact(root["schema"], STAGE2_DATA_PROFILE_SCHEMA, label="data schema")
    _require_exact(root["role"], "stage2_lineage_and_frequency_coverage_only", label="data role")
    _require_contract(root["control_band_contract"], label="data contract")
    _require_exact(root["required_source_families"], _FAMILIES, label="source families")
    _require_exact(root["required_splits"], _SPLITS, label="splits")
    _require_exact(root["minimum_independent_groups_per_family_octave"], 4, label="group floor")
    _require_exact(root["minimum_source_density_ratio"], 0.25, label="source density")
    _require_exact(
        root["one_point_six_khz_sentinel_band_hz"],
        [1425.437949, 1795.939277],
        label="data 1.6 kHz sentinel band",
    )
    _require_exact(
        root["minimum_independent_groups_per_family_sentinel"],
        4,
        label="data sentinel group floor",
    )
    _require_exact(
        root["minimum_sentinel_target_density_ratio"],
        0.25,
        label="data sentinel density",
    )
    _require_exact(root["minimum_native_nyquist_hz"], 2828.4271247462, label="native Nyquist")
    _require_exact(root["recorded_synthetic_lineage_intersection_required"], 0, label="lineage intersection")
    _require_exact(root["component_split_allowed"], False, label="component split")
    _require_exact(
        root["synthetic_fallback_for_missing_manifest_allowed"], False, label="synthetic fallback"
    )
    pretrain = _require_exact_keys(
        root["pretrain_distribution"],
        {"recorded_ratio", "public_synthetic_ratio", "family_balanced", "component_balanced"},
        label="pretrain distribution",
    )
    _require_exact(pretrain, {"recorded_ratio": 0.0, "public_synthetic_ratio": 1.0, "family_balanced": True, "component_balanced": True}, label="pretrain distribution")
    finetune = _require_exact_keys(
        root["finetune_distribution"],
        {"recorded_ratio", "public_synthetic_ratio", "family_balanced", "component_balanced", "session_mixing_probability", "lead_jitter_samples"},
        label="fine-tune distribution",
    )
    _require_exact(
        finetune,
        {"recorded_ratio": 0.7, "public_synthetic_ratio": 0.3, "family_balanced": True, "component_balanced": True, "session_mixing_probability": 0.0, "lead_jitter_samples": 0.0},
        label="fine-tune distribution",
    )
    artifacts = _require_exact_keys(
        root["artifacts"],
        {"manifest_bundle", "lineage_receipt", "frequency_coverage_receipt", "transfer_bootstrap_receipt"},
        label="data artifacts",
    )
    return {
        role: _artifact_intent(artifacts[role], label=f"data artifacts.{role}")
        for role in ("manifest_bundle", "lineage_receipt", "frequency_coverage_receipt", "transfer_bootstrap_receipt")
    }


def _validate_evaluation_profile(payload: Mapping[str, Any]) -> None:
    root = _require_exact_keys(
        dict(payload),
        {
            "schema",
            "role",
            "control_band_contract",
            "objective_octaves_hz",
            "low_mid_attenuation_threshold_db",
            "low_mid_threshold_comparator",
            "two_khz_attenuation_threshold_db",
            "two_khz_threshold_comparator",
            "required_statistics",
            "minimum_independent_groups_per_family_octave",
            "minimum_source_density_ratio",
            "one_point_six_khz_sentinel",
            "do_no_harm",
            "runtime_gate",
            "checkpoint_selection",
            "claim_scope",
        },
        label="Stage-2 evaluation profile",
    )
    _require_exact(root["schema"], STAGE2_EVALUATION_POLICY_SCHEMA, label="evaluation schema")
    _require_exact(root["role"], "stage2_single_point_raw_off_on_evaluation", label="evaluation role")
    _require_contract(root["control_band_contract"], label="evaluation contract")
    _require_exact(root["objective_octaves_hz"], _OBJECTIVE_OCTAVES, label="objective octaves")
    _require_exact(root["low_mid_attenuation_threshold_db"], 0.0, label="low/mid threshold")
    _require_exact(root["low_mid_threshold_comparator"], "strictly_greater_than", label="low/mid comparator")
    _require_exact(root["two_khz_attenuation_threshold_db"], 3.0, label="2 kHz threshold")
    _require_exact(root["two_khz_threshold_comparator"], "greater_than_or_equal", label="2 kHz comparator")
    _require_exact(root["required_statistics"], ["family_mean", "family_worst10_mean", "family_cluster_ci95_lower"], label="required statistics")
    _require_exact(root["minimum_independent_groups_per_family_octave"], 4, label="evaluation group floor")
    _require_exact(root["minimum_source_density_ratio"], 0.25, label="evaluation source density")
    _require_exact(
        root["one_point_six_khz_sentinel"],
        {
            "band_hz": [1425.437949, 1795.939277],
            "attenuation_threshold_db": 0.0,
            "comparator": "strictly_greater_than",
            "required_statistics": [
                "family_mean",
                "family_worst10_mean",
                "family_cluster_ci95_lower",
            ],
            "minimum_independent_groups_per_family": 4,
            "minimum_source_density_ratio": 0.25,
        },
        label="1.6 kHz sentinel policy",
    )
    _require_exact(
        root["do_no_harm"],
        {"observation_octaves_hz": _DNH_OCTAVES, "worst10_amplification_limit_db": 1.0, "comparator": "strictly_less_than"},
        label="do-no-harm policy",
    )
    _require_exact(
        root["runtime_gate"],
        {"inference_p99_ms_strictly_less_than": 3.0, "inference_max_ms_strictly_less_than": 5.333333333333333, "deadline_miss": 0, "xrun": 0, "fallback": 0, "ring_drop_add": 0, "sample_slip": 0, "timing_residual_frames": 0, "excess_backlog": 0},
        label="runtime gate",
    )
    _require_exact(
        root["checkpoint_selection"],
        {"all_frequency_family_dnh_runtime_gates_required": True, "primary_order": "maximize_minimum_frequency_family_dnh_runtime_gate_margin", "secondary_order": "maximize_two_khz_family_equal_mean_attenuation_db", "three_db_is_minimum_not_optimization_target": True, "one_point_six_khz_sentinel_runtime_exact_zero_required": True, "test_data_for_selection_allowed": False},
        label="checkpoint selection",
    )
    _require_exact(
        root["claim_scope"],
        {"physical_single_point_only": True, "spatial_quiet_zone_claim_allowed": False, "full_octave_v3_claim_allowed": False},
        label="claim scope",
    )


def _validate_training_profile(
    payload: Mapping[str, Any], *, expected_role: str
) -> dict[str, tuple[str | None, str | None]]:
    expected_keys = {
        "schema", "role", "steps", "seed", "init_eligible", "control_band_contract",
        "initialization", "resume_policy", "criterion", "checkpoint_required_bindings",
    }
    if expected_role == "canonical_pretrain":
        expected_keys.add("execution")
    root = _require_exact_keys(
        dict(payload),
        expected_keys,
        label=f"Stage-2 {expected_role} training profile",
    )
    _require_exact(root["schema"], STAGE2_TRAINING_PROFILE_SCHEMA, label="training schema")
    _require_exact(root["role"], expected_role, label="training role")
    _require_contract(root["control_band_contract"], label="training contract")
    _require_exact(root["seed"], 20260803, label="training seed")
    expected_steps = 100_000 if expected_role == "canonical_pretrain" else 50_000
    _require_exact(root["steps"], expected_steps, label="training steps")
    _require_exact(root["init_eligible"], expected_role == "canonical_pretrain", label="init eligible")
    init = _require_exact_keys(
        root["initialization"],
        {"mode", "checkpoint_path", "checkpoint_sha256", "legacy_checkpoint_allowed", "stage1_checkpoint_allowed"},
        label="initialization",
    )
    expected_mode = "scratch" if expected_role == "canonical_pretrain" else "completed_stage2_scratch_pretrain_weight_only"
    _require_exact(init["mode"], expected_mode, label="initialization mode")
    if (init["checkpoint_path"] is None) != (init["checkpoint_sha256"] is None):
        raise ValueError("initialization checkpoint path/SHA는 함께 선언해야 합니다")
    if expected_role == "canonical_pretrain" and init["checkpoint_path"] is not None:
        raise ValueError("Stage-2 canonical pretrain은 scratch만 허용하며 init checkpoint를 받을 수 없습니다")
    if init["checkpoint_path"] is not None:
        if not isinstance(init["checkpoint_path"], str):
            raise ValueError("fine-tune init checkpoint path는 string이어야 합니다")
        _require_sha256(init["checkpoint_sha256"], label="fine-tune init checkpoint SHA")
    _require_exact(init["legacy_checkpoint_allowed"], False, label="legacy checkpoint")
    _require_exact(init["stage1_checkpoint_allowed"], False, label="Stage-1 checkpoint")
    _require_exact(
        root["resume_policy"],
        {"automatic_resume_allowed": False, "explicit_same_external_contract_only": True, "full_rng_optimizer_scheduler_state_required": True},
        label="resume policy",
    )
    model_config_intent: tuple[str | None, str | None] = (None, None)
    if expected_role == "canonical_pretrain":
        execution = _require_exact_keys(
            root["execution"],
            {
                "model_config",
                "reference_mode",
                "model_input_channels",
                "error_input_mode",
                "batch_size",
                "target_samples",
                "precision",
                "data_pipeline",
                "telemetry",
                "optimizer",
                "schedule",
                "grad_clip_norm",
                "checkpoint_every_steps",
                "smoke_milestones",
                "required_world_size",
                "run_directory_schema",
                "test_split_for_checkpoint_selection_allowed",
            },
            label="Stage-2 pretrain execution",
        )
        model_config_intent = _artifact_intent(
            execution["model_config"], label="Stage-2 model config"
        )
        _require_exact(execution["reference_mode"], "digital", label="Stage-2 reference mode")
        _require_exact(execution["model_input_channels"], 2, label="Stage-2 model input channels")
        _require_exact(execution["error_input_mode"], "zero", label="Stage-2 error input mode")
        _require_exact(execution["batch_size"], 96, label="Stage-2 A100 smoke batch candidate")
        _require_exact(execution["target_samples"], 16_384, label="Stage-2 target samples")
        _require_exact(execution["precision"], "bf16_forward_fp32_loss", label="Stage-2 precision")
        _require_exact(
            execution["data_pipeline"],
            {
                "schema": "stage2_2khz_bounded_prefetch_v1",
                "loader_workers": 14,
                "prefetch_batches_per_worker": 4,
                "bounded_prefetch_batches": 56,
                "source_cache_items": 64,
                "valid_start_candidates_per_source": 64,
                "valid_start_precompute_from_actual_primary_path": True,
                "source_sha256_reverified_before_cache": True,
                "global_step_order_independent_of_worker_count": True,
                "pin_memory": True,
                "non_blocking_h2d": True,
                "cache_race_allowed": False,
            },
            label="Stage-2 bounded data pipeline",
        )
        _require_exact(
            execution["telemetry"],
            {
                "schema": "stage2_2khz_step_telemetry_v1",
                "raw_per_step_required": True,
                "data_wait_ms_required": True,
                "h2d_ms_required": True,
                "compute_step_ms_required": True,
                "gpu_peak_memory_required": True,
                "throughput_required": True,
                "nvidia_smi_sample_every_steps": 10,
                "smoke_thresholds_must_be_declared_from_observation": True,
                "estimated_memory_pass_allowed": False,
            },
            label="Stage-2 raw performance telemetry",
        )
        _require_exact(
            execution["optimizer"],
            {"name": "adamw", "lr": 0.001, "weight_decay": 0.0001, "betas": [0.9, 0.999]},
            label="Stage-2 optimizer",
        )
        _require_exact(
            execution["schedule"],
            {"name": "linear_warmup_cosine", "warmup_steps": 1250, "total_steps": 100_000, "min_lr": 0.00001},
            label="Stage-2 schedule",
        )
        _require_exact(execution["grad_clip_norm"], 5.0, label="Stage-2 grad clip")
        _require_exact(execution["checkpoint_every_steps"], 500, label="Stage-2 checkpoint interval")
        _require_exact(execution["smoke_milestones"], [200, 500], label="Stage-2 smoke milestones")
        _require_exact(execution["required_world_size"], 1, label="Stage-2 world size")
        _require_exact(
            execution["run_directory_schema"],
            "stage2_pretrain_external_sha_seed_v1",
            label="Stage-2 run directory schema",
        )
        _require_exact(
            execution["test_split_for_checkpoint_selection_allowed"],
            False,
            label="Stage-2 test selection",
        )
    criterion = _require_exact_keys(
        root["criterion"],
        {"schema", "generic_stage1_criterion_allowed", "objective_octaves_hz", "one_point_six_khz_sentinel_band_hz", "one_point_six_khz_minimum_attenuation_db", "two_khz_minimum_attenuation_db", "do_no_harm_octaves_hz", "implementation_receipt"},
        label="criterion",
    )
    _require_exact(criterion["schema"], STAGE2_CRITERION_SCHEMA, label="criterion schema")
    _require_exact(criterion["generic_stage1_criterion_allowed"], False, label="generic Stage-1 criterion")
    _require_exact(criterion["objective_octaves_hz"], _OBJECTIVE_OCTAVES, label="criterion octaves")
    _require_exact(
        criterion["one_point_six_khz_sentinel_band_hz"],
        [1425.437949, 1795.939277],
        label="criterion 1.6 kHz sentinel band",
    )
    _require_exact(
        criterion["one_point_six_khz_minimum_attenuation_db"],
        0.0,
        label="criterion 1.6 kHz threshold",
    )
    _require_exact(criterion["two_khz_minimum_attenuation_db"], 3.0, label="criterion 2 kHz threshold")
    _require_exact(criterion["do_no_harm_octaves_hz"], _DNH_OCTAVES, label="criterion DNH octaves")
    _require_exact(root["checkpoint_required_bindings"], _CHECKPOINT_BINDINGS, label="checkpoint bindings")
    initialization_intent: tuple[str | None, str | None]
    if init["checkpoint_path"] is None:
        initialization_intent = (None, None)
    else:
        initialization_intent = (
            str(init["checkpoint_path"]),
            str(init["checkpoint_sha256"]),
        )
    return {
        "criterion_implementation_receipt": _artifact_intent(
            criterion["implementation_receipt"], label=f"{expected_role} criterion receipt"
        ),
        "initialization_checkpoint": initialization_intent,
        "model_config": model_config_intent,
    }


def _validate_external_contract(
    payload: Mapping[str, Any],
    *,
    stage: str,
    profile_sha256: Mapping[str, str],
    plant_sha256: Mapping[str, str],
    manifest_bundle_sha256: str,
    criterion_receipt_sha256: str,
    init_checkpoint_sha256: str | None,
) -> None:
    root = _require_exact_keys(
        dict(payload),
        {"schema", "stage", "artifact_source_commit_sha", "repository_clean_required", "control_band_contract", "profile_sha256", "plant_sha256", "manifest_bundle_sha256", "criterion_receipt_sha256", "initialization_mode", "init_checkpoint_sha256", "scratch_pretrain_origin_required", "legacy_artifacts_allowed", "automatic_resume_allowed", "training_eligible"},
        label=f"{stage} external contract",
    )
    _require_exact(root["schema"], STAGE2_EXTERNAL_CONTRACT_SCHEMA, label="external contract schema")
    _require_exact(root["stage"], stage, label="external contract stage")
    if not _COMMIT_SHA_RE.fullmatch(str(root["artifact_source_commit_sha"])):
        raise ValueError("external contract artifact source commit은 lowercase 40-hex여야 합니다")
    _require_exact(
        root["repository_clean_required"], True, label="external clean checkout"
    )
    _require_contract(root["control_band_contract"], label="external contract Stage-2 contract")
    _require_exact(root["profile_sha256"], dict(profile_sha256), label="external profile SHA binding")
    _require_exact(root["plant_sha256"], dict(plant_sha256), label="external P/S SHA binding")
    _require_exact(root["manifest_bundle_sha256"], manifest_bundle_sha256, label="external manifest SHA")
    _require_exact(root["criterion_receipt_sha256"], criterion_receipt_sha256, label="external criterion SHA")
    expected_mode = "scratch" if stage == "canonical_pretrain" else "completed_stage2_scratch_pretrain_weight_only"
    _require_exact(root["initialization_mode"], expected_mode, label="external initialization mode")
    _require_exact(root["init_checkpoint_sha256"], init_checkpoint_sha256, label="external init checkpoint SHA")
    _require_exact(root["scratch_pretrain_origin_required"], True, label="scratch origin")
    _require_exact(root["legacy_artifacts_allowed"], False, label="legacy artifacts")
    _require_exact(root["automatic_resume_allowed"], False, label="automatic resume")
    _require_exact(root["training_eligible"], True, label="external training eligible")


def _validate_checkpoint_binding(
    payload: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    external_contract_sha256: str,
    plant_sha256: Mapping[str, str],
    manifest_bundle_sha256: str,
    pretrain_profile_sha256: str,
    evaluation_policy_sha256: str,
) -> None:
    root = _require_exact_keys(
        dict(payload),
        {"schema", "checkpoint_sha256", "external_experiment_contract_sha256", "control_band_contract", "plant_sha256", "plant_binding_runtime_sha256", "manifest_bundle_sha256", "training_profile_sha256", "evaluation_policy_sha256", "model_config_sha256", "criterion_receipt_sha256", "sampler_receipt_sha256", "dnh_calibration_receipt_sha256", "a100_environment_sha256", "smoke_acceptance_sha256", "experiment_role", "completed_steps", "init_eligible", "scratch_pretrain", "legacy_origin", "diagnostic_cpu_test", "completion_receipt_sha256"},
        label="Stage-2 checkpoint binding",
    )
    _require_exact(root["schema"], STAGE2_CHECKPOINT_BINDING_SCHEMA, label="checkpoint binding schema")
    _require_exact(root["checkpoint_sha256"], checkpoint_sha256, label="checkpoint bytes SHA")
    _require_exact(root["external_experiment_contract_sha256"], external_contract_sha256, label="checkpoint external contract SHA")
    _require_contract(root["control_band_contract"], label="checkpoint Stage-2 contract")
    _require_exact(root["plant_sha256"], dict(plant_sha256), label="checkpoint P/S SHA binding")
    for key in (
        "plant_binding_runtime_sha256",
        "model_config_sha256",
        "criterion_receipt_sha256",
        "sampler_receipt_sha256",
        "dnh_calibration_receipt_sha256",
        "a100_environment_sha256",
        "smoke_acceptance_sha256",
    ):
        _require_sha256(root[key], label=f"checkpoint {key}")
    _require_exact(root["manifest_bundle_sha256"], manifest_bundle_sha256, label="checkpoint manifest SHA")
    _require_exact(root["training_profile_sha256"], pretrain_profile_sha256, label="checkpoint train config SHA")
    _require_exact(root["evaluation_policy_sha256"], evaluation_policy_sha256, label="checkpoint evaluation policy SHA")
    _require_exact(root["experiment_role"], "canonical_pretrain", label="checkpoint role")
    _require_exact(root["completed_steps"], 100_000, label="checkpoint completed steps")
    _require_exact(root["init_eligible"], True, label="checkpoint init eligible")
    _require_exact(root["scratch_pretrain"], True, label="checkpoint scratch pretrain")
    _require_exact(root["legacy_origin"], False, label="checkpoint legacy origin")
    _require_exact(root["diagnostic_cpu_test"], False, label="checkpoint CPU diagnostic")
    expected_completion_sha = _sha256_bytes(
        _canonical_json(
            {
                "schema": "stage2_2khz_pretrain_completion_receipt_v2",
                "checkpoint_sha256": checkpoint_sha256,
                "external_experiment_contract_sha256": external_contract_sha256,
                "completed_steps": 100_000,
                "init_eligible": True,
                "scratch_pretrain": True,
                "smoke_acceptance_sha256": root["smoke_acceptance_sha256"],
            }
        )
    )
    _require_exact(
        root["completion_receipt_sha256"],
        expected_completion_sha,
        label="checkpoint completion receipt SHA",
    )


def is_stage2_2khz_profile_config(payload: Mapping[str, Any]) -> bool:
    """generic ``train.py``가 Stage-2 profile을 load_train_config 전에 거부한다."""

    return isinstance(payload, Mapping) and payload.get("schema") in _PROFILE_SCHEMAS


def audit_stage2_2khz_campaign(
    payload: Mapping[str, Any], *, repo_root: str | Path
) -> dict[str, Any]:
    """Stage-2 profile과 현재 artifact chain을 읽기 전용으로 감사한다."""

    root_path = Path(repo_root).resolve(strict=True)
    root = _require_exact_keys(
        dict(payload),
        {"schema", "role", "control_band_contract", "profiles", "external_contracts", "canonical_pretrain_checkpoint", "release_policy"},
        label="Stage-2 campaign profile",
    )
    _require_exact(root["schema"], STAGE2_CAMPAIGN_SCHEMA, label="campaign schema")
    _require_exact(root["role"], STAGE2_CAMPAIGN_ROLE, label="campaign role")
    _require_contract(root["control_band_contract"], label="campaign contract")
    _require_exact(
        root["release_policy"],
        {"generic_stage1_trainer_allowed": False, "legacy_init_or_resume_allowed": False, "automatic_resume_allowed": False, "scratch_pretrain_required": True, "run_directory_before_ready_allowed": False, "gpu_before_ready_allowed": False},
        label="Stage-2 release policy",
    )

    profile_entries = _require_exact_keys(root["profiles"], set(_PROFILE_ROLES), label="campaign profiles")
    profile_payloads: dict[str, dict[str, Any]] = {}
    profile_sha256: dict[str, str] = {}
    checks: list[dict[str, Any]] = []
    for role in _PROFILE_ROLES:
        intent = _artifact_intent(profile_entries[role], label=f"profiles.{role}")
        if intent[0] is None:
            raise ValueError(f"profiles.{role}는 null일 수 없습니다")
        snapshot = _snapshot_intent(root_path, intent, label=f"profiles.{role}")
        if snapshot is None:
            raise ValueError(f"profiles.{role} 파일이 없습니다")
        content, digest = snapshot
        profile_payloads[role] = _read_yaml(content, label=f"profiles.{role}")
        profile_sha256[role] = digest
        checks.append(_check(True, check_id=f"profile_bytes_{role}", detail="profile path/bytes SHA exact"))

    duct_artifacts = _validate_duct_profile(profile_payloads["duct"])
    data_artifacts = _validate_data_profile(profile_payloads["data"])
    _validate_evaluation_profile(profile_payloads["evaluation"])
    pretrain_artifacts = _validate_training_profile(profile_payloads["canonical_pretrain"], expected_role="canonical_pretrain")
    finetune_artifacts = _validate_training_profile(profile_payloads["canonical_finetune"], expected_role="canonical_finetune")
    checks.append(_check(True, check_id="profile_semantics", detail="Stage-2 전용 profile semantic exact"))

    artifact_snapshots: dict[str, tuple[bytes, str] | None] = {}
    artifact_intents = {
        **{f"duct.{key}": value for key, value in duct_artifacts.items()},
        **{f"data.{key}": value for key, value in data_artifacts.items()},
        "canonical_pretrain.criterion_implementation_receipt": pretrain_artifacts["criterion_implementation_receipt"],
        "canonical_pretrain.model_config": pretrain_artifacts["model_config"],
        "canonical_finetune.criterion_implementation_receipt": finetune_artifacts["criterion_implementation_receipt"],
        "canonical_finetune.initialization_checkpoint": finetune_artifacts["initialization_checkpoint"],
    }
    for label, intent in artifact_intents.items():
        snapshot = _snapshot_intent(root_path, intent, label=label)
        artifact_snapshots[label] = snapshot
        checks.append(
            _check(
                snapshot is not None,
                check_id=f"artifact_{label.replace('.', '_')}",
                detail="exact artifact bytes present" if snapshot else "artifact path/SHA가 아직 null 또는 파일 부재",
            )
        )

    external = _require_exact_keys(root["external_contracts"], {"canonical_pretrain", "canonical_finetune"}, label="external contracts")
    external_snapshots: dict[str, tuple[bytes, str] | None] = {}
    for stage in ("canonical_pretrain", "canonical_finetune"):
        intent = _artifact_intent(external[stage], label=f"external_contracts.{stage}")
        snapshot = _snapshot_intent(root_path, intent, label=f"external_contracts.{stage}")
        external_snapshots[stage] = snapshot
        checks.append(_check(snapshot is not None, check_id=f"external_contract_{stage}", detail="external contract bytes present" if snapshot else "external contract 미발행"))

    checkpoint_section = _require_exact_keys(root["canonical_pretrain_checkpoint"], {"checkpoint", "binding"}, label="canonical pretrain checkpoint")
    checkpoint_snapshot = _snapshot_intent(root_path, _artifact_intent(checkpoint_section["checkpoint"], label="canonical_pretrain_checkpoint.checkpoint"), label="canonical_pretrain_checkpoint.checkpoint")
    binding_snapshot = _snapshot_intent(root_path, _artifact_intent(checkpoint_section["binding"], label="canonical_pretrain_checkpoint.binding"), label="canonical_pretrain_checkpoint.binding")
    checks.append(_check(checkpoint_snapshot is not None, check_id="canonical_pretrain_checkpoint", detail="scratch 100k checkpoint bytes present" if checkpoint_snapshot else "scratch 100k checkpoint 미완료"))
    checks.append(_check(binding_snapshot is not None, check_id="canonical_pretrain_checkpoint_binding", detail="checkpoint binding bytes present" if binding_snapshot else "checkpoint binding 미발행"))
    fine_init_snapshot = artifact_snapshots[
        "canonical_finetune.initialization_checkpoint"
    ]
    fine_init_matches_campaign_checkpoint = bool(
        fine_init_snapshot is not None
        and checkpoint_snapshot is not None
        and fine_init_snapshot[1] == checkpoint_snapshot[1]
    )
    checks.append(
        _check(
            fine_init_matches_campaign_checkpoint,
            check_id="canonical_finetune_init_is_exact_stage2_scratch_checkpoint",
            detail=(
                "fine-tune weight-only init가 campaign scratch 100k checkpoint bytes와 exact"
                if fine_init_matches_campaign_checkpoint
                else "fine-tune init가 미지정이거나 campaign scratch checkpoint와 다름"
            ),
        )
    )

    required_for_declared_chain = all(snapshot is not None for snapshot in artifact_snapshots.values())
    plant_sha256 = {
        "primary_path_sha256": artifact_snapshots["duct.primary_path"][1] if artifact_snapshots["duct.primary_path"] else "",
        "secondary_path_sha256": artifact_snapshots["duct.secondary_path"][1] if artifact_snapshots["duct.secondary_path"] else "",
        "plant_binding_sha256": artifact_snapshots["duct.plant_binding"][1] if artifact_snapshots["duct.plant_binding"] else "",
    }
    manifest_sha = artifact_snapshots["data.manifest_bundle"][1] if artifact_snapshots["data.manifest_bundle"] else ""
    pretrain_criterion_sha = artifact_snapshots["canonical_pretrain.criterion_implementation_receipt"][1] if artifact_snapshots["canonical_pretrain.criterion_implementation_receipt"] else ""
    finetune_criterion_sha = artifact_snapshots["canonical_finetune.criterion_implementation_receipt"][1] if artifact_snapshots["canonical_finetune.criterion_implementation_receipt"] else ""

    # Scratch pretrain admission은 fine-tune/checkpoint와 분리한다. 새 P/S, public
    # manifest/lineage/coverage/bootstrap, sampler+DNH criterion 및 pretrain external
    # contract만 exact하면 GPU smoke를 열 수 있어야 한다.
    pretrain_labels = (
        "duct.primary_path",
        "duct.secondary_path",
        "duct.plant_binding",
        "data.manifest_bundle",
        "data.lineage_receipt",
        "data.frequency_coverage_receipt",
        "data.transfer_bootstrap_receipt",
        "canonical_pretrain.criterion_implementation_receipt",
        "canonical_pretrain.model_config",
    )
    pretrain_artifacts_present = all(
        artifact_snapshots[label] is not None for label in pretrain_labels
    )
    pretrain_external_valid = False
    typed_pretrain_ready = False
    typed_pretrain_error: str | None = None
    if pretrain_artifacts_present and external_snapshots["canonical_pretrain"] is not None:
        pretrain_external_snapshot = external_snapshots["canonical_pretrain"]
        assert pretrain_external_snapshot is not None
        _validate_external_contract(
            _read_json(
                pretrain_external_snapshot[0],
                label="canonical pretrain external contract",
            ),
            stage="canonical_pretrain",
            profile_sha256=profile_sha256,
            plant_sha256=plant_sha256,
            manifest_bundle_sha256=manifest_sha,
            criterion_receipt_sha256=pretrain_criterion_sha,
            init_checkpoint_sha256=None,
        )
        pretrain_external_valid = True
        try:
            typed = load_stage2_pretrain_typed_admission(
                repository_root=root_path,
                primary_path_sha256=plant_sha256["primary_path_sha256"],
                secondary_path_sha256=plant_sha256["secondary_path_sha256"],
                plant_binding_ref=duct_artifacts["plant_binding"],
                manifest_ref=data_artifacts["manifest_bundle"],
                lineage_ref=data_artifacts["lineage_receipt"],
                coverage_ref=data_artifacts["frequency_coverage_receipt"],
                bootstrap_ref=data_artifacts["transfer_bootstrap_receipt"],
                criterion_receipt_ref=pretrain_artifacts[
                    "criterion_implementation_receipt"
                ],
            )
            typed_pretrain_ready = typed.status == "READY"
        except (OSError, TypeError, ValueError) as exc:
            typed_pretrain_error = str(exc)
    checks.append(
        _check(
            pretrain_external_valid,
            check_id="canonical_pretrain_external_contract_cross_binding",
            detail=(
                "pretrain external contract가 Stage-2/P-S/manifest/train/eval SHA에 exact"
                if pretrain_external_valid
                else "pretrain-only external contract/artifact chain 미완성"
            ),
        )
    )
    checks.append(
        _check(
            typed_pretrain_ready,
            check_id="typed_stage2_pretrain_execution",
            detail=(
                "typed P/S+public data+sampler+DNH+criterion tensor path READY"
                if typed_pretrain_ready
                else (
                    f"typed pretrain validation BLOCKED: {typed_pretrain_error}"
                    if typed_pretrain_error
                    else "typed pretrain artifact가 아직 없습니다"
                )
            ),
        )
    )
    checks.append(
        _check(
            typed_pretrain_ready,
            check_id="typed_stage2_execution_provenance",
            detail="pretrain typed execution provenance" if typed_pretrain_ready else "typed pretrain provenance 미검증",
        )
    )

    declared_metadata_valid = False
    if (
        required_for_declared_chain
        and fine_init_matches_campaign_checkpoint
        and all(external_snapshots.values())
    ):
        assert external_snapshots["canonical_pretrain"] is not None
        assert external_snapshots["canonical_finetune"] is not None
        _validate_external_contract(
            _read_json(external_snapshots["canonical_pretrain"][0], label="canonical pretrain external contract"),
            stage="canonical_pretrain",
            profile_sha256=profile_sha256,
            plant_sha256=plant_sha256,
            manifest_bundle_sha256=manifest_sha,
            criterion_receipt_sha256=pretrain_criterion_sha,
            init_checkpoint_sha256=None,
        )
        if checkpoint_snapshot is not None:
            _validate_external_contract(
                _read_json(external_snapshots["canonical_finetune"][0], label="canonical fine-tune external contract"),
                stage="canonical_finetune",
                profile_sha256=profile_sha256,
                plant_sha256=plant_sha256,
                manifest_bundle_sha256=manifest_sha,
                criterion_receipt_sha256=finetune_criterion_sha,
                init_checkpoint_sha256=checkpoint_snapshot[1],
            )
            if binding_snapshot is not None:
                _validate_checkpoint_binding(
                    _read_json(binding_snapshot[0], label="canonical pretrain checkpoint binding"),
                    checkpoint_sha256=checkpoint_snapshot[1],
                    external_contract_sha256=external_snapshots["canonical_pretrain"][1],
                    plant_sha256=plant_sha256,
                    manifest_bundle_sha256=manifest_sha,
                    pretrain_profile_sha256=profile_sha256["canonical_pretrain"],
                    evaluation_policy_sha256=profile_sha256["evaluation"],
                )
                declared_metadata_valid = True
    checks.append(_check(declared_metadata_valid, check_id="external_contract_checkpoint_cross_binding", detail="Stage-2 contract/P/S/train/eval SHA cross-binding exact" if declared_metadata_valid else "fine-tune용 외부 계약/checkpoint 결속 chain 미완성"))
    # recorded 70:30 Dataset/selection/test-once authority는 pretrain과 섞지 않는다.
    checks.append(
        _check(
            False,
            check_id="typed_stage2_finetune_recorded_70_30_execution",
            detail="recorded additions/70:30 fine-tune/recorded-val-only selection은 별도 BLOCKED",
        )
    )

    blocked = [str(check["id"]) for check in checks if check["status"] != "PASS"]
    pretrain_blocked = [
        check_id for check_id in blocked if check_id in _PRETRAIN_CHECK_IDS
    ]
    post_pretrain_blocked = [
        check_id for check_id in blocked if check_id not in _PRETRAIN_CHECK_IDS
    ]
    status = (
        "READY_PRETRAIN"
        if typed_pretrain_ready and pretrain_external_valid
        else STAGE2_UNATTESTED_STATUS
        if declared_metadata_valid
        else "BLOCKED"
    )
    contract = Stage2TwoKilohertzContract.canonical()
    return {
        "schema": STAGE2_CAMPAIGN_RESULT_SCHEMA,
        "status": status,
        "eligible": bool(typed_pretrain_ready and pretrain_external_valid),
        "pretrain_smoke_ready": bool(typed_pretrain_ready and pretrain_external_valid),
        "canonical_scratch_pretrain_100k_ready": bool(
            typed_pretrain_ready and pretrain_external_valid
        ),
        "canonical_finetune_ready": False,
        "admission_only": True,
        "control_band_contract_id": contract.contract_id,
        "control_band_contract_sha256": contract.digest(),
        "profile_sha256": profile_sha256,
        "declared_external_checkpoint_cross_binding_valid": declared_metadata_valid,
        "typed_stage2_execution_provenance_attested": bool(typed_pretrain_ready),
        "typed_stage2_pretrain_error": typed_pretrain_error,
        "generic_stage1_trainer_allowed": False,
        "legacy_init_or_resume_allowed": False,
        "scratch_pretrain_required": True,
        "three_db_is_minimum_not_optimization_target": True,
        "checkpoint_selection_primary": "maximize_minimum_frequency_family_dnh_runtime_gate_margin",
        "checkpoint_selection_secondary": "maximize_two_khz_family_equal_mean_attenuation_db",
        "audio_opened": False,
        "trainer_constructed": False,
        "gpu_initialized": False,
        "runs_directory_created": False,
        "checks": checks,
        "blockers": blocked,
        "pretrain_blockers": pretrain_blocked,
        "post_pretrain_blockers": post_pretrain_blocked,
        "pretrain_blocking_requirements": list(_PRETRAIN_ARTIFACT_REQUIREMENTS),
        "post_pretrain_blocking_requirements": list(
            _FINETUNE_ARTIFACT_REQUIREMENTS
        ),
        "blocking_requirements": (
            list(_FINETUNE_ARTIFACT_REQUIREMENTS)
            if typed_pretrain_ready and pretrain_external_valid
            else [*_PRETRAIN_ARTIFACT_REQUIREMENTS, *_FINETUNE_ARTIFACT_REQUIREMENTS]
        ),
        "preflight_input_sha256": _sha256_bytes(
            _canonical_json(
                {
                    "campaign": dict(root),
                    "profile_sha256": profile_sha256,
                }
            )
        ),
    }


def write_stage2_preflight_json_exclusive(path: str | Path, report: Mapping[str, Any]) -> None:
    """명시적 요청에서만 결과를 no-replace JSON으로 기록한다."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(target, flags, 0o644)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_ready_stage2_pretrain_launch(
    payload: Mapping[str, Any], *, repo_root: str | Path
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """READY_PRETRAIN campaign에서 typed admission/train profile/launch anchor를 반환한다.

    run directory나 torch/GPU를 만들지 않는다. runner가 이 반환값을 받은 뒤에만 scratch
    directory와 CUDA context를 만들 수 있다.
    """

    root_path = Path(repo_root).resolve(strict=True)
    report = audit_stage2_2khz_campaign(payload, repo_root=root_path)
    if report.get("status") != "READY_PRETRAIN" or report.get("pretrain_smoke_ready") is not True:
        raise ValueError(f"Stage-2 pretrain campaign이 READY가 아닙니다: {report.get('status')}")
    root = _require_exact_keys(
        dict(payload),
        {"schema", "role", "control_band_contract", "profiles", "external_contracts", "canonical_pretrain_checkpoint", "release_policy"},
        label="Stage-2 campaign profile",
    )
    profile_entries = _require_exact_keys(root["profiles"], set(_PROFILE_ROLES), label="campaign profiles")
    profile_payloads: dict[str, dict[str, Any]] = {}
    profile_sha256: dict[str, str] = {}
    for role in _PROFILE_ROLES:
        intent = _artifact_intent(profile_entries[role], label=f"profiles.{role}")
        snapshot = _snapshot_intent(root_path, intent, label=f"profiles.{role}")
        if snapshot is None:
            raise ValueError(f"Stage-2 profile이 사라졌습니다: {role}")
        profile_payloads[role] = _read_yaml(snapshot[0], label=f"profiles.{role}")
        profile_sha256[role] = snapshot[1]
    duct_artifacts = _validate_duct_profile(profile_payloads["duct"])
    data_artifacts = _validate_data_profile(profile_payloads["data"])
    pretrain_artifacts = _validate_training_profile(
        profile_payloads["canonical_pretrain"], expected_role="canonical_pretrain"
    )
    external_intent = _artifact_intent(
        root["external_contracts"]["canonical_pretrain"],
        label="external_contracts.canonical_pretrain",
    )
    external_snapshot = _snapshot_intent(
        root_path, external_intent, label="external_contracts.canonical_pretrain"
    )
    if external_snapshot is None:
        raise ValueError("Stage-2 pretrain external contract가 사라졌습니다")
    external_payload = _read_json(
        external_snapshot[0], label="canonical pretrain external contract"
    )
    typed = load_stage2_pretrain_typed_admission(
        repository_root=root_path,
        primary_path_sha256=str(duct_artifacts["primary_path"][1]),
        secondary_path_sha256=str(duct_artifacts["secondary_path"][1]),
        plant_binding_ref=(str(duct_artifacts["plant_binding"][0]), str(duct_artifacts["plant_binding"][1])),
        manifest_ref=(str(data_artifacts["manifest_bundle"][0]), str(data_artifacts["manifest_bundle"][1])),
        lineage_ref=(str(data_artifacts["lineage_receipt"][0]), str(data_artifacts["lineage_receipt"][1])),
        coverage_ref=(str(data_artifacts["frequency_coverage_receipt"][0]), str(data_artifacts["frequency_coverage_receipt"][1])),
        bootstrap_ref=(str(data_artifacts["transfer_bootstrap_receipt"][0]), str(data_artifacts["transfer_bootstrap_receipt"][1])),
        criterion_receipt_ref=(
            str(pretrain_artifacts["criterion_implementation_receipt"][0]),
            str(pretrain_artifacts["criterion_implementation_receipt"][1]),
        ),
    )
    anchors = {
        "external_experiment_contract_sha256": external_snapshot[1],
        "artifact_source_commit_sha": external_payload["artifact_source_commit_sha"],
        "pretrain_profile_sha256": profile_sha256["canonical_pretrain"],
        "evaluation_policy_sha256": profile_sha256["evaluation"],
        "model_config_path": pretrain_artifacts["model_config"][0],
        "model_config_sha256": pretrain_artifacts["model_config"][1],
    }
    return typed, profile_payloads["canonical_pretrain"], anchors


__all__ = [
    "STAGE2_CAMPAIGN_RESULT_SCHEMA",
    "STAGE2_CAMPAIGN_ROLE",
    "STAGE2_CAMPAIGN_SCHEMA",
    "STAGE2_CHECKPOINT_BINDING_SCHEMA",
    "STAGE2_EXTERNAL_CONTRACT_SCHEMA",
    "STAGE2_UNATTESTED_STATUS",
    "audit_stage2_2khz_campaign",
    "is_stage2_2khz_profile_config",
    "load_ready_stage2_pretrain_launch",
    "write_stage2_preflight_json_exclusive",
]
