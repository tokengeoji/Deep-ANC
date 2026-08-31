"""READY_PRETRAIN만 소비하는 Stage-2 scratch training runner.

이 모듈은 admission 전에 run directory/CUDA를 만들지 않는다. 자동 resume는
없고, 명시적 checkpoint의 model/optimizer/scheduler/RNG와 전체 Stage-2 SHA가
일치할 때만 다음 global step을 계속한다.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import stat
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.signal import oaconvolve, resample_poly

from ..dsp.stage2_2khz_contract import Stage2TwoKilohertzContract
from ..losses.stage2_2khz_loss import (
    STAGE2_2KHZ_ONE_POINT_SIX_SENTINEL_BAND_HZ,
    Stage2TwoKilohertzLoss,
)
from ..models import build_model
from .causal_secondary_prefix_adapter_v1 import (
    CausalPrefixBatchV1,
    CausalPrefixStateOriginV1,
)
from .stage2_a100_environment import (
    configure_and_collect_stage2_a100_environment,
    stage2_a100_environment_sha256,
)
from .stage2_2khz_campaign import load_ready_stage2_pretrain_launch
from .stage2_2khz_execution import (
    Stage2ActualBatchIdentity,
    Stage2CausalPrefixAdapter,
    Stage2TensorBatch,
    Stage2TwoKilohertzTrainerAdapter,
    require_stage2_actuator_limit,
)
from .stage2_2khz_pretrain_admission import (
    STAGE2_PRETRAIN_AUTHORITY_PATH,
    Stage2PretrainSource,
    Stage2PretrainTypedAdmission,
)
from .stage2_2khz_git_authority import (
    verify_source_commit_ancestor,
    verify_tracked_head_authority,
    verify_tracked_head_file,
)


STAGE2_PRETRAIN_CHECKPOINT_SCHEMA = "stage2_2khz_pretrain_checkpoint_v1"
STAGE2_PRETRAIN_DIAGNOSTIC_CHECKPOINT_SCHEMA = (
    "stage2_2khz_pretrain_cpu_diagnostic_checkpoint_v1"
)
STAGE2_PRETRAIN_RUN_SCHEMA = "stage2_2khz_pretrain_external_sha_seed_v1"
STAGE2_PRETRAIN_RUNNER_SCHEMA = "stage2_2khz_scratch_pretrain_runner_v1"
STAGE2_PRETRAIN_STEP_TELEMETRY_SCHEMA = "stage2_2khz_step_telemetry_v1"
STAGE2_PRETRAIN_SMOKE_PERFORMANCE_SCHEMA = (
    "stage2_2khz_smoke_performance_evidence_v1"
)
STAGE2_PRETRAIN_SMOKE_ACCEPTANCE_SCHEMA = "stage2_2khz_smoke_acceptance_v1"
STAGE2_PRETRAIN_RESUME_EQUIVALENCE_SCHEMA = (
    "stage2_2khz_uninterrupted_resume_equivalence_v1"
)

_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{label}는 lowercase SHA-256이어야 합니다")
    return text


def _inside_repository(root: Path, value: object, *, label: str) -> Path:
    relative = Path(str(value or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}는 repository 내부 상대경로여야 합니다")
    cursor = root
    for part in relative.parts:
        cursor /= part
        try:
            node = cursor.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(node.st_mode):
            raise ValueError(f"{label} 경로에 symlink가 있습니다")
    return root / relative


def _snapshot_regular_file(path: Path) -> tuple[bytes, str]:
    """symlink/TOCTOU를 거부하고 한 descriptor에서 bytes를 고정한다."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"regular checkpoint artifact만 허용합니다: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError(f"checkpoint snapshot 중 파일이 바뀌었습니다: {path}")
    content = b"".join(chunks)
    if len(content) != int(after.st_size):
        raise ValueError(f"checkpoint snapshot 크기가 다릅니다: {path}")
    return content, hashlib.sha256(content).hexdigest()


def _numpy_rng_state_payload() -> dict[str, Any]:
    algorithm, values, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "algorithm": str(algorithm),
        # torch 2.5 legacy serializer는 uint32 storage를 저장하지 못하므로 bit pattern을
        # exact int64 값(0..2**32-1)으로 보존한다. weights_only 안전 타입만 사용한다.
        "values": torch.from_numpy(np.asarray(values, dtype=np.int64).copy()),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_numpy_rng_state(payload: object) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "algorithm",
        "values",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise ValueError("Stage-2 numpy RNG state schema가 다릅니다")
    values = payload["values"]
    if (
        not isinstance(values, torch.Tensor)
        or values.dtype != torch.int64
        or values.ndim != 1
        or bool(torch.any(values < 0))
        or bool(torch.any(values > 0xFFFFFFFF))
    ):
        raise ValueError("Stage-2 numpy RNG values가 안전한 uint32-range int64 tensor가 아닙니다")
    np.random.set_state(
        (
            str(payload["algorithm"]),
            values.cpu().numpy().astype(np.uint32, copy=True),
            int(payload["position"]),
            int(payload["has_gauss"]),
            float(payload["cached_gaussian"]),
        )
    )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        dict(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_exact_clean_checkout(root: Path, expected_commit: str) -> None:
    """외부 계약의 40자리 commit과 현재 clean checkout을 GPU 전에 확인한다."""

    if not _COMMIT_SHA_RE.fullmatch(str(expected_commit)):
        raise ValueError("Stage-2 external contract repository commit이 40-hex가 아닙니다")
    environment = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"}
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Stage-2 exact git checkout을 검증할 수 없습니다") from exc
    if head != expected_commit:
        raise RuntimeError(
            f"Stage-2 checkout HEAD 불일치: expected={expected_commit}, actual={head}"
        )
    if status:
        first = status.splitlines()[0]
        raise RuntimeError(f"Stage-2 checkout이 dirty입니다: {first}")


def _nvidia_smi_l_snapshot(root: Path) -> str:
    """typed READY 뒤, CUDA context 전의 read-only physical GPU inventory."""

    try:
        return subprocess.run(
            ["nvidia-smi", "-L"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Stage-2 A100 nvidia-smi -L inventory를 읽을 수 없습니다") from exc


def _preverify_campaign_checkout(root: Path, campaign: Mapping[str, Any]) -> str:
    """corpus/P/S scan보다 먼저 external contract의 exact commit을 검증한다."""

    campaign_bytes, _, execution_head = verify_tracked_head_file(
        root, "configs/stage2_2khz_campaign.yaml"
    )
    try:
        tracked_campaign = yaml.safe_load(campaign_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("tracked Stage-2 campaign YAML을 읽을 수 없습니다") from exc
    if not isinstance(tracked_campaign, dict) or tracked_campaign != dict(campaign):
        raise ValueError("in-memory Stage-2 campaign이 tracked HEAD bytes와 다릅니다")
    _, _, authority_head = verify_tracked_head_authority(
        root, STAGE2_PRETRAIN_AUTHORITY_PATH
    )
    if authority_head != execution_head:
        raise ValueError("Stage-2 campaign/authority HEAD가 다릅니다")
    try:
        reference = campaign["external_contracts"]["canonical_pretrain"]
        if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
            raise ValueError
        relative = Path(str(reference["path"]))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError
        content, actual_sha = _snapshot_regular_file(root / relative)
        if actual_sha != str(reference["sha256"]):
            raise ValueError
        payload = json.loads(content.decode("utf-8"))
        artifact_source_commit = str(payload["artifact_source_commit_sha"])
        if not _COMMIT_SHA_RE.fullmatch(artifact_source_commit):
            raise ValueError
    except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "Stage-2 external contract/commit을 corpus scan 전에 검증할 수 없습니다"
        ) from exc
    verify_source_commit_ancestor(
        root, artifact_source_commit, head=execution_head
    )
    return execution_head


def stage2_checkpoint_binding_payload(
    *,
    checkpoint_sha256: str,
    bindings: Mapping[str, Any],
    completed_steps: int,
    production_execution: bool = True,
) -> dict[str, Any]:
    """campaign validator가 소비하는 flattened checkpoint sidecar를 만든다."""

    init_eligible = bool(production_execution) and int(completed_steps) == 100_000
    smoke_acceptance_sha256 = bindings["smoke_acceptance_sha256"]
    if bool(production_execution) and int(completed_steps) > 500:
        smoke_acceptance_sha256 = _require_sha256(
            smoke_acceptance_sha256,
            label="Stage-2 checkpoint smoke acceptance SHA",
        )
    elif smoke_acceptance_sha256 is not None:
        raise ValueError(
            "Stage-2 step 500 이하 checkpoint에는 smoke acceptance SHA가 없어야 합니다"
        )
    completion = {
        "schema": "stage2_2khz_pretrain_completion_receipt_v2",
        "checkpoint_sha256": str(checkpoint_sha256),
        "external_experiment_contract_sha256": str(
            bindings["external_experiment_contract_sha256"]
        ),
        "completed_steps": int(completed_steps),
        "init_eligible": bool(init_eligible),
        "scratch_pretrain": True,
        "smoke_acceptance_sha256": smoke_acceptance_sha256,
    }
    return {
        "schema": "stage2_2khz_checkpoint_binding_v2",
        "checkpoint_sha256": str(checkpoint_sha256),
        "external_experiment_contract_sha256": str(
            bindings["external_experiment_contract_sha256"]
        ),
        "control_band_contract": {
            "id": str(bindings["stage2_contract_id"]),
            "sha256": str(bindings["stage2_contract_sha256"]),
        },
        "plant_sha256": {
            "primary_path_sha256": str(bindings["primary_path_sha256"]),
            "secondary_path_sha256": str(bindings["secondary_path_sha256"]),
            "plant_binding_sha256": str(bindings["plant_binding_sha256"]),
        },
        "plant_binding_runtime_sha256": str(
            bindings["plant_binding_runtime_sha256"]
        ),
        "manifest_bundle_sha256": str(bindings["manifest_bundle_sha256"]),
        "training_profile_sha256": str(bindings["training_profile_sha256"]),
        "evaluation_policy_sha256": str(bindings["evaluation_policy_sha256"]),
        "model_config_sha256": str(bindings["model_config_sha256"]),
        "criterion_receipt_sha256": str(bindings["criterion_receipt_sha256"]),
        "sampler_receipt_sha256": str(bindings["sampler_receipt_sha256"]),
        "dnh_calibration_receipt_sha256": str(
            bindings["dnh_calibration_receipt_sha256"]
        ),
        "a100_environment_sha256": str(bindings["a100_environment_sha256"]),
        "smoke_acceptance_sha256": smoke_acceptance_sha256,
        "experiment_role": "canonical_pretrain",
        "completed_steps": int(completed_steps),
        "init_eligible": bool(init_eligible),
        "scratch_pretrain": True,
        "legacy_origin": False,
        "diagnostic_cpu_test": not bool(production_execution),
        "completion_receipt_sha256": hashlib.sha256(
            _canonical_json(completion)
        ).hexdigest(),
    }


def _load_json_ref(
    root: Path,
    value: object,
    *,
    label: str,
    expected_path: Path | None = None,
) -> tuple[dict[str, Any], bytes, str, Path]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} ref가 exact하지 않습니다")
    path = _inside_repository(root, value["path"], label=label)
    if expected_path is not None and path != expected_path:
        raise ValueError(f"{label} 경로가 current run artifact가 아닙니다")
    content, actual_sha = _snapshot_regular_file(path)
    if actual_sha != _require_sha256(value["sha256"], label=f"{label}.sha256"):
        raise ValueError(f"{label} bytes SHA가 다릅니다")
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}는 UTF-8 JSON이어야 합니다") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root가 mapping이 아닙니다")
    return payload, content, actual_sha, path


def _load_binary_ref(
    root: Path,
    value: object,
    *,
    label: str,
    expected_path: Path | None = None,
) -> tuple[bytes, str, Path]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} ref가 exact하지 않습니다")
    path = _inside_repository(root, value["path"], label=label)
    if expected_path is not None and path != expected_path:
        raise ValueError(f"{label} 경로가 current run artifact가 아닙니다")
    content, actual_sha = _snapshot_regular_file(path)
    if actual_sha != _require_sha256(value["sha256"], label=f"{label}.sha256"):
        raise ValueError(f"{label} bytes SHA가 다릅니다")
    return content, actual_sha, path


def _telemetry_rows(
    content: bytes,
    *,
    expected_steps: int,
    expected_batch_size: int,
    expected_target_samples: int,
    utilization_every: int,
    label: str,
) -> list[dict[str, Any]]:
    if not content.endswith(b"\n"):
        raise ValueError(f"{label} JSONL이 완전한 newline으로 끝나지 않습니다")
    try:
        rows = [json.loads(line) for line in content.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} JSONL을 읽을 수 없습니다") from exc
    if len(rows) != expected_steps or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} row 수가 exact {expected_steps}가 아닙니다")
    required_keys = {
        "schema",
        "step",
        "batch_size",
        "global_sample_index_first",
        "global_sample_index_last",
        "data_wait_ms",
        "h2d_ms",
        "compute_step_ms",
        "total_step_ms",
        "samples_per_second",
        "gpu_memory_allocated_bytes",
        "gpu_memory_reserved_bytes",
        "gpu_peak_allocated_bytes",
        "gpu_peak_reserved_bytes",
        "gpu_utilization_percent",
        "nvidia_smi_memory_used_mib",
    }
    numeric_fields = (
        "data_wait_ms",
        "h2d_ms",
        "compute_step_ms",
        "total_step_ms",
        "samples_per_second",
    )
    integer_fields = (
        "global_sample_index_first",
        "global_sample_index_last",
        "gpu_memory_allocated_bytes",
        "gpu_memory_reserved_bytes",
        "gpu_peak_allocated_bytes",
        "gpu_peak_reserved_bytes",
    )
    for expected_step, row in enumerate(rows, start=1):
        if set(row) != required_keys:
            raise ValueError(f"{label} step {expected_step} key 집합이 exact하지 않습니다")
        if (
            row["schema"] != STAGE2_PRETRAIN_STEP_TELEMETRY_SCHEMA
            or row["step"] != expected_step
            or row["batch_size"] != expected_batch_size
        ):
            raise ValueError(f"{label} step/batch/schema가 canonical이 아닙니다")
        for key in numeric_fields:
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label} {key}가 numeric이 아닙니다")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{label} {key}가 finite nonnegative가 아닙니다")
        for key in integer_fields:
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} {key}가 nonnegative integer가 아닙니다")
        if row["global_sample_index_last"] < row["global_sample_index_first"]:
            raise ValueError(f"{label} global sample index 순서가 반대입니다")
        expected_first = (expected_step - 1) * expected_batch_size
        expected_last = expected_step * expected_batch_size - 1
        if (
            row["global_sample_index_first"] != expected_first
            or row["global_sample_index_last"] != expected_last
        ):
            raise ValueError(f"{label} global sample index가 B96 순차 계약과 다릅니다")
        if (
            float(row["compute_step_ms"]) <= 0.0
            or float(row["total_step_ms"]) <= 0.0
            or float(row["samples_per_second"]) <= 0.0
            or float(row["total_step_ms"]) + 1.0e-9
            < float(row["data_wait_ms"])
            + float(row["h2d_ms"])
            + float(row["compute_step_ms"])
        ):
            raise ValueError(f"{label} step timing이 complete wall-time 계약이 아닙니다")
        expected_throughput = (
            expected_batch_size
            * expected_target_samples
            / (float(row["total_step_ms"]) / 1000.0)
        )
        if not math.isclose(
            float(row["samples_per_second"]),
            expected_throughput,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise ValueError(f"{label} throughput이 actual B×samples/wall-time과 다릅니다")
        observed = expected_step % utilization_every == 0
        utilization = row["gpu_utilization_percent"]
        memory_used = row["nvidia_smi_memory_used_mib"]
        if observed:
            if (
                isinstance(utilization, bool)
                or not isinstance(utilization, int)
                or not 0 <= utilization <= 100
                or isinstance(memory_used, bool)
                or not isinstance(memory_used, int)
                or memory_used < 0
            ):
                raise ValueError(f"{label} nvidia-smi 관측값이 불완전합니다")
        elif utilization is not None or memory_used is not None:
            raise ValueError(f"{label} nvidia-smi 관측 주기가 profile과 다릅니다")
    return rows


def _validate_production_checkpoint_state(
    checkpoint: object,
    *,
    expected_bindings: Mapping[str, Any],
    completed_steps: int,
    label: str,
) -> dict[str, Any]:
    required = {
        "schema",
        "completed_steps",
        "init_eligible",
        "bindings",
        "model",
        "optimizer",
        "scheduler",
        "step",
        "torch_rng_state",
        "numpy_rng_state",
        "python_rng_state",
        "cuda_rng_state_all",
        "last_metrics",
    }
    if not isinstance(checkpoint, dict) or set(checkpoint) != required:
        raise ValueError(f"{label} production checkpoint key 집합이 exact하지 않습니다")
    if (
        checkpoint["schema"] != STAGE2_PRETRAIN_CHECKPOINT_SCHEMA
        or checkpoint["step"] != completed_steps
        or checkpoint["completed_steps"] != completed_steps
        or checkpoint["init_eligible"] is not False
        or checkpoint["bindings"] != dict(expected_bindings)
    ):
        raise ValueError(f"{label} production checkpoint semantic이 exact하지 않습니다")

    model = checkpoint["model"]
    if (
        not isinstance(model, Mapping)
        or not model
        or any(
            not isinstance(value, torch.Tensor)
            or not bool(torch.all(torch.isfinite(value)))
            for value in model.values()
        )
    ):
        raise ValueError(f"{label} model state가 finite/nonempty tensor mapping이 아닙니다")
    optimizer = checkpoint["optimizer"]
    if (
        not isinstance(optimizer, Mapping)
        or set(optimizer) != {"state", "param_groups"}
        or not isinstance(optimizer["state"], Mapping)
        or not optimizer["state"]
        or not isinstance(optimizer["param_groups"], list)
        or not optimizer["param_groups"]
        or not _state_all_finite(optimizer)
    ):
        raise ValueError(f"{label} optimizer state가 complete하지 않습니다")
    scheduler = checkpoint["scheduler"]
    if (
        not isinstance(scheduler, Mapping)
        or not scheduler
        or not _state_all_finite(scheduler)
    ):
        raise ValueError(f"{label} scheduler state가 complete하지 않습니다")
    torch_rng = checkpoint["torch_rng_state"]
    cuda_rng = checkpoint["cuda_rng_state_all"]
    if (
        not isinstance(torch_rng, torch.Tensor)
        or torch_rng.dtype != torch.uint8
        or torch_rng.ndim != 1
        or torch_rng.numel() == 0
        or not isinstance(cuda_rng, list)
        or len(cuda_rng) != 1
        or not isinstance(cuda_rng[0], torch.Tensor)
        or cuda_rng[0].dtype != torch.uint8
        or cuda_rng[0].ndim != 1
        or cuda_rng[0].numel() == 0
    ):
        raise ValueError(f"{label} torch/CUDA RNG state가 complete하지 않습니다")
    numpy_rng = checkpoint["numpy_rng_state"]
    if not isinstance(numpy_rng, Mapping) or set(numpy_rng) != {
        "algorithm",
        "values",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise ValueError(f"{label} numpy RNG state schema가 다릅니다")
    numpy_values = numpy_rng["values"]
    if (
        not isinstance(numpy_values, torch.Tensor)
        or numpy_values.dtype != torch.int64
        or numpy_values.ndim != 1
        or numpy_values.numel() == 0
        or bool(torch.any(numpy_values < 0))
        or bool(torch.any(numpy_values > 0xFFFFFFFF))
    ):
        raise ValueError(f"{label} numpy RNG values가 exact uint32 range가 아닙니다")
    python_rng = checkpoint["python_rng_state"]
    if not isinstance(python_rng, tuple) or len(python_rng) != 3:
        raise ValueError(f"{label} Python RNG state가 complete하지 않습니다")
    metrics = checkpoint["last_metrics"]
    if (
        not isinstance(metrics, Mapping)
        or not metrics
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in metrics.values()
        )
        or float(metrics.get("step", -1.0)) != float(completed_steps)
    ):
        raise ValueError(f"{label} last metrics가 finite/complete하지 않습니다")
    return checkpoint


def _smoke_expected_payload(
    *,
    rows: list[dict[str, Any]],
    completed_step: int,
    telemetry_sha256: str,
    environment_sha256: str,
    loader_workers: int,
    bounded_prefetch_batches: int,
) -> dict[str, Any]:
    selected = rows[:completed_step]

    def percentile(key: str, q: float) -> float:
        return float(
            np.percentile(
                np.asarray([float(row[key]) for row in selected], dtype=np.float64),
                q,
            )
        )

    utilization = [
        int(row["gpu_utilization_percent"])
        for row in selected
        if row["gpu_utilization_percent"] is not None
    ]
    return {
        "schema": STAGE2_PRETRAIN_SMOKE_PERFORMANCE_SCHEMA,
        "status": "BLOCKED_PENDING_EMPIRICAL_THRESHOLD_DECISION",
        "canonical_100k_start_eligible": False,
        "completed_step": completed_step,
        "actual_batch_size": 96,
        "loader_workers": loader_workers,
        "bounded_prefetch_batches": bounded_prefetch_batches,
        "pin_memory": True,
        "non_blocking_h2d": True,
        "raw_step_telemetry_sha256": telemetry_sha256,
        "a100_environment_sha256": environment_sha256,
        "data_wait_ms_p50": percentile("data_wait_ms", 50.0),
        "data_wait_ms_p95": percentile("data_wait_ms", 95.0),
        "h2d_ms_p50": percentile("h2d_ms", 50.0),
        "h2d_ms_p95": percentile("h2d_ms", 95.0),
        "compute_step_ms_p50": percentile("compute_step_ms", 50.0),
        "compute_step_ms_p95": percentile("compute_step_ms", 95.0),
        "total_step_ms_p50": percentile("total_step_ms", 50.0),
        "total_step_ms_p95": percentile("total_step_ms", 95.0),
        "samples_per_second_p50": percentile("samples_per_second", 50.0),
        "gpu_utilization_observation_count": len(utilization),
        "gpu_utilization_percent_min": min(utilization),
        "gpu_utilization_percent_median": float(np.median(utilization)),
        "gpu_utilization_percent_max": max(utilization),
        "performance_thresholds_declared_before_observation": False,
        "oom_or_runtime_failure": False,
    }


def _state_exact_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_state_exact_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)  # type: ignore[arg-type]
            and all(_state_exact_equal(a, b) for a, b in zip(left, right))  # type: ignore[arg-type]
        )
    return type(left) is type(right) and left == right


def _state_all_finite(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.all(torch.isfinite(value)))
    if isinstance(value, np.ndarray):
        return not np.issubdtype(value.dtype, np.inexact) or bool(
            np.all(np.isfinite(value))
        )
    if isinstance(value, Mapping):
        return all(_state_all_finite(key) and _state_all_finite(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_state_all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _load_bound_checkpoint(
    root: Path,
    *,
    checkpoint_ref: object,
    binding_ref: object,
    expected_checkpoint_path: Path | None,
    expected_binding_path: Path | None,
    expected_bindings: Mapping[str, Any],
    completed_steps: int,
    label: str,
) -> tuple[dict[str, Any], str, Path]:
    content, checkpoint_sha, checkpoint_path = _load_binary_ref(
        root,
        checkpoint_ref,
        label=f"{label} checkpoint",
        expected_path=expected_checkpoint_path,
    )
    sidecar, _, _, sidecar_path = _load_json_ref(
        root,
        binding_ref,
        label=f"{label} binding",
        expected_path=expected_binding_path,
    )
    if sidecar_path != checkpoint_path.with_suffix(".binding.json"):
        raise ValueError(f"{label} checkpoint/binding 경로가 쌍이 아닙니다")
    expected_sidecar = stage2_checkpoint_binding_payload(
        checkpoint_sha256=checkpoint_sha,
        bindings=expected_bindings,
        completed_steps=completed_steps,
        production_execution=True,
    )
    if sidecar != expected_sidecar:
        raise ValueError(f"{label} checkpoint/binding/contract가 exact하지 않습니다")
    checkpoint = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    checkpoint = _validate_production_checkpoint_state(
        checkpoint,
        expected_bindings=expected_bindings,
        completed_steps=completed_steps,
        label=label,
    )
    return checkpoint, checkpoint_sha, checkpoint_path


def _validate_stage2_smoke_acceptance(
    root: Path,
    *,
    acceptance_path: Path,
    run_dir: Path,
    external_sha256: str,
    environment: Mapping[str, Any],
    environment_sha256: str,
    profile: Mapping[str, Any],
    expected_bindings: Mapping[str, Any],
) -> str:
    """actual 200→explicit-resume→500 bytes와 독립 등가 evidence만 100k를 연다."""

    acceptance_content, acceptance_sha = _snapshot_regular_file(acceptance_path)
    try:
        acceptance = json.loads(acceptance_content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Stage-2 smoke acceptance는 UTF-8 JSON이어야 합니다") from exc
    keys = {
        "schema",
        "status",
        "canonical_100k_start_eligible",
        "external_experiment_contract_sha256",
        "a100_environment_sha256",
        "production_batch_size",
        "run_identity",
        "environment",
        "step_200_checkpoint",
        "step_200_binding",
        "step_500_checkpoint",
        "step_500_binding",
        "smoke_step_200",
        "smoke_step_500",
        "raw_step_telemetry",
        "failure_receipt_absent",
        "resume_chain",
        "uninterrupted_vs_resume_evidence",
    }
    if not isinstance(acceptance, dict) or set(acceptance) != keys:
        raise ValueError("Stage-2 smoke acceptance key 집합이 exact하지 않습니다")
    if (
        acceptance["schema"] != STAGE2_PRETRAIN_SMOKE_ACCEPTANCE_SCHEMA
        or acceptance["status"] != "PASS"
        or acceptance["canonical_100k_start_eligible"] is not True
        or acceptance["external_experiment_contract_sha256"] != external_sha256
        or acceptance["a100_environment_sha256"] != environment_sha256
        or acceptance["production_batch_size"] != 96
        or acceptance["failure_receipt_absent"] is not True
    ):
        raise ValueError("Stage-2 smoke acceptance semantic이 canonical 100k 계약이 아닙니다")
    execution = profile["execution"]
    pipeline = execution["data_pipeline"]
    if int(execution["batch_size"]) != 96:
        raise ValueError("Stage-2 100k acceptance는 actual B96 smoke만 허용합니다")

    identity, _, _, _ = _load_json_ref(
        root,
        acceptance["run_identity"],
        label="Stage-2 smoke run identity",
        expected_path=run_dir / "run_identity.json",
    )
    expected_identity = {
        "schema": STAGE2_PRETRAIN_RUN_SCHEMA,
        "runner_schema": STAGE2_PRETRAIN_RUNNER_SCHEMA,
        "repository_commit_sha": str(expected_bindings["repository_commit_sha"]),
        "external_experiment_contract_sha256": external_sha256,
        "seed": int(profile["seed"]),
        "scratch_pretrain": True,
        "automatic_resume_allowed": False,
        "run_until_step": 200,
        "a100_environment_sha256": environment_sha256,
        "production_batch_size": 96,
        "loader_workers": int(pipeline["loader_workers"]),
        "bounded_prefetch_batches": int(pipeline["bounded_prefetch_batches"]),
    }
    if identity != expected_identity:
        raise ValueError("Stage-2 smoke run identity가 200 scratch 시작을 증명하지 않습니다")
    persisted_environment, _, _, _ = _load_json_ref(
        root,
        acceptance["environment"],
        label="Stage-2 smoke environment",
        expected_path=run_dir / "environment.json",
    )
    if (
        persisted_environment != dict(environment)
        or stage2_a100_environment_sha256(persisted_environment)
        != environment_sha256
    ):
        raise ValueError("Stage-2 smoke environment가 current A100과 다릅니다")

    step_200, step_200_sha, _ = _load_bound_checkpoint(
        root,
        checkpoint_ref=acceptance["step_200_checkpoint"],
        binding_ref=acceptance["step_200_binding"],
        expected_checkpoint_path=run_dir / "step_000200.pt",
        expected_binding_path=run_dir / "step_000200.binding.json",
        expected_bindings=expected_bindings,
        completed_steps=200,
        label="Stage-2 step 200",
    )
    step_500, step_500_sha, _ = _load_bound_checkpoint(
        root,
        checkpoint_ref=acceptance["step_500_checkpoint"],
        binding_ref=acceptance["step_500_binding"],
        expected_checkpoint_path=run_dir / "step_000500.pt",
        expected_binding_path=run_dir / "step_000500.binding.json",
        expected_bindings=expected_bindings,
        completed_steps=500,
        label="Stage-2 step 500",
    )
    resume_chain = acceptance["resume_chain"]
    if resume_chain != {
        "scratch_run_until_step": 200,
        "explicit_resume_from_step": 200,
        "explicit_resume_checkpoint_sha256": step_200_sha,
        "resumed_until_step": 500,
        "resumed_checkpoint_sha256": step_500_sha,
    }:
        raise ValueError("Stage-2 smoke acceptance가 200→explicit resume→500 chain과 다릅니다")

    telemetry_content, telemetry_sha, _ = _load_binary_ref(
        root,
        acceptance["raw_step_telemetry"],
        label="Stage-2 smoke raw telemetry",
        expected_path=run_dir / "step_telemetry.jsonl",
    )
    utilization_every = int(execution["telemetry"]["nvidia_smi_sample_every_steps"])
    rows = _telemetry_rows(
        telemetry_content,
        expected_steps=500,
        expected_batch_size=96,
        expected_target_samples=int(execution["target_samples"]),
        utilization_every=utilization_every,
        label="Stage-2 resumed smoke telemetry",
    )
    lines = telemetry_content.splitlines(keepends=True)
    prefix_200_sha = hashlib.sha256(b"".join(lines[:200])).hexdigest()
    smoke_200, _, _, _ = _load_json_ref(
        root,
        acceptance["smoke_step_200"],
        label="Stage-2 smoke step 200 receipt",
        expected_path=run_dir / "smoke_performance_step_000200.json",
    )
    smoke_500, _, _, _ = _load_json_ref(
        root,
        acceptance["smoke_step_500"],
        label="Stage-2 smoke step 500 receipt",
        expected_path=run_dir / "smoke_performance_step_000500.json",
    )
    smoke_args = {
        "rows": rows,
        "environment_sha256": environment_sha256,
        "loader_workers": int(pipeline["loader_workers"]),
        "bounded_prefetch_batches": int(pipeline["bounded_prefetch_batches"]),
    }
    if smoke_200 != _smoke_expected_payload(
        **smoke_args,
        completed_step=200,
        telemetry_sha256=prefix_200_sha,
    ) or smoke_500 != _smoke_expected_payload(
        **smoke_args,
        completed_step=500,
        telemetry_sha256=telemetry_sha,
    ):
        raise ValueError("Stage-2 smoke receipt가 actual raw telemetry 통계와 다릅니다")
    if (run_dir / "failure.json").exists():
        raise ValueError("Stage-2 smoke run에 OOM/runtime failure receipt가 있습니다")

    equivalence, _, _, equivalence_path = _load_json_ref(
        root,
        acceptance["uninterrupted_vs_resume_evidence"],
        label="Stage-2 uninterrupted/resume equivalence",
    )
    equivalence_keys = {
        "schema",
        "status",
        "external_experiment_contract_sha256",
        "a100_environment_sha256",
        "completed_step",
        "comparison_algorithm",
        "resumed_checkpoint_sha256",
        "uninterrupted_checkpoint",
        "uninterrupted_binding",
        "uninterrupted_run_identity",
        "uninterrupted_raw_step_telemetry",
        "uninterrupted_failure_receipt_absent",
        "model_exact",
        "optimizer_exact",
        "scheduler_exact",
        "rng_exact",
        "last_metrics_exact",
    }
    if not isinstance(equivalence, dict) or set(equivalence) != equivalence_keys:
        raise ValueError("Stage-2 resume equivalence key 집합이 exact하지 않습니다")
    if (
        equivalence["schema"] != STAGE2_PRETRAIN_RESUME_EQUIVALENCE_SCHEMA
        or equivalence["status"] != "PASS"
        or equivalence["external_experiment_contract_sha256"] != external_sha256
        or equivalence["a100_environment_sha256"] != environment_sha256
        or equivalence["completed_step"] != 500
        or equivalence["comparison_algorithm"] != "exact_recursive_torch_state_v1"
        or equivalence["resumed_checkpoint_sha256"] != step_500_sha
        or equivalence["uninterrupted_failure_receipt_absent"] is not True
    ):
        raise ValueError("Stage-2 resume equivalence semantic이 current smoke와 다릅니다")
    uninterrupted, _, uninterrupted_path = _load_bound_checkpoint(
        root,
        checkpoint_ref=equivalence["uninterrupted_checkpoint"],
        binding_ref=equivalence["uninterrupted_binding"],
        expected_checkpoint_path=None,
        expected_binding_path=None,
        expected_bindings=expected_bindings,
        completed_steps=500,
        label="Stage-2 uninterrupted step 500",
    )
    uninterrupted_dir = uninterrupted_path.parent
    if uninterrupted_dir == run_dir or equivalence_path.parent == run_dir:
        raise ValueError("Stage-2 uninterrupted evidence가 resumed run에서 재사용됐습니다")
    uninterrupted_identity, _, _, uninterrupted_identity_path = _load_json_ref(
        root,
        equivalence["uninterrupted_run_identity"],
        label="Stage-2 uninterrupted run identity",
    )
    if uninterrupted_identity_path.parent != uninterrupted_dir:
        raise ValueError("Stage-2 uninterrupted identity/checkpoint directory가 다릅니다")
    expected_uninterrupted_identity = {**expected_identity, "run_until_step": 500}
    if uninterrupted_identity != expected_uninterrupted_identity:
        raise ValueError("Stage-2 uninterrupted run identity가 fresh 500-step이 아닙니다")
    uninterrupted_telemetry, uninterrupted_telemetry_sha, uninterrupted_telemetry_path = (
        _load_binary_ref(
            root,
            equivalence["uninterrupted_raw_step_telemetry"],
            label="Stage-2 uninterrupted raw telemetry",
        )
    )
    if uninterrupted_telemetry_path.parent != uninterrupted_dir:
        raise ValueError("Stage-2 uninterrupted telemetry/checkpoint directory가 다릅니다")
    if uninterrupted_telemetry_sha == telemetry_sha:
        raise ValueError("Stage-2 uninterrupted telemetry가 resumed telemetry 복사본입니다")
    _telemetry_rows(
        uninterrupted_telemetry,
        expected_steps=500,
        expected_batch_size=96,
        expected_target_samples=int(execution["target_samples"]),
        utilization_every=utilization_every,
        label="Stage-2 uninterrupted smoke telemetry",
    )
    if (uninterrupted_dir / "failure.json").exists():
        raise ValueError("Stage-2 uninterrupted run에 OOM/runtime failure receipt가 있습니다")

    comparisons = {
        "model_exact": _state_exact_equal(uninterrupted.get("model"), step_500.get("model")),
        "optimizer_exact": _state_exact_equal(
            uninterrupted.get("optimizer"), step_500.get("optimizer")
        ),
        "scheduler_exact": _state_exact_equal(
            uninterrupted.get("scheduler"), step_500.get("scheduler")
        ),
        "rng_exact": all(
            _state_exact_equal(uninterrupted.get(key), step_500.get(key))
            for key in (
                "torch_rng_state",
                "numpy_rng_state",
                "python_rng_state",
                "cuda_rng_state_all",
            )
        ),
        "last_metrics_exact": _state_exact_equal(
            uninterrupted.get("last_metrics"), step_500.get("last_metrics")
        ),
    }
    if any(equivalence[key] is not value for key, value in comparisons.items()) or not all(
        comparisons.values()
    ):
        raise ValueError("Stage-2 uninterrupted/resume checkpoint가 수치 동등하지 않습니다")
    # step 200이 actual resumed chain의 prefix checkpoint였음을 checkpoint
    # semantic에서도 다시 확인한다. 값을 사용해 linter와 감사에서
    # 누락된 decode로 오인하지 않게 한다.
    if step_200.get("completed_steps") != 200:
        raise ValueError("Stage-2 explicit resume prefix checkpoint step이 200이 아닙니다")
    return acceptance_sha


@dataclass(frozen=True)
class _PreparedStage2Source:
    """source SHA 검증·resample·actual P*n valid-start 계산을 한 immutable cache."""

    samples: np.ndarray
    valid_starts: tuple[int, ...]


class Stage2PublicTensorLoader:
    """manifest row를 global-sample-index 기반 deterministic tensor로 decode한다."""

    def __init__(
        self,
        *,
        repository_root: Path,
        admission: Stage2PretrainTypedAdmission,
        target_samples: int,
        cache_items: int = 32,
        valid_start_candidates: int = 64,
        pin_memory: bool = False,
        model_actuator_limit_abs: float | None = None,
    ) -> None:
        self.root = repository_root
        self.admission = admission
        self.target_samples = int(target_samples)
        if self.target_samples < 4096 or self.target_samples % 256:
            raise ValueError("Stage-2 target_samples는 4096 이상 256의 배수여야 합니다")
        required = int(admission.plant_binding.required_prefix_samples)
        self.prefix_samples = max(256, ((required + 255) // 256) * 256)
        self.lead_samples = int(
            admission.plant_binding.training_timing_contract.digital_reference_lead_samples
        )
        operating_level = admission.plant_binding.source_operating_level
        if operating_level is None:
            raise ValueError(
                "Stage-2 physical source operating level/P-S feasibility binding이 없습니다"
            )
        self.operating_level = operating_level
        if (
            model_actuator_limit_abs is None
            or not math.isfinite(float(model_actuator_limit_abs))
            or not math.isclose(
                float(model_actuator_limit_abs),
                float(operating_level.actuator_limit_abs),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            raise ValueError(
                "Stage-2 model limiter가 physical operating-level actuator limit와 exact하지 않습니다"
            )
        self.model_actuator_limit_abs = float(model_actuator_limit_abs)
        self.sources = {
            int(source.dataset_index): source
            for source in admission.data_binding.sources
        }
        if isinstance(cache_items, bool) or not 1 <= int(cache_items) <= 4096:
            raise ValueError("Stage-2 source cache_items는 1..4096 int여야 합니다")
        if (
            isinstance(valid_start_candidates, bool)
            or not 16 <= int(valid_start_candidates) <= 4096
        ):
            raise ValueError("Stage-2 valid-start 후보 수는 16..4096 int여야 합니다")
        self._cache: OrderedDict[int, _PreparedStage2Source] = OrderedDict()
        self._cache_items = int(cache_items)
        self._valid_start_candidates = int(valid_start_candidates)
        self.pin_memory = bool(pin_memory)
        self._cache_lock = threading.RLock()
        self._source_locks = {index: threading.Lock() for index in self.sources}
        # waveform LRU가 빠져도 실제 P*n로 계산한 valid-start는 source SHA가 같은
        # 현재 run 동안 유지한다. 수십 GB decode cache를 만들지 않으면서 O(N*FIR)
        # 계산은 source당 정확히 한 번만 수행한다.
        self._valid_start_cache: dict[int, tuple[int, ...]] = {}

    def _decode_bytes(self, source: Stage2PretrainSource) -> np.ndarray:
        """held bytes를 한 번 SHA 검증하고 canonical 48 kHz mono로 decode한다."""

        path = self.root / source.relative_path
        content, actual_sha = _snapshot_regular_file(path)
        if actual_sha != source.content_sha256:
            raise RuntimeError(
                f"Stage-2 source bytes가 admission 후 바뀌었습니다: {source.relative_path}"
            )
        values, rate = sf.read(io.BytesIO(content), dtype="float32", always_2d=True)
        if int(rate) != int(source.native_sample_rate):
            raise RuntimeError("Stage-2 source sample rate가 manifest와 다릅니다")
        mono = np.asarray(values.mean(axis=1), dtype=np.float32)
        if mono.size < 2 or not np.all(np.isfinite(mono)):
            raise RuntimeError("Stage-2 source decode가 finite/nonempty가 아닙니다")
        if int(rate) != 48_000:
            divisor = math.gcd(int(rate), 48_000)
            mono = resample_poly(
                mono, 48_000 // divisor, int(rate) // divisor
            ).astype(np.float32)
        mono.setflags(write=False)
        return mono

    def _prepare(self, source: Stage2PretrainSource) -> _PreparedStage2Source:
        index = int(source.dataset_index)
        with self._cache_lock:
            cached = self._cache.get(index)
            if cached is not None:
                self._cache.move_to_end(index)
                return cached
        # 같은 source가 여러 prefetched batch에 겹쳐도 SHA/decode/valid-start 계산은
        # 정확히 한 worker만 수행한다. 서로 다른 source는 병렬로 준비할 수 있다.
        with self._source_locks[index]:
            with self._cache_lock:
                cached = self._cache.get(index)
                if cached is not None:
                    self._cache.move_to_end(index)
                    return cached
            samples = self._decode_bytes(source)
            with self._cache_lock:
                valid_starts = self._valid_start_cache.get(index)
            if valid_starts is None:
                prepared = self._precompute_valid_starts(samples)
                with self._cache_lock:
                    self._valid_start_cache[index] = prepared.valid_starts
            else:
                cached_samples = np.ascontiguousarray(samples, dtype=np.float32)
                cached_samples.setflags(write=False)
                prepared = _PreparedStage2Source(
                    samples=cached_samples,
                    valid_starts=valid_starts,
                )
            with self._cache_lock:
                existing = self._cache.get(index)
                if existing is not None:
                    self._cache.move_to_end(index)
                    return existing
                self._cache[index] = prepared
                self._cache.move_to_end(index)
                while len(self._cache) > self._cache_items:
                    self._cache.popitem(last=False)
            return prepared

    @staticmethod
    def _density_valid(value: np.ndarray) -> bool:
        samples = int(value.size)
        spectrum = np.fft.rfft(np.asarray(value, dtype=np.float64))
        power = spectrum.real**2 + spectrum.imag**2
        frequencies = np.fft.rfftfreq(samples, 1.0 / 48_000)
        contract = Stage2TwoKilohertzContract.canonical()
        densities: list[float] = []
        union_sum = 0.0
        union_bins = 0
        for lower, upper in contract.octave_objective_bands_hz:
            mask = (frequencies >= lower) & (frequencies < upper)
            if not np.any(mask):
                return False
            selected = power[mask]
            densities.append(float(np.mean(selected)))
            union_sum += float(np.sum(selected))
            union_bins += int(np.sum(mask))
        baseline = union_sum / float(union_bins)
        sentinel_mask = (
            frequencies >= STAGE2_2KHZ_ONE_POINT_SIX_SENTINEL_BAND_HZ[0]
        ) & (frequencies < STAGE2_2KHZ_ONE_POINT_SIX_SENTINEL_BAND_HZ[1])
        if not np.any(sentinel_mask):
            return False
        sentinel_density = float(np.mean(power[sentinel_mask]))
        return baseline > 0.0 and all(
            value / baseline >= 0.25 for value in [*densities, sentinel_density]
        )

    def _precompute_valid_starts(self, source: np.ndarray) -> _PreparedStage2Source:
        """소스당 P convolution 한 번으로 exact deterministic valid starts를 만든다.

        기존 구현은 batch sample마다 최대 80개의 ``np.convolve``를 수행했다. 여기서는
        같은 actual P FIR/delay와 target density gate를 유지하되, source SHA로 고정된
        배열에 overlap-add convolution을 한 번만 수행한다. prefix가 FIR history보다
        길다는 typed plant invariant 때문에 target 구간은 crop-local zero-state 계산과
        동일하다.
        """

        needed = self.prefix_samples + self.target_samples + self.lead_samples
        if source.size < needed:
            repeats = int(math.ceil(needed / max(1, source.size))) + 1
            source = np.tile(source, repeats).astype(np.float32, copy=False)
        source = np.ascontiguousarray(source, dtype=np.float32)
        source.setflags(write=False)
        maximum = int(source.size) - needed
        if maximum == 0:
            candidates = [0]
        else:
            candidates = [
                int(round(value))
                for value in np.linspace(
                    0, maximum, num=self._valid_start_candidates
                )
            ]
            candidates = list(dict.fromkeys(candidates))
        operator = self.admission.plant_binding.primary_operator
        fir = np.asarray(operator.post_onset_fir, dtype=np.float64)
        filtered = oaconvolve(
            np.asarray(source, dtype=np.float64), fir, mode="full"
        )[: source.size]
        delay = int(operator.coarse_delay_samples)
        target_d = np.zeros(source.size, dtype=np.float64)
        if delay < source.size:
            target_d[delay:] = filtered[: source.size - delay]
        total = self.prefix_samples + self.target_samples
        valid: list[int] = []
        for start in candidates:
            target = target_d[
                start + self.prefix_samples : start + total
            ]
            if target.size == self.target_samples and self._density_valid(target):
                valid.append(int(start))
        if not valid:
            raise RuntimeError(
                "Stage-2 source의 precomputed actual P*n 5-octave valid start가 없습니다"
            )
        return _PreparedStage2Source(samples=source, valid_starts=tuple(valid))

    def _crop(self, prepared: _PreparedStage2Source, *, seed: int) -> np.ndarray:
        needed = self.prefix_samples + self.target_samples + self.lead_samples
        generator = np.random.Generator(np.random.PCG64(int(seed)))
        position = int(generator.integers(0, len(prepared.valid_starts)))
        start = int(prepared.valid_starts[position])
        crop = np.asarray(prepared.samples[start : start + needed], dtype=np.float32)
        if crop.size != needed:
            raise RuntimeError("Stage-2 precomputed valid start crop 길이가 다릅니다")
        return crop

    def build(self, identity: Stage2ActualBatchIdentity) -> Stage2TensorBatch:
        clean_rows: list[np.ndarray] = []
        for dataset_index, seed, global_sample_index in zip(
            identity.dataset_indices,
            identity.augmentation_seeds,
            identity.global_sample_indices,
            strict=True,
        ):
            source = self.sources.get(int(dataset_index))
            if source is None:
                raise RuntimeError("Stage-2 sampler index가 manifest source mapping에 없습니다")
            crop = self._crop(self._prepare(source), seed=int(seed))
            generator = np.random.Generator(np.random.PCG64(int(seed) ^ 0xD1A5E2))
            gain_db = float(
                generator.uniform(
                    self.operating_level.augmentation_gain_db_minimum,
                    self.operating_level.augmentation_gain_db_maximum,
                )
            )
            gain = float(10.0 ** (gain_db / 20.0))
            polarity = -1.0 if int(generator.integers(0, 2)) else 1.0
            source_peak = float(np.max(np.abs(np.asarray(crop, dtype=np.float64))))
            if not math.isfinite(source_peak) or source_peak <= 0.0:
                raise RuntimeError(
                    f"Stage-2 source crop peak가 유효하지 않습니다: {global_sample_index}"
                )
            scaled = (
                np.asarray(crop, dtype=np.float64)
                / source_peak
                * float(self.operating_level.source_operating_peak_abs)
                * gain
                * polarity
            )
            cap = float(self.operating_level.post_gain_hard_peak_cap_abs)
            scaled = np.clip(scaled, -cap, cap).astype(np.float32)
            actual_peak = float(np.max(np.abs(scaled.astype(np.float64))))
            if (
                not np.all(np.isfinite(scaled))
                or actual_peak <= 0.0
                or actual_peak > cap + 1.0e-12
            ):
                raise RuntimeError(
                    f"Stage-2 post-gain source peak cap 위반: {global_sample_index}"
                )
            clean_rows.append(scaled)
        clean = torch.from_numpy(np.stack(clean_rows, axis=0)).unsqueeze(1)
        total = self.prefix_samples + self.target_samples
        preview = clean[..., self.lead_samples : self.lead_samples + total]
        zeros = torch.zeros_like(preview)
        model_input = torch.cat((preview, zeros), dim=1)
        causal = CausalPrefixBatchV1(
            x_prefix=model_input[..., : self.prefix_samples],
            x_target=model_input[..., self.prefix_samples :],
            source_sha256=identity.source_sha256,
            clean_playback_source_sha256=identity.source_sha256,
            clean_playback_timeline=clean,
            controller_reference_preaugmentation=preview,
            training_timing_contract_sha256=(
                self.admission.plant_binding.training_timing_contract_sha256
            ),
            segment_prefix_start_samples=(0,) * len(identity.source_sha256),
            segment_target_start_samples=(self.prefix_samples,)
            * len(identity.source_sha256),
            global_sample_indices=identity.global_sample_indices,
            state_origin=CausalPrefixStateOriginV1(
                kind="segment_start_zero_state",
                binding_sha256=self.admission.plant_binding.digest(),
                source_sha256=identity.source_sha256,
            ),
        )
        batch = Stage2TensorBatch(
            causal=causal,
            dataset_indices=identity.dataset_indices,
            manifest_row_sha256=identity.manifest_row_sha256,
            augmentation_seeds=identity.augmentation_seeds,
        )
        return _pin_stage2_tensor_batch(batch) if self.pin_memory else batch


def _pin_stage2_tensor_batch(batch: Stage2TensorBatch) -> Stage2TensorBatch:
    causal = batch.causal
    return Stage2TensorBatch(
        causal=CausalPrefixBatchV1(
            x_prefix=causal.x_prefix.pin_memory(),
            x_target=causal.x_target.pin_memory(),
            source_sha256=causal.source_sha256,
            clean_playback_source_sha256=causal.clean_playback_source_sha256,
            clean_playback_timeline=causal.clean_playback_timeline.pin_memory(),
            controller_reference_preaugmentation=(
                causal.controller_reference_preaugmentation.pin_memory()
            ),
            training_timing_contract_sha256=causal.training_timing_contract_sha256,
            segment_prefix_start_samples=causal.segment_prefix_start_samples,
            segment_target_start_samples=causal.segment_target_start_samples,
            global_sample_indices=causal.global_sample_indices,
            state_origin=causal.state_origin,
        ),
        dataset_indices=batch.dataset_indices,
        manifest_row_sha256=batch.manifest_row_sha256,
        augmentation_seeds=batch.augmentation_seeds,
    )


class Stage2DeterministicBatchPrefetcher:
    """global-step 순서를 바꾸지 않는 bounded CPU thread prefetcher."""

    def __init__(
        self,
        *,
        loader: Stage2PublicTensorLoader,
        sampler: Any,
        start_step: int,
        end_step: int,
        workers: int,
        prefetch_batches: int,
    ) -> None:
        if not 1 <= int(workers) <= 64:
            raise ValueError("Stage-2 loader workers는 1..64여야 합니다")
        if not 1 <= int(prefetch_batches) <= 64:
            raise ValueError("Stage-2 bounded prefetch batch 수는 1..64여야 합니다")
        if not 0 <= int(start_step) < int(end_step):
            raise ValueError("Stage-2 prefetch step 범위가 잘못됐습니다")
        self.loader = loader
        self.sampler = sampler
        self.next_submit = int(start_step)
        self.next_yield = int(start_step)
        self.end_step = int(end_step)
        self.prefetch_batches = int(prefetch_batches)
        self.executor = ThreadPoolExecutor(
            max_workers=int(workers), thread_name_prefix="stage2-data"
        )
        self.pending: OrderedDict[
            int, tuple[Stage2ActualBatchIdentity, Future[Stage2TensorBatch]]
        ] = OrderedDict()
        self._fill()

    def _fill(self) -> None:
        while (
            self.next_submit < self.end_step
            and len(self.pending) < self.prefetch_batches
        ):
            step = self.next_submit
            identity = Stage2ActualBatchIdentity.from_sampler(
                self.sampler, global_step=step
            )
            self.pending[step] = (
                identity,
                self.executor.submit(self.loader.build, identity),
            )
            self.next_submit += 1

    def __iter__(self) -> Iterator[
        tuple[int, Stage2ActualBatchIdentity, Stage2TensorBatch]
    ]:
        return self

    def __next__(self) -> tuple[int, Stage2ActualBatchIdentity, Stage2TensorBatch]:
        if self.next_yield >= self.end_step:
            self.close()
            raise StopIteration
        step = self.next_yield
        entry = self.pending.pop(step, None)
        if entry is None:
            self.close()
            raise RuntimeError("Stage-2 prefetch ordered step이 pending queue에 없습니다")
        identity, future = entry
        try:
            batch = future.result()
        except BaseException:
            self.close(cancel=True)
            raise
        self.next_yield += 1
        self._fill()
        return step, identity, batch

    def close(self, *, cancel: bool = False) -> None:
        for _, future in self.pending.values():
            if cancel:
                future.cancel()
        self.pending.clear()
        self.executor.shutdown(wait=True, cancel_futures=cancel)

    def __enter__(self) -> "Stage2DeterministicBatchPrefetcher":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close(cancel=True)


class Stage2ScratchPretrainRunner:
    """200/500 smoke와 100k를 같은 checkpoint 계약으로 실행한다."""

    def __init__(
        self,
        *,
        repository_root: str | Path,
        campaign: Mapping[str, Any],
        run_until_step: int,
        resume: str | Path | None = None,
        smoke_acceptance: str | Path | None = None,
        _allow_cpu_test: bool = False,
    ) -> None:
        self.root = Path(repository_root).resolve(strict=True)
        preverified_commit = _preverify_campaign_checkout(self.root, campaign)
        self.admission, self.profile, self.anchors = load_ready_stage2_pretrain_launch(
            campaign, repo_root=self.root
        )
        execution = self.profile["execution"]
        total_steps = int(execution["schedule"]["total_steps"])
        self.run_until_step = int(run_until_step)
        if not 1 <= self.run_until_step <= total_steps:
            raise ValueError(f"run_until_step은 1..{total_steps} 범위여야 합니다")
        # 실제 A100 200→명시적 resume→500 및 별도 uninterrupted 500의 raw
        # 수치등가 evidence 없이는 CUDA/environment 조회조차 시작하지 않는다.
        # 단순 JSON status=true는 아래 verifier가 실제 checkpoint/telemetry bytes를
        # 다시 계산하므로 admission이 아니다.
        if self.run_until_step > 500 and smoke_acceptance is None:
            raise ValueError(
                "Stage-2 500-step 이후는 raw-bound smoke acceptance가 필요합니다"
            )
        if int(os.environ.get("WORLD_SIZE", "1")) != int(execution["required_world_size"]):
            raise RuntimeError("Stage-2 canonical pretrain은 exact world_size=1입니다")
        if int(self.admission.sampler.batch_size) != int(execution["batch_size"]):
            raise RuntimeError(
                "Stage-2 production batch가 profile/sampler receipt와 다릅니다"
            )
        pipeline = execution.get("data_pipeline")
        if pipeline is None and _allow_cpu_test:
            pipeline = {
                "loader_workers": 1,
                "bounded_prefetch_batches": 1,
                "source_cache_items": 8,
                "valid_start_candidates_per_source": 64,
                "pin_memory": False,
                "non_blocking_h2d": False,
            }
        if not isinstance(pipeline, Mapping):
            raise ValueError("Stage-2 data_pipeline profile이 없습니다")
        self.loader_workers = int(pipeline["loader_workers"])
        self.prefetch_batches = int(pipeline["bounded_prefetch_batches"])
        self.pin_memory = bool(pipeline["pin_memory"]) and not _allow_cpu_test
        self.non_blocking_h2d = bool(pipeline["non_blocking_h2d"]) and not _allow_cpu_test
        if resume is None:
            self.resume = None
        else:
            candidate = Path(resume)
            if not candidate.is_absolute():
                candidate = self.root / candidate
            self.resume = candidate.parent.resolve(strict=True) / candidate.name
        external_sha = str(self.anchors["external_experiment_contract_sha256"])
        seed = int(self.profile["seed"])
        self.run_dir = self.root / "runs" / f"stage2_pretrain_{external_sha[:12]}_{seed}"
        if self.resume is not None and self.resume.parent != self.run_dir:
            raise ValueError("explicit resume checkpoint가 exact Stage-2 contract run directory 밖입니다")
        if self.run_until_step > 500:
            resume_match = (
                re.fullmatch(r"step_([0-9]{6})\.pt", self.resume.name)
                if self.resume is not None
                else None
            )
            if resume_match is None or int(resume_match.group(1)) < 500:
                raise ValueError(
                    "Stage-2 100k는 step 500 이상 checkpoint의 명시적 resume만 허용합니다"
                )
            acceptance_candidate = Path(smoke_acceptance)  # type: ignore[arg-type]
            if acceptance_candidate.is_absolute():
                resolved_acceptance = (
                    acceptance_candidate.parent.resolve(strict=True)
                    / acceptance_candidate.name
                )
                try:
                    resolved_acceptance.relative_to(self.root)
                except ValueError as exc:
                    raise ValueError(
                        "Stage-2 smoke acceptance는 repository 내부여야 합니다"
                    ) from exc
                self.smoke_acceptance_path = resolved_acceptance
            else:
                self.smoke_acceptance_path = _inside_repository(
                    self.root,
                    acceptance_candidate,
                    label="Stage-2 smoke acceptance",
                )
            # symlink/부재/비정규 파일을 CUDA 전에 fail-close한다. 전체 semantic은
            # current A100 environment SHA를 얻은 직후 held snapshot으로 검증한다.
            _snapshot_regular_file(self.smoke_acceptance_path)
        else:
            if smoke_acceptance is not None:
                raise ValueError(
                    "Stage-2 smoke acceptance는 500-step 이후 실행에만 사용합니다"
                )
            self.smoke_acceptance_path = None
        # admission이 돌려준 anchor도 사전 검증한 commit과 동일해야 한다.
        self.execution_repository_commit_sha = preverified_commit
        if _allow_cpu_test:
            self.a100_environment = {
                "schema": "stage2_2khz_cpu_diagnostic_environment_v1",
                "diagnostic_cpu_test": True,
                "production_eligible": False,
            }
            self.device = torch.device("cpu")
        else:
            self.a100_environment = configure_and_collect_stage2_a100_environment(
                torch,
                nvidia_smi_l_output=_nvidia_smi_l_snapshot(self.root),
            )
            self.device = torch.device("cuda:0")
        self.a100_environment_sha256 = stage2_a100_environment_sha256(
            self.a100_environment
        )
        self.smoke_acceptance_sha256: str | None = None
        if self.run_until_step > 500:
            assert self.smoke_acceptance_path is not None
            self.smoke_acceptance_sha256 = _validate_stage2_smoke_acceptance(
                self.root,
                acceptance_path=self.smoke_acceptance_path,
                run_dir=self.run_dir,
                external_sha256=external_sha,
                environment=self.a100_environment,
                environment_sha256=self.a100_environment_sha256,
                profile=self.profile,
                expected_bindings=self._binding_metadata(
                    include_smoke_acceptance=False
                ),
            )
        # 모델 parameter 초기화보다 먼저 seed를 고정해야 fresh scratch run끼리도
        # 같고, 1-step stop→explicit resume가 uninterrupted 결과와 같아진다.
        self._seed_all(seed)

        model_path = self.root / str(self.anchors["model_config_path"])
        if _sha256_file(model_path) != self.anchors["model_config_sha256"]:
            raise RuntimeError("Stage-2 model config bytes SHA가 admission 후 바뀌었습니다")
        model_cfg = yaml.safe_load(model_path.read_text(encoding="utf-8"))
        if not isinstance(model_cfg, dict) or int(model_cfg.get("in_channels", 0)) != 2:
            raise ValueError("Stage-2 Tiny model config가 2-channel digital-ref 계약이 아닙니다")
        model_actuator_limit = require_stage2_actuator_limit(model_cfg)
        self.model = build_model(model_cfg).to(self.device)
        prefix = Stage2CausalPrefixAdapter.from_verified_binding(
            self.admission.plant_binding
        ).to(self.device)
        criterion = Stage2TwoKilohertzLoss(self.admission.loss_config).to(self.device)
        self.consumer = Stage2TwoKilohertzTrainerAdapter.from_verified_components(
            prefix,
            criterion,
            manifest_bundle_sha256=self.admission.data_binding.manifest_bundle_sha256,
            sampler_receipt_sha256=self.admission.sampler_receipt_sha256,
        )
        optimizer_cfg = execution["optimizer"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(optimizer_cfg["lr"]),
            weight_decay=float(optimizer_cfg["weight_decay"]),
            betas=tuple(float(value) for value in optimizer_cfg["betas"]),
        )
        schedule_cfg = execution["schedule"]
        warmup = int(schedule_cfg["warmup_steps"])
        total = int(schedule_cfg["total_steps"])
        minimum_ratio = float(schedule_cfg["min_lr"]) / float(optimizer_cfg["lr"])

        def lr_scale(step: int) -> float:
            if step < warmup:
                return max(1.0 / max(1, warmup), float(step + 1) / float(warmup))
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return minimum_ratio + (1.0 - minimum_ratio) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_scale)
        self.loader = Stage2PublicTensorLoader(
            repository_root=self.root,
            admission=self.admission,
            target_samples=int(execution["target_samples"]),
            cache_items=int(pipeline["source_cache_items"]),
            valid_start_candidates=int(
                pipeline["valid_start_candidates_per_source"]
            ),
            pin_memory=self.pin_memory,
            model_actuator_limit_abs=model_actuator_limit,
        )
        self.grad_clip = float(execution["grad_clip_norm"])
        self.checkpoint_every = int(execution["checkpoint_every_steps"])
        self.smoke_milestones = frozenset(int(value) for value in execution["smoke_milestones"])
        self.diagnostic_cpu_test = bool(_allow_cpu_test)
        if self.diagnostic_cpu_test and self.run_until_step > max(self.smoke_milestones):
            raise ValueError("CPU diagnostic runner는 smoke milestone 이후를 실행할 수 없습니다")
        self.start_step = 0
        if self.resume is not None:
            self._load_resume(self.resume)
        else:
            # 모든 typed artifact, exact clean commit, GPU/model/criterion/optimizer가
            # 구성된 뒤에만 새 scratch run identity를 no-replace로 만든다.
            self.run_dir.mkdir(parents=True, exist_ok=False)
            _write_json_exclusive(
                self.run_dir / "environment.json", self.a100_environment
            )
            _write_json_exclusive(
                self.run_dir / "run_identity.json",
                {
                    "schema": STAGE2_PRETRAIN_RUN_SCHEMA,
                    "runner_schema": STAGE2_PRETRAIN_RUNNER_SCHEMA,
                    "repository_commit_sha": self.execution_repository_commit_sha,
                    "external_experiment_contract_sha256": external_sha,
                    "seed": seed,
                    "scratch_pretrain": True,
                    "automatic_resume_allowed": False,
                    "run_until_step": self.run_until_step,
                    "a100_environment_sha256": self.a100_environment_sha256,
                    "production_batch_size": int(execution["batch_size"]),
                    "loader_workers": self.loader_workers,
                    "bounded_prefetch_batches": self.prefetch_batches,
                },
            )

    def _seed_all(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    def _binding_metadata(
        self, *, include_smoke_acceptance: bool = True
    ) -> dict[str, Any]:
        return {
            "runner_schema": STAGE2_PRETRAIN_RUNNER_SCHEMA,
            "repository_commit_sha": self.execution_repository_commit_sha,
            "stage2_contract_id": Stage2TwoKilohertzContract.canonical().contract_id,
            "stage2_contract_sha256": Stage2TwoKilohertzContract.canonical().digest(),
            "primary_path_sha256": self.admission.plant_binding.primary_path_sha256,
            "secondary_path_sha256": self.admission.plant_binding.secondary_path_sha256,
            "plant_binding_sha256": self.admission.plant_binding.binding_file_sha256,
            "plant_binding_runtime_sha256": self.admission.plant_binding.digest(),
            "manifest_bundle_sha256": self.admission.data_binding.manifest_bundle_sha256,
            "training_profile_sha256": self.anchors["pretrain_profile_sha256"],
            "evaluation_policy_sha256": self.anchors["evaluation_policy_sha256"],
            "model_config_sha256": self.anchors["model_config_sha256"],
            "external_experiment_contract_sha256": self.anchors[
                "external_experiment_contract_sha256"
            ],
            "criterion_receipt_sha256": self.admission.criterion_receipt_sha256,
            "sampler_receipt_sha256": self.admission.sampler_receipt_sha256,
            "dnh_calibration_receipt_sha256": (
                self.admission.dnh_calibration_receipt_sha256
            ),
            "scratch_pretrain": True,
            "legacy_origin": False,
            "automatic_resume_allowed": False,
            "a100_environment_sha256": self.a100_environment_sha256,
            "smoke_acceptance_sha256": (
                self.smoke_acceptance_sha256
                if include_smoke_acceptance
                else None
            ),
        }

    def _load_resume(self, path: Path) -> None:
        try:
            environment_bytes, _ = _snapshot_regular_file(
                self.run_dir / "environment.json"
            )
            persisted_environment = json.loads(environment_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Stage-2 resume environment receipt를 읽을 수 없습니다") from exc
        if (
            stage2_a100_environment_sha256(persisted_environment)
            != self.a100_environment_sha256
            or persisted_environment != self.a100_environment
        ):
            raise ValueError("Stage-2 resume A100 environment가 최초 run과 다릅니다")
        identity_path = self.run_dir / "run_identity.json"
        try:
            identity_bytes, _ = _snapshot_regular_file(identity_path)
            identity = json.loads(identity_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Stage-2 resume run identity를 읽을 수 없습니다") from exc
        expected_identity_keys = {
            "schema",
            "runner_schema",
            "repository_commit_sha",
            "external_experiment_contract_sha256",
            "seed",
            "scratch_pretrain",
            "automatic_resume_allowed",
            "run_until_step",
            "a100_environment_sha256",
            "production_batch_size",
            "loader_workers",
            "bounded_prefetch_batches",
        }
        if not isinstance(identity, dict) or set(identity) != expected_identity_keys:
            raise ValueError("Stage-2 resume run identity key 집합이 exact하지 않습니다")
        expected_identity = {
            "schema": STAGE2_PRETRAIN_RUN_SCHEMA,
            "runner_schema": STAGE2_PRETRAIN_RUNNER_SCHEMA,
            "repository_commit_sha": self.execution_repository_commit_sha,
            "external_experiment_contract_sha256": self.anchors[
                "external_experiment_contract_sha256"
            ],
            "seed": int(self.profile["seed"]),
            "scratch_pretrain": True,
            "automatic_resume_allowed": False,
            "a100_environment_sha256": self.a100_environment_sha256,
            "production_batch_size": int(self.profile["execution"]["batch_size"]),
            "loader_workers": self.loader_workers,
            "bounded_prefetch_batches": self.prefetch_batches,
        }
        if any(identity[key] != value for key, value in expected_identity.items()):
            raise ValueError("Stage-2 resume run identity가 current external contract와 다릅니다")
        # pickle decoder보다 먼저 held regular file SHA와 flattened sidecar의 전체
        # contract binding을 검증한다. sidecar가 없거나 재서명됐으면 torch.load 자체를
        # 호출하지 않는다.
        checkpoint_bytes, checkpoint_sha = _snapshot_regular_file(path)
        sidecar_path = path.with_suffix(".binding.json")
        try:
            sidecar_bytes, _ = _snapshot_regular_file(sidecar_path)
            sidecar = json.loads(sidecar_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Stage-2 resume checkpoint binding sidecar를 읽을 수 없습니다") from exc
        if not isinstance(sidecar, dict):
            raise ValueError("Stage-2 resume checkpoint binding sidecar가 mapping이 아닙니다")
        completed_steps = sidecar.get("completed_steps")
        if isinstance(completed_steps, bool) or not isinstance(completed_steps, int):
            raise ValueError("Stage-2 resume sidecar completed_steps가 정수가 아닙니다")
        resume_bindings = self._binding_metadata(
            include_smoke_acceptance=int(completed_steps) > 500
        )
        expected_sidecar = stage2_checkpoint_binding_payload(
            checkpoint_sha256=checkpoint_sha,
            bindings=resume_bindings,
            completed_steps=completed_steps,
            production_execution=not self.diagnostic_cpu_test,
        )
        if sidecar != expected_sidecar:
            raise ValueError("Stage-2 resume checkpoint SHA/sidecar/contract binding이 다릅니다")
        checkpoint = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
        )
        expected_checkpoint_schema = (
            STAGE2_PRETRAIN_DIAGNOSTIC_CHECKPOINT_SCHEMA
            if self.diagnostic_cpu_test
            else STAGE2_PRETRAIN_CHECKPOINT_SCHEMA
        )
        if not isinstance(checkpoint, dict) or checkpoint.get("schema") != expected_checkpoint_schema:
            raise ValueError("explicit resume checkpoint가 Stage-2 schema가 아닙니다")
        if checkpoint.get("bindings") != resume_bindings:
            raise ValueError("explicit resume checkpoint의 전체 Stage-2 contract SHA가 다릅니다")
        required = {
            "schema",
            "completed_steps",
            "init_eligible",
            "bindings",
            "model",
            "optimizer",
            "scheduler",
            "step",
            "torch_rng_state",
            "numpy_rng_state",
            "python_rng_state",
            "cuda_rng_state_all",
            "last_metrics",
        }
        if set(checkpoint) != required:
            raise ValueError("Stage-2 resume checkpoint key 집합이 exact하지 않습니다")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_step = int(checkpoint["step"])
        if self.start_step != int(completed_steps) or int(checkpoint["completed_steps"]) != int(
            completed_steps
        ):
            raise ValueError("Stage-2 resume checkpoint step이 held sidecar와 다릅니다")
        if not 0 <= self.start_step < self.run_until_step:
            raise ValueError("resume step이 run_until_step 이전이 아닙니다")
        torch.set_rng_state(checkpoint["torch_rng_state"])
        _restore_numpy_rng_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

    def _save_checkpoint(self, step: int, metrics: Mapping[str, float]) -> Path:
        path = self.run_dir / f"step_{step:06d}.pt"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        checkpoint = {
            "schema": (
                STAGE2_PRETRAIN_DIAGNOSTIC_CHECKPOINT_SCHEMA
                if self.diagnostic_cpu_test
                else STAGE2_PRETRAIN_CHECKPOINT_SCHEMA
            ),
            "step": int(step),
            "completed_steps": int(step),
            "init_eligible": (
                not self.diagnostic_cpu_test and int(step) == int(self.profile["steps"])
            ),
            "bindings": self._binding_metadata(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": _numpy_rng_state_payload(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []
            ),
            "last_metrics": dict(metrics),
        }
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                torch.save(checkpoint, stream)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            raise
        checkpoint_sha = _sha256_file(path)
        _write_json_exclusive(
            path.with_suffix(".binding.json"),
            stage2_checkpoint_binding_payload(
                checkpoint_sha256=checkpoint_sha,
                bindings=self._binding_metadata(),
                completed_steps=int(step),
                production_execution=not self.diagnostic_cpu_test,
            ),
        )
        return path

    def _to_device(self, batch: Stage2TensorBatch) -> Stage2TensorBatch:
        causal = batch.causal
        non_blocking = bool(self.non_blocking_h2d)
        if self.device.type == "cuda" and self.pin_memory:
            host_tensors = (
                causal.x_prefix,
                causal.x_target,
                causal.clean_playback_timeline,
                causal.controller_reference_preaugmentation,
            )
            if not all(
                tensor.device.type == "cpu" and tensor.is_pinned()
                for tensor in host_tensors
            ):
                raise RuntimeError("Stage-2 production H2D 입력이 pinned CPU tensor가 아닙니다")
        return Stage2TensorBatch(
            causal=CausalPrefixBatchV1(
                x_prefix=causal.x_prefix.to(self.device, non_blocking=non_blocking),
                x_target=causal.x_target.to(self.device, non_blocking=non_blocking),
                source_sha256=causal.source_sha256,
                clean_playback_source_sha256=causal.clean_playback_source_sha256,
                clean_playback_timeline=causal.clean_playback_timeline.to(
                    self.device, non_blocking=non_blocking
                ),
                controller_reference_preaugmentation=(
                    causal.controller_reference_preaugmentation.to(
                        self.device, non_blocking=non_blocking
                    )
                ),
                training_timing_contract_sha256=(
                    causal.training_timing_contract_sha256
                ),
                segment_prefix_start_samples=causal.segment_prefix_start_samples,
                segment_target_start_samples=causal.segment_target_start_samples,
                global_sample_indices=causal.global_sample_indices,
                state_origin=causal.state_origin,
            ),
            dataset_indices=batch.dataset_indices,
            manifest_row_sha256=batch.manifest_row_sha256,
            augmentation_seeds=batch.augmentation_seeds,
        )

    def _runtime_gpu_sample(self) -> tuple[int | None, int | None]:
        if self.device.type != "cuda":
            return None, None
        try:
            output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                cwd=self.root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            rows = [line.strip() for line in output.splitlines() if line.strip()]
            if len(rows) != 1:
                raise ValueError
            utilization, memory = (part.strip() for part in rows[0].split(","))
            utilization_value = int(utilization)
            memory_value = int(memory)
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            raise RuntimeError("Stage-2 runtime nvidia-smi telemetry를 읽을 수 없습니다") from exc
        if not 0 <= utilization_value <= 100 or memory_value < 0:
            raise RuntimeError("Stage-2 runtime nvidia-smi telemetry 범위가 잘못됐습니다")
        return utilization_value, memory_value

    def _write_smoke_performance_evidence(self, completed_step: int) -> Path:
        telemetry_path = self.run_dir / "step_telemetry.jsonl"
        content, telemetry_sha = _snapshot_regular_file(telemetry_path)
        try:
            rows = [json.loads(line) for line in content.decode("utf-8").splitlines()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Stage-2 raw step telemetry가 손상됐습니다") from exc
        selected = [row for row in rows if int(row.get("step", -1)) <= completed_step]
        if [int(row.get("step", -1)) for row in selected] != list(
            range(1, completed_step + 1)
        ):
            raise RuntimeError("Stage-2 smoke telemetry step sequence가 exact하지 않습니다")

        def percentile(key: str, q: float) -> float:
            values = np.asarray([float(row[key]) for row in selected], dtype=np.float64)
            if values.size != completed_step or not np.all(np.isfinite(values)):
                raise RuntimeError(f"Stage-2 smoke telemetry {key}가 finite/exact하지 않습니다")
            return float(np.percentile(values, q))

        utilization = [
            int(row["gpu_utilization_percent"])
            for row in selected
            if row["gpu_utilization_percent"] is not None
        ]
        if self.diagnostic_cpu_test:
            status = "DIAGNOSTIC_CPU_ONLY"
        elif int(self.profile["execution"]["batch_size"]) != 96:
            status = "BLOCKED_NOT_B96_CANDIDATE"
        elif not utilization:
            status = "BLOCKED_MISSING_GPU_UTILIZATION_OBSERVATION"
        else:
            # 실제 A100 관측 전에 임의 utilization/throughput threshold를 꾸며내지
            # 않는다. raw 분포를 이 receipt에 봉인한 뒤 운영자가 threshold와 함께
            # 별도 acceptance를 발행해야 100k 시작 결정을 내릴 수 있다.
            status = "BLOCKED_PENDING_EMPIRICAL_THRESHOLD_DECISION"
        payload = {
            "schema": STAGE2_PRETRAIN_SMOKE_PERFORMANCE_SCHEMA,
            "status": status,
            "canonical_100k_start_eligible": False,
            "completed_step": int(completed_step),
            "actual_batch_size": int(self.profile["execution"]["batch_size"]),
            "loader_workers": self.loader_workers,
            "bounded_prefetch_batches": self.prefetch_batches,
            "pin_memory": self.pin_memory,
            "non_blocking_h2d": self.non_blocking_h2d,
            "raw_step_telemetry_sha256": telemetry_sha,
            "a100_environment_sha256": self.a100_environment_sha256,
            "data_wait_ms_p50": percentile("data_wait_ms", 50.0),
            "data_wait_ms_p95": percentile("data_wait_ms", 95.0),
            "h2d_ms_p50": percentile("h2d_ms", 50.0),
            "h2d_ms_p95": percentile("h2d_ms", 95.0),
            "compute_step_ms_p50": percentile("compute_step_ms", 50.0),
            "compute_step_ms_p95": percentile("compute_step_ms", 95.0),
            "total_step_ms_p50": percentile("total_step_ms", 50.0),
            "total_step_ms_p95": percentile("total_step_ms", 95.0),
            "samples_per_second_p50": percentile("samples_per_second", 50.0),
            "gpu_utilization_observation_count": len(utilization),
            "gpu_utilization_percent_min": min(utilization) if utilization else None,
            "gpu_utilization_percent_median": (
                float(np.median(utilization)) if utilization else None
            ),
            "gpu_utilization_percent_max": max(utilization) if utilization else None,
            "performance_thresholds_declared_before_observation": False,
            "oom_or_runtime_failure": False,
        }
        path = self.run_dir / f"smoke_performance_step_{completed_step:06d}.json"
        _write_json_exclusive(path, payload)
        return path

    def _write_failure_receipt(self, exc: BaseException) -> None:
        path = self.run_dir / "failure.json"
        if path.exists():
            return
        telemetry = self.run_dir / "step_telemetry.jsonl"
        payload = {
            "schema": "stage2_2khz_pretrain_failure_v1",
            "status": "BLOCKED",
            "canonical_100k_start_eligible": False,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)[:2000],
            "cuda_out_of_memory": isinstance(exc, torch.cuda.OutOfMemoryError),
            "actual_batch_size": int(self.profile["execution"]["batch_size"]),
            "a100_environment_sha256": self.a100_environment_sha256,
            "raw_step_telemetry_sha256": (
                _sha256_file(telemetry) if telemetry.is_file() else None
            ),
        }
        _write_json_exclusive(path, payload)

    def train(self) -> Path:
        self.model.train()
        last_checkpoint: Path | None = None
        metrics_path = self.run_dir / "metrics.jsonl"
        telemetry_path = self.run_dir / "step_telemetry.jsonl"
        utilization_every = int(
            self.profile["execution"].get("telemetry", {}).get(
                "nvidia_smi_sample_every_steps", 10
            )
        )
        try:
            with Stage2DeterministicBatchPrefetcher(
                loader=self.loader,
                sampler=self.admission.sampler,
                start_step=self.start_step,
                end_step=self.run_until_step,
                workers=self.loader_workers,
                prefetch_batches=self.prefetch_batches,
            ) as prefetcher:
                iterator = iter(prefetcher)
                while True:
                    wait_started = time.perf_counter()
                    # wall-step은 ordered batch를 기다리기 시작한 시점부터 잰다.
                    # data_wait를 빼면 CPU starvation이 큰 smoke도 throughput이 높게
                    # 보이므로, canonical 판단용 total/throughput에는 반드시 포함한다.
                    step_started = wait_started
                    try:
                        step_index, identity, cpu_batch = next(iterator)
                    except StopIteration:
                        break
                    data_wait_ms = (time.perf_counter() - wait_started) * 1000.0
                    if self.device.type == "cuda":
                        h2d_start = torch.cuda.Event(enable_timing=True)
                        h2d_end = torch.cuda.Event(enable_timing=True)
                        h2d_start.record()
                        batch = self._to_device(cpu_batch)
                        h2d_end.record()
                        h2d_end.synchronize()
                        h2d_ms = float(h2d_start.elapsed_time(h2d_end))
                    else:
                        h2d_started = time.perf_counter()
                        batch = self._to_device(cpu_batch)
                        h2d_ms = (time.perf_counter() - h2d_started) * 1000.0
                    compute_started = time.perf_counter()
                    self.optimizer.zero_grad(set_to_none=True)
                    autocast = torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.device.type == "cuda",
                    )
                    with autocast:
                        result = self.consumer.compute_loss(self.model, batch, identity)
                    result.loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                    if not torch.isfinite(gradient_norm):
                        raise RuntimeError("Stage-2 gradient norm이 finite하지 않습니다")
                    self.optimizer.step()
                    self.scheduler.step()
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    compute_step_ms = (time.perf_counter() - compute_started) * 1000.0
                    completed = step_index + 1
                    metrics = {
                        **result.metrics,
                        "step": float(completed),
                        "lr": float(self.optimizer.param_groups[0]["lr"]),
                        "gradient_norm": float(gradient_norm.detach()),
                    }
                    with metrics_path.open("a", encoding="utf-8") as stream:
                        stream.write(
                            json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n"
                        )
                        stream.flush()
                    utilization, memory_used_mib = (
                        self._runtime_gpu_sample()
                        if completed % utilization_every == 0
                        else (None, None)
                    )
                    total_step_ms = (time.perf_counter() - step_started) * 1000.0
                    batch_size = len(identity.dataset_indices)
                    telemetry = {
                        "schema": STAGE2_PRETRAIN_STEP_TELEMETRY_SCHEMA,
                        "step": completed,
                        "batch_size": batch_size,
                        "global_sample_index_first": min(identity.global_sample_indices),
                        "global_sample_index_last": max(identity.global_sample_indices),
                        "data_wait_ms": float(data_wait_ms),
                        "h2d_ms": float(h2d_ms),
                        "compute_step_ms": float(compute_step_ms),
                        "total_step_ms": float(total_step_ms),
                        "samples_per_second": float(
                            batch_size * int(self.profile["execution"]["target_samples"])
                            / max(total_step_ms / 1000.0, 1e-12)
                        ),
                        "gpu_memory_allocated_bytes": (
                            int(torch.cuda.memory_allocated(self.device))
                            if self.device.type == "cuda"
                            else 0
                        ),
                        "gpu_memory_reserved_bytes": (
                            int(torch.cuda.memory_reserved(self.device))
                            if self.device.type == "cuda"
                            else 0
                        ),
                        "gpu_peak_allocated_bytes": (
                            int(torch.cuda.max_memory_allocated(self.device))
                            if self.device.type == "cuda"
                            else 0
                        ),
                        "gpu_peak_reserved_bytes": (
                            int(torch.cuda.max_memory_reserved(self.device))
                            if self.device.type == "cuda"
                            else 0
                        ),
                        "gpu_utilization_percent": utilization,
                        "nvidia_smi_memory_used_mib": memory_used_mib,
                    }
                    with telemetry_path.open("a", encoding="utf-8") as stream:
                        stream.write(
                            json.dumps(telemetry, sort_keys=True, allow_nan=False)
                            + "\n"
                        )
                        stream.flush()
                    if (
                        completed in self.smoke_milestones
                        or completed % self.checkpoint_every == 0
                        or completed == self.run_until_step
                    ):
                        last_checkpoint = self._save_checkpoint(completed, metrics)
                    if completed in self.smoke_milestones:
                        self._write_smoke_performance_evidence(completed)
        except BaseException as exc:
            self._write_failure_receipt(exc)
            raise
        if last_checkpoint is None:
            raise RuntimeError("Stage-2 runner가 checkpoint를 하나도 저장하지 않았습니다")
        return last_checkpoint


__all__ = [
    "STAGE2_PRETRAIN_CHECKPOINT_SCHEMA",
    "STAGE2_PRETRAIN_DIAGNOSTIC_CHECKPOINT_SCHEMA",
    "STAGE2_PRETRAIN_RUN_SCHEMA",
    "STAGE2_PRETRAIN_RUNNER_SCHEMA",
    "STAGE2_PRETRAIN_SMOKE_ACCEPTANCE_SCHEMA",
    "STAGE2_PRETRAIN_RESUME_EQUIVALENCE_SCHEMA",
    "Stage2PublicTensorLoader",
    "Stage2ScratchPretrainRunner",
    "stage2_checkpoint_binding_payload",
]
