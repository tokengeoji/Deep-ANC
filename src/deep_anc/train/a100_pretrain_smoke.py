"""A100 canonical-pretrain resume smoke의 독립 계약과 영수증.

``canonical_pretrain``은 campaign ledger SHA를 포함한 완전 계약을 먼저 가져야
한다. 따라서 그 ledger가 요구하는 A100 중단→resume 증거를 같은 canonical run으로
만들려 하면 ``ledger → contract → smoke → ledger`` 순환이 생긴다.

이 모듈은 그 순환을 끊는다. ``a100_pretrain_smoke``는 init 자격이 없는 별도 역할이고,
output/stop budget/resume/ledger를 뺀 *학습 의미*만 ``smoke_target_sha256``으로 canonical
pretrain과 결속한다. 실제 A100 runner는 이 target 아래에서만 산출물을 만들고, campaign
validator는 사람이 적어 둔 boolean이 아니라 checkpoint/telemetry/environment bytes를 다시
열어 equality를 직접 계산한다.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .evaluation_contract import FileSnapshot, snapshot_regular_file, write_json_exclusive
from .experiment_contract import (
    _artifact_fingerprints,
    _git_source_state,
    _json_sha256,
    _normalise,
    validate_embedded_experiment_contract,
)


A100_PRETRAIN_SMOKE_ROLE = "a100_pretrain_smoke"
"""campaign ledger 이전에만 허용되는 init-ineligible A100 pretrain smoke 역할."""

SMOKE_TARGET_SCHEMA_VERSION = 1
SMOKE_PHASE_TELEMETRY_SCHEMA_VERSION = 1
SMOKE_RECEIPT_SCHEMA_VERSION = 1
SMOKE_ENVIRONMENT_RECEIPT_SCHEMA_VERSION = 1
SMOKE_RESUME_INPUT_SCHEMA_VERSION = 1
SMOKE_ROOT = Path("results/training_prerequisites/a100_pretrain_smoke")
SMOKE_RUN_LABELS = frozenset({"uninterrupted", "resumed"})

# bootstrap receipt가 freeze file을 가리키더라도, 실제 smoke interpreter가 나중에
# 바뀌었으면 exact-resume 증거가 아니다. canonical Elice 환경 정책과 같은 값을
# runner preflight 및 published environment 양쪽에서 직접 고정한다.
A100_REQUIRED_TORCH_VERSION = "2.5.1+cu121"
A100_REQUIRED_CUDA_VERSION = "12.1"

# Elice의 nominal A100 80 GB PCIe는 driver-reserved memory 때문에 PyTorch에서
# 79.4 GiB로 보고된다. 80 * 2**30을 요구하면 실제 80 GB 장치를 잘못 거부한다.
# bootstrap hardware gate와 동일하게 79 GiB 이상을 요구해 A100 40 GB/MIG slice는
# 계속 거부하면서, receipt에는 원래 device name/정확한 byte 값을 보존한다.
A100_MIN_USABLE_MEMORY_BYTES = 79 * 1024**3

# role/init/ledger/run-until/output/resume/embedded contract는 의도적으로 target에서
# 제외한다. 그 밖의 resolved training 의미는 명시적으로 모두 기록한다. ``data``와
# ``duct``는 compact P/S, manifest generation, timing contract까지 포함한다.
_TARGET_TRAINING_KEYS = (
    "stage",
    "model_config",
    "data_config",
    "duct_config",
    "model",
    "data",
    "duct",
    "loss",
    "optimizer",
    "schedule",
    "batch_size",
    "num_workers",
    "prefetch_factor",
    "seed",
    "required_world_size",
    "amp",
    "grad_clip",
    "freeze_encoder",
    "val_items",
    "eval_every",
    "log_every",
    "early_stop_patience",
    "recorded_manifest",
    "recorded_ratio",
    "require_measured_primary_path",
    "require_init_checkpoint",
    "require_recorded_manifest",
    "physics_status",
    "trusted_band_hz",
    "loss_start_sample",
    "loss_selection_sha256",
    "best_metric_key",
    "nmse_cvar_scope",
    "determinism_policy",
)
_TARGET_EXCLUDED_ARTIFACTS = frozenset(
    {"campaign_prerequisite", "second_seed_prerequisite", "init_checkpoint"}
)


def _root(value: str | Path) -> Path:
    return Path(os.path.abspath(Path(value)))


def _repo_path(root: Path, value: str | Path, *, label: str) -> Path:
    """repo 내부의 비-symlink 경로만 허용한다.

    leaf ``O_NOFOLLOW``만으로는 ``results/`` 같은 부모 directory symlink가
    repository 밖을 가리키는 경우를 막지 못한다. prerequisite proof는 target
    root 아래의 실제 filesystem object여야 하므로, lexical containment와
    resolved containment를 모두 검사하고 root 이후 어느 parent symlink도
    허용하지 않는다.
    """

    lexical_root = _root(root)
    raw = Path(value).expanduser()
    target = _root(raw if raw.is_absolute() else lexical_root / raw)
    try:
        relative = target.relative_to(lexical_root)
    except ValueError as exc:
        raise ValueError(f"{label}는 저장소 내부여야 합니다: {target}") from exc
    if lexical_root.is_symlink():
        raise ValueError(f"저장소 root는 심볼릭 링크일 수 없습니다: {lexical_root}")
    cursor = lexical_root
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise ValueError(
                f"{label} 경로에 심볼릭 링크가 있어 prerequisite root를 벗어날 수 있습니다: "
                f"{cursor}"
            )
    try:
        target.resolve(strict=False).relative_to(lexical_root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label}의 resolved path가 저장소 밖입니다: {target}") from exc
    return target


def _sha256(value: object, *, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label}가 64자리 SHA-256이 아닙니다")
    return text


def _relative_path(root: Path, path: Path, *, label: str) -> str:
    return _repo_path(root, path, label=label).relative_to(_root(root)).as_posix()


def _snapshot_ref(root: Path, path: str | Path, *, label: str) -> dict[str, str]:
    snapshot = snapshot_regular_file(_repo_path(root, path, label=label))
    return {
        "path": _relative_path(root, snapshot.path, label=label),
        "sha256": snapshot.sha256,
    }


def _snapshot_from_ref(root: Path, value: object, *, label: str) -> FileSnapshot:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} reference key 집합이 정확하지 않습니다")
    raw_path = Path(str(value["path"])).expanduser()
    target = _repo_path(
        root, raw_path if raw_path.is_absolute() else _root(root) / raw_path, label=label
    )
    snapshot = snapshot_regular_file(target)
    if snapshot.sha256 != _sha256(value["sha256"], label=f"{label}.sha256"):
        raise ValueError(f"{label} bytes SHA가 reference와 다릅니다")
    return snapshot


def _json_from_snapshot(snapshot: FileSnapshot, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(snapshot.content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} JSON이 손상됐습니다: {snapshot.path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 최상위는 mapping이어야 합니다")
    return payload


def _exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} key 집합이 정확하지 않습니다: {actual}")
    return value


def build_a100_pretrain_smoke_target(
    cfg: dict,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """ledger/output 없이 canonical pretrain과 비교 가능한 semantic projection.

    이 projection은 resolved config와 실제 artifact bytes를 함께 쓴다. 따라서 YAML
    path만 같은 P/S, source manifest, bootstrap receipt를 다른 bytes로 바꾸어도 target이
    달라진다. ``run_until_step``은 smoke의 운영 stop budget이므로 포함하지 않는다.
    """

    root = _root(repo_root)
    artifacts = {
        name: value
        for name, value in _artifact_fingerprints(cfg, root).items()
        if name not in _TARGET_EXCLUDED_ARTIFACTS
    }
    training = {key: cfg.get(key) for key in _TARGET_TRAINING_KEYS}
    payload: dict[str, Any] = {
        "schema_version": SMOKE_TARGET_SCHEMA_VERSION,
        "source": _normalise(_git_source_state(root)),
        "plant": {
            "primary_path": artifacts.get("primary_path"),
            "secondary_path": artifacts.get("secondary_path"),
            "training_timing_contract": (cfg.get("data") or {}).get(
                "training_timing_contract"
            ),
            "digital_reference_lead_samples": (cfg.get("data") or {}).get(
                "digital_reference_lead_samples"
            ),
        },
        "training": _normalise(training),
        "input_artifacts": _normalise(artifacts),
    }
    payload["sha256"] = _json_sha256(payload)
    return payload


def smoke_run_directory(
    cfg: dict,
    *,
    repo_root: str | Path,
) -> Path:
    """target/label에서만 유도되는 prerequisite-root 전용 smoke run path."""

    root = _root(repo_root)
    target = _sha256(cfg.get("smoke_target_sha256"), label="smoke_target_sha256")
    label = str(cfg.get("a100_smoke_run_label", ""))
    if label not in SMOKE_RUN_LABELS:
        raise ValueError(
            "a100_pretrain_smoke a100_smoke_run_label은 "
            f"{sorted(SMOKE_RUN_LABELS)} 중 하나여야 합니다: {label!r}"
        )
    return _repo_path(root, root / SMOKE_ROOT / target / label, label="A100 smoke run directory")


def validate_a100_pretrain_smoke_config(
    cfg: dict,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """smoke checkpoint/runner가 canonical run·ledger를 우회하지 못하게 검증한다."""

    if str(cfg.get("experiment_role", "")) != A100_PRETRAIN_SMOKE_ROLE:
        raise ValueError("A100 smoke experiment_role이 a100_pretrain_smoke가 아닙니다")
    if cfg.get("init_eligible") is not False:
        raise ValueError("a100_pretrain_smoke는 init_eligible=false여야 합니다")
    if cfg.get("contract_run_dir") is not False:
        raise ValueError("a100_pretrain_smoke는 contract_run_dir=false여야 합니다")
    if cfg.get("campaign_prerequisite") not in (None, "") or cfg.get(
        "campaign_prerequisite_sha256"
    ) not in (None, ""):
        raise ValueError("a100_pretrain_smoke는 campaign prerequisite를 소비하면 안 됩니다")
    if cfg.get("init_ckpt") not in (None, ""):
        raise ValueError("a100_pretrain_smoke는 init_ckpt를 사용할 수 없습니다")

    target = build_a100_pretrain_smoke_target(cfg, repo_root=repo_root)
    declared = _sha256(cfg.get("smoke_target_sha256"), label="smoke_target_sha256")
    if declared != target["sha256"]:
        raise ValueError(
            "a100_pretrain_smoke target digest가 resolved semantic contract와 다릅니다"
        )
    expected_dir = smoke_run_directory(cfg, repo_root=repo_root)
    raw_dir = Path(str(cfg.get("ckpt_dir", ""))).expanduser()
    actual_dir = _root(raw_dir if raw_dir.is_absolute() else _root(repo_root) / raw_dir)
    if actual_dir != expected_dir:
        raise ValueError(
            "a100_pretrain_smoke ckpt_dir는 prerequisite root의 target/label path여야 합니다: "
            f"configured={actual_dir}, expected={expected_dir}"
        )
    validate_embedded_experiment_contract(cfg)
    return target


def _hash_component(digest: "hashlib._Hash", value: Any) -> None:
    """checkpoint state의 type/shape/value를 platform-independent하게 결속한다."""

    if value is None:
        digest.update(b"N\0")
    elif isinstance(value, bool):
        digest.update(b"B1\0" if value else b"B0\0")
    elif isinstance(value, int):
        digest.update(b"I\0" + str(value).encode("ascii") + b"\0")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("smoke checkpoint component에 NaN/Inf가 있습니다")
        digest.update(b"F\0" + struct.pack("!d", value))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S\0" + len(encoded).to_bytes(8, "big") + encoded)
    elif isinstance(value, bytes):
        digest.update(b"Y\0" + len(value).to_bytes(8, "big") + value)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        if tensor.numel() and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("smoke checkpoint tensor에 NaN/Inf가 있습니다")
        header = {
            "dtype": str(tensor.dtype),
            "shape": [int(item) for item in tensor.shape],
        }
        digest.update(
            b"T\0"
            + json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\0"
        )
        if tensor.numel():
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.size and np.issubdtype(array.dtype, np.inexact) and not bool(
            np.isfinite(array).all()
        ):
            raise ValueError("smoke checkpoint ndarray에 NaN/Inf가 있습니다")
        header = {"dtype": str(array.dtype), "shape": [int(item) for item in array.shape]}
        digest.update(
            b"A\0"
            + json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\0"
            + array.tobytes()
            + b"\0"
        )
    elif isinstance(value, np.generic):
        _hash_component(digest, value.item())
    elif isinstance(value, dict):
        digest.update(b"D\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _hash_component(digest, key)
            _hash_component(digest, value[key])
        digest.update(b"\0")
    elif isinstance(value, list):
        digest.update(b"L\0")
        for item in value:
            _hash_component(digest, item)
        digest.update(b"\0")
    elif isinstance(value, tuple):
        digest.update(b"Q\0")
        for item in value:
            _hash_component(digest, item)
        digest.update(b"\0")
    else:
        raise TypeError(
            "smoke checkpoint component를 digest로 만들 수 없습니다: "
            f"{type(value).__name__}"
        )


def checkpoint_component_sha256(state: dict) -> dict[str, str]:
    """A100 exact-resume에 필요한 모든 mutable state component digest."""

    if not isinstance(state, dict):
        raise ValueError("smoke checkpoint 최상위가 mapping이 아닙니다")
    out: dict[str, str] = {}
    components: dict[str, Any] = {
        name: state.get(name)
        for name in ("model", "optimizer", "scheduler", "rng", "training_state")
    }
    # 모델 파라미터만 같아도 best checkpoint 선택 상태나 data stream index가 다르면
    # 다음 resume의 의미가 달라진다. progress는 cfg의 operational path/label은 빼되,
    # 다음 optimizer step에 영향을 주는 selection/data 진행 상태를 한 덩어리로 결속한다.
    components["progress"] = {
        "step": state.get("step"),
        "data_stream": state.get("data_stream"),
        "best_metric": state.get("best_metric"),
    }
    for name, value in components.items():
        if name != "progress" and name not in state:
            raise ValueError(f"smoke checkpoint에 {name} component가 없습니다")
        digest = hashlib.sha256()
        _hash_component(digest, value)
        out[name] = digest.hexdigest()
    return out


def _checkpoint_state_from_snapshot(snapshot: FileSnapshot, *, label: str) -> dict:
    try:
        state = torch.load(
            io.BytesIO(snapshot.content), map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} checkpoint를 읽을 수 없습니다") from exc
    if not isinstance(state, dict) or not isinstance(state.get("cfg"), dict):
        raise ValueError(f"{label} checkpoint에 resolved cfg가 없습니다")
    return state


def _validate_smoke_checkpoint(
    snapshot: FileSnapshot,
    *,
    label: str,
    repo_root: Path,
    expected_target: str,
    require_component_digest: bool = True,
) -> tuple[dict, dict[str, str]]:
    state = _checkpoint_state_from_snapshot(snapshot, label=label)
    cfg = state["cfg"]
    validate_a100_pretrain_smoke_config(cfg, repo_root=repo_root)
    if str(cfg.get("smoke_target_sha256")) != expected_target:
        raise ValueError(f"{label} checkpoint smoke target이 receipt target과 다릅니다")
    stream = state.get("data_stream")
    if not isinstance(stream, dict) or int(stream.get("global_batch_index", -1)) != int(
        state.get("step", -2)
    ):
        raise ValueError(f"{label} checkpoint step/data_stream이 불일치합니다")
    try:
        best_metric = float(state.get("best_metric"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} checkpoint best_metric이 없습니다") from exc
    is_unselected_stop = (
        label == "stopped"
        and snapshot.path.name == "stop.pt"
        and best_metric == float("inf")
        and int(state.get("step", -1)) == int(cfg.get("run_until_step", -2))
        and int(state.get("step", -1)) < int(cfg.get("eval_every", -1))
    )
    if not math.isfinite(best_metric) and not is_unselected_stop:
        raise ValueError(f"{label} checkpoint best_metric이 finite가 아닙니다")
    rng = state.get("rng")
    cuda_rng = rng.get("cuda") if isinstance(rng, dict) else None
    if (
        not isinstance(cuda_rng, list)
        or len(cuda_rng) != 1
        or not isinstance(cuda_rng[0], torch.Tensor)
        or cuda_rng[0].dtype != torch.uint8
    ):
        raise ValueError(
            f"{label} checkpoint에 A100 world1 CUDA RNG state가 정확히 하나 필요합니다"
        )
    # first eval 전 stop.pt의 +inf selection sentinel은 final equality 대상이 아니다.
    # non-finite state를 일반 component digest에 허용하면 canonical resume 검증까지
    # 느슨해지므로, 그 단 하나의 intermediate artifact만 digest 생성을 생략한다.
    return state, (
        checkpoint_component_sha256(state) if require_component_digest else {}
    )


def _phase_telemetry_name(start_step: int, completed_step: int) -> str:
    return f"{int(start_step):06d}_{int(completed_step):06d}.json"


def write_a100_pretrain_smoke_phase_telemetry(
    run_dir: str | Path,
    cfg: dict,
    *,
    start_step: int,
    completed_step: int,
    elapsed_seconds: float,
    device: str,
    cuda_available: bool,
    device_count: int,
    max_memory_allocated_bytes: int,
    max_memory_reserved_bytes: int,
    resume_checkpoint_sha256: str | None,
    final_train_metrics: dict[str, float],
) -> Path:
    """Trainer 종료 지점에서 phase별 immutable A100 telemetry를 발행한다."""

    target = validate_a100_pretrain_smoke_config(
        cfg, repo_root=Path(__file__).resolve().parents[3]
    )
    start = int(start_step)
    completed = int(completed_step)
    elapsed = float(elapsed_seconds)
    if start < 0 or completed <= start or not math.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError("A100 smoke telemetry step/elapsed 값이 유효하지 않습니다")
    metrics: dict[str, float] = {}
    for key, value in sorted(final_train_metrics.items()):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"A100 smoke telemetry metric {key}가 finite가 아닙니다")
        metrics[str(key)] = scalar
    payload = {
        "schema_version": SMOKE_PHASE_TELEMETRY_SCHEMA_VERSION,
        "smoke_target_sha256": target["sha256"],
        "experiment_role": A100_PRETRAIN_SMOKE_ROLE,
        "init_eligible": False,
        "experiment_contract_sha256": str(cfg["experiment_contract_sha256"]),
        "start_step": start,
        "completed_step": completed,
        "run_until_step": int(cfg.get("run_until_step", cfg["schedule"]["total_steps"])),
        "schedule_total_steps": int(cfg["schedule"]["total_steps"]),
        "elapsed_seconds": elapsed,
        "steps_completed": completed - start,
        "steps_per_second": float((completed - start) / elapsed),
        "device": str(device),
        "cuda_available": bool(cuda_available),
        "device_count": int(device_count),
        "amp": str(cfg.get("amp")),
        "max_memory_allocated_bytes": int(max_memory_allocated_bytes),
        "max_memory_reserved_bytes": int(max_memory_reserved_bytes),
        "resume_checkpoint_sha256": (
            None
            if resume_checkpoint_sha256 is None
            else _sha256(
                resume_checkpoint_sha256,
                label="A100 smoke resumed checkpoint SHA",
            )
        ),
        "final_train_metrics": metrics,
    }
    directory = Path(run_dir) / "telemetry"
    path = directory / _phase_telemetry_name(start, completed)
    write_json_exclusive(path, payload)
    return path


def _validate_phase_telemetry(
    snapshot: FileSnapshot,
    *,
    label: str,
    target: str,
    expected_start: int,
    expected_completed: int,
    expected_contract_sha256: str,
    expected_run_until_step: int,
    expected_schedule_total_steps: int,
    expected_resume_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    payload = _exact_keys(
        _json_from_snapshot(snapshot, label=label),
        {
            "schema_version",
            "smoke_target_sha256",
            "experiment_role",
            "init_eligible",
            "experiment_contract_sha256",
            "start_step",
            "completed_step",
            "run_until_step",
            "schedule_total_steps",
            "elapsed_seconds",
            "steps_completed",
            "steps_per_second",
            "device",
            "cuda_available",
            "device_count",
            "amp",
            "max_memory_allocated_bytes",
            "max_memory_reserved_bytes",
            "resume_checkpoint_sha256",
            "final_train_metrics",
        },
        label=label,
    )
    if (
        payload["schema_version"] != SMOKE_PHASE_TELEMETRY_SCHEMA_VERSION
        or payload["smoke_target_sha256"] != target
        or payload["experiment_role"] != A100_PRETRAIN_SMOKE_ROLE
        or payload["init_eligible"] is not False
        or payload["experiment_contract_sha256"] != expected_contract_sha256
        or int(payload["start_step"]) != int(expected_start)
        or int(payload["completed_step"]) != int(expected_completed)
        or int(payload["run_until_step"]) != int(expected_run_until_step)
        or int(payload["schedule_total_steps"]) != int(expected_schedule_total_steps)
        or int(payload["steps_completed"]) != int(expected_completed - expected_start)
        or payload["amp"] != "bf16"
        or payload["cuda_available"] is not True
        or payload["device"] != "cuda:0"
        or int(payload["device_count"]) != 1
        or payload["resume_checkpoint_sha256"] != expected_resume_checkpoint_sha256
    ):
        raise ValueError(f"{label} target/role/contract/step/amp 계약이 다릅니다")
    for key in ("elapsed_seconds", "steps_per_second"):
        if not math.isfinite(float(payload[key])) or float(payload[key]) <= 0.0:
            raise ValueError(f"{label}.{key}가 양의 finite 값이 아닙니다")
    if not isinstance(payload["final_train_metrics"], dict) or not payload[
        "final_train_metrics"
    ]:
        raise ValueError(f"{label}.final_train_metrics가 mapping이 아닙니다")
    for value in payload["final_train_metrics"].values():
        if not math.isfinite(float(value)):
            raise ValueError(f"{label}.final_train_metrics에 non-finite 값이 있습니다")
    allocated = int(payload["max_memory_allocated_bytes"])
    reserved = int(payload["max_memory_reserved_bytes"])
    if allocated <= 0 or reserved < allocated:
        raise ValueError(f"{label} CUDA peak memory telemetry가 유효하지 않습니다")
    return payload


def _validate_environment(snapshot: FileSnapshot) -> dict[str, Any]:
    environment = _json_from_snapshot(snapshot, label="A100 environment receipt")
    devices = environment.get("devices") if isinstance(environment, dict) else None
    if (
        not isinstance(environment, dict)
        or environment.get("cuda_available") is not True
        or environment.get("device_count") != 1
        or not isinstance(devices, list)
        or len(devices) != 1
        or "A100" not in str(devices[0].get("name", ""))
        or int(devices[0].get("total_memory_bytes", 0))
        < A100_MIN_USABLE_MEMORY_BYTES
        or environment.get("deterministic_algorithms") is not True
        or environment.get("cudnn_benchmark") is not False
        or environment.get("cudnn_deterministic") is not True
        or environment.get("cublas_workspace_config") not in {":4096:8", ":16:8"}
        or environment.get("torch") != A100_REQUIRED_TORCH_VERSION
        or environment.get("torch_cuda") != A100_REQUIRED_CUDA_VERSION
    ):
        raise ValueError(
            "A100 environment receipt가 world1/CUDA/결정론 backend 계약을 증명하지 못합니다"
        )
    return environment


def build_a100_pretrain_smoke_environment_receipt(
    *,
    repo_root: str | Path,
    smoke_target_sha256: str,
    uninterrupted_environment: str | Path,
    resumed_environment: str | Path,
) -> dict[str, Any]:
    """두 arm이 같은 A100 결정론 환경에서 실행됐음을 묶는 immutable payload."""

    root = _root(repo_root)
    target = _sha256(smoke_target_sha256, label="smoke_target_sha256")
    return {
        "schema_version": SMOKE_ENVIRONMENT_RECEIPT_SCHEMA_VERSION,
        "smoke_target_sha256": target,
        "experiment_role": A100_PRETRAIN_SMOKE_ROLE,
        "init_eligible": False,
        "uninterrupted_environment": _snapshot_ref(
            root, uninterrupted_environment, label="uninterrupted environment"
        ),
        "resumed_environment": _snapshot_ref(
            root, resumed_environment, label="resumed environment"
        ),
    }


def _validate_environment_receipt(
    snapshot: FileSnapshot, *, root: Path, target: str
) -> dict[str, Any]:
    payload = _exact_keys(
        _json_from_snapshot(snapshot, label="A100 smoke environment receipt"),
        {
            "schema_version",
            "smoke_target_sha256",
            "experiment_role",
            "init_eligible",
            "uninterrupted_environment",
            "resumed_environment",
        },
        label="A100 smoke environment receipt",
    )
    if (
        payload["schema_version"] != SMOKE_ENVIRONMENT_RECEIPT_SCHEMA_VERSION
        or payload["smoke_target_sha256"] != target
        or payload["experiment_role"] != A100_PRETRAIN_SMOKE_ROLE
        or payload["init_eligible"] is not False
    ):
        raise ValueError("A100 smoke environment receipt target/role/init이 다릅니다")
    uninterrupted = _validate_environment(
        _snapshot_from_ref(
            root,
            payload["uninterrupted_environment"],
            label="A100 smoke uninterrupted environment",
        )
    )
    resumed = _validate_environment(
        _snapshot_from_ref(
            root,
            payload["resumed_environment"],
            label="A100 smoke resumed environment",
        )
    )
    if uninterrupted != resumed:
        raise ValueError("A100 smoke 두 arm의 environment receipt가 정확히 같지 않습니다")
    return payload


def build_a100_pretrain_smoke_resume_input(
    *,
    repo_root: str | Path,
    smoke_target_sha256: str,
    resume_checkpoint: str | Path,
    stop_checkpoint: str | Path,
) -> dict[str, Any]:
    """실제 ``--resume stop.pt`` 입력과 immutable stop checkpoint를 결속한다."""

    root = _root(repo_root)
    target = _sha256(smoke_target_sha256, label="smoke_target_sha256")
    resume_snapshot = snapshot_regular_file(resume_checkpoint)
    stop_reference = _snapshot_ref(root, stop_checkpoint, label="smoke stop checkpoint")
    if resume_snapshot.sha256 != stop_reference["sha256"]:
        raise ValueError("resume 입력 bytes가 immutable stop.pt와 다릅니다")
    return {
        "schema_version": SMOKE_RESUME_INPUT_SCHEMA_VERSION,
        "smoke_target_sha256": target,
        "experiment_role": A100_PRETRAIN_SMOKE_ROLE,
        "init_eligible": False,
        "resume_path": _relative_path(
            root, resume_snapshot.path, label="smoke resume checkpoint"
        ),
        "resume_sha256": resume_snapshot.sha256,
        "stop_checkpoint": stop_reference,
    }


def _validate_resume_input(
    snapshot: FileSnapshot,
    *,
    root: Path,
    target: str,
    expected_resume_path: Path,
    expected_stop_reference: dict[str, str],
) -> dict[str, Any]:
    payload = _exact_keys(
        _json_from_snapshot(snapshot, label="A100 smoke resume input"),
        {
            "schema_version",
            "smoke_target_sha256",
            "experiment_role",
            "init_eligible",
            "resume_path",
            "resume_sha256",
            "stop_checkpoint",
        },
        label="A100 smoke resume input",
    )
    if (
        payload["schema_version"] != SMOKE_RESUME_INPUT_SCHEMA_VERSION
        or payload["smoke_target_sha256"] != target
        or payload["experiment_role"] != A100_PRETRAIN_SMOKE_ROLE
        or payload["init_eligible"] is not False
        or _resolve_cfg_path(root, payload["resume_path"]) != expected_resume_path
        or payload["stop_checkpoint"] != expected_stop_reference
    ):
        raise ValueError("A100 smoke resume input target/role/path/stop binding이 다릅니다")
    stop = _snapshot_from_ref(root, payload["stop_checkpoint"], label="A100 smoke stop input")
    if _sha256(payload["resume_sha256"], label="A100 smoke resume input SHA") != stop.sha256:
        raise ValueError("A100 smoke resume input bytes가 immutable stop.pt와 다릅니다")
    return payload


def _resolve_cfg_path(root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return _repo_path(
        root, path if path.is_absolute() else _root(root) / path, label="A100 smoke config path"
    )


def _batch_sequence_sha256(*, target: str, batch_size: int, final_step: int) -> str:
    """global batch index가 만드는 full-run sample sequence의 immutable identity."""

    return _json_sha256(
        {
            "schema_version": 1,
            "smoke_target_sha256": target,
            "start_global_batch_index": 0,
            "exclusive_end_global_batch_index": int(final_step),
            "batch_size": int(batch_size),
        }
    )


def build_a100_pretrain_smoke_artifacts(
    *,
    repo_root: str | Path,
    smoke_target_sha256: str,
    uninterrupted_checkpoint: str | Path,
    stop_checkpoint: str | Path,
    resumed_checkpoint: str | Path,
    uninterrupted_telemetry: str | Path,
    stopped_telemetry: str | Path,
    resumed_telemetry: str | Path,
    environment_receipt: str | Path,
    resume_input: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """runner가 no-replace write할 combined telemetry와 receipt payload를 만든다.

    반환값은 ``(telemetry, receipt)`` 이며, receipt에는 telemetry file reference를
    추가해야 하므로 caller가 telemetry를 먼저 publish한 뒤
    :func:`finalize_a100_pretrain_smoke_receipt`를 호출한다.
    """

    root = _root(repo_root)
    target = _sha256(smoke_target_sha256, label="smoke_target_sha256")
    refs = {
        "uninterrupted_checkpoint": _snapshot_ref(
            root, uninterrupted_checkpoint, label="uninterrupted checkpoint"
        ),
        "stop_checkpoint": _snapshot_ref(root, stop_checkpoint, label="stop checkpoint"),
        "resumed_checkpoint": _snapshot_ref(
            root, resumed_checkpoint, label="resumed checkpoint"
        ),
        "uninterrupted_telemetry": _snapshot_ref(
            root, uninterrupted_telemetry, label="uninterrupted telemetry"
        ),
        "stopped_telemetry": _snapshot_ref(root, stopped_telemetry, label="stopped telemetry"),
        "resumed_telemetry": _snapshot_ref(root, resumed_telemetry, label="resumed telemetry"),
        "environment_receipt": _snapshot_ref(root, environment_receipt, label="environment receipt"),
        "resume_input": _snapshot_ref(root, resume_input, label="resume input"),
    }
    return _build_artifacts_from_refs(root, target, refs)


def _build_artifacts_from_refs(
    root: Path, target: str, refs: dict[str, dict[str, str]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {
        "uninterrupted_checkpoint",
        "stop_checkpoint",
        "resumed_checkpoint",
        "uninterrupted_telemetry",
        "stopped_telemetry",
        "resumed_telemetry",
        "environment_receipt",
        "resume_input",
    }
    if set(refs) != required:
        raise ValueError("A100 smoke artifact reference 집합이 정확하지 않습니다")
    snapshots = {
        name: _snapshot_from_ref(root, value, label=f"A100 smoke {name}")
        for name, value in refs.items()
    }
    full, full_digest = _validate_smoke_checkpoint(
        snapshots["uninterrupted_checkpoint"],
        label="uninterrupted",
        repo_root=root,
        expected_target=target,
    )
    stopped, _ = _validate_smoke_checkpoint(
        snapshots["stop_checkpoint"],
        label="stopped",
        repo_root=root,
        expected_target=target,
        require_component_digest=False,
    )
    resumed, resumed_digest = _validate_smoke_checkpoint(
        snapshots["resumed_checkpoint"],
        label="resumed",
        repo_root=root,
        expected_target=target,
    )
    full_step = int(full["step"])
    stop_step = int(stopped["step"])
    resumed_step = int(resumed["step"])
    if not 200 <= stop_step <= 500 or resumed_step <= stop_step or resumed_step > 500:
        raise ValueError("A100 smoke stop/resumed step은 200–500 범위의 정식 smoke여야 합니다")
    if full_step != resumed_step:
        raise ValueError("uninterrupted/resumed checkpoint 최종 step이 다릅니다")
    full_cfg = full["cfg"]
    stopped_cfg = stopped["cfg"]
    resumed_cfg = resumed["cfg"]
    if full_cfg.get("a100_smoke_run_label") != "uninterrupted":
        raise ValueError("uninterrupted checkpoint label이 잘못됐습니다")
    if stopped_cfg.get("a100_smoke_run_label") != "resumed" or resumed_cfg.get(
        "a100_smoke_run_label"
    ) != "resumed":
        raise ValueError("resumed checkpoint label이 잘못됐습니다")
    if full_cfg.get("resume") not in (None, "") or stopped_cfg.get("resume") not in (
        None,
        "",
    ):
        raise ValueError("A100 smoke 최초 구간은 resume 없이 시작해야 합니다")
    full_dir = smoke_run_directory(full_cfg, repo_root=root)
    resumed_dir = smoke_run_directory(resumed_cfg, repo_root=root)
    target_root = (root / SMOKE_ROOT / target).absolute()
    expected_resume = resumed_dir / "ckpt" / "stop.pt"
    expected_paths = {
        "uninterrupted_checkpoint": full_dir / "ckpt" / "last.pt",
        "stop_checkpoint": resumed_dir / "ckpt" / "stop.pt",
        "resumed_checkpoint": resumed_dir / "ckpt" / "last.pt",
        "uninterrupted_telemetry": full_dir
        / "telemetry"
        / _phase_telemetry_name(0, full_step),
        "stopped_telemetry": resumed_dir
        / "telemetry"
        / _phase_telemetry_name(0, stop_step),
        "resumed_telemetry": resumed_dir
        / "telemetry"
        / _phase_telemetry_name(stop_step, resumed_step),
        "environment_receipt": target_root / "environment_receipt.json",
        "resume_input": target_root / "resume_input.json",
    }
    for name, expected_path in expected_paths.items():
        if snapshots[name].path != expected_path:
            raise ValueError(
                f"A100 smoke {name}는 target prerequisite root의 고정 path여야 합니다: "
                f"actual={snapshots[name].path}, expected={expected_path}"
            )
    # smoke만 immutable sibling stop.pt를 actual --resume input으로 허용한다.
    # canonical role의 resume은 여전히 exact last.pt만 허용된다.
    if _resolve_cfg_path(root, resumed_cfg.get("resume")) != expected_resume:
        raise ValueError("resumed checkpoint cfg.resume이 같은 smoke run의 stop.pt가 아닙니다")
    _validate_resume_input(
        snapshots["resume_input"],
        root=root,
        target=target,
        expected_resume_path=expected_resume,
        expected_stop_reference=refs["stop_checkpoint"],
    )
    for key in ("model", "optimizer", "scheduler", "rng", "training_state", "progress"):
        if full_digest[key] != resumed_digest[key]:
            raise ValueError(f"A100 smoke uninterrupted/resumed {key} state가 다릅니다")
    experiment_contract_sha = str(full_cfg.get("experiment_contract_sha256") or "")
    if (
        not experiment_contract_sha
        or stopped_cfg.get("experiment_contract_sha256") != experiment_contract_sha
        or resumed_cfg.get("experiment_contract_sha256") != experiment_contract_sha
    ):
        raise ValueError("A100 smoke phase들의 experiment contract가 다릅니다")
    if int(full_cfg.get("batch_size", -1)) != 96 or int(full_cfg.get("num_workers", -1)) != 14 or int(
        full_cfg.get("prefetch_factor", -1)
    ) != 4:
        raise ValueError("A100 smoke B96/worker14/prefetch4 계약이 다릅니다")
    if full_cfg.get("amp") != "bf16" or int(full_cfg.get("required_world_size", -1)) != 1:
        raise ValueError("A100 smoke amp/world-size 계약이 다릅니다")
    environments = _validate_environment_receipt(
        snapshots["environment_receipt"], root=root, target=target
    )
    if _resolve_cfg_path(root, environments["uninterrupted_environment"]["path"]) != (
        full_dir / "environment.json"
    ) or _resolve_cfg_path(root, environments["resumed_environment"]["path"]) != (
        resumed_dir / "environment.json"
    ):
        raise ValueError("A100 smoke environment receipt가 각 arm의 environment.json을 가리키지 않습니다")
    _validate_phase_telemetry(
        snapshots["uninterrupted_telemetry"],
        label="uninterrupted telemetry",
        target=target,
        expected_start=0,
        expected_completed=resumed_step,
        expected_contract_sha256=experiment_contract_sha,
        expected_run_until_step=full_step,
        expected_schedule_total_steps=int(full_cfg["schedule"]["total_steps"]),
        expected_resume_checkpoint_sha256=None,
    )
    _validate_phase_telemetry(
        snapshots["stopped_telemetry"],
        label="stopped telemetry",
        target=target,
        expected_start=0,
        expected_completed=stop_step,
        expected_contract_sha256=experiment_contract_sha,
        expected_run_until_step=stop_step,
        expected_schedule_total_steps=int(stopped_cfg["schedule"]["total_steps"]),
        expected_resume_checkpoint_sha256=None,
    )
    _validate_phase_telemetry(
        snapshots["resumed_telemetry"],
        label="resumed telemetry",
        target=target,
        expected_start=stop_step,
        expected_completed=resumed_step,
        expected_contract_sha256=experiment_contract_sha,
        expected_run_until_step=resumed_step,
        expected_schedule_total_steps=int(resumed_cfg["schedule"]["total_steps"]),
        expected_resume_checkpoint_sha256=refs["stop_checkpoint"]["sha256"],
    )
    sequence_sha = _batch_sequence_sha256(
        target=target,
        batch_size=int(full_cfg["batch_size"]),
        final_step=resumed_step,
    )
    telemetry = {
        "schema_version": SMOKE_PHASE_TELEMETRY_SCHEMA_VERSION,
        "smoke_target_sha256": target,
        "experiment_role": A100_PRETRAIN_SMOKE_ROLE,
        "init_eligible": False,
        "stop_step": stop_step,
        "resumed_step": resumed_step,
        "batch_sequence_sha256": sequence_sha,
        "uninterrupted_checkpoint": refs["uninterrupted_checkpoint"],
        "stop_checkpoint": refs["stop_checkpoint"],
        "resumed_checkpoint": refs["resumed_checkpoint"],
        "uninterrupted_phase": refs["uninterrupted_telemetry"],
        "stopped_phase": refs["stopped_telemetry"],
        "resumed_phase": refs["resumed_telemetry"],
    }
    receipt = {
        "schema_version": SMOKE_RECEIPT_SCHEMA_VERSION,
        "smoke_target_sha256": target,
        "experiment_role": A100_PRETRAIN_SMOKE_ROLE,
        "init_eligible": False,
        "stop_step": stop_step,
        "resumed_step": resumed_step,
        "environment_receipt": refs["environment_receipt"],
        "resume_input": refs["resume_input"],
        "uninterrupted_checkpoint": refs["uninterrupted_checkpoint"],
        "stop_checkpoint": refs["stop_checkpoint"],
        "resumed_checkpoint": refs["resumed_checkpoint"],
        "component_sha256": full_digest,
        "batch_sequence_sha256": sequence_sha,
    }
    return telemetry, receipt


def finalize_a100_pretrain_smoke_receipt(
    receipt: dict[str, Any], *, telemetry_reference: dict[str, str]
) -> dict[str, Any]:
    """combined telemetry를 먼저 no-replace 발행한 뒤 receipt에 그 bytes를 결속한다."""

    expected = {
        "schema_version",
        "smoke_target_sha256",
        "experiment_role",
        "init_eligible",
        "stop_step",
        "resumed_step",
        "environment_receipt",
        "resume_input",
        "uninterrupted_checkpoint",
        "stop_checkpoint",
        "resumed_checkpoint",
        "component_sha256",
        "batch_sequence_sha256",
    }
    _exact_keys(receipt, expected, label="A100 smoke receipt payload")
    out = dict(receipt)
    if not isinstance(telemetry_reference, dict) or set(telemetry_reference) != {
        "path",
        "sha256",
    }:
        raise ValueError("A100 smoke telemetry reference key 집합이 정확하지 않습니다")
    out["telemetry"] = dict(telemetry_reference)
    return out


def validate_a100_pretrain_smoke_receipt(
    receipt: dict[str, Any],
    *,
    repo_root: str | Path,
    expected_smoke_target_sha256: str,
) -> dict[str, Any]:
    """campaign ledger가 A100 proof를 직접 재계산하는 fail-closed validator."""

    root = _root(repo_root)
    target = _sha256(expected_smoke_target_sha256, label="expected smoke target")
    payload = _exact_keys(
        receipt,
        {
            "schema_version",
            "smoke_target_sha256",
            "experiment_role",
            "init_eligible",
            "stop_step",
            "resumed_step",
            "environment_receipt",
            "resume_input",
            "uninterrupted_checkpoint",
            "stop_checkpoint",
            "resumed_checkpoint",
            "component_sha256",
            "batch_sequence_sha256",
            "telemetry",
        },
        label="A100 smoke receipt",
    )
    if (
        payload["schema_version"] != SMOKE_RECEIPT_SCHEMA_VERSION
        or payload["smoke_target_sha256"] != target
        or payload["experiment_role"] != A100_PRETRAIN_SMOKE_ROLE
        or payload["init_eligible"] is not False
    ):
        raise ValueError("A100 smoke receipt target/role/init 계약이 다릅니다")
    refs = {
        "uninterrupted_checkpoint": payload["uninterrupted_checkpoint"],
        "stop_checkpoint": payload["stop_checkpoint"],
        "resumed_checkpoint": payload["resumed_checkpoint"],
        "uninterrupted_telemetry": None,
        "stopped_telemetry": None,
        "resumed_telemetry": None,
        "environment_receipt": payload["environment_receipt"],
        "resume_input": payload["resume_input"],
    }
    telemetry_snapshot = _snapshot_from_ref(root, payload["telemetry"], label="A100 smoke telemetry")
    expected_target_root = (root / SMOKE_ROOT / target).absolute()
    if telemetry_snapshot.path != expected_target_root / "telemetry.json":
        raise ValueError("A100 smoke combined telemetry는 target prerequisite root에 있어야 합니다")
    telemetry = _exact_keys(
        _json_from_snapshot(telemetry_snapshot, label="A100 smoke telemetry"),
        {
            "schema_version",
            "smoke_target_sha256",
            "experiment_role",
            "init_eligible",
            "stop_step",
            "resumed_step",
            "batch_sequence_sha256",
            "uninterrupted_checkpoint",
            "stop_checkpoint",
            "resumed_checkpoint",
            "uninterrupted_phase",
            "stopped_phase",
            "resumed_phase",
        },
        label="A100 smoke telemetry",
    )
    if (
        telemetry["schema_version"] != SMOKE_PHASE_TELEMETRY_SCHEMA_VERSION
        or telemetry["smoke_target_sha256"] != target
        or telemetry["experiment_role"] != A100_PRETRAIN_SMOKE_ROLE
        or telemetry["init_eligible"] is not False
        or int(telemetry["stop_step"]) != int(payload["stop_step"])
        or int(telemetry["resumed_step"]) != int(payload["resumed_step"])
        or telemetry["batch_sequence_sha256"] != payload["batch_sequence_sha256"]
    ):
        raise ValueError("A100 smoke telemetry target/role/step binding이 다릅니다")
    for name in ("uninterrupted_checkpoint", "stop_checkpoint", "resumed_checkpoint"):
        if telemetry[name] != payload[name]:
            raise ValueError(f"A100 smoke telemetry {name} reference가 receipt와 다릅니다")
    refs["uninterrupted_telemetry"] = telemetry["uninterrupted_phase"]
    refs["stopped_telemetry"] = telemetry["stopped_phase"]
    refs["resumed_telemetry"] = telemetry["resumed_phase"]
    rebuilt_telemetry, rebuilt_receipt = _build_artifacts_from_refs(root, target, refs)
    # ``_build_artifacts_from_refs``가 fresh snapshots에서 state equality와 phase
    # telemetry를 계산한다. receipt의 사람이 쓴 digest/step도 그 결과와 동일해야 한다.
    for key, value in rebuilt_receipt.items():
        if payload.get(key) != value:
            raise ValueError(f"A100 smoke receipt {key}가 실제 artifact와 다릅니다")
    if telemetry != rebuilt_telemetry:
        raise ValueError("A100 smoke combined telemetry가 실제 artifact와 다릅니다")
    return payload


__all__ = [
    "A100_PRETRAIN_SMOKE_ROLE",
    "SMOKE_ROOT",
    "SMOKE_RUN_LABELS",
    "SMOKE_TARGET_SCHEMA_VERSION",
    "SMOKE_PHASE_TELEMETRY_SCHEMA_VERSION",
    "SMOKE_RECEIPT_SCHEMA_VERSION",
    "SMOKE_ENVIRONMENT_RECEIPT_SCHEMA_VERSION",
    "SMOKE_RESUME_INPUT_SCHEMA_VERSION",
    "A100_REQUIRED_TORCH_VERSION",
    "A100_REQUIRED_CUDA_VERSION",
    "build_a100_pretrain_smoke_target",
    "smoke_run_directory",
    "validate_a100_pretrain_smoke_config",
    "checkpoint_component_sha256",
    "write_a100_pretrain_smoke_phase_telemetry",
    "build_a100_pretrain_smoke_environment_receipt",
    "build_a100_pretrain_smoke_resume_input",
    "build_a100_pretrain_smoke_artifacts",
    "finalize_a100_pretrain_smoke_receipt",
    "validate_a100_pretrain_smoke_receipt",
]
