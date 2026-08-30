"""공식 학습이 immutable schedule을 끝냈다는 no-replace 영수증."""

from __future__ import annotations

import io
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from ..config import validate_canonical_training_policy
from .experiment_contract import (
    CANONICAL_ROLES,
    require_canonical_source_trust,
    validate_embedded_experiment_contract,
)
from .evaluation_contract import snapshot_regular_file
from .reproducibility import REPRODUCIBILITY_FILES


SCHEMA_VERSION = 1
RECEIPT_NAME = "completion.json"


def sha256_file(path: str | Path) -> str:
    return snapshot_regular_file(path).sha256


def _snapshot_checkpoint(path: Path) -> tuple[dict, str]:
    snapshot = snapshot_regular_file(path)
    state = torch.load(
        io.BytesIO(snapshot.content), map_location="cpu", weights_only=False
    )
    if not isinstance(state, dict) or not isinstance(state.get("cfg"), dict):
        raise ValueError(f"completion checkpoint에 resolved cfg가 없습니다: {path}")
    return state, snapshot.sha256


def _model_state_receipt(state: dict, *, label: str) -> dict[str, Any]:
    model = state.get("model")
    if not isinstance(model, dict) or not model:
        raise ValueError(f"{label} checkpoint model state가 없습니다")
    digest = hashlib.sha256()
    structure: list[dict[str, Any]] = []
    for name in sorted(model):
        value = model[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{label} model.{name}이 tensor가 아닙니다")
        tensor = value.detach().cpu().contiguous()
        if tensor.numel() and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{label} model.{name}에 NaN/Inf가 있습니다")
        item = {
            "name": str(name),
            "shape": [int(size) for size in tensor.shape],
            "dtype": str(tensor.dtype),
        }
        structure.append(item)
        digest.update(
            json.dumps(item, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\0")
        if tensor.numel():
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return {"structure": structure, "sha256": digest.hexdigest()}


def _expected_payload(
    ckpt_dir: Path,
    *,
    last_checkpoint: tuple[dict, str] | None = None,
    best_checkpoint: tuple[dict, str] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"
    if not last_path.is_file() or not best_path.is_file():
        raise FileNotFoundError("completion receipt에는 best.pt와 last.pt가 모두 필요합니다")
    last, last_sha256 = last_checkpoint or _snapshot_checkpoint(last_path)
    best, best_sha256 = best_checkpoint or _snapshot_checkpoint(best_path)
    last_cfg = last["cfg"]
    best_cfg = best["cfg"]
    last_contract = validate_embedded_experiment_contract(last_cfg)
    best_contract = validate_embedded_experiment_contract(best_cfg)
    if last_contract["sha256"] != best_contract["sha256"]:
        raise ValueError("best.pt와 last.pt experiment contract가 다릅니다")
    role = str(last_cfg.get("experiment_role", ""))
    if role not in CANONICAL_ROLES:
        raise ValueError(f"completion receipt는 canonical role에만 발급합니다: {role!r}")
    validate_canonical_training_policy(last_cfg)
    validate_canonical_training_policy(best_cfg)
    require_canonical_source_trust(last_cfg, repo_root=repo_root)
    if str(best_cfg.get("experiment_role", "")) != role:
        raise ValueError("best.pt와 last.pt experiment_role이 다릅니다")
    total = int((last_cfg.get("schedule") or {}).get("total_steps", 0))
    step = int(last.get("step", -1))
    if total <= 0 or step != total:
        raise ValueError(
            f"immutable schedule을 완료하지 않았습니다: step={step}, total_steps={total}"
        )
    best_step = int(best.get("step", -1))
    eval_every = int(last_cfg.get("eval_every", 0))
    if (
        best_step <= 0
        or best_step > total
        or eval_every <= 0
        or best_step % eval_every != 0
    ):
        raise ValueError(
            "best.pt step은 완료 run의 실제 eval cadence에 있어야 합니다: "
            f"best_step={best_step}, eval_every={eval_every}, total={total}"
        )
    best_metric = float(best.get("best_metric", float("nan")))
    last_best_metric = float(last.get("best_metric", float("nan")))
    if not math.isfinite(best_metric) or not math.isfinite(last_best_metric):
        raise ValueError("best.pt/last.pt best_metric이 finite가 아닙니다")
    if best_metric != last_best_metric:
        raise ValueError(
            "best.pt 선택 metric과 last.pt 최종 best_metric이 다릅니다: "
            f"best={best_metric!r}, last={last_best_metric!r}"
        )
    best_metric_key = str(best_cfg.get("best_metric_key", ""))
    if not best_metric_key or best_metric_key != str(
        last_cfg.get("best_metric_key", "")
    ):
        raise ValueError("best.pt/last.pt best_metric_key가 없거나 다릅니다")
    best_model = _model_state_receipt(best, label="best.pt")
    last_model = _model_state_receipt(last, label="last.pt")
    if best_model["structure"] != last_model["structure"]:
        raise ValueError("best.pt와 last.pt model state 구조가 다릅니다")
    reproducibility: dict[str, str] = {}
    snapshots = {
        name: snapshot_regular_file(ckpt_dir.parent / name)
        for name in REPRODUCIBILITY_FILES
    }
    try:
        snapshot_cfg = yaml.safe_load(
            snapshots["config_snapshot.yaml"].content.decode("utf-8")
        )
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("canonical config_snapshot.yaml이 손상됐습니다") from exc
    if snapshot_cfg != last_cfg:
        raise ValueError("canonical config_snapshot.yaml이 last.pt resolved cfg와 다릅니다")
    git_rev = snapshots["git_rev.txt"].content.decode("utf-8").strip()
    expected_commit = str((last_contract.get("source") or {}).get("git_commit") or "")
    if not expected_commit or git_rev != expected_commit:
        raise ValueError("canonical git_rev.txt가 experiment contract commit과 다릅니다")
    if not snapshots["pip_freeze.txt"].content.strip():
        raise ValueError("canonical pip_freeze.txt가 비었습니다")
    try:
        environment = json.loads(snapshots["environment.json"].content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("canonical environment.json이 손상됐습니다") from exc
    if not isinstance(environment, dict) or not {
        "python",
        "torch",
        "cuda_available",
        "device_count",
        "devices",
        "deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cublas_workspace_config",
    }.issubset(environment):
        raise ValueError("canonical environment.json 필수 필드가 없습니다")
    if (
        environment["deterministic_algorithms"] is not True
        or environment["cudnn_benchmark"] is not False
        or environment["cudnn_deterministic"] is not True
    ):
        raise ValueError("canonical environment.json 결정론 backend 상태가 아닙니다")
    if bool(environment["cuda_available"]) and environment.get(
        "cublas_workspace_config"
    ) not in {":4096:8", ":16:8"}:
        raise ValueError("canonical CUDA environment의 CUBLAS_WORKSPACE_CONFIG가 잘못됐습니다")
    reproducibility.update(
        {name: snapshot.sha256 for name, snapshot in snapshots.items()}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_role": role,
        "init_eligible": last_cfg.get("init_eligible") is True,
        "experiment_contract_sha256": last_contract["sha256"],
        "loss_selection_sha256": str(last_cfg.get("loss_selection_sha256", "")),
        "schedule_total_steps": total,
        "completed_step": step,
        "best_step": best_step,
        "best_metric": best_metric,
        "best_metric_key": best_metric_key,
        "best_model_state_sha256": best_model["sha256"],
        "last_checkpoint_sha256": last_sha256,
        "best_checkpoint_sha256": best_sha256,
        "reproducibility_sha256": reproducibility,
    }


def validate_completion_receipt(
    ckpt_dir: str | Path,
    *,
    expected_role: str | None = None,
    expected_init_eligible: bool | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    directory = Path(ckpt_dir)
    receipt_path = directory / RECEIPT_NAME
    raw = snapshot_regular_file(receipt_path).content
    try:
        receipt = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"completion receipt가 손상됐습니다: {receipt_path}") from exc
    if not isinstance(receipt, dict):
        raise ValueError("completion receipt 최상위가 mapping이 아닙니다")
    expected = _expected_payload(directory, repo_root=repo_root)
    allowed_keys = set(expected) | {"completed_at_unix_ns"}
    if set(receipt) != allowed_keys:
        raise ValueError(
            "completion receipt key 집합이 정확하지 않습니다: "
            f"extra={sorted(set(receipt) - allowed_keys)}, "
            f"missing={sorted(allowed_keys - set(receipt))}"
        )
    completed_at = receipt.get("completed_at_unix_ns")
    if not isinstance(completed_at, int) or completed_at <= 0:
        raise ValueError("completion receipt completed_at_unix_ns가 잘못됐습니다")
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(
                f"completion receipt {key} 불일치: saved={receipt.get(key)!r}, "
                f"expected={value!r}"
            )
    if expected_role is not None and receipt.get("experiment_role") != expected_role:
        raise ValueError(
            f"completion receipt role 불일치: {receipt.get('experiment_role')!r}"
        )
    if (
        expected_init_eligible is not None
        and receipt.get("init_eligible") is not expected_init_eligible
    ):
        raise ValueError("completion receipt init_eligible 불일치")
    return receipt


def write_completion_receipt(
    ckpt_dir: str | Path, *, repo_root: str | Path | None = None
) -> Path | None:
    """완료한 canonical run에만 영수증을 단 한 번 발급한다."""

    directory = Path(ckpt_dir)
    last_path = directory / "last.pt"
    if not last_path.is_file():
        raise FileNotFoundError(f"last.pt가 없습니다: {last_path}")
    last_checkpoint = _snapshot_checkpoint(last_path)
    last = last_checkpoint[0]
    role = str(last["cfg"].get("experiment_role", ""))
    if role not in CANONICAL_ROLES:
        return None
    payload = _expected_payload(
        directory, last_checkpoint=last_checkpoint, repo_root=repo_root
    )
    payload["completed_at_unix_ns"] = time.time_ns()
    path = directory / RECEIPT_NAME
    directory.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        validate_completion_receipt(directory, repo_root=repo_root)
        return path
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".completion.", suffix=".tmp", dir=directory
    )
    temporary = Path(temporary_name)
    try:
        content = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            validate_completion_receipt(directory, repo_root=repo_root)
        else:
            directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            validate_completion_receipt(directory, repo_root=repo_root)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


__all__ = [
    "RECEIPT_NAME",
    "SCHEMA_VERSION",
    "sha256_file",
    "validate_completion_receipt",
    "write_completion_receipt",
]
