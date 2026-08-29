"""체크포인트 저장/로드 — 모델+옵티마이저+스케줄러+step+RNG 완전 재개."""

from __future__ import annotations

import math
import io
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from .a100_pretrain_smoke import A100_PRETRAIN_SMOKE_ROLE
from .evaluation_contract import FileSnapshot, snapshot_regular_file


WORLD1_CUDA_RNG_ROLES = frozenset(
    {
        "loss_pilot",
        "measured_probe",
        "canonical_pretrain",
        "canonical_finetune",
        A100_PRETRAIN_SMOKE_ROLE,
    }
)


def validate_world1_cuda_rng(state: dict, *, label: str) -> None:
    """A100 world=1 공식 학습 artifact의 CUDA RNG를 exactly-one으로 강제한다."""

    cfg = state.get("cfg") if isinstance(state, dict) else None
    role = str(cfg.get("experiment_role", "")) if isinstance(cfg, dict) else ""
    if role not in WORLD1_CUDA_RNG_ROLES:
        return
    rng = state.get("rng")
    cuda_rng = rng.get("cuda") if isinstance(rng, dict) else None
    if not isinstance(cuda_rng, list) or len(cuda_rng) != 1:
        raise ValueError(
            f"{label} ({role})에는 A100 world1 CUDA RNG state가 정확히 하나 필요합니다"
        )
    cuda_state = cuda_rng[0]
    if (
        not isinstance(cuda_state, torch.Tensor)
        or cuda_state.dtype != torch.uint8
        or cuda_state.ndim != 1
    ):
        raise ValueError(f"{label} ({role}) CUDA RNG tensor schema가 잘못됐습니다")


def _require_finite_nested(name: str, value) -> None:
    if isinstance(value, torch.Tensor):
        if value.numel() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"resume checkpoint {name}에 NaN/Inf가 있습니다")
    elif isinstance(value, dict):
        for key, item in value.items():
            _require_finite_nested(f"{name}.{key}", item)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_nested(f"{name}[{index}]", item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"resume checkpoint {name}에 NaN/Inf가 있습니다")


def _validate_rng_preview(rng: object) -> None:
    if not isinstance(rng, dict):
        raise ValueError("resume checkpoint에 RNG mapping이 없습니다")
    try:
        random.Random().setstate(rng["python"])
        np.random.RandomState().set_state(rng["numpy"])
        torch_state = rng["torch"]
        if not isinstance(torch_state, torch.Tensor):
            raise TypeError("torch RNG가 tensor가 아닙니다")
        if torch_state.dtype != torch.uint8 or torch_state.ndim != 1:
            raise ValueError("torch RNG tensor schema가 잘못됐습니다")
        cuda = rng.get("cuda")
        if cuda is not None:
            if not isinstance(cuda, list):
                raise TypeError("CUDA RNG가 list가 아닙니다")
            for state in cuda:
                if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8:
                    raise ValueError("CUDA RNG tensor schema가 잘못됐습니다")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError("resume checkpoint RNG 상태 preview 검증에 실패했습니다") from exc


def _validate_training_state_preview(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("resume checkpoint training_state schema가 없습니다")
    schema = int(value.get("schema_version", -1))
    if schema == 1:
        expected_keys = {"schema_version", "plant_rng", "nonlinear_rng"}
    elif schema == 2:
        expected_keys = {
            "schema_version", "plant_rng_kind", "plant_rng", "nonlinear_rng"
        }
        if (
            value.get("plant_rng_kind")
            != "not_applicable_frozen_causal_fir"
            or value.get("plant_rng") is not None
        ):
            raise ValueError("resume checkpoint frozen causal plant RNG marker가 다릅니다")
    else:
        raise ValueError("resume checkpoint training_state schema가 없습니다")
    if schema == 2 and set(value) != expected_keys:
        raise ValueError("resume checkpoint training_state key 집합이 exact하지 않습니다")
    for key in ("plant_rng", "nonlinear_rng"):
        state = value.get(key)
        if (key == "nonlinear_rng" or schema == 2) and state is None:
            continue
        if not isinstance(state, dict):
            raise ValueError(f"resume checkpoint {key} 상태가 없습니다")
        try:
            probe = np.random.default_rng()
            probe.bit_generator.state = state
        except (TypeError, ValueError) as exc:
            raise ValueError(f"resume checkpoint {key}를 복원할 수 없습니다") from exc


def _is_unselected_smoke_stop_metric(
    *, checkpoint: Path, state: dict, saved_cfg: dict, best_metric: float
) -> bool:
    """첫 eval 전 bounded smoke stop.pt의 ``+inf`` selection sentinel만 허용한다."""

    if (
        checkpoint.name != "stop.pt"
        or str(saved_cfg.get("experiment_role", "")) != A100_PRETRAIN_SMOKE_ROLE
        or best_metric != float("inf")
    ):
        return False
    try:
        step = int(state.get("step", -1))
        run_until = int(saved_cfg.get("run_until_step", -1))
        eval_every = int(saved_cfg.get("eval_every", -1))
    except (TypeError, ValueError):
        return False
    return 200 <= step <= 500 and step == run_until and step < eval_every


def validate_resume_checkpoint_preview(
    path: str | Path,
    state: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> None:
    """어떤 live state도 바꾸기 전에 exact-resume 전체 schema를 검증한다."""

    checkpoint = Path(path)
    saved_cfg_preview = state.get("cfg") if isinstance(state.get("cfg"), dict) else {}
    is_smoke_stop = (
        str(saved_cfg_preview.get("experiment_role", ""))
        == A100_PRETRAIN_SMOKE_ROLE
        and checkpoint.name == "stop.pt"
    )
    if checkpoint.name != "last.pt" and not is_smoke_stop:
        raise ValueError(f"resume은 last.pt만 허용합니다: {checkpoint}")
    raw_model = model.module if hasattr(model, "module") else model
    expected_model = raw_model.state_dict()
    saved_model = state.get("model")
    if not isinstance(saved_model, dict) or saved_model.keys() != expected_model.keys():
        raise ValueError("resume checkpoint model state key가 현재 모델과 다릅니다")
    for name, expected in expected_model.items():
        saved = saved_model[name]
        if (
            not isinstance(saved, torch.Tensor)
            or saved.shape != expected.shape
            or saved.dtype != expected.dtype
        ):
            raise ValueError(f"resume checkpoint model.{name} shape/dtype가 다릅니다")
    _require_finite_nested("model", saved_model)

    saved_optimizer = state.get("optimizer")
    expected_optimizer = optimizer.state_dict()
    if not isinstance(saved_optimizer, dict) or not isinstance(
        saved_optimizer.get("param_groups"), list
    ):
        raise ValueError("resume checkpoint optimizer 상태 schema가 없습니다")
    if len(saved_optimizer["param_groups"]) != len(expected_optimizer["param_groups"]):
        raise ValueError("resume checkpoint optimizer param group 수가 다릅니다")
    for saved_group, expected_group in zip(
        saved_optimizer["param_groups"], expected_optimizer["param_groups"]
    ):
        if len(saved_group.get("params", [])) != len(expected_group.get("params", [])):
            raise ValueError("resume checkpoint optimizer parameter 수가 다릅니다")
    _require_finite_nested("optimizer", saved_optimizer)

    saved_scheduler = state.get("scheduler")
    if scheduler is not None:
        expected_scheduler = scheduler.state_dict()
        if not isinstance(saved_scheduler, dict):
            raise ValueError("resume checkpoint scheduler schema가 없습니다")
        missing = sorted(set(expected_scheduler) - set(saved_scheduler))
        if missing:
            raise ValueError(f"resume checkpoint scheduler key가 없습니다: {missing}")
        _require_finite_nested("scheduler", saved_scheduler)

    step = int(state.get("step", -1))
    stream = state.get("data_stream")
    if (
        step < 0
        or not isinstance(stream, dict)
        or int(stream.get("schema_version", -1)) != 1
        or int(stream.get("global_batch_index", -1)) != step
    ):
        raise ValueError("resume checkpoint step/data_stream schema가 불일치합니다")
    best_metric = float(state.get("best_metric", float("nan")))
    if not math.isfinite(best_metric) and not _is_unselected_smoke_stop_metric(
        checkpoint=checkpoint,
        state=state,
        saved_cfg=saved_cfg_preview,
        best_metric=best_metric,
    ):
        raise ValueError("resume checkpoint best_metric이 유효하지 않습니다")
    _validate_rng_preview(state.get("rng"))
    validate_world1_cuda_rng(state, label="resume checkpoint")
    _validate_training_state_preview(state.get("training_state"))


def read_checkpoint_snapshot(
    path: str | Path, map_location: str = "cpu"
) -> tuple[dict, FileSnapshot]:
    """동일 FD bytes의 state와 SHA snapshot을 함께 반환한다."""

    snapshot = snapshot_regular_file(path)
    state = torch.load(
        io.BytesIO(snapshot.content),
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint 최상위가 dict가 아닙니다: {path}")
    if "model" not in state or "cfg" not in state:
        raise ValueError(f"checkpoint에 model/cfg가 없습니다: {path}")
    return state, snapshot


def read_checkpoint_state(path: str | Path, map_location: str = "cpu") -> dict:
    """상태를 적용하지 않고 metadata/contract를 먼저 검사하기 위한 로더."""

    state, _ = read_checkpoint_snapshot(path, map_location=map_location)
    return state


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    best_metric: float,
    cfg: dict,
    training_state: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if (path.parent / "completion.json").exists() or (
        path.parent / "completion.json"
    ).is_symlink():
        raise FileExistsError(
            f"완료 receipt가 있는 checkpoint 디렉터리는 변경할 수 없습니다: {path.parent}"
        )
    raw_model = model.module if hasattr(model, "module") else model
    state = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": int(step),
        "data_stream": {
            "schema_version": 1,
            "global_batch_index": int(step),
        },
        "best_metric": float(best_metric),
        "cfg": cfg,
        "training_state": training_state,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    restore_rng: bool = True,
    map_location: str = "cpu",
    preloaded_state: dict | None = None,
) -> dict:
    state = (
        read_checkpoint_state(path, map_location=map_location)
        if preloaded_state is None
        else preloaded_state
    )
    if not isinstance(state, dict) or "model" not in state or "cfg" not in state:
        raise ValueError("preloaded checkpoint state에 model/cfg가 없습니다")
    if restore_rng:
        if optimizer is None:
            raise ValueError("exact resume에는 optimizer가 필요합니다")
        validate_resume_checkpoint_preview(path, state, model, optimizer, scheduler)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(state["model"])
    if optimizer is not None:
        if state.get("optimizer") is None:
            raise ValueError("resume checkpoint에 optimizer 상태가 없습니다")
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        if state.get("scheduler") is None:
            raise ValueError("resume checkpoint에 scheduler 상태가 없습니다")
        scheduler.load_state_dict(state["scheduler"])
    if restore_rng:
        if "rng" not in state:
            raise ValueError("resume checkpoint에 RNG 상태가 없습니다")
        rng = state["rng"]
        try:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError("resume checkpoint RNG 상태를 복원할 수 없습니다") from exc
    return state


__all__ = [
    "load_checkpoint",
    "read_checkpoint_snapshot",
    "read_checkpoint_state",
    "save_checkpoint",
    "validate_resume_checkpoint_preview",
    "validate_world1_cuda_rng",
]
