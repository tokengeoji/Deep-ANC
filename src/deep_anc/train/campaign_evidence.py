"""canonical-pretrain campaign의 원시 증거를 다시 계산하는 검증기.

``canonical_pretrain.json`` 에 사람이 적은 NMSE/gradient/score 숫자를 믿지 않는다.
이 모듈은 ledger가 가리키는 checkpoint, fixed batch, recorded-val ``metrics.npz``를
같은 FD snapshot으로 다시 열고, G0/gradient/선택/실측 probe의 결론을 재계산한다.

receipt의 SHA는 *어떤 bytes를 읽을지* 고정할 뿐 성능 주장 자체는 아니다. 성능 수치는
항상 이 모듈이 raw artifact에서 유도한다.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import tempfile
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import CANONICAL_LOSS_GRID, loss_selection_sha256
from ..data.primary_path import resolve_digital_primary_path
from ..dsp.nonlinear import RandomNonlinear
from ..dsp.secondary_path import DifferentiableSecondaryPath, load_secondary_path
from ..dsp.timing import BandPlan, PlantDelays, TrainingTimingContract, handoff_samples_from_config
from ..losses import ANCLoss
from ..models import build_model
from .evaluation_contract import (
    FileSnapshot,
    publish_directory_noreplace,
    snapshot_regular_file,
    write_json_exclusive,
)
from .experiment_contract import (
    build_experiment_contract,
    validate_embedded_experiment_contract,
)


EVIDENCE_SCHEMA_VERSION = 1
G0_RECEIPT_KIND = "campaign_g0_overfit"
GRADIENT_RECEIPT_KIND = "campaign_gradient_budget"
PILOT_STEPS = 20_000
MEASURED_PROBE_STEPS = 5_000
G0_STEPS = 500
G0_BATCH_SIZE = 4
G0_THRESHOLD_EXCLUSIVE_DB = -6.0
PILOT_TIE_MARGIN_DB = 0.2
PILOT_SELECTION_RULE = "recorded_val_worst10_margin_0.2_db_v1"
CANONICAL_RECORDED_VAL_MANIFEST = "data/manifests/recorded_regrouped.jsonl"
_SHA256 = "0123456789abcdef"


def _root(value: str | Path) -> Path:
    root = Path(os.path.abspath(Path(value).expanduser()))
    if root.is_symlink():
        raise ValueError(f"저장소 root는 심볼릭 링크일 수 없습니다: {root}")
    if not root.is_dir():
        raise ValueError(f"저장소 root가 directory가 아닙니다: {root}")
    return root


def repo_path(root: str | Path, value: str | Path, *, label: str) -> Path:
    """저장소 내부의 symlink 없는 pathname만 prerequisite evidence로 허용한다."""

    base = _root(root)
    raw = Path(value).expanduser()
    target = Path(os.path.abspath(raw if raw.is_absolute() else base / raw))
    try:
        relative = target.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"{label}는 저장소 내부여야 합니다: {target}") from exc
    cursor = base
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise ValueError(
                f"{label} 경로에 심볼릭 링크가 있어 저장소 밖을 가리킬 수 있습니다: {cursor}"
            )
    try:
        target.resolve(strict=False).relative_to(base.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label}의 resolved path가 저장소 밖입니다: {target}") from exc
    return target


def _exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} key 집합이 정확하지 않습니다: {actual}")
    return value


def _sha(value: object, *, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in _SHA256 for character in text):
        raise ValueError(f"{label}가 64자리 SHA-256이 아닙니다")
    return text


def snapshot_reference(
    repo_root: str | Path,
    path: str | Path,
    *,
    label: str,
) -> dict[str, str]:
    """regular-file snapshot의 portable reference를 만든다."""

    root = _root(repo_root)
    snapshot = snapshot_regular_file(repo_path(root, path, label=label))
    return {
        "path": snapshot.path.relative_to(root).as_posix(),
        "sha256": snapshot.sha256,
    }


def snapshot_from_reference(
    repo_root: str | Path,
    value: object,
    *,
    label: str,
) -> FileSnapshot:
    """reference가 가리킨 **동일 FD bytes**를 hash와 함께 돌려준다."""

    root = _root(repo_root)
    entry = _exact_keys(value, {"path", "sha256"}, label=label)
    snapshot = snapshot_regular_file(
        repo_path(root, str(entry["path"]), label=label)
    )
    if snapshot.sha256 != _sha(entry["sha256"], label=f"{label}.sha256"):
        raise ValueError(f"{label} bytes SHA가 reference와 다릅니다")
    return snapshot


def _json_snapshot(snapshot: FileSnapshot, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(snapshot.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} JSON이 손상됐습니다: {snapshot.path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 최상위는 mapping이어야 합니다")
    return payload


def _torch_snapshot(snapshot: FileSnapshot, *, label: str) -> Any:
    try:
        return torch.load(
            io.BytesIO(snapshot.content), map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} torch artifact를 읽을 수 없습니다: {snapshot.path}") from exc


def _write_torch_exclusive(path: Path, payload: Any) -> None:
    """staging 안에서도 overwrite 없는 torch artifact를 만든다."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(f"evidence artifact를 덮어쓸 수 없습니다: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
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


def _write_json_in_staging(path: Path, payload: dict[str, Any]) -> None:
    """``write_json_exclusive``와 같은 no-replace JSON writer를 staging에 사용한다."""

    write_json_exclusive(path, payload)


def _publish_evidence_directory(
    repo_root: str | Path,
    output_dir: str | Path,
    *,
    build,
) -> Path:
    """sibling staging에서 complete evidence set을 원자적으로 공개한다."""

    root = _root(repo_root)
    target = repo_path(root, output_dir, label="campaign evidence output directory")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"campaign evidence directory를 덮어쓸 수 없습니다: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # target parent도 root 이후 symlink가 없는지 다시 닫는다.
    repo_path(root, target.parent, label="campaign evidence output parent")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".staging", dir=target.parent))
    try:
        build(root, staging, target)
        published = publish_directory_noreplace(staging, target)
        staging = None  # type: ignore[assignment]
        return published
    finally:
        if staging is not None and staging.exists():
            # 실패한 private staging은 evidence가 아니며 재사용도 허용하지 않는다.
            import shutil

            shutil.rmtree(staging)


def _reference_for_staged_file(root: Path, staging_file: Path, final_file: Path) -> dict[str, str]:
    snapshot = snapshot_regular_file(staging_file)
    return {
        "path": repo_path(root, final_file, label="published evidence file").relative_to(root).as_posix(),
        "sha256": snapshot.sha256,
    }


def publish_g0_evidence(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    cfg: dict[str, Any],
    model_state: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    steps: int,
    mode: str,
    primary_mode: str,
    require_nmse_db: float | None,
    nmse_only: bool,
    disable_loss_terms: list[str] | tuple[str, ...],
) -> Path:
    """G0의 model+fixed batch만을 source-of-truth로 하는 immutable receipt를 발행한다.

    수치 metric은 intentionally 저장하지 않는다. canonical validator는 final model과
    batch에서 다시 forward하여 G0를 판정한다.
    """

    if set(batch) != {"x", "d"}:
        raise ValueError("G0 evidence batch는 정확히 x,d여야 합니다")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError("G0 evidence model_state가 비었습니다")

    def build(root: Path, staging: Path, target: Path) -> None:
        checkpoint_file = staging / "checkpoint.pt"
        batch_file = staging / "batch.pt"
        protocol = {
            "mode": str(mode),
            "primary_mode": str(primary_mode),
            "steps": int(steps),
            "batch_size": int(batch["x"].shape[0]),
            "require_nmse_db": None if require_nmse_db is None else float(require_nmse_db),
            "nmse_only": bool(nmse_only),
            "disable_loss_terms": [str(item) for item in disable_loss_terms],
        }
        raw_checkpoint = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "kind": G0_RECEIPT_KIND,
            "model": {
                str(name): value.detach().cpu().contiguous().clone()
                for name, value in model_state.items()
            },
            "cfg": cfg,
            "step": int(steps),
            "protocol": protocol,
        }
        raw_batch = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in batch.items()
        }
        _write_torch_exclusive(checkpoint_file, raw_checkpoint)
        _write_torch_exclusive(batch_file, raw_batch)
        receipt = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "kind": G0_RECEIPT_KIND,
            "checkpoint": _reference_for_staged_file(
                root, checkpoint_file, target / checkpoint_file.name
            ),
            "batch": _reference_for_staged_file(root, batch_file, target / batch_file.name),
        }
        _write_json_in_staging(staging / "receipt.json", receipt)

    return _publish_evidence_directory(repo_root, output_dir, build=build) / "receipt.json"


def publish_gradient_budget_evidence(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    checkpoint: str | Path,
    batch: dict[str, torch.Tensor],
) -> Path:
    """pilot checkpoint와 fixed batch를 결속한 gradient-budget receipt를 발행한다."""

    if set(batch) != {"x", "d"}:
        raise ValueError("gradient evidence batch는 정확히 x,d여야 합니다")
    root = _root(repo_root)
    checkpoint_ref = snapshot_reference(root, checkpoint, label="gradient checkpoint")

    def build(base: Path, staging: Path, target: Path) -> None:
        batch_file = staging / "batch.pt"
        raw_batch = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in batch.items()
        }
        _write_torch_exclusive(batch_file, raw_batch)
        receipt = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "kind": GRADIENT_RECEIPT_KIND,
            "checkpoint": checkpoint_ref,
            "batch": _reference_for_staged_file(base, batch_file, target / batch_file.name),
        }
        _write_json_in_staging(staging / "receipt.json", receipt)

    return _publish_evidence_directory(root, output_dir, build=build) / "receipt.json"


def _finite_tensor(value: object, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{label}가 Tensor가 아닙니다")
    if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{label}에 NaN/Inf 또는 빈 tensor가 있습니다")
    return value


def _load_batch(snapshot: FileSnapshot, *, label: str, exact_batch_size: int | None) -> dict[str, torch.Tensor]:
    raw = _torch_snapshot(snapshot, label=label)
    batch = _exact_keys(raw, {"x", "d"}, label=f"{label} batch")
    x = _finite_tensor(batch["x"], label=f"{label}.x").detach().cpu().contiguous()
    d = _finite_tensor(batch["d"], label=f"{label}.d").detach().cpu().contiguous()
    if x.ndim != 3 or d.ndim != 3 or x.shape[0] != d.shape[0] or d.shape[1] != 1:
        raise ValueError(
            f"{label} batch shape이 [B,C,T]/[B,1,T]가 아닙니다: x={tuple(x.shape)}, d={tuple(d.shape)}"
        )
    if x.shape[-1] != d.shape[-1] or x.shape[-1] < 1:
        raise ValueError(f"{label} x/d 시간축이 다르거나 비었습니다")
    if exact_batch_size is not None and int(x.shape[0]) != int(exact_batch_size):
        raise ValueError(
            f"{label} batch size가 승인값과 다릅니다: {int(x.shape[0])} != {int(exact_batch_size)}"
        )
    return {"x": x, "d": d}


def _load_model_state(
    cfg: dict[str, Any], state: object, *, label: str, device: torch.device
) -> torch.nn.Module:
    if not isinstance(cfg.get("model"), dict):
        raise ValueError(f"{label} checkpoint cfg.model이 없습니다")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{label} checkpoint model state가 없습니다")
    model = build_model(cfg["model"])
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError(f"{label} checkpoint model state key가 현재 model과 다릅니다")
    for name, target in expected.items():
        value = _finite_tensor(state[name], label=f"{label}.model.{name}")
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(
                f"{label}.model.{name} shape/dtype가 다릅니다: "
                f"saved={tuple(value.shape)}/{value.dtype}, expected={tuple(target.shape)}/{target.dtype}"
            )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_criterion(
    cfg: dict[str, Any], *, root: Path, model: torch.nn.Module, device: torch.device
) -> tuple[ANCLoss, int]:
    data = cfg.get("data")
    duct = cfg.get("duct")
    loss = cfg.get("loss")
    if not isinstance(data, dict) or not isinstance(duct, dict) or not isinstance(loss, dict):
        raise ValueError("evidence checkpoint resolved data/duct/loss config가 없습니다")
    secondary_cfg = duct.get("secondary_path")
    if not isinstance(secondary_cfg, dict) or not secondary_cfg.get("npz"):
        raise ValueError("evidence checkpoint secondary_path.npz가 없습니다")
    fs = int(data.get("sample_rate", 0))
    secondary = load_secondary_path(
        repo_path(root, str(secondary_cfg["npz"]), label="evidence secondary path")
    )
    if int(secondary.sample_rate) != fs:
        raise ValueError("evidence secondary path sample rate가 config와 다릅니다")
    perturb = data.get("plant_perturbation") or {}
    plant = DifferentiableSecondaryPath(
        secondary,
        handoff_extra_samples=handoff_samples_from_config(duct),
        delay_jitter_range=tuple(perturb.get("delay_jitter_range", [0, 0])),
        gain_db_range=tuple(perturb.get("gain_db", [0.0, 0.0])),
        tilt_db_per_octave_range=tuple(
            perturb.get("gain_tilt_db_per_octave", [0.0, 0.0])
        ),
        allpass_perturb=bool(perturb.get("allpass_perturb", False)),
        seed=int(cfg.get("seed", 0)) + 17,
    ).to(device)
    nonlinear_cfg = data.get("nonlinear") or {}
    nonlinear = RandomNonlinear(
        nonlinear_cfg.get("sef_eta_choices", [10.0]),
        tuple(nonlinear_cfg.get("drive_range", [1.0, 1.0])),
        hardclip_prob=float(nonlinear_cfg.get("hardclip_prob", 0.0)),
        seed=int(cfg.get("seed", 0)) + 29,
    )
    band = BandPlan.resolve(
        plant_trusted_band_hz=secondary.trusted_band_hz(),
        duct_cfg=duct,
        sample_rate=fs,
    )
    criterion = ANCLoss(
        plant,
        loss,
        fs,
        nonlinear=nonlinear,
        cutoff_hz=float((duct.get("acoustics") or {}).get("plane_wave_cutoff_hz", 1633.0)),
        target_band_hz=band.target.as_tuple(),
        trusted_band_hz=band.optimize.as_tuple(),
        limiter_limit=float(model.limit),
    ).to(device)
    criterion.eval()
    return criterion, int(cfg.get("loss_start_sample", -1))


def _recompute_metrics(
    cfg: dict[str, Any], model_state: object, batch: dict[str, torch.Tensor], *, root: Path, label: str
) -> dict[str, float]:
    device = _device()
    model = _load_model_state(cfg, model_state, label=label, device=device)
    criterion, loss_start_sample = _build_criterion(cfg, root=root, model=model, device=device)
    if loss_start_sample < 0:
        raise ValueError(f"{label} loss_start_sample이 없습니다")
    x, d = batch["x"].to(device), batch["d"].to(device)
    with torch.no_grad():
        y = model(x)
        _finite_tensor(y, label=f"{label}.model_output")
        _, metrics = criterion(y, d, loss_start_sample=loss_start_sample)
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError(f"{label} 재계산 metrics가 비었습니다")
    result: dict[str, float] = {}
    for key, value in metrics.items():
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{label} 재계산 metric.{key}가 non-finite입니다")
        result[str(key)] = numeric
    return result


def _recompute_gradient_budget(
    cfg: dict[str, Any], model_state: object, batch: dict[str, torch.Tensor], *, root: Path, label: str
) -> dict[str, float]:
    device = _device()
    model = _load_model_state(cfg, model_state, label=label, device=device)
    criterion, loss_start_sample = _build_criterion(cfg, root=root, model=model, device=device)
    if loss_start_sample < 0:
        raise ValueError(f"{label} loss_start_sample이 없습니다")
    with torch.no_grad():
        y = model(batch["x"].to(device))
    _finite_tensor(y, label=f"{label}.model_output")
    budget = criterion.gradient_budget(
        y,
        batch["d"].to(device),
        loss_start_sample=loss_start_sample,
    )
    if not isinstance(budget, dict) or not budget:
        raise ValueError(f"{label} gradient budget이 비었습니다")
    result: dict[str, float] = {}
    for key, value in budget.items():
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{label} gradient budget.{key}가 non-finite입니다")
        result[str(key)] = numeric
    return result


def _validate_timing(cfg: dict[str, Any], *, root: Path, label: str) -> None:
    """checkpoint에 적힌 timing metadata가 현재 P/S bytes와 실제로 맞는지 재계산한다."""

    data = cfg.get("data")
    duct = cfg.get("duct")
    if not isinstance(data, dict) or not isinstance(duct, dict):
        raise ValueError(f"{label} data/duct config가 없습니다")
    if str(data.get("reference_mode")) != "digital":
        raise ValueError(f"{label} digital-reference evidence가 아닙니다")
    secondary_cfg = duct.get("secondary_path") or {}
    secondary_path = secondary_cfg.get("npz")
    if not secondary_path:
        raise ValueError(f"{label} secondary path가 없습니다")
    secondary = load_secondary_path(
        repo_path(root, str(secondary_path), label=f"{label} secondary path")
    )
    fs = int(data.get("sample_rate", 0))
    # resolver는 normal runtime에서 repository-root 상대 경로를 받는다. evidence
    # validator는 explicit ``repo_root``도 지원하므로, compact P/S path만 absolute로
    # 투영해 global config.REPO_ROOT에 의존하지 않게 한다.
    timing_data = copy.deepcopy(data)
    timing_duct = copy.deepcopy(duct)
    secondary_cfg = timing_duct.get("secondary_path") or {}
    secondary_cfg["npz"] = str(
        repo_path(root, str(secondary_path), label=f"{label} secondary path")
    )
    timing_duct["secondary_path"] = secondary_cfg
    digital_cfg = timing_duct.get("digital_reference") or {}
    primary_path = digital_cfg.get("primary_path_npz")
    if primary_path:
        digital_cfg["primary_path_npz"] = str(
            repo_path(root, str(primary_path), label=f"{label} primary path")
        )
    timing_duct["digital_reference"] = digital_cfg
    primary, _ = resolve_digital_primary_path(timing_data, timing_duct, fs, secondary)
    if primary is None:  # pragma: no cover - resolver contract guard
        raise ValueError(f"{label} compact primary path가 없습니다")
    delays = PlantDelays.from_config(
        duct_cfg=timing_duct,
        secondary_delay_samples=int(secondary.delay_samples),
        primary_delay_samples=int(primary.delay_samples),
        sample_rate=fs,
    )
    actual = TrainingTimingContract.derive(primary_fir=primary.fir, plant_delays=delays)
    declared = TrainingTimingContract.from_data_config(data)
    if declared != actual:
        raise ValueError(f"{label} training timing contract가 실제 P/S와 다릅니다")
    if int(data.get("digital_reference_lead_samples", -1)) != int(
        actual.digital_reference_lead_samples
    ):
        raise ValueError(f"{label} digital-reference lead가 실제 P/S와 다릅니다")
    if int(data.get("d_noise_delay_samples", -1)) != int(primary.delay_samples):
        raise ValueError(f"{label} d_noise_delay_samples가 primary path와 다릅니다")


def _target_contract_and_artifacts(
    canonical_cfg: dict[str, Any], *, root: Path
) -> dict[str, Any]:
    contract = validate_embedded_experiment_contract(canonical_cfg)
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("canonical experiment contract artifact가 없습니다")
    for name in ("primary_path", "secondary_path"):
        item = artifacts.get(name)
        if not isinstance(item, dict) or not item.get("exists") or not item.get("sha256"):
            raise ValueError(f"canonical experiment contract {name} artifact가 불완전합니다")
        # artifact contract가 현재 pathname bytes까지 여전히 가리키는지 확인한다.
        configured = (
            ((canonical_cfg.get("duct") or {}).get("digital_reference") or {}).get("primary_path_npz")
            if name == "primary_path"
            else ((canonical_cfg.get("duct") or {}).get("secondary_path") or {}).get("npz")
        )
        snapshot = snapshot_regular_file(
            repo_path(root, str(configured), label=f"canonical {name}")
        )
        if snapshot.sha256 != str(item["sha256"]):
            raise ValueError(f"canonical {name} bytes가 embedded experiment contract와 다릅니다")
    return contract


def validate_canonical_evidence_target(
    canonical_cfg: dict[str, Any], *, repo_root: str | Path
) -> dict[str, Any]:
    """canonical cfg의 embedded contract와 현재 strict P/S bytes를 함께 닫는다."""

    return _target_contract_and_artifacts(canonical_cfg, root=_root(repo_root))


def _validate_source_and_artifacts(
    evidence_cfg: dict[str, Any],
    *,
    canonical_cfg: dict[str, Any],
    canonical_contract: dict[str, Any],
    root: Path,
    label: str,
    embedded: bool,
) -> dict[str, Any]:
    """evidence가 canonical과 같은 source/bootstrap/P/S generation인지 묶는다."""

    evidence_contract = (
        validate_embedded_experiment_contract(evidence_cfg)
        if embedded
        else build_experiment_contract(evidence_cfg, repo_root=root)
    )
    expected_source = canonical_contract.get("source") or {}
    observed_source = evidence_contract.get("source") or {}
    for key in ("git_commit", "source_tree_sha256"):
        if not expected_source.get(key) or observed_source.get(key) != expected_source.get(key):
            raise ValueError(f"{label} source {key}가 canonical과 다릅니다")
    expected_input = canonical_contract.get("input_generation") or {}
    observed_input = evidence_contract.get("input_generation") or {}
    for key in (
        "bootstrap_receipt_sha256",
        "transfer_manifest_sha256",
        "recorded_transfer_aggregate_sha256",
    ):
        if not expected_input.get(key) or observed_input.get(key) != expected_input.get(key):
            raise ValueError(f"{label} input generation {key}가 canonical과 다릅니다")
    expected_artifacts = canonical_contract.get("artifacts") or {}
    observed_artifacts = evidence_contract.get("artifacts") or {}
    for name in ("primary_path", "secondary_path"):
        expected = expected_artifacts.get(name) or {}
        observed = observed_artifacts.get(name) or {}
        if not expected.get("sha256") or observed.get("sha256") != expected.get("sha256"):
            raise ValueError(f"{label} {name} SHA가 canonical과 다릅니다")
    return evidence_contract


def _validate_loss(
    evidence_cfg: dict[str, Any], canonical_cfg: dict[str, Any], *, label: str,
    allow_any_grid_alpha: bool,
    expected_pair: tuple[float, float] | None,
) -> tuple[float, float]:
    observed = evidence_cfg.get("loss")
    expected = canonical_cfg.get("loss")
    if not isinstance(observed, dict) or not isinstance(expected, dict):
        raise ValueError(f"{label} loss config가 없습니다")
    if set(observed) != set(expected):
        raise ValueError(f"{label} loss key 집합이 canonical과 다릅니다")
    pair = (
        float(observed.get("nmse_cvar_alpha", float("nan"))),
        float(observed.get("lambda_frame", float("nan"))),
    )
    if not all(math.isfinite(item) for item in pair):
        raise ValueError(f"{label} loss alpha/frame이 non-finite입니다")
    for key, value in expected.items():
        if key == "nmse_cvar_alpha" and allow_any_grid_alpha:
            continue
        if observed.get(key) != value:
            raise ValueError(f"{label} loss.{key}가 canonical과 다릅니다")
    if allow_any_grid_alpha:
        if pair not in {(float(a), float(b)) for a, b in CANONICAL_LOSS_GRID}:
            raise ValueError(f"{label} loss alpha×frame이 승인 grid가 아닙니다")
    if expected_pair is not None and pair != expected_pair:
        raise ValueError(f"{label} loss alpha×frame이 pilot winner와 다릅니다")
    digest = str(evidence_cfg.get("loss_selection_sha256", ""))
    if digest != loss_selection_sha256(observed):
        raise ValueError(f"{label} loss_selection_sha256가 embedded loss와 다릅니다")
    return pair


def _validate_derivative_cfg(
    evidence_cfg: dict[str, Any],
    *,
    canonical_cfg: dict[str, Any],
    canonical_contract: dict[str, Any],
    root: Path,
    label: str,
    role: str,
    primary_mode: str,
    embedded: bool,
    expected_pair: tuple[float, float] | None = None,
    allow_any_grid_alpha: bool = False,
) -> tuple[dict[str, Any], tuple[float, float]]:
    if str(evidence_cfg.get("experiment_role", "")) != role:
        raise ValueError(f"{label} experiment_role이 {role!r}가 아닙니다")
    if evidence_cfg.get("init_eligible") is not False:
        raise ValueError(f"{label}는 init_eligible=false여야 합니다")
    if role == "diagnostic_overfit":
        if evidence_cfg.get("contract_run_dir") is not False:
            raise ValueError(f"{label} G0는 contract_run_dir=false여야 합니다")
    elif evidence_cfg.get("contract_run_dir") is not True:
        raise ValueError(f"{label} derivative run은 contract_run_dir=true여야 합니다")
    expected_policy = f"{role}_derivative_v1"
    if evidence_cfg.get("canonical_trust_policy") != expected_policy:
        raise ValueError(f"{label} derivative trust policy가 다릅니다")
    if evidence_cfg.get("model") != canonical_cfg.get("model"):
        raise ValueError(f"{label} model config가 canonical과 다릅니다")
    if evidence_cfg.get("duct") != canonical_cfg.get("duct"):
        raise ValueError(f"{label} duct config가 canonical과 다릅니다")
    observed_data = evidence_cfg.get("data")
    expected_data = canonical_cfg.get("data")
    if not isinstance(observed_data, dict) or not isinstance(expected_data, dict):
        raise ValueError(f"{label} data config가 없습니다")
    if str(observed_data.get("digital_primary_path_mode")) != primary_mode:
        raise ValueError(f"{label} primary path mode가 {primary_mode!r}가 아닙니다")
    if str(evidence_cfg.get("physics_status")) != (
        "measured_primary_path"
        if primary_mode == "measured"
        else f"{primary_mode}_representation_pretrain"
    ):
        raise ValueError(f"{label} physics_status가 primary path mode와 다릅니다")
    for key in ("sample_rate", "reference_mode", "recorded_lead_mode"):
        if observed_data.get(key) != expected_data.get(key):
            raise ValueError(f"{label} data.{key}가 canonical과 다릅니다")
    if int(observed_data.get("digital_reference_lead_samples", -1)) != int(
        expected_data.get("digital_reference_lead_samples", -2)
    ):
        raise ValueError(f"{label} digital-reference lead가 canonical과 다릅니다")
    if evidence_cfg.get("trusted_band_hz") != canonical_cfg.get("trusted_band_hz"):
        raise ValueError(f"{label} trusted band가 canonical과 다릅니다")
    if int(evidence_cfg.get("loss_start_sample", -1)) != int(
        canonical_cfg.get("loss_start_sample", -2)
    ):
        raise ValueError(f"{label} loss_start_sample이 canonical과 다릅니다")
    if int(evidence_cfg.get("seed", -1)) != int(canonical_cfg.get("seed", -2)):
        raise ValueError(f"{label} seed가 canonical과 다릅니다")
    pair = _validate_loss(
        evidence_cfg,
        canonical_cfg,
        label=label,
        allow_any_grid_alpha=allow_any_grid_alpha,
        expected_pair=expected_pair,
    )
    _validate_timing(evidence_cfg, root=root, label=label)
    contract = _validate_source_and_artifacts(
        evidence_cfg,
        canonical_cfg=canonical_cfg,
        canonical_contract=canonical_contract,
        root=root,
        label=label,
        embedded=embedded,
    )
    return contract, pair


def validate_g0_receipt(
    receipt_reference: object,
    *,
    repo_root: str | Path,
    canonical_cfg: dict[str, Any],
    canonical_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """G0 receipt의 raw final model/batch로 trusted NMSE를 다시 계산한다."""

    root = _root(repo_root)
    contract = canonical_contract or _target_contract_and_artifacts(canonical_cfg, root=root)
    receipt_snapshot = snapshot_from_reference(root, receipt_reference, label="campaign G0 receipt")
    receipt = _exact_keys(
        _json_snapshot(receipt_snapshot, label="campaign G0 receipt"),
        {"schema_version", "kind", "checkpoint", "batch"},
        label="campaign G0 receipt",
    )
    if receipt["schema_version"] != EVIDENCE_SCHEMA_VERSION or receipt["kind"] != G0_RECEIPT_KIND:
        raise ValueError("campaign G0 receipt schema/kind가 다릅니다")
    checkpoint_snapshot = snapshot_from_reference(root, receipt["checkpoint"], label="campaign G0 checkpoint")
    batch_snapshot = snapshot_from_reference(root, receipt["batch"], label="campaign G0 batch")
    raw = _exact_keys(
        _torch_snapshot(checkpoint_snapshot, label="campaign G0 checkpoint"),
        {"schema_version", "kind", "model", "cfg", "step", "protocol"},
        label="campaign G0 checkpoint",
    )
    if raw["schema_version"] != EVIDENCE_SCHEMA_VERSION or raw["kind"] != G0_RECEIPT_KIND:
        raise ValueError("campaign G0 checkpoint schema/kind가 다릅니다")
    cfg = raw["cfg"]
    if not isinstance(cfg, dict):
        raise ValueError("campaign G0 checkpoint cfg가 mapping이 아닙니다")
    protocol = _exact_keys(
        raw["protocol"],
        {
            "mode",
            "primary_mode",
            "steps",
            "batch_size",
            "require_nmse_db",
            "nmse_only",
            "disable_loss_terms",
        },
        label="campaign G0 protocol",
    )
    if (
        protocol["mode"] != "nominal"
        or protocol["primary_mode"] != "secondary_surrogate"
        or int(protocol["steps"]) != G0_STEPS
        or int(raw["step"]) != G0_STEPS
        or int(protocol["batch_size"]) != G0_BATCH_SIZE
        or protocol["require_nmse_db"] != G0_THRESHOLD_EXCLUSIVE_DB
        or protocol["nmse_only"] is not False
        or protocol["disable_loss_terms"] != []
    ):
        raise ValueError("campaign G0 protocol이 승인된 500-step nominal control이 아닙니다")
    _validate_derivative_cfg(
        cfg,
        canonical_cfg=canonical_cfg,
        canonical_contract=contract,
        root=root,
        label="campaign G0",
        role="diagnostic_overfit",
        primary_mode="secondary_surrogate",
        embedded=False,
        allow_any_grid_alpha=True,
    )
    batch = _load_batch(batch_snapshot, label="campaign G0", exact_batch_size=G0_BATCH_SIZE)
    metrics = _recompute_metrics(cfg, raw["model"], batch, root=root, label="campaign G0")
    nmse = float(metrics.get("nmse_trusted_db", float("nan")))
    if not math.isfinite(nmse) or nmse >= G0_THRESHOLD_EXCLUSIVE_DB:
        raise ValueError(
            "campaign G0 FAIL: raw model/batch 재계산 trusted NMSE가 "
            f"{G0_THRESHOLD_EXCLUSIVE_DB:.1f} dB 미만이 아닙니다: {nmse:.6f} dB"
        )
    return {
        "receipt": receipt_snapshot,
        "checkpoint": checkpoint_snapshot,
        "batch": batch_snapshot,
        "nmse_trusted_db": nmse,
        "metrics": metrics,
    }


def _load_checkpoint(snapshot: FileSnapshot, *, label: str) -> dict[str, Any]:
    raw = _torch_snapshot(snapshot, label=label)
    if not isinstance(raw, dict):
        raise ValueError(f"{label} checkpoint 최상위가 mapping이 아닙니다")
    cfg = raw.get("cfg")
    if not isinstance(cfg, dict) or not isinstance(raw.get("model"), dict):
        raise ValueError(f"{label} checkpoint cfg/model이 없습니다")
    return raw


def _checkpoint_step(raw: dict[str, Any], *, label: str) -> int:
    try:
        step = int(raw.get("step", -1))
        stream = raw["data_stream"]
        batch_index = int(stream["global_batch_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} checkpoint step/data_stream이 없습니다") from exc
    if step < 1 or batch_index != step:
        raise ValueError(f"{label} checkpoint step/data_stream이 다릅니다")
    return step


def _finite_checkpoint_model(raw: dict[str, Any], *, cfg: dict[str, Any], label: str) -> None:
    # forward를 하지 않고도 model key/shape/dtype/finite를 닫는다. model 생성은
    # raw checkpoint의 config가 현실의 tiny model과 같은지를 동시에 검사한다.
    _load_model_state(cfg, raw["model"], label=label, device=torch.device("cpu"))


def _npz(snapshot: FileSnapshot, *, label: str):
    try:
        return np.load(io.BytesIO(snapshot.content), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} metrics.npz가 손상됐습니다") from exc


def _npz_scalar(data, key: str, *, label: str) -> object:
    if key not in data.files:
        raise ValueError(f"{label} metrics.npz에 {key}가 없습니다")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{label} metrics.npz {key}는 scalar여야 합니다")
    return value.reshape(-1)[0].item()


def _metric_float(data, key: str, *, label: str) -> float:
    value = float(_npz_scalar(data, key, label=label))
    if not math.isfinite(value):
        raise ValueError(f"{label} metrics.npz {key}가 non-finite입니다")
    return value


def _recorded_distribution(values: np.ndarray, *, label: str) -> dict[str, float]:
    """``eval.recorded._distribution``과 같은 raw segment 통계만 다시 계산한다.

    campaign ledger가 신뢰하는 selection metric은 ``metrics.npz``의 사람이 바꿀 수
    있는 scalar가 아니라 이 배열의 CVaR@10%다. 평가 모듈을 import해서 우연히 같은
    scalar를 다시 읽는 일이 없도록 작은 동일 공식을 여기서 고정한다.
    """

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 1 or not bool(np.isfinite(array).all()):
        raise ValueError(f"{label} raw segment metric이 empty/non-finite입니다")
    worst_count = max(1, int(math.ceil(array.size * 0.1)))
    worst = np.sort(array)[-worst_count:]
    return {
        "mean_db": float(np.mean(array)),
        "worst10_mean_db": float(np.mean(worst)),
    }


def _require_recomputed_scalar(
    data,
    key: str,
    expected: float,
    *,
    label: str,
) -> None:
    observed = _metric_float(data, key, label=label)
    # metrics.npz가 float32 segment 배열을 저장하는 경우에도 evaluator가 낸
    # float64 summary와 의미 있게 같은지 확인한다. 이 허용오차는 성능 경계가 아니라
    # 직렬화 오차 전용이며, 임의의 ledger score를 허용하지 않는다.
    if not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"{label} metrics.npz {key} summary scalar가 raw segment 재계산값과 다릅니다: "
            f"saved={observed!r}, recomputed={expected!r}"
        )


def _validate_recorded_metrics(
    metrics_snapshot: FileSnapshot,
    *,
    label: str,
    checkpoint_snapshot: FileSnapshot,
    checkpoint_contract_sha256: str,
    manifest_snapshot: FileSnapshot,
    canonical_contract: dict[str, Any],
    expected_surrogate: bool,
    expected_physics: str,
) -> float:
    """recorded-val raw NPZ의 provenance와 selection metric을 직접 확인한다."""

    with _npz(metrics_snapshot, label=label) as data:
        observed = {
            "split": str(_npz_scalar(data, "split", label=label)),
            "checkpoint_sha256": str(_npz_scalar(data, "checkpoint_sha256", label=label)),
            "manifest_sha256": str(_npz_scalar(data, "manifest_sha256", label=label)),
            "experiment_contract_sha256": str(
                _npz_scalar(data, "experiment_contract_sha256", label=label)
            ),
            "allow_surrogate": bool(_npz_scalar(data, "allow_surrogate", label=label)),
            "physics_status": str(_npz_scalar(data, "physics_status", label=label)),
        }
        expected = {
            "split": "val",
            "checkpoint_sha256": checkpoint_snapshot.sha256,
            "manifest_sha256": manifest_snapshot.sha256,
            "experiment_contract_sha256": checkpoint_contract_sha256,
            "allow_surrogate": expected_surrogate,
            "physics_status": expected_physics,
        }
        if observed != expected:
            raise ValueError(f"{label} recorded-val metrics provenance가 raw checkpoint/manifest와 다릅니다")
        target_artifacts = canonical_contract.get("artifacts") or {}
        for key, artifact in (
            ("primary_path_sha256", target_artifacts.get("primary_path") or {}),
            ("secondary_path_sha256", target_artifacts.get("secondary_path") or {}),
        ):
            if str(_npz_scalar(data, key, label=label)) != str(artifact.get("sha256", "")):
                raise ValueError(f"{label} metrics {key}가 canonical P/S와 다릅니다")
        counts = {
            name: int(_npz_scalar(data, name, label=label))
            for name in ("n_sessions", "n_groups", "n_segments")
        }
        if any(value < 1 for value in counts.values()):
            raise ValueError(f"{label} recorded-val metrics 표본이 비었습니다: {counts}")
        raw_metrics: dict[str, np.ndarray] = {}
        for key in ("per_segment_trusted_db", "per_segment_fullband_db"):
            if key not in data.files:
                raise ValueError(f"{label} metrics.npz에 {key}가 없습니다")
            values = np.asarray(data[key], dtype=np.float64)
            if values.ndim != 1 or values.size != counts["n_segments"]:
                raise ValueError(
                    f"{label} metrics.npz {key} 길이가 n_segments와 다릅니다: "
                    f"{values.shape} vs {counts['n_segments']}"
                )
            raw_metrics[key] = values
        trusted = _recorded_distribution(
            raw_metrics["per_segment_trusted_db"], label=f"{label} trusted"
        )
        fullband = _recorded_distribution(
            raw_metrics["per_segment_fullband_db"], label=f"{label} fullband"
        )
        _require_recomputed_scalar(
            data, "nmse_trusted_mean_db", trusted["mean_db"], label=label
        )
        _require_recomputed_scalar(
            data,
            "nmse_trusted_worst10_mean_db",
            trusted["worst10_mean_db"],
            label=label,
        )
        _require_recomputed_scalar(
            data, "nmse_fullband_mean_db", fullband["mean_db"], label=label
        )
        _require_recomputed_scalar(
            data,
            "nmse_fullband_worst10_mean_db",
            fullband["worst10_mean_db"],
            label=label,
        )
        # probe의 `passed=true` 같은 서술은 전혀 읽지 않는다. 아래 G4 요약은 probe
        # completion의 finite 확인용일 뿐 5k probe에 새 성능 pass threshold를 발명하지
        # 않는다. loss pilot selection은 위에서 재계산한 trusted CVaR@10%만 사용한다.
        for key in (
            "g4_worst_source_trusted_mean_db",
            "g4_worst_source_trusted_worst10_db",
            "g4_worst_octave_worst10_db",
        ):
            _metric_float(data, key, label=label)
    return trusted["worst10_mean_db"]


def _validate_pilot_checkpoint_pair(
    candidate: object,
    *,
    repo_root: Path,
    canonical_cfg: dict[str, Any],
    canonical_contract: dict[str, Any],
    expected_role: str,
    expected_primary_mode: str,
    expected_steps: int,
    expected_pair: tuple[float, float] | None,
    label: str,
) -> dict[str, Any]:
    """best/last/recorded-val three-way provenance을 하나의 derivative run으로 닫는다."""

    row = _exact_keys(
        candidate,
        {"best_checkpoint", "last_checkpoint", "metrics", "manifest"},
        label=label,
    )
    best_snapshot = snapshot_from_reference(repo_root, row["best_checkpoint"], label=f"{label} best checkpoint")
    last_snapshot = snapshot_from_reference(repo_root, row["last_checkpoint"], label=f"{label} last checkpoint")
    metrics_snapshot = snapshot_from_reference(repo_root, row["metrics"], label=f"{label} metrics")
    manifest_snapshot = snapshot_from_reference(repo_root, row["manifest"], label=f"{label} manifest")
    if best_snapshot.path.name != "best.pt" or last_snapshot.path.name != "last.pt":
        raise ValueError(f"{label}는 best.pt와 last.pt raw checkpoint를 가리켜야 합니다")
    expected_manifest = repo_path(
        repo_root, CANONICAL_RECORDED_VAL_MANIFEST, label=f"{label} canonical recorded manifest"
    )
    if manifest_snapshot.path != expected_manifest:
        raise ValueError(f"{label} manifest는 canonical recorded_regrouped.jsonl이어야 합니다")
    best = _load_checkpoint(best_snapshot, label=f"{label} best checkpoint")
    last = _load_checkpoint(last_snapshot, label=f"{label} last checkpoint")
    best_cfg = best["cfg"]
    last_cfg = last["cfg"]
    best_contract, pair = _validate_derivative_cfg(
        best_cfg,
        canonical_cfg=canonical_cfg,
        canonical_contract=canonical_contract,
        root=repo_root,
        label=f"{label} best checkpoint",
        role=expected_role,
        primary_mode=expected_primary_mode,
        embedded=True,
        expected_pair=expected_pair,
        allow_any_grid_alpha=expected_pair is None,
    )
    last_contract, last_pair = _validate_derivative_cfg(
        last_cfg,
        canonical_cfg=canonical_cfg,
        canonical_contract=canonical_contract,
        root=repo_root,
        label=f"{label} last checkpoint",
        role=expected_role,
        primary_mode=expected_primary_mode,
        embedded=True,
        expected_pair=pair,
        allow_any_grid_alpha=False,
    )
    if best_contract["sha256"] != last_contract["sha256"] or pair != last_pair:
        raise ValueError(f"{label} best/last experiment contract 또는 loss pair가 다릅니다")
    best_step = _checkpoint_step(best, label=f"{label} best checkpoint")
    last_step = _checkpoint_step(last, label=f"{label} last checkpoint")
    if last_step != expected_steps or int(last_cfg.get("run_until_step", -1)) != expected_steps:
        raise ValueError(f"{label} last checkpoint가 승인 step까지 완료되지 않았습니다")
    eval_every = int(best_cfg.get("eval_every", 0))
    if best_step > expected_steps or eval_every < 1 or best_step % eval_every:
        raise ValueError(f"{label} best checkpoint step/eval cadence가 잘못됐습니다")
    _finite_checkpoint_model(best, cfg=best_cfg, label=f"{label} best checkpoint")
    _finite_checkpoint_model(last, cfg=last_cfg, label=f"{label} last checkpoint")
    score = _validate_recorded_metrics(
        metrics_snapshot,
        label=label,
        checkpoint_snapshot=best_snapshot,
        checkpoint_contract_sha256=str(best_contract["sha256"]),
        manifest_snapshot=manifest_snapshot,
        canonical_contract=canonical_contract,
        expected_surrogate=expected_primary_mode != "measured",
        expected_physics=(
            "measured_primary_path"
            if expected_primary_mode == "measured"
            else f"{expected_primary_mode}_representation_pretrain"
        ),
    )
    return {
        "best_snapshot": best_snapshot,
        "last_snapshot": last_snapshot,
        "metrics_snapshot": metrics_snapshot,
        "manifest_snapshot": manifest_snapshot,
        "best": best,
        "last": last,
        "pair": pair,
        "score_db": score,
        "contract_sha256": str(best_contract["sha256"]),
    }


def validate_loss_pilot_candidate(
    candidate: object,
    *,
    repo_root: str | Path,
    canonical_cfg: dict[str, Any],
    canonical_contract: dict[str, Any] | None = None,
    label: str = "loss pilot candidate",
) -> dict[str, Any]:
    root = _root(repo_root)
    contract = canonical_contract or _target_contract_and_artifacts(canonical_cfg, root=root)
    result = _validate_pilot_checkpoint_pair(
        candidate,
        repo_root=root,
        canonical_cfg=canonical_cfg,
        canonical_contract=contract,
        expected_role="loss_pilot",
        expected_primary_mode="secondary_surrogate",
        expected_steps=PILOT_STEPS,
        expected_pair=None,
        label=label,
    )
    if int(result["best"]["cfg"].get("run_until_step", -1)) != PILOT_STEPS:
        # cfg is checked above; this branch is intentionally defensive if a future
        # checkpoint schema adds a top-level operational field.
        raise ValueError(f"{label} best checkpoint run_until_step이 승인값과 다릅니다")
    return result


def select_loss_pilot(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """raw recorded-val score에서만 20k loss pilot winner를 유도한다.

    0.7/1.0의 차이가 0.2 dB 이내면 0.85 raw run이 **반드시** 있어야 한다.
    이어서 최저점과 0.2 dB 이내인 후보가 여러 개면 사전 규칙대로 0.7을 택한다.
    alpha=1.0의 '불안정' 분기는 현재 immutable raw failure receipt schema가 없으므로
    의도적으로 자동 승격하지 않는다. malformed/non-finite artifact는 evidence가 아니라
    failure이며, 새 schema를 정의할 때까지 canonical pretrain을 막는다.
    """

    if not isinstance(candidates, list) or not candidates:
        raise ValueError("loss pilot candidate가 비었습니다")
    by_pair: dict[tuple[float, float], dict[str, Any]] = {}
    for candidate in candidates:
        pair = tuple(candidate.get("pair", ()))
        if len(pair) != 2:
            raise ValueError("loss pilot derived pair가 없습니다")
        pair = (float(pair[0]), float(pair[1]))
        score = float(candidate.get("score_db", float("nan")))
        if pair in by_pair or not math.isfinite(score):
            raise ValueError("loss pilot pair 중복 또는 non-finite recorded-val score")
        by_pair[pair] = candidate
    base = {(0.7, 0.0), (1.0, 0.0)}
    optional = (0.85, 0.0)
    pairs = set(by_pair)
    if not base.issubset(pairs) or pairs - (base | {optional}):
        raise ValueError("loss pilot candidate 집합이 승인 alpha×frame grid와 다릅니다")
    base_gap = abs(float(by_pair[(0.7, 0.0)]["score_db"]) - float(by_pair[(1.0, 0.0)]["score_db"]))
    needs_alpha_085 = base_gap <= PILOT_TIE_MARGIN_DB
    if needs_alpha_085 != (optional in pairs):
        raise ValueError(
            "loss pilot 0.7/1.0 margin과 alpha=0.85 candidate 집합이 다릅니다: "
            f"gap={base_gap:.6f} dB, margin={PILOT_TIE_MARGIN_DB:.1f} dB"
        )
    best_score = min(float(row["score_db"]) for row in by_pair.values())
    tied = {
        pair
        for pair, row in by_pair.items()
        if float(row["score_db"]) <= best_score + PILOT_TIE_MARGIN_DB
    }
    winner_pair = (0.7, 0.0) if (0.7, 0.0) in tied else min(
        tied, key=lambda pair: (float(by_pair[pair]["score_db"]), pair)
    )
    return {
        "winner_pair": winner_pair,
        "winner": by_pair[winner_pair],
        "base_gap_db": base_gap,
        "used_alpha_085": optional in pairs,
        "candidates": [by_pair[pair] for pair in sorted(by_pair)],
    }


def validate_gradient_budget_receipt(
    receipt_reference: object,
    *,
    repo_root: str | Path,
    canonical_cfg: dict[str, Any],
    expected_checkpoint_sha256: str,
    expected_pair: tuple[float, float],
    canonical_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """selected pilot raw checkpoint/batch에서 DNH gradient share를 재계산한다."""

    root = _root(repo_root)
    contract = canonical_contract or _target_contract_and_artifacts(canonical_cfg, root=root)
    receipt_snapshot = snapshot_from_reference(root, receipt_reference, label="gradient budget receipt")
    receipt = _exact_keys(
        _json_snapshot(receipt_snapshot, label="gradient budget receipt"),
        {"schema_version", "kind", "checkpoint", "batch"},
        label="gradient budget receipt",
    )
    if receipt["schema_version"] != EVIDENCE_SCHEMA_VERSION or receipt["kind"] != GRADIENT_RECEIPT_KIND:
        raise ValueError("gradient budget receipt schema/kind가 다릅니다")
    checkpoint_snapshot = snapshot_from_reference(root, receipt["checkpoint"], label="gradient budget checkpoint")
    if checkpoint_snapshot.sha256 != _sha(expected_checkpoint_sha256, label="gradient expected checkpoint SHA"):
        raise ValueError("gradient budget checkpoint가 selected loss pilot winner와 다릅니다")
    batch_snapshot = snapshot_from_reference(root, receipt["batch"], label="gradient budget batch")
    raw = _load_checkpoint(checkpoint_snapshot, label="gradient budget checkpoint")
    cfg = raw["cfg"]
    _validate_derivative_cfg(
        cfg,
        canonical_cfg=canonical_cfg,
        canonical_contract=contract,
        root=root,
        label="gradient budget checkpoint",
        role="loss_pilot",
        primary_mode="secondary_surrogate",
        embedded=True,
        expected_pair=expected_pair,
    )
    _finite_checkpoint_model(raw, cfg=cfg, label="gradient budget checkpoint")
    batch = _load_batch(batch_snapshot, label="gradient budget", exact_batch_size=G0_BATCH_SIZE)
    budget = _recompute_gradient_budget(
        cfg, raw["model"], batch, root=root, label="gradient budget"
    )
    share = float(budget.get("dnh", float("nan")))
    if not math.isfinite(share) or not 0.2 <= share <= 0.4:
        raise ValueError(
            "strict-S lambda_dnh raw gradient share가 승인 범위 0.2–0.4가 아닙니다: "
            f"{share!r}"
        )
    return {
        "receipt": receipt_snapshot,
        "checkpoint": checkpoint_snapshot,
        "batch": batch_snapshot,
        "gradient_share": share,
        "budget": budget,
    }


def validate_measured_probe(
    probe: object,
    *,
    repo_root: str | Path,
    canonical_cfg: dict[str, Any],
    expected_pair: tuple[float, float],
    expected_init_checkpoint_sha256: str,
    canonical_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """5k measured probe의 raw checkpoints/metrics에서 completion과 provenance를 확인한다."""

    root = _root(repo_root)
    contract = canonical_contract or _target_contract_and_artifacts(canonical_cfg, root=root)
    row = _exact_keys(
        probe,
        {"best_checkpoint", "last_checkpoint", "metrics", "manifest", "init_checkpoint"},
        label="measured probe",
    )
    init_snapshot = snapshot_from_reference(root, row["init_checkpoint"], label="measured probe init checkpoint")
    if init_snapshot.sha256 != _sha(expected_init_checkpoint_sha256, label="measured probe expected init SHA"):
        raise ValueError("measured probe init checkpoint가 selected pilot winner와 다릅니다")
    pair_result = _validate_pilot_checkpoint_pair(
        {key: row[key] for key in ("best_checkpoint", "last_checkpoint", "metrics", "manifest")},
        repo_root=root,
        canonical_cfg=canonical_cfg,
        canonical_contract=contract,
        expected_role="measured_probe",
        expected_primary_mode="measured",
        expected_steps=MEASURED_PROBE_STEPS,
        expected_pair=expected_pair,
        label="measured probe",
    )
    saved_init = pair_result["best"]["cfg"].get("init_ckpt")
    if not saved_init:
        raise ValueError("measured probe best checkpoint에 init_ckpt가 없습니다")
    saved_init_path = repo_path(root, str(saved_init), label="measured probe saved init checkpoint")
    if saved_init_path != init_snapshot.path:
        raise ValueError("measured probe checkpoint cfg.init_ckpt가 immutable init reference와 다릅니다")
    return {**pair_result, "init_snapshot": init_snapshot}


def make_campaign_evidence_reference(
    repo_root: str | Path, path: str | Path, *, label: str
) -> dict[str, str]:
    """issuer CLI가 raw artifact path를 ledger reference로 바꿀 때 쓰는 public helper."""

    return snapshot_reference(repo_root, path, label=label)


__all__ = [
    "CANONICAL_RECORDED_VAL_MANIFEST",
    "EVIDENCE_SCHEMA_VERSION",
    "G0_BATCH_SIZE",
    "G0_RECEIPT_KIND",
    "G0_STEPS",
    "G0_THRESHOLD_EXCLUSIVE_DB",
    "GRADIENT_RECEIPT_KIND",
    "MEASURED_PROBE_STEPS",
    "PILOT_STEPS",
    "PILOT_SELECTION_RULE",
    "PILOT_TIE_MARGIN_DB",
    "make_campaign_evidence_reference",
    "publish_g0_evidence",
    "publish_gradient_budget_evidence",
    "repo_path",
    "select_loss_pilot",
    "snapshot_from_reference",
    "snapshot_reference",
    "validate_g0_receipt",
    "validate_canonical_evidence_target",
    "validate_gradient_budget_receipt",
    "validate_loss_pilot_candidate",
    "validate_measured_probe",
]
