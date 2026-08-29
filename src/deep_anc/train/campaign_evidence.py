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

from ..config import (
    CANONICAL_DETERMINISM_POLICY,
    CANONICAL_LOSS_GRID,
    CANONICAL_LOSS_PILOT_STEPS,
    CANONICAL_MEASURED_PROBE_POLICY,
    CANONICAL_MEASURED_PROBE_STEPS,
    CANONICAL_RECORDED_MANIFEST,
    canonical_recorded_manifest_for_data,
    loss_selection_sha256,
    validate_canonical_training_policy,
)
from ..data.primary_path import resolve_digital_primary_path
from ..dsp.secondary_path import load_secondary_path
from ..dsp.timing import PlantDelays, TrainingTimingContract, handoff_samples_from_config
from ..losses import ANCLoss
from ..models import build_model
from .criterion_factory import (
    CriterionAdmission,
    admit_criterion_config,
    build_criterion_from_config,
)
from .evaluation_contract import (
    FileSnapshot,
    publish_directory_noreplace,
    snapshot_regular_file,
    validate_persisted_g4_metrics,
    write_json_exclusive,
)
from .experiment_contract import (
    build_experiment_contract,
    validate_embedded_experiment_contract,
)
from .checkpoint import validate_world1_cuda_rng


# v2부터 G0 receipt가 실제 학습 프로세스의 PyTorch/cuDNN/CUBLAS 결정론
# environment snapshot을 SHA로 결속한다. exact-key 검증만으로도 구 receipt를
# 거부할 수 있지만, 같은 schema 번호로 의미가 달라지면 외부 감사자가 구분할 수
# 없으므로 명시적으로 세대를 올린다. gradient receipt도 같은 campaign evidence
# 세대에 속하므로 새 campaign에서 함께 재발행한다.
EVIDENCE_SCHEMA_VERSION = 2
G0_RECEIPT_KIND = "campaign_g0_overfit"
FAILED_G0_RECEIPT_KIND = "campaign_g0_overfit_failed_diagnostic"
G0_DETERMINISM_ENVIRONMENT_KIND = "campaign_g0_determinism_environment"
GRADIENT_RECEIPT_KIND = "campaign_gradient_budget"
PREPILOT_GRADIENT_RECEIPT_KIND = "campaign_prepilot_dnh_output_gradient"
FAILED_G0_GRADIENT_RECEIPT_KIND = "campaign_failed_g0_dnh_output_gradient_diagnostic"
# Gradient receipt v3부터 이 증거가 model parameter-gradient가 아니라 손실의
# ``model output y``에 대한 L2 gradient norm 비율임을 schema에 명시한다. G0와
# gradient receipt는 서로 다른 의미 세대이므로 G0 schema를 불필요하게 올리지 않는다.
GRADIENT_RECEIPT_SCHEMA_VERSION = 3
DNH_GRADIENT_DOMAIN = "model_output_y"
DNH_GRADIENT_NORM = "global_l2"
DNH_GRADIENT_SHARE_MIN = 0.2
DNH_GRADIENT_SHARE_MAX = 0.4
DNH_GRADIENT_TARGET = 0.3
DNH_GRADIENT_RECOMMENDATION_RULE = (
    "keep_if_in_range_else_linear_scale_to_target_then_recompute_v1"
)
PILOT_STEPS = CANONICAL_LOSS_PILOT_STEPS
MEASURED_PROBE_STEPS = CANONICAL_MEASURED_PROBE_STEPS
G0_STEPS = 500
G0_BATCH_SIZE = 4
G0_THRESHOLD_EXCLUSIVE_DB = -6.0
PILOT_TIE_MARGIN_DB = 0.2
PILOT_SELECTION_RULE = (
    "alpha_specific_g0_dnh_precalibrated_measured_probe_"
    "recorded_val_worst_g4_gate_margin_0.2_db_v4"
)
MEASURED_PROBE_SELECTION_SCORE = (
    "measured_probe_recorded_val_worst_g4_gate_margin_db"
)
CANONICAL_RECORDED_VAL_MANIFEST = CANONICAL_RECORDED_MANIFEST
_SHA256 = "0123456789abcdef"

# canonical_pretrain의 campaign prerequisite는 이 evidence들로부터 나중에
# 발행되며, measured probe init/recorded manifest는 역할별 입력이다. 나머지
# artifact는 pilot/probe/canonical 100k가 한 byte도 다르면 안 되는 공통 학습 입력이다.
_ROLE_SPECIFIC_ARTIFACTS = frozenset(
    {"campaign_prerequisite", "init_checkpoint", "recorded_manifest"}
)


def snapshot_g0_determinism_environment() -> dict[str, Any]:
    """G0 모델을 실제로 계산하는 프로세스의 결정론 backend 상태를 캡처한다.

    config에 ``determinism_policy``가 적혀 있다는 사실은 실행 상태의 증거가 아니다.
    특히 ``diagnostic_overfit`` 역할은 일반 진단에서는 결정론 backend를 강제하지
    않으므로, 공식 G0 evidence 경로가 이 live 상태를 별도 결속해야 한다.
    """

    return {
        "schema_version": int(CANONICAL_DETERMINISM_POLICY["schema_version"]),
        "kind": G0_DETERMINISM_ENVIRONMENT_KIND,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_use_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def validate_g0_determinism_environment(
    value: object, *, label: str = "campaign G0 determinism environment"
) -> dict[str, Any]:
    """live/persisted G0 결정론 상태를 canonical 단일 정책과 exact 대조한다."""

    environment = _exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "cuda_available",
            "torch_use_deterministic_algorithms",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "cublas_workspace_config",
        },
        label=label,
    )
    policy = CANONICAL_DETERMINISM_POLICY
    if (
        type(environment["schema_version"]) is not int
        or environment["schema_version"] != int(policy["schema_version"])
        or environment["kind"] != G0_DETERMINISM_ENVIRONMENT_KIND
    ):
        raise ValueError(f"{label} schema/kind가 다릅니다")
    if type(environment["cuda_available"]) is not bool:
        raise ValueError(f"{label} cuda_available은 bool이어야 합니다")
    if (
        environment["torch_use_deterministic_algorithms"]
        is not policy["torch_use_deterministic_algorithms"]
        or environment["cudnn_benchmark"] is not policy["cudnn_benchmark"]
        or environment["cudnn_deterministic"]
        is not policy["cudnn_deterministic"]
    ):
        raise ValueError(f"{label}가 canonical 결정론 backend 상태가 아닙니다")
    workspace = environment["cublas_workspace_config"]
    allowed = set(policy["cublas_workspace_config_allowed"])
    if bool(environment["cuda_available"]) and workspace not in allowed:
        raise ValueError(
            f"{label} CUDA 실행의 CUBLAS_WORKSPACE_CONFIG가 승인값이 아닙니다: "
            f"{workspace!r}"
        )
    if not bool(environment["cuda_available"]) and workspace not in allowed | {None}:
        raise ValueError(f"{label} CUBLAS_WORKSPACE_CONFIG가 잘못됐습니다: {workspace!r}")
    return environment


def configure_g0_evidence_determinism() -> dict[str, Any]:
    """공식 G0 evidence 계산 전에 canonical 결정론 backend를 활성화한다.

    CUDA의 cuBLAS workspace 값은 CUDA context 생성 뒤 바꾸면 효력이 보장되지
    않는다. 따라서 호출자가 프로세스 시작 환경으로 승인값을 제공하지 않았으면
    여기서 값을 대신 채우지 않고 즉시 실패한다.
    """

    policy = CANONICAL_DETERMINISM_POLICY
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if torch.cuda.is_available() and workspace not in set(
        policy["cublas_workspace_config_allowed"]
    ):
        raise RuntimeError(
            "공식 G0 CUDA evidence에는 프로세스 시작 전 "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 또는 :16:8이 필요합니다"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return validate_g0_determinism_environment(
        snapshot_g0_determinism_environment(),
        label="campaign G0 live determinism environment",
    )


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


def _publish_g0_evidence(
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
    determinism_environment: dict[str, Any],
    evidence_kind: str,
) -> Path:
    """G0 model+fixed batch+live 결정론 상태를 지정된 kind로 봉인한다.

    수치 metric은 intentionally 저장하지 않는다. canonical validator는 final model과
    batch에서 다시 forward하여 G0를 판정한다. 실패 kind는 lambda 재추천 진단에만
    쓰며 canonical validator가 구조적으로 거부한다.
    """

    if evidence_kind not in {G0_RECEIPT_KIND, FAILED_G0_RECEIPT_KIND}:
        raise ValueError(f"지원하지 않는 G0 evidence kind입니다: {evidence_kind!r}")

    if set(batch) != {"x", "d"}:
        raise ValueError("G0 evidence batch는 정확히 x,d여야 합니다")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError("G0 evidence model_state가 비었습니다")
    # CLI는 학습 시작 전에 캡처한 값을 넘긴다. publisher를 직접 호출하더라도
    # 현재 live 상태가 canonical 정책이 아니면 evidence directory를 만들기 전에
    # 실패한다. 시작 상태를 true로 위조하고 마지막에 backend를 바꾸는 것도
    # current snapshot exact 대조가 막는다.
    started_environment = validate_g0_determinism_environment(
        copy.deepcopy(determinism_environment),
        label="campaign G0 start determinism environment",
    )
    final_environment = validate_g0_determinism_environment(
        snapshot_g0_determinism_environment(),
        label="campaign G0 publish determinism environment",
    )
    if final_environment != started_environment:
        raise ValueError(
            "campaign G0 학습 시작과 evidence 발행 시점의 결정론 backend 상태가 다릅니다"
        )

    def build(root: Path, staging: Path, target: Path) -> None:
        checkpoint_file = staging / "checkpoint.pt"
        batch_file = staging / "batch.pt"
        environment_file = staging / "environment.json"
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
            "kind": evidence_kind,
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
        _write_json_in_staging(environment_file, started_environment)
        receipt = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "kind": evidence_kind,
            "checkpoint": _reference_for_staged_file(
                root, checkpoint_file, target / checkpoint_file.name
            ),
            "batch": _reference_for_staged_file(root, batch_file, target / batch_file.name),
            "environment": _reference_for_staged_file(
                root, environment_file, target / environment_file.name
            ),
        }
        _write_json_in_staging(staging / "receipt.json", receipt)

    return _publish_evidence_directory(repo_root, output_dir, build=build) / "receipt.json"


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
    determinism_environment: dict[str, Any],
) -> Path:
    """NMSE < -6 dB를 통과한 G0의 campaign-eligible raw receipt를 발행한다."""

    return _publish_g0_evidence(
        repo_root=repo_root,
        output_dir=output_dir,
        cfg=cfg,
        model_state=model_state,
        batch=batch,
        steps=steps,
        mode=mode,
        primary_mode=primary_mode,
        require_nmse_db=require_nmse_db,
        nmse_only=nmse_only,
        disable_loss_terms=disable_loss_terms,
        determinism_environment=determinism_environment,
        evidence_kind=G0_RECEIPT_KIND,
    )


def publish_failed_g0_evidence(
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
    determinism_environment: dict[str, Any],
) -> Path:
    """실패한 G0를 lambda 진단 전용 kind로 봉인한다.

    이 receipt는 :func:`validate_g0_receipt`와 schema-v7 campaign ledger가 절대
    승인하지 않는다. 실패 checkpoint의 weight를 init으로 전이하는 경로도 제공하지
    않는다. 허용되는 유일한 소비자는 다음 fresh G0 contract를 위한 DNH output-y
    gradient 추천이다.
    """

    return _publish_g0_evidence(
        repo_root=repo_root,
        output_dir=output_dir,
        cfg=cfg,
        model_state=model_state,
        batch=batch,
        steps=steps,
        mode=mode,
        primary_mode=primary_mode,
        require_nmse_db=require_nmse_db,
        nmse_only=nmse_only,
        disable_loss_terms=disable_loss_terms,
        determinism_environment=determinism_environment,
        evidence_kind=FAILED_G0_RECEIPT_KIND,
    )


def _gradient_calibration_policy() -> dict[str, Any]:
    return {
        "gradient_domain": DNH_GRADIENT_DOMAIN,
        "gradient_norm": DNH_GRADIENT_NORM,
        "accepted_share_min": DNH_GRADIENT_SHARE_MIN,
        "accepted_share_max": DNH_GRADIENT_SHARE_MAX,
        "target_share": DNH_GRADIENT_TARGET,
        "recommendation_rule": DNH_GRADIENT_RECOMMENDATION_RULE,
    }


def publish_gradient_budget_evidence(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    checkpoint: str | Path,
    batch_artifact: str | Path,
) -> Path:
    """pilot checkpoint와 authoritative G0 batch를 결속한 gradient receipt를 발행한다.

    batch를 새로 직렬화하거나 복제하지 않는다. 모든 alpha의 G0가 공유한 exact
    artifact path/SHA를 직접 참조해야 post-pilot에서 유리한 batch를 바꿔 끼울 수 없다.
    receipt에는 사람이 계산한 share나 추천 λ를 넣지 않고 validator가 현재 λ와 필요 시
    선형 추천 λ의 gradient를 같은 raw bytes에서 다시 계산한다.
    """

    root = _root(repo_root)
    checkpoint_ref = snapshot_reference(root, checkpoint, label="gradient checkpoint")
    batch_ref = snapshot_reference(
        root,
        batch_artifact,
        label="authoritative common G0 gradient batch",
    )

    def build(_base: Path, staging: Path, _target: Path) -> None:
        receipt = {
            "schema_version": GRADIENT_RECEIPT_SCHEMA_VERSION,
            "kind": GRADIENT_RECEIPT_KIND,
            "checkpoint": checkpoint_ref,
            "batch": batch_ref,
            "calibration_policy": _gradient_calibration_policy(),
        }
        _write_json_in_staging(staging / "receipt.json", receipt)

    return _publish_evidence_directory(root, output_dir, build=build) / "receipt.json"


def publish_prepilot_gradient_evidence(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    g0_receipt: str | Path,
) -> Path:
    """approved-G0 raw checkpoint/batch를 alpha별 pre-pilot calibration에 결속한다.

    이 publisher는 G0의 성능 PASS를 주장하지 않는다. G0 receipt와 그 안의
    checkpoint/batch reference가 같은 bytes인지 봉인하며, campaign validator가
    G0 trusted NMSE와 DNH output-gradient share를 모두 다시 계산한다.
    """

    root = _root(repo_root)
    g0_ref = snapshot_reference(root, g0_receipt, label="pre-pilot G0 receipt")
    g0_snapshot = snapshot_from_reference(
        root, g0_ref, label="pre-pilot G0 receipt"
    )
    g0 = _exact_keys(
        _json_snapshot(g0_snapshot, label="pre-pilot G0 receipt"),
        {"schema_version", "kind", "checkpoint", "batch", "environment"},
        label="pre-pilot G0 receipt",
    )
    if (
        g0["schema_version"] != EVIDENCE_SCHEMA_VERSION
        or g0["kind"] != G0_RECEIPT_KIND
    ):
        raise ValueError("pre-pilot source가 canonical G0 receipt가 아닙니다")
    # publisher 시점에도 nested references를 실제로 열어 dangling/resealed G0가
    # calibration pathname을 차지하지 못하게 한다. 성능/정책은 validator가 재검산한다.
    snapshot_from_reference(root, g0["checkpoint"], label="pre-pilot G0 checkpoint")
    snapshot_from_reference(root, g0["batch"], label="pre-pilot G0 batch")

    def build(_base: Path, staging: Path, _target: Path) -> None:
        receipt = {
            "schema_version": GRADIENT_RECEIPT_SCHEMA_VERSION,
            "kind": PREPILOT_GRADIENT_RECEIPT_KIND,
            "g0_receipt": g0_ref,
            "checkpoint": g0["checkpoint"],
            "batch": g0["batch"],
            "calibration_policy": _gradient_calibration_policy(),
        }
        _write_json_in_staging(staging / "receipt.json", receipt)

    return _publish_evidence_directory(root, output_dir, build=build) / "receipt.json"


def publish_failed_g0_gradient_recommendation(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    failed_g0_receipt: str | Path,
    calibration: dict[str, Any],
) -> Path:
    """실패 G0의 다음 fresh-run lambda 추천을 진단 전용 receipt로 봉인한다.

    이 kind는 pre-pilot 승인 kind와 의도적으로 다르다. 따라서 계산 결과가
    0.2--0.4 범위여도 campaign prerequisite의 G0/gradient gate를 열 수 없다.
    저장된 숫자는 편의를 위한 claim일 뿐이며 원본 checkpoint/batch와 정책도 함께
    SHA 결속되어 재계산할 수 있다.
    """

    root = _root(repo_root)
    g0_ref = snapshot_reference(
        root, failed_g0_receipt, label="failed G0 diagnostic receipt"
    )
    g0_snapshot = snapshot_from_reference(
        root, g0_ref, label="failed G0 diagnostic receipt"
    )
    g0 = _exact_keys(
        _json_snapshot(g0_snapshot, label="failed G0 diagnostic receipt"),
        {"schema_version", "kind", "checkpoint", "batch", "environment"},
        label="failed G0 diagnostic receipt",
    )
    if (
        g0["schema_version"] != EVIDENCE_SCHEMA_VERSION
        or g0["kind"] != FAILED_G0_RECEIPT_KIND
    ):
        raise ValueError("lambda 추천 source가 failed-G0 diagnostic receipt가 아닙니다")
    snapshot_from_reference(
        root, g0["checkpoint"], label="failed G0 diagnostic checkpoint"
    )
    snapshot_from_reference(root, g0["batch"], label="failed G0 diagnostic batch")
    snapshot_from_reference(
        root, g0["environment"], label="failed G0 diagnostic environment"
    )
    expected_calibration_keys = {
        "gradient_domain",
        "gradient_norm",
        "accepted_share_min",
        "accepted_share_max",
        "target_share",
        "recommendation_rule",
        "approved",
        "current_lambda_dnh",
        "current_share",
        "recommended_lambda_dnh",
        "recommended_share",
        "current_budget",
        "recommended_budget",
    }
    claim = _exact_keys(
        copy.deepcopy(calibration),
        expected_calibration_keys,
        label="failed G0 gradient calibration claim",
    )
    for key, expected in _gradient_calibration_policy().items():
        if claim.get(key) != expected:
            raise ValueError(
                f"failed G0 gradient calibration claim.{key}가 정책과 다릅니다"
            )
    # allow_nan=False는 diagnostic receipt에도 NaN/Inf claim이 들어가는 것을 막는다.
    try:
        claim = json.loads(json.dumps(claim, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("failed G0 gradient calibration claim이 finite JSON이 아닙니다") from exc
    if not isinstance(claim.get("approved"), bool):
        raise ValueError("failed G0 gradient calibration claim.approved가 bool이 아닙니다")

    def build(_base: Path, staging: Path, _target: Path) -> None:
        receipt = {
            "schema_version": GRADIENT_RECEIPT_SCHEMA_VERSION,
            "kind": FAILED_G0_GRADIENT_RECEIPT_KIND,
            "failed_g0_receipt": g0_ref,
            "checkpoint": g0["checkpoint"],
            "batch": g0["batch"],
            "calibration_policy": _gradient_calibration_policy(),
            "calibration_claim": claim,
            "campaign_eligible": False,
            "required_next_action": "fresh_g0_from_scratch",
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
    cfg: dict[str, Any],
    *,
    root: Path,
    model: torch.nn.Module,
    device: torch.device,
    admission: CriterionAdmission | None = None,
) -> tuple[ANCLoss, int]:
    bundle = build_criterion_from_config(
        cfg,
        repo_root=root,
        limiter_limit=float(model.limit),
        device=device,
        admission=admission,
    )
    criterion = bundle.criterion
    criterion.eval()
    return criterion, int(cfg.get("loss_start_sample", -1))


def _recompute_metrics(
    cfg: dict[str, Any], model_state: object, batch: dict[str, torch.Tensor], *, root: Path, label: str
) -> dict[str, float]:
    admission = admit_criterion_config(
        cfg,
        repo_root=root,
        require_bound=True,
    )
    device = _device()
    model = _load_model_state(cfg, model_state, label=label, device=device)
    criterion, loss_start_sample = _build_criterion(
        cfg,
        root=root,
        model=model,
        device=device,
        admission=admission,
    )
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
    cfg: dict[str, Any],
    model_state: object,
    batch: dict[str, torch.Tensor],
    *,
    root: Path,
    label: str,
    lambda_dnh_override: float | None = None,
) -> dict[str, float]:
    criterion_cfg = cfg
    if lambda_dnh_override is not None:
        value = float(lambda_dnh_override)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} λ_dnh override가 finite 양수가 아닙니다")
        criterion_cfg = copy.deepcopy(cfg)
        loss = criterion_cfg.get("loss")
        if not isinstance(loss, dict):
            raise ValueError(f"{label} loss config가 없습니다")
        loss["lambda_dnh"] = value
    admission = admit_criterion_config(
        criterion_cfg,
        repo_root=root,
        require_bound=True,
    )
    device = _device()
    model = _load_model_state(
        criterion_cfg, model_state, label=label, device=device
    )
    criterion, loss_start_sample = _build_criterion(
        criterion_cfg,
        root=root,
        model=model,
        device=device,
        admission=admission,
    )
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


def calibrate_dnh_output_gradient(
    cfg: dict[str, Any],
    model_state: object,
    batch: dict[str, torch.Tensor],
    *,
    repo_root: str | Path,
    label: str = "DNH output-gradient calibration",
) -> dict[str, Any]:
    """현재 λ를 승인하거나, 범위 밖이면 다음 full rerun용 λ를 추천한다.

    이 함수가 재는 것은 ``‖λ·∂L_dnh/∂y‖ / ‖∂L_nmse/∂y‖``이며
    parameter-gradient가 아니다. 선택된 checkpoint와 고정 batch의 모델 출력 ``y``를
    고정하므로 DNH share는 λ에 선형이다. 범위 밖일 때만 중앙값 0.3으로 선형
    scaling한 뒤 **실제 ANCLoss를 다시 만들어 재계산**한다. 현재 값이 이미
    0.2–0.4이면 target 0.3에 맞춘다는 이유로 계약을 불필요하게 바꾸지 않는다.

    반환된 recommendation은 현재 campaign을 승인하지 않는다. ``approved``가 false면
    issuer는 차단되고, config를 추천값으로 바꾼 뒤 G0/pilot/probe 전체를 새 계약으로
    다시 실행해야 한다.
    """

    loss = cfg.get("loss")
    if not isinstance(loss, dict):
        raise ValueError(f"{label} checkpoint loss config가 없습니다")
    current_lambda = float(loss.get("lambda_dnh", float("nan")))
    if not math.isfinite(current_lambda) or current_lambda <= 0.0:
        raise ValueError(f"{label} checkpoint λ_dnh가 finite 양수가 아닙니다")

    root = _root(repo_root)
    current_budget = _recompute_gradient_budget(
        cfg,
        model_state,
        batch,
        root=root,
        label=f"{label} current",
    )
    current_share = float(current_budget.get("dnh", float("nan")))
    if not math.isfinite(current_share) or current_share <= 0.0:
        raise ValueError(
            f"{label} 현재 model-output y DNH share가 finite 양수가 아닙니다: "
            f"{current_share!r}. 힌지가 비활성인 출력에는 선형 λ 추천을 정의할 수 없습니다"
        )

    approved = DNH_GRADIENT_SHARE_MIN <= current_share <= DNH_GRADIENT_SHARE_MAX
    recommended_lambda = current_lambda
    recommended_budget = current_budget
    if not approved:
        recommended_lambda = current_lambda * DNH_GRADIENT_TARGET / current_share
        if not math.isfinite(recommended_lambda) or recommended_lambda <= 0.0:
            raise ValueError(f"{label} 선형 추천 λ_dnh가 finite 양수가 아닙니다")
        recommended_budget = _recompute_gradient_budget(
            cfg,
            model_state,
            batch,
            root=root,
            label=f"{label} recommended",
            lambda_dnh_override=recommended_lambda,
        )

    recommended_share = float(recommended_budget.get("dnh", float("nan")))
    if not math.isfinite(recommended_share) or not (
        DNH_GRADIENT_SHARE_MIN
        <= recommended_share
        <= DNH_GRADIENT_SHARE_MAX
    ):
        raise ValueError(
            f"{label} 추천 λ 실제 재계산 share가 승인 범위가 아닙니다: "
            f"lambda={recommended_lambda!r}, share={recommended_share!r}"
        )
    if not approved and not math.isclose(
        recommended_share,
        DNH_GRADIENT_TARGET,
        rel_tol=1.0e-4,
        abs_tol=1.0e-6,
    ):
        raise ValueError(
            f"{label} 선형 추천을 실제 재계산했지만 target share와 다릅니다: "
            f"target={DNH_GRADIENT_TARGET}, recomputed={recommended_share!r}"
        )

    return {
        "gradient_domain": DNH_GRADIENT_DOMAIN,
        "gradient_norm": DNH_GRADIENT_NORM,
        "accepted_share_min": DNH_GRADIENT_SHARE_MIN,
        "accepted_share_max": DNH_GRADIENT_SHARE_MAX,
        "target_share": DNH_GRADIENT_TARGET,
        "recommendation_rule": DNH_GRADIENT_RECOMMENDATION_RULE,
        "approved": approved,
        "current_lambda_dnh": current_lambda,
        "current_share": current_share,
        "recommended_lambda_dnh": recommended_lambda,
        "recommended_share": recommended_share,
        "current_budget": current_budget,
        "recommended_budget": recommended_budget,
    }


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


def _common_training_artifacts(artifacts: object, *, label: str) -> dict[str, dict]:
    if not isinstance(artifacts, dict):
        raise ValueError(f"{label} experiment contract artifact가 없습니다")
    common = {
        str(name): item
        for name, item in artifacts.items()
        if str(name) not in _ROLE_SPECIFIC_ARTIFACTS
    }
    if not common:
        raise ValueError(f"{label} 공통 학습 input artifact가 비었습니다")
    return common


def _validate_current_artifact(
    name: str, item: object, *, root: Path, label: str
) -> FileSnapshot:
    """embedded artifact identity와 현재 pathname bytes를 같은 snapshot으로 대조한다."""

    if not isinstance(item, dict) or set(item) != {
        "path",
        "exists",
        "size_bytes",
        "sha256",
    }:
        raise ValueError(f"{label} {name} artifact fingerprint가 불완전합니다")
    if item.get("exists") is not True:
        raise ValueError(f"{label} {name} artifact가 stamp 시점에 존재하지 않았습니다")
    path = repo_path(root, str(item.get("path", "")), label=f"{label} {name}")
    snapshot = snapshot_regular_file(path)
    if len(snapshot.content) != int(item.get("size_bytes", -1)):
        raise ValueError(f"{label} {name} size가 embedded contract와 다릅니다")
    if snapshot.sha256 != str(item.get("sha256", "")):
        raise ValueError(f"{label} {name} bytes가 embedded contract와 다릅니다")
    return snapshot


def _target_contract_and_artifacts(
    canonical_cfg: dict[str, Any], *, root: Path
) -> dict[str, Any]:
    contract = validate_embedded_experiment_contract(canonical_cfg)
    artifacts = _common_training_artifacts(
        contract.get("artifacts"), label="canonical"
    )
    required = {
        "primary_path",
        "secondary_path",
        "rir_bank",
        "bootstrap_receipt",
        "transfer_manifest",
        "source_manifest_generation",
        "recorded_holdout",
    }
    missing = required - set(artifacts)
    if missing:
        raise ValueError(
            "canonical experiment contract의 필수 학습 artifact가 없습니다: "
            f"{sorted(missing)}"
        )
    source_manifests = {
        name for name in artifacts if name.startswith("source_manifest:")
    }
    if not source_manifests:
        raise ValueError("canonical source manifest artifact가 비었습니다")
    for name, item in sorted(artifacts.items()):
        _validate_current_artifact(name, item, root=root, label="canonical")
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
    expected_artifacts = _common_training_artifacts(
        canonical_contract.get("artifacts"), label="canonical"
    )
    observed_artifacts = _common_training_artifacts(
        evidence_contract.get("artifacts"), label=label
    )
    if observed_artifacts != expected_artifacts:
        changed = sorted(
            name
            for name in set(expected_artifacts) | set(observed_artifacts)
            if observed_artifacts.get(name) != expected_artifacts.get(name)
        )
        raise ValueError(
            f"{label} 공통 학습 artifact가 canonical과 다릅니다: {changed}"
        )
    # embedded digest가 유효해도 stamp 이후 pathname bytes가 바뀌었을 수 있다.
    # 모든 public manifest/RIR/holdout/bootstrap/transfer/P/S를 지금 다시 연다.
    for name, item in sorted(observed_artifacts.items()):
        _validate_current_artifact(name, item, root=root, label=label)
    return evidence_contract


def _loss_candidate_identity(
    loss: object, *, label: str
) -> tuple[float, float, float]:
    """후보 identity=(alpha, frame, lambda_dnh)를 strict float로 유도한다."""

    if not isinstance(loss, dict):
        raise ValueError(f"{label} loss config가 없습니다")
    identity = (
        float(loss.get("nmse_cvar_alpha", float("nan"))),
        float(loss.get("lambda_frame", float("nan"))),
        float(loss.get("lambda_dnh", float("nan"))),
    )
    if not all(math.isfinite(item) for item in identity) or identity[2] <= 0.0:
        raise ValueError(f"{label} loss alpha/frame/lambda_dnh가 finite 양수가 아닙니다")
    pair = identity[:2]
    if pair not in {(float(a), float(b)) for a, b in CANONICAL_LOSS_GRID}:
        raise ValueError(f"{label} loss alpha×frame이 승인 grid가 아닙니다")
    return identity


def _validate_loss(
    evidence_cfg: dict[str, Any],
    canonical_cfg: dict[str, Any],
    *,
    label: str,
    allow_any_candidate: bool,
    expected_identity: tuple[float, float, float] | None,
) -> tuple[float, float, float]:
    observed = evidence_cfg.get("loss")
    expected = canonical_cfg.get("loss")
    if not isinstance(observed, dict) or not isinstance(expected, dict):
        raise ValueError(f"{label} loss config가 없습니다")
    if set(observed) != set(expected):
        raise ValueError(f"{label} loss key 집합이 canonical과 다릅니다")
    identity = _loss_candidate_identity(observed, label=label)
    for key, value in expected.items():
        if key in {"nmse_cvar_alpha", "lambda_dnh"} and (
            allow_any_candidate or expected_identity is not None
        ):
            continue
        if observed.get(key) != value:
            raise ValueError(f"{label} loss.{key}가 canonical과 다릅니다")
    if expected_identity is not None and identity != expected_identity:
        raise ValueError(
            f"{label} loss (alpha,frame,lambda_dnh)가 expected candidate와 다릅니다"
        )
    digest = str(evidence_cfg.get("loss_selection_sha256", ""))
    if digest != loss_selection_sha256(observed):
        raise ValueError(f"{label} loss_selection_sha256가 embedded loss와 다릅니다")
    return identity


def _validate_measured_probe_distribution_policy(
    evidence_cfg: dict[str, Any],
    canonical_cfg: dict[str, Any],
    *,
    label: str,
) -> None:
    """probe의 70:30 정책과 현재 recorded generation manifest를 함께 고정한다."""

    expected_manifest = canonical_recorded_manifest_for_data(
        canonical_cfg.get("data") or {}
    )
    for key, required in CANONICAL_MEASURED_PROBE_POLICY.items():
        expected = expected_manifest if key == "recorded_manifest" else required
        if evidence_cfg.get(key) != expected:
            raise ValueError(
                f"{label} {key}가 measured 70:30 probe policy와 다릅니다"
            )


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
    expected_identity: tuple[float, float, float] | None = None,
    allow_any_candidate: bool = False,
) -> tuple[dict[str, Any], tuple[float, float, float]]:
    # checkpoint가 스스로 주장하는 embedded contract를 먼저 검증한 뒤 role policy
    # 전체를 재실행한다. 일부 필드 수기 비교만으로는 optimizer/schedule/batch/data
    # augmentation을 바꾼 derivative가 ledger 후보가 될 수 있다.
    if embedded:
        validate_embedded_experiment_contract(evidence_cfg)
    validate_canonical_training_policy(copy.deepcopy(evidence_cfg))
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
    if role == "measured_probe":
        _validate_measured_probe_distribution_policy(
            evidence_cfg, canonical_cfg, label=label
        )
    identity = _validate_loss(
        evidence_cfg,
        canonical_cfg,
        label=label,
        allow_any_candidate=allow_any_candidate,
        expected_identity=expected_identity,
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
    return contract, identity


def validate_g0_receipt(
    receipt_reference: object,
    *,
    repo_root: str | Path,
    canonical_cfg: dict[str, Any],
    canonical_contract: dict[str, Any] | None = None,
    expected_identity: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """G0 receipt의 raw final model/batch로 trusted NMSE를 다시 계산한다."""

    root = _root(repo_root)
    contract = canonical_contract or _target_contract_and_artifacts(canonical_cfg, root=root)
    receipt_snapshot = snapshot_from_reference(root, receipt_reference, label="campaign G0 receipt")
    receipt = _exact_keys(
        _json_snapshot(receipt_snapshot, label="campaign G0 receipt"),
        {"schema_version", "kind", "checkpoint", "batch", "environment"},
        label="campaign G0 receipt",
    )
    if receipt["schema_version"] != EVIDENCE_SCHEMA_VERSION or receipt["kind"] != G0_RECEIPT_KIND:
        raise ValueError("campaign G0 receipt schema/kind가 다릅니다")
    checkpoint_snapshot = snapshot_from_reference(root, receipt["checkpoint"], label="campaign G0 checkpoint")
    batch_snapshot = snapshot_from_reference(root, receipt["batch"], label="campaign G0 batch")
    environment_snapshot = snapshot_from_reference(
        root, receipt["environment"], label="campaign G0 determinism environment"
    )
    environment = validate_g0_determinism_environment(
        _json_snapshot(
            environment_snapshot, label="campaign G0 determinism environment"
        ),
        label="campaign G0 persisted determinism environment",
    )
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
    _, identity = _validate_derivative_cfg(
        cfg,
        canonical_cfg=canonical_cfg,
        canonical_contract=contract,
        root=root,
        label="campaign G0",
        role="diagnostic_overfit",
        primary_mode="secondary_surrogate",
        embedded=False,
        expected_identity=expected_identity,
        allow_any_candidate=expected_identity is None,
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
        "environment": environment_snapshot,
        "determinism_environment": environment,
        "nmse_trusted_db": nmse,
        "metrics": metrics,
        "identity": identity,
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

    checkpoint_state = _torch_snapshot(checkpoint_snapshot, label=f"{label} checkpoint")
    if not isinstance(checkpoint_state, dict) or not isinstance(
        checkpoint_state.get("cfg"), dict
    ):
        raise ValueError(f"{label} checkpoint에 resolved cfg가 없습니다")
    checkpoint_cfg = checkpoint_state["cfg"]

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
        # campaign 선택도 final G4와 같은 raw population/family/octave/
        # strict-subband 감사기를 통과한다. 20k pilot은 surrogate 물리이므로
        # 실제 덕트 PASS를 주장할 수는 없지만, 추린 session 배열이나
        # 1000–1600 Hz 부대역을 생략한 수기 NPZ로 winner를 만들 수도 없다.
        persisted = validate_persisted_g4_metrics(
            data,
            expected_split="val",
            manifest_bytes=manifest_snapshot.content,
            manifest_path=manifest_snapshot.path,
            checkpoint_cfg=checkpoint_cfg,
            canonical=True,
            surrogate_diagnostic=expected_surrogate,
        )
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
    if not math.isclose(
        float(persisted["trusted"]["worst10_mean_db"]),
        trusted["worst10_mean_db"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):  # pragma: no cover - 동일 raw 계산식 불변식
        raise RuntimeError("campaign/central G4 trusted worst10 재계산이 갈라졌니다")

    # 절대 목표는 저역 평균 하나가 아니라 모든 source family와
    # 1000–1600 Hz를 포함한 모든 strict 부대역, fullband,
    # out-of-band do-no-harm을 동시에 지키는 것이다. 그러므로 candidate
    # score도 각 G4 경계(0 dB, do-no-harm만 허용 증폭 1 dB)로부터의
    # 최악 margin을 쓴다. 낮을수록 좋고 0 이상은 어느 gate인가 실패다.
    strict = persisted["strict_subband"]
    strict_flags = strict["flags"]
    if not (
        strict_flags["g4_trusted_subband_coverage_pass"]
        and strict_flags["g4_trusted_subband_power_pass"]
    ):
        raise ValueError(
            f"{label} candidate selection에 필요한 family×strict-subband "
            "target-energy/group coverage가 부족합니다"
        )
    score_components = {
        "trusted_mean": float(persisted["trusted"]["mean_db"]),
        "trusted_worst10": float(persisted["trusted"]["worst10_mean_db"]),
        "fullband_mean": float(persisted["fullband"]["mean_db"]),
        "worst_family_mean": float(np.max(persisted["source_trusted_mean_db"])),
        "worst_family_worst10": float(
            np.max(persisted["source_trusted_worst10_db"])
        ),
        "worst_family_ci_hi": float(np.max(persisted["source_ci_hi_db"])),
        "worst_strict_subband_mean": float(np.max(strict["mean_db"])),
        "worst_strict_subband_worst10": float(np.max(strict["worst10_db"])),
        "worst_strict_subband_ci_hi": float(np.max(strict["ci_hi_db"])),
        # attenuation는 +가 감쇠, -가 증폭이다. 허용 증폭 threshold를
        # 빼서 다른 0-dB gate와 같은 방향/경계로 바꾼다.
        "do_no_harm": -float(persisted["worst_octave_db"])
        - float(persisted["threshold_db"]),
    }
    if not all(math.isfinite(value) for value in score_components.values()):
        raise ValueError(f"{label} candidate G4 selection margin에 NaN/Inf가 있습니다")
    return max(score_components.values())


def _validate_pilot_checkpoint_pair(
    candidate: object,
    *,
    repo_root: Path,
    canonical_cfg: dict[str, Any],
    canonical_contract: dict[str, Any],
    expected_role: str,
    expected_primary_mode: str,
    expected_steps: int,
    expected_identity: tuple[float, float, float] | None,
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
    expected_manifest_value = canonical_recorded_manifest_for_data(
        canonical_cfg.get("data") or {}
    )
    expected_manifest = repo_path(
        repo_root,
        expected_manifest_value,
        label=f"{label} canonical recorded manifest",
    )
    if manifest_snapshot.path != expected_manifest:
        raise ValueError(
            f"{label} manifest는 current recorded generation의 canonical manifest여야 "
            f"합니다: {expected_manifest_value}"
        )
    best = _load_checkpoint(best_snapshot, label=f"{label} best checkpoint")
    last = _load_checkpoint(last_snapshot, label=f"{label} last checkpoint")
    validate_world1_cuda_rng(best, label=f"{label} best checkpoint")
    validate_world1_cuda_rng(last, label=f"{label} last checkpoint")
    best_cfg = best["cfg"]
    last_cfg = last["cfg"]
    best_contract, identity = _validate_derivative_cfg(
        best_cfg,
        canonical_cfg=canonical_cfg,
        canonical_contract=canonical_contract,
        root=repo_root,
        label=f"{label} best checkpoint",
        role=expected_role,
        primary_mode=expected_primary_mode,
        embedded=True,
        expected_identity=expected_identity,
        allow_any_candidate=expected_identity is None,
    )
    last_contract, last_identity = _validate_derivative_cfg(
        last_cfg,
        canonical_cfg=canonical_cfg,
        canonical_contract=canonical_contract,
        root=repo_root,
        label=f"{label} last checkpoint",
        role=expected_role,
        primary_mode=expected_primary_mode,
        embedded=True,
        expected_identity=identity,
        allow_any_candidate=False,
    )
    if (
        best_contract["sha256"] != last_contract["sha256"]
        or identity != last_identity
    ):
        raise ValueError(
            f"{label} best/last experiment contract 또는 loss candidate identity가 다릅니다"
        )
    if expected_role == "measured_probe":
        recorded_item = (best_contract.get("artifacts") or {}).get(
            "recorded_manifest"
        )
        recorded_snapshot = _validate_current_artifact(
            "recorded_manifest",
            recorded_item,
            root=repo_root,
            label=label,
        )
        if (
            recorded_snapshot.path != manifest_snapshot.path
            or recorded_snapshot.sha256 != manifest_snapshot.sha256
        ):
            raise ValueError(
                f"{label} training recorded_manifest가 recorded-val manifest와 다릅니다"
            )
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
        "identity": identity,
        "score_db": score,
        "contract_sha256": str(best_contract["sha256"]),
    }


def validate_loss_pilot_candidate(
    candidate: object,
    *,
    repo_root: str | Path,
    canonical_cfg: dict[str, Any],
    canonical_contract: dict[str, Any] | None = None,
    expected_identity: tuple[float, float, float] | None = None,
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
        expected_identity=expected_identity,
        label=label,
    )
    if int(result["best"]["cfg"].get("run_until_step", -1)) != PILOT_STEPS:
        # cfg is checked above; this branch is intentionally defensive if a future
        # checkpoint schema adds a top-level operational field.
        raise ValueError(f"{label} best checkpoint run_until_step이 승인값과 다릅니다")
    return result


def select_loss_pilot(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """20k pilot별 5k measured-probe recorded-val score로 winner를 유도한다.

    이 함수에 넘기는 ``score_db``는 surrogate 20k pilot 점수가 아니라,
    각 pilot best.pt를 init으로 사용한 5k measured 70:30 probe의
    recorded-val 최악 G4 gate margin이어야 한다. 이 점수는 trusted/fullband,
    family, strict 부대역, do-no-harm 중 경계에 가장 가까운 항목이다.
    ``selection_score_source``를
    강제해 pilot 점수를 실수로 다시 선택 근거로 쓰는 경로를 닫는다.

    measured probe의 0.7/1.0 차이가 0.2 dB 이내면 0.85의 20k+5k
    raw chain이 **반드시** 있어야 한다.
    이어서 최저점과 0.2 dB 이내인 후보가 여러 개면 사전 규칙대로 0.7을 택한다.
    alpha=1.0의 '불안정' 분기는 현재 immutable raw failure receipt schema가 없으므로
    의도적으로 자동 승격하지 않는다. malformed/non-finite artifact는 evidence가 아니라
    failure이며, 새 schema를 정의할 때까지 canonical pretrain을 막는다.
    """

    if not isinstance(candidates, list) or not candidates:
        raise ValueError("loss pilot candidate가 비었습니다")
    by_alpha: dict[float, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("loss pilot candidate가 mapping이 아닙니다")
        if candidate.get("selection_score_source") != MEASURED_PROBE_SELECTION_SCORE:
            raise ValueError(
                "loss selection score는 후보별 measured-probe recorded-val에서 "
                "유도되어야 합니다"
            )
        identity = tuple(candidate.get("identity", ()))
        if len(identity) != 3:
            raise ValueError(
                "loss pilot derived (alpha,frame,lambda_dnh) identity가 없습니다"
            )
        identity = (
            float(identity[0]),
            float(identity[1]),
            float(identity[2]),
        )
        alpha, frame, lambda_dnh = identity
        if (
            not all(math.isfinite(item) for item in identity)
            or lambda_dnh <= 0.0
            or (alpha, frame)
            not in {(float(a), float(b)) for a, b in CANONICAL_LOSS_GRID}
        ):
            raise ValueError("loss pilot candidate identity가 승인 grid/finite λ가 아닙니다")
        score = float(candidate.get("score_db", float("nan")))
        if alpha in by_alpha or not math.isfinite(score):
            raise ValueError("loss pilot alpha 중복 또는 non-finite recorded-val score")
        by_alpha[alpha] = candidate
    base = {0.7, 1.0}
    optional = 0.85
    alphas = set(by_alpha)
    if not base.issubset(alphas) or alphas - (base | {optional}):
        raise ValueError("loss pilot candidate 집합이 승인 alpha grid와 다릅니다")
    base_gap = abs(
        float(by_alpha[0.7]["score_db"])
        - float(by_alpha[1.0]["score_db"])
    )
    needs_alpha_085 = base_gap <= PILOT_TIE_MARGIN_DB
    if needs_alpha_085 != (optional in alphas):
        raise ValueError(
            "loss pilot 0.7/1.0 margin과 alpha=0.85 candidate 집합이 다릅니다: "
            f"gap={base_gap:.6f} dB, margin={PILOT_TIE_MARGIN_DB:.1f} dB"
        )
    best_score = min(float(row["score_db"]) for row in by_alpha.values())
    tied_alphas = {
        alpha
        for alpha, row in by_alpha.items()
        if float(row["score_db"]) <= best_score + PILOT_TIE_MARGIN_DB
    }
    winner_alpha = 0.7 if 0.7 in tied_alphas else min(
        tied_alphas,
        key=lambda alpha: (float(by_alpha[alpha]["score_db"]), alpha),
    )
    winner_identity = tuple(by_alpha[winner_alpha]["identity"])
    return {
        "winner_identity": winner_identity,
        "winner": by_alpha[winner_alpha],
        "base_gap_db": base_gap,
        "used_alpha_085": optional in alphas,
        "candidates": [by_alpha[alpha] for alpha in sorted(by_alpha)],
    }


def _validate_gradient_calibration_policy(value: object, *, label: str) -> None:
    policy = _exact_keys(
        value,
        {
            "gradient_domain",
            "gradient_norm",
            "accepted_share_min",
            "accepted_share_max",
            "target_share",
            "recommendation_rule",
        },
        label=label,
    )
    if policy != _gradient_calibration_policy():
        raise ValueError(f"{label}가 승인 계약과 다릅니다")


def _validate_calibration_claim(
    claim: object, expected: dict[str, Any], *, label: str
) -> None:
    """저장 claim이 raw 재계산과 수치적으로 같은지 엄격 비교한다."""

    observed = _exact_keys(claim, set(expected), label=label)
    for key, expected_value in expected.items():
        observed_value = observed[key]
        if isinstance(expected_value, dict):
            expected_budget = expected_value
            observed_budget = _exact_keys(
                observed_value, set(expected_budget), label=f"{label}.{key}"
            )
            for budget_key, budget_value in expected_budget.items():
                value = observed_budget[budget_key]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{label}.{key}.{budget_key}가 수치가 아닙니다")
                if not math.isfinite(float(value)) or not math.isclose(
                    float(value),
                    float(budget_value),
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-9,
                ):
                    raise ValueError(
                        f"{label}.{key}.{budget_key}가 raw 재계산과 다릅니다"
                    )
        elif isinstance(expected_value, bool):
            if type(observed_value) is not bool or observed_value is not expected_value:
                raise ValueError(f"{label}.{key}가 raw 재계산과 다릅니다")
        elif isinstance(expected_value, (int, float)):
            if isinstance(observed_value, bool) or not isinstance(
                observed_value, (int, float)
            ):
                raise ValueError(f"{label}.{key}가 수치가 아닙니다")
            if not math.isfinite(float(observed_value)) or not math.isclose(
                float(observed_value),
                float(expected_value),
                rel_tol=1.0e-6,
                abs_tol=1.0e-9,
            ):
                raise ValueError(f"{label}.{key}가 raw 재계산과 다릅니다")
        elif observed_value != expected_value:
            raise ValueError(f"{label}.{key}가 raw 재계산과 다릅니다")


def validate_failed_g0_gradient_recommendation(
    receipt_reference: object,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """failed-G0 recommendation receipt를 재계산하되 campaign 자격은 반환하지 않는다."""

    root = _root(repo_root)
    receipt_snapshot = snapshot_from_reference(
        root, receipt_reference, label="failed G0 gradient recommendation receipt"
    )
    receipt = _exact_keys(
        _json_snapshot(
            receipt_snapshot, label="failed G0 gradient recommendation receipt"
        ),
        {
            "schema_version",
            "kind",
            "failed_g0_receipt",
            "checkpoint",
            "batch",
            "calibration_policy",
            "calibration_claim",
            "campaign_eligible",
            "required_next_action",
        },
        label="failed G0 gradient recommendation receipt",
    )
    if (
        receipt["schema_version"] != GRADIENT_RECEIPT_SCHEMA_VERSION
        or receipt["kind"] != FAILED_G0_GRADIENT_RECEIPT_KIND
        or receipt["campaign_eligible"] is not False
        or receipt["required_next_action"] != "fresh_g0_from_scratch"
    ):
        raise ValueError("failed G0 gradient recommendation의 진단 전용 계약이 다릅니다")
    _validate_gradient_calibration_policy(
        receipt["calibration_policy"],
        label="failed G0 gradient recommendation policy",
    )
    g0_snapshot = snapshot_from_reference(
        root, receipt["failed_g0_receipt"], label="failed G0 diagnostic receipt"
    )
    g0 = _exact_keys(
        _json_snapshot(g0_snapshot, label="failed G0 diagnostic receipt"),
        {"schema_version", "kind", "checkpoint", "batch", "environment"},
        label="failed G0 diagnostic receipt",
    )
    if (
        g0["schema_version"] != EVIDENCE_SCHEMA_VERSION
        or g0["kind"] != FAILED_G0_RECEIPT_KIND
    ):
        raise ValueError("recommendation source가 failed G0 diagnostic kind가 아닙니다")
    checkpoint_snapshot = snapshot_from_reference(
        root, receipt["checkpoint"], label="failed G0 recommendation checkpoint"
    )
    batch_snapshot = snapshot_from_reference(
        root, receipt["batch"], label="failed G0 recommendation batch"
    )
    nested_checkpoint = snapshot_from_reference(
        root, g0["checkpoint"], label="failed G0 nested checkpoint"
    )
    nested_batch = snapshot_from_reference(
        root, g0["batch"], label="failed G0 nested batch"
    )
    if (
        checkpoint_snapshot.path != nested_checkpoint.path
        or checkpoint_snapshot.sha256 != nested_checkpoint.sha256
        or batch_snapshot.path != nested_batch.path
        or batch_snapshot.sha256 != nested_batch.sha256
    ):
        raise ValueError("failed G0 recommendation raw checkpoint/batch가 source와 다릅니다")
    environment_snapshot = snapshot_from_reference(
        root, g0["environment"], label="failed G0 diagnostic environment"
    )
    validate_g0_determinism_environment(
        _json_snapshot(environment_snapshot, label="failed G0 diagnostic environment"),
        label="failed G0 diagnostic persisted environment",
    )
    raw = _exact_keys(
        _torch_snapshot(checkpoint_snapshot, label="failed G0 diagnostic checkpoint"),
        {"schema_version", "kind", "model", "cfg", "step", "protocol"},
        label="failed G0 diagnostic checkpoint",
    )
    if (
        raw["schema_version"] != EVIDENCE_SCHEMA_VERSION
        or raw["kind"] != FAILED_G0_RECEIPT_KIND
    ):
        raise ValueError("failed G0 diagnostic checkpoint schema/kind가 다릅니다")
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
        label="failed G0 diagnostic protocol",
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
        raise ValueError("failed G0 diagnostic protocol이 공식 G0 control과 다릅니다")
    cfg = raw["cfg"]
    if not isinstance(cfg, dict):
        raise ValueError("failed G0 diagnostic cfg가 mapping이 아닙니다")
    validate_canonical_training_policy(copy.deepcopy(cfg))
    if (
        str(cfg.get("experiment_role", "")) != "diagnostic_overfit"
        or cfg.get("init_eligible") is not False
        or cfg.get("contract_run_dir") is not False
    ):
        raise ValueError("failed G0 diagnostic cfg 역할/eligibility가 다릅니다")
    identity = _loss_candidate_identity(cfg.get("loss"), label="failed G0 diagnostic")
    _validate_timing(cfg, root=root, label="failed G0 diagnostic")
    batch = _load_batch(
        batch_snapshot,
        label="failed G0 diagnostic",
        exact_batch_size=G0_BATCH_SIZE,
    )
    metrics = _recompute_metrics(
        cfg, raw["model"], batch, root=root, label="failed G0 diagnostic"
    )
    nmse = float(metrics.get("nmse_trusted_db", float("nan")))
    if not math.isfinite(nmse) or nmse < G0_THRESHOLD_EXCLUSIVE_DB:
        raise ValueError(
            "failed G0 diagnostic raw model/batch가 실제로는 G0 실패가 아닙니다: "
            f"nmse={nmse!r} dB"
        )
    calibration = calibrate_dnh_output_gradient(
        cfg,
        raw["model"],
        batch,
        repo_root=root,
        label="failed G0 diagnostic recommendation",
    )
    _validate_calibration_claim(
        receipt["calibration_claim"],
        calibration,
        label="failed G0 gradient calibration claim",
    )
    return {
        "receipt": receipt_snapshot,
        "failed_g0_receipt": g0_snapshot,
        "checkpoint": checkpoint_snapshot,
        "batch": batch_snapshot,
        "nmse_trusted_db": nmse,
        "identity": identity,
        "calibration": calibration,
        "campaign_eligible": False,
    }


def validate_prepilot_gradient_receipt(
    receipt_reference: object,
    *,
    repo_root: str | Path,
    canonical_cfg: dict[str, Any],
    g0_receipt_reference: object,
    expected_identity: tuple[float, float, float],
    canonical_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """alpha별 approved G0에서 pilot 전 현재 λ의 output-gradient를 승인한다."""

    root = _root(repo_root)
    contract = canonical_contract or _target_contract_and_artifacts(
        canonical_cfg, root=root
    )
    receipt_snapshot = snapshot_from_reference(
        root, receipt_reference, label="pre-pilot gradient receipt"
    )
    receipt = _exact_keys(
        _json_snapshot(receipt_snapshot, label="pre-pilot gradient receipt"),
        {
            "schema_version",
            "kind",
            "g0_receipt",
            "checkpoint",
            "batch",
            "calibration_policy",
        },
        label="pre-pilot gradient receipt",
    )
    if (
        receipt["schema_version"] != GRADIENT_RECEIPT_SCHEMA_VERSION
        or receipt["kind"] != PREPILOT_GRADIENT_RECEIPT_KIND
    ):
        raise ValueError("pre-pilot gradient receipt schema/kind가 다릅니다")
    _validate_gradient_calibration_policy(
        receipt["calibration_policy"], label="pre-pilot gradient calibration policy"
    )
    expected_g0 = snapshot_from_reference(
        root, g0_receipt_reference, label="candidate G0 receipt"
    )
    embedded_g0 = snapshot_from_reference(
        root, receipt["g0_receipt"], label="pre-pilot embedded G0 receipt"
    )
    if (
        embedded_g0.path != expected_g0.path
        or embedded_g0.sha256 != expected_g0.sha256
    ):
        raise ValueError("pre-pilot gradient receipt가 candidate G0 receipt와 다릅니다")

    g0 = validate_g0_receipt(
        receipt["g0_receipt"],
        repo_root=root,
        canonical_cfg=canonical_cfg,
        canonical_contract=contract,
        expected_identity=expected_identity,
    )
    checkpoint_snapshot = snapshot_from_reference(
        root, receipt["checkpoint"], label="pre-pilot gradient checkpoint"
    )
    batch_snapshot = snapshot_from_reference(
        root, receipt["batch"], label="pre-pilot gradient batch"
    )
    if (
        checkpoint_snapshot.path != g0["checkpoint"].path
        or checkpoint_snapshot.sha256 != g0["checkpoint"].sha256
        or batch_snapshot.path != g0["batch"].path
        or batch_snapshot.sha256 != g0["batch"].sha256
    ):
        raise ValueError("pre-pilot gradient raw checkpoint/batch가 G0와 다릅니다")

    raw = _exact_keys(
        _torch_snapshot(checkpoint_snapshot, label="pre-pilot gradient checkpoint"),
        {"schema_version", "kind", "model", "cfg", "step", "protocol"},
        label="pre-pilot gradient checkpoint",
    )
    batch = _load_batch(
        batch_snapshot,
        label="pre-pilot gradient",
        exact_batch_size=G0_BATCH_SIZE,
    )
    calibration = calibrate_dnh_output_gradient(
        raw["cfg"],
        raw["model"],
        batch,
        repo_root=root,
        label="pre-pilot gradient",
    )
    if calibration["approved"] is not True:
        raise ValueError(
            "alpha별 pre-pilot strict-S λ_dnh model-output y gradient share가 "
            f"승인 범위 {DNH_GRADIENT_SHARE_MIN:.1f}–"
            f"{DNH_GRADIENT_SHARE_MAX:.1f}가 아닙니다: "
            f"identity={expected_identity!r}, "
            f"share={calibration['current_share']!r}. 해당 alpha의 G0부터 "
            f"재실행할 추천 λ={calibration['recommended_lambda_dnh']!r}, "
            f"실제 재계산 share={calibration['recommended_share']!r}"
        )
    return {
        "receipt": receipt_snapshot,
        "g0": g0,
        "checkpoint": checkpoint_snapshot,
        "batch": batch_snapshot,
        "identity": expected_identity,
        "gradient_share": float(calibration["current_share"]),
        "calibration": calibration,
    }


def validate_gradient_budget_receipt(
    receipt_reference: object,
    *,
    repo_root: str | Path,
    canonical_cfg: dict[str, Any],
    expected_checkpoint_sha256: str,
    expected_identity: tuple[float, float, float],
    expected_batch_path: str | Path,
    expected_batch_sha256: str,
    canonical_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """selected pilot raw checkpoint/batch에서 DNH output-gradient를 재계산한다.

    승인 대상은 checkpoint에 이미 박힌 **현재 λ**다. 현재 share가 0.2–0.4이면
    그대로 유지하고, 범위 밖이면 재현 가능한 다음-run 추천값까지 실제 재계산하지만
    이 receipt로 현재 campaign을 승인하지 않는다.
    """

    root = _root(repo_root)
    contract = canonical_contract or _target_contract_and_artifacts(canonical_cfg, root=root)
    receipt_snapshot = snapshot_from_reference(root, receipt_reference, label="gradient budget receipt")
    receipt = _exact_keys(
        _json_snapshot(receipt_snapshot, label="gradient budget receipt"),
        {
            "schema_version",
            "kind",
            "checkpoint",
            "batch",
            "calibration_policy",
        },
        label="gradient budget receipt",
    )
    if (
        receipt["schema_version"] != GRADIENT_RECEIPT_SCHEMA_VERSION
        or receipt["kind"] != GRADIENT_RECEIPT_KIND
    ):
        raise ValueError("gradient budget receipt schema/kind가 다릅니다")
    _validate_gradient_calibration_policy(
        receipt["calibration_policy"], label="gradient budget calibration policy"
    )
    checkpoint_snapshot = snapshot_from_reference(root, receipt["checkpoint"], label="gradient budget checkpoint")
    if checkpoint_snapshot.sha256 != _sha(expected_checkpoint_sha256, label="gradient expected checkpoint SHA"):
        raise ValueError("gradient budget checkpoint가 selected loss pilot winner와 다릅니다")
    batch_snapshot = snapshot_from_reference(root, receipt["batch"], label="gradient budget batch")
    authoritative_batch_path = repo_path(
        root,
        expected_batch_path,
        label="authoritative common G0 batch",
    )
    authoritative_batch_sha256 = _sha(
        expected_batch_sha256,
        label="authoritative common G0 batch SHA",
    )
    if (
        batch_snapshot.path != authoritative_batch_path
        or batch_snapshot.sha256 != authoritative_batch_sha256
    ):
        raise ValueError(
            "selected-20k gradient receipt batch가 모든 candidate가 공유한 "
            "authoritative G0 fixed batch path/SHA와 다릅니다"
        )
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
        expected_identity=expected_identity,
    )
    _finite_checkpoint_model(raw, cfg=cfg, label="gradient budget checkpoint")
    batch = _load_batch(batch_snapshot, label="gradient budget", exact_batch_size=G0_BATCH_SIZE)
    calibration = calibrate_dnh_output_gradient(
        cfg,
        raw["model"],
        batch,
        repo_root=root,
        label="gradient budget",
    )
    share = float(calibration["current_share"])
    if calibration["approved"] is not True:
        raise ValueError(
            "strict-S λ_dnh model-output y gradient share가 승인 범위 "
            f"{DNH_GRADIENT_SHARE_MIN:.1f}–{DNH_GRADIENT_SHARE_MAX:.1f}가 "
            f"아닙니다: current_lambda={calibration['current_lambda_dnh']!r}, "
            f"share={share!r}. 다음 full G0/pilot/probe 재실행 추천 λ="
            f"{calibration['recommended_lambda_dnh']!r}, 실제 재계산 share="
            f"{calibration['recommended_share']!r}"
        )
    return {
        "receipt": receipt_snapshot,
        "checkpoint": checkpoint_snapshot,
        "batch": batch_snapshot,
        "gradient_share": share,
        "budget": calibration["current_budget"],
        "calibration": calibration,
    }


def validate_measured_probe(
    probe: object,
    *,
    repo_root: str | Path,
    canonical_cfg: dict[str, Any],
    expected_identity: tuple[float, float, float],
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
        raise ValueError("measured probe init checkpoint가 corresponding 20k pilot best와 다릅니다")
    pair_result = _validate_pilot_checkpoint_pair(
        {key: row[key] for key in ("best_checkpoint", "last_checkpoint", "metrics", "manifest")},
        repo_root=root,
        canonical_cfg=canonical_cfg,
        canonical_contract=contract,
        expected_role="measured_probe",
        expected_primary_mode="measured",
        expected_steps=MEASURED_PROBE_STEPS,
        expected_identity=expected_identity,
        label="measured probe",
    )
    probe_contract = validate_embedded_experiment_contract(
        pair_result["best"]["cfg"]
    )
    embedded_init = _validate_current_artifact(
        "init_checkpoint",
        (probe_contract.get("artifacts") or {}).get("init_checkpoint"),
        root=root,
        label="measured probe",
    )
    if (
        embedded_init.path != init_snapshot.path
        or embedded_init.sha256 != init_snapshot.sha256
    ):
        raise ValueError(
            "measured probe embedded init artifact가 selected pilot snapshot과 다릅니다"
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
    "G0_DETERMINISM_ENVIRONMENT_KIND",
    "G0_RECEIPT_KIND",
    "G0_STEPS",
    "G0_THRESHOLD_EXCLUSIVE_DB",
    "DNH_GRADIENT_DOMAIN",
    "DNH_GRADIENT_NORM",
    "DNH_GRADIENT_RECOMMENDATION_RULE",
    "DNH_GRADIENT_SHARE_MAX",
    "DNH_GRADIENT_SHARE_MIN",
    "DNH_GRADIENT_TARGET",
    "FAILED_G0_GRADIENT_RECEIPT_KIND",
    "FAILED_G0_RECEIPT_KIND",
    "GRADIENT_RECEIPT_KIND",
    "GRADIENT_RECEIPT_SCHEMA_VERSION",
    "MEASURED_PROBE_STEPS",
    "MEASURED_PROBE_SELECTION_SCORE",
    "PILOT_STEPS",
    "PILOT_SELECTION_RULE",
    "PILOT_TIE_MARGIN_DB",
    "PREPILOT_GRADIENT_RECEIPT_KIND",
    "calibrate_dnh_output_gradient",
    "configure_g0_evidence_determinism",
    "make_campaign_evidence_reference",
    "publish_failed_g0_evidence",
    "publish_failed_g0_gradient_recommendation",
    "publish_g0_evidence",
    "publish_gradient_budget_evidence",
    "publish_prepilot_gradient_evidence",
    "repo_path",
    "select_loss_pilot",
    "snapshot_from_reference",
    "snapshot_reference",
    "snapshot_g0_determinism_environment",
    "validate_g0_receipt",
    "validate_canonical_evidence_target",
    "validate_failed_g0_gradient_recommendation",
    "validate_gradient_budget_receipt",
    "validate_g0_determinism_environment",
    "validate_loss_pilot_candidate",
    "validate_measured_probe",
    "validate_prepilot_gradient_receipt",
]
