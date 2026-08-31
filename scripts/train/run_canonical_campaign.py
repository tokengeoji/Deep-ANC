#!/usr/bin/env python3
"""Elice canonical Stage-1 campaign을 한 번에 **한 단계만** 진행한다.

이 진입점은 임의 shell command나 legacy queue를 받지 않는다. 외부에서 SHA-256으로
봉인한 JSON contract를 읽고, 현행 ``bootstrap_all.sh``와 ``scripts/train`` /
``scripts/eval`` canonical CLI만 조합한다. 모든 ``--execute-next`` 직전에는 다음
단계와 exact argv를 담은 상태 JSON을 저장소 밖에 원자 저장한다.

적응형 경계는 의도적으로 자동 결정하지 않는다.

* G0 또는 pre-pilot gradient가 실패하면 추천 lambda를 기존 contract에 몰래
  대입하지 않고 ``ADAPTIVE_LAMBDA_REQUIRED``로 멈춘다.
* alpha 0.7/1.0 measured-probe 차이가 0.2 dB 이내면 alpha 0.85 candidate를 포함한
  새 외부 contract가 올 때까지 멈춘다.
* 중단된 pilot/probe/100k/50k는 checkpoint 경로와 외부 SHA를 둘 다 명시하지
  않으면 재개하지 않는다.

Contract schema v2 (경로는 저장소 밖 regular file이어야 한다)::

  {
    "schema_version": 2,
    "expected_commit": "<40 hex>",
    "expected_holdout_sha256": "<64 hex>",
    "expected_transfer_manifest_sha256": "<64 hex>",
    "campaign": {"seed": 20260803, "second_seed": null},
    "bootstrap": {
      "raw_hash_workers": 8,
      "cublas_workspace_config": ":4096:8",
      "decoder_audit": {
        "expected_audit_sha256": "<64 hex>",
        "expected_file_sha256": "<64 hex>"
      }
    },
    "candidates": [
      {"alpha": "0.7", "lambda_dnh": "0.00075"},
      {"alpha": "1.0", "lambda_dnh": "0.00075"}
    ]
  }

``decoder_audit``은 fresh decode를 의도할 때만 ``null``이다. 후보 숫자는 JSON
number가 아니라 canonical decimal string이다. alpha 0.85가 필요해진 경우 candidate
목록은 0.7, 0.85, 1.0 순서의 새 contract로 다시 봉인한다.

seed 20260903은 독립 contract만 허용한다. 이때 ``campaign.second_seed``에는 외부
primary contract path/SHA, immutable primary borderline selection SHA와 기존 raw
selection이 계산한 seed-neutral campaign SHA를 모두 넣어야 한다.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 2
STATUS_SCHEMA_VERSION = 1
CANONICAL_PRETRAIN_CONFIG = "configs/train_pretrain_tiny.yaml"
CANONICAL_FINETUNE_CONFIG = "configs/train_finetune.yaml"
BOOTSTRAP_RECEIPT = "data/manifests/elice_bootstrap_receipt.json"
HOLDOUT_MANIFEST = "data/manifests/recorded_holdout.json"
TRANSFER_MANIFEST = "data/manifests/elice_transfer_manifest.json"
DECODER_AUDIT_REPORT = "results/provenance/decoder_audit.json"
CANONICAL_LEDGER = "results/training_prerequisites/canonical_pretrain.json"
EVIDENCE_ROOT = "results/training_prerequisites/evidence/canonical_campaign_v1"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_ALPHA_ORDER = ("0.7", "0.85", "1.0")
CANONICAL_STAGE_ORDER = (
    "bootstrap",
    "pre_g0_readiness",
    "g0_all_candidates",
    "prepilot_gradient_all_candidates",
    "loss_pilot_20k_each",
    "loss_pilot_recorded_val_each",
    "measured_probe_5k_each",
    "measured_probe_recorded_val_each",
    "raw_winner_selection",
    "selected20k_gradient",
    "resume_smoke",
    "issue_campaign_ledger",
    "issue_second_seed_prerequisite",
    "canonical_pretrain_100k",
    "finetune_readiness_17_of_17",
    "canonical_finetune_50k",
    "cross_seed_finalize_if_required",
)
_EXECUTABLE_ACTIONS = frozenset(
    {
        "bootstrap",
        "pre_g0_readiness",
        "g0",
        "prepilot_gradient",
        "loss_pilot",
        "loss_pilot_val",
        "measured_probe",
        "measured_probe_val",
        "selected20k_gradient",
        "resume_smoke",
        "issue_campaign_ledger",
        "issue_second_seed_prerequisite",
        "canonical_pretrain",
        "canonical_pretrain_resume",
        "finetune_readiness",
        "canonical_finetune",
        "canonical_finetune_resume",
        "cross_seed_finalize",
    }
)
_LEGACY_TOKENS = (
    "run_pretrain.sh",
    "run_parallel_models.sh",
    "run_job_queue.sh",
    "job_queue.py",
    "queue_gpu0.yaml",
    "queue_gpu1.yaml",
    "bootstrap.sh",
)
_ACTION_TARGETS = {
    "bootstrap": "scripts/elice/bootstrap_all.sh",
    "pre_g0_readiness": "scripts/train/check_finetune.py",
    "g0": "scripts/bench/diagnose_training_overfit.py",
    "prepilot_gradient": "scripts/train/measure_gradient_budget.py",
    "loss_pilot": "scripts/train/train.py",
    "loss_pilot_val": "scripts/eval/evaluate_recorded.py",
    "measured_probe": "scripts/train/train.py",
    "measured_probe_val": "scripts/eval/evaluate_recorded.py",
    "selected20k_gradient": "scripts/train/measure_gradient_budget.py",
    "resume_smoke": "scripts/train/run_a100_pretrain_smoke.py",
    "issue_campaign_ledger": "scripts/train/issue_canonical_pretrain_prerequisite.py",
    "issue_second_seed_prerequisite": "scripts/train/issue_second_seed_prerequisite.py",
    "canonical_pretrain": "scripts/train/train.py",
    "canonical_pretrain_resume": "scripts/train/train.py",
    "finetune_readiness": "scripts/train/check_finetune.py",
    "canonical_finetune": "scripts/train/run_finetune_pipeline.py",
    "canonical_finetune_resume": "scripts/train/run_finetune_pipeline.py",
    "cross_seed_finalize": "scripts/train/run_finetune_pipeline.py",
}


class CampaignError(RuntimeError):
    """사용자 입력 또는 현재 immutable artifact가 campaign 계약과 다름."""


@dataclass(frozen=True)
class Candidate:
    alpha_text: str
    lambda_text: str

    @property
    def alpha(self) -> float:
        return float(self.alpha_text)

    @property
    def lambda_dnh(self) -> float:
        return float(self.lambda_text)

    @property
    def identity(self) -> tuple[float, float, float]:
        return (self.alpha, 0.0, self.lambda_dnh)

    @property
    def key(self) -> str:
        digest = hashlib.sha256(
            f"{self.alpha_text}\0{self.lambda_text}".encode("ascii")
        ).hexdigest()[:12]
        alpha = self.alpha_text.replace(".", "p")
        dnh = self.lambda_text.lower().replace("+", "").replace("-", "m")
        dnh = dnh.replace(".", "p")
        return f"a{alpha}_dnh{dnh}_{digest}"


@dataclass(frozen=True)
class SecondSeedLink:
    primary_contract_path: Path
    primary_contract_sha256: str
    primary_selection_sha256: str
    seed_neutral_campaign_sha256: str


@dataclass(frozen=True)
class CampaignContract:
    path: Path
    sha256: str
    expected_commit: str
    expected_holdout_sha256: str
    expected_transfer_manifest_sha256: str
    raw_hash_workers: int
    cublas_workspace_config: str
    decoder_audit: dict[str, str] | None
    candidates: tuple[Candidate, ...]
    seed: int
    second_seed: SecondSeedLink | None


@dataclass
class Inspection:
    phase: str
    status: str
    next_action: str
    command: list[str] | None
    blockers: list[dict[str, str]]
    details: dict[str, Any]


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise CampaignError(f"{label}를 읽을 수 없습니다: {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CampaignError(f"{label}는 symlink가 아닌 regular file이어야 합니다: {path}")
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            content = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise CampaignError(f"{label}를 snapshot할 수 없습니다: {path}: {exc}") from exc
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(content) != after.st_size
    ):
        raise CampaignError(f"{label}가 검증 중 변경됐습니다: {path}")
    return content


def _json_object(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignError(f"{label} JSON이 손상됐습니다") from exc
    if not isinstance(value, dict):
        raise CampaignError(f"{label} 최상위는 mapping이어야 합니다")
    return value


def _exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise CampaignError(f"{label} key 집합이 정확하지 않습니다: {actual}")
    return value


def _hex(value: object, *, length: int, label: str) -> str:
    text = str(value or "").lower()
    pattern = _HEX40 if length == 40 else _HEX64
    if pattern.fullmatch(text) is None:
        raise CampaignError(f"{label}가 {length}자리 lowercase hex가 아닙니다")
    return text


def _canonical_positive_decimal(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _DECIMAL.fullmatch(value) is None:
        raise CampaignError(f"{label}는 canonical decimal string이어야 합니다")
    try:
        decimal = Decimal(value)
    except InvalidOperation as exc:  # pragma: no cover - regexp 방어
        raise CampaignError(f"{label} decimal이 잘못됐습니다") from exc
    if not decimal.is_finite() or decimal <= 0:
        raise CampaignError(f"{label}는 finite 양수여야 합니다")
    canonical = format(decimal, "f")
    if "." in canonical:
        canonical = canonical.rstrip("0").rstrip(".")
    if value != canonical:
        raise CampaignError(
            f"{label} 표기가 canonical이 아닙니다: {value!r}, expected={canonical!r}"
        )
    return value


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except ValueError:
        return False
    return True


def load_contract(
    path: Path,
    expected_sha256: str,
    *,
    repo_root: Path = REPO_ROOT,
    _seen_paths: frozenset[Path] = frozenset(),
) -> CampaignContract:
    path_identity = path.absolute()
    if path_identity in _seen_paths:
        raise CampaignError("campaign contract primary linkage에 순환이 있습니다")
    seen_paths = _seen_paths | {path_identity}
    expected_sha = _hex(expected_sha256, length=64, label="contract SHA-256")
    if _inside(path, repo_root):
        raise CampaignError(
            "campaign contract는 clean exact checkout을 오염시키지 않도록 저장소 밖에 있어야 합니다"
        )
    content = _regular_bytes(path, label="campaign contract")
    actual_sha = _sha256_bytes(content)
    if actual_sha != expected_sha:
        raise CampaignError(
            f"campaign contract SHA가 외부 anchor와 다릅니다: expected={expected_sha}, actual={actual_sha}"
        )
    payload = _exact_keys(
        _json_object(content, label="campaign contract"),
        {
            "schema_version",
            "expected_commit",
            "expected_holdout_sha256",
            "expected_transfer_manifest_sha256",
            "campaign",
            "bootstrap",
            "candidates",
        },
        label="campaign contract",
    )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise CampaignError("campaign contract schema_version이 다릅니다")
    campaign = _exact_keys(
        payload["campaign"], {"seed", "second_seed"}, label="campaign execution"
    )
    seed = campaign["seed"]
    if type(seed) is not int or seed not in {20260803, 20260903}:
        raise CampaignError("campaign.seed는 exact 20260803 또는 20260903이어야 합니다")
    second_raw = campaign["second_seed"]
    second_seed: SecondSeedLink | None
    if seed == 20260803:
        if second_raw is not None:
            raise CampaignError("primary seed contract의 campaign.second_seed는 null이어야 합니다")
        second_seed = None
    else:
        second_map = _exact_keys(
            second_raw,
            {
                "primary_contract_path",
                "primary_contract_sha256",
                "primary_selection_sha256",
                "seed_neutral_campaign_sha256",
            },
            label="campaign second-seed link",
        )
        primary_path_raw = second_map["primary_contract_path"]
        if not isinstance(primary_path_raw, str) or not primary_path_raw.strip():
            raise CampaignError("second-seed primary_contract_path는 absolute string이어야 합니다")
        primary_path = Path(primary_path_raw)
        if not primary_path.is_absolute() or _inside(primary_path, repo_root):
            raise CampaignError("second-seed primary contract는 저장소 밖 absolute path여야 합니다")
        if primary_path.absolute() == path.absolute():
            raise CampaignError("second-seed contract가 자기 자신을 primary로 가리킬 수 없습니다")
        second_seed = SecondSeedLink(
            primary_contract_path=primary_path.absolute(),
            primary_contract_sha256=_hex(
                second_map["primary_contract_sha256"],
                length=64,
                label="primary campaign contract SHA-256",
            ),
            primary_selection_sha256=_hex(
                second_map["primary_selection_sha256"],
                length=64,
                label="primary selection SHA-256",
            ),
            seed_neutral_campaign_sha256=_hex(
                second_map["seed_neutral_campaign_sha256"],
                length=64,
                label="seed-neutral campaign SHA-256",
            ),
        )
    bootstrap = _exact_keys(
        payload["bootstrap"],
        {"raw_hash_workers", "cublas_workspace_config", "decoder_audit"},
        label="campaign bootstrap",
    )
    workers = bootstrap["raw_hash_workers"]
    if type(workers) is not int or not 1 <= workers <= 32:
        raise CampaignError("bootstrap.raw_hash_workers는 1~32 정수여야 합니다")
    cublas = bootstrap["cublas_workspace_config"]
    if cublas not in {":4096:8", ":16:8"}:
        raise CampaignError("bootstrap.cublas_workspace_config가 승인값이 아닙니다")
    decoder_raw = bootstrap["decoder_audit"]
    decoder: dict[str, str] | None
    if decoder_raw is None:
        decoder = None
    else:
        decoder_map = _exact_keys(
            decoder_raw,
            {"expected_audit_sha256", "expected_file_sha256"},
            label="campaign decoder audit",
        )
        decoder = {
            "expected_audit_sha256": _hex(
                decoder_map["expected_audit_sha256"],
                length=64,
                label="decoder semantic SHA-256",
            ),
            "expected_file_sha256": _hex(
                decoder_map["expected_file_sha256"],
                length=64,
                label="decoder file SHA-256",
            ),
        }
    rows = payload["candidates"]
    if not isinstance(rows, list):
        raise CampaignError("campaign candidates는 list여야 합니다")
    candidates: list[Candidate] = []
    for index, raw in enumerate(rows):
        row = _exact_keys(raw, {"alpha", "lambda_dnh"}, label=f"candidate[{index}]")
        alpha = str(row["alpha"])
        if not isinstance(row["alpha"], str) or alpha not in _ALPHA_ORDER:
            raise CampaignError(f"candidate[{index}].alpha가 승인 decimal string이 아닙니다")
        lambda_text = _canonical_positive_decimal(
            row["lambda_dnh"], label=f"candidate[{index}].lambda_dnh"
        )
        candidates.append(Candidate(alpha, lambda_text))
    alpha_order = tuple(candidate.alpha_text for candidate in candidates)
    if alpha_order not in (("0.7", "1.0"), ("0.7", "0.85", "1.0")):
        raise CampaignError(
            "candidate alpha는 정확히 [0.7,1.0] 또는 [0.7,0.85,1.0] 순서여야 합니다"
        )
    result = CampaignContract(
        path=path.absolute(),
        sha256=actual_sha,
        expected_commit=_hex(payload["expected_commit"], length=40, label="expected commit"),
        expected_holdout_sha256=_hex(
            payload["expected_holdout_sha256"], length=64, label="holdout SHA-256"
        ),
        expected_transfer_manifest_sha256=_hex(
            payload["expected_transfer_manifest_sha256"],
            length=64,
            label="transfer manifest SHA-256",
        ),
        raw_hash_workers=workers,
        cublas_workspace_config=str(cublas),
        decoder_audit=decoder,
        candidates=tuple(candidates),
        seed=seed,
        second_seed=second_seed,
    )
    if second_seed is not None:
        primary = load_contract(
            second_seed.primary_contract_path,
            second_seed.primary_contract_sha256,
            repo_root=repo_root,
            _seen_paths=seen_paths,
        )
        if primary.seed != 20260803 or primary.second_seed is not None:
            raise CampaignError("second-seed link의 primary contract가 exact primary seed가 아닙니다")
        shared = (
            "expected_commit",
            "expected_holdout_sha256",
            "expected_transfer_manifest_sha256",
            "raw_hash_workers",
            "cublas_workspace_config",
            "decoder_audit",
            "candidates",
        )
        mismatched = [name for name in shared if getattr(primary, name) != getattr(result, name)]
        if mismatched:
            raise CampaignError(
                f"second-seed contract가 primary sealed campaign과 다릅니다: {mismatched}"
            )
    return result


def _git_output(repo_root: Path, args: list[str]) -> str:
    environment = dict(os.environ, GIT_NO_REPLACE_OBJECTS="1")
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            env=environment,
            check=True,
            capture_output=True,
            timeout=30,
        )
        return completed.stdout.decode("utf-8", errors="strict").strip()
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise CampaignError(f"git {' '.join(args)} 실패: {exc}") from exc


def validate_exact_source(repo_root: Path, expected_commit: str) -> dict[str, Any]:
    if _git_output(repo_root, ["rev-parse", "--is-inside-work-tree"]) != "true":
        raise CampaignError("저장소가 git working tree가 아닙니다")
    current = _git_output(repo_root, ["rev-parse", "--verify", "HEAD^{commit}"]).lower()
    if current != expected_commit:
        raise CampaignError(
            f"HEAD가 expected commit과 다릅니다: expected={expected_commit}, current={current}"
        )
    replace = _git_output(repo_root, ["for-each-ref", "--format=%(refname)", "refs/replace"])
    grafts = repo_root / ".git" / "info" / "grafts"
    if replace or (grafts.is_file() and grafts.stat().st_size > 0):
        raise CampaignError("git replace/graft가 있어 commit identity를 우회할 수 있습니다")
    flags = _git_output(repo_root, ["ls-files", "-v"])
    suspicious = [line for line in flags.splitlines() if line and (line[0].islower() or line[0] == "S")]
    dirty = _git_output(repo_root, ["status", "--porcelain=v1", "--untracked-files=all"])
    if suspicious or dirty:
        raise CampaignError(
            "canonical campaign은 clean exact checkout만 허용합니다: "
            f"dirty={dirty.splitlines()[:20]}, index_flags={suspicious[:20]}"
        )
    return {"expected_commit": expected_commit, "current_commit": current, "clean": True}


def _execution_file_seal(path: Path, *, repo_root: Path, label: str) -> dict[str, Any]:
    content = _regular_bytes(path, label=label)
    return {
        "path": _relative(path, repo_root),
        "sha256": _sha256_bytes(content),
        "size": len(content),
    }


def build_execution_seal(
    action: str, command: list[str], *, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    """dry-run이 승인한 entrypoint와 child script bytes를 봉인한다."""

    expected_relative = _ACTION_TARGETS.get(action)
    if expected_relative is None:
        raise CampaignError(f"실행 action target whitelist가 없습니다: {action}")
    if len(command) < 2:
        raise CampaignError(f"실행 command에 script target이 없습니다: {command}")
    expected_target = (repo_root / expected_relative).absolute()
    actual_target = Path(command[1]).absolute()
    if actual_target != expected_target:
        raise CampaignError(
            f"실행 target이 action whitelist와 다릅니다: expected={expected_target}, "
            f"actual={actual_target}"
        )
    if action == "bootstrap":
        if command[0] != "bash":
            raise CampaignError("bootstrap executable은 exact bash argv여야 합니다")
    elif Path(command[0]).absolute() != _python(repo_root):
        raise CampaignError("canonical child는 exact .venv/bin/python이어야 합니다")
    entrypoint = Path(__file__).absolute()
    return {
        "schema_version": 1,
        "action": action,
        "argv": list(command),
        "entrypoint": _execution_file_seal(
            entrypoint, repo_root=repo_root, label="campaign entrypoint"
        ),
        "command_target": _execution_file_seal(
            expected_target, repo_root=repo_root, label=f"{action} command target"
        ),
    }


def verify_pre_execution_authority(
    contract: CampaignContract,
    action: str,
    command: list[str],
    expected_seal: dict[str, Any],
    *,
    contract_path: Path,
    expected_contract_sha256: str,
    repo_root: Path = REPO_ROOT,
) -> None:
    """state write/print 뒤 child 직전 source·contract·script bytes를 재검증한다."""

    validate_exact_source(repo_root, contract.expected_commit)
    reloaded = load_contract(
        contract_path, expected_contract_sha256, repo_root=repo_root
    )
    if reloaded != contract:
        raise CampaignError("dry-run 뒤 campaign contract 의미가 바뀌었습니다")
    actual_seal = build_execution_seal(action, command, repo_root=repo_root)
    if actual_seal != expected_seal:
        raise CampaignError(
            "dry-run 뒤 campaign entrypoint 또는 command target bytes/argv가 바뀌었습니다"
        )


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return path.absolute().relative_to(repo_root.absolute()).as_posix()
    except ValueError as exc:
        raise CampaignError(f"artifact가 저장소 밖입니다: {path}") from exc


def _candidate_paths(candidate: Candidate, repo_root: Path) -> dict[str, Path]:
    root = repo_root / EVIDENCE_ROOT / candidate.key
    return {
        "root": root,
        "g0_dir": root / "g0",
        "g0_receipt": root / "g0" / "receipt.json",
        "gradient_dir": root / "prepilot_gradient",
        "gradient_receipt": root / "prepilot_gradient" / "receipt.json",
        "selected_gradient_dir": root / "selected20k_gradient",
        "selected_gradient_receipt": root / "selected20k_gradient" / "receipt.json",
    }


def _python(repo_root: Path) -> Path:
    value = repo_root / ".venv" / "bin" / "python"
    if not value.exists() or not os.access(value, os.X_OK):
        raise CampaignError(f"bootstrap exact venv Python이 없습니다: {value}")
    return value.absolute()


def _append_set(command: list[str], overrides: Iterable[str]) -> list[str]:
    for override in overrides:
        command.extend(["--set", override])
    return command


def _candidate_overrides(
    candidate: Candidate, bootstrap_sha: str, *, role: str, init_ckpt: str | None = None
) -> list[str]:
    common = [
        f"data.bootstrap_receipt_sha256={bootstrap_sha}",
        f"experiment_role={role}",
        "init_eligible=false",
        f"loss.nmse_cvar_alpha={candidate.alpha_text}",
        "loss.lambda_frame=0.0",
        f"loss.lambda_dnh={candidate.lambda_text}",
    ]
    if role == "loss_pilot":
        return [*common, "run_until_step=20000"]
    if role != "measured_probe" or init_ckpt is None:
        raise CampaignError(f"지원하지 않는 derivative role/init 조합: {role!r}")
    return [
        *common,
        "data.digital_primary_path_mode=measured",
        "run_until_step=5000",
        f"init_ckpt={json.dumps(init_ckpt)}",
    ]


def build_bootstrap_command(contract: CampaignContract, repo_root: Path) -> list[str]:
    command = [
        "bash",
        str((repo_root / "scripts/elice/bootstrap_all.sh").absolute()),
        "--expected-commit",
        contract.expected_commit,
        "--expected-holdout-sha256",
        contract.expected_holdout_sha256,
        "--expected-transfer-manifest-sha256",
        contract.expected_transfer_manifest_sha256,
        "--raw-hash-workers",
        str(contract.raw_hash_workers),
        "--no-update",
    ]
    if contract.decoder_audit is not None:
        command.extend(
            [
                "--reuse-decoder-audit",
                "--expected-decoder-audit-sha256",
                contract.decoder_audit["expected_audit_sha256"],
                "--expected-decoder-audit-file-sha256",
                contract.decoder_audit["expected_file_sha256"],
            ]
        )
    return command


def build_g0_command(
    contract: CampaignContract, candidate: Candidate, bootstrap_sha: str, repo_root: Path
) -> list[str]:
    paths = _candidate_paths(candidate, repo_root)
    return [
        str(_python(repo_root)),
        str((repo_root / "scripts/bench/diagnose_training_overfit.py").absolute()),
        "--config",
        CANONICAL_PRETRAIN_CONFIG,
        "--loss-alpha",
        candidate.alpha_text,
        "--loss-lambda-dnh",
        candidate.lambda_text,
        "--bootstrap-receipt-sha256",
        bootstrap_sha,
        "--evidence-dir",
        _relative(paths["g0_dir"], repo_root),
    ]


def build_gradient_command(
    candidate: Candidate, repo_root: Path, *, selected_pilot: Path | None = None
) -> list[str]:
    paths = _candidate_paths(candidate, repo_root)
    command = [
        str(_python(repo_root)),
        str((repo_root / "scripts/train/measure_gradient_budget.py").absolute()),
    ]
    if selected_pilot is None:
        command.extend(["--g0-receipt", _relative(paths["g0_receipt"], repo_root)])
        output = paths["gradient_dir"]
    else:
        command.extend(
            [
                "--checkpoint",
                _relative(selected_pilot, repo_root),
                "--authoritative-g0-receipt",
                _relative(paths["g0_receipt"], repo_root),
            ]
        )
        output = paths["selected_gradient_dir"]
    command.extend(["--out-dir", _relative(output, repo_root)])
    return command


def build_train_command(
    repo_root: Path,
    *,
    config: str,
    overrides: list[str],
    resume: Path | None = None,
) -> list[str]:
    command = [
        str(_python(repo_root)),
        str((repo_root / "scripts/train/train.py").absolute()),
        "--config",
        config,
    ]
    _append_set(command, overrides)
    if resume is not None:
        command.extend(["--resume", str(resume.absolute())])
    return command


def build_recorded_val_command(
    repo_root: Path,
    *,
    checkpoint: Path,
    manifest: Path,
    output: Path,
    allow_surrogate: bool,
) -> list[str]:
    command = [
        str(_python(repo_root)),
        str((repo_root / "scripts/eval/evaluate_recorded.py").absolute()),
        "--ckpt",
        str(checkpoint.absolute()),
        "--manifest",
        str(manifest.absolute()),
        "--split",
        "val",
        "--out",
        str(output.absolute()),
    ]
    if allow_surrogate:
        command.append("--allow-surrogate")
    return command


def _blocked(phase: str, code: str, message: str, **details: Any) -> Inspection:
    return Inspection(
        phase=phase,
        status="BLOCKED",
        next_action=code,
        command=None,
        blockers=[{"code": code, "message": message}],
        details=details,
    )


def _ready(phase: str, action: str, command: list[str], **details: Any) -> Inspection:
    if action not in _EXECUTABLE_ACTIONS:
        raise CampaignError(f"내부 next action whitelist 위반: {action}")
    flattened = "\0".join(command)
    if any(token in flattened for token in _LEGACY_TOKENS):
        raise CampaignError(f"legacy runner가 canonical command에 섞였습니다: {command}")
    return Inspection(
        phase=phase,
        status="READY_TO_EXECUTE",
        next_action=action,
        command=command,
        blockers=[],
        details=details,
    )


def _primary_evidence_transition(
    contract: CampaignContract, inspection: Inspection
) -> Inspection:
    """second seed가 fixed primary evidence pathname을 재발행하지 못하게 한다."""

    if contract.second_seed is None or inspection.status != "READY_TO_EXECUTE":
        return inspection
    return _blocked(
        "second_seed_admission",
        "SECOND_SEED_PRIMARY_EVIDENCE_REQUIRED",
        "seed=20260903은 primary loss-selection evidence/G0/pilot/probe/ledger를 "
        "새로 만들거나 재개하지 않습니다. sealed primary campaign artifact 전체를 "
        "동일 pathname/SHA로 먼저 복원하세요.",
        missing_primary_phase=inspection.phase,
        refused_action=inspection.next_action,
        refused_command_argv=inspection.command,
        **inspection.details,
    )


def _load_bootstrap_receipt(
    contract: CampaignContract, repo_root: Path
) -> tuple[str, dict[str, Any]] | None:
    path = repo_root / BOOTSTRAP_RECEIPT
    if not path.exists() and not path.is_symlink():
        return None
    content = _regular_bytes(path, label="Elice bootstrap receipt")
    payload = _exact_keys(
        _json_object(content, label="Elice bootstrap receipt"),
        {
            "schema_version",
            "expected_commit",
            "canonical_holdout",
            "transfer_manifest",
            "recorded_aggregate_sha256",
            "recorded_subband_coverage",
            "environment",
        },
        label="Elice bootstrap receipt",
    )
    holdout = _exact_keys(
        payload["canonical_holdout"], {"path", "sha256"}, label="bootstrap holdout"
    )
    transfer = _exact_keys(
        payload["transfer_manifest"], {"path", "sha256"}, label="bootstrap transfer"
    )
    if (
        payload["schema_version"] != 2
        or payload["expected_commit"] != contract.expected_commit
        or holdout
        != {"path": HOLDOUT_MANIFEST, "sha256": contract.expected_holdout_sha256}
        or transfer.get("path") != TRANSFER_MANIFEST
    ):
        raise CampaignError("bootstrap receipt가 contract commit/holdout/canonical path와 다릅니다")
    _hex(transfer.get("sha256"), length=64, label="bootstrap transfer manifest SHA-256")
    return _sha256_bytes(content), payload


def _check_anchor_file(path: Path, expected_sha: str, *, label: str) -> dict[str, Any]:
    content = _regular_bytes(path, label=label)
    actual = _sha256_bytes(content)
    if actual != expected_sha:
        raise CampaignError(
            f"{label} SHA가 외부 anchor와 다릅니다: expected={expected_sha}, actual={actual}"
        )
    return {"path": str(path), "sha256": actual, "size": len(content)}


def _inspect_local_admission(
    contract: CampaignContract, repo_root: Path
) -> tuple[dict[str, Any], Inspection | None]:
    source = validate_exact_source(repo_root, contract.expected_commit)
    holdout = _check_anchor_file(
        repo_root / HOLDOUT_MANIFEST,
        contract.expected_holdout_sha256,
        label="canonical holdout",
    )
    transfer = _check_anchor_file(
        repo_root / TRANSFER_MANIFEST,
        contract.expected_transfer_manifest_sha256,
        label="Elice transfer manifest",
    )
    transfer_payload = _json_object(
        _regular_bytes(repo_root / TRANSFER_MANIFEST, label="Elice transfer manifest"),
        label="Elice transfer manifest",
    )
    schema = transfer_payload.get("schema_version")
    context = {"source": source, "holdout": holdout, "transfer": {**transfer, "schema_version": schema}}
    if schema != 2:
        return context, _blocked(
            "local_admission",
            "LOCAL_TRANSFER_SCHEMA_V2_REQUIRED",
            f"canonical campaign에는 101-session schema-v2 transfer가 필요합니다: current={schema!r}",
            local=context,
        )
    receipt = _load_bootstrap_receipt(contract, repo_root)
    previous_bootstrap: dict[str, Any] | None = None
    if receipt is not None:
        previous_sha, previous_payload = receipt
        previous_transfer_sha = str(previous_payload["transfer_manifest"]["sha256"])
        if previous_transfer_sha != contract.expected_transfer_manifest_sha256:
            # docs/05의 같은-commit v1 selector bootstrap → v2 training bootstrap
            # 전이는 canonical bootstrap_all.sh가 receipt를 atomic replace하는 유일한
            # 승인 경로다. 이전 receipt를 학습 authority로 사용하지 않고 dry-run에
            # concrete prior SHA를 남긴 뒤 full bootstrap을 다시 실행한다.
            previous_bootstrap = {
                "path": str(repo_root / BOOTSTRAP_RECEIPT),
                "sha256": previous_sha,
                "transfer_manifest_sha256": previous_transfer_sha,
                "replacement_transfer_manifest_sha256": (
                    contract.expected_transfer_manifest_sha256
                ),
            }
            context["previous_bootstrap_receipt"] = previous_bootstrap
            receipt = None
    if receipt is None:
        if contract.decoder_audit is not None:
            audit_path = repo_root / DECODER_AUDIT_REPORT
            if not audit_path.exists() and not audit_path.is_symlink():
                return context, _blocked(
                    "bootstrap",
                    "DECODER_AUDIT_CACHE_MISSING",
                    "contract가 decoder-audit reuse를 요구하지만 canonical cache path가 "
                    f"없습니다: {audit_path}",
                    local=context,
                )
            try:
                audit_content = _regular_bytes(
                    audit_path, label="reused decoder audit"
                )
                audit_file_sha = _sha256_bytes(audit_content)
                if audit_file_sha != contract.decoder_audit["expected_file_sha256"]:
                    raise CampaignError(
                        "decoder audit file SHA가 contract와 다릅니다: "
                        f"expected={contract.decoder_audit['expected_file_sha256']}, "
                        f"actual={audit_file_sha}"
                    )
                audit_payload = _json_object(
                    audit_content, label="reused decoder audit"
                )
                if (
                    str(audit_payload.get("audit_sha256", ""))
                    != contract.decoder_audit["expected_audit_sha256"]
                ):
                    raise CampaignError(
                        "decoder audit semantic SHA가 contract와 다릅니다"
                    )
                context["decoder_audit"] = {
                    "path": str(audit_path),
                    "file_sha256": audit_file_sha,
                    "audit_sha256": audit_payload["audit_sha256"],
                }
            except CampaignError as exc:
                return context, _blocked(
                    "bootstrap",
                    "DECODER_AUDIT_CACHE_INVALID",
                    str(exc),
                    local=context,
                )
        return context, _ready(
            "bootstrap",
            "bootstrap",
            build_bootstrap_command(contract, repo_root),
            local=context,
            replaces_previous_bootstrap=previous_bootstrap,
        )
    bootstrap_sha, receipt_payload = receipt
    context["bootstrap"] = {
        "path": str(repo_root / BOOTSTRAP_RECEIPT),
        "sha256": bootstrap_sha,
        "schema_version": receipt_payload["schema_version"],
    }
    return context, None


def _lazy_imports(repo_root: Path) -> dict[str, Any]:
    expected_prefix = (repo_root / ".venv").absolute()
    if Path(sys.prefix).absolute() != expected_prefix:
        raise CampaignError(
            "bootstrap 뒤 campaign 검증은 exact .venv interpreter로 다시 실행해야 "
            f"합니다: expected_prefix={expected_prefix}, current_prefix={sys.prefix}"
        )
    source = str(repo_root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        import torch

        from deep_anc.config import canonical_recorded_manifest_for_data, load_train_config
        from deep_anc.train.a100_pretrain_smoke import (
            SMOKE_ROOT,
            build_a100_pretrain_smoke_target,
            validate_a100_pretrain_smoke_receipt,
        )
        from deep_anc.train.campaign_evidence import (
            FAILED_G0_RECEIPT_KIND,
            MEASURED_PROBE_SELECTION_SCORE,
            make_campaign_evidence_reference,
            select_loss_pilot,
            validate_g0_receipt,
            validate_gradient_budget_receipt,
            validate_loss_pilot_candidate,
            validate_measured_probe,
            validate_prepilot_gradient_receipt,
        )
        from deep_anc.train.campaign_prerequisite import (
            validate_canonical_pretrain_prerequisites,
        )
        from deep_anc.train.completion_receipt import validate_completion_receipt
        from deep_anc.train.evaluation_contract import (
            canonical_test_ledger_paths_from_payload,
            seed_neutral_campaign_sha256,
            validate_recorded_val_selection,
            validate_test_open_selection,
        )
        from deep_anc.train.experiment_contract import (
            validate_embedded_experiment_contract,
        )
        from deep_anc.train.finetune_readiness import (
            audit_finetune_completion,
            audit_finetune_readiness,
        )
        from deep_anc.train.process_lock import autostart_state_dir
        from deep_anc.train.second_seed_prerequisite import (
            build_second_seed_prerequisite_payload,
            prerequisite_sha256,
            second_seed_prerequisite_path,
            validate_second_seed_prerequisites,
            validate_second_seed_test_ledger_state,
        )
    except (ImportError, OSError) as exc:
        raise CampaignError(
            "bootstrap 뒤 상태 검사는 exact .venv interpreter로 실행해야 합니다: "
            f"{repo_root / '.venv/bin/python'} ({exc})"
        ) from exc
    return locals()


def _canonical_cfg(
    modules: dict[str, Any],
    contract: CampaignContract,
    bootstrap_sha: str,
    candidate: Candidate,
    *,
    ledger_sha: str = "0" * 64,
    seed: int | None = None,
    second_seed_prerequisite: str | None = None,
    second_seed_prerequisite_sha256: str | None = None,
) -> dict[str, Any]:
    execution_seed = contract.seed if seed is None else seed
    overrides = [
        f"data.bootstrap_receipt_sha256={bootstrap_sha}",
        f"campaign_prerequisite_sha256={ledger_sha}",
        f"loss.nmse_cvar_alpha={candidate.alpha_text}",
        "loss.lambda_frame=0.0",
        f"loss.lambda_dnh={candidate.lambda_text}",
        f"seed={execution_seed}",
    ]
    if second_seed_prerequisite is not None:
        overrides.extend(
            [
                f"second_seed_prerequisite={json.dumps(second_seed_prerequisite)}",
                "second_seed_prerequisite_sha256="
                f"{json.dumps(str(second_seed_prerequisite_sha256))}",
            ]
        )
    return modules["load_train_config"](
        REPO_ROOT / CANONICAL_PRETRAIN_CONFIG,
        overrides,
    )


def _derivative_cfg(
    modules: dict[str, Any],
    contract: CampaignContract,
    bootstrap_sha: str,
    candidate: Candidate,
    *,
    role: str,
    init_ckpt: Path | None = None,
) -> dict[str, Any]:
    init_text = _relative(init_ckpt, REPO_ROOT) if init_ckpt is not None else None
    return modules["load_train_config"](
        REPO_ROOT / CANONICAL_PRETRAIN_CONFIG,
        _candidate_overrides(
            candidate, bootstrap_sha, role=role, init_ckpt=init_text
        ),
    )


def _reference(modules: dict[str, Any], repo_root: Path, path: Path, *, label: str) -> dict[str, str]:
    return modules["make_campaign_evidence_reference"](repo_root, path, label=label)


def _checkpoint(modules: dict[str, Any], path: Path) -> dict[str, Any]:
    snapshot = modules["snapshot_regular_file"](path) if "snapshot_regular_file" in modules else None
    if snapshot is None:
        from deep_anc.train.evaluation_contract import snapshot_regular_file

        snapshot = snapshot_regular_file(path)
    try:
        raw = modules["torch"].load(
            io.BytesIO(snapshot.content), map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CampaignError(f"checkpoint를 읽을 수 없습니다: {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("cfg"), dict):
        raise CampaignError(f"checkpoint에 resolved cfg가 없습니다: {path}")
    return raw


def _current_experiment_contract_sha(
    modules: dict[str, Any], cfg: dict[str, Any], *, label: str
) -> str:
    """현재 resolved cfg의 embedded/top-level/run-dir 계약을 하나로 닫는다."""

    try:
        embedded = modules["validate_embedded_experiment_contract"](cfg)
        digest = _hex(
            embedded.get("sha256"), length=64, label=f"{label} embedded contract SHA"
        )
    except (KeyError, RuntimeError, TypeError, ValueError, CampaignError) as exc:
        raise CampaignError(f"{label} current experiment contract가 유효하지 않습니다: {exc}") from exc
    if cfg.get("experiment_contract_sha256") != digest:
        raise CampaignError(f"{label} top-level experiment contract SHA가 embedded와 다릅니다")
    resolved = cfg.get("resolved_contract_run_dir")
    if not isinstance(resolved, dict) or resolved.get("experiment_contract_sha256") != digest:
        raise CampaignError(f"{label} run-directory contract SHA가 embedded와 다릅니다")
    return digest


def _validated_completion_receipt(
    modules: dict[str, Any],
    ckpt_dir: Path,
    *,
    expected_role: str,
    expected_init_eligible: bool,
    expected_contract_sha256: str,
    repo_root: Path,
) -> dict[str, Any]:
    receipt = modules["validate_completion_receipt"](
        ckpt_dir,
        expected_role=expected_role,
        expected_init_eligible=expected_init_eligible,
        repo_root=repo_root,
    )
    actual = _hex(
        receipt.get("experiment_contract_sha256"),
        length=64,
        label=f"{expected_role} completion receipt contract SHA",
    )
    if actual != expected_contract_sha256:
        raise CampaignError(
            f"{expected_role} completion receipt가 현재 resolved contract와 다릅니다: "
            f"expected={expected_contract_sha256}, actual={actual}"
        )
    return receipt


def _authority_report_view(value: Any) -> Any:
    """fresh audit 비교에서 시각 필드만 제거한다."""

    if isinstance(value, dict):
        return {
            key: _authority_report_view(item)
            for key, item in value.items()
            if key != "checked_at_utc"
        }
    if isinstance(value, list):
        return [_authority_report_view(item) for item in value]
    return value


def _run_progress(
    modules: dict[str, Any], run_dir: Path, expected_step: int
) -> tuple[str, dict[str, Any]]:
    best = run_dir / "ckpt" / "best.pt"
    last = run_dir / "ckpt" / "last.pt"
    if not run_dir.exists() and not run_dir.is_symlink():
        return "absent", {"run_dir": str(run_dir), "best": str(best), "last": str(last)}
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise CampaignError(f"run directory가 symlink/비-directory입니다: {run_dir}")
    if not last.exists() and not best.exists():
        if any(run_dir.iterdir()):
            raise CampaignError(f"checkpoint 없는 nonempty run directory가 있습니다: {run_dir}")
        return "absent", {"run_dir": str(run_dir), "best": str(best), "last": str(last)}
    if not last.is_file():
        raise CampaignError(f"best/partial artifact는 있으나 last.pt가 없습니다: {run_dir}")
    last_raw = _checkpoint(modules, last)
    step = last_raw.get("step")
    if type(step) is not int or not 0 <= step <= expected_step:
        raise CampaignError(f"last.pt step이 범위 밖입니다: {step!r}/{expected_step}")
    detail = {
        "run_dir": str(run_dir),
        "best": str(best),
        "last": str(last),
        "last_step": step,
        "expected_step": expected_step,
    }
    if step < expected_step:
        return "partial", detail
    if not best.is_file():
        raise CampaignError(f"완료 step의 best.pt가 없습니다: {best}")
    return "complete", detail


def _explicit_resume(
    *,
    action: str,
    phase: str,
    base_command: list[str],
    expected_last: Path,
    resume_path: Path | None,
    resume_sha256: str | None,
    details: dict[str, Any],
) -> Inspection:
    if resume_path is None or resume_sha256 is None:
        return _blocked(
            phase,
            "EXPLICIT_RESUME_REQUIRED",
            "partial last.pt를 자동 재개하지 않습니다. --resume-checkpoint와 "
            "--expected-resume-checkpoint-sha256을 함께 지정하세요.",
            expected_resume=str(expected_last),
            **details,
        )
    requested = resume_path.absolute()
    if requested != expected_last.absolute():
        return _blocked(
            phase,
            "RESUME_PATH_MISMATCH",
            f"resume path가 현재 contract run의 exact last.pt가 아닙니다: {requested}",
            expected_resume=str(expected_last),
            **details,
        )
    try:
        expected_sha = _hex(resume_sha256, length=64, label="resume checkpoint SHA-256")
        content = _regular_bytes(requested, label="resume checkpoint")
    except CampaignError as exc:
        return _blocked(phase, "RESUME_INVALID", str(exc), **details)
    actual = _sha256_bytes(content)
    if actual != expected_sha:
        return _blocked(
            phase,
            "RESUME_SHA_MISMATCH",
            f"resume checkpoint SHA 불일치: expected={expected_sha}, actual={actual}",
            **details,
        )
    command = [*base_command, "--resume", str(requested)]
    return _ready(phase, action, command, resume_sha256=actual, **details)


def _smoke_cfg(
    modules: dict[str, Any], bootstrap_sha: str, winner: Candidate, *, seed: int
) -> dict[str, Any]:
    return modules["load_train_config"](
        REPO_ROOT / CANONICAL_PRETRAIN_CONFIG,
        [
            "experiment_role=a100_pretrain_smoke",
            "init_eligible=false",
            "contract_run_dir=false",
            "campaign_prerequisite=null",
            "campaign_prerequisite_sha256=null",
            "second_seed_prerequisite=null",
            "second_seed_prerequisite_sha256=null",
            "init_ckpt=null",
            "a100_smoke_run_label=uninterrupted",
            "run_until_step=500",
            f"data.bootstrap_receipt_sha256={bootstrap_sha}",
            f"loss.nmse_cvar_alpha={winner.alpha_text}",
            f"loss.lambda_dnh={winner.lambda_text}",
            f"seed={seed}",
        ],
    )


def _issuer_command(
    bootstrap_sha: str,
    winner: Candidate,
    chains: list[dict[str, Any]],
    smoke_root: Path,
    repo_root: Path,
) -> list[str]:
    command = [
        str(_python(repo_root)),
        str((repo_root / "scripts/train/issue_canonical_pretrain_prerequisite.py").absolute()),
        "--bootstrap-receipt-sha256",
        bootstrap_sha,
        "--loss-alpha",
        winner.alpha_text,
        "--loss-lambda-dnh",
        winner.lambda_text,
    ]
    selected_gradient = _candidate_paths(winner, repo_root)["selected_gradient_receipt"]
    command.extend(["--gradient-receipt", _relative(selected_gradient, repo_root)])
    for row in chains:
        paths = row["paths"]
        pilot = Path(paths["pilot_dir"])
        probe = Path(paths["probe_dir"])
        command.extend(["--g0-receipt", _relative(Path(paths["g0_receipt"]), repo_root)])
        command.extend(
            ["--prepilot-gradient-receipt", _relative(Path(paths["gradient_receipt"]), repo_root)]
        )
        command.extend(["--pilot-best", _relative(pilot / "ckpt/best.pt", repo_root)])
        command.extend(["--pilot-last", _relative(pilot / "ckpt/last.pt", repo_root)])
        command.extend(
            ["--pilot-metrics", _relative(pilot / "eval_recorded_val/metrics.npz", repo_root)]
        )
        command.extend(["--probe-best", _relative(probe / "ckpt/best.pt", repo_root)])
        command.extend(["--probe-last", _relative(probe / "ckpt/last.pt", repo_root)])
        command.extend(
            ["--probe-metrics", _relative(probe / "eval_recorded_val/metrics.npz", repo_root)]
        )
        command.extend(
            ["--probe-init-checkpoint", _relative(pilot / "ckpt/best.pt", repo_root)]
        )
    command.extend(
        [
            "--smoke-receipt",
            _relative(smoke_root / "receipt.json", repo_root),
            "--smoke-environment-receipt",
            _relative(smoke_root / "environment_receipt.json", repo_root),
            "--smoke-telemetry",
            _relative(smoke_root / "telemetry.json", repo_root),
        ]
    )
    return command


def _second_seed_issuer_command(
    *,
    repo_root: Path,
    bootstrap_sha: str,
    ledger_sha: str,
    winner: Candidate,
    primary_selection: Path,
    smoke_root: Path,
    destination: Path,
) -> list[str]:
    return [
        str(_python(repo_root)),
        str((repo_root / "scripts/train/issue_second_seed_prerequisite.py").absolute()),
        "--config",
        CANONICAL_PRETRAIN_CONFIG,
        "--bootstrap-receipt-sha256",
        bootstrap_sha,
        "--campaign-prerequisite-sha256",
        ledger_sha,
        "--loss-alpha",
        winner.alpha_text,
        "--loss-lambda-dnh",
        winner.lambda_text,
        "--primary-selection",
        str(primary_selection.absolute()),
        "--secondary-smoke-receipt",
        str((smoke_root / "receipt.json").absolute()),
        "--secondary-smoke-environment-receipt",
        str((smoke_root / "environment_receipt.json").absolute()),
        "--secondary-smoke-telemetry",
        str((smoke_root / "telemetry.json").absolute()),
        "--out",
        str(destination.absolute()),
    ]


def _finetune_overrides(
    *, bootstrap_sha: str, winner: Candidate, init_checkpoint: Path, seed: int | None
) -> list[str]:
    values = [
        "data.digital_primary_path_mode=measured",
        f"init_ckpt={json.dumps(str(init_checkpoint.absolute()))}",
        f"data.bootstrap_receipt_sha256={bootstrap_sha}",
        f"loss.nmse_cvar_alpha={winner.alpha_text}",
        f"loss.lambda_dnh={winner.lambda_text}",
    ]
    if seed is not None:
        values.append(f"seed={seed}")
    return values


def _readiness_command(
    repo_root: Path,
    *,
    bootstrap_sha: str,
    winner: Candidate,
    init_checkpoint: Path,
    seed: int,
) -> list[str]:
    command = [
        str(_python(repo_root)),
        str((repo_root / "scripts/train/check_finetune.py").absolute()),
        "--config",
        CANONICAL_FINETUNE_CONFIG,
    ]
    return _append_set(
        command,
        _finetune_overrides(
            bootstrap_sha=bootstrap_sha,
            winner=winner,
            init_checkpoint=init_checkpoint,
            seed=seed,
        ),
    )


def _pre_g0_readiness_command(
    repo_root: Path,
    *,
    bootstrap_sha: str,
    candidate: Candidate,
    out_dir: Path,
) -> list[str]:
    """Init만 비운 measured readiness를 기록하는 16/17 pre-G0 gate."""

    command = [
        str(_python(repo_root)),
        str((repo_root / "scripts/train/check_finetune.py").absolute()),
        "--config",
        CANONICAL_FINETUNE_CONFIG,
    ]
    _append_set(
        command,
        [
            "data.digital_primary_path_mode=measured",
            "init_ckpt=null",
            f"data.bootstrap_receipt_sha256={bootstrap_sha}",
            f"loss.nmse_cvar_alpha={candidate.alpha_text}",
            f"loss.lambda_dnh={candidate.lambda_text}",
        ],
    )
    command.extend(["--out-dir", str(out_dir.absolute())])
    return command


def _finetune_pipeline_command(
    repo_root: Path,
    *,
    bootstrap_sha: str,
    winner: Candidate,
    init_checkpoint: Path,
    seed: int,
) -> list[str]:
    command = [
        str(_python(repo_root)),
        str((repo_root / "scripts/train/run_finetune_pipeline.py").absolute()),
        "--config",
        CANONICAL_FINETUNE_CONFIG,
    ]
    return _append_set(
        command,
        _finetune_overrides(
            bootstrap_sha=bootstrap_sha,
            winner=winner,
            init_checkpoint=init_checkpoint,
            seed=seed,
        ),
    )


def _validated_seed_selection(
    modules: dict[str, Any],
    path: Path,
    *,
    expected_seed: int,
    expected_sha256: str | None = None,
    expected_seed_neutral_sha256: str | None = None,
    expected_experiment_contract_sha256: str | None = None,
) -> dict[str, Any]:
    content = _regular_bytes(path, label=f"seed {expected_seed} recorded-val selection")
    actual_sha = _sha256_bytes(content)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise CampaignError(
            f"seed {expected_seed} selection SHA가 sealed contract와 다릅니다: "
            f"expected={expected_sha256}, actual={actual_sha}"
        )
    payload = _json_object(content, label=f"seed {expected_seed} recorded-val selection")
    try:
        decision = modules["validate_recorded_val_selection"](payload)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CampaignError(f"seed {expected_seed} raw val selection이 유효하지 않습니다: {exc}") from exc
    if payload.get("seed") != expected_seed or payload.get("decision") != decision:
        raise CampaignError(f"seed {expected_seed} selection seed/decision이 raw 재계산과 다릅니다")
    if (
        expected_experiment_contract_sha256 is not None
        and payload.get("experiment_contract_sha256")
        != expected_experiment_contract_sha256
    ):
        raise CampaignError(
            f"seed {expected_seed} selection contract SHA가 현재 canonical run과 다릅니다"
        )
    neutral = _hex(
        payload.get("seed_neutral_campaign_sha256"),
        length=64,
        label=f"seed {expected_seed} seed-neutral campaign SHA",
    )
    if expected_seed_neutral_sha256 is not None and neutral != expected_seed_neutral_sha256:
        raise CampaignError(
            f"seed {expected_seed} seed-neutral linkage가 sealed contract와 다릅니다: "
            f"expected={expected_seed_neutral_sha256}, actual={neutral}"
        )
    return {
        "path": path,
        "sha256": actual_sha,
        "payload": payload,
        "decision": decision,
        "seed_neutral_campaign_sha256": neutral,
    }


def _cross_seed_final_selection_path(repo_root: Path, seed_neutral_sha256: str) -> Path:
    digest = _hex(
        seed_neutral_sha256, length=64, label="cross-seed seed-neutral campaign SHA"
    )
    return (
        repo_root
        / "results/finetune_cross_seed"
        / digest
        / "recorded_val_selection.json"
    )


def _cross_seed_winner_seed(
    primary: dict[str, Any], secondary: dict[str, Any]
) -> int:
    if primary["decision"].get("status") != "borderline":
        raise CampaignError(
            "seed 20260803 raw val 결과가 borderline이 아니므로 second-seed finalize 자격이 없습니다"
        )
    rows = (primary, secondary)
    eligible = [
        row
        for row in rows
        if row["decision"].get("g4_verdict") == "PASS"
        and float(row["decision"].get("minimum_margin_db", float("-inf"))) >= 0.0
    ]
    if not eligible:
        raise CampaignError("두 seed 중 val G4 PASS인 checkpoint가 없어 test를 열 수 없습니다")
    winner = max(
        eligible,
        key=lambda row: (
            float(row["decision"]["minimum_margin_db"]),
            -int(row["payload"]["seed"]),
        ),
    )
    return int(winner["payload"]["seed"])


def _cross_seed_finalize_command(
    repo_root: Path,
    *,
    primary_selection: Path,
    secondary_selection: Path,
    final_selection: Path,
) -> list[str]:
    command = [
        str(_python(repo_root)),
        str((repo_root / "scripts/train/run_finetune_pipeline.py").absolute()),
        "--config",
        CANONICAL_FINETUNE_CONFIG,
    ]
    # winner seed/init/bootstrap/loss는 finalizer가 두 raw selection을 검증한 뒤
    # winner checkpoint embedded cfg에서 exact 재구성한다. caller override 금지.
    command.extend(
        [
            "--cross-seed-selection",
            str(primary_selection.absolute()),
            "--cross-seed-selection",
            str(secondary_selection.absolute()),
            "--cross-seed-final-selection",
            str(final_selection.absolute()),
        ]
    )
    return command


def _validate_second_seed_primary_context(
    modules: dict[str, Any],
    contract: CampaignContract,
    *,
    repo_root: Path,
    bootstrap_sha: str,
    ledger_sha: str,
    winner: Candidate,
) -> dict[str, Any]:
    """secondary GPU 단계 전에 sealed primary borderline chain을 전부 다시 연다."""

    link = contract.second_seed
    if link is None:
        raise CampaignError("second-seed primary context를 primary contract에서 요청했습니다")
    primary_pretrain_cfg = _canonical_cfg(
        modules,
        contract,
        bootstrap_sha,
        winner,
        ledger_sha=ledger_sha,
        seed=20260803,
    )
    modules["validate_canonical_pretrain_prerequisites"](
        primary_pretrain_cfg, repo_root=repo_root
    )
    primary_pretrain_contract_sha = _current_experiment_contract_sha(
        modules, primary_pretrain_cfg, label="primary canonical pretrain"
    )
    primary_pretrain_dir = repo_root / str(primary_pretrain_cfg["ckpt_dir"])
    primary_pretrain_completion = _validated_completion_receipt(
        modules,
        primary_pretrain_dir / "ckpt",
        expected_role="canonical_pretrain",
        expected_init_eligible=True,
        expected_contract_sha256=primary_pretrain_contract_sha,
        repo_root=repo_root,
    )
    primary_init = primary_pretrain_dir / "ckpt/best.pt"
    primary_finetune_cfg = modules["load_train_config"](
        repo_root / CANONICAL_FINETUNE_CONFIG,
        _finetune_overrides(
            bootstrap_sha=bootstrap_sha,
            winner=winner,
            init_checkpoint=primary_init,
            seed=20260803,
        ),
    )
    primary_finetune_contract_sha = _current_experiment_contract_sha(
        modules, primary_finetune_cfg, label="primary canonical fine-tune"
    )
    primary_finetune_dir = repo_root / str(primary_finetune_cfg["ckpt_dir"])
    primary_finetune_completion = _validated_completion_receipt(
        modules,
        primary_finetune_dir / "ckpt",
        expected_role="canonical_finetune",
        expected_init_eligible=False,
        expected_contract_sha256=primary_finetune_contract_sha,
        repo_root=repo_root,
    )
    primary_state_root = modules["autostart_state_dir"](primary_finetune_dir)
    primary_selection_path = primary_state_root / "audit/recorded_val_selection.json"
    primary_selection = _validated_seed_selection(
        modules,
        primary_selection_path,
        expected_seed=20260803,
        expected_sha256=link.primary_selection_sha256,
        expected_seed_neutral_sha256=link.seed_neutral_campaign_sha256,
        expected_experiment_contract_sha256=primary_finetune_contract_sha,
    )
    if primary_selection["decision"].get("status") != "borderline":
        raise CampaignError("sealed primary raw val decision이 numeric/CI borderline이 아닙니다")
    planned_neutral = modules["seed_neutral_campaign_sha256"](
        primary_finetune_cfg
    )
    if planned_neutral != link.seed_neutral_campaign_sha256:
        raise CampaignError("primary resolved config의 seed-neutral digest가 sealed link와 다릅니다")
    test_ledger_state = modules["validate_second_seed_test_ledger_state"](
        primary_selection["payload"],
        primary_selection_path=primary_selection["path"],
        primary_selection_sha256=primary_selection["sha256"],
        repo_root=repo_root,
    )
    return {
        "primary_pretrain_cfg": primary_pretrain_cfg,
        "primary_pretrain_contract_sha": primary_pretrain_contract_sha,
        "primary_pretrain_dir": primary_pretrain_dir,
        "primary_pretrain_completion": primary_pretrain_completion,
        "primary_finetune_cfg": primary_finetune_cfg,
        "primary_finetune_contract_sha": primary_finetune_contract_sha,
        "primary_finetune_dir": primary_finetune_dir,
        "primary_finetune_completion": primary_finetune_completion,
        "primary_state_root": primary_state_root,
        "primary_selection": primary_selection,
        "test_ledger_state": test_ledger_state,
    }


def _inspect_finetune_terminal_authority(
    modules: dict[str, Any],
    *,
    repo_root: Path,
    finetune_cfg: dict[str, Any],
    finetune_dir: Path,
    state_root: Path,
    expected_contract_sha256: str,
    pretrain_dir: Path,
    winner_detail: dict[str, Any],
    run_detail: dict[str, Any],
    selection_path: Path | None = None,
    completion_report_path: Path | None = None,
    status_path: Path | None = None,
) -> Inspection:
    """advisory status와 독립적으로 50k raw completion chain을 재검증한다."""

    status_path = status_path or (state_root / "status.json")
    status_observation: dict[str, Any] = {
        "path": str(status_path),
        "present": False,
        "advisory_only": True,
    }
    if status_path.exists() or status_path.is_symlink():
        try:
            status_payload = _json_object(
                _regular_bytes(status_path, label="advisory fine-tune pipeline status"),
                label="advisory fine-tune pipeline status",
            )
            status_observation.update(
                {
                    "present": True,
                    "phase": status_payload.get("phase"),
                    "exit_code": status_payload.get("exit_code"),
                    "declared_advisory": status_payload.get("advisory"),
                }
            )
        except CampaignError as exc:
            status_observation.update({"present": True, "invalid": str(exc)})

    audit_dir = state_root / "audit"
    selection_path = selection_path or (audit_dir / "recorded_val_selection.json")
    val_metrics = finetune_dir / "eval_recorded_val" / "metrics.npz"
    test_metrics = finetune_dir / "eval_recorded_test" / "metrics.npz"
    completion_report_path = completion_report_path or (audit_dir / "completion.json")
    try:
        completion_receipt = _validated_completion_receipt(
            modules,
            finetune_dir / "ckpt",
            expected_role="canonical_finetune",
            expected_init_eligible=False,
            expected_contract_sha256=expected_contract_sha256,
            repo_root=repo_root,
        )
        selection_content = _regular_bytes(
            selection_path, label="recorded-val selection authority"
        )
        selection_payload = _json_object(
            selection_content, label="recorded-val selection authority"
        )
        modules["validate_test_open_selection"](selection_payload)
        capability_path, consumed_path = modules[
            "canonical_test_ledger_paths_from_payload"
        ](selection_payload, repo_root=repo_root)
        selected = selection_payload.get("selected")
        if not isinstance(selected, dict) or not isinstance(
            selected.get("checkpoint"), str
        ):
            raise CampaignError("recorded-val selection에 selected checkpoint가 없습니다")
        selected_checkpoint = Path(selected["checkpoint"])
        completion_content = _regular_bytes(
            completion_report_path, label="fine-tune terminal completion report"
        )
        persisted_report = _json_object(
            completion_content, label="fine-tune terminal completion report"
        )
        fresh_report = modules["audit_finetune_completion"](
            finetune_cfg,
            checkpoint=selected_checkpoint,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            selection=selection_path,
            test_capability=capability_path,
            test_consumed_marker=consumed_path,
        )
        if not bool(fresh_report.get("ok")) or not bool(
            fresh_report.get("fine_tuning_complete")
        ):
            failed = [
                str(row.get("id"))
                for row in fresh_report.get("checks", [])
                if not bool(row.get("ok"))
            ]
            raise CampaignError(
                f"fresh fine-tune completion audit가 PASS가 아닙니다: {failed}"
            )
        if _authority_report_view(persisted_report) != _authority_report_view(
            fresh_report
        ):
            raise CampaignError(
                "terminal completion report가 현재 raw config/contract/eval/ledger "
                "재검증 결과와 다릅니다"
            )
        capability_content = _regular_bytes(
            capability_path, label="single-use test capability"
        )
        consumed_content = _regular_bytes(
            consumed_path, label="single-use test consumed marker"
        )
        completed_path = capability_path.parent / "completed.json"
        completed_content = _regular_bytes(
            completed_path, label="single-use test completed marker"
        )
        val_content = _regular_bytes(val_metrics, label="canonical recorded val metrics")
        test_content = _regular_bytes(
            test_metrics, label="canonical recorded test metrics"
        )
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        CampaignError,
    ) as exc:
        return _blocked(
            "canonical_finetune",
            "FINETUNE_TERMINAL_AUTHORITY_INVALID",
            "advisory status는 완료 authority가 아닙니다. raw val selection, single-use "
            f"test ledger/metrics와 completion report 재검증이 필요합니다: {exc}",
            winner=winner_detail,
            run=run_detail,
            pipeline_status_observation=status_observation,
        )

    return Inspection(
        phase="complete",
        status="COMPLETE",
        next_action="none",
        command=None,
        blockers=[],
        details={
            "winner": winner_detail,
            "canonical_pretrain": str(pretrain_dir),
            "canonical_finetune": str(finetune_dir),
            "pipeline_status_observation": status_observation,
            "terminal_authority": {
                "experiment_contract_sha256": expected_contract_sha256,
                "completion_receipt_contract_sha256": completion_receipt[
                    "experiment_contract_sha256"
                ],
                "selection": {
                    "path": str(selection_path),
                    "sha256": _sha256_bytes(selection_content),
                },
                "test_capability": {
                    "path": str(capability_path),
                    "sha256": _sha256_bytes(capability_content),
                },
                "test_consumed_marker": {
                    "path": str(consumed_path),
                    "sha256": _sha256_bytes(consumed_content),
                },
                "test_completed_marker": {
                    "path": str(completed_path),
                    "sha256": _sha256_bytes(completed_content),
                },
                "val_metrics_sha256": _sha256_bytes(val_content),
                "test_metrics_sha256": _sha256_bytes(test_content),
                "completion_report": {
                    "path": str(completion_report_path),
                    "sha256": _sha256_bytes(completion_content),
                },
            },
            "scope": (
                "Stage-1 150-1600 Hz training/G4 pipeline complete; "
                "deployment remains separate"
            ),
        },
    )


def _inspect_post_bootstrap(
    contract: CampaignContract,
    repo_root: Path,
    local: dict[str, Any],
    *,
    resume_path: Path | None,
    resume_sha256: str | None,
) -> Inspection:
    modules = _lazy_imports(repo_root)
    # Public snapshot helper is deliberately added after import so checkpoint reads
    # use the same same-FD primitive as the existing validators.
    from deep_anc.train.evaluation_contract import snapshot_regular_file

    modules["snapshot_regular_file"] = snapshot_regular_file
    bootstrap_sha = str(local["bootstrap"]["sha256"])
    canonical_cfgs: dict[str, dict[str, Any]] = {}
    pilot_cfgs: dict[str, dict[str, Any]] = {}
    probe_cfgs: dict[str, dict[str, Any]] = {}
    manifest_paths: set[Path] = set()
    candidate_details: list[dict[str, Any]] = []
    try:
        for candidate in contract.candidates:
            # G0/pilot/probe selection evidence는 docs/05의 고정 primary seed에서만
            # 발행한다. second-seed contract는 이 raw winner를 재사용하되 새 후보
            # selection campaign으로 위장하지 않는다.
            canonical = _canonical_cfg(
                modules, contract, bootstrap_sha, candidate, seed=20260803
            )
            pilot = _derivative_cfg(
                modules, contract, bootstrap_sha, candidate, role="loss_pilot"
            )
            pilot_dir = repo_root / str(pilot["ckpt_dir"])
            probe = _derivative_cfg(
                modules,
                contract,
                bootstrap_sha,
                candidate,
                role="measured_probe",
                init_ckpt=pilot_dir / "ckpt/best.pt",
            )
            manifest_value = modules["canonical_recorded_manifest_for_data"](
                canonical.get("data") or {}
            )
            manifest = Path(manifest_value)
            if not manifest.is_absolute():
                manifest = repo_root / manifest
            canonical_cfgs[candidate.key] = canonical
            pilot_cfgs[candidate.key] = pilot
            probe_cfgs[candidate.key] = probe
            manifest_paths.add(manifest.absolute())
            candidate_details.append(
                {
                    "key": candidate.key,
                    "identity": list(candidate.identity),
                    "pilot_run": str(pilot_dir),
                    "probe_run": str(repo_root / str(probe["ckpt_dir"])),
                }
            )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return _blocked(
            "bootstrap_validation",
            "BOOTSTRAP_OR_CANONICAL_CONFIG_INVALID",
            str(exc),
            local=local,
        )
    if len(manifest_paths) != 1:
        return _blocked(
            "candidate_binding",
            "RECORDED_MANIFEST_DIVERGED",
            f"candidate별 canonical recorded manifest가 갈라졌습니다: {sorted(map(str, manifest_paths))}",
            candidates=candidate_details,
        )
    manifest = next(iter(manifest_paths))

    # 0) docs/05의 G0 admission: canonical init만 비운 measured readiness가
    # 정확히 16/17이어야 GPU selection evidence를 시작한다. G0 CLI 자체에는 이
    # 선행 receipt 인자가 없으므로 state machine이 비용 발생 전에 강제한다.
    pre_candidate = contract.candidates[0]
    try:
        pre_readiness_cfg = modules["load_train_config"](
            repo_root / CANONICAL_FINETUNE_CONFIG,
            [
                "data.digital_primary_path_mode=measured",
                "init_ckpt=null",
                f"data.bootstrap_receipt_sha256={bootstrap_sha}",
                f"loss.nmse_cvar_alpha={pre_candidate.alpha_text}",
                f"loss.lambda_dnh={pre_candidate.lambda_text}",
            ],
        )
        pre_run_dir = repo_root / str(pre_readiness_cfg["ckpt_dir"])
        pre_readiness_dir = (
            modules["autostart_state_dir"](pre_run_dir) / "pre_g0_readiness"
        )
        pre_readiness_path = pre_readiness_dir / "readiness.json"
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return _blocked(
            "pre_g0_readiness",
            "PRE_G0_READINESS_CONFIG_INVALID",
            str(exc),
            local=local,
        )
    if not pre_readiness_path.exists() and not pre_readiness_path.is_symlink():
        return _primary_evidence_transition(contract, _ready(
            "pre_g0_readiness",
            "pre_g0_readiness",
            _pre_g0_readiness_command(
                repo_root,
                bootstrap_sha=bootstrap_sha,
                candidate=pre_candidate,
                out_dir=pre_readiness_dir,
            ),
            expected_gate="16/17 with completed_init_checkpoint as the only failure",
            readiness_report=str(pre_readiness_path),
        ))
    try:
        _regular_bytes(pre_readiness_path, label="pre-G0 readiness report")
        pre_readiness = modules["audit_finetune_readiness"](pre_readiness_cfg)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError, CampaignError) as exc:
        return _blocked(
            "pre_g0_readiness",
            "PRE_G0_READINESS_INVALID",
            str(exc),
            readiness_report=str(pre_readiness_path),
        )
    pre_failed = [
        {"id": str(row.get("id")), "message": str(row.get("message"))}
        for row in pre_readiness.get("checks", [])
        if not bool(row.get("ok"))
    ]
    if [row["id"] for row in pre_failed] != ["completed_init_checkpoint"]:
        return _blocked(
            "pre_g0_readiness",
            "PRE_G0_READINESS_NOT_16_OF_17",
            "G0 전 readiness는 completed_init_checkpoint 하나만 FAIL이어야 합니다: "
            f"actual={[row['id'] for row in pre_failed]}",
            failed_checks=pre_failed,
            readiness_report=str(pre_readiness_path),
        )

    # 1) 모든 G0를 먼저 완성·검증한다. 한 후보를 pilot까지 보낸 뒤 다른 후보의
    # fixed batch mismatch를 발견해 GPU 시간을 낭비하지 않는다.
    g0_rows: dict[str, dict[str, Any]] = {}
    for candidate in contract.candidates:
        paths = _candidate_paths(candidate, repo_root)
        receipt = paths["g0_receipt"]
        if not receipt.exists() and not receipt.is_symlink():
            return _primary_evidence_transition(contract, _ready(
                "g0",
                "g0",
                build_g0_command(contract, candidate, bootstrap_sha, repo_root),
                candidate=candidate.key,
                candidates=candidate_details,
            ))
        try:
            payload = _json_object(_regular_bytes(receipt, label="G0 receipt"), label="G0 receipt")
            if payload.get("kind") == modules["FAILED_G0_RECEIPT_KIND"]:
                return _blocked(
                    "g0",
                    "ADAPTIVE_LAMBDA_REQUIRED",
                    "G0가 -6 dB gate를 통과하지 못했습니다. 기존 weight를 전이하지 말고 "
                    "필요하면 failed-gradient recommendation을 별도로 만든 뒤 새 lambda의 "
                    "외부 contract로 G0부터 다시 실행하세요.",
                    candidate=candidate.key,
                    failed_receipt=str(receipt),
                )
            ref = _reference(modules, repo_root, receipt, label="campaign G0 receipt")
            row = modules["validate_g0_receipt"](
                ref,
                repo_root=repo_root,
                canonical_cfg=canonical_cfgs[candidate.key],
                canonical_contract=canonical_cfgs[candidate.key]["experiment_contract"],
                expected_identity=candidate.identity,
            )
            g0_rows[candidate.key] = row
        except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError, CampaignError) as exc:
            return _blocked(
                "g0",
                "INVALID_G0_ARTIFACT",
                str(exc),
                candidate=candidate.key,
                receipt=str(receipt),
            )
    batch_shas = {str(row["batch"].sha256) for row in g0_rows.values()}
    if len(batch_shas) != 1:
        return _blocked(
            "g0",
            "G0_FIXED_BATCH_MISMATCH",
            f"candidate G0 fixed batch SHA가 다릅니다: {sorted(batch_shas)}",
        )

    # 2) 모든 alpha별 pre-pilot output-y gradient를 먼저 승인한다.
    gradient_rows: dict[str, dict[str, Any]] = {}
    for candidate in contract.candidates:
        paths = _candidate_paths(candidate, repo_root)
        receipt = paths["gradient_receipt"]
        if not receipt.exists() and not receipt.is_symlink():
            return _primary_evidence_transition(contract, _ready(
                "prepilot_gradient",
                "prepilot_gradient",
                build_gradient_command(candidate, repo_root),
                candidate=candidate.key,
            ))
        try:
            g0_ref = _reference(
                modules, repo_root, paths["g0_receipt"], label="candidate G0 receipt"
            )
            gradient_ref = _reference(
                modules, repo_root, receipt, label="prepilot gradient receipt"
            )
            gradient_rows[candidate.key] = modules["validate_prepilot_gradient_receipt"](
                gradient_ref,
                repo_root=repo_root,
                canonical_cfg=canonical_cfgs[candidate.key],
                canonical_contract=canonical_cfgs[candidate.key]["experiment_contract"],
                g0_receipt_reference=g0_ref,
                expected_identity=candidate.identity,
            )
        except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return _blocked(
                "prepilot_gradient",
                "ADAPTIVE_LAMBDA_REQUIRED",
                f"pre-pilot gradient gate가 현재 lambda를 승인하지 않았습니다: {exc}",
                candidate=candidate.key,
                receipt=str(receipt),
            )

    # 3) 모든 20k pilot을 먼저 닫은 뒤, 모든 best.pt의 canonical recorded val을
    # 별도 단계로 닫는다. 상태 JSON의 stage order와 실제 실행 순서를 같게 유지한다.
    pilot_details: dict[str, dict[str, Any]] = {}
    for candidate in contract.candidates:
        pilot_cfg = pilot_cfgs[candidate.key]
        run_dir = repo_root / str(pilot_cfg["ckpt_dir"])
        try:
            progress, detail = _run_progress(modules, run_dir, 20_000)
        except CampaignError as exc:
            return _blocked("loss_pilot", "INVALID_PILOT_RUN", str(exc), candidate=candidate.key)
        overrides = _candidate_overrides(
            candidate, bootstrap_sha, role="loss_pilot"
        )
        train_command = build_train_command(
            repo_root, config=CANONICAL_PRETRAIN_CONFIG, overrides=overrides
        )
        if progress == "absent":
            return _primary_evidence_transition(contract, _ready(
                "loss_pilot",
                "loss_pilot",
                train_command,
                candidate=candidate.key,
                run=detail,
            ))
        if progress == "partial":
            return _primary_evidence_transition(contract, _explicit_resume(
                action="loss_pilot",
                phase="loss_pilot",
                base_command=train_command,
                expected_last=run_dir / "ckpt/last.pt",
                resume_path=resume_path,
                resume_sha256=resume_sha256,
                details={"candidate": candidate.key, "run": detail},
            ))
        pilot_details[candidate.key] = detail

    validated_pilots: dict[str, dict[str, Any]] = {}
    for candidate in contract.candidates:
        pilot_cfg = pilot_cfgs[candidate.key]
        run_dir = repo_root / str(pilot_cfg["ckpt_dir"])
        detail = pilot_details[candidate.key]
        eval_dir = run_dir / "eval_recorded_val"
        metrics = eval_dir / "metrics.npz"
        if not metrics.exists() and not metrics.is_symlink():
            if eval_dir.exists() or eval_dir.is_symlink():
                return _blocked(
                    "loss_pilot_val",
                    "PARTIAL_PILOT_VAL_ARTIFACT",
                    f"no-replace eval directory는 있으나 metrics.npz가 없습니다: {eval_dir}",
                    candidate=candidate.key,
                )
            return _primary_evidence_transition(contract, _ready(
                "loss_pilot_val",
                "loss_pilot_val",
                build_recorded_val_command(
                    repo_root,
                    checkpoint=run_dir / "ckpt/best.pt",
                    manifest=manifest,
                    output=eval_dir,
                    allow_surrogate=True,
                ),
                candidate=candidate.key,
                run=detail,
            ))
        try:
            pilot_row = {
                "best_checkpoint": _reference(
                    modules,
                    repo_root,
                    run_dir / "ckpt/best.pt",
                    label="pilot best",
                ),
                "last_checkpoint": _reference(
                    modules,
                    repo_root,
                    run_dir / "ckpt/last.pt",
                    label="pilot last",
                ),
                "metrics": _reference(
                    modules, repo_root, metrics, label="pilot val metrics"
                ),
                "manifest": _reference(
                    modules, repo_root, manifest, label="recorded manifest"
                ),
            }
            validated_pilots[candidate.key] = modules[
                "validate_loss_pilot_candidate"
            ](
                pilot_row,
                repo_root=repo_root,
                canonical_cfg=canonical_cfgs[candidate.key],
                canonical_contract=canonical_cfgs[candidate.key][
                    "experiment_contract"
                ],
                expected_identity=candidate.identity,
                label=f"loss pilot {candidate.key}",
            )
        except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return _blocked(
                "loss_pilot_val",
                "INVALID_PILOT_VAL_ARTIFACT",
                str(exc),
                candidate=candidate.key,
                metrics=str(metrics),
            )

    # 4) 모든 20k best를 exact init으로 한 5k measured probe를 먼저 닫은 뒤,
    # 모든 measured recorded val을 별도 단계로 닫는다.
    probe_details: dict[str, dict[str, Any]] = {}
    for candidate in contract.candidates:
        pilot_dir = repo_root / str(pilot_cfgs[candidate.key]["ckpt_dir"])
        probe_cfg = probe_cfgs[candidate.key]
        run_dir = repo_root / str(probe_cfg["ckpt_dir"])
        init_rel = _relative(pilot_dir / "ckpt/best.pt", repo_root)
        overrides = _candidate_overrides(
            candidate,
            bootstrap_sha,
            role="measured_probe",
            init_ckpt=init_rel,
        )
        train_command = build_train_command(
            repo_root, config=CANONICAL_PRETRAIN_CONFIG, overrides=overrides
        )
        try:
            progress, detail = _run_progress(modules, run_dir, 5_000)
        except CampaignError as exc:
            return _blocked(
                "measured_probe", "INVALID_MEASURED_PROBE_RUN", str(exc), candidate=candidate.key
            )
        if progress == "absent":
            return _primary_evidence_transition(contract, _ready(
                "measured_probe",
                "measured_probe",
                train_command,
                candidate=candidate.key,
                init_checkpoint=str(pilot_dir / "ckpt/best.pt"),
                run=detail,
            ))
        if progress == "partial":
            return _primary_evidence_transition(contract, _explicit_resume(
                action="measured_probe",
                phase="measured_probe",
                base_command=train_command,
                expected_last=run_dir / "ckpt/last.pt",
                resume_path=resume_path,
                resume_sha256=resume_sha256,
                details={"candidate": candidate.key, "run": detail},
            ))
        probe_details[candidate.key] = detail

    validated_probes: dict[str, dict[str, Any]] = {}
    for candidate in contract.candidates:
        pilot_dir = repo_root / str(pilot_cfgs[candidate.key]["ckpt_dir"])
        probe_cfg = probe_cfgs[candidate.key]
        run_dir = repo_root / str(probe_cfg["ckpt_dir"])
        detail = probe_details[candidate.key]
        eval_dir = run_dir / "eval_recorded_val"
        metrics = eval_dir / "metrics.npz"
        if not metrics.exists() and not metrics.is_symlink():
            if eval_dir.exists() or eval_dir.is_symlink():
                return _blocked(
                    "measured_probe_val",
                    "PARTIAL_PROBE_VAL_ARTIFACT",
                    f"no-replace eval directory는 있으나 metrics.npz가 없습니다: {eval_dir}",
                    candidate=candidate.key,
                )
            return _primary_evidence_transition(contract, _ready(
                "measured_probe_val",
                "measured_probe_val",
                build_recorded_val_command(
                    repo_root,
                    checkpoint=run_dir / "ckpt/best.pt",
                    manifest=manifest,
                    output=eval_dir,
                    allow_surrogate=False,
                ),
                candidate=candidate.key,
                run=detail,
            ))
        try:
            probe_row = {
                "best_checkpoint": _reference(
                    modules,
                    repo_root,
                    run_dir / "ckpt/best.pt",
                    label="probe best",
                ),
                "last_checkpoint": _reference(
                    modules,
                    repo_root,
                    run_dir / "ckpt/last.pt",
                    label="probe last",
                ),
                "metrics": _reference(
                    modules, repo_root, metrics, label="probe val metrics"
                ),
                "manifest": _reference(
                    modules, repo_root, manifest, label="recorded manifest"
                ),
                "init_checkpoint": _reference(
                    modules,
                    repo_root,
                    pilot_dir / "ckpt/best.pt",
                    label="probe init",
                ),
            }
            validated_probes[candidate.key] = modules["validate_measured_probe"](
                probe_row,
                repo_root=repo_root,
                canonical_cfg=canonical_cfgs[candidate.key],
                canonical_contract=canonical_cfgs[candidate.key][
                    "experiment_contract"
                ],
                expected_identity=candidate.identity,
                expected_init_checkpoint_sha256=validated_pilots[candidate.key][
                    "best_snapshot"
                ].sha256,
            )
        except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return _blocked(
                "measured_probe_val",
                "INVALID_PROBE_VAL_ARTIFACT",
                str(exc),
                candidate=candidate.key,
                metrics=str(metrics),
            )

    # 5) raw checkpoint/metrics/manifest validator로만 winner를 유도한다.
    chains: list[dict[str, Any]] = []
    try:
        for candidate in contract.candidates:
            paths = _candidate_paths(candidate, repo_root)
            chains.append(
                {
                    "identity": candidate.identity,
                    "score_db": float(validated_probes[candidate.key]["score_db"]),
                    "selection_score_source": modules[
                        "MEASURED_PROBE_SELECTION_SCORE"
                    ],
                    "g0": g0_rows[candidate.key],
                    "gradient_calibration": gradient_rows[candidate.key],
                    "pilot": validated_pilots[candidate.key],
                    "measured_probe": validated_probes[candidate.key],
                    "paths": {
                        "pilot_dir": repo_root
                        / str(pilot_cfgs[candidate.key]["ckpt_dir"]),
                        "probe_dir": repo_root
                        / str(probe_cfgs[candidate.key]["ckpt_dir"]),
                        "g0_receipt": paths["g0_receipt"],
                        "gradient_receipt": paths["gradient_receipt"],
                    },
                }
            )
        selection = modules["select_loss_pilot"](chains)
    except ValueError as exc:
        text = str(exc)
        if "alpha=0.85" in text:
            return _blocked(
                "winner_selection",
                "ADAPTIVE_ALPHA_085_REQUIRED",
                "0.7/1.0 measured-probe raw margin이 0.2 dB 이내입니다. alpha 0.85와 "
                "그 alpha의 G0-approved lambda를 포함한 새 외부 contract가 필요합니다. "
                f"validator={text}",
                candidate_scores=[
                    {"identity": list(row["identity"]), "score_db": row["score_db"]}
                    for row in chains
                ],
            )
        return _blocked(
            "winner_selection",
            "RAW_WINNER_VALIDATION_FAILED",
            text,
        )
    winner_identity = tuple(float(value) for value in selection["winner_identity"])
    winner = next(
        candidate for candidate in contract.candidates if candidate.identity == winner_identity
    )
    winner_chain = next(row for row in chains if tuple(row["identity"]) == winner_identity)
    winner_detail = {
        "candidate": winner.key,
        "identity": list(winner.identity),
        "base_gap_db": float(selection["base_gap_db"]),
        "used_alpha_085": bool(selection["used_alpha_085"]),
        "scores": [
            {"identity": list(row["identity"]), "score_db": float(row["score_db"])}
            for row in chains
        ],
    }

    # 6) winner 20k의 drift gradient는 winner G0의 concrete batch path를 재사용한다.
    winner_paths = _candidate_paths(winner, repo_root)
    selected_gradient = winner_paths["selected_gradient_receipt"]
    winner_pilot = Path(winner_chain["paths"]["pilot_dir"])
    if not selected_gradient.exists() and not selected_gradient.is_symlink():
        return _primary_evidence_transition(contract, _ready(
            "selected20k_gradient",
            "selected20k_gradient",
            build_gradient_command(
                winner, repo_root, selected_pilot=winner_pilot / "ckpt/best.pt"
            ),
            winner=winner_detail,
        ))
    try:
        gradient_ref = _reference(
            modules, repo_root, selected_gradient, label="selected20k gradient receipt"
        )
        modules["validate_gradient_budget_receipt"](
            gradient_ref,
            repo_root=repo_root,
            canonical_cfg=canonical_cfgs[winner.key],
            canonical_contract=canonical_cfgs[winner.key]["experiment_contract"],
            expected_checkpoint_sha256=winner_chain["pilot"]["best_snapshot"].sha256,
            expected_identity=winner.identity,
            expected_batch_path=winner_chain["g0"]["batch"].path,
            expected_batch_sha256=winner_chain["g0"]["batch"].sha256,
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return _blocked(
            "selected20k_gradient",
            "SELECTED20K_GRADIENT_FAILED",
            str(exc),
            winner=winner_detail,
        )

    # second seed는 어떤 GPU 작업(smoke 포함)보다 먼저 primary ledger→100k→50k→
    # raw borderline selection을 sealed external linkage로 재검증한다. fixed primary
    # evidence pathname을 secondary가 발행/재개하는 경로도 위에서 이미 닫았다.
    ledger_path = repo_root / CANONICAL_LEDGER
    second_seed_context: dict[str, Any] | None = None
    ledger_sha: str | None = None
    if contract.second_seed is not None:
        if not ledger_path.exists() and not ledger_path.is_symlink():
            return _blocked(
                "second_seed_admission",
                "SECOND_SEED_PRIMARY_EVIDENCE_REQUIRED",
                f"primary canonical ledger가 없습니다: {ledger_path}",
                winner=winner_detail,
            )
        try:
            ledger_sha = _sha256_bytes(
                _regular_bytes(ledger_path, label="primary canonical campaign ledger")
            )
            second_seed_context = _validate_second_seed_primary_context(
                modules,
                contract,
                repo_root=repo_root,
                bootstrap_sha=bootstrap_sha,
                ledger_sha=ledger_sha,
                winner=winner,
            )
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            CampaignError,
        ) as exc:
            return _blocked(
                "second_seed_admission",
                "SECOND_SEED_PRIMARY_LINK_INVALID",
                f"second-seed GPU 실행 전 sealed primary borderline chain 검증 실패: {exc}",
                winner=winner_detail,
                primary_contract={
                    "path": str(contract.second_seed.primary_contract_path),
                    "sha256": contract.second_seed.primary_contract_sha256,
                },
            )

    # 7) winner identity의 uninterrupted vs stop+resume A100 smoke. secondary는
    # primary smoke를 재사용하지 않고 seed=20260903 target을 fresh 실행한다.
    try:
        smoke_cfg = _smoke_cfg(
            modules, bootstrap_sha, winner, seed=contract.seed
        )
        target = modules["build_a100_pretrain_smoke_target"](
            smoke_cfg, repo_root=repo_root
        )["sha256"]
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return _blocked("resume_smoke", "SMOKE_TARGET_INVALID", str(exc), winner=winner_detail)
    smoke_root = repo_root / str(modules["SMOKE_ROOT"]) / str(target)
    smoke_receipt = smoke_root / "receipt.json"
    if not smoke_receipt.exists() and not smoke_receipt.is_symlink():
        if smoke_root.exists() or smoke_root.is_symlink():
            return _blocked(
                "resume_smoke",
                "PARTIAL_SMOKE_TARGET",
                "smoke target root가 이미 있으나 terminal receipt가 없습니다. 기존 partial "
                "artifact를 덮어쓰거나 자동 삭제하지 않습니다.",
                target=str(target),
                smoke_root=str(smoke_root),
                winner=winner_detail,
            )
        command = [
            str(_python(repo_root)),
            str((repo_root / "scripts/train/run_a100_pretrain_smoke.py").absolute()),
            "--config",
            CANONICAL_PRETRAIN_CONFIG,
            "--bootstrap-receipt-sha256",
            bootstrap_sha,
            "--stop-step",
            "300",
            "--final-step",
            "500",
            "--loss-alpha",
            winner.alpha_text,
            "--loss-lambda-dnh",
            winner.lambda_text,
            "--seed",
            str(contract.seed),
            "--cublas-workspace-config",
            contract.cublas_workspace_config,
        ]
        return _ready(
            "resume_smoke",
            "resume_smoke",
            command,
            smoke_target=target,
            winner=winner_detail,
        )
    try:
        smoke_payload = _json_object(
            _regular_bytes(smoke_receipt, label="A100 smoke receipt"),
            label="A100 smoke receipt",
        )
        modules["validate_a100_pretrain_smoke_receipt"](
            smoke_payload,
            repo_root=repo_root,
            expected_smoke_target_sha256=str(target),
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError, CampaignError) as exc:
        return _blocked("resume_smoke", "INVALID_SMOKE_RECEIPT", str(exc), winner=winner_detail)

    # 8) primary는 raw winner ledger를 no-replace 발행한다. secondary는 그 fixed
    # ledger를 수정하지 않고, 이미 검증한 primary chain + fresh secondary smoke를
    # 별도 seed-specific prerequisite로 no-replace 발행한다.
    if contract.second_seed is None:
        if not ledger_path.exists() and not ledger_path.is_symlink():
            return _ready(
                "campaign_ledger",
                "issue_campaign_ledger",
                _issuer_command(
                    bootstrap_sha,
                    winner,
                    chains,
                    smoke_root,
                    repo_root,
                ),
                winner=winner_detail,
                smoke_target=target,
            )
        try:
            ledger_sha = _sha256_bytes(
                _regular_bytes(ledger_path, label="canonical campaign ledger")
            )
            final_cfg = _canonical_cfg(
                modules,
                contract,
                bootstrap_sha,
                winner,
                ledger_sha=ledger_sha,
                seed=20260803,
            )
            modules["validate_canonical_pretrain_prerequisites"](
                final_cfg, repo_root=repo_root
            )
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            CampaignError,
        ) as exc:
            return _blocked(
                "campaign_ledger",
                "INVALID_CANONICAL_LEDGER",
                str(exc),
                winner=winner_detail,
            )
    else:
        assert ledger_sha is not None and second_seed_context is not None
        link = contract.second_seed
        primary_selection = second_seed_context["primary_selection"]
        try:
            # Prospective payload의 SHA/path를 issuer와 같은 raw builder로 유도한다.
            # prerequisite 자체는 smoke target projection에서 제외되므로 zero anchor
            # provisional cfg와 published-anchor cfg가 같은 payload를 만든다.
            provisional_path = modules["second_seed_prerequisite_path"](
                link.seed_neutral_campaign_sha256, repo_root=repo_root
            )
            provisional_cfg = _canonical_cfg(
                modules,
                contract,
                bootstrap_sha,
                winner,
                ledger_sha=ledger_sha,
                seed=20260903,
                second_seed_prerequisite=_relative(provisional_path, repo_root),
                second_seed_prerequisite_sha256="0" * 64,
            )
            prerequisite_payload = modules["build_second_seed_prerequisite_payload"](
                provisional_cfg,
                primary_selection=primary_selection["path"],
                smoke_receipt=smoke_root / "receipt.json",
                smoke_environment_receipt=smoke_root / "environment_receipt.json",
                smoke_telemetry=smoke_root / "telemetry.json",
                repo_root=repo_root,
            )
            prerequisite_sha = modules["prerequisite_sha256"](
                prerequisite_payload
            )
            prerequisite_path = modules["second_seed_prerequisite_path"](
                prerequisite_payload["primary"]["seed_neutral_campaign_sha256"],
                repo_root=repo_root,
            )
            if prerequisite_path.absolute() != provisional_path.absolute():
                raise CampaignError("second-seed prerequisite fixed path 유도가 갈라졌습니다")
            if not prerequisite_path.exists() and not prerequisite_path.is_symlink():
                return _ready(
                    "second_seed_prerequisite",
                    "issue_second_seed_prerequisite",
                    _second_seed_issuer_command(
                        repo_root=repo_root,
                        bootstrap_sha=bootstrap_sha,
                        ledger_sha=ledger_sha,
                        winner=winner,
                        primary_selection=primary_selection["path"],
                        smoke_root=smoke_root,
                        destination=prerequisite_path,
                    ),
                    winner=winner_detail,
                    prerequisite={
                        "path": str(prerequisite_path),
                        "prospective_sha256": prerequisite_sha,
                    },
                )
            published = _regular_bytes(
                prerequisite_path, label="second-seed prerequisite"
            )
            if _sha256_bytes(published) != prerequisite_sha:
                raise CampaignError(
                    "published second-seed prerequisite가 현재 raw primary/smoke에서 "
                    "유도한 prospective bytes와 다릅니다"
                )
            final_cfg = _canonical_cfg(
                modules,
                contract,
                bootstrap_sha,
                winner,
                ledger_sha=ledger_sha,
                seed=20260903,
                second_seed_prerequisite=_relative(prerequisite_path, repo_root),
                second_seed_prerequisite_sha256=prerequisite_sha,
            )
            modules["validate_second_seed_prerequisites"](
                final_cfg, repo_root=repo_root
            )
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            CampaignError,
        ) as exc:
            return _blocked(
                "second_seed_prerequisite",
                "SECOND_SEED_PREREQUISITE_INVALID",
                str(exc),
                winner=winner_detail,
            )

    assert ledger_sha is not None
    try:
        pretrain_contract_sha = _current_experiment_contract_sha(
            modules, final_cfg, label="canonical pretrain"
        )
    except (KeyError, RuntimeError, TypeError, ValueError, CampaignError) as exc:
        return _blocked(
            "campaign_ledger",
            "CANONICAL_PRETRAIN_CONTRACT_INVALID",
            str(exc),
            winner=winner_detail,
        )

    # 9) canonical 100k from scratch. Partial state requires exact explicit resume.
    pretrain_dir = repo_root / str(final_cfg["ckpt_dir"])
    pretrain_overrides = [
        f"data.bootstrap_receipt_sha256={bootstrap_sha}",
        f"campaign_prerequisite_sha256={ledger_sha}",
        f"loss.nmse_cvar_alpha={winner.alpha_text}",
        f"loss.lambda_dnh={winner.lambda_text}",
        f"seed={contract.seed}",
    ]
    if contract.second_seed is not None:
        pretrain_overrides.extend(
            [
                "second_seed_prerequisite="
                f"{json.dumps(str(final_cfg['second_seed_prerequisite']))}",
                "second_seed_prerequisite_sha256="
                f"{json.dumps(str(final_cfg['second_seed_prerequisite_sha256']))}",
            ]
        )
    pretrain_command = build_train_command(
        repo_root, config=CANONICAL_PRETRAIN_CONFIG, overrides=pretrain_overrides
    )
    try:
        progress, pretrain_detail = _run_progress(modules, pretrain_dir, 100_000)
    except CampaignError as exc:
        return _blocked("canonical_pretrain", "INVALID_PRETRAIN_RUN", str(exc), winner=winner_detail)
    if progress == "absent":
        return _ready(
            "canonical_pretrain",
            "canonical_pretrain",
            pretrain_command,
            winner=winner_detail,
            run=pretrain_detail,
        )
    if progress == "partial":
        return _explicit_resume(
            action="canonical_pretrain_resume",
            phase="canonical_pretrain",
            base_command=pretrain_command,
            expected_last=pretrain_dir / "ckpt/last.pt",
            resume_path=resume_path,
            resume_sha256=resume_sha256,
            details={"winner": winner_detail, "run": pretrain_detail},
        )
    try:
        _validated_completion_receipt(
            modules,
            pretrain_dir / "ckpt",
            expected_role="canonical_pretrain",
            expected_init_eligible=True,
            expected_contract_sha256=pretrain_contract_sha,
            repo_root=repo_root,
        )
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        CampaignError,
    ) as exc:
        return _blocked(
            "canonical_pretrain",
            "PRETRAIN_COMPLETION_RECEIPT_REQUIRED",
            str(exc),
            winner=winner_detail,
            run=pretrain_detail,
        )

    # 10) readiness report stage, then fresh recomputation before 50k.
    init_checkpoint = pretrain_dir / "ckpt/best.pt"
    finetune_overrides = _finetune_overrides(
        bootstrap_sha=bootstrap_sha,
        winner=winner,
        init_checkpoint=init_checkpoint,
        seed=contract.seed,
    )
    try:
        finetune_cfg = modules["load_train_config"](
            repo_root / CANONICAL_FINETUNE_CONFIG, finetune_overrides
        )
        finetune_contract_sha = _current_experiment_contract_sha(
            modules, finetune_cfg, label="canonical fine-tune"
        )
        if contract.second_seed is not None:
            finetune_neutral = modules["seed_neutral_campaign_sha256"](
                finetune_cfg
            )
            if finetune_neutral != contract.second_seed.seed_neutral_campaign_sha256:
                raise CampaignError(
                    "second-seed 50k resolved config가 sealed seed-neutral linkage와 다릅니다"
                )
        finetune_dir = repo_root / str(finetune_cfg["ckpt_dir"])
        readiness_dir = modules["autostart_state_dir"](finetune_dir) / "audit"
        readiness_path = readiness_dir / "readiness.json"
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return _blocked("finetune_readiness", "FINETUNE_CONFIG_INVALID", str(exc), winner=winner_detail)
    if not readiness_path.exists() and not readiness_path.is_symlink():
        return _ready(
            "finetune_readiness",
            "finetune_readiness",
            _readiness_command(
                repo_root,
                bootstrap_sha=bootstrap_sha,
                winner=winner,
                init_checkpoint=init_checkpoint,
                seed=contract.seed,
            ),
            winner=winner_detail,
            readiness_report=str(readiness_path),
        )
    try:
        # Existing report is only a durable observation. Authority is a fresh audit of
        # current P/S, transfer, recorded bytes and init completion.
        _regular_bytes(readiness_path, label="fine-tune readiness report")
        readiness = modules["audit_finetune_readiness"](finetune_cfg)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError, CampaignError) as exc:
        return _blocked("finetune_readiness", "READINESS_AUDIT_INVALID", str(exc), winner=winner_detail)
    if not bool(readiness.get("ok")):
        failed = [
            {"id": str(row.get("id")), "message": str(row.get("message"))}
            for row in readiness.get("checks", [])
            if not bool(row.get("ok"))
        ]
        return _blocked(
            "finetune_readiness",
            "READINESS_NOT_17_OF_17",
            f"fine-tune readiness가 17/17 PASS가 아닙니다: {[row['id'] for row in failed]}",
            winner=winner_detail,
            failed_checks=failed,
            readiness_report=str(readiness_path),
        )

    # 11) canonical 50k pipeline. It re-runs readiness and handles val/test authority.
    try:
        progress, finetune_detail = _run_progress(modules, finetune_dir, 50_000)
    except CampaignError as exc:
        return _blocked("canonical_finetune", "INVALID_FINETUNE_RUN", str(exc), winner=winner_detail)
    pipeline_command = _finetune_pipeline_command(
        repo_root,
        bootstrap_sha=bootstrap_sha,
        winner=winner,
        init_checkpoint=init_checkpoint,
        seed=contract.seed,
    )
    if progress == "absent":
        return _ready(
            "canonical_finetune",
            "canonical_finetune",
            pipeline_command,
            winner=winner_detail,
            run=finetune_detail,
        )
    if progress == "partial":
        return _explicit_resume(
            action="canonical_finetune_resume",
            phase="canonical_finetune",
            base_command=pipeline_command,
            expected_last=finetune_dir / "ckpt/last.pt",
            resume_path=resume_path,
            resume_sha256=resume_sha256,
            details={"winner": winner_detail, "run": finetune_detail},
        )
    state_root = modules["autostart_state_dir"](finetune_dir)
    selection_path = state_root / "audit/recorded_val_selection.json"
    try:
        _validated_completion_receipt(
            modules,
            finetune_dir / "ckpt",
            expected_role="canonical_finetune",
            expected_init_eligible=False,
            expected_contract_sha256=finetune_contract_sha,
            repo_root=repo_root,
        )
        current_selection = _validated_seed_selection(
            modules,
            selection_path,
            expected_seed=contract.seed,
            expected_seed_neutral_sha256=(
                contract.second_seed.seed_neutral_campaign_sha256
                if contract.second_seed is not None
                else None
            ),
            expected_experiment_contract_sha256=finetune_contract_sha,
        )
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        CampaignError,
    ) as exc:
        return _blocked(
            "canonical_finetune",
            "FINETUNE_VAL_SELECTION_AUTHORITY_REQUIRED",
            f"50k completion 뒤 raw recorded-val selection 검증이 필요합니다: {exc}",
            winner=winner_detail,
            run=finetune_detail,
        )

    if contract.second_seed is None:
        decision_status = current_selection["decision"].get("status")
        if decision_status == "borderline":
            return _blocked(
                "cross_seed_finalize",
                "SECOND_SEED_SEALED_CONTRACT_REQUIRED",
                "seed 20260803 raw val 결과가 numeric/CI borderline입니다. test를 "
                "열지 않고 아래 exact linkage를 넣은 외부 schema-v2 contract로 "
                "seed 20260903 fresh smoke→100k→50k를 실행해야 합니다.",
                winner=winner_detail,
                primary_contract={
                    "path": str(contract.path),
                    "sha256": contract.sha256,
                },
                required_secondary_campaign={
                    "seed": 20260903,
                    "second_seed": {
                        "primary_contract_path": str(contract.path),
                        "primary_contract_sha256": contract.sha256,
                        "primary_selection_sha256": current_selection["sha256"],
                        "seed_neutral_campaign_sha256": current_selection[
                            "seed_neutral_campaign_sha256"
                        ],
                    },
                },
                primary_selection={
                    "path": str(selection_path),
                    "sha256": current_selection["sha256"],
                    "decision": current_selection["decision"],
                },
            )
        if decision_status == "inconclusive_data":
            return _blocked(
                "canonical_finetune",
                "TARGETED_RECORDING_REQUIRED",
                "recorded-val data coverage가 불충분합니다. second seed가 아니라 raw "
                "decision이 지정한 family/subband 추가 수집 뒤 새 generation이 필요합니다.",
                decision=current_selection["decision"],
                selection={
                    "path": str(selection_path),
                    "sha256": current_selection["sha256"],
                },
            )
        if decision_status != "clear_pass":
            return _blocked(
                "canonical_finetune",
                "PRIMARY_VAL_NOT_DEPLOYABLE",
                f"primary raw val status={decision_status!r}는 test를 열 수 없습니다.",
                decision=current_selection["decision"],
            )
        return _inspect_finetune_terminal_authority(
            modules,
            repo_root=repo_root,
            finetune_cfg=finetune_cfg,
            finetune_dir=finetune_dir,
            state_root=state_root,
            expected_contract_sha256=finetune_contract_sha,
            pretrain_dir=pretrain_dir,
            winner_detail=winner_detail,
            run_detail=finetune_detail,
        )

    assert second_seed_context is not None
    if current_selection["decision"].get("status") == "inconclusive_data":
        return _blocked(
            "canonical_finetune",
            "TARGETED_RECORDING_REQUIRED",
            "seed 20260903도 raw val data coverage가 불충분합니다. cross-seed "
            "finalize가 아니라 지정 family/subband 추가 수집이 필요합니다.",
            decision=current_selection["decision"],
            selection={
                "path": str(current_selection["path"]),
                "sha256": current_selection["sha256"],
            },
        )
    primary_selection = second_seed_context["primary_selection"]
    final_selection_path = _cross_seed_final_selection_path(
        repo_root, contract.second_seed.seed_neutral_campaign_sha256
    )
    if not final_selection_path.exists() and not final_selection_path.is_symlink():
        return _ready(
            "cross_seed_finalize",
            "cross_seed_finalize",
            _cross_seed_finalize_command(
                repo_root,
                primary_selection=primary_selection["path"],
                secondary_selection=current_selection["path"],
                final_selection=final_selection_path,
            ),
            winner=winner_detail,
            seed_selections={
                "20260803": {
                    "path": str(primary_selection["path"]),
                    "sha256": primary_selection["sha256"],
                },
                "20260903": {
                    "path": str(current_selection["path"]),
                    "sha256": current_selection["sha256"],
                },
            },
            final_selection=str(final_selection_path),
        )
    try:
        final_content = _regular_bytes(
            final_selection_path, label="cross-seed final selection"
        )
        final_payload = _json_object(
            final_content, label="cross-seed final selection"
        )
        modules["validate_test_open_selection"](final_payload)
        selected_seed = int(final_payload.get("seed"))
        if selected_seed == 20260803:
            terminal_cfg = second_seed_context["primary_finetune_cfg"]
            terminal_dir = second_seed_context["primary_finetune_dir"]
            terminal_state = second_seed_context["primary_state_root"]
            terminal_contract_sha = second_seed_context[
                "primary_finetune_contract_sha"
            ]
            terminal_pretrain = second_seed_context["primary_pretrain_dir"]
        elif selected_seed == 20260903:
            terminal_cfg = finetune_cfg
            terminal_dir = finetune_dir
            terminal_state = state_root
            terminal_contract_sha = finetune_contract_sha
            terminal_pretrain = pretrain_dir
        else:
            raise CampaignError(f"cross-seed final winner seed가 비공식입니다: {selected_seed}")
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        CampaignError,
    ) as exc:
        return _blocked(
            "cross_seed_finalize",
            "CROSS_SEED_FINAL_SELECTION_INVALID",
            str(exc),
            final_selection=str(final_selection_path),
        )
    return _inspect_finetune_terminal_authority(
        modules,
        repo_root=repo_root,
        finetune_cfg=terminal_cfg,
        finetune_dir=terminal_dir,
        state_root=terminal_state,
        expected_contract_sha256=terminal_contract_sha,
        pretrain_dir=terminal_pretrain,
        winner_detail={
            **winner_detail,
            "cross_seed_selected_seed": selected_seed,
            "cross_seed_final_selection_sha256": _sha256_bytes(final_content),
        },
        run_detail={
            "primary": str(second_seed_context["primary_finetune_dir"]),
            "secondary": str(finetune_dir),
        },
        selection_path=final_selection_path,
        completion_report_path=final_selection_path.parent / "completion/completion.json",
        status_path=terminal_state / "status.json",
    )


def inspect_campaign(
    contract: CampaignContract,
    *,
    repo_root: Path = REPO_ROOT,
    resume_path: Path | None = None,
    resume_sha256: str | None = None,
) -> Inspection:
    try:
        local, terminal = _inspect_local_admission(contract, repo_root)
    except CampaignError as exc:
        return _blocked("local_admission", "LOCAL_ADMISSION_FAILED", str(exc))
    if terminal is not None:
        return terminal
    try:
        return _inspect_post_bootstrap(
            contract,
            repo_root,
            local,
            resume_path=resume_path,
            resume_sha256=resume_sha256,
        )
    except CampaignError as exc:
        return _blocked("campaign_inspection", "CAMPAIGN_INSPECTION_FAILED", str(exc), local=local)


def _status_payload(
    contract: CampaignContract,
    inspection: Inspection,
    *,
    execution: dict[str, Any] | None = None,
    execution_seal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "kind": "canonical_elice_campaign_state",
        "authoritative_state_source": "fresh_artifact_validation_not_status_json",
        "contract": {"path": str(contract.path), "sha256": contract.sha256},
        "campaign": {
            "seed": contract.seed,
            "mode": "primary" if contract.second_seed is None else "sealed_second_seed",
            "primary_contract_sha256": (
                contract.second_seed.primary_contract_sha256
                if contract.second_seed is not None
                else None
            ),
            "seed_neutral_campaign_sha256": (
                contract.second_seed.seed_neutral_campaign_sha256
                if contract.second_seed is not None
                else None
            ),
        },
        "expected_commit": contract.expected_commit,
        "canonical_stage_order": list(CANONICAL_STAGE_ORDER),
        "phase": inspection.phase,
        "status": inspection.status,
        "next_action": inspection.next_action,
        "next_command_argv": inspection.command,
        "blockers": inspection.blockers,
        "details": inspection.details,
    }
    if execution is not None:
        payload["execution"] = execution
    if execution_seal is not None:
        payload["execution_seal"] = execution_seal
    return payload


def _validate_state_path(path: Path, repo_root: Path) -> Path:
    candidate = path.absolute()
    if _inside(candidate, repo_root):
        raise CampaignError("state JSON은 clean checkout을 오염시키지 않도록 저장소 밖이어야 합니다")
    parent = candidate.parent
    try:
        info = parent.lstat()
    except OSError as exc:
        raise CampaignError(f"state JSON parent가 없습니다: {parent}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise CampaignError("state JSON parent는 symlink가 아닌 기존 directory여야 합니다")
    if candidate.exists() and candidate.is_symlink():
        raise CampaignError("state JSON target symlink는 허용하지 않습니다")
    return candidate


def atomic_write_state(path: Path, payload: dict[str, Any], *, repo_root: Path = REPO_ROOT) -> None:
    target = _validate_state_path(path, repo_root)
    raw = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument(
        "--state-out",
        type=Path,
        default=None,
        help="저장소 밖 atomic 상태 JSON. --execute-next에서는 필수",
    )
    parser.add_argument(
        "--execute-next",
        action="store_true",
        help="fresh inspection이 만든 exact next command 한 개만 실행",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="partial run을 명시 재개할 exact last.pt",
    )
    parser.add_argument(
        "--expected-resume-checkpoint-sha256",
        default=None,
        help="resume checkpoint의 외부 SHA-256 anchor",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.resume_checkpoint is None) != (
        args.expected_resume_checkpoint_sha256 is None
    ):
        print(
            "[오류] --resume-checkpoint와 --expected-resume-checkpoint-sha256은 함께 필요합니다",
            file=sys.stderr,
        )
        return 2
    if args.execute_next and args.state_out is None:
        print("[오류] --execute-next에는 저장소 밖 --state-out이 필수입니다", file=sys.stderr)
        return 2
    try:
        contract = load_contract(
            args.contract,
            args.expected_contract_sha256,
            repo_root=REPO_ROOT,
        )
        inspection = inspect_campaign(
            contract,
            repo_root=REPO_ROOT,
            resume_path=args.resume_checkpoint,
            resume_sha256=args.expected_resume_checkpoint_sha256,
        )
        execution_seal = None
        if (
            args.execute_next
            and inspection.status == "READY_TO_EXECUTE"
            and inspection.command is not None
        ):
            execution_seal = build_execution_seal(
                inspection.next_action, inspection.command, repo_root=REPO_ROOT
            )
        payload = _status_payload(
            contract, inspection, execution_seal=execution_seal
        )
        if args.state_out is not None:
            atomic_write_state(args.state_out, payload, repo_root=REPO_ROOT)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        if not args.execute_next:
            return 0 if inspection.status in {"READY_TO_EXECUTE", "COMPLETE"} else 1
        if inspection.status != "READY_TO_EXECUTE" or inspection.command is None:
            print(
                f"[중단] next action을 실행할 수 없습니다: {inspection.next_action}",
                file=sys.stderr,
            )
            return 1
        environment = os.environ.copy()
        if inspection.next_action != "bootstrap":
            environment["CUBLAS_WORKSPACE_CONFIG"] = contract.cublas_workspace_config
        assert execution_seal is not None
        try:
            verify_pre_execution_authority(
                contract,
                inspection.next_action,
                inspection.command,
                execution_seal,
                contract_path=args.contract,
                expected_contract_sha256=args.expected_contract_sha256,
                repo_root=REPO_ROOT,
            )
        except (
            CampaignError,
            FileNotFoundError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            blocked = _blocked(
                "pre_execution_authority",
                "PRE_EXECUTION_AUTHORITY_CHANGED",
                f"dry-run 뒤 exact source/contract/script authority가 바뀌었습니다: {exc}",
            )
            blocked_payload = _status_payload(
                contract, blocked, execution_seal=execution_seal
            )
            assert args.state_out is not None
            atomic_write_state(args.state_out, blocked_payload, repo_root=REPO_ROOT)
            print(json.dumps(blocked_payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        completed = subprocess.run(
            inspection.command,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
        )
        execution = {
            "attempted_action": inspection.next_action,
            "argv": inspection.command,
            "returncode": int(completed.returncode),
        }
        # 상태 파일은 authority가 아니다. 실행 뒤 artifact를 fresh 검사해 갱신하되,
        # 검사/상태쓰기 실패가 실제 child exit code를 절대 덮지 않는다. 특히 system
        # Python이 bootstrap을 실행해 .venv를 방금 만든 경우 같은 interpreter의
        # sys.prefix는 바뀌지 않는다. 그 상태에서 project validator를 import하지 않고,
        # 다음 호출을 exact .venv로 요구하는 transition을 명시적으로 기록한다.
        try:
            if (
                inspection.next_action == "bootstrap"
                and int(completed.returncode) == 0
                and Path(sys.prefix).absolute() != (REPO_ROOT / ".venv").absolute()
            ):
                post = _blocked(
                    "bootstrap_transition",
                    "REINVOKE_WITH_EXACT_VENV_REQUIRED",
                    "bootstrap은 성공했습니다. 다음 fresh inspection부터 exact "
                    f"{REPO_ROOT / '.venv/bin/python'}로 이 진입점을 다시 실행하세요.",
                    bootstrap_child_returncode=0,
                )
            else:
                post = inspect_campaign(
                    contract,
                    repo_root=REPO_ROOT,
                    resume_path=None,
                    resume_sha256=None,
                )
            post_payload = _status_payload(contract, post, execution=execution)
            assert args.state_out is not None
            atomic_write_state(args.state_out, post_payload, repo_root=REPO_ROOT)
            print(json.dumps(post_payload, ensure_ascii=False, indent=2, sort_keys=True))
        except BaseException as exc:  # child returncode 보존이 최우선인 terminal 경계
            print(
                "[주의] child 실행 뒤 read-only inspection/state 갱신 실패. "
                f"child returncode={completed.returncode}를 그대로 반환합니다: {exc}",
                file=sys.stderr,
            )
        return int(completed.returncode)
    except (CampaignError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[오류] canonical campaign: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
