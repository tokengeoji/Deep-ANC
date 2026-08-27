"""체크포인트 resume을 같은 실험으로만 제한하는 immutable 계약."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2
_OPERATIONAL_TOP_LEVEL_KEYS = {
    "resume",
    "run_until_step",
    "ckpt_dir",
    "experiment_contract",
    "experiment_contract_sha256",
    "resolved_contract_run_dir",
}

CANONICAL_ROLES = frozenset({"canonical_pretrain", "canonical_finetune"})
"""소스/완료 증명을 생략할 수 없는 공식 학습 역할."""
"""static 실험 의미가 아닌 재개/저장 위치만 contract에서 제외한다.

``run_until_step``은 같은 장기 실험을 smoke/중단/재개하는 operational stop
budget이므로 제외한다. pilot/canonical 구분은 static ``experiment_role``로 한다.

``best_metric`` 수치는 cfg가 아니라 checkpoint 진행 metadata이며 여기에 넣지
않는다. 반면 ``best_metric_key``는 어떤 목적함수로 선택했는지이므로
static 계약에 남는다.
"""


def _normalise(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalise(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("experiment contract에는 NaN/Inf를 기록할 수 없습니다")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(
        f"experiment contract로 직렬화할 수 없는 값입니다: {type(value).__name__}"
    )


def _canonical_config(cfg: dict) -> dict:
    operational = set(_OPERATIONAL_TOP_LEVEL_KEYS)
    # smoke의 label/run root만 두 arm 사이에 달라지는 output metadata다. 이를
    # 모든 role에서 전역 제외하면 canonical/fine-tune cfg에 같은 키를 주입해도
    # resume contract가 감지하지 못한다. 반드시 smoke role에서만 제외한다.
    if str(cfg.get("experiment_role", "")) == "a100_pretrain_smoke":
        operational.update({"a100_smoke_run_label", "resolved_smoke_run_dir"})
    return _normalise(
        {
            key: value
            for key, value in cfg.items()
            if key not in operational
        }
    )


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _artifact_paths(cfg: dict, root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    data = cfg.get("data") or {}
    duct = cfg.get("duct") or {}
    secondary = duct.get("secondary_path") or {}
    digital = duct.get("digital_reference") or {}

    for name, value in (
        ("secondary_path", secondary.get("npz")),
        ("primary_path", digital.get("primary_path_npz")),
        ("rir_bank", data.get("rir_bank")),
        ("recorded_manifest", cfg.get("recorded_manifest")),
        ("bootstrap_receipt", data.get("bootstrap_receipt")),
        ("transfer_manifest", data.get("transfer_manifest")),
        ("campaign_prerequisite", cfg.get("campaign_prerequisite")),
        ("init_checkpoint", cfg.get("init_ckpt")),
    ):
        if value:
            paths[name] = _path(root, value)

    manifest_dir = data.get("noise_manifest_dir")
    mix = data.get("source_mix_ratio") or {}
    if manifest_dir and isinstance(mix, dict):
        directory = _path(root, manifest_dir)
        generation = directory / "manifest_generation.json"
        paths["source_manifest_generation"] = generation
        holdout: Path | None = None
        if generation.is_file():
            try:
                payload = json.loads(generation.read_text(encoding="utf-8"))
                declared = payload.get("holdout") if isinstance(payload, dict) else None
                if declared:
                    holdout = _path(root, declared)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                # 손상된 generation은 그 파일 SHA에 이미 반영되고 readiness에서
                # 구조 오류로 FAIL한다. 여기서 임의 holdout을 추측하지 않는다.
                holdout = None
        if holdout is None:
            explicit = data.get("recorded_holdout")
            holdout = _path(root, explicit) if explicit else directory / "recorded_holdout.json"
        paths["recorded_holdout"] = holdout
        for tag, ratio in sorted(mix.items()):
            if tag != "synthetic" and float(ratio) > 0.0:
                paths[f"source_manifest:{tag}"] = directory / f"{tag}.jsonl"
    return paths


def _artifact_fingerprints(cfg: dict, root: Path) -> dict[str, dict[str, Any]]:
    fingerprints: dict[str, dict[str, Any]] = {}
    for name, path in sorted(_artifact_paths(cfg, root).items()):
        resolved = path.resolve()
        entry: dict[str, Any] = {"path": str(resolved), "exists": resolved.is_file()}
        if resolved.is_file():
            stat = resolved.stat()
            entry.update(size_bytes=int(stat.st_size), sha256=_file_sha256(resolved))
        fingerprints[name] = entry
    return fingerprints


def _portable_source_digest(root: Path, relative_paths: list[str]) -> str:
    """경로+파일 bytes(심볼릭 링크는 링크 문자열)의 portable digest."""

    digest = hashlib.sha256()
    for relative in sorted(set(relative_paths)):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        try:
            if path.is_symlink():
                digest.update(b"SYMLINK\0")
                digest.update(os.readlink(path).encode("utf-8"))
            elif path.is_file():
                digest.update(b"FILE\0")
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
            else:
                digest.update(b"MISSING\0")
        except OSError as exc:
            raise ValueError(f"소스 snapshot을 읽을 수 없습니다: {path}: {exc}") from exc
        digest.update(b"\0")
    return digest.hexdigest()


def _fallback_source_paths(root: Path) -> list[str]:
    paths: list[str] = []
    for directory_name in ("src", "scripts", "configs"):
        directory = root / directory_name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if any(part == "__pycache__" for part in path.parts):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_file() or path.is_symlink():
                paths.append(path.relative_to(root).as_posix())
    return paths


def _git_source_state(root: Path) -> dict[str, Any]:
    """전체 non-ignored worktree와 commit을 결속해 exact-clean 실행만 허용한다."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=20,
        ).stdout.decode("utf-8", errors="strict").strip()
        source_output = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=20,
        ).stdout.decode("utf-8", errors="strict")
        status_output = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=20,
        ).stdout.decode("utf-8", errors="strict")
        replace_output = subprocess.run(
            ["git", "replace", "-l"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=20,
        ).stdout.decode("utf-8", errors="strict")
        flags_output = subprocess.run(
            ["git", "ls-files", "-v"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=20,
        ).stdout.decode("utf-8", errors="strict")
    except (OSError, subprocess.SubprocessError, UnicodeError):
        fallback = _fallback_source_paths(root)
        return {
            "git_commit": None,
            "source_tree_sha256": _portable_source_digest(root, fallback),
            "verifiable": False,
            "clean_exact_commit": False,
            "dirty_paths": [],
            "replace_refs": [],
            "index_flags_clean": False,
        }

    source_paths = [line for line in source_output.splitlines() if line]
    dirty_paths = [line for line in status_output.splitlines() if line]
    replace_refs = [line for line in replace_output.splitlines() if line]
    suspicious_flags = [
        line
        for line in flags_output.splitlines()
        if line and (line[0].islower() or line[0] == "S")
    ]
    return {
        "git_commit": commit,
        "source_tree_sha256": _portable_source_digest(root, source_paths),
        "verifiable": True,
        "clean_exact_commit": not dirty_paths and not replace_refs and not suspicious_flags,
        "dirty_paths": dirty_paths,
        "replace_refs": replace_refs,
        "index_flags_clean": not suspicious_flags,
    }


def build_experiment_contract(
    cfg: dict, *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    """resolved config와 입력 artifact 내용으로 계약 한 벌을 만든다."""

    if repo_root is None:
        from ..config import REPO_ROOT

        root = REPO_ROOT
    else:
        root = Path(repo_root)
    canonical = _canonical_config(cfg)
    source = _git_source_state(Path(root))
    data = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": _json_sha256(canonical),
        "source": source,
        # config_sha만으로 숨기지 않고 transfer generation을 감사 가능한
        # 1급 identity로 노출한다. binder가 stamp 전에 검증·주입한 값이다.
        "input_generation": {
            "bootstrap_receipt_sha256": data.get("bootstrap_receipt_sha256"),
            "transfer_manifest_sha256": data.get("transfer_manifest_sha256"),
            "recorded_transfer_aggregate_sha256": data.get(
                "recorded_transfer_aggregate_sha256"
            ),
        },
        "artifacts": _artifact_fingerprints(cfg, Path(root)),
    }
    payload["sha256"] = _json_sha256(payload)
    return payload


def stamp_experiment_contract(
    cfg: dict, *, repo_root: str | Path | None = None
) -> dict:
    out = copy.deepcopy(cfg)
    contract = build_experiment_contract(out, repo_root=repo_root)
    out["experiment_contract"] = contract
    out["experiment_contract_sha256"] = contract["sha256"]
    return out


def contract_run_directory(
    cfg: dict, *, repo_root: str | Path | None = None
) -> tuple[Path, str]:
    """``runs/<stage>_<contract-sha16>_<seed>`` 유일 실행 디렉터리를 만든다."""

    if repo_root is None:
        from ..config import REPO_ROOT

        root = REPO_ROOT
    else:
        root = Path(repo_root)
    contract = build_experiment_contract(cfg, repo_root=root)
    raw_stage = str(cfg.get("stage") or "train")
    stage = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in raw_stage
    ).strip("-") or "train"
    seed = int(cfg.get("seed", 0))
    name = f"{stage}_{contract['sha256'][:16]}_{seed}"
    return (Path(root) / "runs" / name).resolve(), str(contract["sha256"])


def _verify_embedded(cfg: dict) -> dict:
    contract = cfg.get("experiment_contract")
    digest = cfg.get("experiment_contract_sha256")
    if not isinstance(contract, dict) or not isinstance(digest, str):
        raise ValueError(
            "resume checkpoint에 experiment_contract_sha256가 없습니다 — legacy "
            "artifact는 init_ckpt(weight-only)로만 사용할 수 있습니다"
        )
    if int(contract.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("resume checkpoint experiment contract schema가 다릅니다")
    body = {key: value for key, value in contract.items() if key != "sha256"}
    embedded = str(contract.get("sha256", ""))
    if embedded != _json_sha256(body) or digest != embedded:
        raise ValueError("resume checkpoint experiment contract digest가 손상됐습니다")
    if str(contract.get("config_sha256", "")) != _json_sha256(_canonical_config(cfg)):
        raise ValueError("resume checkpoint cfg가 저장된 experiment contract와 다릅니다")
    return contract


def validate_embedded_experiment_contract(cfg: dict) -> dict:
    """외부 artifact/receipt 검증기가 쓸 embedded 계약 무결성 API."""

    return _verify_embedded(cfg)


def require_exact_source_trust(
    cfg: dict,
    *,
    repo_root: str | Path | None = None,
    roles: frozenset[str] | set[str] | tuple[str, ...] = CANONICAL_ROLES,
) -> dict:
    """지정 역할의 실제 실행은 clean exact source generation만 허용한다."""

    contract = _verify_embedded(cfg)
    role = str(cfg.get("experiment_role", ""))
    if role not in set(roles):
        return contract
    source = contract.get("source")
    if not isinstance(source, dict) or not bool(source.get("verifiable")):
        raise ValueError("canonical 학습은 검증 가능한 git commit이 필요합니다")
    if not bool(source.get("clean_exact_commit")):
        raise ValueError(
            "canonical 학습은 전체 non-ignored worktree가 clean exact commit이어야 합니다: "
            f"dirty={source.get('dirty_paths')}, replace_refs={source.get('replace_refs')}, "
            f"index_flags_clean={source.get('index_flags_clean')}"
        )
    if repo_root is None:
        from ..config import REPO_ROOT

        root = REPO_ROOT
    else:
        root = Path(repo_root)
    current = _git_source_state(Path(root))
    for key in ("git_commit", "source_tree_sha256"):
        if current.get(key) != source.get(key):
            raise ValueError(f"canonical 실행 시점 source {key}가 stamp와 다릅니다")
    if not bool(current.get("clean_exact_commit")):
        raise ValueError("canonical 실행 시점 전체 worktree가 clean exact commit이 아닙니다")
    return contract


def require_canonical_source_trust(
    cfg: dict, *, repo_root: str | Path | None = None
) -> dict:
    """실제 canonical 학습/receipt 발행 시 clean exact commit만 허용한다."""

    return require_exact_source_trust(
        cfg, repo_root=repo_root, roles=CANONICAL_ROLES
    )


def _diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.append(name)
            else:
                differences.extend(_diff_paths(left[key], right[key], name))
        return differences
    if left != right:
        return [prefix or "<root>"]
    return []


def validate_resume_experiment(
    state: dict,
    current_cfg: dict,
    *,
    repo_root: str | Path | None = None,
) -> dict:
    """모델/옵티마이저를 건드리기 전에 resume 계약의 완전 일치를 검사한다."""

    saved_cfg = state.get("cfg")
    if not isinstance(saved_cfg, dict):
        raise ValueError("resume checkpoint에 resolved cfg가 없습니다")
    saved = _verify_embedded(saved_cfg)
    current_stamped = stamp_experiment_contract(current_cfg, repo_root=repo_root)
    current = current_stamped["experiment_contract"]
    if saved["sha256"] != current["sha256"]:
        fields = _diff_paths(
            _canonical_config(saved_cfg), _canonical_config(current_stamped)
        )
        artifact_fields = _diff_paths(
            saved.get("artifacts", {}), current.get("artifacts", {}), "artifacts"
        )
        changed = (fields + artifact_fields)[:12]
        raise ValueError(
            "resume experiment contract 불일치: "
            f"checkpoint={saved['sha256']}, current={current['sha256']}; "
            f"changed={changed}"
        )
    return current_stamped


__all__ = [
    "SCHEMA_VERSION",
    "CANONICAL_ROLES",
    "build_experiment_contract",
    "contract_run_directory",
    "stamp_experiment_contract",
    "validate_embedded_experiment_contract",
    "require_exact_source_trust",
    "require_canonical_source_trust",
    "validate_resume_experiment",
]
