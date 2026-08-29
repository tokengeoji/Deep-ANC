#!/usr/bin/env python3
"""Elice canonical speech 전체에서 recorded DNS speech 5개를 선택한다.

오디오 장치는 열지 않는다. exact-commit bootstrap receipt와 schema-v4 public
manifest를 먼저 검증한 뒤 strict P(z) coverage를 스캔한다. ``--write``는 receipt,
immutable selection-parent manifest, 10.333초 raw, 15초 composite를 하나의
no-replace directory로 원자 발행한다.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[2]
EARLY_SOURCE_EVIDENCE: dict | None = None
EARLY_QUARANTINE_SUMMARY: dict | None = None
PROTECTED_SOURCE_ROOTS = ("src", "scripts", "configs")
SOURCE_QUARANTINE_SCHEMA = "protected_python_cache_quarantine/v1"
CANONICAL_PYCACHE_PREFIX = "/dev/null/deep-anc-selector"
CANONICAL_GIT_EXECUTABLE = "/usr/bin/git"
FAILED_PUBLICATION_SCHEMA = "dns_selection_failed_publication/v1"


class _OwnedBundlePublication:
    def __init__(
        self,
        *,
        destination: Path,
        directory_fd: int,
        device: int,
        inode: int,
        receipt_sha256: str,
        inventory: list[dict[str, object]],
    ) -> None:
        self.destination = destination
        self.directory_fd = directory_fd
        self.device = device
        self.inode = inode
        self.receipt_sha256 = receipt_sha256
        self.inventory = inventory

    def close(self) -> None:
        if self.directory_fd >= 0:
            os.close(self.directory_fd)
            self.directory_fd = -1


def _canonical_isolated_path() -> tuple[str, ...]:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    stdlib = Path(sysconfig.get_path("stdlib")).resolve(strict=True)
    candidates = [
        REPO_ROOT / "src",
        stdlib.parent / f"python{sys.version_info.major}{sys.version_info.minor}.zip",
        stdlib,
        stdlib / "lib-dynload",
        REPO_ROOT / ".venv/lib" / version / "site-packages",
        Path(sys.base_prefix) / "local/lib" / version / "dist-packages",
        Path(sys.base_prefix) / "lib" / version / "dist-packages",
        Path(sys.base_prefix) / "lib/python3/dist-packages",
    ]
    values: list[str] = []
    for index, candidate in enumerate(candidates):
        if index == 1 or candidate.is_dir():
            value = os.path.abspath(os.fspath(candidate))
            if value not in values:
                values.append(value)
    return tuple(values)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _early_git(root: Path, arguments: list[str]) -> bytes:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        return bytes(
            subprocess.run(
                [
                    CANONICAL_GIT_EXECUTABLE,
                    f"--git-dir={root / '.git'}",
                    f"--work-tree={root}",
                    "-c",
                    f"core.worktree={root}",
                    *arguments,
                ],
                cwd=root,
                env=environment,
                check=True,
                capture_output=True,
                timeout=60,
            ).stdout
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"pre-import exact Git 검증 실패: git {' '.join(arguments)}: {exc}"
        ) from exc


def _parse_early_tree(raw: bytes, *, head: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, path_raw = record.split(b"\t", 1)
            fields = metadata.decode("ascii").split()
            if head:
                mode, kind, object_id = fields
                stage = "0"
            else:
                mode, object_id, stage = fields
                kind = "blob"
            path = path_raw.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("pre-import Git tree inventory 파싱 실패") from exc
        if kind != "blob" or stage != "0" or mode not in {"100644", "100755", "120000"}:
            raise RuntimeError(
                "pre-import Git tree에 지원하지 않는 object/stage가 있습니다: "
                f"{path}, mode={mode}, kind={kind}, stage={stage}"
            )
        rows.append({"path": path, "mode": mode, "object_id": object_id})
    rows.sort(key=lambda item: item["path"])
    if len(rows) != len({item["path"] for item in rows}):
        raise RuntimeError("pre-import Git tree path가 중복됩니다")
    return rows


def _early_blob_object_id(path: Path, *, mode: str, algorithm: str) -> str:
    try:
        info = path.lstat()
        if mode == "120000":
            if not stat.S_ISLNK(info.st_mode):
                raise OSError("expected symlink")
            content = os.fsencode(os.readlink(path))
            size = len(content)
        else:
            if not stat.S_ISREG(info.st_mode) or path.is_symlink():
                raise OSError("expected regular file")
            if bool(info.st_mode & stat.S_IXUSR) != (mode == "100755"):
                raise OSError("executable mode mismatch")
            content = None
            size = int(info.st_size)
    except OSError as exc:
        raise RuntimeError(f"pre-import tracked path 검증 실패: {path}: {exc}") from exc
    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise RuntimeError(f"지원하지 않는 Git object format: {algorithm}") from exc
    digest.update(f"blob {size}\0".encode("ascii"))
    if content is not None:
        digest.update(content)
    else:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _hash_regular_file(path: Path) -> tuple[int, str, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise RuntimeError(f"cache quarantine은 regular file만 허용합니다: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return int(info.st_size), digest.hexdigest(), stat.S_IMODE(info.st_mode)


def _protected_source_scan(
    root: Path, *, tracked: set[str]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cache_directories: list[dict[str, object]] = []
    cache_files: list[dict[str, object]] = []
    unexpected: list[str] = []
    for relative_root in PROTECTED_SOURCE_ROOTS:
        protected = root / relative_root
        if not protected.exists():
            continue
        for directory, names, filenames in os.walk(protected, followlinks=False):
            base = Path(directory)
            retained_names: list[str] = []
            for name in names:
                path = base / name
                relative = path.relative_to(root).as_posix()
                if name.endswith(".egg-info") and path.is_dir() and not path.is_symlink():
                    continue
                if name == "__pycache__":
                    info = path.lstat()
                    if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
                        raise RuntimeError(
                            f"protected __pycache__ symlink/special directory 금지: {relative}"
                        )
                    members: list[dict[str, object]] = []
                    for member in sorted(path.iterdir(), key=lambda item: item.name):
                        member_relative = member.relative_to(root).as_posix()
                        if member.is_dir() or member.is_symlink() or member.suffix != ".pyc":
                            raise RuntimeError(
                                "protected __pycache__에는 regular .pyc만 허용합니다: "
                                f"{member_relative}"
                            )
                        if member_relative in tracked:
                            raise RuntimeError(
                                f"tracked bytecode cache는 자동 격리하지 않습니다: {member_relative}"
                            )
                        size, sha256, mode = _hash_regular_file(member)
                        entry = {
                            "source_path": member_relative,
                            "quarantine_path": f"caches/{member_relative}",
                            "size": size,
                            "sha256": sha256,
                            "mode": mode,
                        }
                        members.append(entry)
                        cache_files.append(entry)
                    cache_directories.append(
                        {
                            "source_path": relative,
                            "quarantine_path": f"caches/{relative}",
                            "file_count": len(members),
                        }
                    )
                    continue
                retained_names.append(name)
                if path.is_symlink() and relative not in tracked:
                    unexpected.append(relative)
            names[:] = retained_names
            for name in filenames:
                path = base / name
                relative = path.relative_to(root).as_posix()
                if relative not in tracked:
                    unexpected.append(relative)
    if unexpected:
        raise RuntimeError(
            "pre-import protected source executable injection: "
            f"{sorted(set(unexpected))[:5]}"
        )
    cache_directories.sort(key=lambda item: str(item["source_path"]))
    cache_files.sort(key=lambda item: str(item["source_path"]))
    return cache_directories, cache_files


def _preimport_exact_source(
    root: Path, *, expected_commit: str
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    root = root.resolve(strict=True)
    git_path = root / ".git"
    git_info = git_path.lstat()
    if not stat.S_ISDIR(git_info.st_mode) or git_path.is_symlink():
        raise RuntimeError("pre-import exact source는 root/.git 실제 directory여야 합니다")
    commit = _early_git(root, ["rev-parse", "--verify", "HEAD^{commit}"]).decode(
        "ascii"
    ).strip().lower()
    if commit != expected_commit.lower():
        raise RuntimeError(
            f"pre-import HEAD가 expected commit과 다릅니다: {commit} != {expected_commit}"
        )
    tree_id = _early_git(root, ["rev-parse", "--verify", "HEAD^{tree}"]).decode(
        "ascii"
    ).strip().lower()
    object_format = _early_git(root, ["rev-parse", "--show-object-format"]).decode(
        "ascii"
    ).strip().lower()
    top = Path(
        _early_git(root, ["rev-parse", "--show-toplevel"]).decode("utf-8").strip()
    ).resolve()
    git_dir = Path(
        _early_git(root, ["rev-parse", "--absolute-git-dir"])
        .decode("utf-8")
        .strip()
    ).resolve()
    if top != root or git_dir != git_path:
        raise RuntimeError("pre-import Git top-level/metadata가 actual root와 다릅니다")
    if _early_git(root, ["replace", "-l"]).strip():
        raise RuntimeError("pre-import exact source에서 git replace ref를 금지합니다")
    grafts = git_dir / "info/grafts"
    if grafts.is_file() and grafts.stat().st_size > 0:
        raise RuntimeError("pre-import exact source에서 git grafts를 금지합니다")
    if _early_git(root, ["status", "--porcelain=v1", "-z", "--untracked-files=all"]):
        raise RuntimeError("pre-import exact clean worktree/index가 아닙니다")
    flags = _early_git(root, ["ls-files", "-v", "-z"])
    if any(
        record and (record[:1].islower() or record[:1] == b"S")
        for record in flags.split(b"\0")
    ):
        raise RuntimeError("pre-import assume-unchanged/skip-worktree flag를 금지합니다")
    head_rows = _parse_early_tree(
        _early_git(root, ["ls-tree", "-r", "-z", "--full-tree", commit]),
        head=True,
    )
    index_rows = _parse_early_tree(
        _early_git(root, ["ls-files", "--stage", "-z"]),
        head=False,
    )
    if head_rows != index_rows:
        raise RuntimeError("pre-import Git HEAD/index tree가 exact 일치하지 않습니다")
    tracked = {row["path"] for row in head_rows}
    cache_directories, cache_files = _protected_source_scan(root, tracked=tracked)
    for row in head_rows:
        actual = _early_blob_object_id(
            root / row["path"], mode=row["mode"], algorithm=object_format
        )
        if actual != row["object_id"]:
            raise RuntimeError(
                "pre-import tracked bytes/mode가 HEAD blob과 다릅니다: "
                f"{row['path']} ({actual} != {row['object_id']})"
            )
    evidence = {
        "schema": "exact_clean_git_source/v1",
        "commit": commit,
        "head_tree_object_id": tree_id,
        "git_object_format": object_format,
        "tracked_file_count": len(head_rows),
        "tracked_inventory_sha256": _canonical_json_sha256(head_rows),
        "policy": {
            "tracked_worktree": "exact_HEAD_blob_and_mode",
            "index": "exact_HEAD_tree_no_hidden_flags",
            "nonignored_untracked": "forbidden",
            "protected_ignored_roots": list(PROTECTED_SOURCE_ROOTS),
            "protected_runtime_bytecode": "forbidden",
            "ignored_artifacts_outside_protected_roots": "allowed",
            "replace_refs_and_grafts": "forbidden",
        },
    }
    return evidence, cache_directories, cache_files


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"no-overwrite quarantine destination exists: {destination}")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("source cache quarantine에는 renameat2가 필요합니다")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"no-overwrite quarantine destination exists: {destination}")
        raise OSError(error, os.strerror(error), str(destination))


def _ensure_real_directory(path: Path, *, mode: int = 0o700) -> None:
    path.mkdir(mode=mode, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
        raise RuntimeError(f"quarantine path는 실제 directory여야 합니다: {path}")


def _write_exclusive_fsync(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _validate_quarantined_files(transaction: Path, plan: dict[str, object]) -> None:
    expected_paths = {
        str(dict(entry)["quarantine_path"]) for entry in plan["files"]
    }
    actual_paths: set[str] = set()
    caches = transaction / "caches"
    if not caches.is_dir() or caches.is_symlink():
        raise RuntimeError("quarantine caches root가 실제 directory가 아닙니다")
    for base, names, filenames in os.walk(caches, followlinks=False):
        root = Path(base)
        for name in names:
            path = root / name
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
                raise RuntimeError(f"quarantine archive directory symlink/special 금지: {path}")
        for name in filenames:
            path = root / name
            relative = path.relative_to(transaction).as_posix()
            _hash_regular_file(path)
            actual_paths.add(relative)
    if actual_paths != expected_paths:
        raise RuntimeError(
            "quarantine archive file inventory가 plan과 exact 일치하지 않습니다"
        )
    for raw_entry in plan["files"]:
        entry = dict(raw_entry)
        path = transaction / str(entry["quarantine_path"])
        size, sha256, mode = _hash_regular_file(path)
        if (
            size != entry["size"]
            or sha256 != entry["sha256"]
            or mode != entry["mode"]
        ):
            raise RuntimeError(f"quarantined cache bytes가 manifest와 다릅니다: {path}")


def _quarantine_relative(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError(f"quarantine {field}가 canonical relative path가 아닙니다")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"quarantine {field} path traversal을 금지합니다")
    return path.as_posix()


def _validate_quarantine_plan(
    plan: object, *, root: Path, expected_commit: str, state: str
) -> dict[str, object]:
    required = {
        "schema",
        "state",
        "source_commit",
        "source_repo_root",
        "protected_roots",
        "source_evidence_sha256",
        "cache_directories",
        "files",
        "inventory_sha256",
        "sequence",
        "transaction_id",
    }
    if not isinstance(plan, dict) or set(plan) != required:
        raise RuntimeError("quarantine plan field set이 exact 계약과 다릅니다")
    directories = plan.get("cache_directories")
    files = plan.get("files")
    if (
        plan.get("schema") != SOURCE_QUARANTINE_SCHEMA
        or plan.get("state") != state
        or plan.get("source_commit") != expected_commit
        or plan.get("source_repo_root") != os.path.abspath(root)
        or plan.get("protected_roots") != list(PROTECTED_SOURCE_ROOTS)
        or not isinstance(plan.get("source_evidence_sha256"), str)
        or len(str(plan["source_evidence_sha256"])) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(plan["source_evidence_sha256"])
        )
        or not isinstance(directories, list)
        or not isinstance(files, list)
        or not isinstance(plan.get("inventory_sha256"), str)
        or len(str(plan.get("inventory_sha256") or "")) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(plan.get("inventory_sha256") or "")
        )
        or plan.get("inventory_sha256") != _canonical_json_sha256(files)
        or isinstance(plan.get("sequence"), bool)
        or not isinstance(plan.get("sequence"), int)
        or int(plan["sequence"]) < 0
    ):
        raise RuntimeError("quarantine plan source/inventory 결속이 다릅니다")
    directory_counts: dict[str, int] = {}
    for raw in directories:
        if not isinstance(raw, dict) or set(raw) != {
            "source_path",
            "quarantine_path",
            "file_count",
        }:
            raise RuntimeError("quarantine cache directory ref가 유효하지 않습니다")
        source = _quarantine_relative(raw["source_path"], field="source_path")
        quarantine = _quarantine_relative(
            raw["quarantine_path"], field="quarantine_path"
        )
        source_path = PurePosixPath(source)
        if (
            len(source_path.parts) < 2
            or source_path.parts[0] not in PROTECTED_SOURCE_ROOTS
            or source_path.name != "__pycache__"
            or quarantine != f"caches/{source}"
            or isinstance(raw["file_count"], bool)
            or not isinstance(raw["file_count"], int)
            or int(raw["file_count"]) < 0
            or source in directory_counts
        ):
            raise RuntimeError("quarantine cache directory path/count가 유효하지 않습니다")
        directory_counts[source] = int(raw["file_count"])
    actual_counts = {source: 0 for source in directory_counts}
    seen_files: set[str] = set()
    for raw in files:
        if not isinstance(raw, dict) or set(raw) != {
            "source_path",
            "quarantine_path",
            "size",
            "sha256",
            "mode",
        }:
            raise RuntimeError("quarantine cache file ref가 유효하지 않습니다")
        source = _quarantine_relative(raw["source_path"], field="file.source_path")
        quarantine = _quarantine_relative(
            raw["quarantine_path"], field="file.quarantine_path"
        )
        source_path = PurePosixPath(source)
        parent = source_path.parent.as_posix()
        if (
            parent not in actual_counts
            or source_path.suffix != ".pyc"
            or quarantine != f"caches/{source}"
            or source in seen_files
            or isinstance(raw["size"], bool)
            or not isinstance(raw["size"], int)
            or int(raw["size"]) < 0
            or not isinstance(raw["sha256"], str)
            or len(str(raw["sha256"])) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(raw["sha256"])
            )
            or isinstance(raw["mode"], bool)
            or not isinstance(raw["mode"], int)
            or not 0 <= int(raw["mode"]) <= 0o777
        ):
            raise RuntimeError("quarantine cache file path/hash/mode가 유효하지 않습니다")
        seen_files.add(source)
        actual_counts[parent] += 1
    if actual_counts != directory_counts:
        raise RuntimeError("quarantine cache directory file_count가 inventory와 다릅니다")
    planned = {**plan, "state": "planned"}
    expected_transaction = _canonical_json_sha256(
        {key: value for key, value in planned.items() if key != "transaction_id"}
    )
    if plan.get("transaction_id") != expected_transaction:
        raise RuntimeError("quarantine transaction self-seal이 다릅니다")
    return plan


def _quarantine_root(root: Path) -> Path:
    identity = hashlib.sha256(os.fsencode(root)).hexdigest()[:16]
    return root.parent / ".deep_anc_source_cache_quarantine" / f"{root.name}-{identity}"


def _complete_quarantine_staging(
    *, root: Path, staging: Path, final: Path, plan: dict[str, object]
) -> dict[str, object]:
    _validate_quarantine_plan(
        plan,
        root=root,
        expected_commit=str(plan.get("source_commit") or ""),
        state="planned",
    )
    for raw_directory in plan["cache_directories"]:
        directory = dict(raw_directory)
        source = root / str(directory["source_path"])
        destination = staging / str(directory["quarantine_path"])
        _ensure_real_directory(destination.parent)
        if source.exists() or source.is_symlink():
            if destination.exists() or destination.is_symlink():
                raise RuntimeError(
                    f"quarantine source/destination이 동시에 존재합니다: {source}"
                )
            info = source.lstat()
            if not stat.S_ISDIR(info.st_mode) or source.is_symlink():
                raise RuntimeError(f"quarantine source cache가 실제 directory가 아닙니다: {source}")
            _rename_noreplace(source, destination)
            _fsync_directory(source.parent)
            _fsync_directory(destination.parent)
        elif not destination.is_dir() or destination.is_symlink():
            raise RuntimeError(
                f"중단 quarantine을 복구할 source/destination이 없습니다: {source}"
            )
    _validate_quarantined_files(staging, plan)
    completed = {**plan, "state": "complete"}
    manifest_raw = _canonical_json_bytes(completed) + b"\n"
    manifest_path = staging / "quarantine_manifest.json"
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_raw:
            raise RuntimeError("기존 quarantine completion manifest가 transaction과 다릅니다")
    else:
        _write_exclusive_fsync(manifest_path, manifest_raw)
    for directory in sorted(
        (path for path in staging.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        _fsync_directory(directory)
    _fsync_directory(staging)
    _rename_noreplace(staging, final)
    _fsync_directory(final.parent)
    return {
        "status": "quarantined",
        "transaction_id": plan["transaction_id"],
        "path": os.path.abspath(final),
        "manifest_path": os.path.abspath(final / "quarantine_manifest.json"),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "file_count": len(plan["files"]),
    }


def _resume_quarantine_transactions(
    root: Path, *, commit: str, evidence: dict[str, object]
) -> list[dict[str, object]]:
    base = _quarantine_root(root) / commit
    if not base.exists():
        return []
    if not base.is_dir() or base.is_symlink():
        raise RuntimeError(f"quarantine commit root가 실제 directory가 아닙니다: {base}")
    summaries: list[dict[str, object]] = []
    for staging in sorted(base.glob(".building-*")):
        if not staging.is_dir() or staging.is_symlink():
            raise RuntimeError(f"quarantine staging symlink/special 금지: {staging}")
        plan_path = staging / "quarantine_plan.json"
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"중단 quarantine plan을 검증할 수 없습니다: {staging}") from exc
        transaction_id = str(plan.get("transaction_id") or "")
        validated = _validate_quarantine_plan(
            plan,
            root=root,
            expected_commit=commit,
            state="planned",
        )
        if validated["source_evidence_sha256"] != _canonical_json_sha256(evidence):
            raise RuntimeError(
                "중단 quarantine plan이 현재 exact source evidence와 다릅니다"
            )
        if (
            staging.name != f".building-{transaction_id}"
        ):
            raise RuntimeError(f"중단 quarantine plan self-seal 불일치: {staging}")
        summaries.append(
            _complete_quarantine_staging(
                root=root,
                staging=staging,
                final=base / transaction_id,
                plan=plan,
            )
        )
    return summaries


def _quarantine_cache_directories(
    root: Path,
    *,
    evidence: dict[str, object],
    cache_directories: list[dict[str, object]],
    cache_files: list[dict[str, object]],
) -> dict[str, object]:
    commit = str(evidence["commit"])
    resumed = _resume_quarantine_transactions(root, commit=commit, evidence=evidence)
    if resumed:
        # resume가 source cache 일부를 이동했으므로 caller가 exact inventory를 다시
        # 계산하게 한다.
        return {"status": "resumed", "transactions": resumed}
    if not cache_directories:
        return {"status": "no_cache", "transactions": []}
    base = _quarantine_root(root) / commit
    _ensure_real_directory(base)
    sequence = 0
    while sequence < 10_000:
        plan_core: dict[str, object] = {
            "schema": SOURCE_QUARANTINE_SCHEMA,
            "state": "planned",
            "source_commit": commit,
            "source_repo_root": os.path.abspath(root),
            "protected_roots": list(PROTECTED_SOURCE_ROOTS),
            "source_evidence_sha256": _canonical_json_sha256(evidence),
            "cache_directories": cache_directories,
            "files": cache_files,
            "inventory_sha256": _canonical_json_sha256(cache_files),
            "sequence": sequence,
        }
        transaction_id = _canonical_json_sha256(plan_core)
        final = base / transaction_id
        staging = base / f".building-{transaction_id}"
        if (
            not final.exists()
            and not final.is_symlink()
            and not staging.exists()
            and not staging.is_symlink()
        ):
            break
        sequence += 1
    else:
        raise RuntimeError("cache quarantine transaction namespace가 소진됐습니다")
    plan = {**plan_core, "transaction_id": transaction_id}
    staging = base / f".building-{transaction_id}"
    final = base / transaction_id
    staging.mkdir(mode=0o700, exist_ok=False)
    _fsync_directory(base)
    plan_raw = _canonical_json_bytes(plan) + b"\n"
    _write_exclusive_fsync(staging / "quarantine_plan.json", plan_raw)
    return _complete_quarantine_staging(
        root=root,
        staging=staging,
        final=final,
        plan=plan,
    )


def _load_completed_quarantine(
    root: Path, *, expected_commit: str, transaction: Path
) -> tuple[dict[str, object], bytes]:
    transaction = transaction.expanduser().resolve(strict=True)
    expected_parent = (_quarantine_root(root) / expected_commit).resolve(strict=True)
    if transaction.parent != expected_parent or transaction.is_symlink():
        raise RuntimeError(
            "restore transaction은 현 repository/commit quarantine root 직하여야 합니다"
        )
    manifest_path = transaction / "quarantine_manifest.json"
    try:
        raw = manifest_path.read_bytes()
        plan = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("restore quarantine manifest를 읽을 수 없습니다") from exc
    plan = _validate_quarantine_plan(
        plan,
        root=root,
        expected_commit=expected_commit,
        state="complete",
    )
    transaction_id = str(plan["transaction_id"])
    if transaction.name != transaction_id or raw != _canonical_json_bytes(plan) + b"\n":
        raise RuntimeError("restore quarantine manifest self-seal/source 결속 불일치")
    _validate_quarantined_files(transaction, plan)
    return plan, raw


def _restore_cache_quarantine(
    root: Path, *, expected_commit: str, transaction: str | Path
) -> dict[str, object]:
    transaction_path = Path(transaction)
    if not transaction_path.is_absolute():
        raise RuntimeError("restore quarantine transaction은 absolute path여야 합니다")
    plan, manifest_raw = _load_completed_quarantine(
        root,
        expected_commit=expected_commit,
        transaction=transaction_path,
    )
    transaction_path = transaction_path.resolve(strict=True)
    for raw_directory in plan["cache_directories"]:
        directory = dict(raw_directory)
        destination = root / str(directory["source_path"])
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"restore는 기존 source cache를 overwrite하지 않습니다: {destination}"
            )
        _ensure_real_directory(destination.parent)
        staging = destination.parent / (
            f".{destination.name}.restore-{str(plan['transaction_id'])[:16]}"
        )
        staging.mkdir(mode=0o700, exist_ok=False)
        prefix = f"{directory['source_path']}/"
        members = [
            dict(entry)
            for entry in plan["files"]
            if str(entry["source_path"]).startswith(prefix)
        ]
        if len(members) != int(directory["file_count"]):
            raise RuntimeError("restore cache directory file_count가 manifest와 다릅니다")
        for entry in members:
            relative_member = str(entry["source_path"])[len(prefix) :]
            if not relative_member or "/" in relative_member or "\\" in relative_member:
                raise RuntimeError("restore cache member path가 direct child가 아닙니다")
            source = transaction_path / str(entry["quarantine_path"])
            target = staging / relative_member
            with source.open("rb") as input_handle, target.open("xb") as output_handle:
                for block in iter(lambda: input_handle.read(1024 * 1024), b""):
                    output_handle.write(block)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            os.chmod(target, int(entry["mode"]))
            sync_descriptor = os.open(
                target,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(sync_descriptor)
            finally:
                os.close(sync_descriptor)
            size, sha256, mode = _hash_regular_file(target)
            if (
                size != entry["size"]
                or sha256 != entry["sha256"]
                or mode != entry["mode"]
            ):
                raise RuntimeError(f"restore cache copy 검증 실패: {target}")
        _fsync_directory(staging)
        _rename_noreplace(staging, destination)
        _fsync_directory(destination.parent)
    receipt = {
        "schema": "protected_python_cache_restore/v1",
        "source_commit": expected_commit,
        "source_repo_root": os.path.abspath(root),
        "transaction_id": plan["transaction_id"],
        "quarantine_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "restored_file_count": len(plan["files"]),
    }
    receipt_raw = _canonical_json_bytes(receipt) + b"\n"
    receipt_path = transaction_path / "restore_receipt.json"
    _write_exclusive_fsync(receipt_path, receipt_raw)
    return {
        "status": "restored",
        "transaction_id": plan["transaction_id"],
        "restore_receipt": os.path.abspath(receipt_path),
        "restore_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "file_count": len(plan["files"]),
    }


def _extract_early_argument(name: str, *, required: bool) -> str | None:
    values: list[str] = []
    for index, argument in enumerate(sys.argv[1:]):
        if argument == name:
            actual = index + 2
            if actual >= len(sys.argv):
                raise SystemExit(f"[차단] {name} 뒤에 값이 필요합니다")
            values.append(sys.argv[actual])
        elif argument.startswith(f"{name}="):
            values.append(argument.split("=", 1)[1])
    if len(values) > 1 or (required and len(values) != 1):
        raise SystemExit(f"[차단] {name}은 정확히 한 번 지정해야 합니다")
    return values[0] if values else None


def _early_isolated_preflight() -> None:
    global EARLY_QUARANTINE_SUMMARY, EARLY_SOURCE_EVIDENCE
    expected_flags = (1, 1, 1, 1, 1)
    actual_flags = (
        int(sys.flags.isolated),
        int(sys.flags.ignore_environment),
        int(sys.flags.no_user_site),
        int(sys.flags.no_site),
        int(sys.flags.dont_write_bytecode),
    )
    if actual_flags != expected_flags:
        raise SystemExit(
            "[차단] DNS selector는 `.venv/bin/python -I -S -B "
            "-X pycache_prefix=/dev/null/deep-anc-selector "
            "scripts/data/select_recorded_dns_speech.py ...`로 실행해야 합니다."
        )
    if sys.pycache_prefix != CANONICAL_PYCACHE_PREFIX:
        raise SystemExit(
            "[차단] DNS selector는 canonical -X pycache_prefix=/dev/null/"
            "deep-anc-selector가 필요합니다."
        )
    expected_executable = os.path.abspath(REPO_ROOT / ".venv/bin/python")
    if os.path.abspath(sys.executable) != expected_executable:
        raise SystemExit("[차단] repository canonical .venv interpreter가 아닙니다.")
    expected_commit = _extract_early_argument("--expected-commit", required=True)
    assert expected_commit is not None
    if len(expected_commit) != 40 or any(
        character not in "0123456789abcdefABCDEF" for character in expected_commit
    ):
        raise SystemExit("[차단] --expected-commit은 전체 40자리 SHA여야 합니다")
    restore_transaction = _extract_early_argument(
        "--restore-source-cache-quarantine", required=False
    )
    if restore_transaction is not None:
        try:
            _evidence, cache_directories, cache_files = _preimport_exact_source(
                REPO_ROOT, expected_commit=expected_commit.lower()
            )
            if cache_directories or cache_files:
                raise RuntimeError(
                    "restore 전에 source protected root에 cache가 이미 존재합니다"
                )
            summary = _restore_cache_quarantine(
                REPO_ROOT,
                expected_commit=expected_commit.lower(),
                transaction=restore_transaction,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"[차단] source cache quarantine restore 실패: {exc}") from exc
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        raise SystemExit(0)
    try:
        before, directories, files = _preimport_exact_source(
            REPO_ROOT, expected_commit=expected_commit.lower()
        )
        first = _quarantine_cache_directories(
            REPO_ROOT,
            evidence=before,
            cache_directories=directories,
            cache_files=files,
        )
        # interrupted transaction resume 또는 retained cache 이동 뒤 실제 root를
        # 다시 전수 검증한다. cache가 남으면 한 번 더 새 transaction으로 이동한다.
        middle, directories, files = _preimport_exact_source(
            REPO_ROOT, expected_commit=expected_commit.lower()
        )
        second = _quarantine_cache_directories(
            REPO_ROOT,
            evidence=middle,
            cache_directories=directories,
            cache_files=files,
        )
        after, remaining_directories, remaining_files = _preimport_exact_source(
            REPO_ROOT, expected_commit=expected_commit.lower()
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"[차단] DNS selector pre-import source trust 실패: {exc}") from exc
    if remaining_directories or remaining_files or before != middle or middle != after:
        raise SystemExit("[차단] cache quarantine 전후 exact source evidence가 달라졌습니다")
    EARLY_SOURCE_EVIDENCE = after
    EARLY_QUARANTINE_SUMMARY = {"first": first, "second": second}
    sys.path[:] = list(_canonical_isolated_path())


if __name__ == "__main__":
    _early_isolated_preflight()
else:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.atomic_publish import publish_directory_noreplace  # noqa: E402
from deep_anc.data.holdout_contract import reject_symlink_components  # noqa: E402
from deep_anc.data.manifest_contract import validate_manifest_generation  # noqa: E402
from deep_anc.data.recorded_dns_selection import (  # noqa: E402
    DNS_SELECTION_BUNDLE_ROOT,
    DNS_SELECTION_RECEIPT,
    DNSSelectionError,
    build_dns_selection_payload,
    validate_dns_selection_receipt,
)
from deep_anc.data.transfer_contract import (  # noqa: E402
    TransferContractError,
    bind_recorded_transfer_config,
)
from deep_anc.data.source_trust import (  # noqa: E402
    SourceTrustError,
    canonical_selector_sys_path,
    exact_clean_source_evidence,
    exact_selector_runtime_evidence,
)


DEFAULT_MANIFEST_DIR = "data/manifests/canonical_v4"
DEFAULT_STRICT_PRIMARY = "assets/measured/primary_path_il_strict_5dc06fdd.npz"
DEFAULT_STRICT_PRIMARY_SHA256 = (
    "23fa43f1ec46d5bca6bdad53938b81bb2d2c85afc4eee35e83c555b6c4f0c598"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--bootstrap-receipt-sha256", required=True)
    parser.add_argument(
        "--receipt-sha256",
        help="--verify-existing에서 필수인 외부 selection receipt SHA256 anchor",
    )
    parser.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--strict-primary", default=DEFAULT_STRICT_PRIMARY)
    parser.add_argument(
        "--strict-primary-sha256", default=DEFAULT_STRICT_PRIMARY_SHA256
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    return parser


def _bundle_inventory(directory: Path) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for base, names, filenames in os.walk(directory, followlinks=False):
        root = Path(base)
        for name in names:
            path = root / name
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
                raise DNSSelectionError(f"selection bundle directory symlink/special 금지: {path}")
        for name in filenames:
            path = root / name
            size, sha256, mode = _hash_regular_file(path)
            inventory.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "size": size,
                    "sha256": sha256,
                    "mode": mode,
                }
            )
    inventory.sort(key=lambda item: str(item["path"]))
    return inventory


def _write_bundle_no_replace(
    payload: dict, files: dict[str, bytes]
) -> _OwnedBundlePublication:
    destination = REPO_ROOT / DNS_SELECTION_BUNDLE_ROOT
    reject_symlink_components(destination.parent, root=REPO_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".dns-selection-", dir=destination.parent)
    )
    owned_fd = -1
    try:
        for relative, content in sorted(files.items()):
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        receipt_relative = Path(DNS_SELECTION_RECEIPT).relative_to(
            DNS_SELECTION_BUNDLE_ROOT
        )
        receipt = staging / receipt_relative
        receipt.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        with receipt.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # renameat2 전에 모든 새 directory entry를 영속화한다. 최종 helper가
        # RENAME_NOREPLACE와 destination parent fsync를 담당한다.
        directories = sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        directories.append(staging)
        for directory in directories:
            descriptor = os.open(
                directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        inventory = _bundle_inventory(staging)
        owned_fd = os.open(
            staging,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        owned_info = os.fstat(owned_fd)
        publish_directory_noreplace(staging, destination)
        return _OwnedBundlePublication(
            destination=destination,
            directory_fd=owned_fd,
            device=int(owned_info.st_dev),
            inode=int(owned_info.st_ino),
            receipt_sha256=hashlib.sha256(encoded).hexdigest(),
            inventory=inventory,
        )
    except BaseException:
        if owned_fd >= 0:
            os.close(owned_fd)
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _write_failure_receipt_at(
    publication: _OwnedBundlePublication,
    *,
    payload: dict,
    stage: str,
    failure: BaseException,
    quarantine: Path,
) -> tuple[Path, str]:
    core = {
        "schema": FAILED_PUBLICATION_SCHEMA,
        "kind": "dns_selection_post_publish_failure",
        "failure_stage": stage,
        "failure_type": type(failure).__name__,
        "failure_message": str(failure)[:2000],
        "canonical_bundle_path": os.path.abspath(publication.destination),
        "quarantine_bundle_path": os.path.abspath(quarantine),
        "source_commit": str(payload.get("source_commit") or ""),
        "selection_evidence_sha256": str(payload.get("evidence_sha256") or ""),
        "expected_receipt_sha256": publication.receipt_sha256,
        "owned_directory": {
            "device": publication.device,
            "inode": publication.inode,
        },
        "bundle_inventory_sha256": _canonical_json_sha256(publication.inventory),
    }
    receipt = {**core, "evidence_sha256": _canonical_json_sha256(core)}
    raw = _canonical_json_bytes(receipt) + b"\n"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(
        "publish_failure_receipt.json",
        flags,
        0o600,
        dir_fd=publication.directory_fd,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("failure receipt short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(publication.directory_fd)
    return quarantine / "publish_failure_receipt.json", hashlib.sha256(raw).hexdigest()


def _quarantine_failed_publication(
    publication: _OwnedBundlePublication,
    *,
    payload: dict,
    stage: str,
    failure: BaseException,
) -> dict[str, object]:
    if publication.directory_fd < 0:
        raise DNSSelectionError("closed publication handle은 quarantine할 수 없습니다")
    handle_info = os.fstat(publication.directory_fd)
    try:
        path_info = publication.destination.lstat()
    except OSError as exc:
        raise DNSSelectionError("published bundle canonical path가 사라졌습니다") from exc
    identity = (int(handle_info.st_dev), int(handle_info.st_ino))
    if (
        identity != (publication.device, publication.inode)
        or identity != (int(path_info.st_dev), int(path_info.st_ino))
        or not stat.S_ISDIR(path_info.st_mode)
        or publication.destination.is_symlink()
    ):
        raise DNSSelectionError(
            "published bundle identity가 writer-owned inode와 다르므로 quarantine을 거부합니다"
        )
    if _bundle_inventory(publication.destination) != publication.inventory:
        raise DNSSelectionError(
            "published bundle bytes가 writer-owned inventory와 다르므로 quarantine을 거부합니다"
        )

    failure_root = publication.destination.parent / ".publish-failures"
    _ensure_real_directory(failure_root)
    _fsync_directory(failure_root.parent)
    token = hashlib.sha256(os.urandom(32)).hexdigest()[:24]
    quarantine = failure_root / (
        f"{publication.receipt_sha256[:16]}-{token}"
    )
    _rename_noreplace(publication.destination, quarantine)
    _fsync_directory(publication.destination.parent)
    _fsync_directory(failure_root)
    receipt_path, receipt_sha = _write_failure_receipt_at(
        publication,
        payload=payload,
        stage=stage,
        failure=failure,
        quarantine=quarantine,
    )
    return {
        "status": "quarantined_post_publish_failure",
        "path": os.path.abspath(quarantine),
        "failure_receipt_path": os.path.abspath(receipt_path),
        "failure_receipt_sha256": receipt_sha,
        "expected_selection_receipt_sha256": publication.receipt_sha256,
    }


def _publish_verified_bundle(payload: dict, files: dict[str, bytes]) -> dict:
    """pre/post trust와 receipt 검증을 하나의 owned publish transaction으로 묶는다."""

    _assert_publish_trust(payload)
    publication = _write_bundle_no_replace(payload, files)
    stage = "post_publish_trust"
    try:
        _assert_publish_trust(payload)
        stage = "post_publish_receipt_validation"
        return validate_dns_selection_receipt(
            repo_root=REPO_ROOT,
            expected_receipt_sha256=publication.receipt_sha256,
            require_source_files=True,
        )
    except BaseException as exc:
        try:
            quarantine = _quarantine_failed_publication(
                publication,
                payload=payload,
                stage=stage,
                failure=exc,
            )
        except BaseException as quarantine_exc:
            raise DNSSelectionError(
                "selection post-publish 검증 실패 후 owned bundle quarantine도 "
                f"실패했습니다: validation={exc}; quarantine={quarantine_exc}"
            ) from exc
        raise DNSSelectionError(
            "selection post-publish 검증 실패; writer-owned bundle을 "
            f"격리했습니다: {quarantine}"
        ) from exc
    finally:
        publication.close()


def _assert_publish_trust(payload: dict) -> None:
    """scan receipt와 현재 source/runtime가 publish 경계에서도 동일한지 확인."""

    try:
        source = exact_clean_source_evidence(
            REPO_ROOT,
            expected_commit=str(payload["source_commit"]),
            reject_runtime_bytecode=True,
        )
        freeze = payload["environment_freeze_origin"]
        runtime = exact_selector_runtime_evidence(
            REPO_ROOT,
            freeze_receipt=str(freeze["path"]),
            expected_freeze_sha256=str(freeze["sha256"]),
        )
    except (KeyError, TypeError, SourceTrustError) as exc:
        raise DNSSelectionError(f"selection publish trust 재검증 실패: {exc}") from exc
    if source != payload.get("clean_source"):
        raise DNSSelectionError("selection scan 이후 publish 직전 source가 변경됐습니다")
    if runtime != payload.get("selector_runtime"):
        raise DNSSelectionError("selection scan 이후 publish runtime이 변경됐습니다")


def main(argv: list[str] | None = None) -> int:
    if __name__ == "__main__":
        if EARLY_SOURCE_EVIDENCE is None or EARLY_QUARANTINE_SUMMARY is None:
            raise DNSSelectionError("selector pre-import source trust evidence가 없습니다")
        try:
            imported_source = exact_clean_source_evidence(
                REPO_ROOT,
                expected_commit=str(EARLY_SOURCE_EVIDENCE["commit"]),
                reject_runtime_bytecode=True,
            )
        except SourceTrustError as exc:
            raise DNSSelectionError(
                f"project import 직후 exact source 재검증 실패: {exc}"
            ) from exc
        if imported_source != EARLY_SOURCE_EVIDENCE:
            raise DNSSelectionError(
                "pre-import stdlib source evidence와 project validator가 다릅니다"
            )
    args = _parser().parse_args(argv)
    try:
        if args.verify_existing:
            if args.receipt_sha256 is None:
                raise DNSSelectionError(
                    "--verify-existing은 --receipt-sha256 외부 anchor가 필수입니다"
                )
        elif args.receipt_sha256 is not None:
            raise DNSSelectionError(
                "--receipt-sha256는 --verify-existing에서만 사용합니다"
            )
        if __name__ == "__main__" and tuple(sys.path) != canonical_selector_sys_path(
            REPO_ROOT
        ):
            raise DNSSelectionError("selector canonical isolated sys.path가 변경됐습니다")
        if __name__ == "__main__":
            print(
                "[source-cache-quarantine] "
                + json.dumps(
                    EARLY_QUARANTINE_SUMMARY,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        if args.verify_existing:
            summary = validate_dns_selection_receipt(
                repo_root=REPO_ROOT,
                expected_receipt_sha256=args.receipt_sha256,
                require_source_files=True,
            )
            if summary.get("source_commit") != args.expected_commit.lower():
                raise DNSSelectionError(
                    "existing DNS receipt source_commit이 --expected-commit과 다릅니다"
                )
            if (
                summary.get("bootstrap_receipt_sha256")
                != args.bootstrap_receipt_sha256.lower()
            ):
                raise DNSSelectionError(
                    "existing DNS receipt bootstrap SHA가 CLI 외부 anchor와 다릅니다"
                )
            if summary.get("receipt_sha256") != args.receipt_sha256.lower():
                raise DNSSelectionError(
                    "existing DNS receipt SHA가 CLI 외부 anchor와 다릅니다"
                )
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
            return 0

        manifest_dir = Path(args.manifest_dir)
        if not manifest_dir.is_absolute():
            manifest_dir = REPO_ROOT / manifest_dir
        generation = validate_manifest_generation(
            manifest_dir, required_tags=("speech",), repo_root=REPO_ROOT
        )
        speech = manifest_dir / "speech.jsonl"
        validated = generation.get("_validated_manifest_bytes", {}).get("speech")
        if not isinstance(validated, bytes):
            raise DNSSelectionError(
                "schema-v4 validator의 speech snapshot이 없습니다"
            )
        validated_manifest_sha = hashlib.sha256(validated).hexdigest()

        # selector가 단순 receipt JSON만 보지 않고 bootstrap의 transfer/environment/
        # exact HEAD 체인까지 기존 공식 경계로 검증한다.
        bootstrap_cfg = {
            "bootstrap_receipt": "data/manifests/elice_bootstrap_receipt.json",
            "bootstrap_receipt_sha256": args.bootstrap_receipt_sha256.lower(),
        }
        bind_recorded_transfer_config(bootstrap_cfg, repo_root=REPO_ROOT)

        strict = Path(args.strict_primary)
        if not strict.is_absolute():
            strict = REPO_ROOT / strict
        strict_sha = hashlib.sha256(strict.read_bytes()).hexdigest()
        if strict_sha != args.strict_primary_sha256.lower():
            raise DNSSelectionError(
                "strict primary SHA가 외부 anchor와 다릅니다: "
                f"expected={args.strict_primary_sha256.lower()}, actual={strict_sha}"
            )
        payload, files = build_dns_selection_payload(
            repo_root=REPO_ROOT,
            public_manifest=speech.relative_to(REPO_ROOT).as_posix(),
            bootstrap_receipt="data/manifests/elice_bootstrap_receipt.json",
            bootstrap_receipt_sha256=args.bootstrap_receipt_sha256,
            strict_primary=strict.relative_to(REPO_ROOT).as_posix(),
            expected_commit=args.expected_commit,
            expected_public_manifest_sha256=validated_manifest_sha,
        )
        if args.write:
            summary = _publish_verified_bundle(payload, files)
        else:
            summary = {
                "source_commit": payload["source_commit"],
                "manifest_sha256": payload["public_manifest"]["sha256"],
                "selected_group_ids": sorted(
                    str(item["public_group_id"]) for item in payload["selected"]
                ),
                "evidence_sha256": payload["evidence_sha256"],
            }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (
        DNSSelectionError,
        TransferContractError,
        OSError,
        ValueError,
    ) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
