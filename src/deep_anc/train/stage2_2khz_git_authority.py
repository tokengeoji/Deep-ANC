"""Stage-2 production authority의 clean exact Git anchor verifier.

대용량 raw bytes의 자체 PASS JSON은 권한이 아니다. production authority는 사람이
검토해 ``origin/dev``의 exact HEAD에 커밋한 작은 anchor만 인정한다. 검증은 artifact
scan보다 먼저 수행하며 replace/graft, dirty tree, untracked anchor, working-tree 재서명을
모두 거부한다.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any


STAGE2_AUTHORITY_REMOTE_REF = "refs/remotes/origin/dev"


def _git(root: Path, *args: str, no_replace: bool = True) -> bytes:
    environment = dict(os.environ)
    if no_replace:
        environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            env=environment,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Stage-2 Git authority 검증 실패: {' '.join(args)}") from exc


def _snapshot_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Stage-2 authority anchor는 regular file이어야 합니다")
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
        raise ValueError("Stage-2 authority anchor가 snapshot 중 바뀌었습니다")
    content = b"".join(chunks)
    if len(content) != int(after.st_size):
        raise ValueError("Stage-2 authority anchor snapshot 크기가 다릅니다")
    return content


def verify_tracked_head_file(
    repository_root: str | Path,
    relative_path: str,
) -> tuple[bytes, str, str]:
    """origin/dev exact clean HEAD의 tracked regular-file bytes만 반환한다."""

    root = Path(repository_root).resolve(strict=True)
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Stage-2 authority anchor 경로가 repository 상대경로가 아닙니다")

    # 어떠한 raw/manifest/hash scan보다 먼저 checkout authority를 닫는다.
    head = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    remote_head = _git(root, "rev-parse", STAGE2_AUTHORITY_REMOTE_REF).decode("ascii").strip()
    if head != remote_head:
        raise ValueError("Stage-2 authority는 origin/dev exact HEAD에서만 유효합니다")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Stage-2 authority checkout이 clean하지 않습니다")
    if _git(root, "replace", "-l", no_replace=False).strip():
        raise ValueError("Stage-2 authority repository에 replace object가 있습니다")
    git_dir = Path(_git(root, "rev-parse", "--git-dir").decode("utf-8").strip())
    if not git_dir.is_absolute():
        git_dir = root / git_dir
    grafts = git_dir / "info" / "grafts"
    if grafts.exists() and grafts.read_bytes().strip():
        raise ValueError("Stage-2 authority repository에 graft가 있습니다")

    relative_text = relative.as_posix()
    _git(root, "ls-files", "--error-unmatch", "--", relative_text)
    head_bytes = _git(root, "show", f"HEAD:{relative_text}")
    working_bytes = _snapshot_regular(root / relative)
    if working_bytes != head_bytes:
        raise ValueError("Stage-2 authority anchor가 HEAD blob bytes와 다릅니다")
    return working_bytes, hashlib.sha256(working_bytes).hexdigest(), head


def verify_tracked_head_authority(
    repository_root: str | Path,
    relative_path: str,
) -> tuple[dict[str, Any], str, str]:
    """origin/dev exact clean HEAD의 tracked JSON authority만 반환한다."""

    working_bytes, digest, head = verify_tracked_head_file(
        repository_root, relative_path
    )
    try:
        payload = json.loads(working_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Stage-2 authority anchor는 UTF-8 JSON이어야 합니다") from exc
    if not isinstance(payload, dict):
        raise ValueError("Stage-2 authority anchor root는 mapping이어야 합니다")
    return payload, digest, head


def verify_source_commit_ancestor(
    repository_root: str | Path, source_commit_sha: str, *, head: str
) -> None:
    """artifact source commit이 실제 commit object이며 execution HEAD의 ancestor인지 확인한다."""

    if len(source_commit_sha) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit_sha
    ):
        raise ValueError("Stage-2 artifact source commit은 lowercase 40-hex여야 합니다")
    root = Path(repository_root).resolve(strict=True)
    _git(root, "cat-file", "-e", f"{source_commit_sha}^{{commit}}")
    _git(root, "merge-base", "--is-ancestor", source_commit_sha, head)


__all__ = [
    "STAGE2_AUTHORITY_REMOTE_REF",
    "verify_tracked_head_authority",
    "verify_tracked_head_file",
    "verify_source_commit_ancestor",
]
