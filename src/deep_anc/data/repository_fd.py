"""Repository-relative file access with held dirfd/inode guards.

The live acoustic admission path cannot use ``Path.exists()`` followed by a
second pathname open: a renamed parent or symlink splice between those calls
would make the validation describe different bytes.  This module walks every
parent with ``openat(O_DIRECTORY|O_NOFOLLOW)``, keeps those descriptors open,
and can keep a regular-file descriptor pinned across a legacy semantic
validator.  ``RepositoryFileGuard.verify()`` then rereads the same descriptor
and checks that the lexical name and every held parent still name the same
inodes.

This module never imports an audio backend and never opens an audio device.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
from typing import Any


EXTERNAL_POST_RECEIPT_SUFFIX = ".post_receipt.json"
V5_EXTERNAL_POST_RECEIPT_SUFFIX = EXTERNAL_POST_RECEIPT_SUFFIX


def external_post_receipt_relative_path(raw_relative_path: str) -> str:
    """세대와 무관한 canonical sibling receipt 이름을 반환한다."""

    raw = canonical_relative_path(raw_relative_path, label="live raw path")
    path = PurePosixPath(raw)
    return path.with_name(path.name + EXTERNAL_POST_RECEIPT_SUFFIX).as_posix()


def external_post_receipt_relative_path_v5(raw_relative_path: str) -> str:
    """기존 v5 public alias; 결과 bytes/path를 그대로 유지한다."""

    return external_post_receipt_relative_path(raw_relative_path)


def repository_execution_identity(
    repository_root_path: str | os.PathLike[str], script_relative_path: str
) -> dict[str, Any]:
    """Bind one executable script to a clean exact git checkout."""

    root = repository_root(repository_root_path)
    relative = canonical_relative_path(script_relative_path, label="script path")

    def git(*arguments: str, allow_empty: bool = False) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        if not value and not allow_empty:
            raise RuntimeError(f"git {' '.join(arguments)} 결과가 비었습니다")
        return value

    commit = git("rev-parse", "HEAD")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("repository HEAD가 exact 40-char lowercase commit이 아닙니다")
    if git("status", "--porcelain", "--untracked-files=normal", allow_empty=True):
        raise RuntimeError("live audio는 dirty repository checkout에서 금지됩니다")
    flags = git("ls-files", "-v")
    for line in flags.splitlines():
        if len(line) < 3 or line[1] != " " or not line[0].isalpha():
            raise RuntimeError("git ls-files -v 출력 형식이 비정상입니다")
        marker = line[0]
        if marker.islower() or marker == "S":
            raise RuntimeError(
                "repository tracked file의 assume-unchanged/skip-worktree "
                "index flag를 허용하지 않습니다"
            )
    if git("replace", "-l", allow_empty=True):
        raise RuntimeError("git replacement object가 있는 checkout은 허용하지 않습니다")
    graft_path = Path(git("rev-parse", "--git-path", "info/grafts"))
    if not graft_path.is_absolute():
        graft_path = root / graft_path
    if graft_path.is_file() and graft_path.read_bytes().strip():
        raise RuntimeError("git graft가 있는 checkout은 허용하지 않습니다")
    branch = git("branch", "--show-current", allow_empty=True) or "DETACHED"
    with RepositoryFileGuard(root, relative, label="execution script") as guard:
        snapshot = guard.snapshot()
        committed = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        if committed != guard.bytes:
            raise RuntimeError("execution script bytes가 HEAD blob과 exact 일치하지 않습니다")
        guard.verify()
    return {
        "repository_commit": commit,
        "repository_branch": branch,
        "repository_dirty": False,
        "script_path": relative,
        "script_file_sha256": snapshot["sha256"],
    }


def canonical_relative_path(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError(f"{label}는 canonical repository 상대경로여야 합니다")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{label}는 canonical repository 상대경로여야 합니다")
    return value


def repository_root(value: str | os.PathLike[str]) -> Path:
    root = Path(os.path.abspath(os.fspath(value)))
    try:
        status = os.lstat(root)
    except OSError as error:
        raise ValueError(f"repository root를 읽을 수 없습니다: {root}") from error
    if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode):
        raise ValueError("repository root는 symlink가 아닌 directory여야 합니다")
    return root


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def open_parent_chain(
    repository_root_path: Path,
    relative_path: str,
    *,
    create: bool,
) -> tuple[str, list[tuple[Path, int, int, int]]]:
    """Open and hold root through the target parent without following links."""

    relative = canonical_relative_path(relative_path, label="repository file path")
    parts = PurePosixPath(relative).parts
    root_fd = os.open(repository_root_path, _directory_flags())
    opened: list[tuple[Path, int, int, int]] = []
    try:
        root_status = os.fstat(root_fd)
        opened.append(
            (
                repository_root_path,
                root_fd,
                int(root_status.st_dev),
                int(root_status.st_ino),
            )
        )
        current_fd = root_fd
        cursor = repository_root_path
        for component in parts[:-1]:
            cursor = cursor / component
            try:
                child_fd = os.open(component, _directory_flags(), dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, 0o755, dir_fd=current_fd)
                    os.fsync(current_fd)
                except FileExistsError:
                    pass
                child_fd = os.open(component, _directory_flags(), dir_fd=current_fd)
            child_status = os.fstat(child_fd)
            if not stat.S_ISDIR(child_status.st_mode):
                os.close(child_fd)
                raise NotADirectoryError(
                    f"repository parent가 directory가 아닙니다: {cursor}"
                )
            opened.append(
                (
                    cursor,
                    child_fd,
                    int(child_status.st_dev),
                    int(child_status.st_ino),
                )
            )
            current_fd = child_fd
        verify_parent_chain(opened)
        return parts[-1], opened
    except BaseException:
        close_parent_chain(opened)
        if not opened:
            os.close(root_fd)
        raise


def verify_parent_chain(chain: list[tuple[Path, int, int, int]]) -> None:
    for path, descriptor, expected_dev, expected_ino in chain:
        fd_status = os.fstat(descriptor)
        lexical_status = os.lstat(path)
        if (
            not stat.S_ISDIR(fd_status.st_mode)
            or not stat.S_ISDIR(lexical_status.st_mode)
            or stat.S_ISLNK(lexical_status.st_mode)
            or (fd_status.st_dev, fd_status.st_ino) != (expected_dev, expected_ino)
            or (lexical_status.st_dev, lexical_status.st_ino)
            != (expected_dev, expected_ino)
        ):
            raise RuntimeError(f"repository directory inode가 변경됐습니다: {path}")


def close_parent_chain(chain: list[tuple[Path, int, int, int]]) -> None:
    for _path, descriptor, _dev, _ino in reversed(chain):
        os.close(descriptor)


def _stable_file_stat(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
        int(status.st_mode),
    )


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


class RepositoryFileGuard:
    """Hold one file and its parent dirfds across a semantic validation."""

    def __init__(
        self,
        repository_root_path: str | os.PathLike[str],
        relative_path: str,
        *,
        label: str = "repository file",
    ) -> None:
        self.root = repository_root(repository_root_path)
        self.relative_path = canonical_relative_path(relative_path, label=label)
        self.label = label
        self._filename = ""
        self._chain: list[tuple[Path, int, int, int]] = []
        self._descriptor = -1
        self._stat: tuple[int, int, int, int, int, int] | None = None
        self._bytes: bytes | None = None

    def __enter__(self) -> "RepositoryFileGuard":
        self._filename, self._chain = open_parent_chain(
            self.root, self.relative_path, create=False
        )
        parent_fd = self._chain[-1][1]
        try:
            self._descriptor = os.open(
                self._filename,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
            opened = os.fstat(self._descriptor)
            named = os.stat(
                self._filename, dir_fd=parent_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(named.st_mode)
                or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
            ):
                raise ValueError(
                    f"{self.label}은 symlink 아닌 regular file이어야 합니다: "
                    f"{self.relative_path}"
                )
            payload = _read_descriptor(self._descriptor)
            after = os.fstat(self._descriptor)
            if _stable_file_stat(opened) != _stable_file_stat(after):
                raise RuntimeError(
                    f"{self.label}이 최초 read 중 변경됐습니다: {self.relative_path}"
                )
            if len(payload) != int(after.st_size):
                raise RuntimeError(
                    f"{self.label} size가 read 결과와 다릅니다: {self.relative_path}"
                )
            self._stat = _stable_file_stat(after)
            self._bytes = payload
            self.verify()
            return self
        except BaseException:
            self.close()
            raise

    @property
    def bytes(self) -> bytes:
        if self._bytes is None:
            raise RuntimeError("repository file guard가 열리지 않았습니다")
        return self._bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.bytes).hexdigest()

    @property
    def size(self) -> int:
        return len(self.bytes)

    @property
    def path(self) -> Path:
        return self.root.joinpath(*PurePosixPath(self.relative_path).parts)

    @property
    def parent_identity(self) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (str(path), int(device), int(inode))
            for path, _descriptor, device, inode in self._chain
        )

    def _verify_pinned(self) -> None:
        if self._descriptor < 0 or self._stat is None or self._bytes is None:
            raise RuntimeError("repository file guard가 열리지 않았습니다")
        parent_fd = self._chain[-1][1]
        before = os.fstat(self._descriptor)
        named = os.stat(
            self._filename, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            _stable_file_stat(before) != self._stat
            or not stat.S_ISREG(named.st_mode)
            or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError(
                f"{self.label} inode/stat가 변경됐습니다: {self.relative_path}"
            )
        payload = _read_descriptor(self._descriptor)
        after = os.fstat(self._descriptor)
        if (
            _stable_file_stat(after) != self._stat
            or payload != self._bytes
            or len(payload) != int(after.st_size)
        ):
            raise RuntimeError(
                f"{self.label} bytes가 변경됐습니다: {self.relative_path}"
            )
        named_after = os.stat(
            self._filename, dir_fd=parent_fd, follow_symlinks=False
        )
        if (named_after.st_dev, named_after.st_ino) != (
            after.st_dev,
            after.st_ino,
        ):
            raise RuntimeError(
                f"{self.label} pathname이 retarget됐습니다: {self.relative_path}"
            )
        verify_parent_chain(self._chain)

    def verify(self) -> None:
        self._verify_pinned()

    def snapshot(self) -> dict[str, Any]:
        self.verify()
        assert self._stat is not None
        return {
            "path": self.relative_path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "device": self._stat[0],
            "inode": self._stat[1],
            "size": self._stat[2],
            "mtime_ns": self._stat[3],
            "ctime_ns": self._stat[4],
            "parent_identity": self.parent_identity,
        }

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1
        if self._chain:
            close_parent_chain(self._chain)
            self._chain = []

    def __exit__(self, _exc_type, _exc, _traceback) -> None:  # noqa: ANN001
        self.close()


def read_repository_file_nofollow(
    repository_root_path: str | os.PathLike[str], relative_path: str
) -> dict[str, Any]:
    with RepositoryFileGuard(repository_root_path, relative_path) as guard:
        return guard.snapshot()


def assert_repository_target_fresh_nofollow(
    repository_root_path: str | os.PathLike[str],
    relative_path: str,
    *,
    create_parents: bool = True,
) -> None:
    """Reject an existing leaf without following any component.

    ``create_parents=False`` is the read-only admission form.  If an unopened
    parent is absent the target is necessarily fresh at that instant.  Live
    publication uses the default to create and fsync its dedicated parent,
    then repeats this check immediately before open.
    """

    root = repository_root(repository_root_path)
    relative = canonical_relative_path(relative_path, label="fresh target")
    try:
        filename, chain = open_parent_chain(root, relative, create=create_parents)
    except FileNotFoundError:
        if not create_parents:
            return
        raise
    try:
        try:
            os.stat(filename, dir_fd=chain[-1][1], follow_symlinks=False)
        except FileNotFoundError:
            verify_parent_chain(chain)
            return
        raise FileExistsError(f"기존 target/symlink를 덮어쓰지 않습니다: {relative}")
    finally:
        close_parent_chain(chain)


def publish_repository_bytes_noreplace(
    repository_root_path: str | os.PathLike[str],
    relative_path: str,
    payload: bytes,
    *,
    mode: int = 0o600,
    preserve_recovery_link: bool = False,
    recovery_tag: str = "v5_raw",
) -> dict[str, Any]:
    """Publish bytes through held dirfds without replacement or symlink traversal.

    The temporary inode and final hard link are created in the already-opened
    parent directory.  A pathname rename/splice therefore cannot redirect the
    write outside the repository.  If the final link was made and a later
    verification fails, the final inode is deliberately preserved.
    """

    if type(payload) is not bytes:
        raise TypeError("published payload는 bytes여야 합니다")
    if (
        type(recovery_tag) is not str
        or not recovery_tag
        or any(not (character.islower() or character.isdigit() or character == "_") for character in recovery_tag)
    ):
        raise ValueError("recovery_tag는 소문자/숫자/underscore만 허용합니다")
    root = repository_root(repository_root_path)
    relative = canonical_relative_path(relative_path, label="publish target")
    filename, chain = open_parent_chain(root, relative, create=True)
    parent_fd = chain[-1][1]
    token = hashlib.sha256(os.urandom(32)).hexdigest()[:24]
    staging = f".{filename}.{token}.partial"
    descriptor = -1
    linked = False
    recovery = f".{filename}.{token}.{recovery_tag}_recovery"
    try:
        try:
            os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                f"기존 target/symlink를 덮어쓰지 않습니다: {relative}"
            )
        descriptor = os.open(
            staging,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            mode,
            dir_fd=parent_fd,
        )
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("repository publish short write")
            written += count
        os.fsync(descriptor)
        staged_status = os.fstat(descriptor)
        if not stat.S_ISREG(staged_status.st_mode):
            raise RuntimeError("repository staging inode가 regular file이 아닙니다")
        verify_parent_chain(chain)
        os.link(
            staging,
            filename,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        linked = True
        os.fsync(parent_fd)
        named = os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(named.st_mode)
            or (named.st_dev, named.st_ino)
            != (staged_status.st_dev, staged_status.st_ino)
        ):
            raise RuntimeError("published target inode가 staging inode와 다릅니다")
        actual = _read_descriptor(descriptor)
        after = os.fstat(descriptor)
        if (
            actual != payload
            or int(after.st_size) != len(payload)
            or (after.st_dev, after.st_ino)
            != (named.st_dev, named.st_ino)
        ):
            raise RuntimeError("published target bytes/inode 검증에 실패했습니다")
        verify_parent_chain(chain)
        recovery_relative: str | None = None
        if preserve_recovery_link:
            os.link(
                staging,
                recovery,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            os.fsync(parent_fd)
            recovery_status = os.stat(
                recovery, dir_fd=parent_fd, follow_symlinks=False
            )
            if (recovery_status.st_dev, recovery_status.st_ino) != (
                after.st_dev,
                after.st_ino,
            ):
                raise RuntimeError("raw recovery hardlink inode가 published inode와 다릅니다")
            recovery_relative = str(
                PurePosixPath(relative).parent / recovery
            )
        os.unlink(staging, dir_fd=parent_fd)
        os.fsync(parent_fd)
        verify_parent_chain(chain)
        named_after = os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
        if (named_after.st_dev, named_after.st_ino) != (
            after.st_dev,
            after.st_ino,
        ):
            raise RuntimeError("published target가 최종 검증 중 교체됐습니다")
        return {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
            "device": int(after.st_dev),
            "inode": int(after.st_ino),
            "recovery_path": recovery_relative,
        }
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        # A successful final link is never removed on a later failure.  Only
        # the private staging name is best-effort cleaned up.
        try:
            os.unlink(staging, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileNotFoundError:
            pass
        except OSError:
            if not linked:
                raise
        # A recovery evidence link is never removed here or by the generation writer.
        # Its continued existence is intentional even on successful publication.
        close_parent_chain(chain)


__all__ = [
    "RepositoryFileGuard",
    "assert_repository_target_fresh_nofollow",
    "canonical_relative_path",
    "external_post_receipt_relative_path",
    "external_post_receipt_relative_path_v5",
    "close_parent_chain",
    "open_parent_chain",
    "publish_repository_bytes_noreplace",
    "read_repository_file_nofollow",
    "repository_root",
    "repository_execution_identity",
    "verify_parent_chain",
]
