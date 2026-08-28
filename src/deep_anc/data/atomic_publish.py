"""표준 라이브러리만 사용하는 immutable directory 발행 helper."""

from __future__ import annotations

import ctypes
import errno
import os
import stat
from pathlib import Path


def publish_directory_noreplace(staging: str | Path, target: str | Path) -> Path:
    """same-filesystem staging directory를 renameat2(NOREPLACE)로 발행한다."""

    source = Path(os.path.abspath(Path(staging)))
    destination = Path(os.path.abspath(Path(target)))
    if source.parent != destination.parent:
        raise ValueError("atomic directory publication은 sibling staging만 허용합니다")
    source_stat = source.lstat()
    if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISDIR(source_stat.st_mode):
        raise ValueError("atomic directory staging은 실제 directory여야 합니다")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"artifact directory를 덮어쓸 수 없습니다: {destination}")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("canonical no-replace directory publish에 renameat2가 필요합니다")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(
                f"artifact directory를 덮어쓸 수 없습니다: {destination}"
            )
        raise OSError(error, os.strerror(error), str(destination))
    directory_fd = os.open(
        destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return destination


__all__ = ["publish_directory_noreplace"]
