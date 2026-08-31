#!/usr/bin/env python3
"""병렬 범위(Range) 다운로더 — 연결당 속도 제한 우회.

사용법: ``pget.py URL OUT [N]``
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import queue
import re
import stat
import sys
import tempfile
import threading
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator


MAX_ATTEMPTS = 8
RETRY_DELAY_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 60
PROGRESS_INTERVAL_SECONDS = 5.0
CHUNK_SIZE = 1 << 20
RESUME_RANGE_BYTES = 64 << 20
MAX_CONNECTIONS = 64
RESUME_SCHEMA_VERSION = 2


class DownloadError(RuntimeError):
    """다운로드를 완전한 파일로 마칠 수 없을 때 발생한다."""


@dataclass(frozen=True)
class _RemoteObject:
    total: int
    etag: str | None
    last_modified: str | None

    @property
    def if_range(self) -> str | None:
        # RFC 7233의 If-Range에는 weak ETag를 사용할 수 없다.
        if self.etag and not self.etag.upper().startswith("W/"):
            return self.etag
        return self.last_modified

    @property
    def resume_validator(self) -> tuple[str, str] | None:
        """프로세스가 바뀌어도 같은 bytes임을 입증하는 strong validator."""

        if self.etag and not self.etag.upper().startswith("W/"):
            return ("etag", self.etag)
        return None


def _header(response: object, name: str) -> str | None:
    """HTTPMessage뿐 아니라 테스트의 단순 dict도 대소문자 없이 읽는다."""

    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    if value is None:
        for key, candidate in headers.items():
            if key.lower() == name.lower():
                value = candidate
                break
    if value is None:
        return None
    return str(value).strip()


@contextmanager
def _exclusive_output_lock(output_path: Path) -> Iterator[None]:
    """같은 출력 경로의 다른 pget을 비차단 방식으로 거부한다.

    잠금 파일은 의도적으로 지우지 않는다. 프로세스가 죽으면 커널이 flock을
    자동 해제하므로 남은 파일은 곧바로 재사용할 수 있다. 반대로 unlock 뒤
    파일을 지우면 다른 프로세스가 이미 잡은 inode와 새 inode에 잠금이 갈라질
    수 있어 활성 잠금을 훼손하게 된다.
    """

    lock_path = Path(f"{output_path}.part.lock")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise DownloadError(f"출력 잠금 파일 열기 실패: {lock_path}: {exc}") from exc

    locked = False
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or int(info.st_nlink) != 1
            or int(info.st_uid) != os.geteuid()
        ):
            raise DownloadError(
                f"출력 잠금 파일 ownership/link 계약 불일치: {lock_path}"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise DownloadError(
                    f"같은 출력 경로의 다운로드가 이미 실행 중입니다: {output_path}"
                ) from exc
            raise DownloadError(f"출력 잠금 획득 실패: {lock_path}: {exc}") from exc
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _retry_pause(attempt: int, attempts: int, delay: float) -> None:
    if attempt + 1 < attempts and delay > 0:
        time.sleep(delay)


def _content_length(
    url: str,
    opener: Callable[..., object],
    attempts: int,
    retry_delay: float,
) -> _RemoteObject:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, method="HEAD")
            with opener(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                value = _header(response, "Content-Length")
                etag = _header(response, "ETag")
                last_modified = _header(response, "Last-Modified")
            if value is None:
                raise DownloadError("HEAD 응답에 Content-Length가 없습니다")
            total = int(value)
            if total < 0:
                raise DownloadError(f"잘못된 Content-Length: {value}")
            return _RemoteObject(total, etag, last_modified)
        except Exception as exc:
            last_error = exc
            print(f"[HEAD] retry {attempt + 1}/{attempts}: {exc}", flush=True)
            _retry_pause(attempt, attempts, retry_delay)
    raise DownloadError(f"HEAD 재시도 소진: {last_error}")


def _status_code(response: object) -> int | None:
    status = getattr(response, "status", None)
    if status is None:
        getcode = getattr(response, "getcode", None)
        if getcode is not None:
            status = getcode()
    return status


def _validate_range_response(
    response: object,
    start: int,
    end: int,
    remote: _RemoteObject,
) -> None:
    status = _status_code(response)
    if status != 206:
        raise DownloadError(
            f"서버가 Range 요청을 무시했습니다 (HTTP {status}, 206 필요)"
        )

    value = _header(response, "Content-Range")
    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+)", value or "", re.I)
    if match is None:
        raise DownloadError(f"잘못된 Content-Range: {value!r}")
    actual = tuple(int(part) for part in match.groups())
    if actual != (start, end, remote.total):
        raise DownloadError(
            f"Content-Range 불일치: {value!r}, "
            f"bytes {start}-{end}/{remote.total} 필요"
        )

    response_etag = _header(response, "ETag")
    validator = remote.resume_validator
    if (
        validator is not None
        and validator[0] == "etag"
        and response_etag != validator[1]
    ):
        raise DownloadError(
            "strong ETag Range 응답 결속 실패: "
            f"HEAD={validator[1]!r}, Range={response_etag!r}"
        )
    if remote.etag is not None and response_etag is not None and response_etag != remote.etag:
        raise DownloadError(
            f"ETag 불일치: HEAD={remote.etag!r}, Range={response_etag!r}"
        )
    response_modified = _header(response, "Last-Modified")
    if validator is None and remote.last_modified and response_modified != remote.last_modified:
        raise DownloadError(
            "Last-Modified Range 응답 결속 실패: "
            f"HEAD={remote.last_modified!r}, Range={response_modified!r}"
        )
    if (
        remote.last_modified is not None
        and response_modified is not None
        and response_modified != remote.last_modified
    ):
        raise DownloadError(
            "Last-Modified 불일치: "
            f"HEAD={remote.last_modified!r}, Range={response_modified!r}"
        )


class _ResumeMismatch(ValueError):
    """정상 파일이지만 현재 원격/layout과 재사용 계약이 맞지 않는다."""


@dataclass(frozen=True)
class _PartIdentity:
    device: int
    inode: int
    size: int


def _range_layout(total: int) -> list[tuple[int, int]]:
    """worker 수와 독립적인 고정 크기 durable-resume 블록을 만든다."""

    return [
        (start, min(total - 1, start + RESUME_RANGE_BYTES - 1))
        for start in range(0, total, RESUME_RANGE_BYTES)
    ]


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_kind(path: Path) -> str:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return "missing"
    if stat.S_ISLNK(info.st_mode):
        return "symlink"
    if stat.S_ISREG(info.st_mode):
        return "regular"
    return "non-regular"


def _require_missing_or_regular(path: Path, *, label: str) -> bool:
    kind = _path_kind(path)
    if kind == "missing":
        return False
    if kind != "regular":
        raise DownloadError(f"{label} 경로가 symlink/non-regular입니다: {path}")
    info = path.lstat()
    if int(info.st_nlink) != 1 or int(info.st_uid) != os.geteuid():
        raise DownloadError(f"{label} ownership/link 계약 불일치: {path}")
    return True


def _open_part(path: Path, *, create: bool, total: int) -> tuple[int, _PartIdentity]:
    flags = os.O_RDWR
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise DownloadError(f"임시 파일 안전 열기 실패: {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or int(info.st_nlink) != 1
            or int(info.st_uid) != os.geteuid()
        ):
            raise DownloadError(
                f"임시 파일 ownership/link 계약 불일치: {path}"
            )
        if create:
            os.ftruncate(descriptor, total)
            os.fsync(descriptor)
            info = os.fstat(descriptor)
        if int(info.st_size) != total:
            raise _ResumeMismatch(
                f"part 크기 불일치: {info.st_size} != {total}: {path}"
            )
        identity = _PartIdentity(
            device=int(info.st_dev), inode=int(info.st_ino), size=int(info.st_size)
        )
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _validate_open_part(
    descriptor: int, identity: _PartIdentity, *, path: Path
) -> None:
    info = os.fstat(descriptor)
    current = _PartIdentity(
        device=int(info.st_dev), inode=int(info.st_ino), size=int(info.st_size)
    )
    if not stat.S_ISREG(info.st_mode) or current != identity:
        raise DownloadError(f"다운로드 중 part inode/size가 변경됐습니다: {path}")


def _canonical_state_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _seal_state(payload: dict[str, object]) -> dict[str, object]:
    sealed = {key: value for key, value in payload.items() if key != "state_sha256"}
    sealed["state_sha256"] = hashlib.sha256(_canonical_state_bytes(sealed)).hexdigest()
    return sealed


def _write_resume_state(state_path: Path, payload: dict[str, object]) -> None:
    _require_missing_or_regular(state_path, label="resume sidecar")
    sealed = _seal_state(payload)
    data = _canonical_state_bytes(sealed) + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=state_path.parent,
            prefix=f".{state_path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, state_path)
        temporary = None
        _fsync_directory(state_path.parent)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise DownloadError(f"resume sidecar 원자 저장 실패: {state_path}: {exc}") from exc


def _read_resume_state(state_path: Path) -> dict[str, object]:
    if not _require_missing_or_regular(state_path, label="resume sidecar"):
        raise _ResumeMismatch(f"resume sidecar가 없습니다: {state_path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(state_path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            data = handle.read(4 * 1024 * 1024 + 1)
    except OSError as exc:
        raise DownloadError(f"resume sidecar 안전 읽기 실패: {state_path}: {exc}") from exc
    if not data or len(data) > 4 * 1024 * 1024:
        raise _ResumeMismatch("resume sidecar 크기가 비었거나 제한을 초과했습니다")
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _ResumeMismatch(f"resume sidecar JSON 손상: {exc}") from exc
    if not isinstance(payload, dict):
        raise _ResumeMismatch("resume sidecar root는 object여야 합니다")
    digest = payload.get("state_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "state_sha256"}
    actual = hashlib.sha256(_canonical_state_bytes(unsigned)).hexdigest()
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise _ResumeMismatch("resume sidecar state_sha256가 없습니다")
    if digest != actual:
        raise _ResumeMismatch("resume sidecar self digest 불일치")
    return payload


def _hash_part_range(
    part_path: Path,
    identity: _PartIdentity,
    low: int,
    high: int,
) -> str:
    expected = max(0, high - low + 1)
    descriptor, current = _open_part(part_path, create=False, total=identity.size)
    try:
        if current != identity:
            raise DownloadError(f"part inode가 검증 중 바뀌었습니다: {part_path}")
        _validate_open_part(descriptor, identity, path=part_path)
        os.lseek(descriptor, low, os.SEEK_SET)
        remaining = expected
        digest = hashlib.sha256()
        while remaining:
            chunk = os.read(descriptor, min(CHUNK_SIZE, remaining))
            if not chunk:
                raise DownloadError(
                    f"완료 range local read 부족: {low}-{high}, remaining={remaining}"
                )
            digest.update(chunk)
            remaining -= len(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _expected_resume_state(
    *,
    url: str,
    remote: _RemoteObject,
    ranges: list[tuple[int, int]],
    part_path: Path,
) -> dict[str, object]:
    validator = remote.resume_validator
    if validator is None:
        raise DownloadError("durable resume state에는 strong ETag가 필요합니다")
    return {
        "schema_version": RESUME_SCHEMA_VERSION,
        # signed URL query를 sidecar에 평문으로 남기지 않되 exact URL identity는 결속한다.
        "url_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
        "total": remote.total,
        "validator": {"kind": validator[0], "value": validator[1]},
        "range_bytes": RESUME_RANGE_BYTES,
        "ranges": [[low, high] for low, high in ranges],
        "part_name": part_path.name,
        "completed": [False] * len(ranges),
        "range_sha256": [None] * len(ranges),
    }


def _validate_resume_state(
    payload: dict[str, object],
    expected: dict[str, object],
    *,
    part_path: Path,
    identity: _PartIdentity,
    ranges: list[tuple[int, int]],
) -> None:
    expected_keys = set(expected) | {"state_sha256"}
    if set(payload) != expected_keys:
        raise _ResumeMismatch("resume sidecar field set 불일치")

    integer_fields = ("schema_version", "total", "range_bytes")
    if any(type(payload.get(field)) is not int for field in integer_fields):
        raise _ResumeMismatch("resume 정수 field 타입 불일치")
    string_fields = ("url_sha256", "part_name", "state_sha256")
    if any(type(payload.get(field)) is not str for field in string_fields):
        raise _ResumeMismatch("resume 문자열 field 타입 불일치")
    if not re.fullmatch(r"[0-9a-f]{64}", payload["url_sha256"]):
        raise _ResumeMismatch("resume url_sha256 형식 불일치")

    validator = payload.get("validator")
    expected_validator = expected["validator"]
    if (
        type(validator) is not dict
        or set(validator) != {"kind", "value"}
        or type(validator.get("kind")) is not str
        or type(validator.get("value")) is not str
        or validator != expected_validator
    ):
        raise _ResumeMismatch("resume validator 구조/identity 불일치")

    saved_ranges = payload.get("ranges")
    if type(saved_ranges) is not list or len(saved_ranges) != len(ranges):
        raise _ResumeMismatch("resume ranges 구조 불일치")
    for index, (saved, expected_range) in enumerate(
        zip(saved_ranges, ranges, strict=True)
    ):
        if (
            type(saved) is not list
            or len(saved) != 2
            or any(type(bound) is not int for bound in saved)
            or tuple(saved) != expected_range
        ):
            raise _ResumeMismatch(f"resume range layout 불일치: index={index}")

    for key, value in expected.items():
        if key in {"validator", "ranges", "completed", "range_sha256"}:
            continue
        if type(payload.get(key)) is not type(value) or payload.get(key) != value:
            raise _ResumeMismatch(f"resume identity/layout 불일치: {key}")
    completed = payload.get("completed")
    digests = payload.get("range_sha256")
    if (
        not isinstance(completed, list)
        or len(completed) != len(ranges)
        or any(type(value) is not bool for value in completed)
        or not isinstance(digests, list)
        or len(digests) != len(ranges)
    ):
        raise _ResumeMismatch("resume completed/range_sha256 구조 불일치")
    for index, ((low, high), is_complete, digest) in enumerate(
        zip(ranges, completed, digests, strict=True)
    ):
        if high < low:
            if not is_complete or digest != hashlib.sha256(b"").hexdigest():
                raise _ResumeMismatch(f"empty range 상태 불일치: index={index}")
            continue
        if not is_complete:
            if digest is not None:
                raise _ResumeMismatch(f"미완료 range에 digest가 있습니다: index={index}")
            continue
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise _ResumeMismatch(f"완료 range digest가 유효하지 않습니다: index={index}")
        actual = _hash_part_range(part_path, identity, low, high)
        if actual != digest:
            raise _ResumeMismatch(f"완료 range local SHA-256 불일치: index={index}")


def _quarantine_partial(paths: list[Path], *, reason: str) -> None:
    existing = [
        path
        for path in paths
        if _require_missing_or_regular(path, label="quarantine 대상")
    ]
    if not existing:
        return
    parents = {path.parent.resolve() for path in existing}
    if len(parents) != 1:
        raise DownloadError("quarantine 대상은 같은 디렉터리에 있어야 합니다")
    quarantine_dir: Path | None = None
    try:
        # mkdtemp는 mode 0700의 고유 디렉터리를 원자적으로 생성한다. 외부에
        # 예측 가능한 target 이름을 만든 뒤 os.replace하는 TOCTOU/overwrite를
        # 피하고, freshly-created private directory 안으로만 rename한다.
        quarantine_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{existing[0].name}.quarantine.",
                dir=existing[0].parent,
            )
        )
        for source in existing:
            target = quarantine_dir / source.name
            if _path_kind(target) != "missing":
                raise DownloadError(f"quarantine 내부 target 충돌: {target}")
            source.rename(target)
        _fsync_directory(quarantine_dir)
        _fsync_directory(existing[0].parent)
    except (OSError, DownloadError) as exc:
        raise DownloadError(f"stale partial quarantine 실패: {exc}") from exc
    print(
        f"[resume] 기존 partial을 quarantine하고 새로 시작합니다: {reason}; "
        f"{quarantine_dir}",
        flush=True,
    )


def _remove_regular(path: Path) -> bool:
    try:
        if not _require_missing_or_regular(path, label="임시 파일"):
            return True
        path.unlink()
        _fsync_directory(path.parent)
        return True
    except (OSError, DownloadError) as exc:
        print(f"[WARN] 임시 파일 삭제 실패: {path}: {exc}", flush=True)
        return False


def _prepare_partial(
    *,
    url: str,
    output_path: Path,
    remote: _RemoteObject,
    ranges: list[tuple[int, int]],
) -> tuple[Path, Path, _PartIdentity, dict[str, object] | None, bool]:
    part_path = Path(f"{output_path}.part")
    state_path = Path(f"{output_path}.part.state.json")
    _require_missing_or_regular(output_path, label="최종 출력")
    part_exists = _require_missing_or_regular(part_path, label="part")
    state_exists = _require_missing_or_regular(state_path, label="resume sidecar")
    resumable = remote.resume_validator is not None

    if part_exists or state_exists:
        if not resumable:
            _quarantine_partial(
                [part_path, state_path],
                reason="원격에 strong ETag가 없어 기존 partial을 식별할 수 없음",
            )
        elif not part_exists or not state_exists:
            _quarantine_partial(
                [part_path, state_path], reason="part/sidecar 중 하나만 존재함"
            )
        else:
            descriptor = -1
            try:
                payload = _read_resume_state(state_path)
                descriptor, identity = _open_part(
                    part_path, create=False, total=remote.total
                )
                os.close(descriptor)
                descriptor = -1
                expected = _expected_resume_state(
                    url=url,
                    remote=remote,
                    ranges=ranges,
                    part_path=part_path,
                )
                _validate_resume_state(
                    payload,
                    expected,
                    part_path=part_path,
                    identity=identity,
                    ranges=ranges,
                )
                print(
                    "[resume] 검증된 완료 range를 재사용합니다: "
                    f"{sum(bool(value) for value in payload['completed'])}/{len(ranges)}",
                    flush=True,
                )
                return part_path, state_path, identity, payload, True
            except _ResumeMismatch as exc:
                if descriptor >= 0:
                    os.close(descriptor)
                _quarantine_partial([part_path, state_path], reason=str(exc))
            except BaseException:
                if descriptor >= 0:
                    os.close(descriptor)
                raise

    descriptor, identity = _open_part(part_path, create=True, total=remote.total)
    os.close(descriptor)
    state: dict[str, object] | None = None
    if resumable:
        state = _expected_resume_state(
            url=url,
            remote=remote,
            ranges=ranges,
            part_path=part_path,
        )
        _write_resume_state(state_path, state)
    return part_path, state_path, identity, state, False


def _download_locked(
    url: str,
    output_path: Path,
    connections: int,
    *,
    opener: Callable[..., object],
    attempts: int,
    retry_delay: float,
    progress_interval: float,
) -> None:
    remote = _content_length(url, opener, attempts, retry_delay)
    total = remote.total
    ranges = _range_layout(total)
    print(f"total {total / 1e9:.2f} GB, {connections} connections", flush=True)
    part_path, state_path, part_identity, resume_state, resumed = _prepare_partial(
        url=url,
        output_path=output_path,
        remote=remote,
        ranges=ranges,
    )
    completed = (
        [bool(value) for value in resume_state["completed"]]
        if resume_state is not None
        else [False] * len(ranges)
    )
    done = [
        max(0, high - low + 1) if completed[index] else 0
        for index, (low, high) in enumerate(ranges)
    ]
    errors: list[str | None] = [None] * len(ranges)
    state_lock = threading.Lock()

    def mark_complete(index: int, low: int, high: int) -> None:
        descriptor, current = _open_part(part_path, create=False, total=total)
        try:
            if current != part_identity:
                raise DownloadError(f"part inode가 완료 처리 중 바뀌었습니다: {part_path}")
            _validate_open_part(descriptor, part_identity, path=part_path)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if resume_state is None:
            completed[index] = True
            return
        digest = _hash_part_range(part_path, part_identity, low, high)
        with state_lock:
            next_completed = list(completed)
            next_completed[index] = True
            next_digests = list(resume_state["range_sha256"])
            next_digests[index] = digest
            next_state = dict(resume_state)
            next_state["completed"] = next_completed
            next_state["range_sha256"] = next_digests
            # part fsync -> sidecar atomic replace 순서를 지킨 뒤에만 메모리
            # bitmap도 완료로 승격한다. sidecar 저장 실패 시 재시도/재실행은
            # 해당 range를 다시 받으며 검증되지 않은 진행을 신뢰하지 않는다.
            _write_resume_state(state_path, next_state)
            completed[:] = next_completed
            resume_state.clear()
            resume_state.update(next_state)

    def download_range(index: int) -> None:
        low, high = ranges[index]
        expected = high - low + 1
        if expected <= 0:
            return
        if completed[index]:
            return

        last_error: Exception | None = None
        for attempt in range(attempts):
            connection_start = done[index]
            start = low + connection_start
            try:
                headers = {
                    "Range": f"bytes={start}-{high}",
                    # byte range와 sidecar SHA는 encoded representation이 아니라
                    # HEAD Content-Length가 설명한 identity bytes에 결속한다.
                    "Accept-Encoding": "identity",
                }
                if remote.if_range is not None:
                    headers["If-Range"] = remote.if_range
                request = urllib.request.Request(url, headers=headers)
                with opener(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                    _validate_range_response(response, start, high, remote)
                    descriptor, current = _open_part(
                        part_path, create=False, total=total
                    )
                    if current != part_identity:
                        os.close(descriptor)
                        raise DownloadError(
                            f"part inode가 range write 전에 바뀌었습니다: {part_path}"
                        )
                    with os.fdopen(descriptor, "r+b", buffering=0) as partial:
                        partial.seek(start)
                        remaining = expected - done[index]
                        while True:
                            # remaining+1을 허용해 요청 범위보다 많은 응답도 검출한다.
                            chunk = response.read(min(CHUNK_SIZE, remaining + 1))
                            if not chunk:
                                if remaining:
                                    raise DownloadError(
                                        f"범위 {low}-{high} 응답 부족 "
                                        f"({done[index]}/{expected} bytes)"
                                    )
                                mark_complete(index, low, high)
                                return
                            if len(chunk) > remaining:
                                done[index] = connection_start
                                raise DownloadError(
                                    f"범위 {low}-{high} 응답 초과 "
                                    f"({len(chunk)} > 남은 {remaining} bytes)"
                                )
                            written = partial.write(chunk)
                            if written != len(chunk):
                                raise OSError(
                                    f"임시 파일 쓰기 부족 ({written}/{len(chunk)} bytes)"
                                )
                            done[index] += written
                            remaining -= written
            except Exception as exc:
                # 필요한 바이트를 다 받았어도 EOF 확인에 실패했다면 그 연결분은
                # 검증되지 않은 것이므로 다음 시도에서 다시 받는다.
                if done[index] == expected:
                    done[index] = connection_start
                last_error = exc
                print(
                    f"[{index}] retry {attempt + 1}/{attempts}: {exc}", flush=True
                )
                _retry_pause(attempt, attempts, retry_delay)

        errors[index] = f"범위 {low}-{high} 재시도 소진: {last_error}"

    work: queue.Queue[int] = queue.Queue()
    for index, is_complete in enumerate(completed):
        if not is_complete:
            work.put(index)

    def worker() -> None:
        while True:
            try:
                index = work.get_nowait()
            except queue.Empty:
                return
            try:
                download_range(index)
            finally:
                work.task_done()

    worker_count = min(connections, work.qsize())
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_count)]
    started = time.monotonic()
    for thread in threads:
        thread.start()

    while any(thread.is_alive() for thread in threads):
        deadline = time.monotonic() + progress_interval
        for thread in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)
        received = sum(done)
        elapsed = max(1.0, time.monotonic() - started)
        print(
            f"{received / 1e9:.2f}/{total / 1e9:.2f} GB  "
            f"{received / 1e6 / elapsed:.1f} MB/s",
            flush=True,
        )

    received = sum(done)
    failures = [error for error in errors if error is not None]
    if failures or received != total:
        verified_partial = any(
            completed[index] and high >= low
            for index, (low, high) in enumerate(ranges)
        )
        if resume_state is not None and verified_partial:
            print(
                f"[resume] 검증된 partial을 보존합니다: {part_path}, {state_path}",
                flush=True,
            )
        else:
            _remove_regular(part_path)
            _remove_regular(state_path)
        detail = "; ".join(failures) if failures else "스레드 완료 후 바이트 수 불일치"
        raise DownloadError(f"다운로드 실패 ({received}/{total} bytes): {detail}")

    try:
        if resume_state is not None and not all(completed):
            raise DownloadError("완료 직전 resume bitmap에 미완료 range가 있습니다")
        descriptor, current = _open_part(part_path, create=False, total=total)
        try:
            if current != part_identity:
                raise DownloadError(f"part inode가 최종 교체 전에 바뀌었습니다: {part_path}")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _require_missing_or_regular(output_path, label="최종 출력")
        os.replace(part_path, output_path)
        _fsync_directory(output_path.parent)
        if not _remove_regular(state_path):
            # 최종 output은 이미 rename+directory fsync로 publish되었다. 이 뒤의
            # stale sidecar 정리 실패를 다운로드 실패로 되돌리면 wrapper가 정상
            # archive를 다시 받을 수 있으므로 경고만 남기고 성공으로 종료한다.
            print(
                f"[WARN] 완성 파일은 정상이나 resume sidecar가 남았습니다: {state_path}",
                flush=True,
            )
    except OSError as exc:
        raise DownloadError(f"완성 파일 교체 실패: {exc}") from exc
    print("DONE" + (" (resumed)" if resumed else ""), flush=True)


def download(
    url: str,
    output: str | os.PathLike[str],
    connections: int = 16,
    *,
    opener: Callable[..., object] | None = None,
    attempts: int | None = None,
    retry_delay: float | None = None,
    progress_interval: float | None = None,
) -> None:
    """URL을 병렬 Range 요청으로 받아 성공 시에만 *output*으로 교체한다."""

    if connections <= 0:
        raise DownloadError("연결 수 N은 1 이상이어야 합니다")
    if connections > MAX_CONNECTIONS:
        raise DownloadError(
            f"연결 수 N은 {MAX_CONNECTIONS} 이하여야 합니다: {connections}"
        )
    attempts = MAX_ATTEMPTS if attempts is None else attempts
    retry_delay = RETRY_DELAY_SECONDS if retry_delay is None else retry_delay
    progress_interval = (
        PROGRESS_INTERVAL_SECONDS if progress_interval is None else progress_interval
    )
    if attempts <= 0:
        raise DownloadError("재시도 횟수는 1 이상이어야 합니다")
    if progress_interval <= 0:
        raise DownloadError("진행 로그 주기는 0보다 커야 합니다")
    if opener is None:
        opener = urllib.request.urlopen

    output_path = Path(output)
    with _exclusive_output_lock(output_path):
        _download_locked(
            url,
            output_path,
            connections,
            opener=opener,
            attempts=attempts,
            retry_delay=retry_delay,
            progress_interval=progress_interval,
        )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in (2, 3):
        print("사용법: pget.py URL OUT [N]", file=sys.stderr)
        return 2
    try:
        connections = int(args[2]) if len(args) == 3 else 16
        download(args[0], args[1], connections)
    except (ValueError, DownloadError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
