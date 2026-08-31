#!/usr/bin/env python3
"""Jetson의 검증된 transfer bundle을 Elice exact checkout으로 전송한다.

이 도구는 transfer manifest가 열거한 regular file과 manifest 자체만 ``rsync``로
보낸다. 로컬 full semantic validator, 양쪽 exact Git checkout, 원격 파일 SHA를
각각 검증하며 삭제 옵션은 사용하지 않는다. ``--dry-run``은 로컬 검증까지만 하고
SSH/rsync를 호출하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deep_anc.data.holdout_contract import (  # noqa: E402
    HoldoutContractError,
    read_regular_file_snapshot,
)
from deep_anc.data.source_trust import (  # noqa: E402
    SourceTrustError,
    exact_clean_source_evidence,
)
from deep_anc.data.transfer_contract import (  # noqa: E402
    TransferContractError,
    validate_transfer_manifest,
)


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HOST_RE = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$"
)
_USER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,31}$")
_SAFE_REMOTE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_CANONICAL_MANIFEST = "data/manifests/elice_transfer_manifest.json"
_PEM_BEGIN = b"-----BEGIN "
_PEM_PRIVATE_SUFFIX = b"PRIVATE KEY-----"
_PRIVATE_KEY_HEADERS = {
    _PEM_BEGIN + b"OPENSSH " + _PEM_PRIVATE_SUFFIX,
    _PEM_BEGIN + _PEM_PRIVATE_SUFFIX,
    _PEM_BEGIN + b"RSA " + _PEM_PRIVATE_SUFFIX,
    _PEM_BEGIN + b"EC " + _PEM_PRIVATE_SUFFIX,
}


class PushTransferError(RuntimeError):
    """전송 admission 또는 무결성 실패."""


@dataclass(frozen=True)
class FileEntry:
    path: str
    sha256: str
    size: int

    def payload(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class LocalBundle:
    commit: str
    manifest: FileEntry
    files: tuple[FileEntry, ...]
    schema_version: int

    @property
    def transfer_paths(self) -> tuple[str, ...]:
        return tuple(entry.path for entry in self.files) + (self.manifest.path,)

    @property
    def total_bytes(self) -> int:
        return sum(entry.size for entry in self.files) + self.manifest.size


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int
    user: str
    identity: Path
    remote_repo: str

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


def _safe_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise PushTransferError(f"{label} 상대경로가 유효하지 않습니다")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PushTransferError(f"{label} 상대경로가 저장소 밖을 가리킵니다")
    if any("\n" in part or "\r" in part or "\t" in part for part in path.parts):
        raise PushTransferError(f"{label} 상대경로에 제어문자가 있습니다")
    return path.as_posix()


def _validate_remote_repo(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or value == "/":
        raise PushTransferError("--remote-repo는 root가 아닌 절대 POSIX 경로여야 합니다")
    if "\\" in value or "\x00" in value or any(character.isspace() for character in value):
        raise PushTransferError("--remote-repo에 공백·제어문자·역슬래시를 허용하지 않습니다")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise PushTransferError("--remote-repo 경로가 정규화되지 않았습니다")
    if any(_SAFE_REMOTE_COMPONENT_RE.fullmatch(part) is None for part in path.parts[1:]):
        raise PushTransferError("--remote-repo에는 영숫자/._- 경로 성분만 허용합니다")
    normalized = path.as_posix()
    if normalized != value.rstrip("/"):
        raise PushTransferError("--remote-repo 경로가 정규화되지 않았습니다")
    return normalized


def _validate_identity(path_value: str, *, repo_root: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        raise PushTransferError("--identity는 절대경로여야 합니다")
    absolute = Path(os.path.abspath(candidate))
    try:
        resolved = absolute.resolve(strict=True)
        if resolved != absolute:
            raise PushTransferError("SSH identity 경로에 symlink component를 허용하지 않습니다")
        descriptor = os.open(
            absolute,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise PushTransferError("SSH identity 파일을 읽을 수 없습니다") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise PushTransferError("SSH identity는 regular file이어야 합니다")
        if info.st_uid != os.getuid():
            raise PushTransferError("SSH identity 소유자가 현재 사용자와 다릅니다")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise PushTransferError("SSH identity 권한은 group/other 접근을 허용하면 안 됩니다")
        if info.st_size < 128 or info.st_size > 64 * 1024:
            raise PushTransferError("SSH identity 파일 크기가 private-key 범위를 벗어납니다")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            header = handle.readline(256).rstrip(b"\r\n")
    except OSError as exc:
        raise PushTransferError("SSH identity header를 읽을 수 없습니다") from exc
    finally:
        os.close(descriptor)
    if header not in _PRIVATE_KEY_HEADERS:
        raise PushTransferError("SSH identity가 지원하는 private-key PEM/OpenSSH 형식이 아닙니다")

    try:
        relative = resolved.relative_to(repo_root)
    except ValueError:
        return resolved
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
            cwd=repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        raise PushTransferError("SSH identity Git tracked 검사를 실행할 수 없습니다") from exc
    if tracked.returncode == 0:
        raise PushTransferError("SSH identity가 Git tracked 파일이면 전송할 수 없습니다")
    try:
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--", relative.as_posix()],
            cwd=repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        raise PushTransferError("SSH identity Git ignore 검사를 실행할 수 없습니다") from exc
    if ignored.returncode != 0:
        raise PushTransferError("저장소 안 SSH identity는 .gitignore로 명시 차단되어야 합니다")
    return resolved


def _endpoint_from_args(args: argparse.Namespace, *, repo_root: Path) -> Endpoint:
    if _HOST_RE.fullmatch(args.host or "") is None:
        raise PushTransferError("--host 형식이 유효하지 않습니다")
    if _USER_RE.fullmatch(args.user or "") is None:
        raise PushTransferError("--user 형식이 유효하지 않습니다")
    if type(args.port) is not int or not (1 <= args.port <= 65535):
        raise PushTransferError("--port는 1~65535 정수여야 합니다")
    return Endpoint(
        host=args.host,
        port=args.port,
        user=args.user,
        identity=_validate_identity(args.identity, repo_root=repo_root),
        remote_repo=_validate_remote_repo(args.remote_repo),
    )


def _load_inventory(
    manifest_raw: bytes,
    *,
    manifest_sha256: str,
    manifest_size: int,
) -> tuple[int, tuple[FileEntry, ...]]:
    try:
        payload = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PushTransferError("transfer manifest JSON을 다시 읽을 수 없습니다") from exc
    if not isinstance(payload, Mapping) or type(payload.get("schema_version")) is not int:
        raise PushTransferError("transfer manifest schema_version이 없습니다")
    schema_version = int(payload["schema_version"])
    if schema_version not in {1, 2}:
        raise PushTransferError("transfer manifest schema_version은 1 또는 2여야 합니다")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise PushTransferError("transfer manifest files가 비었습니다")
    entries: list[FileEntry] = []
    seen: set[str] = set()
    for index, raw_entry in enumerate(raw_files):
        if not isinstance(raw_entry, Mapping):
            raise PushTransferError(f"transfer files[{index}] 구조가 유효하지 않습니다")
        path = _safe_relative_path(raw_entry.get("path"), label=f"files[{index}]")
        sha256 = str(raw_entry.get("sha256") or "").lower()
        size = raw_entry.get("size")
        if _SHA256_RE.fullmatch(sha256) is None or type(size) is not int or size < 0:
            raise PushTransferError(f"transfer files[{index}] SHA/size가 유효하지 않습니다")
        if path in seen or path == _CANONICAL_MANIFEST:
            raise PushTransferError("transfer files 경로가 중복되거나 manifest 자체를 포함합니다")
        seen.add(path)
        entries.append(FileEntry(path=path, sha256=sha256, size=size))
    entries.sort(key=lambda entry: entry.path)
    if hashlib.sha256(manifest_raw).hexdigest() != manifest_sha256 or len(manifest_raw) != manifest_size:
        raise PushTransferError("검증 뒤 transfer manifest bytes가 바뀌었습니다")
    return schema_version, tuple(entries)


def _validate_local_bundle(
    *,
    repo_root: Path,
    expected_commit: str,
    expected_manifest_sha256: str,
    manifest_relative: str,
) -> LocalBundle:
    try:
        source = exact_clean_source_evidence(
            repo_root,
            expected_commit=expected_commit,
            reject_runtime_bytecode=False,
        )
    except SourceTrustError as exc:
        raise PushTransferError("로컬 exact clean checkout 검증 실패") from exc
    manifest_path = repo_root / manifest_relative
    try:
        summary = validate_transfer_manifest(
            manifest_path,
            repo_root=repo_root,
            expected_sha256=expected_manifest_sha256,
        )
        snapshot = read_regular_file_snapshot(
            manifest_path,
            root=repo_root,
            label="Jetson transfer manifest post-validation snapshot",
            capture_bytes=True,
        )
    except (OSError, HoldoutContractError, TransferContractError) as exc:
        raise PushTransferError("로컬 transfer bundle full semantic 검증 실패") from exc
    assert snapshot.data is not None
    if snapshot.sha256 != expected_manifest_sha256:
        raise PushTransferError("로컬 transfer manifest 외부 SHA anchor가 다릅니다")
    schema_version, files = _load_inventory(
        snapshot.data,
        manifest_sha256=snapshot.sha256,
        manifest_size=snapshot.size,
    )
    if summary.get("manifest_sha256") != snapshot.sha256:
        raise PushTransferError("full validator와 post-validation manifest snapshot이 다릅니다")
    if summary.get("file_count") != len(files):
        raise PushTransferError("full validator와 rsync inventory file_count가 다릅니다")
    return LocalBundle(
        commit=str(source["commit"]),
        manifest=FileEntry(
            path=manifest_relative,
            sha256=snapshot.sha256,
            size=snapshot.size,
        ),
        files=files,
        schema_version=schema_version,
    )


# 원격에서는 프로젝트 module을 import하지 않는다. 전송 전후에 stdlib만으로 exact
# tracked checkout과 target path/symlink 경계를 확인하고, post 단계에서 manifest에
# 열거한 모든 byte의 size/SHA를 재검산한다.
_REMOTE_CHECK_PROGRAM = r'''
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath

def fail(message):
    print("REMOTE_CHECK_FAIL:" + message, file=sys.stderr)
    raise SystemExit(3)

def git(root, arguments):
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        return subprocess.run(
            ["git", f"--git-dir={root / '.git'}", f"--work-tree={root}",
             "-c", f"core.worktree={root}", *arguments],
            cwd=root, env=env, check=True, capture_output=True, timeout=180,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        fail("git")

def relative(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        fail("path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        fail("path")
    return path

try:
    payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
except Exception:
    fail("payload")
if not isinstance(payload, dict) or set(payload) != {
    "schema_version", "phase", "remote_repo", "expected_commit", "files"
} or payload.get("schema_version") != 1 or payload.get("phase") not in {"pre", "post"}:
    fail("schema")
root_text = payload.get("remote_repo")
if not isinstance(root_text, str) or not root_text.startswith("/"):
    fail("root")
root = Path(root_text)
try:
    if root.resolve(strict=True) != root or root.is_symlink():
        fail("root")
    git_info = (root / ".git").lstat()
except OSError:
    fail("root")
if not stat.S_ISDIR(git_info.st_mode) or (root / ".git").is_symlink():
    fail("gitdir")
commit = str(payload.get("expected_commit") or "")
if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
    fail("commit")
head = git(root, ["rev-parse", "--verify", "HEAD^{commit}"]).decode("ascii").strip().lower()
top = git(root, ["rev-parse", "--show-toplevel"]).decode("utf-8").strip()
git_dir = git(root, ["rev-parse", "--absolute-git-dir"]).decode("utf-8").strip()
if head != commit or Path(top).resolve() != root or Path(git_dir).resolve() != root / ".git":
    fail("head")
if git(root, ["replace", "-l"]).strip():
    fail("replace")
grafts = root / ".git/info/grafts"
if grafts.is_file() and grafts.stat().st_size:
    fail("grafts")
if git(root, ["status", "--porcelain=v1", "-z", "--untracked-files=all"]):
    fail("dirty")
flags = git(root, ["ls-files", "-v", "-z"])
if any(row and (row[:1].islower() or row[:1] == b"S") for row in flags.split(b"\0")):
    fail("index-flags")

object_format = git(root, ["rev-parse", "--show-object-format"]).decode("ascii").strip()
tree_rows = []
for record in git(root, ["ls-tree", "-r", "-z", "--full-tree", commit]).split(b"\0"):
    if not record:
        continue
    try:
        metadata, raw_path = record.split(b"\t", 1)
        mode, kind, object_id = metadata.decode("ascii").split()
        path_text = raw_path.decode("utf-8")
    except Exception:
        fail("tree")
    if kind != "blob" or mode not in {"100644", "100755", "120000"}:
        fail("tree")
    tree_rows.append((path_text, mode, object_id))
index_rows = []
for record in git(root, ["ls-files", "--stage", "-z"]).split(b"\0"):
    if not record:
        continue
    try:
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_id, stage = metadata.decode("ascii").split()
        path_text = raw_path.decode("utf-8")
    except Exception:
        fail("index")
    if stage != "0":
        fail("index")
    index_rows.append((path_text, mode, object_id))
if sorted(tree_rows) != sorted(index_rows):
    fail("index")
for path_text, mode, object_id in tree_rows:
    path = root / path_text
    try:
        info = path.lstat()
        if mode == "120000":
            if not stat.S_ISLNK(info.st_mode):
                fail("tracked")
            content = os.fsencode(os.readlink(path))
        else:
            if not stat.S_ISREG(info.st_mode) or path.is_symlink():
                fail("tracked")
            if bool(info.st_mode & stat.S_IXUSR) != (mode == "100755"):
                fail("tracked")
            content = path.read_bytes()
    except OSError:
        fail("tracked")
    digest = hashlib.new(object_format)
    digest.update(f"blob {len(content)}\0".encode("ascii"))
    digest.update(content)
    if digest.hexdigest() != object_id:
        fail("tracked")

raw_files = payload.get("files")
if not isinstance(raw_files, list) or not raw_files or shutil.which("rsync") is None:
    fail("files")
seen = set()
for item in raw_files:
    if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"}:
        fail("files")
    rel = relative(item.get("path"))
    path_text = rel.as_posix()
    sha256 = item.get("sha256")
    size = item.get("size")
    if path_text in seen or not isinstance(sha256, str) or len(sha256) != 64 \
       or any(character not in "0123456789abcdef" for character in sha256) \
       or type(size) is not int or size < 0:
        fail("files")
    seen.add(path_text)
    current = root
    for part in rel.parts[:-1]:
        current = current / part
        if not os.path.lexists(current):
            break
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            fail("target-parent")
    target = root / path_text
    if os.path.lexists(target):
        info = target.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            fail("target")
    elif payload["phase"] == "post":
        fail("missing")
    if payload["phase"] == "post":
        info = target.stat()
        if info.st_size != size:
            fail("size")
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != sha256:
            fail("sha256")
print("REMOTE_" + payload["phase"].upper() + "_OK")
'''


def _remote_payload(bundle: LocalBundle, endpoint: Endpoint, *, phase: str) -> bytes:
    if phase not in {"pre", "post"}:
        raise ValueError("remote phase는 pre/post여야 합니다")
    files = [entry.payload() for entry in bundle.files]
    files.append(bundle.manifest.payload())
    payload = {
        "schema_version": 1,
        "phase": phase,
        "remote_repo": endpoint.remote_repo,
        "expected_commit": bundle.commit,
        "files": files,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _ssh_base(endpoint: Endpoint) -> list[str]:
    return [
        "ssh",
        "-T",
        "-i",
        os.fspath(endpoint.identity),
        "-p",
        str(endpoint.port),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
    ]


def _run_remote_check(bundle: LocalBundle, endpoint: Endpoint, *, phase: str) -> None:
    command = shlex.join(["python3", "-I", "-B", "-c", _REMOTE_CHECK_PROGRAM])
    try:
        result = subprocess.run(
            [*_ssh_base(endpoint), "--", endpoint.target, command],
            input=_remote_payload(bundle, endpoint, phase=phase),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=3600 if phase == "post" else 600,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PushTransferError(f"원격 {phase}flight를 실행할 수 없습니다") from exc
    expected = f"REMOTE_{phase.upper()}_OK\n".encode("ascii")
    if result.returncode != 0 or result.stdout != expected:
        raise PushTransferError(f"원격 {phase}flight exact checkout/bundle 검증 실패")


def _rsync_command(endpoint: Endpoint) -> list[str]:
    rsh = shlex.join(_ssh_base(endpoint))
    return [
        "rsync",
        "-aR",
        "--partial",
        "--from0",
        "--files-from=-",
        "--protect-args",
        f"--rsh={rsh}",
        "./",
        f"{endpoint.target}:{endpoint.remote_repo}/",
    ]


def _file_list_bytes(bundle: LocalBundle) -> bytes:
    paths = bundle.transfer_paths
    if len(paths) != len(set(paths)) or bundle.manifest.path not in paths:
        raise PushTransferError("rsync file list에 manifest가 없거나 경로가 중복됩니다")
    return b"".join(path.encode("utf-8") + b"\0" for path in paths)


def _run_rsync(bundle: LocalBundle, endpoint: Endpoint, *, repo_root: Path) -> None:
    try:
        result = subprocess.run(
            _rsync_command(endpoint),
            cwd=repo_root,
            input=_file_list_bytes(bundle),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        raise PushTransferError("rsync를 실행할 수 없습니다") from exc
    if result.returncode != 0:
        raise PushTransferError(
            "rsync 전송이 중단됐습니다. 원격 partial 파일을 삭제하지 말고 같은 명령으로 재개하세요"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--remote-repo", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--manifest",
        default=_CANONICAL_MANIFEST,
        help=f"canonical 상대경로(고정값: {_CANONICAL_MANIFEST})",
    )
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        commit = str(args.expected_commit).lower()
        manifest_sha256 = str(args.expected_manifest_sha256).lower()
        if _COMMIT_RE.fullmatch(commit) is None:
            raise PushTransferError("--expected-commit은 lowercase 40자리 SHA여야 합니다")
        if _SHA256_RE.fullmatch(manifest_sha256) is None:
            raise PushTransferError("--expected-manifest-sha256는 lowercase 64자리 SHA-256이어야 합니다")
        manifest_relative = _safe_relative_path(args.manifest, label="--manifest")
        if manifest_relative != _CANONICAL_MANIFEST:
            raise PushTransferError("--manifest는 canonical transfer 경로로 고정됩니다")
        endpoint = _endpoint_from_args(args, repo_root=REPO_ROOT)
        for executable in ("ssh", "rsync"):
            if shutil.which(executable) is None:
                raise PushTransferError(f"필수 실행 파일이 없습니다: {executable}")
        bundle = _validate_local_bundle(
            repo_root=REPO_ROOT,
            expected_commit=commit,
            expected_manifest_sha256=manifest_sha256,
            manifest_relative=manifest_relative,
        )
        if args.dry_run:
            print(
                "[DRY-RUN PASS] local exact checkout/full transfer 검증 완료: "
                f"schema={bundle.schema_version}, files={len(bundle.transfer_paths)}, "
                f"bytes={bundle.total_bytes}, network_calls=0"
            )
            return 0
        _run_remote_check(bundle, endpoint, phase="pre")
        _run_rsync(bundle, endpoint, repo_root=REPO_ROOT)
        _run_remote_check(bundle, endpoint, phase="post")
    except PushTransferError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print(
        "[PASS] Elice transfer bundle 전송/원격 SHA 재검증 완료: "
        f"schema={bundle.schema_version}, files={len(bundle.transfer_paths)}, "
        f"bytes={bundle.total_bytes}, deletion=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
