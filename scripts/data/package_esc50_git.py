#!/usr/bin/env python3
"""공식 ESC-50의 이미 받은 고정 Git 객체를 검증·포장한다(네트워크/checkout/삭제 없음)."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from deep_anc.data.archive_staging import staging_destination

SOURCE = "https://github.com/karolpiczak/ESC-50.git"
COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"


def _independent_repo(source_repo: Path) -> Path:
    repo = source_repo.absolute()
    if ".." in repo.parts or any(p.is_symlink() for p in (repo, *repo.parents)):
        raise ValueError("Git 원본 경로의 상위 이동/링크 거부")
    git_dir = repo / ".git"
    if git_dir.is_symlink() or not git_dir.is_dir():
        raise ValueError("symlink/linked gitdir가 아닌 독립 Git 저장소가 필요합니다")
    if not (git_dir / "objects").is_dir() or not (git_dir / "config").is_file():
        raise ValueError("독립 Git object store/config가 필요합니다")
    forbidden = (git_dir / "commondir", git_dir / "gitdir",
                 git_dir / "objects/info/alternates", git_dir / "objects/info/http-alternates",
                 git_dir / "info/attributes")
    if any(path.exists() or path.is_symlink() for path in forbidden):
        raise ValueError("외부 Git 저장소/대체 objects/로컬 archive attributes는 허용하지 않습니다")
    # 객체 하나/pack 디렉터리의 링크도 막고, partial clone의 promisor pack도 거부한다.
    for root, directories, files in os.walk(git_dir, followlinks=False):
        for name in (*directories, *files):
            item = Path(root) / name
            if item.is_symlink() or item.name.endswith(".promisor"):
                raise ValueError("Git 내부 symlink/promisor 객체는 허용하지 않습니다")
    # Git 시작 과정도 로컬 config를 읽을 수 있으므로 첫 Git 호출 전에 검사한다.
    # 기본 full clone만 대상으로 하며, 모호한 연속 행/확장 설정은 보수적으로 거부한다.
    with (git_dir / "config").open("rb") as handle:
        raw_config = handle.read(1024 * 1024 + 1)
    if len(raw_config) > 1024 * 1024:
        raise ValueError("Git config가 검사 크기 상한을 넘었습니다")
    local_config = raw_config.decode("utf-8")
    if ("\x00" in local_config or any(line.rstrip().endswith("\\") for line in local_config.splitlines())
            or re.search(r"(?im)^\s*\[\s*include(?:if)?(?:\s|\.|\])", local_config)
            or re.search(r"(?im)\b(?:partialclone|promisor|worktreeconfig)\b", local_config)):
        raise ValueError("Git include/partialClone/promisor/연속 config는 허용하지 않습니다")
    return repo


def _git_environment() -> dict[str, str]:
    # 토큰/credential/GIT_DIR/alternate-object/SSH 등 호스트 환경을 상속하지 않는다.
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": os.environ.get("LANG", "C.UTF-8"),
            "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_ALLOW_PROTOCOL": "", "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0",
            "GIT_ATTR_NOSYSTEM": "1"}


def package(source_repo: Path, out: Path) -> dict:
    if os.environ.get("DEEP_ANC_CONTAINER") != "cpu" or not Path("/.dockerenv").is_file():
        raise ValueError("CPU Docker에서만 실행하세요")
    repo = _independent_repo(source_repo)
    destination = staging_destination(out)
    if destination.is_relative_to(repo):
        raise ValueError("기존 Git 원본 내부에 산출물을 만들 수 없습니다")
    environment = _git_environment()
    command = ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.attributesFile=/dev/null",
               "-c", "core.fsmonitor=false", "-C", str(repo)]
    def git(*args):
        return subprocess.run([*command, *args], env=environment,
                              check=True, capture_output=True, text=True).stdout.strip()
    # --no-includes로 외부 config를 읽지 않은 채 로컬 선언부터 검사한다.
    local_config = git("config", "--local", "--no-includes", "--null", "--list")
    for item in local_config.split("\0"):
        key = item.partition("\n")[0].lower()
        if (key.startswith(("include.", "includeif.")) or key.endswith(".promisor")
                or key in {"extensions.partialclone", "extensions.worktreeconfig"}):
            raise ValueError("include/partialClone/promisor/worktree config는 허용하지 않습니다")
    if git("remote", "get-url", "origin") != SOURCE or git("rev-parse", "HEAD") != COMMIT:
        raise ValueError("공식 URL/고정 commit 불일치")
    git("fsck", "--full", "--strict")
    tree = git("rev-parse", COMMIT + "^{tree}")
    if not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise ValueError("고정 commit의 tree ID 검증 실패")
    # 외부 tar.tar.gz.command를 쓰지 않는다. builtin tar stdout만 Python gzip으로 압축한다.
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    name = "ESC-50-" + COMMIT + ".tar.gz"
    archive = destination / name
    archive_command = [*command, "archive", "--format=tar", "--prefix=ESC-50/", COMMIT]
    tar_bytes = 0
    with archive.open("xb") as output:
        process = subprocess.Popen(archive_command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            with process.stdout as source, gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed:
                while block := source.read(1024 * 1024):
                    compressed.write(block)
                    tar_bytes += len(block)
            returncode = process.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, archive_command)
            if tar_bytes == 0:
                raise ValueError("비어 있는 Git archive stdout은 완료로 처리하지 않습니다")
            output.flush()
            os.fsync(output.fileno())
        except BaseException:
            # 정리 실패가 최초 실패를 가리지 않으며 부분 archive는 지우지 않는다.
            try:
                process.kill()
                process.wait()
            except OSError:
                pass
            raise
    digests = {k: hashlib.new(k) for k in ("sha256", "md5")}
    with archive.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            for digest in digests.values():
                digest.update(block)
    sha256, md5 = (digests[k].hexdigest() for k in ("sha256", "md5"))
    receipt = {
        "schema_version": 1, "name": name, "archive_path": str(archive), "source_id": "esc50_git",
        "bytes": archive.stat().st_size, "size": archive.stat().st_size, "sha256": sha256, "md5": md5,
        "source_url": SOURCE, "git_source_url": SOURCE, "git_commit": COMMIT,
        "git_tree": tree, "git_fsck_verified": True,
        "source_verification": "official_git_commit_and_objects", "source_content_verified": True,
        "source_checksum_verified": False, "source_checksum_algorithm": "sha256", "source_checksum": sha256,
        "checksum_origin": "locally_generated_git_archive_not_publisher_archive_digest",
        "official_reference": SOURCE.removesuffix(".git"),
        "license": "CC BY-NC 3.0; ESC-10 CC BY exception; per-file attributions in LICENSE",
        "research_use": "noncommercial_academic", "temporary_local_staging": True,
        "storage_policy": "temporary_local_until_verified_drive_upload", "local_raw_written": True,
        "drive_upload_verified": False, "local_deletion_performed": False,
        "delete_after_verified_upload_required": True, "pcm_qa_verified": False,
        "extraction_performed": False,
    }
    with (destination / "receipt.json").open("x") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    try:
        result = package(args.source_repo, args.out)
    except (ValueError, OSError, subprocess.CalledProcessError):
        print("[실패] 공식 Git 객체/경로/포장 검사 실패; 부분 파일을 보존합니다", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
