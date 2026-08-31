#!/usr/bin/env python3
"""노트북 Stage-2 작업 receipt를 Drive에 발행하거나 read-only로 읽는다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from deep_anc.data.notebook_exchange import (
    DEFAULT_REMOTE_ROOT,
    PHASES,
    STATES,
    NotebookExchangeError,
    assert_exact_checkout,
    build_archive_cache_advisory_receipt,
    build_full_decoder_advisory_receipt,
    publish_status,
    read_remote_statuses,
    sha256_file,
)


MIN_NOTEBOOK_WORK_BYTES = 32 * 1024**3


def _read_json_regular(path: str, *, label: str) -> tuple[Path, dict]:
    candidate = Path(path)
    info = candidate.lstat()
    if candidate.is_symlink() or not candidate.is_file() or info.st_size <= 0:
        raise NotebookExchangeError(f"{label}는 nonempty regular non-symlink JSON이어야 합니다")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise NotebookExchangeError(f"{label}가 UTF-8 JSON이 아닙니다") from exc
    if not isinstance(payload, dict):
        raise NotebookExchangeError(f"{label} root가 object가 아닙니다")
    return candidate.resolve(strict=True), payload


def _write_json_exclusive(path: str, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    audit = commands.add_parser(
        "audit-checkout", help="preflight typed receipt를 O_EXCL로 발행"
    )
    audit.add_argument("--repository-root", default=".")
    audit.add_argument("--expected-commit", required=True)
    audit.add_argument("--work-root", required=True)
    audit.add_argument("--out", required=True)
    audit.add_argument("--rclone", default="rclone")

    archive = commands.add_parser(
        "audit-archive-cache",
        help="exact production archive manifest를 10-file readback advisory로 투영",
    )
    archive.add_argument("--repository-root", default=".")
    archive.add_argument("--expected-commit", required=True)
    archive.add_argument("--manifest", required=True)
    archive.add_argument("--out", required=True)

    decoder = commands.add_parser(
        "audit-full-decoder",
        help="full 37,761 decoder audit을 작은 advisory receipt로 투영",
    )
    decoder.add_argument("--repository-root", default=".")
    decoder.add_argument("--expected-commit", required=True)
    decoder.add_argument("--decoder-audit", required=True)
    decoder.add_argument("--out", required=True)

    publish = commands.add_parser("publish", help="receipt 먼저, status를 마지막에 immutable 발행")
    publish.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    publish.add_argument("--repository-root", default=".")
    publish.add_argument("--expected-commit", required=True)
    publish.add_argument("--phase", choices=PHASES, required=True)
    publish.add_argument("--state", choices=STATES, required=True)
    publish.add_argument("--message", required=True)
    publish.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="KIND=PATH",
        help="phase별 typed artifact. PASS는 필수 kind를 모두 제공해야 합니다",
    )
    publish.add_argument("--rclone", default="rclone")
    publish.add_argument("--timeout-seconds", type=int, default=3600)

    read = commands.add_parser("read", help="Drive status를 쓰지 않고 latest phase를 검증")
    read.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    read.add_argument("--expected-commit", required=True)
    read.add_argument("--rclone", default="rclone")
    read.add_argument("--timeout-seconds", type=int, default=300)
    read.add_argument("--maximum-status-files", type=int, default=256)
    read.add_argument("--require-complete", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "audit-checkout":
            repository = Path(args.repository_root).resolve(strict=True)
            work_root = Path(args.work_root).resolve(strict=True)
            try:
                work_root.relative_to(repository)
            except ValueError:
                pass
            else:
                raise NotebookExchangeError("work-root는 repository 밖이어야 합니다")
            assert_exact_checkout(
                repository_root=repository, expected_commit=args.expected_commit
            )
            executable = shutil.which(args.rclone)
            if executable is None:
                raise NotebookExchangeError("rclone executable을 찾을 수 없습니다")
            version = subprocess.run(
                [executable, "version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.splitlines()
            if not version:
                raise NotebookExchangeError("rclone version을 확인할 수 없습니다")
            free_bytes = int(shutil.disk_usage(work_root).free)
            if free_bytes < MIN_NOTEBOOK_WORK_BYTES:
                raise NotebookExchangeError("work-root free space가 32 GiB 미만입니다")
            payload = {
                "schema": "deep_anc_notebook_checkout_audit_v1",
                "status": "PASS",
                "source_commit": args.expected_commit,
                "repository_clean_exact": True,
                "work_root_outside_repository": True,
                "work_root_free_bytes": free_bytes,
                "rclone_executable_name": Path(executable).name,
                "rclone_version": version[0].strip(),
                "secrets_recorded": False,
            }
            _write_json_exclusive(args.out, payload)
            code = 0
        elif args.command == "audit-archive-cache":
            repository = Path(args.repository_root).resolve(strict=True)
            assert_exact_checkout(
                repository_root=repository, expected_commit=args.expected_commit
            )
            manifest_path, manifest = _read_json_regular(
                args.manifest, label="archive production manifest"
            )
            publisher = repository / "scripts/elice/public_archive_cache.py"
            payload = build_archive_cache_advisory_receipt(
                manifest=manifest,
                manifest_file_sha256=sha256_file(manifest_path),
                expected_commit=args.expected_commit,
                publisher_script_sha256=sha256_file(publisher),
            )
            _write_json_exclusive(args.out, payload)
            code = 0
        elif args.command == "audit-full-decoder":
            repository = Path(args.repository_root).resolve(strict=True)
            assert_exact_checkout(
                repository_root=repository, expected_commit=args.expected_commit
            )
            audit_path, audit_payload = _read_json_regular(
                args.decoder_audit, label="decoder audit"
            )
            payload = build_full_decoder_advisory_receipt(
                report=audit_payload,
                report_file_sha256=sha256_file(audit_path),
                expected_commit=args.expected_commit,
            )
            _write_json_exclusive(args.out, payload)
            code = 0
        elif args.command == "publish":
            typed_artifacts: list[tuple[str, str]] = []
            for value in args.artifact:
                kind, separator, path = value.partition("=")
                if not separator or not kind or not path:
                    raise NotebookExchangeError("--artifact는 KIND=PATH 형식이어야 합니다")
                typed_artifacts.append((kind, path))
            payload = publish_status(
                remote_root=args.remote_root,
                repository_root=args.repository_root,
                expected_commit=args.expected_commit,
                phase=args.phase,
                state=args.state,
                message=args.message,
                receipt_paths=typed_artifacts,
                rclone_executable=args.rclone,
                timeout_seconds=args.timeout_seconds,
            )
            code = 0
        else:
            payload = read_remote_statuses(
                remote_root=args.remote_root,
                expected_commit=args.expected_commit,
                rclone_executable=args.rclone,
                timeout_seconds=args.timeout_seconds,
                maximum_status_files=args.maximum_status_files,
            )
            code = int(args.require_complete and not payload["advisory_complete"])
    except (NotebookExchangeError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"[BLOCKED] notebook exchange: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
