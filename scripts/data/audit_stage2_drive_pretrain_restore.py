#!/usr/bin/env python3
"""Drive Stage-2 partial restore를 감사하거나 local restore를 전수 검증한다."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from deep_anc.data.stage2_drive_pretrain_restore import (
    DEFAULT_ARCHIVE_CACHE_REMOTE_ROOT,
    DEFAULT_SNAPSHOT_REMOTE_ROOT,
    EXPECTED_SNAPSHOT_MANIFEST_SHA256,
    Stage2DriveAuditError,
    audit_stage2_drive_remote,
    build_stage2_drive_restore_anchor,
    verify_local_stage2_partial_restore,
    write_json_exclusive,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    remote = subparsers.add_parser(
        "audit-remote",
        help="rclone cat/lsjson only; remote bytes를 쓰거나 내려받지 않음",
    )
    remote.add_argument("--snapshot-remote-root", default=DEFAULT_SNAPSHOT_REMOTE_ROOT)
    remote.add_argument(
        "--archive-cache-remote-root", default=DEFAULT_ARCHIVE_CACHE_REMOTE_ROOT
    )
    remote.add_argument(
        "--expected-snapshot-manifest-sha256",
        default=EXPECTED_SNAPSHOT_MANIFEST_SHA256,
    )
    remote.add_argument("--rclone", default="rclone")
    remote.add_argument("--timeout-seconds", type=int, default=300)
    remote.add_argument("--output", required=True)
    remote.add_argument("--anchor-output", required=True)

    local = subparsers.add_parser(
        "verify-local",
        help="Elice/local SSD partial restore의 path/size/content SHA를 전수 검증",
    )
    local.add_argument("--anchor", required=True)
    local.add_argument("--restore-root", required=True)
    local.add_argument("--snapshot-manifest", required=True)
    local.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit-remote":
            payload = audit_stage2_drive_remote(
                snapshot_remote_root=args.snapshot_remote_root,
                archive_cache_remote_root=args.archive_cache_remote_root,
                expected_manifest_sha256=args.expected_snapshot_manifest_sha256,
                rclone_executable=args.rclone,
                timeout_seconds=args.timeout_seconds,
            )
            anchor = build_stage2_drive_restore_anchor(payload)
            write_json_exclusive(payload, args.output)
            try:
                write_json_exclusive(anchor, args.anchor_output)
            except BaseException:
                # audit output은 immutable evidence이므로 anchor 실패 시 삭제하지 않는다.
                raise
            summary = {
                "status": payload["status"],
                "public_scratch_pretrain_status": payload[
                    "public_synthetic_scratch_pretrain_readiness"
                ]["status"],
                "snapshot_status": payload["snapshot"]["status"],
                "partial_restore_status": payload["partial_restore"]["status"],
                "partial_restore_files": payload["partial_restore"]["file_count"],
                "partial_restore_bytes": payload["partial_restore"]["byte_count"],
                "fixed_archive_cache_status": payload[
                    "official_fixed_archive_cache"
                ]["status"],
                "evidence_sha256": payload["evidence_sha256"],
                "anchor_evidence_sha256": anchor["evidence_sha256"],
                "output": args.output,
                "anchor_output": args.anchor_output,
            }
        else:
            payload = verify_local_stage2_partial_restore(
                anchor_path=args.anchor,
                restore_root=args.restore_root,
                snapshot_manifest_path=args.snapshot_manifest,
            )
            write_json_exclusive(payload, args.output)
            summary = {
                "status": payload["status"],
                "file_count": payload["file_count"],
                "byte_count": payload["byte_count"],
                "stage2_public_pretrain_ready": payload[
                    "stage2_public_pretrain_ready"
                ],
                "evidence_sha256": payload["evidence_sha256"],
                "output": args.output,
            }
    except (OSError, subprocess.SubprocessError, Stage2DriveAuditError, ValueError) as exc:
        print(f"[BLOCKED] Stage-2 Drive restore audit 실패: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
