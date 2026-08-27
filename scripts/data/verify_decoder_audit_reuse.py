#!/usr/bin/env python3
"""완료된 decoder audit을 canonical bootstrap에서 안전하게 재사용할지 검증한다.

이 도구는 새 PCM decode를 수행하지 않지만, 기존 report 파일 byte SHA와 report 내부
semantic SHA, canonical full-scan recipe, 현재 decoder runtime fingerprint, 그리고
모든 accepted/rejected raw 후보의 경로·SHA-256·크기를 다시 확인한다. 검증 후 raw가
바뀌는 TOCTOU는 ``prepare_noise_pool.py`` transaction의 같은 전수 검증이 막는다.

재사용 실패는 새 audit으로 자동 fallback하지 않는다. 호출자가 원인을 보존한 뒤
명시적으로 새 full audit을 선택해야 한다.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.decoder_audit import (  # noqa: E402
    canonical_json_bytes,
    validate_audit_report_self_digest,
)
from deep_anc.data.holdout_contract import read_regular_file_snapshot  # noqa: E402
from deep_anc.data.manifest_contract import (  # noqa: E402
    read_decoder_audit,
    validate_decoder_audit_raw_inventory,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _path_under_root(value: str, *, root: Path, field: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    absolute = Path(os.path.abspath(candidate))
    try:
        absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field}는 --root 아래여야 합니다: {value}") from exc
    return absolute


def _sha256_argument(value: str, *, field: str) -> str:
    normalised = str(value).lower()
    if _SHA256_RE.fullmatch(normalised) is None:
        raise ValueError(f"{field}에는 64자리 SHA-256이 필요합니다")
    return normalised


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root (기본: .)")
    parser.add_argument(
        "--audit",
        required=True,
        help="기존 decoder_audit.json 경로 (--root 아래)",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        required=True,
        help="audit inventory와 대조할 raw root (--root 아래, 반복 가능)",
    )
    parser.add_argument(
        "--expected-audit-sha256",
        required=True,
        help="report 내부 audit_sha256의 외부 trust anchor",
    )
    parser.add_argument(
        "--expected-file-sha256",
        required=True,
        help="decoder audit JSON file bytes의 외부 trust anchor",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root_candidate = Path(args.root).expanduser()
    if not root_candidate.is_absolute():
        root_candidate = REPO_ROOT / root_candidate
    root = Path(os.path.abspath(root_candidate))
    try:
        if not root.is_dir():
            raise ValueError(f"--root directory가 없습니다: {root}")
        audit_path = _path_under_root(args.audit, root=root, field="--audit")
        raw_roots = [
            _path_under_root(value, root=root, field="--scan-root")
            for value in args.scan_root
        ]
        if len({str(path) for path in raw_roots}) != len(raw_roots):
            raise ValueError("--scan-root에 중복 경로가 있습니다")
        expected_audit_sha = _sha256_argument(
            args.expected_audit_sha256, field="--expected-audit-sha256"
        )
        expected_file_sha = _sha256_argument(
            args.expected_file_sha256, field="--expected-file-sha256"
        )

        file_snapshot = read_regular_file_snapshot(
            audit_path,
            root=root,
            label="재사용 decoder audit report",
        )
        if file_snapshot.sha256 != expected_file_sha:
            raise ValueError(
                "decoder audit report file SHA가 외부 trust anchor와 다릅니다: "
                f"expected={expected_file_sha}, actual={file_snapshot.sha256}"
            )

        audit = read_decoder_audit(
            audit_path,
            repo_root=root,
            label="재사용 decoder audit report",
        )
        report_snapshot = audit["_snapshot"]
        if report_snapshot.sha256 != expected_file_sha:
            raise ValueError("decoder audit report가 검증 중 바뀌었습니다")
        payload = {
            key: value
            for key, value in audit.items()
            if not key.startswith("_")
        }
        actual_audit_sha = validate_audit_report_self_digest(payload)
        if actual_audit_sha != expected_audit_sha:
            raise ValueError(
                "decoder audit semantic SHA가 외부 trust anchor와 다릅니다: "
                f"expected={expected_audit_sha}, actual={actual_audit_sha}"
            )
        validate_decoder_audit_raw_inventory(
            audit,
            repo_root=root,
            raw_roots=raw_roots,
            label="재사용 decoder audit report",
        )
    except (OSError, ValueError) as exc:
        print(f"[실패] decoder audit 재사용 검증 실패: {exc}", file=sys.stderr)
        return 1

    summary = audit["summary"]
    print(
        canonical_json_bytes(
            {
                "status": "PASS",
                "audit": str(audit_path.relative_to(root)),
                "audit_sha256": actual_audit_sha,
                "file_sha256": report_snapshot.sha256,
                "inventory_sha256": audit["inventory_sha256"],
                "accepted_inventory_sha256": audit["accepted_inventory_sha256"],
                "decoder_fingerprint_sha256": audit["decoder_fingerprint_sha256"],
                "candidate_count": summary["candidate_count"],
                "accepted_count": summary["accepted_count"],
                "rejected_count": summary["rejected_count"],
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
