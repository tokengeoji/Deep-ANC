#!/usr/bin/env python3
"""DKITCHEN 환경음의 immutable pre-exclusion selection bundle을 발행한다.

오디오 장치를 열지 않는다. 반드시 live DEMAND manifest에서 recorded-generation
exclusion을 적용하기 전에 ``--write``하고, 이후에는 ``--verify-existing``만 쓴다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _canonical_isolated_import_path() -> tuple[str, ...]:
    """``python -I -S -B``에서도 exact project venv만 다시 연다."""

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


if sys.flags.isolated and sys.flags.no_site:
    sys.path[:] = list(_canonical_isolated_import_path())
else:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.holdout_contract import (  # noqa: E402
    read_regular_file_snapshot,
    reject_symlink_components,
)
from deep_anc.data.recorded_demand_selection import (  # noqa: E402
    DEMAND_BOOTSTRAP_RECEIPT,
    DEMAND_SELECTION_BUNDLE_ROOT,
    DEMAND_SELECTION_RECEIPT,
    DemandSelectionError,
    build_demand_selection_payload,
    validate_demand_selection_receipt,
)
from deep_anc.data.source_trust import canonical_selector_sys_path  # noqa: E402


def _publish_no_replace(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(path.parent, root=REPO_ROOT)
    if path.exists() or path.is_symlink():
        snapshot = read_regular_file_snapshot(
            path,
            root=REPO_ROOT,
            label="existing DEMAND selection bundle file",
            capture_bytes=True,
        )
        if snapshot.data != raw:
            raise DemandSelectionError(
                f"기존 DEMAND selection bundle bytes가 달라 overwrite하지 않습니다: {path}"
            )
        return
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--bootstrap-receipt", default=DEMAND_BOOTSTRAP_RECEIPT
    )
    parser.add_argument("--bootstrap-receipt-sha256", required=True)
    parser.add_argument("--expected-manifest-generation-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--expected-receipt-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if (
            sys.flags.isolated
            and sys.flags.no_site
            and tuple(sys.path) != canonical_selector_sys_path(REPO_ROOT)
        ):
            raise DemandSelectionError(
                "DEMAND selector canonical isolated sys.path가 변경됐습니다"
            )
        if args.bootstrap_receipt != DEMAND_BOOTSTRAP_RECEIPT:
            raise DemandSelectionError(
                "--bootstrap-receipt는 canonical Elice receipt 경로여야 합니다"
            )
        if args.verify_existing and args.expected_receipt_sha256 is None:
            raise DemandSelectionError(
                "--verify-existing은 --expected-receipt-sha256 외부 anchor가 필수입니다"
            )
        if not args.verify_existing and args.expected_receipt_sha256 is not None:
            raise DemandSelectionError(
                "--expected-receipt-sha256는 --verify-existing에서만 사용합니다"
            )
        if args.verify_existing:
            summary = validate_demand_selection_receipt(
                repo_root=REPO_ROOT,
                expected_receipt_sha256=args.expected_receipt_sha256,
                require_source_files=True,
            )
            if summary.get("source_commit") != args.expected_commit.lower():
                raise DemandSelectionError(
                    "existing DEMAND receipt source_commit이 외부 anchor와 다릅니다"
                )
            if (
                summary.get("bootstrap_receipt_sha256")
                != args.bootstrap_receipt_sha256.lower()
            ):
                raise DemandSelectionError(
                    "existing DEMAND bootstrap receipt SHA가 외부 anchor와 다릅니다"
                )
            if (
                summary.get("manifest_generation_sha256")
                != args.expected_manifest_generation_sha256.lower()
            ):
                raise DemandSelectionError(
                    "existing DEMAND manifest_generation SHA가 외부 anchor와 다릅니다"
                )
            receipt_sha = summary["receipt_sha256"]
            evidence_sha = summary["evidence_sha256"]
        else:
            payload, files = build_demand_selection_payload(
                repo_root=REPO_ROOT,
                bootstrap_receipt=args.bootstrap_receipt,
                bootstrap_receipt_sha256=args.bootstrap_receipt_sha256,
                expected_commit=args.expected_commit,
                expected_manifest_generation_sha256=(
                    args.expected_manifest_generation_sha256
                ),
            )
            receipt_raw = (
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            ).encode("utf-8")
            receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
            evidence_sha = payload["evidence_sha256"]
            if args.write:
                bundle = REPO_ROOT / DEMAND_SELECTION_BUNDLE_ROOT
                for relative, raw in sorted(files.items()):
                    _publish_no_replace(bundle / relative, raw)
                # receipt를 마지막에 공개해야 validator가 partial bundle을 승인하지 않는다.
                _publish_no_replace(REPO_ROOT / DEMAND_SELECTION_RECEIPT, receipt_raw)
                validate_demand_selection_receipt(
                    repo_root=REPO_ROOT,
                    expected_receipt_sha256=receipt_sha,
                    require_source_files=True,
                )
    except (OSError, RuntimeError, ValueError, DemandSelectionError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    mode = "verified" if args.verify_existing else ("written" if args.write else "checked")
    print(
        f"DEMAND selection bundle {mode}: {DEMAND_SELECTION_RECEIPT}\n"
        f"receipt sha256: {receipt_sha}\n"
        f"evidence sha256: {evidence_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
