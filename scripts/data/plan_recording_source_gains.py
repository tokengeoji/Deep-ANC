#!/usr/bin/env python3
"""Strict-P 기반 source별 recording gain plan을 무출력으로 발행/검증한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.recording_source_gain import (  # noqa: E402
    RecordingSourceGainError,
    build_recording_source_gain_plan,
    issue_recording_source_gain_plan,
    validate_recording_source_gain_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", default=None)
    parser.add_argument("--source-plan-sha256", default=None)
    parser.add_argument(
        "--strict-primary",
        default="assets/measured/primary_path_il_strict_5dc06fdd.npz",
    )
    parser.add_argument("--strict-primary-sha256", default=None)
    parser.add_argument("--gain-linearity-receipt", default=None)
    parser.add_argument("--gain-linearity-receipt-sha256", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--expected-plan-sha256", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    modes = int(args.write) + int(args.verify_existing)
    if modes > 1:
        parser.error("--write와 --verify-existing은 함께 쓸 수 없습니다")
    try:
        if args.verify_existing:
            if args.out is None or args.expected_plan_sha256 is None:
                parser.error(
                    "--verify-existing에는 --out과 --expected-plan-sha256이 필요합니다"
                )
            summary = validate_recording_source_gain_plan(
                repo_root=REPO_ROOT,
                plan_path=args.out,
                expected_sha256=args.expected_plan_sha256,
            )
            payload = summary["payload"]
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "plan_path": summary["plan_path"],
                        "plan_sha256": summary["plan_sha256"],
                        "row_count": payload["row_count"],
                        "canonical_live_eligible": summary[
                            "canonical_live_eligible"
                        ],
                        "blocker_reasons": payload["blocker_reasons"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0

        if (
            args.source_plan is None
            or args.source_plan_sha256 is None
            or args.strict_primary_sha256 is None
        ):
            parser.error(
                "check/write에는 --source-plan, --source-plan-sha256, "
                "--strict-primary-sha256가 필요합니다"
            )

        if args.write:
            if args.out is None:
                parser.error("--write에는 --out이 필요합니다")
            summary = issue_recording_source_gain_plan(
                repo_root=REPO_ROOT,
                output_path=args.out,
                source_plan=args.source_plan,
                expected_source_plan_sha256=args.source_plan_sha256,
                strict_primary=args.strict_primary,
                expected_strict_primary_sha256=args.strict_primary_sha256,
                gain_linearity_receipt=args.gain_linearity_receipt,
                expected_gain_linearity_receipt_sha256=(
                    args.gain_linearity_receipt_sha256
                ),
            )
            payload = summary["payload"]
            print(
                json.dumps(
                    {
                        "status": (
                            "WROTE_READY_PLAN"
                            if summary["canonical_live_eligible"]
                            else "WROTE_BLOCKED_PLAN"
                        ),
                        "plan_path": summary["plan_path"],
                        "plan_sha256": summary["plan_sha256"],
                        "row_count": payload["row_count"],
                        "canonical_live_eligible": summary[
                            "canonical_live_eligible"
                        ],
                        "blocker_reasons": payload["blocker_reasons"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0

        payload = build_recording_source_gain_plan(
            repo_root=REPO_ROOT,
            source_plan=args.source_plan,
            expected_source_plan_sha256=args.source_plan_sha256,
            strict_primary=args.strict_primary,
            expected_strict_primary_sha256=args.strict_primary_sha256,
            gain_linearity_receipt=args.gain_linearity_receipt,
            expected_gain_linearity_receipt_sha256=(
                args.gain_linearity_receipt_sha256
            ),
        )
        print(
            json.dumps(
                {
                    "status": (
                        "CHECK_ONLY_READY"
                        if payload["canonical_live_eligible"]
                        else "CHECK_ONLY_BLOCKED"
                    ),
                    "source_plan_sha256": payload["source_plan"]["sha256"],
                    "strict_primary_sha256": payload["strict_primary"]["sha256"],
                    "row_count": payload["row_count"],
                    "canonical_live_eligible": payload["canonical_live_eligible"],
                    "blocker_reasons": payload["blocker_reasons"],
                    "evidence_sha256": payload["evidence_sha256"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, RecordingSourceGainError, ValueError) as exc:
        print(f"[BLOCKED] source gain plan 실패: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
