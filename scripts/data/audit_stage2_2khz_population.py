#!/usr/bin/env python3
"""Stage-2 2 kHz recorded/public population을 actual bytes에서 감사한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deep_anc.data.stage2_2khz_population_audit import (
    Stage2PopulationAuditError,
    audit_stage2_2khz_population,
    write_audit_exclusive,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument(
        "--recorded-manifest",
        default="data/manifests/recorded_regrouped.jsonl",
    )
    parser.add_argument(
        "--holdout", default="data/manifests/recorded_holdout.json"
    )
    parser.add_argument("--start-seconds", type=float, default=5.0)
    parser.add_argument("--stop-seconds", type=float, default=65.0)
    parser.add_argument("--nperseg", type=int, default=8192)
    parser.add_argument("--noverlap", type=int, default=4096)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = audit_stage2_2khz_population(
            repository_root=args.repository_root,
            recorded_manifest_path=args.recorded_manifest,
            holdout_path=args.holdout,
            start_seconds=args.start_seconds,
            stop_seconds=args.stop_seconds,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
        )
        write_audit_exclusive(payload, args.output)
    except (OSError, ValueError, Stage2PopulationAuditError) as exc:
        print(f"[오류] Stage-2 population audit 실패: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": payload["status"],
                "public_scratch_pretrain_status": payload[
                    "public_scratch_pretrain_status"
                ],
                "public_scratch_pretrain_blockers": payload[
                    "public_scratch_pretrain_blockers"
                ],
                "recorded_finetune_status": payload["recorded_finetune_status"],
                "recorded_finetune_blocker_count": len(
                    payload["recorded_finetune_blockers"]
                ),
                "legacy_1600_2828_joint_groups": payload[
                    "legacy_1600_2828_joint_recalculation"
                ]["joint_valid_independent_group_count"],
                "minimum_new_recording_slots_lower_bound": payload[
                    "minimum_addition_plan"
                ]["minimum_new_recording_slots_lower_bound"],
                "evidence_sha256": payload["evidence_sha256"],
                "output": str(Path(args.output)),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
