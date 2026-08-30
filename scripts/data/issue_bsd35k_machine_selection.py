#!/usr/bin/env python3
"""공식 BSD35k-CS metadata에서 CC0 machine selection plan을 발행한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from deep_anc.data.bsd35k_machine import (
    build_official_bsd35k_machine_selection,
    write_bsd35k_machine_selection_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        plan = build_official_bsd35k_machine_selection(args.metadata_csv)
        if args.output is not None:
            path, file_sha = write_bsd35k_machine_selection_exclusive(
                args.output, plan
            )
            print(f"[PASS] no-replace selection plan: {path}")
            print(f"[PASS] file SHA-256: {file_sha}")
    except (OSError, ValueError) as error:
        print(f"[BLOCKED] {error}", file=sys.stderr)
        return 2

    summary = plan["selection"]
    print(
        json.dumps(
            {
                "schema_version": plan["schema_version"],
                "selection_plan_sha256": plan["selection_plan_sha256"],
                "selected_clip_count": summary["selected_clip_count"],
                "selected_uploader_count": summary["selected_uploader_count"],
                "split_summary": summary["split_summary"],
                "canonical_source_eligible": plan["authority"][
                    "canonical_source_eligible"
                ],
                "blockers": plan["authority"]["blockers"],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
