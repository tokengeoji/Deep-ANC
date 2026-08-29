#!/usr/bin/env python3
"""v2 aperiodic causal P/S signal-only plan. Live authority는 의도적으로 잠겨 있다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deep_anc.dsp.fullband_causal_aperiodic import build_aperiodic_plan

FULLBAND_CAUSAL_APERIODIC_LIVE_AUTHORITY = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-live", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.execute_live:
        print("[중단] v2 aperiodic live authority SHA가 없어 출력이 잠겨 있습니다", file=sys.stderr)
        return 2
    plan, _ = build_aperiodic_plan()
    if args.output:
        target = args.output.resolve()
        if target.exists():
            print(f"[실패] 기존 plan을 덮어쓰지 않습니다: {target}", file=sys.stderr)
            return 2
        target.parent.mkdir(parents=True, exist_ok=True)
        # authority가 없는 signal-only diagnostic 파일이다. canonical publisher는 별도다.
        with target.open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(
        f"[PASS] v2 aperiodic signal-only {plan['output']['duration_seconds']:.3f}s | "
        f"active {plan['output']['active_duration_seconds']:.3f}s | peak<=0.003"
    )
    print("[잠금] 오디오 출력 0회; live authority=None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
