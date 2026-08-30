#!/usr/bin/env python3
"""v5 immutable raw를 offline 분석한다. 오디오 장치는 접근하지 않는다."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from deep_anc.dsp.fullband_causal_v5_offline import (  # noqa:E402
    analyze_v5_raw_file,
    publish_fixture_analysis_v5,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-envelope", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--synthetic-fixture", action="store_true")
    args = parser.parse_args(argv)
    try:
        envelope = json.loads(args.plan_envelope.read_text(encoding="utf-8"))
        plan = envelope.get("signal_plan", envelope)
        expected_raw = Path(os.path.abspath(ROOT / plan["publisher_contract"]["raw_session_relative_path"]))
        actual_raw = Path(os.path.abspath(args.raw))
        if actual_raw != expected_raw:
            raise ValueError("--raw가 signal plan sealed raw path와 다릅니다")
        analysis, operator = analyze_v5_raw_file(
            plan=plan,
            raw_path=actual_raw,
            repository_root=ROOT,
            synthetic_fixture=args.synthetic_fixture,
        )
        published = publish_fixture_analysis_v5(
            target_directory=args.output_directory, analysis=analysis, operator=operator
        )
    except (FileExistsError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"[실패] v5 offline analysis 거부: {error}", file=sys.stderr)
        return 2
    print(f"[{analysis['status']}] {published}")
    print("[잠금] authority=None; canonical_training_eligible=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
