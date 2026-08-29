#!/usr/bin/env python3
"""J511 RT5640 plug 상태를 오디오 장치를 열지 않고 확인한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.realtime.rt5640_jack import (  # noqa: E402
    JACK_STATES,
    assert_rt5640_jack_state,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", choices=sorted(JACK_STATES), required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval-seconds", type=float, default=0.0)
    parser.add_argument("--card", default="APE")
    args = parser.parse_args(argv)
    try:
        report = assert_rt5640_jack_state(
            args.expect,
            card_id=args.card,
            samples=args.samples,
            interval_seconds=args.interval_seconds,
        )
    except (RuntimeError, ValueError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
