#!/usr/bin/env python3
"""검증된 측정 packet을 SFANC 학습 계획으로 연결. fit/CNN/optimizer 실행 없음."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from deep_anc.train.sfanc_paired import prepare_paired


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, help="measurement_ready --prepare 결과 디렉터리")
    parser.add_argument("--recipe", required=True, help="미확정 null을 실제 결정값으로 채운 명시적 recipe JSON")
    parser.add_argument("--out", required=True, help="존재하지 않는 새 준비 디렉터리")
    args = parser.parse_args(argv)
    try:
        plan = prepare_paired(args.packet, args.recipe, args.out)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"[준비 실패] {error}", file=sys.stderr)
        return 1
    print(f"[준비 완료] {args.out}: 창 수 {plan['window_counts']}; 학습/fit 미실행, test WAV 미열기")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
