#!/usr/bin/env python3
"""기본은 빈 실측 패킷 생성. --input은 무오디오 QA, --prepare는 통과한 train/valid만 분리."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepanc.measurement_ready import create_packet, prepare_measurement_session


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare and args.input is None:
        parser.error("--prepare는 실제 수집한 --input 필요")
    try:
        if args.input is None:
            create_packet(args.out)
            print(f"[빈 수집 계획] {args.out}; 녹음·학습·오디오 실행 없음")
            return 0
        report = prepare_measurement_session(args.input, args.out, prepare=args.prepare)
        print(f"[수집 후 검사] 통과={report['intake_passed']}; 오류 {len(report['errors'])}개; 학습 미실행")
        return 0 if report["intake_passed"] else 2
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(f"[수집 준비 오류] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
