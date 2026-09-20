#!/usr/bin/env python3
"""고역 비교의 정적 준비 전용. 학습/모델 연산 없음. 종료 2=보고서 작성/미해결, 1=입력 오류."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from deep_anc.eval.high_frequency_readiness import prepare_high_frequency


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="null 미확정을 보존한 provisional 비교 계약 JSON")
    parser.add_argument("--out", required=True, help="아직 존재하지 않는 신규 보고서 디렉터리")
    parser.add_argument("--sfanc-manifest", help="기존 SFANC 원천 manifest 경로; 학습/재분할하지 않음")
    parser.add_argument("--capture", help="선택적 capture 메타데이터 파일; 존재/크기/SHA만 확인")
    parser.add_argument("--inventory", help="선택적 공개 자료/Drive inventory 파일; 존재/크기/SHA만 확인")
    args = parser.parse_args(argv)
    try:
        report = prepare_high_frequency(args.config, args.out, sfanc_manifest=args.sfanc_manifest,
                                        capture=args.capture, inventory=args.inventory)
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(f"[준비 입력 오류] {error}", file=sys.stderr)
        return 1
    print(f"[정적 준비 완료] {Path(args.out) / 'report.json'}; 학습 미실행; 미해결 {len(report['blockers'])}개")
    return 0 if report["training_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
