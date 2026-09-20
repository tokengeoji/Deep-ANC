#!/usr/bin/env python3
"""오디오 없는 OMAP SFANC 연속 처리·추가 지연 진단."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from deep_anc.eval.sfanc_stress import StressConfig, run_stress


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="기존 OMAP SFANC 사전학습 산출물")
    parser.add_argument("--out", required=True, help="존재하지 않는 신규 결과 폴더")
    parser.add_argument("--config", required=True, help="진단 조건 JSON, 실측 설정 아님")
    parser.add_argument("--librispeech-root")
    args = parser.parse_args(argv)
    try:
        config = StressConfig(**json.loads(Path(args.config).read_text()))
        report = run_stress(args.run, args.out, config, librispeech_root=args.librispeech_root)
    except (ValueError, OSError, TypeError, RuntimeError) as exc:
        print(f"[진단 실패] {exc}; 부분 산출물은 보존합니다.", file=sys.stderr)
        return 1
    print(f"[진단 완료] {args.out}: {len(report['metrics'])}개 지표; 실측 성능/배포/실시간 주장 불가")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
