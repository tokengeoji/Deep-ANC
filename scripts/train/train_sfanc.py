#!/usr/bin/env python3
"""SFANC FIR 계산 → 과거 REF 선택기 학습 → 독립 평가. 오디오 출력 없음."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from deep_anc.train.sfanc_experiment import ExperimentConfig, run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="존재하지 않는 새 실행 폴더")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--librispeech-root", help="로컬 CC BY 4.0 LibriSpeech root, 다운로드하지 않음")
    args = parser.parse_args(argv)
    try:
        config = ExperimentConfig(**json.loads(Path(args.config).read_text()))
        run_experiment(config, args.out, device=args.device, librispeech_root=args.librispeech_root)
    except (ValueError, OSError, TypeError, RuntimeError) as exc:
        print(f"[실패] {exc}; 부분 산출물은 보존합니다.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
