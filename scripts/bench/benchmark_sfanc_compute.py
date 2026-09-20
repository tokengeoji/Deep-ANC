#!/usr/bin/env python3
"""무오디오 SFANC/미학습 Hybrid 구조 계산시간 CLI. 기존 결과 덮어쓰기 금지."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from deep_anc.eval.sfanc_compute import benchmark_compute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="OMAP S로 준비한 SFANC run 디렉터리")
    parser.add_argument("--out", required=True, help="아직 존재하지 않는 새 결과 디렉터리")
    parser.add_argument("--hybrid-config", default=str(ROOT / "configs/model_tiny.yaml"))
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu"])
    parser.add_argument("--blocks", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    report = benchmark_compute(args.run, args.out, hybrid_config=args.hybrid_config,
        devices=args.devices, blocks=args.blocks, warmup=args.warmup, trials=args.trials,
        torch_threads=args.torch_threads, seed=args.seed)
    print(json.dumps({"report": str(Path(args.out) / "report.json"),
        "deployment_allowed": report["deployment_allowed"], "measurements": report["measurements"]},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
