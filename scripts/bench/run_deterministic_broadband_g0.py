#!/usr/bin/env python3
"""스피커·실측 plant 없이 실제 Tiny forward/streaming 광대역 구조 G0를 실행한다."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# import 전에 작은 감사 작업의 CPU thread를 제한한다. 호출자가 지정한 값은 존중한다.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# CUDA deterministic algorithms는 torch import 전에 이 workspace 계약이 있어야 한다.
# 빠져 있으면 cuBLAS backward가 조용히 비결정적으로 실행되는 대신 즉시 실패한다.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from deep_anc.models.broadband_deterministic_g0 import (  # noqa: E402
    DETERMINISTIC_G0_SEED,
    DETERMINISTIC_G0_STEPS,
    run_deterministic_broadband_g0,
)


def _scalar(npz: np.lib.npyio.NpzFile, key: str) -> object:
    if key not in npz.files:
        raise ValueError(f"plant NPZ에 {key}가 없습니다")
    value = np.asarray(npz[key])
    if value.size != 1:
        raise ValueError(f"plant NPZ {key}가 scalar가 아닙니다")
    return value.reshape(()).item()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config", type=Path, default=ROOT / "configs/model_tiny.yaml"
    )
    parser.add_argument(
        "--strict-primary",
        type=Path,
        default=ROOT / "assets/measured/primary_path_il_strict_5dc06fdd.npz",
    )
    parser.add_argument(
        "--strict-secondary",
        type=Path,
        default=ROOT / "assets/measured/secondary_path_il_strict_5dc06fdd.npz",
    )
    parser.add_argument("--handoff-samples", type=int, default=256)
    parser.add_argument("--steps", type=int, default=DETERMINISTIC_G0_STEPS)
    parser.add_argument("--seed", type=int, default=DETERMINISTIC_G0_SEED)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    model_cfg = yaml.safe_load(args.model_config.resolve().read_text(encoding="utf-8"))
    if not isinstance(model_cfg, dict):
        raise ValueError("model config가 mapping이 아닙니다")
    with np.load(args.strict_primary.resolve(), allow_pickle=False) as primary, np.load(
        args.strict_secondary.resolve(), allow_pickle=False
    ) as secondary:
        primary_rate = int(_scalar(primary, "sample_rate"))
        secondary_rate = int(_scalar(secondary, "sample_rate"))
        if primary_rate != secondary_rate:
            raise ValueError("strict P/S sample rate가 다릅니다")
        primary_capture = str(_scalar(primary, "capture_id"))
        secondary_capture = str(_scalar(secondary, "capture_id"))
        if primary_capture != secondary_capture:
            raise ValueError("strict P/S가 같은 capture가 아닙니다")
        primary_delay = int(_scalar(primary, "delay_samples"))
        secondary_delay = int(_scalar(secondary, "delay_samples"))

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA가 요청됐지만 사용할 수 없습니다")
    receipt = run_deterministic_broadband_g0(
        model_cfg,
        primary_delay_samples=primary_delay,
        secondary_delay_samples=secondary_delay,
        handoff_samples=int(args.handoff_samples),
        sample_rate=primary_rate,
        steps=int(args.steps),
        seed=int(args.seed),
        device=device,
    )
    receipt["strict_capture_id_timing_only"] = primary_capture
    receipt["strict_primary_path_timing_only"] = str(args.strict_primary.resolve())
    receipt["strict_secondary_path_timing_only"] = str(args.strict_secondary.resolve())
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    print(encoded, end="")
    return 0 if receipt["status"] == "PASS_STRUCTURAL_DIAGNOSTIC" else 2


if __name__ == "__main__":
    raise SystemExit(main())
