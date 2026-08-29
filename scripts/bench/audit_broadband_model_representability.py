#!/usr/bin/env python3
"""오디오 없이 Tiny/Base의 광대역 출력 표현력과 limiter 여유를 감사한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

# NumPy/PyTorch를 import하기 전에 기본 BLAS thread 수를 제한한다. 호출자가 값을
# 명시했으면 존중한다. 이 감사는 작은 행렬만 다루므로 thread 폭증이 더 느리다.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from deep_anc.models.broadband_representability import (  # noqa: E402
    broadband_g0_gate_spec,
    checkpoint_polyphase_report,
    load_checkpoint_snapshot,
    output_lattice_contract,
    tone_limiter_feasibility_report,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_tone_diagnostic(
    primary_path: Path,
    secondary_path: Path,
    *,
    source_peaks: list[float],
    limiter_limit: float,
) -> dict[str, Any]:
    from scipy.signal import freqz

    with np.load(primary_path, allow_pickle=False) as primary, np.load(
        secondary_path, allow_pickle=False
    ) as secondary:
        sample_rate = int(primary["sample_rate"])
        if sample_rate != int(secondary["sample_rate"]):
            raise ValueError("strict P/S sample rate가 다릅니다")
        p_band = tuple(float(v) for v in primary["consistency_band_hz"])
        s_band = tuple(float(v) for v in secondary["consistency_band_hz"])
        lo = max(p_band[0], s_band[0])
        hi = min(p_band[1], s_band[1])
        if not 0.0 < lo < hi:
            raise ValueError("strict P/S consistency band 교집합이 없습니다")
        frequency = np.linspace(lo, hi, 4097, dtype=np.float64)
        omega = 2.0 * np.pi * frequency / float(sample_rate)
        _, p_response = freqz(np.asarray(primary["fir"]), worN=omega)
        _, s_response = freqz(np.asarray(secondary["fir"]), worN=omega)
        capture_match = str(primary["capture_id"].item()) == str(
            secondary["capture_id"].item()
        )
    return {
        "role": "stage1_strict_tone_only_diagnostic_not_broadband",
        "primary_path": str(primary_path),
        "primary_sha256": _sha256(primary_path),
        "secondary_path": str(secondary_path),
        "secondary_sha256": _sha256(secondary_path),
        "same_capture": capture_match,
        "frequency_band_hz": [lo, hi],
        "source_peaks": [
            tone_limiter_feasibility_report(
                p_response,
                s_response,
                source_peak=value,
                limiter_limit=limiter_limit,
            )
            for value in source_peaks
        ],
        "canonical_broadband_training_admitted": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "runs/pretrain_tiny_corrected/ckpt/best.pt",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict-primary", type=Path)
    parser.add_argument("--strict-secondary", type=Path)
    parser.add_argument(
        "--source-peak", type=float, action="append", default=[]
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.resolve()
    state, checkpoint_sha = load_checkpoint_snapshot(checkpoint)
    cfg = dict(state["cfg"])
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("checkpoint cfg.model이 없습니다")
    limiter_limit = float((model_cfg.get("limiter") or {}).get("limit", 0.0))
    if limiter_limit <= 0.0:
        raise ValueError("checkpoint model limiter.limit가 없습니다")

    lattice = output_lattice_contract(model_cfg)
    polyphase = checkpoint_polyphase_report(state["model"], model_cfg)
    canonical_contract = cfg.get("experiment_contract_sha256")
    canonical_artifact = bool(
        isinstance(canonical_contract, str)
        and len(canonical_contract) == 64
        and cfg.get("role") in {"canonical_pretrain", "canonical_finetune"}
    )
    best_metric = state.get("best_metric")
    best_metric_value = (
        float(best_metric)
        if best_metric is not None and np.isfinite(float(best_metric))
        else None
    )
    result: dict[str, Any] = {
        "kind": "READ_ONLY_DIAGNOSTIC_broadband_model_representability",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": int(state.get("step", -1)),
        "checkpoint_best_metric": best_metric_value,
        "checkpoint_physics_status": cfg.get("physics_status"),
        "checkpoint_digital_reference_lead_samples": cfg.get(
            "digital_reference_lead_samples",
            (cfg.get("data") or {}).get("digital_reference_lead_samples"),
        ),
        "canonical_checkpoint_contract_present": canonical_artifact,
        "model_lattice": lattice,
        "checkpoint_polyphase": polyphase,
        "limiter": {
            "limit": limiter_limit,
            "io_scale": float(model_cfg.get("io_scale", 0.0)),
            "ninety_percent_output": 0.9 * limiter_limit,
            "prelimit_u_over_limit_at_ninety_percent": float(np.arctanh(0.9)),
        },
        "required_broadband_g0": broadband_g0_gate_spec(),
    }
    if bool(args.strict_primary) != bool(args.strict_secondary):
        raise ValueError("strict tone diagnostic은 --strict-primary/secondary를 함께 요구합니다")
    if args.strict_primary and args.strict_secondary:
        peaks = args.source_peak or [0.003, 0.01, 0.03]
        result["strict_limiter_diagnostic"] = _strict_tone_diagnostic(
            args.strict_primary.resolve(),
            args.strict_secondary.resolve(),
            source_peaks=[float(v) for v in peaks],
            limiter_limit=limiter_limit,
        )

    result["overall_status"] = "BLOCKED"
    result["overall_reasons"] = [
        "canonical fullband causal P/S operator와 exact prefix/state가 아직 없습니다",
        "canonical broadband G0 raw receipt가 아직 없습니다",
        "polyphase positive-branch PASS만으로 실제 plant cancellation을 증명할 수 없습니다",
    ]
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        # 진단도 세대 evidence이므로 덮어쓰지 않는다.
        with output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    print(encoded, end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
