#!/usr/bin/env python3
"""명시 stage된 acoustic 학습 자료의 무출력 QA와 family×split smoke.

Drive 다운로드/원격 조회/학습/장치 실행은 하지 않는다. exit 0은 로컬 자료와 합성
데이터 로더 검사 성공일 뿐 실제 감쇠, 목표대역 신뢰, 학습 완료를 뜻하지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import _resolve_path, load_train_config  # noqa: E402
from deep_anc.data.synth_dataset import SynthANCDataset, validate_prepared_handoff  # noqa: E402
from deep_anc.data.synthetic_signals import KINDS  # noqa: E402
from deep_anc.dsp.secondary_path import load_secondary_path  # noqa: E402


def _band_fraction(values: np.ndarray, fs: int, low=800, high=1600) -> float | None:
    power = np.abs(np.fft.rfft(values.astype(np.float64))) ** 2
    weights = np.full(power.size, 2.0)
    weights[0] = 1
    if values.size % 2 == 0:
        weights[-1] = 1
    power *= weights
    frequency = np.fft.rfftfreq(values.size, 1 / fs)
    total = float(power.sum())
    return float(power[(frequency >= low) & (frequency < high)].sum() / total) if total > 0 else None


def check_acoustic_training_data(cfg: dict) -> dict:
    if cfg["data"].get("require_prepared_data") is not True:
        raise ValueError("이 QA는 require_prepared_data=true 설정만 검사합니다")
    handoff = validate_prepared_handoff(cfg["duct"])
    rows, metadata, synthetic = [], None, []
    for split_index, split in enumerate(("train", "val", "test")):
        dataset = SynthANCDataset(cfg["data"], cfg["duct"], split=split, seed=20260915)
        if metadata is None:
            metadata = dataset.prepared_data_metadata
        elif dataset.prepared_data_metadata != metadata:
            raise ValueError("family/split QA 도중 prepared 입력/감사 기록이 변경됐습니다")
        families = [family for family, ratio in dataset.mix_ratio.items() if ratio > 0]
        for index, family in enumerate(families):
            # 준비 QA는 모든 family를 검사했고 여기서는 각각을 확정 선택한다.
            dataset.mix_ratio = {family: 1.0}
            rng = np.random.default_rng(20260915 + split_index * 100 + index)
            generator = dataset.synthetic_generator(seed=20260915 + index)
            item = dataset._make_item(rng, generator)
            x, disturbance = item["x"].numpy(), item["d"].numpy()
            if (x.shape != (2, dataset.segment) or disturbance.shape != (1, dataset.segment)
                    or not np.isfinite(x).all() or not np.isfinite(disturbance).all()):
                raise ValueError(f"acoustic sample shape/finite 실패: {split}/{family}")
            rows.append({
                "split": split, "source_family": family, "samples": dataset.segment,
                "rir_indices": dataset.rir_indices.tolist(),
                "input_ref_target_power_fraction": _band_fraction(x[0], dataset.fs),
                "disturbance_target_power_fraction": _band_fraction(disturbance[0], dataset.fs),
                "finite": True,
            })
        if split == "train":
            for kind_index, kind in enumerate(KINDS):
                generator = dataset.synthetic_generator(seed=20260915 + kind_index)
                values = [generator.generate(dataset.fs, kind=kind) for _ in range(8)]
                synthetic.append({
                    "kind": kind, "examples": len(values),
                    "mean_target_power_fraction": float(np.mean([
                        _band_fraction(value, dataset.fs) for value in values
                    ])),
                    "mean_below_800_power_fraction": float(np.mean([
                        _band_fraction(value, dataset.fs, 0, 800) for value in values
                    ])),
                })
    secondary = load_secondary_path(_resolve_path(cfg["duct"]["secondary_path"]["npz"]))
    return {
        "data_ready": True, "diagnostic_only": True, "performance_claim_allowed": False,
        "real_time_claim_allowed": False, "training_launched": False,
        "drive_inventory_ready": None, "scope": "explicitly_staged_local_data_only",
        "reference_mode": "acoustic", "digital_reference_lead_samples": 0,
        "source_metadata": cfg["data"].get("source_metadata", {}),
        "prepared_data": metadata, "smoke": rows, "synthetic_coverage": synthetic,
        "secondary_delay_samples": secondary.delay_samples, "handoff_extra_samples": handoff,
        "loss_plant_delay_samples": secondary.delay_samples + handoff,
        "secondary_consistency_band_hz": secondary.consistency_band_hz,
        "notes": [
            "목표대역 에너지 포함은 측정 S 신뢰/감쇠 가능성/성능 입증이 아니다.",
            "입력 REF 비율에는 합성 RIR·마이크 잡음·채널 dropout이 포함된다.",
            "합성 P_ref/P_err의 장치 절대 gain은 미검증이며 F 피드백은 open-loop에 없다.",
            "로컬 SHA 검증은 Drive 원격 원본이나 라이선스 완전성 인증을 대신하지 않는다.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_acoustic_prepared.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--json", action="store_true", help="JSON을 stdout으로 출력")
    args = parser.parse_args(argv)
    try:
        report = check_acoustic_training_data(load_train_config(args.config, args.set))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        report = {"data_ready": False, "diagnostic_only": True, "performance_claim_allowed": False,
                  "drive_inventory_ready": None, "training_launched": False, "error": str(exc)}
        print(json.dumps(report, ensure_ascii=False, allow_nan=False) if args.json else f"준비 실패: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) if args.json
          else f"로컬 준비 QA 통과: {len(report['smoke'])} family×split 표본 — 실기/학습 성공 판정 아님")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
