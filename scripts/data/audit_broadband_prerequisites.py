#!/usr/bin/env python3
"""광대역 2/4/8 kHz 학습 전 실제 P/S와 recorded WAV를 read-only 감사한다.

오디오 장치를 열지 않는다. 기본 실행은 JSON을 stdout에만 쓰며 ``--output``을 지정해도
기존 파일은 덮어쓰지 않는다. 현재 strict v1이나 진단 fullband raw를 광대역 자산으로
승격하는 도구가 아니다.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from deep_anc.data.recorded_broadband_coverage import (
    scan_recorded_broadband_coverage,
    sha256_file,
)
from deep_anc.dsp.control_band_contract import ControlBandContract


REPO_ROOT = Path(__file__).resolve().parents[2]


def _scalar(data: np.lib.npyio.NpzFile, key: str, default: object = None) -> object:
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    if value.size != 1:
        return value.tolist()
    return value.reshape(-1)[0].item()


def _band(data: np.lib.npyio.NpzFile, key: str) -> tuple[float, float] | None:
    if key not in data.files:
        return None
    raw = np.asarray(data[key], dtype=np.float64).reshape(-1)
    if raw.size != 2 or not np.all(np.isfinite(raw)):
        return None
    return (float(raw[0]), float(raw[1]))


def audit_current_plant_pair(
    primary_path: str | Path,
    secondary_path: str | Path,
    *,
    contract: ControlBandContract,
) -> dict[str, Any]:
    primary_file = Path(primary_path).expanduser().resolve()
    secondary_file = Path(secondary_path).expanduser().resolve()
    reasons: list[str] = []
    with np.load(primary_file, allow_pickle=False) as primary, np.load(
        secondary_file, allow_pickle=False
    ) as secondary:
        p_capture = str(_scalar(primary, "capture_id", ""))
        s_capture = str(_scalar(secondary, "capture_id", ""))
        p_raw_sha = str(_scalar(primary, "source_raw_npz_sha256", ""))
        s_raw_sha = str(_scalar(secondary, "source_raw_npz_sha256", ""))
        p_analysis_sha = str(_scalar(primary, "source_analysis_npz_sha256", ""))
        s_analysis_sha = str(_scalar(secondary, "source_analysis_npz_sha256", ""))
        if not p_capture or p_capture != s_capture:
            reasons.append("P/S capture_id가 같지 않습니다")
        if not p_raw_sha or p_raw_sha != s_raw_sha:
            reasons.append("P/S immutable raw SHA가 같지 않습니다")
        if not p_analysis_sha or p_analysis_sha != s_analysis_sha:
            reasons.append("P/S immutable analysis SHA가 같지 않습니다")
        for label, data in (("P", primary), ("S", secondary)):
            if (
                int(_scalar(data, "sample_rate", -1)),
                int(_scalar(data, "calibration_block_size", -1)),
                str(_scalar(data, "calibration_latency", "")),
            ) != (48_000, 256, "low"):
                reasons.append(f"{label}가 48kHz/256/low가 아닙니다")
            if int(_scalar(data, "xrun_count", -1)) != 0:
                reasons.append(f"{label} xrun_count가 0이 아닙니다")
            if str(_scalar(data, "output_pcm_provenance", "")) != "observed_submitted_int16":
                reasons.append(f"{label} observed submitted PCM provenance가 없습니다")
            consistency = _band(data, "consistency_band_hz")
            if consistency is None or consistency[0] > contract.point_control_target_hz[0] or (
                consistency[1] < contract.point_control_target_hz[1]
            ):
                reasons.append(
                    f"{label} consistency band {consistency}가 광대역 target "
                    f"{contract.point_control_target_hz}를 덮지 않습니다"
                )
            excitation = _band(data, "excitation_band_hz")
            if excitation is None or excitation[1] < contract.required_excitation_upper_hz:
                reasons.append(
                    f"{label} excitation upper {None if excitation is None else excitation[1]}Hz가 "
                    f"8k octave 요구 {contract.required_excitation_upper_hz:.3f}Hz 미만입니다"
                )
            artifact_contract = str(_scalar(data, "control_band_contract_sha256", ""))
            if artifact_contract != contract.digest():
                reasons.append(f"{label}에 broadband control-band contract SHA가 없습니다/다릅니다")
            subbands = (
                np.asarray(data["band_consistency_hz"], dtype=np.float64).tolist()
                if "band_consistency_hz" in data.files
                else None
            )
            expected = [list(value) for value in contract.point_control_subbands_hz]
            if subbands != expected:
                reasons.append(f"{label} band consistency rows가 광대역 7개 subband와 다릅니다")

        facts = {
            "same_capture": bool(p_capture and p_capture == s_capture),
            "capture_id": p_capture if p_capture == s_capture else None,
            "same_raw_sha": bool(p_raw_sha and p_raw_sha == s_raw_sha),
            "same_analysis_sha": bool(p_analysis_sha and p_analysis_sha == s_analysis_sha),
            "primary_consistency_band_hz": _band(primary, "consistency_band_hz"),
            "secondary_consistency_band_hz": _band(secondary, "consistency_band_hz"),
            "primary_excitation_band_hz": _band(primary, "excitation_band_hz"),
            "secondary_excitation_band_hz": _band(secondary, "excitation_band_hz"),
        }

    return {
        "status": "PASS" if not reasons else "BLOCKED",
        "reasons": reasons,
        "facts": facts,
        "primary": {
            "path": str(primary_file),
            "size_bytes": primary_file.stat().st_size,
            "sha256": sha256_file(primary_file),
        },
        "secondary": {
            "path": str(secondary_file),
            "size_bytes": secondary_file.stat().st_size,
            "sha256": sha256_file(secondary_file),
        },
    }


def coverage_blockers(
    coverage: dict[str, Any],
    *,
    contract: ControlBandContract,
    minimum_groups: int = 4,
) -> list[str]:
    reasons: list[str] = []
    by_split = coverage["summary"]["by_split_family"]
    for split in ("train", "val", "test"):
        for family in contract.source_families:
            family_result = (by_split.get(split) or {}).get(family)
            if family_result is None:
                reasons.append(f"{split}/{family}: 세션이 없습니다")
                continue
            for index, band in enumerate(contract.point_control_subbands_hz):
                groups = int(
                    family_result["subbands"][index]["joint_pass_independent_groups"]
                )
                if groups < minimum_groups:
                    reasons.append(
                        f"{split}/{family}/{band[0]:.0f}-{band[1]:.0f}Hz: "
                        f"coherence+target-d coverage 독립 group {groups} < {minimum_groups}"
                    )
    return reasons


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary",
        default="assets/measured/primary_path_il_strict_5dc06fdd.npz",
    )
    parser.add_argument(
        "--secondary",
        default="assets/measured/secondary_path_il_strict_5dc06fdd.npz",
    )
    parser.add_argument(
        "--manifest", default="data/manifests/recorded_regrouped.jsonl"
    )
    parser.add_argument("--qa", default="data/manifests/recorded_qa.json")
    parser.add_argument("--start-seconds", type=float, default=5.0)
    parser.add_argument("--stop-seconds", type=float, default=65.0)
    parser.add_argument("--nperseg", type=int, default=8192)
    parser.add_argument("--noverlap", type=int, default=4096)
    parser.add_argument("--minimum-groups", type=int, default=4)
    parser.add_argument("--plant-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = ControlBandContract.broadband_point_control()
    plant = audit_current_plant_pair(args.primary, args.secondary, contract=contract)
    coverage = None
    data_reasons: list[str] = ["--plant-only로 recorded WAV scan을 생략했습니다"]
    if not args.plant_only:
        coverage = scan_recorded_broadband_coverage(
            args.manifest,
            contract=contract,
            qa_path=args.qa,
            start_seconds=args.start_seconds,
            stop_seconds=args.stop_seconds,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
        )
        data_reasons = coverage_blockers(
            coverage, contract=contract, minimum_groups=args.minimum_groups
        )

    payload = {
        "schema": "broadband_prerequisite_audit_v1",
        "role": "diagnostic_only_not_readiness_receipt",
        "status": (
            "PASS"
            if plant["status"] == "PASS" and not data_reasons and coverage is not None
            else "BLOCKED"
        ),
        "control_band_contract": contract.model_dump(mode="json"),
        "control_band_contract_sha256": contract.digest(),
        "plant": plant,
        "recorded_coverage": coverage,
        "recorded_blockers": data_reasons,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "git": _git_state(),
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(text)
        print(f"[saved] {output}", file=sys.stderr)
    print(text, end="")
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
