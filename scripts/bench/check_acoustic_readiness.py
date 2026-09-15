#!/usr/bin/env python3
"""소리 없이 acoustic 설정·지연·S(z) 신뢰대역을 진단한다.

기본은 보고만 하고 exit 0. --require-broadband/--require-band 미달 또는 잘못된
파일/설정은 exit 1. 이 진단의 통과는 실기 실행 허가나 감쇠 성공을 의미하지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from zipfile import BadZipFile

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.eval.acoustic_readiness import check_acoustic_readiness  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime_acoustic.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--json", action="store_true", help="JSON 보고서를 stdout으로 출력")
    parser.add_argument("--require-broadband", action="store_true", help="광대역 도착 필요조건 미달 시 exit 1")
    parser.add_argument("--require-band", nargs=2, type=float, metavar=("LOW", "HIGH"), help="요청 대역 전체의 S(z) 신뢰대역 포함을 요구")
    args = parser.parse_args(argv)
    try:
        report = check_acoustic_readiness(
            args.config, args.overrides, require_broadband=args.require_broadband,
            required_band_hz=args.require_band,
        )
    except (OSError, ValueError, TypeError, KeyError, BadZipFile, yaml.YAMLError) as exc:
        if args.json:
            print(json.dumps({"config_valid": False, "passed": False, "error": str(exc)}, ensure_ascii=False))
        else:
            print(f"[FAIL] acoustic 설정 진단: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        budget = report["causality"]
        path = report["secondary_path"]
        print("acoustic 무출력 진단 — 설정과 S(z) NPZ만 확인")
        print(f"REF→ERR 기하선행: {budget['reference_preview_samples']:.2f} samples ({budget['reference_preview_ms']:.3f} ms)")
        print(f"제어음 지연(S + handoff): {budget['control_delay_samples']} samples ({budget['control_delay_ms']:.3f} ms)")
        print(f"필요 예측: {budget['required_prediction_samples']:.2f} samples ({budget['required_prediction_ms']:.3f} ms)")
        print(f"광대역 도착 필요조건: {'통과' if budget['broadband_arrival_condition_met'] else '미달'} (감쇠 성공 보장 아님)")
        print(f"S(z) 신뢰대역: {path['consistency_band_hz']} Hz, 반복 일관성: {path['repeat_consistency']}")
        print(f"1 kHz 초과 검증 대역: {'있음' if report['band_validation']['validated_band_extends_above_1khz'] else '없음/미검증'}")
        for warning in report["warnings"]:
            print(f"[경고] {warning}")
        if report["failed_requirements"]:
            print(f"[FAIL] 선택 게이트: {', '.join(report['failed_requirements'])}")
        elif args.require_broadband or args.require_band is not None:
            print("[PASS] 요청한 선택 게이트 통과")
        else:
            print("[보고 완료] 선택 게이트 없음. exit 0은 광대역 준비 완료를 뜻하지 않습니다.")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
