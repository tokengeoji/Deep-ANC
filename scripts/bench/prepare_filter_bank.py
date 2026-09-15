#!/usr/bin/env python3
"""합성 FIR bank 준비와 독립 다중 seed 평가. 장치/원본 데이터 다운로드 없음.

새 --out에 bank JSON/NPZ와 평가 JSON/CSV/Markdown을 저장한다. exit 0은 보고서
생성이지 감쇠 성공이 아니다. 학습 실패 bank는 저장하지 않고 누락 사유를 보고한다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.baselines.filter_bank import (  # noqa: E402
    BankConfig, load_bank, new_output_directory, prepare_and_evaluate, save_bank,
)


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(1, f"{self.prog}: 인자 오류: {message}\n")


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument("--out", required=True, help="존재하지 않는 새 결과 폴더")
    for name in ("sample-rate", "hop", "secondary-delay-samples", "stress-preview-samples",
                 "control-length", "train-blocks", "eval-blocks", "window-blocks", "selection-interval-blocks"):
        parser.add_argument("--" + name, type=int)
    for name in ("train-seeds", "validation-seeds", "test-seeds"):
        parser.add_argument("--" + name, nargs="+", type=int)
    for name in ("evaluation-levels", "saturation-levels"):
        parser.add_argument("--" + name, nargs="+", type=float)
    parser.add_argument("--mu", type=float, help="모든 학습/대조군에 동일한 FxNLMS 적응률")
    args = vars(parser.parse_args(argv))
    try:
        destination = new_output_directory(args.pop("out"))
        config = BankConfig(**{key: value for key, value in args.items() if value is not None})
        banks, report = prepare_and_evaluate(config)
        report["artifacts"] = {scenario: {"bank_id": bank.bank_id,
                                         "json": f"{scenario}/bank.json", "npz": f"{scenario}/bank.npz"}
                               for scenario, bank in banks.items()}
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(report["metrics"][0]))
        writer.writeheader()
        for row in report["metrics"]:
            writer.writerow({key: "null" if value is None else value for key, value in row.items()})
        summary = "# 합성 FIR bank / REF-only 선택 진단\n\n"
        summary += "`synthetic_only=true`, `physical_claim_allowed=false`, `performance_claim_allowed=false`.\n\n"
        summary += "조건별 train bank와 validation/test seed를 분리합니다. 실제 음성·음악 평가는 하지 않았습니다.\n\n"
        summary += "전체 수치는 report.json의 metrics와 동일한 metrics.csv에 있으며 증폭/누락 행도 보존합니다.\n\n"
        summary += "\n".join(f"- {warning}" for warning in report["warnings"]) + "\n"
        new_output_directory(destination)
        destination.mkdir(parents=True, exist_ok=False)
        for scenario, bank in banks.items():
            save_bank(bank, destination / scenario)
            restored = load_bank(destination / scenario, expected_context_id=bank.context_id)
            if restored.bank_id != bank.bank_id:
                raise ValueError("저장/복원 bank ID 불일치")
        for filename, content in (("report.json", encoded), ("metrics.csv", buffer.getvalue()),
                                  ("summary.md", summary)):
            with (destination / filename).open("x", encoding="utf-8", newline="") as handle:
                handle.write(content)
    except (OSError, ValueError, TypeError, RuntimeError, FloatingPointError) as exc:
        print(f"[FAIL] 합성 bank: {exc}. 새 경로의 부분 산출물이 있으면 보존합니다.", file=sys.stderr)
        return 1
    invalid = sum(row["status"] != "valid" for row in report["training"])
    print(f"[진단 생성] {destination}: bank {len(banks)}개, 학습 실패 조건 {invalid}개, "
          f"평가 {len(report['runs'])}회 — 실측/실시간 성능 통과 아님")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
