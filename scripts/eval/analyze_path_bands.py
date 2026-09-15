#!/usr/bin/env python3
"""저장된 ESS S 반복을 무출력 진단한다. 신뢰대역/S NPZ를 만들지 않는다.

exit 0은 진단 파일 저장만 뜻하며 품질/성능 통과가 아니다. 입력·출력 오류는 exit 1.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
from zipfile import BadZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.eval.path_band_diagnostics import analyze_path_bands  # noqa: E402


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, f"인자 오류: {message}\n")


def _output_path(value: str) -> Path:
    path = Path(value).expanduser().absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("출력 경로에 심볼릭 링크를 사용할 수 없습니다")
    if path.exists():
        raise FileExistsError(f"기존 출력 경로는 덮어쓰지 않습니다: {path}")
    if any(part.exists() and not part.is_dir() for part in path.parents):
        raise NotADirectoryError("출력의 상위 경로가 디렉터리가 아닙니다")
    return path


def _text(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _artifacts(report: dict) -> dict[str, str]:
    rows = [{"diagnostic_only": True, "performance_claim_allowed": False,
             "promote_secondary_allowed": False, **row} for row in report["metrics"]]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: _text(value) for key, value in row.items()} for row in rows)
    columns = ("mode", "band", "support_status", "coherence_min", "coherence_median",
               "worst_pair_phase_sensitive_cosine", "max_pair_magnitude_difference_db")
    lines = [
        "# ESS 반복 대역 진단", "",
        "`diagnostic_only=true`, `performance_claim_allowed=false`, `promote_secondary_allowed=false`.", "",
        "일관성 0.9는 참고선이다. 가진대역을 신뢰대역으로 바꾸거나 S 모델을 승격하지 않는다.",
        "fixed_origin/정렬 비교는 같은 compact IR을 사용하며 저장된 지연 위상만 다르다.",
        "모든 반복을 사용했다. 입력 SNR·실시간 클록 안정성은 unknown이다.",
        "null은 계산 불가/지원 밖이지 0이나 통과가 아니다. CSV와 아래 표는 같은 값이다.", "",
        f"- 반복 수: {len(report['repeats_used'])}, 제외: 0",
        f"- 저장된 지연: {report['repeat_delay_samples']} samples",
        f"- 지연 max−min: {report['delay_spread_samples']} samples ({report['delay_spread_ms']} ms)",
        f"- 원자료 SHA-256: `{report['source']['sha256']}`", "",
        "## QA와 한계", "",
    ]
    lines.extend(f"- {issue}" for issue in report["issues"])
    lines.extend(["", "## 대역별 결과", "", "| " + " | ".join(columns) + " |",
                  "|" + "---|" * len(columns)])
    lines.extend("| " + " | ".join(_text(row[key]) for key in columns) + " |" for row in rows)
    lines.extend(["", "전체 QA·모든 pair·주파수 bin 결과와 계산 규약은 report.json에 있다.",
                  "저장 실패 시 부분 산출물이 남을 수 있으므로 재시도는 새 디렉터리를 사용한다."])
    return {"report.json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            "metrics.csv": buffer.getvalue(), "summary.md": "\n".join(lines) + "\n"}


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(description=__doc__)
    parser.add_argument("--raw-npz", required=True, help="calibrate_wideband raw_measurement.npz")
    parser.add_argument("--out", required=True, help="아직 존재하지 않는 진단 디렉터리")
    parser.add_argument("--band", type=float, nargs=2, default=[800.0, 1600.0])
    args = parser.parse_args(argv)
    created = False
    out = None
    try:
        out = _output_path(args.out)
        report = analyze_path_bands(args.raw_npz, args.band)
        payloads = _artifacts(report)
        _output_path(str(out))
        out.mkdir(parents=True, exist_ok=False)
        created = True
        for name, text in payloads.items():
            with (out / name).open("x", encoding="utf-8", newline="") as handle:
                handle.write(text)
    except (OSError, ValueError, TypeError, KeyError, BadZipFile) as exc:
        print(f"[FAIL] ESS 대역 진단: {exc}", file=sys.stderr)
        if created:
            print(f"일부 산출물이 남아 있을 수 있습니다: {out}", file=sys.stderr)
        return 1
    print(f"진단 저장: {out} — 성능/신뢰대역/S 승격 판정 아님")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
