#!/usr/bin/env python3
"""녹음된 acoustic REF/ERR를 새 디렉터리에 진단 보고한다. 오디오 장치를 열지 않는다.

exit 0: 모든 cycle의 구간이 완전하고 최소 한 cycle 전체 대역 비교 가능(감쇠 성공 판정 아님).
exit 2: 입력은 읽었지만 비교 불가 또는 불완전 cycle. 진단 산출물은 보존한다.
exit 1: 잘못된 입력·설정·인자 또는 I/O 오류. 기존 경로는 덮어쓰지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
from zipfile import BadZipFile

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import load_runtime_config  # noqa: E402
from deep_anc.eval.acoustic_session import analyze_acoustic_session, sha256_file  # noqa: E402


METRIC_FIELDS = (
    "cycle", "band", "low_hz", "high_hz", "trusted", "off_pre_power", "off_post_power",
    "on_power", "observed_err_reduction_db", "observed_vs_pre_db", "observed_vs_post_db",
    "off_drift_db", "median_observed_db", "p10_observed_db", "worst_observed_db",
    "worst10_mean_db", "n_on_windows", "worst10_window_count", "n_power_invalid_on_windows",
    "emergent_on_energy", "on_minus_baseline_power", "ref_pre_power", "ref_on_power",
    "ref_post_power", "ref_on_vs_pre_db", "ref_on_vs_post_db",
)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: 인자 오류: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(description=__doc__)
    parser.add_argument("--npz", required=True, help="기존 runtime 녹음 NPZ")
    parser.add_argument("--config", required=True, help="녹음 조건과 일치하는 acoustic runtime YAML")
    parser.add_argument("--source-family", required=True, help="소스 종류: speech/music/machine 등")
    parser.add_argument("--out", required=True, help="아직 존재하지 않는 결과 디렉터리")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--startup-guard-seconds", type=float, default=1.0)
    parser.add_argument("--on-warmup-seconds", type=float, default=2.0)
    parser.add_argument("--edge-guard-seconds", type=float, default=0.5)
    parser.add_argument("--power-floor", type=float, default=1e-12)
    return parser


def _new_output_path(value: str) -> Path:
    # resolve()로 심볼릭 링크의 존재를 숨기지 않는다. 상위 링크도 출력 우회 경로다.
    out = Path(value).expanduser()
    if not out.is_absolute():
        out = Path.cwd() / out
    for part in (out, *out.parents):
        if part.is_symlink():
            raise ValueError(f"출력 경로에 심볼릭 링크를 사용할 수 없습니다: {part}")
    if out.exists():
        raise FileExistsError(f"기존 출력 경로는 덮어쓰지 않습니다: {out}")
    for parent in out.parents:
        if parent.exists() and not parent.is_dir():
            raise NotADirectoryError(f"출력의 상위 경로가 디렉터리가 아닙니다: {parent}")
    return out


def _value(value) -> str:
    """CSV와 Markdown의 수치 표현을 공유한다. 계산 불가를 0으로 바꾸지 않는다."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _markdown(value) -> str:
    return _value(value).replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`").replace("\n", " ")


def _context(report: dict) -> dict:
    return {
        "diagnostic_only": report["diagnostic_only"],
        "performance_claim_allowed": report["performance_claim_allowed"],
        "source_family": report["source_family"],
        "recording_context_verified": report["provenance"]["recording_context_verified"],
    }


def metric_records(report: dict) -> list[dict]:
    """대역·무신호 행을 모두 보존하고 계산하지 못한 cycle도 명시적 행으로 남긴다."""
    context = _context(report)
    rows = [
        {**context, "status": "available" if row["observed_err_reduction_db"] is not None else "unavailable",
         "unavailable_reason": None if row["observed_err_reduction_db"] is not None else "power_below_floor",
         **row}
        for row in report["metrics"]
    ]
    for cycle in report["cycles"]:
        if not cycle["usable"]:
            rows.append({
                **context, "status": "unavailable", "unavailable_reason": cycle["reason"],
                **dict.fromkeys(METRIC_FIELDS), "cycle": cycle["cycle"], "band": "unavailable",
            })
    if not rows:
        rows.append({
            **context, "status": "unavailable", "unavailable_reason": "no_complete_cycle",
            **dict.fromkeys(METRIC_FIELDS), "band": "unavailable",
        })
    return rows


def _csv(rows: list[dict]) -> str:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: _value(row.get(key)) for key in fields} for row in rows)
    return buffer.getvalue()


def _summary(report: dict, rows: list[dict]) -> str:
    lines = [
        "# Acoustic 녹음 진단", "",
        "`diagnostic_only=true`, `performance_claim_allowed=false`.", "",
        "서로 다른 시각의 ERR 파워를 비교한 진단이다. 외부 소리의 변화와 REF의 F 누설이 있어 ANC 성능으로 인증하지 않는다.",
        "CSV·표의 null은 계산 불가이며 0 dB가 아니다. 신뢰대역 밖의 결과도 삭제하지 않는다.", "",
        f"- 소스 종류: {_markdown(report['source_family'])}",
        f"- 비교 계산 가능: {_value(report['comparison_available'])}",
        f"- 모든 cycle 완전: {_value(report['all_cycles_complete'])}",
        f"- 녹음 문맥 검증: {_value(report['provenance']['recording_context_verified'])}",
        f"- 원시 녹음: {_markdown(report['provenance']['recording_path'])}",
        f"- 녹음 SHA-256: `{report['provenance']['recording_sha256']}`",
        f"- S SHA-256: `{report['provenance']['secondary_sha256']}`", "",
        "## 경고와 검증 범위", "",
    ]
    lines.extend(f"- {_markdown(issue)}" for issue in report["issues"])
    lines.extend(f"- {_markdown(warning)}" for warning in report["path_readiness"]["warnings"])
    if not report["provenance"]["recording_context_verified"]:
        lines.append("- legacy_recording_context_unverified: 녹음 당시 설정이 없어 사용자 제공 설정과의 실제 일치를 확인하지 못했다.")
    lines.extend(["", "## Cycle", "", "| cycle | usable | reason |", "|---|---|---|"])
    lines.extend(
        f"| {cycle['cycle']} | {_value(cycle['usable'])} | {_markdown(cycle['reason'])} |"
        for cycle in report["cycles"]
    )
    if not report["cycles"]:
        lines.append("| null | false | no_full_on_interval |")
    columns = ("cycle", "band", "trusted", "status", "observed_err_reduction_db",
               "median_observed_db", "worst10_mean_db", "emergent_on_energy")
    lines.extend(["", "## 관측 지표", "", "아래 값은 metrics.csv와 같은 행·수치 표현을 사용한다.", "",
                  "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)])
    lines.extend("| " + " | ".join(_markdown(row.get(key)) for key in columns) + " |" for row in rows)
    lines.extend(["", "## 산출물", "", "- report.json: 전체 진단·구간·원자료 경로와 해시",
                  "- metrics.csv: 모든 대역과 계산 불가 cycle의 지표", "- summary.md: 이 요약"])
    lines.append("- windows.csv: 구간별 원시 파워" if report["windows"] else
                 "- windows.csv: 생성하지 않음 — 유효한 분석 창이 없다.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    created = False
    out = None
    try:
        out = _new_output_path(args.out)
        config_path = Path(args.config).expanduser().resolve()
        config_hash = sha256_file(config_path)
        cfg = load_runtime_config(config_path, args.overrides)
        report = analyze_acoustic_session(
            args.npz, cfg, args.source_family, window_seconds=args.window_seconds,
            startup_guard_seconds=args.startup_guard_seconds, on_warmup_seconds=args.on_warmup_seconds,
            edge_guard_seconds=args.edge_guard_seconds, power_floor=args.power_floor,
        )
        if sha256_file(config_path) != config_hash:
            raise ValueError("분석 중 runtime 설정 파일이 변경됐습니다")
        report["analysis_config"] = {"path": str(config_path), "sha256": config_hash, "overrides": args.overrides}
        report["artifacts"] = {"report": "report.json", "metrics": "metrics.csv", "summary": "summary.md",
                               "windows": "windows.csv" if report["windows"] else None,
                               "windows_omission_reason": None if report["windows"] else "no_valid_windows"}
        rows = metric_records(report)
        payloads = {
            "report.json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            "metrics.csv": _csv(rows), "summary.md": _summary(report, rows),
        }
        if report["windows"]:
            payloads["windows.csv"] = _csv([{**_context(report), **row} for row in report["windows"]])
        # 분석과 직렬화가 끝나기 전에는 결과 디렉터리도 만들지 않는다.
        _new_output_path(str(out))
        out.mkdir(parents=True, exist_ok=False)
        created = True
        for name, text in payloads.items():
            with (out / name).open("x", encoding="utf-8", newline="") as handle:
                handle.write(text)
    except (OSError, ValueError, TypeError, KeyError, BadZipFile, yaml.YAMLError) as exc:
        print(f"[FAIL] acoustic 녹음 진단: {exc}", file=sys.stderr)
        if created:
            print(f"일부 산출물이 남아 있을 수 있습니다: {out}", file=sys.stderr)
        return 1
    print(f"진단 저장: {out} (diagnostic_only=true, performance_claim_allowed=false)")
    if not report["windows"]:
        print("windows.csv 생략: 유효한 분석 창이 없습니다.")
    if report["all_cycles_complete"] and report["comparison_available"]:
        print("비교 계산 가능 — 감쇠 성공 판정이 아닙니다.")
        return 0
    print("비교 불가 또는 불완전 cycle — 진단 자료를 확인하세요.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
