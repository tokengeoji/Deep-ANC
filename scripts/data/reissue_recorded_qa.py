#!/usr/bin/env python3
"""현행 canonical recorded QA를 재발행하거나 기존 report의 authority를 검사한다.

기본 ``--verify``는 raw WAV를 읽지 않고 manifest/config/strict P/S와 report binding만
검사한다. ``--reissue``만 82개 recorded session 전수 QA를 다시 실행한다. 어느 모드도
오디오 장치, microphone, speaker를 열지 않는다.

예시:

  # 과거 lead=116 QA를 current evidence로 쓰지 못하게 확인
  .venv/bin/python scripts/data/reissue_recorded_qa.py --verify \\
    --qa-json data/manifests/recorded_qa.json

  # 현재 regrouped manifest + strict P/S로 새 QA를 results/ 아래에 발행
  .venv/bin/python scripts/data/reissue_recorded_qa.py --reissue
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT  # noqa: E402
from deep_anc.data.recorded_qa_reissue import (  # noqa: E402
    CANONICAL_RECORDED_MANIFEST,
    RecordedQAReissueError,
    build_current_recorded_qa_provenance,
    render_current_recorded_qa_markdown,
    reissue_current_recorded_qa,
    validate_current_recorded_qa_report,
)


DEFAULT_OUT_JSON = "results/data_audit/recorded_qa_reissue/current/recorded_qa.json"
DEFAULT_OUT_MD = "results/data_audit/recorded_qa_reissue/current/recorded_qa.md"


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _write_exclusive(path: Path, text: str) -> None:
    """과거 QA를 덮어쓰지 않고 새 결과도 명시 replace 없이만 쓴다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise RecordedQAReissueError(
            f"출력 경로가 이미 있습니다 (새 run 경로를 지정하세요): {path}"
        ) from exc


def _require_fresh_output_paths(*paths: Path) -> None:
    """긴 전수 QA 전, JSON/MD가 모두 비어 있는지 한 번에 확인한다."""

    occupied = [str(path) for path in paths if path.exists()]
    if occupied:
        raise RecordedQAReissueError(
            "출력 경로가 이미 있습니다 (새 run 경로를 지정하세요): " + ", ".join(occupied)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true", help="기존 QA report binding만 검사")
    mode.add_argument("--reissue", action="store_true", help="전수 QA 후 새 report를 발행")
    parser.add_argument("--qa-json", help="--verify 대상 QA JSON")
    parser.add_argument("--manifest", default=CANONICAL_RECORDED_MANIFEST)
    parser.add_argument("--data-config", default="configs/data_sim.yaml")
    parser.add_argument("--duct-config", default="configs/duct.yaml")
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", default=DEFAULT_OUT_MD)
    parser.add_argument("--block-frames", type=int, default=262_144)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify and not args.qa_json:
        print("[FAIL] --verify에는 --qa-json이 필요합니다", file=sys.stderr)
        return 2
    if args.verify and (
        args.out_json != DEFAULT_OUT_JSON or args.out_md != DEFAULT_OUT_MD
    ):
        print("[FAIL] --verify에서는 출력 옵션을 사용할 수 없습니다", file=sys.stderr)
        return 2

    try:
        expected = build_current_recorded_qa_provenance(
            manifest_path=args.manifest,
            data_config_path=args.data_config,
            duct_config_path=args.duct_config,
        )
        if args.verify:
            qa_json = _repo_path(args.qa_json)
            report = json.loads(qa_json.read_text(encoding="utf-8"))
            validate_current_recorded_qa_report(
                report, expected_provenance=expected, repo_root=REPO_ROOT
            )
            print(
                "[PASS] recorded QA authority: "
                f"manifest={expected['manifest']['path']}, "
                f"lead={expected['timing']['training_timing_contract']['digital_reference_lead_samples']}"
            )
            return 0

        out_json = _repo_path(args.out_json)
        out_md = _repo_path(args.out_md)
        if out_json == out_md:
            raise RecordedQAReissueError("JSON/Markdown 출력 경로는 달라야 합니다")
        _require_fresh_output_paths(out_json, out_md)
        report = reissue_current_recorded_qa(
            manifest_path=args.manifest,
            data_config_path=args.data_config,
            duct_config_path=args.duct_config,
            block_frames=args.block_frames,
        )
        # QA 실패도 forensics를 위해 기록하지만 historical artifact를 덮어쓰지는 않는다.
        _write_exclusive(out_json, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        _write_exclusive(out_md, render_current_recorded_qa_markdown(report))
        verdict = "PASS" if report.get("ok") is True else "FAIL"
        print(
            f"[{verdict}] current recorded QA: "
            f"{report['summary']['valid_sessions']}/{report['summary']['sessions']} sessions"
        )
        print(f"JSON: {out_json}")
        print(f"Markdown: {out_md}")
        return 0 if report.get("ok") is True else 1
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[FAIL] recorded QA authority: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
