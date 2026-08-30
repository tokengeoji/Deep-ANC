#!/usr/bin/env python3
"""공개 raw corpus의 decoder eligibility를 전체 decode로 감사한다.

예시 (Elice의 immutable raw corpus를 repository-relative inventory로 기록):

    .venv/bin/python scripts/data/audit_decoder_eligibility.py \
      --root . --scan-root data/raw \
      --out results/provenance/decoder_audit.json --allow-rejections

``--dry-run``은 파일 발견 순서와 정책만 출력하며 decoder를 열거나 output을 쓰지
않는다. 실제 audit은 reject가 하나라도 있으면 report를 쓴 뒤 종료코드 2를 내므로,
다음 canonical manifest 단계가 실수로 계속되지 않는다. forensic/rebuild 작업에서
rejected row도 필요할 때만 ``--allow-rejections``을 명시한다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.decoder_audit import (  # noqa: E402
    DEFAULT_SEGMENT_FRAMES,
    DEFAULT_SEQUENTIAL_CHUNK_FRAMES,
    MAX_DECODED_PCM_ABS,
    MIN_DECODED_RMS,
    audit_audio_paths,
    canonical_json_bytes,
    discover_audio_files,
    write_audit_report,
)


def _path_under_root(value: str, *, root: Path, field: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    absolute = Path(os.path.abspath(candidate))
    try:
        absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field}는 --root 아래여야 합니다: {value}") from exc
    return absolute


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=".",
        help="inventory relative_path의 기준 directory (기본: repository root)",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        default=None,
        help="--root 아래에서 재귀 탐색할 raw directory/file. 반복 가능 (기본: data/raw)",
    )
    parser.add_argument(
        "--out",
        default="results/provenance/decoder_audit.json",
        help="canonical JSON output path (--root 기준, raw tree와 분리 권장)",
    )
    parser.add_argument(
        "--sequential-chunk-frames",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEQUENTIAL_CHUNK_FRAMES),
        help="필수 full sequential chunk size들 (65536, 262144를 반드시 포함)",
    )
    parser.add_argument(
        "--segment-frames",
        type=int,
        default=DEFAULT_SEGMENT_FRAMES,
        help="deterministic seek grid의 각 segment 최대 frame 수",
    )
    parser.add_argument(
        "--max-decoded-pcm-abs",
        type=float,
        default=MAX_DECODED_PCM_ABS,
    )
    parser.add_argument(
        "--min-decoded-rms",
        type=float,
        default=MIN_DECODED_RMS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="발견한 후보와 정책만 stdout에 출력하고 decode/output write를 하지 않음",
    )
    parser.add_argument(
        "--allow-rejections",
        action="store_true",
        help="reject가 있어도 report 작성 후 0으로 종료 (forensic/rebuild 전용)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root_candidate = Path(args.root).expanduser()
    if not root_candidate.is_absolute():
        root_candidate = REPO_ROOT / root_candidate
    root = Path(os.path.abspath(root_candidate))
    if not root.is_dir():
        print(f"[실패] --root directory가 없습니다: {root}", file=sys.stderr)
        return 1
    values = args.scan_root if args.scan_root is not None else ["data/raw"]
    try:
        scan_roots = [
            _path_under_root(value, root=root, field="--scan-root") for value in values
        ]
        paths = discover_audio_files(scan_roots)
        relative_paths = [
            Path(os.path.abspath(path)).relative_to(root).as_posix() for path in paths
        ]
        out = _path_under_root(args.out, root=root, field="--out")
    except (FileNotFoundError, ValueError) as exc:
        print(f"[실패] {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        payload = {
            "status": "dry_run",
            "root_label": args.root,
            "candidate_count": len(relative_paths),
            "relative_paths": relative_paths,
            "audit_policy": {
                "sequential_chunk_frames": sorted(args.sequential_chunk_frames),
                "segment_frames": args.segment_frames,
                "max_decoded_pcm_abs": args.max_decoded_pcm_abs,
                "min_decoded_rms": args.min_decoded_rms,
            },
        }
        print(canonical_json_bytes(payload).decode("utf-8"))
        return 0

    try:
        report = audit_audio_paths(
            paths,
            root=root,
            root_label=args.root,
            sequential_chunk_frames=args.sequential_chunk_frames,
            segment_frames=args.segment_frames,
            max_decoded_pcm_abs=args.max_decoded_pcm_abs,
            min_decoded_rms=args.min_decoded_rms,
        )
        write_audit_report(report, out)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"[실패] decoder audit을 완료하지 못했습니다: {exc}", file=sys.stderr)
        return 1

    summary = report["summary"]
    print(
        "[완료] decoder audit "
        f"candidate={summary['candidate_count']} "
        f"accept={summary['accepted_count']} reject={summary['rejected_count']}\n"
        f"report={out}\n"
        f"inventory_sha256={report['inventory_sha256']}\n"
        f"accepted_inventory_sha256={report['accepted_inventory_sha256']}"
    )
    if summary["rejected_count"] and not args.allow_rejections:
        print(
            "[차단] reject row가 있습니다. report를 manifest rebuild 입력으로 검토하거나 "
            "forensic 실행이면 --allow-rejections을 명시하세요.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
