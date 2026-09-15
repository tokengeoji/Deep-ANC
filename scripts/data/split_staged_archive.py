#!/usr/bin/env python3
"""임시 원본을 분할하거나 조각을 검증하고, 명시한 새 파일로만 복원한다.

Drive 업로드/네트워크/삭제는 하지 않는다. 검증 성공도 업로드 완료(ready)가 아니다.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.data.archive_parts import restore_archive_parts, split_staged_archive, verify_archive_parts  # noqa: E402


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(1, f"{self.prog}: 인자 오류: {message}\n")


def main(argv=None):
    parser = _Parser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staging-receipt", type=Path)
    mode.add_argument("--verify-manifest", type=Path)
    mode.add_argument("--restore-manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--archive-out", type=Path)
    parser.add_argument("--part-mib", type=int, default=96)
    args = parser.parse_args(argv)
    try:
        if os.environ.get("DEEP_ANC_CONTAINER") != "cpu" or not Path("/.dockerenv").is_file():
            raise ValueError("CPU Docker에서 실행하세요")
        if args.restore_manifest:
            if args.out is not None or args.archive_out is None:
                raise ValueError("복원에는 --out 없이 새 파일 경로 --archive-out이 필요합니다")
            report = restore_archive_parts(args.restore_manifest, args.archive_out)
        elif args.archive_out is not None:
            raise ValueError("--archive-out은 --restore-manifest와 함께만 사용합니다")
        elif args.verify_manifest:
            if args.out is not None:
                raise ValueError("검증 모드는 --out 없이 읽기만 합니다")
            report = verify_archive_parts(args.verify_manifest)
        else:
            if args.out is None:
                raise ValueError("분할에는 기존에 없는 --out 폴더가 필요합니다")
            report = split_staged_archive(args.staging_receipt, args.out, part_mib=args.part_mib)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"[실패] {exc}. 원본/기존 파일/부분 조각/복원 부분 파일은 삭제하지 않습니다.", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
