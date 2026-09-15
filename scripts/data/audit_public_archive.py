#!/usr/bin/env python3
"""검증된 임시 archive의 무추출 PCM QA. 다운로드/Drive/삭제/학습 실행 없음.

성공 exit 0은 archive 내용 QA만 의미한다. metadata-only archive는 음원 READY가
아니다. 항목 오류는 inventory.jsonl/qa.json에 보존하고 exit 1로 끝낸다.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from deep_anc.data.archive_audit import audit_public_archive  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--staging-receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="새 QA 결과 폴더")
    parser.add_argument("--max-entry-mib", type=int, default=64, help="음원별 메모리 상한 1..256MiB; metadata는 chunk 읽기")
    args = parser.parse_args(argv)
    try:
        report = audit_public_archive(args.archive, args.staging_receipt, args.out, max_entry_mib=args.max_entry_mib)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"archive QA 거부: {exc}", file=sys.stderr)
        return 1
    print(f"{report['source_archive_content_qa_status']}: {args.out} (학습/실기 READY 아님)")
    return 0 if report["source_archive_content_qa_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
