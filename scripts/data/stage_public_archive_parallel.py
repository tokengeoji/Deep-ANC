#!/usr/bin/env python3
"""공식 아카이브 하나를 CPU Docker에서 최대 네 HTTPS Range로 임시 staging한다.

명시 동의와 새 출력 폴더가 필요하다. 전체 공식 checksum 통과 후 receipt를 발급한다.
부분 파일을 보존하므로 원본 약 두 배의 여유 공간이 필요하다. 자동 재시도/삭제는 없다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.data.parallel_archive_staging import stage_public_archive_parallel  # noqa: E402


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(1, f"{self.prog}: 인자 오류: {message}\n")


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", type=Path, required=True, help="기존에 없는 독립 임시 staging 폴더")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--confirm-temporary-local-staging", action="store_true")
    args = parser.parse_args(argv)
    try:
        raw = args.catalog.read_bytes()
        catalog = json.loads(raw)
        if catalog.get("schema_version") != 1 or catalog.get("research_use") != "noncommercial_academic":
            raise ValueError("공식 source catalog v1/비상업 학업 용도 메타가 필요합니다")
        receipt = stage_public_archive_parallel(
            catalog["sources"][args.source], args.out,
            confirm_temporary_local_staging=args.confirm_temporary_local_staging,
            workers=args.workers, chunk_bytes=args.chunk_bytes, source_id=args.source,
            catalog_sha256=hashlib.sha256(raw).hexdigest(),
        )
    except Exception as exc:
        # HTTP 예외 URL/본문과 인증 정보를 출력하지 않는다.
        print(f"[실패] {type(exc).__name__}: 병렬 staging 조건/Range/크기/checksum/저장 검증 실패. "
              "부분 파일은 보존하며 자동 재시도·업로드·삭제하지 않습니다.", file=sys.stderr)
        return 1
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
