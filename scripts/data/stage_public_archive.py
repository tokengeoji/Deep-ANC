#!/usr/bin/env python3
"""공식 catalog 아카이브 하나를 CPU Docker의 새 임시 폴더에 staging한다.

사용자 입회 오디오/추출/Drive 인증·업로드/삭제는 하지 않는다. 최신 사용자 승인을
--confirm-temporary-local-staging으로 명시해야 GET을 실행한다. 실패 부분 파일은 보존한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.data.archive_staging import stage_public_archive  # noqa: E402


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(1, f"{self.prog}: 인자 오류: {message}\n")


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", type=Path, required=True, help="기존에 없는 임시 staging 폴더")
    parser.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--confirm-temporary-local-staging", action="store_true")
    args = parser.parse_args(argv)
    try:
        raw = args.catalog.read_bytes()
        catalog = json.loads(raw)
        if catalog.get("schema_version") != 1 or catalog.get("research_use") != "noncommercial_academic":
            raise ValueError("공식 source catalog v1/비상업 학업 용도 메타가 필요합니다")
        receipt = stage_public_archive(
            catalog["sources"][args.source], args.out,
            confirm_temporary_local_staging=args.confirm_temporary_local_staging,
            chunk_bytes=args.chunk_bytes, source_id=args.source,
            catalog_sha256=hashlib.sha256(raw).hexdigest(),
        )
    except Exception as exc:
        # 외부 HTTP 예외 URL/본문을 출력하지 않는다. 성공 receipt와 실패 로그를 분리한다.
        print(f"[실패] {type(exc).__name__}: staging 조건/원본 GET/크기/checksum/저장 검증 실패. "
              "부분 파일은 보존하며 자동 재시도·업로드·삭제하지 않습니다.", file=sys.stderr)
        return 1
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
