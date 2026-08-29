#!/usr/bin/env python3
"""광대역 source contract v2 draft 감사/issued receipt 발행기.

오디오 장치와 원격 저장소를 사용하지 않는다. 현재 production issuer authority는 명시적으로
``None``이므로 ``--issue``는 fail-closed다. 향후 root review로 authority가 열린 뒤에도 48개
local bytes, physical fullband causal P, 실제 9x7 재계산과 EQ git ancestry를 전부 재검증한
acquisition-input receipt만 no-replace 발행하며, live source plan authority는 만들지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_anc.data.broadband_source_contract_v2 import (  # noqa: E402
    SourceContractV2Blocked,
    issue_source_manifest_v2_noreplace,
    source_contract_v2,
    validate_source_manifest_v2,
)


def _json_no_duplicates(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"JSON duplicate key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 최상위가 object가 아닙니다: {path}")
    return value


def _repo_file(value: str) -> Path:
    root = Path(os.path.abspath(ROOT))
    path = Path(value).expanduser()
    candidate = Path(
        os.path.abspath(path if path.is_absolute() else root / path)
    )
    relative = candidate.relative_to(root)
    cursor = root
    if cursor.is_symlink():
        raise ValueError(f"repository root가 symlink입니다: {cursor}")
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise ValueError(f"repository 경로에 symlink가 있습니다: {cursor}")
    if not candidate.is_file():
        raise ValueError(f"repository regular file이 아닙니다: {candidate}")
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-contract", action="store_true")
    parser.add_argument("--campaign")
    parser.add_argument("--draft")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--issue", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_contract:
        if any((args.campaign, args.draft, args.audit_only, args.issue, args.output)):
            raise SystemExit("--print-contract는 다른 인자와 함께 쓸 수 없습니다")
        print(json.dumps(source_contract_v2(), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if not args.campaign or not args.draft or args.audit_only == args.issue:
        raise SystemExit("--campaign/--draft와 --audit-only 또는 --issue 중 정확히 하나가 필요합니다")
    campaign = _json_no_duplicates(_repo_file(args.campaign))
    draft = _json_no_duplicates(_repo_file(args.draft))
    if args.audit_only:
        if args.output:
            raise SystemExit("--audit-only는 output을 쓰지 않습니다")
        result = validate_source_manifest_v2(draft, campaign=campaign)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if result["actual_acquisition_pass"] else 2
    if not args.output:
        raise SystemExit("--issue에는 --output이 필요합니다")
    try:
        result = issue_source_manifest_v2_noreplace(
            draft,
            campaign=campaign,
            repository_root=ROOT,
            output_path=args.output,
        )
    except SourceContractV2Blocked as exc:
        print(f"[BLOCKED] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
