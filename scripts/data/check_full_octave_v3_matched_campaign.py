#!/usr/bin/env python3
"""full-octave v3 physical matched OFF/DL/FxLMS campaign을 읽기만 감사한다.

ALSA, sounddevice, GPU, network를 열지 않고 capture/출력/평가 결과 쓰기를 하지 않는다.
기본 static/null config의 정상 상태는 ``BLOCKED``이다. 이 CLI는 선언 SHA 구조가
완전해도 physical matched pass를 발행하지 않으므로 canonical physical pass가 생길
때까지 항상 nonzero를 반환한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.full_octave_v3_matched_campaign import (  # noqa: E402
    DEFAULT_CONFIG_RELATIVE_PATH,
    load_full_octave_v3_matched_campaign,
)


def _inside_repository(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("config는 repository 내부 상대경로여야 합니다")
    target = REPO_ROOT / candidate
    try:
        target.resolve(strict=False).relative_to(REPO_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("config가 repository 밖을 가리킵니다") from exc
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_RELATIVE_PATH)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="동일한 read-only 검증임을 명시합니다; capture/output/ANC 계산을 시작하지 않습니다.",
    )
    args = parser.parse_args(argv)
    try:
        report = load_full_octave_v3_matched_campaign(
            _inside_repository(args.config), repository_root=REPO_ROOT
        )
    except (OSError, ValueError) as exc:
        print(f"[FAIL] full-octave v3 matched campaign: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        report["dry_run"] = True
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    # checksum/field 구조 검사와 actual physical pass를 절대 혼동하지 않는다. 현재
    # audit module은 self-attested plan/receipt/raw만 읽으므로 exit 0을 발행하지 않는다.
    return 0 if report.get("canonical_matched_physical_pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
