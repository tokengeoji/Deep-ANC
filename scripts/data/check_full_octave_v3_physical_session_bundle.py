#!/usr/bin/env python3
"""full-octave v3 8-input physical raw bundle을 읽기만 감사한다.

이 스크립트는 ALSA/sounddevice/GPU/network를 열지 않고 파일을 생성·수정하지 않는다.
기본 static/null config의 정상 종료 상태는 ``BLOCKED``이다. complete-looking raw bundle도
trusted capture adapter/typed provenance가 없으면
``BLOCKED_UNATTESTED_STRUCTURAL_RAW``이며 이 CLI에는 success exit이 없다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.full_octave_v3_physical_bundle import (  # noqa: E402
    DEFAULT_CONFIG_RELATIVE_PATH,
    load_full_octave_v3_physical_session_bundle,
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
        help="동일한 read-only 검증임을 명시합니다; 어떠한 capture/output도 시작하지 않습니다.",
    )
    args = parser.parse_args(argv)
    try:
        report = load_full_octave_v3_physical_session_bundle(
            _inside_repository(args.config), repository_root=REPO_ROOT
        )
    except (OSError, ValueError) as exc:
        print(f"[FAIL] full-octave v3 physical session bundle: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        report["dry_run"] = True
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    # 선언 SHA 구조 검사는 physical raw provenance가 아니다. future trusted adapter
    # validator가 별도 authority로 추가되기 전까지 이 checker는 항상 nonzero다.
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
