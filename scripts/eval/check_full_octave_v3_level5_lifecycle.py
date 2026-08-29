#!/usr/bin/env python3
"""Full-octave v3 Level-5 lifecycle을 읽기 전용으로 검사한다.

이 명령은 ALSA/sounddevice/GPU/network/subprocess를 열지 않고 capability, receipt,
capture, evaluation 결과를 생성하지 않는다. self-attested config/receipt는 모두
``BLOCKED_UNATTESTED_*``이다. 이 lifecycle에는 independent evaluator trust root가 없으므로
성공 종료 코드는 없다. 훗날 opaque verified receipt를 직접 검증하는 별도 evaluator만
그 자체의 CLI에서 success exit을 가질 수 있다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.eval.full_octave_v3_level5 import (  # noqa: E402
    DEFAULT_CONFIG_RELATIVE_PATH,
    audit_full_octave_v3_level5_lifecycle,
    render_full_octave_v3_level5_markdown,
)


def _inside_repository(value: str, *, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label}는 repository 내부 상대경로여야 합니다")
    target = REPO_ROOT / candidate
    try:
        target.resolve(strict=False).relative_to(REPO_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label}가 repository 밖을 가리킵니다") from exc
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_RELATIVE_PATH)
    parser.add_argument("--markdown", action="store_true", help="JSON 대신 markdown summary 출력")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="동일한 read-only 검사임을 명시합니다; capture/output/evaluation은 시작하지 않습니다.",
    )
    args = parser.parse_args(argv)
    try:
        config_path = _inside_repository(args.config, label="config")
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Level-5 lifecycle config 최상위는 mapping이어야 합니다")
        report = audit_full_octave_v3_level5_lifecycle(payload, repo_root=REPO_ROOT)
    except (OSError, ValueError) as exc:
        print(f"[FAIL] full-octave v3 Level-5 lifecycle: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        report["dry_run"] = True
    if args.markdown:
        print(render_full_octave_v3_level5_markdown(report), end="")
    else:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    # config/manifest/terminal JSON의 자기 선언은 어떤 경우에도 성공이 아니다. independent
    # evaluator trust root가 이 lifecycle 밖에 있으므로 이 checker는 항상 nonzero다.
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
