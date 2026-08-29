#!/usr/bin/env python3
"""Full-octave v3 raw-bound execution envelope를 읽기 전용으로 검사한다.

이 명령은 audio/ALSA/sounddevice, GPU, Trainer, DataLoader, subprocess, run
directory를 사용하지 않는다. 현재 기본 null config의 정상 종료 상태는 ``BLOCKED``다.
형식상 complete한 non-fixture JSON chain도
``BLOCKED_UNATTESTED_EXECUTION_PROVENANCE``이며, 이 CLI에는 success exit이 없다.
``--out``은 명시했을 때만 새 JSON 파일을 O_EXCL로 기록한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.train.full_octave_v3_execution import (  # noqa: E402
    audit_full_octave_v3_execution,
    render_full_octave_v3_execution_markdown,
    write_execution_preflight_json_exclusive,
)


def _inside_repo_relative_path(value: str, *, label: str) -> Path:
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
    parser.add_argument(
        "--config",
        default="configs/full_octave_v3_execution.yaml",
        help="raw-bound execution preflight YAML",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="선택적 no-replace JSON 출력 (repository 내부 상대경로)",
    )
    parser.add_argument("--markdown", action="store_true", help="JSON 대신 markdown summary 출력")
    args = parser.parse_args(argv)
    try:
        config_path = _inside_repo_relative_path(args.config, label="config")
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("execution config 최상위는 mapping이어야 합니다")
        report = audit_full_octave_v3_execution(loaded, repo_root=REPO_ROOT)
        if args.out is not None:
            output = _inside_repo_relative_path(args.out, label="out")
            write_execution_preflight_json_exclusive(output, report)
    except (OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    if args.markdown:
        print(render_full_octave_v3_execution_markdown(report), end="")
    else:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    # 이 checker가 볼 수 있는 것은 self-attested declared SHA structure뿐이다.
    # trusted live publisher/typed validator가 없는 동안에는 어떤 report도 canonical
    # training authority가 아니므로 success exit을 발행하지 않는다.
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
