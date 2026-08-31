#!/usr/bin/env python3
"""Stage-2 2 kHz scratch pretrain/fine-tune profile을 읽기 전용으로 검사한다.

오디오, GPU, Trainer, DataLoader, subprocess, run directory를 사용하지 않는다.
현재 기본 config의 올바른 결과는 새 P/S·manifest·criterion·checkpoint 부재에 따른
``BLOCKED``와 exit 1이다. ``--out``을 명시할 때만 결과 JSON을 O_EXCL로 기록한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.train.stage2_2khz_campaign import (  # noqa: E402
    audit_stage2_2khz_campaign,
    write_stage2_preflight_json_exclusive,
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
        default="configs/stage2_2khz_campaign.yaml",
        help="Stage-2 campaign profile YAML",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="선택적 no-replace JSON 출력 (repository 내부 상대경로)",
    )
    args = parser.parse_args(argv)
    try:
        config_path = _inside_repo_relative_path(args.config, label="config")
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Stage-2 campaign config 최상위는 mapping이어야 합니다")
        report = audit_stage2_2khz_campaign(payload, repo_root=REPO_ROOT)
        if args.out is not None:
            write_stage2_preflight_json_exclusive(
                _inside_repo_relative_path(args.out, label="out"), report
            )
    except (OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    # fine-tune artifact가 없어도 typed P/S+public corpus+criterion+pretrain external
    # contract가 모두 결속되면 scratch pretrain만 먼저 열 수 있다. 이 상태는
    # campaign 전체 READY가 아니라 의도적으로 READY_PRETRAIN이다.
    return 0 if report["status"] == "READY_PRETRAIN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
