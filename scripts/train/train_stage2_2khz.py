#!/usr/bin/env python3
"""Stage-2 2 kHz 전용 scratch pretrain 실행기.

기본 campaign은 실제 새 P/S·public manifest·criterion calibration·external contract가
없으므로 GPU와 run directory 전에 실패한다. 모든 typed admission이 PASS한 뒤에만
``--run-until-step 200``, 500, 100000 순서로 같은 external-contract run을 연다.
자동 재개나 Stage-1/legacy init은 없고, 재개는 ``--resume``으로 exact Stage-2
checkpoint를 명시할 때만 허용한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.train.stage2_2khz_runner import Stage2ScratchPretrainRunner  # noqa: E402


def _repository_path(value: str, *, must_exist: bool, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label}는 repository 내부 상대경로여야 합니다")
    root = REPO_ROOT.resolve(strict=True)
    target = root / candidate
    try:
        resolved = target.resolve(strict=must_exist)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"{label}가 repository 내부의 존재하는 파일이 아닙니다") from exc
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        default="configs/stage2_2khz_campaign.yaml",
        help="Stage-2 campaign YAML (repository 상대경로)",
    )
    parser.add_argument(
        "--run-until-step",
        type=int,
        required=True,
        choices=(200, 500, 100000),
        help="허용된 canonical milestone만 실행",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="같은 external-contract run의 명시적 Stage-2 checkpoint 상대경로",
    )
    parser.add_argument(
        "--smoke-acceptance",
        default=None,
        help=(
            "100k 전용 raw-bound A100 200→resume→500 및 uninterrupted 수치등가 "
            "acceptance JSON 상대경로"
        ),
    )
    args = parser.parse_args(argv)
    try:
        campaign_path = _repository_path(
            args.campaign, must_exist=True, label="campaign"
        )
        payload = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Stage-2 campaign 최상위는 mapping이어야 합니다")
        resume = None
        if args.resume is not None:
            resume = _repository_path(args.resume, must_exist=True, label="resume")
        smoke_acceptance = None
        if args.smoke_acceptance is not None:
            smoke_acceptance = _repository_path(
                args.smoke_acceptance,
                must_exist=True,
                label="smoke acceptance",
            )
        runner = Stage2ScratchPretrainRunner(
            repository_root=REPO_ROOT,
            campaign=payload,
            run_until_step=int(args.run_until_step),
            resume=resume,
            smoke_acceptance=smoke_acceptance,
        )
        checkpoint = runner.train()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] {exc}", file=sys.stderr)
        return 2
    print(f"[PASS] Stage-2 checkpoint: {checkpoint.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
