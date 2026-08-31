#!/usr/bin/env python3
"""Stage-2 actual 200→resume→500 + independent 500 smoke를 발행한다.

이 entrypoint는 canonical 100k run directory를 절대 사용하지 않는다. 두 immutable
smoke namespace에서 실제 A100 checkpoint/telemetry를 만든 뒤, raw bytes를 다시
decode·SHA 검산한 no-replace acceptance JSON만 발행한다. acceptance는 100k init이나
checkpoint가 아니며, 다음 canonical 100k도 ``--resume`` 없이 새 scratch로 시작한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.train.stage2_2khz_runner import (  # noqa: E402
    Stage2ScratchPretrainRunner,
    issue_stage2_smoke_acceptance_no_replace,
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        default="configs/stage2_2khz_campaign.yaml",
        help="Stage-2 campaign YAML (repository 상대경로)",
    )
    return parser


def _require_shared_smoke_semantics(
    resumed: Stage2ScratchPretrainRunner,
    uninterrupted: Stage2ScratchPretrainRunner,
) -> None:
    if resumed.root != uninterrupted.root:
        raise RuntimeError("Stage-2 smoke 두 arm의 repository root가 다릅니다")
    if resumed.profile != uninterrupted.profile:
        raise RuntimeError("Stage-2 smoke 두 arm의 profile이 다릅니다")
    if resumed.a100_environment != uninterrupted.a100_environment:
        raise RuntimeError("Stage-2 smoke 두 arm의 A100 environment가 다릅니다")
    if resumed.a100_environment_sha256 != uninterrupted.a100_environment_sha256:
        raise RuntimeError("Stage-2 smoke 두 arm의 A100 environment SHA가 다릅니다")
    if resumed.execution_repository_commit_sha != uninterrupted.execution_repository_commit_sha:
        raise RuntimeError("Stage-2 smoke 두 arm의 exact repository commit이 다릅니다")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        campaign_path = _repository_path(
            args.campaign, must_exist=True, label="campaign"
        )
        campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
        if not isinstance(campaign, dict):
            raise ValueError("Stage-2 campaign 최상위는 mapping이어야 합니다")

        # arm 1: fresh 0→200. 이 checkpoint만 arm 2의 explicit resume 입력이다.
        first = Stage2ScratchPretrainRunner(
            repository_root=REPO_ROOT,
            campaign=campaign,
            run_until_step=200,
            run_label="resumed",
        )
        step_200 = first.train()

        # arm 2: 같은 resumed namespace에서 immutable step_000200.pt→500.
        resumed = Stage2ScratchPretrainRunner(
            repository_root=REPO_ROOT,
            campaign=campaign,
            run_until_step=500,
            run_label="resumed",
            resume=step_200,
        )
        step_500 = resumed.train()

        # arm 3: 별도 namespace에서 fresh uninterrupted 0→500.
        uninterrupted = Stage2ScratchPretrainRunner(
            repository_root=REPO_ROOT,
            campaign=campaign,
            run_until_step=500,
            run_label="uninterrupted",
        )
        uninterrupted_step_500 = uninterrupted.train()
        _require_shared_smoke_semantics(resumed, uninterrupted)
        if step_500.parent != resumed.run_dir or uninterrupted_step_500.parent != uninterrupted.run_dir:
            raise RuntimeError("Stage-2 smoke checkpoint가 deterministic arm directory 밖에 있습니다")

        acceptance_path, acceptance_sha = issue_stage2_smoke_acceptance_no_replace(
            REPO_ROOT,
            resumed_run_dir=resumed.run_dir,
            uninterrupted_run_dir=uninterrupted.run_dir,
            external_sha256=str(resumed.anchors["external_experiment_contract_sha256"]),
            environment=resumed.a100_environment,
            environment_sha256=resumed.a100_environment_sha256,
            profile=resumed.profile,
            resumed_expected_bindings=resumed._binding_metadata(
                include_smoke_acceptance=False
            ),
            uninterrupted_expected_bindings=uninterrupted._binding_metadata(
                include_smoke_acceptance=False
            ),
        )
    except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
        print(f"[중단] {exc}", file=sys.stderr)
        return 2

    relative_acceptance = acceptance_path.relative_to(REPO_ROOT).as_posix()
    print(f"[PASS] Stage-2 raw smoke acceptance: {relative_acceptance}")
    print(f"[PASS] Stage-2 raw smoke acceptance SHA-256: {acceptance_sha}")
    print("[다음 canonical 100k — smoke checkpoint resume 금지]")
    print(
        ".venv/bin/python scripts/train/train_stage2_2khz.py "
        f"--campaign {args.campaign} --run-until-step 100000 "
        f"--smoke-acceptance {relative_acceptance}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
