#!/usr/bin/env python3
"""v4 continuous-pilot causal P/S signal-only plan (audio authority locked)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.dsp.fullband_causal_v4 import (  # noqa: E402
    CANONICAL_BLOCKER,
    LIVE_AUTHORITY,
    build_plan,
)


def _lexical_repository_target(target: Path, repository_root: Path) -> Path:
    """resolve() 없이 lexical containment와 모든 parent symlink를 검사한다."""

    root = Path(os.path.abspath(os.fspath(repository_root.expanduser())))
    lexical = Path(os.path.abspath(os.fspath(target.expanduser())))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise ValueError("signal plan은 저장소 안에만 저장합니다") from error
    cursor = root
    if cursor.is_symlink():
        raise ValueError("repository root symlink에서는 authority plan을 쓰지 않습니다")
    for part in relative.parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"output parent symlink를 거부합니다: {cursor}")
    if lexical.is_symlink():
        raise ValueError(f"output target symlink를 거부합니다: {lexical}")
    return lexical


def _write_plan_no_replace(
    *, plan: dict, target: Path, repository_root: Path
) -> Path:
    lexical = _lexical_repository_target(target, repository_root)
    lexical.parent.mkdir(parents=True, exist_ok=True)
    # mkdir와 open 사이에 생긴 parent symlink도 한 번 더 검사한다.
    lexical = _lexical_repository_target(lexical, repository_root)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(os.fspath(lexical), flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as handle:
            json.dump(plan, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    directory_fd = os.open(os.fspath(lexical.parent), directory_flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return lexical


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-live", action="store_true")
    parser.add_argument(
        "--excitation-lower-hz",
        type=float,
        default=None,
        help="future contract 검토용 PE 하단. 현 v2 authority를 확장하지 않습니다",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    # 이 분기는 plan 생성, audio backend import, 장치 검사/열기보다 먼저 실행된다.
    if args.execute_live:
        assert LIVE_AUTHORITY is None
        print(
            "[중단] v4 live authority=None; exact plan/raw publisher 검토가 끝나지 않았습니다",
            file=sys.stderr,
        )
        print(f"[차단] {CANONICAL_BLOCKER}", file=sys.stderr)
        return 2

    plan, _ = build_plan(excitation_lower_hz=args.excitation_lower_hz)
    if args.output is not None:
        try:
            _write_plan_no_replace(
                plan=plan,
                target=args.output,
                repository_root=REPO_ROOT,
            )
        except (FileExistsError, OSError, ValueError) as error:
            print(f"[실패] signal plan no-replace 저장 거부: {error}", file=sys.stderr)
            return 2

    print(
        "[PASS] v4 continuous-pilot signal-only | "
        f"{plan['output']['duration_seconds']:.3f}s | "
        f"peak PCM {plan['output']['peak_pcm']} | "
        f"PCM SHA {plan['output']['pcm_sha256']}"
    )
    print("[잠금] 오디오 출력 0회; live authority=None; canonical training=False")
    print(f"[차단] {plan['canonical_blocker']}")
    octave_blocker = plan["plant_identification"].get(
        "125hz_octave_contract_blocker"
    )
    if octave_blocker is not None:
        print(f"[차단] {octave_blocker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
