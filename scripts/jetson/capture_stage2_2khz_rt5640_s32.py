#!/usr/bin/env python3
"""Stage-2 RT5640/J511 S32 capture adapter scaffold의 무음 검사.

기본값은 dry-run이며 ALSA/PCM/audio backend를 열지 않고, speaker 출력이나 raw 파일
쓰기를 하지 않는다. ``--execute-live``도 현재 signal-only plan에서는 backend import
전에 fail-closed한다. 실제 P/S 출력은 별도 actual P/S authority와 J511 물리 확인을
구현한 뒤에만 새 commit의 전용 adapter에서 가능하다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"


def _bootstrap_repository_import() -> None:
    """직접 실행도 현재 checkout의 ``src``만 import하게 한다."""

    imported = sys.modules.get("deep_anc")
    if imported is not None:
        location = getattr(imported, "__file__", None)
        if location is None:
            raise RuntimeError("preloaded deep_anc module의 source 위치가 없습니다")
        try:
            Path(str(location)).resolve().relative_to(SOURCE_ROOT.resolve())
        except ValueError as error:
            raise RuntimeError("foreign preloaded deep_anc module을 거부합니다") from error
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))


_bootstrap_repository_import()

from deep_anc.dsp.stage2_2khz_rt5640_s32_capture import (  # noqa: E402
    Stage2Rt5640S32CaptureBlocked,
    assert_stage2_rt5640_s32_live_capture_blocked,
    build_stage2_rt5640_s32_capture_dry_run_receipt,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="기본값과 같은 무음 static 검사")
    mode.add_argument(
        "--execute-live",
        action="store_true",
        help="현재는 backend import/open 전 fail-closed 되는 예약된 entry point",
    )
    parser.add_argument("--json", action="store_true", help="dry-run receipt를 stdout JSON으로 출력")
    args = parser.parse_args(argv)

    try:
        if args.execute_live:
            assert_stage2_rt5640_s32_live_capture_blocked()
        receipt = build_stage2_rt5640_s32_capture_dry_run_receipt()
    except Stage2Rt5640S32CaptureBlocked as error:
        print(f"[BLOCKED_BEFORE_AUDIO] {error}", file=sys.stderr)
        print("audio_backend_import=0; ALSA_PCM_open=0; speaker_output=0; raw_write=0", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 2

    print("[PASS] Stage-2 RT5640/J511 S32 capture scaffold dry-run; audio was not opened.")
    print(
        "input=APE PCM1/S32_LE, output=APE PCM0/S32_LE, "
        "sample_rate=48000, block_size=256"
    )
    print(
        "audio_backend_import=0; ALSA_PCM_open=0; speaker_output=0; raw_write=0; "
        f"status={receipt['status']}"
    )
    if args.json:
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
