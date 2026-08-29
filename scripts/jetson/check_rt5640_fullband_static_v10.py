#!/usr/bin/env python3
"""RT5640 full-octave P/S의 무음 static 계약을 검사한다.

이 명령은 ALSA/sounddevice/스피커를 전혀 열지 않는다. ``[PASS]``는 오직 config와
125 Hz--8 kHz v3 contract가 일치한다는 뜻이고, 결과 receipt의 상태는 항상 BLOCKED다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.dsp.rt5640_fullband_static_v10 import (  # noqa: E402
    DEFAULT_CONFIG_RELATIVE_PATH,
    load_rt5640_fullband_static_contract,
)


def _repository_relative_path(value: str) -> Path:
    candidate = (REPO_ROOT / value).resolve()
    try:
        candidate.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError("config는 repository 내부 상대경로여야 합니다") from error
    if candidate != (REPO_ROOT / DEFAULT_CONFIG_RELATIVE_PATH).resolve():
        raise ValueError("v10 static gate는 sealed default config만 허용합니다")
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_RELATIVE_PATH)
    args = parser.parse_args(argv)
    try:
        config = _repository_relative_path(args.config)
        receipt = load_rt5640_fullband_static_contract(config)
    except (OSError, ValueError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 2
    print("[PASS] RT5640 fullband v10 static contract only; audio was not opened.")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
