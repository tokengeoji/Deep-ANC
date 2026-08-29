#!/usr/bin/env python3
"""RT5640 S32 20초 level-control meter recipe를 무음으로 검증한다.

이 명령은 ALSA/PCM/sounddevice/GPU를 열지 않고 result file도 만들지 않는다. PASS는
오직 Q15→S32 recipe와 125 Hz--8 kHz contract binding이 맞다는 뜻이며, receipt status는
의도적으로 BLOCKED다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.dsp.rt5640_s32_meter_v10_3 import (  # noqa: E402
    DEFAULT_CONFIG_RELATIVE_PATH_V10_3,
    load_rt5640_s32_meter_static_contract_v10_3,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_RELATIVE_PATH_V10_3)
    args = parser.parse_args(argv)
    try:
        receipt = load_rt5640_s32_meter_static_contract_v10_3(
            args.config, repository_root=REPO_ROOT
        )
    except (OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    print("[PASS] RT5640 S32 level-control static recipe only; audio was not opened.")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
