#!/usr/bin/env python3
"""외부 동기 electrical witness 요구사항을 무출력으로 확인한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.dsp.external_electrical_witness_admission_v1 import (  # noqa: E402
    DEFAULT_CONFIG_RELATIVE_PATH,
    load_external_electrical_witness_static_admission,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="외부 동기 electrical witness의 static admission만 검사합니다. 오디오를 열지 않습니다."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_RELATIVE_PATH)
    args = parser.parse_args(argv)
    try:
        receipt = load_external_electrical_witness_static_admission(
            REPO_ROOT / args.config
        )
    except (OSError, ValueError) as error:
        print(f"[실패] external electrical witness static admission: {error}", file=sys.stderr)
        return 2
    print("[PASS] external electrical witness static requirements only; audio was not opened.")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
