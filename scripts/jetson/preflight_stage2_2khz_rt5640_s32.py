#!/usr/bin/env python3
"""RT5640/J511 Stage-2 S32 actual P/S 전의 읽기 전용 preflight.

기본 동작은 장치/PCM/audio backend를 열지 않는 dry-run이다. J511 ``HP`` 또는
``HS``가 3회 동일하게 감지되고, 전역 PCM stream이 모두 closed이며, APE mux와
sealed actual-P/S same-card S32 config의 config/provenance SHA가 모두 맞아야 PASS한다.
USB AB13X output-master/fallback/S16 및 구형 fallback static receipt는 명시적으로
거절한다. 이 스크립트는 결과 파일을 쓰거나 어떤 소리도 내지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"


def _bootstrap_repository_import() -> None:
    """직접 실행도 현재 checkout의 src만 import한다."""

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

from deep_anc.dsp.rt5640_stage2_s32_preflight import (  # noqa: E402
    PASS_STATUS,
    collect_rt5640_stage2_s32_preflight,
    receipt_to_jsonable,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="기본값과 같은 read-only 점검")
    parser.add_argument("--json", action="store_true", help="immutable in-memory receipt를 stdout JSON으로 출력")
    args = parser.parse_args(argv)

    # 이 행은 preflight PASS/FAIL와 무관하게 실제 PCM/backend/output write가 없음을
    # 명시한다. 이후 P/S adapter는 이 CLI PASS만으로 live stream을 열 수 없다.
    print("audio_backend_import=0; ALSA_PCM_open=0; speaker_output=0; raw_write=0")
    receipt = collect_rt5640_stage2_s32_preflight()
    if args.json:
        print(json.dumps(receipt_to_jsonable(receipt), ensure_ascii=False, sort_keys=True))
    if receipt["status"] == PASS_STATUS:
        print("[PASS] RT5640/J511 Stage-2 S32 read-only preflight; audio was not opened.")
        return 0
    print(
        f"[BLOCKED] {receipt['status']}; "
        f"J511={list(receipt['j511']['observed_states'])}; "
        f"errors={list(receipt['errors'])}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
