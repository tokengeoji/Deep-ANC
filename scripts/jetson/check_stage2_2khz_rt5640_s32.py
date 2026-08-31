#!/usr/bin/env python3
"""Stage-2 RT5640/J511 S32 P/S 준비를 무음으로 검사한다.

기본값과 유일한 동작은 dry-run이다. 이 스크립트는 장치/PCM/backend를 열지 않고
speaker output을 만들지 않으며 result file도 쓰지 않는다. PASS는 APE PCM1→PCM0,
48 kHz/256/S32_LE, Stage-2 Q15→S32 exact plan과 provenance binding만 뜻한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"


def _bootstrap_repository_import() -> None:
    """직접 실행도 현재 checkout ``src``만 import하게 한다."""

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

from deep_anc.dsp.stage2_2khz_rt5640_s32 import (  # noqa: E402
    DEFAULT_CONFIG_RELATIVE_PATH,
    build_stage2_rt5640_s32_planned_transport_provenance,
    build_stage2_rt5640_s32_signal_plan,
    load_stage2_rt5640_s32_static_contract,
    validate_stage2_rt5640_s32_planned_transport_provenance,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_RELATIVE_PATH)
    parser.add_argument("--dry-run", action="store_true", help="명시해도 동작은 기본값과 같습니다")
    parser.add_argument("--json", action="store_true", help="receipt를 stdout JSON으로 출력")
    args = parser.parse_args(argv)
    try:
        receipt = load_stage2_rt5640_s32_static_contract(
            args.config, repository_root=REPO_ROOT
        )
        plan, pcm = build_stage2_rt5640_s32_signal_plan()
        provenance = build_stage2_rt5640_s32_planned_transport_provenance(plan, pcm)
        validate_stage2_rt5640_s32_planned_transport_provenance(provenance, plan, pcm)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 2

    print("[PASS] Stage-2 RT5640/J511 S32 static plan only; audio was not opened.")
    print(
        "input=APE PCM1/S32_LE, output=APE PCM0/S32_LE, "
        f"frames={pcm.shape[0]}, callbacks={plan['expected_callbacks']}"
    )
    print(
        "audio_backend_import=0; hardware_open=0; speaker_output=0; "
        f"status={receipt['status']}"
    )
    if args.json:
        print(
            json.dumps(
                {"static_receipt": receipt, "signal_plan": plan, "planned_provenance": provenance},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
