#!/usr/bin/env python3
"""Fresh meter raw/receipt를 공용 recording-level campaign으로 발행한다.

오디오 장치를 열지 않고 이미 종료된 meter artifact만 읽는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.recording_level_campaign import (  # noqa: E402
    RecordingLevelCampaignError,
    build_recording_level_campaign_payload,
    campaign_receipt_relative_path,
    issue_recording_level_campaign,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meter-raw", required=True)
    parser.add_argument("--meter-receipt", required=True)
    parser.add_argument(
        "--hardware-config",
        default="configs/hardware_jetson.yaml",
        help="meter hardware_identity를 재유도할 YAML",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="검증된 campaign을 고정 results/ 경로에 no-replace 발행",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.write:
            summary = issue_recording_level_campaign(
                repo_root=REPO_ROOT,
                meter_raw=args.meter_raw,
                meter_receipt=args.meter_receipt,
                hardware_config=args.hardware_config,
            )
            output = {
                "campaign_id": summary["campaign_id"],
                "receipt_path": summary["receipt_path"],
                "receipt_size": summary["receipt_size"],
                "receipt_sha256": summary["receipt_sha256"],
            }
            print("[PASS] recording-level campaign no-replace 발행 완료")
        else:
            payload = build_recording_level_campaign_payload(
                repo_root=REPO_ROOT,
                meter_raw=args.meter_raw,
                meter_receipt=args.meter_receipt,
                hardware_config=args.hardware_config,
            )
            output = {
                "campaign_id": payload["campaign_id"],
                "intended_receipt_path": campaign_receipt_relative_path(
                    payload["campaign_id"]
                ),
                "meter": payload["meter"],
                "hardware": payload["hardware"],
                "evidence_sha256": payload["evidence_sha256"],
            }
            print("[PASS] recording-level campaign check-only; 파일을 쓰지 않음")
        print(
            json.dumps(
                output,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    except (FileNotFoundError, OSError, RuntimeError, RecordingLevelCampaignError) as exc:
        print(f"[BLOCKED] recording-level campaign: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
