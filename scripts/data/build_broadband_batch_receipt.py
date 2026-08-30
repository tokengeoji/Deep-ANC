#!/usr/bin/env python3
"""실측 ERR exact-segment 광대역 batch PASS/BLOCKED receipt 발행.

오디오 장치는 열지 않는다. 기존 출력은 덮어쓰지 않으며 BLOCKED도 증거로 저장한 뒤
exit 2를 반환한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.broadband_batch_sampler import build_broadband_batch_receipt  # noqa: E402
from deep_anc.data.manifest import read_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--segment-seconds", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--valid-prefix-samples", type=int, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--edge-trim-seconds", type=float, default=0.5)
    parser.add_argument("--max-segments-per-session", type=int, default=64)
    args = parser.parse_args()

    manifest = Path(args.manifest).expanduser().absolute()
    output = Path(args.output).expanduser().absolute()
    fs = int(args.sample_rate)
    raw_segment = int(round(float(args.segment_seconds) * fs))
    segment = max(256, (raw_segment // 256) * 256)
    trim = int(round(float(args.edge_trim_seconds) * fs))
    receipt = build_broadband_batch_receipt(
        manifest_path=manifest,
        entries=read_manifest(manifest),
        sample_rate=fs,
        segment_samples=segment,
        batch_size=int(args.batch_size),
        valid_prefix_samples=int(args.valid_prefix_samples),
        split=str(args.split),
        edge_trim_samples=trim,
        max_segments_per_session=int(args.max_segments_per_session),
    )
    encoded = (
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(output, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    status = str(receipt["summary"]["status"])
    print(f"[{status}] {output}")
    for blocker in receipt["summary"]["blockers"]:
        print(f"  - {blocker}")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
