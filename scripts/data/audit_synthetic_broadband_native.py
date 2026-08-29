#!/usr/bin/env python3
"""Canonical public manifest의 광대역 native-Nyquist/lineage coverage를 감사한다.

오디오 파일과 장치를 열지 않는다. PASS여도 source spectrum, target-d, P/S가 아직
검증되지 않았으므로 training readiness 영수증으로 사용할 수 없다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deep_anc.data.manifest import read_manifest
from deep_anc.data.synthetic_broadband_coverage import (
    DEFAULT_FAMILY_TAGS,
    audit_synthetic_native_manifest_rows,
)
from deep_anc.dsp.control_band_contract import ControlBandContract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/manifests/canonical_v4"),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_dir = args.manifest_dir.expanduser().resolve()
    entries_by_tag = {}
    for tag in sorted({tag for tags in DEFAULT_FAMILY_TAGS.values() for tag in tags}):
        path = manifest_dir / f"{tag}.jsonl"
        if path.is_file():
            entries_by_tag[tag] = read_manifest(path)
    contract = ControlBandContract.broadband_point_control()
    payload = audit_synthetic_native_manifest_rows(
        entries_by_tag,
        contract=contract,
    )
    payload["manifest_dir"] = str(manifest_dir)
    # manifest_dir를 덧붙였으므로 stdout report는 diagnostic envelope다. 내부 evidence SHA는
    # 순수 audit payload를 봉인하며 readiness artifact로 사용하지 않는다.
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(text)
        print(f"[saved] {output}", file=sys.stderr)
    print(text, end="")
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
