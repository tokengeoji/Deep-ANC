#!/usr/bin/env python3
"""old82 train-only ERR gain을 strict primary 단위에 맞춘 receipt를 만든다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.recorded_level_calibration import (  # noqa: E402
    RecordedLevelCalibrationError,
    build_recorded_level_calibration_payload,
    canonical_recorded_level_calibration_output,
    require_clean_exact_commit,
    validate_recorded_level_calibration_receipt,
    write_recorded_level_calibration_receipt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recorded-manifest",
        default="data/manifests/recorded_regrouped.jsonl",
    )
    parser.add_argument(
        "--strict-primary-npz",
        default="assets/measured/primary_path_il_strict_5dc06fdd.npz",
    )
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source_commit = require_clean_exact_commit(REPO_ROOT)
        payload = build_recorded_level_calibration_payload(
            repo_root=REPO_ROOT,
            recorded_manifest=args.recorded_manifest,
            strict_primary_npz=args.strict_primary_npz,
            source_commit=source_commit,
        )
        output = canonical_recorded_level_calibration_output(REPO_ROOT, args.out)
        path, digest = write_recorded_level_calibration_receipt(payload, output)
        validate_recorded_level_calibration_receipt(
            path,
            expected_sha256=digest,
            repo_root=REPO_ROOT,
            verify_bound_audio=True,
        )
    except (OSError, RecordedLevelCalibrationError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print(f"[PASS] receipt={path.relative_to(REPO_ROOT)} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
