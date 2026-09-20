#!/usr/bin/env python3
"""Validate the measured source and export reproducible training artifacts."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepanc.calibration import DEFAULT_RIR, describe_secondary_path, load_secondary_path, read_coefficients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_RIR)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "secondary_path")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    h = load_secondary_path(args.source, sample_rate=args.sample_rate)
    report = describe_secondary_path(h, args.sample_rate)
    report["source_sha256"] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if args.source.resolve() == DEFAULT_RIR.resolve():
        dsp = ROOT / "firmware" / "omap_l138" / "FxNLMS0" / "ISR.c"
        if not np.array_equal(h, read_coefficients(dsp, "S_hat")):
            raise ValueError("rir.txt and successful FxNLMS S_hat disagree")
        report["matches_successful_dsp"] = True
    if not args.verify_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.output_dir / "secondary_path.npy", h.astype(np.float32), allow_pickle=False)
        np.savetxt(args.output_dir / "secondary_path.csv", np.column_stack((np.arange(len(h)), h)),
                   delimiter=",", header="sample,coefficient", comments="", fmt=["%d", "%.8f"])
        header = "/* Generated from rir.txt; preserve gain, sign and delay. */\n#ifndef DEEPANC_SECONDARY_PATH_H\n#define DEEPANC_SECONDARY_PATH_H\n"
        header += "#define DEEPANC_SECONDARY_SAMPLE_RATE {}\n#define DEEPANC_SECONDARY_TAPS {}\n".format(args.sample_rate, len(h))
        header += "static const float deepanc_secondary_path[DEEPANC_SECONDARY_TAPS] = {\n"
        header += "\n".join("    " + ", ".join("{:.8f}f".format(v) for v in h[i:i + 8]) + "," for i in range(0, len(h), 8))
        header += "\n};\n#endif\n"
        (args.output_dir / "secondary_path.h").write_text(header, encoding="utf-8")
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
