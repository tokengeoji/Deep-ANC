"""Read calibration coefficients without changing their physical interpretation."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RIR = ROOT / "rir.txt"
DEFAULT_METADATA = ROOT / "calibration" / "secondary_path.json"
NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fF]?")


def read_coefficients(path, array_name="S_hat"):
    """Read a named C float initializer or a plain one-column coefficient file.

    Do not scan arbitrary numbers from C source: dimensions and comments are not
    filter coefficients. Reject unparsed tokens instead of silently losing taps.
    """
    path = Path(path)
    source = path.read_text(encoding="utf-8-sig")
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    if "{" in source:
        match = re.search(r"\b" + re.escape(array_name) + r"\s*\[[^]]*\]\s*=\s*\{([^{}]*)\}\s*;", source, re.S)
        if not match:
            raise ValueError("Missing or malformed {} initializer in {}".format(array_name, path))
        source = match.group(1)
    tokens = source.replace(",", " ").split()
    if not tokens or any(NUMBER.fullmatch(token) is None for token in tokens):
        raise ValueError("Expected finite numeric FIR coefficients in {}".format(path))
    coefficients = np.asarray([float(token.rstrip("fF")) for token in tokens], dtype=np.float64)
    if not np.isfinite(coefficients).all() or not np.any(coefficients):
        raise ValueError("FIR must be finite and nonzero")
    return coefficients


def coefficient_sha256(coefficients):
    """Stable hash of ordered little-endian float64 values, independent of EOL."""
    return hashlib.sha256(np.asarray(coefficients, dtype="<f8").tobytes()).hexdigest()


def load_secondary_path(path=DEFAULT_RIR, sample_rate=16000):
    path = Path(path).resolve()
    coefficients = read_coefficients(path)
    if path == DEFAULT_RIR.resolve():
        metadata = json.loads(DEFAULT_METADATA.read_text(encoding="utf-8"))
        if sample_rate != metadata["sample_rate_hz"]:
            raise ValueError("Measured secondary path is {} Hz; resampling is not implicit".format(metadata["sample_rate_hz"]))
        if len(coefficients) != metadata["num_taps"]:
            raise ValueError("Measured secondary path tap count changed")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != metadata["source_sha256"]:
            raise ValueError("rir.txt differs from the authoritative measured source; review calibration metadata")
    elif sample_rate <= 0:
        raise ValueError("A positive sample rate must be supplied for a custom calibration")
    return coefficients


def describe_secondary_path(coefficients, sample_rate=16000):
    h = np.asarray(coefficients, dtype=np.float64)
    energy = np.cumsum(h * h)
    peak = int(np.argmax(np.abs(h)))
    retained = {}
    for taps in (128, 256, 384, len(h)):
        if taps <= len(h):
            retained[str(taps)] = float(energy[taps - 1] / energy[-1])
    return {
        "sample_rate_hz": sample_rate,
        "num_taps": len(h),
        "duration_ms": len(h) * 1000.0 / sample_rate,
        "peak_index_zero_based": peak,
        "peak_time_ms": peak * 1000.0 / sample_rate,
        "peak_value": float(h[peak]),
        "energy": float(energy[-1]),
        "dc_gain": float(h.sum()),
        "coefficient_sha256_float64_le": coefficient_sha256(h),
        "prefix_energy_fraction": retained,
        "default_processing": "all taps; original gain, sign and delay; no resampling",
        "interpretation": "Peak time is not an independently measured transport latency.",
    }
