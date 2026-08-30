#!/usr/bin/env python3
"""Aperiodic fullband causal P/S의 linear-response synthetic core와 provenance loader."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from deep_anc.dsp.control_band_contract import BROADBAND_POINT_CONTROL_SUBBANDS_HZ
from deep_anc.dsp.fullband_causal_aperiodic import FS, GUARD, TARGET_BAND, build_aperiodic_plan, linear_fft_convolve

SCHEMA = "fullband_causal_aperiodic_analysis_v2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_panel_raw_provenance(receipt_path: str | Path) -> dict[str, Any]:
    """임의 SHA 문자열이 아니라 실제 bytes와 same-geometry 계보를 검증한다."""
    path = Path(receipt_path).resolve()
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != "panel_raw_same_geometry_binding_v1":
        raise ValueError("panel provenance schema가 다릅니다")
    loaded = {}
    for name in ("raw", "analysis", "hardware", "level"):
        item = receipt.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"panel {name} binding이 없습니다")
        file = Path(item.get("path", "")).resolve()
        expected = str(item.get("sha256", ""))
        if not file.is_file() or _sha(file) != expected:
            raise ValueError(f"panel {name} 실제 bytes/SHA가 다릅니다")
        loaded[name] = file
    raw_meta = json.loads(loaded["raw"].read_text())
    analysis_meta = json.loads(loaded["analysis"].read_text())
    for field in ("capture_id", "hardware_identity", "geometry_id", "level_evidence_sha256"):
        if not raw_meta.get(field) or raw_meta.get(field) != analysis_meta.get(field):
            raise ValueError(f"panel raw/analysis {field}가 same-geometry가 아닙니다")
    if float(analysis_meta.get("minimum_repeat_consistency", 0.0)) < 0.95:
        raise ValueError("panel repeat consistency가 0.95 미만입니다")
    return {"receipt": receipt, "raw": raw_meta, "analysis": analysis_meta, "files": loaded}


def _slot(plan: Mapping[str, Any], path: str, role: str) -> tuple[int, int, int]:
    burst = next(row for row in plan["layout"] if row["kind"] == f"{path}_{role}_burst")
    guard = next(row for row in plan["layout"] if row["kind"] == f"{path}_{role}_zero_guard")
    if int(guard["start_frame"]) != int(burst["stop_frame"]) or int(guard["frames"]) != GUARD:
        raise ValueError("burst 뒤 exact zero guard가 없습니다")
    return int(burst["start_frame"]), int(burst["frames"]), int(guard["frames"])


def _band_errors(reference: np.ndarray, estimate: np.ndarray) -> list[dict[str, Any]]:
    n = 1 << (max(len(reference), len(estimate)) - 1).bit_length()
    r = np.fft.rfft(reference, n)
    e = np.fft.rfft(estimate, n)
    f = np.fft.rfftfreq(n, 1 / FS)
    out = []
    for i, (lo, hi) in enumerate(BROADBAND_POINT_CONTROL_SUBBANDS_HZ):
        mask = (f >= lo) & (f <= hi if i == 6 else f < hi)
        rel = float(np.linalg.norm(e[mask] - r[mask]) / max(np.linalg.norm(r[mask]), 1e-30))
        agr = float(abs(np.vdot(r[mask], e[mask])) / max(np.linalg.norm(r[mask]) * np.linalg.norm(e[mask]), 1e-30))
        out.append({"band_hz": [lo, hi], "relative_error": rel, "agreement": agr, "passed": rel <= 0.10 and agr >= 0.995})
    return out


def validate_candidate(
    *,
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    responses: Mapping[str, np.ndarray],
    candidates: Mapping[str, Mapping[str, Any]],
    response_evidence: Mapping[str, Mapping[str, Mapping[str, Any]]],
    synthetic_fixture: bool,
) -> dict[str, Any]:
    expected, pcm = build_aperiodic_plan()
    if dict(plan) != expected or not np.array_equal(np.asarray(submitted_pcm), pcm):
        raise ValueError("v2 exact plan/submitted PCM이 다릅니다")
    paths = {}
    overall = True
    for path, channel in (("primary", 0), ("secondary", 1)):
        response = np.asarray(responses[path], np.float64)
        candidate = candidates[path]
        delay = int(candidate["integer_delay_samples"])
        taps = np.asarray(candidate["post_onset_fir"], np.float64).reshape(-1)
        if delay < 0 or delay > 4800 or taps.size not in (1024, 2048, 4096):
            raise ValueError("candidate는 integer delay + post-onset 1024/2048/4096 taps여야 합니다")
        kernel = np.concatenate((np.zeros(delay), taps))
        role_rows = []
        for role in ("fit_a", "fit_b", "holdout"):
            start, length, guard = _slot(plan, path, role)
            x = pcm[start : start + length, channel].astype(np.float64) / 32767.0
            observed = response[start : start + length + guard]
            predicted = linear_fft_convolve(x, kernel)
            predicted = np.pad(predicted, (0, max(0, observed.size - predicted.size)))[: observed.size]
            evidence = response_evidence[path][role]
            snr = list(evidence.get("subband_snr_db", []))
            coherence = list(evidence.get("subband_coherence", []))
            evidence_ok = len(snr) == len(coherence) == 7 and min(snr) >= 20.0 and min(coherence) >= 0.95
            if not synthetic_fixture:
                sha = str(evidence.get("raw_arrays_sha256", ""))
                evidence_ok &= len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)
            bands = _band_errors(observed, predicted)
            guard_residual = observed[length:] - predicted[length:]
            active_rms = float(np.sqrt(np.mean(observed[:length] ** 2)))
            guard_ratio = float(np.sqrt(np.mean(guard_residual**2)) / max(active_rms, 1e-30))
            guard_l1_ratio = float(
                np.sum(np.abs(guard_residual))
                / max(np.sum(np.abs(observed[:length])), 1e-30)
            )
            guard_peak_ratio = float(
                np.max(np.abs(guard_residual)) / max(active_rms, 1e-30)
            )
            final = guard_residual[-(GUARD // 4) :]
            noise_floor = float(evidence.get("input_only_noise_rms", 0.0))
            final_rms = float(np.sqrt(np.mean(final**2)))
            final_ok = final_rms <= max(noise_floor * 10 ** (1 / 20), 1e-12)
            passed = (
                evidence_ok
                and all(row["passed"] for row in bands)
                and guard_ratio <= 0.03
                and guard_l1_ratio <= 0.03
                and guard_peak_ratio <= 0.03
                and final_ok
            )
            role_rows.append({"role": role, "subbands": bands, "guard_residual_rms_ratio": guard_ratio, "guard_residual_l1_ratio": guard_l1_ratio, "guard_induced_peak_ratio": guard_peak_ratio, "full_guard_samples": guard, "final_noise_floor_window_samples": GUARD // 4, "final_guard_rms": final_rms, "evidence_passed": evidence_ok, "passed": passed})
        path_pass = all(row["passed"] for row in role_rows)
        overall &= path_pass
        paths[path] = {"integer_delay_samples": delay, "post_onset_support_samples": taps.size, "artifact_semantics": "integer_delay_plus_post_onset_fir_only", "full_delayed_kernel_role": "diagnostic_only_not_serialized_as_canonical", "roles": role_rows, "passed": path_pass}
    return {"schema": SCHEMA, "status": "PASS" if overall else "BLOCKED", "synthetic_fixture_only": synthetic_fixture, "canonical_training_eligible": False, "canonical_blocker": "production_raw_timewarp_and_panel_byte_binding_not_published", "paths": paths}
