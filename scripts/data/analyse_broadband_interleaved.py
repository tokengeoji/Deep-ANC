#!/usr/bin/env python3
"""광대역 multi-panel P/S raw를 분석·발행하는 오프라인 경로.

저장된 broadband raw-first session과 exact signal plan만 읽는다. panel별
joint-LS/cubic, 공통 P/S 반복 선택, panel 사이 shared phase stitch, 단일
broadband FIR fit을 순서대로 수행한다. raw·plan·level·hardware·meter
SHA와 실제 제출 int16 PCM을 다시 대조하고 ``BroadbandPlantEvidence`` audit가
PASS인 경우에만 분석/P/S 산출물을 no-replace로 발행한다.

이 파일은 ``sounddevice``를 import하지 않고 오디오 장치를 절대 열지 않는다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import math
import os
import stat
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import correlate

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.dsp.broadband_interleaved import (  # noqa: E402
    BROADBAND_CLOCK_PILOT_BAND_HZ,
    validate_submitted_pilot_cross_channel_null,
)
from deep_anc.dsp.broadband_plant_analysis import (  # noqa: E402
    apply_shared_panel_delay,
    band_roundtrip_metrics,
    compact_fir_identifiability_receipt,
    estimate_bulk_delay_samples,
    fit_real_compact_fir,
    fit_shared_panel_delay,
    merge_stitched_panels,
)
from deep_anc.dsp.control_band_contract import (  # noqa: E402
    BROADBAND_GLOBAL_CLOCK_MAX_RESIDUAL_SAMPLES,
    BroadbandPlantEvidence,
    ControlBandContract,
    audit_broadband_plant_evidence,
    max_timing_error_samples_for_attenuation,
)
from deep_anc.dsp.measured_band_path import (  # noqa: E402
    build_every_other_tone_holdout_receipt,
)
from deep_anc.audio_io import (  # noqa: E402
    analyze_int32_input_probe,
    pcm_int32_to_float32,
)
from deep_anc.config import load_yaml  # noqa: E402
from deep_anc.dsp.measurement_level import (  # noqa: E402
    BOOTSTRAP_METER_MAX_AGE_SECONDS,
    atomic_publish_noreplace,
    load_measurement_level_evidence,
    validate_bootstrap_meter_raw,
    validate_measurement_hardware_contract,
)
from deep_anc.dsp.interleaved_probe import (  # noqa: E402
    align_repeats,
    complex_consistency,
)
from deep_anc.dsp.timing import PlantDelays  # noqa: E402
from scripts.data import measure_paths_interleaved as mpi  # noqa: E402
from scripts.data import measure_paths_broadband_interleaved as broadband_measure  # noqa: E402


BROADBAND_ANALYSIS_SCHEMA = "broadband_interleaved_analysis_v2_global_clock"
BROADBAND_PLANT_ARTIFACT_SCHEMA = "broadband_measured_band_plant_v2_raw_derived"
BROADBAND_ANALYSIS_METHOD = (
    "fractional_joint_ls_global_clock_measured_complex_response_v2"
)
BROADBAND_FIR_LENGTH = 1_024
BROADBAND_PRE_ROLL_SAMPLES = 256
BROADBAND_MAXIMUM_DELAY_SAMPLES = 4_800
BROADBAND_HANDOFF_EXTRA_SAMPLES = 256
BROADBAND_MARKER_MAX_BRANCH_WIDTH_SAMPLES = 2_999.0
BROADBAND_MARKER_MIN_NORMALIZED_CORRELATION = 0.20
BROADBAND_MARKER_ALIAS_RELATIVE_SCORE = 0.80
BROADBAND_OUTPUT_PCM_PROVENANCE = "observed_submitted_int16_exact_plan"
DEFAULT_ANALYSIS_NPZ_NAME = "analysis_results.broadband_v5.npz"
DEFAULT_ANALYSIS_JSON_NAME = "analysis_metadata.broadband_v5.json"


def _panel_analysis_bands(
    panel_band_hz: tuple[float, float],
) -> tuple[tuple[float, float], ...]:
    values = [BROADBAND_CLOCK_PILOT_BAND_HZ, panel_band_hz]
    result: list[tuple[float, float]] = []
    for value in values:
        normalized = tuple(float(item) for item in value)
        if normalized not in result:
            result.append(normalized)
    return tuple(result)


def analyse_panel_capture(
    *,
    err: np.ndarray,
    ref: np.ndarray,
    output_pcm_int16: np.ndarray,
    probe: Any,
    period_starts: Sequence[int],
    panel_band_hz: tuple[float, float],
    min_alignment_score: float = mpi.DEFAULT_MIN_ALIGNMENT_SCORE,
    min_kept_repeats: int = 8,
    max_relative_tau_samples: float = mpi.DEFAULT_MAX_RELATIVE_TAU_SAMPLES,
    max_drift_deviation_samples: float = mpi.DEFAULT_MAX_DRIFT_DEVIATION_SAMPLES,
    period_authority_mask: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """한 panel을 compact FIR 없이 분석하고 stitch 가능한 복소 stack을 반환한다."""

    panel_band = tuple(float(value) for value in panel_band_hz)
    if not 0.0 < panel_band[0] < panel_band[1] < probe.sample_rate / 2.0:
        raise ValueError("panel band가 유효하지 않습니다")
    starts = [int(value) for value in period_starts]
    if len(starts) < int(min_kept_repeats) + 1:
        raise ValueError("독립 q/cubic witness를 포함할 panel repeat가 부족합니다")
    frequencies, stacks, separation = mpi.fractional_joint_channel_stacks(
        err=np.asarray(err, dtype=np.float64),
        ref=np.asarray(ref, dtype=np.float64),
        output_pcm_int16=np.asarray(output_pcm_int16),
        probe=probe,
        period_starts=starts,
        fit_band_hz=BROADBAND_CLOCK_PILOT_BAND_HZ,
        max_drift_deviation_samples=float(max_drift_deviation_samples),
        min_valid_periods=int(min_kept_repeats),
        clock_band_hz=BROADBAND_CLOCK_PILOT_BAND_HZ,
    )
    if period_authority_mask is not None:
        authority = np.asarray(period_authority_mask, dtype=np.bool_).reshape(-1)
        observed_valid = np.asarray(separation["valid"], dtype=np.bool_).reshape(-1)
        if authority.size != observed_valid.size:
            raise ValueError("panel period authority mask 길이가 repeat와 다릅니다")
        observed_valid = observed_valid & authority
        if int(observed_valid.sum()) < int(min_kept_repeats):
            raise ValueError("panel row-boundary 제외 뒤 clock-valid repeat가 8개 미만입니다")
        separation["valid"] = observed_valid
        separation["q"] = np.where(
            observed_valid,
            np.asarray(separation["q"], dtype=np.float64),
            np.nan,
        )
    keep, anchor, selection = mpi.select_repeats(
        frequencies=frequencies,
        stacks=stacks,
        sample_rate=probe.sample_rate,
        fit_band_hz=panel_band,
        max_relative_tau_samples=float(max_relative_tau_samples),
        max_drift_deviation_samples=float(max_drift_deviation_samples),
        min_kept_repeats=int(min_kept_repeats),
        initial_keep=separation["valid"],
        observed_drift_samples=separation["common_delay_samples"],
    )

    provisional_taus: dict[str, np.ndarray] = {}
    alignment_scores: dict[str, np.ndarray] = {}
    for drive in ("noise", "cancel"):
        _, tau, score = align_repeats(
            frequencies[drive],
            stacks[drive],
            sample_rate=probe.sample_rate,
            fit_band_hz=panel_band,
            anchor=anchor,
        )
        provisional_taus[drive] = tau
        alignment_scores[drive] = score
    final_keep = (
        keep
        & (alignment_scores["noise"] >= float(min_alignment_score))
        & (alignment_scores["cancel"] >= float(min_alignment_score))
    )
    final_keep[anchor] = True
    if int(final_keep.sum()) < int(min_kept_repeats):
        raise ValueError("두 drive가 함께 유지되는 highband repeat가 8개 미만입니다")
    score_sum = alignment_scores["noise"] + alignment_scores["cancel"]
    common_taus = (
        provisional_taus["noise"] * alignment_scores["noise"]
        + provisional_taus["cancel"] * alignment_scores["cancel"]
    ) / np.maximum(score_sum, np.finfo(np.float64).tiny)

    crosscheck = mpi.separation_crosscheck_metrics(
        frequencies=frequencies,
        joint_stacks=stacks,
        resampled_stacks=separation["crosscheck_transfers"],
        keep=final_keep,
        subbands_hz=_panel_analysis_bands(panel_band),
        overall_band_hz=panel_band,
    )
    aligned_stacks: dict[str, np.ndarray] = {}
    aligned_crosscheck_stacks: dict[str, np.ndarray] = {}
    mean_transfers: dict[str, np.ndarray] = {}
    mean_crosschecks: dict[str, np.ndarray] = {}
    panel_consistency: dict[str, float] = {}
    for drive in ("noise", "cancel"):
        frequency = frequencies[drive]
        phase = np.exp(
            2j
            * np.pi
            * frequency[None, :]
            * common_taus[:, None]
            / float(probe.sample_rate)
        )
        aligned = stacks[drive] * phase
        aligned_check = separation["crosscheck_transfers"][drive] * phase
        aligned_stacks[drive] = aligned[final_keep]
        aligned_crosscheck_stacks[drive] = aligned_check[final_keep]
        mean_transfers[drive] = np.mean(aligned_stacks[drive], axis=0)
        mean_crosschecks[drive] = np.mean(
            aligned_crosscheck_stacks[drive], axis=0
        )
        mask = (frequency >= panel_band[0]) & (frequency <= panel_band[1])
        if int(mask.sum()) < 8:
            raise ValueError(f"{drive} panel requested-band tone이 부족합니다")
        panel_consistency[drive] = complex_consistency(
            aligned_stacks[drive][:, mask]
        )
        if panel_consistency[drive] < mpi.MIN_BAND_CONSISTENCY:
            raise ValueError(
                f"{drive} panel consistency {panel_consistency[drive]:.6f} < "
                f"{mpi.MIN_BAND_CONSISTENCY:.6f}"
            )

    relative_tau = (
        provisional_taus["noise"][final_keep]
        - provisional_taus["cancel"][final_keep]
    )
    relative_max_abs = float(
        np.max(np.abs(relative_tau - np.median(relative_tau)))
    )
    phase_budget = max_timing_error_samples_for_attenuation(
        20.0, panel_band[1], float(probe.sample_rate)
    )
    if relative_max_abs > phase_budget:
        raise ValueError(
            f"panel P-S relative jitter {relative_max_abs:.6f} samples가 "
            f"{panel_band[1]:.0f}Hz 20dB 예산 {phase_budget:.6f}을 넘습니다"
        )

    selection.update(
        {
            "keep": final_keep,
            "anchor": int(anchor),
            "common_alignment_taus": common_taus,
            "relative_tau_kept": relative_tau,
            "relative_tau_max_abs_samples": relative_max_abs,
            "phase_budget_samples": phase_budget,
            "separation_crosscheck": crosscheck,
        }
    )
    return {
        "panel_band_hz": panel_band,
        "frequencies": frequencies,
        "transfers": mean_transfers,
        "crosscheck_transfers": mean_crosschecks,
        "aligned_stacks": aligned_stacks,
        "aligned_crosscheck_stacks": aligned_crosscheck_stacks,
        "panel_consistency": panel_consistency,
        "relative_tau_max_abs_samples": relative_max_abs,
        "separation": separation,
        "selection": selection,
    }


def derive_global_clock_map(
    *,
    period_starts: Sequence[int],
    common_delay_samples: Sequence[float],
    valid: Sequence[bool],
    period_samples: int,
    max_residual_samples: float = BROADBAND_GLOBAL_CLOCK_MAX_RESIDUAL_SAMPLES,
) -> dict[str, Any]:
    """clock-only period deltas를 캡처 전체의 한 ADC→DAC affine map으로 만든다."""

    starts = np.asarray(period_starts, dtype=np.int64).reshape(-1)
    delays = np.asarray(common_delay_samples, dtype=np.float64).reshape(-1)
    mask = np.asarray(valid, dtype=np.bool_).reshape(-1)
    n = int(period_samples)
    if starts.size < 3 or delays.size != starts.size or mask.size != starts.size:
        raise ValueError("global clock witness vector 길이가 잘못됐습니다")
    if n <= 0 or np.any(np.diff(starts) != n):
        raise ValueError("global clock period start가 exact contiguous grid가 아닙니다")
    interval_valid = mask[:-1] & np.isfinite(delays[:-1])
    if not interval_valid[0] or not interval_valid[-1]:
        raise ValueError("global clock 첫/마지막 interval witness가 유효하지 않습니다")
    indices = np.arange(starts.size - 1, dtype=np.float64)
    valid_indices = indices[interval_valid]
    if valid_indices.size < 8:
        raise ValueError("global clock valid interval이 8개 미만입니다")
    observed = delays[:-1]
    filled = np.interp(indices, valid_indices, observed[interval_valid])
    if np.any(np.abs(filled) > mpi.CLOCK_MAX_ABS_PERIOD_DELTA_SAMPLES):
        raise ValueError("global clock period delta가 hard bound를 넘습니다")
    if np.any(np.abs(np.diff(filled)) > mpi.CLOCK_MAX_ADJACENT_CHANGE_SAMPLES):
        raise ValueError("global clock trajectory에 one-sample slip/change가 있습니다")
    offsets = np.r_[0.0, np.cumsum(filled)]
    x = starts.astype(np.float64) - float(starts[0])
    design = np.column_stack((np.ones(x.size, dtype=np.float64), x))
    coefficients, *_ = np.linalg.lstsq(design, offsets, rcond=None)
    fitted = design @ coefficients
    residual = offsets - fitted
    maximum = float(np.max(np.abs(residual)))
    if not math.isfinite(maximum) or maximum > float(max_residual_samples):
        raise ValueError(
            f"global clock affine residual {maximum:.12f} > "
            f"{float(max_residual_samples):.12f} samples"
        )
    payload = {
        "schema": "broadband_global_clock_map_v1",
        "period_samples": n,
        "period_starts": starts.tolist(),
        "valid": mask.tolist(),
        "filled_period_delta_samples": filled.tolist(),
        "period_offsets_samples": offsets.tolist(),
        "intercept_samples": float(coefficients[0]),
        "slope_samples_per_sample": float(coefficients[1]),
        "maximum_residual_samples": maximum,
    }
    return {
        **payload,
        "period_starts": starts,
        "valid": mask,
        "filled_period_delta_samples": filled,
        "period_offsets_samples": offsets,
        "residual_samples": residual,
        "sha256": _sha256_bytes(_canonical_json_bytes(payload)),
    }


def validate_submitted_pilot_global_map(
    *,
    err: np.ndarray,
    ref: np.ndarray,
    output_pcm_int16: np.ndarray,
    probe: Any,
    period_starts: Sequence[int],
    clock_observation: Mapping[str, Any],
    global_clock_map: Mapping[str, Any],
    maximum_residual_samples: float = BROADBAND_GLOBAL_CLOCK_MAX_RESIDUAL_SAMPLES,
) -> dict[str, Any]:
    """실제 int16 pilot 분모로 global clock map을 P/S·ERR/REF에서 재검증한다.

    panel별 high-tone 양자화가 150--600 Hz bin에 만든 누설까지 실제 submitted
    spectrum에 포함해 나눈다. 따라서 intended float pilot이나 결과 highband phase를
    clock authority로 사용하지 않는다.
    """

    starts = np.asarray(period_starts, dtype=np.int64).reshape(-1)
    submitted = np.asarray(output_pcm_int16)
    if submitted.dtype != np.int16 or submitted.ndim != 2 or submitted.shape[1] != 2:
        raise ValueError("submitted pilot 검증에는 실제 [frames,2] int16 PCM이 필요합니다")
    q = np.asarray(clock_observation["q"], dtype=np.float64).reshape(-1)
    valid = np.asarray(clock_observation["valid"], dtype=np.bool_).reshape(-1)
    offsets = np.asarray(
        global_clock_map["period_offsets_samples"], dtype=np.float64
    ).reshape(-1)
    if not (starts.size == q.size == valid.size == offsets.size):
        raise ValueError("submitted pilot/global clock vector 길이가 다릅니다")
    if int(valid.sum()) < 8:
        raise ValueError("submitted pilot trajectory valid period가 8개 미만입니다")
    n = int(probe.period_samples)
    rate = int(probe.sample_rate)
    sample_index = np.arange(n, dtype=np.float64)
    trajectories: dict[str, dict[str, np.ndarray]] = {"err": {}, "ref": {}}
    observed_scores: list[float] = []
    spectra_digest = hashlib.sha256()
    valid_indices = np.flatnonzero(valid)
    cross_channel_null_digest = hashlib.sha256()
    cross_channel_null_max_absolute = 0.0
    cross_channel_null_max_ratio = 0.0
    for index in valid_indices:
        start = int(starts[index])
        if start < 0 or start + n > submitted.shape[0]:
            raise ValueError("submitted pilot null 검증 window가 raw 범위를 벗어납니다")
        null_receipt = validate_submitted_pilot_cross_channel_null(
            submitted[start : start + n],
            sample_rate=rate,
            period_seconds=float(n) / float(rate),
        )
        cross_channel_null_digest.update(np.asarray(start, dtype="<i8").tobytes())
        cross_channel_null_digest.update(
            _canonical_json_bytes(null_receipt)
        )
        for drive in ("noise", "cancel"):
            drive_row = null_receipt["drives"][drive]
            cross_channel_null_max_absolute = max(
                cross_channel_null_max_absolute,
                float(drive_row["cross_channel_max_magnitude"]),
            )
            cross_channel_null_max_ratio = max(
                cross_channel_null_max_ratio,
                float(drive_row["cross_to_main_max_ratio"]),
            )
    for mic_name, mic_values in (("err", err), ("ref", ref)):
        signal = np.asarray(mic_values, dtype=np.float64).reshape(-1)
        for drive, channel in (("noise", 0), ("cancel", 1)):
            bins = np.asarray(probe.bins_for(drive), dtype=np.int64)
            frequency = bins.astype(np.float64) * float(rate) / float(n)
            pilot_mask = (
                (frequency >= BROADBAND_CLOCK_PILOT_BAND_HZ[0])
                & (frequency <= BROADBAND_CLOCK_PILOT_BAND_HZ[1])
            )
            bins = bins[pilot_mask]
            frequency = frequency[pilot_mask]
            if bins.size < 8:
                raise ValueError(f"{drive} submitted pilot tone이 8개 미만입니다")
            stack: list[np.ndarray] = []
            for index in valid_indices:
                start = int(starts[index])
                coordinates = float(start) + sample_index / float(q[index])
                if coordinates[-1] >= signal.size - 1 or start + n > submitted.shape[0]:
                    raise ValueError("submitted pilot trajectory window가 raw 범위를 벗어납니다")
                resampled = mpi.map_coordinates(
                    signal,
                    [coordinates],
                    order=3,
                    mode="nearest",
                    prefilter=True,
                )
                output_spectrum = np.fft.rfft(
                    submitted[start : start + n, channel].astype(np.float64)
                    / float(np.iinfo(np.int16).max)
                )[bins]
                if np.any(np.abs(output_spectrum) <= 1.0e-9):
                    raise ValueError(f"{drive} actual submitted pilot spectrum이 0입니다")
                if mic_name == "err":
                    spectra_digest.update(np.asarray(start, dtype="<i8").tobytes())
                    spectra_digest.update(drive.encode("ascii"))
                    spectra_digest.update(
                        np.asarray(output_spectrum.real, dtype="<f8").tobytes()
                    )
                    spectra_digest.update(
                        np.asarray(output_spectrum.imag, dtype="<f8").tobytes()
                    )
                transfer = np.fft.rfft(resampled)[bins] / output_spectrum
                transfer *= np.exp(
                    2j * np.pi * frequency * float(offsets[index]) / float(rate)
                )
                stack.append(transfer)
            values = np.stack(stack)
            anchor = int(values.shape[0] // 2)
            _, residual_tau, scores = align_repeats(
                frequency,
                values,
                sample_rate=rate,
                fit_band_hz=BROADBAND_CLOCK_PILOT_BAND_HZ,
                span_samples=32.0,
                anchor=anchor,
            )
            residual_tau -= float(np.median(residual_tau))
            if np.any(scores < mpi.CLOCK_MIN_ADJACENT_SCORE):
                raise ValueError(
                    f"{mic_name}/{drive} submitted pilot trajectory score가 0.995 미만입니다"
                )
            observed_scores.extend(float(value) for value in scores)
            trajectories[mic_name][drive] = residual_tau

    vectors = [
        trajectories[mic][drive]
        for mic in ("err", "ref")
        for drive in ("noise", "cancel")
    ]
    maximum_residual = float(max(np.max(np.abs(value)) for value in vectors))
    pairwise_agreement = float(
        max(
            np.max(np.abs(left - right))
            for index, left in enumerate(vectors)
            for right in vectors[index + 1 :]
        )
    )
    limit = float(maximum_residual_samples)
    if maximum_residual > limit or pairwise_agreement > limit:
        raise ValueError(
            "actual submitted pilot의 ERR/REF/P/S global trajectory가 timing 예산을 "
            f"넘습니다: residual={maximum_residual:.12f}, "
            f"agreement={pairwise_agreement:.12f}, limit={limit:.12f}"
        )
    payload = {
        "schema": "actual_submitted_int16_pilot_global_map_validation_v1",
        "valid_period_count": int(valid.sum()),
        "maximum_residual_samples": maximum_residual,
        "pairwise_trajectory_agreement_samples": pairwise_agreement,
        "minimum_alignment_score": float(min(observed_scores)),
        "submitted_pilot_spectra_sha256": spectra_digest.hexdigest(),
        "cross_channel_null_sha256": cross_channel_null_digest.hexdigest(),
        "cross_channel_null_maximum_absolute_observed": (
            cross_channel_null_max_absolute
        ),
        "cross_channel_null_maximum_ratio_observed": cross_channel_null_max_ratio,
        "clock_map_sha256": str(global_clock_map["sha256"]),
        "highband_phase_used_for_map": False,
    }
    return {
        **payload,
        "trajectories": trajectories,
        "sha256": _sha256_bytes(_canonical_json_bytes(payload)),
    }


def estimate_nonperiodic_marker_delay(
    *,
    output_pcm: np.ndarray,
    err: np.ndarray,
    start_frame: int,
    stop_frame: int,
    output_channel: int,
    maximum_delay_samples: int = BROADBAND_MAXIMUM_DELAY_SAMPLES,
) -> dict[str, Any]:
    """channel-separated 비주기 marker로 0..4800의 유일 coarse branch를 고른다."""

    submitted = np.asarray(output_pcm)
    signal = np.asarray(err, dtype=np.float64).reshape(-1)
    start, stop = int(start_frame), int(stop_frame)
    channel = int(output_channel)
    if submitted.dtype != np.int16 or submitted.ndim != 2 or channel not in (0, 1):
        raise ValueError("timing marker submitted PCM/channel이 잘못됐습니다")
    marker = submitted[start:stop, channel].astype(np.float64)
    if marker.size < 1024 or float(np.linalg.norm(marker)) <= 0.0:
        raise ValueError("timing marker PCM이 비었거나 너무 짧습니다")
    response = signal[start : stop + int(maximum_delay_samples)]
    if response.size != marker.size + int(maximum_delay_samples):
        raise ValueError("timing marker 응답 window가 raw 범위를 벗어납니다")
    scores = correlate(response, marker, mode="valid", method="fft")
    energy = np.convolve(response * response, np.ones(marker.size), mode="valid")
    denominator = np.sqrt(np.maximum(energy, np.finfo(np.float64).tiny)) * float(
        np.linalg.norm(marker)
    )
    normalized = np.abs(scores) / denominator
    best = int(np.argmax(normalized))
    best_score = float(normalized[best])
    branch_slices = (
        slice(0, min(3_000, normalized.size)),
        slice(min(3_000, normalized.size), normalized.size),
    )
    branch_peaks = tuple(
        float(np.max(normalized[branch])) if normalized[branch].size else 0.0
        for branch in branch_slices
    )
    candidate_threshold = max(
        BROADBAND_MARKER_MIN_NORMALIZED_CORRELATION,
        BROADBAND_MARKER_ALIAS_RELATIVE_SCORE * best_score,
    )
    candidate_count = int(sum(value >= candidate_threshold for value in branch_peaks))
    if best_score < BROADBAND_MARKER_MIN_NORMALIZED_CORRELATION:
        candidate_count = 0
    if candidate_count != 1:
        raise ValueError(
            "timing marker coarse delay branch가 유일하지 않습니다: "
            f"candidate_count={candidate_count}, branch_peaks={branch_peaks}"
        )
    fraction = 0.0
    if 0 < best < normalized.size - 1:
        y0, y1, y2 = normalized[best - 1 : best + 2]
        curvature = float(y0 - 2.0 * y1 + y2)
        if curvature != 0.0:
            fraction = float(0.5 * (y0 - y2) / curvature)
    coarse = float(best) + fraction
    half_width = 1_000.0
    lower = max(0.0, coarse - half_width)
    upper = min(float(maximum_delay_samples), coarse + half_width)
    width = upper - lower
    if not 0.0 < width < BROADBAND_MARKER_MAX_BRANCH_WIDTH_SAMPLES:
        raise ValueError("timing marker branch width가 0..2999 samples가 아닙니다")
    return {
        "coarse_delay_samples": coarse,
        "peak_normalized_correlation": best_score,
        "alias_branch_peak_normalized_correlation": branch_peaks,
        "alias_candidate_threshold": candidate_threshold,
        "search_lower_samples": lower,
        "search_upper_samples": upper,
        "search_width_samples": width,
        "alias_period_samples": 12_000,
        "alias_candidate_count": candidate_count,
    }


def _apply_global_clock_offset(panel: dict[str, Any], delay_samples: float) -> None:
    """clock-only map에서 정한 하나의 offset을 P/S 모든 복소 배열에 동일 적용한다."""

    for key in ("transfers", "crosscheck_transfers"):
        panel[key] = apply_shared_panel_delay(
            panel["frequencies"], panel[key], delay_samples=delay_samples
        )
    for key in ("aligned_stacks", "aligned_crosscheck_stacks"):
        for drive in ("noise", "cancel"):
            frequency = np.asarray(panel["frequencies"][drive], dtype=np.float64)
            panel[key][drive] = panel[key][drive] * np.exp(
                2j * np.pi * frequency[None, :] * float(delay_samples) / 48_000.0
            )
    panel["global_clock_offset_samples"] = float(delay_samples)


def _failed_compact_subband_rows(
    subbands_hz: Sequence[Sequence[float]],
) -> tuple[dict[str, Any], ...]:
    """식별 불가능한 compact FIR을 유효한 plant처럼 보이게 하지 않는다."""

    return tuple(
        {
            "band_hz": [float(band[0]), float(band[1])],
            "tone_count": 0,
            "complex_agreement": 0.0,
            "relative_error": 1.0,
            "passed": False,
        }
        for band in subbands_hz
    )


def _diagnostic_compact_fit(
    frequencies_hz: np.ndarray,
    measured_transfer: np.ndarray,
    *,
    effective_delay_samples: int,
    fir_length: int,
    sample_rate: int,
    subbands_hz: Sequence[Sequence[float]],
    ridge_relative: float = 1.0e-8,
) -> dict[str, Any]:
    """compact FIR 식별성을 먼저 계측하고 결과를 diagnostic으로만 반환한다.

    measured complex tone은 canonical plant다. 이 함수의 성공 또는 실패는 해당
    measured-band plant의 발행 자격을 바꾸지 않는다.
    """

    frequency = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    measured = np.asarray(measured_transfer, dtype=np.complex128).reshape(-1)
    length = int(fir_length)
    delay = int(effective_delay_samples)
    rate = int(sample_rate)
    design_receipt = compact_fir_identifiability_receipt(
        frequency,
        effective_delay_samples=delay,
        fir_length=length,
        sample_rate=rate,
        ridge_relative=float(ridge_relative),
    )
    identifiable = bool(design_receipt["compact_training_eligible"])
    fir = np.zeros(length, dtype=np.float32)
    reconstructed = np.zeros_like(measured)
    rows = _failed_compact_subband_rows(subbands_hz)
    compact: dict[str, Any] = {
        "fir": fir,
        "reconstructed_transfer": reconstructed,
        "complex_agreement": 0.0,
        "relative_error": 1.0,
        "tone_count": int(frequency.size),
        "fir_length": length,
        "effective_delay_samples": delay,
        "ridge_relative": float(ridge_relative),
        "ridge_absolute": 0.0,
        "passed": False,
    }
    failure_reason = (
        "design_matrix_not_identifiable"
        if not identifiable
        else "compact_fit_or_roundtrip_failed"
    )
    if identifiable:
        try:
            candidate = fit_real_compact_fir(
                frequency,
                measured,
                effective_delay_samples=delay,
                fir_length=length,
                sample_rate=rate,
                ridge_relative=float(ridge_relative),
            )
            candidate_rows = band_roundtrip_metrics(
                frequency,
                measured,
                candidate["reconstructed_transfer"],
                subbands_hz=subbands_hz,
            )
            compact = candidate
            fir = np.asarray(candidate["fir"], dtype=np.float32)
            reconstructed = np.asarray(
                candidate["reconstructed_transfer"], dtype=np.complex128
            )
            rows = candidate_rows
            if bool(candidate["passed"]) and all(row["passed"] for row in rows):
                failure_reason = "none_diagnostic_fit_passed"
        except (np.linalg.LinAlgError, ValueError) as exc:
            failure_reason = f"compact_fit_exception:{type(exc).__name__}"

    spectrum = np.fft.rfft(np.asarray(fir, dtype=np.float64), n=16_384)
    spectrum_frequency = np.fft.rfftfreq(16_384, 1.0 / float(rate))
    valid_lo = float(subbands_hz[0][0])
    valid_hi = float(subbands_hz[-1][1])
    in_band = (spectrum_frequency >= valid_lo) & (spectrum_frequency <= valid_hi)
    out_band = ~in_band

    def maximum_gain_db(mask: np.ndarray) -> float:
        maximum = float(np.max(np.abs(spectrum[mask]))) if np.any(mask) else 0.0
        return float(20.0 * math.log10(max(maximum, 1.0e-15)))

    receipt = {
        "schema_version": "compact_fir_identifiability_diagnostic_v1",
        "compact_role": "diagnostic_only",
        "compact_training_eligible": False,
        "design_identifiability": design_receipt,
        "fir_length": length,
        "tone_count": int(frequency.size),
        "numeric_rank": int(design_receipt["numeric_rank"]),
        "condition_number": design_receipt["condition_number"],
        "ridge_relative": float(ridge_relative),
        "coefficient_l2_norm": float(np.linalg.norm(fir)),
        "coefficient_max_abs": float(np.max(np.abs(fir))) if fir.size else 0.0,
        "in_band_max_gain_db": maximum_gain_db(in_band),
        "out_of_band_max_gain_db": maximum_gain_db(out_band),
        "reason": failure_reason,
    }
    receipt_sha = _sha256_bytes(_canonical_json_bytes(receipt))
    return {
        "fir": fir,
        "reconstructed_transfer": reconstructed,
        "compact": compact,
        "compact_subbands": rows,
        "compact_role": "diagnostic_only",
        "compact_training_eligible": False,
        "compact_identifiability": receipt,
        "compact_identifiability_sha256": receipt_sha,
    }


def stitch_broadband_panels(
    panel_results: Sequence[dict[str, Any]],
    *,
    contract: ControlBandContract,
    fir_length: int = 1_024,
    pre_roll_samples: int = 256,
    maximum_delay_samples: int = 4_800,
    marker_delay_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """다섯 panel을 공통 시간축으로 stitch하고 P/S broadband FIR을 한 번만 적합한다."""

    if contract.role != "broadband_point_control":
        raise ValueError("광대역 panel stitch에는 broadband control contract가 필요합니다")
    if len(panel_results) != len(contract.measurement_panels_hz):
        raise ValueError("measurement panel 수가 계약과 다릅니다")
    panels: list[dict[str, Any]] = []
    for index, (source, expected_band) in enumerate(
        zip(panel_results, contract.measurement_panels_hz, strict=True)
    ):
        actual = tuple(float(value) for value in source["panel_band_hz"])
        if actual != tuple(expected_band):
            raise ValueError(f"panel #{index} band가 계약과 다릅니다")
        panels.append(
            {
                **source,
                "frequencies": {
                    key: np.asarray(value).copy()
                    for key, value in source["frequencies"].items()
                },
                "transfers": {
                    key: np.asarray(value).copy()
                    for key, value in source["transfers"].items()
                },
                "crosscheck_transfers": {
                    key: np.asarray(value).copy()
                    for key, value in source["crosscheck_transfers"].items()
                },
                "aligned_stacks": {
                    key: np.asarray(value).copy()
                    for key, value in source["aligned_stacks"].items()
                },
                "aligned_crosscheck_stacks": {
                    key: np.asarray(value).copy()
                    for key, value in source["aligned_crosscheck_stacks"].items()
                },
            }
        )

    stitch_reports: list[dict[str, Any]] = []
    for index in range(1, len(panels)):
        previous = panels[index - 1]
        current = panels[index]
        overlap = (
            max(previous["panel_band_hz"][0], current["panel_band_hz"][0]),
            min(previous["panel_band_hz"][1], current["panel_band_hz"][1]),
        )
        report = fit_shared_panel_delay(
            reference_frequencies=previous["frequencies"],
            reference_transfers=previous["transfers"],
            current_frequencies=current["frequencies"],
            current_transfers=current["transfers"],
            overlap_band_hz=overlap,
            sample_rate=contract.sample_rate,
        )
        if not report["passed"]:
            raise ValueError(f"panel #{index - 1}→#{index} clock-corrected overlap 검증 실패")
        phase_budget = max_timing_error_samples_for_attenuation(
            contract.measurement_resolution_attenuation_db,
            max(previous["panel_band_hz"][1], current["panel_band_hz"][1]),
            contract.sample_rate,
        )
        if abs(float(report["shared_delay_samples"])) > phase_budget:
            raise ValueError(
                f"panel #{index - 1}→#{index} overlap residual이 clock 예산을 넘습니다"
            )
        report = {
            **report,
            "mode": "validation_only_after_global_clock_map",
            "applied_phase_repair_samples": 0.0,
            "phase_budget_samples": phase_budget,
        }
        stitch_reports.append(report)

    drive_results: dict[str, Any] = {}
    for drive in ("noise", "cancel"):
        frequency, measured, observations = merge_stitched_panels(
            panels, drive=drive
        )
        crosscheck_panels = [
            {**panel, "transfers": panel["crosscheck_transfers"]}
            for panel in panels
        ]
        check_frequency, checked, check_observations = merge_stitched_panels(
            crosscheck_panels, drive=drive
        )
        if not np.array_equal(frequency, check_frequency) or not np.array_equal(
            observations, check_observations
        ):
            raise ValueError("joint-LS와 cubic stitched frequency grid가 다릅니다")
        separation_rows = band_roundtrip_metrics(
            frequency,
            measured,
            checked,
            subbands_hz=contract.point_control_subbands_hz,
            minimum_complex_agreement=mpi.SEPARATION_CROSSCHECK_MIN_AGREEMENT,
            maximum_relative_error=mpi.SEPARATION_CROSSCHECK_MAX_RELATIVE_ERROR,
        )
        if not all(row["passed"] for row in separation_rows):
            raise ValueError(f"{drive} stitched joint-LS/cubic subband crosscheck 실패")
        bounds = (
            marker_delay_bounds.get(drive)
            if marker_delay_bounds is not None
            else (0.0, float(maximum_delay_samples))
        )
        if bounds is None or not 0.0 <= float(bounds[0]) < float(bounds[1]) <= float(
            maximum_delay_samples
        ) or float(bounds[1]) - float(bounds[0]) >= 3_000.0:
            raise ValueError(f"{drive} timing-marker bulk branch가 유효하지 않습니다")
        bulk_fractional = estimate_bulk_delay_samples(
            frequency,
            measured,
            sample_rate=contract.sample_rate,
            minimum_delay_samples=float(bounds[0]),
            maximum_delay_samples=float(bounds[1]),
        )
        bulk_integer = int(round(bulk_fractional))
        effective_delay = bulk_integer - int(pre_roll_samples)
        if effective_delay < 0:
            raise ValueError(f"{drive} bulk delay가 pre-roll보다 작습니다")
        holdout = build_every_other_tone_holdout_receipt(
            frequencies_hz=frequency,
            transfer=measured,
            bulk_delay_fractional_samples=bulk_fractional,
            sample_rate=contract.sample_rate,
            subbands_hz=contract.point_control_subbands_hz,
        )
        if not bool(holdout["passed"]):
            raise ValueError(f"{drive} measured-band every-other-tone holdout 실패")
        compact_diagnostic = _diagnostic_compact_fit(
            frequency,
            measured,
            effective_delay_samples=effective_delay,
            fir_length=int(fir_length),
            sample_rate=contract.sample_rate,
            subbands_hz=contract.point_control_subbands_hz,
        )
        drive_results[drive] = {
            "frequencies_hz": frequency,
            "mean_transfer": measured,
            "observations_per_frequency": observations,
            "bulk_delay_fractional_samples": bulk_fractional,
            "bulk_delay_samples": bulk_integer,
            "effective_delay_samples": effective_delay,
            "fractional_effective_delay_samples": (
                float(bulk_fractional) - float(pre_roll_samples)
            ),
            "pre_roll_samples": int(pre_roll_samples),
            "measured_interpolation_holdout": holdout,
            "measured_interpolation_subbands": tuple(holdout["rows"]),
            "fir": compact_diagnostic["fir"],
            "compact": compact_diagnostic["compact"],
            "compact_subbands": compact_diagnostic["compact_subbands"],
            "compact_role": compact_diagnostic["compact_role"],
            "compact_training_eligible": compact_diagnostic[
                "compact_training_eligible"
            ],
            "compact_identifiability": compact_diagnostic[
                "compact_identifiability"
            ],
            "compact_identifiability_sha256": compact_diagnostic[
                "compact_identifiability_sha256"
            ],
            "separation_subbands": separation_rows,
        }

    # panel stitch가 P/S에 서로 다른 phase repair를 적용하지 않았는지
    # 최종 FIR 적합과 독립적으로 다시 검증한다. 각 panel은 저역 clock
    # pilot과 요청 대역을 같이 담으므로 광대역 phase slope의 별칭을 막는다.
    panel_bulk_delays: dict[str, list[float]] = {"noise": [], "cancel": []}
    for index, panel in enumerate(panels):
        for drive in ("noise", "cancel"):
            bounds = (
                marker_delay_bounds.get(drive)
                if marker_delay_bounds is not None
                else (0.0, float(maximum_delay_samples))
            )
            assert bounds is not None
            delay = estimate_bulk_delay_samples(
                np.asarray(panel["frequencies"][drive], dtype=np.float64),
                np.asarray(panel["transfers"][drive], dtype=np.complex128),
                sample_rate=contract.sample_rate,
                minimum_delay_samples=float(bounds[0]),
                maximum_delay_samples=float(bounds[1]),
            )
            if not math.isfinite(delay):
                raise ValueError(f"panel #{index} {drive} bulk delay가 finite가 아닙니다")
            panel_bulk_delays[drive].append(float(delay))
    panel_relative_delays = tuple(
        primary - secondary
        for primary, secondary in zip(
            panel_bulk_delays["noise"],
            panel_bulk_delays["cancel"],
            strict=True,
        )
    )
    final_fractional_relative = float(
        drive_results["noise"]["bulk_delay_fractional_samples"]
        - drive_results["cancel"]["bulk_delay_fractional_samples"]
    )
    panel_relative_deviations = tuple(
        abs(value - final_fractional_relative) for value in panel_relative_delays
    )

    consistency_vectors: dict[str, list[float]] = {"noise": [], "cancel": []}
    relative_jitter: list[float] = []
    for band in contract.point_control_subbands_hz:
        contributing = [
            panel
            for panel in panels
            if max(panel["panel_band_hz"][0], band[0])
            < min(panel["panel_band_hz"][1], band[1])
        ]
        if not contributing:
            raise ValueError(f"control subband {band}를 덮는 panel이 없습니다")
        relative_jitter.append(
            max(float(panel["relative_tau_max_abs_samples"]) for panel in contributing)
        )
        for drive in ("noise", "cancel"):
            values: list[float] = []
            for panel in contributing:
                frequency = np.asarray(panel["frequencies"][drive])
                mask = (frequency >= band[0]) & (frequency <= band[1])
                if int(mask.sum()) >= 4:
                    values.append(
                        complex_consistency(panel["aligned_stacks"][drive][:, mask])
                    )
            if not values:
                raise ValueError(f"{drive} control subband {band} consistency tone이 없습니다")
            consistency_vectors[drive].append(min(values))
    if any(value < mpi.MIN_BAND_CONSISTENCY for values in consistency_vectors.values() for value in values):
        raise ValueError("stitched control subband consistency가 0.95 미만입니다")

    return {
        "status": "PASS",
        "control_band_contract_sha256": contract.digest(),
        "panel_stitch": stitch_reports,
        "applied_per_drive_phase_repair_samples": (0.0,) * (2 * len(panels)),
        "panels": panels,
        "drives": drive_results,
        "primary_consistency": tuple(consistency_vectors["noise"]),
        "secondary_consistency": tuple(consistency_vectors["cancel"]),
        "relative_phase_jitter_samples": tuple(relative_jitter),
        "panel_bulk_delay_fractional_samples": {
            drive: tuple(values) for drive, values in panel_bulk_delays.items()
        },
        "panel_primary_minus_secondary_bulk_delay_samples": panel_relative_delays,
        "panel_relative_delay_deviation_samples": panel_relative_deviations,
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, complex):
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            raise ValueError("JSON evidence에 NaN/Inf complex를 쓸 수 없습니다")
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON evidence에 NaN/Inf를 쓸 수 없습니다")
    return value


def _repository_path(
    value: str | Path,
    *,
    repository_root: str | Path,
    label: str,
) -> Path:
    root = Path(repository_root).resolve()
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    result = candidate.resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label}는 저장소 안에 있어야 합니다: {result}") from exc
    return result


def _bound_metadata_path(
    value: Any,
    *,
    repository_root: Path,
    expected_path: Path | None,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"raw metadata에 {label} path가 필요합니다")
    actual = _repository_path(value, repository_root=repository_root, label=label)
    if expected_path is not None and actual != expected_path.resolve():
        raise ValueError(
            f"raw metadata {label} path가 명시한 파일과 다릅니다: "
            f"raw={actual}, requested={expected_path.resolve()}"
        )
    if not actual.is_file():
        raise FileNotFoundError(f"{label} 파일이 없습니다: {actual}")
    return actual


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label}는 64자리 lowercase SHA-256이어야 합니다")
    return digest


def _parse_utc(value: Any, *, label: str) -> dt.datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} UTC timestamp가 필요합니다")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} UTC timestamp를 읽을 수 없습니다: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label}에 timezone이 필요합니다")
    normalized = parsed.astimezone(dt.timezone.utc)
    if normalized.utcoffset() != dt.timedelta(0):  # pragma: no cover - astimezone invariant
        raise ValueError(f"{label}을 UTC로 정규화할 수 없습니다")
    return normalized


def _read_json_snapshot(path: Path, *, label: str) -> tuple[dict[str, Any], bytes, str]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} JSON을 읽을 수 없습니다: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON root는 mapping이어야 합니다")
    return value, payload, _sha256_bytes(payload)


def _clip_count_int32(value: np.ndarray) -> int:
    normalized = pcm_int32_to_float32(np.asarray(value, dtype=np.int32)).astype(
        np.float64
    )
    return int(np.count_nonzero(np.abs(normalized) >= 0.99))


def _validate_clean_telemetry(
    telemetry: Any, *, expected_frames: int, sample_rate: int
) -> int:
    if not isinstance(telemetry, dict):
        raise ValueError("raw telemetry mapping이 필요합니다")
    try:
        xrun_count = int(telemetry["xrun_count"])
        unexpected = int(telemetry["unexpected_status_count"])
        captured_frames = int(telemetry["captured_frames"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("raw telemetry의 xrun/status/captured_frames가 필요합니다") from exc
    failures = {
        "xrun": xrun_count != 0,
        "unexpected_callback_status": unexpected != 0,
        "callback_error": bool(telemetry.get("callback_error")),
        "stream_abort_error": bool(telemetry.get("stream_abort_error")),
        "stream_close_error": bool(telemetry.get("stream_close_error")),
        "output_stop_unconfirmed": telemetry.get("output_stop_confirmed") is not True,
        "capture_incomplete": telemetry.get("completed") is not True,
        "captured_frames_mismatch": captured_frames != int(expected_frames),
        "termination_signal": telemetry.get("termination_signal") is not None,
    }
    bad = [label for label, failed in failures.items() if failed]
    if bad:
        raise ValueError("광대역 raw telemetry gate 실패: " + ", ".join(bad))
    nominal = expected_frames / float(sample_rate)
    try:
        stored_nominal = float(telemetry["nominal_output_seconds"])
        stored_hard_max = float(telemetry["hard_max_output_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("raw telemetry nominal/hard-max witness가 필요합니다") from exc
    if not math.isclose(stored_nominal, nominal, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(
        stored_hard_max,
        nominal + mpi.LIVE_WATCHDOG_GRACE_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("raw telemetry nominal/hard-max가 exact plan과 다릅니다")
    return xrun_count


def _reverify_bound_files(bound_files: Mapping[str, Mapping[str, Any]]) -> None:
    for label, item in bound_files.items():
        path = Path(str(item["path"])).resolve()
        expected = str(item["sha256"])
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"분석 중 {label} bytes가 변경됐습니다: {path}")


def load_broadband_raw_capture(
    *,
    session_dir: str | Path,
    plan_path: str | Path,
    hardware_path: str | Path,
    repository_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """broadband raw-first session을 한 byte snapshot으로 읽고 계보를 다시 검증한다."""

    root = Path(repository_root).resolve()
    session = _repository_path(
        session_dir, repository_root=root, label="broadband raw session"
    )
    if not session.is_dir():
        raise FileNotFoundError(f"broadband raw session이 없습니다: {session}")
    raw_path = session / "raw_measurement.npz"
    sidecar_path = session / "metadata.json"
    if not raw_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("broadband raw_measurement.npz/metadata.json 쌍이 필요합니다")
    raw_bytes = raw_path.read_bytes()
    raw_sha = _sha256_bytes(raw_bytes)
    try:
        with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as archive:
            required = {
                "metadata_json",
                "submitted_output_pcm_int16",
                "input_raw_int32",
                "preflight_raw_int32",
                *broadband_measure.CALLBACK_TIME_INFO_FIELDS,
            }
            if set(archive.files) != required:
                raise ValueError(
                    f"broadband raw array schema가 다릅니다: {sorted(archive.files)}"
                )
            metadata = json.loads(str(archive["metadata_json"].item()))
            submitted = np.asarray(archive["submitted_output_pcm_int16"])
            input_raw = np.asarray(archive["input_raw_int32"])
            preflight_raw = np.asarray(archive["preflight_raw_int32"])
            callback_time_info = {
                name: np.asarray(archive[name])
                for name in broadband_measure.CALLBACK_TIME_INFO_FIELDS
            }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"broadband raw NPZ를 읽을 수 없습니다: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("broadband raw embedded metadata는 mapping이어야 합니다")
    sidecar, _sidecar_bytes, sidecar_sha = _read_json_snapshot(
        sidecar_path, label="broadband raw sidecar"
    )
    if _canonical_json_bytes(sidecar) != _canonical_json_bytes(metadata):
        raise ValueError("raw NPZ embedded metadata와 metadata.json이 다릅니다")

    if metadata.get("raw_capture_schema") != broadband_measure.BROADBAND_RAW_CAPTURE_SCHEMA:
        raise ValueError(f"broadband raw schema가 다릅니다: {metadata.get('raw_capture_schema')!r}")
    if metadata.get("method") != broadband_measure.BROADBAND_METHOD:
        raise ValueError("broadband raw method가 다릅니다")
    if metadata.get("status") != "PASS" or metadata.get("valid") is not True:
        raise ValueError("PASS/valid broadband raw만 승격할 수 있습니다")
    if metadata.get("invalid_reasons") != []:
        raise ValueError(f"broadband raw에 결함이 기록됐습니다: {metadata.get('invalid_reasons')!r}")
    if metadata.get("analysis_status") != "NOT_RUN_RAW_FIRST":
        raise ValueError("broadband raw-first analysis_status가 다릅니다")
    if metadata.get("post_capture_binding") != {"valid": True, "error": None}:
        raise ValueError("broadband raw의 post-capture TOCTOU binding이 PASS가 아닙니다")
    callback_summary = broadband_measure.validate_callback_time_info(
        callback_time_info,
        expected_frames=int(input_raw.shape[0]),
    )
    if metadata.get("callback_timing") != callback_summary:
        raise ValueError("raw callback timing summary가 저장 배열 재검산과 다릅니다")
    capture_id = str(metadata.get("capture_id", "")).strip()
    if not capture_id:
        raise ValueError("broadband capture_id가 비었습니다")

    sample_rate = int(metadata.get("sample_rate", -1))
    block_size = int(metadata.get("block_size", -1))
    latency = str(metadata.get("latency", ""))
    if (sample_rate, block_size, latency) != (48_000, 256, "low"):
        raise ValueError("broadband raw는 48kHz/256/low여야 합니다")
    exact_channels = {
        "error_mic": 0,
        "reference_mic": 1,
        "noise_out": 0,
        "cancel_out": 1,
    }
    if metadata.get("channel_map") != exact_channels:
        raise ValueError("broadband raw channel map이 exact 0/1 계약과 다릅니다")
    exact_confirmations = {
        "speaker_output": True,
        "user_present": True,
        "volume_minimum": True,
        "routing_and_geometry": True,
        "same_amplifier_setting": True,
    }
    if metadata.get("operator_confirmations") != exact_confirmations:
        raise ValueError("broadband raw operator confirmation 5개가 exact PASS가 아닙니다")

    requested_plan = _repository_path(
        plan_path, repository_root=root, label="exact broadband plan"
    )
    requested_hardware = _repository_path(
        hardware_path, repository_root=root, label="hardware YAML"
    )
    plan_binding = metadata.get("plan")
    hardware_binding = metadata.get("hardware")
    if not isinstance(plan_binding, dict) or not isinstance(hardware_binding, dict):
        raise ValueError("raw metadata plan/hardware binding이 필요합니다")
    bound_plan = _bound_metadata_path(
        plan_binding.get("path"),
        repository_root=root,
        expected_path=requested_plan,
        label="exact broadband plan",
    )
    bound_hardware = _bound_metadata_path(
        hardware_binding.get("path"),
        repository_root=root,
        expected_path=requested_hardware,
        label="hardware YAML",
    )
    plan, _plan_bytes, plan_file_sha = _read_json_snapshot(
        bound_plan, label="exact broadband plan"
    )
    hardware_sha = _sha256_file(bound_hardware)
    if _require_sha256(plan_binding.get("file_sha256"), label="plan file SHA") != plan_file_sha:
        raise ValueError("raw에 박힌 plan file SHA가 실제 bytes와 다릅니다")
    plan_payload_sha = _sha256_bytes(_canonical_json_bytes(plan))
    if _require_sha256(plan_binding.get("payload_sha256"), label="plan payload SHA") != plan_payload_sha:
        raise ValueError("raw에 박힌 plan payload SHA가 다릅니다")
    if _require_sha256(hardware_binding.get("sha256"), label="hardware SHA") != hardware_sha:
        raise ValueError("raw에 박힌 hardware SHA가 실제 bytes와 다릅니다")
    if plan_binding.get("schema") != broadband_measure.BROADBAND_MEASUREMENT_PLAN_SCHEMA:
        raise ValueError("raw plan schema binding이 다릅니다")

    hardware_config = load_yaml(bound_hardware)
    audio, hardware_channels = validate_measurement_hardware_contract(hardware_config)
    if hardware_channels != exact_channels:
        raise ValueError("hardware channel map이 broadband raw와 다릅니다")
    if (int(audio["sample_rate"]), int(audio["block_size"]), str(audio["latency"])) != (
        sample_rate,
        block_size,
        latency,
    ):
        raise ValueError("hardware audio timing이 broadband raw와 다릅니다")
    expected_plan, expected_pcm = broadband_measure.build_signal_plan(
        hardware_path=bound_hardware
    )
    if plan != expected_plan:
        raise ValueError("exact plan이 현재 broadband contract/hardware/recipe와 다릅니다")
    broadband_measure.validate_plan_pcm_exact(plan, expected_pcm)
    if plan.get("control_band_contract_sha256") != ControlBandContract.broadband_point_control().digest():
        raise ValueError("plan control-band contract SHA가 현재 계약과 다릅니다")
    if metadata.get("control_band_contract_sha256") != plan["control_band_contract_sha256"]:
        raise ValueError("raw/plan control-band contract SHA가 다릅니다")

    expected_shape = tuple(expected_pcm.shape)
    if submitted.dtype != np.int16 or submitted.shape != expected_shape:
        raise ValueError("submitted_output_pcm_int16 shape/dtype이 exact plan과 다릅니다")
    if not np.array_equal(submitted, expected_pcm):
        raise ValueError("실제 제출 int16 PCM이 exact plan PCM과 한 code라도 다릅니다")
    submitted_sha = _sha256_bytes(submitted.tobytes(order="C"))
    expected_pcm_sha = _require_sha256(plan["output"].get("pcm_sha256"), label="plan PCM SHA")
    if submitted_sha != expected_pcm_sha or plan_binding.get("pcm_sha256") != expected_pcm_sha:
        raise ValueError("submitted/plan PCM SHA binding이 다릅니다")
    if metadata.get("submitted_pcm_sha256") != submitted_sha:
        raise ValueError("raw top-level submitted PCM SHA가 실제 배열과 다릅니다")
    broadband_measure.validate_live_authority_binding(
        {
            "path": bound_plan,
            "file_sha256": plan_file_sha,
            "payload_sha256": plan_payload_sha,
            "payload": plan,
        },
        submitted,
    )
    if input_raw.dtype != np.int32 or input_raw.shape != (expected_shape[0], 2):
        raise ValueError("input_raw_int32 shape/dtype이 exact plan과 다릅니다")
    if preflight_raw.dtype != np.int32 or preflight_raw.ndim != 2 or preflight_raw.shape[1] != 2:
        raise ValueError("preflight_raw_int32은 int32 [frames,2]여야 합니다")
    expected_preflight_frames = int(
        round(
            (
                broadband_measure.DEFAULT_INPUT_PREFLIGHT_SECONDS
                - mpi.cw.DEFAULT_PROBE_SETTLE_SECONDS
            )
            * sample_rate
        )
    )
    if preflight_raw.shape[0] != expected_preflight_frames:
        raise ValueError("preflight raw가 3초 total/1초 settle 계약과 다릅니다")
    if not math.isclose(
        float(metadata.get("input_preflight_seconds", float("nan"))),
        broadband_measure.DEFAULT_INPUT_PREFLIGHT_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("raw input_preflight_seconds가 3초가 아닙니다")

    measurement_report = analyze_int32_input_probe(input_raw)
    preflight_report = analyze_int32_input_probe(preflight_raw)
    if not all(bool(row.get("valid")) for row in measurement_report["channels"][:2]):
        raise ValueError("broadband capture ERR/REF raw channel이 유효하지 않습니다")
    if not all(bool(row.get("valid")) for row in preflight_report["channels"][:2]):
        raise ValueError("broadband preflight ERR/REF raw channel이 유효하지 않습니다")
    stored_preflight = metadata.get("preflight")
    if not isinstance(stored_preflight, dict):
        raise ValueError("raw preflight report가 필요합니다")
    if stored_preflight.get("frames") != preflight_report["frames"] or stored_preflight.get("channels") != preflight_report["channels"]:
        raise ValueError("raw preflight report가 preflight_raw_int32 재계산과 다릅니다")
    if int(stored_preflight.get("sample_rate", -1)) != sample_rate or not math.isclose(
        float(stored_preflight.get("settle_seconds", float("nan"))),
        mpi.cw.DEFAULT_PROBE_SETTLE_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("raw preflight sample-rate/settle witness가 다릅니다")
    resolved_devices = metadata.get("resolved_devices")
    if not isinstance(resolved_devices, dict) or not all(
        isinstance(resolved_devices.get(name), int) for name in ("input", "output")
    ):
        raise ValueError("raw resolved input/output device witness가 필요합니다")
    if int(stored_preflight.get("device", -1)) != int(resolved_devices["input"]):
        raise ValueError("preflight device와 capture resolved input이 다릅니다")

    xrun_count = _validate_clean_telemetry(
        metadata.get("telemetry"),
        expected_frames=expected_shape[0],
        sample_rate=sample_rate,
    )
    capture_clip_count = _clip_count_int32(input_raw)
    preflight_clip_count = _clip_count_int32(preflight_raw)
    output_clip_count = int(
        np.count_nonzero(
            np.abs(submitted.astype(np.int32)) >= np.iinfo(np.int16).max
        )
    )
    clip_count = capture_clip_count + preflight_clip_count + output_clip_count
    if clip_count != 0:
        raise ValueError(
            "broadband raw clip count가 0이 아닙니다: "
            f"capture={capture_clip_count}, preflight={preflight_clip_count}, output={output_clip_count}"
        )

    hardware_identity = metadata.get("hardware_identity")
    if not isinstance(hardware_identity, dict):
        raise ValueError("raw hardware_identity mapping이 필요합니다")
    level_binding = metadata.get("level_evidence")
    meter_binding = metadata.get("meter")
    if not isinstance(level_binding, dict) or not isinstance(meter_binding, dict):
        raise ValueError("raw level/meter binding이 필요합니다")
    level_path = _bound_metadata_path(
        level_binding.get("path"),
        repository_root=root,
        expected_path=None,
        label="measurement level evidence",
    )
    level = load_measurement_level_evidence(level_path, repository_root=root)
    level_sha = _require_sha256(level.get("_evidence_sha256"), label="verified level SHA")
    if level_sha != _require_sha256(level_binding.get("sha256"), label="raw level SHA"):
        raise ValueError("raw/verified measurement level evidence SHA가 다릅니다")
    if level.get("hardware_identity") != hardware_identity:
        raise ValueError("level evidence/raw hardware identity가 다릅니다")
    meter_path = _bound_metadata_path(
        meter_binding.get("path"),
        repository_root=root,
        expected_path=None,
        label="fresh bootstrap meter raw",
    )
    meter = validate_bootstrap_meter_raw(
        meter_path,
        repository_root=root,
        expected_hardware_identity=hardware_identity,
        require_fresh=False,
    )
    meter_sha = _require_sha256(meter.get("sha256"), label="verified meter raw SHA")
    if meter_sha != _require_sha256(meter_binding.get("raw_sha256"), label="raw meter SHA"):
        raise ValueError("raw/verified fresh meter SHA가 다릅니다")
    broadband_measure.validate_meter_followup_binding(
        meter_metadata=meter["metadata"],
        plan_binding={
            "path": str(plan_binding["path"]),
            "file_sha256": plan_file_sha,
            "payload_sha256": plan_payload_sha,
            "pcm_sha256": expected_pcm_sha,
        },
        hardware_path=bound_hardware,
        hardware_sha256=hardware_sha,
        level_evidence_path=level_path,
        level_evidence_sha256=level_sha,
        raw_session_dir=session,
    )
    if Path(meter["receipt_path"]).resolve() != _repository_path(
        meter_binding.get("receipt_path"), repository_root=root, label="meter receipt"
    ):
        raise ValueError("raw meter receipt path binding이 다릅니다")
    if Path(meter["path"]).resolve() != meter_path:
        raise ValueError("verified meter path가 raw binding과 다릅니다")
    if meter["metadata"].get("resolved_devices") != resolved_devices:
        raise ValueError("meter/capture PortAudio device mapping이 다릅니다")
    if meter["completed_at_utc"].isoformat() != meter_binding.get("completed_at_utc"):
        raise ValueError("meter completion timestamp가 raw binding과 다릅니다")
    if not math.isclose(
        float(meter["meter_ch0_dbfs"]),
        float(meter_binding.get("meter_ch0_dbfs", float("nan"))),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("meter ch0 dBFS가 raw binding과 다릅니다")
    try:
        stored_freshness = float(meter_binding["freshness_max_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("raw meter freshness_max_seconds witness가 필요합니다") from exc
    if not math.isclose(
        stored_freshness,
        float(BOOTSTRAP_METER_MAX_AGE_SECONDS),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("raw meter freshness max가 canonical 계약과 다릅니다")
    capture_started = _parse_utc(
        metadata.get("started_at_utc"), label="raw started_at_utc"
    )
    capture_completed = _parse_utc(
        metadata.get("completed_at_utc"), label="raw completed_at_utc"
    )
    stream_started = _parse_utc(
        metadata.get("telemetry", {}).get("stream_started_at_utc"),
        label="raw telemetry.stream_started_at_utc",
    )
    meter_completed = meter["completed_at_utc"].astimezone(dt.timezone.utc)
    meter_to_stream_seconds = (stream_started - meter_completed).total_seconds()
    if not 0.0 <= meter_to_stream_seconds <= stored_freshness:
        raise ValueError(
            "fresh meter→broadband stream freshness 위반: "
            f"{meter_to_stream_seconds:.3f}s"
        )
    if not capture_started <= stream_started <= capture_completed:
        raise ValueError("raw capture/stream start/completion UTC 순서가 잘못됐습니다")

    bound_files: dict[str, dict[str, str]] = {
        "raw_measurement": {"path": str(raw_path), "sha256": raw_sha},
        "raw_metadata_sidecar": {"path": str(sidecar_path), "sha256": sidecar_sha},
        "exact_plan": {"path": str(bound_plan), "sha256": plan_file_sha},
        "hardware": {"path": str(bound_hardware), "sha256": hardware_sha},
        "measurement_level_evidence": {"path": str(level_path), "sha256": level_sha},
        "fresh_meter_raw": {"path": str(meter_path), "sha256": meter_sha},
        "fresh_meter_receipt": {
            "path": str(Path(meter["receipt_path"]).resolve()),
            "sha256": _sha256_file(Path(meter["receipt_path"])),
        },
    }
    for key in ("meter_raw", "interleaved_raw"):
        entry = level.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"level evidence {key} binding이 필요합니다")
        bound = _repository_path(
            entry.get("path"), repository_root=root, label=f"level {key}"
        )
        bound_files[f"level_{key}"] = {
            "path": str(bound),
            "sha256": _require_sha256(entry.get("sha256"), label=f"level {key} SHA"),
        }
    _reverify_bound_files(bound_files)
    return {
        "repository_root": root,
        "session_dir": session,
        "raw_path": raw_path,
        "metadata_path": sidecar_path,
        "metadata": metadata,
        "raw_sha256": raw_sha,
        "metadata_sha256": sidecar_sha,
        "plan_path": bound_plan,
        "plan": plan,
        "plan_file_sha256": plan_file_sha,
        "plan_payload_sha256": plan_payload_sha,
        "plan_pcm_sha256": expected_pcm_sha,
        "hardware_path": bound_hardware,
        "hardware_sha256": hardware_sha,
        "level_evidence_path": level_path,
        "level_evidence_sha256": level_sha,
        "meter_path": meter_path,
        "meter_sha256": meter_sha,
        "meter_receipt_path": Path(meter["receipt_path"]).resolve(),
        "meter_receipt_sha256": bound_files["fresh_meter_receipt"]["sha256"],
        "meter_to_stream_seconds": meter_to_stream_seconds,
        "hardware_identity": hardware_identity,
        "submitted_output_pcm_int16": submitted,
        "input_raw_int32": input_raw,
        "preflight_raw_int32": preflight_raw,
        "callback_time_info": callback_time_info,
        "measurement_report": measurement_report,
        "preflight_report": preflight_report,
        "xrun_count": xrun_count,
        "clip_count": clip_count,
        "clip_breakdown": {
            "capture_input": capture_clip_count,
            "preflight_input": preflight_clip_count,
            "submitted_output": output_clip_count,
        },
        "bound_files": bound_files,
    }


def _same_band(actual: Any, expected: Sequence[float]) -> bool:
    try:
        values = tuple(float(value) for value in actual)
    except (TypeError, ValueError):
        return False
    return len(values) == 2 and all(
        math.isclose(value, float(reference), rel_tol=0.0, abs_tol=1e-9)
        for value, reference in zip(values, expected, strict=True)
    )


def _band_metric_vector(
    stitched: Mapping[str, Any],
    *,
    row_name: str,
    metric_name: str,
    reducer: str,
    contract: ControlBandContract,
    require_pass: bool = True,
) -> tuple[float, ...]:
    per_drive: list[list[float]] = []
    for drive in ("noise", "cancel"):
        rows = stitched["drives"][drive].get(row_name)
        if not isinstance(rows, (list, tuple)) or len(rows) != len(
            contract.point_control_subbands_hz
        ):
            raise ValueError(f"{drive} {row_name} 7-band row가 필요합니다")
        values: list[float] = []
        for index, (row, expected_band) in enumerate(
            zip(rows, contract.point_control_subbands_hz, strict=True)
        ):
            if not isinstance(row, Mapping) or not _same_band(
                row.get("band_hz"), expected_band
            ):
                raise ValueError(f"{drive} {row_name} #{index} band가 계약과 다릅니다")
            value = float(row.get(metric_name, float("nan")))
            if not math.isfinite(value) or (
                bool(require_pass) and row.get("passed") is not True
            ):
                raise ValueError(f"{drive} {row_name} #{index}가 PASS/finite가 아닙니다")
            values.append(value)
        per_drive.append(values)
    paired = zip(per_drive[0], per_drive[1], strict=True)
    if reducer == "min":
        return tuple(min(primary, secondary) for primary, secondary in paired)
    if reducer == "max":
        return tuple(max(primary, secondary) for primary, secondary in paired)
    raise ValueError(f"알 수 없는 band reducer: {reducer}")


def run_broadband_analysis(
    capture: Mapping[str, Any],
    *,
    contract: ControlBandContract | None = None,
) -> dict[str, Any]:
    """exact layout의 5개 panel을 raw에서 분석한 뒤 단일 P/S FIR로 stitch한다."""

    active_contract = contract or ControlBandContract.broadband_point_control()
    if active_contract.role != "broadband_point_control":
        raise ValueError("broadband analysis에 broadband point-control 계약이 필요합니다")
    metadata = capture["metadata"]
    plan = capture["plan"]
    if metadata.get("control_band_contract_sha256") != active_contract.digest():
        raise ValueError("capture/control contract SHA가 다릅니다")
    sample_rate = int(metadata["sample_rate"])
    output_pcm = np.asarray(capture["submitted_output_pcm_int16"])
    converted = pcm_int32_to_float32(capture["input_raw_int32"])
    err = converted[:, int(metadata["channel_map"]["error_mic"])].astype(
        np.float64
    )
    ref = converted[:, int(metadata["channel_map"]["reference_mic"])].astype(
        np.float64
    )
    rows = [
        row for row in plan.get("layout", []) if row.get("kind") == "analysis_panel"
    ]
    if len(rows) != len(active_contract.measurement_panels_hz):
        raise ValueError("exact plan에 analysis_panel 5개가 필요합니다")
    repeats = int(plan.get("recipe", {}).get("repeats_per_panel", -1))
    if repeats < 9:
        raise ValueError("panel당 반복이 clock witness 8개를 만들기에 부족합니다")

    marker_rows = {
        row.get("kind"): row
        for row in plan.get("layout", [])
        if row.get("kind")
        in {
            "primary_nonperiodic_timing_marker",
            "secondary_nonperiodic_timing_marker",
        }
    }
    if set(marker_rows) != {
        "primary_nonperiodic_timing_marker",
        "secondary_nonperiodic_timing_marker",
    }:
        raise ValueError("P/S channel-separated nonperiodic timing marker가 필요합니다")
    marker_reports: dict[str, dict[str, Any]] = {}
    for drive, kind, channel in (
        ("noise", "primary_nonperiodic_timing_marker", 0),
        ("cancel", "secondary_nonperiodic_timing_marker", 1),
    ):
        row = marker_rows[kind]
        if int(row.get("output_channel", -1)) != channel or row.get(
            "other_channel_silent"
        ) is not True:
            raise ValueError(f"{drive} timing marker channel/silence 계약이 다릅니다")
        marker_reports[drive] = estimate_nonperiodic_marker_delay(
            output_pcm=output_pcm,
            err=err,
            start_frame=int(row["start_frame"]),
            stop_frame=int(row["stop_frame"]),
            output_channel=channel,
        )
    marker_delay_bounds = {
        drive: (
            float(report["search_lower_samples"]),
            float(report["search_upper_samples"]),
        )
        for drive, report in marker_reports.items()
    }
    timing_marker_pcm_sha256 = _sha256_bytes(
        b"".join(
            np.asarray(
                output_pcm[int(marker_rows[kind]["start_frame"]) : int(marker_rows[kind]["stop_frame"])],
                dtype="<i2",
            ).tobytes(order="C")
            for kind in (
                "primary_nonperiodic_timing_marker",
                "secondary_nonperiodic_timing_marker",
            )
        )
    )

    periodic_kinds = {"warmup_panel_0", "analysis_panel", "lowband_clock_anchor"}
    periodic_rows = [
        row for row in plan.get("layout", []) if row.get("kind") in periodic_kinds
    ]
    transition_rows = [
        row for row in periodic_rows if row.get("kind") == "lowband_clock_anchor"
    ]
    if len(transition_rows) != 4 or any(
        int(row.get("frames", -1)) != 10 * 6_000 for row in transition_rows
    ):
        raise ValueError(
            "global clock transition anchor는 exact 4×(guard 1 + analysis 9) periods여야 합니다"
        )
    global_period_starts: list[int] = []
    row_period_ranges: dict[int, tuple[int, int]] = {}
    for row_index, row in enumerate(periodic_rows):
        start = int(row["start_frame"])
        frames = int(row["frames"])
        if frames <= 0 or frames % 6_000:
            raise ValueError("global clock periodic row가 6000-sample 정수배가 아닙니다")
        begin = len(global_period_starts)
        global_period_starts.extend(range(start, start + frames, 6_000))
        row_period_ranges[row_index] = (begin, len(global_period_starts))
    if np.any(np.diff(np.asarray(global_period_starts, dtype=np.int64)) != 6_000):
        raise ValueError("warmup부터 마지막 panel까지 global clock chain이 연속이 아닙니다")
    global_probe = broadband_measure.build_clock_piloted_panel_probe(
        sample_rate=sample_rate,
        period_seconds=float(plan["recipe"]["period_seconds"]),
        panel_band_hz=tuple(active_contract.measurement_panels_hz[0]),
        amplitude=float(plan["recipe"]["amplitude"]),
    )
    global_clock_observation = mpi.observe_period_clock_ratios(
        err=err,
        ref=ref,
        probe=global_probe,
        period_starts=global_period_starts,
        max_drift_deviation_samples=mpi.DEFAULT_MAX_DRIFT_DEVIATION_SAMPLES,
        min_valid_periods=8,
        clock_band_hz=BROADBAND_CLOCK_PILOT_BAND_HZ,
    )
    # 서로 다른 submitted PCM row 경계의 양쪽 period는 drift 부호에 따라 반대
    # row의 tail/head를 포함할 수 있다. 양의 drift만 가정해 새 row의 첫 period만
    # 버리면 음의 drift에서 이전 row의 마지막 period가 오염된다. 따라서 경계
    # 직전/직후 interval을 모두 버리고, transition 내부 8개 exact adjacent
    # witness만 authority로 사용한다.
    global_valid_mask = np.asarray(
        global_clock_observation["valid"], dtype=np.bool_
    ).copy()
    for row_index in range(1, len(periodic_rows)):
        begin, _ = row_period_ranges[row_index]
        global_valid_mask[begin - 1] = False
        global_valid_mask[begin] = False
    global_clock_observation["valid"] = global_valid_mask
    global_clock_observation["q"] = np.where(
        global_valid_mask,
        np.asarray(global_clock_observation["q"], dtype=np.float64),
        np.nan,
    )
    global_clock_map = derive_global_clock_map(
        period_starts=global_period_starts,
        common_delay_samples=global_clock_observation["common_delay_samples"],
        valid=global_clock_observation["valid"],
        period_samples=6_000,
    )
    pilot_validation_probe = broadband_measure.build_clock_piloted_panel_probe(
        sample_rate=sample_rate,
        period_seconds=float(plan["recipe"]["period_seconds"]),
        panel_band_hz=tuple(active_contract.measurement_panels_hz[0]),
        amplitude=float(plan["recipe"]["amplitude"]),
    )
    submitted_pilot_validation = validate_submitted_pilot_global_map(
        err=err,
        ref=ref,
        output_pcm_int16=output_pcm,
        probe=pilot_validation_probe,
        period_starts=global_period_starts,
        clock_observation=global_clock_observation,
        global_clock_map=global_clock_map,
    )
    global_valid = np.asarray(global_clock_observation["valid"], dtype=np.bool_)
    transition_anchor_valid_counts: list[int] = []
    for row_index, row in enumerate(periodic_rows):
        if row.get("kind") != "lowband_clock_anchor":
            continue
        begin, stop = row_period_ranges[row_index]
        count = int(global_valid[begin + 1 : stop - 1].sum())
        if count != 8:
            raise ValueError(
                f"transition anchor #{len(transition_anchor_valid_counts)} valid adjacent가 "
                f"{count}/8입니다"
            )
        transition_anchor_valid_counts.append(count)
    offset_by_start = {
        int(start): float(offset)
        for start, offset in zip(
            global_clock_map["period_starts"],
            global_clock_map["period_offsets_samples"],
            strict=True,
        )
    }

    panel_results: list[dict[str, Any]] = []
    period_starts_by_panel: list[np.ndarray] = []
    clock_valid_repeats: list[int] = []
    clock_min_scores: list[float] = []
    panel_summaries: list[dict[str, Any]] = []
    for index, (row, panel_band) in enumerate(
        zip(rows, active_contract.measurement_panels_hz, strict=True)
    ):
        if int(row.get("panel_index", -1)) != index:
            raise ValueError(f"analysis panel #{index} index가 다릅니다")
        probe = broadband_measure.build_clock_piloted_panel_probe(
            sample_rate=sample_rate,
            period_seconds=float(plan["recipe"]["period_seconds"]),
            panel_band_hz=tuple(panel_band),
            amplitude=float(plan["recipe"]["amplitude"]),
        )
        start = int(row.get("start_frame", -1))
        stop = int(row.get("stop_frame", -1))
        frames = int(row.get("frames", -1))
        expected_frames = repeats * int(probe.period_samples)
        if (
            start < 0
            or stop != start + frames
            or frames != expected_frames
            or stop > output_pcm.shape[0]
        ):
            raise ValueError(f"analysis panel #{index} layout/frame가 다릅니다")
        starts = start + np.arange(repeats, dtype=np.int64) * int(
            probe.period_samples
        )
        # 누적 drift 부호와 관계없이 앞/뒤 row의 PCM이 섞일 수 있는 panel
        # 양끝 period는 highband P/S transfer authority에서도 제외한다.
        period_authority_mask = np.ones(repeats, dtype=np.bool_)
        period_authority_mask[[0, -1]] = False
        result = analyse_panel_capture(
            err=err,
            ref=ref,
            output_pcm_int16=output_pcm,
            probe=probe,
            period_starts=starts.tolist(),
            panel_band_hz=tuple(panel_band),
            period_authority_mask=period_authority_mask,
        )
        if not _same_band(result.get("panel_band_hz"), panel_band):
            raise ValueError(f"panel #{index} analysis result band가 다릅니다")
        selection_for_offset = result.get("selection")
        if not isinstance(selection_for_offset, Mapping):
            raise ValueError(f"panel #{index} repeat selection witness가 없습니다")
        selected_anchor = int(selection_for_offset.get("anchor", -1))
        if not 0 <= selected_anchor < starts.size:
            raise ValueError(f"panel #{index} clock anchor index가 잘못됐습니다")
        nominal_anchor_start = int(starts[selected_anchor])
        if nominal_anchor_start not in offset_by_start:
            raise ValueError(f"panel #{index} anchor가 global clock chain에 없습니다")
        _apply_global_clock_offset(result, offset_by_start[nominal_anchor_start])
        separation = result.get("separation")
        if not isinstance(separation, Mapping):
            raise ValueError(f"panel #{index} clock separation witness가 없습니다")
        valid = np.asarray(separation.get("valid"), dtype=np.bool_).reshape(-1)
        err_score = np.asarray(separation.get("err_score"), dtype=np.float64).reshape(
            -1
        )
        ref_score = np.asarray(separation.get("ref_score"), dtype=np.float64).reshape(
            -1
        )
        if not (valid.size == err_score.size == ref_score.size == repeats):
            raise ValueError(f"panel #{index} clock vector 길이가 layout과 다릅니다")
        valid_count = int(valid.sum())
        if valid_count < 8:
            raise ValueError(f"panel #{index} clock valid repeat가 8개 미만입니다")
        selected_scores = np.concatenate((err_score[valid], ref_score[valid]))
        if selected_scores.size == 0 or not np.all(np.isfinite(selected_scores)):
            raise ValueError(f"panel #{index} clock score가 finite가 아닙니다")
        minimum_score = float(np.min(selected_scores))
        selection = result.get("selection")
        if not isinstance(selection, Mapping):
            raise ValueError(f"panel #{index} repeat selection witness가 없습니다")
        keep = np.asarray(selection.get("keep"), dtype=np.bool_).reshape(-1)
        if keep.size != repeats or np.any(keep & ~valid) or int(keep.sum()) < 8:
            raise ValueError(f"panel #{index} final keep가 clock-valid 8개 계약을 어겼습니다")
        panel_results.append(result)
        period_starts_by_panel.append(starts)
        clock_valid_repeats.append(valid_count)
        clock_min_scores.append(minimum_score)
        panel_summaries.append(
            {
                "panel_index": index,
                "band_hz": [float(value) for value in panel_band],
                "period_start_frame": int(starts[0]),
                "period_stop_frame": int(starts[-1] + probe.period_samples),
                "clock_valid_repeats": valid_count,
                "clock_min_adjacent_score_observed": minimum_score,
                "kept_repeats": int(keep.sum()),
                "primary_panel_consistency": float(
                    result["panel_consistency"]["noise"]
                ),
                "secondary_panel_consistency": float(
                    result["panel_consistency"]["cancel"]
                ),
                "relative_phase_jitter_samples": float(
                    result["relative_tau_max_abs_samples"]
                ),
            }
        )

    stitched = stitch_broadband_panels(
        panel_results,
        contract=active_contract,
        fir_length=BROADBAND_FIR_LENGTH,
        pre_roll_samples=BROADBAND_PRE_ROLL_SAMPLES,
        maximum_delay_samples=BROADBAND_MAXIMUM_DELAY_SAMPLES,
        marker_delay_bounds=marker_delay_bounds,
    )
    if stitched.get("status") != "PASS" or stitched.get(
        "control_band_contract_sha256"
    ) != active_contract.digest():
        raise ValueError("broadband stitch result가 PASS/current contract가 아닙니다")
    separation_agreement = _band_metric_vector(
        stitched,
        row_name="separation_subbands",
        metric_name="complex_agreement",
        reducer="min",
        contract=active_contract,
    )
    separation_error = _band_metric_vector(
        stitched,
        row_name="separation_subbands",
        metric_name="relative_error",
        reducer="max",
        contract=active_contract,
    )
    compact_agreement = _band_metric_vector(
        stitched,
        row_name="compact_subbands",
        metric_name="complex_agreement",
        reducer="min",
        contract=active_contract,
        require_pass=False,
    )
    compact_error = _band_metric_vector(
        stitched,
        row_name="compact_subbands",
        metric_name="relative_error",
        reducer="max",
        contract=active_contract,
        require_pass=False,
    )
    interpolation_agreement = _band_metric_vector(
        stitched,
        row_name="measured_interpolation_subbands",
        metric_name="complex_agreement",
        reducer="min",
        contract=active_contract,
    )
    interpolation_error = _band_metric_vector(
        stitched,
        row_name="measured_interpolation_subbands",
        metric_name="relative_error",
        reducer="max",
        contract=active_contract,
    )
    return {
        "panels": panel_results,
        "period_starts_by_panel": period_starts_by_panel,
        "panel_summaries": panel_summaries,
        "stitched": stitched,
        "clock_valid_repeats": tuple(clock_valid_repeats),
        "clock_min_adjacent_score_observed": tuple(clock_min_scores),
        "separation_crosscheck_agreement": separation_agreement,
        "separation_crosscheck_relative_error": separation_error,
        "compact_roundtrip_agreement": compact_agreement,
        "compact_roundtrip_relative_error": compact_error,
        "measured_interpolation_agreement": interpolation_agreement,
        "measured_interpolation_relative_error": interpolation_error,
        "timing_markers": marker_reports,
        "marker_delay_bounds": marker_delay_bounds,
        "global_clock_observation": global_clock_observation,
        "global_clock_map": global_clock_map,
        "global_clock_map_sha256": str(global_clock_map["sha256"]),
        "panel_clock_offsets_samples": tuple(
            float(panel["global_clock_offset_samples"]) for panel in panel_results
        ),
        "transition_anchor_valid_counts": tuple(transition_anchor_valid_counts),
        "clock_trajectory_agreement_samples": float(
            submitted_pilot_validation[
                "pairwise_trajectory_agreement_samples"
            ]
        ),
        "submitted_pilot_validation": submitted_pilot_validation,
        "fixed_clock_pilot_sha256": str(
            submitted_pilot_validation["submitted_pilot_spectra_sha256"]
        ),
        "intended_float_pilot_sha256": str(
            plan["recipe"]["fixed_clock_pilot_sha256"]
        ),
        "timing_marker_pcm_sha256": timing_marker_pcm_sha256,
        "applied_per_drive_phase_repair_samples": stitched[
            "applied_per_drive_phase_repair_samples"
        ],
    }


def _analysis_arrays(
    capture: Mapping[str, Any], analysis: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    contract = ControlBandContract.broadband_point_control()
    stitched = analysis["stitched"]
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(BROADBAND_ANALYSIS_SCHEMA),
        "method": np.asarray(BROADBAND_ANALYSIS_METHOD),
        "capture_id": np.asarray(str(capture["metadata"]["capture_id"])),
        "source_raw_npz_path": np.asarray(
            str(Path(capture["raw_path"]).relative_to(capture["repository_root"]))
        ),
        "source_raw_npz_sha256": np.asarray(str(capture["raw_sha256"])),
        "source_metadata_json_sha256": np.asarray(
            str(capture["metadata_sha256"])
        ),
        "exact_plan_path": np.asarray(
            str(Path(capture["plan_path"]).relative_to(capture["repository_root"]))
        ),
        "exact_plan_file_sha256": np.asarray(str(capture["plan_file_sha256"])),
        "exact_plan_payload_sha256": np.asarray(
            str(capture["plan_payload_sha256"])
        ),
        "exact_plan_pcm_sha256": np.asarray(str(capture["plan_pcm_sha256"])),
        "hardware_path": np.asarray(
            str(Path(capture["hardware_path"]).relative_to(capture["repository_root"]))
        ),
        "hardware_sha256": np.asarray(str(capture["hardware_sha256"])),
        "measurement_level_evidence_path": np.asarray(
            str(
                Path(capture["level_evidence_path"]).relative_to(
                    capture["repository_root"]
                )
            )
        ),
        "measurement_level_evidence_sha256": np.asarray(
            str(capture["level_evidence_sha256"])
        ),
        "fresh_meter_raw_path": np.asarray(
            str(Path(capture["meter_path"]).relative_to(capture["repository_root"]))
        ),
        "fresh_meter_raw_sha256": np.asarray(str(capture["meter_sha256"])),
        "fresh_meter_receipt_sha256": np.asarray(
            str(capture["meter_receipt_sha256"])
        ),
        "fresh_meter_to_stream_seconds": np.asarray(
            float(capture["meter_to_stream_seconds"]), dtype=np.float64
        ),
        "control_band_contract_sha256": np.asarray(contract.digest()),
        "sample_rate": np.asarray(48_000, dtype=np.int64),
        "block_size": np.asarray(256, dtype=np.int64),
        "latency": np.asarray("low"),
        "output_pcm_provenance": np.asarray(BROADBAND_OUTPUT_PCM_PROVENANCE),
        "observed_submitted_pcm": np.asarray(True, dtype=np.bool_),
        "xrun_count": np.asarray(int(capture["xrun_count"]), dtype=np.int64),
        "clip_count": np.asarray(int(capture["clip_count"]), dtype=np.int64),
        "excitation_panels_hz": np.asarray(
            contract.measurement_panels_hz, dtype=np.float64
        ),
        "verified_subbands_hz": np.asarray(
            contract.point_control_subbands_hz, dtype=np.float64
        ),
        "primary_consistency": np.asarray(
            stitched["primary_consistency"], dtype=np.float64
        ),
        "secondary_consistency": np.asarray(
            stitched["secondary_consistency"], dtype=np.float64
        ),
        "clock_valid_repeats": np.asarray(
            analysis["clock_valid_repeats"], dtype=np.int64
        ),
        "clock_min_adjacent_score_observed": np.asarray(
            analysis["clock_min_adjacent_score_observed"], dtype=np.float64
        ),
        "relative_phase_jitter_samples": np.asarray(
            stitched["relative_phase_jitter_samples"], dtype=np.float64
        ),
        "separation_crosscheck_agreement": np.asarray(
            analysis["separation_crosscheck_agreement"], dtype=np.float64
        ),
        "separation_crosscheck_relative_error": np.asarray(
            analysis["separation_crosscheck_relative_error"], dtype=np.float64
        ),
        "compact_roundtrip_agreement": np.asarray(
            analysis["compact_roundtrip_agreement"], dtype=np.float64
        ),
        "compact_roundtrip_relative_error": np.asarray(
            analysis["compact_roundtrip_relative_error"], dtype=np.float64
        ),
        "measured_interpolation_agreement": np.asarray(
            analysis["measured_interpolation_agreement"], dtype=np.float64
        ),
        "measured_interpolation_relative_error": np.asarray(
            analysis["measured_interpolation_relative_error"], dtype=np.float64
        ),
        "hardware_identity_json": np.asarray(
            _canonical_json_bytes(_json_safe(capture["hardware_identity"])).decode(
                "utf-8"
            )
        ),
        "measurement_report_json": np.asarray(
            _canonical_json_bytes(_json_safe(capture["measurement_report"])).decode(
                "utf-8"
            )
        ),
        "preflight_report_json": np.asarray(
            _canonical_json_bytes(_json_safe(capture["preflight_report"])).decode(
                "utf-8"
            )
        ),
        "panel_stitch_json": np.asarray(
            _canonical_json_bytes(_json_safe(stitched["panel_stitch"])).decode(
                "utf-8"
            )
        ),
        "timing_marker_pcm_sha256": np.asarray(
            str(analysis["timing_marker_pcm_sha256"])
        ),
        "fixed_clock_pilot_sha256": np.asarray(
            str(analysis["fixed_clock_pilot_sha256"])
        ),
        "intended_float_pilot_sha256": np.asarray(
            str(analysis["intended_float_pilot_sha256"])
        ),
        "submitted_pilot_validation_sha256": np.asarray(
            str(analysis["submitted_pilot_validation"]["sha256"])
        ),
        "submitted_pilot_spectra_sha256": np.asarray(
            str(
                analysis["submitted_pilot_validation"][
                    "submitted_pilot_spectra_sha256"
                ]
            )
        ),
        "submitted_pilot_cross_channel_null_sha256": np.asarray(
            str(
                analysis["submitted_pilot_validation"][
                    "cross_channel_null_sha256"
                ]
            )
        ),
        "submitted_pilot_cross_channel_max_absolute": np.asarray(
            float(
                analysis["submitted_pilot_validation"][
                    "cross_channel_null_maximum_absolute_observed"
                ]
            ),
            dtype=np.float64,
        ),
        "submitted_pilot_cross_channel_max_ratio": np.asarray(
            float(
                analysis["submitted_pilot_validation"][
                    "cross_channel_null_maximum_ratio_observed"
                ]
            ),
            dtype=np.float64,
        ),
        "submitted_pilot_validation_json": np.asarray(
            _canonical_json_bytes(
                _json_safe(analysis["submitted_pilot_validation"])
            ).decode("utf-8")
        ),
        "global_clock_input_domain": np.asarray(
            "actual_submitted_int16_period_spectrum_not_intended_float"
        ),
        "global_clock_map_sha256": np.asarray(
            str(analysis["global_clock_map_sha256"])
        ),
        "global_clock_slope_samples_per_sample": np.asarray(
            float(analysis["global_clock_map"]["slope_samples_per_sample"]),
            dtype=np.float64,
        ),
        "global_clock_intercept_samples": np.asarray(
            float(analysis["global_clock_map"]["intercept_samples"]),
            dtype=np.float64,
        ),
        "global_clock_max_residual_samples": np.asarray(
            float(analysis["global_clock_map"]["maximum_residual_samples"]),
            dtype=np.float64,
        ),
        "global_clock_period_starts": np.asarray(
            analysis["global_clock_map"]["period_starts"], dtype=np.int64
        ),
        "global_clock_period_offsets_samples": np.asarray(
            analysis["global_clock_map"]["period_offsets_samples"], dtype=np.float64
        ),
        "global_clock_residual_samples": np.asarray(
            analysis["global_clock_map"]["residual_samples"], dtype=np.float64
        ),
        "transition_anchor_valid_counts": np.asarray(
            analysis["transition_anchor_valid_counts"], dtype=np.int64
        ),
        "panel_clock_offsets_samples": np.asarray(
            analysis["panel_clock_offsets_samples"], dtype=np.float64
        ),
        "clock_trajectory_agreement_samples": np.asarray(
            analysis["clock_trajectory_agreement_samples"], dtype=np.float64
        ),
        "applied_per_drive_phase_repair_samples": np.asarray(
            analysis["applied_per_drive_phase_repair_samples"], dtype=np.float64
        ),
    }
    clock_array_names = (
        "valid",
        "q",
        "common_delay_samples",
        "err_delay_samples",
        "ref_delay_samples",
        "err_score",
        "ref_score",
        "err_subwindow_spread_samples",
        "ref_subwindow_spread_samples",
        "err_ref_delta_samples",
        "adjacent_change_samples",
        "drift_deviation_samples",
    )
    selection_array_names = (
        "keep",
        "common_alignment_taus",
        "relative_tau_kept",
    )
    for index, (panel, starts) in enumerate(
        zip(
            analysis["panels"],
            analysis["period_starts_by_panel"],
            strict=True,
        )
    ):
        prefix = f"panel_{index}"
        arrays[f"{prefix}_band_hz"] = np.asarray(
            panel["panel_band_hz"], dtype=np.float64
        )
        arrays[f"{prefix}_period_starts"] = np.asarray(starts, dtype=np.int64)
        arrays[f"{prefix}_primary_consistency"] = np.asarray(
            panel["panel_consistency"]["noise"], dtype=np.float64
        )
        arrays[f"{prefix}_secondary_consistency"] = np.asarray(
            panel["panel_consistency"]["cancel"], dtype=np.float64
        )
        arrays[f"{prefix}_relative_phase_jitter_samples"] = np.asarray(
            panel["relative_tau_max_abs_samples"], dtype=np.float64
        )
        separation = panel["separation"]
        for name in clock_array_names:
            if name not in separation:
                raise ValueError(f"panel #{index} separation에 {name}이 없습니다")
            arrays[f"{prefix}_clock_{name}"] = np.asarray(separation[name])
        for scalar_name in ("drift_samples_per_period", "drift_ppm"):
            arrays[f"{prefix}_clock_{scalar_name}"] = np.asarray(
                separation[scalar_name], dtype=np.float64
            )
        selection = panel["selection"]
        for name in selection_array_names:
            if name not in selection:
                raise ValueError(f"panel #{index} selection에 {name}이 없습니다")
            arrays[f"{prefix}_selection_{name}"] = np.asarray(selection[name])
        arrays[f"{prefix}_selection_anchor"] = np.asarray(
            selection["anchor"], dtype=np.int64
        )
        for drive in ("noise", "cancel"):
            frequency = np.asarray(panel["frequencies"][drive], dtype=np.float64)
            transfer = np.asarray(panel["transfers"][drive], dtype=np.complex128)
            checked = np.asarray(
                panel["crosscheck_transfers"][drive], dtype=np.complex128
            )
            aligned = np.asarray(
                panel["aligned_stacks"][drive], dtype=np.complex128
            )
            aligned_checked = np.asarray(
                panel["aligned_crosscheck_stacks"][drive], dtype=np.complex128
            )
            arrays[f"{prefix}_{drive}_frequencies_hz"] = frequency
            arrays[f"{prefix}_{drive}_transfer_real"] = transfer.real
            arrays[f"{prefix}_{drive}_transfer_imag"] = transfer.imag
            arrays[f"{prefix}_{drive}_crosscheck_real"] = checked.real
            arrays[f"{prefix}_{drive}_crosscheck_imag"] = checked.imag
            arrays[f"{prefix}_{drive}_aligned_stacks_real"] = aligned.real
            arrays[f"{prefix}_{drive}_aligned_stacks_imag"] = aligned.imag
            arrays[f"{prefix}_{drive}_aligned_crosscheck_stacks_real"] = (
                aligned_checked.real
            )
            arrays[f"{prefix}_{drive}_aligned_crosscheck_stacks_imag"] = (
                aligned_checked.imag
            )

    for drive in ("noise", "cancel"):
        result = stitched["drives"][drive]
        frequency = np.asarray(result["frequencies_hz"], dtype=np.float64)
        transfer = np.asarray(result["mean_transfer"], dtype=np.complex128)
        reconstructed = np.asarray(
            result["compact"]["reconstructed_transfer"], dtype=np.complex128
        )
        arrays[f"{drive}_frequencies_hz"] = frequency
        arrays[f"{drive}_mean_transfer_real"] = transfer.real
        arrays[f"{drive}_mean_transfer_imag"] = transfer.imag
        arrays[f"{drive}_observations_per_frequency"] = np.asarray(
            result["observations_per_frequency"], dtype=np.int64
        )
        arrays[f"{drive}_bulk_delay_fractional_samples"] = np.asarray(
            result["bulk_delay_fractional_samples"], dtype=np.float64
        )
        arrays[f"{drive}_bulk_delay_samples"] = np.asarray(
            result["bulk_delay_samples"], dtype=np.int64
        )
        arrays[f"{drive}_effective_delay_samples"] = np.asarray(
            result["effective_delay_samples"], dtype=np.int64
        )
        arrays[f"{drive}_fractional_effective_delay_samples"] = np.asarray(
            result["fractional_effective_delay_samples"], dtype=np.float64
        )
        arrays[f"{drive}_pre_roll_samples"] = np.asarray(
            result["pre_roll_samples"], dtype=np.int64
        )
        arrays[f"{drive}_fir"] = np.asarray(result["fir"], dtype=np.float32)
        arrays[f"{drive}_compact_reconstructed_real"] = reconstructed.real
        arrays[f"{drive}_compact_reconstructed_imag"] = reconstructed.imag
        arrays[f"{drive}_separation_agreement"] = np.asarray(
            [row["complex_agreement"] for row in result["separation_subbands"]],
            dtype=np.float64,
        )
        arrays[f"{drive}_separation_relative_error"] = np.asarray(
            [row["relative_error"] for row in result["separation_subbands"]],
            dtype=np.float64,
        )
        arrays[f"{drive}_compact_agreement"] = np.asarray(
            [row["complex_agreement"] for row in result["compact_subbands"]],
            dtype=np.float64,
        )
        arrays[f"{drive}_compact_relative_error"] = np.asarray(
            [row["relative_error"] for row in result["compact_subbands"]],
            dtype=np.float64,
        )
        arrays[f"{drive}_measured_interpolation_agreement"] = np.asarray(
            [
                row["complex_agreement"]
                for row in result["measured_interpolation_subbands"]
            ],
            dtype=np.float64,
        )
        arrays[f"{drive}_measured_interpolation_relative_error"] = np.asarray(
            [
                row["relative_error"]
                for row in result["measured_interpolation_subbands"]
            ],
            dtype=np.float64,
        )
        arrays[f"{drive}_compact_role"] = np.asarray(result["compact_role"])
        arrays[f"{drive}_compact_training_eligible"] = np.asarray(
            result["compact_training_eligible"], dtype=np.bool_
        )
        arrays[f"{drive}_compact_identifiability_sha256"] = np.asarray(
            result["compact_identifiability_sha256"]
        )
        arrays[f"{drive}_compact_identifiability_json"] = np.asarray(
            _canonical_json_bytes(result["compact_identifiability"]).decode("utf-8")
        )
    return arrays


def _build_evidence(
    capture: Mapping[str, Any],
    analysis: Mapping[str, Any],
    *,
    analysis_sha256: str,
) -> BroadbandPlantEvidence:
    contract = ControlBandContract.broadband_point_control()
    stitched = analysis["stitched"]
    capture_id = str(capture["metadata"]["capture_id"])
    primary = stitched["drives"]["noise"]
    secondary = stitched["drives"]["cancel"]
    if int(primary["pre_roll_samples"]) != int(secondary["pre_roll_samples"]):
        raise ValueError("P/S broadband compact FIR pre-roll이 다릅니다")
    plant_delays = PlantDelays(
        primary_delay_samples=int(primary["effective_delay_samples"]),
        secondary_delay_samples=int(secondary["effective_delay_samples"]),
        handoff_samples=BROADBAND_HANDOFF_EXTRA_SAMPLES,
        sample_rate=contract.sample_rate,
    )
    lead = plant_delays.lead()
    if lead.is_clamped:
        raise ValueError("광대역 digital-reference lead가 0으로 clamp됩니다")
    return BroadbandPlantEvidence(
        control_band_contract_sha256=contract.digest(),
        primary_capture_id=capture_id,
        secondary_capture_id=capture_id,
        primary_raw_sha256=str(capture["raw_sha256"]),
        secondary_raw_sha256=str(capture["raw_sha256"]),
        primary_analysis_sha256=analysis_sha256,
        secondary_analysis_sha256=analysis_sha256,
        exact_plan_file_sha256=str(capture["plan_file_sha256"]),
        exact_plan_payload_sha256=str(capture["plan_payload_sha256"]),
        exact_plan_pcm_sha256=str(capture["plan_pcm_sha256"]),
        measurement_level_evidence_sha256=str(
            capture["level_evidence_sha256"]
        ),
        fresh_meter_raw_sha256=str(capture["meter_sha256"]),
        fresh_meter_receipt_sha256=str(capture["meter_receipt_sha256"]),
        timing_marker_pcm_sha256=str(analysis["timing_marker_pcm_sha256"]),
        fixed_clock_pilot_sha256=str(analysis["fixed_clock_pilot_sha256"]),
        submitted_pilot_validation_sha256=str(
            analysis["submitted_pilot_validation"]["sha256"]
        ),
        submitted_pilot_cross_channel_null_sha256=str(
            analysis["submitted_pilot_validation"]["cross_channel_null_sha256"]
        ),
        submitted_pilot_cross_channel_max_absolute=float(
            analysis["submitted_pilot_validation"][
                "cross_channel_null_maximum_absolute_observed"
            ]
        ),
        submitted_pilot_cross_channel_max_ratio=float(
            analysis["submitted_pilot_validation"][
                "cross_channel_null_maximum_ratio_observed"
            ]
        ),
        global_clock_input_domain=(
            "actual_submitted_int16_period_spectrum_not_intended_float"
        ),
        global_clock_map_sha256=str(analysis["global_clock_map_sha256"]),
        global_clock_slope_samples_per_sample=float(
            analysis["global_clock_map"]["slope_samples_per_sample"]
        ),
        global_clock_intercept_samples=float(
            analysis["global_clock_map"]["intercept_samples"]
        ),
        global_clock_max_residual_samples=float(
            analysis["global_clock_map"]["maximum_residual_samples"]
        ),
        clock_trajectory_agreement_samples=float(
            analysis["clock_trajectory_agreement_samples"]
        ),
        transition_anchor_valid_counts=tuple(
            int(value) for value in analysis["transition_anchor_valid_counts"]
        ),
        callback_timing_valid=bool(
            capture["metadata"]["callback_timing"]["valid"]
        ),
        callback_sample_slip_count=int(
            capture["metadata"]["callback_timing"]["sample_slip_count"]
        ),
        panel_clock_offsets_samples=tuple(
            float(value) for value in analysis["panel_clock_offsets_samples"]
        ),
        applied_per_drive_phase_repair_samples=tuple(
            float(value)
            for value in analysis["applied_per_drive_phase_repair_samples"]
        ),
        primary_marker_delay_samples=float(
            analysis["timing_markers"]["noise"]["coarse_delay_samples"]
        ),
        secondary_marker_delay_samples=float(
            analysis["timing_markers"]["cancel"]["coarse_delay_samples"]
        ),
        primary_marker_branch_width_samples=float(
            analysis["timing_markers"]["noise"]["search_width_samples"]
        ),
        secondary_marker_branch_width_samples=float(
            analysis["timing_markers"]["cancel"]["search_width_samples"]
        ),
        primary_marker_alias_candidate_count=int(
            analysis["timing_markers"]["noise"]["alias_candidate_count"]
        ),
        secondary_marker_alias_candidate_count=int(
            analysis["timing_markers"]["cancel"]["alias_candidate_count"]
        ),
        primary_bulk_delay_fractional_samples=float(
            primary["bulk_delay_fractional_samples"]
        ),
        secondary_bulk_delay_fractional_samples=float(
            secondary["bulk_delay_fractional_samples"]
        ),
        primary_bulk_delay_samples=int(primary["bulk_delay_samples"]),
        secondary_bulk_delay_samples=int(secondary["bulk_delay_samples"]),
        primary_effective_delay_samples=int(primary["effective_delay_samples"]),
        secondary_effective_delay_samples=int(secondary["effective_delay_samples"]),
        pre_roll_samples=int(primary["pre_roll_samples"]),
        handoff_extra_samples=int(plant_delays.handoff_samples),
        derived_lead_samples=int(lead),
        panel_primary_minus_secondary_bulk_delay_samples=tuple(
            float(value)
            for value in stitched[
                "panel_primary_minus_secondary_bulk_delay_samples"
            ]
        ),
        panel_relative_delay_deviation_samples=tuple(
            float(value)
            for value in stitched["panel_relative_delay_deviation_samples"]
        ),
        sample_rate=int(capture["metadata"]["sample_rate"]),
        block_size=int(capture["metadata"]["block_size"]),
        latency=str(capture["metadata"]["latency"]),
        observed_submitted_pcm=True,
        excitation_panels_hz=contract.measurement_panels_hz,
        verified_subbands_hz=contract.point_control_subbands_hz,
        primary_consistency=tuple(
            float(value) for value in stitched["primary_consistency"]
        ),
        secondary_consistency=tuple(
            float(value) for value in stitched["secondary_consistency"]
        ),
        clock_valid_repeats=tuple(
            int(value) for value in analysis["clock_valid_repeats"]
        ),
        clock_min_adjacent_score_observed=tuple(
            float(value)
            for value in analysis["clock_min_adjacent_score_observed"]
        ),
        relative_phase_jitter_samples=tuple(
            float(value) for value in stitched["relative_phase_jitter_samples"]
        ),
        separation_crosscheck_agreement=tuple(
            float(value) for value in analysis["separation_crosscheck_agreement"]
        ),
        separation_crosscheck_relative_error=tuple(
            float(value)
            for value in analysis["separation_crosscheck_relative_error"]
        ),
        measured_interpolation_agreement=tuple(
            float(value) for value in analysis["measured_interpolation_agreement"]
        ),
        measured_interpolation_relative_error=tuple(
            float(value)
            for value in analysis["measured_interpolation_relative_error"]
        ),
        primary_compact_role=str(primary["compact_role"]),
        secondary_compact_role=str(secondary["compact_role"]),
        primary_compact_training_eligible=bool(
            primary["compact_training_eligible"]
        ),
        secondary_compact_training_eligible=bool(
            secondary["compact_training_eligible"]
        ),
        primary_compact_identifiability_sha256=str(
            primary["compact_identifiability_sha256"]
        ),
        secondary_compact_identifiability_sha256=str(
            secondary["compact_identifiability_sha256"]
        ),
        compact_roundtrip_agreement=tuple(
            float(value) for value in analysis["compact_roundtrip_agreement"]
        ),
        compact_roundtrip_relative_error=tuple(
            float(value) for value in analysis["compact_roundtrip_relative_error"]
        ),
        xrun_count=int(capture["xrun_count"]),
        clip_count=int(capture["clip_count"]),
    )


def _broadband_plant_arrays(
    *,
    role: str,
    capture: Mapping[str, Any],
    analysis: Mapping[str, Any],
    analysis_path: Path,
    analysis_sha256: str,
    evidence: BroadbandPlantEvidence,
) -> dict[str, np.ndarray]:
    if role not in {"primary", "secondary"}:
        raise ValueError(f"알 수 없는 broadband plant role: {role}")
    contract = ControlBandContract.broadband_point_control()
    drive = "noise" if role == "primary" else "cancel"
    output_channel = 0 if role == "primary" else 1
    result = analysis["stitched"]["drives"][drive]
    consistency = (
        evidence.primary_consistency
        if role == "primary"
        else evidence.secondary_consistency
    )
    separation_rows = result["separation_subbands"]
    compact_rows = result["compact_subbands"]
    transfer = np.asarray(result["mean_transfer"], dtype=np.complex128)
    transfer_sha = _sha256_bytes(transfer.tobytes(order="C"))
    evidence_bytes = _canonical_json_bytes(evidence.model_dump(mode="json"))
    evidence_sha = _sha256_bytes(evidence_bytes)
    root = Path(capture["repository_root"])
    return {
        "schema_version": np.asarray(BROADBAND_PLANT_ARTIFACT_SCHEMA),
        "plant_role": np.asarray(role),
        "role": np.asarray(role),
        "measured_band_training_eligible": np.asarray(False, dtype=np.bool_),
        "measured_band_training_status": np.asarray(
            "blocked_until_fullband_persistently_exciting_causal_history"
        ),
        "compact_role": np.asarray(result["compact_role"]),
        "compact_training_eligible": np.asarray(
            result["compact_training_eligible"], dtype=np.bool_
        ),
        "compact_identifiability_sha256": np.asarray(
            result["compact_identifiability_sha256"]
        ),
        "compact_identifiability_json": np.asarray(
            _canonical_json_bytes(result["compact_identifiability"]).decode("utf-8")
        ),
        "fir": np.asarray(result["fir"], dtype=np.float32),
        "fir_length": np.asarray(len(result["fir"]), dtype=np.int64),
        "delay_samples": np.asarray(
            result["effective_delay_samples"], dtype=np.int64
        ),
        "effective_delay_samples": np.asarray(
            result["effective_delay_samples"], dtype=np.int64
        ),
        "bulk_delay_samples": np.asarray(
            result["bulk_delay_samples"], dtype=np.int64
        ),
        "bulk_delay_fractional_samples": np.asarray(
            result["bulk_delay_fractional_samples"], dtype=np.float64
        ),
        "fractional_effective_delay_samples": np.asarray(
            result["fractional_effective_delay_samples"], dtype=np.float64
        ),
        "primary_bulk_delay_fractional_samples": np.asarray(
            evidence.primary_bulk_delay_fractional_samples, dtype=np.float64
        ),
        "secondary_bulk_delay_fractional_samples": np.asarray(
            evidence.secondary_bulk_delay_fractional_samples, dtype=np.float64
        ),
        "primary_bulk_delay_samples": np.asarray(
            evidence.primary_bulk_delay_samples, dtype=np.int64
        ),
        "secondary_bulk_delay_samples": np.asarray(
            evidence.secondary_bulk_delay_samples, dtype=np.int64
        ),
        "primary_effective_delay_samples": np.asarray(
            evidence.primary_effective_delay_samples, dtype=np.int64
        ),
        "secondary_effective_delay_samples": np.asarray(
            evidence.secondary_effective_delay_samples, dtype=np.int64
        ),
        "handoff_extra_samples": np.asarray(
            evidence.handoff_extra_samples, dtype=np.int64
        ),
        "derived_lead_samples": np.asarray(
            evidence.derived_lead_samples, dtype=np.int64
        ),
        "panel_primary_minus_secondary_bulk_delay_samples": np.asarray(
            evidence.panel_primary_minus_secondary_bulk_delay_samples,
            dtype=np.float64,
        ),
        "panel_relative_delay_deviation_samples": np.asarray(
            evidence.panel_relative_delay_deviation_samples, dtype=np.float64
        ),
        "pre_roll_samples": np.asarray(
            result["pre_roll_samples"], dtype=np.int64
        ),
        "delay_semantics": np.asarray("effective_zeros_before_compact_fir"),
        "sample_rate": np.asarray(48_000, dtype=np.int64),
        "calibration_block_size": np.asarray(256, dtype=np.int64),
        "calibration_latency": np.asarray("low"),
        "error_mic_channel": np.asarray(0, dtype=np.int64),
        "reference_mic_channel": np.asarray(1, dtype=np.int64),
        "noise_output_channel": np.asarray(0, dtype=np.int64),
        "cancel_output_channel": np.asarray(1, dtype=np.int64),
        "output_channel": np.asarray(drive),
        "output_channel_index": np.asarray(output_channel, dtype=np.int64),
        "method": np.asarray(BROADBAND_ANALYSIS_METHOD),
        "output_pcm_provenance": np.asarray(BROADBAND_OUTPUT_PCM_PROVENANCE),
        "observed_submitted_pcm": np.asarray(True, dtype=np.bool_),
        "amplitude": np.asarray(0.003, dtype=np.float64),
        "analysis_period_seconds": np.asarray(0.125, dtype=np.float64),
        "capture_id": np.asarray(str(capture["metadata"]["capture_id"])),
        "xrun_count": np.asarray(evidence.xrun_count, dtype=np.int64),
        "clip_count": np.asarray(evidence.clip_count, dtype=np.int64),
        "repeats": np.asarray(min(evidence.clock_valid_repeats), dtype=np.int64),
        "excitation_panels_hz": np.asarray(
            evidence.excitation_panels_hz, dtype=np.float64
        ),
        "excitation_band_hz": np.asarray(
            contract.point_control_target_hz, dtype=np.float64
        ),
        "verified_subbands_hz": np.asarray(
            evidence.verified_subbands_hz, dtype=np.float64
        ),
        "consistency_band_hz": np.asarray(
            contract.point_control_target_hz, dtype=np.float64
        ),
        "band_consistency_hz": np.asarray(
            contract.point_control_subbands_hz, dtype=np.float64
        ),
        "band_consistency": np.asarray(consistency, dtype=np.float64),
        "fullband_consistency": np.asarray(min(consistency), dtype=np.float64),
        "clock_valid_repeats": np.asarray(
            evidence.clock_valid_repeats, dtype=np.int64
        ),
        "clock_min_adjacent_score_observed": np.asarray(
            evidence.clock_min_adjacent_score_observed, dtype=np.float64
        ),
        "relative_phase_jitter_samples": np.asarray(
            evidence.relative_phase_jitter_samples, dtype=np.float64
        ),
        "separation_crosscheck_agreement": np.asarray(
            [row["complex_agreement"] for row in separation_rows],
            dtype=np.float64,
        ),
        "separation_crosscheck_relative_error": np.asarray(
            [row["relative_error"] for row in separation_rows], dtype=np.float64
        ),
        "compact_roundtrip_agreement": np.asarray(
            [row["complex_agreement"] for row in compact_rows], dtype=np.float64
        ),
        "compact_roundtrip_relative_error": np.asarray(
            [row["relative_error"] for row in compact_rows], dtype=np.float64
        ),
        "measured_interpolation_agreement": np.asarray(
            [
                row["complex_agreement"]
                for row in result["measured_interpolation_subbands"]
            ],
            dtype=np.float64,
        ),
        "measured_interpolation_relative_error": np.asarray(
            [
                row["relative_error"]
                for row in result["measured_interpolation_subbands"]
            ],
            dtype=np.float64,
        ),
        "measured_interpolation_receipt_json": np.asarray(
            _canonical_json_bytes(result["measured_interpolation_holdout"]).decode(
                "utf-8"
            )
        ),
        "worst_drive_separation_crosscheck_agreement": np.asarray(
            evidence.separation_crosscheck_agreement, dtype=np.float64
        ),
        "worst_drive_separation_crosscheck_relative_error": np.asarray(
            evidence.separation_crosscheck_relative_error, dtype=np.float64
        ),
        "worst_drive_compact_roundtrip_agreement": np.asarray(
            evidence.compact_roundtrip_agreement, dtype=np.float64
        ),
        "worst_drive_compact_roundtrip_relative_error": np.asarray(
            evidence.compact_roundtrip_relative_error, dtype=np.float64
        ),
        "tone_frequencies_hz": np.asarray(
            result["frequencies_hz"], dtype=np.float64
        ),
        "measured_frequencies_hz": np.asarray(
            result["frequencies_hz"], dtype=np.float64
        ),
        "measured_transfer_real": transfer.real,
        "measured_transfer_imag": transfer.imag,
        "aligned_mean_transfer_real": transfer.real,
        "aligned_mean_transfer_imag": transfer.imag,
        "aligned_mean_transfer_sha256": np.asarray(transfer_sha),
        "observations_per_frequency": np.asarray(
            result["observations_per_frequency"], dtype=np.int64
        ),
        "control_band_contract_sha256": np.asarray(contract.digest()),
        "source_raw_npz_path": np.asarray(
            str(Path(capture["raw_path"]).relative_to(root))
        ),
        "source_raw_npz_sha256": np.asarray(str(capture["raw_sha256"])),
        "source_analysis_npz_path": np.asarray(
            str(Path(analysis_path).relative_to(root))
        ),
        "source_analysis_npz_sha256": np.asarray(analysis_sha256),
        "source_analysis_sha256": np.asarray(analysis_sha256),
        "source_plan_path": np.asarray(
            str(Path(capture["plan_path"]).relative_to(root))
        ),
        "source_plan_file_sha256": np.asarray(str(capture["plan_file_sha256"])),
        "source_plan_payload_sha256": np.asarray(
            str(capture["plan_payload_sha256"])
        ),
        "source_plan_pcm_sha256": np.asarray(str(capture["plan_pcm_sha256"])),
        "source_hardware_path": np.asarray(
            str(Path(capture["hardware_path"]).relative_to(root))
        ),
        "source_hardware_sha256": np.asarray(str(capture["hardware_sha256"])),
        "measurement_level_evidence_path": np.asarray(
            str(Path(capture["level_evidence_path"]).relative_to(root))
        ),
        "measurement_level_evidence_sha256": np.asarray(
            str(capture["level_evidence_sha256"])
        ),
        "fresh_meter_raw_path": np.asarray(
            str(Path(capture["meter_path"]).relative_to(root))
        ),
        "fresh_meter_raw_sha256": np.asarray(str(capture["meter_sha256"])),
        "fresh_meter_receipt_sha256": np.asarray(
            str(capture["meter_receipt_sha256"])
        ),
        "broadband_plant_evidence_json": np.asarray(
            evidence_bytes.decode("utf-8")
        ),
        "broadband_plant_evidence_sha256": np.asarray(evidence_sha),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_identity(path: Path) -> tuple[int, int, int, str]:
    status = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"발행 temp/final은 symlink 없는 regular file이어야 합니다: {path}")
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_size),
        _sha256_file(path),
    )


def _unlink_if_owned(path: Path, identity: tuple[int, int, int, str]) -> bool:
    """rollback은 우리가 발행한 inode인 경우에만 이름을 제거한다."""

    try:
        current = _file_identity(path)
    except FileNotFoundError:
        return False
    except (OSError, ValueError):
        # 경쟁자가 symlink/다른 타입으로 교체했다면 절대 삭제하지 않는다.
        return False
    if current != identity:
        return False
    path.unlink()
    return True


def _write_npz_temp(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_temp(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())


def _publication_target(
    value: str | Path,
    *,
    repository_root: Path,
    suffix: str,
    label: str,
) -> Path:
    target = _repository_path(
        value, repository_root=repository_root, label=label
    )
    if target.suffix != suffix:
        raise ValueError(f"{label}는 {suffix} 파일이어야 합니다: {target}")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"{label} 상위 디렉토리가 없습니다: {target.parent}")
    return target


def _validate_publication_targets(
    targets: Mapping[str, Path], *, protected_paths: Sequence[Path]
) -> None:
    values = [path.resolve() for path in targets.values()]
    if len(set(values)) != len(values):
        raise ValueError("분석/P/S 발행 target은 모두 달라야 합니다")
    protected = {path.resolve() for path in protected_paths}
    for label, path in targets.items():
        if path.resolve() in protected:
            raise ValueError(f"{label} target이 immutable input과 같습니다: {path}")
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"기존 {label}은 덮어쓰지 않습니다: {path}")


def _analysis_metadata(
    *,
    capture: Mapping[str, Any],
    analysis: Mapping[str, Any],
    evidence: BroadbandPlantEvidence,
    audit: Any,
    analysis_npz_path: Path,
    analysis_npz_sha256: str,
    primary_path: Path | None,
    primary_sha256: str | None,
    secondary_path: Path | None,
    secondary_sha256: str | None,
) -> dict[str, Any]:
    stitched = analysis["stitched"]
    root = Path(capture["repository_root"])
    outputs: dict[str, Any] = {
        "analysis_npz": {
            "path": str(analysis_npz_path.relative_to(root)),
            "sha256": analysis_npz_sha256,
        }
    }
    if primary_path is not None and secondary_path is not None:
        outputs.update(
            {
                "primary_npz": {
                    "path": str(primary_path.relative_to(root)),
                    "sha256": primary_sha256,
                },
                "secondary_npz": {
                    "path": str(secondary_path.relative_to(root)),
                    "sha256": secondary_sha256,
                },
            }
        )
    return {
        "schema": BROADBAND_ANALYSIS_SCHEMA,
        "status": "PASS",
        "method": BROADBAND_ANALYSIS_METHOD,
        "capture_id": str(capture["metadata"]["capture_id"]),
        "same_capture_for_primary_secondary": True,
        "sample_rate": 48_000,
        "block_size": 256,
        "latency": "low",
        "output_pcm_provenance": BROADBAND_OUTPUT_PCM_PROVENANCE,
        "raw": {
            "path": str(Path(capture["raw_path"]).relative_to(root)),
            "sha256": str(capture["raw_sha256"]),
            "metadata_path": str(Path(capture["metadata_path"]).relative_to(root)),
            "metadata_sha256": str(capture["metadata_sha256"]),
            "xrun_count": int(capture["xrun_count"]),
            "clip_count": int(capture["clip_count"]),
            "clip_breakdown": dict(capture["clip_breakdown"]),
        },
        "plan": {
            "path": str(Path(capture["plan_path"]).relative_to(root)),
            "file_sha256": str(capture["plan_file_sha256"]),
            "payload_sha256": str(capture["plan_payload_sha256"]),
            "pcm_sha256": str(capture["plan_pcm_sha256"]),
        },
        "hardware": {
            "path": str(Path(capture["hardware_path"]).relative_to(root)),
            "sha256": str(capture["hardware_sha256"]),
            "identity": capture["hardware_identity"],
        },
        "measurement_level_evidence": {
            "path": str(Path(capture["level_evidence_path"]).relative_to(root)),
            "sha256": str(capture["level_evidence_sha256"]),
        },
        "fresh_meter": {
            "path": str(Path(capture["meter_path"]).relative_to(root)),
            "raw_sha256": str(capture["meter_sha256"]),
            "receipt_sha256": str(capture["meter_receipt_sha256"]),
            "meter_to_stream_seconds": float(
                capture["meter_to_stream_seconds"]
            ),
        },
        "control_band_contract_sha256": evidence.control_band_contract_sha256,
        "global_clock": {
            "map_sha256": str(analysis["global_clock_map_sha256"]),
            "input_domain": evidence.global_clock_input_domain,
            "submitted_pilot_validation_sha256": (
                evidence.submitted_pilot_validation_sha256
            ),
            "submitted_pilot_cross_channel_null_sha256": (
                evidence.submitted_pilot_cross_channel_null_sha256
            ),
            "submitted_pilot_cross_channel_max_absolute": (
                evidence.submitted_pilot_cross_channel_max_absolute
            ),
            "submitted_pilot_cross_channel_max_ratio": (
                evidence.submitted_pilot_cross_channel_max_ratio
            ),
            "maximum_residual_samples": float(
                analysis["global_clock_map"]["maximum_residual_samples"]
            ),
            "trajectory_agreement_samples": (
                evidence.clock_trajectory_agreement_samples
            ),
            "highband_phase_used_for_map": False,
        },
        "broadband_plant_evidence_sha256": _sha256_bytes(
            _canonical_json_bytes(evidence.model_dump(mode="json"))
        ),
        "panels": analysis["panel_summaries"],
        "panel_stitch": stitched["panel_stitch"],
        "timing": {
            "primary_bulk_delay_fractional_samples": (
                evidence.primary_bulk_delay_fractional_samples
            ),
            "secondary_bulk_delay_fractional_samples": (
                evidence.secondary_bulk_delay_fractional_samples
            ),
            "primary_bulk_delay_samples": evidence.primary_bulk_delay_samples,
            "secondary_bulk_delay_samples": evidence.secondary_bulk_delay_samples,
            "primary_effective_delay_samples": (
                evidence.primary_effective_delay_samples
            ),
            "secondary_effective_delay_samples": (
                evidence.secondary_effective_delay_samples
            ),
            "pre_roll_samples": evidence.pre_roll_samples,
            "handoff_extra_samples": evidence.handoff_extra_samples,
            "derived_lead_samples": evidence.derived_lead_samples,
            "panel_primary_minus_secondary_bulk_delay_samples": list(
                evidence.panel_primary_minus_secondary_bulk_delay_samples
            ),
            "panel_relative_delay_deviation_samples": list(
                evidence.panel_relative_delay_deviation_samples
            ),
        },
        "primary": {
            "bulk_delay_fractional_samples": float(
                stitched["drives"]["noise"]["bulk_delay_fractional_samples"]
            ),
            "bulk_delay_samples": int(
                stitched["drives"]["noise"]["bulk_delay_samples"]
            ),
            "effective_delay_samples": int(
                stitched["drives"]["noise"]["effective_delay_samples"]
            ),
            "canonical_representation": "measured_complex_tones",
            "measured_band_training_eligible": False,
            "compact_role": "diagnostic_only",
            "compact_identifiability": stitched["drives"]["noise"][
                "compact_identifiability"
            ],
        },
        "secondary": {
            "bulk_delay_fractional_samples": float(
                stitched["drives"]["cancel"]["bulk_delay_fractional_samples"]
            ),
            "bulk_delay_samples": int(
                stitched["drives"]["cancel"]["bulk_delay_samples"]
            ),
            "effective_delay_samples": int(
                stitched["drives"]["cancel"]["effective_delay_samples"]
            ),
            "canonical_representation": "measured_complex_tones",
            "measured_band_training_eligible": False,
            "compact_role": "diagnostic_only",
            "compact_identifiability": stitched["drives"]["cancel"][
                "compact_identifiability"
            ],
        },
        "broadband_plant_evidence": evidence.model_dump(mode="json"),
        "audit": audit.model_dump(mode="json"),
        "outputs": outputs,
    }


def publish_broadband_analysis(
    *,
    capture: Mapping[str, Any],
    analysis: Mapping[str, Any],
    analysis_npz_path: str | Path,
    analysis_json_path: str | Path,
    primary_path: str | Path | None = None,
    secondary_path: str | Path | None = None,
    publish: bool,
) -> dict[str, Any]:
    """audit PASS 이후에만 broadband analysis/P/S 네 파일을 no-replace 발행한다."""

    root = Path(capture["repository_root"]).resolve()
    analysis_npz = _publication_target(
        analysis_npz_path,
        repository_root=root,
        suffix=".npz",
        label="broadband analysis NPZ",
    )
    analysis_json = _publication_target(
        analysis_json_path,
        repository_root=root,
        suffix=".json",
        label="broadband analysis JSON",
    )
    targets: dict[str, Path] = {
        "broadband analysis NPZ": analysis_npz,
        "broadband analysis JSON": analysis_json,
    }
    primary: Path | None = None
    secondary: Path | None = None
    if publish:
        if primary_path is None or secondary_path is None:
            raise ValueError("--publish에는 --primary-out/--secondary-out이 모두 필요합니다")
        primary = _publication_target(
            primary_path,
            repository_root=root,
            suffix=".npz",
            label="broadband primary NPZ",
        )
        secondary = _publication_target(
            secondary_path,
            repository_root=root,
            suffix=".npz",
            label="broadband secondary NPZ",
        )
        targets["broadband primary NPZ"] = primary
        targets["broadband secondary NPZ"] = secondary
        _validate_publication_targets(
            targets,
            protected_paths=(capture["raw_path"], capture["metadata_path"]),
        )

    token = uuid.uuid4().hex
    analysis_temp = analysis_npz.parent / f".{analysis_npz.name}.{token}.partial"
    temporary_paths: list[Path] = [analysis_temp]
    promoted: list[tuple[Path, tuple[int, int, int, str]]] = []
    try:
        _write_npz_temp(analysis_temp, _analysis_arrays(capture, analysis))
        analysis_sha = _sha256_file(analysis_temp)
        evidence = _build_evidence(
            capture, analysis, analysis_sha256=analysis_sha
        )
        contract = ControlBandContract.broadband_point_control()
        audit = audit_broadband_plant_evidence(contract, evidence)
        audit.raise_if_blocked()
        _reverify_bound_files(capture["bound_files"])

        if not publish:
            metadata = _analysis_metadata(
                capture=capture,
                analysis=analysis,
                evidence=evidence,
                audit=audit,
                analysis_npz_path=analysis_npz,
                analysis_npz_sha256=analysis_sha,
                primary_path=None,
                primary_sha256=None,
                secondary_path=None,
                secondary_sha256=None,
            )
            return {
                "status": "PASS",
                "published": False,
                "analysis_sha256": analysis_sha,
                "evidence": evidence,
                "audit": audit,
                "metadata": metadata,
                "paths": {},
            }

        assert primary is not None and secondary is not None
        primary_temp = primary.parent / f".{primary.name}.{token}.partial"
        secondary_temp = secondary.parent / f".{secondary.name}.{token}.partial"
        analysis_json_temp = (
            analysis_json.parent / f".{analysis_json.name}.{token}.partial"
        )
        temporary_paths.extend((primary_temp, secondary_temp, analysis_json_temp))
        _write_npz_temp(
            primary_temp,
            _broadband_plant_arrays(
                role="primary",
                capture=capture,
                analysis=analysis,
                analysis_path=analysis_npz,
                analysis_sha256=analysis_sha,
                evidence=evidence,
            ),
        )
        _write_npz_temp(
            secondary_temp,
            _broadband_plant_arrays(
                role="secondary",
                capture=capture,
                analysis=analysis,
                analysis_path=analysis_npz,
                analysis_sha256=analysis_sha,
                evidence=evidence,
            ),
        )
        primary_sha = _sha256_file(primary_temp)
        secondary_sha = _sha256_file(secondary_temp)
        metadata = _analysis_metadata(
            capture=capture,
            analysis=analysis,
            evidence=evidence,
            audit=audit,
            analysis_npz_path=analysis_npz,
            analysis_npz_sha256=analysis_sha,
            primary_path=primary,
            primary_sha256=primary_sha,
            secondary_path=secondary,
            secondary_sha256=secondary_sha,
        )
        _write_json_temp(analysis_json_temp, metadata)
        analysis_json_sha = _sha256_file(analysis_json_temp)
        _reverify_bound_files(capture["bound_files"])
        _validate_publication_targets(
            targets,
            protected_paths=(capture["raw_path"], capture["metadata_path"]),
        )
        publication = (
            (analysis_temp, analysis_npz),
            (primary_temp, primary),
            (secondary_temp, secondary),
            # JSON receipt를 마지막에 노출한다. 정상 종료에서 JSON이
            # 보이면 앞선 analysis/P/S 세 파일이 먼저 fsync된 상태다.
            (analysis_json_temp, analysis_json),
        )
        for temporary, target in publication:
            identity = _file_identity(temporary)
            try:
                atomic_publish_noreplace(temporary, target)
            except BaseException:
                # helper가 hard-link를 만든 뒤 directory fsync에서 실패한
                # 경우까지 포함하되, 경쟁자가 교체한 target은 손대지 않는다.
                _unlink_if_owned(target, identity)
                raise
            promoted.append((target, identity))
            _fsync_directory(target.parent)
        expected_shas = {
            analysis_npz: analysis_sha,
            primary: primary_sha,
            secondary: secondary_sha,
            analysis_json: analysis_json_sha,
        }
        for path, expected_sha in expected_shas.items():
            if _sha256_file(path) != expected_sha:
                raise RuntimeError(f"발행 후 SHA 불일치: {path}")
        _reverify_bound_files(capture["bound_files"])
        return {
            "status": "PASS",
            "published": True,
            "analysis_sha256": analysis_sha,
            "evidence": evidence,
            "audit": audit,
            "metadata": metadata,
            "paths": {
                "analysis_npz": analysis_npz,
                "analysis_json": analysis_json,
                "primary": primary,
                "secondary": secondary,
            },
            "sha256": {
                "analysis_npz": analysis_sha,
                "analysis_json": _sha256_file(analysis_json),
                "primary": primary_sha,
                "secondary": secondary_sha,
            },
        }
    except BaseException:
        for path in temporary_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        for path, identity in promoted:
            try:
                _unlink_if_owned(path, identity)
            except OSError:
                pass
        cleanup_paths = [*temporary_paths, *(path for path, _ in promoted)]
        for parent in {path.parent for path in cleanup_paths}:
            try:
                _fsync_directory(parent)
            except OSError:
                pass
        raise
    finally:
        for path in temporary_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_session", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--hardware", type=Path, default=Path("configs/hardware_jetson.yaml")
    )
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--primary-out", type=Path)
    parser.add_argument("--secondary-out", type=Path)
    parser.add_argument("--analysis-npz", type=Path)
    parser.add_argument("--analysis-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        capture = load_broadband_raw_capture(
            session_dir=args.raw_session,
            plan_path=args.plan,
            hardware_path=args.hardware,
            repository_root=REPO_ROOT,
        )
        analysis = run_broadband_analysis(capture)
        analysis_npz = args.analysis_npz or (
            capture["session_dir"] / DEFAULT_ANALYSIS_NPZ_NAME
        )
        analysis_json = args.analysis_json or (
            capture["session_dir"] / DEFAULT_ANALYSIS_JSON_NAME
        )
        result = publish_broadband_analysis(
            capture=capture,
            analysis=analysis,
            analysis_npz_path=analysis_npz,
            analysis_json_path=analysis_json,
            primary_path=args.primary_out,
            secondary_path=args.secondary_out,
            publish=bool(args.publish),
        )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            f"[실패] 광대역 raw 분석/발행: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    if result["published"]:
        paths = result["paths"]
        print(
            "[PASS] 광대역 raw-derived P/S 발행\n"
            f"  analysis={paths['analysis_npz']}\n"
            f"  metadata={paths['analysis_json']}\n"
            f"  primary={paths['primary']}\n"
            f"  secondary={paths['secondary']}",
            file=sys.stderr,
        )
    else:
        print(
            "[PASS] 광대역 raw-derived audit-only; 파일은 발행하지 않았습니다. "
            f"analysis_sha256={result['analysis_sha256']}",
            file=sys.stderr,
        )
    return 0


__all__ = [
    "BROADBAND_ANALYSIS_SCHEMA",
    "BROADBAND_PLANT_ARTIFACT_SCHEMA",
    "analyse_panel_capture",
    "load_broadband_raw_capture",
    "publish_broadband_analysis",
    "run_broadband_analysis",
    "stitch_broadband_panels",
]


if __name__ == "__main__":
    raise SystemExit(main())
