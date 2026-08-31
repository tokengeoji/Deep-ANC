"""Stage-2 2 kHz same-capture P/S offline shared-q/MIMO analyzer.

이 모듈은 오디오 backend를 import하지 않는다. immutable 24초 signal raw만 받아
low-band full-capture shared affine q, fit-a/fit-b MIMO transfer cross-fit, untouched
holdout, nonaffine/change-point reject 및 compact P/S 후보를 계산한다.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize_scalar

from deep_anc.dsp.broadband_plant_analysis import (
    band_roundtrip_metrics,
    compact_fir_identifiability_receipt,
    estimate_bulk_delay_samples,
    fit_real_compact_fir,
)
from deep_anc.dsp.stage2_2khz_contract import (
    STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ,
    Stage2TwoKilohertzContract,
)
from deep_anc.data.repository_fd import publish_repository_bytes_noreplace

from .stage2_2khz_measurement import (
    MAX_TIMING_RESIDUAL_SAMPLES,
    MIN_RESPONSE_TO_NOISE_DB,
    MIN_SUBBAND_CONSISTENCY,
    SAMPLE_RATE,
    STAGE2_MEASUREMENT_RESULT_SCHEMA,
    Stage2MeasurementError,
    _array_sha256,
    _canonical_json_bytes,
    _safe_relative_path,
    admit_stage2_relative_ps_candidate,
    validate_submitted_pcm,
)
from .stage2_2khz_live import (
    load_stage2_meter_raw_bytes,
    load_stage2_raw_bytes,
    validate_stage2_duplex_telemetry,
)


ANALYSIS_SCHEMA = "stage2_2khz_shared_q_mimo_analysis_v1"
SHARED_Q_SCHEMA = "stage2_2khz_shared_affine_q_v1"
MAX_Q_PPM = 1_000.0
Q_SELECTION_BAND_HZ = (88.3883476483, 600.0)
Q_EPOCH_FRAMES = 2 * SAMPLE_RATE
ROBUSTNESS_EPOCH_FRAMES = SAMPLE_RATE
MAX_ACOUSTIC_DELAY_SAMPLES = 4_800
STFT_FRAMES = 16_384
STFT_HOP = 8_192
MIMO_MAX_INPUT_CONDITION = 20.0
COMPACT_FIR_LENGTH = 1_024
COMPACT_PRE_ROLL = 256


def _payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(dict(value))).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pcm_float(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.dtype("<i2"):
        return array.astype(np.float64) / 32768.0
    if array.dtype == np.dtype("<i4"):
        return array.astype(np.float64) / 2147483648.0
    raise Stage2MeasurementError("PCM dtype은 exact int16 또는 int32여야 합니다")


def _clock_phase_rows(
    submitted: np.ndarray, captured: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """각 2초 epoch의 low-band 2×2 MIMO transfer phase를 actual PCM에서 계산한다."""

    frequency = np.fft.rfftfreq(STFT_FRAMES, 1.0 / SAMPLE_RATE)
    band_indices = np.flatnonzero(
        (frequency >= Q_SELECTION_BAND_HZ[0])
        & (frequency <= Q_SELECTION_BAND_HZ[1])
    )
    phases: list[np.ndarray] = []
    magnitudes: list[np.ndarray] = []
    centers: list[float] = []
    worst_condition = 0.0
    for start in range(0, len(submitted), Q_EPOCH_FRAMES):
        stop = start + Q_EPOCH_FRAMES
        x = _stft_matrix(submitted[start:stop])
        y = _stft_matrix(captured[start:stop])
        transfer = np.zeros((4, band_indices.size), dtype=np.complex128)
        for output_offset, bin_index in enumerate(band_indices):
            design = x[:, bin_index, :]
            target = y[:, bin_index, :]
            singular = np.linalg.svd(design, compute_uv=False)
            condition = float(
                singular[0] / max(singular[-1], np.finfo(np.float64).tiny)
            )
            worst_condition = max(worst_condition, condition)
            gram = design.conj().T @ design
            regularization = 1.0e-10 * max(
                float(np.trace(gram).real / 2.0), np.finfo(np.float64).tiny
            )
            fitted = np.linalg.solve(
                gram + regularization * np.eye(2), design.conj().T @ target
            )
            transfer[:, output_offset] = fitted.reshape(-1)
        phases.append(np.angle(transfer))
        magnitudes.append(np.abs(transfer))
        centers.append(0.5 * (start + stop - 1) / SAMPLE_RATE)
    if not math.isfinite(worst_condition) or worst_condition > MIMO_MAX_INPUT_CONDITION:
        raise Stage2MeasurementError(
            "shared-q low-band MIMO input condition이 "
            f"{MIMO_MAX_INPUT_CONDITION:.1f} 초과입니다: {worst_condition:.3f}"
        )
    return (
        np.asarray(centers, dtype=np.float64),
        np.asarray(phases, dtype=np.float64),
        frequency[band_indices].astype(np.float64),
        np.asarray(magnitudes, dtype=np.float64),
        worst_condition,
    )


def _circular_clock_fit(
    times: np.ndarray,
    phase: np.ndarray,
    frequency: np.ndarray,
    magnitude: np.ndarray,
) -> tuple[float, float, dict[str, Any]]:
    """고정 LTI phase를 view×bin circular intercept로 profile-out한다."""

    weights = np.median(np.asarray(magnitude, dtype=np.float64), axis=0)
    positive = weights > np.finfo(np.float64).tiny
    if not np.all(positive) or not np.all(np.isfinite(weights)):
        raise Stage2MeasurementError("shared-q view/bin transfer magnitude가 finite positive가 아닙니다")
    weights = weights / float(np.mean(weights))

    def objective(ppm: float) -> float:
        correction = (
            2.0
            * np.pi
            * times[:, None, None]
            * frequency[None, None, :]
            * (float(ppm) * 1.0e-6)
        )
        phasor = np.exp(1j * (phase - correction))
        coherence = np.abs(np.mean(phasor, axis=0))
        return float(np.sum(weights * (1.0 - coherence)) / np.sum(weights))

    grid = np.linspace(-MAX_Q_PPM, MAX_Q_PPM, 2001, dtype=np.float64)
    values = np.asarray([objective(value) for value in grid], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < -1.0e-12):
        raise Stage2MeasurementError("shared-q circular likelihood가 finite nonnegative가 아닙니다")
    best = int(np.argmin(values))
    if best in {0, len(grid) - 1}:
        raise Stage2MeasurementError("shared-q circular likelihood optimum이 ±1000 ppm 경계입니다")
    fit = minimize_scalar(
        objective,
        bounds=(float(grid[best - 1]), float(grid[best + 1])),
        method="bounded",
        options={"xatol": 1.0e-9},
    )
    if not fit.success or not math.isfinite(float(fit.fun)):
        raise Stage2MeasurementError("shared-q circular likelihood refine가 실패했습니다")
    local = np.flatnonzero(
        (values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:])
    ) + 1
    ranked = sorted((float(values[index]), int(index)) for index in local)
    runner = ranked[1][0] if len(ranked) > 1 else None
    return float(fit.x), float(fit.fun), {
        "grid_step_ppm": 1.0,
        "grid_points": int(grid.size),
        "all_interior_basins_enumerated": True,
        "interior_basin_count": int(local.size),
        "runner_up_to_best_objective_ratio": (
            None
            if runner is None
            else float(runner / max(float(fit.fun), np.finfo(np.float64).tiny))
        ),
        "fixed_lti_phase_profiled_as_view_bin_intercept": True,
    }


def estimate_shared_affine_q(
    submitted_pcm: np.ndarray, captured_pcm: np.ndarray
) -> dict[str, Any]:
    """full-capture low-band aperiodic likelihood로 네 view 공통 q를 고정한다."""

    submitted = _pcm_float(submitted_pcm)
    captured = _pcm_float(captured_pcm)
    if submitted.shape != captured.shape or submitted.ndim != 2 or submitted.shape[1] != 2:
        raise Stage2MeasurementError("shared-q submitted/captured shape가 다릅니다")
    if len(submitted) % Q_EPOCH_FRAMES or len(submitted) < 3 * Q_EPOCH_FRAMES:
        raise Stage2MeasurementError("shared-q는 2초 epoch exact 3개 이상이어야 합니다")

    view_names = (
        "primary_ERR",
        "primary_REF",
        "secondary_ERR",
        "secondary_REF",
    )
    q = 1.0
    refinement_rows: list[dict[str, Any]] = []
    basin_audit: dict[str, Any] | None = None
    worst_condition = 0.0
    # q mismatch가 한 2초 STFT 안에서도 transfer를 퍼뜨리므로, 같은 shared objective를
    # fixed 4회 재적용한다. view별 보정은 한 번도 적용하지 않는다.
    for iteration in range(4):
        warped_iteration = _warp_submitted_to_capture(submitted_pcm, q)
        times, phase, frequency, magnitude, condition = _clock_phase_rows(
            warped_iteration, captured
        )
        delta_ppm, objective, iteration_basin = _circular_clock_fit(
            times, phase, frequency, magnitude
        )
        if basin_audit is None:
            basin_audit = iteration_basin
        q *= 1.0 + delta_ppm * 1.0e-6
        worst_condition = max(worst_condition, condition)
        refinement_rows.append(
            {
                "iteration": iteration,
                "shared_delta_ppm": delta_ppm,
                "objective": objective,
                "q_ratio_after_update": q,
            }
        )
    warped_final = _warp_submitted_to_capture(submitted_pcm, q)
    times, phase, frequency, magnitude, condition = _clock_phase_rows(
        warped_final, captured
    )
    residual_ppm, objective, _ = _circular_clock_fit(
        times, phase, frequency, magnitude
    )
    worst_condition = max(worst_condition, condition)
    q *= 1.0 + residual_ppm * 1.0e-6
    ppm = (q - 1.0) * 1.0e6
    if not math.isfinite(ppm) or abs(ppm) > MAX_Q_PPM:
        raise Stage2MeasurementError(
            f"shared q가 ±{MAX_Q_PPM:.0f} ppm search 내부가 아닙니다: {ppm:+.3f} ppm"
        )
    correction = (
        2.0
        * np.pi
        * times[:, None, None]
        * frequency[None, None, :]
        * (residual_ppm * 1.0e-6)
    )
    corrected = phase - correction
    intercept = np.angle(np.mean(np.exp(1j * corrected), axis=0))
    residual_phase = np.angle(np.exp(1j * (corrected - intercept[None, :, :])))
    residual_samples = (
        residual_phase
        / (2.0 * np.pi * frequency[None, None, :])
        * SAMPLE_RATE
    )
    fixed_weight = np.median(magnitude, axis=0)
    frequency_grid = np.broadcast_to(frequency[None, :], fixed_weight.shape)
    common_epoch_residual = (
        np.sum(
            fixed_weight[None, :, :]
            * frequency_grid[None, :, :]
            * residual_phase,
            axis=(1, 2),
        )
        * SAMPLE_RATE
        / (2.0 * np.pi * np.sum(fixed_weight * frequency_grid**2))
    )
    maximum_view_likelihood_residual = float(np.max(np.abs(residual_samples)))
    maximum_residual = float(np.max(np.abs(common_epoch_residual)))
    if maximum_residual > MAX_TIMING_RESIDUAL_SAMPLES:
        raise Stage2MeasurementError(
            "shared affine q common timing residual이 "
            f"{MAX_TIMING_RESIDUAL_SAMPLES:.6f} sample을 초과합니다: {maximum_residual:.6f}"
        )
    boundary_jumps = np.diff(common_epoch_residual)
    change_point = bool(
        boundary_jumps.size
        and np.max(np.abs(boundary_jumps)) > MAX_TIMING_RESIDUAL_SAMPLES
    )
    if change_point:
        raise Stage2MeasurementError("shared-q 2초 boundary에서 change-point/sample-slip을 검출했습니다")

    # 네 acoustic view에 따로 q를 허용하지 않는다. 각 view의 독립 slope를 진단용으로만
    # 계산하고, capture span 끝에서 shared slope와 0.270208 sample보다 벌어지면 reject한다.
    view_q_ppm: dict[str, float] = {}
    for view_index, name in enumerate(view_names):
        view_ppm, _, _ = _circular_clock_fit(
            times,
            phase[:, view_index : view_index + 1, :],
            frequency,
            magnitude[:, view_index : view_index + 1, :],
        )
        view_q_ppm[name] = view_ppm
    elapsed_frames = float((np.max(times) - np.min(times)) * SAMPLE_RATE)
    maximum_view_q_divergence = max(
        abs(value - residual_ppm) for value in view_q_ppm.values()
    ) * 1.0e-6 * elapsed_frames
    if maximum_view_q_divergence > MAX_TIMING_RESIDUAL_SAMPLES:
        raise Stage2MeasurementError(
            "view-specific q가 shared affine q와 capture span에서 "
            f"{MAX_TIMING_RESIDUAL_SAMPLES:.6f} sample보다 벌어집니다: "
            f"{maximum_view_q_divergence:.6f}"
        )

    payload: dict[str, Any] = {
        "schema": SHARED_Q_SCHEMA,
        "selection_input": "full_capture_low_band_known_stereo_codes_to_err_ref",
        "selection_band_hz": list(Q_SELECTION_BAND_HZ),
        "q_model": "single_affine",
        "q_ratio": q,
        "q_ppm": ppm,
        "search_bound_ppm": [-MAX_Q_PPM, MAX_Q_PPM],
        "search_boundary_optimum": False,
        "single_shared_q_for_all_p_s_err_ref_views": True,
        "circular_likelihood_objective": objective,
        "basin_audit": basin_audit,
        "shared_q_refinement_rows": refinement_rows,
        "final_residual_ppm_applied_to_q": residual_ppm,
        "maximum_timing_residual_samples": maximum_residual,
        "maximum_view_likelihood_residual_samples": maximum_view_likelihood_residual,
        "timing_residual_limit_samples": MAX_TIMING_RESIDUAL_SAMPLES,
        "ambiguity_envelope_validation_samples": maximum_residual,
        "epoch_frames": Q_EPOCH_FRAMES,
        "all_epoch_boundaries_tested": True,
        "change_point_detected": False,
        "nonaffine_drift_detected": False,
        "one_sample_insert_drop_detected": False,
        "view_specific_q_detected": False,
        "view_q_ppm_diagnostic": view_q_ppm,
        "maximum_view_q_divergence_samples": maximum_view_q_divergence,
        "observation_count": int(len(times) * len(view_names) * len(frequency)),
        "low_band_mimo_maximum_input_condition": worst_condition,
        "common_epoch_residual_samples": common_epoch_residual.tolist(),
        "common_epoch_boundary_jump_samples": boundary_jumps.tolist(),
        "epoch_center_seconds": times.tolist(),
    }
    payload["sha256"] = _payload_sha256(payload)
    return payload


def _warp_submitted_to_capture(submitted: np.ndarray, q: float) -> np.ndarray:
    source = _pcm_float(submitted)
    query = float(q) * np.arange(len(source), dtype=np.float64)
    grid = np.arange(len(source), dtype=np.float64)
    warped = np.column_stack(
        [np.interp(query, grid, source[:, channel], left=0.0, right=0.0) for channel in (0, 1)]
    )
    return warped


def _stft_matrix(signal: np.ndarray) -> np.ndarray:
    value = np.asarray(signal, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or len(value) < STFT_FRAMES:
        raise Stage2MeasurementError("MIMO STFT 입력 shape/길이가 잘못됐습니다")
    starts = np.arange(0, len(value) - STFT_FRAMES + 1, STFT_HOP, dtype=np.int64)
    window = np.hanning(STFT_FRAMES).astype(np.float64)
    frames = np.stack([value[start : start + STFT_FRAMES] * window[:, None] for start in starts])
    return np.fft.rfft(frames, axis=1)


def _fit_role_transfer(
    warped_input: np.ndarray,
    captured: np.ndarray,
    *,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    x = _stft_matrix(warped_input[start:stop])
    y = _stft_matrix(captured[start:stop])
    if x.shape != y.shape:
        raise Stage2MeasurementError("MIMO role STFT input/output shape가 다릅니다")
    frequency = np.fft.rfftfreq(STFT_FRAMES, 1.0 / SAMPLE_RATE)
    transfer = np.zeros((frequency.size, 2, 2), dtype=np.complex128)
    maximum_condition = 0.0
    for bin_index in range(frequency.size):
        design = x[:, bin_index, :]
        target = y[:, bin_index, :]
        gram = design.conj().T @ design
        singular = np.linalg.eigvalsh((gram + gram.conj().T) * 0.5)
        condition = float(singular[-1] / max(float(singular[0]), np.finfo(np.float64).tiny))
        maximum_condition = max(maximum_condition, condition)
        regularization = 1.0e-10 * max(float(np.trace(gram).real / 2.0), np.finfo(np.float64).tiny)
        transfer[bin_index] = np.linalg.solve(
            gram + regularization * np.eye(2), design.conj().T @ target
        )
    band = (frequency >= Q_SELECTION_BAND_HZ[0]) & (
        frequency <= STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ[-1][1]
    )
    band_conditions: list[float] = []
    for bin_index in np.flatnonzero(band):
        design = x[:, bin_index, :]
        singular = np.linalg.svd(design, compute_uv=False)
        band_conditions.append(float(singular[0] / max(singular[-1], np.finfo(np.float64).tiny)))
    worst_band_condition = max(band_conditions)
    if not math.isfinite(worst_band_condition) or worst_band_condition > MIMO_MAX_INPUT_CONDITION:
        raise Stage2MeasurementError(
            f"Stage-2 MIMO input condition이 {MIMO_MAX_INPUT_CONDITION:.1f} 초과입니다: "
            f"{worst_band_condition:.3f}"
        )
    return frequency, transfer, worst_band_condition


def _complex_agreement(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.complex128).reshape(-1)
    b = np.asarray(right, dtype=np.complex128).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0
    return float(abs(complex(np.vdot(a, b))) / denominator)


def _subband_rows(
    frequency: np.ndarray,
    fit_a: np.ndarray,
    fit_b: np.ndarray,
    holdout: np.ndarray,
) -> list[dict[str, Any]]:
    frozen = 0.5 * (fit_a + fit_b)
    rows: list[dict[str, Any]] = []
    for path_index, path in enumerate(("primary", "secondary")):
        for mic_index, microphone in enumerate(("ERR", "REF")):
            for subband_index, (lower, upper) in enumerate(
                STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ
            ):
                mask = (frequency >= lower) & (frequency < upper)
                if int(mask.sum()) < 8:
                    raise Stage2MeasurementError("Stage-2 subband STFT bin이 8개 미만입니다")
                a = fit_a[mask, path_index, mic_index]
                b = fit_b[mask, path_index, mic_index]
                h = holdout[mask, path_index, mic_index]
                f = frozen[mask, path_index, mic_index]
                fit_consistency = _complex_agreement(a, b)
                holdout_consistency = _complex_agreement(f, h)
                noise = max(
                    float(np.linalg.norm(a - b) / math.sqrt(2.0)),
                    float(np.linalg.norm(h - f)),
                    np.finfo(np.float64).tiny,
                )
                snr = 20.0 * math.log10(
                    float(np.linalg.norm(f)) / noise
                )
                row = {
                    "path": path,
                    "microphone": microphone,
                    "subband_index": subband_index,
                    "band_hz": [float(lower), float(upper)],
                    "fit_a_fit_b_consistency": fit_consistency,
                    "untouched_holdout_consistency": holdout_consistency,
                    "response_to_noise_db": snr,
                }
                if fit_consistency < MIN_SUBBAND_CONSISTENCY:
                    raise Stage2MeasurementError(
                        f"{path}/{microphone}/{lower:.1f}-{upper:.1f} fit consistency "
                        f"{fit_consistency:.6f}<0.95"
                    )
                if holdout_consistency < MIN_SUBBAND_CONSISTENCY:
                    raise Stage2MeasurementError(
                        f"{path}/{microphone}/{lower:.1f}-{upper:.1f} holdout consistency "
                        f"{holdout_consistency:.6f}<0.95"
                    )
                if snr < MIN_RESPONSE_TO_NOISE_DB:
                    raise Stage2MeasurementError(
                        f"{path}/{microphone}/{lower:.1f}-{upper:.1f} response-to-noise "
                        f"{snr:.3f}dB<20dB"
                    )
                rows.append(row)
    return rows


def _compact_path(
    frequency: np.ndarray, transfer: np.ndarray, *, label: str
) -> dict[str, Any]:
    upper = STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ[-1][1]
    mask = (frequency >= STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ[0][0]) & (
        frequency <= upper
    )
    selected_frequency = frequency[mask]
    selected_transfer = transfer[mask]
    bulk = estimate_bulk_delay_samples(
        selected_frequency,
        selected_transfer,
        sample_rate=SAMPLE_RATE,
        minimum_delay_samples=0.0,
        maximum_delay_samples=float(MAX_ACOUSTIC_DELAY_SAMPLES),
        coarse_resolution_samples=0.25,
    )
    bulk_integer = int(math.floor(bulk))
    effective = bulk_integer - COMPACT_PRE_ROLL
    if effective < 0:
        raise Stage2MeasurementError(f"{label} bulk delay가 compact pre-roll보다 짧습니다")
    # 1024-tap unrestricted FIR을 2.8 kHz 이하의 부분대역만으로 식별하면 측정되지
    # 않은 null-space가 남는다. expensive SVD 전에 이 명백한 production blocker를
    # 닫는다. fullband PE 식별 신호 또는 식별 가능한 저차 파라미터화가 필요하다.
    if float(selected_frequency[0]) > 0.0 or float(selected_frequency[-1]) < SAMPLE_RATE / 2.0:
        raise Stage2MeasurementError(
            f"{label} compact FIR identifiability 실패: measured_band="
            f"[{float(selected_frequency[0]):.6f},{float(selected_frequency[-1]):.6f}]Hz, "
            "unrestricted_1024tap_requires_fullband_PE_or_lower_identifiable_parameterization"
        )
    identifiability = compact_fir_identifiability_receipt(
        selected_frequency,
        effective_delay_samples=effective,
        fir_length=COMPACT_FIR_LENGTH,
        sample_rate=SAMPLE_RATE,
    )
    if identifiability["compact_training_eligible"] is not True:
        raise Stage2MeasurementError(
            f"{label} compact FIR identifiability 실패: "
            f"rank={identifiability['numeric_rank']}/{COMPACT_FIR_LENGTH}, "
            f"condition={identifiability['condition_number']}"
        )
    compact = fit_real_compact_fir(
        selected_frequency,
        selected_transfer,
        effective_delay_samples=effective,
        fir_length=COMPACT_FIR_LENGTH,
        sample_rate=SAMPLE_RATE,
    )
    roundtrip = band_roundtrip_metrics(
        selected_frequency,
        selected_transfer,
        np.asarray(compact["reconstructed_transfer"]),
        subbands_hz=STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ,
    )
    if compact["passed"] is not True or not all(row["passed"] for row in roundtrip):
        raise Stage2MeasurementError(f"{label} compact FIR round-trip gate가 실패했습니다")
    return {
        "fir": np.asarray(compact["fir"], dtype=np.float64),
        "delay_samples": effective,
        "bulk_delay_samples": bulk_integer,
        "bulk_delay_fractional_samples": bulk,
        "pre_roll_samples": COMPACT_PRE_ROLL,
        "delay_semantics": "delay_samples_zeros_before_compact_fir",
        "fractional_delay_encoding": "exactly_once_in_compact_fir",
        "sample_rate": SAMPLE_RATE,
        "consistency_band_hz": [
            STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ[0][0],
            upper,
        ],
        "compact_complex_agreement": compact["complex_agreement"],
        "compact_relative_error": compact["relative_error"],
        "compact_fir_identifiability_receipt": identifiability,
        "roundtrip_subbands": list(roundtrip),
    }


def _analyse_stage2_capture(
    plan: Mapping[str, Any],
    *,
    submitted_pcm: np.ndarray,
    captured_pcm: np.ndarray,
    raw_npz_sha256: str,
    capture_metadata: Mapping[str, Any],
    transport_counters: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """immutable Stage-2 raw를 relative P/S candidate까지 fail-closed 분석한다."""

    validate_submitted_pcm(plan, submitted_pcm)
    captured = np.asarray(captured_pcm)
    if captured.dtype != np.dtype("<i4") or captured.shape != np.asarray(submitted_pcm).shape:
        raise Stage2MeasurementError("Stage-2 captured PCM dtype/shape가 잘못됐습니다")
    if type(raw_npz_sha256) is not str or len(raw_npz_sha256) != 64:
        raise Stage2MeasurementError("Stage-2 raw NPZ SHA-256이 필요합니다")
    if capture_metadata.get("schema") != "stage2_2khz_same_capture_ps_raw_v1":
        raise Stage2MeasurementError("Stage-2 physical raw metadata가 아닙니다")
    if capture_metadata.get("physical_acoustic_capture") is not True:
        raise Stage2MeasurementError("synthetic/diagnostic raw를 physical Stage-2로 승격할 수 없습니다")

    q_receipt = estimate_shared_affine_q(submitted_pcm, captured)
    policy = plan.get("robustness_epoch_policy")
    if policy != {
        "epoch_frames": ROBUSTNESS_EPOCH_FRAMES,
        "total_nonoverlapping_epochs": 24,
        "minimum_kept_epochs_per_role": 8,
        "epochs_are_distinct_aperiodic_observations": True,
        "legacy_periodic_repeat_indices_applicable": False,
        "fake_repeat_index_synthesis_allowed": False,
    }:
        raise Stage2MeasurementError("Stage-2 24×1초 independent epoch policy가 다릅니다")
    epoch_rows: list[dict[str, Any]] = []
    common_residuals = q_receipt["common_epoch_residual_samples"]
    for epoch in range(24):
        start = epoch * ROBUSTNESS_EPOCH_FRAMES
        stop = start + ROBUSTNESS_EPOCH_FRAMES
        role_row = next(
            (
                row
                for row in plan["role_layout"]
                if int(row["start_frame_in_capture"]) <= start
                and stop <= int(row["stop_frame_in_capture"])
            ),
            None,
        )
        q_epoch = start // Q_EPOCH_FRAMES
        if role_row is None:
            raise Stage2MeasurementError("independent epoch가 role에 완전히 결박되지 않았습니다")
        residual = float(common_residuals[q_epoch])
        minimum_score = float(1.0 - q_receipt["circular_likelihood_objective"])
        kept = bool(
            math.isfinite(residual)
            and abs(residual) <= MAX_TIMING_RESIDUAL_SAMPLES
            and math.isfinite(minimum_score)
            and minimum_score > 0.0
        )
        epoch_rows.append(
            {
                "epoch_index": epoch,
                "role": str(role_row["role"]),
                "start_frame": start,
                "stop_frame": stop,
                "common_timing_residual_samples": residual,
                "minimum_known_code_likelihood_score": minimum_score,
                "kept": kept,
            }
        )
    kept_per_role = {
        role: sum(row["kept"] and row["role"] == role for row in epoch_rows)
        for role in ("fit_a", "fit_b", "untouched_holdout")
    }
    if any(value < 8 for value in kept_per_role.values()):
        raise Stage2MeasurementError("independent 1초 kept epoch가 role당 8개 미만입니다")
    warped = _warp_submitted_to_capture(submitted_pcm, float(q_receipt["q_ratio"]))
    captured_float = _pcm_float(captured)
    transfers: dict[str, np.ndarray] = {}
    conditions: dict[str, float] = {}
    frequency: np.ndarray | None = None
    for row in plan["role_layout"]:
        role = str(row["role"])
        role_frequency, transfer, condition = _fit_role_transfer(
            warped,
            captured_float,
            start=int(row["start_frame_in_capture"]),
            stop=int(row["stop_frame_in_capture"]),
        )
        if frequency is not None and not np.array_equal(frequency, role_frequency):
            raise Stage2MeasurementError("role별 STFT frequency grid가 다릅니다")
        frequency = role_frequency
        transfers[role] = transfer
        conditions[role] = condition
    assert frequency is not None
    rows = _subband_rows(
        frequency,
        transfers["fit_a"],
        transfers["fit_b"],
        transfers["untouched_holdout"],
    )
    frozen = 0.5 * (transfers["fit_a"] + transfers["fit_b"])
    primary = _compact_path(frequency, frozen[:, 0, 0], label="primary/ERR")
    secondary = _compact_path(frequency, frozen[:, 1, 0], label="secondary/ERR")

    required_transport = {"xrun", "clip", "callback_status"}
    if set(transport_counters) != required_transport:
        raise Stage2MeasurementError("raw-derived transport counter key가 exact하지 않습니다")
    if any(
        type(transport_counters[name]) is not int or transport_counters[name] != 0
        for name in required_transport
    ):
        raise Stage2MeasurementError("raw-derived xrun/clip/callback status는 exact 0이어야 합니다")
    counters = {
        "xrun": int(transport_counters["xrun"]),
        "clip": int(transport_counters["clip"]),
        "callback_status": int(transport_counters["callback_status"]),
        "sample_slip": 0,
        "sample_drop": 0,
        "sample_add": 0,
    }
    receipt: dict[str, Any] = {
        "schema": STAGE2_MEASUREMENT_RESULT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "actual_submitted_pcm_sha256": plan["actual_submitted_pcm"]["sha256"],
        "same_capture_ps": True,
        "raw_publish_mode": "no_replace",
        "analysis_publish_mode": "no_replace",
        "raw_capture_sha256": raw_npz_sha256,
        "counters": counters,
        "holdout_policy": {
            "used_for_fit": False,
            "used_for_support_selection": False,
            "used_for_threshold_selection": False,
            "used_for_predeclared_shared_q_nuisance_likelihood": True,
            "evaluated_after_fit_frozen": True,
        },
        "clock_witness": {
            "kind": "submitted_aperiodic_shared_q_acoustic_likelihood",
            "continuous_frames": int(plan["signal_frames"]),
            "gap_frames": 0,
            "single_shared_q_for_all_p_s_err_ref_views": True,
            "selection_input": q_receipt["selection_input"],
            "selection_band_hz": q_receipt["selection_band_hz"],
            "q_model": "single_affine",
            "search_boundary_optimum": False,
            "ambiguity_envelope_validation_samples": q_receipt[
                "ambiguity_envelope_validation_samples"
            ],
            "maximum_timing_residual_samples": q_receipt[
                "maximum_timing_residual_samples"
            ],
            "counters": counters,
        },
        "absolute_transport_claims": {
            "absolute_hardware_frame_identity_claimed": False,
            "callback_before_start_drop_observed_claimed": False,
            "hardware_counter_slip_zero_claimed": False,
            "relative_ps_lead_only": True,
        },
        "nonaffine_change_point_audit": {
            "transport_256_callback_contiguity_tested": True,
            "transport_semantics": "software_accounting_not_absolute_hardware_slip",
            "acoustic_q_epoch_frames": Q_EPOCH_FRAMES,
            "all_acoustic_q_epoch_boundaries_tested": True,
            "all_256_frame_acoustic_q_boundaries_tested": False,
            "change_point_detected": False,
            "nonaffine_drift_detected": False,
            "one_sample_insert_drop_detected": False,
            "view_specific_q_detected": False,
            "affine_model_frozen_before_holdout": True,
            "holdout_failure_refit_performed": False,
        },
        "relative_delay_scope": {
            "playback_to_err_acoustic_delay_included_in_primary": True,
            "playback_to_err_acoustic_delay_included_in_secondary": True,
            "common_intercept_claimed_separately": False,
            "common_time_gauge_cancels_in_p_minus_s": True,
            "manual_lead_allowed": False,
        },
        "path_subbands": rows,
        "thresholds_relaxed": False,
        "capture_generation": {
            "adapter_schema": "stage2_2khz_live_capture_adapter_v1",
            "reviewed_live_adapter_implemented": True,
            "physical_acoustic_capture": True,
            "synthetic_or_diagnostic": False,
            "clean_exact_commit": bool(capture_metadata.get("clean_exact_commit")),
            "native_raw_published_no_replace": True,
        },
    }
    admission = admit_stage2_relative_ps_candidate(plan, receipt)
    analysis: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "status": "RELATIVE_PS_TRAINING_PLANT_CANDIDATE_PASS",
        "plan_sha256": plan["plan_sha256"],
        "raw_capture_sha256": raw_npz_sha256,
        "submitted_pcm_sha256": _array_sha256(np.asarray(submitted_pcm)),
        "captured_pcm_sha256": _array_sha256(captured),
        "authority_scope": "single_point_relative_ps_lead_only",
        "absolute_hardware_frame_clock_authority": False,
        "shared_q": q_receipt,
        "mimo_input_condition_by_role": conditions,
        "robustness_epochs": {
            "epoch_frames": ROBUSTNESS_EPOCH_FRAMES,
            "total_epoch_count": len(epoch_rows),
            "rows": epoch_rows,
            "kept_per_role": kept_per_role,
            "minimum_kept_epochs_per_role": 8,
            "legacy_repeat_indices_applicable": False,
        },
        "path_subbands": rows,
        "primary": {key: value for key, value in primary.items() if key != "fir"},
        "secondary": {key: value for key, value in secondary.items() if key != "fir"},
        "relative_p_minus_s_bulk_delay_samples": float(
            primary["bulk_delay_fractional_samples"]
            - secondary["bulk_delay_fractional_samples"]
        ),
        "typed_admission": admission,
        "automatic_training_config_update_allowed": False,
    }
    analysis["analysis_sha256"] = _payload_sha256(analysis)
    arrays = {
        "frequency_hz": frequency.astype(np.float64),
        "fit_a_transfer": transfers["fit_a"].astype(np.complex128),
        "fit_b_transfer": transfers["fit_b"].astype(np.complex128),
        "untouched_holdout_transfer": transfers["untouched_holdout"].astype(np.complex128),
        "primary_fir": np.asarray(primary["fir"], dtype=np.float64),
        "secondary_fir": np.asarray(secondary["fir"], dtype=np.float64),
    }
    return analysis, arrays


def analyse_stage2_raw_bytes(
    plan: Mapping[str, Any], payload: bytes
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """self-attested metric이 아니라 native raw arrays에서 모든 gate를 재계산한다."""

    loaded = load_stage2_raw_bytes(payload)
    metadata = loaded["metadata"]
    submitted = loaded["submitted_pcm"]
    captured = loaded["captured_pcm"]
    if metadata.get("plan_sha256") != plan.get("plan_sha256"):
        raise Stage2MeasurementError("Stage-2 raw가 exact plan SHA에 결박되지 않았습니다")
    execution = metadata.get("execution_identity")
    if not isinstance(execution, Mapping):
        raise Stage2MeasurementError("Stage-2 raw에 clean execution identity가 없습니다")
    commit = execution.get("repository_commit")
    if (
        type(commit) is not str
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or execution.get("repository_dirty") is not False
    ):
        raise Stage2MeasurementError("Stage-2 raw execution identity가 clean exact commit이 아닙니다")
    scalar = metadata.get("telemetry_scalar")
    if not isinstance(scalar, Mapping):
        raise Stage2MeasurementError("Stage-2 raw에 native telemetry scalar가 없습니다")
    telemetry = dict(scalar)
    telemetry.update(loaded["telemetry_arrays"])
    telemetry["actual_submitted_pcm"] = submitted.copy()
    telemetry["capture_valid_mask"] = loaded["telemetry_arrays"]["capture_valid_mask"]
    telemetry["submitted_valid_mask"] = loaded["telemetry_arrays"]["submitted_valid_mask"]
    # 저장된 receipt를 신뢰하지 않고 raw callback arrays와 PCM을 다시 검증한다.
    validate_stage2_duplex_telemetry(
        telemetry,
        captured_adc_pcm=captured,
        expected_submitted_pcm=submitted,
    )
    clip_count = int(
        np.count_nonzero(np.abs(captured.astype(np.int64)) >= 2**31 - 1)
    )
    if clip_count:
        raise Stage2MeasurementError("Stage-2 raw ADC clipping을 직접 검출했습니다")
    physical_metadata = dict(metadata)
    physical_metadata["clean_exact_commit"] = True
    return _analyse_stage2_capture(
        plan,
        submitted_pcm=submitted,
        captured_pcm=captured,
        raw_npz_sha256=loaded["raw_npz_sha256"],
        capture_metadata=physical_metadata,
        transport_counters={
            "xrun": int(telemetry["xrun_count"]),
            "clip": clip_count,
            "callback_status": int(telemetry["status_present_count"]),
        },
    )


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


def publish_stage2_analysis_no_replace(
    repository_root: str,
    plan: Mapping[str, Any],
    analysis: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    native_raw_payload: bytes,
    meter_raw_payload: bytes,
) -> dict[str, Any]:
    """native raw 재계산 뒤 production raw/analysis/P/S/binding을 no-replace 발행한다."""

    core = dict(analysis)
    claimed = core.pop("analysis_sha256", None)
    if claimed != _payload_sha256(core):
        raise Stage2MeasurementError("Stage-2 analysis core SHA가 payload와 다릅니다")
    required_arrays = {
        "frequency_hz",
        "fit_a_transfer",
        "fit_b_transfer",
        "untouched_holdout_transfer",
        "primary_fir",
        "secondary_fir",
    }
    if set(arrays) != required_arrays:
        raise Stage2MeasurementError("Stage-2 analysis array key set이 exact하지 않습니다")
    if np.asarray(arrays["primary_fir"]).dtype != np.float64 or np.asarray(
        arrays["secondary_fir"]
    ).dtype != np.float64:
        raise Stage2MeasurementError("Stage-2 candidate FIR은 exact float64여야 합니다")
    native = load_stage2_raw_bytes(native_raw_payload)
    meter = load_stage2_meter_raw_bytes(plan, meter_raw_payload)
    if native["raw_npz_sha256"] != analysis.get("raw_capture_sha256"):
        raise Stage2MeasurementError("analysis가 exact native raw bytes에 결박되지 않았습니다")
    if meter["admission"]["passed"] is not True:
        raise Stage2MeasurementError("official meter raw가 -50.1±2 dBFS PASS가 아닙니다")
    native_metadata = native["metadata"]
    meter_metadata = meter["metadata"]
    native_execution = native_metadata.get("execution_identity")
    meter_execution = meter_metadata.get("execution_identity")
    if not isinstance(native_execution, Mapping) or not isinstance(
        meter_execution, Mapping
    ):
        raise Stage2MeasurementError("meter/signal raw execution identity가 없습니다")
    source_capture_commit_sha = native_execution.get("repository_commit")
    if (
        type(source_capture_commit_sha) is not str
        or len(source_capture_commit_sha) != 40
        or any(character not in "0123456789abcdef" for character in source_capture_commit_sha)
        or native_execution.get("repository_dirty") is not False
        or meter_execution.get("repository_commit") != source_capture_commit_sha
        or meter_execution.get("repository_dirty") is not False
    ):
        raise Stage2MeasurementError("meter/signal source capture commit이 clean exact로 일치하지 않습니다")
    capture_id = native_metadata.get("capture_id")
    if (
        type(capture_id) is not str
        or not capture_id
        or meter_metadata.get("capture_id") != capture_id
        or native_metadata.get("same_amplifier_setting_meter_to_signal") is not True
        or meter_metadata.get("same_amplifier_setting_meter_to_signal") is not True
    ):
        raise Stage2MeasurementError("meter/signal capture ID 또는 same-amplifier 결박이 다릅니다")
    scalar = native_metadata.get("telemetry_scalar")
    if not isinstance(scalar, Mapping):
        raise Stage2MeasurementError("native raw telemetry scalar가 없습니다")
    xrun_count = int(scalar.get("xrun_count", -1))
    callback_status = int(scalar.get("status_present_count", -1))
    clip_count = int(
        np.count_nonzero(
            np.abs(native["captured_pcm"].astype(np.int64)) >= 2**31 - 1
        )
    )
    q = analysis.get("shared_q")
    if not isinstance(q, Mapping) or any(
        q.get(key) is not False
        for key in (
            "change_point_detected",
            "nonaffine_drift_detected",
            "one_sample_insert_drop_detected",
            "view_specific_q_detected",
        )
    ):
        raise Stage2MeasurementError("shared-q slip/nonaffine/view-specific gate가 PASS가 아닙니다")
    sample_slip_count = 0
    if (xrun_count, callback_status, clip_count) != (0, 0, 0):
        raise Stage2MeasurementError("native raw xrun/status/clip이 exact 0이 아닙니다")

    contract = Stage2TwoKilohertzContract.canonical()
    if plan.get("contract") != {
        "id": contract.contract_id,
        "sha256": contract.digest(),
        "payload": contract.model_dump(mode="json"),
    }:
        raise Stage2MeasurementError("plan의 Stage-2 contract payload/SHA가 canonical과 다릅니다")
    submitted = np.ascontiguousarray(native["submitted_pcm"], dtype="<i2")
    captured = np.ascontiguousarray(native["captured_pcm"], dtype="<i4")
    submitted_plain_sha = hashlib.sha256(submitted.tobytes(order="C")).hexdigest()

    canonical_raw_payload = _npz_bytes(
        stage2_raw_schema=np.asarray("stage2_2khz_raw_capture_npz_v1"),
        stage2_contract_sha256=np.asarray(contract.digest()),
        capture_id=np.asarray(capture_id),
        sample_rate=np.asarray(SAMPLE_RATE, dtype=np.int64),
        block_size=np.asarray(256, dtype=np.int64),
        submitted_output_pcm=submitted,
        captured_input_pcm=captured,
        xrun_count=np.asarray(xrun_count, dtype=np.int64),
        clip_count=np.asarray(clip_count, dtype=np.int64),
        sample_slip_count=np.asarray(sample_slip_count, dtype=np.int64),
        callback_status_failures=np.asarray(callback_status, dtype=np.int64),
        native_raw_capture_sha256=np.asarray(native["raw_npz_sha256"]),
        plan_sha256=np.asarray(plan["plan_sha256"]),
        source_capture_commit_sha=np.asarray(source_capture_commit_sha),
    )
    canonical_raw_sha = _sha256_bytes(canonical_raw_payload)

    def path_consistency(path: str) -> np.ndarray:
        rows = sorted(
            (
                row
                for row in analysis["path_subbands"]
                if row["path"] == path and row["microphone"] == "ERR"
            ),
            key=lambda row: int(row["subband_index"]),
        )
        if len(rows) != 6:
            raise Stage2MeasurementError(f"{path}/ERR consistency row가 exact 6개가 아닙니다")
        return np.asarray(
            [
                min(
                    float(row["fit_a_fit_b_consistency"]),
                    float(row["untouched_holdout_consistency"]),
                )
                for row in rows
            ],
            dtype=np.float64,
        )

    primary_consistency = path_consistency("primary")
    secondary_consistency = path_consistency("secondary")
    timing_residual = float(q["maximum_timing_residual_samples"])
    canonical_analysis_payload = _npz_bytes(
        stage2_analysis_schema=np.asarray("stage2_2khz_analysis_npz_v1"),
        analysis_status=np.asarray("PASS"),
        stage2_contract_sha256=np.asarray(contract.digest()),
        capture_id=np.asarray(capture_id),
        raw_capture_sha256=np.asarray(canonical_raw_sha),
        sample_rate=np.asarray(SAMPLE_RATE, dtype=np.int64),
        physical_subbands_hz=np.asarray(
            contract.physical_identification_subbands_hz, dtype=np.float64
        ),
        primary_band_consistency=primary_consistency,
        secondary_band_consistency=secondary_consistency,
        primary_delay_samples=np.asarray(analysis["primary"]["delay_samples"], dtype=np.int64),
        secondary_delay_samples=np.asarray(analysis["secondary"]["delay_samples"], dtype=np.int64),
        timing_residual_max_samples=np.asarray(timing_residual, dtype=np.float64),
        submitted_output_pcm_sha256=np.asarray(submitted_plain_sha),
        native_analysis_core_sha256=np.asarray(claimed),
        source_capture_commit_sha=np.asarray(source_capture_commit_sha),
    )
    canonical_analysis_sha = _sha256_bytes(canonical_analysis_payload)

    epoch_rows = analysis.get("robustness_epochs", {}).get("rows")
    if not isinstance(epoch_rows, list) or len(epoch_rows) != 24:
        raise Stage2MeasurementError("production P/S에는 actual 24×1초 epoch row가 필요합니다")
    epoch_roles = np.asarray([str(row["role"]) for row in epoch_rows])
    epoch_starts = np.asarray([int(row["start_frame"]) for row in epoch_rows], dtype=np.int64)
    epoch_stops = np.asarray([int(row["stop_frame"]) for row in epoch_rows], dtype=np.int64)
    epoch_kept = np.asarray([bool(row["kept"]) for row in epoch_rows], dtype=bool)
    if not np.all(epoch_kept):
        raise Stage2MeasurementError("role당 exact 8개뿐이므로 24 independent epoch 모두 kept여야 합니다")

    artifacts = plan.get("artifacts", {})
    targets = {
        "raw_capture": _safe_relative_path(
            artifacts.get("raw_capture"), suffix=".npz", label="canonical raw path"
        ),
        "analysis": _safe_relative_path(
            artifacts.get("analysis"), suffix=".npz", label="canonical analysis path"
        ),
        "analysis_arrays": _safe_relative_path(
            artifacts.get("analysis_arrays"), suffix=".npz", label="analysis arrays path"
        ),
        "primary_candidate": _safe_relative_path(
            artifacts.get("primary_candidate"), suffix=".npz", label="primary candidate path"
        ),
        "secondary_candidate": _safe_relative_path(
            artifacts.get("secondary_candidate"), suffix=".npz", label="secondary candidate path"
        ),
        "relative_clock_receipt": _safe_relative_path(
            artifacts.get("relative_clock_receipt"), suffix=".json", label="clock receipt path"
        ),
        "measurement_level_evidence": _safe_relative_path(
            artifacts.get("measurement_level_evidence"),
            suffix=".json",
            label="measurement level evidence path",
        ),
        "analysis_receipt": _safe_relative_path(
            artifacts.get("analysis_receipt"), suffix=".json", label="analysis receipt path"
        ),
        "plant_binding": _safe_relative_path(
            artifacts.get("plant_binding"), suffix=".json", label="plant binding path"
        ),
    }

    analysis_array_payload = _npz_bytes(
        **{name: np.asarray(value) for name, value in arrays.items()}
    )
    core_sha = str(claimed)

    def plant_payload(path: str, fir_key: str, consistency: np.ndarray) -> bytes:
        item = analysis[path]
        bulk = float(item["bulk_delay_fractional_samples"])
        fractional = (bulk + 0.5) % 1.0 - 0.5
        return _npz_bytes(
            stage2_path_schema=np.asarray("stage2_2khz_causal_path_npz_v1"),
            measurement_status=np.asarray("PASS"),
            canonical_training_eligible=np.asarray(True),
            stage2_contract_id=np.asarray(contract.contract_id),
            stage2_contract_sha256=np.asarray(contract.digest()),
            role=np.asarray(path),
            fir=np.asarray(arrays[fir_key], dtype=np.float64),
            delay_samples=np.asarray(item["delay_samples"], dtype=np.int64),
            fractional_delay_samples=np.asarray(fractional, dtype=np.float64),
            delay_semantics=np.asarray("effective_zeros_before_compact_fir"),
            sample_rate=np.asarray(SAMPLE_RATE, dtype=np.int64),
            capture_id=np.asarray(capture_id),
            excitation_band_hz=np.asarray(
                [contract.required_excitation_lower_hz, contract.required_excitation_upper_hz],
                dtype=np.float64,
            ),
            band_consistency_hz=np.asarray(
                contract.physical_identification_subbands_hz, dtype=np.float64
            ),
            band_consistency=np.asarray(consistency, dtype=np.float64),
            independent_epoch_role_names=epoch_roles,
            independent_epoch_start_frames=epoch_starts,
            independent_epoch_stop_frames=epoch_stops,
            independent_epoch_kept=epoch_kept,
            repeated_slot_count=np.asarray(0, dtype=np.int64),
            timing_residual_max_samples=np.asarray(timing_residual, dtype=np.float64),
            xrun_count=np.asarray(xrun_count, dtype=np.int64),
            clip_count=np.asarray(clip_count, dtype=np.int64),
            sample_slip_count=np.asarray(sample_slip_count, dtype=np.int64),
            callback_status_failures=np.asarray(callback_status, dtype=np.int64),
            output_pcm_provenance=np.asarray("observed_submitted_int16"),
            source_raw_npz_path=np.asarray(targets["raw_capture"]),
            source_raw_npz_sha256=np.asarray(canonical_raw_sha),
            source_analysis_npz_path=np.asarray(targets["analysis"]),
            source_analysis_npz_sha256=np.asarray(canonical_analysis_sha),
            source_capture_commit_sha=np.asarray(source_capture_commit_sha),
            calibration_block_size=np.asarray(256, dtype=np.int64),
            error_mic_channel=np.asarray(0, dtype=np.int64),
            reference_mic_channel=np.asarray(1, dtype=np.int64),
        )

    primary_payload = plant_payload("primary", "primary_fir", primary_consistency)
    secondary_payload = plant_payload("secondary", "secondary_fir", secondary_consistency)
    level_evidence = {
        "schema": "measurement_level_evidence_v2_bootstrap_pair",
        "passed": True,
        "sample_rate": SAMPLE_RATE,
        "probe_amplitude": 0.003,
        "same_amplifier_setting": True,
        "meter_ch0_dbfs": float(meter["admission"]["meter_level_dbfs"]),
        "meter_target_dbfs": float(meter["admission"]["target_dbfs"]),
        "meter_tolerance_db": float(meter["admission"]["tolerance_db"]),
        "meter_raw_sha256": meter["raw_npz_sha256"],
        "native_signal_raw_sha256": native["raw_npz_sha256"],
        "capture_id": capture_id,
    }
    payloads = {
        "raw_capture": canonical_raw_payload,
        "analysis": canonical_analysis_payload,
        "analysis_arrays": analysis_array_payload,
        "primary_candidate": primary_payload,
        "secondary_candidate": secondary_payload,
        "measurement_level_evidence": _canonical_json_bytes(level_evidence),
    }
    published: dict[str, dict[str, Any]] = {}
    for name in (
        "raw_capture",
        "analysis",
        "analysis_arrays",
        "primary_candidate",
        "secondary_candidate",
        "measurement_level_evidence",
    ):
        published[name] = publish_repository_bytes_noreplace(
            repository_root,
            targets[name],
            payloads[name],
            mode=0o600,
            recovery_tag=f"stage2_{name}",
        )
    clock_receipt = {
        "schema": "stage2_2khz_relative_clock_model_v1",
        "status": "PASS",
        "control_band_contract_sha256": contract.digest(),
        "raw_capture_sha256": published["raw_capture"]["sha256"],
        "analysis_sha256": published["analysis"]["sha256"],
        "submitted_output_pcm_sha256": submitted_plain_sha,
        "relative_shared_q_model_pass": True,
        "absolute_hardware_frame_clock": False,
        "submitted_stereo_known_code_bound": True,
        "xrun_clip_status_slip_zero": True,
    }
    published["relative_clock_receipt"] = publish_repository_bytes_noreplace(
        repository_root,
        targets["relative_clock_receipt"],
        _canonical_json_bytes(clock_receipt),
        mode=0o600,
        recovery_tag="stage2_relative_clock_receipt",
    )
    final_analysis = {
        **dict(analysis),
        "analysis_core_sha256": core_sha,
        "native_raw_capture": {
            "path": artifacts["native_raw_capture"],
            "sha256": native["raw_npz_sha256"],
        },
        "published_dependencies": {
            name: {
                "path": item["path"],
                "sha256": item["sha256"],
                "size": item["size"],
            }
            for name, item in published.items()
        },
        "final_commit_marker": True,
    }
    final_analysis.pop("analysis_sha256", None)
    final_analysis["analysis_sha256"] = _payload_sha256(final_analysis)
    published["analysis_receipt"] = publish_repository_bytes_noreplace(
        repository_root,
        targets["analysis_receipt"],
        _canonical_json_bytes(final_analysis),
        mode=0o600,
        recovery_tag="stage2_analysis_receipt",
    )
    binding = {
        "schema": "stage2_2khz_plant_binding_v1",
        "status": "PASS",
        "canonical_training_eligible": True,
        "fixture_only": False,
        "control_band_contract": {"id": contract.contract_id, "sha256": contract.digest()},
        "sample_rate_hz": SAMPLE_RATE,
        "block_size": 256,
        "verified_physical_subbands_hz": [
            list(row) for row in contract.physical_identification_subbands_hz
        ],
        "minimum_subband_consistency": 0.95,
        "maximum_timing_residual_samples": MAX_TIMING_RESIDUAL_SAMPLES,
        "minimum_independent_epochs_per_role": 8,
        "periodic_repeat_indices_allowed": False,
        "handoff_extra_samples": 256,
        "lead_source": "PlantDelays.lead()",
        "primary_path": {
            "path": published["primary_candidate"]["path"],
            "sha256": published["primary_candidate"]["sha256"],
        },
        "secondary_path": {
            "path": published["secondary_candidate"]["path"],
            "sha256": published["secondary_candidate"]["sha256"],
        },
        "raw_capture": {
            "path": published["raw_capture"]["path"],
            "sha256": published["raw_capture"]["sha256"],
        },
        "analysis": {
            "path": published["analysis"]["path"],
            "sha256": published["analysis"]["sha256"],
        },
        "measurement_level_evidence": {
            "path": published["measurement_level_evidence"]["path"],
            "sha256": published["measurement_level_evidence"]["sha256"],
        },
        "relative_clock_model_receipt": {
            "path": published["relative_clock_receipt"]["path"],
            "sha256": published["relative_clock_receipt"]["sha256"],
        },
        "err_channel_index": 0,
        "reference_channel_index": 1,
        "source_capture_commit_sha": source_capture_commit_sha,
    }
    published["plant_binding"] = publish_repository_bytes_noreplace(
        repository_root,
        targets["plant_binding"],
        _canonical_json_bytes(binding),
        mode=0o600,
        recovery_tag="stage2_plant_binding",
    )
    return {"analysis": final_analysis, "binding": binding, "published": published}


__all__ = [
    "ANALYSIS_SCHEMA",
    "COMPACT_FIR_LENGTH",
    "MIMO_MAX_INPUT_CONDITION",
    "Q_SELECTION_BAND_HZ",
    "SHARED_Q_SCHEMA",
    "analyse_stage2_raw_bytes",
    "estimate_shared_affine_q",
    "publish_stage2_analysis_no_replace",
]
