"""Stage-2 2 kHz diagnostic stream의 순수 offline clock 감사.

이 모듈은 파일, 오디오 장치, 네트워크를 열지 않는다.  Canonical diagnostic
submitted PCM과 별도 input stream에서 얻은 가변 길이 captured PCM만 받아서 다음을
검증한다.

* output/input raw의 coarse timeline 포함 관계
* 8개 49/98-PCM two-tone slot이 공유하는 단 하나의 affine sample-rate ratio ``q``
* slot, tone, microphone view가 그 global ``q``와 일치하는지
* ±1000 ppm search의 interior optimum, distinct runner-up ambiguity, phase stability

각 slot마다 서로 다른 q를 적용해 결과를 구제하지 않는다.  Slot-local 추정치는 오직
global affine 가설을 반증하는 view로만 사용한다.  따라서 독립 DAC/ADC clock이 한
stream 안에서 rate regime을 바꾸면 receipt는 finite FAIL로 남는다.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import correlate

from .fullband_causal_v5 import (
    CLOCK_HARD_MAX_RESIDUAL_SAMPLES,
    CLOCK_VIEW_MAX_ENDPOINT_DISAGREEMENT_SAMPLES,
)
from .stage2_2khz_measurement_v2 import (
    BLOCK_SIZE,
    DIAGNOSTIC_ANALYSIS_FRAMES,
    SAMPLE_RATE,
    Stage2MeasurementV2Error,
    build_stage2_v2_live_safe_fallback_plan,
)


DIAGNOSTIC_CLOCK_SCHEMA = "stage2_2khz_diagnostic_global_affine_clock_v1"
MAX_CLOCK_PPM = 1_000.0
GLOBAL_GRID_STEP_PPM = 1.0
RATE_BOUNDARY_GUARD_PPM = 1.0
DISTINCT_BASIN_EXCLUSION_PPM = 8.0
MIN_DISTINCT_BASIN_OBJECTIVE_GAP = 0.01
MIN_VIEW_PROJECTION_SNR_DB = 40.0
MIN_GLOBAL_COHERENCE = 0.995
PHASE_BLOCK_FRAMES = 2_048
PHASE_HOP_FRAMES = 1_024
MIN_PHASE_OBSERVATIONS = 16
MAX_VIEW_AMPLITUDE_CV = 0.20
ALIGNMENT_EXCLUSION_BLOCKS = 8
MIN_ALIGNMENT_CORRELATION = 0.15
MIN_ALIGNMENT_RUNNER_UP_GAP = 0.005
QUIET_SETTLE_FRAMES = 8_192
FINITE_DB_LIMIT = 600.0


def _payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finite_tree(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, label=f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, label=f"{label}[{index}]")
        return
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise Stage2MeasurementV2Error(f"{label}가 finite가 아닙니다")


def _finite_db_ratio(numerator: float, denominator: float) -> float:
    top = float(numerator)
    bottom = float(denominator)
    if not (math.isfinite(top) and math.isfinite(bottom)) or top < 0.0 or bottom < 0.0:
        raise Stage2MeasurementV2Error("clock projection power가 finite non-negative가 아닙니다")
    if top == 0.0 and bottom == 0.0:
        return 0.0
    if bottom == 0.0:
        return FINITE_DB_LIMIT
    if top == 0.0:
        return -FINITE_DB_LIMIT
    return float(
        np.clip(10.0 * math.log10(top / bottom), -FINITE_DB_LIMIT, FINITE_DB_LIMIT)
    )


def _normalise_capture(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 2 or source.shape[1] != 2 or source.shape[0] <= 0:
        raise Stage2MeasurementV2Error("diagnostic captured PCM shape가 [frames,2]가 아닙니다")
    if not np.issubdtype(source.dtype, np.number) or not np.all(np.isfinite(source)):
        raise Stage2MeasurementV2Error("diagnostic captured PCM이 finite numeric이 아닙니다")
    if source.dtype == np.dtype("<i4") or source.dtype == np.int32:
        return np.ascontiguousarray(source, dtype=np.float64) / 2147483648.0
    if source.dtype == np.dtype("<i2") or source.dtype == np.int16:
        return np.ascontiguousarray(source, dtype=np.float64) / 32768.0
    converted = np.ascontiguousarray(source, dtype=np.float64)
    if float(np.max(np.abs(converted))) >= 1.0:
        raise Stage2MeasurementV2Error("float diagnostic captured PCM은 |x|<1이어야 합니다")
    return converted


def _validate_inputs(
    plan: Mapping[str, Any],
    submitted_diagnostic_pcm: np.ndarray,
    captured_pcm: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    canonical, full = build_stage2_v2_live_safe_fallback_plan()
    if not isinstance(plan, Mapping) or dict(plan) != canonical:
        raise Stage2MeasurementV2Error("diagnostic clock plan이 exact canonical v2가 아닙니다")
    phase_stop = int(canonical["live_phase_contract"]["diagnostic_phase_stop_frame"])
    expected = np.asarray(full[:phase_stop])
    submitted = np.asarray(submitted_diagnostic_pcm)
    if (
        submitted.dtype != np.dtype("<i2")
        or submitted.shape != expected.shape
        or not np.array_equal(submitted, expected)
    ):
        raise Stage2MeasurementV2Error(
            "diagnostic submitted PCM이 exact canonical phase-1 slice가 아닙니다"
        )
    captured = _normalise_capture(captured_pcm)
    if len(captured) < len(submitted):
        raise Stage2MeasurementV2Error(
            "split input raw는 canonical diagnostic 출력 전체를 포함해야 합니다"
        )
    rows = [dict(row) for row in canonical["nonlinearity_diagnostics"]["slots"]]
    if len(rows) != 8:
        raise Stage2MeasurementV2Error("diagnostic two-tone slot은 exact 8개여야 합니다")
    return canonical, submitted, captured, rows


def _block_rms(value: np.ndarray) -> np.ndarray:
    frames = len(value) // BLOCK_SIZE * BLOCK_SIZE
    if frames < BLOCK_SIZE:
        raise Stage2MeasurementV2Error("coarse alignment에 256-frame block이 없습니다")
    shaped = np.asarray(value[:frames], dtype=np.float64).reshape(-1, BLOCK_SIZE, 2)
    return np.sqrt(np.mean(shaped * shaped, axis=(1, 2)))


def _coarse_alignment(submitted: np.ndarray, captured: np.ndarray) -> dict[str, Any]:
    """input이 output보다 먼저 시작하고 나중에 끝난 split-stream 포함 관계를 찾는다."""

    output_feature = _block_rms(submitted.astype(np.float64) / 32768.0)
    input_feature = _block_rms(captured)
    if len(input_feature) < len(output_feature):
        raise Stage2MeasurementV2Error("input block timeline이 output timeline보다 짧습니다")
    output_norm = float(np.linalg.norm(output_feature))
    if output_norm <= 0.0:
        raise Stage2MeasurementV2Error("diagnostic submitted envelope가 exact zero입니다")
    numerator = correlate(input_feature, output_feature, mode="valid", method="fft")
    input_power = np.convolve(
        input_feature * input_feature,
        np.ones(len(output_feature), dtype=np.float64),
        mode="valid",
    )
    denominator = output_norm * np.sqrt(np.maximum(input_power, 1.0e-300))
    scores = np.asarray(numerator / denominator, dtype=np.float64)
    best_index = int(np.argmax(scores))
    best = float(scores[best_index])
    eligible = np.ones(len(scores), dtype=np.bool_)
    lower = max(0, best_index - ALIGNMENT_EXCLUSION_BLOCKS)
    upper = min(len(scores), best_index + ALIGNMENT_EXCLUSION_BLOCKS + 1)
    eligible[lower:upper] = False
    if np.any(eligible):
        runner_up = float(np.max(scores[eligible]))
        gap = best - runner_up
        runner_up_block = int(np.flatnonzero(eligible)[np.argmax(scores[eligible])])
    else:
        runner_up = None
        runner_up_block = None
        gap = 1.0
    passed = bool(
        best >= MIN_ALIGNMENT_CORRELATION
        and gap >= MIN_ALIGNMENT_RUNNER_UP_GAP
    )
    return {
        "method": "256_frame_rms_normalized_valid_cross_correlation",
        "input_contains_complete_output_timeline_required": True,
        "coarse_capture_offset_blocks": best_index,
        "coarse_capture_offset_samples": best_index * BLOCK_SIZE,
        "coarse_uncertainty_samples": BLOCK_SIZE,
        "correlation": best,
        "runner_up_block": runner_up_block,
        "runner_up_correlation": runner_up,
        "runner_up_gap": gap,
        "minimum_correlation": MIN_ALIGNMENT_CORRELATION,
        "minimum_runner_up_gap": MIN_ALIGNMENT_RUNNER_UP_GAP,
        "passed": passed,
    }


def _complex_observations(
    captured: np.ndarray,
    *,
    start: int,
    stop: int,
    microphone: int,
    frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stop - start != DIAGNOSTIC_ANALYSIS_FRAMES:
        raise Stage2MeasurementV2Error("clock view가 exact diagnostic analysis length가 아닙니다")
    if start < 0 or stop > len(captured) or microphone not in (0, 1):
        raise Stage2MeasurementV2Error("clock view가 captured raw 경계를 벗어났습니다")
    window = np.hanning(PHASE_BLOCK_FRAMES)
    centres: list[float] = []
    coefficients: list[complex] = []
    amplitudes: list[float] = []
    for block_start in range(start, stop - PHASE_BLOCK_FRAMES + 1, PHASE_HOP_FRAMES):
        indices = np.arange(block_start, block_start + PHASE_BLOCK_FRAMES, dtype=np.float64)
        coefficient = np.sum(
            captured[block_start : block_start + PHASE_BLOCK_FRAMES, microphone]
            * window
            * np.exp(-2j * np.pi * float(frequency_hz) * indices / SAMPLE_RATE)
        )
        centres.append(block_start + (PHASE_BLOCK_FRAMES - 1) / 2.0)
        coefficients.append(complex(coefficient))
        amplitudes.append(float(abs(coefficient)))
    if len(coefficients) < MIN_PHASE_OBSERVATIONS:
        raise Stage2MeasurementV2Error("clock view phase observation이 부족합니다")
    values = np.asarray(coefficients, dtype=np.complex128)
    magnitude = np.asarray(amplitudes, dtype=np.float64)
    if np.any(magnitude <= 0.0) or not np.all(np.isfinite(values)):
        raise Stage2MeasurementV2Error("clock view complex projection이 finite nonzero가 아닙니다")
    return np.asarray(centres, dtype=np.float64), values / magnitude, magnitude


def _projection_noise_power(
    captured: np.ndarray,
    *,
    start: int,
    stop: int,
    microphone: int,
    frequency_hz: float,
) -> float:
    frames = min(stop - start, DIAGNOSTIC_ANALYSIS_FRAMES)
    if frames < PHASE_BLOCK_FRAMES or start < 0 or start + frames > len(captured):
        raise Stage2MeasurementV2Error("clock quiet projection 경계가 잘못됐습니다")
    window = np.hanning(PHASE_BLOCK_FRAMES)
    powers: list[float] = []
    for block_start in range(
        start,
        start + DIAGNOSTIC_ANALYSIS_FRAMES - PHASE_BLOCK_FRAMES + 1,
        PHASE_HOP_FRAMES,
    ):
        indices = np.arange(
            block_start, block_start + PHASE_BLOCK_FRAMES, dtype=np.float64
        )
        coefficient = np.sum(
            captured[block_start : block_start + PHASE_BLOCK_FRAMES, microphone]
            * window
            * np.exp(-2j * np.pi * float(frequency_hz) * indices / SAMPLE_RATE)
        )
        powers.append(float(abs(coefficient) ** 2))
    if len(powers) < MIN_PHASE_OBSERVATIONS or not np.all(np.isfinite(powers)):
        raise Stage2MeasurementV2Error("clock quiet projection이 finite하지 않습니다")
    return float(np.median(powers))


def _objective_grid(views: Sequence[Mapping[str, Any]], ppm_grid: np.ndarray) -> np.ndarray:
    if not views:
        raise Stage2MeasurementV2Error("clock search view가 비었습니다")
    objective = np.zeros(len(ppm_grid), dtype=np.float64)
    deltas = ppm_grid * 1.0e-6
    for view in views:
        centres = np.asarray(view["centres"], dtype=np.float64)
        unit = np.asarray(view["unit"], dtype=np.complex128)
        angular = 2.0 * np.pi * float(view["frequency_hz"]) * centres / SAMPLE_RATE
        corrected = unit[:, None] * np.exp(-1j * angular[:, None] * deltas[None, :])
        coherence = np.abs(np.mean(corrected, axis=0))
        objective += 1.0 - coherence
    objective /= float(len(views))
    return objective


def _scalar_objective(views: Sequence[Mapping[str, Any]], ppm: float) -> float:
    return float(_objective_grid(views, np.asarray([ppm], dtype=np.float64))[0])


def _search_rate(views: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Any]:
    """±1000 ppm 전체 grid, interior refinement, distinct-basin ambiguity gate."""

    ppm_grid = np.linspace(
        -MAX_CLOCK_PPM,
        MAX_CLOCK_PPM,
        int(round(2.0 * MAX_CLOCK_PPM / GLOBAL_GRID_STEP_PPM)) + 1,
    )
    objective = _objective_grid(views, ppm_grid)
    best_index = int(np.argmin(objective))
    grid_best_ppm = float(ppm_grid[best_index])
    boundary = best_index in (0, len(ppm_grid) - 1)
    if boundary:
        selected_ppm = grid_best_ppm
        selected_objective = float(objective[best_index])
        refinement_success = False
    else:
        fit = minimize_scalar(
            lambda value: _scalar_objective(views, float(value)),
            bounds=(float(ppm_grid[best_index - 1]), float(ppm_grid[best_index + 1])),
            method="bounded",
            options={"xatol": 1.0e-8},
        )
        refinement_success = bool(
            fit.success and math.isfinite(float(fit.x)) and math.isfinite(float(fit.fun))
        )
        selected_ppm = float(fit.x) if refinement_success else grid_best_ppm
        selected_objective = (
            float(fit.fun) if refinement_success else float(objective[best_index])
        )
    interior_passed = bool(
        refinement_success
        and selected_ppm > -MAX_CLOCK_PPM + RATE_BOUNDARY_GUARD_PPM
        and selected_ppm < MAX_CLOCK_PPM - RATE_BOUNDARY_GUARD_PPM
    )

    local_minima = [
        index
        for index in range(1, len(objective) - 1)
        if objective[index] <= objective[index - 1]
        and objective[index] <= objective[index + 1]
    ]
    runner_candidates = [
        index
        for index in local_minima
        if abs(float(ppm_grid[index]) - selected_ppm) >= DISTINCT_BASIN_EXCLUSION_PPM
    ]
    if runner_candidates:
        runner_index = min(runner_candidates, key=lambda index: float(objective[index]))
        runner_ppm: float | None = float(ppm_grid[runner_index])
        runner_objective: float | None = float(objective[runner_index])
        objective_gap = runner_objective - selected_objective
    else:
        runner_ppm = None
        runner_objective = None
        objective_gap = 1.0
    ambiguity_passed = bool(objective_gap >= MIN_DISTINCT_BASIN_OBJECTIVE_GAP)
    coherence = 1.0 - selected_objective
    result = {
        "label": label,
        "bounds_ppm": [-MAX_CLOCK_PPM, MAX_CLOCK_PPM],
        "grid_step_ppm": GLOBAL_GRID_STEP_PPM,
        "grid_best_ppm": grid_best_ppm,
        "selected_ppm": selected_ppm,
        "selected_rate_ratio": 1.0 + selected_ppm * 1.0e-6,
        "objective": selected_objective,
        "coherence": coherence,
        "boundary_optimum": boundary,
        "boundary_guard_ppm": RATE_BOUNDARY_GUARD_PPM,
        "interior_optimum_passed": interior_passed,
        "runner_up_ppm": runner_ppm,
        "runner_up_objective": runner_objective,
        "runner_up_objective_gap": objective_gap,
        "minimum_runner_up_objective_gap": MIN_DISTINCT_BASIN_OBJECTIVE_GAP,
        "ambiguity_passed": ambiguity_passed,
        "refinement_success": refinement_success,
        "passed": bool(interior_passed and ambiguity_passed),
    }
    _finite_tree(result, label=label)
    return result


def _phase_stability(
    view: Mapping[str, Any], *, global_ppm: float
) -> tuple[float, float]:
    centres = np.asarray(view["centres"], dtype=np.float64)
    unit = np.asarray(view["unit"], dtype=np.complex128)
    frequency = float(view["frequency_hz"])
    angular = 2.0 * np.pi * frequency * centres / SAMPLE_RATE
    corrected = unit * np.exp(-1j * angular * global_ppm * 1.0e-6)
    reference = np.sum(corrected)
    if abs(reference) <= 0.0:
        return FINITE_DB_LIMIT, 0.0
    residual = np.angle(corrected * np.conj(reference))
    residual_samples = np.abs(residual) * SAMPLE_RATE / (2.0 * np.pi * frequency)
    return float(np.max(residual_samples)), float(abs(np.mean(corrected)))


def estimate_stage2_diagnostic_global_clock(
    plan: Mapping[str, Any],
    submitted_diagnostic_pcm: np.ndarray,
    captured_pcm: np.ndarray,
) -> dict[str, Any]:
    """8-slot diagnostic raw가 하나의 global affine clock을 공유하는지 판정한다.

    구조/byte 계약 위반은 예외다.  유효 raw에서 관측한 boundary, ambiguity, SNR 또는
    stability 실패는 immutable evidence로 저장할 수 있도록 finite FAIL receipt로
    반환한다.
    """

    canonical, submitted, captured, rows = _validate_inputs(
        plan, submitted_diagnostic_pcm, captured_pcm
    )
    alignment = _coarse_alignment(submitted, captured)
    offset = int(alignment["coarse_capture_offset_samples"])
    # 실측 a68d13…에서 stream 시작 직후 8,192 frame은 ERR/REF
    # DC decay가 -40 dBFS 수준으로 남아 있었고, 그 뒤의 동일 길이
    # window는 -70/-61 dBFS였다. 출력 pre-zero 안의 settled window만
    # noise projection으로 사용하고 stream-open transient를 THD 증거로
    # 오인하지 않는다.
    quiet_start = offset + QUIET_SETTLE_FRAMES
    quiet_stop = quiet_start + DIAGNOSTIC_ANALYSIS_FRAMES
    if quiet_stop > len(captured):
        raise Stage2MeasurementV2Error("coarse aligned quiet window가 captured raw 밖입니다")

    views: list[dict[str, Any]] = []
    slot_views: dict[int, list[dict[str, Any]]] = {}
    for slot_index, row in enumerate(rows):
        start = offset + int(row["analysis_start_frame"])
        stop = offset + int(row["analysis_stop_frame"])
        if start < 0 or stop > len(captured):
            raise Stage2MeasurementV2Error(
                "coarse aligned diagnostic analysis window가 captured raw 밖입니다"
            )
        for microphone, microphone_name in enumerate(("ERR", "REF")):
            for tone_index, frequency in enumerate(row["fundamental_frequencies_hz"]):
                centres, unit, magnitude = _complex_observations(
                    captured,
                    start=start,
                    stop=stop,
                    microphone=microphone,
                    frequency_hz=float(frequency),
                )
                noise_power = _projection_noise_power(
                    captured,
                    start=quiet_start,
                    stop=quiet_stop,
                    microphone=microphone,
                    frequency_hz=float(frequency),
                )
                signal_power = float(np.median(magnitude * magnitude))
                snr = _finite_db_ratio(signal_power, noise_power)
                amplitude_cv = float(np.std(magnitude) / np.mean(magnitude))
                view = {
                    "slot_index": slot_index,
                    "path": str(row["path"]),
                    "pair_index": int(row["pair_index"]),
                    "level_pcm": int(row["level_pcm"]),
                    "microphone": microphone_name,
                    "tone_index": tone_index,
                    "frequency_hz": float(frequency),
                    "centres": centres,
                    "unit": unit,
                    "projection_snr_db": snr,
                    "amplitude_cv": amplitude_cv,
                }
                views.append(view)
                slot_views.setdefault(slot_index, []).append(view)

    global_search = _search_rate(views, label="all_slots_all_tones_all_microphones")
    global_ppm = float(global_search["selected_ppm"])
    slot_receipts: list[dict[str, Any]] = []
    for slot_index, local_views in slot_views.items():
        search = _search_rate(local_views, label=f"slot_{slot_index}")
        disagreement = (
            abs(float(search["selected_ppm"]) - global_ppm)
            * 1.0e-6
            * DIAGNOSTIC_ANALYSIS_FRAMES
        )
        passed = bool(
            search["passed"]
            and disagreement <= CLOCK_VIEW_MAX_ENDPOINT_DISAGREEMENT_SAMPLES
        )
        slot_receipts.append(
            {
                "slot_index": slot_index,
                "path": str(rows[slot_index]["path"]),
                "pair_index": int(rows[slot_index]["pair_index"]),
                "level_pcm": int(rows[slot_index]["level_pcm"]),
                "search": search,
                "global_endpoint_disagreement_samples": disagreement,
                "passed": passed,
            }
        )

    view_receipts: list[dict[str, Any]] = []
    for view in views:
        search = _search_rate([view], label=(
            f"slot_{view['slot_index']}_{view['microphone']}_{int(view['frequency_hz'])}Hz"
        ))
        disagreement = (
            abs(float(search["selected_ppm"]) - global_ppm)
            * 1.0e-6
            * DIAGNOSTIC_ANALYSIS_FRAMES
        )
        residual_samples, coherence = _phase_stability(view, global_ppm=global_ppm)
        passed = bool(
            search["passed"]
            and float(view["projection_snr_db"]) >= MIN_VIEW_PROJECTION_SNR_DB
            and float(view["amplitude_cv"]) <= MAX_VIEW_AMPLITUDE_CV
            and disagreement <= CLOCK_VIEW_MAX_ENDPOINT_DISAGREEMENT_SAMPLES
            and residual_samples <= CLOCK_HARD_MAX_RESIDUAL_SAMPLES
        )
        view_receipts.append(
            {
                "slot_index": int(view["slot_index"]),
                "path": str(view["path"]),
                "pair_index": int(view["pair_index"]),
                "level_pcm": int(view["level_pcm"]),
                "microphone": str(view["microphone"]),
                "tone_index": int(view["tone_index"]),
                "frequency_hz": float(view["frequency_hz"]),
                "projection_snr_db": float(view["projection_snr_db"]),
                "amplitude_cv": float(view["amplitude_cv"]),
                "search": search,
                "global_endpoint_disagreement_samples": disagreement,
                "maximum_phase_residual_samples": residual_samples,
                "global_corrected_coherence": coherence,
                "passed": passed,
            }
        )

    passed = bool(
        alignment["passed"]
        and global_search["passed"]
        and float(global_search["coherence"]) >= MIN_GLOBAL_COHERENCE
        and all(row["passed"] for row in slot_receipts)
        and all(row["passed"] for row in view_receipts)
    )
    receipt: dict[str, Any] = {
        "schema": DIAGNOSTIC_CLOCK_SCHEMA,
        "signal_plan_sha256": canonical["canonical_payload_sha256"],
        "submitted_phase_frames": len(submitted),
        "captured_frames": len(captured),
        "clock_model": "one_global_affine_rate_ratio_all_8_slots",
        "slot_local_rates_are_diagnostic_only": True,
        "per_slot_clock_correction_may_not_grant_pass": True,
        "alignment": alignment,
        "settled_quiet_window": {
            "capture_start_frame": quiet_start,
            "capture_stop_frame": quiet_stop,
            "stream_start_settle_frames_excluded": QUIET_SETTLE_FRAMES,
        },
        "global_search": global_search,
        "thresholds": {
            "maximum_absolute_clock_ppm": MAX_CLOCK_PPM,
            "grid_step_ppm": GLOBAL_GRID_STEP_PPM,
            "rate_boundary_guard_ppm": RATE_BOUNDARY_GUARD_PPM,
            "distinct_basin_exclusion_ppm": DISTINCT_BASIN_EXCLUSION_PPM,
            "minimum_distinct_basin_objective_gap": MIN_DISTINCT_BASIN_OBJECTIVE_GAP,
            "minimum_view_projection_snr_db": MIN_VIEW_PROJECTION_SNR_DB,
            "minimum_global_coherence": MIN_GLOBAL_COHERENCE,
            "maximum_view_endpoint_disagreement_samples": (
                CLOCK_VIEW_MAX_ENDPOINT_DISAGREEMENT_SAMPLES
            ),
            "maximum_phase_residual_samples": CLOCK_HARD_MAX_RESIDUAL_SAMPLES,
            "maximum_view_amplitude_cv": MAX_VIEW_AMPLITUDE_CV,
        },
        "slot_rows": slot_receipts,
        "view_rows": view_receipts,
        "passed": passed,
        "diagnostic_linearity_may_run": passed,
        "ps_phase_may_start": False,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    _finite_tree(receipt, label="diagnostic_clock_receipt")
    return receipt


__all__ = [
    "DIAGNOSTIC_CLOCK_SCHEMA",
    "MAX_CLOCK_PPM",
    "estimate_stage2_diagnostic_global_clock",
]
