"""Stage-2 v2 DPSS P/S의 순수 offline analyzer.

audio/file/network를 열지 않는다. immutable submitted/captured arrays에서 diagnostic
linearity, common affine clock, coarse delay + fixed DPSS subspace fit, fit-a/fit-b 교차검증,
untouched holdout 및 3 dB actuator feasibility를 계산한다.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from .fullband_causal_v5 import (
    CLOCK_HARD_MAX_RESIDUAL_SAMPLES,
    CLOCK_VIEW_MAX_ENDPOINT_DISAGREEMENT_SAMPLES,
    _clock_local_accessor_v5,
    _clock_waveform_rows_v5,
    _waveform_transfer_bank_v5,
)
from .fullband_causal_v6 import global_grid_basin_search_v6
from .stage2_2khz_contract import Stage2TwoKilohertzContract
from .stage2_2khz_measurement_v2 import (
    FIT_ROLES,
    HOLDOUT_ROLE,
    LIVE_SAFE_BAND_HZ,
    MAX_GRAM_CONDITION,
    PATHS,
    PATH_CHANNEL,
    PERIOD,
    SAMPLE_RATE,
    SUPPORT_SAMPLES,
    Stage2HoldoutAccessLedger,
    Stage2MeasurementV2Error,
    _array_sha256,
    _central_period,
    _payload_sha256,
    audit_stage2_v2_live_safe_dpss_gram,
    build_stage2_bandlimited_dpss_basis,
    validate_stage2_v2_live_safe_fallback_plan,
)
from .stage2_2khz_level_contract import (
    MINIMUM_ACTUATOR_HEADROOM_DB,
    validate_stage2_physical_operating_level_evidence,
)


ANALYSIS_SCHEMA = "stage2_2khz_dpss_physical_analysis_v2"
DIAGNOSTIC_SCHEMA = "stage2_2khz_multilevel_linearity_preflight_v2"
MIN_FUNDAMENTAL_SNR_DB = 40.0
MAX_THD_IMD_DBC = -30.0
MAX_INPUT_THD_IMD_DBC = -45.0
MIN_DISTORTION_NOISE_MARGIN_DB = 10.0
MAX_GAIN_RATIO_ERROR_DB = 1.0
MAX_GAIN_RATIO_PHASE_ERROR_SAMPLES = 0.10
MAX_DIAGNOSTIC_ADC_PEAK_ABS = 0.40
ABSOLUTE_ADC_SAFETY_LIMIT = 0.50
MIN_COMPLEX_AGREEMENT = 0.95
MAX_COMPLEX_RELATIVE_ERROR = 0.25
MAX_MAGNITUDE_RATIO_ERROR_DB = 1.0
MIN_RESPONSE_TO_NOISE_DB = 40.0
MAX_COARSE_DELAY_SAMPLES = 4_800
COARSE_PRE_ROLL_SAMPLES = 256
FINITE_DB_LIMIT = 600.0


def _finite_capture(value: np.ndarray, *, minimum_frames: int) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 2 or source.shape[0] < minimum_frames or source.shape[1] != 2:
        raise Stage2MeasurementV2Error("captured raw shape가 Stage-2 v2와 다릅니다")
    if not np.issubdtype(source.dtype, np.number) or not np.all(np.isfinite(source)):
        raise Stage2MeasurementV2Error("captured raw가 finite numeric array가 아닙니다")
    if source.dtype == np.int32:
        return source.astype(np.float64) / 2147483648.0
    if source.dtype == np.int16:
        return source.astype(np.float64) / 32768.0
    return source.astype(np.float64)


def _expanded_bins(indices: set[int], size: int) -> set[int]:
    return {
        candidate
        for index in indices
        for candidate in (index - 1, index, index + 1)
        if 0 < candidate < size
    }


def _power(spectrum: np.ndarray, indices: set[int]) -> float:
    selected = sorted(_expanded_bins(indices, spectrum.size))
    return float(np.sum(np.abs(spectrum[selected]) ** 2)) if selected else 0.0


def _finite_db_ratio(numerator: float, denominator: float) -> float:
    """canonical JSON에 ±inf를 넣지 않는 conservative finite dB ratio."""

    top = float(numerator)
    bottom = float(denominator)
    if not (math.isfinite(top) and math.isfinite(bottom)) or top < 0.0 or bottom < 0.0:
        raise Stage2MeasurementV2Error("power ratio가 finite non-negative가 아닙니다")
    if top == 0.0 and bottom == 0.0:
        return 0.0
    if bottom == 0.0:
        return FINITE_DB_LIMIT
    if top == 0.0:
        return -FINITE_DB_LIMIT
    return float(np.clip(10.0 * math.log10(top / bottom), -FINITE_DB_LIMIT, FINITE_DB_LIMIT))


def _distortion_row(values: np.ndarray, submitted: np.ndarray, pair: tuple[int, int], noise: np.ndarray) -> dict[str, Any]:
    samples = np.asarray(values, dtype=np.float64)
    source = np.asarray(submitted, dtype=np.float64)
    quiet = np.asarray(noise, dtype=np.float64)
    if samples.shape != source.shape or samples.size != 24_000 or quiet.size != 24_000:
        raise Stage2MeasurementV2Error("diagnostic analysis window가 exact 0.5초가 아닙니다")
    spectrum = np.fft.rfft(samples)
    input_spectrum = np.fft.rfft(source)
    noise_spectrum = np.fft.rfft(quiet)
    first, second = pair
    fundamentals = {first // 2, second // 2}
    harmonics = {
        multiplier * tone // 2
        for tone in pair
        for multiplier in (2, 3)
        if multiplier * tone < SAMPLE_RATE / 2
    } - fundamentals
    products_hz = {
        abs(second - first),
        first + second,
        abs(2 * first - second),
        abs(2 * second - first),
        2 * first + second,
        first + 2 * second,
    }
    products = {
        value // 2 for value in products_hz if 0 < value < SAMPLE_RATE / 2
    } - fundamentals - harmonics
    fundamental_power = _power(spectrum, fundamentals)
    noise_power = _power(noise_spectrum, fundamentals)
    thd = _finite_db_ratio(_power(spectrum, harmonics), fundamental_power)
    imd = _finite_db_ratio(_power(spectrum, products), fundamental_power)
    input_fundamental = _power(input_spectrum, fundamentals)
    input_thd = _finite_db_ratio(_power(input_spectrum, harmonics), input_fundamental)
    input_imd = _finite_db_ratio(_power(input_spectrum, products), input_fundamental)
    snr = _finite_db_ratio(fundamental_power, noise_power)
    distortion_indices = _expanded_bins(harmonics | products, spectrum.size)
    # quiet 구간에서 관측한 가장 큰 bin을 모든 distortion detection line에 배치한
    # conservative upper bound다. exact-zero quiet는 -600 dBc로 canonicalize한다.
    maximum_quiet_bin_power = (
        max((float(abs(noise_spectrum[index]) ** 2) for index in distortion_indices), default=0.0)
    )
    distortion_noise_upper_power = maximum_quiet_bin_power * len(distortion_indices)
    distortion_noise_upper_dbc = _finite_db_ratio(
        distortion_noise_upper_power, fundamental_power
    )
    distortion_noise_margin = (
        MAX_THD_IMD_DBC - distortion_noise_upper_dbc
    )
    passed = bool(
        snr >= MIN_FUNDAMENTAL_SNR_DB
        and thd <= MAX_THD_IMD_DBC
        and imd <= MAX_THD_IMD_DBC
        and input_thd <= MAX_INPUT_THD_IMD_DBC
        and input_imd <= MAX_INPUT_THD_IMD_DBC
        and distortion_noise_margin >= MIN_DISTORTION_NOISE_MARGIN_DB
    )
    return {
        "fundamental_snr_db": snr,
        "thd_dbc": thd,
        "imd_dbc": imd,
        "submitted_thd_dbc": input_thd,
        "submitted_imd_dbc": input_imd,
        "distortion_noise_upper_bound_dbc": distortion_noise_upper_dbc,
        "distortion_threshold_noise_margin_db": distortion_noise_margin,
        "fundamental_complex": [
            [float(spectrum[index].real), float(spectrum[index].imag)]
            for index in sorted(fundamentals)
        ],
        "passed": passed,
    }


def analyse_stage2_v2_diagnostic_preflight(
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    captured_diagnostic_pcm: np.ndarray,
    *,
    transport_counters: Mapping[str, int],
) -> dict[str, Any]:
    """phase-1 raw만으로 두 output의 49/98 PCM level/THD를 fail-closed 판정한다."""

    owned_plan, owned_submitted = validate_stage2_v2_live_safe_fallback_plan(
        plan, submitted_pcm
    )
    phase_stop = int(owned_plan["live_phase_contract"]["diagnostic_phase_stop_frame"])
    captured = _finite_capture(captured_diagnostic_pcm, minimum_frames=phase_stop)
    if captured.shape[0] != phase_stop:
        raise Stage2MeasurementV2Error("diagnostic raw는 phase-1 exact frame까지만 있어야 합니다")
    expected_counters = {"xrun": 0, "clip": 0, "callback_status": 0}
    if dict(transport_counters) != expected_counters:
        raise Stage2MeasurementV2Error("diagnostic transport xrun/clip/callback은 exact 0이어야 합니다")
    per_channel_peak = np.max(np.abs(captured), axis=0)
    diagnostic_peak = float(np.max(per_channel_peak))
    adc_peak_passed = bool(
        diagnostic_peak < MAX_DIAGNOSTIC_ADC_PEAK_ABS
        and np.all(np.abs(captured) < ABSOLUTE_ADC_SAFETY_LIMIT)
    )
    quiet = captured[:24_000]
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, int], dict[int, dict[str, Any]]] = {}
    for row in owned_plan["nonlinearity_diagnostics"]["slots"]:
        start, stop = int(row["analysis_start_frame"]), int(row["analysis_stop_frame"])
        active = PATH_CHANNEL[str(row["path"])]
        submitted_window = owned_submitted[start:stop, active].astype(np.float64)
        pair = tuple(int(value) for value in row["fundamental_frequencies_hz"])
        for mic, mic_name in enumerate(("ERR", "REF")):
            result = _distortion_row(
                captured[start:stop, mic], submitted_window, pair, quiet[:, mic]
            )
            result.update(
                {
                    "path": row["path"],
                    "pair_index": int(row["pair_index"]),
                    "level_pcm": int(row["level_pcm"]),
                    "microphone": mic_name,
                }
            )
            rows.append(result)
            grouped.setdefault(
                (str(row["path"]), int(row["pair_index"]), mic), {}
            )[int(row["level_pcm"])] = result
    linearity_rows: list[dict[str, Any]] = []
    for (path, pair_index, mic), levels in grouped.items():
        if set(levels) != {49, 98}:
            raise Stage2MeasurementV2Error("diagnostic 49/98 level pair가 완전하지 않습니다")
        low = np.asarray(
            [complex(*value) for value in levels[49]["fundamental_complex"]]
        )
        high = np.asarray(
            [complex(*value) for value in levels[98]["fundamental_complex"]]
        )
        ratios = high / np.where(np.abs(low) > 0.0, low, np.nan + 0j)
        gain_error_db = float(
            np.max(np.abs(20.0 * np.log10(np.maximum(np.abs(ratios), 1.0e-300) / 2.0)))
        )
        frequencies = np.asarray(
            next(
                row["fundamental_frequencies_hz"]
                for row in owned_plan["nonlinearity_diagnostics"]["slots"]
                if row["path"] == path and int(row["pair_index"]) == pair_index
            ),
            dtype=np.float64,
        )
        phase_radians = np.abs(np.angle(ratios))
        phase_samples_by_tone = phase_radians * SAMPLE_RATE / (
            2.0 * math.pi * frequencies
        )
        phase_error_samples = float(np.max(phase_samples_by_tone))
        passed = bool(
            np.all(np.isfinite(ratios))
            and gain_error_db <= MAX_GAIN_RATIO_ERROR_DB
            and phase_error_samples <= MAX_GAIN_RATIO_PHASE_ERROR_SAMPLES
            and levels[49]["passed"]
            and levels[98]["passed"]
        )
        linearity_rows.append(
            {
                "path": path,
                "pair_index": pair_index,
                "microphone": "ERR" if mic == 0 else "REF",
                "high_to_low_gain_error_db": gain_error_db,
                "high_to_low_phase_error_samples": phase_error_samples,
                "high_to_low_phase_error_samples_by_tone": phase_samples_by_tone.tolist(),
                "passed": passed,
            }
        )
    passed = bool(
        adc_peak_passed
        and rows
        and all(row["passed"] for row in rows)
        and all(row["passed"] for row in linearity_rows)
    )
    receipt: dict[str, Any] = {
        "schema": DIAGNOSTIC_SCHEMA,
        "signal_plan_sha256": owned_plan["canonical_payload_sha256"],
        "actual_submitted_pcm_sha256": owned_plan["actual_submitted_pcm"]["sha256"],
        "diagnostic_captured_snapshot_sha256": _array_sha256(
            np.asarray(captured_diagnostic_pcm)
        ),
        "diagnostic_phase_frames": phase_stop,
        "transport_counters": expected_counters,
        "thresholds": {
            "minimum_fundamental_snr_db": MIN_FUNDAMENTAL_SNR_DB,
            "maximum_thd_imd_dbc": MAX_THD_IMD_DBC,
            "maximum_submitted_thd_imd_dbc": MAX_INPUT_THD_IMD_DBC,
            "minimum_distortion_threshold_noise_margin_db": MIN_DISTORTION_NOISE_MARGIN_DB,
            "maximum_gain_ratio_error_db": MAX_GAIN_RATIO_ERROR_DB,
            "maximum_gain_ratio_phase_error_samples": MAX_GAIN_RATIO_PHASE_ERROR_SAMPLES,
            "maximum_diagnostic_adc_peak_abs_exclusive": MAX_DIAGNOSTIC_ADC_PEAK_ABS,
            "absolute_adc_safety_limit_exclusive": ABSOLUTE_ADC_SAFETY_LIMIT,
        },
        "adc_peak_abs_by_channel": per_channel_peak.tolist(),
        "diagnostic_adc_peak_abs": diagnostic_peak,
        "adc_peak_passed": adc_peak_passed,
        "distortion_rows": rows,
        "linearity_rows": linearity_rows,
        "passed": passed,
        "ps_phase_may_start": passed,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt


def _clock_adapter_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    layout = json.loads(json.dumps(plan["layout"], allow_nan=False))
    for row in layout:
        if row.get("role") == HOLDOUT_ROLE:
            row["role"] = "holdout"
    payload: dict[str, Any] = {
        "schema": "stage2_v2_clock_adapter_to_fullband_v5_primitive",
        "source_plan_sha256": plan["canonical_payload_sha256"],
        "actual_submitted_pcm_sha256": plan["actual_submitted_pcm"]["sha256"],
        "layout": layout,
        "clock_contract": {
            "primary_pilot_bins": list(plan["pilot"]["primary_bins"]),
            "secondary_pilot_bins": list(plan["pilot"]["secondary_bins"]),
        },
    }
    payload["canonical_payload_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def estimate_stage2_v2_common_clock(
    plan: Mapping[str, Any], submitted_pcm: np.ndarray, captured_pcm: np.ndarray
) -> dict[str, Any]:
    owned_plan, owned_submitted = validate_stage2_v2_live_safe_fallback_plan(
        plan, submitted_pcm
    )
    boundary = int(
        owned_plan["live_phase_contract"]["diagnostic_phase_stop_frame"]
    )
    ps_submitted = np.asarray(owned_submitted[boundary:])
    captured = np.asarray(captured_pcm)
    if captured.dtype != np.dtype("<i4") or captured.shape != ps_submitted.shape:
        raise Stage2MeasurementV2Error("PS-local captured raw shape/dtype가 submitted와 다릅니다")
    raise Stage2MeasurementV2Error(
        "Stage-2 PS-local global-grid clock 분석은 raw-first capture commit에서 "
        "의도적으로 BLOCKED입니다; 검증 완료된 offline clock module 결속 전에는 "
        "P/S candidate authority를 만들 수 없습니다"
    )


def _capture_on_dac_grid(captured: np.ndarray, ratio: float, frames: int) -> np.ndarray:
    source = np.asarray(captured, dtype=np.float64)
    query = np.arange(frames, dtype=np.float64) / float(ratio)
    available = int(np.count_nonzero(query <= len(source) - 1))
    if available <= 0:
        raise Stage2MeasurementV2Error("clock-corrected capture interpolation support가 없습니다")
    axis = np.arange(len(source), dtype=np.float64)
    result = np.zeros((frames, 2), dtype=np.float64)
    result[:available] = np.column_stack(
        [
            CubicSpline(axis, source[:, channel], extrapolate=False)(
                query[:available]
            )
            for channel in range(2)
        ]
    )
    if not np.all(np.isfinite(result[:available])):
        raise Stage2MeasurementV2Error("clock-corrected capture가 non-finite입니다")
    return result


def _estimate_coarse_zeros(plan: Mapping[str, Any], submitted: np.ndarray, captured: np.ndarray) -> tuple[int, int]:
    zeros: list[int] = []
    for path in PATHS:
        accumulated = np.zeros(PERIOD, dtype=np.complex128)
        channel = PATH_CHANNEL[path]
        for role in FIT_ROLES:
            x = _central_period(plan, submitted, role=role, path=path).astype(np.float64)
            row = next(
                value
                for value in plan["layout"]
                if value.get("kind") == "pe_slot"
                and value.get("role") == role
                and value.get("path") == path
            )
            y = captured[
                int(row["central_start_frame"]) : int(row["central_stop_frame"]), 0
            ]
            accumulated += np.conj(np.fft.fft(x[:, channel])) * np.fft.fft(y)
        correlation = np.fft.ifft(accumulated).real
        bulk = int(np.argmax(np.abs(correlation[: MAX_COARSE_DELAY_SAMPLES + 1])))
        if bulk < COARSE_PRE_ROLL_SAMPLES:
            raise Stage2MeasurementV2Error("Stage-2 P/S bulk delay가 pre-roll보다 짧습니다")
        zeros.append(bulk - COARSE_PRE_ROLL_SAMPLES)
    return zeros[0], zeros[1]


def _projected_design(period: np.ndarray, basis: np.ndarray, zeros: tuple[int, int]) -> np.ndarray:
    # FIR operator는 deployment/training과 같은 normalized PCM -> normalized ADC
    # 단위여야 한다. actual submitted int16 bytes는 Gram/lineage의 권위 자료지만,
    # 여기서 raw integer를 그대로 쓰면 fitted FIR gain이 1/32768로 축소된다.
    source = np.asarray(period, dtype=np.float64) / 32768.0
    basis_transfer = np.fft.rfft(basis, n=PERIOD, axis=0)
    columns: list[np.ndarray] = []
    for channel in range(2):
        shifted = np.roll(source[:, channel], int(zeros[channel]))
        transfer = np.fft.rfft(shifted)[:, None] * basis_transfer
        columns.append(np.fft.irfft(transfer, n=PERIOD, axis=0))
    return np.ascontiguousarray(np.concatenate(columns, axis=1), dtype=np.float64)


def _complex_agreement(left: np.ndarray, right: np.ndarray) -> float:
    denominator = math.sqrt(
        max(float(np.vdot(left, left).real), np.finfo(np.float64).tiny)
        * max(float(np.vdot(right, right).real), np.finfo(np.float64).tiny)
    )
    return float(abs(np.vdot(left, right)) / denominator)


def _holdout_prediction_metrics(
    observed: np.ndarray, predicted: np.ndarray
) -> dict[str, float | bool]:
    observed_band = np.asarray(observed, dtype=np.complex128)
    predicted_band = np.asarray(predicted, dtype=np.complex128)
    if (
        observed_band.shape != predicted_band.shape
        or observed_band.ndim != 1
        or observed_band.size == 0
        or not np.all(np.isfinite(observed_band))
        or not np.all(np.isfinite(predicted_band))
    ):
        raise Stage2MeasurementV2Error("holdout spectrum pair가 finite/equal shape가 아닙니다")
    observed_norm = max(
        float(np.linalg.norm(observed_band)), np.finfo(np.float64).tiny
    )
    predicted_norm = max(
        float(np.linalg.norm(predicted_band)), np.finfo(np.float64).tiny
    )
    residual = observed_band - predicted_band
    relative_error = float(np.linalg.norm(residual) / observed_norm)
    nmse_db = _finite_db_ratio(
        float(np.vdot(residual, residual).real),
        float(np.vdot(observed_band, observed_band).real),
    )
    magnitude_error_db = abs(20.0 * math.log10(predicted_norm / observed_norm))
    agreement = _complex_agreement(observed_band, predicted_band)
    return {
        "complex_agreement": agreement,
        "complex_relative_error": relative_error,
        "complex_nmse_db": nmse_db,
        "magnitude_ratio_error_db": magnitude_error_db,
        "passed": bool(
            agreement >= MIN_COMPLEX_AGREEMENT
            and relative_error <= MAX_COMPLEX_RELATIVE_ERROR
            and magnitude_error_db <= MAX_MAGNITUDE_RATIO_ERROR_DB
        ),
    }


def _actuator_feasibility(
    primary_fir: np.ndarray,
    secondary_fir: np.ndarray,
    *,
    operating_level_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        level = validate_stage2_physical_operating_level_evidence(
            operating_level_evidence
        )
    except (TypeError, ValueError) as exc:
        raise Stage2MeasurementV2Error(str(exc)) from exc
    source_peak = float(level["source_operating_peak_abs"])
    limit = float(level["actuator_limit_abs"])
    if not (math.isfinite(source_peak) and math.isfinite(limit) and 0.0 < source_peak <= 1.0 and 0.0 < limit <= 1.0):
        raise Stage2MeasurementV2Error("source operating peak/actuator limit가 유효하지 않습니다")
    frequency = np.fft.rfftfreq(65_536, 1.0 / SAMPLE_RATE)
    p = np.fft.rfft(np.asarray(primary_fir, dtype=np.float64), n=65_536)
    s = np.fft.rfft(np.asarray(secondary_fir, dtype=np.float64), n=65_536)
    policy = level["feasibility_policy"]
    target_attenuation_db = float(policy["target_attenuation_db"])
    minimum_headroom_db = float(policy["minimum_additional_headroom_db"])
    if minimum_headroom_db != MINIMUM_ACTUATOR_HEADROOM_DB:
        raise Stage2MeasurementV2Error("physical level actuator headroom policy가 다릅니다")
    cancellation_fraction = 1.0 - 10.0 ** (-target_attenuation_db / 20.0)
    contract = Stage2TwoKilohertzContract.canonical()
    rows: list[dict[str, Any]] = []
    for center, (lower, upper) in zip(
        contract.octave_objective_centers_hz,
        contract.octave_objective_bands_hz,
        strict=True,
    ):
        mask = (frequency >= lower) & (frequency < upper)
        ratio = np.abs(p[mask]) / np.maximum(np.abs(s[mask]), np.finfo(np.float64).tiny)
        required = source_peak * cancellation_fraction * ratio
        worst_required = float(np.max(required))
        margin = 20.0 * math.log10(limit / max(worst_required, np.finfo(np.float64).tiny))
        rows.append(
            {
                "center_hz": float(center),
                "band_hz": [float(lower), float(upper)],
                "worst_required_control_peak_abs_for_3db": worst_required,
                "actuator_headroom_db": margin,
                "passed": bool(margin >= minimum_headroom_db),
            }
        )
    authority_mask = (frequency >= LIVE_SAFE_BAND_HZ[0]) & (
        frequency < LIVE_SAFE_BAND_HZ[1]
    )
    ideal_control = np.zeros_like(p)
    ideal_control[authority_mask] = -cancellation_fraction * p[authority_mask] / np.where(
        np.abs(s[authority_mask]) > np.finfo(np.float64).tiny,
        s[authority_mask],
        np.finfo(np.float64).tiny + 0j,
    )
    # 주파수별 |P/S|만으로 broadband crest/summation을 숨기지 않는다. 같은
    # 65,536-point bandlimited ideal controller의 impulse L1은 모든 |x[n]|<=source_peak
    # 입력에 대한 보수적 L-infinity bound다. 이 gate는 actuator amplitude 충분조건일
    # 뿐이며 causal lead/phase 가능성을 주장하지 않는다.
    ideal_impulse = np.fft.irfft(ideal_control, n=65_536)
    broadband_required_bound = source_peak * float(np.sum(np.abs(ideal_impulse)))
    broadband_margin = 20.0 * math.log10(
        limit / max(broadband_required_bound, np.finfo(np.float64).tiny)
    )
    broadband = {
        "method": "bandlimited_ideal_control_impulse_l1_induced_linf_upper_bound",
        "authority_band_hz": list(LIVE_SAFE_BAND_HZ),
        "source_peak_assumption_abs": source_peak,
        "worst_case_required_control_peak_upper_bound_abs": broadband_required_bound,
        "actuator_headroom_db": broadband_margin,
        "passed": bool(broadband_margin >= minimum_headroom_db),
        "amplitude_admission_only_no_causality_or_attenuation_claim": True,
    }
    return {
        "operating_level_evidence_sha256": level["canonical_payload_sha256"],
        "source_operating_peak_abs": source_peak,
        "actuator_limit_abs": limit,
        "target_attenuation_db": target_attenuation_db,
        "minimum_additional_headroom_db": minimum_headroom_db,
        "rows": rows,
        "broadband_time_domain_peak_bound": broadband,
        "passed": bool(all(row["passed"] for row in rows) and broadband["passed"]),
    }


def analyse_stage2_v2_capture(
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    captured_diagnostic_pcm: np.ndarray,
    captured_ps_pcm: np.ndarray,
    *,
    diagnostic_transport_counters: Mapping[str, int],
    ps_transport_counters: Mapping[str, int],
    operating_level_evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """서로 다른 ADC clock stream인 diagnostic/PS raw에서 candidate P/S를 계산한다."""

    owned_plan, owned_submitted = validate_stage2_v2_live_safe_fallback_plan(
        plan, submitted_pcm
    )
    try:
        physical_level = validate_stage2_physical_operating_level_evidence(
            operating_level_evidence
        )
    except (TypeError, ValueError) as exc:
        raise Stage2MeasurementV2Error(str(exc)) from exc
    if physical_level["signal_plan_sha256"] != owned_plan["canonical_payload_sha256"]:
        raise Stage2MeasurementV2Error("physical operating level evidence plan SHA가 다릅니다")
    boundary = int(
        owned_plan["live_phase_contract"]["diagnostic_phase_stop_frame"]
    )
    diagnostic_raw = np.asarray(captured_diagnostic_pcm)
    ps_raw = np.asarray(captured_ps_pcm)
    if diagnostic_raw.dtype != np.dtype("<i4") or diagnostic_raw.shape != (
        boundary,
        2,
    ):
        raise Stage2MeasurementV2Error("diagnostic raw가 exact phase-1 int32 shape가 아닙니다")
    if ps_raw.dtype != np.dtype("<i4") or ps_raw.shape != (
        len(owned_submitted) - boundary,
        2,
    ):
        raise Stage2MeasurementV2Error("PS raw가 exact phase-2 int32 shape가 아닙니다")
    diagnostic_receipt = analyse_stage2_v2_diagnostic_preflight(
        owned_plan,
        owned_submitted,
        diagnostic_raw,
        transport_counters=diagnostic_transport_counters,
    )
    if diagnostic_receipt["passed"] is not True:
        raise Stage2MeasurementV2Error("recomputed diagnostic raw gate가 실패했습니다")
    if dict(ps_transport_counters) != {
        "xrun": 0,
        "clip": 0,
        "callback_status": 0,
    }:
        raise Stage2MeasurementV2Error("PS transport counter가 exact zero가 아닙니다")
    clock = estimate_stage2_v2_common_clock(owned_plan, owned_submitted, ps_raw)
    corrected_ps = _capture_on_dac_grid(
        _finite_capture(ps_raw, minimum_frames=len(ps_raw)),
        float(clock["estimated_rate_ratio"]),
        len(ps_raw),
    )
    maximum_operator_stop = max(
        int(row["central_stop_frame"]) - boundary
        for row in owned_plan["layout"]
        if row.get("kind") == "pe_slot"
    )
    available_clock_frames = int(
        np.count_nonzero(
            np.arange(len(ps_raw), dtype=np.float64)
            / float(clock["estimated_rate_ratio"])
            <= len(ps_raw) - 1
        )
    )
    if maximum_operator_stop > available_clock_frames:
        raise Stage2MeasurementV2Error("PS clock correction이 operator holdout 끝까지 지원하지 않습니다")
    corrected = np.zeros_like(owned_submitted, dtype=np.float64)
    corrected[:boundary] = _finite_capture(
        diagnostic_raw, minimum_frames=boundary
    )
    corrected[boundary:] = corrected_ps
    zeros = _estimate_coarse_zeros(owned_plan, owned_submitted, corrected)
    gram = audit_stage2_v2_live_safe_dpss_gram(
        owned_plan, owned_submitted, zeros_by_path=zeros
    )
    if gram["numerical_subspace_passed"] is not True:
        raise Stage2MeasurementV2Error("actual shifted DPSS Gram rank/condition이 실패했습니다")
    basis, basis_receipt = build_stage2_bandlimited_dpss_basis()
    dof = basis.shape[1]
    ledger = Stage2HoldoutAccessLedger(owned_plan, corrected)
    coefficients: dict[str, np.ndarray] = {}
    role_firs: dict[str, np.ndarray] = {}
    role_conditions: dict[str, float] = {}
    for role in FIT_ROLES:
        designs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for path in PATHS:
            submitted_period = _central_period(
                owned_plan, owned_submitted, role=role, path=path
            )
            captured_period = ledger.read_fit_period(role=role, path=path)
            designs.append(_projected_design(submitted_period, basis, zeros))
            targets.append(np.asarray(captured_period, dtype=np.float64))
        design = np.concatenate(designs, axis=0)
        target = np.concatenate(targets, axis=0)
        singular = np.linalg.svd(design, compute_uv=False)
        condition = float(singular[0] / singular[-1])
        if not math.isfinite(condition) or condition**2 > MAX_GRAM_CONDITION:
            raise Stage2MeasurementV2Error("fit role DPSS design condition이 실패했습니다")
        coefficient, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if int(rank) != 2 * dof:
            raise Stage2MeasurementV2Error("fit role DPSS design rank가 부족합니다")
        coefficients[role] = coefficient
        role_conditions[role] = condition**2
        role_firs[role] = np.stack(
            (
                basis @ coefficient[:dof],
                basis @ coefficient[dof:],
            ),
            axis=0,
        ).transpose(0, 2, 1)
    frozen_coefficient = 0.5 * (coefficients["fit_a"] + coefficients["fit_b"])
    frozen_firs = np.stack(
        (basis @ frozen_coefficient[:dof], basis @ frozen_coefficient[dof:]), axis=0
    ).transpose(0, 2, 1)
    ledger.freeze_fit()

    frequency = np.fft.rfftfreq(PERIOD, 1.0 / SAMPLE_RATE)
    contract = Stage2TwoKilohertzContract.canonical()
    subband_rows: list[dict[str, Any]] = []
    quiet = corrected[:PERIOD]
    for path_index, path in enumerate(PATHS):
        submitted_period = _central_period(
            owned_plan, owned_submitted, role=HOLDOUT_ROLE, path=path
        )
        design = _projected_design(submitted_period, basis, zeros)
        observed = np.asarray(ledger.read_holdout_period(path=path), dtype=np.float64)
        predicted = design @ frozen_coefficient
        for mic, mic_name in enumerate(("ERR", "REF")):
            observed_spectrum = np.fft.rfft(observed[:, mic])
            predicted_spectrum = np.fft.rfft(predicted[:, mic])
            noise_spectrum = np.fft.rfft(quiet[:, mic])
            fit_a_transfer = np.fft.rfft(
                role_firs["fit_a"][path_index, mic], n=PERIOD
            )
            fit_b_transfer = np.fft.rfft(
                role_firs["fit_b"][path_index, mic], n=PERIOD
            )
            for lower, upper in contract.physical_identification_subbands_hz:
                mask = (frequency >= lower) & (frequency < upper)
                fit_agreement = _complex_agreement(
                    fit_a_transfer[mask], fit_b_transfer[mask]
                )
                holdout_metrics = _holdout_prediction_metrics(
                    observed_spectrum[mask], predicted_spectrum[mask]
                )
                holdout_agreement = float(holdout_metrics["complex_agreement"])
                response_power = float(np.mean(np.abs(observed_spectrum[mask]) ** 2))
                noise_power = max(
                    float(np.mean(np.abs(noise_spectrum[mask]) ** 2)),
                    np.finfo(np.float64).tiny,
                )
                response_snr = 10.0 * math.log10(response_power / noise_power)
                passed = bool(
                    fit_agreement >= MIN_COMPLEX_AGREEMENT
                    and holdout_agreement >= MIN_COMPLEX_AGREEMENT
                    and holdout_metrics["passed"] is True
                    and response_snr >= MIN_RESPONSE_TO_NOISE_DB
                )
                subband_rows.append(
                    {
                        "path": path,
                        "microphone": mic_name,
                        "band_hz": [float(lower), float(upper)],
                        "fit_a_fit_b_complex_agreement": fit_agreement,
                        "untouched_holdout_complex_agreement": holdout_agreement,
                        "untouched_holdout_complex_relative_error": holdout_metrics[
                            "complex_relative_error"
                        ],
                        "untouched_holdout_complex_nmse_db": holdout_metrics[
                            "complex_nmse_db"
                        ],
                        "untouched_holdout_magnitude_ratio_error_db": holdout_metrics[
                            "magnitude_ratio_error_db"
                        ],
                        "response_to_noise_db": response_snr,
                        "passed": passed,
                    }
                )
    if not all(row["passed"] for row in subband_rows):
        raise Stage2MeasurementV2Error("fit crosscheck/untouched holdout subband가 실패했습니다")
    actuator = _actuator_feasibility(
        frozen_firs[0, 0],
        frozen_firs[1, 0],
        operating_level_evidence=physical_level,
    )
    if actuator["passed"] is not True:
        raise Stage2MeasurementV2Error("실측 |P/S|의 3 dB actuator headroom이 부족합니다")
    analysis: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "status": "DPSS_RELATIVE_PS_CANDIDATE_PASS_NOT_YET_PUBLISHED",
        "signal_plan_sha256": owned_plan["canonical_payload_sha256"],
        "actual_submitted_pcm_sha256": owned_plan["actual_submitted_pcm"]["sha256"],
        "diagnostic_captured_pcm_sha256": _array_sha256(diagnostic_raw),
        "ps_captured_pcm_sha256": _array_sha256(ps_raw),
        "diagnostic_receipt_sha256": diagnostic_receipt["canonical_payload_sha256"],
        "basis_definition_sha256": basis_receipt["canonical_payload_sha256"],
        "basis_array_sha256": basis_receipt["basis_sha256"],
        "authority_band_hz": list(LIVE_SAFE_BAND_HZ),
        "unrestricted_1024tap_authority_claimed": False,
        "coarse_zeros_before_fir_samples": list(zeros),
        "clock_receipt": clock,
        "shifted_projected_gram_receipt": gram,
        "fit_role_projected_gram_conditions": role_conditions,
        "holdout_access_ledger": ledger.receipt(),
        "subband_rows": subband_rows,
        "actuator_feasibility": actuator,
        "physical_operating_level_evidence_sha256": physical_level[
            "canonical_payload_sha256"
        ],
        "two_stream_contract": owned_plan["live_phase_contract"],
        "diagnostic_transport_counters": dict(diagnostic_transport_counters),
        "ps_transport_counters": dict(ps_transport_counters),
        "training_eval_consumer_basis_binding_verified": False,
        "canonical_training_eligible": False,
    }
    analysis["canonical_payload_sha256"] = _payload_sha256(analysis)
    arrays = {
        "primary_fir_by_mic": np.ascontiguousarray(frozen_firs[0], dtype=np.float64),
        "secondary_fir_by_mic": np.ascontiguousarray(frozen_firs[1], dtype=np.float64),
        "fit_a_primary_fir_by_mic": np.ascontiguousarray(role_firs["fit_a"][0], dtype=np.float64),
        "fit_a_secondary_fir_by_mic": np.ascontiguousarray(role_firs["fit_a"][1], dtype=np.float64),
        "fit_b_primary_fir_by_mic": np.ascontiguousarray(role_firs["fit_b"][0], dtype=np.float64),
        "fit_b_secondary_fir_by_mic": np.ascontiguousarray(role_firs["fit_b"][1], dtype=np.float64),
        "dpss_basis": np.asarray(basis, dtype=np.float64),
    }
    return analysis, arrays


__all__ = [
    "ANALYSIS_SCHEMA",
    "DIAGNOSTIC_SCHEMA",
    "analyse_stage2_v2_capture",
    "analyse_stage2_v2_diagnostic_preflight",
    "estimate_stage2_v2_common_clock",
]
