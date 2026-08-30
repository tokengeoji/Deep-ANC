"""Committed v6 checkpoint capture를 위한 fail-closed live-delay 분석 코어.

오디오 장치를 열거나 artifact를 쓰지 않는다. exact v6 plan/PCM과 v6 duplex
telemetry를 먼저 검증하고, :func:`estimate_common_clock_v6`가 PASS한 뒤에만 P/S
operator 식별을 연다. terminal clock은 clock 검증에만, operator holdout은 고정된
최종 식 이후의 단 한 번의 admission에만 사용한다.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.interpolate import CubicSpline

from ..audio_duplex_v5 import DUPLEX_TELEMETRY_SCHEMA as _V5_TELEMETRY_SCHEMA
from ..audio_duplex_v6 import DUPLEX_TELEMETRY_SCHEMA as V6_TELEMETRY_SCHEMA
from .control_band_contract import BroadbandFullOctaveContractV3
from .fullband_causal_v5 import MAX_CONDITION, _exact_condition_audit_with_shifts_v5
from .fullband_causal_v6 import (
    CLOCK_BINS,
    CLOCK_EPOCHS,
    PERIOD,
    RAW_DEFAULT,
    SCHEMA as PLAN_SCHEMA,
    V6ClockAdmissionError,
    _canonicalize_condition_receipt_v6,
    build_plan_v6,
    estimate_common_clock_v6,
)
from .fullband_live_delay_core import (
    COMPACT_PRE_ROLL_SAMPLES,
    COMPACT_SUPPORT_SAMPLES,
    FINAL_FIXED_AVERAGE_WEIGHTS,
    FIT_ROLES,
    FULL_CAUSAL_SUPPORT_SAMPLES,
    MAX_BAND_RELATIVE_RESIDUAL,
    MAX_COARSE_DELAY_SAMPLES,
    MAX_COMPACT_ROUNDTRIP_RELATIVE,
    MAX_FRACTIONAL_ROUNDTRIP_ERROR_SAMPLES,
    MIN_BAND_COMPLEX_VECTOR_AGREEMENT,
    MIN_RESPONSE_TO_NOISE_DB,
    MIN_TARGET_BIN_DENSITY,
    MICROPHONES,
    PATHS,
    _array_sha256,
    _characterize_full_candidate,
    _compact_roundtrip,
    _fit_candidate,
    _payload_sha256,
    _score_candidate_role,
    derive_stationary_err_timing,
    validate_duplex_telemetry_auxiliary as _validate_v5_telemetry,
)


EXPECTED_PLAN_SHA256 = (
    "8b37213a13131a071e10527c948580c906dfd914a1134e98a640ead259ba42f7"
)
EXPECTED_PCM_SHA256 = (
    "4e8a66b983af872192624bd6759282058cfe4a845460111a24bcd684b22551a3"
)
ANALYSIS_SCHEMA = "fullband_committed_v6_live_delay_core_v1"
SHIFTED_CONDITION_SCHEMA = "fullband_causal_shifted_exact_gram_condition_v6"
OPERATOR_RECEIPT_SCHEMA = "fullband_v6_fixed_average_compact_operator_receipt_v1"
OPERATOR_ARRAY_NAMES = frozenset(
    {
        "primary_compact_fir_by_mic",
        "secondary_compact_fir_by_mic",
        "primary_zeros_before_fir",
        "secondary_zeros_before_fir",
        "support_samples",
        "separate_fractional_phase_applications",
    }
)
ANALYSIS_KEYS = frozenset(
    {
        "schema",
        "status",
        "signal_plan",
        "duplex_telemetry_auxiliary",
        "clock",
        "broadband_noise",
        "control_band_contract",
        "control_band_contract_sha256",
        "full_unshifted_causal_identification",
        "timing",
        "compact_refit",
        "candidate_fit_cross_preterminal_scores",
        "final_fixed_average",
        "holdout_policy",
        "captured_raw_binding",
        "raw_publisher_bound",
        "canonical_training_eligible",
        "hardware_slip_authority_available",
        "analysis_sha256",
    }
)
OPERATOR_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "signal_plan_payload_sha256",
        "actual_submitted_pcm_sha256",
        "captured_adc_pcm_sha256",
        "duplex_telemetry_receipt_sha256",
        "clock_receipt_sha256",
        "timing_receipt_sha256",
        "shifted_condition_payload_sha256",
        "fixed_average_formula_payload_sha256",
        "final_score_payload_sha256",
        "broadband_noise_receipt_sha256",
        "operator_array_sha256",
        "raw_publisher_bound",
        "canonical_training_eligible",
        "hardware_sample_slip_authority_available",
        "canonical_payload_sha256",
    }
)


def validate_committed_v6_plan_and_derive_windows(
    plan: Mapping[str, Any], submitted_pcm: np.ndarray
) -> tuple[dict[str, Any], dict[tuple[str, str], tuple[int, int]]]:
    """Pinned plan bytes에서만 fit_a/fit_b/holdout 중앙 PE window를 유도한다."""

    submitted = np.asarray(submitted_pcm)
    if submitted.dtype != np.dtype("<i2") or submitted.ndim != 2 or submitted.shape[1] != 2:
        raise ValueError("v6 actual submitted PCM은 exact <i2 [frame,2]여야 합니다")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("v6 signal plan schema가 다릅니다")
    if plan.get("canonical_payload_sha256") != EXPECTED_PLAN_SHA256:
        raise ValueError("committed v6 plan SHA가 아닙니다")
    payload = {key: value for key, value in plan.items() if key != "canonical_payload_sha256"}
    if _payload_sha256(payload) != EXPECTED_PLAN_SHA256:
        raise ValueError("v6 plan payload가 pinned SHA와 일치하지 않습니다")
    if _array_sha256(submitted) != EXPECTED_PCM_SHA256:
        raise ValueError("committed v6 actual PCM SHA가 아닙니다")
    if plan.get("actual_submitted_pcm_sha256") != EXPECTED_PCM_SHA256:
        raise ValueError("v6 plan의 PCM SHA가 pinned PCM과 다릅니다")

    expected_plan, expected_pcm = build_plan_v6(
        raw_session_relative_path=str(plan.get("raw_session_relative_path", RAW_DEFAULT))
    )
    if dict(plan) != expected_plan or not np.array_equal(submitted, expected_pcm):
        raise ValueError("입력 plan/PCM이 exact committed v6 builder 결과가 아닙니다")

    layout = list(plan["layout"])
    if int(layout[0]["start_frame"]) != 0 or int(layout[-1]["stop_frame"]) != len(submitted):
        raise ValueError("v6 layout의 처음/끝 frame이 PCM과 다릅니다")
    if any(int(left["stop_frame"]) != int(right["start_frame"]) for left, right in zip(layout, layout[1:])):
        raise ValueError("v6 layout이 contiguous하지 않습니다")
    expected_order = [
        ("clock_block", "fit_pre_0", "primary"),
        ("clock_block", "fit_pre_0", "secondary"),
        ("near_white_pe_slot", "fit_a", "primary"),
        ("near_white_pe_slot", "fit_a", "secondary"),
        ("clock_block", "fit_pre_1", "primary"),
        ("clock_block", "fit_pre_1", "secondary"),
        ("near_white_pe_slot", "fit_b", "primary"),
        ("near_white_pe_slot", "fit_b", "secondary"),
        ("clock_block", "fit_pre_2", "primary"),
        ("clock_block", "fit_pre_2", "secondary"),
        ("near_white_pe_slot", "holdout", "primary"),
        ("near_white_pe_slot", "holdout", "secondary"),
        ("clock_block", "terminal_post_holdout", "primary"),
        ("clock_block", "terminal_post_holdout", "secondary"),
    ]
    actual_order = [
        (row["kind"], row.get("epoch", row.get("role")), row["path"])
        for row in layout
    ]
    if actual_order != expected_order:
        raise ValueError("v6 layout 순서가 exact checkpoint/PE order가 아닙니다")

    windows: dict[tuple[str, str], tuple[int, int]] = {}
    for role in ("fit_a", "fit_b", "holdout"):
        for path in PATHS:
            rows = [row for row in layout if row.get("role") == role and row["path"] == path]
            if len(rows) != 1:
                raise ValueError(f"v6 {role}/{path} PE slot이 exact 하나가 아닙니다")
            row = rows[0]
            start = int(row["central_start_frame"])
            stop = int(row["central_stop_frame"])
            if stop - start != PERIOD or row["continuous_clock_pilot_present"] is not False:
                raise ValueError("v6 PE 중앙 period/continuous-pilot 계약이 다릅니다")
            windows[(role, path)] = (start, stop)

    receipt = {
        "schema": "fullband_committed_v6_plan_layout_v1",
        "signal_plan_payload_sha256": EXPECTED_PLAN_SHA256,
        "actual_submitted_pcm_sha256": EXPECTED_PCM_SHA256,
        "window_source": "exact_v6_plan_layout_only_no_external_mapping",
        "fit_roles": list(FIT_ROLES),
        "terminal_operator_holdout_role": "holdout",
        "clock_fit_epochs": list(CLOCK_EPOCHS[:3]),
        "clock_terminal_validation_epoch": CLOCK_EPOCHS[3],
        "windows": [
            {"role": role, "path": path, "start_frame": windows[(role, path)][0], "stop_frame": windows[(role, path)][1]}
            for role in ("fit_a", "fit_b", "holdout")
            for path in PATHS
        ],
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt, windows


def validate_duplex_telemetry_v6(
    telemetry: Mapping[str, Any],
    *,
    captured_adc_pcm: np.ndarray,
    expected_submitted_pcm: np.ndarray,
) -> dict[str, Any]:
    """동일 transport 필드를 가진 v6 schema만 v5 검증 primitive로 exact 검사한다."""

    if telemetry.get("schema") != V6_TELEMETRY_SCHEMA:
        raise ValueError("audio_duplex_v6 telemetry schema가 아닙니다")
    adapted = dict(telemetry)
    pre_open = {
        name: adapted.pop(name, None)
        for name in (
            "pre_open_monotonic_started",
            "pre_open_monotonic_completed",
            "pre_open_monotonic_elapsed_seconds",
        )
    }
    capture_started = telemetry.get("capture_monotonic_started")
    if not all(type(value) is float and np.isfinite(value) for value in pre_open.values()):
        raise ValueError("v6 pre-open monotonic telemetry가 exact finite float가 아닙니다")
    if type(capture_started) is not float or not np.isfinite(capture_started):
        raise ValueError("v6 capture monotonic start가 exact finite float가 아닙니다")
    if (
        pre_open["pre_open_monotonic_started"] < 0.0
        or pre_open["pre_open_monotonic_completed"]
        < pre_open["pre_open_monotonic_started"]
        or pre_open["pre_open_monotonic_elapsed_seconds"] < 0.0
        or not math.isclose(
            pre_open["pre_open_monotonic_completed"]
            - pre_open["pre_open_monotonic_started"],
            pre_open["pre_open_monotonic_elapsed_seconds"],
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        or capture_started < pre_open["pre_open_monotonic_completed"]
    ):
        raise ValueError("v6 pre-open/capture monotonic 순서 또는 elapsed가 잘못됐습니다")
    adapted["schema"] = _V5_TELEMETRY_SCHEMA
    receipt = _validate_v5_telemetry(
        adapted,
        captured_adc_pcm=captured_adc_pcm,
        expected_submitted_pcm=expected_submitted_pcm,
    )
    receipt = dict(receipt)
    receipt.pop("sha256", None)
    receipt["schema"] = "fullband_duplex_telemetry_auxiliary_receipt_v6"
    receipt["source_schema"] = V6_TELEMETRY_SCHEMA
    receipt["v5_transport_validator_reused_without_v5_schema_acceptance"] = True
    receipt["pre_open_timing"] = pre_open
    receipt["sha256"] = _payload_sha256(receipt)
    return receipt


def _bounded_resample(
    captured: np.ndarray,
    *,
    start: int,
    stop: int,
    q: float,
    label: str,
    access_log: list[dict[str, Any]],
) -> np.ndarray:
    query = np.arange(start, stop, dtype=np.float64) / float(q)
    lower = max(0, int(math.floor(float(query[0]))) - 3)
    upper = min(len(captured), int(math.ceil(float(query[-1]))) + 4)
    if query[0] < lower or query[-1] > upper - 1 or upper - lower < 4:
        raise ValueError("v6 q-corrected window가 bounded capture support 밖입니다")
    local = np.array(captured[lower:upper], dtype=np.float64, copy=True, order="C")
    grid = np.arange(lower, upper, dtype=np.float64)
    result = np.column_stack(
        [CubicSpline(grid, local[:, mic], extrapolate=False)(query) for mic in range(2)]
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("v6 q-corrected window에 non-finite가 있습니다")
    access_log.append(
        {
            "access_label": label,
            "logical_dac_start_frame": start,
            "logical_dac_stop_frame": stop,
            "owned_adc_start_frame": lower,
            "owned_adc_stop_frame": upper,
            "owned_local_capture_sha256": _array_sha256(local),
        }
    )
    return result


def _role_rows_v6(
    submitted: np.ndarray,
    captured: np.ndarray,
    windows: Mapping[tuple[str, str], tuple[int, int]],
    *,
    role: str,
    q: float,
    access_log: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    x_rows, y_rows = [], []
    for path in PATHS:
        start, stop = windows[(role, path)]
        x_rows.append(submitted[start:stop].astype(np.float64))
        y_rows.append(
            _bounded_resample(
                captured,
                start=start,
                stop=stop,
                q=q,
                label=f"operator:{role}:{path}",
                access_log=access_log,
            )
        )
    return np.stack(x_rows), np.stack(y_rows)


def _broadband_clock_half_difference_noise_v6(
    plan: Mapping[str, Any],
    captured: np.ndarray,
    *,
    q: float,
    access_log: list[dict[str, Any]],
    bands: Sequence[Sequence[float]],
) -> tuple[tuple[np.ndarray, ...], np.ndarray, dict[str, Any]]:
    """세 preterminal epoch의 q-corrected repeat half-difference를 broadband noise로 쓴다."""

    spectra: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    rows = [
        row
        for row in plan["layout"]
        if row.get("kind") == "clock_block" and row.get("stage") == "preterminal_fit"
    ]
    if len(rows) != 6:
        raise ValueError("v6 broadband noise에는 exact 6 preterminal clock blocks가 필요합니다")
    for row in rows:
        repeats = []
        for index, start in enumerate(row["central_repeat_starts"]):
            repeats.append(
                _bounded_resample(
                    captured,
                    start=int(start),
                    stop=int(start) + PERIOD,
                    q=q,
                    label=f"noise:{row['epoch']}:{row['path']}:repeat_{index}",
                    access_log=access_log,
                )
            )
        half_difference = (repeats[0] - repeats[1]) / 2.0
        spectrum = np.fft.rfft(half_difference, axis=0)
        spectra.append(spectrum)
        sources.append(
            {
                "epoch": row["epoch"],
                "path": row["path"],
                "complex_half_difference_spectrum_sha256": _array_sha256(spectrum),
            }
        )
    frequency = np.fft.rfftfreq(PERIOD, 1.0 / 48_000)
    band_rows = []
    fixed = set(CLOCK_BINS)
    for index, (lower, upper) in enumerate(bands):
        mask = (frequency >= float(lower)) & (frequency <= float(upper))
        indices = np.flatnonzero(mask)
        power = np.mean(
            np.stack([np.abs(value[mask]) ** 2 for value in spectra]), axis=(0, 1)
        )
        band_rows.append(
            {
                "band_index": index,
                "band_hz": [float(lower), float(upper)],
                "broadband_bin_count": int(len(indices)),
                "nonfixed_clock_bin_count": int(sum(int(value) not in fixed for value in indices)),
                "mean_half_difference_power_by_mic": np.asarray(power).tolist(),
            }
        )
    all_bins = np.ones(PERIOD // 2 + 1, dtype=np.bool_)
    receipt = {
        "schema": "fullband_v6_q_corrected_broadband_repeat_half_difference_noise_v1",
        "clock_fit_epochs": list(CLOCK_EPOCHS[:3]),
        "terminal_clock_epoch_excluded": CLOCK_EPOCHS[3],
        "terminal_clock_used_for_noise_fit_or_tuning": False,
        "source_block_count": len(sources),
        "sources": sources,
        "noise_definition": "q_corrected_complex_fft_of_(central_repeat_0-central_repeat_1)/2",
        "aggregation_used_by_score": "mean_power_across_six_preterminal_path_epoch_views",
        "fixed_clock_bins_only": False,
        "all_rfft_bins_available": True,
        "rfft_bin_count": int(len(all_bins)),
        "band_rows": band_rows,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return tuple(spectra), all_bins, receipt


def exact_shifted_condition_audit_v6(
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    *,
    zeros_by_path: Sequence[int],
    support: int = COMPACT_SUPPORT_SAMPLES,
) -> dict[str, Any]:
    """Exact v6 builder/PCM에 실제 P/S shift를 적용한 1024-support Gram audit."""

    validate_committed_v6_plan_and_derive_windows(plan, submitted_pcm)
    if len(zeros_by_path) != 2 or any(type(value) is not int or value < 0 for value in zeros_by_path):
        raise ValueError("v6 shifted Gram zeros는 exact non-negative int P/S 두 개여야 합니다")
    receipt = _exact_condition_audit_with_shifts_v5(
        plan,
        np.asarray(submitted_pcm),
        support=int(support),
        zeros_by_path=tuple(int(value) for value in zeros_by_path),
        schema=SHIFTED_CONDITION_SCHEMA,
    )
    if receipt["signal_plan_payload_sha256"] != EXPECTED_PLAN_SHA256:
        raise AssertionError("v6 shifted Gram receipt가 pinned plan과 결속되지 않았습니다")
    return _canonicalize_condition_receipt_v6(receipt)


def _score_v6(**kwargs: Any) -> list[dict[str, Any]]:
    rows = _score_candidate_role(**kwargs)
    for row in rows:
        row["noise_authority"] = "q_corrected_broadband_clock_repeat_half_difference"
        row["fixed_clock_bins_only"] = False
    return rows


def _roundtrip_v6(**kwargs: Any) -> dict[str, Any]:
    receipt = _compact_roundtrip(**kwargs)
    receipt.pop("canonical_payload_sha256", None)
    receipt["schema"] = "broadband_repeat_half_difference_cross_role_representation_gate_v6"
    for row in receipt["rows"]:
        row["broadband_repeat_half_difference_noise_power"] = row.pop(
            "pilot_exact_null_noise_power"
        )
    receipt["uncertainty_definition"] = (
        "max(q-corrected broadband repeat half-difference noise power,"
        "min(full observed residual power,compact observed residual power))"
    )
    receipt["fixed_clock_bins_only"] = False
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt


def validate_analysis_operator_v6(
    analysis: Mapping[str, Any], operator: Mapping[str, Any]
) -> dict[str, Any]:
    """post publisher 직전 v6 analysis/operator self-binding을 exact 검증한다."""

    if not isinstance(analysis, Mapping) or set(analysis) != ANALYSIS_KEYS:
        raise ValueError("offline analysis top-level key 집합이 exact v6가 아닙니다")
    if analysis.get("schema") != ANALYSIS_SCHEMA:
        raise ValueError("offline analysis가 exact v6 core schema가 아닙니다")
    if analysis.get("status") != "OFFLINE_MATH_PASS_RAW_PUBLISHER_AUTHORITY_UNBOUND":
        raise ValueError("offline analysis status가 exact v6 PASS 상태가 아닙니다")
    if (
        analysis.get("raw_publisher_bound") is not False
        or analysis.get("canonical_training_eligible") is not False
        or analysis.get("hardware_slip_authority_available") is not False
    ):
        raise ValueError("offline analysis가 canonical/slip 권한을 확대했습니다")
    analysis_payload = {
        key: value for key, value in analysis.items() if key != "analysis_sha256"
    }
    if analysis.get("analysis_sha256") != _payload_sha256(analysis_payload):
        raise ValueError("offline analysis self SHA가 유효하지 않습니다")

    if not isinstance(operator, Mapping) or set(operator) != OPERATOR_ARRAY_NAMES | {"receipt"}:
        raise ValueError("offline operator는 exact 6 arrays+receipt여야 합니다")
    receipt = operator.get("receipt")
    if not isinstance(receipt, Mapping) or set(receipt) != OPERATOR_RECEIPT_KEYS:
        raise ValueError("offline operator receipt key 집합이 exact v6가 아닙니다")
    if receipt.get("schema") != OPERATOR_RECEIPT_SCHEMA:
        raise ValueError("offline operator receipt가 exact v6 schema가 아닙니다")
    receipt_payload = {
        key: value for key, value in receipt.items() if key != "canonical_payload_sha256"
    }
    if receipt.get("canonical_payload_sha256") != _payload_sha256(receipt_payload):
        raise ValueError("offline operator receipt self SHA가 유효하지 않습니다")
    if (
        receipt.get("raw_publisher_bound") is not False
        or receipt.get("signal_plan_payload_sha256") != EXPECTED_PLAN_SHA256
        or receipt.get("actual_submitted_pcm_sha256") != EXPECTED_PCM_SHA256
        or receipt.get("canonical_training_eligible") is not False
        or receipt.get("hardware_sample_slip_authority_available") is not False
    ):
        raise ValueError("offline operator receipt가 canonical/slip 권한을 확대했습니다")
    plan = analysis.get("signal_plan")
    if (
        not isinstance(plan, Mapping)
        or plan.get("schema") != "fullband_committed_v6_plan_layout_v1"
        or plan.get("signal_plan_payload_sha256") != EXPECTED_PLAN_SHA256
        or plan.get("actual_submitted_pcm_sha256") != EXPECTED_PCM_SHA256
    ):
        raise ValueError("analysis signal-plan receipt가 pinned v6 plan/PCM이 아닙니다")
    expected_plan, expected_pcm = build_plan_v6()
    expected_plan_receipt, _ = validate_committed_v6_plan_and_derive_windows(
        expected_plan, expected_pcm
    )
    if dict(plan) != expected_plan_receipt:
        raise ValueError("analysis signal-plan receipt가 committed v6 layout과 다릅니다")

    final = analysis.get("final_fixed_average")
    if not isinstance(final, Mapping) or set(final) != {
        "formula", "roundtrip_on_fit_roles", "score", "operator_receipt"
    }:
        raise ValueError("analysis final_fixed_average key 집합이 exact하지 않습니다")
    if final.get("operator_receipt") != receipt:
        raise ValueError("analysis 내부와 returned operator receipt가 다릅니다")
    compact = analysis.get("compact_refit")
    if not isinstance(compact, Mapping) or set(compact) != {
        "support_samples",
        "different_P_S_zeros",
        "candidate_roundtrip",
        "shifted_support_1024_exact_condition_receipt",
    }:
        raise ValueError("analysis compact_refit key 집합이 exact하지 않습니다")

    def component_sha(
        value: Any, *, schema: str, sha_key: str, label: str,
        exact_keys: set[str],
    ) -> str:
        if not isinstance(value, Mapping) or set(value) != exact_keys:
            raise ValueError(f"{label} key 집합이 exact v6가 아닙니다")
        if value.get("schema") != schema:
            raise ValueError(f"{label} schema가 exact v6가 아닙니다")
        payload = {key: item for key, item in value.items() if key != sha_key}
        observed = value.get(sha_key)
        if observed != _payload_sha256(payload):
            raise ValueError(f"{label} self SHA가 유효하지 않습니다")
        return str(observed)

    telemetry_sha = component_sha(
        analysis["duplex_telemetry_auxiliary"],
        schema="fullband_duplex_telemetry_auxiliary_receipt_v6",
        sha_key="sha256",
        label="duplex telemetry receipt",
        exact_keys={
            "schema", "source_schema", "callback_count",
            "exact_256_frame_callbacks", "captured_frames", "submitted_frames",
            "resolved_input_device", "resolved_output_device",
            "device_indices_are_auxiliary_capture_evidence",
            "device_identity_binding_authority",
            "device_identity_must_be_bound_by_raw_adapter",
            "timestamps_finite_strict_monotonic", "timestamp_delta_diagnostics",
            "statuses_all_zero", "portaudio_xrun_status_witness",
            "hardware_slip_authority", "timestamps_used_to_estimate_clock_q",
            "slip_samples_field_expected_or_fabricated",
            "actual_submitted_pcm_exact_match", "actual_submitted_pcm_sha256",
            "capture_valid_mask_all_true", "submitted_valid_mask_all_true",
            "valid_mask_sha256", "array_sha256",
            "pre_open_timing",
            "v5_transport_validator_reused_without_v5_schema_acceptance", "sha256",
        },
    )
    clock = analysis["clock"]
    clock_sha = component_sha(
        clock,
        schema="fullband_causal_common_clock_v6",
        sha_key="canonical_payload_sha256",
        label="clock receipt",
        exact_keys={
            "schema", "signal_plan_payload_sha256", "actual_submitted_pcm_sha256",
            "fixed_line_bins", "preoptimizer_snr_admission",
            "terminal_preoptimizer_snr_admission", "global_search",
            "view_global_search", "view_rate_ratios",
            "maximum_view_endpoint_disagreement_samples", "interpolation",
            "selected_rate_ratio", "selected_ppm",
            "cubic_linear_endpoint_disagreement_samples",
            "maximum_terminal_phase_error_samples", "clock_fit_epochs",
            "clock_terminal_validation_epoch", "operator_holdout_accessed",
            "accessed_clock_frame_ranges", "captured_full_sha256_computed",
            "passed", "canonical_payload_sha256",
        },
    )
    if (
        clock.get("passed") is not True
        or clock.get("operator_holdout_accessed") is not False
        or clock.get("signal_plan_payload_sha256") != EXPECTED_PLAN_SHA256
        or clock.get("actual_submitted_pcm_sha256") != EXPECTED_PCM_SHA256
    ):
        raise ValueError("clock receipt가 pinned v6 PASS가 아닙니다")
    timing_sha = component_sha(
        analysis["timing"],
        schema="fullband_fit_roles_stationary_err_timing_v1",
        sha_key="sha256",
        label="timing receipt",
        exact_keys={
            "schema", "timing_authority_microphone", "REF_role", "fit_roles",
            "holdout_used_for_threshold_support_peak_or_candidate_tuning",
            "stationarity", "paths", "plant_delays", "lead", "sha256",
        },
    )
    shifted_sha = component_sha(
        compact["shifted_support_1024_exact_condition_receipt"],
        schema=SHIFTED_CONDITION_SCHEMA,
        sha_key="canonical_payload_sha256",
        label="shifted condition receipt",
        exact_keys={
            "schema", "signal_plan_payload_sha256", "actual_submitted_pcm_sha256",
            "condition_scope", "fit_roles", "support_samples",
            "zeros_before_fir_samples", "role_condition_numbers",
            "joint_fit_condition_number", "periodic_normal_matrix_gram_condition_number",
            "minimum_eigenvalue", "maximum_eigenvalue", "maximum_allowed",
            "longer_supports", "operator_definition", "operator_definition_sha256",
            "operator_normal_vector_relative_error",
            "operator_quadratic_form_relative_error",
            "operator_quadratic_form_maximum_allowed",
            "operator_quadratic_form_crosscheck_passed",
            "operator_quadratic_form_probe_receipts", "passed",
            "numeric_canonicalization", "canonical_payload_sha256",
        },
    )
    shifted = compact["shifted_support_1024_exact_condition_receipt"]
    shifted_roles = shifted.get("role_condition_numbers")
    if (
        shifted.get("passed") is not True
        or shifted.get("signal_plan_payload_sha256") != EXPECTED_PLAN_SHA256
        or shifted.get("actual_submitted_pcm_sha256") != EXPECTED_PCM_SHA256
        or shifted.get("support_samples") != COMPACT_SUPPORT_SAMPLES
        or shifted.get("maximum_allowed") != MAX_CONDITION
        or type(shifted.get("joint_fit_condition_number")) not in {int, float}
        or float(shifted["joint_fit_condition_number"]) > MAX_CONDITION
        or not isinstance(shifted_roles, Mapping)
        or set(shifted_roles) != set(FIT_ROLES)
        or any(
            type(value) not in {int, float} or float(value) > MAX_CONDITION
            for value in shifted_roles.values()
        )
        or shifted.get("operator_quadratic_form_crosscheck_passed") is not True
        or type(shifted.get("operator_quadratic_form_relative_error"))
        not in {int, float}
        or type(shifted.get("operator_quadratic_form_maximum_allowed"))
        not in {int, float}
        or float(shifted["operator_quadratic_form_relative_error"])
        > float(shifted["operator_quadratic_form_maximum_allowed"])
    ):
        raise ValueError("shifted condition receipt가 pinned v6 PASS 의미와 다릅니다")
    formula_sha = component_sha(
        final["formula"],
        schema="fullband_v6_predeclared_fixed_average_formula_v1",
        sha_key="canonical_payload_sha256",
        label="fixed-average formula",
        exact_keys={
            "schema", "fixed_before_operator_holdout_open", "candidate_order",
            "candidate_weights", "selection_or_holdout_dependent_weighting",
            "full_candidate_array_sha256", "compact_candidate_array_sha256",
            "fixed_average_full_array_sha256", "fixed_average_compact_array_sha256",
            "canonical_payload_sha256",
        },
    )
    formula = final["formula"]
    full_hashes = formula.get("full_candidate_array_sha256")
    compact_hashes = formula.get("compact_candidate_array_sha256")
    if (
        formula.get("fixed_before_operator_holdout_open") is not True
        or formula.get("candidate_order") != list(FIT_ROLES)
        or formula.get("candidate_weights") != list(FINAL_FIXED_AVERAGE_WEIGHTS)
        or formula.get("selection_or_holdout_dependent_weighting") is not False
        or not isinstance(full_hashes, Mapping)
        or set(full_hashes) != set(FIT_ROLES)
        or not isinstance(compact_hashes, Mapping)
        or set(compact_hashes) != set(FIT_ROLES)
        or any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in [
                *full_hashes.values(),
                *compact_hashes.values(),
                formula.get("fixed_average_full_array_sha256"),
                formula.get("fixed_average_compact_array_sha256"),
            ]
        )
    ):
        raise ValueError("fixed-average formula 의미가 exact v6와 다릅니다")
    score = final["score"]
    score_sha = component_sha(
        score,
        schema="fullband_v6_fixed_average_fit_terminal_score_v1",
        sha_key="canonical_payload_sha256",
        label="final score",
        exact_keys={
            "schema", "fixed_average_formula_payload_sha256", "rows",
            "expected_rows", "physical_subband_count", "physical_subbands_hz",
            "all_rows_passed", "holdout_used_only_for_terminal_admission",
            "canonical_payload_sha256",
        },
    )
    if (
        score.get("fixed_average_formula_payload_sha256") != formula_sha
        or score.get("expected_rows") != 96
        or score.get("physical_subband_count") != 8
        or score.get("all_rows_passed") is not True
        or score.get("holdout_used_only_for_terminal_admission") is not True
        or not isinstance(score.get("rows"), list)
        or len(score["rows"]) != 96
        or any(row.get("passed") is not True for row in score["rows"])
    ):
        raise ValueError("final score가 exact 8-subband/96-row PASS가 아닙니다")
    noise_sha = component_sha(
        analysis["broadband_noise"],
        schema="fullband_v6_q_corrected_broadband_repeat_half_difference_noise_v1",
        sha_key="canonical_payload_sha256",
        label="broadband noise receipt",
        exact_keys={
            "schema", "clock_fit_epochs", "terminal_clock_epoch_excluded",
            "terminal_clock_used_for_noise_fit_or_tuning", "source_block_count",
            "sources", "noise_definition", "aggregation_used_by_score",
            "fixed_clock_bins_only", "all_rfft_bins_available", "rfft_bin_count",
            "band_rows", "canonical_payload_sha256",
        },
    )
    expected_component_sha = {
        "duplex_telemetry_receipt_sha256": telemetry_sha,
        "clock_receipt_sha256": clock_sha,
        "timing_receipt_sha256": timing_sha,
        "shifted_condition_payload_sha256": shifted_sha,
        "fixed_average_formula_payload_sha256": formula_sha,
        "final_score_payload_sha256": score_sha,
        "broadband_noise_receipt_sha256": noise_sha,
    }
    if any(receipt.get(key) != value for key, value in expected_component_sha.items()):
        raise ValueError("operator receipt component SHA splice가 감지됐습니다")
    captured_binding = analysis.get("captured_raw_binding")
    if captured_binding != {
        "captured_adc_pcm_sha256": receipt.get("captured_adc_pcm_sha256"),
        "actual_submitted_pcm_sha256": EXPECTED_PCM_SHA256,
    }:
        raise ValueError("analysis captured raw binding과 operator receipt가 다릅니다")
    contract = BroadbandFullOctaveContractV3.canonical()
    physical_bands = [
        list(value) for value in contract.physical_identification_subbands_hz
    ]
    if (
        analysis.get("control_band_contract") != contract.model_dump(mode="json")
        or analysis.get("control_band_contract_sha256") != contract.digest()
        or score.get("physical_subbands_hz") != physical_bands
    ):
        raise ValueError("analysis control-band contract가 canonical full-octave v3가 아닙니다")
    score_row_keys = {
        "candidate_role", "evaluation_role", "relation", "path", "microphone",
        "band_index", "band_hz", "target_bins", "noise_bins",
        "target_bin_density_above_noise_20db", "response_to_noise_db",
        "noise_conditioned_relative_residual",
        "complex_vector_agreement_not_coherence", "independent_coherence_claimed",
        "passed", "noise_authority", "fixed_clock_bins_only",
    }
    expected_score_coordinates = {
        (evaluation_role, path, microphone, band_index)
        for evaluation_role in (*FIT_ROLES, "holdout")
        for path in PATHS
        for microphone in MICROPHONES
        for band_index in range(len(physical_bands))
    }
    observed_score_coordinates: set[tuple[str, str, str, int]] = set()
    for row in score["rows"]:
        if not isinstance(row, Mapping) or set(row) != score_row_keys:
            raise ValueError("final score row key 집합이 exact v6가 아닙니다")
        evaluation_role = row.get("evaluation_role")
        expected_relation = (
            "terminal_holdout"
            if evaluation_role == "holdout"
            else "preterminal_fit_role"
        )
        band_index = row.get("band_index")
        if (
            row.get("candidate_role") != "fixed_average"
            or evaluation_role not in {*FIT_ROLES, "holdout"}
            or row.get("relation") != expected_relation
            or row.get("path") not in PATHS
            or row.get("microphone") not in MICROPHONES
            or type(band_index) is not int
            or band_index not in range(len(physical_bands))
            or row.get("band_hz") != physical_bands[band_index]
            or row.get("independent_coherence_claimed") is not False
            or row.get("noise_authority")
            != "q_corrected_broadband_clock_repeat_half_difference"
            or row.get("fixed_clock_bins_only") is not False
            or row.get("noise_conditioned_relative_residual")
            > MAX_BAND_RELATIVE_RESIDUAL
            or row.get("complex_vector_agreement_not_coherence")
            < MIN_BAND_COMPLEX_VECTOR_AGREEMENT
            or row.get("response_to_noise_db") < MIN_RESPONSE_TO_NOISE_DB
            or row.get("target_bin_density_above_noise_20db")
            < MIN_TARGET_BIN_DENSITY
        ):
            raise ValueError("final score row 의미가 exact v6 PASS가 아닙니다")
        observed_score_coordinates.add(
            (evaluation_role, row["path"], row["microphone"], band_index)
        )
    if observed_score_coordinates != expected_score_coordinates:
        raise ValueError("final score의 3 role×2 path×2 mic×8 band coverage가 exact하지 않습니다")

    noise = analysis["broadband_noise"]
    if (
        noise.get("clock_fit_epochs") != list(CLOCK_EPOCHS[:3])
        or noise.get("terminal_clock_epoch_excluded") != CLOCK_EPOCHS[3]
        or noise.get("terminal_clock_used_for_noise_fit_or_tuning") is not False
        or noise.get("source_block_count") != 6
        or noise.get("fixed_clock_bins_only") is not False
        or noise.get("all_rfft_bins_available") is not True
        or noise.get("rfft_bin_count") != PERIOD // 2 + 1
        or not isinstance(noise.get("band_rows"), list)
        or len(noise["band_rows"]) != len(physical_bands)
        or any(
            row.get("band_index") != index
            or row.get("band_hz") != physical_bands[index]
            or type(row.get("broadband_bin_count")) is not int
            or row["broadband_bin_count"] < 8
            or type(row.get("nonfixed_clock_bin_count")) is not int
            or row["nonfixed_clock_bin_count"] < 8
            for index, row in enumerate(noise["band_rows"])
        )
    ):
        raise ValueError("broadband noise receipt 의미가 exact v6가 아닙니다")

    holdout = analysis.get("holdout_policy")
    expected_execution_order = [
        "exact_plan_and_v6_telemetry",
        "common_clock_pass",
        "full_and_compact_fit_after_clock",
        "final_formula_fixed_and_hashed",
        "operator_holdout_first_open",
        "terminal_holdout_pass",
    ]
    if (
        not isinstance(holdout, Mapping)
        or set(holdout) != {
            "operator_holdout_first_open_after_final_formula_hash",
            "execution_order", "preterminal_bounded_capture_access",
            "terminal_bounded_capture_access",
            "terminal_clock_used_for_q_selection_fit_noise_or_tuning",
            "terminal_clock_used_only_for_validation",
            "operator_holdout_used_for_clock",
            "captured_full_sha_computed_after_terminal_score",
        }
        or holdout.get("operator_holdout_first_open_after_final_formula_hash") is not True
        or holdout.get("execution_order") != expected_execution_order
        or holdout.get("terminal_clock_used_for_q_selection_fit_noise_or_tuning") is not False
        or holdout.get("terminal_clock_used_only_for_validation") is not True
        or holdout.get("operator_holdout_used_for_clock") is not False
        or holdout.get("captured_full_sha_computed_after_terminal_score") is not True
        or not isinstance(holdout.get("preterminal_bounded_capture_access"), list)
        or not isinstance(holdout.get("terminal_bounded_capture_access"), list)
        or any(
            "holdout" in row.get("access_label", "")
            for row in holdout["preterminal_bounded_capture_access"]
        )
        or any(
            not row.get("access_label", "").startswith("operator:holdout:")
            for row in holdout["terminal_bounded_capture_access"]
        )
    ):
        raise ValueError("operator holdout 정책/실행 순서가 exact v6가 아닙니다")
    declared = receipt.get("operator_array_sha256")
    if not isinstance(declared, Mapping) or set(declared) != OPERATOR_ARRAY_NAMES:
        raise ValueError("operator receipt의 exact 6 array SHA key가 다릅니다")
    actual: dict[str, str] = {}
    for name in sorted(OPERATOR_ARRAY_NAMES):
        if not isinstance(operator[name], np.ndarray):
            raise ValueError(f"operator {name}은 exact ndarray여야 합니다")
        actual[name] = _array_sha256(operator[name])
    for name in ("primary_compact_fir_by_mic", "secondary_compact_fir_by_mic"):
        if operator[name].dtype != np.dtype("<f8") or operator[name].shape != (2, 1_024):
            raise ValueError(f"operator {name} dtype/shape이 exact하지 않습니다")
        if not np.all(np.isfinite(operator[name])):
            raise ValueError(f"operator {name}에 non-finite가 있습니다")
    scalar_expectations = {
        "support_samples": 1_024,
        "separate_fractional_phase_applications": 0,
    }
    for name in (
        "primary_zeros_before_fir",
        "secondary_zeros_before_fir",
        *scalar_expectations,
    ):
        if operator[name].dtype != np.dtype("<i8") or operator[name].shape != ():
            raise ValueError(f"operator {name} dtype/shape이 exact scalar <i8이 아닙니다")
    if any(int(operator[name]) < 0 for name in ("primary_zeros_before_fir", "secondary_zeros_before_fir")):
        raise ValueError("operator P/S zeros가 음수입니다")
    if any(int(operator[name]) != expected for name, expected in scalar_expectations.items()):
        raise ValueError("operator support/fractional phase scalar가 exact하지 않습니다")
    reconstructed_compact = np.stack(
        [
            operator["primary_compact_fir_by_mic"],
            operator["secondary_compact_fir_by_mic"],
        ],
        axis=1,
    )
    if (
        final["formula"].get("fixed_average_compact_array_sha256")
        != _array_sha256(reconstructed_compact)
    ):
        raise ValueError("operator FIR가 fixed-average compact formula와 다릅니다")
    zeros = [
        int(operator["primary_zeros_before_fir"]),
        int(operator["secondary_zeros_before_fir"]),
    ]
    timing = analysis["timing"]
    timing_paths = timing.get("paths") if isinstance(timing, Mapping) else None
    if not isinstance(timing_paths, Mapping) or zeros != [
        timing_paths.get("primary", {}).get("zeros_before_compact_fir_samples"),
        timing_paths.get("secondary", {}).get("zeros_before_compact_fir_samples"),
    ]:
        raise ValueError("operator P/S zeros가 timing receipt와 다릅니다")
    shifted = compact["shifted_support_1024_exact_condition_receipt"]
    if (
        compact.get("different_P_S_zeros") != zeros
        or shifted.get("zeros_before_fir_samples") != zeros
        or compact.get("support_samples") != int(operator["support_samples"])
        or shifted.get("support_samples") != int(operator["support_samples"])
    ):
        raise ValueError("operator zeros/support가 compact/shifted receipt와 다릅니다")
    if actual != dict(declared):
        raise ValueError("operator actual 6 arrays와 receipt SHA가 다릅니다")
    validation = {
        "schema": "fullband_v6_analysis_operator_self_binding_validation_v1",
        "analysis_sha256": analysis["analysis_sha256"],
        "operator_receipt_sha256": receipt["canonical_payload_sha256"],
        "operator_array_sha256": actual,
        "exact_six_arrays": True,
        "passed": True,
    }
    validation["canonical_payload_sha256"] = _payload_sha256(validation)
    return validation


def analyze_committed_v6_live_delay(
    *,
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    captured_adc_pcm: np.ndarray,
    duplex_telemetry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """v6 plan→clock→P/S LS→compact→terminal holdout의 단일 entrypoint."""

    submitted = np.asarray(submitted_pcm)
    captured = np.asarray(captured_adc_pcm)
    if captured.dtype != np.dtype("<i4") or captured.shape != submitted.shape:
        raise ValueError("v6 actual captured ADC는 submitted와 같은 exact <i4여야 합니다")
    plan_receipt, windows = validate_committed_v6_plan_and_derive_windows(plan, submitted)
    telemetry_receipt = validate_duplex_telemetry_v6(
        duplex_telemetry,
        captured_adc_pcm=captured,
        expected_submitted_pcm=submitted,
    )

    # 이 호출이 실패하면 아래의 어떤 LS/operator primitive도 실행되지 않는다.
    clock = estimate_common_clock_v6(
        plan=plan, submitted_pcm=submitted, captured_pcm=captured
    )
    if clock.get("passed") is not True or clock.get("operator_holdout_accessed") is not False:
        raise V6ClockAdmissionError(
            "v6 common clock receipt가 operator admission을 열 수 없습니다",
            stage="clock_receipt_admission",
            optimizer_started=True,
            available_receipt=clock,
        )
    q = float(clock["selected_rate_ratio"])
    execution_order = ["exact_plan_and_v6_telemetry", "common_clock_pass"]

    contract = BroadbandFullOctaveContractV3.canonical()
    bands = contract.physical_identification_subbands_hz
    preterminal_access: list[dict[str, Any]] = []
    noise_spectra, noise_bins, noise_receipt = _broadband_clock_half_difference_noise_v6(
        plan,
        captured,
        q=q,
        access_log=preterminal_access,
        bands=bands,
    )
    fit_rows = {
        role: _role_rows_v6(
            submitted,
            captured,
            windows,
            role=role,
            q=q,
            access_log=preterminal_access,
        )
        for role in FIT_ROLES
    }
    full_candidates = {
        role: _fit_candidate(
            x_rows=fit_rows[role][0],
            y_rows=fit_rows[role][1],
            role=role,
            support=FULL_CAUSAL_SUPPORT_SAMPLES,
            zeros=(0, 0),
        )
        for role in FIT_ROLES
    }
    characterized = {
        role: _characterize_full_candidate(full_candidates[role]) for role in FIT_ROLES
    }
    timing, plants = derive_stationary_err_timing(characterized)
    zeros = (plants.primary_delay_samples, plants.secondary_delay_samples)
    compact_candidates = {
        role: _fit_candidate(
            x_rows=fit_rows[role][0],
            y_rows=fit_rows[role][1],
            role=role,
            support=COMPACT_SUPPORT_SAMPLES,
            zeros=zeros,
        )
        for role in FIT_ROLES
    }
    shifted_condition = exact_shifted_condition_audit_v6(
        plan, submitted, zeros_by_path=zeros
    )
    if shifted_condition["passed"] is not True:
        raise ValueError("v6 shifted compact exact Gram condition gate 실패")
    execution_order.append("full_and_compact_fit_after_clock")

    roundtrips = {}
    candidate_scores = {}
    for role in FIT_ROLES:
        opposite = "fit_b" if role == "fit_a" else "fit_a"
        roundtrips[role] = _roundtrip_v6(
            full_candidate=full_candidates[role],
            compact_candidate=compact_candidates[role],
            x_rows=fit_rows[opposite][0],
            y_rows=fit_rows[opposite][1],
            zeros=zeros,
            timing=timing,
            bands=bands,
            exact_zero_noise_bins=noise_bins,
            noise_spectra=noise_spectra,
            evaluation_role=opposite,
        )
        rows = []
        for evaluation in FIT_ROLES:
            rows.extend(
                _score_v6(
                    candidate_role=role,
                    evaluation_role=evaluation,
                    compact_fir=compact_candidates[role]["fir_by_mic_path"],
                    x_rows=fit_rows[evaluation][0],
                    y_rows=fit_rows[evaluation][1],
                    zeros=zeros,
                    bands=bands,
                    exact_zero_noise_bins=noise_bins,
                    noise_spectra=noise_spectra,
                )
            )
        if len(rows) != 64 or not all(row["passed"] for row in rows):
            raise ValueError("v6 candidate fit/cross 8 physical subband gate 실패")
        candidate_scores[role] = rows

    final_full = sum(
        weight * full_candidates[role]["fir_by_mic_path"]
        for role, weight in zip(FIT_ROLES, FINAL_FIXED_AVERAGE_WEIGHTS)
    )
    final_compact = sum(
        weight * compact_candidates[role]["fir_by_mic_path"]
        for role, weight in zip(FIT_ROLES, FINAL_FIXED_AVERAGE_WEIGHTS)
    )
    final_formula = {
        "schema": "fullband_v6_predeclared_fixed_average_formula_v1",
        "fixed_before_operator_holdout_open": True,
        "candidate_order": list(FIT_ROLES),
        "candidate_weights": list(FINAL_FIXED_AVERAGE_WEIGHTS),
        "selection_or_holdout_dependent_weighting": False,
        "full_candidate_array_sha256": {
            role: _array_sha256(full_candidates[role]["fir_by_mic_path"]) for role in FIT_ROLES
        },
        "compact_candidate_array_sha256": {
            role: _array_sha256(compact_candidates[role]["fir_by_mic_path"]) for role in FIT_ROLES
        },
        "fixed_average_full_array_sha256": _array_sha256(final_full),
        "fixed_average_compact_array_sha256": _array_sha256(final_compact),
    }
    final_formula["canonical_payload_sha256"] = _payload_sha256(final_formula)
    execution_order.append("final_formula_fixed_and_hashed")

    final_full_candidate = {"role": "fixed_average", "fir_by_mic_path": final_full}
    final_compact_candidate = {"role": "fixed_average", "fir_by_mic_path": final_compact}
    final_fit_rows: list[dict[str, Any]] = []
    final_roundtrip = {}
    for role in FIT_ROLES:
        final_roundtrip[role] = _roundtrip_v6(
            full_candidate=final_full_candidate,
            compact_candidate=final_compact_candidate,
            x_rows=fit_rows[role][0],
            y_rows=fit_rows[role][1],
            zeros=zeros,
            timing=timing,
            bands=bands,
            exact_zero_noise_bins=noise_bins,
            noise_spectra=noise_spectra,
            evaluation_role=role,
        )
        final_fit_rows.extend(
            _score_v6(
                candidate_role="fixed_average",
                evaluation_role=role,
                compact_fir=final_compact,
                x_rows=fit_rows[role][0],
                y_rows=fit_rows[role][1],
                zeros=zeros,
                bands=bands,
                exact_zero_noise_bins=noise_bins,
                noise_spectra=noise_spectra,
            )
        )
    if len(final_fit_rows) != 64 or not all(row["passed"] for row in final_fit_rows):
        raise ValueError("v6 fixed average preterminal 8 physical subband gate 실패")

    # Final formula가 byte-hash로 고정된 뒤 여기서 처음 operator holdout을 연다.
    holdout_access: list[dict[str, Any]] = []
    holdout = _role_rows_v6(
        submitted,
        captured,
        windows,
        role="holdout",
        q=q,
        access_log=holdout_access,
    )
    execution_order.append("operator_holdout_first_open")
    terminal_rows = _score_v6(
        candidate_role="fixed_average",
        evaluation_role="holdout",
        compact_fir=final_compact,
        x_rows=holdout[0],
        y_rows=holdout[1],
        zeros=zeros,
        bands=bands,
        exact_zero_noise_bins=noise_bins,
        noise_spectra=noise_spectra,
    )
    if len(terminal_rows) != 32 or not all(row["passed"] for row in terminal_rows):
        raise ValueError("v6 terminal operator holdout 8 physical subband gate 실패")
    execution_order.append("terminal_holdout_pass")

    final_score = {
        "schema": "fullband_v6_fixed_average_fit_terminal_score_v1",
        "fixed_average_formula_payload_sha256": final_formula["canonical_payload_sha256"],
        "rows": final_fit_rows + terminal_rows,
        "expected_rows": 96,
        "physical_subband_count": len(bands),
        "physical_subbands_hz": [list(value) for value in bands],
        "all_rows_passed": True,
        "holdout_used_only_for_terminal_admission": True,
    }
    final_score["canonical_payload_sha256"] = _payload_sha256(final_score)
    captured_sha = _array_sha256(captured)

    operator_arrays = {
        "primary_compact_fir_by_mic": np.asarray(final_compact[:, 0], dtype="<f8"),
        "secondary_compact_fir_by_mic": np.asarray(final_compact[:, 1], dtype="<f8"),
        "primary_zeros_before_fir": np.asarray(zeros[0], dtype="<i8"),
        "secondary_zeros_before_fir": np.asarray(zeros[1], dtype="<i8"),
        "support_samples": np.asarray(COMPACT_SUPPORT_SAMPLES, dtype="<i8"),
        "separate_fractional_phase_applications": np.asarray(0, dtype="<i8"),
    }
    operator_receipt = {
        "schema": OPERATOR_RECEIPT_SCHEMA,
        "signal_plan_payload_sha256": EXPECTED_PLAN_SHA256,
        "actual_submitted_pcm_sha256": EXPECTED_PCM_SHA256,
        "captured_adc_pcm_sha256": captured_sha,
        "duplex_telemetry_receipt_sha256": telemetry_receipt["sha256"],
        "clock_receipt_sha256": clock["canonical_payload_sha256"],
        "timing_receipt_sha256": timing["sha256"],
        "shifted_condition_payload_sha256": shifted_condition["canonical_payload_sha256"],
        "fixed_average_formula_payload_sha256": final_formula["canonical_payload_sha256"],
        "final_score_payload_sha256": final_score["canonical_payload_sha256"],
        "broadband_noise_receipt_sha256": noise_receipt["canonical_payload_sha256"],
        "operator_array_sha256": {
            key: _array_sha256(value) for key, value in sorted(operator_arrays.items())
        },
        "raw_publisher_bound": False,
        "canonical_training_eligible": False,
        "hardware_sample_slip_authority_available": False,
    }
    operator_receipt["canonical_payload_sha256"] = _payload_sha256(operator_receipt)
    operator = {**operator_arrays, "receipt": operator_receipt}
    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "status": "OFFLINE_MATH_PASS_RAW_PUBLISHER_AUTHORITY_UNBOUND",
        "signal_plan": plan_receipt,
        "duplex_telemetry_auxiliary": telemetry_receipt,
        "clock": clock,
        "broadband_noise": noise_receipt,
        "control_band_contract": contract.model_dump(mode="json"),
        "control_band_contract_sha256": contract.digest(),
        "full_unshifted_causal_identification": {
            "support_samples": FULL_CAUSAL_SUPPORT_SAMPLES,
            "max_delay_scan_samples": MAX_COARSE_DELAY_SAMPLES,
            "characterized": characterized,
        },
        "timing": timing,
        "compact_refit": {
            "support_samples": COMPACT_SUPPORT_SAMPLES,
            "different_P_S_zeros": list(zeros),
            "candidate_roundtrip": roundtrips,
            "shifted_support_1024_exact_condition_receipt": shifted_condition,
        },
        "candidate_fit_cross_preterminal_scores": candidate_scores,
        "final_fixed_average": {
            "formula": final_formula,
            "roundtrip_on_fit_roles": final_roundtrip,
            "score": final_score,
            "operator_receipt": operator_receipt,
        },
        "holdout_policy": {
            "operator_holdout_first_open_after_final_formula_hash": True,
            "execution_order": execution_order,
            "preterminal_bounded_capture_access": preterminal_access,
            "terminal_bounded_capture_access": holdout_access,
            "terminal_clock_used_for_q_selection_fit_noise_or_tuning": False,
            "terminal_clock_used_only_for_validation": True,
            "operator_holdout_used_for_clock": False,
            "captured_full_sha_computed_after_terminal_score": True,
        },
        "captured_raw_binding": {
            "captured_adc_pcm_sha256": captured_sha,
            "actual_submitted_pcm_sha256": EXPECTED_PCM_SHA256,
        },
        "raw_publisher_bound": False,
        "canonical_training_eligible": False,
        "hardware_slip_authority_available": False,
    }
    analysis["analysis_sha256"] = _payload_sha256(analysis)
    validate_analysis_operator_v6(analysis, operator)
    return analysis, operator


__all__ = [
    "ANALYSIS_SCHEMA",
    "EXPECTED_PCM_SHA256",
    "EXPECTED_PLAN_SHA256",
    "analyze_committed_v6_live_delay",
    "exact_shifted_condition_audit_v6",
    "validate_committed_v6_plan_and_derive_windows",
    "validate_duplex_telemetry_v6",
    "validate_analysis_operator_v6",
]
