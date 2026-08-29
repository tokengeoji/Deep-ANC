"""보존된 v6 clock block의 short-time 비-affine 진단.

이 모듈은 P/S, delay, 학습 authority를 만들지 않는다. v6의 고정 8-line clock
block에서 각 line의 국소 주파수 scale만 다시 계산하여, 하나의 affine ``q``를
가정하기 전에 실제 capture가 시간에 따라 움직였는지 기록한다.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from .fullband_causal_v6 import (
    CLOCK_BINS,
    FS,
    MAX_CLOCK_PPM,
    PERIOD,
    V6ClockAdmissionError,
    build_plan_v6,
    estimate_common_clock_v6,
    preoptimizer_spectral_line_admission_v6,
)


SCHEMA = "fullband_causal_v6_short_time_clock_forensics_v1"
WINDOW_SAMPLES = 8_192
HOP_SAMPLES = 1_024
MODE_HISTOGRAM_WIDTH_PPM = 250.0
MODE_MINIMUM_SEPARATION_PPM = 1_000.0
MODE_MEMBERSHIP_RADIUS_PPM = 500.0
MODE_MINIMUM_MEMBERS = 10
FAILURE_SCHEMA = "fullband_causal_v6_live_delay_failure_v1"
FAILURE_KEYS = {
    "schema",
    "status",
    "raw",
    "external_post_receipt",
    "failure_stage",
    "optimizer_started",
    "error",
    "available_snr_receipt",
    "analysis_published",
    "operator_published",
    "canonical_training_eligible",
    "hardware_sample_slip_authority",
    "failure_payload_sha256",
}


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label}가 lowercase SHA-256이 아닙니다")
    return value


def validate_failure_binding_v6(
    value: Mapping[str, Any],
    *,
    raw_relative_path: str,
    raw_file_sha256: str,
    receipt_relative_path: str,
    receipt_file_sha256: str,
) -> dict[str, Any]:
    """기존 v6 failure JSON을 raw/receipt bytes와 교차 결속한다."""

    if not isinstance(value, Mapping) or set(value) != FAILURE_KEYS:
        raise ValueError("v6 failure key 집합이 exact하지 않습니다")
    failure = dict(value)
    if failure["schema"] != FAILURE_SCHEMA or failure["status"] != "FAILED":
        raise ValueError("v6 failure schema/status가 다릅니다")
    expected_raw_sha = _require_sha256(raw_file_sha256, label="raw file SHA")
    expected_receipt_sha = _require_sha256(
        receipt_file_sha256, label="post receipt file SHA"
    )
    if failure["raw"] != {
        "path": raw_relative_path,
        "file_sha256": expected_raw_sha,
    }:
        raise ValueError("v6 failure raw binding이 admitted raw와 다릅니다")
    if failure["external_post_receipt"] != {
        "path": receipt_relative_path,
        "file_sha256": expected_receipt_sha,
    }:
        raise ValueError("v6 failure receipt binding이 admitted receipt와 다릅니다")
    if (
        type(failure["failure_stage"]) is not str
        or not failure["failure_stage"]
        or type(failure["optimizer_started"]) is not bool
        or type(failure["error"]) is not str
        or not failure["error"]
        or (
            failure["available_snr_receipt"] is not None
            and not isinstance(failure["available_snr_receipt"], Mapping)
        )
    ):
        raise ValueError("v6 failure stage/error/available receipt 계약이 잘못됐습니다")
    if (
        failure["analysis_published"] is not False
        or failure["operator_published"] is not False
        or failure["canonical_training_eligible"] is not False
        or failure["hardware_sample_slip_authority"] is not False
    ):
        raise ValueError("v6 failure가 금지된 authority를 주장합니다")
    declared = _require_sha256(
        failure["failure_payload_sha256"], label="failure payload SHA"
    )
    core = {
        key: item
        for key, item in failure.items()
        if key != "failure_payload_sha256"
    }
    if hashlib.sha256(_canonical_json_bytes(core)).hexdigest() != declared:
        raise ValueError("v6 failure payload SHA가 내용과 다릅니다")
    # 이 도구는 2026-08-29 affine ambiguity 실패의 forensic 전용이다. 다른 단계의
    # failure를 같은 관련 artifact로 조용히 표시하지 않는다.
    if (
        failure["failure_stage"] != "global_grid_basin_search"
        or failure["optimizer_started"] is not True
        or "multimodal ambiguous" not in failure["error"]
    ):
        raise ValueError("v6 failure가 대상 global clock ambiguity가 아닙니다")
    return failure


def replay_affine_clock_admission_v6(
    *,
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    captured_pcm: np.ndarray,
) -> dict[str, Any]:
    """현재 코드로 affine admission을 재실행하되 어떤 authority도 발행하지 않는다."""

    try:
        receipt = estimate_common_clock_v6(
            plan=plan,
            submitted_pcm=submitted_pcm,
            captured_pcm=captured_pcm,
        )
    except V6ClockAdmissionError as error:
        return {
            "schema": "fullband_causal_v6_affine_clock_admission_replay_v1",
            "authority": "diagnostic_only_no_clock_no_plant_no_training_authority",
            "passed": False,
            "failure_stage": error.stage,
            "optimizer_started": error.optimizer_started,
            "error": f"{type(error).__name__}: {error}",
            "available_receipt": error.available_receipt,
        }
    return {
        "schema": "fullband_causal_v6_affine_clock_admission_replay_v1",
        "authority": "diagnostic_only_no_clock_no_plant_no_training_authority",
        "passed": True,
        "failure_stage": None,
        "optimizer_started": True,
        "error": None,
        "available_receipt": receipt,
    }


def _validate_exact_inputs(
    plan: Mapping[str, Any], submitted_pcm: np.ndarray, captured_pcm: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if not isinstance(plan, Mapping):
        raise ValueError("short-time 진단 plan은 mapping이어야 합니다")
    expected_plan, expected_pcm = build_plan_v6(
        raw_session_relative_path=str(plan.get("raw_session_relative_path", ""))
    )
    submitted = np.asarray(submitted_pcm)
    captured = np.asarray(captured_pcm)
    if (
        dict(plan) != expected_plan
        or submitted.dtype != expected_pcm.dtype
        or submitted.shape != expected_pcm.shape
        or not submitted.flags.c_contiguous
        or not np.array_equal(submitted, expected_pcm)
    ):
        raise ValueError("short-time 진단 입력이 exact v6 plan/PCM이 아닙니다")
    if (
        captured.dtype != np.dtype("<i4")
        or captured.shape != submitted.shape
        or not captured.flags.c_contiguous
    ):
        raise ValueError("short-time 진단 capture는 exact C-contiguous int32 [frame,2]여야 합니다")
    return expected_plan, submitted, captured.astype(np.float64, copy=False)


def _clock_rows(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = [row for row in plan["layout"] if row.get("kind") == "clock_block"]
    if len(rows) != 8:
        raise ValueError("short-time 진단에는 exact v6 clock block 8개가 필요합니다")
    return rows


def _short_time_line_rate_ppm(
    signal: np.ndarray,
    *,
    absolute_start_frame: int,
    frequencies_hz: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = HOP_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """한 연속 clock 구간의 line별 인접-window fractional scale을 계산한다."""

    value = np.asarray(signal, dtype=np.float64)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if (
        value.ndim != 1
        or value.size < window_samples + hop_samples
        or not np.all(np.isfinite(value))
    ):
        raise ValueError("short-time clock signal 길이가 부족합니다")
    if (
        frequencies.ndim != 1
        or frequencies.size == 0
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies <= 0.0)
        or np.any(frequencies >= FS / 2.0)
    ):
        raise ValueError("short-time clock frequency가 잘못됐습니다")
    if (
        type(absolute_start_frame) is not int
        or absolute_start_frame < 0
        or type(window_samples) is not int
        or type(hop_samples) is not int
        or window_samples <= 0
        or hop_samples <= 0
        or window_samples % hop_samples
    ):
        raise ValueError("short-time window/hop 계약이 잘못됐습니다")

    local_starts = np.arange(
        0, value.size - window_samples + 1, hop_samples, dtype=np.int64
    )
    if local_starts.size < 2:
        raise ValueError("short-time clock window가 두 개 미만입니다")
    window = np.hanning(window_samples)
    local_sample = np.arange(window_samples, dtype=np.float64)
    phase_rows = []
    for frequency in frequencies:
        carrier = np.exp(-2j * np.pi * frequency * local_sample / FS)
        phase = []
        for local_start in local_starts:
            frame = value[local_start : local_start + window_samples]
            frame = (frame - float(np.mean(frame))) * window
            absolute = int(absolute_start_frame) + int(local_start)
            phasor = np.sum(frame * carrier) * np.exp(
                -2j * np.pi * frequency * absolute / FS
            )
            numerical_floor = (
                64.0
                * np.finfo(np.float64).eps
                * max(float(np.sum(np.abs(frame))), 1.0)
            )
            if abs(phasor) <= numerical_floor:
                raise ValueError("short-time clock line phasor가 수치적으로 식별되지 않습니다")
            phase.append(float(np.angle(phasor)))
        phase_rows.append(np.unwrap(np.asarray(phase, dtype=np.float64)))
    phase_array = np.asarray(phase_rows)
    rate_ppm = (
        np.diff(phase_array, axis=1)
        / (2.0 * np.pi * (hop_samples / FS) * frequencies[:, None])
        * 1.0e6
    )
    midpoint_frames = (
        int(absolute_start_frame)
        + local_starts[:-1]
        + (window_samples - 1.0) / 2.0
        + hop_samples / 2.0
    )
    return rate_ppm, midpoint_frames.astype(np.float64)


def _deterministic_modes(values_ppm: np.ndarray) -> list[dict[str, Any]]:
    """진단 표시용 1-D histogram mode를 결정론적으로 요약한다.

    mode는 admission gate가 아니며, 폭/간격/반경은 report에 그대로 공개한다.
    """

    values = np.asarray(values_ppm, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("clock mode 입력이 finite 1-D가 아닙니다")
    width = MODE_HISTOGRAM_WIDTH_PPM
    lower = math.floor(float(np.min(values)) / width) * width
    upper = math.ceil(float(np.max(values)) / width) * width
    if upper <= lower:
        upper = lower + width
    edges = np.arange(lower, upper + width, width, dtype=np.float64)
    counts, _ = np.histogram(values, edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    peaks = [
        index
        for index, count in enumerate(counts)
        if count > 0
        and count >= counts[max(0, index - 1)]
        and count >= counts[min(len(counts) - 1, index + 1)]
    ]
    ordered = sorted(
        peaks,
        key=lambda index: (-int(counts[index]), abs(float(centers[index])), float(centers[index])),
    )
    selected: list[int] = []
    for index in ordered:
        if int(counts[index]) < MODE_MINIMUM_MEMBERS:
            continue
        if all(
            abs(float(centers[index] - centers[other]))
            >= MODE_MINIMUM_SEPARATION_PPM
            for other in selected
        ):
            selected.append(index)
    result = []
    for index in sorted(selected, key=lambda item: float(centers[item])):
        member = np.abs(values - centers[index]) <= MODE_MEMBERSHIP_RADIUS_PPM
        result.append(
            {
                "histogram_center_ppm": float(centers[index]),
                "histogram_count": int(counts[index]),
                "membership_count": int(np.sum(member)),
                "membership_median_ppm": float(np.median(values[member])),
            }
        )
    return result


def diagnose_short_time_clock_v6(
    *,
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    captured_pcm: np.ndarray,
) -> dict[str, Any]:
    """Exact v6 raw의 local frequency scale을 계산하는 diagnostic-only report."""

    exact_plan, submitted, captured = _validate_exact_inputs(
        plan, submitted_pcm, captured_pcm
    )
    rows = _clock_rows(exact_plan)
    pre_rows = [row for row in rows if row["stage"] == "preterminal_fit"]
    terminal_rows = [row for row in rows if row["stage"] == "terminal_validation"]
    pre_repeats = np.stack(
        [
            np.stack(
                [captured[start : start + PERIOD] for start in row["central_repeat_starts"]]
            )
            for row in pre_rows
        ]
    )
    terminal_repeats = np.stack(
        [
            np.stack(
                [captured[start : start + PERIOD] for start in row["central_repeat_starts"]]
            )
            for row in terminal_rows
        ]
    )
    pre_snr = preoptimizer_spectral_line_admission_v6(pre_repeats)
    terminal_snr = preoptimizer_spectral_line_admission_v6(terminal_repeats)

    frequencies = np.asarray(CLOCK_BINS, dtype=np.float64) * FS / PERIOD
    all_step_medians: list[float] = []
    all_step_mads: list[float] = []
    block_reports = []
    for row in rows:
        start = int(row["central_repeat_starts"][0])
        stop = start + 2 * PERIOD
        view_rates = []
        midpoints: np.ndarray | None = None
        for microphone in range(2):
            rate, observed_midpoints = _short_time_line_rate_ppm(
                captured[start:stop, microphone],
                absolute_start_frame=start,
                frequencies_hz=frequencies,
            )
            view_rates.append(rate)
            if midpoints is None:
                midpoints = observed_midpoints
            elif not np.array_equal(midpoints, observed_midpoints):
                raise AssertionError("ERR/REF short-time midpoint가 다릅니다")
        stacked = np.concatenate(view_rates, axis=0)
        median = np.median(stacked, axis=0)
        mad = np.median(np.abs(stacked - median[None, :]), axis=0)
        all_step_medians.extend(float(value) for value in median)
        all_step_mads.extend(float(value) for value in mad)
        assert midpoints is not None
        block_reports.append(
            {
                "epoch": str(row["epoch"]),
                "stage": str(row["stage"]),
                "path": str(row["path"]),
                "central_frame_range": [start, stop],
                "step_count": int(median.size),
                "rate_ppm_median": float(np.median(median)),
                "rate_ppm_p05": float(np.percentile(median, 5.0)),
                "rate_ppm_p95": float(np.percentile(median, 95.0)),
                "cross_line_mic_mad_ppm_median": float(np.median(mad)),
                "steps": [
                    {
                        "midpoint_frame": float(frame),
                        "rate_ppm_median": float(rate),
                        "cross_line_mic_mad_ppm": float(spread),
                    }
                    for frame, rate, spread in zip(midpoints, median, mad, strict=True)
                ],
            }
        )

    rate_values = np.asarray(all_step_medians, dtype=np.float64)
    mad_values = np.asarray(all_step_mads, dtype=np.float64)
    inside = np.abs(rate_values) <= MAX_CLOCK_PPM
    report = {
        "schema": SCHEMA,
        "authority": "diagnostic_only_no_clock_no_plant_no_training_authority",
        "analysis_admission_eligible": False,
        "clock_estimate_authority": False,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
        "attenuation_assessed": False,
        "plant_identification_assessed": False,
        "signal_plan_payload_sha256": exact_plan["canonical_payload_sha256"],
        "actual_submitted_pcm_sha256": exact_plan["actual_submitted_pcm_sha256"],
        "captured_array_sha256": _array_sha256(np.asarray(captured_pcm)),
        "algorithm": {
            "scope": "central_two_periods_only_half_open_prefix_suffix_excluded",
            "window": "numpy_hanning",
            "window_samples": WINDOW_SAMPLES,
            "hop_samples": HOP_SAMPLES,
            "fixed_line_bins": list(CLOCK_BINS),
            "microphones": ["ERR", "REF"],
            "per_step_reducer": "median_of_8_lines_x_2_microphones",
            "spread": "median_absolute_deviation_without_consistency_scaling",
            "declared_affine_search_bounds_ppm": [-MAX_CLOCK_PPM, MAX_CLOCK_PPM],
            "mode_histogram_width_ppm": MODE_HISTOGRAM_WIDTH_PPM,
            "mode_minimum_separation_ppm": MODE_MINIMUM_SEPARATION_PPM,
            "mode_membership_radius_ppm": MODE_MEMBERSHIP_RADIUS_PPM,
            "mode_minimum_members": MODE_MINIMUM_MEMBERS,
            "mode_summary_is_admission_gate": False,
            "local_phasor_quality_scope": (
                "numerical_nonzero_only_outer_full_period_preoptimizer_snr_admission_applies"
            ),
        },
        "preoptimizer_snr": {
            "preterminal": pre_snr,
            "terminal": terminal_snr,
        },
        "summary": {
            "step_count": int(rate_values.size),
            "rate_ppm_median": float(np.median(rate_values)),
            "rate_ppm_p01": float(np.percentile(rate_values, 1.0)),
            "rate_ppm_p05": float(np.percentile(rate_values, 5.0)),
            "rate_ppm_p95": float(np.percentile(rate_values, 95.0)),
            "rate_ppm_p99": float(np.percentile(rate_values, 99.0)),
            "rate_ppm_minimum": float(np.min(rate_values)),
            "rate_ppm_maximum": float(np.max(rate_values)),
            "declared_affine_search_bounds_member_count": int(np.sum(inside)),
            "declared_affine_search_bounds_member_fraction": float(np.mean(inside)),
            "cross_line_mic_mad_ppm_median": float(np.median(mad_values)),
            "cross_line_mic_mad_ppm_p95": float(np.percentile(mad_values, 95.0)),
            "diagnostic_modes": _deterministic_modes(rate_values),
        },
        "blocks": block_reports,
    }
    return report


__all__ = [
    "SCHEMA",
    "diagnose_short_time_clock_v6",
    "replay_affine_clock_admission_v6",
    "validate_failure_binding_v6",
]
