"""Stage-2 2 kHz P/S 스트림 전용 공통 ADC/DAC clock 추정기.

진단 스트림과 P/S 스트림은 서로 다른 PortAudio stream이므로 하나의 연속 시간축으로
이어 붙일 수 없다. 이 모듈은 두 번째 P/S stream의 local frame 0부터 정확히 594,944
frame만 받아, untouched operator holdout을 열지 않고 continuous disjoint pilot의 위상으로
공통 rate ratio ``q``를 추정한다.

전역 선택은 ±1000 ppm의 1 ppm grid에서 모든 interior basin을 찾은 뒤 수행한다.
선택 basin은 actual waveform을 cubic/linear로 다시 sampling하는 국소 objective로
검증하며, search boundary, runner-up ambiguity, path/microphone별 q 불일치 및 terminal
pilot phase 오차를 모두 fail-closed한다. 파일, 오디오 장치, 네트워크는 열지 않는다.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from .fullband_causal_v5 import (
    CLOCK_HARD_MAX_RESIDUAL_SAMPLES,
    CLOCK_VIEW_MAX_ENDPOINT_DISAGREEMENT_SAMPLES,
)
from .stage2_2khz_measurement_v2 import (
    FIT_ROLES,
    HOLDOUT_ROLE,
    PATH_CHANNEL,
    PERIOD,
    Stage2MeasurementV2Error,
    _array_sha256,
    _payload_sha256,
    build_stage2_v2_live_safe_fallback_plan,
)


CLOCK_SCHEMA = "stage2_2khz_ps_local_common_clock_v1"
MAX_CLOCK_PPM = 1_000.0
GLOBAL_GRID_STEP_PPM = 1.0
MIN_RUNNER_UP_OBJECTIVE_RATIO = 4.0
LOCAL_REFINEMENT_HALF_WIDTH_PPM = 1.0
LOCAL_REFINEMENT_BOUNDARY_GUARD_PPM = 0.005
MAX_CUBIC_LINEAR_ENDPOINT_DISAGREEMENT_SAMPLES = (
    CLOCK_HARD_MAX_RESIDUAL_SAMPLES
)


def _finite_tree(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, label=f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, label=f"{label}[{index}]")
        return
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise Stage2MeasurementV2Error(f"{label}에 non-finite clock 값이 있습니다")


def _global_grid_basin_search(
    objective: Callable[[float], float],
    *,
    label: str,
) -> tuple[float, dict[str, Any]]:
    """±1000 ppm 전체 grid와 모든 interior basin을 deterministic하게 검사한다."""

    ppm_grid = np.linspace(
        -MAX_CLOCK_PPM,
        MAX_CLOCK_PPM,
        int(round(2.0 * MAX_CLOCK_PPM / GLOBAL_GRID_STEP_PPM)) + 1,
    )
    ratio_grid = 1.0 + ppm_grid * 1.0e-6
    values = np.asarray([objective(float(q)) for q in ratio_grid], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise Stage2MeasurementV2Error(
            f"{label} global clock objective가 finite non-negative가 아닙니다"
        )
    if int(np.argmin(values)) in {0, len(values) - 1}:
        raise Stage2MeasurementV2Error(
            f"{label} global clock optimum이 ±1000 ppm boundary입니다"
        )
    local = (
        np.flatnonzero(
            (values[1:-1] <= values[:-2])
            & (values[1:-1] <= values[2:])
        )
        + 1
    )
    if local.size < 2:
        raise Stage2MeasurementV2Error(
            f"{label} global clock runner-up interior basin이 없습니다"
        )

    basins: list[dict[str, Any]] = []
    for index in local:
        refined = minimize_scalar(
            objective,
            bounds=(float(ratio_grid[index - 1]), float(ratio_grid[index + 1])),
            method="bounded",
            options={"xatol": 1.0e-12},
        )
        if (
            not refined.success
            or not math.isfinite(float(refined.x))
            or not math.isfinite(float(refined.fun))
            or float(refined.fun) < 0.0
        ):
            raise Stage2MeasurementV2Error(
                f"{label} global clock basin refinement가 실패했습니다"
            )
        basins.append(
            {
                "grid_index": int(index),
                "grid_ppm": float(ppm_grid[index]),
                "refined_rate_ratio": float(refined.x),
                "refined_ppm": (float(refined.x) - 1.0) * 1.0e6,
                "objective": float(refined.fun),
            }
        )
    basins.sort(key=lambda row: (row["objective"], row["grid_index"]))
    best, runner_up = basins[:2]
    if min(float(values[0]), float(values[-1])) <= float(best["objective"]):
        raise Stage2MeasurementV2Error(
            f"{label} global clock boundary objective가 selected basin보다 작습니다"
        )
    runner_ratio = float(runner_up["objective"]) / max(
        float(best["objective"]), np.finfo(np.float64).tiny
    )
    if (
        not math.isfinite(runner_ratio)
        or runner_ratio < MIN_RUNNER_UP_OBJECTIVE_RATIO
    ):
        raise Stage2MeasurementV2Error(
            f"{label} global clock runner-up basin이 모호합니다"
        )
    receipt: dict[str, Any] = {
        "schema": "stage2_2khz_global_clock_grid_basin_v1",
        "label": label,
        "bounds_ppm": [-MAX_CLOCK_PPM, MAX_CLOCK_PPM],
        "grid_step_ppm": GLOBAL_GRID_STEP_PPM,
        "grid_points": int(len(ratio_grid)),
        "all_interior_basins_refined": True,
        "boundary_minimum_rejected": True,
        "basins": basins,
        "selected_rate_ratio": float(best["refined_rate_ratio"]),
        "selected_ppm": float(best["refined_ppm"]),
        "runner_up_to_best_objective_ratio": runner_ratio,
        "minimum_runner_up_objective_ratio": MIN_RUNNER_UP_OBJECTIVE_RATIO,
        "runner_up_ambiguity_gate_passed": True,
    }
    _finite_tree(receipt, label=label)
    return float(best["refined_rate_ratio"]), receipt


def _ps_local_rows(
    plan: Mapping[str, Any], *, ps_frames: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    phase = plan["live_phase_contract"]
    ps_start = int(phase["ps_phase_start_frame"])
    ps_stop = int(phase["ps_phase_stop_frame"])
    stream = phase["ps_stream"]
    if (
        ps_stop - ps_start != ps_frames
        or int(stream["local_start_frame"]) != 0
        or int(stream["local_stop_frame"]) != ps_frames
        or int(stream["logical_plan_start_frame"]) != ps_start
        or int(stream["logical_plan_stop_frame"]) != ps_stop
        or phase.get("common_clock_scope") != "ps_stream_only_local_coordinates"
        or phase.get("adc_clock_is_restarted_between_streams") is not True
    ):
        raise Stage2MeasurementV2Error("Stage-2 P/S local stream contract가 다릅니다")

    lead = next(row for row in plan["layout"] if row.get("kind") == "pilot_only_lead")
    tail = next(row for row in plan["layout"] if row.get("kind") == "pilot_only_tail")
    fit: dict[str, list[dict[str, Any]]] = {}
    terminal: dict[str, list[dict[str, Any]]] = {}
    for path in PATH_CHANNEL:
        rows = [
            {
                "name": "pilot_lead_reference",
                "role": "clock_fit_reference",
                "path": path,
                "global_start_frame": int(lead["start_frame"]) + PERIOD,
            }
        ]
        for role in FIT_ROLES:
            selected = next(
                row
                for row in plan["layout"]
                if row.get("kind") == "pe_slot"
                and row.get("path") == path
                and row.get("role") == role
            )
            rows.append(
                {
                    "name": f"{role}_{path}",
                    "role": role,
                    "path": path,
                    "global_start_frame": int(selected["central_start_frame"]),
                }
            )
        fit[path] = rows
        terminal[path] = [
            {
                "name": "pilot_tail_terminal",
                "role": "terminal_clock_validation",
                "path": path,
                "global_start_frame": int(tail["start_frame"]) + PERIOD,
            }
        ]

    holdout_ranges = [
        (
            int(row["central_start_frame"]) - ps_start,
            int(row["central_stop_frame"]) - ps_start,
        )
        for row in plan["layout"]
        if row.get("kind") == "pe_slot" and row.get("role") == HOLDOUT_ROLE
    ]
    if len(holdout_ranges) != 2:
        raise Stage2MeasurementV2Error("Stage-2 untouched holdout range가 exact 2개가 아닙니다")
    for rows in tuple(fit.values()) + tuple(terminal.values()):
        for row in rows:
            local = int(row["global_start_frame"]) - ps_start
            row["local_start_frame"] = local
            row["local_stop_frame"] = local + PERIOD
            if local < 0 or local + PERIOD > ps_frames:
                raise Stage2MeasurementV2Error("Stage-2 clock row가 P/S local stream 밖입니다")
            if any(
                max(local, lower) < min(local + PERIOD, upper)
                for lower, upper in holdout_ranges
            ):
                raise Stage2MeasurementV2Error(
                    "Stage-2 clock fit/terminal row가 untouched holdout을 엽니다"
                )
    return fit, terminal


def _actual_transfer_observations(
    plan: Mapping[str, Any],
    submitted: np.ndarray,
    captured: np.ndarray,
    rows_by_path: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]]]:
    observations: dict[str, np.ndarray] = {}
    access: dict[str, list[dict[str, Any]]] = {}
    for path, channel in PATH_CHANNEL.items():
        bins = np.asarray(plan["pilot"][f"{path}_bins"], dtype=np.int64)
        values: list[np.ndarray] = []
        receipts: list[dict[str, Any]] = []
        for row in rows_by_path[path]:
            start = int(row["local_start_frame"])
            stop = start + PERIOD
            submitted_period = submitted[start:stop].astype(np.float64)
            captured_period = captured[start:stop]
            submitted_spectrum = np.fft.rfft(submitted_period, axis=0)
            captured_spectrum = np.fft.rfft(captured_period, axis=0)
            opposite = 1 - channel
            if float(np.max(np.abs(submitted_spectrum[bins, opposite]))) > 1.0e-8:
                raise Stage2MeasurementV2Error(
                    "Stage-2 clock matching bin의 반대 DAC actual DFT가 exact zero가 아닙니다"
                )
            denominator = submitted_spectrum[bins, channel]
            if float(np.min(np.abs(denominator))) <= 1.0:
                raise Stage2MeasurementV2Error(
                    "Stage-2 clock actual submitted denominator가 너무 작습니다"
                )
            values.append(captured_spectrum[bins, :] / denominator[:, None])
            receipts.append(
                {
                    "name": str(row["name"]),
                    "role": str(row["role"]),
                    "path": path,
                    "global_start_frame": int(row["global_start_frame"]),
                    "local_start_frame": start,
                    "local_stop_frame": stop,
                    "submitted_window_sha256": _array_sha256(submitted[start:stop]),
                    "captured_window_sha256": _array_sha256(captured_period),
                }
            )
        observations[path] = np.stack(values)
        access[path] = receipts
    return observations, access


def _phase_objective(
    plan: Mapping[str, Any],
    observations: Mapping[str, np.ndarray],
    rows_by_path: Mapping[str, Sequence[Mapping[str, Any]]],
    rate_ratio: float,
    *,
    selected_view: tuple[str, int] | None = None,
) -> float:
    total = 0.0
    paths = (selected_view[0],) if selected_view is not None else tuple(PATH_CHANNEL)
    for path in paths:
        value = observations[path]
        if selected_view is not None:
            value = value[:, :, selected_view[1] : selected_view[1] + 1]
        bins = np.asarray(plan["pilot"][f"{path}_bins"], dtype=np.float64)
        centers = np.asarray(
            [float(row["local_start_frame"]) + PERIOD / 2.0 for row in rows_by_path[path]],
            dtype=np.float64,
        )
        unit = value / np.maximum(np.abs(value), np.finfo(np.float64).tiny)
        correction = np.exp(
            -2j
            * np.pi
            * centers[:, None, None]
            * bins[None, :, None]
            * (float(rate_ratio) - 1.0)
            / PERIOD
        )
        corrected = unit * correction
        mean = np.mean(corrected, axis=0, keepdims=True)
        mean /= np.maximum(np.abs(mean), np.finfo(np.float64).tiny)
        total += float(np.mean(np.abs(corrected - mean) ** 2))
    return total


def _bounded_accessors(
    captured: np.ndarray,
    captured_raw: np.ndarray,
    rows_by_path: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    q_min = 1.0 - MAX_CLOCK_PPM * 1.0e-6
    q_max = 1.0 + MAX_CLOCK_PPM * 1.0e-6
    for path, rows in rows_by_path.items():
        for row in rows:
            start = int(row["local_start_frame"])
            stop = start + PERIOD
            query_bounds = (
                start / q_min,
                start / q_max,
                (stop - 1) / q_min,
                (stop - 1) / q_max,
            )
            lower = max(0, int(math.floor(min(query_bounds))) - 3)
            upper = min(len(captured), int(math.ceil(max(query_bounds))) + 4)
            if upper - lower < PERIOD:
                raise Stage2MeasurementV2Error(
                    "Stage-2 bounded clock interpolation support가 부족합니다"
                )
            local = np.array(captured[lower:upper], dtype=np.float64, copy=True, order="C")
            grid = np.arange(lower, upper, dtype=np.float64)
            cubic = tuple(
                CubicSpline(grid, local[:, mic], extrapolate=False) for mic in range(2)
            )
            linear = tuple(
                (
                    lambda query, mic=mic, grid=grid, local=local: np.interp(
                        query, grid, local[:, mic]
                    )
                )
                for mic in range(2)
            )
            result[(path, str(row["name"]))] = {
                "cubic": cubic,
                "linear": linear,
                "lower": lower,
                "upper": upper,
                "receipt": {
                    "path": path,
                    "name": str(row["name"]),
                    "role": str(row["role"]),
                    "local_dac_start_frame": start,
                    "local_dac_stop_frame": stop,
                    "owned_adc_start_frame": lower,
                    "owned_adc_stop_frame": upper,
                    "owned_local_capture_sha256": _array_sha256(
                        captured_raw[lower:upper]
                    ),
                },
            }
    return result


def _corrected_transfer_banks(
    plan: Mapping[str, Any],
    submitted: np.ndarray,
    rows_by_path: Mapping[str, Sequence[Mapping[str, Any]]],
    accessors: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    rate_ratio: float,
    interpolation_kind: str,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for path, channel in PATH_CHANNEL.items():
        bins = np.asarray(plan["pilot"][f"{path}_bins"], dtype=np.int64)
        values: list[np.ndarray] = []
        for row in rows_by_path[path]:
            start = int(row["local_start_frame"])
            query = np.arange(start, start + PERIOD, dtype=np.float64) / float(
                rate_ratio
            )
            accessor = accessors[(path, str(row["name"]))]
            if query[0] < int(accessor["lower"]) or query[-1] > int(accessor["upper"]) - 1:
                raise Stage2MeasurementV2Error(
                    "Stage-2 q-corrected clock row가 owned capture support 밖입니다"
                )
            interpolators = accessor[interpolation_kind]
            corrected = np.column_stack(
                [interpolator(query) for interpolator in interpolators]
            )
            if not np.all(np.isfinite(corrected)):
                raise Stage2MeasurementV2Error(
                    "Stage-2 q-corrected clock waveform이 non-finite입니다"
                )
            captured_spectrum = np.fft.rfft(corrected, axis=0)
            submitted_spectrum = np.fft.rfft(
                submitted[start : start + PERIOD].astype(np.float64), axis=0
            )
            denominator = submitted_spectrum[bins, channel]
            values.append(captured_spectrum[bins, :] / denominator[:, None])
        result[path] = np.stack(values)
    return result


def _bank_objective(
    banks: Mapping[str, np.ndarray], *, selected_view: tuple[str, int] | None = None
) -> float:
    total = 0.0
    paths = (selected_view[0],) if selected_view is not None else tuple(PATH_CHANNEL)
    for path in paths:
        value = banks[path]
        if selected_view is not None:
            value = value[:, :, selected_view[1] : selected_view[1] + 1]
        unit = value / np.maximum(np.abs(value), np.finfo(np.float64).tiny)
        mean = np.mean(unit, axis=0, keepdims=True)
        mean /= np.maximum(np.abs(mean), np.finfo(np.float64).tiny)
        total += float(np.mean(np.abs(unit - mean) ** 2))
    return total


def _local_exact_refinement(
    objective: Callable[[float], float],
    *,
    center_ratio: float,
    label: str,
) -> tuple[float, dict[str, Any]]:
    half = LOCAL_REFINEMENT_HALF_WIDTH_PPM * 1.0e-6
    lower = float(center_ratio) - half
    upper = float(center_ratio) + half
    hard_lower = 1.0 - MAX_CLOCK_PPM * 1.0e-6
    hard_upper = 1.0 + MAX_CLOCK_PPM * 1.0e-6
    if lower <= hard_lower or upper >= hard_upper:
        raise Stage2MeasurementV2Error(
            f"{label} exact refinement basin이 global clock boundary에 닿습니다"
        )
    fit = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    if (
        not fit.success
        or not math.isfinite(float(fit.x))
        or not math.isfinite(float(fit.fun))
        or float(fit.fun) < 0.0
    ):
        raise Stage2MeasurementV2Error(f"{label} exact clock refinement가 실패했습니다")
    guard = LOCAL_REFINEMENT_BOUNDARY_GUARD_PPM * 1.0e-6
    if float(fit.x) - lower <= guard or upper - float(fit.x) <= guard:
        raise Stage2MeasurementV2Error(
            f"{label} exact clock refinement optimum이 local boundary입니다"
        )
    receipt = {
        "label": label,
        "center_rate_ratio": float(center_ratio),
        "bounds_rate_ratio": [lower, upper],
        "bounds_ppm": [(lower - 1.0) * 1.0e6, (upper - 1.0) * 1.0e6],
        "selected_rate_ratio": float(fit.x),
        "selected_ppm": (float(fit.x) - 1.0) * 1.0e6,
        "objective": float(fit.fun),
        "boundary_guard_ppm": LOCAL_REFINEMENT_BOUNDARY_GUARD_PPM,
        "interior_optimum_passed": True,
    }
    _finite_tree(receipt, label=label)
    return float(fit.x), receipt


def estimate_stage2_ps_local_clock(
    plan: Mapping[str, Any],
    ps_submitted_pcm: np.ndarray,
    ps_captured_pcm: np.ndarray,
) -> dict[str, Any]:
    """실제 두 번째 stream의 local raw에서 공통 affine clock rate ratio를 구한다."""

    canonical_plan, canonical_full = build_stage2_v2_live_safe_fallback_plan()
    if not isinstance(plan, Mapping) or dict(plan) != canonical_plan:
        raise Stage2MeasurementV2Error("Stage-2 clock plan이 exact canonical v2가 아닙니다")
    phase = canonical_plan["live_phase_contract"]
    start = int(phase["ps_phase_start_frame"])
    stop = int(phase["ps_phase_stop_frame"])
    expected_submitted = np.asarray(canonical_full[start:stop])
    submitted = np.asarray(ps_submitted_pcm)
    captured_raw = np.asarray(ps_captured_pcm)
    if (
        submitted.dtype != np.dtype("<i2")
        or submitted.shape != expected_submitted.shape
        or not np.array_equal(submitted, expected_submitted)
    ):
        raise Stage2MeasurementV2Error(
            "Stage-2 clock submitted PCM이 exact 594944-frame canonical PS slice가 아닙니다"
        )
    if captured_raw.dtype != np.dtype("<i4") or captured_raw.shape != submitted.shape:
        raise Stage2MeasurementV2Error(
            "Stage-2 clock captured PCM은 exact 594944-frame int32 PS raw여야 합니다"
        )
    captured = captured_raw.astype(np.float64) / 2147483648.0
    fit_rows, terminal_rows = _ps_local_rows(canonical_plan, ps_frames=len(submitted))
    fit_observations, fit_access = _actual_transfer_observations(
        canonical_plan, submitted, captured, fit_rows
    )

    global_ratio, global_search = _global_grid_basin_search(
        lambda q: _phase_objective(
            canonical_plan, fit_observations, fit_rows, q
        ),
        label="combined_path_microphone_phase",
    )
    view_searches: dict[str, Any] = {}
    for path in PATH_CHANNEL:
        for mic, mic_name in enumerate(("ERR", "REF")):
            name = f"{path}_{mic_name}"
            _view_ratio, view_searches[name] = _global_grid_basin_search(
                lambda q, path=path, mic=mic: _phase_objective(
                    canonical_plan,
                    fit_observations,
                    fit_rows,
                    q,
                    selected_view=(path, mic),
                ),
                label=name,
            )

    all_rows = {
        path: list(fit_rows[path]) + list(terminal_rows[path]) for path in PATH_CHANNEL
    }
    accessors = _bounded_accessors(captured, captured_raw, all_rows)

    selected: dict[str, float] = {}
    exact_refinements: dict[str, Any] = {}
    for kind in ("cubic", "linear"):
        selected[kind], exact_refinements[kind] = _local_exact_refinement(
            lambda q, kind=kind: _bank_objective(
                _corrected_transfer_banks(
                    canonical_plan,
                    submitted,
                    fit_rows,
                    accessors,
                    rate_ratio=q,
                    interpolation_kind=kind,
                )
            ),
            center_ratio=global_ratio,
            label=f"combined_{kind}_waveform",
        )
    cubic_linear_endpoint = abs(selected["cubic"] - selected["linear"]) * len(
        submitted
    )
    if cubic_linear_endpoint > MAX_CUBIC_LINEAR_ENDPOINT_DISAGREEMENT_SAMPLES:
        raise Stage2MeasurementV2Error(
            "Stage-2 cubic/linear clock endpoint crosscheck가 실패했습니다"
        )

    view_ratios: dict[str, float] = {}
    view_refinements: dict[str, Any] = {}
    for path in PATH_CHANNEL:
        for mic, mic_name in enumerate(("ERR", "REF")):
            name = f"{path}_{mic_name}"
            view_ratios[name], view_refinements[name] = _local_exact_refinement(
                lambda q, path=path, mic=mic: _bank_objective(
                    _corrected_transfer_banks(
                        canonical_plan,
                        submitted,
                        fit_rows,
                        accessors,
                        rate_ratio=q,
                        interpolation_kind="cubic",
                    ),
                    selected_view=(path, mic),
                ),
                center_ratio=selected["cubic"],
                label=f"{name}_cubic_waveform",
            )
    view_endpoint = (max(view_ratios.values()) - min(view_ratios.values())) * len(
        submitted
    )
    if view_endpoint > CLOCK_VIEW_MAX_ENDPOINT_DISAGREEMENT_SAMPLES:
        raise Stage2MeasurementV2Error(
            "Stage-2 path/microphone shared-q endpoint consensus가 실패했습니다"
        )

    maximum_terminal_phase_error = 0.0
    terminal_rows_receipt: list[dict[str, Any]] = []
    for kind in ("cubic", "linear"):
        fit_banks = _corrected_transfer_banks(
            canonical_plan,
            submitted,
            fit_rows,
            accessors,
            rate_ratio=selected[kind],
            interpolation_kind=kind,
        )
        terminal_banks = _corrected_transfer_banks(
            canonical_plan,
            submitted,
            terminal_rows,
            accessors,
            rate_ratio=selected[kind],
            interpolation_kind=kind,
        )
        for path in PATH_CHANNEL:
            bins = np.asarray(canonical_plan["pilot"][f"{path}_bins"], dtype=np.float64)
            reference = np.mean(fit_banks[path], axis=0)
            terminal_value = terminal_banks[path][0]
            phase_error = np.angle(terminal_value * np.conj(reference))
            sample_error = np.abs(phase_error) * PERIOD / (
                2.0 * np.pi * bins[:, None]
            )
            observed = float(np.max(sample_error))
            maximum_terminal_phase_error = max(maximum_terminal_phase_error, observed)
            terminal_rows_receipt.append(
                {
                    "interpolation_kind": kind,
                    "path": path,
                    "maximum_phase_error_samples": observed,
                }
            )
    if maximum_terminal_phase_error > CLOCK_HARD_MAX_RESIDUAL_SAMPLES:
        raise Stage2MeasurementV2Error(
            "Stage-2 terminal pilot clock phase validation이 실패했습니다"
        )

    receipt: dict[str, Any] = {
        "schema": CLOCK_SCHEMA,
        "signal_plan_sha256": canonical_plan["canonical_payload_sha256"],
        "clock_scope": "ps_stream_only_local_coordinates",
        "ps_stream_global_frame_range": [start, stop],
        "ps_stream_local_frame_range": [0, len(submitted)],
        "ps_stream_frames": int(len(submitted)),
        "ps_submitted_pcm_sha256": _array_sha256(submitted),
        "ps_captured_pcm_sha256": _array_sha256(captured_raw),
        "global_search": global_search,
        "view_searches": view_searches,
        "exact_waveform_refinements": exact_refinements,
        "view_waveform_refinements": view_refinements,
        "estimated_rate_ratio": selected["cubic"],
        "estimated_ppm": (selected["cubic"] - 1.0) * 1.0e6,
        "linear_rate_ratio": selected["linear"],
        "cubic_linear_endpoint_disagreement_samples": float(
            cubic_linear_endpoint
        ),
        "maximum_cubic_linear_endpoint_disagreement_samples": (
            MAX_CUBIC_LINEAR_ENDPOINT_DISAGREEMENT_SAMPLES
        ),
        "view_rate_ratios": view_ratios,
        "maximum_view_endpoint_disagreement_samples": float(view_endpoint),
        "maximum_allowed_view_endpoint_disagreement_samples": (
            CLOCK_VIEW_MAX_ENDPOINT_DISAGREEMENT_SAMPLES
        ),
        "terminal_phase_rows": terminal_rows_receipt,
        "maximum_terminal_phase_error_samples": float(
            maximum_terminal_phase_error
        ),
        "maximum_allowed_terminal_phase_error_samples": (
            CLOCK_HARD_MAX_RESIDUAL_SAMPLES
        ),
        "fit_accessed_waveform_receipts_by_path": fit_access,
        "bounded_owned_capture_receipts": [
            accessors[key]["receipt"] for key in sorted(accessors)
        ],
        "operator_holdout_accessed": False,
        "diagnostic_stream_accessed": False,
        "concatenated_single_stream_clock_claimed": False,
        "shared_q_across_paths_and_microphones": True,
        "captured_ps_local_sha256_computed": True,
        "passed": True,
    }
    _finite_tree(receipt, label="stage2_ps_local_clock_receipt")
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt


__all__ = [
    "CLOCK_SCHEMA",
    "GLOBAL_GRID_STEP_PPM",
    "MAX_CLOCK_PPM",
    "MIN_RUNNER_UP_OBJECTIVE_RATIO",
    "estimate_stage2_ps_local_clock",
]
