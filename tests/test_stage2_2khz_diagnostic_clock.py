from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

from deep_anc.dsp.stage2_2khz_diagnostic_clock import (
    MAX_CLOCK_PPM,
    _search_rate,
    estimate_stage2_diagnostic_global_clock,
)
from deep_anc.dsp.stage2_2khz_measurement_v2 import (
    build_stage2_v2_live_safe_fallback_plan,
)


PRE_ROLL = 4_096
POST_ROLL = 8_192


@pytest.fixture(scope="module")
def diagnostic_bundle() -> tuple[dict, np.ndarray, np.ndarray]:
    plan, full = build_stage2_v2_live_safe_fallback_plan()
    stop = int(plan["live_phase_contract"]["diagnostic_phase_stop_frame"])
    submitted = np.ascontiguousarray(full[:stop], dtype="<i2")
    source = submitted.astype(np.float64) / 32768.0
    response = np.zeros_like(source)
    filters = (
        (
            np.r_[np.zeros(73), [8.0, 2.0, -1.0]],
            np.r_[np.zeros(91), [5.0, 1.0]],
        ),
        (
            np.r_[np.zeros(113), [4.0, 1.0]],
            np.r_[np.zeros(67), [7.0, -1.5, 0.5]],
        ),
    )
    for output_channel in range(2):
        for microphone in range(2):
            response[:, microphone] += lfilter(
                filters[output_channel][microphone],
                [1.0],
                source[:, output_channel],
            )
    assert float(np.max(np.abs(response))) < 0.10
    return plan, submitted, response


def _to_int32(value: np.ndarray) -> np.ndarray:
    assert float(np.max(np.abs(value))) < 1.0
    return np.rint(value * 2147483648.0).astype("<i4")


def _render_global_async(response: np.ndarray, ppm: float) -> np.ndarray:
    ratio = 1.0 + float(ppm) * 1.0e-6
    frames = PRE_ROLL + int(np.ceil(len(response) / ratio)) + POST_ROLL
    capture_index = np.arange(frames, dtype=np.float64)
    source_index = ratio * (capture_index - PRE_ROLL)
    rendered = np.column_stack(
        [
            np.interp(
                source_index,
                np.arange(len(response), dtype=np.float64),
                response[:, microphone],
                left=0.0,
                right=0.0,
            )
            for microphone in range(2)
        ]
    )
    return _to_int32(rendered)


def _render_slot_async(
    plan: dict, response: np.ndarray, slot_ppm: list[float]
) -> np.ndarray:
    rows = list(plan["nonlinearity_diagnostics"]["slots"])
    assert len(rows) == len(slot_ppm) == 8
    rendered = np.zeros((PRE_ROLL + len(response) + POST_ROLL, 2), dtype=np.float64)
    for row, ppm in zip(rows, slot_ppm, strict=True):
        start, stop = int(row["start_frame"]), int(row["stop_frame"])
        frames = stop - start
        source_index = (1.0 + float(ppm) * 1.0e-6) * np.arange(
            frames, dtype=np.float64
        )
        for microphone in range(2):
            rendered[PRE_ROLL + start : PRE_ROLL + stop, microphone] = np.interp(
                source_index,
                np.arange(frames, dtype=np.float64),
                response[start:stop, microphone],
                left=0.0,
                right=0.0,
            )
    return _to_int32(rendered)


def test_stable_async_two_by_two_lti_has_one_global_clock(
    diagnostic_bundle: tuple[dict, np.ndarray, np.ndarray],
) -> None:
    plan, submitted, response = diagnostic_bundle
    captured = _render_global_async(response, 250.0)

    receipt = estimate_stage2_diagnostic_global_clock(plan, submitted, captured)

    assert receipt["passed"] is True
    assert receipt["diagnostic_linearity_may_run"] is True
    assert receipt["ps_phase_may_start"] is False
    assert receipt["alignment"]["coarse_capture_offset_samples"] == PRE_ROLL
    assert receipt["alignment"]["passed"] is True
    assert receipt["settled_quiet_window"] == {
        "capture_start_frame": PRE_ROLL + 8_192,
        "capture_stop_frame": PRE_ROLL + 32_192,
        "stream_start_settle_frames_excluded": 8_192,
    }
    assert receipt["global_search"]["interior_optimum_passed"] is True
    assert receipt["global_search"]["ambiguity_passed"] is True
    assert receipt["global_search"]["selected_ppm"] == pytest.approx(250.0, abs=0.1)
    assert all(row["passed"] for row in receipt["slot_rows"])
    assert all(row["passed"] for row in receipt["view_rows"])
    assert len(receipt["canonical_payload_sha256"]) == 64


def test_slot_dependent_clock_regimes_fail_global_affine_gate(
    diagnostic_bundle: tuple[dict, np.ndarray, np.ndarray],
) -> None:
    plan, submitted, response = diagnostic_bundle
    captured = _render_slot_async(
        plan,
        response,
        [-500.0, 500.0, -500.0, 500.0, -500.0, 500.0, -500.0, 500.0],
    )

    receipt = estimate_stage2_diagnostic_global_clock(plan, submitted, captured)

    assert receipt["alignment"]["passed"] is True
    assert receipt["passed"] is False
    assert receipt["diagnostic_linearity_may_run"] is False
    assert receipt["global_search"]["coherence"] < 0.995
    assert all(row["passed"] is False for row in receipt["slot_rows"])
    assert max(
        row["global_endpoint_disagreement_samples"] for row in receipt["slot_rows"]
    ) > 10.0


def test_condensed_observed_rate_pattern_remains_clock_stability_fail(
    diagnostic_bundle: tuple[dict, np.ndarray, np.ndarray],
) -> None:
    plan, submitted, response = diagnostic_bundle
    # 실제 raw를 fixture로 복사하지 않는다. 관측한 부호 전환과 ±3.3 kppm regime만
    # 합성 LTI에 축약해, severe nonlinearity가 없어도 clock gate가 닫히는지 검증한다.
    captured = _render_slot_async(
        plan,
        response,
        [-730.0, 3_280.0, 3_360.0, -606.0, 2_730.0, 3_376.0, -639.0, -603.0],
    )

    receipt = estimate_stage2_diagnostic_global_clock(plan, submitted, captured)

    assert receipt["alignment"]["passed"] is True
    assert receipt["passed"] is False
    assert receipt["diagnostic_linearity_may_run"] is False
    assert receipt["global_search"]["coherence"] < 0.80
    assert any(row["search"]["boundary_optimum"] for row in receipt["slot_rows"])
    assert any(row["passed"] is False for row in receipt["view_rows"])


def test_exact_plus_1000_ppm_is_a_fail_closed_boundary(
    diagnostic_bundle: tuple[dict, np.ndarray, np.ndarray],
) -> None:
    plan, submitted, response = diagnostic_bundle
    captured = _render_global_async(response, MAX_CLOCK_PPM)

    receipt = estimate_stage2_diagnostic_global_clock(plan, submitted, captured)

    assert receipt["passed"] is False
    assert receipt["global_search"]["selected_ppm"] == pytest.approx(
        MAX_CLOCK_PPM, abs=0.1
    )
    assert receipt["global_search"]["boundary_optimum"] is True
    assert receipt["global_search"]["interior_optimum_passed"] is False


def test_equal_distinct_clock_basins_are_explicitly_ambiguous() -> None:
    centres = np.arange(22, dtype=np.float64) * 1_024.0 + 1_024.0
    views: list[dict[str, np.ndarray | float]] = []
    for ppm in (-800.0, -800.0, 800.0, 800.0):
        for frequency in (752.0, 2_200.0):
            unit = np.exp(
                2j
                * np.pi
                * frequency
                * centres
                / 48_000.0
                * ppm
                * 1.0e-6
            )
            views.append(
                {"centres": centres, "unit": unit, "frequency_hz": frequency}
            )

    search = _search_rate(views, label="synthetic_equal_distinct_basins")

    assert search["interior_optimum_passed"] is True
    assert search["runner_up_ppm"] is not None
    assert abs(search["runner_up_ppm"] - search["selected_ppm"]) > 1_000.0
    assert search["runner_up_objective_gap"] < 1.0e-5
    assert search["ambiguity_passed"] is False
    assert search["passed"] is False
