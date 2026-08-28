from __future__ import annotations

import copy
import time

import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve

from deep_anc.audio_duplex_v5 import (
    DUPLEX_TELEMETRY_SCHEMA,
    capture_duplex_v5,
)
import deep_anc.dsp.fullband_live_delay_core as live_core
from deep_anc.dsp.fullband_causal_v5 import BLOCK, FS, build_plan_v5
from deep_anc.dsp.fullband_live_delay_core import (
    EXPECTED_PCM_SHA256,
    EXPECTED_PLAN_SHA256,
    FULL_CAUSAL_SUPPORT_SAMPLES,
    _array_sha256,
    _characterize_full_candidate,
    _compact_roundtrip,
    _fit_candidate,
    _payload_sha256,
    _predict,
    analyze_committed_v5_live_delay,
    derive_stationary_err_timing,
    validate_committed_plan_and_derive_windows,
    validate_duplex_telemetry_auxiliary,
)


def _telemetry(submitted_pcm: np.ndarray) -> dict[str, object]:
    submitted = np.asarray(submitted_pcm)
    frame_count = len(submitted)
    count = frame_count // BLOCK
    index = np.arange(count, dtype=np.int64)
    step = BLOCK / FS
    current = index.astype(np.float64) * step
    return {
        "schema": DUPLEX_TELEMETRY_SCHEMA,
        "callback_frame_semantics": "software_accounting_only_not_hardware_slip_witness",
        "portaudio_xrun_status_witness": True,
        "hardware_sample_slip_authority": False,
        "watchdog_coverage": "host_wait_until_planned_frames_plus_grace_not_hardware_deadline_witness",
        "sample_rate_hz": FS,
        "block_size": BLOCK,
        "latency": "low",
        "channels": [2, 2],
        "resolved_input_device": 1,
        "resolved_output_device": 2,
        "capture_monotonic_started": 10.0,
        "capture_monotonic_completed": 10.0 + frame_count / FS,
        "capture_monotonic_elapsed_seconds": frame_count / FS,
        "watchdog_grace_seconds": 2.0,
        "input_dtype": "<i4",
        "output_dtype": "<i2",
        "callback_sequence": index.copy(),
        "callback_start_frames": index * BLOCK,
        "callback_frame_counts": np.full(count, BLOCK, dtype=np.int64),
        "input_buffer_adc_time": current - 0.002,
        "output_buffer_dac_time": current + 0.002,
        "callback_current_time": current,
        "callback_status_bitmask": np.zeros(count, dtype=np.uint32),
        "xrun_count": 0,
        "status_present_count": 0,
        "captured_frames": frame_count,
        "submitted_frames": frame_count,
        "completed": True,
        "callback_error": None,
        "canonical_invalid_reasons": [],
        "stream_stop_error": None,
        "stream_abort_error": None,
        "stream_close_error": None,
        "termination_signal": None,
        "normal_stop_completed": True,
        "output_stop_confirmed": True,
        "actual_submitted_pcm": submitted.copy(),
        "capture_valid_mask": np.ones(frame_count, dtype=np.bool_),
        "submitted_valid_mask": np.ones(frame_count, dtype=np.bool_),
    }


class _CallbackStop(Exception):
    pass


class _CallbackAbort(Exception):
    pass


class _CaptureBackend:
    CallbackStop = _CallbackStop
    CallbackAbort = _CallbackAbort

    def __init__(self, captured_adc_pcm: np.ndarray) -> None:
        self.captured = np.asarray(captured_adc_pcm)

    def Stream(self, **kwargs):  # noqa: N802, ANN003, ANN201
        captured = self.captured

        class Stream:
            def start(self) -> None:
                for start in range(0, len(captured), BLOCK):
                    index = start // BLOCK
                    output = np.zeros((BLOCK, 2), dtype="<i2")
                    try:
                        kwargs["callback"](
                            captured[start : start + BLOCK],
                            output,
                            BLOCK,
                            {
                                "inputBufferAdcTime": index + 0.1,
                                "outputBufferDacTime": index + 0.2,
                                "currentTime": index + 0.3,
                            },
                            None,
                        )
                    except _CallbackStop:
                        break

            def stop(self, *, ignore_errors: bool) -> None:
                assert ignore_errors is False

            def abort(self, *, ignore_errors: bool) -> None:
                assert ignore_errors is False

            def close(self, *, ignore_errors: bool) -> None:
                assert ignore_errors is False

        return Stream()


def _fractional_impulse(delay: float, gain: float) -> np.ndarray:
    result = np.zeros(FULL_CAUSAL_SUPPORT_SAMPLES, dtype=np.float64)
    center = int(np.floor(delay))
    radius = 96
    index = np.arange(center - radius, center + radius + 1)
    window = np.kaiser(len(index), 10.0)
    # 실제 덕트/11.3 kHz identification 범위처럼 band-limited fractional delay.
    cutoff_cycles = 0.235
    offset = index.astype(np.float64) - delay
    result[index] = (
        gain * 2.0 * cutoff_cycles * np.sinc(2.0 * cutoff_cycles * offset) * window
    )
    return result


def _fractional_affine_raw(
    submitted: np.ndarray, *, q: float
) -> tuple[np.ndarray, dict[str, float]]:
    truth = {
        "primary_ERR": 1642.25,
        "secondary_ERR": 1500.70,
        "primary_REF": 1600.20,
        "secondary_REF": 1460.30,
    }
    primary = np.stack(
        [
            _fractional_impulse(truth["primary_ERR"], 5000.0),
            _fractional_impulse(truth["primary_REF"], 3700.0),
        ]
    )
    secondary = np.stack(
        [
            _fractional_impulse(truth["secondary_ERR"], -4400.0),
            _fractional_impulse(truth["secondary_REF"], 3300.0),
        ]
    )
    full = []
    for mic in range(2):
        response = fftconvolve(submitted[:, 0].astype(np.float64), primary[mic])
        response += fftconvolve(submitted[:, 1].astype(np.float64), secondary[mic])
        full.append(response)
    full_dac = np.column_stack(full)
    dac_grid = np.arange(len(full_dac), dtype=np.float64)
    query = np.arange(len(submitted), dtype=np.float64) * q
    sampled = np.column_stack(
        [CubicSpline(dac_grid, full_dac[:, mic])(query) for mic in range(2)]
    )
    captured = np.rint(sampled).astype("<i4")
    return captured, truth


def test_only_exact_committed_plan_pcm_can_derive_internal_windows() -> None:
    plan, submitted = build_plan_v5()
    receipt, windows = validate_committed_plan_and_derive_windows(plan, submitted)
    assert receipt["signal_plan_payload_sha256"] == EXPECTED_PLAN_SHA256
    assert receipt["actual_submitted_pcm_sha256"] == EXPECTED_PCM_SHA256
    assert receipt["window_source"].endswith("no_external_mapping")
    assert len(windows) == 6

    changed = copy.deepcopy(plan)
    changed["layout"][1]["central_start_frame"] += 1
    with pytest.raises(ValueError, match="payload|committed"):
        validate_committed_plan_and_derive_windows(changed, submitted)
    wrong_pcm = submitted.copy()
    wrong_pcm[0, 0] += 1
    with pytest.raises(ValueError, match="PCM SHA"):
        validate_committed_plan_and_derive_windows(plan, wrong_pcm)


def test_actual_capture_return_integrates_with_duplex_validator() -> None:
    submitted = np.arange(2 * BLOCK * 4, dtype="<i2").reshape(BLOCK * 4, 2)
    expected_captured = np.arange(
        2 * BLOCK * 4, dtype="<i4"
    ).reshape(BLOCK * 4, 2)
    captured, telemetry = capture_duplex_v5(
        _CaptureBackend(expected_captured),
        submitted_pcm=submitted,
        input_device=1,
        output_device=2,
    )
    assert np.array_equal(captured, expected_captured)
    receipt = validate_duplex_telemetry_auxiliary(
        telemetry,
        captured_adc_pcm=captured,
        expected_submitted_pcm=submitted,
    )
    assert receipt["source_schema"] == DUPLEX_TELEMETRY_SCHEMA
    assert receipt["resolved_input_device"] == 1
    assert receipt["resolved_output_device"] == 2
    assert receipt["device_identity_binding_authority"] is False
    assert receipt["device_identity_must_be_bound_by_raw_adapter"] is True
    assert receipt["actual_submitted_pcm_exact_match"] is True
    assert receipt["capture_valid_mask_all_true"] is True
    assert receipt["submitted_valid_mask_all_true"] is True


def test_duplex_parser_is_auxiliary_exact_and_rejects_bad_evidence() -> None:
    plan, submitted = build_plan_v5()
    del plan
    captured = np.zeros(submitted.shape, dtype="<i4")
    telemetry = _telemetry(submitted)
    receipt = validate_duplex_telemetry_auxiliary(
        telemetry,
        captured_adc_pcm=captured,
        expected_submitted_pcm=submitted,
    )
    assert receipt["hardware_slip_authority"] is False
    assert receipt["timestamps_used_to_estimate_clock_q"] is False
    assert receipt["slip_samples_field_expected_or_fabricated"] is False

    nonmonotonic = copy.deepcopy(telemetry)
    nonmonotonic["input_buffer_adc_time"][2] = np.nan  # type: ignore[index]
    with pytest.raises(ValueError, match="finite strict-monotonic"):
        validate_duplex_telemetry_auxiliary(
            nonmonotonic,
            captured_adc_pcm=captured,
            expected_submitted_pcm=submitted,
        )
    status = copy.deepcopy(telemetry)
    status["callback_status_bitmask"][3] = 2  # type: ignore[index]
    with pytest.raises(ValueError, match="status"):
        validate_duplex_telemetry_auxiliary(
            status,
            captured_adc_pcm=captured,
            expected_submitted_pcm=submitted,
        )

    extra = copy.deepcopy(telemetry)
    extra["slip_samples"] = np.zeros(len(submitted) // BLOCK, dtype=np.int64)
    with pytest.raises(ValueError, match="key 집합이 exact"):
        validate_duplex_telemetry_auxiliary(
            extra,
            captured_adc_pcm=captured,
            expected_submitted_pcm=submitted,
        )
    removed = copy.deepcopy(telemetry)
    del removed["watchdog_coverage"]
    with pytest.raises(ValueError, match="key 집합이 exact"):
        validate_duplex_telemetry_auxiliary(
            removed,
            captured_adc_pcm=captured,
            expected_submitted_pcm=submitted,
        )

    wrong_schema = copy.deepcopy(telemetry)
    wrong_schema["schema"] = "fullband_causal_v5_duplex_telemetry_v1"
    with pytest.raises(ValueError, match="schema"):
        validate_duplex_telemetry_auxiliary(
            wrong_schema,
            captured_adc_pcm=captured,
            expected_submitted_pcm=submitted,
        )
    for key, value in (
        ("resolved_input_device", True),
        ("resolved_input_device", "1"),
        ("resolved_output_device", -1),
    ):
        wrong_device = copy.deepcopy(telemetry)
        wrong_device[key] = value
        with pytest.raises(ValueError, match="stream 계약"):
            validate_duplex_telemetry_auxiliary(
                wrong_device,
                captured_adc_pcm=captured,
                expected_submitted_pcm=submitted,
            )
    for key in ("capture_valid_mask", "submitted_valid_mask"):
        false_mask = copy.deepcopy(telemetry)
        false_mask[key][0] = False  # type: ignore[index]
        with pytest.raises(ValueError, match="valid mask"):
            validate_duplex_telemetry_auxiliary(
                false_mask,
                captured_adc_pcm=captured,
                expected_submitted_pcm=submitted,
            )

    wrong_actual = copy.deepcopy(telemetry)
    wrong_actual["actual_submitted_pcm"][0, 0] += 1  # type: ignore[index]
    with pytest.raises(ValueError, match="actual submitted PCM"):
        validate_duplex_telemetry_auxiliary(
            wrong_actual,
            captured_adc_pcm=captured,
            expected_submitted_pcm=submitted,
        )

    terminated = copy.deepcopy(telemetry)
    terminated["termination_signal"] = 15
    with pytest.raises(ValueError, match="completion/status/error"):
        validate_duplex_telemetry_auxiliary(
            terminated,
            captured_adc_pcm=captured,
            expected_submitted_pcm=submitted,
        )


@pytest.fixture(scope="module")
def analyzed_fractional_fixture() -> tuple[dict, dict, dict[str, float], float]:
    started = time.monotonic()
    plan, submitted = build_plan_v5()
    q = 1.0 + 173.25e-6
    captured, truth = _fractional_affine_raw(submitted, q=q)
    analysis, operator = analyze_committed_v5_live_delay(
        plan=plan,
        submitted_pcm=submitted,
        captured_adc_pcm=captured,
        duplex_telemetry=_telemetry(submitted),
    )
    return analysis, operator, truth, time.monotonic() - started


def test_fractional_affine_raw_recovers_q_bulk_compact_and_roundtrip(
    analyzed_fractional_fixture: tuple[dict, dict, dict[str, float], float]
) -> None:
    analysis, operator, truth, _ = analyzed_fractional_fixture
    assert analysis["clock"]["selected_rate_ratio"] == pytest.approx(
        1.0 + 173.25e-6, abs=2.0e-8
    )
    assert analysis["clock"]["callback_timestamp_q_used"] is False
    assert analysis["clock"]["separate_marker_used"] is False
    timing = analysis["timing"]
    assert timing["paths"]["primary"]["bulk_peak_samples"] == pytest.approx(
        truth["primary_ERR"], abs=0.08
    )
    assert timing["paths"]["secondary"]["bulk_peak_samples"] == pytest.approx(
        truth["secondary_ERR"], abs=0.08
    )
    assert int(operator["primary_zeros_before_fir"]) == 1386
    assert int(operator["secondary_zeros_before_fir"]) == 1245
    assert timing["lead"]["samples"] == 115
    assert int(operator["separate_fractional_phase_applications"]) == 0
    assert analysis["compact_refit"]["fractional_shape_inside_FIR_once"] is True
    assert all(
        receipt["passed"]
        for receipt in analysis["compact_refit"]["roundtrip"].values()
    )
    assert analysis["live_delay_authority_available"] is None
    assert analysis["raw_publisher_bound"] is False
    assert analysis["canonical_training_eligible"] is False


def test_candidates_are_preterminal_and_fixed_average_is_the_scored_operator(
    analyzed_fractional_fixture: tuple[dict, dict, dict[str, float], float]
) -> None:
    analysis, operator = analyzed_fractional_fixture[:2]
    for candidate in ("fit_a", "fit_b"):
        receipt = analysis["candidate_fit_cross_preterminal_scores"][candidate]
        assert len(receipt["rows"]) == 64
        assert receipt["all_rows_passed"] is True
        assert receipt["holdout_used_for_threshold_support_or_candidate_tuning"] is False
        assert receipt["scored_zeros_before_fir_samples"] == [1386, 1245]
        relations = {row["relation"] for row in receipt["rows"]}
        assert relations == {"fit", "cross"}
        assert all(row["evaluation_role"] != "holdout" for row in receipt["rows"])

    final = analysis["final_fixed_average"]
    score = final["score"]
    assert score["evaluation_order"] == ["fit_a", "fit_b", "terminal_holdout"]
    assert len(score["rows"]) == 96 and score["all_rows_passed"] is True
    assert [row["evaluation_role"] for row in score["rows"][::32]] == [
        "fit_a",
        "fit_b",
        "holdout",
    ]
    assert {row["relation"] for row in score["rows"]} == {
        "preterminal_fit_role",
        "terminal_holdout",
    }
    for row in score["rows"]:
        assert row["target_bins"] >= 8 and row["noise_bins"] >= 8
        assert row["target_bin_density_above_noise_20db"] >= 0.95
        assert row["response_to_noise_db"] >= 20.0
        assert row["independent_coherence_claimed"] is False
        assert "complex_vector_agreement_not_coherence" in row

    rebuilt = np.stack(
        (
            operator["primary_compact_fir_by_mic"],
            operator["secondary_compact_fir_by_mic"],
        ),
        axis=1,
    )
    formula = final["formula"]
    for candidate in ("fit_a", "fit_b"):
        assert analysis["candidate_fit_cross_preterminal_scores"][candidate][
            "scored_compact_fir_array_sha256"
        ] == formula["compact_candidate_array_sha256"][candidate]
    assert _array_sha256(rebuilt) == formula["fixed_average_compact_array_sha256"]
    assert _array_sha256(rebuilt) == score["scored_compact_fir_array_sha256"]
    assert score["scored_zeros_before_fir_samples"] == [1386, 1245]
    assert score["fixed_average_formula_payload_sha256"] == formula[
        "canonical_payload_sha256"
    ]
    assert final["returned_operator_is_exact_scored_fixed_average"] is True
    assert all(
        receipt["passed"]
        for receipt in final["roundtrip_on_fit_roles"].values()
    )
    receipt = operator["receipt"]
    for key, value in operator.items():
        if key != "receipt":
            assert receipt["operator_array_sha256"][key] == _array_sha256(value)
    payload = {key: value for key, value in receipt.items() if key != "canonical_payload_sha256"}
    assert receipt["canonical_payload_sha256"] == _payload_sha256(payload)
    assert receipt["captured_adc_pcm_sha256"] == analysis["captured_raw_binding"][
        "captured_adc_pcm_sha256"
    ]
    assert receipt["raw_publisher_bound"] is False
    assert receipt["live_delay_authority_available"] is None
    assert receipt["canonical_training_eligible"] is False


def test_full_and_compact_lsmr_receipts_are_finite_converged_and_gated(
    analyzed_fractional_fixture: tuple[dict, dict, dict[str, float], float]
) -> None:
    analysis, operator = analyzed_fractional_fixture[:2]
    groups = (
        analysis["full_unshifted_causal_identification"]["candidates"],
        analysis["compact_refit"]["candidate_receipts"],
    )
    for candidates in groups:
        for candidate in candidates.values():
            for fit in candidate["fit_receipts"]:
                diagnostics = fit["lsmr_diagnostics"]
                assert set(diagnostics) == {
                    "istop",
                    "itn",
                    "normr",
                    "normar",
                    "norma",
                    "conda",
                    "normx",
                }
                assert np.all(
                    np.isfinite(np.asarray(list(diagnostics.values()), dtype=np.float64))
                )
                assert fit["solution_x_all_finite"] is True
                assert fit["coefficient_fir_all_finite"] is True
                assert fit["coefficient_fir_shape"][0] == 2
                assert diagnostics["istop"] in fit["accepted_istop_codes"] == [1, 2]
                assert diagnostics["itn"] < fit["max_iterations"]
                assert fit["max_iterations_exhausted"] is False
                assert fit["relative_residual"] <= fit["maximum_relative_residual"]
                assert (
                    fit["normal_equation_relative_residual"]
                    <= fit["maximum_normal_equation_relative_residual"]
                )
                assert fit["independent_residual_gates_passed"] is True


def test_shifted_exact_condition_and_holdout_execution_order_are_bound(
    analyzed_fractional_fixture: tuple[dict, dict, dict[str, float], float]
) -> None:
    analysis, operator = analyzed_fractional_fixture[:2]
    condition = analysis["compact_refit"][
        "shifted_support_1024_exact_condition_receipt"
    ]
    assert condition["schema"] == "fullband_causal_shifted_exact_gram_condition_v5"
    assert condition["zeros_before_fir_samples"] == [1386, 1245]
    assert condition["operator_definition"]["zeros_before_fir_samples"] == {
        "primary": 1386,
        "secondary": 1245,
    }
    assert condition["operator_definition_sha256"] == _payload_sha256(
        condition["operator_definition"]
    )
    assert condition["operator_quadratic_form_crosscheck_passed"] is True
    assert condition["passed"] is True
    assert analysis["compact_refit"][
        "unshifted_condition_receipt_reused_for_shifted_operator"
    ] is False
    assert analysis["clock"]["operator_holdout_used_for_clock_validation"] is False
    assert analysis["clock"]["captured_adc_full_sha256_computed"] is False
    assert analysis["holdout_policy"][
        "captured_full_sha_computed_only_after_terminal_score"
    ] is True
    for interpolation in ("cubic", "linear"):
        clock = analysis["clock"][interpolation]
        assert clock["validation_policy"] == "pilot_tail_only_pre_operator_holdout"
        assert clock["operator_holdout_used_for_clock_validation"] is False
        assert set(clock["clock_validation_rows_by_path"]) == {"primary", "secondary"}
        assert all(
            rows == ["tail"]
            for rows in clock["clock_validation_rows_by_path"].values()
        )
    order = analysis["holdout_policy"]["execution_order"]
    assert order.index("operator_holdout_first_open") > order.index(
        "fixed_average_fit_a_fit_b_scoring"
    )
    assert order[-1] == "fixed_average_terminal_holdout_scoring"
    receipt = analysis["final_fixed_average"]["operator_receipt"]
    assert receipt["duplex_telemetry_receipt_sha256"] == analysis[
        "duplex_telemetry_auxiliary"
    ]["sha256"]
    assert receipt["clock_receipt_sha256"] == analysis["clock"]["sha256"]
    assert receipt["timing_receipt_sha256"] == analysis["timing"]["sha256"]
    assert receipt["final_representation_roundtrip_bundle_sha256"] == analysis[
        "final_fixed_average"
    ]["roundtrip_bundle"]["canonical_payload_sha256"]
    assert receipt["representation_threshold_contract_sha256"] == analysis[
        "final_fixed_average"
    ]["representation_threshold_contract"]["canonical_payload_sha256"]
    assert receipt["score_threshold_contract_sha256"] == analysis[
        "final_fixed_average"
    ]["score_threshold_contract"]["canonical_payload_sha256"]
    spliced = copy.deepcopy(analysis)
    spliced["clock"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="stale payload SHA"):
        live_core._validate_operator_component_bindings(spliced, operator)

    for path, key, value in (
        (("clock",), "selected_rate_ratio", 1.0001),
        (("final_fixed_average", "formula"), "candidate_weights", [0.4, 0.6]),
        (("final_fixed_average", "score_threshold_contract"), "minimum_response_to_noise_db", 19.0),
        (("compact_refit", "shifted_support_1024_exact_condition_receipt"), "zeros_before_fir_samples", [1, 2]),
    ):
        stale = copy.deepcopy(analysis)
        node = stale
        for part in path:
            node = node[part]
        node[key] = value
        with pytest.raises(ValueError, match="stale payload SHA"):
            live_core._validate_operator_component_bindings(stale, operator)
    spliced_operator = copy.deepcopy(operator)
    spliced_operator["primary_compact_fir_by_mic"][0, 0] += 1.0
    with pytest.raises(ValueError, match="array SHA splice"):
        live_core._validate_operator_component_bindings(analysis, spliced_operator)


def _solver_fixture(support: int = 16) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260829)
    x_rows = rng.standard_normal((2, live_core.PERIOD, 2))
    fir = np.zeros((2, 2, support), dtype=np.float64)
    fir[:, 0, 2] = [0.7, 0.5]
    fir[:, 1, 5] = [-0.4, 0.3]
    return x_rows, _predict(x_rows, fir, zeros=(0, 0))


def test_actual_lsmr_failures_close_stop_residual_normal_and_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normal_limit = live_core.MAX_SOLVER_NORMAL_EQUATION_RELATIVE_RESIDUAL
    iteration_limit = live_core.LSMR_MAX_ITERATIONS
    x_rows, represented = _solver_fixture()
    independent = np.random.default_rng(71).standard_normal(represented.shape)
    with pytest.raises(ValueError, match="전체 relative residual"):
        _fit_candidate(
            x_rows=x_rows,
            y_rows=independent,
            role="negative_residual",
            support=16,
            zeros=(0, 0),
        )

    with pytest.raises(ValueError, match="istop"):
        _fit_candidate(
            x_rows=np.zeros_like(x_rows),
            y_rows=np.ones_like(represented),
            role="negative_stop",
            support=16,
            zeros=(0, 0),
        )

    with pytest.raises(ValueError, match="finite"):
        bad_target = represented.copy()
        bad_target[0, 0, 0] = np.nan
        _fit_candidate(
            x_rows=x_rows,
            y_rows=bad_target,
            role="negative_nonfinite",
            support=16,
            zeros=(0, 0),
        )

    monkeypatch.setattr(live_core, "MAX_SOLVER_NORMAL_EQUATION_RELATIVE_RESIDUAL", 0.0)
    with pytest.raises(ValueError, match="normal-equation"):
        _fit_candidate(
            x_rows=x_rows,
            y_rows=represented,
            role="negative_normal",
            support=16,
            zeros=(0, 0),
        )
    monkeypatch.setattr(
        live_core,
        "MAX_SOLVER_NORMAL_EQUATION_RELATIVE_RESIDUAL",
        normal_limit,
    )
    monkeypatch.setattr(live_core, "LSMR_MAX_ITERATIONS", 1)
    with pytest.raises(ValueError, match="maxiter"):
        _fit_candidate(
            x_rows=x_rows,
            y_rows=represented,
            role="negative_maxiter",
            support=16,
            zeros=(0, 0),
        )
    monkeypatch.setattr(live_core, "LSMR_MAX_ITERATIONS", iteration_limit)


def _full_candidate(*, early: bool = False, tail: bool = False) -> dict[str, object]:
    fir = np.zeros((2, 2, FULL_CAUSAL_SUPPORT_SAMPLES), dtype=np.float64)
    for mic in range(2):
        fir[mic, 0, 1642] = 1.0
        fir[mic, 1, 1501] = 1.0
    if early:
        fir[0, 0, 1200] = 0.1
    if tail:
        fir[0, 0, 2500] = 0.1
    return {"role": "fit_a", "fir_by_mic_path": fir}


@pytest.mark.parametrize("kind", ["early", "tail"])
def test_early_tail_are_diagnostic_but_unrepresentable_roundtrip_fails(
    kind: str,
) -> None:
    full = _full_candidate(early=kind == "early", tail=kind == "tail")
    characterized = _characterize_full_candidate(full)
    path = characterized["paths"]["primary_ERR"]
    assert path[f"{kind}_energy_diagnostic_exceeded"] is True
    assert path["energy_ratios_are_noise_sensitive_diagnostics_not_admission"] is True
    assert path["representation_admission_authority"] is False

    plan, submitted = build_plan_v5()
    x_by_role = {}
    for role in live_core.FIT_ROLES:
        x_by_role[role] = np.stack([
            submitted[row["central_start_frame"]:row["central_stop_frame"]].astype(np.float64)
            for slot_path in ("primary", "secondary")
            for row in [next(row for row in plan["layout"] if row.get("path") == slot_path and row.get("role") == role)]
        ])
    observed = {
        role: _predict(x_by_role[role], full["fir_by_mic_path"], zeros=(0, 0))
        for role in live_core.FIT_ROLES
    }
    fitted_full = _fit_candidate(
        x_rows=x_by_role["fit_a"], y_rows=observed["fit_a"], role="fit_a",
        support=FULL_CAUSAL_SUPPORT_SAMPLES, zeros=(0, 0),
    )
    fitted_compact = _fit_candidate(
        x_rows=x_by_role["fit_a"], y_rows=observed["fit_a"], role="fit_a",
        support=live_core.COMPACT_SUPPORT_SAMPLES, zeros=(1386, 1245),
    )
    timing_paths = {}
    for path_index, slot_path in enumerate(("primary", "secondary")):
        _, peak = live_core._continuous_peak(fitted_compact["fir_by_mic_path"][0, path_index])
        timing_paths[slot_path] = {"fractional_residual_samples": peak - live_core.COMPACT_PRE_ROLL_SAMPLES}
    zero_noise = np.zeros((live_core.PERIOD // 2 + 1, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="representation gate"):
        _compact_roundtrip(
            full_candidate=fitted_full, compact_candidate=fitted_compact,
            x_rows=x_by_role["fit_b"], y_rows=observed["fit_b"],
            zeros=(1386, 1245), timing={"paths": timing_paths},
            bands=plan["control_band_contract"]["physical_identification_subbands_hz"],
            exact_zero_noise_bins=np.ones(live_core.PERIOD // 2 + 1, dtype=np.bool_),
            noise_spectra=(zero_noise, zero_noise), evaluation_role="fit_b",
        )


def test_wrong_fit_role_peak_is_not_repaired_by_holdout() -> None:
    paths_a = {}
    paths_b = {}
    for path, peak in (("primary", 1642.2), ("secondary", 1500.7)):
        for microphone in ("ERR", "REF"):
            paths_a[f"{path}_{microphone}"] = {"continuous_peak_samples": peak}
            paths_b[f"{path}_{microphone}"] = {"continuous_peak_samples": peak}
    paths_b["primary_ERR"] = {"continuous_peak_samples": 1643.0}
    with pytest.raises(ValueError, match="stationarity"):
        derive_stationary_err_timing(
            {
                "fit_a": {"paths": paths_a},
                "fit_b": {"paths": paths_b},
            }
        )


def test_analysis_api_rejects_external_window_mapping() -> None:
    plan, submitted = build_plan_v5()
    with pytest.raises(TypeError):
        analyze_committed_v5_live_delay(
            plan=plan,
            submitted_pcm=submitted,
            captured_adc_pcm=np.zeros_like(submitted, dtype="<i4"),
            duplex_telemetry=_telemetry(submitted),
            marker_windows=[],  # type: ignore[call-arg]
        )


def test_production_dimension_exact_compact_20db_noise_representation_passes() -> None:
    plan, submitted = build_plan_v5()
    x_by_role = {}
    for role in live_core.FIT_ROLES:
        x_rows = []
        for path in ("primary", "secondary"):
            row = next(row for row in plan["layout"] if row.get("path") == path and row.get("role") == role)
            x_rows.append(submitted[row["central_start_frame"]:row["central_stop_frame"]].astype(np.float64))
        x_by_role[role] = np.stack(x_rows)
    zeros = (1386, 1245)
    truth = np.zeros((2, 2, live_core.COMPACT_SUPPORT_SAMPLES), dtype=np.float64)
    truth[:, :, live_core.COMPACT_PRE_ROLL_SAMPLES] = ((0.8, -0.6), (0.5, 0.7))
    rng = np.random.default_rng(20260829)
    observed_by_role = {}
    noise_by_role = {}
    for role in live_core.FIT_ROLES:
        clean = _predict(x_by_role[role], truth, zeros=zeros)
        noise = rng.normal(size=clean.shape)
        noise *= np.sqrt(np.mean(clean**2)) / (10.0 * np.sqrt(np.mean(noise**2)))
        observed_by_role[role] = clean + noise
        noise_by_role[role] = noise
    full = _fit_candidate(
        x_rows=x_by_role["fit_a"], y_rows=observed_by_role["fit_a"],
        role="fit_a", support=FULL_CAUSAL_SUPPORT_SAMPLES, zeros=(0, 0),
    )
    compact = _fit_candidate(
        x_rows=x_by_role["fit_a"], y_rows=observed_by_role["fit_a"],
        role="fit_a", support=live_core.COMPACT_SUPPORT_SAMPLES, zeros=zeros,
    )
    noise_spectra = tuple(
        np.fft.rfft(noise_by_role[role][0], axis=0) for role in live_core.FIT_ROLES
    )
    timing_paths = {}
    for path_index, path in enumerate(("primary", "secondary")):
        _, peak = live_core._continuous_peak(compact["fir_by_mic_path"][0, path_index])
        timing_paths[path] = {"fractional_residual_samples": peak - live_core.COMPACT_PRE_ROLL_SAMPLES}
    receipt = _compact_roundtrip(
        full_candidate=full, compact_candidate=compact,
        x_rows=x_by_role["fit_b"], y_rows=observed_by_role["fit_b"], zeros=zeros,
        timing={"paths": timing_paths},
        bands=plan["control_band_contract"]["physical_identification_subbands_hz"],
        exact_zero_noise_bins=np.ones(live_core.PERIOD // 2 + 1, dtype=np.bool_),
        noise_spectra=noise_spectra, evaluation_role="fit_b",
    )
    assert receipt["passed"] is True
    assert len(receipt["rows"]) == 32


def test_holdout_only_mutation_cannot_change_preterminal_clock_or_fit_inputs() -> None:
    plan, submitted = build_plan_v5()
    captured, _ = _fractional_affine_raw(submitted, q=1.0 + 173.0e-6)
    initial_clock = live_core.estimate_clock_cubic_linear_crosscheck(
        plan=plan, submitted_pcm=submitted, captured_adc_pcm=captured
    )
    mutated = captured.copy()
    q = initial_clock["selected_rate_ratio"]
    for row in plan["layout"]:
        if row.get("role") == "holdout":
            start = max(0, int(np.floor(row["central_start_frame"] / q)) - 3)
            stop = min(len(mutated), int(np.ceil((row["central_stop_frame"] - 1) / q)) + 4)
            mutated[start:stop] ^= np.int32(0x15555)
    clocks = [
        initial_clock,
        live_core.estimate_clock_cubic_linear_crosscheck(
            plan=plan, submitted_pcm=submitted, captured_adc_pcm=mutated
        ),
    ]
    assert clocks[0]["selected_rate_ratio"] == clocks[1]["selected_rate_ratio"]
    _, windows = validate_committed_plan_and_derive_windows(plan, submitted)
    fit_hashes = []
    preterminal_selection_hashes = []
    for value, clock in zip((captured, mutated), clocks, strict=True):
        access_log: list[dict[str, object]] = []
        rows = {
            role: live_core._role_rows(
                submitted=submitted, captured_adc_pcm=value, windows=windows,
                role=role, q=clock["selected_rate_ratio"], access_log=access_log,
            )
            for role in live_core.FIT_ROLES
        }
        fit_hashes.append({
            role: (_array_sha256(pair[0]), _array_sha256(pair[1]))
            for role, pair in rows.items()
        })
        preterminal_selection_hashes.append(_payload_sha256({
            "q": clock["selected_rate_ratio"],
            "fit_input_hashes": fit_hashes[-1],
            "candidate_formula": "fit_a_fit_b_then_fixed_average_0.5_0.5",
            "timing_source": "fit_a_fit_b_only",
        }))
        assert all("holdout" not in row["access_label"] for row in access_log)
    assert fit_hashes[0] == fit_hashes[1]
    assert preterminal_selection_hashes[0] == preterminal_selection_hashes[1]
