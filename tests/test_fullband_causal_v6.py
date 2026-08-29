from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import deep_anc.dsp.fullband_causal_v6 as v6
from deep_anc.dsp.fullband_live_raw_v5 import LIVE_RAW_SCHEMA_V6
from deep_anc.dsp.fullband_causal_v5 import PERIOD, SEEDS, build_plan_v5, near_white_period
from deep_anc.dsp.fullband_causal_v6 import (
    CLOCK_BINS,
    CLOCK_BLOCK_FRAMES,
    CLOCK_EPOCHS,
    CLOCK_PREFIX,
    CLOCK_SUFFIX,
    RAW_DEFAULT,
    TOTAL_FRAMES,
    build_plan_v6,
    estimate_common_clock_v6,
    exact_condition_audit_v6,
    global_grid_basin_search_v6,
    preoptimizer_spectral_line_admission_v6,
    synthesize_affine_capture_v6,
)


def test_v5_builder_schema_and_default_raw_are_unchanged() -> None:
    plan, pcm = build_plan_v5()
    assert plan["schema"] == "fullband_causal_time_separated_near_white_v5"
    assert plan["publisher_contract"]["raw_session_relative_path"] == (
        "results/fullband_causal_v5/raw_capture.npz"
    )
    assert plan["canonical_payload_sha256"] == (
        "32a79b3700b457dc40373dc4dd0969301287baea7100b1ec5edd86ea907ee127"
    )
    assert pcm.shape == (557_056, 2)


def test_v6_exact_signal_layout_power_full_period_and_holdout_policy() -> None:
    plan, pcm = build_plan_v6()
    assert plan["schema"] == v6.SCHEMA
    assert plan["raw_session_relative_path"] == RAW_DEFAULT
    assert plan["canonical_payload_sha256"] == (
        "8b37213a13131a071e10527c948580c906dfd914a1134e98a640ead259ba42f7"
    )
    assert plan["actual_submitted_pcm_sha256"] == (
        "4e8a66b983af872192624bd6759282058cfe4a845460111a24bcd684b22551a3"
    )
    assert pcm.dtype == np.int16
    assert pcm.shape == (TOTAL_FRAMES, 2) == (1_179_648, 2)
    assert plan["duration_seconds"] == pytest.approx(24.576)
    assert plan["publisher_contract"]["raw_session_relative_path"] == RAW_DEFAULT
    assert plan["publisher_contract"]["raw_npz_schema"] == LIVE_RAW_SCHEMA_V6
    assert plan["publisher_contract"]["must_not_exist_before_capture"] is True
    assert plan["actual_submitted_peak_pcm"] == 98

    excitation = plan["clock_excitation"]
    assert excitation["fixed_line_bins"] == list(CLOCK_BINS)
    assert excitation["line_count"] == 8
    assert excitation["bin_gcd_with_period"] == 1
    assert excitation["bin_difference_gcd"] == 1
    assert excitation["actual_fundamental_period_samples"] == PERIOD
    assert excitation["actual_line_to_guard_minimum_db"] > 34.0
    assert -0.25 <= excitation["path_only_vs_meter_db"] <= 0.0
    assert excitation["peak_pcm"] <= 98

    clock = [row for row in plan["layout"] if row["kind"] == "clock_block"]
    pe = [row for row in plan["layout"] if row["kind"] == "near_white_pe_slot"]
    assert len(clock) == 8 and len(pe) == 6
    assert all(row["stop_frame"] - row["start_frame"] == CLOCK_BLOCK_FRAMES for row in clock)
    assert CLOCK_BLOCK_FRAMES == 3 * PERIOD
    assert all(row["prefix_samples"] == CLOCK_PREFIX == PERIOD // 2 for row in clock)
    assert all(row["suffix_samples"] == CLOCK_SUFFIX == PERIOD // 2 for row in clock)

    for row in clock:
        active = int(row["active_channel"])
        opposite = 1 - active
        block = pcm[row["start_frame"] : row["stop_frame"]]
        first, second = row["central_repeat_starts"]
        first_period = pcm[first : first + PERIOD, active]
        second_period = pcm[second : second + PERIOD, active]
        assert np.array_equal(first_period, second_period)
        assert np.array_equal(block[:CLOCK_PREFIX, active], first_period[-CLOCK_PREFIX:])
        assert np.array_equal(block[-CLOCK_SUFFIX:, active], first_period[:CLOCK_SUFFIX])
        assert np.all(block[:, opposite] == 0)
        assert row["opposite_channel_actual_max_abs_dft"] == 0.0
        spectrum = np.fft.rfft(first_period.astype(np.float64))
        assert np.all(np.abs(spectrum[np.asarray(CLOCK_BINS)]) > 1.0)

    for row in pe:
        slot = pcm[row["start_frame"] : row["stop_frame"]]
        assert row["peak_pcm"] == 49
        assert row["continuous_clock_pilot_present"] is False
        assert np.max(np.abs(slot[:, int(row["active_channel"])].astype(np.int32))) == 49
        assert np.all(slot[:, 1 - int(row["active_channel"])] == 0)
        assert row["actual_submitted_not_above_meter"] is True
        assert row["actual_submitted_total_power"] <= row["official_meter_total_power"]
        assert row["actual_submitted_vs_meter_db"] == pytest.approx(
            -0.4354513906504307
        )
        expected_pe = near_white_period(SEEDS[(row["path"], row["role"])])
        assert np.array_equal(
            pcm[
                row["central_start_frame"] : row["central_stop_frame"],
                row["active_channel"],
            ],
            expected_pe,
        )

    power = plan["active_block_power_contract"]
    assert power["pe_slot_count"] == 6
    assert power["worst_pe_slot_vs_meter_db"] == pytest.approx(
        -0.4354513906504307
    )
    assert power["all_active_blocks_not_above_meter"] is True

    policy = plan["holdout_access_policy"]
    assert policy["clock_fit_epochs"] == list(CLOCK_EPOCHS[:3])
    assert policy["clock_terminal_validation_epoch"] == CLOCK_EPOCHS[3]
    assert policy["operator_holdout_used_for_clock_fit_snr_basin_or_selection"] is False
    assert policy["clock_fit_epoch_rows"] == [
        "fit_pre_0",
        "fit_pre_0",
        "fit_pre_1",
        "fit_pre_1",
        "fit_pre_2",
        "fit_pre_2",
    ]
    assert policy["terminal_epoch_rows"] == [
        "terminal_post_holdout",
        "terminal_post_holdout",
    ]


def test_v6_pure_pe_exact_1024_gram_receipt_is_literal_and_not_v5_value() -> None:
    plan, pcm = build_plan_v6()
    receipt = exact_condition_audit_v6(plan, pcm)
    assert receipt["schema"] == "fullband_causal_exact_gram_condition_v6"
    assert receipt["signal_plan_payload_sha256"] == plan["canonical_payload_sha256"]
    assert receipt["actual_submitted_pcm_sha256"] == plan["actual_submitted_pcm_sha256"]
    assert receipt["zeros_before_fir_samples"] == [0, 0]
    assert receipt["support_samples"] == 1_024
    assert receipt["role_condition_numbers"]["fit_a"] == pytest.approx(
        2.7725128483014365
    )
    assert receipt["role_condition_numbers"]["fit_b"] == pytest.approx(
        3.1970213983201465
    )
    assert receipt["joint_fit_condition_number"] == pytest.approx(
        1.9649539063087111
    )
    assert receipt["joint_fit_condition_number"] != pytest.approx(
        9.058033530917806
    )
    assert receipt["minimum_eigenvalue"] == pytest.approx(112010848.95588191)
    assert receipt["maximum_eigenvalue"] == pytest.approx(220096155.20481518)
    assert receipt["operator_quadratic_form_crosscheck_passed"] is True
    assert receipt["passed"] is True
    assert receipt["canonical_payload_sha256"] == (
        "211f581296d9d99927241a08c7a1096615246d68fe6702db8ff241cf1f582034"
    )


def test_v6_global_grid_finds_true_basin_not_single_start_local_trap() -> None:
    def objective(ratio: float) -> float:
        ppm = (ratio - 1.0) * 1.0e6
        return min(1.0 + (ppm + 895.0) ** 2, 10.0 + (ppm + 185.0) ** 2)

    ratio, receipt = global_grid_basin_search_v6(objective)
    assert (ratio - 1.0) * 1.0e6 == pytest.approx(-895.0, abs=1.0e-4)
    assert len(receipt["basins"]) == 2
    assert receipt["all_interior_basins_refined"] is True
    assert receipt["runner_up_to_best_objective_ratio"] == pytest.approx(10.0)
    assert receipt["unique_basin_passed"] is True


def test_v6_global_grid_rejects_boundary_and_ambiguous_basins() -> None:
    with pytest.raises(ValueError, match="search boundary"):
        global_grid_basin_search_v6(
            lambda ratio: ((ratio - 1.0) * 1.0e6 + 1_000.0) ** 2
        )

    def ambiguous(ratio: float) -> float:
        ppm = (ratio - 1.0) * 1.0e6
        return min(1.0 + (ppm + 500.0) ** 2, 2.0 + (ppm - 500.0) ** 2)

    with pytest.raises(ValueError, match="multimodal ambiguous"):
        global_grid_basin_search_v6(ambiguous)


def test_v6_estimator_preserves_ambiguous_global_basin_receipt_and_stops_before_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, submitted = build_plan_v6()
    captured = synthesize_affine_capture_v6(
        submitted,
        primary_fir_by_mic=np.asarray([[1.0], [0.8]], dtype=np.float64),
        secondary_fir_by_mic=np.asarray([[0.7], [-0.9]], dtype=np.float64),
        rate_ratio=1.0,
    )
    original_search = global_grid_basin_search_v6
    calls: list[bool] = []

    def ambiguous_first_search(
        _objective, *, grid_step_ppm=v6.GLOBAL_GRID_STEP_PPM, require_unique=True
    ):  # noqa: ANN001, ANN202
        calls.append(require_unique)
        if len(calls) != 1:
            raise AssertionError("ambiguous global basin 뒤 view optimizer가 실행됐습니다")

        def ambiguous(ratio: float) -> float:
            ppm = (ratio - 1.0) * 1.0e6
            return min(1.0 + (ppm + 500.0) ** 2, 2.0 + (ppm - 500.0) ** 2)

        return original_search(
            ambiguous,
            grid_step_ppm=grid_step_ppm,
            require_unique=require_unique,
        )

    monkeypatch.setattr(v6, "global_grid_basin_search_v6", ambiguous_first_search)
    with pytest.raises(v6.V6ClockAdmissionError, match="multimodal ambiguous") as caught:
        estimate_common_clock_v6(
            plan=plan,
            submitted_pcm=submitted,
            captured_pcm=captured,
        )

    failure = caught.value
    assert failure.stage == "global_grid_basin_search"
    assert failure.optimizer_started is True
    assert calls == [False]
    assert set(failure.available_receipt) == {
        "preterminal_preoptimizer_snr_admission",
        "terminal_preoptimizer_snr_admission",
        "global_search",
    }
    assert failure.available_receipt[
        "preterminal_preoptimizer_snr_admission"
    ]["passed"] is True
    assert failure.available_receipt[
        "terminal_preoptimizer_snr_admission"
    ]["passed"] is True
    global_search = failure.available_receipt["global_search"]
    assert global_search["unique_basin_passed"] is False
    assert len(global_search["basins"]) == 2
    assert global_search["runner_up_to_best_objective_ratio"] == pytest.approx(2.0)
    assert global_search["minimum_unique_basin_objective_ratio"] == pytest.approx(4.0)


def test_v6_estimator_preserves_failed_view_basin_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, submitted = build_plan_v6()
    captured = synthesize_affine_capture_v6(
        submitted,
        primary_fir_by_mic=np.asarray([[1.0], [0.8]], dtype=np.float64),
        secondary_fir_by_mic=np.asarray([[0.7], [-0.9]], dtype=np.float64),
        rate_ratio=1.0,
    )
    original_search = global_grid_basin_search_v6
    calls: list[bool] = []

    def fail_first_view(
        objective, *, grid_step_ppm=v6.GLOBAL_GRID_STEP_PPM, require_unique=True
    ):  # noqa: ANN001, ANN202
        calls.append(require_unique)
        if len(calls) == 1:
            return original_search(
                objective,
                grid_step_ppm=grid_step_ppm,
                require_unique=require_unique,
            )

        def ambiguous(ratio: float) -> float:
            ppm = (ratio - 1.0) * 1.0e6
            return min(1.0 + (ppm + 500.0) ** 2, 2.0 + (ppm - 500.0) ** 2)

        return original_search(
            ambiguous,
            grid_step_ppm=grid_step_ppm,
            require_unique=require_unique,
        )

    monkeypatch.setattr(v6, "global_grid_basin_search_v6", fail_first_view)
    with pytest.raises(
        v6.V6ClockAdmissionError, match="path/mic clock objective가 multimodal"
    ) as caught:
        estimate_common_clock_v6(
            plan=plan,
            submitted_pcm=submitted,
            captured_pcm=captured,
        )

    failure = caught.value
    assert failure.stage == "view_global_grid_basin_search/primary_ERR"
    assert failure.optimizer_started is True
    assert calls == [False, False]
    assert failure.available_receipt["failed_view"] == "primary_ERR"
    failed = failure.available_receipt["completed_view_searches"]["primary_ERR"]
    assert failed["unique_basin_passed"] is False
    assert len(failed["basins"]) == 2


def test_v6_preoptimizer_snr_failure_stops_before_global_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, pcm = build_plan_v6()
    captured = np.zeros_like(pcm, dtype=np.float64)

    def forbidden(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("SNR admission 실패 뒤 optimizer가 실행되면 안 됩니다")

    monkeypatch.setattr(v6, "global_grid_basin_search_v6", forbidden)
    with pytest.raises(v6.V6ClockAdmissionError, match="pre-optimizer clock SNR admission") as caught:
        estimate_common_clock_v6(plan=plan, submitted_pcm=pcm, captured_pcm=captured)
    assert caught.value.stage == "preterminal_preoptimizer_snr_admission"
    assert caught.value.optimizer_started is False
    assert caught.value.available_receipt["passed"] is False
    assert caught.value.available_receipt["optimizer_was_run"] is False
    assert caught.value.available_receipt["canonical_payload_sha256"]


@pytest.mark.parametrize("ppm", [-1_000.0, 1_000.0])
def test_v6_preoptimizer_hann_gate_survives_pure_signal_worst_clock_offset(
    ppm: float,
) -> None:
    plan, submitted = build_plan_v6()
    primary = np.asarray([[1.0], [0.8]], dtype=np.float64)
    secondary = np.asarray([[0.7], [-0.9]], dtype=np.float64)
    captured = synthesize_affine_capture_v6(
        submitted,
        primary_fir_by_mic=primary,
        secondary_fir_by_mic=secondary,
        rate_ratio=1.0 + ppm * 1.0e-6,
    )
    repeats = v6._nominal_repeats(captured, v6._clock_rows(plan, terminal=False))
    receipt = preoptimizer_spectral_line_admission_v6(repeats)
    assert receipt["window"] == "full_P_symmetric_Hann_numpy_hanning"
    assert receipt["target_bin_offsets"] == [-2, -1, 0, 1, 2]
    assert receipt["local_guard_offsets"] == [
        -10,
        -9,
        -8,
        -7,
        -6,
        6,
        7,
        8,
        9,
        10,
    ]
    assert receipt["minimum_observed_snr_db"] > 30.0
    assert receipt["minimum_observed_snr_db"] >= receipt["minimum_snr_db"] == 20.0


def test_current_low_snr_v5_raw_is_explicit_preoptimizer_snr_failure() -> None:
    raw_path = Path("results/fullband_causal_v5/raw_capture.npz")
    envelope_path = Path("assets/contracts/fullband_causal_v5_signal_plan.json")
    if not raw_path.is_file() or not envelope_path.is_file():
        pytest.skip("현장 v5 raw는 repository artifact가 아니므로 없는 환경에서는 skip")
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    plan = envelope["signal_plan"]
    lead = next(row for row in plan["layout"] if row["kind"] == "pilot_only_lead")
    with np.load(raw_path, allow_pickle=False) as payload:
        captured = payload["captured_pcm"]
    repeats = np.stack(
        [
            captured[lead["start_frame"] : lead["start_frame"] + PERIOD],
            captured[
                lead["start_frame"] + PERIOD : lead["start_frame"] + 2 * PERIOD
            ],
        ]
    )[None]
    primary_bins = plan["clock_contract"]["primary_pilot_bins"][:8]
    with pytest.raises(ValueError, match="pre-optimizer clock SNR admission"):
        preoptimizer_spectral_line_admission_v6(
            repeats, fixed_bins=primary_bins, minimum_snr_db=20.0
        )


@pytest.fixture(scope="module")
def known_q_fixture() -> tuple[dict, np.ndarray, np.ndarray, dict, float]:
    plan, submitted = build_plan_v6()
    taps = 1_200
    primary = np.zeros((2, taps), dtype=np.float64)
    secondary = np.zeros((2, taps), dtype=np.float64)
    primary[:, 500] = [7_000.0, 5_000.0]
    secondary[:, 420] = [-6_000.0, 4_500.0]
    q = 1.0 + 173.25e-6
    captured = synthesize_affine_capture_v6(
        submitted,
        primary_fir_by_mic=primary,
        secondary_fir_by_mic=secondary,
        rate_ratio=q,
        noise_rms=0.1,
        seed=20260829,
    )
    receipt = estimate_common_clock_v6(
        plan=plan, submitted_pcm=submitted, captured_pcm=captured
    )
    return plan, submitted, captured, receipt, q


def test_v6_synthetic_known_q_passes_all_fixed_gates(
    known_q_fixture: tuple[dict, np.ndarray, np.ndarray, dict, float]
) -> None:
    plan, submitted, _, receipt, q = known_q_fixture
    assert receipt["passed"] is True
    assert receipt["selected_rate_ratio"] == pytest.approx(q, abs=5.0e-10)
    assert receipt["maximum_view_endpoint_disagreement_samples"] <= 0.05
    assert receipt["cubic_linear_endpoint_disagreement_samples"] <= (
        v6.MAX_CUBIC_LINEAR_ENDPOINT_DISAGREEMENT_SAMPLES
    )
    assert receipt["maximum_terminal_phase_error_samples"] <= (
        v6.MAX_TERMINAL_PHASE_ERROR_SAMPLES
    )
    assert receipt["preoptimizer_snr_admission"]["optimizer_was_run"] is False
    assert receipt["operator_holdout_accessed"] is False
    assert receipt["clock_fit_epochs"] == list(CLOCK_EPOCHS[:3])
    holdout = [
        tuple(value)
        for value in plan["holdout_access_policy"]["operator_holdout_frame_ranges"]
    ]
    accessed = [tuple(value) for value in receipt["accessed_clock_frame_ranges"]]
    assert not any(max(a, c) < min(b, d) for a, b in accessed for c, d in holdout)
    assert receipt["actual_submitted_pcm_sha256"] == plan["actual_submitted_pcm_sha256"]
    assert submitted.shape[0] == TOTAL_FRAMES
    for kind in ("cubic", "linear"):
        interpolation = receipt["interpolation"][kind]
        assert interpolation["preterminal_repeat"]["passed"] is True
        assert interpolation["terminal_repeat"]["passed"] is True


def test_v6_operator_holdout_mutation_cannot_change_clock_receipt(
    known_q_fixture: tuple[dict, np.ndarray, np.ndarray, dict, float]
) -> None:
    plan, submitted, captured, original, _ = known_q_fixture
    changed = captured.copy()
    for start, stop in plan["holdout_access_policy"]["operator_holdout_frame_ranges"]:
        changed[start:stop] = np.random.default_rng(start).normal(
            0.0, 1.0e9, (stop - start, 2)
        )
    repeated = estimate_common_clock_v6(
        plan=plan, submitted_pcm=submitted, captured_pcm=changed
    )
    assert repeated == original


def test_v6_terminal_mutation_is_rejected_without_refitting_q(
    known_q_fixture: tuple[dict, np.ndarray, np.ndarray, dict, float]
) -> None:
    plan, submitted, captured, _, _ = known_q_fixture
    changed = captured.copy()
    terminal = [
        row
        for row in plan["layout"]
        if row.get("kind") == "clock_block"
        and row.get("stage") == "terminal_validation"
    ]
    for row in terminal:
        start, stop = int(row["start_frame"]), int(row["stop_frame"])
        changed[start:stop] = np.roll(changed[start:stop], 4, axis=0)
    with pytest.raises(ValueError, match="terminal clock phase validation"):
        estimate_common_clock_v6(
            plan=plan, submitted_pcm=submitted, captured_pcm=changed
        )


def test_v6_rejects_plan_line_or_holdout_policy_mutation_before_analysis() -> None:
    plan, submitted = build_plan_v6()
    changed = copy.deepcopy(plan)
    changed["clock_excitation"]["fixed_line_bins"][0] += 1
    with pytest.raises(ValueError, match="exact v6 plan"):
        estimate_common_clock_v6(
            plan=changed,
            submitted_pcm=submitted,
            captured_pcm=np.zeros_like(submitted, dtype=np.float64),
        )

    changed = copy.deepcopy(plan)
    changed["holdout_access_policy"][
        "operator_holdout_used_for_clock_fit_snr_basin_or_selection"
    ] = True
    with pytest.raises(ValueError, match="exact v6 plan"):
        estimate_common_clock_v6(
            plan=changed,
            submitted_pcm=submitted,
            captured_pcm=np.zeros_like(submitted, dtype=np.float64),
        )
