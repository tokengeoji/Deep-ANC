from __future__ import annotations

import copy

import numpy as np
import pytest

from deep_anc.audio_duplex_v5 import DUPLEX_TELEMETRY_SCHEMA as V5_TELEMETRY_SCHEMA
from deep_anc.audio_duplex_v6 import DUPLEX_TELEMETRY_SCHEMA as V6_TELEMETRY_SCHEMA
from deep_anc.dsp.fullband_causal_v5 import BLOCK, FS
from deep_anc.dsp.fullband_causal_v6 import (
    V6ClockAdmissionError,
    build_plan_v6,
    estimate_common_clock_v6,
    synthesize_affine_capture_v6,
)
import deep_anc.dsp.fullband_live_delay_core_v6 as core
import deep_anc.dsp.fullband_live_post_v6 as post
from deep_anc.dsp.fullband_live_delay_core_v6 import (
    EXPECTED_PCM_SHA256,
    EXPECTED_PLAN_SHA256,
    analyze_committed_v6_live_delay,
    exact_shifted_condition_audit_v6,
    validate_committed_v6_plan_and_derive_windows,
)


def _telemetry(submitted_pcm: np.ndarray, *, schema: str = V6_TELEMETRY_SCHEMA) -> dict:
    submitted = np.asarray(submitted_pcm)
    frames = len(submitted)
    count = frames // BLOCK
    index = np.arange(count, dtype="<i8")
    current = index.astype("<f8") * BLOCK / FS
    return {
        "schema": schema,
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
        "pre_open_monotonic_started": 9.0,
        "pre_open_monotonic_completed": 10.0,
        "pre_open_monotonic_elapsed_seconds": 1.0,
        "capture_monotonic_started": 10.0,
        "capture_monotonic_completed": 10.0 + frames / FS,
        "capture_monotonic_elapsed_seconds": frames / FS,
        "watchdog_grace_seconds": 2.0,
        "input_dtype": "<i4",
        "output_dtype": "<i2",
        "callback_sequence": index.copy(),
        "callback_start_frames": index * BLOCK,
        "callback_frame_counts": np.full(count, BLOCK, dtype="<i8"),
        "input_buffer_adc_time": current - 0.002,
        "output_buffer_dac_time": current + 0.002,
        "callback_current_time": current,
        "callback_status_bitmask": np.zeros(count, dtype="<u4"),
        "xrun_count": 0,
        "status_present_count": 0,
        "captured_frames": frames,
        "submitted_frames": frames,
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
        "capture_valid_mask": np.ones(frames, dtype=np.bool_),
        "submitted_valid_mask": np.ones(frames, dtype=np.bool_),
    }


@pytest.fixture(scope="module")
def synthetic_capture() -> tuple[dict, np.ndarray, np.ndarray, dict]:
    plan, submitted = build_plan_v6()
    taps = 1_200
    primary = np.zeros((2, taps), dtype=np.float64)
    secondary = np.zeros((2, taps), dtype=np.float64)
    primary[:, 500] = [7_000.0, 5_000.0]
    secondary[:, 420] = [-6_000.0, 4_500.0]
    captured = np.rint(
        synthesize_affine_capture_v6(
            submitted,
            primary_fir_by_mic=primary,
            secondary_fir_by_mic=secondary,
            rate_ratio=1.0 + 10.0e-6,
            noise_rms=0.01,
            seed=20260829,
        )
    ).astype("<i4")
    return plan, submitted, captured, _telemetry(submitted)


@pytest.fixture(scope="module")
def analyzed(
    synthetic_capture: tuple[dict, np.ndarray, np.ndarray, dict]
) -> tuple[dict, dict]:
    plan, submitted, captured, telemetry = synthetic_capture
    return analyze_committed_v6_live_delay(
        plan=plan,
        submitted_pcm=submitted,
        captured_adc_pcm=captured,
        duplex_telemetry=telemetry,
    )


def test_exact_v6_plan_pcm_windows_and_shifted_gram_are_pinned() -> None:
    plan, submitted = build_plan_v6()
    receipt, windows = validate_committed_v6_plan_and_derive_windows(plan, submitted)
    assert receipt["signal_plan_payload_sha256"] == EXPECTED_PLAN_SHA256
    assert receipt["actual_submitted_pcm_sha256"] == EXPECTED_PCM_SHA256
    assert list(windows) == [
        ("fit_a", "primary"),
        ("fit_a", "secondary"),
        ("fit_b", "primary"),
        ("fit_b", "secondary"),
        ("holdout", "primary"),
        ("holdout", "secondary"),
    ]
    shifted = exact_shifted_condition_audit_v6(
        plan, submitted, zeros_by_path=(244, 164)
    )
    assert shifted["schema"] == core.SHIFTED_CONDITION_SCHEMA
    assert shifted["zeros_before_fir_samples"] == [244, 164]
    assert shifted["passed"] is True

    changed = copy.deepcopy(plan)
    changed["layout"][2]["central_start_frame"] += 1
    with pytest.raises(ValueError, match="payload|committed"):
        validate_committed_v6_plan_and_derive_windows(changed, submitted)


def test_low_snr_structured_clock_failure_prevents_any_ls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, submitted = build_plan_v6()
    captured = np.zeros(submitted.shape, dtype="<i4")

    def forbidden(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("clock PASS 전 LS가 호출됐습니다")

    monkeypatch.setattr(core, "_fit_candidate", forbidden)
    with pytest.raises(V6ClockAdmissionError) as caught:
        analyze_committed_v6_live_delay(
            plan=plan,
            submitted_pcm=submitted,
            captured_adc_pcm=captured,
            duplex_telemetry=_telemetry(submitted),
        )
    assert caught.value.stage == "preterminal_preoptimizer_snr_admission"
    assert caught.value.optimizer_started is False
    assert caught.value.available_receipt["passed"] is False


@pytest.mark.parametrize("schema", [V5_TELEMETRY_SCHEMA, "tampered_v6"])
def test_wrong_telemetry_or_v5_splice_rejected_before_clock_and_ls(
    synthetic_capture: tuple[dict, np.ndarray, np.ndarray, dict],
    monkeypatch: pytest.MonkeyPatch,
    schema: str,
) -> None:
    plan, submitted, captured, _ = synthetic_capture

    def forbidden(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("잘못된 telemetry 뒤 clock/LS가 호출됐습니다")

    monkeypatch.setattr(core, "estimate_common_clock_v6", forbidden)
    monkeypatch.setattr(core, "_fit_candidate", forbidden)
    with pytest.raises(ValueError, match="audio_duplex_v6"):
        analyze_committed_v6_live_delay(
            plan=plan,
            submitted_pcm=submitted,
            captured_adc_pcm=captured,
            duplex_telemetry=_telemetry(submitted, schema=schema),
        )


def test_known_delay_full_pipeline_passes_all_eight_physical_subbands(
    analyzed: tuple[dict, dict],
) -> None:
    analysis, operator = analyzed
    assert analysis["schema"] == core.ANALYSIS_SCHEMA
    assert analysis["clock"]["passed"] is True
    assert analysis["hardware_slip_authority_available"] is False
    assert operator["receipt"]["hardware_sample_slip_authority_available"] is False
    assert analysis["clock"]["selected_ppm"] == pytest.approx(10.0, abs=0.001)
    assert analysis["timing"]["paths"]["primary"]["zeros_before_compact_fir_samples"] == 244
    assert analysis["timing"]["paths"]["secondary"]["zeros_before_compact_fir_samples"] == 164
    score = analysis["final_fixed_average"]["score"]
    assert score["expected_rows"] == 96
    assert score["physical_subband_count"] == 8
    assert {row["band_index"] for row in score["rows"]} == set(range(8))
    assert all(row["passed"] for row in score["rows"])
    assert operator["primary_compact_fir_by_mic"].shape == (2, 1_024)
    assert operator["secondary_compact_fir_by_mic"].shape == (2, 1_024)


def test_noise_is_broadband_preterminal_half_difference_not_fixed_lines(
    analyzed: tuple[dict, dict],
) -> None:
    analysis, _ = analyzed
    noise = analysis["broadband_noise"]
    assert noise["source_block_count"] == 6
    assert noise["terminal_clock_used_for_noise_fit_or_tuning"] is False
    assert noise["fixed_clock_bins_only"] is False
    assert noise["all_rfft_bins_available"] is True
    assert len(noise["band_rows"]) == 8
    assert noise["band_rows"][-1]["nonfixed_clock_bin_count"] > 1_000
    assert all(
        np.all(np.isfinite(row["mean_half_difference_power_by_mic"]))
        for row in noise["band_rows"]
    )
    score_rows = analysis["final_fixed_average"]["score"]["rows"]
    assert all(row["fixed_clock_bins_only"] is False for row in score_rows)


def test_holdout_is_first_opened_only_after_final_formula_hash(
    analyzed: tuple[dict, dict],
) -> None:
    analysis, _ = analyzed
    policy = analysis["holdout_policy"]
    order = policy["execution_order"]
    assert order.index("final_formula_fixed_and_hashed") < order.index(
        "operator_holdout_first_open"
    )
    assert policy["terminal_clock_used_for_q_selection_fit_noise_or_tuning"] is False
    assert all(
        "holdout" not in row["access_label"]
        for row in policy["preterminal_bounded_capture_access"]
    )
    assert all(
        "operator:holdout:" in row["access_label"]
        for row in policy["terminal_bounded_capture_access"]
    )


def test_actual_core_output_satisfies_post_publisher_contract(
    analyzed: tuple[dict, dict],
    synthetic_capture: tuple[dict, np.ndarray, np.ndarray, dict],
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis, operator = analyzed
    validation = core.validate_analysis_operator_v6(analysis, operator)
    assert validation["passed"] is True
    assert validation["exact_six_arrays"] is True
    receipt_sha = "a" * 64
    execution = {
        "repository_commit": "d" * 40,
        "repository_branch": "work/test-v6",
        "repository_dirty": False,
        "adapter_path": "scripts/data/measure_paths_fullband_causal_v6.py",
        "adapter_file_sha256": "e" * 64,
    }
    raw_arrays = {
        "captured_pcm": analysis["captured_raw_binding"]["captured_adc_pcm_sha256"],
        "actual_submitted_pcm": analysis["captured_raw_binding"]["actual_submitted_pcm_sha256"],
    }
    _plan, submitted, captured, telemetry = synthetic_capture
    admitted_arrays = {
        "actual_submitted_pcm": submitted,
        "captured_pcm": captured,
        "capture_valid_mask": telemetry["capture_valid_mask"],
        "submitted_valid_mask": telemetry["submitted_valid_mask"],
        **{name: telemetry[name] for name in post.TELEMETRY_ARRAY_FIELDS},
    }
    telemetry_scalars = {
        name: value
        for name, value in telemetry.items()
        if name
        not in set(post.TELEMETRY_ARRAY_FIELDS)
        | {
            "actual_submitted_pcm",
            "capture_valid_mask",
            "submitted_valid_mask",
        }
    }

    monkeypatch.setattr(
        post,
        "load_external_post_capture_receipt_v6",
        lambda **_kwargs: {
            "receipt_file_sha256": receipt_sha,
            "raw": {
                "metadata": {
                    "array_sha256": raw_arrays,
                    "session": dict(execution),
                    "duplex_telemetry_scalars": telemetry_scalars,
                },
                "arrays": admitted_arrays,
            },
        },
    )
    monkeypatch.setattr(
        post,
        "repository_execution_identity",
        lambda _root, script: {
            "repository_commit": execution["repository_commit"],
            "repository_branch": execution["repository_branch"],
            "repository_dirty": False,
            "script_path": script,
            "script_file_sha256": execution["adapter_file_sha256"],
        },
    )

    def recompute(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["plan"] == _plan
        assert kwargs["submitted_pcm"] is submitted
        assert kwargs["captured_adc_pcm"] is captured
        assert kwargs["duplex_telemetry"]["callback_sequence"] is telemetry[
            "callback_sequence"
        ]
        return analysis, operator

    monkeypatch.setattr(post, "analyze_committed_v6_live_delay", recompute)
    published = {}

    def fake_publish(_root, directory, *, analysis_bytes, operator_bytes):
        published["directory"] = directory
        published["analysis_bytes"] = analysis_bytes
        published["operator_bytes"] = operator_bytes
        return {"published": True}

    monkeypatch.setattr(post, "_publish_analysis_directory_noreplace", fake_publish)
    result = post.publish_live_delay_analysis_v6(
        repository_root=tmp_path,
        output_directory_relative_path="results/fullband_causal_v6/synthetic",
        external_receipt_relative_path="results/fullband_causal_v6/external.json",
        external_receipt_file_sha256=receipt_sha,
        plan_envelope_path="assets/contracts/v6.json",
        live_authority_path="assets/contracts/v6_authority.json",
        meter_raw_path="results/fullband_causal_v6/meter.npz",
        level_evidence_path="assets/measured/level.json",
        hardware_path="results/fullband_causal_v6/hardware.json",
        analysis_execution_identity=execution,
        analysis=analysis,
        operator=operator,
    )
    assert result == {"published": True}
    assert published["directory"] == "results/fullband_causal_v6/synthetic"
    assert published["analysis_bytes"]
    assert published["operator_bytes"]


def test_analysis_operator_validator_rejects_v5_missing_array_and_hash_splices(
    analyzed: tuple[dict, dict],
) -> None:
    analysis, operator = analyzed

    wrong_schema = copy.deepcopy(analysis)
    wrong_schema["schema"] = "fullband_committed_v5_live_delay_core_v3"
    with pytest.raises(ValueError, match="exact v6 core schema"):
        core.validate_analysis_operator_v6(wrong_schema, operator)

    missing = dict(operator)
    del missing["support_samples"]
    with pytest.raises(ValueError, match="exact 6 arrays"):
        core.validate_analysis_operator_v6(analysis, missing)

    bad_analysis_hash = copy.deepcopy(analysis)
    bad_analysis_hash["analysis_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="analysis self SHA"):
        core.validate_analysis_operator_v6(bad_analysis_hash, operator)

    bad_operator = copy.deepcopy(operator)
    bad_operator["receipt"]["canonical_payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="receipt self SHA"):
        core.validate_analysis_operator_v6(analysis, bad_operator)

    extra_key = copy.deepcopy(analysis)
    extra_key["v5_or_fake_payload"] = {"passed": True}
    extra_key["analysis_sha256"] = core._payload_sha256(
        {key: value for key, value in extra_key.items() if key != "analysis_sha256"}
    )
    with pytest.raises(ValueError, match="top-level key"):
        core.validate_analysis_operator_v6(extra_key, operator)

    relabeled_operator = copy.deepcopy(operator)
    relabeled_operator["receipt"]["signal_plan_payload_sha256"] = "f" * 64
    relabeled_operator["receipt"]["canonical_payload_sha256"] = core._payload_sha256(
        {
            key: value
            for key, value in relabeled_operator["receipt"].items()
            if key != "canonical_payload_sha256"
        }
    )
    relabeled_analysis = copy.deepcopy(analysis)
    relabeled_analysis["final_fixed_average"]["operator_receipt"] = relabeled_operator[
        "receipt"
    ]
    relabeled_analysis["analysis_sha256"] = core._payload_sha256(
        {
            key: value
            for key, value in relabeled_analysis.items()
            if key != "analysis_sha256"
        }
    )
    with pytest.raises(ValueError, match="pinned|권한"):
        core.validate_analysis_operator_v6(relabeled_analysis, relabeled_operator)

    wrong_shape_operator = copy.deepcopy(operator)
    wrong_shape_operator["support_samples"] = np.asarray([1_024], dtype="<i8")
    wrong_shape_operator["receipt"]["operator_array_sha256"]["support_samples"] = (
        core._array_sha256(wrong_shape_operator["support_samples"])
    )
    wrong_shape_operator["receipt"]["canonical_payload_sha256"] = core._payload_sha256(
        {
            key: value
            for key, value in wrong_shape_operator["receipt"].items()
            if key != "canonical_payload_sha256"
        }
    )
    wrong_shape_analysis = copy.deepcopy(analysis)
    wrong_shape_analysis["final_fixed_average"]["operator_receipt"] = (
        wrong_shape_operator["receipt"]
    )
    wrong_shape_analysis["analysis_sha256"] = core._payload_sha256(
        {
            key: value
            for key, value in wrong_shape_analysis.items()
            if key != "analysis_sha256"
        }
    )
    with pytest.raises(ValueError, match="dtype/shape"):
        core.validate_analysis_operator_v6(wrong_shape_analysis, wrong_shape_operator)

    fir_splice = copy.deepcopy(operator)
    fir_splice["primary_compact_fir_by_mic"][0, 0] += 1.0
    fir_splice["receipt"]["operator_array_sha256"][
        "primary_compact_fir_by_mic"
    ] = core._array_sha256(fir_splice["primary_compact_fir_by_mic"])
    fir_splice["receipt"]["canonical_payload_sha256"] = core._payload_sha256(
        {
            key: value
            for key, value in fir_splice["receipt"].items()
            if key != "canonical_payload_sha256"
        }
    )
    fir_analysis = copy.deepcopy(analysis)
    fir_analysis["final_fixed_average"]["operator_receipt"] = fir_splice["receipt"]
    fir_analysis["analysis_sha256"] = core._payload_sha256(
        {
            key: value
            for key, value in fir_analysis.items()
            if key != "analysis_sha256"
        }
    )
    with pytest.raises(ValueError, match="fixed-average compact"):
        core.validate_analysis_operator_v6(fir_analysis, fir_splice)

    zero_splice = copy.deepcopy(operator)
    zero_splice["primary_zeros_before_fir"] = np.asarray(
        int(zero_splice["primary_zeros_before_fir"]) + 1, dtype="<i8"
    )
    zero_splice["receipt"]["operator_array_sha256"][
        "primary_zeros_before_fir"
    ] = core._array_sha256(zero_splice["primary_zeros_before_fir"])
    zero_splice["receipt"]["canonical_payload_sha256"] = core._payload_sha256(
        {
            key: value
            for key, value in zero_splice["receipt"].items()
            if key != "canonical_payload_sha256"
        }
    )
    zero_analysis = copy.deepcopy(analysis)
    zero_analysis["final_fixed_average"]["operator_receipt"] = zero_splice["receipt"]
    zero_analysis["analysis_sha256"] = core._payload_sha256(
        {
            key: value
            for key, value in zero_analysis.items()
            if key != "analysis_sha256"
        }
    )
    with pytest.raises(ValueError, match="timing receipt"):
        core.validate_analysis_operator_v6(zero_analysis, zero_splice)

    score_link_operator = copy.deepcopy(operator)
    score_link_analysis = copy.deepcopy(analysis)
    score_link_analysis["final_fixed_average"]["score"][
        "fixed_average_formula_payload_sha256"
    ] = "f" * 64
    score_link_analysis["final_fixed_average"]["score"][
        "canonical_payload_sha256"
    ] = core._payload_sha256(
        {
            key: value
            for key, value in score_link_analysis["final_fixed_average"]["score"].items()
            if key != "canonical_payload_sha256"
        }
    )
    score_link_operator["receipt"]["final_score_payload_sha256"] = (
        score_link_analysis["final_fixed_average"]["score"][
            "canonical_payload_sha256"
        ]
    )
    score_link_operator["receipt"]["canonical_payload_sha256"] = core._payload_sha256(
        {
            key: value
            for key, value in score_link_operator["receipt"].items()
            if key != "canonical_payload_sha256"
        }
    )
    score_link_analysis["final_fixed_average"]["operator_receipt"] = (
        score_link_operator["receipt"]
    )
    score_link_analysis["analysis_sha256"] = core._payload_sha256(
        {
            key: value
            for key, value in score_link_analysis.items()
            if key != "analysis_sha256"
        }
    )
    with pytest.raises(ValueError, match="final score"):
        core.validate_analysis_operator_v6(score_link_analysis, score_link_operator)

    shifted_operator = copy.deepcopy(operator)
    shifted_analysis = copy.deepcopy(analysis)
    shifted_receipt = shifted_analysis["compact_refit"][
        "shifted_support_1024_exact_condition_receipt"
    ]
    shifted_receipt["passed"] = False
    shifted_receipt["canonical_payload_sha256"] = core._payload_sha256(
        {
            key: value
            for key, value in shifted_receipt.items()
            if key != "canonical_payload_sha256"
        }
    )
    shifted_operator["receipt"]["shifted_condition_payload_sha256"] = (
        shifted_receipt["canonical_payload_sha256"]
    )
    shifted_operator["receipt"]["canonical_payload_sha256"] = core._payload_sha256(
        {
            key: value
            for key, value in shifted_operator["receipt"].items()
            if key != "canonical_payload_sha256"
        }
    )
    shifted_analysis["final_fixed_average"]["operator_receipt"] = shifted_operator[
        "receipt"
    ]
    shifted_analysis["analysis_sha256"] = core._payload_sha256(
        {
            key: value
            for key, value in shifted_analysis.items()
            if key != "analysis_sha256"
        }
    )
    with pytest.raises(ValueError, match="shifted condition"):
        core.validate_analysis_operator_v6(shifted_analysis, shifted_operator)

    holdout_analysis = copy.deepcopy(analysis)
    holdout_analysis["holdout_policy"]["execution_order"][-2:] = reversed(
        holdout_analysis["holdout_policy"]["execution_order"][-2:]
    )
    holdout_analysis["analysis_sha256"] = core._payload_sha256(
        {
            key: value
            for key, value in holdout_analysis.items()
            if key != "analysis_sha256"
        }
    )
    with pytest.raises(ValueError, match="holdout 정책"):
        core.validate_analysis_operator_v6(holdout_analysis, operator)


def test_terminal_mutation_fails_clock_before_ls_but_holdout_does_not_change_clock(
    synthetic_capture: tuple[dict, np.ndarray, np.ndarray, dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, submitted, captured, telemetry = synthetic_capture
    original_clock = estimate_common_clock_v6(
        plan=plan, submitted_pcm=submitted, captured_pcm=captured
    )
    holdout_changed = captured.copy()
    for start, stop in plan["holdout_access_policy"]["operator_holdout_frame_ranges"]:
        holdout_changed[start:stop] = 0
    assert estimate_common_clock_v6(
        plan=plan, submitted_pcm=submitted, captured_pcm=holdout_changed
    ) == original_clock

    terminal_changed = captured.copy()
    for row in plan["layout"]:
        if row.get("stage") == "terminal_validation":
            start, stop = int(row["start_frame"]), int(row["stop_frame"])
            terminal_changed[start:stop] = np.roll(terminal_changed[start:stop], 4, axis=0)

    def forbidden(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("terminal clock failure 뒤 LS가 호출됐습니다")

    monkeypatch.setattr(core, "_fit_candidate", forbidden)
    with pytest.raises(V6ClockAdmissionError) as caught:
        analyze_committed_v6_live_delay(
            plan=plan,
            submitted_pcm=submitted,
            captured_adc_pcm=terminal_changed,
            duplex_telemetry=telemetry,
        )
    assert caught.value.stage == "terminal_clock_validation"
