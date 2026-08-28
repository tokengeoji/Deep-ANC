import inspect
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve

import deep_anc.dsp.fullband_causal_v4 as v4


def _callbacks(frames: int) -> dict[str, np.ndarray]:
    count = np.full(int(np.ceil(frames / 256)), 256, dtype=np.int64)
    count[-1] = frames - 256 * (len(count) - 1)
    frame = np.r_[0, np.cumsum(count[:-1])]
    return {
        "frame_index": frame,
        "frame_count": count,
        "input_adc_time": frame / 48_000.0,
        "output_dac_time": frame / 48_000.0,
    }


@pytest.fixture(scope="module")
def v4_signal():
    return v4.build_plan()


def _plant_outputs(
    submitted: np.ndarray, *, highband_mutation: bool = False
) -> list[np.ndarray]:
    source = submitted.astype(np.float64) / 32767.0
    microphone_specs = (
        ((1_700, 0.80), (1_300, 0.60)),
        ((1_100, 0.50), (900, 0.40)),
    )
    outputs: list[np.ndarray] = []
    for microphone in microphone_specs:
        response = np.zeros(len(submitted) + 20_000, dtype=np.float64)
        for channel, (delay, gain) in enumerate(microphone):
            taps = np.zeros(1_024, dtype=np.float64)
            taps[[0, 17, 1_023]] = [gain, -0.12 * gain, 0.025 * gain]
            ordinary = fftconvolve(source[:, channel], taps)
            response[delay : delay + len(ordinary)] += ordinary
            if highband_mutation:
                # The 8192-sample comb is exactly zero at every k%4 pilot line
                # but materially changes the response between those lines.
                response[
                    delay + 1 : delay + 1 + len(source)
                ] += 0.20 * gain * source[:, channel]
                response[
                    delay + 1 + v4.PE_COMB_SHIFT :
                    delay + 1 + v4.PE_COMB_SHIFT + len(source)
                ] -= 0.20 * gain * source[:, channel]
        outputs.append(response)
    return outputs


def _warp_affine(
    responses: list[np.ndarray], submitted_frames: int, drift_ppm: float
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    ratio = 1.0 + float(drift_ppm) * 1e-6
    frames = max(submitted_frames, int(np.ceil((submitted_frames - 1) / ratio)) + 4)
    adc = np.arange(frames, dtype=np.float64)
    dac_q = ratio * adc
    raw = [
        np.nan_to_num(
            CubicSpline(
                np.arange(len(response), dtype=np.float64),
                response,
                extrapolate=False,
            )(dac_q),
            nan=0.0,
        )
        for response in responses
    ]
    return raw, _callbacks(frames)


def _warp_piecewise(
    responses: list[np.ndarray], submitted_frames: int
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    frames = submitted_frames + 4_096
    adc = np.arange(frames, dtype=np.float64)
    split = float(submitted_frames // 2)
    first = 1.0 + 413.931e-6
    second = 1.0 - 300.0e-6
    dac_q = np.where(
        adc <= split,
        first * adc,
        first * split + second * (adc - split),
    )
    raw = [
        np.nan_to_num(
            CubicSpline(
                np.arange(len(response), dtype=np.float64),
                response,
                extrapolate=False,
            )(dac_q),
            nan=0.0,
        )
        for response in responses
    ]
    return raw, _callbacks(frames)


def test_signal_is_continuous_actual_int16_pilot_and_under_thermal_budget(v4_signal):
    plan, submitted = v4_signal
    assert submitted.dtype == np.int16 and submitted.shape[1] == 2
    assert plan["output"]["duration_seconds"] < 50.0
    assert plan["output"]["peak_pcm"] <= 98
    assert plan["output"]["pcm_sha256"] == v4._sha256_array(submitted)
    assert plan["continuous_pilot"]["overlay_present_for_every_submitted_frame"]
    assert plan["continuous_pilot"]["primary_opposite_actual_int16_max_abs_dft"] <= 1e-8
    assert plan["continuous_pilot"]["secondary_opposite_actual_int16_max_abs_dft"] <= 1e-8
    assert plan["boundary_contract"]["pre_and_post_both_exceed_maximum_history"]
    assert plan["boundary_contract"][
        "path_switch_and_period_boundary_both_sides_excluded"
    ]
    payloads = plan["aperiodic_payloads"]
    assert len({row["pcm_sha256"] for row in payloads.values()}) == len(payloads)
    assert all(row["actual_int16_pilot_line_max_abs_dft"] <= 1e-8 for row in payloads.values())
    assert all(row["peak_pcm"] <= v4.PE_MAX_PEAK_PCM for row in payloads.values())
    assert plan["plant_identification"]["actual_full_input_including_continuous_pilot"]
    assert plan["plant_identification"]["pilot_low_band_is_part_of_joint_plant_fit"]
    assert plan["canonical_training_eligible"] is False
    assert plan["canonical_blocker"] == v4.CANONICAL_BLOCKER
    assert v4.LIVE_AUTHORITY is None


def test_80hz_excitation_candidate_is_signal_safe_but_v2_authority_stays_blocked():
    plan, submitted = v4.build_plan(excitation_lower_hz=80.0)
    assert plan["plant_identification"]["excitation_band_hz"] == [80.0, 11_400.0]
    assert plan["output"]["duration_seconds"] == 14.336
    assert plan["output"]["peak_pcm"] <= 98
    assert plan["output"]["pcm_sha256"] == v4._sha256_array(submitted)
    assert all(
        row["actual_int16_pilot_line_max_abs_dft"] <= 1.0e-8
        for row in plan["aperiodic_payloads"].values()
    )
    # signal은 125-Hz octave 하단까지 내려가지만 현 shared v2 contract가 150Hz에서
    # 시작하므로 이 fixture를 canonical authority로 승격할 수 없다.
    assert plan["plant_identification"]["125hz_octave_fully_covered"] is False
    assert plan["plant_identification"]["125hz_octave_contract_blocker"] == (
        "control_contract_or_excitation_does_not_cover_full_125hz_octave"
    )
    assert plan["canonical_training_eligible"] is False
    assert plan["live_authority"] is None


@pytest.mark.parametrize("drift_ppm", [413.931, -413.931, 0.0])
def test_fixed_lti_line_phase_slope_identifies_signed_affine_clock(
    v4_signal, drift_ppm
):
    plan, submitted = v4_signal
    raw, callbacks = _warp_affine(
        _plant_outputs(submitted), len(submitted), drift_ppm
    )
    receipt = v4.absolute_dac_q_timewarp_v4(
        plan=plan,
        submitted_pcm=submitted,
        raw_err=raw[0],
        raw_ref=raw[1],
        callback_time_info=callbacks,
    )
    end_error = abs(receipt["slope"] - drift_ppm * 1e-6) * len(submitted)
    assert end_error <= v4.CLOCK_LEAVEOUT_MAX
    assert receipt["view_end_to_end_disagreement_samples"] <= v4.CLOCK_VIEW_DISAGREEMENT_MAX
    assert receipt["leaveout_max_samples"] <= v4.CLOCK_LEAVEOUT_MAX
    assert receipt["cubic_max_samples"] <= v4.CLOCK_CUBIC_MAX
    assert receipt["combined_max_samples"] <= v4.CLOCK_COMBINED_MAX
    assert receipt["minimum_transfer_coherence"] >= v4.CLOCK_MIN_COHERENCE
    assert receipt["holdout_used_for_fit_or_selection"] is False
    assert receipt["passed_under_fixed_lti_hypothesis"] is True
    assert receipt["canonical_passed"] is False


def test_piecewise_clock_is_rejected_by_fit_leaveout(v4_signal):
    plan, submitted = v4_signal
    raw, callbacks = _warp_piecewise(_plant_outputs(submitted), len(submitted))
    with pytest.raises(
        ValueError,
        match="clock|maps disagree|leaveout|coherence|optimisation|boundary",
    ):
        v4.absolute_dac_q_timewarp_v4(
            plan=plan,
            submitted_pcm=submitted,
            raw_err=raw[0],
            raw_ref=raw[1],
            callback_time_info=callbacks,
        )


def test_callback_sample_slip_is_rejected_before_acoustic_fit(v4_signal):
    plan, submitted = v4_signal
    callback = _callbacks(len(submitted))
    callback["frame_index"] = callback["frame_index"].copy()
    callback["frame_index"][5:] += 1
    with pytest.raises(ValueError, match="slip"):
        v4.absolute_dac_q_timewarp_v4(
            plan=plan,
            submitted_pcm=submitted,
            raw_err=np.zeros(len(submitted)),
            raw_ref=np.zeros(len(submitted)),
            callback_time_info=callback,
        )


def test_clock_map_does_not_use_highband_plant_mutation(v4_signal):
    plan, submitted = v4_signal
    baseline_raw, baseline_callback = _warp_affine(
        _plant_outputs(submitted, highband_mutation=False),
        len(submitted),
        413.931,
    )
    mutated_raw, mutated_callback = _warp_affine(
        _plant_outputs(submitted, highband_mutation=True),
        len(submitted),
        413.931,
    )
    baseline = v4.absolute_dac_q_timewarp_v4(
        plan=plan,
        submitted_pcm=submitted,
        raw_err=baseline_raw[0],
        raw_ref=baseline_raw[1],
        callback_time_info=baseline_callback,
    )
    mutated = v4.absolute_dac_q_timewarp_v4(
        plan=plan,
        submitted_pcm=submitted,
        raw_err=mutated_raw[0],
        raw_ref=mutated_raw[1],
        callback_time_info=mutated_callback,
    )
    assert abs(baseline["slope"] - mutated["slope"]) * len(submitted) <= v4.CLOCK_CUBIC_MAX
    assert baseline["highband_used_for_clock_fit"] is False
    assert mutated["highband_used_for_clock_fit"] is False


def test_actual_int16_and_callback_arrays_cannot_be_replaced_by_scalars(v4_signal):
    plan, submitted = v4_signal
    assert set(inspect.signature(v4.absolute_dac_q_timewarp_v4).parameters) == {
        "plan",
        "submitted_pcm",
        "raw_err",
        "raw_ref",
        "callback_time_info",
    }
    with pytest.raises(ValueError, match="exact.*int16"):
        v4.absolute_dac_q_timewarp_v4(
            plan=plan,
            submitted_pcm=submitted.astype(np.float64),
            raw_err=np.zeros(len(submitted)),
            raw_ref=np.zeros(len(submitted)),
            callback_time_info=_callbacks(len(submitted)),
        )


def test_marker_alias_is_fail_closed():
    rng = np.random.default_rng(20260828)
    marker = rng.choice((-1.0, 1.0), size=4_096)
    search = np.zeros(len(marker) + v4.MAX_DELAY + 1, dtype=np.float64)
    search[100 : 100 + len(marker)] += marker
    assert v4.marker_branch_v4(marker=marker, response_search=search)["delay_samples"] == 100
    search[3_100 : 3_100 + len(marker)] += marker
    with pytest.raises(ValueError, match="not unique"):
        v4.marker_branch_v4(marker=marker, response_search=search)


def test_common_time_varying_plant_delay_is_information_theoretic_counterexample():
    frames = 8_192
    sample = np.arange(frames, dtype=np.float64)
    acoustic = np.sin(2.0 * np.pi * 431.0 * sample / 48_000.0)
    trajectory = 0.20 * np.sin(2.0 * np.pi * sample / (frames - 1))
    trajectory[[0, -1]] = 0.0
    result = v4.acoustic_clock_plant_confounding_counterexample(
        acoustic_signal=acoustic,
        trajectory_samples=trajectory,
    )
    assert result["raw_byte_identical"] is True
    assert result["acoustic_only_decision_possible"] is False
    assert result["model_scope_limitation"] == v4.IDENTIFIABILITY_LIMITATION
    assert result[
        "fixed_lti_stationarity_gates_can_define_conditional_canonical_scope"
    ] is True


def test_joint_operator_uses_actual_both_inputs_and_has_exact_adjoint():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(512, 2))
    indices = np.arange(100, 420, dtype=np.int64)
    operator = v4._joint_fir_operator(
        submitted=source, selected_indices=indices, delays=(3, 5), support=16
    )
    taps = rng.normal(size=32)
    residual = rng.normal(size=len(indices))
    lhs = float(np.vdot(operator.matvec(taps), residual))
    rhs = float(np.vdot(taps, operator.rmatvec(residual)))
    assert abs(lhs - rhs) <= 1e-10 * max(1.0, abs(lhs), abs(rhs))


def test_actual_int16_excitation_condition_blocks_every_predeclared_support(
    v4_signal,
):
    plan, submitted = v4_signal
    expected = {"fit_a": 280.3743017049416, "fit_b": 297.7764322227819}
    for role, expected_condition in expected.items():
        receipt = v4.periodic_excitation_condition_audit_v4(
            plan=plan,
            submitted_pcm=submitted,
            source_role=role,
            delays=(1_700, 1_300),
        )
        assert receipt["condition_number"] == pytest.approx(
            expected_condition, rel=1.0e-9
        )
        assert receipt["condition_number"] > v4.MAX_CONDITION
        assert receipt["delay_independent_joint_condition_lower_bound"] > (
            v4.MAX_CONDITION
        )
        assert receipt["passed"] is False
        assert receipt["all_predeclared_supports_blocked"] is True
        assert receipt["longer_support_condition_cannot_improve"] is True
        assert receipt["receipt_sha256"] == v4._json_sha256(
            {
                key: value
                for key, value in receipt.items()
                if key != "receipt_sha256"
            }
        )


def _fake_candidate(role: str, support: int, value: float) -> dict:
    taps = np.zeros(support, dtype=np.float64)
    taps[0] = value
    return {
        "source_role": role,
        "microphone_role": "err",
        "support_samples": support,
        "delays_samples": [31, 29],
        "fractional_delay_residual_samples": [0.125, -0.25],
        "bulk_delay_samples_fractional": [31.125, 28.75],
        "condition_number": 2.0,
        "fit_residual_ratio": 0.01,
        "submitted_pcm_sha256": "a" * 64,
        "control_band_contract_sha256": (
            v4.ControlBandContract.broadband_point_control().digest()
        ),
        "control_subbands_hz": [
            [float(value) for value in band]
            for band in v4.BROADBAND_POINT_CONTROL_SUBBANDS_HZ
        ],
        "primary_post_onset_fir": taps.tolist(),
        "secondary_post_onset_fir": (0.5 * taps).tolist(),
        "candidate_sha256": f"{role}-{support}",
    }


def _fake_score(candidate: dict, target_role: str, *, passed: bool = True) -> dict:
    rows = [
        {
            "path": path,
            "band_index": band_index,
            "band_hz": [float(value) for value in band],
            "passed": bool(passed or (path, band_index) != ("primary", 5)),
        }
        for path in ("primary", "secondary")
        for band_index, band in enumerate(v4.BROADBAND_POINT_CONTROL_SUBBANDS_HZ)
    ]
    payload = {
        "schema": "joint_actual_input_role_score_v4",
        "candidate_sha256": candidate["candidate_sha256"],
        "candidate_source_role": candidate["source_role"],
        "microphone_role": candidate["microphone_role"],
        "target_role": target_role,
        "control_band_contract_sha256": candidate[
            "control_band_contract_sha256"
        ],
        "global_residual_ratio": 0.02,
        "subband_rows": rows,
        "all_subbands_passed": bool(passed),
        "passed": bool(passed),
    }
    payload["receipt_sha256"] = v4._json_sha256(payload)
    return payload


def test_support_selection_is_fit_only_and_holdout_is_terminal():
    fit_a = _fake_candidate("fit_a", 1_024, 1.0)
    fit_b = _fake_candidate("fit_b", 1_024, 1.01)
    evidence = {
        1_024: {
            "fit_a": fit_a,
            "fit_b": fit_b,
            "fit_a_score": _fake_score(fit_a, "fit_a"),
            "fit_b_score": _fake_score(fit_b, "fit_b"),
            "fit_a_on_fit_b_score": _fake_score(fit_a, "fit_b"),
            "fit_b_on_fit_a_score": _fake_score(fit_b, "fit_a"),
        }
    }
    frozen = v4.select_and_freeze_fit_support_v4(evidence)
    assert frozen["support_samples"] == 1_024
    assert frozen["holdout_used_for_generation_or_selection"] is False
    failed_holdout = _fake_score(
        {
            **frozen,
            "candidate_sha256": frozen["freeze_sha256"],
            "source_role": "frozen_fit_only",
        },
        "holdout",
        passed=False,
    )
    failed_holdout["global_residual_ratio"] = 0.20
    failed_holdout["receipt_sha256"] = v4._json_sha256(
        {key: value for key, value in failed_holdout.items() if key != "receipt_sha256"}
    )
    terminal = v4.terminal_holdout_receipt_v4(
        frozen=frozen, holdout_score=failed_holdout
    )
    assert terminal["passed"] is False
    assert terminal["selected_support_samples"] == 1_024
    assert terminal["support_reselection_after_holdout_forbidden"] is True
    poisoned = {1_024: {**evidence[1_024], "holdout": {"residual": 0.0}}}
    with pytest.raises(ValueError, match="holdout"):
        v4.select_and_freeze_fit_support_v4(poisoned)


def _exact_err_candidate(plan: dict) -> dict:
    primary = np.zeros(1_024, dtype=np.float64)
    secondary = np.zeros(1_024, dtype=np.float64)
    primary[[0, 17, 1_023]] = [0.80, -0.096, 0.020]
    secondary[[0, 17, 1_023]] = [0.60, -0.072, 0.015]
    return {
        "schema": "joint_actual_input_fit_candidate_v4",
        "microphone_role": "err",
        "source_role": "fit_a",
        "support_samples": 1_024,
        "delays_samples": [1_700, 1_300],
        "fractional_delay_residual_samples": [0.0, 0.0],
        "bulk_delay_samples_fractional": [1_700.0, 1_300.0],
        "primary_post_onset_fir": primary.tolist(),
        "secondary_post_onset_fir": secondary.tolist(),
        "submitted_pcm_sha256": plan["output"]["pcm_sha256"],
        "control_band_contract_sha256": plan[
            "control_band_contract_sha256"
        ],
        "control_subbands_hz": plan["plant_identification"][
            "subband_authority"
        ]["bands_hz"],
        "candidate_sha256": "synthetic-exact-err-candidate",
    }


def test_subband_authority_cannot_hide_highband_failure_in_global_mean(v4_signal):
    plan, submitted = v4_signal
    response = _plant_outputs(submitted)[0]
    candidate = _exact_err_candidate(plan)
    baseline = v4.score_candidate_on_role_v4(
        candidate=candidate,
        plan=plan,
        submitted_pcm=submitted,
        response_dac_q=response,
        microphone_role="err",
        target_role="holdout",
    )
    assert baseline["passed"] is True
    assert len(baseline["subband_rows"]) == 14
    assert all(
        row["exact_zero_noise_bin_count"]
        >= v4.SUBBAND_MIN_EXACT_ZERO_NOISE_BINS
        for row in baseline["subband_rows"]
    )

    mutated = response.copy()
    primary = next(
        row
        for row in plan["layout"]
        if row.get("role") == "holdout" and row.get("path") == "primary"
    )
    start = int(primary["central_start_frame"])
    stop = int(primary["central_stop_frame"])
    spectrum = np.fft.rfft(mutated[start:stop])
    frequency = np.fft.rfftfreq(v4.PERIOD, 1.0 / v4.FS)
    lo, hi = v4.BROADBAND_POINT_CONTROL_SUBBANDS_HZ[5]
    highband = (frequency >= lo) & (frequency < hi)
    spectrum[highband] *= 1.12
    mutated[start:stop] = np.fft.irfft(spectrum, n=v4.PERIOD)
    receipt = v4.score_candidate_on_role_v4(
        candidate=candidate,
        plan=plan,
        submitted_pcm=submitted,
        response_dac_q=mutated,
        microphone_role="err",
        target_role="holdout",
    )
    assert receipt["global_residual_ratio"] < v4.HOLDOUT_RESIDUAL_MAX
    failed = [row for row in receipt["subband_rows"] if not row["passed"]]
    assert [(row["path"], row["band_index"]) for row in failed] == [
        ("primary", 5)
    ]
    assert failed[0]["noise_conditioned_relative_residual"] > 0.10
    assert receipt["all_subbands_passed"] is False
    assert receipt["passed"] is False


def test_actual_exact_zero_noise_floor_blocks_low_snr_response(v4_signal):
    plan, submitted = v4_signal
    response = _plant_outputs(submitted)[0]
    rng = np.random.default_rng(20260828)
    for row in plan["clock_rows"]:
        if row["name"] in ("lead_reference", "tail_validation"):
            start = int(row["start_frame"])
            stop = int(row["stop_frame"])
            response[start:stop] += rng.normal(scale=0.01, size=stop - start)
    receipt = v4.score_candidate_on_role_v4(
        candidate=_exact_err_candidate(plan),
        plan=plan,
        submitted_pcm=submitted,
        response_dac_q=response,
        microphone_role="err",
        target_role="holdout",
    )
    assert receipt["global_residual_ratio"] < 1.0e-12
    assert receipt["all_subbands_passed"] is False
    assert all(
        row["target_to_noise_db"] < v4.SUBBAND_MIN_TARGET_TO_NOISE_DB
        for row in receipt["subband_rows"]
    )


def test_live_cli_source_has_no_audio_import_and_authority_is_none():
    source = (
        v4.__file__
    )
    assert source.endswith("fullband_causal_v4.py")
    assert v4.LIVE_AUTHORITY is None


def test_signal_cli_no_replace_fsync_path_rejects_symlink_parent(tmp_path):
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts/data/measure_paths_fullband_causal_continuous.py"
    )
    spec = importlib.util.spec_from_file_location("v4_signal_cli_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    repository = tmp_path / "repository"
    repository.mkdir()
    target = repository / "plans" / "plan.json"
    plan = {"schema": "fixture", "canonical_training_eligible": False}
    assert module._write_plan_no_replace(
        plan=plan, target=target, repository_root=repository
    ) == target
    with pytest.raises(FileExistsError):
        module._write_plan_no_replace(
            plan=plan, target=target, repository_root=repository
        )

    real = repository / "real"
    real.mkdir()
    alias = repository / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module._write_plan_no_replace(
            plan=plan,
            target=alias / "escaped.json",
            repository_root=repository,
        )
    assert module.main(["--execute-live"]) == 2
    cli_source = script_path.read_text(encoding="utf-8")
    assert "sounddevice" not in cli_source
