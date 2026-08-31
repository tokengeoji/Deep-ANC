from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.signal import lfilter

from deep_anc.audio_duplex_stage2 import (
    OUTPUT_MASTER_TELEMETRY_SCHEMA,
    OutputMasterCaptureFailure,
)
from deep_anc.dsp.stage2_2khz_diagnostic_clock import DIAGNOSTIC_CLOCK_SCHEMA
from deep_anc.dsp.stage2_2khz_measurement_v2 import (
    DPSS_REPRESENTATION_GUARD_HZ,
    _payload_sha256,
    build_stage2_v2_live_safe_fallback_plan,
)
from deep_anc.dsp.stage2_2khz_level_contract import (
    build_stage2_physical_operating_level_evidence,
)
import deep_anc.dsp.stage2_2khz_output_master_ps_v3 as ps_v3_module
from deep_anc.dsp.stage2_2khz_output_master_diagnostic import (
    POST_ROLL_FRAMES,
    PRE_ROLL_FRAMES,
    analyse_and_publish_output_master_clock_receipt,
    output_master_session_targets,
    publish_output_master_raw_no_replace,
    validate_output_master_success_telemetry,
)
from deep_anc.dsp.stage2_2khz_output_master_ps_v3 import (
    PS_V3_ADMISSION_SCHEMA,
    PS_V3_PLAN_SCHEMA,
    PS_V3_RAW_STRUCTURE_SCHEMA,
    analyse_stage2_output_master_diagnostic_linearity_v3,
    analyse_stage2_output_master_ps_v3_capture,
    assess_stage2_output_master_ps_v3_admission,
    build_stage2_output_master_ps_v3_plan,
    estimate_stage2_output_master_ps_clock_and_resample,
    inspect_stage2_output_master_ps_v3_raw_structure,
    output_master_ps_v3_session_targets,
    publish_stage2_output_master_diagnostic_linearity_v3_no_replace,
    publish_stage2_output_master_ps_v3_physical_level_no_replace,
    publish_stage2_output_master_ps_v3_partial_raw_no_replace,
    publish_stage2_output_master_ps_v3_raw_no_replace,
    run_stage2_output_master_ps_v3_if_admitted,
    validate_output_master_diagnostic_clock_publication,
)


ROOT = Path(__file__).resolve().parents[1]


def _capture_metadata(capture_id: str = "c" * 32) -> dict:
    hardware_identity = {
        "schema": "synthetic_hardware_identity_v1",
        "physical_fingerprint": {"fixture": "two-clock-two-by-two-lti"},
    }
    devices = {"input": 5, "output": 24}
    authority = {
        "schema": "stage2_2khz_output_master_origin_dev_exact_bundle_v1",
        "branch": "dev",
        "head": "a" * 40,
        "origin_dev": "a" * 40,
        "critical_file_sha256": {"fixture.py": "b" * 64},
    }
    return {
        "capture_id": capture_id,
        "repository_execution_identity": {
            "repository_commit": "a" * 40,
            "repository_branch": "dev",
            "repository_dirty": False,
            "script_path": "scripts/data/fixture.py",
            "script_file_sha256": "b" * 64,
        },
        "measurement_git_authority": authority,
        "hardware_config_sha256": "d" * 64,
        "resolved_devices": devices,
        "fresh_meter": {
            "path": "results/stage2_v3_fixture/meter_raw.npz",
            "sha256": "1" * 64,
            "receipt_path": "results/stage2_v3_fixture/meter_receipt.json",
            "receipt_sha256": "2" * 64,
            "capture_id": "meter-fixture",
            "completed_at_utc": "2026-09-01T00:00:00+00:00",
            "age_seconds": 1.0,
            "meter_ch0_dbfs": -50.1,
            "freshness_max_seconds": 600,
            "resolved_devices": devices,
            "physical_fingerprint": hardware_identity["physical_fingerprint"],
            "hardware_identity": hardware_identity,
            "calibration_evidence": {
                "path": "results/stage2_v3_fixture/calibration.json",
                "sha256": "3" * 64,
            },
        },
        "operator_confirmations": {
            "speaker_output": True,
            "user_present": True,
            "volume_fixed_after_meter_adjustment": True,
            "routing_and_geometry": True,
            "same_amplifier_setting": True,
        },
    }


def _fake_reopen_bound_meter(_root, binding, *, require_fresh):
    del require_fresh
    return {
        "verified_meter": {
            "sha256": binding["meter_raw_artifact"]["sha256"],
            "metadata": {
                "resolved_devices": binding["resolved_devices"],
                "hardware_identity": binding["hardware_identity"],
            },
        },
        "artifact_payload_sha256": {
            "meter_raw_artifact": binding["meter_raw_artifact"]["sha256"],
            "meter_receipt_artifact": binding["meter_receipt_artifact"]["sha256"],
            "calibration_evidence_artifact": binding[
                "calibration_evidence_artifact"
            ]["sha256"],
        },
    }


def _success_telemetry(submitted: np.ndarray, captured: np.ndarray) -> dict:
    output_blocks = len(submitted) // 256
    input_blocks = len(captured) // 256
    output_sequence = np.arange(output_blocks, dtype="<i8")
    input_sequence = np.arange(input_blocks, dtype="<i8")
    return {
        "schema": OUTPUT_MASTER_TELEMETRY_SCHEMA,
        "transport": "independent_input_output_streams_output_clock_master",
        "output_clock_owner": "outputstream_callback_only",
        "input_role": "raw_witness_only_never_output_pacing",
        "cross_clock_timestamp_alignment_used": False,
        "input_output_frame_identity_claimed": False,
        "hardware_sample_slip_authority": False,
        "legacy_combined_duplex_used": False,
        "completed": True,
        "normal_stop_completed": True,
        "failure_events": [],
        "canonical_output_frames": len(submitted),
        "submitted_output_frames": len(submitted),
        "captured_input_frames": len(captured),
        "pre_roll_requested_frames": PRE_ROLL_FRAMES,
        "post_roll_requested_frames": POST_ROLL_FRAMES,
        "pre_roll_observed_input_frames": PRE_ROLL_FRAMES,
        "post_roll_observed_input_frames": POST_ROLL_FRAMES,
        "input_frame_cursor_at_output_start": PRE_ROLL_FRAMES,
        "input_frame_cursor_at_output_complete": len(captured) - POST_ROLL_FRAMES,
        "capture_monotonic_elapsed_seconds": 1.0,
        "actual_submitted_pcm": np.asarray(submitted, dtype="<i2"),
        "capture_valid_mask": np.ones(len(captured), dtype="bool"),
        "submitted_valid_mask": np.ones(len(submitted), dtype="bool"),
        "input_callback_sequence": input_sequence,
        "input_callback_start_frames": input_sequence * 256,
        "input_callback_frame_counts": np.full(input_blocks, 256, dtype="<i8"),
        "input_buffer_adc_time": 10.0 + input_sequence.astype("<f8") / 100.0,
        "input_callback_current_time": 20.0 + input_sequence.astype("<f8") / 100.0,
        "input_callback_status_bitmask": np.zeros(input_blocks, dtype="<u4"),
        "output_callback_sequence": output_sequence,
        "output_callback_start_frames": output_sequence * 256,
        "output_callback_frame_counts": np.full(output_blocks, 256, dtype="<i8"),
        "output_buffer_dac_time": 30.0 + output_sequence.astype("<f8") / 100.0,
        "output_callback_current_time": 40.0 + output_sequence.astype("<f8") / 100.0,
        "output_callback_status_bitmask": np.zeros(output_blocks, dtype="<u4"),
    }


def _passing_clock_stub(plan, submitted, captured):
    receipt = {
        "schema": DIAGNOSTIC_CLOCK_SCHEMA,
        "signal_plan_sha256": plan["canonical_payload_sha256"],
        "submitted_phase_frames": len(submitted),
        "captured_frames": len(captured),
        "passed": True,
        "diagnostic_linearity_may_run": True,
        "ps_phase_may_start": False,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt


@pytest.fixture(scope="module")
def durable_diagnostic(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage2_ps_v3_diagnostic")
    session = "results/stage2_2khz_output_master_diagnostic/pass_session"
    plan, full = build_stage2_v2_live_safe_fallback_plan()
    boundary = int(plan["live_phase_contract"]["diagnostic_phase_stop_frame"])
    submitted = np.asarray(full[:boundary])
    captured = np.zeros(
        (len(submitted) + PRE_ROLL_FRAMES + POST_ROLL_FRAMES, 2), dtype="<i4"
    )
    telemetry = _success_telemetry(submitted, captured)
    raw = publish_output_master_raw_no_replace(
        str(root),
        session,
        plan,
        full,
        captured_pcm=captured,
        telemetry=telemetry,
        capture_metadata=_capture_metadata("a" * 32),
    )
    clock = analyse_and_publish_output_master_clock_receipt(
        str(root),
        session,
        plan,
        full,
        raw_publication=raw,
        clock_estimator=_passing_clock_stub,
    )
    publication = {
        "path": clock["path"],
        "sha256": clock["sha256"],
    }
    return root, session, plan, full, raw, publication


def _render_two_by_two_lti(submitted: np.ndarray) -> np.ndarray:
    source = np.asarray(submitted, dtype=np.float64) / 32768.0
    response = np.zeros_like(source)
    # secondary가 primary보다 이르고 약 2배 강한 causal same-capture fixture다.
    filters = (
        ((700, 0.5), (710, 0.3)),
        ((520, 1.0), (530, 0.6)),
    )
    for output_channel in range(2):
        for microphone in range(2):
            delay, gain = filters[output_channel][microphone]
            response[delay:, microphone] += (
                gain * source[:-delay, output_channel]
            )
    return response


def _render_variable_input_clock(
    response: np.ndarray, *, ppm: float = 250.0
) -> np.ndarray:
    q = 1.0 + ppm * 1.0e-6
    needed = (
        PRE_ROLL_FRAMES
        + int(np.ceil(len(response) / q))
        + POST_ROLL_FRAMES
    )
    frames = int(math.ceil(needed / 256.0) * 256)
    input_index = np.arange(frames, dtype=np.float64)
    output_index = q * (input_index - PRE_ROLL_FRAMES)
    captured = np.column_stack(
        [
            np.interp(
                output_index,
                np.arange(len(response), dtype=np.float64),
                response[:, microphone],
                left=0.0,
                right=0.0,
            )
            for microphone in range(2)
        ]
    )
    return np.rint(captured * 2147483648.0).astype("<i4")


def _physical_level_fixture(signal_plan_sha256: str) -> dict:
    names = (
        "meter_raw",
        "meter_receipt",
        "calibration",
        "diagnostic_raw",
        "authorization",
        "ps_raw",
    )
    refs = {
        name: {
            "path": f"results/stage2_v3_fixture/{name}.bin",
            "sha256": f"{index + 1:064x}",
        }
        for index, name in enumerate(names)
    }
    return build_stage2_physical_operating_level_evidence(
        signal_plan_sha256=signal_plan_sha256,
        capture_id="stage2-output-master-v3-synthetic-lti",
        hardware_identity={"fixture": "two-clock-two-by-two-lti"},
        meter_raw_artifact=refs["meter_raw"],
        meter_receipt_artifact=refs["meter_receipt"],
        calibration_evidence_artifact=refs["calibration"],
        diagnostic_raw_artifact=refs["diagnostic_raw"],
        diagnostic_authorization_artifact=refs["authorization"],
        ps_raw_artifact=refs["ps_raw"],
    )


@pytest.fixture(scope="module")
def positive_end_to_end(tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        ps_v3_module, "_reopen_bound_meter", _fake_reopen_bound_meter
    )
    root = tmp_path_factory.mktemp("stage2_ps_v3_positive")
    session = "results/stage2_2khz_output_master_diagnostic/lti_session"
    ps_session = "results/stage2_2khz_output_master_ps_v3/lti_session"
    plan, full = build_stage2_v2_live_safe_fallback_plan()
    boundary = int(plan["live_phase_contract"]["diagnostic_phase_stop_frame"])

    diagnostic_submitted = np.asarray(full[:boundary])
    diagnostic_input = _render_variable_input_clock(
        _render_two_by_two_lti(diagnostic_submitted)
    )
    diagnostic_telemetry = _success_telemetry(
        diagnostic_submitted, diagnostic_input
    )
    diagnostic_raw = publish_output_master_raw_no_replace(
        str(root),
        session,
        plan,
        full,
        captured_pcm=diagnostic_input,
        telemetry=diagnostic_telemetry,
        capture_metadata=_capture_metadata("c" * 32),
    )
    clock = analyse_and_publish_output_master_clock_receipt(
        str(root), session, plan, full, raw_publication=diagnostic_raw
    )
    clock_ref = {"path": clock["path"], "sha256": clock["sha256"]}
    linearity = publish_stage2_output_master_diagnostic_linearity_v3_no_replace(
        str(root), session, clock_ref, plan, full
    )
    linearity_ref = {"path": linearity["path"], "sha256": linearity["sha256"]}

    ps_submitted = np.asarray(full[boundary:])
    ps_input = _render_variable_input_clock(
        _render_two_by_two_lti(ps_submitted)
    )
    ps_telemetry = _success_telemetry(ps_submitted, ps_input)
    diagnostic = analyse_stage2_output_master_diagnostic_linearity_v3(
        plan,
        full,
        repository_root=str(root),
        diagnostic_session_relative_path=session,
        diagnostic_clock_publication=clock_ref,
    )
    ps_clock = estimate_stage2_output_master_ps_clock_and_resample(
        plan, full, ps_input, ps_telemetry
    )
    ps_raw = publish_stage2_output_master_ps_v3_raw_no_replace(
        str(root),
        ps_session,
        plan,
        full,
        diagnostic_session_relative_path=session,
        diagnostic_clock_publication=clock_ref,
        diagnostic_linearity_publication=linearity_ref,
        captured_ps_input_pcm=ps_input,
        ps_telemetry=ps_telemetry,
        capture_metadata=_capture_metadata("c" * 32),
    )
    ps_raw_ref = {"path": ps_raw["path"], "sha256": ps_raw["sha256"]}
    level = publish_stage2_output_master_ps_v3_physical_level_no_replace(
        str(root),
        ps_session,
        ps_raw_ref,
        plan,
        full,
        diagnostic_session_relative_path=session,
        diagnostic_clock_publication=clock_ref,
        diagnostic_linearity_publication=linearity_ref,
    )
    analysis, arrays = analyse_stage2_output_master_ps_v3_capture(
        plan,
        full,
        repository_root=str(root),
        diagnostic_session_relative_path=session,
        diagnostic_clock_publication=clock_ref,
        diagnostic_linearity_publication=linearity_ref,
        ps_session_relative_path=ps_session,
        ps_raw_publication=ps_raw_ref,
        physical_level_publication={
            "path": level["path"],
            "sha256": level["sha256"],
        },
    )
    monkeypatch.undo()
    return plan, diagnostic, ps_clock, analysis, arrays, {
        "root": root,
        "diagnostic_session": session,
        "ps_session": ps_session,
        "full": full,
        "clock_ref": clock_ref,
        "linearity_ref": linearity_ref,
        "ps_raw_ref": ps_raw_ref,
        "ps_input": ps_input,
        "ps_telemetry": ps_telemetry,
    }


def test_v3_plan_separates_output_and_variable_input_clock_axes() -> None:
    plan, full = build_stage2_v2_live_safe_fallback_plan()
    v3, submitted = build_stage2_output_master_ps_v3_plan(plan, full)

    assert v3["schema"] == PS_V3_PLAN_SCHEMA
    assert v3["ps_output_frames"] == 594_944
    assert v3["ps_output_seconds"] == pytest.approx(12.394666666666666)
    assert submitted.shape == (594_944, 2)
    assert v3["transport"]["input_frame_count_is_independent_and_variable"] is True
    assert v3["transport"]["input_output_frame_identity_claimed"] is False
    assert v3["transport"]["legacy_combined_duplex_allowed"] is False
    assert v3["ps_stream_may_open"] is False
    assert v3["plant_identification_eligible"] is False


def test_two_clock_lti_recovers_q_and_clock_corrected_diagnostic(
    positive_end_to_end,
) -> None:
    _plan, diagnostic, ps_clock, _analysis, _arrays, _artifacts = positive_end_to_end

    assert DPSS_REPRESENTATION_GUARD_HZ == 100.0
    assert diagnostic["receipt"]["passed"] is True
    assert diagnostic["receipt"]["ps_phase_may_start"] is True
    assert all(
        diagnostic["receipt"]["output_grid_analysis_receipts"][kind]["passed"]
        for kind in ("cubic", "linear")
    )
    clock = ps_clock["receipt"]["ps_local_clock_receipt"]
    assert clock["estimated_ppm"] == pytest.approx(250.0, abs=0.02)
    assert ps_clock["receipt"]["relative_p_minus_s_time_gauge_cancels"] is True
    assert ps_clock["receipt"]["absolute_hardware_clock_authority_claimed"] is False
    assert ps_clock["cubic_output_grid_pcm"].shape == (594_944, 2)
    assert ps_clock["linear_output_grid_pcm"].shape == (594_944, 2)


def test_end_to_end_dpss_fit_holdout_preserves_lowest_band_and_relative_delay(
    positive_end_to_end,
) -> None:
    _plan, _diagnostic, _ps_clock, analysis, arrays, _artifacts = positive_end_to_end

    assert analysis["status"] == (
        "PASS_RELATIVE_PS_CANDIDATE_PLANT_BINDING_STILL_REQUIRED"
    )
    assert analysis["relative_ps_authority"] is True
    assert analysis["absolute_hardware_clock_authority_claimed"] is False
    assert analysis["canonical_training_eligible"] is False
    assert analysis["cubic_coarse_zeros_before_fir_samples"] == [444, 264]
    assert analysis["linear_coarse_zeros_before_fir_samples"] == [444, 264]
    assert (
        analysis["cubic_coarse_zeros_before_fir_samples"][1]
        - analysis["cubic_coarse_zeros_before_fir_samples"][0]
        == -180
    )
    for key in ("cubic_physical_subband_rows", "linear_physical_subband_rows"):
        rows = analysis[key]
        lowest = [row for row in rows if row["band_hz"] == [88.3883476483, 150.0]]
        assert len(lowest) == 4
        assert all(row["passed"] for row in lowest)
        assert min(row["untouched_holdout_complex_agreement"] for row in lowest) >= 0.95
        assert max(row["untouched_holdout_magnitude_ratio_error_db"] for row in lowest) <= 1.0
    assert all(row["passed"] for row in analysis["cubic_linear_fir_subband_rows"])
    assert analysis["cubic_actuator_feasibility"]["passed"] is True
    assert analysis["linear_actuator_feasibility"]["passed"] is True
    assert arrays["primary_fir_by_mic"].shape == (2, 1024)
    assert arrays["secondary_fir_by_mic"].shape == (2, 1024)


def test_ps_success_raw_is_no_replace_after_full_revalidation(
    positive_end_to_end, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _diagnostic, _clock, _analysis, _arrays, artifacts = positive_end_to_end
    monkeypatch.setattr(
        ps_v3_module, "_reopen_bound_meter", _fake_reopen_bound_meter
    )
    before = (artifacts["root"] / artifacts["ps_raw_ref"]["path"]).read_bytes()
    with pytest.raises(FileExistsError):
        publish_stage2_output_master_ps_v3_raw_no_replace(
            str(artifacts["root"]),
            artifacts["ps_session"],
            plan,
            artifacts["full"],
            diagnostic_session_relative_path=artifacts["diagnostic_session"],
            diagnostic_clock_publication=artifacts["clock_ref"],
            diagnostic_linearity_publication=artifacts["linearity_ref"],
            captured_ps_input_pcm=artifacts["ps_input"],
            ps_telemetry=artifacts["ps_telemetry"],
            capture_metadata=_capture_metadata("c" * 32),
        )
    assert (artifacts["root"] / artifacts["ps_raw_ref"]["path"]).read_bytes() == before


def test_full_durable_diagnostic_and_same_physical_binding_open_exactly_one_ps_call(
    positive_end_to_end, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _diagnostic, _clock, _analysis, _arrays, artifacts = positive_end_to_end
    monkeypatch.setattr(
        ps_v3_module, "_reopen_bound_meter", _fake_reopen_bound_meter
    )
    admission = ps_v3_module._assess_stage2_output_master_ps_v3_admission_after_cli_authority(
        plan,
        artifacts["full"],
        repository_root=str(artifacts["root"]),
        diagnostic_session_relative_path=artifacts["diagnostic_session"],
        diagnostic_clock_publication=artifacts["clock_ref"],
        diagnostic_linearity_publication=artifacts["linearity_ref"],
        ps_capture_metadata=_capture_metadata("c" * 32),
    )

    assert admission["blockers"] == []
    assert admission["ps_stream_may_open"] is True
    assert admission["ps_backend_calls_allowed"] == 1
    assert admission["plant_identification_eligible"] is False


def test_public_executor_cannot_invoke_callback_even_with_real_artifact_mappings(
    positive_end_to_end,
) -> None:
    plan, _diagnostic, _clock, _analysis, _arrays, artifacts = positive_end_to_end
    calls: list[str] = []
    result = run_stage2_output_master_ps_v3_if_admitted(
        plan,
        artifacts["full"],
        capture_callable=lambda **_kwargs: calls.append("capture"),
        repository_root=str(artifacts["root"]),
        diagnostic_session_relative_path=artifacts["diagnostic_session"],
        diagnostic_clock_publication=artifacts["clock_ref"],
        diagnostic_linearity_publication=artifacts["linearity_ref"],
        ps_capture_metadata=_capture_metadata("c" * 32),
    )

    assert result["admission"]["status"] == (
        "BLOCKED_PUBLIC_LIBRARY_CANNOT_AUTHORIZE_LIVE_AUDIO"
    )
    assert result["capture_callable_invoked"] is False
    assert result["ps_backend_calls"] == 0
    assert calls == []


def test_post_capture_continuity_or_publisher_failure_preserves_recovery_raw(
    positive_end_to_end, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _diagnostic, _clock, _analysis, _arrays, artifacts = positive_end_to_end
    monkeypatch.setattr(
        ps_v3_module, "_reopen_bound_meter", _fake_reopen_bound_meter
    )

    def fail_after_capture(*_args, **_kwargs):
        raise RuntimeError("simulated post-capture continuity expiry")

    monkeypatch.setattr(
        ps_v3_module,
        "publish_stage2_output_master_ps_v3_raw_no_replace",
        fail_after_capture,
    )
    recovery_session = (
        "results/stage2_2khz_output_master_ps_v3/post_capture_recovery_fixture"
    )
    result = ps_v3_module._run_stage2_output_master_ps_v3_after_cli_authority(
        plan,
        artifacts["full"],
        capture_callable=lambda **_kwargs: (
            artifacts["ps_input"],
            artifacts["ps_telemetry"],
        ),
        repository_root=str(artifacts["root"]),
        diagnostic_session_relative_path=artifacts["diagnostic_session"],
        diagnostic_clock_publication=artifacts["clock_ref"],
        diagnostic_linearity_publication=artifacts["linearity_ref"],
        ps_session_relative_path=recovery_session,
        ps_capture_metadata=_capture_metadata("c" * 32),
    )

    assert result["status"] == (
        "FAILED_POST_CAPTURE_VALIDATION_RECOVERY_RAW_PRESERVED"
    )
    assert result["ps_backend_calls"] == 1
    assert result["ps_raw_written"] is False
    assert result["automatic_retry_allowed"] is False
    partial_path = artifacts["root"] / result["partial_raw_publication"]["path"]
    assert partial_path.is_file()
    with np.load(partial_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        assert np.array_equal(archive["captured_pcm"], artifacts["ps_input"])
        assert np.array_equal(
            archive["actual_submitted_pcm"],
            artifacts["ps_telemetry"]["actual_submitted_pcm"],
        )
        assert metadata["partial_capture_never_promotable"] is True
        assert metadata["automatic_retry_allowed"] is False
        assert metadata["plant_identification_eligible"] is False


def test_ps_transport_failure_publishes_partial_raw_never_promotable(
    tmp_path: Path,
) -> None:
    plan, full = build_stage2_v2_live_safe_fallback_plan()
    boundary = int(plan["live_phase_contract"]["diagnostic_phase_stop_frame"])
    submitted = np.asarray(full[boundary:])
    captured = np.zeros((PRE_ROLL_FRAMES, 2), dtype="<i4")
    telemetry = _success_telemetry(
        submitted,
        np.zeros(
            (len(submitted) + PRE_ROLL_FRAMES + POST_ROLL_FRAMES, 2),
            dtype="<i4",
        ),
    )
    telemetry["completed"] = False
    telemetry["normal_stop_completed"] = False
    telemetry["failure_events"] = [{"role": "input", "message": "fixture"}]
    telemetry["captured_input_frames"] = len(captured)
    telemetry["capture_valid_mask"] = np.ones(len(captured), dtype="bool")
    input_blocks = len(captured) // 256
    for name in (
        "input_callback_sequence",
        "input_callback_start_frames",
        "input_callback_frame_counts",
        "input_buffer_adc_time",
        "input_callback_current_time",
        "input_callback_status_bitmask",
    ):
        telemetry[name] = np.asarray(telemetry[name])[:input_blocks]
    actual = np.zeros_like(submitted)
    submitted_mask = np.zeros(len(submitted), dtype="bool")
    failure = OutputMasterCaptureFailure(
        "fixture failure",
        captured,
        actual,
        np.ones(len(captured), dtype="bool"),
        submitted_mask,
        telemetry,
    )
    session = "results/stage2_2khz_output_master_ps_v3/partial_fixture"
    publication = publish_stage2_output_master_ps_v3_partial_raw_no_replace(
        str(tmp_path),
        session,
        plan,
        full,
        failure=failure,
        capture_metadata=_capture_metadata("f" * 32),
        diagnostic_session_relative_path=(
            "results/stage2_2khz_output_master_diagnostic/fixture"
        ),
        diagnostic_clock_publication={
            "path": "results/stage2_2khz_output_master_diagnostic/fixture/clock_receipt.json",
            "sha256": "4" * 64,
        },
        diagnostic_linearity_publication={
            "path": "results/stage2_2khz_output_master_diagnostic/fixture/linearity_v3.json",
            "sha256": "5" * 64,
        },
    )
    with np.load(tmp_path / publication["path"], allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        assert metadata["partial_capture_never_promotable"] is True
        assert metadata["automatic_retry_allowed"] is False
        assert metadata["plant_identification_eligible"] is False
        assert metadata["canonical_training_eligible"] is False


def test_structural_adapter_preserves_unequal_lengths_without_plant_authority() -> None:
    plan, full = build_stage2_v2_live_safe_fallback_plan()
    _v3, submitted = build_stage2_output_master_ps_v3_plan(plan, full)
    captured = np.zeros(
        (len(submitted) + PRE_ROLL_FRAMES + POST_ROLL_FRAMES, 2), dtype="<i4"
    )
    receipt = inspect_stage2_output_master_ps_v3_raw_structure(
        plan, full, captured, _success_telemetry(submitted, captured)
    )

    assert receipt["schema"] == PS_V3_RAW_STRUCTURE_SCHEMA
    assert receipt["output_clock_frames"] == len(submitted)
    assert receipt["input_clock_frames"] == len(captured)
    assert receipt["input_output_length_difference_frames"] == 12_288
    assert receipt["input_output_frame_identity_claimed"] is False
    assert receipt["ps_clock_authority_granted"] is False
    assert receipt["plant_identification_eligible"] is False


def test_missing_or_even_passed_diagnostic_clock_never_calls_ps_backend(
    durable_diagnostic,
) -> None:
    root, session, plan, full, _raw, publication = durable_diagnostic
    calls: list[str] = []

    def forbidden_capture(**_kwargs):
        calls.append("capture")
        raise AssertionError("P/S backend를 호출하면 안 됩니다")

    missing = run_stage2_output_master_ps_v3_if_admitted(
        plan, full, capture_callable=forbidden_capture
    )
    assert missing["ps_backend_calls"] == 0
    assert missing["admission"]["blockers"] == [
        "TRACKED_CLI_REPOSITORY_AND_PHYSICAL_PREOPEN_AUTHORITY_REQUIRED"
    ]

    passed_clock = run_stage2_output_master_ps_v3_if_admitted(
        plan,
        full,
        capture_callable=forbidden_capture,
        repository_root=str(root),
        diagnostic_session_relative_path=session,
        diagnostic_clock_publication=publication,
    )
    assert passed_clock["ps_backend_calls"] == 0
    assert passed_clock["capture_callable_invoked"] is False
    assert calls == []


def test_durable_clock_validation_reopens_and_cross_binds_actual_raw(
    durable_diagnostic,
) -> None:
    root, session, plan, full, raw, publication = durable_diagnostic
    loaded = validate_output_master_diagnostic_clock_publication(
        str(root), session, publication, plan, full
    )

    assert loaded["receipt"]["raw_artifact"] == {
        "path": raw["path"],
        "sha256": raw["sha256"],
    }
    assert loaded["receipt"]["passed"] is True
    assert loaded["receipt"]["ps_phase_may_start"] is False
    assert loaded["receipt"]["canonical_training_eligible"] is False


def test_self_attested_pass_json_is_rejected_before_raw_use(tmp_path: Path) -> None:
    plan, full = build_stage2_v2_live_safe_fallback_plan()
    session = "results/stage2_2khz_output_master_diagnostic/forged_session"
    target = output_master_session_targets(session)["clock_receipt"]
    path = tmp_path / target
    path.parent.mkdir(parents=True)
    payload = b'{"passed":true}\n'
    path.write_bytes(payload)
    publication = {"path": target, "sha256": hashlib.sha256(payload).hexdigest()}

    with pytest.raises(Exception, match="schema|canonical"):
        validate_output_master_diagnostic_clock_publication(
            str(tmp_path),
            session,
            publication,
            plan,
            full,
        )


def test_public_linearity_analyzer_rejects_caller_supplied_durable_mapping() -> None:
    plan, full = build_stage2_v2_live_safe_fallback_plan()
    forged = {"receipt": {"passed": True}, "raw": {"captured_pcm": np.zeros((1, 2))}}

    with pytest.raises(TypeError):
        analyse_stage2_output_master_diagnostic_linearity_v3(plan, full, forged)


@pytest.mark.parametrize(
    "field,delta",
    [
        ("input_frame_cursor_at_output_start", 256),
        ("input_frame_cursor_at_output_complete", -256),
        ("pre_roll_observed_input_frames", 256),
        ("post_roll_observed_input_frames", 256),
    ],
)
def test_output_master_marker_must_match_raw_callback_axis(field: str, delta: int) -> None:
    plan, full = build_stage2_v2_live_safe_fallback_plan()
    boundary = int(plan["live_phase_contract"]["diagnostic_phase_stop_frame"])
    submitted = np.asarray(full[:boundary])
    captured = np.zeros(
        (len(submitted) + PRE_ROLL_FRAMES + POST_ROLL_FRAMES, 2), dtype="<i4"
    )
    telemetry = _success_telemetry(submitted, captured)
    telemetry[field] += delta

    with pytest.raises(Exception, match="marker"):
        validate_output_master_success_telemetry(
            telemetry,
            captured_pcm=captured,
            expected_submitted_pcm=submitted,
        )


@pytest.mark.parametrize("mutation", ["hardware_sha", "devices", "meter_sha", "git_head"])
def test_diagnostic_to_ps_physical_continuity_rejects_changed_route_or_level(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    monkeypatch.setattr(
        ps_v3_module, "_reopen_bound_meter", _fake_reopen_bound_meter
    )
    diagnostic = _capture_metadata("e" * 32)
    expected = ps_v3_module._validate_physical_capture_metadata(diagnostic)
    observed = json.loads(json.dumps(diagnostic))
    if mutation == "hardware_sha":
        observed["hardware_config_sha256"] = "9" * 64
    elif mutation == "devices":
        observed["resolved_devices"]["output"] = 25
        observed["fresh_meter"]["resolved_devices"]["output"] = 25
    elif mutation == "meter_sha":
        observed["fresh_meter"]["sha256"] = "8" * 64
    else:
        observed["measurement_git_authority"]["head"] = "f" * 40
        observed["measurement_git_authority"]["origin_dev"] = "f" * 40
        observed["repository_execution_identity"]["repository_commit"] = "f" * 40

    with pytest.raises(Exception, match="continuity"):
        ps_v3_module._physical_continuity_binding(
            "/unused",
            expected,
            observed,
            require_fresh_meter=True,
        )


def test_cli_default_is_dry_run_and_never_imports_audio_backend(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    path = ROOT / "scripts/data/measure_paths_stage2_2khz_v3.py"
    spec = importlib.util.spec_from_file_location("stage2_ps_v3_cli_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.main([]) == 0
    output = capsys.readouterr().out
    assert "무음 dry-run" in output
    assert "legacy combined authority=false" in output
    assert "sounddevice import/open=0; P/S output=0; raw write=0" in output


def test_cli_execute_live_is_blocked_after_clean_identity_without_backend(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    path = ROOT / "scripts/data/measure_paths_stage2_2khz_v3.py"
    spec = importlib.util.spec_from_file_location("stage2_ps_v3_live_cli_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "repository_execution_identity",
        lambda *_args: {
            "repository_commit": "a" * 40,
            "repository_branch": "dev",
            "repository_dirty": False,
            "script_path": module.ADAPTER_PATH,
            "script_file_sha256": "b" * 64,
        },
    )

    imports: list[str] = []
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: imports.append(name),
    )
    assert module.main(["--execute-live"]) == 2
    captured = capsys.readouterr()
    assert "BLOCKED_BEFORE_AUDIO" in captured.err
    assert imports == []


@pytest.mark.parametrize(
    "identity_branch,identity_commit,authority_head",
    [
        ("main", "a" * 40, "a" * 40),
        ("dev", "b" * 40, "a" * 40),
    ],
)
def test_cli_git_authority_rejects_wrong_branch_or_clean_unpushed_head(
    identity_branch: str, identity_commit: str, authority_head: str
) -> None:
    path = ROOT / "scripts/data/measure_paths_stage2_2khz_v3.py"
    spec = importlib.util.spec_from_file_location("stage2_ps_v3_git_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    identity = {
        "repository_branch": identity_branch,
        "repository_commit": identity_commit,
        "repository_dirty": False,
    }
    authority = {
        "branch": "dev",
        "head": authority_head,
        "origin_dev": authority_head,
    }

    with pytest.raises(ValueError, match="origin/dev"):
        module._validate_v3_git_authority(identity, authority)


def test_legacy_combined_cli_is_explicitly_disabled_before_live_import() -> None:
    path = ROOT / "scripts/data/measure_paths_stage2_2khz.py"
    spec = importlib.util.spec_from_file_location("stage2_legacy_combined_block", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.LEGACY_COMBINED_LIVE_AUTHORITY_DISABLED is True
    assert module.main(["--execute-live"]) == 2
    estimate_stage2_output_master_ps_clock_and_resample,
