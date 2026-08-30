from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import deep_anc.realtime.physical_clock_witness as witness
from deep_anc.audio_io import float32_to_pcm_int16
from deep_anc.dsp import fullband_causal_v4 as causal_v4
from deep_anc.realtime.clock_telemetry import payload_sha256, sha256_file


def _time_domain(callback_count: int) -> dict[str, object]:
    return {
        "finite_count": callback_count,
        "missing_or_nonfinite_count": 0,
        "strict_monotonic_violation_count": 0,
        "frame_step_violation_count": 0,
        "maximum_absolute_frame_step_error_samples": 0.0,
    }


def _write_case(
    root: Path,
    *,
    control_leak: bool = False,
    fallback_silence_blocks: int = 0,
    include_time_domains: bool = True,
    maximum_input_backlog_samples: int = causal_v4.BLOCK,
    maximum_output_backlog_samples: int = causal_v4.BLOCK,
) -> tuple[Path, Path, Path]:
    session = root / "session.npz"
    clock = root / "session.runtime_clock.json"
    plan_path = root / "physical_witness_plan.json"
    start = causal_v4.PERIOD
    plan = witness.build_runtime_physical_witness_plan(
        session_npz_target=session,
        clock_receipt_target=clock,
        hardware_fingerprint_sha256="a" * 64,
        analysis_start_sample=start,
        synthetic_fixture=True,
    )
    witness.write_runtime_physical_witness_plan_exclusive(plan_path, plan)

    # One full-period guard on both sides lets the bounded +/-1000 ppm search
    # interpolate without using an edge.  Every structural fixture is marked
    # synthetic in the predeclared no-replace plan.
    frames = int(plan["analysis_stop_sample"]) + causal_v4.PERIOD
    period = causal_v4.continuous_pilot_period()[:, 0]
    source_pcm = np.tile(
        period, int(np.ceil(frames / causal_v4.PERIOD))
    )[:frames].astype(np.int16, copy=False)
    source = source_pcm.astype(np.float32) / np.float32(32767.0)
    control = np.zeros(frames, dtype=np.float32)
    if control_leak:
        control[:] = source
    # Cyclic fixed-LTI fixtures avoid an onset transient.  They test only the
    # estimator contract, never physical/canonical authority.
    ref = np.roll(source, 100).astype(np.float32) * np.float32(0.5)
    err = np.roll(source, 200).astype(np.float32) * np.float32(0.4)
    anc_gain = np.ones(frames, dtype=np.float32)

    callback_starts = np.arange(0, frames, causal_v4.BLOCK, dtype=np.int64)
    callbacks = []
    for frame in callback_starts:
        seconds = float(frame / causal_v4.FS)
        callbacks.append(
            {
                "callback_start_frame": int(frame),
                "callback_frame_count": causal_v4.BLOCK,
                "input_buffer_adc_time": 10.0 + seconds,
                "output_buffer_dac_time": 10.1 + seconds,
                "callback_current_time": 10.2 + seconds,
                "completed": True,
            }
        )
    count = len(callbacks)
    telemetry = {
        "schema_version": "realtime_clock_telemetry_v1",
        "authority_status": "INCONCLUSIVE",
        "structural_status": "PASS",
        "input_device": "APE:1 synthetic fixture",
        "output_device": "Audio:0 synthetic fixture",
        "clock_semantics": {
            "noise_and_cancel_outputs_share_one_output_stream_device_clock": True,
            "adc_dac_drift_is_not_noise_cancel_relative_output_phase": True,
            "callback_frame_counter_is_application_observed_not_physical_adc_proof": True,
        },
        "callback_summary": {
            "callback_count": count,
            "completed_callback_count": count,
            "incomplete_callback_count": 0,
            "pending_callback_count": 0,
            "portaudio_status_callback_count": 0,
            "callback_host_deadline_miss_count": 0,
            "omitted_callback_record_count": 0,
            "stored_callback_record_count": count,
            "application_observed_frames": frames,
            "application_observed_seconds": frames / causal_v4.FS,
        },
        "runtime_counters_final": {
            "xrun_count": 0,
            "deadline_miss_count": 0,
            "input_ring_drop_samples": 0,
            "output_ring_drop_samples": 0,
            "input_ring_overrun_blocks": 0,
            "output_ring_overrun_blocks": 0,
            "input_ring_underrun_blocks": 0,
            "output_ring_underrun_blocks": 0,
            "ring_add_samples": 0,
            "fallback_silence_blocks": fallback_silence_blocks,
            "watchdog_trip_counts": {},
        },
        "maximum_input_backlog_samples": maximum_input_backlog_samples,
        "maximum_output_backlog_samples": maximum_output_backlog_samples,
        "allowed_input_backlog_samples": causal_v4.BLOCK,
        "allowed_output_backlog_samples": causal_v4.BLOCK,
        "issue_counts": {},
        "callbacks": callbacks,
    }
    if include_time_domains:
        telemetry["time_domains"] = {
            "input_buffer_adc_time": _time_domain(count),
            "output_buffer_dac_time": _time_domain(count),
            "callback_current_time": _time_domain(count),
        }
    telemetry_sha = payload_sha256(telemetry)
    with session.open("xb") as handle:
        np.savez_compressed(
            handle,
            fs=causal_v4.FS,
            runtime_clock_telemetry_sha256=np.asarray(telemetry_sha),
            runtime_clock_authority_status=np.asarray("INCONCLUSIVE"),
            source=source,
            control=control,
            ref=ref,
            err=err,
            anc_gain=anc_gain,
        )
    bundle = {
        "schema_version": "realtime_clock_receipt_bundle_v1",
        "authority_status": "INCONCLUSIVE",
        "runtime_clock_telemetry_sha256": telemetry_sha,
        "recording_npz": str(session.resolve()),
        "recording_npz_sha256": sha256_file(session),
        "runtime_clock_telemetry": telemetry,
    }
    clock.write_text(
        json.dumps(
            bundle,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return plan_path, session, clock


@pytest.fixture(scope="module")
def valid_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("runtime-physical-witness")
    paths = _write_case(root)
    evidence = witness.audit_runtime_physical_clock_witness(
        plan_path=paths[0], session_npz_path=paths[1], clock_receipt_path=paths[2]
    )
    return paths, evidence


def test_predeclared_plan_separates_clock_pilot_from_control_frequency_claims(tmp_path):
    plan = witness.build_runtime_physical_witness_plan(
        session_npz_target=tmp_path / "session.npz",
        clock_receipt_target=tmp_path / "session.runtime_clock.json",
        hardware_fingerprint_sha256="b" * 64,
        analysis_start_sample=causal_v4.PERIOD,
        synthetic_fixture=True,
    )
    assert plan["period_count"] == witness.MINIMUM_PERIOD_COUNT == 44
    assert plan["observed_seconds"] >= 30.0
    assert max(plan["continuous_reserved_pilot"]["pilot_frequencies_hz"]) <= 600.0
    scope = plan["authority_scope"]
    assert scope["pilot_band_is_clock_witness_not_control_or_evaluation_band"]
    assert scope["control_attenuation_assessed"] is False
    assert scope["octave_125_hz_band_hz"] == pytest.approx(
        [125.0 / np.sqrt(2.0), 125.0 * np.sqrt(2.0)]
    )
    assert scope["octave_125_hz_fully_covered_by_pilot"] is False
    assert scope["point_control_union_150_11314_claimed_by_witness"] is False
    with pytest.raises(ValueError, match="synthetic_fixture"):
        witness.build_runtime_physical_witness_plan(
            session_npz_target=tmp_path / "other.npz",
            clock_receipt_target=tmp_path / "other.json",
            hardware_fingerprint_sha256="b" * 64,
            analysis_start_sample=causal_v4.PERIOD,
            synthetic_fixture=np.bool_(True),
        )
    with pytest.raises(ValueError, match="30"):
        witness.build_runtime_physical_witness_plan(
            session_npz_target=tmp_path / "short.npz",
            clock_receipt_target=tmp_path / "short.json",
            hardware_fingerprint_sha256="b" * 64,
            analysis_start_sample=causal_v4.PERIOD,
            synthetic_fixture=True,
            period_count=43,
        )


def test_valid_synthetic_clock_fit_never_becomes_physical_pass(valid_fixture, tmp_path):
    _, evidence = valid_fixture
    assert evidence["status"] == "FIXTURE_ONLY_PASS"
    assert evidence["synthetic_fixture"] is True
    assert evidence["conditional_physical_timing_pass"] is False
    assert evidence["independent_clock_authority_pass"] is False
    assert evidence["canonical_runtime_pass"] is False
    assert evidence["deployment_eligible"] is False
    assert evidence["clock_telemetry_authority_remains"] == "INCONCLUSIVE"
    assert evidence["highband_target_or_attenuation_used_for_clock_fit"] is False
    assert evidence["octave_125_hz_fully_assessed"] is False
    assert evidence["point_control_union_150_11314_assessed"] is False
    assert evidence["observed_seconds"] >= 30.0
    fit = evidence["clock_fit"]
    assert fit["combined_max_samples"] <= causal_v4.CLOCK_COMBINED_MAX
    assert fit["hard_20db_11314hz_max_samples"] == pytest.approx(
        0.06755189029558946
    )
    assert fit["highband_target_or_attenuation_used_for_clock_fit"] is False
    assert fit["segment_stationarity"]["change_point_count"] == 0
    assert fit["segment_stationarity"]["sample_slip_count"] == 0
    backlog = evidence["ring_backlog_witness"]
    assert backlog["maximum_input_backlog_samples"] == causal_v4.BLOCK
    assert backlog["maximum_output_backlog_samples"] == causal_v4.BLOCK
    assert backlog["maximum_excess_input_backlog_samples"] == 0
    assert backlog["maximum_excess_output_backlog_samples"] == 0
    assert len(evidence["source_submitted_pcm_sha256"]) == 64
    assert len(evidence["stereo_submitted_pcm_sha256"]) == 64

    receipt = tmp_path / "witness.json"
    path, digest = witness.write_runtime_physical_witness_receipt_exclusive(
        receipt, evidence
    )
    assert path == receipt and len(digest) == 64
    with pytest.raises(FileExistsError):
        witness.write_runtime_physical_witness_receipt_exclusive(receipt, evidence)


def test_receipt_writer_rejects_synthetic_physical_status(valid_fixture, tmp_path):
    _, evidence = valid_fixture
    forged = dict(evidence)
    forged["status"] = "CONDITIONAL_PASS"
    forged["conditional_physical_timing_pass"] = True
    forged.pop("evidence_sha256")
    forged["evidence_sha256"] = witness._json_sha256(forged)
    with pytest.raises(ValueError, match="synthetic fixture"):
        witness.write_runtime_physical_witness_receipt_exclusive(
            tmp_path / "forged.json", forged
        )


def test_control_pilot_line_leakage_blocks_before_clock_fit(tmp_path):
    paths = _write_case(tmp_path, control_leak=True)
    evidence = witness.audit_runtime_physical_clock_witness(
        plan_path=paths[0], session_npz_path=paths[1], clock_receipt_path=paths[2]
    )
    assert evidence["status"] == "BLOCKED"
    assert "control PCM" in evidence["blockers"][0]


@pytest.mark.parametrize(
    "updates, expected",
    [
        ({"fallback_silence_blocks": 1}, "fallback_silence_blocks"),
        ({"include_time_domains": False}, "time raw summary"),
        (
            {"maximum_output_backlog_samples": causal_v4.BLOCK + 1},
            "maximum excess backlog",
        ),
    ],
)
def test_runtime_counter_or_timestamp_evidence_cannot_be_omitted(
    tmp_path, updates, expected
):
    paths = _write_case(tmp_path, **updates)
    evidence = witness.audit_runtime_physical_clock_witness(
        plan_path=paths[0], session_npz_path=paths[1], clock_receipt_path=paths[2]
    )
    assert evidence["status"] == "BLOCKED"
    assert expected in evidence["blockers"][0]


def test_v4_path_subset_uses_only_actual_requested_source_path():
    period = causal_v4.continuous_pilot_period()
    submitted = np.column_stack(
        (period[:, 0], np.zeros(causal_v4.PERIOD, dtype=np.int16))
    )
    plan = {
        "clock_rows": [
            {
                "name": "fit",
                "start_frame": 0,
                "stop_frame": causal_v4.PERIOD,
                "purpose": "fit",
            }
        ]
    }
    signal = np.roll(submitted[:, 0].astype(np.float64), 10)
    bank = causal_v4._transfer_bank(
        plan=plan,
        submitted=submitted,
        signals={"ref": signal},
        rate_ratio=1.0,
        method="linear",
        purposes=("fit",),
        paths=("primary",),
    )
    assert set(bank) == {("ref", "primary", "fit")}
    assert float(np.min(np.abs(bank[("ref", "primary", "fit")][0]))) > 0.0
    # Default remains the P/S measurement behavior and therefore rejects a
    # missing actual secondary denominator.  No fake secondary is inserted.
    with pytest.raises(ValueError, match="denominator"):
        causal_v4._transfer_bank(
            plan=plan,
            submitted=submitted,
            signals={"ref": signal},
            rate_ratio=1.0,
            method="linear",
            purposes=("fit",),
        )


def test_runtime_pcm_sha_is_exact_converter_output(valid_fixture):
    paths, evidence = valid_fixture
    with np.load(paths[1], allow_pickle=False) as archive:
        reconstructed = float32_to_pcm_int16(archive["source"])
    assert witness._array_sha256(reconstructed) == evidence[
        "source_submitted_pcm_sha256"
    ]


@pytest.mark.parametrize("maximum_backlog", [0, causal_v4.BLOCK])
def test_ring_backlog_scheduler_race_allows_zero_or_one_hop(maximum_backlog):
    receipt = witness._validate_ring_backlog(
        {
            "maximum_input_backlog_samples": maximum_backlog,
            "maximum_output_backlog_samples": maximum_backlog,
            "allowed_input_backlog_samples": causal_v4.BLOCK,
            "allowed_output_backlog_samples": causal_v4.BLOCK,
        }
    )
    assert receipt["maximum_excess_input_backlog_samples"] == 0
    assert receipt["maximum_excess_output_backlog_samples"] == 0


def test_ring_backlog_allowed_plus_one_is_blocked():
    with pytest.raises(ValueError, match="maximum excess backlog"):
        witness._validate_ring_backlog(
            {
                "maximum_input_backlog_samples": causal_v4.BLOCK,
                "maximum_output_backlog_samples": causal_v4.BLOCK + 1,
                "allowed_input_backlog_samples": causal_v4.BLOCK,
                "allowed_output_backlog_samples": causal_v4.BLOCK,
            }
        )
