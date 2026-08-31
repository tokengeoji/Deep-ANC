from __future__ import annotations

import pytest

from deep_anc.dsp.control_band_contract import ControlBandContract
from deep_anc.eval.broadband_runtime import (
    BroadbandRuntimeEvidence,
    audit_broadband_runtime_evidence,
)


def _evidence(**updates):
    contract = ControlBandContract.broadband_point_control()
    raw = {
        "control_band_contract_sha256": contract.digest(),
        "experiment_contract_sha256": "1" * 64,
        "training_timing_contract_sha256": "2" * 64,
        "checkpoint_sha256": "3" * 64,
        "deployment_artifact_sha256": "4" * 64,
        "deployment_metadata_sha256": "9" * 64,
        "runtime_session_sha256": "a" * 64,
        "runtime_log_sha256": "5" * 64,
        "primary_path_sha256": "6" * 64,
        "secondary_path_sha256": "7" * 64,
        "hardware_fingerprint_sha256": "8" * 64,
        "runtime_physical_witness_file_sha256": "b" * 64,
        "runtime_physical_witness_evidence_sha256": "c" * 64,
        "model_name": "hybrid_anc_tiny",
        "engine": "ort",
        "power_mode": "MAXN",
        "sample_rate": 48_000,
        "block_size": 256,
        "handoff_extra_samples": 256,
        "plant_lead_samples": 115,
        "checkpoint_lead_samples": 115,
        "deployment_lead_samples": 115,
        "runtime_lead_samples": 115,
        "observed_seconds": 60.0,
        "callback_count": 11_250,
        "inference_p50_ms": 1.1,
        "inference_p95_ms": 1.5,
        "inference_p99_ms": 1.8,
        "inference_max_ms": 2.4,
        "deadline_miss_count": 0,
        "engine_error_blocks": 0,
        "xrun_count": 0,
        "input_ring_drop_samples": 0,
        "output_ring_drop_samples": 0,
        "ring_add_samples": 0,
        "maximum_input_backlog_samples": 256,
        "maximum_output_backlog_samples": 256,
        "allowed_input_backlog_samples": 256,
        "allowed_output_backlog_samples": 256,
        "maximum_excess_input_backlog_samples": 0,
        "maximum_excess_output_backlog_samples": 0,
        "fallback_silence_blocks": 0,
        "watchdog_trip_count": 0,
        "sample_slip_count": 0,
        "conditional_physical_timing_pass": True,
        "independent_clock_authority_pass": True,
    }
    raw.update(updates)
    return BroadbandRuntimeEvidence.model_validate(raw)


@pytest.mark.parametrize("maximum_backlog", [0, 256])
def test_broadband_runtime_passes_with_normal_scheduler_backlog(maximum_backlog):
    contract = ControlBandContract.broadband_point_control()
    audit = audit_broadband_runtime_evidence(
        contract,
        _evidence(
            maximum_input_backlog_samples=maximum_backlog,
            maximum_output_backlog_samples=maximum_backlog,
        ),
        expected_plant_lead_samples=115,
    )
    assert audit.ok
    assert audit.block_deadline_ms == 1000.0 * 256 / 48_000
    assert audit.degrees_per_sample_8khz == 60.0


def test_one_sample_slip_blocks_high_frequency_runtime_claim():
    audit = audit_broadband_runtime_evidence(
        ControlBandContract.broadband_point_control(),
        _evidence(sample_slip_count=1),
        expected_plant_lead_samples=115,
    )
    assert not audit.ok
    assert any("sample slip=1" in reason for reason in audit.reasons)


def test_mismatched_checkpoint_or_runtime_lead_is_blocked():
    audit = audit_broadband_runtime_evidence(
        ControlBandContract.broadband_point_control(),
        _evidence(runtime_lead_samples=116),
        expected_plant_lead_samples=115,
    )
    assert not audit.ok
    assert any("lead" in reason for reason in audit.reasons)


def test_p99_margin_and_hard_deadline_are_separate_gates():
    contract = ControlBandContract.broadband_point_control()
    p99 = audit_broadband_runtime_evidence(
        contract,
        _evidence(inference_p99_ms=3.1, inference_max_ms=4.0),
        expected_plant_lead_samples=115,
    )
    assert not p99.ok
    assert any("P99" in reason for reason in p99.reasons)

    deadline = audit_broadband_runtime_evidence(
        contract,
        _evidence(inference_p99_ms=2.9, inference_max_ms=5.4),
        expected_plant_lead_samples=115,
    )
    assert not deadline.ok
    assert any("deadline" in reason for reason in deadline.reasons)


def test_allowed_plus_one_backlog_cannot_pass_despite_fast_inference():
    audit = audit_broadband_runtime_evidence(
        ControlBandContract.broadband_point_control(),
        _evidence(
            maximum_output_backlog_samples=257,
            maximum_excess_output_backlog_samples=1,
        ),
        expected_plant_lead_samples=115,
    )
    assert not audit.ok
    assert any("excess backlog" in reason for reason in audit.reasons)
    assert any("absolute backlog" in reason for reason in audit.reasons)


def test_declared_excess_backlog_must_be_exactly_derived():
    with pytest.raises(ValueError, match="exact하게 유도"):
        _evidence(
            maximum_output_backlog_samples=257,
            maximum_excess_output_backlog_samples=0,
        )


def test_fallback_stays_exact_zero_even_with_normal_backlog():
    audit = audit_broadband_runtime_evidence(
        ControlBandContract.broadband_point_control(),
        _evidence(fallback_silence_blocks=1),
        expected_plant_lead_samples=115,
    )
    assert not audit.ok
    assert any("fallback silence=1" in reason for reason in audit.reasons)


def test_engine_exception_silence_cannot_pass_as_fast_inference():
    audit = audit_broadband_runtime_evidence(
        ControlBandContract.broadband_point_control(),
        _evidence(engine_error_blocks=1),
        expected_plant_lead_samples=115,
    )
    assert not audit.ok
    assert any("engine error block=1" in reason for reason in audit.reasons)


def test_all_equal_but_wrong_lead_is_blocked_by_external_strict_plant():
    audit = audit_broadband_runtime_evidence(
        ControlBandContract.broadband_point_control(),
        _evidence(
            plant_lead_samples=109,
            checkpoint_lead_samples=109,
            deployment_lead_samples=109,
            runtime_lead_samples=109,
        ),
        expected_plant_lead_samples=115,
    )
    assert not audit.ok
    assert any("expected=115" in reason for reason in audit.reasons)


def test_conditional_acoustic_witness_without_independent_clock_cannot_pass():
    audit = audit_broadband_runtime_evidence(
        ControlBandContract.broadband_point_control(),
        _evidence(independent_clock_authority_pass=False),
        expected_plant_lead_samples=115,
    )
    assert not audit.ok
    assert any("electrical clock" in reason for reason in audit.reasons)
