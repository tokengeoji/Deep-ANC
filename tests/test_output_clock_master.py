from __future__ import annotations

import hashlib

import numpy as np
import pytest
from pydantic import ValidationError

from deep_anc.audio_io import float32_to_pcm_int16
from deep_anc.dsp.control_band_contract import BroadbandFullOctaveContractV3
from deep_anc.dsp.timing import TrainingTimingContract
from deep_anc.realtime.output_clock_master import (
    CanonicalErrZeroReceipt,
    OutputClockMasterAdmission,
    OutputClockMasterBlocked,
    OutputClockMasterScheduler,
    OutputDiscontinuityCounters,
    RefOnlyModelInputContract,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _timing() -> TrainingTimingContract:
    # raw lead = S(0) + handoff(256) - P pre-FIR zeros(140) = 116.
    return TrainingTimingContract(
        primary_zeros_before_fir_samples=140,
        primary_fir_peak_offset_samples=4,
        primary_effective_delay_samples=144,
        secondary_delay_samples=0,
        handoff_samples=256,
        sample_rate=48_000,
        raw_digital_reference_lead_samples=116,
        digital_reference_lead_samples=116,
        synthetic_total_advance_samples=260,
    )


def _err_zero_receipt() -> CanonicalErrZeroReceipt:
    return CanonicalErrZeroReceipt(
        canonical_population_sha256=_sha("canonical population"),
        item_receipts_sha256=_sha("item receipts"),
        item_count=17,
        total_error_feature_samples=17 * 256,
    )


def _admission(
    *,
    error_dropout_probability: float = 1.0,
    include_err_zero_receipt: bool = False,
    equivalence_error: float = 5.0e-6,
) -> OutputClockMasterAdmission:
    v3 = BroadbandFullOctaveContractV3.canonical()
    timing = _timing()
    mode = RefOnlyModelInputContract(
        error_dropout_probability=error_dropout_probability
    )
    receipt = _err_zero_receipt() if include_err_zero_receipt else None
    return OutputClockMasterAdmission(
        digital_reference_lead_samples=timing.digital_reference_lead_samples,
        control_band_contract=v3,
        control_band_contract_sha256=v3.digest(),
        training_timing_contract=timing,
        training_timing_contract_sha256=timing.digest(),
        experiment_contract_sha256=_sha("experiment"),
        checkpoint_sha256=_sha("checkpoint"),
        deployment_artifact_sha256=_sha("deployment"),
        model_input_contract=mode,
        model_input_mode_sha256=mode.digest(),
        canonical_train_err_zero_receipt=receipt,
        canonical_train_err_zero_receipt_sha256=(
            None if receipt is None else receipt.digest()
        ),
        ref_only_ablation_receipt_sha256=_sha("ablation"),
        ref_only_g0_receipt_sha256=_sha("g0"),
        ref_only_validation_receipt_sha256=_sha("validation"),
        offline_streaming_equivalence_receipt_sha256=_sha("equivalence"),
        offline_streaming_max_abs_error=equivalence_error,
    )


def _source(offset: float) -> np.ndarray:
    return np.linspace(-0.02 + offset, 0.02 + offset, 256, dtype=np.float32)


def _complete_job(
    scheduler: OutputClockMasterScheduler,
    *,
    control_value: float,
    target_delta: int = 0,
    error_feature: np.ndarray | None = None,
) -> tuple[object, np.ndarray]:
    job = scheduler.claim_inference_job()
    assert job is not None
    control = np.full(256, control_value, dtype=np.float32)
    scheduler.submit_inference_result(
        job_id=job.job_id,
        source_callback_index=job.source_callback_index,
        target_output_callback_index=job.target_output_callback_index + target_delta,
        reference_used=job.reference,
        error_feature_used=(
            job.error_feature if error_feature is None else error_feature
        ),
        control=control,
    )
    return job, control


def test_admission_binds_exact_v3_timing_model_input_and_receipts() -> None:
    admission = _admission()

    assert admission.control_band_contract == BroadbandFullOctaveContractV3.canonical()
    assert admission.control_band_contract_sha256 == admission.control_band_contract.digest()
    assert admission.training_timing_contract_sha256 == _timing().digest()
    assert admission.model_input_contract.reference_dropout_probability == 0.0
    assert admission.model_input_contract.error_dropout_probability == 1.0
    assert admission.handoff_samples == 256
    assert admission.physical_performance_pass is False
    assert admission.deployment_eligible is False
    assert len(admission.digest()) == 64


def test_admission_rejects_wrong_v3_or_timing_digest_and_equivalence() -> None:
    payload = _admission().model_dump(mode="python")
    payload["control_band_contract_sha256"] = _sha("wrong-v3")
    with pytest.raises(ValidationError, match="control_band_contract_sha256"):
        OutputClockMasterAdmission.model_validate(payload)

    payload = _admission().model_dump(mode="python")
    payload["training_timing_contract_sha256"] = _sha("wrong-timing")
    with pytest.raises(ValidationError, match="training_timing_contract_sha256"):
        OutputClockMasterAdmission.model_validate(payload)

    with pytest.raises(ValidationError, match="offline-streaming equivalence"):
        _admission(equivalence_error=1.0001e-5)


def test_admission_err_dropout_branch_is_fail_closed() -> None:
    with pytest.raises(ValidationError, match="ERR exact-zero receipt"):
        _admission(error_dropout_probability=0.5)

    admitted = _admission(
        error_dropout_probability=0.5,
        include_err_zero_receipt=True,
    )
    assert admitted.canonical_train_err_zero_receipt is not None
    assert admitted.canonical_train_err_zero_receipt.nonzero_error_feature_sample_count == 0


def test_output_clock_timeline_prime_then_exact_k_plus_one_control() -> None:
    scheduler = OutputClockMasterScheduler(_admission())
    scheduler.request_anc_on()
    u0 = _source(0.0)

    prime = scheduler.output_callback(
        callback_index=0,
        global_output_frame_start=0,
        future_source=u0,
        anc_gain=0.0,
    )
    assert prime.receipt.state == "startup_prime"
    assert prime.receipt.startup_prime is True
    assert prime.receipt.protocol_prime is True
    assert prime.receipt.performance_window_included is False
    assert prime.receipt.inference_ran_in_output_callback is False
    assert np.all(prime.stereo_pcm_s16[:, 1] == 0)
    # lead 116: 실제 NS는 첫 116 samples가 silence이고 이후 U0가 시작된다.
    assert np.all(prime.stereo_pcm_s16[:116, 0] == 0)

    job0, y0 = _complete_job(scheduler, control_value=0.01)
    assert job0.source_callback_index == 0
    assert job0.target_output_callback_index == 1
    assert job0.source_frame_start == 0
    assert job0.target_output_frame_start == 256
    assert job0.reference.flags.writeable is False
    assert np.array_equal(job0.reference, u0)
    assert np.count_nonzero(job0.error_feature) == 0

    u1 = _source(0.001)
    active = scheduler.output_callback(
        callback_index=1,
        global_output_frame_start=256,
        future_source=u1,
        anc_gain=np.ones(256, dtype=np.float32),
    )
    assert active.receipt.state == "anc_on"
    assert active.receipt.performance_window_included is True
    assert active.receipt.control_source_callback_index == 0
    assert active.receipt.control_reference_frame_start == 0
    assert active.receipt.generated_source_frame_start == 256
    assert active.receipt.reference_frame_start == 256
    assert active.receipt.playback_source_frame_start == 256 - 116
    assert active.receipt.anc_gain_frame_start == 256
    assert active.receipt.actual_control_output_frame_start == 256
    assert np.array_equal(active.stereo_pcm_s16[:, 1], float32_to_pcm_int16(y0))
    assert active.stereo_pcm_s16.flags.writeable is False

    receipt = scheduler.close_evidence_window(
        # callback race에 따라 absolute backlog 0/256은 모두 정상이다. excess만 0 고정.
        discontinuity_counters=OutputDiscontinuityCounters(
            maximum_absolute_backlog_samples=256
        )
    )
    assert receipt.performance_output_block_count == 1
    assert receipt.terminal_tail_target_callback_index == 2
    assert receipt.physical_performance_pass is False
    assert receipt.run_realtime_integrated is False
    assert len(receipt.digest()) == 64


def test_actual_float_to_s16_and_stereo_payload_shas_are_bound() -> None:
    scheduler = OutputClockMasterScheduler(_admission())
    source = _source(0.0)
    block = scheduler.output_callback(
        callback_index=0,
        global_output_frame_start=0,
        future_source=source,
        anc_gain=0.0,
    )

    expected_source_pcm = float32_to_pcm_int16(source)
    assert block.receipt.generated_source_pcm_s16_sha256 == hashlib.sha256(
        expected_source_pcm.tobytes()
    ).hexdigest()
    assert block.receipt.submitted_stereo_pcm_s16_sha256 == hashlib.sha256(
        block.stereo_pcm_s16.tobytes()
    ).hexdigest()
    assert block.receipt.anc_gain_min == 0.0
    assert block.receipt.anc_gain_max == 0.0


@pytest.mark.parametrize("target_delta", [-1, 1])
def test_one_block_early_or_late_result_permanently_blocks(target_delta: int) -> None:
    scheduler = OutputClockMasterScheduler(_admission())
    scheduler.request_anc_on()
    scheduler.output_callback(
        callback_index=0,
        global_output_frame_start=0,
        future_source=_source(0.0),
        anc_gain=0.0,
    )

    with pytest.raises(OutputClockMasterBlocked, match=r"정확히 k\+1"):
        _complete_job(
            scheduler,
            control_value=0.0,
            target_delta=target_delta,
        )
    assert scheduler.blocked_reason is not None
    with pytest.raises(OutputClockMasterBlocked):
        scheduler.claim_inference_job()


def test_missing_y_does_not_emit_fallback_and_blocks() -> None:
    scheduler = OutputClockMasterScheduler(_admission())
    scheduler.request_anc_on()
    scheduler.output_callback(
        callback_index=0,
        global_output_frame_start=0,
        future_source=_source(0.0),
        anc_gain=0.0,
    )

    with pytest.raises(OutputClockMasterBlocked, match="underflow/late"):
        scheduler.output_callback(
            callback_index=1,
            global_output_frame_start=256,
            future_source=_source(0.001),
            anc_gain=1.0,
        )


def test_nonzero_err_feature_and_stale_job_are_fail_closed() -> None:
    scheduler = OutputClockMasterScheduler(_admission())
    scheduler.request_anc_on()
    scheduler.output_callback(
        callback_index=0,
        global_output_frame_start=0,
        future_source=_source(0.0),
        anc_gain=0.0,
    )
    nonzero_err = np.zeros(256, dtype=np.float32)
    nonzero_err[3] = np.float32(1.0e-7)
    with pytest.raises(OutputClockMasterBlocked, match="exact zero"):
        _complete_job(
            scheduler,
            control_value=0.0,
            error_feature=nonzero_err,
        )

    scheduler = OutputClockMasterScheduler(_admission())
    scheduler.request_anc_on()
    scheduler.output_callback(
        callback_index=0,
        global_output_frame_start=0,
        future_source=_source(0.0),
        anc_gain=0.0,
    )
    job, _ = _complete_job(scheduler, control_value=0.0)
    with pytest.raises(OutputClockMasterBlocked, match="stale/reused/unknown"):
        scheduler.submit_inference_result(
            job_id=job.job_id,
            source_callback_index=job.source_callback_index,
            target_output_callback_index=job.target_output_callback_index,
            reference_used=job.reference,
            error_feature_used=job.error_feature,
            control=np.zeros(256, dtype=np.float32),
        )


def test_reset_while_anc_requested_forces_new_zero_gain_prime() -> None:
    scheduler = OutputClockMasterScheduler(_admission())
    scheduler.request_anc_on()
    scheduler.output_callback(
        callback_index=0,
        global_output_frame_start=0,
        future_source=_source(0.0),
        anc_gain=0.0,
    )
    _complete_job(scheduler, control_value=0.01)
    scheduler.output_callback(
        callback_index=1,
        global_output_frame_start=256,
        future_source=_source(0.001),
        anc_gain=1.0,
    )

    # callback 1이 만든 아직 claim되지 않은 tail은 reset 경계에서 폐기되고, callback 2는
    # control을 절대로 재사용하지 않고 exact-zero prime이 된다.
    scheduler.request_reset()
    reset_prime = scheduler.output_callback(
        callback_index=2,
        global_output_frame_start=512,
        future_source=_source(0.002),
        anc_gain=0.0,
    )
    assert reset_prime.receipt.state == "reset_prime"
    assert reset_prime.receipt.prime_reason == "reset"
    assert reset_prime.receipt.protocol_prime is True
    assert reset_prime.receipt.performance_window_included is False
    assert np.all(reset_prime.stereo_pcm_s16[:, 1] == 0)

    _complete_job(scheduler, control_value=-0.01)
    active = scheduler.output_callback(
        callback_index=3,
        global_output_frame_start=768,
        future_source=_source(0.003),
        anc_gain=1.0,
    )
    assert active.receipt.state == "anc_on"
    assert active.receipt.epoch == 1
    receipt = scheduler.close_evidence_window(
        discontinuity_counters=OutputDiscontinuityCounters()
    )
    assert receipt.reset_count == 1
    assert receipt.performance_output_block_count == 2


def test_output_frame_slip_and_nonzero_runtime_counter_block_receipt() -> None:
    scheduler = OutputClockMasterScheduler(_admission())
    with pytest.raises(OutputClockMasterBlocked, match="sample slip"):
        scheduler.output_callback(
            callback_index=0,
            global_output_frame_start=1,
            future_source=_source(0.0),
            anc_gain=0.0,
        )

    scheduler = OutputClockMasterScheduler(_admission())
    scheduler.request_anc_on()
    scheduler.output_callback(
        callback_index=0,
        global_output_frame_start=0,
        future_source=_source(0.0),
        anc_gain=0.0,
    )
    _complete_job(scheduler, control_value=0.0)
    scheduler.output_callback(
        callback_index=1,
        global_output_frame_start=256,
        future_source=_source(0.001),
        anc_gain=1.0,
    )
    with pytest.raises(OutputClockMasterBlocked, match="허용 계약"):
        scheduler.close_evidence_window(
            discontinuity_counters=OutputDiscontinuityCounters(
                deadline_miss_count=1
            )
        )

    scheduler = OutputClockMasterScheduler(_admission())
    scheduler.request_anc_on()
    scheduler.output_callback(
        callback_index=0,
        global_output_frame_start=0,
        future_source=_source(0.0),
        anc_gain=0.0,
    )
    _complete_job(scheduler, control_value=0.0)
    scheduler.output_callback(
        callback_index=1,
        global_output_frame_start=256,
        future_source=_source(0.001),
        anc_gain=1.0,
    )
    with pytest.raises(OutputClockMasterBlocked, match="excess-backlog"):
        scheduler.close_evidence_window(
            discontinuity_counters=OutputDiscontinuityCounters(
                maximum_absolute_backlog_samples=257,
                maximum_excess_backlog_samples=1,
            )
        )


def test_anc_off_to_on_cannot_skip_reprime() -> None:
    scheduler = OutputClockMasterScheduler(_admission())
    off = scheduler.output_callback(
        callback_index=0,
        global_output_frame_start=0,
        future_source=_source(0.0),
        anc_gain=0.0,
    )
    assert off.receipt.state == "anc_off"
    assert scheduler.claim_inference_job() is None

    scheduler.request_anc_on()
    prime = scheduler.output_callback(
        callback_index=1,
        global_output_frame_start=256,
        future_source=_source(0.001),
        anc_gain=0.0,
    )
    assert prime.receipt.state == "startup_prime"
    assert prime.receipt.performance_window_included is False
    with pytest.raises(OutputClockMasterBlocked, match="underflow/late"):
        scheduler.output_callback(
            callback_index=2,
            global_output_frame_start=512,
            future_source=_source(0.002),
            anc_gain=1.0,
        )


def test_completed_anc_off_to_on_transition_is_labeled_rearm_prime() -> None:
    scheduler = OutputClockMasterScheduler(_admission())
    scheduler.request_anc_on()
    scheduler.output_callback(
        callback_index=0,
        global_output_frame_start=0,
        future_source=_source(0.0),
        anc_gain=0.0,
    )
    _complete_job(scheduler, control_value=0.0)
    scheduler.output_callback(
        callback_index=1,
        global_output_frame_start=256,
        future_source=_source(0.001),
        anc_gain=1.0,
    )
    scheduler.request_anc_off()
    scheduler.output_callback(
        callback_index=2,
        global_output_frame_start=512,
        future_source=_source(0.002),
        anc_gain=0.0,
    )
    scheduler.request_anc_on()
    rearm = scheduler.output_callback(
        callback_index=3,
        global_output_frame_start=768,
        future_source=_source(0.003),
        anc_gain=0.0,
    )
    assert rearm.receipt.state == "rearm_prime"
    assert rearm.receipt.prime_reason == "anc_on_transition"
    assert rearm.receipt.performance_window_included is False
