from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
import torch

from deep_anc.dsp.control_band_contract import BroadbandFullOctaveContractV3
from deep_anc.dsp.timing import PlantDelays, TrainingTimingContract
from deep_anc.eval.full_octave_v3 import (
    FullOctaveV3MatchedSegment,
    evaluate_full_octave_v3_matched_segments,
)
from deep_anc.losses.broadband_loss import (
    BroadbandFullOctaveLossConfigV3,
    BroadbandFullOctaveLossPrimitiveV3,
    CausalFIRPathData,
)
from deep_anc.train.causal_secondary_prefix_adapter_v1 import (
    CausalPrefixBatchV1,
    CausalPrefixStateOriginV1,
    CausalSecondaryPrefixAdapterV1,
)
from deep_anc.train.full_octave_causal_plant_binding_v4 import (
    FULL_OCTAVE_CAUSAL_PLANT_AUTHORITY_SCHEMA_V4,
    FullOctaveCausalPlantBindingV4,
)
from deep_anc.train.full_octave_v3_consumers import (
    FullOctaveV3EvaluationIdentity,
    FullOctaveV3FxLMSConfig,
    FullOctaveV3MatchedFxLMSEvaluator,
    FullOctaveV3TrainerConsumer,
)


FS = 48_000


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _operator(
    *,
    role: str,
    fir: tuple[float, ...],
    coarse_delay: int,
    handoff: int,
    authority_sha: str,
) -> CausalFIRPathData:
    values = np.ascontiguousarray(np.asarray(fir, dtype=np.float64))
    return CausalFIRPathData(
        role=role,  # type: ignore[arg-type]
        post_onset_fir=values,
        coarse_delay_samples=coarse_delay,
        fractional_delay_samples=0.0,
        support_samples=len(values),
        sample_rate=FS,
        handoff_extra_samples=handoff,
        operator_file_sha256=_sha(f"{role}-file"),
        operator_internal_sha256=_sha(f"{role}-internal"),
        fir_sha256=hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        authority_sha256=authority_sha,
        source_path=f"fixture/full-octave/{role}.npz",
    )


def _binding() -> FullOctaveCausalPlantBindingV4:
    authority_sha = _sha("plant-authority")
    delays = PlantDelays(
        primary_delay_samples=2,
        secondary_delay_samples=1,
        handoff_samples=256,
        sample_rate=FS,
    )
    timing = TrainingTimingContract.derive(
        primary_fir=(1.0,), plant_delays=delays
    )
    contract = BroadbandFullOctaveContractV3.canonical()
    return FullOctaveCausalPlantBindingV4._for_test_fixture(
        control_band_contract=contract,
        control_band_contract_sha256=contract.digest(),
        training_timing_contract=timing,
        training_timing_contract_sha256=timing.digest(),
        primary_operator=_operator(
            role="primary",
            fir=(1.0,),
            coarse_delay=2,
            handoff=0,
            authority_sha=authority_sha,
        ),
        secondary_operator=_operator(
            role="secondary",
            fir=(0.35, -0.1, 0.05),
            coarse_delay=1,
            handoff=256,
            authority_sha=authority_sha,
        ),
        verified_physical_subbands_hz=contract.physical_identification_subbands_hz,
        raw_capture_sha256=_sha("raw"),
        analysis_sha256=_sha("analysis"),
        primary_raw_capture_sha256=_sha("raw"),
        secondary_raw_capture_sha256=_sha("raw"),
        primary_analysis_sha256=_sha("analysis"),
        secondary_analysis_sha256=_sha("analysis"),
        plant_authority_sha256=authority_sha,
        electrical_witness_receipt_sha256=_sha("witness"),
        err_channel_index=0,
        err_channel_selection_sha256=_sha("err-selection"),
        reference_channel_index=0,
        reference_channel_selection_sha256=_sha("ref-selection"),
        authority_schema=FULL_OCTAVE_CAUSAL_PLANT_AUTHORITY_SCHEMA_V4,
        block_size=256,
        schema_version="full_octave_causal_plant_binding_v4",
    )


class _GainStreamingController(torch.nn.Module):
    hop = 128
    context = 256
    in_channels = 2

    def __init__(self, gain: float = -0.12) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(gain, dtype=torch.float32))

    def init_states(
        self, batch: int = 1, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        return torch.zeros((batch, 1, 1), device=device)

    def streaming_step(
        self, x_block: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.gain * x_block[:, :1] + 0.0 * state
        return output, output[..., -1:]


def _equal_octave_signal(samples: int, *, phase_offset: float) -> torch.Tensor:
    contract = BroadbandFullOctaveContractV3.canonical()
    time = torch.arange(samples, dtype=torch.float32) / float(FS)
    result = torch.zeros(samples, dtype=torch.float32)
    for index, (center, (lo, hi)) in enumerate(
        zip(
            contract.octave_objective_centers_hz,
            contract.equal_weight_octave_objective_bands_hz,
            strict=True,
        )
    ):
        # octave 폭에 비례한 power로 각 band의 PSD를 평탄하게 만든다.
        amplitude = 3.0e-4 * math.sqrt((hi - lo) / (176.7766952966 - 88.3883476483))
        result = result + float(amplitude) * torch.sin(
            2.0 * math.pi * float(center) * time + phase_offset + index * 0.13
        )
    return result


def _batch(
    adapter: CausalSecondaryPrefixAdapterV1,
    *,
    batch_size: int,
    target_samples: int,
) -> CausalPrefixBatchV1:
    prefix = 512
    lead = int(adapter.binding.training_timing_contract.digital_reference_lead_samples)
    total = prefix + target_samples
    signals = torch.stack(
        [_equal_octave_signal(total + lead, phase_offset=0.19 * index) for index in range(batch_size)],
        dim=0,
    ).unsqueeze(1)
    preview = signals[..., lead : lead + total]
    x = torch.cat((preview, torch.zeros_like(preview)), dim=1)
    sources = tuple(_sha(f"source-{index}") for index in range(batch_size))
    return CausalPrefixBatchV1(
        x_prefix=x[..., :prefix],
        x_target=x[..., prefix:],
        source_sha256=sources,
        clean_playback_source_sha256=sources,
        clean_playback_timeline=signals,
        controller_reference_preaugmentation=preview,
        training_timing_contract_sha256=adapter.binding.training_timing_contract_sha256,
        segment_prefix_start_samples=(0,) * batch_size,
        segment_target_start_samples=(prefix,) * batch_size,
        global_sample_indices=tuple(range(batch_size)),
        state_origin=CausalPrefixStateOriginV1(
            kind="segment_start_zero_state",
            binding_sha256=adapter.binding_sha256,
            source_sha256=sources,
        ),
    )


def _loss() -> BroadbandFullOctaveLossPrimitiveV3:
    contract = BroadbandFullOctaveContractV3.canonical()
    config = BroadbandFullOctaveLossConfigV3(
        control_band_contract=contract,
        control_band_contract_sha256=contract.digest(),
        lambda_dnh=0.08,
    )
    return BroadbandFullOctaveLossPrimitiveV3(config)


def _fixture_adapter() -> CausalSecondaryPrefixAdapterV1:
    return CausalSecondaryPrefixAdapterV1._for_test_fixture(_binding())


def test_trainer_consumer_uses_adapter_primary_secondary_and_seven_octave_loss() -> None:
    adapter = _fixture_adapter()
    consumer = FullOctaveV3TrainerConsumer._for_test_fixture(adapter, _loss())
    model = _GainStreamingController()
    batch = _batch(adapter, batch_size=4, target_samples=16_384)

    result = consumer.compute_loss(model, batch)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert model.gain.grad is not None and torch.isfinite(model.gain.grad)
    assert result.metrics["v3_consumer_causal_prefix_used"] == 1.0
    assert result.metrics["v3_consumer_actual_secondary_output_used"] == 1.0
    assert result.metrics["v3_consumer_equal_octave_count"] == 7.0
    assert result.metrics["v3_consumer_canonical_training_claim"] == 0.0
    assert torch.equal(
        result.causal_result.error_target,
        result.causal_result.primary_target + result.causal_result.secondary_target,
    )
    # target만 S에 통과시킨 값은 prefix tail을 잃으므로 consumer result와 같을 수 없다.
    target_only = adapter.secondary_path(result.causal_result.y_target)
    assert not torch.allclose(target_only, result.causal_result.secondary_target)
    assert all(f"v3_octave_{index}_objective_db" in result.metrics for index in range(7))


def test_fixture_cannot_enter_public_trainer_or_matched_fxlms_consumer() -> None:
    adapter = _fixture_adapter()
    with pytest.raises(ValueError, match="test fixture"):
        FullOctaveV3TrainerConsumer(adapter, _loss())
    with pytest.raises(ValueError, match="test fixture"):
        FullOctaveV3MatchedFxLMSEvaluator(adapter)


def test_matched_fxlms_uses_same_binding_reference_prefix_and_target_crop() -> None:
    adapter = _fixture_adapter()
    evaluator = FullOctaveV3MatchedFxLMSEvaluator._for_test_fixture(
        adapter,
        fxlms=FullOctaveV3FxLMSConfig(control_length=64, mu=0.02),
    )
    model = _GainStreamingController()
    batch = _batch(adapter, batch_size=1, target_samples=4_096)
    causal, segment = evaluator.evaluate(
        model,
        batch,
        identity=FullOctaveV3EvaluationIdentity(
            session_id="fixture-session",
            source_family="speech",
            group_id="fixture-group",
        ),
    )
    assert segment.causal_plant_binding_sha256 == adapter.binding_sha256
    assert segment.session_id == "fixture-session"
    assert segment.group_id == "fixture-group"
    assert segment.disturbance_off.shape == (causal.target_samples,)
    assert segment.error_deep_anc.shape == segment.error_fxlms.shape
    assert not np.array_equal(segment.error_deep_anc, segment.error_fxlms)
    assert segment.evaluation_domain == "surrogate_matched_causal_ps_not_physical"


def test_matched_fxlms_rejects_batched_items_without_per_item_identity_receipts() -> None:
    adapter = _fixture_adapter()
    evaluator = FullOctaveV3MatchedFxLMSEvaluator._for_test_fixture(adapter)
    with pytest.raises(ValueError, match="batch size=1"):
        evaluator.evaluate(
            _GainStreamingController(),
            _batch(adapter, batch_size=2, target_samples=4_096),
            identity=FullOctaveV3EvaluationIdentity(
                session_id="one-session",
                source_family="speech",
                group_id="one-group",
            ),
        )


def _raw_campaign(*, deep_scale: np.ndarray, fxlms_scale: np.ndarray):
    contract = BroadbandFullOctaveContractV3.canonical()
    samples = 16_384
    time = np.arange(samples, dtype=np.float64) / float(FS)
    tones = []
    for index, (center, (lo, hi)) in enumerate(
        zip(
            contract.octave_objective_centers_hz,
            contract.equal_weight_octave_objective_bands_hz,
            strict=True,
        )
    ):
        amplitude = 0.002 * math.sqrt((hi - lo) / (176.7766952966 - 88.3883476483))
        tones.append(amplitude * np.sin(2.0 * math.pi * center * time + index * 0.11))
    tones = np.asarray(tones)
    disturbance = tones.sum(axis=0)
    deep = (deep_scale[:, None] * tones).sum(axis=0)
    fxlms = (fxlms_scale[:, None] * tones).sum(axis=0)
    segments = []
    for family in contract.source_families:
        for group in range(4):
            segments.append(
                FullOctaveV3MatchedSegment(
                    session_id=f"{family}-{group}",
                    source_family=family,
                    group_id=f"{family}-{group}",
                    error_position_id="surrogate_center",
                    sample_rate=FS,
                    disturbance_off=disturbance,
                    error_deep_anc=deep,
                    error_fxlms=fxlms,
                    causal_plant_binding_sha256="a" * 64,
                )
            )
    return segments


def test_full_octave_evaluator_requires_positive_all_octaves_and_highband_fxlms_win() -> None:
    passed = evaluate_full_octave_v3_matched_segments(
        _raw_campaign(
            deep_scale=np.full(7, 0.40),
            fxlms_scale=np.full(7, 0.65),
        ),
        contract=BroadbandFullOctaveContractV3.canonical(),
        n_resamples=100,
    )
    assert passed["status"] == "PASS"
    assert passed["canonical_training_or_physical_g4_claim"] is False
    assert len(passed["cells"]) == 4 * 7
    high = [cell for cell in passed["cells"] if cell["octave_center_hz"] >= 2000.0]
    assert len(high) == 4 * 3
    assert all(cell["matched_fxlms_superiority_pass"] for cell in high)

    failed = evaluate_full_octave_v3_matched_segments(
        _raw_campaign(
            deep_scale=np.asarray((0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.80)),
            fxlms_scale=np.asarray((0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.30)),
        ),
        contract=BroadbandFullOctaveContractV3.canonical(),
        n_resamples=100,
    )
    assert failed["status"] == "BLOCKED"
    failed_8k = [cell for cell in failed["cells"] if cell["octave_center_hz"] == 8000.0]
    assert failed_8k and all(not cell["matched_fxlms_superiority_pass"] for cell in failed_8k)
