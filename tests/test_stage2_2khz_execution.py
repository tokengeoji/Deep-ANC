from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from deep_anc.dsp.stage2_2khz_contract import Stage2TwoKilohertzContract
from deep_anc.dsp.timing import PlantDelays, TrainingTimingContract
from deep_anc.losses.broadband_loss import CausalFIRPathData
from deep_anc.losses.stage2_2khz_loss import (
    STAGE2_2KHZ_ONE_POINT_SIX_SENTINEL_BAND_HZ,
    Stage2TwoKilohertzLoss,
    Stage2TwoKilohertzLossConfig,
)
from deep_anc.train.causal_secondary_prefix_adapter_v1 import (
    CausalPrefixBatchV1,
    CausalPrefixStateOriginV1,
)
from deep_anc.train.stage2_2khz_binding import (
    STAGE2_2KHZ_ANALYSIS_SCHEMA,
    STAGE2_2KHZ_PATH_NPZ_SCHEMA,
    STAGE2_2KHZ_PLANT_BINDING_SCHEMA,
    STAGE2_2KHZ_RAW_CAPTURE_SCHEMA,
    STAGE2_2KHZ_RELATIVE_CLOCK_MODEL_SCHEMA,
    Stage2TwoKilohertzPlantBinding,
    _load_self_attested_stage2_2khz_plant_binding_for_test,
    load_stage2_2khz_plant_binding,
)
from deep_anc.train.stage2_2khz_execution import (
    Stage2ActualBatchIdentity,
    Stage2CausalPrefixAdapter,
    Stage2FamilyComponentBatchSampler,
    Stage2SamplerRecord,
    Stage2TensorBatch,
    Stage2TwoKilohertzTrainerAdapter,
)


FAMILIES = ("speech", "music", "environment", "machine")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _loss(*, sampler_sha: str = "b" * 64) -> Stage2TwoKilohertzLoss:
    contract = Stage2TwoKilohertzContract.canonical()
    config = Stage2TwoKilohertzLossConfig(
        control_band_contract=contract,
        control_band_contract_sha256=contract.digest(),
        lambda_dnh=0.1,
        dnh_calibration_receipt_sha256="a" * 64,
        dnh_observed_gradient_share=0.3,
        family_balanced_sampler_receipt_sha256=sampler_sha,
    )
    return Stage2TwoKilohertzLoss(config)


def _objective_noise(batch: int = 4, samples: int = 16_384) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260831)
    spectrum = torch.zeros((batch, samples // 2 + 1), dtype=torch.complex64)
    frequencies = torch.fft.rfftfreq(samples, 1.0 / 48_000)
    selected = (frequencies >= 88.3883476483) & (frequencies <= 2828.4271247462)
    real = torch.randn((batch, int(selected.sum())), generator=generator)
    imag = torch.randn((batch, int(selected.sum())), generator=generator)
    spectrum[:, selected] = torch.complex(real, imag)
    return torch.fft.irfft(spectrum, n=samples, dim=-1)


def _band_component(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    samples = int(value.shape[-1])
    spectrum = torch.fft.rfft(value, dim=-1)
    frequency = torch.fft.rfftfreq(samples, 1.0 / 48_000).to(value.device)
    mask = (frequency >= lower) & (frequency <= upper)
    filtered = torch.zeros_like(spectrum)
    filtered[..., mask] = spectrum[..., mask]
    return torch.fft.irfft(filtered, n=samples, dim=-1)


def test_stage2_loss_keeps_five_equal_octaves_and_two_khz_is_secondary() -> None:
    criterion = _loss()
    d = _objective_noise()
    y = 0.01 * d
    families = FAMILIES
    just_pass, metrics_pass = criterion(
        y,
        d,
        -0.30 * d,
        source_families=families,
    )
    better, metrics_better = criterion(
        y,
        d,
        -0.75 * d,
        source_families=families,
    )
    assert metrics_pass["stage2_exact_equal_octave_weight"] == pytest.approx(0.2)
    assert metrics_better["stage2_two_khz_positive_guard"] == pytest.approx(0.0)
    assert float(better.detach()) < float(just_pass.detach())


def test_stage2_loss_two_khz_failure_is_not_hidden_by_low_octaves() -> None:
    criterion = _loss()
    d = _objective_noise()
    two_k = _band_component(d, 1414.2135623731, 2828.4271247462)
    secondary = -0.75 * (d - two_k)
    loss, metrics = criterion(
        0.01 * d,
        d,
        secondary,
        source_families=FAMILIES,
    )
    assert torch.isfinite(loss)
    assert metrics["stage2_two_khz_objective_nmse_db"] > -0.2
    assert metrics["stage2_two_khz_positive_guard"] == pytest.approx(0.0)


def test_stage2_loss_one_point_six_near_zero_is_not_hidden_by_two_khz_mean() -> None:
    criterion = _loss()
    d = _objective_noise()
    sentinel = _band_component(d, *STAGE2_2KHZ_ONE_POINT_SIX_SENTINEL_BAND_HZ)
    secondary = -0.75 * (d - sentinel)
    loss, metrics = criterion(
        0.01 * d,
        d,
        secondary,
        source_families=FAMILIES,
    )
    assert torch.isfinite(loss)
    assert metrics["stage2_two_khz_objective_nmse_db"] < -3.0
    assert metrics["stage2_one_point_six_khz_sentinel_nmse_db"] > -0.2
    assert metrics["stage2_one_point_six_khz_sentinel_positive_guard"] > 0.0
    assert metrics["stage2_one_point_six_khz_minimum_attenuation_db"] == pytest.approx(6.0)


def test_stage2_loss_cannot_hide_one_failed_family_behind_three_good_families() -> None:
    criterion = _loss()
    target = _objective_noise(batch=4)
    balanced_loss, _ = criterion(
        torch.zeros_like(target),
        target,
        -(1.0 - 10.0 ** (-8.0 / 20.0)) * target,
        source_families=FAMILIES,
    )
    secondary = -0.9 * target
    secondary[0] = 0.0  # speech만 0 dB, 나머지 family는 약 -20 dB
    actuator = torch.zeros_like(target)
    loss, metrics = criterion(
        actuator,
        target,
        secondary,
        source_families=FAMILIES,
    )
    assert torch.isfinite(loss)
    assert loss > balanced_loss
    assert all(
        metrics[f"stage2_octave_{index}_family_worst_nmse_db"] > -0.1
        for index in range(5)
    )
    assert metrics["stage2_two_khz_positive_guard"] == pytest.approx(0.0)


def test_stage2_loss_rejects_missing_density_and_unbalanced_actual_batch() -> None:
    criterion = _loss()
    d = _objective_noise()
    without_two_k = d - _band_component(d, 1414.2135623731, 2828.4271247462)
    with pytest.raises(ValueError, match="target-density"):
        criterion(
            0.01 * d,
            without_two_k,
            -0.5 * without_two_k,
            source_families=FAMILIES,
        )
    with pytest.raises(ValueError, match="개수가 균등"):
        criterion(
            0.01 * d,
            d,
            -0.5 * d,
            source_families=("speech", "speech", "music", "machine"),
        )


def test_stage2_loss_penalizes_4k_8k_actuator_output_and_has_finite_gradient() -> None:
    criterion = _loss()
    d = _objective_noise()
    samples = int(d.shape[-1])
    time = torch.arange(samples, dtype=torch.float32) / 48_000.0
    high = (torch.sin(2.0 * torch.pi * 4000.0 * time) + torch.sin(2.0 * torch.pi * 8000.0 * time))
    y_low = (0.01 * d).requires_grad_()
    y_high = (0.01 * d + high.unsqueeze(0).expand_as(d)).requires_grad_()
    secondary = (-0.75 * d).requires_grad_()
    low, low_metrics = criterion(y_low, d, secondary, source_families=FAMILIES)
    high_loss, high_metrics = criterion(y_high, d, secondary, source_families=FAMILIES)
    assert high_metrics["stage2_dnh"] > low_metrics["stage2_dnh"]
    high_loss.backward()
    assert y_high.grad is not None and torch.isfinite(y_high.grad).all()
    assert secondary.grad is not None and torch.isfinite(secondary.grad).all()


def _sampler() -> Stage2FamilyComponentBatchSampler:
    records: list[Stage2SamplerRecord] = []
    index = 0
    for family in FAMILIES:
        for component in range(3):
            for item in range(2):
                records.append(
                    Stage2SamplerRecord(
                        dataset_index=index,
                        source_family=family,
                        component_id=f"{family}-{component}",
                        split="train",
                        source_sha256=_sha(f"{family}-{component}-{item}"),
                    )
                )
                index += 1
    return Stage2FamilyComponentBatchSampler(
        records,
        batch_size=8,
        seed=20260803,
        manifest_bundle_sha256="c" * 64,
        sampler_receipt_sha256="b" * 64,
    )


def test_stage2_sampler_is_actual_family_balanced_and_global_step_deterministic() -> None:
    sampler = _sampler()
    first = sampler.batch_records(17)
    second = sampler.batch_records(17)
    assert first == second
    assert len(first) == 8
    assert {family: sum(row.source_family == family for row in first) for family in FAMILIES} == {
        family: 2 for family in FAMILIES
    }
    assert sampler.batch_indices(18) != sampler.batch_indices(17)
    receipt = sampler.expected_receipt_payload()
    assert receipt["source_mix_random_tag_selector_used"] is False
    assert receipt["actual_target_density_recheck_required"] is True


def test_stage2_sampler_resume_worker_ddp_identity_is_numerically_equivalent() -> None:
    sampler = _sampler()
    original = Stage2ActualBatchIdentity.from_sampler(
        sampler,
        global_step=91,
        rank=1,
        world_size=2,
        worker_id=0,
        num_workers=1,
    )
    resumed_other_worker = Stage2ActualBatchIdentity.from_sampler(
        sampler,
        global_step=91,
        rank=1,
        world_size=2,
        worker_id=3,
        num_workers=8,
    )
    assert resumed_other_worker == original
    other_rank = Stage2ActualBatchIdentity.from_sampler(
        sampler,
        global_step=91,
        rank=0,
        world_size=2,
    )
    assert set(other_rank.global_sample_indices).isdisjoint(original.global_sample_indices)
    assert other_rank.augmentation_seeds != original.augmentation_seeds


def _operator(role: str, *, handoff: int) -> CausalFIRPathData:
    fir = np.asarray([1.0], dtype="<f8")
    return CausalFIRPathData(
        role=role,
        post_onset_fir=fir,
        coarse_delay_samples=0,
        fractional_delay_samples=0.0,
        support_samples=1,
        sample_rate=48_000,
        handoff_extra_samples=handoff,
        operator_file_sha256=_sha(f"{role}-file"),
        operator_internal_sha256=_sha(f"{role}-internal"),
        fir_sha256=hashlib.sha256(fir.tobytes()).hexdigest(),
        authority_sha256=_sha("binding-authority"),
        source_path=f"artifacts/{role}.npz",
    )


def _fixture_binding() -> Stage2TwoKilohertzPlantBinding:
    contract = Stage2TwoKilohertzContract.canonical()
    primary = _operator("primary", handoff=0)
    secondary = _operator("secondary", handoff=256)
    delays = PlantDelays(
        primary_delay_samples=0,
        secondary_delay_samples=0,
        handoff_samples=256,
        sample_rate=48_000,
    )
    timing = TrainingTimingContract.derive(
        primary_fir=primary.post_onset_fir,
        plant_delays=delays,
    )
    return Stage2TwoKilohertzPlantBinding(
        control_band_contract=contract,
        control_band_contract_sha256=contract.digest(),
        training_timing_contract=timing,
        training_timing_contract_sha256=timing.digest(),
        primary_operator=primary,
        secondary_operator=secondary,
        primary_path_sha256=_sha("primary-path"),
        secondary_path_sha256=_sha("secondary-path"),
        raw_capture_sha256=_sha("raw"),
        analysis_sha256=_sha("analysis"),
        measurement_level_evidence_sha256=_sha("level"),
        relative_clock_model_receipt_sha256=_sha("clock"),
        verified_physical_subbands_hz=tuple(contract.physical_identification_subbands_hz),
        err_channel_index=0,
        reference_channel_index=0,
        block_size=256,
        binding_file_sha256=_sha("binding-file"),
        source_capture_commit_sha="c" * 40,
        fixture_only=True,
    )


class _ScaleStreaming(nn.Module):
    hop = 256
    in_channels = 1

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(-0.25))

    def init_states(self, batch: int = 1, device: torch.device | str = "cpu") -> None:
        del batch, device
        return None

    def streaming_step(self, x_block: torch.Tensor, state: None) -> tuple[torch.Tensor, None]:
        return self.scale * x_block[:, :1], state


def test_stage2_trainer_adapter_consumes_prefix_ps_actual_batch_and_backpropagates() -> None:
    binding = _fixture_binding()
    adapter = Stage2CausalPrefixAdapter._for_test_fixture(binding)
    criterion = _loss()
    consumer = Stage2TwoKilohertzTrainerAdapter._for_test_fixture(
        adapter,
        criterion,
        manifest_bundle_sha256="c" * 64,
        sampler_receipt_sha256="b" * 64,
    )
    sampler = _sampler()
    identity = Stage2ActualBatchIdentity.from_sampler(sampler, global_step=3)
    batch_size = len(identity.source_sha256)
    total = 512 + 16_384
    clean = _objective_noise(batch=batch_size, samples=total + 256).unsqueeze(1)
    preview = clean[..., 256 : 256 + total]
    causal_batch = CausalPrefixBatchV1(
        x_prefix=preview[..., :512],
        x_target=preview[..., 512:],
        source_sha256=identity.source_sha256,
        clean_playback_source_sha256=identity.source_sha256,
        clean_playback_timeline=clean,
        controller_reference_preaugmentation=preview,
        training_timing_contract_sha256=binding.training_timing_contract_sha256,
        segment_prefix_start_samples=(0,) * batch_size,
        segment_target_start_samples=(512,) * batch_size,
        global_sample_indices=identity.global_sample_indices,
        state_origin=CausalPrefixStateOriginV1(
            kind="segment_start_zero_state",
            binding_sha256=binding.digest(),
            source_sha256=identity.source_sha256,
        ),
    )
    batch = Stage2TensorBatch(
        causal=causal_batch,
        dataset_indices=identity.dataset_indices,
        manifest_row_sha256=identity.manifest_row_sha256,
        augmentation_seeds=identity.augmentation_seeds,
    )
    model = _ScaleStreaming()
    result = consumer.compute_loss(model, batch, identity)
    result.loss.backward()
    assert result.metrics["stage2_consumer_actual_secondary_output_used"] == 1.0
    assert result.metrics["stage2_actual_family_balance_rechecked"] == 1.0
    assert model.scale.grad is not None and torch.isfinite(model.scale.grad)


def _write_npz(path: Path, **values: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **values)
    return _file_sha(path)


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return _file_sha(path)


def _production_binding_files(root: Path, *, corrupt_analysis_pcm_sha: bool = False) -> Path:
    contract = Stage2TwoKilohertzContract.canonical()
    artifacts = root / "artifacts"
    raw_path = artifacts / "raw.npz"
    submitted = np.arange(1024, dtype=np.int16).reshape(512, 2)
    captured = submitted.astype(np.int32) * 1024
    raw_sha = _write_npz(
        raw_path,
        stage2_raw_schema=np.asarray(STAGE2_2KHZ_RAW_CAPTURE_SCHEMA),
        stage2_contract_sha256=np.asarray(contract.digest()),
        capture_id=np.asarray("capture-stage2"),
        sample_rate=np.asarray(48_000),
        block_size=np.asarray(256),
        submitted_output_pcm=submitted,
        captured_input_pcm=captured,
        xrun_count=np.asarray(0),
        clip_count=np.asarray(0),
        sample_slip_count=np.asarray(0),
        callback_status_failures=np.asarray(0),
        source_capture_commit_sha=np.asarray("c" * 40),
    )
    submitted_sha = hashlib.sha256(submitted.tobytes(order="C")).hexdigest()
    analysis_path = artifacts / "analysis.npz"
    analysis_sha = _write_npz(
        analysis_path,
        stage2_analysis_schema=np.asarray(STAGE2_2KHZ_ANALYSIS_SCHEMA),
        analysis_status=np.asarray("PASS"),
        stage2_contract_sha256=np.asarray(contract.digest()),
        capture_id=np.asarray("capture-stage2"),
        raw_capture_sha256=np.asarray(raw_sha),
        sample_rate=np.asarray(48_000),
        physical_subbands_hz=np.asarray(contract.physical_identification_subbands_hz),
        primary_band_consistency=np.asarray([0.99] * 6),
        secondary_band_consistency=np.asarray([0.99] * 6),
        primary_delay_samples=np.asarray(10),
        secondary_delay_samples=np.asarray(4),
        timing_residual_max_samples=np.asarray(0.2),
        submitted_output_pcm_sha256=np.asarray(
            "0" * 64 if corrupt_analysis_pcm_sha else submitted_sha
        ),
        source_capture_commit_sha=np.asarray("c" * 40),
    )

    common = {
        "stage2_path_schema": np.asarray(STAGE2_2KHZ_PATH_NPZ_SCHEMA),
        "measurement_status": np.asarray("PASS"),
        "canonical_training_eligible": np.asarray(True),
        "stage2_contract_id": np.asarray(contract.contract_id),
        "stage2_contract_sha256": np.asarray(contract.digest()),
        "fractional_delay_samples": np.asarray(0.0),
        "delay_semantics": np.asarray("effective_zeros_before_compact_fir"),
        "sample_rate": np.asarray(48_000),
        "capture_id": np.asarray("capture-stage2"),
        "excitation_band_hz": np.asarray([80.0, 2828.4271247462]),
        "band_consistency_hz": np.asarray(contract.physical_identification_subbands_hz),
        "independent_epoch_role_names": np.asarray(
            [role for role in ("fit_a", "fit_b", "untouched_holdout") for _ in range(8)]
        ),
        "independent_epoch_start_frames": np.arange(24, dtype=np.int64) * 48_000,
        "independent_epoch_stop_frames": (np.arange(24, dtype=np.int64) + 1) * 48_000,
        "independent_epoch_kept": np.ones(24, dtype=bool),
        "repeated_slot_count": np.asarray(0),
        "timing_residual_max_samples": np.asarray(0.2),
        "xrun_count": np.asarray(0),
        "clip_count": np.asarray(0),
        "sample_slip_count": np.asarray(0),
        "callback_status_failures": np.asarray(0),
        "output_pcm_provenance": np.asarray("observed_submitted_int16"),
        "source_raw_npz_path": np.asarray("artifacts/raw.npz"),
        "source_raw_npz_sha256": np.asarray(raw_sha),
        "source_analysis_npz_path": np.asarray("artifacts/analysis.npz"),
        "source_analysis_npz_sha256": np.asarray(analysis_sha),
        "calibration_block_size": np.asarray(256),
        "error_mic_channel": np.asarray(0),
        "reference_mic_channel": np.asarray(1),
        "source_capture_commit_sha": np.asarray("c" * 40),
    }
    primary_path = artifacts / "primary.npz"
    primary_sha = _write_npz(
        primary_path,
        **common,
        role=np.asarray("primary"),
        fir=np.asarray([1.0, 0.25], dtype="<f8"),
        delay_samples=np.asarray(10),
        band_consistency=np.asarray([0.99] * 6),
    )
    secondary_path = artifacts / "secondary.npz"
    secondary_sha = _write_npz(
        secondary_path,
        **common,
        role=np.asarray("secondary"),
        fir=np.asarray([0.5, -0.125], dtype="<f8"),
        delay_samples=np.asarray(4),
        band_consistency=np.asarray([0.99] * 6),
    )
    level_path = artifacts / "level.json"
    level_sha = _write_json(
        level_path,
        {
            "schema": "measurement_level_evidence_v2_bootstrap_pair",
            "passed": True,
            "sample_rate": 48_000,
            "probe_amplitude": 0.003,
            "same_amplifier_setting": True,
        },
    )
    clock_path = artifacts / "clock.json"
    clock_sha = _write_json(
        clock_path,
        {
            "schema": STAGE2_2KHZ_RELATIVE_CLOCK_MODEL_SCHEMA,
            "status": "PASS",
            "control_band_contract_sha256": contract.digest(),
            "raw_capture_sha256": raw_sha,
            "analysis_sha256": analysis_sha,
            "submitted_output_pcm_sha256": submitted_sha,
            "relative_shared_q_model_pass": True,
            "absolute_hardware_frame_clock": False,
            "submitted_stereo_known_code_bound": True,
            "xrun_clip_status_slip_zero": True,
        },
    )
    binding_path = artifacts / "binding.json"
    _write_json(
        binding_path,
        {
            "schema": STAGE2_2KHZ_PLANT_BINDING_SCHEMA,
            "status": "PASS",
            "canonical_training_eligible": True,
            "fixture_only": False,
            "control_band_contract": {"id": contract.contract_id, "sha256": contract.digest()},
            "sample_rate_hz": 48_000,
            "block_size": 256,
            "verified_physical_subbands_hz": [
                list(row) for row in contract.physical_identification_subbands_hz
            ],
            "minimum_subband_consistency": 0.95,
            "maximum_timing_residual_samples": 0.270208,
            "minimum_independent_epochs_per_role": 8,
            "periodic_repeat_indices_allowed": False,
            "handoff_extra_samples": 256,
            "lead_source": "PlantDelays.lead()",
            "primary_path": {"path": "artifacts/primary.npz", "sha256": primary_sha},
            "secondary_path": {"path": "artifacts/secondary.npz", "sha256": secondary_sha},
            "raw_capture": {"path": "artifacts/raw.npz", "sha256": raw_sha},
            "analysis": {"path": "artifacts/analysis.npz", "sha256": analysis_sha},
            "measurement_level_evidence": {"path": "artifacts/level.json", "sha256": level_sha},
            "relative_clock_model_receipt": {"path": "artifacts/clock.json", "sha256": clock_sha},
            "err_channel_index": 0,
            "reference_channel_index": 1,
            "source_capture_commit_sha": "c" * 40,
        },
    )
    return binding_path


def test_stage2_production_binding_recomputes_raw_analysis_ps_and_lead(tmp_path: Path) -> None:
    path = _production_binding_files(tmp_path)
    with pytest.raises(ValueError, match="Git authority"):
        load_stage2_2khz_plant_binding(
            path.relative_to(tmp_path),
            repository_root=tmp_path,
            expected_binding_file_sha256=_file_sha(path),
        )
    binding = _load_self_attested_stage2_2khz_plant_binding_for_test(
        path.relative_to(tmp_path),
        repository_root=tmp_path,
        expected_binding_file_sha256=_file_sha(path),
    )
    assert binding.fixture_only is True
    assert binding.training_timing_contract.digital_reference_lead_samples == 250
    assert binding.primary_operator.coarse_delay_samples == 10
    assert binding.secondary_operator.handoff_extra_samples == 256


def test_stage2_binding_rejects_analysis_not_bound_to_submitted_pcm(tmp_path: Path) -> None:
    path = _production_binding_files(tmp_path, corrupt_analysis_pcm_sha=True)
    with pytest.raises(ValueError, match="submitted PCM bytes"):
        _load_self_attested_stage2_2khz_plant_binding_for_test(
            path.relative_to(tmp_path), repository_root=tmp_path
        )
