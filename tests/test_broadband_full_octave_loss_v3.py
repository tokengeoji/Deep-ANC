from __future__ import annotations

import math

import pytest
import torch
from pydantic import ValidationError

from deep_anc.dsp.control_band_contract import (
    BROADBAND_V3_OCTAVE_OBJECTIVE_BANDS_HZ,
    BroadbandFullOctaveContractV3,
    ControlBandContract,
)
from deep_anc.losses.broadband_loss import (
    BROADBAND_FULL_OCTAVE_ADMISSION_BLOCKERS,
    BROADBAND_FULL_OCTAVE_DNH_SCHEMA_VERSION,
    BROADBAND_FULL_OCTAVE_TRAINING_BLOCKER,
    BROADBAND_DNH_SCHEMA_VERSION,
    BroadbandFullOctaveLossConfigV3,
    BroadbandFullOctaveLossPrimitiveV3,
    BroadbandLossConfig,
)


def _config() -> BroadbandFullOctaveLossConfigV3:
    contract = BroadbandFullOctaveContractV3.canonical()
    return BroadbandFullOctaveLossConfigV3(
        control_band_contract=contract,
        control_band_contract_sha256=contract.digest(),
        lambda_dnh=0.12,  # diagnostic test value; calibration receipt is intentionally absent
    )


def _spectral_pair(
    *,
    batch: int = 8,
    samples: int = 32_768,
    target_amplitudes: tuple[float, ...] = (1.0,) * 7,
    residual_db: tuple[float, ...] = (-12.0,) * 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """정확한 objective-bin에서 d/e를 따로 만들고 time-domain으로 돌린다."""

    assert len(target_amplitudes) == len(residual_db) == 7
    bins = samples // 2 + 1
    target_spectrum = torch.zeros((batch, bins), dtype=torch.complex64)
    error_spectrum = torch.zeros_like(target_spectrum)
    for index, ((lo, hi), amplitude, db) in enumerate(
        zip(
            BROADBAND_V3_OCTAVE_OBJECTIVE_BANDS_HZ,
            target_amplitudes,
            residual_db,
            strict=True,
        )
    ):
        lo_bin = int(math.ceil(lo * samples / 48_000))
        hi_bin = int(math.floor(hi * samples / 48_000))
        if index != 6:
            hi_bin = min(hi_bin, int(math.ceil(hi * samples / 48_000)) - 1)
        phase = torch.linspace(0.0, 0.7, hi_bin - lo_bin + 1)
        values = float(amplitude) * torch.exp(1j * phase)
        target_spectrum[:, lo_bin : hi_bin + 1] = values
        error_spectrum[:, lo_bin : hi_bin + 1] = values * (
            10.0 ** (float(db) / 20.0)
        )
    target = torch.fft.irfft(target_spectrum, n=samples)
    error = torch.fft.irfft(error_spectrum, n=samples)
    return target, error


def test_v3_config_requires_exact_inline_payload_and_digest_without_v2_promotion() -> None:
    config = _config()
    assert config.control_band_contract_sha256 == (
        "53579b9ff8419ac19fb2458c29a3e8a94ffbb2eeb88cc07f34b76c68033989f2"
    )
    assert config.training_admission_status == BROADBAND_FULL_OCTAVE_TRAINING_BLOCKER
    assert config.canonical_training_eligible is False
    assert config.legacy_v2_checkpoint_automatic_promotion_allowed is False
    assert config.training_admission_blockers == (
        BROADBAND_FULL_OCTAVE_ADMISSION_BLOCKERS
    )
    assert config.diagnostic_design_hyperparameters_not_campaign_authority is True
    assert config.campaign_hyperparameter_authority_sha256 is None
    assert config.dnh_schema_version == BROADBAND_FULL_OCTAVE_DNH_SCHEMA_VERSION
    assert config.dnh_schema_version != BROADBAND_DNH_SCHEMA_VERSION

    payload = config.model_dump(mode="python")
    payload["control_band_contract_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="payload.*SHA"):
        BroadbandFullOctaveLossConfigV3.model_validate(payload)

    v2 = ControlBandContract.broadband_point_control()
    payload = config.model_dump(mode="python")
    payload["control_band_contract"] = v2.model_dump(mode="python")
    payload["control_band_contract_sha256"] = v2.digest()
    with pytest.raises(ValidationError):
        BroadbandFullOctaveLossConfigV3.model_validate(payload)

    # 반대 방향도 extra=forbid로 닫힌다. v3 payload를 기존 v2 config가 먹지 않는다.
    with pytest.raises(ValidationError):
        BroadbandLossConfig.parse(
            {"control_band_contract": config.control_band_contract.model_dump(mode="json")}
        )


def test_v3_covers_missing_88_to_150_and_does_not_make_1600_an_octave() -> None:
    contract = BroadbandFullOctaveContractV3.canonical()
    assert contract.equal_weight_octave_objective_bands_hz[0] == (
        88.3883476483,
        176.7766952966,
    )
    assert contract.octave_objective_centers_hz == (
        125.0,
        250.0,
        500.0,
        1000.0,
        2000.0,
        4000.0,
        8000.0,
    )
    assert 1600.0 not in contract.octave_objective_centers_hz
    objective_owner = [
        index
        for index, (lo, hi) in enumerate(
            contract.equal_weight_octave_objective_bands_hz
        )
        if lo <= 1600.0 < hi
    ]
    stage1_owner = [
        index
        for index, (lo, hi) in enumerate(contract.stage1_low_guard_subbands_hz)
        if lo <= 1600.0 < hi
    ]
    assert objective_owner == [4]
    assert stage1_owner == []  # Stage-1의 1600 Hz 상단은 half-open이다.

    primitive = BroadbandFullOctaveLossPrimitiveV3(_config())
    target, _ = _spectral_pair(residual_db=(-80.0,) * 7)
    time = torch.arange(target.shape[-1], dtype=torch.float32) / 48_000.0
    error_100 = 0.05 * torch.sin(2.0 * math.pi * 100.0 * time)
    error = error_100.unsqueeze(0).repeat(target.shape[0], 1)
    _, metrics = primitive(
        torch.zeros_like(target),
        target,
        error - target,
    )
    assert metrics["v3_octave_0_objective_db"] > 0.0
    assert metrics["v3_octave_0_objective_db"] > metrics["v3_octave_1_objective_db"]


def test_seven_octaves_are_target_normalized_and_equal_weighted_despite_high_energy() -> None:
    primitive = BroadbandFullOctaveLossPrimitiveV3(_config())
    expected = (-20.0, -18.0, -16.0, -14.0, -12.0, -10.0, 1.0)
    # 고역은 더 넓을 뿐 아니라 bin당 target amplitude도 1.8배다. fullband 합계라면
    # 고역이 지배하지만 octave objective는 각자 d 에너지로 나눈 scalar를 1/7로 쓴다.
    target, error = _spectral_pair(
        target_amplitudes=(1.0, 1.0, 1.0, 1.0, 1.8, 1.8, 1.8),
        residual_db=expected,
    )
    _, metrics = primitive(torch.zeros_like(target), target, error - target)
    observed = tuple(metrics[f"v3_octave_{index}_objective_db"] for index in range(7))
    assert observed == pytest.approx(expected, abs=0.03)
    assert metrics["nmse_v3_octave_equal_db"] == pytest.approx(
        sum(expected) / 7.0, abs=0.03
    )
    assert metrics["v3_equal_octave_weight"] == pytest.approx(1.0 / 7.0)
    # 평균은 음수여도 8 kHz octave 실패가 high guard에서 사라지지 않는다.
    assert metrics["nmse_v3_octave_equal_db"] < 0.0
    assert metrics["nmse_v3_high_worst_objective_db"] > 0.9
    assert metrics["v3_low_high_positive_guard"] > 0.9
    assert metrics["v3_design_hyperparameters_diagnostic_only"] == 1.0
    assert metrics["v3_campaign_hyperparameter_authority_present"] == 0.0


def test_v3_nmse_is_scale_invariant_for_quiet_but_valid_batches() -> None:
    primitive = BroadbandFullOctaveLossPrimitiveV3(_config())
    observed: list[float] = []
    for scale in (1.0, 1.0e-3, 1.0e-5, 1.0e-7):
        target, error = _spectral_pair(
            target_amplitudes=(scale,) * 7,
            residual_db=(-20.0,) * 7,
        )
        _, metrics = primitive(torch.zeros_like(target), target, error - target)
        observed.append(metrics["nmse_v3_octave_equal_db"])
    assert observed == pytest.approx((-20.0,) * 4, abs=0.03)


def test_exact_and_near_cancellation_have_finite_loss_and_gradient() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        for residual_scale in (0.0, 1.0e-5):
            primitive = BroadbandFullOctaveLossPrimitiveV3(_config())
            target, _ = _spectral_pair(target_amplitudes=(1.0,) * 7)
            target = target.to(device)
            secondary = (-(1.0 - residual_scale) * target).detach().requires_grad_(True)
            loss, metrics = primitive(torch.zeros_like(target), target, secondary)
            assert torch.isfinite(loss)
            assert metrics["v3_relative_nmse_floor_db"] == pytest.approx(-80.0)
            loss.backward()
            assert secondary.grad is not None
            assert torch.isfinite(secondary.grad).all()
            if residual_scale == 0.0:
                assert torch.count_nonzero(secondary.grad) == 0


def test_additive_worst_guard_preserves_each_octave_one_seventh_baseline() -> None:
    primitive = BroadbandFullOctaveLossPrimitiveV3(_config())
    baseline = (-20.0, -18.0, -16.0, -14.0, -12.0, -10.0, -8.0)
    changed = (-19.0, *baseline[1:])
    losses: list[float] = []
    for residual in (baseline, changed):
        target, error = _spectral_pair(residual_db=residual)
        loss, metrics = primitive(torch.zeros_like(target), target, error - target)
        assert metrics["v3_stage1_positive_guard"] == pytest.approx(0.0)
        assert metrics["v3_low_high_positive_guard"] == pytest.approx(0.0)
        losses.append(float(loss.detach()))
    # worst octave(-8 dB)는 그대로이므로 non-worst 한 octave +1 dB의 영향은 exact 1/7이다.
    assert losses[1] - losses[0] == pytest.approx(1.0 / 7.0, abs=0.03)


def test_stage1_four_band_positive_guard_and_valid_item_floor_are_independent() -> None:
    primitive = BroadbandFullOctaveLossPrimitiveV3(_config())
    raw = (
        torch.full((8,), -12.0),
        torch.full((8,), 2.0),
        torch.full((8,), -10.0),
        torch.full((8,), -8.0),
    )
    ratios = torch.ones((8, 4))
    objectives, _, _ = primitive._qualified_objectives(
        raw,
        ratios,
        torch.ones_like(ratios, dtype=torch.bool),
        bands_hz=primitive.stage1_guard_bands_hz,
        role="v3_stage1_guard",
    )
    assert float(torch.relu(objectives).sum()) == pytest.approx(2.0)

    target, error = _spectral_pair(batch=3)
    with pytest.raises(ValueError, match="valid_items=3, required=4"):
        primitive(torch.zeros_like(target), target, error - target)


def test_control_union_outside_actuator_output_is_do_no_harm_penalized() -> None:
    primitive = BroadbandFullOctaveLossPrimitiveV3(_config())
    target, error = _spectral_pair(residual_db=(-12.0,) * 7)
    samples = target.shape[-1]
    time = torch.arange(samples, dtype=torch.float32) / 48_000.0
    inside = (0.03 * torch.sin(2.0 * math.pi * 1000.0 * time)).repeat(
        target.shape[0], 1
    )
    outside = (0.03 * torch.sin(2.0 * math.pi * 12000.0 * time)).repeat(
        target.shape[0], 1
    )
    _, inside_metrics = primitive(inside, target, error - target)
    _, outside_metrics = primitive(outside, target, error - target)
    assert inside_metrics["v3_dnh"] == pytest.approx(0.0, abs=1.0e-5)
    assert outside_metrics["v3_dnh"] > 1.0
    assert outside_metrics["v3_dnh_gradient_share_calibrated"] == 0.0
    assert outside_metrics["v3_dnh_replaces_physical_err_g4"] == 0.0


def test_v3_dnh_denominator_is_detached_but_outside_energy_has_gradient() -> None:
    primitive = BroadbandFullOctaveLossPrimitiveV3(_config())
    target, error = _spectral_pair(residual_db=(-12.0,) * 7)
    samples = target.shape[-1]
    index = torch.arange(samples, dtype=torch.float32)
    inside_amplitude = torch.tensor(0.03, requires_grad=True)
    outside_amplitude = torch.tensor(0.30, requires_grad=True)
    # exact FFT bins: 1500 Hz is protected, 12 kHz is outside the control union.
    inside = torch.sin(2.0 * math.pi * 1024.0 * index / samples)
    outside = torch.sin(2.0 * math.pi * 8192.0 * index / samples)
    waveform = inside_amplitude * inside + outside_amplitude * outside
    actuator = waveform.unsqueeze(0).repeat(target.shape[0], 1)
    loss, metrics = primitive(actuator, target, error - target)
    loss.backward()
    assert metrics["v3_dnh"] > 0.0
    assert inside_amplitude.grad is not None
    assert outside_amplitude.grad is not None
    assert float(abs(inside_amplitude.grad)) < 1.0e-5
    assert float(abs(outside_amplitude.grad)) > 1.0e-3


def test_actuator_dnh_pass_does_not_replace_physical_error_amplification_gate() -> None:
    primitive = BroadbandFullOctaveLossPrimitiveV3(_config())
    target, _ = _spectral_pair(residual_db=(-12.0,) * 7)
    # control union 안 y만 쓰면 actuator leakage DNH는 0이다. 하지만 S*y를 +d로
    # 두면 e=d+S*y=2d, 즉 모든 octave에서 약 +6.02 dB 증폭한다.
    samples = target.shape[-1]
    time = torch.arange(samples, dtype=torch.float32) / 48_000.0
    inside_y = (0.03 * torch.sin(2.0 * math.pi * 1000.0 * time)).repeat(
        target.shape[0], 1
    )
    _, metrics = primitive(inside_y, target, target)
    assert metrics["v3_dnh"] == pytest.approx(0.0, abs=1.0e-5)
    assert metrics["nmse_v3_octave_equal_db"] == pytest.approx(
        20.0 * math.log10(2.0), abs=0.03
    )
    assert metrics["v3_actual_family_balanced_batch_receipt_consumed"] == 0.0
