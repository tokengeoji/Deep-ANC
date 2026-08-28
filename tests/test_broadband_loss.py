"""광대역 equal-subband 손실의 fail-closed 계약."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from deep_anc.dsp.control_band_contract import ControlBandContract
from deep_anc.dsp.measured_band_path import (
    MeasuredBandPath,
    MeasuredBandPathData,
)
from deep_anc.dsp.secondary_path import (
    DifferentiableSecondaryPath,
    SecondaryPathData,
)
from deep_anc.losses import ANCLoss, BroadbandANCLoss, BroadbandLossConfig


FS = 48_000
SAMPLES = 4_800


def _identity_plant() -> MeasuredBandPath:
    contract = ControlBandContract.broadband_point_control()
    frequencies = np.arange(104.0, 11_400.0 + 0.1, 16.0)
    data = MeasuredBandPathData.from_arrays(
        role="secondary",
        sample_rate=FS,
        frequencies_hz=frequencies,
        transfer=np.ones(frequencies.size, dtype=np.complex128),
        bulk_delay_samples=0,
        bulk_delay_fractional_samples=0.0,
        pre_roll_samples=0,
        effective_delay_samples=0,
        fractional_effective_delay_samples=0.0,
        delay_semantics="effective_zeros_before_compact_fir",
        valid_band_hz=contract.point_control_target_hz,
        control_band_contract_sha256=contract.digest(),
        source_analysis_sha256="a" * 64,
        plant_evidence_sha256="b" * 64,
        subbands_hz=contract.point_control_subbands_hz,
        source_path="broadband-loss-test",
    )
    return MeasuredBandPath(data)


def _legacy_identity_plant() -> DifferentiableSecondaryPath:
    return DifferentiableSecondaryPath(
        SecondaryPathData(
            fir=np.asarray([1.0], dtype=np.float32),
            delay_samples=0,
            sample_rate=FS,
            fit_improvement_db=100.0,
            coherence_median=1.0,
            excitation_band_hz=(150.0, 8000.0 * math.sqrt(2.0)),
            consistency_band_hz=(150.0, 8000.0 * math.sqrt(2.0)),
            source_path="legacy-broadband-loss-test",
        )
    )


def _cfg(**overrides: object) -> dict[str, object]:
    # 주 NMSE만 수치 비교하되 DNH는 광대역 계약상 끌 수 없으므로 켜 둔다. fixture의
    # 파형은 point-control union 안의 정확한 FFT bin뿐이라 DNH 값은 정확히 0이다.
    raw: dict[str, object] = {
        "mrstft_ffts": [256],
        "lambda_mrstft": 0.0,
        "lambda_pow": 0.0,
        "lambda_dnh": 0.01,
        "lambda_frame": 0.0,
        "lambda_sat": 0.0,
        "band_weight": "trusted_only",
        "nmse_cvar_alpha": 0.0,
    }
    raw.update(overrides)
    return raw


def _flat_subband_components(samples: int = SAMPLES) -> torch.Tensor:
    """각 authority subband의 모든 FFT bin에 같은 power density를 넣는다."""

    contract = ControlBandContract.broadband_point_control()
    bins = samples // 2 + 1
    spectra = torch.zeros((7, bins), dtype=torch.complex64)
    generator = torch.Generator().manual_seed(20260828 + samples)
    for index, (lo, hi) in enumerate(contract.point_control_subbands_hz):
        lo_bin = int(math.ceil(lo * samples / FS))
        hi_bin = (
            int(math.floor(hi * samples / FS))
            if index == 6
            else int(math.ceil(hi * samples / FS)) - 1
        )
        phase = 2.0 * torch.pi * torch.rand(
            hi_bin - lo_bin + 1,
            generator=generator,
        )
        spectra[index, lo_bin : hi_bin + 1] = torch.polar(
            torch.ones_like(phase), phase
        )
    return torch.fft.irfft(spectra, n=samples)


def _spectral_case(
    residual_by_band: tuple[float, ...],
    *,
    batch_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(residual_by_band) == 7
    components = _flat_subband_components()
    d = components.sum(dim=0)
    # e = d + y = residual*d, 항등 S이므로 y=(residual-1)*d_component.
    y = sum(
        (
            (float(residual) - 1.0) * component
            for residual, component in zip(residual_by_band, components, strict=True)
        ),
        start=torch.zeros_like(d),
    )
    return (
        y.view(1, 1, -1).repeat(batch_size, 1, 1),
        d.view(1, 1, -1).repeat(batch_size, 1, 1),
    )


def _criterion(**overrides: object) -> BroadbandANCLoss:
    return BroadbandANCLoss(_identity_plant(), _cfg(**overrides), FS).eval()


def test_stage1_class_and_config_are_not_widened_by_broadband_keys() -> None:
    """기존 Stage-1 parser/constructor가 광대역 키를 조용히 받으면 역할이 섞인다."""

    with pytest.raises(ValueError, match="subband_guard_alpha"):
        ANCLoss(
            _legacy_identity_plant(),
            {"subband_guard_alpha": 0.7},
            FS,
            trusted_band_hz=(150.0, 1600.0),
        )
    assert BroadbandLossConfig().nmse_objective == "trusted_band"
    broadband = _criterion()
    authority = ControlBandContract.broadband_point_control()
    assert broadband.point_control_subbands_hz == authority.point_control_subbands_hz
    assert broadband.control_band_contract_sha256 == authority.digest()
    with pytest.raises(ValueError, match="Stage-1"):
        BroadbandANCLoss(
            _identity_plant(),
            _cfg(),
            FS,
            control_band_contract=ControlBandContract.stage1_strict(),
        )


def test_lowband_success_cannot_hide_highband_failure() -> None:
    """저역 네 band만 -20 dB여도 고역 worst/CVaR는 0 dB로 남아야 한다."""

    y, d = _spectral_case((0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0))
    _, metrics = _criterion()(y, d, perturb={"jitter": 0})

    assert metrics["nmse_low_worst_db"] < -19.9
    assert metrics["nmse_high_worst_db"] == pytest.approx(0.0, abs=2.0e-4)
    assert metrics["nmse_subband_worst_db"] == pytest.approx(0.0, abs=2.0e-4)
    assert metrics["nmse_subband_guard_cvar_db"] == pytest.approx(0.0, abs=2.0e-4)
    assert metrics["nmse_subband_guard_cvar_db"] > metrics["nmse_subband_equal_db"]


def test_highband_success_cannot_hide_lowband_failure() -> None:
    """고역 세 band만 -20 dB여도 저역 worst/CVaR는 0 dB로 남아야 한다."""

    y, d = _spectral_case((1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1))
    _, metrics = _criterion()(y, d, perturb={"jitter": 0})

    assert metrics["nmse_high_worst_db"] < -19.9
    assert metrics["nmse_low_worst_db"] == pytest.approx(0.0, abs=2.0e-4)
    assert metrics["nmse_subband_worst_db"] == pytest.approx(0.0, abs=2.0e-4)
    assert metrics["nmse_subband_guard_cvar_db"] == pytest.approx(0.0, abs=2.0e-4)
    assert metrics["nmse_subband_guard_cvar_db"] > metrics["nmse_subband_equal_db"]


def test_equal_subband_normalization_ignores_energy_and_bin_count() -> None:
    """대역 폭/에너지가 크게 달라도 같은 잔차비는 정확히 같은 NMSE여야 한다."""

    samples = 8192
    contract = ControlBandContract.broadband_point_control()
    components = _flat_subband_components(samples)
    # 각 band의 bin당 density는 같지만, 150--300Hz와 5.657--11.314kHz의 총
    # 에너지는 bin 수만큼 크게 다르다. item 레벨도 80 dB 범위로 달리한다.
    # 수치 EPS 자체가 물리 에너지에 섞이는 비정상적 near-zero fixture는 피한다.
    scale = torch.tensor([1.0e-1, 1.0, 10.0, 1.0e3]).view(-1, 1, 1)
    d = components.sum(dim=0).view(1, 1, -1) * scale
    y = (-0.5 * d).clone()

    _, metrics = _criterion()(y, d, perturb={"jitter": 0})
    objectives = [
        metrics[f"nmse_subband_{index}_{int(round(lo))}_{int(round(hi))}_objective_db"]
        for index, (lo, hi) in enumerate(contract.point_control_subbands_hz)
    ]
    assert objectives == pytest.approx([-6.0206] * 7, abs=2.0e-4)
    assert metrics["nmse_subband_equal_db"] == pytest.approx(-6.0206, abs=2.0e-4)
    assert metrics["nmse_subband_guard_cvar_db"] == pytest.approx(-6.0206, abs=2.0e-4)


def test_gradient_reaches_every_control_subband() -> None:
    """대역 CVaR에 들지 않은 band도 equal baseline 때문에 0이 아닌 gradient를 받는다."""

    y, d = _spectral_case((1.0,) * 7)
    # residual=1은 y=0이므로 leaf tensor로 다시 만든다.
    y = y.detach().requires_grad_(True)
    loss, _ = _criterion()(y, d, perturb={"jitter": 0})
    loss.backward()

    assert y.grad is not None
    assert torch.isfinite(y.grad).all()
    gradient_spectrum = torch.fft.rfft(y.grad[:, 0], dim=-1)
    contract = ControlBandContract.broadband_point_control()
    for index, (lo, hi) in enumerate(contract.point_control_subbands_hz):
        lo_bin = int(math.ceil(lo * SAMPLES / FS))
        hi_bin = (
            int(math.floor(hi * SAMPLES / FS))
            if index == 6
            else int(math.ceil(hi * SAMPLES / FS)) - 1
        )
        assert float(
            gradient_spectrum[:, lo_bin : hi_bin + 1].abs().sum()
        ) > 1.0e-7, (lo, hi)


def test_target_density_rejects_a_batch_with_one_empty_subband() -> None:
    components = _flat_subband_components()
    d = components[:-1].sum(dim=0).view(1, 1, -1).repeat(4, 1, 1)
    y = torch.zeros_like(d)

    with pytest.raises(ValueError, match="target-d density.*5656"):
        _criterion()(y, d, perturb={"jitter": 0})


def test_target_density_requires_item_cvar_minimum_per_subband() -> None:
    components = _flat_subband_components()
    full = components.sum(dim=0)
    without_high = components[:-1].sum(dim=0)
    d = full.view(1, 1, -1).repeat(4, 1, 1)
    d[0, 0] = without_high
    y = torch.zeros_like(d)

    with pytest.raises(ValueError, match="valid_items=3, required=4"):
        _criterion()(y, d, perturb={"jitter": 0})


def test_dnh_is_the_exact_fft_bin_complement_of_point_control_union() -> None:
    """DNH가 연속 구간뿐 아니라 실제 FFT bin에서도 보호 대역과 겹치지 않는다."""

    criterion = _criterion()
    plan = criterion.do_no_harm
    assert plan is not None
    protected_lo, protected_hi = criterion.control_band_contract.point_control_target_hz
    assert plan.protected.as_tuple() == pytest.approx((protected_lo, protected_hi))
    assert plan.bands[0].band.lo_hz == 0.0
    assert plan.bands[-1].band.hi_hz == FS / 2.0

    control_bins: set[int] = set()
    for index, (lo, hi) in enumerate(criterion.point_control_subbands_hz):
        lo_bin = int(math.ceil(lo * SAMPLES / FS))
        if index == len(criterion.point_control_subbands_hz) - 1:
            hi_bin = int(math.floor(hi * SAMPLES / FS))
        else:
            hi_bin = int(math.ceil(hi * SAMPLES / FS)) - 1
        control_bins.update(range(lo_bin, hi_bin + 1))

    dnh_bin_membership: dict[int, int] = {}
    for item in plan.bands:
        lo, hi = item.band.as_tuple()
        lo_bin, hi_bin = criterion._dnh_band_bins(SAMPLES, lo, hi)
        for index in range(lo_bin, hi_bin + 1):
            dnh_bin_membership[index] = dnh_bin_membership.get(index, 0) + 1

    dnh_bins = set(dnh_bin_membership)
    assert control_bins.isdisjoint(dnh_bins)
    assert control_bins | dnh_bins == set(range(SAMPLES // 2 + 1))
    assert set(dnh_bin_membership.values()) == {1}


def test_broadband_config_cannot_disable_or_partially_list_dnh() -> None:
    with pytest.raises(ValueError, match="끌 수 없습니다"):
        BroadbandLossConfig(lambda_dnh=0.0)
    with pytest.raises(ValueError, match="직접 열거"):
        BroadbandLossConfig(
            do_no_harm_bands=[[0.0, 150.0, -18.3, 1.0]],
        )
    with pytest.raises(ValueError, match="minimum_target_d_density_ratio"):
        BroadbandLossConfig(minimum_target_d_density_ratio=0.249)
    with pytest.raises(ValueError, match="nmse_cvar_min_k"):
        BroadbandLossConfig(nmse_cvar_min_k=3)
    with pytest.raises(ValueError, match="lambda_mrstft"):
        BroadbandLossConfig(lambda_mrstft=0.1)
    with pytest.raises(ValueError, match="lambda_frame"):
        BroadbandLossConfig(lambda_frame=0.1)


def test_broadband_rejects_legacy_differentiable_fir_plant() -> None:
    with pytest.raises(TypeError, match="DifferentiableSecondaryPath"):
        BroadbandANCLoss(_legacy_identity_plant(), _cfg(), FS)


def test_actuator_dnh_zero_output_has_exact_zero_loss_and_gradient() -> None:
    criterion = _criterion()
    y = torch.zeros((4, 1, SAMPLES), requires_grad=True)
    loss, _ = criterion._actuator_output_dnh(
        y, n_fft=criterion._linear_dtft_size(SAMPLES)
    )
    loss.backward()
    assert float(loss) == 0.0
    assert y.grad is not None
    assert torch.count_nonzero(y.grad) == 0


def test_actuator_dnh_gradient_exists_only_outside_control_union() -> None:
    criterion = _criterion()
    n_fft = criterion._linear_dtft_size(SAMPLES)
    spectrum = torch.zeros((1, n_fft // 2 + 1), dtype=torch.complex64)
    # 하나는 control, 하나는 upper outside. 분모 detach 때문에 control tone에는
    # DNH gradient가 없어야 한다.
    control_bin = int(round(1000.0 * n_fft / FS))
    outside_bin = int(round(16_000.0 * n_fft / FS))
    spectrum[0, control_bin] = 1.0 + 0.0j
    spectrum[0, outside_bin] = 0.1 + 0.0j
    y = torch.fft.irfft(spectrum, n=n_fft)[..., :SAMPLES].view(1, 1, -1)
    y = y.detach().requires_grad_(True)
    loss, _ = criterion._actuator_output_dnh(y, n_fft=n_fft)
    loss.backward()
    gradient = torch.fft.rfft(y.grad[:, 0], n=n_fft).abs()
    # finite crop 때문에 exact-bin leakage가 생기므로 절대 0 대신 outside 대비 수치
    # roundoff 수준을 요구한다.
    assert float(gradient[0, outside_bin]) > 1.0e-8
    assert float(gradient[0, control_bin]) < float(gradient[0, outside_bin]) * 1.0e-5


def test_finite_sequence_dtft_refuses_settle_crop_instead_of_circular_ifft() -> None:
    criterion = _criterion()
    assert criterion._linear_dtft_size(SAMPLES) == SAMPLES
    y = torch.randn(4, 1, SAMPLES)
    d = torch.randn(4, 1, SAMPLES)
    with pytest.raises(ValueError, match="loss_start_sample=0"):
        criterion(y, d, loss_start_sample=1)


def test_measured_frequency_product_matches_full_linear_convolution_dtft() -> None:
    """H(f)Y(f)가 truncated/circular가 아닌 full linear convolution과 같다."""

    contract = ControlBandContract.broadband_point_control()
    frequencies = np.arange(104.0, 11_400.0 + 0.1, 16.0)
    fir = np.asarray([0.7, -0.12, 0.04], dtype=np.float64)
    tap_index = np.arange(fir.size, dtype=np.float64)
    measured = np.exp(
        -2j * np.pi * frequencies[:, None] * tap_index[None, :] / FS
    ) @ fir
    data = MeasuredBandPathData.from_arrays(
        role="secondary",
        sample_rate=FS,
        frequencies_hz=frequencies,
        transfer=measured,
        bulk_delay_samples=0,
        bulk_delay_fractional_samples=0.0,
        pre_roll_samples=0,
        effective_delay_samples=0,
        fractional_effective_delay_samples=0.0,
        delay_semantics="effective_zeros_before_compact_fir",
        valid_band_hz=contract.point_control_target_hz,
        control_band_contract_sha256=contract.digest(),
        source_analysis_sha256="c" * 64,
        plant_evidence_sha256="d" * 64,
        subbands_hz=contract.point_control_subbands_hz,
        source_path="known-fir-frequency-only",
    )
    path = MeasuredBandPath(data)
    # 6000 samples gives 8 Hz DFT bins, so 104+16k measured tones are exact bins.
    generator = torch.Generator().manual_seed(7)
    y = torch.randn(6000, generator=generator, dtype=torch.float64)
    full_linear = np.convolve(y.numpy(), fir, mode="full")
    query = torch.tensor([152.0, 1592.0, 8008.0, 11_304.0], dtype=torch.float64)
    bins = (query / 8.0).to(dtype=torch.int64)
    product = path.response_at(query) * torch.fft.rfft(y)[bins]
    time = np.arange(full_linear.size, dtype=np.float64)
    direct = np.exp(
        -2j * np.pi * query.numpy()[:, None] * time[None, :] / FS
    ) @ full_linear
    assert product.detach().numpy() == pytest.approx(
        direct, rel=2.0e-10, abs=2.0e-10
    )
