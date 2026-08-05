"""trusted-band ANC 손실과 fullband 관측 지표 검증."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deep_anc.config import REPO_ROOT, load_yaml
from deep_anc.dsp.secondary_path import (
    DifferentiableSecondaryPath,
    SecondaryPathData,
    load_secondary_path,
)
from deep_anc.losses import ANCLoss, intersect_frequency_bands


FS = 8000


def _identity_plant() -> DifferentiableSecondaryPath:
    data = SecondaryPathData(
        fir=np.array([1.0], dtype=np.float32),
        delay_samples=0,
        sample_rate=FS,
        fit_improvement_db=100.0,
        coherence_median=1.0,
        excitation_band_hz=(150.0, 600.0),
        source_path="test",
    )
    return DifferentiableSecondaryPath(data)


def _loss_cfg(objective: str | None = None) -> dict:
    cfg = {
        "mrstft_ffts": [256],
        "lambda_mrstft": 0.0,
        "lambda_pow": 0.0,
        "lambda_clip": 0.0,
        "clip_margin": 0.18,
        "band_weight": "curriculum_a",
    }
    if objective is not None:
        cfg["nmse_objective"] = objective
    return cfg


def test_measured_and_target_band_intersection() -> None:
    duct = load_yaml(REPO_ROOT / "configs/duct.yaml")
    sp = load_secondary_path(REPO_ROOT / duct["secondary_path"]["npz"])
    target = tuple(duct["acoustics"]["realistic_target_band_hz"])

    # 구동 대역이 아니라 **재현이 검증된 대역**으로 교집합을 낸다.
    # S npz 의 consistency_band 는 2026-08-05 재분석으로 [150,1600] 이 됐지만,
    # duct.yaml 의 현실적 목표 대역이 800Hz 에서 끝나므로 교집합은 800 에서 잘린다.
    assert sp.trusted_band_hz() == (150.0, 1600.0)
    trusted = intersect_frequency_bands(sp.trusted_band_hz(), target, 24_000.0)
    assert trusted == (150.0, 800.0)


def test_empty_or_invalid_band_fails_fast() -> None:
    with pytest.raises(ValueError, match="교집이 비어"):
        intersect_frequency_bands((150.0, 600.0), (800.0, 1000.0), 4000.0)
    with pytest.raises(ValueError, match="잘못된 주파수 대역"):
        intersect_frequency_bands((600.0, 150.0), (80.0, 800.0), 4000.0)
    with pytest.raises(ValueError, match="trusted_band_hz"):
        ANCLoss(
            _identity_plant(),
            _loss_cfg("trusted_band"),
            FS,
        )


def test_trusted_nmse_is_objective_and_fullband_is_reported() -> None:
    samples = FS
    t = torch.arange(samples, dtype=torch.float32) / FS
    low = torch.sin(2.0 * torch.pi * 300.0 * t)
    high = torch.sin(2.0 * torch.pi * 1200.0 * t)
    d = (low + high).view(1, 1, -1)
    y = (-low).view(1, 1, -1)

    criterion = ANCLoss(
        _identity_plant(),
        _loss_cfg("trusted_band"),
        FS,
        target_band_hz=(80.0, 800.0),
        trusted_band_hz=(150.0, 600.0),
    ).eval()
    _, metrics = criterion(y, d, perturb={"jitter": 0})

    assert metrics["nmse_db"] == metrics["nmse_trusted_db"]
    assert metrics["nmse_trusted_db"] < -80.0
    # 동일 에너지의 두 tone 중 하나만 제거 → fullband -3.01dB.
    assert metrics["nmse_fullband_db"] == pytest.approx(-3.0103, abs=0.02)


def test_legacy_constructor_keeps_fullband_objective() -> None:
    torch.manual_seed(0)
    d = torch.randn(2, 1, 4096) * 0.02
    y = -0.5 * d
    criterion = ANCLoss(_identity_plant(), _loss_cfg(), FS).eval()

    loss, metrics = criterion(y, d, perturb={"jitter": 0})

    assert torch.isfinite(loss)
    assert "nmse_trusted_db" not in metrics
    assert metrics["nmse_db"] == metrics["nmse_fullband_db"]
    assert metrics["nmse_db"] == pytest.approx(-6.0206, abs=0.01)


@pytest.mark.parametrize("samples", [4095, 4096])
def test_trusted_full_spectrum_matches_time_nmse_with_fft_endpoints(samples: int) -> None:
    """DC/Nyquist 포함 대역의 one-sided FFT 가중치를 홀수/짝수 길이에서 검증."""
    torch.manual_seed(samples)
    d = torch.randn(2, 1, samples) * 0.02
    y = -0.5 * d
    criterion = ANCLoss(
        _identity_plant(),
        _loss_cfg("trusted_band"),
        FS,
        trusted_band_hz=(0.0, FS / 2.0),
    ).eval()

    _, metrics = criterion(y, d, perturb={"jitter": 0})

    assert metrics["nmse_trusted_db"] == pytest.approx(
        metrics["nmse_fullband_db"], abs=1.0e-5
    )
    assert metrics["nmse_trusted_db"] == pytest.approx(-6.0206, abs=0.01)


def test_trusted_nmse_gradient_flows() -> None:
    t = torch.arange(FS, dtype=torch.float32) / FS
    d = torch.sin(2.0 * torch.pi * 300.0 * t).view(1, 1, -1)
    y = torch.zeros_like(d, requires_grad=True)
    criterion = ANCLoss(
        _identity_plant(),
        _loss_cfg("trusted_band"),
        FS,
        trusted_band_hz=(150.0, 600.0),
    ).eval()

    loss, _ = criterion(y, d, perturb={"jitter": 0})
    loss.backward()

    assert y.grad is not None
    assert torch.isfinite(y.grad).all()
    assert float(y.grad.norm()) > 0.0
