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
from deep_anc.dsp.timing import BandPlan
from deep_anc.losses import (
    ANCLoss,
    LossConfig,
    band_weights,
    intersect_frequency_bands,
)


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


def _loss_cfg(objective: str | None = None, **overrides) -> dict:
    """NMSE 항만 남긴 기준 설정 — 값 항등식을 검증하는 테스트용.

    새 항(대역밖 힌지·프레임·포화)은 여기서 전부 꺼 두고, 각각을 자기 테스트에서
    **켜서** 검증한다. 켜져 있는지 확인하지 않은 항은 죽은 항이 되고, 이 저장소는
    이미 그 방식으로 lambda_pow / lambda_clip 을 잃었다.
    """

    cfg = {
        "mrstft_ffts": [256],
        "lambda_mrstft": 0.0,
        "lambda_pow": 0.0,
        "lambda_dnh": 0.0,
        "lambda_frame": 0.0,
        "lambda_sat": 0.0,
        "band_weight": "curriculum_a",
    }
    if objective is not None:
        cfg["nmse_objective"] = objective
    cfg.update(overrides)
    return cfg


def test_measured_and_target_band_intersection() -> None:
    duct = load_yaml(REPO_ROOT / "configs/duct.yaml")
    sp = load_secondary_path(REPO_ROOT / duct["secondary_path"]["npz"])
    target = tuple(duct["acoustics"]["realistic_target_band_hz"])

    # (1) 규칙 자체를 고정된 입력으로 못박는다 — 설정이 바뀌어도 의미는 안 바뀐다.
    #     구동 대역이 아니라 **재현이 검증된 대역**으로 교집합을 낸다.
    assert intersect_frequency_bands((150.0, 1600.0), (80.0, 800.0), 24_000.0) == (
        150.0,
        800.0,
    )

    # (2) 현재 설정에 그 규칙을 적용한 결과가 실제 손실 대역과 같은가.
    #     기대값을 숫자로 적어 두면 duct.yaml 을 넓힐 때마다 여기가 갈라진다 —
    #     그것이 발생기 A(같은 유도가 여러 벌) 이므로 규칙으로 유도한다.
    #     S npz 의 consistency_band 는 2026-08-05 재분석으로 [150,1600] 이 됐고,
    #     실효 손실 대역은 duct.yaml 의 현실적 목표 대역과의 교집합이다.
    assert sp.trusted_band_hz() == (150.0, 1600.0)
    trusted = intersect_frequency_bands(sp.trusted_band_hz(), target, 24_000.0)
    assert trusted == (
        max(sp.trusted_band_hz()[0], float(target[0])),
        min(sp.trusted_band_hz()[1], float(target[1])),
    )

    # (3) 대역 유도의 단일 출처(BandPlan)가 같은 답을 준다. 손실 대역은 보수적으로
    #     좁고, 보고 대역은 플랜트 신뢰대역 전체다 — 넓지 않으면 절대목표 1(고역도
    #     제거)을 **검증할 방법이 없다**.
    plan = BandPlan.resolve(
        plant_trusted_band_hz=sp.trusted_band_hz(), duct_cfg=duct, sample_rate=48_000
    )
    assert plan.optimize.as_tuple() == trusted
    assert plan.measure.as_tuple() == sp.trusted_band_hz()


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


# ======================================================================================
# 절대목표 2 — 집계가 최악값을 향하는가
# ======================================================================================
def _band_noise(batch: int, samples: int, seed: int) -> torch.Tensor:
    """150–600Hz 안에만 에너지가 있는 [B, 1, T] 신호."""

    g = torch.Generator().manual_seed(seed)
    spec = torch.zeros(batch, samples // 2 + 1, dtype=torch.complex64)
    lo = int(np.ceil(150.0 * samples / FS))
    hi = int(np.floor(600.0 * samples / FS))
    real = torch.randn(batch, hi - lo + 1, generator=g)
    imag = torch.randn(batch, hi - lo + 1, generator=g)
    spec[:, lo : hi + 1] = torch.complex(real, imag)
    return torch.fft.irfft(spec, n=samples).unsqueeze(1).float()


def _per_item_gradient_share(alpha: float, gains: list[float]) -> tuple[float, np.ndarray]:
    """항등 플랜트에서 아이템별 잔차 이득을 주고 그래디언트 배분을 잰다.

    ``e_i = (1 + gain_i)·d_i`` 가 되도록 ``y_i = gain_i·d_i`` 를 만든다.
    """

    samples = FS
    d = _band_noise(len(gains), samples, seed=7)
    y = (d * torch.tensor(gains).view(-1, 1, 1)).clone().requires_grad_(True)
    criterion = ANCLoss(
        _identity_plant(),
        _loss_cfg("trusted_band", nmse_cvar_alpha=alpha, nmse_cvar_q=0.25),
        FS,
        trusted_band_hz=(150.0, 600.0),
    ).eval()
    loss, _ = criterion(y, d, perturb={"jitter": 0})
    loss.backward()
    per_item = y.grad.reshape(len(gains), -1).norm(dim=-1).detach().numpy()
    return float(loss.detach()), per_item / max(per_item.sum(), 1e-30)


# 최악 4개는 +6.02 dB 증폭, 최상 12개는 −40 dB 상쇄. 실측 분포(96개 중 8개 증폭,
# CVaR25 −0.03 dB, 중앙값 −18.8 dB)를 최소 재현한 배치다.
_SPLIT_GAINS = [1.0] * 4 + [-0.99] * 12


def test_mean_aggregation_gives_the_worst_item_the_smallest_gradient() -> None:
    """**결함 재현**: dB 산술평균의 그래디언트는 잔차에 반비례한다.

    실측 log-log 회귀 기울기 −0.94(synth) / −1.02(recorded), corr −0.99. 즉 이미
    −40 dB 인 아이템이 그래디언트를 독식하고, 증폭 중인 아이템이 가장 덜 배운다 —
    절대목표 2(최악값)와 정확히 반대 방향이다.
    """

    _, share = _per_item_gradient_share(alpha=0.0, gains=_SPLIT_GAINS)

    assert share[:4].mean() < share[4:].mean(), (
        "평균 집계가 최악 아이템에 더 큰 그래디언트를 준다면 이 결함은 없었다"
    )
    assert share[:4].sum() < 0.01
    assert share[4:].sum() > 0.99


def test_cvar_aggregation_flips_the_gradient_budget_toward_the_worst_items() -> None:
    """CVaR 로 바꾸면 배분이 뒤집힌다 (실측: 최악4 3.4% → 43.4%, 최상4 16.0% → 0.0%)."""

    _, mean_share = _per_item_gradient_share(alpha=0.0, gains=_SPLIT_GAINS)
    _, cvar_share = _per_item_gradient_share(alpha=1.0, gains=_SPLIT_GAINS)

    worst = slice(0, 4)
    assert mean_share[worst].sum() < 0.01, "평균 집계는 최악 4개를 사실상 무시한다"
    assert cvar_share[worst].sum() == pytest.approx(1.0, abs=1e-6)
    assert cvar_share[worst].sum() > 100.0 * mean_share[worst].sum()
    # 순수 CVaR 은 상위 k 밖 아이템의 그래디언트를 0 으로 만든다 (k = max(min_k, qB) = 4).
    assert int((cvar_share > 1e-12).sum()) == 4


def test_amplifying_the_worst_items_costs_more_under_cvar_than_under_the_mean() -> None:
    """같은 출력에 대해 CVaR 집계가 **더 큰 손실**을 매긴다."""

    mean_loss, _ = _per_item_gradient_share(alpha=0.0, gains=_SPLIT_GAINS)
    mixed_loss, _ = _per_item_gradient_share(alpha=0.7, gains=_SPLIT_GAINS)
    cvar_loss, _ = _per_item_gradient_share(alpha=1.0, gains=_SPLIT_GAINS)

    # mean = (4×6.02 + 12×(−40))/16 = −28.49, CVaR@25%(k=4) = +6.02
    assert mean_loss == pytest.approx(-28.49, abs=0.2)
    assert cvar_loss == pytest.approx(6.02, abs=0.2)
    assert mixed_loss == pytest.approx(0.3 * mean_loss + 0.7 * cvar_loss, abs=0.05)
    assert mean_loss < mixed_loss < cvar_loss


def test_cvar_alpha_zero_reproduces_the_previous_arithmetic_mean() -> None:
    """재현성 계약 — alpha=0 이면 예전 손실과 **완전히 같은 값**이다."""

    d = _band_noise(8, FS, seed=11)
    y = -0.9 * d
    criterion = ANCLoss(
        _identity_plant(),
        _loss_cfg("trusted_band", nmse_cvar_alpha=0.0),
        FS,
        trusted_band_hz=(150.0, 600.0),
    ).eval()
    _, metrics = criterion(y, d, perturb={"jitter": 0})

    assert metrics["nmse_db"] == pytest.approx(metrics["nmse_trusted_mean_db"], abs=1e-6)


# ======================================================================================
# 절대목표 1 — 대역 밖 do-no-harm
# ======================================================================================
def _tone(freq: float, samples: int, amp: float = 1.0) -> torch.Tensor:
    t = torch.arange(samples, dtype=torch.float32) / FS
    return amp * torch.sin(2.0 * torch.pi * freq * t)


def _dnh_criterion(**overrides) -> ANCLoss:
    cfg = _loss_cfg("trusted_band", lambda_dnh=1.0, **overrides)
    return ANCLoss(
        _identity_plant(), cfg, FS, trusted_band_hz=(150.0, 600.0)
    ).eval()


def test_do_no_harm_bands_are_derived_by_subtracting_the_trusted_band() -> None:
    """대역 목록을 손으로 적지 않는다 — 신뢰 대역을 빼서 만든다.

    S npz 의 consistency_band 가 [150,600] → [150,1600] 으로 넓어져도 여기서 고칠
    것이 없어야 한다. 리터럴 대역 목록은 **여섯 번째 복붙**이 된다 (발생기 A).
    """

    narrow = _dnh_criterion().do_no_harm
    assert narrow is not None
    lows = [b.band.as_tuple() for b in narrow.bands if b.band.hi_hz <= 150.0]
    # 88.4 Hz 는 G4 옥타브 125 의 하단 경계다. 손실 대역이 옥타브를 가로지르면
    # 대역별 상한이 옥타브 상한으로 합쳐지지 않는다 (dsp/do_no_harm.py 의 정리).
    assert lows == [
        (0.0, 20.0),
        (20.0, 80.0),
        (80.0, pytest.approx(88.388, abs=1e-2)),
        (pytest.approx(88.388, abs=1e-2), 150.0),
    ]
    assert all(b.band.lo_hz >= 600.0 for b in narrow.bands if b.band.lo_hz > 150.0)

    wide = ANCLoss(
        _identity_plant(),
        _loss_cfg("trusted_band", lambda_dnh=1.0),
        FS,
        trusted_band_hz=(150.0, 1000.0),
    ).eval().do_no_harm
    assert wide is not None
    # 신뢰 대역이 넓어지면 감시 대역이 자동으로 물러난다.
    assert all(
        hi <= 150.0 or lo >= 1000.0
        for lo, hi, _, _ in wide.as_tuples()
    )


def test_out_of_band_amplification_is_penalised() -> None:
    """결함 3 이 손실 안에서 비용을 갖는가 (실측 1633–6000Hz 최악 +19.9 dB)."""

    samples = FS
    d = (_tone(300.0, samples) + _tone(2000.0, samples, amp=0.01)).view(1, 1, -1)
    y = _tone(2000.0, samples, amp=0.1).view(1, 1, -1)

    criterion = _dnh_criterion()
    _, metrics = criterion(y, d, perturb={"jitter": 0})

    # 대역이 옥타브 경계(2828.4 Hz)에서 잘리므로 2000 Hz 순음은 [1633, 2828] 에 든다.
    assert metrics["dnh_1633_2828_max_db"] == pytest.approx(20.0, abs=0.5)
    assert metrics["dnh"] > 0.0

    off = ANCLoss(
        _identity_plant(),
        _loss_cfg("trusted_band", lambda_dnh=0.0),
        FS,
        trusted_band_hz=(150.0, 600.0),
    ).eval()
    loss_on, _ = criterion(y, d, perturb={"jitter": 0})
    loss_off, _ = off(y, d, perturb={"jitter": 0})
    assert float(loss_on) > float(loss_off)


def test_out_of_band_hinge_asks_for_silence_not_cancellation() -> None:
    """**단측 확인** — 신뢰 못 하는 대역에서 요구하는 것은 '상쇄'가 아니라 '침묵'이다.

    2026-08-06 에 계약이 바뀌었다. 마진이 게이트 임계에서 유도되면서(+6.0 → −18.27 dB)
    "대역 밖을 완벽히 상쇄하면 벌점 0" 이 성립하지 않는다. **성립할 수 없다** —

    이 항의 판정량은 ``bandpower(S·y)`` 이고 그것은 설계상 **∠S 를 쓰지 않는다**
    (비신뢰 대역에서 못 믿는 것이 바로 위상이기 때문이다). 위상을 모르면 "완벽 상쇄"와
    "완벽 증폭"은 같은 숫자다. 그리고 게이트는 최악값을 본다. 그래서 ``|S·y| ≈ |d|`` 는
    **동전 던지기**이고, 유도 마진은 그것을 벌한다. 옛 마진 +6.0 은 이 동전 던지기를
    공짜로 만들었고, 그 상태로 학습한 모델이 게이트를 8.5 dB 차이로 FAIL 했다.

    지켜야 하는 진짜 단측 성질은 남아 있고 여기서 강제한다:
    **최소는 침묵이고, 벌점은 |S·y| 에 대해 단조이며, 상쇄를 보상하지 않는다.**
    """

    samples = FS
    d_oob = _tone(2000.0, samples, amp=0.05)
    d = (_tone(300.0, samples) + d_oob).view(1, 1, -1)
    criterion = _dnh_criterion()

    quiet = torch.zeros(1, 1, samples)
    cancelling = (-d_oob).view(1, 1, -1)
    inverted = d_oob.view(1, 1, -1)  # 같은 크기, 반대 위상
    amplifying = (3.0 * d_oob).view(1, 1, -1)

    values = []
    for y in (quiet, cancelling, inverted, amplifying):
        s_y = criterion.plant(y, {"jitter": 0}).squeeze(1)
        value, _ = criterion._do_no_harm(s_y, d.squeeze(1))
        values.append(float(value))

    # (a) 침묵이 유일한 최소이고 정확히 0 이다.
    assert values[0] == 0.0
    # (b) 위상 무관 — 상쇄와 증폭이 크기가 같으면 값이 **같다**.
    assert values[1] == pytest.approx(values[2], rel=1e-6)
    # (c) 단조 — 크기를 키우면 벌점이 커진다.
    assert values[0] < values[1] < values[3]

    # (d) 침묵에서는 그래디언트가 정확히 0 이다 (relu 비활성 구간).
    y = quiet.clone().requires_grad_(True)
    s_y = criterion.plant(y, {"jitter": 0}).squeeze(1)
    value, _ = criterion._do_no_harm(s_y, d.squeeze(1))
    value.backward()
    assert float(y.grad.abs().max()) == 0.0

    # (e) 상쇄 방향의 그래디언트는 **출력을 줄이는 쪽**이지 늘리는 쪽이 아니다.
    y = cancelling.clone().requires_grad_(True)
    s_y = criterion.plant(y, {"jitter": 0}).squeeze(1)
    value, _ = criterion._do_no_harm(s_y, d.squeeze(1))
    value.backward()
    assert float((y.grad * y.detach()).sum()) > 0.0, (
        "그래디언트가 대역 밖 출력을 키우는 쪽을 가리킵니다 — 이 항은 상쇄를 "
        "요구해서는 안 됩니다"
    )


def test_out_of_band_hinge_ignores_bands_where_the_disturbance_is_absent() -> None:
    """교란이 없는 대역에서 **침묵한 모델이 벌점을 받지 않는다.**

    회귀 방어. 분자에 ``_EPS`` 를 더하면 ``S·y = 0`` 인 모델의 비율이 ``_EPS/_EPS =
    0 dB`` 가 되어 음수 마진을 넘는다. 마진이 +6.0 이던 시절에는 0 < 6 이라 가려져
    있었고, 게이트에서 마진을 유도하는 순간 드러났다 — 아무 소리도 내지 않는 모델이
    벌점 8.39 를 받았다.
    """

    samples = FS
    d = _tone(300.0, samples).view(1, 1, -1)  # 300 Hz 단일 순음 — 나머지 대역은 비어 있다
    criterion = _dnh_criterion()
    s_y = criterion.plant(torch.zeros(1, 1, samples), {"jitter": 0}).squeeze(1)
    value, metrics = criterion._do_no_harm(s_y, d.squeeze(1))
    assert float(value) == 0.0, f"침묵한 모델이 벌점 {float(value):.3f} 을 받았습니다"
    assert metrics["dnh_worst_db"] < -100.0


def test_do_no_harm_ignores_the_phase_of_the_secondary_path() -> None:
    """판정량이 bandpower(S·y) 라 ∠S 와 무관하다.

    비신뢰 대역에서 못 믿는 것은 **위상**이다. 부호를 뒤집어도 값이 같다는 것은
    이 항이 위상을 쓰지 않는다는 직접 증거다.
    """

    samples = FS
    d = (_tone(300.0, samples) + _tone(2000.0, samples, amp=0.01)).view(1, 1, -1)
    criterion = _dnh_criterion()
    s_y = criterion.plant(_tone(2000.0, samples, amp=0.1).view(1, 1, -1), {"jitter": 0})

    positive, _ = criterion._do_no_harm(s_y.squeeze(1), d.squeeze(1))
    negative, _ = criterion._do_no_harm(-s_y.squeeze(1), d.squeeze(1))
    assert float(positive) == pytest.approx(float(negative), abs=1e-9)


def test_do_no_harm_band_overlapping_the_trusted_band_is_rejected() -> None:
    """겹치면 양측 목표와 단측 힌지가 서로 상쇄하고, 지표로는 보이지 않는다."""

    with pytest.raises(ValueError, match="겹칩니다"):
        ANCLoss(
            _identity_plant(),
            _loss_cfg(
                "trusted_band", lambda_dnh=0.1, do_no_harm_bands=[[400.0, 800.0, 6.0, 1.0]]
            ),
            FS,
            trusted_band_hz=(150.0, 600.0),
        )


def test_do_no_harm_without_a_trusted_band_is_rejected() -> None:
    """'대역 밖'이 정의되지 않았는데 힌지를 거는 설정은 조용히 통과시키지 않는다."""

    with pytest.raises(ValueError, match="trusted_band_hz"):
        ANCLoss(_identity_plant(), _loss_cfg(lambda_dnh=0.12), FS)


# ======================================================================================
# 시간 국소성
# ======================================================================================
def test_frame_aggregation_sees_the_burst_that_the_segment_fft_hides() -> None:
    """1.5초 한 FFT 는 순간 증폭을 희석한다 (실측 전체 +0.00 vs 0.125s 프레임 +10.82 dB)."""

    samples = 8192
    d = _tone(300.0, samples).view(1, 1, -1)
    y = (-0.99 * d).clone()
    # 프레임 2([2048,4096), Hann 중심 3072)의 한가운데에 증폭 버스트를 놓는다.
    burst = slice(2560, 3584)
    y[..., burst] = d[..., burst]  # e = 2d → 그 구간만 +6 dB 증폭

    criterion = ANCLoss(
        _identity_plant(),
        _loss_cfg(
            "trusted_band",
            lambda_frame=1.0,
            nmse_frame_samples=2048,
            nmse_frame_hop=1024,
        ),
        FS,
        trusted_band_hz=(150.0, 600.0),
    ).eval()
    _, metrics = criterion(y, d, perturb={"jitter": 0})

    assert metrics["nmse_trusted_db"] < 0.0, "세그먼트 FFT 로는 '좋아 보인다'"
    assert metrics["nmse_trusted_worst_db"] < 0.0
    assert metrics["frame_worst_db"] > 0.0, "프레임으로 보면 증폭 버스트가 드러난다"
    # 세그먼트 지표가 감춘 크기 — 실측에서는 전체 +0.00 vs 프레임 +10.82 dB 였다.
    assert metrics["frame_worst_db"] > metrics["nmse_trusted_worst_db"] + 5.0

    # 버스트를 없애면 프레임 지표도 같이 내려간다 (항이 실제로 버스트를 보고 있다).
    _, quiet = criterion(-0.99 * d, d, perturb={"jitter": 0})
    assert quiet["frame_worst_db"] < -30.0
    assert quiet["frame"] < metrics["frame"]


def test_silent_frames_do_not_hijack_the_frame_cvar() -> None:
    """무음 프레임은 마이크 자기잡음이 지배해 CVaR 이 의미 없는 곳을 고른다."""

    samples = 8192
    d = _tone(300.0, samples).view(1, 1, -1).clone()
    d[..., : 2 * 1024] *= 1.0e-4  # 앞 두 프레임은 사실상 무음
    y = -0.99 * d

    criterion = ANCLoss(
        _identity_plant(),
        _loss_cfg(
            "trusted_band",
            lambda_frame=1.0,
            nmse_frame_samples=2048,
            nmse_frame_hop=1024,
            nmse_frame_silence_db=-40.0,
        ),
        FS,
        trusted_band_hz=(150.0, 600.0),
    ).eval()
    fr_db, valid = criterion._framed_band_nmse_db(
        (d + criterion.plant(y, {"jitter": 0})).squeeze(1), d.squeeze(1), (150.0, 600.0)
    )
    assert bool(valid.any())
    assert not bool(valid[0, 0]), "무음 프레임이 유효로 남으면 게이트가 느슨한 것이다"


# ======================================================================================
# MR-STFT 정규화
# ======================================================================================
def test_mrstft_normalisation_is_per_item_not_per_batch() -> None:
    """배치 전체 노름은 가장 큰 아이템이 항을 독식한다.

    실측 배치 d 레벨 편차 45~66 dB, top1 아이템이 배치 에너지의 17~46%, 하위 절반
    합계 2%. 아이템별 비율이 같으면 레벨을 바꿔도 항의 값이 변하면 안 된다.
    """

    samples = 4096
    base = _band_noise(2, samples, seed=3)
    gains = torch.tensor([-0.5, -0.99]).view(-1, 1, 1)

    def mrstft(scale0: float) -> float:
        d = base.clone()
        d[0] *= scale0
        y = d * gains
        criterion = ANCLoss(
            _identity_plant(),
            _loss_cfg("trusted_band", lambda_mrstft=1.0, band_weight="fullband"),
            FS,
            trusted_band_hz=(150.0, 600.0),
        ).eval()
        return criterion(y, d, perturb={"jitter": 0})[1]["mrstft"]

    quiet, loud = mrstft(1.0), mrstft(100.0)
    assert quiet == pytest.approx(loud, rel=1e-4), (
        "한 아이템의 레벨이 항의 값을 바꾸면 배치 전체 노름으로 정규화하고 있는 것이다"
    )


# ======================================================================================
# 죽은 항 처리
# ======================================================================================
def test_saturation_penalty_replaces_the_structurally_dead_clip_term() -> None:
    """구 clip 항은 상한이 4.0e−4 로 **고정**이었고 새 항은 그 벽이 없다."""

    limit, margin = 0.2, 2.0
    u = torch.full((1, 1, 64), 3.0 * limit, requires_grad=True)
    y = limit * torch.tanh(u / limit)

    criterion = ANCLoss(
        _identity_plant(),
        _loss_cfg("trusted_band", lambda_sat=1.0, sat_margin=margin),
        FS,
        trusted_band_hz=(150.0, 600.0),
        limiter_limit=limit,
    ).eval()
    l_sat, u_over_limit = criterion.saturation_penalty(y)

    # 죽은 항의 상한 — 리미터가 있는 한 절대 넘을 수 없던 값.
    dead_bound = (limit - 0.18) ** 2
    assert float(torch.relu(y.abs() - 0.18).pow(2).mean()) <= dead_bound
    assert float(l_sat) > 1000.0 * dead_bound

    assert float(u_over_limit.abs().max()) == pytest.approx(3.0, abs=1e-3)
    assert float(l_sat) == pytest.approx((3.0 - margin) ** 2, abs=1e-3)

    # 벌점이 리미터를 **통과해** 활성 u 에 단위 이득으로 닿는가: ∂l/∂u = 2(|u|/L − m)/L.
    l_sat.backward()
    assert float(u.grad.sum()) == pytest.approx(2.0 * (3.0 - margin) / limit, rel=1e-3)


def test_saturation_penalty_is_zero_inside_the_linear_region() -> None:
    """포화하지 않은 출력에는 비용이 없다 — 상쇄에 필요한 출력을 벌하지 않는다."""

    limit = 0.2
    u = torch.full((1, 1, 64), 1.5 * limit)
    y = limit * torch.tanh(u / limit)
    criterion = ANCLoss(
        _identity_plant(),
        _loss_cfg("trusted_band", lambda_sat=1.0, sat_margin=2.0),
        FS,
        trusted_band_hz=(150.0, 600.0),
        limiter_limit=limit,
    ).eval()
    l_sat, _ = criterion.saturation_penalty(y)
    assert float(l_sat) == 0.0


def test_limiter_limit_disagreement_is_rejected() -> None:
    """같은 물리량(리미터 한계)을 두 곳에서 정하면 실패한다.

    예전에는 ``loss.clip_margin`` 과 ``model.limiter.limit`` 이 서로 다른 두 곳에
    적힌 같은 물리량이었고, 부등식 하나가 대조의 전부였다.
    """

    with pytest.raises(ValueError, match="limiter"):
        ANCLoss(
            _identity_plant(),
            _loss_cfg("trusted_band", limiter_limit=0.5),
            FS,
            trusted_band_hz=(150.0, 600.0),
            limiter_limit=0.2,
        )


# ======================================================================================
# 설정 경계 검증
# ======================================================================================
def test_unknown_loss_key_is_rejected() -> None:
    """오타 난 키가 **조용히 무시**되면 설정이 죽은 채로 남는다."""

    with pytest.raises(Exception, match="lambda_mrstf"):
        LossConfig.parse({"lambda_mrstf": 1.0})


def test_deprecated_clip_keys_are_not_silently_ignored() -> None:
    """폐기된 키는 무시가 아니라 경고다 — 오래된 설정이 조용히 도는 것을 막는다."""

    with pytest.warns(DeprecationWarning, match="폐기"):
        LossConfig.parse({"lambda_clip": 1.0, "clip_margin": 0.18})


def test_invalid_cvar_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="nmse_cvar_q"):
        LossConfig.parse({"nmse_cvar_q": 0.0})
    with pytest.raises(ValueError, match="nmse_cvar_alpha"):
        LossConfig.parse({"nmse_cvar_alpha": 1.5})
    with pytest.raises(ValueError, match="nmse_cvar_min_k"):
        LossConfig.parse({"nmse_cvar_min_k": 0})
    with pytest.raises(ValueError, match="sat_margin"):
        LossConfig.parse({"sat_margin": 0.0})


def test_do_no_harm_bands_and_edges_cannot_both_be_given() -> None:
    """대역을 두 곳에서 유도하는 설정 자체를 거부한다 (발생기 A)."""

    with pytest.raises(ValueError, match="동시에 지정"):
        LossConfig.parse(
            {"do_no_harm_bands": [[20.0, 80.0, 6.0, 1.0]], "do_no_harm_edges_hz": [20.0, 80.0]}
        )


def test_shipped_training_configs_pass_the_loss_schema() -> None:
    """출하 설정이 스키마를 통과하는가 (오기각 방지)."""

    for name in ("train_pretrain.yaml", "train_pretrain_tiny_wide.yaml"):
        cfg = load_yaml(REPO_ROOT / "configs" / name)
        parsed = LossConfig.parse(cfg["loss"])
        assert parsed.nmse_cvar_alpha > 0.0
        assert parsed.lambda_dnh > 0.0
        assert parsed.band_weight == "trusted_only"


def test_trusted_only_band_weight_zeroes_everything_outside_the_trusted_band() -> None:
    """MR-STFT 는 양측 항이다 — 신뢰 대역 밖에 가중을 남기면 상쇄를 요구하게 된다."""

    w = band_weights(256, FS, "trusted_only", trusted_band_hz=(150.0, 600.0))
    freqs = torch.fft.rfftfreq(256) * FS
    inside = (freqs >= 150.0) & (freqs <= 600.0)
    assert float(w[inside].min()) == 1.0
    assert float(w[~inside].max()) == 0.0
    with pytest.raises(ValueError, match="trusted_band_hz"):
        band_weights(256, FS, "trusted_only")


# ======================================================================================
# loss_start_sample
# ======================================================================================
def test_loss_start_sample_drops_the_structurally_uncancellable_prefix() -> None:
    """앞구간은 y 가 무엇이든 e = d 라 상쇄량을 잴 수 없다.

    실측 하한(실측 d, trusted 150–600Hz, N=1721): mean −20.3 / CVaR10 −10.1 /
    worst −4.8 dB. CVaR 로 바꾸는 순간 이 최악 아이템에 그래디언트가 집중된다.
    """

    samples, prefix = 4096, 512
    d = _band_noise(4, samples, seed=5)
    polluted = d.clone()
    polluted[..., :prefix] *= 50.0  # 앞구간에만 큰 신호

    criterion = ANCLoss(
        _identity_plant(),
        _loss_cfg("trusted_band"),
        FS,
        trusted_band_hz=(150.0, 600.0),
    ).eval()
    y = -0.9 * d

    clean_skip = criterion(y, d, loss_start_sample=prefix, perturb={"jitter": 0})[1]
    dirty_skip = criterion(y, polluted, loss_start_sample=prefix, perturb={"jitter": 0})[1]
    dirty_none = criterion(y, polluted, loss_start_sample=0, perturb={"jitter": 0})[1]

    assert dirty_skip["nmse_db"] == pytest.approx(clean_skip["nmse_db"], abs=1e-4)
    assert dirty_none["nmse_db"] > clean_skip["nmse_db"] + 1.0


def test_do_no_harm_covers_the_whole_complement_of_the_trusted_band() -> None:
    """감시하지 않는 주파수 구간이 하나라도 남으면 거기서 증폭은 **공짜**다.

    결함 3(2–8 kHz 를 15–22 dB 증폭)이 손실 안에서 비용이 0 이었던 이유가 정확히
    그것이다. 경계 목록은 여집합을 어디서 자를지만 정하고, 덮는 범위는 항상
    ``[0, Nyquist] − 신뢰대역`` 전체여야 한다.
    """

    plan = _dnh_criterion().do_no_harm
    assert plan is not None
    covered = sorted(band.band.as_tuple() for band in plan.bands)
    # 인접 구간이 빈틈 없이 이어지는가.
    assert covered[0][0] == 0.0
    assert covered[-1][1] == pytest.approx(FS / 2.0)
    for (_, hi), (lo, _) in zip(covered, covered[1:]):
        assert lo == pytest.approx(hi) or (hi, lo) == (150.0, 600.0)
    # 신뢰 대역만 구멍으로 남는다.
    holes = [
        (hi, lo) for (_, hi), (lo, _) in zip(covered, covered[1:]) if lo > hi
    ]
    assert holes == [(150.0, 600.0)]
