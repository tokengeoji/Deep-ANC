"""λ 가 항끼리의 그래디언트 예산 밖으로 **조용히 흘러가지 못하게** 한다.

왜 이 파일이 있는가
------------------
2026-08-06 감사: "λ_dnh 가 새 대역 구성에서 재교정되지 않았다 — 실측 그래디언트 비
**1333%**(설계 목표 20~40%)". 그런데 그 1333% 를 만든 계산은 저장소 어디에도 없었다.
숫자만 문서에 남고 재현 수단이 없으면, 다음 사람은 그 숫자를 믿거나 무시하는 것 말고
할 수 있는 일이 없다.

이제 측정은 :meth:`ANCLoss.gradient_budget` 하나이고(:meth:`forward` 와 **같은 항
dict** 를 쓴다), 이 파일이 출하 설정을 그 측정에 걸어 둔다. λ 가 예산을 벗어나면
학습을 돌리기 전에 여기서 실패한다.

2026-08-06 에 무엇이 바뀌었나
----------------------------
do-no-harm 마진이 ``+6.0`` 에서 게이트 유도값 ``−18.27 dB`` 로 24 dB 조여졌다(커밋
83c6954). 힌지가 훨씬 자주 활성화되므로 같은 λ 의 예산 몫이 그만큼 커진다. 실측::

    λ_dnh = 0.12 (옛 출하값)  →  예산비 41.8   ← 목적함수를 42배로 덮는다
λ_dnh = 0.00025           →  strict-S fixture 예산비 0.088 (승인 하한 미달)
λ_dnh = 0.00075           →  strict-S fixture 예산비 0.264 (현행 출하값)

같은 측정을 실제 체크포인트(``pretrain_tiny_corrected``)의 출력으로도 했고 0.394 로
일치했다. 즉 아래 픽스처(실측 S(z) + 합성 y)는 실기 y 를 대표한다.

⚠ 아직 검증되지 않은 것 두 가지 — 숨기지 않는다
------------------------------------------------
1. **"예산 0.2~0.4" 라는 규칙이 이 항에 맞는가.** do-no-harm 은 목적항이 아니라
   **제약**이다. 제약을 벌점으로 강제할 때 적정 λ 는 "예산 몫" 이 아니라 "실행 가능성을
   만드는 최소값" 으로 정해야 할 수 있다. λ 를 크게 두면 학습 초기에 y→0 으로 무너질
   위험이 있고(그러면 상쇄가 0 이다), 작게 두면 G4 를 못 넘을 위험이 있다. 20k step
   ablation 이 필요하고, 그것은 GPU 가 있는 곳에서만 할 수 있다.
2. **``frame`` 항의 몫이 0.905 로 목표 위에 있다.** λ_frame=0.5 에서 측정된 값이다.
   이번에 함께 바꾸지 않았다 — 검증 없이 두 축을 동시에 움직이면 ablation 이 무엇을
   말하는지 알 수 없기 때문이다. 같은 ablation 에서 함께 본다.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from deep_anc.config import REPO_ROOT
from deep_anc.dsp.secondary_path import DifferentiableSecondaryPath, load_secondary_path
from deep_anc.losses.anc_loss import ANCLoss

FS = 48_000
SAMPLES = 1 << 14
BATCH = 4
TRUSTED = (150.0, 1600.0)

# 설계 목표 — 보조항은 목적함수의 0.2~0.4 배 몫을 갖는다.
BUDGET_LO, BUDGET_HI = 0.2, 0.4


def _plant() -> DifferentiableSecondaryPath:
    """**실측 S(z)** 를 쓴다. 합성 FIR 을 쓰면 대표성이 사라진다.

    난수 FIR 로 만든 플랜트는 같은 y 에 대해 대역 밖으로 훨씬 많이 퍼뜨렸다 —
    같은 λ 에서 예산비가 4.05 vs 실측 0.39 로 10배 차이가 났다. 그 픽스처로 λ 를
    고르면 실기에서 10배 틀린 값을 쓰게 된다. 플랜트가 없으면 판정하지 않는다
    (없는 것을 통과로 세지 않는다).
    """

    npz = REPO_ROOT / "assets" / "measured" / "secondary_path_il.npz"
    if not npz.is_file():
        pytest.skip(f"실측 S(z) 아티팩트가 없습니다: {npz}")
    return DifferentiableSecondaryPath(load_secondary_path(npz), handoff_extra_samples=256)


def _band_signal(lo: float, hi: float, seed: int) -> torch.Tensor:
    freqs = np.fft.rfftfreq(SAMPLES, 1.0 / FS)
    rng = np.random.default_rng(seed)
    spec = np.zeros((BATCH, freqs.size), dtype=complex)
    mask = (freqs >= lo) & (freqs <= hi)
    spec[:, mask] = rng.normal(size=(BATCH, int(mask.sum()))) + 1j * rng.normal(
        size=(BATCH, int(mask.sum()))
    )
    wave = np.fft.irfft(spec, n=SAMPLES)
    wave = 0.05 * wave / (np.abs(wave).max(axis=-1, keepdims=True) + 1e-12)
    return torch.from_numpy(wave).float().unsqueeze(1)


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    """학습 중간을 닮은 (y, d). ``y=0`` 이면 dnh 가 0 이라 예산이 정의되지 않는다."""

    d = _band_signal(20.0, 8000.0, seed=1)
    y = -0.7 * _band_signal(TRUSTED[0], TRUSTED[1], seed=2)
    return y, d


def _budget(lambda_dnh: float) -> dict[str, float]:
    cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "train_finetune.yaml").read_text(encoding="utf-8")
    )
    loss_cfg = dict(cfg["loss"])
    loss_cfg["lambda_dnh"] = float(lambda_dnh)
    criterion = ANCLoss(_plant(), loss_cfg, FS, trusted_band_hz=TRUSTED).eval()
    y, d = _fixture()
    return criterion.gradient_budget(y, d, perturb={"jitter": 0})


# --------------------------------------------------------------------------- 양성 대조
def test_shipped_lambda_dnh_is_the_calibrated_non_swamping_value() -> None:
    cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "train_finetune.yaml").read_text(encoding="utf-8")
    )
    shipped = float(cfg["loss"]["lambda_dnh"])
    # 이 fixture는 역사적 고정 파형 smoke이고 실제 대표 학습 출력과 같은
    # 출력 분포를 보장하지 않는다. 실행별 strict-S 측정은 campaign ledger가
    # 검증한다. 여기서는 재교정값이 legacy 0.001로 되돌아가 목적함수를
    # 덮는 회귀와 0/극소값을 잡는다.
    assert shipped == pytest.approx(0.00075)
    share = _budget(shipped)["dnh"]
    assert 0.0 < share < BUDGET_HI


def test_the_nmse_term_is_the_denominator_and_is_alive() -> None:
    budget = _budget(0.001)
    assert budget["nmse"] == pytest.approx(1.0)
    # 보조항이 전부 죽어 있으면 예산 자체가 무의미하다.
    assert budget["mrstft"] > 0.0 and budget["frame"] > 0.0


def test_other_terms_are_recorded_so_a_silent_drift_is_visible() -> None:
    """dnh 말고 다른 항이 목적함수를 덮는 것도 잡는다.

    ``frame`` 은 현재 0.905 로 목표(0.2~0.4) 위에 있다. 이번에 함께 바꾸지 않은 이유는
    docstring 에 적었다 — 여기서는 **그 사실이 조용해지지 않게** 상한만 걸어 둔다.
    3.0 을 넘으면 그 항이 목적함수를 덮는 수준이므로 멈춘다.
    """

    budget = _budget(0.001)
    for name in ("mrstft", "frame", "sat"):
        assert budget[name] < 3.0, (
            f"{name} 항의 몫이 {budget[name]:.2f} 입니다 — 목적함수를 덮습니다"
        )
    assert budget["frame"] == pytest.approx(0.905, abs=0.15), (
        f"frame 몫이 {budget['frame']:.3f} 로 바뀌었습니다 — λ_frame 을 건드렸다면 "
        "docstring 의 미검증 항목을 갱신하세요"
    )


# --------------------------------------------------------------------------- 음성 대조
def test_the_old_lambda_would_swamp_the_objective() -> None:
    """마진을 조인 뒤 옛 λ=0.12 가 목적함수를 몇 배로 덮는지 **숫자로** 고정한다.

    이 테스트가 없으면 위 테스트는 "λ 가 목표 안에 있다" 만 말하고, **왜 바꿔야 했는지**
    를 잃는다. 그리고 누가 λ 를 0.12 로 되돌리면 조용히 지나간다.
    """

    share = _budget(0.12)["dnh"]
    assert share > 10.0, (
        f"옛 λ=0.12 의 몫이 {share:.2f} 입니다 — 10 을 넘지 않으면 이 픽스처가 힌지를 "
        "제대로 활성화하지 못한 것이고, 그러면 위 테스트도 의미가 없습니다"
    )


def test_zero_lambda_kills_the_term_entirely() -> None:
    assert _budget(0.0)["dnh"] == 0.0


def test_budget_scales_linearly_with_lambda() -> None:
    """예산비가 λ 에 선형이어야 한다 — 아니면 측정이 항을 잘못 잡고 있는 것이다."""

    a, b = _budget(0.001)["dnh"], _budget(0.002)["dnh"]
    assert b == pytest.approx(2.0 * a, rel=1e-3)
