"""ANC 학습 손실 — 미분가능 플랜트를 통과한 에러신호를 직접 최소화.

    y = model(x)  →  y_nl = G_nl(y)  →  e = d + S(y_nl)

    L = A[NMSE_trusted(dB)]                      절대목표 2: 최악값
      + λ_dnh · Σ_b w_b · A[relu(대역밖 증폭_b − margin_b)]   절대목표 1: 악화 금지
      + λ_frame · A[프레임별 NMSE_trusted(dB)]   절대목표 2: 시간 국소성
      + λ_mrstft · A[아이템별 MR-STFT]
      + λ_sat · 리미터 이전 활성 포화 벌점

여기서 ``A[·]`` 는 **평균이 아니라 (평균, CVaR) 혼합** 집계다.

왜 이렇게 바뀌었나 (2026-08-05 실측)
-----------------------------------
1. **집계가 최악값과 반대 방향이었다.** ``nmse_trusted_db.mean()`` 의 아이템별
   그래디언트는 잔차 RMS 에 **반비례**한다 (log-log 회귀 기울기 −0.94(synth) /
   −1.02(recorded), corr −0.99). 최악 4개가 그래디언트의 3.4%, 최상 4개가 16.0% 를
   가져갔다. 그래서 "trusted 평균 −19 dB" 인 모델이 96개 중 8개를 **증폭**하고
   CVaR25 는 −0.03 dB 였다. CVaR25 로 바꾸면 배분이 43.4% / 0.0% 로 뒤집힌다.
2. **대역 밖 악화를 막는 항이 아예 없었다.** 반노이즈 파워가 원소음 대비 서브소닉
   +30.5 dB, 600–1000Hz +25.2 dB, 1633–6000Hz +19.9 dB 까지 올라가는데 손실에는
   비용이 0 이었다 (결함 3 = 절대목표 1 정면 위반). 게이트에만 있었다.
3. **λ_pow / λ_clip 은 구조적으로 죽어 있었다.** 모델이 ``y = L·tanh(u/L)`` 라
   ``relu(|y|−0.18)²`` 의 **상한이 4.0e−4 로 고정**이다. 실측 그래디언트 기여
   1.1e−8 / 2.3e−9. 진짜 위험인 리미터 이전 활성 포화(복원 |u| = 7.8·L →
   tanh′ ≈ 1e−5)는 아무도 안 보고 있었다.
4. **MR-STFT 가 배치 전체 노름으로 정규화**돼 있었다. 배치 안 d 레벨 편차가 실측
   45~66 dB 라 한 아이템이 항의 17~46% 를 독식한다.
5. **1.5초 한 FFT 가 트랜지언트를 가렸다.** 전체구간 최악 +0.00 dB 인 같은 배치가
   0.125초 프레임에서는 최악 **+10.82 dB** 였다.

신뢰 대역과 그 밖의 비대칭 (설계의 핵심)
--------------------------------------
* **신뢰 대역 안** — 양측. "줄여라". S(z) 의 크기와 **위상**을 둘 다 믿는다.
* **신뢰 대역 밖** — 단측. "키우지 마라". 판정량이 ``bandpower(S·y)`` 라 ∠S 와
  무관하고, ``relu`` 라 '상쇄하라'는 그래디언트를 **절대** 만들지 않는다.
  |S| 오차는 힌지 임계만 옮기며, 그 대역의 정답 출력이 0 이므로 과벌점은 항상
  안전한 방향이다.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..dsp.nonlinear import RandomNonlinear
from ..dsp.secondary_path import DifferentiableSecondaryPath
# 대역 산술의 단일 출처. 예전에는 이 파일과 eval/metrics.py 에 같은 함수가 한 벌씩
# 있었고 입력 처리와 오류 메시지가 이미 갈라져 있었다 (발생기 A).
from ..dsp.timing import FrequencyBand, intersect_frequency_bands
from .config import DoNoHarmPlan, LossConfig

_EPS = 1.0e-10

_DNH_LOG_FLOOR = 1.0e-30
"""do-no-harm 비율의 로그 하한 (−300 dB). **분자에 더하지 않기 위한 장치다.**

분자에 ``_EPS`` 를 더하면 ``S·y = 0`` 인 모델의 비율이 ``_EPS/_EPS = 0 dB`` 가 되어
힌지 마진(−18.27 dB)을 넘고, **아무 소리도 내지 않는 모델이 벌점을 받는다.** 상대비에만
하한을 걸면 ``num=0 → −300 dB → relu 비활성 → 벌점 0·그래디언트 0`` 이 정확히 성립한다.
"""


def band_weights(
    fft_size: int,
    sample_rate: int,
    scheme: str,
    cutoff_hz: float = 1633.0,
    target_band_hz: tuple[float, float] = (80.0, 1000.0),
    *,
    trusted_band_hz: tuple[float, float] | None = None,
) -> torch.Tensor:
    """rfft 빈별 가중 벡터. 목표 대역은 duct.yaml acoustics.realistic_target_band_hz
    가 단일 출처다 (감사 L9 — trainer 가 주입)."""
    freqs = torch.fft.rfftfreq(fft_size) * sample_rate
    w = torch.ones_like(freqs)
    lo, hi = float(target_band_hz[0]), float(target_band_hz[1])
    if scheme == "curriculum_a":
        w = torch.where((freqs >= lo) & (freqs <= hi), torch.full_like(w, 3.0), w)
        w = torch.where(freqs > cutoff_hz, torch.full_like(w, 0.25), w)
        w = torch.where(freqs < 40.0, torch.full_like(w, 0.1), w)
    elif scheme == "fullband":
        w = torch.where(freqs < 20.0, torch.full_like(w, 0.1), w)
    elif scheme == "trusted_only":
        # MR-STFT 는 **양측** 항이다. 신뢰 대역 밖에 가중을 남기면 S(z) 의 위상을
        # 못 믿는 곳에서 상쇄를 요구하게 되고, 그 잘못된 위상이 신뢰 대역 성능까지
        # 갉아먹는다. 대역 밖 압력은 전부 단측 do-no-harm 힌지가 담당한다.
        if trusted_band_hz is None:
            raise ValueError("band_weight=trusted_only 는 trusted_band_hz 가 필요합니다")
        t_lo, t_hi = (float(v) for v in trusted_band_hz)
        w = torch.where(
            (freqs >= t_lo) & (freqs <= t_hi), torch.ones_like(w), torch.zeros_like(w)
        )
    else:
        raise ValueError(f"알 수 없는 band_weight: {scheme}")
    return w


def _stft_mag(x: torch.Tensor, fft_size: int, hop: int, window: torch.Tensor) -> torch.Tensor:
    """x: [B, T] → |STFT| [B, F, frames]."""
    spec = torch.stft(
        x,
        n_fft=fft_size,
        hop_length=hop,
        win_length=fft_size,
        window=window,
        center=True,
        return_complex=True,
    )
    return spec.abs()


class ANCLoss(nn.Module):
    def __init__(
        self,
        plant: DifferentiableSecondaryPath,
        loss_cfg: dict,
        sample_rate: int,
        nonlinear: RandomNonlinear | None = None,
        cutoff_hz: float = 1633.0,
        target_band_hz: tuple[float, float] = (80.0, 1000.0),
        trusted_band_hz: tuple[float, float] | None = None,
        limiter_limit: float | None = None,
    ) -> None:
        super().__init__()
        self.plant = plant
        self.nonlinear = nonlinear
        self.sample_rate = sample_rate
        # 설정 검증은 여기 한 번뿐이다. 모르는 키는 여기서 죽는다 — 조용히 무시되는
        # 설정이 다음 사람을 속이는 것이 이 저장소의 반복 결함이었다.
        cfg = LossConfig.parse(loss_cfg)
        self.loss_config = cfg

        self.ffts = [int(v) for v in cfg.mrstft_ffts]
        self.lambda_mrstft = float(cfg.lambda_mrstft)
        self.lambda_pow = float(cfg.lambda_pow)
        self.lambda_dnh = float(cfg.lambda_dnh)
        self.lambda_frame = float(cfg.lambda_frame)
        self.lambda_sat = float(cfg.lambda_sat)
        self.sat_margin = float(cfg.sat_margin)
        self.sat_ratio_eps = float(cfg.sat_ratio_eps)

        # 리미터 한계는 **모델이 단일 출처**다. 예전에는 loss.clip_margin 과
        # model.limiter.limit 이 서로 다른 두 곳에 적힌 같은 물리량이었다.
        if limiter_limit is not None and cfg.limiter_limit is not None:
            if abs(float(limiter_limit) - float(cfg.limiter_limit)) > 1.0e-12:
                raise ValueError(
                    f"loss.limiter_limit({cfg.limiter_limit}) 이 모델 limiter.limit"
                    f"({limiter_limit}) 과 다릅니다 — 같은 물리량을 두 곳에서 정하지 "
                    "마세요"
                )
        resolved_limit = (
            float(limiter_limit)
            if limiter_limit is not None
            else (0.2 if cfg.limiter_limit is None else float(cfg.limiter_limit))
        )
        if not math.isfinite(resolved_limit) or resolved_limit <= 0.0:
            raise ValueError(f"limiter limit 은 유한한 양수여야 합니다: {resolved_limit}")
        self.limit = resolved_limit

        # ---- 최악값 집계 ----
        # dB 산술평균 = 비율의 기하평균이고, 실측상 ∂/∂y ∝ 1/잔차RMS 라 이미 −40 dB 인
        # 아이템이 그래디언트를 독식한다. 절대목표 2 는 최악값 문제이므로 뒤집는다.
        # q=0.25 근거: 평가 G4 는 worst10(=CVaR@10%)을 본다 → 학습은 그보다 약간 넓게
        # 잡아 게이트를 독립 검증으로 남긴다. 실패 질량 실측(synth val 96개): 하위 25%
        # 가 ≥ −0.03 dB, recorded 는 50% 가 ≥ −0.1 dB. q=0.10 은 과집중, q=0.50 은 평균 회귀.
        self.nmse_cvar_q = float(cfg.nmse_cvar_q)
        self.nmse_cvar_alpha = float(cfg.nmse_cvar_alpha)
        self.nmse_cvar_min_k = int(cfg.nmse_cvar_min_k)

        # 직접 ANCLoss를 생성하는 기존 코드는 trusted band를 넘기지 않으므로
        # 기존 fullband 동작을 유지한다. Trainer는 항상 실측∩목표 대역을 주입한다.
        default_objective = "trusted_band" if trusted_band_hz is not None else "fullband"
        self.nmse_objective = str(cfg.nmse_objective or default_objective)
        if trusted_band_hz is None:
            if self.nmse_objective == "trusted_band":
                raise ValueError("nmse_objective=trusted_band 이면 trusted_band_hz가 필요합니다")
            self.trusted_band_hz = None
        else:
            self.trusted_band_hz = intersect_frequency_bands(
                trusted_band_hz,
                trusted_band_hz,
                sample_rate / 2.0,
            )

        # ---- 대역 밖 악화 금지 ----
        # 대역 목록을 리터럴로 박지 않는다. 보호 대역(= 개선을 요구하는 대역)을 빼서
        # 만들기 때문에 S npz 의 신뢰대역이 [150,600] → [150,1600] 으로 넓어져도
        # 여기서 고칠 것이 없고, 겹침이 **구조적으로 불가능**하다.
        self.do_no_harm: DoNoHarmPlan | None = None
        self.dnh_band_floor = 10.0 ** (float(cfg.dnh_band_floor_db) / 10.0)
        if self.lambda_dnh > 0.0:
            if self.trusted_band_hz is None:
                raise ValueError(
                    "lambda_dnh > 0 이면 trusted_band_hz 가 필요합니다 — 무엇이 "
                    "'대역 밖'인지 정의되지 않으면 힌지를 걸 수 없습니다"
                )
            self.do_no_harm = cfg.resolve_do_no_harm(
                protected=FrequencyBand(
                    lo_hz=self.trusted_band_hz[0], hi_hz=self.trusted_band_hz[1]
                ),
                nyquist_hz=sample_rate / 2.0,
            )

        # ---- 프레임 집계 ----
        self.frame_samples = int(cfg.nmse_frame_samples)
        self.frame_hop = int(cfg.frame_hop_samples())
        self.frame_silence_db = float(cfg.nmse_frame_silence_db)
        if self.lambda_frame > 0.0 and self.trusted_band_hz is None:
            raise ValueError("lambda_frame > 0 이면 trusted_band_hz 가 필요합니다")
        # λ=0 후보도 frame metric은 계속 산출한다. 버퍼를 λ 조건에 묶으면
        # metric-only 모드가 첫 forward에서 AttributeError로 죽어 관측 자체가 불가능하다.
        if self.trusted_band_hz is not None:
            self.register_buffer(
                "frame_win", torch.hann_window(self.frame_samples), persistent=False
            )

        scheme = str(cfg.band_weight)
        for fft_size in self.ffts:
            self.register_buffer(
                f"w_{fft_size}",
                band_weights(
                    fft_size,
                    sample_rate,
                    scheme,
                    cutoff_hz,
                    target_band_hz,
                    trusted_band_hz=self.trusted_band_hz,
                ),
                persistent=False,
            )
            self.register_buffer(
                f"win_{fft_size}", torch.hann_window(fft_size), persistent=False
            )

    # ------------------------------------------------------------------ 집계
    def _cvar(self, values: torch.Tensor) -> torch.Tensor:
        """상위 q 분위(=최악) 평균.

        ``k`` 에 하한을 두어 소배치/DDP 에서 유효 배치가 너무 줄어드는 것을 막는다.
        **주의: DDP 에서 topk 는 랭크 로컬이다.** world>1 이면 각 랭크의 상위 q 합집합이
        글로벌 상위 q 를 덮으므로 실효 분위가 조금 넓어진다. cfg_snapshot 에 그 사실을
        남긴다.
        """

        flat = values.reshape(-1)
        n = flat.numel()
        if n == 0:
            return values.new_zeros(())
        k = min(n, max(self.nmse_cvar_min_k, int(math.ceil(self.nmse_cvar_q * n))))
        return flat.topk(k).values.mean()

    def _worst_aggregate(self, values: torch.Tensor) -> torch.Tensor:
        """평균 + CVaR 혼합. ``alpha=0`` 이면 기존 산술평균과 **완전히 동일**하다.

        ``alpha=0.7`` 근거: 순수 CVaR(alpha=1)은 (1−q) 아이템의 그래디언트를 0 으로
        만들어 유효 배치를 1/4 로 줄인다. 실측 ||∇|| 이 mean 24.97 vs CVaR25 17.10 으로
        비슷해 혼합해도 LR 재조정이 필요 없다.
        """

        if values.numel() == 0:
            return values.new_zeros(())
        if self.nmse_cvar_alpha <= 0.0:
            return values.mean()
        return (
            (1.0 - self.nmse_cvar_alpha) * values.mean()
            + self.nmse_cvar_alpha * self._cvar(values)
        )

    # ------------------------------------------------------------------ 대역
    def _band_bins(self, samples: int, lo_hz: float, hi_hz: float) -> tuple[int, int]:
        lo_bin = max(0, int(math.ceil(lo_hz * samples / self.sample_rate)))
        hi_bin = min(samples // 2, int(math.floor(hi_hz * samples / self.sample_rate)))
        return lo_bin, hi_bin

    def _band_nmse_db(
        self,
        e: torch.Tensor,
        d: torch.Tensor,
        band_hz: tuple[float, float],
        *,
        include_upper: bool = True,
    ) -> torch.Tensor:
        """e/d [B,T]의 주어진 대역 NMSE [B]를 미분가능하게 계산."""
        samples = e.shape[-1]
        if samples < 2:
            raise ValueError(f"NMSE FFT에 필요한 샘플이 부족합니다: {samples}")
        lo, hi = band_hz
        lo_bin, hi_bin = self._band_bins(samples, lo, hi)
        if not include_upper:
            # [lo, hi) 규약. hi가 FFT bin 사이면 floor와 같고, 정확히 bin이면 그
            # 경계 bin을 다음 인접 subband에만 귀속한다. 기본값은 기존 Stage-1의
            # 양끝 포함 동작을 그대로 보존한다.
            hi_bin = min(
                hi_bin,
                int(math.ceil(float(hi) * samples / self.sample_rate)) - 1,
            )
        if lo_bin > hi_bin:
            raise ValueError(
                f"세그먼트 {samples}샘플 FFT에 trusted band {band_hz} bin이 없습니다"
            )

        E = torch.fft.rfft(e, dim=-1, norm="ortho")[..., lo_bin : hi_bin + 1]
        D = torch.fft.rfft(d, dim=-1, norm="ortho")[..., lo_bin : hi_bin + 1]
        e_pow = E.real.square() + E.imag.square()
        d_pow = D.real.square() + D.imag.square()

        # one-sided FFT Parseval 가중치. DC/Nyquist 외 bin은 음수 주파수와
        # 짝이 있으므로 2배한다. 대역 비율의 물리적 에너지를 유지한다.
        weights = torch.full(
            (hi_bin - lo_bin + 1,),
            2.0,
            dtype=e_pow.dtype,
            device=e.device,
        )
        if lo_bin == 0:
            weights[0] = 1.0
        if samples % 2 == 0 and hi_bin == samples // 2:
            weights[-1] = 1.0
        e_band_pow = (e_pow * weights).sum(dim=-1)
        d_band_pow = (d_pow * weights).sum(dim=-1)
        return 10.0 * torch.log10((e_band_pow + _EPS) / (d_band_pow + _EPS))

    def _main_nmse_objective(
        self,
        e: torch.Tensor,
        d: torch.Tensor,
        nmse_fullband_db: torch.Tensor,
        nmse_trusted_db: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """주 목적 NMSE를 반환하는 확장 지점.

        Stage-1 :class:`ANCLoss`의 계산은 기존과 바이트 단위로 같은 수식이다. 최종
        광대역 역할은 이 메서드만 재정의해 일곱 subband를 독립 정규화한다. 이렇게
        분리하면 기존 150--1600 Hz 설정에 광대역 키를 섞거나, 광대역 구현을 위해
        :meth:`_forward_fp32` 전체를 복제할 필요가 없다.
        """

        del e, d  # 기본 구현은 위에서 계산한 연속 대역/fullband 값만 사용한다.
        if self.nmse_objective == "trusted_band":
            assert nmse_trusted_db is not None
            return self._worst_aggregate(nmse_trusted_db), {}
        return self._worst_aggregate(nmse_fullband_db), {}

    def _dnh_band_bins(self, samples: int, lo_hz: float, hi_hz: float) -> tuple[int, int]:
        """DNH bin 범위 확장 지점. Stage-1은 기존 양끝 포함 규약을 유지한다."""

        return self._band_bins(samples, lo_hz, hi_hz)

    # -------------------------------------------------------- 대역 밖 do-no-harm
    def _do_no_harm(
        self, s_y: torch.Tensor, d: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """대역 밖 '악화 금지' 단측 힌지 (s_y, d: [B, T]).

        1) 판정량이 ``bandpower(S·y) = Σ|S(f)|²|Y(f)|²`` 라 **S 의 위상과 무관**하다.
           비신뢰 대역에서 못 믿는 것은 위상이므로, 그나마 믿을 수 있는 크기응답만 쓴다.
           ``e = d + S·y`` 기반 힌지는 위상 오차가 그대로 들어온다.
        2) 단측(relu)이라 '그 대역을 상쇄하라'는 그래디언트를 **절대** 만들지 않는다.
           오직 '반노이즈를 그 대역에 쏟아붓지 마라' 만 요구한다.
        3) dB 힌지를 쓴다. ratio 힌지는 스케일이 레짐에 따라 14배 흔들려(λ 7.5e−3 vs
           1.06e−1) 단일 λ 가 성립하지 않는다. dB 힌지는 1.3배(0.110 vs 0.140)다.
        """

        plan = self.do_no_harm
        assert plan is not None
        samples = s_y.shape[-1]
        SY = torch.fft.rfft(s_y, dim=-1, norm="ortho")
        D = torch.fft.rfft(d, dim=-1, norm="ortho")
        sy_pow = SY.real.square() + SY.imag.square()
        d_pow = D.real.square() + D.imag.square()
        # 분모 하한 — 교란이 사실상 없는 대역에서는 비율이 정의되지 않는다. 이것이
        # 없으면 ``_EPS/_EPS = 0 dB`` 가 되어 **아무 소리도 내지 않는 모델이 벌점을
        # 받는다** (마진이 양수(+6.0)일 때는 0 dB < 6.0 이라 가려져 있었고, 게이트에서
        # 유도한 음수 마진으로 바꾸는 순간 드러났다). 하한은 신호 자신의 전체 전력에
        # 상대적이라 스케일 불변이다.
        floor = self.dnh_band_floor * d_pow.sum(dim=-1)

        total = s_y.new_zeros(())
        metrics: dict[str, float] = {}
        worst = -math.inf
        for item in plan.bands:
            lo, hi = item.band.as_tuple()
            lo_bin, hi_bin = self._dnh_band_bins(samples, lo, hi)
            if lo_bin > hi_bin:
                continue
            num = sy_pow[..., lo_bin : hi_bin + 1].sum(dim=-1)
            den = torch.maximum(d_pow[..., lo_bin : hi_bin + 1].sum(dim=-1), floor)
            # num 에 EPS 를 더하지 않는다 — 출력이 정확히 0 이면 벌점도 정확히 0 이어야
            # 한다. 로그 하한은 상대비에만 걸어 그 성질을 지킨다.
            ratio_db = 10.0 * torch.log10(num / (den + _EPS) + _DNH_LOG_FLOOR)
            # 최악값 집계 — 대역 밖도 평균은 음수인데 최악이 +30 dB 다.
            total = total + float(item.weight) * self._worst_aggregate(
                F.relu(ratio_db - float(item.margin_db))
            )
            band_max = float(ratio_db.max().detach())
            metrics[f"dnh_{int(lo)}_{int(hi)}_max_db"] = band_max
            worst = max(worst, band_max)
        if math.isfinite(worst):
            metrics["dnh_worst_db"] = worst
        return total, metrics

    # ------------------------------------------------------------ 프레임 국소성
    def _framed_band_nmse_db(
        self, e: torch.Tensor, d: torch.Tensor, band_hz: tuple[float, float]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """프레임별 대역 NMSE [B, frames] 와 유효 마스크.

        1.5초를 한 FFT 로 재면 트랜지언트가 희석된다 — 실측 전체구간 최악 +0.00 dB 인
        같은 배치가 0.125초 프레임에서는 최악 +10.82 dB 였다.
        프레임 8192(48kHz 에서 170 ms)는 150–600Hz 에 77 bin 을 준다(bin 5.86 Hz).
        """

        win = self.frame_win.to(dtype=e.dtype, device=e.device)
        n = self.frame_samples
        E = torch.stft(
            e, n, self.frame_hop, n, win, center=False, return_complex=True
        )
        D = torch.stft(
            d, n, self.frame_hop, n, win, center=False, return_complex=True
        )
        lo_bin, hi_bin = self._band_bins(n, band_hz[0], band_hz[1])
        if lo_bin > hi_bin:
            raise ValueError(
                f"프레임 {n}샘플 FFT 에 trusted band {band_hz} bin 이 없습니다"
            )
        e_pow = E[:, lo_bin : hi_bin + 1].abs().square().sum(dim=1)
        d_pow = D[:, lo_bin : hi_bin + 1].abs().square().sum(dim=1)
        # 무음 프레임 배제 — 그 프레임의 d 대역파워가 아이템 최대 대비 −40 dB 미만이면
        # 비율이 마이크 자기잡음에 지배돼 CVaR 이 의미 없는 프레임을 고른다.
        floor = d_pow.amax(dim=1, keepdim=True) * (
            10.0 ** (self.frame_silence_db / 10.0)
        )
        valid = d_pow > floor
        ratio_db = 10.0 * torch.log10((e_pow + _EPS) / (d_pow + _EPS))
        return ratio_db, valid

    # --------------------------------------------------------------- 출력 포화
    def saturation_penalty(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """리미터 **이전** 활성 포화 벌점과 복원된 ``|u|/L``.

        구 ``l_clip = relu(|y| − clip_margin)²`` 는 **구조적으로 죽어 있었다**: 모델
        출력이 ``y = L·tanh(u/L)`` 라 ``|y| < L`` 이 항상 성립하고, 따라서 그 항의
        **상한이 (0.2−0.18)² = 4.0e−4 로 고정**돼 있었다. 실측값 8.45e−7(synth) /
        0.0(recorded), ``||∇l_clip|| / ||∇l_nmse|| = 1.1e−8``.

        진짜 위험은 리미터 이전 활성의 포화다. 실측 복원값 ``|u|max = 7.77·L`` →
        ``tanh′ ≈ 1e−5`` 라 그 샘플들의 그래디언트가 사실상 죽는다(전체의 0.22%).

        ``atanh`` 는 모델 ``tanh`` 의 정확한 역이라 ``u`` 를 재구성할 수 있고, 그 미분
        ``1/(1−(y/L)²)`` 이 ``tanh`` 미분을 정확히 상쇄해 벌점이 ``u`` 에 **단위 이득**
        으로 닿는다 (``∂l/∂u = 2(|u|/L − margin)/L``).

        ``clamp`` 는 fp32 에서 ``atanh`` 가 발산하는 것을 막는다. 잘리는 범위
        (``|u|/L > atanh(1−sat_ratio_eps) ≈ 4.95``)에서는 그래디언트가 0 이지만 벌점은
        단조라, 학습이 ``u`` 를 낮추면 그 샘플들이 다시 보이기 시작한다 — 국소최소
        함정이 없다.

        **전제**: 리미터가 모델의 마지막 연산이다 (hybrid_anc.py:116, 179). 모델 구조를
        바꾸면 여기가 조용히 틀린 값을 재구성한다.
        """

        ratio = (y / self.limit).clamp(
            -1.0 + self.sat_ratio_eps, 1.0 - self.sat_ratio_eps
        )
        u_over_limit = torch.atanh(ratio)
        l_sat = F.relu(u_over_limit.abs() - self.sat_margin).pow(2).mean()
        return l_sat, u_over_limit

    # ---------------------------------------------------------------- forward
    # ------------------------------------------------------------ 항별 그래디언트 예산
    def gradient_budget(
        self,
        y: torch.Tensor,
        d: torch.Tensor,
        perturb: dict | None = None,
        *,
        loss_start_sample: int = 0,
        nl_params: dict | None = None,
    ) -> dict[str, float]:
        """각 항이 ``∂/∂y`` 예산에서 차지하는 몫: ``‖λ·∇term‖ / ‖∇nmse‖``.

        여기서 ``y``는 리미터를 통과한 **모델 출력 파형**이다. 이 값은 model
        parameter-gradient가 아니다. 파라미터 Jacobian을 곱하기 전 손실 표면에서
        보조항과 NMSE의 상대 크기를 재는 진단이며, 문서와 campaign receipt도 반드시
        ``model_output_y`` gradient라고 표기해야 한다.

        왜 손실 안에 있는가
        ------------------
        λ 를 고를 때마다 이 값을 재야 하는데, 재는 코드가 스크립트 안에 있으면 항 목록을
        거기서 다시 적게 된다. 그러면 항이 추가·삭제될 때 조용히 갈라진다 — 실제로
        2026-08-06 감사가 "λ_dnh 가 새 대역 구성에서 재교정되지 않았다(비 1333%)" 를
        찾았을 때, 그 1333% 를 만든 계산은 저장소 어디에도 남아 있지 않았다.
        :meth:`forward` 와 **같은 dict** 를 쓰므로 갈라질 수 없다.

        해석: ``nmse`` 는 항상 1.0 이다(분모). 설계 목표는 보조항이 0.2~0.4 다 —
        그보다 크면 목적함수가 보조항에 끌려가고, 훨씬 작으면 그 항이 죽은 것이다.

        ⚠ 값은 ``y`` 에 의존한다. ``y=0`` 에서는 dnh 가 0 이라 무의미하므로, 학습 중간을
        닮은 y(신뢰대역 일부 상쇄 + 대역 밖 누설)에서 재라.

        ``loss_start_sample`` / ``perturb`` / ``nl_params`` 는 :meth:`forward` 와
        같은 의미다. 예산 측정이 학습이 실제로 버리는 plant 정착 구간을 포함하면,
        같은 S(z)라도 전혀 다른 목적함수의 비율을 보고하게 된다. 따라서 campaign
        evidence는 Trainer가 쓴 값을 그대로 넘겨야 한다.
        """

        y = y.detach().clone().requires_grad_(True)
        total, _ = self.forward(
            y,
            d,
            loss_start_sample=loss_start_sample,
            perturb=perturb,
            nl_params=nl_params,
        )
        terms = self._last_terms
        norms: dict[str, float] = {}
        for name, (weight, term) in terms.items():
            if float(weight) == 0.0 or not term.requires_grad:
                norms[name] = 0.0
                continue
            (grad,) = torch.autograd.grad(
                float(weight) * term, y, retain_graph=True, allow_unused=True
            )
            norms[name] = 0.0 if grad is None else float(grad.norm())
        base = norms.get("nmse", 0.0)
        if base <= 0.0:
            raise ValueError(
                "nmse 항의 그래디언트가 0 입니다 — 예산비를 정의할 분모가 없습니다. "
                "y 가 목적함수에 영향을 주는 값인지 확인하세요"
            )
        return {name: value / base for name, value in norms.items()}

    def forward(
        self,
        y: torch.Tensor,
        d: torch.Tensor,
        loss_start_sample: int = 0,
        perturb: dict | None = None,
        nl_params: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """y, d: [B, 1, T] (물리 스케일). 반환: (total_loss, metrics).

        - loss_start_sample: 플랜트를 전체 길이에 적용한 **뒤** NMSE/MR-STFT 에서 제외할
          앞구간. 잘린 y 에 플랜트를 걸면 경계 뒤 ~(지연+FIR) 구간의 S·y 기여가
          사라지므로 반드시 여기서 잘라야 한다 (리뷰 확정 결함 #2/#5).
          값의 단일 출처는 ``dsp.timing.PlantSettle`` 이다.
        - perturb/nl_params: 폐루프 학습에서 되먹임 경로와 동일한 플랜트/비선형을
          쓰기 위한 외부 주입 (리뷰 확정 결함 #6). None 이면 내부 샘플링.
        - FFT(플랜트·STFT)는 bf16 미지원 → 손실 전체 FP32 고정.
        """
        if y.is_cuda:
            autocast_off = torch.autocast("cuda", enabled=False)
        else:
            import contextlib

            autocast_off = contextlib.nullcontext()
        with autocast_off:
            return self._forward_fp32(
                y.float(), d.float(), loss_start_sample, perturb, nl_params
            )

    def _forward_fp32(
        self,
        y: torch.Tensor,
        d: torch.Tensor,
        loss_start_sample: int = 0,
        perturb: dict | None = None,
        nl_params: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        batch = y.shape[0]

        # 평가(eval) 모드에서는 비선형/섭동 없이 결정적으로 계산 — val 지표 일관성
        y_nl = y
        if self.training and self.nonlinear is not None:
            if nl_params is None:
                nl_params = self.nonlinear.sample(batch)
            y_nl = self.nonlinear.apply_torch(y, nl_params)

        if perturb is None:
            perturb = self.plant.sample_perturbation() if self.training else {"jitter": 0}
        s_y = self.plant(y_nl, perturb)
        e = d + s_y

        skip = int(loss_start_sample)
        e_flat = e.squeeze(1)[..., skip:]
        d_flat = d.squeeze(1)[..., skip:]
        s_y_flat = s_y.squeeze(1)[..., skip:]

        # 전대역 NMSE (dB) — do-no-harm 관측용
        e_pow = e_flat.pow(2).sum(dim=-1)
        d_pow = d_flat.pow(2).sum(dim=-1)
        nmse_fullband_db = 10.0 * torch.log10((e_pow + _EPS) / (d_pow + _EPS))

        nmse_trusted_db: torch.Tensor | None = None
        if self.trusted_band_hz is not None:
            nmse_trusted_db = self._band_nmse_db(e_flat, d_flat, self.trusted_band_hz)

        l_nmse, nmse_objective_metrics = self._main_nmse_objective(
            e_flat,
            d_flat,
            nmse_fullband_db,
            nmse_trusted_db,
        )

        # 다중해상도 STFT (주파수 가중) — e 의 스펙트럼 에너지를 d 대비로 정규화
        l_mrstft = y.new_zeros(())
        for fft_size in self.ffts:
            w = getattr(self, f"w_{fft_size}").view(1, -1, 1)
            win = getattr(self, f"win_{fft_size}")
            E = _stft_mag(e_flat, fft_size, fft_size // 4, win) * w
            D = _stft_mag(d_flat, fft_size, fft_size // 4, win) * w
            # **아이템별** 정규화. 배치 전체 노름은 가장 큰 아이템이 항을 독식한다 —
            # 실측 배치 d 레벨 편차 45~66 dB, top1 아이템이 배치 에너지의 17~46%,
            # 하위 절반 합계 2%.
            sc = torch.linalg.norm(E, dim=(1, 2)) / (
                torch.linalg.norm(D, dim=(1, 2)) + _EPS
            )
            l1 = E.sum(dim=(1, 2)) / (D.sum(dim=(1, 2)) + _EPS)
            l_mrstft = l_mrstft + self._worst_aggregate(sc) + self._worst_aggregate(l1)
        l_mrstft = l_mrstft / len(self.ffts)

        # 대역 밖 악화 금지 (단측, S 위상 무관)
        l_dnh = y.new_zeros(())
        dnh_metrics: dict[str, float] = {}
        if self.do_no_harm is not None and self.do_no_harm.bands:
            l_dnh, dnh_metrics = self._do_no_harm(s_y_flat, d_flat)

        # 프레임 국소성
        l_frame = y.new_zeros(())
        frame_worst_db: float | None = None
        frame_valid_count: int | None = None
        # frame은 학습 가중치와 별개로 항상 관측한다. 가중치가 0인 안전한
        # warm-up/candidate에서도 순간 증폭을 숨기면, "학습은 안정적"이라는 말과
        # "국소적으로도 안전"이라는 말을 구분할 수 없다.
        if self.trusted_band_hz is not None:
            if e_flat.shape[-1] >= self.frame_samples:
                fr_db, valid = self._framed_band_nmse_db(
                    e_flat, d_flat, self.trusted_band_hz
                )
                frame_valid_count = int(valid.sum().detach())
                if bool(valid.any()):
                    selected = fr_db[valid]
                    l_frame = self._worst_aggregate(selected)
                    frame_worst_db = float(selected.max().detach())

        # 리미터 이전 활성 포화 (구 l_clip 대체 — 근거는 saturation_penalty 참조)
        l_sat, u_over_limit = self.saturation_penalty(y)
        l_pow = y.pow(2).mean()  # λ=0 기본 — 지표로만 유지

        # 항을 이름과 함께 모아 둔다. forward 는 **합만** 하고, 예산 측정
        # (gradient_budget)은 같은 dict 를 쓴다 — 항 목록을 두 곳에 적으면 그것이
        # 발생기 A 이고, 실제로 λ 재교정 때마다 목록이 갈라졌다.
        self._last_terms = {
            "nmse": (1.0, l_nmse),
            "mrstft": (self.lambda_mrstft, l_mrstft),
            "dnh": (self.lambda_dnh, l_dnh),
            "frame": (self.lambda_frame, l_frame),
            "sat": (self.lambda_sat, l_sat),
            "pow": (self.lambda_pow, l_pow),
        }
        total = sum(
            (weight * term for weight, term in self._last_terms.values()),
            start=y.new_zeros(()),
        )
        metrics = {
            "loss": float(total.detach()),
            "nmse_db": float(l_nmse.detach()),
            "nmse_fullband_db": float(nmse_fullband_db.mean().detach()),
            "nmse_fullband_worst_db": float(nmse_fullband_db.max().detach()),
            "mrstft": float(l_mrstft.detach()),
            "dnh": float(l_dnh.detach()),
            "frame": float(l_frame.detach()),
            "sat": float(l_sat.detach()),
            "sat_u_over_limit_max": float(u_over_limit.abs().max().detach()),
            "out_pow": float(l_pow.detach()),
        }
        if frame_worst_db is not None:
            metrics["frame_worst_db"] = frame_worst_db
        if frame_valid_count is not None:
            metrics["frame_valid_count"] = float(frame_valid_count)
        metrics.update(nmse_objective_metrics)
        metrics.update(dnh_metrics)
        if nmse_trusted_db is not None:
            mean_db = float(nmse_trusted_db.mean().detach())
            # 기존 대시보드/로그가 읽는 alias 는 **평균**으로 유지한다. 목적함수 값은
            # nmse_db 에 있고, 선택 기준은 nmse_trusted_cvar_db 다.
            metrics["nmse_trusted_db"] = mean_db
            metrics["nmse_trusted_mean_db"] = mean_db
            metrics["nmse_trusted_cvar_db"] = float(self._cvar(nmse_trusted_db).detach())
            metrics["nmse_trusted_worst_db"] = float(nmse_trusted_db.max().detach())
        return total, metrics
