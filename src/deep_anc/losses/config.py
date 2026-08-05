"""손실 설정의 **경계 검증** — YAML dict 가 손실 안으로 들어오는 유일한 문.

왜 pydantic 인가
---------------
2026-08-05 결함 군집 분석: 확인된 결함의 78% 가 "같은 물리량을 두 곳에서 따로 유도"
또는 "게이트가 통과만 해 봤다" 에서 나왔다. 손실 설정에는 그 두 가지가 다 있었다.

* ``loss.clip_margin`` 과 모델 ``limiter.limit`` 은 **서로 다른 두 곳에 적힌 같은 물리량**
  이었고, trainer 가 부등식 하나로 대조하는 것이 전부였다. 실제로는 그 항 자체가 죽어
  있었고(상한 4.0e−4, 그래디언트 기여 1.1e−8) 아무도 몰랐다.
* ``loss:`` 블록은 그냥 ``dict.get`` 으로 읽혔다. 오타 난 키는 **조용히 무시**된다.
  죽은 설정은 다음 사람을 속인다 — ``configs/eval*.yaml`` 의 ``trusted_band_hz`` 가
  어떤 코드에도 읽히지 않으면서 3개 파일에 남아 있던 것과 같은 종류다.

그래서 이 모듈은 ``extra="forbid"`` 로 **모르는 키를 즉시 거부**하고, 값 범위를 생성
시점에 강제하며, 폐기된 키는 조용히 무시하지 않고 경고를 낸다.

핫패스 금지: 이 검증은 ANCLoss 생성 시 **1회** 돈다. 스텝마다 돌지 않는다.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, model_validator

from ..dsp.timing import FrequencyBand

__all__ = [
    "DEFAULT_DO_NO_HARM_EDGES_HZ",
    "DoNoHarmBand",
    "DoNoHarmPlan",
    "LossConfig",
]


_FROZEN = ConfigDict(frozen=True, extra="forbid")


DEFAULT_DO_NO_HARM_EDGES_HZ: tuple[float, ...] = (
    20.0,
    80.0,
    150.0,
    600.0,
    1000.0,
    1633.0,
    6000.0,
    20000.0,
)
"""대역 밖 감시 구간을 자르는 경계.

이 값들은 **대역 자체가 아니라 자를 자리**다. 실제 do-no-harm 대역은 여기서 신뢰
대역(개선을 요구하는 대역)을 빼서 만든다 — 그래서 S(z) 신뢰대역이 [150,600] 에서
[150,1600] 으로 넓어져도 손으로 고칠 것이 없다. 대역 목록을 리터럴로 박아 두면
그것이 **여섯 번째 복붙**이 된다(발생기 A).

경계 선택 근거(실측 2026-08-05, pretrain_tiny synth B=32, 10log10 bandpower(S·y)/d):
20–80 +30.5 / 80–150 +24.4 / 600–1000 +25.2 / 1000–1633 +22.3 / 1633–6000 +19.9 /
6000–20000 +17.8 dB (최악값). 1633Hz 는 덕트 평면파 컷오프이자 S(z) 가진 상한이다.
"""


class DoNoHarmBand(BaseModel):
    """대역 밖 '악화 금지' 힌지 한 구간.

    ``margin_db`` 는 "시뮬레이터가 말하는 증폭 중 몇 dB 까지는 |S| 측정오차로 보고
    봐준다" 는 뜻이다. 1633Hz 위의 |S| 는 가진조차 하지 않은 구간의 FIR 외삽이므로
    관대해야 한다. 결함 3 의 15~22 dB 증폭은 6 dB 마진을 훌쩍 넘으므로 그래도 잡힌다.
    """

    model_config = _FROZEN

    band: FrequencyBand
    margin_db: float
    weight: float

    @model_validator(mode="after")
    def _validate(self) -> "DoNoHarmBand":
        if not math.isfinite(self.margin_db) or self.margin_db < 0.0:
            raise ValueError(
                f"do_no_harm margin_db 는 유한한 0 이상 값이어야 합니다: {self.margin_db}"
            )
        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError(
                f"do_no_harm weight 는 유한한 0 이상 값이어야 합니다: {self.weight}"
            )
        return self

    def as_tuple(self) -> tuple[float, float, float, float]:
        lo, hi = self.band.as_tuple()
        return (lo, hi, float(self.margin_db), float(self.weight))


class DoNoHarmPlan(BaseModel):
    """do-no-harm 대역 집합. **개선을 요구하는 대역과 겹칠 수 없다.**

    왜 겹치면 안 되는가: 같은 주파수에 "여기를 줄여라"(양측 NMSE)와 "여기를 키우지
    마라"(단측 힌지)를 동시에 걸면 두 항이 서로 상쇄하고, 어느 쪽이 이겼는지 지표로는
    보이지 않는다. 겹침을 **생성 시점에 거부**해야 그 상태가 존재할 수 없다.
    """

    model_config = _FROZEN

    protected: FrequencyBand
    """개선을 요구하는 대역(= 손실의 trusted band). 여기에는 힌지를 걸지 않는다."""

    bands: tuple[DoNoHarmBand, ...]

    @model_validator(mode="after")
    def _validate(self) -> "DoNoHarmPlan":
        for item in self.bands:
            lo, hi = item.band.as_tuple()
            p_lo, p_hi = self.protected.as_tuple()
            if lo < p_hi and p_lo < hi:
                raise ValueError(
                    f"do_no_harm 대역 [{lo:g}, {hi:g}] 가 개선 요구 대역 "
                    f"[{p_lo:g}, {p_hi:g}] 와 겹칩니다 — 같은 주파수에 양측 목표와 "
                    "단측 힌지를 동시에 걸면 서로 상쇄합니다"
                )
        return self

    @classmethod
    def derive(
        cls,
        *,
        protected: FrequencyBand,
        nyquist_hz: float,
        edges_hz: Sequence[float] = DEFAULT_DO_NO_HARM_EDGES_HZ,
        margin_db: float = 6.0,
        weight_below: float = 1.0,
        weight_above: float = 2.0,
    ) -> "DoNoHarmPlan":
        """경계 목록에서 **보호 대역을 빼서** do-no-harm 대역을 만든다.

        두 가지가 구조적으로 보장된다.

        1. **겹침 불가** — 빼고 남은 것만 쓰기 때문이다.
        2. **빈틈 불가** — 결과는 ``[0, Nyquist] − protected`` 를 정확히 덮는다.
           경계 목록은 그 여집합을 어디서 자를지만 정한다. 감시하지 않는 주파수 구간이
           하나라도 남으면 모델이 거기에 출력을 쏟아부어도 비용이 0 이고, 그것이 바로
           결함 3 이 손실 안에서 공짜였던 이유다.
        """

        nyquist = float(nyquist_hz)
        if not math.isfinite(nyquist) or nyquist <= 0.0:
            raise ValueError(f"Nyquist 주파수는 유한한 양수여야 합니다: {nyquist_hz}")
        edges = [float(v) for v in edges_hz]
        if len(edges) < 2:
            raise ValueError(f"do_no_harm 경계는 2개 이상이어야 합니다: {edges_hz!r}")
        if any(not math.isfinite(v) or v < 0.0 for v in edges):
            raise ValueError(f"do_no_harm 경계는 유한한 0 이상 값이어야 합니다: {edges_hz!r}")
        if any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError(f"do_no_harm 경계는 증가해야 합니다: {edges_hz!r}")
        # 여집합 전체를 덮도록 양끝을 [0, Nyquist] 까지 늘린다.
        if edges[0] > 0.0:
            edges.insert(0, 0.0)
        if nyquist > edges[-1]:
            edges.append(nyquist)

        p_lo, p_hi = protected.as_tuple()
        bands: list[DoNoHarmBand] = []
        for lo, hi in zip(edges, edges[1:]):
            hi = min(hi, nyquist)
            if hi <= lo:
                continue
            # 보호 대역을 뺀 나머지 조각만 남긴다.
            if lo < p_lo:
                bands.append(
                    DoNoHarmBand(
                        band=FrequencyBand(lo_hz=lo, hi_hz=min(hi, p_lo)),
                        margin_db=float(margin_db),
                        weight=float(weight_below),
                    )
                )
            if hi > p_hi:
                bands.append(
                    DoNoHarmBand(
                        band=FrequencyBand(lo_hz=max(lo, p_hi), hi_hz=hi),
                        margin_db=float(margin_db),
                        weight=float(weight_above),
                    )
                )
        return cls(protected=protected, bands=tuple(bands))

    @classmethod
    def from_spec(
        cls,
        spec: Sequence[Sequence[float]],
        *,
        protected: FrequencyBand,
        nyquist_hz: float,
    ) -> "DoNoHarmPlan":
        """설정이 대역을 직접 적었을 때. ``[lo, hi, margin_db, weight]`` 목록."""

        bands: list[DoNoHarmBand] = []
        for row in spec:
            values = list(row)
            if len(values) != 4:
                raise ValueError(
                    "do_no_harm_bands 항목은 [lo_hz, hi_hz, margin_db, weight] "
                    f"형식이어야 합니다: {row!r}"
                )
            lo, hi, margin_db, weight = (float(v) for v in values)
            bands.append(
                DoNoHarmBand(
                    band=FrequencyBand.parse(
                        (lo, hi), name="do_no_harm", nyquist_hz=float(nyquist_hz)
                    ),
                    margin_db=margin_db,
                    weight=weight,
                )
            )
        return cls(protected=protected, bands=tuple(bands))

    def as_tuples(self) -> tuple[tuple[float, float, float, float], ...]:
        return tuple(item.as_tuple() for item in self.bands)


class LossConfig(BaseModel):
    """``configs/*.yaml`` 의 ``loss:`` 블록. 모르는 키는 거부한다."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ---- 목적함수 ----
    nmse_objective: Literal["trusted_band", "fullband"] | None = None
    """None 이면 trusted_band_hz 주입 여부로 결정한다(기존 동작 보존)."""

    # ---- 최악값 집계 (절대목표 2) ----
    nmse_cvar_q: float = 0.25
    nmse_cvar_alpha: float = 0.7
    nmse_cvar_min_k: int = 4

    # ---- 대역 밖 악화 금지 (절대목표 1) ----
    lambda_dnh: float = 0.12
    do_no_harm_bands: list[list[float]] | None = None
    do_no_harm_edges_hz: list[float] | None = None
    dnh_margin_db: float = 6.0
    dnh_weight_below: float = 1.0
    dnh_weight_above: float = 2.0

    # ---- 시간 국소성 ----
    lambda_frame: float = 0.0
    nmse_frame_samples: int = 8192
    nmse_frame_hop: int | None = None
    nmse_frame_silence_db: float = -40.0

    # ---- 스펙트럼 형상 ----
    mrstft_ffts: list[int] = [256, 512, 1024, 2048]
    lambda_mrstft: float = 1.0
    band_weight: Literal["curriculum_a", "fullband", "trusted_only"] = "curriculum_a"

    # ---- 출력 제약 ----
    lambda_pow: float = 0.0
    lambda_sat: float = 1.0
    sat_margin: float = 2.0
    sat_ratio_eps: float = 1.0e-4
    limiter_limit: float | None = None

    # ---- 폐기된 키 (조용히 무시하지 않는다) ----
    lambda_clip: float | None = None
    clip_margin: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> "LossConfig":
        if not 0.0 < self.nmse_cvar_q <= 1.0:
            raise ValueError(f"loss.nmse_cvar_q 는 (0,1] 이어야 합니다: {self.nmse_cvar_q}")
        if not 0.0 <= self.nmse_cvar_alpha <= 1.0:
            raise ValueError(
                f"loss.nmse_cvar_alpha 는 [0,1] 이어야 합니다: {self.nmse_cvar_alpha}"
            )
        if self.nmse_cvar_min_k < 1:
            raise ValueError(
                f"loss.nmse_cvar_min_k 는 1 이상이어야 합니다: {self.nmse_cvar_min_k}"
            )
        for name in (
            "lambda_dnh",
            "lambda_frame",
            "lambda_mrstft",
            "lambda_pow",
            "lambda_sat",
            "dnh_margin_db",
            "dnh_weight_below",
            "dnh_weight_above",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"loss.{name} 는 유한한 0 이상 값이어야 합니다: {value}")
        if not math.isfinite(self.sat_margin) or self.sat_margin <= 0.0:
            raise ValueError(
                f"loss.sat_margin 은 유한한 양수여야 합니다: {self.sat_margin} "
                "(tanh'(2)=0.071 — 이 지점부터 유효 그래디언트가 1/14 이하다)"
            )
        if not 0.0 < self.sat_ratio_eps < 1.0:
            raise ValueError(
                f"loss.sat_ratio_eps 는 (0,1) 이어야 합니다: {self.sat_ratio_eps}"
            )
        if self.limiter_limit is not None and not (
            math.isfinite(self.limiter_limit) and self.limiter_limit > 0.0
        ):
            raise ValueError(
                f"loss.limiter_limit 은 유한한 양수여야 합니다: {self.limiter_limit}"
            )
        if not self.mrstft_ffts:
            raise ValueError("loss.mrstft_ffts 가 비었습니다")
        if any(int(v) < 4 for v in self.mrstft_ffts):
            raise ValueError(f"loss.mrstft_ffts 는 4 이상이어야 합니다: {self.mrstft_ffts}")
        if int(self.nmse_frame_samples) < 4:
            raise ValueError(
                f"loss.nmse_frame_samples 는 4 이상이어야 합니다: {self.nmse_frame_samples}"
            )
        if self.nmse_frame_hop is not None and int(self.nmse_frame_hop) < 1:
            raise ValueError(
                f"loss.nmse_frame_hop 은 1 이상이어야 합니다: {self.nmse_frame_hop}"
            )
        if self.do_no_harm_bands is not None and self.do_no_harm_edges_hz is not None:
            raise ValueError(
                "loss.do_no_harm_bands 와 do_no_harm_edges_hz 를 동시에 지정할 수 "
                "없습니다 — 대역을 두 곳에서 유도하는 것이 바로 막으려는 결함입니다"
            )
        self._warn_deprecated()
        return self

    def _warn_deprecated(self) -> None:
        dead: list[str] = []
        if self.clip_margin is not None:
            dead.append(f"clip_margin={self.clip_margin}")
        if self.lambda_clip is not None:
            dead.append(f"lambda_clip={self.lambda_clip}")
        if not dead:
            return
        message = (
            "loss." + ", loss.".join(dead) + " 는 폐기됐고 아무 효과가 없습니다. "
            "모델 출력이 y = L·tanh(u/L) 라 |y| < L 이 항상 성립하고, "
            "relu(|y|−clip_margin)² 의 상한이 (0.2−0.18)² = 4.0e−4 로 고정돼 있었습니다 "
            "(실측 그래디언트 기여 1.1e−8 = 구조적으로 죽은 항). "
            "리미터 **이전** 활성 포화를 보는 lambda_sat / sat_margin 으로 대체됐습니다."
        )
        warnings.warn(message, DeprecationWarning, stacklevel=4)

    @classmethod
    def parse(cls, raw: dict[str, Any] | None) -> "LossConfig":
        """YAML dict → 검증된 설정. 모르는 키는 여기서 죽는다."""

        return cls.model_validate(dict(raw or {}))

    def frame_hop_samples(self) -> int:
        if self.nmse_frame_hop is not None:
            return int(self.nmse_frame_hop)
        return max(1, int(self.nmse_frame_samples) // 2)

    def resolve_do_no_harm(
        self, *, protected: FrequencyBand, nyquist_hz: float
    ) -> DoNoHarmPlan:
        """설정에서 do-no-harm 대역 계획을 만든다 (유일한 경로)."""

        if self.do_no_harm_bands is not None:
            return DoNoHarmPlan.from_spec(
                self.do_no_harm_bands, protected=protected, nyquist_hz=nyquist_hz
            )
        edges = (
            DEFAULT_DO_NO_HARM_EDGES_HZ
            if self.do_no_harm_edges_hz is None
            else tuple(float(v) for v in self.do_no_harm_edges_hz)
        )
        return DoNoHarmPlan.derive(
            protected=protected,
            nyquist_hz=nyquist_hz,
            edges_hz=edges,
            margin_db=float(self.dnh_margin_db),
            weight_below=float(self.dnh_weight_below),
            weight_above=float(self.dnh_weight_above),
        )
