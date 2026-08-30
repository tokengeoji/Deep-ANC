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

from ..dsp.do_no_harm import (
    MAX_OUT_OF_BAND_AMPLIFICATION_DB,
    gate_consistent_margin_db,
    octave_boundary_edges_hz,
    worst_case_amplification_db,
)
from ..dsp.timing import FrequencyBand

__all__ = [
    "DEFAULT_DNH_MARGIN_DB",
    "DEFAULT_DO_NO_HARM_EDGES_HZ",
    "DoNoHarmBand",
    "DoNoHarmPlan",
    "LossConfig",
]


_FROZEN = ConfigDict(frozen=True, extra="forbid")


DEFAULT_DNH_MARGIN_DB: float = gate_consistent_margin_db()
"""do-no-harm 힌지 마진 — **G4 게이트 임계에서 유도된다. 여기에 숫자를 쓰지 마라.**

2026-08-06 이전에는 ``6.0`` 이 리터럴로 적혀 있었고, 게이트는 ``eval/recorded.py`` 에
``1.0`` 이라고 따로 적혀 있었다. 두 값은 서로를 모른 채 살았고 대조 코드도 테스트도
없었다 — 실측 결과 **힌지를 정확히 만족한 모델이 게이트를 8.5 dB 차이로 FAIL** 했다
(4000 Hz 옥타브 −9.53 dB, 한계 −1.0 dB). 유도 근거는 ``dsp/do_no_harm.py`` 참조.
"""


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

    ``margin_db`` 는 ``10·log10(bandpower(S·y)/bandpower(d))`` 를 이 값까지 봐준다는
    뜻이다. **음수가 정상이다** — 게이트가 보는 것은 ``e = d + S·y`` 이고, 위상이
    동상으로 맞으면 ``|e| = |d| + |S·y|`` 가 되기 때문이다. 옥타브를 1.0 dB 이상 키우지
    않으려면 반노이즈가 교란보다 **18.3 dB 아래**여야 한다(:mod:`deep_anc.dsp.do_no_harm`).

    2026-08-06 이전 주석은 "``margin_db`` 는 |S| 측정오차를 봐주는 값" 이라며 ``+6.0`` 을
    정당화했다. 방향이 반대였다 — |S| 를 **과소평가**했다면 실제 증폭은 손실이 아는 것보다
    나쁘다. 플랜트 불확실성은 마진을 **좁히는** 쪽으로만 작용해야 한다.
    """

    model_config = _FROZEN

    band: FrequencyBand
    margin_db: float
    weight: float

    @model_validator(mode="after")
    def _validate(self) -> "DoNoHarmBand":
        if not math.isfinite(self.margin_db):
            raise ValueError(
                f"do_no_harm margin_db 는 유한한 값이어야 합니다: {self.margin_db}"
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
        p_lo, p_hi = self.protected.as_tuple()
        nyquist = max((item.band.as_tuple()[1] for item in self.bands), default=0.0)
        octave_edges = (
            octave_boundary_edges_hz(nyquist_hz=nyquist) if nyquist > 0.0 else ()
        )
        for item in self.bands:
            lo, hi = item.band.as_tuple()
            if lo < p_hi and p_lo < hi:
                raise ValueError(
                    f"do_no_harm 대역 [{lo:g}, {hi:g}] 가 개선 요구 대역 "
                    f"[{p_lo:g}, {p_hi:g}] 와 겹칩니다 — 같은 주파수에 양측 목표와 "
                    "단측 힌지를 동시에 걸면 서로 상쇄합니다"
                )
            # 옥타브를 가로지르면 대역별 상한이 옥타브 상한으로 합쳐지지 않는다.
            crossing = [edge for edge in octave_edges if lo < edge - 1e-6 and edge < hi - 1e-6]
            if crossing:
                raise ValueError(
                    f"do_no_harm 대역 [{lo:g}, {hi:g}] 가 G4 옥타브 경계 "
                    f"{[round(v, 1) for v in crossing]} Hz 를 가로지릅니다 — 이러면 "
                    "대역 전체 비율을 만족한 채로 한 옥타브에 에너지를 몰아넣을 수 "
                    "있고, 실측에서 그 자유도가 게이트를 3.1 dB 더 나쁘게 만들었습니다. "
                    "경계를 옥타브에 맞춰 자르세요 (DoNoHarmPlan.derive 는 자동으로 합칩니다)"
                )
            # 게이트 임계에서 유도한 상한보다 느슨한 마진은 보장이 아니라 희망이다.
            allowed = gate_consistent_margin_db()
            if item.margin_db > allowed + 1e-9:
                raise ValueError(
                    f"do_no_harm 대역 [{lo:g}, {hi:g}] 의 margin_db="
                    f"{item.margin_db:+.2f} 는 G4 임계 "
                    f"{MAX_OUT_OF_BAND_AMPLIFICATION_DB:+.1f} dB 가 허용하는 상한 "
                    f"{allowed:+.2f} dB 를 넘습니다 — 이 마진을 정확히 만족한 모델은 "
                    f"옥타브를 최대 {worst_case_amplification_db(item.margin_db):.2f} dB "
                    "증폭할 수 있어 게이트에서 FAIL 합니다 "
                    "(2026-08-06 실측: margin +6.0 → 4000 Hz 옥타브 −9.53 dB)"
                )
        return self

    @classmethod
    def derive(
        cls,
        *,
        protected: FrequencyBand,
        nyquist_hz: float,
        edges_hz: Sequence[float] = DEFAULT_DO_NO_HARM_EDGES_HZ,
        margin_db: float = DEFAULT_DNH_MARGIN_DB,
        weight_below: float = 1.0,
        weight_above: float = 2.0,
    ) -> "DoNoHarmPlan":
        """경계 목록에서 **보호 대역을 빼서** do-no-harm 대역을 만든다.

        세 가지가 구조적으로 보장된다.

        1. **겹침 불가** — 빼고 남은 것만 쓰기 때문이다.
        2. **빈틈 불가** — 결과는 ``[0, Nyquist] − protected`` 를 정확히 덮는다.
           경계 목록은 그 여집합을 어디서 자를지만 정한다. 감시하지 않는 주파수 구간이
           하나라도 남으면 모델이 거기에 출력을 쏟아부어도 비용이 0 이고, 그것이 바로
           결함 3 이 손실 안에서 공짜였던 이유다.
        3. **옥타브를 가로지르지 않는다** — G4 게이트의 옥타브 경계를 절단점에 항상
           합친다. 이것이 없으면 대역별 상한이 옥타브 상한으로 **합쳐지지 않는다**.
           실측(2026-08-06): ``[1633, 6000]`` 이 옥타브 2000·4000 을 함께 덮고 있어서,
           대역 전체 비율을 정확히 만족시키면서 에너지를 2000 옥타브에 몰아넣으면
           그 옥타브만 −12.63 dB 가 됐다(고르게 퍼뜨렸을 때는 −8.99 dB). 즉 손실을
           만족한 채로 게이트를 3.1 dB 더 나쁘게 만드는 자유도가 남아 있었다.
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
        # 게이트 옥타브 경계를 절단점에 합친다. 설정이 무엇을 적든 이것은 빠질 수 없다 —
        # 빠지면 보장이 정리(定理)가 아니라 희망이 된다.
        edges = sorted(set(edges) | set(octave_boundary_edges_hz(nyquist_hz=nyquist)))
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
    dnh_margin_db: float = DEFAULT_DNH_MARGIN_DB
    dnh_weight_below: float = 1.0
    dnh_weight_above: float = 2.0
    dnh_band_floor_db: float = -60.0
    """대역 교란 전력의 하한 — 전체 전력 대비. 이보다 조용한 대역은 **비율이 정의되지 않는다.**

    비율 힌지에는 분모가 필요하다. 교란이 사실상 없는 대역에서 ``P_sy/P_d`` 를 그대로
    쓰면 두 잡음바닥을 나누는 것이고, 값이 신호가 아니라 수치 오차로 정해진다. 하한을
    **신호 자신의 전체 전력에 상대적으로** 두면 스케일 불변이면서, 조용한 대역에 출력을
    쏟아붓는 것은 여전히 비싸다(분모가 하한에 고정될 뿐 0 이 되지 않는다).

    −60 dB 의 뜻: 그 대역이 전체 교란 전력의 100만분의 1 미만이면 하한을 쓴다. 실측
    녹음(소음·음성·음악)은 어느 옥타브도 이보다 조용하지 않다 — 이 값이 실제로 작동하는
    것은 합성 순음 시험처럼 대역이 통째로 비는 경우뿐이다.
    """

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
            "dnh_weight_below",
            "dnh_weight_above",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"loss.{name} 는 유한한 0 이상 값이어야 합니다: {value}")
        # dnh_margin_db 는 **음수가 정상**이다 (게이트 임계에서 유도되며 -18.27 dB).
        # 상한 검사는 대역이 만들어지는 DoNoHarmPlan 에서 G4 임계와 직접 대조한다 —
        # 여기서 숫자를 한 번 더 적으면 그것이 세 번째 유도가 된다.
        if not math.isfinite(self.dnh_margin_db):
            raise ValueError(
                f"loss.dnh_margin_db 는 유한한 값이어야 합니다: {self.dnh_margin_db}"
            )
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
