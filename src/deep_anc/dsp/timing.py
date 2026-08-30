"""지연·정렬·대역 부기의 **단일 출처**.

왜 이 모듈이 있는가
------------------
2026-08-05 결함 군집 분석: 확인된 결함 18건 중 9건이 "같은 물리량을 두 곳 이상에서
따로 유도하고 아무도 대조하지 않는다" 는 하나의 발생기에서 나왔다. 실제 사고들 —

* ``lead`` 를 ``trainer.py`` 와 ``finetune_readiness.py`` 가 각자 유도해 109 와 113 으로
  갈라졌다 (커밋 aaeef41).
* ``excitation_band_hz`` 를 ``consistency_band_hz`` 자리에 썼다. 둘 다 ``tuple[float,float]``
  이라 대입이 막히지 않았다.
* ``intersect(sp.trusted_band_hz(), realistic_target_band_hz, fs/2)`` 라는 **같은 한 줄이
  5개 파일에 복붙**돼 있었고, ``intersect_frequency_bands`` 자체도 두 번 정의돼 있었다.
* 전후 비교가 서로 다른 플랜트(S 지연 1342 vs 1465)에서 나온 값이었는데 아무도 못 막았다.

그래서 이 모듈의 규칙은 하나다: **지연/lead/정렬/대역 수치는 여기에서만 유도되고,
나머지 코드는 전부 읽기만 한다.**

설계 원칙
--------
1. **서로 다른 물리량은 서로 다른 타입.** ``int`` 하나로 lead·벌크지연·handoff 를 다
   표현하면 섞인다. 실제로 섞였다.
2. **frozen.** 유도된 값이 나중에 조용히 바뀌면 단일 출처가 무의미하다.
3. **생성 경로를 좁힌다.** :class:`Lead` 는 :meth:`PlantDelays.lead` 로만 만들 수 있다.
   손으로 ``Lead(samples=113)`` 이라고 쓰는 것이 **구조적으로 불가능**하다.
4. **검증은 생성 시점에.** 음수 지연, ``hi <= lo`` 인 대역, 인과성 위반은 만들어질 수 없다.

핫패스 금지
----------
오디오 콜백 본체·샘플 단위 루프에서는 쓰지 마라. 마감이 5.33 ms 다. 이 모듈은
**경계**(설정 로드, 아티팩트 로드, 게이트 판정, 학습/평가 셋업)에서 블록당 1회 이하로
호출되는 것을 전제한다. pydantic 검증 실측 비용은 건당 1.4 µs 다.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, model_validator

__all__ = [
    "DEFAULT_TARGET_BAND_HZ",
    "BandPlan",
    "FrequencyBand",
    "Lead",
    "PlantDelays",
    "PlantFingerprint",
    "PlantSettle",
    "TrainingTimingContract",
    "handoff_samples_from_config",
    "intersect_frequency_bands",
    "target_band_from_config",
]


DEFAULT_TARGET_BAND_HZ: tuple[float, float] = (80.0, 1000.0)
"""``duct.yaml acoustics.realistic_target_band_hz`` 가 없을 때의 기본 목표 대역.

이 기본값은 다섯 곳에 리터럴로 복붙돼 있었다(trainer / eval.recorded /
evaluate_offline / evaluate_session / render_anc_demo). 한 곳만 고치면 나머지가
조용히 옛 값을 쓴다 — 그래서 상수로 올린다.
"""


_FROZEN = ConfigDict(frozen=True, extra="forbid")

# ``Lead`` 를 손으로 만들 수 없게 하는 토큰. 이 객체는 이 모듈 밖으로 나가지 않는다.
_DERIVE_TOKEN = object()


# --------------------------------------------------------------------------------------
# 대역
# --------------------------------------------------------------------------------------
class FrequencyBand(BaseModel):
    """``[lo, hi]`` Hz 구간. ``lo < hi`` 와 유한성이 생성 시점에 강제된다.

    ``tuple[float, float]`` 대신 이 타입을 쓰는 이유는 **의미가 다른 대역끼리 섞이는
    것을 막기 위해서**다. 실제 사고: 구동 대역(``excitation_band_hz``, 64-1648Hz)을
    신뢰 대역(``consistency_band_hz``, 150-1600Hz) 자리에 넣어도 아무 일도 일어나지
    않았다. 어떤 대역인지는 :class:`BandPlan` 의 **필드 이름**이 들고 있다.
    """

    model_config = _FROZEN

    lo_hz: float
    hi_hz: float

    @model_validator(mode="after")
    def _validate(self) -> "FrequencyBand":
        if not (math.isfinite(self.lo_hz) and math.isfinite(self.hi_hz)):
            raise ValueError(
                f"주파수 대역은 유한한 값이어야 합니다: [{self.lo_hz}, {self.hi_hz}]"
            )
        if not 0.0 <= self.lo_hz < self.hi_hz:
            raise ValueError(
                f"잘못된 주파수 대역: [{self.lo_hz}, {self.hi_hz}]"
            )
        return self

    @classmethod
    def parse(
        cls, value: Any, *, name: str = "주파수", nyquist_hz: float | None = None
    ) -> "FrequencyBand":
        """설정/아티팩트에서 온 ``[lo, hi]`` 를 검증해 대역으로 만든다."""

        try:
            values = list(value)
        except TypeError as exc:  # pragma: no cover - 방어
            raise ValueError(f"{name} 대역은 [lo, hi] 형식이어야 합니다: {value!r}") from exc
        if len(values) != 2:
            raise ValueError(f"{name} 대역은 [lo, hi] 형식이어야 합니다: {value!r}")
        try:
            lo, hi = float(values[0]), float(values[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 대역은 [lo, hi] 형식이어야 합니다: {value!r}") from exc
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"{name} 대역은 유한한 값이어야 합니다: {value!r}")
        if not 0.0 <= lo < hi:
            raise ValueError(f"잘못된 주파수 대역 ({name}): {value!r}")
        if nyquist_hz is not None and hi > float(nyquist_hz):
            raise ValueError(
                f"잘못된 주파수 대역 ({name}): {value!r} (Nyquist={float(nyquist_hz):g}Hz)"
            )
        return cls(lo_hz=lo, hi_hz=hi)

    def intersect(self, other: "FrequencyBand") -> "FrequencyBand":
        lo = max(self.lo_hz, other.lo_hz)
        hi = min(self.hi_hz, other.hi_hz)
        if lo >= hi:
            raise ValueError(
                f"주파수 대역 교집이 비어 있습니다: {self.as_tuple()} ∩ {other.as_tuple()}"
            )
        return FrequencyBand(lo_hz=lo, hi_hz=hi)

    def as_tuple(self) -> tuple[float, float]:
        return (float(self.lo_hz), float(self.hi_hz))

    def covers(self, other: "FrequencyBand") -> bool:
        return self.lo_hz <= other.lo_hz and other.hi_hz <= self.hi_hz

    def __str__(self) -> str:  # pragma: no cover - 표시용
        return f"{self.lo_hz:.0f}–{self.hi_hz:.0f}Hz"


def intersect_frequency_bands(
    first: Sequence[float],
    second: Sequence[float],
    nyquist_hz: float,
) -> tuple[float, float]:
    """두 유효 주파수 대역의 교집을 반환하고 빈 교집·범위 위반을 fail-fast 한다.

    **이 함수는 이 저장소에서 유일한 정의다.** 예전에는 ``losses/anc_loss.py`` 와
    ``eval/metrics.py`` 에 하나씩, 의미는 같고 입력 처리와 오류 메시지가 다른 두 벌이
    있었다. 두 벌이 있으면 언젠가 갈라진다 — 두 벌은 이미 갈라져 있었다.
    """

    nyquist = float(nyquist_hz)
    if not math.isfinite(nyquist) or nyquist <= 0.0:
        raise ValueError(f"Nyquist 주파수는 유한한 양수여야 합니다: {nyquist_hz}")
    a = FrequencyBand.parse(first, name="첫 번째", nyquist_hz=nyquist)
    b = FrequencyBand.parse(second, name="두 번째", nyquist_hz=nyquist)
    return a.intersect(b).as_tuple()


def target_band_from_config(duct_cfg: dict) -> FrequencyBand:
    """``duct.yaml acoustics.realistic_target_band_hz`` — 목표 대역의 단일 출처."""

    acoustics = (duct_cfg or {}).get("acoustics", {}) or {}
    raw = acoustics.get("realistic_target_band_hz", list(DEFAULT_TARGET_BAND_HZ))
    return FrequencyBand.parse(raw, name="realistic_target_band_hz")


class BandPlan(BaseModel):
    """손실 대역과 보고 대역을 **명시적으로 분리**한 단일 출처.

    왜 두 개인가
    ------------
    한 개면 둘 중 하나를 잃는다.

    * **좁게** 두면(현행 ``optimize``) 평가도 같이 좁아져 **절대목표 1(고역도 제거)을
      검증할 방법이 없다.** 실제로 S(z) 를 1600Hz 까지 복구해 놓고도 목표 대역이
      800Hz 라 800–1600Hz 를 보지도 못하고 있었다.
    * **넓게** 두면 손실 대역이 함께 넓어져, 결함 3(대역 밖 15–22 dB 증폭)이 살아 있는
      상태에서 gradient 가 고역으로 쏠려 150–600Hz 가 나빠진다.

    그래서 ``optimize`` 는 "여기서 개선을 요구한다"(보수적), ``measure`` 는 "여기서
    측정만 한다"(넓게)로 **타입 수준에서 갈라 둔다**. 이 분리는 대역 밖 do-no-harm
    손실항의 설계와 짝을 이룬다 — 신뢰 대역 밖은 "개선 요구"가 아니라 "악화 금지"다.

    현재 배선
    --------
    ``optimize`` 는 기존 동작과 **정확히 같다**: ``S 신뢰대역 ∩ 목표대역 ∩ Nyquist``.
    ``measure`` 는 새로 생긴 값이고 아직 소비처가 없다 — 대역 확대는 do-no-harm 손실항과
    **함께** 들어가야 하므로 이 단계에서는 계산만 해 둔다(구조 정리 단계).
    """

    model_config = _FROZEN

    plant_trusted: FrequencyBand
    """S(z) 아티팩트가 신고한 신뢰 대역 (``consistency_band_hz``). 단일 출처는 NPZ."""

    target: FrequencyBand
    """``duct.yaml acoustics.realistic_target_band_hz``. 덕트 물리가 정하는 목표."""

    optimize: FrequencyBand
    """손실이 개선을 요구하는 대역. = ``plant_trusted ∩ target ∩ Nyquist``."""

    measure: FrequencyBand
    """보고·평가가 측정하는 대역. = ``plant_trusted ∩ Nyquist``. optimize 보다 넓다."""

    nyquist_hz: float

    @model_validator(mode="after")
    def _validate(self) -> "BandPlan":
        if not math.isfinite(self.nyquist_hz) or self.nyquist_hz <= 0.0:
            raise ValueError(f"Nyquist 주파수는 유한한 양수여야 합니다: {self.nyquist_hz}")
        if not self.measure.covers(self.optimize):
            raise ValueError(
                "보고 대역이 손실 대역을 덮지 않습니다 — 최적화하는 곳을 측정하지 "
                f"못합니다: measure={self.measure.as_tuple()} "
                f"optimize={self.optimize.as_tuple()}"
            )
        return self

    @classmethod
    def resolve(
        cls,
        *,
        plant_trusted_band_hz: Sequence[float],
        duct_cfg: dict,
        sample_rate: int | float,
    ) -> "BandPlan":
        """대역 계획을 **유일하게** 만드는 경로.

        이 저장소에서 ``intersect(sp.trusted_band_hz(), realistic_target_band_hz, fs/2)``
        를 손으로 쓰는 여섯 번째 복붙이 나오지 않게 하는 것이 이 메서드의 목적이다.
        """

        nyquist = float(sample_rate) / 2.0
        if not math.isfinite(nyquist) or nyquist <= 0.0:
            raise ValueError(f"sample_rate는 유한한 양수여야 합니다: {sample_rate}")
        trusted = FrequencyBand.parse(
            plant_trusted_band_hz, name="S 신뢰", nyquist_hz=nyquist
        )
        target = target_band_from_config(duct_cfg)
        nyquist_band = FrequencyBand(lo_hz=0.0, hi_hz=nyquist)
        measure = trusted.intersect(nyquist_band)
        optimize = measure.intersect(target)
        return cls(
            plant_trusted=trusted,
            target=target,
            optimize=optimize,
            measure=measure,
            nyquist_hz=nyquist,
        )


# --------------------------------------------------------------------------------------
# 지연과 lead
# --------------------------------------------------------------------------------------
def handoff_samples_from_config(duct_cfg: dict) -> int:
    """``duct.yaml secondary_path.handoff_extra_samples`` — handoff 의 단일 출처.

    ``.get("handoff_extra_samples", DEFAULT_HANDOFF_SAMPLES)`` 가 7개 파일에 복붙돼
    있었다. 기본값 분기가 여러 벌이면 하나만 고쳐도 아무도 모른다.
    """

    from ..config import DEFAULT_HANDOFF_SAMPLES

    secondary = (duct_cfg or {}).get("secondary_path", {}) or {}
    value = secondary.get("handoff_extra_samples", DEFAULT_HANDOFF_SAMPLES)
    handoff = int(value)
    if handoff < 0:
        raise ValueError(f"handoff_extra_samples 는 0 이상이어야 합니다: {value!r}")
    return handoff


class Lead(BaseModel):
    """digital-reference lead(샘플). **:meth:`PlantDelays.lead` 로만 만들 수 있다.**

    lead 는 "미래를 얼마나 미리 주는가"가 아니라 **P 와 S 의 지연 부기에서 유도되는
    양**이다::

        lead = max(0, S_bulk_delay + handoff − P_bulk_delay)

    이 관계를 손으로 다시 쓰면 반드시 갈라진다 — 실제로 trainer 와 게이트가 109 와
    113 으로 갈라졌다. 그래서 생성자를 직접 부르면 ``TypeError`` 다.

    실측 기준값(2026-08-05, 캡처 225546_f7b0fecd): S 1462 + handoff 256 − P 1602 = **116**.
    독립 캡처 9건에서 115/116/116/116/116/116/116/116/117 로 재현된다. 절대 지연은
    클록 드리프트(364~729ppm) 때문에 재현되지 않지만 **P−S = 140** 과 lead 는 재현된다.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    samples: int
    """실제로 쓰이는 lead. 인과성 때문에 0 으로 클램프된 값이다."""

    raw_samples: int
    """클램프 전 유도값. 음수면 "예측이 필요하다"는 뜻이고 숨기면 안 된다."""

    secondary_delay_samples: int
    handoff_samples: int
    primary_delay_samples: int

    token: Any = None

    @model_validator(mode="after")
    def _validate(self) -> "Lead":
        if self.token is not _DERIVE_TOKEN:
            raise TypeError(
                "Lead 는 PlantDelays.lead() 로만 만들 수 있습니다 — lead 를 손으로 쓰는 "
                "순간 그것이 두 번째 유도가 되고, 실제로 그렇게 109 와 113 이 갈라졌습니다"
            )
        expected_raw = (
            int(self.secondary_delay_samples)
            + int(self.handoff_samples)
            - int(self.primary_delay_samples)
        )
        if int(self.raw_samples) != expected_raw:
            raise ValueError(
                f"lead 유도 관계 위반: {self.raw_samples} != S {self.secondary_delay_samples}"
                f" + handoff {self.handoff_samples} − P {self.primary_delay_samples}"
            )
        if int(self.samples) != max(0, expected_raw):
            raise ValueError(
                f"lead 클램프 규약 위반: {self.samples} != max(0, {expected_raw})"
            )
        return self

    @property
    def is_clamped(self) -> bool:
        """유도값이 음수라 0 으로 잘렸는가. 잘렸다면 예측 요구가 남아 있다는 뜻이다."""

        return int(self.raw_samples) < 0

    def __int__(self) -> int:
        return int(self.samples)

    def __str__(self) -> str:  # pragma: no cover - 표시용
        return str(int(self.samples))


class PlantDelays(BaseModel):
    """측정된 플랜트의 지연 부기. lead 의 **유일한** 발원지.

    ``primary_delay_samples`` 와 ``secondary_delay_samples`` 는 반드시 **같은 캡처·같은
    앵커**에서 나온 값이어야 한다. 절대 지연은 캡처 간에 재현되지 않는다(실측 범위
    low-latency 1565~1659). 재현되는 물리량은 ``P − S`` 하나뿐이다.
    """

    model_config = _FROZEN

    primary_delay_samples: int
    secondary_delay_samples: int
    handoff_samples: int
    sample_rate: int

    @model_validator(mode="after")
    def _validate(self) -> "PlantDelays":
        for name in (
            "primary_delay_samples",
            "secondary_delay_samples",
            "handoff_samples",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} 는 0 이상이어야 합니다: {value}")
        if int(self.sample_rate) <= 0:
            raise ValueError(f"sample_rate 는 양수여야 합니다: {self.sample_rate}")
        return self

    @classmethod
    def from_config(
        cls,
        *,
        duct_cfg: dict,
        secondary_delay_samples: int,
        primary_delay_samples: int,
        sample_rate: int,
    ) -> "PlantDelays":
        """duct.yaml 의 handoff 와 실측 P/S 지연을 묶는다."""

        return cls(
            primary_delay_samples=int(primary_delay_samples),
            secondary_delay_samples=int(secondary_delay_samples),
            handoff_samples=handoff_samples_from_config(duct_cfg),
            sample_rate=int(sample_rate),
        )

    def lead(self) -> Lead:
        """digital-reference lead 를 유도한다. **저장소에서 유일한 유도 지점이다.**"""

        raw = (
            int(self.secondary_delay_samples)
            + int(self.handoff_samples)
            - int(self.primary_delay_samples)
        )
        return Lead(
            samples=max(0, raw),
            raw_samples=raw,
            secondary_delay_samples=int(self.secondary_delay_samples),
            handoff_samples=int(self.handoff_samples),
            primary_delay_samples=int(self.primary_delay_samples),
            token=_DERIVE_TOKEN,
        )

    @property
    def relative_delay_samples(self) -> int:
        """P − S. **이 측정의 유일한 물리 불변량**(실측 9건에서 139~141)."""

        return int(self.primary_delay_samples) - int(self.secondary_delay_samples)

    @property
    def secondary_total_delay_samples(self) -> int:
        """학습 플랜트가 실제로 쓰는 S 총지연 = 벌크지연 + handoff."""

        return int(self.secondary_delay_samples) + int(self.handoff_samples)

    def settle(self, *, fir_taps: int) -> "PlantSettle":
        """정착 구간을 유도한다. :meth:`PlantSettle.derive` 로 위임한다."""

        return PlantSettle.derive(
            secondary_delay_samples=int(self.secondary_delay_samples),
            handoff_samples=int(self.handoff_samples),
            fir_taps=int(fir_taps),
            sample_rate=int(self.sample_rate),
        )


class TrainingTimingContract(BaseModel):
    """합성·실측 학습 브랜치가 공유하는 digital-reference 시간축 계약.

    합성 브랜치의 ``d`` 는 NPZ ``delay_samples`` 만큼 0을 먼저
    놓고 compact P(z) FIR을 적용한다. 즉 NPZ 값은 FIR 피크까지의
    총지연이 아니라 **compact FIR 앞의 0 샘플 수**다. 따라서
    상호상관으로 관측되는 재생→ERR 지연은::

        primary_effective = primary_zeros_before_fir + argmax(abs(primary_fir))
        synthetic_total_advance = primary_effective + configured_lead

    이다. 실측 timeline lead도 이 총 선행량에서만 유도한다. 산식을 dataset과
    readiness에 각각 쓰면 한쪽은 pre-FIR 0만, 다른 쪽은 FIR 최대
    탭까지 세는 회귀가
    생기므로 모든 소비자는 이 immutable 객체를 전달받아 값만 읽는다.
    """

    model_config = _FROZEN

    schema_version: int = 2
    primary_zeros_before_fir_samples: int
    primary_fir_peak_offset_samples: int
    primary_effective_delay_samples: int
    secondary_delay_samples: int
    handoff_samples: int
    sample_rate: int
    raw_digital_reference_lead_samples: int
    digital_reference_lead_samples: int
    synthetic_total_advance_samples: int

    @model_validator(mode="after")
    def _validate(self) -> "TrainingTimingContract":
        if int(self.schema_version) != 2:
            raise ValueError(
                f"지원하지 않는 training timing schema: {self.schema_version}"
            )
        for name in (
            "primary_zeros_before_fir_samples",
            "primary_fir_peak_offset_samples",
            "secondary_delay_samples",
            "handoff_samples",
            "digital_reference_lead_samples",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} 는 0 이상이어야 합니다: {getattr(self, name)}")
        effective = int(self.primary_zeros_before_fir_samples) + int(
            self.primary_fir_peak_offset_samples
        )
        if int(self.primary_effective_delay_samples) != effective:
            raise ValueError(
                "primary effective 지연 유도 관계 위반: "
                f"{self.primary_effective_delay_samples} != "
                f"{self.primary_zeros_before_fir_samples} + "
                f"{self.primary_fir_peak_offset_samples}"
            )
        if int(self.sample_rate) <= 0:
            raise ValueError(f"sample_rate 는 양수여야 합니다: {self.sample_rate}")
        raw_lead = (
            int(self.secondary_delay_samples)
            + int(self.handoff_samples)
            - int(self.primary_zeros_before_fir_samples)
        )
        if int(self.raw_digital_reference_lead_samples) != raw_lead:
            raise ValueError(
                "raw lead 유도 관계 위반: "
                f"{self.raw_digital_reference_lead_samples} != "
                f"{self.secondary_delay_samples} + {self.handoff_samples} - "
                f"{self.primary_zeros_before_fir_samples}"
            )
        clamped_lead = max(0, raw_lead)
        if int(self.digital_reference_lead_samples) != clamped_lead:
            raise ValueError(
                "clamped lead 유도 관계 위반: "
                f"{self.digital_reference_lead_samples} != max(0, {raw_lead})"
            )
        total = effective + int(self.digital_reference_lead_samples)
        if int(self.synthetic_total_advance_samples) != total:
            raise ValueError(
                "합성 총 선행량 유도 관계 위반: "
                f"{self.synthetic_total_advance_samples} != {effective} + "
                f"{self.digital_reference_lead_samples}"
            )
        return self

    @classmethod
    def derive(
        cls,
        *,
        primary_fir: Sequence[float],
        plant_delays: PlantDelays,
    ) -> "TrainingTimingContract":
        """실제 P(z)와 :class:`PlantDelays`에서 계약을 만드는 유일한 경로.

        lead int를 받지 않는다. 반드시 ``PlantDelays.lead()``의 보호된
        유도 경로를 통해야 하며, P/S/handoff/raw/clamped 관계가 계약 자체에
        남아 checkpoint만 읽어도 재검증할 수 있다.
        """

        zeros_before_fir = int(plant_delays.primary_delay_samples)
        effective = cls.primary_effective_from_fir(zeros_before_fir, primary_fir)
        peak = effective - zeros_before_fir
        derived_lead = plant_delays.lead()
        lead = int(derived_lead.samples)
        return cls(
            primary_zeros_before_fir_samples=zeros_before_fir,
            primary_fir_peak_offset_samples=int(peak),
            primary_effective_delay_samples=effective,
            secondary_delay_samples=int(plant_delays.secondary_delay_samples),
            handoff_samples=int(plant_delays.handoff_samples),
            sample_rate=int(plant_delays.sample_rate),
            raw_digital_reference_lead_samples=int(derived_lead.raw_samples),
            digital_reference_lead_samples=lead,
            synthetic_total_advance_samples=effective + lead,
        )

    @staticmethod
    def primary_effective_from_fir(
        primary_zeros_before_fir_samples: int,
        primary_fir: Sequence[float],
    ) -> int:
        """P(z) 피크 지연만 필요한 감사 코드의 단일 유도 경로."""

        try:
            taps = [float(value) for value in primary_fir]
        except (TypeError, ValueError) as exc:
            raise ValueError("P(z) FIR 은 유한한 1차원 수열이어야 합니다") from exc
        zeros = int(primary_zeros_before_fir_samples)
        if zeros < 0 or not taps or not all(math.isfinite(value) for value in taps):
            raise ValueError("P(z) pre-FIR zeros/FIR 계약 위반")
        return zeros + max(range(len(taps)), key=lambda index: abs(taps[index]))

    @classmethod
    def from_data_config(cls, data_cfg: dict) -> "TrainingTimingContract":
        raw = (data_cfg or {}).get("training_timing_contract")
        if not isinstance(raw, dict):
            raise ValueError(
                "data.training_timing_contract 가 없습니다 — timeline lead는 실제 P(z) "
                "FIR을 포함한 단일 계약에서만 유도할 수 있습니다"
            )
        return cls.model_validate(raw)

    def recorded_lead_samples(self, recorded_delay_samples: float) -> int:
        """한 recorded session의 timeline lead ``K'``를 유도한다."""

        observed = float(recorded_delay_samples)
        if not math.isfinite(observed) or observed < 0.0:
            raise ValueError(
                f"recorded source→ERR 지연은 유한한 0 이상이어야 합니다: {observed}"
            )
        raw = int(round(float(self.synthetic_total_advance_samples) - observed))
        return max(0, raw)

    def recorded_total_advance_samples(
        self, *, recorded_delay_samples: float, mode: str
    ) -> float:
        """dataset이 실제로 만드는 recorded 총 선행량을 반환한다."""

        observed = float(recorded_delay_samples)
        if mode == "constant":
            lead = int(self.digital_reference_lead_samples)
        elif mode == "timeline":
            lead = self.recorded_lead_samples(observed)
        else:
            raise ValueError(f"지원하지 않는 recorded lead mode: {mode!r}")
        return observed + float(lead)

    def digest(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PlantSettle(BaseModel):
    """``e = d`` 로 고정되는 앞구간 길이. **학습 손실과 평가 warmup 의 단일 출처.**

    y 가 무엇이든 ``e[0:N] = d[0:N]`` 인 구간이 존재한다. S(z) 의 총지연(벌크지연 +
    handoff) 동안은 반노이즈가 아예 도달하지 않고, 그 뒤 FIR 길이만큼은 임펄스응답이
    아직 채워지지 않는다. 즉::

        N = secondary_delay + handoff + fir_taps

    이 구간을 손실에 넣으면 **달성 불가능한 목표**를 최적화하게 된다. 실측(2026-08-05):
    실측 d 로 이 구간을 포함한 채 완전상쇄를 가정해도 trusted 대역 하한이
    mean −20.3 / CVaR10 −10.1 / **worst −4.8 dB** 다. 합성 d 는 P(z) 지연 때문에 그
    구간이 비어 있어 −59.8 dB — 즉 **합성만 공짜였고 실측은 계속 노출돼 있었다.**
    평균 집계에서는 영향이 0.03 dB 로 작아 보이지만, CVaR 로 바꾸는 순간 하한이
    −4.8 dB 인 최악 아이템에 그래디언트가 집중된다.

    왜 타입인가
    ----------
    학습은 ``trainer.loss_start_sample``, 평가는 ``eval.recorded.resolve_warmup_samples``
    가 각자 "앞을 얼마나 버릴지"를 정했고 두 숫자는 **같은 양이 아니었다**(학습 0,
    평가 12000). 발생기 A 의 전형이다. 이제 두 곳 모두 이 클래스가 낸 값을 읽는다.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    samples: int
    secondary_delay_samples: int
    handoff_samples: int
    fir_taps: int
    sample_rate: int

    token: Any = None

    @model_validator(mode="after")
    def _validate(self) -> "PlantSettle":
        if self.token is not _DERIVE_TOKEN:
            raise TypeError(
                "PlantSettle 은 PlantSettle.derive() 로만 만들 수 있습니다 — 손으로 쓰는 "
                "순간 그것이 두 번째 유도가 되고, 학습 0 과 평가 12000 이 그렇게 갈라졌습니다"
            )
        for name in ("secondary_delay_samples", "handoff_samples", "fir_taps"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} 는 0 이상이어야 합니다: {getattr(self, name)}")
        if int(self.fir_taps) < 1:
            raise ValueError(f"fir_taps 는 1 이상이어야 합니다: {self.fir_taps}")
        if int(self.sample_rate) <= 0:
            raise ValueError(f"sample_rate 는 양수여야 합니다: {self.sample_rate}")
        expected = (
            int(self.secondary_delay_samples)
            + int(self.handoff_samples)
            + int(self.fir_taps)
        )
        if int(self.samples) != expected:
            raise ValueError(
                f"정착 구간 유도 관계 위반: {self.samples} != S "
                f"{self.secondary_delay_samples} + handoff {self.handoff_samples} + FIR "
                f"{self.fir_taps}"
            )
        return self

    @classmethod
    def derive(
        cls,
        *,
        secondary_delay_samples: int,
        handoff_samples: int,
        fir_taps: int,
        sample_rate: int,
    ) -> "PlantSettle":
        """**저장소에서 정착 구간을 유도하는 유일한 지점.**"""

        return cls(
            samples=(
                int(secondary_delay_samples) + int(handoff_samples) + int(fir_taps)
            ),
            secondary_delay_samples=int(secondary_delay_samples),
            handoff_samples=int(handoff_samples),
            fir_taps=int(fir_taps),
            sample_rate=int(sample_rate),
            token=_DERIVE_TOKEN,
        )

    @property
    def milliseconds(self) -> float:
        return 1000.0 * float(self.samples) / float(self.sample_rate)

    def __int__(self) -> int:
        return int(self.samples)

    def describe(self) -> str:  # pragma: no cover - 표시용
        return (
            f"{self.samples} samples ({self.milliseconds:.1f} ms) = S "
            f"{self.secondary_delay_samples} + handoff {self.handoff_samples} + FIR "
            f"{self.fir_taps}"
        )


# --------------------------------------------------------------------------------------
# 플랜트 지문
# --------------------------------------------------------------------------------------
class PlantFingerprint(BaseModel):
    """두 결과가 **같은 플랜트**에서 나왔는지 판정하기 위한 식별자.

    2026-08-04 사고: 파인튜닝 전 기준선은 S 지연 1342 / lead 109 / surrogate 물리였고
    후는 1465 / 113 / measured 였다. **서로 다른 물리**인데 "1.30 dB 개선"이라고 적혔고,
    비교를 막는 장치가 아무 데도 없었다. metrics 산출물이 지문을 들고 다니지 않으면
    이 사고는 구조적으로 반복된다.
    """

    model_config = _FROZEN

    primary_delay_samples: int
    secondary_delay_samples: int
    handoff_samples: int
    lead_samples: int
    """**지연 부기에서 유도된** lead. ``PlantDelays.lead()`` 의 값이다."""

    sample_rate: int
    physics_status: str
    optimize_band_hz: tuple[float, float]
    secondary_sha256: str | None = None
    primary_sha256: str | None = None
    capture_id: str | None = None

    configured_lead_samples: int | None = None
    """그 실행이 **실제로 쓴** lead. 유도값과 다를 수 있고, 다르다는 사실 자체가
    지문의 일부다.

    유도값만 기록하면 "지연은 같은데 서로 다른 lead 로 돌린 두 실행"이 같은 플랜트로
    보인다. 실제로 lead 를 4~7 샘플 바꾸면 150-600Hz 상쇄가 1 dB 넘게 달라지므로
    (실측 lead 스캔: 0 → −5.54 dB, 116 → −6.53 dB) 비교 가능성 판정에 반드시 들어가야
    한다. 유도값과의 일치 여부는 ``check_lead_agreement`` 가 따로 판정한다 — 지문은
    "무엇이었는가"를 기록할 뿐 "옳았는가"를 판정하지 않는다."""

    @classmethod
    def build(
        cls,
        *,
        delays: PlantDelays,
        lead: Lead,
        physics_status: str,
        bands: BandPlan,
        secondary_sha256: str | None = None,
        primary_sha256: str | None = None,
        capture_id: str | None = None,
        configured_lead_samples: int | None = None,
    ) -> "PlantFingerprint":
        if int(lead.samples) != int(delays.lead().samples):
            raise ValueError(
                f"lead 가 지연 부기와 맞지 않습니다: {lead.samples} != "
                f"{delays.lead().samples}"
            )
        return cls(
            primary_delay_samples=int(delays.primary_delay_samples),
            secondary_delay_samples=int(delays.secondary_delay_samples),
            handoff_samples=int(delays.handoff_samples),
            lead_samples=int(lead.samples),
            sample_rate=int(delays.sample_rate),
            physics_status=str(physics_status),
            optimize_band_hz=bands.optimize.as_tuple(),
            secondary_sha256=secondary_sha256,
            primary_sha256=primary_sha256,
            capture_id=capture_id,
            configured_lead_samples=(
                None if configured_lead_samples is None else int(configured_lead_samples)
            ),
        )

    def digest(self) -> str:
        """지문의 안정적인 16진 요약. 산출물에 한 칸으로 박아 둘 때 쓴다."""

        payload = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def differences(self, other: "PlantFingerprint") -> list[str]:
        """다른 필드 이름을 나열한다. 비어 있으면 같은 플랜트다."""

        mine = self.model_dump()
        theirs = other.model_dump()
        diffs: list[str] = []
        for key in sorted(mine):
            left, right = mine[key], theirs.get(key)
            if left is None or right is None:
                # 한쪽만 없는 필드(sha256 등)는 "다르다"고 단정하지 않는다.
                if left is None and right is None:
                    continue
                if key in {"secondary_sha256", "primary_sha256", "capture_id"}:
                    continue
            if left != right:
                diffs.append(f"{key}: {left!r} != {right!r}")
        return diffs

    def assert_same(self, other: "PlantFingerprint", *, context: str = "") -> None:
        diffs = self.differences(other)
        if diffs:
            where = f"[{context}] " if context else ""
            raise ValueError(
                f"{where}서로 다른 플랜트의 결과는 비교할 수 없습니다: "
                + ", ".join(diffs)
                + ". 같은 플랜트로 기준선을 다시 평가하세요."
            )
