"""안전장치 — anc_project 실기 검증 패턴 계승 + DL 특화 워치독.

전 모드(dl/fxlms) 공통 적용 (docs/06):
  1. 시작 시 ANC OFF          2. FadeGate 페이드 온/오프
  3. NaN 검출 + 출력 리미터    4. 출력 포화/RMS 워치독
  5. 출력 DC 차단 + DC 워치독  6. 발산 워치독
  7. 데드라인 미스율 워치독    8. 핸드오프 백로그 워치독

2026-08-06 실시간 감사에서 확정된 결함과 그 **발생기**
--------------------------------------------------------
감사에서 나온 6건은 증상이 서로 다르지만 발생기는 두 개뿐이다.

**발생기 B ("실패해본 적 없는 게이트")** — 워치독이 "감시 중"이라고 주장하는데
그 주장이 반증된 적이 없다.

* **S1** 클립 워치독이 DL 경로에서 **죽은 코드**였다. 모델 출력은
  ``0.2·tanh(y/0.2)`` 라 엄격히 ``|y| < 0.2`` 인데 워치독은 ``|y| > 0.2`` 를 셌다.
  clip_fraction 은 **구조적으로 항상 0** 이었다 — 리미터와 같은 값을 재고 있었다.
* **S3** 발산 워치독이 ``baseline_power == 0`` 이면 **조용히 비활성**됐다.
  감시할 수 없는 상태와 정상인 상태를 구분하지 않았다.
* **S6** NaN 이 ``nan_to_num`` 으로 조용히 삼켜졌다. 메시지가 없어 무한 무음을
  정상으로 오인한다.

→ 대응: 워치독을 :class:`WatchdogId` 로 **열거 가능**하게 만들고, 각각이 실제로
발동하는 것을 테스트가 확인한다(``tests/test_realtime_safety.py``). 열거에 있는데
발동 시나리오가 없으면 테스트가 실패한다. 감시 불가능한 상태는 **fail-closed** 다.

**발생기 A ("두 도메인 간 시간 정렬 부기")** — 같은 물리량을 두 곳에서 따로 유도하고
대조하지 않는다.

* **백로그 비대칭**: 출력은 1 hop 만 허용하는데 입력은 ``hop*8`` 을 허용했다.
  두 리터럴이 서로 다른 파일 위치에 흩어져 있었고 학습 플랜트의 ``handoff_extra_samples``
  (256 = 1 hop)와 대조되지 않았다. 추론이 뒤처지면 실효 핸드오프가 8 hop(42.7 ms)까지
  조용히 늘어나 **상쇄가 증폭으로 바뀐다**.
* **S2** 데드라인 워치독이 ``elif had_data: streak = 0`` 이라 교대 미스(1,0,1,0)를
  영원히 못 잡았다. 상쇄음이 절반만 나가는 가장 나쁜 음향 상태가 무한 지속된다.

→ 대응: 백로그 예산은 :class:`PipelineHandoffBudget` **한 곳에서만** 유도되고
입출력 비대칭은 생성 시점에 거부된다. 스트릭 카운터는 저장소에서 전부 제거하고
:class:`RateWindow` (슬라이딩 윈도 사건율)로 대체했다 — 교대 실패에 면역이다.

핫패스 규약
----------
콜백 마감은 5.33 ms 다. 이 모듈에서 블록당 하는 일은 numpy 축약 몇 개와 deque
갱신뿐이다. pydantic 검증은 **생성 시점(설정 로드)** 에만 돈다.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from ..dsp.filters import DCBlocker
from ..dsp.timing import handoff_samples_from_config

__all__ = [
    "BlockObservation",
    "BlockVerdict",
    "FadeGate",
    "HARDWARE_PROTECTION_WATCHDOGS",
    "OutputBlockReport",
    "PipelineHandoffBudget",
    "PowerEMA",
    "RateWindow",
    "SafetyLimits",
    "SafetySupervisor",
    "SafetyWindows",
    "WatchdogId",
]


_FROZEN = ConfigDict(frozen=True, extra="forbid")
_FROZEN_ANY = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

# ``PipelineHandoffBudget`` 을 손으로 만들 수 없게 하는 토큰 (dsp/timing.Lead 와 같은 규약).
_BUDGET_TOKEN = object()


# ======================================================================================
# 워치독 목록 — 이 열거가 "감시하고 있다"는 주장의 단일 출처다.
# ======================================================================================
class WatchdogId(str, Enum):
    """실시간 워치독 식별자.

    여기 이름을 추가하면 ``tests/test_realtime_safety.py`` 가 **그것을 발동시키는
    시나리오**를 요구한다. 발동시킬 수 없는 워치독은 죽은 코드다 — S1 이 정확히
    그랬다.
    """

    NONFINITE_OUTPUT = "runtime_nonfinite_output"
    OUTPUT_DC = "runtime_output_dc"
    OUTPUT_SATURATION = "runtime_output_saturation"
    OUTPUT_RMS = "runtime_output_rms"
    DEADLINE_MISS_RATE = "runtime_deadline_miss_rate"
    DIVERGENCE = "runtime_divergence"
    HANDOFF_BACKLOG = "runtime_handoff_backlog"


HARDWARE_PROTECTION_WATCHDOGS: frozenset[WatchdogId] = frozenset(
    {WatchdogId.NONFINITE_OUTPUT, WatchdogId.OUTPUT_DC}
)
"""**측정 모드에서도 절대 무력화되지 않는** 워치독.

나머지는 음향 성능 보호라 캘리브레이션(의도적으로 큰 처프를 내는 모드)에서는
자문(advisory)으로 내릴 수 있다. 그러나 DC 와 NaN 은 **스피커 보이스코일**과
직결된 하드웨어 보호다 — 어떤 모드에서도 mute 한다.
"""


# ======================================================================================
# 설정 (경계에서 1회 검증)
# ======================================================================================
_LEGACY_KEYS: dict[str, str] = {
    "clip_streak_mute": (
        "연속 클립 스트릭은 (a) DL 경로에서 리미터와 같은 값을 재 항상 0 이었고 "
        "(b) 교대 실패에 취약하다 — saturation_rate_mute / output_rms_ratio 가 대체한다"
    ),
    "deadline_miss_mute": (
        "연속 미스 스트릭은 교대 미스(1,0,1,0)를 영원히 못 잡는다 — "
        "deadline_miss_rate_mute(슬라이딩 윈도 미스율)가 대체한다"
    ),
}


class SafetyLimits(BaseModel):
    """``runtime.yaml safety:`` 의 검증된 표현. 경계에서 한 번만 만든다.

    임계값을 dict 에서 ``cfg.get(...)`` 로 그때그때 꺼내면 (a) 오타가 조용히
    기본값으로 떨어지고 (b) 물리적으로 불가능한 값(음수 비율, ratio>1)이 통과한다.
    여기서는 둘 다 생성 시점에 막힌다.
    """

    model_config = _FROZEN

    # ---- 출력 리미터 ----
    control_limit: float = 0.2
    """상쇄 출력 피크 제한. 모델 소프트 리미터와 **같은 값**이라는 점이 S1 의 핵심이다."""

    fade_ms: float = 20.0

    # ---- S1: 포화 워치독 (리미터와 다른 값을 잰다) ----
    saturation_ratio: float = 0.95
    """``|y| > saturation_ratio × control_limit`` 이면 그 샘플은 포화로 센다.

    tanh 리미터는 ``|y| < limit`` 를 보장할 뿐 ``0.95×limit`` 는 막지 않는다
    (``tanh(u) > 0.95`` ⟺ ``u > 1.83``). 따라서 이 지표는 **실제로 발동 가능**하다.
    """

    saturation_fraction: float = 0.5
    """블록을 '포화'로 판정하는 샘플 비율.

    진폭 A 의 정현파는 ``|y| > 0.95A`` 인 샘플이 ``(2/π)·arccos(0.95) ≈ 0.20`` 뿐이다.
    0.5 를 넘으려면 파형이 사각파에 가까워야 한다 = 소프트 리미터가 신호를 지배하는
    상태 = 출력이 더 이상 의도한 안티노이즈가 아니다.
    """

    saturation_window_s: float = 1.0
    saturation_rate_mute: float = 0.5

    # ---- S1b: 출력 RMS 상한 ----
    output_rms_ratio: float = 0.6
    """``rms > output_rms_ratio × control_limit`` 인 블록은 과출력으로 센다.

    limit 진폭의 정현파 rms 는 ``0.707×limit`` 이므로 0.6 은 "정현파 최대 출력에
    근접한 상태가 지속됨"을 잡는다. 피크가 아니라 **에너지** 기준이라 포화 워치독과
    잡는 파형이 다르다.
    """

    rms_window_s: float = 1.0
    rms_rate_mute: float = 0.5

    # ---- S2: 데드라인 미스율 ----
    deadline_window_s: float = 1.0
    deadline_miss_rate_mute: float = 0.2
    """최근 ``deadline_window_s`` 중 미스 비율이 이 값을 넘으면 mute.

    스트릭이 아니라 **비율**인 이유: 교대 미스(1,0,1,0)는 스트릭을 매 블록 리셋시켜
    영원히 잡히지 않았다. 그 상태가 음향적으로 가장 나쁘다 — 상쇄음이 절반만 나가
    남은 절반은 그대로 방사되고, 위상이 끊겨 오히려 증폭될 수 있다.
    """

    # ---- S3: 발산 ----
    divergence_ratio: float = 4.0
    divergence_hold_s: float = 0.5
    divergence_rate_mute: float = 0.5
    baseline_floor_power: float = 1.0e-8
    """이보다 작은 베이스라인 파워는 '측정된 것이 없다'로 취급한다(≈ −80 dBFS)."""

    baseline_grace_s: float = 3.0
    """유효 베이스라인 없이 ANC 를 켜 둘 수 있는 시간. 넘으면 **fail-closed** mute.

    예전에는 ``baseline_power == 0`` 이면 발산 워치독이 조용히 비활성이었다.
    감시할 수 없는 상태는 안전한 상태가 아니다.
    """

    # ---- S7: 출력 DC ----
    output_dc_limit_ratio: float = 0.05
    """``|DC| > output_dc_limit_ratio × control_limit`` 가 지속되면 mute."""

    output_dc_window_s: float = 0.2
    output_dc_blocker_r: float = 0.9995
    """출력 DC 차단기 극점. ``f_c = (1−r)·fs/2π ≈ 3.8 Hz`` (48 kHz).

    입력에만 DCBlocker 가 있었고 **출력에는 아무 보호도 없었다**. 모델이 DC 0.19 를
    내면 앰프가 보이스코일에 DC 를 계속 흘린다(하드웨어 손상). 차단기는 신뢰대역
    하단 150 Hz 에서 위상을 1.5° 만 돌리므로 상쇄 성능에 사실상 영향이 없다.
    """

    # ---- 백로그(핸드오프) ----
    handoff_window_s: float = 1.0
    handoff_drop_rate_mute: float = 0.2

    # ---- 모드 ----
    measurement_mode: bool = False
    """캘리브레이션 등 의도적으로 큰 출력을 내는 모드.

    음향 성능 워치독은 자문으로 내려가지만 :data:`HARDWARE_PROTECTION_WATCHDOGS` 는
    그대로 mute 한다.
    """

    legacy_notes: tuple[str, ...] = ()
    """설정에 남아 있는 폐기된 키에 대한 안내 (런타임 시작 시 출력)."""

    @model_validator(mode="after")
    def _validate(self) -> "SafetyLimits":
        if not 0.0 < self.control_limit <= 1.0:
            raise ValueError(f"control_limit 은 (0, 1] 이어야 합니다: {self.control_limit}")
        if self.fade_ms < 0.0:
            raise ValueError(f"fade_ms 는 0 이상이어야 합니다: {self.fade_ms}")
        if not 0.0 < self.saturation_ratio < 1.0:
            raise ValueError(
                "saturation_ratio 는 (0, 1) 이어야 합니다 — 1.0 이면 리미터와 같은 값을 "
                f"재게 되어 DL 경로에서 죽은 워치독이 됩니다 (S1): {self.saturation_ratio}"
            )
        for name in ("saturation_fraction", "output_rms_ratio", "output_dc_limit_ratio"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} 은 (0, 1] 이어야 합니다: {value}")
        for name in (
            "saturation_rate_mute",
            "rms_rate_mute",
            "deadline_miss_rate_mute",
            "divergence_rate_mute",
            "handoff_drop_rate_mute",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} 은 (0, 1] 이어야 합니다: {value}")
        for name in (
            "saturation_window_s",
            "rms_window_s",
            "deadline_window_s",
            "divergence_hold_s",
            "output_dc_window_s",
            "handoff_window_s",
            "baseline_grace_s",
        ):
            value = float(getattr(self, name))
            if not value > 0.0:
                raise ValueError(f"{name} 은 양수여야 합니다: {value}")
        if self.divergence_ratio <= 1.0:
            raise ValueError(
                f"divergence_ratio 는 1 보다 커야 합니다: {self.divergence_ratio}"
            )
        if self.baseline_floor_power <= 0.0:
            raise ValueError(
                f"baseline_floor_power 는 양수여야 합니다: {self.baseline_floor_power}"
            )
        if not 0.0 < self.output_dc_blocker_r < 1.0:
            raise ValueError(
                f"output_dc_blocker_r 은 (0, 1) 이어야 합니다: {self.output_dc_blocker_r}"
            )
        return self

    # ---- 유도 임계값 (여기가 단일 출처 — 호출부에서 곱하지 말 것) ----
    @property
    def saturation_threshold(self) -> float:
        return float(self.control_limit) * float(self.saturation_ratio)

    @property
    def rms_threshold(self) -> float:
        return float(self.control_limit) * float(self.output_rms_ratio)

    @property
    def dc_threshold(self) -> float:
        return float(self.control_limit) * float(self.output_dc_limit_ratio)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "SafetyLimits":
        """``runtime.yaml`` 의 ``safety:`` 블록을 검증해 만든다.

        폐기된 키는 **조용히 무시하지 않는다** — 무엇이 왜 없어졌는지 안내를 남긴다.
        모르는 키는 오타일 수 있으므로 거부한다(``extra="forbid"``).
        """

        raw = dict(cfg or {})
        notes: list[str] = []
        for key, why in _LEGACY_KEYS.items():
            if key in raw:
                raw.pop(key)
                notes.append(f"safety.{key} 는 더 이상 쓰이지 않습니다 — {why}")
        raw.pop("legacy_notes", None)
        return cls(**raw, legacy_notes=tuple(notes))


class SafetyWindows(BaseModel):
    """초 → 블록 수 변환의 **단일 지점**.

    예전에는 ``int(seconds * sample_rate / block_size)`` 가 감시자 생성자 안에
    한 번, 다른 곳에 또 한 번 나타났다. 같은 변환이 두 곳에 있으면 갈라진다
    (군집 A). 여기서 한 번에 유도하고 나머지는 읽기만 한다.
    """

    model_config = _FROZEN

    sample_rate: int
    block_size: int
    saturation_blocks: int
    rms_blocks: int
    deadline_blocks: int
    divergence_blocks: int
    dc_blocks: int
    handoff_blocks: int
    baseline_grace_blocks: int

    @model_validator(mode="after")
    def _validate(self) -> "SafetyWindows":
        if self.sample_rate <= 0 or self.block_size <= 0:
            raise ValueError(
                f"sample_rate/block_size 는 양수여야 합니다: {self.sample_rate}/{self.block_size}"
            )
        for name, value in self.model_dump().items():
            if name.endswith("_blocks") and int(value) < 1:
                raise ValueError(f"{name} 은 1 이상이어야 합니다: {value}")
        return self

    @staticmethod
    def blocks_for(seconds: float, sample_rate: int, block_size: int) -> int:
        return max(1, int(round(float(seconds) * int(sample_rate) / int(block_size))))

    @classmethod
    def derive(
        cls, limits: SafetyLimits, sample_rate: int, block_size: int
    ) -> "SafetyWindows":
        def blocks(seconds: float) -> int:
            return cls.blocks_for(seconds, sample_rate, block_size)

        return cls(
            sample_rate=int(sample_rate),
            block_size=int(block_size),
            saturation_blocks=blocks(limits.saturation_window_s),
            rms_blocks=blocks(limits.rms_window_s),
            deadline_blocks=blocks(limits.deadline_window_s),
            divergence_blocks=blocks(limits.divergence_hold_s),
            dc_blocks=blocks(limits.output_dc_window_s),
            handoff_blocks=blocks(limits.handoff_window_s),
            baseline_grace_blocks=blocks(limits.baseline_grace_s),
        )


# ======================================================================================
# 파이프라인 백로그 예산 — 입출력 비대칭을 **구조적으로 불가능**하게 만든다
# ======================================================================================
class PipelineHandoffBudget(BaseModel):
    """콜백↔추론 링버퍼의 백로그 허용치. :meth:`derive` 로만 만들 수 있다.

    왜 타입인가
    ----------
    ``keep_backlog=frames`` (콜백) 와 ``keep_backlog=self.hop * 8`` (추론 스레드) 이
    서로 다른 파일 위치에 리터럴로 흩어져 있었고, **학습 플랜트가 가정한 핸드오프
    (duct.yaml ``handoff_extra_samples`` = 256 = 1 hop)와 대조되지 않았다.**
    추론이 뒤처지면 입력단이 최대 8 hop(42.7 ms) 오래된 블록을 처리하므로 실효
    핸드오프가 학습 가정의 8배가 된다 — 그 지연에서는 안티노이즈의 위상이 뒤집혀
    **상쇄가 증폭으로 바뀐다**.

    ``pop_latest(n, keep_backlog=K)`` 는 백로그를 K 로 줄인 뒤 그 중 **가장 오래된**
    n 을 꺼낸다. 따라서 처리되는 블록은 쓰기 헤드보다 ``K/hop − 1`` hop 만큼 낡아 있다.
    ``K = hop`` 이면 추가 지연이 0 이고 실효 핸드오프는 설계값 1 hop 그대로다.
    """

    model_config = _FROZEN_ANY

    hop_samples: int
    handoff_samples: int
    """duct.yaml 의 학습 플랜트 핸드오프. **단일 출처는 dsp.timing 이다.**"""

    input_keep_backlog_samples: int
    output_keep_backlog_samples: int
    token: Any = None

    @model_validator(mode="after")
    def _validate(self) -> "PipelineHandoffBudget":
        if self.token is not _BUDGET_TOKEN:
            raise TypeError(
                "PipelineHandoffBudget 는 derive() 로만 만들 수 있습니다 — 백로그 예산을 "
                "손으로 쓰는 순간 그것이 두 번째 유도가 되고, 실제로 그렇게 입력 8 hop / "
                "출력 1 hop 의 비대칭이 생겼습니다"
            )
        if self.hop_samples <= 0:
            raise ValueError(f"hop 은 양수여야 합니다: {self.hop_samples}")
        if self.handoff_samples != self.hop_samples:
            raise ValueError(
                "실시간 파이프라인의 핸드오프는 정확히 1 hop 입니다. duct.yaml "
                f"handoff_extra_samples={self.handoff_samples} 가 hop={self.hop_samples} 와 "
                "다르면 학습 플랜트와 런타임이 서로 다른 지연을 가정하게 됩니다"
            )
        for name in ("input_keep_backlog_samples", "output_keep_backlog_samples"):
            value = int(getattr(self, name))
            if value < self.hop_samples:
                raise ValueError(
                    f"{name} 은 최소 1 hop({self.hop_samples}) 이어야 합니다: {value}"
                )
        if self.input_keep_backlog_samples != self.output_keep_backlog_samples:
            raise ValueError(
                "입력/출력 백로그가 비대칭입니다: "
                f"입력 {self.input_keep_backlog_samples} vs 출력 "
                f"{self.output_keep_backlog_samples}. 한쪽만 늘리면 실효 핸드오프가 "
                "조용히 늘어나 학습 플랜트 가정이 깨집니다 (상쇄→증폭)"
            )
        if self.effective_handoff_samples != self.handoff_samples:
            raise ValueError(
                f"실효 핸드오프 {self.effective_handoff_samples} 가 학습 가정 "
                f"{self.handoff_samples} 와 다릅니다"
            )
        return self

    @property
    def effective_handoff_samples(self) -> int:
        """최악의 경우 실효 핸드오프. 백로그가 1 hop 을 넘는 만큼 그대로 더해진다."""

        extra_in = max(0, int(self.input_keep_backlog_samples) - int(self.hop_samples))
        extra_out = max(0, int(self.output_keep_backlog_samples) - int(self.hop_samples))
        return int(self.handoff_samples) + extra_in + extra_out

    @classmethod
    def derive(cls, *, duct_cfg: Mapping[str, Any] | None, hop: int) -> "PipelineHandoffBudget":
        """duct.yaml 의 핸드오프(**단일 출처**)에서 백로그 예산을 유도한다."""

        handoff = handoff_samples_from_config(dict(duct_cfg or {}))
        hop_samples = int(hop)
        return cls(
            hop_samples=hop_samples,
            handoff_samples=int(handoff),
            input_keep_backlog_samples=hop_samples,
            output_keep_backlog_samples=hop_samples,
            token=_BUDGET_TOKEN,
        )


# ======================================================================================
# 슬라이딩 윈도 (스트릭 카운터의 대체물)
# ======================================================================================
class RateWindow:
    """최근 N 블록의 사건 발생률.

    **스트릭 카운터를 쓰지 마라.** ``elif ok: streak = 0`` 은 교대 실패(1,0,1,0)를
    영원히 못 잡는다. 실기에서 데드라인 워치독이 정확히 그 이유로 무력했다.
    """

    __slots__ = ("capacity", "min_blocks", "_buf", "_hits")

    def __init__(self, capacity: int, min_blocks: int | None = None) -> None:
        self.capacity = max(1, int(capacity))
        # 판정에 필요한 최소 표본. 창이 다 찰 때까지 기다리면 반응이 느려지고,
        # 1블록으로 판정하면 잡음에 흔들린다 — 1/4 창을 절충으로 쓴다.
        default_min = max(1, (self.capacity + 3) // 4)
        self.min_blocks = max(1, int(min_blocks)) if min_blocks else default_min
        self._buf: deque[int] = deque(maxlen=self.capacity)
        self._hits = 0

    def update(self, hit: bool) -> None:
        if len(self._buf) == self.capacity:
            self._hits -= self._buf[0]
        value = 1 if hit else 0
        self._buf.append(value)
        self._hits += value

    @property
    def count(self) -> int:
        return len(self._buf)

    @property
    def rate(self) -> float:
        return (self._hits / len(self._buf)) if self._buf else 0.0

    def exceeds(self, threshold: float) -> bool:
        return len(self._buf) >= self.min_blocks and self.rate > float(threshold)

    def reset(self) -> None:
        self._buf.clear()
        self._hits = 0


class MeanWindow:
    """최근 N 블록의 산술평균 — DC 처럼 **부호가 있는** 양에 쓴다."""

    __slots__ = ("capacity", "min_blocks", "_buf", "_sum")

    def __init__(self, capacity: int, min_blocks: int | None = None) -> None:
        self.capacity = max(1, int(capacity))
        default_min = max(1, (self.capacity + 3) // 4)
        self.min_blocks = max(1, int(min_blocks)) if min_blocks else default_min
        self._buf: deque[float] = deque(maxlen=self.capacity)
        self._sum = 0.0

    def update(self, value: float) -> None:
        if len(self._buf) == self.capacity:
            self._sum -= self._buf[0]
        item = float(value)
        self._buf.append(item)
        self._sum += item

    @property
    def count(self) -> int:
        return len(self._buf)

    @property
    def mean(self) -> float:
        return (self._sum / len(self._buf)) if self._buf else 0.0

    def exceeds_magnitude(self, threshold: float) -> bool:
        return len(self._buf) >= self.min_blocks and abs(self.mean) > float(threshold)

    def reset(self) -> None:
        self._buf.clear()
        self._sum = 0.0


# ======================================================================================
# 블록 단위 관측/판정 (핫패스 — frozen dataclass, pydantic 아님)
# ======================================================================================
@dataclass(frozen=True, slots=True)
class OutputBlockReport:
    """:meth:`SafetySupervisor.limit_output` 의 산출물.

    ``clipped_fraction`` 과 ``saturated_fraction`` 은 **다른 것을 잰다**:
    전자는 하드 리미터(``|y| > limit``)에 걸린 비율이고 — DL 모델은 tanh 때문에
    구조적으로 0 이다 — 후자는 리미터 **직전 영역**(``> 0.95×limit``) 비율이라
    실제로 발동한다. 둘을 같은 값으로 재는 것이 S1 의 결함이었다.
    """

    signal: np.ndarray
    frames: int
    nonfinite_fraction: float
    clipped_fraction: float
    saturated_fraction: float
    rms: float
    dc_in: float
    """DC 차단기 **이전** 블록 평균 = 모델이 내려던 DC. 워치독은 이것을 본다."""

    dc_out: float
    """실제로 앰프로 나가는 블록 평균. 차단기가 살아 있으면 ≈ 0 이다."""


@dataclass(frozen=True, slots=True)
class BlockObservation:
    """워치독이 한 블록에 대해 보는 모든 것."""

    anc_on: bool
    output: OutputBlockReport
    error_power: float
    baseline_power: float
    baseline_valid: bool
    had_output_data: bool
    stale_input_samples: int = 0

    def __post_init__(self) -> None:
        # 블록당 1회 — 5.33 ms 마감에서 비교 몇 개는 무시할 수 있다.
        # (numpy 스칼라 호출은 이 기기에서 10 µs 라 math 를 쓴다)
        if not math.isfinite(self.error_power) or self.error_power < 0.0:
            raise ValueError(f"error_power 가 유효하지 않습니다: {self.error_power}")
        if not math.isfinite(self.baseline_power) or self.baseline_power < 0.0:
            raise ValueError(f"baseline_power 가 유효하지 않습니다: {self.baseline_power}")
        if int(self.stale_input_samples) < 0:
            raise ValueError(
                f"stale_input_samples 는 0 이상이어야 합니다: {self.stale_input_samples}"
            )


@dataclass(frozen=True, slots=True)
class BlockVerdict:
    mute: bool
    fired: tuple[WatchdogId, ...] = ()
    messages: tuple[str, ...] = ()


_CLEAN = BlockVerdict(False, (), ())


# ======================================================================================
# 감시자
# ======================================================================================
class SafetySupervisor:
    """콜백 안에서 매 블록 호출되는 안전 감시자 — 상태와 자동 mute 판단.

    설계 규칙 세 가지 (감사에서 배운 것):

    1. **리미터와 다른 값을 재라.** 보호 장치와 같은 임계값을 세는 감시자는 죽는다.
    2. **스트릭을 쓰지 마라.** 전부 슬라이딩 윈도 비율이다.
    3. **감시할 수 없으면 끈다.** 베이스라인이 없으면 발산을 판정할 수 없고,
       판정할 수 없는 상태로 상쇄음을 계속 내보내지 않는다 (fail-closed).
    """

    def __init__(self, cfg: Mapping[str, Any] | None, sample_rate: int, block_size: int) -> None:
        self.limits = SafetyLimits.from_config(cfg)
        self.windows = SafetyWindows.derive(self.limits, sample_rate, block_size)
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)

        # 하위 호환: 호출부가 읽던 이름 (리미터 값 자체는 여전히 단일 출처다).
        self.control_limit = float(self.limits.control_limit)
        # 핫패스에서 pydantic property 를 매 블록 다시 계산하지 않기 위한 사본.
        # limits 가 frozen 이라 이 사본이 갈라질 수 없다.
        self._limit = float(self.limits.control_limit)
        self._saturation_threshold = float(self.limits.saturation_threshold)

        self._dc_blocker = DCBlocker(self.limits.output_dc_blocker_r)
        self._saturation = RateWindow(self.windows.saturation_blocks)
        self._rms = RateWindow(self.windows.rms_blocks)
        self._deadline = RateWindow(self.windows.deadline_blocks)
        self._divergence = RateWindow(self.windows.divergence_blocks)
        self._handoff = RateWindow(self.windows.handoff_blocks)
        self._dc = MeanWindow(self.windows.dc_blocks)
        self._blocks_without_baseline = 0

        self.trip_counts: dict[WatchdogId, int] = {item: 0 for item in WatchdogId}

    # ---------------------------------------------------------------- 출력 처리
    def limit_output(self, y: np.ndarray) -> OutputBlockReport:
        """NaN 방어 → 하드 리미터 → **출력 DC 차단** → 통계.

        DC 차단기가 리미터 뒤에 오는 이유: 리미터를 먼저 걸어야 발산한 값(1e30)이
        차단기 상태를 오염시키지 않는다.

        핫패스다. 이 Jetson 에서 numpy 축약 한 번이 15~30 µs 이므로 **패스 수를 센다**:
        정상 블록에서 도는 축약은 sum(1) + max|·|(1) + dot(1) + sum(1) = 4 회이고
        나머지(클립 비율·비유한 비율 카운트)는 피크가 임계값을 넘을 때만 돈다.
        측정값 ≈ 150 µs/블록 (마감 5330 µs).
        """

        values = np.asarray(y, dtype=np.float32).reshape(-1)
        frames = int(values.size)
        if frames == 0:
            empty = np.empty(0, dtype=np.float32)
            return OutputBlockReport(empty, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # 합 하나로 DC 와 비유한 검출을 동시에 한다. NaN/±Inf 가 하나라도 있으면
        # float64 누적합이 반드시 비유한이 된다 (float32 오버플로는 누적 dtype 으로 회피).
        total = float(values.sum(dtype=np.float64))
        nonfinite_fraction = 0.0
        if not math.isfinite(total):
            nonfinite_fraction = float(
                frames - int(np.count_nonzero(np.isfinite(values)))
            ) / frames
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            total = float(values.sum(dtype=np.float64))
        dc_in = total / frames

        limit = self._limit
        peak = float(np.max(np.abs(values)))
        if peak > limit:
            clipped_fraction = float(
                int(np.count_nonzero(np.abs(values) > limit))
            ) / frames
            values = np.clip(values, -limit, limit)
            peak = limit
        else:
            clipped_fraction = 0.0

        saturation_threshold = self._saturation_threshold
        if peak > saturation_threshold:
            saturated_fraction = float(
                int(np.count_nonzero(np.abs(values) > saturation_threshold))
            ) / frames
        else:
            saturated_fraction = 0.0
        energy = float(np.dot(values, values))
        rms = math.sqrt(energy / frames) if energy > 0.0 else 0.0

        # 포화·RMS 는 **모델이 낸 신호**(리미터 직후)에서 잰다. 차단기는 3.8 Hz
        # 고역통과라 이 대역에서 사실상 항등이고, 판정 대상은 모델 상태다.
        out = self._dc_blocker.process(values)
        # 차단기 과도응답이 리미터를 0.05% 넘길 수 있다. "출력은 절대 리미터를 넘지
        # 않는다"는 불변식을 재기 위해 제자리에서 한 번 더 조인다 (lfilter 가 준
        # 새 배열이므로 호출자 버퍼를 건드리지 않는다).
        np.clip(out, -limit, limit, out=out)
        dc_out = float(out.sum(dtype=np.float64)) / frames
        return OutputBlockReport(
            signal=out,
            frames=frames,
            nonfinite_fraction=nonfinite_fraction,
            clipped_fraction=clipped_fraction,
            saturated_fraction=saturated_fraction,
            rms=rms,
            dc_in=dc_in,
            dc_out=dc_out,
        )

    # ---------------------------------------------------------------- 블록 판정
    def check_block(self, obs: BlockObservation) -> BlockVerdict:
        """블록마다 호출. ``mute=True`` 면 ANC 를 즉시 OFF 해야 한다."""

        if not obs.anc_on:
            # 게이트가 닫혀 있으면 스피커로 나가는 것이 없다. 다음 ON 은 깨끗한
            # 창에서 시작해야 하므로 상태를 비운다.
            self.reset_windows()
            return _CLEAN

        fired: list[WatchdogId] = []
        messages: list[str] = []
        out = obs.output
        limits = self.limits

        def fire(watchdog: WatchdogId, text: str) -> None:
            fired.append(watchdog)
            messages.append(f"[{watchdog.value}] {text}")
            self.trip_counts[watchdog] += 1

        # ---- S6: NaN/Inf — 조용히 삼키지 않는다 ----
        if out.nonfinite_fraction > 0.0:
            fire(
                WatchdogId.NONFINITE_OUTPUT,
                f"엔진 출력에 NaN/Inf 가 {out.nonfinite_fraction:.1%} 섞였습니다 — "
                "해당 샘플을 0 으로 대체하고 ANC 를 끕니다 (무한 무음을 정상으로 "
                "오인하지 않기 위해 반드시 보고합니다)",
            )

        # ---- S7: 출력 DC (하드웨어 보호) ----
        self._dc.update(out.dc_in)
        if self._dc.exceeds_magnitude(limits.dc_threshold):
            fire(
                WatchdogId.OUTPUT_DC,
                f"상쇄 출력에 DC {self._dc.mean:+.4f} 가 "
                f"{limits.output_dc_window_s:.2f}s 지속됩니다 (한계 "
                f"±{limits.dc_threshold:.4f}) — 스피커 보이스코일 보호를 위해 ANC OFF. "
                "출력 DC 차단기가 이미 제거하고 있으나 모델 상태가 비정상입니다",
            )

        # ---- S1: 포화 (리미터와 다른 값을 잰다) ----
        self._saturation.update(out.saturated_fraction >= limits.saturation_fraction)
        if self._saturation.exceeds(limits.saturation_rate_mute):
            fire(
                WatchdogId.OUTPUT_SATURATION,
                f"상쇄 출력이 리미터 근처(|y| > {limits.saturation_threshold:.3f})에 "
                f"눌려 있습니다: 최근 {self._saturation.count} 블록 중 "
                f"{self._saturation.rate:.0%} — 소프트 리미터가 신호를 지배하면 출력은 "
                "더 이상 의도한 안티노이즈가 아닙니다. 볼륨/모델 확인",
            )

        # ---- S1b: 출력 RMS 상한 ----
        self._rms.update(out.rms > limits.rms_threshold)
        if self._rms.exceeds(limits.rms_rate_mute):
            fire(
                WatchdogId.OUTPUT_RMS,
                f"상쇄 출력 RMS 가 상한 {limits.rms_threshold:.3f} 을 최근 "
                f"{self._rms.count} 블록 중 {self._rms.rate:.0%} 초과합니다 — 과출력",
            )

        # ---- S2: 데드라인 미스율 (스트릭 아님) ----
        self._deadline.update(not obs.had_output_data)
        if self._deadline.exceeds(limits.deadline_miss_rate_mute):
            fire(
                WatchdogId.DEADLINE_MISS_RATE,
                f"추론이 콜백을 따라가지 못합니다: 최근 {self._deadline.count} 블록 중 "
                f"{self._deadline.rate:.0%} 가 무음(허용 "
                f"{limits.deadline_miss_rate_mute:.0%}) — 상쇄음이 띄엄띄엄 나가면 "
                "위상이 끊겨 증폭이 될 수 있습니다. 엔진을 trt 로 바꾸거나 tiny 모델/"
                "폴백을 사용하세요",
            )

        # ---- 백로그: 학습 플랜트가 가정한 핸드오프를 넘었는가 ----
        self._handoff.update(int(obs.stale_input_samples) > 0)
        if self._handoff.exceeds(limits.handoff_drop_rate_mute):
            fire(
                WatchdogId.HANDOFF_BACKLOG,
                f"입력 백로그가 1 hop 을 넘어 최근 {self._handoff.count} 블록 중 "
                f"{self._handoff.rate:.0%} 에서 오래된 입력을 버렸습니다 — 실효 핸드오프가 "
                "학습 플랜트 가정에서 이탈했습니다 (이 상태의 안티노이즈는 상쇄가 아니라 "
                "증폭이 될 수 있습니다)",
            )

        # ---- S3: 발산 — 감시할 수 없으면 끈다 (fail-closed) ----
        if not obs.baseline_valid:
            self._blocks_without_baseline += 1
            self._divergence.reset()
            if self._blocks_without_baseline >= self.windows.baseline_grace_blocks:
                fire(
                    WatchdogId.DIVERGENCE,
                    f"발산 워치독이 판정할 베이스라인이 없습니다(파워 "
                    f"{obs.baseline_power:.3e} ≤ 하한 {limits.baseline_floor_power:.1e}) — "
                    f"{limits.baseline_grace_s:.1f}s 동안 감시 불가 상태였으므로 ANC 를 "
                    "끕니다. ANC OFF 상태에서 소음을 먼저 켜 베이스라인을 잡으세요",
                )
        else:
            self._blocks_without_baseline = 0
            self._divergence.update(
                obs.error_power > limits.divergence_ratio * obs.baseline_power
            )
            if self._divergence.exceeds(limits.divergence_rate_mute):
                ratio_db = 10.0 * float(
                    np.log10(
                        max(obs.error_power, 1e-30) / max(obs.baseline_power, 1e-30)
                    )
                )
                fire(
                    WatchdogId.DIVERGENCE,
                    f"에러 파워가 베이스라인 대비 {ratio_db:+.1f}dB 로 최근 "
                    f"{self._divergence.count} 블록 중 {self._divergence.rate:.0%} 동안 "
                    "발산 기준을 넘었습니다 — 자동 OFF",
                )

        if not fired:
            return _CLEAN

        if limits.measurement_mode:
            mute = any(item in HARDWARE_PROTECTION_WATCHDOGS for item in fired)
            if not mute:
                messages = [
                    f"{text} (측정 모드: 자문만, ANC 유지)" for text in messages
                ]
        else:
            mute = True

        # 창을 비운다. mute 면 다음 ON 시도가 남은 창 때문에 즉시 다시 꺼지지 않게
        # 하기 위해서고, 자문(측정 모드)이면 같은 메시지가 매 블록 쏟아지는 것을
        # 막기 위해서다 — 창 하나(≈1s)에 한 번만 보고된다.
        self.reset_windows()
        # 메시지는 verdict 만 들고 다닌다 — 감시자 안에 사본을 하나 더 두면
        # 어느 쪽이 진짜인지 갈라진다 (군집 A 와 같은 발생기).
        return BlockVerdict(mute=mute, fired=tuple(fired), messages=tuple(messages))

    # ---------------------------------------------------------------- 상태
    def reset_windows(self) -> None:
        for window in (
            self._saturation,
            self._rms,
            self._deadline,
            self._divergence,
            self._handoff,
        ):
            window.reset()
        self._dc.reset()
        self._blocks_without_baseline = 0

    def baseline_is_valid(self, baseline_power: float, initialized: bool) -> bool:
        """베이스라인이 발산 판정에 쓸 수 있는 값인가. **판정 규칙의 단일 출처.**"""

        return bool(
            initialized
            and math.isfinite(float(baseline_power))
            and float(baseline_power) > float(self.limits.baseline_floor_power)
        )


# ======================================================================================
# 기존 블록 (변경 없음)
# ======================================================================================
class FadeGate:
    """0↔1 선형 페이드 게이트 (anc_project/main_realtime_anc.py 이식)."""

    def __init__(self, fade_samples: int, initial: float = 0.0) -> None:
        self.fade_samples = max(0, int(fade_samples))
        self.current = float(initial)
        self.target = float(initial)
        self.remaining = 0

    def set_target(self, target: float) -> None:
        target = float(np.clip(target, 0.0, 1.0))
        if target == self.target:
            return
        self.target = target
        self.remaining = self.fade_samples
        if self.fade_samples == 0:
            self.current = self.target

    def process(self, frames: int) -> np.ndarray:
        if frames <= 0:
            return np.empty(0, dtype=np.float32)
        if self.remaining <= 0 or self.current == self.target:
            self.current = self.target
            self.remaining = 0
            return np.full(frames, self.current, dtype=np.float32)
        count = min(frames, self.remaining)
        start = self.current
        ramp = start + (self.target - start) * (
            np.arange(1, count + 1, dtype=np.float64) / self.remaining
        )
        output = np.empty(frames, dtype=np.float32)
        output[:count] = ramp.astype(np.float32)
        self.current = float(ramp[-1])
        self.remaining -= count
        if count < frames:
            output[count:] = np.float32(self.target)
            self.current = self.target
            self.remaining = 0
        return output


class PowerEMA:
    """지수이동평균 파워 미터 (anc_project 이식)."""

    def __init__(self, sample_rate: int, time_constant: float = 0.5) -> None:
        self.sample_rate = sample_rate
        self.time_constant = max(1.0e-3, float(time_constant))
        self.value = 0.0
        self.initialized = False

    def update(self, block: np.ndarray) -> float:
        values = np.asarray(block, dtype=np.float64)
        power = float(np.mean(values * values)) if values.size else 0.0
        alpha = float(np.exp(-values.size / (self.sample_rate * self.time_constant)))
        if not self.initialized:
            self.value = power
            self.initialized = True
        else:
            self.value = alpha * self.value + (1.0 - alpha) * power
        return self.value
