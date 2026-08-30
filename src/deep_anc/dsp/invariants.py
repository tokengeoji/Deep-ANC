"""교차 도메인 불변식 검사기 — 두 스트림이 만나는 경계에서 **같은 코드**로 돈다.

왜 이 모듈이 있는가
------------------
2026-08-05 결함 군집 분석의 결론: 게이트 9개가 전부 PASS 인데 플랜트가 33%(형상 기준
50%) 틀렸고, recorded QA 는 80/80 PASS 인데 학습 데이터의 시간축이 붕괴해 있었다.
공통 원인은 검사가 없었던 게 아니라 **각 도메인이 자기 안만 봤다**는 것이다.

* 측정은 채널마다 따로 분석해 ``P−S 상대 τ`` 를 볼 수 없었다.
* QA 는 파일 하나하나의 RMS/clip/길이만 보고 **채널 사이의 관계**를 한 번도 안 봤다.
* 평가는 metrics 에 플랜트 지문을 남기지 않아 다른 플랜트끼리 비교해도 막지 못했다.
* 게이트와 trainer 는 lead 를 각자 유도해 서로 어긋난 채로 둘 다 "통과"했다.

여기 있는 네 검사는 전부 **두 값을 대조**한다. 각각을 자기 도메인 안에서 재구현하지
말고 이 모듈을 호출하라 — 재구현이 곧 다음 사고다.

핫패스 금지: 오디오 콜백 안에서 부르지 마라. 경계에서 블록당 1회 이하로 쓴다.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from .interleaved_probe import relative_tau_outliers
from .timing import Lead, PlantDelays, PlantFingerprint

if TYPE_CHECKING:  # pragma: no cover - 순환 import 를 피하려 타입에서만 참조한다
    from ..data.timeline import DelayTrajectory

__all__ = [
    "MAX_STREAM_DELAY_P95_P5_SAMPLES",
    "MAX_STREAM_DELAY_ROBUST_STD_SAMPLES",
    "MIN_STREAM_DELAY_VALID_WINDOWS",
    "MIN_STREAM_DELAY_VALID_WINDOW_RATIO",
    "InvariantResult",
    "InvariantViolation",
    "check_corpus_disjoint",
    "check_lead_agreement",
    "check_measured_delay_agreement",
    "check_plant_fingerprint_match",
    "check_relative_tau_constancy",
    "check_stream_coherence",
    "check_stream_delay_stability",
    "derive_playback_to_error_delay_samples",
    "measure_stream_delay_trajectory",
]


MIN_COHERENCE_SAMPLES = 256
"""코히런스 추정에 요구하는 최소 길이(샘플).

이보다 짧으면 세그먼트가 1개뿐이라 코히런스가 **항상 1** 로 나오거나(자기 자신과의
정규화) 0/0 으로 NaN 이 된다. 어느 쪽이든 판정에 쓸 수 없는데, 두 경우 모두
``값 < 임계`` 가 False 라 **통과처럼 보인다**. 그래서 값을 지어내지 않고 거부한다.
"""


class InvariantViolation(ValueError):
    """불변식이 깨졌다. 값이 아니라 **두 값의 관계**가 틀렸다는 뜻이다."""


class InvariantResult(BaseModel):
    """검사 결과. 통과/실패와 **측정값**을 함께 들고 다닌다.

    측정값을 함께 반환하는 이유: 통과했다는 말만 남으면 다음 사람이 마진을 알 수 없다.
    "PASS" 만 기록된 게이트 9개가 전부 무용지물이었던 것이 정확히 그 사고다.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    ok: bool
    detail: str
    measured: dict[str, Any] = {}

    def raise_if_failed(self) -> "InvariantResult":
        if not self.ok:
            raise InvariantViolation(f"[{self.name}] {self.detail}")
        return self


def check_relative_tau_constancy(
    tau_primary: Sequence[float] | np.ndarray,
    tau_secondary: Sequence[float] | np.ndarray,
    *,
    tolerance_samples: float = 3.0,
    name: str = "relative_tau_constancy",
) -> InvariantResult:
    """P−S 상대 τ 가 반복에 무관한 상수인가.

    두 채널은 **같은 DAC·같은 출력 스트림**의 인터리브다. 따라서 τ_P − τ_S 는 설계
    원리상 상수여야 하고, 튀면 출력 버퍼가 한쪽 채널에서만 미끄러진 것이다.

    실측(2026-08-04 사고): ``[0, 1.2, ..., 1.4, 32.1, 32.2, 31.7, 30.3, 29.1]`` —
    반복 11 에서 1.4 → 32 샘플 점프. 게이트는 ``delay_spread 32`` 를 아티팩트가
    스스로 써 넣은 허용치 48 과 비교해 **통과시켰다**. range 하나로는 "11개가 1.2,
    5개가 32" 라는 구조를 볼 수 없다.

    임계 3.0 의 근거(캡처 10건, 236 반복): 정상군 최대 편차 **1.99**, 오염군 최소
    **4.32**. 그 사이가 비어 있어 3.0 이 양쪽에 1.5x / 1.44x 여유를 남긴다. MAD 스케일
    임계는 쓰지 않는다 — 오염이 과반이면 MAD 가 부풀어 슬립 블록을 통째로 통과시킨다.
    """

    outliers, deviation, centre = relative_tau_outliers(
        tau_primary, tau_secondary, tolerance_samples=float(tolerance_samples)
    )
    indices = [int(i) for i in np.flatnonzero(outliers)]
    relative = np.asarray(tau_primary, dtype=np.float64).reshape(-1) - np.asarray(
        tau_secondary, dtype=np.float64
    ).reshape(-1)
    step = float(np.max(np.abs(np.diff(relative)))) if relative.size > 1 else 0.0
    measured = {
        "centre_samples": float(centre),
        "max_deviation_samples": float(np.max(deviation)) if deviation.size else 0.0,
        "max_consecutive_step_samples": step,
        "outlier_indices": indices,
        "tolerance_samples": float(tolerance_samples),
    }
    if indices:
        return InvariantResult(
            name=name,
            ok=False,
            detail=(
                f"P−S 상대 τ 가 상수가 아닙니다: 중앙값 {centre:.2f} 에서 최대 "
                f"{measured['max_deviation_samples']:.2f} 샘플 벗어남 "
                f"(허용 {float(tolerance_samples):.2f}), 위반 반복 {indices} — "
                "출력 버퍼 프레임 슬립"
            ),
            measured=measured,
        )
    return InvariantResult(
        name=name,
        ok=True,
        detail=(
            f"P−S 상대 τ 가 상수입니다: 중앙 {centre:.2f} 샘플, 최대 편차 "
            f"{measured['max_deviation_samples']:.2f} ≤ {float(tolerance_samples):.2f}"
        ),
        measured=measured,
    )


def _median_coherence(
    a: np.ndarray, b: np.ndarray, *, sample_rate: int, band_hz: tuple[float, float], nperseg: int
) -> float:
    from scipy.signal import coherence

    n = int(min(a.size, b.size))
    # 짧은 신호로도 "계산은 된다" — 그리고 그 결과가 NaN 이면 ``NaN < 임계`` 가 False 라
    # **통과처럼 보인다.** 조용히 통과하는 것이 이 저장소에서 반복된 실패 방식이므로,
    # 의미 있는 추정이 불가능한 길이는 값을 지어내지 않고 거부한다.
    if n < MIN_COHERENCE_SAMPLES:
        raise ValueError(
            f"코히런스를 추정하기에 신호가 너무 짧습니다: {n} < {MIN_COHERENCE_SAMPLES} "
            "samples — 짧은 신호의 코히런스는 1 에 가깝게 편향되거나 NaN 이 됩니다"
        )
    if n < nperseg:
        nperseg = max(64, 1 << int(math.floor(math.log2(max(n, 64)))))
    freqs, values = coherence(a[:n], b[:n], fs=float(sample_rate), nperseg=int(nperseg))
    mask = (freqs >= float(band_hz[0])) & (freqs <= float(band_hz[1]))
    if not np.any(mask):
        raise ValueError(
            f"대역 {band_hz} 에 코히런스 빈이 없습니다 (nperseg={nperseg}, fs={sample_rate})"
        )
    value = float(np.median(values[mask]))
    if not math.isfinite(value):
        raise ValueError(
            "코히런스가 유한하지 않습니다 (한쪽 신호의 대역 내 전력이 0 일 수 있습니다) — "
            "판정 불가를 통과로 세지 않습니다"
        )
    return value


def check_stream_coherence(
    playback: Sequence[float] | np.ndarray,
    capture: Sequence[float] | np.ndarray,
    *,
    sample_rate: int,
    band_hz: tuple[float, float] = (150.0, 600.0),
    min_coherence: float = 0.60,
    nperseg: int = 8192,
    name: str = "stream_coherence",
    control: Sequence[float] | np.ndarray | None = None,
) -> InvariantResult:
    """재생 신호와 캡처 신호의 대응이 살아 있는가 (coh², 대역 중앙값).

    이것이 **학습이 배워야 하는 관계 그 자체**다. recorded QA 가 80/80 PASS 를 낸
    이유는 무엇을 봤는가가 아니라 무엇을 **안 봤는가**다 — RMS·clip·길이·메타데이터만
    보고 채널 사이의 시간 관계를 한 번도 보지 않았다.

    실측(2026-08-04, 150–600Hz 중앙값, nperseg 8192):
    ``coh²(source→ERR) = 0.021~0.126`` 인데 ``coh²(REF→ERR) = 0.959~0.991`` 이었다.
    두 값을 **함께** 보면 "음향은 멀쩡한데 소프트웨어 타임베이스가 깨졌다" 는 진단까지
    자동으로 나온다. 그래서 ``control`` 인자로 음향 대조군을 함께 받는다.

    임계 0.60 은 정상(0.96~0.99)과 붕괴(0.02~0.13) 사이의 넓은 골짜기에서 골랐다.
    """

    a = np.asarray(playback, dtype=np.float64).reshape(-1)
    b = np.asarray(capture, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2:
        raise ValueError("코히런스 검사에는 두 신호 각각 2샘플 이상이 필요합니다")
    value = _median_coherence(
        a, b, sample_rate=int(sample_rate), band_hz=tuple(band_hz), nperseg=int(nperseg)
    )
    measured: dict[str, Any] = {
        "coherence": value,
        "band_hz": [float(band_hz[0]), float(band_hz[1])],
        "min_coherence": float(min_coherence),
        "nperseg": int(nperseg),
    }
    control_value: float | None = None
    if control is not None:
        c = np.asarray(control, dtype=np.float64).reshape(-1)
        control_value = _median_coherence(
            c, b, sample_rate=int(sample_rate), band_hz=tuple(band_hz), nperseg=int(nperseg)
        )
        measured["control_coherence"] = control_value

    if value < float(min_coherence):
        detail = (
            f"재생→캡처 {band_hz[0]:.0f}-{band_hz[1]:.0f}Hz 결맞음 {value:.3f} < "
            f"{float(min_coherence):.2f} — 학습이 배워야 할 관계가 없습니다"
        )
        if control_value is not None and control_value >= float(min_coherence):
            detail += (
                f" (음향 대조군은 {control_value:.3f} 로 정상 = 음향이 아니라 "
                "녹음 소프트웨어 타임베이스 문제)"
            )
        return InvariantResult(name=name, ok=False, detail=detail, measured=measured)
    return InvariantResult(
        name=name,
        ok=True,
        detail=(
            f"재생→캡처 결맞음 {value:.3f} ≥ {float(min_coherence):.2f} "
            f"({band_hz[0]:.0f}-{band_hz[1]:.0f}Hz)"
        ),
        measured=measured,
    )


def check_plant_fingerprint_match(
    before: PlantFingerprint,
    after: PlantFingerprint,
    *,
    name: str = "plant_fingerprint_match",
) -> InvariantResult:
    """두 결과가 같은 플랜트에서 나왔는가.

    2026-08-04 사고: 전 = S 지연 1342 / lead 109 / surrogate, 후 = 1465 / 113 / measured
    를 비교해 "1.30 dB 개선" 이라고 적었다. 서로 다른 물리다.
    """

    diffs = before.differences(after)
    measured = {
        "before_digest": before.digest(),
        "after_digest": after.digest(),
        "differences": diffs,
    }
    if diffs:
        return InvariantResult(
            name=name,
            ok=False,
            detail=(
                "서로 다른 플랜트의 결과는 비교할 수 없습니다: "
                + ", ".join(diffs)
                + ". 같은 플랜트로 기준선을 다시 평가하세요."
            ),
            measured=measured,
        )
    return InvariantResult(
        name=name,
        ok=True,
        detail=f"같은 플랜트입니다 (digest {before.digest()[:12]})",
        measured=measured,
    )


def check_lead_agreement(
    configured_lead: int,
    delays: PlantDelays,
    *,
    tolerance_samples: int = 0,
    name: str = "lead_agreement",
) -> InvariantResult:
    """설정된 lead 가 **측정된 지연에서 유도되는 값**과 맞는가.

    lead 는 자유 변수처럼 보이지만 아니다 — ``S + handoff − P`` 로 결정된다. 설정과
    유도값이 갈라진 채로 양쪽이 각자 "통과"한 것이 커밋 aaeef41 의 사고다(109 vs 113).

    ``tolerance_samples`` 는 init checkpoint 선택에만 쓰는 허용 오차다. 배포·학습
    설정에는 0 을 써라 — 실측 감쇠 계산상 δ=16 샘플이면 600Hz 에서 오히려 +1.40 dB
    증폭이다.
    """

    derived: Lead = delays.lead()
    mismatch = abs(int(configured_lead) - int(derived.samples))
    measured = {
        "configured_lead_samples": int(configured_lead),
        "derived_lead_samples": int(derived.samples),
        "raw_lead_samples": int(derived.raw_samples),
        "mismatch_samples": int(mismatch),
        "tolerance_samples": int(tolerance_samples),
        "primary_delay_samples": int(delays.primary_delay_samples),
        "secondary_delay_samples": int(delays.secondary_delay_samples),
        "handoff_samples": int(delays.handoff_samples),
        "relative_delay_samples": int(delays.relative_delay_samples),
    }
    if mismatch > int(tolerance_samples):
        return InvariantResult(
            name=name,
            ok=False,
            detail=(
                f"lead 가 P/S 지연에서 유도되는 값과 다릅니다: 설정={int(configured_lead)}, "
                f"유도={int(derived.samples)} "
                f"(S {delays.secondary_delay_samples} + handoff {delays.handoff_samples} "
                f"− P {delays.primary_delay_samples}), 차이 {mismatch} > "
                f"허용 {int(tolerance_samples)} samples"
            ),
            measured=measured,
        )
    return InvariantResult(
        name=name,
        ok=True,
        detail=(
            f"lead {int(configured_lead)} 가 P/S 지연 유도값과 정합합니다 "
            f"(S {delays.secondary_delay_samples} + handoff {delays.handoff_samples} "
            f"− P {delays.primary_delay_samples} = {int(derived.samples)})"
        ),
        measured=measured,
    )


def coherence_from_delay_jitter(freq_hz: float, robust_std_samples: float, sample_rate: float) -> float:
    """지연 지터 σ 가 주파수 ``f`` 에서 남기는 크기제곱 코히런스 상한.

    ``coh²(f) = exp(-(2π f σ / fs)²)``. 지연 궤적의 요동을 가우시안 위상 요동으로 보면
    나오는 표준 결과이고, 이 저장소의 실측이 그것을 지지한다 — 재정렬 47 세션에서
    고역 coh² 와 σ 의 상관이 **r = −0.812**, 저역 예측 p50 0.992 vs 실측 0.947,
    고역 예측 0.936 vs 실측 0.824 (나머지 0.11 은 가산 잡음 바닥).

    **지연 자체는 코히런스를 낮추지 않는다.** 낮추는 것은 지연의 *변동*이다. 그래서
    정렬이 잘 된 세션도 지터가 크면 고역이 무너진다.
    """

    import math

    return float(math.exp(-((2.0 * math.pi * float(freq_hz) * float(robust_std_samples) / float(sample_rate)) ** 2)))


def delay_jitter_limit_for_band(
    top_hz: float, min_coherence: float, sample_rate: float
) -> float:
    """제어를 주장하는 대역 상단에서 코히런스를 지키는 **지터 상한**을 유도한다.

    ``σ_max = (fs / 2πf_top) · √(−ln coh_min)``. :func:`coherence_from_delay_jitter` 의 역이다.

    왜 이것이 필요한가 (2026-08-06)
    ------------------------------
    이전 상한 ``8.0`` 은 저역의 정상/오염 분리에서만 정해졌다(정상 1.2~3.0 vs 오염
    25~120). 그런데 σ=8.0 에서 **1600 Hz 의 달성 가능 coh² 는 0.060** 이다. 즉 그 게이트를
    통과한 세션이 절대목표 1의 고역 절반에 대해 아무 정보도 담고 있지 않을 수 있었고,
    실제로 고역을 보는 게이트는 저장소에 **하나도 없었다**.

    임계를 손으로 고르지 않는다 — 신뢰대역 상단과 이미 선언된 코히런스 요구
    (:data:`MIN_STREAM_COHERENCE`)에서 유도한다.
    """

    import math

    if not 0.0 < float(min_coherence) < 1.0:
        raise ValueError(f"min_coherence 는 (0,1) 이어야 합니다: {min_coherence}")
    if float(top_hz) <= 0.0 or float(sample_rate) <= 0.0:
        raise ValueError(f"주파수는 양수여야 합니다: top={top_hz}, fs={sample_rate}")
    return float(
        (float(sample_rate) / (2.0 * math.pi * float(top_hz)))
        * math.sqrt(-math.log(float(min_coherence)))
    )


MIN_STREAM_COHERENCE = 0.60
"""``source → ERR`` 크기제곱 코히런스의 하한. **제어를 주장하는 대역 전체에 적용된다.**

실측 정상 0.96~0.99 / 붕괴 0.02~0.13 사이의 넓은 골짜기에서 고른 값이다.
2026-08-06 이전에는 이 값이 ``recorded_qa`` 안에 있었고 **150–600 Hz 에만** 적용됐다.
신뢰대역이 [150,1600] 으로 넓어진 뒤에도 그대로였다 — 즉 요구 대역의 위쪽 절반은
아무도 보지 않았다. 지금은 이 하나의 선언에서 고역 게이트와 지터 상한이 함께 유도된다.
"""

CONTROLLED_BAND_TOP_HZ = 1600.0
"""제어를 주장하는 대역의 상단. 지터 상한의 유도원이다.

S(z) 복구 후 신뢰대역이 [150, 1600] 이 됐다(HANDOFF 플랜트 복구 결과). 1633 Hz 는 덕트
평면파 컷오프이고 그 위는 가진조차 하지 않았다.
"""

MAX_STREAM_DELAY_ROBUST_STD_SAMPLES = delay_jitter_limit_for_band(
    CONTROLLED_BAND_TOP_HZ, MIN_STREAM_COHERENCE, 48_000.0
)
"""``playback → capture`` 지연 궤적 산포(1.4826×MAD)의 상한. **유도값 ≈ 3.41 샘플.**

옛 값 ``8.0`` 은 실측 전수의 골짜기에서 정했다 (2026-08-06, ``data/recorded_broken``
앞 30 초, 단일 추정기)::

    정상군 = 재정렬된 47 세션 (source_aligned → ERR)   robust-std 1.221 ~ 2.992
    오염군 = 같은 세션의 원본 80 개 (source → ERR)     robust-std 25.159 ~ 120.620

골짜기 [2.992, 25.159] 는 통째로 비어 있고(8.4배) 8.0 은 그 기하평균(8.68)의 내림이었다.
**그 근거는 여전히 맞다 — 다만 부족하다.** σ=8.0 을 허용하면 1600 Hz 에서 달성 가능한
coh² 가 **0.060** 이다. 붕괴한 세션은 걸러내지만, 통과한 세션의 고역이 쓸모없을 수
있고 그것을 보는 게이트가 없었다.

그래서 임계를 대역에서 유도한다: ``σ_max = (fs/2πf_top)·√(−ln coh_min)``.
f_top = 1600 Hz, coh_min = 0.60 → **3.413 샘플**.

이 변경은 게이트를 **강화**한다(8.0 → 3.41). 실측 47 세션은 σ 최대 2.992 이므로 전부
그대로 통과한다 — 즉 기존 데이터를 버리지 않으면서 고역 보호를 얻는다. 오염군 최소
25.159 와의 여유도 7.4배로 오히려 커진다.
"""

MAX_STREAM_DELAY_P95_P5_SAMPLES = 48.0
"""``playback → capture`` 지연 궤적 변동폭(p95−p5)의 상한.

같은 전수 측정::

    정상군 47 세션   p95−p5   4.834 ~  11.980
    오염군 80 세션   p95−p5 228.136 ~ 370.770

골짜기 [11.98, 228.14] 는 19배로 비어 있고, 기하평균 52.3 을 내림해 48.0 을 쓴다
(정상군 최대의 4.0배, 오염군 최소의 1/4.8). p95−p5 를 쓰고 min/max range 를 쓰지
않는 이유: range 는 창 하나가 로브를 건너뛰면 통째로 날아간다 — 반증 #17 이 인용한
세션 101813 의 창 30개 중 27개가 125~150 인데 이상치 3개(479/487/6311)가 range 를
6186 으로 만들어 정상 세션을 기각시킨 것이 정확히 그 실패다."""

MIN_STREAM_DELAY_VALID_WINDOWS = 8
"""판정에 필요한 최소 유효창 수.

유효창이 없으면 통계가 NaN 이고 ``NaN > 임계`` 는 False 라 **조용히 통과**한다.
그래서 "측정 불가"를 통과가 아니라 실패로 떨어뜨린다. 실측 30 초 발췌의 유효창은
정상군 410~476 / 오염군 335~476 이라 8 은 어느 실데이터도 건드리지 않는다 —
이 값이 발동한다는 것은 추정 자체가 실패했다는 뜻이다."""

MIN_STREAM_DELAY_VALID_WINDOW_RATIO = 0.77
"""유효창 **비율** 하한 — 로버스트 통계의 fail-open 구멍을 막는다.

로버스트 통계만으로는 부족하다는 것을 직접 확인했다: 창마다 지연을 1200 샘플씩
옮긴 합성 신호를 넣으면 추정기가 추적을 놓쳐 91 창 중 **17 창만** 유효해지고, 그
살아남은 17 창끼리는 서로 붙어 있어 robust-std 0.56 / p95−p5 2.39 로 **PASS 한다.**
"추적을 놓쳤다"가 "안정하다"로 보고되는 것이다.

⚠ 2026-08-06 통합 검증 — **0.50 이었고, 그 값이 실제로 뚫렸다.**
이전 주석은 "이 축은 정상/오염을 가르는 축이 아니라 판정 가능성만 가른다"고 적고
하한을 오염군 최저(0.704) **아래**인 0.50 에 두었다. 그런데 판정량(robust-std,
p95−p5)은 **살아남은 창만** 보고, 탐색범위(``coarse_search_samples``) 밖으로 튄 창은
통계에서 아예 빠진다. 그래서 세션 일부만 프레임 슬립으로 어긋나면 어긋난 구간이
통째로 버려지고 남은 구간은 깨끗해 보인다 — 유일한 목격자가 이 비율 하나뿐이다.

실측 재현(실제 QA 코드경로 ``_validate_alignment``, 출하 임계, 세션
``20260804_102441_file`` 의 source_aligned 를 앞 구간만 이동):
  * 앞 15% 를  700 샘플 이동 → vwr 0.651 / 오류 **0건** (구 임계 0.50 에서 PASS)
  * 앞 15% 를 2000 샘플(42 ms) 이동 → vwr 0.653 / 오류 **0건**
  * 앞 30% 를  700/2000 샘플 이동 → vwr 0.695 / robust-std 1.2~1.3 / p95−p5 5.2
    (전 축 PASS. 같은 신호를 옛 원시 std/range 게이트는 std 913.7 / range 2050 으로 잡았다)
즉 30 초 중 4.5 초가 42 ms 어긋난 세션이 QA 를 전부 통과하고 있었다.

임계 0.77 의 근거(전수 실측):
  * 정상군 47 세션(재정렬본) 유효창 비율 — 전 파일 0.857~0.997 / QA 발췌 0.861~0.979
  * 프레임 슬립 주입                     0.651 ~ 0.695
  * 추적 실패 합성 신호                  0.187
골짜기 [0.695, 0.857] 의 기하평균 0.7717 을 내림해 0.77 을 쓴다 — 정상군 최저의
1/1.11, 슬립 주입 최대의 1.11배로 양쪽에 같은 여유를 둔다. 오염군(원본 source.wav,
최저 0.704)도 이제 이 축에서 함께 떨어지지만 그것은 오기각이 아니다: 그 무리는
robust-std 25~121 로 어차피 FAIL 이고, 한 결함이 두 축에 걸리는 것은 정상이다.
**이 값을 낮추려면 위 슬립 주입을 다시 돌려 무엇이 통과하게 되는지 먼저 적어라.**"""


def measure_stream_delay_trajectory(
    playback: Sequence[float] | np.ndarray,
    capture: Sequence[float] | np.ndarray,
    *,
    sample_rate: int,
    settings: Any | None = None,
) -> "DelayTrajectory":
    """재생→캡처 지연 궤적. **자체 구현이 아니라 단일 출처로 위임한다.**

    2026-08-06 반증 #14/#18 — 이 함수는 예전에 자기 추정기를 들고 있었다: 대역제한도
    PHAT 도 없이 1 초 창 광대역 상관의 ``|argmax|`` 를 ±8000 에서 취하고 생 std/range
    로 판정했다. 같은 ``(source_aligned, ERR)`` 쌍에 대해
    :func:`deep_anc.data.timeline.estimate_lag_track` 은 robust-std **1.84** 를,
    이 함수는 **1106.55** 를 냈다 — **600배**. 세션 101813 의 창 30개 중 27개가
    125~150 인데 이상치 3개(479/487/6311)가 통계를 통째로 날린 것이다.
    coh²=0.926 인 신호가 129 ms 를 떠다닐 수는 없으니 데이터가 아니라 추정기가 틀렸다.

    그 결과 제대로 재정렬된 47 세션 중 **22개(47%)** 가 QA 에서 오기각됐다.

    그래서 지금 이 함수에는 상관 코드가 한 줄도 없다. 추정은
    :func:`deep_anc.data.timeline.measure_delay_trajectory` 만 한다 — 두 번째 유도가
    생기려면 그 함수를 복사해야 하고, 그건 눈에 띈다.

    import 방향에 대해: ``dsp`` 가 ``data`` 를 부르는 것은 층 순서상 역방향이라
    함수 안에서 지연 import 한다(모듈 최상단에서는 순환이다 — ``timeline`` 이
    :func:`check_stream_coherence` 를 쓴다). 층을 지키려고 **구현을 복제하는 것**이
    이 저장소에서 실제로 사고를 냈으므로, 층보다 단일 출처를 우선한다.
    """

    from ..data.timeline import measure_delay_trajectory as _measure

    return _measure(playback, capture, sample_rate=int(sample_rate), settings=settings)


def check_stream_delay_stability(
    playback: Sequence[float] | np.ndarray,
    capture: Sequence[float] | np.ndarray,
    *,
    sample_rate: int,
    max_robust_std_samples: float = MAX_STREAM_DELAY_ROBUST_STD_SAMPLES,
    max_p95_p5_samples: float = MAX_STREAM_DELAY_P95_P5_SAMPLES,
    min_valid_windows: int = MIN_STREAM_DELAY_VALID_WINDOWS,
    min_valid_window_ratio: float = MIN_STREAM_DELAY_VALID_WINDOW_RATIO,
    settings: Any | None = None,
    name: str = "stream_delay_stability",
) -> InvariantResult:
    """재생→캡처 지연이 세션 내내 **한 값**인가.

    코히런스와 짝을 이룬다. 코히런스는 관계의 유무를, 이 검사는 관계의 **시간적
    안정성**을 본다. 둘 다 봐야 하는 이유: 지연이 천천히 떠다니면 긴 창 코히런스는
    떨어지지만 짧은 창에서는 살아 있어, 코히런스 하나만으로는 "음향이 나쁘다"와
    "타임베이스가 떠다닌다"를 구분할 수 없다.

    판정량은 **로버스트 통계뿐**이다 — 임계 근거는
    :data:`MAX_STREAM_DELAY_ROBUST_STD_SAMPLES` /
    :data:`MAX_STREAM_DELAY_P95_P5_SAMPLES` 의 전수 측정에 적혀 있다. 원시 std 와 ptp
    는 진단으로 함께 남기지만 판정에 쓰지 않는다: 실측 정상 세션에서 유효창의 1~2%
    가 덕트 공진 로브를 한 칸 건너뛰면 raw std 가 1.6 → 17.4 로 부푸는데(세션 121132)
    시간축은 멀쩡하다. 이상치 몇 개로 좋은 데이터를 버리지 않으려면 판정량 자체가
    이상치에 둔감해야 한다.
    """

    trajectory = measure_stream_delay_trajectory(
        playback, capture, sample_rate=int(sample_rate), settings=settings
    )
    measured: dict[str, Any] = {
        **trajectory.as_metadata(),
        "max_robust_std_samples": float(max_robust_std_samples),
        "max_p95_p5_samples": float(max_p95_p5_samples),
        "min_valid_windows": int(min_valid_windows),
        "min_valid_window_ratio": float(min_valid_window_ratio),
    }
    reasons: list[str] = []
    if not trajectory.measurable(int(min_valid_windows)):
        # "판정 불가"는 통과가 아니다.
        reasons.append(
            f"지연을 잴 수 있는 유효창이 {trajectory.valid_windows}개로 최소 "
            f"{int(min_valid_windows)}개에 미달합니다 (전체 창 {trajectory.windows}) — "
            "무음이거나 지연이 탐색 범위를 벗어났습니다"
        )
    elif trajectory.valid_window_ratio < float(min_valid_window_ratio):
        # 살아남은 창끼리만 보면 로버스트 통계도 "안정" 으로 보인다. 추적을 놓친 것을
        # 안정으로 보고하지 않으려면 놓친 비율 자체를 판정에 넣어야 한다.
        reasons.append(
            f"지연을 추적할 수 있는 창이 {trajectory.valid_windows}/{trajectory.windows} "
            f"({trajectory.valid_window_ratio:.3f}) 로 하한 "
            f"{float(min_valid_window_ratio):.2f} 미만입니다 — 지연이 창별 탐색 범위를 "
            "벗어났거나 대역 내 내용이 없어 추적을 놓쳤습니다"
        )
    else:
        if trajectory.robust_std_samples > float(max_robust_std_samples):
            reasons.append(
                f"지연 robust-std {trajectory.robust_std_samples:.2f} > "
                f"{float(max_robust_std_samples):.2f} 샘플"
            )
        if trajectory.p95_p5_samples > float(max_p95_p5_samples):
            reasons.append(
                f"지연 p95−p5 {trajectory.p95_p5_samples:.2f} > "
                f"{float(max_p95_p5_samples):.2f} 샘플"
            )
    if reasons:
        return InvariantResult(
            name=name,
            ok=False,
            detail=(
                "재생→캡처 지연이 세션 안에서 떠다닙니다: "
                + "; ".join(reasons)
                + " — 서로 다른 클록 도메인을 인덱스로만 정렬한 결과입니다"
            ),
            measured=measured,
        )
    return InvariantResult(
        name=name,
        ok=True,
        detail=(
            f"재생→캡처 지연이 안정합니다: 중앙 "
            f"{trajectory.median_samples:.2f} 샘플, robust-std "
            f"{trajectory.robust_std_samples:.2f} ≤ {float(max_robust_std_samples):.2f}, "
            f"p95−p5 {trajectory.p95_p5_samples:.2f} ≤ {float(max_p95_p5_samples):.2f} "
            f"(유효창 {trajectory.valid_windows}/{trajectory.windows})"
        ),
        measured=measured,
    )


def check_measured_delay_agreement(
    observed_delay_samples: float,
    derived_delay_samples: float,
    *,
    tolerance_samples: float = 64.0,
    observation_count: int = 0,
    name: str = "measured_delay_agreement",
    observed_label: str = "실측 세션 source→ERR",
    derived_label: str = "P(z) 측정에서 유도",
) -> InvariantResult:
    """**같은 물리량을 두 방법으로 잰 값**이 일치하는가 (발생기 A 의 정면 대응).

    이 저장소에서 반복된 사고는 전부 같은 모양이다 — 같은 지연을 두 곳에서 따로
    유도하고 **아무도 대조하지 않았다**. 이 검사는 그 대조 자체를 코드로 만든다.

    2026-08-05 감사(D2)가 찾은 실제 불일치::

        관측: 실측 세션의 source→ERR 지연
              포락선 상관 80세션 중앙값 1672
              반송파 상관 40세션 1663 ± 73
              제어기 시작지연 스윕 직접파 ~1670
        유도: D_P(1602) + P_FIR 무게중심 tap(369) ≈ 1971

        차이 ≈ 250~280 샘플 (5~6 ms). 비용은 계열별 +0.71 ~ +2.39 dB.

    세 방법이 서로 일치하는데 유도값만 어긋난다 — 즉 관측이 옳고 부기가 틀렸다.

    허용 오차 64 샘플의 근거(임의의 숫자가 아니다)
    ------------------------------------------
    1. **관측의 불확도**: 세션 간 SD 가 73 샘플이므로 N 세션 중앙값의 표준오차는
       73/√N 이다. 게이트가 요구하는 최소 세션 수 8 에서 SE ≈ 26 샘플 → 64 는 2.5σ.
    2. **기하학적 의미**: 48kHz 에서 64 샘플 = 1.33 ms = 공기 중 0.46 m. 덕트의
       NS→ERR 거리가 1.10 m 이므로 허용 오차는 기하의 42% 다. 이보다 크면 마이크가
       어디 있는지 모르는 것과 같다.
    3. **검출력**: 실제 결함 250 샘플은 이 허용치의 3.9배이자 9.6σ 다. 놓칠 수 없다.

    ``observation_count`` 를 주면 표본이 부족할 때 **통과시키지 않고** 판정 불가로
    떨어뜨린다 — 표본 1개짜리 중앙값으로 부기를 승인하면 이 게이트도 자기증명이 된다.
    """

    observed = float(observed_delay_samples)
    derived = float(derived_delay_samples)
    tolerance = float(tolerance_samples)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"허용 오차는 유한한 0 이상이어야 합니다: {tolerance_samples!r}")
    mismatch = abs(observed - derived)
    measured: dict[str, Any] = {
        "observed_delay_samples": observed,
        "derived_delay_samples": derived,
        "mismatch_samples": mismatch,
        "tolerance_samples": tolerance,
        "observation_count": int(observation_count),
    }
    if not (math.isfinite(observed) and math.isfinite(derived)):
        return InvariantResult(
            name=name,
            ok=False,
            detail=(
                f"지연 교차검증에 유한한 값이 없습니다: {observed_label}={observed!r}, "
                f"{derived_label}={derived!r}"
            ),
            measured=measured,
        )
    if mismatch > tolerance:
        return InvariantResult(
            name=name,
            ok=False,
            detail=(
                f"같은 지연을 두 방법으로 잰 값이 다릅니다: {observed_label} "
                f"{observed:.0f} vs {derived_label} {derived:.0f} — 차이 "
                f"{mismatch:.0f} > 허용 {tolerance:.0f} 샘플 "
                f"({mismatch / 48.0:.2f} ms @48kHz). 합성 d 가 실측 d 와 다른 위치에 "
                "놓이며, 이 오차는 학습으로 흡수되지 않습니다"
            ),
            measured=measured,
        )
    return InvariantResult(
        name=name,
        ok=True,
        detail=(
            f"{observed_label} {observed:.0f} 와 {derived_label} {derived:.0f} 가 "
            f"{mismatch:.0f} ≤ {tolerance:.0f} 샘플로 일치합니다"
        ),
        measured=measured,
    )


def _clip_key(value: str) -> str:
    """코퍼스 클립의 비교 키 — 경로가 달라도 같은 원본이면 같은 키가 된다.

    합성 매니페스트는 ``data/raw/music/fma_small/033/033012.mp3`` 처럼 저장소 경로를
    들고, 실측 ``sources.csv`` 는 ``033012.mp3`` 처럼 basename 만 들고 있다. 둘을
    직접 비교하면 **영원히 교집합이 비어 보인다** — 누수 게이트가 있는데도 아무것도
    잡지 못하는 가장 흔한 실패 방식이다. 그래서 양쪽을 basename 으로 정규화한다.
    """

    text = str(value).strip().replace("\\", "/")
    return text.rsplit("/", 1)[-1].casefold()


def check_corpus_disjoint(
    recorded_clips_by_family: dict[str, Sequence[str]],
    synthetic_clips: Sequence[str] | dict[str, Sequence[str]],
    *,
    name: str = "corpus_disjoint",
    synthetic_splits: dict[str, str] | None = None,
) -> InvariantResult:
    """실측이 재생한 원본과 합성 학습 스트림의 원본이 **겹치지 않는가**.

    왜 이것이 불변식인가
    --------------------
    같은 오디오가 두 브랜치에 동시에 들어가면 모델은 **같은 입력에 상충하는 정답**을
    받는다. 합성 브랜치는 이상적 P/S 라 −18 dB 까지 상쇄 가능하고, 실측 브랜치는
    실제 플랜트라 천장이 훨씬 낮다. 같은 곡에서 반대 방향 gradient 가 온다.

    2026-08-05 감사(D1) 실측::

        계열          실측이 쓴 원본   합성 풀과 교집합
        music              60          **60 (100%)**   ← 55개는 합성 *train* 에 있다
        speech            218               0
        machine           188               0
        environment       225               0

    그리고 **music 만 개선되지 않았다** (+0.09 dB, 나머지는 −0.85 ~ −2.05 dB).
    누수를 검사하는 게이트가 없었다 — 전형적인 군집 B(반증된 적 없는 게이트) 결함이다.

    ``synthetic_splits`` 를 주면 겹친 클립이 합성의 어느 split 에 있었는지 함께
    보고한다. train 에 있으면 누수이고, test 에 있으면 **평가 오염**이라 더 나쁘다.
    """

    if isinstance(synthetic_clips, dict):
        synthetic_pairs = [
            (tag, item) for tag, values in synthetic_clips.items() for item in values
        ]
    else:
        synthetic_pairs = [("", item) for item in synthetic_clips]
    synthetic_index: dict[str, str] = {}
    for tag, item in synthetic_pairs:
        synthetic_index.setdefault(_clip_key(item), tag)

    splits = {_clip_key(key): str(value) for key, value in (synthetic_splits or {}).items()}

    families: dict[str, dict[str, Any]] = {}
    offenders: list[str] = []
    total_overlap = 0
    for family in sorted(recorded_clips_by_family):
        keys = {_clip_key(item) for item in recorded_clips_by_family[family]}
        overlap = sorted(keys.intersection(synthetic_index))
        total_overlap += len(overlap)
        ratio = (len(overlap) / len(keys)) if keys else 0.0
        split_counts: dict[str, int] = {}
        for key in overlap:
            split_counts[splits.get(key, "unknown")] = (
                split_counts.get(splits.get(key, "unknown"), 0) + 1
            )
        families[family] = {
            "recorded_clips": len(keys),
            "overlap_clips": len(overlap),
            "overlap_ratio": ratio,
            "overlap_by_synthetic_split": split_counts,
            "examples": overlap[:5],
        }
        if overlap:
            detail_split = (
                " ".join(f"{key}={value}" for key, value in sorted(split_counts.items()))
                or "split 미상"
            )
            offenders.append(
                f"{family} {len(overlap)}/{len(keys)} ({ratio * 100:.0f}%; {detail_split})"
            )

    measured: dict[str, Any] = {
        "families": families,
        "synthetic_clips": len(synthetic_index),
        "total_overlap_clips": total_overlap,
    }
    if offenders:
        return InvariantResult(
            name=name,
            ok=False,
            detail=(
                "실측과 합성이 같은 원본 오디오를 씁니다: "
                + ", ".join(offenders)
                + " — 같은 소리에 상충하는 정답이 주어져 gradient 가 서로를 지웁니다. "
                "겹치는 원본을 합성 풀에서 빼거나 실측을 다른 원본으로 다시 수집하세요"
            ),
            measured=measured,
        )
    return InvariantResult(
        name=name,
        ok=True,
        detail=(
            f"실측 {sum(item['recorded_clips'] for item in families.values())}개 원본과 "
            f"합성 {len(synthetic_index)}개 원본이 서로소입니다"
        ),
        measured=measured,
    )


def derive_playback_to_error_delay_samples(
    bulk_delay_samples: int, fir: Sequence[float] | np.ndarray
) -> float:
    """측정된 P(z) 로부터 **재생→ERR 마이크 지연**을 유도한다 (D2 의 유도 쪽 절반).

    관측 쪽은 실측 세션에서 source.wav 와 ERR 채널의 상호상관 최대점으로 잰다.
    따라서 유도 쪽도 **같은 추정기**를 써야 비교가 성립한다 — 상호상관 최대점은 임펄스
    응답의 최대점에 대응하므로 ``벌크지연 + argmax|h|`` 다.

    무게중심(에너지 1차 모멘트)을 쓰면 안 되는 이유: 잔향 꼬리가 무게중심을 뒤로
    끌어당긴다. 실측 P(z) 에서 argmax 는 tap 247 인데 무게중심은 363.7 로 **117 샘플**
    차이가 난다. 두 추정기를 섞으면 그 차이가 그대로 가짜 불일치로 보고된다 — 이
    저장소에서 반복된 "같은 양을 다르게 유도" 사고와 정확히 같은 모양이다.

    실측(2026-08-05, 캡처 225546_f7b0fecd)::

        P 벌크지연 1602 + argmax 247 = 1849  (유도)
        실측 세션 source→ERR                 ≈ 1663~1672  (관측, 독립 3방법 일치)
        차이 ≈ 180 샘플 (3.7 ms)

    이 불일치를 검사하는 게이트가 없었다.
    """

    taps = np.asarray(fir, dtype=np.float64).reshape(-1)
    if taps.size == 0 or not np.all(np.isfinite(taps)):
        raise ValueError("P(z) FIR 이 비었거나 유한하지 않습니다")
    if int(bulk_delay_samples) < 0:
        raise ValueError(f"벌크지연은 0 이상이어야 합니다: {bulk_delay_samples!r}")
    return float(int(bulk_delay_samples) + int(np.argmax(np.abs(taps))))
