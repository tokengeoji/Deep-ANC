"""재생↔녹음 시간축 부기의 **단일 출처**.

왜 이 모듈이 있는가
------------------
2026-08-05 결함 군집 분석의 발생기 A: "같은 물리량을 두 곳 이상에서 따로 유도하고
아무도 대조하지 않는다". 녹음 파이프라인에서 그 물리량은 **재생 시각과 캡처 시각의
대응**이었고, ``scripts/data/record_duct.py`` 는 그것을 유도하지 않고 **단언**했다::

    cursor = {"in": 0, "out": 0}      # 둘 다 0 에서 출발
    ...                                # 매 콜백 둘 다 frames 만큼 전진
    # ⇒ source[t] 와 mics[t] 가 같은 물리 시각이라는 주장 = 왕복지연 0 이라는 가정

실제 왕복지연은 상수도 아니다. AB13X USB DAC 의 재생 엔드포인트가 UAC1 **ADAPTIVE**
(full speed, 피드백 엔드포인트 없음)라 장치 내부 PLL 이 호스트 데이터율을 추종하며
헌팅한다 — 주기 4~5 초, 진폭 259~407 샘플(5.4~8.5 ms). 이 흔들림은 xrun/status 로
전혀 보고되지 않고(무음 40 초 프로브에서 status 0회), PortAudio 를 완전히 배제한
aplay/arecord 직결 경로에서도 같은 파형으로 재현된다.

그 결과 실측 80 세션은 ``coh²(source→ERR, 150-600Hz) = 0.021~0.126`` 인데
``coh²(REF→ERR) = 0.959~0.993`` 이다 — **음향은 멀쩡하고 인덱스 배정만 틀렸다.**

복구 원리
--------
REF 마이크는 ERR 마이크와 **같은 ADC**를 탄다. 따라서 REF 는 재생 신호가 실제로 언제
방출됐는지에 대한 **ADC 시간축 위의 증인**이다. ``source → REF`` 로 시변 지연 L(t) 를
추정해 ``source_aligned[t] = source(t − L(t))`` 로 되감으면 source 가 ADC 시간축으로
옮겨온다. 추정에 쓰지 않은 ERR 채널로 검증하면(홀드아웃) coh² 가 0.02~0.07 →
0.91~0.96 으로 회복된다.

규칙
----
**재생↔녹음 정렬 수치는 이 모듈에서만 유도된다.** ``record_duct.py`` 도,
``recorded_qa.py`` 도, 재정렬 스크립트도 전부 :func:`align_source_to_adc` /
:func:`estimate_lag_track` 를 호출하기만 한다. 여섯 번째 복붙이 나오면 그것이 다음 사고다.

핫패스 금지
----------
오디오 콜백 안에서 부르지 마라. 이 모듈은 캡처가 **끝난 뒤** 저장 시점, QA, 재정렬
스크립트에서만 돈다.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from ..dsp.invariants import check_stream_coherence
from ..dsp.timing import FrequencyBand

__all__ = [
    "TIMELINE_METHOD",
    "LagTrack",
    "TimelineReport",
    "TimelineSettings",
    "align_source_to_adc",
    "estimate_lag_track",
    "median_coherence",
    "warp_by_lag_track",
]


TIMELINE_METHOD = "ref_witness_warp_v1"
"""``session.json`` 의 ``timeline.method``. 워프 알고리즘을 바꾸면 이 문자열도 바꿔라 —
그래야 이미 만들어 둔 ``source_aligned.wav`` 를 재생성해야 하는지 판정할 수 있다."""


_FROZEN = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------------------
class TimelineSettings(BaseModel):
    """시간축 추정 파라미터. 값의 근거는 전부 실측이다.

    * ``track_band_hz`` 하한 150Hz — 그 아래는 스피커 저역 SNR 이 8~10 dB 라 위상이
      못 미덥다. 상한 700Hz — 덕트 1차 횡모드 아래를 유지해 로브 점프를 줄인다.
    * ``window_seconds`` 0.25 s — 헌팅 주기 4~5 초의 1/16 이라 창 안에서 지연이
      거의 상수다(국소 기울기 최대 ±165 샘플/초 × 0.25 s ≈ 41 샘플).
    * ``coarse_search_samples`` 600 — 관측된 지연 궤적 진폭(ptp 259~407)에 여유.
    * ``refine_search_samples`` 48 — 덕트 공진(약 144 샘플 주기) 로브로 건너뛰지 않을
      만큼 좁게. 이 재탐색이 없으면 창별 추정이 한 로브씩 튄다.
    """

    model_config = _FROZEN

    sample_rate: int
    track_band_hz: tuple[float, float] = (150.0, 700.0)
    window_seconds: float = 0.25
    hop_seconds: float = 0.0625
    coarse_search_samples: int = 600
    refine_search_samples: int = 48
    max_bulk_lag_samples: int = 4800
    verify_seconds: float = 30.0
    """검증 지표(coh², 잔여 지연 궤적)를 계산할 앞부분 길이. **워프 자체는 전 구간**에
    적용된다 — 여기서 줄이는 것은 판정 비용뿐이다.

    30 초인 이유: ``recorded_qa`` 의 ``alignment_max_seconds`` 와 같은 값이라 수집
    시점 판정과 QA 판정이 **같은 구간**을 본다. 다르게 두면 "수집은 통과했는데 QA 는
    떨어진다"가 되고, 그 불일치를 설명하려고 또 두 번째 부기가 생긴다.
    실측 비용: 70초 세션 기준 35 s → 18 s.
    """

    min_window_rms: float = 2.0e-4
    min_coarse_quality: float = 0.25
    min_refine_quality: float = 0.15
    """PHAT 피크 하한. 정규화 규약상 "대역 안 모든 위상이 정렬 = 1.0" 이므로 덕트
    잔향이 있는 실제 신호에서는 0.2~0.5 가 정상이다. 실측 스윕(세션 121917):
    0.35/0.25 → 유효창 0.870, 0.25/0.15 → 0.945, 0.15/0.05 → 0.993 이고 **지연 통계는
    셋 다 같다**(중앙 1539~1540, ptp 272~273). 즉 임계를 낮춰도 궤적이 오염되지 않는다 —
    레일 검출이 진짜 실패를 따로 잡기 때문이다. 0.25/0.15 를 쓴다: 게이트 0.90 에 여유를
    주면서, 내용이 없는 창을 유효로 세지는 않는다."""

    @model_validator(mode="after")
    def _validate(self) -> "TimelineSettings":
        if int(self.sample_rate) <= 0:
            raise ValueError(f"sample_rate 는 양수여야 합니다: {self.sample_rate}")
        band = FrequencyBand.parse(
            self.track_band_hz, name="track_band_hz", nyquist_hz=self.sample_rate / 2.0
        )
        del band  # 검증만 한다 (생성 시점 강제)
        if not 0.0 < self.hop_seconds <= self.window_seconds:
            raise ValueError(
                f"hop({self.hop_seconds})은 0 < hop <= window({self.window_seconds}) 여야 합니다"
            )
        for name in ("coarse_search_samples", "refine_search_samples", "max_bulk_lag_samples"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} 는 양수여야 합니다: {getattr(self, name)}")
        if self.refine_search_samples > self.coarse_search_samples:
            raise ValueError("refine 탐색 폭이 coarse 보다 넓으면 2단 추정의 의미가 없습니다")
        for name in ("min_window_rms", "min_coarse_quality", "min_refine_quality"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} 는 유한한 0 이상 값이어야 합니다: {value}")
        return self

    @property
    def window_samples(self) -> int:
        return max(256, int(round(self.window_seconds * self.sample_rate)))

    @property
    def hop_samples(self) -> int:
        return max(1, int(round(self.hop_seconds * self.sample_rate)))


# --------------------------------------------------------------------------------------
# 결과 타입
# --------------------------------------------------------------------------------------
class LagTrack(BaseModel):
    """창별 지연 궤적. numpy 배열을 그대로 들고 다니되 통계는 여기서만 계산한다.

    통계를 호출부에서 각자 계산하면 그것이 두 번째 유도다 — ``lag_std`` 를
    "유효창만" 으로 재는 곳과 "전체 창" 으로 재는 곳이 갈리면 게이트가 갈린다.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    times_s: Any
    lag_samples: Any
    quality: Any
    valid: Any
    sample_rate: int

    @property
    def valid_window_ratio(self) -> float:
        valid = np.asarray(self.valid, dtype=bool)
        if valid.size == 0:
            return 0.0
        return float(np.count_nonzero(valid) / valid.size)

    def _valid_lags(self) -> np.ndarray:
        lags = np.asarray(self.lag_samples, dtype=np.float64)
        valid = np.asarray(self.valid, dtype=bool)
        return lags[valid]

    @property
    def median_samples(self) -> float:
        lags = self._valid_lags()
        return float(np.median(lags)) if lags.size else float("nan")

    @property
    def std_samples(self) -> float:
        lags = self._valid_lags()
        return float(np.std(lags)) if lags.size >= 2 else float("nan")

    @property
    def ptp_samples(self) -> float:
        lags = self._valid_lags()
        return float(np.ptp(lags)) if lags.size else float("nan")

    @property
    def robust_std_samples(self) -> float:
        """1.4826 × MAD. **게이트는 이 값을 본다.**

        원시 std 를 게이트에 쓰면 안 되는 이유는 실측으로 확인됐다: 세션 121917 의
        재정렬 후 잔여 궤적은 창 1071개 중 **16개(1.5%)** 가 덕트 공진 로브 한 칸
        건너뛴 62~76 샘플이고, 이 16개가 std 를 2.2 → 7.8 로 부풀린다. p95−p5 는
        7.16 으로 멀쩡하다. 원시 std 는 진단용으로 계속 보고한다 — 숨기면 안 된다.
        """

        lags = self._valid_lags()
        if lags.size < 2:
            return float("nan")
        return float(1.4826 * np.median(np.abs(lags - np.median(lags))))

    @property
    def p95_p5_samples(self) -> float:
        lags = self._valid_lags()
        if lags.size < 2:
            return float("nan")
        return float(np.percentile(lags, 95.0) - np.percentile(lags, 5.0))

    def summary(self) -> dict[str, float]:
        return {
            "lag_median_samples": self.median_samples,
            "lag_std_samples": self.std_samples,
            "lag_robust_std_samples": self.robust_std_samples,
            "lag_ptp_samples": self.ptp_samples,
            "lag_p95_p5_samples": self.p95_p5_samples,
            "valid_window_ratio": self.valid_window_ratio,
            "windows": float(np.asarray(self.valid, dtype=bool).size),
        }


class TimelineReport(BaseModel):
    """세션 하나의 시간축 QA 결과. ``session.json['timeline']`` 에 그대로 들어간다.

    ``*_before`` 를 함께 남기는 이유: 재정렬이 실제로 무엇을 고쳤는지 숫자로 남지
    않으면, 다음 사람이 "원래 괜찮았던 것 아니냐" 를 반증할 수 없다.
    """

    model_config = _FROZEN

    method: str = TIMELINE_METHOD
    witness_channel: int = 1
    track_band_hz: tuple[float, float]
    track_window_s: float
    track_hop_s: float

    valid_window_ratio: float
    raw_lag_median_samples: float
    raw_lag_ptp_samples: float
    raw_lag_std_samples: float

    aligned_lag_median_samples: float
    aligned_lag_std_samples: float
    aligned_lag_robust_std_samples: float
    aligned_lag_p95_p5_samples: float
    aligned_valid_window_ratio: float

    coh2_150_600_before: float
    coh2_150_600_after: float
    coh2_600_1600_before: float
    coh2_600_1600_after: float
    coh2_ref_err_150_600: float

    @model_validator(mode="after")
    def _validate(self) -> "TimelineReport":
        if not self.method.strip():
            raise ValueError("timeline.method 가 비어 있습니다")
        if self.witness_channel < 0:
            raise ValueError(f"witness_channel 은 0 이상이어야 합니다: {self.witness_channel}")
        FrequencyBand.parse(self.track_band_hz, name="track_band_hz")
        for name in (
            "valid_window_ratio",
            "aligned_valid_window_ratio",
            "coh2_150_600_before",
            "coh2_150_600_after",
            "coh2_600_1600_before",
            "coh2_600_1600_after",
            "coh2_ref_err_150_600",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 는 0..1 이어야 합니다: {value}")
        return self

    def as_metadata(self) -> dict[str, Any]:
        return self.model_dump()


# --------------------------------------------------------------------------------------
# 내부 헬퍼
# --------------------------------------------------------------------------------------
def _next_pow2(n: int) -> int:
    return 1 << int(np.ceil(np.log2(max(2, n))))


def _band_mask(n_fft: int, sample_rate: int, band: tuple[float, float]) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
    return (freqs >= float(band[0])) & (freqs <= float(band[1]))


def _rfft(values: np.ndarray) -> np.ndarray:
    try:
        from scipy.fft import rfft as _scipy_rfft

        return _scipy_rfft(values, axis=-1, workers=-1)
    except ImportError:  # pragma: no cover - scipy 는 requirements 에 있다
        return np.fft.rfft(values, axis=-1)


def _irfft(values: np.ndarray, n: int) -> np.ndarray:
    try:
        from scipy.fft import irfft as _scipy_irfft

        return _scipy_irfft(values, n=n, axis=-1, workers=-1)
    except ImportError:  # pragma: no cover
        return np.fft.irfft(values, n=n, axis=-1)


def _gcc_phat_lags(
    capture_segs: np.ndarray,
    source_segs: np.ndarray,
    *,
    centre_lags: np.ndarray,
    search: int,
    sample_rate: int,
    band: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """창 여러 개의 ``capture(t) ≈ source(t − L)`` 을 한 번에 푼다.

    ``capture_segs`` (W, win), ``source_segs`` (W, win + 2·search) 를 받아
    ``(lag_samples, quality)`` 를 각각 (W,) 로 돌려준다. quality 는 대역 안 모든 위상이
    정렬됐을 때 1.0 이 되도록 정규화한 PHAT 피크라 대역폭·창길이에 무관하게 비교할 수 있다.

    FFT 길이는 ``next_pow2(win + 2·search)`` 로 충분하다. 순환 겹침은 지연 인덱스
    ``k ∈ [0, 2·search]`` 구간을 오염시키지 않는다 — 겹치는 항의 capture 인덱스가
    ``n − 2·search ≥ win`` 이라 항상 제로패딩 영역이기 때문이다. 이걸 모르고
    ``next_pow2(len(src)+len(cap))`` 를 쓰면 FFT 가 2배가 되고 그만큼 그냥 느려진다.
    """

    win = int(capture_segs.shape[-1])
    src_len = int(source_segs.shape[-1])
    if src_len != win + 2 * search:
        raise ValueError(f"source 창 길이가 맞지 않습니다: {src_len} != {win} + 2*{search}")
    n_fft = _next_pow2(src_len)
    mask = _band_mask(n_fft, sample_rate, band)
    active = int(np.count_nonzero(mask))
    if active == 0:
        raise ValueError(f"대역 {band} 에 FFT 빈이 없습니다 (n_fft={n_fft})")

    taper = np.hanning(win)
    count = int(capture_segs.shape[0])
    cap = np.zeros((count, n_fft), dtype=np.float64)
    cap[:, :win] = capture_segs * taper
    src = np.zeros((count, n_fft), dtype=np.float64)
    src[:, :src_len] = source_segs

    spectrum = _rfft(src) * np.conj(_rfft(cap))
    magnitude = np.abs(spectrum)
    np.divide(spectrum, magnitude, out=spectrum, where=magnitude > 1e-20)
    spectrum[:, ~mask] = 0.0
    corr = _irfft(spectrum, n_fft)[:, : 2 * search + 1]

    # src_seg[i] = source(t0 − centre − search + i), cap_seg[m] = capture(t0 + m)
    # corr[k] = Σ src[i]·cap[i−k] 이므로  L = centre + search − k,  k ∈ [0, 2·search]
    k = np.argmax(corr, axis=-1)
    rows = np.arange(count)
    peak = corr[rows, k]
    quality = np.clip(peak * n_fft / (2.0 * active), 0.0, 1.0)

    # 포물선 보간 (서브샘플). 탐색 끝단에 붙은 창은 보간하지 않는다.
    inner = (k > 0) & (k < corr.shape[-1] - 1)
    safe_k = np.clip(k, 1, corr.shape[-1] - 2)
    y0 = corr[rows, safe_k - 1]
    y1 = corr[rows, safe_k]
    y2 = corr[rows, safe_k + 1]
    denom = y0 - 2.0 * y1 + y2
    delta = np.where(np.abs(denom) > 1e-30, 0.5 * (y0 - y2) / np.where(denom == 0, 1.0, denom), 0.0)
    delta = np.clip(np.where(inner, delta, 0.0), -1.0, 1.0)
    lag = centre_lags.astype(np.float64) + float(search) - (k + delta)
    return lag, quality


def _bulk_lag(
    capture: np.ndarray,
    source: np.ndarray,
    *,
    sample_rate: int,
    band: tuple[float, float],
    max_lag: int,
) -> float:
    """세션 전체의 거친 벌크 지연. 창별 탐색의 중심을 잡는 데만 쓴다."""

    take = min(capture.size, source.size, 20 * sample_rate)
    cap = capture[:take].astype(np.float64)
    src = source[:take].astype(np.float64)
    n_fft = _next_pow2(take + max_lag + 1)
    cap_p = np.zeros(n_fft)
    cap_p[:take] = cap * np.hanning(take)
    src_p = np.zeros(n_fft)
    src_p[:take] = src
    spectrum = np.fft.rfft(src_p) * np.conj(np.fft.rfft(cap_p))
    magnitude = np.abs(spectrum)
    spectrum = np.where(magnitude > 1e-20, spectrum / np.maximum(magnitude, 1e-20), 0.0)
    spectrum[~_band_mask(n_fft, sample_rate, band)] = 0.0
    corr = np.fft.irfft(spectrum, n_fft)
    # capture(t) = source(t − L), L ≥ 0  ⇒  corr[n_fft − L] 이 최대
    tail = corr[n_fft - max_lag : n_fft]
    if tail.size == 0:
        return 0.0
    return float(max_lag - int(np.argmax(tail)))


def _medfilt1d(values: np.ndarray, size: int) -> np.ndarray:
    if size <= 1 or values.size == 0:
        return values.astype(np.float64, copy=True)
    half = size // 2
    padded = np.pad(values.astype(np.float64), half, mode="edge")
    strided = np.lib.stride_tricks.sliding_window_view(padded, size)
    return np.median(strided, axis=-1)


def _interpolate_missing(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    filled = values.astype(np.float64, copy=True)
    if not np.any(valid):
        return np.zeros_like(filled)
    index = np.arange(filled.size, dtype=np.float64)
    filled = np.interp(index, index[valid], filled[valid])
    return filled


# --------------------------------------------------------------------------------------
# 공개 API
# --------------------------------------------------------------------------------------
def median_coherence(
    reference: Sequence[float] | np.ndarray,
    capture: Sequence[float] | np.ndarray,
    *,
    sample_rate: int,
    band_hz: tuple[float, float],
    nperseg: int = 8192,
) -> float:
    """대역 중앙값 coh². **불변식 모듈의 구현을 그대로 쓴다** (재구현 금지).

    ``check_stream_coherence`` 는 임계 비교까지 하지만 우리는 값만 필요할 때가 있다.
    그래도 같은 코드가 계산해야 QA·게이트·리포트의 숫자가 갈라지지 않는다.
    """

    result = check_stream_coherence(
        reference,
        capture,
        sample_rate=int(sample_rate),
        band_hz=(float(band_hz[0]), float(band_hz[1])),
        min_coherence=0.0,
        nperseg=int(nperseg),
    )
    return float(result.measured["coherence"])


def estimate_lag_track(
    source: np.ndarray,
    capture: np.ndarray,
    settings: TimelineSettings,
    *,
    centre_lag: int | None = None,
) -> LagTrack:
    """``capture(t) ≈ source(t − L(t))`` 의 L(t) 를 2단으로 추정한다.

    1단계: 벌크 지연 중심에서 ``±coarse_search_samples`` 거친 탐색.
    2단계: 결측을 보간하고 ``medfilt(9)`` 로 기준선을 만든 뒤 ``±refine_search_samples``
    로 좁혀 재탐색. 좁히지 않으면 덕트 공진 로브(약 144 샘플)로 한 칸씩 튄다.
    """

    src = np.asarray(source, dtype=np.float64).reshape(-1)
    cap = np.asarray(capture, dtype=np.float64).reshape(-1)
    fs = int(settings.sample_rate)
    band = (float(settings.track_band_hz[0]), float(settings.track_band_hz[1]))
    win = settings.window_samples
    hop = settings.hop_samples

    if centre_lag is None:
        centre_lag = int(
            round(
                _bulk_lag(
                    cap,
                    src,
                    sample_rate=fs,
                    band=band,
                    max_lag=int(settings.max_bulk_lag_samples),
                )
            )
        )

    coarse = int(settings.coarse_search_samples)
    refine = int(settings.refine_search_samples)
    n = min(src.size, cap.size)
    # 창 시작점은 source 를 centre-coarse 만큼 앞에서 읽을 수 있어야 한다.
    first = max(0, centre_lag + coarse)
    starts = np.arange(first, n - win + 1, hop, dtype=np.int64)
    if starts.size == 0:
        raise ValueError(
            f"시간축 추정에 필요한 길이가 부족합니다: {n} 샘플 (창 {win}, 지연 {centre_lag})"
        )

    def _stage(centres: np.ndarray, search: int, min_quality: float):
        lags = np.full(starts.size, np.nan, dtype=np.float64)
        quality = np.zeros(starts.size, dtype=np.float64)
        span = win + 2 * search
        lows = starts - centres - search
        usable = (lows >= 0) & (lows + span <= src.size) & (starts + win <= cap.size)
        rms_local = np.zeros(starts.size, dtype=np.float64)
        # 청크 단위 배치 FFT. 창 하나씩 돌면 같은 크기 FFT 를 수천 번 새로 계획한다
        # (실측: 세션당 27초 → 배치 후 2초대).
        chunk = 128
        index = np.flatnonzero(usable)
        for begin in range(0, index.size, chunk):
            take = index[begin : begin + chunk]
            src_segs = np.stack([src[lo : lo + span] for lo in lows[take]])
            rms_local[take] = np.sqrt(np.mean(np.square(src_segs), axis=1))
            loud = rms_local[take] >= settings.min_window_rms
            if not np.any(loud):
                continue
            rows = take[loud]
            cap_segs = np.stack([cap[t0 : t0 + win] for t0 in starts[rows]])
            lag, q = _gcc_phat_lags(
                cap_segs,
                src_segs[loud],
                centre_lags=centres[rows].astype(np.float64),
                search=search,
                sample_rate=fs,
                band=band,
            )
            lags[rows] = lag
            quality[rows] = q
        valid = np.isfinite(lags) & (quality >= float(min_quality))
        return lags, quality, valid

    lags, quality, valid = _stage(
        np.full(starts.size, centre_lag, dtype=np.int64), coarse, settings.min_coarse_quality
    )
    if not np.any(valid):
        return LagTrack(
            times_s=(starts + win / 2.0) / fs,
            lag_samples=np.zeros(starts.size),
            quality=quality,
            valid=np.zeros(starts.size, dtype=bool),
            sample_rate=fs,
        )

    baseline = _medfilt1d(_interpolate_missing(lags, valid), 9)

    lags2, quality2, valid2 = _stage(
        np.rint(baseline).astype(np.int64), refine, settings.min_refine_quality
    )
    # 탐색 창 가장자리에 붙은 추정은 "찾았다"가 아니라 "못 찾았다"이다. 레일에 붙은
    # 값을 유효로 세면 valid_window_ratio 가 실패를 성공으로 보고한다.
    railed = np.abs(lags2 - baseline) >= float(refine) - 1.0
    valid2 = valid2 & ~railed
    smoothed = _medfilt1d(_interpolate_missing(lags2, valid2), 5)
    final = np.where(valid2, lags2, smoothed)
    return LagTrack(
        times_s=(starts + win / 2.0) / fs,
        lag_samples=final,
        quality=quality2,
        valid=valid2,
        sample_rate=fs,
    )


def warp_by_lag_track(
    source: np.ndarray, track: LagTrack, *, length: int | None = None
) -> np.ndarray:
    """``out[t] = source(t − L(t))`` — 4점 라그랑주 3차 분수지연으로 전 샘플 보간.

    L(t) 는 유효 창의 (시간, 지연) 점을 자연 3차 스플라인으로 보간해 얻는다.
    ``scipy`` 가 없어도 돌도록 실패 시 선형 보간으로 낮춘다(정밀도만 떨어진다).
    """

    src = np.asarray(source, dtype=np.float64).reshape(-1)
    n = int(length if length is not None else src.size)
    times = np.asarray(track.times_s, dtype=np.float64)
    lags = np.asarray(track.lag_samples, dtype=np.float64)
    valid = np.asarray(track.valid, dtype=bool)
    if not np.any(valid):
        raise ValueError("유효한 지연 추정 창이 없어 재정렬할 수 없습니다")

    t_samples = times[valid] * float(track.sample_rate)
    lag_valid = lags[valid]
    grid = np.arange(n, dtype=np.float64)
    if t_samples.size < 4:
        # 3점 이하로는 3차 스플라인이 성립하지 않는다. 정밀도가 떨어지는 것이지
        # 실패가 아니므로 선형으로 낮춘다 — 판정은 valid_window_ratio 가 한다.
        lag_dense = np.interp(grid, t_samples, lag_valid)
    else:
        try:
            from scipy.interpolate import CubicSpline

            spline = CubicSpline(t_samples, lag_valid, extrapolate=True)
            lag_dense = np.asarray(spline(grid), dtype=np.float64)
            # 스플라인 외삽은 양 끝에서 폭주할 수 있다 — 관측 범위로 클램프한다.
            lo, hi = float(np.min(lag_valid)), float(np.max(lag_valid))
            margin = max(4.0, 0.25 * (hi - lo))
            lag_dense = np.clip(lag_dense, lo - margin, hi + margin)
        except ImportError:  # pragma: no cover - scipy 는 requirements 에 있다
            lag_dense = np.interp(grid, t_samples, lag_valid)

    positions = grid - lag_dense
    base = np.floor(positions).astype(np.int64)
    frac = positions - base

    out = np.zeros(n, dtype=np.float64)
    idx = np.stack([base - 1, base, base + 1, base + 2], axis=0)
    ok = (idx[0] >= 0) & (idx[3] < src.size)
    idx = np.clip(idx, 0, src.size - 1)
    f = frac
    # 4점 라그랑주 3차 계수
    c0 = -f * (f - 1.0) * (f - 2.0) / 6.0
    c1 = (f + 1.0) * (f - 1.0) * (f - 2.0) / 2.0
    c2 = -(f + 1.0) * f * (f - 2.0) / 2.0
    c3 = (f + 1.0) * f * (f - 1.0) / 6.0
    out = c0 * src[idx[0]] + c1 * src[idx[1]] + c2 * src[idx[2]] + c3 * src[idx[3]]
    out[~ok] = 0.0
    return out.astype(np.float32)


def align_source_to_adc(
    source: np.ndarray,
    witness: np.ndarray,
    holdout: np.ndarray,
    sample_rate: int,
    *,
    settings: TimelineSettings | None = None,
) -> tuple[np.ndarray, TimelineReport]:
    """재생 소스를 ADC 시간축으로 되감는다.

    Parameters
    ----------
    source
        재생한 디지털 소스 (``source.wav``).
    witness
        시간축 증인 = REF 마이크(ch1). **추정에만** 쓴다.
    holdout
        검증용 = ERR 마이크(ch0). **추정에 쓰지 않는다.** 이 홀드아웃이 없으면
        "추정에 쓴 채널로 추정 결과를 검증" 하는 자기증명이 되고, 그건 게이트가 아니다.
    """

    fs = int(sample_rate)
    cfg = settings or TimelineSettings(sample_rate=fs)
    if int(cfg.sample_rate) != fs:
        raise ValueError(f"TimelineSettings.sample_rate({cfg.sample_rate}) != {fs}")

    src = np.asarray(source, dtype=np.float64).reshape(-1)
    ref = np.asarray(witness, dtype=np.float64).reshape(-1)
    err = np.asarray(holdout, dtype=np.float64).reshape(-1)
    n = min(src.size, ref.size, err.size)
    src, ref, err = src[:n], ref[:n], err[:n]

    track = estimate_lag_track(src, ref, cfg)
    if np.count_nonzero(np.asarray(track.valid, dtype=bool)) < 2:
        # 추정이 통째로 실패했다. 예외로 터뜨리지 않고 **측정값이 실패를 말하게** 둔다 —
        # 호출부(record_duct / 재정렬 스크립트 / QA)가 전부 같은 방식으로 판정할 수
        # 있어야 하고, "검사했는데 통과 못 했다" 와 "검사하지 않았다" 가 리포트에서
        # 구분돼야 한다. 워프하지 않은 원본이 그대로 검증에 들어간다.
        aligned = src.astype(np.float32)
    else:
        aligned = warp_by_lag_track(src, track, length=n)

    # 검증은 전부 홀드아웃(ERR)으로 한다. 구간은 recorded_qa 와 같은 앞 30 초다.
    take = min(n, max(4 * cfg.window_samples, int(round(cfg.verify_seconds * fs))))
    raw_v = src[:take]                                        # 재정렬 전 소스
    aligned_v = np.asarray(aligned[:take], dtype=np.float64)  # 재정렬 후 소스
    err_v = err[:take]                                        # 홀드아웃
    witness_v = ref[:take]                                    # 음향 대조군
    before = median_coherence(raw_v, err_v, sample_rate=fs, band_hz=(150.0, 600.0))
    after = median_coherence(aligned_v, err_v, sample_rate=fs, band_hz=(150.0, 600.0))
    before_hi = median_coherence(raw_v, err_v, sample_rate=fs, band_hz=(600.0, 1600.0))
    after_hi = median_coherence(aligned_v, err_v, sample_rate=fs, band_hz=(600.0, 1600.0))
    ref_err = median_coherence(witness_v, err_v, sample_rate=fs, band_hz=(150.0, 600.0))

    def _finite(value: float) -> float:
        return float(value) if math.isfinite(float(value)) else 0.0

    residual = estimate_lag_track(aligned_v, err_v, cfg)

    report = TimelineReport(
        method=TIMELINE_METHOD,
        witness_channel=1,
        track_band_hz=(float(cfg.track_band_hz[0]), float(cfg.track_band_hz[1])),
        track_window_s=float(cfg.window_seconds),
        track_hop_s=float(cfg.hop_seconds),
        valid_window_ratio=_finite(track.valid_window_ratio),
        raw_lag_median_samples=track.median_samples,
        raw_lag_ptp_samples=track.ptp_samples,
        raw_lag_std_samples=track.std_samples,
        aligned_lag_median_samples=residual.median_samples,
        aligned_lag_std_samples=residual.std_samples,
        aligned_lag_robust_std_samples=residual.robust_std_samples,
        aligned_lag_p95_p5_samples=residual.p95_p5_samples,
        aligned_valid_window_ratio=_finite(residual.valid_window_ratio),
        coh2_150_600_before=_finite(before),
        coh2_150_600_after=_finite(after),
        coh2_600_1600_before=_finite(before_hi),
        coh2_600_1600_after=_finite(after_hi),
        coh2_ref_err_150_600=_finite(ref_err),
    )
    return aligned, report
