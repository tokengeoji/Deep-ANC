"""DAC↔ADC 클록이 비동기인 환경에서 P(z)/S(z) 를 **동시에** 식별하기 위한 신호 설계.

문제
----
재생은 USB(AB13X), 녹음은 Tegra APE I²S 다. 서로 다른 클록 도메인이라 "출력 샘플 번호 ↔
녹음 샘플 번호" 대응이 시간에 따라 흔들린다(wander). 저장된 측정 4건을 오프라인 재분석한
결과가 이를 못박는다.

* 재생 프로그램을 기준으로 본 반복 간 coherence: **0.08 ~ 0.17**
* 같은 녹음을 ERR/REF(둘 다 같은 ADC 클록) 기준으로 보면: **0.9915 ~ 0.9976**
* 크기 |H| 는 반복 간 std **0.08 dB** 로 완벽히 재현되고, 흔들리는 것은 **위상뿐**이다.

즉 덕트도 마이크도 레벨도 정상이고, 깨진 것은 시간축 대응 하나다. 자극 진폭을 4배 올려도
coherence 가 개선되지 않았다는 사실이 레벨 가설을 직접 반증한다.

해법
----
warp 는 위상 계수 ``exp(-j w D(t))`` 로 나타난다. **두 출력 채널은 같은 DAC·같은 스트림**을
지나므로 D(t) 가 동일하다. 따라서 두 경로를 *정확히 같은 시각에* 측정하면 D 가 아무리
빨리 흔들려도 두 경로에 공통으로 실리고, 우리가 실제로 필요로 하는 **S 와 P 의 상대 관계**
에서는 상쇄된다.

동시에 재생하면서 두 경로를 분리하려면 주파수를 나눈다.

    ch0(소음 스피커)  → 짝수 인덱스 톤    f = f0 + 2k*df
    ch1(상쇄 스피커)  → 홀수 인덱스 톤    f = f0 + (2k+1)*df

송신할 DAC 명령에서는 빈 집합이 서로소다. 그러나 비동기 USB DAC와 Tegra ADC 사이의
sample-rate 오차는 ADC 격자에서 톤을 fractional bin으로 옮긴다. 그러면 guard=1 정수 FFT는
바로 옆 채널 톤을 누설시키므로 **측정 분석에 그대로 쓰면 안 된다**. 측정 도구는 원시
시간영역에서 반복별 실제 주기 비율 ``q``를 관측하고, 두 채널의 모든 톤을 공동 real LS로
분리하며, 독립 cubic 재표본화 교차검증까지 통과시킨다. 이 모듈의 정수 빈은 송신 probe와
compact-FIR 격자를 정의할 뿐 비동기 녹음의 분리 증거가 아니다.

절대 지연은 이 방법으로 얻지 못한다(그것이 바로 흔들리는 양이다). 대신 이미 검증된
기하로 고정한다 — TDOA 실측 ERR−REF 가 noise 146 / cancel −93 샘플이고 기하 예측이
140 / −126 으로 일치한다. ANC 에 필요한 것은 절대 지연이 아니라 S 와 P 의 상대 지연이며,
실기 ANC 가 +6dB 로 동작하는 이유도 소음과 상쇄음이 같은 스트림을 지나 warp 가
``e = d + S*y`` 에서 상쇄되기 때문이다.

크레스트 팩터
------------
멀티톤은 위상을 잘못 주면 크레스트가 20dB 를 넘어 같은 피크에서 음향 에너지를 잃는다.
Schroeder 위상은 선형 스윕과 비슷한 낮은 크레스트를 준다. 여기에 소수의 반복 최적화를
더해 실측 크레스트를 리포트에 남긴다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "InterleavedProbe",
    "build_interleaved_probe",
    "align_repeats",
    "channel_impulse_response",
    "complex_consistency",
    "crest_factor_db",
    "dewarp_recording",
    "estimate_repeat_delay",
    "estimate_transfer",
    "relative_tau_outliers",
    "schroeder_phases",
    "timebase_drift",
    "tone_snr_db",
    "track_warp",
]


def crest_factor_db(signal: np.ndarray) -> float:
    values = np.asarray(signal, dtype=np.float64)
    rms = float(np.sqrt(np.mean(values**2)))
    peak = float(np.max(np.abs(values)))
    if rms <= 0.0 or peak <= 0.0:
        return float("inf")
    return 20.0 * np.log10(peak / rms)


def schroeder_phases(count: int) -> np.ndarray:
    """Schroeder(1970) 위상 — 평탄 스펙트럼 멀티톤의 크레스트를 최소화한다."""

    index = np.arange(count, dtype=np.float64)
    return -np.pi * index * (index + 1.0) / max(1, count)


@dataclass(frozen=True)
class InterleavedProbe:
    """동시 재생용 2채널 자극과 그 분석에 필요한 인덱스."""

    sample_rate: int
    period_samples: int
    """한 주기 길이. FFT 길이와 같아야 빈이 정확히 맞아떨어진다."""

    noise_signal: np.ndarray
    cancel_signal: np.ndarray
    noise_bins: np.ndarray
    cancel_bins: np.ndarray
    frequencies_hz: np.ndarray
    """모든 자극 톤의 주파수 (noise_bins ∪ cancel_bins, 정렬)."""

    def crest_db(self) -> tuple[float, float]:
        return crest_factor_db(self.noise_signal), crest_factor_db(self.cancel_signal)

    def bins_for(self, drive: str) -> np.ndarray:
        if drive == "noise":
            return self.noise_bins
        if drive == "cancel":
            return self.cancel_bins
        raise ValueError(f"drive 는 'noise' 또는 'cancel' 이어야 합니다: {drive!r}")

    def bin_step(self, drive: str) -> int:
        """한 채널 안에서 인접 톤 사이의 빈 간격. IR 복원 스케일의 분모다."""

        selected = self.bins_for(drive)
        if selected.size < 2:
            raise ValueError("톤이 2개 미만이면 간격을 정의할 수 없습니다")
        steps = np.unique(np.diff(selected))
        if steps.size != 1:
            raise ValueError(f"톤 간격이 균일하지 않습니다: {steps}")
        return int(steps[0])

    def guard_bins(self) -> int:
        """서로 다른 채널의 가장 가까운 두 톤 사이 빈 거리."""

        return int(np.min(np.abs(self.noise_bins[:, None] - self.cancel_bins[None, :])))


def build_interleaved_probe(
    *,
    sample_rate: int,
    period_seconds: float,
    band_hz: tuple[float, float],
    amplitude: float,
    tone_spacing_hz: float | None = None,
) -> InterleavedProbe:
    """서로소 빈 집합을 쓰는 2채널 동시 자극을 만든다.

    ``period_samples`` 는 DAC 명령에서 정확히 정수 주기가 되도록 잡는다. 따라서 송신
    배열과 해당 스펙트럼은 정확히 재구성할 수 있다. 비동기 ADC 녹음에서는 이 사실만으로
    채널 분리가 보장되지 않으며, 측정 도구의 q 관측+joint LS가 별도로 필요하다.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate 는 양수여야 합니다")
    if not 0.0 < amplitude <= 1.0:
        raise ValueError("amplitude 는 (0, 1] 이어야 합니다")
    low, high = float(band_hz[0]), float(band_hz[1])
    if not 0.0 < low < high < sample_rate / 2.0:
        raise ValueError(f"band_hz 가 유효하지 않습니다: {band_hz}")

    period = int(round(period_seconds * sample_rate))
    if period <= 0:
        raise ValueError("period_seconds 가 너무 짧습니다")
    resolution = sample_rate / period
    spacing = float(tone_spacing_hz) if tone_spacing_hz else resolution
    step = max(1, int(round(spacing / resolution)))
    # 두 채널을 번갈아 배치하려면 step 이 짝수여야 각 채널의 간격이 균일해진다.
    if step % 2:
        step += 1

    first = int(np.ceil(low / resolution))
    last = int(np.floor(high / resolution))
    bins = np.arange(first, last + 1, step // 2, dtype=int)
    if bins.size < 8:
        raise ValueError(
            f"자극 톤이 너무 적습니다({bins.size}개) — period_seconds 를 늘리거나 "
            "tone_spacing_hz 를 줄이세요"
        )
    noise_bins = bins[0::2]
    cancel_bins = bins[1::2]

    def synthesize(selected: np.ndarray) -> np.ndarray:
        spectrum = np.zeros(period // 2 + 1, dtype=np.complex128)
        phases = schroeder_phases(selected.size)
        spectrum[selected] = np.exp(1j * phases)
        signal = np.fft.irfft(spectrum, period)
        peak = float(np.max(np.abs(signal)))
        return (signal / peak * amplitude).astype(np.float32)

    return InterleavedProbe(
        sample_rate=int(sample_rate),
        period_samples=period,
        noise_signal=synthesize(noise_bins),
        cancel_signal=synthesize(cancel_bins),
        noise_bins=noise_bins,
        cancel_bins=cancel_bins,
        frequencies_hz=bins * resolution,
    )


def estimate_transfer(
    recording: np.ndarray,
    probe: InterleavedProbe,
    *,
    drive: str,
) -> tuple[np.ndarray, np.ndarray]:
    """한 주기 녹음에서 지정한 채널의 전달함수를 추정한다.

    반환: (주파수 Hz, 복소 전달함수). 해당 채널의 빈에서만 값이 정의된다 —
    다른 채널의 빈은 애초에 그 채널이 구동하지 않았으므로 물리적으로 의미가 없다.
    """

    if drive == "noise":
        selected, reference = probe.noise_bins, probe.noise_signal
    elif drive == "cancel":
        selected, reference = probe.cancel_bins, probe.cancel_signal
    else:
        raise ValueError(f"drive 는 'noise' 또는 'cancel' 이어야 합니다: {drive!r}")

    segment = np.asarray(recording, dtype=np.float64)
    if segment.size != probe.period_samples:
        raise ValueError(
            f"녹음 구간 길이가 주기와 다릅니다: {segment.size} != {probe.period_samples}"
        )
    observed = np.fft.rfft(segment)
    driven = np.fft.rfft(np.asarray(reference, dtype=np.float64))
    denominator = driven[selected]
    # 구동이 0 인 빈은 나눗셈이 정의되지 않는다. 설계상 일어나지 않지만 조용히
    # inf 를 만들지 않도록 명시적으로 막는다.
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError("구동 스펙트럼에 0 인 빈이 있습니다 — 자극 설계를 확인하세요")
    resolution = probe.sample_rate / probe.period_samples
    return selected * resolution, observed[selected] / denominator


def channel_impulse_response(
    probe: InterleavedProbe,
    transfer: np.ndarray,
    *,
    drive: str,
    pre_roll: int = 0,
) -> np.ndarray:
    """한 채널의 빈-희소 전달함수를 시간영역 IR 로 되돌린다.

    이 채널은 ``bin_step`` 마다 하나씩만 빈을 갖는다. 나머지를 0 으로 두고 역변환하면
    결과는 길이 ``period_samples / bin_step`` 인 alias 조각들로 나뉘고 진폭은
    ``1/bin_step`` 로 줄어든다(빗살 곱셈의 시간영역 쌍대). 따라서 ``bin_step`` 을 곱해
    원래 스케일로 되돌리고, 첫 조각만 잘라 쓴다. 덕트 IR(약 50ms)이 이 길이(기본
    250ms)보다 훨씬 짧아야 복제본이 겹치지 않는다 — 겹치면 IR 이 자기 자신과 더해져
    조용히 틀린다.

    ``pre_roll`` 만큼 전체 IFFT 를 먼저 순환 이동시켜 대역제한 IR 의 **선행 링잉을
    온셋 앞쪽에 남긴 뒤** 첫 조각을 자른다. 이동 전에 첫 조각을 잘라 그 조각만
    ``np.roll`` 하면 안 된다. 기본 interleave 에서 noise 는 even bin 이라 조각이
    periodic 이지만 cancel 은 odd bin 이라 반주기마다 부호가 바뀌는 anti-periodic
    조각이다. 먼저 자르면 cancel 의 앞쪽으로 감긴 pre-roll 부호가 뒤집히지 않아 측정한
    복소 전달함수와 다른 FIR 이 만들어진다.
    """

    selected = probe.bins_for(drive)
    step = probe.bin_step(drive)
    values = np.asarray(transfer, dtype=np.complex128).reshape(-1)
    if values.size != selected.size:
        raise ValueError(
            f"전달함수 길이가 톤 개수와 다릅니다: {values.size} != {selected.size}"
        )
    if pre_roll < 0:
        raise ValueError("pre_roll 은 음수일 수 없습니다")

    spectrum = np.zeros(probe.period_samples // 2 + 1, dtype=np.complex128)
    spectrum[selected] = values
    full = np.fft.irfft(spectrum, probe.period_samples) * float(step)
    period = probe.period_samples // step
    if pre_roll >= period:
        raise ValueError(f"pre_roll 이 복원 주기({period})보다 큽니다")
    # 전체 N-point 주기열에서 이동한 뒤 잘라야 odd-bin 채널의 anti-periodic 부호까지
    # 보존된다. ``np.roll(full[:period], pre_roll)`` 은 even-bin 채널에서만 동치다.
    return np.roll(full, int(pre_roll))[:period]


def tone_snr_db(
    signal_spectrum: np.ndarray,
    noise_spectrum: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    """구동 톤 빈에서의 신호 대 배경잡음 비(dB).

    ``noise_spectrum`` 은 **출력 장치를 열지 않은** preflight 를 같은 길이·같은 FFT 로
    변환한 것이어야 한다. 그래야 분모가 실제 환경 잡음이 된다.
    """

    sig = np.abs(np.asarray(signal_spectrum).reshape(-1)[bins])
    noise = np.abs(np.asarray(noise_spectrum).reshape(-1)[bins])
    return 20.0 * np.log10((sig + 1e-30) / (noise + 1e-30))


# ---------------------------------------------------------------------------
# warp 추적 — 실측이 강제한 추가 단계
#
# 이 모듈의 원래 설계는 "두 경로에 공통으로 실리는 warp 는 상대 관계에서 상쇄된다"
# 였고 그 주장 자체는 맞다. 그러나 2026-08-04 실측이 warp 의 **크기**를 처음으로
# 정확히 재어 놓았고, 값이 설계 가정보다 훨씬 컸다.
#
#   창 1초 안에서 재생↔녹음 대응이 100~200 샘플 움직인다.
#   반복 간 |H| 비는 1.000 (크기는 완벽히 재현), 위상은 순수 시간이동조차 아니다
#   (직선 적합 잔차 1.8~3.9 rad).
#
# 결과적으로 정수주기 FFT 가정이 깨진다 — guard=1 이면 한 채널의 톤이 이웃 채널
# 빈으로 샌다. 아래의 오래된 비선형 warp 보정은 진단 전용이다. official 측정은 raw
# adjacent-cycle q witness와 fractional-frequency joint LS를 사용하고, cubic affine
# 재표본화 결과와 독립적으로 일치해야 한다.
#
# 아래 두 함수가 그 단계다. 실측 검증치(같은 캡처, 8회 반복):
#
#   보정 없음                반복 일관성 0.05
#   창 43ms · 평활 없음      반복 일관성 0.84 / 0.85   ← 최선
#   창 43ms · 평활 25점      반복 일관성 0.54 / 0.55
#
# 평활이 **악화**시킨다는 사실이 중요하다. ±150 샘플의 요동이 추적 잡음이 아니라
# 실제 타임베이스 거동이라는 뜻이다. 그리고 0.85 는 파인튜닝 게이트의 0.90 에 아직
# 못 미친다 — 이 함수들은 문제를 줄이지만 없애지는 못한다.
# ---------------------------------------------------------------------------

DEFAULT_TRACK_WINDOW = 2048     # 43ms. 실측 스윕에서 가장 높은 일관성을 준 값
DEFAULT_TRACK_SEARCH = 4096


def track_warp(
    reference: np.ndarray,
    observed: np.ndarray,
    *,
    window: int = DEFAULT_TRACK_WINDOW,
    hop: int | None = None,
    search: int = DEFAULT_TRACK_SEARCH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """짧은 창마다 ``observed`` 가 ``reference`` 보다 얼마나 늦는지 추적한다.

    반환 ``(centres, delays, peaks)`` — 창 중심 인덱스, 지연(샘플, 포물선 보간으로
    소수점까지), 정규화 상관 첨두값. ``peaks`` 는 진단용이다: 값이 낮은 구간은
    추적을 믿을 수 없다는 뜻이므로 조용히 넘어가지 말고 리포트에 남긴다.

    창을 짧게 할수록 warp 를 잘 따라가지만 상관 첨두가 넓어져 위치가 흐려진다.
    실측에서 2048 샘플이 최적이었다.
    """

    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    obs = np.asarray(observed, dtype=np.float64).reshape(-1)
    if window <= 0 or search <= 0:
        raise ValueError("window 와 search 는 양수여야 합니다")
    step = int(hop) if hop else max(1, window // 16)
    limit = min(ref.size, obs.size) - window - search
    if limit <= 0:
        raise ValueError("신호가 창+탐색범위보다 짧습니다")

    centres: list[float] = []
    delays: list[float] = []
    peaks: list[float] = []
    for start in range(0, limit, step):
        a = obs[start : start + window + search]
        b = ref[start : start + window]
        a = a - a.mean()
        b = b - b.mean()
        corr = np.correlate(a, b, mode="valid")
        index = int(np.argmax(corr))
        if 0 < index < corr.size - 1:
            y0, y1, y2 = corr[index - 1], corr[index], corr[index + 1]
            denominator = y0 - 2.0 * y1 + y2
            fraction = 0.5 * (y0 - y2) / denominator if denominator != 0.0 else 0.0
        else:
            fraction = 0.0
        # 정규화는 **실제로 맞춰진 구간**으로 해야 값이 상관계수처럼 읽힌다.
        norm = float(np.linalg.norm(a[index : index + window]) * np.linalg.norm(b))
        centres.append(start + window / 2.0)
        delays.append(float(index) + float(fraction))
        peaks.append(float(corr[index] / norm) if norm > 0.0 else 0.0)
    return np.asarray(centres), np.asarray(delays), np.asarray(peaks)


def dewarp_recording(
    observed: np.ndarray,
    centres: np.ndarray,
    delays: np.ndarray,
    peaks: np.ndarray | None = None,
    *,
    min_peak: float = 0.0,
) -> np.ndarray:
    """추적한 궤적으로 녹음을 재생 타임베이스에 되돌린다.

    ``corrected[n] = observed(n + D(n))`` — 재생 시각 n 에 나간 소리가 녹음에서는
    ``n + D(n)`` 에 있으므로, 그 자리에서 값을 끌어와 재생 격자에 다시 얹는다.
    D(n) 은 살아남은 창 중심 사이를 선형 보간한다. **평활하지 않는다** — 실측에서
    평활이 일관성을 떨어뜨렸고, 그것이 요동이 실재한다는 증거다.

    ``min_peak`` 은 상관이 약한 창을 버린다. 신호가 아직 도달하지 않은 앞머리나
    잘려나간 꼬리에서는 추적이 엉뚱한 값을 내는데, 그런 점 하나가 선형 보간을 통해
    **주변 수천 샘플을 함께 망가뜨린다**. 버린 구간은 이웃 사이를 잇는다.
    """

    signal = np.asarray(observed, dtype=np.float64).reshape(-1)
    centre_values = np.asarray(centres, dtype=np.float64).reshape(-1)
    delay_values = np.asarray(delays, dtype=np.float64).reshape(-1)
    if peaks is not None and min_peak > 0.0:
        keep = np.asarray(peaks, dtype=np.float64).reshape(-1) >= float(min_peak)
        if keep.sum() < 2:
            raise ValueError(
                f"min_peak={min_peak} 를 넘는 추적점이 {int(keep.sum())}개뿐입니다"
            )
        centre_values, delay_values = centre_values[keep], delay_values[keep]
    index = np.arange(signal.size, dtype=np.float64)
    trajectory = np.interp(index, centre_values, delay_values)
    return np.interp(index + trajectory, index, signal, left=0.0, right=0.0)


# ---------------------------------------------------------------------------
# 반복 정렬 — 시간영역 온셋 대신 전달함수의 위상 기울기를 쓴다
#
# 대역제한 IR 은 선행 링잉이 길어 에너지 온셋 검출이 흔들린다. 그 흔들림이 그대로
# "반복 지연 지터"로 보고되면, 실제로는 안정적인 측정이 게이트에서 떨어진다.
#
# 전달함수 위에서 재면 이 문제가 없다. 반복 k 의 관측을
#     H_k(f) = H(f) · exp(-j 2π f τ_k / fs)
# 로 두고 τ_k 를 |Σ_f H_k(f)* H_ref(f) e^{j2πfτ/fs}| 최대화로 찾는다. 합을 **재현되는
# 대역에서만** 취하는 것이 중요하다 — 재현되지 않는 대역을 넣으면 그 잡음이 τ 추정을
# 끌고 가 정렬이 오히려 나빠진다(실측 확인).
# ---------------------------------------------------------------------------


def estimate_repeat_delay(
    frequencies_hz: np.ndarray,
    transfer: np.ndarray,
    reference: np.ndarray,
    *,
    sample_rate: int,
    span_samples: float = 512.0,
    resolution_samples: float = 0.25,
    fit_band_hz: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """``transfer`` 가 ``reference`` 보다 **얼마나 늦는지** τ(샘플)와 정렬 후 상관을 준다.

    부호 규약: τ > 0 이면 이 반복이 기준보다 늦다. 정렬은 ``H·exp(+j2πfτ/fs)`` 로 되돌린다.
    "보정량"이 아니라 "관측된 지연"으로 두어야 두 채널의 τ 차이가 그대로 상대 지연이 된다.
    """

    freq = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    observed = np.asarray(transfer, dtype=np.complex128).reshape(-1)
    anchor = np.asarray(reference, dtype=np.complex128).reshape(-1)
    if not (freq.size == observed.size == anchor.size):
        raise ValueError("주파수·관측·기준의 길이가 같아야 합니다")
    if span_samples <= 0 or resolution_samples <= 0:
        raise ValueError("span 과 resolution 은 양수여야 합니다")

    mask = (
        np.ones(freq.size, dtype=bool)
        if fit_band_hz is None
        else (freq >= float(fit_band_hz[0])) & (freq <= float(fit_band_hz[1]))
    )
    if int(mask.sum()) < 8:
        raise ValueError(f"적합 대역 안의 톤이 너무 적습니다: {int(mask.sum())}개")

    cross = observed[mask].conj() * anchor[mask]
    taus = np.arange(-span_samples, span_samples + resolution_samples, resolution_samples)
    scores = np.abs(
        cross @ np.exp(-2j * np.pi * np.outer(taus, freq[mask]) / float(sample_rate)).T
    )
    index = int(np.argmax(scores))
    if 0 < index < scores.size - 1:
        y0, y1, y2 = scores[index - 1], scores[index], scores[index + 1]
        denominator = y0 - 2.0 * y1 + y2
        fraction = 0.5 * (y0 - y2) / denominator if denominator != 0.0 else 0.0
    else:
        fraction = 0.0
    norm = float(np.linalg.norm(observed[mask]) * np.linalg.norm(anchor[mask]))
    return (
        float(taus[index] + fraction * resolution_samples),
        float(scores[index] / norm) if norm > 0.0 else 0.0,
    )


def complex_consistency(stack: np.ndarray) -> float:
    """반복 간 복소 전달함수의 평균 쌍별 정규화 내적 (0~1)."""

    values = np.asarray(stack, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(f"[repeats, bins] 이고 repeats>=2 여야 합니다: {values.shape}")
    scores = []
    for i in range(values.shape[0]):
        for j in range(i + 1, values.shape[0]):
            a, b = values[i], values[j]
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            scores.append(abs(complex(np.vdot(a, b))) / denominator if denominator else 0.0)
    return float(np.mean(scores))


def timebase_drift(taus: np.ndarray) -> tuple[np.ndarray, float]:
    """반복별 국소 타임베이스 드리프트(샘플/주기)와 그 중앙값.

    재생 클록과 녹음 클록의 상대 오프셋은 주기당 일정한 기울기로 나타난다
    (실측 364~729 ppm). 그 기울기에서 크게 벗어나는 반복은 스트림이 정상상태가
    아니었다는 뜻이며, 그 주기 안에서 warp 가 비선형으로 움직여 FFT 의 정수주기
    가정을 깬다. 판정 근거는 **타임베이스 관측 하나**이고 결과를 보지 않는다.

    실측(20260804_235822_03f4c088): 중앙 4.38 샘플/주기인데 반복 0 의 국소값은
    10.44, 반복 1 은 5.60 이다 — 워밍업 4주기(0.5s)로는 스트림이 정상상태에 못 든다.
    """

    t = np.asarray(taus, dtype=np.float64).reshape(-1)
    if t.size < 3:
        raise ValueError(f"드리프트 판정에는 반복 3개 이상이 필요합니다: {t.size}")
    step = np.diff(t)
    centred = np.concatenate([[step[0]], 0.5 * (step[:-1] + step[1:]), [step[-1]]])
    return centred, float(np.median(step))


def relative_tau_outliers(
    tau_primary: np.ndarray,
    tau_secondary: np.ndarray,
    *,
    tolerance_samples: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """P−S 상대 τ 의 연속성 위반을 찾는다. ``(위반 마스크, 편차, 중앙값)``.

    두 채널은 **같은 DAC·같은 출력 스트림**의 인터리브다. 따라서 τ_P − τ_S 는
    설계 원리상 반복에 무관한 상수여야 하고, 이 값이 튀면 출력 버퍼가 한쪽
    채널에서만 미끄러진 것이다(실측 32 샘플 점프).

    기준(anchor) 반복은 τ 가 양쪽 모두 0 으로 **구조적으로** 고정되므로 중앙값
    계산에서 제외한다. 중앙값 편차를 쓰되 **MAD 로 스케일하지 않는다** — 오염
    반복이 과반이면 MAD 가 부풀어(실측 1.35/1.95 → 허용치 12/17 샘플) 슬립
    블록을 통째로 통과시킨다. 고정 임계가 이 자료에서 유일하게 견고하다.
    """

    a = np.asarray(tau_primary, dtype=np.float64).reshape(-1)
    b = np.asarray(tau_secondary, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"두 채널의 반복 수가 다릅니다: {a.size} != {b.size}")
    if float(tolerance_samples) <= 0.0:
        raise ValueError(f"tolerance_samples 는 양수여야 합니다: {tolerance_samples}")
    relative = a - b
    body = relative[1:] if relative.size > 1 else relative
    centre = float(np.median(body))
    deviation = np.abs(relative - centre)
    return deviation > float(tolerance_samples), deviation, centre


def align_repeats(
    frequencies_hz: np.ndarray,
    stack: np.ndarray,
    *,
    sample_rate: int,
    fit_band_hz: tuple[float, float] | None = None,
    span_samples: float = 512.0,
    anchor: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """반복별 시간이동을 제거하고 ``(정렬된 stack, τ 배열, 정렬 신뢰도 배열)`` 을 준다.

    신뢰도는 정렬 후 기준과의 정규화 상관이다. 이 값이 낮은 반복은 **τ 탐색이 봉우리를
    찾지 못한 것**이며, 그 τ 는 시간축 정보가 아니라 잡음이다. 호출자는 이 값으로
    실패한 반복을 걸러야 한다 — 걸러내지 않으면 잡음 τ 하나가 평균과 일관성을 함께
    망가뜨린다(실측: 상대 τ 편차가 2 → 128 샘플로 튀었다).

    평균을 기준으로 반복 정제하면 좋아질 것 같지만 실측에서는 **악화됐다** —
    정렬이 틀린 반복이 평균을 오염시켜 다음 회차를 더 끌고 갔다. 고정 기준이
    이 자료에서는 더 안전하다.

    ``anchor`` 는 기준으로 삼을 반복 인덱스다. 기본 0 은 **안전한 기본값이 아니다** —
    2026-08-05 실측에서 워밍업 4주기(0.5s) 직후의 첫 분석 주기는 아직 정상상태가
    아니고(국소 드리프트 10.4 vs 정상 4.4 샘플/주기), 그것을 기준으로 삼으면 전
    반복의 τ 가 함께 틀어진다. 호출자는 드리프트가 정상인 구간의 **중앙** 반복을
    지정해야 한다. 실측: 앵커 0 → 150-1600Hz 0.9689, 중앙 앵커 → 0.9991.
    """

    freq = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    values = np.asarray(stack, dtype=np.complex128)
    if not 0 <= int(anchor) < values.shape[0]:
        raise ValueError(f"anchor 가 범위 밖입니다: {anchor} / {values.shape[0]}")
    reference = values[int(anchor)]
    taus = np.zeros(values.shape[0], dtype=np.float64)
    scores = np.zeros(values.shape[0], dtype=np.float64)
    aligned = np.empty_like(values)
    for index in range(values.shape[0]):
        tau, score = estimate_repeat_delay(
            freq, values[index], reference, sample_rate=sample_rate,
            span_samples=span_samples, fit_band_hz=fit_band_hz,
        )
        taus[index] = tau
        scores[index] = score
        aligned[index] = values[index] * np.exp(
            2j * np.pi * freq * tau / float(sample_rate)
        )
    return aligned, taus, scores
