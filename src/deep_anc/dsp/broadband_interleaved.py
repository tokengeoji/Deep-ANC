"""광대역 P/S panel에 저역 clock pilot을 함께 넣는 결정적 probe 설계.

고역 dense tone만으로 만든 2026-08-27 진단 캡처는 adjacent-cycle clock witness가
0/64였다. panel 사이 anchor만으로는 8초 panel 내부의 affine timebase를 직접 관측할 수
없으므로, 모든 panel에 150--600 Hz pilot을 같은 주기·같은 DAC stream으로 함께 넣는다.

이 probe의 bin 집합은 pilot과 panel 사이에 gap이 있을 수 있다. 따라서 panel별 sparse
IR IFFT에는 사용하지 않고 fractional joint-LS로 복소 전달을 얻은 뒤, 겹치는 panel을
공통 시간축으로 stitch해 하나의 broadband compact FIR을 적합해야 한다.

이 모듈은 파형만 만들며 오디오 장치를 열지 않는다.
"""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

import numpy as np

from .interleaved_probe import InterleavedProbe, schroeder_phases


BROADBAND_CLOCK_PILOT_BAND_HZ = (150.0, 600.0)
BROADBAND_MARKER_BAND_HZ = (150.0, 11_313.708498984761)
BROADBAND_MARKER_SECONDS = 0.25
BROADBAND_MARKER_GUARD_SECONDS = 0.125
BROADBAND_MARKER_FREQUENCY_RESOLUTION_HZ = 4.0
BROADBAND_CANONICAL_PANEL_BANDS_HZ = (
    (100.0, 1800.0),
    (1400.0, 3200.0),
    (2800.0, 6000.0),
    (5400.0, 8500.0),
    (7800.0, 11400.0),
)
SUBMITTED_PILOT_CROSS_CHANNEL_MAX_ABSOLUTE = 1.0e-8
SUBMITTED_PILOT_CROSS_CHANNEL_MAX_RATIO = 1.0e-12


def _absolute_bin_phases(bins: np.ndarray, *, salt: int) -> np.ndarray:
    """선택 tone 수와 무관한 결정적 phase를 만든다.

    기존 Schroeder phase는 ``selected.size``에 의존해 panel이 바뀌면 공통 pilot의
    phase도 바뀌었다. global-clock pilot은 절대 bin과 channel salt만으로 정해져야 한다.
    이 함수는 암호학적 난수 용도가 아니라 exact 재생 가능한 위상표의 단일 출처다.
    """

    selected = np.asarray(bins, dtype=np.int64).reshape(-1)
    salt_word = np.uint64(
        (int(salt) * 0xC2B2AE3D27D4EB4F) & ((1 << 64) - 1)
    )
    phase_words = selected.astype(np.uint64) * np.uint64(
        0x9E3779B185EBCA87
    ) + salt_word
    mantissa = (phase_words >> np.uint64(11)).astype(np.float64)
    return 2.0 * np.pi * mantissa / float(1 << 53)


def fixed_clock_pilot_complex_spectrum(
    *, sample_rate: int, period_seconds: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """모든 panel/transition이 공유하는 pilot bin과 unit complex coefficient.

    반환 coefficient는 amplitude normalization 이전의 권위 값이다. 실제 submitted
    PCM SHA와 함께 plan에 봉인되며 panel high-tone phase와 독립이다.
    """

    rate = int(sample_rate)
    period = int(round(float(period_seconds) * rate))
    if rate <= 0 or period <= 0:
        raise ValueError("sample_rate/period는 양수여야 합니다")
    resolution = rate / period
    bins = _merge_integer_bins(
        bands_hz=(BROADBAND_CLOCK_PILOT_BAND_HZ,),
        resolution_hz=resolution,
        nyquist_hz=rate / 2.0,
    )
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for drive, parity, salt in (("noise", 0, 11), ("cancel", 1, 29)):
        selected = bins[(bins % 2) == parity]
        result[drive] = (
            selected,
            # pilot bin 집합은 모든 panel에서 같으므로 독립 Schroeder 위상도
            # 모든 panel에서 exact 동일하면서 random phase보다 crest가 낮다.
            np.exp(1j * schroeder_phases(selected.size)),
        )
    return result


def fixed_clock_pilot_sha256(*, sample_rate: int, period_seconds: float) -> str:
    digest = hashlib.sha256()
    for drive in ("noise", "cancel"):
        bins, coefficients = fixed_clock_pilot_complex_spectrum(
            sample_rate=sample_rate, period_seconds=period_seconds
        )[drive]
        digest.update(drive.encode("ascii"))
        digest.update(np.asarray(bins, dtype="<i8").tobytes())
        digest.update(np.asarray(coefficients.real, dtype="<f8").tobytes())
        digest.update(np.asarray(coefficients.imag, dtype="<f8").tobytes())
    return digest.hexdigest()


def validate_submitted_pilot_cross_channel_null(
    pcm_period: np.ndarray,
    *,
    sample_rate: int,
    period_seconds: float,
) -> dict[str, object]:
    """actual int16 PCM의 pilot bin에서 반대 DAC channel이 null인지 검증한다.

    absolute-bin parity와 반주기 대칭은 int16 양자화 뒤에도 유지돼야 한다.
    이 조건이 깨지면 한 mic bin을 ``P*X0`` 또는 ``S*X1``로 분리할 수 없으므로
    actual-spectrum global clock 분석을 즉시 막는다.
    """

    pcm = np.asarray(pcm_period)
    rate = int(sample_rate)
    period = int(round(float(period_seconds) * rate))
    if pcm.dtype != np.int16 or pcm.shape != (period, 2):
        raise ValueError("submitted pilot null 검증에는 exact one-period [N,2] int16이 필요합니다")
    spectra = [
        np.fft.rfft(pcm[:, channel].astype(np.float64)) for channel in (0, 1)
    ]
    fixed = fixed_clock_pilot_complex_spectrum(
        sample_rate=rate, period_seconds=float(period_seconds)
    )
    rows: dict[str, dict[str, object]] = {}
    digest = hashlib.sha256()
    for drive, channel in (("noise", 0), ("cancel", 1)):
        bins = np.asarray(fixed[drive][0], dtype=np.int64)
        main = spectra[channel][bins]
        cross = spectra[1 - channel][bins]
        main_min = float(np.min(np.abs(main)))
        cross_max = float(np.max(np.abs(cross)))
        ratio = cross_max / max(main_min, np.finfo(np.float64).tiny)
        passed = bool(
            main_min > 0.0
            and cross_max <= SUBMITTED_PILOT_CROSS_CHANNEL_MAX_ABSOLUTE
            and ratio <= SUBMITTED_PILOT_CROSS_CHANNEL_MAX_RATIO
        )
        if not passed:
            raise ValueError(
                f"{drive} actual int16 pilot의 반대 channel null이 깨졌습니다: "
                f"absolute={cross_max:.12g}, ratio={ratio:.12g}"
            )
        digest.update(drive.encode("ascii"))
        for values in (main, cross):
            digest.update(np.asarray(values.real, dtype="<f8").tobytes())
            digest.update(np.asarray(values.imag, dtype="<f8").tobytes())
        rows[drive] = {
            "main_min_magnitude": main_min,
            "cross_channel_max_magnitude": cross_max,
            "cross_to_main_max_ratio": ratio,
            "passed": passed,
        }
    return {
        "schema": "submitted_int16_pilot_cross_channel_null_v1",
        "maximum_absolute": SUBMITTED_PILOT_CROSS_CHANNEL_MAX_ABSOLUTE,
        "maximum_ratio": SUBMITTED_PILOT_CROSS_CHANNEL_MAX_RATIO,
        "drives": rows,
        "spectrum_sha256": digest.hexdigest(),
        "passed": True,
    }


def build_nonperiodic_timing_markers(
    *,
    sample_rate: int,
    duration_seconds: float = BROADBAND_MARKER_SECONDS,
    amplitude: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """P-only/S-only slot에 넣을 결정적 4 Hz-grid 비주기 marker를 만든다.

    marker는 channel별로 별도 slot에서 재생된다. 0.25초 FFT grid는 4 Hz이고
    delay alias 주기 12,000 samples가 0..4,800 coarse 탐색보다 크다.
    """

    rate = int(sample_rate)
    frames = int(round(float(duration_seconds) * rate))
    amp = float(amplitude)
    if rate <= 0 or frames <= 0 or not 0.0 < amp <= 1.0:
        raise ValueError("timing marker sample-rate/duration/amplitude가 잘못됐습니다")
    resolution = rate / frames
    if not math.isclose(
        resolution,
        BROADBAND_MARKER_FREQUENCY_RESOLUTION_HZ,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("timing marker FFT grid는 exact 4 Hz여야 합니다")
    first = int(math.ceil(BROADBAND_MARKER_BAND_HZ[0] / resolution))
    last = int(math.floor(BROADBAND_MARKER_BAND_HZ[1] / resolution))
    bins = np.arange(first, last + 1, dtype=np.int64)
    output = np.zeros((frames, 2), dtype=np.float32)
    channel_sha: dict[str, str] = {}
    for channel, (drive, salt) in enumerate((("noise", 101), ("cancel", 211))):
        spectrum = np.zeros(frames // 2 + 1, dtype=np.complex128)
        spectrum[bins] = np.exp(1j * _absolute_bin_phases(bins, salt=salt))
        signal = np.fft.irfft(spectrum, n=frames)
        peak = float(np.max(np.abs(signal)))
        if not math.isfinite(peak) or peak <= 0.0:
            raise ValueError("timing marker 합성 peak가 유효하지 않습니다")
        output[:, channel] = np.asarray(signal / peak * amp, dtype=np.float32)
        channel_sha[drive] = hashlib.sha256(
            np.asarray(output[:, channel], dtype="<f4").tobytes()
        ).hexdigest()
    metadata: dict[str, object] = {
        "duration_seconds": float(duration_seconds),
        "frames": frames,
        "frequency_resolution_hz": resolution,
        "frequency_gcd_hz": resolution,
        "delay_alias_period_samples": int(round(rate / resolution)),
        "band_hz": [float(BROADBAND_MARKER_BAND_HZ[0]), float(BROADBAND_MARKER_BAND_HZ[1])],
        "bins": [int(first), int(last)],
        "float32_channel_sha256": channel_sha,
    }
    return output, metadata


def _merge_integer_bins(
    *,
    bands_hz: Sequence[Sequence[float]],
    resolution_hz: float,
    nyquist_hz: float,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for index, raw in enumerate(bands_hz):
        if len(raw) != 2:
            raise ValueError(f"excitation band #{index}는 [lo, hi]여야 합니다")
        lo, hi = (float(value) for value in raw)
        if not (
            math.isfinite(lo)
            and math.isfinite(hi)
            and 0.0 < lo < hi < float(nyquist_hz)
        ):
            raise ValueError(f"excitation band #{index}가 유효하지 않습니다: {raw!r}")
        first = int(math.ceil(lo / resolution_hz))
        last = int(math.floor(hi / resolution_hz))
        if last < first:
            raise ValueError(f"excitation band #{index}에 정수 FFT bin이 없습니다")
        rows.append(np.arange(first, last + 1, dtype=np.int64))
    bins = np.unique(np.concatenate(rows))
    if bins.size < 16:
        raise ValueError("clock-piloted interleaved probe의 tone이 16개 미만입니다")
    return bins


def build_clock_piloted_panel_probe(
    *,
    sample_rate: int,
    period_seconds: float,
    panel_band_hz: tuple[float, float],
    amplitude: float,
    clock_pilot_band_hz: tuple[float, float] = BROADBAND_CLOCK_PILOT_BAND_HZ,
) -> InterleavedProbe:
    """한 high-band panel과 공통 low-band clock pilot을 동시에 합성한다.

    union bin을 정렬한 뒤 두 DAC 채널에 번갈아 배치한다. 각 채널 파형은 독립적으로
    ``amplitude`` peak에 맞춘다. 두 채널의 합이 아니라 실제 DAC channel별 peak 계약이다.
    """

    rate = int(sample_rate)
    period = int(round(float(period_seconds) * rate))
    amp = float(amplitude)
    if rate <= 0 or period <= 0:
        raise ValueError("sample_rate/period는 양수여야 합니다")
    if not math.isfinite(amp) or not 0.0 < amp <= 1.0:
        raise ValueError("amplitude는 유한한 (0, 1]이어야 합니다")
    resolution = rate / period
    union_bins = _merge_integer_bins(
        bands_hz=(clock_pilot_band_hz, panel_band_hz),
        resolution_hz=resolution,
        nyquist_hz=rate / 2.0,
    )
    # panel별 시작 주파수가 달라도 overlap tone의 DAC 역할이 바뀌면 안 된다. union의
    # 상대 index로 번갈아 나누면 gap 길이에 따라 같은 2.8 kHz bin이 P였다가 S가 될 수
    # 있다. 절대 FFT-bin parity를 역할로 고정해야 인접 panel에 동일 drive·동일 frequency
    # phase anchor가 남고, interpolation 없이 phase stitch를 재검산할 수 있다.
    noise_bins = union_bins[(union_bins % 2) == 0]
    cancel_bins = union_bins[(union_bins % 2) == 1]
    if min(noise_bins.size, cancel_bins.size) < 8:
        raise ValueError("각 출력 채널에 tone이 8개 이상 필요합니다")

    fixed_pilot = fixed_clock_pilot_complex_spectrum(
        sample_rate=rate, period_seconds=float(period_seconds)
    )

    def raw_signal(selected: np.ndarray, *, drive: str, salt: int) -> np.ndarray:
        spectrum = np.zeros(period // 2 + 1, dtype=np.complex128)
        pilot_bins, pilot_coefficients = fixed_pilot[drive]
        pilot_positions = np.searchsorted(selected, pilot_bins)
        if np.any(pilot_positions >= selected.size) or not np.array_equal(
            selected[pilot_positions], pilot_bins
        ):
            raise ValueError(f"{drive} panel에 fixed clock pilot bin이 없습니다")
        high_mask = np.ones(selected.size, dtype=np.bool_)
        high_mask[pilot_positions] = False
        high_bins = selected[high_mask]
        spectrum[pilot_bins] = pilot_coefficients
        # high-tone phase는 panel별 독립 Schroeder 최적화를 사용해 crest를 낮춘다.
        # fixed pilot coefficient에는 손대지 않으므로 clock authority와 독립이다.
        spectrum[high_bins] = np.exp(1j * schroeder_phases(high_bins.size))
        return np.fft.irfft(spectrum, period)

    # panel별 peak normalization은 공통 pilot amplitude/phase를 바꾸므로 금지한다.
    # canonical 다섯 panel의 실제 peak 최댓값으로 channel별 scale을 한 번만 정한다.
    global_peaks = {"noise": 0.0, "cancel": 0.0}
    for canonical_panel in BROADBAND_CANONICAL_PANEL_BANDS_HZ:
        canonical_union = _merge_integer_bins(
            bands_hz=(clock_pilot_band_hz, canonical_panel),
            resolution_hz=resolution,
            nyquist_hz=rate / 2.0,
        )
        for drive, parity, salt in (("noise", 0, 41), ("cancel", 1, 73)):
            canonical_selected = canonical_union[(canonical_union % 2) == parity]
            global_peaks[drive] = max(
                global_peaks[drive],
                float(
                    np.max(
                        np.abs(
                            raw_signal(
                                canonical_selected, drive=drive, salt=salt
                            )
                        )
                    )
                ),
            )

    def synthesize(selected: np.ndarray, *, drive: str, salt: int) -> np.ndarray:
        signal = raw_signal(selected, drive=drive, salt=salt)
        peak = global_peaks[drive]
        if not math.isfinite(peak) or peak <= 0.0:
            raise ValueError("clock-piloted probe 합성 peak가 유효하지 않습니다")
        # float64을 유지해야 서로 다른 high-tone panel의 FFT round-off가 fixed
        # pilot complex coefficient에 float32 양자화 노이즈로 섞이지 않는다.
        return np.asarray(signal / peak * amp, dtype=np.float64)

    return InterleavedProbe(
        sample_rate=rate,
        period_samples=period,
        noise_signal=synthesize(noise_bins, drive="noise", salt=41),
        cancel_signal=synthesize(cancel_bins, drive="cancel", salt=73),
        noise_bins=noise_bins,
        cancel_bins=cancel_bins,
        frequencies_hz=union_bins.astype(np.float64) * resolution,
    )


__all__ = [
    "BROADBAND_CLOCK_PILOT_BAND_HZ",
    "BROADBAND_CANONICAL_PANEL_BANDS_HZ",
    "BROADBAND_MARKER_BAND_HZ",
    "BROADBAND_MARKER_SECONDS",
    "BROADBAND_MARKER_GUARD_SECONDS",
    "build_nonperiodic_timing_markers",
    "build_clock_piloted_panel_probe",
    "fixed_clock_pilot_complex_spectrum",
    "fixed_clock_pilot_sha256",
    "validate_submitted_pilot_cross_channel_null",
]
