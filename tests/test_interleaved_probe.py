"""주파수 인터리브 동시 자극이 **비동기 클록 wander 아래에서도** 상대 전달을 복원하는지 검증한다.

이 테스트가 이 설계의 존재 이유를 그대로 검사한다. 실측에서 확인된 사실은:

* 재생 프로그램 기준 반복 간 coherence 0.08~0.17 (요구 0.9)
* 같은 녹음을 ERR/REF 기준으로 보면 0.99+
* 크기 |H| 는 반복 간 std 0.08dB 로 재현, **위상만** 무작위

즉 고장은 DAC↔ADC 시간축 대응 하나다. 여기서는 그 wander 를 **시뮬레이션으로 주입**하고,
순차 측정(기존 방식)은 무너지지만 동시 인터리브 측정은 살아남는지를 대조한다.
하드웨어 없이 설계의 옳고 그름을 가르는 것이 목적이다.
"""

from __future__ import annotations

import numpy as np
import pytest

from deep_anc.dsp.interleaved_probe import (
    build_interleaved_probe,
    align_repeats,
    channel_impulse_response,
    complex_consistency,
    crest_factor_db,
    dewarp_recording,
    estimate_repeat_delay,
    estimate_transfer,
    relative_tau_outliers,
    schroeder_phases,
    timebase_drift,
    tone_snr_db,
    track_warp,
)

FS = 48000
BAND = (80.0, 1600.0)


def make_probe(period_seconds: float = 2.0, amplitude: float = 0.2):
    return build_interleaved_probe(
        sample_rate=FS,
        period_seconds=period_seconds,
        band_hz=BAND,
        amplitude=amplitude,
    )


def duct_response(frequencies: np.ndarray, *, delay_samples: float, seed: int) -> np.ndarray:
    """공진이 있는 그럴듯한 덕트 전달함수. 축방향 공진 70/210/350/489/629 Hz."""

    omega = 2.0 * np.pi * frequencies / FS
    response = np.zeros_like(frequencies, dtype=np.complex128)
    rng = np.random.default_rng(seed)
    for centre in (70.0, 210.0, 350.0, 489.0, 629.0, 900.0, 1300.0):
        gain = rng.uniform(0.5, 1.5)
        width = rng.uniform(12.0, 30.0)
        response += gain / (1.0 + 1j * (frequencies - centre) / width)
    return response * np.exp(-1j * omega * delay_samples)


def apply_warp(signal: np.ndarray, warp_samples: np.ndarray) -> np.ndarray:
    """ADC 가 자기 클록으로 음향장을 다시 샘플링하는 것을 모사한다.

    **warp 는 합쳐진 음향 신호에 걸린다.** 채널별 신호에 따로 거는 모델은 틀렸다 —
    두 스피커는 같은 DAC 타임베이스로 소리를 내고, 어긋나는 것은 그 음향장을 ADC 가
    언제 샘플링하는가이다. 이 구분이 설계의 성패를 가른다(공기 중 상쇄는 클록과 무관하다).
    """

    index = np.arange(signal.size, dtype=np.float64)
    return np.interp(index - warp_samples, index, signal, left=0.0, right=0.0)


def wander_trajectory(period: int, amplitude_samples: float, seed: int = 5) -> np.ndarray:
    """실측과 같은 성격의 느린 무작위 wander. 단조 드리프트가 아니다.

    크기 근거: 저장된 8초 ESS 측정에서 반복 간 위상차의 1차 적합 잔차가 분석 창 길이에
    비례해 커졌다 — 0.77초 창 0.12 rad, 1.36초 0.24 rad, 2.26초 2.33 rad, 3.70초 3.99 rad.
    600Hz 에서 0.25 rad 은 3.2 샘플에 해당하므로, 1~2초 창의 현실적인 wander 는 이 정도다.
    """

    rng = np.random.default_rng(seed)
    coarse = rng.normal(0.0, amplitude_samples, 12)
    return np.interp(np.arange(period), np.linspace(0, period - 1, coarse.size), coarse)


def synth_recording(probe, warp_samples, *, seed=0, noise_dbfs=-70.0):
    """두 채널을 동시에 재생했을 때의 ERR 마이크 녹음을 합성한다."""

    period = probe.period_samples
    frequencies = np.fft.rfftfreq(period, 1.0 / FS)
    primary = duct_response(frequencies, delay_samples=1489.0, seed=11)
    secondary = duct_response(frequencies, delay_samples=1342.0, seed=23)

    def through(signal, response):
        return np.fft.irfft(np.fft.rfft(np.asarray(signal, dtype=np.float64)) * response, period)

    field = through(probe.noise_signal, primary) + through(probe.cancel_signal, secondary)
    rng = np.random.default_rng(seed)
    error = apply_warp(field, warp_samples) + rng.normal(
        0.0, 10.0 ** (noise_dbfs / 20.0), period
    )
    return error, primary, secondary, frequencies


def relative_error_db(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(
        20.0 * np.log10(np.linalg.norm(estimate - truth) / (np.linalg.norm(truth) + 1e-18))
    )


def test_bin_sets_are_disjoint_and_interleaved():
    probe = make_probe()
    assert not set(probe.noise_bins.tolist()) & set(probe.cancel_bins.tolist())
    merged = np.sort(np.concatenate([probe.noise_bins, probe.cancel_bins]))
    # 번갈아 배치되어야 인접 빈 사이 warp 위상차가 최소가 된다.
    owner = np.isin(merged, probe.noise_bins).astype(int)
    assert np.all(np.diff(owner) != 0), "두 채널의 톤이 번갈아 나와야 한다"


def test_tones_stay_inside_requested_band():
    probe = make_probe()
    assert probe.frequencies_hz.min() >= BAND[0] - 1.0
    assert probe.frequencies_hz.max() <= BAND[1] + 1.0


def test_crest_factor_is_low_enough_to_carry_energy():
    """크레스트가 높으면 같은 피크 제한에서 음향 에너지를 잃는다 — 실제로 겪은 실패다."""

    probe = make_probe()
    noise_crest, cancel_crest = probe.crest_db()
    assert noise_crest < 12.0, f"소음 채널 크레스트 {noise_crest:.1f}dB 가 너무 높다"
    assert cancel_crest < 12.0, f"상쇄 채널 크레스트 {cancel_crest:.1f}dB 가 너무 높다"


def test_schroeder_beats_random_phase_crest():
    count = 300
    rng = np.random.default_rng(0)
    period = 1 << 14
    bins = np.arange(50, 50 + count)

    def build(phases):
        spectrum = np.zeros(period // 2 + 1, dtype=np.complex128)
        spectrum[bins] = np.exp(1j * phases)
        return np.fft.irfft(spectrum, period)

    assert crest_factor_db(build(schroeder_phases(count))) < crest_factor_db(
        build(rng.uniform(0, 2 * np.pi, count))
    )


def test_no_crosstalk_between_channels_without_warp():
    """빈 집합이 서로소이므로 한 채널의 강한 톤이 다른 채널 추정을 오염시키면 안 된다."""

    probe = make_probe()
    error, primary, secondary, _ = synth_recording(
        probe, np.zeros(probe.period_samples), noise_dbfs=-200.0
    )
    _, estimated_p = estimate_transfer(error, probe, drive="noise")
    _, estimated_s = estimate_transfer(error, probe, drive="cancel")
    assert relative_error_db(estimated_p, primary[probe.noise_bins]) < -60.0
    assert relative_error_db(estimated_s, secondary[probe.cancel_bins]) < -60.0


def measure_ratio(probe, error):
    """S 를 P 의 빈 격자로 보간해 상대 전달 S/P 를 만든다."""

    freq_p, estimated_p = estimate_transfer(error, probe, drive="noise")
    freq_s, estimated_s = estimate_transfer(error, probe, drive="cancel")
    interpolated_s = np.interp(freq_p, freq_s, estimated_s.real) + 1j * np.interp(
        freq_p, freq_s, estimated_s.imag
    )
    return freq_p, interpolated_s / estimated_p


@pytest.mark.parametrize(
    ("period_seconds", "wander_samples"),
    [(1.0, 1.5), (1.0, 3.2), (2.0, 1.5), (2.0, 3.2)],
)
def test_relative_transfer_survives_realistic_wander(period_seconds, wander_samples):
    """설계의 핵심 주장 — 동시에 재면 S/P 는 현실적 wander 에서 살아남는다.

    (창, wander) 조합은 실측에서 유도한 값이다(`wander_trajectory` 주석 참조).
    """

    probe = make_probe(period_seconds=period_seconds)
    warp = wander_trajectory(probe.period_samples, wander_samples)
    error, primary, secondary, _ = synth_recording(probe, warp)
    freq_p, measured_ratio = measure_ratio(probe, error)
    truth_ratio = secondary[probe.noise_bins] / primary[probe.noise_bins]

    band = (freq_p >= 150.0) & (freq_p <= 600.0)
    error_db = relative_error_db(measured_ratio[band], truth_ratio[band])
    assert error_db < -20.0, f"상대 전달 오차 {error_db:.1f} dB — warp 상쇄가 실패했다"


def test_long_window_degrades_as_measured_on_hardware():
    """긴 창은 무너져야 한다 — 기존 도구가 8초 ESS 를 쓴 것이 실패의 직접 원인이었다.

    이 대조가 없으면 위 통과가 "애초에 문제가 없었다"는 뜻일 수도 있다.
    """

    probe = make_probe(period_seconds=2.0)
    short = wander_trajectory(probe.period_samples, 3.2)
    long_window = wander_trajectory(probe.period_samples, 40.0)

    def error_for(warp):
        error, primary, secondary, _ = synth_recording(probe, warp)
        freq_p, ratio = measure_ratio(probe, error)
        truth = secondary[probe.noise_bins] / primary[probe.noise_bins]
        band = (freq_p >= 150.0) & (freq_p <= 600.0)
        return relative_error_db(ratio[band], truth[band])

    assert error_for(short) < error_for(long_window) - 10.0


def test_sequential_measurement_fails_under_the_same_wander():
    """대조군 — 순차 측정(기존 방식)은 무너져야 한다.

    wander 크기는 **측정 간격**에서 나온다. 기존 도구는 NS epoch 7회를 다 재생한 뒤 CS
    epoch 를 재생하므로 두 경로가 약 50초 떨어져 있고, 그 사이 누적 wander 는 8초 창에서
    관측된 4 rad(600Hz 기준 약 50 샘플)보다 크다. 동시 측정(위 테스트)의 1~2초 간격과
    같은 값을 쓰면 대조가 성립하지 않는다 — 실제로 3.2 샘플에서는 순차 측정도 -22dB 로
    멀쩡했다. 이것이 "간격이 문제다"라는 이 설계의 근거 그 자체다.
    """

    probe = make_probe()
    period = probe.period_samples
    frequencies = np.fft.rfftfreq(period, 1.0 / FS)
    primary = duct_response(frequencies, delay_samples=1489.0, seed=11)
    secondary = duct_response(frequencies, delay_samples=1342.0, seed=23)
    selected = probe.noise_bins

    def measure(response, warp):
        field = np.fft.irfft(
            np.fft.rfft(np.asarray(probe.noise_signal, dtype=np.float64)) * response, period
        )
        observed = np.fft.rfft(apply_warp(field, warp))
        reference = np.fft.rfft(np.asarray(probe.noise_signal, dtype=np.float64))
        return observed[selected] / reference[selected]

    # 50초 간격 = 8초 창에서 관측된 4 rad(600Hz 기준 ~50 샘플) 이상의 누적 wander
    estimated_p = measure(primary, wander_trajectory(period, 50.0, seed=1))
    estimated_s = measure(secondary, wander_trajectory(period, 50.0, seed=2))
    measured_ratio = estimated_s / estimated_p
    truth_ratio = secondary[selected] / primary[selected]

    frequencies_selected = selected * FS / period
    band = (frequencies_selected >= 150.0) & (frequencies_selected <= 600.0)
    assert relative_error_db(measured_ratio[band], truth_ratio[band]) > -12.0, (
        "순차 측정이 wander 에서 멀쩡하다면 애초에 고칠 문제가 없다는 뜻이다"
    )


def test_rejects_band_outside_nyquist():
    with pytest.raises(ValueError):
        build_interleaved_probe(
            sample_rate=FS, period_seconds=2.0, band_hz=(80.0, 30000.0), amplitude=0.2
        )


def test_rejects_too_few_tones():
    with pytest.raises(ValueError):
        build_interleaved_probe(
            sample_rate=FS, period_seconds=0.01, band_hz=(80.0, 100.0), amplitude=0.2
        )


def test_rejects_segment_length_mismatch():
    probe = make_probe()
    with pytest.raises(ValueError):
        estimate_transfer(np.zeros(probe.period_samples - 1), probe, drive="noise")


# ---------------------------------------------------------------------------
# IR 복원 — 빈-희소 전달함수를 시간영역으로 되돌리는 부분
#
# 이 경로에는 **조용히 틀리기 쉬운 스케일 인자**가 하나 있다. 채널은 bin_step 마다
# 하나씩만 빈을 가지므로, 나머지를 0 으로 두고 역변환하면 진폭이 1/bin_step 로 줄고
# 결과가 period/bin_step 마다 반복된다. 인자를 빠뜨려도 IR 모양은 그대로라 눈으로는
# 보이지 않고, 학습·평가에 들어가는 플랜트 이득만 조용히 틀어진다.
# ---------------------------------------------------------------------------


def band_limited_delay_plant(frequencies: np.ndarray, *, delay_samples: float, gain: float):
    omega = 2.0 * np.pi * frequencies / FS
    return gain * np.exp(-1j * omega * delay_samples)


def reconstruct(probe, plant_fn, *, drive: str, pre_roll: int = 256):
    period = probe.period_samples
    frequencies = np.fft.rfftfreq(period, 1.0 / FS)
    primary = plant_fn(frequencies)

    def through(signal):
        return np.fft.irfft(
            np.fft.rfft(np.asarray(signal, dtype=np.float64)) * primary, period
        )

    field = through(probe.noise_signal) + through(probe.cancel_signal)
    _, transfer = estimate_transfer(field, probe, drive=drive)
    return channel_impulse_response(probe, transfer, drive=drive, pre_roll=pre_roll)


@pytest.mark.parametrize("spacing_hz", [2.0, 4.0, 8.0])
def test_impulse_response_amplitude_is_independent_of_tone_spacing(spacing_hz):
    """스케일 인자가 빠지면 톤 간격을 바꿀 때 진폭이 그 비율만큼 달라진다."""

    probe = build_interleaved_probe(
        sample_rate=FS,
        period_seconds=1.0,
        band_hz=(70.0, 1610.0),
        amplitude=0.02,
        tone_spacing_hz=spacing_hz,
    )
    ir = reconstruct(
        probe,
        lambda f: band_limited_delay_plant(f, delay_samples=1489.0, gain=0.25),
        drive="noise",
    )
    # 순수 지연·이득 0.25 의 대역제한 IR 은 대역폭 비율만큼의 첨두를 갖는다.
    # 대역이 같으므로 톤 간격과 무관하게 같은 값이어야 한다.
    peak = float(np.max(np.abs(ir)))
    expected = 0.25 * (1610.0 - 70.0) * 2.0 / FS
    assert peak == pytest.approx(expected, rel=0.05), (
        f"spacing {spacing_hz}Hz 에서 첨두 {peak:.6f} != 기대 {expected:.6f} — "
        "bin_step 스케일 보정이 빠졌을 때 정확히 이렇게 어긋난다"
    )


def test_impulse_response_places_onset_at_delay_plus_pre_roll():
    probe = build_interleaved_probe(
        sample_rate=FS, period_seconds=1.0, band_hz=(70.0, 1610.0),
        amplitude=0.02, tone_spacing_hz=4.0,
    )
    ir = reconstruct(
        probe,
        lambda f: band_limited_delay_plant(f, delay_samples=1489.0, gain=0.25),
        drive="noise",
        pre_roll=256,
    )
    assert int(np.argmax(np.abs(ir))) == 1489 + 256


def test_pre_roll_keeps_leading_ringing_in_front_of_the_onset():
    """대역제한 IR 은 첨두 **앞쪽**으로도 링잉한다.

    지연이 작으면 pre_roll 없이는 그 링잉이 배열 앞에 담길 자리가 없어 주기 끝으로
    감겨 들어간다. 그러면 ``_model_from_repeat_irs`` 가 잘라내는 compact FIR 이
    선행 성분을 잃는다 — 극성·군지연이 미묘하게 틀어지는 방식이다.
    """

    probe = build_interleaved_probe(
        sample_rate=FS, period_seconds=1.0, band_hz=(70.0, 1610.0),
        amplitude=0.02, tone_spacing_hz=4.0,
    )
    plant = lambda f: band_limited_delay_plant(f, delay_samples=3.0, gain=0.25)  # noqa: E731
    wrapped = reconstruct(probe, plant, drive="noise", pre_roll=0)
    rolled = reconstruct(probe, plant, drive="noise", pre_roll=256)

    assert int(np.argmax(np.abs(wrapped))) == 3
    assert int(np.argmax(np.abs(rolled))) == 3 + 256
    # pre_roll 이 없으면 첨두 **직전** 표본이 배열 끝에 가 있다.
    peak = float(np.max(np.abs(wrapped)))
    assert abs(float(wrapped[-1])) > 0.5 * peak
    # pre_roll 을 주면 그 자리에 멀리 떨어진 꼬리만 남는다.
    assert abs(float(rolled[-1])) < 0.1 * float(np.max(np.abs(rolled)))


@pytest.mark.parametrize("drive", ["noise", "cancel"])
def test_pre_roll_round_trips_complex_transfer_for_both_bin_parities(drive):
    """최종 FIR 이 even P뿐 아니라 odd S의 복소 전달함수도 그대로 재현해야 한다.

    기본 probe의 noise bin은 even이라 반주기 alias가 periodic이고, cancel bin은 odd라
    anti-periodic이다. 첨두 위치만 검사하면 cancel pre-roll의 감긴 앞부분 부호가 틀려도
    통과하므로, 실제 저장 계약처럼 effective delay를 적용한 전체 복소 응답을 비교한다.
    """

    probe = build_interleaved_probe(
        sample_rate=FS,
        period_seconds=0.125,
        band_hz=(60.0, 1650.0),
        amplitude=0.02,
        tone_spacing_hz=16.0,
    )
    selected = probe.bins_for(drive)
    frequencies = selected * FS / probe.period_samples
    rng = np.random.default_rng(20260814)
    taps = rng.normal(size=113) * np.exp(-np.arange(113) / 18.0)
    omega = 2.0 * np.pi * frequencies / FS
    measured = np.exp(-1j * np.outer(omega, np.arange(taps.size))) @ taps
    pre_roll = 256

    compact = channel_impulse_response(
        probe, measured, drive=drive, pre_roll=pre_roll
    )
    reconstructed = (
        np.exp(-1j * np.outer(omega, np.arange(compact.size))) @ compact
    ) * np.exp(1j * omega * pre_roll)

    np.testing.assert_allclose(reconstructed, measured, rtol=1e-11, atol=1e-11)


def test_channel_impulse_response_rejects_pre_roll_beyond_the_period():
    probe = build_interleaved_probe(
        sample_rate=FS, period_seconds=1.0, band_hz=(70.0, 1610.0),
        amplitude=0.02, tone_spacing_hz=4.0,
    )
    with pytest.raises(ValueError):
        channel_impulse_response(
            probe,
            np.zeros(probe.noise_bins.size),
            drive="noise",
            pre_roll=probe.period_samples,
        )


def test_relative_delay_of_the_two_paths_survives_wander():
    """동시 측정의 핵심 주장 — wander 가 있어도 **P 와 S 의 차이**는 보존된다."""

    probe = build_interleaved_probe(
        sample_rate=FS, period_seconds=1.0, band_hz=(70.0, 1610.0),
        amplitude=0.02, tone_spacing_hz=4.0,
    )
    period = probe.period_samples
    frequencies = np.fft.rfftfreq(period, 1.0 / FS)
    p_delay, s_delay = 1489.0, 1342.0
    primary = band_limited_delay_plant(frequencies, delay_samples=p_delay, gain=0.25)
    secondary = band_limited_delay_plant(frequencies, delay_samples=s_delay, gain=0.25)

    def through(signal, response):
        return np.fft.irfft(
            np.fft.rfft(np.asarray(signal, dtype=np.float64)) * response, period
        )

    field = through(probe.noise_signal, primary) + through(probe.cancel_signal, secondary)
    warped = apply_warp(field, wander_trajectory(period, 3.2, seed=7))

    onsets = {}
    for drive in ("noise", "cancel"):
        _, transfer = estimate_transfer(warped, probe, drive=drive)
        ir = channel_impulse_response(probe, transfer, drive=drive, pre_roll=256)
        onsets[drive] = int(np.argmax(np.abs(ir))) - 256
    assert onsets["noise"] - onsets["cancel"] == pytest.approx(p_delay - s_delay, abs=1)


def test_default_spacing_gives_guard_one():
    """기본(=주파수 분해능) 간격이라야 인접 빈이 서로 다른 채널이 된다.

    guard 가 커지면 두 채널의 톤이 주파수축에서 멀어지고, 그만큼 두 경로가 **서로 다른
    주파수에서** 관측된다. 동시 측정의 이점을 스스로 깎는 설정이므로 게이트가 1 을 강제한다.
    """

    probe = build_interleaved_probe(
        sample_rate=FS, period_seconds=1.0, band_hz=(70.0, 1610.0), amplitude=0.02
    )
    assert probe.guard_bins() == 1
    assert probe.bin_step("noise") == 2
    assert probe.bin_step("cancel") == 2


def test_wider_tone_spacing_widens_the_guard():
    probe = build_interleaved_probe(
        sample_rate=FS, period_seconds=1.0, band_hz=(70.0, 1610.0),
        amplitude=0.02, tone_spacing_hz=4.0,
    )
    assert probe.guard_bins() == 2
    assert probe.bin_step("noise") == 4


def test_channel_impulse_response_rejects_length_mismatch():
    probe = make_probe(period_seconds=1.0)
    with pytest.raises(ValueError):
        channel_impulse_response(
            probe, np.zeros(probe.noise_bins.size - 1), drive="noise"
        )


def test_tone_snr_uses_the_matching_bins():
    probe = make_probe(period_seconds=1.0)
    spectrum = np.zeros(probe.period_samples // 2 + 1, dtype=np.complex128)
    spectrum[probe.noise_bins] = 10.0
    floor = np.full(probe.period_samples // 2 + 1, 1.0, dtype=np.complex128)
    snr = tone_snr_db(spectrum, floor, probe.noise_bins)
    assert np.allclose(snr, 20.0)
    assert np.all(tone_snr_db(spectrum, floor, probe.cancel_bins) < -100.0)


# ---------------------------------------------------------------------------
# warp 추적 — 실측이 강제한 단계
# ---------------------------------------------------------------------------


def band_limited_source(n: int, seed: int, *, top_bin: int | None = None) -> np.ndarray:
    """실제 자극과 같은 성격의 대역제한 신호.

    광대역 백색잡음을 쓰면 안 된다 — 소수점 지연을 선형보간으로 만드는 순간
    고역이 지워지고, 되돌려도 복원되지 않는다. 그러면 테스트가 추적 코드가 아니라
    보간의 대역손실을 재게 된다. 실제 자극은 70-1610Hz(Nyquist 의 1/15)라
    이 문제가 없다.
    """

    rng = np.random.default_rng(seed)
    spectrum = np.zeros(n // 2 + 1, dtype=np.complex128)
    top = top_bin if top_bin is not None else n // 30
    count = top - 4
    spectrum[4:top] = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, count))
    signal = np.fft.irfft(spectrum, n)
    return signal / float(np.max(np.abs(signal)))


def warped_copy(source: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    index = np.arange(source.size, dtype=np.float64)
    return np.interp(index - trajectory, index, source, left=0.0, right=0.0)


def test_track_warp_recovers_a_known_trajectory():
    n = 60000
    source = band_limited_source(n, seed=3)
    index = np.arange(n, dtype=np.float64)
    truth = 400.0 + 60.0 * np.sin(2.0 * np.pi * index / 30000.0)
    observed = warped_copy(source, truth)

    centres, delays, peaks = track_warp(source, observed, window=2048, hop=512)
    good = peaks >= 0.5
    # 추적은 "관측 시각"에서의 지연을 준다 — 참값도 같은 시각에서 봐야 한다.
    expected = np.interp(centres[good] + delays[good], index, truth)
    error = np.abs(delays[good] - expected)
    assert float(np.mean(good)) > 0.8, f"신뢰 가능한 추적점 {np.mean(good):.1%}"
    # 창 하나에 지연 하나를 주므로, 창 안에서 참 궤적이 움직인 만큼은 원리상 못 줄인다.
    # 이 궤적은 2048 샘플 창 안에서 26 샘플 움직인다(실측 하드웨어의 4배 기울기).
    assert float(np.median(error)) < 1.0
    assert float(np.max(error)) < 5.0


def test_dewarp_undoes_a_known_trajectory():
    n = 60000
    source = band_limited_source(n, seed=4)
    index = np.arange(n, dtype=np.float64)
    truth = 300.0 + 40.0 * np.sin(2.0 * np.pi * index / 25000.0)
    observed = warped_copy(source, truth)

    centres, delays, peaks = track_warp(source, observed, window=2048, hop=256)
    corrected = dewarp_recording(observed, centres, delays, peaks, min_peak=0.5)

    core = slice(6000, n - 6000)
    before = float(np.corrcoef(observed[core], source[core])[0, 1])
    after = float(np.corrcoef(corrected[core], source[core])[0, 1])
    assert before < 0.5
    assert after > 0.95, f"보정 전 {before:.3f} → 후 {after:.3f}"


def test_dewarp_rejects_a_trajectory_with_no_trustworthy_points():
    with pytest.raises(ValueError):
        dewarp_recording(
            np.zeros(1000), np.array([10.0, 500.0]), np.array([1.0, 2.0]),
            np.array([0.1, 0.1]), min_peak=0.5,
        )


def test_track_warp_rejects_short_signals():
    with pytest.raises(ValueError):
        track_warp(np.zeros(100), np.zeros(100), window=2048, search=4096)


# ---------------------------------------------------------------------------
# 반복 정렬 — 주파수영역
# ---------------------------------------------------------------------------


def test_estimate_repeat_delay_recovers_a_known_shift():
    freq = np.linspace(150.0, 1200.0, 200)
    rng = np.random.default_rng(11)
    reference = rng.normal(size=freq.size) + 1j * rng.normal(size=freq.size)
    shift = 37.25
    observed = reference * np.exp(-2j * np.pi * freq * shift / FS)
    tau, score = estimate_repeat_delay(freq, observed, reference, sample_rate=FS)
    assert tau == pytest.approx(shift, abs=0.3)
    assert score > 0.99


def test_estimate_repeat_delay_reports_low_score_for_unrelated_data():
    freq = np.linspace(150.0, 1200.0, 200)
    rng = np.random.default_rng(12)
    a = rng.normal(size=freq.size) + 1j * rng.normal(size=freq.size)
    b = rng.normal(size=freq.size) + 1j * rng.normal(size=freq.size)
    _, score = estimate_repeat_delay(freq, a, b, sample_rate=FS)
    assert score < 0.5, "무관한 두 관측이 높은 신뢰도를 받으면 이상치 제거가 무력해진다"


def test_align_repeats_removes_per_repeat_shifts():
    freq = np.linspace(150.0, 1200.0, 200)
    rng = np.random.default_rng(13)
    truth = rng.normal(size=freq.size) + 1j * rng.normal(size=freq.size)
    shifts = np.array([0.0, 12.0, -31.0, 47.5, -8.25])
    stack = np.stack([
        truth * np.exp(-2j * np.pi * freq * s / FS) for s in shifts
    ])
    assert complex_consistency(stack) < 0.9
    aligned, taus, scores = align_repeats(freq, stack, sample_rate=FS)
    assert complex_consistency(aligned) > 0.999
    assert np.allclose(taus, shifts, atol=0.3)
    assert np.all(scores > 0.99)


def test_align_repeats_flags_a_corrupted_repeat():
    """정렬 신뢰도가 낮은 반복은 τ 가 아니라 잡음이다 — 호출자가 버릴 수 있어야 한다."""

    freq = np.linspace(150.0, 1200.0, 200)
    rng = np.random.default_rng(14)
    truth = rng.normal(size=freq.size) + 1j * rng.normal(size=freq.size)
    stack = np.stack([
        truth,
        truth * np.exp(-2j * np.pi * freq * 9.0 / FS),
        rng.normal(size=freq.size) + 1j * rng.normal(size=freq.size),   # 손상
    ])
    _, _, scores = align_repeats(freq, stack, sample_rate=FS)
    assert scores[0] > 0.99 and scores[1] > 0.99
    assert scores[2] < 0.5


def test_complex_consistency_needs_two_repeats():
    with pytest.raises(ValueError):
        complex_consistency(np.ones((1, 10), dtype=complex))


# ---------------------------------------------------------------------------
# 2026-08-05 결함 1 — 오염 반복이 official 로 들어간 사건의 회귀 테스트.
#
# 아래 배열은 전부 실측이다(캡처 20260804_235822_03f4c088). 합성이 아니라 실제로
# 게이트를 통과해버린 값이라, 이 테스트가 깨지면 같은 사건이 다시 일어난다는 뜻이다.
# ---------------------------------------------------------------------------

MEASURED_NOISE_TAU = np.array([
    0.0, 15.41, 20.52, 25.78, 30.36, 34.93, 39.28, 43.53,
    47.78, 52.33, 56.73, 55.65, 36.84, 17.56, -2.39, -21.83,
])
MEASURED_CANCEL_TAU = np.array([
    0.0, 14.21, 19.39, 24.68, 29.27, 33.63, 37.87, 42.07,
    46.65, 50.85, 55.37, 23.54, 4.66, -14.19, -32.65, -50.89,
])


def test_relative_tau_gate_rejects_the_measured_frame_slip():
    """P−S 상대 τ 는 같은 출력 스트림의 불변량이다 — 튀면 버퍼 슬립이다."""

    bad, deviation, centre = relative_tau_outliers(
        MEASURED_NOISE_TAU, MEASURED_CANCEL_TAU, tolerance_samples=3.0
    )
    assert np.array_equal(np.flatnonzero(bad), np.array([11, 12, 13, 14, 15]))
    # 앵커(반복 0)는 τ 가 양쪽 모두 구조적으로 0 이라 중앙값 계산에서 빠진다.
    assert 1.0 < centre < 1.6
    # 정상군 최대 편차와 오염군 최소 편차 사이가 실제로 비어 있어야 임계가 의미 있다.
    assert deviation[np.flatnonzero(~bad)].max() < 2.0
    assert deviation[np.flatnonzero(bad)].min() > 4.0


def test_relative_tau_gate_survives_contamination_majority():
    """오염이 과반이어도 통과시키면 안 된다 — MAD 스케일 임계가 실패하는 지점이다.

    실측 캡처 2건(225546, 225856)에서 MAD 가 1.35/1.95 로 부풀어 MAD 기반 허용치가
    12~17 샘플이 됐고 32 샘플 슬립 블록을 통째로 통과시켰다. 고정 임계는 그렇지 않다.
    """

    relative = np.concatenate([np.full(4, 1.2), np.full(12, 33.0)])
    bad, _, _ = relative_tau_outliers(
        relative, np.zeros_like(relative), tolerance_samples=3.0
    )
    # 중앙값이 오염군 쪽으로 넘어가므로 이번엔 **정상군 4개**가 이탈로 잡힌다.
    # 어느 쪽이든 두 무리를 갈라놓는 것이 목적이고, 통째로 통과시키지 않는 것이 핵심이다.
    assert int(bad.sum()) == 4
    mad = np.median(np.abs(relative[1:] - np.median(relative[1:])))
    assert mad == 0.0 or 3.0 < 3.0 * 1.4826 * mad, "MAD 스케일이었다면 임계가 부풀었다"


def test_relative_tau_gate_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        relative_tau_outliers(np.zeros(5), np.zeros(6), tolerance_samples=3.0)
    with pytest.raises(ValueError):
        relative_tau_outliers(np.zeros(5), np.zeros(5), tolerance_samples=0.0)


def test_timebase_drift_flags_the_measured_warmup_transient():
    """워밍업 4주기(0.5s)로는 스트림이 정상상태에 못 든다 — 실측 τ 궤적이 증거다."""

    # 판정은 두 채널의 평균 궤적으로 한다 — warp 는 공통 성분이라 평균이 잡음을 줄인다.
    common = 0.5 * (MEASURED_NOISE_TAU + MEASURED_CANCEL_TAU)
    drift, median = timebase_drift(common)
    assert median == pytest.approx(4.38, abs=0.02)
    deviation = np.abs(drift - median)
    assert deviation[0] == pytest.approx(10.44, abs=0.02)   # 정상상태 아님
    assert deviation[1] == pytest.approx(5.60, abs=0.02)
    assert deviation[2:10].max() < 1.0                      # 정상 구간
    # 반복 10 부터는 프레임 슬립 전이 — 드리프트 게이트도 독립적으로 이것을 잡는다.
    assert deviation[10] > 2.0
    assert np.all(deviation[[0, 1]] > 2.0)                  # 임계 2.0 으로 기각된다


def test_timebase_drift_needs_three_repeats():
    with pytest.raises(ValueError):
        timebase_drift(np.array([0.0, 1.0]))


def test_align_repeats_anchor_selects_the_reference_repeat():
    freq = np.linspace(150.0, 1200.0, 200)
    rng = np.random.default_rng(21)
    truth = rng.normal(size=freq.size) + 1j * rng.normal(size=freq.size)
    shifts = np.array([0.0, 12.0, -31.0, 47.5, -8.25])
    stack = np.stack([truth * np.exp(-2j * np.pi * freq * s / FS) for s in shifts])

    _, taus, scores = align_repeats(freq, stack, sample_rate=FS, anchor=3)
    # 앵커는 자기 자신과의 상관이 1 이고 τ 가 0 이다.
    assert taus[3] == pytest.approx(0.0, abs=1e-9)
    assert scores[3] == pytest.approx(1.0, abs=1e-9)
    # 나머지 τ 는 앵커 기준으로 재정의된다 — 절대값이 아니라 규약에 달린 양이다.
    assert np.allclose(taus, shifts - shifts[3], atol=0.3)


def test_align_repeats_rejects_anchor_out_of_range():
    freq = np.linspace(150.0, 1200.0, 64)
    stack = np.ones((3, freq.size), dtype=complex)
    for anchor in (-1, 3):
        with pytest.raises(ValueError, match="anchor"):
            align_repeats(freq, stack, sample_rate=FS, anchor=anchor)
