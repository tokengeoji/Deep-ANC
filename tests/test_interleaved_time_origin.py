"""무오디오 합성 경로의 FIR+delay 시간 원점과 gain/극성 왕복 회귀."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "measurement_time_origin", ROOT / "scripts/data/measure_paths_interleaved.py"
)
measurement = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measurement)


def analyse(*, drive="noise", pre_roll=256, delay=1400.0, gain=.25, low=60.0):
    probe = measurement.build_interleaved_probe(
        sample_rate=48000, period_seconds=.125, band_hz=(low, 1650.0), amplitude=.02
    )
    frequencies = np.fft.rfftfreq(probe.period_samples, 1 / probe.sample_rate)
    truth = gain * np.exp(-2j * np.pi * frequencies * delay / probe.sample_rate)
    output = np.asarray(probe.noise_signal, dtype=np.float64) + probe.cancel_signal
    one_period = np.fft.irfft(np.fft.rfft(output) * truth, probe.period_samples)
    model = measurement.analyse_channel(
        err=np.tile(one_period, 3), probe=probe, drive=drive,
        period_starts=[0, probe.period_samples, 2 * probe.period_samples],
        fir_length=probe.period_samples // 2, pre_roll=pre_roll, max_delay_samples=2000,
        fit_band_hz=(150., 1200.), consistency_band_hz=(150., 600.),
        min_alignment_score=.5, min_kept_repeats=2,
    )
    return probe, model, truth[probe.bins_for(drive)]


@pytest.mark.parametrize("drive", ["noise", "cancel"])
@pytest.mark.parametrize("pre_roll", [0, 128, 256])
@pytest.mark.parametrize("gain,delay", [(.25, 1400.0), (-.4, 1400.25)])
@pytest.mark.parametrize("low", [60.0, 68.0])
def test_saved_delay_and_fir_reconstruct_original_complex_transfer(drive, pre_roll, gain, delay, low):
    # 두 first-bin parity 모두 검사: 어느 채널이 홀수 빈인지 이름에 의존하면 안 된다.
    probe, model, truth = analyse(drive=drive, pre_roll=pre_roll, gain=gain, delay=delay, low=low)
    saved_delay = model["delay_samples"]
    frequencies = model["frequencies_hz"]
    omega = 2 * np.pi * frequencies / probe.sample_rate
    reconstructed = np.exp(
        -1j * np.outer(omega, np.arange(model["fir"].size) + saved_delay)
    ) @ model["fir"]
    np.testing.assert_allclose(reconstructed, truth, rtol=2e-6, atol=1e-8)
    assert saved_delay + pre_roll == model["bulk_delay_samples"]
    assert model["delay_fractional"] == pytest.approx(delay, abs=.01)
    assert model["time_origin_convention"] == "bulk_delay_minus_pre_roll_v1"
    assert np.argmax(np.abs(model["fir"])) + saved_delay == round(delay)


def test_official_arrays_record_time_origin_without_changing_fir(tmp_path):
    probe, model, _ = analyse(drive="cancel")
    arrays = measurement._official_arrays(
        model=model, relative_delay_spread=0, max_delay_jitter_samples=48, fs=48000,
        consistency=1., band_hz=(60., 1650.), amplitude=.02, block_size=256,
        latency="high", output_channel="right", repeats=3, xrun_count=0,
        capture_id="synthetic-only", probe=probe, drive="cancel",
        snr_db=np.full(probe.cancel_bins.size, 100.), period_seconds=.125,
    )
    destination = tmp_path / "synthetic.npz"
    np.savez(destination, **arrays)
    with np.load(destination, allow_pickle=False) as restored:
        np.testing.assert_array_equal(restored["fir"], model["fir"])
        assert restored["delay_samples"].item() == 1144
        assert restored["bulk_delay_samples"].item() == 1400
        assert restored["pre_roll_samples"].item() == 256
        assert restored["time_origin_convention"].item() == "bulk_delay_minus_pre_roll_v1"


def test_pre_roll_larger_than_bulk_delay_does_not_save_negative_delay():
    with pytest.raises(ValueError, match="비인과"):
        analyse(pre_roll=256, delay=100.)


@pytest.mark.parametrize("pre_roll", [-1, True, 1.5])
def test_invalid_pre_roll_is_not_silently_coerced(pre_roll):
    with pytest.raises(ValueError, match="정수"):
        analyse(pre_roll=pre_roll)


def test_live_entrypoint_without_confirmation_does_not_open_audio(monkeypatch):
    monkeypatch.setattr(measurement, "load_yaml", lambda *_: pytest.fail("하드웨어 검사 금지"))
    assert measurement.main([]) == 2
