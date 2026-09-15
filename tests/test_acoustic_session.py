"""acoustic 녹음의 관측 감소량을 검증한다. 실제 덕트 성능 시험이 아니다."""

import json
from pathlib import Path

import numpy as np
import pytest

from deep_anc.config import load_runtime_config
from deep_anc.eval.acoustic_session import analyze_acoustic_session


FS = 8000


@pytest.fixture
def session_config(tmp_path):
    cfg = load_runtime_config("configs/runtime_acoustic.yaml")
    cfg["hardware"]["audio"].update(sample_rate=FS, block_size=256, latency="low")
    path = tmp_path / "secondary.npz"
    np.savez(
        path, fir=np.array([0.5], dtype=np.float32), delay_samples=24,
        sample_rate=FS, calibration_block_size=256, calibration_latency="low",
        output_channel="cancel", consistency_band_hz=[150.0, 600.0],
        excitation_band_hz=[80.0, 3000.0], coherence_median=0.96,
    )
    cfg["duct"]["secondary_path"].update(npz=str(path), handoff_extra_samples=256)
    cfg["duct"]["positions_m"].update(reference_mic=0.1, error_mic=1.1)
    cfg["duct"]["duct"]["speed_of_sound_mps"] = 343.0
    return cfg


def _session(*, seconds=17, on_intervals=((3, 13),), scales=(0.5,), frequency=None):
    t = np.arange(seconds * FS, dtype=np.float64) / FS
    base = 0.04 * np.sin(2 * np.pi * (300 if frequency is None else frequency) * t)
    if frequency is None:
        base += 0.04 * np.sin(2 * np.pi * 1500 * t)
    gain = np.zeros(t.size, dtype=np.float32)
    err = base.copy()
    for (start, stop), scale in zip(on_intervals, scales):
        sl = slice(int(start * FS), int(stop * FS))
        gain[sl] = 1.0
        err[sl] *= scale
    return {
        "fs": np.int64(FS), "err": err, "ref": base.copy(),
        "source": np.zeros(t.size, dtype=np.float32),
        "control": 0.005 * np.sin(2 * np.pi * 300 * t) * gain,
        "anc_gain": gain,
    }


def _analyze(tmp_path, cfg, data, **options):
    path = tmp_path / "session.npz"
    np.savez(path, **data)
    return analyze_acoustic_session(path, cfg, "machine", **options)


def _rows(report, band):
    return [row for row in report["metrics"] if row["band"] == band]


def _row(report, band):
    rows = _rows(report, band)
    assert len(rows) == 1, (band, report["metrics"])
    return rows[0]


def test_half_amplitude_unequal_durations_are_observations_not_physical_pass(tmp_path, session_config):
    report = _analyze(tmp_path, session_config, _session())
    assert report["diagnostic_only"] is True
    assert report["performance_claim_allowed"] is False
    assert report["comparison_available"] is True
    assert report["all_cycles_complete"] is True
    for band in ("full", "low_0_1000", "high_1000_nyquist", "trusted"):
        row = _row(report, band)
        assert row["observed_err_reduction_db"] == pytest.approx(6.0206, abs=0.03)
        assert row["median_observed_db"] == pytest.approx(6.0206, abs=0.03)
        assert row["worst10_mean_db"] == pytest.approx(6.0206, abs=0.03)
        assert row["n_on_windows"] >= 4


def test_late_on_amplification_is_not_lost_to_short_off_baseline(tmp_path, session_config):
    data = _session()
    # 이전 segment_stats는 짧은 OFF 길이만큼만 ON을 잘라 뒤의 악화를 버렸다.
    data["err"][6 * FS:13 * FS] = 2.0 * data["ref"][6 * FS:13 * FS]
    row = _row(_analyze(tmp_path, session_config, data), "full")
    assert row["n_on_windows"] >= 4
    assert row["observed_err_reduction_db"] < 0
    assert row["worst_observed_db"] == pytest.approx(-6.0206, abs=0.03)
    assert row["worst10_mean_db"] == pytest.approx(-6.0206, abs=0.03)


def test_each_on_island_keeps_its_own_off_on_off_comparison(tmp_path, session_config):
    data = _session(seconds=23, on_intervals=((3, 9), (12, 18)), scales=(0.5, 2.0))
    report = _analyze(tmp_path, session_config, data)
    assert len(report["cycles"]) == 2
    assert all(cycle["usable"] for cycle in report["cycles"])
    rows = _rows(report, "full")
    assert len(rows) == 2
    assert [row["observed_err_reduction_db"] for row in rows] == pytest.approx([6.0206, -6.0206], abs=0.03)


def test_missing_post_off_makes_the_cycle_unusable(tmp_path, session_config):
    data = _session(seconds=13)
    report = _analyze(tmp_path, session_config, data)
    assert report["comparison_available"] is False
    assert report["all_cycles_complete"] is False
    assert report["cycles"]
    assert not any(cycle["usable"] for cycle in report["cycles"])
    assert all(row["observed_err_reduction_db"] is None for row in report["metrics"])


def test_reference_increase_alone_does_not_create_attenuation(tmp_path, session_config):
    data = _session(scales=(1.0,))
    data["ref"][3 * FS:13 * FS] *= 2.0
    report = _analyze(tmp_path, session_config, data)
    row = _row(report, "full")
    assert row["observed_err_reduction_db"] == pytest.approx(0.0, abs=0.03)
    assert row["ref_on_power"] / row["ref_pre_power"] == pytest.approx(4.0, rel=0.01)
    assert report["performance_claim_allowed"] is False


def test_source_dip_during_on_is_never_certified_as_anc_success(tmp_path, session_config):
    data = _session()
    data["ref"][3 * FS:13 * FS] *= 0.5
    # 외부 음원 자체가 작아진 경우에는 앞뒤 OFF가 같아도 인과 효과를 단정하지 않는다.
    report = _analyze(tmp_path, session_config, data)
    row = _row(report, "full")
    assert row["observed_err_reduction_db"] == pytest.approx(6.0206, abs=0.03)
    assert row["ref_on_power"] < row["ref_pre_power"]
    assert report["diagnostic_only"] and not report["performance_claim_allowed"]


def test_lower_post_off_baseline_prevents_initial_baseline_only_gain(tmp_path, session_config):
    data = _session()
    data["err"][13 * FS:] *= 0.5
    row = _row(_analyze(tmp_path, session_config, data), "full")
    assert row["off_post_power"] < row["off_pre_power"]
    assert row["observed_err_reduction_db"] == pytest.approx(0.0, abs=0.03)


def test_long_secondary_tail_is_removed_from_post_off_without_extra_handoff(tmp_path, session_config):
    path = Path(session_config["duct"]["secondary_path"]["npz"])
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    fir = np.zeros(129, dtype=np.float32)
    fir[-1] = 0.5
    values.update(delay_samples=2 * FS, fir=fir)
    np.savez(path, **values)
    data = _session(seconds=22, on_intervals=((5, 15),))
    tail = 2 * FS + len(fir) - 1
    # 출력은 이미 OFF지만 측정 ERR에는 앞선 상쇄음이 아직 도착하는 구간이다.
    data["err"][15 * FS:15 * FS + tail] *= 0.5
    report = _analyze(tmp_path, session_config, data)
    assert report["selection"]["secondary_tail_samples"] == tail
    assert report["selection"]["handoff_added_to_recorded_output"] is False
    assert report["cycles"][0]["intervals"]["off_post"]["start"] >= 15 * FS + tail
    assert _row(report, "full")["observed_err_reduction_db"] == pytest.approx(6.0206, abs=0.03)


def test_incomplete_last_cycle_is_retained_beside_complete_first_cycle(tmp_path, session_config):
    data = _session(seconds=18, on_intervals=((3, 9), (12, 18)), scales=(0.5, 2.0))
    report = _analyze(tmp_path, session_config, data)
    assert len(report["cycles"]) == 2
    assert report["cycles"][0]["usable"] is True
    assert report["cycles"][1]["usable"] is False
    assert report["cycles"][1]["reason"]
    assert report["comparison_available"] is True
    assert report["all_cycles_complete"] is False
    assert len(_rows(report, "full")) == 1


@pytest.mark.parametrize("level", [0.0, 1e-6])
def test_unexcited_or_below_floor_baseline_is_null_not_zero_db(tmp_path, session_config, level):
    data = _session()
    data["err"] *= level
    row = _row(_analyze(tmp_path, session_config, data), "full")
    for key in ("observed_err_reduction_db", "median_observed_db", "p10_observed_db", "worst_observed_db", "worst10_mean_db"):
        assert row[key] is None


def test_zero_on_error_is_not_an_exact_infinite_attenuation(tmp_path, session_config):
    data = _session(scales=(0.0,))
    row = _row(_analyze(tmp_path, session_config, data), "full")
    assert row["observed_err_reduction_db"] is None


def test_extreme_finite_power_ratio_keeps_entire_report_json_finite(tmp_path, session_config):
    """실기에서 의미 없는 극단값으로 나눗셈 overflow 방지만 검증한다."""
    data = _session(scales=(1.0,))
    # 각 톤의 OFF 진폭 1e150 / ON 진폭 1e-5 → 전력비 1e310.
    # 신호 제곱과 FFT 전력은 유한하고 ON 전력도 기본 floor 1e-12보다 크다.
    # 이 조건에서 전력을 먼저 나누면 float64 범위를 넘지만 로그 차이는 유한하다.
    data["err"] *= 1e150 / 0.04
    on = data["anc_gain"] >= 0.999
    data["err"][on] = data["ref"][on] * (1e-5 / 0.04)
    report = _analyze(tmp_path, session_config, data)
    row = _row(report, "full")
    assert row["on_power"] > report["selection"]["power_floor"]
    assert row["off_pre_power"] > np.finfo(np.float64).max * row["on_power"]
    assert row["observed_err_reduction_db"] == pytest.approx(3100.0, abs=0.03)
    assert report["diagnostic_only"] and not report["performance_claim_allowed"]
    json.dumps(report, allow_nan=False)


def test_emergent_high_band_energy_is_not_hidden_by_missing_baseline_source(tmp_path, session_config):
    data = _session(scales=(1.0,), frequency=300)
    t = np.arange(data["err"].size) / FS
    added = 0.04 * np.sin(2 * np.pi * 1500 * t) * data["anc_gain"]
    data["err"] += added
    data["control"] = 2.0 * added
    report = _analyze(tmp_path, session_config, data)
    high = _row(report, "high_1000_nyquist")
    assert high["emergent_on_energy"] is True
    assert high["observed_err_reduction_db"] is None
    assert high["on_power"] > high["off_pre_power"]
    assert _row(report, "full")["observed_err_reduction_db"] < -2.9


def test_trusted_requires_whole_octave_inside_measured_band(tmp_path, session_config):
    report = _analyze(tmp_path, session_config, _session())
    assert _row(report, "octave_250")["trusted"] is True
    row = _row(report, "octave_500")
    assert row["high_hz"] > 600
    assert row["trusted"] is False
    assert _row(report, "high_1000_nyquist")["trusted"] is False


def test_1khz_boundary_is_counted_once_across_low_and_high(tmp_path, session_config):
    report = _analyze(tmp_path, session_config, _session(frequency=1000))
    full = _row(report, "full")
    low = _row(report, "low_0_1000")
    high = _row(report, "high_1000_nyquist")
    # 고정창 FFT/창함수 선택과 무관하게 경계의 에너지가 중복되거나 누락되면 안 된다.
    for key in ("off_pre_power", "on_power", "off_post_power"):
        assert low[key] + high[key] == pytest.approx(full[key], rel=1e-6, abs=1e-12)
        assert high[key] > low[key]


def test_missing_consistency_band_does_not_promote_excitation_band(tmp_path, session_config):
    path = Path(session_config["duct"]["secondary_path"]["npz"])
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files if key != "consistency_band_hz"}
    np.savez(path, **values)
    report = _analyze(tmp_path, session_config, _session())
    assert all(row["trusted"] is False for row in report["metrics"])


@pytest.mark.parametrize("bad_field", ["err", "ref", "source", "control", "anc_gain"])
def test_nonfinite_samples_are_rejected_even_outside_analysis_windows(tmp_path, session_config, bad_field):
    data = _session()
    data[bad_field][0] = np.nan
    with pytest.raises(ValueError):
        _analyze(tmp_path, session_config, data)


@pytest.mark.parametrize("bad_field", ["ref", "control", "anc_gain"])
def test_different_array_lengths_are_rejected(tmp_path, session_config, bad_field):
    data = _session()
    data[bad_field] = data[bad_field][:-1]
    with pytest.raises(ValueError):
        _analyze(tmp_path, session_config, data)


def test_multichannel_shape_is_not_silently_flattened(tmp_path, session_config):
    data = _session()
    data["err"] = data["err"].reshape(2, -1)
    with pytest.raises(ValueError):
        _analyze(tmp_path, session_config, data)


def test_recording_sample_rate_must_match_config_and_secondary(tmp_path, session_config):
    data = _session()
    data["fs"] = np.int64(16000)
    with pytest.raises(ValueError, match="sample_rate|샘플|fs"):
        _analyze(tmp_path, session_config, data)


@pytest.mark.parametrize("key", ["fs", "err", "ref", "source", "control", "anc_gain"])
def test_required_recording_fields_are_not_invented(tmp_path, session_config, key):
    data = _session()
    data.pop(key)
    with pytest.raises((ValueError, KeyError)):
        _analyze(tmp_path, session_config, data)
