"""학습 없는 합성 FFT/구간/세션 통계 검증."""

import copy
import json

import numpy as np
import pytest

from deep_anc.eval.high_frequency_metrics import (
    band_powers, evaluate_trace, metric_row, paired_session_comparison,
)


def tone(frequency=2000, count=256, amplitude=.1):
    return amplitude * np.cos(2 * np.pi * frequency * np.arange(count) / 16000)


def trace(method="dl", factor=.5, session="s1", source="x1", d=None, context="a" * 64, **kwargs):
    d = tone() if d is None else d
    return evaluate_trace(d, factor * d, control=np.zeros_like(d), raw_control=np.zeros_like(d),
        regions={"onset": [[0, 64]], "transition": [[64, 128]], "steady": [[128, len(d)]]},
        session_id=session, source_id=source, source_kind="noise", method=method,
        sample_rate=16000, control_limit=.2, power_floor=1e-12,
        comparison_context_sha256=context, **kwargs)


def paired(records, **kwargs):
    options = dict(candidate_method="dl", baseline_method="base", region="steady", band="high",
                   confidence=.95, bootstrap_replicates=200, seed=71)
    options.update(kwargs)
    return paired_session_comparison(records, **options)


@pytest.mark.parametrize("frequency,band,power", [
    (0, "low", 1.), (500, "low", .5), (1000, "priority", .5),
    (1600, "tail", .5), (8000, "tail", 1.),
])
def test_exact_boundary_tones_and_nyquist(frequency, band, power):
    values = tone(frequency, 1600, 1.)
    result = band_powers(values, sample_rate=16000)
    assert result[band] == pytest.approx(power, abs=1e-12)
    assert result["full"] == pytest.approx(power, abs=1e-12)
    assert result["low"] + result["high"] == pytest.approx(result["full"], abs=1e-12)
    assert result["priority"] + result["tail"] == pytest.approx(result["high"], abs=1e-12)


@pytest.mark.parametrize("count", [1, 31, 32, 511])
def test_parseval_for_odd_and_even_lengths(count):
    values = np.random.default_rng(3).normal(size=count)
    result = band_powers(values, sample_rate=16000)
    assert result["low"] + result["high"] == pytest.approx(np.mean(values**2), rel=1e-13)


def test_attenuation_null_amplification_and_clip_are_preserved():
    record = trace()
    assert metric_row(record, "steady", "high")["attenuation_db"] == pytest.approx(20 * np.log10(2))
    amplified = trace(factor=2.)
    row = metric_row(amplified, "steady", "high")
    assert row["amplification"] and row["attenuation_db"] < 0
    zero = trace(d=np.zeros(256))
    assert metric_row(zero, "full", "high")["attenuation_db"] is None
    d = np.zeros(256)
    noisy = evaluate_trace(d, tone(), control=np.full(256, .3), raw_control=np.full(256, .5),
        regions={"onset": [], "transition": [], "steady": [[0, 256]]},
        session_id="s", source_id="x", source_kind="silence", method="dl",
        sample_rate=16000, control_limit=.2, power_floor=1e-12)
    row = metric_row(noisy, "steady", "high")
    assert row["attenuation_db"] is None and row["emergent_error_energy"]
    assert row["limited_samples"] == row["post_limit_exceeded_samples"] == 256
    assert noisy["fairness_unverified"]
    assert metric_row(noisy, "onset", "high")["samples"] == 0
    assert metric_row(noisy, "onset", "high")["attenuation_db"] is None
    json.dumps(noisy, allow_nan=False)


def test_disjoint_phase_intervals_are_not_concatenated_for_fft():
    d = np.r_[np.ones(16), -np.ones(16)]
    result = evaluate_trace(d, d * .5, control=np.zeros(32), raw_control=np.zeros(32),
        regions={"onset": [], "transition": [], "steady": [[0, 16], [16, 32]]},
        session_id="s", source_id="x", source_kind="noise", method="dl",
        sample_rate=16000, control_limit=.2, power_floor=1e-12)
    assert metric_row(result, "steady", "high")["baseline_power"] == 0
    assert metric_row(result, "full", "high")["baseline_power"] > 0
    assert metric_row(result, "steady", "low")["baseline_power"] == 1


def test_session_is_independent_unit_not_number_of_crops():
    records = []
    for source in ("a", "b", "c", "d"):
        records.extend([trace(session="many", source=source), trace("base", 1., session="many", source=source)])
    records.extend([trace(factor=1., session="single"), trace("base", 1., session="single")])
    result = paired(records)
    assert result["session_count"] == 2
    assert result["mean_delta_db"] == pytest.approx(10 * np.log10(2))
    assert result["bootstrap_ci_db"] is not None
    assert result == paired(list(reversed(records)))
    assert not result["fairness_unverified"]
    assert not result["superiority_proven"]


def test_session_energy_aggregation_precedes_db_conversion():
    records = [trace(source="quiet", d=tone(amplitude=.01)), trace("base", 1., source="quiet", d=tone(amplitude=.01)),
               trace(factor=1., source="loud", d=tone(amplitude=.1)),
               trace("base", 1., source="loud", d=tone(amplitude=.1))]
    result = paired(records)
    assert result["mean_delta_db"] == pytest.approx(10 * np.log10((.01**2 + .1**2) / (.005**2 + .1**2)))
    assert result["bootstrap_ci_db"] is None  # 세션 하나를 crop 여러 개로 부풀리지 않음.


def test_missing_pair_and_null_session_prevent_global_claim():
    missing = paired([trace(), trace("base", 1.), trace(session="missing")])
    assert not missing["paired_complete"]
    assert missing["mean_delta_db"] is missing["bootstrap_ci_db"] is None
    assert any(row["status"] == "missing_method" for row in missing["pairs"])
    silent = paired([trace(), trace("base", 1.),
                     trace(session="silent", d=np.zeros(256)), trace("base", 1., session="silent", d=np.zeros(256))])
    assert silent["mean_delta_db"] is None
    assert len(silent["sessions"]) == 2


@pytest.mark.parametrize("field,value", [
    ("disturbance_sha256_f64le", "b" * 64), ("comparison_context_sha256", "b" * 64),
    ("sample_rate", 48000), ("control_limit", .3), ("power_floor", 1e-10),
    ("source_kind", "music"), ("samples", 511),
])
def test_pair_mismatch_rejected(field, value):
    records = [trace(), trace("base", 1.)]
    records[1][field] = value
    with pytest.raises(ValueError):
        paired(records)


def test_missing_context_is_explicit_and_duplicate_pair_rejected():
    records = [trace(context=None), trace("base", 1., context=None)]
    assert paired(records)["fairness_unverified"]
    with pytest.raises(ValueError, match="중복"):
        paired(records + [copy.deepcopy(records[0])])


@pytest.mark.parametrize("regions", [
    {}, {"onset": [[0, 65]], "transition": [[64, 128]], "steady": []},
    {"onset": [[0, 0]], "transition": [], "steady": []},
    {"onset": [[0, 257]], "transition": [], "steady": []},
    {"onset": [[False, 64]], "transition": [], "steady": []},
])
def test_invalid_regions_rejected(regions):
    with pytest.raises(ValueError):
        evaluate_trace(tone(), tone(), control=np.zeros(256), raw_control=np.zeros(256),
            regions=regions, session_id="s", source_id="x", source_kind="noise", method="dl",
            sample_rate=16000, control_limit=.2, power_floor=1e-12)


@pytest.mark.parametrize("value", [[], [[1.]], [np.nan], [np.inf], [1j], [True]])
def test_invalid_wave_rejected(value):
    with pytest.raises(ValueError):
        band_powers(value, sample_rate=16000)


@pytest.mark.parametrize("options", [{"confidence": 1.}, {"confidence": 0.},
    {"bootstrap_replicates": 0}, {"seed": True}, {"candidate_method": "base"}])
def test_invalid_statistics_settings(options):
    with pytest.raises(ValueError):
        paired([trace(), trace("base", 1.)], **options)
