"""SFANC FIR 설계의 독립 인과 합성 oracle. 실기 감쇠 검증이 아니다."""

import json

import numpy as np
import pytest

from deep_anc.baselines import sfanc_design as design


def direct_causal(values, fir, delay=0):
    """라이브러리 컨볼루션을 쓰지 않는 표본별 독립 기준."""
    result = np.zeros(len(values), dtype=np.float64)
    for n in range(len(values)):
        for k in range(len(fir)):
            index = n - delay - k
            if 0 <= index < len(values):
                result[n] += fir[k] * values[index]
    return result


def direct_residual(x, d, s, w, delay=0):
    y = direct_causal(x, w)
    return y, d + direct_causal(y, s, delay)


def test_scores_match_independent_loops_and_preserve_inputs():
    rng = np.random.default_rng(19)
    x, d = rng.normal(size=(2, 91))
    s = np.array([0.0, 0.0, -0.6, 0.12])
    weights = rng.normal(scale=0.08, size=(3, 5))
    copies = [a.copy() for a in (x, d, s, weights)]
    result = design.score_control_filters(x, d, s, weights, secondary_delay_samples=7,
                                          warmup_samples=16, effort_penalty=0.07)
    assert result["residual"].shape == result["control"].shape == (3, 91)
    for i, w in enumerate(weights):
        y, e = direct_residual(x, d, s, w, 7)
        np.testing.assert_allclose(result["control"][i], y, atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(result["residual"][i], e, atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(result["secondary_output"][i], e - d, atol=1e-14)
        assert result["residual_mse"][i] == pytest.approx(np.mean(e[16:]**2), rel=1e-13)
        assert result["control_mse"][i] == pytest.approx(np.mean(y[16:]**2), rel=1e-13)
        assert result["objective"][i] == pytest.approx(np.mean(e[16:]**2) + .07*np.mean(y[16:]**2))
    assert result["baseline_mse"] == pytest.approx(np.mean(d[16:]**2))
    assert result["evaluated_samples"] == 75
    for original, before in zip((x, d, s, weights), copies):
        np.testing.assert_array_equal(original, before)


def test_secondary_leading_delay_and_explicit_delay_are_each_applied_once():
    x = np.zeros(64)
    x[4] = 1
    s, w = np.array([0., 0., -.5, .125]), np.array([0., .3])
    result = design.score_control_filters(x, np.zeros(64), s, w, secondary_delay_samples=6)
    assert result["residual"].shape == (1, 64)
    assert np.argmax(np.abs(result["residual"][0])) == 4 + 1 + 2 + 6
    np.testing.assert_array_equal(result["secondary_output"][0, :13], 0)
    np.testing.assert_allclose(result["residual"][0], direct_causal(direct_causal(x, w), s, 6), atol=1e-15)


@pytest.mark.parametrize("secondary_gain", [-.7, .7])
def test_closed_form_scalar_solution_fixes_sign_and_mean_square_scaling(secondary_gain):
    x = np.random.default_rng(4).normal(size=500)
    d = .25*x
    ridge, effort = .02, .03
    fit = design.fit_control_fir(x, d, [secondary_gain], control_length=1,
                                 warmup_samples=20, regularization=ridge, effort_penalty=effort)
    power = np.mean(x[20:]**2)
    expected = -secondary_gain*.25*power / ((secondary_gain**2 + effort)*power + ridge)
    assert fit.coefficients[0] == pytest.approx(expected, abs=1e-13)
    assert np.sign(fit.coefficients[0]) == -np.sign(secondary_gain)
    assert fit.diagnostics["normal_equation_gradient_norm"] < 1e-12
    assert fit.diagnostics["objective"] == pytest.approx(
        fit.diagnostics["residual_mse"] + effort*fit.diagnostics["control_mse"]
        + ridge*np.dot(fit.coefficients, fit.coefficients), abs=1e-14)
    assert fit.coefficients.dtype == np.float64 and not fit.coefficients.flags.writeable
    json.dumps(fit.diagnostics, allow_nan=False)


def test_known_fir_is_recovered_and_cancels_unseen_white_reference():
    rng = np.random.default_rng(88)
    x = rng.normal(scale=.2, size=1400)
    secondary = np.array([0., 0., -.5, .09, -.02])
    expected = np.array([.24, -.06, .015, .007])
    disturbance = -direct_causal(direct_causal(x, expected), secondary, 9)
    saved = secondary.copy()
    fit = design.fit_control_fir(x, disturbance, secondary, control_length=4,
                                 secondary_delay_samples=9, warmup_samples=17,
                                 regularization=1e-11)
    np.testing.assert_allclose(fit.coefficients, expected, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(secondary, saved)
    unseen = rng.normal(scale=.2, size=1900)
    d = -direct_causal(direct_causal(unseen, expected), secondary, 9)
    score = design.score_control_filters(unseen, d, secondary, fit.coefficients,
                                         secondary_delay_samples=9, warmup_samples=17)
    assert score["residual_mse"][0] < score["baseline_mse"] * 1e-12


def test_finite_difference_gradient_matches_independent_causal_design():
    rng = np.random.default_rng(23)
    x, d = rng.normal(scale=.2, size=(2, 80))
    s = np.array([0., -.7, .13])
    w = np.array([.2, -.05, .04])
    delay, warmup, ridge, effort = 5, 12, .003, .02
    control_columns = np.column_stack([direct_causal(x, row) for row in np.eye(len(w))])
    columns = np.column_stack([direct_causal(control_columns[:, i], s, delay) for i in range(len(w))])
    y, e = direct_residual(x, d, s, w, delay)
    gradient = 2*(columns[warmup:].T @ e[warmup:] / (len(x)-warmup)
                  + effort*control_columns[warmup:].T @ y[warmup:] / (len(x)-warmup) + ridge*w)

    def cost(weights):
        scored = design.score_control_filters(x, d, s, weights, secondary_delay_samples=delay,
                                              warmup_samples=warmup, effort_penalty=effort)
        return scored["objective"][0] + ridge*np.dot(weights, weights)

    epsilon = 1e-6
    finite_difference = np.array([(cost(w + epsilon*v) - cost(w - epsilon*v))/(2*epsilon)
                                  for v in np.eye(len(w))])
    np.testing.assert_allclose(finite_difference, gradient, atol=1e-10, rtol=1e-7)
    fit = design.fit_control_fir(x, d, s, control_length=3, secondary_delay_samples=delay,
                                 warmup_samples=warmup, regularization=ridge, effort_penalty=effort)
    assert fit.diagnostics["normal_equation_gradient_norm"] < 1e-12


def test_warmup_is_removed_after_plant_and_keeps_earlier_control_tail():
    x = np.zeros(32)
    x[3] = 1.0
    score = design.score_control_filters(x, np.zeros(32), [.5, .2], [1.0],
                                         secondary_delay_samples=8, warmup_samples=10)
    assert score["residual_mse"][0] == pytest.approx((.5**2 + .2**2)/22)
    assert score["control_mse"][0] == 0
    # warmup 이전 reference를 지우면 위 잔차를 잘못 잃어버린다.
    erased = x.copy()
    erased[:10] = 0
    wrong = design.score_control_filters(erased, np.zeros(32), [.5, .2], [1.0],
                                         secondary_delay_samples=8, warmup_samples=10)
    assert wrong["residual_mse"][0] == 0


def test_future_reference_and_disturbance_cannot_change_earlier_score_waveforms():
    rng = np.random.default_rng(36)
    x, d = rng.normal(size=(2, 120))
    changed_x, changed_d = x.copy(), d.copy()
    changed_x[71:] = rng.normal(size=49)*3
    changed_d[71:] = rng.normal(size=49)*4
    args = ([.3, -.1], [[.2, -.04], [-.1, .06]])
    a = design.score_control_filters(x, d, *args, secondary_delay_samples=5)
    b = design.score_control_filters(changed_x, changed_d, *args, secondary_delay_samples=5)
    for key in ("control", "secondary_output", "residual"):
        np.testing.assert_allclose(a[key][:, :71], b[key][:, :71], atol=2e-14, rtol=1e-12)


def test_unpredictable_white_reference_with_insufficient_preview_is_not_magically_cancelled():
    rng = np.random.default_rng(302)
    x = rng.normal(scale=.1, size=6000)
    fit = design.fit_control_fir(x, .2*x, [-.5], control_length=12,
                                 secondary_delay_samples=40, warmup_samples=64,
                                 regularization=1e-6)
    unseen = rng.normal(scale=.1, size=10000)
    score = design.score_control_filters(unseen, .2*unseen, [-.5], fit.coefficients,
                                         secondary_delay_samples=40, warmup_samples=64)
    assert score["residual_mse"][0] >= .95*score["baseline_mse"]
    assert fit.diagnostics["secondary_delay_samples"] == 40


@pytest.mark.parametrize("changes", [
    {"control_length": 0}, {"control_length": True}, {"control_length": 1.2},
    {"secondary_delay_samples": -1}, {"secondary_delay_samples": True},
    {"secondary_delay_samples": 64}, {"warmup_samples": 64}, {"warmup_samples": -1},
    {"warmup_samples": 63}, {"regularization": 0}, {"regularization": -1},
    {"regularization": np.nan}, {"regularization": True}, {"effort_penalty": -.1},
    {"effort_penalty": np.inf}, {"max_coefficient_norm": 0},
])
def test_invalid_fit_parameters_fail_before_solving(changes):
    options = {"control_length": 3, "warmup_samples": 4} | changes
    with pytest.raises(ValueError):
        design.fit_control_fir(np.ones(64), np.ones(64), [-.5], **options)


@pytest.mark.parametrize("x,d,s", [
    ([], [], [1.]), (np.zeros(64), np.ones(64), [1.]),
    (np.ones(64), np.ones(63), [1.]), (np.ones((8, 8)), np.ones(64), [1.]),
    (np.full(64, np.nan), np.ones(64), [1.]), (np.ones(64), np.ones(64), [0., 0.]),
    (np.ones(64), np.ones(64), [np.inf]), (np.ones(64), np.ones(64), [[1.]]),
    (np.ones(64, dtype=complex), np.ones(64), [1.]),
    (np.ones(64), np.ones(64), [*np.zeros(80), 1.]),
])
def test_invalid_short_or_uninformative_records_fail(x, d, s):
    with pytest.raises(ValueError):
        design.fit_control_fir(x, d, s, control_length=3, warmup_samples=4)


def test_norm_violation_is_rejected_without_clipping_or_mutation():
    x = np.linspace(-1, 1, 120)
    s = np.array([-.5])
    before = s.copy()
    fit = design.fit_control_fir(x, x, s, control_length=1, warmup_samples=0, regularization=1e-10)
    assert fit.coefficients[0] > 1.99
    with pytest.raises(ValueError, match="norm"):
        design.fit_control_fir(x, x, s, control_length=1, warmup_samples=0,
                               regularization=1e-10, max_coefficient_norm=.1)
    np.testing.assert_array_equal(s, before)


def test_resource_limits_are_checked_before_large_design_allocation(monkeypatch):
    monkeypatch.setattr(design, "MAX_DESIGN_ELEMENTS", 100)
    with pytest.raises(ValueError, match="크기"):
        design.fit_control_fir(np.ones(64), np.ones(64), [1], control_length=3, warmup_samples=0)
    monkeypatch.setattr(design, "MAX_SCORE_ELEMENTS", 100)
    with pytest.raises(ValueError, match="크기"):
        design.score_control_filters(np.ones(64), np.ones(64), [1], [[1], [1]])


def test_silence_and_delayed_outside_record_scoring_are_finite():
    for delay in (0, 100):
        result = design.score_control_filters(np.zeros(64), np.zeros(64), [1.], [[0.], [1.]],
                                              secondary_delay_samples=delay)
        np.testing.assert_array_equal(result["residual_mse"], 0)
        np.testing.assert_array_equal(result["control_mse"], 0)
        assert result["baseline_mse"] == 0


@pytest.mark.parametrize("weights", [[], [np.nan], [np.inf], [[1], [1, 2]], [1+1j], [True]])
def test_invalid_coefficients_are_rejected(weights):
    with pytest.raises(ValueError):
        design.score_control_filters(np.ones(16), np.ones(16), [1.], weights)
