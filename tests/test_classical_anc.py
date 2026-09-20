"""최적화/모델학습 없는 작은 합성 plain FxLMS/FxNLMS 수치 회귀."""

import hashlib

import numpy as np
import pytest
from scipy import signal

from deep_anc.eval.classical_anc import run_classical_anc


def test_analytic_scalar_distinguishes_plain_and_normalized():
    # x=2, S=3 -> xf=6, e=d=4, mu=.1. plain w=-2.4, NLMS w=-2.4/37.
    args = ([2.0], [4.0], [3.0])
    plain = run_classical_anc(*args, algorithm="fxlms", control_length=1,
                              mu=.1, normalization_epsilon=1.0)
    normalized = run_classical_anc(*args, algorithm="fxnlms", control_length=1,
                                   mu=.1, normalization_epsilon=1.0)
    np.testing.assert_allclose(plain["final_weights"], [-2.4], rtol=1e-15)
    np.testing.assert_allclose(normalized["final_weights"], [-2.4 / 37], rtol=1e-15)
    for result in (plain, normalized):
        np.testing.assert_array_equal(result["control"], [0.0])
        np.testing.assert_array_equal(result["residual"], [4.0])
    assert not plain["settings"]["normalized"]
    assert normalized["settings"]["normalized"]
    assert plain["settings"]["normalization_epsilon"] is None


@pytest.mark.parametrize("algorithm", ["fxlms", "fxnlms"])
def test_analytic_block_sum_not_mean_or_sequential_update(algorithm):
    # Xf=[6,12], e=[4,5], gradient=84; normalizer=180+1.
    result = run_classical_anc([2., 4.], [4., 5.], [3.], algorithm=algorithm,
                               control_length=1, block_samples=2, mu=.01,
                               normalization_epsilon=1.)
    expected = -.84 / (181 if algorithm == "fxnlms" else 1)
    np.testing.assert_allclose(result["final_weights"], [expected], rtol=1e-15)
    np.testing.assert_array_equal(result["control"], 0)
    assert result["settings"]["update_convention"] == "block_sum_gradient"


@pytest.mark.parametrize("algorithm", ["fxlms", "fxnlms"])
@pytest.mark.parametrize("delay", [0, 3, 40])
@pytest.mark.parametrize("block", [1, 13])
def test_delays_plant_and_filtered_x_match_independent_recurrence(algorithm, delay, block):
    rng = np.random.default_rng(550)
    x = rng.normal(0, .08, 119)
    d = signal.lfilter([0.] * 8 + [.04, -.02], [1.], x)
    s, estimate = np.array([0., -.4, .2]), np.array([.02, -.37, .19])
    estimate_delay = delay + 2
    length, mu, eps = 4, .001, .003
    result = run_classical_anc(x, d, s, algorithm=algorithm, additional_delay_samples=delay,
                               control_length=length, block_samples=block, mu=mu,
                               normalization_epsilon=eps, secondary_estimate=estimate,
                               estimate_delay_samples=estimate_delay)
    expected_u, weights = np.zeros_like(x), np.zeros(length)
    xf = np.convolve(np.pad(x, (estimate_delay, 0)), estimate)[:x.size]
    xp, xfp = np.pad(x, (length - 1, 0)), np.pad(xf, (length - 1, 0))
    for start in range(0, x.size, block):
        end = min(start + block, x.size)
        for n in range(start, end):
            expected_u[n] = np.dot(weights, xp[n:n + length][::-1])
        error = d + np.convolve(np.pad(expected_u, (delay, 0)), s)[:x.size]
        gradient, power = np.zeros(length), 0.
        for n in range(start, end):
            taps = xfp[n:n + length][::-1]
            gradient += error[n] * taps
            power += np.dot(taps, taps)
        weights -= mu * gradient / (power + eps if algorithm == "fxnlms" else 1.)
    np.testing.assert_allclose(result["control"], expected_u, atol=1e-16, rtol=1e-12)
    np.testing.assert_allclose(result["final_weights"], weights, atol=1e-16, rtol=1e-12)
    expected_e = d + np.convolve(np.pad(result["control"], (delay, 0)), s)[:x.size]
    np.testing.assert_allclose(result["residual"], expected_e, atol=1e-16, rtol=1e-12)
    assert result["settings"]["estimate_source"] == "explicit_argument"
    assert not result["settings"]["estimate_matches_plant"]
    assert not result["settings"]["estimate_delay_matches_plant"]


@pytest.mark.parametrize("algorithm", ["fxlms", "fxnlms"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_control_is_actual_dac_command_with_no_extra_inversion(algorithm, sign):
    result = run_classical_anc(np.full(16, .5), np.full(16, .05), [sign],
                               algorithm=algorithm, control_length=1, mu=.1)
    assert result["control"][0] == 0
    assert np.sign(result["control"][1]) == -sign
    assert abs(result["residual"][-1]) < .05
    np.testing.assert_allclose(result["residual"], .05 + sign * result["control"])


@pytest.mark.parametrize("algorithm", ["fxlms", "fxnlms"])
def test_hard_clip_freezes_adaptation_through_delayed_plant_tail(algorithm):
    x = np.array([1., 100., 0., 0., 0., 0., 0., 1.])
    result = run_classical_anc(x, np.full(x.size, .1), [1., .5, .1],
                               algorithm=algorithm, control_length=1, mu=.1,
                               additional_delay_samples=3, secondary_estimate=[1.],
                               estimate_delay_samples=0, control_limit=.2)
    assert result["limited_samples"] == 1
    assert result["raw_control"][1] < -.2
    assert result["control"][1] == -.2
    # n=1 clip -> exclusive 2 + delay 3 + tail 2 = 7; n=7부터 적응 재개.
    assert result["clipped_guard_blocks"] == 6
    assert result["adapted_blocks"] == 2
    expected = .1 + np.convolve(np.pad(result["control"], (3, 0)), [1., .5, .1])[:x.size]
    np.testing.assert_allclose(result["residual"], expected, atol=1e-15)


@pytest.mark.parametrize("algorithm", ["fxlms", "fxnlms"])
def test_weight_projection_and_zero_reference_are_reported(algorithm):
    limited = run_classical_anc([1.], [1.], [1.], algorithm=algorithm,
                                control_length=1, mu=1., weight_norm_limit=.01)
    assert limited["weight_limited_blocks"] == 1
    assert limited["weight_norm"] == pytest.approx(.01)
    silent = run_classical_anc(np.zeros(20), np.ones(20) * .1, [.5], algorithm=algorithm)
    assert silent["adapted_blocks"] == 0
    np.testing.assert_array_equal(silent["control"], 0)
    np.testing.assert_array_equal(silent["residual"], .1)


@pytest.mark.parametrize("reference", [1e-8, 1e-200])
def test_plain_epsilon_does_not_create_a_hidden_power_gate(reference):
    result = run_classical_anc([reference], [1.], [1.], algorithm="fxlms", mu=.1,
                               control_length=1, normalization_epsilon=1.)
    assert result["adapted_blocks"] == 1
    np.testing.assert_allclose(result["final_weights"], [-.1 * reference], rtol=1e-15, atol=0)


@pytest.mark.parametrize("algorithm", ["fxlms", "fxnlms"])
def test_future_inputs_do_not_change_completed_outputs_and_calls_reset(algorithm):
    rng = np.random.default_rng(15)
    x, d, s = rng.normal(0, .04, 93), rng.normal(0, .01, 93), np.array([.31, -.1])
    originals = [value.copy() for value in (x, d, s)]
    first = run_classical_anc(x, d, s, algorithm=algorithm, block_samples=7)
    second = run_classical_anc(x, d, s, algorithm=algorithm, block_samples=7)
    for key in ("control", "residual", "final_weights"):
        np.testing.assert_array_equal(first[key], second[key])
    for value, saved in zip((x, d, s), originals):
        np.testing.assert_array_equal(value, saved)
    x[51:] += .1
    d[51:] -= .2
    changed = run_classical_anc(x, d, s, algorithm=algorithm, block_samples=7)
    np.testing.assert_array_equal(first["control"][:51], changed["control"][:51])
    np.testing.assert_array_equal(first["residual"][:51], changed["residual"][:51])


def test_omap_original_all_500_taps_and_full_precision_are_preserved():
    from deepanc.calibration import load_secondary_path
    s = load_secondary_path()
    original = s.copy()
    x = np.random.default_rng(36).normal(0, .02, 1024)
    d = signal.lfilter([0] * 128 + [.05], [1.], x)
    result = run_classical_anc(x, d, s, algorithm="fxnlms", block_samples=32,
                               additional_delay_samples=8)
    np.testing.assert_array_equal(s, original)
    expected = d + np.convolve(np.pad(result["control"], (8, 0)), s)[:x.size]
    np.testing.assert_allclose(result["residual"], expected, atol=1e-16, rtol=1e-12)
    settings = result["settings"]
    assert settings["secondary_taps"] == settings["estimate_taps"] == 500
    assert settings["secondary_sha256_f64le"] == hashlib.sha256(s.astype("<f8").tobytes()).hexdigest()
    assert settings["estimate_sha256_f64le"] == settings["secondary_sha256_f64le"]
    assert settings["estimate_matches_plant"] and settings["estimate_delay_matches_plant"]
    assert not any(result["claims"].values())


@pytest.mark.parametrize("name,value", [
    ("reference", []), ("reference", [[1.]]), ("reference", [np.nan]),
    ("reference", [1j]), ("reference", [True]), ("reference", ["1"]),
    ("disturbance", [np.inf]), ("disturbance", [1., 2.]),
    ("secondary", []), ("secondary", [0.]), ("secondary", [np.inf]),
    ("secondary_estimate", [0.]), ("secondary_estimate", [np.nan]),
])
def test_invalid_signals_fail_before_computation(name, value):
    args = dict(reference=[.1], disturbance=[.01], secondary=[.5], algorithm="fxlms")
    args[name] = value
    with pytest.raises(ValueError):
        run_classical_anc(**args)


@pytest.mark.parametrize("options", [
    {"algorithm": "FxLMS"}, {"algorithm": None}, {"algorithm": []},
    {"additional_delay_samples": -1}, {"estimate_delay_samples": 1.5},
    {"estimate_delay_samples": True}, {"control_length": 0}, {"block_samples": 0},
    {"mu": 0}, {"mu": np.inf}, {"mu": "0.1"}, {"control_limit": np.nan},
    {"normalization_epsilon": 0}, {"normalization_epsilon": True},
    {"weight_norm_limit": 0},
])
def test_invalid_settings_fail(options):
    args = {"algorithm": "fxnlms", **options}
    with pytest.raises(ValueError):
        run_classical_anc([.1], [.01], [.5], **args)


def test_large_design_matrix_rejected():
    with pytest.raises(ValueError, match="행렬"):
        run_classical_anc(np.ones(8192), np.zeros(8192), [1.], algorithm="fxlms",
                           block_samples=8192, control_length=4096)
