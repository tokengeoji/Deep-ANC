"""SFANC 비교용 FxNLMS의 부호·시간축·제한·원본 보존 회귀."""

import numpy as np
import pytest
from scipy import signal

from deep_anc.eval.sfanc_fxnlms import run_fxnlms


@pytest.mark.parametrize("delay", [0, 1, 7, 32, 93])
@pytest.mark.parametrize("block", [1, 17, 32])
def test_residual_matches_independent_full_convolution(delay, block):
    rng = np.random.default_rng(71)
    x = rng.normal(0, .04, 373)
    secondary = np.array([0, -.231234567890123, .1234567890123, .05])
    d = signal.lfilter([.04, -.01], [1], np.roll(x, 18))
    result = run_fxnlms(x, d, secondary, additional_delay_samples=delay, block_samples=block)
    expected = np.convolve(np.pad(result["control"], (delay, 0)), secondary)[:x.size] + d
    np.testing.assert_allclose(result["residual"], expected, rtol=1e-12, atol=1e-15)
    assert result["residual"].dtype == np.float64
    assert np.max(np.abs(result["control"])) <= .2
    assert not any(result["claims"].values())


@pytest.mark.parametrize("secondary", [[.5], [-.5]])
def test_actual_dac_sign_and_samplewise_learning(secondary):
    x = np.full(128, .1)
    d = np.full(128, .025)
    result = run_fxnlms(x, d, secondary, control_length=1, block_samples=1, mu=.1)
    assert result["raw_control"][0] == 0
    assert np.sign(result["raw_control"][-1]) == -np.sign(secondary[0])
    assert abs(result["residual"][-1]) < 1e-6


def test_no_implicit_one_hop_delay_and_partial_last_block():
    result = run_fxnlms(np.ones(10) * .1, np.ones(10) * .01, [1.0],
                        control_length=1, block_samples=4, mu=.5)
    np.testing.assert_array_equal(result["control"][:4], 0)
    assert result["control"][4] < 0
    assert result["residual"][4] < .01
    assert result["adapted_blocks"] == 3
    np.testing.assert_allclose(result["residual"], .01 + result["control"], atol=1e-15)


def test_zero_reference_cannot_generate_control_from_disturbance():
    d = np.linspace(-.1, .1, 111)
    result = run_fxnlms(np.zeros(111), d, [.5, -.1], block_samples=13)
    np.testing.assert_array_equal(result["control"], 0)
    np.testing.assert_array_equal(result["residual"], d)
    assert result["adapted_blocks"] == 0
    assert result["weight_norm"] == 0


def test_hard_limit_and_tail_guard_are_visible(monkeypatch):
    from deep_anc.eval import sfanc_fxnlms as module

    class ScriptedCore:
        def __init__(self, *args, **kwargs):
            self.index = 0
            self.w = np.zeros(1, dtype=np.float32)

        def generate_block(self, reference):
            output = np.full(reference.size, 1.0 if self.index == 0 else .01)
            self.index += 1
            return output

        def adapt_block(self, error, enabled=True):
            from deep_anc.baselines.fxlms_core import AdaptationResult
            calls.append(enabled)
            return AdaptationResult(enabled, 1.0, 0.0, 0.0, False, "scripted")

    calls = []
    monkeypatch.setattr(module, "FxLMSController", ScriptedCore)
    result = module.run_fxnlms(np.ones(24), np.zeros(24), [1, .5, .1],
                               control_length=1, block_samples=4,
                               additional_delay_samples=5, control_limit=.2)
    # clip block exclusive 끝 4 + delay 5 + tail 2 = 11; block 12부터 적응 재개.
    assert calls == [False, False, False, True, True, True]
    assert result["limited_samples"] == 4
    assert result["clipped_guard_blocks"] == 3
    np.testing.assert_array_equal(result["control"][:4], .2)
    expected = np.convolve(np.pad(result["control"], (5, 0)), [1, .5, .1])[:24]
    np.testing.assert_allclose(result["residual"], expected)


def test_inputs_not_mutated_and_each_call_resets_every_state():
    x = np.linspace(-.1, .1, 99)
    d = np.sin(np.arange(99)) * .04
    s = np.array([0.0, .32123456789123456, -.03])
    originals = [value.copy() for value in (x, d, s)]
    first = run_fxnlms(x, d, s, additional_delay_samples=5)
    second = run_fxnlms(x, d, s, additional_delay_samples=5)
    for key in ("control", "raw_control", "residual", "final_weights"):
        np.testing.assert_array_equal(first[key], second[key])
    for value, original in zip((x, d, s), originals):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("delay", [0, 3, 27])
def test_samplewise_filtered_x_alignment_matches_independent_recurrence(delay):
    rng = np.random.default_rng(229)
    x = rng.normal(0, .1, 128)
    d = signal.lfilter([0] * 11 + [.04, -.01], [1], x)
    s = np.array([-.31, .2, .05])
    length, mu = 4, .025
    result = run_fxnlms(x, d, s, additional_delay_samples=delay,
                        control_length=length, block_samples=1, mu=mu)
    xf = np.convolve(np.pad(x, (delay, 0)), s)[:x.size]
    weights = np.zeros(length)
    control = np.zeros_like(x)
    x_padded = np.pad(x, (length - 1, 0))
    xf_padded = np.pad(xf, (length - 1, 0))
    for n in range(x.size):
        control[n] = np.dot(weights, x_padded[n:n + length][::-1])
        error = d[n]
        for k, coefficient in enumerate(s):
            if n - delay - k >= 0:
                error += coefficient * control[n - delay - k]
        taps = xf_padded[n:n + length][::-1]
        power = np.dot(taps, taps)
        if power > 1e-12:
            weights -= mu * error * taps / (power + 1e-12)
    assert result["limited_samples"] == result["weight_limited_blocks"] == 0
    np.testing.assert_allclose(result["control"], control, atol=2e-9, rtol=5e-6)
    np.testing.assert_allclose(result["final_weights"], weights, atol=2e-8, rtol=5e-6)


def test_future_reference_and_error_do_not_change_completed_outputs():
    rng = np.random.default_rng(301)
    x = rng.normal(0, .05, 133)
    d = rng.normal(0, .01, 133)
    original = run_fxnlms(x, d, [-.5, .2], block_samples=17)
    x[75:] = rng.normal(0, .3, x.size - 75)
    d[75:] = rng.normal(0, .4, d.size - 75)
    changed = run_fxnlms(x, d, [-.5, .2], block_samples=17)
    # 75는 블록 경계가 아니다. 현재 블록 ERR도 현재 출력에 소급 반영되지 않는다.
    np.testing.assert_array_equal(original["control"][:75], changed["control"][:75])
    np.testing.assert_array_equal(original["residual"][:75], changed["residual"][:75])


@pytest.mark.parametrize("field,bad", [
    ("reference", []), ("reference", [[.1]]), ("reference", [np.nan]),
    ("reference", [True]), ("reference", [1j]), ("reference", ["1"]),
    ("disturbance", [np.inf]), ("disturbance", [.1, .2]),
    ("secondary", []), ("secondary", [0, 0]), ("secondary", [np.nan]),
    ("secondary", [1e7]),
])
def test_invalid_waves_rejected(field, bad):
    args = {"reference": [.1], "disturbance": [.02], "secondary": [.5]}
    args[field] = bad
    with pytest.raises(ValueError):
        run_fxnlms(**args)


@pytest.mark.parametrize("kwargs", [
    {"additional_delay_samples": -1}, {"additional_delay_samples": 1.5},
    {"additional_delay_samples": True}, {"control_length": 0},
    {"block_samples": 0}, {"block_samples": np.nan}, {"mu": 0},
    {"mu": 2.01}, {"mu": np.nan}, {"mu": "0.1"}, {"mu": True},
    {"control_limit": 0}, {"control_limit": np.inf},
])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        run_fxnlms([.1], [.02], [.5], **kwargs)


def test_matrix_allocation_guard():
    with pytest.raises(ValueError, match="행렬"):
        run_fxnlms(np.ones(8192), np.zeros(8192), [1.0],
                   block_samples=8192, control_length=4096)


def test_original_omap_500_tap_plant_preserves_full_precision():
    from deepanc.calibration import load_secondary_path
    secondary = load_secondary_path()
    rng = np.random.default_rng(911)
    x = rng.normal(0, .03, 2048)
    d = signal.lfilter([0] * 128 + [.05], [1], x)
    result = run_fxnlms(x, d, secondary, additional_delay_samples=7)
    expected = np.convolve(np.pad(result["control"], (7, 0)), secondary)[:x.size] + d
    assert result["settings"]["secondary_taps"] == 500
    np.testing.assert_allclose(result["residual"], expected, atol=1e-15, rtol=1e-12)
