"""연속 FIR 수치 회귀. 실제 음향 감쇠·오디오 deadline 검증이 아니다."""

import json

import numpy as np
import pytest

from deep_anc.baselines import sfanc_stream as stream


def direct_varying(x, weights):
    return np.array([sum(weights[n, k] * x[n-k] for k in range(weights.shape[1]) if n >= k)
                     for n in range(len(x))], dtype=np.float64)


@pytest.mark.parametrize("length", [1, 2, 17, 128])
def test_constant_filter_matches_direct_convolution_and_random_chunks(length):
    rng = np.random.default_rng(length)
    x, w = rng.normal(size=719), rng.normal(size=(3, length))
    core = stream.StreamingFIRBank(w, initial_index=2)
    outputs = []
    position = 0
    while position < len(x):
        end = min(len(x), position + int(rng.integers(1, 54)))
        outputs.append(core.process(x[position:end]))
        position = end
    expected = direct_varying(x, np.broadcast_to(w[2], (len(x), length)))
    np.testing.assert_allclose(np.concatenate(outputs), expected, rtol=1e-12, atol=1e-12)
    whole = stream.StreamingFIRBank(w, initial_index=2).process(x)
    np.testing.assert_allclose(np.concatenate(outputs), whole, rtol=1e-12, atol=1e-12)
    assert core.diagnostics["processed_samples"] == len(x)


@pytest.mark.parametrize("transition", [0, 1, 4, 23])
def test_switch_preserves_history_and_matches_independent_coefficients(transition):
    rng = np.random.default_rng(14)
    x, w = rng.normal(size=91), rng.normal(size=(2, 9))
    core = stream.StreamingFIRBank(w, transition_samples=transition)
    before = core.process(x[:37])
    core.request_filter(1)
    after = core.process(x[37:])
    weights = np.tile(w[0], (len(x), 1))
    for n in range(37, len(x)):
        alpha = 1.0 if not transition else min((n-36)/transition, 1.0)
        weights[n] = (1-alpha)*w[0] + alpha*w[1]
    np.testing.assert_allclose(np.r_[before, after], direct_varying(x, weights), atol=1e-12)
    assert core.transition_remaining == 0
    np.testing.assert_array_equal(core.current_coefficients, w[1])


def test_switch_on_impulse_tail_does_not_erase_previous_input():
    core = stream.StreamingFIRBank([[0, 0, 0, 0], [0, 0, 0, 1]])
    np.testing.assert_array_equal(core.process([1, 0]), [0, 0])
    core.request_filter(1)
    np.testing.assert_array_equal(core.process([0, 0, 0]), [0, 1, 0])


def test_transition_is_chunk_independent_including_mid_transition_requests():
    rng = np.random.default_rng(244)
    w, x = rng.normal(size=(3, 7)), rng.normal(size=100)

    def run(chunks):
        core = stream.StreamingFIRBank(w, transition_samples=13)
        events = {9: 1, 14: 2, 19: 0, 41: 1}
        results = []
        for start, end in chunks:
            if start in events:
                core.request_filter(events[start])
            results.append(core.process(x[start:end]))
        return np.concatenate(results), core

    coarse, a = run([(0, 9), (9, 14), (14, 19), (19, 41), (41, 100)])
    fine, b = run([(i, i+1) for i in range(100)])
    np.testing.assert_allclose(coarse, fine, atol=1e-13, rtol=1e-13)
    assert a.diagnostics == b.diagnostics
    np.testing.assert_array_equal(a.current_coefficients, b.current_coefficients)


def test_mid_transition_restarts_from_last_actual_coefficient():
    core = stream.StreamingFIRBank([[0.0], [1.0], [-1.0]], transition_samples=4)
    core.request_filter(1)
    np.testing.assert_allclose(core.process([1, 1]), [.25, .5])
    assert core.transition_remaining == 2
    core.request_filter(2)
    np.testing.assert_array_equal(core.current_coefficients, [.5])
    np.testing.assert_allclose(core.process([1]*5), [.125, -.25, -.625, -1, -1])
    assert core.transition_remaining == 0


def test_same_target_request_is_noop_before_during_and_after_transition():
    core = stream.StreamingFIRBank([[0], [1]], transition_samples=4)
    initial = core.diagnostics
    core.request_filter(0)
    assert core.diagnostics == initial
    core.request_filter(1)
    core.process([1])
    during = core.diagnostics
    core.request_filter(1)
    assert core.diagnostics == during
    np.testing.assert_allclose(core.process([1]*3), [.5, .75, 1])
    completed = core.diagnostics
    core.request_filter(1)
    assert core.diagnostics == completed


def test_future_reference_changes_do_not_change_earlier_output():
    rng = np.random.default_rng(89)
    w, x = rng.normal(size=(3, 9)), rng.normal(size=120)
    changed = x.copy()
    changed[72:] = rng.normal(size=48)*10
    outputs = []
    for values in (x, changed):
        core = stream.StreamingFIRBank(w, transition_samples=100)
        core.request_filter(2)
        outputs.append(core.process(values))
    np.testing.assert_array_equal(outputs[0][:72], outputs[1][:72])


def test_limit_only_clips_output_and_keeps_input_history_and_raw_statistics():
    w, x = [[2, -1]], np.array([1., 0, -1, 0, .25])
    linear = stream.StreamingFIRBank(w)
    clipped = stream.StreamingFIRBank(w, control_limit=.5)
    raw = linear.process(x)
    output = np.r_[clipped.process(x[:2]), clipped.process(x[2:])]
    np.testing.assert_array_equal(output, np.clip(raw, -.5, .5))
    assert clipped.diagnostics["limited_samples"] == np.count_nonzero(np.abs(raw) > .5)
    assert clipped.diagnostics["raw_peak"] == 2
    assert clipped.diagnostics["output_peak"] == .5
    assert clipped.diagnostics["limiter_kind"] == "hard_clip"
    assert not clipped.diagnostics["deployment_allowed"]
    assert not clipped.diagnostics["realtime_validated"]
    json.dumps(clipped.diagnostics, allow_nan=False)


def test_reset_restores_initial_candidate_history_statistics_and_pending_transition():
    w = [[0, 1, 0], [2, 0, -1]]
    core = stream.StreamingFIRBank(w, initial_index=1, transition_samples=4, control_limit=.5)
    original = core.diagnostics
    expected = core.process([1, 2, 3])
    core.request_filter(0)
    core.process([4])
    assert core.transition_remaining == 3
    core.reset()
    assert core.diagnostics == original
    np.testing.assert_array_equal(core.process([1, 2, 3]), expected)


def test_bank_inputs_and_observable_copies_cannot_modify_internal_state():
    w = np.array([[.25, -.125], [1., 0.]])
    saved = w.copy()
    core = stream.StreamingFIRBank(w)
    np.testing.assert_array_equal(w, saved)
    w[:] = 42
    current = core.current_coefficients
    current[:] = 12
    diagnostic = core.diagnostics
    diagnostic["processed_samples"] = 99
    x = np.array([1., 0, 0])
    x_saved = x.copy()
    np.testing.assert_array_equal(core.process(x), [.25, -.125, 0])
    np.testing.assert_array_equal(x, x_saved)
    assert core.diagnostics["processed_samples"] == 3


@pytest.mark.parametrize("bank", [[], [1], [[1], [1, 2]], np.empty((0, 3)),
                                    [[True]], [[1+1j]], [["1"]], [[np.nan]], [[np.inf]]])
def test_invalid_bank(bank):
    with pytest.raises(ValueError):
        stream.StreamingFIRBank(bank)


@pytest.mark.parametrize("kwargs", [
    {"initial_index": -1}, {"initial_index": 2}, {"initial_index": True},
    {"initial_index": 1.0}, {"transition_samples": -1}, {"transition_samples": .5},
    {"transition_samples": True}, {"transition_samples": 1_000_001},
    {"control_limit": 0}, {"control_limit": -.5}, {"control_limit": np.inf},
    {"control_limit": np.nan}, {"control_limit": True}, {"control_limit": ".5"},
    {"control_limit": 10**1000},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        stream.StreamingFIRBank([[0], [1]], **kwargs)


@pytest.mark.parametrize("reference", [[], [[1]], [True], [1+1j], ["1"], [np.nan], [np.inf]])
def test_invalid_process_leaves_state_unchanged(reference):
    core = stream.StreamingFIRBank([[0, 0], [1, .5]], transition_samples=3)
    core.process([1])
    core.request_filter(1)
    before = core.diagnostics
    with pytest.raises(ValueError):
        core.process(reference)
    assert core.diagnostics == before
    np.testing.assert_allclose(core.process([0]), [1/6])


@pytest.mark.parametrize("index", [-1, 2, True, .5, np.nan])
def test_invalid_request_leaves_state_unchanged(index):
    core = stream.StreamingFIRBank([[0], [1]], transition_samples=4)
    core.request_filter(1)
    before = core.diagnostics
    with pytest.raises(ValueError):
        core.request_filter(index)
    assert core.diagnostics == before
    np.testing.assert_array_equal(core.process([1]), [.25])


def test_overflow_is_not_hidden_by_clipping_and_does_not_change_state():
    core = stream.StreamingFIRBank([[1e308, 0]], control_limit=.2)
    before = core.diagnostics
    with pytest.raises(ValueError, match="FP64"):
        core.process([2., 0])
    assert core.diagnostics == before
    np.testing.assert_array_equal(core.process([0., 0]), [0., 0])


def test_resource_limits_fail_before_large_computation(monkeypatch):
    monkeypatch.setattr(stream, "MAX_TAPS", 2)
    with pytest.raises(ValueError, match="크기"):
        stream.StreamingFIRBank([[1, 2, 3]])
    monkeypatch.setattr(stream, "MAX_BANK_COEFFICIENTS", 3)
    with pytest.raises(ValueError, match="크기"):
        stream.StreamingFIRBank([[1, 2], [3, 4]])
    core = stream.StreamingFIRBank([[1, 2]])
    monkeypatch.setattr(stream, "MAX_BLOCK_SAMPLES", 3)
    with pytest.raises(ValueError, match="크기"):
        core.process([1]*4)
    monkeypatch.setattr(stream, "MAX_FILTER_PRODUCTS", 3)
    with pytest.raises(ValueError, match="크기"):
        core.process([1, 2])
    assert core.diagnostics["processed_samples"] == 0


def test_integer_numpy_configuration_and_noncontiguous_input():
    core = stream.StreamingFIRBank(np.array([[1, -1], [0, 1]], dtype=np.int16),
                                 initial_index=np.int64(0), transition_samples=np.int32(1),
                                 control_limit=np.float32(100))
    core.request_filter(np.int64(1))
    out = core.process(np.arange(10, dtype=np.int16)[::2])
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, [0, 0, 2, 4, 6])


def test_transition_overflow_is_atomic_even_after_a_good_prefix():
    core = stream.StreamingFIRBank([[0, 0], [1e308, 0]], transition_samples=4)
    core.process([.1])
    core.request_filter(1)
    before = core.diagnostics
    with pytest.raises(ValueError, match="FP64"):
        core.process([.1, .1, 2])
    assert core.diagnostics == before
    np.testing.assert_array_equal(core.current_coefficients, [0, 0])
    np.testing.assert_allclose(core.process([.1]), [2.5e306])
    assert core.transition_remaining == 3


def test_large_opposite_coefficients_do_not_overflow_during_interpolation():
    core = stream.StreamingFIRBank([[1e308], [-1e308]], transition_samples=2)
    core.request_filter(1)
    np.testing.assert_array_equal(core.process([.1]), [0])
    np.testing.assert_array_equal(core.current_coefficients, [0])
    np.testing.assert_allclose(core.process([.1]), [-1e307])


def test_request_before_any_sample_restarts_from_original_not_unapplied_target():
    core = stream.StreamingFIRBank([[0], [1], [-1]], transition_samples=4)
    core.request_filter(1)
    core.request_filter(2)
    np.testing.assert_allclose(core.process([1]*4), [-.25, -.5, -.75, -1])


def test_generic_bank_does_not_require_zero_candidate():
    core = stream.StreamingFIRBank([[2], [3]], initial_index=1)
    np.testing.assert_array_equal(core.process([1, -1]), [3, -3])
