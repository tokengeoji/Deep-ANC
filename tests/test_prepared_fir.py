"""논문에서 착안한 계수 전달 규약의 합성 검증. Jetson/덕트 성능 시험이 아니다."""

from dataclasses import replace
import subprocess
import sys

import numpy as np
import pytest
from scipy import signal

from deep_anc.baselines.prepared_fir import FilterProposal, PreparedFxNLMSController
from deep_anc.dsp.filters import StreamingFIR, SampleDelay


def controller(**changes):
    options = dict(
        sample_rate=8000, hop=8, secondary_delay_samples=3, handoff_extra_samples=8,
        control_length=2, reference_peak_limit=1.0, control_limit=0.9,
        prepared_l1_limit=2.0, residual_norm_limit=2.0, mu=0.2, leakage=0.0,
        transition_samples=16, max_proposal_age_samples=10000,
    )
    options.update(changes)
    secondary = options.pop("s_hat", np.array([-0.5]))
    return PreparedFxNLMSController(secondary, **options)


def proposal(c, coefficients=(0.1, 0.0), **changes):
    fields = dict(context_id=c.context_id, generation=c.generation, revision=0,
                  observed_until_sample=c.sample_cursor, coefficients=coefficients,
                  source_id="독립 합성 train 후보")
    fields.update(changes)
    return FilterProposal(**fields)


def advance(c, ref=None, err=None, enabled=False):
    ref = np.zeros(c.hop) if ref is None else ref
    err = np.zeros(c.hop) if err is None else err
    y = c.generate_block(ref)
    result = c.adapt_block(err, enabled=enabled)
    return y, result


def test_zero_start_and_adaptation_is_opt_in():
    c = controller()
    assert c.generation == 0 and c.sample_cursor == 0
    for _ in range(12):
        y, result = advance(c, np.full(8, 0.2), np.full(8, 0.1))
        np.testing.assert_array_equal(y, 0)
        assert not result.adapted
    np.testing.assert_array_equal(c.residual_weights, 0)


def test_immutable_proposal_and_safe_weight_snapshots():
    c = controller()
    coefficients = np.array([0.1, 0.0])
    p = proposal(c, coefficients)
    coefficients[0] = 99
    assert p.coefficients == (0.1, 0.0)
    assert c.submit(p).accepted
    advance(c)
    c.residual_weights[:] = 99
    c.prepared_coefficients[:] = 99
    assert np.max(np.abs(c.prepared_coefficients)) < 1
    np.testing.assert_array_equal(c.residual_weights, 0)


def test_common_history_crossfade_matches_explicit_causal_convolution():
    c = controller(control_length=3, transition_samples=11)
    rng = np.random.default_rng(11)
    ref = rng.uniform(-0.1, 0.1, 80)
    first = np.array([0.2, -0.1, 0.05])
    second = np.array([-0.1, 0.15, 0.0])
    output = []
    assert c.submit(proposal(c, first)).accepted
    for start in range(0, len(ref), c.hop):
        if start == 32:
            assert c.submit(proposal(c, second, revision=1)).accepted
        output.append(advance(c, ref[start:start + c.hop])[0])
    old = signal.lfilter(first, [1], ref)
    new = signal.lfilter(second, [1], ref)
    expected = old.copy()
    expected[:11] *= np.arange(1, 12) / 11
    alpha = np.minimum(1, np.arange(1, len(ref) - 32 + 1) / 11)
    expected[32:] = (1 - alpha) * old[32:] + alpha * new[32:]
    np.testing.assert_allclose(np.concatenate(output), expected, atol=1e-8, rtol=1e-6)


def test_future_input_changes_cannot_change_current_output():
    a, b = controller(), controller()
    assert a.submit(proposal(a, (0.1, -0.1))).accepted
    assert b.submit(proposal(b, (0.1, -0.1))).accepted
    left, right = np.linspace(-0.1, 0.1, 32), np.linspace(-0.1, 0.1, 32)
    right[21:] = 0.7
    ya = np.concatenate([advance(a, left[i:i + 8])[0] for i in range(0, 32, 8)])
    yb = np.concatenate([advance(b, right[i:i + 8])[0] for i in range(0, 32, 8)])
    np.testing.assert_array_equal(ya[:21], yb[:21])


@pytest.mark.parametrize("fields,reason", [
    ({"context_id": "다른조건"}, "context_mismatch"),
    ({"generation": 1}, "generation_mismatch"),
    ({"observed_until_sample": 1}, "future_observation"),
    ({"coefficients": (0.1,)}, "coefficient_shape"),
    ({"coefficients": (2.1, 0)}, "coefficient_l1_limit"),
])
def test_invalid_proposal_does_not_modify_controller(fields, reason):
    c = controller()
    decision = c.submit(proposal(c, **fields))
    assert not decision.accepted and decision.reason == reason
    np.testing.assert_array_equal(c.prepared_coefficients, 0)
    assert c.sample_cursor == 0


def test_revision_expiry_nested_transition_and_pending_block_are_rejected():
    c = controller(max_proposal_age_samples=8)
    old = proposal(c)
    assert c.submit(old).accepted
    assert c.submit(old).reason == "stale_revision"
    assert c.submit(replace(old, revision=1)).reason == "transition_in_progress"
    c.generate_block(np.zeros(8))
    assert c.submit(replace(old, revision=1)).reason == "pending_error_block"
    with pytest.raises(RuntimeError, match="ERR"):
        c.generate_block(np.zeros(8))
    c.adapt_block(np.zeros(8))
    advance(c)
    assert c.submit(replace(old, revision=1)).reason == "expired_observation"
    assert c.submit(proposal(c, revision=1)).accepted


def test_context_is_bound_to_secondary_delay_handoff_and_limits():
    a = controller()
    assert a.context_id == controller().context_id
    for changed in (controller(handoff_extra_samples=0), controller(secondary_delay_samples=4),
                    controller(control_limit=0.2), controller(s_hat=np.array([0.5]))):
        assert changed.submit(proposal(a)).reason == "context_mismatch"


def test_caller_cannot_mutate_secondary_after_context_hash_was_created():
    original = np.array([-0.5], dtype=np.float32)
    a = controller(s_hat=original)
    b = controller(s_hat=original.copy())
    original[0] = 0.5
    assert a.context_id == b.context_id
    for _ in range(20):
        ya, _ = advance(a, np.full(8, 0.1), np.full(8, 0.01), enabled=True)
        yb, _ = advance(b, np.full(8, 0.1), np.full(8, 0.01), enabled=True)
        np.testing.assert_array_equal(ya, yb)
        np.testing.assert_array_equal(a.residual_weights, b.residual_weights)


def test_extreme_diagnostic_overflow_resets_instead_of_returning_infinity():
    # 실기 성능과 무관한 float32 코어 진단 경계: 유한 입력이어도 norm은 overflow할 수 있다.
    c = controller(hop=256, secondary_delay_samples=0, handoff_extra_samples=0,
                   control_length=1, s_hat=np.array([1e6]), reference_peak_limit=1e6,
                   control_limit=1e6)
    c.generate_block(np.full(256, 1e6))
    with np.errstate(over="ignore"):
        with pytest.raises(FloatingPointError, match="수치 범위"):
            c.adapt_block(np.full(256, 1e6), enabled=True)
    assert c.generation == 1 and c.sample_cursor == 0
    np.testing.assert_array_equal(c.residual_weights, 0)


def test_adaptation_waits_through_fade_and_exact_secondary_tail():
    c = controller(s_hat=np.array([-0.5, 0.1, 0.05]), secondary_delay_samples=5,
                   transition_samples=10)
    assert c.secondary_tail_samples == 5 + 8 + 3 - 1
    assert c.submit(proposal(c)).accepted
    observed = [advance(c, np.full(8, 0.1), np.full(8, 0.01), enabled=True)[1].adapted
                for _ in range(6)]
    # 10샘플 fade + 15샘플 S/handoff 꼬리: 시작24인 블록도 완전히 건너뛴다.
    assert observed[:4] == [False] * 4
    assert all(observed[4:])


def test_combined_limiter_is_visible_and_delayed_clipping_freezes_adaptation():
    c = controller(control_limit=0.01, transition_samples=1)
    assert c.submit(proposal(c, (1.0, 0.0))).accepted
    y, result = advance(c, np.full(8, 0.5), np.full(8, 0.1), enabled=True)
    assert np.max(np.abs(y)) <= 0.010000001
    assert c.clip_samples == 8 and not result.adapted
    assert c.last_guard_reason == "output_clipping_guard"
    for _ in range(2):
        _, result = advance(c, np.zeros(8), np.full(8, 0.1), enabled=True)
        assert not result.adapted


def test_residual_corrects_prepared_error_and_survives_new_base_proposal():
    c = controller(control_length=1, transition_samples=8)
    plant, primary = StreamingFIR(np.array([-0.5]), 11), SampleDelay(11)
    assert c.submit(proposal(c, (0.1,))).accepted
    rng = np.random.default_rng(123)
    powers = []
    for _ in range(220):
        ref = rng.uniform(-0.1, 0.1, 8).astype(np.float32)
        y = c.generate_block(ref)
        err = 0.1 * primary.process(ref) + plant.process(y)
        c.adapt_block(err, enabled=True)
        powers.append(np.mean(err**2))
    assert np.mean(powers[-20:]) < np.mean(powers[3:13]) * 1e-3
    assert c.residual_weights[0] == pytest.approx(0.1, abs=1e-5)
    before = c.residual_weights
    assert c.submit(proposal(c, (0.08,), revision=1)).accepted
    advance(c, np.ones(8) * 0.1, np.ones(8) * 0.01, enabled=True)
    np.testing.assert_array_equal(c.residual_weights, before)


@pytest.mark.parametrize("channel,value", [("REF", np.nan), ("REF", 1.1), ("ERR", np.inf)])
def test_bad_input_resets_history_and_invalidates_old_generation(channel, value):
    c = controller()
    old = proposal(c)
    assert c.submit(old).accepted
    if channel == "ERR":
        c.generate_block(np.ones(8) * 0.1)
    with pytest.raises(ValueError, match="초기화"):
        if channel == "ERR":
            c.adapt_block(np.full(8, value), enabled=True)
        else:
            c.generate_block(np.full(8, value))
    assert c.generation == 1 and c.sample_cursor == 0
    np.testing.assert_array_equal(c.prepared_coefficients, 0)
    np.testing.assert_array_equal(c.residual_weights, 0)
    assert c.submit(old).reason == "generation_mismatch"


@pytest.mark.parametrize("channel", ["REF", "ERR"])
def test_ragged_array_conversion_failure_also_resets_controller(channel):
    c = controller()
    old = proposal(c)
    assert c.submit(old).accepted
    advance(c, np.full(8, 0.1))
    if channel == "ERR":
        c.generate_block(np.zeros(8))
    with pytest.raises(ValueError, match="배열 변환 실패"):
        method = c.generate_block if channel == "REF" else c.adapt_block
        method([[0.1], [0.2, 0.3]])
    assert c.generation == 1 and c.sample_cursor == 0
    assert c.submit(old).reason == "generation_mismatch"
    np.testing.assert_array_equal(c.prepared_coefficients, 0)


@pytest.mark.parametrize("change", [
    {"hop": 0}, {"hop": 8.5}, {"handoff_extra_samples": -1}, {"transition_samples": 0},
    {"max_proposal_age_samples": -1}, {"reference_peak_limit": np.nan}, {"leakage": None},
    {"mu": 3}, {"s_hat": np.ones((2, 2))}, {"s_hat": np.zeros(3)},
])
def test_bad_settings_fail_before_any_processing(change):
    with pytest.raises(ValueError):
        controller(**change)


@pytest.mark.parametrize("coefficients", [(np.nan, 0), (np.inf, 0), ((0.1, 0),), ("0.1", "0")])
def test_bad_coefficient_payload_is_rejected(coefficients):
    with pytest.raises(ValueError):
        proposal(controller(), coefficients)


def test_import_does_not_require_neural_or_audio_runtimes():
    code = '''
import importlib.abc
import sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'sounddevice', 'onnxruntime', 'tensorrt'}:
            raise RuntimeError(fullname)
sys.meta_path.insert(0, Reject())
import deep_anc.baselines.prepared_fir
'''
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, run.stderr
