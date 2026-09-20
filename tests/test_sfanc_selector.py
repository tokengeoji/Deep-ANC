"""REF-only 오프라인 선택기: 특징, 비용 학습, 검증 복원, 실패-폐쇄 회귀."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy.signal import welch
import torch

import deep_anc.train.sfanc_selector as selector
from deep_anc.train.sfanc_selector import (
    RefSpectrumSelector,
    predict_selector,
    reference_features,
    train_selector,
)


@pytest.fixture(autouse=True)
def _small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _examples(count: int = 48, seed: int = 1):
    rng = np.random.default_rng(seed)
    labels = np.arange(count) % 3
    frequency = np.asarray([5, 17, 27])[labels]
    phase = rng.uniform(-np.pi, np.pi, (count, 1))
    reference = 0.1 * np.sin(2 * np.pi * frequency[:, None] * np.arange(256) / 64 + phase)
    reference += rng.normal(0, 0.001, reference.shape)
    features = reference_features(reference, n_fft=64)
    costs = np.full((count, 3), 1.2)
    costs[np.arange(count), labels] = 0.1
    return reference, features, costs


def test_reference_features_match_hann_welch_and_keep_source_gain():
    reference = np.random.default_rng(5).normal(0, 0.1, (2, 384))
    original = reference.copy()
    features = reference_features(reference, n_fft=64)
    _, psd = welch(reference, fs=1.0, window="hann", nperseg=64, noverlap=32,
                   nfft=64, detrend=False, axis=-1)
    expected = np.log(np.maximum(psd / psd.sum(axis=-1, keepdims=True), 1e-12))
    assert features.shape == (2, 2, 33)
    assert features.dtype == np.float32
    np.testing.assert_allclose(features[:, 0], expected, atol=1e-6)
    rms = np.sqrt(np.mean(reference**2, axis=-1))
    np.testing.assert_allclose(features[:, 1, 0], np.log(rms), atol=1e-6)
    np.testing.assert_array_equal(features[:, 1], np.repeat(features[:, 1, :1], 33, axis=1))
    louder = reference_features(3 * reference, n_fft=64)
    np.testing.assert_allclose(louder[:, 0], features[:, 0], atol=1e-6)
    np.testing.assert_allclose(louder[:, 1] - features[:, 1], np.log(3), atol=1e-6)
    np.testing.assert_array_equal(reference, original)


@pytest.mark.parametrize("reference", [np.zeros(1), np.zeros(9), np.ones(1), np.ones((2, 3))])
def test_silence_and_short_windows_are_finite(reference):
    features = reference_features(reference)
    assert features.shape == ((1 if reference.ndim == 1 else len(reference)), 2, 257)
    assert np.isfinite(features).all()


def test_dc_is_not_detrended():
    features = reference_features(np.ones(512))
    assert features[0, 0].argmax() == 0
    assert features[0, 1, 0] == 0


@pytest.mark.parametrize("reference", [
    np.empty(0), np.empty((0, 8)), np.empty((2, 0)), np.ones((2, 1, 8)),
    np.array([np.nan]), np.array([np.inf]), np.array([1j]),
    np.array([True]), np.array(["1"]), np.array([1e300]),
])
def test_bad_reference_fails(reference):
    with pytest.raises(ValueError):
        reference_features(reference)


@pytest.mark.parametrize("n_fft", [0, 1, 16_385, 3.5, True])
def test_fft_bounds(n_fft):
    with pytest.raises(ValueError, match="n_fft"):
        reference_features(np.ones(16), n_fft=n_fft)


def test_element_limit_checked_before_fft(monkeypatch):
    monkeypatch.setattr(selector, "_MAX_ELEMENTS", 10)
    with pytest.raises(ValueError, match="elements"):
        reference_features(np.ones(11))
    with pytest.raises(ValueError, match="output exceeds"):
        reference_features(np.ones(3), n_fft=64)


def test_model_output_shape_and_backward():
    model = RefSpectrumSelector(candidate_count=3, bins=33, width=8)
    assert model.candidate_count == 3
    features = torch.randn(4, 2, 33, requires_grad=True)
    logits = model(features)
    assert logits.shape == (4, 3)
    logits.square().mean().backward()
    assert torch.isfinite(logits).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError, match="features"):
        model(torch.zeros(4, 1, 33))


@pytest.mark.parametrize("kwargs", [
    {"candidate_count": 0}, {"candidate_count": 257}, {"candidate_count": True},
    {"bins": 1}, {"bins": 8194}, {"width": 0}, {"width": 257},
])
def test_model_bounds(kwargs):
    arguments = {"candidate_count": 3, "bins": 33, "width": 8, **kwargs}
    with pytest.raises(ValueError):
        RefSpectrumSelector(**arguments)


def test_soft_labels_include_cost_regret_and_ties():
    costs = np.array([[1.0, 1.0, 1.5], [10.0, 10.5, 11.0]])
    targets = selector._targets(costs, 0.5).numpy()
    expected = np.exp(-(costs - costs.min(axis=1, keepdims=True)) / 0.5)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(targets, expected, rtol=1e-6)
    assert targets[0, 0] == targets[0, 1]
    assert 0 < targets[0, 2] < targets[0, 0]
    extreme = selector._targets(np.array([[0.0, 1e30]]), 1e-300).numpy()
    np.testing.assert_array_equal(extreme, [[1, 0]])


def test_training_learns_cost_selected_candidate_and_is_repeatable():
    _, train, costs = _examples()
    reference, validation, validation_costs = _examples(24, seed=9)
    original = train.copy()
    rng_state = torch.random.get_rng_state().clone()
    results = [train_selector(train, costs, validation, validation_costs,
                              epochs=18, batch_size=16, learning_rate=0.003, seed=11)
               for _ in range(2)]
    first, second = results
    assert first.best_validation_loss <= 0.11
    assert first.best_validation_loss < first.initial_validation_loss
    assert first.history == second.history
    assert first.best_epoch == second.best_epoch
    assert first.best_epoch >= 1
    assert len(first.history) == 18
    assert not first.model.training
    assert all(np.isfinite(value) for row in first.history for value in row.values())
    for name, parameter in first.model.state_dict().items():
        torch.testing.assert_close(parameter, second.model.state_dict()[name], rtol=0, atol=0)
    chosen = predict_selector(first.model, reference, n_fft=64)
    assert chosen.dtype == np.int64 and chosen.shape == (24,)
    selected_cost = validation_costs[np.arange(24), chosen].mean()
    assert selected_cost == pytest.approx(first.best_validation_loss)
    np.testing.assert_array_equal(train, original)
    assert torch.equal(rng_state, torch.random.get_rng_state())


def test_best_state_is_deep_copy_not_last_epoch(monkeypatch):
    _, features, costs = _examples(6)

    def run(epochs):
        calls = 0

        def evaluate(*args):
            nonlocal calls
            calls += 1
            # 첫 검증=10, epoch별 train=0, validation=1,2,3. 첫 epoch만 최상.
            selected = 10.0 if calls == 1 else (0.0 if calls % 2 == 0 else float(calls // 2))
            return 0.0, selected, 0.0

        monkeypatch.setattr(selector, "_evaluate", evaluate)
        return train_selector(features, costs, features, costs, epochs=epochs, batch_size=6, seed=2)

    short = run(1)
    longer = run(3)
    assert short.best_epoch == longer.best_epoch == 1
    for name, parameter in short.model.state_dict().items():
        torch.testing.assert_close(parameter, longer.model.state_dict()[name], rtol=0, atol=0)


def test_initial_model_remains_best_on_equal_validation_cost():
    _, features, costs = _examples(6)
    result = train_selector(features, costs, features, np.ones_like(costs), epochs=2, seed=4)
    with torch.random.fork_rng():
        torch.manual_seed(4)
        initial = RefSpectrumSelector(3, 33)
    assert result.best_epoch == 0
    assert result.best_validation_loss == result.initial_validation_loss == 1.0
    for name, parameter in initial.state_dict().items():
        torch.testing.assert_close(parameter, result.model.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("parameter,value", [
    ("epochs", 0), ("epochs", 10_001), ("epochs", 1.5), ("batch_size", 0),
    ("batch_size", 65_537), ("seed", -1), ("seed", 2**32), ("seed", True),
    ("learning_rate", 0), ("learning_rate", float("nan")), ("temperature", 0),
    ("temperature", float("inf")), ("device", "meta"),
    ("risk_weight", -1), ("risk_weight", float("nan")), ("risk_weight", float("inf")),
    ("risk_weight", True),
])
def test_bad_training_options_fail(parameter, value):
    _, features, costs = _examples(6)
    options = {"epochs": 1, parameter: value}
    with pytest.raises(ValueError):
        train_selector(features, costs, features, costs, **options)


@pytest.mark.parametrize("position,bad", [
    (0, np.zeros((0, 2, 33))), (0, np.zeros((6, 33))),
    (0, np.full((6, 2, 33), np.nan)), (1, np.full((6, 3), -1.0)),
    (1, np.full((6, 3), np.inf)), (1, np.zeros((5, 3))),
    (2, np.zeros((6, 2, 32))), (3, np.zeros((6, 2))),
    (3, np.zeros((6, 0))),
])
def test_bad_training_arrays_fail(position, bad):
    _, features, costs = _examples(6)
    arrays = [features, costs, features, costs]
    arrays[position] = bad
    with pytest.raises(ValueError):
        train_selector(*arrays, epochs=1)


def test_cuda_unavailable_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _, features, costs = _examples(6)
    with pytest.raises(ValueError, match="CUDA.*unavailable"):
        train_selector(features, costs, features, costs, device="cuda", epochs=1)


def test_unavailable_cuda_index_fails_before_allocation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    _, features, costs = _examples(6)
    with pytest.raises(ValueError, match="index is unavailable"):
        train_selector(features, costs, features, costs, device="cuda:1", epochs=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="실제 CUDA 장치 필요")
def test_cuda_training_fp32_finite_and_repeatable():
    _, features, costs = _examples(12)
    rng_state = torch.cuda.get_rng_state().clone()
    first = train_selector(features, costs, features, costs, epochs=3,
                           batch_size=6, seed=7, device="cuda:0")
    second = train_selector(features, costs, features, costs, epochs=3,
                            batch_size=6, seed=7, device="cuda:0")
    assert first.history == second.history
    assert torch.equal(rng_state, torch.cuda.get_rng_state())
    for name, parameter in first.model.state_dict().items():
        assert parameter.device.type == "cuda" and parameter.dtype == torch.float32
        assert torch.isfinite(parameter).all()
        torch.testing.assert_close(parameter, second.model.state_dict()[name], rtol=0, atol=0)
    assert all(np.isfinite(value) for row in first.history for value in row.values())


def test_training_fp32_even_with_different_default_dtype():
    _, features, costs = _examples(6)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        result = train_selector(features, costs, features, costs, epochs=1)
    finally:
        torch.set_default_dtype(previous)
    assert all(parameter.dtype == torch.float32 for parameter in result.model.parameters())


def test_nonfinite_logits_and_gradients_are_rejected(monkeypatch):
    _, features, costs = _examples(6)
    original_forward = RefSpectrumSelector.forward
    monkeypatch.setattr(RefSpectrumSelector, "forward", lambda self, x: original_forward(self, x) * np.nan)
    with pytest.raises(FloatingPointError, match="logits"):
        train_selector(features, costs, features, costs, epochs=1)
    monkeypatch.setattr(RefSpectrumSelector, "forward", original_forward)
    original_init = RefSpectrumSelector.__init__

    def bad_gradient(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        next(self.parameters()).register_hook(lambda gradient: gradient * np.nan)

    monkeypatch.setattr(RefSpectrumSelector, "__init__", bad_gradient)
    with pytest.raises(FloatingPointError, match="gradient"):
        train_selector(features, costs, features, costs, epochs=1)


def test_inference_is_ref_only_preserves_mode_and_checks_shape():
    model = RefSpectrumSelector(3, 33)
    reference, _, _ = _examples(70)
    result = predict_selector(model, reference, n_fft=64)
    assert result.shape == (70,) and model.training
    np.testing.assert_array_equal(predict_selector(model, reference, n_fft=64, device="cpu:0"), result)
    assert set(inspect.signature(predict_selector).parameters) == {"model", "reference", "n_fft", "device"}
    assert not any("test" in key for key in inspect.signature(train_selector).parameters)
    with pytest.raises(ValueError, match="features"):
        predict_selector(model, reference, n_fft=32)
    assert model.training


def test_nonfinite_inference_is_rejected():
    model = RefSpectrumSelector(3, 33)
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="inference logits"):
        predict_selector(model, np.ones(128), n_fft=64)
    assert model.training


def test_asymmetric_regret_penalizes_risky_logit_more_than_ce():
    costs = np.array([[0.0, 1.0, 100.0]])
    targets = selector._targets(costs, 0.05)
    logits = torch.zeros((1, 3), requires_grad=True)
    cross_entropy, regret = selector._loss_terms(logits, targets, selector._regrets(costs))
    ce_gradient = torch.autograd.grad(cross_entropy.mean(), logits, retain_graph=True)[0]
    risk_gradient = torch.autograd.grad(cross_entropy.mean() + regret.mean(), logits)[0]
    # CE는 두 오답의 심각도를 거의 같게 보지만 실제 비용은 100배 차이다.
    torch.testing.assert_close(ce_gradient[0, 1], ce_gradient[0, 2], atol=1e-7, rtol=0)
    assert risk_gradient[0, 2] > risk_gradient[0, 1]
    regrets = selector._regrets(costs)
    after_ce = (torch.softmax(logits.detach() - 0.01 * ce_gradient, dim=1) * regrets).sum()
    after_risk = (torch.softmax(logits.detach() - 0.01 * risk_gradient, dim=1) * regrets).sum()
    assert after_risk < after_ce


def test_risk_zero_keeps_ce_objective_and_reports_regret():
    _, features, costs = _examples(6)
    result = train_selector(features, costs, features, costs, epochs=2, seed=6, risk_weight=0)
    assert all(row["risk_weight"] == 0 for row in result.history)
    assert all(row["train_expected_regret"] >= 0 and row["validation_expected_regret"] >= 0
               for row in result.history)
    logits = torch.zeros((2, 3), requires_grad=True)
    targets = selector._targets(costs[:2], 0.05)
    cross_entropy, regret = selector._loss_terms(logits, targets, selector._regrets(costs[:2]))
    expected = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
    actual = cross_entropy.mean() + 0.0 * regret.mean()
    assert torch.equal(actual, expected)
    expected_gradient = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, logits)[0]
    assert torch.equal(actual_gradient, expected_gradient)


def test_nonfinite_expected_regret_is_not_clipped():
    with pytest.raises(FloatingPointError, match="expected regret"):
        selector._loss_terms(torch.zeros(1, 2), torch.ones(1, 2) / 2,
                             torch.full((1, 2), float("inf")))
