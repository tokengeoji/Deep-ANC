"""SFANC 오프라인 준비/학습의 합성 통합 회귀. 음향 성능 승리를 요구하지 않는다."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from deep_anc.baselines.sfanc_design import score_control_filters
from deep_anc.train import sfanc_experiment as experiment


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def small_config():
    return experiment.ExperimentConfig(
        control_length=8, bank_train_samples=2048, window_samples=512, n_fft=64,
        secondary_delay_samples=1, primary_delay_samples=8,
        train_groups=1, validation_groups=1, test_groups=1, epochs=2, batch_size=8,
    )


@pytest.fixture(scope="module")
def completed(small_config, tmp_path_factory):
    output = tmp_path_factory.mktemp("sfanc_integration") / "run"
    report = experiment.run_experiment(small_config, output, device="cpu")
    return output, report


def test_small_cpu_run_and_exact_artifact_restore(completed, small_config):
    output, report = completed
    assert report == json.loads((output / "report.json").read_text())
    assert report["training"]["device"] == "cpu"
    assert len(report["training"]["history"]) == 2
    assert not report["deployment_allowed"]
    assert not report["physical_performance_claim_allowed"]
    assert not report["real_time_claim_allowed"]
    assert not report["protocol"]["continuous_stream_evaluated"]
    assert not report["protocol"]["transition_evaluated"]
    assert not report["protocol"]["test_used_for_checkpoint"]
    model, weights, artifact = experiment.load_selector(output)
    restored, restored_weights, restored_artifact = experiment.load_selector(output)
    assert weights.shape == (9, small_config.control_length)
    np.testing.assert_array_equal(weights, restored_weights)
    assert artifact["bank_sha256"] == restored_artifact["bank_sha256"]
    values = np.random.default_rng(4).normal(scale=.03, size=small_config.window_samples)
    feature = experiment.reference_features(values, n_fft=small_config.n_fft)
    with torch.no_grad():
        assert torch.equal(model(torch.from_numpy(feature)), restored(torch.from_numpy(feature)))
    for name, expected in report["artifacts"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected


def test_existing_output_is_rejected_before_recomputation(completed, small_config, monkeypatch):
    output, _ = completed
    before = (output / "report.json").read_bytes()
    monkeypatch.setattr(experiment, "design_bank", lambda *_: pytest.fail("기존 출력이면 필터 계산 금지"))
    with pytest.raises(FileExistsError):
        experiment.run_experiment(small_config, output, device="cpu")
    assert (output / "report.json").read_bytes() == before


@pytest.mark.parametrize("filename", ["bank.npz", "selector.pt"])
def test_artifact_hash_tampering_is_rejected(completed, tmp_path, filename):
    source, _ = completed
    changed = tmp_path / "changed"
    shutil.copytree(source, changed)
    with (changed / filename).open("ab") as handle:
        handle.write(b"damaged")
    with pytest.raises(ValueError, match="SHA"):
        experiment.load_selector(changed)


def test_group_disjointness_finite_costs_and_separate_low_high_metrics(completed):
    _, report = completed
    groups, hashes = {}, {}
    for split, rows in report["datasets"].items():
        assert len(rows) == 10  # 9개 음원 조건 + 명시적 무음 진단
        groups[split] = {row["group_id"] for row in rows}
        # 인공 무음은 모든 split에서 같은 0 배열인 의도된 경계 검사다.
        hashes[split] = {row["crop_sha256"] for row in rows if row["kind"] != "silence"}
        assert all(row["observed_until_sample"] == row["apply_at_sample"]
                   < row["scored_from_sample"] for row in rows)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert groups[left].isdisjoint(groups[right])
        assert hashes[left].isdisjoint(hashes[right])
    for split in ("validation", "test"):
        for result in report["evaluation"][split].values():
            assert np.isfinite(result["mean_cost"])
            assert np.isfinite(result["mean_regret"])
            assert result["mean_regret"] >= -1e-7
            assert sum(result["selection_counts"]) == len(report["datasets"][split])
            assert {row["band"] for row in result["metrics"]} == {
                "fullband", "below_1000", "target_1000_1600", "above_1600"}
            for row in result["metrics"]:
                assert np.isfinite(row["baseline_power"]) and np.isfinite(row["residual_power"])
                assert row["reduction_db"] is None or np.isfinite(row["reduction_db"])
    json.dumps(report, allow_nan=False)


def test_future_suffix_does_not_change_observed_features(small_config, monkeypatch):
    primary, secondary, _ = experiment.paths(small_config)
    rng = np.random.default_rng(28)
    first = rng.normal(scale=.02, size=2*small_config.window_samples)
    changed = first.copy()
    changed[small_config.window_samples:] = rng.normal(scale=.08, size=small_config.window_samples)
    weights = np.zeros((2, small_config.control_length))
    weights[1, 0] = .05

    def prepare(value):
        monkeypatch.setattr(experiment, "procedural_sources", lambda *_: iter([
            (value, {"source_id": "synthetic:fixed", "group_id": "independent_group", "split": "train"})]))
        return experiment.prepare_split(small_config, "train", weights, primary, secondary)

    left, right = prepare(first), prepare(changed)
    np.testing.assert_array_equal(left["features"], right["features"])
    np.testing.assert_array_equal(left["simple_features"], right["simple_features"])
    assert left["rows"][0]["crop_sha256"] != right["rows"][0]["crop_sha256"]
    assert np.isfinite(left["costs"]).all() and np.isfinite(right["costs"]).all()


def test_zero_output_in_observation_window_matches_scoring_after_complete_settling(small_config):
    primary, secondary, _ = experiment.paths(small_config)
    reference = np.random.default_rng(67).normal(scale=.02, size=2*small_config.window_samples)
    weights = np.array([.12, -.05, .03, .01, -.005, .003, -.002, .001])
    disturbance = np.convolve(reference, primary)[:len(reference)]
    warmup = max(len(primary)-1, len(secondary)-1 + small_config.secondary_delay_samples
                 + small_config.control_length-1)
    start = small_config.window_samples + warmup
    whole = score_control_filters(reference, disturbance, secondary, weights,
                                   secondary_delay_samples=small_config.secondary_delay_samples,
                                   warmup_samples=start)
    # 실제 첫 창의 출력을 0으로 놓은 독립 direct-convolution 기준이다.
    command = np.convolve(reference, weights)[:len(reference)]
    command[:small_config.window_samples] = 0
    contribution = np.convolve(command, secondary)[:len(reference)]
    delay = small_config.secondary_delay_samples
    if delay:
        contribution = np.pad(contribution, (delay, 0))[:len(reference)]
    error = disturbance + contribution
    np.testing.assert_allclose(whole["residual"][0, start:], error[start:], atol=1e-14, rtol=1e-12)
    assert whole["residual_mse"][0] == pytest.approx(np.mean(error[start:]**2), rel=1e-12)


def test_omap_path_keeps_all_authoritative_taps_and_source_hash(small_config, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from deepanc.calibration import DEFAULT_RIR, load_secondary_path
    before = DEFAULT_RIR.read_bytes()
    config = replace(small_config, secondary_kind="omap_measured", secondary_delay_samples=0,
                     window_samples=1024)
    _, secondary, provenance = experiment.paths(config)
    assert secondary.shape == (500,)
    np.testing.assert_array_equal(secondary, load_secondary_path(sample_rate=16000))
    assert provenance["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert provenance["preserve_all_taps_gain_sign_delay"] is True
    assert provenance["jetson_output_path_measured"] is False
    assert provenance["additional_delay_samples"] == 0
    assert provenance["primary_kind"].startswith("synthetic")
    assert DEFAULT_RIR.read_bytes() == before


def test_frequency_band_powers_add_to_fullband():
    values = np.random.default_rng(53).normal(size=(3, 1024))
    power = experiment._band_power(values, 16000)
    np.testing.assert_allclose(power["below_1000"] + power["target_1000_1600"] + power["above_1600"],
                               power["fullband"], atol=1e-14, rtol=1e-13)
