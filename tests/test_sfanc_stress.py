"""연속 SFANC의 인과 scheduler, plant 정렬, 진단 보고서 회귀."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import signal
import torch

from deep_anc.eval import sfanc_stress as stress
from deep_anc.train.sfanc_experiment import save_selector
from deep_anc.train.sfanc_selector import RefSpectrumSelector


@pytest.mark.parametrize("lag", [0, 1, 7, 32, 55])
@pytest.mark.parametrize("block", [1, 3, 8, 31])
def test_selection_sees_completed_past_and_never_blocks_audio(lag, block):
    x = np.arange(80, dtype=float) / 100
    seen = []
    def choose(past):
        seen.append(past.copy())
        return 1
    result = stress.selected_stream(x, [[0], [.1]], choose, window_samples=16,
                  block_samples=block, selector_delay_samples=lag, control_limit=None)
    expected = .1*x
    expected[:min(16+lag, len(x))] = 0
    np.testing.assert_allclose(result["control"], expected, atol=1e-15)
    assert len(seen) == 4
    for i, past in enumerate(seen):
        np.testing.assert_array_equal(past, x[16*i:16*(i+1)])
    assert all(row["apply_at_sample"] == row["observed_until_sample"]+lag
               for row in result["decisions"])


def test_future_suffix_cannot_change_prior_outputs_or_decisions():
    rng = np.random.default_rng(18)
    x = rng.normal(size=160)
    changed = x.copy()
    changed[91:] *= -100
    bank = np.array([[0, 0, 0], [.03, -.02, .01], [-.01, .02, .04]])
    def rollout(a, block):
        return stress.selected_stream(a, bank, lambda p: 1 if p.mean() > 0 else 2,
                   window_samples=32, block_samples=block, selector_delay_samples=9,
                   transition_samples=17, control_limit=.2)
    left, right = rollout(x, 3), rollout(changed, 11)
    np.testing.assert_allclose(left["control"][:91], right["control"][:91], atol=1e-14)
    assert [r for r in left["decisions"] if r["observed_until_sample"] <= 91] == [
           r for r in right["decisions"] if r["observed_until_sample"] <= 91]
    np.testing.assert_allclose(left["control"], rollout(x, 23)["control"], atol=1e-14)


def test_silence_uses_zero_without_calling_cnn():
    result = stress.selected_stream(np.zeros(160), [[0], [2]],
                lambda _: pytest.fail("무음은 선택기 호출 금지"), window_samples=32)
    assert not result["control"].any()
    assert all(event["candidate"] == 0 for event in result["decisions"])


@pytest.mark.parametrize("bank", [[], np.empty((0, 3)), [[1, 0]]])
def test_zero_candidate_validation(bank):
    with pytest.raises(ValueError):
        stress.selected_stream(np.ones(64), bank, lambda _: 0, window_samples=16)


@pytest.mark.parametrize("delay", [0, 1, 17, 100])
def test_delay_applied_exactly_once_and_polarity_preserved(delay):
    rng = np.random.default_rng(1)
    x, u = rng.normal(size=(2, 100))
    secondary = np.array([0., -.2, .5, -.1])
    full = np.convolve(u, secondary)
    expected = x + np.pad(full, (delay, 0))[:len(x)]
    np.testing.assert_allclose(stress.control_residual(x, u, secondary, delay), expected, atol=1e-13)


def test_clipping_is_before_secondary_path_not_on_error():
    x = np.ones(256)
    result = stress.selected_stream(x, [[0], [10]], lambda _: 1,
                       window_samples=32, control_limit=.2)
    assert result["diagnostics"]["limited_samples"] > 0
    e = stress.control_residual(np.zeros_like(x), result["control"], [2.], 3)
    assert e.max() == pytest.approx(.4)
    assert np.max(np.abs(result["control"])) <= .2


@pytest.mark.parametrize("n", [255, 256])
def test_fft_band_powers_match_parseval(n):
    x = np.random.default_rng(n).normal(size=n)
    rows = {r["band"]: r for r in stress.band_metrics(x, x*.5)}
    assert rows["fullband"]["baseline_power"] == pytest.approx(np.mean(x*x))
    assert rows["fullband"]["residual_power"] == pytest.approx(np.mean(x*x)/4)
    assert rows["fullband"]["reduction_db"] == pytest.approx(20*np.log10(2))
    assert sum(rows[b]["baseline_power"] for b in ("below_1000", "target_1000_1600", "above_1600")) == pytest.approx(np.mean(x*x))


def test_silent_baseline_reports_emergent_energy_not_fake_db():
    rows = stress.band_metrics(np.zeros(256), np.ones(256))
    assert all(r["reduction_db"] is None for r in rows)
    assert rows[0]["emergent_error_energy"]


@pytest.mark.parametrize("settings", [
    {"samples": 2050}, {"samples": True}, {"fit_samples": 1024},
    {"additional_delays": [-1]}, {"additional_delays": [0, 0]},
    {"primary_delays": []}, {"selector_delay_samples": -.5},
    {"control_limit": float("nan")}, {"control_limit": True},
    {"fxnlms_mu": 3}, {"block_samples": 0}])
def test_invalid_config_rejected(settings):
    with pytest.raises(ValueError):
        stress.StressConfig(**settings)


def test_sources_reproducible_no_actual_music_claim():
    config = stress.StressConfig(samples=2048)
    first = list(stress.diagnostic_sources(config))
    second = list(stress.diagnostic_sources(config))
    assert len(first) == 6
    for (name, x, meta), (other, y, _) in zip(first, second):
        assert name == other
        np.testing.assert_array_equal(x, y)
        assert not meta["music"] and not meta["measured_ref_err"]
    changed = list(stress.diagnostic_sources(replace(config, seed=config.seed+1)))
    assert not np.array_equal(first[0][1], changed[0][1])


@pytest.fixture
def artifact(tmp_path):
    from deepanc.calibration import load_secondary_path
    root = tmp_path/"artifact"
    root.mkdir()
    bank = np.zeros((2, 8))
    bank[1, 0] = -.01
    with (root/"bank.npz").open("xb") as stream:
        np.savez(stream, coefficients=bank, secondary=load_secondary_path(sample_rate=16000))
    bank_sha = hashlib.sha256((root/"bank.npz").read_bytes()).hexdigest()
    model = RefSpectrumSelector(2, 33)
    with torch.no_grad():
        for value in model.parameters():
            value.zero_()
    selector_sha = save_selector(model, root, bank_sha,
                         SimpleNamespace(n_fft=64, window_samples=512, sample_rate=16000))
    prior = {"config": {"secondary_kind": "omap_measured", "regularization": 1e-7, "effort_penalty": .002},
             "training": {"fixed_candidate_from_train": 1},
             "artifacts": {"bank.npz": bank_sha, "selector.pt": selector_sha}}
    (root/"report.json").write_text(json.dumps(prior))
    return root


def test_omap_small_end_to_end_report_and_no_clobber(artifact, tmp_path):
    config = stress.StressConfig(samples=2048, fit_samples=2048,
                                primary_delays=(128,), additional_delays=(3,))
    out = tmp_path/"diagnostic"
    report = stress.run_stress(artifact, out, config)
    saved = json.loads((out/"report.json").read_text())
    assert saved["schema"] == report["schema"]
    assert len(report["runs"]) == 30
    assert report["continuous_simulation_evaluated"]
    for name in ("deployment_allowed", "real_time_claim_allowed", "physical_performance_claim_allowed",
                 "audio_devices_opened", "music_evaluated", "model_retrained", "test_used_for_tuning"):
        assert report[name] is False
    assert report["hybrid_ancnet_attenuation_comparison"]["status"] == "not_evaluated"
    assert all(row["output_peak"] <= config.control_limit for row in report["runs"])
    assert {row["method"] for row in report["metrics"]} == {
        "zero", "frozen_train_best_fir", "frozen_sfanc",
        "scenario_fitted_fir_not_frozen", "cold_block_fxnlms"}
    before = (out/"report.json").read_bytes()
    with pytest.raises(FileExistsError):
        stress.run_stress(artifact, out, config)
    assert (out/"report.json").read_bytes() == before


def test_standalone_cli_without_pythonpath(artifact, tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = tmp_path/"config.json"
    config.write_text(json.dumps({"samples": 2048, "fit_samples": 2048,
                                  "primary_delays": [128], "additional_delays": [0]}))
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = tmp_path/"standalone"
    result = subprocess.run([sys.executable, str(root/"scripts/bench/benchmark_sfanc_stress.py"),
                "--run", str(artifact), "--out", str(out), "--config", str(config)],
                cwd=root, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert (out/"report.json").is_file()
