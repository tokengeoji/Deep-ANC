"""사전 FIR 합성 벤치마크의 분리·지연·대조군·무출력/무덮어쓰기 검증."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench" / "benchmark_prepared_fir.py"
SMALL = dict(sample_rate=8000, hop=32, secondary_delay_samples=64,
             stress_preview_samples=5, control_length=8, train_blocks=280,
             eval_blocks=128, train_seed=11, heldout_seed=101)


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("prepared_fir_bench", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report(bench):
    return bench.run_benchmark(**SMALL)


def metric(report, variant, *, scenario="causal_toy", family="white", period="early", band="fullband"):
    return next(row for row in report["metrics"] if row["variant"] == variant
                and row["scenario"] == scenario and row["source_family"] == family
                and row["period"] == period and row["band"] == band)


def arguments(out):
    args = ["--out", str(out)]
    for key, value in SMALL.items():
        args.extend(["--" + key.replace("_", "-"), str(value)])
    return args


def test_training_and_heldout_are_independent_and_not_a_selector(report, bench):
    assert report["training"]["scenario"] == "causal_toy"
    assert report["training"]["seed"] == 11
    assert report["training"]["seed"] not in report["heldout"]["family_seeds"].values()
    assert report["candidate_count"] == 1
    assert not report["selector_implemented"] and not report["cnn_trained"]
    assert report["synthetic_only"] and report["diagnostic_only"]
    assert not report["performance_claim_allowed"] and not report["real_time_claim_allowed"]
    assert not report["plant"]["measured_secondary_path_used"]
    assert len(report["metrics"]) == 2 * 5 * 4 * 4 * 4
    assert {row["band"] for row in report["metrics"]} >= {"below_800", "target_800_1600"}
    assert all(not row["performance_claim_allowed"] for row in report["metrics"])
    stress = report["scenarios"]["long_delay_stress"]
    assert stress["uses_same_candidate_without_retraining"]
    assert stress["primary_preview_samples"] != stress["training_primary_preview_samples"]
    assert any("불일치" in warning and "최적성" in warning for warning in report["warnings"])
    configuration = report["controller_configuration"]
    assert configuration["mu"] == 0.25 and configuration["leakage"] == 0.0
    assert configuration["prepared_l1_limit"] == configuration["residual_norm_limit"] == 2.0
    assert configuration["max_proposal_age_samples"] == SMALL["hop"]
    reconstructed = bench.PreparedFxNLMSController(
        np.asarray(report["plant"]["synthetic_secondary_fir"], dtype=np.float32), **configuration)
    assert reconstructed.context_id == report["context_id"]


def test_causal_warmstart_and_gain_change_have_separate_benefits(report):
    warm = metric(report, "prepared_fxnlms")["reduction_db"]
    cold = metric(report, "cold_fxnlms")["reduction_db"]
    assert warm > cold + 5.0
    corrected = metric(report, "prepared_fxnlms", period="steady")["reduction_db"]
    fixed = metric(report, "fixed_fir", period="steady")["reduction_db"]
    assert corrected > fixed + 2.0
    assert all(row["adapted_blocks"] == 0 for row in report["runs"] if row["variant"] == "fixed_fir")
    assert all(row["control_peak"] <= 0.100001 for row in report["metrics"])


def test_long_delay_white_noise_is_not_reported_as_broadband_success(report):
    row = metric(report, "prepared_fxnlms", scenario="long_delay_stress", period="steady")
    assert row["reduction_db"] < 1.0
    assert report["plant"]["effective_control_delay_samples"] == 96
    assert report["plant"]["stress_reference_preview_samples"] == 5
    assert any("주기" in warning and "white" in warning for warning in report["warnings"])


def test_zero_control_is_real_baseline_and_power_is_not_fabricated(report, bench):
    for row in report["metrics"]:
        if row["variant"] == "zero_control" and row["comparison_available"]:
            assert row["reduction_db"] == pytest.approx(0.0, abs=1e-12)
    zeros = np.zeros(64, dtype=np.float32)
    result = {"error": zeros.copy(), "control": zeros.copy()}
    rows = bench._metrics(zeros, zeros, result, {"silent": (0, 64)}, fs=8000,
                          scenario="synthetic", family="silent", variant="zero_control")
    assert all(row["reduction_db"] is None and not row["comparison_available"] for row in rows)
    result["error"] = np.ones(64, dtype=np.float32)
    rows = bench._metrics(zeros, zeros, result, {"silent": (0, 64)}, fs=8000,
                          scenario="synthetic", family="silent", variant="zero_control")
    assert rows[0]["emergent_error_energy"] and rows[0]["reduction_db"] is None


def test_simulated_control_impulse_has_exact_total_delay_and_polarity(bench):
    hop = 8
    plant = bench._Plant(np.array([-0.5, -0.08], dtype=np.float32), 19, hop)
    previous = np.zeros(hop, dtype=np.float32)
    errors = []
    for block in range(6):
        errors.extend(plant.step(previous))
        previous = np.zeros(hop, dtype=np.float32)
        if block == 0:
            previous[0] = 1.0
    assert errors[19] == pytest.approx(-0.5)
    assert errors[20] == pytest.approx(-0.08)
    assert np.count_nonzero(errors) == 2


def test_clipped_training_candidate_is_invalid_without_retry(bench):
    class UnsafeTrainer:
        def generate_block(self, reference):
            return np.ones_like(reference)

        def adapt_block(self, *args, **kwargs):
            pytest.fail("clipping 후 적응하면 안 됨")

    with pytest.raises(ValueError, match="후보 무효"):
        bench._rollout(np.zeros(64, dtype=np.float32), np.zeros(64, dtype=np.float32),
                       UnsafeTrainer(), hop=8, s_hat=np.array([-0.5], dtype=np.float32),
                       effective_delay=8, adapt=True, control_limit=0.1, training=True)


@pytest.mark.parametrize("override", [
    {"train_seed": 101}, {"train_seed": 103}, {"sample_rate": 3000},
    {"hop": 0}, {"secondary_delay_samples": -1}, {"stress_preview_samples": 96},
    {"eval_blocks": 4}, {"saturation_level": float("nan")},
])
def test_invalid_input_rejected_before_any_output(bench, override):
    with pytest.raises(ValueError):
        bench.run_benchmark(**(SMALL | override))


def test_cli_reports_same_json_csv_and_will_not_overwrite(bench, tmp_path):
    out = tmp_path / "new"
    assert bench.main(arguments(out)) == 0
    saved = json.loads((out / "report.json").read_text())
    with (out / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(saved["metrics"])
    for csv_row, json_row in zip(rows, saved["metrics"]):
        assert csv_row["reduction_db"] == ("null" if json_row["reduction_db"] is None else str(json_row["reduction_db"]))
    assert "performance_claim_allowed=false" in (out / "summary.md").read_text()
    before = (out / "report.json").read_bytes()
    assert bench.main(arguments(out)) == 1
    assert (out / "report.json").read_bytes() == before
    assert {path.suffix for path in out.iterdir()} == {".json", ".csv", ".md"}


@pytest.mark.parametrize("kind", ["existing", "symlink", "broken", "ancestor", "parent_escape"])
def test_output_rejection_never_runs_analysis(bench, tmp_path, monkeypatch, kind):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    paths = {"existing": real, "symlink": link, "broken": tmp_path / "broken",
             "ancestor": link / "new", "parent_escape": link / ".." / "escaped"}
    paths["broken"].symlink_to(tmp_path / "missing")
    monkeypatch.setattr(bench, "run_benchmark", lambda **_: pytest.fail("분석하면 안 됨"))
    assert bench.main(arguments(paths[kind])) == 1
    assert not (real / "new").exists() and not (tmp_path / "escaped").exists()


def test_subprocess_never_imports_audio_or_gpu_libraries(tmp_path):
    code = """
import importlib.abc, runpy, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'sounddevice', 'onnxruntime', 'tensorrt'} or fullname.startswith('deep_anc.audio_io'):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Guard())
script, *arguments = sys.argv[1:]
sys.argv = [script] + arguments
runpy.run_path(script, run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", code, str(SCRIPT), *arguments(tmp_path / "out")],
                            capture_output=True, text=True, cwd=ROOT, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "실제 감쇠/성능 통과 아님" in result.stdout


def test_nonlinear_stress_is_explicit_not_a_claim(bench):
    result = bench.run_benchmark(**(SMALL | {"saturation_level": 0.025}))
    assert result["plant"]["saturation_level"] == 0.025
    assert not result["performance_claim_allowed"]
    assert any("비선형" in warning and "증명" in warning for warning in result["warnings"])
