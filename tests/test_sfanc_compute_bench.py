"""계산시간 보고서 계약 회귀. 특정 지연/ANC 성능 통과값을 강제하지 않는다."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from deepanc.calibration import load_secondary_path
from deep_anc.eval import sfanc_compute as bench
from deep_anc.train.sfanc_selector import RefSpectrumSelector


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def artifact(tmp_path):
    """외부 로컬 run에 의존하지 않는 명시적 미학습 수치 fixture."""
    directory = tmp_path / "fixture"
    directory.mkdir()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4)
        model = RefSpectrumSelector(3, 33)
    weights = np.array([[0., 0.], [.1, -.01], [-.1, .01]])
    np.savez(directory / "bank.npz", coefficients=weights, secondary=load_secondary_path())
    bank_sha = hashlib.sha256((directory / "bank.npz").read_bytes()).hexdigest()
    torch.save({"schema": "sfanc_ref_spectrum_selector.v1", "bank_sha256": bank_sha,
        "candidate_count": 3, "bins": 33, "n_fft": 64, "window_samples": 256,
        "sample_rate": 16000, "deployment_allowed": False, "state_dict": model.state_dict()},
        directory / "selector.pt")
    report = {"config": {"secondary_kind": "omap_measured"}, "test_fixture_untrained": True,
        "artifacts": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                      for name in ("bank.npz", "selector.pt")}}
    (directory / "report.json").write_text(json.dumps(report))
    return directory


def test_summary_percentiles_and_rate_budgets():
    summary = bench.timing_summary([1, 2, 3, 4], samples=16)
    assert summary["median_ms"] == 2.5
    assert summary["p99_ms"] == pytest.approx(3.97)
    assert summary["max_ms"] == 4
    assert summary["duration_budget_comparison_only"]["16000"]["duration_ms"] == 1
    assert summary["duration_budget_comparison_only"]["16000"]["observed_calls_over_duration"] == 3
    assert summary["duration_budget_comparison_only"]["48000"]["duration_ms"] == pytest.approx(1/3)


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf], [-1], [[1]]])
def test_invalid_measurements(values):
    with pytest.raises(ValueError):
        bench.timing_summary(values, samples=16)


def test_cpu_report_contract_and_preservation(artifact, tmp_path):
    original = {p.name: p.read_bytes() for p in artifact.iterdir()}
    previous_threads = torch.get_num_threads()
    rng_state = torch.get_rng_state().clone()
    result = bench.benchmark_compute(artifact, tmp_path / "output", blocks=(16, 256),
                                     warmup=1, trials=2, torch_threads=1)
    assert torch.get_num_threads() == previous_threads
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert result == json.loads((tmp_path / "output/report.json").read_text())
    for field in ("audio_opened", "end_to_end_latency_measured", "anc_attenuation_compared",
                  "real_time_claim_allowed", "deployment_allowed", "components_run_concurrently",
                  "hybrid_trained_for_omap_16khz"):
        assert result[field] is False
    assert result["source"]["omap_secondary_exact"]
    assert result["source"]["hybrid_parameters"] == 1164809
    assert len(result["measurements"]) == 6
    assert all(m["trials"] == 2 and m["max_ms"] >= m["median_ms"] >= 0
               for m in result["measurements"])
    assert [m["mode"] for m in result["measurements"][:2]] == ["steady", "switch_each_block"]
    assert {p.name: p.read_bytes() for p in artifact.iterdir()} == original


def test_existing_output_rejected_before_read(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "_load_omap", lambda *_: pytest.fail("existing output: no load"))
    with pytest.raises(FileExistsError):
        bench.benchmark_compute("missing", tmp_path)


@pytest.mark.parametrize("kwargs", [{"trials": 0}, {"warmup": -1}, {"torch_threads": True},
    {"blocks": []}, {"blocks": [16, 16]}, {"blocks": [0]}, {"devices": ["cpu", "cpu"]},
    {"devices": ["mps"]}, {"seed": -1}])
def test_bad_options_rejected_before_read(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(bench, "_load_omap", lambda *_: pytest.fail("invalid options: no load"))
    with pytest.raises(ValueError):
        bench.benchmark_compute("missing", tmp_path / "out", **kwargs)
    assert not (tmp_path / "out").exists()


def test_cuda_unavailable_is_not_cpu_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        bench.benchmark_compute("missing", tmp_path / "out", devices=("cuda",))


def test_hash_failure_before_output(artifact, tmp_path):
    with (artifact / "bank.npz").open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError, match="SHA"):
        bench.benchmark_compute(artifact, tmp_path / "out", warmup=0, trials=1)
    assert not (tmp_path / "out").exists()


def test_secondary_changes_are_not_accepted(artifact):
    np.savez(artifact / "bank.npz", coefficients=np.array([[0.], [.1], [-.1]]),
             secondary=load_secondary_path()[1:])
    report = json.loads((artifact / "report.json").read_text())
    report["artifacts"]["bank.npz"] = hashlib.sha256((artifact / "bank.npz").read_bytes()).hexdigest()
    (artifact / "report.json").write_text(json.dumps(report))
    checkpoint = torch.load(artifact / "selector.pt", weights_only=True)
    checkpoint["bank_sha256"] = report["artifacts"]["bank.npz"]
    torch.save(checkpoint, artifact / "selector.pt")
    report["artifacts"]["selector.pt"] = hashlib.sha256((artifact / "selector.pt").read_bytes()).hexdigest()
    (artifact / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="전체 탭"):
        bench._load_omap(artifact)


def test_measure_warmup_and_cuda_synchronization(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d: calls.append("sync"))
    result = bench._measure(lambda: calls.append("op") or np.zeros(2),
                            warmup=2, trials=3, device=torch.device("cuda"))
    assert len(result) == 3
    assert calls == ["sync", "op", "sync"] * 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="실제 CUDA 필요; CPU 대체 금지")
def test_cuda_report_contract(artifact, tmp_path):
    result = bench.benchmark_compute(artifact, tmp_path / "cuda", devices=("cuda",),
                                     blocks=(16,), warmup=1, trials=1)
    gpu_rows = [row for row in result["measurements"] if row["device"] == "cuda"]
    assert len(gpu_rows) == 2
    assert result["environment"]["cuda_device"]
    assert not result["end_to_end_latency_measured"]


def test_cli_help_no_audio():
    completed = subprocess.run([sys.executable, str(ROOT / "scripts/bench/benchmark_sfanc_compute.py"),
                                "--help"], text=True, capture_output=True, check=True)
    assert "--trials" in completed.stdout
    assert "무오디오" in completed.stdout
