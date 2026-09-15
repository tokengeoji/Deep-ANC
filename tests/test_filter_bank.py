"""합성 bank 재현/출처·분리·REF-only 인과성·누락/증폭·무출력 CLI 회귀."""

from dataclasses import replace
import csv
import importlib.util
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from deep_anc.baselines import filter_bank as fb


SMALL = dict(sample_rate=8000, hop=32, secondary_delay_samples=32, stress_preview_samples=5,
             control_length=8, train_blocks=128, eval_blocks=32, window_blocks=4,
             selection_interval_blocks=8, train_seeds=(1, 2), validation_seeds=(10, 11),
             test_seeds=(20, 21), evaluation_levels=(0.04,), saturation_levels=(0.0,))
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench" / "prepare_filter_bank.py"


@pytest.fixture(scope="module")
def config():
    return fb.BankConfig(**SMALL)


@pytest.fixture(scope="module")
def bank(config):
    return fb.build_synthetic_bank(config, "long_delay_stress")


@pytest.fixture(scope="module")
def experiment(config):
    return fb.prepare_and_evaluate(config)


def test_train_candidates_do_not_depend_on_validation_or_test(config, bank):
    changed = replace(config, validation_seeds=(30,), test_seeds=(40,))
    rebuilt = fb.build_synthetic_bank(changed, "long_delay_stress")
    assert len(bank.candidates) == len(fb.TRAIN_FAMILIES) * len(config.train_seeds)
    assert bank.context_id == rebuilt.context_id
    assert bank.candidates == rebuilt.candidates
    assert bank.bank_id != rebuilt.bank_id  # 분리 프로토콜 메타도 hash로 보존한다.
    assert {c.train_seed for c in bank.candidates} == set(config.train_seeds)
    assert config.controller_configuration["mu"] == 0.05


@pytest.mark.parametrize("override", [
    {"validation_seeds": (1,)}, {"train_seeds": (1, 1)}, {"test_seeds": ()},
    {"sample_rate": 3200}, {"hop": 0}, {"eval_blocks": 1},
    {"stress_preview_samples": 64}, {"evaluation_levels": (float("nan"),)},
    {"saturation_levels": (-1.0,)}, {"mu": True},
    {"evaluation_levels": (1e40,)}, {"evaluation_levels": (float(np.finfo(np.float64).max),)},
    {"saturation_levels": (1e-300,)}, {"saturation_levels": (1e40,)},
])
def test_invalid_configuration_or_seed_overlap_is_rejected(override):
    with pytest.raises(ValueError):
        fb.BankConfig(**(SMALL | override))


def test_condition_specific_training_and_source_sha(experiment, config):
    banks, report = experiment
    assert set(banks) == set(fb.SCENARIOS)
    assert banks["causal_toy"].context_id == banks["long_delay_stress"].context_id
    assert banks["causal_toy"].bank_id != banks["long_delay_stress"].bank_id
    for candidate in banks["causal_toy"].candidates:
        source = fb.synthetic_reference(config, candidate.source_family, candidate.train_seed, "train")
        assert fb._sha(source.tobytes()) == candidate.reference_sha256
    assert not report["real_data_used"] and not report["speech_music_evaluated"]
    assert not report["physical_claim_allowed"] and not report["performance_claim_allowed"]
    assert not report["split_protocol"]["validation_tuning_used"]
    assert len(report["runs"]) == 2 * 4 * 7 * 4
    assert all(event["observed_until_sample"] <= event["apply_at_sample"]
               for run in report["runs"] for event in run["events"])
    assert all(row["adapted_blocks"] == 0 for row in report["runs"] if row["variant"] == "fixed_fir")


def test_artifact_roundtrip_reproduces_features_coefficients_and_identity(bank, tmp_path):
    target = tmp_path / "bank"
    fb.save_bank(bank, target)
    restored = fb.load_bank(target, expected_context_id=bank.context_id)
    assert restored == bank
    with pytest.raises(FileExistsError):
        fb.save_bank(bank, target)
    with pytest.raises(ValueError, match="context"):
        fb.load_bank(target, expected_context_id="0" * 64)


def test_bank_candidate_copies_mutable_inputs(bank):
    weights = np.asarray(bank.candidates[0].coefficients, dtype=np.float32)
    copied = replace(bank.candidates[0], coefficients=weights)
    before = copied.coefficients
    weights[:] = 100
    assert isinstance(copied.coefficients, tuple) and copied.coefficients == before
    with pytest.raises(ValueError):
        replace(copied, feature=(float("nan"),) * 6)


def _rewrite_artifact(path, *, metadata_change=None, array_change=None, resign=False):
    metadata = json.loads((path / "bank.json").read_text())
    with np.load(path / "bank.npz", allow_pickle=False) as arrays:
        coefficients, features = arrays["coefficients"].copy(), arrays["features"].copy()
    if array_change:
        coefficients, features = array_change(coefficients, features)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, coefficients=coefficients, features=features)
    raw = buffer.getvalue()
    (path / "bank.npz").write_bytes(raw)
    metadata["npz_sha256"] = fb._sha(raw)
    if metadata_change:
        metadata_change(metadata)
    if resign:
        logical = {key: value for key, value in metadata.items() if key not in ("bank_id", "npz_sha256")}
        metadata["bank_id"] = fb._bank_id(logical, coefficients, features)
    (path / "bank.json").write_text(json.dumps(metadata))


@pytest.mark.parametrize("kind", ["file_hash", "shape", "nan", "logical_hash", "source_hash", "decoder", "schema", "split", "truncated"])
def test_loader_rejects_corrupt_unknown_or_wrong_provenance(bank, tmp_path, kind):
    target = tmp_path / kind
    fb.save_bank(bank, target)
    if kind == "file_hash":
        with (target / "bank.npz").open("ab") as handle:
            handle.write(b"broken")
    elif kind == "shape":
        _rewrite_artifact(target, array_change=lambda c, f: (c[:, :-1], f))
    elif kind == "nan":
        def bad(c, f):
            c[0, 0] = np.nan
            return c, f
        _rewrite_artifact(target, array_change=bad)
    elif kind == "logical_hash":
        _rewrite_artifact(target, array_change=lambda c, f: (c * np.float32(0.5), f))
    elif kind == "source_hash":
        _rewrite_artifact(target, metadata_change=lambda m: m["candidates"][0].update(reference_sha256="0" * 64), resign=True)
    elif kind == "decoder":
        _rewrite_artifact(target, metadata_change=lambda m: m.update(decoder="unknown"), resign=True)
    elif kind == "schema":
        _rewrite_artifact(target, metadata_change=lambda m: m.update(schema="unknown"))
    elif kind == "split":
        _rewrite_artifact(target, metadata_change=lambda m: m["config"].update(validation_seeds=[1]))
    else:
        (target / "bank.json").write_text("{")
    with pytest.raises(ValueError):
        fb.load_bank(target, expected_context_id=bank.context_id)


def _observe_until_choice(selector, value=0.04):
    decision = selector.choose(generation=selector.generation, consumed_until_sample=selector.sample_cursor)
    while decision.reason == "window_or_interval_wait":
        selector.observe(np.full(selector.bank.config.hop, value, dtype=np.float32),
                         end_sample_exclusive=selector.sample_cursor + selector.bank.config.hop)
        decision = selector.choose(generation=selector.generation, consumed_until_sample=selector.sample_cursor)
    return decision


def test_selector_has_no_error_target_or_future_input_and_retains_same_index(bank):
    assert set(inspect.signature(fb.PastRefSelector.choose).parameters) == {"self", "generation", "consumed_until_sample"}
    selector = fb.PastRefSelector(bank)
    first = _observe_until_choice(selector)
    assert first.proposal.observed_until_sample == selector.sample_cursor
    with pytest.raises(RuntimeError, match="acknowledge"):
        selector.choose(generation=0, consumed_until_sample=selector.sample_cursor)
    selector.acknowledge(first, accepted=True)
    same = _observe_until_choice(selector)
    assert same.proposal is None and same.reason == "same_candidate_hold"
    assert same.candidate_id == first.candidate_id
    with pytest.raises(ValueError):
        selector.choose(generation=0, consumed_until_sample=selector.sample_cursor - bank.config.hop)
    with pytest.raises(ValueError):
        selector.choose(generation=True, consumed_until_sample=selector.sample_cursor)


def test_rejected_proposal_is_not_remembered_as_installed_and_reset_clears_history(bank):
    selector = fb.PastRefSelector(bank)
    first = _observe_until_choice(selector)
    selector.acknowledge(first, accepted=False)
    second = _observe_until_choice(selector)
    assert second.proposal is not None and second.proposal.revision > first.proposal.revision
    selector.acknowledge(second, accepted=True)
    selector.reset(generation=1)
    assert selector.sample_cursor == 0
    assert selector.choose(generation=1, consumed_until_sample=0).reason == "window_or_interval_wait"
    with pytest.raises(ValueError):
        selector.choose(generation=0, consumed_until_sample=0)


def test_future_suffix_changes_neither_earlier_output_nor_selection(bank, config):
    reference = fb.synthetic_reference(config, "switch", 10, "validation")
    changed = reference.copy()
    boundary = 16 * config.hop
    changed[boundary:] *= -1
    disturbance = np.zeros_like(reference)
    future_error = disturbance.copy()
    future_error[boundary:] = 0.1
    first = fb._rollout(config, bank, reference, disturbance, "selected_fxnlms", 0)
    second = fb._rollout(config, bank, changed, future_error, "selected_fxnlms", 0)
    np.testing.assert_array_equal(first["control"][:boundary], second["control"][:boundary])
    assert [e for e in first["events"] if e["apply_at_sample"] < boundary] == [e for e in second["events"] if e["apply_at_sample"] < boundary]


def test_invalid_observation_and_silence_do_not_produce_candidates(bank):
    selector = fb.PastRefSelector(bank)
    with pytest.raises(ValueError):
        selector.observe(np.full(bank.config.hop, np.nan), end_sample_exclusive=bank.config.hop)
    with pytest.raises(ValueError):
        selector.observe(np.zeros(bank.config.hop), end_sample_exclusive=float(bank.config.hop))
    assert selector.sample_cursor == 0
    assert _observe_until_choice(selector, 0).reason == "unavailable_reference_power"


def test_missing_and_amplified_bands_are_not_removed(experiment):
    _, report = experiment
    silence = [r for r in report["metrics"] if r["source_family"] == "silence"]
    assert silence and all(r["reduction_db"] is None for r in silence)
    assert any(r["amplified"] is True for r in report["metrics"])
    assert any(r["unavailable_rows"] > 0 for r in report["aggregate"])
    for row in report["metrics"]:
        if row["variant"] == "zero_control" and row["comparison_available"]:
            assert row["reduction_db"] == 0.0


def test_aggregate_distinguishes_failure_silence_emergence_and_finite_amplification(config):
    """정의할 수 없는 dB를 0으로 바꾸지 않고 실패/신규 에너지를 별도 센다."""
    description = dict(scenario="causal_toy", split="test", variant="selected_fxnlms")
    zero = np.zeros(config.eval_blocks * config.hop, dtype=np.float32)
    signal = np.full_like(zero, 0.1)
    rows = []
    rows.extend(fb._metric_rows(config, description, zero, {"error": zero, "control": zero}))
    rows.extend(fb._metric_rows(config, description, zero, {"error": signal, "control": zero}))
    rows.extend(fb._metric_rows(config, description, signal, {"error": 2 * signal, "control": zero}))
    rows.extend(fb._metric_rows(config, description, signal, None, failure=""))
    summary = next(row for row in fb._aggregate(rows) if row["band"] == "fullband")
    assert summary["total_rows"] == 4
    assert summary["available_rows"] == 1 and summary["unavailable_rows"] == 3
    assert summary["emergent_rows"] == 1 and summary["failed_rows"] == 1
    assert summary["amplified_rows"] == 1
    assert summary["worst_db"] == pytest.approx(-20 * np.log10(2), abs=1e-6)
    assert summary["statistics_scope"] == "available_finite_db_rows_only"
    assert all(row["unavailable_reason"] == "unspecified_run_failure" for row in rows if row["run_failed"])
    json.dumps({"rows": rows, "aggregate": fb._aggregate(rows)}, allow_nan=False)


def test_extreme_representable_reference_levels_keep_report_json_finite(config):
    """실기 의미가 없는 극단값의 직렬화 회귀이며 유효 음압/성능 시험이 아니다."""
    extreme = replace(config, evaluation_levels=(fb.MAX_EVALUATION_LEVEL,),
                      train_seeds=(1,), validation_seeds=(10,), test_seeds=(20,))
    for family in fb.EVAL_FAMILIES:
        reference = fb.synthetic_reference(extreme, family, 20, "test", level=fb.MAX_EVALUATION_LEVEL)
        assert np.isfinite(reference).all()
    _, report = fb.prepare_and_evaluate(extreme)
    json.dumps(report, allow_nan=False)
    assert any(row["failed_rows"] > 0 for row in report["aggregate"])
    assert all(run["status"] == "complete" for run in report["runs"] if run["variant"] == "zero_control")
    assert not report["physical_claim_allowed"]


@pytest.mark.parametrize("level", [fb.MIN_SATURATION_LEVEL, fb.MAX_SATURATION_LEVEL])
def test_extreme_supported_saturation_does_not_create_nan_from_silence(config, level):
    bounded = replace(config, saturation_levels=(level,))
    plant = fb._Plant(bounded, saturation=level)
    for drive in (np.zeros(config.hop, dtype=np.float32),
                  np.full(config.hop, 0.1, dtype=np.float32),
                  np.zeros(config.hop, dtype=np.float32)):
        assert np.isfinite(plant.step(drive)).all()


def test_training_failure_preserves_all_missing_comparison_rows(config, monkeypatch):
    def fail(*_):
        raise fb.BankTrainingError([{"candidate_id": "low:seed=1", "status": "invalid", "reason": "clipping"}])
    monkeypatch.setattr(fb, "build_synthetic_bank", fail)
    banks, report = fb.prepare_and_evaluate(config)
    assert not banks and all(r["status"] == "invalid" for r in report["training"])
    missing = [r for r in report["metrics"] if r["variant"] in ("fixed_fir", "selected_fxnlms")]
    assert missing and all(r["unavailable_reason"] == "missing_invalid_training_bank" and r["error_power"] is None for r in missing)
    assert all(row["run_failed"] for row in missing)
    for row in report["aggregate"]:
        if row["variant"] in ("fixed_fir", "selected_fxnlms"):
            assert row["failed_rows"] == row["total_rows"]
            assert row["emergent_rows"] == 0


def test_actual_training_clipping_invalidates_bank_without_seed_retry(config, monkeypatch):
    class OverflowTrainer(fb.FxLMSController):
        def generate_block(self, reference):
            return np.ones_like(reference)

    monkeypatch.setattr(fb, "FxLMSController", OverflowTrainer)
    with pytest.raises(fb.BankTrainingError) as result:
        fb.build_synthetic_bank(config, "causal_toy")
    assert len(result.value.diagnostics) == len(fb.TRAIN_FAMILIES) * len(config.train_seeds)
    assert all(row["status"] == "invalid" and "clipping" in row["reason"] for row in result.value.diagnostics)


def test_cli_outputs_artifacts_without_audio_or_gpu_imports_and_refuses_existing(tmp_path):
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
    args = ["--out", str(tmp_path / "new")]
    for key, value in SMALL.items():
        args.append("--" + key.replace("_", "-"))
        args.extend(str(v) for v in value) if isinstance(value, tuple) else args.append(str(value))
    command = [sys.executable, "-c", code, str(SCRIPT), *args]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "new" / "report.json").read_text())
    assert len(report["artifacts"]) == 2
    with (tmp_path / "new" / "metrics.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(report["metrics"])
    for row, original_row in zip(csv_rows, report["metrics"]):
        assert row["reduction_db"] == ("null" if original_row["reduction_db"] is None else str(original_row["reduction_db"]))
    assert "physical_claim_allowed=false" in (tmp_path / "new" / "summary.md").read_text()
    original = (tmp_path / "new" / "report.json").read_bytes()
    assert subprocess.run(command, cwd=ROOT, capture_output=True, timeout=30).returncode == 1
    assert (tmp_path / "new" / "report.json").read_bytes() == original


@pytest.mark.parametrize("kind", ["existing", "broken", "ancestor", "escape"])
def test_cli_rejects_output_before_analysis(tmp_path, monkeypatch, kind):
    spec = importlib.util.spec_from_file_location("filter_bank_cli", SCRIPT)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "missing")
    target = {"existing": real, "broken": broken, "ancestor": link / "new", "escape": link / ".." / "escape"}[kind]
    monkeypatch.setattr(cli, "prepare_and_evaluate", lambda *_: pytest.fail("분석 호출 금지"))
    assert cli.main(["--out", str(target)]) == 1


def test_nonlinear_and_unseen_levels_are_explicit_stress(config):
    nonlinear = replace(config, evaluation_levels=(0.025, 0.06), saturation_levels=(0.03,),
                        validation_seeds=(10,), test_seeds=(20,))
    _, report = fb.prepare_and_evaluate(nonlinear)
    assert {r["level"] for r in report["runs"]} == {0.025, 0.06}
    assert {r["saturation_level"] for r in report["runs"]} == {0.03}
    assert not report["physical_claim_allowed"]
