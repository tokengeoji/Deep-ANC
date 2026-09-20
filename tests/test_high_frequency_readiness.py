"""학습을 실행하지 않는 정적 계약·파일 보존·test 노출 회귀."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from deep_anc.eval import high_frequency_readiness as readiness


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/high_frequency_comparison.json").read_text())


@pytest.fixture
def inputs(tmp_path, config):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    splits = {}
    for index, split in enumerate(("train", "validation", "test")):
        path = corpus / f"{index}.flac"
        path.write_bytes(b"static file existence fixture, not decodable audio" + bytes([index]))
        splits[split] = [{"path": path.name, "source_id": f"source-{index}", "group_id": f"group-{index}",
                          "speaker": str(index), "book": str(index), "split": split,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "sfanc_librispeech_sources.v1", "root": str(corpus),
        "sample_rate": 16000, "simulation_pretraining_only": True, "measured_ref_err": False, "splits": splits}))
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "report.json").write_text(json.dumps({"datasets": {"test": splits["test"]}}))
    config["sfanc_manifest"] = str(manifest)
    config["observed_run_directories"] = [str(previous)]
    configuration = tmp_path / "config.json"
    configuration.write_text(json.dumps(config))
    return configuration, manifest, corpus, previous


def test_provisional_contract_keeps_nulls(config):
    original = copy.deepcopy(config)
    assert readiness.validate_contract(config) == original
    assert all(config[key] is None for key in readiness.DECISIONS)


@pytest.mark.parametrize("change", [{}, {"unknown": 1}, {"sample_rate": 48000}, {"stage": "train"},
    {"polarity": "e=d-S*u"}, {"additional_delay_samples": True}, {"control_limit": 0},
    {"minimum_high_band_advantage_db": float("nan")}, {"candidate_selection": []},
    {"candidate_selection": ["wavenet"]}, {"observed_run_directories": "run"}])
def test_bad_contract_fails_closed(config, change):
    if not change:
        config = {}
    else:
        config.update(change)
    with pytest.raises(ValueError):
        readiness.validate_contract(config)


def test_static_report_never_runs_training_and_preserves_inputs(inputs, tmp_path):
    configuration, manifest, corpus, previous = inputs
    originals = {str(p): p.read_bytes() for p in (configuration, manifest, previous / "report.json", *corpus.iterdir())}
    result = readiness.prepare_high_frequency(configuration, tmp_path / "out")
    assert result["static_preparation_completed"]
    for name in ("training_ready", "training_executed", "optimizer_created", "model_instantiated",
                 "forward_backward_executed", "bank_fitting_executed", "audio_devices_opened", "deployment_allowed"):
        assert result[name] is False
    assert result["secondary"]["taps"] == 500
    assert result["secondary"]["all_taps_gain_sign_delay_preserved"]
    assert result["data"]["prior_observed_test_in_current_test"] == 1
    assert result["data"]["cross_split_overlap_count"] == 0
    assert not result["data"]["audio_decoded"]
    assert result["candidates"]["hybrid_ancnet"]["legacy_training_sample_rate"] == 48000
    assert not result["candidates"]["causal_controller"]["is_hybrid_ancnet"]
    assert not result["candidates"]["hybrid_ancnet"]["required_for_sfanc_training"]
    assert result["candidates"]["sfanc"]["paired_training_bridge_exists"]
    assert not result["candidates"]["sfanc"]["comparison_contract_adapter_required"]
    assert result["classical_baselines"]["validation_search_runner_exists"]
    assert result["classical_baselines"]["band_and_session_metrics_exists"]
    assert not result["classical_baselines"]["strong_tuning_ready"]
    assert result["protocol"]["mimii_usage"].startswith("train_only")
    assert "measured_ref_err_not_verified" in {row["code"] for row in result["physical_claim_blockers"]}
    assert result == json.loads((tmp_path / "out/report.json").read_text())
    assert {name: Path(name).read_bytes() for name in originals} == originals


def test_file_missing_and_split_leakage_are_blockers(inputs, tmp_path):
    configuration, manifest, corpus, _ = inputs
    value = json.loads(manifest.read_text())
    value["splits"]["test"][0]["path"] = "missing.flac"
    value["splits"]["test"][0]["group_id"] = value["splits"]["train"][0]["group_id"]
    manifest.write_text(json.dumps(value))
    result = readiness.prepare_high_frequency(configuration, tmp_path / "out")
    codes = {row["code"] for row in result["blockers"]}
    assert {"source_files_missing", "source_split_leakage"} <= codes


def test_missing_manifest_and_prior_history_are_not_pass(config, tmp_path):
    config["sfanc_manifest"] = "missing"
    config["observed_run_directories"] = ["missing"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = readiness.prepare_high_frequency(path, tmp_path / "out", repository_root=tmp_path)
    assert {"source_manifest_missing", "prior_test_history_incomplete"} <= {r["code"] for r in result["blockers"]}
    assert not result["classical_baselines"]["plain_and_normalized_module_exists"]
    assert not result["classical_baselines"]["validation_search_runner_exists"]
    assert not result["candidates"]["sfanc"]["paired_training_bridge_exists"]
    assert result["candidates"]["sfanc"]["comparison_contract_adapter_required"]


def test_existing_output_rejected_before_config_load(tmp_path, monkeypatch):
    monkeypatch.setattr(readiness, "_json", lambda *_: pytest.fail("기존 출력이면 읽기 시작 금지"))
    with pytest.raises(FileExistsError):
        readiness.prepare_high_frequency("missing", tmp_path)


def test_optional_files_are_not_capture_or_license_certification(inputs, tmp_path):
    configuration, manifest, _, _ = inputs
    result = readiness.prepare_high_frequency(configuration, tmp_path / "out", capture=manifest, inventory=manifest)
    for name in ("capture_file_check", "inventory_file_check"):
        assert result[name]["file_exists"]
        assert len(result[name]["sha256"]) == 64
        assert result[name]["contents_verified"] is False
    assert not result["training_ready"]


@pytest.mark.parametrize("mutate", ["escape", "unknown_schema", "empty_split"])
def test_invalid_manifest_refused(inputs, tmp_path, mutate):
    configuration, manifest, _, _ = inputs
    value = json.loads(manifest.read_text())
    if mutate == "escape":
        value["splits"]["test"][0]["path"] = "../outside.flac"
    elif mutate == "unknown_schema":
        value["schema"] = "unknown"
    else:
        value["splits"]["test"] = []
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        readiness.prepare_high_frequency(configuration, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema": "one", "schema": "two"}')
    with pytest.raises(ValueError, match="중복"):
        readiness._json(path)


@pytest.mark.parametrize("content", ["[]", "null", '"text"'])
def test_non_object_json_rejected(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="객체"):
        readiness._json(path)


def test_missing_optional_file_remains_blocker(inputs, tmp_path):
    configuration, _, _, _ = inputs
    result = readiness.prepare_high_frequency(configuration, tmp_path / "out", inventory=tmp_path / "missing")
    assert "provided_inventory_file_missing" in {row["code"] for row in result["blockers"]}


@pytest.mark.parametrize("field", ["speaker", "book"])
def test_prior_test_identity_cannot_move_to_training_after_group_rename(inputs, tmp_path, field):
    configuration, manifest, _, _ = inputs
    value = json.loads(manifest.read_text())
    value["splits"]["train"][0][field] = "2"
    value["splits"]["test"][0][field] = "new-identity"
    manifest.write_text(json.dumps(value))
    result = readiness.prepare_high_frequency(configuration, tmp_path / "out")
    assert result["data"]["prior_observed_test_in_training_or_validation"] == 1
    assert "prior_test_group_moved_into_training_or_validation" in {r["code"] for r in result["blockers"]}


def test_empty_history_identity_is_not_verified(inputs, tmp_path):
    configuration, _, _, previous = inputs
    (previous / "report.json").write_text(json.dumps({"datasets": {"test": [{}]}}))
    result = readiness.prepare_high_frequency(configuration, tmp_path / "out")
    assert not result["data"]["declared_history_readable"]
    assert "prior_test_history_incomplete" in {r["code"] for r in result["blockers"]}


def test_output_symlink_parent_rejected_before_read(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(readiness, "_json", lambda *_: pytest.fail("심볼릭 출력 경로는 읽기 전 거부"))
    with pytest.raises(ValueError, match="심볼릭"):
        readiness.prepare_high_frequency("missing", link / "out")
    assert not (real / "out").exists()


def test_metadata_size_limit_applies_before_hash(tmp_path, monkeypatch):
    path = tmp_path / "big"
    path.write_bytes(b"123456789")
    monkeypatch.setattr(readiness, "MAX_METADATA_BYTES", 8)
    monkeypatch.setattr(readiness, "_sha", lambda *_: pytest.fail("대용량 원본 해시하지 않음"))
    with pytest.raises(ValueError, match="메타데이터"):
        readiness._file_check(path)


def test_no_ml_import_or_execution_in_static_module():
    source = ROOT / "src/deep_anc/eval/high_frequency_readiness.py"
    tree = ast.parse(source.read_text())
    imported = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imported += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any("torch" in name or "train" in name or "scipy" in name for name in imported)
    forbidden = {"fit_control_fir", "design_bank", "Trainer", "run_experiment", "train_selector",
                 "HybridANCNet", "CausalController", "RefSpectrumSelector", "backward", "step"}
    calls = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
             for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
    assert not calls & forbidden


def test_cli_blocked_exit_and_no_artifacts_other_than_report(inputs, tmp_path):
    configuration, _, _, _ = inputs
    completed = subprocess.run([sys.executable, str(ROOT / "scripts/train/prepare_high_frequency.py"),
        "--config", str(configuration), "--out", str(tmp_path / "cli")], capture_output=True, text=True)
    assert completed.returncode == 2, completed.stderr
    assert "학습 미실행" in completed.stdout
    assert [p.name for p in (tmp_path / "cli").iterdir()] == ["report.json"]


def test_cli_unknown_flag_does_not_start_training(tmp_path):
    completed = subprocess.run([sys.executable, str(ROOT / "scripts/train/prepare_high_frequency.py"),
        "--config", "missing", "--out", str(tmp_path / "out"), "--train"], capture_output=True, text=True)
    assert completed.returncode == 2
    assert not (tmp_path / "out").exists()
