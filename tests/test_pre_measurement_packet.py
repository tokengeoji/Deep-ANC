"""실측 전 통합 패킷은 실제 수집·학습·탐색을 시작하지 않는다."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.prepare_before_measurement import prepare_before_measurement, ROOT
from scripts.eval import tune_classical_baselines as baseline_cli


def test_packet_creates_only_templates_and_preserves_secondary(tmp_path):
    source = ROOT / "rir.txt"
    original = source.read_bytes()
    out = tmp_path / "packet"
    report = prepare_before_measurement(out)
    assert report["packet_created"] and len(report["generated_files"]) == 9
    assert all((out / name).is_file() for name in report["generated_files"])
    for key in ("training_executed", "bank_fitting_executed", "validation_search_executed",
                "audio_devices_opened", "firmware_changed", "raw_audio_downloaded",
                "capture_ready", "training_ready", "physical_performance_claim_allowed"):
        assert report[key] is False
    assert not list(out.rglob("*.wav")) and not list(out.rglob("*.pt"))
    assert source.read_bytes() == original
    recipe = json.loads((out / "sfanc_recipe.template.json").read_text())
    assert recipe["additional_delay_samples"] is None
    assert recipe["cost"]["band_weights"]["remaining_high"] is None
    commands = json.loads((out / "commands.json").read_text())
    assert not commands["executed"] and not commands["actual_training_or_validation_search_authorized"]


def test_generated_baseline_template_is_incomplete_not_runnable(tmp_path):
    out = tmp_path / "packet"
    prepare_before_measurement(out)
    result = baseline_cli.execute_plan(out / "baseline_search.template.json", tmp_path / "check")
    assert not result["static_plan_complete"] and not result["search_executed"]
    with pytest.raises(ValueError, match="미확정"):
        baseline_cli.execute_plan(out / "baseline_search.template.json", tmp_path / "forbidden",
                                  run_validation_search=True)
    assert not (tmp_path / "forbidden").exists()


def test_source_inventory_is_reference_not_audio_certification(tmp_path):
    inventory = tmp_path / "inventory.json"
    inventory.write_text('{"metadata_verified": true, "data_ready": false}')
    report = prepare_before_measurement(tmp_path / "packet", source_inventory=inventory)
    evidence = report["public_source_policy"]["source_inventory"]
    assert evidence["provided"] and not evidence["content_certified"]
    assert len(evidence["sha256"]) == 64
    assert report["public_source_policy"]["public_audio_is_not_measured_ref_err"]


def test_invalid_inventory_rejected_before_any_packet_write(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ValueError):
        prepare_before_measurement(tmp_path / "out", source_inventory=missing)
    assert not (tmp_path / "out").exists()


def test_existing_and_symlink_outputs_not_changed(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(ValueError):
        prepare_before_measurement(tmp_path)
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        prepare_before_measurement(tmp_path / "link/out")
    assert marker.read_text() == "keep"


def test_cli_is_directly_executable_without_training_option(tmp_path):
    completed = subprocess.run([sys.executable, str(ROOT / "tools/prepare_before_measurement.py"),
                                "--out", str(tmp_path / "cli")], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "학습·측정·튜닝 미실행" in completed.stdout
    rejected = subprocess.run([sys.executable, str(ROOT / "tools/prepare_before_measurement.py"),
                              "--out", str(tmp_path / "train"), "--train"], capture_output=True, text=True)
    assert rejected.returncode == 2 and not (tmp_path / "train").exists()
