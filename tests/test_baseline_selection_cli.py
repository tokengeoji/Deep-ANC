"""baseline CLI의 기본 무실행·명시 승인·test 선검사 회귀."""
import importlib.util
import json
import os
from pathlib import Path
import wave

import numpy as np
import pytest
from deepanc import measurement_ready

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("tune_classical_cli", ROOT / "scripts/eval/tune_classical_baselines.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def write_json(path, data):
    path.write_text(json.dumps(data))
    return path


def complete_plan(tmp_path):
    # 3×16-frame 합성 intake fixture. 실녹음/모델/탐색을 실행하지 않는다.
    measurement_ready.create_packet(tmp_path / "capture")
    for index, split in enumerate(("train", "valid", "test")):
        folder = tmp_path / "capture/raw" / split / f"session_{split}_001"
        sidecar = folder / "capture.json"
        meta = json.loads(sidecar.read_text())
        for key in ("operator", "capture_utc", "board_model", "board_revision", "firmware_revision",
                    "capture_method", "clock_source"):
            meta[key] = "synthetic-unit-fixture"
        meta.update(source_family="noise", source_corpus="fixture", source_recording_id=f"source-{index}",
                    source_recording_ids=[f"parent-{index}"], source_group_ids=[f"group-{index}"], gain_profile_id="profile-A")
        meta["gains"] = {name: {"value": i, "unit": "dB"} for i, name in enumerate(measurement_ready.GAIN_NAMES)}
        meta["attestations"] = {name: True for name in measurement_ready.ATTESTATIONS}
        meta["recording_loss"] = {"dropped_frames": 0, "duplicate_frames": 0, "sequence_check_method": "fixture"}
        if split == "test":
            meta["final_test_not_used_for_tuning"] = True
        write_json(sidecar, meta)
        with wave.open(str(folder / "noise.wav"), "wb") as handle:
            handle.setparams((2, 2, 16000, 16, "NONE", "not compressed"))
            handle.writeframes((np.arange(32, dtype="<i2") + index * 100 + 1).tobytes())
    qa = measurement_ready.prepare_measurement_session(tmp_path / "capture/raw", tmp_path / "intake", prepare=True)
    assert qa["intake_passed"]
    plan = cli.plan_template()
    for name, value in {
        "additional_delay_samples": 0, "estimate_delay_samples": 0, "control_limit": .2,
        "power_floor": 1e-12, "max_limited_fraction": 0, "max_low_amplification_db": 0,
        "max_case_runs": 2, "runtime_conditions": {"mode": "synthetic_unit_fixture_not_hardware"},
        "validation_case_manifest": "cases.json",
        "measurement_packet": "intake",
    }.items():
        plan[name] = value
    for item in plan["candidates"]:
        item.update(mu=.01, control_length=2, block_samples=1, normalization_epsilon=1e-3, weight_norm_limit=1.)
    cases = {"schema": "classical_validation_cases.v1", "cases": [{
        "split": "validation", "session_id": "session_valid_001", "source_id": "source-1", "source_kind": "noise",
        "reference_path": "intake/train_valid/valid/session_valid_001/noise_reference.wav",
        "disturbance_path": "intake/train_valid/valid/session_valid_001/noise_disturbance.wav",
        "regions": {"onset": [[0, 4]], "transition": [], "steady": [[4, 16]]},
    }]}
    write_json(tmp_path / "cases.json", cases)
    return write_json(tmp_path / "plan.json", plan), cases


def test_unknown_plan_is_written_without_wave_or_search(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_read_pcm", lambda *_: pytest.fail("파형 접근 금지"))
    monkeypatch.setattr(cli, "select_validation_baselines", lambda *_: pytest.fail("탐색 금지"))
    plan = write_json(tmp_path / "plan.json", cli.plan_template())
    result = cli.execute_plan(plan, tmp_path / "out")
    assert not result["static_plan_complete"]
    assert result["missing_fields"]
    assert not result["pcm_read"] and not result["search_executed"]
    assert cli.main(["--plan", str(plan), "--out", str(tmp_path / "cli")]) == 2


def test_complete_plan_default_still_does_not_read_wav(tmp_path, monkeypatch):
    plan, _ = complete_plan(tmp_path)
    monkeypatch.setattr(cli, "_read_pcm", lambda *_: pytest.fail("기본은 파형 미열기"))
    result = cli.execute_plan(plan, tmp_path / "out")
    assert result["static_plan_complete"] and result["validation_case_count"] == 1
    assert not result["search_executed"]


@pytest.mark.parametrize("split", ["train", "test"])
def test_late_nonvalidation_case_rejected_before_any_wave(tmp_path, monkeypatch, split):
    plan, cases = complete_plan(tmp_path)
    cases["cases"].append({**cases["cases"][0], "split": split})
    write_json(tmp_path / "cases.json", cases)
    monkeypatch.setattr(cli, "_read_pcm", lambda *_: pytest.fail("WAV 열기 전 test 거부 필요"))
    with pytest.raises(ValueError, match="validation"):
        cli.execute_plan(plan, tmp_path / "out", run_validation_search=True)
    assert not (tmp_path / "out").exists()


def test_run_switch_only_dispatches_explicit_parameters_to_mock(tmp_path, monkeypatch):
    plan, _ = complete_plan(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "_read_pcm", lambda *a, **k: np.arange(16, dtype=float) / 32768)
    def fake(cases, candidates, **kwargs):
        calls.append((cases, candidates, kwargs))
        return {"mock_only": True}
    monkeypatch.setattr(cli, "select_validation_baselines", fake)
    assert cli.execute_plan(plan, tmp_path / "out", run_validation_search=True) == {"mock_only": True}
    assert len(calls) == 1
    cases, candidates, values = calls[0]
    assert cases[0]["split"] == "validation" and len(candidates) == 2
    assert len(values["secondary"]) == 500 and values["additional_delay_samples"] == 0
    assert values["control_limit"] == .2 and values["sample_rate"] == 16000
    binding = values["runtime_conditions"]["validation_input_binding"]
    assert not binding["train_wav_opened"] and not binding["test_wav_opened"]
    assert len(binding["cases"]) == 1 and binding["cases"][0]["source_id"] == "source-1"


@pytest.mark.parametrize("change", [{"sample_rate": 48000}, {"extra": 1}, {"control_limit": 2},
                                  {"max_case_runs": True}, {"secondary_path": "other.txt"}])
def test_invalid_contract_never_dispatches(tmp_path, monkeypatch, change):
    plan = cli.plan_template()
    plan.update(change)
    path = write_json(tmp_path / "plan.json", plan)
    monkeypatch.setattr(cli, "_read_pcm", lambda *_: pytest.fail("파형 접근 금지"))
    with pytest.raises(ValueError):
        cli.execute_plan(path, tmp_path / "out", run_validation_search=True)


def test_pcm16_fullscale_conversion_preserves_relative_gain(tmp_path):
    pcm = np.array([-32768, -1200, 0, 2400, 32767], dtype="<i2")
    path = tmp_path / "x.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, 16000, len(pcm), "NONE", "not compressed"))
        handle.writeframes(pcm.tobytes())
    np.testing.assert_array_equal(cli._read_pcm(path), pcm.astype(float) / 32768)


def test_existing_output_and_symlink_parent_rejected(tmp_path):
    path = write_json(tmp_path / "plan.json", cli.plan_template())
    with pytest.raises(ValueError):
        cli.execute_plan(path, tmp_path)
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        cli.execute_plan(path, tmp_path / "link/out")


def prohibit_pcm(monkeypatch):
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.resolve().suffix.lower() == ".wav":
            pytest.fail("메타데이터 선검사/기본 준비 중 WAV 접근")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    monkeypatch.setattr(wave, "open", lambda *a, **k: pytest.fail("wave.open 실행 금지"))
    monkeypatch.setattr(cli, "select_validation_baselines", lambda *a, **k: pytest.fail("실제 탐색 실행 금지"))


def test_complete_packet_prepare_reads_zero_pcm_and_preserves_binding(tmp_path, monkeypatch):
    path, _ = complete_plan(tmp_path)
    prohibit_pcm(monkeypatch)
    report = cli.execute_plan(path, tmp_path / "out")
    binding = report["validation_input_binding"]
    assert report["static_plan_complete"] and not report["pcm_read"]
    assert not any(Path(name).suffix == ".wav" for name in binding["metadata_sha256"])
    assert binding["cases"][0]["source_group_ids"] == ["group-1"]


@pytest.mark.parametrize("kind", ["test_path", "train_path", "case_identity", "machine", "lock", "qa_identity",
                                  "manifest_test", "symlink_valid", "symlink_train", "capture_changed", "parent_symlink",
                                  "preparation_sha", "case_length", "missing_packet"])
def test_packet_metadata_boundaries_reject_before_any_pcm(tmp_path, monkeypatch, kind):
    path, cases = complete_plan(tmp_path)
    manifest = tmp_path / "intake/train_valid/manifest.jsonl"
    if kind in ("test_path", "train_path", "case_identity", "machine", "case_length"):
        case = cases["cases"][0]
        if kind == "test_path": case["reference_path"] = "capture/raw/test/session_test_001/noise.wav"
        elif kind == "train_path": case["reference_path"] = "intake/train_valid/train/session_train_001/noise_reference.wav"
        elif kind == "case_identity": case["source_id"] = "forged-source"
        elif kind == "machine": case["source_kind"] = "machine"
        else: case["regions"]["steady"] = [[4, 17]]
        write_json(tmp_path / "cases.json", cases)
    elif kind == "lock":
        target = tmp_path / "intake/final_test_lock.json"
        value = json.loads(target.read_text()); value["training_and_tuning_allowed"] = True; write_json(target, value)
    elif kind == "qa_identity":
        target = tmp_path / "intake/report.json"
        value = json.loads(target.read_text()); value["recordings"][0]["source_group_ids"] = ["forged-group"]
        write_json(target, value)
    elif kind == "manifest_test":
        manifest.write_text(manifest.read_text() + json.dumps({"split": "test"}) + "\n")
    elif kind in ("symlink_valid", "symlink_train"):
        split = "valid" if kind == "symlink_valid" else "train"
        target = tmp_path / f"intake/train_valid/{split}/session_{split}_001/noise_reference.wav"
        target.rename(target.with_suffix(".saved"))
        target.symlink_to(tmp_path / "capture/raw/test/session_test_001/noise.wav")
    elif kind == "parent_symlink":
        target = tmp_path / "intake/train_valid/valid/session_valid_001"
        target.rename(target.with_name("moved"))
        target.symlink_to(target.with_name("moved"), target_is_directory=True)
    elif kind == "capture_changed":
        target = tmp_path / "capture/raw/test/session_test_001/capture.json"
        value = json.loads(target.read_text()); value["operator"] = "changed"; write_json(target, value)
    elif kind == "preparation_sha":
        target = tmp_path / "intake/train_valid/preparation.json"
        value = json.loads(target.read_text()); value["recordings"][0]["reference_pcm_sha256"] = "0" * 64
        write_json(target, value)
    else:
        value = json.loads(path.read_text()); value["measurement_packet"] = None; write_json(path, value)
    prohibit_pcm(monkeypatch)
    with pytest.raises(ValueError):
        cli.execute_plan(path, tmp_path / "out", run_validation_search=True)
    assert not (tmp_path / "out").exists()


def guard_nonvalid_pcm(monkeypatch, tmp_path):
    allowed = tmp_path / "intake/train_valid/valid"
    original_open, original_wave = Path.open, wave.open
    def guarded(path, *args, **kwargs):
        if path.suffix.lower() == ".wav" and not path.is_relative_to(allowed):
            pytest.fail("baseline search가 train/test PCM을 열었습니다")
        return original_open(path, *args, **kwargs)
    def guarded_wave(path, *args, **kwargs):
        if not Path(path).is_relative_to(allowed):
            pytest.fail("baseline search가 train/test WAV를 열었습니다")
        return original_wave(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    monkeypatch.setattr(wave, "open", guarded_wave)


def test_explicit_mock_search_reads_only_hash_verified_valid_pcm(tmp_path, monkeypatch):
    path, _ = complete_plan(tmp_path)
    guard_nonvalid_pcm(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(cli, "select_validation_baselines", lambda *a, **k: calls.append((a, k)) or {"mock": True})
    assert cli.execute_plan(path, tmp_path / "out", run_validation_search=True) == {"mock": True}
    arrays = calls[0][0][0][0]
    np.testing.assert_array_equal(arrays["reference"], (np.arange(0, 32, 2) + 101) / 32768.)
    np.testing.assert_array_equal(arrays["disturbance"], (np.arange(1, 32, 2) + 101) / 32768.)


def test_valid_pcm_changed_after_qa_never_reaches_search(tmp_path, monkeypatch):
    path, _ = complete_plan(tmp_path)
    target = tmp_path / "intake/train_valid/valid/session_valid_001/noise_reference.wav"
    data = bytearray(target.read_bytes()); data[-1] ^= 1; target.write_bytes(data)
    guard_nonvalid_pcm(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "select_validation_baselines", lambda *a, **k: pytest.fail("SHA 불일치 뒤 탐색 금지"))
    with pytest.raises(ValueError, match="PCM SHA"):
        cli.execute_plan(path, tmp_path / "out", run_validation_search=True)


@pytest.mark.parametrize("complete,expected", [(False, 2), (True, 0)])
def test_cli_exit_status_preserves_incomplete_selection(monkeypatch, complete, expected):
    monkeypatch.setattr(cli, "execute_plan", lambda *a, **k: {"frozen_selection": {"payload": {"selection_complete": complete}}})
    assert cli.main(["--plan", "mock", "--out", "mock", "--run-validation-search"]) == expected


@pytest.mark.parametrize("kind", ["plan_symlink", "cases_symlink", "plan_wav", "cases_wav", "qa_symlink",
                                  "manifest_symlink", "plan_fifo", "cases_fifo", "manifest_fifo"])
def test_metadata_alias_to_locked_wav_and_nonregular_files_are_never_read(tmp_path, monkeypatch, kind):
    plan, _ = complete_plan(tmp_path)
    locked = tmp_path / "capture/raw/test/session_test_001/noise.wav"
    if kind == "plan_symlink":
        alias = tmp_path / "alias.json"; alias.symlink_to(locked); plan = alias
    elif kind == "plan_wav":
        plan = locked
    elif kind in ("cases_symlink", "cases_wav"):
        if kind == "cases_symlink":
            alias = tmp_path / "alias.json"; alias.symlink_to(locked)
        else:
            alias = locked
        value = json.loads(plan.read_text()); value["validation_case_manifest"] = str(alias); write_json(plan, value)
    elif kind in ("qa_symlink", "manifest_symlink"):
        target = tmp_path / ("intake/report.json" if kind == "qa_symlink" else "intake/train_valid/manifest.jsonl")
        target.rename(target.with_suffix(".saved")); target.symlink_to(locked)
    else:
        target = {"plan_fifo": plan, "cases_fifo": tmp_path / "cases.json",
                  "manifest_fifo": tmp_path / "intake/train_valid/manifest.jsonl"}[kind]
        target.rename(target.with_suffix(".saved")); os.mkfifo(target)
    prohibit_pcm(monkeypatch)
    with pytest.raises(ValueError):
        cli.execute_plan(plan, tmp_path / "out", run_validation_search=True)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["mimii", "machine", "cross_split_group"])
def test_rehashed_capture_metadata_cannot_override_source_policy(tmp_path, monkeypatch, kind):
    plan, _ = complete_plan(tmp_path)
    target = tmp_path / "capture/raw/valid/session_valid_001/capture.json"
    value = json.loads(target.read_text())
    if kind == "mimii": value["source_corpus"] = "MIMII_DG_fan"
    elif kind == "machine": value["source_family"] = "machine"
    else: value["source_group_ids"] = ["group-0"]
    write_json(target, value)
    qa_path = tmp_path / "intake/report.json"
    qa = json.loads(qa_path.read_text())
    item = next(item for item in qa["capture_metadata"] if item["metadata"]["split"] == "valid")
    item.update(metadata=value, sha256=cli._sha(target))
    row = next(row for row in qa["recordings"] if row["split"] == "valid")
    row.update(source_family=value["source_family"], source_kind=value["source_family"], source_group_ids=value["source_group_ids"])
    write_json(qa_path, qa)
    prohibit_pcm(monkeypatch)
    with pytest.raises(ValueError):
        cli.execute_plan(plan, tmp_path / "out", run_validation_search=True)
