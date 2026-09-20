"""실제 학습/모델/SVD 없이 paired 준비, 고정 FIR 수치, mock orchestration만 검사."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from deepanc import measurement_ready
from deep_anc.train import sfanc_paired as paired

FIT_ADAPTER = paired._fit_candidate
SELECTOR_ADAPTER = paired._train_selector


def write_json(path, value):
    path.write_text(json.dumps(value))


def write_wav(path, values):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(np.asarray(values, dtype="<i2").tobytes())


@pytest.fixture(autouse=True)
def prohibit_learning(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("이 회귀에서 실제 fitting/학습/모델 저장 실행 금지")
    for name in ("_fit_candidate", "_train_selector", "_save_selector"):
        monkeypatch.setattr(paired, name, forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)


@pytest.fixture
def recipe():
    # 회귀용 명시 숫자이며 추천/승인된 학습 설정이 아니다.
    value = paired.recipe_template()
    value.update(window_samples=1024, stride_samples=2048, n_fft=64, control_length=2,
                 additional_delay_samples=4, control_limit=0.04, maximum_windows_per_split=4)
    value["cost"] = {"band_weights": {"low": 1., "priority_high": 1., "remaining_high": 1.},
                     "effort_penalty": 0., "clipping_penalty": 0., "power_floor": 1e-12}
    value["bank"] = {"regularization": 0.01, "effort_penalty": 0., "candidates": [
        {"name": "fixture", "reference": "train/session_train_001/noise_reference.wav", "start_sample": 0, "sample_count": 1024}]}
    value["training"] = {"epochs": 1, "batch_size": 1, "learning_rate": 0.001,
                         "seed": 1, "temperature": 0.1, "risk_weight": 0.1}
    return value


@pytest.fixture
def packet(tmp_path, recipe):
    raw_packet = tmp_path / "raw_packet"
    measurement_ready.create_packet(raw_packet)
    raw = raw_packet / "raw"
    rng = np.random.default_rng(42)
    for index, split in enumerate(("train", "valid", "test")):
        directory = raw / split / f"session_{split}_001"
        sidecar = directory / "capture.json"
        metadata = json.loads(sidecar.read_text())
        for name in ("operator", "capture_utc", "board_model", "board_revision", "firmware_revision",
                     "capture_method", "clock_source"):
            metadata[name] = "fixture-not-measurement"
        metadata.update(source_family="speech", source_corpus="fixture", source_recording_id=f"source-{index}",
                        source_recording_ids=[f"parent-{index}"], source_group_ids=[f"speaker:{index}", f"book:{index}"],
                        gain_profile_id="profile-A")
        metadata["gains"] = {name: {"value": i, "unit": "dB"} for i, name in enumerate(measurement_ready.GAIN_NAMES)}
        metadata["attestations"] = {name: True for name in measurement_ready.ATTESTATIONS}
        metadata["recording_loss"] = {"dropped_frames": 0, "duplicate_frames": 0, "sequence_check_method": "fixture"}
        if split == "test":
            metadata["final_test_not_used_for_tuning"] = True
        write_json(sidecar, metadata)
        write_wav(directory / "noise.wav", rng.integers(-4000, 4000, (2048, 2)))
    output = tmp_path / "intake"
    report = measurement_ready.prepare_measurement_session(raw, output, prepare=True)
    assert report["intake_passed"], report.get("errors")
    recipe_path = tmp_path / "recipe.json"
    write_json(recipe_path, recipe)
    return output, recipe_path, raw


def protect_locked_wav(monkeypatch, raw):
    original = Path.open
    def guarded(path, *args, **kwargs):
        resolved = path.resolve()
        if resolved.suffix == ".wav" and resolved.is_relative_to(raw / "test"):
            pytest.fail("bridge가 locked test WAV를 열었습니다")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    original_wave = wave.open
    def guarded_wave(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).resolve().is_relative_to(raw / "test"):
            pytest.fail("bridge가 locked test WAV를 열었습니다")
        return original_wave(path, *args, **kwargs)
    monkeypatch.setattr(wave, "open", guarded_wave)


def prepare(packet, tmp_path):
    directory, recipe_path, _ = packet
    out = tmp_path / "plan"
    return out, paired.prepare_paired(directory, recipe_path, out)


def test_template_is_unknown_nonrunnable_and_fresh():
    first = paired.recipe_template()
    first["training"]["epochs"] = 77
    assert paired.recipe_template()["training"]["epochs"] is None
    with pytest.raises(ValueError, match="null"):
        paired.validate_recipe(paired.recipe_template())


@pytest.mark.parametrize("change", ["unknown", "delay", "limit", "weight", "nan", "shape", "candidate", "budget"])
def test_explicit_recipe_fail_closed(recipe, change):
    if change == "unknown": recipe["synthetic_primary"] = 128
    elif change == "delay": recipe["additional_delay_samples"] = None
    elif change == "limit": recipe["control_limit"] = 2
    elif change == "weight": recipe["cost"]["band_weights"]["remaining_high"] = 0
    elif change == "nan": recipe["cost"]["power_floor"] = float("nan")
    elif change == "shape": recipe["window_samples"] = 256
    elif change == "candidate": recipe["bank"]["candidates"][0]["name"] = "zero"
    elif change == "budget": recipe["maximum_windows_per_split"] = 100001
    with pytest.raises(ValueError):
        paired.validate_recipe(recipe)


def test_prepare_and_reload_preserve_raw_and_never_open_locked_test(packet, tmp_path, monkeypatch):
    directory, _, raw = packet
    before = {str(path): path.read_bytes() for path in directory.rglob("*.wav")}
    protect_locked_wav(monkeypatch, raw)
    monkeypatch.setattr(paired, "_features", lambda *a: pytest.fail("정적 준비에서 특징 실행 금지"))
    out, plan = prepare(packet, tmp_path)
    assert paired.load_preparation(out) == plan
    assert plan["window_counts"] == {"train": 1, "valid": 1}
    for flag in ("training_executed", "bank_fitting_executed", "model_instantiated", "labels_materialized", "deployment_allowed", "locked_test_wav_opened"):
        assert plan[flag] is False
    assert {path.name for path in out.iterdir()} == {"plan.json", "ready.json"}
    assert before == {name: Path(name).read_bytes() for name in before}
    assert not any(Path(path).suffix == ".wav" and "test" in Path(path).parts for path in plan["input_sha256"])


@pytest.mark.parametrize("kind", ["recipe", "mono_wav", "sidecar", "plan", "forged_ready"])
def test_inputs_and_plan_are_hash_bound(packet, tmp_path, kind):
    out, plan = prepare(packet, tmp_path)
    if kind == "recipe":
        recipe_path = packet[1]
        value = json.loads(recipe_path.read_text()); value["training"]["seed"] = 8
        write_json(recipe_path, value)
    elif kind == "mono_wav":
        path = Path(plan["records"][0]["reference_path"])
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    elif kind == "sidecar":
        path = packet[2] / "test/session_test_001/capture.json"
        value = json.loads(path.read_text()); value["operator"] = "changed"; write_json(path, value)
    else:
        path = out / "plan.json"
        value = json.loads(path.read_text()); value["records"][0]["split"] = "test"; write_json(path, value)
        if kind == "forged_ready":
            write_json(out / "ready.json", {"schema": "sfanc_paired_ready.v1", "plan_sha256": paired._sha(path)})
    with pytest.raises(ValueError):
        paired.load_preparation(out)


@pytest.mark.parametrize("kind", ["test_row", "duplicate", "escape", "valid_fit", "group_leak", "source_leak"])
def test_manifest_and_split_guards_before_fitting(packet, tmp_path, kind, monkeypatch):
    directory, recipe_path, raw = packet
    protect_locked_wav(monkeypatch, raw)
    manifest = directory / "train_valid/manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if kind in ("test_row", "duplicate", "escape"):
        if kind == "test_row": rows.append({"split": "test", "reference": "test/forbidden.wav"})
        elif kind == "duplicate": rows.append(rows[0])
        else: rows[0]["disturbance"] = "../test/forbidden.wav"
        manifest.write_text("\n".join(json.dumps(row) for row in rows))
        monkeypatch.setattr(paired, "_pcm_info", lambda *a: pytest.fail("형식 오류는 WAV 전에 거부"))
    elif kind == "valid_fit":
        value = json.loads(recipe_path.read_text())
        value["bank"]["candidates"][0]["reference"] = next(row["reference"] for row in rows if row["split"] == "valid")
        write_json(recipe_path, value)
    else:
        qa_path = directory / "report.json"
        value = json.loads(qa_path.read_text())
        train = next(row for row in value["recordings"] if row["split"] == "train")
        valid = next(row for row in value["recordings"] if row["split"] == "valid")
        field = "source_group_ids" if kind == "group_leak" else "source_recording_ids"
        valid[field] = deepcopy(train[field])
        write_json(qa_path, value)
    with pytest.raises(ValueError):
        paired.prepare_paired(directory, recipe_path, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_output_existing_and_symlink_parents_rejected_before_inputs(tmp_path):
    occupied = tmp_path / "occupied"; occupied.mkdir()
    alias = tmp_path / "alias"; alias.symlink_to(tmp_path, target_is_directory=True)
    for output in (occupied, alias / "new"):
        with pytest.raises((ValueError, FileExistsError)):
            paired.prepare_paired("absent", "absent", output)


def test_labels_use_actual_hard_limit_and_exact_secondary_and_extra_delay(recipe):
    rng = np.random.default_rng(50)
    x, d = rng.normal(0, .07, (2, 2048))
    bank = np.array([[0., 0.], [2., -.4]])
    result = paired.paired_label(x, d, bank, recipe)
    raw = np.convolve(x, bank[1])[:len(x)]; raw[:1024] = 0
    limited = np.clip(raw, -.04, .04)
    delayed = np.pad(limited, (4, 0))[:len(x)]
    error = d + np.convolve(delayed, paired.load_secondary_path(sample_rate=16000))[:len(x)]
    start = result["scored_from_sample"]
    assert start == 1024 + 499 + 1 + 4
    assert result["costs"][0] == pytest.approx(1.)
    assert result["costs"][1] == pytest.approx(np.mean(error[start:]**2) / np.mean(d[start:]**2), rel=2e-7)
    detail = result["details"][1]
    assert sum(detail["residual_band_power"].values()) == pytest.approx(np.mean(error[start:]**2))
    assert detail["limited_samples"] == np.count_nonzero(np.abs(raw[1024:]) > .04)
    assert detail["output_peak"] <= .04 and detail["raw_peak"] > .04
    assert result["continuous_switching_evaluated"] is False
    assert np.isfinite(result["costs"]).all()


def test_past_ref_features_never_depend_on_future_or_error_and_keep_raw_gain(recipe):
    rng = np.random.default_rng(7)
    x, d = rng.normal(0, .01, (2, 2048))
    bank = np.array([[0., 0.], [.1, 0.]])
    first = paired.paired_label(x, d, bank, recipe)
    changed = x.copy(); changed[1024:] *= -3
    second = paired.paired_label(changed, -d, bank, recipe)
    np.testing.assert_array_equal(first["features"], second["features"])
    scaled = paired.paired_label(x*2, d, bank, recipe)
    np.testing.assert_allclose(scaled["features"][1]-first["features"][1], np.log(2), rtol=1e-6)


@pytest.mark.parametrize("kind", ["complex", "boolean", "nan", "nonzero_first", "wrong_length", "extra_candidate", "non_pcm", "overflow"])
def test_numeric_inputs_fail_closed(recipe, kind):
    x = np.ones(2048)*.01; d = x.copy(); bank = np.array([[0., 0.], [.1, 0.]])
    if kind == "complex": x = x.astype(complex)
    elif kind == "boolean": bank = bank.astype(bool)
    elif kind == "nan": d[0] = np.nan
    elif kind == "nonzero_first": bank[0, 0] = 1
    elif kind == "wrong_length": x = x[:-1]
    elif kind == "extra_candidate": bank = np.zeros((3, 2))
    elif kind == "non_pcm": d[0] = 1.1
    elif kind == "overflow": x[:] = 1; bank[1, :] = np.finfo(np.float64).max
    with pytest.raises(ValueError): paired.paired_label(x, d, bank, recipe)


def test_internal_symlink_never_opens_a_hidden_test_wav(packet, tmp_path, monkeypatch):
    directory, recipe_path, _ = packet
    reference = directory / "train_valid/train/session_train_001/noise_reference.wav"
    hidden = directory / "train_valid/test/forbidden.wav"
    hidden.parent.mkdir()
    reference.rename(hidden)
    reference.symlink_to(hidden)
    monkeypatch.setattr(paired, "_pcm_info", lambda *a: pytest.fail("symlink는 PCM 전에 거부"))
    with pytest.raises(ValueError, match="상대경로"):
        paired.prepare_paired(directory, recipe_path, tmp_path / "bad")


def test_qa_identity_cannot_override_original_capture_metadata(packet, tmp_path):
    directory, recipe_path, _ = packet
    qa_path = directory / "report.json"
    value = json.loads(qa_path.read_text())
    value["recordings"][0]["source_group_ids"] = ["invented-different-group"]
    write_json(qa_path, value)
    with pytest.raises(ValueError, match="capture 메타데이터 불일치"):
        paired.prepare_paired(directory, recipe_path, tmp_path / "bad")


@pytest.mark.parametrize("relative", ["report.json", "final_test_lock.json", "train_valid/manifest.jsonl", "train_valid/preparation.json"])
def test_metadata_symlinks_never_read_locked_wav(packet, tmp_path, monkeypatch, relative):
    directory, recipe_path, raw = packet
    metadata = directory / relative
    metadata.rename(metadata.with_suffix(metadata.suffix + ".original"))
    metadata.symlink_to(raw / "test/session_test_001/noise.wav")
    protect_locked_wav(monkeypatch, raw)
    with pytest.raises(ValueError, match="symlink"):
        paired.prepare_paired(directory, recipe_path, tmp_path / "bad")


@pytest.mark.parametrize("filename", ["ready.json", "plan.json"])
def test_published_plan_symlinks_never_hash_locked_wav(packet, tmp_path, monkeypatch, filename):
    out, _ = prepare(packet, tmp_path)
    path = out / filename
    path.rename(path.with_suffix(".original"))
    path.symlink_to(packet[2] / "test/session_test_001/noise.wav")
    protect_locked_wav(monkeypatch, packet[2])
    with pytest.raises(ValueError, match="symlink"):
        paired.load_preparation(out)


def test_materialize_train_valid_but_reject_test_before_loading(packet, tmp_path, monkeypatch):
    out, _ = prepare(packet, tmp_path)
    protect_locked_wav(monkeypatch, packet[2])
    for split in ("train", "valid"):
        values = paired.materialize_paired_labels(out, np.array([[0., 0.], [.1, 0.]]), split)
        assert values["features"].shape == (1, 2, 33)
        assert values["costs"].shape == (1, 2)
        assert len(values["rows"]) == 1
    monkeypatch.setattr(paired, "load_preparation", lambda *a: pytest.fail("test 접근 전에 거부"))
    with pytest.raises(ValueError, match="test"):
        paired.materialize_paired_labels("missing", np.zeros((2, 2)), "test")


def test_training_default_denies_before_loading_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(paired, "load_preparation", lambda *a: pytest.fail("명시 승인 전에 로드 금지"))
    with pytest.raises(PermissionError):
        paired.train_paired("missing", tmp_path / "train", device="cpu")
    assert not (tmp_path / "train").exists()


def test_mock_orchestration_only_train_fit_valid_selection_no_test(packet, tmp_path, monkeypatch):
    preparation, plan = prepare(packet, tmp_path)
    protect_locked_wav(monkeypatch, packet[2])
    calls = []
    def fit(x, d, secondary, recipe):
        expected = paired._read_pcm(plan["records"][0]["reference_path"], 0, 1024)
        np.testing.assert_array_equal(x, expected)
        assert len(secondary) == 500
        calls.append("fit_train_mock")
        return SimpleNamespace(coefficients=np.array([.1, 0.]), diagnostics={"mock": True})
    def train(training, valid, recipe, device):
        assert training["rows"][0]["session"] == "session_train_001"
        assert valid["rows"][0]["session"] == "session_valid_001"
        calls.append("train_selector_mock")
        return SimpleNamespace(best_epoch=0, history=[], best_validation_loss=.9)
    def save(path, result, recipe, digest):
        assert len(digest) == 64
        path.write_bytes(b"mock-not-a-model")
        calls.append("save_mock")
    monkeypatch.setattr(paired, "_fit_candidate", fit)
    monkeypatch.setattr(paired, "_train_selector", train)
    monkeypatch.setattr(paired, "_save_selector", save)
    output = tmp_path / "mock_training"
    result = paired.train_paired(preparation, output, approve_training=True, device="cpu")
    assert calls == ["fit_train_mock", "train_selector_mock", "save_mock"]
    assert result["bank_fit_split"] == "train" and result["checkpoint_selection_split"] == "valid"
    assert not result["test_opened"] and not result["deployment_allowed"]
    assert not result["primary_is_synthetic"] and not result["physical_performance_claim_allowed"]
    assert result["artifacts"]["bank.npz"] == paired._sha(output / "bank.npz")
    with np.load(output / "bank.npz") as bank:
        np.testing.assert_array_equal(bank["secondary"], paired.load_secondary_path(sample_rate=16000))
    with pytest.raises(FileExistsError):
        paired.train_paired(preparation, output, approve_training=True, device="cpu")


def test_future_fit_adapter_passes_only_explicit_recipe_to_mock(recipe, monkeypatch):
    from deep_anc.baselines import sfanc_design
    calls = []
    def mocked(x, d, secondary, **kwargs):
        calls.append(kwargs)
        return "mock-fit-no-svd"
    monkeypatch.setattr(sfanc_design, "fit_control_fir", mocked)
    assert FIT_ADAPTER(np.zeros(1024), np.zeros(1024), np.zeros(500), recipe) == "mock-fit-no-svd"
    assert calls == [{"control_length": 2, "secondary_delay_samples": 4, "warmup_samples": 504,
                      "regularization": .01, "effort_penalty": 0.}]


def test_future_selector_adapter_passes_validation_costs_to_mock(recipe, monkeypatch):
    from deep_anc.train import sfanc_selector
    calls = []
    def mocked(*args, **kwargs):
        calls.append((args, kwargs))
        return "mock-selector-no-model"
    monkeypatch.setattr(sfanc_selector, "train_selector", mocked)
    training = {"features": "train-x", "costs": "train-cost"}
    valid = {"features": "valid-x", "costs": "valid-cost"}
    assert SELECTOR_ADAPTER(training, valid, recipe, "cpu") == "mock-selector-no-model"
    assert calls == [(("train-x", "train-cost", "valid-x", "valid-cost"), {"device": "cpu", **recipe["training"]})]


def test_cli_help_unknown_flags_and_null_recipe_never_train(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/train/prepare_sfanc_paired.py"
    helped = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert helped.returncode == 0 and "optimizer" in helped.stdout
    refused = subprocess.run([sys.executable, str(script), "--train"], capture_output=True, text=True)
    assert refused.returncode == 2
    recipe = tmp_path / "null.json"; write_json(recipe, paired.recipe_template())
    failed = subprocess.run([sys.executable, str(script), "--packet", "missing", "--recipe", str(recipe),
                             "--out", str(tmp_path / "bad")], capture_output=True, text=True)
    assert failed.returncode == 1 and "null" in failed.stderr
    assert not (tmp_path / "bad").exists()
