"""Behavioral checks for causal paired ANC training and safe resumption."""

import copy
import json
from pathlib import Path
import wave

import numpy as np
import pytest
import torch

from deepanc.data import PairedWaveDataset, load_manifest, read_pcm
from deepanc.model import CausalController
from deepanc.secondary_path import CausalSecondaryPath
from deepanc.train import main


ROOT = Path(__file__).resolve().parents[1]


def write_wav(path, values, rate=16000):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(np.asarray(values, dtype="<i2").tobytes())


def make_manifest(tmp_path, frames=150):
    entries = []
    for index, split in enumerate(("train", "valid")):
        rng = np.random.default_rng(index)
        write_wav(tmp_path / f"x{index}.wav", rng.integers(-5000, 5000, frames))
        write_wav(tmp_path / f"d{index}.wav", rng.integers(-2000, 2000, frames))
        entries.append({"reference": f"x{index}.wav", "disturbance": f"d{index}.wav",
                        "session": f"session{index}", "split": split,
                        "anc_enabled": False, "signal_domain": "raw_pcm"})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")
    return manifest, entries


def rewrite_manifest(manifest, entries):
    manifest.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")


def test_controller_is_causal_and_bounded():
    torch.manual_seed(8)
    model = CausalController(channels=4, dilations=(1, 2), max_output=0.08)
    reference = torch.randn(2, 100)
    altered = reference.clone()
    altered[:, 60:] = 100 * torch.randn(2, 40)
    expected = model(reference)
    actual = model(altered)
    torch.testing.assert_close(actual[:, :60], expected[:, :60], atol=0, rtol=0)
    assert actual.shape == reference.shape
    assert actual.abs().max() <= 0.08
    torch.testing.assert_close(model(torch.zeros_like(reference)), torch.zeros_like(reference), atol=0, rtol=0)


def test_chunk_context_matches_full_recording_and_keeps_pcm_gain(tmp_path):
    manifest, _ = make_manifest(tmp_path)
    records = load_manifest(manifest, 16000)
    model = CausalController(channels=3, kernel_size=3, dilations=(1, 2))
    secondary = CausalSecondaryPath([0.02, 0.1, -0.07, 0.01], delay_samples=3)
    context = model.receptive_field - 1 + secondary.history_samples + 1
    dataset = PairedWaveDataset(records, "train", chunk_samples=37, context_samples=context)
    full_reference = torch.from_numpy(read_pcm(records[0].reference, 0, records[0].frames))[None]
    full_disturbance = torch.from_numpy(read_pcm(records[0].disturbance, 0, records[0].frames))[None]
    expected = full_disturbance + secondary(model(full_reference))
    reconstructed = []
    for batch in dataset:
        actual = batch["disturbance"][None] + secondary(model(batch["reference"][None]))
        reconstructed.append(actual[:, context:][:, batch["mask"]])
    torch.testing.assert_close(torch.cat(reconstructed, dim=1), expected, atol=2e-8, rtol=1e-5)
    write_wav(tmp_path / "quiet.wav", [4096, -4096])
    write_wav(tmp_path / "loud.wav", [8192, -8192])
    np.testing.assert_array_equal(read_pcm(tmp_path / "quiet.wav", 0, 2) * 2,
                                  read_pcm(tmp_path / "loud.wav", 0, 2))


@pytest.mark.parametrize("violation,match", [
    ("session", "Session leakage"), ("file", "Recording leakage"),
    ("content", "Audio-content leakage"), ("length", "lengths differ"),
    ("rate", "16000 Hz"), ("anc", "ANC OFF"), ("domain", "raw_pcm"),
])
def test_manifest_rejects_invalid_pairing_and_leakage(tmp_path, violation, match):
    manifest, entries = make_manifest(tmp_path)
    if violation == "session":
        entries[1]["session"] = entries[0]["session"]
    elif violation == "file":
        entries[1]["reference"] = entries[0]["reference"]
    elif violation == "content":
        (tmp_path / "x1.wav").write_bytes((tmp_path / "x0.wav").read_bytes())
    elif violation == "length":
        write_wav(tmp_path / "d0.wav", [1, 2])
    elif violation == "rate":
        write_wav(tmp_path / "d0.wav", np.arange(150), rate=8000)
    elif violation == "anc":
        entries[0]["anc_enabled"] = True
    elif violation == "domain":
        entries[0]["signal_domain"] = "dc_blocked"
    rewrite_manifest(manifest, entries)
    with pytest.raises(ValueError, match=match):
        load_manifest(manifest, 16000)


def test_actual_training_checkpoint_resume_and_incompatible_config(tmp_path):
    config = json.loads((ROOT / "configs/anc_train.json").read_text())
    config["secondary_path"] = str(ROOT / "rir.txt")
    config["model"] = {"channels": 3, "kernel_size": 3, "dilations": [1, 2], "max_output": 0.1}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "run"
    arguments = ["--config", str(config_path), "--output", str(output), "--device", "cpu", "--smoke-test"]
    assert main(arguments) == 0
    first = torch.load(output / "last.pt", weights_only=True)
    assert first["epoch"] == 1 and first["synthetic"] is True
    assert first["optimizer"]["state"]  # Actual backward/optimizer step occurred.
    assert (output / "best.pt").is_file()
    assert first["validation"]["samples"] == 1024
    assert first["validation"]["output_peak"] <= 0.1
    assert main(arguments + ["--resume", str(output / "last.pt"), "--epochs", "2"]) == 0
    second = torch.load(output / "last.pt", weights_only=True)
    assert second["epoch"] == 2
    assert any(not torch.equal(value, second["model"][key]) for key, value in first["model"].items())
    for artifact in ("metrics.jsonl", "best.pt", "run.json"):
        foreign_output = tmp_path / artifact.replace(".", "_")
        foreign_output.mkdir()
        marker = foreign_output / artifact
        marker.write_bytes(b"unrelated existing experiment")
        with pytest.raises(FileExistsError, match="different run"):
            main(arguments + ["--output", str(foreign_output), "--resume", str(output / "last.pt"), "--epochs", "3"])
        assert marker.read_bytes() == b"unrelated existing experiment"
    incompatible = copy.deepcopy(config)
    incompatible["secondary_delay_samples"] = 1
    config_path.write_text(json.dumps(incompatible))
    with pytest.raises(ValueError, match="Incompatible resume"):
        main(arguments + ["--resume", str(output / "last.pt"), "--epochs", "3"])


def test_explicit_cuda_does_not_silently_fall_back(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        main(["--config", str(ROOT / "configs/anc_train.json"), "--smoke-test",
              "--device", "cuda", "--output", str(tmp_path / "run")])


def test_truncated_wav_never_broadcasts_one_sample_over_a_chunk(tmp_path):
    manifest, _ = make_manifest(tmp_path)
    source = tmp_path / "x0.wav"
    source.write_bytes(source.read_bytes()[:46])  # PCM header still declares 150 samples.
    with pytest.raises(ValueError, match="Truncated recording"):
        read_pcm(source, 0, 150)
    with pytest.raises(ValueError, match="Truncated recording"):
        load_manifest(manifest, 16000)


def test_prepared_pcm_manifest_runs_actual_dataset_training_branch(tmp_path):
    # Synthetic PCM fixture exercises the recorded-data code path, not hardware.
    from tools.prepare_recordings import prepare_recordings

    raw = tmp_path / "raw"
    for index, split in enumerate(("train", "valid")):
        session = raw / split / f"session-{index}"
        session.mkdir(parents=True)
        samples = np.random.default_rng(index).integers(-2000, 2000, (1024, 2), dtype=np.int16)
        with wave.open(str(session / "fixture.wav"), "wb") as stream:
            stream.setnchannels(2)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(samples.astype("<i2").tobytes())
    manifest = prepare_recordings(raw, tmp_path / "prepared", anc_off=True, raw_pcm=True)
    config = json.loads((ROOT / "configs/anc_train.json").read_text())
    config.update(secondary_path=str(ROOT / "rir.txt"), epochs=1, chunk_samples=512)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "recorded-format-run"
    assert main(["--config", str(config_path), "--manifest", str(manifest),
                 "--device", "cpu", "--output", str(output)]) == 0
    result = torch.load(output / "last.pt", weights_only=True)
    assert result["epoch"] == 1 and not result["synthetic"]
    assert result["validation"]["samples"] == 1024
    assert np.isfinite(result["validation"]["loss"])
