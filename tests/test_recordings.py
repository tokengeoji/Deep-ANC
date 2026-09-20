"""Prepared recordings must preserve calibration units and independent sessions."""

import json
import shutil
import struct
import wave

import pytest

from tools.prepare_recordings import prepare_recordings


def write_stereo(path, left, right, sample_rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    interleaved = [value for pair in zip(left, right) for value in pair]
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(struct.pack("<" + "h" * len(interleaved), *interleaved))


def inputs(tmp_path):
    source = tmp_path / "raw"
    write_stereo(source / "train" / "capture-a" / "noise.wav",
                 [-32768, 0, 32767, -123], [91, -401, 12345, 2])
    write_stereo(source / "valid" / "capture-b" / "noise.wav",
                 [103, 2001, -67, -324], [62, 598, -2145, 33])
    return source, tmp_path / "prepared"


def prepare(source, output):
    return prepare_recordings(source, output, anc_off=True, raw_pcm=True)


def test_exact_channel_pcm_gain_synchronization_and_manifest(tmp_path):
    source, output = inputs(tmp_path)
    sidecar = source / "train" / "capture-a" / "capture.json"
    sidecar.write_text(json.dumps({"amplifier_gain": "unknown", "adc_gain_db": 3}))
    manifest = prepare(source, output)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert {item["split"] for item in records} == {"train", "valid"}
    for item in records:
        assert item["anc_enabled"] is False
        assert item["signal_domain"] == "raw_pcm"
        original = source / item["split"] / item["session"] / "noise.wav"
        with wave.open(str(original), "rb") as stream:
            stereo = stream.readframes(stream.getnframes())
        for offset, channel in enumerate(("reference", "disturbance")):
            with wave.open(str(manifest.parent / item[channel]), "rb") as stream:
                assert stream.getnchannels() == 1
                assert stream.getsampwidth() == 2
                assert stream.getframerate() == 16000
                assert stream.getnframes() == 4
                actual = stream.readframes(4)
            expected = b"".join(stereo[i + 2 * offset:i + 2 * offset + 2]
                                for i in range(0, len(stereo), 4))
            assert actual == expected
    report = json.loads((output / "preparation.json").read_text())
    assert report["capture_metadata"]["train/capture-a"]["amplifier_gain"] == "unknown"


def test_prepared_manifest_loads_in_trainer_without_gain_changes(tmp_path):
    from deepanc.data import load_manifest, read_pcm

    source, output = inputs(tmp_path)
    manifest = prepare(source, output)
    records = load_manifest(manifest, 16000)
    training = next(record for record in records if record.split == "train")
    assert read_pcm(training.reference, 0, training.frames).tolist() == [
        value / 32768 for value in (-32768, 0, 32767, -123)]
    assert read_pcm(training.disturbance, 0, training.frames).tolist() == [
        value / 32768 for value in (91, -401, 12345, 2)]


def test_rejects_same_session_across_splits_before_writing(tmp_path):
    source, output = inputs(tmp_path)
    (source / "valid" / "capture-b").rename(source / "valid" / "capture-a")
    with pytest.raises(ValueError, match="Session leakage"):
        prepare(source, output)
    assert not output.exists()


def test_rejects_copied_audio_with_new_session_and_filename(tmp_path):
    source, output = inputs(tmp_path)
    shutil.copyfile(source / "train" / "capture-a" / "noise.wav",
                    source / "valid" / "capture-b" / "renamed.wav")
    with pytest.raises(ValueError, match="Identical PCM audio"):
        prepare(source, output)
    assert not output.exists()


def test_rejects_single_copied_channel_across_splits(tmp_path):
    source, output = inputs(tmp_path)
    write_stereo(source / "valid" / "capture-b" / "noise.wav",
                 [91, -401, 12345, 2], [62, 598, -2145, 33])
    with pytest.raises(ValueError, match="Identical PCM audio"):
        prepare(source, output)
    assert not output.exists()


def test_validates_all_headers_before_creating_output(tmp_path):
    source, output = inputs(tmp_path)
    write_stereo(source / "valid" / "capture-b" / "noise.wav",
                 [100, 101], [-200, -201], sample_rate=48000)
    with pytest.raises(ValueError, match="16000 Hz"):
        prepare(source, output)
    assert not output.exists()


def test_rejects_truncated_recording_before_writing(tmp_path):
    source, output = inputs(tmp_path)
    path = source / "valid" / "capture-b" / "noise.wav"
    path.write_bytes(path.read_bytes()[:-3])
    with pytest.raises(ValueError, match="Incomplete|Truncated"):
        prepare(source, output)
    assert not output.exists()


@pytest.mark.parametrize("anc_off,raw_pcm", [(False, False), (True, False), (False, True)])
def test_requires_capture_attestations(tmp_path, anc_off, raw_pcm):
    source, output = inputs(tmp_path)
    with pytest.raises(ValueError, match="attestations"):
        prepare_recordings(source, output, anc_off=anc_off, raw_pcm=raw_pcm)
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    source, output = inputs(tmp_path)
    output.mkdir()
    sentinel = output / "manifest.jsonl"
    sentinel.write_text("existing user data")
    with pytest.raises(ValueError, match="refusing overwrite"):
        prepare(source, output)
    assert sentinel.read_text() == "existing user data"


def test_input_output_must_be_disjoint(tmp_path):
    source, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        prepare(source, source / "prepared")
    assert not (source / "prepared").exists()


def test_requires_both_nonempty_splits(tmp_path):
    source = tmp_path / "raw"
    write_stereo(source / "train" / "capture-a" / "noise.wav", [1, 2], [3, 4])
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Both train and valid"):
        prepare(source, output)
    assert not output.exists()
