"""SFANC 음성 원천의 합성 파일 fixture 검사: 실제 녹음/감쇠 검증 아님."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from deep_anc.data.sfanc_sources import iter_librispeech_crops, prepare_librispeech_manifest


@pytest.fixture
def libri(tmp_path):
    root = tmp_path / "LibriSpeech"
    root.mkdir()
    (root / "LICENSE.TXT").write_text(
        "LibriSpeech fixture: Creative Commons Attribution 4.0 International License.\n"
    )
    speakers, chapters = [], []
    # 1--book100--2--book200--3은 하나의 연결요소여야 한다.
    identities = [(1, 11, 100), (2, 21, 100), (2, 22, 200), (3, 31, 200)]
    identities += [(speaker, speaker * 10 + 1, speaker * 100) for speaker in range(4, 10)]
    for speaker in range(1, 10):
        speakers.append(f"{speaker} | F | dev-clean | 1.0 | Reader {speaker}")
    for speaker, chapter, book in identities:
        chapters.append(f"{chapter} | {speaker} | 1.0 | dev-clean | 5 | {book} | Chapter | Book")
        folder = root / "dev-clean" / str(speaker) / str(chapter)
        folder.mkdir(parents=True)
        for utterance in range(3):
            path = folder / f"{speaker}-{chapter}-{utterance:04d}.flac"
            samples = np.random.default_rng(chapter * 10 + utterance).uniform(-0.03, 0.06, 192)
            sf.write(path, samples, 16000, subtype="PCM_16")
    (root / "SPEAKERS.TXT").write_text("\n".join(speakers) + "\n")
    (root / "CHAPTERS.TXT").write_text("\n".join(chapters) + "\n")
    return root


def _prepare(root, **kwargs):
    return prepare_librispeech_manifest(root, crop_samples=64, **kwargs)


def _rows(manifest):
    return [row for rows in manifest["splits"].values() for row in rows]


def _snapshot(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_deterministic_connected_component_split_preserves_every_source(libri):
    before = _snapshot(libri)
    manifest = _prepare(libri, seed=17)
    assert manifest == _prepare(libri, seed=17)
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["inventory"]["utterances"] == 30
    assert manifest["inventory"]["components"] == 7
    assert len(_rows(manifest)) == 30
    assert manifest["simulation_pretraining_only"] and not manifest["measured_ref_err"]
    assert not manifest["physical_claim_allowed"] and not manifest["deployment_allowed"]
    assert not manifest["resampling"] and not manifest["normalization"]
    assert not manifest["provenance"]["official_archive_checksum_verified"]
    groups = {row["group_id"] for row in _rows(manifest) if row["speaker"] in ("1", "2", "3")}
    assert len(groups) == 1
    for field in ("speaker", "book", "group_id", "source_id"):
        seen = set()
        for split, rows in manifest["splits"].items():
            values = {row[field] for row in rows}
            assert values and not (values & seen), (field, split)
            seen.update(values)
    for row in _rows(manifest):
        assert not Path(row["path"]).is_absolute()
        assert row["license"] == "CC-BY-4.0"
        assert row["duration_s"] == 192 / 16000
        assert row["sha256"] == before[row["path"]]
        assert len(row["crops"]) == 2
        assert row["crops"][0]["start"] + 64 <= row["crops"][1]["start"]
    assert before == _snapshot(libri)


def test_limits_use_group_round_robin_after_unchanged_split(libri):
    full = _prepare(libri, seed=3)
    small = _prepare(libri, seed=3, max_files_per_split={"train": 3, "validation": 2, "test": 1})
    original = {row["source_id"]: row for row in _rows(full)}
    assert [len(rows) for rows in small["splits"].values()] == [3, 2, 1]
    assert len({row["group_id"] for row in small["splits"]["train"]}) == 3
    assert all(row == original[row["source_id"]] for row in _rows(small))
    changed = _prepare(libri, seed=8)
    assert [(r["source_id"], r["split"]) for r in _rows(full)] != [
        (r["source_id"], r["split"]) for r in _rows(changed)]


def test_crop_reads_are_bounded_exact_unscaled_and_read_only(libri, monkeypatch):
    manifest = _prepare(libri, max_files_per_split={"train": 2, "validation": 1, "test": 1})
    before = _snapshot(libri)
    expected = {}
    for row in manifest["splits"]["train"]:
        audio, rate = sf.read(libri / row["path"], dtype="float32")
        assert rate == 16000
        for crop in row["crops"]:
            expected[(row["source_id"], crop["start"])] = audio[crop["start"]:crop["start"] + 64]
    original = sf.SoundFile.read
    read_sizes = []

    def checked_read(stream, frames=-1, *args, **kwargs):
        read_sizes.append(frames)
        assert 1 <= frames <= 64
        return original(stream, frames, *args, **kwargs)

    monkeypatch.setattr(sf.SoundFile, "read", checked_read)
    loaded = list(iter_librispeech_crops(manifest, "train"))
    assert len(loaded) == 4 and read_sizes == [64] * 4
    for audio, metadata in loaded:
        assert audio.dtype == np.float32 and audio.shape == (64,)
        assert metadata["split"] == "train" and metadata["group_id"]
        assert metadata["amplitude_domain"] == "decoded_source_audio_not_calibrated_adc_dac"
        np.testing.assert_array_equal(audio, expected[(metadata["source_id"], metadata["crop_start"])])
        assert np.max(np.abs(audio)) < 0.07  # peak normalization을 하면 실패한다.
    assert before == _snapshot(libri)


@pytest.mark.parametrize("problem,match", [
    ("license", "license"), ("speaker", "subset"), ("chapter", "reader"),
    ("book", "book"), ("rate", "16 kHz"), ("stereo", "mono"), ("symlink", "symlink"),
])
def test_preparation_rejects_missing_contract(libri, tmp_path, problem, match):
    audio = next((libri / "dev-clean").rglob("*.flac"))
    if problem == "license":
        (libri / "LICENSE.TXT").write_text("Unknown terms")
    elif problem == "speaker":
        path = libri / "SPEAKERS.TXT"
        path.write_text(path.read_text().replace("dev-clean", "test-clean"))
    elif problem in ("chapter", "book"):
        path = libri / "CHAPTERS.TXT"
        rows = path.read_text().splitlines()
        fields = rows[0].split("|")
        fields[1 if problem == "chapter" else 5] = "999" if problem == "chapter" else "unknown"
        rows[0] = "|".join(fields)
        path.write_text("\n".join(rows) + "\n")
    elif problem == "symlink":
        outside = tmp_path / "elsewhere.flac"
        audio.rename(outside)
        audio.symlink_to(outside)
    else:
        sf.write(audio, np.ones((192, 2) if problem == "stereo" else 192) * 0.1,
                 48000 if problem == "rate" else 16000)
    with pytest.raises(ValueError, match=match):
        _prepare(libri)


def test_too_few_components_fail_without_splitting_speakers(libri):
    path = libri / "CHAPTERS.TXT"
    changed = []
    for line in path.read_text().splitlines():
        fields = line.split("|")
        fields[5] = "100"
        changed.append("|".join(fields))
    path.write_text("\n".join(changed) + "\n")
    with pytest.raises(ValueError, match="3개 독립"):
        _prepare(libri)


@pytest.mark.parametrize("change,match", [
    ("audio", "SHA"), ("metadata", "SHA"), ("traversal", "상대경로"),
    ("speaker_overlap", "speaker 누출"), ("book_overlap", "book 누출"),
    ("range", "crop"), ("overlap", "crop"), ("oversize", "crop"),
])
def test_iterator_fails_closed_on_changed_inputs_or_tampered_manifest(libri, change, match):
    manifest = copy.deepcopy(_prepare(libri))
    row = manifest["splits"]["train"][0]
    if change == "audio":
        sf.write(libri / row["path"], np.zeros(192), 16000)
    elif change == "metadata":
        path = libri / "LICENSE.TXT"
        path.write_text(path.read_text() + "changed\n")
    elif change == "traversal":
        row["path"] = "../elsewhere.flac"
    elif change.endswith("_overlap"):
        field = change.split("_")[0]
        row[field] = manifest["splits"]["test"][0][field]
    elif change == "range":
        row["crops"][0]["start"] = 192
    elif change == "overlap":
        row["crops"][1]["start"] = row["crops"][0]["start"]
    elif change == "oversize":
        row["crops"][0]["frames"] = 8193
    with pytest.raises(ValueError, match=match):
        list(iter_librispeech_crops(manifest, "train"))


@pytest.mark.parametrize("kwargs", [
    {"seed": -1}, {"seed": True}, {"crop_samples": 8193}, {"crop_samples": 0},
    {"max_crops_per_file": 3}, {"max_files_per_split": {"train": 0}},
    {"max_files_per_split": {"val": 1}},
])
def test_invalid_limits_are_rejected_before_loading(libri, kwargs):
    with pytest.raises(ValueError):
        prepare_librispeech_manifest(libri, **kwargs)
