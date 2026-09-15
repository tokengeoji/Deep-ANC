"""합성 fixture만 사용한다. 공식 corpus의 실제 QA/다운로드를 대신하지 않는다."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from deep_anc.data.manifest import read_manifest
from deep_anc.data.public_corpus import FAMILIES, inspect_audio, prepare_public_corpus, sha256_file


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def _audio(path: Path, seed: int, *, sr: int = 16000, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.random.default_rng(seed).normal(0, 0.05, (4096, channels))
    # FMA fixture는 .mp3 경로 아래 WAV PCM; 실제 MP3 codec 가용성 검증은 아니다.
    sf.write(path, values, sr, format="FLAC" if path.suffix == ".flac" else "WAV")


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    rows = []
    for index in range(3):
        filename = f"{index + 1}-src-{index}.wav"
        rows.append({"filename": filename, "src_file": str(index + 11), "fold": str(index + 1), "category": "synthetic_fixture"})
        _audio(raw / "esc50" / "ESC-50-master" / "audio" / filename, index + 1)
    _write_csv(raw / "esc50" / "ESC-50-master" / "meta" / "esc50.csv", list(rows[0]), rows)

    tracks = raw / "music" / "fma_metadata" / "tracks.csv"
    tracks.parent.mkdir(parents=True)
    with tracks.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["", "artist", "set", "set", "track"])
        writer.writerow(["", "id", "subset", "split", "license"])
        writer.writerow(["track_id", "", "", "", ""])
        for index in range(3):
            writer.writerow([index + 1, index + 101, "small", "training", "fixture-license"])
            _audio(raw / "music" / "fma_small" / "000" / f"{index + 1:06d}.mp3", index + 11)

    libri = raw / "speech" / "LibriSpeech"
    speaker_rows = ["; official-shape synthetic fixture"]
    for index, subset in enumerate(("train-clean-100", "dev-clean", "test-clean")):
        speaker = str(index + 201)
        folder = libri / subset / speaker / "1"
        utterance = f"{speaker}-1-0000"
        _audio(folder / f"{utterance}.flac", index + 21)
        (folder / f"{speaker}-1.trans.txt").write_text(f"{utterance} SYNTHETIC TEST\n")
        speaker_rows.append(f"{speaker} | M | {subset} | 0.1 | Fixture {index}")
    (libri / "SPEAKERS.TXT").write_text("\n".join(speaker_rows) + "\n")

    for family, offset in (("demand", 31), ("machine", 41)):
        root = raw / family
        rows = []
        for index in range(3):
            relative = f"environment_{index}/ch01.wav"
            _audio(root / relative, index + offset, channels=2 if family == "demand" else 1)
            rows.append({"path": relative, "group_id": f"group-{index}", "official_split": "",
                         "license": "fixture-license", "source_recording_id": f"original-{index}"})
        _write_csv(root / "source_index.csv", list(rows[0]), rows)
        (root / "source_index_meta.json").write_text(json.dumps({
            "source_family": family, "expected_inventory_complete": True,
            "inventory_origin": "synthetic fixture expected file inventory",
            "grouping_basis": "synthetic fixture groups, not real device/section evidence",
        }))
    return raw, tmp_path / "manifests"


def _issues(report: dict) -> set[str]:
    return {issue["code"] for issue in report["issues"]}


def test_complete_corpus_writes_compatible_group_manifests(corpus):
    raw, out = corpus
    report = prepare_public_corpus(raw, out)
    assert report["data_ready"] is True
    assert report["performance_claim_allowed"] is False
    assert report["network_or_download_performed"] is False
    assert set(path.name for path in out.iterdir()) == {"qa.json", "inventory.jsonl", *(f"{family}.jsonl" for family in FAMILIES)}
    inventory = [json.loads(line) for line in (out / "inventory.jsonl").read_text().splitlines()]
    assert len(inventory) == 15
    for family in FAMILIES:
        assert report["split_counts"][family] == {"train": 1, "val": 1, "test": 1}
        path = out / f"{family}.jsonl"
        assert sha256_file(path) == report["manifest_sha256"][path.name]
        entries = read_manifest(path)
        for entry in entries:
            assert Path(entry["path"]).is_file()
            assert entry["sha256"] == sha256_file(Path(entry["path"]))
            assert entry["frames"] == 4096
            assert entry["duration_s"] == 4096 / 16000
            assert entry["qa_valid"] is True
            assert entry["band_power"]["nyquist_hz"] == 8000
            assert entry["band_power"]["snr_and_trusted_band"] == "unknown"
            assert entry["metadata"]["license"]
    speech = read_manifest(out / "speech.jsonl")
    assert {entry["metadata"]["official_subset"]: entry["split"] for entry in speech} == {"train-clean-100": "train", "dev-clean": "val", "test-clean": "test"}
    for relative, digest in report["source_metadata_sha256"].items():
        assert sha256_file(raw / relative) == digest


def test_writer_output_is_accepted_by_strict_prepared_loader(corpus, tmp_path):
    from deep_anc.config import load_train_config
    from deep_anc.data.synth_dataset import validate_prepared_data

    raw, out = corpus
    writer_report = prepare_public_corpus(raw, out)
    cfg = load_train_config("configs/train_acoustic_prepared.yaml")
    rirs = {key: np.zeros((6, 12), dtype=np.float32) for key in ("p_ref", "p_err", "f_fb")}
    for index in range(6):
        for key, delay in (("p_ref", 2), ("p_err", 5), ("f_fb", 3)):
            rirs[key][index, delay] = 0.5 + index * 0.01
    bank = tmp_path / "fixture_rir_bank.npz"
    np.savez(bank, **rirs, sample_rate=16000)
    cfg["data"].update(noise_manifest_dir=str(out), sample_rate=16000, rir_bank=str(bank))
    metadata, validated_rirs = validate_prepared_data(cfg["data"])
    assert metadata["data_ready"] is True
    assert metadata["performance_claim_allowed"] is False
    assert metadata["split_counts"] == writer_report["split_counts"]
    assert metadata["manifest_sha256"] == writer_report["manifest_sha256"]
    assert metadata["source_metadata_sha256"] == writer_report["source_metadata_sha256"]
    for key in rirs:
        np.testing.assert_array_equal(validated_rirs[key], rirs[key])
    # 실제 writer 산출물 이후 원본 변경은 strict loader에서도 거부된다.
    _audio(next((raw / "esc50").rglob("*.wav")), 8122)
    with pytest.raises(ValueError, match="SHA-256"):
        validate_prepared_data(cfg["data"])


@pytest.mark.parametrize("remove_audio_too", [False, True])
def test_librispeech_transcript_inventory_completeness_boundary(corpus, remove_audio_too):
    raw, out = corpus
    # 기존 train speaker의 별도 chapter를 합성한다. 공식 archive 전체가 아니다.
    chapter = raw / "speech/LibriSpeech/train-clean-100/201/2"
    audio = chapter / "201-2-0000.flac"
    transcript = chapter / "201-2.trans.txt"
    _audio(audio, 8931)
    transcript.write_text("201-2-0000 SYNTHETIC EXTRA CHAPTER\n")
    before = prepare_public_corpus(raw, out.parent / "before_missing_chapter")
    assert before["data_ready"]
    assert before["split_counts"]["speech"]["train"] == 2
    transcript.unlink()
    if remove_audio_too:
        audio.unlink()
    report = prepare_public_corpus(raw, out)
    completeness = report["inventory_completeness"]
    assert completeness["official_archive_completeness_verified"] is False
    assert completeness["speech_expected_files_basis"] == "present_official_transcript_rows"
    if remove_audio_too:
        # 외부 완전 inventory가 없으므로 원래 chapter가 있었다는 사실을 모른다.
        # data_ready를 공식 archive 전체 완전성으로 해석해서는 안 된다는 회귀다.
        assert report["data_ready"]
        assert report["split_counts"]["speech"]["train"] == 1
        assert completeness["speech_missing_transcript_and_audio"] == "unknown_without_external_complete_inventory"
    else:
        assert not report["data_ready"]
        assert "audio_missing_from_metadata" in _issues(report)
        assert completeness["speech_missing_transcript_with_remaining_audio"] == "rejected"


def test_reuse_revalidates_without_writes(corpus):
    raw, out = corpus
    first = prepare_public_corpus(raw, out)
    timestamps = {path.name: path.stat().st_mtime_ns for path in out.iterdir()}
    assert prepare_public_corpus(raw, out, reuse=True) == first
    assert timestamps == {path.name: path.stat().st_mtime_ns for path in out.iterdir()}


@pytest.mark.parametrize("changed", ["audio", "metadata", "manifest", "qa", "extra_file"])
def test_reuse_rejects_changes(corpus, changed):
    raw, out = corpus
    prepare_public_corpus(raw, out)
    if changed == "audio":
        _audio(next((raw / "esc50").rglob("*.wav")), 987)
    elif changed == "metadata":
        path = next((raw / "esc50").rglob("esc50.csv"))
        path.write_text(path.read_text().replace("synthetic_fixture", "modified"))
    elif changed == "manifest":
        (out / "esc50.jsonl").write_text("modified\n")
    elif changed == "qa":
        (out / "qa.json").write_text("{}\n")
    else:
        (out / "extra").write_text("keep me")
    original = {path.name: path.read_bytes() for path in out.iterdir()}
    with pytest.raises(ValueError, match="재사용"):
        prepare_public_corpus(raw, out, reuse=True)
    assert original == {path.name: path.read_bytes() for path in out.iterdir()}


def test_existing_output_rejected_before_audio_read(corpus, monkeypatch):
    raw, out = corpus
    out.mkdir()
    (out / "keep").write_text("untouched")
    def forbidden(*args, **kwargs):
        raise AssertionError("기존 출력은 PCM 읽기 전에 거부해야 함")
    monkeypatch.setattr(sf, "SoundFile", forbidden)
    with pytest.raises(FileExistsError):
        prepare_public_corpus(raw, out)
    assert (out / "keep").read_text() == "untouched"


@pytest.mark.parametrize("damage", ["missing", "corrupt", "silent", "nan", "inf", "low_sample_rate"])
def test_bad_audio_preserved_and_no_training_manifests(corpus, damage):
    raw, out = corpus
    path = next((raw / "esc50").rglob("*.wav"))
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_bytes(b"not an audio file")
    elif damage == "low_sample_rate":
        _audio(path, 412, sr=2000)
    else:
        values = np.zeros(4096)
        if damage != "silent":
            values[-1] = np.nan if damage == "nan" else np.inf
        sf.write(path, values, 16000, subtype="FLOAT")
    report = prepare_public_corpus(raw, out)
    assert report["data_ready"] is False
    assert set(path.name for path in out.iterdir()) == {"qa.json", "inventory.jsonl"}
    entries = [json.loads(line) for line in (out / "inventory.jsonl").read_text().splitlines()]
    assert len(entries) == 15
    assert "audio_or_entry_invalid" in _issues(report) or "target_band_unsupported" in _issues(report)


def test_full_pcm_scan_reaches_late_nonfinite(tmp_path):
    path = tmp_path / "late.wav"
    samples = np.ones(150000) * .01
    samples[-1] = np.nan
    sf.write(path, samples, 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="NaN/Inf"):
        inspect_audio(path)


def test_changed_audio_during_pcm_audit_is_rejected(corpus, monkeypatch):
    from deep_anc.data import public_corpus
    raw, out = corpus
    original_inspect = public_corpus.inspect_audio
    changed = False
    def inspect_and_change(path):
        nonlocal changed
        result = original_inspect(path)
        if not changed:
            changed = True
            _audio(path, 8999)
        return result
    monkeypatch.setattr(public_corpus, "inspect_audio", inspect_and_change)
    report = prepare_public_corpus(raw, out)
    assert not report["data_ready"]
    assert any("내용이 변경" in issue["detail"] for issue in report["issues"])


def test_nonfinite_index_metadata_becomes_qa_failure(corpus):
    raw, out = corpus
    path = raw / "machine" / "source_index_meta.json"
    metadata = json.loads(path.read_text())
    metadata["bad_number"] = float("nan")
    path.write_text(json.dumps(metadata))
    report = prepare_public_corpus(raw, out)
    assert not report["data_ready"]
    assert "source_metadata_invalid" in _issues(report)
    assert (out / "qa.json").is_file()


def test_metadata_mutation_during_pcm_read_is_rejected(corpus, monkeypatch):
    from deep_anc.data import public_corpus
    raw, out = corpus
    original_inspect = public_corpus.inspect_audio
    changed = False
    def inspect_and_change(path):
        nonlocal changed
        result = original_inspect(path)
        if not changed:
            changed = True
            metadata = next((raw / "esc50").rglob("esc50.csv"))
            metadata.write_text(metadata.read_text().replace("synthetic_fixture", "changed"))
        return result
    monkeypatch.setattr(public_corpus, "inspect_audio", inspect_and_change)
    report = prepare_public_corpus(raw, out)
    assert not report["data_ready"]
    assert "metadata_changed_during_audit" in _issues(report)


@pytest.mark.parametrize("problem", ["same_group_duplicate", "cross_group_duplicate", "cross_family_duplicate"])
def test_all_exact_duplicates_fail_without_selection(corpus, problem):
    raw, out = corpus
    source = next((raw / "esc50").rglob("*.wav"))
    if problem == "same_group_duplicate":
        path = next((raw / "esc50").rglob("esc50.csv"))
        rows = list(csv.DictReader(path.open()))
        duplicate = dict(rows[0], filename="duplicate.wav")
        rows.append(duplicate)
        _write_csv(path, list(rows[0]), rows)
        shutil.copyfile(source, source.parent / "duplicate.wav")
    elif problem == "cross_group_duplicate":
        dest = next(path for path in (raw / "esc50").rglob("*.wav") if path != source)
        shutil.copyfile(source, dest)
    else:
        shutil.copyfile(source, next((raw / "machine").rglob("*.wav")))
    report = prepare_public_corpus(raw, out)
    assert report["data_ready"] is False
    assert "duplicate_sha256" in _issues(report)


def test_source_group_not_file_split(corpus):
    raw, out = corpus
    path = next((raw / "esc50").rglob("esc50.csv"))
    rows = list(csv.DictReader(path.open()))
    rows.append(dict(rows[0], filename="sibling.wav", fold="5"))
    _write_csv(path, list(rows[0]), rows)
    _audio(path.parent.parent / "audio" / "sibling.wav", 512)
    report = prepare_public_corpus(raw, out)
    assert report["data_ready"] is True
    entries = read_manifest(out / "esc50.jsonl")
    sibling_group = [entry for entry in entries if entry["group_id"] == "esc50:src:11"]
    assert len(sibling_group) == 2
    assert len({entry["split"] for entry in sibling_group}) == 1


@pytest.mark.parametrize("problem", ["missing_csv", "missing_inventory_attestation", "unknown_completeness", "unexpected_audio", "missing_family", "dns", "missing_license", "unsafe_path", "recording_cross_group", "partial_split", "group_cross_split"])
def test_metadata_and_inventory_errors_fail_closed(corpus, problem):
    raw, out = corpus
    path = raw / "machine" / "source_index.csv"
    rows = list(csv.DictReader(path.open()))
    if problem == "missing_csv":
        path.unlink()
    elif problem == "missing_inventory_attestation":
        (path.parent / "source_index_meta.json").unlink()
    elif problem == "unknown_completeness":
        metadata = path.parent / "source_index_meta.json"
        value = json.loads(metadata.read_text())
        value["expected_inventory_complete"] = False
        metadata.write_text(json.dumps(value))
    elif problem == "unexpected_audio":
        _audio(path.parent / "unlisted.wav", 998)
    elif problem == "missing_family":
        shutil.rmtree(raw / "machine")
    elif problem == "dns":
        _audio(raw / "dns" / "unsupported.wav", 999)
    else:
        if problem == "missing_license":
            rows[0]["license"] = ""
        elif problem == "unsafe_path":
            rows[0]["path"] = "../escape.wav"
        elif problem == "recording_cross_group":
            rows[1]["source_recording_id"] = rows[0]["source_recording_id"]
        elif problem == "partial_split":
            rows[0]["official_split"] = "train"
        else:
            for row, split in zip(rows, ("train", "val", "test")):
                row["official_split"] = split
            rows[1]["group_id"] = rows[0]["group_id"]
        _write_csv(path, list(rows[0]), rows)
    report = prepare_public_corpus(raw, out)
    assert report["data_ready"] is False
    assert not (out / "machine.jsonl").exists()
    if problem == "group_cross_split":
        assert "group_cross_split" in _issues(report)
    if problem == "recording_cross_group":
        assert "recording_cross_group" in _issues(report)


def test_audio_symlink_and_parent_symlink_rejected(corpus, tmp_path):
    raw, out = corpus
    target = next((raw / "machine").rglob("*.wav"))
    target.with_name("linked.wav").symlink_to(target)
    assert not prepare_public_corpus(raw, out)["data_ready"]
    link = tmp_path / "parent_link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prepare_public_corpus(raw, link / ".." / "not_created")


@pytest.mark.parametrize("ratios", [{"train": .9, "val": .1, "test": 0}, {"train": .9, "val": np.nan, "test": .1}, {"train": .9}, {"train": .8, "val": .1, "test": .2}])
def test_invalid_split_ratios(corpus, ratios):
    raw, out = corpus
    with pytest.raises(ValueError, match="비율"):
        prepare_public_corpus(raw, out, split_ratios=ratios)
    assert not out.exists()


def test_band_power_parseval_and_boundary(tmp_path):
    path = tmp_path / "tones.wav"
    sr = 8192
    t = np.arange(65536) / sr
    values = .1 * np.sin(2 * np.pi * 400 * t) + .2 * np.sin(2 * np.pi * 1000 * t) + .3 * np.sin(2 * np.pi * 2000 * t)
    sf.write(path, values, sr, subtype="FLOAT")
    report = inspect_audio(path)
    powers = report["band_power"]
    assert powers["low_0_800_hz"] == pytest.approx(.1 ** 2 / 2, rel=1e-6)
    assert powers["target_800_1600_hz"] == pytest.approx(.2 ** 2 / 2, rel=1e-6)
    assert powers["outside_above_1600_hz"] == pytest.approx(.3 ** 2 / 2, rel=1e-6)
    assert powers["total"] == pytest.approx(float(np.mean(values ** 2)), rel=1e-6)


def test_cli_success_failure_and_help(corpus):
    raw, out = corpus
    script = Path(__file__).resolve().parents[1] / "scripts/data/prepare_acoustic_corpus.py"
    help_run = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert help_run.returncode == 0
    assert "--reuse" in help_run.stdout
    run = subprocess.run([sys.executable, str(script), "--raw-root", str(raw), "--out", str(out)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    duplicate = subprocess.run([sys.executable, str(script), "--raw-root", str(raw), "--out", str(out)], capture_output=True, text=True)
    assert duplicate.returncode == 1


def test_import_has_no_torch_audio_device_or_legacy_dependency():
    code = "import sys; import deep_anc.data.public_corpus; assert 'torch' not in sys.modules; assert 'sounddevice' not in sys.modules; assert not any('anc_project' in key for key in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
