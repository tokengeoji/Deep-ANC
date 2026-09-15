"""원본 다운로드 없이 합성 archive/receipt만 검사한다."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import zipfile

import numpy as np
import pytest
import soundfile as sf

from deep_anc.data.archive_audit import audit_public_archive
from deep_anc.data.archive_staging import ESC50_APPROVED_COMMIT, ESC50_GIT_URL, validate_staging_provenance


def audio_bytes(*, kind="WAV", frames=4096, bad=None, channels=1):
    data = np.random.default_rng(11).normal(0, .03, (frames, channels))
    if bad is not None:
        data[-1, 0] = bad
    buffer = io.BytesIO()
    sf.write(buffer, data, 16000, format=kind, subtype="FLOAT" if kind == "WAV" else "PCM_16")
    return buffer.getvalue()


def make_archive(tmp_path, entries, kind="zip"):
    archive = tmp_path / ("fixture.zip" if kind == "zip" else "fixture.tar.gz")
    if kind == "zip":
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
            for name, data, member_kind in entries:
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_DEFLATED
                if member_kind == "link":
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                elif member_kind == "directory":
                    info.external_attr = (stat.S_IFDIR | 0o755) << 16
                stream.writestr(info, data)
    else:
        with tarfile.open(archive, "w:gz") as stream:
            for name, data, member_kind in entries:
                info = tarfile.TarInfo(name)
                if member_kind == "link":
                    info.type = tarfile.SYMTYPE
                    info.linkname = "elsewhere.wav"
                elif member_kind == "hardlink":
                    info.type = tarfile.LNKTYPE
                    info.linkname = "elsewhere.wav"
                elif member_kind == "directory":
                    info.type = tarfile.DIRTYPE
                else:
                    info.size = len(data)
                stream.addfile(info, io.BytesIO(data) if info.isfile() else None)
    return archive, write_receipt(archive)


def write_receipt(archive):
    data = archive.read_bytes()
    receipt = archive.parent / "receipt.json"
    receipt.write_text(json.dumps({
        "schema_version": 1, "name": archive.name, "archive_path": str(archive),
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "md5": hashlib.md5(data).hexdigest(),
        "source_checksum_algorithm": "sha256", "source_checksum": hashlib.sha256(data).hexdigest(),
        "source_checksum_verified": True, "research_use": "noncommercial_academic",
        "license": "synthetic-fixture-not-real-corpus-license",
    }))
    return receipt


def rows(out):
    return [json.loads(line) for line in (out / "inventory.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_all_audio_pcm_and_metadata_without_extraction(tmp_path, kind):
    archive, receipt = make_archive(tmp_path, [
        ("audio/", b"", "directory"), ("audio/one.wav", audio_bytes(), "file"),
        ("audio/two.flac", audio_bytes(kind="FLAC"), "file"),
        ("meta/tracks.csv", b"official-shape synthetic fixture\n", "file"),
    ], kind)
    before = {path.name for path in tmp_path.iterdir()}
    out = tmp_path / "qa"
    report = audit_public_archive(archive, receipt, out)
    assert report["source_archive_content_qa_passed"]
    assert report["audio_pcm_qa_passed"]
    assert report["source_archive_content_qa_status"] == "audio_pcm_qa_passed"
    assert not report["data_ready"] and not report["training_ready"]
    assert not report["physical_performance_claim_allowed"]
    assert not report["extraction_performed"] and not report["local_deletion_performed"]
    assert before | {"qa"} == {path.name for path in tmp_path.iterdir()}
    assert {path.name for path in out.iterdir()} == {"inventory.jsonl", "qa.json"}
    assert report["counts"] == {"entries": 4, "audio": 2, "audio_passed": 2, "metadata": 1, "directories": 1, "failed": 0}
    for row in rows(out):
        if row.get("is_audio"):
            assert row["audio"]["frames"] == 4096
            assert row["audio"]["sample_rate"] == 16000
            assert row["audio"]["band_power"]["target_supported_by_sample_rate"]


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_large_metadata_streams_beyond_audio_memory_cap(tmp_path, kind, monkeypatch):
    metadata = b"a,b,c\n" * 400000
    archive, receipt = make_archive(tmp_path, [("fma_metadata/tracks.csv", metadata, "file")], kind)
    monkeypatch.setattr(sf, "SoundFile", lambda *args, **kwargs: pytest.fail("metadata-only는PCM decoder를사용하지않음"))
    out = tmp_path / "qa"
    report = audit_public_archive(archive, receipt, out, max_entry_mib=1)
    assert report["source_archive_content_qa_status"] == "metadata_only"
    assert report["metadata_only"] and report["source_archive_content_qa_passed"]
    assert not report["audio_pcm_qa_passed"] and not report["data_ready"]
    assert rows(out)[0]["sha256"] == hashlib.sha256(metadata).hexdigest()


@pytest.mark.parametrize("bad_name", ["../escape.wav", "/absolute.wav", "C:/drive.wav", "folder/../../escape.wav", "bad\\path.wav"])
@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_unsafe_names_fail_without_opening_payload(tmp_path, bad_name, kind, monkeypatch):
    archive, receipt = make_archive(tmp_path, [(bad_name, audio_bytes(), "file")], kind)
    monkeypatch.setattr(sf, "SoundFile", lambda *args, **kwargs: pytest.fail("unsafe항목decoder접근금지"))
    out = tmp_path / "qa"
    report = audit_public_archive(archive, receipt, out)
    assert not report["source_archive_content_qa_passed"]
    assert rows(out)[0]["qa_status"] == "failed"
    assert not (tmp_path.parent / "escape.wav").exists()


@pytest.mark.parametrize("kind,member_kind", [("zip", "link"), ("tar", "link"), ("tar", "hardlink")])
def test_links_never_read(tmp_path, kind, member_kind, monkeypatch):
    archive, receipt = make_archive(tmp_path, [("linked.wav", b"target", member_kind)], kind)
    monkeypatch.setattr(sf, "SoundFile", lambda *args, **kwargs: pytest.fail("link는읽으면안됨"))
    out = tmp_path / "qa"
    assert not audit_public_archive(archive, receipt, out)["source_archive_content_qa_passed"]
    assert rows(out)[0]["kind"] == "link_or_special"


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_duplicate_normalized_paths_are_not_selected(tmp_path, kind):
    archive, receipt = make_archive(tmp_path, [("a.wav", audio_bytes(), "file"), ("./a.wav", audio_bytes(), "file")], kind)
    out = tmp_path / "qa"
    report = audit_public_archive(archive, receipt, out)
    assert not report["source_archive_content_qa_passed"]
    assert len(rows(out)) == 2 and rows(out)[1]["qa_status"] == "failed"


@pytest.mark.parametrize("damage", ["corrupt", "nan", "inf", "oversized", "channels"])
def test_bad_audio_preserved_and_remaining_audio_still_checked(tmp_path, damage):
    payload = b"not audio" if damage == "corrupt" else audio_bytes(
        bad=np.nan if damage == "nan" else np.inf if damage == "inf" else None,
        frames=400000 if damage == "oversized" else 4096,
        channels=65 if damage == "channels" else 1,
    )
    archive, receipt = make_archive(tmp_path, [("bad.wav", payload, "file"), ("good.wav", audio_bytes(), "file")])
    out = tmp_path / "qa"
    report = audit_public_archive(archive, receipt, out, max_entry_mib=1)
    assert not report["source_archive_content_qa_passed"]
    assert report["counts"]["audio"] == 2
    assert report["counts"]["failed"] == 1
    assert report["counts"]["audio_passed"] == 1
    assert report["traversal_complete"]


@pytest.mark.parametrize("field,value", [("source_checksum_verified", False), ("schema_version", True),
                                         ("name", "different.zip"), ("bytes", 1), ("sha256", "0" * 64),
                                         ("source_checksum", "0" * 64), ("license", "")])
def test_invalid_receipt_rejected_before_output(tmp_path, field, value):
    archive, receipt = make_archive(tmp_path, [("a.wav", audio_bytes(), "file")])
    metadata = json.loads(receipt.read_text())
    metadata[field] = value
    receipt.write_text(json.dumps(metadata))
    out = tmp_path / "qa"
    with pytest.raises(ValueError):
        audit_public_archive(archive, receipt, out)
    assert not out.exists()


@pytest.mark.parametrize("target", ["archive", "receipt"])
def test_changes_during_audit_reject_old_pass(tmp_path, target, monkeypatch):
    from deep_anc.data import archive_audit
    archive, receipt = make_archive(tmp_path, [("a.wav", audio_bytes(), "file")])
    original_inspect = archive_audit.inspect_audio
    def mutate_after_inspection(buffer):
        result = original_inspect(buffer)
        with (archive if target == "archive" else receipt).open("ab") as stream:
            stream.write(b"changed during QA")
        return result
    monkeypatch.setattr(archive_audit, "inspect_audio", mutate_after_inspection)
    out = tmp_path / "qa"
    report = audit_public_archive(archive, receipt, out)
    assert not report["source_archive_content_qa_passed"]
    assert any(issue["code"] == target + "_changed_during_audit" for issue in report["issues"])


def test_existing_output_and_symlink_rejected(tmp_path):
    archive, receipt = make_archive(tmp_path, [("metadata.txt", b"metadata", "file")])
    out = tmp_path / "qa"
    out.mkdir()
    (out / "keep").write_text("untouched")
    with pytest.raises(FileExistsError):
        audit_public_archive(archive, receipt, out)
    assert (out / "keep").read_text() == "untouched"
    link = tmp_path / "linked.zip"
    link.symlink_to(archive)
    with pytest.raises(ValueError, match="링크"):
        audit_public_archive(link, receipt, tmp_path / "another")


def test_tar_nonzero_trailing_content_is_not_ignored(tmp_path):
    archive, _ = make_archive(tmp_path, [("metadata.txt", b"metadata", "file")], "tar")
    content = gzip.decompress(archive.read_bytes())
    archive.write_bytes(gzip.compress(content + b"hidden nonzero data"))
    receipt = write_receipt(archive)
    report = audit_public_archive(archive, receipt, tmp_path / "qa")
    assert not report["source_archive_content_qa_passed"]
    assert not report["traversal_complete"]


@pytest.mark.parametrize("remaining_zero_blocks", [0, 1])
def test_tar_missing_termination_blocks_fails_even_with_valid_gzip(tmp_path, remaining_zero_blocks):
    archive, _ = make_archive(tmp_path, [("metadata.txt", b"metadata", "file")], "tar")
    content = gzip.decompress(archive.read_bytes())
    # 한 header + 한 padded data block 뒤 종결을 잘라낸 유효한 gzip stream.
    archive.write_bytes(gzip.compress(content[:1024 + 512 * remaining_zero_blocks]))
    receipt = write_receipt(archive)
    report = audit_public_archive(archive, receipt, tmp_path / "qa")
    assert not report["source_archive_content_qa_passed"]
    assert not report["traversal_complete"]
    assert report["counts"]["metadata"] == 1


def test_tar_invalid_header_after_valid_member_fails(tmp_path):
    archive, _ = make_archive(tmp_path, [("metadata.txt", b"metadata", "file")], "tar")
    content = gzip.decompress(archive.read_bytes())
    archive.write_bytes(gzip.compress(content[:1024] + b"X" * 512 + content[1536:]))
    receipt = write_receipt(archive)
    report = audit_public_archive(archive, receipt, tmp_path / "qa")
    assert not report["source_archive_content_qa_passed"]
    assert not report["traversal_complete"]


def test_tar_pax_metadata_bound_and_no_retained_member_list(tmp_path, monkeypatch):
    from deep_anc.data import archive_audit
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as stream:
        for index in range(5):
            info = tarfile.TarInfo(f"metadata_{index}.txt")
            info.size = 1
            info.pax_headers = {"fixture": "a" * 100}
            stream.addfile(info, io.BytesIO(b"a"))
    receipt = write_receipt(archive)
    original = archive_audit._BoundedTarInfo._proc_member
    retained_counts = []
    def check_retention(self, current):
        retained_counts.append(len(current.members))
        return original(self, current)
    monkeypatch.setattr(archive_audit._BoundedTarInfo, "_proc_member", check_retention)
    report = audit_public_archive(archive, receipt, tmp_path / "qa")
    assert report["source_archive_content_qa_passed"]
    assert max(retained_counts) <= 1
    monkeypatch.setattr(archive_audit, "MAX_TAR_HEADER_BYTES", 64)
    report = audit_public_archive(archive, receipt, tmp_path / "too_large")
    assert not report["source_archive_content_qa_passed"]


def test_empty_directory_archive_is_not_content_pass(tmp_path):
    archive, receipt = make_archive(tmp_path, [("./", b"", "directory")])
    report = audit_public_archive(archive, receipt, tmp_path / "qa")
    assert not report["source_archive_content_qa_passed"]


def test_cli_metadata_only_and_no_torch_audio_device_import(tmp_path):
    archive, receipt = make_archive(tmp_path, [("tracks.csv", b"metadata", "file")])
    script = Path(__file__).resolve().parents[1] / "scripts/data/audit_public_archive.py"
    result = subprocess.run([sys.executable, str(script), "--archive", str(archive), "--staging-receipt", str(receipt), "--out", str(tmp_path / "qa")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "metadata_only" in result.stdout
    result = subprocess.run([sys.executable, "-c", "import sys; import deep_anc.data.archive_audit; assert 'torch' not in sys.modules; assert 'sounddevice' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def git_receipt(receipt):
    value = json.loads(receipt.read_text())
    value.update(source_checksum_verified=False, source_content_verified=True,
                 source_verification="official_git_commit_and_objects", git_fsck_verified=True,
                 git_source_url=ESC50_GIT_URL, git_commit=ESC50_APPROVED_COMMIT)
    return value


def test_git_archive_provenance_never_becomes_official_archive_checksum(tmp_path):
    archive, receipt = make_archive(tmp_path, [("ESC-50/audio/fixture.wav", audio_bytes(), "file")], "tar")
    value = git_receipt(receipt)
    receipt.write_text(json.dumps(value))
    report = audit_public_archive(archive, receipt, tmp_path / "qa")
    assert report["source_archive_content_qa_passed"]
    assert report["source_content_verified"]
    assert report["source_checksum_verified"] is False
    assert report["source_checksum_scope"] == "locally_generated_archive"
    assert report["git_commit"] == ESC50_APPROVED_COMMIT
    assert report["provenance_attestation_only"]
    assert not report["training_ready"] and not report["data_ready"]


@pytest.mark.parametrize("field,value", [
    ("source_checksum_verified", True), ("source_content_verified", False),
    ("source_verification", "unknown"), ("git_fsck_verified", 1),
    ("git_source_url", "https://github.com/unapproved/ESC-50.git"),
    ("git_commit", "0" * 40), ("source_checksum_algorithm", "md5"),
    ("source_checksum_scope", "official_archive_bytes"), ("source_checksum", "0" * 64),
])
def test_git_receipt_requires_exact_approved_provenance(tmp_path, field, value):
    archive, receipt = make_archive(tmp_path, [("README", b"fixture", "file")], "tar")
    data = git_receipt(receipt)
    data[field] = value
    with pytest.raises(ValueError):
        validate_staging_provenance(data)


def test_checksum_and_git_provenance_must_not_mix(tmp_path):
    archive, receipt = make_archive(tmp_path, [("README", b"fixture", "file")])
    data = json.loads(receipt.read_text())
    official = validate_staging_provenance(data)
    assert official["source_checksum_verified"] is True
    assert official["source_checksum_scope"] == "official_archive_bytes"
    data["git_commit"] = ESC50_APPROVED_COMMIT
    with pytest.raises(ValueError, match="충돌"):
        validate_staging_provenance(data)
