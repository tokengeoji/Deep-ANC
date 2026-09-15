"""조각/원본 보존·정확한 순서/해시·모든 파일 검증. 실제 다운로드/업로드 없음."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from deep_anc.data import archive_parts as parts
from deep_anc.data.archive_staging import ESC50_APPROVED_COMMIT, ESC50_GIT_URL


DATA = (b"0123456789abcdef" * (2 * parts.MIB // 16 + 2))[:2 * parts.MIB + 19]


@pytest.fixture
def staged(tmp_path):
    directory = tmp_path / "staged"
    directory.mkdir()
    archive = directory / "fixture.zip"
    archive.write_bytes(DATA)
    receipt = {"schema_version": 1, "temporary_local_staging": True, "name": archive.name,
               "archive_path": str(archive), "bytes": len(DATA), "size": len(DATA),
               "sha256": hashlib.sha256(DATA).hexdigest(), "md5": hashlib.md5(DATA).hexdigest(),
               "source_url": "https://www.openslr.org/resources/12/fixture.zip", "license": "CC BY 4.0",
               "source_checksum_verified": True, "source_checksum_algorithm": "sha1",
               "source_checksum": hashlib.sha1(DATA).hexdigest()}
    path = directory / "receipt.json"
    path.write_text(json.dumps(receipt))
    return path, archive


def test_split_and_verify_exact_order_sizes_hashes_without_restoring(staged, tmp_path):
    receipt, archive = staged
    out = tmp_path / "parts"
    manifest = parts.split_staged_archive(receipt, out, part_mib=1, chunk_bytes=4093)
    assert manifest["part_count"] == 3
    assert [p["name"] for p in manifest["parts"]] == [f"fixture.zip.part-{i:04d}-of-0003" for i in range(3)]
    assert [p["size"] for p in manifest["parts"]] == [parts.MIB, parts.MIB, 19]
    assert all(p["size"] < parts.MAX_PART_BYTES_EXCLUSIVE for p in manifest["parts"])
    assert not manifest["ready"] and not manifest["drive_upload_verified"]
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in out.iterdir()}
    verified = parts.verify_archive_parts(out / "manifest.json", chunk_bytes=4093)
    assert verified["sha256"] == hashlib.sha256(DATA).hexdigest()
    assert verified["md5"] == hashlib.md5(DATA).hexdigest() and verified["bytes"] == len(DATA)
    assert verified["local_parts_integrity_verified"] and not verified["ready"]
    assert not verified["restored_file_written"] and not verified["local_deletion_performed"]
    assert before == {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in out.iterdir()}
    assert archive.read_bytes() == DATA


@pytest.mark.parametrize("kind", ["checksum_flag", "sha", "size", "official_checksum", "path"])
def test_invalid_original_receipt_fails_before_output(staged, tmp_path, kind):
    receipt, _ = staged
    metadata = json.loads(receipt.read_text())
    if kind == "checksum_flag":
        metadata["source_checksum_verified"] = False
    elif kind == "sha":
        metadata["sha256"] = "0" * 64
    elif kind == "size":
        metadata["size"] += 1
    elif kind == "official_checksum":
        metadata["source_checksum"] = "0" * 40
    else:
        other = tmp_path / metadata["name"]
        other.write_bytes(DATA)
        metadata["archive_path"] = str(other)
    receipt.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        parts.split_staged_archive(receipt, tmp_path / "new", part_mib=1)
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("part_mib", [0, 100, 101, True, 1.5])
def test_parts_must_be_strictly_below_100_mib(staged, tmp_path, part_mib):
    with pytest.raises(ValueError):
        parts.split_staged_archive(staged[0], tmp_path / "new", part_mib=part_mib)
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("kind", ["content", "missing", "extra", "order", "number", "size", "hash", "full_hash", "symlink"])
def test_verifier_checks_every_file_and_does_not_modify_failures(staged, tmp_path, kind):
    out = tmp_path / "parts"
    manifest = parts.split_staged_archive(staged[0], out, part_mib=1)
    first = out / manifest["parts"][0]["name"]
    if kind == "content":
        with first.open("r+b") as handle:
            handle.write(b"!")
    elif kind == "missing":
        first.rename(tmp_path / "preserved_part")
    elif kind == "extra":
        (out / "unexpected.part").write_bytes(b"extra")
    elif kind == "order":
        manifest["parts"][0], manifest["parts"][1] = manifest["parts"][1], manifest["parts"][0]
    elif kind == "number":
        manifest["parts"][0]["index"] = False
    elif kind == "size":
        manifest["parts"][0]["size"] -= 1
    elif kind == "hash":
        manifest["parts"][0]["sha256"] = "0" * 64
    elif kind == "full_hash":
        manifest["archive"]["sha256"] = "0" * 64
    else:
        saved = tmp_path / "preserved_part"
        first.rename(saved)
        first.symlink_to(saved)
    (out / "manifest.json").write_text(json.dumps(manifest))
    before = {p.name for p in out.iterdir()}
    with pytest.raises((ValueError, FileNotFoundError)):
        parts.verify_archive_parts(out / "manifest.json")
    assert {p.name for p in out.iterdir()} == before
    assert staged[1].read_bytes() == DATA


def test_failed_write_preserves_partial_part_and_original(staged, tmp_path, monkeypatch):
    def failure(_):
        raise OSError("mock disk write failure")
    monkeypatch.setattr(parts.os, "fsync", failure)
    out = tmp_path / "failed"
    with pytest.raises(OSError):
        parts.split_staged_archive(staged[0], out, part_mib=1)
    assert not (out / "manifest.json").exists()
    assert len(list(out.iterdir())) == 1
    assert next(out.iterdir()).stat().st_size == parts.MIB
    assert staged[1].read_bytes() == DATA


@pytest.mark.parametrize("kind", ["existing", "broken", "ancestor", "parent"])
def test_output_path_refusal_preserves_files(staged, tmp_path, kind):
    real = tmp_path / "real"
    real.mkdir()
    (real / "keep").write_text("keep")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "missing")
    out = {"existing": real, "broken": broken, "ancestor": link / "new", "parent": tmp_path / "x" / ".." / "new"}[kind]
    with pytest.raises((ValueError, FileExistsError)):
        parts.split_staged_archive(staged[0], out)
    assert (real / "keep").read_text() == "keep"


def test_cli_split_verify_and_no_output_on_failure(staged, tmp_path, monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts/data/split_staged_archive.py"
    spec = importlib.util.spec_from_file_location("split_staged_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setenv("DEEP_ANC_CONTAINER", "cpu")
    out = tmp_path / "cli_parts"
    arguments = ["--staging-receipt", str(staged[0]), "--out", str(out), "--part-mib", "1"]
    assert cli.main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["ready"] is False
    assert cli.main(["--verify-manifest", str(out / "manifest.json")]) == 0
    assert json.loads(capsys.readouterr().out)["local_parts_integrity_verified"]
    assert cli.main(arguments) == 1
    assert capsys.readouterr().out == ""


def test_git_provenance_keeps_archive_checksum_unverified(staged, tmp_path):
    receipt, _ = staged
    metadata = json.loads(receipt.read_text())
    metadata.update(source_url=ESC50_GIT_URL, source_checksum_verified=False,
                    source_content_verified=True, source_verification="official_git_commit_and_objects",
                    git_source_url=ESC50_GIT_URL, git_commit=ESC50_APPROVED_COMMIT, git_fsck_verified=True,
                    source_checksum_algorithm="sha256", source_checksum=metadata["sha256"])
    receipt.write_text(json.dumps(metadata))
    out = tmp_path / "git_parts"
    manifest = parts.split_staged_archive(receipt, out, part_mib=1)
    assert manifest["archive"]["source_checksum_verified"] is False
    assert manifest["archive"]["source_content_verified"] is True
    assert manifest["archive"]["source_checksum_scope"] == "locally_generated_archive"
    verified = parts.verify_archive_parts(out / "manifest.json")
    assert verified["source_checksum_verified"] is False
    assert verified["source_content_verified"] is True and verified["local_parts_integrity_verified"] is True
    assert verified["provenance_attestation_only"] is True and verified["ready"] is False


@pytest.mark.parametrize("field,value", [("git_commit", "0" * 40), ("git_fsck_verified", False),
                                         ("source_checksum_verified", True), ("source_checksum", "0" * 64)])
def test_invalid_git_provenance_does_not_create_parts(staged, tmp_path, field, value):
    receipt, _ = staged
    metadata = json.loads(receipt.read_text())
    metadata.update(source_url=ESC50_GIT_URL, source_checksum_verified=False,
                    source_content_verified=True, source_verification="official_git_commit_and_objects",
                    git_source_url=ESC50_GIT_URL, git_commit=ESC50_APPROVED_COMMIT, git_fsck_verified=True,
                    source_checksum_algorithm="sha256", source_checksum=metadata["sha256"])
    metadata[field] = value
    receipt.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        parts.split_staged_archive(receipt, tmp_path / "new", part_mib=1)
    assert not (tmp_path / "new").exists()


@pytest.fixture
def restorable(staged, tmp_path):
    directory = tmp_path / "parts"
    manifest = parts.split_staged_archive(staged[0], directory, part_mib=1, chunk_bytes=4093)
    return directory / "manifest.json", manifest


def test_restore_new_file_matches_original_and_keeps_inputs_read_only(restorable, staged, tmp_path):
    path, manifest = restorable
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns, p.read_bytes()) for p in path.parent.iterdir()}
    restored = tmp_path / "restored" / "fixture.zip"
    report = parts.restore_archive_parts(path, restored, chunk_bytes=4093)
    assert restored.read_bytes() == staged[1].read_bytes() == DATA
    assert report["bytes"] == report["size"] == len(DATA)
    assert report["sha256"] == hashlib.sha256(DATA).hexdigest()
    assert report["md5"] == hashlib.md5(DATA).hexdigest()
    assert report["source_checksum"] == hashlib.sha1(DATA).hexdigest()
    assert report["source_checksum_verified"] and report["source_content_verified"]
    assert report["schema"] == "staged_archive_parts_restore.v1"
    assert report["archive_path"] == str(restored) and report["manifest_path"] == str(path)
    assert report["restored_file_written"] and report["restored_file_integrity_verified"]
    assert report["output_readback_verified"] and report["temporary_local_staging"]
    assert not any(report[key] for key in ("ready", "drive_upload_verified", "local_deletion_performed",
                                         "pcm_qa_verified", "extraction_performed"))
    assert before == {p.name: (p.stat().st_size, p.stat().st_mtime_ns, p.read_bytes()) for p in path.parent.iterdir()}
    assert parts.verify_archive_parts(path)["sha256"] == report["sha256"]


def test_restore_preserves_git_provenance_without_promoting_publisher_checksum(staged, tmp_path):
    receipt, _ = staged
    metadata = json.loads(receipt.read_text())
    metadata.update(source_url=ESC50_GIT_URL, source_checksum_verified=False,
                    source_content_verified=True, source_verification="official_git_commit_and_objects",
                    git_source_url=ESC50_GIT_URL, git_commit=ESC50_APPROVED_COMMIT, git_fsck_verified=True,
                    source_checksum_algorithm="sha256", source_checksum=metadata["sha256"])
    receipt.write_text(json.dumps(metadata))
    directory = tmp_path / "git_parts"
    parts.split_staged_archive(receipt, directory, part_mib=1)
    report = parts.restore_archive_parts(directory / "manifest.json", tmp_path / "restored.zip")
    assert report["source_checksum_verified"] is False and report["source_content_verified"] is True
    assert report["source_checksum_scope"] == "locally_generated_archive"
    assert report["git_commit"] == ESC50_APPROVED_COMMIT and report["provenance_attestation_only"]
    assert report["restored_file_integrity_verified"] and not report["ready"]


@pytest.mark.parametrize("kind", ["content", "missing", "extra", "order", "hash", "full_hash", "md5", "symlink"])
def test_restore_preflight_failure_creates_no_output(restorable, tmp_path, kind):
    path, manifest = restorable
    first = path.parent / manifest["parts"][0]["name"]
    if kind == "content":
        with first.open("r+b") as output:
            output.write(b"!")
    elif kind == "missing":
        first.rename(tmp_path / "preserved_part")
    elif kind == "extra":
        (path.parent / "extra_file").write_text("fixture")
    elif kind == "order":
        manifest["parts"].reverse()
    elif kind == "hash":
        manifest["parts"][0]["sha256"] = "0" * 64
    elif kind == "full_hash":
        manifest["archive"]["sha256"] = "0" * 64
    elif kind == "md5":
        manifest["archive"]["md5"] = "0" * 32
    else:
        saved = tmp_path / "preserved_part"
        first.rename(saved)
        first.symlink_to(saved)
    path.write_text(json.dumps(manifest))
    names_before = {p.name for p in path.parent.iterdir()}
    output = tmp_path / "new_directory" / "restored.zip"
    with pytest.raises((ValueError, FileNotFoundError)):
        parts.restore_archive_parts(path, output)
    assert not output.parent.exists()
    assert names_before == {p.name for p in path.parent.iterdir()}


@pytest.mark.parametrize("kind", ["file", "directory", "broken_link", "ancestor_link", "inside", "nested_inside", "parent_move", "broad"])
def test_restore_destination_refusal_never_overwrites(restorable, staged, tmp_path, kind):
    path, _ = restorable
    existing = tmp_path / "existing"
    existing.mkdir()
    kept = existing / "keep"
    kept.write_bytes(b"preserve")
    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "missing")
    link = tmp_path / "link"
    link.symlink_to(existing, target_is_directory=True)
    output = {"file": kept, "directory": existing, "broken_link": broken,
              "ancestor_link": link / "new", "inside": path.parent / "restored.zip",
              "nested_inside": path.parent / "nested" / "restored.zip",
              "parent_move": tmp_path / "x" / ".." / "restored.zip", "broad": Path("/tmp")}[kind]
    names = {p.name for p in path.parent.iterdir()}
    with pytest.raises((ValueError, FileExistsError)):
        parts.restore_archive_parts(path, output)
    assert kept.read_bytes() == b"preserve" and staged[1].read_bytes() == DATA
    assert names == {p.name for p in path.parent.iterdir()}


@pytest.mark.parametrize("kind", ["manifest", "part", "shorter", "longer"])
def test_restore_detects_input_change_after_preflight(restorable, tmp_path, monkeypatch, kind):
    path, manifest = restorable
    first = path.parent / manifest["parts"][0]["name"]
    original_verify = parts.verify_archive_parts

    def mutate_after_verification(*args, **kwargs):
        verified = original_verify(*args, **kwargs)
        if kind == "manifest":
            path.write_text(path.read_text() + " ")
        elif kind == "part":
            with first.open("r+b") as output:
                output.write(b"!")
        elif kind == "shorter":
            with first.open("r+b") as output:
                output.truncate(4093)
        else:
            with first.open("ab") as output:
                output.write(b"!")
        return verified

    monkeypatch.setattr(parts, "verify_archive_parts", mutate_after_verification)
    output = tmp_path / "restored.zip"
    with pytest.raises(ValueError):
        parts.restore_archive_parts(path, output, chunk_bytes=4093)
    if kind == "manifest":
        assert not output.exists()
    else:
        assert output.is_file() and output.stat().st_size > 0


@pytest.mark.parametrize("kind", ["manifest", "part", "extra", "output", "fsync"])
def test_restore_failure_after_copy_preserves_file_and_rejects_readback_corruption(restorable, tmp_path, monkeypatch, kind):
    path, manifest = restorable
    first = path.parent / manifest["parts"][0]["name"]
    real_fsync = parts.os.fsync

    def mutate_during_flush(fd):
        real_fsync(fd)
        if kind == "manifest":
            path.write_text(path.read_text() + " ")
        elif kind == "part":
            with first.open("r+b") as output:
                output.write(b"!")
        elif kind == "extra":
            (path.parent / "extra").write_bytes(b"fixture")
        elif kind == "output":
            parts.os.pwrite(fd, b"!", 0)
        else:
            raise OSError("fixture fsync failure")

    monkeypatch.setattr(parts.os, "fsync", mutate_during_flush)
    output = tmp_path / "restored.zip"
    with pytest.raises((ValueError, OSError)):
        parts.restore_archive_parts(path, output, chunk_bytes=4093)
    assert output.is_file() and output.stat().st_size == len(DATA)
    assert output.read_bytes() == (b"!" + DATA[1:] if kind == "output" else DATA)


class _BoundedRestoreStream:
    def __init__(self, handle, *, limit, reads, writes, short_write=False):
        self.handle = handle
        self.limit = limit
        self.reads = reads
        self.writes = writes
        self.short_write = short_write

    def __enter__(self):
        self.handle.__enter__()
        return self

    def __exit__(self, *args):
        return self.handle.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def read(self, size=-1):
        assert 0 < size <= self.limit
        self.reads.append(size)
        return self.handle.read(size)

    def write(self, value):
        assert 0 < len(value) <= self.limit
        self.writes.append(len(value))
        return self.handle.write(value[:-1] if self.short_write else value)


def test_restore_rejects_same_inode_same_size_write_to_already_read_output(restorable, tmp_path, monkeypatch):
    path, _ = restorable
    destination = tmp_path / "restored.zip"
    real_open = Path.open
    mutation = {}

    class ChangedAfterRead(_BoundedRestoreStream):
        def read(self, size=-1):
            original_block = super().read(size)
            if original_block and not mutation:
                # 첫 블록은 원본 byte 그대로 반환한 뒤, 별도 FD로 이미 읽은 위치를 변경한다.
                # 해시 입력/파일 크기/inode는 그대로여서 이전 구현은 잘못 통과하던 경우다.
                before = destination.stat()
                with real_open(destination, "r+b") as other:
                    assert parts.os.pwrite(other.fileno(), b"!", 0) == 1
                after = destination.stat()
                mutation.update(before=before, after=after, returned=original_block)
            return original_block

    def wrapped_open(p, mode="r", *args, **kwargs):
        handle = real_open(p, mode, *args, **kwargs)
        if p == destination and mode == "x+b":
            return ChangedAfterRead(handle, limit=4093, reads=[], writes=[])
        return handle

    monkeypatch.setattr(Path, "open", wrapped_open)
    with pytest.raises(ValueError, match="출력 경로/파일 상태"):
        parts.restore_archive_parts(path, destination, chunk_bytes=4093)
    assert mutation["returned"] == DATA[:4093]
    before, after = mutation["before"], mutation["after"]
    assert (before.st_dev, before.st_ino, before.st_size) == (after.st_dev, after.st_ino, after.st_size)
    assert (before.st_mtime_ns, before.st_ctime_ns) != (after.st_mtime_ns, after.st_ctime_ns)
    assert destination.read_bytes() == b"!" + DATA[1:]
    assert parts.verify_archive_parts(path)["sha256"] == hashlib.sha256(DATA).hexdigest()


@pytest.mark.parametrize("short_write", [False, True])
def test_restore_uses_bounded_reads_writes_and_preserves_short_write_failure(restorable, tmp_path, monkeypatch, short_write):
    path, manifest = restorable
    destination = tmp_path / "restored.zip"
    originals = {path.parent / part["name"] for part in manifest["parts"]}
    real_open = Path.open
    reads, writes = [], []

    def checked_open(p, mode="r", *args, **kwargs):
        handle = real_open(p, mode, *args, **kwargs)
        if p in originals or p == destination:
            return _BoundedRestoreStream(handle, limit=4093, reads=reads, writes=writes,
                                         short_write=short_write and p == destination)
        return handle

    monkeypatch.setattr(Path, "open", checked_open)
    if short_write:
        with pytest.raises(OSError, match="쓰기"):
            parts.restore_archive_parts(path, destination, chunk_bytes=4093)
        assert destination.stat().st_size == 4092
    else:
        report = parts.restore_archive_parts(path, destination, chunk_bytes=4093)
        assert report["restored_file_integrity_verified"]
    assert reads and writes and max(reads) <= 4093 and max(writes) <= 4093


@pytest.mark.parametrize("chunk_bytes", [0, True, 65 * parts.MIB])
def test_restore_rejects_invalid_memory_chunk_before_output(restorable, tmp_path, chunk_bytes):
    output = tmp_path / "restored.zip"
    with pytest.raises(ValueError):
        parts.restore_archive_parts(restorable[0], output, chunk_bytes=chunk_bytes)
    assert not output.exists()


def test_restore_cli_success_and_invalid_mode_combinations(restorable, tmp_path, monkeypatch, capsys):
    path, _ = restorable
    script = Path(__file__).resolve().parents[1] / "scripts/data/split_staged_archive.py"
    spec = importlib.util.spec_from_file_location("restore_staged_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setenv("DEEP_ANC_CONTAINER", "cpu")
    output = tmp_path / "restored.zip"
    arguments = ["--restore-manifest", str(path), "--archive-out", str(output)]
    assert cli.main(arguments) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["restored_file_written"] and report["output_readback_verified"] and not report["ready"]
    for arguments in (
        arguments,
        ["--restore-manifest", str(path)],
        ["--restore-manifest", str(path), "--archive-out", str(tmp_path / "new"), "--out", str(tmp_path / "other")],
        ["--verify-manifest", str(path), "--archive-out", str(tmp_path / "new")],
        ["--staging-receipt", str(path), "--archive-out", str(tmp_path / "new")],
    ):
        assert cli.main(arguments) == 1
        captured = capsys.readouterr()
        assert captured.out == "" and "[실패]" in captured.err
    assert output.read_bytes() == DATA and not (tmp_path / "new").exists()
