"""공개 archive cache의 고정 allowlist/no-replace 경계."""

from __future__ import annotations

import bz2
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import types
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.elice import public_archive_cache as cache


def _zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members:
            archive.writestr(name, payload)


def _tar_bz2(path: Path, members: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w:bz2") as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(payload))


def _tiny_spec(
    path: Path,
    *,
    archive_id: str = "tiny",
    target: str = "data/raw/noise/tiny.zip",
    prefix: str = "root/",
    count: int = 1,
    total_bytes: int | None = 3,
) -> cache.ArchiveSpec:
    return cache.ArchiveSpec(
        archive_id=archive_id,
        corpus="fixture",
        url=f"https://example.invalid/{path.name}",
        filename=path.name,
        canonical_target=target,
        archive_format="tar.bz2" if path.name.endswith(".tar.bz2") else "zip",
        expected_size=path.stat().st_size,
        provider_checksum_kind="md5",
        provider_checksum=hashlib.md5(path.read_bytes()).hexdigest(),  # noqa: S324
        provider_etag=None,
        member_prefix=prefix,
        expected_wav_count=count,
        expected_wav_bytes=total_bytes,
    )


def test_fixed_allowlist_is_exactly_dns3_demand6_mimii1():
    assert cache.EXPECTED_IDS == (
        "dns_noise_000",
        "dns_noise_001",
        "dns_speech_000",
        "demand_dkitchen",
        "demand_dwashing",
        "demand_ooffice",
        "demand_ohallway",
        "demand_tmetro",
        "demand_tcar",
        "mimii_fan",
    )
    assert {spec.corpus for spec in cache.ARCHIVE_SPECS} == {
        "dns3",
        "demand6",
        "mimii1",
    }
    serialized = cache._canonical_json(  # noqa: SLF001
        [spec.__dict__ for spec in cache.ARCHIVE_SPECS]
    ).decode("utf-8")
    assert "librispeech" not in serialized.casefold()
    assert "esc50" not in serialized.casefold()
    assert "fma_small" not in serialized.casefold()


@pytest.mark.parametrize("suffix", [".zip", ".tar.bz2"])
def test_archive_validator_accepts_crc_structure_and_provider_checksum(
    tmp_path: Path, suffix: str
):
    archive = tmp_path / f"tiny{suffix}"
    members = [("root/a.wav", b"abc")]
    (_zip if suffix == ".zip" else _tar_bz2)(archive, members)
    spec = _tiny_spec(archive)

    result = cache.validate_archive(archive, spec)

    assert result["archive_size"] == archive.stat().st_size
    assert result["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert result["wav_count"] == 1
    assert result["wav_bytes"] == 3
    assert len(str(result["member_inventory_sha256"])) == 64


@pytest.mark.parametrize("name", ["../x.wav", "/root/x.wav", "root//x.wav", "root\\x.wav"])
def test_zip_validator_rejects_traversal_and_non_posix_paths(tmp_path: Path, name: str):
    archive = tmp_path / "bad.zip"
    _zip(archive, [(name, b"abc")])
    spec = _tiny_spec(archive)

    with pytest.raises(cache.ArchiveCacheError):
        cache.validate_archive(archive, spec)


def test_zip_validator_rejects_symlink_and_duplicate_members(tmp_path: Path):
    symlink_archive = tmp_path / "symlink.zip"
    with zipfile.ZipFile(symlink_archive, "w") as archive:
        info = zipfile.ZipInfo("root/a.wav")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "target")
    symlink_spec = _tiny_spec(
        symlink_archive, total_bytes=len(b"target")
    )

    with pytest.raises(cache.ArchiveCacheError, match="non-regular"):
        cache.validate_archive(symlink_archive, symlink_spec)

    duplicate_archive = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        _zip(
            duplicate_archive,
            [("root/a.wav", b"abc"), ("root/a.wav", b"abc")],
        )
    duplicate_spec = _tiny_spec(
        duplicate_archive, count=2, total_bytes=6
    )
    with pytest.raises(cache.ArchiveCacheError, match="duplicate"):
        cache.validate_archive(duplicate_archive, duplicate_spec)


def test_tar_validator_rejects_link_member(tmp_path: Path):
    archive_path = tmp_path / "link.tar.bz2"
    with tarfile.open(archive_path, "w:bz2") as archive:
        member = tarfile.TarInfo("root/a.wav")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        archive.addfile(member)
    spec = _tiny_spec(archive_path, total_bytes=0)

    with pytest.raises(cache.ArchiveCacheError, match="non-regular"):
        cache.validate_archive(archive_path, spec)


def test_corrupted_zip_crc_and_bzip2_stream_are_blocked(tmp_path: Path):
    zip_path = tmp_path / "crc.zip"
    _zip(zip_path, [("root/a.wav", b"abcdefghijklmno")])
    with zipfile.ZipFile(zip_path) as archive:
        member = archive.infolist()[0]
        raw = bytearray(zip_path.read_bytes())
        filename_length = int.from_bytes(raw[member.header_offset + 26 : member.header_offset + 28], "little")
        extra_length = int.from_bytes(raw[member.header_offset + 28 : member.header_offset + 30], "little")
        payload_offset = member.header_offset + 30 + filename_length + extra_length
    raw[payload_offset] ^= 0xFF
    zip_path.write_bytes(raw)
    zip_spec = _tiny_spec(zip_path, total_bytes=15)
    with pytest.raises(cache.ArchiveCacheError, match="ZIP"):
        cache.validate_archive(zip_path, zip_spec)

    tar_path = tmp_path / "crc.tar.bz2"
    _tar_bz2(tar_path, [("root/a.wav", b"abcdefghijklmno")])
    compressed = bytearray(tar_path.read_bytes())
    compressed[len(compressed) // 2] ^= 0xFF
    tar_path.write_bytes(compressed)
    tar_spec = _tiny_spec(tar_path, total_bytes=15)
    with pytest.raises(cache.ArchiveCacheError, match="bzip2"):
        cache.validate_archive(tar_path, tar_spec)


def test_tar_validator_rejects_traversal_and_duplicate_members(tmp_path: Path):
    traversal = tmp_path / "traversal.tar.bz2"
    _tar_bz2(traversal, [("../outside.wav", b"abc")])
    with pytest.raises(cache.ArchiveCacheError, match="traversal"):
        cache.validate_archive(traversal, _tiny_spec(traversal))

    duplicate = tmp_path / "duplicate.tar.bz2"
    _tar_bz2(
        duplicate,
        [("root/a.wav", b"abc"), ("root/a.wav", b"def")],
    )
    with pytest.raises(cache.ArchiveCacheError, match="duplicate"):
        cache.validate_archive(
            duplicate, _tiny_spec(duplicate, count=2, total_bytes=6)
        )


def test_provider_md5_size_and_member_inventory_mismatch_are_blocked(tmp_path: Path):
    archive = tmp_path / "tiny.zip"
    _zip(archive, [("root/a.wav", b"abc")])
    spec = _tiny_spec(archive)

    with pytest.raises(cache.ArchiveCacheError, match="provider MD5"):
        cache.validate_archive(archive, replace(spec, provider_checksum="0" * 32))
    with pytest.raises(cache.ArchiveCacheError, match="size"):
        cache.validate_archive(archive, replace(spec, expected_size=spec.expected_size + 1))
    with pytest.raises(cache.ArchiveCacheError, match="WAV count"):
        cache.validate_archive(archive, replace(spec, expected_wav_count=2))


def test_copy_no_replace_preserves_source_and_never_overwrites_target(tmp_path: Path):
    source = tmp_path / "source.zip"
    source.write_bytes(b"source-bytes")
    source_before = (source.stat().st_ino, source.stat().st_mtime_ns, source.read_bytes())
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target = target_dir / "archive.zip"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    assert cache._copy_no_replace(  # noqa: SLF001
        source, target, size=source.stat().st_size, sha256=digest
    ) == "restored"
    assert target.read_bytes() == b"source-bytes"
    assert (source.stat().st_ino, source.stat().st_mtime_ns, source.read_bytes()) == source_before
    assert cache._copy_no_replace(  # noqa: SLF001
        source, target, size=source.stat().st_size, sha256=digest
    ) == "already_exact"

    target.write_bytes(b"different")
    with pytest.raises(cache.ArchiveCacheError, match="이미 다른"):
        cache._copy_no_replace(  # noqa: SLF001
            source, target, size=source.stat().st_size, sha256=digest
        )
    assert target.read_bytes() == b"different"


def test_failed_copy_preserves_forensic_staging_and_does_not_publish(tmp_path: Path):
    source = tmp_path / "source.zip"
    source.write_bytes(b"source-bytes")
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target = target_dir / "archive.zip"

    with pytest.raises(cache.ArchiveCacheError, match="forensic staging"):
        cache._copy_no_replace(  # noqa: SLF001
            source,
            target,
            size=source.stat().st_size,
            sha256="0" * 64,
        )

    assert not target.exists()
    staged = list(target_dir.glob(".archive.zip.archive-cache-restore.*"))
    assert len(staged) == 1
    assert staged[0].read_bytes() == source.read_bytes()


def test_safe_target_rejects_symlink_parent_before_any_outside_write(tmp_path: Path):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    (repo / "data").symlink_to(outside, target_is_directory=True)

    with pytest.raises(cache.ArchiveCacheError, match="symlink/non-directory"):
        cache._safe_target(repo, "data/raw/noise/archive.zip")  # noqa: SLF001

    assert list(outside.iterdir()) == []


def test_pget_failure_preserves_partial_in_unique_campaign_staging(tmp_path: Path):
    pget = tmp_path / "pget_fixture.py"
    pget.write_text(
        "from pathlib import Path\n"
        "class DownloadError(RuntimeError):\n"
        "    pass\n"
        "def _remove_regular(path):\n"
        "    Path(path).unlink(missing_ok=True)\n"
        "    return True\n"
        "def download(url, output, connections):\n"
        "    part = Path(str(output) + '.part')\n"
        "    part.write_bytes(b'forensic-partial')\n"
        "    _remove_regular(part)\n"
        "    raise DownloadError('fixture failure')\n",
        encoding="utf-8",
    )
    trusted_source = pget.read_bytes()
    executed = tmp_path / "unverified-pget-executed"
    pget.write_text(
        "from pathlib import Path\n"
        f"Path({str(executed)!r}).write_text('executed', encoding='utf-8')\n"
        "class DownloadError(RuntimeError):\n"
        "    pass\n"
        "def download(url, output, connections):\n"
        "    raise DownloadError('unverified replacement')\n",
        encoding="utf-8",
    )
    output = tmp_path / "archive.zip"
    spec = cache.ArchiveSpec(
        archive_id="fixture",
        corpus="fixture",
        url="https://example.invalid/archive.zip",
        filename=output.name,
        canonical_target="data/raw/noise/archive.zip",
        archive_format="zip",
        expected_size=1,
        provider_checksum_kind="none",
        provider_checksum=None,
        provider_etag=None,
        member_prefix="root/",
        expected_wav_count=1,
        expected_wav_bytes=1,
    )

    with pytest.raises(cache.ArchiveCacheError, match="partial을 보존"):
        cache._run_pget(  # noqa: SLF001
            pget, spec, output, trusted_source=trusted_source
        )

    assert Path(f"{output}.part").read_bytes() == b"forensic-partial"
    assert not executed.exists()


def test_trusted_tracked_pget_blob_executes_dataclass_from_held_bytes():
    pget = Path(cache.__file__).with_name("pget.py").resolve()
    module_name = "_archive_cache_actual_pget_compile_regression"

    namespace = cache._exec_trusted_source(  # noqa: SLF001
        pget.read_bytes(), filename=pget, module_name=module_name
    )

    assert namespace["DownloadError"].__name__ == "DownloadError"
    assert callable(namespace["download"])
    assert module_name not in sys.modules


def test_exclusive_manifest_write_handles_short_writes_and_preserves_zero_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    real_write = cache.os.write

    def short_write(descriptor, data):
        return real_write(descriptor, bytes(data[:3]))

    monkeypatch.setattr(cache.os, "write", short_write)
    target = tmp_path / "manifest.json"
    payload = b'{"complete":true}\n'
    assert cache._write_exclusive_fsynced(target, payload) == hashlib.sha256(  # noqa: SLF001
        payload
    ).hexdigest()
    assert target.read_bytes() == payload

    zero_target = tmp_path / "zero.json"
    calls = 0

    def then_zero(descriptor, data):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, bytes(data[:2]))
        return 0

    monkeypatch.setattr(cache.os, "write", then_zero)
    with pytest.raises(cache.ArchiveCacheError, match="forensic staging"):
        cache._write_exclusive_fsynced(zero_target, payload)  # noqa: SLF001
    assert zero_target.read_bytes() == payload[:2]


def _cache_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, str, tuple[cache.ArchiveSpec, ...]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    pget = repo / "scripts/elice/pget.py"
    pget.parent.mkdir(parents=True)
    pget.write_text("# restore fixture\n", encoding="utf-8")
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    specs: list[cache.ArchiveSpec] = []
    entries: list[dict[str, object]] = []
    for index in range(2):
        scratch = tmp_path / f"source-{index}.zip"
        payload = f"wav-{index}".encode()
        _zip(scratch, [(f"root{index}/a.wav", payload)])
        spec = _tiny_spec(
            scratch,
            archive_id=f"fixture_{index}",
            target=f"data/raw/noise/fixture-{index}.zip",
            prefix=f"root{index}/",
            total_bytes=len(payload),
        )
        details = cache.validate_archive(scratch, spec)
        entry = cache._manifest_entry(spec, details)  # noqa: SLF001
        cached = incoming / str(entry["cache_path"])
        cached.parent.mkdir(parents=True)
        cached.write_bytes(scratch.read_bytes())
        specs.append(spec)
        entries.append(entry)
    monkeypatch.setattr(cache, "ARCHIVE_SPECS", tuple(specs))
    monkeypatch.setattr(cache, "EXPECTED_IDS", tuple(spec.archive_id for spec in specs))
    monkeypatch.setattr(cache, "_validate_aggregate", lambda _entries: None)
    monkeypatch.setattr(cache, "_verify_exact_source", lambda *_args, **_kwargs: None)
    script_sha = cache._hash_file(Path(cache.__file__).resolve())["sha256"]  # noqa: SLF001
    pget_sha = cache._hash_file(pget)["sha256"]  # noqa: SLF001
    commit = "a" * 40
    manifest = {
        "archive_count": len(entries),
        "archives": entries,
        "authority": cache.AUTHORITY,
        "excluded_corpora": ["esc50", "fma_small", "fma_metadata", "librispeech"],
        "kind": cache.MANIFEST_KIND,
        "publisher_commit": commit,
        "publisher_entry_script_sha256": script_sha,
        "publisher_pget_sha256": pget_sha,
        "schema_version": cache.SCHEMA_VERSION,
    }
    raw = cache._canonical_json(manifest)  # noqa: SLF001
    manifest_path = incoming / "manifest.json"
    manifest_path.write_bytes(raw)
    return repo, incoming, manifest_path, hashlib.sha256(raw).hexdigest(), tuple(specs)


def _restore_argv(repo: Path, incoming: Path, manifest: Path, digest: str) -> list[str]:
    return [
        "restore",
        "--cache-root",
        str(incoming),
        "--manifest",
        str(manifest),
        "--expected-manifest-sha256",
        digest,
        "--expected-commit",
        "a" * 40,
        "--repo-root",
        str(repo),
    ]


def _consume_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, str, tuple[cache.ArchiveSpec, ...], dict[str, Path]]:
    repo = tmp_path / "consume-repo"
    repo.mkdir()
    pget = repo / "scripts/elice/pget.py"
    pget.parent.mkdir(parents=True)
    pget.write_text("# held consume fixture\n", encoding="utf-8")
    incoming = tmp_path / "consume-incoming"
    incoming.mkdir()
    seeds = (
        (
            "dns_noise_000",
            "noise.tar.bz2",
            "data/raw/noise/shard000.tar.bz2",
            "datasets_fullband/noise_fullband/",
            [("datasets_fullband/noise_fullband/good.wav", b"held-dns-good")],
        ),
        (
            "demand_dkitchen",
            "DKITCHEN_48k.zip",
            "data/raw/noise/demand/DKITCHEN_48k.zip",
            "DKITCHEN/",
            [("DKITCHEN/ch01.wav", b"held-demand-good")],
        ),
    )
    specs: list[cache.ArchiveSpec] = []
    entries: list[dict[str, object]] = []
    cached_by_id: dict[str, Path] = {}
    for archive_id, filename, target, prefix, members in seeds:
        scratch = tmp_path / filename
        (_tar_bz2 if filename.endswith(".tar.bz2") else _zip)(scratch, members)
        spec = _tiny_spec(
            scratch,
            archive_id=archive_id,
            target=target,
            prefix=prefix,
            count=len(members),
            total_bytes=sum(len(payload) for _name, payload in members),
        )
        details = cache.validate_archive(scratch, spec)
        entry = cache._manifest_entry(spec, details)  # noqa: SLF001
        cached = incoming / str(entry["cache_path"])
        cached.parent.mkdir(parents=True)
        cached.write_bytes(scratch.read_bytes())
        specs.append(spec)
        entries.append(entry)
        cached_by_id[archive_id] = cached
    monkeypatch.setattr(cache, "ARCHIVE_SPECS", tuple(specs))
    monkeypatch.setattr(cache, "EXPECTED_IDS", tuple(spec.archive_id for spec in specs))
    monkeypatch.setattr(cache, "_validate_aggregate", lambda _entries: None)
    monkeypatch.setattr(cache, "_verify_exact_source", lambda *_args, **_kwargs: None)
    script_sha = cache._hash_file(Path(cache.__file__).resolve())["sha256"]  # noqa: SLF001
    pget_sha = cache._hash_file(pget)["sha256"]  # noqa: SLF001
    manifest_payload = {
        "archive_count": len(entries),
        "archives": entries,
        "authority": cache.AUTHORITY,
        "excluded_corpora": ["esc50", "fma_small", "fma_metadata", "librispeech"],
        "kind": cache.MANIFEST_KIND,
        "publisher_commit": "a" * 40,
        "publisher_entry_script_sha256": script_sha,
        "publisher_pget_sha256": pget_sha,
        "schema_version": cache.SCHEMA_VERSION,
    }
    manifest_raw = cache._canonical_json(manifest_payload)  # noqa: SLF001
    manifest = incoming / "manifest.json"
    manifest.write_bytes(manifest_raw)
    return (
        repo,
        incoming,
        manifest,
        hashlib.sha256(manifest_raw).hexdigest(),
        tuple(specs),
        cached_by_id,
    )


def _consume_argv(repo: Path, incoming: Path, manifest: Path, digest: str) -> list[str]:
    return [
        "consume",
        "--cache-root",
        str(incoming),
        "--manifest",
        str(manifest),
        "--expected-manifest-sha256",
        digest,
        "--expected-commit",
        "a" * 40,
        "--repo-root",
        str(repo),
    ]


def _verify_consumed_argv(
    repo: Path,
    incoming: Path,
    manifest: Path,
    digest: str,
    *,
    decoder_audit: Path | None = None,
) -> list[str]:
    # The production manifest populates all four cache-owned raw roots.  This
    # two-archive fixture leaves two roots empty; materialise them only after
    # consume so the consume-time exact extracted-tree audit remains faithful.
    for relative in cache.CACHE_TRAINING_WAV_ROOTS:
        (repo / relative).mkdir(parents=True, exist_ok=True)
    arguments = [
        "verify-consumed-raw",
        "--cache-root",
        str(incoming),
        "--manifest",
        str(manifest),
        "--expected-manifest-sha256",
        digest,
        "--expected-commit",
        "a" * 40,
        "--repo-root",
        str(repo),
    ]
    if decoder_audit is not None:
        arguments.extend(("--decoder-audit", str(decoder_audit)))
    return arguments


def test_consume_extracts_tar_and_zip_through_held_fds_and_is_restart_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    repo, incoming, manifest, digest, _specs, _cached = _consume_fixture(
        tmp_path, monkeypatch
    )

    assert cache.main(_consume_argv(repo, incoming, manifest, digest)) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["state"] == "held_fd_extracted_pending_exact_raw_and_decoder_audits"
    assert first["authority"] == cache.AUTHORITY
    assert first["extracted_file_count"] == 2
    assert (repo / "data/raw/noise/dns_fullband/datasets_fullband/noise_fullband/good.wav").read_bytes() == b"held-dns-good"
    assert (repo / "data/raw/noise/demand/DKITCHEN/ch01.wav").read_bytes() == b"held-demand-good"
    assert not (repo / "data/raw/noise/shard000.tar.bz2").exists()
    assert not (repo / "data/raw/noise/demand/DKITCHEN_48k.zip").exists()
    origin = repo / first["origin_receipt"]
    assert hashlib.sha256(origin.read_bytes()).hexdigest() == first["origin_receipt_sha256"]

    # A full-bootstrap retry may safely consume the same external anchor.  It
    # byte-compares every existing raw member and never overwrites it.
    assert cache.main(_consume_argv(repo, incoming, manifest, digest)) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["origin_receipt_sha256"] == first["origin_receipt_sha256"]


def test_full_consume_accepts_only_exact_cache_only_working_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    repo, incoming, manifest, digest, _specs, _cached = _consume_fixture(
        tmp_path, monkeypatch
    )
    assert cache.main(_restore_argv(repo, incoming, manifest, digest)) == 0
    capsys.readouterr()
    demand_archive = repo / "data/raw/noise/demand/DKITCHEN_48k.zip"
    assert demand_archive.is_file()

    assert cache.main(_consume_argv(repo, incoming, manifest, digest)) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "held_fd_extracted_pending_exact_raw_and_decoder_audits"
    assert (repo / "data/raw/noise/demand/DKITCHEN/ch01.wav").read_bytes() == b"held-demand-good"
    assert demand_archive.is_file()

    # A different pre-existing working archive is never ignored merely because
    # its filename is fixed.
    demand_archive.write_bytes(b"different-working-archive")
    assert cache.main(_consume_argv(repo, incoming, manifest, digest)) == 2


def test_consume_path_swap_cannot_substitute_archive_between_validation_and_extract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, incoming, manifest, digest, _specs, cached = _consume_fixture(
        tmp_path, monkeypatch
    )
    real_expected = cache._expected_extracted_outputs  # noqa: SLF001
    swapped = False

    def swap_after_all_held_validation(validated):
        nonlocal swapped
        result = real_expected(validated)
        if not swapped:
            swapped = True
            original = cached["demand_dkitchen"]
            moved = original.with_name(f"{original.name}.validated-inode")
            original.rename(moved)
            _zip(original, [("DKITCHEN/ch01.wav", b"path-swap-evil")])
        return result

    monkeypatch.setattr(cache, "_expected_extracted_outputs", swap_after_all_held_validation)
    args = cache._build_parser().parse_args(  # noqa: SLF001
        _consume_argv(repo, incoming, manifest, digest)
    )

    with pytest.raises(cache.ArchiveCacheError, match="held archive"):
        cache.consume(args)

    # The final metadata/SHA gate blocks the receipt.  If the member was
    # published before that final gate it came from the held good inode, never
    # from the replacement pathname.
    target = repo / "data/raw/noise/demand/DKITCHEN/ch01.wav"
    if target.exists():
        assert target.read_bytes() == b"held-demand-good"
    origin_root = repo / cache.CACHE_ORIGIN_DIRECTORY
    assert not origin_root.exists() or not any(origin_root.iterdir())


def test_consume_final_held_sha_inode_gate_rejects_same_size_output_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, incoming, manifest, digest, _specs, _cached = _consume_fixture(
        tmp_path, monkeypatch
    )
    real_audit = cache._audit_existing_extracted_subset  # noqa: SLF001
    replaced = False

    def replace_after_size_only_audit(*args, **kwargs):
        nonlocal replaced
        result = real_audit(*args, **kwargs)
        if kwargs.get("require_complete") and not replaced:
            replaced = True
            target = repo / "data/raw/noise/demand/DKITCHEN/ch01.wav"
            original_size = target.stat().st_size
            replacement = target.with_name(".same-size-replacement")
            replacement.write_bytes(b"X" * original_size)
            os.replace(replacement, target)
        return result

    monkeypatch.setattr(
        cache, "_audit_existing_extracted_subset", replace_after_size_only_audit
    )
    args = cache._build_parser().parse_args(  # noqa: SLF001
        _consume_argv(repo, incoming, manifest, digest)
    )

    with pytest.raises(cache.ArchiveCacheError, match="held published"):
        cache.consume(args)

    intent, completion = cache._consumption_receipt_paths(  # noqa: SLF001
        digest, "a" * 40
    )
    assert (repo / intent).is_file()
    assert not (repo / completion).exists()
    origin_root = repo / cache.CACHE_ORIGIN_DIRECTORY
    assert not origin_root.exists() or not any(origin_root.iterdir())


def test_consume_rechecks_held_outputs_after_origin_receipt_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, incoming, manifest, digest, _specs, _cached = _consume_fixture(
        tmp_path, monkeypatch
    )
    real_origin = cache._publish_cache_origin_receipt  # noqa: SLF001
    replaced = False

    def publish_origin_then_replace(*args, **kwargs):
        nonlocal replaced
        result = real_origin(*args, **kwargs)
        if not replaced:
            replaced = True
            target = repo / "data/raw/noise/demand/DKITCHEN/ch01.wav"
            replacement = target.with_name(".origin-hook-same-size")
            replacement.write_bytes(b"X" * target.stat().st_size)
            os.replace(replacement, target)
        return result

    monkeypatch.setattr(cache, "_publish_cache_origin_receipt", publish_origin_then_replace)
    args = cache._build_parser().parse_args(  # noqa: SLF001
        _consume_argv(repo, incoming, manifest, digest)
    )

    with pytest.raises(cache.ArchiveCacheError, match="held published"):
        cache.consume(args)

    _intent, completion = cache._consumption_receipt_paths(  # noqa: SLF001
        digest, "a" * 40
    )
    assert not (repo / completion).exists()


def test_verify_consumed_raw_holds_all_paths_and_binds_decoder_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, incoming, manifest, digest, _specs, _cached = _consume_fixture(
        tmp_path, monkeypatch
    )
    consume_args = cache._build_parser().parse_args(  # noqa: SLF001
        _consume_argv(repo, incoming, manifest, digest)
    )
    result = cache.consume(consume_args)
    inventory = json.loads(
        (repo / result["consumed_member_inventory"]).read_text(encoding="utf-8")
    )
    decoder = {
        "inventory": [
            {
                "content_sha256": row["sha256"],
                "content_size": row["size"],
                "decision": "accept",
                "relative_path": row["path"],
            }
            for row in inventory["rows"]
        ],
        "schema_version": 1,
        "status": "complete",
    }
    semantic = json.dumps(
        decoder,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    decoder["audit_sha256"] = hashlib.sha256(semantic).hexdigest()
    decoder_path = repo / "results/provenance/decoder_audit.json"
    decoder_path.parent.mkdir(parents=True)
    decoder_path.write_bytes(cache._canonical_json(decoder))  # noqa: SLF001
    verify_args = cache._build_parser().parse_args(  # noqa: SLF001
        _verify_consumed_argv(
            repo,
            incoming,
            manifest,
            digest,
            decoder_audit=decoder_path,
        )
    )

    verified = cache.verify_consumed_raw(verify_args)

    assert verified["decoder_audit_file_sha256"] == hashlib.sha256(
        decoder_path.read_bytes()
    ).hexdigest()
    assert (
        verified["decoder_cache_projection_sha256"]
        == verified["current_output_projection_sha256"]
    )

    decoder["inventory"][0]["content_sha256"] = "0" * 64
    semantic = json.dumps(
        {key: value for key, value in decoder.items() if key != "audit_sha256"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    decoder["audit_sha256"] = hashlib.sha256(semantic).hexdigest()
    decoder_path.write_bytes(cache._canonical_json(decoder))  # noqa: SLF001
    with pytest.raises(cache.ArchiveCacheError, match="decoder audit/cache"):
        cache.verify_consumed_raw(verify_args)


def test_verify_consumed_raw_rejects_early_path_replacement_before_final_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, incoming, manifest, digest, _specs, _cached = _consume_fixture(
        tmp_path, monkeypatch
    )
    cache.consume(
        cache._build_parser().parse_args(  # noqa: SLF001
            _consume_argv(repo, incoming, manifest, digest)
        )
    )
    real_hold = cache._hold_published_target  # noqa: SLF001
    replaced = False

    def hold_then_replace(target, **kwargs):
        nonlocal replaced
        held = real_hold(target, **kwargs)
        if not replaced:
            replaced = True
            replacement = target.path.with_name(".verify-same-size-replacement")
            replacement.write_bytes(b"X" * int(kwargs["size"]))
            os.replace(replacement, target.path)
        return held

    monkeypatch.setattr(cache, "_hold_published_target", hold_then_replace)
    verify_args = cache._build_parser().parse_args(  # noqa: SLF001
        _verify_consumed_argv(repo, incoming, manifest, digest)
    )

    with pytest.raises(cache.ArchiveCacheError, match="held published"):
        cache.verify_consumed_raw(verify_args)


def test_verify_consumed_raw_rejects_extra_wav_added_during_decoder_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, incoming, manifest, digest, _specs, _cached = _consume_fixture(
        tmp_path, monkeypatch
    )
    result = cache.consume(
        cache._build_parser().parse_args(  # noqa: SLF001
            _consume_argv(repo, incoming, manifest, digest)
        )
    )
    inventory = json.loads(
        (repo / result["consumed_member_inventory"]).read_text(encoding="utf-8")
    )
    decoder = {
        "inventory": [
            {
                "content_sha256": row["sha256"],
                "content_size": row["size"],
                "relative_path": row["path"],
            }
            for row in inventory["rows"]
        ],
        "schema_version": 1,
        "status": "complete",
    }
    decoder["audit_sha256"] = hashlib.sha256(
        json.dumps(
            decoder,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    decoder_path = repo / "results/provenance/decoder_audit.json"
    decoder_path.parent.mkdir(parents=True)
    decoder_path.write_bytes(cache._canonical_json(decoder))  # noqa: SLF001
    verify_args = cache._build_parser().parse_args(  # noqa: SLF001
        _verify_consumed_argv(
            repo, incoming, manifest, digest, decoder_audit=decoder_path
        )
    )
    real_decoder = cache._validate_decoder_audit_projection  # noqa: SLF001

    def validate_then_add_extra(*args, **kwargs):
        bound = real_decoder(*args, **kwargs)
        extra = repo / "data/raw/noise/demand/DKITCHEN/unexpected.wav"
        extra.write_bytes(b"extra-cache-wav")
        return bound

    monkeypatch.setattr(
        cache, "_validate_decoder_audit_projection", validate_then_add_extra
    )
    with pytest.raises(cache.ArchiveCacheError, match="cache raw WAV exact-set"):
        cache.verify_consumed_raw(verify_args)


def test_decoder_projection_only_is_rejected_because_caller_scan_cannot_be_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, incoming, manifest, digest, _specs, _cached = _consume_fixture(
        tmp_path, monkeypatch
    )
    result = cache.consume(
        cache._build_parser().parse_args(  # noqa: SLF001
            _consume_argv(repo, incoming, manifest, digest)
        )
    )
    inventory = json.loads(
        (repo / result["consumed_member_inventory"]).read_text(encoding="utf-8")
    )
    decoder = {
        "inventory": [
            {
                "relative_path": row["path"],
                "content_size": row["size"],
                "content_sha256": row["sha256"],
            }
            for row in inventory["rows"]
        ],
        "schema_version": 1,
        "status": "complete",
    }
    semantic = json.dumps(
        decoder,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    decoder["audit_sha256"] = hashlib.sha256(semantic).hexdigest()
    decoder_path = repo / "results/provenance/decoder_audit.json"
    decoder_path.parent.mkdir(parents=True)
    decoder_path.write_bytes(cache._canonical_json(decoder))  # noqa: SLF001
    target = repo / str(inventory["rows"][0]["path"])
    target.write_bytes(b"X" * target.stat().st_size)
    projection_argv = _verify_consumed_argv(
        repo,
        incoming,
        manifest,
        digest,
        decoder_audit=decoder_path,
    ) + ["--decoder-projection-only"]
    projection_args = cache._build_parser().parse_args(projection_argv)  # noqa: SLF001

    with pytest.raises(cache.ArchiveCacheError, match="pathname replacement"):
        cache.verify_consumed_raw(projection_args)

    normal_args = cache._build_parser().parse_args(  # noqa: SLF001
        _verify_consumed_argv(
            repo,
            incoming,
            manifest,
            digest,
            decoder_audit=decoder_path,
        )
    )
    with pytest.raises(cache.ArchiveCacheError, match="held SHA"):
        cache.verify_consumed_raw(normal_args)


def test_decoder_audit_ancestor_symlink_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, incoming, manifest, digest, _specs, _cached = _consume_fixture(
        tmp_path, monkeypatch
    )
    cache.consume(
        cache._build_parser().parse_args(  # noqa: SLF001
            _consume_argv(repo, incoming, manifest, digest)
        )
    )
    outside = tmp_path / "outside-results"
    outside.mkdir()
    (repo / "results").symlink_to(outside, target_is_directory=True)
    decoder_path = repo / "results/provenance/decoder_audit.json"
    verify_args = cache._build_parser().parse_args(  # noqa: SLF001
        _verify_consumed_argv(
            repo,
            incoming,
            manifest,
            digest,
            decoder_audit=decoder_path,
        )
    )

    with pytest.raises(cache.ArchiveCacheError, match="symlink/non-directory"):
        cache.verify_consumed_raw(verify_args)


def test_interrupted_consume_intent_blocks_plain_bootstrap_and_exact_anchor_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, incoming, manifest, digest, _specs, _cached = _consume_fixture(
        tmp_path, monkeypatch
    )
    real_extract_zip = cache._extract_held_zip  # noqa: SLF001

    class SimulatedPowerLoss(BaseException):
        pass

    def extract_then_interrupt(repo_arg, held, spec, published):
        real_extract_zip(repo_arg, held, spec, published)
        raise SimulatedPowerLoss("after all fixture raw publication")

    monkeypatch.setattr(cache, "_extract_held_zip", extract_then_interrupt)
    consume_args = cache._build_parser().parse_args(  # noqa: SLF001
        _consume_argv(repo, incoming, manifest, digest)
    )
    with pytest.raises(SimulatedPowerLoss):
        cache.consume(consume_args)

    intent, completion = cache._consumption_receipt_paths(  # noqa: SLF001
        digest, "a" * 40
    )
    assert (repo / intent).is_file()
    assert not (repo / completion).exists()
    assert (repo / "data/raw/noise/demand/DKITCHEN/ch01.wav").is_file()
    guard_args = cache._build_parser().parse_args(  # noqa: SLF001
        [
            "guard-plain",
            "--expected-commit",
            "a" * 40,
            "--repo-root",
            str(repo),
        ]
    )
    with pytest.raises(cache.ArchiveCacheError, match="plain bootstrap"):
        cache.guard_plain(guard_args)

    monkeypatch.setattr(cache, "_extract_held_zip", real_extract_zip)
    result = cache.consume(consume_args)
    assert result["consumption_intent"] == intent
    assert result["consumption_completion"] == completion
    assert (repo / completion).is_file()
    with pytest.raises(cache.ArchiveCacheError, match="plain bootstrap"):
        cache.guard_plain(guard_args)


def test_consume_validates_every_held_archive_before_publishing_any_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, incoming, manifest, digest, _specs, cached = _consume_fixture(
        tmp_path, monkeypatch
    )
    damaged = cached["demand_dkitchen"]
    payload = bytearray(damaged.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    damaged.write_bytes(payload)

    assert cache.main(_consume_argv(repo, incoming, manifest, digest)) == 2

    assert not (repo / "data/raw/noise/dns_fullband").exists()
    assert not (repo / "data/raw/noise/demand/DKITCHEN").exists()
    origin_root = repo / cache.CACHE_ORIGIN_DIRECTORY
    assert not origin_root.exists() or not any(origin_root.iterdir())


def test_restore_cli_validates_complete_cache_then_no_replace_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    repo, incoming, manifest, digest, specs = _cache_fixture(tmp_path, monkeypatch)
    source_state = {
        path: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        for path in incoming.rglob("*.zip")
    }

    assert cache.main(_restore_argv(repo, incoming, manifest, digest)) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "archives_restored_no_raw_authority"
    assert result["authority"] == cache.AUTHORITY
    origin = repo / result["origin_receipt"]
    origin_payload = json.loads(origin.read_text(encoding="utf-8"))
    assert origin_payload["kind"] == cache.CACHE_ORIGIN_KIND
    assert origin_payload["authority"] == cache.CACHE_ORIGIN_AUTHORITY
    assert origin_payload["manifest_sha256"] == digest
    assert origin_payload["publisher_commit"] == "a" * 40
    assert hashlib.sha256(origin.read_bytes()).hexdigest() == result["origin_receipt_sha256"]
    assert not (repo / "data/manifests/elice_bootstrap_receipt.json").exists()
    for spec in specs:
        assert (repo / spec.canonical_target).read_bytes()
    assert {
        path: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        for path in incoming.rglob("*.zip")
    } == source_state


def test_restore_rehashes_every_target_immediately_before_origin_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, incoming, manifest, digest, specs = _cache_fixture(tmp_path, monkeypatch)
    calls = 0

    def mutate_after_copy(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            target = repo / specs[0].canonical_target
            replacement = target.with_name(f".{target.name}.replacement")
            replacement.write_bytes(b"forged-valid-looking-archive")
            os.replace(replacement, target)
        return None

    monkeypatch.setattr(cache, "_verify_exact_source", mutate_after_copy)
    args = cache._build_parser().parse_args(  # noqa: SLF001
        _restore_argv(repo, incoming, manifest, digest)
    )

    with pytest.raises(cache.ArchiveCacheError, match="receipt precondition"):
        cache.restore(args)

    origin_root = repo / cache.CACHE_ORIGIN_DIRECTORY
    assert not origin_root.exists() or not any(origin_root.iterdir())


def test_restore_parent_swap_is_blocked_before_dirfd_target_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, incoming, manifest, digest, _specs = _cache_fixture(tmp_path, monkeypatch)
    real_copy = cache._copy_no_replace_at  # noqa: SLF001
    outside = tmp_path / "outside-via-symlink"
    moved = tmp_path / "renamed-validated-parent"
    outside.mkdir()
    swapped = False

    def swap_parent_then_copy(source, target, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            target.path.parent.rename(moved)
            target.path.parent.symlink_to(outside, target_is_directory=True)
        return real_copy(source, target, **kwargs)

    monkeypatch.setattr(cache, "_copy_no_replace_at", swap_parent_then_copy)
    args = cache._build_parser().parse_args(  # noqa: SLF001
        _restore_argv(repo, incoming, manifest, digest)
    )

    with pytest.raises(cache.ArchiveCacheError, match="parent가 validation 뒤 교체"):
        cache.restore(args)

    assert list(outside.iterdir()) == []
    assert not list(moved.glob("fixture-*.zip"))


def test_held_target_rejects_intermediate_ancestor_rename_and_symlink_swap(
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    (repo / "data/raw/noise").mkdir(parents=True)
    target = cache._open_safe_target(  # noqa: SLF001
        repo, "data/raw/noise/archive.zip"
    )
    moved = tmp_path / "moved-data"
    try:
        (repo / "data").rename(moved)
        (repo / "data").symlink_to(moved, target_is_directory=True)

        with pytest.raises(cache.ArchiveCacheError, match="parent가 validation 뒤 교체"):
            cache._assert_target_parent_still_named(target)  # noqa: SLF001
    finally:
        target.close()

    assert not (moved / "raw/noise/archive.zip").exists()


@pytest.mark.parametrize("through_dirfd", [False, True])
def test_exact_target_rejects_same_inode_write_at_hash_eof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    through_dirfd: bool,
):
    repo = tmp_path / "repo"
    target = repo / "data/raw/noise/member.wav"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"trusted-bytes")
    expected = hashlib.sha256(target.read_bytes()).hexdigest()
    real_read = cache.os.read
    mutated = False

    def mutate_on_eof(descriptor: int, amount: int) -> bytes:
        nonlocal mutated
        chunk = real_read(descriptor, amount)
        if not chunk and not mutated:
            mutated = True
            with target.open("r+b", buffering=0) as writer:
                writer.write(b"X" * target.stat().st_size)
                os.fsync(writer.fileno())
        return chunk

    monkeypatch.setattr(cache.os, "read", mutate_on_eof)
    if through_dirfd:
        handle = cache._open_safe_target(  # noqa: SLF001
            repo, "data/raw/noise/member.wav", create_parents=False
        )
        try:
            with pytest.raises(cache.ArchiveCacheError, match="readback"):
                cache._verify_exact_target(  # noqa: SLF001
                    handle,
                    size=len(b"trusted-bytes"),
                    sha256=expected,
                    label="fixture",
                )
        finally:
            handle.close()
    else:
        with pytest.raises(cache.ArchiveCacheError, match="readback"):
            cache._verify_exact_path(  # noqa: SLF001
                target,
                size=len(b"trusted-bytes"),
                sha256=expected,
                label="fixture",
            )
    assert target.read_bytes() == b"X" * len(b"trusted-bytes")


def test_restore_and_consume_target_mount_space_fail_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, incoming, manifest, digest, specs, _cached = _consume_fixture(
        tmp_path, monkeypatch
    )
    no_space = type(
        "NoSpace",
        (),
        {
            "f_bavail": 0,
            "f_frsize": 4096,
            "f_bsize": 4096,
            "f_favail": 1_000_000,
        },
    )()
    monkeypatch.setattr(cache.os, "fstatvfs", lambda _descriptor: no_space)

    restore_args = cache._build_parser().parse_args(  # noqa: SLF001
        _restore_argv(repo, incoming, manifest, digest)
    )
    with pytest.raises(cache.ArchiveCacheError, match="target filesystem"):
        cache.restore(restore_args)
    assert all(not (repo / spec.canonical_target).exists() for spec in specs)
    origin_root = repo / cache.CACHE_ORIGIN_DIRECTORY
    assert not origin_root.exists() or not any(origin_root.iterdir())

    consume_args = cache._build_parser().parse_args(  # noqa: SLF001
        _consume_argv(repo, incoming, manifest, digest)
    )
    with pytest.raises(cache.ArchiveCacheError, match="target filesystem"):
        cache.consume(consume_args)
    intent, completion = cache._consumption_receipt_paths(  # noqa: SLF001
        digest, "a" * 40
    )
    assert not (repo / intent).exists()
    assert not (repo / completion).exists()
    assert not (repo / "data/raw/noise/demand/DKITCHEN/ch01.wav").exists()


def test_consume_target_mount_inode_gate_fails_before_intent_or_raw_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, incoming, manifest, digest, _specs, _cached = _consume_fixture(
        tmp_path, monkeypatch
    )
    no_inodes = type(
        "NoInodes",
        (),
        {
            "f_bavail": 1 << 40,
            "f_frsize": 4096,
            "f_bsize": 4096,
            "f_favail": 0,
        },
    )()
    monkeypatch.setattr(cache.os, "fstatvfs", lambda _descriptor: no_inodes)

    args = cache._build_parser().parse_args(  # noqa: SLF001
        _consume_argv(repo, incoming, manifest, digest)
    )
    with pytest.raises(cache.ArchiveCacheError, match="inode 부족"):
        cache.consume(args)
    intent, completion = cache._consumption_receipt_paths(  # noqa: SLF001
        digest, "a" * 40
    )
    assert not (repo / intent).exists()
    assert not (repo / completion).exists()
    assert not (repo / "data/raw/noise/demand/DKITCHEN/ch01.wav").exists()


def test_consume_space_gate_includes_receipt_parent_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, incoming, manifest, digest, _specs, _cached = _consume_fixture(
        tmp_path, monkeypatch
    )
    real_fstatvfs = os.fstatvfs
    seen_receipt_parent = False

    def receipt_parent_out_of_space(descriptor: int):
        nonlocal seen_receipt_parent
        named = os.readlink(f"/proc/self/fd/{descriptor}")
        if ".archive_cache_consumptions" in named or ".archive_cache_origins" in named:
            seen_receipt_parent = True
            return types.SimpleNamespace(
                f_bavail=0,
                f_frsize=4096,
                f_bsize=4096,
                f_favail=0,
            )
        return real_fstatvfs(descriptor)

    monkeypatch.setattr(cache.os, "fstatvfs", receipt_parent_out_of_space)
    args = cache._build_parser().parse_args(  # noqa: SLF001
        _consume_argv(repo, incoming, manifest, digest)
    )

    with pytest.raises(cache.ArchiveCacheError, match="target filesystem"):
        cache.consume(args)
    assert seen_receipt_parent
    intent, completion = cache._consumption_receipt_paths(  # noqa: SLF001
        digest, "a" * 40
    )
    assert not (repo / intent).exists()
    assert not (repo / completion).exists()


def test_restore_cli_rejects_external_sha_unknown_key_symlink_and_wrong_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, incoming, manifest, digest, specs = _cache_fixture(tmp_path, monkeypatch)
    assert cache.main(_restore_argv(repo, incoming, manifest, "0" * 64)) == 2
    assert not (repo / specs[0].canonical_target).exists()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["untrusted"] = True
    raw = cache._canonical_json(payload)  # noqa: SLF001
    manifest.write_bytes(raw)
    assert cache.main(
        _restore_argv(repo, incoming, manifest, hashlib.sha256(raw).hexdigest())
    ) == 2
    assert not (repo / specs[0].canonical_target).exists()

    # Restore the anchored manifest, then replace the first cached object by a symlink.
    del payload["untrusted"]
    raw = cache._canonical_json(payload)  # noqa: SLF001
    manifest.write_bytes(raw)
    first_entry = payload["archives"][0]
    first = incoming / first_entry["cache_path"]
    real = first.with_suffix(".real")
    first.rename(real)
    first.symlink_to(real)
    assert cache.main(
        _restore_argv(repo, incoming, manifest, hashlib.sha256(raw).hexdigest())
    ) == 2
    assert not (repo / specs[0].canonical_target).exists()

    first.unlink()
    real.rename(first)
    wrong_target = repo / specs[0].canonical_target
    wrong_target.parent.mkdir(parents=True)
    wrong_target.write_bytes(b"do-not-overwrite")
    assert cache.main(
        _restore_argv(repo, incoming, manifest, hashlib.sha256(raw).hexdigest())
    ) == 2
    assert wrong_target.read_bytes() == b"do-not-overwrite"


def test_restore_rejects_missing_duplicate_allowlist_and_validates_all_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, incoming, manifest, _digest, specs = _cache_fixture(tmp_path, monkeypatch)
    original = json.loads(manifest.read_text(encoding="utf-8"))

    missing = dict(original)
    missing["archives"] = list(original["archives"][:-1])
    missing["archive_count"] = 1
    raw = cache._canonical_json(missing)  # noqa: SLF001
    manifest.write_bytes(raw)
    assert cache.main(
        _restore_argv(repo, incoming, manifest, hashlib.sha256(raw).hexdigest())
    ) == 2
    assert not (repo / specs[0].canonical_target).exists()

    duplicate = dict(original)
    duplicate["archives"] = [original["archives"][0], original["archives"][0]]
    raw = cache._canonical_json(duplicate)  # noqa: SLF001
    manifest.write_bytes(raw)
    assert cache.main(
        _restore_argv(repo, incoming, manifest, hashlib.sha256(raw).hexdigest())
    ) == 2
    assert not (repo / specs[0].canonical_target).exists()

    raw = cache._canonical_json(original)  # noqa: SLF001
    manifest.write_bytes(raw)
    second = incoming / original["archives"][1]["cache_path"]
    corrupted = bytearray(second.read_bytes())
    corrupted[len(corrupted) // 2] ^= 0xFF
    second.write_bytes(corrupted)
    assert cache.main(
        _restore_argv(repo, incoming, manifest, hashlib.sha256(raw).hexdigest())
    ) == 2
    assert not (repo / specs[0].canonical_target).exists()


def test_rclone_upload_uses_immutable_shared_client_throttle_check_and_cat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    local = tmp_path / "object"
    local.write_bytes(b"object-bytes")
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    checked: list[list[str]] = []
    popen_commands: list[list[str]] = []

    monkeypatch.setattr(cache, "_run_checked", lambda command: checked.append(list(command)))

    class _Process:
        def __init__(self, command, **_kwargs):
            popen_commands.append(list(command))
            self.stdout = io.BytesIO(local.read_bytes())
            self.returncode = 0

        def wait(self):
            return self.returncode

    monkeypatch.setattr(cache.subprocess, "Popen", _Process)
    cache._rclone_upload_and_verify(  # noqa: SLF001
        local,
        "gdrive:DeepANC/cache/object",
        rclone=Path("/usr/bin/rclone"),
        expected_sha256=digest,
    )

    assert "copyto" in checked[0]
    assert "--immutable" in checked[0]
    assert checked[0][checked[0].index("--transfers") + 1] == "1"
    assert checked[0][checked[0].index("--checkers") + 1] == "1"
    assert checked[0][checked[0].index("--tpslimit") + 1] == "2"
    assert "check" in checked[1] and "--download" in checked[1]
    assert checked[1][checked[1].index("--transfers") + 1] == "1"
    assert checked[1][checked[1].index("check") + 1] == str(local.parent)
    assert checked[1][checked[1].index("--include") + 1] == f"/{local.name}"
    assert checked[1][checked[1].index("check") + 2].endswith("/cache")
    assert "cat" in popen_commands[0]
    assert all(word not in command for command in checked for word in ("sync", "delete", "move"))


@pytest.mark.skipif(shutil.which("rclone") is None, reason="rclone CLI가 설치되지 않음")
def test_rclone_real_local_backend_accepts_directory_scoped_exact_file_check(
    tmp_path: Path,
):
    local_dir = tmp_path / "source"
    remote_dir = tmp_path / "remote"
    local_dir.mkdir()
    remote_dir.mkdir()
    local = local_dir / "archive.bin"
    local.write_bytes(b"actual-rclone-check-regression")
    digest = hashlib.sha256(local.read_bytes()).hexdigest()

    cache._rclone_upload_and_verify(  # noqa: SLF001
        local,
        str(remote_dir / local.name),
        rclone=Path(shutil.which("rclone") or ""),
        expected_sha256=digest,
    )

    assert (remote_dir / local.name).read_bytes() == local.read_bytes()


def test_publish_is_sequential_and_manifest_is_uploaded_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    tracked_pget = repo / "scripts/elice/pget.py"
    tracked_pget.parent.mkdir(parents=True)
    tracked_pget.write_text("# publisher fixture\n", encoding="utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()
    seed = tmp_path / "seed.zip"
    _zip(seed, [("root/a.wav", b"abc")])
    spec = _tiny_spec(seed, archive_id="fixture_publish")
    monkeypatch.setattr(cache, "ARCHIVE_SPECS", (spec,))
    monkeypatch.setattr(cache, "EXPECTED_IDS", (spec.archive_id,))
    monkeypatch.setattr(cache, "_repo_root_from_script", lambda: repo)
    monkeypatch.setattr(cache, "_verify_exact_source", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cache, "_provider_head", lambda _spec: None)
    monkeypatch.setattr(cache, "_validate_aggregate", lambda _entries: None)

    active = 0
    max_active = 0

    def fake_download(_pget, _spec, output, **_kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        output.write_bytes(seed.read_bytes())
        active -= 1

    uploaded: list[tuple[str, str]] = []

    def fake_upload(local, remote, **kwargs):
        uploaded.append((Path(local).name, remote))
        assert kwargs["expected_sha256"] == hashlib.sha256(Path(local).read_bytes()).hexdigest()

    monkeypatch.setattr(cache, "_run_pget", fake_download)
    monkeypatch.setattr(cache, "_rclone_upload_and_verify", fake_upload)
    args = cache._build_parser().parse_args(  # noqa: SLF001
        [
            "publish",
            "--staging-root",
            str(staging),
            "--remote-root",
            "gdrive:DeepANC/archive_cache",
            "--expected-commit",
            "a" * 40,
            "--rclone",
            "/bin/true",
        ]
    )

    result = cache.publish(args)

    assert max_active == 1
    assert uploaded[0][0] == seed.name
    assert "/archives/v1/fixture_publish/" in uploaded[0][1]
    assert uploaded[-1][0] == "archive_cache_manifest.json"
    assert "/manifests/v1/sha256_" in uploaded[-1][1]
    assert result["authority"] == cache.AUTHORITY
    assert (Path(result["staging_manifest"]).parent / seed.name).read_bytes() == seed.read_bytes()
    published_manifest = json.loads(Path(result["staging_manifest"]).read_text(encoding="utf-8"))
    assert published_manifest["publisher_pget_sha256"] == hashlib.sha256(
        tracked_pget.read_bytes()
    ).hexdigest()


def test_publish_blocks_insufficient_total_staging_before_provider_or_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    tracked_pget = repo / "scripts/elice/pget.py"
    tracked_pget.parent.mkdir(parents=True)
    tracked_pget.write_text("# publisher fixture\n", encoding="utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()
    seed = tmp_path / "seed.zip"
    _zip(seed, [("root/a.wav", b"abc")])
    spec = _tiny_spec(seed, archive_id="fixture_space")
    monkeypatch.setattr(cache, "ARCHIVE_SPECS", (spec,))
    monkeypatch.setattr(cache, "_repo_root_from_script", lambda: repo)
    monkeypatch.setattr(cache, "_verify_exact_source", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cache.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": spec.expected_size})(),
    )
    provider_calls: list[str] = []
    upload_calls: list[str] = []
    monkeypatch.setattr(
        cache, "_provider_head", lambda selected: provider_calls.append(selected.archive_id)
    )
    monkeypatch.setattr(
        cache,
        "_rclone_upload_and_verify",
        lambda _local, remote, **_kwargs: upload_calls.append(remote),
    )
    args = cache._build_parser().parse_args(  # noqa: SLF001
        [
            "publish",
            "--staging-root",
            str(staging),
            "--remote-root",
            "gdrive:DeepANC/archive_cache",
            "--expected-commit",
            "a" * 40,
            "--rclone",
            "/bin/true",
        ]
    )

    with pytest.raises(cache.ArchiveCacheError, match="전체 manifest-last"):
        cache.publish(args)

    assert provider_calls == []
    assert upload_calls == []
    assert list(staging.iterdir()) == []


def _exact_source_repo(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, str, str, str, types.ModuleType]:
    repo = tmp_path / "repo"
    public_script = repo / "scripts/elice/public_archive_cache.py"
    pget = repo / "scripts/elice/pget.py"
    source_trust = repo / "src/deep_anc/data/source_trust.py"
    for source, target in (
        (Path(cache.__file__), public_script),
        (Path(cache.__file__).with_name("pget.py"), pget),
        (
            Path(cache.__file__).parents[2] / "src/deep_anc/data/source_trust.py",
            source_trust,
        ),
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "archive-cache@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Archive Cache Test"], cwd=repo, check=True
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "archive cache exact tree fixture"],
        cwd=repo,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    script_sha = hashlib.sha256(public_script.read_bytes()).hexdigest()
    committed_pget_sha = hashlib.sha256(pget.read_bytes()).hexdigest()
    module_name = f"archive_cache_exact_{tmp_path.name}"
    copied = types.ModuleType(module_name)
    copied.__file__ = str(public_script)
    sys.modules[module_name] = copied
    exec(compile(public_script.read_bytes(), str(public_script), "exec"), copied.__dict__)
    return (
        repo,
        public_script,
        pget,
        source_trust,
        commit,
        script_sha,
        committed_pget_sha,
        copied,
    )


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_publisher_exact_tree_rejects_hidden_pget_blob_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str
):
    (
        repo,
        _public_script,
        pget,
        _source_trust,
        commit,
        script_sha,
        committed_pget_sha,
        copied,
    ) = _exact_source_repo(tmp_path)
    subprocess.run(["git", "update-index", flag, str(pget.relative_to(repo))], cwd=repo, check=True)
    pget.write_bytes(pget.read_bytes() + b"\n# hidden mutation\n")

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    marker = tmp_path / "path-git-was-executed"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        'printf called > "$FAKE_GIT_MARKER"\n'
        "exit 97\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    monkeypatch.setenv("FAKE_GIT_MARKER", str(marker))
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ.get('PATH', '')}")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "forged-git-dir"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "forged-work-tree"))
    try:
        with pytest.raises(copied.ArchiveCacheError, match="assume-unchanged/skip-worktree"):
            copied._verify_exact_source(
                repo, commit, script_sha, committed_pget_sha
            )
        assert not marker.exists()
    finally:
        sys.modules.pop(copied.__name__, None)


def test_source_trust_swap_is_never_executed_after_held_blob_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (
        repo,
        _public_script,
        _pget,
        source_trust,
        commit,
        script_sha,
        committed_pget_sha,
        copied,
    ) = _exact_source_repo(tmp_path)
    marker = tmp_path / "swapped-source-trust-executed"
    original_read = copied._read_regular_bytes
    swapped = False

    def read_then_swap(path: Path, *, label: str):
        nonlocal swapped
        raw = original_read(path, label=label)
        if path == source_trust and not swapped:
            swapped = True
            source_trust.write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
                "class SourceTrustError(ValueError):\n"
                "    pass\n"
                "def exact_clean_source_evidence(*args, **kwargs):\n"
                "    return {}\n",
                encoding="utf-8",
            )
        return raw

    monkeypatch.setattr(copied, "_read_regular_bytes", read_then_swap)
    try:
        with pytest.raises(copied.ArchiveCacheError, match="robust exact-tree"):
            copied._verify_exact_source(repo, commit, script_sha, committed_pget_sha)
        assert not marker.exists()
    finally:
        sys.modules.pop(copied.__name__, None)


def test_cli_rejects_nonisolated_python_before_shadowable_import(tmp_path: Path):
    script = tmp_path / "scripts/elice/public_archive_cache.py"
    script.parent.mkdir(parents=True)
    shutil.copy2(Path(cache.__file__), script)
    marker = tmp_path / "shadow-hashlib-executed"
    (script.parent / "hashlib.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "python -I -B" in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    "remote",
    ["", "relative/path", "/absolute", "gdrive:/absolute", "gdrive:a/../b", "https://x"],
)
def test_remote_root_rejects_non_rclone_and_traversal_values(remote: str):
    with pytest.raises(cache.ArchiveCacheError):
        cache._validate_remote_root(remote)  # noqa: SLF001
