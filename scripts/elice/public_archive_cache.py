#!/usr/bin/env python3
"""Official public-archive cache publisher and no-replace restorer.

The cache is transport acceleration only.  It never attests extracted raw data,
decoder success, a training manifest, or bootstrap readiness.  The allowlist is
intentionally compiled into this file so a URL, archive name, extraction target,
or corpus cannot be supplied from a manifest or the command line.
"""

from __future__ import annotations

# This file is an authority-bearing CLI entrypoint.  Refuse a non-isolated
# interpreter before importing any module that could be shadowed by the script
# directory or PYTHONPATH.  Library-style imports (focused tests) remain
# possible, but every real CLI invocation must use ``python -I -B``.
import sys as _bootstrap_sys

if __name__ == "__main__" and (
    not _bootstrap_sys.flags.isolated
    or not _bootstrap_sys.flags.ignore_environment
    or not _bootstrap_sys.flags.dont_write_bytecode
):
    raise SystemExit(
        "[BLOCKED] public_archive_cache.py는 격리된 `python -I -B`로만 실행할 수 있습니다"
    )

import argparse
import bz2
import hashlib
import json
import os
import re
import resource
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import types
import unicodedata
import urllib.request
import uuid
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
MANIFEST_KIND = "deep_anc_public_archive_cache"
AUTHORITY = "transport_acceleration_only_not_raw_or_training_authority"
CACHE_ORIGIN_KIND = "deep_anc_archive_cache_origin_receipt"
CACHE_ORIGIN_AUTHORITY = "cache_origin_only_not_official_raw_or_training_authority"
CACHE_ORIGIN_DIRECTORY = "data/raw/noise/.archive_cache_origins"
CACHE_CONSUMPTION_DIRECTORY = "data/raw/noise/.archive_cache_consumptions"
CACHE_CONSUMPTION_INTENT_KIND = "deep_anc_archive_cache_consumption_intent"
CACHE_CONSUMPTION_COMPLETION_KIND = "deep_anc_archive_cache_consumption_completion"
CACHE_CONSUMPTION_AUTHORITY = (
    "cache_transport_state_only_requires_exact_raw_and_decoder_authority"
)
CACHE_TRAINING_WAV_ROOTS = (
    "data/raw/noise/dns_fullband",
    "data/raw/noise/speech",
    "data/raw/noise/demand",
    "data/raw/noise/machine/fan",
)
GIT_EXECUTABLE = "/usr/bin/git"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
REMOTE_ROOT = re.compile(r"[A-Za-z0-9_.-]+:[^\t\r\n]+")
READ_CHUNK = 8 * 1024 * 1024
STAGING_HEADROOM_BYTES = 512 * 1024 * 1024
STAGING_HEADROOM_INODES = 64
CACHE_RECEIPT_RESERVE_BYTES = 64 * 1024 * 1024
RCLONE_THROTTLE = (
    "--transfers",
    "1",
    "--checkers",
    "1",
    "--tpslimit",
    "2",
    "--drive-pacer-min-sleep",
    "500ms",
)


class ArchiveCacheError(RuntimeError):
    """Fail-closed cache contract error."""


@dataclass(frozen=True)
class ArchiveSpec:
    archive_id: str
    corpus: str
    url: str
    filename: str
    canonical_target: str
    archive_format: str
    expected_size: int
    provider_checksum_kind: str
    provider_checksum: str | None
    provider_etag: str | None
    member_prefix: str
    expected_wav_count: int
    expected_wav_bytes: int | None
    demand_environment: str | None = None


@dataclass
class _SafeTargetHandle:
    path: Path
    parent_fd: int
    name: str
    parent_identity: tuple[int, int]
    directory_chain: list[tuple[Path, int, tuple[int, int]]]

    def close(self) -> None:
        for _path, descriptor, _identity in reversed(self.directory_chain):
            if descriptor >= 0:
                os.close(descriptor)
        self.directory_chain.clear()
        self.parent_fd = -1


@dataclass
class _HeldArchive:
    """One cache inode held from manifest validation through extraction.

    The cache pathname is deliberately informational after this object is
    created.  Validation, archive readers, extraction, and the final SHA
    readback all operate on this descriptor (or a duplicate of it), so a
    same-UID rename/replacement cannot substitute different bytes between the
    external-manifest gate and extraction.
    """

    path: Path
    descriptor: int
    identity: tuple[int, int]
    size: int
    mtime_ns: int
    ctime_ns: int

    def reader(self) -> BinaryIO:
        duplicate = os.dup(self.descriptor)
        handle = os.fdopen(duplicate, "rb")
        handle.seek(0)
        return handle

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


@dataclass
class _PublishedMember:
    archive_id: str
    relative: str
    size: int
    sha256: str
    descriptor: int
    identity: tuple[int, int]
    mtime_ns: int
    ctime_ns: int

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


DNS_BASE = "https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"
ZENODO = "https://zenodo.org/records"
ARCHIVE_SPECS: tuple[ArchiveSpec, ...] = (
    ArchiveSpec(
        "dns_noise_000",
        "dns3",
        f"{DNS_BASE}/noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2",
        "datasets_fullband.noise_fullband.audioset_000.tar.bz2",
        "data/raw/noise/shard000.tar.bz2",
        "tar.bz2",
        5_364_611_964,
        "none",
        None,
        "0x8D9B5BADC55F6BB",
        "datasets_fullband/noise_fullband/",
        8_000,
        None,
    ),
    ArchiveSpec(
        "dns_noise_001",
        "dns3",
        f"{DNS_BASE}/noise_fullband/datasets_fullband.noise_fullband.audioset_001.tar.bz2",
        "datasets_fullband.noise_fullband.audioset_001.tar.bz2",
        "data/raw/noise/shard001.tar.bz2",
        "tar.bz2",
        5_357_916_291,
        "none",
        None,
        "0x8D9B5B0DB55DD9B",
        "datasets_fullband/noise_fullband/",
        8_000,
        None,
    ),
    ArchiveSpec(
        "dns_speech_000",
        "dns3",
        f"{DNS_BASE}/clean_fullband/datasets_fullband.clean_fullband.read_speech_000_0.00_3.75.tar.bz2",
        "datasets_fullband.clean_fullband.read_speech_000_0.00_3.75.tar.bz2",
        "data/raw/noise/speech000.tar.bz2",
        "tar.bz2",
        4_664_045_287,
        "none",
        None,
        "0x8D9B5BA782095C9",
        "datasets_fullband/clean_fullband/read_speech/",
        8_065,
        8_000_834_860,
    ),
    ArchiveSpec(
        "demand_dkitchen",
        "demand6",
        f"{ZENODO}/1227121/files/DKITCHEN_48k.zip?download=1",
        "DKITCHEN_48k.zip",
        "data/raw/noise/demand/DKITCHEN_48k.zip",
        "zip",
        336_992_458,
        "md5",
        "b4d38241fbd50d8a17f8742ca6870c10",
        None,
        "DKITCHEN/",
        16,
        460_806_848,
        "DKITCHEN",
    ),
    ArchiveSpec(
        "demand_dwashing",
        "demand6",
        f"{ZENODO}/1227121/files/DWASHING_48k.zip?download=1",
        "DWASHING_48k.zip",
        "data/raw/noise/demand/DWASHING_48k.zip",
        "zip",
        306_101_499,
        "md5",
        "ecf765f12b8d3ada7ef0ec664b8f8d73",
        None,
        "DWASHING/",
        16,
        460_806_848,
        "DWASHING",
    ),
    ArchiveSpec(
        "demand_ooffice",
        "demand6",
        f"{ZENODO}/1227121/files/OOFFICE_48k.zip?download=1",
        "OOFFICE_48k.zip",
        "data/raw/noise/demand/OOFFICE_48k.zip",
        "zip",
        277_643_831,
        "md5",
        "6f87edf8a6b03f17b3f693af1754aab4",
        None,
        "OOFFICE/",
        16,
        460_806_848,
        "OOFFICE",
    ),
    ArchiveSpec(
        "demand_ohallway",
        "demand6",
        f"{ZENODO}/1227121/files/OHALLWAY_48k.zip?download=1",
        "OHALLWAY_48k.zip",
        "data/raw/noise/demand/OHALLWAY_48k.zip",
        "zip",
        252_905_617,
        "md5",
        "cb9227a75d2c1342de0b6548da4bbb1b",
        None,
        "OHALLWAY/",
        16,
        460_806_848,
        "OHALLWAY",
    ),
    ArchiveSpec(
        "demand_tmetro",
        "demand6",
        f"{ZENODO}/1227121/files/TMETRO_48k.zip?download=1",
        "TMETRO_48k.zip",
        "data/raw/noise/demand/TMETRO_48k.zip",
        "zip",
        367_513_573,
        "md5",
        "00d895020233f94348aafce7140d671f",
        None,
        "TMETRO/",
        16,
        460_806_848,
        "TMETRO",
    ),
    ArchiveSpec(
        "demand_tcar",
        "demand6",
        f"{ZENODO}/1227121/files/TCAR_48k.zip?download=1",
        "TCAR_48k.zip",
        "data/raw/noise/demand/TCAR_48k.zip",
        "zip",
        373_520_251,
        "md5",
        "8550f03e8356d8054ae845e8b4b6c773",
        None,
        "TCAR/",
        16,
        460_806_848,
        "TCAR",
    ),
    ArchiveSpec(
        "mimii_fan",
        "mimii1",
        f"{ZENODO}/6529888/files/fan.zip?download=1",
        "fan.zip",
        "data/raw/noise/mimii_fan.zip",
        "zip",
        928_511_244,
        "md5",
        "a1a9b488934a82426bacc933d87aacde",
        None,
        "fan/",
        3_600,
        1_152_158_400,
    ),
)

EXPECTED_IDS = tuple(spec.archive_id for spec in ARCHIVE_SPECS)
SPEC_BY_ID = {spec.archive_id: spec for spec in ARCHIVE_SPECS}
MANIFEST_KEYS = {
    "archive_count",
    "archives",
    "authority",
    "excluded_corpora",
    "kind",
    "publisher_commit",
    "publisher_entry_script_sha256",
    "publisher_pget_sha256",
    "schema_version",
}
ENTRY_KEYS = {
    "archive_format",
    "archive_id",
    "archive_sha256",
    "archive_size",
    "cache_path",
    "canonical_target",
    "corpus",
    "filename",
    "member_inventory_sha256",
    "member_content_inventory_sha256",
    "member_prefix",
    "provider_checksum",
    "provider_checksum_kind",
    "provider_etag",
    "regular_file_bytes",
    "regular_file_count",
    "source_url",
    "output_content_inventory_sha256",
    "wav_bytes",
    "wav_count",
}


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive_fsynced(path: Path, data: bytes) -> str:
    """Write all bytes without replace, fsync, then read back the content SHA."""

    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    offset = 0
    try:
        while offset < len(data):
            written = os.write(descriptor, memoryview(data)[offset:])
            if written <= 0:
                raise ArchiveCacheError(
                    f"exclusive file short write; forensic staging을 보존합니다: {path}"
                )
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    local_sha256 = _hash_file(path)["sha256"]
    if path.stat().st_size != len(data) or local_sha256 != _sha256_bytes(data):
        raise ArchiveCacheError(
            f"exclusive file local fsync/readback 불일치; remote publish 금지: {path}"
        )
    return local_sha256


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path, algorithms: Iterable[str] = ("sha256",)) -> dict[str, str]:
    hashers = {name: hashlib.new(name) for name in algorithms}
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ArchiveCacheError(f"regular single-link archive가 아닙니다: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while True:
                chunk = handle.read(READ_CHUNK)
                if not chunk:
                    break
                for hasher in hashers.values():
                    hasher.update(chunk)
    finally:
        os.close(descriptor)
    return {name: hasher.hexdigest() for name, hasher in hashers.items()}


def _require_absolute_directory(path: Path, *, label: str, writable: bool) -> Path:
    if not path.is_absolute():
        raise ArchiveCacheError(f"{label}는 절대경로여야 합니다")
    try:
        info = path.lstat()
    except OSError as exc:
        raise ArchiveCacheError(f"{label}를 읽을 수 없습니다: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ArchiveCacheError(f"{label}는 symlink가 아닌 기존 디렉터리여야 합니다")
    resolved = path.resolve(strict=True)
    access = os.R_OK | os.X_OK | (os.W_OK if writable else 0)
    if not os.access(resolved, access):
        raise ArchiveCacheError(f"{label} 권한이 부족합니다: {resolved}")
    return resolved


def _ensure_outside(path: Path, repo: Path, *, label: str) -> None:
    try:
        path.relative_to(repo)
    except ValueError:
        pass
    else:
        raise ArchiveCacheError(f"{label}는 저장소 밖이어야 합니다: {path}")
    try:
        repo.relative_to(path)
    except ValueError:
        return
    raise ArchiveCacheError(f"{label}는 저장소의 상위 디렉터리일 수 없습니다: {path}")


def _regular_no_symlink(path: Path, *, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ArchiveCacheError(f"{label}를 읽을 수 없습니다: {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ArchiveCacheError(f"{label}는 symlink가 아닌 regular file이어야 합니다: {path}")
    return info


def _normalized_member(name: str) -> str:
    if not name or "\\" in name or "\x00" in name:
        raise ArchiveCacheError(f"안전하지 않은 archive member path: {name!r}")
    normalized_unicode = unicodedata.normalize("NFC", name)
    if normalized_unicode != name:
        raise ArchiveCacheError(f"Unicode-normalized archive member가 아닙니다: {name!r}")
    raw_parts = (name[:-1] if name.endswith("/") else name).split("/")
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in raw_parts):
        raise ArchiveCacheError(f"traversal/absolute archive member를 거부합니다: {name!r}")
    normalized = pure.as_posix()
    if normalized.startswith("/") or normalized == ".":
        raise ArchiveCacheError(f"안전하지 않은 archive member path: {name!r}")
    return normalized.rstrip("/")


def _inventory_digest(rows: Sequence[tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for name, size in sorted(rows):
        digest.update(_canonical_json({"path": name, "size": size}))
    return digest.hexdigest()


def _content_inventory_digest(
    rows: Sequence[tuple[str, int, str]],
    *,
    spec: ArchiveSpec | None = None,
) -> str:
    digest = hashlib.sha256()
    for name, size, sha256 in sorted(rows):
        # Production manifests can only contain the compiled ARCHIVE_SPECS and
        # therefore always take the fixed untouched-raw mapping below.  Local
        # validator fixtures use synthetic archive ids; retaining their member
        # path keeps the format validator independently testable without
        # inventing a production extraction destination for those ids.
        has_output_mapping = spec is not None and (
            spec.archive_id in ("dns_noise_000", "dns_noise_001", "dns_speech_000")
            or spec.archive_id.startswith("demand_")
            or spec.archive_id == "mimii_fan"
        )
        if not has_output_mapping:
            path = name
        else:
            assert spec is not None
            path = _extracted_member_target(spec, name)
        digest.update(
            _canonical_json({"path": path, "sha256": sha256, "size": size})
        )
    return digest.hexdigest()


def _validate_member_rows(
    spec: ArchiveSpec,
    rows: Sequence[tuple[str, int]],
) -> dict[str, object]:
    if not rows:
        raise ArchiveCacheError(f"archive에 regular file이 없습니다: {spec.archive_id}")
    names = [name for name, _size in rows]
    folded = [name.casefold() for name in names]
    if len(set(names)) != len(names) or len(set(folded)) != len(folded):
        raise ArchiveCacheError(f"duplicate/case-colliding member: {spec.archive_id}")
    if any(not name.startswith(spec.member_prefix) for name in names):
        raise ArchiveCacheError(f"allowlist prefix 밖 member: {spec.archive_id}")
    wav_rows = [(name, size) for name, size in rows if name.casefold().endswith(".wav")]
    if len(wav_rows) != len(rows):
        raise ArchiveCacheError(f"WAV가 아닌 regular member를 거부합니다: {spec.archive_id}")
    if len(wav_rows) != spec.expected_wav_count:
        raise ArchiveCacheError(
            f"WAV count 불일치: {spec.archive_id}: {len(wav_rows)} != {spec.expected_wav_count}"
        )
    wav_bytes = sum(size for _name, size in wav_rows)
    if spec.expected_wav_bytes is not None and wav_bytes != spec.expected_wav_bytes:
        raise ArchiveCacheError(
            f"WAV bytes 불일치: {spec.archive_id}: {wav_bytes} != {spec.expected_wav_bytes}"
        )
    if spec.demand_environment is not None:
        expected = {
            f"{spec.demand_environment}/ch{index:02d}.wav"
            for index in range(1, 17)
        }
        if set(names) != expected:
            raise ArchiveCacheError(f"DEMAND ch01..ch16 구조 불일치: {spec.archive_id}")
    return {
        "member_inventory_sha256": _inventory_digest(rows),
        "regular_file_count": len(rows),
        "regular_file_bytes": sum(size for _name, size in rows),
        "wav_count": len(wav_rows),
        "wav_bytes": wav_bytes,
    }


def _validate_tar_bz2(path: Path, spec: ArchiveSpec) -> dict[str, object]:
    rows: list[tuple[str, int]] = []
    content_rows: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    # bzip2.open forces the compressed stream CRC check through EOF. tarfile may
    # otherwise stop after the final tar member before noticing trailing damage.
    try:
        with bz2.open(path, "rb") as compressed:
            while compressed.read(READ_CHUNK):
                pass
    except (OSError, EOFError) as exc:
        raise ArchiveCacheError(f"bzip2 integrity 실패: {spec.archive_id}: {exc}") from exc
    try:
        with tarfile.open(path, mode="r:bz2") as archive:
            for member in archive:
                normalized = _normalized_member(member.name)
                folded = normalized.casefold()
                if folded in seen:
                    raise ArchiveCacheError(f"duplicate/case-colliding tar member: {member.name}")
                seen.add(folded)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ArchiveCacheError(
                        f"tar link/device/fifo 등 non-regular member를 거부합니다: {member.name}"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ArchiveCacheError(f"tar member stream을 열 수 없습니다: {member.name}")
                member_digest = hashlib.sha256()
                member_bytes = 0
                with extracted:
                    while True:
                        chunk = extracted.read(READ_CHUNK)
                        if not chunk:
                            break
                        member_digest.update(chunk)
                        member_bytes += len(chunk)
                if member_bytes != int(member.size):
                    raise ArchiveCacheError(f"tar member size readback 불일치: {member.name}")
                rows.append((normalized, int(member.size)))
                content_rows.append(
                    (normalized, int(member.size), member_digest.hexdigest())
                )
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ArchiveCacheError(f"tar integrity 실패: {spec.archive_id}: {exc}") from exc
    return {
        **_validate_member_rows(spec, rows),
        "member_content_inventory_sha256": _content_inventory_digest(content_rows),
        "output_content_inventory_sha256": _content_inventory_digest(
            content_rows, spec=spec
        ),
    }


def _validate_zip(path: Path, spec: ArchiveSpec) -> dict[str, object]:
    rows: list[tuple[str, int]] = []
    content_rows: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                normalized = _normalized_member(member.filename)
                folded = normalized.casefold()
                if folded in seen:
                    raise ArchiveCacheError(
                        f"duplicate/case-colliding ZIP member: {member.filename}"
                    )
                seen.add(folded)
                mode = (member.external_attr >> 16) & 0xFFFF
                kind = stat.S_IFMT(mode)
                is_directory = member.is_dir()
                if is_directory:
                    if kind not in (0, stat.S_IFDIR):
                        raise ArchiveCacheError(f"ZIP directory mode 불일치: {member.filename}")
                    continue
                if member.flag_bits & 0x1:
                    raise ArchiveCacheError(f"encrypted ZIP member를 거부합니다: {member.filename}")
                if kind not in (0, stat.S_IFREG):
                    raise ArchiveCacheError(
                        f"ZIP symlink/device 등 non-regular member를 거부합니다: {member.filename}"
                    )
                rows.append((normalized, int(member.file_size)))
            bad = archive.testzip()
            if bad is not None:
                raise ArchiveCacheError(f"ZIP CRC 실패 member: {bad}")
            for member in archive.infolist():
                if member.is_dir():
                    continue
                normalized = _normalized_member(member.filename)
                member_digest = hashlib.sha256()
                member_bytes = 0
                with archive.open(member, "r") as extracted:
                    while True:
                        chunk = extracted.read(READ_CHUNK)
                        if not chunk:
                            break
                        member_digest.update(chunk)
                        member_bytes += len(chunk)
                if member_bytes != int(member.file_size):
                    raise ArchiveCacheError(
                        f"ZIP member size readback 불일치: {member.filename}"
                    )
                content_rows.append(
                    (normalized, int(member.file_size), member_digest.hexdigest())
                )
    except (zipfile.BadZipFile, zlib.error, OSError, EOFError) as exc:
        raise ArchiveCacheError(f"ZIP integrity 실패: {spec.archive_id}: {exc}") from exc
    return {
        **_validate_member_rows(spec, rows),
        "member_content_inventory_sha256": _content_inventory_digest(content_rows),
        "output_content_inventory_sha256": _content_inventory_digest(
            content_rows, spec=spec
        ),
    }


def validate_archive(path: Path, spec: ArchiveSpec) -> dict[str, object]:
    info = _regular_no_symlink(path, label="archive")
    if info.st_size != spec.expected_size:
        raise ArchiveCacheError(
            f"archive size 불일치: {spec.archive_id}: {info.st_size} != {spec.expected_size}"
        )
    algorithms = ["sha256"]
    if spec.provider_checksum_kind == "md5":
        algorithms.append("md5")
    digests = _hash_file(path, algorithms)
    if spec.provider_checksum_kind == "md5":
        if digests["md5"] != spec.provider_checksum:
            raise ArchiveCacheError(
                f"provider MD5 불일치: {spec.archive_id}: {digests['md5']}"
            )
    elif spec.provider_checksum_kind != "none":
        raise ArchiveCacheError(f"지원하지 않는 provider checksum kind: {spec.archive_id}")
    structure = (
        _validate_tar_bz2(path, spec)
        if spec.archive_format == "tar.bz2"
        else _validate_zip(path, spec)
    )
    return {
        "archive_sha256": digests["sha256"],
        "archive_size": info.st_size,
        "provider_checksum_kind": spec.provider_checksum_kind,
        "provider_checksum": spec.provider_checksum,
        "provider_etag": spec.provider_etag,
        **structure,
    }


def _open_held_cache_member(cache_root: Path, relative: str) -> _HeldArchive:
    """Open a fixed cache member by dirfd traversal and keep its inode held."""

    if not relative or "\\" in relative:
        raise ArchiveCacheError("cache_path가 POSIX 상대경로가 아닙니다")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ArchiveCacheError("cache_path traversal을 거부합니다")
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(cache_root, directory_flags)
    except OSError as exc:
        raise ArchiveCacheError(f"cache root nofollow open 실패: {cache_root}: {exc}") from exc
    try:
        for part in pure.parts[:-1]:
            try:
                child_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            except OSError as exc:
                raise ArchiveCacheError(
                    f"cache path intermediate symlink/non-directory 거부: {relative}: {exc}"
                ) from exc
            os.close(directory_fd)
            directory_fd = child_fd
        try:
            descriptor = os.open(
                pure.parts[-1],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise ArchiveCacheError(
                f"cached archive nofollow openat 실패: {relative}: {exc}"
            ) from exc
    finally:
        os.close(directory_fd)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(descriptor)
        raise ArchiveCacheError(
            f"cached archive는 regular single-link file이어야 합니다: {relative}"
        )
    return _HeldArchive(
        path=cache_root.joinpath(*pure.parts),
        descriptor=descriptor,
        identity=(int(info.st_dev), int(info.st_ino)),
        size=int(info.st_size),
        mtime_ns=int(info.st_mtime_ns),
        ctime_ns=int(info.st_ctime_ns),
    )


def _hash_held_archive(
    held: _HeldArchive,
    algorithms: Iterable[str] = ("sha256",),
    *,
    require_initial_metadata: bool,
) -> dict[str, str]:
    """Hash a held inode without reopening its pathname."""

    before = os.fstat(held.descriptor)
    identity = (int(before.st_dev), int(before.st_ino))
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or identity != held.identity
        or int(before.st_size) != held.size
        or (
            require_initial_metadata
            and (
                int(before.st_mtime_ns) != held.mtime_ns
                or int(before.st_ctime_ns) != held.ctime_ns
            )
        )
    ):
        raise ArchiveCacheError(f"held archive inode/metadata가 바뀌었습니다: {held.path}")
    hashers = {name: hashlib.new(name) for name in algorithms}
    offset = 0
    while offset < held.size:
        chunk = os.pread(held.descriptor, min(READ_CHUNK, held.size - offset), offset)
        if not chunk:
            raise ArchiveCacheError(f"held archive short read: {held.path}: offset={offset}")
        for hasher in hashers.values():
            hasher.update(chunk)
        offset += len(chunk)
    after = os.fstat(held.descriptor)
    if (
        (int(after.st_dev), int(after.st_ino)) != held.identity
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or int(after.st_size) != held.size
        or (
            require_initial_metadata
            and (
                int(after.st_mtime_ns) != held.mtime_ns
                or int(after.st_ctime_ns) != held.ctime_ns
            )
        )
    ):
        raise ArchiveCacheError(f"held archive readback 중 inode/metadata가 바뀌었습니다: {held.path}")
    return {name: hasher.hexdigest() for name, hasher in hashers.items()}


def _validate_held_tar_bz2(
    held: _HeldArchive, spec: ArchiveSpec
) -> tuple[dict[str, object], tuple[tuple[str, int, str], ...]]:
    rows: list[tuple[str, int]] = []
    content_rows: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    try:
        with held.reader() as raw, bz2.BZ2File(raw, "rb") as compressed:
            while compressed.read(READ_CHUNK):
                pass
    except (OSError, EOFError) as exc:
        raise ArchiveCacheError(f"bzip2 integrity 실패: {spec.archive_id}: {exc}") from exc
    try:
        with held.reader() as raw, tarfile.open(fileobj=raw, mode="r:bz2") as archive:
            for member in archive:
                normalized = _normalized_member(member.name)
                folded = normalized.casefold()
                if folded in seen:
                    raise ArchiveCacheError(
                        f"duplicate/case-colliding tar member: {member.name}"
                    )
                seen.add(folded)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ArchiveCacheError(
                        "tar link/device/fifo 등 non-regular member를 거부합니다: "
                        f"{member.name}"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ArchiveCacheError(f"tar member stream을 열 수 없습니다: {member.name}")
                digest = hashlib.sha256()
                consumed = 0
                with extracted:
                    while True:
                        chunk = extracted.read(READ_CHUNK)
                        if not chunk:
                            break
                        digest.update(chunk)
                        consumed += len(chunk)
                if consumed != int(member.size):
                    raise ArchiveCacheError(f"tar member size readback 불일치: {member.name}")
                rows.append((normalized, int(member.size)))
                content_rows.append(
                    (normalized, int(member.size), digest.hexdigest())
                )
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ArchiveCacheError(f"tar integrity 실패: {spec.archive_id}: {exc}") from exc
    immutable_rows = tuple(content_rows)
    return (
        {
            **_validate_member_rows(spec, rows),
            "member_content_inventory_sha256": _content_inventory_digest(
                immutable_rows
            ),
            "output_content_inventory_sha256": _content_inventory_digest(
                immutable_rows, spec=spec
            ),
        },
        immutable_rows,
    )


def _validate_held_zip(
    held: _HeldArchive, spec: ArchiveSpec
) -> tuple[dict[str, object], tuple[tuple[str, int, str], ...]]:
    rows: list[tuple[str, int]] = []
    content_rows: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    try:
        with held.reader() as raw, zipfile.ZipFile(raw) as archive:
            for member in archive.infolist():
                normalized = _normalized_member(member.filename)
                folded = normalized.casefold()
                if folded in seen:
                    raise ArchiveCacheError(
                        f"duplicate/case-colliding ZIP member: {member.filename}"
                    )
                seen.add(folded)
                mode = (member.external_attr >> 16) & 0xFFFF
                kind = stat.S_IFMT(mode)
                if member.is_dir():
                    if kind not in (0, stat.S_IFDIR):
                        raise ArchiveCacheError(
                            f"ZIP directory mode 불일치: {member.filename}"
                        )
                    continue
                if member.flag_bits & 0x1:
                    raise ArchiveCacheError(
                        f"encrypted ZIP member를 거부합니다: {member.filename}"
                    )
                if kind not in (0, stat.S_IFREG):
                    raise ArchiveCacheError(
                        "ZIP symlink/device 등 non-regular member를 거부합니다: "
                        f"{member.filename}"
                    )
                rows.append((normalized, int(member.file_size)))
            bad = archive.testzip()
            if bad is not None:
                raise ArchiveCacheError(f"ZIP CRC 실패 member: {bad}")
            for member in archive.infolist():
                if member.is_dir():
                    continue
                normalized = _normalized_member(member.filename)
                digest = hashlib.sha256()
                consumed = 0
                with archive.open(member, "r") as extracted:
                    while True:
                        chunk = extracted.read(READ_CHUNK)
                        if not chunk:
                            break
                        digest.update(chunk)
                        consumed += len(chunk)
                if consumed != int(member.file_size):
                    raise ArchiveCacheError(
                        f"ZIP member size readback 불일치: {member.filename}"
                    )
                content_rows.append(
                    (normalized, int(member.file_size), digest.hexdigest())
                )
    except (zipfile.BadZipFile, zlib.error, OSError, EOFError) as exc:
        raise ArchiveCacheError(f"ZIP integrity 실패: {spec.archive_id}: {exc}") from exc
    immutable_rows = tuple(content_rows)
    return (
        {
            **_validate_member_rows(spec, rows),
            "member_content_inventory_sha256": _content_inventory_digest(
                immutable_rows
            ),
            "output_content_inventory_sha256": _content_inventory_digest(
                immutable_rows, spec=spec
            ),
        },
        immutable_rows,
    )


def _validate_held_archive(
    held: _HeldArchive, spec: ArchiveSpec
) -> tuple[dict[str, object], tuple[tuple[str, int, str], ...]]:
    if held.size != spec.expected_size:
        raise ArchiveCacheError(
            f"archive size 불일치: {spec.archive_id}: {held.size} != {spec.expected_size}"
        )
    algorithms = ["sha256"]
    if spec.provider_checksum_kind == "md5":
        algorithms.append("md5")
    digests = _hash_held_archive(
        held, algorithms, require_initial_metadata=True
    )
    if spec.provider_checksum_kind == "md5":
        if digests["md5"] != spec.provider_checksum:
            raise ArchiveCacheError(
                f"provider MD5 불일치: {spec.archive_id}: {digests['md5']}"
            )
    elif spec.provider_checksum_kind != "none":
        raise ArchiveCacheError(f"지원하지 않는 provider checksum kind: {spec.archive_id}")
    if spec.archive_format == "tar.bz2":
        structure, rows = _validate_held_tar_bz2(held, spec)
    elif spec.archive_format == "zip":
        structure, rows = _validate_held_zip(held, spec)
    else:
        raise ArchiveCacheError(f"지원하지 않는 archive format: {spec.archive_id}")
    return (
        {
            "archive_sha256": digests["sha256"],
            "archive_size": held.size,
            "provider_checksum_kind": spec.provider_checksum_kind,
            "provider_checksum": spec.provider_checksum,
            "provider_etag": spec.provider_etag,
            **structure,
        },
        rows,
    )


def _validate_aggregate(entries: Sequence[Mapping[str, object]]) -> None:
    by_id = {str(entry["archive_id"]): entry for entry in entries}
    if tuple(by_id) != EXPECTED_IDS:
        raise ArchiveCacheError("aggregate allowlist/order가 고정 계약과 다릅니다")
    noise = [by_id["dns_noise_000"], by_id["dns_noise_001"]]
    if sum(int(entry["wav_count"]) for entry in noise) != 16_000:
        raise ArchiveCacheError("DNS noise aggregate WAV count가 16,000이 아닙니다")
    if sum(int(entry["wav_bytes"]) for entry in noise) != 15_360_708_826:
        raise ArchiveCacheError("DNS noise aggregate WAV bytes가 고정 계약과 다릅니다")
    demand = [entry for key, entry in by_id.items() if key.startswith("demand_")]
    if sum(int(entry["wav_count"]) for entry in demand) != 96:
        raise ArchiveCacheError("DEMAND aggregate WAV count가 96이 아닙니다")
    if sum(int(entry["wav_bytes"]) for entry in demand) != 2_764_841_088:
        raise ArchiveCacheError("DEMAND aggregate WAV bytes가 고정 계약과 다릅니다")


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve(strict=True).parents[2]


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    """Read one held regular single-link inode without following a symlink."""

    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ArchiveCacheError(f"{label} nofollow open 실패: {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ArchiveCacheError(f"{label}는 regular single-link file이어야 합니다: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, READ_CHUNK)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        try:
            named = path.lstat()
        except OSError as exc:
            raise ArchiveCacheError(f"{label} pathname readback 실패: {path}: {exc}") from exc
        identity = (int(before.st_dev), int(before.st_ino))
        if (
            (int(after.st_dev), int(after.st_ino)) != identity
            or (int(named.st_dev), int(named.st_ino)) != identity
            or not stat.S_ISREG(after.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or after.st_nlink != 1
            or named.st_nlink != 1
            or int(after.st_size) != int(before.st_size)
            or int(named.st_size) != int(before.st_size)
            or int(after.st_mtime_ns) != int(before.st_mtime_ns)
            or int(after.st_ctime_ns) != int(before.st_ctime_ns)
            or int(named.st_mtime_ns) != int(before.st_mtime_ns)
            or int(named.st_ctime_ns) != int(before.st_ctime_ns)
        ):
            raise ArchiveCacheError(f"{label}가 read 중/pathname에서 바뀌었습니다: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _exec_trusted_source(raw: bytes, *, filename: Path, module_name: str) -> dict[str, object]:
    """Execute the already-verified bytes, never a path reopened after validation."""

    if module_name in sys.modules:
        raise ArchiveCacheError(f"trusted source 임시 module name 충돌: {module_name}")
    module = types.ModuleType(module_name)
    module.__file__ = str(filename)
    module.__package__ = None
    sys.modules[module_name] = module
    try:
        # dataclasses resolves postponed annotations through sys.modules while
        # decorating classes, hence the temporary registration above.  Remove
        # it immediately after exec; function globals remain the held dict.
        exec(compile(raw, str(filename), "exec"), module.__dict__)
    finally:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
    return module.__dict__


def _verify_exact_source(
    repo: Path,
    expected_commit: str,
    script_sha256: str,
    pget_sha256: str,
) -> bytes:
    if not HEX40.fullmatch(expected_commit):
        raise ArchiveCacheError("expected commit은 lowercase 전체 40자리 SHA여야 합니다")
    expected_script = (repo / "scripts/elice/public_archive_cache.py").resolve(strict=True)
    current_script_path = Path(__file__).resolve(strict=True)
    if current_script_path != expected_script:
        raise ArchiveCacheError("archive-cache entry script가 requested repository 소속이 아닙니다")
    git_path = Path(GIT_EXECUTABLE)
    try:
        git_info = git_path.lstat()
        git_dir_info = (repo / ".git").lstat()
    except OSError as exc:
        raise ArchiveCacheError(f"trusted Git executable/repository metadata 검증 실패: {exc}") from exc
    if (
        not stat.S_ISREG(git_info.st_mode)
        or git_path.is_symlink()
        or git_info.st_nlink != 1
        or not os.access(git_path, os.X_OK)
    ):
        raise ArchiveCacheError(f"trusted Git은 executable regular non-symlink여야 합니다: {git_path}")
    if not stat.S_ISDIR(git_dir_info.st_mode) or (repo / ".git").is_symlink():
        raise ArchiveCacheError("exact source는 root/.git 실제 directory checkout이어야 합니다")
    git_environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    git_environment["GIT_NO_REPLACE_OBJECTS"] = "1"

    def git_bytes(*arguments: str) -> bytes:
        try:
            return subprocess.run(
                [
                    GIT_EXECUTABLE,
                    f"--git-dir={repo / '.git'}",
                    f"--work-tree={repo}",
                    "-c",
                    f"core.worktree={repo}",
                    *arguments,
                ],
                cwd=repo,
                env=git_environment,
                capture_output=True,
                check=True,
                timeout=60,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise ArchiveCacheError(
                f"trusted Git exact-source 명령 실패: {' '.join(arguments)}: {exc}"
            ) from exc

    head = git_bytes("rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip().lower()
    if head != expected_commit:
        raise ArchiveCacheError(f"현재 HEAD와 expected commit이 다릅니다: {head}")
    if git_bytes("for-each-ref", "--format=%(refname)", "refs/replace").strip():
        raise ArchiveCacheError("git replace ref가 있어 exact source를 신뢰할 수 없습니다")
    grafts = repo / ".git/info/grafts"
    if grafts.is_file() and grafts.stat().st_size > 0:
        raise ArchiveCacheError("legacy git grafts가 있어 exact source를 신뢰할 수 없습니다")
    flags = git_bytes("ls-files", "-v", "-z")
    if any(
        record and (record[:1].islower() or record[:1] == b"S")
        for record in flags.split(b"\0")
    ):
        raise ArchiveCacheError("assume-unchanged/skip-worktree index flag가 있습니다")
    committed_sources: dict[str, bytes] = {}
    for relative in (
        "scripts/elice/public_archive_cache.py",
        "scripts/elice/pget.py",
        "src/deep_anc/data/source_trust.py",
    ):
        path = repo / relative
        committed = git_bytes("cat-file", "blob", f"{expected_commit}:{relative}")
        current = _read_regular_bytes(path, label="tracked exact-source dependency")
        if current != committed:
            raise ArchiveCacheError(f"tracked exact-source dependency blob 불일치: {relative}")
        committed_sources[relative] = committed
    source_trust_path = repo / "src/deep_anc/data/source_trust.py"
    namespace = _exec_trusted_source(
        committed_sources["src/deep_anc/data/source_trust.py"],
        filename=source_trust_path,
        module_name=f"_deep_anc_archive_cache_source_trust_{uuid.uuid4().hex}",
    )
    try:
        namespace["exact_clean_source_evidence"](
            repo,
            expected_commit=expected_commit,
            # Entry/source-trust/pget are loaded by exact source path under -B;
            # unrelated regular __pycache__ below protected roots is therefore
            # non-executable here and follows the shared checker default policy.
            reject_runtime_bytecode=False,
        )
    except namespace["SourceTrustError"] as exc:
        raise ArchiveCacheError(f"robust exact-tree source 검증 실패: {exc}") from exc
    committed_script_sha256 = _sha256_bytes(
        committed_sources["scripts/elice/public_archive_cache.py"]
    )
    if committed_script_sha256 != script_sha256:
        raise ArchiveCacheError("실행 중 archive-cache entry script bytes가 바뀌었습니다")
    committed_pget = committed_sources["scripts/elice/pget.py"]
    if _sha256_bytes(committed_pget) != pget_sha256:
        raise ArchiveCacheError("실행 중 tracked pget.py bytes가 바뀌었습니다")
    return committed_pget


def _validate_remote_root(value: str) -> str:
    if not REMOTE_ROOT.fullmatch(value) or value.endswith(":"):
        raise ArchiveCacheError("remote root는 rclone remote:path 형식이어야 합니다")
    _remote, relative = value.split(":", 1)
    if relative.startswith("/") or "\\" in relative or "//" in relative:
        raise ArchiveCacheError("remote root path는 상대 POSIX 경로여야 합니다")
    if any(part in ("", ".", "..") for part in PurePosixPath(relative).parts):
        raise ArchiveCacheError("remote root에 empty/traversal component가 있습니다")
    if any(character in value for character in ("@", "?", "#")):
        raise ArchiveCacheError("remote root에 secret/URL로 오인할 문자를 허용하지 않습니다")
    return value.rstrip("/")


def _remote_join(root: str, relative: str) -> str:
    return f"{root}/{relative}"


def _provider_head(spec: ArchiveSpec) -> None:
    request = urllib.request.Request(spec.url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            etag = response.headers.get("ETag")
    except OSError as exc:
        raise ArchiveCacheError(f"provider HEAD 실패: {spec.archive_id}: {exc}") from exc
    if length is None or int(length) != spec.expected_size:
        raise ArchiveCacheError(
            f"provider Content-Length 불일치: {spec.archive_id}: {length!r}"
        )
    if spec.provider_etag is not None:
        normalized = (etag or "").strip().strip('"')
        if normalized != spec.provider_etag:
            raise ArchiveCacheError(
                f"provider ETag 불일치: {spec.archive_id}: {etag!r}"
            )


def _run_pget(
    pget: Path,
    spec: ArchiveSpec,
    output: Path,
    *,
    trusted_source: bytes | None = None,
) -> None:
    if output.exists() or output.is_symlink():
        raise ArchiveCacheError(f"unique staging output이 이미 존재합니다: {output}")
    # The general pget CLI deliberately removes unverified partials.  A cache
    # campaign has a stricter forensic contract: no local temporary bytes may
    # be deleted when the overall publish fails.  Execute a held snapshot (the
    # publisher passes the expected-commit blob) so a path swap after source
    # validation can never execute unverified Python.
    source = (
        _read_regular_bytes(pget, label="tracked pget snapshot")
        if trusted_source is None
        else trusted_source
    )
    namespace = _exec_trusted_source(
        source,
        filename=pget,
        module_name=f"_deep_anc_archive_cache_pget_{uuid.uuid4().hex}",
    )

    def preserve_partial(path: Path) -> bool:
        print(f"[preserve] archive-cache temporary: {path}", flush=True)
        return True

    download = namespace["download"]
    download.__globals__["_remove_regular"] = preserve_partial
    try:
        download(spec.url, str(output), 4)
    except namespace["DownloadError"] as exc:
        raise ArchiveCacheError(f"pget 실패; partial을 보존합니다: {spec.archive_id}: {exc}") from exc


def _run_checked(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def _rclone_base(rclone: Path) -> list[str]:
    return [str(rclone), "--no-update-modtime"]


def _rclone_upload_and_verify(
    local: Path,
    remote: str,
    *,
    rclone: Path,
    expected_sha256: str,
) -> None:
    if _hash_file(local)["sha256"] != expected_sha256:
        raise ArchiveCacheError(f"remote upload 직전 local SHA-256 불일치: {local}")
    if "/" not in remote:
        raise ArchiveCacheError(f"content-addressed remote file path 형식 오류: {remote}")
    remote_parent, remote_name = remote.rsplit("/", 1)
    if remote_name != local.name:
        raise ArchiveCacheError(
            f"rclone check basename 결속 실패: local={local.name}, remote={remote_name}"
        )
    _run_checked(
        [
            *_rclone_base(rclone),
            "copyto",
            str(local),
            remote,
            "--immutable",
            *RCLONE_THROTTLE,
        ]
    )
    _run_checked(
        [
            *_rclone_base(rclone),
            "check",
            str(local.parent),
            remote_parent,
            "--one-way",
            "--download",
            "--include",
            f"/{local.name}",
            "--transfers",
            "1",
            "--checkers",
            "1",
            "--tpslimit",
            "2",
            "--drive-pacer-min-sleep",
            "500ms",
        ]
    )
    process = subprocess.Popen(
        [
            *_rclone_base(rclone),
            "cat",
            remote,
            "--transfers",
            "1",
            "--checkers",
            "1",
            "--tpslimit",
            "2",
            "--drive-pacer-min-sleep",
            "500ms",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdout is not None
    digest = hashlib.sha256()
    while True:
        chunk = process.stdout.read(READ_CHUNK)
        if not chunk:
            break
        digest.update(chunk)
    process.wait()
    if process.returncode != 0:
        raise ArchiveCacheError(f"rclone cat readback 실패: {remote}")
    if digest.hexdigest() != expected_sha256:
        raise ArchiveCacheError(
            f"rclone cat SHA-256 readback 불일치: {remote}: {digest.hexdigest()}"
        )


def _archive_cache_path(spec: ArchiveSpec, details: Mapping[str, object]) -> str:
    return (
        f"archives/v1/{spec.archive_id}/bytes_{details['archive_size']}/"
        f"sha256_{details['archive_sha256']}/{spec.filename}"
    )


def _manifest_entry(spec: ArchiveSpec, details: Mapping[str, object]) -> dict[str, object]:
    return {
        "archive_format": spec.archive_format,
        "archive_id": spec.archive_id,
        "archive_sha256": details["archive_sha256"],
        "archive_size": details["archive_size"],
        "cache_path": _archive_cache_path(spec, details),
        "canonical_target": spec.canonical_target,
        "corpus": spec.corpus,
        "filename": spec.filename,
        "member_inventory_sha256": details["member_inventory_sha256"],
        "member_content_inventory_sha256": details[
            "member_content_inventory_sha256"
        ],
        "member_prefix": spec.member_prefix,
        "provider_checksum": spec.provider_checksum,
        "provider_checksum_kind": spec.provider_checksum_kind,
        "provider_etag": spec.provider_etag,
        "regular_file_bytes": details["regular_file_bytes"],
        "regular_file_count": details["regular_file_count"],
        "source_url": spec.url,
        "output_content_inventory_sha256": details[
            "output_content_inventory_sha256"
        ],
        "wav_bytes": details["wav_bytes"],
        "wav_count": details["wav_count"],
    }


def publish(args: argparse.Namespace) -> dict[str, object]:
    repo = _repo_root_from_script()
    staging_root = _require_absolute_directory(
        Path(args.staging_root), label="--staging-root", writable=True
    )
    _ensure_outside(staging_root, repo, label="--staging-root")
    remote_root = _validate_remote_root(args.remote_root)
    expected_commit = args.expected_commit.lower()
    if not HEX40.fullmatch(expected_commit):
        raise ArchiveCacheError("--expected-commit은 lowercase 전체 40자리 SHA여야 합니다")
    rclone = Path(args.rclone).resolve(strict=True)
    pget = (repo / "scripts/elice/pget.py").resolve(strict=True)
    _regular_no_symlink(pget, label="tracked pget")
    script_sha256 = _hash_file(Path(__file__).resolve(strict=True))["sha256"]
    pget_sha256 = _hash_file(pget)["sha256"]
    trusted_pget = _verify_exact_source(
        repo, expected_commit, script_sha256, pget_sha256
    )
    required_staging = sum(spec.expected_size for spec in ARCHIVE_SPECS) + STAGING_HEADROOM_BYTES
    available_staging = shutil.disk_usage(staging_root).free
    if available_staging < required_staging:
        raise ArchiveCacheError(
            "전체 manifest-last campaign staging 여유공간 부족: "
            f"available={available_staging}, required={required_staging}"
        )
    run_dir = staging_root / (
        f"deep_anc_public_archive_cache_{int(time.time())}_{os.getpid()}_{uuid.uuid4().hex}"
    )
    run_dir.mkdir(mode=0o700)
    entries: list[dict[str, object]] = []
    for spec in ARCHIVE_SPECS:
        trusted_pget = _verify_exact_source(
            repo, expected_commit, script_sha256, pget_sha256
        )
        available = shutil.disk_usage(run_dir).free
        if available < spec.expected_size + 128 * 1024 * 1024:
            raise ArchiveCacheError(
                f"sequential archive staging 여유공간 부족: {spec.archive_id}: {available}"
            )
        archive_path = run_dir / spec.filename
        _provider_head(spec)
        _run_pget(pget, spec, archive_path, trusted_source=trusted_pget)
        details = validate_archive(archive_path, spec)
        _verify_exact_source(repo, expected_commit, script_sha256, pget_sha256)
        entry = _manifest_entry(spec, details)
        remote = _remote_join(remote_root, str(entry["cache_path"]))
        _rclone_upload_and_verify(
            archive_path,
            remote,
            rclone=rclone,
            expected_sha256=str(details["archive_sha256"]),
        )
        entries.append(entry)
    _validate_aggregate(entries)
    _verify_exact_source(repo, expected_commit, script_sha256, pget_sha256)
    manifest = {
        "archive_count": len(entries),
        "archives": entries,
        "authority": AUTHORITY,
        "excluded_corpora": ["esc50", "fma_small", "fma_metadata", "librispeech"],
        "kind": MANIFEST_KIND,
        "publisher_commit": expected_commit,
        "publisher_entry_script_sha256": script_sha256,
        "publisher_pget_sha256": pget_sha256,
        "schema_version": SCHEMA_VERSION,
    }
    manifest_bytes = _canonical_json(manifest)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    manifest_path = run_dir / "archive_cache_manifest.json"
    local_manifest = _write_exclusive_fsynced(manifest_path, manifest_bytes)
    if local_manifest != manifest_sha256:
        raise ArchiveCacheError("manifest local SHA-256 internal invariant 불일치")
    _verify_exact_source(repo, expected_commit, script_sha256, pget_sha256)
    relative = (
        f"manifests/v1/sha256_{manifest_sha256}/archive_cache_manifest.json"
    )
    _rclone_upload_and_verify(
        manifest_path,
        _remote_join(remote_root, relative),
        rclone=rclone,
        expected_sha256=manifest_sha256,
    )
    return {
        "archive_count": len(entries),
        "authority": AUTHORITY,
        "manifest_remote_path": relative,
        "manifest_sha256": manifest_sha256,
        "staging_manifest": str(manifest_path),
    }


def _load_manifest(
    path: Path,
    *,
    cache_root: Path,
    expected_sha256: str,
    expected_commit: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    _regular_no_symlink(path, label="archive cache manifest")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(cache_root)
    except ValueError as exc:
        raise ArchiveCacheError("manifest는 --cache-root 안에 있어야 합니다") from exc
    raw = resolved.read_bytes()
    if _sha256_bytes(raw) != expected_sha256:
        raise ArchiveCacheError("archive cache manifest 외부 SHA-256 anchor 불일치")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveCacheError(f"manifest JSON을 읽을 수 없습니다: {exc}") from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != raw:
        raise ArchiveCacheError("manifest는 canonical JSON + 단일 newline이어야 합니다")
    if set(payload) != MANIFEST_KEYS:
        raise ArchiveCacheError("manifest top-level exact key set 불일치")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ArchiveCacheError("archive cache schema_version 불일치")
    if payload.get("kind") != MANIFEST_KIND or payload.get("authority") != AUTHORITY:
        raise ArchiveCacheError("archive cache kind/authority 불일치")
    if payload.get("publisher_commit") != expected_commit:
        raise ArchiveCacheError("archive cache publisher commit과 expected commit이 다릅니다")
    script_digest = payload.get("publisher_entry_script_sha256")
    if not isinstance(script_digest, str) or not HEX64.fullmatch(script_digest):
        raise ArchiveCacheError("publisher entry script SHA-256 형식 불일치")
    pget_digest = payload.get("publisher_pget_sha256")
    if not isinstance(pget_digest, str) or not HEX64.fullmatch(pget_digest):
        raise ArchiveCacheError("publisher tracked pget.py SHA-256 형식 불일치")
    if payload.get("excluded_corpora") != [
        "esc50",
        "fma_small",
        "fma_metadata",
        "librispeech",
    ]:
        raise ArchiveCacheError("excluded corpus 계약 불일치")
    entries = payload.get("archives")
    if not isinstance(entries, list) or payload.get("archive_count") != len(ARCHIVE_SPECS):
        raise ArchiveCacheError("manifest archive_count 불일치")
    if len(entries) != len(ARCHIVE_SPECS) or any(not isinstance(item, dict) for item in entries):
        raise ArchiveCacheError("manifest는 고정 allowlist 10개를 모두 포함해야 합니다")
    ids = tuple(str(item.get("archive_id")) for item in entries)
    if ids != EXPECTED_IDS:
        raise ArchiveCacheError("manifest archive allowlist/order가 다릅니다")
    return payload, entries  # type: ignore[return-value]


def _validate_manifest_entry(entry: Mapping[str, object], spec: ArchiveSpec) -> None:
    if set(entry) != ENTRY_KEYS:
        raise ArchiveCacheError(f"manifest entry exact key set 불일치: {spec.archive_id}")
    fixed = {
        "archive_format": spec.archive_format,
        "archive_id": spec.archive_id,
        "archive_size": spec.expected_size,
        "canonical_target": spec.canonical_target,
        "corpus": spec.corpus,
        "filename": spec.filename,
        "member_prefix": spec.member_prefix,
        "provider_checksum": spec.provider_checksum,
        "provider_checksum_kind": spec.provider_checksum_kind,
        "provider_etag": spec.provider_etag,
        "source_url": spec.url,
        "wav_count": spec.expected_wav_count,
    }
    for key, expected in fixed.items():
        if entry.get(key) != expected:
            raise ArchiveCacheError(f"manifest 고정 field 불일치: {spec.archive_id}.{key}")
    if spec.expected_wav_bytes is not None and entry.get("wav_bytes") != spec.expected_wav_bytes:
        raise ArchiveCacheError(f"manifest WAV bytes 불일치: {spec.archive_id}")
    digest = entry.get("archive_sha256")
    if not isinstance(digest, str) or not HEX64.fullmatch(digest):
        raise ArchiveCacheError(f"manifest archive SHA-256 형식 불일치: {spec.archive_id}")
    for key in (
        "member_content_inventory_sha256",
        "output_content_inventory_sha256",
    ):
        content_digest = entry.get(key)
        if not isinstance(content_digest, str) or not HEX64.fullmatch(content_digest):
            raise ArchiveCacheError(
                f"manifest content inventory SHA-256 형식 불일치: {spec.archive_id}.{key}"
            )
    expected_path = _archive_cache_path(spec, entry)
    if entry.get("cache_path") != expected_path:
        raise ArchiveCacheError(f"content-addressed cache path 불일치: {spec.archive_id}")


def _safe_cache_member(cache_root: Path, relative: str) -> Path:
    if not relative or "\\" in relative:
        raise ArchiveCacheError("cache_path가 POSIX 상대경로가 아닙니다")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ArchiveCacheError("cache_path traversal을 거부합니다")
    cursor = cache_root
    for part in pure.parts:
        cursor = cursor / part
        info = cursor.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ArchiveCacheError(f"cache path symlink를 거부합니다: {cursor}")
    resolved = cursor.resolve(strict=True)
    try:
        resolved.relative_to(cache_root)
    except ValueError as exc:
        raise ArchiveCacheError("cache member가 root 밖으로 벗어났습니다") from exc
    _regular_no_symlink(resolved, label="cached archive")
    return resolved


def _open_safe_target(
    repo: Path, relative: str, *, create_parents: bool = True
) -> _SafeTargetHandle:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or len(pure.parts) == 0
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ArchiveCacheError("canonical target traversal을 거부합니다")
    target = repo.joinpath(*pure.parts)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(repo, flags)
    except OSError as exc:
        raise ArchiveCacheError(f"repository root nofollow open 실패: {repo}: {exc}") from exc
    cursor = repo
    root_info = os.fstat(directory_fd)
    directory_chain: list[tuple[Path, int, tuple[int, int]]] = [
        (repo, directory_fd, (int(root_info.st_dev), int(root_info.st_ino)))
    ]
    try:
        for part in pure.parts[:-1]:
            cursor = cursor / part
            try:
                child_fd = os.open(part, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                if not create_parents:
                    raise ArchiveCacheError(
                        f"canonical existing target parent가 없습니다: {cursor}"
                    )
                try:
                    os.mkdir(part, 0o755, dir_fd=directory_fd)
                    os.fsync(directory_fd)
                    child_fd = os.open(part, flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise ArchiveCacheError(
                        f"canonical target parent no-replace mkdir 실패: {cursor}: {exc}"
                    ) from exc
            except OSError as exc:
                raise ArchiveCacheError(
                    f"canonical target parent symlink/non-directory 거부: {cursor}: {exc}"
                ) from exc
            directory_fd = child_fd
            child_info = os.fstat(child_fd)
            directory_chain.append(
                (cursor, child_fd, (int(child_info.st_dev), int(child_info.st_ino)))
            )
        parent_info = os.fstat(directory_fd)
        return _SafeTargetHandle(
            path=target,
            parent_fd=directory_fd,
            name=pure.parts[-1],
            parent_identity=(int(parent_info.st_dev), int(parent_info.st_ino)),
            directory_chain=directory_chain,
        )
    except BaseException:
        for _path, descriptor, _identity in reversed(directory_chain):
            os.close(descriptor)
        raise


def _safe_target(repo: Path, relative: str) -> Path:
    """Compatibility path resolver; authority writes use the held-fd variant."""

    handle = _open_safe_target(repo, relative)
    try:
        return handle.path
    finally:
        handle.close()


def _assert_target_parent_still_named(handle: _SafeTargetHandle) -> None:
    for named_path, descriptor, expected_identity in handle.directory_chain:
        try:
            held = os.fstat(descriptor)
            named = named_path.lstat()
        except OSError as exc:
            raise ArchiveCacheError(
                f"canonical target held directory identity 검증 실패: {named_path}: {exc}"
            ) from exc
        identity = (int(held.st_dev), int(held.st_ino))
        if (
            identity != expected_identity
            or (int(named.st_dev), int(named.st_ino)) != identity
            or not stat.S_ISDIR(named.st_mode)
            or stat.S_ISLNK(named.st_mode)
        ):
            raise ArchiveCacheError(
                "canonical target parent가 validation 뒤 교체되었습니다 (ancestor): "
                f"{named_path}"
            )
    if handle.directory_chain[-1][2] != handle.parent_identity:
        raise ArchiveCacheError(
            f"canonical target held parent identity invariant 불일치: {handle.path.parent}"
        )


def _read_safe_target_bytes(repo: Path, relative: str, *, label: str) -> bytes:
    """Read an existing repository file through a held nofollow ancestor chain."""

    handle = _open_safe_target(repo, relative, create_parents=False)
    try:
        _assert_target_parent_still_named(handle)
        try:
            descriptor = os.open(
                handle.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=handle.parent_fd,
            )
        except OSError as exc:
            raise ArchiveCacheError(
                f"{label} nofollow openat 실패: {handle.path}: {exc}"
            ) from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ArchiveCacheError(
                    f"{label}는 regular single-link file이어야 합니다: {handle.path}"
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, READ_CHUNK)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            named = os.stat(
                handle.name, dir_fd=handle.parent_fd, follow_symlinks=False
            )
            identity = (int(before.st_dev), int(before.st_ino))
            if (
                (int(after.st_dev), int(after.st_ino)) != identity
                or (int(named.st_dev), int(named.st_ino)) != identity
                or not stat.S_ISREG(after.st_mode)
                or not stat.S_ISREG(named.st_mode)
                or after.st_nlink != 1
                or named.st_nlink != 1
                or int(after.st_size) != int(before.st_size)
                or int(named.st_size) != int(before.st_size)
                or int(after.st_mtime_ns) != int(before.st_mtime_ns)
                or int(after.st_ctime_ns) != int(before.st_ctime_ns)
                or int(named.st_mtime_ns) != int(before.st_mtime_ns)
                or int(named.st_ctime_ns) != int(before.st_ctime_ns)
            ):
                raise ArchiveCacheError(
                    f"{label}가 read 중/pathname에서 바뀌었습니다: {handle.path}"
                )
            _assert_target_parent_still_named(handle)
            return b"".join(chunks)
        except OSError as exc:
            raise ArchiveCacheError(
                f"{label} held readback 실패: {handle.path}: {exc}"
            ) from exc
        finally:
            os.close(descriptor)
    finally:
        handle.close()


def _verify_exact_target(
    handle: _SafeTargetHandle,
    *,
    size: int,
    sha256: str,
    label: str,
    expected_nlink: int = 1,
) -> None:
    """Hash a target through its held parent fd and bind the still-named parent."""

    _assert_target_parent_still_named(handle)
    try:
        descriptor = os.open(
            handle.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=handle.parent_fd,
        )
    except OSError as exc:
        raise ArchiveCacheError(f"{label} nofollow openat 실패: {handle.path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != expected_nlink
            or before.st_size != size
        ):
            raise ArchiveCacheError(f"{label} inode/size 계약 불일치: {handle.path}")
        while True:
            chunk = os.read(descriptor, READ_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        named = os.stat(handle.name, dir_fd=handle.parent_fd, follow_symlinks=False)
        os.lseek(descriptor, 0, os.SEEK_SET)
        confirmation = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, READ_CHUNK)
            if not chunk:
                break
            confirmation.update(chunk)
        confirmed = os.fstat(descriptor)
        identity = (int(before.st_dev), int(before.st_ino))
        if (
            (int(after.st_dev), int(after.st_ino)) != identity
            or (int(confirmed.st_dev), int(confirmed.st_ino)) != identity
            or (int(named.st_dev), int(named.st_ino)) != identity
            or not stat.S_ISREG(after.st_mode)
            or not stat.S_ISREG(confirmed.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or after.st_nlink != expected_nlink
            or confirmed.st_nlink != expected_nlink
            or named.st_nlink != expected_nlink
            or after.st_size != size
            or confirmed.st_size != size
            or named.st_size != size
            or int(after.st_mtime_ns) != int(before.st_mtime_ns)
            or int(after.st_ctime_ns) != int(before.st_ctime_ns)
            or int(confirmed.st_mtime_ns) != int(before.st_mtime_ns)
            or int(confirmed.st_ctime_ns) != int(before.st_ctime_ns)
            or int(named.st_mtime_ns) != int(before.st_mtime_ns)
            or int(named.st_ctime_ns) != int(before.st_ctime_ns)
            or digest.hexdigest() != sha256
            or confirmation.hexdigest() != sha256
        ):
            raise ArchiveCacheError(f"{label} SHA/path identity readback 불일치: {handle.path}")
    except OSError as exc:
        raise ArchiveCacheError(f"{label} openat readback 실패: {handle.path}: {exc}") from exc
    finally:
        os.close(descriptor)
    _assert_target_parent_still_named(handle)


def _matches(path: Path, *, size: int, sha256: str) -> bool:
    try:
        _verify_exact_path(
            path,
            size=size,
            sha256=sha256,
            label="existing canonical archive",
        )
    except ArchiveCacheError:
        return False
    return True


def _verify_exact_path(path: Path, *, size: int, sha256: str, label: str) -> None:
    """Hash one held inode and ensure the pathname still names it at readback."""

    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ArchiveCacheError(f"{label} nofollow open 실패: {path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != size
        ):
            raise ArchiveCacheError(f"{label} inode/size 계약 불일치: {path}")
        while True:
            chunk = os.read(descriptor, READ_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        try:
            named = path.lstat()
        except OSError as exc:
            raise ArchiveCacheError(f"{label} pathname readback 실패: {path}: {exc}") from exc
        os.lseek(descriptor, 0, os.SEEK_SET)
        confirmation = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, READ_CHUNK)
            if not chunk:
                break
            confirmation.update(chunk)
        confirmed = os.fstat(descriptor)
        identity = (int(before.st_dev), int(before.st_ino))
        if (
            (int(after.st_dev), int(after.st_ino)) != identity
            or (int(confirmed.st_dev), int(confirmed.st_ino)) != identity
            or (int(named.st_dev), int(named.st_ino)) != identity
            or not stat.S_ISREG(after.st_mode)
            or not stat.S_ISREG(confirmed.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or after.st_nlink != 1
            or confirmed.st_nlink != 1
            or named.st_nlink != 1
            or after.st_size != size
            or confirmed.st_size != size
            or named.st_size != size
            or int(after.st_mtime_ns) != int(before.st_mtime_ns)
            or int(after.st_ctime_ns) != int(before.st_ctime_ns)
            or int(confirmed.st_mtime_ns) != int(before.st_mtime_ns)
            or int(confirmed.st_ctime_ns) != int(before.st_ctime_ns)
            or int(named.st_mtime_ns) != int(before.st_mtime_ns)
            or int(named.st_ctime_ns) != int(before.st_ctime_ns)
            or digest.hexdigest() != sha256
            or confirmation.hexdigest() != sha256
        ):
            raise ArchiveCacheError(f"{label} SHA/path identity readback 불일치: {path}")
    finally:
        os.close(descriptor)


def _copy_no_replace(source: Path, target: Path, *, size: int, sha256: str) -> str:
    if target.exists() or target.is_symlink():
        if _matches(target, size=size, sha256=sha256):
            return "already_exact"
        raise ArchiveCacheError(f"canonical archive target가 이미 다른 bytes/종류입니다: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.archive-cache-restore.", dir=target.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    written = 0
    try:
        source_descriptor = os.open(
            source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        source_info = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_info.st_mode) or source_info.st_nlink != 1:
            os.close(source_descriptor)
            raise ArchiveCacheError(f"restore source inode 계약 불일치: {source}")
        with os.fdopen(source_descriptor, "rb") as reader, os.fdopen(
            descriptor, "wb"
        ) as writer:
            while True:
                chunk = reader.read(READ_CHUNK)
                if not chunk:
                    break
                writer.write(chunk)
                digest.update(chunk)
                written += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if written != size or digest.hexdigest() != sha256:
            raise ArchiveCacheError(
                f"restore copy readback 불일치; forensic staging을 보존합니다: {temporary}"
            )
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise ArchiveCacheError(
                f"canonical target no-replace 경합; staging을 보존합니다: {temporary}"
            ) from exc
        directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # Deliberately preserve the exclusive staging file on every failure.
        raise
    temporary.unlink()
    return "restored"


def _copy_no_replace_at(
    source: Path,
    target: _SafeTargetHandle,
    *,
    size: int,
    sha256: str,
) -> str:
    """No-replace restore using one held parent fd for every target operation."""

    _assert_target_parent_still_named(target)
    try:
        os.stat(target.name, dir_fd=target.parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ArchiveCacheError(f"canonical target openat 검증 실패: {target.path}: {exc}") from exc
    else:
        try:
            _verify_exact_target(
                target,
                size=size,
                sha256=sha256,
                label="existing canonical archive",
            )
        except ArchiveCacheError as exc:
            raise ArchiveCacheError(
                f"canonical archive target가 이미 다른 bytes/종류입니다: {target.path}"
            ) from exc
        return "already_exact"

    temporary_name = f".{target.name}.archive-cache-restore.{uuid.uuid4().hex}"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=target.parent_fd)
    except OSError as exc:
        raise ArchiveCacheError(
            f"restore exclusive openat staging 실패: {target.path.parent / temporary_name}: {exc}"
        ) from exc
    digest = hashlib.sha256()
    written = 0
    try:
        try:
            source_descriptor = os.open(
                source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
        except BaseException:
            os.close(descriptor)
            raise
        source_info = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_info.st_mode) or source_info.st_nlink != 1:
            os.close(source_descriptor)
            os.close(descriptor)
            raise ArchiveCacheError(f"restore source inode 계약 불일치: {source}")
        reader_handle = os.fdopen(source_descriptor, "rb")
        writer_handle = os.fdopen(descriptor, "wb")
        with reader_handle as reader, writer_handle as writer:
            while True:
                chunk = reader.read(READ_CHUNK)
                if not chunk:
                    break
                writer.write(chunk)
                digest.update(chunk)
                written += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if written != size or digest.hexdigest() != sha256:
            raise ArchiveCacheError(
                "restore copy readback 불일치; forensic staging을 보존합니다: "
                f"{target.path.parent / temporary_name}"
            )
        _assert_target_parent_still_named(target)
        try:
            os.link(
                temporary_name,
                target.name,
                src_dir_fd=target.parent_fd,
                dst_dir_fd=target.parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise ArchiveCacheError(
                "canonical target no-replace 경합; staging을 보존합니다: "
                f"{target.path.parent / temporary_name}"
            ) from exc
        os.fsync(target.parent_fd)
        _verify_exact_target(
            target,
            size=size,
            sha256=sha256,
            label="restored canonical archive",
            expected_nlink=2,
        )
    except BaseException:
        # Deliberately preserve the exclusive staging entry on every failure.
        raise
    os.unlink(temporary_name, dir_fd=target.parent_fd)
    os.fsync(target.parent_fd)
    _verify_exact_target(
        target,
        size=size,
        sha256=sha256,
        label="restored canonical archive",
    )
    return "restored"


def _extracted_member_target(spec: ArchiveSpec, member_name: str) -> str:
    """Map a fixed archive member to the existing untouched-raw layout."""

    normalized = _normalized_member(member_name)
    if spec.archive_id in ("dns_noise_000", "dns_noise_001"):
        root = PurePosixPath("data/raw/noise/dns_fullband")
    elif spec.archive_id == "dns_speech_000":
        root = PurePosixPath("data/raw/noise/speech")
    elif spec.archive_id.startswith("demand_"):
        root = PurePosixPath("data/raw/noise/demand")
    elif spec.archive_id == "mimii_fan":
        root = PurePosixPath("data/raw/noise/machine")
    else:
        raise ArchiveCacheError(f"extract target mapping이 없는 archive: {spec.archive_id}")
    return (root / PurePosixPath(normalized)).as_posix()


def _expected_extracted_outputs(
    validated: Sequence[
        tuple[
            ArchiveSpec,
            Mapping[str, object],
            _HeldArchive,
            Sequence[tuple[str, int, str]],
        ]
    ],
) -> dict[str, int]:
    expected: dict[str, int] = {}
    folded: set[str] = set()
    for spec, _entry, _held, rows in validated:
        for member_name, size, _sha256 in rows:
            target = _extracted_member_target(spec, member_name)
            key = target.casefold()
            if key in folded or target in expected:
                raise ArchiveCacheError(f"archive 사이 extracted target 충돌: {target}")
            folded.add(key)
            expected[target] = int(size)
    return expected


def _extracted_roots() -> tuple[str, ...]:
    return (
        "data/raw/noise/dns_fullband",
        "data/raw/noise/speech",
        "data/raw/noise/demand",
        "data/raw/noise/machine",
    )


def _audit_existing_extracted_subset(
    repo: Path,
    expected: Mapping[str, int],
    *,
    allowed_working_archives: Mapping[str, tuple[int, str]],
    require_complete: bool,
) -> None:
    """Reject unexpected/symlink/hardlinked raw before or after publication."""

    seen: set[str] = set()
    expected_directories: set[str] = set()
    for relative in expected:
        pure = PurePosixPath(relative)
        for stop in range(1, len(pure.parts)):
            expected_directories.add(PurePosixPath(*pure.parts[:stop]).as_posix())
    for root_relative in _extracted_roots():
        root = repo.joinpath(*PurePosixPath(root_relative).parts)
        if not root.exists() and not root.is_symlink():
            continue
        try:
            root_info = root.lstat()
        except OSError as exc:
            raise ArchiveCacheError(f"extracted raw root lstat 실패: {root}: {exc}") from exc
        if root.is_symlink() or not stat.S_ISDIR(root_info.st_mode):
            raise ArchiveCacheError(f"extracted raw root symlink/non-directory 거부: {root}")
        for current_text, directories, files in os.walk(root, followlinks=False):
            current = Path(current_text)
            for name in directories:
                child = current / name
                info = child.lstat()
                relative = child.relative_to(repo).as_posix()
                if (
                    child.is_symlink()
                    or not stat.S_ISDIR(info.st_mode)
                    or relative not in expected_directories
                ):
                    raise ArchiveCacheError(
                        f"예상 밖 extracted directory/symlink를 거부합니다: {child}"
                    )
            for name in files:
                child = current / name
                info = child.lstat()
                relative = child.relative_to(repo).as_posix()
                if relative in allowed_working_archives:
                    archive_size, archive_sha256 = allowed_working_archives[relative]
                    _verify_exact_path(
                        child,
                        size=archive_size,
                        sha256=archive_sha256,
                        label="cache-only working archive ignored by held-fd consume",
                    )
                    continue
                if (
                    child.is_symlink()
                    or not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or relative not in expected
                    or int(info.st_size) != expected[relative]
                ):
                    raise ArchiveCacheError(
                        f"예상 밖/불일치 extracted raw file을 거부합니다: {child}"
                    )
                if relative in seen:
                    raise ArchiveCacheError(f"중복 extracted raw path: {relative}")
                seen.add(relative)
    if require_complete and seen != set(expected):
        missing = sorted(set(expected).difference(seen))[:5]
        raise ArchiveCacheError(
            "held-fd extraction 결과가 완전하지 않습니다: "
            f"seen={len(seen)}, expected={len(expected)}, missing={missing}"
        )


def _hold_published_target(
    target: _SafeTargetHandle,
    *,
    archive_id: str,
    relative: str,
    size: int,
    sha256: str,
) -> _PublishedMember:
    _assert_target_parent_still_named(target)
    try:
        descriptor = os.open(
            target.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=target.parent_fd,
        )
    except OSError as exc:
        raise ArchiveCacheError(
            f"published raw held openat 실패: {target.path}: {exc}"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or int(info.st_size) != size
        ):
            raise ArchiveCacheError(
                f"published raw held inode 계약 불일치: {target.path}"
            )
        digest = hashlib.sha256()
        offset = 0
        while offset < size:
            chunk = os.pread(descriptor, min(READ_CHUNK, size - offset), offset)
            if not chunk:
                raise ArchiveCacheError(
                    f"published raw held short read: {target.path}: offset={offset}"
                )
            digest.update(chunk)
            offset += len(chunk)
        if digest.hexdigest() != sha256:
            raise ArchiveCacheError(f"published raw held SHA 불일치: {target.path}")
        after = os.fstat(descriptor)
        named = os.stat(target.name, dir_fd=target.parent_fd, follow_symlinks=False)
        identity = (int(info.st_dev), int(info.st_ino))
        if (
            (int(after.st_dev), int(after.st_ino)) != identity
            or (int(named.st_dev), int(named.st_ino)) != identity
            or not stat.S_ISREG(after.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or after.st_nlink != 1
            or named.st_nlink != 1
            or int(after.st_size) != size
            or int(named.st_size) != size
            or int(after.st_mtime_ns) != int(info.st_mtime_ns)
            or int(after.st_ctime_ns) != int(info.st_ctime_ns)
            or int(named.st_mtime_ns) != int(info.st_mtime_ns)
            or int(named.st_ctime_ns) != int(info.st_ctime_ns)
        ):
            raise ArchiveCacheError(
                f"published raw held path identity 불일치: {target.path}"
            )
        _assert_target_parent_still_named(target)
        return _PublishedMember(
            archive_id=archive_id,
            relative=relative,
            size=size,
            sha256=sha256,
            descriptor=descriptor,
            identity=identity,
            mtime_ns=int(info.st_mtime_ns),
            ctime_ns=int(info.st_ctime_ns),
        )
    except BaseException:
        os.close(descriptor)
        raise


def _publish_member_stream_no_replace(
    repo: Path,
    relative: str,
    reader: BinaryIO,
    *,
    archive_id: str,
    expected_size: int,
) -> _PublishedMember:
    """Publish one decompressed member through a held parent dirfd."""

    target = _open_safe_target(repo, relative)
    temporary_name = f".{target.name}.archive-cache-extract.{uuid.uuid4().hex}"
    descriptor = -1
    try:
        _assert_target_parent_still_named(target)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(temporary_name, flags, 0o600, dir_fd=target.parent_fd)
        except OSError as exc:
            raise ArchiveCacheError(
                "extracted member exclusive staging 실패: "
                f"{target.path.parent / temporary_name}: {exc}"
            ) from exc
        digest = hashlib.sha256()
        written = 0
        with os.fdopen(descriptor, "wb") as writer:
            descriptor = -1
            while True:
                chunk = reader.read(READ_CHUNK)
                if not chunk:
                    break
                writer.write(chunk)
                digest.update(chunk)
                written += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if written != expected_size:
            raise ArchiveCacheError(
                "extracted member size 불일치; forensic staging을 보존합니다: "
                f"{target.path.parent / temporary_name}: {written} != {expected_size}"
            )
        member_sha256 = digest.hexdigest()
        _assert_target_parent_still_named(target)
        try:
            os.link(
                temporary_name,
                target.name,
                src_dir_fd=target.parent_fd,
                dst_dir_fd=target.parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            try:
                _verify_exact_target(
                    target,
                    size=expected_size,
                    sha256=member_sha256,
                    label="existing extracted raw member",
                )
            except ArchiveCacheError as exc:
                raise ArchiveCacheError(
                    "extracted raw target가 이미 다른 bytes/종류입니다; "
                    f"forensic staging을 보존합니다: {target.path}"
                ) from exc
            os.unlink(temporary_name, dir_fd=target.parent_fd)
            os.fsync(target.parent_fd)
            return _hold_published_target(
                target,
                archive_id=archive_id,
                relative=relative,
                size=expected_size,
                sha256=member_sha256,
            )
        os.fsync(target.parent_fd)
        _verify_exact_target(
            target,
            size=expected_size,
            sha256=member_sha256,
            label="held-fd extracted raw member",
            expected_nlink=2,
        )
        os.unlink(temporary_name, dir_fd=target.parent_fd)
        os.fsync(target.parent_fd)
        _verify_exact_target(
            target,
            size=expected_size,
            sha256=member_sha256,
            label="held-fd extracted raw member",
        )
        return _hold_published_target(
            target,
            archive_id=archive_id,
            relative=relative,
            size=expected_size,
            sha256=member_sha256,
        )
    except BaseException:
        # Any completed staging file is intentionally retained for forensics.
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        target.close()


def _extract_held_tar_bz2(
    repo: Path,
    held: _HeldArchive,
    spec: ArchiveSpec,
    published: list[_PublishedMember],
) -> tuple[int, int]:
    file_count = 0
    byte_count = 0
    try:
        with held.reader() as raw, tarfile.open(fileobj=raw, mode="r:bz2") as archive:
            seen: set[str] = set()
            for member in archive:
                normalized = _normalized_member(member.name)
                folded = normalized.casefold()
                if folded in seen:
                    raise ArchiveCacheError(
                        f"duplicate/case-colliding tar member: {member.name}"
                    )
                seen.add(folded)
                if member.isdir():
                    continue
                if not member.isfile() or not normalized.casefold().endswith(".wav"):
                    raise ArchiveCacheError(
                        f"held extractor가 non-WAV/non-regular tar member를 거부합니다: {member.name}"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ArchiveCacheError(f"tar member stream을 열 수 없습니다: {member.name}")
                with extracted:
                    published.append(
                        _publish_member_stream_no_replace(
                            repo,
                            _extracted_member_target(spec, normalized),
                            extracted,
                            archive_id=spec.archive_id,
                            expected_size=int(member.size),
                        )
                    )
                file_count += 1
                byte_count += int(member.size)
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ArchiveCacheError(
            f"held-fd tar extraction 실패: {spec.archive_id}: {exc}"
        ) from exc
    return file_count, byte_count


def _extract_held_zip(
    repo: Path,
    held: _HeldArchive,
    spec: ArchiveSpec,
    published: list[_PublishedMember],
) -> tuple[int, int]:
    file_count = 0
    byte_count = 0
    try:
        with held.reader() as raw, zipfile.ZipFile(raw) as archive:
            seen: set[str] = set()
            for member in archive.infolist():
                normalized = _normalized_member(member.filename)
                folded = normalized.casefold()
                if folded in seen:
                    raise ArchiveCacheError(
                        f"duplicate/case-colliding ZIP member: {member.filename}"
                    )
                seen.add(folded)
                mode = (member.external_attr >> 16) & 0xFFFF
                kind = stat.S_IFMT(mode)
                if member.is_dir():
                    if kind not in (0, stat.S_IFDIR):
                        raise ArchiveCacheError(
                            f"ZIP directory mode 불일치: {member.filename}"
                        )
                    continue
                if member.flag_bits & 0x1 or kind not in (0, stat.S_IFREG):
                    raise ArchiveCacheError(
                        f"held extractor가 encrypted/non-regular ZIP member를 거부합니다: {member.filename}"
                    )
                if not normalized.casefold().endswith(".wav"):
                    raise ArchiveCacheError(
                        f"held extractor가 non-WAV ZIP member를 거부합니다: {member.filename}"
                    )
                with archive.open(member, "r") as extracted:
                    published.append(
                        _publish_member_stream_no_replace(
                            repo,
                            _extracted_member_target(spec, normalized),
                            extracted,
                            archive_id=spec.archive_id,
                            expected_size=int(member.file_size),
                        )
                    )
                file_count += 1
                byte_count += int(member.file_size)
    except (zipfile.BadZipFile, zlib.error, OSError, EOFError) as exc:
        raise ArchiveCacheError(
            f"held-fd ZIP extraction 실패: {spec.archive_id}: {exc}"
        ) from exc
    return file_count, byte_count


def _ensure_consumption_fd_budget(member_count: int) -> None:
    """Reserve enough process-local fds to hold every published WAV to receipt."""

    required = member_count + len(ARCHIVE_SPECS) + 512
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    unlimited = hard == resource.RLIM_INFINITY
    if not unlimited and hard < required:
        raise ArchiveCacheError(
            "held-fd consume에 필요한 RLIMIT_NOFILE hard limit이 부족합니다: "
            f"hard={hard}, required={required}"
        )
    if soft < required:
        try:
            resource.setrlimit(
                resource.RLIMIT_NOFILE,
                (required, hard),
            )
        except (OSError, ValueError) as exc:
            raise ArchiveCacheError(
                "held-fd consume RLIMIT_NOFILE soft limit을 올릴 수 없습니다: "
                f"soft={soft}, hard={hard}, required={required}: {exc}"
            ) from exc
    confirmed, _confirmed_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if confirmed < required:
        raise ArchiveCacheError(
            f"held-fd consume descriptor budget 불충족: {confirmed} < {required}"
        )


def _require_target_filesystem_space(
    repo: Path,
    requirements: Mapping[str, int],
    *,
    stage_every_entry: bool,
    label: str,
) -> None:
    """Check the actual target mount before publishing the first byte.

    ``df(repo)`` is insufficient when ``data/raw`` is a separate mount.  Walk
    every fixed destination with the same no-symlink dirfd traversal used by
    publication, group it by the parent directory's device, and require the
    missing final bytes plus one sequential staging member and fixed headroom.
    The check is intentionally conservative; availability failure must happen
    before a consume intent or a restored archive is published.
    """

    grouped: dict[int, dict[str, int]] = {}
    for relative, raw_size in requirements.items():
        size = int(raw_size)
        if size < 0:
            raise ArchiveCacheError(f"{label} target size가 음수입니다: {relative}")
        target = _open_safe_target(repo, relative)
        try:
            parent = os.fstat(target.parent_fd)
            filesystem = os.fstatvfs(target.parent_fd)
            fragment_size = int(filesystem.f_frsize or filesystem.f_bsize)
            available = int(filesystem.f_bavail) * fragment_size
            available_inodes = int(filesystem.f_favail)
            device = int(parent.st_dev)
            group = grouped.setdefault(
                device,
                {
                    "available": available,
                    "available_inodes": available_inodes,
                    "missing": 0,
                    "missing_inodes": 0,
                    "staging_peak": 0,
                },
            )
            group["available"] = min(group["available"], available)
            group["available_inodes"] = min(
                group["available_inodes"], available_inodes
            )
            try:
                os.stat(target.name, dir_fd=target.parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                group["missing"] += size
                group["missing_inodes"] += 1
                needs_stage = True
            except OSError as exc:
                raise ArchiveCacheError(
                    f"{label} target preflight openat 실패: {target.path}: {exc}"
                ) from exc
            else:
                needs_stage = stage_every_entry
            if needs_stage:
                group["staging_peak"] = max(group["staging_peak"], size)
        finally:
            target.close()
    for device, group in grouped.items():
        active = group["missing"] > 0 or group["staging_peak"] > 0
        required = (
            group["missing"]
            + group["staging_peak"]
            + (STAGING_HEADROOM_BYTES if active else 0)
        )
        if group["available"] < required:
            raise ArchiveCacheError(
                f"{label} target filesystem 여유공간 부족: device={device}, "
                f"available={group['available']}, required={required}, "
                f"missing={group['missing']}, staging_peak={group['staging_peak']}"
            )
        # One temporary/staging inode is sufficient because members are
        # published sequentially.  A fixed receipt/directory reserve covers
        # intent, member inventory, origin, completion and crash forensics.
        required_inodes = (
            group["missing_inodes"]
            + (1 if active else 0)
            + (STAGING_HEADROOM_INODES if active else 0)
        )
        if group["available_inodes"] < required_inodes:
            raise ArchiveCacheError(
                f"{label} target filesystem inode 부족: device={device}, "
                f"available_inodes={group['available_inodes']}, "
                f"required_inodes={required_inodes}, "
                f"missing_inodes={group['missing_inodes']}"
            )


def _verify_published_members_held(
    repo: Path,
    published: Sequence[_PublishedMember],
    expected: Mapping[str, int],
) -> str:
    """Final SHA+inode+named-path recheck while every output fd remains held."""

    by_path = {item.relative: item for item in published}
    if len(by_path) != len(published) or set(by_path) != set(expected):
        raise ArchiveCacheError(
            "held published member set 불일치: "
            f"held={len(by_path)}, expected={len(expected)}"
        )
    inventory = hashlib.sha256()
    for relative in sorted(expected):
        item = by_path[relative]
        if item.size != expected[relative]:
            raise ArchiveCacheError(f"held published member size 계약 불일치: {relative}")
        before = os.fstat(item.descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or int(before.st_size) != item.size
            or (int(before.st_dev), int(before.st_ino)) != item.identity
            or int(before.st_mtime_ns) != item.mtime_ns
            or int(before.st_ctime_ns) != item.ctime_ns
        ):
            raise ArchiveCacheError(f"held published inode가 바뀌었습니다: {relative}")
        digest = hashlib.sha256()
        offset = 0
        while offset < item.size:
            chunk = os.pread(
                item.descriptor,
                min(READ_CHUNK, item.size - offset),
                offset,
            )
            if not chunk:
                raise ArchiveCacheError(
                    f"held published final short read: {relative}: offset={offset}"
                )
            digest.update(chunk)
            offset += len(chunk)
        if digest.hexdigest() != item.sha256:
            raise ArchiveCacheError(f"held published final SHA 불일치: {relative}")
        target = _open_safe_target(repo, relative)
        try:
            _assert_target_parent_still_named(target)
            try:
                named_fd = os.open(
                    target.name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=target.parent_fd,
                )
            except OSError as exc:
                raise ArchiveCacheError(
                    f"held published final named openat 실패: {relative}: {exc}"
                ) from exc
            try:
                named = os.fstat(named_fd)
                if (
                    not stat.S_ISREG(named.st_mode)
                    or named.st_nlink != 1
                    or int(named.st_size) != item.size
                    or (int(named.st_dev), int(named.st_ino)) != item.identity
                    or int(named.st_mtime_ns) != item.mtime_ns
                    or int(named.st_ctime_ns) != item.ctime_ns
                ):
                    raise ArchiveCacheError(
                        f"held published final named inode 불일치: {relative}"
                    )
            finally:
                os.close(named_fd)
            after = os.fstat(item.descriptor)
            if (
                (int(after.st_dev), int(after.st_ino)) != item.identity
                or after.st_nlink != 1
                or int(after.st_size) != item.size
                or int(after.st_mtime_ns) != item.mtime_ns
                or int(after.st_ctime_ns) != item.ctime_ns
            ):
                raise ArchiveCacheError(
                    f"held published final readback identity 불일치: {relative}"
                )
            _assert_target_parent_still_named(target)
        finally:
            target.close()
        inventory.update(
            _canonical_json(
                {
                    "path": relative,
                    "sha256": item.sha256,
                    "size": item.size,
                }
            )
        )
    return inventory.hexdigest()


def _publish_cache_origin_receipt(
    repo: Path,
    *,
    manifest_sha256: str,
    expected_commit: str,
    script_sha256: str,
    pget_sha256: str,
    entries: Sequence[Mapping[str, object]],
) -> tuple[str, str]:
    payload = {
        "archives": [
            {
                "archive_id": entry["archive_id"],
                "archive_sha256": entry["archive_sha256"],
                "archive_size": entry["archive_size"],
                "canonical_target": entry["canonical_target"],
            }
            for entry in entries
        ],
        "authority": CACHE_ORIGIN_AUTHORITY,
        "kind": CACHE_ORIGIN_KIND,
        "manifest_sha256": manifest_sha256,
        "publisher_commit": expected_commit,
        "restorer_entry_script_sha256": script_sha256,
        "restorer_pget_sha256": pget_sha256,
        "schema_version": 1,
    }
    raw = _canonical_json(payload)
    digest = _sha256_bytes(raw)
    relative = _cache_origin_receipt_path(manifest_sha256, expected_commit)
    target = _open_safe_target(repo, relative)
    try:
        _assert_target_parent_still_named(target)
        try:
            os.stat(target.name, dir_fd=target.parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(target.name, flags, 0o600, dir_fd=target.parent_fd)
            offset = 0
            try:
                while offset < len(raw):
                    written = os.write(descriptor, memoryview(raw)[offset:])
                    if written <= 0:
                        raise ArchiveCacheError(
                            f"cache origin receipt short write; file을 보존합니다: {target.path}"
                        )
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(target.parent_fd)
        except OSError as exc:
            raise ArchiveCacheError(
                f"cache origin receipt openat 검증 실패: {target.path}: {exc}"
            ) from exc
        _verify_exact_target(
            target,
            size=len(raw),
            sha256=digest,
            label="cache origin receipt",
        )
    finally:
        target.close()
    return relative, digest


def _verify_optional_working_archives(
    repo: Path, entries: Sequence[Mapping[str, object]]
) -> None:
    """If cache-only targets exist, require all of their exact manifest bytes."""

    for entry in entries:
        target = _open_safe_target(repo, str(entry["canonical_target"]))
        try:
            _assert_target_parent_still_named(target)
            try:
                os.stat(target.name, dir_fd=target.parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ArchiveCacheError(
                    f"cache-only working archive openat 실패: {target.path}: {exc}"
                ) from exc
            _verify_exact_target(
                target,
                size=int(entry["archive_size"]),
                sha256=str(entry["archive_sha256"]),
                label="cache-only working archive",
            )
        finally:
            target.close()


def _consumption_receipt_paths(
    manifest_sha256: str, expected_commit: str
) -> tuple[str, str]:
    stem = f"{manifest_sha256}.{expected_commit}"
    return (
        f"{CACHE_CONSUMPTION_DIRECTORY}/consume_intent.{stem}.json",
        f"{CACHE_CONSUMPTION_DIRECTORY}/consume_complete.{stem}.json",
    )


def _cache_origin_receipt_path(manifest_sha256: str, expected_commit: str) -> str:
    return (
        f"{CACHE_ORIGIN_DIRECTORY}/archive_cache_origin."
        f"{manifest_sha256}.{expected_commit}.json"
    )


def _consumption_inventory_path(manifest_sha256: str, expected_commit: str) -> str:
    return (
        f"{CACHE_CONSUMPTION_DIRECTORY}/consume_inventory."
        f"{manifest_sha256}.{expected_commit}.json"
    )


def _publish_safe_json_receipt(
    repo: Path,
    *,
    relative: str,
    payload: Mapping[str, object],
    label: str,
) -> str:
    raw = _canonical_json(payload)
    digest = _sha256_bytes(raw)
    target = _open_safe_target(repo, relative)
    try:
        _assert_target_parent_still_named(target)
        try:
            os.stat(target.name, dir_fd=target.parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(target.name, flags, 0o600, dir_fd=target.parent_fd)
            offset = 0
            try:
                while offset < len(raw):
                    written = os.write(descriptor, memoryview(raw)[offset:])
                    if written <= 0:
                        raise ArchiveCacheError(
                            f"{label} short write; file을 보존합니다: {target.path}"
                        )
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(target.parent_fd)
        except OSError as exc:
            raise ArchiveCacheError(
                f"{label} openat 검증 실패: {target.path}: {exc}"
            ) from exc
        _verify_exact_target(
            target,
            size=len(raw),
            sha256=digest,
            label=label,
        )
    finally:
        target.close()
    return digest


def _validate_consumption_directory(
    repo: Path,
    *,
    allowed_relatives: Sequence[str],
) -> None:
    directory = repo.joinpath(*PurePosixPath(CACHE_CONSUMPTION_DIRECTORY).parts)
    if not directory.exists() and not directory.is_symlink():
        return
    info = directory.lstat()
    if directory.is_symlink() or not stat.S_ISDIR(info.st_mode):
        raise ArchiveCacheError(
            f"cache consumption marker directory symlink/non-directory: {directory}"
        )
    allowed_names = {PurePosixPath(item).name for item in allowed_relatives}
    for child in directory.iterdir():
        child_info = child.lstat()
        if (
            child.name not in allowed_names
            or child.is_symlink()
            or not stat.S_ISREG(child_info.st_mode)
            or child_info.st_nlink != 1
        ):
            raise ArchiveCacheError(
                f"matching external anchor 밖 cache consumption marker를 거부합니다: {child}"
            )


def _publish_consumption_intent(
    repo: Path,
    *,
    manifest_sha256: str,
    expected_commit: str,
    script_sha256: str,
    pget_sha256: str,
    entries: Sequence[Mapping[str, object]],
    expected_outputs: Mapping[str, int],
) -> tuple[str, str, str]:
    intent_relative, completion_relative = _consumption_receipt_paths(
        manifest_sha256, expected_commit
    )
    inventory_relative = _consumption_inventory_path(
        manifest_sha256, expected_commit
    )
    _validate_consumption_directory(
        repo,
        allowed_relatives=(
            intent_relative,
            inventory_relative,
            completion_relative,
        ),
    )
    output_inventory = hashlib.sha256()
    for path, size in sorted(expected_outputs.items()):
        output_inventory.update(_canonical_json({"path": path, "size": size}))
    payload = {
        "archive_count": len(entries),
        "archive_manifest_sha256": manifest_sha256,
        "authority": CACHE_CONSUMPTION_AUTHORITY,
        "expected_output_bytes": sum(expected_outputs.values()),
        "expected_output_count": len(expected_outputs),
        "expected_output_path_size_inventory_sha256": output_inventory.hexdigest(),
        "kind": CACHE_CONSUMPTION_INTENT_KIND,
        "publisher_commit": expected_commit,
        "restorer_entry_script_sha256": script_sha256,
        "restorer_pget_sha256": pget_sha256,
        "schema_version": 1,
        "state": "in_progress_or_completed_requires_matching_external_anchors",
    }
    digest = _publish_safe_json_receipt(
        repo,
        relative=intent_relative,
        payload=payload,
        label="cache consumption intent",
    )
    return intent_relative, digest, completion_relative


def _publish_consumption_inventory(
    repo: Path,
    *,
    manifest_sha256: str,
    expected_commit: str,
    entries: Sequence[Mapping[str, object]],
    published: Sequence[_PublishedMember],
) -> tuple[str, str]:
    rows = [
        {
            "archive_id": item.archive_id,
            "path": item.relative,
            "sha256": item.sha256,
            "size": item.size,
        }
        for item in sorted(published, key=lambda selected: selected.relative)
    ]
    by_archive: dict[str, list[tuple[str, int, str]]] = {}
    for row in rows:
        by_archive.setdefault(str(row["archive_id"]), []).append(
            (str(row["path"]), int(row["size"]), str(row["sha256"]))
        )
    entry_by_id = {str(entry["archive_id"]): entry for entry in entries}
    if set(by_archive) != set(entry_by_id):
        raise ArchiveCacheError("consumption inventory archive_id set 불일치")
    for archive_id, content_rows in by_archive.items():
        digest = hashlib.sha256()
        for path, size, sha256 in sorted(content_rows):
            digest.update(
                _canonical_json({"path": path, "sha256": sha256, "size": size})
            )
        if digest.hexdigest() != entry_by_id[archive_id][
            "output_content_inventory_sha256"
        ]:
            raise ArchiveCacheError(
                f"external manifest output content inventory 불일치: {archive_id}"
            )
    payload = {
        "archive_manifest_sha256": manifest_sha256,
        "authority": CACHE_CONSUMPTION_AUTHORITY,
        "kind": "deep_anc_archive_cache_consumed_member_inventory",
        "output_bytes": sum(int(row["size"]) for row in rows),
        "output_count": len(rows),
        "publisher_commit": expected_commit,
        "rows": rows,
        "schema_version": 1,
    }
    relative = _consumption_inventory_path(manifest_sha256, expected_commit)
    digest = _publish_safe_json_receipt(
        repo,
        relative=relative,
        payload=payload,
        label="cache consumed member inventory",
    )
    return relative, digest


def _publish_consumption_completion(
    repo: Path,
    *,
    relative: str,
    manifest_sha256: str,
    expected_commit: str,
    intent_relative: str,
    intent_sha256: str,
    inventory_relative: str,
    inventory_sha256: str,
    origin_relative: str,
    origin_sha256: str,
    output_count: int,
    output_bytes: int,
    output_inventory_sha256: str,
) -> str:
    payload = {
        "archive_manifest_sha256": manifest_sha256,
        "authority": CACHE_CONSUMPTION_AUTHORITY,
        "intent_path": intent_relative,
        "intent_sha256": intent_sha256,
        "kind": CACHE_CONSUMPTION_COMPLETION_KIND,
        "member_inventory_path": inventory_relative,
        "member_inventory_sha256": inventory_sha256,
        "origin_receipt_path": origin_relative,
        "origin_receipt_sha256": origin_sha256,
        "output_bytes": output_bytes,
        "output_count": output_count,
        "output_path_size_sha256_inventory_sha256": output_inventory_sha256,
        "publisher_commit": expected_commit,
        "schema_version": 1,
        "state": "held_fd_consume_complete_pending_exact_raw_and_decoder_authority",
    }
    return _publish_safe_json_receipt(
        repo,
        relative=relative,
        payload=payload,
        label="cache consumption completion",
    )


def guard_plain(args: argparse.Namespace) -> dict[str, object]:
    """Reject cache-origin raw when matching external anchors are absent."""

    repo = Path(args.repo_root).resolve(strict=True)
    expected_commit = args.expected_commit.lower()
    if not HEX40.fullmatch(expected_commit):
        raise ArchiveCacheError("--expected-commit은 lowercase 전체 40자리여야 합니다")
    script_sha256 = _hash_file(Path(__file__).resolve(strict=True))["sha256"]
    pget_sha256 = _hash_file(repo / "scripts/elice/pget.py")["sha256"]
    _verify_exact_source(repo, expected_commit, script_sha256, pget_sha256)
    directory = repo.joinpath(*PurePosixPath(CACHE_CONSUMPTION_DIRECTORY).parts)
    if not directory.exists() and not directory.is_symlink():
        return {
            "authority": AUTHORITY,
            "state": "no_archive_cache_consumption_marker",
        }
    info = directory.lstat()
    if directory.is_symlink() or not stat.S_ISDIR(info.st_mode):
        raise ArchiveCacheError(
            f"cache consumption marker directory가 안전하지 않습니다: {directory}"
        )
    # Even an empty directory can be the crash residue between mkdir and the
    # O_EXCL intent write.  Plain bootstrap must not infer that no cache raw was
    # published; matching external anchors are required to resume or attest it.
    entries = list(directory.iterdir())
    raise ArchiveCacheError(
        "cache consumption intent/completion directory가 있어 plain bootstrap으로 "
        "raw를 재사용할 수 없습니다. matching archive-cache external anchors가 "
        f"필수입니다: entries={len(entries)}"
    )


def _read_canonical_json_receipt(
    repo: Path,
    relative: str,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, object], str]:
    raw = _read_safe_target_bytes(repo, relative, label=label)
    digest = _sha256_bytes(raw)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ArchiveCacheError(f"{label} SHA-256 binding 불일치")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveCacheError(f"{label} JSON parse 실패: {exc}") from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != raw:
        raise ArchiveCacheError(f"{label}는 canonical JSON이어야 합니다")
    return payload, digest


def _validate_consumption_intent_payload(
    payload: Mapping[str, object],
    *,
    manifest_sha256: str,
    expected_commit: str,
    script_sha256: str,
    pget_sha256: str,
    entries: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
) -> None:
    keys = {
        "archive_count",
        "archive_manifest_sha256",
        "authority",
        "expected_output_bytes",
        "expected_output_count",
        "expected_output_path_size_inventory_sha256",
        "kind",
        "publisher_commit",
        "restorer_entry_script_sha256",
        "restorer_pget_sha256",
        "schema_version",
        "state",
    }
    path_size = hashlib.sha256()
    for row in rows:
        path_size.update(
            _canonical_json({"path": row["path"], "size": row["size"]})
        )
    expected_bytes = sum(int(row["size"]) for row in rows)
    if set(payload) != keys or payload != {
        "archive_count": len(entries),
        "archive_manifest_sha256": manifest_sha256,
        "authority": CACHE_CONSUMPTION_AUTHORITY,
        "expected_output_bytes": expected_bytes,
        "expected_output_count": len(rows),
        "expected_output_path_size_inventory_sha256": path_size.hexdigest(),
        "kind": CACHE_CONSUMPTION_INTENT_KIND,
        "publisher_commit": expected_commit,
        "restorer_entry_script_sha256": script_sha256,
        "restorer_pget_sha256": pget_sha256,
        "schema_version": 1,
        "state": "in_progress_or_completed_requires_matching_external_anchors",
    }:
        raise ArchiveCacheError("cache consumption intent schema/content 불일치")


def _validate_cache_origin_payload(
    payload: Mapping[str, object],
    *,
    manifest_sha256: str,
    expected_commit: str,
    script_sha256: str,
    pget_sha256: str,
    entries: Sequence[Mapping[str, object]],
) -> None:
    expected = {
        "archives": [
            {
                "archive_id": entry["archive_id"],
                "archive_sha256": entry["archive_sha256"],
                "archive_size": entry["archive_size"],
                "canonical_target": entry["canonical_target"],
            }
            for entry in entries
        ],
        "authority": CACHE_ORIGIN_AUTHORITY,
        "kind": CACHE_ORIGIN_KIND,
        "manifest_sha256": manifest_sha256,
        "publisher_commit": expected_commit,
        "restorer_entry_script_sha256": script_sha256,
        "restorer_pget_sha256": pget_sha256,
        "schema_version": 1,
    }
    if payload != expected:
        raise ArchiveCacheError("cache origin receipt schema/content 불일치")


def _is_cache_training_wav_path(relative: str) -> bool:
    return any(
        relative == root or relative.startswith(f"{root}/")
        for root in CACHE_TRAINING_WAV_ROOTS
    )


def _walk_held_cache_wavs(descriptor: int, relative: str) -> set[str]:
    """List WAVs below one held nofollow root without reopening its pathname."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    before = os.fstat(descriptor)
    if not stat.S_ISDIR(before.st_mode):
        raise ArchiveCacheError(f"cache raw root가 directory가 아닙니다: {relative}")
    found: set[str] = set()
    try:
        names = sorted(os.listdir(descriptor))
    except OSError as exc:
        raise ArchiveCacheError(f"cache raw root list 실패: {relative}: {exc}") from exc
    for name in names:
        if name in ("", ".", "..") or "/" in name or "\x00" in name:
            raise ArchiveCacheError(f"cache raw unsafe entry name: {relative}/{name!r}")
        try:
            named = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ArchiveCacheError(
                f"cache raw entry stat 실패: {relative}/{name}: {exc}"
            ) from exc
        child_relative = f"{relative}/{name}"
        if stat.S_ISDIR(named.st_mode):
            try:
                child = os.open(name, flags, dir_fd=descriptor)
            except OSError as exc:
                raise ArchiveCacheError(
                    f"cache raw child nofollow open 실패: {child_relative}: {exc}"
                ) from exc
            try:
                held = os.fstat(child)
                if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
                    raise ArchiveCacheError(
                        f"cache raw child가 stat/open 사이 교체됐습니다: {child_relative}"
                    )
                found.update(_walk_held_cache_wavs(child, child_relative))
                after = os.fstat(child)
                current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if (
                    (held.st_dev, held.st_ino, held.st_mtime_ns, held.st_ctime_ns)
                    != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns)
                    or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino)
                    or not stat.S_ISDIR(current.st_mode)
                ):
                    raise ArchiveCacheError(
                        f"cache raw child가 walk 중 변경됐습니다: {child_relative}"
                    )
            finally:
                os.close(child)
        elif stat.S_ISREG(named.st_mode):
            if name.casefold().endswith(".wav"):
                if child_relative in found:
                    raise ArchiveCacheError(f"cache raw WAV 중복: {child_relative}")
                found.add(child_relative)
        else:
            raise ArchiveCacheError(
                f"cache raw root의 symlink/device/non-regular를 거부합니다: {child_relative}"
            )
    after = os.fstat(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ArchiveCacheError(f"cache raw directory가 walk 중 변경됐습니다: {relative}")
    return found


def _hold_cache_raw_roots(repo: Path) -> list[_SafeTargetHandle]:
    handles: list[_SafeTargetHandle] = []
    try:
        for relative in CACHE_TRAINING_WAV_ROOTS:
            handle = _open_safe_target(
                repo, f"{relative}/.__exact_wav_set_sentinel__", create_parents=False
            )
            _assert_target_parent_still_named(handle)
            handles.append(handle)
        return handles
    except BaseException:
        for handle in reversed(handles):
            handle.close()
        raise


def _verify_held_cache_raw_wav_set(
    handles: Sequence[_SafeTargetHandle], expected_paths: set[str]
) -> None:
    actual: set[str] = set()
    for relative, handle in zip(CACHE_TRAINING_WAV_ROOTS, handles, strict=True):
        _assert_target_parent_still_named(handle)
        actual.update(_walk_held_cache_wavs(handle.parent_fd, relative))
        _assert_target_parent_still_named(handle)
    if actual != expected_paths:
        extra = sorted(actual - expected_paths)[:20]
        missing = sorted(expected_paths - actual)[:20]
        raise ArchiveCacheError(
            "cache raw WAV exact-set 불일치: "
            f"extra={extra}, missing={missing}, actual={len(actual)}, "
            f"expected={len(expected_paths)}"
        )


def _validate_decoder_audit_projection(
    repo: Path,
    audit_argument: str,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    audit_path = Path(audit_argument)
    if audit_path.is_absolute():
        audit_path = Path(os.path.abspath(audit_path))
    else:
        audit_path = Path(os.path.abspath(repo / audit_path))
    try:
        audit_relative = audit_path.relative_to(repo).as_posix()
    except ValueError as exc:
        raise ArchiveCacheError("--decoder-audit는 repository 아래여야 합니다") from exc
    raw = _read_safe_target_bytes(
        repo, audit_relative, label="decoder audit cache projection"
    )
    try:
        audit = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveCacheError(f"decoder audit JSON parse 실패: {exc}") from exc
    if not isinstance(audit, dict) or _canonical_json(audit) != raw:
        raise ArchiveCacheError("decoder audit는 canonical JSON이어야 합니다")
    inventory = audit.get("inventory")
    if (
        audit.get("schema_version") != 1
        or audit.get("status") != "complete"
        or not isinstance(inventory, list)
    ):
        raise ArchiveCacheError("decoder audit 완료/schema/inventory 계약 불일치")
    audit_sha256 = audit.get("audit_sha256")
    if not isinstance(audit_sha256, str) or not HEX64.fullmatch(audit_sha256):
        raise ArchiveCacheError("decoder audit semantic SHA 형식 불일치")
    semantic_basis = {key: value for key, value in audit.items() if key != "audit_sha256"}
    semantic_raw = json.dumps(
        semantic_basis,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(semantic_raw).hexdigest() != audit_sha256:
        raise ArchiveCacheError("decoder audit semantic SHA self-digest 불일치")
    by_path: dict[str, tuple[int, str]] = {}
    for record in inventory:
        if not isinstance(record, dict):
            raise ArchiveCacheError("decoder audit inventory row가 object가 아닙니다")
        relative = record.get("relative_path")
        size = record.get("content_size")
        sha256 = record.get("content_sha256")
        if (
            not isinstance(relative, str)
            or relative in by_path
            or type(size) is not int
            or size < 0
            or not isinstance(sha256, str)
            or not HEX64.fullmatch(sha256)
        ):
            raise ArchiveCacheError("decoder audit inventory path/size/SHA 불일치")
        by_path[relative] = (size, sha256)
    expected_cache_paths = {str(row["path"]) for row in rows}
    decoder_cache_paths = {
        relative for relative in by_path if _is_cache_training_wav_path(relative)
    }
    if decoder_cache_paths != expected_cache_paths:
        raise ArchiveCacheError(
            "decoder audit cache raw exact-set 불일치: "
            f"extra={sorted(decoder_cache_paths - expected_cache_paths)[:20]}, "
            f"missing={sorted(expected_cache_paths - decoder_cache_paths)[:20]}"
        )
    projection = hashlib.sha256()
    for row in rows:
        relative = str(row["path"])
        expected = (int(row["size"]), str(row["sha256"]))
        if by_path.get(relative) != expected:
            raise ArchiveCacheError(
                f"decoder audit/cache external inventory 불일치: {relative}"
            )
        projection.update(
            _canonical_json(
                {"path": relative, "sha256": expected[1], "size": expected[0]}
            )
        )
    return {
        "decoder_audit_path": audit_relative,
        "decoder_audit_file_sha256": _sha256_bytes(raw),
        "decoder_audit_semantic_sha256": audit_sha256,
        "decoder_cache_projection_sha256": projection.hexdigest(),
    }


def verify_consumed_raw(args: argparse.Namespace) -> dict[str, object]:
    """Rebind current raw bytes to the externally anchored member inventory."""

    if args.decoder_projection_only:
        raise ArchiveCacheError(
            "--decoder-projection-only는 caller raw hash와 projection 사이 pathname "
            "replacement를 결속할 수 없어 폐기됐습니다. held-fd current raw "
            "검증을 사용하세요"
        )
    repo = Path(args.repo_root).resolve(strict=True)
    cache_root = _require_absolute_directory(
        Path(args.cache_root), label="--cache-root", writable=False
    )
    _ensure_outside(cache_root, repo, label="--cache-root")
    expected_manifest_sha256 = args.expected_manifest_sha256.lower()
    expected_commit = args.expected_commit.lower()
    if not HEX64.fullmatch(expected_manifest_sha256) or not HEX40.fullmatch(
        expected_commit
    ):
        raise ArchiveCacheError("manifest SHA-256 또는 expected commit 형식이 잘못됐습니다")
    script_sha256 = _hash_file(Path(__file__).resolve(strict=True))["sha256"]
    pget_sha256 = _hash_file(repo / "scripts/elice/pget.py")["sha256"]
    _verify_exact_source(repo, expected_commit, script_sha256, pget_sha256)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        raise ArchiveCacheError("--manifest는 절대경로여야 합니다")
    payload, entries = _load_manifest(
        manifest_path,
        cache_root=cache_root,
        expected_sha256=expected_manifest_sha256,
        expected_commit=expected_commit,
    )
    if payload["publisher_entry_script_sha256"] != script_sha256:
        raise ArchiveCacheError("publisher와 raw verifier entry script SHA가 다릅니다")
    if payload["publisher_pget_sha256"] != pget_sha256:
        raise ArchiveCacheError("publisher와 raw verifier pget SHA가 다릅니다")
    for spec, entry in zip(ARCHIVE_SPECS, entries, strict=True):
        _validate_manifest_entry(entry, spec)

    intent_relative, completion_relative = _consumption_receipt_paths(
        expected_manifest_sha256, expected_commit
    )
    inventory_relative = _consumption_inventory_path(
        expected_manifest_sha256, expected_commit
    )
    _validate_consumption_directory(
        repo,
        allowed_relatives=(intent_relative, inventory_relative, completion_relative),
    )
    completion, completion_sha256 = _read_canonical_json_receipt(
        repo,
        completion_relative,
        label="cache consumption completion",
    )
    completion_keys = {
        "archive_manifest_sha256",
        "authority",
        "intent_path",
        "intent_sha256",
        "kind",
        "member_inventory_path",
        "member_inventory_sha256",
        "origin_receipt_path",
        "origin_receipt_sha256",
        "output_bytes",
        "output_count",
        "output_path_size_sha256_inventory_sha256",
        "publisher_commit",
        "schema_version",
        "state",
    }
    if set(completion) != completion_keys:
        raise ArchiveCacheError("cache consumption completion exact key set 불일치")
    if (
        completion.get("archive_manifest_sha256") != expected_manifest_sha256
        or completion.get("publisher_commit") != expected_commit
        or completion.get("authority") != CACHE_CONSUMPTION_AUTHORITY
        or completion.get("kind") != CACHE_CONSUMPTION_COMPLETION_KIND
        or completion.get("schema_version") != 1
        or completion.get("intent_path") != intent_relative
        or completion.get("member_inventory_path") != inventory_relative
        or completion.get("state")
        != "held_fd_consume_complete_pending_exact_raw_and_decoder_authority"
    ):
        raise ArchiveCacheError("cache consumption completion external binding 불일치")
    intent_sha256 = completion.get("intent_sha256")
    inventory_sha256 = completion.get("member_inventory_sha256")
    origin_relative = completion.get("origin_receipt_path")
    origin_sha256 = completion.get("origin_receipt_sha256")
    expected_origin_relative = (
        f"{CACHE_ORIGIN_DIRECTORY}/archive_cache_origin."
        f"{expected_manifest_sha256}.{expected_commit}.json"
    )
    if any(
        not isinstance(value, str) or not HEX64.fullmatch(value)
        for value in (intent_sha256, inventory_sha256, origin_sha256)
    ) or origin_relative != expected_origin_relative:
        raise ArchiveCacheError("cache consumption completion SHA/path 형식 불일치")
    intent, _actual_intent_sha256 = _read_canonical_json_receipt(
        repo,
        intent_relative,
        label="cache consumption intent",
        expected_sha256=str(intent_sha256),
    )
    origin, _actual_origin_sha256 = _read_canonical_json_receipt(
        repo,
        str(origin_relative),
        label="cache origin receipt",
        expected_sha256=str(origin_sha256),
    )
    inventory, actual_inventory_sha256 = _read_canonical_json_receipt(
        repo,
        inventory_relative,
        label="cache consumed member inventory",
        expected_sha256=str(inventory_sha256),
    )
    inventory_keys = {
        "archive_manifest_sha256",
        "authority",
        "kind",
        "output_bytes",
        "output_count",
        "publisher_commit",
        "rows",
        "schema_version",
    }
    if set(inventory) != inventory_keys or (
        inventory.get("archive_manifest_sha256") != expected_manifest_sha256
        or inventory.get("publisher_commit") != expected_commit
        or inventory.get("authority") != CACHE_CONSUMPTION_AUTHORITY
        or inventory.get("kind") != "deep_anc_archive_cache_consumed_member_inventory"
        or inventory.get("schema_version") != 1
    ):
        raise ArchiveCacheError("cache consumed member inventory binding 불일치")
    rows = inventory.get("rows")
    if not isinstance(rows, list):
        raise ArchiveCacheError("cache consumed member inventory rows가 list가 아닙니다")
    entry_by_id = {str(entry["archive_id"]): entry for entry in entries}
    grouped: dict[str, list[tuple[str, int, str]]] = {}
    validated_rows: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    total_bytes = 0
    current_projection = hashlib.sha256()
    ordered_paths: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "archive_id",
            "path",
            "sha256",
            "size",
        }:
            raise ArchiveCacheError("cache consumed member row exact key set 불일치")
        archive_id = row.get("archive_id")
        relative = row.get("path")
        sha256 = row.get("sha256")
        size = row.get("size")
        if (
            not isinstance(archive_id, str)
            or archive_id not in entry_by_id
            or not isinstance(relative, str)
            or not isinstance(sha256, str)
            or not HEX64.fullmatch(sha256)
            or type(size) is not int
            or size < 0
            or relative in seen_paths
        ):
            raise ArchiveCacheError("cache consumed member row value 불일치")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
            raise ArchiveCacheError(f"cache consumed member path traversal: {relative}")
        seen_paths.add(relative)
        ordered_paths.append(relative)
        grouped.setdefault(archive_id, []).append((relative, size, sha256))
        validated_rows.append(
            {
                "archive_id": archive_id,
                "path": relative,
                "sha256": sha256,
                "size": size,
            }
        )
        total_bytes += size
        current_projection.update(
            _canonical_json({"path": relative, "sha256": sha256, "size": size})
        )
    if set(grouped) != set(entry_by_id):
        raise ArchiveCacheError("cache consumed inventory archive set 불일치")
    if ordered_paths != sorted(ordered_paths):
        raise ArchiveCacheError("cache consumed inventory rows가 path 순으로 canonical하지 않습니다")
    for archive_id, content_rows in grouped.items():
        digest = hashlib.sha256()
        for relative, size, sha256 in sorted(content_rows):
            digest.update(
                _canonical_json(
                    {"path": relative, "sha256": sha256, "size": size}
                )
            )
        if digest.hexdigest() != entry_by_id[archive_id][
            "output_content_inventory_sha256"
        ]:
            raise ArchiveCacheError(
                f"externally anchored output content projection 불일치: {archive_id}"
            )
    if (
        type(inventory.get("output_count")) is not int
        or type(inventory.get("output_bytes")) is not int
        or type(completion.get("output_count")) is not int
        or type(completion.get("output_bytes")) is not int
        or inventory.get("output_count") != len(rows)
        or inventory.get("output_bytes") != total_bytes
        or completion.get("output_count") != len(rows)
        or completion.get("output_bytes") != total_bytes
        or completion.get("output_path_size_sha256_inventory_sha256")
        != current_projection.hexdigest()
        or actual_inventory_sha256 != inventory_sha256
    ):
        raise ArchiveCacheError("cache consumed inventory count/bytes/completion 불일치")
    _validate_consumption_intent_payload(
        intent,
        manifest_sha256=expected_manifest_sha256,
        expected_commit=expected_commit,
        script_sha256=script_sha256,
        pget_sha256=pget_sha256,
        entries=entries,
        rows=validated_rows,
    )
    _validate_cache_origin_payload(
        origin,
        manifest_sha256=expected_manifest_sha256,
        expected_commit=expected_commit,
        script_sha256=script_sha256,
        pget_sha256=pget_sha256,
        entries=entries,
    )
    expected_cache_paths = {str(row["path"]) for row in validated_rows}
    raw_root_handles = _hold_cache_raw_roots(repo)
    try:
        _verify_held_cache_raw_wav_set(raw_root_handles, expected_cache_paths)
        decoder_binding: dict[str, object] = {}
        if args.decoder_audit is not None:
            decoder_binding = _validate_decoder_audit_projection(
                repo, args.decoder_audit, validated_rows
            )
            if (
                decoder_binding["decoder_cache_projection_sha256"]
                != current_projection.hexdigest()
            ):
                raise ArchiveCacheError("decoder/cache projection aggregate 불일치")
        _verify_held_cache_raw_wav_set(raw_root_handles, expected_cache_paths)
        # Open and retain every current raw inode.  A replacement after an early
        # row's hash remains visible at the final all-path identity pass; an
        # in-place same-size write changes mtime/ctime and is rejected as well.
        _ensure_consumption_fd_budget(len(validated_rows))
        held_current: list[_PublishedMember] = []
        expected_sizes = {
            str(row["path"]): int(row["size"]) for row in validated_rows
        }
        try:
            for row in validated_rows:
                target = _open_safe_target(repo, str(row["path"]))
                try:
                    held_current.append(
                        _hold_published_target(
                            target,
                            archive_id=str(row["archive_id"]),
                            relative=str(row["path"]),
                            size=int(row["size"]),
                            sha256=str(row["sha256"]),
                        )
                    )
                finally:
                    target.close()
            held_projection = _verify_published_members_held(
                repo, held_current, expected_sizes
            )
            if held_projection != current_projection.hexdigest():
                raise ArchiveCacheError("current held raw projection 불일치")
            if decoder_binding and (
                decoder_binding["decoder_cache_projection_sha256"] != held_projection
            ):
                raise ArchiveCacheError("decoder/cache projection aggregate 불일치")
            _verify_exact_source(repo, expected_commit, script_sha256, pget_sha256)
            if (
                _verify_published_members_held(repo, held_current, expected_sizes)
                != held_projection
            ):
                raise ArchiveCacheError("current raw가 verifier 실행 중 바뀌었습니다")
            _verify_held_cache_raw_wav_set(raw_root_handles, expected_cache_paths)
            result: dict[str, object] = {
                "authority": CACHE_CONSUMPTION_AUTHORITY,
                "completion_path": completion_relative,
                "completion_sha256": completion_sha256,
                "current_output_count": len(rows),
                "current_output_bytes": total_bytes,
                "current_output_projection_sha256": held_projection,
                "inventory_path": inventory_relative,
                "inventory_sha256": actual_inventory_sha256,
                "manifest_sha256": expected_manifest_sha256,
                "state": (
                    "current_raw_and_decoder_audit_match_externally_anchored_inventory"
                    if decoder_binding
                    else "current_raw_matches_externally_anchored_consumed_member_inventory"
                ),
            }
            result.update(decoder_binding)
            return result
        finally:
            for item in held_current:
                item.close()
    finally:
        for handle in reversed(raw_root_handles):
            handle.close()


def restore(args: argparse.Namespace) -> dict[str, object]:
    repo = Path(args.repo_root).resolve(strict=True)
    cache_root = _require_absolute_directory(
        Path(args.cache_root), label="--cache-root", writable=False
    )
    _ensure_outside(cache_root, repo, label="--cache-root")
    expected_sha256 = args.expected_manifest_sha256.lower()
    expected_commit = args.expected_commit.lower()
    if not HEX64.fullmatch(expected_sha256):
        raise ArchiveCacheError("--expected-manifest-sha256는 lowercase 64자리여야 합니다")
    if not HEX40.fullmatch(expected_commit):
        raise ArchiveCacheError("--expected-commit은 lowercase 전체 40자리여야 합니다")
    script_sha256 = _hash_file(Path(__file__).resolve(strict=True))["sha256"]
    pget_sha256 = _hash_file(repo / "scripts/elice/pget.py")["sha256"]
    _verify_exact_source(repo, expected_commit, script_sha256, pget_sha256)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        raise ArchiveCacheError("--manifest는 절대경로여야 합니다")
    _payload, entries = _load_manifest(
        manifest_path,
        cache_root=cache_root,
        expected_sha256=expected_sha256,
        expected_commit=expected_commit,
    )
    if _payload["publisher_entry_script_sha256"] != script_sha256:
        raise ArchiveCacheError("publisher와 restore entry script SHA-256이 다릅니다")
    if _payload["publisher_pget_sha256"] != pget_sha256:
        raise ArchiveCacheError("publisher와 restore tracked pget.py SHA-256이 다릅니다")
    validated: list[tuple[ArchiveSpec, dict[str, object], Path]] = []
    for spec, entry in zip(ARCHIVE_SPECS, entries, strict=True):
        _validate_manifest_entry(entry, spec)
        source = _safe_cache_member(cache_root, str(entry["cache_path"]))
        details = validate_archive(source, spec)
        for key in (
            "archive_sha256",
            "archive_size",
            "member_inventory_sha256",
            "member_content_inventory_sha256",
            "output_content_inventory_sha256",
            "regular_file_bytes",
            "regular_file_count",
            "wav_bytes",
            "wav_count",
        ):
            if details[key] != entry.get(key):
                raise ArchiveCacheError(f"cached archive/manifest 불일치: {spec.archive_id}.{key}")
        validated.append((spec, entry, source))
    _validate_aggregate(entries)
    targets: list[_SafeTargetHandle] = []
    states: list[dict[str, str]] = []
    try:
        for spec, _entry, _source in validated:
            targets.append(_open_safe_target(repo, spec.canonical_target))
        restore_space_requirements = {
            spec.canonical_target: int(entry["archive_size"])
            for spec, entry, _source in validated
        }
        restore_space_requirements[
            _cache_origin_receipt_path(expected_sha256, expected_commit)
        ] = CACHE_RECEIPT_RESERVE_BYTES
        _require_target_filesystem_space(
            repo,
            restore_space_requirements,
            stage_every_entry=False,
            label="archive restore",
        )
        _verify_exact_source(repo, expected_commit, script_sha256, pget_sha256)
        # Validate the complete incoming set before publishing even the first target.
        for (spec, entry, source), target in zip(validated, targets, strict=True):
            state = _copy_no_replace_at(
                source,
                target,
                size=int(entry["archive_size"]),
                sha256=str(entry["archive_sha256"]),
            )
            states.append(
                {
                    "archive_id": spec.archive_id,
                    "canonical_target": spec.canonical_target,
                    "state": state,
                }
            )
        _verify_exact_source(repo, expected_commit, script_sha256, pget_sha256)
        # Keep every validated parent dirfd held through the final target
        # readback and receipt publication; do not reopen a pathname that may
        # have become an intermediate symlink.
        for (_spec, entry, _source), target in zip(validated, targets, strict=True):
            _verify_exact_target(
                target,
                size=int(entry["archive_size"]),
                sha256=str(entry["archive_sha256"]),
                label="restored canonical archive receipt precondition",
            )
        origin_relative, origin_sha256 = _publish_cache_origin_receipt(
            repo,
            manifest_sha256=expected_sha256,
            expected_commit=expected_commit,
            script_sha256=script_sha256,
            pget_sha256=pget_sha256,
            entries=entries,
        )
    finally:
        for target in targets:
            target.close()
    return {
        "archive_count": len(states),
        "archives": states,
        "authority": AUTHORITY,
        "manifest_sha256": expected_sha256,
        "origin_receipt": origin_relative,
        "origin_receipt_sha256": origin_sha256,
        "state": "archives_restored_no_raw_authority",
    }


def consume(args: argparse.Namespace) -> dict[str, object]:
    """Verify and extract fixed cache archives without reopening their paths.

    This is the only cache-backed full-bootstrap handoff.  It deliberately does
    not call :func:`restore` and does not ask a shell extractor to reopen the
    restored canonical filenames.  All ten cache inodes stay open from their
    manifest comparison through member publication and the final SHA readback.
    The result remains transport/origin evidence only; the bootstrap's exact
    raw inventory and decoder audit are still the raw/training authority.
    """

    repo = Path(args.repo_root).resolve(strict=True)
    cache_root = _require_absolute_directory(
        Path(args.cache_root), label="--cache-root", writable=False
    )
    _ensure_outside(cache_root, repo, label="--cache-root")
    expected_sha256 = args.expected_manifest_sha256.lower()
    expected_commit = args.expected_commit.lower()
    if not HEX64.fullmatch(expected_sha256):
        raise ArchiveCacheError("--expected-manifest-sha256는 lowercase 64자리여야 합니다")
    if not HEX40.fullmatch(expected_commit):
        raise ArchiveCacheError("--expected-commit은 lowercase 전체 40자리여야 합니다")
    script_sha256 = _hash_file(Path(__file__).resolve(strict=True))["sha256"]
    pget_sha256 = _hash_file(repo / "scripts/elice/pget.py")["sha256"]
    _verify_exact_source(repo, expected_commit, script_sha256, pget_sha256)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        raise ArchiveCacheError("--manifest는 절대경로여야 합니다")
    payload, entries = _load_manifest(
        manifest_path,
        cache_root=cache_root,
        expected_sha256=expected_sha256,
        expected_commit=expected_commit,
    )
    if payload["publisher_entry_script_sha256"] != script_sha256:
        raise ArchiveCacheError("publisher와 consume entry script SHA-256이 다릅니다")
    if payload["publisher_pget_sha256"] != pget_sha256:
        raise ArchiveCacheError("publisher와 consume tracked pget.py SHA-256이 다릅니다")

    held_archives: list[_HeldArchive] = []
    published_members: list[_PublishedMember] = []
    validated: list[
        tuple[ArchiveSpec, dict[str, object], _HeldArchive, tuple[tuple[str, int], ...]]
    ] = []
    try:
        # Open all fixed objects first.  A later pathname replacement cannot
        # change any byte source used by validation or extraction.
        for spec, entry in zip(ARCHIVE_SPECS, entries, strict=True):
            _validate_manifest_entry(entry, spec)
            held = _open_held_cache_member(cache_root, str(entry["cache_path"]))
            held_archives.append(held)
        for (spec, entry), held in zip(
            zip(ARCHIVE_SPECS, entries, strict=True), held_archives, strict=True
        ):
            details, rows = _validate_held_archive(held, spec)
            for key in (
                "archive_sha256",
                "archive_size",
                "member_inventory_sha256",
                "member_content_inventory_sha256",
                "output_content_inventory_sha256",
                "regular_file_bytes",
                "regular_file_count",
                "wav_bytes",
                "wav_count",
            ):
                if details[key] != entry.get(key):
                    raise ArchiveCacheError(
                        f"cached held archive/manifest 불일치: {spec.archive_id}.{key}"
                    )
            validated.append((spec, entry, held, rows))
        _validate_aggregate(entries)
        expected_outputs = _expected_extracted_outputs(validated)
        _ensure_consumption_fd_budget(len(expected_outputs))
        allowed_working_archives = {
            str(entry["canonical_target"]): (
                int(entry["archive_size"]),
                str(entry["archive_sha256"]),
            )
            for entry in entries
        }
        _verify_optional_working_archives(repo, entries)
        _audit_existing_extracted_subset(
            repo,
            expected_outputs,
            allowed_working_archives=allowed_working_archives,
            require_complete=False,
        )
        intent_relative, completion_relative = _consumption_receipt_paths(
            expected_sha256, expected_commit
        )
        inventory_relative = _consumption_inventory_path(
            expected_sha256, expected_commit
        )
        consume_space_requirements = dict(expected_outputs)
        consume_space_requirements.update(
            {
                intent_relative: CACHE_RECEIPT_RESERVE_BYTES,
                completion_relative: CACHE_RECEIPT_RESERVE_BYTES,
                inventory_relative: CACHE_RECEIPT_RESERVE_BYTES,
                _cache_origin_receipt_path(
                    expected_sha256, expected_commit
                ): CACHE_RECEIPT_RESERVE_BYTES,
            }
        )
        _require_target_filesystem_space(
            repo,
            consume_space_requirements,
            stage_every_entry=True,
            label="archive consume",
        )
        _verify_exact_source(repo, expected_commit, script_sha256, pget_sha256)
        published_intent_relative, intent_sha256, published_completion_relative = (
            _publish_consumption_intent(
                repo,
                manifest_sha256=expected_sha256,
                expected_commit=expected_commit,
                script_sha256=script_sha256,
                pget_sha256=pget_sha256,
                entries=entries,
                expected_outputs=expected_outputs,
            )
        )
        if (
            published_intent_relative != intent_relative
            or published_completion_relative != completion_relative
        ):
            raise ArchiveCacheError("cache consumption receipt path 재유도 불일치")

        extracted_files = 0
        extracted_bytes = 0
        for spec, entry, held, _rows in validated:
            if spec.archive_format == "tar.bz2":
                file_count, byte_count = _extract_held_tar_bz2(
                    repo, held, spec, published_members
                )
            elif spec.archive_format == "zip":
                file_count, byte_count = _extract_held_zip(
                    repo, held, spec, published_members
                )
            else:
                raise ArchiveCacheError(
                    f"지원하지 않는 held extractor format: {spec.archive_id}"
                )
            if (
                file_count != int(entry["regular_file_count"])
                or byte_count != int(entry["regular_file_bytes"])
            ):
                raise ArchiveCacheError(
                    f"held extraction count/bytes 불일치: {spec.archive_id}"
                )
            extracted_files += file_count
            extracted_bytes += byte_count

        # Detect same-inode mutation during decompression.  Pathname replacement
        # is harmless because it never changes this held inode; in-place byte
        # mutation is not harmless and prevents receipt publication.
        for _spec, entry, held, _rows in validated:
            digest = _hash_held_archive(
                held, ("sha256",), require_initial_metadata=True
            )["sha256"]
            if digest != entry["archive_sha256"]:
                raise ArchiveCacheError(
                    f"held archive가 extraction 중 변경되었습니다: {held.path}"
                )
        _audit_existing_extracted_subset(
            repo,
            expected_outputs,
            allowed_working_archives=allowed_working_archives,
            require_complete=True,
        )
        _verify_optional_working_archives(repo, entries)
        output_inventory_sha256 = _verify_published_members_held(
            repo, published_members, expected_outputs
        )
        inventory_relative, inventory_sha256 = _publish_consumption_inventory(
            repo,
            manifest_sha256=expected_sha256,
            expected_commit=expected_commit,
            entries=entries,
            published=published_members,
        )
        output_inventory_sha256 = _verify_published_members_held(
            repo, published_members, expected_outputs
        )
        _verify_exact_source(repo, expected_commit, script_sha256, pget_sha256)
        origin_relative, origin_sha256 = _publish_cache_origin_receipt(
            repo,
            manifest_sha256=expected_sha256,
            expected_commit=expected_commit,
            script_sha256=script_sha256,
            pget_sha256=pget_sha256,
            entries=entries,
        )
        output_inventory_sha256 = _verify_published_members_held(
            repo, published_members, expected_outputs
        )
        completion_sha256 = _publish_consumption_completion(
            repo,
            relative=completion_relative,
            manifest_sha256=expected_sha256,
            expected_commit=expected_commit,
            intent_relative=intent_relative,
            intent_sha256=intent_sha256,
            inventory_relative=inventory_relative,
            inventory_sha256=inventory_sha256,
            origin_relative=origin_relative,
            origin_sha256=origin_sha256,
            output_count=extracted_files,
            output_bytes=extracted_bytes,
            output_inventory_sha256=output_inventory_sha256,
        )
        output_inventory_sha256 = _verify_published_members_held(
            repo, published_members, expected_outputs
        )
    finally:
        for member in published_members:
            member.close()
        for held in held_archives:
            held.close()
    return {
        "archive_count": len(validated),
        "authority": AUTHORITY,
        "extracted_file_count": extracted_files,
        "extracted_regular_bytes": extracted_bytes,
        "manifest_sha256": expected_sha256,
        "consumption_completion": completion_relative,
        "consumption_completion_sha256": completion_sha256,
        "consumption_intent": intent_relative,
        "consumption_intent_sha256": intent_sha256,
        "consumed_member_inventory": inventory_relative,
        "consumed_member_inventory_sha256": inventory_sha256,
        "origin_receipt": origin_relative,
        "origin_receipt_sha256": origin_sha256,
        "state": "held_fd_extracted_pending_exact_raw_and_decoder_audits",
    }


def verify_manifest(args: argparse.Namespace) -> dict[str, object]:
    """Cheap external-anchor gate; archive bytes are verified again by restore."""

    repo = Path(args.repo_root).resolve(strict=True)
    cache_root = _require_absolute_directory(
        Path(args.cache_root), label="--cache-root", writable=False
    )
    _ensure_outside(cache_root, repo, label="--cache-root")
    expected_sha256 = args.expected_manifest_sha256.lower()
    expected_commit = args.expected_commit.lower()
    if not HEX64.fullmatch(expected_sha256) or not HEX40.fullmatch(expected_commit):
        raise ArchiveCacheError("manifest SHA-256 또는 expected commit 형식이 잘못됐습니다")
    script_sha256 = _hash_file(Path(__file__).resolve(strict=True))["sha256"]
    pget_sha256 = _hash_file(repo / "scripts/elice/pget.py")["sha256"]
    _verify_exact_source(repo, expected_commit, script_sha256, pget_sha256)
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        raise ArchiveCacheError("--manifest는 절대경로여야 합니다")
    payload, entries = _load_manifest(
        manifest,
        cache_root=cache_root,
        expected_sha256=expected_sha256,
        expected_commit=expected_commit,
    )
    if payload["publisher_entry_script_sha256"] != script_sha256:
        raise ArchiveCacheError("publisher와 verify entry script SHA-256이 다릅니다")
    if payload["publisher_pget_sha256"] != pget_sha256:
        raise ArchiveCacheError("publisher와 verify tracked pget.py SHA-256이 다릅니다")
    for spec, entry in zip(ARCHIVE_SPECS, entries, strict=True):
        _validate_manifest_entry(entry, spec)
        _safe_cache_member(cache_root, str(entry["cache_path"]))
    return {
        "archive_count": len(entries),
        "authority": AUTHORITY,
        "manifest_sha256": expected_sha256,
        "state": "manifest_anchor_verified_archive_bytes_not_yet_verified",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DNS3+DEMAND6+MIMII1 fixed public archive cache"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish_parser = subparsers.add_parser(
        "publish", help="sequential download, validate, immutable upload, manifest-last publish"
    )
    publish_parser.add_argument("--staging-root", required=True)
    publish_parser.add_argument("--remote-root", required=True)
    publish_parser.add_argument("--expected-commit", required=True)
    publish_parser.add_argument("--rclone", default=shutil.which("rclone") or "rclone")
    restore_parser = subparsers.add_parser(
        "restore", help="verify read-only incoming cache and no-replace-stage canonical archives"
    )
    restore_parser.add_argument("--cache-root", required=True)
    restore_parser.add_argument("--manifest", required=True)
    restore_parser.add_argument("--expected-manifest-sha256", required=True)
    restore_parser.add_argument("--expected-commit", required=True)
    restore_parser.add_argument("--repo-root", required=True)
    consume_parser = subparsers.add_parser(
        "consume",
        help=(
            "verify fixed cache inodes and extract through held fds; "
            "exact raw/decoder audits remain authoritative"
        ),
    )
    consume_parser.add_argument("--cache-root", required=True)
    consume_parser.add_argument("--manifest", required=True)
    consume_parser.add_argument("--expected-manifest-sha256", required=True)
    consume_parser.add_argument("--expected-commit", required=True)
    consume_parser.add_argument("--repo-root", required=True)
    guard_parser = subparsers.add_parser(
        "guard-plain",
        help="reject plain bootstrap when any cache-consumption marker directory exists",
    )
    guard_parser.add_argument("--expected-commit", required=True)
    guard_parser.add_argument("--repo-root", required=True)
    raw_verify_parser = subparsers.add_parser(
        "verify-consumed-raw",
        help="rehash current raw against externally anchored per-member cache inventory",
    )
    raw_verify_parser.add_argument("--cache-root", required=True)
    raw_verify_parser.add_argument("--manifest", required=True)
    raw_verify_parser.add_argument("--expected-manifest-sha256", required=True)
    raw_verify_parser.add_argument("--expected-commit", required=True)
    raw_verify_parser.add_argument("--repo-root", required=True)
    raw_verify_parser.add_argument(
        "--decoder-audit",
        help=(
            "optional completed decoder audit whose path/size/content_sha256 rows "
            "must exactly contain every externally anchored cache output"
        ),
    )
    raw_verify_parser.add_argument(
        "--decoder-projection-only",
        action="store_true",
        help=(
            "deprecated and always rejected: a caller-side raw scan cannot be "
            "atomically bound to a later projection comparison"
        ),
    )
    verify_parser = subparsers.add_parser(
        "verify-manifest", help="verify the external manifest anchor without publishing targets"
    )
    verify_parser.add_argument("--cache-root", required=True)
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.add_argument("--expected-manifest-sha256", required=True)
    verify_parser.add_argument("--expected-commit", required=True)
    verify_parser.add_argument("--repo-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "publish":
            result = publish(args)
        elif args.command == "restore":
            result = restore(args)
        elif args.command == "consume":
            result = consume(args)
        elif args.command == "guard-plain":
            result = guard_plain(args)
        elif args.command == "verify-consumed-raw":
            result = verify_consumed_raw(args)
        else:
            result = verify_manifest(args)
    except (ArchiveCacheError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"[BLOCKED] {exc}", file=sys.stderr)
        return 2
    print(_canonical_json(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
