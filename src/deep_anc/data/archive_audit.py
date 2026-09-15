"""검증된 임시 ZIP/TAR.GZ의 무추출 전수 검사. Drive/다운로드/삭제는 하지 않는다.

음원 한 개만 제한된 메모리에 보관하고 metadata는 chunk 해시만 계산한다.
정책 상한 초과도 실패로 기록하며 조용히 선별하거나 디스크로 spill하지 않는다.
공식 source의 진위는 staging receipt의 출처 선언에 의존한다(서명 인증 아님).
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tarfile
from typing import Any, BinaryIO
import zipfile
import zlib

import soundfile as sf

from .public_corpus import inspect_audio
from .archive_staging import validate_staging_provenance

CHUNK_BYTES = 1024 * 1024
MAX_ENTRIES = 100000
MAX_NAME_BYTES = 4096
MAX_TOTAL_NAME_BYTES = 16 * 1024 * 1024
MAX_ZIP_DIRECTORY_BYTES = 64 * 1024 * 1024
MAX_TAR_HEADER_BYTES = 1024 * 1024
MAX_AUDIO_CHANNELS = 64
MAX_PCM_SAMPLES = 100000000
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3"}


def _safe_path(value: str | Path) -> Path:
    path = Path(value).absolute()
    if ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("상위 이동/심볼릭 링크 경로를 허용하지 않습니다")
    return path.resolve()


def _json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _hashes(path: Path, algorithm: str = "sha256") -> dict[str, str]:
    digests = {name: hashlib.new(name) for name in {"sha256", "md5", algorithm}}
    with path.open("rb") as stream:
        while block := stream.read(CHUNK_BYTES):
            for digest in digests.values():
                digest.update(block)
    return {name: digest.hexdigest() for name, digest in digests.items()}


def _receipt(archive: Path, path: Path) -> tuple[dict, str, dict]:
    if path.stat().st_size > CHUNK_BYTES:
        raise ValueError("staging receipt가 1MiB 상한을 초과했습니다")
    with path.open("rb") as stream:
        raw = stream.read(CHUNK_BYTES + 1)
    if len(raw) > CHUNK_BYTES:
        raise ValueError("staging receipt가 1MiB 상한을 초과했습니다")
    value = json.loads(raw)
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("staging receipt schema_version=1이 필요합니다")
    provenance = validate_staging_provenance(value)
    if value.get("name") != archive.name or not isinstance(value.get("archive_path"), str):
        raise ValueError("receipt archive name/path가 일치해야 합니다")
    if not Path(value["archive_path"]).is_absolute() or _safe_path(value["archive_path"]) != archive:
        raise ValueError("receipt archive_path가 현재 원본과 다릅니다")
    if type(value.get("bytes")) is not int or value["bytes"] <= 0 or value["bytes"] != archive.stat().st_size:
        raise ValueError("receipt 원본 byte 수가 다릅니다")
    algorithm = value.get("source_checksum_algorithm")
    lengths = {"sha256": 64, "sha1": 40, "md5": 32}
    if algorithm not in lengths:
        raise ValueError("receipt source checksum 알고리즘 오류")
    for key, length in (("sha256", 64), ("md5", 32), ("source_checksum", lengths[algorithm])):
        if not isinstance(value.get(key), str) or not re.fullmatch(r"[a-f0-9]{%d}" % length, value[key]):
            raise ValueError(f"receipt {key} 형식 오류")
    if value.get("research_use") != "noncommercial_academic" or not isinstance(value.get("license"), str) or not value["license"].strip():
        raise ValueError("비상업 학업 용도와 license provenance가 필요합니다")
    return value, hashlib.sha256(raw).hexdigest(), provenance


def _member_name(name: str) -> str:
    if not isinstance(name, str) or not name or len(name.encode("utf-8")) > MAX_NAME_BYTES:
        raise ValueError("비어 있거나 너무 긴 archive 항목 이름")
    if "\\" in name or any(ord(char) < 32 or ord(char) == 127 for char in name):
        raise ValueError("archive 경로의 역슬래시/제어문자 거부")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
        raise ValueError("archive 절대/상위 이동/드라이브 경로 거부")
    return path.as_posix().rstrip("/") or "."


def _zip_directory_preflight(path: Path) -> int:
    """ZipFile이 중앙 목록 전체를 읽기 전에 ZIP/ZIP64 크기와 개수를 제한한다."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(max(0, size - 65557))
        tail_start = stream.tell()
        tail = stream.read(65557)
        index = tail.rfind(b"PK\x05\x06")
        if index < 0 or len(tail) - index < 22:
            raise ValueError("ZIP 종료 목록이 없습니다")
        _, disk, cd_disk, disk_count, count, cd_bytes, cd_offset, comment_len = struct.unpack("<4s4H2LH", tail[index:index + 22])
        eocd_position = tail_start + index
        if index + 22 + comment_len != len(tail) or disk or cd_disk or disk_count != count:
            raise ValueError("다중 disk/모호한 ZIP 종료 목록 거부")
        if count == 65535 or cd_bytes == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
            if eocd_position < 20:
                raise ValueError("ZIP64 locator 누락")
            stream.seek(eocd_position - 20)
            signature, locator_disk, zip64_offset, disks = struct.unpack("<4sLQL", stream.read(20))
            if signature != b"PK\x06\x07" or locator_disk or disks != 1:
                raise ValueError("다중 disk/잘못된 ZIP64 locator")
            if zip64_offset + 56 > eocd_position - 20:
                raise ValueError("ZIP64 종료 목록 offset 오류")
            stream.seek(zip64_offset)
            record = stream.read(56)
            if len(record) != 56:
                raise ValueError("ZIP64 종료 목록 잘림")
            signature, record_size, _, _, disk, cd_disk, disk_count, count, cd_bytes, cd_offset = struct.unpack("<4sQ2H2L4Q", record)
            if signature != b"PK\x06\x06" or not 44 <= record_size <= CHUNK_BYTES or disk or cd_disk or disk_count != count:
                raise ValueError("ZIP64 종료 목록 규약 오류")
        if count > MAX_ENTRIES or cd_bytes > MAX_ZIP_DIRECTORY_BYTES:
            raise ValueError("ZIP 항목 수/중앙 목록 메모리 상한 초과")
        if cd_offset + cd_bytes > eocd_position:
            raise ValueError("ZIP 중앙 목록 경계 오류")
        return int(count)


class _BoundedTarInfo(tarfile.TarInfo):
    """TAR parser의 extended-header 통로드/중첩을 제한한다. sparse는 거부한다."""
    @classmethod
    def fromtarfile(cls, archive):
        return cls._fromtarfile(archive)

    @classmethod
    def _fromtarfile(cls, archive, *, dircheck=True):
        buf = archive.fileobj.read(tarfile.BLOCKSIZE)
        if len(buf) != tarfile.BLOCKSIZE:
            raise ValueError("TAR 종결 zero blocks가 없거나 헤더가 잘렸습니다")
        if not any(buf):
            second = archive.fileobj.read(tarfile.BLOCKSIZE)
            if len(second) != tarfile.BLOCKSIZE or any(second):
                raise ValueError("TAR 종결에는 연속된 두 zero blocks가 필요합니다")
            archive._audit_termination_verified = True
            raise tarfile.EOFHeaderError("검증된 TAR 종결")
        try:
            if hasattr(cls, "_frombuf"):
                obj = cls._frombuf(buf, archive.encoding, archive.errors, dircheck=dircheck)
            else:  # 보안 패치 전 Python tarfile의 공개 parser 규약.
                obj = cls.frombuf(buf, archive.encoding, archive.errors)
        except (tarfile.InvalidHeaderError, tarfile.TruncatedHeaderError) as exc:
            raise ValueError("손상된 TAR 헤더") from exc
        obj.offset = archive.fileobj.tell() - tarfile.BLOCKSIZE
        return obj._proc_member(archive)

    def _proc_member(self, archive):
        depth = getattr(archive, "_audit_header_depth", 0)
        if depth >= 32:
            raise ValueError("TAR 확장 헤더 중첩 상한 초과")
        if self.type in (tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK, tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.SOLARIS_XHDTYPE) and self.size > MAX_TAR_HEADER_BYTES:
            raise ValueError("TAR 확장 헤더 메모리 상한 초과")
        if self.type == tarfile.GNUTYPE_SPARSE:
            raise ValueError("sparse TAR 항목은 지원하지 않습니다")
        archive._audit_header_depth = depth + 1
        try:
            result = super()._proc_member(archive)
            for values in (archive.pax_headers, result.pax_headers):
                if len(values) > 4096 or sum(len(key.encode("utf-8", "surrogateescape")) + len(value.encode("utf-8", "surrogateescape")) for key, value in values.items()) > MAX_TAR_HEADER_BYTES:
                    raise ValueError("TAR 누적 PAX metadata 메모리 상한 초과")
            return result
        finally:
            archive._audit_header_depth = depth

    def _proc_gnusparse_00(self, *args):
        raise ValueError("sparse TAR 항목은 지원하지 않습니다")

    _proc_gnusparse_01 = _proc_gnusparse_00
    _proc_gnusparse_10 = _proc_gnusparse_00


class _MemberAudit:
    def __init__(self, stream: BinaryIO, max_audio_bytes: int):
        self.stream = stream
        self.max_audio_bytes = max_audio_bytes
        self.seen: set[str] = set()
        self.name_bytes = 0
        self.counts = {"entries": 0, "audio": 0, "audio_passed": 0, "metadata": 0, "directories": 0, "failed": 0}

    def process(self, name: str, size: int, kind: str, opener, *, extra_issue: str | None = None) -> None:
        self.counts["entries"] += 1
        if self.counts["entries"] > MAX_ENTRIES:
            raise ValueError("archive 항목 수 상한 초과; 이후 목록은 미확인")
        row: dict[str, Any] = {"name": name, "declared_bytes": size, "kind": kind, "issues": []}
        try:
            normalized = _member_name(name)
            row["normalized_name"] = normalized
            if normalized in self.seen:
                raise ValueError("중복 archive 항목 이름(정규화 경로 기준)")
            self.name_bytes += len(normalized.encode("utf-8"))
            if self.name_bytes > MAX_TOTAL_NAME_BYTES:
                raise ValueError("archive 이름 목록 메모리 상한 초과")
            self.seen.add(normalized)
            if extra_issue:
                raise ValueError(extra_issue)
            if kind == "directory":
                self.counts["directories"] += 1
                if size:
                    raise ValueError("directory에 비어 있지 않은 payload가 있습니다")
                row["qa_status"] = "directory"
            elif kind != "file":
                raise ValueError("링크/장치/특수 archive 항목은 읽지 않고 거부합니다")
            else:
                if type(size) is not int or size < 0:
                    raise ValueError("archive 항목 크기 오류")
                audio = PurePosixPath(normalized).suffix.lower() in AUDIO_SUFFIXES
                row["is_audio"] = audio
                self.counts["audio" if audio else "metadata"] += 1
                if audio and size > self.max_audio_bytes:
                    raise ValueError("audio_entry_memory_limit 초과: 디스크로 풀지 않습니다")
                digest = hashlib.sha256()
                decoded_bytes = 0
                buffer = io.BytesIO() if audio else None
                with opener() as member:
                    while block := member.read(CHUNK_BYTES):
                        if not isinstance(block, bytes) or len(block) > CHUNK_BYTES:
                            raise ValueError("archive entry bounded read 규약 오류")
                        decoded_bytes += len(block)
                        if decoded_bytes > size or (audio and decoded_bytes > self.max_audio_bytes):
                            raise ValueError("archive 항목 실제 byte 수/메모리 상한 불일치")
                        digest.update(block)
                        if buffer is not None:
                            buffer.write(block)
                if decoded_bytes != size:
                    raise ValueError("archive 항목 byte 수 잘림")
                row.update(decoded_bytes=decoded_bytes, sha256=digest.hexdigest())
                if buffer is not None:
                    buffer.seek(0)
                    # inspect_audio의 65536-frame block 배열도 채널 상한으로 제한한다.
                    with sf.SoundFile(buffer) as probe:
                        if not 1 <= probe.channels <= MAX_AUDIO_CHANNELS:
                            raise ValueError("audio 채널 메모리 상한 초과")
                        if len(probe) * probe.channels > MAX_PCM_SAMPLES:
                            raise ValueError("audio decoded PCM 표본 수 정책 상한 초과")
                    buffer.seek(0)
                    row["audio"] = inspect_audio(buffer)
                    buffer.close()
                    self.counts["audio_passed"] += 1
                row["qa_status"] = "passed"
        except (ValueError, OSError, RuntimeError, EOFError, zipfile.BadZipFile, zlib.error, NotImplementedError) as exc:
            self.counts["failed"] += 1
            row["qa_status"] = "failed"
            row["issues"].append({"code": "member_qa_failed", "detail": str(exc)})
        self.stream.write(_json(row))
        self.stream.flush()


def _scan_zip(path: Path, auditor: _MemberAudit) -> None:
    expected_count = _zip_directory_preflight(path)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != expected_count:
            raise ValueError("ZIP 중앙 목록 항목 수 불일치")
        for member in members:
            mode = stat.S_IFMT(member.external_attr >> 16)
            if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
                kind = "link_or_special"
            elif member.is_dir() or mode == stat.S_IFDIR:
                kind = "directory"
            elif mode in (0, stat.S_IFREG):
                kind = "file"
            else:
                kind = "link_or_special"
            auditor.process(member.filename, member.file_size, kind,
                            lambda member=member: archive.open(member, "r"),
                            extra_issue="암호화 ZIP 항목은 지원하지 않습니다" if member.flag_bits & 1 else None)


def _scan_tar_gz(path: Path, auditor: _MemberAudit) -> None:
    # 순방향 TAR + 메모리 audio entry: gzip를 매 음원마다 되감는 O(N²) 경로를 피한다.
    with gzip.open(path, "rb") as decompressed:
        with tarfile.open(fileobj=decompressed, mode="r|", tarinfo=_BoundedTarInfo) as archive:
            while (member := archive.next()) is not None:
                kind = "directory" if member.isdir() else "file" if member.isfile() else "link_or_special"
                auditor.process(member.name, member.size, kind,
                                lambda member=member: archive.extractfile(member))
                # 링크를 추적하지 않으므로 과거 TarInfo/PAX를 메모리에 보관할 필요가 없다.
                archive.members.clear()
            if not getattr(archive, "_audit_termination_verified", False):
                raise ValueError("TAR 종결을 검증하지 못했습니다")
            # EOF 뒤 숨겨진 비zero 자료는 거부하며 gzip EOF/CRC도 끝까지 확인한다.
            while block := archive.fileobj.read(CHUNK_BYTES):
                if any(block):
                    raise ValueError("TAR 종료 뒤 비zero 자료가 있습니다")


def audit_public_archive(archive_path: str | Path, staging_receipt: str | Path,
                         out_dir: str | Path, *, max_entry_mib: int = 64) -> dict[str, Any]:
    """검증된 archive를 읽고 새 폴더에 qa.json/inventory.jsonl만 독점 생성한다.

    잘못된 receipt/hash는 항목 접근 전에 거부한다. 순회 중 항목 오류는 보존하며
    archive 구조 오류 이후 미확인 항목은 traversal_complete=false로 구분한다.
    파일 쓰기 실패 시 부분 결과가 남을 수 있다. 자동 재사용/삭제/재전송은 없다.
    """
    archive, receipt_path, out = (_safe_path(path) for path in (archive_path, staging_receipt, out_dir))
    if not archive.is_file() or not receipt_path.is_file():
        raise ValueError("archive와 staging receipt 파일이 필요합니다")
    if out.exists():
        raise FileExistsError("기존 QA 결과 폴더는 덮어쓰지 않습니다")
    if out == archive.parent or out in archive.parents or out in receipt_path.parents:
        raise ValueError("archive/receipt를 포함하는 폴더를 출력으로 사용할 수 없습니다")
    if type(max_entry_mib) is not int or not 1 <= max_entry_mib <= 256:
        raise ValueError("max_entry_mib는 1..256 정수여야 합니다")
    receipt, receipt_hash, provenance = _receipt(archive, receipt_path)
    initial = _hashes(archive, receipt["source_checksum_algorithm"])
    if (initial["sha256"] != receipt["sha256"] or initial["md5"] != receipt["md5"]
            or initial[receipt["source_checksum_algorithm"]] != receipt["source_checksum"]):
        raise ValueError("archive와 staging receipt SHA-256/선언된 checksum 불일치")
    lower = archive.name.lower()
    if not (lower.endswith(".zip") or lower.endswith(".tar.gz") or lower.endswith(".tgz")):
        raise ValueError("ZIP/TAR.GZ/TGZ 원본만 지원합니다")
    out.mkdir(parents=True, exist_ok=False)
    errors = []
    complete = False
    with (out / "inventory.jsonl").open("xb") as inventory:
        auditor = _MemberAudit(inventory, max_entry_mib * CHUNK_BYTES)
        try:
            (_scan_zip if lower.endswith(".zip") else _scan_tar_gz)(archive, auditor)
            complete = True
        except (ValueError, OSError, RuntimeError, EOFError, tarfile.TarError, zipfile.BadZipFile, struct.error, zlib.error) as exc:
            errors.append({"code": "archive_traversal_failed", "detail": str(exc)})
    unchanged = False
    try:
        final = _hashes(archive, receipt["source_checksum_algorithm"])
        unchanged = final == initial and archive.stat().st_size == receipt["bytes"]
        if not unchanged:
            errors.append({"code": "archive_changed_during_audit", "detail": "원본 archive hash/크기가 검사 도중 변경됐습니다"})
        if _hashes(receipt_path)["sha256"] != receipt_hash:
            errors.append({"code": "receipt_changed_during_audit", "detail": "staging receipt가 검사 도중 변경됐습니다"})
    except (ValueError, OSError) as exc:
        errors.append({"code": "input_recheck_failed", "detail": str(exc)})
    counts = auditor.counts
    if counts["audio"] + counts["metadata"] == 0:
        errors.append({"code": "no_regular_files", "detail": "검사할 일반 파일이 없습니다"})
    passed = complete and unchanged and not errors and counts["failed"] == 0 and counts["entries"] > 0
    metadata_only = passed and counts["audio"] == 0 and counts["metadata"] > 0
    report = {
        "schema_version": 1, "diagnostic_only": True, "data_ready": False,
        "training_ready": False, "physical_performance_claim_allowed": False,
        "performance_claim_allowed": False, "drive_upload_verified": False,
        "extraction_performed": False, "local_deletion_performed": False,
        "archive_path": str(archive), "archive_sha256": initial["sha256"],
        "staging_receipt_sha256": receipt_hash, **provenance,
        "source_checksum_algorithm": receipt["source_checksum_algorithm"],
        "source_checksum": receipt["source_checksum"], "license": receipt["license"],
        "source_archive_content_qa_passed": passed,
        "source_archive_content_qa_status": "metadata_only" if metadata_only else "audio_pcm_qa_passed" if passed and counts["audio"] else "failed",
        "audio_pcm_qa_passed": passed and counts["audio"] > 0 and counts["audio"] == counts["audio_passed"],
        "metadata_only": metadata_only, "traversal_complete": complete,
        "archive_unchanged": unchanged, "counts": counts, "issues": errors,
        "inventory_sha256": _hashes(out / "inventory.jsonl")["sha256"],
        "resource_limits": {"max_audio_entry_bytes": max_entry_mib * CHUNK_BYTES,
                            "metadata_read_chunk_bytes": CHUNK_BYTES, "max_entries": MAX_ENTRIES,
                            "max_total_name_bytes": MAX_TOTAL_NAME_BYTES,
                            "max_audio_channels": MAX_AUDIO_CHANNELS, "max_pcm_samples_per_audio": MAX_PCM_SAMPLES},
        "limitations": [
            "학습 group/split/출처 완전성/라이선스 진위/Drive 업로드/실기 감쇠 검증이 아님",
            "출처 검증은 staging receipt의 선언에 의존; Git fsck/upstream 인증을 여기서 다시 실행하지 않음",
            "메타데이터는 내용 SHA와목록만 보존; tracks.csv 등을 의미적으로 파싱하지 않음",
            "중첩 archive는 일반 비오디오 파일로 해시만 검사하며 재귀 추출하지 않음",
            "음원별 메모리/PCM 상한 초과도 실패; 조용한 선별이나 디스크 spill 없음",
            "TAR 확장 헤더는 제한된 parser로 해석한 최종 member 이름을 검사; sparse/링크 거부",
        ],
    }
    with (out / "qa.json").open("xb") as stream:
        stream.write(_json(report))
    return report
