"""검증된 임시 원본의 작은 조각 생성/읽기 전용 검증/명시적인 새 파일 복원.

네트워크·업로드·삭제는 하지 않는다. 실패한 새 조각/복원 부분 파일은 보존한다.
공식/로컬 checksum은 staging receipt의 출처 구분을 유지하고 웹에서 다시 조회하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

from .archive_staging import staging_destination, validate_staging_provenance
from .drive_transfer import validate_source

MIB = 1024 * 1024
MAX_PART_BYTES_EXCLUSIVE = 100 * MIB
SCHEMA = "staged_archive_parts.v1"


def _file(path) -> Path:
    result = Path(path).expanduser().absolute()
    if ".." in result.parts or any(p.is_symlink() for p in (result, *result.parents)):
        raise ValueError("입력의 상위 이동/심볼릭 링크는 허용하지 않습니다")
    if not result.is_file():
        raise FileNotFoundError(f"명시한 일반 파일이 필요합니다: {result}")
    return result


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: {minimum} 이상 정수가 필요합니다")
    return value


def _chunk_size(value):
    if _integer(value, "chunk_bytes") > 64 * MIB:
        raise ValueError("메모리 청크는 64MiB 이하여야 합니다")
    return value


def _source(archive):
    provenance = validate_staging_provenance(archive)
    if provenance["source_verification"] == "official_archive_checksum":
        validate_source({"name": archive["name"], "url": archive["source_url"],
                         "checksum_algorithm": archive["source_checksum_algorithm"],
                         "checksum": archive["source_checksum"], "license": archive["license"]})
    elif (not isinstance(archive.get("name"), str)
          or not re.fullmatch(r"[A-Za-z0-9_.-]+", archive["name"]) or archive["name"] in {".", ".."}
          or not isinstance(archive.get("license"), str) or not archive["license"].strip()
          or archive.get("source_url") not in {provenance["git_source_url"], provenance["git_source_url"].removesuffix(".git")}):
        raise ValueError("승인된 Git 원본의 이름/URL/라이선스가 잘못되었습니다")
    size = _integer(archive["bytes"], "archive bytes")
    if _integer(archive["size"], "archive size") != size:
        raise ValueError("원본 bytes/size 메타가 다릅니다")
    for name, length in (("sha256", 64), ("md5", 32)):
        if not isinstance(archive.get(name), str) or not re.fullmatch(f"[a-f0-9]{{{length}}}", archive[name]):
            raise ValueError(f"원본 {name} 형식이 잘못되었습니다")
    return size


def _digests(algorithm):
    return {name: hashlib.new(name) for name in {"sha256", "md5", algorithm}}


def _matches(archive, size, digests):
    if size != archive["bytes"]:
        raise ValueError("원본/결합 byte 수가 staging receipt와 다릅니다")
    wanted = {"sha256": archive["sha256"], "md5": archive["md5"]}
    official = archive["source_checksum_algorithm"]
    if digests[official].hexdigest() != archive["source_checksum"]:
        raise ValueError("원본/결합의 출처에 선언된 checksum이 다릅니다")
    if any(digests[name].hexdigest() != value for name, value in wanted.items()):
        raise ValueError("원본/결합 SHA-256 또는 MD5가 다릅니다")


def _part_name(name, index, count):
    return f"{name}.part-{index:04d}-of-{count:04d}"


def split_staged_archive(staging_receipt, out, *, part_mib=96, chunk_bytes=MIB):
    """원본 전체를 먼저 검사하고 1..99MiB 조각을 새 폴더에 만든다."""
    _integer(part_mib, "part_mib")
    part_bytes = part_mib * MIB
    if part_bytes >= MAX_PART_BYTES_EXCLUSIVE:
        raise ValueError("각 조각은 100MiB 미만이어야 합니다(기본 96MiB)")
    _chunk_size(chunk_bytes)
    destination = staging_destination(out)
    receipt_path = _file(staging_receipt)
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    if receipt.get("schema_version") != 1 or receipt.get("temporary_local_staging") is not True:
        raise ValueError("임시 staging receipt v1이 필요합니다")
    size = _source(receipt)
    archive_path = _file(receipt["archive_path"])
    if archive_path != receipt_path.parent / receipt["name"]:
        raise ValueError("원본 name/path가 staging receipt 폴더와 일치하지 않습니다")
    if archive_path.stat().st_size != size:
        raise ValueError("원본 파일 크기가 receipt와 다릅니다")
    count = (size + part_bytes - 1) // part_bytes
    if count > 9999:
        raise ValueError("4자리 조각 번호의 최대 개수를 넘었습니다")
    original_hashes = _digests(receipt["source_checksum_algorithm"])
    original_size = 0
    with archive_path.open("rb") as original:
        while block := original.read(chunk_bytes):
            original_size += len(block)
            for digest in original_hashes.values():
                digest.update(block)
    _matches(receipt, original_size, original_hashes)  # 원본 불일치면 출력 폴더조차 만들지 않는다.
    staging_destination(destination)
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    combined = _digests(receipt["source_checksum_algorithm"])
    written = 0
    parts = []
    with archive_path.open("rb") as original:
        for index in range(count):
            name = _part_name(receipt["name"], index, count)
            needed = min(part_bytes, size - written)
            part_hashes = _digests("sha256")
            part_size = 0
            with (destination / name).open("xb") as output:
                while part_size < needed:
                    block = original.read(min(chunk_bytes, needed - part_size))
                    if not block:
                        raise ValueError("분할 중 원본이 짧아졌습니다; 부분 조각을 보존합니다")
                    if output.write(block) != len(block):
                        raise OSError("조각 쓰기가 완료되지 않았습니다")
                    for digest in (*part_hashes.values(), *combined.values()):
                        digest.update(block)
                    part_size += len(block)
                    written += len(block)
                output.flush()
                os.fsync(output.fileno())
                if os.fstat(output.fileno()).st_size != part_size:
                    raise ValueError("조각 파일 크기 불일치; 부분 조각을 보존합니다")
            parts.append({"index": index, "name": name, "size": part_size,
                          "sha256": part_hashes["sha256"].hexdigest(), "md5": part_hashes["md5"].hexdigest()})
        if original.read(1):
            raise ValueError("분할 중 원본 크기가 늘었습니다; 부분 조각을 보존합니다")
    _matches(receipt, written, combined)
    archive = {key: receipt[key] for key in ("name", "bytes", "size", "sha256", "md5", "source_url",
                                            "source_checksum_algorithm", "source_checksum",
                                            "source_checksum_verified", "license")}
    archive.update(validate_staging_provenance(receipt))
    manifest = {"schema": SCHEMA, "archive": archive, "part_count": count,
                "part_size_bytes": part_bytes, "max_part_bytes_exclusive": MAX_PART_BYTES_EXCLUSIVE,
                "parts": parts, "ordering": "zero_based_index_in_manifest_order",
                "staging_receipt_path": str(receipt_path),
                "staging_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "local_parts_integrity_verified": True, "drive_upload_verified": False,
                "ready": False, "local_deletion_performed": False, "research_use": "noncommercial_academic",
                "note": "로컬 조각 작성 byte/크기/hash 검증만 완료. Drive 부분/전체 업로드 완료는 별도 검증 필요."}
    with (destination / "manifest.json").open("x", encoding="utf-8") as output:
        output.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return manifest


def verify_archive_parts(parts_manifest, *, chunk_bytes=MIB):
    """명시 목록을 모두 다시 읽어 검증한다. 복원 파일·receipt 파일을 쓰지 않는다."""
    _chunk_size(chunk_bytes)
    path = _file(parts_manifest)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("지원하는 조각 manifest가 아닙니다")
    archive = manifest["archive"]
    size = _source(archive)
    count = _integer(manifest["part_count"], "part_count")
    part_bytes = _integer(manifest["part_size_bytes"], "part_size_bytes")
    if (count > 9999 or part_bytes >= MAX_PART_BYTES_EXCLUSIVE
            or manifest.get("max_part_bytes_exclusive") != MAX_PART_BYTES_EXCLUSIVE
            or count != (size + part_bytes - 1) // part_bytes
            or len(manifest["parts"]) != count):
        raise ValueError("조각 개수/최대 크기/전체 크기 규약이 다릅니다")
    names = [_part_name(archive["name"], index, count) for index in range(count)]
    if {item.name for item in path.parent.iterdir()} != {path.name, *names}:
        raise ValueError("manifest 외 조각 파일의 정확한 개수/이름이 다릅니다(누락 또는 추가 파일)")
    combined = _digests(archive["source_checksum_algorithm"])
    combined_size = 0
    for index, part in enumerate(manifest["parts"]):
        if _integer(part["index"], "part index", 0) != index or part["name"] != names[index]:
            raise ValueError("조각 순서/번호/이름이 다릅니다")
        expected_size = min(part_bytes, size - combined_size)
        if _integer(part["size"], "part size") != expected_size:
            raise ValueError("조각의 지정 크기가 전체 순서와 다릅니다")
        part_path = _file(path.parent / part["name"])
        if part_path.stat().st_size != expected_size:
            raise ValueError("조각 파일 크기가 다릅니다")
        digests = _digests("sha256")
        consumed = 0
        with part_path.open("rb") as stream:
            while block := stream.read(chunk_bytes):
                consumed += len(block)
                for digest in (*digests.values(), *combined.values()):
                    digest.update(block)
        if consumed != expected_size or any(digests[key].hexdigest() != part[key] for key in ("sha256", "md5")):
            raise ValueError("조각 byte/SHA-256/MD5 불일치")
        combined_size += consumed
    _matches(archive, combined_size, combined)
    return {"schema": "staged_archive_parts_verification.v1", "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "name": archive["name"], "bytes": combined_size, "part_count": count,
            "sha256": combined["sha256"].hexdigest(), "md5": combined["md5"].hexdigest(),
            **validate_staging_provenance(archive), "local_parts_integrity_verified": True,
            "drive_upload_verified": False, "ready": False, "restored_file_written": False,
            "local_deletion_performed": False, "license": archive["license"]}


def _stat_identity(status):
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


def _file_identity(path):
    return _stat_identity(_file(path).stat())


def restore_archive_parts(parts_manifest, archive_out, *, chunk_bytes=MIB):
    """전체 조각을 먼저 검증한 뒤 별도 새 파일에 복원하고 출력도 다시 읽어 검사한다.

    결합 중 조각별/전체 checksum, manifest/입력 파일 변경을 다시 검사한다.
    실패 시 이미 만든 부분 파일은 보존하며 성공 report를 반환하지 않는다.
    Drive 다운로드/검증/삭제는 없고, 복원 성공도 데이터/PCM READY가 아니다.
    """
    _chunk_size(chunk_bytes)
    path = _file(parts_manifest)
    destination = staging_destination(archive_out)
    if destination.is_relative_to(path.parent):
        raise ValueError("복원 출력은 원본 parts 폴더 밖의 새 파일이어야 합니다")
    verified = verify_archive_parts(path, chunk_bytes=chunk_bytes)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != verified["manifest_sha256"]:
        raise ValueError("사전 검증 후 manifest가 변경되었습니다")
    manifest = json.loads(raw)
    archive = manifest["archive"]
    expected_size = archive["bytes"]
    inputs = [(part, _file(path.parent / part["name"])) for part in manifest["parts"]]
    identities = {part_path: _file_identity(part_path) for _, part_path in inputs}
    combined = _digests(archive["source_checksum_algorithm"])
    written = 0
    # 검증 동안 출력 경로가 바뀌었는지도 검사하고, exclusive open으로 기존 파일을 보호한다.
    staging_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_destination(destination)
    with destination.open("x+b") as output:
        for part, part_path in inputs:
            if _file_identity(part_path) != identities[part_path]:
                raise ValueError("복원 전 조각이 변경되었습니다; 부분 출력을 보존합니다")
            digests = _digests("sha256")
            consumed = 0
            with part_path.open("rb") as stream:
                while block := stream.read(min(chunk_bytes, part["size"] - consumed + 1)):
                    if consumed + len(block) > part["size"]:
                        raise ValueError("복원 중 조각 크기가 늘었습니다; 부분 출력을 보존합니다")
                    if output.write(block) != len(block):
                        raise OSError("복원 byte 쓰기가 완료되지 않았습니다")
                    for digest in (*digests.values(), *combined.values()):
                        digest.update(block)
                    consumed += len(block)
                    written += len(block)
            if (consumed != part["size"]
                    or any(digests[key].hexdigest() != part[key] for key in ("sha256", "md5"))
                    or _file_identity(part_path) != identities[part_path]):
                raise ValueError("복원 중 조각 byte/hash/상태 변경; 부분 출력을 보존합니다")
        _matches(archive, written, combined)
        output.flush()
        os.fsync(output.fileno())
        # 이미 읽은 구간의 동일 inode/동일 크기 덮어쓰기도 탐지하도록 검증 전 상태를 고정한다.
        output_identity = _stat_identity(os.fstat(output.fileno()))
        if output_identity[2] != expected_size:
            raise ValueError("복원 출력 파일 크기가 다릅니다; 부분 출력을 보존합니다")
        output.seek(0)
        readback = _digests(archive["source_checksum_algorithm"])
        readback_size = 0
        while block := output.read(chunk_bytes):
            readback_size += len(block)
            for digest in readback.values():
                digest.update(block)
        _matches(archive, readback_size, readback)
        if (_file(path).read_bytes() != raw
                or {item.name for item in path.parent.iterdir()} != {path.name, *(part["name"] for part, _ in inputs)}
                or any(_file_identity(part_path) != identities[part_path] for _, part_path in inputs)):
            raise ValueError("복원 중 manifest/조각 목록/입력이 변경되었습니다; 출력을 보존합니다")
        if (_stat_identity(os.fstat(output.fileno())) != output_identity
                or _file_identity(destination) != output_identity):
            raise ValueError("복원 출력 경로/파일 상태가 변경되었습니다; 성공으로 처리하지 않습니다")
    return {**verified, "schema": "staged_archive_parts_restore.v1", "archive_path": str(destination),
            "manifest_path": str(path), "size": written, "restored_file_written": True,
            "restored_file_integrity_verified": True, "output_readback_verified": True,
            "source_url": archive["source_url"], "source_checksum_algorithm": archive["source_checksum_algorithm"],
            "source_checksum": archive["source_checksum"], "temporary_local_staging": True,
            "pcm_qa_verified": False, "extraction_performed": False,
            "note": "로컬 복원 byte/hash 검증만 완료. Drive 검증·PCM QA·데이터 READY는 별도입니다."}
