"""CPU Docker의 명시적으로 승인된 임시 원본 staging. 업로드/분할/삭제는 하지 않는다.

공식 출처 GET만 수행한다. 공식 checksum 검증 후에만 receipt를 발급하고 실패한
부분 파일은 보존한다. 환경 표시값 검사는 사용자 승인이나 보안 격리를 대체하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from urllib.request import Request

from .drive_transfer import _open_source, validate_source


ESC50_GIT_URL = "https://github.com/karolpiczak/ESC-50.git"
ESC50_APPROVED_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"


def validate_staging_provenance(value: dict) -> dict:
    """receipt 출처 선언을 분류한다. 네트워크/Git/서명 인증을 수행하지 않는다.

    공식 배포 checksum과 공식 Git 객체→로컬 생성 archive hash는 다른 증거다.
    Git variant에서 source_checksum_verified=false를 true로 승격하지 않는다.
    반환은 공유 소비자의 보수적 provenance metadata이며 데이터/PCM READY가 아니다.
    """
    if not isinstance(value, dict):
        raise ValueError("staging provenance는 객체여야 합니다")
    algorithm = value.get("source_checksum_algorithm")
    lengths = {"md5": 32, "sha1": 40, "sha256": 64}
    checksum = value.get("source_checksum")
    if (algorithm not in lengths or not isinstance(checksum, str)
            or not re.fullmatch(r"[a-f0-9]{%d}" % lengths[algorithm], checksum)):
        raise ValueError("staging source checksum 알고리즘/형식 오류")
    checksum_verified = value.get("source_checksum_verified")
    if checksum_verified is True:
        if (value.get("source_verification") not in (None, "official_archive_checksum")
                or value.get("source_content_verified", True) is not True
                or value.get("source_checksum_scope", "official_archive_bytes") != "official_archive_bytes"
                or any(key in value for key in ("git_source_url", "git_commit", "git_fsck_verified"))):
            raise ValueError("공식 archive checksum과 Git 출처 선언이 충돌합니다")
        return {"source_verification": "official_archive_checksum", "source_content_verified": True,
                "source_checksum_verified": True, "source_checksum_scope": "official_archive_bytes",
                "provenance_attestation_only": True}
    if (checksum_verified is not False or value.get("source_content_verified") is not True
            or value.get("source_verification") != "official_git_commit_and_objects"
            or value.get("git_fsck_verified") is not True
            or value.get("git_source_url") != ESC50_GIT_URL
            or value.get("git_commit") != ESC50_APPROVED_COMMIT
            or algorithm != "sha256" or checksum != value.get("sha256")
            or value.get("source_checksum_scope", "locally_generated_archive") != "locally_generated_archive"):
        raise ValueError("검증 완료된 공식 archive checksum 또는 승인된 ESC-50 Git 객체 출처가 필요합니다")
    return {"source_verification": "official_git_commit_and_objects", "source_content_verified": True,
            "source_checksum_verified": False, "source_checksum_scope": "locally_generated_archive",
            "git_source_url": ESC50_GIT_URL, "git_commit": ESC50_APPROVED_COMMIT,
            "git_fsck_verified": True, "provenance_attestation_only": True}


def staging_destination(value) -> Path:
    path = Path(value).expanduser().absolute()
    if ".." in path.parts:
        raise ValueError("임시 staging 경로에 상위 이동(..)을 사용할 수 없습니다")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("임시 staging 경로에 심볼릭 링크를 사용할 수 없습니다")
    repo = Path(__file__).resolve().parents[3]
    broad = {Path("/"), Path("/tmp"), Path("/var/tmp"), Path("/home"), Path("/workspace"),
             Path.home(), repo, Path.cwd()}
    if len(path.parts) < 3 or path in broad:
        raise ValueError("루트/홈/작업공간 등 광범위 경로가 아닌 새 임시 하위 폴더가 필요합니다")
    if path.exists():
        raise FileExistsError("기존 staging 폴더는 재사용하거나 덮어쓰지 않습니다")
    for parent in path.parents:
        if parent.exists() and not parent.is_dir():
            raise NotADirectoryError("staging 상위 경로가 디렉터리가 아닙니다")
    return path


def stage_public_archive(source: dict, out, *, confirm_temporary_local_staging=False,
                         chunk_bytes=8 * 1024 * 1024, opener=None,
                         source_id=None, catalog_sha256=None) -> dict:
    """정확히 한 source GET. EOF/길이/공식 hash 검증 실패 시 자동 재시도하지 않는다."""
    if os.environ.get("DEEP_ANC_CONTAINER") != "cpu" or not Path("/.dockerenv").is_file():
        raise ValueError("CPU Docker에서만 임시 staging할 수 있습니다")
    if confirm_temporary_local_staging is not True:
        raise ValueError("--confirm-temporary-local-staging 명시 동의가 필요합니다")
    validate_source(source)
    if source["name"] == "receipt.json":
        raise ValueError("receipt.json은 staging 검증 기록의 예약 이름입니다")
    if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= 64 * 1024 * 1024:
        raise ValueError("chunk_bytes는 1..64MiB 범위의 정수여야 합니다")
    destination = staging_destination(out)
    request = Request(source["url"], headers={"Accept-Encoding": "identity"}, method="GET")
    open_source = _open_source if opener is None else opener
    expected = hashlib.new(source["checksum_algorithm"])
    sha256, md5 = hashlib.sha256(), hashlib.md5()
    downloaded = 0
    with open_source(request, timeout=120) as stream:
        # 주입 opener를 쓰더라도 최종 출처를 검사한다. 실제 중간 redirect는 _open_source가 검사한다.
        final_url = stream.geturl()
        validate_source({**source, "url": final_url})
        if getattr(stream, "status", None) != 200 or stream.headers.get("Content-Range"):
            raise ValueError("원본 전체 GET(HTTP 200)만 허용하며 자동 재개/Range는 지원하지 않습니다")
        if stream.headers.get("Content-Encoding", "identity").strip().lower() != "identity":
            raise ValueError("압축 HTTP 전송은 원본 byte 검증을 위해 거부합니다")
        length_header = stream.headers.get("Content-Length")
        total = None
        if length_header is not None:
            if not isinstance(length_header, str) or not length_header.isdigit() or int(length_header) <= 0:
                raise ValueError("원본 Content-Length가 올바르지 않습니다")
            total = int(length_header)
        # Content-Length가 없는 출처도 EOF+공식 checksum으로 완료를 검사할 수 있다.
        staging_destination(destination)
        destination.mkdir(parents=True, exist_ok=False, mode=0o700)
        archive_path = destination / source["name"]
        with archive_path.open("xb") as output:
            while True:
                block = stream.read(chunk_bytes)
                if not block:
                    break
                if not isinstance(block, bytes) or len(block) > chunk_bytes:
                    raise ValueError("원본 스트림의 bounded byte 읽기 규약이 잘못되었습니다")
                if output.write(block) != len(block):
                    raise OSError("원본 byte 쓰기가 완료되지 않았습니다")
                for digest in (expected, sha256, md5):
                    digest.update(block)
                downloaded += len(block)
                if total is not None and downloaded > total:
                    raise ValueError("원본이 Content-Length보다 깁니다; 부분 파일을 보존합니다")
            output.flush()
            os.fsync(output.fileno())
            if os.fstat(output.fileno()).st_size != downloaded:
                raise ValueError("저장된 원본 파일 크기 검증에 실패했습니다")
        if downloaded <= 0 or (total is not None and downloaded != total):
            raise ValueError("원본이 비었거나 전송 길이가 다릅니다; 부분 파일을 보존합니다")
        if expected.hexdigest() != source["checksum"]:
            raise ValueError("공식 원본 checksum 불일치; 실패 파일을 보존하며 receipt를 발급하지 않습니다")
    receipt = {
        "schema_version": 1, "name": source["name"], "archive_path": str(archive_path),
        "source_id": source_id, "catalog_sha256": catalog_sha256,
        "source_url": source["url"], "final_source_url": final_url,
        "bytes": downloaded, "size": downloaded, "content_length": total,
        "sha256": sha256.hexdigest(), "md5": md5.hexdigest(),
        "source_checksum_algorithm": source["checksum_algorithm"], "source_checksum": source["checksum"],
        "source_checksum_verified": True, "license": source["license"],
        "official_reference": source.get("official_reference"), "research_use": "noncommercial_academic",
        "storage_policy": "temporary_local_until_verified_drive_upload", "local_raw_written": True,
        "temporary_local_staging": True, "drive_upload_verified": False,
        "delete_after_verified_upload_required": True, "local_deletion_performed": False,
        "pcm_qa_verified": False, "extraction_performed": False,
    }
    encoded = json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with (destination / "receipt.json").open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    return receipt
