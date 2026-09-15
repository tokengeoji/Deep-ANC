"""공식 아카이브의 검증된 HTTPS Range를 최대 네 연결로 임시 staging한다.

CPU Docker와 명시 동의가 필요하다. 기존 파일/부분 다운로드를 재사용하지 않고,
성공해도 부분 파일을 삭제하지 않는다(병합 중 디스크는 원본 약 두 배 필요).
재시도·추출·Drive 업로드·삭제는 하지 않는다. 공식 전체 checksum이 최종 기준이다.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from urllib.request import Request

from .archive_staging import staging_destination
from .drive_transfer import _open_source, validate_source


_CONTENT_RANGE = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)")
_ETAG = re.compile(r'(?:W/)?"[\x21\x23-\x7e\x80-\xff]*"')
_PARTS_DIRECTORY = "range_parts"


def _response_metadata(stream, source, start, end, *, total=None, final_url=None,
                       etag=None, require_same_entity=False):
    resolved = stream.geturl()
    validate_source({**source, "url": resolved})
    if getattr(stream, "status", None) != 206:
        raise ValueError("정확한 HTTP 206 Range 응답이 필요합니다; 자동 재시도하지 않습니다")
    if stream.headers.get("Content-Encoding", "identity").strip().lower() != "identity":
        raise ValueError("Range 응답의 Content-Encoding은 identity여야 합니다")
    ranges = stream.headers.get("Accept-Ranges")
    if ranges is not None and ranges.strip().lower() != "bytes":
        raise ValueError("서버가 byte Range 지원을 명시적으로 확인하지 않았습니다")
    header = stream.headers.get("Content-Range", "")
    match = _CONTENT_RANGE.fullmatch(header) if isinstance(header, str) else None
    if match is None:
        raise ValueError("정확한 Content-Range 시작/끝/전체 크기가 필요합니다")
    actual_start, actual_end, actual_total = map(int, match.groups())
    if (actual_start != start or actual_end != end or actual_total <= end
            or (total is not None and actual_total != total)):
        raise ValueError("요청 범위 또는 전체 크기가 Content-Range와 다릅니다")
    length = stream.headers.get("Content-Length")
    if length is not None and (not isinstance(length, str) or not length.isdigit()
                               or int(length) != end - start + 1):
        raise ValueError("Range Content-Length가 요청 길이와 다릅니다")
    actual_etag = stream.headers.get("ETag")
    if actual_etag is not None and (not isinstance(actual_etag, str) or not _ETAG.fullmatch(actual_etag)):
        raise ValueError("유효한 HTTP entity tag가 필요합니다")
    if require_same_entity and (resolved != final_url or actual_etag != etag):
        raise ValueError("Range 간 최종 출처 또는 ETag가 달라졌습니다")
    return actual_total, resolved, actual_etag


def _read(stream, count):
    block = stream.read(count)
    if not isinstance(block, bytes) or len(block) > count:
        raise ValueError("원본 스트림의 bounded byte 읽기 규약이 잘못되었습니다")
    return block


def _request(url, start, end, etag=None):
    headers = {"Accept-Encoding": "identity", "Range": f"bytes={start}-{end}"}
    # weak ETag는 If-Range에 쓸 수 없다. 응답 간 비교와 전체 checksum은 항상 검사한다.
    if etag is not None and not etag.startswith("W/"):
        headers["If-Range"] = etag
    return Request(url, headers=headers, method="GET")


def stage_public_archive_parallel(source: dict, out, *, confirm_temporary_local_staging=False,
                                  workers=4, chunk_bytes=8 * 1024 * 1024, opener=None,
                                  source_id=None, catalog_sha256=None) -> dict:
    """한 번의 0-0 probe 후 겹치지 않는 Range를 병렬 저장하고 공식 hash로 병합 검증한다.

    실패 시 진행 중 연결도 다음 bounded read 경계에서 중단한다. 서버 응답 대기는
    최대 timeout까지 걸릴 수 있다. 자동 재시도/부분 재사용/자동 삭제는 없다.
    ETag가 없으면 부재를 포함한 응답 일관성과 공식 checksum으로 검증한다.
    """
    if os.environ.get("DEEP_ANC_CONTAINER") != "cpu" or not Path("/.dockerenv").is_file():
        raise ValueError("CPU Docker에서만 임시 staging할 수 있습니다")
    if confirm_temporary_local_staging is not True:
        raise ValueError("--confirm-temporary-local-staging 명시 동의가 필요합니다")
    validate_source(source)
    if source["name"] in {"receipt.json", _PARTS_DIRECTORY}:
        raise ValueError("receipt.json/range_parts는 staging 검증 기록의 예약 이름입니다")
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers는 1..4 범위의 정수여야 합니다")
    if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= 64 * 1024 * 1024:
        raise ValueError("chunk_bytes는 1..64MiB 범위의 정수여야 합니다")
    destination = staging_destination(out)
    open_source = _open_source if opener is None else opener
    with open_source(_request(source["url"], 0, 0), timeout=120) as probe:
        total, final_url, etag = _response_metadata(probe, source, 0, 0)
        if len(_read(probe, 1)) != 1 or _read(probe, 1):
            raise ValueError("0-0 probe 응답 byte 길이가 다릅니다")
    staging_destination(destination)
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    parts_directory = destination / _PARTS_DIRECTORY
    parts_directory.mkdir(mode=0o700)
    count = min(workers, total)
    ranges = [(index, total * index // count, total * (index + 1) // count - 1)
              for index in range(count)]
    stop = threading.Event()

    def download(index, start, end):
        if stop.is_set():
            raise RuntimeError("다른 Range가 실패하여 시작하지 않습니다")
        part = parts_directory / f"part-{index:02d}.bin"
        digest, downloaded = hashlib.sha256(), 0
        try:
            with open_source(_request(source["url"], start, end, etag), timeout=120) as stream:
                _response_metadata(stream, source, start, end, total=total, final_url=final_url,
                                   etag=etag, require_same_entity=True)
                with part.open("xb") as output:
                    while downloaded < end - start + 1:
                        if stop.is_set():
                            raise RuntimeError("다른 Range가 실패하여 부분 파일을 보존하고 중단합니다")
                        block = _read(stream, min(chunk_bytes, end - start + 1 - downloaded))
                        if not block:
                            raise ValueError("Range 응답이 요청 길이보다 짧습니다")
                        if output.write(block) != len(block):
                            raise OSError("Range byte 쓰기가 완료되지 않았습니다")
                        digest.update(block)
                        downloaded += len(block)
                    if _read(stream, 1):
                        raise ValueError("Range 응답이 요청 길이보다 깁니다")
                    output.flush()
                    os.fsync(output.fileno())
                    if os.fstat(output.fileno()).st_size != downloaded:
                        raise ValueError("저장된 Range 크기 검증에 실패했습니다")
            return {"index": index, "start": start, "end_inclusive": end, "bytes": downloaded,
                    "sha256": digest.hexdigest(), "path": str(part)}
        except BaseException:
            stop.set()
            raise

    completed = []
    with ThreadPoolExecutor(max_workers=count, thread_name_prefix="archive-range") as pool:
        futures = [pool.submit(download, *item) for item in ranges]
        try:
            for future in as_completed(futures):
                completed.append(future.result())
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
    completed.sort(key=lambda item: item["index"])
    expected = hashlib.new(source["checksum_algorithm"])
    sha256, md5 = hashlib.sha256(), hashlib.md5()
    downloaded = 0
    archive_path = destination / source["name"]
    with archive_path.open("xb") as output:
        for item in completed:
            path = Path(item["path"])
            if path.is_symlink() or not path.is_file():
                raise ValueError("병합할 부분 파일이 변경되었습니다")
            digest, part_size = hashlib.sha256(), 0
            with path.open("rb") as part:
                while True:
                    block = part.read(chunk_bytes)
                    if not block:
                        break
                    if output.write(block) != len(block):
                        raise OSError("병합 byte 쓰기가 완료되지 않았습니다")
                    digest.update(block)
                    for accumulator in (expected, sha256, md5):
                        accumulator.update(block)
                    part_size += len(block)
                    downloaded += len(block)
                    if part_size > item["bytes"]:
                        raise ValueError("병합할 부분 파일 크기가 증가했습니다")
            if part_size != item["bytes"] or digest.hexdigest() != item["sha256"]:
                raise ValueError("병합할 부분 파일의 크기/hash가 변경되었습니다")
        output.flush()
        os.fsync(output.fileno())
        if os.fstat(output.fileno()).st_size != downloaded or downloaded != total:
            raise ValueError("병합된 원본 전체 크기가 다릅니다")
    if expected.hexdigest() != source["checksum"]:
        raise ValueError("공식 원본 checksum 불일치; 파일을 보존하며 receipt를 발급하지 않습니다")
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
        "transfer_mode": "parallel_verified_ranges", "range_workers": count,
        "source_etag": etag, "range_support_verified_by": "GET bytes=0-0 HTTP 206",
        "range_parts": completed, "range_parts_retained": True,
        "automatic_retry_performed": False,
    }
    encoded = json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with (destination / "receipt.json").open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    return receipt
