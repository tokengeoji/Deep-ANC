"""공식 아카이브 → 메모리 청크 → Drive. 원본을 디스크에 저장하지 않는다.

실행은 사용자가 승인하고 인증한 전송 환경에서만 한다. 연결 플러그인의 인증을
추출하거나 공유 설정/기존 파일을 변경하지 않는다. 중단 시 자동 재개하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

ALLOWED_SOURCE_HOSTS = {"www.openslr.org", "openslr.org", "us.openslr.org", "openslr.elda.org",
                        "os.unil.cloud.switch.ch", "zenodo.org"}
API = "https://www.googleapis.com/drive/v3/files"


class _NoAuthenticatedRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("인증된 Drive 요청은 redirect를 따라가지 않습니다")


def _validate_source_url(address: str) -> None:
    url = urlparse(address)
    if url.scheme != "https" or url.hostname not in ALLOWED_SOURCE_HOSTS or url.username or url.password:
        raise ValueError("허용된 공식 HTTPS 음원 출처/redirect만 사용할 수 있습니다")
    if url.port not in (None, 443) or url.fragment:
        raise ValueError("출처/redirect URL의 포트/fragment가 올바르지 않습니다")


class _ValidatedSourceRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # 중간 hop도 연결 전에 검사한다. 최종 주소만 검사하면 충분하지 않다.
        _validate_source_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_source(request, *, timeout):
    _validate_source_url(request.full_url)
    return build_opener(_ValidatedSourceRedirect()).open(request, timeout=timeout)


def validate_source(source: dict) -> None:
    _validate_source_url(source.get("url", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", source.get("name", "")) or source["name"] in {".", ".."}:
        raise ValueError("안전한 아카이브 파일 이름이 필요합니다")
    algorithm = source.get("checksum_algorithm")
    lengths = {"md5": 32, "sha1": 40, "sha256": 64}
    if algorithm not in lengths or not re.fullmatch(r"[a-f0-9]{%d}" % lengths[algorithm], source.get("checksum", "")):
        raise ValueError("공식 배포 체크섬이 필요합니다")
    if not isinstance(source.get("license"), str) or not source["license"]:
        raise ValueError("라이선스 메타데이터가 필요합니다")


class GoogleDriveUploadAPI:
    """인증 callback은 토큰을 메모리로만 전달한다. URL/HTTP 본문을 오류에 노출하지 않는다."""
    def __init__(self, token_provider):
        self.token_provider = token_provider

    def _request(self, method, url, body=None, headers=None):
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.netloc != "www.googleapis.com":
            raise ValueError("Drive 이외의 주소에 인증을 전달하지 않습니다")
        h = {"Authorization": "Bearer " + self.token_provider(), **(headers or {})}
        try:
            response = build_opener(_NoAuthenticatedRedirect()).open(
                Request(url, data=body, headers=h, method=method), timeout=120)
        except Exception as exc:
            # 308은 urllib HTTPError로 전달되며 정상적인 중간 청크 ACK다.
            if getattr(exc, "code", None) == 308:
                return 308, dict(exc.headers), {}
            raise RuntimeError(f"Drive 요청 실패 (HTTP {getattr(exc, 'code', 'unknown')}); 자동 재개하지 않습니다") from None
        with response:
            data = response.read()
            return response.status, dict(response.headers), json.loads(data) if data else {}

    def check_destination(self, folder_id, name):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", folder_id):
            raise ValueError("대상 Drive 폴더 ID가 필요합니다")
        _, _, folder = self._request("GET", API + "/" + folder_id + "?" + urlencode(
            {"fields": "id,mimeType,trashed,capabilities(canAddChildren)", "supportsAllDrives": "true"}))
        if folder.get("mimeType") != "application/vnd.google-apps.folder" or folder.get("trashed") or not folder.get("capabilities", {}).get("canAddChildren"):
            raise ValueError("쓰기가 가능한 대상 폴더가 아닙니다")
        query = f"'{folder_id}' in parents and name = '{name}' and trashed = false"
        _, _, matches = self._request("GET", API + "?" + urlencode(
            {"q": query, "fields": "files(id,name),nextPageToken", "pageSize": 100,
             "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"}))
        if matches.get("files") or matches.get("nextPageToken"):
            raise FileExistsError("대상 이름이 이미 있습니다. 기존 파일을 검증/재사용하고 중복 업로드하지 마세요")

    def initiate(self, source, folder_id, total):
        metadata = {"name": source["name"], "parents": [folder_id],
                    "description": "Deep-ANC 비상업 학업 실험 원본; 출처 " + source["url"],
                    "appProperties": {"source_checksum": source["checksum"],
                                      "source_checksum_algorithm": source["checksum_algorithm"]}}
        _, headers, _ = self._request("POST", "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&supportsAllDrives=true",
            json.dumps(metadata).encode(), {"Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": "application/octet-stream", "X-Upload-Content-Length": str(total)})
        location = headers.get("Location") or headers.get("location")
        if not location:
            raise RuntimeError("Drive 업로드 세션 주소가 없습니다")
        if urlparse(location).scheme != "https" or urlparse(location).netloc != "www.googleapis.com":
            raise ValueError("Drive 세션 주소가 올바르지 않습니다")
        return location

    def put(self, session, chunk, start, total):
        return self._request("PUT", session, chunk, {"Content-Type": "application/octet-stream",
            "Content-Length": str(len(chunk)), "Content-Range": f"bytes {start}-{start + len(chunk) - 1}/{total}"})

    def metadata(self, file_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
            raise ValueError("완료된 파일 ID가 올바르지 않습니다")
        return self._request("GET", API + "/" + file_id + "?" + urlencode(
            {"fields": "id,name,size,md5Checksum,webViewLink,parents", "supportsAllDrives": "true"}))[2]


def stream_archive_to_drive(source, folder_id, api, *, opener=_open_source, chunk_bytes=8 * 1024 * 1024):
    """체크섬 확인 전 마지막 청크를 보내지 않아 잘못된 원본을 완료 처리하지 않는다.

    HTTP 실패/모호한 ACK는 즉시 중단한다. 진행 중 세션은 Drive에 남을 수 있지만
    기존 파일을 지우거나 덮어쓰지 않는다. 중단된 세션의 복구/만료 확인은 별도다.
    """
    validate_source(source)
    if type(chunk_bytes) is not int or chunk_bytes < 256 * 1024 or chunk_bytes % (256 * 1024):
        raise ValueError("청크는 256KiB 이상의 배수여야 합니다")
    api.check_destination(folder_id, source["name"])
    request = Request(source["url"], headers={"Accept-Encoding": "identity"})
    with opener(request, timeout=120) as stream:
        _validate_source_url(stream.geturl())
        if stream.headers.get("Content-Encoding", "identity") != "identity":
            raise ValueError("압축 HTTP 전송은 크기 검증을 할 수 없습니다")
        try:
            total = int(stream.headers["Content-Length"])
        except (KeyError, ValueError, TypeError):
            raise ValueError("원본 Content-Length가 필요합니다") from None
        if total <= 0:
            raise ValueError("빈 원본은 업로드하지 않습니다")
        session = api.initiate(source, folder_id, total)
        expected = hashlib.new(source["checksum_algorithm"])
        md5, sha256 = hashlib.md5(), hashlib.sha256()
        offset, completed = 0, None
        while offset < total:
            need = min(chunk_bytes, total - offset)
            chunk = bytearray()
            while len(chunk) < need:
                piece = stream.read(need - len(chunk))
                if not piece:
                    raise ValueError("원본 전송이 중단됐습니다; 완료 청크를 보내지 않습니다")
                chunk.extend(piece)
            for digest in (expected, md5, sha256):
                digest.update(chunk)
            last = offset + len(chunk) == total
            if last and (stream.read(1) or expected.hexdigest() != source["checksum"]):
                raise ValueError("공식 원본 크기/체크섬 불일치; 마지막 청크를 보류합니다")
            status, headers, payload = api.put(session, bytes(chunk), offset, total)
            if last:
                if status not in (200, 201) or not payload.get("id"):
                    raise RuntimeError("Drive 완료 응답을 확인하지 못했습니다")
                completed = payload["id"]
            elif status != 308 or (headers.get("Range") or headers.get("range")) != f"bytes=0-{offset + len(chunk) - 1}":
                raise RuntimeError("Drive ACK 범위 불일치; 자동 재전송하지 않습니다")
            offset += len(chunk)
        meta = api.metadata(completed)
        if int(meta.get("size", -1)) != total or meta.get("md5Checksum") != md5.hexdigest() or meta.get("name") != source["name"] or folder_id not in meta.get("parents", []):
            raise ValueError("완료된 Drive 파일의 크기/hash/이름/폴더 검증 실패; 파일은 보존합니다")
        return {"schema_version": 1, "file_id": completed, "web_view_link": meta.get("webViewLink"),
                "name": source["name"], "size": total, "sha256": sha256.hexdigest(),
                "md5": md5.hexdigest(), "source_checksum_verified": True,
                "drive_checksum_verified": True, "local_raw_written": False,
                "license": source["license"], "research_use": "noncommercial_academic",
                "source_url": source["url"], "pcm_qa_verified": False}
