"""Drive 메타데이터 관측의 보수적 진단. 음원 다운로드/원격 쓰기는 하지 않는다."""
from __future__ import annotations

import re


def analyze_drive_inventory(value: dict) -> dict:
    """목록 관측과 과거 QA를 실데이터 검증으로 승격하지 않는다."""
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Drive inventory schema_version=1이 필요합니다")
    if value.get("storage_policy") != "drive_only_no_local_raw":
        raise ValueError("원본은 Drive 전용이어야 합니다")
    if value.get("research_use") != "noncommercial_academic":
        raise ValueError("학업용 비상업 연구 용도를 명시해야 합니다")
    files = value.get("files")
    if not isinstance(files, list):
        raise ValueError("files 목록이 필요합니다")
    seen = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("파일 메타데이터는 객체여야 합니다")
        fid, name = item.get("id"), item.get("name")
        if not isinstance(fid, str) or not re.fullmatch(r"[\w-]+", fid, re.ASCII):
            raise ValueError("Drive file ID가 올바르지 않습니다")
        if fid in seen:
            raise ValueError("중복 Drive file ID")
        seen.add(fid)
        if not isinstance(name, str) or not name or any(c in name for c in "\x00\r\n"):
            raise ValueError("파일 이름이 올바르지 않습니다")
        size = item.get("size")
        if size is not None and (type(size) is not int or size < 0):
            raise ValueError("size는 0 이상 정수 또는 null이어야 합니다")
        if not isinstance(item.get("mime_type"), str):
            raise ValueError("mime_type이 필요합니다")

    archive = value.get("multipart_archive")
    report = {"schema_version": 1, "diagnostic_only": True,
              "storage_policy": value["storage_policy"], "local_raw_downloaded": False,
              "training_ready": False, "physical_performance_claim_allowed": False,
              "metadata_file_count": len(files), "archive": None,
              "notes": ["목록과 크기 관측은 파일 내용 hash/PCM 검증을 대신하지 않습니다.",
                        "목록에 안 보이는 파일은 누락 확정이 아니라 미확인입니다.",
                        "과거 QA는 현재 checkout 및 acoustic 학습 준비의 PASS가 아닙니다."]}
    if archive is not None:
        if not isinstance(archive, dict):
            raise ValueError("multipart_archive는 객체여야 합니다")
        count = archive.get("expected_parts")
        size = archive.get("expected_total_bytes")
        prefix = archive.get("prefix")
        if type(count) is not int or not 1 <= count <= 10000:
            raise ValueError("expected_parts가 올바르지 않습니다")
        if type(size) is not int or size <= 0 or not isinstance(prefix, str) or not prefix:
            raise ValueError("아카이브 크기/prefix가 올바르지 않습니다")
        pattern = re.compile(re.escape(prefix) + r"\.part-(\d{4})-of-(\d{4})$")
        parts = {}
        for item in files:
            match = pattern.fullmatch(item["name"])
            if not match:
                continue
            index, total = map(int, match.groups())
            if total != count or not 0 <= index < count:
                raise ValueError("multipart 번호/총개수가 계약과 다릅니다")
            if index in parts:
                raise ValueError("같은 part 번호의 복수 파일: 자동 선택하지 않습니다")
            if item.get("mime_type") == "application/vnd.google-apps.folder" or not item.get("size"):
                raise ValueError("part는 크기가 있는 일반 파일이어야 합니다")
            parts[index] = item
        missing = [i for i in range(count) if i not in parts]
        observed = sum(p["size"] for p in parts.values())
        report["archive"] = {
            "expected_parts": count, "observed_parts": len(parts),
            "unconfirmed_part_indices": missing, "observed_bytes": observed,
            "expected_total_bytes": size, "size_and_count_match": not missing and observed == size,
            "content_hash_verified": False, "restoration_ready": False,
        }
    families = value.get("source_families", [])
    if not isinstance(families, list):
        raise ValueError("source_families는 목록이어야 합니다")
    family_seen = set()
    by_id = {item["id"]: item for item in files}
    for family in families:
        if not isinstance(family, dict) or family.get("name") in family_seen:
            raise ValueError("소스 계열 중복 또는 비정상 메타데이터")
        if family.get("name") not in {"speech", "music", "esc50", "machine", "demand", "dns_fullband"}:
            raise ValueError("알 수 없는 소스 계열")
        family_seen.add(family["name"])
        for key in ("raw_folder_id", "manifest_file_id"):
            if family.get(key) not in seen:
                raise ValueError(f"{key}가 관측된 files에 없습니다")
        raw, manifest = by_id[family["raw_folder_id"]], by_id[family["manifest_file_id"]]
        if raw["mime_type"] != "application/vnd.google-apps.folder":
            raise ValueError("raw_folder_id는 관측된 폴더여야 합니다")
        if manifest["id"] == raw["id"] or manifest["mime_type"] == "application/vnd.google-apps.folder":
            raise ValueError("manifest_file_id는 raw 폴더와 다른 일반 파일이어야 합니다")
        if manifest["name"] != family["name"] + ".jsonl" or not manifest.get("size"):
            raise ValueError("manifest는 계열명.jsonl 형식의 비어 있지 않은 파일이어야 합니다")
    report["source_families_observed"] = sorted(family_seen)
    report["raw_pcm_qa_current"] = "not_performed"
    return report
