"""Stage-2 public pretrain용 Google Drive partial restore의 fail-closed 감사.

이 모듈은 rclone의 read-only ``cat``/``lsjson``만 사용해 고정 Jetson snapshot을
그 snapshot의 SHA manifest와 대조한다. Drive metadata만으로 decoder, lineage,
source-density 또는 training readiness를 주장하지 않는다. FMA/LibriSpeech/ESC-50
partial restore를 빠르게 검증하기 위한 anchor와 local restore receipt만 만든다.

DNS/DEMAND/MIMII fixed archive cache는 별도 external manifest와 held-fd 검증이
필수다. 그 cache가 없으면 이 snapshot이 완전한 public pretrain bundle로 승격되는
경로는 없다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


STAGE2_DRIVE_AUDIT_SCHEMA = "stage2_drive_public_restore_audit_v1"
STAGE2_DRIVE_ANCHOR_SCHEMA = "stage2_drive_public_restore_anchor_v1"
STAGE2_LOCAL_RESTORE_RECEIPT_SCHEMA = "stage2_drive_local_restore_receipt_v1"

DEFAULT_SNAPSHOT_REMOTE_ROOT = (
    "gdrive:DeepANC/jetson_data_backup_20260827"
)
DEFAULT_ARCHIVE_CACHE_REMOTE_ROOT = "gdrive:DeepANC/public_archive_cache"
SNAPSHOT_MANIFEST_RELATIVE_PATH = "data_backup_manifest.sha256"
EXPECTED_SNAPSHOT_MANIFEST_SHA256 = (
    "1dd9fef8d796cc1f27fbf5d434d640c8b80554e16f04b6bfac0d3403c748bea2"
)
EXPECTED_SNAPSHOT_FILE_COUNT = 13_428
EXPECTED_SNAPSHOT_BYTE_COUNT = 17_439_445_191

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REMOTE = re.compile(r"^[A-Za-z0-9_.-]+:[A-Za-z0-9_.\-/]+$")
_MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  (data/[!-~]+)$")
_ARCHIVE_MANIFEST = re.compile(
    r"(?:^|/)manifests/v1/sha256_([0-9a-f]{64})/archive_cache_manifest\.json$"
)


class Stage2DriveAuditError(ValueError):
    """Drive evidence나 local restore가 exact 계약을 만족하지 않을 때 발생한다."""


@dataclass(frozen=True)
class RestoreCohort:
    name: str
    prefix: str
    expected_file_count: int
    expected_byte_count: int
    required_extension: str
    expected_required_extension_count: int
    expected_required_extension_bytes: int
    stage2_family_role: str


CANONICAL_RESTORE_COHORTS: tuple[RestoreCohort, ...] = (
    RestoreCohort(
        name="fma_music",
        prefix="data/raw/music/",
        expected_file_count=8_003,
        expected_byte_count=8_235_886_703,
        required_extension=".mp3",
        expected_required_extension_count=8_000,
        expected_required_extension_bytes=7_975_016_002,
        stage2_family_role="music",
    ),
    RestoreCohort(
        name="librispeech_speech",
        prefix="data/raw/speech/",
        expected_file_count=2_805,
        expected_byte_count=360_289_905,
        required_extension=".flac",
        expected_required_extension_count=2_703,
        expected_required_extension_bytes=359_034_309,
        stage2_family_role="speech",
    ),
    RestoreCohort(
        name="esc50_environment",
        prefix="data/raw/noise/esc50/",
        expected_file_count=2_011,
        expected_byte_count=884_047_129,
        required_extension=".wav",
        expected_required_extension_count=2_000,
        expected_required_extension_bytes=882_088_000,
        stage2_family_role="environment_diagnostic_not_machine_session_lineage",
    ),
)

LEGACY_MANIFEST_PATHS = (
    "data/manifests/speech.jsonl",
    "data/manifests/music.jsonl",
    "data/manifests/esc50.jsonl",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_remote(value: str, *, label: str) -> str:
    if not _REMOTE.fullmatch(value) or value.endswith(":"):
        raise Stage2DriveAuditError(f"{label}가 안전한 rclone remote:path가 아닙니다")
    _remote, path = value.split(":", 1)
    parts = PurePosixPath(path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise Stage2DriveAuditError(f"{label}에 empty/traversal component가 있습니다")
    return value.rstrip("/")


def _safe_data_path(value: str) -> str:
    if (
        not value
        or "\\" in value
        or unicodedata.normalize("NFC", value) != value
        or not value.startswith("data/")
    ):
        raise Stage2DriveAuditError(f"snapshot path가 canonical data 상대경로가 아닙니다: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise Stage2DriveAuditError(f"snapshot path에 traversal/empty component가 있습니다: {value!r}")
    return path.as_posix()


def parse_snapshot_sha_manifest(content: bytes) -> dict[str, str]:
    """13k-line sha256sum 형식 bytes를 duplicate/traversal 없이 읽는다."""

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Stage2DriveAuditError("snapshot manifest가 UTF-8이 아닙니다") from exc
    if not text.endswith("\n") or "\r" in text:
        raise Stage2DriveAuditError("snapshot manifest newline 형식이 canonical하지 않습니다")
    result: dict[str, str] = {}
    ordered: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = _MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise Stage2DriveAuditError(f"snapshot manifest line 형식 오류: {number}")
        digest, raw_path = match.groups()
        path = _safe_data_path(raw_path)
        if path in result:
            raise Stage2DriveAuditError(f"snapshot manifest path 중복: {path}")
        result[path] = digest
        ordered.append(path)
    if ordered != sorted(ordered):
        raise Stage2DriveAuditError("snapshot manifest path가 bytewise 정렬돼 있지 않습니다")
    return result


def parse_rclone_lsjson(
    content: bytes,
    *,
    data_prefix: str | None,
) -> list[dict[str, Any]]:
    """rclone lsjson output을 PII 없는 path/size/SHA projection으로 줄인다."""

    try:
        rows = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2DriveAuditError("rclone lsjson output이 유효한 UTF-8 JSON이 아닙니다") from exc
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise Stage2DriveAuditError("rclone lsjson 최상위는 object list여야 합니다")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if bool(row.get("IsDir")):
            raise Stage2DriveAuditError(f"--files-only listing에 directory가 있습니다: #{index}")
        raw_path = str(row.get("Path") or "")
        if data_prefix is None:
            if (
                not raw_path
                or "\\" in raw_path
                or PurePosixPath(raw_path).is_absolute()
                or any(part in {"", ".", ".."} for part in PurePosixPath(raw_path).parts)
            ):
                raise Stage2DriveAuditError(f"archive-cache listing path 오류: {raw_path!r}")
            path = PurePosixPath(raw_path).as_posix()
        else:
            path = _safe_data_path(f"{data_prefix.rstrip('/')}/{raw_path}")
        if path in seen:
            raise Stage2DriveAuditError(f"rclone listing path 중복: {path}")
        seen.add(path)
        size = row.get("Size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise Stage2DriveAuditError(f"rclone listing size 오류: {path}")
        hashes = row.get("Hashes")
        if not isinstance(hashes, dict):
            raise Stage2DriveAuditError(f"rclone listing hash가 없습니다: {path}")
        sha256 = str(hashes.get("sha256") or "")
        if not _HEX64.fullmatch(sha256):
            raise Stage2DriveAuditError(f"Drive SHA-256 metadata가 없습니다: {path}")
        normalized.append({"path": path, "size": size, "sha256": sha256})
    return sorted(normalized, key=lambda item: item["path"])


def _projection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [
        {
            "path": str(row["path"]),
            "size": int(row["size"]),
            "sha256": str(row["sha256"]),
        }
        for row in sorted(rows, key=lambda item: str(item["path"]))
    ]
    extensions = Counter(PurePosixPath(row["path"]).suffix.lower() for row in normalized)
    return {
        "file_count": len(normalized),
        "byte_count": sum(row["size"] for row in normalized),
        "extension_counts": dict(sorted(extensions.items())),
        "path_size_sha256_projection_sha256": _digest(normalized),
    }


def build_stage2_drive_audit_from_evidence(
    *,
    snapshot_remote_root: str,
    archive_cache_remote_root: str,
    snapshot_manifest_bytes: bytes,
    snapshot_listing_bytes: bytes,
    archive_cache_listing_bytes: bytes | None,
    archive_cache_query_returncode: int,
    expected_manifest_sha256: str = EXPECTED_SNAPSHOT_MANIFEST_SHA256,
    expected_snapshot_file_count: int = EXPECTED_SNAPSHOT_FILE_COUNT,
    expected_snapshot_byte_count: int = EXPECTED_SNAPSHOT_BYTE_COUNT,
    cohorts: Sequence[RestoreCohort] = CANONICAL_RESTORE_COHORTS,
) -> dict[str, Any]:
    """remote bytes/metadata evidence를 pure fail-closed receipt로 바꾼다."""

    snapshot_root = _safe_remote(snapshot_remote_root, label="snapshot remote root")
    archive_root = _safe_remote(archive_cache_remote_root, label="archive cache remote root")
    if not _HEX64.fullmatch(expected_manifest_sha256):
        raise Stage2DriveAuditError("expected snapshot manifest SHA-256 형식 오류")
    actual_manifest_sha = hashlib.sha256(snapshot_manifest_bytes).hexdigest()
    if actual_manifest_sha != expected_manifest_sha256:
        raise Stage2DriveAuditError("Drive snapshot manifest SHA-256이 external anchor와 다릅니다")
    declared = parse_snapshot_sha_manifest(snapshot_manifest_bytes)
    listed = parse_rclone_lsjson(snapshot_listing_bytes, data_prefix="data")
    listed_by_path = {str(row["path"]): row for row in listed}
    if set(declared) != set(listed_by_path):
        raise Stage2DriveAuditError(
            "Drive snapshot actual object set과 manifest path set이 다릅니다: "
            f"missing={len(set(declared) - set(listed_by_path))}, "
            f"extra={len(set(listed_by_path) - set(declared))}"
        )
    mismatches = [
        path
        for path, expected in declared.items()
        if listed_by_path[path]["sha256"] != expected
    ]
    if mismatches:
        raise Stage2DriveAuditError(
            f"Drive snapshot SHA-256 metadata mismatch가 있습니다: {len(mismatches)}"
        )
    snapshot_projection = _projection(listed)
    if snapshot_projection["file_count"] != int(expected_snapshot_file_count):
        raise Stage2DriveAuditError("Drive snapshot file count가 고정 anchor와 다릅니다")
    if snapshot_projection["byte_count"] != int(expected_snapshot_byte_count):
        raise Stage2DriveAuditError("Drive snapshot byte count가 고정 anchor와 다릅니다")

    cohort_results: list[dict[str, Any]] = []
    restore_rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        prefix = _safe_data_path(cohort.prefix.rstrip("/")) + "/"
        selected = [row for row in listed if str(row["path"]).startswith(prefix)]
        projection = _projection(selected)
        required = [
            row
            for row in selected
            if PurePosixPath(str(row["path"])).suffix.lower()
            == cohort.required_extension
        ]
        required_bytes = sum(int(row["size"]) for row in required)
        passed = (
            projection["file_count"] == cohort.expected_file_count
            and projection["byte_count"] == cohort.expected_byte_count
            and len(required) == cohort.expected_required_extension_count
            and required_bytes == cohort.expected_required_extension_bytes
        )
        if not passed:
            raise Stage2DriveAuditError(f"Drive partial restore cohort가 고정 anchor와 다릅니다: {cohort.name}")
        cohort_results.append(
            {
                "name": cohort.name,
                "prefix": prefix,
                "stage2_family_role": cohort.stage2_family_role,
                **projection,
                "required_extension": cohort.required_extension,
                "required_extension_file_count": len(required),
                "required_extension_byte_count": required_bytes,
                "status": "PASS_EXACT_REMOTE_METADATA_AND_MANIFEST_SHA",
            }
        )
        restore_rows.extend(selected)
    if len({row["path"] for row in restore_rows}) != len(restore_rows):
        raise Stage2DriveAuditError("partial restore cohort prefix가 겹칩니다")
    restore_projection = _projection(restore_rows)

    archive_candidates: list[dict[str, Any]] = []
    archive_file_count = archive_byte_count = 0
    if archive_cache_listing_bytes is not None:
        archive_rows = parse_rclone_lsjson(
            archive_cache_listing_bytes,
            data_prefix=None,
        )
        archive_file_count = len(archive_rows)
        archive_byte_count = sum(int(row["size"]) for row in archive_rows)
        for row in archive_rows:
            match = _ARCHIVE_MANIFEST.search(str(row["path"]))
            if match is not None:
                archive_candidates.append(
                    {
                        "path": row["path"],
                        "path_declared_sha256": match.group(1),
                        "drive_metadata_sha256": row["sha256"],
                        "path_and_metadata_sha256_match": (
                            match.group(1) == row["sha256"]
                        ),
                    }
                )
    archive_present = bool(archive_file_count)
    archive_status = (
        "PRESENT_REMOTE_METADATA_ONLY_NOT_HELD_FD_VERIFIED"
        if archive_present
        else "BLOCKED_ARCHIVE_CACHE_ABSENT"
    )

    legacy_manifests = {
        path: declared.get(path) for path in LEGACY_MANIFEST_PATHS
    }
    if any(value is None for value in legacy_manifests.values()):
        raise Stage2DriveAuditError("snapshot에 legacy speech/music/ESC manifest가 없습니다")
    blockers = [
        "STAGE2_LINEAGE_AND_FREQUENCY_MANIFEST_BUNDLE_NOT_VERIFIED",
        "REMOTE_METADATA_DOES_NOT_VERIFY_DECODER_OR_SOURCE_DENSITY",
    ]
    if not archive_present:
        blockers.extend(
            (
                "DNS_FIXED_ARCHIVES_AND_MANIFEST_ABSENT",
                "DEMAND_FIXED_ARCHIVES_AND_MANIFEST_ABSENT",
                "MIMII_FIXED_ARCHIVE_AND_MANIFEST_ABSENT",
            )
        )
    else:
        blockers.append("ARCHIVE_CACHE_REQUIRES_LOCAL_HELD_FD_CONTENT_VALIDATION")
    payload: dict[str, Any] = {
        "schema": STAGE2_DRIVE_AUDIT_SCHEMA,
        "authority": "read_only_remote_inventory_and_partial_restore_input_only",
        "status": "BLOCKED",
        "public_synthetic_scratch_pretrain_readiness": {
            "status": "BLOCKED",
            "blockers": sorted(blockers),
            "recorded_population_required": False,
        },
        "snapshot": {
            "remote_root": snapshot_root,
            "manifest_relative_path": SNAPSHOT_MANIFEST_RELATIVE_PATH,
            "manifest_sha256": actual_manifest_sha,
            "manifest_line_count": len(declared),
            "actual_object_set_matches_manifest": True,
            "actual_sha256_metadata_matches_manifest": True,
            **snapshot_projection,
            "status": "PASS_EXACT_REMOTE_OBJECT_SET_AND_SHA256_METADATA",
        },
        "partial_restore": {
            "status": "PASS_INPUT_ELIGIBLE_NOT_TRAINING_READY",
            "cohorts": cohort_results,
            **restore_projection,
            "legacy_manifest_sha256": legacy_manifests,
            "requires_local_content_rehash_after_restore": True,
        },
        "official_fixed_archive_cache": {
            "remote_root": archive_root,
            "query_returncode": int(archive_cache_query_returncode),
            "file_count": archive_file_count,
            "byte_count": archive_byte_count,
            "manifest_candidates": archive_candidates,
            "status": archive_status,
        },
        "safety": {
            "remote_operations": ["rclone cat", "rclone lsjson"],
            "remote_write_operations": 0,
            "files_downloaded": 0,
            "audio_opened": False,
            "gpu_initialized": False,
            "elice_contacted": False,
        },
        "limitations": [
            "Drive SHA metadata exact match는 local decoder와 source-density 검증을 대체하지 않습니다",
            "ESC source-file lineage는 MIMII machine/session lineage를 대체하지 않습니다",
            "partial restore PASS는 Stage-2 public scratch-pretrain admission PASS가 아닙니다",
        ],
    }
    payload["evidence_sha256"] = _digest(payload)
    return payload


def _run_rclone(
    executable: Path,
    arguments: Sequence[str],
    *,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(executable), *arguments],
        check=False,
        capture_output=True,
        timeout=timeout_seconds,
    )


def audit_stage2_drive_remote(
    *,
    snapshot_remote_root: str = DEFAULT_SNAPSHOT_REMOTE_ROOT,
    archive_cache_remote_root: str = DEFAULT_ARCHIVE_CACHE_REMOTE_ROOT,
    expected_manifest_sha256: str = EXPECTED_SNAPSHOT_MANIFEST_SHA256,
    rclone_executable: str | Path = "rclone",
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """현재 Drive를 cat/lsjson만으로 읽고 actual audit payload를 만든다."""

    snapshot_root = _safe_remote(snapshot_remote_root, label="snapshot remote root")
    archive_root = _safe_remote(archive_cache_remote_root, label="archive cache remote root")
    if timeout_seconds < 1 or timeout_seconds > 1800:
        raise Stage2DriveAuditError("rclone timeout은 1--1800초여야 합니다")
    raw_executable = str(rclone_executable)
    located = shutil.which(raw_executable) if "/" not in raw_executable else raw_executable
    if not located:
        raise Stage2DriveAuditError("rclone executable을 찾을 수 없습니다")
    executable_candidate = Path(located)
    candidate_node = executable_candidate.lstat()
    if stat.S_ISLNK(candidate_node.st_mode):
        raise Stage2DriveAuditError("rclone executable symlink는 허용하지 않습니다")
    executable = executable_candidate.resolve(strict=True)
    node = executable.stat()
    if not stat.S_ISREG(node.st_mode):
        raise Stage2DriveAuditError("rclone executable은 regular non-symlink여야 합니다")

    manifest_result = _run_rclone(
        executable,
        ["cat", f"{snapshot_root}/{SNAPSHOT_MANIFEST_RELATIVE_PATH}"],
        timeout_seconds=timeout_seconds,
    )
    if manifest_result.returncode != 0:
        raise Stage2DriveAuditError("rclone cat snapshot manifest가 실패했습니다")
    listing_result = _run_rclone(
        executable,
        ["lsjson", f"{snapshot_root}/data", "--recursive", "--files-only", "--hash"],
        timeout_seconds=timeout_seconds,
    )
    if listing_result.returncode != 0:
        raise Stage2DriveAuditError("rclone lsjson snapshot data가 실패했습니다")
    archive_result = _run_rclone(
        executable,
        ["lsjson", archive_root, "--recursive", "--files-only", "--hash"],
        timeout_seconds=timeout_seconds,
    )
    archive_bytes = archive_result.stdout if archive_result.returncode == 0 else None
    return build_stage2_drive_audit_from_evidence(
        snapshot_remote_root=snapshot_root,
        archive_cache_remote_root=archive_root,
        snapshot_manifest_bytes=manifest_result.stdout,
        snapshot_listing_bytes=listing_result.stdout,
        archive_cache_listing_bytes=archive_bytes,
        archive_cache_query_returncode=archive_result.returncode,
        expected_manifest_sha256=expected_manifest_sha256,
    )


def build_stage2_drive_restore_anchor(audit: Mapping[str, Any]) -> dict[str, Any]:
    """큰 remote audit에서 Git에 둘 수 있는 partial-restore anchor만 추린다."""

    if audit.get("schema") != STAGE2_DRIVE_AUDIT_SCHEMA:
        raise Stage2DriveAuditError("Drive audit schema가 다릅니다")
    evidence = str(audit.get("evidence_sha256") or "")
    body = dict(audit)
    body.pop("evidence_sha256", None)
    if not _HEX64.fullmatch(evidence) or _digest(body) != evidence:
        raise Stage2DriveAuditError("Drive audit evidence SHA가 유효하지 않습니다")
    snapshot = audit.get("snapshot")
    restore = audit.get("partial_restore")
    archive = audit.get("official_fixed_archive_cache")
    if not all(isinstance(value, Mapping) for value in (snapshot, restore, archive)):
        raise Stage2DriveAuditError("Drive audit section이 누락됐습니다")
    if snapshot.get("status") != "PASS_EXACT_REMOTE_OBJECT_SET_AND_SHA256_METADATA":
        raise Stage2DriveAuditError("exact snapshot PASS 없이는 anchor를 만들 수 없습니다")
    if restore.get("status") != "PASS_INPUT_ELIGIBLE_NOT_TRAINING_READY":
        raise Stage2DriveAuditError("partial restore projection PASS가 아닙니다")
    anchor: dict[str, Any] = {
        "schema": STAGE2_DRIVE_ANCHOR_SCHEMA,
        "authority": "partial_public_restore_transport_anchor_not_training_authority",
        "source_audit_evidence_sha256": evidence,
        "snapshot_remote_root": snapshot["remote_root"],
        "snapshot_manifest_relative_path": snapshot["manifest_relative_path"],
        "snapshot_manifest_sha256": snapshot["manifest_sha256"],
        "snapshot_file_count": snapshot["file_count"],
        "snapshot_byte_count": snapshot["byte_count"],
        "snapshot_projection_sha256": snapshot[
            "path_size_sha256_projection_sha256"
        ],
        "restore_cohorts": restore["cohorts"],
        "restore_file_count": restore["file_count"],
        "restore_byte_count": restore["byte_count"],
        "restore_projection_sha256": restore[
            "path_size_sha256_projection_sha256"
        ],
        "legacy_manifest_sha256": restore["legacy_manifest_sha256"],
        "fixed_archive_cache_status_at_audit": archive["status"],
        "stage2_public_pretrain_ready": False,
        "recorded_population_required_for_this_restore": False,
    }
    anchor["evidence_sha256"] = _digest(anchor)
    return anchor


def _load_and_validate_anchor(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage2DriveAuditError("restore anchor를 읽을 수 없습니다") from exc
    if not isinstance(payload, dict) or payload.get("schema") != STAGE2_DRIVE_ANCHOR_SCHEMA:
        raise Stage2DriveAuditError("restore anchor schema가 다릅니다")
    evidence = str(payload.get("evidence_sha256") or "")
    body = dict(payload)
    body.pop("evidence_sha256", None)
    if not _HEX64.fullmatch(evidence) or _digest(body) != evidence:
        raise Stage2DriveAuditError("restore anchor evidence SHA가 다릅니다")
    return payload


def verify_local_stage2_partial_restore(
    *,
    anchor_path: str | Path,
    restore_root: str | Path,
    snapshot_manifest_path: str | Path,
) -> dict[str, Any]:
    """Elice local SSD의 partial restore를 manifest content SHA로 전수 검증한다."""

    anchor_candidate = Path(anchor_path)
    if stat.S_ISLNK(anchor_candidate.lstat().st_mode):
        raise Stage2DriveAuditError("restore anchor symlink는 허용하지 않습니다")
    anchor_file = anchor_candidate.resolve(strict=True)
    if not anchor_file.is_file():
        raise Stage2DriveAuditError("restore anchor는 regular non-symlink여야 합니다")
    anchor = _load_and_validate_anchor(anchor_file)
    root_candidate = Path(restore_root)
    if stat.S_ISLNK(root_candidate.lstat().st_mode):
        raise Stage2DriveAuditError("restore root symlink는 허용하지 않습니다")
    root = root_candidate.resolve(strict=True)
    if not root.is_dir():
        raise Stage2DriveAuditError("restore root는 regular directory여야 합니다")
    manifest_candidate = Path(snapshot_manifest_path)
    if stat.S_ISLNK(manifest_candidate.lstat().st_mode):
        raise Stage2DriveAuditError("snapshot manifest symlink는 허용하지 않습니다")
    manifest_path = manifest_candidate.resolve(strict=True)
    if not manifest_path.is_file():
        raise Stage2DriveAuditError("snapshot manifest는 regular non-symlink여야 합니다")
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != anchor["snapshot_manifest_sha256"]:
        raise Stage2DriveAuditError("local snapshot manifest SHA가 anchor와 다릅니다")
    declared = parse_snapshot_sha_manifest(manifest_bytes)

    actual_rows: list[dict[str, Any]] = []
    expected_paths: set[str] = set()
    for raw_cohort in anchor["restore_cohorts"]:
        if not isinstance(raw_cohort, dict):
            raise Stage2DriveAuditError("restore anchor cohort가 object가 아닙니다")
        prefix = _safe_data_path(str(raw_cohort.get("prefix") or "").rstrip("/")) + "/"
        expected_paths.update(path for path in declared if path.startswith(prefix))
        cohort_root = root / prefix.rstrip("/")
        if not cohort_root.is_dir() or cohort_root.is_symlink():
            raise Stage2DriveAuditError(f"local restore cohort가 없습니다: {prefix}")
        for directory, directory_names, filenames in os.walk(cohort_root, followlinks=False):
            base = Path(directory)
            for name in directory_names:
                child = base / name
                if child.is_symlink():
                    raise Stage2DriveAuditError(f"local restore directory symlink 금지: {child}")
            for name in filenames:
                path = base / name
                node = path.lstat()
                if not stat.S_ISREG(node.st_mode) or stat.S_ISLNK(node.st_mode):
                    raise Stage2DriveAuditError(f"local restore regular file 위반: {path}")
                relative = path.relative_to(root).as_posix()
                relative = _safe_data_path(relative)
                actual_rows.append(
                    {
                        "path": relative,
                        "size": node.st_size,
                        "sha256": sha256_file(path),
                    }
                )
    by_path = {row["path"]: row for row in actual_rows}
    if len(by_path) != len(actual_rows) or set(by_path) != expected_paths:
        raise Stage2DriveAuditError(
            "local partial restore path set이 snapshot projection과 다릅니다"
        )
    mismatch = [
        path for path, row in by_path.items() if row["sha256"] != declared[path]
    ]
    if mismatch:
        raise Stage2DriveAuditError(
            f"local partial restore content SHA mismatch가 있습니다: {len(mismatch)}"
        )
    projection = _projection(actual_rows)
    if (
        projection["file_count"] != anchor["restore_file_count"]
        or projection["byte_count"] != anchor["restore_byte_count"]
        or projection["path_size_sha256_projection_sha256"]
        != anchor["restore_projection_sha256"]
    ):
        raise Stage2DriveAuditError("local partial restore aggregate anchor가 다릅니다")
    payload: dict[str, Any] = {
        "schema": STAGE2_LOCAL_RESTORE_RECEIPT_SCHEMA,
        "authority": "local_partial_restore_content_verified_not_training_authority",
        "status": "PASS_PARTIAL_RESTORE_ONLY",
        "anchor_file_sha256": sha256_file(anchor_file),
        "anchor_evidence_sha256": anchor["evidence_sha256"],
        "snapshot_manifest_file_sha256": sha256_file(manifest_path),
        "restore_root": str(root),
        **projection,
        "stage2_public_pretrain_ready": False,
        "remaining_blockers": [
            "DNS_DEMAND_MIMII_FIXED_ARCHIVE_CACHE_OR_OFFICIAL_DOWNLOAD_REQUIRED",
            "STAGE2_LINEAGE_AND_FREQUENCY_MANIFEST_BUNDLE_REQUIRED",
            "LOCAL_DECODER_AND_SOURCE_DENSITY_AUDIT_REQUIRED",
        ],
    }
    payload["evidence_sha256"] = _digest(payload)
    return payload


def write_json_exclusive(payload: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "CANONICAL_RESTORE_COHORTS",
    "DEFAULT_ARCHIVE_CACHE_REMOTE_ROOT",
    "DEFAULT_SNAPSHOT_REMOTE_ROOT",
    "EXPECTED_SNAPSHOT_BYTE_COUNT",
    "EXPECTED_SNAPSHOT_FILE_COUNT",
    "EXPECTED_SNAPSHOT_MANIFEST_SHA256",
    "LEGACY_MANIFEST_PATHS",
    "RestoreCohort",
    "SNAPSHOT_MANIFEST_RELATIVE_PATH",
    "STAGE2_DRIVE_ANCHOR_SCHEMA",
    "STAGE2_DRIVE_AUDIT_SCHEMA",
    "STAGE2_LOCAL_RESTORE_RECEIPT_SCHEMA",
    "Stage2DriveAuditError",
    "audit_stage2_drive_remote",
    "build_stage2_drive_audit_from_evidence",
    "build_stage2_drive_restore_anchor",
    "parse_rclone_lsjson",
    "parse_snapshot_sha_manifest",
    "sha256_file",
    "verify_local_stage2_partial_restore",
    "write_json_exclusive",
]
