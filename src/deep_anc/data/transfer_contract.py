"""Jetson→Elice immutable 학습 입력 bundle 계약 (표준 라이브러리 전용)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .holdout_contract import (
    FileSnapshot,
    HoldoutContractError,
    read_regular_file_snapshot,
    reject_symlink_components,
    validate_holdout_contract,
)


TRANSFER_SCHEMA_VERSION = 1
EXPECTED_RECORDED_SESSIONS = 82
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ROLES = frozenset(
    {
        "recorded",
        "rir_bank",
        "strict_ps_raw",
        "strict_ps_analysis",
        "strict_primary_npz",
        "strict_secondary_npz",
        "recorded_manifest",
        "lineage_tracks",
        "librispeech_chapters_metadata",
        "esc50_metadata",
        "canonical_holdout",
        "source_pool_v1_csv",
        "source_pool_v2_csv",
        "provenance_report",
    }
)
_ROLE_PREFIXES = {
    "recorded": "data/recorded/",
    "rir_bank": "data/rir_bank/",
    "strict_ps_raw": "results/",
    "strict_ps_analysis": "results/",
    "strict_primary_npz": "assets/measured/",
    "strict_secondary_npz": "assets/measured/",
    "recorded_manifest": "data/manifests/",
    "lineage_tracks": "data/raw/music/fma_metadata/",
    "librispeech_chapters_metadata": "data/raw/speech/LibriSpeech/",
    "esc50_metadata": "data/raw/noise/esc50/ESC-50-master/meta/",
    "canonical_holdout": "data/manifests/",
    "source_pool_v1_csv": "data/source_pool/",
    "source_pool_v2_csv": "data/source_pool_v2/",
    "provenance_report": "results/provenance/",
}


class TransferContractError(ValueError):
    """전송 manifest 또는 그 파일 집합이 학습 입력 계약을 위반했다."""


@dataclass(frozen=True)
class RecordedTrainingSnapshot:
    """학습 loader가 소비하는 검증된 transfer generation의 recorded 부분."""

    repo_root: Path
    bootstrap_receipt: FileSnapshot | None
    transfer_manifest: FileSnapshot
    recorded_manifest: FileSnapshot
    recorded_aggregate_sha256: str
    recorded_files: dict[str, FileSnapshot]

    def _relative_recorded_path(self, path: str | Path) -> str:
        candidate = Path(os.path.abspath(os.fspath(path)))
        try:
            relative = candidate.relative_to(self.repo_root).as_posix()
        except ValueError as exc:
            raise TransferContractError(
                f"recorded loader path가 transfer repository 밖입니다: {candidate}"
            ) from exc
        if relative not in self.recorded_files:
            raise TransferContractError(
                f"recorded loader path가 검증된 transfer exact 집합에 없습니다: {relative}"
            )
        return relative

    def has_recorded_file(self, path: str | Path) -> bool:
        candidate = Path(os.path.abspath(os.fspath(path)))
        try:
            relative = candidate.relative_to(self.repo_root).as_posix()
        except ValueError:
            return False
        return relative in self.recorded_files

    @staticmethod
    def _same_stat(left: FileSnapshot, right: FileSnapshot) -> bool:
        return (
            left.device,
            left.inode,
            left.size,
            left.mtime_ns,
            left.ctime_ns,
        ) == (
            right.device,
            right.inode,
            right.size,
            right.mtime_ns,
            right.ctime_ns,
        )

    def read_verified_recorded_file(self, path: str | Path) -> bytes:
        """transfer 검증과 같은 inode/stat/content인 한 FD bytes만 반환한다."""

        relative = self._relative_recorded_path(path)
        expected = self.recorded_files[relative]
        try:
            current = read_regular_file_snapshot(
                self.repo_root / relative,
                root=self.repo_root,
                label=f"recorded training input {relative}",
            )
        except HoldoutContractError as exc:
            raise TransferContractError(str(exc)) from exc
        if (
            current.sha256 != expected.sha256
            or current.size != expected.size
            or not self._same_stat(current, expected)
        ):
            raise TransferContractError(
                "recorded training input이 검증된 transfer snapshot 이후 변경됐습니다: "
                f"{relative}"
            )
        assert current.data is not None
        return current.data

    def assert_recorded_file_unchanged(self, path: str | Path) -> None:
        """이미 decode해 cache한 파일도 매 사용 전 inode/stat 변화로 차단한다."""

        relative = self._relative_recorded_path(path)
        expected = self.recorded_files[relative]
        try:
            candidate = reject_symlink_components(
                self.repo_root / relative,
                root=self.repo_root,
            )
        except HoldoutContractError as exc:
            raise TransferContractError(str(exc)) from exc
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(candidate, flags)
        except OSError as exc:
            raise TransferContractError(
                f"recorded cached input을 재검증할 수 없습니다: {relative}: {exc}"
            ) from exc
        try:
            current = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        try:
            pathname = candidate.lstat()
        except FileNotFoundError as exc:
            raise TransferContractError(
                f"recorded cached input이 사라졌습니다: {relative}"
            ) from exc
        actual = (
            int(current.st_dev),
            int(current.st_ino),
            int(current.st_size),
            int(current.st_mtime_ns),
            int(current.st_ctime_ns),
        )
        declared = (
            expected.device,
            expected.inode,
            expected.size,
            expected.mtime_ns,
            expected.ctime_ns,
        )
        if (
            not stat.S_ISREG(current.st_mode)
            or stat.S_ISLNK(pathname.st_mode)
            or (pathname.st_dev, pathname.st_ino) != (current.st_dev, current.st_ino)
            or actual != declared
        ):
            raise TransferContractError(
                "recorded cached input이 검증된 transfer snapshot 이후 변경됐습니다: "
                f"{relative}"
            )


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TransferContractError(f"transfer manifest JSON 중복 키: {key}")
        result[key] = value
    return result


def _relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TransferContractError(f"{field}는 비어 있지 않은 문자열이어야 합니다")
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise TransferContractError(f"{field}는 canonical 저장소 상대 POSIX 경로여야 합니다")
    return path.as_posix()


def _canonical_recorded_aggregate(entries: list[dict[str, object]]) -> str:
    evidence = [
        {
            "path": str(entry["path"]),
            "sha256": str(entry["sha256"]),
            "size": int(entry["size"]),
        }
        for entry in sorted(entries, key=lambda item: str(item["path"]))
    ]
    raw = json.dumps(
        evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _no_replace_head_commit(repo_root: Path) -> str:
    """receipt의 bootstrap commit을 현재 checkout과 no-replace로 교차검증한다."""

    environment = dict(os.environ, GIT_NO_REPLACE_OBJECTS="1")
    try:
        value = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=repo_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip().lower()
    except (OSError, subprocess.SubprocessError) as exc:
        raise TransferContractError(
            "bootstrap receipt를 사용할 때는 검증 가능한 git HEAD가 필요합니다"
        ) from exc
    if not _COMMIT_RE.fullmatch(value):
        raise TransferContractError("현재 git HEAD commit SHA가 유효하지 않습니다")
    return value


def _walk_recorded_tree(root: Path, *, repo_root: Path) -> tuple[set[str], int]:
    reject_symlink_components(root, root=repo_root)
    if not root.is_dir():
        raise TransferContractError(f"recorded root가 directory가 아닙니다: {root}")
    top_level_sessions: set[str] = set()
    files: set[str] = set()
    for current_text, dir_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_text)
        for name in list(dir_names):
            child = current / name
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise TransferContractError(f"recorded tree symlink directory 금지: {child}")
            if current == root:
                top_level_sessions.add(name)
        for name in file_names:
            child = current / name
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise TransferContractError(f"recorded tree regular file 위반: {child}")
            files.add(child.relative_to(repo_root).as_posix())
    for session in sorted(top_level_sessions):
        metadata = (root / session / "session.json").relative_to(repo_root).as_posix()
        if metadata not in files:
            raise TransferContractError(f"recorded session.json 누락: {session}")
    return files, len(top_level_sessions)


def validate_transfer_manifest(
    path: str | Path,
    *,
    repo_root: str | Path,
    expected_sha256: str,
) -> dict[str, object]:
    """manifest와 모든 전송 파일을 한 fd snapshot씩 hash하고 exact 집합을 검증한다."""

    root = Path(os.path.abspath(repo_root))
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(
        expected_sha256.lower()
    ):
        raise TransferContractError("expected transfer manifest SHA-256이 유효하지 않습니다")
    try:
        manifest = read_regular_file_snapshot(
            path, root=root, label="Jetson transfer manifest"
        )
    except HoldoutContractError as exc:
        raise TransferContractError(str(exc)) from exc
    if manifest.sha256 != expected_sha256.lower():
        raise TransferContractError(
            "Jetson transfer manifest SHA-256 불일치: "
            f"expected={expected_sha256.lower()}, actual={manifest.sha256}"
        )
    assert manifest.data is not None
    try:
        payload = json.loads(
            manifest.data.decode("utf-8"), object_pairs_hook=_object_without_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransferContractError(f"transfer manifest JSON 오류: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != TRANSFER_SCHEMA_VERSION:
        raise TransferContractError("transfer manifest schema_version은 1이어야 합니다")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise TransferContractError("transfer manifest files가 비었습니다")

    entries: list[dict[str, object]] = []
    validated_file_snapshots: dict[str, FileSnapshot] = {}
    paths: set[str] = set()
    by_role: dict[str, list[str]] = {role: [] for role in _ROLES}
    for index, raw_entry in enumerate(raw_files):
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "path",
            "role",
            "sha256",
            "size",
        }:
            raise TransferContractError(
                f"files[{index}]는 path/role/sha256/size만 정확히 포함해야 합니다"
            )
        relative = _relative_path(raw_entry["path"], field=f"files[{index}].path")
        role = raw_entry["role"]
        digest = raw_entry["sha256"]
        size = raw_entry["size"]
        if role not in _ROLES:
            raise TransferContractError(f"files[{index}].role이 허용 목록 밖입니다: {role!r}")
        if not relative.startswith(_ROLE_PREFIXES[str(role)]):
            raise TransferContractError(
                f"files[{index}] path가 role={role!r} 허용 prefix 밖입니다: {relative}"
            )
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise TransferContractError(f"files[{index}].sha256이 유효하지 않습니다")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise TransferContractError(f"files[{index}].size는 0 이상 정수여야 합니다")
        if relative in paths:
            raise TransferContractError(f"transfer manifest 중복 path: {relative}")
        paths.add(relative)
        try:
            snapshot = read_regular_file_snapshot(
                root / relative,
                root=root,
                label=f"transfer file #{index}",
                capture_bytes=role == "recorded_manifest",
            )
        except HoldoutContractError as exc:
            raise TransferContractError(str(exc)) from exc
        if snapshot.size != size or snapshot.sha256 != digest:
            raise TransferContractError(
                f"transfer file size/SHA 불일치: {relative}; "
                f"declared=({size},{digest}), actual=({snapshot.size},{snapshot.sha256})"
            )
        entry = {"path": relative, "role": role, "sha256": digest, "size": size}
        entries.append(entry)
        validated_file_snapshots[relative] = snapshot
        by_role[str(role)].append(relative)

    if by_role["rir_bank"] != ["data/rir_bank/duct_rirs_v1.npz"]:
        raise TransferContractError("rir_bank role은 official duct_rirs_v1.npz 정확히 1개여야 합니다")
    for role in ("strict_primary_npz", "strict_secondary_npz"):
        if len(by_role[role]) != 1 or not by_role[role][0].endswith(".npz"):
            raise TransferContractError(f"{role} NPZ는 정확히 1개여야 합니다")
    for role in ("strict_ps_raw", "strict_ps_analysis"):
        if not by_role[role]:
            raise TransferContractError(f"{role} 증거가 하나 이상 필요합니다")
    if by_role["recorded_manifest"] != ["data/manifests/recorded_regrouped.jsonl"]:
        raise TransferContractError("recorded_manifest role은 canonical regrouped JSONL 1개여야 합니다")
    if by_role["lineage_tracks"] != ["data/raw/music/fma_metadata/tracks.csv"]:
        raise TransferContractError("lineage_tracks role은 canonical FMA tracks.csv 1개여야 합니다")
    if by_role["librispeech_chapters_metadata"] != [
        "data/raw/speech/LibriSpeech/CHAPTERS.TXT"
    ]:
        raise TransferContractError(
            "librispeech_chapters_metadata role은 canonical CHAPTERS.TXT 1개여야 합니다"
        )
    if by_role["esc50_metadata"] != [
        "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv"
    ]:
        raise TransferContractError(
            "esc50_metadata role은 canonical ESC-50 esc50.csv 1개여야 합니다"
        )
    if by_role["canonical_holdout"] != ["data/manifests/recorded_holdout.json"]:
        raise TransferContractError("canonical_holdout role은 recorded_holdout.json 정확히 1개여야 합니다")
    if by_role["source_pool_v1_csv"] != ["data/source_pool/sources.csv"]:
        raise TransferContractError("source_pool_v1_csv role은 v1 sources.csv 정확히 1개여야 합니다")
    if by_role["source_pool_v2_csv"] != ["data/source_pool_v2/sources.csv"]:
        raise TransferContractError("source_pool_v2_csv role은 v2 sources.csv 정확히 1개여야 합니다")
    if len(by_role["provenance_report"]) != 1:
        raise TransferContractError("provenance_report role은 content-addressed report 1개여야 합니다")

    rir_value = _relative_path(payload.get("rir_bank"), field="rir_bank")
    if by_role["rir_bank"] != [rir_value]:
        raise TransferContractError("rir_bank pointer와 files role이 다릅니다")
    strict = payload.get("strict_ps")
    if not isinstance(strict, dict) or set(strict) != {
        "analysis",
        "capture_root",
        "primary_npz",
        "raw",
        "secondary_npz",
    }:
        raise TransferContractError("strict_ps pointer 구조가 불완전합니다")
    expected_strict = {
        "raw": sorted(by_role["strict_ps_raw"]),
        "analysis": sorted(by_role["strict_ps_analysis"]),
        "primary_npz": by_role["strict_primary_npz"][0],
        "secondary_npz": by_role["strict_secondary_npz"][0],
    }
    capture_root = _relative_path(strict.get("capture_root"), field="strict_ps.capture_root")
    if not capture_root.startswith("results/") or any(
        not value.startswith(capture_root.rstrip("/") + "/")
        for value in [*expected_strict["raw"], *expected_strict["analysis"]]
    ):
        raise TransferContractError(
            "strict_ps raw/analysis는 하나의 canonical capture_root 아래여야 합니다"
        )
    actual_strict = {
        "raw": [_relative_path(value, field="strict_ps.raw") for value in strict["raw"]]
        if isinstance(strict["raw"], list)
        else None,
        "analysis": [
            _relative_path(value, field="strict_ps.analysis") for value in strict["analysis"]
        ]
        if isinstance(strict["analysis"], list)
        else None,
        "primary_npz": _relative_path(strict["primary_npz"], field="strict_ps.primary_npz"),
        "secondary_npz": _relative_path(
            strict["secondary_npz"], field="strict_ps.secondary_npz"
        ),
    }
    if actual_strict != expected_strict:
        raise TransferContractError("strict_ps pointer와 files role exact 집합이 다릅니다")

    if payload.get("recorded_manifest") != by_role["recorded_manifest"][0]:
        raise TransferContractError("recorded_manifest pointer와 role이 다릅니다")
    if payload.get("lineage_tracks") != by_role["lineage_tracks"][0]:
        raise TransferContractError("lineage_tracks pointer와 role이 다릅니다")
    if (
        payload.get("librispeech_chapters_metadata")
        != by_role["librispeech_chapters_metadata"][0]
    ):
        raise TransferContractError(
            "librispeech_chapters_metadata pointer와 role이 다릅니다"
        )
    if payload.get("esc50_metadata") != by_role["esc50_metadata"][0]:
        raise TransferContractError("esc50_metadata pointer와 role이 다릅니다")
    try:
        holdout_summary = validate_holdout_contract(
            root / "data/manifests/recorded_holdout.json",
            repo_root=root,
        )
    except (OSError, HoldoutContractError) as exc:
        raise TransferContractError(
            f"canonical holdout lineage와 transfer bundle을 결속할 수 없습니다: {exc}"
        ) from exc
    lineage_summary = holdout_summary.get("lineage")
    if not isinstance(lineage_summary, dict):
        raise TransferContractError("canonical holdout lineage summary가 없습니다")
    recorded_tree_summary = holdout_summary.get("recorded_tree")
    if not isinstance(recorded_tree_summary, dict):
        raise TransferContractError("canonical holdout recorded-tree summary가 없습니다")
    lineage_tree_sha256 = recorded_tree_summary.get("metadata_snapshot_sha256")
    lineage_tree_content_sha256 = recorded_tree_summary.get("content_snapshot_sha256")
    lineage_tree_file_count = recorded_tree_summary.get("file_count")
    if (
        not isinstance(lineage_tree_sha256, str)
        or not _SHA256_RE.fullmatch(lineage_tree_sha256)
        or not isinstance(lineage_tree_content_sha256, str)
        or not _SHA256_RE.fullmatch(lineage_tree_content_sha256)
        or isinstance(lineage_tree_file_count, bool)
        or not isinstance(lineage_tree_file_count, int)
        or lineage_tree_file_count <= 0
    ):
        raise TransferContractError("canonical recorded-tree metadata 증거가 유효하지 않습니다")
    entry_by_path = {str(entry["path"]): entry for entry in entries}
    if (
        entry_by_path[by_role["recorded_manifest"][0]]["sha256"]
        != lineage_summary.get("regrouped_manifest_sha256")
        or entry_by_path[by_role["lineage_tracks"][0]]["sha256"]
        != lineage_summary.get("tracks_csv_sha256")
        or entry_by_path[by_role["librispeech_chapters_metadata"][0]]["sha256"]
        != lineage_summary.get("librispeech_chapters_sha256")
        or by_role["librispeech_chapters_metadata"][0]
        != lineage_summary.get("librispeech_chapters_path")
        or entry_by_path[by_role["esc50_metadata"][0]]["sha256"]
        != lineage_summary.get("esc50_metadata_sha256")
        or by_role["esc50_metadata"][0]
        != lineage_summary.get("esc50_metadata_path")
    ):
        raise TransferContractError(
            "transfer recorded_manifest/FMA tracks/LibriSpeech CHAPTERS/ESC-50 증거가 "
            "canonical provenance lineage와 다릅니다"
        )
    source_csv_paths = holdout_summary.get("sources_csv")
    source_csv_hashes = holdout_summary.get("sources_csv_sha256")
    report_path = holdout_summary.get("provenance_report")
    report_sha256 = holdout_summary.get("provenance_report_sha256")
    if (
        source_csv_paths
        != ["data/source_pool/sources.csv", "data/source_pool_v2/sources.csv"]
        or not isinstance(source_csv_hashes, dict)
        or not isinstance(report_path, str)
        or not isinstance(report_sha256, str)
    ):
        raise TransferContractError("canonical holdout provenance file summary가 불완전합니다")
    if by_role["provenance_report"] != [report_path]:
        raise TransferContractError(
            "transfer provenance_report가 canonical holdout pointer와 다릅니다"
        )
    expected_canonical_provenance = {
        "holdout": {
            "path": by_role["canonical_holdout"][0],
            "sha256": holdout_summary.get("sha256"),
        },
        "sources_csv": {
            "source_pool": {
                "path": by_role["source_pool_v1_csv"][0],
                "sha256": source_csv_hashes.get("source_pool"),
            },
            "source_pool_v2": {
                "path": by_role["source_pool_v2_csv"][0],
                "sha256": source_csv_hashes.get("source_pool_v2"),
            },
        },
        "provenance_report": {"path": report_path, "sha256": report_sha256},
    }
    canonical_provenance = payload.get("canonical_provenance")
    if canonical_provenance != expected_canonical_provenance:
        raise TransferContractError(
            "canonical_provenance pointer/SHA가 검증된 holdout bundle과 다릅니다"
        )
    for item in (
        expected_canonical_provenance["holdout"],
        *expected_canonical_provenance["sources_csv"].values(),
        expected_canonical_provenance["provenance_report"],
    ):
        entry = entry_by_path.get(str(item["path"]))
        if entry is None or entry["sha256"] != item["sha256"]:
            raise TransferContractError(
                f"canonical provenance transfer entry SHA 불일치: {item['path']}"
            )

    recorded_entries = [entry for entry in entries if entry["role"] == "recorded"]
    recorded = payload.get("recorded")
    if not isinstance(recorded, dict) or set(recorded) != {
        "aggregate_sha256",
        "file_count",
        "root",
        "session_count",
        "source_metadata_file_count",
        "source_metadata_snapshot_sha256",
        "source_content_snapshot_sha256",
        "total_bytes",
    }:
        raise TransferContractError("recorded aggregate 구조가 불완전합니다")
    if recorded.get("root") != "data/recorded":
        raise TransferContractError("recorded.root는 data/recorded여야 합니다")
    tree_files, actual_sessions = _walk_recorded_tree(
        root / "data/recorded", repo_root=root
    )
    declared_recorded_paths = {str(entry["path"]) for entry in recorded_entries}
    if tree_files != declared_recorded_paths:
        missing = sorted(tree_files - declared_recorded_paths)[:5]
        extra = sorted(declared_recorded_paths - tree_files)[:5]
        raise TransferContractError(
            f"recorded tree와 manifest exact 파일 집합이 다릅니다: missing={missing}, extra={extra}"
        )
    aggregate = _canonical_recorded_aggregate(recorded_entries)
    if lineage_tree_file_count != len(recorded_entries):
        raise TransferContractError(
            "canonical provenance recorded-tree file_count와 transfer exact 집합이 다릅니다: "
            f"provenance={lineage_tree_file_count}, transfer={len(recorded_entries)}"
        )
    expected_recorded = {
        "root": "data/recorded",
        "session_count": EXPECTED_RECORDED_SESSIONS,
        "file_count": len(recorded_entries),
        "total_bytes": sum(int(entry["size"]) for entry in recorded_entries),
        "aggregate_sha256": aggregate,
        "source_metadata_file_count": lineage_tree_file_count,
        "source_metadata_snapshot_sha256": lineage_tree_sha256,
        "source_content_snapshot_sha256": lineage_tree_content_sha256,
    }
    if actual_sessions != EXPECTED_RECORDED_SESSIONS or recorded != expected_recorded:
        raise TransferContractError(
            f"recorded aggregate 불일치: sessions={actual_sessions}, "
            f"expected={expected_recorded}, declared={recorded}"
        )
    return {
        "manifest_sha256": manifest.sha256,
        "file_count": len(entries),
        "total_bytes": sum(int(entry["size"]) for entry in entries),
        "recorded_session_count": actual_sessions,
        "recorded_aggregate_sha256": aggregate,
        "canonical_holdout_sha256": holdout_summary["sha256"],
        "_validated_transfer_manifest_snapshot": manifest,
        "_validated_recorded_manifest_snapshot": validated_file_snapshots[
            by_role["recorded_manifest"][0]
        ],
        "_validated_recorded_file_snapshots": {
            str(entry["path"]): validated_file_snapshots[str(entry["path"])]
            for entry in recorded_entries
        },
    }


def bind_recorded_transfer_config(
    data_cfg: dict[str, Any],
    *,
    repo_root: str | Path,
) -> RecordedTrainingSnapshot:
    """외부 manifest SHA를 검증하고 recorded aggregate를 resolved config에 주입한다.

    ``transfer_manifest_sha256``은 bootstrap/CLI가 별도 채널로 전달한 trust anchor다.
    aggregate는 그 검증된 manifest에서만 가져오며, 이미 설정돼 있으면 exact 비교한다.
    """

    if not isinstance(data_cfg, dict):
        raise TransferContractError("data config는 mapping이어야 합니다")
    root = Path(os.path.abspath(os.fspath(Path(repo_root).expanduser())))
    receipt_value = data_cfg.get("bootstrap_receipt")
    expected_receipt_sha = data_cfg.get("bootstrap_receipt_sha256")
    receipt_snapshot: FileSnapshot | None = None
    receipt_payload: dict[str, Any] | None = None
    if receipt_value is not None or expected_receipt_sha is not None:
        if receipt_value != "data/manifests/elice_bootstrap_receipt.json":
            raise TransferContractError(
                "data.bootstrap_receipt는 canonical 경로로 고정됩니다: "
                "data/manifests/elice_bootstrap_receipt.json"
            )
        if (
            not isinstance(expected_receipt_sha, str)
            or not _SHA256_RE.fullmatch(expected_receipt_sha.lower())
        ):
            raise TransferContractError(
                "data.bootstrap_receipt_sha256에 외부 전달된 64자리 SHA-256이 필요합니다"
            )
        try:
            receipt_snapshot = read_regular_file_snapshot(
                root / str(receipt_value),
                root=root,
                label="Elice bootstrap receipt",
            )
        except HoldoutContractError as exc:
            raise TransferContractError(str(exc)) from exc
        if receipt_snapshot.sha256 != expected_receipt_sha.lower():
            raise TransferContractError(
                "Elice bootstrap receipt SHA-256이 외부 trust anchor와 다릅니다"
            )
        assert receipt_snapshot.data is not None
        try:
            receipt_payload = json.loads(
                receipt_snapshot.data.decode("utf-8"),
                object_pairs_hook=_object_without_duplicates,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransferContractError(f"bootstrap receipt JSON 오류: {exc}") from exc
        if not isinstance(receipt_payload, dict) or set(receipt_payload) != {
            "schema_version",
            "expected_commit",
            "canonical_holdout",
            "transfer_manifest",
            "recorded_aggregate_sha256",
            "environment",
        }:
            raise TransferContractError("bootstrap receipt schema/필드가 불완전합니다")
        if (
            receipt_payload.get("schema_version") != 1
            or not isinstance(receipt_payload.get("expected_commit"), str)
            or not _COMMIT_RE.fullmatch(str(receipt_payload["expected_commit"]))
        ):
            raise TransferContractError("bootstrap receipt schema_version/commit이 유효하지 않습니다")
        current_commit = _no_replace_head_commit(root)
        if current_commit != receipt_payload["expected_commit"]:
            raise TransferContractError(
                "bootstrap receipt expected_commit과 현재 no-replace git HEAD가 다릅니다: "
                f"receipt={receipt_payload['expected_commit']}, current={current_commit}"
            )
        holdout_receipt = receipt_payload.get("canonical_holdout")
        transfer_receipt = receipt_payload.get("transfer_manifest")
        environment_receipt = receipt_payload.get("environment")
        if (
            not isinstance(holdout_receipt, dict)
            or set(holdout_receipt) != {"path", "sha256"}
            or holdout_receipt.get("path") != "data/manifests/recorded_holdout.json"
            or not isinstance(holdout_receipt.get("sha256"), str)
            or not _SHA256_RE.fullmatch(str(holdout_receipt["sha256"]))
            or not isinstance(transfer_receipt, dict)
            or set(transfer_receipt) != {"path", "sha256"}
            or transfer_receipt.get("path")
            != "data/manifests/elice_transfer_manifest.json"
            or not isinstance(transfer_receipt.get("sha256"), str)
            or not _SHA256_RE.fullmatch(str(transfer_receipt["sha256"]))
            or not isinstance(environment_receipt, dict)
            or set(environment_receipt)
            != {
                "freeze_receipt",
                "freeze_receipt_sha256",
                "torch_version",
                "torch_cuda",
            }
            or environment_receipt.get("freeze_receipt")
            != ".venv/environment-freeze.txt"
            or environment_receipt.get("torch_version") != "2.5.1+cu121"
            or environment_receipt.get("torch_cuda") != "12.1"
            or not isinstance(environment_receipt.get("freeze_receipt_sha256"), str)
            or not _SHA256_RE.fullmatch(
                str(environment_receipt["freeze_receipt_sha256"])
            )
            or not isinstance(receipt_payload.get("recorded_aggregate_sha256"), str)
            or not _SHA256_RE.fullmatch(
                str(receipt_payload["recorded_aggregate_sha256"])
            )
        ):
            raise TransferContractError("bootstrap receipt trust-chain 필드가 유효하지 않습니다")
        try:
            freeze_snapshot = read_regular_file_snapshot(
                root / str(environment_receipt["freeze_receipt"]),
                root=root,
                label="bootstrap environment freeze receipt",
                capture_bytes=False,
            )
        except HoldoutContractError as exc:
            raise TransferContractError(str(exc)) from exc
        if freeze_snapshot.sha256 != environment_receipt["freeze_receipt_sha256"]:
            raise TransferContractError(
                "bootstrap 이후 environment freeze receipt가 변경됐습니다"
            )
        configured_manifest = data_cfg.get("transfer_manifest")
        if configured_manifest is not None and configured_manifest != transfer_receipt["path"]:
            raise TransferContractError(
                "data.transfer_manifest가 bootstrap receipt pointer와 다릅니다"
            )
        configured_manifest_sha = data_cfg.get("transfer_manifest_sha256")
        if (
            configured_manifest_sha is not None
            and configured_manifest_sha != transfer_receipt["sha256"]
        ):
            raise TransferContractError(
                "data.transfer_manifest_sha256가 bootstrap receipt와 다릅니다"
            )
        data_cfg["bootstrap_receipt_sha256"] = expected_receipt_sha.lower()
        data_cfg["transfer_manifest"] = transfer_receipt["path"]
        data_cfg["transfer_manifest_sha256"] = transfer_receipt["sha256"]

    manifest_value = data_cfg.get("transfer_manifest")
    expected_manifest_sha = data_cfg.get("transfer_manifest_sha256")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise TransferContractError("data.transfer_manifest가 필요합니다")
    if (
        not isinstance(expected_manifest_sha, str)
        or not _SHA256_RE.fullmatch(expected_manifest_sha.lower())
    ):
        raise TransferContractError(
            "data.transfer_manifest_sha256에 외부 검증한 64자리 SHA-256이 필요합니다"
        )
    configured_path = Path(manifest_value).expanduser()
    if not configured_path.is_absolute():
        configured_path = root / configured_path
    configured_path = Path(os.path.abspath(os.fspath(configured_path)))
    canonical_path = root / "data/manifests/elice_transfer_manifest.json"
    if configured_path != canonical_path:
        raise TransferContractError(
            "data.transfer_manifest는 canonical 경로로 고정됩니다: "
            "data/manifests/elice_transfer_manifest.json"
        )
    summary = validate_transfer_manifest(
        configured_path,
        repo_root=root,
        expected_sha256=expected_manifest_sha.lower(),
    )
    aggregate = str(summary["recorded_aggregate_sha256"])
    if receipt_payload is not None:
        if summary["canonical_holdout_sha256"] != receipt_payload["canonical_holdout"]["sha256"]:
            raise TransferContractError(
                "bootstrap receipt holdout SHA와 transfer validator 결과가 다릅니다"
            )
        if aggregate != receipt_payload["recorded_aggregate_sha256"]:
            raise TransferContractError(
                "bootstrap receipt recorded aggregate와 transfer validator 결과가 다릅니다"
            )
    declared_aggregate = data_cfg.get("recorded_transfer_aggregate_sha256")
    if declared_aggregate is not None and declared_aggregate != aggregate:
        raise TransferContractError(
            "data.recorded_transfer_aggregate_sha256가 검증된 transfer manifest와 "
            f"다릅니다: declared={declared_aggregate!r}, actual={aggregate}"
        )
    data_cfg["transfer_manifest_sha256"] = expected_manifest_sha.lower()
    data_cfg["recorded_transfer_aggregate_sha256"] = aggregate
    manifest_snapshot = summary["_validated_transfer_manifest_snapshot"]
    recorded_manifest_snapshot = summary["_validated_recorded_manifest_snapshot"]
    recorded_files = summary["_validated_recorded_file_snapshots"]
    if (
        not isinstance(manifest_snapshot, FileSnapshot)
        or not isinstance(recorded_manifest_snapshot, FileSnapshot)
        or not isinstance(recorded_files, dict)
        or not recorded_files
    ):
        raise TransferContractError("validated transfer snapshot 내부 증거가 불완전합니다")
    if recorded_manifest_snapshot.data is None:
        raise TransferContractError("validated recorded manifest bytes snapshot이 없습니다")
    return RecordedTrainingSnapshot(
        repo_root=root,
        bootstrap_receipt=receipt_snapshot,
        transfer_manifest=manifest_snapshot,
        recorded_manifest=recorded_manifest_snapshot,
        recorded_aggregate_sha256=aggregate,
        recorded_files=dict(recorded_files),
    )


def validate_recorded_training_snapshot(
    data_cfg: dict[str, Any],
    *,
    repo_root: str | Path,
) -> RecordedTrainingSnapshot:
    """readiness/loader가 공유하는 이름: 검증과 resolved aggregate 결속을 함께 한다."""

    return bind_recorded_transfer_config(data_cfg, repo_root=repo_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        summary = validate_transfer_manifest(
            args.path,
            repo_root=args.repo_root,
            expected_sha256=args.expected_sha256,
        )
    except (OSError, TransferContractError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 1
    print(
        "[transfer] immutable bundle 확인: "
        f"manifest_sha256={summary['manifest_sha256']}, "
        f"files={summary['file_count']}, recorded_sessions={summary['recorded_session_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
