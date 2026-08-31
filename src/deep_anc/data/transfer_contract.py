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
from .recorded_generation import (
    COMBINED_SESSION_COUNT,
    PARENT_MANIFEST,
    RecordedGenerationError,
    validate_recorded_generation,
)
from .source_trust import (
    SourceTrustError,
    exact_clean_source_evidence,
    validate_environment_freeze_source_commit,
)


TRANSFER_SCHEMA_VERSION = 1
TRANSFER_GENERATION_SCHEMA_VERSION = 2
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
        "level_meter_raw",
        "level_meter_receipt",
        "recorded_manifest",
        "parent_recorded_manifest",
        "recorded_generation",
        "recorded_level_calibration",
        "recording_level_campaign",
        "recording_level_meter_raw",
        "recording_level_meter_receipt",
        "recording_source_gain_plan",
        "recording_gain_linearity_receipt",
        "recording_gain_linearity_plan",
        "recording_gain_linearity_raw",
        "recorded_source_plan",
        "recorded_source_selection",
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
    "level_meter_raw": "results/",
    "level_meter_receipt": "results/",
    "recorded_manifest": "data/manifests/",
    "parent_recorded_manifest": "data/manifests/",
    "recorded_generation": "data/manifests/recorded_generations/",
    "recorded_level_calibration": "data/manifests/recorded_level_calibration/",
    "recording_level_campaign": "results/recording_level_campaigns/",
    "recording_level_meter_raw": "results/",
    "recording_level_meter_receipt": "results/",
    "recording_source_gain_plan": "results/",
    "recording_gain_linearity_receipt": "results/",
    "recording_gain_linearity_plan": "results/",
    "recording_gain_linearity_raw": "results/",
    "recorded_source_plan": "data/source_plans/recorded_additions/",
    "recorded_source_selection": "data/source_plans/recorded_additions/",
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
    recorded_generation: FileSnapshot | None
    recorded_generation_summary: dict[str, Any] | None
    recorded_level_calibration: FileSnapshot | None
    recorded_aggregate_sha256: str
    recorded_files: dict[str, FileSnapshot]
    recorded_subband_coverage_report: FileSnapshot | None
    recorded_subband_coverage_receipt: dict[str, Any] | None

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

    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        value = subprocess.run(
            [
                "git",
                f"--git-dir={repo_root / '.git'}",
                f"--work-tree={repo_root}",
                "-c",
                f"core.worktree={repo_root}",
                "rev-parse",
                "--verify",
                "HEAD^{commit}",
            ],
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
    schema_version = payload.get("schema_version") if isinstance(payload, dict) else None
    if (
        type(schema_version) is not int
        or schema_version
        not in {TRANSFER_SCHEMA_VERSION, TRANSFER_GENERATION_SCHEMA_VERSION}
    ):
        raise TransferContractError("transfer manifest schema_version은 1 또는 2여야 합니다")
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
        role_prefix_ok = relative.startswith(_ROLE_PREFIXES[str(role)])
        if role == "recorded" and schema_version == TRANSFER_GENERATION_SCHEMA_VERSION:
            role_prefix_ok = relative.startswith("data/recorded/") or relative.startswith(
                "data/recorded_additions/"
            )
        if not role_prefix_ok:
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
                capture_bytes=role
                in {
                    "recorded_manifest",
                    "level_meter_receipt",
                    "recording_level_campaign",
                    "recording_level_meter_receipt",
                },
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
    if schema_version == TRANSFER_SCHEMA_VERSION:
        if by_role["recorded_manifest"] != [PARENT_MANIFEST]:
            raise TransferContractError(
                "schema v1 recorded_manifest role은 canonical regrouped JSONL 1개여야 합니다"
            )
        if any(
            by_role[role]
            for role in (
                "parent_recorded_manifest",
                "recorded_generation",
                "recorded_level_calibration",
                "recording_level_campaign",
                "recording_level_meter_raw",
                "recording_level_meter_receipt",
                "recording_source_gain_plan",
                "recording_gain_linearity_receipt",
                "recording_gain_linearity_plan",
                "recording_gain_linearity_raw",
                "recorded_source_plan",
                "recorded_source_selection",
            )
        ):
            raise TransferContractError("schema v1에는 recorded generation role을 넣을 수 없습니다")
    else:
        if len(by_role["recorded_manifest"]) != 1:
            raise TransferContractError("schema v2 combined recorded_manifest는 정확히 1개여야 합니다")
        if by_role["parent_recorded_manifest"] != [PARENT_MANIFEST]:
            raise TransferContractError(
                "schema v2 parent_recorded_manifest는 기존 82세션 manifest여야 합니다"
            )
        if len(by_role["recorded_generation"]) != 1 or len(by_role["recorded_source_plan"]) != 1:
            raise TransferContractError(
                "schema v2 recorded_generation/source_plan role은 각각 정확히 1개여야 합니다"
            )
        if len(by_role["recorded_level_calibration"]) != 1:
            raise TransferContractError(
                "schema v2 recorded_level_calibration receipt role은 정확히 1개여야 합니다"
            )
        campaign_count = len(by_role["recording_level_campaign"])
        if (
            campaign_count <= 0
            or len(by_role["recording_level_meter_raw"]) != campaign_count
            or len(by_role["recording_level_meter_receipt"]) != campaign_count
        ):
            raise TransferContractError(
                "schema v2 recording-level campaign/raw/receipt role은 같은 양수 개수여야 합니다"
            )
        gain_role_counts = {
            role: len(by_role[role])
            for role in (
                "recording_source_gain_plan",
                "recording_gain_linearity_receipt",
                "recording_gain_linearity_plan",
                "recording_gain_linearity_raw",
            )
        }
        if set(gain_role_counts.values()) != {1}:
            raise TransferContractError(
                "schema v2 canonical additions에는 source-gain/linearity authority "
                "4개 role이 각각 정확히 1개여야 합니다"
            )
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

    entry_by_path = {str(entry["path"]): entry for entry in entries}
    # tracked measurement_level_evidence.json이 참조하는 큰 meter raw와 sibling
    # receipt는 strict P/S capture_root와 다른 경로에 있으므로 별도 role로
    # 봉인한다. pointer가 있는 manifest에서는 두 role의 path/SHA 관계를 반드시
    # 검증하고, fixture/구형 manifest처럼 pointer가 없는 경우에는 기존 계약을
    # 유지한다.
    level_meter = payload.get("level_meter")
    level_roles_present = bool(
        by_role["level_meter_raw"] or by_role["level_meter_receipt"]
    )
    if level_meter is None:
        if level_roles_present:
            raise TransferContractError(
                "level_meter role이 있지만 top-level level_meter pointer가 없습니다"
            )
    else:
        if not isinstance(level_meter, dict) or set(level_meter) != {"raw", "receipt"}:
            raise TransferContractError("level_meter pointer 구조가 불완전합니다")
        if len(by_role["level_meter_raw"]) != 1 or len(
            by_role["level_meter_receipt"]
        ) != 1:
            raise TransferContractError(
                "level_meter raw/receipt role은 각각 정확히 1개여야 합니다"
            )
        expected_level = {
            "raw": by_role["level_meter_raw"][0],
            "receipt": by_role["level_meter_receipt"][0],
        }
        if level_meter != expected_level:
            raise TransferContractError("level_meter pointer와 role이 다릅니다")
        try:
            receipt_bytes = validated_file_snapshots[expected_level["receipt"]].data
            if receipt_bytes is None:
                raise ValueError("receipt bytes snapshot이 없습니다")
            receipt_payload = json.loads(receipt_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise TransferContractError("level meter receipt JSON을 읽을 수 없습니다") from exc
        raw_entry = entry_by_path[expected_level["raw"]]
        expected_receipt = {
            "raw_path": expected_level["raw"],
            "raw_sha256": raw_entry["sha256"],
            "schema": "measurement_level_meter_raw_receipt_v1",
        }
        if receipt_payload != expected_receipt:
            raise TransferContractError(
                "level meter receipt가 transfer raw path/SHA/schema와 다릅니다"
            )

    if payload.get("recorded_manifest") != by_role["recorded_manifest"][0]:
        raise TransferContractError("recorded_manifest pointer와 role이 다릅니다")
    calibration_pointer = payload.get("recorded_level_calibration")
    calibration_snapshot: FileSnapshot | None = None
    recording_campaign_pointer = payload.get("recording_level_campaigns")
    if schema_version == TRANSFER_SCHEMA_VERSION:
        if calibration_pointer is not None:
            raise TransferContractError(
                "schema v1에는 recorded_level_calibration pointer를 넣을 수 없습니다"
            )
        if recording_campaign_pointer is not None:
            raise TransferContractError(
                "schema v1에는 recording_level_campaigns pointer를 넣을 수 없습니다"
            )
    else:
        if calibration_pointer != by_role["recorded_level_calibration"][0]:
            raise TransferContractError(
                "recorded_level_calibration pointer와 role이 다릅니다"
            )
        from .recorded_level_calibration import (
            RecordedLevelCalibrationError,
            validate_recorded_level_calibration_receipt,
        )

        calibration_entry = entry_by_path[str(calibration_pointer)]
        calibration_snapshot = validated_file_snapshots[str(calibration_pointer)]
        try:
            calibration = validate_recorded_level_calibration_receipt(
                root / str(calibration_pointer),
                expected_sha256=str(calibration_entry["sha256"]),
                repo_root=root,
                verify_bound_audio=False,
                verify_current_commit=True,
            )
        except RecordedLevelCalibrationError as exc:
            raise TransferContractError(
                f"recorded level calibration receipt 검증 실패: {exc}"
            ) from exc
        # ``verify_bound_audio=False``는 4 GiB recorded WAV를 두 번 읽지
        # 않기 위한 선택이지, receipt가 선언한 audio SHA를 믿으라는
        # 뜻이 아니다. 이미 위 files loop에서 safe snapshot으로 검증한
        # role=recorded entry map과 82×2 ref를 exact 대조해 stale/재봉인
        # calibration receipt가 실제 전송 WAV와 다른 경로를 닫는다.
        calibration_sessions = calibration.payload.get("sessions")
        if not isinstance(calibration_sessions, list) or len(calibration_sessions) != 82:
            raise TransferContractError(
                "recorded level calibration session ref가 exact 82세션이 아닙니다"
            )
        audio_ref_count = 0
        for session_index, session in enumerate(calibration_sessions):
            if not isinstance(session, dict):
                raise TransferContractError(
                    f"recorded level calibration session #{session_index}가 mapping이 아닙니다"
                )
            for field in ("source_aligned", "mics"):
                ref = session.get(field)
                if not isinstance(ref, dict) or set(ref) != {"path", "sha256", "size"}:
                    raise TransferContractError(
                        "recorded level calibration audio ref schema가 다릅니다: "
                        f"session={session.get('session_id')!r}, field={field}"
                    )
                relative = str(ref.get("path") or "")
                entry = entry_by_path.get(relative)
                expected_ref = (
                    None
                    if entry is None
                    else {
                        "path": entry.get("path"),
                        "sha256": entry.get("sha256"),
                        "size": entry.get("size"),
                    }
                )
                if entry is None or entry.get("role") != "recorded" or ref != expected_ref:
                    raise TransferContractError(
                        "recorded level calibration audio ref가 transfer recorded bytes와 "
                        "다릅니다: "
                        f"session={session.get('session_id')!r}, field={field}, path={relative!r}"
                    )
                audio_ref_count += 1
        if audio_ref_count != 164:
            raise TransferContractError(
                "recorded level calibration audio ref는 exact 164개여야 합니다"
            )
    generation_summary: dict[str, Any] | None = None
    if schema_version == TRANSFER_GENERATION_SCHEMA_VERSION:
        generation_pointer = payload.get("recorded_generation")
        if generation_pointer != by_role["recorded_generation"][0]:
            raise TransferContractError("recorded_generation pointer와 role이 다릅니다")
        generation_entry = next(
            entry for entry in entries if entry["path"] == generation_pointer
        )
        try:
            generation_summary = validate_recorded_generation(
                root / str(generation_pointer),
                repo_root=root,
                expected_sha256=str(generation_entry["sha256"]),
                require_source_files=False,
            )
        except RecordedGenerationError as exc:
            raise TransferContractError(f"recorded generation 검증 실패: {exc}") from exc
        combined_ref = generation_summary["recorded_manifest"]
        additions_summary = generation_summary["additions"]
        if (
            not isinstance(combined_ref, dict)
            or combined_ref.get("path") != by_role["recorded_manifest"][0]
            or combined_ref.get("sha256")
            != next(
                entry["sha256"]
                for entry in entries
                if entry["path"] == by_role["recorded_manifest"][0]
            )
            or not isinstance(additions_summary, dict)
            or not isinstance(additions_summary.get("source_plan"), dict)
            or additions_summary["source_plan"].get("path")
            != by_role["recorded_source_plan"][0]
            or additions_summary["source_plan"].get("sha256")
            != next(
                entry["sha256"]
                for entry in entries
                if entry["path"] == by_role["recorded_source_plan"][0]
            )
        ):
            raise TransferContractError(
                "schema v2 combined manifest/source plan이 recorded generation과 다릅니다"
            )
        expected_campaigns = additions_summary.get("recording_level_campaigns")
        if (
            not isinstance(expected_campaigns, list)
            or not expected_campaigns
            or recording_campaign_pointer != expected_campaigns
        ):
            raise TransferContractError(
                "schema v2 recording_level_campaigns pointer가 generation과 다릅니다"
            )
        campaign_paths: list[str] = []
        meter_raw_paths: list[str] = []
        meter_receipt_paths: list[str] = []
        gain_authority_paths: dict[str, set[str]] = {
            "source_gain_plan": set(),
            "gain_linearity_receipt": set(),
            "gain_linearity_plan": set(),
            "gain_linearity_raw": set(),
        }
        seen_campaign_ids: set[str] = set()
        from .recording_level_campaign import (
            RecordingLevelCampaignError,
            validate_recording_level_campaign,
        )
        from .recording_gain_linearity import (
            RecordingGainLinearityError,
            validate_gain_linearity_receipt,
        )
        from .recording_source_gain import (
            RecordingSourceGainError,
            validate_recording_source_gain_plan,
        )

        gain_role_for_field = {
            "source_gain_plan": "recording_source_gain_plan",
            "gain_linearity_receipt": "recording_gain_linearity_receipt",
            "gain_linearity_plan": "recording_gain_linearity_plan",
            "gain_linearity_raw": "recording_gain_linearity_raw",
        }
        gain_campaign_keys = {
            "campaign_id",
            "campaign",
            "meter_raw",
            "meter_receipt",
            "hardware_config",
        } | set(gain_role_for_field)

        for index, campaign_ref in enumerate(expected_campaigns):
            campaign_keys = (
                frozenset(campaign_ref) if isinstance(campaign_ref, dict) else frozenset()
            )
            if campaign_keys != frozenset(gain_campaign_keys):
                raise TransferContractError(
                    f"schema v2 recording level campaign #{index} summary에 "
                    "source-gain/linearity authority 4종이 필수입니다"
                )
            campaign_id = campaign_ref.get("campaign_id")
            if not isinstance(campaign_id, str) or campaign_id in seen_campaign_ids:
                raise TransferContractError(
                    "schema v2 recording level campaign_id가 누락/중복됐습니다"
                )
            seen_campaign_ids.add(campaign_id)
            for field in ("campaign", "meter_raw", "meter_receipt", "hardware_config"):
                ref = campaign_ref.get(field)
                if (
                    not isinstance(ref, dict)
                    or set(ref) != {"path", "size", "sha256"}
                    or not isinstance(ref.get("path"), str)
                    or not isinstance(ref.get("sha256"), str)
                ):
                    raise TransferContractError(
                        f"schema v2 recording level campaign #{index} {field} ref 오류"
                    )
            campaign_file = campaign_ref["campaign"]
            meter_raw = campaign_ref["meter_raw"]
            meter_receipt = campaign_ref["meter_receipt"]
            hardware_config = campaign_ref["hardware_config"]
            campaign_paths.append(str(campaign_file["path"]))
            meter_raw_paths.append(str(meter_raw["path"]))
            meter_receipt_paths.append(str(meter_receipt["path"]))
            for ref in (campaign_file, meter_raw, meter_receipt):
                entry = entry_by_path.get(str(ref["path"]))
                if (
                    entry is None
                    or entry.get("sha256") != ref.get("sha256")
                    or entry.get("size") != ref.get("size")
                ):
                    raise TransferContractError(
                        "schema v2 recording level campaign transfer path/SHA/size 불일치"
                    )
            try:
                hardware_snapshot = read_regular_file_snapshot(
                    root / str(hardware_config["path"]),
                    root=root,
                    label=f"recording level hardware config #{index}",
                )
            except HoldoutContractError as exc:
                raise TransferContractError(str(exc)) from exc
            if (
                hardware_snapshot.sha256 != hardware_config.get("sha256")
                or hardware_snapshot.size != hardware_config.get("size")
            ):
                raise TransferContractError(
                    "schema v2 recording level hardware config SHA/size 불일치"
                )
            try:
                campaign_summary = validate_recording_level_campaign(
                    repo_root=root,
                    campaign_receipt=str(campaign_file["path"]),
                    expected_sha256=str(campaign_file["sha256"]),
                    require_fresh=False,
                )
            except (OSError, RecordingLevelCampaignError, ValueError) as exc:
                raise TransferContractError(
                    f"schema v2 recording level campaign 재검증 실패: {exc}"
                ) from exc
            campaign_payload = campaign_summary["payload"]
            if (
                campaign_summary.get("campaign_id") != campaign_id
                or campaign_payload.get("meter", {}).get("raw") != meter_raw
                or campaign_payload.get("meter", {}).get("receipt") != meter_receipt
                or campaign_payload.get("hardware", {}).get("config")
                != hardware_config
            ):
                raise TransferContractError(
                    "schema v2 recording level campaign refs가 campaign JSON에서 재유도되지 않습니다"
                )
            for field, role in gain_role_for_field.items():
                ref = campaign_ref.get(field)
                if (
                    not isinstance(ref, dict)
                    or set(ref) != {"path", "size", "sha256"}
                ):
                    raise TransferContractError(
                        f"schema v2 recording level campaign #{index} {field} ref 오류"
                    )
                path = str(ref["path"])
                entry = entry_by_path.get(path)
                if (
                    entry is None
                    or entry.get("role") != role
                    or entry.get("sha256") != ref.get("sha256")
                    or entry.get("size") != ref.get("size")
                ):
                    raise TransferContractError(
                        f"schema v2 {field} transfer role/path/SHA/size 불일치"
                    )
                gain_authority_paths[field].add(path)
            try:
                source_gain = validate_recording_source_gain_plan(
                    repo_root=root,
                    plan_path=str(campaign_ref["source_gain_plan"]["path"]),
                    expected_sha256=str(
                        campaign_ref["source_gain_plan"]["sha256"]
                    ),
                )
                gain_linearity = validate_gain_linearity_receipt(
                    repo_root=root,
                    receipt_path=str(
                        campaign_ref["gain_linearity_receipt"]["path"]
                    ),
                    expected_sha256=str(
                        campaign_ref["gain_linearity_receipt"]["sha256"]
                    ),
                )
            except (
                OSError,
                RecordingGainLinearityError,
                RecordingSourceGainError,
                ValueError,
            ) as exc:
                raise TransferContractError(
                    f"schema v2 source-gain/linearity authority 재검증 실패: {exc}"
                ) from exc
            source_payload = source_gain.get("payload")
            gain_payload = gain_linearity.get("payload")
            if (
                source_gain.get("canonical_live_eligible") is not True
                or gain_linearity.get("passed") is not True
                or not isinstance(source_payload, dict)
                or not isinstance(gain_payload, dict)
                or not isinstance(gain_payload.get("hardware"), dict)
                or source_payload.get("gain_linearity_receipt")
                != campaign_ref["gain_linearity_receipt"]
                or gain_payload.get("plan") != campaign_ref["gain_linearity_plan"]
                or gain_payload.get("raw") != campaign_ref["gain_linearity_raw"]
                or source_payload.get("gain_linearity_hardware")
                != gain_payload.get("hardware")
                or gain_payload.get("hardware", {}).get("path")
                != hardware_config.get("path")
                or gain_payload.get("hardware", {}).get("sha256")
                != hardware_config.get("sha256")
                or gain_payload.get("hardware", {}).get("size")
                != hardware_config.get("size")
            ):
                raise TransferContractError(
                    "schema v2 source-gain/linearity/campaign hardware 결속 불일치"
                )
        if (
            sorted(campaign_paths) != sorted(by_role["recording_level_campaign"])
            or sorted(meter_raw_paths)
            != sorted(by_role["recording_level_meter_raw"])
            or sorted(meter_receipt_paths)
            != sorted(by_role["recording_level_meter_receipt"])
        ):
            raise TransferContractError(
                "schema v2 recording-level campaign/raw/receipt role exact 집합이 generation과 다릅니다"
            )
        gain_role_paths = {
            field: set(by_role[role]) for field, role in gain_role_for_field.items()
        }
        if gain_authority_paths != gain_role_paths:
            raise TransferContractError(
                "schema v2 gain authority role exact 집합이 generation campaign refs와 다릅니다"
            )
        source_selections = additions_summary.get("source_selection", {})
        if not isinstance(source_selections, dict):
            raise TransferContractError(
                "schema v2 recorded source selection summary가 mapping이 아닙니다"
            )
        selection_kinds = (
            "external_dns_speech_selection",
            "external_demand_environment_selection",
        )
        selection_refs: dict[str, Any] = {}
        for selection_kind in selection_kinds:
            source_selection = source_selections.get(selection_kind)
            if source_selection is None:
                continue
            if not isinstance(source_selection, dict) or not isinstance(
                source_selection.get("bundle_files"), list
            ):
                raise TransferContractError(
                    f"schema v2 {selection_kind} bundle summary가 유효하지 않습니다"
                )
            for ref in source_selection["bundle_files"]:
                if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
                    raise TransferContractError(
                        f"schema v2 {selection_kind} bundle ref가 유효하지 않습니다"
                    )
                ref_path = str(ref["path"])
                if ref_path in selection_refs:
                    raise TransferContractError(
                        "schema v2 source selection bundle path가 selection 간 중복됩니다"
                    )
                selection_refs[ref_path] = ref
        expected_selection = sorted(selection_refs)
        if sorted(by_role["recorded_source_selection"]) != expected_selection:
            raise TransferContractError(
                "schema v2 recorded_source_selection role이 generation bundle union과 다릅니다"
            )
        for path in expected_selection:
            entry = next(
                (item for item in entries if str(item.get("path")) == path),
                None,
            )
            ref = selection_refs[path]
            if (
                not isinstance(entry, dict)
                or entry.get("sha256") != ref.get("sha256")
                or entry.get("size") != ref.get("size")
            ):
                raise TransferContractError(
                    "schema v2 source selection bundle path/SHA/size가 generation과 다릅니다"
                )
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
    lineage_manifest_path = (
        by_role["recorded_manifest"][0]
        if schema_version == TRANSFER_SCHEMA_VERSION
        else by_role["parent_recorded_manifest"][0]
    )
    if (
        entry_by_path[lineage_manifest_path]["sha256"]
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
    declared_recorded_paths = {str(entry["path"]) for entry in recorded_entries}
    aggregate = _canonical_recorded_aggregate(recorded_entries)
    if schema_version == TRANSFER_SCHEMA_VERSION:
        expected_recorded_keys = {
            "aggregate_sha256",
            "file_count",
            "root",
            "session_count",
            "source_metadata_file_count",
            "source_metadata_snapshot_sha256",
            "source_content_snapshot_sha256",
            "total_bytes",
        }
        if not isinstance(recorded, dict) or set(recorded) != expected_recorded_keys:
            raise TransferContractError("schema v1 recorded aggregate 구조가 불완전합니다")
        if recorded.get("root") != "data/recorded":
            raise TransferContractError("schema v1 recorded.root는 data/recorded여야 합니다")
        tree_files, actual_sessions = _walk_recorded_tree(
            root / "data/recorded", repo_root=root
        )
        if lineage_tree_file_count != len(recorded_entries):
            raise TransferContractError(
                "canonical provenance recorded-tree file_count와 transfer exact 파일 집합이 다릅니다: "
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
        expected_sessions = EXPECTED_RECORDED_SESSIONS
    else:
        assert generation_summary is not None
        additions_summary = generation_summary["additions"]
        parent_summary = generation_summary["parent"]
        if not isinstance(additions_summary, dict) or not isinstance(parent_summary, dict):
            raise TransferContractError("recorded generation parent/additions summary가 없습니다")
        parent_root = str(parent_summary.get("root"))
        additions_root = str(additions_summary.get("root"))
        parent_files, parent_sessions = _walk_recorded_tree(root / parent_root, repo_root=root)
        additions_files, additions_sessions = _walk_recorded_tree(
            root / additions_root, repo_root=root
        )
        tree_files = parent_files | additions_files
        actual_sessions = parent_sessions + additions_sessions
        parent_declared = {
            path for path in declared_recorded_paths if path.startswith(parent_root + "/")
        }
        if parent_declared != parent_files or len(parent_files) != lineage_tree_file_count:
            raise TransferContractError(
                "schema v2 parent 82 exact 파일 집합이 holdout anchor와 다릅니다"
            )
        expected_recorded = {
            "root": "recorded_generation",
            "parent_root": parent_root,
            "additions_root": additions_root,
            "generation_manifest": by_role["recorded_generation"][0],
            "session_count": COMBINED_SESSION_COUNT,
            "file_count": len(recorded_entries),
            "total_bytes": sum(int(entry["size"]) for entry in recorded_entries),
            "aggregate_sha256": aggregate,
            "source_metadata_file_count": lineage_tree_file_count,
            "source_metadata_snapshot_sha256": lineage_tree_sha256,
            "source_content_snapshot_sha256": lineage_tree_content_sha256,
        }
        expected_sessions = COMBINED_SESSION_COUNT
        if not isinstance(recorded, dict) or set(recorded) != set(expected_recorded):
            raise TransferContractError("schema v2 recorded aggregate 구조가 불완전합니다")
    if tree_files != declared_recorded_paths:
        missing = sorted(tree_files - declared_recorded_paths)[:5]
        extra = sorted(declared_recorded_paths - tree_files)[:5]
        raise TransferContractError(
            f"recorded tree와 manifest exact 파일 집합이 다릅니다: missing={missing}, extra={extra}"
        )
    if actual_sessions != expected_sessions or recorded != expected_recorded:
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
        "_validated_recorded_generation_snapshot": (
            None
            if schema_version == TRANSFER_SCHEMA_VERSION
            else validated_file_snapshots[by_role["recorded_generation"][0]]
        ),
        "_validated_recorded_generation_summary": generation_summary,
        "recorded_level_calibration": (
            None
            if calibration_snapshot is None
            else {
                "path": str(calibration_pointer),
                "sha256": calibration_snapshot.sha256,
            }
        ),
        "_validated_recorded_level_calibration_snapshot": calibration_snapshot,
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
    coverage_snapshot: FileSnapshot | None = None
    coverage_receipt: dict[str, Any] | None = None
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
        receipt_schema = receipt_payload.get("schema_version") if isinstance(receipt_payload, dict) else None
        expected_receipt_keys = {
            "schema_version",
            "expected_commit",
            "canonical_holdout",
            "transfer_manifest",
            "recorded_aggregate_sha256",
            "environment",
        }
        if receipt_schema == 2:
            expected_receipt_keys.add("recorded_subband_coverage")
        if not isinstance(receipt_payload, dict) or set(receipt_payload) != expected_receipt_keys:
            raise TransferContractError("bootstrap receipt schema/필드가 불완전합니다")
        if (
            receipt_schema not in {1, 2}
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
        try:
            exact_clean_source_evidence(
                root, expected_commit=str(receipt_payload["expected_commit"])
            )
        except SourceTrustError as exc:
            raise TransferContractError(
                f"bootstrap receipt clean exact source 재검증 실패: {exc}"
            ) from exc
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
        if receipt_schema == 2:
            raw_coverage = receipt_payload.get("recorded_subband_coverage")
            coverage_keys = {
                "path",
                "sha256",
                "evidence_sha256",
                "manifest_sha256",
                "training_timing_contract_sha256",
                "coverage_contract_sha256",
                "all_requested_splits_pass",
            }
            if (
                not isinstance(raw_coverage, dict)
                or set(raw_coverage) != coverage_keys
                or not isinstance(raw_coverage.get("path"), str)
                or not str(raw_coverage["path"]).startswith(
                    "results/data_audit/recorded_subband_coverage/"
                )
                or not str(raw_coverage["path"]).endswith(".json")
                or any(
                    not isinstance(raw_coverage.get(key), str)
                    or not _SHA256_RE.fullmatch(str(raw_coverage[key]))
                    for key in (
                        "sha256",
                        "evidence_sha256",
                        "manifest_sha256",
                        "training_timing_contract_sha256",
                        "coverage_contract_sha256",
                    )
                )
                or Path(str(raw_coverage["path"])).stem
                != str(raw_coverage["coverage_contract_sha256"])
                or not isinstance(raw_coverage.get("all_requested_splits_pass"), bool)
            ):
                raise TransferContractError(
                    "bootstrap receipt recorded_subband_coverage 필드가 유효하지 않습니다"
                )
            try:
                coverage_snapshot = read_regular_file_snapshot(
                    root / str(raw_coverage["path"]),
                    root=root,
                    label="bootstrap recorded subband coverage report",
                )
            except HoldoutContractError as exc:
                raise TransferContractError(str(exc)) from exc
            if coverage_snapshot.sha256 != raw_coverage["sha256"]:
                raise TransferContractError(
                    "bootstrap 이후 recorded subband coverage report가 변경됐습니다"
                )
            assert coverage_snapshot.data is not None
            try:
                coverage_payload = json.loads(
                    coverage_snapshot.data.decode("utf-8"),
                    object_pairs_hook=_object_without_duplicates,
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TransferContractError(
                    f"recorded subband coverage report JSON 오류: {exc}"
                ) from exc
            for receipt_key, report_key in (
                ("evidence_sha256", "evidence_sha256"),
                ("manifest_sha256", "manifest"),
                ("training_timing_contract_sha256", "training_timing_contract_sha256"),
                ("coverage_contract_sha256", "coverage_contract_sha256"),
                ("all_requested_splits_pass", "all_requested_splits_pass"),
            ):
                report_value = coverage_payload.get(report_key)
                if report_key == "manifest" and isinstance(report_value, dict):
                    report_value = report_value.get("sha256")
                if report_value != raw_coverage[receipt_key]:
                    raise TransferContractError(
                        "bootstrap coverage receipt와 report semantic field가 다릅니다: "
                        f"{receipt_key}"
                    )
            coverage_receipt = dict(raw_coverage)
        try:
            freeze_snapshot = read_regular_file_snapshot(
                root / str(environment_receipt["freeze_receipt"]),
                root=root,
                label="bootstrap environment freeze receipt",
                capture_bytes=True,
            )
        except HoldoutContractError as exc:
            raise TransferContractError(str(exc)) from exc
        if freeze_snapshot.sha256 != environment_receipt["freeze_receipt_sha256"]:
            raise TransferContractError(
                "bootstrap 이후 environment freeze receipt가 변경됐습니다"
            )
        assert freeze_snapshot.data is not None
        try:
            validate_environment_freeze_source_commit(
                freeze_snapshot.data,
                expected_commit=str(receipt_payload["expected_commit"]),
            )
        except SourceTrustError as exc:
            raise TransferContractError(
                f"bootstrap environment freeze source 결속 실패: {exc}"
            ) from exc
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
    recorded_generation_snapshot = summary.get(
        "_validated_recorded_generation_snapshot"
    )
    recorded_generation_summary = summary.get(
        "_validated_recorded_generation_summary"
    )
    recorded_level_calibration_snapshot = summary.get(
        "_validated_recorded_level_calibration_snapshot"
    )
    declared_generation = data_cfg.get("recorded_generation")
    declared_generation_sha = data_cfg.get("recorded_generation_sha256")
    if isinstance(recorded_generation_snapshot, FileSnapshot):
        generation_relative = recorded_generation_snapshot.path.relative_to(root).as_posix()
        if declared_generation in (None, "") and declared_generation_sha in (None, ""):
            # bootstrap receipt의 외부 SHA -> transfer SHA -> generation FileSnapshot은
            # 이미 하나의 검증된 trust chain이다. schema v2에서 같은 값을 모든 runner
            # CLI에 반복 입력하게 하지 않고 이 snapshot에서 원자 materialize한다.
            data_cfg["recorded_generation"] = generation_relative
            data_cfg["recorded_generation_sha256"] = recorded_generation_snapshot.sha256
        elif declared_generation in (None, "") or declared_generation_sha in (None, ""):
            raise TransferContractError(
                "schema v2 recorded_generation path/SHA는 둘 다 비우거나 둘 다 exact여야 합니다"
            )
        elif (
            declared_generation != generation_relative
            or declared_generation_sha != recorded_generation_snapshot.sha256
        ):
            raise TransferContractError(
                "schema v2 transfer의 data.recorded_generation/path SHA가 검증값과 다릅니다: "
                f"expected=({generation_relative},{recorded_generation_snapshot.sha256}), "
                f"declared=({declared_generation},{declared_generation_sha})"
            )
    elif declared_generation not in (None, "") or declared_generation_sha not in (None, ""):
        raise TransferContractError(
            "schema v1 transfer에 data.recorded_generation 선언을 결합할 수 없습니다"
        )
    declared_calibration = data_cfg.get("recorded_level_calibration")
    declared_calibration_sha = data_cfg.get("recorded_level_calibration_sha256")
    if isinstance(recorded_level_calibration_snapshot, FileSnapshot):
        calibration_relative = (
            recorded_level_calibration_snapshot.path.relative_to(root).as_posix()
        )
        if declared_calibration in (None, "") and declared_calibration_sha in (None, ""):
            # generation과 같은 transfer trust chain에서 두 필드를 한 번에 만든다.
            # path만 또는 SHA만 사용자가 채운 상태는 아래에서 실패 폐쇄한다.
            data_cfg["recorded_level_calibration"] = calibration_relative
            data_cfg["recorded_level_calibration_sha256"] = (
                recorded_level_calibration_snapshot.sha256
            )
        elif declared_calibration in (None, "") or declared_calibration_sha in (None, ""):
            raise TransferContractError(
                "schema v2 recorded_level_calibration path/SHA는 둘 다 비우거나 "
                "둘 다 exact여야 합니다"
            )
        elif (
            declared_calibration != calibration_relative
            or declared_calibration_sha != recorded_level_calibration_snapshot.sha256
        ):
            raise TransferContractError(
                "schema v2 transfer의 data.recorded_level_calibration/path SHA가 "
                "검증값과 다릅니다: "
                f"expected=({calibration_relative},"
                f"{recorded_level_calibration_snapshot.sha256}), "
                f"declared=({declared_calibration},{declared_calibration_sha})"
            )
    elif declared_calibration not in (None, "") or declared_calibration_sha not in (
        None,
        "",
    ):
        raise TransferContractError(
            "schema v1 transfer에 data.recorded_level_calibration 선언을 결합할 수 없습니다"
        )
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
        recorded_generation=(
            recorded_generation_snapshot
            if isinstance(recorded_generation_snapshot, FileSnapshot)
            else None
        ),
        recorded_generation_summary=(
            dict(recorded_generation_summary)
            if isinstance(recorded_generation_summary, dict)
            else None
        ),
        recorded_level_calibration=(
            recorded_level_calibration_snapshot
            if isinstance(recorded_level_calibration_snapshot, FileSnapshot)
            else None
        ),
        recorded_aggregate_sha256=aggregate,
        recorded_files=dict(recorded_files),
        recorded_subband_coverage_report=coverage_snapshot,
        recorded_subband_coverage_receipt=coverage_receipt,
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
