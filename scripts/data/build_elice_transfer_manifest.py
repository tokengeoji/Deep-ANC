#!/usr/bin/env python3
"""Jetson의 immutable Elice 학습 입력을 canonical transfer manifest로 결속한다."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.holdout_contract import (  # noqa: E402
    HoldoutContractError,
    read_regular_file_snapshot,
    reject_symlink_components,
    snapshot_regular_tree_metadata,
    validate_holdout_contract,
)
from deep_anc.data.transfer_contract import (  # noqa: E402
    EXPECTED_RECORDED_SESSIONS,
    TransferContractError,
    _canonical_recorded_aggregate,
    _walk_recorded_tree,
    validate_transfer_manifest,
)
from deep_anc.data.recorded_generation import (  # noqa: E402
    COMBINED_SESSION_COUNT,
    PARENT_MANIFEST,
    RecordedGenerationError,
    validate_recorded_generation,
)
from deep_anc.data.recorded_level_calibration import (  # noqa: E402
    RecordedLevelCalibrationError,
    validate_recorded_level_calibration_receipt,
)


OUTPUT = "data/manifests/elice_transfer_manifest.json"
TRANSFER_HISTORY_DIR = "data/manifests/elice_transfer_history"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _level_meter_support(
    args: argparse.Namespace, *, repo_root: Path
) -> tuple[str, str] | None:
    """tracked level evidence가 참조하는 raw/receipt를 transfer bundle에 결속한다.

    레벨 JSON은 git tracked 파일이고, 큰 NPZ는 보통 git-ignored다. 따라서 JSON만
    clone된 Elice에서 pytest가 조용히 빠지지 않도록 raw와 canonical sibling receipt를
    별도 role로 봉인한다. fixture처럼 level evidence가 없는 최소 builder 입력은
    기존 schema v1 동작을 유지한다.
    """

    raw_arg = getattr(args, "level_meter_raw", None)
    receipt_arg = getattr(args, "level_meter_receipt", None)
    if (raw_arg is None) != (receipt_arg is None):
        raise TransferContractError(
            "level_meter_raw와 level_meter_receipt는 함께 지정해야 합니다"
        )
    evidence_path = repo_root / "assets/measured/measurement_level_evidence.json"
    if raw_arg is None and not evidence_path.is_file():
        return None
    if raw_arg is None:
        try:
            evidence_snapshot = read_regular_file_snapshot(
                evidence_path,
                root=repo_root,
                label="tracked measurement level evidence",
            )
            payload = json.loads(evidence_snapshot.data.decode("utf-8"))
            raw_value = payload["meter_raw"]["path"]
            expected_raw_sha = str(payload["meter_raw"]["sha256"]).lower()
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransferContractError(
                "tracked measurement level evidence의 meter_raw 참조를 읽을 수 없습니다"
            ) from exc
        raw = _relative(str(raw_value), field="level_meter_raw", prefix="results/")
        receipt = PurePosixPath(raw).with_name(
            f"{PurePosixPath(raw).stem}.receipt.json"
        ).as_posix()
    else:
        raw = _relative(str(raw_arg), field="level_meter_raw", prefix="results/")
        receipt = _relative(
            str(receipt_arg), field="level_meter_receipt", prefix="results/"
        )
        expected_raw_sha = None

    raw_entry = _entry(raw, "level_meter_raw", repo_root=repo_root)
    receipt_entry = _entry(receipt, "level_meter_receipt", repo_root=repo_root)
    if expected_raw_sha is not None and raw_entry["sha256"] != expected_raw_sha:
        raise TransferContractError(
            "tracked measurement level evidence와 level meter raw SHA가 다릅니다"
        )
    try:
        receipt_payload = json.loads(
            (repo_root / receipt).read_bytes().decode("utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransferContractError("level meter receipt JSON을 읽을 수 없습니다") from exc
    expected_receipt = {
        "raw_path": raw,
        "raw_sha256": raw_entry["sha256"],
        "schema": "measurement_level_meter_raw_receipt_v1",
    }
    if receipt_payload != expected_receipt:
        raise TransferContractError(
            "level meter receipt가 raw path/SHA/schema와 exact하게 일치하지 않습니다"
        )
    return raw, receipt


def _relative(value: str, *, field: str, prefix: str | None = None) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise TransferContractError(f"{field}는 canonical 저장소 상대경로여야 합니다")
    result = path.as_posix()
    if prefix is not None and not result.startswith(prefix):
        raise TransferContractError(f"{field}는 {prefix!r} 아래여야 합니다: {result}")
    return result


def _entry(relative: str, role: str, *, repo_root: Path) -> dict[str, object]:
    try:
        snapshot = read_regular_file_snapshot(
            repo_root / relative,
            root=repo_root,
            label=f"transfer input {role}",
            capture_bytes=False,
        )
    except HoldoutContractError as exc:
        raise TransferContractError(str(exc)) from exc
    return {
        "path": relative,
        "role": role,
        "sha256": snapshot.sha256,
        "size": snapshot.size,
    }


def _build_payload_v1(args: argparse.Namespace, *, repo_root: Path) -> dict[str, object]:
    canonical_holdout = "data/manifests/recorded_holdout.json"
    holdout_summary = validate_holdout_contract(
        repo_root / canonical_holdout,
        repo_root=repo_root,
        expected_sha256=args.expected_holdout_sha256,
    )
    canonical_source_csv = holdout_summary["sources_csv"]
    canonical_source_csv_hashes = holdout_summary["sources_csv_sha256"]
    canonical_report = str(holdout_summary["provenance_report"])
    canonical_report_sha256 = str(holdout_summary["provenance_report_sha256"])
    recorded_root = _relative(args.recorded_root, field="recorded_root")
    if recorded_root != "data/recorded":
        raise TransferContractError("recorded_root는 data/recorded로 고정됩니다")
    recorded_before = snapshot_regular_tree_metadata(
        repo_root / recorded_root,
        repo_root=repo_root,
        label="Jetson recorded transfer source",
    )
    tree_files, session_count = _walk_recorded_tree(
        repo_root / recorded_root, repo_root=repo_root
    )
    if session_count != EXPECTED_RECORDED_SESSIONS:
        raise TransferContractError(
            f"recorded session 수가 {EXPECTED_RECORDED_SESSIONS}가 아닙니다: {session_count}"
        )
    entries = [
        _entry(relative, "recorded", repo_root=repo_root)
        for relative in sorted(tree_files)
    ]
    recorded_after = snapshot_regular_tree_metadata(
        repo_root / recorded_root,
        repo_root=repo_root,
        label="Jetson recorded transfer source 재검증",
    )
    if (
        recorded_before.entries != recorded_after.entries
        or recorded_before.sha256 != recorded_after.sha256
        or recorded_before.content_entries != recorded_after.content_entries
        or recorded_before.content_sha256 != recorded_after.content_sha256
    ):
        raise TransferContractError(
            "recorded tree가 transfer content hashing 중 변경됐습니다"
        )

    rir = _relative(args.rir_bank, field="rir_bank", prefix="data/rir_bank/")
    if rir != "data/rir_bank/duct_rirs_v1.npz":
        raise TransferContractError("rir_bank는 official data/rir_bank/duct_rirs_v1.npz여야 합니다")
    recorded_manifest = _relative(
        args.recorded_manifest,
        field="recorded_manifest",
        prefix="data/manifests/",
    )
    if recorded_manifest != "data/manifests/recorded_regrouped.jsonl":
        raise TransferContractError("recorded_manifest는 canonical recorded_regrouped.jsonl이어야 합니다")
    lineage_tracks = _relative(
        args.lineage_tracks,
        field="lineage_tracks",
        prefix="data/raw/music/fma_metadata/",
    )
    if lineage_tracks != "data/raw/music/fma_metadata/tracks.csv":
        raise TransferContractError("lineage_tracks는 canonical FMA tracks.csv여야 합니다")
    librispeech_chapters = _relative(
        args.librispeech_chapters_metadata,
        field="librispeech_chapters_metadata",
        prefix="data/raw/speech/LibriSpeech/",
    )
    if librispeech_chapters != "data/raw/speech/LibriSpeech/CHAPTERS.TXT":
        raise TransferContractError(
            "librispeech_chapters_metadata는 canonical LibriSpeech CHAPTERS.TXT여야 합니다"
        )
    esc50_metadata = _relative(
        args.esc50_metadata,
        field="esc50_metadata",
        prefix="data/raw/noise/esc50/ESC-50-master/meta/",
    )
    if esc50_metadata != "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv":
        raise TransferContractError(
            "esc50_metadata는 canonical ESC-50 meta/esc50.csv여야 합니다"
        )
    primary = _relative(
        args.primary_npz, field="primary_npz", prefix="assets/measured/"
    )
    secondary = _relative(
        args.secondary_npz, field="secondary_npz", prefix="assets/measured/"
    )
    raw_paths = sorted(
        {_relative(value, field="strict_raw", prefix="results/") for value in args.strict_raw}
    )
    analysis_paths = sorted(
        {
            _relative(value, field="strict_analysis", prefix="results/")
            for value in args.strict_analysis
        }
    )
    if len(raw_paths) != len(args.strict_raw) or len(analysis_paths) != len(args.strict_analysis):
        raise TransferContractError("strict raw/analysis 경로에 중복이 있습니다")
    capture_candidates = [*raw_paths, *analysis_paths]
    capture_root = PurePosixPath(
        os.path.commonpath([str(PurePosixPath(value).parent) for value in capture_candidates])
    ).as_posix()
    if capture_root == "results" or not capture_root.startswith("results/"):
        raise TransferContractError(
            "strict raw/analysis는 results/<한 capture root>/ 아래에 함께 있어야 합니다"
        )

    fixed = [
        (rir, "rir_bank"),
        (recorded_manifest, "recorded_manifest"),
        (lineage_tracks, "lineage_tracks"),
        (librispeech_chapters, "librispeech_chapters_metadata"),
        (esc50_metadata, "esc50_metadata"),
        (primary, "strict_primary_npz"),
        (secondary, "strict_secondary_npz"),
        *((value, "strict_ps_raw") for value in raw_paths),
        *((value, "strict_ps_analysis") for value in analysis_paths),
        (canonical_holdout, "canonical_holdout"),
        (str(canonical_source_csv[0]), "source_pool_v1_csv"),
        (str(canonical_source_csv[1]), "source_pool_v2_csv"),
        (canonical_report, "provenance_report"),
    ]
    level_meter = _level_meter_support(args, repo_root=repo_root)
    if level_meter is not None:
        fixed.extend(
            [
                (level_meter[0], "level_meter_raw"),
                (level_meter[1], "level_meter_receipt"),
            ]
        )
    if len({value for value, _role in fixed}) != len(fixed):
        raise TransferContractError("서로 다른 transfer role이 같은 파일을 가리킵니다")
    entries.extend(_entry(value, role, repo_root=repo_root) for value, role in fixed)
    entries.sort(key=lambda item: str(item["path"]))
    recorded_entries = [entry for entry in entries if entry["role"] == "recorded"]
    payload = {
        "schema_version": 1,
        "files": entries,
        "recorded": {
            "root": recorded_root,
            "session_count": session_count,
            "file_count": len(recorded_entries),
            "total_bytes": sum(int(entry["size"]) for entry in recorded_entries),
            "aggregate_sha256": _canonical_recorded_aggregate(recorded_entries),
            "source_metadata_file_count": recorded_before.file_count,
            "source_metadata_snapshot_sha256": recorded_before.sha256,
            "source_content_snapshot_sha256": recorded_before.content_sha256,
        },
        "rir_bank": rir,
        "recorded_manifest": recorded_manifest,
        "lineage_tracks": lineage_tracks,
        "librispeech_chapters_metadata": librispeech_chapters,
        "esc50_metadata": esc50_metadata,
        "canonical_provenance": {
            "holdout": {
                "path": canonical_holdout,
                "sha256": holdout_summary["sha256"],
            },
            "sources_csv": {
                "source_pool": {
                    "path": canonical_source_csv[0],
                    "sha256": canonical_source_csv_hashes["source_pool"],
                },
                "source_pool_v2": {
                    "path": canonical_source_csv[1],
                    "sha256": canonical_source_csv_hashes["source_pool_v2"],
                },
            },
            "provenance_report": {
                "path": canonical_report,
                "sha256": canonical_report_sha256,
            },
        },
        "strict_ps": {
            "capture_root": capture_root,
            "raw": raw_paths,
            "analysis": analysis_paths,
            "primary_npz": primary,
            "secondary_npz": secondary,
        },
    }
    if level_meter is not None:
        payload["level_meter"] = {
            "raw": level_meter[0],
            "receipt": level_meter[1],
        }
    return payload


def _build_payload_v2(args: argparse.Namespace, *, repo_root: Path) -> dict[str, object]:
    calibration_arg = getattr(args, "recorded_level_calibration", None)
    if not calibration_arg:
        raise TransferContractError(
            "schema-v2 transfer에는 --recorded-level-calibration이 필요합니다"
        )
    calibration_relative = _relative(
        str(calibration_arg),
        field="recorded_level_calibration",
        prefix="data/manifests/recorded_level_calibration/",
    )
    calibration_entry = _entry(
        calibration_relative, "recorded_level_calibration", repo_root=repo_root
    )
    try:
        validate_recorded_level_calibration_receipt(
            repo_root / calibration_relative,
            expected_sha256=str(calibration_entry["sha256"]),
            repo_root=repo_root,
            verify_bound_audio=False,
            verify_current_commit=True,
        )
    except RecordedLevelCalibrationError as exc:
        raise TransferContractError(
            f"recorded level calibration receipt 검증 실패: {exc}"
        ) from exc
    generation_relative = _relative(
        args.recorded_generation,
        field="recorded_generation",
        prefix="data/manifests/recorded_generations/",
    )
    generation_snapshot = read_regular_file_snapshot(
        repo_root / generation_relative,
        root=repo_root,
        label="recorded generation transfer source",
    )
    try:
        generation = validate_recorded_generation(
            repo_root / generation_relative,
            repo_root=repo_root,
            expected_sha256=generation_snapshot.sha256,
            require_source_files=True,
        )
    except RecordedGenerationError as exc:
        raise TransferContractError(f"recorded generation 검증 실패: {exc}") from exc

    legacy_values = dict(vars(args))
    legacy_values["recorded_root"] = "data/recorded"
    legacy_values["recorded_manifest"] = PARENT_MANIFEST
    legacy_values["recorded_generation"] = None
    legacy = _build_payload_v1(argparse.Namespace(**legacy_values), repo_root=repo_root)
    files = [dict(item) for item in legacy["files"]]
    parent_manifest_entries = [
        item for item in files if item["role"] == "recorded_manifest"
    ]
    if len(parent_manifest_entries) != 1:
        raise TransferContractError("legacy parent recorded manifest role이 유일하지 않습니다")
    parent_manifest_entries[0]["role"] = "parent_recorded_manifest"

    additions = generation.get("additions")
    combined = generation.get("combined")
    recorded_manifest_ref = generation.get("recorded_manifest")
    if (
        not isinstance(additions, dict)
        or not isinstance(combined, dict)
        or not isinstance(recorded_manifest_ref, dict)
        or not isinstance(additions.get("source_plan"), dict)
    ):
        raise TransferContractError("recorded generation summary가 불완전합니다")
    additions_root = str(additions.get("root"))
    additions_files, additions_sessions = _walk_recorded_tree(
        repo_root / additions_root, repo_root=repo_root
    )
    if additions_sessions != generation.get("addition_session_count"):
        raise TransferContractError("recorded generation additions session 수가 다릅니다")
    files.extend(
        _entry(relative, "recorded", repo_root=repo_root)
        for relative in sorted(additions_files)
    )
    combined_manifest = str(recorded_manifest_ref.get("path"))
    source_plan = str(additions["source_plan"].get("path"))
    files.extend(
        [
            _entry(combined_manifest, "recorded_manifest", repo_root=repo_root),
            _entry(generation_relative, "recorded_generation", repo_root=repo_root),
            _entry(source_plan, "recorded_source_plan", repo_root=repo_root),
            calibration_entry,
        ]
    )
    level_campaigns = additions.get("recording_level_campaigns")
    if not isinstance(level_campaigns, list) or not level_campaigns:
        raise TransferContractError(
            "schema-v2 generation에는 recording_level_campaigns가 하나 이상 필요합니다"
        )
    campaign_roles = {
        "campaign": "recording_level_campaign",
        "meter_raw": "recording_level_meter_raw",
        "meter_receipt": "recording_level_meter_receipt",
    }
    for index, campaign in enumerate(level_campaigns):
        if not isinstance(campaign, dict) or set(campaign) != {
            "campaign_id",
            "campaign",
            "meter_raw",
            "meter_receipt",
            "hardware_config",
        }:
            raise TransferContractError(
                f"recording level campaign summary #{index}가 불완전합니다"
            )
        hardware_ref = campaign.get("hardware_config")
        if (
            not isinstance(hardware_ref, dict)
            or set(hardware_ref) != {"path", "sha256", "size"}
        ):
            raise TransferContractError(
                f"recording level campaign #{index} hardware config ref가 불완전합니다"
            )
        hardware_entry = _entry(
            str(hardware_ref.get("path")),
            # tracked checkout 파일이므로 transfer role에는 추가하지 않고 bytes만 재대조한다.
            "recording_level_campaign",
            repo_root=repo_root,
        )
        if (
            hardware_entry["sha256"] != hardware_ref.get("sha256")
            or hardware_entry["size"] != hardware_ref.get("size")
        ):
            raise TransferContractError(
                f"recording level campaign #{index} hardware config SHA/size 불일치"
            )
        for field, role in campaign_roles.items():
            ref = campaign.get(field)
            if (
                not isinstance(ref, dict)
                or set(ref) != {"path", "sha256", "size"}
            ):
                raise TransferContractError(
                    f"recording level campaign #{index} {field} ref가 불완전합니다"
                )
            entry = _entry(str(ref.get("path")), role, repo_root=repo_root)
            if (
                entry["sha256"] != ref.get("sha256")
                or entry["size"] != ref.get("size")
            ):
                raise TransferContractError(
                    f"recording level campaign #{index} {field} SHA/size 불일치"
                )
            files.append(entry)
    source_selections = additions.get("source_selection", {})
    if not isinstance(source_selections, dict):
        raise TransferContractError("recorded source selection summary가 mapping이 아닙니다")
    selection_kinds = (
        "external_dns_speech_selection",
        "external_demand_environment_selection",
    )
    for selection_kind in selection_kinds:
        source_selection = source_selections.get(selection_kind)
        if source_selection is None:
            continue
        if not isinstance(source_selection, dict) or not isinstance(
            source_selection.get("bundle_files"), list
        ):
            raise TransferContractError(
                f"recorded {selection_kind} bundle summary가 유효하지 않습니다"
            )
        for index, ref in enumerate(source_selection["bundle_files"]):
            if (
                not isinstance(ref, dict)
                or set(ref) != {"path", "sha256", "size"}
                or not isinstance(ref.get("path"), str)
            ):
                raise TransferContractError(
                    f"recorded {selection_kind} bundle ref #{index}가 유효하지 않습니다"
                )
            entry = _entry(
                str(ref["path"]),
                "recorded_source_selection",
                repo_root=repo_root,
            )
            if entry["sha256"] != ref.get("sha256") or entry["size"] != ref.get(
                "size"
            ):
                raise TransferContractError(
                    f"recorded {selection_kind} bundle ref #{index} path/SHA/size 불일치"
                )
            files.append(entry)
    if len({str(item["path"]) for item in files}) != len(files):
        raise TransferContractError("schema v2 transfer files에 중복 path가 있습니다")
    files.sort(key=lambda item: str(item["path"]))
    recorded_entries = [item for item in files if item["role"] == "recorded"]
    parent_recorded = legacy["recorded"]
    if not isinstance(parent_recorded, dict):
        raise TransferContractError("legacy parent recorded aggregate가 없습니다")
    return {
        "schema_version": 2,
        "files": files,
        "recorded": {
            "root": "recorded_generation",
            "parent_root": "data/recorded",
            "additions_root": additions_root,
            "generation_manifest": generation_relative,
            "session_count": COMBINED_SESSION_COUNT,
            "file_count": len(recorded_entries),
            "total_bytes": sum(int(entry["size"]) for entry in recorded_entries),
            "aggregate_sha256": _canonical_recorded_aggregate(recorded_entries),
            "source_metadata_file_count": parent_recorded[
                "source_metadata_file_count"
            ],
            "source_metadata_snapshot_sha256": parent_recorded[
                "source_metadata_snapshot_sha256"
            ],
            "source_content_snapshot_sha256": parent_recorded[
                "source_content_snapshot_sha256"
            ],
        },
        "rir_bank": legacy["rir_bank"],
        "recorded_manifest": combined_manifest,
        "recorded_generation": generation_relative,
        "recorded_level_calibration": calibration_relative,
        "recording_level_campaigns": level_campaigns,
        "lineage_tracks": legacy["lineage_tracks"],
        "librispeech_chapters_metadata": legacy[
            "librispeech_chapters_metadata"
        ],
        "esc50_metadata": legacy["esc50_metadata"],
        "canonical_provenance": legacy["canonical_provenance"],
        "strict_ps": legacy["strict_ps"],
        **(
            {"level_meter": legacy["level_meter"]}
            if "level_meter" in legacy
            else {}
        ),
    }


def build_payload(args: argparse.Namespace, *, repo_root: Path) -> dict[str, object]:
    if getattr(args, "recorded_generation", None):
        return _build_payload_v2(args, repo_root=repo_root)
    return _build_payload_v1(args, repo_root=repo_root)


def _manifest_schema(data: bytes, *, label: str) -> int:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransferContractError(f"{label} JSON을 읽을 수 없습니다") from exc
    schema = payload.get("schema_version") if isinstance(payload, dict) else None
    if type(schema) is not int or schema not in {1, 2}:
        raise TransferContractError(f"{label} schema_version은 1 또는 2여야 합니다")
    return schema


def _archive_existing_manifest(
    path: Path,
    *,
    snapshot_sha256: str,
    snapshot_data: bytes,
    repo_root: Path,
) -> Path:
    """기존 canonical manifest를 content-addressed hard-link로 먼저 보존한다."""

    history = repo_root / TRANSFER_HISTORY_DIR
    reject_symlink_components(history.parent, root=repo_root)
    history.mkdir(mode=0o755, parents=False, exist_ok=True)
    reject_symlink_components(history, root=repo_root)
    archive = history / f"elice_transfer_manifest.{snapshot_sha256}.json"
    if archive.exists():
        existing_archive = read_regular_file_snapshot(
            archive,
            root=repo_root,
            label="existing archived transfer manifest",
        )
        if (
            existing_archive.sha256 != snapshot_sha256
            or existing_archive.data != snapshot_data
        ):
            raise TransferContractError(
                "content-addressed transfer history bytes가 기존 manifest와 다릅니다"
            )
        return archive
    try:
        os.link(path, archive, follow_symlinks=False)
    except FileExistsError:
        raced = read_regular_file_snapshot(
            archive,
            root=repo_root,
            label="raced archived transfer manifest",
        )
        if raced.sha256 != snapshot_sha256 or raced.data != snapshot_data:
            raise TransferContractError(
                "transfer history 발행 race의 기존 bytes가 다릅니다"
            )
    directory_fd = os.open(history, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return archive


def _publish_no_replace(
    path: Path,
    data: bytes,
    *,
    repo_root: Path,
    rotate_existing_sha256: str | None = None,
) -> None:
    expected = repo_root / OUTPUT
    if Path(os.path.abspath(path)) != Path(os.path.abspath(expected)):
        raise TransferContractError(f"출력은 {OUTPUT}로 고정됩니다")
    reject_symlink_components(path.parent, root=repo_root)
    if rotate_existing_sha256 is not None:
        rotate_existing_sha256 = rotate_existing_sha256.lower()
        if not _SHA256_RE.fullmatch(rotate_existing_sha256):
            raise TransferContractError(
                "rotate-existing-transfer-sha256는 64자리 lowercase SHA-256이어야 합니다"
            )

    lock_path = path.parent / ".elice-transfer.publish.lock"
    reject_symlink_components(lock_path, root=repo_root, allow_missing_leaf=True)
    lock_descriptor = os.open(
        lock_path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
            raise TransferContractError("transfer publish lock은 regular file이어야 합니다")
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        if path.exists():
            existing = read_regular_file_snapshot(
                path, root=repo_root, label="existing transfer manifest"
            )
            assert existing.data is not None
            if existing.data == data:
                return
            if rotate_existing_sha256 is None:
                raise TransferContractError(
                    "기존 transfer manifest bytes가 다릅니다. 자동 overwrite하지 않습니다"
                )
            if existing.sha256 != rotate_existing_sha256:
                raise TransferContractError(
                    "기존 transfer manifest SHA가 명시한 회전 anchor와 다릅니다"
                )
            old_schema = _manifest_schema(
                existing.data, label="기존 transfer manifest"
            )
            new_schema = _manifest_schema(data, label="새 transfer manifest")
            if (old_schema, new_schema) != (1, 2):
                raise TransferContractError(
                    "transfer manifest 회전은 검증된 schema v1→v2 전이에만 허용됩니다"
                )
            validate_transfer_manifest(
                path,
                repo_root=repo_root,
                expected_sha256=existing.sha256,
            )
            _archive_existing_manifest(
                path,
                snapshot_sha256=existing.sha256,
                snapshot_data=existing.data,
                repo_root=repo_root,
            )
        elif rotate_existing_sha256 is not None:
            raise TransferContractError(
                "회전할 기존 canonical transfer manifest가 없습니다"
            )

        descriptor, temp_name = tempfile.mkstemp(
            prefix=".elice-transfer.", dir=path.parent
        )
        temporary = Path(temp_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if path.exists():
                # 새 bytes도 실제 bundle 전체를 통과한 뒤에만 canonical 이름을
                # 원자적으로 교체한다. 실패하면 v1 canonical과 history가 모두 남는다.
                validate_transfer_manifest(
                    temporary,
                    repo_root=repo_root,
                    expected_sha256=hashlib.sha256(data).hexdigest(),
                )
                current = read_regular_file_snapshot(
                    path, root=repo_root, label="pre-replace transfer manifest"
                )
                if current.sha256 != rotate_existing_sha256:
                    raise TransferContractError(
                        "transfer manifest가 회전 검증 뒤 변경됐습니다"
                    )
                os.replace(temporary, path)
            else:
                os.link(temporary, path, follow_symlinks=False)
            directory_fd = os.open(
                path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-root", default="data/recorded")
    parser.add_argument("--rir-bank", required=True)
    parser.add_argument("--strict-raw", action="append", required=True)
    parser.add_argument("--strict-analysis", action="append", required=True)
    parser.add_argument("--primary-npz", required=True)
    parser.add_argument("--secondary-npz", required=True)
    parser.add_argument(
        "--level-meter-raw",
        default=None,
        help="tracked measurement level evidence가 참조하는 PASS meter raw (기본: evidence에서 자동 발견)",
    )
    parser.add_argument(
        "--level-meter-receipt",
        default=None,
        help="PASS meter raw의 canonical .receipt.json (기본: raw sibling 자동 유도)",
    )
    parser.add_argument("--expected-holdout-sha256", required=True)
    parser.add_argument(
        "--recorded-manifest",
        default="data/manifests/recorded_regrouped.jsonl",
    )
    parser.add_argument(
        "--recorded-generation",
        default=None,
        help=(
            "parent 82 + additions 19를 봉인한 generation.json. 지정하면 transfer schema v2와 "
            "combined 101 manifest를 사용합니다"
        ),
    )
    parser.add_argument(
        "--recorded-level-calibration",
        default=None,
        help=(
            "schema-v2에 포함할 old82 train-only strict-P level calibration receipt. "
            "--recorded-generation 사용 시 필수"
        ),
    )
    parser.add_argument(
        "--lineage-tracks",
        default="data/raw/music/fma_metadata/tracks.csv",
    )
    parser.add_argument(
        "--librispeech-chapters-metadata",
        default="data/raw/speech/LibriSpeech/CHAPTERS.TXT",
    )
    parser.add_argument(
        "--esc50-metadata",
        default="data/raw/noise/esc50/ESC-50-master/meta/esc50.csv",
    )
    parser.add_argument("--out", default=OUTPUT)
    parser.add_argument(
        "--rotate-existing-transfer-sha256",
        default=None,
        help=(
            "기존 schema-v1 canonical manifest를 content-addressed history로 보존하고 "
            "schema-v2로 원자 전환할 때 요구되는 기존 파일 SHA-256"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_payload(args, repo_root=REPO_ROOT)
        data = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        output = REPO_ROOT / args.out
        _publish_no_replace(
            output,
            data,
            repo_root=REPO_ROOT,
            rotate_existing_sha256=args.rotate_existing_transfer_sha256,
        )
        digest = hashlib.sha256(data).hexdigest()
        summary = validate_transfer_manifest(
            output,
            repo_root=REPO_ROOT,
            expected_sha256=digest,
        )
    except (OSError, HoldoutContractError, TransferContractError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print(
        f"transfer manifest: {output}\n"
        f"sha256: {digest}\n"
        f"files: {summary['file_count']}, recorded_sessions: {summary['recorded_session_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
