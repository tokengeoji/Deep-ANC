#!/usr/bin/env python3
"""Jetson의 immutable Elice 학습 입력을 canonical transfer manifest로 결속한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


OUTPUT = "data/manifests/elice_transfer_manifest.json"


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


def build_payload(args: argparse.Namespace, *, repo_root: Path) -> dict[str, object]:
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
    if len({value for value, _role in fixed}) != len(fixed):
        raise TransferContractError("서로 다른 transfer role이 같은 파일을 가리킵니다")
    entries.extend(_entry(value, role, repo_root=repo_root) for value, role in fixed)
    entries.sort(key=lambda item: str(item["path"]))
    recorded_entries = [entry for entry in entries if entry["role"] == "recorded"]
    return {
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


def _publish_no_replace(path: Path, data: bytes, *, repo_root: Path) -> None:
    expected = repo_root / OUTPUT
    if Path(os.path.abspath(path)) != Path(os.path.abspath(expected)):
        raise TransferContractError(f"출력은 {OUTPUT}로 고정됩니다")
    reject_symlink_components(path.parent, root=repo_root)
    if path.exists():
        existing = read_regular_file_snapshot(
            path, root=repo_root, label="existing transfer manifest"
        )
        if existing.data != data:
            raise TransferContractError(
                "기존 transfer manifest bytes가 다릅니다. 자동 overwrite하지 않습니다"
            )
        return
    descriptor, temp_name = tempfile.mkstemp(prefix=".elice-transfer.", dir=path.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-root", default="data/recorded")
    parser.add_argument("--rir-bank", required=True)
    parser.add_argument("--strict-raw", action="append", required=True)
    parser.add_argument("--strict-analysis", action="append", required=True)
    parser.add_argument("--primary-npz", required=True)
    parser.add_argument("--secondary-npz", required=True)
    parser.add_argument("--expected-holdout-sha256", required=True)
    parser.add_argument(
        "--recorded-manifest",
        default="data/manifests/recorded_regrouped.jsonl",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_payload(args, repo_root=REPO_ROOT)
        data = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        output = REPO_ROOT / args.out
        _publish_no_replace(output, data, repo_root=REPO_ROOT)
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
