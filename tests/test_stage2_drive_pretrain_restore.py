from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deep_anc.data.stage2_drive_pretrain_restore import (
    RestoreCohort,
    Stage2DriveAuditError,
    build_stage2_drive_audit_from_evidence,
    build_stage2_drive_restore_anchor,
    parse_snapshot_sha_manifest,
    verify_local_stage2_partial_restore,
    write_json_exclusive,
)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _fixture():
    files = {
        "data/manifests/esc50.jsonl": b"esc-manifest\n",
        "data/manifests/music.jsonl": b"music-manifest\n",
        "data/manifests/speech.jsonl": b"speech-manifest\n",
        "data/raw/music/fma_small/a.mp3": b"music",
        "data/raw/noise/esc50/audio/a.wav": b"environment",
        "data/raw/speech/LibriSpeech/a.flac": b"speech",
    }
    manifest = "".join(
        f"{_sha(content)}  {path}\n" for path, content in sorted(files.items())
    ).encode("utf-8")
    listing = []
    for path, content in sorted(files.items()):
        listing.append(
            {
                "Path": path.removeprefix("data/"),
                "Size": len(content),
                "IsDir": False,
                "Hashes": {"sha256": _sha(content), "md5": "ignored"},
                "ID": "must-not-enter-receipt",
            }
        )
    cohorts = (
        RestoreCohort(
            "music",
            "data/raw/music/",
            1,
            len(files["data/raw/music/fma_small/a.mp3"]),
            ".mp3",
            1,
            len(files["data/raw/music/fma_small/a.mp3"]),
            "music",
        ),
        RestoreCohort(
            "speech",
            "data/raw/speech/",
            1,
            len(files["data/raw/speech/LibriSpeech/a.flac"]),
            ".flac",
            1,
            len(files["data/raw/speech/LibriSpeech/a.flac"]),
            "speech",
        ),
        RestoreCohort(
            "environment",
            "data/raw/noise/esc50/",
            1,
            len(files["data/raw/noise/esc50/audio/a.wav"]),
            ".wav",
            1,
            len(files["data/raw/noise/esc50/audio/a.wav"]),
            "environment",
        ),
    )
    return files, manifest, json.dumps(listing).encode("utf-8"), cohorts


def _audit_fixture():
    files, manifest, listing, cohorts = _fixture()
    audit = build_stage2_drive_audit_from_evidence(
        snapshot_remote_root="gdrive:DeepANC/snapshot",
        archive_cache_remote_root="gdrive:DeepANC/public_archive_cache",
        snapshot_manifest_bytes=manifest,
        snapshot_listing_bytes=listing,
        archive_cache_listing_bytes=None,
        archive_cache_query_returncode=3,
        expected_manifest_sha256=_sha(manifest),
        expected_snapshot_file_count=len(files),
        expected_snapshot_byte_count=sum(len(value) for value in files.values()),
        cohorts=cohorts,
    )
    return files, manifest, audit


def test_remote_snapshot_exact_but_public_pretrain_stays_blocked_without_archives() -> None:
    _files, _manifest, audit = _audit_fixture()
    assert audit["snapshot"]["status"] == (
        "PASS_EXACT_REMOTE_OBJECT_SET_AND_SHA256_METADATA"
    )
    assert audit["partial_restore"]["status"] == (
        "PASS_INPUT_ELIGIBLE_NOT_TRAINING_READY"
    )
    assert audit["official_fixed_archive_cache"]["status"] == (
        "BLOCKED_ARCHIVE_CACHE_ABSENT"
    )
    assert audit["public_synthetic_scratch_pretrain_readiness"]["status"] == (
        "BLOCKED"
    )
    assert audit["safety"]["remote_write_operations"] == 0
    assert "must-not-enter-receipt" not in json.dumps(audit)


def test_remote_listing_content_sha_mismatch_fails_closed() -> None:
    files, manifest, listing, cohorts = _fixture()
    rows = json.loads(listing)
    rows[0]["Hashes"]["sha256"] = "0" * 64
    with pytest.raises(Stage2DriveAuditError, match="mismatch"):
        build_stage2_drive_audit_from_evidence(
            snapshot_remote_root="gdrive:DeepANC/snapshot",
            archive_cache_remote_root="gdrive:DeepANC/cache",
            snapshot_manifest_bytes=manifest,
            snapshot_listing_bytes=json.dumps(rows).encode(),
            archive_cache_listing_bytes=None,
            archive_cache_query_returncode=3,
            expected_manifest_sha256=_sha(manifest),
            expected_snapshot_file_count=len(files),
            expected_snapshot_byte_count=sum(len(value) for value in files.values()),
            cohorts=cohorts,
        )


def test_snapshot_manifest_rejects_duplicate_and_traversal() -> None:
    digest = "0" * 64
    with pytest.raises(Stage2DriveAuditError, match="중복"):
        parse_snapshot_sha_manifest(
            f"{digest}  data/a\n{digest}  data/a\n".encode()
        )
    with pytest.raises(Stage2DriveAuditError, match="형식|traversal"):
        parse_snapshot_sha_manifest(f"{digest}  data/../secret\n".encode())


def test_anchor_is_small_and_preserves_partial_not_ready_boundary() -> None:
    _files, _manifest, audit = _audit_fixture()
    anchor = build_stage2_drive_restore_anchor(audit)
    assert anchor["stage2_public_pretrain_ready"] is False
    assert anchor["restore_file_count"] == 3
    assert anchor["fixed_archive_cache_status_at_audit"] == (
        "BLOCKED_ARCHIVE_CACHE_ABSENT"
    )
    assert len(json.dumps(anchor)) < 10_000


def _materialize_local_restore(tmp_path: Path):
    files, manifest, audit = _audit_fixture()
    root = tmp_path / "incoming"
    for relative, content in files.items():
        if not relative.startswith("data/raw/"):
            continue
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    manifest_path = tmp_path / "data_backup_manifest.sha256"
    manifest_path.write_bytes(manifest)
    anchor_path = tmp_path / "anchor.json"
    write_json_exclusive(build_stage2_drive_restore_anchor(audit), anchor_path)
    return root, manifest_path, anchor_path


def test_local_restore_rehashes_every_file_and_emits_partial_receipt(tmp_path: Path) -> None:
    root, manifest, anchor = _materialize_local_restore(tmp_path)
    receipt = verify_local_stage2_partial_restore(
        anchor_path=anchor,
        restore_root=root,
        snapshot_manifest_path=manifest,
    )
    assert receipt["status"] == "PASS_PARTIAL_RESTORE_ONLY"
    assert receipt["file_count"] == 3
    assert receipt["stage2_public_pretrain_ready"] is False


def test_local_restore_tamper_and_extra_file_fail_closed(tmp_path: Path) -> None:
    root, manifest, anchor = _materialize_local_restore(tmp_path)
    target = root / "data/raw/music/fma_small/a.mp3"
    target.write_bytes(b"tampered")
    with pytest.raises(Stage2DriveAuditError, match="content SHA|aggregate"):
        verify_local_stage2_partial_restore(
            anchor_path=anchor,
            restore_root=root,
            snapshot_manifest_path=manifest,
        )

    target.write_bytes(b"music")
    (target.parent / "extra.mp3").write_bytes(b"extra")
    with pytest.raises(Stage2DriveAuditError, match="path set"):
        verify_local_stage2_partial_restore(
            anchor_path=anchor,
            restore_root=root,
            snapshot_manifest_path=manifest,
        )


def test_local_restore_directory_symlink_fails_closed(tmp_path: Path) -> None:
    root, manifest, anchor = _materialize_local_restore(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "data/raw/music/link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(Stage2DriveAuditError, match="symlink"):
        verify_local_stage2_partial_restore(
            anchor_path=anchor,
            restore_root=root,
            snapshot_manifest_path=manifest,
        )
