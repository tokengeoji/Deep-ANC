from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deep_anc.train.stage2_2khz_pretrain_admission import (
    _validate_recorded_synthetic_lineage,
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_holdout(root: Path) -> tuple[Path, str, str]:
    clips = [
        {
            "family": "speech",
            "clip": "recorded-source.wav",
            "content_sha256": "a" * 64,
            "lineage_keys": ["recorded_lineage:one"],
        }
    ]
    clips_sha = hashlib.sha256(_canonical_json(clips)).hexdigest()
    payload = {
        "clip_lineage": {
            "schema_version": 1,
            "metadata": {
                "librispeech_chapters": {
                    "path": "data/raw/speech/LibriSpeech/CHAPTERS.TXT",
                    "sha256": "1" * 64,
                    "size": 1,
                },
                "fma_tracks": {
                    "path": "data/raw/music/fma_metadata/tracks.csv",
                    "sha256": "2" * 64,
                    "size": 1,
                },
                "esc50": {
                    "path": "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv",
                    "sha256": "3" * 64,
                    "size": 1,
                },
            },
            "clips": clips,
            "clips_sha256": clips_sha,
        },
        "families": {"speech": ["recorded-source.wav"]},
    }
    path = root / "data/manifests/recorded_holdout.json"
    path.parent.mkdir(parents=True)
    raw = _canonical_json(payload)
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest(), clips_sha


def _receipt(holdout_sha: str, clips_sha: str) -> dict[str, object]:
    return {
        "recorded_holdout": {
            "path": "data/manifests/recorded_holdout.json",
            "sha256": holdout_sha,
        },
        "recorded_clip_count": 1,
        "recorded_clip_lineage_sha256": clips_sha,
        "recorded_synthetic_intersection_algorithm": (
            "transitive_basename_content_sha256_lineage_keys_v1"
        ),
        "recorded_synthetic_lineage_intersection_count": 0,
        "actual_recorded_holdout_bytes_consumed": True,
    }


def _manifest_item() -> dict[str, object]:
    return {
        "split": "train",
        "path": "data/public/unseen-source.wav",
        "content_sha256": "b" * 64,
        "lineage_keys": ["public_lineage:one"],
    }


@pytest.mark.parametrize("overlap_axis", ["basename", "content_sha256", "lineage_keys"])
def test_forged_zero_receipt_cannot_hide_actual_recorded_synthetic_overlap(
    tmp_path: Path,
    overlap_axis: str,
) -> None:
    _, holdout_sha, clips_sha = _write_holdout(tmp_path)
    item = _manifest_item()
    if overlap_axis == "basename":
        item["path"] = "data/public/recorded-source.wav"
    elif overlap_axis == "content_sha256":
        item["content_sha256"] = "a" * 64
    else:
        item["lineage_keys"] = ["recorded_lineage:one"]

    with pytest.raises(ValueError, match="actual recorded/synthetic 교집합"):
        _validate_recorded_synthetic_lineage(
            tmp_path,
            manifest_payload={"items": [item]},
            lineage_receipt=_receipt(holdout_sha, clips_sha),
        )


def test_recorded_holdout_file_sha_is_rechecked_before_lineage_admission(
    tmp_path: Path,
) -> None:
    path, holdout_sha, clips_sha = _write_holdout(tmp_path)
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="bytes SHA"):
        _validate_recorded_synthetic_lineage(
            tmp_path,
            manifest_payload={"items": [_manifest_item()]},
            lineage_receipt=_receipt(holdout_sha, clips_sha),
        )


def test_disjoint_actual_holdout_and_manifest_pass_recomputation(tmp_path: Path) -> None:
    _, holdout_sha, clips_sha = _write_holdout(tmp_path)

    _validate_recorded_synthetic_lineage(
        tmp_path,
        manifest_payload={"items": [_manifest_item()]},
        lineage_receipt=_receipt(holdout_sha, clips_sha),
    )
