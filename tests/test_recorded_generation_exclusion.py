from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from deep_anc.data.holdout_contract import read_regular_file_snapshot
from deep_anc.data.recorded_generation import (
    ADDITION_SESSION_COUNT,
    SOURCE_PLAN_FIELDS,
    _derive_librispeech_identity_component_map,
)
from deep_anc.data.recorded_generation_exclusion import (
    RecordedGenerationExclusionError,
    derive_recorded_generation_exclusion,
    find_recorded_generation_overlaps,
    generation_excluded_basenames,
    generation_excluded_public_groups,
    validate_recorded_generation_exclusion,
)
from deep_anc.data.public_lineage import parse_librispeech_chapters_bytes


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _summary_fixture(root: Path) -> tuple[dict, dict]:
    generation = root / "data/manifests/recorded_generations/highband-v1/generation.json"
    generation.parent.mkdir(parents=True)
    generation.write_text('{"fixture":true}\n', encoding="utf-8")
    plan = root / "data/source_plans/recorded_additions/highband-v1.csv"
    plan.parent.mkdir(parents=True)

    chapters = root / "data/raw/speech/LibriSpeech/CHAPTERS.TXT"
    chapters.parent.mkdir(parents=True)
    chapters.write_text("100 | 7 | 0 | 0 | 0 | 900\n", encoding="utf-8")
    by_identity, _ = _derive_librispeech_identity_component_map(
        parse_librispeech_chapters_bytes(chapters.read_bytes())
    )
    libri_component = by_identity["librivox_reader:7"]

    rows: list[dict[str, str]] = []
    sessions: list[dict] = []
    for index in range(ADDITION_SESSION_COUNT):
        number = index + 2
        source_path = f"data/source_pool_v2/speech/speech_{index:03d}.wav"
        source_kind = "source_pool_row"
        raw_path = ""
        raw_sha = ""
        raw_lineage = ""
        authority = [f"clip_identity:speech_{index:03d}.wav"]
        if index == 0:
            source_kind = "external_librispeech_file"
            source_path = (
                "data/raw/speech/LibriSpeech/dev-clean/7/100/7-100-0001.flac"
            )
            raw_path = source_path
            raw_sha = _sha("external-libri-raw")
            raw_lineage = libri_component
            authority = [
                "clip_identity:7-100-0001.flac",
                f"librispeech_component:{libri_component}",
            ]
        source_sha = raw_sha or _sha(f"source-{index}")
        row = {field: "" for field in SOURCE_PLAN_FIELDS}
        row.update(
            {
                "source_kind": source_kind,
                "path": source_path,
                "seconds": "15.0",
                "start_seconds": "0.0",
                "source_family": "speech",
                "group_id": f"speech-group-{index}",
                "lineage_key": f"speech-lineage-{index}",
                "split": "train",
                "source_file_sha256": source_sha,
                "raw_member_path": raw_path,
                "raw_member_sha256": raw_sha,
                "raw_member_lineage_key": raw_lineage,
                "transform": "identity",
                "transform_repeat_count": "1",
            }
        )
        rows.append(row)
        sessions.append(
            {
                "source_row_number": number,
                "source_kind": source_kind,
                "source_family": "speech",
                "source_path": source_path,
                "raw_member_path": raw_path,
                "source_file_sha256": source_sha,
                "raw_member_sha256": raw_sha,
                "raw_member_lineage_key": raw_lineage,
                "authority_components": sorted(authority),
            }
        )
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    plan_snapshot = read_regular_file_snapshot(
        plan, root=root, label="fixture plan", capture_bytes=True
    )
    generation_snapshot = read_regular_file_snapshot(
        generation, root=root, label="fixture generation", capture_bytes=True
    )
    summary = {
        "generation_id": "highband-v1",
        "_validated_generation_snapshot": generation_snapshot,
        "additions": {
            "source_plan": {
                "path": plan.relative_to(root).as_posix(),
                "sha256": plan_snapshot.sha256,
                "size": plan_snapshot.size,
            },
            "sessions": sessions,
        },
    }
    return summary, {"libri_component": libri_component}


def test_external_libri_content_and_transitive_lineage_are_generation_identities(
    tmp_path: Path,
) -> None:
    summary, fixture = _summary_fixture(tmp_path)
    evidence = derive_recorded_generation_exclusion(summary, repo_root=tmp_path)

    identity = evidence["identities"][0]
    assert identity["source_file_sha256"] == _sha("external-libri-raw")
    assert identity["raw_member_sha256"] == _sha("external-libri-raw")
    assert (
        f"librispeech_component:{fixture['libri_component']}"
        in identity["authority_components"]
    )

    # 파일명과 bytes가 달라도 같은 reader/book transitive component면 누수다.
    entries = {
        "speech": [
            {
                "path": "/public/7-100-9999.flac",
                "content_sha256": _sha("different-public-bytes"),
                "lineage_keys": ["gutenberg_book:900", "librivox_reader:7"],
                "group_id": "public-lineage-fixture",
                "split": "train",
            }
        ]
    }
    overlaps = find_recorded_generation_overlaps(
        evidence, entries, repo_root=tmp_path
    )
    assert len(overlaps) == 1
    assert "authority_component" in overlaps[0]["dimensions"]

    evidence["identities"][1]["authority_components"].append(
        "fma_identity:fma_artist:42"
    )
    music_overlaps = find_recorded_generation_overlaps(
        evidence,
        {
            "music": [
                {
                    "path": "/public/different-track.mp3",
                    "content_sha256": _sha("different-music"),
                    "lineage_keys": ["fma_artist:42", "fma_album:999"],
                    "group_id": "public-lineage-music",
                    "split": "train",
                }
            ]
        },
        repo_root=tmp_path,
    )
    assert music_overlaps[0]["dimensions"] == ["authority_component"]


def test_generation_exclusion_rejects_sidecar_identity_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary, _fixture = _summary_fixture(tmp_path)
    evidence = derive_recorded_generation_exclusion(summary, repo_root=tmp_path)
    monkeypatch.setattr(
        "deep_anc.data.recorded_generation_exclusion.validate_recorded_generation",
        lambda *args, **kwargs: summary,
    )

    assert (
        validate_recorded_generation_exclusion(evidence, repo_root=tmp_path)
        == evidence
    )
    tampered = {**evidence, "identities": [dict(row) for row in evidence["identities"]]}
    tampered["identities"][0]["raw_member_sha256"] = _sha("forged")
    tampered["identities_sha256"] = hashlib.sha256(
        json.dumps(
            tampered["identities"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(RecordedGenerationExclusionError, match="재유도한 값"):
        validate_recorded_generation_exclusion(tampered, repo_root=tmp_path)


def test_dns_public_group_authority_excludes_every_component_member(tmp_path: Path):
    summary, _fixture = _summary_fixture(tmp_path)
    evidence = derive_recorded_generation_exclusion(summary, repo_root=tmp_path)
    public_group = "public-lineage-" + "a" * 64
    evidence["identities"][0]["authority_components"].append(
        f"public_group:{public_group}"
    )
    evidence["identities"][0]["authority_components"].sort()

    assert generation_excluded_public_groups(evidence) == {public_group}
    overlaps = find_recorded_generation_overlaps(
        evidence,
        {
            "speech": [
                {
                    # 선택 파일과 basename/content가 달라도 같은 public DSU
                    # component member면 반드시 제외된다.
                    "path": "/public/another-member.wav",
                    "content_sha256": _sha("different-member"),
                    "lineage_keys": ["dns_book:123", "dns_reader:456"],
                    "group_id": public_group,
                    "split": "train",
                }
            ]
        },
        repo_root=tmp_path,
    )
    assert len(overlaps) == 1
    assert overlaps[0]["dimensions"] == ["authority_component"]


def test_demand_unique_basename_excludes_only_one_public_group_of_16(tmp_path: Path):
    summary, _fixture = _summary_fixture(tmp_path)
    evidence = derive_recorded_generation_exclusion(summary, repo_root=tmp_path)
    public_group = "public-lineage-" + "d" * 64
    unique = "environment-demand-dkitchen-ch01-deadbeef0000.wav"
    identity = evidence["identities"][0]
    identity["source_path"] = f"data/immutable/{unique}"
    identity["raw_member_path"] = f"data/immutable/{unique}"
    identity["authority_components"] = [
        "public_lineage_key:demand_environment:DKITCHEN",
        f"public_group:{public_group}",
    ]
    assert generation_excluded_basenames(evidence) == {
        unique.casefold(),
        *{
            Path(row["source_path"]).name.casefold()
            for row in evidence["identities"][1:]
        },
    }
    assert "ch01.wav" not in generation_excluded_basenames(evidence)
    assert generation_excluded_public_groups(evidence) == {public_group}
    public_rows = [
        {
            "path": f"/public/DKITCHEN/ch{index:02d}.wav",
            "content_sha256": _sha(f"demand-{index}"),
            "lineage_keys": ["demand_environment:DKITCHEN"],
            "group_id": public_group,
            "split": "train",
        }
        for index in range(1, 17)
    ]
    overlaps = find_recorded_generation_overlaps(
        evidence, {"demand": public_rows}, repo_root=tmp_path
    )
    assert len(overlaps) == 16
    assert {tuple(item["dimensions"]) for item in overlaps} == {
        ("authority_component",)
    }
