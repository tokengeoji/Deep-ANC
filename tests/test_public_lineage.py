"""공개 코퍼스 authoritative lineage와 component 원자성 계약."""

from __future__ import annotations

from pathlib import Path

import pytest

import deep_anc.data.public_lineage as public_lineage

from deep_anc.data.public_lineage import (
    ESC50_METADATA,
    FMA_TRACKS,
    LIBRISPEECH_CHAPTERS,
    PUBLIC_LINEAGE_SCHEMA,
    PublicLineageBlocked,
    PublicLineageError,
    build_public_lineage,
    canonical_json_sha256,
    demand_lineage_keys,
    dns_audioset_lineage_keys,
    dns_speech_lineage_keys,
    esc50_lineage_keys,
    fma_lineage_keys,
    librispeech_lineage_keys,
    mimii_lineage_keys,
    parse_esc50_metadata_bytes,
    parse_fma_tracks_bytes,
    parse_librispeech_chapters_bytes,
    validate_public_manifest_lineage,
)


def _holdout(*, family: str = "environment", keys: list[str] | None = None) -> dict:
    rows = [
        {
            "family": family,
            "clip": "recorded.wav",
            "content_sha256": "f" * 64,
            "lineage_keys": keys or ["recorded:source"],
        }
    ]
    return {
        "schema_version": 1,
        "metadata": {
            "librispeech_chapters": {
                "path": LIBRISPEECH_CHAPTERS,
                "sha256": "a" * 64,
                "size": 1,
            },
            "fma_tracks": {"path": FMA_TRACKS, "sha256": "b" * 64, "size": 1},
            "esc50": {"path": ESC50_METADATA, "sha256": "c" * 64, "size": 1},
        },
        "clips": rows,
        "clips_sha256": canonical_json_sha256(rows),
    }


def _write_fma(root: Path) -> None:
    path = root / FMA_TRACKS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "track,artist,album\n"
        "id,id,id\n"
        "1,10,20\n"
        "2,10,21\n"
        "3,30,31\n",
        encoding="utf-8",
    )


def test_librispeech_chapters_uses_reader_and_gutenberg_book_not_chapter() -> None:
    chapters = parse_librispeech_chapters_bytes(
        b"; comment\n149897 | 2277 | 12.3 | train-clean-100 | 999 | 5267\n"
    )
    assert librispeech_lineage_keys("2277-149897-0001.flac", chapters) == (
        "librivox_reader:2277",
        "gutenberg_book:5267",
    )
    with pytest.raises(PublicLineageError, match="speaker.*reader"):
        librispeech_lineage_keys("9999-149897-0001.flac", chapters)


def test_public_dsu_handles_adversarial_reverse_merge_chain_without_recursion() -> None:
    """identity merge 순서가 역순이어도 large component를 안전하게 닫는다."""

    values = [f"node-{index:05d}" for index in range(2_048)]
    dsu = public_lineage._DisjointSet(values)
    for index in range(len(values) - 1, 0, -1):
        dsu.union(values[index], values[index - 1])

    # 구 구현은 마지막 find에서 재귀 깊이가 1,000을 넘어 RecursionError가 났다.
    assert {dsu.find(value) for value in values} == {dsu.find(values[0])}


def test_fma_and_esc_metadata_parsers_expose_official_source_relationships() -> None:
    tracks = parse_fma_tracks_bytes(
        b"track,artist,album\nid,id,id\n1,10,20\n2,10,21\n"
    )
    assert fma_lineage_keys("000001.mp3", tracks) == (
        "fma_artist:10",
        "fma_album:20",
    )
    esc = parse_esc50_metadata_bytes(
        b"filename,fold,target,category,esc10,src_file,take\n"
        b"1-1-A-1.wav,1,1,dog,True,777,1\n"
        b"1-2-B-1.wav,1,1,dog,True,777,2\n"
    )
    assert esc50_lineage_keys("1-1-a-1.wav", esc) == ("esc50_src:777",)
    assert esc50_lineage_keys("1-2-B-1.wav", esc) == ("esc50_src:777",)


def test_demand_mimii_and_dns_path_parsers_are_strict(tmp_path: Path) -> None:
    demand = tmp_path / "demand"
    assert demand_lineage_keys(
        demand / "DKITCHEN/ch01.wav", tag_root=demand
    ) == ("demand_environment:DKITCHEN",)
    with pytest.raises(PublicLineageBlocked):
        demand_lineage_keys(demand / "UNKNOWN/ch01.wav", tag_root=demand)

    machine = tmp_path / "machine"
    assert mimii_lineage_keys(
        machine / "6_dB/fan/id_00/normal/00000000.wav", tag_root=machine
    ) == ("mimii_fan_machine:00",)
    assert mimii_lineage_keys(
        machine / "fan/test/section_00_source_test_anomaly_0000_m-n_W.wav",
        tag_root=machine,
    ) == ("mimii_dg_fan_section:00",)
    assert mimii_lineage_keys(
        machine / "fan/train/section_00_target_train_normal_0001_m-n_Z.wav",
        tag_root=machine,
    ) == ("mimii_dg_fan_section:00",)
    with pytest.raises(PublicLineageBlocked):
        mimii_lineage_keys(machine / "6_dB/fan/normal.wav", tag_root=machine)

    assert dns_speech_lineage_keys("book_12_chp_3_reader_4.wav") == (
        "dns_reader:4",
        "dns_book:12",
    )
    assert dns_audioset_lineage_keys("AbCdEf_123-.wav") == (
        "audioset_video:AbCdEf_123-",
    )
    with pytest.raises(PublicLineageBlocked):
        dns_audioset_lineage_keys("not-an-official-id.wav")


def test_public_dsu_excludes_whole_lineage_and_content_components(tmp_path: Path) -> None:
    _write_fma(tmp_path)
    music_root = tmp_path / "data/raw/music"
    entries = {
        "music": [
            {"path": str(music_root / "fma_small/000/000001.mp3"), "content_sha256": "1" * 64},
            {"path": str(music_root / "fma_small/000/000002.mp3"), "content_sha256": "2" * 64},
            {"path": str(music_root / "fma_small/000/000003.mp3"), "content_sha256": "f" * 64},
        ]
    }
    # 1/2는 같은 artist component, 3은 basename이 달라도 holdout content SHA와 같다.
    # 전부 빠지면 빈 학습 pool을 만드는 대신 명시적으로 BLOCKED다.
    with pytest.raises(PublicLineageBlocked, match="모든 component"):
        build_public_lineage(
            entries,
            tag_roots={"music": [music_root]},
            repo_root=tmp_path,
            holdout_lineage=_holdout(family="music", keys=["fma_artist:10"]),
        )


def test_public_dsu_keeps_independent_component_and_assigns_one_group(tmp_path: Path) -> None:
    _write_fma(tmp_path)
    music_root = tmp_path / "data/raw/music"
    entries = {
        "music": [
            {"path": str(music_root / "fma_small/000/000001.mp3"), "content_sha256": "1" * 64},
            {"path": str(music_root / "fma_small/000/000002.mp3"), "content_sha256": "2" * 64},
            {"path": str(music_root / "fma_small/000/000003.mp3"), "content_sha256": "3" * 64},
        ]
    }
    built = build_public_lineage(
        entries,
        tag_roots={"music": [music_root]},
        repo_root=tmp_path,
        holdout_lineage=_holdout(family="music", keys=["fma_artist:10"]),
    )
    kept = built.entries_by_tag["music"]
    assert [Path(row["path"]).stem for row in kept] == ["000003"]
    assert built.excluded_by_tag == {"music": 2}
    assert kept[0]["lineage_schema"] == PUBLIC_LINEAGE_SCHEMA


def test_public_dsu_excludes_recorded_source_pool_basename(tmp_path: Path) -> None:
    _write_fma(tmp_path)
    music_root = tmp_path / "data/raw/music"
    entries = {
        "music": [
            {"path": str(music_root / "fma_small/000/000003.mp3"), "content_sha256": "3" * 64},
        ]
    }
    with pytest.raises(PublicLineageBlocked, match="모든 component"):
        build_public_lineage(
            entries,
            tag_roots={"music": [music_root]},
            repo_root=tmp_path,
            holdout_lineage=_holdout(family="music", keys=["unrelated:source"]),
            extra_excluded_basenames=["000003.mp3"],
        )


def test_dns_speech_uses_explicit_namespace_without_crosswalk(tmp_path: Path) -> None:
    root = tmp_path / "data/raw/noise/speech"
    root.mkdir(parents=True)
    marker = tmp_path / "data/raw/noise/speech000.tar.bz2.extracted"
    marker.write_text("book_12_chp_3_reader_4.wav\n", encoding="utf-8")
    built = build_public_lineage(
        {
            "speech": [
                {
                    "path": str(root / "book_12_chp_3_reader_4.wav"),
                    "content_sha256": "1" * 64,
                }
            ]
        },
        tag_roots={"speech": [root]},
        repo_root=tmp_path,
        holdout_lineage=_holdout(family="speech"),
    )
    assert built.entries_by_tag["speech"][0]["lineage_keys"] == [
        "dns_book:12",
        "dns_reader:4",
    ]
    assert built.evidence["crosswalk_policy"] == {
        "dns_read_speech_to_librispeech": "namespace_disjoint_no_official_crosswalk",
        "cross_namespace_overlap_checks": ["content_sha256", "basename"],
    }


def test_missing_authoritative_metadata_is_blocked(tmp_path: Path) -> None:
    music_root = tmp_path / "data/raw/music"
    music_root.mkdir(parents=True)
    with pytest.raises(PublicLineageBlocked, match="metadata"):
        build_public_lineage(
            {
                "music": [
                    {
                        "path": str(music_root / "fma_small/000/000001.mp3"),
                        "content_sha256": "1" * 64,
                    }
                ]
            },
            tag_roots={"music": [music_root]},
            repo_root=tmp_path,
            holdout_lineage=_holdout(),
        )


def test_manifest_rejects_component_or_identity_split_crossing() -> None:
    base = {
        "path": "/raw/a.wav",
        "content_sha256": "1" * 64,
        "lineage_schema": PUBLIC_LINEAGE_SCHEMA,
        "lineage_keys": ["source:one"],
        "group_id": "public-lineage-" + "1" * 64,
    }
    with pytest.raises(PublicLineageError, match="split"):
        validate_public_manifest_lineage(
            {
                "esc50": [
                    {**base, "split": "train"},
                    {**base, "path": "/raw/b.wav", "split": "val"},
                ]
            }
        )
