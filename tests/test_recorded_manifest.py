"""실측 매니페스트의 이식성·그룹 분할·메타데이터 회귀 테스트."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from deep_anc.data.manifest import (
    MANIFEST_PATH_BASE,
    assign_splits,
    read_manifest,
    validate_group_splits,
    validate_source_family,
    write_manifest,
)
from scripts.data.make_recorded_manifest import build_recorded_entries


def test_manifest_resolves_only_explicit_manifest_relative_paths(tmp_path):
    manifest = tmp_path / "data" / "manifests" / "recorded.jsonl"
    session = tmp_path / "data" / "recorded" / "session-a"
    session.mkdir(parents=True)
    absolute_legacy = tmp_path / "legacy-absolute"

    write_manifest(
        [
            {
                "path": "../recorded/session-a",
                "path_base": MANIFEST_PATH_BASE,
                "split": "train",
            },
            {"path": "legacy-relative/session-b", "split": "val"},
            {"path": str(absolute_legacy), "split": "test"},
        ],
        manifest,
    )

    entries = read_manifest(manifest)

    assert entries[0]["path"] == str(session.resolve())
    # marker가 없는 legacy 경로는 CWD 의미를 바꾸지 않는다.
    assert entries[1]["path"] == "legacy-relative/session-b"
    assert entries[2]["path"] == str(absolute_legacy)


def test_manifest_relative_marker_rejects_absolute_path(tmp_path):
    with pytest.raises(ValueError, match="상대 경로"):
        write_manifest(
            [
                {
                    "path": str((tmp_path / "session").resolve()),
                    "path_base": MANIFEST_PATH_BASE,
                    "split": "train",
                }
            ],
            tmp_path / "manifest.jsonl",
        )


def test_group_split_is_leak_free_deterministic_and_input_order_independent():
    entries = [
        {
            "path": f"session-{group}-{repeat}",
            "group_id": f"speaker-{group}",
            "source_family": "speech",
        }
        for group in range(8)
        for repeat in range(2)
    ]
    ratios = {"train": 0.5, "val": 0.25, "test": 0.25}

    first = assign_splits(entries, ratios, seed=73)
    second = assign_splits(list(reversed(entries)), ratios, seed=73)

    def group_map(items):
        mapping: dict[str, set[str]] = {}
        for item in items:
            mapping.setdefault(item["group_id"], set()).add(item["split"])
        return mapping

    first_map = group_map(first)
    second_map = group_map(second)
    assert all(len(splits) == 1 for splits in first_map.values())
    assert {key: next(iter(value)) for key, value in first_map.items()} == {
        key: next(iter(value)) for key, value in second_map.items()
    }
    assert [entry.get("split") for entry in entries] == [None] * len(entries)


def test_source_family_stratification_covers_all_splits_when_groups_suffice():
    entries = [
        {
            "path": f"{family}-{group}-{repeat}",
            "group_id": f"{family}-group-{group:02d}",
            "source_family": family,
        }
        for family in ("speech", "music", "environment")
        for group in range(10)
        for repeat in range(2)
    ]
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

    assigned = assign_splits(
        entries,
        ratios,
        seed=20260803,
        group_key="group_id",
        stratify_key="source_family",
    )
    repeated = assign_splits(
        list(reversed(entries)),
        ratios,
        seed=20260803,
        group_key="group_id",
        stratify_key="source_family",
    )

    mapping = {entry["group_id"]: entry["split"] for entry in assigned}
    assert mapping == {entry["group_id"]: entry["split"] for entry in repeated}
    for family in ("speech", "music", "environment"):
        family_groups = {
            entry["group_id"]: entry["split"]
            for entry in assigned
            if entry["source_family"] == family
        }
        counts = {
            split: sum(value == split for value in family_groups.values())
            for split in ("train", "val", "test")
        }
        assert counts == {"train": 8, "val": 1, "test": 1}


def test_source_family_stratification_minimum_coverage_for_three_groups():
    entries = [
        {
            "path": f"speech-{group}",
            "group_id": f"speaker-{group}",
            "source_family": "speech",
        }
        for group in range(3)
    ]
    assigned = assign_splits(
        entries,
        {"train": 0.8, "val": 0.1, "test": 0.1},
        seed=7,
        stratify_key="source_family",
    )

    assert {entry["split"] for entry in assigned} == {"train", "val", "test"}


def test_partial_group_metadata_is_rejected():
    with pytest.raises(ValueError, match="모든 항목"):
        assign_splits(
            [{"path": "a", "group_id": "g-a"}, {"path": "b"}],
            {"train": 0.8, "val": 0.1, "test": 0.1},
            seed=1,
        )


def test_grouped_entries_cannot_disable_group_assignment():
    with pytest.raises(ValueError, match="비활성화"):
        assign_splits(
            [
                {"path": "a", "group_id": "same"},
                {"path": "b", "group_id": "same"},
            ],
            {"train": 0.5, "val": 0.0, "test": 0.5},
            seed=1,
            group_key=None,
        )


def test_manifest_rejects_group_cross_split_on_write_and_read(tmp_path):
    entries = [
        {"path": "session-a", "group_id": "same-source", "split": "train"},
        {"path": "session-b", "group_id": "same-source", "split": "test"},
    ]
    manifest = tmp_path / "leaky.jsonl"

    with pytest.raises(ValueError, match="여러 split"):
        write_manifest(entries, manifest)

    manifest.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="여러 split"):
        read_manifest(manifest, split="train")


def test_manifest_rejects_conflicting_source_family_within_group(tmp_path):
    with pytest.raises(ValueError, match="source_family가 일관되지"):
        write_manifest(
            [
                {
                    "path": "session-a",
                    "group_id": "same-source",
                    "source_family": "speech",
                    "split": "train",
                },
                {
                    "path": "session-b",
                    "group_id": "same-source",
                    "source_family": "music",
                    "split": "train",
                },
            ],
            tmp_path / "inconsistent.jsonl",
        )


@pytest.mark.parametrize("value", ["", " speech", "speech/music", "bad\nfamily"])
def test_source_family_validation_rejects_ambiguous_ids(value):
    with pytest.raises(ValueError, match="source_family"):
        validate_source_family(value)


def _write_session(session: Path, metadata: dict | None) -> None:
    session.mkdir(parents=True)
    sf.write(session / "mics.wav", np.zeros((480, 2), dtype=np.float32), 48_000)
    if metadata is not None:
        (session / "session.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )


def test_recorded_manifest_builder_adds_portable_paths_and_legacy_metadata(tmp_path):
    data_root = tmp_path / "portable-data"
    recorded = data_root / "recorded"
    manifest = data_root / "manifests" / "recorded.jsonl"
    # 계열마다 test 하한(G4 MIN_GROUPS_PER_FAMILY) + train 최소 1 을 만족해야 하므로
    # 계열당 5 그룹을 만든다. speaker-001 은 **두 세션이 한 그룹**이라 그룹 원자성을
    # 검사하는 자리이고, legacy-session 은 메타데이터 승격을 검사하는 자리다.
    shared = {"group_id": "speaker-001", "source_family": "speech"}
    _write_session(recorded / "session-a", shared)
    _write_session(recorded / "session-b", shared)
    for index in range(2, 6):
        _write_session(
            recorded / f"session-speech-{index}",
            {"group_id": f"speaker-{index:03d}", "source_family": "speech"},
        )
    _write_session(
        recorded / "legacy-session",
        {"program": {"type": "music"}},
    )
    for index in range(1, 5):
        _write_session(
            recorded / f"session-music-{index}",
            {"group_id": f"album-{index:03d}", "source_family": "music"},
        )

    entries = build_recorded_entries(
        recorded,
        manifest,
        seed=19,
        ratios={"train": 0.5, "val": 0.0, "test": 0.5},
    )

    assert len(entries) == 11
    assert all(entry["path_base"] == MANIFEST_PATH_BASE for entry in entries)
    assert all(not Path(entry["path"]).is_absolute() for entry in entries)
    assert {entry["source_family"] for entry in entries} == {"speech", "music"}
    shared_splits = {
        entry["split"] for entry in entries if entry["group_id"] == "speaker-001"
    }
    assert len(shared_splits) == 1
    legacy = next(entry for entry in entries if entry["session_id"] == "legacy-session")
    assert legacy["group_id"] == "legacy-session"
    assert set(legacy["metadata_inferred"]) == {
        "session_id",
        "group_id",
        "source_family",
    }

    write_manifest(entries, manifest)
    loaded = read_manifest(manifest)
    assert {Path(entry["path"]).name for entry in loaded} >= {
        "session-a",
        "session-b",
        "legacy-session",
    }
    assert len(loaded) == len(entries)
    assert all(Path(entry["path"]).is_absolute() for entry in loaded)

    moved_root = tmp_path / "moved-data"
    shutil.copytree(data_root, moved_root)
    moved = read_manifest(moved_root / "manifests" / "recorded.jsonl")
    assert all(Path(entry["path"]).is_relative_to(moved_root) for entry in moved)


def _stratified_entries(groups_per_family: dict[str, int]) -> list[dict]:
    return [
        {
            "path": f"{family}-{group}-{repeat}",
            "group_id": f"{family}-group-{group:02d}",
            "source_family": family,
        }
        for family, count in groups_per_family.items()
        for group in range(count)
        for repeat in range(2)
    ]


def test_group_floor_beats_the_ratio_when_the_ratio_would_starve_val_and_test():
    """8:1:1 비율만으로는 G4 가 요구하는 계열당 4 그룹이 절대 안 나온다.

    실측 근거(2026-08-06): 실측 80세션/64그룹을 8:1:1 로 나누면 val·test 가 계열당
    1~2 그룹이었다. 비율로 4 를 얻으려면 계열당 40 그룹(=160세션)이 필요하고 그건
    스피커 시간 2배다. 그래서 하한을 먼저 확보한다.
    """
    from deep_anc.eval.recorded import MIN_GROUPS_PER_FAMILY

    entries = _stratified_entries({f: 20 for f in ("speech", "music", "environment")})
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

    without_floor = assign_splits(
        entries, ratios, seed=20260803, group_key="group_id", stratify_key="source_family"
    )
    with_floor = assign_splits(
        entries,
        ratios,
        seed=20260803,
        group_key="group_id",
        stratify_key="source_family",
        min_units_per_split={"val": MIN_GROUPS_PER_FAMILY, "test": MIN_GROUPS_PER_FAMILY},
    )

    def group_counts(assigned: list[dict], family: str) -> dict[str, int]:
        mapping = {
            entry["group_id"]: entry["split"]
            for entry in assigned
            if entry["source_family"] == family
        }
        return {
            split: sum(value == split for value in mapping.values())
            for split in ("train", "val", "test")
        }

    for family in ("speech", "music", "environment"):
        # 비율만 쓰면 하한 미달 — 이것이 고치려는 상태다.
        assert group_counts(without_floor, family) == {"train": 16, "val": 2, "test": 2}
        # 하한을 걸면 만족하고, 남는 것은 train 으로 간다.
        counts = group_counts(with_floor, family)
        assert counts["val"] >= MIN_GROUPS_PER_FAMILY
        assert counts["test"] >= MIN_GROUPS_PER_FAMILY
        assert sum(counts.values()) == 20
        assert counts["train"] >= 1


def test_group_floor_fails_closed_when_a_family_has_too_few_groups():
    """만족시킬 수 없으면 조용히 적게 주지 않고 실패한다.

    실측: machine 계열은 재정렬 후 8 그룹뿐이라 4+4+1 = 9 를 만족할 수 없다.
    조용히 통과시키면 그 사실은 학습이 끝난 뒤 G4 판정 불가로만 드러난다.
    """
    entries = _stratified_entries({"speech": 20, "machine": 8})
    with pytest.raises(ValueError, match=r"machine.*8 개뿐"):
        assign_splits(
            entries,
            {"train": 0.8, "val": 0.1, "test": 0.1},
            seed=20260803,
            group_key="group_id",
            stratify_key="source_family",
            min_units_per_split={"val": 4, "test": 4},
        )


def test_group_floor_keeps_splits_leak_free_and_deterministic():
    """하한을 걸어도 그룹은 여전히 원자 단위이고 입력 순서에 무관해야 한다."""
    entries = _stratified_entries({"speech": 12, "music": 15})
    kwargs = dict(
        seed=20260803,
        group_key="group_id",
        stratify_key="source_family",
        min_units_per_split={"val": 4, "test": 4},
    )
    forward = assign_splits(entries, {"train": 0.8, "val": 0.1, "test": 0.1}, **kwargs)
    backward = assign_splits(
        list(reversed(entries)), {"train": 0.8, "val": 0.1, "test": 0.1}, **kwargs
    )
    mapping = {entry["group_id"]: entry["split"] for entry in forward}
    assert mapping == {entry["group_id"]: entry["split"] for entry in backward}
    validate_group_splits(forward)


def test_manifest_builder_refuses_to_lower_the_g4_group_floor(tmp_path):
    """manifest 쪽에서 게이트 하한을 낮추는 것을 구조적으로 막는다 (발생기 A)."""
    from deep_anc.eval.recorded import MIN_GROUPS_PER_FAMILY

    with pytest.raises(ValueError, match=r"강화 방향"):
        build_recorded_entries(
            tmp_path / "nonexistent",
            tmp_path / "m.jsonl",
            min_groups_per_split=MIN_GROUPS_PER_FAMILY - 1,
        )
