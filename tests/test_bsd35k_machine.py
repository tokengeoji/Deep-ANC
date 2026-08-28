from __future__ import annotations

import copy
import hashlib
import json

import pytest

from deep_anc.data import bsd35k_machine as bsd


def _row(sound_id: int, uploader: str, *, license_url: str = bsd.CC0_LICENSE_URL):
    return {
        "sound_id": str(sound_id),
        "class": "fx-m",
        "class_idx": "404",
        "class_top": "fx",
        "confidence": "",
        "uploader": uploader,
        "license": license_url,
        "title": f"machine {sound_id}",
        "tags": "machine,motor",
        "description": "fixture",
    }


def _fixture_plan():
    rows = []
    sound_id = 1
    # 큰 component가 train으로 가고도 val/test 각각 16 uploader를 강제해야 한다.
    counts = (30, 9, 8, 7, 6, 5, 4, 3) + (1,) * 40
    for uploader_index, count in enumerate(counts):
        for _ in range(count):
            rows.append(_row(sound_id, f"uploader-{uploader_index:02d}"))
            sound_id += 1
    rows.append(
        _row(
            sound_id,
            "excluded-by",
            license_url="https://creativecommons.org/licenses/by/4.0/",
        )
    )
    metadata_sha = "a" * 64
    return bsd._build_selection_payload(
        metadata_size=123,
        metadata_sha256=metadata_sha,
        rows=rows,
        expected_row_count=len(rows),
        expected_fx_m_count=len(rows),
        expected_cc0_count=len(rows) - 1,
        expected_uploader_count=len(counts),
    )


def test_split_is_uploader_disjoint_deterministic_and_minimum_four():
    first = _fixture_plan()
    second = _fixture_plan()
    assert first == second
    by_split = {
        split: {
            row["uploader"] for row in first["entries"] if row["split"] == split
        }
        for split in bsd.SPLITS
    }
    assert all(
        len(uploaders) >= bsd.MINIMUM_UPLOADERS_PER_SPLIT
        for uploaders in by_split.values()
    )
    assert by_split["train"].isdisjoint(by_split["val"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["val"].isdisjoint(by_split["test"])
    assert {row["license"] for row in first["entries"]} == {bsd.CC0_LICENSE_URL}
    assert {row["uploader"] for row in first["entries"]}.isdisjoint({"excluded-by"})
    assert first["authority"]["canonical_source_eligible"] is False


def test_duplicate_sound_id_is_rejected():
    rows = [_row(1, f"uploader-{index:02d}") for index in range(48)]
    with pytest.raises(ValueError, match="중복 sound_id"):
        bsd._build_selection_payload(
            metadata_size=123,
            metadata_sha256="a" * 64,
            rows=rows,
            expected_row_count=len(rows),
            expected_fx_m_count=len(rows),
            expected_cc0_count=len(rows),
            expected_uploader_count=48,
        )


def test_official_builder_rejects_nonofficial_bytes(tmp_path):
    path = tmp_path / "metadata.csv"
    path.write_text(",".join(bsd.REQUIRED_COLUMNS) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="size"):
        bsd.build_official_bsd35k_machine_selection(path)


def test_validator_rejects_tampered_official_shape_before_claim():
    # Fixture payload의 internal SHA 검증 자체를 먼저 확인한다. 공식 validator는 그 뒤
    # official count/bytes까지 추가로 닫는다.
    plan = _fixture_plan()
    tampered = copy.deepcopy(plan)
    tampered["entries"][0]["split"] = "test"
    with pytest.raises(ValueError, match="plan SHA"):
        bsd.validate_bsd35k_machine_selection(tampered)


def test_exclusive_writer_rejects_existing_and_symlink_parent(tmp_path, monkeypatch):
    plan = _fixture_plan()
    # writer의 official validator만 fixture 전용 구조 검증으로 대체한다. O_EXCL/fsync와
    # symlink 방어가 selection 내용과 독립이라는 것을 검사한다.
    def validate_fixture(value):
        payload = dict(value)
        claimed = payload.pop("selection_plan_sha256")
        assert claimed == bsd._json_sha256(payload)

    monkeypatch.setattr(bsd, "validate_bsd35k_machine_selection", validate_fixture)
    target = tmp_path / "plans" / "selection.json"
    path, digest = bsd.write_bsd35k_machine_selection_exclusive(target, plan)
    assert path == target
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    assert json.loads(target.read_text(encoding="utf-8"))["authority"][
        "canonical_source_eligible"
    ] is False
    with pytest.raises(FileExistsError):
        bsd.write_bsd35k_machine_selection_exclusive(target, plan)

    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        bsd.write_bsd35k_machine_selection_exclusive(alias / "plan.json", plan)
