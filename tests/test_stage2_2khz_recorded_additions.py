"""Stage-2 2 kHz 47-slot recorded-additions preflight 계약."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from deep_anc.config import REPO_ROOT
from deep_anc.data.recorded_generation import (
    RecordedGenerationError,
    validate_generation_id,
    validate_stage2_2khz_generation_id,
    validate_stage2_2khz_recorded_additions_source_plan_bytes,
)
from deep_anc.data.stage2_2khz_recorded_additions import (
    STAGE2_2KHZ_ADDITION_SESSION_COUNT,
    STAGE2_2KHZ_RECORDED_GENERATION_ID,
    STAGE2_2KHZ_SOURCE_PLAN_FIELDS,
    Stage2TwoKhzRecordedAdditionsError,
    stage2_2khz_required_slots,
    validate_stage2_2khz_source_plan_bytes,
)


def _load_batch_module():
    name = "record_session_batch_stage2_2khz_test"
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts/data/record_session_batch.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rows(*, source_hashes: dict[str, str] | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for slot in stage2_2khz_required_slots():
        rows.append(
            {
                "source_kind": "source_pool_row",
                "path": f"data/stage2_sources/{slot.slot_id}.wav",
                "seconds": "15.0",
                "start_seconds": "0.0",
                "source_family": slot.source_family,
                "group_id": f"group-{slot.slot_id}",
                "lineage_key": f"lineage-{slot.slot_id}",
                "split": slot.split,
                "source_file_sha256": (
                    source_hashes[slot.slot_id]
                    if source_hashes is not None
                    else _sha(slot.slot_id)
                ),
                "raw_member_path": "",
                "raw_member_sha256": "",
                "raw_member_lineage_key": "",
                "authority_metadata_sha256": "",
                "inventory_path": "",
                "inventory_sha256": "",
                "transform": "identity",
                "transform_repeat_count": "1",
                "stage2_slot_id": slot.slot_id,
                "stage2_required_objective_octaves_hz": (
                    "[" + ",".join(str(value) for value in slot.required_objective_octaves_hz) + "]"
                ),
                "stage2_one_point_six_khz_sentinel_required": "true",
                "stage2_conditioning_allowed": str(slot.conditioning_allowed).lower(),
                "stage2_untouched_natural_unseen_required": str(
                    slot.untouched_natural_unseen_required
                ).lower(),
                "stage2_training_or_model_selection_use_allowed": str(
                    slot.training_or_model_selection_use_allowed
                ).lower(),
            }
        )
    return rows


def _plan_bytes(rows: list[dict[str, str]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=STAGE2_2KHZ_SOURCE_PLAN_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def test_stage2_slot_constant_binds_all_47_to_two_khz_and_sentinel() -> None:
    slots = stage2_2khz_required_slots()
    assert len(slots) == STAGE2_2KHZ_ADDITION_SESSION_COUNT == 47
    assert {slot.split for slot in slots} == {"train", "val", "test"}
    assert sum(slot.split == "train" for slot in slots) == 16
    assert sum(slot.split == "val" for slot in slots) == 15
    assert sum(slot.split == "test" for slot in slots) == 16
    assert all(2000 in slot.required_objective_octaves_hz for slot in slots)
    assert all(slot.one_point_six_khz_sentinel_required for slot in slots)
    assert sum(slot.conditioning_allowed for slot in slots) == 16
    assert sum(slot.untouched_natural_unseen_required for slot in slots) == 16
    assert not any(
        slot.training_or_model_selection_use_allowed for slot in slots if slot.split == "test"
    )


def test_stage2_source_plan_accepts_exact_47_slot_contract() -> None:
    payload = _plan_bytes(_rows())
    result = validate_stage2_2khz_source_plan_bytes(
        payload, generation_id=STAGE2_2KHZ_RECORDED_GENERATION_ID
    )
    assert result["source_plan_row_count"] == 47
    assert result["split_counts"] == {"train": 16, "val": 15, "test": 16}
    assert result["all_slots_require_2000_hz"] is True
    assert result["all_slots_require_1600_hz_sentinel"] is True
    # recorded_generation의 public wrapper도 동일한 Stage-2 preflight만 허용한다.
    assert validate_stage2_2khz_recorded_additions_source_plan_bytes(
        payload, generation_id=STAGE2_2KHZ_RECORDED_GENERATION_ID
    ) == result


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("stage2_required_objective_octaves_hz", "[125,1000]", "objective octave"),
        (
            "stage2_one_point_six_khz_sentinel_required",
            "false",
            "1.6 kHz sentinel",
        ),
        ("stage2_conditioning_allowed", "false", "split policy"),
    ],
)
def test_stage2_source_plan_rejects_weakened_slot_contract(
    field: str, value: str, needle: str
) -> None:
    rows = _rows()
    rows[0][field] = value
    with pytest.raises(Stage2TwoKhzRecordedAdditionsError, match=needle):
        validate_stage2_2khz_source_plan_bytes(
            _plan_bytes(rows), generation_id=STAGE2_2KHZ_RECORDED_GENERATION_ID
        )


def test_stage2_source_plan_rejects_reused_lineage_or_content() -> None:
    rows = _rows()
    rows[1]["lineage_key"] = rows[0]["lineage_key"]
    with pytest.raises(Stage2TwoKhzRecordedAdditionsError, match="독립 component"):
        validate_stage2_2khz_source_plan_bytes(
            _plan_bytes(rows), generation_id=STAGE2_2KHZ_RECORDED_GENERATION_ID
        )

    rows = _rows()
    rows[1]["source_file_sha256"] = rows[0]["source_file_sha256"]
    with pytest.raises(Stage2TwoKhzRecordedAdditionsError, match="재사용"):
        validate_stage2_2khz_source_plan_bytes(
            _plan_bytes(rows), generation_id=STAGE2_2KHZ_RECORDED_GENERATION_ID
        )


def test_stage1_generation_id_path_remains_separate() -> None:
    assert validate_stage2_2khz_generation_id(
        STAGE2_2KHZ_RECORDED_GENERATION_ID
    ) == STAGE2_2KHZ_RECORDED_GENERATION_ID
    with pytest.raises(RecordedGenerationError):
        validate_generation_id(STAGE2_2KHZ_RECORDED_GENERATION_ID)
    with pytest.raises(Stage2TwoKhzRecordedAdditionsError):
        validate_stage2_2khz_generation_id("stage1-coverage-v4-gainprobe006")


def test_batch_stage2_preflight_is_dry_run_only_and_never_opens_audio(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    batch = _load_batch_module()
    monkeypatch.setattr(batch, "REPO_ROOT", tmp_path)
    source_hashes: dict[str, str] = {}
    for index, slot in enumerate(stage2_2khz_required_slots()):
        source = tmp_path / "data/stage2_sources" / f"{slot.slot_id}.wav"
        source.parent.mkdir(parents=True, exist_ok=True)
        # 15초 window를 실제 decoder가 읽을 수 있게만 만든다. 각 source는 서로
        # 다른 bytes여야 하며 오디오 출력은 없다.
        sf.write(
            source,
            np.full(1_500, (index + 1) / 1_000.0, dtype=np.float32),
            100,
            subtype="FLOAT",
        )
        source_hashes[slot.slot_id] = hashlib.sha256(source.read_bytes()).hexdigest()
    plan = (
        tmp_path
        / batch.SOURCE_PLAN_ROOT
        / f"{STAGE2_2KHZ_RECORDED_GENERATION_ID}.csv"
    )
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_bytes(_plan_bytes(_rows(source_hashes=source_hashes)))
    out_root = tmp_path / batch.ADDITIONS_ROOT / STAGE2_2KHZ_RECORDED_GENERATION_ID

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Stage-2 dry-run이 child/audio process를 실행했습니다")

    monkeypatch.setattr(batch.subprocess, "Popen", forbidden)
    assert batch.main(
        [
            "--sources",
            str(plan),
            "--out-root",
            str(out_root),
            "--stage2-2khz-canonical-generation",
            STAGE2_2KHZ_RECORDED_GENERATION_ID,
            "--dry-run",
        ]
    ) == 0
    assert not out_root.exists()
    assert "Stage-2 2 kHz preflight PASS" in capsys.readouterr().out

    with pytest.raises(SystemExit) as excinfo:
        batch.main(
            [
                "--sources",
                str(plan),
                "--out-root",
                str(out_root),
                "--stage2-2khz-canonical-generation",
                STAGE2_2KHZ_RECORDED_GENERATION_ID,
                "--confirm-user-present",
                "--confirm-volume-minimum",
                "--confirm-routing-and-geometry",
            ]
        )
    assert excinfo.value.code == 2
    assert not out_root.exists()
