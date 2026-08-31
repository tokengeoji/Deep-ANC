"""Stage-2 2 kHz 47-slot recorded-additions의 무출력 source-plan 계약.

이 모듈은 source WAV를 고르거나, 오디오 장치를 열거나, session/generation을
발행하지 않는다. 현재 82세션 actual-bytes population audit가 산출한 최소
47개 독립 component slot을 CSV bytes 수준에서만 고정한다. 따라서 이 validator의
PASS는 physical P/S, source-gain, 실제 ERR/coherence 또는 ANC 성능 PASS가 아니다.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping

from deep_anc.data.manifest import validate_group_id, validate_source_family
from deep_anc.data.recorded_generation import SOURCE_PLAN_FIELDS


STAGE2_2KHZ_RECORDED_GENERATION_ID = "stage2-2khz-47slot-v1"
"""Stage-1 19-row generation과 절대 섞지 않는 Stage-2 plan 식별자."""

STAGE2_2KHZ_ADDITION_SESSION_COUNT = 47
STAGE2_2KHZ_SPLITS = ("train", "val", "test")
STAGE2_2KHZ_FAMILIES = ("speech", "music", "environment", "machine")
STAGE2_2KHZ_REQUIRED_OBJECTIVE_HZ = (125, 250, 500, 1000, 2000)
STAGE2_2KHZ_SENTINEL_CENTER_HZ = 1600

# `stage2_2khz_population_20260831_v3.json`의 deterministic deficit-layering
# 결과를 코드 상수로 결속한다. 값은 신규 *독립 component* 수이며 session 반복으로
# 채울 수 없다.
STAGE2_2KHZ_DEFICITS: dict[tuple[str, str], dict[int, int]] = {
    ("train", "speech"): {125: 1, 250: 0, 500: 0, 1000: 3, 2000: 4},
    ("train", "music"): {125: 0, 250: 0, 500: 0, 1000: 1, 2000: 4},
    ("train", "environment"): {125: 0, 250: 0, 500: 0, 1000: 0, 2000: 4},
    ("train", "machine"): {125: 0, 250: 0, 500: 0, 1000: 1, 2000: 4},
    ("val", "speech"): {125: 2, 250: 0, 500: 0, 1000: 3, 2000: 4},
    ("val", "music"): {125: 1, 250: 0, 500: 0, 1000: 4, 2000: 4},
    ("val", "environment"): {125: 2, 250: 0, 500: 1, 1000: 1, 2000: 3},
    ("val", "machine"): {125: 1, 250: 0, 500: 0, 1000: 1, 2000: 4},
    ("test", "speech"): {125: 2, 250: 0, 500: 0, 1000: 3, 2000: 4},
    ("test", "music"): {125: 0, 250: 0, 500: 0, 1000: 3, 2000: 4},
    ("test", "environment"): {125: 0, 250: 0, 500: 0, 1000: 3, 2000: 4},
    ("test", "machine"): {125: 1, 250: 0, 500: 0, 1000: 4, 2000: 4},
}

# Sentinel 부족은 2 kHz 부족과 같은 47개 slot에 겹쳐야 하며, 별도 녹음 수로
# 부풀릴 수 없다.
STAGE2_2KHZ_SENTINEL_DEFICITS: dict[tuple[str, str], int] = {
    key: max(value.values()) for key, value in STAGE2_2KHZ_DEFICITS.items()
}

STAGE2_2KHZ_SOURCE_PLAN_EXTRA_FIELDS = (
    "stage2_slot_id",
    "stage2_required_objective_octaves_hz",
    "stage2_one_point_six_khz_sentinel_required",
    "stage2_conditioning_allowed",
    "stage2_untouched_natural_unseen_required",
    "stage2_training_or_model_selection_use_allowed",
)
STAGE2_2KHZ_SOURCE_PLAN_FIELDS = (
    *SOURCE_PLAN_FIELDS,
    *STAGE2_2KHZ_SOURCE_PLAN_EXTRA_FIELDS,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Stage2TwoKhzRecordedAdditionsError(ValueError):
    """Stage-2 47-slot source-plan/generation 식별자가 계약을 위반했다."""


@dataclass(frozen=True)
class Stage2TwoKhzSlot:
    """하나의 새 독립 component가 동시에 충족해야 할 최소 조건."""

    slot_id: str
    split: str
    source_family: str
    required_objective_octaves_hz: tuple[int, ...]
    one_point_six_khz_sentinel_required: bool
    conditioning_allowed: bool
    untouched_natural_unseen_required: bool
    training_or_model_selection_use_allowed: bool


def validate_stage2_2khz_generation_id(value: object) -> str:
    """오직 Stage-2 47-slot generation ID만 허용한다."""

    if value != STAGE2_2KHZ_RECORDED_GENERATION_ID:
        raise Stage2TwoKhzRecordedAdditionsError(
            "Stage-2 2 kHz canonical source plan generation-id가 아닙니다: "
            f"expected={STAGE2_2KHZ_RECORDED_GENERATION_ID!r}, actual={value!r}"
        )
    return STAGE2_2KHZ_RECORDED_GENERATION_ID


def stage2_2khz_required_slots() -> tuple[Stage2TwoKhzSlot, ...]:
    """audit의 deterministic deficit-layering slot을 재현한다."""

    slots: list[Stage2TwoKhzSlot] = []
    for split in STAGE2_2KHZ_SPLITS:
        for family in STAGE2_2KHZ_FAMILIES:
            deficits = STAGE2_2KHZ_DEFICITS[(split, family)]
            sentinel_deficit = STAGE2_2KHZ_SENTINEL_DEFICITS[(split, family)]
            slot_count = max((*deficits.values(), sentinel_deficit), default=0)
            for ordinal in range(1, slot_count + 1):
                required = tuple(
                    center
                    for center in STAGE2_2KHZ_REQUIRED_OBJECTIVE_HZ
                    if deficits[center] >= ordinal
                )
                slots.append(
                    Stage2TwoKhzSlot(
                        slot_id=f"{split}-{family}-new-lineage-{ordinal:02d}",
                        split=split,
                        source_family=family,
                        required_objective_octaves_hz=required,
                        one_point_six_khz_sentinel_required=sentinel_deficit >= ordinal,
                        conditioning_allowed=split == "train",
                        untouched_natural_unseen_required=split == "test",
                        training_or_model_selection_use_allowed=split != "test",
                    )
                )
    if len(slots) != STAGE2_2KHZ_ADDITION_SESSION_COUNT:
        raise RuntimeError("Stage-2 2 kHz 47-slot 상수 자체가 일관되지 않습니다")
    return tuple(slots)


def _canonical_octave_json(values: tuple[int, ...]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def _parse_exact_bool(value: object, *, field: str, row_number: int) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise Stage2TwoKhzRecordedAdditionsError(
        f"source plan {row_number}행 {field}는 lower-case true/false여야 합니다"
    )


def _validate_relative_source_path(value: object, *, row_number: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Stage2TwoKhzRecordedAdditionsError(
            f"source plan {row_number}행 path가 비어 있습니다"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise Stage2TwoKhzRecordedAdditionsError(
            f"source plan {row_number}행 path는 canonical 저장소 상대 POSIX 경로여야 합니다"
        )
    return path.as_posix()


def _validate_sha256(value: object, *, field: str, row_number: int) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Stage2TwoKhzRecordedAdditionsError(
            f"source plan {row_number}행 {field}는 64자리 소문자 SHA-256이어야 합니다"
        )
    return value


def _parse_exact_seconds(value: object, *, row_number: int) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise Stage2TwoKhzRecordedAdditionsError(
            f"source plan {row_number}행 seconds가 숫자가 아닙니다"
        ) from exc
    if not math.isfinite(seconds) or not math.isclose(seconds, 15.0, rel_tol=0.0, abs_tol=1e-9):
        raise Stage2TwoKhzRecordedAdditionsError(
            f"source plan {row_number}행은 exact 15초여야 합니다: {value!r}"
        )
    return seconds


def _parse_nonnegative_start(value: object, *, row_number: int) -> float:
    try:
        start = float(value)
    except (TypeError, ValueError) as exc:
        raise Stage2TwoKhzRecordedAdditionsError(
            f"source plan {row_number}행 start_seconds가 숫자가 아닙니다"
        ) from exc
    if not math.isfinite(start) or start < 0.0:
        raise Stage2TwoKhzRecordedAdditionsError(
            f"source plan {row_number}행 start_seconds는 0 이상 finite여야 합니다"
        )
    return start


def validate_stage2_2khz_source_plan_bytes(
    source_plan_bytes: bytes,
    *,
    generation_id: object,
) -> dict[str, Any]:
    """47-row Stage-2 source-plan의 구조·slot·독립성 선언을 순수 검증한다.

    WAV 존재/decoder, parent82 connected-component closure, source-to-ERR joint
    density 및 P/S/gain authority는 의도적으로 이 함수의 범위 밖이다. 이 함수의
    성공을 live 녹음·generation 발행 허가로 해석하면 안 된다.
    """

    generation = validate_stage2_2khz_generation_id(generation_id)
    if not isinstance(source_plan_bytes, bytes):
        raise Stage2TwoKhzRecordedAdditionsError("source plan bytes가 아닙니다")
    try:
        text = source_plan_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Stage2TwoKhzRecordedAdditionsError("Stage-2 source plan UTF-8 오류") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != STAGE2_2KHZ_SOURCE_PLAN_FIELDS:
        raise Stage2TwoKhzRecordedAdditionsError(
            "Stage-2 source plan header는 exact 계약이어야 합니다: "
            f"expected={STAGE2_2KHZ_SOURCE_PLAN_FIELDS}, actual={tuple(reader.fieldnames or ())}"
        )
    raw_rows = list(reader)
    if len(raw_rows) != STAGE2_2KHZ_ADDITION_SESSION_COUNT:
        raise Stage2TwoKhzRecordedAdditionsError(
            "Stage-2 2 kHz source plan은 정확히 47개의 새 독립 component row가 필요합니다: "
            f"actual={len(raw_rows)}"
        )

    expected_by_slot = {slot.slot_id: slot for slot in stage2_2khz_required_slots()}
    observed_slots: set[str] = set()
    observed_groups: set[str] = set()
    observed_lineages: set[str] = set()
    observed_sources: set[str] = set()
    observed_source_hashes: set[str] = set()
    normalized_rows: list[dict[str, Any]] = []

    for row_number, row in enumerate(raw_rows, start=2):
        if None in row or set(row) != set(STAGE2_2KHZ_SOURCE_PLAN_FIELDS):
            raise Stage2TwoKhzRecordedAdditionsError(
                f"source plan {row_number}행 field가 header와 다릅니다"
            )
        slot_id = row["stage2_slot_id"]
        expected = expected_by_slot.get(slot_id)
        if expected is None or slot_id in observed_slots:
            raise Stage2TwoKhzRecordedAdditionsError(
                f"source plan {row_number}행 stage2_slot_id가 누락/중복/미지정입니다: {slot_id!r}"
            )
        observed_slots.add(slot_id)

        path = _validate_relative_source_path(row["path"], row_number=row_number)
        _parse_exact_seconds(row["seconds"], row_number=row_number)
        start_seconds = _parse_nonnegative_start(row["start_seconds"], row_number=row_number)
        try:
            family = validate_source_family(row["source_family"])
            group_id = validate_group_id(row["group_id"])
            lineage_key = validate_group_id(row["lineage_key"])
        except ValueError as exc:
            raise Stage2TwoKhzRecordedAdditionsError(
                f"source plan {row_number}행 family/group/lineage 오류: {exc}"
            ) from exc
        split = row["split"]
        if split not in STAGE2_2KHZ_SPLITS:
            raise Stage2TwoKhzRecordedAdditionsError(
                f"source plan {row_number}행 split이 Stage-2 계약 밖입니다: {split!r}"
            )
        if family != expected.source_family or split != expected.split:
            raise Stage2TwoKhzRecordedAdditionsError(
                f"source plan {row_number}행 family/split이 slot과 다릅니다: "
                f"slot={slot_id}, actual={family}/{split}"
            )
        source_sha = _validate_sha256(
            row["source_file_sha256"], field="source_file_sha256", row_number=row_number
        )
        if group_id in observed_groups or lineage_key in observed_lineages:
            raise Stage2TwoKhzRecordedAdditionsError(
                f"source plan {row_number}행은 신규 독립 component가 아닙니다: "
                f"group={group_id!r}, lineage={lineage_key!r}"
            )
        if path in observed_sources or source_sha in observed_source_hashes:
            raise Stage2TwoKhzRecordedAdditionsError(
                f"source plan {row_number}행은 같은 source path/content를 다른 slot에 재사용합니다"
            )
        observed_groups.add(group_id)
        observed_lineages.add(lineage_key)
        observed_sources.add(path)
        observed_source_hashes.add(source_sha)

        expected_octaves = _canonical_octave_json(expected.required_objective_octaves_hz)
        if row["stage2_required_objective_octaves_hz"] != expected_octaves:
            raise Stage2TwoKhzRecordedAdditionsError(
                f"source plan {row_number}행 objective octave가 slot과 다릅니다: "
                f"expected={expected_octaves}, actual={row['stage2_required_objective_octaves_hz']!r}"
            )
        if 2000 not in expected.required_objective_octaves_hz:
            raise RuntimeError("Stage-2 slot이 2 kHz objective를 잃었습니다")
        actual_sentinel = _parse_exact_bool(
            row["stage2_one_point_six_khz_sentinel_required"],
            field="stage2_one_point_six_khz_sentinel_required",
            row_number=row_number,
        )
        if actual_sentinel is not expected.one_point_six_khz_sentinel_required or not actual_sentinel:
            raise Stage2TwoKhzRecordedAdditionsError(
                f"source plan {row_number}행은 1.6 kHz sentinel을 mandatory true로 결속해야 합니다"
            )
        checks = {
            "stage2_conditioning_allowed": expected.conditioning_allowed,
            "stage2_untouched_natural_unseen_required": expected.untouched_natural_unseen_required,
            "stage2_training_or_model_selection_use_allowed": expected.training_or_model_selection_use_allowed,
        }
        for field, expected_value in checks.items():
            if _parse_exact_bool(row[field], field=field, row_number=row_number) is not expected_value:
                raise Stage2TwoKhzRecordedAdditionsError(
                    f"source plan {row_number}행 {field}가 split policy와 다릅니다"
                )
        normalized_rows.append(
            {
                "source_row_number": row_number,
                "slot_id": slot_id,
                "path": path,
                "seconds": 15.0,
                "start_seconds": start_seconds,
                "source_family": family,
                "group_id": group_id,
                "lineage_key": lineage_key,
                "split": split,
                "source_file_sha256": source_sha,
                "required_objective_octaves_hz": list(expected.required_objective_octaves_hz),
                "one_point_six_khz_sentinel_required": True,
                "conditioning_allowed": expected.conditioning_allowed,
                "untouched_natural_unseen_required": expected.untouched_natural_unseen_required,
                "training_or_model_selection_use_allowed": expected.training_or_model_selection_use_allowed,
            }
        )

    if observed_slots != set(expected_by_slot):
        missing = sorted(set(expected_by_slot) - observed_slots)
        raise Stage2TwoKhzRecordedAdditionsError(
            f"Stage-2 source plan slot이 완전하지 않습니다: missing={missing}"
        )
    split_counts = {
        split: sum(row["split"] == split for row in normalized_rows)
        for split in STAGE2_2KHZ_SPLITS
    }
    expected_split_counts = {"train": 16, "val": 15, "test": 16}
    if split_counts != expected_split_counts:
        raise RuntimeError("Stage-2 split slot 상수 자체가 일관되지 않습니다")
    return {
        "generation_id": generation,
        "source_plan_sha256": hashlib.sha256(source_plan_bytes).hexdigest(),
        "source_plan_row_count": len(normalized_rows),
        "split_counts": split_counts,
        "all_slots_require_2000_hz": True,
        "all_slots_require_1600_hz_sentinel": True,
        "rows": normalized_rows,
    }
