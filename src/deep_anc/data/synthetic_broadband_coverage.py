"""Synthetic public-corpus의 광대역 native-Nyquist 계보 게이트.

48 kHz로 리샘플된 tensor가 있다는 사실은 원본이 8 kHz octave 전체를 담았다는
증거가 아니다. 이 모듈은 canonical public manifest의 원본 sample rate, content SHA,
lineage group과 split을 직접 검사한다. 실제 source spectral density와 덕트 target-d
coverage는 별도 raw-derived receipt가 필요하므로 이 결과만으로 학습 readiness를
PASS시키지는 않는다.

오디오 파일이나 장치를 열지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from ..dsp.control_band_contract import ControlBandContract


SYNTHETIC_BROADBAND_NATIVE_AUDIT_SCHEMA = (
    "synthetic_broadband_native_manifest_audit_v1"
)
REQUIRED_SPLITS = ("train", "val", "test")
MIN_INDEPENDENT_GROUPS_PER_SPLIT_FAMILY_BAND = 4

# built-in synthetic tone generator는 independent public lineage가 아니므로 여기서 세지
# 않는다. ESC-50 전체를 machine으로 중복 집계하는 것도 금지한다. 16 kHz MIMII fan만
# 있는 현재 machine tag는 8 kHz octave 상단을 통과할 수 없다는 사실을 그대로 드러낸다.
DEFAULT_FAMILY_TAGS: dict[str, tuple[str, ...]] = {
    "speech": ("speech",),
    "music": ("music",),
    "environment": ("dns_fullband", "demand", "esc50"),
    "machine": ("machine",),
}

_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in _HEX for character in text)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _normalise_family_tags(
    contract: ControlBandContract,
    family_tags: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    if set(family_tags) != set(contract.source_families):
        raise ValueError(
            "family tag mapping은 speech/music/environment/machine을 정확히 한 번씩 "
            "정의해야 합니다"
        )
    result: dict[str, tuple[str, ...]] = {}
    owner: dict[str, str] = {}
    for family in contract.source_families:
        tags = tuple(str(value).strip() for value in family_tags[family])
        if not tags or any(not value for value in tags):
            raise ValueError(f"{family} family의 public tag가 비었습니다")
        if len(set(tags)) != len(tags):
            raise ValueError(f"{family} family tag가 중복됩니다")
        for tag in tags:
            previous = owner.setdefault(tag, family)
            if previous != family:
                raise ValueError(
                    f"public tag {tag!r}를 {previous}/{family}에 중복 집계할 수 없습니다"
                )
        result[family] = tags
    if "synthetic" in owner:
        raise ValueError("built-in synthetic generator는 independent public group이 아닙니다")
    return result


def audit_synthetic_native_manifest_rows(
    entries_by_tag: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    contract: ControlBandContract,
    family_tags: Mapping[str, Sequence[str]] = DEFAULT_FAMILY_TAGS,
    minimum_groups: int = MIN_INDEPENDENT_GROUPS_PER_SPLIT_FAMILY_BAND,
) -> dict[str, Any]:
    """원본 sample rate와 lineage 기준 split×family×band coverage를 재집계한다.

    이 함수는 manifest metadata만 감사한다. canonical manifest 생성기가 decoder 전수
    audit로 header/sample-rate/content SHA를 결속한 뒤 사용해야 한다.
    """

    if contract.role != "broadband_point_control":
        raise ValueError("synthetic 광대역 감사에는 broadband contract가 필요합니다")
    floor = int(minimum_groups)
    if floor < MIN_INDEPENDENT_GROUPS_PER_SPLIT_FAMILY_BAND:
        raise ValueError("독립 group 하한을 4보다 낮출 수 없습니다")
    mapping = _normalise_family_tags(contract, family_tags)

    reasons: list[str] = []
    malformed: list[str] = []
    assignment_by_group: dict[str, tuple[str, str]] = {}
    group_by_content: dict[str, str] = {}
    # 동일 group의 여러 파일은 한 번만 세되, 그 group이 해당 band를 native로 덮으면 된다.
    covered_groups: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    row_counts: dict[str, int] = defaultdict(int)

    known_tags = {tag for tags in mapping.values() for tag in tags}
    missing_tags = sorted(tag for tag in known_tags if tag not in entries_by_tag)
    for tag in missing_tags:
        reasons.append(f"public manifest tag가 없습니다: {tag}")

    for family, tags in mapping.items():
        for tag in tags:
            for index, raw in enumerate(entries_by_tag.get(tag, ())):
                label = f"{tag}[{index}]"
                row_counts[tag] += 1
                row_tag = str(raw.get("tag", "")).strip()
                split = str(raw.get("split", "")).strip()
                group = str(raw.get("group_id", "")).strip()
                content_sha = str(raw.get("content_sha256", "")).strip()
                sample_rate_raw = raw.get("sample_rate")
                if row_tag != tag:
                    malformed.append(f"{label}: row tag가 manifest tag와 다릅니다")
                    continue
                if split not in REQUIRED_SPLITS:
                    malformed.append(f"{label}: split이 train/val/test가 아닙니다")
                    continue
                if not group:
                    malformed.append(f"{label}: canonical lineage group_id가 없습니다")
                    continue
                if not _is_sha256(content_sha):
                    malformed.append(f"{label}: lowercase content SHA-256이 없습니다")
                    continue
                if isinstance(sample_rate_raw, bool) or not isinstance(
                    sample_rate_raw, int
                ):
                    malformed.append(f"{label}: native sample_rate가 정수가 아닙니다")
                    continue
                sample_rate = int(sample_rate_raw)
                if sample_rate <= 0:
                    malformed.append(f"{label}: native sample_rate가 양수가 아닙니다")
                    continue

                assignment = (split, family)
                previous_assignment = assignment_by_group.setdefault(group, assignment)
                if previous_assignment != assignment:
                    malformed.append(
                        f"{label}: group {group!r}가 split/family를 넘나듭니다 "
                        f"({previous_assignment} -> {assignment})"
                    )
                    continue
                previous_group = group_by_content.setdefault(content_sha, group)
                if previous_group != group:
                    malformed.append(
                        f"{label}: 같은 content SHA가 여러 독립 group으로 갈라졌습니다"
                    )
                    continue

                native_nyquist = sample_rate / 2.0
                for band_index, band in enumerate(contract.point_control_subbands_hz):
                    if native_nyquist >= float(band[1]):
                        covered_groups[(split, family, band_index)].add(group)

    if malformed:
        reasons.extend(malformed)

    cells: list[dict[str, Any]] = []
    for split in REQUIRED_SPLITS:
        for family in contract.source_families:
            for band_index, band in enumerate(contract.point_control_subbands_hz):
                groups = covered_groups[(split, family, band_index)]
                count = len(groups)
                passed = count >= floor
                cells.append(
                    {
                        "split": split,
                        "source_family": family,
                        "band_hz": [float(band[0]), float(band[1])],
                        "native_eligible_independent_groups": count,
                        "minimum_groups": floor,
                        "passed": passed,
                    }
                )
                if not passed:
                    reasons.append(
                        f"{split}/{family}/{band[0]:.0f}-{band[1]:.0f}Hz: "
                        f"native-Nyquist 독립 group {count} < {floor}"
                    )

    payload: dict[str, Any] = {
        "schema": SYNTHETIC_BROADBAND_NATIVE_AUDIT_SCHEMA,
        "role": "diagnostic_only_not_spectral_or_training_readiness",
        "status": "PASS" if not reasons else "BLOCKED",
        "control_band_contract_sha256": contract.digest(),
        "required_native_nyquist_hz": contract.required_excitation_upper_hz,
        "family_tags": {key: list(value) for key, value in mapping.items()},
        "minimum_independent_groups_per_split_family_band": floor,
        "manifest_row_counts": dict(sorted(row_counts.items())),
        "cells": cells,
        "reasons": reasons,
        "limitations": [
            "manifest sample_rate는 canonical decoder audit에 결속되어야 합니다",
            "source spectral density는 이 감사가 확인하지 않습니다",
            "실제 ERR target-d coverage와 broadband P/S는 별도 receipt가 필요합니다",
            "built-in synthetic generator는 independent public lineage로 세지 않습니다",
        ],
    }
    payload["evidence_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


__all__ = [
    "DEFAULT_FAMILY_TAGS",
    "MIN_INDEPENDENT_GROUPS_PER_SPLIT_FAMILY_BAND",
    "REQUIRED_SPLITS",
    "SYNTHETIC_BROADBAND_NATIVE_AUDIT_SCHEMA",
    "audit_synthetic_native_manifest_rows",
]
