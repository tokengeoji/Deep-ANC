"""Recorded generation additions와 public synthetic corpus의 누수 계약.

기존 canonical holdout은 immutable parent 82세션만 표현한다. schema v2 transfer의
추가 세션 원본(특히 source-pool 밖의 external raw)은 그 holdout에 없으므로, 별도
generation report/source-plan 증거에서 identity를 재유도해 public manifest 세대에
결속한다. 이 모듈은 오디오를 열지 않고 path/SHA/lineage 증거만 다룬다.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .holdout_contract import HoldoutContractError, read_regular_file_snapshot
from .recorded_generation import (
    ADDITION_SESSION_COUNT,
    SOURCE_PLAN_FIELDS,
    RecordedGenerationError,
    _derive_librispeech_identity_component_map,
    validate_recorded_generation,
)
from . import public_lineage


RECORDED_GENERATION_EXCLUSION_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RecordedGenerationExclusionError(ValueError):
    """Generation exclusion 증거가 불완전하거나 현재 bytes와 다르다."""


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_ref(snapshot: Any, *, repo_root: Path) -> dict[str, object]:
    return {
        "path": snapshot.path.relative_to(repo_root).as_posix(),
        "sha256": snapshot.sha256,
        "size": int(snapshot.size),
    }


def _read_plan_rows(raw: bytes) -> dict[int, dict[str, str]]:
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
        if tuple(reader.fieldnames or ()) != SOURCE_PLAN_FIELDS:
            raise RecordedGenerationExclusionError(
                "recorded generation source plan header가 canonical exact 계약과 다릅니다"
            )
        rows = list(reader)
    except UnicodeDecodeError as exc:
        raise RecordedGenerationExclusionError(
            "recorded generation source plan UTF-8 오류"
        ) from exc
    if len(rows) != ADDITION_SESSION_COUNT:
        raise RecordedGenerationExclusionError(
            f"recorded generation source plan은 정확히 {ADDITION_SESSION_COUNT}행이어야 합니다"
        )
    return {number: row for number, row in enumerate(rows, start=2)}


def derive_recorded_generation_exclusion(
    summary: Mapping[str, Any], *, repo_root: str | Path
) -> dict[str, Any]:
    """검증된 :func:`validate_recorded_generation` 결과에서 exclusion을 유도한다.

    source plan을 별도 pathname read로 믿지 않는다. generation validator가 결속한
    path/SHA를 같은 FD snapshot으로 다시 확인하고, session별 source/raw SHA와 lineage가
    plan 행과 정확히 같은지 대조한 뒤 canonical identity 목록을 만든다.
    """

    root = Path(os.path.abspath(os.fspath(repo_root)))
    generation_snapshot = summary.get("_validated_generation_snapshot")
    additions = summary.get("additions")
    if generation_snapshot is None or not isinstance(additions, Mapping):
        raise RecordedGenerationExclusionError(
            "validate_recorded_generation의 검증 snapshot/additions가 없습니다"
        )
    generation_id = summary.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise RecordedGenerationExclusionError("recorded generation_id가 없습니다")
    plan_ref = additions.get("source_plan")
    sessions = additions.get("sessions")
    if (
        not isinstance(plan_ref, Mapping)
        or set(plan_ref) != {"path", "sha256", "size"}
        or not isinstance(sessions, list)
        or len(sessions) != ADDITION_SESSION_COUNT
    ):
        raise RecordedGenerationExclusionError(
            "recorded generation source_plan/session summary가 불완전합니다"
        )
    plan_path = root / str(plan_ref.get("path") or "")
    try:
        plan_snapshot = read_regular_file_snapshot(
            plan_path,
            root=root,
            label="recorded generation exclusion source plan",
            capture_bytes=True,
        )
    except HoldoutContractError as exc:
        raise RecordedGenerationExclusionError(str(exc)) from exc
    if (
        plan_snapshot.sha256 != plan_ref.get("sha256")
        or plan_snapshot.size != plan_ref.get("size")
        or plan_snapshot.data is None
    ):
        raise RecordedGenerationExclusionError(
            "recorded generation exclusion source plan path/SHA/size 불일치"
        )
    rows = _read_plan_rows(plan_snapshot.data)

    identities: list[dict[str, Any]] = []
    seen_rows: set[int] = set()
    for session in sessions:
        if not isinstance(session, Mapping):
            raise RecordedGenerationExclusionError(
                "recorded generation addition session summary가 mapping이 아닙니다"
            )
        row_number = session.get("source_row_number")
        if (
            isinstance(row_number, bool)
            or not isinstance(row_number, int)
            or row_number not in rows
            or row_number in seen_rows
        ):
            raise RecordedGenerationExclusionError(
                "recorded generation addition source_row_number가 누락/중복됐습니다"
            )
        seen_rows.add(row_number)
        row = rows[row_number]
        source_sha = str(row.get("source_file_sha256") or "")
        raw_sha_text = str(row.get("raw_member_sha256") or "")
        raw_sha = raw_sha_text or None
        source_path = str(row.get("path") or "")
        raw_path_text = str(row.get("raw_member_path") or "")
        raw_path = raw_path_text or None
        raw_lineage_text = str(row.get("raw_member_lineage_key") or "")
        raw_lineage = raw_lineage_text or None
        authority = session.get("authority_components")
        if (
            _SHA256_RE.fullmatch(source_sha) is None
            or (raw_sha is not None and _SHA256_RE.fullmatch(raw_sha) is None)
            or not source_path
            or not isinstance(authority, list)
            or not authority
            or authority != sorted(set(str(item) for item in authority))
            or session.get("source_path") != source_path
            or (session.get("raw_member_path") or None) != raw_path
            or session.get("source_file_sha256") != source_sha
            or (session.get("raw_member_sha256") or None) != raw_sha
            or (session.get("raw_member_lineage_key") or None) != raw_lineage
            or session.get("source_kind") != row.get("source_kind")
            or session.get("source_family") != row.get("source_family")
        ):
            raise RecordedGenerationExclusionError(
                f"recorded generation addition row {row_number} identity가 source plan과 다릅니다"
            )
        identities.append(
            {
                "source_row_number": row_number,
                "source_kind": str(row["source_kind"]),
                "source_family": str(row["source_family"]),
                "source_path": source_path,
                "source_file_sha256": source_sha,
                "raw_member_path": raw_path,
                "raw_member_sha256": raw_sha,
                "raw_member_lineage_key": raw_lineage,
                "authority_components": list(authority),
            }
        )
    if seen_rows != set(rows):
        raise RecordedGenerationExclusionError(
            "recorded generation source plan의 모든 행이 addition session과 대응하지 않습니다"
        )
    identities.sort(key=lambda item: int(item["source_row_number"]))
    evidence: dict[str, Any] = {
        "schema_version": RECORDED_GENERATION_EXCLUSION_SCHEMA_VERSION,
        "generation_id": generation_id,
        "generation": _file_ref(generation_snapshot, repo_root=root),
        "source_plan": _file_ref(plan_snapshot, repo_root=root),
        "identity_count": len(identities),
        "identities": identities,
        "identities_sha256": _canonical_json_sha256(identities),
    }
    return evidence


def validate_recorded_generation_exclusion(
    evidence: object, *, repo_root: str | Path
) -> dict[str, Any]:
    """sidecar exclusion을 generation report/source plan에서 독립 재유도한다."""

    if not isinstance(evidence, dict):
        raise RecordedGenerationExclusionError(
            "recorded_generation_exclusion이 mapping이 아닙니다"
        )
    generation_ref = evidence.get("generation")
    source_plan_ref = evidence.get("source_plan")
    identities = evidence.get("identities")
    if (
        type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version")
        != RECORDED_GENERATION_EXCLUSION_SCHEMA_VERSION
        or not isinstance(generation_ref, dict)
        or set(generation_ref) != {"path", "sha256", "size"}
        or not isinstance(source_plan_ref, dict)
        or set(source_plan_ref) != {"path", "sha256", "size"}
        or _SHA256_RE.fullmatch(str(generation_ref.get("sha256") or "")) is None
        or _SHA256_RE.fullmatch(str(source_plan_ref.get("sha256") or "")) is None
        or not isinstance(generation_ref.get("path"), str)
        or not isinstance(source_plan_ref.get("path"), str)
        or type(generation_ref.get("size")) is not int
        or int(generation_ref["size"]) <= 0
        or type(source_plan_ref.get("size")) is not int
        or int(source_plan_ref["size"]) <= 0
        or type(evidence.get("identity_count")) is not int
        or evidence.get("identity_count") != ADDITION_SESSION_COUNT
        or not isinstance(identities, list)
        or len(identities) != ADDITION_SESSION_COUNT
        or _SHA256_RE.fullmatch(str(evidence.get("identities_sha256") or "")) is None
        or evidence.get("identities_sha256") != _canonical_json_sha256(identities)
    ):
        raise RecordedGenerationExclusionError(
            "recorded_generation_exclusion schema/generation ref가 유효하지 않습니다"
        )
    for index, identity in enumerate(identities):
        if (
            not isinstance(identity, dict)
            or set(identity)
            != {
                "source_row_number",
                "source_kind",
                "source_family",
                "source_path",
                "source_file_sha256",
                "raw_member_path",
                "raw_member_sha256",
                "raw_member_lineage_key",
                "authority_components",
            }
            or type(identity.get("source_row_number")) is not int
            or not isinstance(identity.get("source_kind"), str)
            or not isinstance(identity.get("source_family"), str)
            or not isinstance(identity.get("source_path"), str)
            or _SHA256_RE.fullmatch(
                str(identity.get("source_file_sha256") or "")
            )
            is None
            or identity.get("raw_member_path") is not None
            and not isinstance(identity.get("raw_member_path"), str)
            or identity.get("raw_member_sha256") is not None
            and _SHA256_RE.fullmatch(str(identity.get("raw_member_sha256"))) is None
            or identity.get("raw_member_lineage_key") is not None
            and not isinstance(identity.get("raw_member_lineage_key"), str)
            or not isinstance(identity.get("authority_components"), list)
            or identity.get("authority_components")
            != sorted(set(str(item) for item in identity.get("authority_components", [])))
        ):
            raise RecordedGenerationExclusionError(
                f"recorded_generation_exclusion identity #{index} schema가 유효하지 않습니다"
            )
    root = Path(os.path.abspath(os.fspath(repo_root)))
    try:
        summary = validate_recorded_generation(
            root / str(generation_ref["path"]),
            repo_root=root,
            expected_sha256=str(generation_ref["sha256"]),
            require_source_files=False,
        )
    except (OSError, RecordedGenerationError) as exc:
        raise RecordedGenerationExclusionError(
            f"recorded generation exclusion report 검증 실패: {exc}"
        ) from exc
    derived = derive_recorded_generation_exclusion(summary, repo_root=root)
    if evidence != derived:
        raise RecordedGenerationExclusionError(
            "recorded_generation_exclusion이 현재 generation/source plan에서 재유도한 값과 다릅니다"
        )
    return derived


def _identity_sets(evidence: Mapping[str, Any]) -> dict[str, set[str]]:
    identities = evidence.get("identities")
    if not isinstance(identities, list):
        raise RecordedGenerationExclusionError(
            "recorded generation exclusion identities가 목록이 아닙니다"
        )
    basenames: set[str] = set()
    contents: set[str] = set()
    raw_lineages: set[str] = set()
    authority: set[str] = set()
    for row in identities:
        if not isinstance(row, Mapping):
            raise RecordedGenerationExclusionError(
                "recorded generation exclusion identity가 mapping이 아닙니다"
            )
        for field in ("source_path", "raw_member_path"):
            value = row.get(field)
            if isinstance(value, str) and value:
                basenames.add(Path(value).name.casefold())
        for field in ("source_file_sha256", "raw_member_sha256"):
            value = row.get(field)
            if isinstance(value, str) and value:
                contents.add(value)
        raw_lineage = row.get("raw_member_lineage_key")
        if isinstance(raw_lineage, str) and raw_lineage:
            raw_lineages.add(raw_lineage)
        components = row.get("authority_components")
        if isinstance(components, list):
            authority.update(str(value) for value in components)
    return {
        "basenames": basenames,
        "contents": contents,
        "raw_lineages": raw_lineages,
        "authority": authority,
    }


def _librispeech_component_map(repo_root: Path) -> dict[str, str]:
    try:
        snapshot = read_regular_file_snapshot(
            repo_root / public_lineage.LIBRISPEECH_CHAPTERS,
            root=repo_root,
            label="generation exclusion LibriSpeech CHAPTERS",
            capture_bytes=True,
        )
    except HoldoutContractError as exc:
        raise RecordedGenerationExclusionError(str(exc)) from exc
    assert snapshot.data is not None
    try:
        chapters = public_lineage.parse_librispeech_chapters_bytes(snapshot.data)
        by_identity, _components = _derive_librispeech_identity_component_map(chapters)
    except (ValueError, RecordedGenerationError) as exc:
        raise RecordedGenerationExclusionError(
            f"generation exclusion LibriSpeech component 재유도 실패: {exc}"
        ) from exc
    return by_identity


def find_recorded_generation_overlaps(
    evidence: Mapping[str, Any],
    entries_by_tag: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    repo_root: str | Path,
) -> list[dict[str, Any]]:
    """Public manifest entry와 addition identity의 교집합을 독립 계산한다."""

    sets = _identity_sets(evidence)
    needs_libri = any(
        value.startswith("librispeech_component:") for value in sets["authority"]
    )
    root = Path(os.path.abspath(os.fspath(repo_root)))
    libri_components = _librispeech_component_map(root) if needs_libri else {}
    overlaps: list[dict[str, Any]] = []
    for tag, entries in sorted(entries_by_tag.items()):
        for index, entry in enumerate(entries):
            path = str(entry.get("path") or "")
            content = str(entry.get("content_sha256") or "")
            keys_raw = entry.get("lineage_keys")
            keys = (
                {str(value) for value in keys_raw}
                if isinstance(keys_raw, list)
                else set()
            )
            basename = Path(path).name.casefold()
            entry_authority = {
                f"clip_identity:{basename}",
                f"public_group:{str(entry.get('group_id') or '')}",
                *(f"esc50_identity:{key}" for key in keys),
                *(f"fma_identity:{key}" for key in keys),
                *(f"public_lineage_key:{key}" for key in keys),
            }
            for key in keys:
                component = libri_components.get(key)
                if component is not None:
                    entry_authority.add(f"librispeech_component:{component}")
            dimensions: list[str] = []
            if basename in sets["basenames"]:
                dimensions.append("basename")
            if content in sets["contents"]:
                dimensions.append("content_sha256")
            if keys.intersection(sets["raw_lineages"]):
                dimensions.append("raw_member_lineage_key")
            if entry_authority.intersection(sets["authority"]):
                dimensions.append("authority_component")
            if dimensions:
                overlaps.append(
                    {
                        "tag": str(tag),
                        "entry_index": index,
                        "path": path,
                        "split": str(entry.get("split") or ""),
                        "group_id": str(entry.get("group_id") or ""),
                        "dimensions": dimensions,
                    }
                )
    return overlaps


def generation_excluded_basenames(evidence: Mapping[str, Any]) -> set[str]:
    """Producer의 public DSU 구성 전에 적용할 canonical basename 집합."""

    return set(_identity_sets(evidence)["basenames"])


def generation_excluded_public_groups(evidence: Mapping[str, Any]) -> set[str]:
    """DNS 등 public source가 속한 component 전체를 producer에서 제외한다."""

    prefix = "public_group:"
    return {
        value.removeprefix(prefix)
        for value in _identity_sets(evidence)["authority"]
        if value.startswith(prefix) and value != prefix
    }


__all__ = [
    "RECORDED_GENERATION_EXCLUSION_SCHEMA_VERSION",
    "RecordedGenerationExclusionError",
    "derive_recorded_generation_exclusion",
    "find_recorded_generation_overlaps",
    "generation_excluded_basenames",
    "generation_excluded_public_groups",
    "validate_recorded_generation_exclusion",
]
