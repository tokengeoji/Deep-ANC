"""Jetson 로컬 source를 population-v3 관점에서 읽기 전용으로 감사한다.

이 모듈은 legacy public/recorded/source-pool manifest와 unreferenced audio tree를 실제
로컬 bytes에 대조한다. 존재/크기/SHA, bounded decoder probe, 실제 sample rate/Nyquist,
family/split 및 lineage *mapping 후보*를 재계산한다. legacy payload는 절대로 v3
manifest로 승격하지 않으며 decoded PCM, causal-P ERR, density도 발행하지 않는다.

canonical fullband causal P authority와 ``POPULATION_V3_AUTHORITY``가 없으므로 이
auditor가 반환하는 최종 상태는 항상 ``BLOCKED``다. 이는 availability 보고서이지 학습
admission이나 물리 성능 증거가 아니다. 네트워크와 오디오 장치를 사용하지 않는다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..dsp.control_band_contract import (
    BroadbandFullOctaveContractV3,
    REQUIRED_SOURCE_FAMILIES,
)
from .broadband_population_contract_v3 import (
    CausalPrimaryOperatorV3,
    POPULATION_V3_AUTHORITY,
    PopulationCoverageContractV3,
)
from .public_lineage import (
    PublicLineageError,
    esc50_lineage_keys,
    fma_lineage_keys,
    librispeech_lineage_keys,
    parse_esc50_metadata_bytes,
    parse_fma_tracks_bytes,
    parse_librispeech_chapters_bytes,
)


_FROZEN = ConfigDict(frozen=True, extra="forbid")
_HEX = frozenset("0123456789abcdef")
_AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"})
_INPUT_KINDS = (
    "public_jsonl",
    "recorded_jsonl",
    "source_pool_csv",
    "unreferenced_audio_tree",
)
_METADATA_PATHS = {
    "librispeech": "data/raw/speech/LibriSpeech/CHAPTERS.TXT",
    "fma": "data/raw/music/fma_metadata/tracks.csv",
    "esc50": "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv",
}


class PopulationAvailabilityV3Blocked(RuntimeError):
    """availability evidence를 canonical population authority로 사용할 수 없음."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise ValueError(f"{name}가 lowercase SHA-256이 아닙니다")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_no_duplicates(raw: bytes, *, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label}: JSON duplicate key {key}")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=pairs)


def _normal_relative_path(path: str) -> str:
    candidate = Path(str(path))
    if not str(path).strip() or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("repository 상대경로가 정규화되지 않았습니다")
    return candidate.as_posix()


def _resolve_within_root(
    root: Path,
    path_value: str | Path,
    *,
    base: Path | None = None,
) -> tuple[Path, str]:
    root = root.resolve(strict=True)
    raw = Path(path_value)
    candidate = raw if raw.is_absolute() else (base or root) / raw
    resolved = Path(os.path.abspath(candidate))
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("경로가 repository root 밖입니다") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError("경로에 symlink가 있습니다")
    return resolved, relative.as_posix()


class AvailabilityInputV3(BaseModel):
    model_config = _FROZEN

    kind: Literal[
        "public_jsonl",
        "recorded_jsonl",
        "source_pool_csv",
        "unreferenced_audio_tree",
    ]
    path: str
    source_family: Literal["speech", "music", "environment", "machine"] | None = None

    @model_validator(mode="after")
    def _validate_input(self) -> "AvailabilityInputV3":
        _normal_relative_path(self.path)
        if self.kind == "unreferenced_audio_tree" and self.source_family is None:
            raise ValueError("unreferenced tree에는 source_family가 필요합니다")
        if self.kind != "unreferenced_audio_tree" and self.source_family is not None:
            raise ValueError("manifest input에는 source_family override를 허용하지 않습니다")
        return self


class LocalEvidenceFileV3(BaseModel):
    model_config = _FROZEN

    path: str
    size_bytes: int = Field(gt=0)
    sha256: str

    @model_validator(mode="after")
    def _validate_file(self) -> "LocalEvidenceFileV3":
        _normal_relative_path(self.path)
        _require_sha("local evidence file SHA", self.sha256)
        return self


class AvailabilityInputEvidenceV3(BaseModel):
    model_config = _FROZEN

    kind: str
    path: str
    source_family: str | None
    status: Literal["PASS", "MISSING", "INVALID"]
    file: LocalEvidenceFileV3 | None
    detected_legacy_schema: str
    entries_seen: int = Field(ge=0)
    entries_emitted: int = Field(ge=0)
    error: str | None
    legacy_automatic_promotion: Literal[False] = False
    mapping_only: Literal[True] = True


class MetadataEvidenceV3(BaseModel):
    model_config = _FROZEN

    role: Literal["librispeech", "fma", "esc50"]
    status: Literal["PASS", "MISSING", "INVALID"]
    file: LocalEvidenceFileV3 | None
    mapped_records: int = Field(ge=0)
    error: str | None
    authority_for_v3_manifest: Literal[False] = False


class SourceAvailabilityCandidateV3(BaseModel):
    model_config = _FROZEN

    candidate_id: str
    input_kind: str
    input_path: str
    entry_index: int = Field(gt=0)
    detected_legacy_schema: str
    legacy_automatic_promotion: Literal[False] = False
    mapping_only: Literal[True] = True
    declared_audio_path: str
    resolved_audio_path: str | None
    origin_role: Literal[
        "direct_corpus_native_candidate",
        "recorded_playback_composite",
        "processed_source_pool_composite",
        "unreferenced_native_candidate",
    ]
    file_status: Literal["PRESENT", "MISSING", "UNSAFE"]
    actual_size_bytes: int | None
    actual_sha256: str | None
    declared_sha256: str | None
    declared_sha_matches: bool | None
    decoder_probe_status: Literal["PASS", "FAIL", "NOT_RUN"]
    decoder_probe_scope: Literal["header_first_last_1024_frames", "none"]
    decoder_error: str | None
    full_decode_verified: Literal[False] = False
    actual_sample_rate_hz: int | None
    actual_header_nyquist_hz: float | None
    actual_channels: int | None
    actual_frames: int | None
    actual_format: str | None
    actual_subtype: str | None
    declared_sample_rate_hz: int | None
    declared_sample_rate_matches: bool | None
    native_nyquist_verified: bool
    native_nyquist_hz: float | None
    native_physical_nyquist_coverage: tuple[bool, ...] | None
    native_objective_octave_nyquist_coverage: tuple[bool, ...] | None
    full_target_native_nyquist: bool
    source_family: Literal["speech", "music", "environment", "machine"] | None
    source_family_basis: str
    split: Literal["train", "val", "test"] | None
    split_basis: str
    semantic_lineage_keys: tuple[str, ...]
    semantic_lineage_basis: tuple[str, ...]
    semantic_lineage_available: bool
    mapping_component_id: str | None
    mapping_component_authoritative: Literal[False] = False
    decoded_pcm_artifact_sha256: None = None
    p_applied_err_artifact_sha256: None = None
    density_recomputed: Literal[False] = False
    availability_status: Literal[
        "MAPPING_CANDIDATE",
        "PARTIAL_MAPPING_CANDIDATE",
        "UNAVAILABLE",
    ]
    qualification_limitations: tuple[str, ...]
    blockers: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_candidate(self) -> "SourceAvailabilityCandidateV3":
        if not self.candidate_id.strip() or not self.detected_legacy_schema.strip():
            raise ValueError("availability candidate ID/schema가 비었습니다")
        if self.actual_sha256 is not None:
            _require_sha("actual source SHA", self.actual_sha256)
        if self.declared_sha256 is not None:
            _require_sha("declared source SHA", self.declared_sha256)
        if self.file_status == "PRESENT":
            if self.actual_size_bytes is None or self.actual_size_bytes <= 0:
                raise ValueError("PRESENT source의 실제 byte size가 없습니다")
            if self.actual_sha256 is None:
                raise ValueError("PRESENT source의 실제 SHA가 없습니다")
        elif self.actual_size_bytes is not None or self.actual_sha256 is not None:
            raise ValueError("없는/unsafe source에 실제 file evidence가 있습니다")
        if self.decoder_probe_status == "PASS":
            values = (
                self.actual_sample_rate_hz,
                self.actual_channels,
                self.actual_frames,
            )
            if any(value is None or value <= 0 for value in values):
                raise ValueError("decoder PASS에 실제 audio metadata가 없습니다")
            if self.decoder_probe_scope != "header_first_last_1024_frames":
                raise ValueError("decoder PASS probe scope가 다릅니다")
        if self.actual_header_nyquist_hz is not None and (
            not math.isfinite(self.actual_header_nyquist_hz)
            or self.actual_header_nyquist_hz <= 0.0
        ):
            raise ValueError("actual header Nyquist가 유효하지 않습니다")
        if self.native_nyquist_verified != (self.native_nyquist_hz is not None):
            raise ValueError("native Nyquist verified flag/value가 모순됩니다")
        if self.native_nyquist_hz is not None and (
            not math.isfinite(self.native_nyquist_hz) or self.native_nyquist_hz <= 0.0
        ):
            raise ValueError("verified native Nyquist가 유효하지 않습니다")
        if self.native_physical_nyquist_coverage is not None and len(
            self.native_physical_nyquist_coverage
        ) != 8:
            raise ValueError("physical Nyquist mask는 정확히 8개여야 합니다")
        if self.native_objective_octave_nyquist_coverage is not None and len(
            self.native_objective_octave_nyquist_coverage
        ) != 7:
            raise ValueError("objective Nyquist mask는 정확히 7개여야 합니다")
        if not self.native_nyquist_verified and (
            self.native_physical_nyquist_coverage is not None
            or self.native_objective_octave_nyquist_coverage is not None
            or self.full_target_native_nyquist
        ):
            raise ValueError("unverified native source에 Nyquist coverage가 있습니다")
        if self.mapping_component_id is not None:
            _require_sha("mapping component ID", self.mapping_component_id)
        if self.availability_status == "MAPPING_CANDIDATE" and (
            self.decoder_probe_status != "PASS"
            or self.source_family is None
            or self.split is None
            or not self.semantic_lineage_available
        ):
            raise ValueError("MAPPING_CANDIDATE 필수 evidence가 없습니다")
        expected_limitations: list[str] = []
        if self.native_physical_nyquist_coverage is not None and not all(
            self.native_physical_nyquist_coverage
        ):
            expected_limitations.append("native_nyquist_partial_physical_band_coverage")
        if self.native_objective_octave_nyquist_coverage is not None and not all(
            self.native_objective_octave_nyquist_coverage
        ):
            expected_limitations.append("native_nyquist_partial_objective_octave_coverage")
        if tuple(sorted(expected_limitations)) != self.qualification_limitations:
            raise ValueError("candidate의 대역별 qualification limitation이 Nyquist mask와 다릅니다")
        if any("nyquist" in blocker and "full_v3_target" in blocker for blocker in self.blockers):
            raise ValueError("부분 Nyquist coverage를 전역 canonical blocker로 사용할 수 없습니다")
        return self


class AvailabilityCellV3(BaseModel):
    model_config = _FROZEN

    split: Literal["train", "val", "test"]
    source_family: Literal["speech", "music", "environment", "machine"]
    rows: int = Field(ge=0)
    present_decodable_rows: int = Field(ge=0)
    mapping_candidate_rows: int = Field(ge=0)
    full_target_nyquist_mapping_rows: int = Field(ge=0)
    mapping_lineage_components: int = Field(ge=0)
    mapping_native_components_per_physical_band: tuple[int, ...]
    mapping_native_components_per_objective_octave: tuple[int, ...]
    canonical_qualified_components_per_physical_band: tuple[Literal[0], ...]
    canonical_qualified_components_per_objective_octave: tuple[Literal[0], ...]
    physical_component_band_deficit: Literal[32] = 32
    objective_component_octave_deficit: Literal[28] = 28
    status: Literal["BLOCKED"] = "BLOCKED"

    @model_validator(mode="after")
    def _validate_cell(self) -> "AvailabilityCellV3":
        if len(self.canonical_qualified_components_per_physical_band) != 8:
            raise ValueError("availability cell physical vector가 8개가 아닙니다")
        if len(self.canonical_qualified_components_per_objective_octave) != 7:
            raise ValueError("availability cell objective vector가 7개가 아닙니다")
        if len(self.mapping_native_components_per_physical_band) != 8:
            raise ValueError("mapping native physical vector가 8개가 아닙니다")
        if len(self.mapping_native_components_per_objective_octave) != 7:
            raise ValueError("mapping native objective vector가 7개가 아닙니다")
        return self


class AvailabilitySummaryV3(BaseModel):
    model_config = _FROZEN

    manifest_entries_total: int = Field(ge=0)
    candidates_reported: int = Field(ge=0)
    unique_resolved_audio_paths: int = Field(ge=0)
    files_present: int = Field(ge=0)
    files_missing: int = Field(ge=0)
    unsafe_paths: int = Field(ge=0)
    decoder_probe_pass: int = Field(ge=0)
    decoder_probe_fail: int = Field(ge=0)
    declared_sha_present: int = Field(ge=0)
    declared_sha_match: int = Field(ge=0)
    declared_sha_mismatch: int = Field(ge=0)
    source_family_mapped: int = Field(ge=0)
    split_mapped: int = Field(ge=0)
    semantic_lineage_mapped: int = Field(ge=0)
    mapping_candidates: int = Field(ge=0)
    full_target_native_nyquist_rows: int = Field(ge=0)
    direct_native_present: int = Field(ge=0)
    direct_native_full_target_nyquist: int = Field(ge=0)
    content_sha_duplicate_groups: int = Field(ge=0)
    mapping_components_crossing_splits: int = Field(ge=0)
    mapping_components_crossing_families: int = Field(ge=0)
    present_components_crossing_splits: int = Field(ge=0)
    present_components_crossing_families: int = Field(ge=0)
    canonical_population_candidates: Literal[0] = 0
    canonical_physical_component_band_deficit: Literal[384] = 384
    canonical_objective_component_octave_deficit: Literal[336] = 336


class CausalPAvailabilityV3(BaseModel):
    model_config = _FROZEN

    requested_path: str | None
    discovered_operator_payload_paths: tuple[str, ...]
    status: Literal["MISSING", "INVALID", "STRUCTURAL_ONLY"]
    candidate_file: LocalEvidenceFileV3 | None
    fir_file: LocalEvidenceFileV3 | None
    operator: CausalPrimaryOperatorV3 | None
    fullband_causal_p_authority: Literal[False] = False
    authority: None = None
    blockers: tuple[str, ...]


class LineageMappingConflictV3(BaseModel):
    model_config = _FROZEN

    mapping_component_id: str
    conflict_axes: tuple[Literal["split", "source_family"], ...]
    splits: tuple[str, ...]
    source_families: tuple[str, ...]
    candidate_count: int = Field(gt=0)
    present_candidate_count: int = Field(ge=0)
    mapping_candidate_count: int = Field(ge=0)
    input_paths: tuple[str, ...]
    canonical_lineage_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_conflict(self) -> "LineageMappingConflictV3":
        _require_sha("lineage mapping component", self.mapping_component_id)
        expected: list[str] = []
        if len(self.splits) > 1:
            expected.append("split")
        if len(self.source_families) > 1:
            expected.append("source_family")
        if tuple(expected) != self.conflict_axes or not expected:
            raise ValueError("lineage conflict axes와 실제 split/family가 다릅니다")
        return self


class PopulationAvailabilityReportV3(BaseModel):
    model_config = _FROZEN

    schema_version: Literal["broadband_population_availability_audit_v3_bandwise_v1"] = (
        "broadband_population_availability_audit_v3_bandwise_v1"
    )
    role: Literal["read_only_mapping_availability_not_population_manifest"] = (
        "read_only_mapping_availability_not_population_manifest"
    )
    status: Literal["BLOCKED"] = "BLOCKED"
    authority: None = None
    control_band_contract: BroadbandFullOctaveContractV3
    control_band_contract_sha256: str
    population_contract: PopulationCoverageContractV3
    population_contract_sha256: str
    inputs: tuple[AvailabilityInputEvidenceV3, ...]
    metadata_evidence: tuple[MetadataEvidenceV3, ...]
    causal_primary: CausalPAvailabilityV3
    candidates: tuple[SourceAvailabilityCandidateV3, ...]
    lineage_mapping_conflicts: tuple[LineageMappingConflictV3, ...]
    cells: tuple[AvailabilityCellV3, ...]
    summary: AvailabilitySummaryV3
    canonical_population_manifest_issued: Literal[False] = False
    decoded_pcm_issued: Literal[False] = False
    p_applied_err_issued: Literal[False] = False
    density_recomputed: Literal[False] = False
    legacy_v1_v2_automatic_promotion: Literal[False] = False
    blockers: tuple[str, ...]
    safety: Mapping[str, object]
    evidence_sha256: str

    @model_validator(mode="after")
    def _validate_report(self) -> "PopulationAvailabilityReportV3":
        control = BroadbandFullOctaveContractV3.canonical()
        population = PopulationCoverageContractV3.canonical()
        if self.control_band_contract.model_dump(mode="json") != control.model_dump(
            mode="json"
        ) or self.control_band_contract_sha256 != control.digest():
            raise ValueError("availability report control-band v3 payload/SHA가 다릅니다")
        if self.population_contract.model_dump(mode="json") != population.model_dump(
            mode="json"
        ) or self.population_contract_sha256 != population.digest():
            raise ValueError("availability report population v3 payload/SHA가 다릅니다")
        expected_cells = {
            (split, family)
            for split in ("train", "val", "test")
            for family in REQUIRED_SOURCE_FAMILIES
        }
        actual_cells = {(cell.split, cell.source_family) for cell in self.cells}
        if actual_cells != expected_cells or len(self.cells) != len(expected_cells):
            raise ValueError("availability report가 12개 split×family cell을 덮지 않습니다")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("availability candidate_id가 중복됐습니다")
        if sum(item.entries_emitted for item in self.inputs) != len(self.candidates):
            raise ValueError("input emitted count와 availability candidate 수가 다릅니다")
        manifest_entries = sum(
            item.entries_seen
            for item in self.inputs
            if item.kind != "unreferenced_audio_tree"
        )
        expected_summary, recomputed_cells, recomputed_conflicts = _summary_and_cells(
            self.candidates,
            manifest_entries_total=manifest_entries,
        )
        if self.summary != expected_summary or self.cells != recomputed_cells:
            raise ValueError("availability summary/cell이 candidate 재집계와 다릅니다")
        if self.lineage_mapping_conflicts != recomputed_conflicts:
            raise ValueError("lineage mapping conflict가 candidate 재집계와 다릅니다")
        payload = self.model_dump(mode="json", exclude={"evidence_sha256"})
        if self.evidence_sha256 != _digest(payload):
            raise ValueError("availability report evidence SHA가 payload와 다릅니다")
        if POPULATION_V3_AUTHORITY is not None:
            raise ValueError("이 auditor는 authority 설정 상태에서 사용할 수 없습니다")
        return self

    def digest(self) -> str:
        return self.evidence_sha256


class _DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            low, high = sorted((root_left, root_right))
            self.parent[high] = low


class _LineageResolver:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._loaded: dict[str, Mapping[Any, Any] | None] = {}
        self.evidence: dict[str, MetadataEvidenceV3] = {}

    def _load(self, role: str) -> Mapping[Any, Any] | None:
        if role in self._loaded:
            return self._loaded[role]
        relative = _METADATA_PATHS[role]
        path = self.root / relative
        parser = {
            "librispeech": parse_librispeech_chapters_bytes,
            "fma": parse_fma_tracks_bytes,
            "esc50": parse_esc50_metadata_bytes,
        }[role]
        if not path.is_file() or path.is_symlink():
            self._loaded[role] = None
            self.evidence[role] = MetadataEvidenceV3(
                role=role,
                status="MISSING",
                file=None,
                mapped_records=0,
                error="metadata file absent",
            )
            return None
        try:
            raw = path.read_bytes()
            parsed = parser(raw)
            reference = LocalEvidenceFileV3(
                path=relative,
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        except (OSError, ValueError, PublicLineageError) as exc:
            self._loaded[role] = None
            self.evidence[role] = MetadataEvidenceV3(
                role=role,
                status="INVALID",
                file=None,
                mapped_records=0,
                error=f"{type(exc).__name__}: {exc}",
            )
            return None
        self._loaded[role] = parsed
        self.evidence[role] = MetadataEvidenceV3(
            role=role,
            status="PASS",
            file=reference,
            mapped_records=len(parsed),
            error=None,
        )
        return parsed

    def keys_for_clip(self, clip: str) -> tuple[tuple[str, ...], str | None]:
        path = Path(str(clip))
        name = path.name
        stem = path.stem
        try:
            if path.suffix.casefold() == ".flac" and len(stem.split("-")) >= 3:
                metadata = self._load("librispeech")
                if metadata is None:
                    return (), "librispeech_metadata_absent_or_invalid"
                return tuple(librispeech_lineage_keys(name, metadata)), None
            if path.suffix.casefold() == ".mp3" and stem.isdigit():
                metadata = self._load("fma")
                if metadata is None:
                    return (), "fma_metadata_absent_or_invalid"
                return tuple(fma_lineage_keys(name, metadata)), None
            if path.suffix.casefold() == ".wav" and "-" in stem:
                metadata = self._load("esc50")
                if metadata is None:
                    return (), "esc50_metadata_absent_or_invalid"
                return tuple(esc50_lineage_keys(name, metadata)), None
        except (KeyError, PublicLineageError, ValueError) as exc:
            return (), f"semantic_lineage_mapping_failed:{exc}"
        return (), "semantic_lineage_mapping_rule_absent"


def _manifest_file_evidence(root: Path, relative: str) -> LocalEvidenceFileV3:
    path, normalized = _resolve_within_root(root, relative)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(normalized)
    return LocalEvidenceFileV3(
        path=normalized,
        size_bytes=path.stat().st_size,
        sha256=_sha256_file(path),
    )


def _family_from_entry(entry: Mapping[str, Any]) -> tuple[str | None, str]:
    explicit = str(entry.get("source_family") or "").strip().casefold()
    if explicit in REQUIRED_SOURCE_FAMILIES:
        return explicit, "explicit_source_family"
    tag = str(entry.get("tag") or "").strip().casefold()
    mapping = {
        "speech": "speech",
        "music": "music",
        "esc50": "environment",
        "environment": "environment",
        "demand": "environment",
        "machine": "machine",
        "mimii": "machine",
    }
    if tag in mapping:
        return mapping[tag], f"legacy_tag:{tag}"
    return None, "missing_or_unknown"


def _split_from_entry(entry: Mapping[str, Any]) -> tuple[str | None, str]:
    value = str(entry.get("split") or "").strip().casefold()
    if value in {"train", "val", "test"}:
        return value, "explicit_split"
    return None, "missing_or_invalid"


def _declared_sha(entry: Mapping[str, Any]) -> str | None:
    for key in ("sha256", "content_sha256", "native_content_sha256"):
        value = entry.get(key)
        if value is not None and str(value).strip():
            text = str(value).strip()
            _require_sha(f"{key}", text)
            return text
    return None


def _declared_rate(entry: Mapping[str, Any]) -> int | None:
    for key in ("native_sample_rate_hz", "sample_rate_hz", "sample_rate"):
        value = entry.get(key)
        if value is not None and str(value).strip():
            if isinstance(value, bool):
                return None
            try:
                result = int(value)
            except (TypeError, ValueError):
                return None
            return result if result > 0 else None
    return None


def _probe_audio(path: Path) -> dict[str, Any]:
    try:
        with sf.SoundFile(path, mode="r") as handle:
            frames = int(handle.frames)
            sample_rate = int(handle.samplerate)
            channels = int(handle.channels)
            if frames <= 0 or sample_rate <= 0 or channels <= 0:
                raise ValueError("audio header frames/fs/channels가 양수가 아닙니다")
            first = handle.read(
                min(frames, 1024), dtype="float32", always_2d=True
            )
            if frames > 1024:
                handle.seek(max(0, frames - 1024))
                last = handle.read(1024, dtype="float32", always_2d=True)
            else:
                last = first
            if first.size == 0 or last.size == 0:
                raise ValueError("bounded decoder probe가 빈 PCM을 반환했습니다")
            if not np.all(np.isfinite(first)) or not np.all(np.isfinite(last)):
                raise ValueError("bounded decoder probe PCM이 finite하지 않습니다")
            return {
                "status": "PASS",
                "error": None,
                "sample_rate": sample_rate,
                "channels": channels,
                "frames": frames,
                "format": str(handle.format),
                "subtype": str(handle.subtype),
            }
    except (OSError, RuntimeError, ValueError, sf.LibsndfileError) as exc:
        return {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "sample_rate": None,
            "channels": None,
            "frames": None,
            "format": None,
            "subtype": None,
        }


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = _json_no_duplicates(raw, label=f"{path}:{line_number}")
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row가 object가 아닙니다")
            value = dict(value)
            value["__entry_index__"] = line_number
            rows.append(value)
    return rows


def _parse_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_family", "group_id", "path", "sample_rate_hz", "clips"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("source pool CSV 필수 header가 없습니다")
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(reader, start=2):
            value = dict(row)
            value["__entry_index__"] = index
            rows.append(value)
        return rows


def _legacy_schema(kind: str, rows: Sequence[Mapping[str, Any]]) -> str:
    if kind == "public_jsonl":
        return "legacy_public_audio_jsonl_unversioned"
    if kind == "recorded_jsonl":
        versions = {str(row.get("lineage_schema") or "unknown") for row in rows}
        suffix = "+".join(sorted(versions))
        return f"legacy_recorded_jsonl:{suffix}"
    if kind == "source_pool_csv":
        return "legacy_source_pool_csv_unversioned"
    return "unreferenced_audio_tree_no_manifest"


def _lineage_for_entry(
    *,
    entry: Mapping[str, Any],
    kind: str,
    declared_audio_path: str,
    resolver: _LineageResolver,
) -> tuple[tuple[str, ...], tuple[str, ...], list[str]]:
    keys: set[str] = set()
    basis: set[str] = set()
    blockers: list[str] = []
    explicit_component = str(entry.get("lineage_component_id") or "").strip()
    if explicit_component:
        keys.add(f"legacy_component:{explicit_component}")
        basis.add("explicit_legacy_lineage_component_id")
    group = str(entry.get("group_id") or "").strip()
    if group:
        keys.add(f"legacy_group:{group}")
        basis.add("explicit_legacy_group_id")
    source_pool_group = str(entry.get("source_pool_group_id") or group).strip()
    if source_pool_group and kind in {"recorded_jsonl", "source_pool_csv"}:
        keys.add(f"source_pool_group:{source_pool_group}")
        basis.add("source_pool_group_link")

    clips: list[str]
    raw_clips = entry.get("clips")
    if kind == "source_pool_csv":
        try:
            parsed = json.loads(str(raw_clips))
            if not isinstance(parsed, list) or not all(
                isinstance(item, str) and item.strip() for item in parsed
            ):
                raise ValueError("clips가 non-empty string list가 아닙니다")
            clips = list(parsed)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            clips = []
            blockers.append(f"source_pool_clips_invalid:{exc}")
    elif kind in {"public_jsonl", "unreferenced_audio_tree"}:
        clips = [declared_audio_path]
    else:
        clips = []
    for clip in clips:
        clip_keys, error = resolver.keys_for_clip(clip)
        if clip_keys:
            keys.update(clip_keys)
            basis.add("authoritative_legacy_metadata_mapping_candidate")
        elif error is not None:
            blockers.append(error)
    if not keys:
        blockers.append("semantic_lineage_absent")
    return tuple(sorted(keys)), tuple(sorted(basis)), blockers


def _candidate_draft(
    *,
    root: Path,
    spec: AvailabilityInputV3,
    entry: Mapping[str, Any],
    input_path: str,
    legacy_schema: str,
    resolver: _LineageResolver,
) -> dict[str, Any]:
    index = int(entry["__entry_index__"])
    family, family_basis = _family_from_entry(entry)
    split, split_basis = _split_from_entry(entry)
    kind = spec.kind
    declared = str(entry.get("path") or "").strip()
    origin_role = {
        "public_jsonl": "direct_corpus_native_candidate",
        "recorded_jsonl": "recorded_playback_composite",
        "source_pool_csv": "processed_source_pool_composite",
        "unreferenced_audio_tree": "unreferenced_native_candidate",
    }[kind]
    if kind == "unreferenced_audio_tree":
        family = spec.source_family
        family_basis = "scan_input_family"
    blockers: list[str] = [
        "legacy_or_unmanifested_mapping_only",
        "declared_source_sha_missing",
        "decoded_pcm_artifact_absent",
        "full_decode_not_verified",
        "causal_p_applied_err_absent",
        "density_not_recomputed",
    ]
    try:
        declared_sha = _declared_sha(entry)
    except ValueError as exc:
        declared_sha = None
        blockers.append(f"declared_sha_invalid:{exc}")
    if declared_sha is not None:
        blockers.remove("declared_source_sha_missing")
    declared_rate = _declared_rate(entry)

    if kind == "recorded_jsonl":
        base = (root / input_path).parent
        directory, _ = _resolve_within_root(
            root,
            declared,
            base=base if str(entry.get("path_base")) == "manifest" else root,
        )
        audio_value: str | Path = directory / "source.wav"
    else:
        audio_value = declared
    resolved_path: Path | None = None
    relative: str | None = None
    file_status: str
    try:
        resolved_path, relative = _resolve_within_root(root, audio_value)
        if resolved_path.is_file():
            file_status = "PRESENT"
        else:
            file_status = "MISSING"
            blockers.append("audio_file_missing")
    except (OSError, ValueError) as exc:
        file_status = "UNSAFE"
        blockers.append(f"audio_path_unsafe:{exc}")

    actual_size: int | None = None
    actual_sha: str | None = None
    declared_sha_match: bool | None = None
    probe = {
        "status": "NOT_RUN",
        "error": None,
        "sample_rate": None,
        "channels": None,
        "frames": None,
        "format": None,
        "subtype": None,
    }
    if file_status == "PRESENT" and resolved_path is not None:
        actual_size = resolved_path.stat().st_size
        if actual_size <= 0:
            file_status = "MISSING"
            blockers.append("audio_file_empty")
            actual_size = None
        else:
            actual_sha = _sha256_file(resolved_path)
            declared_sha_match = (
                actual_sha == declared_sha if declared_sha is not None else None
            )
            if declared_sha_match is False:
                blockers.append("declared_sha_mismatch")
            probe = _probe_audio(resolved_path)
            if probe["status"] != "PASS":
                blockers.append("decoder_bounded_probe_failed")
    actual_rate = probe["sample_rate"]
    rate_match = (
        actual_rate == declared_rate
        if actual_rate is not None and declared_rate is not None
        else None
    )
    if rate_match is False:
        blockers.append("declared_sample_rate_mismatch")
    header_nyquist = actual_rate / 2.0 if actual_rate is not None else None
    direct_native_role = origin_role in {
        "direct_corpus_native_candidate",
        "unreferenced_native_candidate",
    }
    native_nyquist = header_nyquist if direct_native_role else None
    control = BroadbandFullOctaveContractV3.canonical()
    physical_mask = (
        tuple(
            native_nyquist >= band[1]
            for band in control.physical_identification_subbands_hz
        )
        if native_nyquist is not None
        else None
    )
    objective_mask = (
        tuple(
            native_nyquist >= band[1]
            for band in control.equal_weight_octave_objective_bands_hz
        )
        if native_nyquist is not None
        else None
    )
    full_nyquist = bool(physical_mask and all(physical_mask))
    qualification_limitations: list[str] = []
    if physical_mask is not None and not all(physical_mask):
        qualification_limitations.append(
            "native_nyquist_partial_physical_band_coverage"
        )
    if objective_mask is not None and not all(objective_mask):
        qualification_limitations.append(
            "native_nyquist_partial_objective_octave_coverage"
        )
    semantic_keys, semantic_basis, lineage_blockers = _lineage_for_entry(
        entry=entry,
        kind=kind,
        declared_audio_path=declared,
        resolver=resolver,
    )
    blockers.extend(lineage_blockers)
    if family is None:
        blockers.append("source_family_missing_or_invalid")
    if split is None:
        blockers.append("split_missing_or_invalid")
    if origin_role in {
        "recorded_playback_composite",
        "processed_source_pool_composite",
    }:
        blockers.append("immutable_native_origin_not_bound")
    semantic_available = bool(semantic_keys)
    mapping_ready = (
        file_status == "PRESENT"
        and probe["status"] == "PASS"
        and declared_sha_match is not False
        and rate_match is not False
        and family is not None
        and split is not None
        and semantic_available
    )
    if mapping_ready:
        status = "MAPPING_CANDIDATE"
    elif file_status == "PRESENT" and probe["status"] == "PASS":
        status = "PARTIAL_MAPPING_CANDIDATE"
    else:
        status = "UNAVAILABLE"
    return {
        "candidate_id": f"{input_path}:{index}",
        "input_kind": kind,
        "input_path": input_path,
        "entry_index": index,
        "detected_legacy_schema": legacy_schema,
        "declared_audio_path": declared,
        "resolved_audio_path": relative,
        "origin_role": origin_role,
        "file_status": file_status,
        "actual_size_bytes": actual_size,
        "actual_sha256": actual_sha,
        "declared_sha256": declared_sha,
        "declared_sha_matches": declared_sha_match,
        "decoder_probe_status": probe["status"],
        "decoder_probe_scope": (
            "header_first_last_1024_frames" if probe["status"] == "PASS" else "none"
        ),
        "decoder_error": probe["error"],
        "actual_sample_rate_hz": actual_rate,
        "actual_header_nyquist_hz": header_nyquist,
        "actual_channels": probe["channels"],
        "actual_frames": probe["frames"],
        "actual_format": probe["format"],
        "actual_subtype": probe["subtype"],
        "declared_sample_rate_hz": declared_rate,
        "declared_sample_rate_matches": rate_match,
        "native_nyquist_verified": native_nyquist is not None,
        "native_nyquist_hz": native_nyquist,
        "native_physical_nyquist_coverage": physical_mask,
        "native_objective_octave_nyquist_coverage": objective_mask,
        "full_target_native_nyquist": full_nyquist,
        "source_family": family,
        "source_family_basis": family_basis,
        "split": split,
        "split_basis": split_basis,
        "semantic_lineage_keys": semantic_keys,
        "semantic_lineage_basis": semantic_basis,
        "semantic_lineage_available": semantic_available,
        "availability_status": status,
        "qualification_limitations": tuple(sorted(qualification_limitations)),
        "blockers": tuple(sorted(set(blockers))),
        "_lineage_keys": semantic_keys,
    }


def _read_manifest_input(
    root: Path,
    spec: AvailabilityInputV3,
    *,
    resolver: _LineageResolver,
) -> tuple[AvailabilityInputEvidenceV3, list[dict[str, Any]]]:
    try:
        path, relative = _resolve_within_root(root, spec.path)
    except ValueError as exc:
        return (
            AvailabilityInputEvidenceV3(
                kind=spec.kind,
                path=spec.path,
                source_family=spec.source_family,
                status="INVALID",
                file=None,
                detected_legacy_schema="unreadable_input",
                entries_seen=0,
                entries_emitted=0,
                error=str(exc),
            ),
            [],
        )
    if not path.is_file():
        return (
            AvailabilityInputEvidenceV3(
                kind=spec.kind,
                path=relative,
                source_family=spec.source_family,
                status="MISSING",
                file=None,
                detected_legacy_schema="missing_input",
                entries_seen=0,
                entries_emitted=0,
                error="input manifest absent",
            ),
            [],
        )
    try:
        rows = _parse_csv(path) if spec.kind == "source_pool_csv" else _parse_jsonl(path)
        schema = _legacy_schema(spec.kind, rows)
        drafts = [
            _candidate_draft(
                root=root,
                spec=spec,
                entry=row,
                input_path=relative,
                legacy_schema=schema,
                resolver=resolver,
            )
            for row in rows
        ]
        evidence = AvailabilityInputEvidenceV3(
            kind=spec.kind,
            path=relative,
            source_family=spec.source_family,
            status="PASS",
            file=_manifest_file_evidence(root, relative),
            detected_legacy_schema=schema,
            entries_seen=len(rows),
            entries_emitted=len(drafts),
            error=None,
        )
        return evidence, drafts
    except (OSError, ValueError, json.JSONDecodeError, csv.Error) as exc:
        return (
            AvailabilityInputEvidenceV3(
                kind=spec.kind,
                path=relative,
                source_family=spec.source_family,
                status="INVALID",
                file=(
                    _manifest_file_evidence(root, relative) if path.stat().st_size > 0 else None
                ),
                detected_legacy_schema="invalid_input",
                entries_seen=0,
                entries_emitted=0,
                error=f"{type(exc).__name__}: {exc}",
            ),
            [],
        )


def _read_tree_input(
    root: Path,
    spec: AvailabilityInputV3,
    *,
    resolver: _LineageResolver,
    referenced_paths: set[str],
) -> tuple[AvailabilityInputEvidenceV3, list[dict[str, Any]]]:
    try:
        path, relative = _resolve_within_root(root, spec.path)
    except ValueError as exc:
        return (
            AvailabilityInputEvidenceV3(
                kind=spec.kind,
                path=spec.path,
                source_family=spec.source_family,
                status="INVALID",
                file=None,
                detected_legacy_schema="unreadable_tree",
                entries_seen=0,
                entries_emitted=0,
                error=str(exc),
            ),
            [],
        )
    if not path.is_dir():
        return (
            AvailabilityInputEvidenceV3(
                kind=spec.kind,
                path=relative,
                source_family=spec.source_family,
                status="MISSING",
                file=None,
                detected_legacy_schema="missing_audio_tree",
                entries_seen=0,
                entries_emitted=0,
                error="audio tree absent",
            ),
            [],
        )
    audio_paths = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and not candidate.is_symlink()
        and candidate.suffix.casefold() in _AUDIO_SUFFIXES
    )
    unreferenced = [
        candidate
        for candidate in audio_paths
        if candidate.relative_to(root).as_posix() not in referenced_paths
    ]
    schema = _legacy_schema(spec.kind, [])
    drafts: list[dict[str, Any]] = []
    for index, candidate in enumerate(unreferenced, start=1):
        row = {
            "__entry_index__": index,
            "path": candidate.relative_to(root).as_posix(),
            "source_family": spec.source_family,
        }
        drafts.append(
            _candidate_draft(
                root=root,
                spec=spec,
                entry=row,
                input_path=relative,
                legacy_schema=schema,
                resolver=resolver,
            )
        )
    evidence = AvailabilityInputEvidenceV3(
        kind=spec.kind,
        path=relative,
        source_family=spec.source_family,
        status="PASS",
        file=None,
        detected_legacy_schema=schema,
        entries_seen=len(audio_paths),
        entries_emitted=len(drafts),
        error=None,
    )
    return evidence, drafts


def _files_containing_schema(root: Path, token: bytes) -> tuple[str, ...]:
    hits: list[str] = []
    excluded = {".git", ".venv", "node_modules", "__pycache__"}
    for path in root.rglob("*.json"):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if any(part in excluded for part in relative.parts) or path.is_symlink():
            continue
        try:
            with path.open("rb") as handle:
                tail = b""
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    combined = tail + chunk
                    if token in combined:
                        hits.append(relative.as_posix())
                        break
                    tail = combined[-max(0, len(token) - 1) :]
        except OSError:
            continue
    return tuple(sorted(hits))


def _audit_causal_p(
    root: Path,
    requested: str | None,
) -> CausalPAvailabilityV3:
    discovered = _files_containing_schema(
        root, b"broadband_population_causal_primary_operator_v3"
    )
    if requested is None:
        return CausalPAvailabilityV3(
            requested_path=None,
            discovered_operator_payload_paths=discovered,
            status="MISSING",
            candidate_file=None,
            fir_file=None,
            operator=None,
            blockers=(
                "canonical fullband causal P authority path was not supplied",
                "POPULATION_V3_AUTHORITY is None",
            ),
        )
    try:
        path, relative = _resolve_within_root(root, requested)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(relative)
        raw = path.read_bytes()
        payload = _json_no_duplicates(raw, label=relative)
        if isinstance(payload, dict) and set(payload) == {"operator"}:
            payload = payload["operator"]
        operator = CausalPrimaryOperatorV3.model_validate(payload)
        fir_path, fir_relative = _resolve_within_root(root, operator.fir_file.path)
        if not fir_path.is_file() or fir_path.is_symlink():
            raise FileNotFoundError(fir_relative)
        if fir_path.stat().st_size != operator.fir_file.size_bytes:
            raise ValueError("causal P FIR size가 operator payload와 다릅니다")
        if _sha256_file(fir_path) != operator.fir_file.sha256:
            raise ValueError("causal P FIR SHA가 operator payload와 다릅니다")
        if fir_path.stat().st_size % 4:
            raise ValueError("causal P FIR raw size가 float32 배수가 아닙니다")
        taps = np.fromfile(fir_path, dtype="<f4")
        if taps.size == 0 or not np.all(np.isfinite(taps)):
            raise ValueError("causal P FIR이 비었거나 non-finite입니다")
        return CausalPAvailabilityV3(
            requested_path=relative,
            discovered_operator_payload_paths=discovered,
            status="STRUCTURAL_ONLY",
            candidate_file=LocalEvidenceFileV3(
                path=relative,
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            ),
            fir_file=LocalEvidenceFileV3(
                path=fir_relative,
                size_bytes=fir_path.stat().st_size,
                sha256=operator.fir_file.sha256,
            ),
            operator=operator,
            blockers=(
                "operator bytes are structurally valid but no external authority issuer exists",
                "POPULATION_V3_AUTHORITY is None",
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return CausalPAvailabilityV3(
            requested_path=requested,
            discovered_operator_payload_paths=discovered,
            status="INVALID",
            candidate_file=None,
            fir_file=None,
            operator=None,
            blockers=(f"causal P candidate invalid: {type(exc).__name__}: {exc}",),
        )


def _assign_mapping_components(drafts: list[dict[str, Any]]) -> None:
    dsu = _DSU()
    candidate_nodes: list[str] = []
    for draft in drafts:
        node = f"candidate:{draft['candidate_id']}"
        candidate_nodes.append(node)
        keys = list(draft.pop("_lineage_keys"))
        if draft["actual_sha256"] is not None:
            keys.append(f"content_sha256:{draft['actual_sha256']}")
        if not keys:
            draft["mapping_component_id"] = None
            continue
        for key in keys:
            dsu.union(node, key)
    members: dict[str, set[str]] = defaultdict(set)
    for value in dsu.parent:
        members[dsu.find(value)].add(value)
    component_ids = {
        root: hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()
        for root, values in members.items()
    }
    for draft, node in zip(drafts, candidate_nodes, strict=True):
        draft["mapping_component_id"] = (
            component_ids[dsu.find(node)] if node in dsu.parent else None
        )


def _summary_and_cells(
    candidates: Sequence[SourceAvailabilityCandidateV3],
    *,
    manifest_entries_total: int,
) -> tuple[
    AvailabilitySummaryV3,
    tuple[AvailabilityCellV3, ...],
    tuple[LineageMappingConflictV3, ...],
]:
    content_counts = Counter(
        candidate.actual_sha256
        for candidate in candidates
        if candidate.actual_sha256 is not None
    )
    component_splits: dict[str, set[str]] = defaultdict(set)
    component_families: dict[str, set[str]] = defaultdict(set)
    present_component_splits: dict[str, set[str]] = defaultdict(set)
    present_component_families: dict[str, set[str]] = defaultdict(set)
    component_candidates: dict[str, list[SourceAvailabilityCandidateV3]] = defaultdict(list)
    for candidate in candidates:
        component = candidate.mapping_component_id
        if component is None:
            continue
        component_candidates[component].append(candidate)
        if candidate.split:
            component_splits[component].add(candidate.split)
            if candidate.file_status == "PRESENT":
                present_component_splits[component].add(candidate.split)
        if candidate.source_family:
            component_families[component].add(candidate.source_family)
            if candidate.file_status == "PRESENT":
                present_component_families[component].add(candidate.source_family)
    conflicts: list[LineageMappingConflictV3] = []
    for component in sorted(component_candidates):
        splits = tuple(sorted(component_splits[component]))
        families = tuple(sorted(component_families[component]))
        axes: list[str] = []
        if len(splits) > 1:
            axes.append("split")
        if len(families) > 1:
            axes.append("source_family")
        if not axes:
            continue
        selected = component_candidates[component]
        conflicts.append(
            LineageMappingConflictV3(
                mapping_component_id=component,
                conflict_axes=tuple(axes),
                splits=splits,
                source_families=families,
                candidate_count=len(selected),
                present_candidate_count=sum(
                    candidate.file_status == "PRESENT" for candidate in selected
                ),
                mapping_candidate_count=sum(
                    candidate.availability_status == "MAPPING_CANDIDATE"
                    for candidate in selected
                ),
                input_paths=tuple(sorted({candidate.input_path for candidate in selected})),
            )
        )
    cells: list[AvailabilityCellV3] = []
    for split in ("train", "val", "test"):
        for family in REQUIRED_SOURCE_FAMILIES:
            selected = [
                candidate
                for candidate in candidates
                if candidate.split == split and candidate.source_family == family
            ]
            mapping = [
                candidate
                for candidate in selected
                if candidate.availability_status == "MAPPING_CANDIDATE"
            ]
            cells.append(
                AvailabilityCellV3(
                    split=split,
                    source_family=family,
                    rows=len(selected),
                    present_decodable_rows=sum(
                        candidate.decoder_probe_status == "PASS"
                        for candidate in selected
                    ),
                    mapping_candidate_rows=len(mapping),
                    full_target_nyquist_mapping_rows=sum(
                        candidate.full_target_native_nyquist for candidate in mapping
                    ),
                    mapping_lineage_components=len(
                        {
                            candidate.mapping_component_id
                            for candidate in mapping
                            if candidate.mapping_component_id is not None
                        }
                    ),
                    mapping_native_components_per_physical_band=tuple(
                        len(
                            {
                                candidate.mapping_component_id
                                for candidate in mapping
                                if candidate.mapping_component_id is not None
                                and candidate.native_physical_nyquist_coverage is not None
                                and candidate.native_physical_nyquist_coverage[band_index]
                            }
                        )
                        for band_index in range(8)
                    ),
                    mapping_native_components_per_objective_octave=tuple(
                        len(
                            {
                                candidate.mapping_component_id
                                for candidate in mapping
                                if candidate.mapping_component_id is not None
                                and candidate.native_objective_octave_nyquist_coverage is not None
                                and candidate.native_objective_octave_nyquist_coverage[band_index]
                            }
                        )
                        for band_index in range(7)
                    ),
                    canonical_qualified_components_per_physical_band=(0,) * 8,
                    canonical_qualified_components_per_objective_octave=(0,) * 7,
                )
            )
    direct_roles = {
        "direct_corpus_native_candidate",
        "unreferenced_native_candidate",
    }
    summary = AvailabilitySummaryV3(
        manifest_entries_total=manifest_entries_total,
        candidates_reported=len(candidates),
        unique_resolved_audio_paths=len(
            {
                candidate.resolved_audio_path
                for candidate in candidates
                if candidate.resolved_audio_path is not None
            }
        ),
        files_present=sum(candidate.file_status == "PRESENT" for candidate in candidates),
        files_missing=sum(candidate.file_status == "MISSING" for candidate in candidates),
        unsafe_paths=sum(candidate.file_status == "UNSAFE" for candidate in candidates),
        decoder_probe_pass=sum(
            candidate.decoder_probe_status == "PASS" for candidate in candidates
        ),
        decoder_probe_fail=sum(
            candidate.decoder_probe_status == "FAIL" for candidate in candidates
        ),
        declared_sha_present=sum(
            candidate.declared_sha256 is not None for candidate in candidates
        ),
        declared_sha_match=sum(
            candidate.declared_sha_matches is True for candidate in candidates
        ),
        declared_sha_mismatch=sum(
            candidate.declared_sha_matches is False for candidate in candidates
        ),
        source_family_mapped=sum(
            candidate.source_family is not None for candidate in candidates
        ),
        split_mapped=sum(candidate.split is not None for candidate in candidates),
        semantic_lineage_mapped=sum(
            candidate.semantic_lineage_available for candidate in candidates
        ),
        mapping_candidates=sum(
            candidate.availability_status == "MAPPING_CANDIDATE"
            for candidate in candidates
        ),
        full_target_native_nyquist_rows=sum(
            candidate.full_target_native_nyquist for candidate in candidates
        ),
        direct_native_present=sum(
            candidate.origin_role in direct_roles and candidate.file_status == "PRESENT"
            for candidate in candidates
        ),
        direct_native_full_target_nyquist=sum(
            candidate.origin_role in direct_roles
            and candidate.file_status == "PRESENT"
            and candidate.full_target_native_nyquist
            for candidate in candidates
        ),
        content_sha_duplicate_groups=sum(value > 1 for value in content_counts.values()),
        mapping_components_crossing_splits=sum(
            len(splits) > 1 for splits in component_splits.values()
        ),
        mapping_components_crossing_families=sum(
            len(families) > 1 for families in component_families.values()
        ),
        present_components_crossing_splits=sum(
            len(splits) > 1 for splits in present_component_splits.values()
        ),
        present_components_crossing_families=sum(
            len(families) > 1 for families in present_component_families.values()
        ),
    )
    return summary, tuple(cells), tuple(conflicts)


def audit_population_v3_availability(
    *,
    repository_root: str | Path,
    inputs: Sequence[AvailabilityInputV3],
    causal_p_authority_path: str | None = None,
) -> PopulationAvailabilityReportV3:
    """실제 로컬 bytes를 inventory하되 v3 manifest 발행은 항상 닫는다."""

    root = Path(repository_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repository_root가 directory가 아닙니다")
    specs = tuple(AvailabilityInputV3.model_validate(value) for value in inputs)
    if not specs:
        raise ValueError("최소 한 개 availability input이 필요합니다")
    resolver = _LineageResolver(root)
    evidence_by_index: dict[int, AvailabilityInputEvidenceV3] = {}
    drafts: list[dict[str, Any]] = []
    manifest_entries_total = 0
    referenced_paths: set[str] = set()

    for index, spec in enumerate(specs):
        if spec.kind == "unreferenced_audio_tree":
            continue
        evidence, current = _read_manifest_input(root, spec, resolver=resolver)
        evidence_by_index[index] = evidence
        manifest_entries_total += evidence.entries_seen
        drafts.extend(current)
        referenced_paths.update(
            str(item["resolved_audio_path"])
            for item in current
            if item["resolved_audio_path"] is not None
        )
    for index, spec in enumerate(specs):
        if spec.kind != "unreferenced_audio_tree":
            continue
        evidence, current = _read_tree_input(
            root,
            spec,
            resolver=resolver,
            referenced_paths=referenced_paths,
        )
        evidence_by_index[index] = evidence
        drafts.extend(current)
    _assign_mapping_components(drafts)
    candidates = tuple(SourceAvailabilityCandidateV3.model_validate(item) for item in drafts)
    summary, cells, lineage_conflicts = _summary_and_cells(
        candidates,
        manifest_entries_total=manifest_entries_total,
    )
    causal_p = _audit_causal_p(root, causal_p_authority_path)
    input_evidence = tuple(evidence_by_index[index] for index in range(len(specs)))
    blockers = [
        "canonical fullband causal P authority is absent",
        "population-v3 decoded PCM artifacts are absent",
        "P-applied ERR and per-band density were not recomputed",
        "legacy v1/v2 and unmanifested inputs are mapping-only",
        "POPULATION_V3_AUTHORITY is None",
        "untouched Level-5 challenge remains separate and absent",
    ]
    blockers.extend(
        f"input {item.path} is {item.status}" for item in input_evidence if item.status != "PASS"
    )
    control = BroadbandFullOctaveContractV3.canonical()
    population = PopulationCoverageContractV3.canonical()
    payload: dict[str, Any] = {
        "control_band_contract": control,
        "control_band_contract_sha256": control.digest(),
        "population_contract": population,
        "population_contract_sha256": population.digest(),
        "inputs": input_evidence,
        "metadata_evidence": tuple(
            resolver.evidence[key] for key in sorted(resolver.evidence)
        ),
        "causal_primary": causal_p,
        "candidates": candidates,
        "lineage_mapping_conflicts": lineage_conflicts,
        "cells": cells,
        "summary": summary,
        "blockers": tuple(blockers),
        "safety": {
            "audio_devices_opened": False,
            "speaker_output_count": 0,
            "network_access": False,
            "drive_access": False,
            "elice_access": False,
            "source_files_modified": False,
            "decoder_probe": "header plus first/last at most 1024 frames",
        },
    }
    serializable = {
        key: (
            value.model_dump(mode="json")
            if isinstance(value, BaseModel)
            else [
                item.model_dump(mode="json") if isinstance(item, BaseModel) else item
                for item in value
            ]
            if isinstance(value, tuple)
            else value
        )
        for key, value in payload.items()
    }
    serializable.update(
        {
            "schema_version": "broadband_population_availability_audit_v3_bandwise_v1",
            "role": "read_only_mapping_availability_not_population_manifest",
            "status": "BLOCKED",
            "authority": None,
            "canonical_population_manifest_issued": False,
            "decoded_pcm_issued": False,
            "p_applied_err_issued": False,
            "density_recomputed": False,
            "legacy_v1_v2_automatic_promotion": False,
        }
    )
    serializable["evidence_sha256"] = _digest(serializable)
    return PopulationAvailabilityReportV3.model_validate(serializable)


__all__ = [
    "AvailabilityCellV3",
    "AvailabilityInputEvidenceV3",
    "AvailabilityInputV3",
    "AvailabilitySummaryV3",
    "CausalPAvailabilityV3",
    "LineageMappingConflictV3",
    "MetadataEvidenceV3",
    "PopulationAvailabilityReportV3",
    "PopulationAvailabilityV3Blocked",
    "SourceAvailabilityCandidateV3",
    "audit_population_v3_availability",
]
