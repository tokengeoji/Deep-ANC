"""자연 분포를 보존하는 broadband-v3 population/source coverage 계약.

v2의 ``한 clip의 9개 구간 중 8개가 일곱 대역을 모두 통과`` 조건은 speech/music의
자연스러운 spectral sparsity를 거부하고 shaped all-band source만 남긴다. v3는 density
하한 0.25, 실제 P-applied ERR 재계산, 독립 lineage 하한을 낮추지 않는다. 대신 자격의
단위를 ``clip 전체``에서 ``(component, item, band)``로 옮긴다.

* 한 item은 실제 에너지가 있는 일부 physical/objective band에만 자격을 얻을 수 있다.
* split×family×band마다 서로 다른 lineage component가 최소 4개여야 한다.
* family-balanced batch마다 physical 8구간과 objective octave 7구간 각각 valid item이
  최소 4개여야 한다.

현재 실제 source bytes와 canonical fullband causal P authority가 없으므로
``POPULATION_V3_AUTHORITY``는 ``None``이고 어떤 structural fixture도 학습 authority가
되지 않는다. 이 모듈은 네트워크나 오디오 장치를 사용하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.signal import fftconvolve

from ..dsp.control_band_contract import (
    BroadbandFullOctaveContractV3,
    REQUIRED_SOURCE_FAMILIES,
)


_FROZEN = ConfigDict(frozen=True, extra="forbid")
_HEX = frozenset("0123456789abcdef")
REQUIRED_SPLITS = ("train", "val", "test")
MIN_DENSITY_RATIO = 0.25
MIN_COMPONENTS_PER_SPLIT_FAMILY_BAND = 4
MIN_VALID_ITEMS_PER_BATCH_BAND = 4
MAX_QUALIFIED_ITEMS_V3 = 65_536
MAX_BATCH_SIZE_V3 = 256
STRUCTURAL_BATCH_SEARCH_ATTEMPTS_V3 = 256
POPULATION_V3_SCAFFOLD_BLOCKERS = (
    "POPULATION_V3_AUTHORITY is None",
    "EXTERNAL_MANIFEST_AUTHORITY_NOT_BOUND",
    "CONNECTED_COMPONENT_AUTHORITY_NOT_BOUND",
    "INTERVAL_ALIAS_AUTHORITY_NOT_BOUND",
    "LOCAL_FILE_RECOMPUTATION_IS_NOT_EXTERNAL_RAW_AUTHORITY",
)
POPULATION_V3_AUTHORITY: dict[str, str] | None = None


class PopulationV3Blocked(RuntimeError):
    """v3 population/batch를 canonical authority로 사용할 수 없음."""


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
    text = str(value)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{name}가 lowercase SHA-256이 아닙니다")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PopulationCoverageContractV3(BaseModel):
    """v1/v2를 수정하지 않는 별도 immutable population 계약."""

    model_config = _FROZEN

    schema_version: Literal["broadband_population_coverage_contract_v3"] = (
        "broadband_population_coverage_contract_v3"
    )
    role: Literal["population_and_batch_band_coverage"] = (
        "population_and_batch_band_coverage"
    )
    control_band_contract: BroadbandFullOctaveContractV3
    control_band_contract_sha256: str
    required_splits: tuple[Literal["train", "val", "test"], ...] = REQUIRED_SPLITS
    required_families: tuple[str, ...] = REQUIRED_SOURCE_FAMILIES
    minimum_density_ratio: Literal[0.25] = MIN_DENSITY_RATIO
    minimum_independent_components_per_split_family_physical_band: Literal[4] = 4
    minimum_independent_components_per_split_family_objective_octave: Literal[4] = 4
    minimum_valid_items_per_batch_physical_band: Literal[4] = 4
    minimum_valid_items_per_batch_objective_octave: Literal[4] = 4
    candidate_may_qualify_partial_bands: Literal[True] = True
    all_bands_per_clip_required: Literal[False] = False
    adaptive_eq_or_band_shaping_allowed: Literal[False] = False
    repeat_or_loop_allowed: Literal[False] = False
    native_nyquist_checked_per_qualified_band: Literal[True] = True
    decoded_pcm_and_sha_required: Literal[True] = True
    p_applied_err_recomputation_required: Literal[True] = True
    legacy_v2_automatic_promotion_allowed: Literal[False] = False
    untouched_level5_challenge_separate_required: Literal[True] = True
    authority: None = None

    @model_validator(mode="after")
    def _validate_contract(self) -> "PopulationCoverageContractV3":
        canonical = BroadbandFullOctaveContractV3.canonical()
        if self.control_band_contract.model_dump(mode="json") != canonical.model_dump(
            mode="json"
        ):
            raise ValueError("population v3에는 exact canonical Broadband v3가 필요합니다")
        if self.control_band_contract_sha256 != self.control_band_contract.digest():
            raise ValueError("inline Broadband v3 payload와 SHA가 다릅니다")
        if tuple(self.required_splits) != REQUIRED_SPLITS:
            raise ValueError("population v3 split은 train/val/test 정확히 세 개입니다")
        if tuple(self.required_families) != REQUIRED_SOURCE_FAMILIES:
            raise ValueError("population v3 family 집합이 다릅니다")
        return self

    @classmethod
    def canonical(cls) -> "PopulationCoverageContractV3":
        control = BroadbandFullOctaveContractV3.canonical()
        return cls(
            control_band_contract=control,
            control_band_contract_sha256=control.digest(),
        )

    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class LocalFileReferenceV3(BaseModel):
    model_config = _FROZEN

    path: str
    size_bytes: int = Field(gt=0)
    sha256: str

    @model_validator(mode="after")
    def _validate_ref(self) -> "LocalFileReferenceV3":
        candidate = Path(self.path)
        if not self.path or candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("file reference는 정규화된 repository 상대경로여야 합니다")
        _require_sha("file sha256", self.sha256)
        return self


class CausalPrimaryOperatorV3(BaseModel):
    model_config = _FROZEN

    schema_version: Literal["broadband_population_causal_primary_operator_v3"] = (
        "broadband_population_causal_primary_operator_v3"
    )
    role: Literal["physical_fullband_causal_primary_for_population"] = (
        "physical_fullband_causal_primary_for_population"
    )
    control_band_contract_sha256: str
    sample_rate_hz: Literal[48_000] = 48_000
    fir_file: LocalFileReferenceV3
    fir_dtype: Literal["little_endian_float32_mono_raw"] = (
        "little_endian_float32_mono_raw"
    )
    delay_samples: int = Field(ge=0)
    verified_lower_hz: float
    verified_upper_hz: float
    causal: Literal[True] = True
    physical_measurement: Literal[True] = True
    canonical_training_eligible: Literal[True] = True
    operator_receipt_sha256: str

    @model_validator(mode="after")
    def _validate_operator(self) -> "CausalPrimaryOperatorV3":
        control = BroadbandFullOctaveContractV3.canonical()
        if self.control_band_contract_sha256 != control.digest():
            raise ValueError("causal P가 exact Broadband v3 계약과 다릅니다")
        if (
            not math.isfinite(self.verified_lower_hz)
            or not math.isfinite(self.verified_upper_hz)
            or self.verified_lower_hz < 0.0
            or self.verified_upper_hz <= self.verified_lower_hz
        ):
            raise ValueError("causal P verified band가 finite한 양의 구간이 아닙니다")
        if self.verified_lower_hz > control.physical_identification_subbands_hz[0][0]:
            raise ValueError("causal P가 125 Hz octave 하단을 덮지 않습니다")
        if self.verified_upper_hz < control.physical_identification_subbands_hz[-1][1]:
            raise ValueError("causal P가 8 kHz octave 상단을 덮지 않습니다")
        _require_sha("operator_receipt_sha256", self.operator_receipt_sha256)
        return self


class PopulationItemClaimV3(BaseModel):
    model_config = _FROZEN

    item_id: str
    start_frame: int = Field(ge=0)
    n_frames: int = Field(gt=0)
    physical_density_ratios: tuple[float, ...]
    physical_valid_bands: tuple[bool, ...]
    objective_octave_density_ratios: tuple[float, ...]
    objective_octave_valid_bands: tuple[bool, ...]

    @model_validator(mode="after")
    def _validate_item(self) -> "PopulationItemClaimV3":
        if not self.item_id.strip():
            raise ValueError("item_id가 비었습니다")
        if len(self.physical_density_ratios) != 8 or len(self.physical_valid_bands) != 8:
            raise ValueError("physical band claim은 정확히 8개여야 합니다")
        if (
            len(self.objective_octave_density_ratios) != 7
            or len(self.objective_octave_valid_bands) != 7
        ):
            raise ValueError("objective octave claim은 정확히 7개여야 합니다")
        values = (*self.physical_density_ratios, *self.objective_octave_density_ratios)
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise ValueError("density ratio는 유한한 0 이상이어야 합니다")
        return self


class PopulationCandidateV3(BaseModel):
    model_config = _FROZEN

    candidate_id: str
    split: Literal["train", "val", "test"]
    source_family: Literal["speech", "music", "environment", "machine"]
    lineage_component_id: str
    immutable_native_source: LocalFileReferenceV3
    native_sample_rate_hz: int = Field(gt=0)
    native_nyquist_hz: float
    native_probe_receipt_sha256: str
    decoded_pcm_file: LocalFileReferenceV3
    decoded_pcm_dtype: Literal["little_endian_float32_mono_raw"] = (
        "little_endian_float32_mono_raw"
    )
    decoded_sample_rate_hz: Literal[48_000] = 48_000
    decoded_frames: int = Field(gt=0)
    decoded_transform_receipt_sha256: str
    p_applied_err_file: LocalFileReferenceV3
    p_applied_err_dtype: Literal["little_endian_float32_mono_raw"] = (
        "little_endian_float32_mono_raw"
    )
    p_operator_receipt_sha256: str
    valid_prefix_samples: int = Field(ge=0)
    adaptive_eq_or_band_shaping: Literal[False] = False
    repeated_or_looped: Literal[False] = False
    unmodified_level5_challenge: Literal[False] = False
    legacy_v2_promoted: Literal[False] = False
    items: tuple[PopulationItemClaimV3, ...]

    @model_validator(mode="after")
    def _validate_candidate(self) -> "PopulationCandidateV3":
        if not self.candidate_id.strip() or not self.lineage_component_id.strip():
            raise ValueError("candidate/lineage component ID가 비었습니다")
        expected_nyquist = self.native_sample_rate_hz / 2.0
        if not math.isclose(
            self.native_nyquist_hz, expected_nyquist, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise ValueError("native Nyquist가 native sample rate/2와 다릅니다")
        for name in (
            "native_probe_receipt_sha256",
            "decoded_transform_receipt_sha256",
            "p_operator_receipt_sha256",
        ):
            _require_sha(name, str(getattr(self, name)))
        if not self.items:
            raise ValueError("candidate에는 최소 한 개의 item이 필요합니다")
        ids = [item.item_id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate item_id가 중복됐습니다")
        return self


class UntouchedLevel5PolicyV3(BaseModel):
    model_config = _FROZEN

    schema_version: Literal["untouched_level5_challenge_reservation_v3"] = (
        "untouched_level5_challenge_reservation_v3"
    )
    candidate_ids_in_population: tuple[()] = ()
    training_use_allowed: Literal[False] = False
    validation_or_model_selection_use_allowed: Literal[False] = False
    required_after_model_and_contract_lock: Literal[True] = True
    required_families: tuple[str, ...] = REQUIRED_SOURCE_FAMILIES
    reservation_receipt_sha256: str

    @model_validator(mode="after")
    def _validate_level5(self) -> "UntouchedLevel5PolicyV3":
        if tuple(self.required_families) != REQUIRED_SOURCE_FAMILIES:
            raise ValueError("Level-5 challenge family 집합이 다릅니다")
        _require_sha("reservation_receipt_sha256", self.reservation_receipt_sha256)
        return self


class PopulationManifestV3(BaseModel):
    model_config = _FROZEN

    schema_version: Literal["broadband_population_source_manifest_v3"] = (
        "broadband_population_source_manifest_v3"
    )
    role: Literal["population_source_coverage_not_level5_challenge"] = (
        "population_source_coverage_not_level5_challenge"
    )
    contract: PopulationCoverageContractV3
    contract_sha256: str
    causal_primary: CausalPrimaryOperatorV3
    segment_samples: int = Field(ge=4096)
    candidates: tuple[PopulationCandidateV3, ...]
    untouched_level5_policy: UntouchedLevel5PolicyV3
    legacy_v2_manifest_sha256: None = None
    legacy_v2_automatic_promotion: Literal[False] = False

    @model_validator(mode="after")
    def _validate_manifest(self) -> "PopulationManifestV3":
        canonical = PopulationCoverageContractV3.canonical()
        if self.contract.model_dump(mode="json") != canonical.model_dump(mode="json"):
            raise ValueError("manifest가 exact canonical population v3 계약이 아닙니다")
        if self.contract_sha256 != self.contract.digest():
            raise ValueError("population contract payload/SHA가 다릅니다")
        if self.segment_samples % 256:
            raise ValueError("population item 길이는 256의 배수여야 합니다")
        if not self.candidates:
            raise ValueError("population candidates가 비었습니다")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate_id가 중복됐습니다")
        item_ids = [
            item.item_id for candidate in self.candidates for item in candidate.items
        ]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("population 전체에서 item_id가 중복됐습니다")
        return self

    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class QualifiedPopulationItemV3(BaseModel):
    model_config = _FROZEN

    item_id: str
    candidate_id: str
    split: Literal["train", "val", "test"]
    source_family: Literal["speech", "music", "environment", "machine"]
    lineage_component_id: str
    start_frame: int = Field(ge=0)
    n_frames: int = Field(gt=0)
    physical_valid_bands: tuple[bool, ...]
    objective_octave_valid_bands: tuple[bool, ...]

    @model_validator(mode="after")
    def _validate_qualified(self) -> "QualifiedPopulationItemV3":
        if not self.item_id.strip() or not self.candidate_id.strip():
            raise ValueError("qualified item/candidate ID가 비었습니다")
        if not self.lineage_component_id.strip():
            raise ValueError("qualified lineage component ID가 비었습니다")
        if len(self.physical_valid_bands) != 8:
            raise ValueError("qualified physical mask는 정확히 8개여야 합니다")
        if len(self.objective_octave_valid_bands) != 7:
            raise ValueError("qualified objective mask는 정확히 7개여야 합니다")
        if not any(self.physical_valid_bands) or not any(
            self.objective_octave_valid_bands
        ):
            raise ValueError("qualified item은 실제 유효 band가 필요합니다")
        return self


class PopulationCoverageRowV3(BaseModel):
    model_config = _FROZEN

    split: str
    source_family: str
    band_role: Literal["physical_identification", "objective_octave"]
    band_index: int = Field(ge=0)
    band_hz: tuple[float, float]
    independent_lineage_components: int = Field(ge=0)
    minimum_required: Literal[4] = 4
    passed: bool

    @model_validator(mode="after")
    def _validate_coverage_row(self) -> "PopulationCoverageRowV3":
        lower, upper = self.band_hz
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("coverage band가 finite한 증가 구간이 아닙니다")
        if self.passed != (self.independent_lineage_components >= 4):
            raise ValueError("coverage PASS와 independent component 수가 모순됩니다")
        return self


class PopulationAuditV3(BaseModel):
    model_config = _FROZEN

    schema_version: Literal["broadband_population_source_audit_v3"] = (
        "broadband_population_source_audit_v3"
    )
    role: Literal["local_recomputation_scaffold_not_external_raw_authority"] = (
        "local_recomputation_scaffold_not_external_raw_authority"
    )
    contract: PopulationCoverageContractV3
    contract_sha256: str
    manifest_sha256: str
    structural_status: Literal["PASS", "BLOCKED"]
    canonical_status: Literal["BLOCKED"] = "BLOCKED"
    authority: None = None
    external_manifest_authority_bound: Literal[False] = False
    connected_component_authority_bound: Literal[False] = False
    interval_alias_authority_bound: Literal[False] = False
    actual_raw_manifest_authority_bound: Literal[False] = False
    blockers: tuple[str, ...]
    candidates_verified: int = Field(gt=0)
    qualified_items: tuple[QualifiedPopulationItemV3, ...] = Field(
        max_length=MAX_QUALIFIED_ITEMS_V3
    )
    coverage: tuple[PopulationCoverageRowV3, ...]
    all_bands_per_clip_required: Literal[False] = False
    density_threshold_lowered: Literal[False] = False
    untouched_level5_challenge_separate: Literal[True] = True
    physical_performance_pass: Literal[False] = False

    @model_validator(mode="after")
    def _validate_audit(self) -> "PopulationAuditV3":
        canonical = PopulationCoverageContractV3.canonical()
        if self.contract.model_dump(mode="json") != canonical.model_dump(mode="json"):
            raise ValueError("audit가 exact canonical population v3 계약이 아닙니다")
        if self.contract_sha256 != canonical.digest():
            raise ValueError("audit population contract payload/SHA가 다릅니다")
        _require_sha("manifest_sha256", self.manifest_sha256)
        ids = [item.item_id for item in self.qualified_items]
        if len(ids) != len(set(ids)):
            raise ValueError("audit qualified item_id가 중복됐습니다")
        candidate_assignment: dict[str, tuple[str, str, str]] = {}
        lineage_assignment: dict[str, tuple[str, str]] = {}
        for item in self.qualified_items:
            candidate_value = (
                item.split,
                item.source_family,
                item.lineage_component_id,
            )
            prior_candidate = candidate_assignment.get(item.candidate_id)
            if prior_candidate is not None and prior_candidate != candidate_value:
                raise ValueError("audit candidate가 split/family/lineage를 넘나듭니다")
            candidate_assignment[item.candidate_id] = candidate_value
            lineage_value = (item.split, item.source_family)
            prior_lineage = lineage_assignment.get(item.lineage_component_id)
            if prior_lineage is not None and prior_lineage != lineage_value:
                raise ValueError("audit lineage component가 split/family를 넘나듭니다")
            lineage_assignment[item.lineage_component_id] = lineage_value

        control = canonical.control_band_contract
        expected: dict[tuple[str, str, str, int], tuple[float, float]] = {}
        for split in REQUIRED_SPLITS:
            for family in REQUIRED_SOURCE_FAMILIES:
                for role, bands in (
                    (
                        "physical_identification",
                        control.physical_identification_subbands_hz,
                    ),
                    ("objective_octave", control.equal_weight_octave_objective_bands_hz),
                ):
                    for index, band in enumerate(bands):
                        expected[(split, family, role, index)] = tuple(
                            float(value) for value in band
                        )
        actual: dict[tuple[str, str, str, int], PopulationCoverageRowV3] = {}
        for row in self.coverage:
            key = (row.split, row.source_family, row.band_role, row.band_index)
            if key in actual:
                raise ValueError("audit coverage key가 중복됐습니다")
            actual[key] = row
        if set(actual) != set(expected):
            raise ValueError("audit coverage가 split×family×band 전체를 덮지 않습니다")
        if any(actual[key].band_hz != band for key, band in expected.items()):
            raise ValueError("audit coverage band가 inline v3와 다릅니다")
        for key, row in actual.items():
            split, family, role, band_index = key
            attribute = (
                "physical_valid_bands"
                if role == "physical_identification"
                else "objective_octave_valid_bands"
            )
            lineage_count = len(
                {
                    item.lineage_component_id
                    for item in self.qualified_items
                    if item.split == split
                    and item.source_family == family
                    and bool(getattr(item, attribute)[band_index])
                }
            )
            if row.independent_lineage_components != lineage_count:
                raise ValueError(
                    "audit coverage independent lineage count가 qualified item과 다릅니다"
                )
        expected_status = "PASS" if all(row.passed for row in self.coverage) else "BLOCKED"
        if self.structural_status != expected_status:
            raise ValueError("audit structural status와 coverage 결과가 모순됩니다")
        missing_blockers = set(POPULATION_V3_SCAFFOLD_BLOCKERS) - set(self.blockers)
        if missing_blockers:
            raise ValueError(
                "audit가 external authority scaffold blocker를 누락했습니다: "
                f"{sorted(missing_blockers)!r}"
            )
        return self

    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class PopulationBatchPlanV3(BaseModel):
    """Schema/consistency scaffold; ``model_validate``는 admission parser가 아니다."""

    model_config = _FROZEN

    schema_version: Literal["broadband_population_batch_plan_v3"] = (
        "broadband_population_batch_plan_v3"
    )
    role: Literal["incomplete_structural_search_not_admission_parser"] = (
        "incomplete_structural_search_not_admission_parser"
    )
    contract_sha256: str
    population_audit_sha256: str
    population_audit_rng_entropy_material_sha256: str
    split: Literal["train", "val"]
    batch_index: int = Field(ge=0)
    seed: int = Field(ge=0)
    batch_size: int = Field(gt=0, le=MAX_BATCH_SIZE_V3)
    selected_items: tuple[QualifiedPopulationItemV3, ...] = Field(
        max_length=MAX_BATCH_SIZE_V3
    )
    selected_item_ids: tuple[str, ...] = Field(max_length=MAX_BATCH_SIZE_V3)
    selected_candidate_ids: tuple[str, ...] = Field(max_length=MAX_BATCH_SIZE_V3)
    selected_lineage_component_ids: tuple[str, ...] = Field(
        max_length=MAX_BATCH_SIZE_V3
    )
    family_counts: tuple[tuple[str, int], ...]
    physical_valid_item_counts: tuple[int, ...]
    objective_octave_valid_item_counts: tuple[int, ...]
    physical_distinct_lineage_counts: tuple[int, ...]
    objective_octave_distinct_lineage_counts: tuple[int, ...]
    minimum_valid_items_per_band: Literal[4] = 4
    structural_status: Literal["PASS"] = "PASS"
    canonical_training_status: Literal["BLOCKED"] = "BLOCKED"
    external_manifest_authority_bound: Literal[False] = False
    connected_component_authority_bound: Literal[False] = False
    interval_alias_authority_bound: Literal[False] = False
    actual_raw_manifest_authority_bound: Literal[False] = False
    component_uniform_long_run_sampler_proven: Literal[False] = False
    feasibility_search_complete: Literal[False] = False
    feasibility_false_negative_possible: Literal[True] = True
    feasibility_search_attempt_limit: Literal[256] = STRUCTURAL_BATCH_SEARCH_ATTEMPTS_V3
    standalone_model_validate_is_admission_parser: Literal[False] = False
    authority: None = None

    @model_validator(mode="after")
    def _validate_plan(self) -> "PopulationBatchPlanV3":
        contract = PopulationCoverageContractV3.canonical()
        if self.contract_sha256 != contract.digest():
            raise ValueError("batch plan population contract SHA가 다릅니다")
        _require_sha("population_audit_sha256", self.population_audit_sha256)
        _require_sha(
            "population_audit_rng_entropy_material_sha256",
            self.population_audit_rng_entropy_material_sha256,
        )
        expected_entropy_sha = hashlib.sha256(
            bytes.fromhex(self.population_audit_sha256)
        ).hexdigest()
        if self.population_audit_rng_entropy_material_sha256 != expected_entropy_sha:
            raise ValueError("population audit RNG entropy material SHA가 다릅니다")
        if self.batch_size <= len(REQUIRED_SOURCE_FAMILIES):
            raise ValueError("v3 structural batch_size=4는 금지됩니다")
        if len(self.selected_items) != self.batch_size:
            raise ValueError("batch selected item 증거 수가 batch_size와 다릅니다")
        expected_item_ids = tuple(item.item_id for item in self.selected_items)
        expected_candidate_ids = tuple(item.candidate_id for item in self.selected_items)
        expected_lineage_ids = tuple(
            item.lineage_component_id for item in self.selected_items
        )
        if (
            self.selected_item_ids != expected_item_ids
            or self.selected_candidate_ids != expected_candidate_ids
            or self.selected_lineage_component_ids != expected_lineage_ids
        ):
            raise ValueError("batch item/candidate/lineage 벡터가 선택 증거와 다릅니다")
        if len(set(self.selected_item_ids)) != self.batch_size:
            raise ValueError("batch item 수/고유성이 batch_size와 다릅니다")
        if len(set(self.selected_lineage_component_ids)) != self.batch_size:
            raise ValueError("batch에서 같은 lineage component를 두 번 선택할 수 없습니다")
        if any(not item_id.strip() for item_id in self.selected_item_ids):
            raise ValueError("batch item_id가 비었습니다")
        if any(item.split != self.split for item in self.selected_items):
            raise ValueError("batch selected item의 split이 plan과 다릅니다")
        candidate_assignment: dict[str, tuple[str, str]] = {}
        lineage_assignment: dict[str, str] = {}
        for item in self.selected_items:
            candidate_value = (item.source_family, item.lineage_component_id)
            prior_candidate = candidate_assignment.get(item.candidate_id)
            if prior_candidate is not None and prior_candidate != candidate_value:
                raise ValueError("batch candidate가 family/lineage를 넘나듭니다")
            candidate_assignment[item.candidate_id] = candidate_value
            prior_family = lineage_assignment.get(item.lineage_component_id)
            if prior_family is not None and prior_family != item.source_family:
                raise ValueError("batch lineage component가 family를 넘나듭니다")
            lineage_assignment[item.lineage_component_id] = item.source_family
        if self.batch_size % len(REQUIRED_SOURCE_FAMILIES):
            raise ValueError("batch_size가 네 family로 나뉘지 않습니다")
        quota = self.batch_size // len(REQUIRED_SOURCE_FAMILIES)
        expected_families = tuple(
            (
                family,
                sum(item.source_family == family for item in self.selected_items),
            )
            for family in REQUIRED_SOURCE_FAMILIES
        )
        if self.family_counts != expected_families or any(
            count != quota for _, count in expected_families
        ):
            raise ValueError("batch family count가 정확히 균형이 아닙니다")
        physical_item_counts = tuple(
            sum(item.physical_valid_bands[index] for item in self.selected_items)
            for index in range(8)
        )
        octave_item_counts = tuple(
            sum(
                item.objective_octave_valid_bands[index]
                for item in self.selected_items
            )
            for index in range(7)
        )
        physical_lineage_counts = tuple(
            len(
                {
                    item.lineage_component_id
                    for item in self.selected_items
                    if item.physical_valid_bands[index]
                }
            )
            for index in range(8)
        )
        octave_lineage_counts = tuple(
            len(
                {
                    item.lineage_component_id
                    for item in self.selected_items
                    if item.objective_octave_valid_bands[index]
                }
            )
            for index in range(7)
        )
        if self.physical_valid_item_counts != physical_item_counts:
            raise ValueError("batch physical valid item count가 선택 증거와 다릅니다")
        if self.objective_octave_valid_item_counts != octave_item_counts:
            raise ValueError("batch objective valid item count가 선택 증거와 다릅니다")
        if self.physical_distinct_lineage_counts != physical_lineage_counts:
            raise ValueError("batch physical distinct lineage count가 선택 증거와 다릅니다")
        if self.objective_octave_distinct_lineage_counts != octave_lineage_counts:
            raise ValueError("batch objective distinct lineage count가 선택 증거와 다릅니다")
        if len(physical_item_counts) != 8 or min(physical_item_counts) < 4:
            raise ValueError("batch physical band valid item 수가 4 미만입니다")
        if len(octave_item_counts) != 7 or min(octave_item_counts) < 4:
            raise ValueError("batch objective octave valid item 수가 4 미만입니다")
        if min(physical_lineage_counts) < 4:
            raise ValueError("batch physical band distinct lineage가 4 미만입니다")
        if min(octave_lineage_counts) < 4:
            raise ValueError("batch objective octave distinct lineage가 4 미만입니다")
        return self

    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class CurrentPopulationV3Gate(BaseModel):
    model_config = _FROZEN

    contract: PopulationCoverageContractV3
    contract_sha256: str
    authority: None = None
    status: Literal["BLOCKED"] = "BLOCKED"
    blockers: tuple[str, ...]


def current_population_v3_gate() -> CurrentPopulationV3Gate:
    contract = PopulationCoverageContractV3.canonical()
    return CurrentPopulationV3Gate(
        contract=contract,
        contract_sha256=contract.digest(),
        blockers=(
            "canonical broadband-v3 population manifest actual bytes are absent",
            "canonical fullband physical causal P authority is absent",
            "untouched post-lock Level-5 challenge reservation has no live authority",
            "POPULATION_V3_AUTHORITY is None",
        ),
    )


def density_ratios_v3(
    values: np.ndarray,
    *,
    sample_rate: int,
    bands_hz: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """대역별 평균 PSD를 전체 계약 대역의 bin-weighted 평균 PSD로 정규화."""

    signal = np.asarray(values, dtype=np.float64)
    fs = int(sample_rate)
    if signal.ndim != 1 or signal.size < 2 or not np.all(np.isfinite(signal)):
        raise ValueError("density 입력은 finite 1-D signal이어야 합니다")
    if fs <= 0:
        raise ValueError("sample rate는 양수여야 합니다")
    power = np.abs(np.fft.rfft(signal, norm="ortho")) ** 2
    means: list[float] = []
    total_power = 0.0
    total_bins = 0
    for index, raw_band in enumerate(bands_hz):
        lower, upper = float(raw_band[0]), float(raw_band[1])
        lower_bin = max(0, int(math.ceil(lower * signal.size / fs)))
        upper_bin = min(power.size - 1, int(math.floor(upper * signal.size / fs)))
        if index != len(bands_hz) - 1:
            upper_bin = min(
                upper_bin, int(math.ceil(upper * signal.size / fs)) - 1
            )
        if lower_bin > upper_bin:
            raise ValueError(f"FFT에 band bin이 없습니다: {(lower, upper)}")
        selected = power[lower_bin : upper_bin + 1]
        means.append(float(np.mean(selected)))
        total_power += float(np.sum(selected))
        total_bins += int(selected.size)
    flat = total_power / total_bins
    if flat <= np.finfo(np.float64).tiny:
        return tuple(0.0 for _ in means)
    return tuple(value / flat for value in means)


def apply_causal_primary_v3(
    source: np.ndarray, *, fir: np.ndarray, delay_samples: int
) -> np.ndarray:
    """manifest publisher와 auditor가 공유할 exact P application primitive."""

    values = np.asarray(source, dtype=np.float32)
    taps = np.asarray(fir, dtype=np.float32)
    delay = int(delay_samples)
    if (
        values.ndim != 1
        or taps.ndim != 1
        or values.size < 1
        or taps.size < 1
        or delay < 0
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(taps))
    ):
        raise ValueError("causal P source/FIR/delay가 유효하지 않습니다")
    output = np.zeros(values.size, dtype=np.float32)
    if delay < values.size:
        convolved = fftconvolve(
            values.astype(np.float64), taps.astype(np.float64), mode="full"
        )
        count = values.size - delay
        output[delay:] = np.asarray(convolved[:count], dtype=np.float32)
    return output


def _resolve_local_file(root: Path, reference: LocalFileReferenceV3) -> Path:
    root = root.resolve(strict=True)
    path = Path(os.path.abspath(root / reference.path))
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("file reference가 repository root 밖입니다") from exc
    current = root
    for part in Path(reference.path).parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError("file reference에 symlink가 있습니다")
    if not path.is_file() or path.stat().st_size != reference.size_bytes:
        raise ValueError(f"local file/size가 다릅니다: {reference.path}")
    if _sha256_file(path) != reference.sha256:
        raise ValueError(f"local file SHA가 다릅니다: {reference.path}")
    return path


def _read_float32_raw(root: Path, reference: LocalFileReferenceV3) -> np.ndarray:
    path = _resolve_local_file(root, reference)
    if reference.size_bytes % 4:
        raise ValueError(f"float32 raw byte size가 4의 배수가 아닙니다: {reference.path}")
    return np.fromfile(path, dtype="<f4").astype(np.float32, copy=False)


def audit_population_manifest_v3(
    manifest: PopulationManifestV3, *, repository_root: str | Path
) -> PopulationAuditV3:
    """actual bytes와 P-applied ERR를 재계산하되 canonical authority는 열지 않는다."""

    root = Path(repository_root).resolve(strict=True)
    contract = PopulationCoverageContractV3.canonical()
    if manifest.contract_sha256 != contract.digest():
        raise ValueError("manifest population v3 계약 SHA가 current canonical과 다릅니다")
    fir = _read_float32_raw(root, manifest.causal_primary.fir_file)
    if fir.size < 1:
        raise ValueError("causal P FIR이 비었습니다")

    physical_bands = contract.control_band_contract.physical_identification_subbands_hz
    octave_bands = contract.control_band_contract.equal_weight_octave_objective_bands_hz
    component_assignment: dict[str, tuple[str, str]] = {}
    native_sha_component: dict[str, str] = {}
    decoded_sha_component: dict[str, str] = {}
    qualified: list[QualifiedPopulationItemV3] = []

    for candidate in manifest.candidates:
        assignment = (candidate.split, candidate.source_family)
        prior = component_assignment.get(candidate.lineage_component_id)
        if prior is not None and prior != assignment:
            raise ValueError("lineage component가 split/family를 넘나듭니다")
        component_assignment[candidate.lineage_component_id] = assignment
        for digest, seen, label in (
            (
                candidate.immutable_native_source.sha256,
                native_sha_component,
                "native source",
            ),
            (candidate.decoded_pcm_file.sha256, decoded_sha_component, "decoded PCM"),
        ):
            previous_component = seen.get(digest)
            if previous_component is not None and previous_component != candidate.lineage_component_id:
                raise ValueError(f"같은 {label} SHA가 여러 lineage component로 분할됐습니다")
            seen[digest] = candidate.lineage_component_id

        _resolve_local_file(root, candidate.immutable_native_source)
        decoded = _read_float32_raw(root, candidate.decoded_pcm_file)
        stored_err = _read_float32_raw(root, candidate.p_applied_err_file)
        if decoded.size != candidate.decoded_frames or stored_err.size != decoded.size:
            raise ValueError(f"{candidate.candidate_id}: decoded/P-applied frame 수가 다릅니다")
        if candidate.p_operator_receipt_sha256 != manifest.causal_primary.operator_receipt_sha256:
            raise ValueError(f"{candidate.candidate_id}: causal P receipt SHA가 다릅니다")
        recomputed_err = apply_causal_primary_v3(
            decoded,
            fir=fir,
            delay_samples=manifest.causal_primary.delay_samples,
        )
        if recomputed_err.tobytes(order="C") != stored_err.tobytes(order="C"):
            raise ValueError(f"{candidate.candidate_id}: persisted P-applied ERR가 재계산과 다릅니다")
        minimum_prefix = manifest.causal_primary.delay_samples + fir.size - 1
        if candidate.valid_prefix_samples < minimum_prefix:
            raise ValueError(f"{candidate.candidate_id}: causal P valid prefix가 너무 짧습니다")

        for item in candidate.items:
            if item.n_frames != manifest.segment_samples:
                raise ValueError(f"{item.item_id}: segment length가 manifest와 다릅니다")
            stop = item.start_frame + item.n_frames
            if item.start_frame < candidate.valid_prefix_samples or stop > stored_err.size:
                raise ValueError(f"{item.item_id}: segment가 causal valid range 밖입니다")
            segment = stored_err[item.start_frame:stop]
            physical_density = density_ratios_v3(
                segment,
                sample_rate=48_000,
                bands_hz=physical_bands,
            )
            octave_density = density_ratios_v3(
                segment,
                sample_rate=48_000,
                bands_hz=octave_bands,
            )
            physical_valid = tuple(value >= MIN_DENSITY_RATIO for value in physical_density)
            octave_valid = tuple(value >= MIN_DENSITY_RATIO for value in octave_density)
            if not np.allclose(
                item.physical_density_ratios,
                physical_density,
                rtol=1.0e-10,
                atol=1.0e-12,
            ) or tuple(item.physical_valid_bands) != physical_valid:
                raise ValueError(f"{item.item_id}: physical density/mask 재계산 불일치")
            if not np.allclose(
                item.objective_octave_density_ratios,
                octave_density,
                rtol=1.0e-10,
                atol=1.0e-12,
            ) or tuple(item.objective_octave_valid_bands) != octave_valid:
                raise ValueError(f"{item.item_id}: objective density/mask 재계산 불일치")
            if not any(physical_valid) or not any(octave_valid):
                raise ValueError(f"{item.item_id}: 어떤 v3 band에도 자격이 없습니다")
            for band, valid in zip(physical_bands, physical_valid, strict=True):
                if valid and candidate.native_nyquist_hz < band[1]:
                    raise ValueError(f"{item.item_id}: native Nyquist로 physical band를 덮지 못합니다")
            for band, valid in zip(octave_bands, octave_valid, strict=True):
                if valid and candidate.native_nyquist_hz < band[1]:
                    raise ValueError(f"{item.item_id}: native Nyquist로 objective octave를 덮지 못합니다")
            if len(qualified) >= MAX_QUALIFIED_ITEMS_V3:
                raise ValueError(
                    f"qualified item은 {MAX_QUALIFIED_ITEMS_V3}개를 넘을 수 없습니다"
                )
            qualified.append(
                QualifiedPopulationItemV3(
                    item_id=item.item_id,
                    candidate_id=candidate.candidate_id,
                    split=candidate.split,
                    source_family=candidate.source_family,
                    lineage_component_id=candidate.lineage_component_id,
                    start_frame=item.start_frame,
                    n_frames=item.n_frames,
                    physical_valid_bands=physical_valid,
                    objective_octave_valid_bands=octave_valid,
                )
            )

    coverage: list[PopulationCoverageRowV3] = []
    blockers: list[str] = []
    for split in REQUIRED_SPLITS:
        for family in REQUIRED_SOURCE_FAMILIES:
            selected = [
                item
                for item in qualified
                if item.split == split and item.source_family == family
            ]
            for role, bands, attribute in (
                ("physical_identification", physical_bands, "physical_valid_bands"),
                ("objective_octave", octave_bands, "objective_octave_valid_bands"),
            ):
                for index, band in enumerate(bands):
                    components = {
                        item.lineage_component_id
                        for item in selected
                        if bool(getattr(item, attribute)[index])
                    }
                    passed = len(components) >= MIN_COMPONENTS_PER_SPLIT_FAMILY_BAND
                    coverage.append(
                        PopulationCoverageRowV3(
                            split=split,
                            source_family=family,
                            band_role=role,
                            band_index=index,
                            band_hz=tuple(float(value) for value in band),
                            independent_lineage_components=len(components),
                            passed=passed,
                        )
                    )
                    if not passed:
                        blockers.append(
                            f"{split}/{family}/{role}#{index}: independent components "
                            f"{len(components)} < 4"
                        )

    structural_status = "PASS" if not blockers else "BLOCKED"
    canonical_blockers = list(blockers)
    canonical_blockers.extend(
        (
            *POPULATION_V3_SCAFFOLD_BLOCKERS,
            "local file recomputation scaffold cannot self-issue training authority",
        )
    )
    return PopulationAuditV3(
        contract=contract,
        contract_sha256=contract.digest(),
        manifest_sha256=manifest.digest(),
        structural_status=structural_status,
        blockers=tuple(canonical_blockers),
        candidates_verified=len(manifest.candidates),
        qualified_items=tuple(qualified),
        coverage=tuple(coverage),
    )


def plan_structural_batch_v3(
    audit: PopulationAuditV3,
    *,
    split: Literal["train", "val"],
    batch_size: int,
    batch_index: int,
    seed: int,
) -> PopulationBatchPlanV3:
    """family→lineage→item set-cover만 증명하며 training을 열지 않는다."""

    if audit.structural_status != "PASS":
        raise PopulationV3Blocked("population structural coverage가 BLOCKED입니다")
    size = int(batch_size)
    if size <= len(REQUIRED_SOURCE_FAMILIES):
        raise ValueError("v3 structural batch_size=4는 금지됩니다")
    if size > MAX_BATCH_SIZE_V3:
        raise ValueError(f"v3 batch_size는 {MAX_BATCH_SIZE_V3}를 넘을 수 없습니다")
    if size % len(REQUIRED_SOURCE_FAMILIES):
        raise ValueError("batch_size는 네 family로 정확히 나뉘는 양수여야 합니다")
    if int(batch_index) < 0:
        raise ValueError("batch_index는 0 이상이어야 합니다")
    if int(seed) < 0:
        raise ValueError("seed는 0 이상이어야 합니다")
    quota = size // len(REQUIRED_SOURCE_FAMILIES)
    pool = [item for item in audit.qualified_items if item.split == split]
    audit_bytes = bytes.fromhex(audit.digest())
    audit_entropy = tuple(
        int.from_bytes(audit_bytes[offset : offset + 4], "little")
        for offset in range(0, len(audit_bytes), 4)
    )
    selected: list[QualifiedPopulationItemV3] | None = None
    for attempt in range(STRUCTURAL_BATCH_SEARCH_ATTEMPTS_V3):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [int(seed), int(batch_index), int(attempt), 0x5033, *audit_entropy]
            )
        )
        remaining = list(pool)
        chosen: list[QualifiedPopulationItemV3] = []
        family_counts = {family: 0 for family in REQUIRED_SOURCE_FAMILIES}
        physical_counts = [0] * 8
        octave_counts = [0] * 7
        physical_lineages = [set() for _ in range(8)]
        octave_lineages = [set() for _ in range(7)]
        for _ in range(size):
            best_score = -1
            best_by_lineage: dict[
                tuple[str, str], list[QualifiedPopulationItemV3]
            ] = {}
            for item in remaining:
                if family_counts[item.source_family] >= quota:
                    continue
                item_gain = sum(
                    count < MIN_VALID_ITEMS_PER_BATCH_BAND and valid
                    for count, valid in zip(
                        physical_counts, item.physical_valid_bands, strict=True
                    )
                ) + sum(
                    count < MIN_VALID_ITEMS_PER_BATCH_BAND and valid
                    for count, valid in zip(
                        octave_counts, item.objective_octave_valid_bands, strict=True
                    )
                )
                lineage_gain = sum(
                    len(lineages) < MIN_VALID_ITEMS_PER_BATCH_BAND
                    and valid
                    and item.lineage_component_id not in lineages
                    for lineages, valid in zip(
                        physical_lineages, item.physical_valid_bands, strict=True
                    )
                ) + sum(
                    len(lineages) < MIN_VALID_ITEMS_PER_BATCH_BAND
                    and valid
                    and item.lineage_component_id not in lineages
                    for lineages, valid in zip(
                        octave_lineages,
                        item.objective_octave_valid_bands,
                        strict=True,
                    )
                )
                score = 2 * lineage_gain + item_gain
                lineage_key = (item.source_family, item.lineage_component_id)
                if score > best_score:
                    best_score = int(score)
                    best_by_lineage = {lineage_key: [item]}
                elif score == best_score:
                    best_by_lineage.setdefault(lineage_key, []).append(item)
            if not best_by_lineage:
                break
            lineage_keys = sorted(best_by_lineage)
            lineage_key = lineage_keys[int(rng.integers(len(lineage_keys)))]
            candidate_groups: dict[str, list[QualifiedPopulationItemV3]] = {}
            for candidate_item in best_by_lineage[lineage_key]:
                candidate_groups.setdefault(candidate_item.candidate_id, []).append(
                    candidate_item
                )
            candidate_ids = sorted(candidate_groups)
            candidate_id = candidate_ids[int(rng.integers(len(candidate_ids)))]
            candidate_items = sorted(
                candidate_groups[candidate_id], key=lambda value: value.item_id
            )
            item = candidate_items[int(rng.integers(len(candidate_items)))]
            chosen.append(item)
            remaining = [
                candidate
                for candidate in remaining
                if candidate.lineage_component_id != item.lineage_component_id
            ]
            family_counts[item.source_family] += 1
            for index, valid in enumerate(item.physical_valid_bands):
                physical_counts[index] += int(valid)
                if valid:
                    physical_lineages[index].add(item.lineage_component_id)
            for index, valid in enumerate(item.objective_octave_valid_bands):
                octave_counts[index] += int(valid)
                if valid:
                    octave_lineages[index].add(item.lineage_component_id)
        if (
            len(chosen) == size
            and set(family_counts.values()) == {quota}
            and all(count >= MIN_VALID_ITEMS_PER_BATCH_BAND for count in physical_counts)
            and all(count >= MIN_VALID_ITEMS_PER_BATCH_BAND for count in octave_counts)
            and all(
                len(lineages) >= MIN_VALID_ITEMS_PER_BATCH_BAND
                for lineages in physical_lineages
            )
            and all(
                len(lineages) >= MIN_VALID_ITEMS_PER_BATCH_BAND
                for lineages in octave_lineages
            )
        ):
            selected = chosen
            break
    if selected is None:
        raise PopulationV3Blocked(
            "family-balanced batch에서 physical/objective band별 valid item>=4 "
            "및 distinct lineage>=4를 구성할 수 없습니다. "
            f"{STRUCTURAL_BATCH_SEARCH_ATTEMPTS_V3}-attempt randomized greedy는 "
            "incomplete feasibility search이므로 "
            "실제 feasible population에서도 false negative가 가능합니다"
        )
    family_counts = tuple(
        (family, sum(item.source_family == family for item in selected))
        for family in REQUIRED_SOURCE_FAMILIES
    )
    physical_counts = tuple(
        sum(item.physical_valid_bands[index] for item in selected) for index in range(8)
    )
    octave_counts = tuple(
        sum(item.objective_octave_valid_bands[index] for item in selected)
        for index in range(7)
    )
    physical_lineage_counts = tuple(
        len(
            {
                item.lineage_component_id
                for item in selected
                if item.physical_valid_bands[index]
            }
        )
        for index in range(8)
    )
    octave_lineage_counts = tuple(
        len(
            {
                item.lineage_component_id
                for item in selected
                if item.objective_octave_valid_bands[index]
            }
        )
        for index in range(7)
    )
    return PopulationBatchPlanV3(
        contract_sha256=audit.contract_sha256,
        population_audit_sha256=audit.digest(),
        population_audit_rng_entropy_material_sha256=hashlib.sha256(
            bytes.fromhex(audit.digest())
        ).hexdigest(),
        split=split,
        batch_index=int(batch_index),
        seed=int(seed),
        batch_size=size,
        selected_items=tuple(selected),
        selected_item_ids=tuple(item.item_id for item in selected),
        selected_candidate_ids=tuple(item.candidate_id for item in selected),
        selected_lineage_component_ids=tuple(
            item.lineage_component_id for item in selected
        ),
        family_counts=family_counts,
        physical_valid_item_counts=physical_counts,
        objective_octave_valid_item_counts=octave_counts,
        physical_distinct_lineage_counts=physical_lineage_counts,
        objective_octave_distinct_lineage_counts=octave_lineage_counts,
    )


__all__ = [
    "CausalPrimaryOperatorV3",
    "CurrentPopulationV3Gate",
    "LocalFileReferenceV3",
    "MAX_BATCH_SIZE_V3",
    "MAX_QUALIFIED_ITEMS_V3",
    "MIN_COMPONENTS_PER_SPLIT_FAMILY_BAND",
    "MIN_DENSITY_RATIO",
    "MIN_VALID_ITEMS_PER_BATCH_BAND",
    "POPULATION_V3_AUTHORITY",
    "POPULATION_V3_SCAFFOLD_BLOCKERS",
    "PopulationAuditV3",
    "PopulationBatchPlanV3",
    "PopulationCandidateV3",
    "PopulationCoverageContractV3",
    "PopulationItemClaimV3",
    "PopulationManifestV3",
    "PopulationV3Blocked",
    "STRUCTURAL_BATCH_SEARCH_ATTEMPTS_V3",
    "UntouchedLevel5PolicyV3",
    "apply_causal_primary_v3",
    "audit_population_manifest_v3",
    "current_population_v3_gate",
    "density_ratios_v3",
    "plan_structural_batch_v3",
]
