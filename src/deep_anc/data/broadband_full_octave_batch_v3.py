"""Broadband v3 전용 결정적 batch primitive.

기존 :mod:`deep_anc.data.broadband_batch_sampler`의 v2 receipt/sampler를 확장하거나
자동 승격하지 않는다. 이 모듈은 별도 ``PopulationAuditV3``의 **실제 payload SHA**를
입력으로 받아 8개 physical-identification band와 7개 objective octave를 다시
결속하고, global sample index의 순수 함수로 family-balanced batch를 만든다.

현재 population audit와 batch plan은 구조적 가능성만 증명한다. live v5 causal P/S
training envelope가 없으므로 모든 출력은 canonical training ``BLOCKED``다.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..dsp.control_band_contract import (
    BROADBAND_V3_OCTAVE_OBJECTIVE_BANDS_HZ,
    BROADBAND_V3_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ,
    BroadbandFullOctaveContractV3,
    resolve_control_band_contract,
)
from .broadband_population_contract_v3 import (
    MAX_BATCH_SIZE_V3,
    MIN_VALID_ITEMS_PER_BATCH_BAND,
    PopulationAuditV3,
    PopulationBatchPlanV3,
    PopulationV3Blocked,
    plan_structural_batch_v3,
)


FULL_OCTAVE_BATCH_PRIMITIVE_SCHEMA = "broadband_full_octave_batch_primitive_v3"
FULL_OCTAVE_GLOBAL_ITEM_SCHEMA = "broadband_full_octave_global_item_v3"
FULL_OCTAVE_BATCH_BLOCKER = "BLOCKED_INCOMPLETE_BROADBAND_V3_ADMISSION"
FULL_OCTAVE_BATCH_ADMISSION_BLOCKERS = (
    "POPULATION_V3_AUTHORITY is None",
    "MISSING_LIVE_V5_CAUSAL_AUTHORITY_ENVELOPE",
    "MISSING_OUTPUT_Y_GRADIENT_SHARE_0P2_0P4_CALIBRATION",
    "MISSING_ACTUAL_FAMILY_BALANCED_BATCH_RECEIPT_BINDING",
    "MISSING_CAUSAL_PREFIX_OPERATOR_TIMING_BINDING",
    "EXTERNAL_MANIFEST_AUTHORITY_NOT_BOUND",
    "CONNECTED_COMPONENT_AUTHORITY_NOT_BOUND",
    "INTERVAL_ALIAS_AUTHORITY_NOT_BOUND",
    "LOCAL_FILE_RECOMPUTATION_IS_NOT_EXTERNAL_RAW_AUTHORITY",
)
REQUIRED_FAMILIES = ("speech", "music", "environment", "machine")

_FROZEN = ConfigDict(frozen=True, extra="forbid")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label}는 lowercase SHA-256이어야 합니다")
    return text


class FullOctaveBatchPlanV3(BaseModel):
    """한 global batch의 scaffold. ``model_validate``는 admission parser가 아니다."""

    model_config = _FROZEN

    schema_version: Literal["broadband_full_octave_batch_primitive_v3"] = (
        FULL_OCTAVE_BATCH_PRIMITIVE_SCHEMA
    )
    role: Literal["global_index_scaffold_requires_authority_bound_parser"] = (
        "global_index_scaffold_requires_authority_bound_parser"
    )
    control_band_contract: BroadbandFullOctaveContractV3
    control_band_contract_sha256: str
    physical_identification_population_receipt_sha256: str
    population_audit_rng_entropy_material_sha256: str
    structural_population_plan: PopulationBatchPlanV3
    structural_population_plan_sha256: str
    seed: int
    split: Literal["train", "val"]
    global_batch_index: int
    first_global_sample_index: int
    batch_size: int = Field(le=MAX_BATCH_SIZE_V3)
    selected_item_ids: tuple[str, ...] = Field(max_length=MAX_BATCH_SIZE_V3)
    selected_candidate_ids: tuple[str, ...] = Field(max_length=MAX_BATCH_SIZE_V3)
    selected_lineage_component_ids: tuple[str, ...] = Field(
        max_length=MAX_BATCH_SIZE_V3
    )
    family_counts: tuple[tuple[str, int], ...]
    physical_identification_valid_item_counts: tuple[int, ...]
    objective_octave_valid_item_counts: tuple[int, ...]
    physical_identification_distinct_lineage_counts: tuple[int, ...]
    objective_octave_distinct_lineage_counts: tuple[int, ...]
    minimum_valid_items_per_band: Literal[4] = 4
    legacy_v2_automatic_promotion_allowed: Literal[False] = False
    canonical_training_eligible: Literal[False] = False
    external_manifest_authority_bound: Literal[False] = False
    connected_component_authority_bound: Literal[False] = False
    interval_alias_authority_bound: Literal[False] = False
    actual_raw_manifest_authority_bound: Literal[False] = False
    component_uniform_long_run_sampler_proven: Literal[False] = False
    feasibility_search_complete: Literal[False] = False
    feasibility_false_negative_possible: Literal[True] = True
    feasibility_search_attempt_limit: Literal[256] = 256
    standalone_model_validate_is_admission_parser: Literal[False] = False
    training_admission_status: Literal[
        "BLOCKED_INCOMPLETE_BROADBAND_V3_ADMISSION"
    ] = FULL_OCTAVE_BATCH_BLOCKER
    training_admission_blockers: tuple[str, ...] = FULL_OCTAVE_BATCH_ADMISSION_BLOCKERS
    authority: None = None

    @model_validator(mode="after")
    def _validate_plan(self) -> "FullOctaveBatchPlanV3":
        canonical = BroadbandFullOctaveContractV3.canonical()
        if self.control_band_contract.model_dump(mode="json") != canonical.model_dump(
            mode="json"
        ):
            raise ValueError("batch plan의 inline v3 contract가 canonical과 다릅니다")
        if self.control_band_contract_sha256 != self.control_band_contract.digest():
            raise ValueError("batch plan의 inline v3 payload/SHA가 다릅니다")
        _require_sha256(
            self.physical_identification_population_receipt_sha256,
            label="physical population receipt SHA",
        )
        _require_sha256(
            self.population_audit_rng_entropy_material_sha256,
            label="population audit RNG entropy material SHA",
        )
        _require_sha256(
            self.structural_population_plan_sha256,
            label="structural population plan SHA",
        )
        if self.structural_population_plan.digest() != self.structural_population_plan_sha256:
            raise ValueError("inline structural population plan과 SHA가 다릅니다")
        if (
            self.structural_population_plan.population_audit_sha256
            != self.physical_identification_population_receipt_sha256
            or self.structural_population_plan.population_audit_rng_entropy_material_sha256
            != self.population_audit_rng_entropy_material_sha256
        ):
            raise ValueError("structural plan이 다른 population receipt/entropy에 결속됐습니다")
        if self.global_batch_index < 0 or self.seed < 0:
            raise ValueError("batch index/seed는 0 이상이어야 합니다")
        if self.batch_size <= 4 or self.batch_size % len(REQUIRED_FAMILIES):
            raise ValueError(
                "v3 batch는 4보다 크고 네 family로 정확히 나뉘어야 합니다"
            )
        if self.first_global_sample_index != self.global_batch_index * self.batch_size:
            raise ValueError("first global sample index가 batch index와 다릅니다")
        if len(self.selected_item_ids) != self.batch_size:
            raise ValueError("batch item 수가 batch_size와 다릅니다")
        if len(set(self.selected_item_ids)) != len(self.selected_item_ids):
            raise ValueError("v3 batch selected_item_ids가 중복됐습니다")
        if len(set(self.selected_lineage_component_ids)) != self.batch_size:
            raise ValueError("v3 batch에서 같은 lineage component를 두 번 선택할 수 없습니다")
        structural = self.structural_population_plan
        if (
            structural.split != self.split
            or structural.batch_index != self.global_batch_index
            or structural.seed != self.seed
            or structural.batch_size != self.batch_size
            or structural.selected_item_ids != self.selected_item_ids
            or structural.selected_candidate_ids != self.selected_candidate_ids
            or structural.selected_lineage_component_ids
            != self.selected_lineage_component_ids
            or structural.feasibility_search_complete
            != self.feasibility_search_complete
            or structural.feasibility_false_negative_possible
            != self.feasibility_false_negative_possible
            or structural.feasibility_search_attempt_limit
            != self.feasibility_search_attempt_limit
        ):
            raise ValueError("outer batch plan이 inline structural selection과 다릅니다")
        expected_quota = self.batch_size // len(REQUIRED_FAMILIES)
        if self.family_counts != tuple(
            (family, expected_quota) for family in REQUIRED_FAMILIES
        ):
            raise ValueError("v3 batch family가 정확히 균형이 아닙니다")
        if len(self.physical_identification_valid_item_counts) != 8:
            raise ValueError("physical-identification count는 정확히 8개여야 합니다")
        if len(self.objective_octave_valid_item_counts) != 7:
            raise ValueError("objective-octave count는 정확히 7개여야 합니다")
        if (
            self.physical_identification_valid_item_counts
            != structural.physical_valid_item_counts
            or self.objective_octave_valid_item_counts
            != structural.objective_octave_valid_item_counts
            or self.physical_identification_distinct_lineage_counts
            != structural.physical_distinct_lineage_counts
            or self.objective_octave_distinct_lineage_counts
            != structural.objective_octave_distinct_lineage_counts
        ):
            raise ValueError("outer batch count가 inline structural selection과 다릅니다")
        if any(
            int(count) < MIN_VALID_ITEMS_PER_BATCH_BAND
            for count in (
                *self.physical_identification_valid_item_counts,
                *self.objective_octave_valid_item_counts,
            )
        ):
            raise ValueError("v3 batch의 band별 valid item이 4개 미만입니다")
        if len(self.physical_identification_distinct_lineage_counts) != 8 or any(
            int(count) < MIN_VALID_ITEMS_PER_BATCH_BAND
            for count in self.physical_identification_distinct_lineage_counts
        ):
            raise ValueError("v3 batch의 physical band별 distinct lineage가 4개 미만입니다")
        if len(self.objective_octave_distinct_lineage_counts) != 7 or any(
            int(count) < MIN_VALID_ITEMS_PER_BATCH_BAND
            for count in self.objective_octave_distinct_lineage_counts
        ):
            raise ValueError("v3 batch의 objective band별 distinct lineage가 4개 미만입니다")
        if tuple(self.training_admission_blockers) != (
            FULL_OCTAVE_BATCH_ADMISSION_BLOCKERS
        ):
            raise ValueError("v3 batch admission blocker 집합을 변경할 수 없습니다")
        return self

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.model_dump(mode="json"))).hexdigest()


class FullOctaveGlobalItemV3(BaseModel):
    """중단/재개에 안정적인 global sample index 한 항목의 선택 증거."""

    model_config = _FROZEN

    schema_version: Literal["broadband_full_octave_global_item_v3"] = (
        FULL_OCTAVE_GLOBAL_ITEM_SCHEMA
    )
    global_sample_index: int
    global_batch_index: int
    batch_offset: int
    batch_size: int = Field(le=MAX_BATCH_SIZE_V3)
    selected_item_id: str
    selected_candidate_id: str
    selected_lineage_component_id: str
    batch_plan_sha256: str
    physical_identification_population_receipt_sha256: str
    external_manifest_authority_bound: Literal[False] = False
    connected_component_authority_bound: Literal[False] = False
    interval_alias_authority_bound: Literal[False] = False
    actual_raw_manifest_authority_bound: Literal[False] = False
    standalone_model_validate_is_admission_parser: Literal[False] = False
    canonical_training_eligible: Literal[False] = False
    training_admission_status: Literal[
        "BLOCKED_INCOMPLETE_BROADBAND_V3_ADMISSION"
    ] = FULL_OCTAVE_BATCH_BLOCKER
    training_admission_blockers: tuple[str, ...] = FULL_OCTAVE_BATCH_ADMISSION_BLOCKERS

    @model_validator(mode="after")
    def _validate_item(self) -> "FullOctaveGlobalItemV3":
        if self.global_sample_index < 0 or self.global_batch_index < 0:
            raise ValueError("global sample/batch index는 0 이상이어야 합니다")
        if self.batch_size <= 4 or self.batch_size % len(REQUIRED_FAMILIES):
            raise ValueError("global item의 v3 batch size가 유효하지 않습니다")
        expected_batch, expected_offset = divmod(
            self.global_sample_index, self.batch_size
        )
        if (
            self.global_batch_index != expected_batch
            or self.batch_offset != expected_offset
        ):
            raise ValueError("global sample index의 batch/offset 분해가 다릅니다")
        if self.batch_offset < 0 or not all(
            (
                self.selected_item_id,
                self.selected_candidate_id,
                self.selected_lineage_component_id,
            )
        ):
            raise ValueError("batch offset/item/candidate/lineage id가 유효하지 않습니다")
        _require_sha256(self.batch_plan_sha256, label="batch plan SHA")
        _require_sha256(
            self.physical_identification_population_receipt_sha256,
            label="physical population receipt SHA",
        )
        if tuple(self.training_admission_blockers) != (
            FULL_OCTAVE_BATCH_ADMISSION_BLOCKERS
        ):
            raise ValueError("v3 global item admission blocker 집합을 변경할 수 없습니다")
        return self


class BroadbandFullOctaveBatchPrimitiveV3:
    """exact v3 population receipt에서 global-index batch를 계획한다."""

    def __init__(
        self,
        *,
        control_band_contract: Mapping[str, Any] | BroadbandFullOctaveContractV3,
        control_band_contract_sha256: str,
        physical_identification_population_receipt: Mapping[str, Any]
        | PopulationAuditV3,
        physical_identification_population_receipt_sha256: str,
        split: Literal["train", "val"],
        batch_size: int,
        seed: int,
    ) -> None:
        resolved = resolve_control_band_contract(control_band_contract)
        if type(resolved) is not BroadbandFullOctaveContractV3:
            raise ValueError("v2 control-band contract를 v3 batch로 승격할 수 없습니다")
        canonical = BroadbandFullOctaveContractV3.canonical()
        if resolved.model_dump(mode="json") != canonical.model_dump(mode="json"):
            raise ValueError("v3 batch에는 exact canonical inline contract가 필요합니다")
        supplied_contract_sha = _require_sha256(
            control_band_contract_sha256,
            label="control-band contract SHA",
        )
        if supplied_contract_sha != resolved.digest() or supplied_contract_sha != canonical.digest():
            raise ValueError("v3 batch inline contract/SHA가 다릅니다")

        audit = (
            physical_identification_population_receipt
            if isinstance(physical_identification_population_receipt, PopulationAuditV3)
            else PopulationAuditV3.model_validate(
                dict(physical_identification_population_receipt)
            )
        )
        supplied_population_sha = _require_sha256(
            physical_identification_population_receipt_sha256,
            label="physical population receipt SHA",
        )
        if audit.digest() != supplied_population_sha:
            raise ValueError("physical-identification population receipt payload/SHA가 다릅니다")
        if audit.structural_status != "PASS":
            raise PopulationV3Blocked("physical population structural audit가 BLOCKED입니다")
        if (
            audit.contract.control_band_contract.model_dump(mode="json")
            != resolved.model_dump(mode="json")
            or audit.contract.control_band_contract_sha256 != supplied_contract_sha
            or audit.contract_sha256 != audit.contract.digest()
        ):
            raise ValueError("physical population receipt가 같은 inline v3 계약에 결속되지 않았습니다")
        self._validate_population_coverage(audit)

        size = int(batch_size)
        if size <= 4:
            raise ValueError(
                "batch_size=4는 band별 valid item>=4와 family balance를 동시에 "
                "만족하려면 네 item 모두가 모든 band를 통과해야 하므로 v3에서 금지합니다"
            )
        if size > MAX_BATCH_SIZE_V3:
            raise ValueError(f"v3 batch_size는 {MAX_BATCH_SIZE_V3}를 넘을 수 없습니다")
        if size % len(REQUIRED_FAMILIES):
            raise ValueError("v3 batch_size는 네 family로 정확히 나뉘어야 합니다")
        if int(seed) < 0:
            raise ValueError("v3 batch seed는 0 이상이어야 합니다")
        if split not in {"train", "val"}:
            raise ValueError("v3 batch split은 train/val만 허용합니다")

        self.contract = resolved
        self.contract_sha256 = supplied_contract_sha
        self.population_audit = audit
        self.population_receipt_sha256 = supplied_population_sha
        self.split = split
        self.batch_size = size
        self.seed = int(seed)

    def _validate_population_coverage(self, audit: PopulationAuditV3) -> None:
        """receipt status 문자열만 믿지 않고 exact 8+7 coverage rows를 확인한다."""

        expected: dict[tuple[str, str, str, int], tuple[float, float]] = {}
        for split in ("train", "val", "test"):
            for family in REQUIRED_FAMILIES:
                for index, band in enumerate(
                    BROADBAND_V3_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ
                ):
                    expected[(split, family, "physical_identification", index)] = band
                for index, band in enumerate(BROADBAND_V3_OCTAVE_OBJECTIVE_BANDS_HZ):
                    expected[(split, family, "objective_octave", index)] = band
        observed: dict[tuple[str, str, str, int], tuple[float, float]] = {}
        for row in audit.coverage:
            key = (row.split, row.source_family, row.band_role, int(row.band_index))
            if key in observed:
                raise ValueError(f"population coverage row가 중복됐습니다: {key!r}")
            band = tuple(float(value) for value in row.band_hz)
            if key not in expected or band != expected[key]:
                raise ValueError(f"population coverage row의 role/band가 v3와 다릅니다: {key!r}")
            if (
                not row.passed
                or int(row.independent_lineage_components) < 4
                or int(row.minimum_required) != 4
            ):
                raise PopulationV3Blocked(f"population coverage가 4-component gate 실패: {key!r}")
            observed[key] = band
        missing = sorted(set(expected) - set(observed))
        if missing:
            raise ValueError(
                "population receipt에 physical/objective coverage row가 누락됐습니다: "
                f"{missing[:3]!r} (total={len(missing)})"
            )

    def plan_for_batch_index(self, global_batch_index: int) -> FullOctaveBatchPlanV3:
        index = int(global_batch_index)
        if index < 0:
            raise ValueError("global batch index는 0 이상이어야 합니다")
        structural = plan_structural_batch_v3(
            self.population_audit,
            split=self.split,
            batch_size=self.batch_size,
            batch_index=index,
            seed=self.seed,
        )
        if (
            structural.contract_sha256 != self.population_audit.contract_sha256
            or structural.population_audit_sha256 != self.population_receipt_sha256
        ):
            raise RuntimeError("structural planner가 다른 contract/population SHA를 반환했습니다")
        return FullOctaveBatchPlanV3(
            control_band_contract=self.contract,
            control_band_contract_sha256=self.contract_sha256,
            physical_identification_population_receipt_sha256=(
                self.population_receipt_sha256
            ),
            population_audit_rng_entropy_material_sha256=(
                structural.population_audit_rng_entropy_material_sha256
            ),
            structural_population_plan=structural,
            structural_population_plan_sha256=structural.digest(),
            seed=self.seed,
            split=self.split,
            global_batch_index=index,
            first_global_sample_index=index * self.batch_size,
            batch_size=self.batch_size,
            selected_item_ids=structural.selected_item_ids,
            selected_candidate_ids=structural.selected_candidate_ids,
            selected_lineage_component_ids=(
                structural.selected_lineage_component_ids
            ),
            family_counts=structural.family_counts,
            physical_identification_valid_item_counts=(
                structural.physical_valid_item_counts
            ),
            objective_octave_valid_item_counts=(
                structural.objective_octave_valid_item_counts
            ),
            physical_identification_distinct_lineage_counts=(
                structural.physical_distinct_lineage_counts
            ),
            objective_octave_distinct_lineage_counts=(
                structural.objective_octave_distinct_lineage_counts
            ),
        )

    def validate_serialized_plan(
        self, payload: Mapping[str, Any] | FullOctaveBatchPlanV3
    ) -> FullOctaveBatchPlanV3:
        """유일한 authority-bound parser: receipt/config으로 plan을 재계획한다."""

        observed = (
            payload
            if isinstance(payload, FullOctaveBatchPlanV3)
            else FullOctaveBatchPlanV3.model_validate(dict(payload))
        )
        if (
            observed.physical_identification_population_receipt_sha256
            != self.population_receipt_sha256
            or observed.control_band_contract_sha256 != self.contract_sha256
            or observed.split != self.split
            or observed.batch_size != self.batch_size
            or observed.seed != self.seed
        ):
            raise ValueError("serialized plan이 현재 primitive authority/config와 다릅니다")
        expected = self.plan_for_batch_index(observed.global_batch_index)
        if observed != expected:
            raise ValueError("serialized plan이 population-bound 결정적 재계획과 다릅니다")
        return observed

    def item_for_global_sample_index(
        self, global_sample_index: int
    ) -> FullOctaveGlobalItemV3:
        index = int(global_sample_index)
        if index < 0:
            raise ValueError("global sample index는 0 이상이어야 합니다")
        batch_index, offset = divmod(index, self.batch_size)
        plan = self.plan_for_batch_index(batch_index)
        return FullOctaveGlobalItemV3(
            global_sample_index=index,
            global_batch_index=batch_index,
            batch_offset=offset,
            batch_size=self.batch_size,
            selected_item_id=plan.selected_item_ids[offset],
            selected_candidate_id=plan.selected_candidate_ids[offset],
            selected_lineage_component_id=(
                plan.selected_lineage_component_ids[offset]
            ),
            batch_plan_sha256=plan.digest(),
            physical_identification_population_receipt_sha256=(
                self.population_receipt_sha256
            ),
        )


__all__ = [
    "BROADBAND_V3_OCTAVE_OBJECTIVE_BANDS_HZ",
    "BROADBAND_V3_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ",
    "BroadbandFullOctaveBatchPrimitiveV3",
    "FULL_OCTAVE_BATCH_ADMISSION_BLOCKERS",
    "FULL_OCTAVE_BATCH_BLOCKER",
    "FULL_OCTAVE_BATCH_PRIMITIVE_SCHEMA",
    "FULL_OCTAVE_GLOBAL_ITEM_SCHEMA",
    "FullOctaveBatchPlanV3",
    "FullOctaveGlobalItemV3",
]
