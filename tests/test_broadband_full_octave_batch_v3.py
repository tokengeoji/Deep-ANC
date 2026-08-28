from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from deep_anc.data.broadband_full_octave_batch_v3 import (
    FULL_OCTAVE_BATCH_ADMISSION_BLOCKERS,
    FULL_OCTAVE_BATCH_BLOCKER,
    BroadbandFullOctaveBatchPrimitiveV3,
    FullOctaveBatchPlanV3,
)
from deep_anc.data.broadband_population_contract_v3 import (
    MAX_BATCH_SIZE_V3,
    POPULATION_V3_SCAFFOLD_BLOCKERS,
    PopulationAuditV3,
    PopulationBatchPlanV3,
    PopulationCoverageContractV3,
    PopulationCoverageRowV3,
    PopulationV3Blocked,
    QualifiedPopulationItemV3,
)
from deep_anc.dsp.control_band_contract import (
    BroadbandFullOctaveContractV3,
    ControlBandContract,
)


_FAMILIES = ("speech", "music", "environment", "machine")


def _population_audit(*, segments_per_lineage: int = 1) -> PopulationAuditV3:
    population_contract = PopulationCoverageContractV3.canonical()
    control = population_contract.control_band_contract
    items: list[QualifiedPopulationItemV3] = []
    # family마다 8개 independent component. 각 item은 parity가 같은 일부 band만
    # 자격을 갖고, 모든 8/7 band를 동시에 통과하는 item은 하나도 없다.
    for split in ("train", "val", "test"):
        for family in _FAMILIES:
            for lineage_index in range(8):
                physical = tuple(
                    lineage_index % 2 == band_index % 2 for band_index in range(8)
                )
                objective = tuple(
                    lineage_index % 2 == band_index % 2 for band_index in range(7)
                )
                for segment_index in range(segments_per_lineage):
                    items.append(
                        QualifiedPopulationItemV3(
                            item_id=(
                                f"{split}-{family}-{lineage_index}-{segment_index}"
                            ),
                            candidate_id=(
                                f"candidate-{split}-{family}-{lineage_index}"
                            ),
                            split=split,
                            source_family=family,
                            lineage_component_id=(
                                f"component-{split}-{family}-{lineage_index}"
                            ),
                            start_frame=4096
                            * (lineage_index * segments_per_lineage + segment_index),
                            n_frames=4096,
                            physical_valid_bands=physical,
                            objective_octave_valid_bands=objective,
                        )
                    )

    coverage: list[PopulationCoverageRowV3] = []
    for split in ("train", "val", "test"):
        for family in _FAMILIES:
            for index, band in enumerate(control.physical_identification_subbands_hz):
                coverage.append(
                    PopulationCoverageRowV3(
                        split=split,
                        source_family=family,
                        band_role="physical_identification",
                        band_index=index,
                        band_hz=band,
                        independent_lineage_components=4,
                        passed=True,
                    )
                )
            for index, band in enumerate(
                control.equal_weight_octave_objective_bands_hz
            ):
                coverage.append(
                    PopulationCoverageRowV3(
                        split=split,
                        source_family=family,
                        band_role="objective_octave",
                        band_index=index,
                        band_hz=band,
                        independent_lineage_components=4,
                        passed=True,
                    )
                )
    return PopulationAuditV3(
        contract=population_contract,
        contract_sha256=population_contract.digest(),
        manifest_sha256="a" * 64,
        structural_status="PASS",
        blockers=(
            *POPULATION_V3_SCAFFOLD_BLOCKERS,
            "canonical training remains BLOCKED",
        ),
        candidates_verified=len(items),
        qualified_items=tuple(items),
        coverage=tuple(coverage),
    )


def _primitive(
    *,
    batch_size: int = 16,
    seed: int = 20260803,
    audit: PopulationAuditV3 | None = None,
):
    control = BroadbandFullOctaveContractV3.canonical()
    resolved_audit = _population_audit() if audit is None else audit
    return BroadbandFullOctaveBatchPrimitiveV3(
        control_band_contract=control.model_dump(mode="python"),
        control_band_contract_sha256=control.digest(),
        physical_identification_population_receipt=resolved_audit,
        physical_identification_population_receipt_sha256=resolved_audit.digest(),
        split="train",
        batch_size=batch_size,
        seed=seed,
    )


def test_partial_band_items_form_family_balanced_global_index_batch() -> None:
    primitive = _primitive()
    plan = primitive.plan_for_batch_index(37)
    assert plan.first_global_sample_index == 37 * 16
    assert plan.split == "train"
    assert plan.family_counts == tuple((family, 4) for family in _FAMILIES)
    assert min(plan.physical_identification_valid_item_counts) >= 4
    assert min(plan.objective_octave_valid_item_counts) >= 4
    assert min(plan.physical_identification_distinct_lineage_counts) >= 4
    assert min(plan.objective_octave_distinct_lineage_counts) >= 4
    assert len(set(plan.selected_lineage_component_ids)) == plan.batch_size
    assert plan.physical_identification_population_receipt_sha256 == (
        primitive.population_audit.digest()
    )
    assert plan.canonical_training_eligible is False
    assert plan.external_manifest_authority_bound is False
    assert plan.connected_component_authority_bound is False
    assert plan.interval_alias_authority_bound is False
    assert plan.actual_raw_manifest_authority_bound is False
    assert plan.component_uniform_long_run_sampler_proven is False
    assert plan.feasibility_search_complete is False
    assert plan.feasibility_false_negative_possible is True
    assert plan.feasibility_search_attempt_limit == 256
    assert plan.standalone_model_validate_is_admission_parser is False
    assert plan.structural_population_plan.feasibility_search_complete is False
    assert plan.structural_population_plan.feasibility_false_negative_possible is True
    assert plan.training_admission_status == FULL_OCTAVE_BATCH_BLOCKER
    assert plan.training_admission_blockers == FULL_OCTAVE_BATCH_ADMISSION_BLOCKERS

    selected = {
        item.item_id: item for item in primitive.population_audit.qualified_items
    }
    assert all(
        not all(selected[item_id].physical_valid_bands)
        and not all(selected[item_id].objective_octave_valid_bands)
        for item_id in plan.selected_item_ids
    )


def test_partial_band_set_cover_uses_distinct_lineages_despite_many_segments() -> None:
    audit = _population_audit(segments_per_lineage=3)
    plan = _primitive(audit=audit, seed=812).plan_for_batch_index(5)
    selected = {
        item.item_id: item for item in plan.structural_population_plan.selected_items
    }

    assert all(
        not all(selected[item_id].physical_valid_bands)
        and not all(selected[item_id].objective_octave_valid_bands)
        for item_id in plan.selected_item_ids
    )
    assert min(plan.physical_identification_valid_item_counts) >= 4
    assert min(plan.objective_octave_valid_item_counts) >= 4
    assert min(plan.physical_identification_distinct_lineage_counts) >= 4
    assert min(plan.objective_octave_distinct_lineage_counts) >= 4


def test_global_sample_index_is_resume_deterministic_and_receipt_sha_bound() -> None:
    first = _primitive(seed=91)
    second = _primitive(seed=91)
    index = 1_000_003
    selected_a = first.item_for_global_sample_index(index)
    selected_b = second.item_for_global_sample_index(index)
    assert selected_a == selected_b
    assert selected_a.global_batch_index == index // 16
    assert selected_a.batch_offset == index % 16
    assert selected_a.physical_identification_population_receipt_sha256 == (
        first.population_audit.digest()
    )

    control = BroadbandFullOctaveContractV3.canonical()
    audit = _population_audit()
    with pytest.raises(ValueError, match="payload/SHA"):
        BroadbandFullOctaveBatchPrimitiveV3(
            control_band_contract=control,
            control_band_contract_sha256=control.digest(),
            physical_identification_population_receipt=audit,
            physical_identification_population_receipt_sha256="b" * 64,
            split="train",
            batch_size=16,
            seed=91,
        )


def test_batch_four_is_rejected_even_if_four_items_could_claim_every_band() -> None:
    # valid item>=4를 batch=4로 허용하면 네 item 모두가 모든 octave를 통과해야만 한다.
    # 자연 source의 partial-band assignment를 다시 all-seven clip 요구로 바꾸므로 금지한다.
    with pytest.raises(ValueError, match="batch_size=4"):
        _primitive(batch_size=4)


def test_serialized_plan_rejects_forged_duplicate_items_and_split() -> None:
    plan = _primitive().plan_for_batch_index(3)
    payload = plan.model_dump(mode="python")
    forged_ids = list(payload["selected_item_ids"])
    forged_ids[1] = forged_ids[0]
    payload["selected_item_ids"] = forged_ids
    with pytest.raises(ValidationError, match="selected_item_ids가 중복"):
        FullOctaveBatchPlanV3.model_validate(payload)

    payload = plan.model_dump(mode="python")
    payload["split"] = "test"
    with pytest.raises(ValidationError):
        FullOctaveBatchPlanV3.model_validate(payload)


def test_serialized_plan_seals_candidate_lineage_vectors_and_counts() -> None:
    primitive = _primitive()
    plan = primitive.plan_for_batch_index(11)
    assert primitive.validate_serialized_plan(plan.model_dump(mode="json")) == plan

    forged_admission = plan.model_dump(mode="json")
    forged_admission["standalone_model_validate_is_admission_parser"] = True
    with pytest.raises(ValidationError):
        FullOctaveBatchPlanV3.model_validate(forged_admission)

    forged_authority = plan.model_dump(mode="json")
    forged_authority["external_manifest_authority_bound"] = True
    with pytest.raises(ValidationError):
        FullOctaveBatchPlanV3.model_validate(forged_authority)

    forged_candidate = plan.model_dump(mode="python")
    forged_candidate["selected_candidate_ids"] = list(
        forged_candidate["selected_candidate_ids"]
    )
    forged_candidate["selected_candidate_ids"][0] = "forged-candidate"
    with pytest.raises(ValidationError, match="structural selection"):
        FullOctaveBatchPlanV3.model_validate(forged_candidate)

    forged_lineage = plan.model_dump(mode="python")
    forged_lineage["structural_population_plan"][
        "selected_lineage_component_ids"
    ] = list(
        forged_lineage["structural_population_plan"][
            "selected_lineage_component_ids"
        ]
    )
    forged_lineage["structural_population_plan"][
        "selected_lineage_component_ids"
    ][0] = "forged-lineage"
    with pytest.raises(ValidationError, match="item/candidate/lineage"):
        FullOctaveBatchPlanV3.model_validate(forged_lineage)

    forged_count = plan.model_dump(mode="python")
    forged_count["physical_identification_distinct_lineage_counts"] = list(
        forged_count["physical_identification_distinct_lineage_counts"]
    )
    forged_count["physical_identification_distinct_lineage_counts"][0] += 1
    with pytest.raises(ValidationError, match="outer batch count"):
        FullOctaveBatchPlanV3.model_validate(forged_count)

    # 내부 벡터와 SHA를 모두 다시 맞춘 self-consistent 위조도
    # exact population receipt에 결속된 primitive의 결정적 재계획과 비교하면 차단된다.
    resealed = plan.model_dump(mode="json")
    structural_payload = resealed["structural_population_plan"]
    structural_payload["selected_items"][0]["candidate_id"] = "resealed-forgery"
    structural_payload["selected_candidate_ids"][0] = "resealed-forgery"
    structural = PopulationBatchPlanV3.model_validate(structural_payload)
    resealed["structural_population_plan"] = structural.model_dump(mode="json")
    resealed["structural_population_plan_sha256"] = structural.digest()
    resealed["selected_candidate_ids"][0] = "resealed-forgery"
    standalone = FullOctaveBatchPlanV3.model_validate(resealed)
    assert standalone != plan
    assert standalone.standalone_model_validate_is_admission_parser is False
    with pytest.raises(ValueError, match="결정적 재계획"):
        primitive.validate_serialized_plan(standalone)


def test_one_lineage_many_segments_cannot_forge_band_quota() -> None:
    audit = _population_audit()
    base = audit.qualified_items[0]
    selected: list[QualifiedPopulationItemV3] = []
    for family_index, family in enumerate(_FAMILIES):
        for item_index in range(4):
            band_zero = family_index == 0
            selected.append(
                QualifiedPopulationItemV3(
                    item_id=f"selected-{family}-{item_index}",
                    candidate_id=f"candidate-{family}-{item_index}",
                    split="train",
                    source_family=family,
                    lineage_component_id=(
                        "speech-one-lineage"
                        if band_zero
                        else f"lineage-{family}-{item_index}"
                    ),
                    start_frame=base.n_frames * item_index,
                    n_frames=base.n_frames,
                    physical_valid_bands=(band_zero, True, True, True, True, True, True, True),
                    objective_octave_valid_bands=(band_zero, True, True, True, True, True, True),
                )
            )
    item_counts_physical = tuple(
        sum(item.physical_valid_bands[index] for item in selected)
        for index in range(8)
    )
    item_counts_objective = tuple(
        sum(item.objective_octave_valid_bands[index] for item in selected)
        for index in range(7)
    )
    lineage_counts_physical = tuple(
        len(
            {
                item.lineage_component_id
                for item in selected
                if item.physical_valid_bands[index]
            }
        )
        for index in range(8)
    )
    lineage_counts_objective = tuple(
        len(
            {
                item.lineage_component_id
                for item in selected
                if item.objective_octave_valid_bands[index]
            }
        )
        for index in range(7)
    )

    assert item_counts_physical[0] == item_counts_objective[0] == 4
    assert lineage_counts_physical[0] == lineage_counts_objective[0] == 1
    with pytest.raises(ValidationError, match="lineage component"):
        PopulationBatchPlanV3(
            contract_sha256=audit.contract_sha256,
            population_audit_sha256=audit.digest(),
            population_audit_rng_entropy_material_sha256=hashlib.sha256(
                bytes.fromhex(audit.digest())
            ).hexdigest(),
            split="train",
            batch_index=0,
            seed=1,
            batch_size=16,
            selected_items=tuple(selected),
            selected_item_ids=tuple(item.item_id for item in selected),
            selected_candidate_ids=tuple(item.candidate_id for item in selected),
            selected_lineage_component_ids=tuple(
                item.lineage_component_id for item in selected
            ),
            family_counts=tuple((family, 4) for family in _FAMILIES),
            physical_valid_item_counts=item_counts_physical,
            objective_octave_valid_item_counts=item_counts_objective,
            physical_distinct_lineage_counts=lineage_counts_physical,
            objective_octave_distinct_lineage_counts=lineage_counts_objective,
        )


def test_population_audit_sha_is_rng_entropy_and_changes_plan_identity() -> None:
    first_audit = _population_audit()
    payload = first_audit.model_dump(mode="python")
    payload["manifest_sha256"] = "b" * 64
    second_audit = PopulationAuditV3.model_validate(payload)
    first = _primitive(seed=417, audit=first_audit).plan_for_batch_index(29)
    second = _primitive(seed=417, audit=second_audit).plan_for_batch_index(29)

    assert first.physical_identification_population_receipt_sha256 != (
        second.physical_identification_population_receipt_sha256
    )
    assert first.population_audit_rng_entropy_material_sha256 != (
        second.population_audit_rng_entropy_material_sha256
    )
    assert first.population_audit_rng_entropy_material_sha256 == hashlib.sha256(
        bytes.fromhex(first.physical_identification_population_receipt_sha256)
    ).hexdigest()
    assert first.digest() != second.digest()


def test_batch_size_has_fail_closed_upper_bound() -> None:
    with pytest.raises(ValueError, match=str(MAX_BATCH_SIZE_V3)):
        _primitive(batch_size=MAX_BATCH_SIZE_V3 + 4)


def test_incomplete_greedy_search_is_schema_sealed_and_error_is_explicit() -> None:
    primitive = _primitive(batch_size=36)
    with pytest.raises(
        PopulationV3Blocked, match="incomplete feasibility search.*false negative"
    ):
        primitive.plan_for_batch_index(0)


def test_v2_control_contract_cannot_open_v3_batch() -> None:
    v2 = ControlBandContract.broadband_point_control()
    audit = _population_audit()
    with pytest.raises(ValueError, match="승격"):
        BroadbandFullOctaveBatchPrimitiveV3(
            control_band_contract=v2.model_dump(mode="python"),
            control_band_contract_sha256=v2.digest(),
            physical_identification_population_receipt=audit,
            physical_identification_population_receipt_sha256=audit.digest(),
            split="train",
            batch_size=16,
            seed=1,
        )


def test_population_receipt_must_contain_all_exact_eight_plus_seven_rows() -> None:
    audit = _population_audit()
    payload = audit.model_dump(mode="python")
    payload["coverage"] = payload["coverage"][1:]
    # PopulationAuditV3 자체와 adapter가 같은 exact 8+7 row 집합을 요구한다.
    with pytest.raises(ValidationError, match="전체를 덮지 않습니다"):
        PopulationAuditV3.model_validate(payload)
