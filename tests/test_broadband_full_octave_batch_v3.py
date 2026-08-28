from __future__ import annotations

import pytest
from pydantic import ValidationError

from deep_anc.data.broadband_full_octave_batch_v3 import (
    FULL_OCTAVE_BATCH_ADMISSION_BLOCKERS,
    FULL_OCTAVE_BATCH_BLOCKER,
    BroadbandFullOctaveBatchPrimitiveV3,
    FullOctaveBatchPlanV3,
)
from deep_anc.data.broadband_population_contract_v3 import (
    PopulationAuditV3,
    PopulationCoverageContractV3,
    PopulationCoverageRowV3,
    QualifiedPopulationItemV3,
)
from deep_anc.dsp.control_band_contract import (
    BroadbandFullOctaveContractV3,
    ControlBandContract,
)


_FAMILIES = ("speech", "music", "environment", "machine")


def _population_audit() -> PopulationAuditV3:
    population_contract = PopulationCoverageContractV3.canonical()
    control = population_contract.control_band_contract
    items: list[QualifiedPopulationItemV3] = []
    # family마다 8개 independent component. 각 item은 parity가 같은 일부 band만
    # 자격을 갖고, 모든 8/7 band를 동시에 통과하는 item은 하나도 없다.
    for split in ("train", "val", "test"):
        for family in _FAMILIES:
            for item_index in range(8):
                physical = tuple(
                    item_index % 2 == band_index % 2 for band_index in range(8)
                )
                objective = tuple(
                    item_index % 2 == band_index % 2 for band_index in range(7)
                )
                items.append(
                    QualifiedPopulationItemV3(
                        item_id=f"{split}-{family}-{item_index}",
                        candidate_id=f"candidate-{split}-{family}-{item_index}",
                        split=split,
                        source_family=family,
                        lineage_component_id=f"component-{split}-{family}-{item_index}",
                        start_frame=4096 * item_index,
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
            "POPULATION_V3_AUTHORITY is None",
            "canonical training remains BLOCKED",
        ),
        candidates_verified=len(items),
        qualified_items=tuple(items),
        coverage=tuple(coverage),
    )


def _primitive(*, batch_size: int = 16, seed: int = 20260803):
    control = BroadbandFullOctaveContractV3.canonical()
    audit = _population_audit()
    return BroadbandFullOctaveBatchPrimitiveV3(
        control_band_contract=control.model_dump(mode="python"),
        control_band_contract_sha256=control.digest(),
        physical_identification_population_receipt=audit,
        physical_identification_population_receipt_sha256=audit.digest(),
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
    assert plan.physical_identification_population_receipt_sha256 == (
        primitive.population_audit.digest()
    )
    assert plan.canonical_training_eligible is False
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
