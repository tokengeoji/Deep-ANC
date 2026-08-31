from __future__ import annotations

import hashlib

import numpy as np
import pytest

from deep_anc.data.stage2_2khz_population_audit import (
    Stage2PopulationAuditError,
    audit_stage2_lineage_assignments,
    build_stage2_coverage_cells,
    build_stage2_data_readiness,
    build_stage2_sentinel_cells,
    measure_stage2_recorded_signals,
    plan_minimum_multioctave_additions,
    write_audit_exclusive,
)
from deep_anc.dsp.stage2_2khz_contract import Stage2TwoKilohertzContract


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_actual_signal_measurement_has_five_objectives_and_legacy_panel() -> None:
    rng = np.random.default_rng(20260831)
    source = rng.normal(0.0, 0.01, 65_536)
    result = measure_stage2_recorded_signals(
        source,
        source,
        sample_rate=48_000,
        contract=Stage2TwoKilohertzContract.canonical(),
        nperseg=4096,
        noverlap=2048,
    )

    assert len(result["source_density_ratio"]) == 5
    assert len(result["target_err_density_ratio"]) == 5
    assert len(result["source_err_coherence"]) == 5
    assert all(result["source_density_pass"])
    assert all(result["target_err_density_pass"])
    assert all(result["source_err_coherence_pass"])
    assert all(result["population_joint_valid"])
    assert len(result["legacy_physical_joint_valid"]) == 7
    assert result["one_point_six_khz_sentinel_source_density_pass"] == (True,)
    assert result["one_point_six_khz_sentinel_target_err_density_pass"] == (True,)
    assert result["one_point_six_khz_sentinel_source_err_coherence_pass"] == (True,)
    assert result["one_point_six_khz_sentinel_population_joint_valid"] == (True,)


def _session_rows(
    *,
    groups: int = 4,
    fail_two_khz: bool = False,
    fail_sentinel: bool = False,
):
    rows = []
    for split in ("train", "val", "test"):
        for family in ("speech", "music", "environment", "machine"):
            for group in range(groups):
                valid = [True] * 5
                if fail_two_khz and split == "train" and family == "speech":
                    valid[-1] = False
                rows.append(
                    {
                        "session_id": f"{split}-{family}-{group}",
                        "split": split,
                        "source_family": family,
                        "group_id": f"{split}-{family}-component-{group}",
                        "source_density_pass": valid,
                        "target_path_joint_valid": valid,
                        "population_joint_valid": valid,
                        "one_point_six_khz_sentinel_source_density_pass": [
                            not fail_sentinel
                        ],
                        "one_point_six_khz_sentinel_target_path_joint_valid": [
                            not fail_sentinel
                        ],
                        "one_point_six_khz_sentinel_population_joint_valid": [
                            not fail_sentinel
                        ],
                    }
                )
    return rows


def test_coverage_has_every_split_family_octave_and_never_hides_missing_two_khz() -> None:
    contract = Stage2TwoKilohertzContract.canonical()
    passed = build_stage2_coverage_cells(_session_rows(), contract=contract)
    assert len(passed) == 3 * 4 * 5
    assert all(row["population_joint_pass"] for row in passed)

    failed = build_stage2_coverage_cells(
        _session_rows(fail_two_khz=True), contract=contract
    )
    target = [
        row
        for row in failed
        if row["split"] == "train"
        and row["source_family"] == "speech"
        and row["octave_center_hz"] == 2000.0
    ]
    assert len(target) == 1
    assert target[0]["population_joint_independent_groups"] == 0
    assert target[0]["population_joint_component_deficit"] == 4
    assert not target[0]["population_joint_pass"]
    assert all(
        row["population_joint_pass"]
        for row in failed
        if row is not target[0]
    )


def test_sentinel_population_is_a_separate_twelve_cell_fail_closed_gate() -> None:
    contract = Stage2TwoKilohertzContract.canonical()
    passed = build_stage2_sentinel_cells(_session_rows(), contract=contract)
    assert len(passed) == 3 * 4
    assert all(row["population_joint_pass"] for row in passed)

    failed = build_stage2_sentinel_cells(
        _session_rows(fail_sentinel=True), contract=contract
    )
    assert all(row["population_joint_component_deficit"] == 4 for row in failed)
    assert not any(row["population_joint_pass"] for row in failed)


def test_sentinel_deficit_is_layered_into_minimum_recording_slots() -> None:
    contract = Stage2TwoKilohertzContract.canonical()
    cells = build_stage2_coverage_cells(_session_rows(), contract=contract)
    sentinel = build_stage2_sentinel_cells(
        _session_rows(fail_sentinel=True), contract=contract
    )
    plan = plan_minimum_multioctave_additions(
        cells,
        contract=contract,
        sentinel_cells=sentinel,
    )
    assert plan["minimum_new_recording_slots_lower_bound"] == 3 * 4 * 4
    assert all(
        slot["one_point_six_khz_sentinel_required"]
        for row in plan["plans"]
        for slot in row["slots"]
    )


def test_three_groups_are_underpowered_even_with_many_sessions() -> None:
    contract = Stage2TwoKilohertzContract.canonical()
    rows = _session_rows(groups=3)
    # 같은 세 component의 session을 복제해도 independent group 수는 늘지 않는다.
    duplicates = []
    for index, row in enumerate(rows):
        duplicate = dict(row)
        duplicate["session_id"] = f"duplicate-{index}"
        duplicates.append(duplicate)
    cells = build_stage2_coverage_cells(rows + duplicates, contract=contract)
    assert all(row["population_joint_independent_groups"] == 3 for row in cells)
    assert all(row["population_joint_component_deficit"] == 1 for row in cells)
    assert not any(row["population_joint_pass"] for row in cells)


def test_minimum_plan_layers_multiple_octave_deficits_into_same_new_source() -> None:
    contract = Stage2TwoKilohertzContract.canonical()
    cells = build_stage2_coverage_cells(
        _session_rows(fail_two_khz=True), contract=contract
    )
    speech_train = [
        row
        for row in cells
        if row["split"] == "train" and row["source_family"] == "speech"
    ]
    speech_train[0]["population_joint_component_deficit"] = 2
    speech_train[1]["population_joint_component_deficit"] = 1
    plan = plan_minimum_multioctave_additions(cells, contract=contract)
    selected = next(
        row
        for row in plan["plans"]
        if row["split"] == "train" and row["source_family"] == "speech"
    )

    assert selected[
        "minimum_new_components_if_each_slot_covers_all_listed_octaves"
    ] == 4
    assert selected["slots"][0]["required_objective_octaves_hz"] == [
        125.0,
        250.0,
        2000.0,
    ]
    assert selected["slots"][1]["required_objective_octaves_hz"] == [
        125.0,
        2000.0,
    ]
    assert plan["final_unseen_policy"][
        "conditioned_training_stimulus_may_not_fill_test_slots"
    ]


def test_lineage_audit_detects_component_wav_and_original_clip_crossings() -> None:
    base = {
        "split": "train",
        "group_id": "speech-lineage-a",
        "source_pool_wav_sha256": _sha("wav-a"),
        "source_pool_wav_path": "data/source_pool/speech/speech_000.wav",
        "original_clips": ["speaker-book-000.flac"],
    }
    clean = audit_stage2_lineage_assignments(
        [base, {**base, "group_id": "speech-lineage-b"}]
    )
    assert clean["status"] == "PASS"

    crossing = audit_stage2_lineage_assignments(
        [base, {**base, "split": "test"}]
    )
    assert crossing["status"] == "BLOCKED"
    assert crossing["cross_split_counts"] == {
        "component": 1,
        "composite_wav_sha": 1,
        "composite_wav_path": 1,
        "original_clip": 1,
    }


def test_lineage_assignment_without_sha_or_clips_fails_closed() -> None:
    with pytest.raises(Stage2PopulationAuditError, match="불완전"):
        audit_stage2_lineage_assignments(
            [
                {
                    "split": "train",
                    "group_id": "g",
                    "source_pool_wav_sha256": "bad",
                    "source_pool_wav_path": "x.wav",
                    "original_clips": [],
                }
            ]
        )


def test_noncanonical_but_stronger_excitation_contract_is_not_silently_accepted() -> None:
    stronger = Stage2TwoKilohertzContract(required_excitation_lower_hz=60.0)
    signal = np.ones(8192, dtype=np.float64)
    with pytest.raises(Stage2PopulationAuditError, match="exact canonical"):
        measure_stage2_recorded_signals(
            signal,
            signal,
            sample_rate=48_000,
            contract=stronger,
            nperseg=1024,
            noverlap=512,
        )


def test_audit_writer_is_no_replace(tmp_path) -> None:
    target = tmp_path / "audit.json"
    write_audit_exclusive({"status": "BLOCKED", "evidence_sha256": _sha("x")}, target)
    with pytest.raises(FileExistsError):
        write_audit_exclusive(
            {"status": "PASS", "evidence_sha256": _sha("y")}, target
        )
    assert '"status": "BLOCKED"' in target.read_text(encoding="utf-8")


def test_recorded_deficit_does_not_block_independent_public_pretrain_axis() -> None:
    contract = Stage2TwoKilohertzContract.canonical()
    cells = build_stage2_coverage_cells(
        _session_rows(fail_two_khz=True), contract=contract
    )
    readiness = build_stage2_data_readiness(
        cells=cells,
        lineage_audit={"status": "PASS"},
        public_inventory={"status": "PASS", "blockers": []},
        contract=contract,
    )

    public = readiness["public_synthetic_scratch_pretrain"]
    recorded = readiness["recorded_measured_finetune"]
    assert public["status"] == "PASS"
    assert not public["recorded_population_required"]
    assert recorded["status"] == "BLOCKED"
    assert recorded["blockers"] == [
        "RECORDED_POPULATION_DEFICIT:train:speech:2000Hz:4"
    ]


def test_recorded_sentinel_deficit_does_not_block_public_pretrain_axis() -> None:
    contract = Stage2TwoKilohertzContract.canonical()
    readiness = build_stage2_data_readiness(
        cells=build_stage2_coverage_cells(_session_rows(), contract=contract),
        sentinel_cells=build_stage2_sentinel_cells(
            _session_rows(fail_sentinel=True), contract=contract
        ),
        lineage_audit={"status": "PASS"},
        public_inventory={"status": "PASS", "blockers": []},
        contract=contract,
    )
    assert readiness["public_synthetic_scratch_pretrain"]["status"] == "PASS"
    recorded = readiness["recorded_measured_finetune"]
    assert recorded["status"] == "BLOCKED"
    assert len(recorded["blockers"]) == 12
    assert all("RECORDED_SENTINEL_POPULATION_DEFICIT" in value for value in recorded["blockers"])


def test_public_deficit_does_not_demote_independent_recorded_axis() -> None:
    contract = Stage2TwoKilohertzContract.canonical()
    readiness = build_stage2_data_readiness(
        cells=build_stage2_coverage_cells(_session_rows(), contract=contract),
        lineage_audit={"status": "PASS"},
        public_inventory={
            "status": "BLOCKED",
            "blockers": ["CANONICAL_MIMII_MACHINE_MANIFEST_ABSENT"],
        },
        contract=contract,
    )

    assert readiness["public_synthetic_scratch_pretrain"]["status"] == "BLOCKED"
    assert readiness["recorded_measured_finetune"] == {
        "status": "PASS",
        "blockers": [],
        "scope": "recorded_source_target_joint_density_and_lineage_population_only",
        "public_pretrain_population_substitutes_for_recorded_deficits": False,
        "full_training_admission": False,
    }


def test_readiness_rejects_contradictory_public_pass_with_blocker() -> None:
    contract = Stage2TwoKilohertzContract.canonical()
    with pytest.raises(Stage2PopulationAuditError, match="모순"):
        build_stage2_data_readiness(
            cells=build_stage2_coverage_cells(_session_rows(), contract=contract),
            lineage_audit={"status": "PASS"},
            public_inventory={"status": "PASS", "blockers": ["STALE"]},
            contract=contract,
        )
