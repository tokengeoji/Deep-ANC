from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from deep_anc.dsp.reference_mode_causality import (
    AcousticReferenceRuntimeEvidence,
    assess_acoustic_reference,
    build_current_reference_mode_audit,
    first_transverse_mode_hz,
    load_current_causality_snapshot,
    phase_error_budget,
    propagating_rectangular_mode_count,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def current_audit():
    return build_current_reference_mode_audit(repo_root=REPO_ROOT)


def _complete_acoustic_evidence(
    *,
    advance: float = 600.0,
    adc: float = 10.0,
    inference: float = 100.0,
    dac: float = 20.0,
    secondary: float = 50.0,
    include_common_timeline: bool = True,
) -> AcousticReferenceRuntimeEvidence:
    return AcousticReferenceRuntimeEvidence(
        ref_to_err_advance_samples=advance,
        ref_to_err_advance_receipt_sha256=_sha("advance"),
        adc_observation_latency_samples=adc,
        adc_observation_receipt_sha256=_sha("adc"),
        inference_p99_samples=inference,
        inference_receipt_sha256=_sha("inference"),
        dac_output_latency_samples=dac,
        dac_output_receipt_sha256=_sha("dac"),
        secondary_acoustic_delay_samples=secondary,
        secondary_acoustic_receipt_sha256=_sha("secondary"),
        common_runtime_timeline_receipt_sha256=(
            _sha("common timeline") if include_common_timeline else None
        ),
    )


def test_current_strict_ps_files_are_same_capture_and_sha_bound(current_audit) -> None:
    timing = current_audit.timing
    primary = timing.primary
    secondary = timing.secondary

    assert primary.artifact_sha256 == (
        "23fa43f1ec46d5bca6bdad53938b81bb2d2c85afc4eee35e83c555b6c4f0c598"
    )
    assert secondary.artifact_sha256 == (
        "883c09364c00ad7aecdc038e38d0b6f8a49140fdc4b9788c2f5b0fc4686c2bee"
    )
    assert primary.capture_id == secondary.capture_id == (
        "5ac1313488c8434bb4d672a36503df59"
    )
    assert primary.source_raw_npz_sha256 == secondary.source_raw_npz_sha256 == (
        "31d563b163fe7dcb3f6b85e30e491a6775947e7f1b988690c3668fd13464b347"
    )
    assert (
        primary.source_analysis_npz_sha256
        == secondary.source_analysis_npz_sha256
        == "064ff82cc5c4ed4febabff856394c22fa4db69510251c1c96eaa9f87789fba94"
    )
    assert primary.sample_rate == secondary.sample_rate == 48_000
    assert primary.calibration_block_size == secondary.calibration_block_size == 256
    assert primary.xrun_count == secondary.xrun_count == 0
    assert timing.strict_capture_hardware_input_card == "APE"
    assert timing.strict_capture_hardware_output_product == "AB13X USB Audio"


def test_lead_and_handoff_are_only_contract_derived(current_audit) -> None:
    timing = current_audit.timing
    delays = timing.plant_delays
    contract = timing.training_timing_contract

    assert delays.primary_delay_samples == 1386
    assert delays.secondary_delay_samples == 1245
    assert delays.handoff_samples == 256
    assert delays.lead().samples == timing.derived_lead_samples == 115
    assert delays.lead().raw_samples == timing.derived_raw_lead_samples == 115
    assert contract.digital_reference_lead_samples == 115
    assert contract.primary_fir_peak_offset_samples == 245
    assert contract.primary_effective_delay_samples == 1631
    assert contract.synthetic_total_advance_samples == 1746
    assert contract.digest() == (
        "1d6723bbfbad1371fab9d38e827c59789eba35a98fae67e478c44e1fdb0061db"
    )
    # NPZ bulk peak 1642는 별도 estimator metadata다. TrainingTimingContract가 실제
    # compact FIR에서 얻은 1631을 덮어쓰거나 lead 계산에 사용하면 안 된다.
    assert timing.primary.bulk_delay_samples == 1642
    assert timing.primary.bulk_delay_samples != contract.primary_effective_delay_samples


def test_current_geometry_is_config_evidence_not_field_verified(current_audit) -> None:
    geometry = current_audit.geometry

    assert geometry.config_sha256 == (
        "a7091a3ebf4fe37ddd4503ddd84a22e36c0f17acbf978a65f0b30cdbb7fce5ff"
    )
    assert geometry.cross_section_m == (0.105, 0.105)
    assert geometry.interior_length_m == 1.19
    assert geometry.computed_first_transverse_mode_hz == pytest.approx(
        1633.3333333333335
    )
    assert geometry.configured_plane_wave_cutoff_hz == 1633.0
    assert geometry.geometry_authority == "repository_config_only_not_field_verified"
    assert geometry.error_mic_position_authority == "repository_comment_provisional"
    # X=0.1→1.1m의 기하 추정일 뿐 canonical REF→ERR 측정이 아니다.
    assert geometry.geometric_ref_to_err_advance_samples == pytest.approx(
        1000.0 / 343.0 * 48.0
    )


def test_phase_budget_and_rectangular_mode_math() -> None:
    _, phase_20_degree, samples_20_at_8k = phase_error_budget(
        frequency_hz=8_000.0,
        target_attenuation_db=20.0,
        sample_rate=48_000,
    )
    assert phase_20_degree == pytest.approx(5.7319679651977244)
    assert samples_20_at_8k == pytest.approx(0.09553279941996208)
    _, _, upper_edge = phase_error_budget(
        frequency_hz=8_000.0 * math.sqrt(2.0),
        target_attenuation_db=20.0,
        sample_rate=48_000,
    )
    assert upper_edge == pytest.approx(0.06755189029558946)

    kwargs = dict(width_m=0.105, height_m=0.105, speed_of_sound_mps=343.0)
    assert first_transverse_mode_hz(**kwargs) == pytest.approx(1633.3333333333335)
    assert propagating_rectangular_mode_count(frequency_hz=2_000.0, **kwargs) == 3
    assert propagating_rectangular_mode_count(frequency_hz=4_000.0, **kwargs) == 8
    assert propagating_rectangular_mode_count(frequency_hz=8_000.0, **kwargs) == 22


@pytest.mark.parametrize(
    ("frequency", "attenuation"),
    [(0.0, 20.0), (24_000.0, 20.0), (1_000.0, 0.0)],
)
def test_phase_budget_rejects_invalid_frequency_or_target(
    frequency: float, attenuation: float
) -> None:
    with pytest.raises(ValueError):
        phase_error_budget(
            frequency_hz=frequency,
            target_attenuation_db=attenuation,
            sample_rate=48_000,
        )


def test_all_seven_octaves_have_phase_and_five_point_spatial_contract(
    current_audit,
) -> None:
    rows = current_audit.phase_budgets_20db

    assert tuple(row.center_hz for row in rows) == (
        125.0,
        250.0,
        500.0,
        1000.0,
        2000.0,
        4000.0,
        8000.0,
    )
    assert tuple(row.modal_regime for row in rows) == (
        "plane_wave_band",
        "plane_wave_band",
        "plane_wave_band",
        "plane_wave_band",
        "crosses_first_transverse_cutoff",
        "higher_order_band",
        "higher_order_band",
    )
    assert tuple(row.propagating_mode_count_at_center for row in rows) == (
        1,
        1,
        1,
        1,
        3,
        8,
        22,
    )
    assert all(row.minimum_spatial_err_positions_for_quiet_zone == 5 for row in rows)
    assert all(row.single_point_is_quiet_zone_evidence is False for row in rows)
    assert all(row.physical_performance_claim is False for row in rows)
    assert rows[-1].maximum_timing_error_samples_at_upper_edge == pytest.approx(
        0.06755189029558946
    )


def test_current_mode_split_digital_is_structural_acoustic_is_blocked(
    current_audit,
) -> None:
    digital = current_audit.digital_reference
    acoustic = current_audit.acoustic_reference

    assert digital.causality_status == "CONDITIONALLY_CAUSAL"
    assert digital.derived_lead_samples == 115
    assert digital.strict_plant_trusted_band_hz == (150.0, 1600.0)
    assert digital.broadband_125_to_8000_octave_status == "BLOCKED"
    assert acoustic.causality_status == "BLOCKED"
    assert acoustic.required_latency_samples is None
    assert acoustic.causal_margin_samples is None
    assert acoustic.geometric_estimate_is_canonical_measurement is False
    assert set(acoustic.missing_or_unbound_terms) == {
        "ref_to_err_advance_samples",
        "adc_observation_latency_samples",
        "inference_p99_samples",
        "dac_output_latency_samples",
        "secondary_acoustic_delay_samples",
        "common_runtime_timeline_receipt_sha256",
    }
    # strict S=1245는 알려져 있지만 분해된 acoustic runtime latency로 재사용하지 않는다.
    assert acoustic.strict_secondary_calibration_delay_samples == 1245
    assert acoustic.strict_secondary_is_decomposed_runtime_latency is False
    assert current_audit.overall_broadband_deployment_status == "BLOCKED"
    assert current_audit.physical_attenuation_pass is False


def test_new_natural_sound_origin_changes_reference_causality(current_audit) -> None:
    replay, live = current_audit.natural_sound_routes

    assert replay.origin == "new_file_or_recording_replayed_by_jetson"
    assert replay.reference_mode == "digital_reference"
    assert replay.causality_status == "CONDITIONALLY_CAUSAL"
    assert live.origin == "live_sound_first_observed_by_upstream_ref_mic"
    assert live.reference_mode == "acoustic_reference"
    assert live.causality_status == "BLOCKED"


def test_acoustic_evidence_requires_value_sha_pairs_and_common_timeline() -> None:
    with pytest.raises(ValidationError, match="함께 있어야"):
        AcousticReferenceRuntimeEvidence(ref_to_err_advance_samples=140.0)

    geometry, timing = load_current_causality_snapshot(REPO_ROOT)
    missing_common = assess_acoustic_reference(
        geometry=geometry,
        timing=timing,
        evidence=_complete_acoustic_evidence(include_common_timeline=False),
    )
    assert missing_common.causality_status == "BLOCKED"
    assert missing_common.missing_or_unbound_terms == (
        "common_runtime_timeline_receipt_sha256",
    )
    assert missing_common.causal_margin_samples is None


def test_complete_same_timeline_acoustic_evidence_computes_without_double_counting(
    current_audit,
) -> None:
    result = assess_acoustic_reference(
        geometry=current_audit.geometry,
        timing=current_audit.timing,
        evidence=_complete_acoustic_evidence(),
    )

    # inference=100은 256-sample handoff 안의 deadline 증거이며 required latency에
    # 다시 더하지 않는다: ADC10 + handoff256 + DAC20 + acoustic S50 = 336.
    assert result.inference_deadline_met is True
    assert result.required_latency_samples == 336.0
    assert result.causal_margin_samples == 264.0
    assert result.causality_status == "CONDITIONALLY_CAUSAL"
    # 현 strict P/S highband 부재 때문에 broadband deployment는 여전히 BLOCKED다.
    assert result.broadband_125_to_8000_octave_status == "BLOCKED"
    assert result.physical_attenuation_pass is False


@pytest.mark.parametrize(
    "evidence",
    [
        _complete_acoustic_evidence(advance=300.0),
        _complete_acoustic_evidence(inference=257.0),
    ],
)
def test_negative_margin_or_missed_inference_deadline_blocks_acoustic(
    current_audit, evidence: AcousticReferenceRuntimeEvidence
) -> None:
    result = assess_acoustic_reference(
        geometry=current_audit.geometry,
        timing=current_audit.timing,
        evidence=evidence,
    )
    assert result.causality_status == "BLOCKED"
    assert result.broadband_random_live_sound_status == "BLOCKED"
    assert result.periodic_predictive_sound_status == "INCONCLUSIVE"


def test_audit_digest_is_stable_for_current_read_only_evidence(current_audit) -> None:
    # 경로 내용이나 계약이 바뀌면 의도적으로 갱신해야 하는 현장 snapshot digest다.
    assert current_audit.digest() == (
        "c8ed8b01b1df74d8c3fed80bbff961cac94cf4e0eb510d5e62015695810db35b"
    )
