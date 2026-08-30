from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

from deep_anc.dsp.control_band_contract import (
    BROADBAND_V3_OCTAVE_OBJECTIVE_BANDS_HZ,
    BROADBAND_V3_OCTAVE_OBJECTIVE_CENTERS_HZ,
    BROADBAND_V3_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ,
    BROADBAND_V3_STAGE1_LOW_GUARD_SUBBANDS_HZ,
    BROADBAND_MEASUREMENT_PANELS_HZ,
    BROADBAND_POINT_CONTROL_SUBBANDS_HZ,
    BroadbandFullOctaveContractV3,
    BroadbandPlantEvidence,
    ControlBandContract,
    OCTAVE_8K_UPPER_HZ,
    audit_broadband_plant_evidence,
    max_timing_error_samples_for_attenuation,
    phase_error_degrees,
    resolve_control_band_contract,
)


_STAGE1_FROZEN_DIGEST = (
    "cf6216ce4bae35fd449b29c726c8b2c7d7d2f9a83adcde1b0b29fead642d0619"
)
_BROADBAND_V2_FROZEN_DIGEST = (
    "73c8fdf013fec94a3b8697d3be1353a5d59c33f8fd2b5973127fc159328f8047"
)
_BROADBAND_V3_FROZEN_DIGEST = (
    "53579b9ff8419ac19fb2458c29a3e8a94ffbb2eeb88cc07f34b76c68033989f2"
)
_STAGE1_FROZEN_JSON = (
    '{"contract_id":"stage1_strict_150_1600_v1",'
    '"high_band_requirement":"do_no_harm_only",'
    '"low_band_requirement":"positive_attenuation",'
    '"matched_fxlms_required":false,'
    '"measurement_panels_hz":[[150.0,1600.0]],'
    '"measurement_resolution_attenuation_db":10.0,'
    '"minimum_spatial_error_positions":1,'
    '"octave_centers_hz":[125.0,250.0,500.0,1000.0,1600.0],'
    '"point_control_subbands_hz":[[150.0,300.0],[300.0,600.0],'
    '[600.0,1000.0],[1000.0,1600.0]],'
    '"point_control_target_hz":[150.0,1600.0],'
    '"required_excitation_upper_hz":1600.0,'
    '"role":"stage1_strict","sample_rate":48000,'
    '"schema_version":"control_band_contract_v2",'
    '"source_families":["speech","music","environment","machine"],'
    '"spatial_validation_required":false}'
)
_BROADBAND_V2_FROZEN_JSON = (
    '{"contract_id":"broadband_point_control_150_11314_v2",'
    '"high_band_requirement":"matched_fxlms_superiority",'
    '"low_band_requirement":"positive_attenuation",'
    '"matched_fxlms_required":true,'
    '"measurement_panels_hz":[[100.0,1800.0],[1400.0,3200.0],'
    '[2800.0,6000.0],[5400.0,8500.0],[7800.0,11400.0]],'
    '"measurement_resolution_attenuation_db":20.0,'
    '"minimum_spatial_error_positions":5,'
    '"octave_centers_hz":[125.0,250.0,500.0,1000.0,1600.0,2000.0,4000.0,8000.0],'
    '"point_control_subbands_hz":[[150.0,300.0],[300.0,600.0],'
    '[600.0,1000.0],[1000.0,1600.0],[1600.0,2828.42712474619],'
    '[2828.42712474619,5656.85424949238],'
    '[5656.85424949238,11313.70849898476]],'
    '"point_control_target_hz":[150.0,11313.70849898476],'
    '"required_excitation_upper_hz":11313.70849898476,'
    '"role":"broadband_point_control","sample_rate":48000,'
    '"schema_version":"control_band_contract_v2",'
    '"source_families":["speech","music","environment","machine"],'
    '"spatial_validation_required":true}'
)


def _sha(character: str) -> str:
    return character * 64


def _canonical_contract_json(contract) -> str:
    return json.dumps(
        contract.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _valid_evidence(contract: ControlBandContract) -> BroadbandPlantEvidence:
    n_bands = len(contract.point_control_subbands_hz)
    n_panels = len(contract.measurement_panels_hz)
    phase_limits = tuple(
        max_timing_error_samples_for_attenuation(
            contract.measurement_resolution_attenuation_db,
            band[1],
            contract.sample_rate,
        )
        for band in contract.point_control_subbands_hz
    )
    return BroadbandPlantEvidence(
        control_band_contract_sha256=contract.digest(),
        primary_capture_id="capture-v2",
        secondary_capture_id="capture-v2",
        primary_raw_sha256=_sha("a"),
        secondary_raw_sha256=_sha("a"),
        primary_analysis_sha256=_sha("b"),
        secondary_analysis_sha256=_sha("b"),
        exact_plan_file_sha256=_sha("d"),
        exact_plan_payload_sha256=_sha("e"),
        exact_plan_pcm_sha256=_sha("f"),
        measurement_level_evidence_sha256=_sha("c"),
        fresh_meter_raw_sha256=_sha("1"),
        fresh_meter_receipt_sha256=_sha("2"),
        timing_marker_pcm_sha256=_sha("3"),
        fixed_clock_pilot_sha256=_sha("4"),
        submitted_pilot_validation_sha256=_sha("8"),
        submitted_pilot_cross_channel_null_sha256=_sha("9"),
        submitted_pilot_cross_channel_max_absolute=0.0,
        submitted_pilot_cross_channel_max_ratio=0.0,
        global_clock_input_domain=(
            "actual_submitted_int16_period_spectrum_not_intended_float"
        ),
        global_clock_map_sha256=_sha("5"),
        global_clock_slope_samples_per_sample=2.4835888604 / 6000.0,
        global_clock_intercept_samples=0.0,
        global_clock_max_residual_samples=0.05,
        clock_trajectory_agreement_samples=0.05,
        transition_anchor_valid_counts=(8, 8, 8, 8),
        callback_timing_valid=True,
        callback_sample_slip_count=0,
        panel_clock_offsets_samples=(0.0,) * n_panels,
        applied_per_drive_phase_repair_samples=(0.0,) * (2 * n_panels),
        primary_marker_delay_samples=500.0,
        secondary_marker_delay_samples=450.0,
        primary_marker_branch_width_samples=2000.0,
        secondary_marker_branch_width_samples=1450.0,
        primary_marker_alias_candidate_count=1,
        secondary_marker_alias_candidate_count=1,
        primary_bulk_delay_fractional_samples=500.125,
        secondary_bulk_delay_fractional_samples=450.125,
        primary_bulk_delay_samples=500,
        secondary_bulk_delay_samples=450,
        primary_effective_delay_samples=244,
        secondary_effective_delay_samples=194,
        pre_roll_samples=256,
        handoff_extra_samples=256,
        derived_lead_samples=206,
        panel_primary_minus_secondary_bulk_delay_samples=(50.0,) * n_panels,
        panel_relative_delay_deviation_samples=(0.0,) * n_panels,
        sample_rate=48_000,
        block_size=256,
        latency="low",
        observed_submitted_pcm=True,
        excitation_panels_hz=contract.measurement_panels_hz,
        verified_subbands_hz=contract.point_control_subbands_hz,
        primary_consistency=(0.951,) * n_bands,
        secondary_consistency=(0.951,) * n_bands,
        clock_valid_repeats=(8,) * n_panels,
        clock_min_adjacent_score_observed=(0.995,) * n_panels,
        relative_phase_jitter_samples=tuple(value * 0.99 for value in phase_limits),
        separation_crosscheck_agreement=(0.999,) * n_bands,
        separation_crosscheck_relative_error=(0.01,) * n_bands,
        measured_interpolation_agreement=(0.995,) * n_bands,
        measured_interpolation_relative_error=(0.10,) * n_bands,
        primary_compact_role="diagnostic_only",
        secondary_compact_role="diagnostic_only",
        primary_compact_training_eligible=False,
        secondary_compact_training_eligible=False,
        primary_compact_identifiability_sha256=_sha("6"),
        secondary_compact_identifiability_sha256=_sha("7"),
        compact_roundtrip_agreement=(0.995,) * n_bands,
        compact_roundtrip_relative_error=(0.10,) * n_bands,
        xrun_count=0,
        clip_count=0,
    )


def test_legacy_stage1_and_broadband_v2_serialization_and_digest_are_frozen():
    stage1 = ControlBandContract.stage1_strict()
    broadband_v2 = ControlBandContract.broadband_point_control()

    assert _canonical_contract_json(stage1) == _STAGE1_FROZEN_JSON
    assert stage1.digest() == _STAGE1_FROZEN_DIGEST
    assert _canonical_contract_json(broadband_v2) == _BROADBAND_V2_FROZEN_JSON
    assert broadband_v2.digest() == _BROADBAND_V2_FROZEN_DIGEST

    # 기존 payload의 직접 model_validate 행동도 v3 resolver와 무관하게 유지한다.
    restored = ControlBandContract.model_validate(
        broadband_v2.model_dump(mode="python")
    )
    assert _canonical_contract_json(restored) == _BROADBAND_V2_FROZEN_JSON
    assert restored.digest() == _BROADBAND_V2_FROZEN_DIGEST


def test_v3_separates_identification_objectives_and_stage1_guards():
    contract = BroadbandFullOctaveContractV3.canonical()

    assert contract.physical_identification_subbands_hz == (
        BROADBAND_V3_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ
    )
    assert contract.octave_objective_centers_hz == (
        BROADBAND_V3_OCTAVE_OBJECTIVE_CENTERS_HZ
    )
    assert contract.equal_weight_octave_objective_bands_hz == (
        BROADBAND_V3_OCTAVE_OBJECTIVE_BANDS_HZ
    )
    assert contract.stage1_low_guard_subbands_hz == (
        BROADBAND_V3_STAGE1_LOW_GUARD_SUBBANDS_HZ
    )
    assert len(contract.physical_identification_subbands_hz) == 8
    assert len(contract.equal_weight_octave_objective_bands_hz) == 7
    assert len(contract.stage1_low_guard_subbands_hz) == 4
    assert contract.physical_identification_subbands_hz != (
        contract.equal_weight_octave_objective_bands_hz
    )
    assert contract.required_excitation_lower_hz == 80.0
    assert contract.required_excitation_upper_hz == 11313.7084989848
    assert contract.legacy_v2_automatic_promotion_allowed is False
    assert contract.requires_exact_v3_contract_sha256 is True
    assert contract.digest() == _BROADBAND_V3_FROZEN_DIGEST

    with pytest.raises(ValidationError, match="frozen"):
        contract.required_excitation_lower_hz = 79.0


@pytest.mark.parametrize(
    "field,value,match",
    [
        (
            "physical_identification_subbands_hz",
            BROADBAND_V3_OCTAVE_OBJECTIVE_BANDS_HZ,
            "physical identification",
        ),
        (
            "equal_weight_octave_objective_bands_hz",
            BROADBAND_V3_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ,
            "equal-weight octave objective",
        ),
        (
            "stage1_low_guard_subbands_hz",
            BROADBAND_V3_STAGE1_LOW_GUARD_SUBBANDS_HZ[:-1],
            "Stage-1 low guard",
        ),
        ("required_excitation_lower_hz", 80.0000000001, "80 Hz 이하"),
        ("required_excitation_upper_hz", 11_313.7084989847, "11,313"),
    ],
)
def test_v3_rejects_band_role_confusion_and_incomplete_excitation(
    field, value, match
):
    payload = BroadbandFullOctaveContractV3.canonical().model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValidationError, match=match):
        BroadbandFullOctaveContractV3.model_validate(payload)


def test_explicit_resolver_never_promotes_v2_to_v3():
    broadband_v2 = ControlBandContract.broadband_point_control()
    broadband_v3 = BroadbandFullOctaveContractV3.canonical()

    resolved_v2 = resolve_control_band_contract(
        broadband_v2.model_dump(mode="python")
    )
    resolved_v3 = resolve_control_band_contract(
        broadband_v3.model_dump(mode="python")
    )
    assert type(resolved_v2) is ControlBandContract
    assert resolved_v2.digest() == _BROADBAND_V2_FROZEN_DIGEST
    assert type(resolved_v3) is BroadbandFullOctaveContractV3
    assert resolved_v3.digest() == _BROADBAND_V3_FROZEN_DIGEST

    with pytest.raises(ValidationError):
        BroadbandFullOctaveContractV3.model_validate(
            broadband_v2.model_dump(mode="python")
        )
    with pytest.raises(ValueError, match="지원하지 않는"):
        resolve_control_band_contract({"schema_version": "control_band_contract_v4"})


def test_v3_rejects_legacy_promotion_flag_even_with_other_exact_fields():
    payload = BroadbandFullOctaveContractV3.canonical().model_dump(mode="python")
    payload["legacy_v2_automatic_promotion_allowed"] = True

    with pytest.raises(ValidationError):
        BroadbandFullOctaveContractV3.model_validate(payload)


def test_v3_identification_decimal_grid_rejects_one_ulp_change():
    payload = BroadbandFullOctaveContractV3.canonical().model_dump(mode="python")
    bands = list(payload["physical_identification_subbands_hz"])
    bands[0] = (math.nextafter(bands[0][0], math.inf), bands[0][1])
    payload["physical_identification_subbands_hz"] = tuple(bands)

    with pytest.raises(ValidationError, match="physical identification"):
        BroadbandFullOctaveContractV3.model_validate(payload)


def test_broadband_contract_covers_complete_8khz_octave_and_all_families():
    contract = ControlBandContract.broadband_point_control()

    assert contract.point_control_subbands_hz == BROADBAND_POINT_CONTROL_SUBBANDS_HZ
    assert contract.measurement_panels_hz == BROADBAND_MEASUREMENT_PANELS_HZ
    assert contract.point_control_target_hz == (150.0, OCTAVE_8K_UPPER_HZ)
    assert OCTAVE_8K_UPPER_HZ == pytest.approx(11_313.708498984761)
    assert contract.required_excitation_upper_hz > 8_000.0
    assert contract.source_families == ("speech", "music", "environment", "machine")
    assert contract.minimum_spatial_error_positions == 5
    assert contract.matched_fxlms_required is True
    assert contract.measurement_resolution_attenuation_db == 20.0
    assert len(contract.digest()) == 64


def test_stage1_contract_remains_diagnostic_outside_1600hz():
    contract = ControlBandContract.stage1_strict()

    assert contract.point_control_target_hz == (150.0, 1600.0)
    assert contract.high_band_requirement == "do_no_harm_only"
    assert contract.matched_fxlms_required is False
    assert contract.spatial_validation_required is False


def test_broadband_contract_rejects_8khz_endpoint_instead_of_octave_upper():
    payload = ControlBandContract.broadband_point_control().model_dump(mode="python")
    payload["point_control_target_hz"] = (150.0, 8000.0)
    payload["point_control_subbands_hz"] = (
        *BROADBAND_POINT_CONTROL_SUBBANDS_HZ[:-1],
        (BROADBAND_POINT_CONTROL_SUBBANDS_HZ[-1][0], 8000.0),
    )
    payload["required_excitation_upper_hz"] = 8000.0

    with pytest.raises(ValidationError, match="광대역 point-control subband|11.314"):
        ControlBandContract.model_validate(payload)


def test_phase_budget_matches_2_4_8khz_one_sample_and_10db_limits():
    assert phase_error_degrees(1.0, 2000.0, 48_000.0) == pytest.approx(15.0)
    assert phase_error_degrees(1.0, 4000.0, 48_000.0) == pytest.approx(30.0)
    assert phase_error_degrees(1.0, 8000.0, 48_000.0) == pytest.approx(60.0)
    assert max_timing_error_samples_for_attenuation(10.0, 2000.0, 48_000.0) == pytest.approx(
        1.213, abs=0.001
    )
    assert max_timing_error_samples_for_attenuation(10.0, 4000.0, 48_000.0) == pytest.approx(
        0.606, abs=0.001
    )
    assert max_timing_error_samples_for_attenuation(10.0, 8000.0, 48_000.0) == pytest.approx(
        0.303, abs=0.001
    )


def test_broadband_plant_gate_passes_every_subband_at_the_boundary():
    contract = ControlBandContract.broadband_point_control()
    result = audit_broadband_plant_evidence(contract, _valid_evidence(contract))

    assert result.ok
    assert result.status == "PASS"
    assert result.reasons == ()
    assert result.max_timing_error_samples_by_subband[-1] == pytest.approx(
        0.06755189029558945
    )


def test_broadband_plant_gate_rejects_one_bad_highband_clock_and_consistency():
    contract = ControlBandContract.broadband_point_control()
    good = _valid_evidence(contract)
    bad_primary = (*good.primary_consistency[:-1], 0.949)
    bad_clock = (*good.clock_valid_repeats[:-1], 7)
    evidence = BroadbandPlantEvidence.model_validate(
        {
            **good.model_dump(mode="python"),
            "primary_consistency": bad_primary,
            "clock_valid_repeats": bad_clock,
        }
    )

    result = audit_broadband_plant_evidence(contract, evidence)

    assert not result.ok
    assert any("clock valid repeat" in reason for reason in result.reasons)
    assert any("P consistency" in reason for reason in result.reasons)


def test_broadband_plant_gate_rejects_submitted_pilot_cross_channel_leakage():
    contract = ControlBandContract.broadband_point_control()
    good = _valid_evidence(contract)
    evidence = BroadbandPlantEvidence.model_validate(
        {
            **good.model_dump(mode="python"),
            "submitted_pilot_cross_channel_max_absolute": 1.01e-8,
            "submitted_pilot_cross_channel_max_ratio": 1.01e-12,
        }
    )

    result = audit_broadband_plant_evidence(contract, evidence)

    assert not result.ok
    assert any("absolute null" in reason for reason in result.reasons)
    assert any("상대 null" in reason for reason in result.reasons)


def test_broadband_plant_gate_rejects_forged_lead_and_panel_relative_delay():
    contract = ControlBandContract.broadband_point_control()
    good = _valid_evidence(contract)
    evidence = BroadbandPlantEvidence.model_validate(
        {
            **good.model_dump(mode="python"),
            "derived_lead_samples": good.derived_lead_samples + 1,
            "panel_primary_minus_secondary_bulk_delay_samples": (
                *good.panel_primary_minus_secondary_bulk_delay_samples[:-1],
                50.5,
            ),
            # 의도적으로 0을 유지해 final P−S에서 재계산되지 않는
            # forged deviation이 따로 적발되는지도 검증한다.
            "panel_relative_delay_deviation_samples": (0.0,)
            * len(contract.measurement_panels_hz),
        }
    )

    result = audit_broadband_plant_evidence(contract, evidence)

    assert not result.ok
    assert any("PlantDelays.lead" in reason for reason in result.reasons)
    assert any("final broadband delay" in reason for reason in result.reasons)
    assert any("20dB" in reason for reason in result.reasons)


def test_broadband_plant_gate_rejects_stage1_contract_even_with_self_consistent_evidence():
    stage1 = ControlBandContract.stage1_strict()
    evidence = _valid_evidence(stage1)

    result = audit_broadband_plant_evidence(stage1, evidence)

    assert not result.ok
    assert any("stage1 strict" in reason for reason in result.reasons)


def test_broadband_evidence_rejects_non_sha_provenance():
    contract = ControlBandContract.broadband_point_control()
    payload = _valid_evidence(contract).model_dump(mode="python")
    payload["primary_raw_sha256"] = "legacy"

    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        BroadbandPlantEvidence.model_validate(payload)


def test_timing_budget_rejects_nonphysical_inputs():
    with pytest.raises(ValueError, match="양수"):
        max_timing_error_samples_for_attenuation(0.0, 8000.0, 48_000.0)
    with pytest.raises(ValueError, match="frequency"):
        phase_error_degrees(1.0, -1.0, 48_000.0)
    assert math.isfinite(max_timing_error_samples_for_attenuation(20.0, 8000.0, 48_000.0))
