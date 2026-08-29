from __future__ import annotations

import copy
import hashlib

import pytest

from deep_anc.data.broadband_recording_campaign import (
    MAX_SOURCE_CREST_FACTOR_DB,
    MIN_NATIVE_SAMPLE_RATE_HZ,
    RECORDING_SECONDS,
    REQUIRED_NATIVE_BANDWIDTH_UPPER_HZ,
    build_missing_source_campaign,
    calculate_minimum_new_groups,
    validate_campaign_candidate_set,
    validate_source_candidate_metadata,
)
from deep_anc.data.broadband_coverage_receipt import (
    NATIVE_BANDWIDTH_RESAMPLED_ONCE_ROLE,
    NATIVE_EXACT_TARGET_RATE_ROLE,
)
from deep_anc.dsp.control_band_contract import ControlBandContract


def _diagnostic(*, groups: int = 0):
    contract = ControlBandContract.broadband_point_control()
    family_row = {
        "sessions": 4,
        "subbands": [
            {"joint_pass_independent_groups": groups}
            for _ in contract.point_control_subbands_hz
        ],
    }
    return {
        "schema": "broadband_prerequisite_audit_v1",
        "recorded_coverage": {
            "schema": "recorded_broadband_coverage_diagnostic_v1",
            "role": "diagnostic_only_not_campaign_receipt",
            "control_band_contract_sha256": contract.digest(),
            "summary": {
                "by_split_family": {
                    split: {
                        family: copy.deepcopy(family_row)
                        for family in contract.source_families
                    }
                    for split in ("train", "val", "test")
                }
            },
        },
    }


def _candidate():
    matrix = [[0.5] * 7 for _ in range(9)]
    return {
        "split": "train",
        "source_family": "speech",
        "native_sample_rate_hz": 48_000,
        "processed_sample_rate_hz": 48_000,
        "processing_role": NATIVE_EXACT_TARGET_RATE_ROLE,
        "resample_count": 0,
        "resampler_algorithm": None,
        "verified_resampler_passband_upper_hz": None,
        "resampler_frequency_response_sha256": None,
        "synthetic_bandwidth_claimed": False,
        "lossless": True,
        "duration_seconds": 15.0,
        "crest_factor_db": 12.0,
        "source_density_ratios": copy.deepcopy(matrix),
        "predicted_err_density_ratios": copy.deepcopy(matrix),
        "native_content_sha256": hashlib.sha256(b"new-source").hexdigest(),
        "processed_content_sha256": hashlib.sha256(b"new-source").hexdigest(),
        "transform_receipt_sha256": hashlib.sha256(b"transform").hexdigest(),
        "lineage_component_id": "speech-new-speaker-and-book-component",
    }


def _candidate_for_rate(native_rate: int):
    candidate = _candidate()
    candidate["native_sample_rate_hz"] = native_rate
    if native_rate != 48_000:
        candidate["processing_role"] = NATIVE_BANDWIDTH_RESAMPLED_ONCE_ROLE
        candidate["resample_count"] = 1
        candidate["resampler_algorithm"] = "polyphase_fir"
        candidate["verified_resampler_passband_upper_hz"] = (
            REQUIRED_NATIVE_BANDWIDTH_UPPER_HZ
        )
        candidate["resampler_frequency_response_sha256"] = hashlib.sha256(
            f"response-{native_rate}".encode()
        ).hexdigest()
        candidate["processed_content_sha256"] = hashlib.sha256(
            f"processed-{native_rate}".encode()
        ).hexdigest()
    return candidate


def test_zero_highband_groups_require_exactly_48_new_sources_and_12_minutes():
    result = calculate_minimum_new_groups(_diagnostic(groups=0))
    assert result["minimum_new_source_groups"] == 48
    assert result["minimum_new_recording_sessions"] == 48
    assert result["minimum_audible_seconds"] == 48 * RECORDING_SECONDS == 720.0
    assert len(result["rows"]) == 3 * 4
    assert all(row["minimum_new_all_band_groups"] == 4 for row in result["rows"])


def test_cell_minimum_is_maximum_band_deficit_not_sum():
    diagnostic = _diagnostic(groups=4)
    target = diagnostic["recorded_coverage"]["summary"]["by_split_family"]["test"][
        "music"
    ]["subbands"]
    target[0]["joint_pass_independent_groups"] = 3
    target[1]["joint_pass_independent_groups"] = 2
    target[-1]["joint_pass_independent_groups"] = 0
    result = calculate_minimum_new_groups(diagnostic)
    row = next(
        row
        for row in result["rows"]
        if row["split"] == "test" and row["source_family"] == "music"
    )
    assert row["deficit_by_band"] == [1, 2, 0, 0, 0, 0, 4]
    assert row["minimum_new_all_band_groups"] == 4
    assert result["minimum_new_source_groups"] == 4


def test_missing_plan_never_fabricates_source_or_lineage():
    plan = build_missing_source_campaign(_diagnostic(groups=0))
    assert plan["status"].startswith("BLOCKED_")
    assert len(plan["slots"]) == 48
    assert all(slot["source"] is None for slot in plan["slots"])
    assert all(slot["lineage_component"] is None for slot in plan["slots"])
    assert len({slot["slot_id"] for slot in plan["slots"]}) == 48
    assert all(
        slot["requirements"]["native_audio"]["minimum_sample_rate_hz"]
        == MIN_NATIVE_SAMPLE_RATE_HZ
        for slot in plan["slots"]
    )


def test_valid_native_48k_all_band_candidate_passes_metadata_preflight():
    result = validate_source_candidate_metadata(
        _candidate(), expected_split="train", expected_family="speech"
    )
    assert result["status"] == "PASS_METADATA_ONLY"


@pytest.mark.parametrize("native_rate", [16_000, 22_050])
def test_native_rate_below_octave_nyquist_cannot_claim_coverage(native_rate):
    candidate = _candidate_for_rate(native_rate)
    with pytest.raises(ValueError, match="22628Hz 미만"):
        validate_source_candidate_metadata(
            candidate, expected_split="train", expected_family="speech"
        )


@pytest.mark.parametrize("native_rate", [24_000, 44_100, 48_000, 96_000])
def test_native_coverage_rates_use_appropriate_single_transform(native_rate):
    result = validate_source_candidate_metadata(
        _candidate_for_rate(native_rate),
        expected_split="train",
        expected_family="speech",
    )
    assert result["status"] == "PASS_METADATA_ONLY"


def test_exact_native_rate_floor_22628_is_physical_not_dac_rate():
    candidate = _candidate_for_rate(MIN_NATIVE_SAMPLE_RATE_HZ)
    result = validate_source_candidate_metadata(
        candidate, expected_split="train", expected_family="speech"
    )
    assert result["status"] == "PASS_METADATA_ONLY"


def test_processing_label_cannot_disguise_native_rate():
    candidate = _candidate_for_rate(96_000)
    candidate["processing_role"] = NATIVE_EXACT_TARGET_RATE_ROLE
    with pytest.raises(ValueError, match="모순"):
        validate_source_candidate_metadata(
            candidate, expected_split="train", expected_family="speech"
        )


def test_some_band_in_each_segment_cannot_fake_simultaneous_low_and_high_coverage():
    candidate = _candidate()
    for index in range(7):
        candidate["predicted_err_density_ratios"][index][index] = 0.1
    with pytest.raises(ValueError, match="all-seven-band PASS segment가 8개 미만"):
        validate_source_candidate_metadata(
            candidate, expected_split="train", expected_family="speech"
        )


def test_crest_ceiling_must_be_met_without_processing():
    candidate = _candidate()
    candidate["crest_factor_db"] = MAX_SOURCE_CREST_FACTOR_DB + 0.01
    with pytest.raises(ValueError, match="crest"):
        validate_source_candidate_metadata(
            candidate, expected_split="train", expected_family="speech"
        )


def _candidate_set(plan):
    candidates = {}
    for index, slot in enumerate(plan["slots"]):
        candidate = _candidate()
        candidate["split"] = slot["split"]
        candidate["source_family"] = slot["source_family"]
        candidate["native_content_sha256"] = hashlib.sha256(
            f"content-{index}".encode()
        ).hexdigest()
        candidate["processed_content_sha256"] = candidate[
            "native_content_sha256"
        ]
        candidate["transform_receipt_sha256"] = hashlib.sha256(
            f"transform-{index}".encode()
        ).hexdigest()
        candidate["lineage_component_id"] = f"lineage-component-{index}"
        candidates[slot["slot_id"]] = candidate
    return candidates


def test_candidate_set_is_exactly_48_unique_content_and_lineage_components():
    plan = build_missing_source_campaign(_diagnostic(groups=0))
    candidates = _candidate_set(plan)
    result = validate_campaign_candidate_set(
        plan,
        candidates,
        forbidden_content_sha256=[],
        forbidden_lineage_component_ids=[],
    )
    assert result == {
        "status": "PASS_METADATA_SET_ONLY",
        "candidate_count": 48,
        "unique_native_content_count": 48,
        "unique_processed_content_count": 48,
        "unique_transform_receipt_count": 48,
        "unique_lineage_component_count": 48,
    }


def test_same_speaker_artist_or_machine_component_cannot_be_split_into_fake_groups():
    plan = build_missing_source_campaign(_diagnostic(groups=0))
    candidates = _candidate_set(plan)
    first, second = list(candidates)[:2]
    candidates[second]["lineage_component_id"] = candidates[first][
        "lineage_component_id"
    ]
    with pytest.raises(ValueError, match="lineage를 여러 group"):
        validate_campaign_candidate_set(
            plan,
            candidates,
            forbidden_content_sha256=[],
            forbidden_lineage_component_ids=[],
        )


def test_existing_corpus_content_or_lineage_cannot_be_reused():
    plan = build_missing_source_campaign(_diagnostic(groups=0))
    candidates = _candidate_set(plan)
    first = next(iter(candidates.values()))
    with pytest.raises(ValueError, match="기존 recorded/synthetic content"):
        validate_campaign_candidate_set(
            plan,
            candidates,
            forbidden_content_sha256=[first["native_content_sha256"]],
            forbidden_lineage_component_ids=[],
        )
