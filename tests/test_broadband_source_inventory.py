from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deep_anc.data.broadband_coverage_receipt import (
    NATIVE_BANDWIDTH_RESAMPLED_ONCE_ROLE,
    NATIVE_EXACT_TARGET_RATE_ROLE,
)
from deep_anc.data.broadband_recording_campaign import (
    BROADBAND_RECORDING_CAMPAIGN_SCHEMA,
    REQUIRED_FAMILIES,
    REQUIRED_SPLITS,
)
from deep_anc.data.broadband_source_inventory import (
    BROADBAND_SOURCE_ACQUISITION_MANIFEST_SCHEMA,
    SOURCE_ACQUISITION_MANIFEST_ROLES,
    acquisition_manifest_contract,
    audit_acquisition_manifest,
    inspect_existing_pipeline_capability,
    required_campaign_slots,
)
from deep_anc.dsp.control_band_contract import ControlBandContract


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _campaign() -> dict:
    slots = []
    for split in REQUIRED_SPLITS:
        for family in REQUIRED_FAMILIES:
            for index in range(4):
                slots.append(
                    {
                        "slot_id": f"{split}-{family}-{index + 1:02d}",
                        "split": split,
                        "source_family": family,
                        "source": None,
                        "lineage_component": None,
                        "requirements": {},
                    }
                )
    return {
        "schema": BROADBAND_RECORDING_CAMPAIGN_SCHEMA,
        "role": "missing_source_specification_not_live_plan",
        "status": "BLOCKED_MISSING_VERIFIED_SOURCES_AND_FULLBAND_P",
        "control_band_contract_sha256": ControlBandContract.broadband_point_control().digest(),
        "minimum": {},
        "slots": slots,
        "live_preconditions": [],
        "evidence_sha256": _sha("campaign"),
    }


def _lineage_keys(family: str) -> list[str]:
    return {
        "speech": ["raw_content", "speaker", "book_or_work", "recording_session"],
        "music": [
            "raw_content",
            "perceptual_audio_alias",
            "artist",
            "album_or_release",
            "recording_session",
        ],
        "environment": [
            "raw_content",
            "original_recording_id",
            "field_recording_session",
            "captured_event",
        ],
        "machine": [
            "raw_content",
            "physical_machine_unit",
            "operating_run",
            "recording_session",
        ],
    }[family]


def _candidate(
    slot: dict[str, str],
    *,
    native_rate: int = 48_000,
    duration_seconds: float = 20.0,
    lossless: bool = True,
) -> dict:
    identity = slot["slot_id"]
    native_sha = _sha(f"native-{identity}")
    if native_rate == 48_000:
        processed_sha = native_sha
        role = NATIVE_EXACT_TARGET_RATE_ROLE
        resample_count = 0
        algorithm = None
        passband = None
        response_sha = None
    else:
        processed_sha = _sha(f"processed-{identity}")
        role = NATIVE_BANDWIDTH_RESAMPLED_ONCE_ROLE
        resample_count = 1
        algorithm = "polyphase_fir"
        passband = min(native_rate / 2.0, 11_400.0)
        response_sha = _sha(f"response-{identity}")
    matrix = [[0.30] * 7 for _ in range(9)]
    frame_count = int(round(duration_seconds * native_rate))
    contiguous_frames = int(round(15.0 * native_rate))
    decoded_sha = _sha(f"decoded-{identity}")
    metadata = {
        "split": slot["split"],
        "source_family": slot["source_family"],
        "native_sample_rate_hz": native_rate,
        "processed_sample_rate_hz": 48_000,
        "processing_role": role,
        "resample_count": resample_count,
        "resampler_algorithm": algorithm,
        "verified_resampler_passband_upper_hz": passband,
        "resampler_frequency_response_sha256": response_sha,
        "synthetic_bandwidth_claimed": False,
        "lossless": lossless,
        "duration_seconds": duration_seconds,
        "crest_factor_db": 10.0,
        "source_density_ratios": matrix,
        "predicted_err_density_ratios": matrix,
        "native_content_sha256": native_sha,
        "processed_content_sha256": processed_sha,
        "transform_receipt_sha256": _sha(f"transform-{identity}"),
        "lineage_component_id": f"lineage-{identity}",
    }
    return {
        "slot_id": identity,
        "candidate_metadata": metadata,
        "origin_audio": {
            "storage_locator": f"acquisition/{identity}.wav",
            "size_bytes": max(frame_count * 3, 1),
            "immutable_source_sha256": native_sha,
            "container": "WAV",
            "codec": "PCM",
            "subtype": "PCM_24",
            "lossless": lossless,
            "native_sample_rate_hz": native_rate,
            "native_nyquist_hz": native_rate / 2.0,
            "channels": 1,
            "frame_count": frame_count,
            "duration_seconds": duration_seconds,
            "contiguous_window_start_frame": 0,
            "contiguous_window_frames": contiguous_frames,
            "header_evidence_sha256": _sha(f"header-{identity}"),
        },
        "decode_provenance": {
            "decoder_fingerprint_sha256": _sha(f"decoder-{identity}"),
            "decoder_receipt_sha256": _sha(f"decoder-receipt-{identity}"),
            "decoded_native_pcm_sha256": decoded_sha,
            "decoded_sample_rate_hz": native_rate,
            "decoded_channels": 1,
            "decoded_frames": frame_count,
        },
        "spectral_evidence": {
            "evidence_sha256": _sha(f"spectrum-{identity}"),
            "decoded_native_pcm_sha256": decoded_sha,
            "control_band_contract_sha256": ControlBandContract.broadband_point_control().digest(),
            "canonical_fullband_plant_evidence_sha256": _sha("physical-fullband-P"),
            "window_start_frame": 0,
            "window_frames": contiguous_frames,
            "analysed_upper_hz": 11_400.0,
            "actual_11314_hz_covered": True,
            "source_density_ratios": matrix,
            "predicted_err_density_ratios": matrix,
        },
        "lineage_evidence": {
            "component_id": f"lineage-{identity}",
            "authority_sha256": _sha(f"lineage-authority-{identity}"),
            "family_values_sha256": _sha(f"lineage-values-{identity}"),
            "connected_component_keys": _lineage_keys(slot["source_family"]),
            "dsu_verified": True,
        },
        "corpus_disjointness": {
            "authority_sha256": _sha(f"disjoint-{identity}"),
            "raw_content_disjoint": True,
            "processed_content_disjoint": True,
            "lineage_component_disjoint": True,
        },
    }


def _manifest(campaign: dict, candidates: list[dict], *, status: str) -> dict:
    value = {
        "schema": BROADBAND_SOURCE_ACQUISITION_MANIFEST_SCHEMA,
        "role": SOURCE_ACQUISITION_MANIFEST_ROLES[status],
        "status": status,
        "contract_sha256": acquisition_manifest_contract()["contract_sha256"],
        "control_band_contract_sha256": ControlBandContract.broadband_point_control().digest(),
        "campaign_evidence_sha256": campaign["evidence_sha256"],
        "candidates": candidates,
    }
    value["evidence_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return value


def test_campaign_has_exact_48_slots_and_empty_inventory_is_exactly_48_short() -> None:
    campaign = _campaign()
    assert len(required_campaign_slots(campaign)) == 48
    result = audit_acquisition_manifest(campaign, None)
    assert result["status"] == "BLOCKED"
    assert result["eligible_candidate_count"] == 0
    assert result["candidate_deficit"] == 48
    assert all(row["deficit"] == 4 for row in result["by_split_family"])


def test_complete_verified_manifest_can_bridge_all_campaign_slots() -> None:
    campaign = _campaign()
    candidates = [_candidate(slot) for slot in required_campaign_slots(campaign)]
    result = audit_acquisition_manifest(
        campaign, _manifest(campaign, candidates, status="PASS")
    )
    assert result["status"] == "PASS"
    assert result["eligible_candidate_count"] == 48
    assert result["candidate_deficit"] == 0


@pytest.mark.parametrize("native_rate", [16_000, 22_050])
def test_16k_and_22050_are_not_8k_octave_candidates(native_rate: int) -> None:
    campaign = _campaign()
    slot = required_campaign_slots(campaign)[0]
    candidate = _candidate(slot, native_rate=native_rate)
    result = audit_acquisition_manifest(
        campaign, _manifest(campaign, [candidate], status="INCOMPLETE")
    )
    row = next(item for item in result["candidate_results"] if item["slot_id"] == slot["slot_id"])
    assert row["status"] == "REJECTED"
    assert any("native_rate" in reason for reason in row["reasons"])
    assert result["eligible_candidate_count"] == 0


def test_five_second_lossless_source_cannot_be_repeated_or_concatenated() -> None:
    campaign = _campaign()
    slot = required_campaign_slots(campaign)[0]
    candidate = _candidate(slot, native_rate=44_100, duration_seconds=5.0)
    result = audit_acquisition_manifest(
        campaign, _manifest(campaign, [candidate], status="INCOMPLETE")
    )
    row = next(item for item in result["candidate_results"] if item["slot_id"] == slot["slot_id"])
    assert "contiguous_native_window_shorter_than_15s" in row["reasons"]
    assert result["eligible_candidate_count"] == 0


def test_lossy_source_stays_blocked_even_with_decoder_and_decoded_pcm_sha() -> None:
    campaign = _campaign()
    slot = required_campaign_slots(campaign)[0]
    candidate = _candidate(slot, native_rate=44_100, lossless=False)
    candidate["origin_audio"].update(
        {"container": "MPEG", "codec": "MP3", "subtype": "MPEG_LAYER_III"}
    )
    result = audit_acquisition_manifest(
        campaign, _manifest(campaign, [candidate], status="INCOMPLETE")
    )
    row = next(item for item in result["candidate_results"] if item["slot_id"] == slot["slot_id"])
    assert "native_source_not_lossless_current_policy" in row["reasons"]
    assert candidate["decode_provenance"]["decoder_fingerprint_sha256"]
    assert candidate["decode_provenance"]["decoded_native_pcm_sha256"]


def test_duplicate_lineage_invalidates_all_alias_slots() -> None:
    campaign = _campaign()
    slots = required_campaign_slots(campaign)[:2]
    candidates = [_candidate(slot) for slot in slots]
    candidates[1]["candidate_metadata"]["lineage_component_id"] = candidates[0][
        "candidate_metadata"
    ]["lineage_component_id"]
    candidates[1]["lineage_evidence"]["component_id"] = candidates[0][
        "lineage_evidence"
    ]["component_id"]
    result = audit_acquisition_manifest(
        campaign, _manifest(campaign, candidates, status="INCOMPLETE")
    )
    rows = [
        item
        for item in result["candidate_results"]
        if item["slot_id"] in {slot["slot_id"] for slot in slots}
    ]
    assert all("duplicate_lineage_component_id" in row["reasons"] for row in rows)
    assert result["eligible_candidate_count"] == 0


def test_extra_unknown_candidate_cannot_hide_behind_complete_48_slots() -> None:
    campaign = _campaign()
    candidates = [_candidate(slot) for slot in required_campaign_slots(campaign)]
    extra = _candidate(required_campaign_slots(campaign)[0])
    extra["slot_id"] = "train-speech-unknown"
    candidates.append(extra)
    result = audit_acquisition_manifest(
        campaign, _manifest(campaign, candidates, status="PASS")
    )
    assert result["status"] == "BLOCKED"
    assert "unknown_or_duplicate_candidate_slots" in result["top_level_reasons"]


def test_current_repository_has_placeholder_and_validator_but_no_publisher() -> None:
    root = Path(__file__).resolve().parents[1]
    result = inspect_existing_pipeline_capability(root)
    assert result["placeholder_campaign_builder_present"] is True
    assert result["ready_source_plan_validator_present"] is True
    assert result["ready_source_plan_publisher_present"] is False
    assert result["default_ready_source_plan_exists"] is False
    assert result["can_currently_make_actual_48_source_live_plan"] is False
