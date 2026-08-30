from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from deep_anc.data.broadband_recording_campaign import (
    BROADBAND_RECORDING_CAMPAIGN_SCHEMA,
    REQUIRED_FAMILIES,
    REQUIRED_SPLITS,
)
from deep_anc.data.broadband_source_contract_v2 import (
    BOUNDARY_FADE_FRAMES,
    BROADBAND_SOURCE_MANIFEST_V2_SCHEMA,
    SourceContractV2Blocked,
    boundary_fade_coefficients_sha256,
    issue_source_manifest_v2_noreplace,
    source_contract_v2,
    validate_source_manifest_v2,
)
from deep_anc.data.broadband_source_inventory import required_campaign_slots
from deep_anc.data.recorded_v2_capture import SOURCE_FRAMES
from deep_anc.dsp.control_band_contract import ControlBandContract


ISSUER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/data/issue_broadband_source_manifest_v2.py"
)


def _issuer_module():
    spec = importlib.util.spec_from_file_location(
        "_broadband_source_v2_issuer_probe", ISSUER_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def test_issuer_repo_file_rejects_internal_symlink_parent(tmp_path) -> None:
    module = _issuer_module()
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "draft.json").write_text("{}", encoding="utf-8")
    (repository / "linked").symlink_to(outside, target_is_directory=True)
    module.ROOT = repository

    with pytest.raises(ValueError, match="symlink"):
        module._repo_file("linked/draft.json")


def _ref(label: str, *, size: int = 97) -> dict:
    return {"path": f"evidence/{label}", "size_bytes": size, "sha256": _sha(label)}


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
        "evidence_sha256": _sha("campaign-v2"),
    }


def _component(
    seed: str,
    *,
    source_family: str,
    assigned_split: str,
    lossless: bool = False,
    native_rate: int = 44_100,
    native_channels: int = 2,
    native_seconds: float = 20.0,
    excerpt_seconds: float = 15.0,
) -> dict:
    native_frames = int(round(native_rate * native_seconds))
    processed_frames = int(round(48_000 * native_seconds))
    native_excerpt_frames = int(round(native_rate * excerpt_seconds))
    processed_excerpt_frames = int(round(48_000 * excerpt_seconds))
    original_sha = _sha(f"original-{seed}")
    decoded_sha = _sha(f"decoded-{seed}")
    processed_pcm_sha = _sha(f"processed-pcm-{seed}")
    excerpt_native_sha = _sha(f"excerpt-native-{seed}")
    excerpt_processed_sha = _sha(f"excerpt-processed-{seed}")
    return {
        "component_id": f"component-{seed}",
        "lineage_component_id": f"lineage-{seed}",
        "source_family": source_family,
        "assigned_split": assigned_split,
        "original": {
            "file": {
                "path": f"source/{seed}.{'flac' if lossless else 'mp3'}",
                "size_bytes": 12345,
                "sha256": original_sha,
            },
            "encoding": {
                "container": "FLAC" if lossless else "MPEG",
                "codec": "FLAC" if lossless else "MP3",
                "subtype": "PCM_24" if lossless else "MPEG_LAYER_III",
                "lossless": lossless,
                "native_sample_rate_hz": native_rate,
                "native_nyquist_hz": native_rate / 2.0,
                "channels": native_channels,
                "frame_count": native_frames,
            },
            "header_receipt": _ref(f"header-{seed}.json"),
        },
        "decode": {
            "decoder_runtime_fingerprint_sha256": _sha(f"decoder-{seed}"),
            "decoder_receipt": _ref(f"decoder-{seed}.json"),
            "original_file_sha256": original_sha,
            "decoded_pcm_file": {
                "path": f"decoded/{seed}.f32le",
                "size_bytes": native_frames * 4,
                "sha256": decoded_sha,
            },
            "decoded_pcm_sha256": decoded_sha,
            "pcm_dtype": "little_endian_float32_mono_raw",
            "sample_rate_hz": native_rate,
            "channels": 1,
            "frames": native_frames,
        },
        "processed": {
            "wav_file": _ref(f"processed-{seed}.wav", size=processed_frames * 4 + 44),
            "pcm_file": {
                "path": f"processed/{seed}.f32le",
                "size_bytes": processed_frames * 4,
                "sha256": processed_pcm_sha,
            },
            "transform_receipt": _ref(f"transform-{seed}.json"),
            "input_decoded_pcm_sha256": decoded_sha,
            "processed_pcm_sha256": processed_pcm_sha,
            "pcm_dtype": "little_endian_float32_mono_raw",
            "sample_rate_hz": 48_000,
            "channels": 1,
            "frames": processed_frames,
            "resample_count": 0 if native_rate == 48_000 else 1,
            "resampler": None
            if native_rate == 48_000
            else {
                "algorithm": "polyphase_fir",
                "implementation_fingerprint_sha256": _sha(f"resampler-{seed}"),
                "frequency_response_receipt": _ref(f"resampler-response-{seed}.json"),
                "verified_passband_upper_hz": 11_400.0,
            },
        },
        "excerpt": {
            "native_start_frame": 0,
            "native_frames": native_excerpt_frames,
            "native_excerpt_pcm_file": {
                "path": f"excerpt/{seed}.native.f32le",
                "size_bytes": native_excerpt_frames * 4,
                "sha256": excerpt_native_sha,
            },
            "native_excerpt_pcm_sha256": excerpt_native_sha,
            "processed_start_frame": 0,
            "processed_frames": processed_excerpt_frames,
            "processed_excerpt_pcm_file": {
                "path": f"excerpt/{seed}.f32le",
                "size_bytes": processed_excerpt_frames * 4,
                "sha256": excerpt_processed_sha,
            },
            "processed_excerpt_pcm_sha256": excerpt_processed_sha,
        },
        "pre_eq_spectral_crest": {
            "receipt": _ref(f"component-spectrum-{seed}.json"),
            "decoded_pcm_sha256": decoded_sha,
            "native_excerpt_pcm_sha256": excerpt_native_sha,
            "processed_excerpt_pcm_sha256": excerpt_processed_sha,
            "control_band_contract_sha256": ControlBandContract.broadband_point_control().digest(),
            "point_control_subbands_hz": [
                list(band)
                for band in ControlBandContract.broadband_point_control().point_control_subbands_hz
            ],
            "analysed_upper_hz": 11_400.0,
            "actual_native_bandwidth_verified": True,
            "density_ratios_7": [0.30] * 7,
            "crest_factor_db": 10.0,
            "boundary_or_eq_used_for_evidence": False,
        },
    }


def _candidate(
    slot: dict[str, str],
    *,
    mode: str = "single_long_form",
    lossless: bool = False,
    shared_component_seed: str | None = None,
) -> dict:
    candidate_id = f"candidate-{slot['slot_id']}"
    if mode == "single_long_form":
        seeds = [shared_component_seed or slot["slot_id"]]
        components = [
            _component(
                seeds[0],
                source_family=slot["source_family"],
                assigned_split=slot["split"],
                lossless=lossless,
            )
        ]
    else:
        seeds = [
            f"{slot['slot_id']}-short-{index}" for index in range(3)
        ]
        if shared_component_seed:
            seeds[0] = shared_component_seed
        components = [
            _component(
                seed,
                source_family=slot["source_family"],
                assigned_split=slot["split"],
                lossless=True,
                native_seconds=5.0,
                excerpt_seconds=5.0,
            )
            for seed in seeds
        ]
    ordered = [row["component_id"] for row in components]
    pre_eq_sha = _sha(f"pre-eq-{candidate_id}")
    boundaries = []
    cumulative = 0
    for left, right in zip(components, components[1:], strict=False):
        cumulative += left["excerpt"]["processed_frames"]
        boundaries.append(
            {
                "left_component_id": left["component_id"],
                "right_component_id": right["component_id"],
                "output_frame": cumulative,
                "fade_frames_each_side": BOUNDARY_FADE_FRAMES,
                "coefficient_q15_sha256": boundary_fade_coefficients_sha256(),
                "receipt": _ref(f"boundary-{candidate_id}-{cumulative}.json"),
            }
        )
    rank = 0 if mode == "single_long_form" and lossless else 1 if mode == "single_long_form" else 2
    lineages = sorted(row["lineage_component_id"] for row in components)
    union_sha = hashlib.sha256(
        json.dumps(lineages, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    matrix = [[0.30] * 7 for _ in range(9)]
    submitted_sha = _sha(f"submitted-{candidate_id}")
    return {
        "slot_id": slot["slot_id"],
        "candidate_id": candidate_id,
        "split": slot["split"],
        "source_family": slot["source_family"],
        "mode": mode,
        "components": components,
        "composition": {
            "algorithm": "sequential_no_overlap_linear_q15_boundary_fade_v1",
            "ordered_component_ids": ordered,
            "output_frames": SOURCE_FRAMES,
            "boundaries": boundaries,
            "pre_eq_pcm_sha256": pre_eq_sha,
            "recipe_receipt": _ref(f"composition-{candidate_id}.json"),
        },
        "lineage_union": {
            "component_lineage_ids": lineages,
            "union_identity_sha256": union_sha,
            "dsu_authority_sha256": _sha(f"dsu-{candidate_id}"),
            "all_components_same_family": True,
            "no_component_reuse_across_candidates_or_splits": True,
        },
        "selection_evidence": {
            "selected_preference_rank": rank,
            "eligible_better_rank_candidate_count": 0,
            "inventory_authority_sha256": _sha("inventory-v2"),
            "receipt": _ref(f"selection-{candidate_id}.json"),
            "reason": "fixture preference chain",
        },
        "eq_transform": {
            "policy_id": "identity-eq-v2",
            "input_pre_eq_pcm_sha256": pre_eq_sha,
            "output_post_eq_pcm_sha256": pre_eq_sha,
            "receipt": _ref(f"eq-{candidate_id}.json"),
            "actual_crest_increase_db": 0.0,
            "adaptive_to_candidate": False,
        },
        "final_submission": {
            "role": "coverage_source_not_unmodified_level5_challenge",
            "post_eq_pcm_sha256": pre_eq_sha,
            "processed_wav_file": _ref(f"final-{candidate_id}.wav", size=SOURCE_FRAMES * 4 + 44),
            "submitted_q15le_file": {
                "path": f"submitted/{candidate_id}.s16le",
                "size_bytes": SOURCE_FRAMES * 2,
                "sha256": submitted_sha,
            },
            "submitted_q15_pcm_sha256": submitted_sha,
            "sample_rate_hz": 48_000,
            "frames": SOURCE_FRAMES,
            "dtype": "little_endian_int16_mono_raw",
            "gain_q15": 4000,
            "peak_int16": 4000,
            "crest_factor_db": 10.0,
            "control_band_contract_sha256": ControlBandContract.broadband_point_control().digest(),
            "point_control_subbands_hz": [
                list(band)
                for band in ControlBandContract.broadband_point_control().point_control_subbands_hz
            ],
            "segment_start_frames": [12_000 + 72_000 * index for index in range(9)],
            "segment_frames": 72_000,
            "source_density_ratios_9x7": matrix,
            "source_density_pass_counts_7": [9] * 7,
            "predicted_err_density_ratios_9x7": matrix,
            "predicted_err_density_pass_counts_7": [9] * 7,
            "spectral_crest_receipt": _ref(f"final-spectrum-{candidate_id}.json"),
            "canonical_fullband_plant_evidence_sha256": _sha("plant-v2"),
            "unmodified_natural_challenge": False,
        },
        "corpus_disjointness": {
            "authority_sha256": _sha(f"disjoint-{candidate_id}"),
            "component_ids": sorted(ordered),
            "all_raw_content_disjoint": True,
            "all_decoded_and_processed_content_disjoint": True,
            "all_lineage_components_disjoint": True,
        },
    }


def _eq_policy_set() -> dict:
    return {
        "mode": "none",
        "predeclared_policy_commit_sha": "a" * 40,
        "candidate_analysis_commit_sha": "b" * 40,
        "ancestry_receipt": _ref("eq-ancestry.json"),
        "authority_file": _ref("eq-policy.json"),
        "policies": [
            {
                "policy_id": "identity-eq-v2",
                "scope": "global",
                "source_family": None,
                "fir_coefficients_sha256": _sha("identity-fir"),
                "frequency_response_receipt": _ref("identity-response.json"),
                "taps": 1,
                "actual_peak_boost_db": 0.0,
                "actual_max_attenuation_db": 0.0,
                "maximum_crest_increase_db": 0.0,
                "adaptive_to_candidate": False,
            }
        ],
    }


def _manifest(campaign: dict, candidates: list[dict]) -> dict:
    value = {
        "schema": BROADBAND_SOURCE_MANIFEST_V2_SCHEMA,
        "role": "candidate_evidence_not_live_source_plan",
        "status": "DRAFT",
        "synthetic_fixture": True,
        "contract_sha256": source_contract_v2()["contract_sha256"],
        "control_band_contract_sha256": ControlBandContract.broadband_point_control().digest(),
        "campaign_evidence_sha256": campaign["evidence_sha256"],
        "physical_fullband_plant_evidence": {
            "path": "plant/fullband.json",
            "size_bytes": 123,
            "sha256": _sha("plant-v2"),
        },
        "eq_policy_set": _eq_policy_set(),
        "selection_inventory_authority_sha256": _sha("inventory-v2"),
        "unmodified_level5_challenge_required": True,
        "candidates": candidates,
    }
    value["evidence_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return value


def _audit_one(candidate: dict) -> dict:
    campaign = _campaign()
    return validate_source_manifest_v2(
        _manifest(campaign, [candidate]), campaign=campaign
    )


def _rejection_for(result: dict, slot_id: str) -> str | None:
    for row in result["rejected"]:
        if row["slot_id"] == slot_id and row["reason"] != "candidate_missing":
            return row["reason"]
    return None


def test_v2_is_separate_and_does_not_change_v1_or_open_live_authority() -> None:
    contract = source_contract_v2()
    assert contract["v1_unchanged"] is True
    assert contract["live_authority"] is None
    assert contract["issuer_authority"] is None
    assert contract["natural_unmodified_level5_challenge_required"] is True
    assert contract["component_pre_eq_all_seven_band_density_minimum"] == 0.25
    assert contract["joint_causal_operator_npz_schema"] == (
        "fullband_causal_joint_fir_operator_npz_v4"
    )
    assert contract["final_per_band_passing_segments_minimum"] == 8
    assert contract["final_segment_frames"] == 72_000
    assert contract["final_segment_start_frames"] == [
        12_000 + 72_000 * index for index in range(9)
    ]


def test_immutable_mp3_long_form_is_structurally_allowed_with_full_decode_chain() -> None:
    campaign = _campaign()
    slot = required_campaign_slots(campaign)[0]
    result = _audit_one(_candidate(slot, lossless=False))
    assert _rejection_for(result, slot["slot_id"]) is None


def test_component_family_and_split_must_match_campaign_slot() -> None:
    campaign = _campaign()
    slot = required_campaign_slots(campaign)[0]
    candidate = _candidate(slot)
    candidate["components"][0]["assigned_split"] = "test"
    result = _audit_one(candidate)
    assert "family/split" in (_rejection_for(result, slot["slot_id"]) or "")
    assert result["actual_verified_candidate_count"] == 0
    assert result["status"] == "BLOCKED"  # other 47 slots and no local bytes/plant


def test_lossless_long_form_has_priority_rank_zero() -> None:
    campaign = _campaign()
    slot = required_campaign_slots(campaign)[0]
    candidate = _candidate(slot, lossless=True)
    candidate["selection_evidence"]["selected_preference_rank"] = 1
    result = _audit_one(candidate)
    assert "우선순위" in (_rejection_for(result, slot["slot_id"]) or "")


@pytest.mark.parametrize("native_rate", [16_000, 22_050])
def test_every_component_still_requires_native_8k_octave_bandwidth(native_rate: int) -> None:
    campaign = _campaign()
    slot = required_campaign_slots(campaign)[0]
    candidate = _candidate(slot)
    component = candidate["components"][0]
    component["original"]["encoding"]["native_sample_rate_hz"] = native_rate
    component["original"]["encoding"]["native_nyquist_hz"] = native_rate / 2.0
    component["decode"]["sample_rate_hz"] = native_rate
    result = _audit_one(candidate)
    assert "native fs/Nyquist" in (_rejection_for(result, slot["slot_id"]) or "")


def test_compressed_source_missing_decoder_fingerprint_is_rejected() -> None:
    campaign = _campaign()
    slot = required_campaign_slots(campaign)[0]
    candidate = _candidate(slot)
    candidate["components"][0]["decode"]["decoder_runtime_fingerprint_sha256"] = ""
    result = _audit_one(candidate)
    assert "decoder fingerprint" in (_rejection_for(result, slot["slot_id"]) or "")


def test_short_sequence_requires_three_components_and_exact_boundary_receipts() -> None:
    campaign = _campaign()
    slot = required_campaign_slots(campaign)[0]
    candidate = _candidate(slot, mode="multi_component_sequence")
    result = _audit_one(candidate)
    assert _rejection_for(result, slot["slot_id"]) is None
    broken = copy.deepcopy(candidate)
    broken["composition"]["boundaries"][0]["coefficient_q15_sha256"] = _sha("wrong-fade")
    result = _audit_one(broken)
    assert "boundary fade" in (_rejection_for(result, slot["slot_id"]) or "")


def test_two_short_components_cannot_replace_minimum_three() -> None:
    campaign = _campaign()
    slot = required_campaign_slots(campaign)[0]
    candidate = _candidate(slot, mode="multi_component_sequence")
    candidate["components"] = candidate["components"][:2]
    result = _audit_one(candidate)
    assert "component 수" in (_rejection_for(result, slot["slot_id"]) or "")


def test_boundary_or_eq_cannot_create_component_native_bandwidth_evidence() -> None:
    campaign = _campaign()
    slot = required_campaign_slots(campaign)[0]
    candidate = _candidate(slot, mode="multi_component_sequence")
    candidate["components"][0]["pre_eq_spectral_crest"][
        "boundary_or_eq_used_for_evidence"
    ] = True
    result = _audit_one(candidate)
    assert "native pre-EQ" in (_rejection_for(result, slot["slot_id"]) or "")


def test_final_predicted_err_9x7_gate_cannot_be_lowered_by_composition() -> None:
    campaign = _campaign()
    slot = required_campaign_slots(campaign)[0]
    candidate = _candidate(slot, mode="multi_component_sequence")
    for row in candidate["final_submission"]["predicted_err_density_ratios_9x7"]:
        row[-1] = 0.01
    candidate["final_submission"]["predicted_err_density_pass_counts_7"][-1] = 0
    result = _audit_one(candidate)
    assert "band별 PASS segment" in (_rejection_for(result, slot["slot_id"]) or "")


@pytest.mark.parametrize("band_index", [0, 6])
def test_final_density_fails_low_and_high_bands_independently(band_index: int) -> None:
    campaign = _campaign()
    slot = required_campaign_slots(campaign)[0]
    candidate = _candidate(slot)
    # 다른 여섯 band는 9/9 PASS인 채 선택한 저역 또는 고역만 7/9로 만든다.
    for row in candidate["final_submission"]["source_density_ratios_9x7"][:2]:
        row[band_index] = 0.01
    candidate["final_submission"]["source_density_pass_counts_7"][band_index] = 7
    result = _audit_one(candidate)
    assert "band별 PASS segment" in (_rejection_for(result, slot["slot_id"]) or "")


def test_global_component_reuse_invalidates_both_candidates_across_slots() -> None:
    campaign = _campaign()
    slots = required_campaign_slots(campaign)[:2]
    candidates = [
        _candidate(slots[0], shared_component_seed="forbidden-shared"),
        _candidate(slots[1], shared_component_seed="forbidden-shared"),
    ]
    result = validate_source_manifest_v2(
        _manifest(campaign, candidates), campaign=campaign
    )
    reasons = {
        row["slot_id"]: row["reason"]
        for row in result["rejected"]
        if "component_reused" in row["reason"]
    }
    assert set(reasons) == {slot["slot_id"] for slot in slots}


def test_bounded_eq_rejects_candidate_adaptive_and_excessive_policy() -> None:
    campaign = _campaign()
    slot = required_campaign_slots(campaign)[0]
    candidate = _candidate(slot)
    candidate["eq_transform"]["adaptive_to_candidate"] = True
    result = _audit_one(candidate)
    assert "EQ input/adaptive" in (_rejection_for(result, slot["slot_id"]) or "")

    manifest = _manifest(campaign, [_candidate(slot)])
    manifest["eq_policy_set"]["mode"] = "global_fixed"
    manifest["eq_policy_set"]["policies"][0]["actual_peak_boost_db"] = 12.1
    manifest["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "evidence_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="bounded"):
        validate_source_manifest_v2(manifest, campaign=campaign)


def test_complete_48_fixture_remains_structural_only_not_actual_pass() -> None:
    campaign = _campaign()
    candidates = [
        _candidate(slot, lossless=False)
        for slot in required_campaign_slots(campaign)
    ]
    result = validate_source_manifest_v2(
        _manifest(campaign, candidates), campaign=campaign
    )
    assert result["status"] == "STRUCTURAL_ONLY_NOT_PUBLISHABLE"
    assert result["structurally_valid_candidate_count"] == 48
    assert result["actual_verified_candidate_count"] == 0
    assert result["actual_acquisition_pass"] is False


def test_issuer_is_fail_closed_before_writing(tmp_path: Path) -> None:
    campaign = _campaign()
    candidates = [_candidate(slot) for slot in required_campaign_slots(campaign)]
    target = tmp_path / "issued.json"
    with pytest.raises(SourceContractV2Blocked, match="issuer authority"):
        issue_source_manifest_v2_noreplace(
            _manifest(campaign, candidates),
            campaign=campaign,
            repository_root=tmp_path,
            output_path=target,
        )
    assert not target.exists()
