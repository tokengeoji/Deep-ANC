from __future__ import annotations

import copy
import hashlib
import json

import pytest

from deep_anc.data.broadband_coverage_receipt import (
    BROADBAND_COVERAGE_RECEIPT_SCHEMA,
    BROADBAND_SOURCE_TRANSFORM_SCHEMA,
    NATIVE_BANDWIDTH_RESAMPLED_ONCE_ROLE,
    NATIVE_EXACT_TARGET_RATE_ROLE,
    build_broadband_coverage_policy,
    minimum_native_sample_rate_hz,
    seal_broadband_coverage_receipt,
    validate_broadband_coverage_receipt,
)
from deep_anc.dsp.control_band_contract import ControlBandContract


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _file(label: str):
    return {"path": f"evidence/{label}.bin", "size_bytes": 1, "sha256": _sha(label)}


def _canonical_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _reseal_transform(source):
    encoded = _canonical_bytes(source["transform_receipt"]["payload"])
    source["transform_receipt"]["file"] = {
        "path": f"evidence/transform-{hashlib.sha256(encoded).hexdigest()}.json",
        "size_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _source(label: str, *, native_rate: int = 48_000):
    raw_native = _file(f"source-native-{label}")
    if native_rate == 48_000:
        processed = copy.deepcopy(raw_native)
        role = NATIVE_EXACT_TARGET_RATE_ROLE
        count = 0
        resampler = None
    else:
        processed = _file(f"source-processed-{label}")
        role = NATIVE_BANDWIDTH_RESAMPLED_ONCE_ROLE
        count = 1
        resampler = {
            "algorithm": "polyphase_fir",
            "implementation": "test-resample-poly",
            "version": "1.0",
            "parameters_sha256": _sha(f"parameters-{label}"),
            "verified_passband_upper_hz": 11_400.0,
            "frequency_response_evidence": _file(f"response-{label}"),
        }
    source = {
        "raw_native_file": raw_native,
        "processed_file": processed,
        "native_sample_rate": native_rate,
        "native_nyquist_hz": native_rate / 2.0,
        "transform_receipt": {
            "file": {},
            "payload": {
                "schema": BROADBAND_SOURCE_TRANSFORM_SCHEMA,
                "processing_role": role,
                "raw_native_sha256": raw_native["sha256"],
                "processed_sha256": processed["sha256"],
                "input_sample_rate_hz": native_rate,
                "output_sample_rate_hz": 48_000,
                "resample_count": count,
                "native_bandwidth_coverage_verified": True,
                "synthetic_bandwidth_claimed": False,
                "lossless_native": True,
                "resampler": resampler,
            },
        },
    }
    _reseal_transform(source)
    return source


def _segments(*, passed: bool = True):
    coherence = 0.9 if passed else 0.2
    return [
        {
            "start_frame": index * 65_536,
            "n_frames": 65_536,
            "coherence": coherence,
            "target_density_ratio": 1.0,
        }
        for index in range(8)
    ]


def _band_row(band, *, passed: bool = True):
    segments = _segments(passed=passed)
    coherence_count = 8 if passed else 0
    return {
        "band_hz": list(band),
        "n_segments": 8,
        "coherence_pass_segments": coherence_count,
        "target_density_pass_segments": 8,
        "joint_pass_segments": coherence_count,
        "median_coherence": 0.9 if passed else 0.2,
        "median_target_density_ratio": 1.0,
        "segments": segments,
    }


def _summary(contract, sessions):
    rows = []
    blockers = []
    for split in ("train", "val", "test"):
        for family in contract.source_families:
            selected = [
                row for row in sessions if row["split"] == split and row["source_family"] == family
            ]
            for band_index, band in enumerate(contract.point_control_subbands_hz):
                groups = {
                    row["group_id"]
                    for row in selected
                    if row["bands"][band_index]["joint_pass_segments"] >= 8
                }
                passed = len(groups) >= 4
                rows.append(
                    {
                        "split": split,
                        "source_family": family,
                        "band_hz": list(band),
                        "qualifying_independent_groups": len(groups),
                        "passed": passed,
                    }
                )
                if not passed:
                    blockers.append(
                        f"{split}/{family}/{band[0]:.0f}-{band[1]:.0f}Hz groups {len(groups)} < 4"
                    )
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "session_count": len(sessions),
        "rows": rows,
        "blockers": blockers,
    }


def _receipt():
    contract = ControlBandContract.broadband_point_control()
    sessions = []
    for split in ("train", "val", "test"):
        for family in contract.source_families:
            for group_index in range(4):
                label = f"{split}-{family}-{group_index}"
                sessions.append(
                    {
                        "session_id": label,
                        "split": split,
                        "source_family": family,
                        "group_id": f"group-{label}",
                        "lineage_id": f"lineage-{label}",
                        "source": _source(label),
                        "mics": {
                            "file": _file(f"mics-{label}"),
                            "sample_rate": 48_000,
                        },
                        "alignment": {
                            "receipt_sha256": _sha(f"align-{label}"),
                            "method": "pilot_fractional_warp",
                            "subsample_alignment": True,
                            "clock_witness": True,
                            "timing_jitter_samples_by_subband": [0.0] * 7,
                        },
                        "bands": [
                            _band_row(band) for band in contract.point_control_subbands_hz
                        ],
                    }
                )
    payload = {
        "schema": BROADBAND_COVERAGE_RECEIPT_SCHEMA,
        "role": "campaign_readiness_not_diagnostic",
        "control_band_contract_sha256": contract.digest(),
        "policy": build_broadband_coverage_policy(),
        "manifest": _file("manifest"),
        "plant_evidence": _file("plant"),
        "training_timing_contract_sha256": _sha("timing"),
        "alignment_policy_sha256": _sha("alignment-policy"),
        "coverage_algorithm_sha256": _sha("coverage-algorithm"),
        "sessions": sessions,
        "summary": _summary(contract, sessions),
    }
    return contract, seal_broadband_coverage_receipt(payload)


def test_full_split_family_band_receipt_passes_structural_reaudit(tmp_path):
    contract, receipt = _receipt()
    result = validate_broadband_coverage_receipt(
        receipt,
        contract=contract,
        repository_root=tmp_path,
        require_local_files=False,
    )
    assert result["status"] == "STRUCTURAL_ONLY_NOT_CAMPAIGN_ELIGIBLE"
    assert result["campaign_readiness_eligible"] is False
    assert result["session_count"] == 48
    assert len(result["rows"]) == 3 * 4 * 7


def test_one_highband_group_failure_is_not_hidden_by_global_average(tmp_path):
    contract, receipt = _receipt()
    broken = copy.deepcopy(receipt)
    target = next(
        row
        for row in broken["sessions"]
        if row["split"] == "test" and row["source_family"] == "music"
    )
    target["bands"][-1] = _band_row(contract.point_control_subbands_hz[-1], passed=False)
    broken["summary"] = _summary(contract, broken["sessions"])
    broken = seal_broadband_coverage_receipt(broken)
    with pytest.raises(ValueError, match="BLOCKED"):
        validate_broadband_coverage_receipt(
            broken,
            contract=contract,
            repository_root=tmp_path,
            require_local_files=False,
        )


def test_native_rate_floor_is_derived_from_8khz_octave_upper():
    contract = ControlBandContract.broadband_point_control()
    assert minimum_native_sample_rate_hz(contract) == 22_628


@pytest.mark.parametrize("native_rate", [16_000, 22_050])
def test_native_source_below_octave_nyquist_cannot_claim_coverage(
    tmp_path, native_rate
):
    contract, receipt = _receipt()
    broken = copy.deepcopy(receipt)
    broken["sessions"][0]["source"] = _source(
        f"too-low-{native_rate}", native_rate=native_rate
    )
    broken = seal_broadband_coverage_receipt(broken)
    with pytest.raises(ValueError, match="native Nyquist"):
        validate_broadband_coverage_receipt(
            broken,
            contract=contract,
            repository_root=tmp_path,
            require_local_files=False,
        )


@pytest.mark.parametrize("native_rate", [24_000, 44_100, 48_000, 96_000])
def test_native_bandwidth_valid_rates_use_appropriate_transform(tmp_path, native_rate):
    contract, receipt = _receipt()
    receipt["sessions"][0]["source"] = _source(
        f"valid-{native_rate}", native_rate=native_rate
    )
    receipt = seal_broadband_coverage_receipt(receipt)
    result = validate_broadband_coverage_receipt(
        receipt,
        contract=contract,
        repository_root=tmp_path,
        require_local_files=False,
    )
    assert result["status"] == "STRUCTURAL_ONLY_NOT_CAMPAIGN_ELIGIBLE"
    assert result["campaign_readiness_eligible"] is False


def test_old_v2_fixture_is_not_silently_promoted(tmp_path):
    contract, receipt = _receipt()
    receipt["schema"] = "recorded_broadband_coverage_receipt_v2"
    receipt = seal_broadband_coverage_receipt(receipt)
    with pytest.raises(ValueError, match="schema"):
        validate_broadband_coverage_receipt(
            receipt,
            contract=contract,
            repository_root=tmp_path,
            require_local_files=False,
        )


def test_native_exact_role_cannot_disguise_44100_source(tmp_path):
    contract, receipt = _receipt()
    source = _source("disguised-44100", native_rate=44_100)
    source["processed_file"] = copy.deepcopy(source["raw_native_file"])
    payload = source["transform_receipt"]["payload"]
    payload["processing_role"] = NATIVE_EXACT_TARGET_RATE_ROLE
    payload["processed_sha256"] = payload["raw_native_sha256"]
    payload["resample_count"] = 0
    payload["resampler"] = None
    _reseal_transform(source)
    receipt["sessions"][0]["source"] = source
    receipt = seal_broadband_coverage_receipt(receipt)
    with pytest.raises(ValueError, match="exact-48k 역할"):
        validate_broadband_coverage_receipt(
            receipt,
            contract=contract,
            repository_root=tmp_path,
            require_local_files=False,
        )


def test_resampled_source_must_preserve_full_required_passband(tmp_path):
    contract, receipt = _receipt()
    source = _source("narrow-passband", native_rate=44_100)
    source["transform_receipt"]["payload"]["resampler"][
        "verified_passband_upper_hz"
    ] = 11_000.0
    _reseal_transform(source)
    receipt["sessions"][0]["source"] = source
    receipt = seal_broadband_coverage_receipt(receipt)
    with pytest.raises(ValueError, match="passband"):
        validate_broadband_coverage_receipt(
            receipt,
            contract=contract,
            repository_root=tmp_path,
            require_local_files=False,
        )


def test_transform_file_must_bind_exact_embedded_payload(tmp_path):
    contract, receipt = _receipt()
    source = receipt["sessions"][0]["source"]
    source["transform_receipt"]["payload"]["lossless_native"] = False
    receipt = seal_broadband_coverage_receipt(receipt)
    with pytest.raises(ValueError, match="lossless|canonical payload"):
        validate_broadband_coverage_receipt(
            receipt,
            contract=contract,
            repository_root=tmp_path,
            require_local_files=False,
        )


def test_same_native_source_sha_cannot_be_counted_as_independent_groups(tmp_path):
    contract, receipt = _receipt()
    broken = copy.deepcopy(receipt)
    broken["sessions"][1]["source"] = copy.deepcopy(
        broken["sessions"][0]["source"]
    )
    broken = seal_broadband_coverage_receipt(broken)
    with pytest.raises(ValueError, match="같은 native source SHA"):
        validate_broadband_coverage_receipt(
            broken,
            contract=contract,
            repository_root=tmp_path,
            require_local_files=False,
        )


def test_same_processed_source_sha_cannot_be_counted_as_independent_groups(tmp_path):
    contract, receipt = _receipt()
    first = _source("processed-alias-first", native_rate=44_100)
    second = _source("processed-alias-second", native_rate=44_100)
    second["processed_file"] = copy.deepcopy(first["processed_file"])
    second["transform_receipt"]["payload"]["processed_sha256"] = first[
        "processed_file"
    ]["sha256"]
    _reseal_transform(second)
    receipt["sessions"][0]["source"] = first
    receipt["sessions"][1]["source"] = second
    receipt = seal_broadband_coverage_receipt(receipt)
    with pytest.raises(ValueError, match="같은 processed source SHA"):
        validate_broadband_coverage_receipt(
            receipt,
            contract=contract,
            repository_root=tmp_path,
            require_local_files=False,
        )


def test_receipt_semantic_tamper_breaks_evidence_sha(tmp_path):
    contract, receipt = _receipt()
    receipt["sessions"][0]["bands"][0]["joint_pass_segments"] = 0
    with pytest.raises(ValueError, match="evidence SHA"):
        validate_broadband_coverage_receipt(
            receipt,
            contract=contract,
            repository_root=tmp_path,
            require_local_files=False,
        )
