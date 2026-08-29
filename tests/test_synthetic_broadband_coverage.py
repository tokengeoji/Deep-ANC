from __future__ import annotations

import copy

import pytest

from deep_anc.data.synthetic_broadband_coverage import (
    DEFAULT_FAMILY_TAGS,
    audit_synthetic_native_manifest_rows,
)
from deep_anc.dsp.control_band_contract import ControlBandContract


def _rows(*, machine_rate: int = 48_000):
    entries = {
        tag: [] for tags in DEFAULT_FAMILY_TAGS.values() for tag in tags
    }
    family_primary_tag = {
        "speech": "speech",
        "music": "music",
        "environment": "dns_fullband",
        "machine": "machine",
    }
    serial = 0
    for split in ("train", "val", "test"):
        for family, tag in family_primary_tag.items():
            for group_index in range(4):
                serial += 1
                entries[tag].append(
                    {
                        "path": f"{tag}/{split}/{group_index}.wav",
                        "tag": tag,
                        "split": split,
                        "group_id": f"{family}-{split}-{group_index}",
                        "content_sha256": f"{serial:064x}",
                        "sample_rate": machine_rate if family == "machine" else 48_000,
                    }
                )
    return entries


def test_native_manifest_gate_passes_only_with_every_family_split_and_band():
    result = audit_synthetic_native_manifest_rows(
        _rows(), contract=ControlBandContract.broadband_point_control()
    )
    assert result["status"] == "PASS"
    assert result["reasons"] == []
    assert len(result["cells"]) == 3 * 4 * 7
    assert all(row["passed"] for row in result["cells"])


def test_16khz_machine_cannot_claim_complete_8khz_octave_after_upsampling():
    result = audit_synthetic_native_manifest_rows(
        _rows(machine_rate=16_000),
        contract=ControlBandContract.broadband_point_control(),
    )
    assert result["status"] == "BLOCKED"
    failures = [
        row
        for row in result["cells"]
        if row["source_family"] == "machine" and not row["passed"]
    ]
    assert failures
    assert all(row["band_hz"][1] > 8_000.0 for row in failures)
    assert {row["split"] for row in failures} == {"train", "val", "test"}


def test_same_lineage_group_cannot_cross_split():
    entries = _rows()
    entries["speech"][4]["group_id"] = entries["speech"][0]["group_id"]
    result = audit_synthetic_native_manifest_rows(
        entries, contract=ControlBandContract.broadband_point_control()
    )
    assert result["status"] == "BLOCKED"
    assert any("split/family" in reason for reason in result["reasons"])


def test_same_content_cannot_be_split_into_multiple_independent_groups():
    entries = _rows()
    entries["music"][4]["content_sha256"] = entries["music"][0][
        "content_sha256"
    ]
    result = audit_synthetic_native_manifest_rows(
        entries, contract=ControlBandContract.broadband_point_control()
    )
    assert result["status"] == "BLOCKED"
    assert any("content SHA" in reason for reason in result["reasons"])


def test_missing_declared_tag_is_fail_closed_even_if_other_tag_has_enough_rows():
    entries = _rows()
    del entries["demand"]
    result = audit_synthetic_native_manifest_rows(
        entries, contract=ControlBandContract.broadband_point_control()
    )
    assert result["status"] == "BLOCKED"
    assert "public manifest tag가 없습니다: demand" in result["reasons"]


def test_builtin_synthetic_generator_cannot_be_counted_as_public_lineage():
    mapping = copy.deepcopy(DEFAULT_FAMILY_TAGS)
    mapping["machine"] = ("machine", "synthetic")
    with pytest.raises(ValueError, match="independent public group"):
        audit_synthetic_native_manifest_rows(
            _rows(),
            contract=ControlBandContract.broadband_point_control(),
            family_tags=mapping,
        )


def test_minimum_group_floor_cannot_be_weakened():
    with pytest.raises(ValueError, match="4"):
        audit_synthetic_native_manifest_rows(
            _rows(),
            contract=ControlBandContract.broadband_point_control(),
            minimum_groups=3,
        )
