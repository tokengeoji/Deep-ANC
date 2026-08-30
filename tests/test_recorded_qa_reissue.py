"""Historical recorded QA가 current regrouped/strict timing evidence로 승격되지 않는지."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from deep_anc.data.recorded_qa_reissue import (
    CANONICAL_RECORDED_MANIFEST,
    RECORDED_QA_REISSUE_SCHEMA,
    RecordedQAReissueError,
    build_current_recorded_qa_provenance,
    render_current_recorded_qa_markdown,
    validate_current_recorded_qa_report,
)


def _write_npz(path: Path, *, delay: int, peak: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fir = np.zeros(32, dtype=np.float32)
    fir[peak] = 1.0
    np.savez(
        path,
        fir=fir,
        delay_samples=np.asarray(delay, dtype=np.int64),
        sample_rate=np.asarray(48_000, dtype=np.int64),
        excitation_band_hz=np.asarray([64.0, 1648.0]),
        consistency_band_hz=np.asarray([150.0, 1600.0]),
    )


def _current_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = tmp_path / CANONICAL_RECORDED_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "path": "../recorded/session-a",
                "path_base": "manifest",
                "duration_s": 1.5,
                "sample_rate": 48_000,
                "channels": 2,
                "tag": "recorded",
                "session_id": "session-a",
                "group_id": "component-a",
                "source_family": "speech",
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    primary = tmp_path / "assets/measured/primary.npz"
    secondary = tmp_path / "assets/measured/secondary.npz"
    # S 1245 + handoff 256 - P 1386 = current strict lead 115.
    _write_npz(primary, delay=1386, peak=4)
    _write_npz(secondary, delay=1245, peak=2)
    data = tmp_path / "configs/data.yaml"
    duct = tmp_path / "configs/duct.yaml"
    data.parent.mkdir(parents=True)
    data.write_text(
        yaml.safe_dump(
            {
                "sample_rate": 48_000,
                "segment_seconds": 1.5,
                "reference_mode": "digital",
                # Pretrain default must not decide current recorded QA timing.
                "digital_primary_path_mode": "secondary_surrogate",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    duct.write_text(
        yaml.safe_dump(
            {
                "secondary_path": {
                    "npz": "assets/measured/secondary.npz",
                    "handoff_extra_samples": 256,
                },
                "digital_reference": {
                    "primary_path_npz": "assets/measured/primary.npz"
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest, data, duct


def _valid_report(provenance: dict, *, root: Path) -> dict:
    contract = provenance["timing"]["training_timing_contract"]
    lead = int(contract["digital_reference_lead_samples"])
    return {
        "ok": True,
        "manifest": str(root / CANONICAL_RECORDED_MANIFEST),
        "settings": {
            "reference_mode": "digital",
            "sample_rate": 48_000,
            "segment_samples": 72_000,
            "digital_reference_lead_samples": lead,
            "minimum_frames": 72_000 + lead + 1,
        },
        "summary": {"sessions": 82, "valid_sessions": 82},
        "provenance": provenance,
    }


def test_binding_uses_canonical_manifest_and_plant_derived_lead(tmp_path: Path):
    manifest, data, duct = _current_tree(tmp_path)

    provenance = build_current_recorded_qa_provenance(
        repo_root=tmp_path,
        manifest_path=manifest,
        data_config_path=data,
        duct_config_path=duct,
    )

    assert provenance["schema"] == RECORDED_QA_REISSUE_SCHEMA
    assert provenance["manifest"]["path"] == CANONICAL_RECORDED_MANIFEST
    contract = provenance["timing"]["training_timing_contract"]
    assert contract["digital_reference_lead_samples"] == 115
    assert provenance["timing"]["plant_delays_lead_samples"] == 115
    assert provenance["inputs"]["primary_path_mode"] == "measured"
    assert provenance["timing"]["training_timing_contract_sha256"]


def test_historical_report_without_binding_is_rejected(tmp_path: Path):
    manifest, data, duct = _current_tree(tmp_path)
    expected = build_current_recorded_qa_provenance(
        repo_root=tmp_path,
        manifest_path=manifest,
        data_config_path=data,
        duct_config_path=duct,
    )
    historical = {
        "ok": True,
        "manifest": "data/manifests/recorded_train.jsonl",
        "settings": {"digital_reference_lead_samples": 116},
    }

    with pytest.raises(RecordedQAReissueError, match="provenance"):
        validate_current_recorded_qa_report(historical, expected_provenance=expected)


def test_binding_rejects_changed_manifest_or_lead(tmp_path: Path):
    manifest, data, duct = _current_tree(tmp_path)
    expected = build_current_recorded_qa_provenance(
        repo_root=tmp_path,
        manifest_path=manifest,
        data_config_path=data,
        duct_config_path=duct,
    )
    report = _valid_report(expected, root=tmp_path)
    validate_current_recorded_qa_report(
        report, expected_provenance=expected, repo_root=tmp_path
    )

    stale_manifest = _valid_report(expected, root=tmp_path)
    stale_manifest["provenance"] = dict(expected)
    stale_manifest["provenance"]["manifest"] = dict(expected["manifest"])
    stale_manifest["provenance"]["manifest"]["sha256"] = "0" * 64
    with pytest.raises(RecordedQAReissueError, match="provenance"):
        validate_current_recorded_qa_report(
            stale_manifest, expected_provenance=expected, repo_root=tmp_path
        )

    misleading_visible_manifest = _valid_report(expected, root=tmp_path)
    misleading_visible_manifest["manifest"] = str(
        tmp_path / "data/manifests/recorded_train.jsonl"
    )
    with pytest.raises(RecordedQAReissueError, match="visible manifest"):
        validate_current_recorded_qa_report(
            misleading_visible_manifest, expected_provenance=expected, repo_root=tmp_path
        )

    stale_lead = _valid_report(expected, root=tmp_path)
    stale_lead["settings"] = dict(stale_lead["settings"])
    stale_lead["settings"]["digital_reference_lead_samples"] = 116
    stale_lead["settings"]["minimum_frames"] += 1
    with pytest.raises(RecordedQAReissueError, match="lead"):
        validate_current_recorded_qa_report(
            stale_lead, expected_provenance=expected, repo_root=tmp_path
        )


def test_noncanonical_manifest_cannot_be_reissued(tmp_path: Path):
    _, data, duct = _current_tree(tmp_path)
    old_manifest = tmp_path / "data/manifests/recorded_train.jsonl"
    old_manifest.write_text("", encoding="utf-8")

    with pytest.raises(RecordedQAReissueError, match="recorded_regrouped"):
        build_current_recorded_qa_provenance(
            repo_root=tmp_path,
            manifest_path=old_manifest,
            data_config_path=data,
            duct_config_path=duct,
        )


def test_rendered_reissue_report_exposes_authority_binding(tmp_path: Path):
    manifest, data, duct = _current_tree(tmp_path)
    provenance = build_current_recorded_qa_provenance(
        repo_root=tmp_path,
        manifest_path=manifest,
        data_config_path=data,
        duct_config_path=duct,
    )
    report = _valid_report(provenance, root=tmp_path)
    report["summary"].update(
        {"duration_s": 0.0, "splits": {}, "source_families": {}}
    )
    rendered = render_current_recorded_qa_markdown(report)
    assert "Current authority binding" in rendered
    assert "strict P/S-derived lead: 115 samples" in rendered
