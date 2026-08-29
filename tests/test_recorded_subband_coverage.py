"""recorded strict 부대역 coverage 사전계산 증거 회귀 테스트."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deep_anc.data.recorded_subband_coverage import (
    CANONICAL_MAX_SEGMENTS_PER_SESSION,
    RECORDED_SUBBAND_COVERAGE_SCHEMA_VERSION,
    seal_recorded_subband_coverage_report,
    validate_recorded_subband_coverage_report,
)
from deep_anc.dsp.timing import PlantDelays, TrainingTimingContract
from scripts.data import audit_recorded_subband_coverage as audit_script
from scripts.eval import evaluate_recorded as evaluate_script


FS = 8_000
FAMILIES = ("environment", "machine", "music", "speech")


def test_audit_and_official_evaluator_share_full_population_default():
    assert CANONICAL_MAX_SEGMENTS_PER_SESSION == 64
    assert (
        audit_script._parser().parse_args([]).max_segments_per_session
        == CANONICAL_MAX_SEGMENTS_PER_SESSION
    )
    assert (
        evaluate_script.build_parser()
        .parse_args(["--ckpt", "unused.pt"])
        .max_segments_per_session
        == CANONICAL_MAX_SEGMENTS_PER_SESSION
    )


def _fixture(tmp_path: Path) -> tuple[Path, dict, list[dict]]:
    timing = TrainingTimingContract.derive(
        primary_fir=np.asarray([1.0], dtype=np.float32),
        plant_delays=PlantDelays(
            primary_delay_samples=4,
            secondary_delay_samples=5,
            handoff_samples=2,
            sample_rate=FS,
        ),
    )
    cfg = {
        "model": {"hop": 4},
        "data": {
            "sample_rate": FS,
            "segment_seconds": 1.0,
            "digital_reference_lead_samples": int(
                timing.digital_reference_lead_samples
            ),
            "training_timing_contract": timing.model_dump(),
            "closed_loop": {
                "warmup_seconds": 0.0,
                "feedback_delay_samples": [0, 0],
            },
        },
    }
    entries = []
    for split in ("train", "val", "test"):
        for family in FAMILIES:
            for group_index in range(4):
                entries.append(
                    {
                        "path": str(tmp_path / f"{split}-{family}-{group_index}"),
                        "split": split,
                        "session_id": f"session-{split}-{family}-{group_index}",
                        "group_id": f"group-{split}-{family}-{group_index}",
                        "source_family": family,
                    }
                )
    manifest = tmp_path / "recorded.jsonl"
    manifest.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries),
        encoding="utf-8",
    )
    return manifest, cfg, entries


def _segments(entries, _data_cfg, **_kwargs):
    time = np.arange(FS, dtype=np.float64) / FS
    target = sum(
        np.sin(2.0 * np.pi * frequency * time)
        for frequency in (200.0, 400.0, 800.0, 1_200.0)
    ).astype(np.float32)
    for entry in entries:
        yield SimpleNamespace(
            d=target,
            source_family=entry["source_family"],
            group_id=entry["group_id"],
        )


def test_audit_script_generates_no_replace_report_and_recomputes_wav_exact(
    tmp_path, monkeypatch
):
    manifest, cfg, entries = _fixture(tmp_path)
    report_dir = tmp_path / "coverage"
    monkeypatch.setattr(audit_script, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(audit_script, "load_train_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        audit_script, "load_and_audit_recorded_manifest", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(audit_script, "read_manifest_bytes", lambda *_args, **_kwargs: entries)
    monkeypatch.setattr(audit_script, "iter_recorded_segments", _segments)

    assert audit_script.main(
        [
            "--manifest",
            str(manifest),
            "--canonical-out-dir",
            str(report_dir),
            "--edge-trim-seconds",
            "0",
        ]
    ) == 0
    reports = list(report_dir.glob("*.json"))
    assert len(reports) == 1
    report = reports[0]
    assert report.stem == json.loads(report.read_text())["coverage_contract_sha256"]
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["schema_version"] == RECORDED_SUBBAND_COVERAGE_SCHEMA_VERSION
    assert payload["segment_seconds"] == 1.0
    assert payload["segment_samples"] == FS
    assert payload["all_requested_splits_pass"] is True
    assert len(payload["evidence_sha256"]) == 64

    calls = {"count": 0}

    def counted_segments(*args, **kwargs):
        calls["count"] += 1
        yield from _segments(*args, **kwargs)

    monkeypatch.setattr(audit_script, "iter_recorded_segments", counted_segments)
    assert audit_script.main(
        [
            "--manifest",
            str(manifest),
            "--canonical-out-dir",
            str(report_dir),
            "--edge-trim-seconds",
            "0",
        ]
    ) == 0
    assert calls["count"] == 3
    monkeypatch.setattr(audit_script, "iter_recorded_segments", _segments)
    with pytest.raises(FileExistsError):
        audit_script.main(
            ["--manifest", str(manifest), "--out", str(report), "--edge-trim-seconds", "0"]
        )


def test_canonical_existing_forged_pass_cannot_bypass_fresh_weak_wav(
    tmp_path, monkeypatch
):
    manifest, cfg, entries = _fixture(tmp_path)
    report_dir = tmp_path / "coverage"
    monkeypatch.setattr(audit_script, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(audit_script, "load_train_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        audit_script, "load_and_audit_recorded_manifest", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(audit_script, "read_manifest_bytes", lambda *_args, **_kwargs: entries)
    monkeypatch.setattr(audit_script, "iter_recorded_segments", _segments)
    common = [
        "--manifest",
        str(manifest),
        "--canonical-out-dir",
        str(report_dir),
        "--edge-trim-seconds",
        "0",
    ]
    assert audit_script.main(common) == 0
    report = next(report_dir.glob("*.json"))
    forged_pass = json.loads(report.read_text(encoding="utf-8"))
    assert forged_pass["all_requested_splits_pass"] is True
    report.write_text(
        json.dumps(
            seal_recorded_subband_coverage_report(forged_pass),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    def weak_segments(selected, _data_cfg, **_kwargs):
        time = np.arange(FS, dtype=np.float64) / FS
        target = np.sin(2.0 * np.pi * 200.0 * time).astype(np.float32)
        for entry in selected:
            yield SimpleNamespace(
                d=target,
                source_family=entry["source_family"],
                group_id=entry["group_id"],
            )

    monkeypatch.setattr(audit_script, "iter_recorded_segments", weak_segments)
    with pytest.raises(ValueError, match="fresh 재계산 bytes"):
        audit_script.main(common)


def test_validator_rejects_stale_timing_and_resealed_forged_group_aggregate(
    tmp_path, monkeypatch
):
    manifest, cfg, entries = _fixture(tmp_path)
    report = tmp_path / "coverage.json"
    monkeypatch.setattr(audit_script, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(audit_script, "load_train_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        audit_script, "load_and_audit_recorded_manifest", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(audit_script, "read_manifest_bytes", lambda *_args, **_kwargs: entries)
    monkeypatch.setattr(audit_script, "iter_recorded_segments", _segments)
    assert audit_script.main(
        ["--manifest", str(manifest), "--out", str(report), "--edge-trim-seconds", "0"]
    ) == 0

    original = json.loads(report.read_text(encoding="utf-8"))
    mutations = (
        lambda payload: payload["manifest"].update(path=str(tmp_path / "other.jsonl")),
        lambda payload: payload["manifest"].update(size_bytes=payload["manifest"]["size_bytes"] + 1),
        lambda payload: payload["manifest"].update(sha256="0" * 64),
        lambda payload: payload.update(min_source_energy_density_ratio=0.20),
        lambda payload: payload.update(min_groups_per_family=3),
    )
    for mutate in mutations:
        forged = copy.deepcopy(original)
        mutate(forged)
        report.write_text(
            json.dumps(
                seal_recorded_subband_coverage_report(forged),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="불일치"):
            validate_recorded_subband_coverage_report(
                report,
                manifest_path=manifest,
                data_cfg=cfg["data"],
                model_hop=4,
                required_families=FAMILIES,
                configured_min_groups_per_family=4,
                edge_trim_seconds=0.0,
            )
    report.write_text(
        json.dumps(original, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="group 하한"):
        validate_recorded_subband_coverage_report(
            report,
            manifest_path=manifest,
            data_cfg=cfg["data"],
            model_hop=4,
            required_families=FAMILIES,
            configured_min_groups_per_family=3,
            edge_trim_seconds=0.0,
        )

    stale_cfg = json.loads(json.dumps(cfg))
    stale_timing = TrainingTimingContract.derive(
        primary_fir=np.asarray([1.0], dtype=np.float32),
        plant_delays=PlantDelays(
            primary_delay_samples=3,
            secondary_delay_samples=5,
            handoff_samples=2,
            sample_rate=FS,
        ),
    )
    stale_cfg["data"]["training_timing_contract"] = stale_timing.model_dump()
    stale_cfg["data"]["digital_reference_lead_samples"] = int(
        stale_timing.digital_reference_lead_samples
    )
    with pytest.raises(ValueError, match="불일치"):
        validate_recorded_subband_coverage_report(
            report,
            manifest_path=manifest,
            data_cfg=stale_cfg["data"],
            model_hop=4,
            required_families=FAMILIES,
            configured_min_groups_per_family=4,
            edge_trim_seconds=0.0,
        )

    stale_segment_cfg = json.loads(json.dumps(cfg))
    stale_segment_cfg["data"]["segment_seconds"] = 0.5
    with pytest.raises(ValueError, match="segment_seconds|segment_samples|불일치"):
        validate_recorded_subband_coverage_report(
            report,
            manifest_path=manifest,
            data_cfg=stale_segment_cfg["data"],
            model_hop=4,
            required_families=FAMILIES,
            configured_min_groups_per_family=4,
            edge_trim_seconds=0.0,
        )

    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["splits"]["val"]["rows"][0]["n_covered_groups"] = 3
    payload["splits"]["val"]["rows"][0]["group_power_pass"] = True
    report.write_text(
        json.dumps(
            seal_recorded_subband_coverage_report(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="covered group 수|위조"):
        validate_recorded_subband_coverage_report(
            report,
            manifest_path=manifest,
            data_cfg=cfg["data"],
            model_hop=4,
            required_families=FAMILIES,
            configured_min_groups_per_family=4,
            edge_trim_seconds=0.0,
        )
