"""Parent 82 + 별도 additions 17 recorded generation 계약."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import numpy as np
import soundfile as sf

import deep_anc.data.recorded_generation as generation
import deep_anc.data.transfer_contract as transfer_contract
from deep_anc import config as config_module
from deep_anc.data.holdout_contract import snapshot_regular_tree_metadata
from deep_anc.realtime.noise_gen import NoiseProgram, render_recording_file_window


GENERATION_ID = "highband-v1"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_progress_seconds(path: Path, *, row_number: int, seconds: float) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    for row in rows:
        if int(row["source_row_number"]) == row_number:
            row["seconds"] = str(seconds)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_recorded_session_artifacts(
    session: Path, *, seconds: float, metadata: dict, source_path: Path
) -> None:
    frames = int(round(seconds * 48_000))
    program = metadata["program"]
    program_instance = NoiseProgram(
        program,
        48_000,
        file_bytes=source_path.read_bytes(),
    )
    expected_source = render_recording_file_window(
        program_instance,
        frames,
        sample_rate=48_000,
    )
    session.mkdir(parents=True, exist_ok=True)
    sf.write(
        session / "mics.wav",
        np.column_stack((expected_source * 0.5, expected_source)).astype(np.float32),
        48_000,
        subtype="PCM_32",
    )
    sf.write(
        session / "source.wav",
        expected_source,
        48_000,
        subtype="FLOAT",
    )
    sf.write(
        session / "source_aligned.wav",
        expected_source,
        48_000,
        subtype="FLOAT",
    )
    metadata["artifacts"] = [
        {
            "path": name,
            "size_bytes": (session / name).stat().st_size,
            "sha256": _sha((session / name).read_bytes()),
        }
        for name in ("mics.wav", "source.wav", "source_aligned.wav")
    ]
    _write_json(session / "session.json", metadata)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    root = tmp_path
    parent_root = root / generation.PARENT_ROOT
    parent_manifest = root / generation.PARENT_MANIFEST
    parent_manifest.parent.mkdir(parents=True, exist_ok=True)
    families = ("speech", "music", "environment", "machine")
    monkeypatch.setattr(
        generation,
        "EXPECTED_ADDITION_FAMILY_KIND_COUNTS",
        {(family, generation.SOURCE_KIND_POOL): (5 if family == "speech" else 4)
         for family in families},
    )
    parent_rows = []
    for index in range(generation.PARENT_SESSION_COUNT):
        session_id = f"parent-{index:03d}"
        session = parent_root / session_id
        _write_json(session / "session.json", {"session_id": session_id})
        (session / "mics.wav").write_bytes(f"parent-{index}".encode())
        family = families[index % len(families)]
        parent_rows.append(
            {
                "path": f"../recorded/{session_id}",
                "path_base": "manifest",
                "duration_s": 70.0,
                "sample_rate": 48_000,
                "channels": 2,
                "tag": "recorded",
                "session_id": session_id,
                "group_id": f"parent-lineage-{index:03d}",
                "source_family": family,
                "metadata_inferred": [],
                "source_pool_group_id": f"parent-source-{index:03d}",
                "lineage_schema": "parent/v2",
                "split": ("train", "val", "test")[index % 3],
            }
        )
    parent_manifest.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in parent_rows),
        encoding="utf-8",
    )
    holdout_path = root / generation.PARENT_HOLDOUT
    _write_json(holdout_path, {"fixture": "immutable-parent-holdout"})
    provenance = root / "results/provenance/parent.json"
    _write_json(provenance, {"fixture": "immutable-parent-provenance"})
    lineage_files = {
        "tracks": root / "data/raw/music/fma_metadata/tracks.csv",
        "chapters": root / "data/raw/speech/LibriSpeech/CHAPTERS.TXT",
        "esc50": root / "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv",
        "source_pool": root / "data/source_pool/sources.csv",
        "source_pool_v2": root / "data/source_pool_v2/sources.csv",
    }
    for name, path in lineage_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-{name}\n".encode())
    parent_tree = snapshot_regular_tree_metadata(
        parent_root, repo_root=root, label="fixture parent tree"
    )
    holdout_summary = {
        "sha256": _sha(holdout_path.read_bytes()),
        "lineage": {
            "regrouped_manifest": generation.PARENT_MANIFEST,
            "regrouped_manifest_sha256": _sha(parent_manifest.read_bytes()),
            "regrouped_row_count": generation.PARENT_SESSION_COUNT,
            "tracks_csv_sha256": _sha(lineage_files["tracks"].read_bytes()),
            "librispeech_chapters_path": lineage_files["chapters"].relative_to(root).as_posix(),
            "librispeech_chapters_sha256": _sha(lineage_files["chapters"].read_bytes()),
            "esc50_metadata_path": lineage_files["esc50"].relative_to(root).as_posix(),
            "esc50_metadata_sha256": _sha(lineage_files["esc50"].read_bytes()),
        },
        "recorded_tree": {
            "file_count": parent_tree.file_count,
            "metadata_snapshot_sha256": parent_tree.sha256,
            "content_snapshot_sha256": parent_tree.content_sha256,
        },
        "provenance_report": provenance.relative_to(root).as_posix(),
        "provenance_report_sha256": _sha(provenance.read_bytes()),
        "sources_csv": [
            lineage_files["source_pool"].relative_to(root).as_posix(),
            lineage_files["source_pool_v2"].relative_to(root).as_posix(),
        ],
        "sources_csv_sha256": {
            "source_pool": _sha(lineage_files["source_pool"].read_bytes()),
            "source_pool_v2": _sha(lineage_files["source_pool_v2"].read_bytes()),
        },
    }

    def validate_holdout(path, *, repo_root, expected_sha256=None):
        assert Path(path) == root / generation.PARENT_HOLDOUT
        assert Path(repo_root) == root
        if expected_sha256 is not None and expected_sha256 != holdout_summary["sha256"]:
            raise AssertionError("fixture expected holdout SHA mismatch")
        return json.loads(json.dumps(holdout_summary))

    monkeypatch.setattr(generation, "validate_holdout_contract", validate_holdout)

    plan_path = root / generation.SOURCE_PLAN_ROOT / f"{GENERATION_ID}.csv"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_rows = []
    planned_splits = {
        "speech": iter(("train", "train", "val", "test", "test")),
        "music": iter(("val", "val", "test", "test")),
        "environment": iter(("val", "test", "test", "test")),
        "machine": iter(("train", "val", "test", "test")),
    }
    for index in range(generation.ADDITION_SESSION_COUNT):
        source = root / "data/generation_sources" / f"source-{index:02d}.wav"
        source.parent.mkdir(parents=True, exist_ok=True)
        source_time = np.arange(2400, dtype=np.float64) / 48_000.0
        sf.write(
            source,
            (0.2 * np.sin(2.0 * np.pi * (300.0 + index) * source_time)).astype(
                np.float32
            ),
            48_000,
            subtype="FLOAT",
        )
        family = families[index % len(families)]
        plan_rows.append(
            {
                "source_kind": generation.SOURCE_KIND_POOL,
                "path": source.relative_to(root).as_posix(),
                "seconds": "0.01",
                "start_seconds": "0.0",
                "source_family": family,
                "group_id": f"addition-source-{index:02d}",
                "lineage_key": f"addition-lineage-{index:02d}",
                "split": next(planned_splits[family]),
                "source_file_sha256": _sha(source.read_bytes()),
                "raw_member_path": "",
                "raw_member_sha256": "",
                "raw_member_lineage_key": "",
                "transform": "identity",
                "transform_repeat_count": "1",
            }
        )
    monkeypatch.setattr(
        generation,
        "CANONICAL_ADDITION_SECONDS_BY_KIND",
        {
            generation.SOURCE_KIND_POOL: 0.01,
            generation.SOURCE_KIND_EXTERNAL: 15.0,
            generation.SOURCE_KIND_EXTERNAL_LIBRISPEECH: 15.0,
        },
    )
    monkeypatch.setattr(
        generation,
        "CANONICAL_SOURCE_POOL_ADDITIONS",
        {
            row["path"]: (
                row["source_family"],
                float(row["start_seconds"]),
                row["split"],
            )
            for row in plan_rows
        },
    )
    with plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=generation.SOURCE_PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(plan_rows)
    plan_sha = _sha(plan_path.read_bytes())
    canonical_rows = {
        row["path"]: {
            "source_family": row["source_family"],
            "group_id": row["group_id"],
            "clips": [f"fixture-{index}"],
        }
        for index, row in enumerate(plan_rows)
    }
    component_by_path = {
        row["path"]: row["lineage_key"] for row in plan_rows
    }
    monkeypatch.setattr(
        generation,
        "_canonical_source_lineage",
        lambda _root: {
            "rows": canonical_rows,
            "component_by_path": component_by_path,
            "active_components": set(),
            "authority_tokens_by_path": {
                row["path"]: {f"source_pool_component:{row['lineage_key']}"}
                for row in plan_rows
            },
            "evidence_sha256": "e" * 64,
        },
    )
    monkeypatch.setattr(
        generation,
        "_canonical_source_selection_evidence",
        lambda _root, _lineage: {
            "schema": generation.SOURCE_SELECTION_CONTRACT_SCHEMA,
            "fixture": True,
            "evidence_sha256": "s" * 64,
        },
    )

    additions_root = root / generation.ADDITIONS_ROOT / GENERATION_ID
    progress_rows = []
    for index, row in enumerate(plan_rows, start=2):
        session_id = f"addition-{index - 2:02d}"
        session = additions_root / session_id
        metadata = {
            "session_id": session_id,
            "program": {
                "type": "file",
                "frequency": 300.0,
                "amplitude": 0.1,
                "band": [80.0, 1000.0],
                "file": row["path"],
                "file_start_seconds": float(row["start_seconds"]),
            },
            "source_family": row["source_family"],
            "group_id": row["group_id"],
            "seconds": 0.01,
            "sample_rate": 48_000,
            "block_size": 256,
            "channels": {
                "err_mic": 0,
                "ref_mic": 1,
                "noise_out": 0,
                "cancel_out": 1,
            },
            "safety_confirmations": {
                "user_present": True,
                "volume_minimum": True,
                "routing_and_geometry": True,
            },
            "timeline": {
                "method": "ref_witness_warp_v1",
                "witness_channel": 1,
                "usable_for_digital_reference": True,
                "valid_window_ratio": 0.99,
                "aligned_lag_median_samples": 143.0,
                "aligned_lag_robust_std_samples": 1.0,
                "coh2_150_600_after": 0.95,
            },
            "preassigned_split": row["split"],
            "collection_plan": {
                "status": "exact",
                "source_list": plan_path.relative_to(root).as_posix(),
                "source_list_sha256": plan_sha,
                "source_row_number": index,
                "lineage_key": row["lineage_key"],
                "preassigned_split": row["split"],
                "split_source": "csv",
                "source_file_sha256": row["source_file_sha256"],
                "start_seconds": float(row["start_seconds"]),
            },
        }
        _write_recorded_session_artifacts(
            session,
            seconds=0.01,
            metadata=metadata,
            source_path=root / row["path"],
        )
        progress_rows.append(
            {
                "source_row_number": str(index),
                "session_id": session_id,
                "seconds": "0.01",
                "verdict": "ok",
            }
        )
    progress_path = additions_root / "batch_progress.csv"
    with progress_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source_row_number", "session_id", "seconds", "verdict"),
        )
        writer.writeheader()
        writer.writerows(progress_rows)

    directory = root / generation.GENERATION_ROOT / GENERATION_ID
    directory.mkdir(parents=True, exist_ok=True)
    combined = generation.build_combined_manifest_bytes(
        repo_root=root,
        generation_id=GENERATION_ID,
        expected_holdout_sha256=holdout_summary["sha256"],
    )
    (directory / "recorded.jsonl").write_bytes(combined)
    payload = generation.build_recorded_generation_payload(
        repo_root=root,
        generation_id=GENERATION_ID,
        expected_holdout_sha256=holdout_summary["sha256"],
    )
    report = directory / "generation.json"
    report.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report, holdout_summary


def _schema_v2_transfer_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    report: Path,
    holdout: dict,
) -> tuple[Path, object]:
    """실제 transfer builder/validator가 소비하는 generation-99 bundle을 만든다."""

    fixed_files = {
        "data/rir_bank/duct_rirs_v1.npz": b"rir",
        "results/strict_ps/raw.npz": b"raw",
        "results/strict_ps/analysis.json": b"analysis",
        "assets/measured/primary.npz": b"primary",
        "assets/measured/secondary.npz": b"secondary",
    }
    for relative, raw in fixed_files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    spec = importlib.util.spec_from_file_location(
        "recorded_generation_transfer_builder_test",
        Path(__file__).resolve().parents[1]
        / "scripts/data/build_elice_transfer_manifest.py",
    )
    builder = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)
    monkeypatch.setattr(builder, "REPO_ROOT", root)
    monkeypatch.setattr(
        builder,
        "validate_holdout_contract",
        lambda _path, *, repo_root, expected_sha256=None: json.loads(
            json.dumps(holdout)
        ),
    )
    monkeypatch.setattr(
        transfer_contract,
        "validate_holdout_contract",
        lambda _path, *, repo_root, expected_sha256=None: json.loads(
            json.dumps(holdout)
        ),
    )
    args = Namespace(
        recorded_root="data/recorded",
        rir_bank="data/rir_bank/duct_rirs_v1.npz",
        strict_raw=["results/strict_ps/raw.npz"],
        strict_analysis=["results/strict_ps/analysis.json"],
        primary_npz="assets/measured/primary.npz",
        secondary_npz="assets/measured/secondary.npz",
        expected_holdout_sha256=holdout["sha256"],
        recorded_manifest=generation.PARENT_MANIFEST,
        recorded_generation=report.relative_to(root).as_posix(),
        allow_missing_generation_source_files=False,
        lineage_tracks="data/raw/music/fma_metadata/tracks.csv",
        librispeech_chapters_metadata="data/raw/speech/LibriSpeech/CHAPTERS.TXT",
        esc50_metadata="data/raw/noise/esc50/ESC-50-master/meta/esc50.csv",
        out=builder.OUTPUT,
    )
    payload = builder.build_payload(args, repo_root=root)
    transfer = root / builder.OUTPUT
    transfer.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return transfer, builder


def test_generation_preserves_parent82_and_binds_exact_additions17(tmp_path, monkeypatch):
    report, _holdout = _fixture(tmp_path, monkeypatch)
    summary = generation.validate_recorded_generation(
        report,
        repo_root=tmp_path,
        expected_sha256=_sha(report.read_bytes()),
    )
    assert summary["parent_session_count"] == 82
    assert summary["addition_session_count"] == 17
    assert summary["recorded_session_count"] == 99
    assert summary["recorded_manifest"]["path"].endswith("/recorded.jsonl")


def test_canonical_config_accepts_only_matching_generation_manifest_path_and_sha():
    cfg = {
        "recorded_manifest": (
            "data/manifests/recorded_generations/highband-v1/recorded.jsonl"
        ),
        "data": {
            "recorded_generation": (
                "data/manifests/recorded_generations/highband-v1/generation.json"
            ),
            "recorded_generation_sha256": "a" * 64,
        },
    }
    assert config_module._canonical_recorded_manifest_binding(cfg)
    assert config_module.canonical_recorded_manifest_for_data(cfg["data"]) == (
        "data/manifests/recorded_generations/highband-v1/recorded.jsonl"
    )
    cfg["data"]["recorded_generation"] = (
        "data/manifests/recorded_generations/other-v1/generation.json"
    )
    assert not config_module._canonical_recorded_manifest_binding(cfg)

    with pytest.raises(ValueError, match="함께 선언"):
        config_module.canonical_recorded_manifest_for_data(
            {
                "recorded_generation": (
                    "data/manifests/recorded_generations/highband-v1/generation.json"
                )
            }
        )
    assert config_module.canonical_recorded_manifest_for_data({}) == (
        config_module.CANONICAL_RECORDED_MANIFEST
    )


@pytest.mark.parametrize("role", ["measured_probe", "canonical_finetune"])
def test_bound_schema2_materializes_dynamic_recorded_manifest(role, monkeypatch):
    calls = []
    monkeypatch.setattr(
        config_module,
        "_enforce_pretrain_derivative_policy",
        lambda cfg: calls.append(cfg["recorded_manifest"]),
    )
    monkeypatch.setattr(
        config_module,
        "_enforce_canonical_finetune_policy",
        lambda cfg: calls.append(cfg["recorded_manifest"]),
    )
    cfg = {
        "experiment_role": role,
        "recorded_manifest": config_module.CANONICAL_RECORDED_MANIFEST,
        "data": {
            "recorded_generation": (
                "data/manifests/recorded_generations/highband-coverage-v1/generation.json"
            ),
            "recorded_generation_sha256": "a" * 64,
        },
    }
    config_module._materialize_bound_recorded_manifest(cfg)
    expected = "data/manifests/recorded_generations/highband-coverage-v1/recorded.jsonl"
    assert cfg["recorded_manifest"] == expected
    assert calls == [expected]
    cfg["recorded_manifest"] = "data/manifests/forged.jsonl"
    with pytest.raises(ValueError, match="검증된 transfer generation"):
        config_module._materialize_bound_recorded_manifest(cfg)
    cfg["data"]["recorded_generation"] = (
        "data/manifests/recorded_generations/-bad/generation.json"
    )
    cfg["recorded_manifest"] = (
        "data/manifests/recorded_generations/-bad/recorded.jsonl"
    )
    assert not config_module._canonical_recorded_manifest_binding(cfg)


def test_generation_rejects_changed_parent_session(tmp_path, monkeypatch):
    report, _holdout = _fixture(tmp_path, monkeypatch)
    target = tmp_path / generation.PARENT_ROOT / "parent-000/mics.wav"
    target.write_bytes(b"replaced-parent")
    with pytest.raises(generation.RecordedGenerationError, match="parent 82 recorded tree"):
        generation.validate_recorded_generation(report, repo_root=tmp_path)


def test_generation_rejects_missing_or_replaced_addition(tmp_path, monkeypatch):
    report, _holdout = _fixture(tmp_path, monkeypatch)
    target = tmp_path / generation.ADDITIONS_ROOT / GENERATION_ID / "addition-00/mics.wav"
    target.unlink()
    with pytest.raises(generation.RecordedGenerationError, match="artifact exact"):
        generation.validate_recorded_generation(report, repo_root=tmp_path)


def test_generation_rejects_source_plan_or_session_provenance_forgery(tmp_path, monkeypatch):
    report, _holdout = _fixture(tmp_path, monkeypatch)
    session_json = (
        tmp_path / generation.ADDITIONS_ROOT / GENERATION_ID / "addition-00/session.json"
    )
    metadata = json.loads(session_json.read_text(encoding="utf-8"))
    metadata["collection_plan"]["preassigned_split"] = "test"
    _write_json(session_json, metadata)
    with pytest.raises(generation.RecordedGenerationError, match="collection_plan/CSV"):
        generation.validate_recorded_generation(report, repo_root=tmp_path)


def test_generation_rejects_valid_wav_resealed_with_wrong_playback(tmp_path, monkeypatch):
    report, _holdout = _fixture(tmp_path, monkeypatch)
    session = tmp_path / generation.ADDITIONS_ROOT / GENERATION_ID / "addition-00"
    source_wav = session / "source.wav"
    replacement = (0.05 * np.cos(2.0 * np.pi * 700.0 * np.arange(480) / 48_000.0)).astype(
        np.float32
    )
    sf.write(source_wav, replacement, 48_000, subtype="FLOAT")
    metadata_path = session / "session.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifact = next(item for item in metadata["artifacts"] if item["path"] == "source.wav")
    artifact["size_bytes"] = source_wav.stat().st_size
    artifact["sha256"] = _sha(source_wav.read_bytes())
    _write_json(metadata_path, metadata)
    with pytest.raises(generation.RecordedGenerationError, match=r"source\.wav.*재유도"):
        generation.validate_recorded_generation(report, repo_root=tmp_path)


def test_generation_rejects_program_start_self_statement(tmp_path, monkeypatch):
    report, _holdout = _fixture(tmp_path, monkeypatch)
    metadata_path = (
        tmp_path / generation.ADDITIONS_ROOT / GENERATION_ID / "addition-00/session.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["program"]["file_start_seconds"] = 0.001
    _write_json(metadata_path, metadata)
    with pytest.raises(generation.RecordedGenerationError, match="top-level metadata"):
        generation.validate_recorded_generation(report, repo_root=tmp_path)


def test_generation_rejects_wrong_family_split_coverage_matrix(tmp_path, monkeypatch):
    report, _holdout = _fixture(tmp_path, monkeypatch)
    plan = tmp_path / generation.SOURCE_PLAN_ROOT / f"{GENERATION_ID}.csv"
    with plan.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["source_family"] == "speech" and rows[0]["split"] == "train"
    rows[0]["split"] = "val"
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=generation.SOURCE_PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(generation.RecordedGenerationError, match="split"):
        generation.validate_recorded_generation(report, repo_root=tmp_path)


def test_generation_rejects_combined_manifest_parent_rewrite(tmp_path, monkeypatch):
    report, _holdout = _fixture(tmp_path, monkeypatch)
    combined = report.parent / "recorded.jsonl"
    rows = [json.loads(line) for line in combined.read_text(encoding="utf-8").splitlines()]
    parent = next(row for row in rows if row["session_id"] == "parent-000")
    parent["split"] = "test" if parent["split"] != "test" else "train"
    combined.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(generation.RecordedGenerationError, match="combined manifest"):
        generation.validate_recorded_generation(report, repo_root=tmp_path)


def test_generation_external_sha_is_required_when_declared(tmp_path, monkeypatch):
    report, _holdout = _fixture(tmp_path, monkeypatch)
    with pytest.raises(generation.RecordedGenerationError, match="외부 SHA-256"):
        generation.validate_recorded_generation(
            report, repo_root=tmp_path, expected_sha256="0" * 64
        )


@pytest.mark.parametrize("forged_schema", [True, 1.0])
def test_generation_schema_rejects_bool_and_float(tmp_path, monkeypatch, forged_schema):
    report, _holdout = _fixture(tmp_path, monkeypatch)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["schema_version"] = forged_schema
    report.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(generation.RecordedGenerationError, match="schema_version"):
        generation.validate_recorded_generation(report, repo_root=tmp_path)


def test_metadata_dsu_unions_different_clips_from_same_esc50_source():
    rows = {
        "data/source_pool/environment/environment_000.wav": {
            "source_family": "environment",
            "group_id": "env-a",
            "clips": ["a.wav"],
        },
        "data/source_pool_v2/environment/environment_011.wav": {
            "source_family": "environment",
            "group_id": "env-b",
            "clips": ["b.wav"],
        },
        "data/source_pool_v2/environment/environment_012.wav": {
            "source_family": "environment",
            "group_id": "env-c",
            "clips": ["c.wav"],
        },
    }
    by_path, components = generation._derive_source_component_map(
        rows,
        fma_tracks={},
        librispeech_chapters={},
        esc50_metadata={"a.wav": "same-src", "b.wav": "same-src", "c.wav": "free-src"},
    )
    assert by_path["data/source_pool/environment/environment_000.wav"] == by_path[
        "data/source_pool_v2/environment/environment_011.wav"
    ]
    assert by_path["data/source_pool_v2/environment/environment_012.wav"] != by_path[
        "data/source_pool/environment/environment_000.wav"
    ]
    assert sorted(len(members) for members in components.values()) == [1, 2]


def test_librispeech_component_map_uses_transitive_reader_book_bridge():
    by_identity, components = generation._derive_librispeech_identity_component_map(
        {
            152373: (2035, 10),
            152374: (2035, 20),
            152375: (999, 20),
            128104: (1272, 30),
        }
    )
    assert by_identity["librivox_reader:2035"] == by_identity["gutenberg_book:10"]
    assert by_identity["librivox_reader:2035"] == by_identity["librivox_reader:999"]
    assert by_identity["librivox_reader:1272"] != by_identity["librivox_reader:2035"]
    assert sorted(len(members) for members in components.values()) == [2, 4]


def test_canonical_population_requires_pool8_dns5_external_esc4():
    split_plan = {
        "speech": ("train", "train", "val", "test", "test"),
        "music": ("val", "val", "test", "test"),
        "environment": ("val", "test", "test", "test"),
        "machine": ("train", "val", "test", "test"),
    }
    rows = []
    for family, splits in split_plan.items():
        for index, split in enumerate(splits):
            if family == "machine":
                kind = generation.SOURCE_KIND_EXTERNAL
            elif family == "speech":
                kind = generation.SOURCE_KIND_EXTERNAL_DNS_SPEECH
            else:
                kind = generation.SOURCE_KIND_POOL
            rows.append(
                {"source_family": family, "split": split, "source_kind": kind}
            )
    generation._validate_addition_population(rows)
    next(row for row in rows if row["source_family"] == "speech")[
        "source_kind"
    ] = generation.SOURCE_KIND_POOL
    with pytest.raises(generation.RecordedGenerationError, match="source-pool 8"):
        generation._validate_addition_population(rows)


def test_exact_highband_addition_inventory_and_split_matrix_are_frozen():
    assert generation.CANONICAL_SOURCE_POOL_ADDITIONS == {
        "data/source_pool/environment/environment_008.wav": ("environment", 54.1, "test"),
        "data/source_pool_v2/environment/environment_012.wav": ("environment", 3.0, "test"),
        "data/source_pool_v2/environment/environment_004.wav": ("environment", 5.9, "test"),
        "data/source_pool_v2/environment/environment_017.wav": ("environment", 26.2, "val"),
        "data/source_pool/music/music_007.wav": ("music", 54.8, "test"),
        "data/source_pool_v2/music/music_007.wav": ("music", 12.8, "test"),
        "data/source_pool_v2/music/music_012.wav": ("music", 17.1, "val"),
        "data/source_pool_v2/music/music_017.wav": ("music", 20.1, "val"),
    }
    assert generation.REJECTED_SOURCE_POOL_SPEECH_ADDITIONS == {
        "data/source_pool/speech/speech_002.wav": ("speech", 51.0, "test"),
        "data/source_pool/speech/speech_013.wav": ("speech", 13.75, "test"),
        "data/source_pool_v2/speech/speech_002.wav": ("speech", 0.75, "train"),
        "data/source_pool_v2/speech/speech_016.wav": ("speech", 51.75, "val"),
        "data/source_pool_v2/speech/speech_019.wav": ("speech", 50.0, "train"),
    }
    assert generation.CANONICAL_EXTERNAL_LIBRISPEECH_FILES == {
        "data/raw/speech/LibriSpeech/dev-clean/2035/152373/2035-152373-0013.flac": (
            3.0,
            "train",
        ),
        "data/raw/speech/LibriSpeech/dev-clean/1272/128104/1272-128104-0004.flac": (
            0.75,
            "train",
        ),
        "data/raw/speech/LibriSpeech/dev-clean/6241/61943/6241-61943-0027.flac": (
            0.5,
            "test",
        ),
        "data/raw/speech/LibriSpeech/dev-clean/2412/153948/2412-153948-0006.flac": (
            0.25,
            "test",
        ),
    }
    observed = {}
    for family, _start, split in generation.CANONICAL_SOURCE_POOL_ADDITIONS.values():
        observed[(family, split)] = observed.get((family, split), 0) + 1
    for split in generation.CANONICAL_EXTERNAL_ESC_SPLITS.values():
        observed[("machine", split)] = observed.get(("machine", split), 0) + 1
    for split, count in {"train": 2, "val": 1, "test": 2}.items():
        observed[("speech", split)] = count
    assert observed == {
        key: value
        for key, value in generation.EXPECTED_ADDITION_FAMILY_SPLIT_COUNTS.items()
        if value
    }


def test_source_selection_rejects_shared_transitive_authority_component(
    tmp_path, monkeypatch
):
    paths = {
        "data/source_pool/speech/speech_002.wav": ("speech", 51.0, "test"),
        "data/source_pool_v2/speech/speech_016.wav": ("speech", 51.75, "val"),
    }
    components = {
        path: f"speech-source-lineage-{index:012d}"
        for index, path in enumerate(paths, start=1)
    }
    shared = "librispeech_component:speech-librispeech-lineage-d697786cc484"
    monkeypatch.setattr(generation, "CANONICAL_SOURCE_POOL_ADDITIONS", paths)
    monkeypatch.setattr(
        generation,
        "CANONICAL_SPEECH_SELECTION_EVIDENCE",
        {
            path: {
                "start_seconds": start,
                "split": split,
                "component": components[path],
                "covered_segment_counts": [9, 9, 1, 1],
                "max_density_ratios": [1.0, 1.0, 0.3, 0.26],
            }
            for path, (_family, start, split) in paths.items()
        },
    )
    monkeypatch.setattr(
        generation,
        "_snapshot",
        lambda *_args, **_kwargs: type(
            "Snapshot",
            (),
            {
                "sha256": generation.SOURCE_SELECTION_STRICT_PRIMARY_SHA256,
                "path": tmp_path / generation.SOURCE_SELECTION_STRICT_PRIMARY_PATH,
                "size": 1,
            },
        )(),
    )
    lineage = {
        "component_by_path": components,
        "active_components": set(),
        "authority_tokens_by_path": {
            path: {f"source_pool_component:{components[path]}", shared}
            for path in paths
        },
    }
    with pytest.raises(
        generation.RecordedGenerationError,
        match="overlap=librispeech_component",
    ):
        generation._canonical_source_selection_evidence(tmp_path, lineage)


def test_unique_lineage_string_cannot_disguise_active_source_component(tmp_path, monkeypatch):
    report, _holdout = _fixture(tmp_path, monkeypatch)
    plan = tmp_path / generation.SOURCE_PLAN_ROOT / f"{GENERATION_ID}.csv"
    with plan.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    forged = "environment-source-lineage-deadbeef0000"
    rows[0]["lineage_key"] = forged
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=generation.SOURCE_PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # 공격자가 CSV의 문자열만 새 이름으로 바꿔도 component는 source path와 권위
    # ESC/FMA/Libri metadata에서 다시 유도되므로 선언값을 신뢰하지 않는다.
    original = generation._canonical_source_lineage(tmp_path)
    first_path = rows[0]["path"]
    original["active_components"] = {original["component_by_path"][first_path]}
    monkeypatch.setattr(generation, "_canonical_source_lineage", lambda _root: original)
    with pytest.raises(generation.RecordedGenerationError, match="metadata DSU"):
        generation.validate_recorded_generation(report, repo_root=tmp_path)


def test_transfer_schema_v2_requires_generation_and_loads_exact99(tmp_path, monkeypatch):
    report, holdout = _fixture(tmp_path, monkeypatch)
    transfer, builder = _schema_v2_transfer_fixture(
        tmp_path, monkeypatch, report=report, holdout=holdout
    )
    payload = json.loads(transfer.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["recorded"]["session_count"] == 99
    summary = transfer_contract.validate_transfer_manifest(
        transfer,
        repo_root=tmp_path,
        expected_sha256=_sha(transfer.read_bytes()),
    )
    assert summary["recorded_session_count"] == 99
    assert summary["_validated_recorded_manifest_snapshot"].path == report.parent / "recorded.jsonl"

    data_cfg = {
        "transfer_manifest": builder.OUTPUT,
        "transfer_manifest_sha256": _sha(transfer.read_bytes()),
        "recorded_generation": report.relative_to(tmp_path).as_posix(),
        "recorded_generation_sha256": _sha(report.read_bytes()),
    }
    snapshot = transfer_contract.bind_recorded_transfer_config(
        data_cfg, repo_root=tmp_path
    )
    assert snapshot.recorded_generation is not None
    assert snapshot.recorded_generation_summary is not None
    assert snapshot.recorded_generation_summary["recorded_session_count"] == 99
    auto_cfg = {
        "transfer_manifest": builder.OUTPUT,
        "transfer_manifest_sha256": _sha(transfer.read_bytes()),
        "recorded_generation": None,
        "recorded_generation_sha256": None,
    }
    transfer_contract.bind_recorded_transfer_config(auto_cfg, repo_root=tmp_path)
    assert auto_cfg["recorded_generation"] == report.relative_to(tmp_path).as_posix()
    assert auto_cfg["recorded_generation_sha256"] == _sha(report.read_bytes())
    partial_cfg = dict(auto_cfg)
    partial_cfg["recorded_generation_sha256"] = None
    with pytest.raises(transfer_contract.TransferContractError, match="둘 다"):
        transfer_contract.bind_recorded_transfer_config(
            partial_cfg, repo_root=tmp_path
        )
    forged_cfg = dict(data_cfg)
    forged_cfg["recorded_generation_sha256"] = "0" * 64
    with pytest.raises(transfer_contract.TransferContractError, match="path SHA"):
        transfer_contract.bind_recorded_transfer_config(
            forged_cfg, repo_root=tmp_path
        )

    transfer_payload = json.loads(transfer.read_text(encoding="utf-8"))
    transfer_payload["schema_version"] = 2.0
    transfer.write_text(json.dumps(transfer_payload) + "\n", encoding="utf-8")
    with pytest.raises(transfer_contract.TransferContractError, match="schema_version"):
        transfer_contract.validate_transfer_manifest(
            transfer,
            repo_root=tmp_path,
            expected_sha256=_sha(transfer.read_bytes()),
        )
    transfer_payload["schema_version"] = 2
    transfer.write_text(
        json.dumps(transfer_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    target = tmp_path / generation.ADDITIONS_ROOT / GENERATION_ID / "addition-16/mics.wav"
    target.write_bytes(b"replaced")
    with pytest.raises(transfer_contract.TransferContractError, match="size/SHA 불일치"):
        transfer_contract.validate_transfer_manifest(
            transfer,
            repo_root=tmp_path,
            expected_sha256=_sha(transfer.read_bytes()),
        )


def test_schema_v2_generation_survives_official_config_campaign_chain(
    tmp_path, monkeypatch
):
    """99-session generation이 모든 공식 config 해석 경계에서 하나로 유지된다."""

    report, holdout = _fixture(tmp_path, monkeypatch)
    transfer, _builder = _schema_v2_transfer_fixture(
        tmp_path, monkeypatch, report=report, holdout=holdout
    )
    transfer_sha = _sha(transfer.read_bytes())
    transfer_summary = transfer_contract.validate_transfer_manifest(
        transfer,
        repo_root=tmp_path,
        expected_sha256=transfer_sha,
    )
    assert transfer_summary["recorded_session_count"] == 99

    freeze = tmp_path / ".venv/environment-freeze.txt"
    freeze.parent.mkdir(parents=True)
    freeze.write_bytes(b"torch==2.5.1+cu121\n")
    commit = "c" * 40
    monkeypatch.setattr(
        transfer_contract, "_no_replace_head_commit", lambda _root: commit
    )
    monkeypatch.setattr(
        transfer_contract,
        "exact_clean_source_evidence",
        lambda _root, *, expected_commit=None: {
            "schema": "exact_clean_git_source/v1",
            "commit": commit,
        },
    )
    receipt_payload = {
        "schema_version": 1,
        "expected_commit": commit,
        "canonical_holdout": {
            "path": generation.PARENT_HOLDOUT,
            "sha256": transfer_summary["canonical_holdout_sha256"],
        },
        "transfer_manifest": {
            "path": "data/manifests/elice_transfer_manifest.json",
            "sha256": transfer_sha,
        },
        "recorded_aggregate_sha256": transfer_summary[
            "recorded_aggregate_sha256"
        ],
        "environment": {
            "freeze_receipt": ".venv/environment-freeze.txt",
            "freeze_receipt_sha256": _sha(freeze.read_bytes()),
            "torch_version": "2.5.1+cu121",
            "torch_cuda": "12.1",
        },
    }
    receipt = tmp_path / "data/manifests/elice_bootstrap_receipt.json"
    receipt.write_text(
        json.dumps(receipt_payload, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    receipt_sha = _sha(receipt.read_bytes())

    # 이 회귀의 대상은 실제 YAML→transfer validator→generation materializer다.
    # 별도 테스트가 강제하는 RIR bytes와 source-tree 기반 run-dir 계산만 격리한다.
    monkeypatch.setattr(config_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(config_module, "_require_canonical_rir_bank", lambda _cfg: None)

    from deep_anc.train import a100_pretrain_smoke, experiment_contract

    def contract_digest(cfg):
        identity = {
            "role": cfg.get("experiment_role"),
            "alpha": (cfg.get("loss") or {}).get("nmse_cvar_alpha"),
            "generation": (cfg.get("data") or {}).get("recorded_generation"),
        }
        return _sha(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        )

    def fake_contract_run_directory(cfg, *, repo_root):
        digest = contract_digest(cfg)
        return Path(repo_root) / "runs" / digest[:16], digest

    def fake_stamp_experiment_contract(cfg, *, repo_root):
        digest = contract_digest(cfg)
        resolved = dict(cfg)
        resolved["experiment_contract"] = {"sha256": digest}
        resolved["experiment_contract_sha256"] = digest
        return resolved

    monkeypatch.setattr(
        experiment_contract,
        "contract_run_directory",
        fake_contract_run_directory,
    )
    monkeypatch.setattr(
        experiment_contract,
        "stamp_experiment_contract",
        fake_stamp_experiment_contract,
    )
    smoke_target = _sha(b"schema-v2-smoke-target")
    monkeypatch.setattr(
        a100_pretrain_smoke,
        "build_a100_pretrain_smoke_target",
        lambda _cfg, *, repo_root: {"sha256": smoke_target},
    )
    monkeypatch.setattr(
        a100_pretrain_smoke,
        "smoke_run_directory",
        lambda _cfg, *, repo_root: Path(repo_root)
        / "results/training_prerequisites/a100_pretrain_smoke"
        / smoke_target
        / "uninterrupted",
    )
    monkeypatch.setattr(
        a100_pretrain_smoke,
        "validate_a100_pretrain_smoke_config",
        lambda _cfg, *, repo_root: None,
    )

    anchor = f"data.bootstrap_receipt_sha256={receipt_sha}"
    pretrain_config = "configs/train_pretrain_tiny.yaml"
    stages = {
        "g0_diagnostic": config_module.load_train_config(
            pretrain_config,
            [
                anchor,
                "experiment_role=diagnostic_overfit",
                "init_eligible=false",
                "contract_run_dir=false",
                "schedule.total_steps=1",
            ],
        ),
        "loss_pilot": config_module.load_train_config(
            pretrain_config,
            [
                anchor,
                "experiment_role=loss_pilot",
                "init_eligible=false",
                "run_until_step=20000",
            ],
        ),
        "measured_probe": config_module.load_train_config(
            pretrain_config,
            [
                anchor,
                "experiment_role=measured_probe",
                "init_eligible=false",
                "run_until_step=5000",
                "data.digital_primary_path_mode=measured",
                "init_ckpt=runs/pilot/best.pt",
            ],
        ),
        "smoke": config_module.load_train_config(
            pretrain_config,
            [
                anchor,
                "experiment_role=a100_pretrain_smoke",
                "init_eligible=false",
                "contract_run_dir=false",
                "campaign_prerequisite=null",
                "campaign_prerequisite_sha256=null",
                "a100_smoke_run_label=uninterrupted",
                "run_until_step=200",
            ],
        ),
        "canonical_100k": config_module.load_train_config(
            pretrain_config,
            [anchor, f"campaign_prerequisite_sha256={'f' * 64}"],
        ),
        "finetune": config_module.load_train_config(
            "configs/train_finetune.yaml",
            [
                anchor,
                "data.digital_primary_path_mode=measured",
                "init_ckpt=runs/canonical/best.pt",
            ],
        ),
    }

    issuer_path = (
        Path(__file__).resolve().parents[1]
        / "scripts/train/issue_canonical_pretrain_prerequisite.py"
    )
    issuer_spec = importlib.util.spec_from_file_location(
        "schema_v2_campaign_issuer_test", issuer_path
    )
    issuer = importlib.util.module_from_spec(issuer_spec)
    assert issuer_spec.loader is not None
    sys.modules[issuer_spec.name] = issuer
    issuer_spec.loader.exec_module(issuer)
    stages["issuer"] = issuer._canonical_cfg(
        pretrain_config,
        receipt_sha,
        "f" * 64,
        loss_alpha=0.7,
        loss_lambda_dnh=0.00075,
    )

    expected_generation = report.relative_to(tmp_path).as_posix()
    expected_generation_sha = _sha(report.read_bytes())
    expected_manifest = (report.parent / "recorded.jsonl").relative_to(
        tmp_path
    ).as_posix()
    assert len((tmp_path / expected_manifest).read_text().splitlines()) == 99
    for label, cfg in stages.items():
        assert cfg["data"]["recorded_generation"] == expected_generation, label
        assert (
            cfg["data"]["recorded_generation_sha256"]
            == expected_generation_sha
        ), label
        assert (
            config_module.canonical_recorded_manifest_for_data(cfg["data"])
            == expected_manifest
        ), label
    assert stages["measured_probe"]["recorded_manifest"] == expected_manifest
    assert stages["finetune"]["recorded_manifest"] == expected_manifest

    # transfer와 다른 generation SHA 또는 combined manifest로 우회하면 config
    # 해석 단계에서 학습 contract를 만들기 전에 닫혀야 한다.
    with pytest.raises(transfer_contract.TransferContractError, match="path SHA"):
        config_module.load_train_config(
            pretrain_config,
            [
                anchor,
                f"data.recorded_generation={expected_generation}",
                f"data.recorded_generation_sha256={'0' * 64}",
                f"campaign_prerequisite_sha256={'f' * 64}",
            ],
        )
    with pytest.raises(ValueError, match="recorded_manifest"):
        config_module.load_train_config(
            pretrain_config,
            [
                anchor,
                "experiment_role=measured_probe",
                "init_eligible=false",
                "run_until_step=5000",
                "data.digital_primary_path_mode=measured",
                "init_ckpt=runs/pilot/best.pt",
                "recorded_manifest=data/manifests/forged.jsonl",
            ],
        )


def test_external_composite_rederives_esc_identity_transform_and_output_sha(
    tmp_path, monkeypatch
):
    report, holdout = _fixture(tmp_path, monkeypatch)
    source_lineage = generation._canonical_source_lineage(tmp_path)
    raw_member = (
        tmp_path
        / "data/raw/noise/esc50/ESC-50-master/audio/1-28808-A-43.wav"
    )
    raw_member.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(5 * 44_100, dtype=np.float64) / 44_100.0
    sf.write(
        raw_member,
        (0.2 * np.sin(2 * np.pi * 700.0 * time)).astype(np.float32),
        44_100,
        subtype="PCM_16",
    )
    raw_lineage = "esc50_src:28808"
    digest12 = hashlib.sha256(raw_lineage.encode()).hexdigest()[:12]
    output = (
        tmp_path
        / generation.SOURCE_PLAN_ROOT
        / (
            f"{GENERATION_ID}_sources/"
            + generation.CANONICAL_EXTERNAL_ESC_OUTPUT_NAMES[raw_member.name]
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(generation._canonical_external_composite_bytes(raw_member))

    plan = tmp_path / generation.SOURCE_PLAN_ROOT / f"{GENERATION_ID}.csv"
    with plan.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[3].update(
        {
            "source_kind": generation.SOURCE_KIND_EXTERNAL,
            "path": output.relative_to(tmp_path).as_posix(),
            "seconds": "15.0",
            "start_seconds": "0.0",
            "source_family": "machine",
            "group_id": f"machine-esc50-source-{digest12}",
            "lineage_key": f"machine-external-lineage-{digest12}",
            "source_file_sha256": _sha(output.read_bytes()),
            "raw_member_path": raw_member.relative_to(tmp_path).as_posix(),
            "raw_member_sha256": _sha(raw_member.read_bytes()),
            "raw_member_lineage_key": raw_lineage,
            "authority_metadata_sha256": "c" * 64,
            "inventory_path": "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv",
            "inventory_sha256": "c" * 64,
            "transform": generation.EXTERNAL_TRANSFORM,
            "transform_repeat_count": str(generation.EXTERNAL_REPEAT_COUNT),
        }
    )
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=generation.SOURCE_PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    plan_sha = _sha(plan.read_bytes())

    additions = tmp_path / generation.ADDITIONS_ROOT / GENERATION_ID
    for session_json in additions.glob("*/session.json"):
        metadata = json.loads(session_json.read_text(encoding="utf-8"))
        metadata["collection_plan"]["source_list_sha256"] = plan_sha
        if metadata["collection_plan"]["source_row_number"] == 5:
            metadata["program"]["file"] = rows[3]["path"]
            metadata["program"]["file_start_seconds"] = 0.0
            metadata["source_family"] = "machine"
            metadata["group_id"] = rows[3]["group_id"]
            metadata["seconds"] = 15.0
            metadata["collection_plan"].update(
                {
                    "lineage_key": rows[3]["lineage_key"],
                    "source_file_sha256": rows[3]["source_file_sha256"],
                    "start_seconds": 0.0,
                }
            )
            _write_recorded_session_artifacts(
                session_json.parent,
                seconds=15.0,
                metadata=metadata,
                source_path=output,
            )
        else:
            _write_json(session_json, metadata)

    source_lineage.update(
        {
            "active_identity_keys": set(),
            "esc50_metadata": {raw_member.name.casefold(): "28808"},
            "esc50_metadata_sha256": "c" * 64,
            "esc50_metadata_path": "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv",
            "esc50_authority": {raw_member.name: ("28808", "car_horn")},
        }
    )
    monkeypatch.setattr(
        generation,
        "EXPECTED_ADDITION_FAMILY_KIND_COUNTS",
        {
            ("speech", generation.SOURCE_KIND_POOL): 5,
            ("music", generation.SOURCE_KIND_POOL): 4,
            ("environment", generation.SOURCE_KIND_POOL): 4,
            ("machine", generation.SOURCE_KIND_POOL): 3,
            ("machine", generation.SOURCE_KIND_EXTERNAL): 1,
        },
    )
    monkeypatch.setattr(
        generation, "_canonical_source_lineage", lambda _root: source_lineage
    )
    _rewrite_progress_seconds(
        additions / "batch_progress.csv", row_number=5, seconds=15.0
    )
    report.unlink()
    (report.parent / "recorded.jsonl").unlink()
    combined = generation.build_combined_manifest_bytes(
        repo_root=tmp_path,
        generation_id=GENERATION_ID,
        expected_holdout_sha256=holdout["sha256"],
    )
    (report.parent / "recorded.jsonl").write_bytes(combined)
    payload = generation.build_recorded_generation_payload(
        repo_root=tmp_path,
        generation_id=GENERATION_ID,
        expected_holdout_sha256=holdout["sha256"],
    )
    report.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    assert generation.validate_recorded_generation(report, repo_root=tmp_path)[
        "recorded_session_count"
    ] == 99

    pool_path = rows[2]["path"]
    original_tokens = set(source_lineage["authority_tokens_by_path"][pool_path])
    source_lineage["authority_tokens_by_path"][pool_path].add(
        f"esc50_identity:{raw_lineage}"
    )
    with pytest.raises(generation.RecordedGenerationError, match="서로 같은 권위"):
        generation.validate_recorded_generation(report, repo_root=tmp_path)
    source_lineage["authority_tokens_by_path"][pool_path] = original_tokens

    source_lineage["active_identity_keys"] = {("esc50", raw_lineage)}
    with pytest.raises(generation.RecordedGenerationError, match="active82 disjoint"):
        generation.validate_recorded_generation(report, repo_root=tmp_path)


def test_external_librispeech_rederives_transitive_component_and_window(
    tmp_path, monkeypatch
):
    report, holdout = _fixture(tmp_path, monkeypatch)
    source_lineage = generation._canonical_source_lineage(tmp_path)
    source = (
        tmp_path
        / "data/raw/speech/LibriSpeech/dev-clean/2035/152373/"
        "2035-152373-0013.flac"
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16_000
    time = np.arange(18 * sample_rate, dtype=np.float64) / sample_rate
    sf.write(
        source,
        (0.1 * np.sin(2 * np.pi * 900.0 * time)).astype(np.float32),
        sample_rate,
        format="FLAC",
        subtype="PCM_16",
    )
    chapters = {
        152373: (2035, 10),
        # candidate reader -> 다른 book -> active reader로 이어지는 transitive bridge.
        152374: (2035, 20),
        152375: (999, 20),
        128104: (1272, 30),
    }
    by_identity, _components = generation._derive_librispeech_identity_component_map(
        chapters
    )
    component = by_identity["librivox_reader:2035"]
    digest12 = hashlib.sha256(component.encode()).hexdigest()[:12]
    transcript = source.with_name("2035-152373.trans.txt")
    transcript.write_text(
        "2035-152373-0013 FIXTURE AUTHORITATIVE UTTERANCE\n",
        encoding="utf-8",
    )

    plan = tmp_path / generation.SOURCE_PLAN_ROOT / f"{GENERATION_ID}.csv"
    with plan.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0].update(
        {
            "source_kind": generation.SOURCE_KIND_EXTERNAL_LIBRISPEECH,
            "path": source.relative_to(tmp_path).as_posix(),
            "seconds": "15.0",
            "start_seconds": "3.0",
            "source_family": "speech",
            "group_id": f"speech-librispeech-source-{digest12}",
            "lineage_key": f"speech-external-lineage-{digest12}",
            "source_file_sha256": _sha(source.read_bytes()),
            "raw_member_path": source.relative_to(tmp_path).as_posix(),
            "raw_member_sha256": _sha(source.read_bytes()),
            "raw_member_lineage_key": component,
            "authority_metadata_sha256": "b" * 64,
            "inventory_path": transcript.relative_to(tmp_path).as_posix(),
            "inventory_sha256": _sha(transcript.read_bytes()),
            "transform": generation.EXTERNAL_LIBRISPEECH_TRANSFORM,
            "transform_repeat_count": "1",
        }
    )
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=generation.SOURCE_PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    plan_sha = _sha(plan.read_bytes())

    additions = tmp_path / generation.ADDITIONS_ROOT / GENERATION_ID
    for session_json in additions.glob("*/session.json"):
        metadata = json.loads(session_json.read_text(encoding="utf-8"))
        metadata["collection_plan"]["source_list_sha256"] = plan_sha
        if metadata["collection_plan"]["source_row_number"] == 2:
            metadata["program"]["file"] = rows[0]["path"]
            metadata["program"]["file_start_seconds"] = 3.0
            metadata["source_family"] = "speech"
            metadata["group_id"] = rows[0]["group_id"]
            metadata["seconds"] = 15.0
            metadata["collection_plan"].update(
                {
                    "lineage_key": rows[0]["lineage_key"],
                    "source_file_sha256": rows[0]["source_file_sha256"],
                    "start_seconds": 3.0,
                }
            )
            _write_recorded_session_artifacts(
                session_json.parent,
                seconds=15.0,
                metadata=metadata,
                source_path=source,
            )
        else:
            _write_json(session_json, metadata)

    source_lineage.update(
        {
            "librispeech_chapters": chapters,
            "librispeech_component_by_identity": by_identity,
            "active_librispeech_components": set(),
            "librispeech_chapters_sha256": "b" * 64,
        }
    )
    monkeypatch.setattr(
        generation,
        "EXPECTED_ADDITION_FAMILY_KIND_COUNTS",
        {
            ("speech", generation.SOURCE_KIND_POOL): 4,
            ("speech", generation.SOURCE_KIND_EXTERNAL_LIBRISPEECH): 1,
            ("music", generation.SOURCE_KIND_POOL): 4,
            ("environment", generation.SOURCE_KIND_POOL): 4,
            ("machine", generation.SOURCE_KIND_POOL): 4,
        },
    )
    monkeypatch.setattr(
        generation, "_canonical_source_lineage", lambda _root: source_lineage
    )
    _rewrite_progress_seconds(
        additions / "batch_progress.csv", row_number=2, seconds=15.0
    )
    report.unlink()
    (report.parent / "recorded.jsonl").unlink()
    combined = generation.build_combined_manifest_bytes(
        repo_root=tmp_path,
        generation_id=GENERATION_ID,
        expected_holdout_sha256=holdout["sha256"],
    )
    (report.parent / "recorded.jsonl").write_bytes(combined)
    payload = generation.build_recorded_generation_payload(
        repo_root=tmp_path,
        generation_id=GENERATION_ID,
        expected_holdout_sha256=holdout["sha256"],
    )
    report.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    assert generation.validate_recorded_generation(report, repo_root=tmp_path)[
        "recorded_session_count"
    ] == 99

    other_speech_pool_path = rows[4]["path"]
    original_tokens = set(
        source_lineage["authority_tokens_by_path"][other_speech_pool_path]
    )
    source_lineage["authority_tokens_by_path"][other_speech_pool_path].add(
        f"librispeech_component:{component}"
    )
    with pytest.raises(generation.RecordedGenerationError, match="서로 같은 권위"):
        generation.validate_recorded_generation(report, repo_root=tmp_path)
    source_lineage["authority_tokens_by_path"][other_speech_pool_path] = original_tokens

    # candidate의 직접 key(reader2035/book10)는 active가 아니어도 reader999가 book20을
    # 통해 같은 CHAPTERS component에 있으면 fail-closed 해야 한다.
    assert "librivox_reader:999" not in generation.public_lineage.librispeech_lineage_keys(
        source.name, chapters
    )
    source_lineage["active_librispeech_components"] = {
        by_identity["librivox_reader:999"]
    }
    with pytest.raises(generation.RecordedGenerationError, match="transitive component"):
        generation.validate_recorded_generation(report, repo_root=tmp_path)

    source_lineage["active_librispeech_components"] = set()
    source_lineage["librispeech_chapters_sha256"] = "d" * 64
    with pytest.raises(generation.RecordedGenerationError, match="CHAPTERS SHA"):
        generation.validate_recorded_generation(report, repo_root=tmp_path)
