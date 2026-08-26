"""Jetson→Elice immutable transfer manifest의 fail-closed 계약."""

from __future__ import annotations

import hashlib
import io
import importlib.util
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import numpy as np
import soundfile as sf

import deep_anc.data.transfer_contract as transfer_contract
from deep_anc.data.holdout_contract import snapshot_regular_tree_metadata
from deep_anc.data.transfer_contract import (
    TransferContractError,
    _canonical_recorded_aggregate,
    bind_recorded_transfer_config,
    validate_transfer_manifest,
)
from deep_anc.data.recorded_dataset import RecordedANCDataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    path = REPO_ROOT / "scripts/data/build_elice_transfer_manifest.py"
    spec = importlib.util.spec_from_file_location("_elice_transfer_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _wav_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, value, 48_000, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def _write_transfer_bundle(
    root: Path, *, dataset_ready: bool = False
) -> tuple[Path, str, list[dict[str, object]]]:
    files: list[dict[str, object]] = []

    def add(relative: str, role: str, payload: bytes) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files.append(
            {
                "path": relative,
                "role": role,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        )

    for index in range(82):
        metadata = {"session_id": f"session-{index:03d}"}
        add(
            f"data/recorded/session-{index:03d}/session.json",
            "recorded",
            json.dumps(metadata).encode(),
        )
    if dataset_ready:
        samples = 1024
        time = np.arange(samples, dtype=np.float32) / 48_000.0
        source = np.sin(2.0 * np.pi * 300.0 * time).astype(np.float32)
        mics = np.stack([source * 0.5, source * 0.25], axis=1)
        add("data/recorded/session-000/mics.wav", "recorded", _wav_bytes(mics))
        add(
            "data/recorded/session-000/source_aligned.wav",
            "recorded",
            _wav_bytes(source),
        )
    source_metadata = snapshot_regular_tree_metadata(
        root / "data/recorded",
        repo_root=root,
        label="fixture recorded source",
    )
    add("data/rir_bank/duct_rirs_v1.npz", "rir_bank", b"fixture-rir")
    add("results/strict_ps/raw_measurement.npz", "strict_ps_raw", b"fixture-raw")
    add("results/strict_ps/analysis.json", "strict_ps_analysis", b"fixture-analysis")
    add("assets/measured/primary_path_il.npz", "strict_primary_npz", b"fixture-p")
    add("assets/measured/secondary_path_il.npz", "strict_secondary_npz", b"fixture-s")
    recorded_manifest_bytes = (
        (
            json.dumps(
                {
                    "session_id": "session-000",
                    "source_family": "machine",
                    "group_id": "machine-lineage-fixture",
                    "source_pool_group_id": "machine-source-pool",
                    "split": "train",
                    "path_base": "manifest",
                    "path": "../recorded/session-000",
                }
            )
            + "\n"
        ).encode()
        if dataset_ready
        else b'{"fixture":"regrouped"}\n'
    )
    add(
        "data/manifests/recorded_regrouped.jsonl",
        "recorded_manifest",
        recorded_manifest_bytes,
    )
    add(
        "data/raw/music/fma_metadata/tracks.csv",
        "lineage_tracks",
        b"fixture-tracks\n",
    )
    add(
        "data/raw/speech/LibriSpeech/CHAPTERS.TXT",
        "librispeech_chapters_metadata",
        b"fixture-chapters\n",
    )
    add(
        "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv",
        "esc50_metadata",
        b"fixture-esc50-metadata\n",
    )
    add(
        "data/manifests/recorded_holdout.json",
        "canonical_holdout",
        b'{"fixture":"canonical-holdout"}\n',
    )
    add("data/source_pool/sources.csv", "source_pool_v1_csv", b"fixture-v1-csv\n")
    add("data/source_pool_v2/sources.csv", "source_pool_v2_csv", b"fixture-v2-csv\n")
    report_bytes = b'{"fixture":"canonical-provenance"}\n'
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    report_relative = (
        "results/provenance/"
        f"source_pool_provenance_report.{report_sha}.json"
    )
    add(report_relative, "provenance_report", report_bytes)
    files.sort(key=lambda item: str(item["path"]))
    recorded_entries = [item for item in files if item["role"] == "recorded"]
    payload = {
        "schema_version": 1,
        "files": files,
        "recorded": {
            "root": "data/recorded",
            "session_count": 82,
            "file_count": len(recorded_entries),
            "total_bytes": sum(int(item["size"]) for item in recorded_entries),
            "aggregate_sha256": _canonical_recorded_aggregate(recorded_entries),
            "source_metadata_file_count": source_metadata.file_count,
            "source_metadata_snapshot_sha256": source_metadata.sha256,
            "source_content_snapshot_sha256": source_metadata.content_sha256,
        },
        "rir_bank": "data/rir_bank/duct_rirs_v1.npz",
        "recorded_manifest": "data/manifests/recorded_regrouped.jsonl",
        "lineage_tracks": "data/raw/music/fma_metadata/tracks.csv",
        "librispeech_chapters_metadata": (
            "data/raw/speech/LibriSpeech/CHAPTERS.TXT"
        ),
        "esc50_metadata": (
            "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv"
        ),
        "canonical_provenance": {
            "holdout": {
                "path": "data/manifests/recorded_holdout.json",
                "sha256": hashlib.sha256(
                    (root / "data/manifests/recorded_holdout.json").read_bytes()
                ).hexdigest(),
            },
            "sources_csv": {
                "source_pool": {
                    "path": "data/source_pool/sources.csv",
                    "sha256": hashlib.sha256(
                        (root / "data/source_pool/sources.csv").read_bytes()
                    ).hexdigest(),
                },
                "source_pool_v2": {
                    "path": "data/source_pool_v2/sources.csv",
                    "sha256": hashlib.sha256(
                        (root / "data/source_pool_v2/sources.csv").read_bytes()
                    ).hexdigest(),
                },
            },
            "provenance_report": {
                "path": report_relative,
                "sha256": report_sha,
            },
        },
        "strict_ps": {
            "capture_root": "results/strict_ps",
            "raw": ["results/strict_ps/raw_measurement.npz"],
            "analysis": ["results/strict_ps/analysis.json"],
            "primary_npz": "assets/measured/primary_path_il.npz",
            "secondary_npz": "assets/measured/secondary_path_il.npz",
        },
    }
    manifest = root / "data/manifests/elice_transfer_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest(), files


def _fixture_holdout_summary(repo_root: Path) -> dict[str, object]:
    recorded_tree = snapshot_regular_tree_metadata(
        repo_root / "data/recorded",
        repo_root=repo_root,
        label="fixture canonical recorded tree",
    )
    report = next(
        (repo_root / "results/provenance").glob(
            "source_pool_provenance_report.*.json"
        )
    )
    return {
        "sha256": hashlib.sha256(
            (repo_root / "data/manifests/recorded_holdout.json").read_bytes()
        ).hexdigest(),
        "sources_csv": [
            "data/source_pool/sources.csv",
            "data/source_pool_v2/sources.csv",
        ],
        "sources_csv_sha256": {
            "source_pool": hashlib.sha256(
                (repo_root / "data/source_pool/sources.csv").read_bytes()
            ).hexdigest(),
            "source_pool_v2": hashlib.sha256(
                (repo_root / "data/source_pool_v2/sources.csv").read_bytes()
            ).hexdigest(),
        },
        "provenance_report": report.relative_to(repo_root).as_posix(),
        "provenance_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "lineage": {
            "regrouped_manifest_sha256": hashlib.sha256(
                (repo_root / "data/manifests/recorded_regrouped.jsonl").read_bytes()
            ).hexdigest(),
            "tracks_csv_sha256": hashlib.sha256(
                (repo_root / "data/raw/music/fma_metadata/tracks.csv").read_bytes()
            ).hexdigest(),
            "librispeech_chapters_path": (
                "data/raw/speech/LibriSpeech/CHAPTERS.TXT"
            ),
            "librispeech_chapters_sha256": hashlib.sha256(
                (
                    repo_root
                    / "data/raw/speech/LibriSpeech/CHAPTERS.TXT"
                ).read_bytes()
            ).hexdigest(),
            "esc50_metadata_path": (
                "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv"
            ),
            "esc50_metadata_sha256": hashlib.sha256(
                (
                    repo_root
                    / "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv"
                ).read_bytes()
            ).hexdigest(),
        },
        "recorded_tree": {
            "file_count": recorded_tree.file_count,
            "metadata_snapshot_sha256": recorded_tree.sha256,
            "content_snapshot_sha256": recorded_tree.content_sha256,
        },
    }


def _write_bootstrap_receipt(
    root: Path,
    *,
    transfer_sha256: str,
) -> tuple[Path, str, dict[str, object]]:
    summary = validate_transfer_manifest(
        root / "data/manifests/elice_transfer_manifest.json",
        repo_root=root,
        expected_sha256=transfer_sha256,
    )
    freeze = root / ".venv/environment-freeze.txt"
    freeze.parent.mkdir(parents=True, exist_ok=True)
    freeze.write_bytes(b"torch==2.5.1+cu121\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    code_anchor = root / "receipt-code-anchor.txt"
    code_anchor.write_text("fixture code\n", encoding="utf-8")
    subprocess.run(["git", "add", code_anchor.name], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "fixture code",
        ],
        cwd=root,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "schema_version": 1,
        "expected_commit": commit,
        "canonical_holdout": {
            "path": "data/manifests/recorded_holdout.json",
            "sha256": summary["canonical_holdout_sha256"],
        },
        "transfer_manifest": {
            "path": "data/manifests/elice_transfer_manifest.json",
            "sha256": transfer_sha256,
        },
        "recorded_aggregate_sha256": summary[
            "recorded_aggregate_sha256"
        ],
        "environment": {
            "freeze_receipt": ".venv/environment-freeze.txt",
            "freeze_receipt_sha256": hashlib.sha256(
                freeze.read_bytes()
            ).hexdigest(),
            "torch_version": "2.5.1+cu121",
            "torch_cuda": "12.1",
        },
    }
    receipt = root / "data/manifests/elice_bootstrap_receipt.json"
    receipt.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt, hashlib.sha256(receipt.read_bytes()).hexdigest(), summary


def _dataset_config(**anchors: object) -> dict[str, object]:
    return {
        "sample_rate": 48_000,
        "segment_seconds": 256 / 48_000,
        "reference_mode": "digital",
        "digital_reference_lead_samples": 0,
        "closed_loop": {"feedback_delay_samples": [1, 2]},
        **anchors,
    }


@pytest.fixture(autouse=True)
def _fixture_holdout_lineage(monkeypatch, tmp_path: Path):
    def validate(_path, *, repo_root, expected_sha256=None):
        summary = _fixture_holdout_summary(Path(repo_root))
        if expected_sha256 is not None and summary["sha256"] != expected_sha256:
            raise TransferContractError("fixture expected holdout SHA mismatch")
        return summary

    monkeypatch.setattr(transfer_contract, "validate_holdout_contract", validate)


def test_transfer_manifest_validates_exact_82_session_bundle(tmp_path: Path) -> None:
    manifest, digest, files = _write_transfer_bundle(tmp_path)
    summary = validate_transfer_manifest(
        manifest, repo_root=tmp_path, expected_sha256=digest
    )
    assert summary["recorded_session_count"] == 82
    assert summary["file_count"] == len(files)


def test_transfer_manifest_detects_file_tampering_from_same_declared_path(
    tmp_path: Path,
) -> None:
    manifest, digest, _files = _write_transfer_bundle(tmp_path)
    target = tmp_path / "results/strict_ps/analysis.json"
    target.write_bytes(b"forged-analysis")
    with pytest.raises(TransferContractError, match="size/SHA 불일치"):
        validate_transfer_manifest(manifest, repo_root=tmp_path, expected_sha256=digest)


def test_transfer_manifest_binds_canonical_source_csv_content(tmp_path: Path) -> None:
    manifest, digest, _files = _write_transfer_bundle(tmp_path)
    (tmp_path / "data/source_pool/sources.csv").write_bytes(b"forged-source-csv\n")
    with pytest.raises(TransferContractError, match="size/SHA 불일치"):
        validate_transfer_manifest(manifest, repo_root=tmp_path, expected_sha256=digest)


def test_transfer_manifest_rejects_rebound_canonical_provenance_pointer(
    tmp_path: Path,
) -> None:
    manifest, _digest, _files = _write_transfer_bundle(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["canonical_provenance"]["sources_csv"]["source_pool"]["sha256"] = "0" * 64
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(TransferContractError, match="canonical_provenance pointer/SHA"):
        validate_transfer_manifest(manifest, repo_root=tmp_path, expected_sha256=digest)


def test_transfer_manifest_rejects_symlinked_recorded_file(tmp_path: Path) -> None:
    manifest, digest, _files = _write_transfer_bundle(tmp_path)
    target = tmp_path / "data/recorded/session-000/session.json"
    saved = target.with_name("saved.json")
    saved.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(saved)
    with pytest.raises(TransferContractError, match="symlink"):
        validate_transfer_manifest(manifest, repo_root=tmp_path, expected_sha256=digest)


def test_transfer_manifest_rejects_unlisted_recorded_file(tmp_path: Path) -> None:
    manifest, digest, _files = _write_transfer_bundle(tmp_path)
    (tmp_path / "data/recorded/session-000/unlisted.wav").write_bytes(b"extra")
    with pytest.raises(TransferContractError, match="exact 파일 집합"):
        validate_transfer_manifest(manifest, repo_root=tmp_path, expected_sha256=digest)


def test_builder_scans_recorded_and_publishes_no_replace_canonical_json(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, _digest, _files = _write_transfer_bundle(tmp_path)
    manifest.unlink()
    builder = _load_builder()
    monkeypatch.setattr(builder, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        builder,
        "validate_holdout_contract",
        lambda _path, *, repo_root, expected_sha256=None: _fixture_holdout_summary(
            Path(repo_root)
        ),
    )
    args = Namespace(
        recorded_root="data/recorded",
        rir_bank="data/rir_bank/duct_rirs_v1.npz",
        strict_raw=["results/strict_ps/raw_measurement.npz"],
        strict_analysis=["results/strict_ps/analysis.json"],
        primary_npz="assets/measured/primary_path_il.npz",
        secondary_npz="assets/measured/secondary_path_il.npz",
        expected_holdout_sha256=hashlib.sha256(
            (tmp_path / "data/manifests/recorded_holdout.json").read_bytes()
        ).hexdigest(),
        recorded_manifest="data/manifests/recorded_regrouped.jsonl",
        lineage_tracks="data/raw/music/fma_metadata/tracks.csv",
        librispeech_chapters_metadata=(
            "data/raw/speech/LibriSpeech/CHAPTERS.TXT"
        ),
        esc50_metadata=(
            "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv"
        ),
        out=builder.OUTPUT,
    )
    payload = builder.build_payload(args, repo_root=tmp_path)
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    builder._publish_no_replace(manifest, data, repo_root=tmp_path)
    builder._publish_no_replace(manifest, data, repo_root=tmp_path)
    assert manifest.read_bytes() == data
    with pytest.raises(TransferContractError, match="overwrite"):
        builder._publish_no_replace(manifest, data + b" ", repo_root=tmp_path)


def test_recorded_config_binder_injects_only_validated_transfer_aggregate(
    tmp_path: Path,
) -> None:
    manifest, digest, _files = _write_transfer_bundle(tmp_path)
    data_cfg = {
        "transfer_manifest": manifest.relative_to(tmp_path).as_posix(),
        "transfer_manifest_sha256": digest,
    }

    snapshot = bind_recorded_transfer_config(data_cfg, repo_root=tmp_path)

    assert data_cfg["recorded_transfer_aggregate_sha256"] == (
        snapshot.recorded_aggregate_sha256
    )
    assert snapshot.bootstrap_receipt is None
    data_cfg["recorded_transfer_aggregate_sha256"] = "0" * 64
    with pytest.raises(TransferContractError, match="검증된 transfer manifest"):
        bind_recorded_transfer_config(data_cfg, repo_root=tmp_path)


def test_recorded_config_binder_chains_external_bootstrap_receipt_to_transfer(
    tmp_path: Path,
) -> None:
    _manifest, transfer_sha, _files = _write_transfer_bundle(tmp_path)
    receipt, receipt_sha, summary = _write_bootstrap_receipt(
        tmp_path,
        transfer_sha256=transfer_sha,
    )
    data_cfg = {
        "bootstrap_receipt": receipt.relative_to(tmp_path).as_posix(),
        "bootstrap_receipt_sha256": receipt_sha,
    }

    snapshot = bind_recorded_transfer_config(data_cfg, repo_root=tmp_path)

    assert snapshot.bootstrap_receipt is not None
    assert snapshot.bootstrap_receipt.sha256 == receipt_sha
    assert data_cfg["transfer_manifest_sha256"] == transfer_sha
    assert data_cfg["recorded_transfer_aggregate_sha256"] == summary[
        "recorded_aggregate_sha256"
    ]


def test_recorded_config_binder_rejects_receipt_chain_tampering(
    tmp_path: Path,
) -> None:
    _manifest, transfer_sha, _files = _write_transfer_bundle(tmp_path)
    receipt, receipt_sha, _summary = _write_bootstrap_receipt(
        tmp_path,
        transfer_sha256=transfer_sha,
    )
    data_cfg = {
        "bootstrap_receipt": receipt.relative_to(tmp_path).as_posix(),
        "bootstrap_receipt_sha256": receipt_sha,
    }
    (tmp_path / ".venv/environment-freeze.txt").write_bytes(b"forged freeze\n")

    with pytest.raises(TransferContractError, match="environment freeze receipt"):
        bind_recorded_transfer_config(data_cfg, repo_root=tmp_path)


def test_recorded_config_binder_rejects_transfer_changed_after_receipt(
    tmp_path: Path,
) -> None:
    manifest, transfer_sha, _files = _write_transfer_bundle(tmp_path)
    receipt, receipt_sha, _summary = _write_bootstrap_receipt(
        tmp_path,
        transfer_sha256=transfer_sha,
    )
    data_cfg = {
        "bootstrap_receipt": receipt.relative_to(tmp_path).as_posix(),
        "bootstrap_receipt_sha256": receipt_sha,
    }
    manifest.write_bytes(manifest.read_bytes() + b" ")

    with pytest.raises(TransferContractError, match="manifest SHA-256 불일치"):
        bind_recorded_transfer_config(data_cfg, repo_root=tmp_path)


def test_recorded_config_binder_rejects_receipt_from_another_commit(
    tmp_path: Path,
) -> None:
    _manifest, transfer_sha, _files = _write_transfer_bundle(tmp_path)
    receipt, _receipt_sha, _summary = _write_bootstrap_receipt(
        tmp_path,
        transfer_sha256=transfer_sha,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["expected_commit"] = "0" * 40
    receipt.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    data_cfg = {
        "bootstrap_receipt": receipt.relative_to(tmp_path).as_posix(),
        "bootstrap_receipt_sha256": hashlib.sha256(
            receipt.read_bytes()
        ).hexdigest(),
    }

    with pytest.raises(TransferContractError, match="no-replace git HEAD"):
        bind_recorded_transfer_config(data_cfg, repo_root=tmp_path)


def test_recorded_dataset_consumes_validated_manifest_and_audio_snapshot(
    tmp_path: Path,
) -> None:
    manifest, transfer_sha, _files = _write_transfer_bundle(
        tmp_path,
        dataset_ready=True,
    )
    receipt, receipt_sha, _summary = _write_bootstrap_receipt(
        tmp_path,
        transfer_sha256=transfer_sha,
    )
    data_cfg = _dataset_config(
        bootstrap_receipt=receipt.relative_to(tmp_path).as_posix(),
        bootstrap_receipt_sha256=receipt_sha,
    )
    dataset = RecordedANCDataset(
        "data/manifests/recorded_regrouped.jsonl",
        data_cfg,
        transfer_repo_root=tmp_path,
    )
    loaded = dataset._session(0)
    assert all(value.shape == (1024,) for value in loaded)
    assert dataset.describe()["transfer_manifest_sha256"] == transfer_sha

    target = tmp_path / "data/recorded/session-000/mics.wav"
    before = target.stat()
    forged = bytearray(target.read_bytes())
    forged[-1] ^= 1
    target.write_bytes(forged)
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert target.stat().st_size == before.st_size
    assert target.stat().st_mtime_ns == before.st_mtime_ns

    with pytest.raises(TransferContractError, match="snapshot 이후 변경"):
        dataset._session(0)


def test_recorded_dataset_and_resume_reject_post_validation_wav_tampering(
    tmp_path: Path,
) -> None:
    manifest, transfer_sha, _files = _write_transfer_bundle(
        tmp_path,
        dataset_ready=True,
    )
    data_cfg = _dataset_config(
        transfer_manifest=manifest.relative_to(tmp_path).as_posix(),
        transfer_manifest_sha256=transfer_sha,
    )
    dataset = RecordedANCDataset(
        "data/manifests/recorded_regrouped.jsonl",
        data_cfg,
        transfer_repo_root=tmp_path,
    )
    target = tmp_path / "data/recorded/session-000/source_aligned.wav"
    before = target.stat()
    forged = bytearray(target.read_bytes())
    forged[-1] ^= 1
    target.write_bytes(forged)
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(TransferContractError, match="snapshot 이후 변경"):
        dataset._session(0)
    with pytest.raises(TransferContractError, match="size/SHA 불일치"):
        bind_recorded_transfer_config(
            {
                "transfer_manifest": manifest.relative_to(tmp_path).as_posix(),
                "transfer_manifest_sha256": transfer_sha,
            },
            repo_root=tmp_path,
        )
