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
from deep_anc.data.holdout_contract import (
    read_regular_file_snapshot,
    snapshot_regular_tree_metadata,
)
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
    ignore = root / ".gitignore"
    ignore.write_text(
        "/data/\n/results/\n/assets/\n/.venv/\n",
        encoding="utf-8",
    )
    code_anchor = root / "receipt-code-anchor.txt"
    code_anchor.write_text("fixture code\n", encoding="utf-8")
    tracked_fixture_paths = [code_anchor.name, ignore.name]
    for relative in (
        "scripts/elice/public_archive_cache.py",
        "scripts/elice/pget.py",
    ):
        if (root / relative).is_file():
            tracked_fixture_paths.append(relative)
    subprocess.run(["git", "add", *tracked_fixture_paths], cwd=root, check=True)
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
    freeze.write_bytes(
        (
            "-e git+https://github.com/Roka-jsj/Deep-ANC.git@"
            f"{commit}#egg=deep_anc\n"
            "torch==2.5.1+cu121\n"
        ).encode("utf-8")
    )
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


_CACHE_ARCHIVE_IDS = (
    "dns_noise_000",
    "dns_noise_001",
    "dns_speech_000",
    "demand_dkitchen",
    "demand_dwashing",
    "demand_ooffice",
    "demand_ohallway",
    "demand_tmetro",
    "demand_tcar",
    "mimii_fan",
)
_CACHE_ARCHIVE_ORIGIN = {
    "dns_noise_000": (5_364_611_964, "data/raw/noise/shard000.tar.bz2"),
    "dns_noise_001": (5_357_916_291, "data/raw/noise/shard001.tar.bz2"),
    "dns_speech_000": (4_664_045_287, "data/raw/noise/speech000.tar.bz2"),
    "demand_dkitchen": (336_992_458, "data/raw/noise/demand/DKITCHEN_48k.zip"),
    "demand_dwashing": (306_101_499, "data/raw/noise/demand/DWASHING_48k.zip"),
    "demand_ooffice": (277_643_831, "data/raw/noise/demand/OOFFICE_48k.zip"),
    "demand_ohallway": (252_905_617, "data/raw/noise/demand/OHALLWAY_48k.zip"),
    "demand_tmetro": (367_513_573, "data/raw/noise/demand/TMETRO_48k.zip"),
    "demand_tcar": (373_520_251, "data/raw/noise/demand/TCAR_48k.zip"),
    "mimii_fan": (928_511_244, "data/raw/noise/mimii_fan.zip"),
}
_CACHE_ARCHIVE_OUTPUT = {
    "dns_noise_000": ("data/raw/noise/dns_fullband/noise000", 8_000, None),
    "dns_noise_001": ("data/raw/noise/dns_fullband/noise001", 8_000, None),
    "dns_speech_000": ("data/raw/noise/speech/speech000", 8_065, 8_000_834_860),
    "demand_dkitchen": ("data/raw/noise/demand/DKITCHEN", 16, 460_806_848),
    "demand_dwashing": ("data/raw/noise/demand/DWASHING", 16, 460_806_848),
    "demand_ooffice": ("data/raw/noise/demand/OOFFICE", 16, 460_806_848),
    "demand_ohallway": ("data/raw/noise/demand/OHALLWAY", 16, 460_806_848),
    "demand_tmetro": ("data/raw/noise/demand/TMETRO", 16, 460_806_848),
    "demand_tcar": ("data/raw/noise/demand/TCAR", 16, 460_806_848),
    "mimii_fan": ("data/raw/noise/machine/fan", 3_600, 1_152_158_400),
}


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_non_null_cache_v3_binding(
    root: Path,
    *,
    commit: str = "a" * 40,
    manifest_sha: str = "b" * 64,
    external_root: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Write a minimal but semantically complete ten-archive schema-v3 chain."""

    script = root / "scripts/elice/public_archive_cache.py"
    pget = root / "scripts/elice/pget.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_bytes(b"fixture public archive cache entry\n")
    pget.write_bytes(b"fixture pget entry\n")
    script_sha = hashlib.sha256(script.read_bytes()).hexdigest()
    pget_sha = hashlib.sha256(pget.read_bytes()).hexdigest()

    rows: list[dict[str, object]] = []
    for archive_id in _CACHE_ARCHIVE_IDS:
        prefix, count, declared_bytes = _CACHE_ARCHIVE_OUTPUT[archive_id]
        total = count if declared_bytes is None else declared_bytes
        base_size, remainder = divmod(total, count)
        for index in range(count):
            size = base_size + (1 if index < remainder else 0)
            relative = f"{prefix}/{index:05d}.wav"
            rows.append(
                {
                    "archive_id": archive_id,
                    "path": relative,
                    "sha256": hashlib.sha256(
                        f"{archive_id}:{index}:{size}".encode()
                    ).hexdigest(),
                    "size": size,
                }
            )
    rows.sort(key=lambda row: str(row["path"]))
    output_bytes = sum(int(row["size"]) for row in rows)
    content_projection = hashlib.sha256()
    path_size_projection = hashlib.sha256()
    for row in rows:
        content_projection.update(
            _canonical_json_bytes(
                {
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "size": row["size"],
                }
            )
        )
        path_size_projection.update(
            _canonical_json_bytes({"path": row["path"], "size": row["size"]})
        )

    archive_digests = {
        archive_id: hashlib.sha256(archive_id.encode()).hexdigest()
        for archive_id in _CACHE_ARCHIVE_IDS
    }
    if external_root is not None:
        entries: list[dict[str, object]] = []
        for archive_id in _CACHE_ARCHIVE_IDS:
            archive_rows = [row for row in rows if row["archive_id"] == archive_id]
            archive_bytes = sum(int(row["size"]) for row in archive_rows)
            output_digest = hashlib.sha256()
            for row in archive_rows:
                output_digest.update(
                    _canonical_json_bytes(
                        {
                            "path": row["path"],
                            "sha256": row["sha256"],
                            "size": row["size"],
                        }
                    )
                )
            archive_size, canonical_target = _CACHE_ARCHIVE_ORIGIN[archive_id]
            archive_sha = archive_digests[archive_id]
            filename = f"fixture-{archive_id}.archive"
            entries.append(
                {
                    "archive_format": "fixture",
                    "archive_id": archive_id,
                    "archive_sha256": archive_sha,
                    "archive_size": archive_size,
                    "cache_path": (
                        f"archives/v1/{archive_id}/bytes_{archive_size}/"
                        f"sha256_{archive_sha}/{filename}"
                    ),
                    "canonical_target": canonical_target,
                    "corpus": "fixture",
                    "filename": filename,
                    "member_inventory_sha256": hashlib.sha256(
                        f"members:{archive_id}".encode()
                    ).hexdigest(),
                    "member_content_inventory_sha256": hashlib.sha256(
                        f"content:{archive_id}".encode()
                    ).hexdigest(),
                    "member_prefix": "fixture/",
                    "provider_checksum": None,
                    "provider_checksum_kind": "none",
                    "provider_etag": None,
                    "regular_file_bytes": archive_bytes,
                    "regular_file_count": len(archive_rows),
                    "source_url": f"https://example.invalid/{archive_id}",
                    "output_content_inventory_sha256": output_digest.hexdigest(),
                    "wav_bytes": archive_bytes,
                    "wav_count": len(archive_rows),
                }
            )
        external_root.mkdir(parents=True)
        external_manifest = external_root / "manifest.json"
        external_manifest.write_bytes(
            _canonical_json_bytes(
                {
                    "archive_count": 10,
                    "archives": entries,
                    "authority": "transport_acceleration_only_not_raw_or_training_authority",
                    "excluded_corpora": [
                        "esc50",
                        "fma_small",
                        "fma_metadata",
                        "librispeech",
                    ],
                    "kind": "deep_anc_public_archive_cache",
                    "publisher_commit": commit,
                    "publisher_entry_script_sha256": script_sha,
                    "publisher_pget_sha256": pget_sha,
                    "schema_version": 1,
                }
            )
        )
        manifest_sha = hashlib.sha256(external_manifest.read_bytes()).hexdigest()

    stem = f"{manifest_sha}.{commit}"
    marker = root / "data/raw/noise/.archive_cache_consumptions"
    origin_dir = root / "data/raw/noise/.archive_cache_origins"
    marker.mkdir(parents=True, exist_ok=True)
    origin_dir.mkdir(parents=True, exist_ok=True)
    intent_relative = (
        f"data/raw/noise/.archive_cache_consumptions/consume_intent.{stem}.json"
    )
    inventory_relative = (
        f"data/raw/noise/.archive_cache_consumptions/consume_inventory.{stem}.json"
    )
    completion_relative = (
        f"data/raw/noise/.archive_cache_consumptions/consume_complete.{stem}.json"
    )
    origin_relative = (
        f"data/raw/noise/.archive_cache_origins/archive_cache_origin.{stem}.json"
    )

    intent = {
        "archive_count": 10,
        "archive_manifest_sha256": manifest_sha,
        "authority": "cache_transport_state_only_requires_exact_raw_and_decoder_authority",
        "expected_output_bytes": output_bytes,
        "expected_output_count": len(rows),
        "expected_output_path_size_inventory_sha256": path_size_projection.hexdigest(),
        "kind": "deep_anc_archive_cache_consumption_intent",
        "publisher_commit": commit,
        "restorer_entry_script_sha256": script_sha,
        "restorer_pget_sha256": pget_sha,
        "schema_version": 1,
        "state": "in_progress_or_completed_requires_matching_external_anchors",
    }
    inventory = {
        "archive_manifest_sha256": manifest_sha,
        "authority": "cache_transport_state_only_requires_exact_raw_and_decoder_authority",
        "kind": "deep_anc_archive_cache_consumed_member_inventory",
        "output_bytes": output_bytes,
        "output_count": len(rows),
        "publisher_commit": commit,
        "rows": rows,
        "schema_version": 1,
    }
    origin = {
        "archives": [
            {
                "archive_id": archive_id,
                "archive_sha256": archive_digests[archive_id],
                "archive_size": _CACHE_ARCHIVE_ORIGIN[archive_id][0],
                "canonical_target": _CACHE_ARCHIVE_ORIGIN[archive_id][1],
            }
            for archive_id in _CACHE_ARCHIVE_IDS
        ],
        "authority": "cache_origin_only_not_official_raw_or_training_authority",
        "kind": "deep_anc_archive_cache_origin_receipt",
        "manifest_sha256": manifest_sha,
        "publisher_commit": commit,
        "restorer_entry_script_sha256": script_sha,
        "restorer_pget_sha256": pget_sha,
        "schema_version": 1,
    }
    for relative, payload in (
        (intent_relative, intent),
        (inventory_relative, inventory),
        (origin_relative, origin),
    ):
        (root / relative).write_bytes(_canonical_json_bytes(payload))
    intent_sha = hashlib.sha256((root / intent_relative).read_bytes()).hexdigest()
    inventory_sha = hashlib.sha256((root / inventory_relative).read_bytes()).hexdigest()
    origin_sha = hashlib.sha256((root / origin_relative).read_bytes()).hexdigest()
    completion = {
        "archive_manifest_sha256": manifest_sha,
        "authority": "cache_transport_state_only_requires_exact_raw_and_decoder_authority",
        "intent_path": intent_relative,
        "intent_sha256": intent_sha,
        "kind": "deep_anc_archive_cache_consumption_completion",
        "member_inventory_path": inventory_relative,
        "member_inventory_sha256": inventory_sha,
        "origin_receipt_path": origin_relative,
        "origin_receipt_sha256": origin_sha,
        "output_bytes": output_bytes,
        "output_count": len(rows),
        "output_path_size_sha256_inventory_sha256": content_projection.hexdigest(),
        "publisher_commit": commit,
        "schema_version": 1,
        "state": "held_fd_consume_complete_pending_exact_raw_and_decoder_authority",
    }
    (root / completion_relative).write_bytes(_canonical_json_bytes(completion))

    decoder = {
        "inventory": [
            {
                "content_sha256": row["sha256"],
                "content_size": row["size"],
                "relative_path": row["path"],
            }
            for row in rows
        ],
        "schema_version": 1,
        "status": "complete",
    }
    decoder["audit_sha256"] = hashlib.sha256(
        json.dumps(
            decoder,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    decoder_path = root / "results/provenance/decoder_audit.json"
    decoder_path.parent.mkdir(parents=True, exist_ok=True)
    decoder_path.write_bytes(_canonical_json_bytes(decoder))

    binding = {
        "archive_manifest_sha256": manifest_sha,
        "authority": "cache_transport_state_bound_to_exact_decoder_inventory",
        "completion_path": completion_relative,
        "completion_sha256": hashlib.sha256(
            (root / completion_relative).read_bytes()
        ).hexdigest(),
        "member_inventory_path": inventory_relative,
        "member_inventory_sha256": inventory_sha,
        "output_path_size_sha256_inventory_sha256": content_projection.hexdigest(),
        "decoder_audit_path": "results/provenance/decoder_audit.json",
        "decoder_audit_file_sha256": hashlib.sha256(decoder_path.read_bytes()).hexdigest(),
        "decoder_audit_semantic_sha256": decoder["audit_sha256"],
        "decoder_cache_projection_sha256": content_projection.hexdigest(),
    }
    receipt = {
        "schema_version": 3,
        "expected_commit": commit,
        "archive_cache_consumption": binding,
    }
    return receipt, binding


def test_schema_v3_distinguishes_no_cache_from_cache_marker_residue(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": 3,
        "expected_commit": "a" * 40,
        "archive_cache_consumption": None,
    }
    assert (
        transfer_contract._validate_archive_cache_bootstrap_binding(  # noqa: SLF001
            tmp_path, payload
        )
        is None
    )

    marker = tmp_path / "data/raw/noise/.archive_cache_consumptions"
    marker.mkdir(parents=True)
    with pytest.raises(TransferContractError, match="no-cache receipt"):
        transfer_contract._validate_archive_cache_bootstrap_binding(  # noqa: SLF001
            tmp_path, payload
        )


def test_schema_v3_rejects_unbound_cache_object_before_training(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": 3,
        "expected_commit": "a" * 40,
        "archive_cache_consumption": {},
    }
    with pytest.raises(TransferContractError, match="exact field set"):
        transfer_contract._validate_archive_cache_bootstrap_binding(  # noqa: SLF001
            tmp_path, payload
        )


def test_schema_v3_non_null_cache_chain_passes_transfer_binding_and_detects_tamper(
    tmp_path: Path,
) -> None:
    payload, binding = _write_non_null_cache_v3_binding(tmp_path)

    assert transfer_contract._validate_archive_cache_bootstrap_binding(  # noqa: SLF001
        tmp_path, payload
    ) == binding

    decoder = tmp_path / "results/provenance/decoder_audit.json"
    original = decoder.read_bytes()
    decoder.write_bytes(original.replace(b'"status":"complete"', b'"status":"tampered"'))
    with pytest.raises(TransferContractError, match="SHA가 bootstrap receipt"):
        transfer_contract._validate_archive_cache_bootstrap_binding(  # noqa: SLF001
            tmp_path, payload
        )


def test_schema_v3_rejects_resealed_extra_cache_root_decoder_wav(
    tmp_path: Path,
) -> None:
    payload, binding = _write_non_null_cache_v3_binding(tmp_path)
    decoder_path = tmp_path / "results/provenance/decoder_audit.json"
    decoder = json.loads(decoder_path.read_text(encoding="utf-8"))
    decoder["inventory"].append(
        {
            "content_sha256": "f" * 64,
            "content_size": 17,
            "relative_path": "data/raw/noise/demand/DKITCHEN/unexpected.wav",
        }
    )
    decoder["audit_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in decoder.items() if key != "audit_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    decoder_path.write_bytes(_canonical_json_bytes(decoder))
    binding["decoder_audit_file_sha256"] = hashlib.sha256(
        decoder_path.read_bytes()
    ).hexdigest()
    binding["decoder_audit_semantic_sha256"] = decoder["audit_sha256"]

    with pytest.raises(TransferContractError, match="cache raw exact-set"):
        transfer_contract._validate_archive_cache_bootstrap_binding(  # noqa: SLF001
            tmp_path, payload
        )


def test_schema_v3_non_null_cache_receipt_passes_full_training_config_bind(
    tmp_path: Path,
) -> None:
    _manifest, transfer_sha, _files = _write_transfer_bundle(tmp_path)
    script = tmp_path / "scripts/elice/public_archive_cache.py"
    pget = tmp_path / "scripts/elice/pget.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_bytes(b"fixture public archive cache entry\n")
    pget.write_bytes(b"fixture pget entry\n")
    receipt, _old_sha, _summary = _write_bootstrap_receipt(
        tmp_path, transfer_sha256=transfer_sha
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    commit = str(payload["expected_commit"])
    external_root = tmp_path.parent / f"{tmp_path.name}-external-cache"
    cache_receipt, binding = _write_non_null_cache_v3_binding(
        tmp_path, commit=commit, external_root=external_root
    )
    contract_sha = "4" * 64
    coverage = (
        tmp_path
        / "results/data_audit/recorded_subband_coverage"
        / f"{contract_sha}.json"
    )
    coverage.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "evidence_sha256": "5" * 64,
        "manifest": {"sha256": "6" * 64},
        "training_timing_contract_sha256": "7" * 64,
        "coverage_contract_sha256": contract_sha,
        "all_requested_splits_pass": False,
    }
    coverage.write_text(json.dumps(report_payload, sort_keys=True) + "\n")
    payload.update(
        {
            "schema_version": 3,
            "archive_cache_consumption": cache_receipt[
                "archive_cache_consumption"
            ],
            "recorded_subband_coverage": {
                "path": coverage.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(coverage.read_bytes()).hexdigest(),
                "evidence_sha256": report_payload["evidence_sha256"],
                "manifest_sha256": report_payload["manifest"]["sha256"],
                "training_timing_contract_sha256": report_payload[
                    "training_timing_contract_sha256"
                ],
                "coverage_contract_sha256": contract_sha,
                "all_requested_splits_pass": False,
            },
        }
    )
    receipt.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    data_cfg = {
        "bootstrap_receipt": receipt.relative_to(tmp_path).as_posix(),
        "bootstrap_receipt_sha256": receipt_sha,
        "archive_cache_root": str(external_root.absolute()),
        "archive_cache_manifest": str((external_root / "manifest.json").absolute()),
        "archive_cache_manifest_sha256": binding["archive_manifest_sha256"],
    }

    with pytest.raises(TransferContractError, match="external anchor가 모두 필요"):
        bind_recorded_transfer_config(
            {
                "bootstrap_receipt": receipt.relative_to(tmp_path).as_posix(),
                "bootstrap_receipt_sha256": receipt_sha,
            },
            repo_root=tmp_path,
        )

    snapshot = bind_recorded_transfer_config(data_cfg, repo_root=tmp_path)

    assert snapshot.bootstrap_receipt is not None
    assert snapshot.bootstrap_receipt.sha256 == receipt_sha
    assert data_cfg["transfer_manifest_sha256"] == transfer_sha
    assert (
        transfer_contract._validate_archive_cache_bootstrap_binding(  # noqa: SLF001
            tmp_path, payload
        )
        == binding
    )


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


def test_builder_rotates_schema_v1_to_v2_with_content_addressed_history(
    tmp_path: Path, monkeypatch
) -> None:
    builder = _load_builder()
    manifest = tmp_path / builder.OUTPUT
    manifest.parent.mkdir(parents=True)
    old = b'{"schema_version":1}\n'
    new = b'{"schema_version":2}\n'
    old_sha = hashlib.sha256(old).hexdigest()
    manifest.write_bytes(old)

    def validate(path, *, repo_root, expected_sha256):
        payload = Path(path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        assert Path(repo_root) == tmp_path
        return {}

    monkeypatch.setattr(builder, "validate_transfer_manifest", validate)
    builder._publish_no_replace(
        manifest,
        new,
        repo_root=tmp_path,
        rotate_existing_sha256=old_sha,
    )

    history = (
        tmp_path
        / builder.TRANSFER_HISTORY_DIR
        / f"elice_transfer_manifest.{old_sha}.json"
    )
    assert manifest.read_bytes() == new
    assert history.read_bytes() == old


def test_builder_rotation_requires_exact_old_sha_and_v1_to_v2_transition(
    tmp_path: Path, monkeypatch
) -> None:
    builder = _load_builder()
    manifest = tmp_path / builder.OUTPUT
    manifest.parent.mkdir(parents=True)
    old = b'{"schema_version":1}\n'
    manifest.write_bytes(old)
    old_sha = hashlib.sha256(old).hexdigest()
    monkeypatch.setattr(builder, "validate_transfer_manifest", lambda *a, **k: {})

    with pytest.raises(TransferContractError, match="회전 anchor"):
        builder._publish_no_replace(
            manifest,
            b'{"schema_version":2}\n',
            repo_root=tmp_path,
            rotate_existing_sha256="0" * 64,
        )
    assert manifest.read_bytes() == old

    with pytest.raises(TransferContractError, match="v1→v2"):
        builder._publish_no_replace(
            manifest,
            b'{"schema_version":1,"changed":true}\n',
            repo_root=tmp_path,
            rotate_existing_sha256=old_sha,
        )
    assert manifest.read_bytes() == old
    assert not (tmp_path / builder.TRANSFER_HISTORY_DIR).exists()


def test_builder_rotation_keeps_v1_canonical_if_new_bundle_validation_fails(
    tmp_path: Path, monkeypatch
) -> None:
    builder = _load_builder()
    manifest = tmp_path / builder.OUTPUT
    manifest.parent.mkdir(parents=True)
    old = b'{"schema_version":1}\n'
    new = b'{"schema_version":2}\n'
    old_sha = hashlib.sha256(old).hexdigest()
    manifest.write_bytes(old)

    def validate(path, *, repo_root, expected_sha256):
        del repo_root, expected_sha256
        if Path(path).read_bytes() == new:
            raise TransferContractError("injected new schema validation failure")
        return {}

    monkeypatch.setattr(builder, "validate_transfer_manifest", validate)
    with pytest.raises(TransferContractError, match="injected"):
        builder._publish_no_replace(
            manifest,
            new,
            repo_root=tmp_path,
            rotate_existing_sha256=old_sha,
        )

    history = (
        tmp_path
        / builder.TRANSFER_HISTORY_DIR
        / f"elice_transfer_manifest.{old_sha}.json"
    )
    assert manifest.read_bytes() == old
    assert history.read_bytes() == old


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


def _schema_v2_binding_summary(root: Path) -> tuple[dict[str, object], dict[str, str]]:
    paths = {
        "transfer": "data/manifests/elice_transfer_manifest.json",
        "manifest": "data/manifests/recorded_regrouped_101.jsonl",
        "generation": "data/manifests/recorded_generations/fixture/generation.json",
        "calibration": "data/manifests/recorded_level_calibration/fixture.json",
        "recorded": "data/recorded/session-000/session.json",
    }
    for name, relative in paths.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-{name}\n".encode("utf-8"))

    def snapshot(name: str):
        return read_regular_file_snapshot(
            root / paths[name], root=root, label=f"fixture {name}"
        )

    recorded_snapshot = snapshot("recorded")
    return (
        {
            "recorded_aggregate_sha256": "a" * 64,
            "canonical_holdout_sha256": "b" * 64,
            "_validated_transfer_manifest_snapshot": snapshot("transfer"),
            "_validated_recorded_manifest_snapshot": snapshot("manifest"),
            "_validated_recorded_generation_snapshot": snapshot("generation"),
            "_validated_recorded_generation_summary": {"fixture": True},
            "_validated_recorded_level_calibration_snapshot": snapshot(
                "calibration"
            ),
            "_validated_recorded_file_snapshots": {
                paths["recorded"]: recorded_snapshot
            },
        },
        paths,
    )


def test_schema_v2_binder_materializes_transfer_validated_level_calibration(
    tmp_path: Path, monkeypatch
) -> None:
    summary, paths = _schema_v2_binding_summary(tmp_path)
    transfer_snapshot = summary["_validated_transfer_manifest_snapshot"]
    monkeypatch.setattr(
        transfer_contract,
        "validate_transfer_manifest",
        lambda *args, **kwargs: summary,
    )
    data_cfg = {
        "transfer_manifest": paths["transfer"],
        "transfer_manifest_sha256": transfer_snapshot.sha256,
        "recorded_generation": None,
        "recorded_generation_sha256": None,
        "recorded_level_calibration": None,
        "recorded_level_calibration_sha256": None,
    }

    snapshot = bind_recorded_transfer_config(data_cfg, repo_root=tmp_path)

    calibration_snapshot = summary[
        "_validated_recorded_level_calibration_snapshot"
    ]
    assert snapshot.recorded_level_calibration == calibration_snapshot
    assert data_cfg["recorded_level_calibration"] == paths["calibration"]
    assert data_cfg["recorded_level_calibration_sha256"] == (
        calibration_snapshot.sha256
    )


@pytest.mark.parametrize(
    ("declared_path", "declared_sha", "message"),
    [
        ("data/manifests/recorded_level_calibration/fixture.json", None, "둘 다"),
        (None, "0" * 64, "둘 다"),
        ("data/manifests/recorded_level_calibration/other.json", "0" * 64, "검증값"),
    ],
)
def test_schema_v2_binder_rejects_partial_or_mismatched_level_calibration(
    tmp_path: Path,
    monkeypatch,
    declared_path: str | None,
    declared_sha: str | None,
    message: str,
) -> None:
    summary, paths = _schema_v2_binding_summary(tmp_path)
    transfer_snapshot = summary["_validated_transfer_manifest_snapshot"]
    monkeypatch.setattr(
        transfer_contract,
        "validate_transfer_manifest",
        lambda *args, **kwargs: summary,
    )
    data_cfg = {
        "transfer_manifest": paths["transfer"],
        "transfer_manifest_sha256": transfer_snapshot.sha256,
        "recorded_generation": None,
        "recorded_generation_sha256": None,
        "recorded_level_calibration": declared_path,
        "recorded_level_calibration_sha256": declared_sha,
    }

    with pytest.raises(TransferContractError, match=message):
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


def test_recorded_config_binder_rejects_resealed_stale_editable_freeze(
    tmp_path: Path,
) -> None:
    _manifest, transfer_sha, _files = _write_transfer_bundle(tmp_path)
    receipt, _receipt_sha, _summary = _write_bootstrap_receipt(
        tmp_path,
        transfer_sha256=transfer_sha,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    current = str(payload["expected_commit"])
    stale = ("0" if current[0] != "0" else "1") + current[1:]
    freeze = tmp_path / ".venv/environment-freeze.txt"
    freeze.write_bytes(freeze.read_bytes().replace(current.encode(), stale.encode()))
    payload["environment"]["freeze_receipt_sha256"] = hashlib.sha256(
        freeze.read_bytes()
    ).hexdigest()
    receipt.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    resealed_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()

    with pytest.raises(TransferContractError, match="source 결속 실패"):
        bind_recorded_transfer_config(
            {
                "bootstrap_receipt": receipt.relative_to(tmp_path).as_posix(),
                "bootstrap_receipt_sha256": resealed_sha,
            },
            repo_root=tmp_path,
        )


def test_bootstrap_binder_rejects_dirty_source_after_clean_receipt(
    tmp_path: Path,
) -> None:
    _manifest, transfer_sha, _files = _write_transfer_bundle(tmp_path)
    receipt, receipt_sha, _summary = _write_bootstrap_receipt(
        tmp_path,
        transfer_sha256=transfer_sha,
    )
    (tmp_path / "receipt-code-anchor.txt").write_text(
        "mutated after bootstrap\n", encoding="utf-8"
    )
    with pytest.raises(TransferContractError, match="clean exact source"):
        bind_recorded_transfer_config(
            {
                "bootstrap_receipt": receipt.relative_to(tmp_path).as_posix(),
                "bootstrap_receipt_sha256": receipt_sha,
            },
            repo_root=tmp_path,
        )


def test_bootstrap_binder_allows_ignored_runtime_cache_but_not_code_injection(
    tmp_path: Path,
) -> None:
    _manifest, transfer_sha, _files = _write_transfer_bundle(tmp_path)
    receipt, receipt_sha, _summary = _write_bootstrap_receipt(
        tmp_path,
        transfer_sha256=transfer_sha,
    )
    cache = tmp_path / "scripts/__pycache__/bootstrap.cpython-310.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"ordinary runtime cache")
    with (tmp_path / ".git/info/exclude").open("a", encoding="utf-8") as handle:
        handle.write("__pycache__/\n*.pyc\n/scripts/injected.py\n")
    config = {
        "bootstrap_receipt": receipt.relative_to(tmp_path).as_posix(),
        "bootstrap_receipt_sha256": receipt_sha,
    }
    bind_recorded_transfer_config(config, repo_root=tmp_path)

    (tmp_path / "scripts/injected.py").write_text("raise SystemExit('forged')\n")
    with pytest.raises(TransferContractError, match="clean exact source"):
        bind_recorded_transfer_config(
            {
                "bootstrap_receipt": receipt.relative_to(tmp_path).as_posix(),
                "bootstrap_receipt_sha256": receipt_sha,
            },
            repo_root=tmp_path,
        )


def test_bootstrap_receipt_rejects_resealed_coverage_report_forgery(
    tmp_path: Path,
) -> None:
    _manifest, transfer_sha, _files = _write_transfer_bundle(tmp_path)
    receipt, _old_sha, _summary = _write_bootstrap_receipt(
        tmp_path,
        transfer_sha256=transfer_sha,
    )
    contract_sha = "4" * 64
    coverage = (
        tmp_path
        / "results/data_audit/recorded_subband_coverage"
        / f"{contract_sha}.json"
    )
    coverage.parent.mkdir(parents=True)
    report_payload = {
        "evidence_sha256": "5" * 64,
        "manifest": {"sha256": "6" * 64},
        "training_timing_contract_sha256": "7" * 64,
        "coverage_contract_sha256": contract_sha,
        "all_requested_splits_pass": False,
    }
    coverage.write_text(json.dumps(report_payload, sort_keys=True) + "\n")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    payload["recorded_subband_coverage"] = {
        "path": coverage.relative_to(tmp_path).as_posix(),
        "sha256": hashlib.sha256(coverage.read_bytes()).hexdigest(),
        "evidence_sha256": report_payload["evidence_sha256"],
        "manifest_sha256": report_payload["manifest"]["sha256"],
        "training_timing_contract_sha256": report_payload[
            "training_timing_contract_sha256"
        ],
        "coverage_contract_sha256": contract_sha,
        "all_requested_splits_pass": False,
    }
    receipt.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    valid = bind_recorded_transfer_config(
        {
            "bootstrap_receipt": receipt.relative_to(tmp_path).as_posix(),
            "bootstrap_receipt_sha256": receipt_sha,
        },
        repo_root=tmp_path,
    )
    assert valid.recorded_subband_coverage_receipt is not None

    # 공격자가 report payload와 자체 evidence SHA를 함께 다시 봉인해도, 별도 채널로
    # 전달된 bootstrap receipt SHA 아래 file SHA는 바꿀 수 없다.
    report_payload["all_requested_splits_pass"] = True
    report_payload["evidence_sha256"] = "8" * 64
    coverage.write_text(json.dumps(report_payload, sort_keys=True) + "\n")
    with pytest.raises(TransferContractError, match="coverage report가 변경"):
        bind_recorded_transfer_config(
            {
                "bootstrap_receipt": receipt.relative_to(tmp_path).as_posix(),
                "bootstrap_receipt_sha256": receipt_sha,
            },
            repo_root=tmp_path,
        )


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
