from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from deep_anc.data.public_lineage import canonical_json_sha256
from deep_anc.data.transfer_contract import TransferContractError
from deep_anc.data import stage2_pretrain_data_issuer as issuer_module
from deep_anc.data.stage2_pretrain_data_issuer import (
    REQUIRED_CANONICAL_TAGS,
    Stage2PretrainDataIssueError,
    build_coverage_receipt,
    build_lineage_receipt,
    build_manifest_items,
    build_transfer_bootstrap_receipt,
    canonical_json_bytes,
    publish_payloads_noreplace,
    seal_published_payloads_noreplace,
    sha256_bytes,
    stage2_recorded_public_intersection,
    validate_elice_bootstrap_inputs,
)
from deep_anc.dsp.stage2_2khz_contract import (
    STAGE2_2KHZ_SOURCE_FAMILIES,
    Stage2TwoKilohertzContract,
)
from deep_anc.train.stage2_2khz_pretrain_admission import (
    STAGE2_PUBLIC_MANIFEST_BUNDLE_SCHEMA,
    STAGE2_TRANSFER_BOOTSTRAP_RECEIPT_SCHEMA,
    validate_stage2_pretrain_data_candidate,
)


@pytest.fixture(autouse=True)
def _fixture_stage2_physical_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """이 파일의 public-data fixture는 physical P/S bytes를 만들지 않는다.

    실제 role/authority 재검증은 transfer-contract fixture에서 별도로 수행한다.
    여기서는 public manifest/cache regression이 actual 4 GiB corpus를 요구하지 않도록
    Stage-2-only helper 경계를 typed reference로 대체한다.
    """

    monkeypatch.setattr(
        issuer_module,
        "require_stage2_2khz_physical_transfer_manifest",
        lambda *_args, **_kwargs: {
            "plant_binding": {
                "path": "results/stage2_2khz_ps_v3/plant_binding.json",
                "sha256": "a" * 64,
            },
            "physical_authority": {
                "path": "authority/stage2_2khz_physical.json",
                "sha256": "b" * 64,
            },
        },
    )


def _write_holdout(root: Path) -> None:
    clips = [
        {
            "family": "speech",
            "clip": "held-recorded.wav",
            "content_sha256": "f" * 64,
            "lineage_keys": ["recorded:test"],
        }
    ]
    payload = {
        "clip_lineage": {
            "schema_version": 1,
            "metadata": {
                "librispeech_chapters": {
                    "path": "data/raw/speech/LibriSpeech/CHAPTERS.TXT",
                    "sha256": "1" * 64,
                    "size": 1,
                },
                "fma_tracks": {
                    "path": "data/raw/music/fma_metadata/tracks.csv",
                    "sha256": "2" * 64,
                    "size": 1,
                },
                "esc50": {
                    "path": "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv",
                    "sha256": "3" * 64,
                    "size": 1,
                },
            },
            "clips": clips,
            "clips_sha256": canonical_json_sha256(clips),
        },
        "families": {"speech": ["held-recorded.wav"]},
    }
    path = root / "data/manifests/recorded_holdout.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical_json_bytes(payload))


def _write_no_cache_elice_bootstrap(root: Path, *, commit: str) -> dict[str, str]:
    holdout = root / "data/manifests/recorded_holdout.json"
    transfer = root / "data/manifests/elice_transfer_manifest.json"
    freeze = root / ".venv/environment-freeze.txt"
    transfer.write_bytes(canonical_json_bytes({"fixture": "transfer"}))
    freeze.parent.mkdir(parents=True, exist_ok=True)
    freeze.write_text("fixture environment freeze\n", encoding="utf-8")
    payload = {
        "schema_version": 3,
        "expected_commit": commit,
        "canonical_holdout": {
            "path": "data/manifests/recorded_holdout.json",
            "sha256": sha256_bytes(holdout.read_bytes()),
        },
        "transfer_manifest": {
            "path": "data/manifests/elice_transfer_manifest.json",
            "sha256": sha256_bytes(transfer.read_bytes()),
        },
        "recorded_aggregate_sha256": "4" * 64,
        "archive_cache_consumption": None,
        "recorded_subband_coverage": {"fixture": "not-consumed-here"},
        "environment": {
            "freeze_receipt": ".venv/environment-freeze.txt",
            "freeze_receipt_sha256": sha256_bytes(freeze.read_bytes()),
            "torch_version": "2.5.1+cu121",
            "torch_cuda": "12.1",
        },
    }
    receipt = root / "data/manifests/elice_bootstrap_receipt.json"
    receipt.write_bytes(canonical_json_bytes(payload))
    return {
        "path": "data/manifests/elice_bootstrap_receipt.json",
        "sha256": sha256_bytes(receipt.read_bytes()),
    }


def _broadband(index: int, frames: int = 16384) -> np.ndarray:
    rng = np.random.default_rng(1000 + index)
    return (0.05 * rng.standard_normal(frames)).astype(np.float32)


def _validated_entries(root: Path) -> dict[str, list[dict[str, object]]]:
    rows: dict[str, list[dict[str, object]]] = {
        tag: [] for tag in REQUIRED_CANONICAL_TAGS
    }
    # 네 family 각각 train/val/test×4 component. environment의 12개를 세 public
    # corpus에 나눠 실제 six-manifest input을 모두 소비한다.
    family_tags = {
        "speech": ("speech",),
        "music": ("music",),
        "machine": ("machine",),
        "environment": ("demand", "dns_fullband", "esc50", "esc50"),
    }
    counter = 0
    for family in STAGE2_2KHZ_SOURCE_FAMILIES:
        tags = family_tags[family]
        for split in ("train", "val", "test"):
            for component_index in range(4):
                tag = tags[component_index % len(tags)]
                directory = root / "data/raw/public" / tag / split
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{family}_{split}_{component_index}.wav"
                sf.write(path, _broadband(counter), 48_000, subtype="PCM_16")
                content = path.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                rows[tag].append(
                    {
                        "path": str(path),
                        "content_sha256": digest,
                        "content_size": len(content),
                        "split": split,
                        "group_id": f"public-lineage-{counter:064x}",
                        "lineage_schema": "public-corpus-lineage/v2",
                        "lineage_keys": [f"fixture:{counter}"],
                        "sample_rate": 48_000,
                    }
                )
                counter += 1
    assert all(rows[tag] for tag in REQUIRED_CANONICAL_TAGS)
    return rows


def _candidate_payloads(root: Path) -> dict[str, dict[str, object]]:
    _write_holdout(root)
    commit = "1" * 40
    elice_bootstrap_ref = _write_no_cache_elice_bootstrap(root, commit=commit)
    plant_sha = "a" * 64
    items = build_manifest_items(root, _validated_entries(root))
    contract = Stage2TwoKilohertzContract.canonical()
    manifest: dict[str, object] = {
        "schema": STAGE2_PUBLIC_MANIFEST_BUNDLE_SCHEMA,
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract": {
            "id": contract.contract_id,
            "sha256": contract.digest(),
        },
        "required_source_families": list(STAGE2_2KHZ_SOURCE_FAMILIES),
        "required_splits": ["train", "val", "test"],
        "recorded_artifacts_required_for_pretrain": False,
        "test_split_for_checkpoint_selection_allowed": False,
        "source_inventory_commit_sha": commit,
        "items": items,
    }
    manifest_sha = sha256_bytes(canonical_json_bytes(manifest))
    lineage = build_lineage_receipt(
        root,
        items=items,
        manifest_bundle_sha256=manifest_sha,
        source_inventory_commit_sha=commit,
    )
    coverage = build_coverage_receipt(
        root,
        items=items,
        manifest_bundle_sha256=manifest_sha,
        plant_binding_file_sha256=plant_sha,
        source_inventory_commit_sha=commit,
    )
    transfer = {
        "schema": STAGE2_TRANSFER_BOOTSTRAP_RECEIPT_SCHEMA,
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract_sha256": contract.digest(),
        "manifest_bundle_sha256": manifest_sha,
        "elice_bootstrap_receipt": elice_bootstrap_ref,
        "existing_instance_cache_reused": False,
        "all_declared_source_bytes_rehashed": True,
        "stale_run_or_checkpoint_auto_resume_allowed": False,
        "scratch_new_run_directory_required": True,
        "source_inventory_commit_sha": commit,
    }
    return {
        "manifest_bundle.json": manifest,
        "lineage_receipt.json": lineage,
        "frequency_coverage_receipt.json": coverage,
        "transfer_bootstrap_receipt.json": transfer,
        "issuer_receipt.json": {
            "schema": "fixture-stage2-issuer",
            "source_inventory_commit_sha": commit,
        },
    }


def test_no_cache_bootstrap_cannot_claim_reused_cache(tmp_path: Path) -> None:
    _write_holdout(tmp_path)
    commit = "1" * 40
    ref = _write_no_cache_elice_bootstrap(tmp_path, commit=commit)
    inputs = validate_elice_bootstrap_inputs(
        tmp_path,
        source_inventory_commit_sha=commit,
        expected_bootstrap_receipt_sha256=ref["sha256"],
    )
    assert inputs["archive_cache_reused"] is False
    transfer = build_transfer_bootstrap_receipt(
        manifest_bundle_sha256="a" * 64,
        source_inventory_commit_sha=commit,
        bootstrap_inputs=inputs,
    )
    assert transfer["elice_bootstrap_receipt"] == ref
    assert transfer["existing_instance_cache_reused"] is False

    (tmp_path / "data/raw/noise/.archive_cache_consumptions").mkdir(parents=True)
    with pytest.raises(
        Stage2PretrainDataIssueError, match="archive-cache provenance"
    ):
        validate_elice_bootstrap_inputs(
            tmp_path,
            source_inventory_commit_sha=commit,
            expected_bootstrap_receipt_sha256=ref["sha256"],
        )


def test_canonical_stage2_bootstrap_rejects_legacy_transfer_without_physical_v3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_holdout(tmp_path)
    commit = "1" * 40
    ref = _write_no_cache_elice_bootstrap(tmp_path, commit=commit)

    def reject(*_args: object, **_kwargs: object) -> object:
        raise TransferContractError("legacy transfer schema v1")

    monkeypatch.setattr(
        issuer_module,
        "require_stage2_2khz_physical_transfer_manifest",
        reject,
    )
    with pytest.raises(
        Stage2PretrainDataIssueError,
        match="physical P/S schema-v3 transfer",
    ):
        validate_elice_bootstrap_inputs(
            tmp_path,
            source_inventory_commit_sha=commit,
            expected_bootstrap_receipt_sha256=ref["sha256"],
            require_stage2_physical_transfer=True,
        )


def test_candidate_rejects_no_cache_bootstrap_claimed_as_reused(tmp_path: Path) -> None:
    payloads = _candidate_payloads(tmp_path)
    payloads["transfer_bootstrap_receipt.json"][
        "existing_instance_cache_reused"
    ] = True
    output = tmp_path / "data/manifests/stage2_2khz/fixture"
    output.parent.mkdir(parents=True)
    digests = publish_payloads_noreplace(output, payloads)

    with pytest.raises(ValueError, match="idempotent cache/scratch"):
        validate_stage2_pretrain_data_candidate(
            repository_root=tmp_path,
            manifest_ref=(
                "data/manifests/stage2_2khz/fixture/manifest_bundle.json",
                digests["manifest_bundle.json"],
            ),
            lineage_ref=(
                "data/manifests/stage2_2khz/fixture/lineage_receipt.json",
                digests["lineage_receipt.json"],
            ),
            coverage_ref=(
                "data/manifests/stage2_2khz/fixture/frequency_coverage_receipt.json",
                digests["frequency_coverage_receipt.json"],
            ),
            bootstrap_ref=(
                "data/manifests/stage2_2khz/fixture/transfer_bootstrap_receipt.json",
                digests["transfer_bootstrap_receipt.json"],
            ),
            plant_binding_file_sha256="a" * 64,
            workers=1,
        )


def test_actual_source_bytes_issue_and_independent_candidate_revalidation(
    tmp_path: Path,
) -> None:
    payloads = _candidate_payloads(tmp_path)
    output = tmp_path / "data/manifests/stage2_2khz/fixture"
    output.parent.mkdir(parents=True)
    digests = publish_payloads_noreplace(output, payloads)

    refs = {
        "manifest_ref": (
            "data/manifests/stage2_2khz/fixture/manifest_bundle.json",
            digests["manifest_bundle.json"],
        ),
        "lineage_ref": (
            "data/manifests/stage2_2khz/fixture/lineage_receipt.json",
            digests["lineage_receipt.json"],
        ),
        "coverage_ref": (
            "data/manifests/stage2_2khz/fixture/frequency_coverage_receipt.json",
            digests["frequency_coverage_receipt.json"],
        ),
        "bootstrap_ref": (
            "data/manifests/stage2_2khz/fixture/transfer_bootstrap_receipt.json",
            digests["transfer_bootstrap_receipt.json"],
        ),
    }
    binding = validate_stage2_pretrain_data_candidate(
        repository_root=tmp_path,
        plant_binding_file_sha256="a" * 64,
        workers=1,
        **refs,
    )
    parallel_binding = validate_stage2_pretrain_data_candidate(
        repository_root=tmp_path,
        plant_binding_file_sha256="a" * 64,
        workers=4,
        **refs,
    )
    assert len(binding.records) == 48
    assert len(binding.sources) == 48
    assert parallel_binding == binding

    completion, completion_sha = seal_published_payloads_noreplace(
        output,
        artifact_sha256=digests,
        validated_record_count=len(binding.records),
    )
    assert completion["git_training_authority_granted"] is False
    assert completion["validated_record_count"] == 48
    assert sha256_bytes((output / "publication_complete.json").read_bytes()) == completion_sha


def test_completion_is_no_replace_and_preserves_first_bytes(tmp_path: Path) -> None:
    payloads = _candidate_payloads(tmp_path)
    output = tmp_path / "data/manifests/stage2_2khz/fixture"
    output.parent.mkdir(parents=True)
    digests = publish_payloads_noreplace(output, payloads)
    seal_published_payloads_noreplace(
        output,
        artifact_sha256=digests,
        validated_record_count=48,
    )
    original = (output / "publication_complete.json").read_bytes()

    with pytest.raises(Stage2PretrainDataIssueError, match="missing/extra residue"):
        seal_published_payloads_noreplace(
            output,
            artifact_sha256=digests,
            validated_record_count=48,
        )
    assert (output / "publication_complete.json").read_bytes() == original


def test_partial_publish_remains_forensic_and_cannot_be_retried_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "partial"
    payloads = _candidate_payloads(tmp_path)
    original_write = issuer_module._write_bytes_exclusive
    calls = 0

    def fail_after_first_artifact(path: Path, content: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("simulated publication interruption")
        original_write(path, content)

    monkeypatch.setattr(
        issuer_module,
        "_write_bytes_exclusive",
        fail_after_first_artifact,
    )
    with pytest.raises(OSError, match="simulated publication interruption"):
        publish_payloads_noreplace(output, payloads)
    assert (output / "publication_intent.json").is_file()
    assert not (output / "publication_complete.json").exists()
    partial_bytes = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }

    with pytest.raises(Stage2PretrainDataIssueError, match="이미 있습니다"):
        publish_payloads_noreplace(output, payloads)
    assert {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    } == partial_bytes


def test_extra_residue_blocks_completion(tmp_path: Path) -> None:
    payloads = _candidate_payloads(tmp_path)
    output = tmp_path / "data/manifests/stage2_2khz/fixture"
    output.parent.mkdir(parents=True)
    digests = publish_payloads_noreplace(output, payloads)
    (output / "unexpected.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(Stage2PretrainDataIssueError, match="missing/extra residue"):
        seal_published_payloads_noreplace(
            output,
            artifact_sha256=digests,
            validated_record_count=48,
        )
    assert not (output / "publication_complete.json").exists()


def test_coverage_workers_are_byte_deterministic(tmp_path: Path) -> None:
    _write_holdout(tmp_path)
    items = build_manifest_items(tmp_path, _validated_entries(tmp_path))
    manifest_sha = "e" * 64
    serial = build_coverage_receipt(
        tmp_path,
        items=items,
        manifest_bundle_sha256=manifest_sha,
        plant_binding_file_sha256="a" * 64,
        source_inventory_commit_sha="1" * 40,
        workers=1,
    )
    parallel = build_coverage_receipt(
        tmp_path,
        items=items,
        manifest_bundle_sha256=manifest_sha,
        plant_binding_file_sha256="a" * 64,
        source_inventory_commit_sha="1" * 40,
        workers=4,
    )
    assert canonical_json_bytes(serial) == canonical_json_bytes(parallel)


def test_manifest_input_requires_exact_six_canonical_tags(tmp_path: Path) -> None:
    entries = _validated_entries(tmp_path)
    missing = deepcopy(entries)
    missing.pop("demand")
    with pytest.raises(Stage2PretrainDataIssueError, match="exact 6종"):
        build_manifest_items(tmp_path, missing)

    extra = deepcopy(entries)
    extra["unexpected"] = list(extra["speech"])
    with pytest.raises(Stage2PretrainDataIssueError, match="exact 6종"):
        build_manifest_items(tmp_path, extra)


def test_component_cannot_cross_split(tmp_path: Path) -> None:
    entries = _validated_entries(tmp_path)
    speech = entries["speech"]
    train = next(row for row in speech if row["split"] == "train")
    val = next(row for row in speech if row["split"] == "val")
    val["group_id"] = train["group_id"]

    with pytest.raises(Stage2PretrainDataIssueError, match="split을 가로지릅니다"):
        build_manifest_items(tmp_path, entries)


def test_component_cannot_cross_source_family(tmp_path: Path) -> None:
    entries = _validated_entries(tmp_path)
    speech = next(row for row in entries["speech"] if row["split"] == "train")
    music = next(row for row in entries["music"] if row["split"] == "train")
    music["group_id"] = speech["group_id"]

    with pytest.raises(Stage2PretrainDataIssueError, match="source family"):
        build_manifest_items(tmp_path, entries)


def test_original_lineage_cannot_be_split_across_components(tmp_path: Path) -> None:
    entries = _validated_entries(tmp_path)
    speech = [row for row in entries["speech"] if row["split"] == "train"]
    speech[1]["lineage_keys"] = list(speech[0]["lineage_keys"])

    with pytest.raises(Stage2PretrainDataIssueError, match="lineage key"):
        build_manifest_items(tmp_path, entries)


@pytest.mark.parametrize("workers", [0, 17])
def test_coverage_workers_are_bounded(tmp_path: Path, workers: int) -> None:
    with pytest.raises(Stage2PretrainDataIssueError, match="workers"):
        build_coverage_receipt(
            tmp_path,
            items=[],
            manifest_bundle_sha256="e" * 64,
            plant_binding_file_sha256="a" * 64,
            source_inventory_commit_sha="1" * 40,
            workers=workers,
        )


def test_source_mutation_after_issue_is_rejected(tmp_path: Path) -> None:
    payloads = _candidate_payloads(tmp_path)
    output = tmp_path / "data/manifests/stage2_2khz/fixture"
    output.parent.mkdir(parents=True)
    digests = publish_payloads_noreplace(output, payloads)
    source = next((tmp_path / "data/raw/public").rglob("*.wav"))
    source.write_bytes(source.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="bytes SHA"):
        validate_stage2_pretrain_data_candidate(
            repository_root=tmp_path,
            manifest_ref=(
                "data/manifests/stage2_2khz/fixture/manifest_bundle.json",
                digests["manifest_bundle.json"],
            ),
            lineage_ref=(
                "data/manifests/stage2_2khz/fixture/lineage_receipt.json",
                digests["lineage_receipt.json"],
            ),
            coverage_ref=(
                "data/manifests/stage2_2khz/fixture/frequency_coverage_receipt.json",
                digests["frequency_coverage_receipt.json"],
            ),
            bootstrap_ref=(
                "data/manifests/stage2_2khz/fixture/transfer_bootstrap_receipt.json",
                digests["transfer_bootstrap_receipt.json"],
            ),
            plant_binding_file_sha256="a" * 64,
        )


def test_no_replace_output_preserves_first_generation(tmp_path: Path) -> None:
    output = tmp_path / "new"
    first = {"manifest_bundle.json": {"value": 1}}
    publish_payloads_noreplace(output, first)
    original = (output / "manifest_bundle.json").read_bytes()

    with pytest.raises(Stage2PretrainDataIssueError, match="이미 있습니다"):
        publish_payloads_noreplace(output, {"manifest_bundle.json": {"value": 2}})
    assert (output / "manifest_bundle.json").read_bytes() == original


def test_generic_public_basename_does_not_merge_unrelated_components() -> None:
    recorded = [
        {
            "clip": "recorded.wav",
            "content_sha256": "a" * 64,
            "lineage_keys": ["recorded:one"],
        }
    ]
    public = [
        {
            "component_id": "c1",
            "path": "a/ch01.wav",
            "content_sha256": "b" * 64,
            "lineage_keys": ["demand:A"],
        },
        {
            "component_id": "c2",
            "path": "b/ch01.wav",
            "content_sha256": "c" * 64,
            "lineage_keys": ["demand:B"],
        },
    ]
    assert (
        stage2_recorded_public_intersection(
            recorded_rows=recorded,
            public_items=public,
        )
        == 0
    )


def test_recorded_overlap_expands_only_within_canonical_component() -> None:
    recorded = [
        {
            "clip": "recorded.wav",
            "content_sha256": "a" * 64,
            "lineage_keys": ["recorded:one"],
        }
    ]
    public = [
        {
            "component_id": "c1",
            "path": "a/recorded.wav",
            "content_sha256": "b" * 64,
            "lineage_keys": ["public:one"],
        },
        {
            "component_id": "c1",
            "path": "a/second.wav",
            "content_sha256": "c" * 64,
            "lineage_keys": ["public:one"],
        },
        {
            "component_id": "c2",
            "path": "b/recorded.wav.generic",
            "content_sha256": "d" * 64,
            "lineage_keys": ["public:two"],
        },
    ]
    assert (
        stage2_recorded_public_intersection(
            recorded_rows=recorded,
            public_items=public,
        )
        == 2
    )


def test_cli_rejects_unbounded_worker_before_source_scan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.data import issue_stage2_pretrain_data as cli

    result = cli.main(
        [
            "--expected-commit",
            "1" * 40,
            "--plant-binding",
            "authority/plant.json",
            "--expected-plant-binding-sha256",
            "a" * 64,
            "--workers",
            "17",
        ]
    )
    assert result == 2
    assert "--workers는 1..16" in capsys.readouterr().err


def test_cli_rejects_output_outside_stage2_generation_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.data import issue_stage2_pretrain_data as cli

    manifest = tmp_path / "data/manifests/canonical_v4"
    manifest.mkdir(parents=True)
    plant = tmp_path / "authority/plant.json"
    plant.parent.mkdir(parents=True)
    plant.write_text("{}\n", encoding="utf-8")
    bootstrap = tmp_path / "data/manifests/receipt.json"
    bootstrap.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cli, "exact_clean_source_evidence", lambda *args, **kwargs: {})

    result = cli.main(
        [
            "--expected-commit",
            "1" * 40,
            "--manifest-dir",
            "data/manifests/canonical_v4",
            "--plant-binding",
            "authority/plant.json",
            "--expected-plant-binding-sha256",
            "a" * 64,
            "--bootstrap-receipt",
            "data/manifests/receipt.json",
            "--output-dir",
            "outside-stage2",
        ]
    )
    assert result == 2
    assert "stage2_2khz 아래" in capsys.readouterr().err


def test_cli_existing_output_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.data import issue_stage2_pretrain_data as cli

    manifest = tmp_path / "data/manifests/canonical_v4"
    manifest.mkdir(parents=True)
    plant = tmp_path / "authority/plant.json"
    plant.parent.mkdir(parents=True)
    plant.write_text("{}\n", encoding="utf-8")
    bootstrap = tmp_path / "data/manifests/receipt.json"
    bootstrap.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "data/manifests/stage2_2khz/existing"
    output.mkdir(parents=True)
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cli, "exact_clean_source_evidence", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        cli,
        "build_stage2_pretrain_data_payloads",
        lambda *args, **kwargs: {"manifest_bundle.json": {"fixture": True}},
    )

    result = cli.main(
        [
            "--expected-commit",
            "1" * 40,
            "--manifest-dir",
            "data/manifests/canonical_v4",
            "--plant-binding",
            "authority/plant.json",
            "--expected-plant-binding-sha256",
            "a" * 64,
            "--bootstrap-receipt",
            "data/manifests/receipt.json",
            "--output-dir",
            "data/manifests/stage2_2khz/existing",
        ]
    )
    assert result == 2
    assert "이미 있습니다" in capsys.readouterr().err
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"
