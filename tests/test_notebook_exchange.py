from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from deep_anc.data.notebook_exchange import (
    PHASES,
    NotebookExchangeError,
    _mark_summary_semantically_verified,
    _validate_phase_artifact_payloads,
    _validate_typed_receipt_payload,
    assert_exact_checkout,
    build_status,
    canonical_json_bytes,
    payload_sha256,
    read_remote_statuses,
    summarize_statuses,
    validate_status,
)
from deep_anc.dsp.stage2_2khz_contract import (
    STAGE2_2KHZ_CONTRACT_ID,
    STAGE2_2KHZ_OBJECTIVE_BANDS_HZ,
    Stage2TwoKilohertzContract,
)


COMMIT = "a" * 40
SCHEMAS = {
    "checkout_audit": "deep_anc_notebook_checkout_audit_v1",
    "restore_receipt": "stage2_drive_local_restore_receipt_v1",
    "archive_cache_receipt": "deep_anc_notebook_archive_cache_readback_v1",
    "decoder_qa_receipt": "deep_anc_notebook_full_decoder_receipt_v2",
    "manifest_bundle": "stage2_2khz_public_manifest_bundle_v1",
    "lineage_receipt": "stage2_2khz_public_lineage_receipt_v2",
    "frequency_coverage_receipt": "stage2_2khz_public_frequency_coverage_v2",
    "transfer_bootstrap_receipt": "stage2_2khz_transfer_bootstrap_receipt_v1",
}
REQUIRED = {
    "preflight": ("checkout_audit",),
    "drive_partial_restore": ("restore_receipt",),
    "public_archive_cache": ("archive_cache_receipt",),
    "decoder_qa": ("decoder_qa_receipt",),
    "lineage_manifest": ("manifest_bundle", "lineage_receipt"),
    "frequency_coverage": ("manifest_bundle", "frequency_coverage_receipt"),
    "bundle_publish": (
        "manifest_bundle",
        "lineage_receipt",
        "frequency_coverage_receipt",
        "transfer_bootstrap_receipt",
    ),
}
FAMILIES = ("speech", "music", "environment", "machine")
SPLITS = ("train", "val", "test")
ARCHIVE_SIZES = {
    "dns_noise_000": 5_364_611_964,
    "dns_noise_001": 5_357_916_291,
    "dns_speech_000": 4_664_045_287,
    "demand_dkitchen": 336_992_458,
    "demand_dwashing": 306_101_499,
    "demand_ooffice": 277_643_831,
    "demand_ohallway": 252_905_617,
    "demand_tmetro": 367_513_573,
    "demand_tcar": 373_520_251,
    "mimii_fan": 928_511_244,
}


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _with_evidence(payload: dict) -> dict:
    payload = copy.deepcopy(payload)
    payload["evidence_sha256"] = _sha(payload)
    return payload


def artifact(kind: str, name: str | None = None) -> dict:
    name = name or f"{kind}.json"
    digest = hashlib.sha256(kind.encode("utf-8")).hexdigest()
    return {
        "kind": kind,
        "name": name,
        "schema": SCHEMAS[kind],
        "size_bytes": 123,
        "sha256": digest,
        "remote_path": f"receipts/sha256_{digest}/{name}",
    }


def _semantic_artifact(kind: str, payload: dict) -> dict:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    return {
        "kind": kind,
        "name": f"{kind}.json",
        "schema": SCHEMAS[kind],
        "size_bytes": len(raw),
        "sha256": digest,
        "remote_path": f"receipts/sha256_{digest}/{kind}.json",
        "_payload": payload,
    }


def checkout_payload() -> dict:
    return {
        "schema": SCHEMAS["checkout_audit"],
        "status": "PASS",
        "source_commit": COMMIT,
        "repository_clean_exact": True,
        "work_root_outside_repository": True,
        "work_root_free_bytes": 40 * 1024**3,
        "rclone_executable_name": "rclone",
        "rclone_version": "rclone v1.70.0",
        "secrets_recorded": False,
    }


def restore_payload() -> dict:
    return _with_evidence(
        {
            "schema": SCHEMAS["restore_receipt"],
            "authority": "local_partial_restore_content_verified_not_training_authority",
            "status": "PASS_PARTIAL_RESTORE_ONLY",
            "anchor_file_sha256": "1" * 64,
            "anchor_evidence_sha256": "2" * 64,
            "snapshot_manifest_file_sha256": "3" * 64,
            "restore_root": "/mnt/ssd/stage2_restore",
            "file_count": 12_819,
            "byte_count": 9_480_223_737,
            "extension_counts": {
                ".mp3": 8_000,
                ".flac": 2_703,
                ".wav": 2_000,
                ".meta": 116,
            },
            "path_size_sha256_projection_sha256": "4" * 64,
            "stage2_public_pretrain_ready": False,
            "remaining_blockers": [
                "DNS_DEMAND_MIMII_FIXED_ARCHIVE_CACHE_OR_OFFICIAL_DOWNLOAD_REQUIRED",
                "STAGE2_LINEAGE_AND_FREQUENCY_MANIFEST_BUNDLE_REQUIRED",
                "LOCAL_DECODER_AND_SOURCE_DENSITY_AUDIT_REQUIRED",
            ],
        }
    )


def archive_payload() -> dict:
    return _with_evidence(
        {
            "schema": SCHEMAS["archive_cache_receipt"],
            "status": "PASS",
            "authority": "transport_readback_advisory_not_raw_or_training_authority",
            "source_commit": COMMIT,
            "production_manifest_sha256": "5" * 64,
            "production_manifest_remote_path": (
                f"manifests/v1/sha256_{'5' * 64}/archive_cache_manifest.json"
            ),
            "archive_count": 10,
            "archive_total_bytes": 18_229_762_015,
            "archive_sizes_by_id": ARCHIVE_SIZES,
            "archive_sha256_by_id": {
                key: hashlib.sha256(key.encode()).hexdigest() for key in ARCHIVE_SIZES
            },
            "immutable_content_addressed_archive_paths": True,
            "publisher_source_sha256_verified": True,
            "publisher_archive_and_manifest_readback_enforced": True,
            "canonical_training_authority": False,
        }
    )


def decoder_payload() -> dict:
    return _with_evidence(
        {
            "schema": SCHEMAS["decoder_qa_receipt"],
            "status": "PASS",
            "authority": "full_decoder_projection_not_training_authority",
            "source_commit": COMMIT,
            "decoder_audit_file_sha256": "6" * 64,
            "decoder_audit_semantic_sha256": "7" * 64,
            "inventory_sha256": "8" * 64,
            "accepted_inventory_sha256": "9" * 64,
            "candidate_count": 37_761,
            "accepted_count": 36_868,
            "rejected_count": 893,
            "cohort_counts": {
                "dns_fullband": 16_000,
                "speech": 8_065,
                "music": 8_000,
                "demand": 96,
                "machine": 3_600,
                "esc50": 2_000,
            },
            "accepted_cohort_counts": {
                "dns_fullband": 15_553,
                "speech": 7_971,
                "music": 7_941,
                "demand": 96,
                "machine": 3_600,
                "esc50": 1_707,
            },
            "rejected_cohort_counts": {
                "dns_fullband": 447,
                "speech": 94,
                "music": 59,
                "demand": 0,
                "machine": 0,
                "esc50": 293,
            },
            "full_sequential_chunk_frames": [65_536, 262_144],
            "full_inventory_rows_consumed": True,
            "partial_only": False,
            "canonical_training_authority": False,
        }
    )


def manifest_payload() -> dict:
    contract = Stage2TwoKilohertzContract.canonical()
    items = []
    index = 0
    for split in SPLITS:
        for family in FAMILIES:
            for component in range(4):
                component_id = f"{split}-{family}-{component}"
                items.append(
                    {
                        "dataset_index": index,
                        "source_family": family,
                        "component_id": component_id,
                        "split": split,
                        "path": f"data/stage2/{split}/{family}-{component}.wav",
                        "content_sha256": hashlib.sha256(component_id.encode()).hexdigest(),
                        "content_size": 65_536,
                        "native_sample_rate": 48_000,
                        "native_nyquist_hz": 24_000.0,
                        "lineage_keys": [f"fixture:{component_id}"],
                    }
                )
                index += 1
    return {
        "schema": SCHEMAS["manifest_bundle"],
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract": {
            "id": STAGE2_2KHZ_CONTRACT_ID,
            "sha256": contract.digest(),
        },
        "required_source_families": list(FAMILIES),
        "required_splits": list(SPLITS),
        "recorded_artifacts_required_for_pretrain": False,
        "test_split_for_checkpoint_selection_allowed": False,
        "source_inventory_commit_sha": COMMIT,
        "items": items,
    }


def lineage_payload(manifest_sha: str, item_count: int) -> dict:
    return {
        "schema": SCHEMAS["lineage_receipt"],
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract_sha256": Stage2TwoKilohertzContract.canonical().digest(),
        "manifest_bundle_sha256": manifest_sha,
        "verified_item_count": item_count,
        "component_cross_split_count": 0,
        "source_sha_cross_split_count": 0,
        "original_lineage_cross_split_count": 0,
        "recorded_synthetic_lineage_intersection_count": 0,
        "actual_manifest_rows_consumed": True,
        "recorded_holdout": {
            "path": "data/manifests/recorded_holdout.json",
            "sha256": "d" * 64,
        },
        "recorded_clip_count": 682,
        "recorded_clip_lineage_sha256": "e" * 64,
        "recorded_synthetic_intersection_algorithm": (
            "transitive_basename_content_sha256_lineage_keys_v1"
        ),
        "actual_recorded_holdout_bytes_consumed": True,
        "source_inventory_commit_sha": COMMIT,
    }


def coverage_payload(manifest: dict, manifest_sha: str) -> dict:
    cells = {
        (split, family): [
            {
                "dataset_index": row["dataset_index"],
                "component_id": row["component_id"],
                "path": row["path"],
                "content_sha256": row["content_sha256"],
            }
            for row in manifest["items"]
            if row["split"] == split and row["source_family"] == family
        ]
        for split in SPLITS
        for family in FAMILIES
    }
    octave = {
        split: {
            family: [copy.deepcopy(cells[(split, family)]) for _ in range(5)]
            for family in FAMILIES
        }
        for split in SPLITS
    }
    sentinel = {
        split: {
            family: copy.deepcopy(cells[(split, family)]) for family in FAMILIES
        }
        for split in SPLITS
    }
    return {
        "schema": SCHEMAS["frequency_coverage_receipt"],
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract_sha256": Stage2TwoKilohertzContract.canonical().digest(),
        "manifest_bundle_sha256": manifest_sha,
        "actual_source_bytes_recomputed": True,
        "plant_binding_file_sha256": "b" * 64,
        "source_density_algorithm": (
            "mono_mean_welch_nperseg8192_noverlap4096_detrend_false_v1"
        ),
        "octave_objective_bands_hz": [
            list(values) for values in STAGE2_2KHZ_OBJECTIVE_BANDS_HZ
        ],
        "minimum_source_density_ratio": 0.25,
        "minimum_independent_components_per_family_octave": 4,
        "qualified_sources_by_split_family_octave": octave,
        "one_point_six_khz_sentinel_band_hz": [1425.437949, 1795.939277],
        "qualified_sources_by_split_family_one_point_six_khz_sentinel": sentinel,
        "source_inventory_commit_sha": COMMIT,
    }


def transfer_payload(manifest_sha: str) -> dict:
    return {
        "schema": SCHEMAS["transfer_bootstrap_receipt"],
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract_sha256": Stage2TwoKilohertzContract.canonical().digest(),
        "manifest_bundle_sha256": manifest_sha,
        "existing_instance_cache_reused": True,
        "all_declared_source_bytes_rehashed": True,
        "stale_run_or_checkpoint_auto_resume_allowed": False,
        "scratch_new_run_directory_required": True,
        "source_inventory_commit_sha": COMMIT,
    }


def test_status_round_trip_and_tamper_rejected() -> None:
    status = build_status(
        source_commit=COMMIT,
        phase="preflight",
        state="PASS",
        message="exact checkout 확인",
        artifacts=[artifact("checkout_audit")],
        created_at_utc="2026-08-31T12:00:00Z",
    )
    assert validate_status(status, expected_commit=COMMIT) == status
    changed = copy.deepcopy(status)
    changed["message"] = "변조"
    with pytest.raises(NotebookExchangeError, match="payload SHA"):
        validate_status(changed, expected_commit=COMMIT)


def test_latest_status_per_phase_is_advisory_not_canonical() -> None:
    statuses = []
    for index, phase in enumerate(PHASES):
        statuses.append(
            build_status(
                source_commit=COMMIT,
                phase=phase,
                state="PASS",
                message=f"{phase} 완료",
                artifacts=[artifact(kind) for kind in REQUIRED[phase]],
                created_at_utc=f"2026-08-31T12:{index:02d}:00Z",
            )
        )
    unverified = summarize_statuses(statuses, expected_commit=COMMIT)
    assert unverified["all_phase_statuses_structurally_pass"] is True
    assert unverified["semantic_receipts_verified"] is False
    assert unverified["advisory_complete"] is False
    summary = _mark_summary_semantically_verified(unverified)
    assert summary["all_required_phases_pass"] is True
    assert summary["advisory_complete"] is True
    assert summary["completion_scope"] == "ADVISORY_COMPLETE"
    assert summary["canonical_pretrain_ready"] is False


def test_same_second_conflicting_status_is_rejected_as_ambiguous() -> None:
    first = build_status(
        source_commit=COMMIT,
        phase="preflight",
        state="PASS",
        message="첫 상태",
        artifacts=[artifact("checkout_audit")],
        created_at_utc="2026-08-31T12:00:00Z",
    )
    second = build_status(
        source_commit=COMMIT,
        phase="preflight",
        state="FAIL",
        message="같은 초의 다른 상태",
        artifacts=[],
        created_at_utc="2026-08-31T12:00:00Z",
    )
    with pytest.raises(NotebookExchangeError, match="latest가 모호"):
        summarize_statuses([first, second], expected_commit=COMMIT)


def test_checkout_audit_requires_detached_exact_head(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "dev"], cwd=repository, check=True)
    (repository / "tracked.txt").write_text("exact\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(NotebookExchangeError, match="detached HEAD"):
        assert_exact_checkout(repository_root=repository, expected_commit=commit)
    subprocess.run(["git", "checkout", "-q", "--detach", commit], cwd=repository, check=True)
    assert_exact_checkout(repository_root=repository, expected_commit=commit)


def test_wrong_commit_and_non_content_addressed_artifact_rejected() -> None:
    with pytest.raises(NotebookExchangeError, match="expected commit"):
        summarize_statuses([], expected_commit="bad")
    bad = artifact("checkout_audit")
    bad["remote_path"] = "receipts/receipt.json"
    with pytest.raises(NotebookExchangeError, match="content-addressed"):
        build_status(
            source_commit=COMMIT,
            phase="preflight",
            state="PASS",
            message="bad artifact",
            artifacts=[bad],
        )


@pytest.mark.parametrize("phase", PHASES)
def test_pass_phase_cannot_use_empty_artifact_list(phase: str) -> None:
    with pytest.raises(NotebookExchangeError, match="artifact kind"):
        build_status(
            source_commit=COMMIT,
            phase=phase,
            state="PASS",
            message="증거 없는 완료",
            artifacts=[],
        )


@pytest.mark.parametrize("kind", tuple(SCHEMAS))
def test_minimal_fake_receipt_is_rejected_for_all_eight_kinds(kind: str) -> None:
    fake = {"schema": SCHEMAS[kind], "status": "PASS"}
    with pytest.raises(NotebookExchangeError):
        _validate_typed_receipt_payload(kind, fake, expected_commit=COMMIT)


def test_realistic_receipts_pass_individual_semantic_validation() -> None:
    manifest = manifest_payload()
    manifest_sha = _semantic_artifact("manifest_bundle", manifest)["sha256"]
    values = {
        "checkout_audit": checkout_payload(),
        "restore_receipt": restore_payload(),
        "archive_cache_receipt": archive_payload(),
        "decoder_qa_receipt": decoder_payload(),
        "manifest_bundle": manifest,
        "lineage_receipt": lineage_payload(manifest_sha, len(manifest["items"])),
        "frequency_coverage_receipt": coverage_payload(manifest, manifest_sha),
        "transfer_bootstrap_receipt": transfer_payload(manifest_sha),
    }
    for kind, payload in values.items():
        assert (
            _validate_typed_receipt_payload(kind, payload, expected_commit=COMMIT)
            == SCHEMAS[kind]
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("all_accept", "partition"),
        ("cohort_swap", "partition"),
    ],
)
def test_decoder_receipt_requires_real_accept_reject_partition(
    mutation: str, message: str
) -> None:
    payload = decoder_payload()
    payload.pop("evidence_sha256")
    if mutation == "all_accept":
        payload["accepted_count"] = 37_761
        payload["rejected_count"] = 0
    else:
        payload["accepted_cohort_counts"]["esc50"] += 1
        payload["rejected_cohort_counts"]["esc50"] -= 1
    payload = _with_evidence(payload)
    with pytest.raises(NotebookExchangeError, match=message):
        _validate_typed_receipt_payload(
            "decoder_qa_receipt", payload, expected_commit=COMMIT
        )


def test_bundle_cross_sha_and_qualified_component_identity_are_verified() -> None:
    manifest = manifest_payload()
    manifest_item = _semantic_artifact("manifest_bundle", manifest)
    manifest_sha = manifest_item["sha256"]
    artifacts = [
        manifest_item,
        _semantic_artifact(
            "lineage_receipt", lineage_payload(manifest_sha, len(manifest["items"]))
        ),
        _semantic_artifact(
            "frequency_coverage_receipt", coverage_payload(manifest, manifest_sha)
        ),
        _semantic_artifact("transfer_bootstrap_receipt", transfer_payload(manifest_sha)),
    ]
    _validate_phase_artifact_payloads(
        phase="bundle_publish", artifacts=artifacts, expected_commit=COMMIT
    )

    tampered = copy.deepcopy(artifacts)
    coverage = tampered[2]["_payload"]
    coverage["qualified_sources_by_split_family_octave"]["train"]["speech"][0][0][
        "component_id"
    ] = "fake-component"
    with pytest.raises(NotebookExchangeError, match="manifest actual row"):
        _validate_phase_artifact_payloads(
            phase="bundle_publish", artifacts=tampered, expected_commit=COMMIT
        )

    wrong_sha = copy.deepcopy(artifacts)
    wrong_sha[3]["_payload"]["manifest_bundle_sha256"] = "f" * 64
    with pytest.raises(NotebookExchangeError, match="transfer receipt"):
        _validate_phase_artifact_payloads(
            phase="bundle_publish", artifacts=wrong_sha, expected_commit=COMMIT
        )


def test_artifact_bundle_sha_binds_typed_artifacts() -> None:
    status = build_status(
        source_commit=COMMIT,
        phase="bundle_publish",
        state="PASS",
        message="bundle 완료",
        artifacts=[artifact(kind) for kind in REQUIRED["bundle_publish"]],
    )
    changed = copy.deepcopy(status)
    changed["artifact_bundle_sha256"] = "0" * 64
    changed["payload_sha256"] = payload_sha256(changed)
    with pytest.raises(NotebookExchangeError, match="artifact bundle SHA"):
        validate_status(changed, expected_commit=COMMIT)


def test_remote_readback_rehashes_and_semantically_validates_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = json.dumps(checkout_payload(), ensure_ascii=False, sort_keys=True).encode()
    item = {
        "kind": "checkout_audit",
        "name": "checkout_audit.json",
        "schema": SCHEMAS["checkout_audit"],
        "size_bytes": len(receipt),
        "sha256": hashlib.sha256(receipt).hexdigest(),
    }
    item["remote_path"] = f"receipts/sha256_{item['sha256']}/{item['name']}"
    status = build_status(
        source_commit=COMMIT,
        phase="preflight",
        state="PASS",
        message="preflight",
        artifacts=[item],
        created_at_utc="2026-08-31T12:00:00Z",
    )
    status_name = (
        f"20260831T120000Z_{COMMIT[:12]}_preflight_"
        f"{status['payload_sha256'][:16]}.json"
    )

    def fake_run(args, *, timeout_seconds):
        del timeout_seconds
        target = args[-1]
        if "lsjson" in args:
            return json.dumps([{"Path": status_name, "IsDir": False}]).encode()
        if target.endswith(f"status/{status_name}"):
            return json.dumps(status).encode()
        if target.endswith(str(item["remote_path"])):
            return receipt
        raise AssertionError(args)

    monkeypatch.setattr("deep_anc.data.notebook_exchange._run_rclone", fake_run)
    summary = read_remote_statuses(
        remote_root="gdrive:DeepANC/notebook_exchange/stage2_2khz_v1",
        expected_commit=COMMIT,
    )
    assert summary["remote_artifacts_verified"] == 1
    assert summary["canonical_pretrain_ready"] is False

    def corrupted(args, *, timeout_seconds):
        result = fake_run(args, timeout_seconds=timeout_seconds)
        if args[-1].endswith(str(item["remote_path"])):
            return result + b"x"
        return result

    monkeypatch.setattr("deep_anc.data.notebook_exchange._run_rclone", corrupted)
    with pytest.raises(NotebookExchangeError, match="size"):
        read_remote_statuses(
            remote_root="gdrive:DeepANC/notebook_exchange/stage2_2khz_v1",
            expected_commit=COMMIT,
        )
