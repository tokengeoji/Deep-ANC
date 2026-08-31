from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
import numpy as np
import soundfile as sf

from deep_anc.dsp.stage2_2khz_contract import Stage2TwoKilohertzContract
from deep_anc.train.stage2_2khz_campaign import (
    STAGE2_CAMPAIGN_SCHEMA,
    STAGE2_UNATTESTED_STATUS,
    audit_stage2_2khz_campaign,
)
from deep_anc.train.stage2_2khz_execution import (
    Stage2FamilyComponentBatchSampler,
    Stage2SamplerRecord,
)
from deep_anc.train.stage2_2khz_pretrain_admission import (
    STAGE2_CRITERION_IMPLEMENTATION_RECEIPT_SCHEMA,
    STAGE2_DNH_CALIBRATION_RECEIPT_SCHEMA,
    STAGE2_PUBLIC_COVERAGE_RECEIPT_SCHEMA,
    STAGE2_PUBLIC_LINEAGE_RECEIPT_SCHEMA,
    STAGE2_PUBLIC_MANIFEST_BUNDLE_SCHEMA,
    STAGE2_TRANSFER_BOOTSTRAP_RECEIPT_SCHEMA,
    _load_self_attested_stage2_pretrain_data_binding_for_test,
    _validate_manifest_bundle,
)
from tests.test_stage2_2khz_execution import _production_binding_files


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_FILES = {
    "duct": "stage2_2khz_duct.yaml",
    "data": "stage2_2khz_data.yaml",
    "evaluation": "stage2_2khz_eval.yaml",
    "canonical_pretrain": "stage2_2khz_train_pretrain.yaml",
    "canonical_finetune": "stage2_2khz_train_finetune.yaml",
}


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write(root: Path, relative: str, content: bytes) -> str:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return _sha(content)


def _write_json(root: Path, relative: str, payload: dict) -> str:
    return _write(
        root,
        relative,
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
    )


def _write_yaml(root: Path, relative: str, payload: dict) -> str:
    return _write(
        root,
        relative,
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).encode("utf-8"),
    )


def _default_campaign() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "configs/stage2_2khz_campaign.yaml").read_text(encoding="utf-8")
    )


def _copy_profiles(root: Path) -> tuple[dict[str, dict], dict]:
    profiles: dict[str, dict] = {}
    campaign = _default_campaign()
    for role, filename in PROFILE_FILES.items():
        profiles[role] = yaml.safe_load(
            (REPO_ROOT / f"configs/{filename}").read_text(encoding="utf-8")
        )
        digest = _write_yaml(root, f"configs/{filename}", profiles[role])
        campaign["profiles"][role] = {
            "path": f"configs/{filename}",
            "sha256": digest,
        }
    model_relative = str(
        profiles["canonical_pretrain"]["execution"]["model_config"]["path"]
    )
    model_path = REPO_ROOT / model_relative
    _write(root, model_relative, model_path.read_bytes())
    return profiles, campaign


def _complete_self_attested_chain(root: Path) -> dict:
    profiles, campaign = _copy_profiles(root)
    artifact_shas = {
        "primary": _write(root, "artifacts/primary.npz", b"stage2-primary"),
        "secondary": _write(root, "artifacts/secondary.npz", b"stage2-secondary"),
        "plant_binding": _write_json(root, "artifacts/plant-binding.json", {"self_attested": True}),
        "manifest": _write_json(root, "artifacts/manifests.json", {"families": 4}),
        "lineage": _write_json(root, "artifacts/lineage.json", {"intersection": 0}),
        "coverage": _write_json(root, "artifacts/coverage.json", {"groups": 4}),
        "bootstrap": _write_json(root, "artifacts/bootstrap.json", {"exact": True}),
        "pretrain_criterion": _write_json(root, "artifacts/pretrain-criterion.json", {"dedicated": True}),
        "finetune_criterion": _write_json(root, "artifacts/finetune-criterion.json", {"dedicated": True}),
        "checkpoint": _write(root, "artifacts/canonical-pretrain.pt", b"self-attested-stage2-checkpoint"),
        "smoke_acceptance": _write_json(
            root,
            "artifacts/smoke-acceptance.json",
            {"fixture_only": True},
        ),
    }
    profiles["duct"]["artifacts"] = {
        "primary_path": {"path": "artifacts/primary.npz", "sha256": artifact_shas["primary"]},
        "secondary_path": {"path": "artifacts/secondary.npz", "sha256": artifact_shas["secondary"]},
        "plant_binding": {"path": "artifacts/plant-binding.json", "sha256": artifact_shas["plant_binding"]},
    }
    profiles["data"]["artifacts"] = {
        "manifest_bundle": {"path": "artifacts/manifests.json", "sha256": artifact_shas["manifest"]},
        "lineage_receipt": {"path": "artifacts/lineage.json", "sha256": artifact_shas["lineage"]},
        "frequency_coverage_receipt": {"path": "artifacts/coverage.json", "sha256": artifact_shas["coverage"]},
        "transfer_bootstrap_receipt": {"path": "artifacts/bootstrap.json", "sha256": artifact_shas["bootstrap"]},
    }
    profiles["canonical_pretrain"]["criterion"]["implementation_receipt"] = {
        "path": "artifacts/pretrain-criterion.json",
        "sha256": artifact_shas["pretrain_criterion"],
    }
    profiles["canonical_finetune"]["criterion"]["implementation_receipt"] = {
        "path": "artifacts/finetune-criterion.json",
        "sha256": artifact_shas["finetune_criterion"],
    }
    profiles["canonical_finetune"]["initialization"]["checkpoint_path"] = (
        "artifacts/canonical-pretrain.pt"
    )
    profiles["canonical_finetune"]["initialization"]["checkpoint_sha256"] = (
        artifact_shas["checkpoint"]
    )

    profile_sha256: dict[str, str] = {}
    for role, filename in PROFILE_FILES.items():
        digest = _write_yaml(root, f"configs/{filename}", profiles[role])
        profile_sha256[role] = digest
        campaign["profiles"][role]["sha256"] = digest

    contract = Stage2TwoKilohertzContract.canonical()
    plant_sha256 = {
        "primary_path_sha256": artifact_shas["primary"],
        "secondary_path_sha256": artifact_shas["secondary"],
        "plant_binding_sha256": artifact_shas["plant_binding"],
    }

    def external(stage: str, criterion_sha: str, init_sha: str | None) -> dict:
        return {
            "schema": "stage2_2khz_external_experiment_contract_v2",
            "stage": stage,
            "artifact_source_commit_sha": "a" * 40,
            "repository_clean_required": True,
            "control_band_contract": {
                "id": contract.contract_id,
                "sha256": contract.digest(),
            },
            "profile_sha256": profile_sha256,
            "plant_sha256": plant_sha256,
            "manifest_bundle_sha256": artifact_shas["manifest"],
            "criterion_receipt_sha256": criterion_sha,
            "initialization_mode": (
                "scratch"
                if stage == "canonical_pretrain"
                else "completed_stage2_scratch_pretrain_weight_only"
            ),
            "init_checkpoint_sha256": init_sha,
            "scratch_pretrain_origin_required": True,
            "legacy_artifacts_allowed": False,
            "automatic_resume_allowed": False,
            "training_eligible": True,
        }

    pretrain_external_sha = _write_json(
        root,
        "artifacts/pretrain-external.json",
        external("canonical_pretrain", artifact_shas["pretrain_criterion"], None),
    )
    finetune_external_sha = _write_json(
        root,
        "artifacts/finetune-external.json",
        external(
            "canonical_finetune",
            artifact_shas["finetune_criterion"],
            artifact_shas["checkpoint"],
        ),
    )
    campaign["external_contracts"] = {
        "canonical_pretrain": {
            "path": "artifacts/pretrain-external.json",
            "sha256": pretrain_external_sha,
        },
        "canonical_finetune": {
            "path": "artifacts/finetune-external.json",
            "sha256": finetune_external_sha,
        },
    }
    checkpoint_binding = {
        "schema": "stage2_2khz_checkpoint_binding_v2",
        "checkpoint_sha256": artifact_shas["checkpoint"],
        "external_experiment_contract_sha256": pretrain_external_sha,
        "control_band_contract": {
            "id": contract.contract_id,
            "sha256": contract.digest(),
        },
        "plant_sha256": plant_sha256,
        "plant_binding_runtime_sha256": "1" * 64,
        "manifest_bundle_sha256": artifact_shas["manifest"],
        "training_profile_sha256": profile_sha256["canonical_pretrain"],
        "evaluation_policy_sha256": profile_sha256["evaluation"],
        "model_config_sha256": "2" * 64,
        "criterion_receipt_sha256": artifact_shas["pretrain_criterion"],
        "sampler_receipt_sha256": "3" * 64,
        "dnh_calibration_receipt_sha256": "4" * 64,
        "a100_environment_sha256": "5" * 64,
        "smoke_acceptance_sha256": artifact_shas["smoke_acceptance"],
        "experiment_role": "canonical_pretrain",
        "completed_steps": 100_000,
        "init_eligible": True,
        "scratch_pretrain": True,
        "legacy_origin": False,
        "diagnostic_cpu_test": False,
        "completion_receipt_sha256": _sha(
            json.dumps(
                {
                    "schema": "stage2_2khz_pretrain_completion_receipt_v2",
                    "checkpoint_sha256": artifact_shas["checkpoint"],
                    "external_experiment_contract_sha256": pretrain_external_sha,
                    "completed_steps": 100_000,
                    "init_eligible": True,
                    "scratch_pretrain": True,
                    "smoke_acceptance_sha256": artifact_shas[
                        "smoke_acceptance"
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ),
    }
    binding_sha = _write_json(
        root, "artifacts/checkpoint-binding.json", checkpoint_binding
    )
    campaign["canonical_pretrain_checkpoint"] = {
        "checkpoint": {
            "path": "artifacts/canonical-pretrain.pt",
            "sha256": artifact_shas["checkpoint"],
        },
        "binding": {
            "path": "artifacts/checkpoint-binding.json",
            "sha256": binding_sha,
        },
    }
    return campaign


def _complete_typed_pretrain_only_chain(root: Path) -> tuple[dict, Path]:
    profiles, campaign = _copy_profiles(root)
    contract = Stage2TwoKilohertzContract.canonical()
    binding_path = _production_binding_files(root)
    binding_sha = _sha(binding_path.read_bytes())
    primary_path = root / "artifacts/primary.npz"
    secondary_path = root / "artifacts/secondary.npz"
    profiles["duct"]["artifacts"] = {
        "primary_path": {
            "path": "artifacts/primary.npz",
            "sha256": _sha(primary_path.read_bytes()),
        },
        "secondary_path": {
            "path": "artifacts/secondary.npz",
            "sha256": _sha(secondary_path.read_bytes()),
        },
        "plant_binding": {
            "path": "artifacts/binding.json",
            "sha256": binding_sha,
        },
    }

    items: list[dict] = []
    records: list[Stage2SamplerRecord] = []
    index = 0
    for split in ("train", "val", "test"):
        for family in ("speech", "music", "environment", "machine"):
            for component in range(4):
                relative = f"data/stage2/{split}/{split}-{family}-{component}.wav"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                generator = np.random.Generator(np.random.PCG64(10_000 + index))
                sf.write(
                    path,
                    generator.normal(0.0, 0.05, 8192).astype(np.float32),
                    48_000,
                )
                content_sha = _sha(path.read_bytes())
                component_id = f"{split}-{family}-{component}"
                items.append(
                    {
                        "dataset_index": index,
                        "source_family": family,
                        "component_id": component_id,
                        "split": split,
                        "path": relative,
                        "content_sha256": content_sha,
                        "content_size": path.stat().st_size,
                        "native_sample_rate": 48_000,
                        "native_nyquist_hz": 24_000.0,
                        "lineage_keys": [f"fixture_lineage:{component_id}"],
                    }
                )
                records.append(
                    Stage2SamplerRecord(
                        dataset_index=index,
                        source_family=family,
                        component_id=component_id,
                        split=split,
                        source_sha256=content_sha,
                    )
                )
                index += 1
    manifest_payload = {
        "schema": STAGE2_PUBLIC_MANIFEST_BUNDLE_SCHEMA,
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract": {
            "id": contract.contract_id,
            "sha256": contract.digest(),
        },
        "required_source_families": [
            "speech",
            "music",
            "environment",
            "machine",
        ],
        "required_splits": ["train", "val", "test"],
        "recorded_artifacts_required_for_pretrain": False,
        "test_split_for_checkpoint_selection_allowed": False,
        "source_inventory_commit_sha": "a" * 40,
        "items": items,
    }
    manifest_path = root / "artifacts/stage2-manifest.json"
    manifest_sha = _write_json(
        root, "artifacts/stage2-manifest.json", manifest_payload
    )
    recorded_clips = [
        {
            "family": "speech",
            "clip": "recorded-fixture-only.wav",
            "content_sha256": "e" * 64,
            "lineage_keys": ["fixture_recorded:only"],
        }
    ]
    recorded_clip_lineage_sha = _sha(
        json.dumps(
            recorded_clips,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    recorded_holdout_sha = _write_json(
        root,
        "data/manifests/recorded_holdout.json",
        {
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
                "clips": recorded_clips,
                "clips_sha256": recorded_clip_lineage_sha,
            },
            "families": {"speech": ["recorded-fixture-only.wav"]},
        },
    )
    lineage_sha = _write_json(
        root,
        "artifacts/stage2-lineage.json",
        {
            "schema": STAGE2_PUBLIC_LINEAGE_RECEIPT_SCHEMA,
            "status": "PASS",
            "canonical_pretrain_eligible": True,
            "control_band_contract_sha256": contract.digest(),
            "manifest_bundle_sha256": manifest_sha,
            "verified_item_count": len(items),
            "component_cross_split_count": 0,
            "source_sha_cross_split_count": 0,
            "original_lineage_cross_split_count": 0,
            "recorded_synthetic_lineage_intersection_count": 0,
            "actual_manifest_rows_consumed": True,
            "recorded_holdout": {
                "path": "data/manifests/recorded_holdout.json",
                "sha256": recorded_holdout_sha,
            },
            "recorded_clip_count": len(recorded_clips),
            "recorded_clip_lineage_sha256": recorded_clip_lineage_sha,
            "recorded_synthetic_intersection_algorithm": (
                "transitive_basename_content_sha256_lineage_keys_v1"
            ),
            "actual_recorded_holdout_bytes_consumed": True,
            "source_inventory_commit_sha": "a" * 40,
        },
    )
    qualified_octave: dict[str, dict[str, list[list[dict]]]] = {}
    qualified_sentinel: dict[str, dict[str, list[dict]]] = {}
    for split in ("train", "val", "test"):
        qualified_octave[split] = {}
        qualified_sentinel[split] = {}
        for family in ("speech", "music", "environment", "machine"):
            entries = [
                {
                    "dataset_index": row["dataset_index"],
                    "component_id": row["component_id"],
                    "path": row["path"],
                    "content_sha256": row["content_sha256"],
                }
                for row in items
                if row["split"] == split and row["source_family"] == family
            ]
            qualified_octave[split][family] = [list(entries) for _ in range(5)]
            qualified_sentinel[split][family] = list(entries)
    coverage_sha = _write_json(
        root,
        "artifacts/stage2-coverage.json",
        {
            "schema": STAGE2_PUBLIC_COVERAGE_RECEIPT_SCHEMA,
            "status": "PASS",
            "canonical_pretrain_eligible": True,
            "control_band_contract_sha256": contract.digest(),
            "manifest_bundle_sha256": manifest_sha,
            "actual_source_bytes_recomputed": True,
            "plant_binding_file_sha256": binding_sha,
            "source_density_algorithm": (
                "mono_mean_welch_nperseg8192_noverlap4096_detrend_false_v1"
            ),
            "octave_objective_bands_hz": [
                [88.3883476483, 176.7766952966],
                [176.7766952966, 353.5533905933],
                [353.5533905933, 707.1067811865],
                [707.1067811865, 1414.2135623731],
                [1414.2135623731, 2828.4271247462],
            ],
            "minimum_source_density_ratio": 0.25,
            "minimum_independent_components_per_family_octave": 4,
            "qualified_sources_by_split_family_octave": qualified_octave,
            "one_point_six_khz_sentinel_band_hz": [1425.437949, 1795.939277],
            "qualified_sources_by_split_family_one_point_six_khz_sentinel": (
                qualified_sentinel
            ),
            "source_inventory_commit_sha": "a" * 40,
        },
    )
    bootstrap_sha = _write_json(
        root,
        "artifacts/stage2-bootstrap.json",
        {
            "schema": STAGE2_TRANSFER_BOOTSTRAP_RECEIPT_SCHEMA,
            "status": "PASS",
            "canonical_pretrain_eligible": True,
            "control_band_contract_sha256": contract.digest(),
            "manifest_bundle_sha256": manifest_sha,
            "existing_instance_cache_reused": True,
            "all_declared_source_bytes_rehashed": True,
            "stale_run_or_checkpoint_auto_resume_allowed": False,
            "scratch_new_run_directory_required": True,
            "source_inventory_commit_sha": "a" * 40,
        },
    )
    profiles["data"]["artifacts"] = {
        "manifest_bundle": {
            "path": "artifacts/stage2-manifest.json",
            "sha256": manifest_sha,
        },
        "lineage_receipt": {
            "path": "artifacts/stage2-lineage.json",
            "sha256": lineage_sha,
        },
        "frequency_coverage_receipt": {
            "path": "artifacts/stage2-coverage.json",
            "sha256": coverage_sha,
        },
        "transfer_bootstrap_receipt": {
            "path": "artifacts/stage2-bootstrap.json",
            "sha256": bootstrap_sha,
        },
    }

    provisional_sampler = Stage2FamilyComponentBatchSampler(
        records,
        batch_size=96,
        seed=20260803,
        manifest_bundle_sha256=manifest_sha,
        sampler_receipt_sha256="0" * 64,
    )
    sampler_sha = _write_json(
        root,
        "artifacts/stage2-sampler.json",
        provisional_sampler.expected_receipt_payload(),
    )
    dnh_sha = _write_json(
        root,
        "artifacts/stage2-dnh.json",
        {
            "schema": STAGE2_DNH_CALIBRATION_RECEIPT_SCHEMA,
            "status": "PASS",
            "canonical_pretrain_eligible": True,
            "control_band_contract_sha256": contract.digest(),
            "plant_binding_file_sha256": binding_sha,
            "manifest_bundle_sha256": manifest_sha,
            "sampler_receipt_sha256": sampler_sha,
            "actual_causal_secondary_output": True,
            "actual_family_balanced_batch": True,
            "lambda_dnh": 0.001,
            "output_y_gradient_share": 0.3,
            "calibration_batch_sha256": "5" * 64,
        },
    )
    implementation = (
        (
            "loss_implementation",
            "src/deep_anc/losses/stage2_2khz_loss.py",
            "stage2_2khz_dedicated_loss_v1",
        ),
        (
            "trainer_adapter_implementation",
            "src/deep_anc/train/stage2_2khz_execution.py",
            "stage2_2khz_trainer_adapter_v1",
        ),
        (
            "scratch_runner_implementation",
            "src/deep_anc/train/stage2_2khz_runner.py",
            "stage2_2khz_scratch_pretrain_runner_v1",
        ),
    )
    criterion_payload = {
        "schema": STAGE2_CRITERION_IMPLEMENTATION_RECEIPT_SCHEMA,
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract_sha256": contract.digest(),
        "plant_binding_file_sha256": binding_sha,
        "manifest_bundle_sha256": manifest_sha,
        "sampler_receipt": {
            "path": "artifacts/stage2-sampler.json",
            "sha256": sampler_sha,
        },
        "dnh_calibration_receipt": {
            "path": "artifacts/stage2-dnh.json",
            "sha256": dnh_sha,
        },
        "batch_size": 96,
        "seed": 20260803,
        "generic_stage1_loss_used": False,
        "full_octave_v3_loss_used": False,
    }
    for key, relative, schema in implementation:
        source = REPO_ROOT / relative
        source_sha = _write(root, relative, source.read_bytes())
        criterion_payload[key] = {
            "path": relative,
            "sha256": source_sha,
            "schema": schema,
        }
    criterion_sha = _write_json(
        root, "artifacts/stage2-criterion.json", criterion_payload
    )
    profiles["canonical_pretrain"]["criterion"]["implementation_receipt"] = {
        "path": "artifacts/stage2-criterion.json",
        "sha256": criterion_sha,
    }

    profile_sha: dict[str, str] = {}
    for role, filename in PROFILE_FILES.items():
        digest = _write_yaml(root, f"configs/{filename}", profiles[role])
        campaign["profiles"][role]["sha256"] = digest
        profile_sha[role] = digest
    plant_sha = {
        "primary_path_sha256": _sha(primary_path.read_bytes()),
        "secondary_path_sha256": _sha(secondary_path.read_bytes()),
        "plant_binding_sha256": binding_sha,
    }
    external_sha = _write_json(
        root,
        "artifacts/stage2-pretrain-external.json",
        {
            "schema": "stage2_2khz_external_experiment_contract_v2",
            "stage": "canonical_pretrain",
            "artifact_source_commit_sha": "a" * 40,
            "repository_clean_required": True,
            "control_band_contract": {
                "id": contract.contract_id,
                "sha256": contract.digest(),
            },
            "profile_sha256": profile_sha,
            "plant_sha256": plant_sha,
            "manifest_bundle_sha256": manifest_sha,
            "criterion_receipt_sha256": criterion_sha,
            "initialization_mode": "scratch",
            "init_checkpoint_sha256": None,
            "scratch_pretrain_origin_required": True,
            "legacy_artifacts_allowed": False,
            "automatic_resume_allowed": False,
            "training_eligible": True,
        },
    )
    campaign["external_contracts"]["canonical_pretrain"] = {
        "path": "artifacts/stage2-pretrain-external.json",
        "sha256": external_sha,
    }
    return campaign, manifest_path


def _load_fixture_data_binding(
    root: Path,
    *,
    manifest_sha: str | None = None,
    lineage_sha: str | None = None,
    coverage_sha: str | None = None,
    bootstrap_sha: str | None = None,
):
    def digest(relative: str, override: str | None) -> str:
        return override or _sha((root / relative).read_bytes())

    return _load_self_attested_stage2_pretrain_data_binding_for_test(
        repository_root=root,
        manifest_ref=(
            "artifacts/stage2-manifest.json",
            digest("artifacts/stage2-manifest.json", manifest_sha),
        ),
        lineage_ref=(
            "artifacts/stage2-lineage.json",
            digest("artifacts/stage2-lineage.json", lineage_sha),
        ),
        coverage_ref=(
            "artifacts/stage2-coverage.json",
            digest("artifacts/stage2-coverage.json", coverage_sha),
        ),
        bootstrap_ref=(
            "artifacts/stage2-bootstrap.json",
            digest("artifacts/stage2-bootstrap.json", bootstrap_sha),
        ),
        plant_binding_file_sha256=_sha(
            (root / "artifacts/binding.json").read_bytes()
        ),
    )


def test_default_stage2_profile_is_blocked_before_gpu_or_run_directory() -> None:
    report = audit_stage2_2khz_campaign(_default_campaign(), repo_root=REPO_ROOT)

    assert report["status"] == "BLOCKED"
    assert report["eligible"] is False
    assert report["audio_opened"] is False
    assert report["trainer_constructed"] is False
    assert report["gpu_initialized"] is False
    assert report["runs_directory_created"] is False
    assert report["control_band_contract_sha256"] == (
        Stage2TwoKilohertzContract.canonical().digest()
    )
    assert report["scratch_pretrain_required"] is True
    assert report["legacy_init_or_resume_allowed"] is False
    assert report["three_db_is_minimum_not_optimization_target"] is True
    assert report["checkpoint_selection_primary"].startswith("maximize_minimum")
    assert report["checkpoint_selection_secondary"].startswith(
        "maximize_two_khz"
    )


def test_complete_self_attested_chain_binds_contract_ps_train_eval_but_stays_blocked(
    tmp_path: Path,
) -> None:
    report = audit_stage2_2khz_campaign(
        _complete_self_attested_chain(tmp_path), repo_root=tmp_path
    )

    assert report["status"] == STAGE2_UNATTESTED_STATUS
    assert report["declared_external_checkpoint_cross_binding_valid"] is True
    assert report["typed_stage2_execution_provenance_attested"] is False
    assert report["eligible"] is False
    assert "typed_stage2_execution_provenance" in report["blockers"]


def test_self_attested_typed_pretrain_chain_cannot_become_ready(
    tmp_path: Path,
) -> None:
    campaign, _ = _complete_typed_pretrain_only_chain(tmp_path)
    report = audit_stage2_2khz_campaign(campaign, repo_root=tmp_path)

    assert report["status"] == "BLOCKED"
    assert report["pretrain_smoke_ready"] is False
    assert report["canonical_scratch_pretrain_100k_ready"] is False
    assert report["canonical_finetune_ready"] is False
    assert "typed_stage2_pretrain_execution" in report["blockers"]
    assert "typed_stage2_finetune_recorded_70_30_execution" in report["blockers"]


def test_actual_manifest_lineage_crossing_is_rejected_without_trusting_receipt(
    tmp_path: Path,
) -> None:
    _, manifest_path = _complete_typed_pretrain_only_chain(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = next(row for row in payload["items"] if row["split"] == "train")
    val = next(row for row in payload["items"] if row["split"] == "val")
    val["lineage_keys"] = list(train["lineage_keys"])
    digest = _write_json(
        tmp_path, "artifacts/stage2-manifest-crossing.json", payload
    )
    with pytest.raises(ValueError, match="original lineage.*split"):
        _validate_manifest_bundle(
            tmp_path,
            path="artifacts/stage2-manifest-crossing.json",
            sha256=digest,
        )


def test_same_source_bytes_cannot_be_split_into_fake_components(
    tmp_path: Path,
) -> None:
    _, manifest_path = _complete_typed_pretrain_only_chain(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload["items"]
        if row["split"] == "train" and row["source_family"] == "environment"
    ]
    source = tmp_path / rows[0]["path"]
    duplicate = tmp_path / rows[1]["path"]
    duplicate.write_bytes(source.read_bytes())
    rows[1]["content_sha256"] = _sha(duplicate.read_bytes())
    rows[1]["content_size"] = duplicate.stat().st_size
    digest = _write_json(
        tmp_path,
        "artifacts/stage2-manifest-duplicate-source.json",
        payload,
    )

    with pytest.raises(ValueError, match="동일 source bytes SHA.*component_id"):
        _validate_manifest_bundle(
            tmp_path,
            path="artifacts/stage2-manifest-duplicate-source.json",
            sha256=digest,
        )


def test_coverage_receipt_is_recomputed_from_exact_manifest_source_bytes(
    tmp_path: Path,
) -> None:
    _complete_typed_pretrain_only_chain(tmp_path)

    binding = _load_fixture_data_binding(tmp_path)

    assert len(binding.records) == 48
    assert len(binding.sources) == 48


@pytest.mark.parametrize("coverage_axis", ["octave", "sentinel"])
def test_four_files_from_one_component_cannot_masquerade_as_four_components(
    tmp_path: Path,
    coverage_axis: str,
) -> None:
    _complete_typed_pretrain_only_chain(tmp_path)
    path = tmp_path / "artifacts/stage2-coverage.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if coverage_axis == "octave":
        entries = payload["qualified_sources_by_split_family_octave"]["train"][
            "speech"
        ][4]
        payload["qualified_sources_by_split_family_octave"]["train"]["speech"][
            4
        ] = [deepcopy(entries[0]) for _ in range(4)]
    else:
        entries = payload[
            "qualified_sources_by_split_family_one_point_six_khz_sentinel"
        ]["train"]["speech"]
        payload[
            "qualified_sources_by_split_family_one_point_six_khz_sentinel"
        ]["train"]["speech"] = [deepcopy(entries[0]) for _ in range(4)]
    forged_sha = _write_json(
        tmp_path,
        "artifacts/stage2-coverage.json",
        payload,
    )

    with pytest.raises(ValueError, match="qualified IDs.*actual manifest/source bytes"):
        _load_fixture_data_binding(tmp_path, coverage_sha=forged_sha)


@pytest.mark.parametrize("field", ["component_id", "path", "content_sha256"])
def test_coverage_identity_fields_must_reverse_map_to_manifest(
    tmp_path: Path,
    field: str,
) -> None:
    _complete_typed_pretrain_only_chain(tmp_path)
    path = tmp_path / "artifacts/stage2-coverage.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["qualified_sources_by_split_family_octave"]["val"]["music"][
        2
    ][0]
    entry[field] = "0" * 64 if field == "content_sha256" else f"forged-{field}"
    forged_sha = _write_json(
        tmp_path,
        "artifacts/stage2-coverage.json",
        payload,
    )

    with pytest.raises(ValueError, match="qualified IDs.*actual manifest/source bytes"):
        _load_fixture_data_binding(tmp_path, coverage_sha=forged_sha)


def test_resealed_zero_source_bytes_cannot_keep_claimed_coverage(
    tmp_path: Path,
) -> None:
    _, manifest_path = _complete_typed_pretrain_only_chain(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = next(
        item
        for item in manifest["items"]
        if item["split"] == "test"
        and item["source_family"] == "machine"
        and item["component_id"].endswith("-0")
    )
    source_path = tmp_path / row["path"]
    sf.write(source_path, np.zeros(8192, dtype=np.float32), 48_000)
    row["content_sha256"] = _sha(source_path.read_bytes())
    row["content_size"] = source_path.stat().st_size
    manifest_sha = _write_json(
        tmp_path,
        "artifacts/stage2-manifest.json",
        manifest,
    )

    receipt_shas: dict[str, str] = {}
    for name in ("lineage", "coverage", "bootstrap"):
        relative = f"artifacts/stage2-{name}.json"
        payload = json.loads((tmp_path / relative).read_text(encoding="utf-8"))
        payload["manifest_bundle_sha256"] = manifest_sha
        if name == "coverage":
            for split in ("train", "val", "test"):
                for family in ("speech", "music", "environment", "machine"):
                    groups = payload[
                        "qualified_sources_by_split_family_octave"
                    ][split][family]
                    sentinel = payload[
                        "qualified_sources_by_split_family_one_point_six_khz_sentinel"
                    ][split][family]
                    for entries in [*groups, sentinel]:
                        for entry in entries:
                            if entry["dataset_index"] == row["dataset_index"]:
                                entry["content_sha256"] = row["content_sha256"]
        receipt_shas[name] = _write_json(tmp_path, relative, payload)

    with pytest.raises(ValueError, match="actual source bytes.*distinct component"):
        _load_fixture_data_binding(
            tmp_path,
            manifest_sha=manifest_sha,
            lineage_sha=receipt_shas["lineage"],
            coverage_sha=receipt_shas["coverage"],
            bootstrap_sha=receipt_shas["bootstrap"],
        )


def test_external_contract_cannot_swap_stage2_primary_path_sha(tmp_path: Path) -> None:
    campaign = _complete_self_attested_chain(tmp_path)
    path = tmp_path / "artifacts/pretrain-external.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["plant_sha256"]["primary_path_sha256"] = "0" * 64
    campaign["external_contracts"]["canonical_pretrain"]["sha256"] = _write_json(
        tmp_path, "artifacts/pretrain-external.json", payload
    )
    with pytest.raises(ValueError, match="P/S SHA"):
        audit_stage2_2khz_campaign(campaign, repo_root=tmp_path)


def test_checkpoint_binding_cannot_swap_evaluation_policy_sha(tmp_path: Path) -> None:
    campaign = _complete_self_attested_chain(tmp_path)
    path = tmp_path / "artifacts/checkpoint-binding.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evaluation_policy_sha256"] = "0" * 64
    campaign["canonical_pretrain_checkpoint"]["binding"]["sha256"] = _write_json(
        tmp_path, "artifacts/checkpoint-binding.json", payload
    )
    with pytest.raises(ValueError, match="evaluation policy SHA"):
        audit_stage2_2khz_campaign(campaign, repo_root=tmp_path)


def test_stage2_profile_rejects_threshold_relaxation_and_legacy_init(
    tmp_path: Path,
) -> None:
    profiles, campaign = _copy_profiles(tmp_path)
    profiles["evaluation"]["two_khz_attenuation_threshold_db"] = 2.9
    campaign["profiles"]["evaluation"]["sha256"] = _write_yaml(
        tmp_path,
        "configs/stage2_2khz_eval.yaml",
        profiles["evaluation"],
    )
    with pytest.raises(ValueError, match="2 kHz threshold"):
        audit_stage2_2khz_campaign(campaign, repo_root=tmp_path)

    profiles, campaign = _copy_profiles(tmp_path)
    profiles["canonical_pretrain"]["initialization"]["mode"] = "legacy_warm_start"
    campaign["profiles"]["canonical_pretrain"]["sha256"] = _write_yaml(
        tmp_path,
        "configs/stage2_2khz_train_pretrain.yaml",
        profiles["canonical_pretrain"],
    )
    with pytest.raises(ValueError, match="initialization mode"):
        audit_stage2_2khz_campaign(campaign, repo_root=tmp_path)


def test_stage2_profile_binds_b96_and_bounded_prefetch_exactly(tmp_path: Path) -> None:
    profiles, campaign = _copy_profiles(tmp_path)
    profiles["canonical_pretrain"]["execution"]["batch_size"] = 16
    campaign["profiles"]["canonical_pretrain"]["sha256"] = _write_yaml(
        tmp_path,
        "configs/stage2_2khz_train_pretrain.yaml",
        profiles["canonical_pretrain"],
    )
    with pytest.raises(ValueError, match="A100 smoke batch candidate"):
        audit_stage2_2khz_campaign(campaign, repo_root=tmp_path)

    profiles, campaign = _copy_profiles(tmp_path)
    profiles["canonical_pretrain"]["execution"]["data_pipeline"][
        "bounded_prefetch_batches"
    ] = 55
    campaign["profiles"]["canonical_pretrain"]["sha256"] = _write_yaml(
        tmp_path,
        "configs/stage2_2khz_train_pretrain.yaml",
        profiles["canonical_pretrain"],
    )
    with pytest.raises(ValueError, match="bounded data pipeline"):
        audit_stage2_2khz_campaign(campaign, repo_root=tmp_path)


def test_profile_bytes_sha_drift_is_rejected(tmp_path: Path) -> None:
    profiles, campaign = _copy_profiles(tmp_path)
    del profiles
    path = tmp_path / "configs/stage2_2khz_eval.yaml"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="bytes SHA"):
        audit_stage2_2khz_campaign(campaign, repo_root=tmp_path)


def test_stage2_preflight_source_has_no_audio_gpu_trainer_or_subprocess_imports() -> None:
    sources = [
        REPO_ROOT / "src/deep_anc/train/stage2_2khz_campaign.py",
        REPO_ROOT / "scripts/train/check_stage2_2khz_campaign.py",
    ]
    tree = ast.parse("\n".join(path.read_text(encoding="utf-8") for path in sources))
    imported: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    assert not {"sounddevice", "torch", "subprocess"} & imported
    assert "Trainer" not in names


def _load_train_entry():
    spec = importlib.util.spec_from_file_location(
        "stage2_profile_train_entry", REPO_ROOT / "scripts/train/train.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "schema",
    [
        STAGE2_CAMPAIGN_SCHEMA,
        "stage2_2khz_duct_profile_v1",
        "stage2_2khz_data_profile_v1",
        "stage2_2khz_evaluation_policy_v1",
        "stage2_2khz_training_profile_v1",
    ],
)
def test_generic_train_entry_blocks_every_stage2_profile_before_config_trainer_or_run_dir(
    schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_entry = _load_train_entry()
    called = {"load_train_config": False, "trainer": False}
    monkeypatch.setattr(train_entry, "load_yaml", lambda _: {"schema": schema})

    def unexpected_load(*_args, **_kwargs):
        called["load_train_config"] = True
        raise AssertionError("load_train_config must not be called")

    class UnexpectedTrainer:
        def __init__(self, *_args, **_kwargs):
            called["trainer"] = True
            raise AssertionError("Trainer must not be constructed")

    monkeypatch.setattr(train_entry, "load_train_config", unexpected_load)
    monkeypatch.setattr(train_entry, "Trainer", UnexpectedTrainer)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "ignored.yaml"])
    assert train_entry.main() == 2
    assert called == {"load_train_config": False, "trainer": False}
