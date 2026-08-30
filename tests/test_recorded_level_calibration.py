"""historical ERR level receipt와 plant-domain sampler 회귀."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from deep_anc.data.recorded_dataset import (
    PLANT_DOMAIN_SAMPLING_MODE,
    RecordedANCDataset,
)
from deep_anc.data.recorded_level_calibration import (
    CURRENT_DOMAIN,
    HISTORICAL_DOMAIN,
    SCHEMA,
    WELCH_RECIPE,
    RecordedLevelCalibrationError,
    canonical_recorded_level_calibration_output,
    require_clean_exact_commit,
    require_recorded_level_calibration_config,
    validate_recorded_level_calibration_receipt,
    write_recorded_level_calibration_receipt,
)
from deep_anc.data.resumable_stream import indexed_rng
from deep_anc.data.source_trust import exact_clean_source_evidence


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()


def _ref(root: Path, relative: str) -> dict[str, object]:
    raw = (root / relative).read_bytes()
    return {"path": relative, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _write_receipt(root: Path) -> tuple[Path, str, list[str]]:
    (root / ".gitignore").write_text(
        "/data/manifests/recorded_level_calibration/\n",
        encoding="utf-8",
    )
    (root / "data/manifests").mkdir(parents=True)
    (root / "data/manifests/recorded_regrouped.jsonl").write_text("fixture\n")
    (root / "assets/measured").mkdir(parents=True)
    (root / "assets/measured/primary.npz").write_bytes(b"strict-primary")
    (root / "src/deep_anc/data").mkdir(parents=True)
    (root / "src/deep_anc/data/recorded_level_calibration.py").write_bytes(
        b"fixture implementation identity\n"
    )
    (root / "data/recorded/shared").mkdir(parents=True)
    samples = np.full(4096, 0.1, dtype=np.float32)
    sf.write(root / "data/recorded/shared/source_aligned.wav", samples, 48_000, subtype="FLOAT")
    sf.write(
        root / "data/recorded/shared/mics.wav",
        np.stack([samples, samples * 0.5], axis=1),
        48_000,
        subtype="FLOAT",
    )
    source_ref = _ref(root, "data/recorded/shared/source_aligned.wav")
    mics_ref = _ref(root, "data/recorded/shared/mics.wav")

    sessions: list[dict[str, object]] = []
    ids: list[str] = []
    cohorts: dict[str, dict[str, object]] = {}
    for prefix, cohort, gain in (
        ("20260804", "historical_20260804", 2.0),
        ("20260806", "historical_20260806", 4.0),
    ):
        member_ids: list[str] = []
        train_ids: list[str] = []
        for index in range(41):
            session_id = f"{prefix}_{index:06d}_file"
            split = "train" if index < 30 else ("val" if index < 35 else "test")
            ids.append(session_id)
            member_ids.append(session_id)
            if split == "train":
                train_ids.append(session_id)
            sessions.append(
                {
                    "session_id": session_id,
                    "split": split,
                    "source_family": ("speech", "music", "environment", "machine")[index % 4],
                    "cohort": cohort,
                    "plant_domain": HISTORICAL_DOMAIN,
                    "source_aligned": source_ref,
                    "mics": mics_ref,
                    "observed_to_strict_power_ratio_db": -20.0 * np.log10(gain),
                    "subband_observed_to_strict_power_ratio_db": [-20.0 * np.log10(gain)] * 4,
                    "raw_err_abs_peak": 0.1,
                    "calibrated_err_abs_peak": 0.1 * gain,
                }
            )
        fitted = -20.0 * np.log10(gain)
        cohorts[cohort] = {
            "fit_split": "train",
            "train_fit_count": len(train_ids),
            "fit_session_ids": sorted(train_ids),
            "member_session_ids": sorted(member_ids),
            "fitted_observed_to_strict_power_ratio_db": fitted,
            "err_amplitude_gain": gain,
            "heldout_residual_diagnostics": {},
        }
    implementation = _ref(root, "src/deep_anc/data/recorded_level_calibration.py")
    analysis = {
        "schema": SCHEMA,
        "welch_recipe": WELCH_RECIPE,
        "power_ratio_definition": "10log10(sum(abs(CSD_source_ERR)^2/PSD_source)/sum(PSD_source*abs(H_strict)^2))",
        "cohort_gain_definition": "10**(-median_train_power_ratio_db/20)",
        "shape_definition": "aggregate_CSD_over_PSD_then_best_integer_delay_and_complex_scalar",
        "implementation_sha256": implementation["sha256"],
    }
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "fixture"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    clean_source = exact_clean_source_evidence(
        root,
        expected_commit=commit,
        reject_runtime_bytecode=True,
    )
    payload = {
        "schema": SCHEMA,
        "source_commit": commit,
        "source_tree_clean_at_issue": True,
        "clean_source": clean_source,
        "analysis_contract": analysis,
        "analysis_contract_sha256": hashlib.sha256(_json_bytes(analysis)).hexdigest(),
        "implementation_source": implementation,
        "purpose": "old82_ERR_to_current_strict_primary_level_only",
        "reference_mode": "digital",
        "apply_to": ["ERR", "d"],
        "forbidden_apply_to": ["source_aligned", "REF", "acoustic_reference"],
        "fit_policy": {
            "allowed_split": "train",
            "heldout_splits_are_diagnostics_only": ["val", "test"],
            "wav_mutation": False,
        },
        "welch_recipe": WELCH_RECIPE,
        "recorded_manifest": _ref(root, "data/manifests/recorded_regrouped.jsonl"),
        "strict_primary_npz": _ref(root, "assets/measured/primary.npz"),
        "cohorts": cohorts,
        "plant_shape_diagnostic": {
            "definition": analysis["shape_definition"],
            "band_hz": [150.0, 1600.0],
            "delay_search_samples": [-2048, 2048],
            "best_relative_delay_samples": 0,
            "complex_agreement": 0.988,
            "relative_error_after_scalar_and_delay": 0.15,
            "fit_scope": "train_split_only_shape_diagnostic",
            "interpretation": "scalar_level_calibration_does_not_replace_plant_shape_ablation",
            "required_ablation_domains": [HISTORICAL_DOMAIN, CURRENT_DOMAIN],
        },
        "quality_gate": {
            "thresholds": {
                "heldout_split_median_max_abs_db": 1.0,
                "all_session_residual_max_abs_db": 6.0,
                "train_complex_agreement_min": 0.95,
                "train_complex_relative_error_max": 0.25,
                "calibrated_err_abs_peak_max": 0.8,
            },
            "observed": {
                "heldout_split_median_max_abs_db": 0.0,
                "all_session_residual_max_abs_db": 0.0,
                "train_complex_agreement": 0.988,
                "train_complex_relative_error": 0.15,
                "calibrated_err_abs_peak": 0.4,
            },
            "pass": True,
            "threshold_policy": "predeclared_not_result_tuned",
        },
        "sessions": sorted(sessions, key=lambda item: str(item["session_id"])),
    }
    receipt = root / "data/manifests/recorded_level_calibration/receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(_json_bytes(payload))
    return receipt, hashlib.sha256(receipt.read_bytes()).hexdigest(), ids


def _init_clean_source_repo(root: Path) -> str:
    (root / "scripts").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "scripts/calibration_issuer.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (root / "configs/fixture.yaml").write_text("value: 1\n", encoding="utf-8")
    (root / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "fixture"], cwd=root, check=True
    )
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


@pytest.mark.parametrize(
    "mutation",
    (
        "replace",
        "graft",
        "assume_unchanged",
        "skip_worktree",
        "tracked_bytes",
        "tracked_mode",
        "staged_index_blob",
        "ignored_executable",
    ),
)
def test_calibration_issuer_rejects_nonexact_source_state(
    tmp_path: Path,
    mutation: str,
) -> None:
    commit = _init_clean_source_repo(tmp_path)
    assert require_clean_exact_commit(tmp_path) == commit
    tracked = tmp_path / "scripts/calibration_issuer.py"
    if mutation == "replace":
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=tmp_path,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        replacement = subprocess.run(
            ["git", "commit-tree", tree],
            cwd=tmp_path,
            check=True,
            input="replacement commit\n",
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        subprocess.run(
            ["git", "replace", commit, replacement], cwd=tmp_path, check=True
        )
    elif mutation == "graft":
        (tmp_path / ".git/info/grafts").write_text(commit + "\n", encoding="ascii")
    elif mutation in {"assume_unchanged", "skip_worktree"}:
        option = (
            "--assume-unchanged"
            if mutation == "assume_unchanged"
            else "--skip-worktree"
        )
        subprocess.run(
            ["git", "update-index", option, "scripts/calibration_issuer.py"],
            cwd=tmp_path,
            check=True,
        )
    elif mutation == "tracked_bytes":
        tracked.write_text("VALUE = 2\n", encoding="utf-8")
    elif mutation == "tracked_mode":
        tracked.chmod(0o755)
    elif mutation == "staged_index_blob":
        tracked.write_text("VALUE = 2\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "scripts/calibration_issuer.py"],
            cwd=tmp_path,
            check=True,
        )
    else:
        with (tmp_path / ".git/info/exclude").open("a", encoding="utf-8") as handle:
            handle.write("/scripts/injected.py\n")
        injected = tmp_path / "scripts/injected.py"
        injected.write_text("raise SystemExit\n", encoding="utf-8")
        injected.chmod(0o755)

    with pytest.raises(RecordedLevelCalibrationError, match="clean exact source"):
        require_clean_exact_commit(tmp_path)


def test_receipt_rejects_clean_source_rebinding_even_if_resealed(
    tmp_path: Path,
) -> None:
    receipt, _digest, _ids = _write_receipt(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["clean_source"]["tracked_inventory_sha256"] = "9" * 64
    receipt.write_bytes(_json_bytes(payload))
    forged_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with pytest.raises(RecordedLevelCalibrationError, match="clean_source evidence"):
        validate_recorded_level_calibration_receipt(
            receipt,
            expected_sha256=forged_sha,
            repo_root=tmp_path,
            verify_current_commit=True,
        )


def test_receipt_rejects_val_session_in_train_fit(tmp_path: Path) -> None:
    receipt, digest, _ids = _write_receipt(tmp_path)
    validated = validate_recorded_level_calibration_receipt(
        receipt, expected_sha256=digest, repo_root=tmp_path
    )
    assert validated.err_gain_by_session["20260804_000000_file"] == 2.0
    payload = json.loads(receipt.read_text())
    payload["cohorts"]["historical_20260804"]["fit_session_ids"].append(
        "20260804_000035_file"
    )
    receipt.write_bytes(_json_bytes(payload))
    forged_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with pytest.raises(RecordedLevelCalibrationError, match="train split"):
        validate_recorded_level_calibration_receipt(
            receipt, expected_sha256=forged_sha, repo_root=tmp_path
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("sha256", "NOT-A-SHA"), ("size", 0), ("size", True)),
)
def test_receipt_rejects_malformed_audio_ref_without_rereading_audio(
    tmp_path: Path, field: str, value: object
) -> None:
    receipt, _digest, _ids = _write_receipt(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["sessions"][0]["mics"][field] = value
    receipt.write_bytes(_json_bytes(payload))
    forged_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with pytest.raises(RecordedLevelCalibrationError, match="path/size/SHA"):
        validate_recorded_level_calibration_receipt(
            receipt,
            expected_sha256=forged_sha,
            repo_root=tmp_path,
            verify_bound_audio=False,
        )


def test_receipt_rejects_result_tuned_shape_tolerance(tmp_path: Path) -> None:
    receipt, _digest, _ids = _write_receipt(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["plant_shape_diagnostic"]["relative_error_after_scalar_and_delay"] = 0.26
    payload["quality_gate"]["observed"]["train_complex_relative_error"] = 0.26
    receipt.write_bytes(_json_bytes(payload))
    forged_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with pytest.raises(RecordedLevelCalibrationError, match="threshold"):
        validate_recorded_level_calibration_receipt(
            receipt, expected_sha256=forged_sha, repo_root=tmp_path
        )


def test_receipt_rejects_post_gain_err_peak_outside_fixed_headroom(
    tmp_path: Path,
) -> None:
    receipt, _digest, ids = _write_receipt(tmp_path)
    payload = json.loads(receipt.read_text())
    target = next(item for item in payload["sessions"] if item["session_id"] == ids[0])
    cohort = payload["cohorts"][target["cohort"]]
    target["raw_err_abs_peak"] = 0.81 / float(cohort["err_amplitude_gain"])
    target["calibrated_err_abs_peak"] = 0.81
    receipt.write_bytes(_json_bytes(payload))
    forged_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with pytest.raises(RecordedLevelCalibrationError, match="0.8 안전 상한"):
        validate_recorded_level_calibration_receipt(
            receipt, expected_sha256=forged_sha, repo_root=tmp_path
        )


def test_receipt_writer_is_no_replace_and_rejects_symlink_parent(
    tmp_path: Path,
) -> None:
    output = tmp_path / "new" / "authority.json"
    written, digest = write_recorded_level_calibration_receipt(
        {"schema": "fixture", "value": 1}, output
    )
    assert written == output
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError, match="덮어쓰지"):
        write_recorded_level_calibration_receipt(
            {"schema": "fixture", "value": 2}, output
        )
    assert json.loads(output.read_text()) == {"schema": "fixture", "value": 1}

    actual_parent = tmp_path / "actual-parent"
    actual_parent.mkdir()
    symlink_parent = tmp_path / "symlink-parent"
    symlink_parent.symlink_to(actual_parent, target_is_directory=True)
    with pytest.raises(RecordedLevelCalibrationError, match="symlink"):
        write_recorded_level_calibration_receipt(
            {"schema": "fixture"}, symlink_parent / "forbidden.json"
        )
    assert not (actual_parent / "forbidden.json").exists()


@pytest.mark.parametrize(
    "output",
    (
        "/tmp/escape.json",
        "data/manifests/recorded_level_calibration/../escape.json",
        "data/manifests/other/escape.json",
        "data/manifests/recorded_level_calibration/not-json.txt",
    ),
)
def test_canonical_receipt_output_cannot_escape_repository_namespace(
    tmp_path: Path, output: str
) -> None:
    with pytest.raises(RecordedLevelCalibrationError, match="저장소 상대"):
        canonical_recorded_level_calibration_output(tmp_path, output)
    accepted = canonical_recorded_level_calibration_output(
        tmp_path,
        "data/manifests/recorded_level_calibration/clean-exact.json",
    )
    assert accepted == (
        tmp_path / "data/manifests/recorded_level_calibration/clean-exact.json"
    )


def test_measured_run_requires_external_receipt_sha_before_run_dir(tmp_path: Path) -> None:
    receipt, digest, _ids = _write_receipt(tmp_path)
    cfg = {
        "experiment_role": "measured_probe",
        "recorded_ratio": 0.7,
        "data": {"reference_mode": "digital"},
    }
    with pytest.raises(RecordedLevelCalibrationError, match="외부 SHA"):
        require_recorded_level_calibration_config(cfg, repo_root=tmp_path)
    cfg["data"].update(
        recorded_level_calibration=receipt.relative_to(tmp_path).as_posix(),
        recorded_level_calibration_sha256=digest,
    )
    assert require_recorded_level_calibration_config(cfg, repo_root=tmp_path) is not None
    cfg["data"]["reference_mode"] = "acoustic"
    with pytest.raises(RecordedLevelCalibrationError, match="digital-reference"):
        require_recorded_level_calibration_config(cfg, repo_root=tmp_path)


def test_dataset_applies_err_only_and_exact_half_current_schedule(tmp_path: Path) -> None:
    receipt, digest, ids = _write_receipt(tmp_path)
    manifest = tmp_path / "data/manifests/combined.jsonl"
    rows = []
    families = ("environment", "machine", "music", "speech")
    historical_ids = (ids[0], ids[1], ids[2], ids[3])
    for family, historical_id in zip(families, historical_ids, strict=True):
        for domain, session_id in (
            (HISTORICAL_DOMAIN, historical_id),
            (CURRENT_DOMAIN, f"current-{family}"),
        ):
            directory = tmp_path / f"data/fixture/{session_id}"
            directory.mkdir(parents=True)
            source = np.full(4096, 0.2, dtype=np.float32)
            mics = np.stack([np.full(4096, 0.1), np.full(4096, 0.03)], axis=1)
            sf.write(directory / "source_aligned.wav", source, 48_000, subtype="FLOAT")
            sf.write(directory / "mics.wav", mics, 48_000, subtype="FLOAT")
            (directory / "session.json").write_text(
                json.dumps(
                    {
                        "recording_level_campaign": {
                            "plant_domain": domain,
                        }
                    }
                )
            )
            rows.append(
                {
                    "path": str(directory),
                    "session_id": session_id,
                    "source_family": family,
                    "group_id": f"{family}-{domain}",
                    "source_pool_group_id": f"pool-{family}-{domain}",
                    "plant_domain": domain,
                    "split": "train",
                }
            )
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dataset = RecordedANCDataset(
        manifest,
        {
            "sample_rate": 48_000,
            "segment_seconds": 0.02,
            "reference_mode": "digital",
            "recorded_sampling": PLANT_DOMAIN_SAMPLING_MODE,
            "recorded_current_strict_min_fraction": 0.5,
            "recorded_level_calibration": receipt.relative_to(tmp_path).as_posix(),
            "recorded_level_calibration_sha256": digest,
            "recorded_augment": {"enabled": False},
            "closed_loop": {"feedback_delay_samples": [16, 17]},
        },
        transfer_repo_root=tmp_path,
    )
    domains = []
    selected_families = []
    for global_index in range(8):
        rng = indexed_rng(7, 0x524543, global_index, 0)
        index = dataset._sample_session_index(rng, global_index=global_index)
        selected_families.append(rows[index]["source_family"])
        domains.append(dataset._plant_domain(rows[index]))
    assert domains == [HISTORICAL_DOMAIN, CURRENT_DOMAIN] * 4
    assert selected_families == [value for family in families for value in (family, family)]
    assert dataset.current_strict_item_fraction == 0.5

    historical_index = next(
        index for index, row in enumerate(rows) if row["session_id"] == historical_ids[0]
    )
    current_index = next(
        index for index, row in enumerate(rows) if row["session_id"] == "current-environment"
    )
    historical_err, historical_ref, historical_source = dataset._session(historical_index)
    current_err, current_ref, current_source = dataset._session(current_index)
    assert np.allclose(historical_err, 0.2, atol=1e-6)  # Aug04 gain=2, ERR만
    assert np.allclose(current_err, 0.1, atol=1e-6)
    assert np.allclose(historical_ref, current_ref)
    assert np.allclose(historical_source, current_source)


def test_sampler_blocks_family_without_current_train_domain(tmp_path: Path) -> None:
    receipt, digest, ids = _write_receipt(tmp_path)
    directory = tmp_path / "data/only-historical"
    directory.mkdir(parents=True)
    source = np.zeros(4096, dtype=np.float32)
    sf.write(directory / "source_aligned.wav", source, 48_000, subtype="FLOAT")
    sf.write(directory / "mics.wav", np.stack([source, source], axis=1), 48_000, subtype="FLOAT")
    (directory / "session.json").write_text("{}")
    manifest = tmp_path / "data/manifests/incomplete.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "path": str(directory),
                "session_id": ids[0],
                "source_family": "speech",
                "group_id": "speech-component",
                "source_pool_group_id": "speech-pool",
                "split": "train",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="모든 family"):
        RecordedANCDataset(
            manifest,
            {
                "sample_rate": 48_000,
                "segment_seconds": 0.02,
                "reference_mode": "digital",
                "recorded_sampling": PLANT_DOMAIN_SAMPLING_MODE,
                "recorded_level_calibration": receipt.relative_to(tmp_path).as_posix(),
                "recorded_level_calibration_sha256": digest,
                "recorded_augment": {"enabled": False},
                "closed_loop": {"feedback_delay_samples": [16, 17]},
            },
            transfer_repo_root=tmp_path,
        )
