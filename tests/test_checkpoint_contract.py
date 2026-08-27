"""resume experiment 계약과 fail-closed 상태 복원 회귀 테스트."""

from __future__ import annotations

import json
import hashlib
import random
import shutil
import subprocess
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from deep_anc.config import REPO_ROOT, load_train_config, load_yaml
from deep_anc.data.resumable_stream import indexed_rng, worker_global_item_indices
from deep_anc.train.checkpoint import load_checkpoint, save_checkpoint
from deep_anc.train.campaign_prerequisite import (
    CANONICAL_PATH as CANONICAL_PREREQUISITE_PATH,
    validate_canonical_pretrain_prerequisites,
)
from deep_anc.train.completion_receipt import (
    validate_completion_receipt,
    write_completion_receipt,
)
from deep_anc.train.experiment_contract import (
    build_experiment_contract,
    require_canonical_source_trust,
    stamp_experiment_contract,
    validate_resume_experiment,
)
from deep_anc.train.reproducibility import _publish_or_validate
from deep_anc.train.trainer import (
    cfg_snapshot,
    configure_canonical_determinism,
    validate_canonical_run_entry,
    validate_init_checkpoint_role,
    validate_resume_physics,
    validate_training_world_size,
)


_BOOTSTRAP_SHA = "a" * 64
_PREREQUISITE_SHA = "d" * 64


def _load_bound_canonical(path, overrides: list[str] | None = None) -> dict:
    """정책 단위 테스트용 검증 완료 transfer generation stub.

    transfer bytes 자체의 공격 검증은 test_elice_transfer_contract가 담당한다.
    여기서는 binder가 stamp 전에 주입한 resolved 필드가 계약에 남는 경계만 쓴다.
    """

    def _bind(data: dict, *, repo_root):
        assert data["bootstrap_receipt_sha256"] == _BOOTSTRAP_SHA
        data.update(
            transfer_manifest="data/manifests/elice_transfer_manifest.json",
            transfer_manifest_sha256="b" * 64,
            recorded_transfer_aggregate_sha256="c" * 64,
        )

    values = list(overrides or [])
    values.extend(
        [
            f"data.bootstrap_receipt_sha256={_BOOTSTRAP_SHA}",
            f"campaign_prerequisite_sha256={_PREREQUISITE_SHA}",
        ]
    )
    with patch(
        "deep_anc.data.transfer_contract.bind_recorded_transfer_config", _bind
    ), patch(
        "deep_anc.train.experiment_contract._git_source_state",
        return_value={
            "git_commit": "1" * 40,
            "source_tree_sha256": "2" * 64,
            "verifiable": True,
            "clean_exact_commit": True,
            "dirty_paths": [],
            "replace_refs": [],
            "index_flags_clean": True,
        },
    ):
        return load_train_config(path, values)


def _config(tmp_path) -> dict:
    secondary = tmp_path / "secondary.npz"
    primary = tmp_path / "primary.npz"
    rir = tmp_path / "rir.npz"
    recorded = tmp_path / "recorded.jsonl"
    for path, payload in (
        (secondary, b"secondary-v1"),
        (primary, b"primary-v1"),
        (rir, b"rir-v1"),
        (recorded, b'{"split":"train"}\n'),
    ):
        path.write_bytes(payload)
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    source_manifest = manifest_dir / "speech.jsonl"
    source_manifest.write_text('{"path":"speech.wav","split":"train"}\n')
    holdout = manifest_dir / "recorded_holdout.json"
    holdout.write_text('{"families":{}}\n')
    generation = manifest_dir / "manifest_generation.json"
    generation.write_text(
        '{"holdout":' + repr(str(holdout)).replace("'", '"') + '}\n'
    )
    return {
        "stage": "open_loop",
        "model": {"name": "hybrid_anc_tiny", "channels": 128},
        "data": {
            "sample_rate": 48_000,
            "reference_mode": "digital",
            "digital_primary_path_mode": "measured",
            "digital_reference_lead_samples": 113,
            "rir_bank": str(rir),
            "noise_manifest_dir": str(manifest_dir),
            "source_mix_ratio": {"synthetic": 0.5, "speech": 0.5},
        },
        "duct": {
            "secondary_path": {"npz": str(secondary), "handoff_extra_samples": 256},
            "digital_reference": {"primary_path_npz": str(primary)},
        },
        "loss": {
            "nmse_objective": "trusted_band",
            "nmse_cvar_alpha": 0.7,
            "band_weight": "trusted_only",
        },
        "optimizer": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-4},
        "schedule": {"warmup_steps": 1_000, "total_steps": 50_000, "min_lr": 1e-6},
        "recorded_manifest": str(recorded),
        "recorded_ratio": 0.7,
        "batch_size": 16,
        "seed": 20260803,
        "require_measured_primary_path": True,
    }


def _training_state() -> dict:
    return {
        "schema_version": 1,
        "plant_rng": deepcopy(np.random.default_rng(17).bit_generator.state),
        "nonlinear_rng": None,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda cfg: cfg["model"].update(channels=64),
        lambda cfg: cfg["loss"].update(nmse_cvar_alpha=1.0),
        lambda cfg: cfg["optimizer"].update(lr=2e-4),
        lambda cfg: cfg["schedule"].update(total_steps=40_000),
        lambda cfg: cfg.update(recorded_ratio=0.5),
        lambda cfg: cfg["data"].update(digital_reference_lead_samples=116),
    ],
)
def test_resume_rejects_any_training_semantics_change(tmp_path, mutation):
    cfg = _config(tmp_path)
    saved = cfg_snapshot(cfg, trusted_band_hz=(150.0, 1_600.0))
    current = deepcopy(cfg)
    mutation(current)
    current = cfg_snapshot(current, trusted_band_hz=(150.0, 1_600.0))
    with pytest.raises(ValueError, match="experiment contract 불일치"):
        validate_resume_physics({"cfg": saved}, current)


def test_resume_rejects_artifact_content_change_at_the_same_path(tmp_path):
    cfg = _config(tmp_path)
    saved = cfg_snapshot(cfg, trusted_band_hz=(150.0, 1_600.0))
    primary = tmp_path / "primary.npz"
    primary.write_bytes(b"primary-v2")
    current = cfg_snapshot(cfg, trusted_band_hz=(150.0, 1_600.0))

    with pytest.raises(ValueError, match="artifacts.primary_path"):
        validate_resume_experiment({"cfg": saved}, current)


def test_resume_contract_includes_git_generation_and_holdout_fingerprints(tmp_path):
    cfg = _config(tmp_path)
    saved = cfg_snapshot(cfg, trusted_band_hz=(150.0, 1_600.0))
    contract = saved["experiment_contract"]
    assert contract["source"]["git_commit"]
    assert contract["artifacts"]["source_manifest_generation"]["exists"]
    assert contract["artifacts"]["recorded_holdout"]["exists"]

    holdout = tmp_path / "manifests" / "recorded_holdout.json"
    holdout.write_text('{"families":{"speech":[]}}\n')
    current = cfg_snapshot(cfg, trusted_band_hz=(150.0, 1_600.0))
    with pytest.raises(ValueError, match="artifacts.recorded_holdout"):
        validate_resume_experiment({"cfg": saved}, current)


def test_best_metric_progress_is_metadata_not_a_static_contract_field(tmp_path):
    cfg = _config(tmp_path)
    saved = cfg_snapshot(cfg, trusted_band_hz=(150.0, 1_600.0))
    state = {"cfg": saved, "best_metric": -3.0}
    state["best_metric"] = -9.0
    current = cfg_snapshot(cfg, trusted_band_hz=(150.0, 1_600.0))
    validate_resume_experiment(state, current)


def test_checkpoint_restore_reproduces_global_rng_and_requires_complete_state(tmp_path):
    cfg = cfg_snapshot(_config(tmp_path), trusted_band_hz=(150.0, 1_600.0))
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    checkpoint = tmp_path / "last.pt"
    save_checkpoint(
        checkpoint, model, optimizer, scheduler, 3, -1.0, cfg,
        training_state=_training_state(),
    )

    expected = (random.random(), float(np.random.rand()), torch.rand(4))
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    load_checkpoint(checkpoint, model, optimizer, scheduler, restore_rng=True)
    actual = (random.random(), float(np.random.rand()), torch.rand(4))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])

    incomplete = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incomplete.pop("optimizer")
    broken = tmp_path / "broken" / "last.pt"
    broken.parent.mkdir()
    torch.save(incomplete, broken)
    with pytest.raises(ValueError, match="optimizer 상태"):
        load_checkpoint(broken, model, optimizer, scheduler, restore_rng=True)


def test_rng_restore_error_is_not_silently_ignored(tmp_path):
    cfg = cfg_snapshot(_config(tmp_path), trusted_band_hz=(150.0, 1_600.0))
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "last.pt"
    save_checkpoint(
        checkpoint, model, optimizer, None, 0, 0.0, cfg,
        training_state=_training_state(),
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state["rng"]["python"] = "not-a-random-state"
    torch.save(state, checkpoint)

    with pytest.raises(RuntimeError, match="RNG 상태"):
        load_checkpoint(checkpoint, model, optimizer, restore_rng=True)


class _IndexedToyDataset(IterableDataset):
    def __init__(self, *, start_batch: int, batch_size: int) -> None:
        self.start_batch = int(start_batch)
        self.batch_size = int(batch_size)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        for index in worker_global_item_indices(
            start_batch_index=self.start_batch,
            batch_size=self.batch_size,
            worker_id=worker_id,
            num_workers=workers,
        ):
            rng = indexed_rng(20260826, 7, index)
            x = torch.tensor(rng.normal(size=3), dtype=torch.float32)
            y = torch.tensor([0.4 * x[0] - 0.2 * x[1] + 0.1], dtype=torch.float32)
            yield {"index": index, "x": x, "y": y}


def _toy_components():
    torch.manual_seed(91)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 - 0.01 * min(step, 50)
    )
    return model, optimizer, scheduler


def _toy_run(model, optimizer, scheduler, *, start: int, steps: int, workers: int):
    batch_size = 4
    loader = DataLoader(
        _IndexedToyDataset(start_batch=start, batch_size=batch_size),
        batch_size=batch_size,
        num_workers=workers,
        prefetch_factor=2 if workers else None,
    )
    batches = []
    iterator = iter(loader)
    for _ in range(steps):
        batch = next(iterator)
        batches.append(batch["index"].clone())
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["x"])
        loss = torch.nn.functional.mse_loss(prediction, batch["y"])
        loss.backward()
        optimizer.step()
        scheduler.step()
    return batches


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def test_uninterrupted_and_global_index_resume_are_numerically_identical(tmp_path):
    """N step와 K+resume N step의 batch/model/optimizer/scheduler가 worker 수 변경에도 같다."""

    total, cut = 9, 4
    full_model, full_optimizer, full_scheduler = _toy_components()
    full_batches = _toy_run(
        full_model, full_optimizer, full_scheduler, start=0, steps=total, workers=2
    )

    part_model, part_optimizer, part_scheduler = _toy_components()
    first_batches = _toy_run(
        part_model, part_optimizer, part_scheduler, start=0, steps=cut, workers=1
    )
    cfg = cfg_snapshot(_config(tmp_path), trusted_band_hz=(150.0, 1_600.0))
    checkpoint = tmp_path / "exact" / "last.pt"
    save_checkpoint(
        checkpoint,
        part_model,
        part_optimizer,
        part_scheduler,
        cut,
        -1.0,
        cfg,
        training_state=_training_state(),
    )

    resumed_model, resumed_optimizer, resumed_scheduler = _toy_components()
    state = load_checkpoint(
        checkpoint,
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        restore_rng=True,
    )
    assert state["data_stream"]["global_batch_index"] == cut
    remaining_batches = _toy_run(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        start=cut,
        steps=total - cut,
        workers=3,
    )

    assert all(
        torch.equal(left, right)
        for left, right in zip(full_batches, first_batches + remaining_batches)
    )
    _assert_nested_equal(full_model.state_dict(), resumed_model.state_dict())
    _assert_nested_equal(full_optimizer.state_dict(), resumed_optimizer.state_dict())
    _assert_nested_equal(full_scheduler.state_dict(), resumed_scheduler.state_dict())


def test_canonical_tiny_config_matches_the_base_training_contract():
    base = load_yaml(REPO_ROOT / "configs/train_pretrain.yaml")
    tiny = load_yaml(REPO_ROOT / "configs/train_pretrain_tiny.yaml")
    for key in (
        "stage",
        "data_config",
        "duct_config",
        "batch_size",
        "num_workers",
        "prefetch_factor",
        "optimizer",
        "schedule",
        "amp",
        "grad_clip",
        "loss",
        "val_items",
        "eval_every",
        "log_every",
        "early_stop_patience",
    ):
        assert tiny[key] == base[key], key
    assert tiny["model_config"] == "configs/model_tiny.yaml"
    assert tiny["experiment_role"] == "canonical_pretrain"
    assert tiny["init_eligible"] is True
    assert tiny["contract_run_dir"] is True
    assert tiny["ckpt_dir"] == "runs"
    assert tiny["seed"] == 20260803
    assert tiny["required_world_size"] == 1
    assert "tiny_wide" not in tiny["model_config"]


def test_official_training_is_single_gpu_and_ddp_resume_fails_closed():
    canonical = load_yaml(REPO_ROOT / "configs/train_pretrain_tiny.yaml")
    finetune = load_yaml(REPO_ROOT / "configs/train_finetune.yaml")
    assert canonical["required_world_size"] == finetune["required_world_size"] == 1
    validate_training_world_size(canonical, 1)
    with pytest.raises(RuntimeError, match="world-size 계약"):
        validate_training_world_size(canonical, 2)
    with pytest.raises(RuntimeError, match="DDP exact-resume"):
        validate_training_world_size({"resume": "last.pt"}, 2)


def test_canonical_config_requires_bootstrap_and_prerequisite_anchors_before_stamp():
    with pytest.raises(ValueError, match="bootstrap_receipt_sha256"):
        load_train_config(REPO_ROOT / "configs/train_pretrain_tiny.yaml")

    def _bind(data: dict, *, repo_root):
        data.update(
            transfer_manifest="data/manifests/elice_transfer_manifest.json",
            transfer_manifest_sha256="b" * 64,
            recorded_transfer_aggregate_sha256="c" * 64,
        )

    with patch(
        "deep_anc.data.transfer_contract.bind_recorded_transfer_config", _bind
    ):
        with pytest.raises(ValueError, match="campaign prerequisite"):
            load_train_config(
                REPO_ROOT / "configs/train_pretrain_tiny.yaml",
                [f"data.bootstrap_receipt_sha256={_BOOTSTRAP_SHA}"],
            )


def test_approved_alpha_frame_pilot_grid_has_distinct_20k_contracts():
    base = _load_bound_canonical(REPO_ROOT / "configs/train_pretrain_tiny.yaml")
    run_dirs = set()
    for alpha in (0.7, 1.0):
        for frame in (0.5, 0.2):
            pilot = load_train_config(
                REPO_ROOT / "configs/train_pretrain_tiny.yaml",
                [
                    "experiment_role=loss_pilot",
                    "init_eligible=false",
                    f"loss.nmse_cvar_alpha={alpha}",
                    f"loss.lambda_frame={frame}",
                    "run_until_step=20000",
                ],
            )
            assert pilot["loss"]["nmse_cvar_alpha"] == alpha
            assert pilot["loss"]["lambda_frame"] == frame
            assert pilot["experiment_role"] == "loss_pilot"
            assert pilot["init_eligible"] is False
            assert pilot["run_until_step"] == 20_000
            run_dirs.add(pilot["ckpt_dir"])
    assert len(run_dirs) == 4
    assert base["ckpt_dir"] not in run_dirs


def test_measured_primary_five_k_probe_is_an_explicit_distinct_contract():
    base = _load_bound_canonical(REPO_ROOT / "configs/train_pretrain_tiny.yaml")
    probe = load_train_config(
        REPO_ROOT / "configs/train_pretrain_tiny.yaml",
        [
            "experiment_role=measured_probe",
            "init_eligible=false",
            "data.digital_primary_path_mode=measured",
            "run_until_step=5000",
            "init_ckpt=runs/selected-20k/ckpt/best.pt",
        ],
    )
    assert probe["data"]["digital_primary_path_mode"] == "measured"
    assert probe["experiment_role"] == "measured_probe"
    assert probe["init_eligible"] is False
    assert probe["run_until_step"] == 5_000
    assert probe["init_ckpt"].endswith("selected-20k/ckpt/best.pt")
    assert probe["ckpt_dir"] != base["ckpt_dir"]


def test_canonical_smoke_stop_budget_keeps_the_long_run_contract():
    full = _load_bound_canonical(REPO_ROOT / "configs/train_pretrain_tiny.yaml")
    smoke = _load_bound_canonical(
        REPO_ROOT / "configs/train_pretrain_tiny.yaml",
        ["run_until_step=500"],
    )
    assert smoke["run_until_step"] == 500
    assert smoke["ckpt_dir"] == full["ckpt_dir"]
    assert (
        smoke["resolved_contract_run_dir"]["experiment_contract_sha256"]
        == full["resolved_contract_run_dir"]["experiment_contract_sha256"]
    )
    saved = cfg_snapshot(smoke, trusted_band_hz=(150.0, 1_600.0))
    current = cfg_snapshot(full, trusted_band_hz=(150.0, 1_600.0))
    with patch(
        "deep_anc.train.experiment_contract._git_source_state",
        return_value=full["experiment_contract"]["source"],
    ):
        validate_resume_experiment({"cfg": saved}, current)


@pytest.mark.parametrize(
    "saved_cfg, message",
    [
        (
            {"experiment_role": "loss_pilot", "init_eligible": False},
            "experiment_role",
        ),
        (
            {"experiment_role": "canonical_pretrain", "init_eligible": False},
            "init_eligible=true",
        ),
    ],
)
def test_pilot_or_ineligible_checkpoint_cannot_initialize_official_finetune(
    saved_cfg, message
):
    finetune = {
        "require_init_checkpoint": True,
        "required_init_experiment_role": "canonical_pretrain",
        "require_init_eligible": True,
    }
    with pytest.raises(ValueError, match=message):
        validate_init_checkpoint_role({"cfg": saved_cfg}, finetune)
    validate_init_checkpoint_role(
        {
            "cfg": {
                "experiment_role": "canonical_pretrain",
                "init_eligible": True,
            }
        },
        finetune,
    )


@pytest.mark.parametrize(
    "override, message",
    [
        ("contract_run_dir=false", "contract_run_dir=true"),
        (
            "data.recorded_sampling=uniform_session",
            "family_lineage_session_balanced",
        ),
    ],
)
def test_official_finetune_cannot_override_contract_run_or_lineage_sampler(
    override, message
):
    with pytest.raises(ValueError, match=message):
        load_train_config(
            REPO_ROOT / "configs/train_finetune.yaml",
            ["data.digital_primary_path_mode=measured", override],
        )


@pytest.mark.parametrize(
    "override",
    [
        "experiment_role=loss_pilot",
        "require_init_checkpoint=false",
        "required_init_experiment_role=loss_pilot",
        "require_init_eligible=false",
        "readiness.require_completed_init_checkpoint=false",
        "readiness.min_recorded_sessions=1",
    ],
)
def test_canonical_finetune_trust_policy_cannot_be_weakened(override):
    with pytest.raises(ValueError, match="trust policy"):
        load_train_config(
            REPO_ROOT / "configs/train_finetune.yaml",
            ["data.digital_primary_path_mode=measured", override],
        )


@pytest.mark.parametrize(
    "override",
    [
        "stage=closed_loop",
        "schedule.total_steps=1",
        "amp=null",
        "freeze_encoder=true",
        "model.encoder.channels=64",
        "seed=1",
        "loss.lambda_dnh=0.0",
        "data.recorded_augment.mix_probability=0.1",
        "recorded_ratio=0.1",
    ],
)
def test_canonical_finetune_full_training_policy_cannot_be_weakened(override):
    with pytest.raises(ValueError, match="trust policy"):
        load_train_config(
            REPO_ROOT / "configs/train_finetune.yaml",
            ["data.digital_primary_path_mode=measured", override],
        )


@pytest.mark.parametrize(
    "override",
    [
        "schedule.total_steps=500",
        "stage=closed_loop",
        "amp=null",
        "model.tcn.repeats=1",
        "optimizer.lr=0.01",
        "seed=1",
        "init_eligible=false",
    ],
)
def test_canonical_pretrain_cannot_be_promoted_from_a_changed_contract(override):
    with pytest.raises(ValueError, match="canonical_pretrain trust policy"):
        load_train_config(
            REPO_ROOT / "configs/train_pretrain_tiny.yaml", [override]
        )


def test_canonical_loss_identity_must_match_between_pretrain_and_finetune():
    pretrain = _load_bound_canonical(REPO_ROOT / "configs/train_pretrain_tiny.yaml")
    finetune = _load_bound_canonical(
        REPO_ROOT / "configs/train_finetune.yaml",
        ["data.digital_primary_path_mode=measured"],
    )
    validate_init_checkpoint_role({"cfg": pretrain}, finetune)
    mismatched = _load_bound_canonical(
        REPO_ROOT / "configs/train_pretrain_tiny.yaml",
        ["loss.nmse_cvar_alpha=1.0"],
    )
    with pytest.raises(ValueError, match="loss selection"):
        validate_init_checkpoint_role({"cfg": mismatched}, finetune)


def _prerequisite_components(tmp_path, *, gradient_share: float = 0.3):
    source_root = tmp_path / "campaign-source"
    (source_root / "src").mkdir(parents=True)
    (source_root / "src" / "training.py").write_text("VALUE = 1\n")
    (source_root / ".gitignore").write_text("/results/\n/runs/\n")
    cfg = _load_bound_canonical(REPO_ROOT / "configs/train_pretrain_tiny.yaml")
    artifact_paths = [
        cfg["duct"]["secondary_path"]["npz"],
        cfg["duct"]["digital_reference"]["primary_path_npz"],
        cfg["data"]["bootstrap_receipt"],
        cfg["data"]["transfer_manifest"],
    ]
    for relative in artifact_paths:
        target = source_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"fixture:{relative}\n".encode())
    subprocess.run(["git", "init", "-q"], cwd=source_root, check=True)
    subprocess.run(["git", "add", "."], cwd=source_root, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=tests", "-c",
            "user.email=tests@example.invalid", "commit", "-qm", "source",
        ],
        cwd=source_root,
        check=True,
    )

    evidence_dir = source_root / "results" / "training_prerequisites" / "evidence"
    evidence_dir.mkdir(parents=True)
    evidence = evidence_dir / "verified.json"
    evidence.write_text('{"verified":true}\n')
    evidence_ref = {
        "path": str(evidence.relative_to(source_root)),
        "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
    }
    environment = evidence_dir / "a100-environment.json"
    environment.write_text(
        json.dumps(
            {
                "python": "fixture",
                "torch": "2.5.1+cu121",
                "torch_cuda": "12.1",
                "cuda_available": True,
                "device_count": 1,
                "devices": [
                    {"index": 0, "name": "NVIDIA A100-SXM4-40GB", "capability": [8, 0]}
                ],
                "deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cublas_workspace_config": ":4096:8",
            },
            sort_keys=True,
        )
        + "\n"
    )
    environment_ref = {
        "path": str(environment.relative_to(source_root)),
        "sha256": hashlib.sha256(environment.read_bytes()).hexdigest(),
    }
    provisional = stamp_experiment_contract(cfg, repo_root=source_root)
    source = provisional["experiment_contract"]["source"]
    artifacts = provisional["experiment_contract"]["artifacts"]
    candidates = []
    for alpha, frame, score in (
        (0.7, 0.5, -4.0),
        (0.7, 0.2, -3.0),
        (1.0, 0.5, -2.0),
        (1.0, 0.2, -1.0),
    ):
        candidates.append(
            {
                "alpha": alpha,
                "lambda_frame": frame,
                "run_until_step": 20_000,
                "experiment_role": "loss_pilot",
                "init_eligible": False,
                "score_db": score,
                "evidence": evidence_ref,
            }
        )
    ledger = {
        "schema_version": 2,
        "source": {
            "git_commit": source["git_commit"],
            "source_tree_sha256": source["source_tree_sha256"],
            "bootstrap_receipt_sha256": cfg["data"]["bootstrap_receipt_sha256"],
            "primary_path_sha256": artifacts["primary_path"]["sha256"],
            "secondary_path_sha256": artifacts["secondary_path"]["sha256"],
        },
        "g0": {
            "evidence": evidence_ref,
            "all_finite": True,
            "nmse_trusted_db": -6.1,
            "threshold_exclusive_db": -6.0,
        },
        "gradient_budget": {
            "evidence": evidence_ref,
            "strict_ps": True,
            "lambda_dnh": cfg["loss"]["lambda_dnh"],
            "gradient_share": gradient_share,
            "loss_start_sample": cfg["loss_start_sample"],
        },
        "loss_pilot_selection": {
            "selection_rule": "minimum_recorded_val_score_db",
            "conditional_alpha_085_triggered": False,
            "candidates": candidates,
            "winner": {"alpha": 0.7, "lambda_frame": 0.5},
        },
        "measured_probe": {
            "evidence": evidence_ref,
            "experiment_role": "measured_probe",
            "init_eligible": False,
            "run_until_step": 5_000,
            "strict_ps": True,
            "passed": True,
            "alpha": 0.7,
            "lambda_frame": 0.5,
        },
        "a100_smoke_resume": {
            "evidence": evidence_ref,
            "environment_receipt": environment_ref,
            "device": "NVIDIA A100-SXM4-40GB",
            "cuda": True,
            "amp": "bf16",
            "stop_step": 300,
            "resumed_step": 500,
            "model_equal": True,
            "optimizer_equal": True,
            "scheduler_equal": True,
            "batch_sequence_equal": True,
        },
    }
    ledger_path = source_root / CANONICAL_PREREQUISITE_PATH
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(ledger, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cfg["campaign_prerequisite_sha256"] = hashlib.sha256(
        ledger_path.read_bytes()
    ).hexdigest()
    cfg = stamp_experiment_contract(cfg, repo_root=source_root)
    return cfg, source_root, ledger_path


def test_canonical_pretrain_prerequisites_bind_every_campaign_gate(tmp_path):
    cfg, source_root, ledger_path = _prerequisite_components(tmp_path)
    ledger = validate_canonical_pretrain_prerequisites(
        cfg, repo_root=source_root
    )
    assert ledger["g0"]["nmse_trusted_db"] < -6.0
    assert ledger["a100_smoke_resume"]["batch_sequence_equal"] is True

    evidence = source_root / json.loads(ledger_path.read_text())["g0"]["evidence"]["path"]
    evidence.write_text('{"verified":false}\n')
    with pytest.raises(ValueError, match="evidence bytes SHA"):
        validate_canonical_pretrain_prerequisites(cfg, repo_root=source_root)


def test_canonical_pretrain_prerequisite_rejects_gradient_share_outside_budget(
    tmp_path,
):
    cfg, source_root, _ = _prerequisite_components(
        tmp_path, gradient_share=0.199999
    )
    with pytest.raises(ValueError, match="0.2–0.4"):
        validate_canonical_pretrain_prerequisites(cfg, repo_root=source_root)


def _completion_components(tmp_path):
    source_root = tmp_path / "source"
    (source_root / "src").mkdir(parents=True)
    (source_root / "src" / "training.py").write_text("VALUE = 1\n")
    (source_root / ".gitignore").write_text("/runs/\n")
    subprocess.run(["git", "init", "-q"], cwd=source_root, check=True)
    subprocess.run(["git", "add", "src", ".gitignore"], cwd=source_root, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=tests", "-c",
            "user.email=tests@example.invalid", "commit", "-qm", "source",
        ],
        cwd=source_root,
        check=True,
    )
    cfg = _load_bound_canonical(REPO_ROOT / "configs/train_pretrain_tiny.yaml")
    cfg["data"].update(
        bootstrap_receipt="data/manifests/elice_bootstrap_receipt.json",
        bootstrap_receipt_sha256="a" * 64,
        transfer_manifest="data/manifests/elice_transfer_manifest.json",
        transfer_manifest_sha256="b" * 64,
        recorded_transfer_aggregate_sha256="c" * 64,
    )
    cfg = stamp_experiment_contract(cfg, repo_root=source_root)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    ckpt_dir = tmp_path / "run" / "ckpt"
    ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root, check=True,
        capture_output=True, text=True,
    ).stdout
    (ckpt_dir.parent / "config_snapshot.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (ckpt_dir.parent / "git_rev.txt").write_text(commit, encoding="utf-8")
    (ckpt_dir.parent / "pip_freeze.txt").write_text("fixture==1\n", encoding="utf-8")
    (ckpt_dir.parent / "environment.json").write_text(
        json.dumps(
            {
                "python": "fixture",
                "torch": "fixture",
                "cuda_available": False,
                "device_count": 0,
                "devices": [],
                "deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cublas_workspace_config": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    save_checkpoint(
        ckpt_dir / "best.pt", model, optimizer, scheduler, 500, -1.0, cfg,
        training_state=_training_state(),
    )
    return cfg, model, optimizer, scheduler, ckpt_dir, source_root


@pytest.mark.parametrize("stop_kind, stopped_step", [("run_until", 4), ("early_stop", 9)])
def test_operational_stop_never_creates_canonical_completion_receipt(
    tmp_path, stop_kind, stopped_step
):
    cfg, model, optimizer, scheduler, ckpt_dir, source_root = _completion_components(tmp_path)
    if stop_kind == "run_until":
        cfg["run_until_step"] = stopped_step
    save_checkpoint(
        ckpt_dir / "last.pt", model, optimizer, scheduler, stopped_step, -1.0, cfg,
        training_state=_training_state(),
    )
    with pytest.raises(ValueError, match="immutable schedule"):
        write_completion_receipt(ckpt_dir, repo_root=source_root)
    assert not (ckpt_dir / "completion.json").exists()


def test_completion_receipt_is_no_replace_and_binds_exact_checkpoint_bytes(tmp_path):
    cfg, model, optimizer, scheduler, ckpt_dir, source_root = _completion_components(tmp_path)
    save_checkpoint(
        ckpt_dir / "last.pt", model, optimizer, scheduler,
        cfg["schedule"]["total_steps"], -1.0, cfg,
        training_state=_training_state(),
    )
    receipt = write_completion_receipt(ckpt_dir, repo_root=source_root)
    assert receipt == ckpt_dir / "completion.json"
    validate_completion_receipt(
        ckpt_dir, expected_role="canonical_pretrain", expected_init_eligible=True,
        repo_root=source_root,
    )
    original = receipt.read_bytes()
    write_completion_receipt(ckpt_dir, repo_root=source_root)
    assert receipt.read_bytes() == original

    state = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
    state["best_metric"] = -2.0
    torch.save(state, ckpt_dir / "last.pt")
    with pytest.raises(ValueError, match="선택 metric|last_checkpoint_sha256"):
        validate_completion_receipt(ckpt_dir, repo_root=source_root)


def test_completion_rejects_best_not_selected_by_final_last_metric(tmp_path):
    cfg, model, optimizer, scheduler, ckpt_dir, source_root = _completion_components(tmp_path)
    save_checkpoint(
        ckpt_dir / "last.pt", model, optimizer, scheduler,
        cfg["schedule"]["total_steps"], -2.0, cfg,
        training_state=_training_state(),
    )
    with pytest.raises(ValueError, match="선택 metric"):
        write_completion_receipt(ckpt_dir, repo_root=source_root)
    assert not (ckpt_dir / "completion.json").exists()


def test_canonical_determinism_requires_cuda_workspace_and_sets_backend(
    monkeypatch,
):
    cfg = _load_bound_canonical(REPO_ROOT / "configs/train_pretrain_tiny.yaml")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        configure_canonical_determinism(cfg, cuda_available=True)

    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    try:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        configure_canonical_determinism(cfg, cuda_available=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
    finally:
        torch.use_deterministic_algorithms(previous[0])
        torch.backends.cudnn.benchmark = previous[1]
        torch.backends.cudnn.deterministic = previous[2]


def test_canonical_reproducibility_artifact_rejects_symlink(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"same\n")
    target = tmp_path / "config_snapshot.yaml"
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="regular-file snapshot"):
        _publish_or_validate(target, b"same\n")


def test_forged_or_symlinked_completion_receipt_is_rejected_without_replacement(
    tmp_path,
):
    cfg, model, optimizer, scheduler, ckpt_dir, source_root = _completion_components(tmp_path)
    save_checkpoint(
        ckpt_dir / "last.pt", model, optimizer, scheduler,
        cfg["schedule"]["total_steps"], -1.0, cfg,
        training_state=_training_state(),
    )
    receipt = ckpt_dir / "completion.json"
    receipt.write_text('{"schema_version":1}\n')
    forged = receipt.read_bytes()
    with pytest.raises(ValueError, match="receipt"):
        write_completion_receipt(ckpt_dir, repo_root=source_root)
    assert receipt.read_bytes() == forged

    receipt.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n")
    receipt.symlink_to(outside)
    with pytest.raises(ValueError, match="regular-file snapshot"):
        validate_completion_receipt(ckpt_dir, repo_root=source_root)


def test_canonical_contract_fails_without_git_but_keeps_portable_source_digest(
    tmp_path,
):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "model.py").write_text("VALUE = 1\n")
    diagnostic = build_experiment_contract(
        {"experiment_role": "diagnostic"}, repo_root=tmp_path
    )
    assert diagnostic["source"]["verifiable"] is False
    assert len(diagnostic["source"]["source_tree_sha256"]) == 64
    canonical = stamp_experiment_contract(
        {"experiment_role": "canonical_pretrain"}, repo_root=tmp_path
    )
    with pytest.raises(ValueError, match="git commit"):
        require_canonical_source_trust(canonical)


def test_canonical_source_trust_requires_clean_unchanged_exact_commit(tmp_path):
    cfg, _, _, _, _, source_root = _completion_components(tmp_path)
    require_canonical_source_trust(cfg, repo_root=source_root)
    (source_root / "src" / "training.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="source_tree_sha256|clean exact"):
        require_canonical_source_trust(cfg, repo_root=source_root)


def test_canonical_source_trust_rejects_dirty_docs_or_tests_too(tmp_path):
    cfg, _, _, _, _, source_root = _completion_components(tmp_path)
    (source_root / "docs").mkdir()
    (source_root / "docs" / "note.md").write_text("untracked\n")
    with pytest.raises(ValueError, match="clean exact|source_tree_sha256"):
        require_canonical_source_trust(cfg, repo_root=source_root)


def test_resume_preview_rejects_nonfinite_optimizer_before_model_mutation(tmp_path):
    cfg = cfg_snapshot(_config(tmp_path), trusted_band_hz=(150.0, 1_600.0))
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    checkpoint = tmp_path / "resume" / "last.pt"
    save_checkpoint(
        checkpoint, model, optimizer, scheduler, 1, -1.0, cfg,
        training_state=_training_state(),
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    first_slot = next(iter(state["optimizer"]["state"].values()))
    first_slot["exp_avg"].fill_(float("nan"))
    torch.save(state, checkpoint)

    target = torch.nn.Linear(2, 1)
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    target_scheduler = torch.optim.lr_scheduler.LambdaLR(
        target_optimizer, lambda step: 1.0
    )
    before = deepcopy(target.state_dict())
    with pytest.raises(ValueError, match="NaN/Inf"):
        load_checkpoint(
            checkpoint, target, target_optimizer, target_scheduler, restore_rng=True
        )
    _assert_nested_equal(before, target.state_dict())


def test_canonical_run_entry_refuses_overwrite_wrong_resume_and_completed_reentry(
    tmp_path, monkeypatch
):
    cfg, model, optimizer, scheduler, fixture_ckpt_dir, source_root = _completion_components(tmp_path)
    import deep_anc.train.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "REPO_ROOT", source_root)
    monkeypatch.setattr(
        trainer_module, "bind_recorded_transfer_config", lambda data, repo_root: None
    )
    monkeypatch.setattr(
        trainer_module,
        "validate_canonical_pretrain_prerequisites",
        lambda cfg, repo_root: {},
    )
    run_dir = source_root / "runs" / "canonical"
    validate_canonical_run_entry(cfg, run_dir, None)
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("existing\n")
    with pytest.raises(FileExistsError, match="덮어쓸 수 없습니다"):
        validate_canonical_run_entry(cfg, run_dir, None)

    expected_last = run_dir / "ckpt" / "last.pt"
    with pytest.raises(ValueError, match="exact last.pt"):
        validate_canonical_run_entry(cfg, run_dir, run_dir / "ckpt" / "best.pt")
    validate_canonical_run_entry(cfg, run_dir, expected_last)

    save_checkpoint(
        run_dir / "ckpt" / "best.pt", model, optimizer, scheduler, 500, -1.0, cfg,
        training_state=_training_state(),
    )
    save_checkpoint(
        expected_last, model, optimizer, scheduler,
        cfg["schedule"]["total_steps"], -1.0, cfg,
        training_state=_training_state(),
    )
    for name in (
        "config_snapshot.yaml",
        "git_rev.txt",
        "pip_freeze.txt",
        "environment.json",
    ):
        shutil.copy2(fixture_ckpt_dir.parent / name, run_dir / name)
    write_completion_receipt(run_dir / "ckpt", repo_root=source_root)
    with pytest.raises(FileExistsError, match="재진입"):
        validate_canonical_run_entry(cfg, run_dir, expected_last)


def test_checkpoint_save_uses_unique_temp_and_cleans_failure(tmp_path, monkeypatch):
    cfg = cfg_snapshot(_config(tmp_path), trusted_band_hz=(150.0, 1_600.0))
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "ckpt" / "last.pt"
    save_checkpoint(
        checkpoint, model, optimizer, None, 0, 0.0, cfg,
        training_state=_training_state(),
    )
    assert checkpoint.is_file()
    assert not list(checkpoint.parent.glob(".last.pt.*.tmp"))

    original_save = torch.save

    def fail_save(*args, **kwargs):
        raise RuntimeError("injected save failure")

    monkeypatch.setattr(torch, "save", fail_save)
    failed = tmp_path / "failed" / "last.pt"
    with pytest.raises(RuntimeError, match="injected"):
        save_checkpoint(
            failed, model, optimizer, None, 0, 0.0, cfg,
            training_state=_training_state(),
        )
    assert not failed.exists()
    assert not list(failed.parent.glob(".last.pt.*.tmp"))
    monkeypatch.setattr(torch, "save", original_save)
