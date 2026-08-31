"""resume experiment 계약과 fail-closed 상태 복원 회귀 테스트."""

from __future__ import annotations

import json
import hashlib
import os
import random
import shutil
import subprocess
from types import SimpleNamespace
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from deep_anc.config import (
    A100_PRETRAIN_SMOKE_POLICY_VERSION,
    CANONICAL_FINETUNE_READINESS_POLICY,
    CANONICAL_LOSS_PILOT_STEPS,
    REPO_ROOT,
    load_train_config,
    load_yaml,
    validate_canonical_training_policy,
)
import deep_anc.config as config_module
from deep_anc.data.resumable_stream import indexed_rng, worker_global_item_indices
from deep_anc.dsp.timing import FrequencyBand
from deep_anc.train.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    validate_resume_checkpoint_preview,
)
from deep_anc.train.campaign_prerequisite import (
    CANONICAL_PATH as CANONICAL_PREREQUISITE_PATH,
    validate_canonical_pretrain_prerequisites,
)
import deep_anc.train.campaign_prerequisite as campaign_prerequisite_module
import deep_anc.train.campaign_evidence as campaign_evidence_module
from deep_anc.train.campaign_evidence import PILOT_SELECTION_RULE
from deep_anc.train.a100_pretrain_smoke import (
    A100_MIN_USABLE_MEMORY_BYTES,
    A100_PRETRAIN_SMOKE_ROLE,
    SMOKE_ROOT,
    build_a100_pretrain_smoke_artifacts,
    build_a100_pretrain_smoke_environment_receipt,
    build_a100_pretrain_smoke_resume_input,
    build_a100_pretrain_smoke_target,
    finalize_a100_pretrain_smoke_receipt,
)
import deep_anc.train.a100_pretrain_smoke as a100_smoke_module
from deep_anc.train.evaluation_contract import snapshot_regular_file
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
from deep_anc.model_input import (
    canonical_stage1_model_input_payload,
    resolve_stage1_model_input_contract,
)
from deep_anc.train.finetune_readiness import audit_init_checkpoint
from deep_anc.train.reproducibility import _publish_or_validate
from deep_anc.train.trainer import (
    cfg_snapshot,
    configure_canonical_determinism,
    validate_canonical_run_entry,
    validate_init_checkpoint_role,
    validate_measured_probe_init_chain,
    validate_resume_physics,
    validate_training_world_size,
)


_BOOTSTRAP_SHA = "a" * 64
_PREREQUISITE_SHA = "d" * 64


def _a100_environment_payload(*, total_memory_bytes: int) -> dict:
    return {
        "python": "fixture",
        "torch": "2.5.1+cu121",
        "torch_cuda": "12.1",
        "cuda_available": True,
        "device_count": 1,
        "devices": [
            {
                "index": 0,
                "name": "NVIDIA A100 80GB PCIe",
                "capability": [8, 0],
                "total_memory_bytes": total_memory_bytes,
            }
        ],
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": ":4096:8",
    }


def test_a100_environment_accepts_actual_nominal_80gb_usable_memory(tmp_path):
    """PyTorch가 79.4 GiB로 보고하는 실제 Elice A100 80GB PCIe를 허용한다."""

    path = tmp_path / "environment.json"
    path.write_text(
        json.dumps(
            _a100_environment_payload(total_memory_bytes=85_293_072_384),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        a100_smoke_module._validate_environment(snapshot_regular_file(path))["devices"][0]
        ["total_memory_bytes"]
        == 85_293_072_384
    )

    path.write_text(
        json.dumps(
            _a100_environment_payload(
                total_memory_bytes=A100_MIN_USABLE_MEMORY_BYTES - 1
            ),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="world1/CUDA/결정론"):
        a100_smoke_module._validate_environment(snapshot_regular_file(path))


def _load_bound_canonical(
    path,
    overrides: list[str] | None = None,
    *,
    campaign_anchor: bool = True,
    recorded_generation: bool = False,
) -> dict:
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
        if recorded_generation:
            data.update(
                recorded_generation=(
                    "data/manifests/recorded_generations/"
                    "stage1-coverage-v2/generation.json"
                ),
                recorded_generation_sha256="d" * 64,
            )

    values = list(overrides or [])
    values.append(f"data.bootstrap_receipt_sha256={_BOOTSTRAP_SHA}")
    if campaign_anchor:
        values.append(f"campaign_prerequisite_sha256={_PREREQUISITE_SHA}")
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


@pytest.mark.parametrize("role", ["loss_pilot", "measured_probe"])
def test_selection_derivative_resume_requires_exactly_one_cuda_rng(
    tmp_path, role
):
    """pilot/probe도 preview와 실제 resume 모두 world1 CUDA RNG 누락을 거부한다."""

    cfg = cfg_snapshot(_config(tmp_path), trusted_band_hz=(150.0, 1_600.0))
    cfg["experiment_role"] = role
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    checkpoint = tmp_path / role / "last.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        1,
        -1.0,
        cfg,
        training_state=_training_state(),
    )
    missing = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing["rng"]["cuda"] = None
    torch.save(missing, checkpoint)
    assert missing["rng"]["cuda"] is None
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="CUDA RNG state가 정확히 하나"):
        validate_resume_checkpoint_preview(
            checkpoint, missing, model, optimizer, scheduler
        )
    with pytest.raises(ValueError, match="CUDA RNG state가 정확히 하나"):
        load_checkpoint(
            checkpoint, model, optimizer, scheduler, restore_rng=True
        )
    _assert_nested_equal(before, model.state_dict())

    missing["rng"]["cuda"] = [torch.zeros(32, dtype=torch.uint8)]
    torch.save(missing, checkpoint)
    validate_resume_checkpoint_preview(
        checkpoint, missing, model, optimizer, scheduler
    )


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


@pytest.mark.parametrize(
    ("config_name", "overrides"),
    [
        ("train_pretrain_tiny.yaml", []),
        ("train_finetune.yaml", ["data.digital_primary_path_mode=measured"]),
    ],
)
def test_canonical_ref_only_input_is_resolved_and_bound_to_contract_and_checkpoint(
    config_name, overrides
):
    cfg = _load_bound_canonical(REPO_ROOT / "configs" / config_name, overrides)
    contract = resolve_stage1_model_input_contract(cfg["data"])
    assert contract is not None
    assert cfg["data_model_input_contract_config"] == (
        "configs/stage1_ref_only_input.yaml"
    )
    assert cfg["data"]["model_input_contract"] == (
        canonical_stage1_model_input_payload()
    )
    assert cfg["model_input_contract_sha256"] == contract.digest()

    checkpoint_cfg = cfg_snapshot(cfg, trusted_band_hz=(150.0, 1_600.0))
    assert checkpoint_cfg["model_input_contract_sha256"] == contract.digest()
    validate_canonical_training_policy(checkpoint_cfg)

    baseline = build_experiment_contract(cfg, repo_root=REPO_ROOT)
    changed = deepcopy(cfg)
    changed["data"]["model_input_contract"]["error_dropout_probability"] = 0.5
    assert build_experiment_contract(changed, repo_root=REPO_ROOT)["sha256"] != baseline[
        "sha256"
    ]
    with pytest.raises(ValueError, match="trust policy"):
        validate_canonical_training_policy(changed)


def test_canonical_ref_only_input_missing_or_conflicting_config_is_rejected():
    with pytest.raises(ValueError, match="trust policy"):
        _load_bound_canonical(
            REPO_ROOT / "configs/train_pretrain_tiny.yaml",
            ["data.model_input_contract=null"],
        )
    with pytest.raises(ValueError, match="Stage-1 canonical model input"):
        _load_bound_canonical(
            REPO_ROOT / "configs/train_pretrain_tiny.yaml",
            ["data.model_input_contract.error_dropout_probability=0.5"],
        )


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


def test_canonical_roles_reject_missing_regular_rir_before_contract_stamp():
    original = config_module._resolve_path
    missing = REPO_ROOT / "data/rir_bank/__missing_canonical_fixture__.npz"

    def redirect(value):
        if str(value) == "data/rir_bank/duct_rirs_v1.npz":
            return missing
        return original(value)

    with patch.object(config_module, "_resolve_path", side_effect=redirect):
        with pytest.raises(ValueError, match="RIR bank regular file"):
            _load_bound_canonical(REPO_ROOT / "configs/train_pretrain_tiny.yaml")


def test_approved_frame_metric_only_alpha_pilots_have_distinct_20k_contracts():
    base = _load_bound_canonical(REPO_ROOT / "configs/train_pretrain_tiny.yaml")
    run_dirs = set()
    for alpha in (0.7, 1.0):
        pilot = _load_bound_canonical(
            REPO_ROOT / "configs/train_pretrain_tiny.yaml",
            [
                "experiment_role=loss_pilot",
                "init_eligible=false",
                f"loss.nmse_cvar_alpha={alpha}",
                "loss.lambda_frame=0.0",
                "run_until_step=20000",
            ],
        )
        assert pilot["loss"]["nmse_cvar_alpha"] == alpha
        assert pilot["loss"]["lambda_frame"] == 0.0
        assert pilot["experiment_role"] == "loss_pilot"
        assert pilot["init_eligible"] is False
        assert pilot["run_until_step"] == 20_000
        run_dirs.add(pilot["ckpt_dir"])
    assert len(run_dirs) == 2
    assert base["ckpt_dir"] not in run_dirs


def test_measured_primary_five_k_probe_is_an_explicit_distinct_contract():
    base = _load_bound_canonical(REPO_ROOT / "configs/train_pretrain_tiny.yaml")
    probe = _load_bound_canonical(
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
    assert probe["require_measured_primary_path"] is True
    assert probe["require_init_checkpoint"] is True
    assert probe["require_init_completion_receipt"] is False
    assert probe["require_recorded_manifest"] is True
    assert probe["required_init_experiment_role"] == "loss_pilot"
    assert probe["require_init_eligible"] is False
    assert probe["recorded_manifest"] == "data/manifests/recorded_regrouped.jsonl"
    assert probe["recorded_ratio"] == 0.7
    assert probe["readiness"] == CANONICAL_FINETUNE_READINESS_POLICY
    assert len(probe["readiness"]) == 21
    assert probe["readiness"]["recorded_subband_coverage_report_dir"] == (
        "results/data_audit/recorded_subband_coverage"
    )
    assert probe["readiness"]["recorded_source_pool_csv"] == [
        "data/source_pool_v2/sources.csv",
        "data/source_pool/sources.csv",
    ]
    assert probe["experiment_contract"]["artifacts"]["recorded_manifest"][
        "exists"
    ]
    assert probe["experiment_contract"]["artifacts"]["init_checkpoint"][
        "path"
    ].endswith("selected-20k/ckpt/best.pt")
    assert probe["ckpt_dir"] != base["ckpt_dir"]


@pytest.mark.parametrize(
    ("config", "overrides", "expected_recorded_manifest"),
    [
        ("configs/train_pretrain_tiny.yaml", [], None),
        (
            "configs/train_pretrain_tiny.yaml",
            [
                "experiment_role=loss_pilot",
                "init_eligible=false",
                "run_until_step=20000",
            ],
            None,
        ),
        (
            "configs/train_pretrain_tiny.yaml",
            [
                "experiment_role=measured_probe",
                "init_eligible=false",
                "data.digital_primary_path_mode=measured",
                "run_until_step=5000",
                "init_ckpt=runs/selected-20k/ckpt/best.pt",
            ],
            (
                "data/manifests/recorded_generations/"
                "stage1-coverage-v2/recorded.jsonl"
            ),
        ),
        (
            "configs/train_finetune.yaml",
            ["data.digital_primary_path_mode=measured"],
            (
                "data/manifests/recorded_generations/"
                "stage1-coverage-v2/recorded.jsonl"
            ),
        ),
    ],
)
def test_schema2_bootstrap_atomically_resolves_all_official_role_manifests(
    config, overrides, expected_recorded_manifest
):
    """bootstrap SHA 하나로 pilot/probe/canonical/fine-tune이 같은 99세대를 본다."""

    cfg = _load_bound_canonical(
        REPO_ROOT / config,
        overrides,
        recorded_generation=True,
    )
    assert cfg["data"]["recorded_generation"] == (
        "data/manifests/recorded_generations/"
        "stage1-coverage-v2/generation.json"
    )
    assert cfg["data"]["recorded_generation_sha256"] == "d" * 64
    assert cfg.get("recorded_manifest") == expected_recorded_manifest


@pytest.mark.parametrize(
    "override",
    [
        "recorded_ratio=0.3",
        "recorded_manifest=data/manifests/recorded_train.jsonl",
        "require_recorded_manifest=false",
        "require_measured_primary_path=false",
        "require_init_checkpoint=false",
        "require_init_completion_receipt=true",
        "required_init_experiment_role=canonical_pretrain",
        "require_init_eligible=true",
        "readiness.max_init_lead_mismatch_samples=0",
        "readiness.min_path_consistency=0.9",
        (
            "readiness.recorded_subband_coverage_report_dir="
            "results/data_audit/weakened"
        ),
        "readiness.recorded_source_pool_csv=data/source_pool_v2/sources.csv",
    ],
)
def test_measured_probe_cannot_weaken_recorded_70_30_or_selected_init_policy(
    override,
):
    with pytest.raises(ValueError, match="measured_probe trust policy"):
        _load_bound_canonical(
            REPO_ROOT / "configs/train_pretrain_tiny.yaml",
            [
                "experiment_role=measured_probe",
                "init_eligible=false",
                "data.digital_primary_path_mode=measured",
                "run_until_step=5000",
                "init_ckpt=runs/selected-20k/ckpt/best.pt",
                override,
            ],
        )


def _measured_probe_init_fixture(tmp_path: Path, *, last_step: int = 20_000):
    pilot = _load_bound_canonical(
        REPO_ROOT / "configs/train_pretrain_tiny.yaml",
        [
            "experiment_role=loss_pilot",
            "init_eligible=false",
            "run_until_step=20000",
        ],
    )
    checkpoint_dir = tmp_path / "pilot" / "ckpt"
    checkpoint_dir.mkdir(parents=True)
    best = checkpoint_dir / "best.pt"
    last = checkpoint_dir / "last.pt"
    best_state = {
        "cfg": pilot,
        "model": {"weight": torch.ones(1)},
        "step": 15_000,
        "best_metric": -1.0,
    }
    torch.save(best_state, best)
    torch.save(
        {
            "cfg": pilot,
            "model": {"weight": torch.ones(1)},
            "step": last_step,
            "best_metric": -1.0,
        },
        last,
    )
    probe = _load_bound_canonical(
        REPO_ROOT / "configs/train_pretrain_tiny.yaml",
        [
            "experiment_role=measured_probe",
            "init_eligible=false",
            "data.digital_primary_path_mode=measured",
            "run_until_step=5000",
            f"init_ckpt={best}",
        ],
    )
    return best_state, probe, best


def test_measured_probe_requires_same_loss_completed_20k_pilot_before_gpu(tmp_path):
    best_state, probe, best = _measured_probe_init_fixture(tmp_path)
    validate_measured_probe_init_chain(best_state, probe, best)

    incomplete = torch.load(best.with_name("last.pt"), weights_only=False)
    incomplete["step"] = 19_999
    torch.save(incomplete, best.with_name("last.pt"))
    with pytest.raises(ValueError, match="정확히 20k 완료"):
        validate_measured_probe_init_chain(best_state, probe, best)


def test_readiness_accepts_exact_20k_pilot_without_completion_receipt(tmp_path):
    best_state, probe, best = _measured_probe_init_fixture(tmp_path)
    pilot = best_state["cfg"]
    trusted_band = FrequencyBand.parse(pilot["trusted_band_hz"])

    result = audit_init_checkpoint(
        best,
        expected_model_cfg=probe["model"],
        expected_lead=int(probe["digital_reference_lead_samples"]),
        expected_optimize_band=trusted_band,
        max_lead_mismatch_samples=16,
        require_completed=True,
        require_completion_receipt=False,
        expected_completion_step=CANONICAL_LOSS_PILOT_STEPS,
        required_experiment_role="loss_pilot",
        require_init_eligible=False,
        expected_loss_selection_sha256=probe["loss_selection_sha256"],
    )

    assert not best.with_name("completion.json").exists()
    assert result["completion_step"] == CANONICAL_LOSS_PILOT_STEPS
    assert result["completion_target_step"] == CANONICAL_LOSS_PILOT_STEPS
    assert result["completion_receipt"] is None


def test_readiness_rejects_incomplete_20k_pilot_without_receipt_requirement(tmp_path):
    best_state, probe, best = _measured_probe_init_fixture(
        tmp_path, last_step=CANONICAL_LOSS_PILOT_STEPS - 1
    )
    pilot = best_state["cfg"]

    with pytest.raises(ValueError, match="last step=19999, target=20000"):
        audit_init_checkpoint(
            best,
            expected_model_cfg=probe["model"],
            expected_lead=int(probe["digital_reference_lead_samples"]),
            expected_optimize_band=FrequencyBand.parse(pilot["trusted_band_hz"]),
            max_lead_mismatch_samples=16,
            require_completed=True,
            require_completion_receipt=False,
            expected_completion_step=CANONICAL_LOSS_PILOT_STEPS,
            required_experiment_role="loss_pilot",
            require_init_eligible=False,
            expected_loss_selection_sha256=probe["loss_selection_sha256"],
        )


def test_canonical_init_keeps_full_schedule_and_completion_receipt_requirement(
    tmp_path,
):
    canonical = _load_bound_canonical(
        REPO_ROOT / "configs/train_pretrain_tiny.yaml"
    )
    checkpoint_dir = tmp_path / "canonical" / "ckpt"
    checkpoint_dir.mkdir(parents=True)
    best = checkpoint_dir / "best.pt"
    last = checkpoint_dir / "last.pt"
    base_state = {
        "cfg": canonical,
        "model": {"weight": torch.ones(1)},
        "best_metric": -1.0,
    }
    torch.save({**base_state, "step": 90_000}, best)
    torch.save({**base_state, "step": 99_999}, last)
    kwargs = {
        "expected_model_cfg": canonical["model"],
        "expected_lead": int(canonical["digital_reference_lead_samples"]),
        "expected_optimize_band": FrequencyBand.parse(
            canonical["trusted_band_hz"]
        ),
        "require_completed": True,
        "require_completion_receipt": True,
        "expected_completion_step": None,
        "required_experiment_role": "canonical_pretrain",
        "require_init_eligible": True,
        "expected_loss_selection_sha256": canonical["loss_selection_sha256"],
    }

    with pytest.raises(ValueError, match="last step=99999, target=100000"):
        audit_init_checkpoint(best, **kwargs)

    torch.save({**base_state, "step": 100_000}, last)
    with pytest.raises(ValueError, match="completion.json"):
        audit_init_checkpoint(best, **kwargs)


def test_measured_probe_rejects_different_alpha_before_gpu(tmp_path):
    best_state, _probe, best = _measured_probe_init_fixture(tmp_path)
    different = _load_bound_canonical(
        REPO_ROOT / "configs/train_pretrain_tiny.yaml",
        [
            "experiment_role=measured_probe",
            "init_eligible=false",
            "data.digital_primary_path_mode=measured",
            "run_until_step=5000",
            f"init_ckpt={best}",
            "loss.nmse_cvar_alpha=1.0",
        ],
    )
    with pytest.raises(ValueError, match="loss selection"):
        validate_measured_probe_init_chain(best_state, different, best)


@pytest.mark.parametrize(
    ("role", "primary_mode", "overrides"),
    [
        (
            "loss_pilot",
            "secondary_surrogate",
            [
                "experiment_role=loss_pilot",
                "init_eligible=false",
                "run_until_step=20000",
            ],
        ),
        (
            "measured_probe",
            "measured",
            [
                "experiment_role=measured_probe",
                "init_eligible=false",
                "data.digital_primary_path_mode=measured",
                "run_until_step=5000",
                "init_ckpt=runs/selected-20k/ckpt/best.pt",
            ],
        ),
    ],
)
def test_campaign_replays_full_derivative_policy_from_checkpoint(
    role, primary_mode, overrides
):
    canonical = _load_bound_canonical(REPO_ROOT / "configs/train_pretrain_tiny.yaml")
    derivative = _load_bound_canonical(
        REPO_ROOT / "configs/train_pretrain_tiny.yaml",
        overrides,
    )
    forged = deepcopy(derivative)
    forged["optimizer"]["lr"] = 0.01
    forged = stamp_experiment_contract(forged, repo_root=REPO_ROOT)
    with pytest.raises(ValueError, match=rf"{role} trust policy"):
        campaign_evidence_module._validate_derivative_cfg(
            forged,
            canonical_cfg=canonical,
            canonical_contract=canonical["experiment_contract"],
            root=REPO_ROOT,
            label=f"forged {role}",
            role=role,
            primary_mode=primary_mode,
            embedded=True,
            expected_identity=(0.7, 0.0, 0.00075),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            [
                "experiment_role=loss_pilot",
                "init_eligible=false",
                "run_until_step=19999",
            ],
            "run_until_step",
        ),
        (
            [
                "experiment_role=loss_pilot",
                "init_eligible=false",
                "run_until_step=20000",
                "init_ckpt=runs/other/ckpt/best.pt",
            ],
            "from-scratch",
        ),
        (
            [
                "experiment_role=measured_probe",
                "init_eligible=false",
                "data.digital_primary_path_mode=measured",
                "run_until_step=4999",
                "init_ckpt=runs/selected-20k/ckpt/best.pt",
            ],
            "run_until_step",
        ),
        (
            [
                "experiment_role=measured_probe",
                "init_eligible=false",
                "data.digital_primary_path_mode=measured",
                "run_until_step=5000",
            ],
            "selected 20k",
        ),
    ],
)
def test_strict_selection_derivative_admission_rejects_wrong_budget_or_init(
    overrides, message
):
    with pytest.raises(ValueError, match=message):
        _load_bound_canonical(REPO_ROOT / "configs/train_pretrain_tiny.yaml", overrides)


def test_strict_selection_derivative_requires_bootstrap_anchor_before_stamp():
    with pytest.raises(ValueError, match="bootstrap_receipt_sha256"):
        load_train_config(
            REPO_ROOT / "configs/train_pretrain_tiny.yaml",
            [
                "experiment_role=loss_pilot",
                "init_eligible=false",
                "run_until_step=20000",
            ],
        )


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


def test_secondary_prerequisite_provisional_cfg_resolves_with_real_loader():
    """issuer는 prerequisite 발행 전 exists=false artifact로 prospective SHA를 만든다."""

    neutral = "a" * 64
    path = (
        "results/training_prerequisites/second_seed/"
        f"{neutral}/seed_20260903.json"
    )
    cfg = _load_bound_canonical(
        REPO_ROOT / "configs/train_pretrain_tiny.yaml",
        [
            "seed=20260903",
            f"second_seed_prerequisite={path}",
            f"second_seed_prerequisite_sha256=\"{'0' * 64}\"",
        ],
    )

    assert cfg["seed"] == 20260903
    assert cfg["second_seed_prerequisite"] == path
    assert cfg["second_seed_prerequisite_sha256"] == "0" * 64
    artifact = cfg["experiment_contract"]["artifacts"][
        "second_seed_prerequisite"
    ]
    assert artifact["exists"] is False


def test_a100_pretrain_smoke_has_same_target_but_never_uses_canonical_runs():
    canonical = _load_bound_canonical(REPO_ROOT / "configs/train_pretrain_tiny.yaml")
    common = [
        f"experiment_role={A100_PRETRAIN_SMOKE_ROLE}",
        "init_eligible=false",
        "contract_run_dir=false",
        "campaign_prerequisite=null",
        "campaign_prerequisite_sha256=null",
        "a100_smoke_run_label=uninterrupted",
        "run_until_step=500",
    ]
    smoke = _load_bound_canonical(
        REPO_ROOT / "configs/train_pretrain_tiny.yaml",
        common,
        campaign_anchor=False,
    )
    resumed = _load_bound_canonical(
        REPO_ROOT / "configs/train_pretrain_tiny.yaml",
        common[:-2]
        + ["a100_smoke_run_label=resumed", "run_until_step=300"],
        campaign_anchor=False,
    )
    assert smoke["init_eligible"] is False
    assert smoke["smoke_target_sha256"] == resumed["smoke_target_sha256"]
    assert smoke["experiment_contract_sha256"] == resumed[
        "experiment_contract_sha256"
    ]
    assert smoke["ckpt_dir"].startswith(
        "results/training_prerequisites/a100_pretrain_smoke/"
    )
    assert not smoke["ckpt_dir"].startswith("runs/")
    assert build_a100_pretrain_smoke_target(
        canonical, repo_root=REPO_ROOT
    )["sha256"] == smoke["smoke_target_sha256"]


@pytest.mark.parametrize("alpha", [0.7, 0.85, 1.0])
def test_a100_smoke_target_binds_the_selected_loss_grid_alpha(alpha):
    common = [
        f"experiment_role={A100_PRETRAIN_SMOKE_ROLE}",
        "init_eligible=false",
        "contract_run_dir=false",
        "campaign_prerequisite=null",
        "campaign_prerequisite_sha256=null",
        "a100_smoke_run_label=uninterrupted",
        "run_until_step=500",
        f"loss.nmse_cvar_alpha={alpha}",
    ]
    smoke = _load_bound_canonical(
        REPO_ROOT / "configs/train_pretrain_tiny.yaml",
        common,
        campaign_anchor=False,
    )
    canonical = _load_bound_canonical(
        REPO_ROOT / "configs/train_pretrain_tiny.yaml",
        [f"loss.nmse_cvar_alpha={alpha}"],
    )
    assert smoke["loss"]["nmse_cvar_alpha"] == alpha
    assert build_a100_pretrain_smoke_target(
        canonical, repo_root=REPO_ROOT
    )["sha256"] == smoke["smoke_target_sha256"]


def test_a100_operational_fields_are_not_ignored_by_canonical_contract(tmp_path):
    """label/run-root exclusion은 smoke role에만 한정돼야 한다."""

    canonical = _config(tmp_path)
    canonical["experiment_role"] = "canonical_pretrain"
    baseline = build_experiment_contract(canonical, repo_root=REPO_ROOT)
    injected = deepcopy(canonical)
    injected["a100_smoke_run_label"] = "resumed"
    injected["resolved_smoke_run_dir"] = {"smoke_target_sha256": "a" * 64}
    assert build_experiment_contract(injected, repo_root=REPO_ROOT)["sha256"] != baseline[
        "sha256"
    ]


def test_a100_pretrain_smoke_rejects_semantic_weakening_and_role_output_misuse():
    base = [
        f"experiment_role={A100_PRETRAIN_SMOKE_ROLE}",
        "init_eligible=false",
        "contract_run_dir=false",
        "campaign_prerequisite=null",
        "campaign_prerequisite_sha256=null",
        "a100_smoke_run_label=uninterrupted",
        "run_until_step=500",
    ]
    with pytest.raises(ValueError, match="B96|batch_size"):
        _load_bound_canonical(
            REPO_ROOT / "configs/train_pretrain_tiny.yaml",
            base + ["batch_size=95"],
            campaign_anchor=False,
        )
    with pytest.raises(ValueError, match="a100_smoke_run_label"):
        _load_bound_canonical(
            REPO_ROOT / "configs/train_pretrain_tiny.yaml",
            base[:-2] + ["a100_smoke_run_label=other", "run_until_step=500"],
            campaign_anchor=False,
        )
    with pytest.raises(ValueError, match="run_until_step"):
        _load_bound_canonical(
            REPO_ROOT / "configs/train_pretrain_tiny.yaml",
            base[:-1] + ["run_until_step=501"],
            campaign_anchor=False,
        )


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
            "family_plant_domain_component_session_balanced",
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
        "require_init_completion_receipt=false",
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
        "loss.lambda_dnh=.nan",
        "loss.lambda_dnh=.inf",
        "loss.lambda_dnh=true",
        "data.recorded_augment.mix_probability=0.1",
        "data.allow_missing_source_manifests=true",
        "data.noise_manifest_dir=data/manifests",
        "data.rir_bank=data/rir_bank/other.npz",
        "data.level_dbfs=[-20,-20]",
        "data.snr_mic_noise_db=[0,0]",
        "data.dc_hum_prob=0.0",
        "data.nonlinear.hardclip_prob=0.1",
        "data.plant_perturbation.delay_jitter_range=[0,1]",
        "data.closed_loop.feedback_delay_samples=[1,2]",
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
        "data.allow_missing_source_manifests=true",
        "data.noise_manifest_dir=data/manifests",
        "data.rir_bank=data/rir_bank/other.npz",
        "data.level_dbfs=[-20,-20]",
        "data.snr_mic_noise_db=[0,0]",
        "data.dc_hum_prob=0.0",
        "data.nonlinear.hardclip_prob=0.1",
        "data.plant_perturbation.delay_jitter_range=[0,1]",
        "data.closed_loop.feedback_delay_samples=[1,2]",
        "recorded_manifest=data/manifests/recorded_regrouped.jsonl",
        "recorded_ratio=0.7",
        "require_recorded_manifest=true",
    ],
)
def test_canonical_pretrain_cannot_be_promoted_from_a_changed_contract(override):
    with pytest.raises(ValueError, match="canonical_pretrain trust policy"):
        load_train_config(
            REPO_ROOT / "configs/train_pretrain_tiny.yaml", [override]
        )


@pytest.mark.parametrize(
    ("role", "primary_mode", "extra", "init"),
    [
        ("loss_pilot", "secondary_surrogate", "run_until_step=20000", None),
        (
            "measured_probe",
            "measured",
            "run_until_step=5000",
            "init_ckpt=runs/selected-20k/ckpt/best.pt",
        ),
    ],
)
def test_canonical_selection_derivatives_cannot_replace_missing_source_families(
    role, primary_mode, extra, init
):
    """pilot/probe도 누락 public source를 synthetic으로 대체할 수 없다."""

    with pytest.raises(ValueError, match="allow_missing_source_manifests"):
        _load_bound_canonical(
            REPO_ROOT / "configs/train_pretrain_tiny.yaml",
            [
                f"experiment_role={role}",
                "init_eligible=false",
                f"data.digital_primary_path_mode={primary_mode}",
                extra,
                *([init] if init else []),
                "data.allow_missing_source_manifests=true",
            ],
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
    mismatched_lambda = _load_bound_canonical(
        REPO_ROOT / "configs/train_pretrain_tiny.yaml",
        ["loss.lambda_dnh=0.000375"],
    )
    with pytest.raises(ValueError, match="loss selection"):
        validate_init_checkpoint_role({"cfg": mismatched_lambda}, finetune)


def _sha_ref(source_root: Path, path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(source_root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_smoke_phase_telemetry(
    path: Path,
    *,
    cfg: dict,
    start_step: int,
    completed_step: int,
    resume_checkpoint_sha256: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "smoke_target_sha256": cfg["smoke_target_sha256"],
                "experiment_role": A100_PRETRAIN_SMOKE_ROLE,
                "init_eligible": False,
                "experiment_contract_sha256": cfg["experiment_contract_sha256"],
                "start_step": start_step,
                "completed_step": completed_step,
                "run_until_step": completed_step,
                "schedule_total_steps": cfg["schedule"]["total_steps"],
                "elapsed_seconds": 1.0,
                "steps_completed": completed_step - start_step,
                "steps_per_second": float(completed_step - start_step),
                "device": "cuda:0",
                "cuda_available": True,
                "device_count": 1,
                "amp": "bf16",
                "max_memory_allocated_bytes": 1,
                "max_memory_reserved_bytes": 1,
                "resume_checkpoint_sha256": resume_checkpoint_sha256,
                "final_train_metrics": {"loss": -1.0},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _a100_smoke_ledger_refs(source_root: Path, canonical_cfg: dict) -> dict:
    """schema-v4 ledger test용 실제 checkpoint/resume receipt를 만든다."""

    target = build_a100_pretrain_smoke_target(
        canonical_cfg, repo_root=source_root
    )["sha256"]

    def smoke_cfg(label: str, *, run_until: int, resume: str | None = None) -> dict:
        cfg = deepcopy(canonical_cfg)
        cfg.update(
            experiment_role=A100_PRETRAIN_SMOKE_ROLE,
            canonical_trust_policy=A100_PRETRAIN_SMOKE_POLICY_VERSION,
            init_eligible=False,
            contract_run_dir=False,
            campaign_prerequisite=None,
            campaign_prerequisite_sha256=None,
            init_ckpt=None,
            a100_smoke_run_label=label,
            smoke_target_sha256=target,
            run_until_step=run_until,
        )
        run_dir = source_root / SMOKE_ROOT / target / label
        cfg["ckpt_dir"] = str(run_dir.relative_to(source_root))
        cfg["resolved_smoke_run_dir"] = {
            "schema": "results/training_prerequisites/a100_pretrain_smoke/<target>/<label>",
            "smoke_target_sha256": target,
        }
        if resume is not None:
            cfg["resume"] = resume
        return stamp_experiment_contract(cfg, repo_root=source_root)

    full_cfg = smoke_cfg("uninterrupted", run_until=500)
    resumed_dir = source_root / SMOKE_ROOT / target / "resumed"
    resumed_last = resumed_dir / "ckpt" / "last.pt"
    stop_path = resumed_dir / "ckpt" / "stop.pt"
    stopped_cfg = smoke_cfg("resumed", run_until=300)
    resumed_cfg = smoke_cfg("resumed", run_until=500, resume=str(stop_path))
    assert full_cfg["experiment_contract_sha256"] == resumed_cfg[
        "experiment_contract_sha256"
    ]

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    full_path = source_root / SMOKE_ROOT / target / "uninterrupted" / "ckpt" / "last.pt"
    save_checkpoint(
        full_path,
        model,
        optimizer,
        scheduler,
        500,
        -1.0,
        full_cfg,
        training_state=_training_state(),
    )
    state = torch.load(full_path, map_location="cpu", weights_only=False)
    # 실제 A100 smoke는 world1 CUDA RNG 한 벌을 저장해야 한다. 이 CPU fixture도
    # receipt validator가 그 schema를 직접 검사하도록 같은 구조를 materialize한다.
    state["rng"]["cuda"] = [torch.zeros(32, dtype=torch.uint8)]
    torch.save(state, full_path)
    stopped_state = deepcopy(state)
    stopped_state["cfg"] = stopped_cfg
    stopped_state["step"] = 300
    stopped_state["data_stream"] = {"schema_version": 1, "global_batch_index": 300}
    # default smoke stop=300은 eval_every=500 전이므로 selection이 아직 없다.
    stopped_state["best_metric"] = float("inf")
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stopped_state, stop_path)
    target_root = resumed_dir.parent
    resume_input_path = target_root / "resume_input.json"
    resume_input_path.write_text(
        json.dumps(
            build_a100_pretrain_smoke_resume_input(
                repo_root=source_root,
                smoke_target_sha256=target,
                resume_checkpoint=stop_path,
                stop_checkpoint=stop_path,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    resumed_state = deepcopy(state)
    resumed_state["cfg"] = resumed_cfg
    resumed_state["step"] = 500
    resumed_state["data_stream"] = {"schema_version": 1, "global_batch_index": 500}
    resumed_temporary = resumed_last.with_name(".last-resumed.tmp")
    torch.save(resumed_state, resumed_temporary)
    os.replace(resumed_temporary, resumed_last)

    full_dir = full_path.parent.parent
    _write_smoke_phase_telemetry(
        full_dir / "telemetry" / "000000_000500.json",
        cfg=full_cfg,
        start_step=0,
        completed_step=500,
    )
    _write_smoke_phase_telemetry(
        resumed_dir / "telemetry" / "000000_000300.json",
        cfg=stopped_cfg,
        start_step=0,
        completed_step=300,
    )
    _write_smoke_phase_telemetry(
        resumed_dir / "telemetry" / "000300_000500.json",
        cfg=resumed_cfg,
        start_step=300,
        completed_step=500,
        resume_checkpoint_sha256=_sha_ref(source_root, stop_path)["sha256"],
    )
    environment_payload = {
        "python": "fixture",
        "torch": "2.5.1+cu121",
        "torch_cuda": "12.1",
        "cuda_available": True,
        "device_count": 1,
        "devices": [
            {
                "index": 0,
                "name": "NVIDIA A100-SXM4-40GB",
                "capability": [8, 0],
                "total_memory_bytes": 80 * 1024**3,
            }
        ],
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": ":4096:8",
    }
    full_environment = full_dir / "environment.json"
    environment = resumed_dir / "environment.json"
    for path in (full_environment, environment):
        path.write_text(
            json.dumps(environment_payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    environment_receipt_path = target_root / "environment_receipt.json"
    environment_receipt_path.write_text(
        json.dumps(
            build_a100_pretrain_smoke_environment_receipt(
                repo_root=source_root,
                smoke_target_sha256=target,
                uninterrupted_environment=full_environment,
                resumed_environment=environment,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    combined, receipt = build_a100_pretrain_smoke_artifacts(
        repo_root=source_root,
        smoke_target_sha256=target,
        uninterrupted_checkpoint=full_path,
        stop_checkpoint=stop_path,
        resumed_checkpoint=resumed_last,
        uninterrupted_telemetry=full_dir / "telemetry" / "000000_000500.json",
        stopped_telemetry=resumed_dir / "telemetry" / "000000_000300.json",
        resumed_telemetry=resumed_dir / "telemetry" / "000300_000500.json",
        environment_receipt=environment_receipt_path,
        resume_input=resume_input_path,
    )
    combined_path = target_root / "telemetry.json"
    combined_path.write_text(
        json.dumps(combined, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = finalize_a100_pretrain_smoke_receipt(
        receipt, telemetry_reference=_sha_ref(source_root, combined_path)
    )
    receipt_path = target_root / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "evidence": _sha_ref(source_root, receipt_path),
        "environment_receipt": _sha_ref(source_root, environment_receipt_path),
        "telemetry": _sha_ref(source_root, combined_path),
    }


def _prerequisite_components(tmp_path):
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
                    {
                        "index": 0,
                        "name": "NVIDIA A100-SXM4-40GB",
                        "capability": [8, 0],
                        "total_memory_bytes": 80 * 1024**3,
                    }
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
    # v7 ledger는 각 alpha의 G0/calibration/20k pilot/5k measured probe를
    # identity=(alpha,frame,lambda_dnh) chain으로 보관한다. 이 fixture의 P/S는 unit-test용 bytes라 raw DSP 재계산은
    # 아래 helper에서 stub하고, A100 smoke chain만 독립 검증한다.
    candidates = [
        {
            "g0": {"receipt": evidence_ref},
            "gradient_calibration": {"receipt": evidence_ref},
            "pilot": {"fixture_identity": [0.7, 0.0, 0.00075]},
            "measured_probe": {"fixture_identity": [0.7, 0.0, 0.00075]},
        },
        {
            "g0": {"receipt": evidence_ref},
            "gradient_calibration": {"receipt": evidence_ref},
            "pilot": {"fixture_identity": [1.0, 0.0, 0.000375]},
            "measured_probe": {"fixture_identity": [1.0, 0.0, 0.000375]},
        },
    ]
    ledger = {
        "schema_version": 7,
        "source": {
            "git_commit": source["git_commit"],
            "source_tree_sha256": source["source_tree_sha256"],
            "bootstrap_receipt_sha256": cfg["data"]["bootstrap_receipt_sha256"],
            "archive_cache_manifest_sha256": cfg["data"].get(
                "archive_cache_manifest_sha256"
            ),
            "primary_path_sha256": artifacts["primary_path"]["sha256"],
            "secondary_path_sha256": artifacts["secondary_path"]["sha256"],
        },
        "gradient_budget": {"receipt": evidence_ref},
        "loss_pilot_selection": {
            "selection_rule": PILOT_SELECTION_RULE,
            "candidates": candidates,
        },
        "a100_smoke_resume": _a100_smoke_ledger_refs(source_root, cfg),
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


def _validate_fixture_campaign(
    cfg: dict,
    source_root: Path,
    *,
    pilot_scores: tuple[float, float] = (-4.0, -2.0),
    probe_scores: tuple[float, float] = (-4.0, -2.0),
    g0_batch_shas: tuple[str, str] = ("c" * 64, "c" * 64),
    g0_batch_paths: tuple[str, str] = ("shared-batch.pt", "shared-batch.pt"),
):
    """A100 smoke fixture에서는 새 raw-DSP validators만 분리한다.

    이 파일의 fake ``*.npz``는 transfer/source/a100 receipt test를 빠르게 하기 위한
    text fixture다. schema-v7 raw G0/calibration/pilot/probe 계산은
    ``tests/test_campaign_evidence.py``의 real raw artifact fixture에서 다룬다.
    """

    pilots = iter(
        (
            ((0.7, 0.0, 0.00075), float(pilot_scores[0]), "a" * 64),
            ((1.0, 0.0, 0.000375), float(pilot_scores[1]), "b" * 64),
        )
    )
    probes = iter(
        (
            ((0.7, 0.0, 0.00075), float(probe_scores[0]), "a" * 64),
            ((1.0, 0.0, 0.000375), float(probe_scores[1]), "b" * 64),
        )
    )

    def pilot(*_args, **_kwargs):
        identity, score, sha = next(pilots)
        assert tuple(_kwargs["expected_identity"]) == identity
        return {
            "identity": identity,
            "score_db": score,
            "best_snapshot": SimpleNamespace(sha256=sha),
        }

    def probe(*_args, **kwargs):
        identity, score, init_sha = next(probes)
        assert tuple(kwargs["expected_identity"]) == identity
        assert kwargs["expected_init_checkpoint_sha256"] == init_sha
        return {"identity": identity, "score_db": score}

    g0_rows = iter(
        (
            ((0.7, 0.0, 0.00075), g0_batch_paths[0], g0_batch_shas[0]),
            ((1.0, 0.0, 0.000375), g0_batch_paths[1], g0_batch_shas[1]),
        )
    )

    def g0(*_args, **_kwargs):
        identity, batch_path, batch_sha = next(g0_rows)
        return {
            "identity": identity,
            "batch": SimpleNamespace(
                path=source_root / "results/evidence" / batch_path,
                sha256=batch_sha,
            ),
        }

    def gradient(*_args, **kwargs):
        assert kwargs["expected_batch_path"] == (
            source_root / "results/evidence" / g0_batch_paths[0]
        )
        assert kwargs["expected_batch_sha256"] == g0_batch_shas[0]
        return {}

    with patch.object(
        campaign_prerequisite_module,
        "validate_canonical_evidence_target",
        return_value=cfg["experiment_contract"],
    ), patch.object(
        campaign_prerequisite_module,
        "validate_g0_receipt",
        side_effect=g0,
    ), patch.object(
        campaign_prerequisite_module,
        "validate_prepilot_gradient_receipt",
        return_value={"gradient_share": 0.3},
    ), patch.object(
        campaign_prerequisite_module,
        "validate_loss_pilot_candidate",
        side_effect=pilot,
    ), patch.object(
        campaign_prerequisite_module,
        "validate_gradient_budget_receipt",
        side_effect=gradient,
    ), patch.object(
        campaign_prerequisite_module,
        "validate_measured_probe",
        side_effect=probe,
    ):
        return validate_canonical_pretrain_prerequisites(cfg, repo_root=source_root)


def test_canonical_pretrain_prerequisites_bind_every_campaign_gate(tmp_path):
    cfg, source_root, ledger_path = _prerequisite_components(tmp_path)
    ledger = _validate_fixture_campaign(cfg, source_root)
    assert "g0" not in ledger
    assert "measured_probe" not in ledger
    assert all(
        set(candidate) == {"g0", "gradient_calibration", "pilot", "measured_probe"}
        for candidate in ledger["loss_pilot_selection"]["candidates"]
    )
    assert set(ledger["a100_smoke_resume"]) == {
        "evidence",
        "environment_receipt",
        "telemetry",
    }

    # old scalar schema must not remain an accidental manual-evidence escape hatch.
    legacy = json.loads(ledger_path.read_text())
    legacy["schema_version"] = 6
    ledger_path.write_text(json.dumps(legacy, sort_keys=True) + "\n", encoding="utf-8")
    refreshed = _restamp_campaign_anchor(cfg, source_root, ledger_path)
    with pytest.raises(ValueError, match="schema_version"):
        _validate_fixture_campaign(refreshed, source_root)


def test_campaign_requires_the_same_fixed_g0_batch_for_every_alpha(tmp_path):
    cfg, source_root, _ = _prerequisite_components(tmp_path)
    with pytest.raises(ValueError, match="authoritative fixed batch SHA"):
        _validate_fixture_campaign(
            cfg,
            source_root,
            g0_batch_shas=("c" * 64, "d" * 64),
        )


def test_campaign_allows_same_g0_batch_bytes_in_candidate_receipts_but_locks_authority(tmp_path):
    cfg, source_root, _ = _prerequisite_components(tmp_path)
    _validate_fixture_campaign(
        cfg,
        source_root,
        g0_batch_paths=("alpha07-batch.pt", "alpha10-batch.pt"),
    )


def test_campaign_winner_is_derived_from_measured_probe_not_surrogate_pilot(
    tmp_path,
):
    cfg, source_root, _ = _prerequisite_components(tmp_path)
    # surrogate pilot은 alpha=0.7을 선호하지만 measured probe는 alpha=1.0을
    # 0.2 dB 밖으로 명확히 선호한다. canonical cfg가 0.7이므로 반드시 거부된다.
    with pytest.raises(ValueError, match="winner.*canonical pretrain loss"):
        _validate_fixture_campaign(
            cfg,
            source_root,
            pilot_scores=(-5.0, -2.0),
            probe_scores=(-1.0, -2.0),
        )


def test_campaign_rejects_legacy_v5_single_winner_probe_layout(tmp_path):
    cfg, source_root, ledger_path = _prerequisite_components(tmp_path)
    legacy = json.loads(ledger_path.read_text(encoding="utf-8"))
    chains = legacy["loss_pilot_selection"]["candidates"]
    legacy["schema_version"] = 6
    legacy["loss_pilot_selection"]["candidates"] = [
        chain["pilot"] for chain in chains
    ]
    legacy["measured_probe"] = chains[0]["measured_probe"]
    ledger_path.write_text(
        json.dumps(legacy, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    refreshed = _restamp_campaign_anchor(cfg, source_root, ledger_path)
    with pytest.raises(ValueError, match="campaign prerequisite key 집합"):
        _validate_fixture_campaign(refreshed, source_root)


def test_canonical_pretrain_prerequisite_rejects_manual_v4_gradient_fields(tmp_path):
    cfg, source_root, ledger_path = _prerequisite_components(tmp_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["gradient_budget"] = {
        "evidence": ledger["loss_pilot_selection"]["candidates"][0]["g0"]["receipt"],
        "strict_ps": True,
        "lambda_dnh": cfg["loss"]["lambda_dnh"],
        "gradient_share": 0.3,
        "loss_start_sample": cfg["loss_start_sample"],
    }
    ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")
    refreshed = _restamp_campaign_anchor(cfg, source_root, ledger_path)
    with pytest.raises(ValueError, match="gradient budget key 집합"):
        _validate_fixture_campaign(refreshed, source_root)


def _restamp_campaign_anchor(cfg: dict, source_root: Path, ledger_path: Path) -> dict:
    refreshed = deepcopy(cfg)
    refreshed["campaign_prerequisite_sha256"] = hashlib.sha256(
        ledger_path.read_bytes()
    ).hexdigest()
    return stamp_experiment_contract(refreshed, repo_root=source_root)


def test_a100_smoke_receipt_target_tamper_rejects_canonical_ledger(tmp_path):
    cfg, source_root, ledger_path = _prerequisite_components(tmp_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    receipt_path = source_root / ledger["a100_smoke_resume"]["evidence"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["smoke_target_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    ledger["a100_smoke_resume"]["evidence"] = _sha_ref(source_root, receipt_path)
    ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")
    refreshed = _restamp_campaign_anchor(cfg, source_root, ledger_path)
    with pytest.raises(ValueError, match="target/role/init"):
        _validate_fixture_campaign(refreshed, source_root)


def test_a100_smoke_role_misuse_rejects_canonical_ledger(tmp_path):
    cfg, source_root, ledger_path = _prerequisite_components(tmp_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    receipt_path = source_root / ledger["a100_smoke_resume"]["evidence"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["experiment_role"] = "canonical_pretrain"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    ledger["a100_smoke_resume"]["evidence"] = _sha_ref(source_root, receipt_path)
    ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")
    refreshed = _restamp_campaign_anchor(cfg, source_root, ledger_path)
    with pytest.raises(ValueError, match="target/role/init"):
        _validate_fixture_campaign(refreshed, source_root)


def test_a100_smoke_resumed_checkpoint_tamper_rejects_ledger(tmp_path):
    cfg, source_root, ledger_path = _prerequisite_components(tmp_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    receipt_path = source_root / ledger["a100_smoke_resume"]["evidence"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    resumed_path = source_root / receipt["resumed_checkpoint"]["path"]
    state = torch.load(resumed_path, map_location="cpu", weights_only=False)
    first = next(iter(state["model"].values()))
    first.add_(1.0)
    torch.save(state, resumed_path)
    receipt["resumed_checkpoint"] = _sha_ref(source_root, resumed_path)

    telemetry_path = source_root / receipt["telemetry"]["path"]
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    telemetry["resumed_checkpoint"] = receipt["resumed_checkpoint"]
    telemetry_path.write_text(json.dumps(telemetry, sort_keys=True) + "\n", encoding="utf-8")
    receipt["telemetry"] = _sha_ref(source_root, telemetry_path)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    ledger["a100_smoke_resume"]["evidence"] = _sha_ref(source_root, receipt_path)
    ledger["a100_smoke_resume"]["telemetry"] = _sha_ref(source_root, telemetry_path)
    ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")
    refreshed = _restamp_campaign_anchor(cfg, source_root, ledger_path)
    with pytest.raises(ValueError, match="model state"):
        _validate_fixture_campaign(refreshed, source_root)


def test_a100_smoke_resume_input_mismatch_rejects_ledger(tmp_path):
    """final last.pt의 resume pathname도 immutable stop-input receipt와 같아야 한다."""

    cfg, source_root, ledger_path = _prerequisite_components(tmp_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    receipt_path = source_root / ledger["a100_smoke_resume"]["evidence"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    resumed_path = source_root / receipt["resumed_checkpoint"]["path"]
    state = torch.load(resumed_path, map_location="cpu", weights_only=False)
    state["cfg"]["resume"] = "results/training_prerequisites/not-the-stop/ckpt/last.pt"
    torch.save(state, resumed_path)
    receipt["resumed_checkpoint"] = _sha_ref(source_root, resumed_path)

    telemetry_path = source_root / receipt["telemetry"]["path"]
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    telemetry["resumed_checkpoint"] = receipt["resumed_checkpoint"]
    telemetry_path.write_text(json.dumps(telemetry, sort_keys=True) + "\n", encoding="utf-8")
    receipt["telemetry"] = _sha_ref(source_root, telemetry_path)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    ledger["a100_smoke_resume"]["evidence"] = _sha_ref(source_root, receipt_path)
    ledger["a100_smoke_resume"]["telemetry"] = _sha_ref(source_root, telemetry_path)
    ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")
    refreshed = _restamp_campaign_anchor(cfg, source_root, ledger_path)
    with pytest.raises(ValueError, match="cfg.resume"):
        _validate_fixture_campaign(refreshed, source_root)


def test_a100_smoke_default_stop_before_first_eval_allows_only_inf_sentinel(tmp_path):
    """default 300→500 smoke는 stop 시점에 아직 validation best가 없다."""

    _, source_root, ledger_path = _prerequisite_components(tmp_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    receipt_path = source_root / ledger["a100_smoke_resume"]["evidence"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    stop_path = source_root / receipt["stop_checkpoint"]["path"]
    state = torch.load(stop_path, map_location="cpu", weights_only=False)
    assert state["step"] == state["cfg"]["run_until_step"] == 300
    assert state["cfg"]["eval_every"] == 500
    assert state["best_metric"] == float("inf")

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    validate_resume_checkpoint_preview(stop_path, state, model, optimizer, scheduler)

    invalid = deepcopy(state)
    invalid["best_metric"] = float("-inf")
    invalid_path = tmp_path / "invalid" / "stop.pt"
    invalid_path.parent.mkdir()
    torch.save(invalid, invalid_path)
    with pytest.raises(ValueError, match="best_metric"):
        validate_resume_checkpoint_preview(
            invalid_path, invalid, model, optimizer, scheduler
        )


def test_a100_smoke_rejects_actual_torch_cuda_environment_drift(tmp_path):
    """freeze receipt만 같아도 실제 A100 interpreter가 바뀌면 smoke 증거가 아니다."""

    cfg, source_root, ledger_path = _prerequisite_components(tmp_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    receipt_path = source_root / ledger["a100_smoke_resume"]["evidence"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    environment_receipt_path = source_root / receipt["environment_receipt"]["path"]
    environment_receipt = json.loads(
        environment_receipt_path.read_text(encoding="utf-8")
    )
    environment_path = source_root / environment_receipt["uninterrupted_environment"][
        "path"
    ]
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["torch"] = "2.5.2+cu121"
    environment_path.write_text(
        json.dumps(environment, sort_keys=True) + "\n", encoding="utf-8"
    )
    environment_receipt["uninterrupted_environment"] = _sha_ref(
        source_root, environment_path
    )
    environment_receipt_path.write_text(
        json.dumps(environment_receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt["environment_receipt"] = _sha_ref(source_root, environment_receipt_path)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    ledger["a100_smoke_resume"]["evidence"] = _sha_ref(source_root, receipt_path)
    ledger["a100_smoke_resume"]["environment_receipt"] = _sha_ref(
        source_root, environment_receipt_path
    )
    ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")
    refreshed = _restamp_campaign_anchor(cfg, source_root, ledger_path)
    with pytest.raises(ValueError, match="world1/CUDA/결정론"):
        _validate_fixture_campaign(refreshed, source_root)


def test_a100_smoke_rejects_target_root_parent_symlink_escape(tmp_path):
    """lexical ``results/...`` 경로라도 parent symlink 밖 artifact는 인정하지 않는다."""

    cfg, source_root, _ = _prerequisite_components(tmp_path)
    target_root = next(
        (source_root / SMOKE_ROOT).iterdir()
    )
    smoke_parent = target_root.parent
    external = tmp_path / "external-smoke-root"
    shutil.move(str(smoke_parent), external)
    smoke_parent.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="심볼릭"):
        _validate_fixture_campaign(cfg, source_root)


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
