from __future__ import annotations

import hashlib
import json
import random
import runpy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

from deep_anc.dsp.stage2_2khz_contract import (
    STAGE2_2KHZ_SOURCE_FAMILIES,
    Stage2TwoKilohertzContract,
)
from deep_anc.dsp.timing import PlantDelays, TrainingTimingContract
from deep_anc.losses.broadband_loss import CausalFIRPathData
from deep_anc.losses.stage2_2khz_loss import Stage2TwoKilohertzLossConfig
from deep_anc.dsp.stage2_2khz_level_contract import (
    canonical_stage2_operating_level_contract,
)
from deep_anc.train.stage2_2khz_binding import (
    Stage2SourceOperatingLevelBinding,
    Stage2TwoKilohertzPlantBinding,
)
from deep_anc.train.stage2_2khz_execution import (
    Stage2ActualBatchIdentity,
    Stage2CausalPrefixAdapter,
    Stage2FamilyComponentBatchSampler,
    Stage2SamplerRecord,
    Stage2TwoKilohertzTrainerAdapter,
)
from deep_anc.train.stage2_2khz_pretrain_admission import (
    Stage2PretrainDataBinding,
    Stage2PretrainSource,
    Stage2PretrainTypedAdmission,
)
from deep_anc.train import stage2_2khz_runner as runner_module


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _operator(role: str, *, handoff: int) -> CausalFIRPathData:
    fir = np.asarray([1.0], dtype="<f8")
    return CausalFIRPathData(
        role=role,
        post_onset_fir=fir,
        coarse_delay_samples=0,
        fractional_delay_samples=0.0,
        support_samples=1,
        sample_rate=48_000,
        handoff_extra_samples=handoff,
        operator_file_sha256=_sha(f"{role}-file"),
        operator_internal_sha256=_sha(f"{role}-internal"),
        fir_sha256=hashlib.sha256(fir.tobytes()).hexdigest(),
        authority_sha256=_sha("runner-test-binding"),
        source_path=f"artifacts/{role}.npz",
    )


def _fixture_binding() -> Stage2TwoKilohertzPlantBinding:
    primary = _operator("primary", handoff=0)
    secondary = _operator("secondary", handoff=256)
    delays = PlantDelays(
        primary_delay_samples=0,
        secondary_delay_samples=0,
        handoff_samples=256,
        sample_rate=48_000,
    )
    timing = TrainingTimingContract.derive(
        primary_fir=primary.post_onset_fir, plant_delays=delays
    )
    contract = Stage2TwoKilohertzContract.canonical()
    level = canonical_stage2_operating_level_contract()
    actuator = float(level["actuator_limit_abs"])
    return Stage2TwoKilohertzPlantBinding(
        control_band_contract=contract,
        control_band_contract_sha256=contract.digest(),
        training_timing_contract=timing,
        training_timing_contract_sha256=timing.digest(),
        primary_operator=primary,
        secondary_operator=secondary,
        primary_path_sha256=_sha("primary-path"),
        secondary_path_sha256=_sha("secondary-path"),
        raw_capture_sha256=_sha("raw"),
        analysis_sha256=_sha("analysis"),
        measurement_level_evidence_sha256=_sha("level"),
        relative_clock_model_receipt_sha256=_sha("clock"),
        verified_physical_subbands_hz=tuple(
            contract.physical_identification_subbands_hz
        ),
        err_channel_index=0,
        reference_channel_index=1,
        block_size=256,
        binding_file_sha256=_sha("binding-file"),
        source_capture_commit_sha="c" * 40,
        source_operating_level=Stage2SourceOperatingLevelBinding(
            planned_contract_sha256=level["canonical_payload_sha256"],
            physical_evidence_file_sha256=_sha("physical-level-file"),
            physical_evidence_payload_sha256=_sha("physical-level-payload"),
            source_operating_peak_abs=float(level["source_operating_peak_abs"]),
            actuator_limit_abs=actuator,
            augmentation_gain_db_minimum=float(
                level["augmentation_gain_db"]["minimum"]
            ),
            augmentation_gain_db_maximum=float(
                level["augmentation_gain_db"]["maximum"]
            ),
            post_gain_hard_peak_cap_abs=float(
                level["augmentation_gain_db"]["post_gain_hard_peak_cap_abs"]
            ),
            minimum_observed_actuator_headroom_db=4.0,
            broadband_required_control_peak_upper_bound_abs=(
                actuator / 10.0 ** (4.0 / 20.0)
            ),
            actuator_feasibility_passed=True,
            fixture_only=True,
        ),
        fixture_only=True,
    )


def _admission(root: Path) -> Stage2PretrainTypedAdmission:
    records: list[Stage2SamplerRecord] = []
    sources: list[Stage2PretrainSource] = []
    for index, family in enumerate(STAGE2_2KHZ_SOURCE_FAMILIES):
        relative = f"data/{family}.wav"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        generator = np.random.Generator(np.random.PCG64(10_000 + index))
        sf.write(path, generator.normal(0.0, 0.05, 12_000).astype(np.float32), 48_000)
        digest = _file_sha(path)
        records.append(
            Stage2SamplerRecord(
                dataset_index=index,
                source_family=family,
                component_id=f"{family}-component",
                split="train",
                source_sha256=digest,
            )
        )
        sources.append(
            Stage2PretrainSource(
                dataset_index=index,
                relative_path=relative,
                content_sha256=digest,
                native_sample_rate=48_000,
            )
        )
    manifest_sha = _sha("manifest")
    sampler_sha = _sha("sampler")
    sampler = Stage2FamilyComponentBatchSampler(
        records,
        batch_size=4,
        seed=20260803,
        manifest_bundle_sha256=manifest_sha,
        sampler_receipt_sha256=sampler_sha,
    )
    contract = Stage2TwoKilohertzContract.canonical()
    loss_config = Stage2TwoKilohertzLossConfig(
        control_band_contract=contract,
        control_band_contract_sha256=contract.digest(),
        lambda_dnh=0.001,
        dnh_calibration_receipt_sha256=_sha("dnh"),
        dnh_observed_gradient_share=0.3,
        family_balanced_sampler_receipt_sha256=sampler_sha,
    )
    return Stage2PretrainTypedAdmission(
        plant_binding=_fixture_binding(),
        data_binding=Stage2PretrainDataBinding(
            manifest_bundle_sha256=manifest_sha,
            lineage_receipt_sha256=_sha("lineage"),
            frequency_coverage_receipt_sha256=_sha("coverage"),
            transfer_bootstrap_receipt_sha256=_sha("bootstrap"),
            records=tuple(records),
            sources=tuple(sources),
        ),
        sampler=sampler,
        loss_config=loss_config,
        criterion_receipt_sha256=_sha("criterion"),
        dnh_calibration_receipt_sha256=_sha("dnh"),
        sampler_receipt_sha256=sampler_sha,
    )


class _ScaleStreaming(nn.Module):
    hop = 256

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(-0.25))

    def init_states(self, batch: int = 1, device: torch.device | str = "cpu") -> None:
        del batch, device
        return None

    def streaming_step(
        self, x_block: torch.Tensor, state: None
    ) -> tuple[torch.Tensor, None]:
        return self.scale * x_block[:, :1], state


def _launch_material(root: Path) -> tuple[Stage2PretrainTypedAdmission, dict, dict]:
    model_path = root / "configs/model-test.yaml"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(
        "in_channels: 2\nlimiter:\n  limit: 0.00299072265625\n",
        encoding="utf-8",
    )
    profile = {
        "steps": 100_000,
        "seed": 20260803,
        "execution": {
            "schedule": {
                "total_steps": 100_000,
                "warmup_steps": 2,
                "min_lr": 0.00001,
            },
            "required_world_size": 1,
            "optimizer": {
                "lr": 0.001,
                "weight_decay": 0.0001,
                "betas": [0.9, 0.999],
            },
            "target_samples": 4096,
            "batch_size": 4,
            "grad_clip_norm": 5.0,
            "checkpoint_every_steps": 500,
            "smoke_milestones": [1, 2],
        },
    }
    anchors = {
        "repository_commit_sha": "a" * 40,
        "external_experiment_contract_sha256": "e" * 64,
        "pretrain_profile_sha256": "f" * 64,
        "evaluation_policy_sha256": "9" * 64,
        "model_config_path": "configs/model-test.yaml",
        "model_config_sha256": _file_sha(model_path),
    }
    return _admission(root), profile, anchors


def _artifact_ref(root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": _file_sha(path)}


def _acceptance_checkpoint(
    root: Path,
    run_dir: Path,
    *,
    step: int,
    bindings: dict,
) -> tuple[Path, dict]:
    path = run_dir / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    generator = np.random.RandomState(123)
    _, values, position, has_gauss, cached = generator.get_state()
    checkpoint = {
        "schema": runner_module.STAGE2_PRETRAIN_CHECKPOINT_SCHEMA,
        "completed_steps": step,
        "init_eligible": False,
        "bindings": bindings,
        "model": {"weight": torch.tensor([1.25], dtype=torch.float32)},
        "optimizer": {
            "state": {
                0: {
                    "step": torch.tensor(float(step)),
                    "exp_avg": torch.tensor([0.25]),
                    "exp_avg_sq": torch.tensor([0.5]),
                }
            },
            "param_groups": [{"params": [0], "lr": 0.001}],
        },
        "scheduler": {"last_epoch": step, "_step_count": step + 1},
        "step": step,
        "torch_rng_state": torch.arange(32, dtype=torch.uint8),
        "numpy_rng_state": {
            "algorithm": "MT19937",
            "values": torch.from_numpy(values.astype(np.int64)),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        },
        "python_rng_state": random.Random(123).getstate(),
        "cuda_rng_state_all": [torch.arange(16, dtype=torch.uint8)],
        "last_metrics": {"step": float(step), "loss": 0.5},
    }
    torch.save(checkpoint, path)
    sidecar = runner_module.stage2_checkpoint_binding_payload(
        checkpoint_sha256=_file_sha(path),
        bindings=bindings,
        completed_steps=step,
        production_execution=True,
    )
    sidecar_path = path.with_suffix(".binding.json")
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
    return path, checkpoint


def _acceptance_telemetry(
    path: Path,
    *,
    total_step_ms: float,
    target_samples: int,
) -> list[dict]:
    rows: list[dict] = []
    for step in range(1, 501):
        rows.append(
            {
                "schema": runner_module.STAGE2_PRETRAIN_STEP_TELEMETRY_SCHEMA,
                "step": step,
                "batch_size": 96,
                "global_sample_index_first": (step - 1) * 96,
                "global_sample_index_last": step * 96 - 1,
                "data_wait_ms": 1.0,
                "h2d_ms": 1.0,
                "compute_step_ms": total_step_ms - 2.0,
                "total_step_ms": total_step_ms,
                "samples_per_second": 96 * target_samples / (total_step_ms / 1000.0),
                "gpu_memory_allocated_bytes": 1024,
                "gpu_memory_reserved_bytes": 2048,
                "gpu_peak_allocated_bytes": 1024,
                "gpu_peak_reserved_bytes": 2048,
                "gpu_utilization_percent": 90 if step % 10 == 0 else None,
                "nvidia_smi_memory_used_mib": 4096 if step % 10 == 0 else None,
            }
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def test_stage2_runner_explicit_resume_is_numerically_equal_to_uninterrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uninterrupted_root = tmp_path / "uninterrupted"
    resumed_root = tmp_path / "resumed"
    uninterrupted_root.mkdir()
    resumed_root.mkdir()
    materials = {
        uninterrupted_root.resolve(): _launch_material(uninterrupted_root),
        resumed_root.resolve(): _launch_material(resumed_root),
    }

    def ready(_campaign: dict, *, repo_root: Path):
        return materials[Path(repo_root).resolve()]

    monkeypatch.setattr(runner_module, "load_ready_stage2_pretrain_launch", ready)
    monkeypatch.setattr(
        runner_module, "_preverify_campaign_checkout", lambda *_: "a" * 40
    )
    monkeypatch.setattr(runner_module, "_verify_exact_clean_checkout", lambda *_: None)
    monkeypatch.setattr(runner_module, "build_model", lambda _cfg: _ScaleStreaming())
    monkeypatch.setattr(
        Stage2CausalPrefixAdapter,
        "from_verified_binding",
        classmethod(lambda cls, binding: cls._for_test_fixture(binding)),
    )
    monkeypatch.setattr(
        Stage2TwoKilohertzTrainerAdapter,
        "from_verified_components",
        classmethod(
            lambda cls, adapter, criterion, **kwargs: cls._for_test_fixture(
                adapter, criterion, **kwargs
            )
        ),
    )

    uninterrupted = runner_module.Stage2ScratchPretrainRunner(
        repository_root=uninterrupted_root,
        campaign={},
        run_until_step=2,
        _allow_cpu_test=True,
    ).train()
    first = runner_module.Stage2ScratchPretrainRunner(
        repository_root=resumed_root,
        campaign={},
        run_until_step=1,
        _allow_cpu_test=True,
    ).train()
    resumed = runner_module.Stage2ScratchPretrainRunner(
        repository_root=resumed_root,
        campaign={},
        run_until_step=2,
        resume=first,
        _allow_cpu_test=True,
    ).train()

    direct_state = torch.load(uninterrupted, map_location="cpu", weights_only=False)
    resumed_state = torch.load(resumed, map_location="cpu", weights_only=False)
    assert direct_state["schema"] == runner_module.STAGE2_PRETRAIN_DIAGNOSTIC_CHECKPOINT_SCHEMA
    assert direct_state["init_eligible"] is False
    assert direct_state["step"] == resumed_state["step"] == 2
    assert direct_state["bindings"] == resumed_state["bindings"]
    for name, tensor in direct_state["model"].items():
        assert torch.equal(tensor, resumed_state["model"][name])
    assert direct_state["optimizer"] == resumed_state["optimizer"]
    assert direct_state["scheduler"] == resumed_state["scheduler"]
    assert direct_state["last_metrics"] == resumed_state["last_metrics"]
    assert direct_state["bindings"]["a100_environment_sha256"]

    telemetry_rows = [
        json.loads(line)
        for line in (
            uninterrupted_root
            / "runs"
            / "stage2_pretrain_eeeeeeeeeeee_20260803"
            / "step_telemetry.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in telemetry_rows] == [1, 2]
    assert all(row["schema"] == runner_module.STAGE2_PRETRAIN_STEP_TELEMETRY_SCHEMA for row in telemetry_rows)
    assert all(row["batch_size"] == 4 for row in telemetry_rows)
    assert all(
        row["total_step_ms"] >= row["data_wait_ms"] + row["h2d_ms"]
        for row in telemetry_rows
    )
    assert all(
        row["samples_per_second"]
        == pytest.approx(4 * 4096 / (row["total_step_ms"] / 1000.0))
        for row in telemetry_rows
    )
    smoke = json.loads(
        (
            uninterrupted_root
            / "runs"
            / "stage2_pretrain_eeeeeeeeeeee_20260803"
            / "smoke_performance_step_000002.json"
        ).read_text(encoding="utf-8")
    )
    assert smoke["status"] == "DIAGNOSTIC_CPU_ONLY"
    assert smoke["canonical_100k_start_eligible"] is False

    sidecar_path = first.with_suffix(".binding.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["checkpoint_sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
    load_calls = 0

    def forbidden_torch_load(*_args, **_kwargs):
        nonlocal load_calls
        load_calls += 1
        raise AssertionError("untrusted checkpoint에 torch.load를 호출하면 안 됩니다")

    monkeypatch.setattr(runner_module.torch, "load", forbidden_torch_load)
    with pytest.raises(ValueError, match="SHA/sidecar/contract binding"):
        runner_module.Stage2ScratchPretrainRunner(
            repository_root=resumed_root,
            campaign={},
            run_until_step=2,
            resume=first,
            _allow_cpu_test=True,
        )
    assert load_calls == 0


def test_default_campaign_blocks_runner_before_cuda_or_run_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    campaign = __import__("yaml").safe_load(
        (root / "configs/stage2_2khz_campaign.yaml").read_text(encoding="utf-8")
    )

    def unexpected_cuda() -> bool:
        raise AssertionError("BLOCKED admission 뒤 CUDA를 조회하면 안 됩니다")

    def unexpected_admission(*_args, **_kwargs):
        raise AssertionError("clean exact commit 전 corpus/P-S admission scan을 하면 안 됩니다")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda)
    monkeypatch.setattr(
        runner_module, "load_ready_stage2_pretrain_launch", unexpected_admission
    )
    before = tuple((root / "runs").glob("stage2_pretrain_*"))
    with pytest.raises(
        ValueError, match="origin/dev exact HEAD|external contract/commit"
    ):
        runner_module.Stage2ScratchPretrainRunner(
            repository_root=root,
            campaign=campaign,
            run_until_step=200,
        )
    assert tuple((root / "runs").glob("stage2_pretrain_*")) == before


def test_100k_is_blocked_before_cuda_without_raw_smoke_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, profile, anchors = _launch_material(tmp_path)
    monkeypatch.setattr(
        runner_module,
        "load_ready_stage2_pretrain_launch",
        lambda *_args, **_kwargs: (admission, profile, anchors),
    )
    monkeypatch.setattr(
        runner_module, "_preverify_campaign_checkout", lambda *_: "a" * 40
    )

    def forbidden_environment(*_args, **_kwargs):
        raise AssertionError("raw smoke acceptance 없이 CUDA/environment를 열면 안 됩니다")

    monkeypatch.setattr(
        runner_module,
        "configure_and_collect_stage2_a100_environment",
        forbidden_environment,
    )
    before = tuple((tmp_path / "runs").glob("stage2_pretrain_*"))
    with pytest.raises(ValueError, match="raw-bound smoke acceptance"):
        runner_module.Stage2ScratchPretrainRunner(
            repository_root=tmp_path,
            campaign={},
            run_until_step=100_000,
        )
    assert tuple((tmp_path / "runs").glob("stage2_pretrain_*")) == before


@pytest.mark.parametrize(
    ("run_until_step", "run_label", "resume", "message"),
    (
        (200, None, None, "200-step"),
        (200, "uninterrupted", None, "200-step"),
        (500, "resumed", None, "resumed 500-step"),
        (500, "uninterrupted", "runs/not-a-resume.pt", "uninterrupted 500-step"),
        (100_000, "resumed", None, "canonical 100k"),
        (100_000, None, "runs/not-a-resume.pt", "canonical 100k"),
        (300, "resumed", None, "200, 500 또는 canonical"),
    ),
)
def test_production_run_labels_fail_closed_before_gpu_or_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_until_step: int,
    run_label: str | None,
    resume: str | None,
    message: str,
) -> None:
    """smoke label은 CLI 편의값이 아니라 production namespace admission이다."""

    admission, profile, anchors = _launch_material(tmp_path)
    monkeypatch.setattr(
        runner_module,
        "load_ready_stage2_pretrain_launch",
        lambda *_args, **_kwargs: (admission, profile, anchors),
    )
    monkeypatch.setattr(
        runner_module, "_preverify_campaign_checkout", lambda *_args: "a" * 40
    )
    monkeypatch.setattr(
        runner_module,
        "configure_and_collect_stage2_a100_environment",
        lambda *_args, **_kwargs: pytest.fail("invalid smoke run이 GPU를 열면 안 됩니다"),
    )

    with pytest.raises(ValueError, match=message):
        runner_module.Stage2ScratchPretrainRunner(
            repository_root=tmp_path,
            campaign={},
            run_until_step=run_until_step,
            run_label=run_label,
            resume=resume,
        )
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    ("run_until_step", "run_label", "requires_smoke_acceptance"),
    (
        (200, "resumed", False),
        (500, "uninterrupted", False),
        (100_000, None, True),
    ),
)
def test_existing_fresh_run_directory_fails_before_cuda_model_or_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_until_step: int,
    run_label: str | None,
    requires_smoke_acceptance: bool,
) -> None:
    """fresh arm collision은 no-replace일 뿐 아니라 GPU 비용 전 경계다."""

    admission, profile, anchors = _launch_material(tmp_path)
    profile = {
        **profile,
        "execution": {
            **profile["execution"],
            "data_pipeline": {
                "loader_workers": 1,
                "bounded_prefetch_batches": 1,
                "source_cache_items": 8,
                "valid_start_candidates_per_source": 64,
                "pin_memory": True,
                "non_blocking_h2d": True,
            },
        },
    }
    monkeypatch.setattr(
        runner_module,
        "load_ready_stage2_pretrain_launch",
        lambda *_args, **_kwargs: (admission, profile, anchors),
    )
    monkeypatch.setattr(
        runner_module, "_preverify_campaign_checkout", lambda *_args: "a" * 40
    )
    monkeypatch.setattr(
        runner_module,
        "configure_and_collect_stage2_a100_environment",
        lambda *_args, **_kwargs: pytest.fail(
            "existing fresh run directory collision 전에 CUDA를 열면 안 됩니다"
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "build_model",
        lambda *_args, **_kwargs: pytest.fail(
            "existing fresh run directory collision 전에 model을 만들면 안 됩니다"
        ),
    )

    run_dir = runner_module.stage2_pretrain_run_directory(
        tmp_path,
        external_experiment_contract_sha256="e" * 64,
        seed=20260803,
        run_label=run_label,
    )
    run_dir.mkdir(parents=True)
    smoke_acceptance: Path | None = None
    if requires_smoke_acceptance:
        smoke_acceptance = tmp_path / "smoke_acceptance.json"
        smoke_acceptance.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="기존 Stage-2 fresh .* run directory"):
        runner_module.Stage2ScratchPretrainRunner(
            repository_root=tmp_path,
            campaign={},
            run_until_step=run_until_step,
            run_label=run_label,
            smoke_acceptance=smoke_acceptance,
        )
    assert tuple(run_dir.iterdir()) == ()


def test_resumed_500_keeps_existing_resumed_namespace_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resumed arm은 immutable 200-step namespace를 새 run collision으로 오인하지 않는다."""

    admission, profile, anchors = _launch_material(tmp_path)
    profile = {
        **profile,
        "execution": {
            **profile["execution"],
            "data_pipeline": {
                "loader_workers": 1,
                "bounded_prefetch_batches": 1,
                "source_cache_items": 8,
                "valid_start_candidates_per_source": 64,
                "pin_memory": True,
                "non_blocking_h2d": True,
            },
        },
    }
    monkeypatch.setattr(
        runner_module,
        "load_ready_stage2_pretrain_launch",
        lambda *_args, **_kwargs: (admission, profile, anchors),
    )
    monkeypatch.setattr(
        runner_module, "_preverify_campaign_checkout", lambda *_args: "a" * 40
    )
    monkeypatch.setattr(runner_module, "_nvidia_smi_l_snapshot", lambda *_args: "A100")

    def expected_environment_probe(*_args, **_kwargs):
        raise RuntimeError("resumed 500 reached environment admission")

    monkeypatch.setattr(
        runner_module,
        "configure_and_collect_stage2_a100_environment",
        expected_environment_probe,
    )
    run_dir = runner_module.stage2_pretrain_run_directory(
        tmp_path,
        external_experiment_contract_sha256="e" * 64,
        seed=20260803,
        run_label="resumed",
    )
    run_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="resumed 500 reached environment admission"):
        runner_module.Stage2ScratchPretrainRunner(
            repository_root=tmp_path,
            campaign={},
            run_until_step=500,
            run_label="resumed",
            resume=run_dir / "step_000200.pt",
        )
    assert tuple(run_dir.iterdir()) == ()


def test_raw_smoke_acceptance_recomputes_chain_and_binds_100k_checkpoint(
    tmp_path: Path,
) -> None:
    external_sha = "e" * 64
    environment = {"schema": "actual-a100-fixture", "device": "A100 80GB"}
    environment_sha = runner_module.stage2_a100_environment_sha256(environment)
    contract = Stage2TwoKilohertzContract.canonical()
    common_bindings = {
        "runner_schema": runner_module.STAGE2_PRETRAIN_RUNNER_SCHEMA,
        "repository_commit_sha": "a" * 40,
        "stage2_contract_id": contract.contract_id,
        "stage2_contract_sha256": contract.digest(),
        "primary_path_sha256": "1" * 64,
        "secondary_path_sha256": "2" * 64,
        "plant_binding_sha256": "3" * 64,
        "plant_binding_runtime_sha256": "4" * 64,
        "manifest_bundle_sha256": "5" * 64,
        "training_profile_sha256": "6" * 64,
        "evaluation_policy_sha256": "7" * 64,
        "model_config_sha256": "8" * 64,
        "external_experiment_contract_sha256": external_sha,
        "criterion_receipt_sha256": "9" * 64,
        "sampler_receipt_sha256": "b" * 64,
        "dnh_calibration_receipt_sha256": "c" * 64,
        "scratch_pretrain": True,
        "legacy_origin": False,
        "automatic_resume_allowed": False,
        "a100_environment_sha256": environment_sha,
        "smoke_acceptance_sha256": None,
    }
    resumed_bindings = {
        **common_bindings,
        "run_kind": "smoke",
        "run_label": "resumed",
    }
    uninterrupted_bindings = {
        **common_bindings,
        "run_kind": "smoke",
        "run_label": "uninterrupted",
    }
    profile = {
        "seed": 20260803,
        "execution": {
            "batch_size": 96,
            "target_samples": 16_384,
            "data_pipeline": {
                "loader_workers": 14,
                "bounded_prefetch_batches": 56,
            },
            "telemetry": {"nvidia_smi_sample_every_steps": 10},
        },
    }
    run_dir = runner_module.stage2_pretrain_run_directory(
        tmp_path,
        external_experiment_contract_sha256=external_sha,
        seed=20260803,
        run_label="resumed",
    )
    independent_dir = runner_module.stage2_pretrain_run_directory(
        tmp_path,
        external_experiment_contract_sha256=external_sha,
        seed=20260803,
        run_label="uninterrupted",
    )
    run_dir.mkdir(parents=True)
    independent_dir.mkdir(parents=True)
    identity = runner_module._stage2_smoke_run_identity(
        repository_commit_sha="a" * 40,
        external_sha256=external_sha,
        seed=20260803,
        run_label="resumed",
        run_until_step=200,
        environment_sha256=environment_sha,
        loader_workers=14,
        bounded_prefetch_batches=56,
    )
    identity_path = run_dir / "run_identity.json"
    environment_path = run_dir / "environment.json"
    identity_path.write_text(json.dumps(identity, sort_keys=True), encoding="utf-8")
    environment_path.write_text(json.dumps(environment, sort_keys=True), encoding="utf-8")
    step_200_path, _ = _acceptance_checkpoint(
        tmp_path, run_dir, step=200, bindings=resumed_bindings
    )
    step_500_path, step_500 = _acceptance_checkpoint(
        tmp_path, run_dir, step=500, bindings=resumed_bindings
    )
    uninterrupted_path, _ = _acceptance_checkpoint(
        tmp_path, independent_dir, step=500, bindings=uninterrupted_bindings
    )
    telemetry_path = run_dir / "step_telemetry.jsonl"
    rows = _acceptance_telemetry(
        telemetry_path, total_step_ms=10.0, target_samples=16_384
    )
    independent_telemetry_path = independent_dir / "step_telemetry.jsonl"
    _acceptance_telemetry(
        independent_telemetry_path,
        total_step_ms=11.0,
        target_samples=16_384,
    )
    independent_identity_path = independent_dir / "run_identity.json"
    independent_identity_path.write_text(
        json.dumps(
            runner_module._stage2_smoke_run_identity(
                repository_commit_sha="a" * 40,
                external_sha256=external_sha,
                seed=20260803,
                run_label="uninterrupted",
                run_until_step=500,
                environment_sha256=environment_sha,
                loader_workers=14,
                bounded_prefetch_batches=56,
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (independent_dir / "environment.json").write_text(
        json.dumps(environment, sort_keys=True), encoding="utf-8"
    )

    telemetry_bytes = telemetry_path.read_bytes()
    prefix_sha = hashlib.sha256(
        b"".join(telemetry_bytes.splitlines(keepends=True)[:200])
    ).hexdigest()
    smoke_kwargs = {
        "rows": rows,
        "environment_sha256": environment_sha,
        "loader_workers": 14,
        "bounded_prefetch_batches": 56,
    }
    smoke_200_path = run_dir / "smoke_performance_step_000200.json"
    smoke_500_path = run_dir / "smoke_performance_step_000500.json"
    smoke_200_path.write_text(
        json.dumps(
            runner_module._smoke_expected_payload(
                **smoke_kwargs,
                completed_step=200,
                telemetry_sha256=prefix_sha,
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    smoke_500_path.write_text(
        json.dumps(
            runner_module._smoke_expected_payload(
                **smoke_kwargs,
                completed_step=500,
                telemetry_sha256=_file_sha(telemetry_path),
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    acceptance_path, acceptance_sha = runner_module.issue_stage2_smoke_acceptance_no_replace(
        tmp_path,
        resumed_run_dir=run_dir,
        uninterrupted_run_dir=independent_dir,
        external_sha256=external_sha,
        environment=environment,
        environment_sha256=environment_sha,
        profile=profile,
        resumed_expected_bindings=resumed_bindings,
        uninterrupted_expected_bindings=uninterrupted_bindings,
    )
    assert acceptance_sha == _file_sha(acceptance_path)
    assert acceptance_path == run_dir / "smoke_acceptance.json"
    assert (run_dir / "uninterrupted_vs_resume_equivalence.json").is_file()
    with pytest.raises(FileExistsError, match="덮어쓸 수 없습니다"):
        runner_module.issue_stage2_smoke_acceptance_no_replace(
            tmp_path,
            resumed_run_dir=run_dir,
            uninterrupted_run_dir=independent_dir,
            external_sha256=external_sha,
            environment=environment,
            environment_sha256=environment_sha,
            profile=profile,
            resumed_expected_bindings=resumed_bindings,
            uninterrupted_expected_bindings=uninterrupted_bindings,
        )

    # acceptance JSON의 PASS scalar만 믿지 않는다. 실제 weights bytes가 변하면
    # existing no-replace receipt를 다시 읽을 때 즉시 실패해야 한다.
    with step_500_path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="SHA|checkpoint"):
        runner_module._validate_stage2_smoke_acceptance(
            tmp_path,
            acceptance_path=acceptance_path,
            resumed_run_dir=run_dir,
            uninterrupted_run_dir=independent_dir,
            external_sha256=external_sha,
            environment=environment,
            environment_sha256=environment_sha,
            profile=profile,
            resumed_expected_bindings=resumed_bindings,
            uninterrupted_expected_bindings=uninterrupted_bindings,
        )

    with pytest.raises(ValueError, match="smoke acceptance SHA"):
        runner_module.stage2_checkpoint_binding_payload(
            checkpoint_sha256="d" * 64,
            bindings=resumed_bindings,
            completed_steps=100_000,
            production_execution=True,
        )
    final_bindings = {
        **resumed_bindings,
        "run_kind": "canonical",
        "run_label": None,
        "smoke_acceptance_sha256": acceptance_sha,
    }
    sidecar = runner_module.stage2_checkpoint_binding_payload(
        checkpoint_sha256="d" * 64,
        bindings=final_bindings,
        completed_steps=100_000,
        production_execution=True,
    )
    assert sidecar["smoke_acceptance_sha256"] == acceptance_sha

    incomplete = dict(step_500)
    incomplete.pop("optimizer")
    with pytest.raises(ValueError, match="key 집합"):
        runner_module._validate_production_checkpoint_state(
            incomplete,
            expected_bindings=resumed_bindings,
            completed_steps=500,
            label="tampered",
        )


def test_smoke_launchers_keep_canonical_and_independent_runs_separate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI가 smoke arm을 canonical 100k namespace와 섞지 않는지 고정한다."""

    scripts_root = Path(__file__).resolve().parents[1] / "scripts" / "train"
    campaign_path = tmp_path / "configs" / "stage2_2khz_campaign.yaml"
    campaign_path.parent.mkdir(parents=True)
    campaign_path.write_text("{}\n", encoding="utf-8")
    calls: list[dict[str, object]] = []
    issued: list[dict[str, object]] = []

    class FakeRunner:
        def __init__(self, **kwargs: object) -> None:
            calls.append(dict(kwargs))
            self.root = Path(str(kwargs["repository_root"])).resolve()
            self.profile = {"fixture": "shared"}
            self.a100_environment = {"fixture": "A100"}
            self.a100_environment_sha256 = "a" * 64
            self.execution_repository_commit_sha = "b" * 40
            self.anchors = {"external_experiment_contract_sha256": "c" * 64}
            self.run_label = kwargs.get("run_label")
            label = str(self.run_label or "canonical")
            self.run_dir = self.root / "runs" / f"fixture_{label}"
            self.run_until_step = int(kwargs["run_until_step"])

        def train(self) -> Path:
            return self.run_dir / f"step_{self.run_until_step:06d}.pt"

        def _binding_metadata(self, **kwargs: object) -> dict[str, object]:
            return {
                "fixture": "binding",
                "run_label": self.run_label,
                "include_smoke_acceptance": kwargs.get("include_smoke_acceptance"),
            }

    def fake_issue(repository_root: Path, **kwargs: object) -> tuple[Path, str]:
        assert Path(repository_root).resolve() == tmp_path.resolve()
        issued.append(dict(kwargs))
        resumed_dir = Path(str(kwargs["resumed_run_dir"]))
        return resumed_dir / "smoke_acceptance.json", "d" * 64

    smoke_module = runpy.run_path(
        str(scripts_root / "run_stage2_2khz_pretrain_smoke.py")
    )
    smoke_main = smoke_module["main"]
    smoke_main.__globals__["REPO_ROOT"] = tmp_path
    smoke_main.__globals__["Stage2ScratchPretrainRunner"] = FakeRunner
    smoke_main.__globals__["issue_stage2_smoke_acceptance_no_replace"] = fake_issue
    assert smoke_main(["--campaign", "configs/stage2_2khz_campaign.yaml"]) == 0

    assert [call["run_until_step"] for call in calls] == [200, 500, 500]
    assert [call["run_label"] for call in calls] == [
        "resumed",
        "resumed",
        "uninterrupted",
    ]
    assert calls[0].get("resume") is None
    assert calls[1]["resume"] == tmp_path / "runs" / "fixture_resumed" / "step_000200.pt"
    assert calls[2].get("resume") is None
    assert len(issued) == 1
    assert issued[0]["resumed_run_dir"] == tmp_path / "runs" / "fixture_resumed"
    assert issued[0]["uninterrupted_run_dir"] == tmp_path / "runs" / "fixture_uninterrupted"
    smoke_output = capsys.readouterr().out
    assert "--run-until-step 100000" in smoke_output
    assert "--resume" not in smoke_output
    assert "--run-label" not in smoke_output

    # generic launcher도 label을 runner에 그대로 넘기며 canonical 위치를 추론하지 않는다.
    calls.clear()
    generic_module = runpy.run_path(str(scripts_root / "train_stage2_2khz.py"))
    generic_main = generic_module["main"]
    generic_main.__globals__["REPO_ROOT"] = tmp_path
    generic_main.__globals__["Stage2ScratchPretrainRunner"] = FakeRunner
    assert generic_main(
        [
            "--campaign",
            "configs/stage2_2khz_campaign.yaml",
            "--run-until-step",
            "500",
            "--run-label",
            "uninterrupted",
        ]
    ) == 0
    assert len(calls) == 1
    assert calls[0]["run_label"] == "uninterrupted"
    assert calls[0].get("resume") is None

    canonical = runner_module.stage2_pretrain_run_directory(
        tmp_path,
        external_experiment_contract_sha256="c" * 64,
        seed=20260803,
    )
    resumed = runner_module.stage2_pretrain_run_directory(
        tmp_path,
        external_experiment_contract_sha256="c" * 64,
        seed=20260803,
        run_label="resumed",
    )
    uninterrupted = runner_module.stage2_pretrain_run_directory(
        tmp_path,
        external_experiment_contract_sha256="c" * 64,
        seed=20260803,
        run_label="uninterrupted",
    )
    assert len({canonical, resumed, uninterrupted}) == 3
    assert canonical.name.startswith("stage2_pretrain_")
    assert resumed.name.startswith("stage2_pretrain_smoke_")
    assert uninterrupted.name.startswith("stage2_pretrain_smoke_")


def test_bounded_prefetch_order_is_worker_count_and_resume_independent(
    tmp_path: Path,
) -> None:
    admission = _admission(tmp_path)

    class IdentityLoader:
        @staticmethod
        def build(identity):
            # 역순 sleep으로 completion 순서를 의도적으로 흔든다.
            time.sleep(0.002 * (4 - (identity.global_step % 4)))
            return identity.global_sample_indices

    def collect(*, workers: int, start: int, end: int):
        with runner_module.Stage2DeterministicBatchPrefetcher(
            loader=IdentityLoader(),
            sampler=admission.sampler,
            start_step=start,
            end_step=end,
            workers=workers,
            prefetch_batches=4,
        ) as batches:
            return [
                (step, identity.global_sample_indices, value)
                for step, identity, value in batches
            ]

    single = collect(workers=1, start=0, end=6)
    parallel = collect(workers=4, start=0, end=6)
    resumed_tail = collect(workers=3, start=3, end=6)
    assert single == parallel
    assert resumed_tail == single[3:]


def test_source_prepare_cache_has_no_parallel_decode_race(tmp_path: Path) -> None:
    admission = _admission(tmp_path)
    loader = runner_module.Stage2PublicTensorLoader(
        repository_root=tmp_path,
        admission=admission,
        target_samples=4096,
        cache_items=8,
        valid_start_candidates=64,
        model_actuator_limit_abs=float(
            admission.plant_binding.source_operating_level.actuator_limit_abs
        ),
    )
    source = admission.data_binding.sources[0]
    original = loader._decode_bytes
    count = 0
    count_lock = threading.Lock()

    def counted(value):
        nonlocal count
        with count_lock:
            count += 1
        time.sleep(0.02)
        return original(value)

    loader._decode_bytes = counted  # type: ignore[method-assign]
    with ThreadPoolExecutor(max_workers=8) as executor:
        prepared = list(executor.map(lambda _: loader._prepare(source), range(8)))
    assert count == 1
    assert all(value is prepared[0] for value in prepared)
    assert prepared[0].valid_starts


def test_valid_start_cache_survives_waveform_lru_eviction(tmp_path: Path) -> None:
    admission = _admission(tmp_path)
    loader = runner_module.Stage2PublicTensorLoader(
        repository_root=tmp_path,
        admission=admission,
        target_samples=4096,
        cache_items=1,
        valid_start_candidates=64,
        model_actuator_limit_abs=float(
            admission.plant_binding.source_operating_level.actuator_limit_abs
        ),
    )
    calls = 0
    original = loader._precompute_valid_starts

    def counted(samples):
        nonlocal calls
        calls += 1
        return original(samples)

    loader._precompute_valid_starts = counted  # type: ignore[method-assign]
    first, second = admission.data_binding.sources[:2]
    loader._prepare(first)
    loader._prepare(second)
    loader._prepare(first)
    assert calls == 2
    assert set(loader._valid_start_cache) == {
        int(first.dataset_index),
        int(second.dataset_index),
    }


def test_actual_batch_consumes_physical_source_peak_and_limiter_feasibility(
    tmp_path: Path,
) -> None:
    admission = _admission(tmp_path)
    level = admission.plant_binding.source_operating_level
    assert level is not None
    loader = runner_module.Stage2PublicTensorLoader(
        repository_root=tmp_path,
        admission=admission,
        target_samples=4096,
        cache_items=8,
        valid_start_candidates=64,
        model_actuator_limit_abs=level.actuator_limit_abs,
    )
    identity = Stage2ActualBatchIdentity.from_sampler(
        admission.sampler, global_step=3
    )
    batch = loader.build(identity)
    clean = batch.causal.clean_playback_timeline.detach().cpu().numpy()
    peaks = np.max(np.abs(clean), axis=(1, 2))
    assert np.all(peaks > 0.0)
    assert np.all(peaks <= level.post_gain_hard_peak_cap_abs + 1.0e-12)
    minimum_gain = 10.0 ** (level.augmentation_gain_db_minimum / 20.0)
    assert np.all(
        peaks
        >= level.source_operating_peak_abs * minimum_gain - 2.0e-10
    )

    with pytest.raises(ValueError, match="model limiter"):
        runner_module.Stage2PublicTensorLoader(
            repository_root=tmp_path,
            admission=admission,
            target_samples=4096,
            model_actuator_limit_abs=level.actuator_limit_abs / 2.0,
        )
    with pytest.raises(ValueError, match="3 dB|headroom"):
        Stage2SourceOperatingLevelBinding(
            planned_contract_sha256=level.planned_contract_sha256,
            physical_evidence_file_sha256=level.physical_evidence_file_sha256,
            physical_evidence_payload_sha256=level.physical_evidence_payload_sha256,
            source_operating_peak_abs=level.source_operating_peak_abs,
            actuator_limit_abs=level.actuator_limit_abs,
            augmentation_gain_db_minimum=level.augmentation_gain_db_minimum,
            augmentation_gain_db_maximum=level.augmentation_gain_db_maximum,
            post_gain_hard_peak_cap_abs=level.post_gain_hard_peak_cap_abs,
            minimum_observed_actuator_headroom_db=2.99,
            broadband_required_control_peak_upper_bound_abs=(
                level.actuator_limit_abs / 10.0 ** (2.99 / 20.0)
            ),
            actuator_feasibility_passed=True,
            fixture_only=True,
        )
