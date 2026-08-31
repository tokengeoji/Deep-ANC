from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from deep_anc.dsp.stage2_2khz_contract import Stage2TwoKilohertzContract
from deep_anc.dsp.timing import PlantDelays, TrainingTimingContract
from deep_anc.losses.broadband_loss import CausalFIRPathData
from deep_anc.train.causal_secondary_prefix_adapter_v1 import (
    CausalPrefixBatchV1,
    CausalPrefixStateOriginV1,
)
from deep_anc.train.stage2_2khz_binding import Stage2TwoKilohertzPlantBinding
from deep_anc.train.stage2_2khz_execution import (
    Stage2ActualBatchIdentity,
    Stage2CausalPrefixAdapter,
    Stage2FamilyComponentBatchSampler,
    Stage2SamplerRecord,
    Stage2TensorBatch,
)
from deep_anc.train.stage2_2khz_pretrain_issuer import (
    DNH_SHARE_MAX,
    DNH_SHARE_MIN,
    DNH_SHARE_TARGET,
    Stage2DNHCalibrationSnapshot,
    build_canonical_pretrain_external_contract,
    build_criterion_receipt,
    build_dnh_receipt,
    calibrate_dnh_from_reloaded_batch,
    load_calibration_batch,
    load_calibration_batch_bytes,
    publish_calibration_batch_no_replace,
    snapshot_actual_stage2_batch,
)
from deep_anc.train.stage2_2khz_pretrain_admission import (
    _require_rebuilt_model_initial_state_sha256,
    _rebuild_model_initial_state_sha256,
)


FAMILIES = ("speech", "music", "environment", "machine")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _operator(role: str, *, handoff: int) -> CausalFIRPathData:
    fir = np.asarray([1.0, -0.15, 0.05], dtype="<f8")
    return CausalFIRPathData(
        role=role,
        post_onset_fir=fir,
        coarse_delay_samples=0,
        fractional_delay_samples=0.0,
        support_samples=3,
        sample_rate=48_000,
        handoff_extra_samples=handoff,
        operator_file_sha256=_sha(f"{role}-file"),
        operator_internal_sha256=_sha(f"{role}-internal"),
        fir_sha256=hashlib.sha256(fir.tobytes()).hexdigest(),
        authority_sha256=_sha("binding-authority"),
        source_path=f"artifacts/{role}.npz",
    )


def _binding() -> Stage2TwoKilohertzPlantBinding:
    contract = Stage2TwoKilohertzContract.canonical()
    primary = _operator("primary", handoff=0)
    secondary = _operator("secondary", handoff=256)
    timing = TrainingTimingContract.derive(
        primary_fir=primary.post_onset_fir,
        plant_delays=PlantDelays(
            primary_delay_samples=0,
            secondary_delay_samples=0,
            handoff_samples=256,
            sample_rate=48_000,
        ),
    )
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
        verified_physical_subbands_hz=tuple(contract.physical_identification_subbands_hz),
        err_channel_index=0,
        reference_channel_index=0,
        block_size=256,
        binding_file_sha256=_sha("binding-file"),
        source_capture_commit_sha="c" * 40,
        fixture_only=True,
    )


def _sampler() -> Stage2FamilyComponentBatchSampler:
    records = tuple(
        Stage2SamplerRecord(
            dataset_index=index,
            source_family=family,
            component_id=f"{family}-component",
            split="train",
            source_sha256=_sha(f"source-{family}"),
        )
        for index, family in enumerate(FAMILIES)
    )
    return Stage2FamilyComponentBatchSampler(
        records,
        batch_size=4,
        seed=20260803,
        manifest_bundle_sha256=_sha("manifest"),
        sampler_receipt_sha256=_sha("sampler-receipt"),
    )


def _noise(batch: int, samples: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260901)
    spectrum = torch.zeros((batch, samples // 2 + 1), dtype=torch.complex64)
    frequency = torch.fft.rfftfreq(samples, 1.0 / 48_000)
    selected = (frequency >= 88.0) & (frequency <= 12_000.0)
    spectrum[:, selected] = torch.complex(
        torch.randn((batch, int(selected.sum())), generator=generator),
        torch.randn((batch, int(selected.sum())), generator=generator),
    )
    return torch.fft.irfft(spectrum, n=samples, dim=-1)


class _ScaleStreaming(nn.Module):
    hop = 256

    def init_states(self, batch: int = 1, device: torch.device | str = "cpu") -> None:
        del batch, device
        return None

    def streaming_step(
        self, x_block: torch.Tensor, state: None
    ) -> tuple[torch.Tensor, None]:
        return 0.25 * x_block[:, :1], state


def _actual_snapshot():
    binding = _binding()
    sampler = _sampler()
    identity = Stage2ActualBatchIdentity.from_sampler(sampler, global_step=0)
    batch_size = len(identity.source_families)
    prefix = 512
    target = 4096
    lead = binding.training_timing_contract.digital_reference_lead_samples
    clean = _noise(batch_size, prefix + target + lead).unsqueeze(1)
    preview = clean[..., lead : lead + prefix + target]
    zeros = torch.zeros_like(preview)
    model_input = torch.cat((preview, zeros), dim=1)
    causal = CausalPrefixBatchV1(
        x_prefix=model_input[..., :prefix],
        x_target=model_input[..., prefix:],
        source_sha256=identity.source_sha256,
        clean_playback_source_sha256=identity.source_sha256,
        clean_playback_timeline=clean,
        controller_reference_preaugmentation=preview,
        training_timing_contract_sha256=binding.training_timing_contract_sha256,
        segment_prefix_start_samples=(0,) * batch_size,
        segment_target_start_samples=(prefix,) * batch_size,
        global_sample_indices=identity.global_sample_indices,
        state_origin=CausalPrefixStateOriginV1(
            kind="segment_start_zero_state",
            binding_sha256=binding.digest(),
            source_sha256=identity.source_sha256,
        ),
    )
    tensor_batch = Stage2TensorBatch(
        causal=causal,
        dataset_indices=identity.dataset_indices,
        manifest_row_sha256=identity.manifest_row_sha256,
        augmentation_seeds=identity.augmentation_seeds,
    )
    result = Stage2CausalPrefixAdapter._for_test_fixture(binding)(
        _ScaleStreaming(), causal
    )
    snapshot = snapshot_actual_stage2_batch(
        tensor_batch=tensor_batch,
        identity=identity,
        causal_result=result,
        binding=binding,
        model_config_sha256=_sha("model-config"),
        model_initial_state_sha256_value=_sha("model-state"),
    )
    return binding, snapshot


def test_actual_tensor_npz_reload_drives_dnh_calibration_and_receipt(tmp_path: Path) -> None:
    binding, snapshot = _actual_snapshot()
    path = tmp_path / "actual_calibration_batch.npz"
    batch_sha = publish_calibration_batch_no_replace(path, snapshot)
    reloaded = load_calibration_batch(path, expected_sha256=batch_sha)
    calibration = calibrate_dnh_from_reloaded_batch(reloaded, binding=binding)
    assert calibration["selected_target_share"] == DNH_SHARE_TARGET
    assert DNH_SHARE_MIN <= calibration["output_y_gradient_share"] <= DNH_SHARE_MAX
    assert [row["target_share"] for row in calibration["candidate_results"]] == [
        DNH_SHARE_MIN,
        DNH_SHARE_TARGET,
        DNH_SHARE_MAX,
    ]
    receipt = build_dnh_receipt(
        snapshot=reloaded,
        calibration_batch_path="results/actual_calibration_batch.npz",
        calibration_batch_sha256=batch_sha,
        calibration=calibration,
    )
    assert receipt["calibration_batch"] == {
        "path": "results/actual_calibration_batch.npz",
        "sha256": batch_sha,
    }
    assert receipt["actual_causal_secondary_output"] is True
    assert receipt["actual_family_balanced_batch"] is True
    assert receipt["model_config_sha256"] == snapshot.metadata["model_config_sha256"]
    assert receipt["model_initial_state_sha256"] == snapshot.metadata[
        "model_initial_state_sha256"
    ]


def test_dnh_model_state_binding_rebuilds_actual_config_and_rejects_tamper() -> None:
    """receipt scalar가 아니라 actual model construction으로 initial state를 검사한다."""

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs/model_tiny_stage2_2khz.yaml"
    )
    config_bytes = config_path.read_bytes()
    model_config = yaml.safe_load(config_bytes.decode("utf-8"))
    assert isinstance(model_config, dict)
    initial_state_sha = _rebuild_model_initial_state_sha256(
        model_config, seed=20260803
    )
    _require_rebuilt_model_initial_state_sha256(
        model_config,
        seed=20260803,
        expected_sha256=initial_state_sha,
    )

    # immutable NPZ metadata가 criterion/profile config bytes와 같은 state를 주장할 때만
    # admission까지 갈 수 있다. config 하나라도 바꾸면 fresh state digest가 달라진다.
    tampered = dict(model_config)
    tampered["io_scale"] = float(model_config["io_scale"]) * 1.5
    with pytest.raises(ValueError, match="actual config\\+seed"):
        _require_rebuilt_model_initial_state_sha256(
            tampered,
            seed=20260803,
            expected_sha256=initial_state_sha,
        )

    _binding, snapshot = _actual_snapshot()
    bound_metadata = dict(snapshot.metadata)
    bound_metadata["model_config_sha256"] = _sha(config_bytes.decode("utf-8"))
    bound_metadata["model_initial_state_sha256"] = initial_state_sha
    bound_snapshot = Stage2DNHCalibrationSnapshot(
        arrays=snapshot.arrays,
        metadata=bound_metadata,
    )
    calibration = calibrate_dnh_from_reloaded_batch(bound_snapshot, binding=_binding)
    receipt = build_dnh_receipt(
        snapshot=bound_snapshot,
        calibration_batch_path="results/actual_calibration_batch.npz",
        calibration_batch_sha256=_sha("actual-calibration-batch"),
        calibration=calibration,
    )
    assert receipt["model_config_sha256"] == _sha(config_bytes.decode("utf-8"))
    assert receipt["model_initial_state_sha256"] == initial_state_sha


def test_snapshot_bytes_are_independently_reopenable_for_admission(tmp_path: Path) -> None:
    """admission은 path 재-open race가 아닌 nofollow snapshot bytes를 사용한다."""

    _binding, snapshot = _actual_snapshot()
    path = tmp_path / "actual_calibration_batch.npz"
    digest = publish_calibration_batch_no_replace(path, snapshot)
    reloaded = load_calibration_batch_bytes(path.read_bytes(), expected_sha256=digest)

    assert reloaded.metadata == snapshot.metadata
    assert set(reloaded.arrays) == set(snapshot.arrays)


def test_issue_cli_default_output_root_is_gitignored_results() -> None:
    """default issue는 raw NPZ를 tracked working tree에 만들면 안 된다."""

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/train/issue_stage2_2khz_pretrain_contract.py"
    )
    spec = importlib.util.spec_from_file_location("stage2_pretrain_issuer_cli_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    parsed = module._parser().parse_args(["issue", "--generation", "unit-test"])

    assert parsed.output_root == "results/stage2_2khz_pretrain_contracts"


def test_calibration_artifact_is_no_replace_and_tamper_fails(tmp_path: Path) -> None:
    _, snapshot = _actual_snapshot()
    path = tmp_path / "actual_calibration_batch.npz"
    digest = publish_calibration_batch_no_replace(path, snapshot)
    with pytest.raises(FileExistsError):
        publish_calibration_batch_no_replace(path, snapshot)
    content = bytearray(path.read_bytes())
    content[-9] ^= 0x01
    path.write_bytes(content)
    with pytest.raises(ValueError, match="file SHA"):
        load_calibration_batch(path, expected_sha256=digest)


def test_external_contract_builder_is_scratch_and_exact() -> None:
    profile = {
        key: _sha(key)
        for key in (
            "duct",
            "data",
            "evaluation",
            "canonical_pretrain",
            "canonical_finetune",
        )
    }
    plant = {
        "primary_path_sha256": _sha("P"),
        "secondary_path_sha256": _sha("S"),
        "plant_binding_sha256": _sha("binding"),
    }
    payload = build_canonical_pretrain_external_contract(
        artifact_source_commit_sha="a" * 40,
        profile_sha256=profile,
        plant_sha256=plant,
        manifest_bundle_sha256=_sha("manifest"),
        criterion_receipt_sha256=_sha("criterion"),
    )
    assert payload["initialization_mode"] == "scratch"
    assert payload["init_checkpoint_sha256"] is None
    assert payload["legacy_artifacts_allowed"] is False
    assert payload["automatic_resume_allowed"] is False


def test_criterion_receipt_is_exactly_compatible_with_current_admission_schema(
    tmp_path: Path,
) -> None:
    implementation_specs = (
        "src/deep_anc/losses/stage2_2khz_loss.py",
        "src/deep_anc/train/stage2_2khz_execution.py",
        "src/deep_anc/train/stage2_2khz_runner.py",
    )
    for relative in implementation_specs:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"fixture bytes for {relative}\n", encoding="utf-8")
    payload = build_criterion_receipt(
        repository_root=tmp_path,
        plant_binding_file_sha256=_sha("binding-file"),
        manifest_bundle_sha256=_sha("manifest"),
        sampler_receipt_path="artifacts/sampler.json",
        sampler_receipt_sha256=_sha("sampler"),
        dnh_receipt_path="artifacts/dnh.json",
        dnh_receipt_sha256=_sha("dnh"),
        model_config_path="configs/model_tiny_stage2_2khz.yaml",
        model_config_sha256=_sha("model-config"),
        model_initial_state_sha256_value=_sha("model-initial-state"),
        batch_size=96,
        seed=20260803,
    )
    # load_stage2_pretrain_typed_admission의 exact key 집합과 동기화한다.
    assert set(payload) == {
        "schema",
        "status",
        "canonical_pretrain_eligible",
        "control_band_contract_sha256",
        "plant_binding_file_sha256",
        "manifest_bundle_sha256",
        "loss_implementation",
        "trainer_adapter_implementation",
        "scratch_runner_implementation",
        "sampler_receipt",
        "dnh_calibration_receipt",
        "model_config",
        "model_initial_state_sha256",
        "batch_size",
        "seed",
        "generic_stage1_loss_used",
        "full_octave_v3_loss_used",
    }
    assert payload["batch_size"] == 96
    assert payload["seed"] == 20260803
    assert payload["generic_stage1_loss_used"] is False
    assert payload["full_octave_v3_loss_used"] is False
    assert payload["model_config"] == {
        "path": "configs/model_tiny_stage2_2khz.yaml",
        "sha256": _sha("model-config"),
    }
    assert payload["model_initial_state_sha256"] == _sha("model-initial-state")
    for key in (
        "loss_implementation",
        "trainer_adapter_implementation",
        "scratch_runner_implementation",
    ):
        assert set(payload[key]) == {"path", "sha256", "schema"}


def test_cli_dry_run_uses_no_audio_gpu_network_and_writes_nothing(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    del tmp_path
    output_relative = "results/stage2-pretrain-issuer-dry-run-must-not-exist"
    output = repository_root / output_relative
    assert not output.exists()
    completed = subprocess.run(
        [
            sys.executable,
            str(
                repository_root
                / "scripts/train/issue_stage2_2khz_pretrain_contract.py"
            ),
            "--repository-root",
            str(repository_root),
            "issue",
            "--generation",
            "dry-run-test",
            "--output-root",
            output_relative,
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["dry_run"] is True
    assert payload["audio_used"] is False
    assert payload["gpu_used"] is False
    assert payload["network_used"] is False
    assert not output.exists()


def test_external_runtime_snapshot_accepts_gitignored_artifact_bytes(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script_path = (
        repository_root / "scripts/train/issue_stage2_2khz_pretrain_contract.py"
    )
    spec = importlib.util.spec_from_file_location("stage2_pretrain_issue_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    (tmp_path / ".gitignore").write_text("/runtime/\n", encoding="utf-8")
    artifact = tmp_path / "runtime/manifest.json"
    artifact.parent.mkdir(parents=True)
    content = b'{"actual":"manifest-bytes"}\n'
    artifact.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    assert module._snapshot_runtime_ref(
        tmp_path,
        ("runtime/manifest.json", digest),
        label="ignored manifest regression",
    ) == digest
    with pytest.raises(ValueError, match="SHA"):
        module._snapshot_runtime_ref(
            tmp_path,
            ("runtime/manifest.json", "0" * 64),
            label="ignored manifest regression",
        )
