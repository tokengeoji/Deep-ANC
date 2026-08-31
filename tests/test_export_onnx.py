from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import deep_anc.eval.broadband_runtime as runtime
from deep_anc.dsp.control_band_contract import ControlBandContract
from deep_anc.dsp.timing import PlantDelays, TrainingTimingContract
from deep_anc.eval.broadband_runtime import (
    RUNTIME_DEPLOYMENT_METADATA_SCHEMA,
    verify_runtime_deployment_identity,
)
from deep_anc.realtime.clock_telemetry import sha256_file
from deep_anc.train.experiment_contract import stamp_experiment_contract


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train/export_onnx.py"


def _load_exporter():
    name = "deep_anc_export_onnx_transaction_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stage_pair(root: Path) -> tuple[Path, Path]:
    artifact = root / ".model.onnx.fixture.partial"
    metadata = root / ".model.json.fixture.partial"
    artifact.write_bytes(b"verified-onnx-bytes")
    metadata.write_bytes(b'{"verified":true}\n')
    return artifact, metadata


def test_export_pair_publishes_both_leaves_without_replace(tmp_path):
    exporter = _load_exporter()
    staged_artifact, staged_metadata = _stage_pair(tmp_path)
    artifact = tmp_path / "model.onnx"
    metadata = tmp_path / "model.json"

    state = exporter._publish_export_pair(
        staged_artifact=staged_artifact,
        staged_metadata=staged_metadata,
        artifact=artifact,
        metadata=metadata,
    )

    assert state == "published"
    assert artifact.read_bytes() == staged_artifact.read_bytes()
    assert metadata.read_bytes() == staged_metadata.read_bytes()
    with pytest.raises(FileExistsError, match="이미 모두 존재"):
        exporter._publish_export_pair(
            staged_artifact=staged_artifact,
            staged_metadata=staged_metadata,
            artifact=artifact,
            metadata=metadata,
        )


def test_exact_artifact_only_crash_residue_recovers_metadata(tmp_path):
    exporter = _load_exporter()
    staged_artifact, staged_metadata = _stage_pair(tmp_path)
    artifact = tmp_path / "model.onnx"
    metadata = tmp_path / "model.json"
    artifact.write_bytes(staged_artifact.read_bytes())

    state = exporter._publish_export_pair(
        staged_artifact=staged_artifact,
        staged_metadata=staged_metadata,
        artifact=artifact,
        metadata=metadata,
    )

    assert state == "recovered"
    assert artifact.read_bytes() == b"verified-onnx-bytes"
    assert metadata.read_bytes() == b'{"verified":true}\n'


def test_different_artifact_orphan_is_preserved_and_rejected(tmp_path):
    exporter = _load_exporter()
    staged_artifact, staged_metadata = _stage_pair(tmp_path)
    artifact = tmp_path / "model.onnx"
    metadata = tmp_path / "model.json"
    artifact.write_bytes(b"unrelated-existing-artifact")

    with pytest.raises(RuntimeError, match="orphan ONNX artifact"):
        exporter._publish_export_pair(
            staged_artifact=staged_artifact,
            staged_metadata=staged_metadata,
            artifact=artifact,
            metadata=metadata,
        )

    assert artifact.read_bytes() == b"unrelated-existing-artifact"
    assert not metadata.exists()


def test_second_leaf_failure_rolls_back_only_created_artifact(
    tmp_path, monkeypatch
):
    exporter = _load_exporter()
    staged_artifact, staged_metadata = _stage_pair(tmp_path)
    artifact = tmp_path / "model.onnx"
    metadata = tmp_path / "model.json"
    original = exporter._publish_one_noreplace

    def fail_metadata(staged, final, **kwargs):
        if final == metadata:
            raise RuntimeError("fixture metadata publication failure")
        return original(staged, final, **kwargs)

    monkeypatch.setattr(exporter, "_publish_one_noreplace", fail_metadata)
    with pytest.raises(RuntimeError, match="metadata publication failure"):
        exporter._publish_export_pair(
            staged_artifact=staged_artifact,
            staged_metadata=staged_metadata,
            artifact=artifact,
            metadata=metadata,
        )

    assert not artifact.exists()
    assert not metadata.exists()
    assert staged_artifact.exists() and staged_metadata.exists()


def test_losing_concurrent_publisher_does_not_remove_winner_pair(
    tmp_path, monkeypatch
):
    exporter = _load_exporter()
    staged_artifact, staged_metadata = _stage_pair(tmp_path)
    winner_artifact = tmp_path / ".winner-model.onnx.partial"
    winner_metadata = tmp_path / ".winner-model.json.partial"
    winner_artifact.write_bytes(staged_artifact.read_bytes())
    winner_metadata.write_bytes(b'{"winner":true}\n')
    artifact = tmp_path / "model.onnx"
    metadata = tmp_path / "model.json"
    original = exporter._publish_one_noreplace
    injected = False

    def publish_winner_before_loser_metadata(staged, final, **kwargs):
        nonlocal injected
        if final == metadata and not injected:
            injected = True
            assert exporter._publish_export_pair(
                staged_artifact=winner_artifact,
                staged_metadata=winner_metadata,
                artifact=artifact,
                metadata=metadata,
            ) == "recovered"
        return original(staged, final, **kwargs)

    monkeypatch.setattr(
        exporter, "_publish_one_noreplace", publish_winner_before_loser_metadata
    )
    with pytest.raises(RuntimeError, match="기존 ONNX metadata"):
        exporter._publish_export_pair(
            staged_artifact=staged_artifact,
            staged_metadata=staged_metadata,
            artifact=artifact,
            metadata=metadata,
        )

    assert artifact.read_bytes() == winner_artifact.read_bytes()
    assert metadata.read_bytes() == winner_metadata.read_bytes()


def test_checkpoint_snapshot_stays_bound_when_path_is_atomically_replaced(
    tmp_path, monkeypatch
):
    exporter = _load_exporter()
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint-A-exact-bytes")
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"checkpoint-B-different")
    expected_sha = exporter._sha256(checkpoint)
    loaded = {"cfg": {}, "model": {}}

    def replace_path_while_loading(handle, **_kwargs):
        assert handle.read() == b"checkpoint-A-exact-bytes"
        replacement.replace(checkpoint)
        return loaded

    monkeypatch.setattr(exporter.torch, "load", replace_path_while_loading)
    state, checkpoint_sha, checkpoint_size = exporter._load_checkpoint_snapshot(
        checkpoint
    )

    assert state is loaded
    assert checkpoint_sha == expected_sha
    assert checkpoint_size == len(b"checkpoint-A-exact-bytes")
    with pytest.raises(RuntimeError, match="현재 checkpoint bytes"):
        exporter._require_exact_regular(
            checkpoint,
            expected_size=checkpoint_size,
            expected_sha256=checkpoint_sha,
            label="현재 checkpoint",
        )


def test_ort_failure_leaves_no_final_or_staging_orphan(tmp_path, monkeypatch):
    exporter = _load_exporter()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-fixture")
    output = tmp_path / "export/model.onnx"
    timing = TrainingTimingContract.derive(
        primary_fir=[1.0],
        plant_delays=PlantDelays(
            primary_delay_samples=1_000,
            secondary_delay_samples=859,
            handoff_samples=256,
            sample_rate=48_000,
        ),
    )
    state = {
        "cfg": {
            "model": {"name": "hybrid_anc_tiny"},
            "data": {"training_timing_contract": timing.model_dump()},
            "control_band_contract_sha256": "a" * 64,
        },
        "model": {},
    }

    class FakeModel:
        in_channels = 2
        hop = 128
        win = 256

        def load_state_dict(self, _state):
            return None

        def eval(self):
            return self

        def init_states(self, _batch, _device):
            return []

    class SessionOptions:
        intra_op_num_threads = 0
        inter_op_num_threads = 0

    def fake_export(_wrapper, _inputs, handle, **_kwargs):
        handle.write(b"staged-onnx-before-ort-failure")

    monkeypatch.setattr(exporter.torch, "load", lambda *_args, **_kwargs: state)
    monkeypatch.setattr(exporter, "build_model", lambda _cfg: FakeModel())
    monkeypatch.setattr(exporter, "ExportWrapper", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(exporter, "state_names", lambda _model: [])
    monkeypatch.setattr(exporter, "flatten_states", lambda _states: [])
    monkeypatch.setattr(exporter.torch.onnx, "export", fake_export)
    monkeypatch.setattr(
        exporter,
        "validate_embedded_experiment_contract",
        lambda _cfg: {"sha256": "b" * 64, "artifacts": {}},
    )
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            SessionOptions=SessionOptions,
            InferenceSession=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("fixture ORT open failure")
            ),
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--ckpt",
            str(checkpoint),
            "--out",
            str(output),
        ],
    )

    with pytest.raises(RuntimeError, match="ORT open failure"):
        exporter.main()

    assert not output.exists()
    assert not output.with_suffix(".json").exists()
    assert not list(output.parent.glob("*.partial"))


def _runtime_identity_fixture(root: Path):
    primary = root / "primary.npz"
    secondary = root / "secondary.npz"
    primary.write_bytes(b"strict-primary-fixture")
    secondary.write_bytes(b"strict-secondary-fixture")
    delays = PlantDelays(
        primary_delay_samples=1_000,
        secondary_delay_samples=859,
        handoff_samples=256,
        sample_rate=48_000,
    )
    timing = TrainingTimingContract.derive(
        primary_fir=[1.0], plant_delays=delays
    )
    contract = ControlBandContract.broadband_point_control()
    cfg = {
        "model": {"name": "hybrid_anc_tiny"},
        "data": {
            "digital_reference_lead_samples": 115,
            "training_timing_contract": timing.model_dump(),
        },
        "duct": {
            "secondary_path": {"npz": str(secondary)},
            "digital_reference": {"primary_path_npz": str(primary)},
        },
        "control_band_contract_sha256": contract.digest(),
    }
    stamped = stamp_experiment_contract(cfg, repo_root=root)
    checkpoint = root / "model.pt"
    torch.save({"cfg": stamped}, checkpoint)
    artifact = root / "model.onnx"
    artifact.write_bytes(b"canonical-onnx-fixture")
    metadata = artifact.with_suffix(".json")
    metadata_payload = {
        "schema_version": RUNTIME_DEPLOYMENT_METADATA_SCHEMA,
        "model_name": "hybrid_anc_tiny",
        "engine": "ort",
        "experiment_contract_sha256": stamped["experiment_contract_sha256"],
        "control_band_contract_sha256": contract.digest(),
        "training_timing_contract_sha256": timing.digest(),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "digital_reference_lead_samples": 115,
        "deployment_artifact_path": str(artifact.resolve()),
        "deployment_artifact_sha256": sha256_file(artifact),
        "primary_path_sha256": sha256_file(primary),
        "secondary_path_sha256": sha256_file(secondary),
    }
    metadata.write_text(
        json.dumps(metadata_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    runtime_cfg = {
        "controller": "dl",
        "engine": {
            "type": "ort",
            "ckpt": str(checkpoint),
            "onnx": str(artifact),
        },
    }
    plant = SimpleNamespace(
        primary_path_sha256=sha256_file(primary),
        secondary_path_sha256=sha256_file(secondary),
    )
    return contract, runtime_cfg, plant, checkpoint, artifact, metadata


def test_runtime_deployment_identity_reopens_actual_checkpoint_and_sidecar(tmp_path):
    contract, runtime_cfg, plant, checkpoint, artifact, metadata = (
        _runtime_identity_fixture(tmp_path)
    )

    identity = verify_runtime_deployment_identity(
        contract=contract,
        runtime_cfg=runtime_cfg,
        plant=plant,
        repo_root=tmp_path,
    )

    assert identity.checkpoint_sha256 == sha256_file(checkpoint)
    assert identity.deployment_artifact_sha256 == sha256_file(artifact)
    assert identity.deployment_metadata_sha256 == sha256_file(metadata)
    assert identity.checkpoint_lead_samples == identity.deployment_lead_samples == 115


def test_runtime_deployment_identity_rejects_actual_artifact_tamper(tmp_path):
    contract, runtime_cfg, plant, _checkpoint, artifact, _metadata = (
        _runtime_identity_fixture(tmp_path)
    )
    artifact.write_bytes(b"tampered-after-export")

    with pytest.raises(ValueError, match="deployment_artifact_sha256"):
        verify_runtime_deployment_identity(
            contract=contract,
            runtime_cfg=runtime_cfg,
            plant=plant,
            repo_root=tmp_path,
        )


def test_runtime_identity_rejects_checkpoint_path_swap_during_load(
    tmp_path, monkeypatch
):
    contract, runtime_cfg, plant, checkpoint, _artifact, _metadata = (
        _runtime_identity_fixture(tmp_path)
    )
    original_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    replacement = tmp_path / "replacement.pt"
    torch.save({**original_state, "replacement": True}, replacement)
    real_load = torch.load

    def replace_after_held_load(handle, **kwargs):
        value = real_load(handle, **kwargs)
        replacement.replace(checkpoint)
        return value

    monkeypatch.setattr(torch, "load", replace_after_held_load)
    with pytest.raises(ValueError, match="runtime checkpoint"):
        verify_runtime_deployment_identity(
            contract=contract,
            runtime_cfg=runtime_cfg,
            plant=plant,
            repo_root=tmp_path,
        )


def test_deployment_snapshot_rejects_artifact_path_swap_while_hashing(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "model.pt"
    artifact = tmp_path / "model.onnx"
    metadata = tmp_path / "model.json"
    checkpoint.write_bytes(b"checkpoint")
    artifact.write_bytes(b"artifact-A")
    metadata.write_bytes(b"metadata")
    replacement = tmp_path / "artifact-B.onnx"
    replacement.write_bytes(b"artifact-B")
    runtime_cfg = {
        "controller": "dl",
        "digital_reference_lead_samples": 115,
        "engine": {
            "type": "ort",
            "ckpt": str(checkpoint),
            "onnx": str(artifact),
        },
    }
    plant = SimpleNamespace(
        timing=SimpleNamespace(digital_reference_lead_samples=115),
        primary_path_sha256="a" * 64,
        secondary_path_sha256="b" * 64,
    )
    original_hash = runtime._sha256_handle
    calls = 0

    def replace_artifact_after_hash(handle):
        nonlocal calls
        digest = original_hash(handle)
        calls += 1
        if calls == 2:
            replacement.replace(artifact)
        return digest

    monkeypatch.setattr(runtime, "_sha256_handle", replace_artifact_after_hash)
    with pytest.raises(ValueError, match="deployment_artifact path.*교체"):
        runtime.snapshot_runtime_deployment_files(
            runtime_cfg=runtime_cfg,
            plant=plant,
            repo_root=tmp_path,
        )
