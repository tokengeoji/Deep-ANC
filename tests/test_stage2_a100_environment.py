from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from deep_anc.train.stage2_a100_environment import (
    STAGE2_A100_ENVIRONMENT_SCHEMA,
    STAGE2_A100_MIN_VISIBLE_MEMORY_BYTES,
    configure_and_collect_stage2_a100_environment,
    validate_stage2_a100_environment_payload,
)


def _valid_payload() -> dict:
    return {
        "schema": STAGE2_A100_ENVIRONMENT_SCHEMA,
        "torch_version": "2.5.1+cu121",
        "torch_cuda_version": "12.1",
        "cuda_available": True,
        "cuda_initialized_before_configuration": False,
        "cuda_initialized_after_probe": True,
        "device_count": 1,
        "current_device": 0,
        "device_name": "NVIDIA A100 80GB PCIe",
        "visible_total_memory_bytes": STAGE2_A100_MIN_VISIBLE_MEMORY_BYTES + 1,
        "bf16_supported": True,
        "world_size": 1,
        "rank": 0,
        "local_rank": 0,
        "cuda_visible_devices": "0",
        "nvidia_smi_l_sha256": "a" * 64,
        "mig_detected": False,
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cudnn_allow_tf32": False,
        "matmul_allow_tf32": False,
    }


@pytest.mark.parametrize(
    ("name", "memory", "mig"),
    [
        ("NVIDIA A100-PCIE-40GB", 40 * 1024**3, False),
        ("NVIDIA A100 80GB PCIe MIG 4g.40gb", 40 * 1024**3, True),
    ],
)
def test_stage2_environment_rejects_40gb_and_mig(
    name: str, memory: int, mig: bool
) -> None:
    payload = _valid_payload()
    payload["device_name"] = name
    payload["visible_total_memory_bytes"] = memory
    payload["mig_detected"] = mig
    with pytest.raises(ValueError, match="A100 80GB|visible memory|MIG"):
        validate_stage2_a100_environment_payload(payload)


class _FakeCuda:
    def __init__(self) -> None:
        self.initialized = False
        self.available_calls = 0

    def is_initialized(self) -> bool:
        return self.initialized

    def is_available(self) -> bool:
        self.available_calls += 1
        self.initialized = True
        return True

    def device_count(self) -> int:
        return 1

    def current_device(self) -> int:
        return 0

    def get_device_properties(self, _index: int) -> SimpleNamespace:
        return SimpleNamespace(
            name="NVIDIA A100 80GB PCIe",
            total_memory=STAGE2_A100_MIN_VISIBLE_MEMORY_BYTES + 1,
        )

    def is_bf16_supported(self) -> bool:
        return True


class _FakeTorch:
    __version__ = "2.5.1+cu121"
    version = SimpleNamespace(cuda="12.1")

    def __init__(self) -> None:
        self.cuda = _FakeCuda()
        self.backends = SimpleNamespace(
            cudnn=SimpleNamespace(
                deterministic=False, benchmark=True, allow_tf32=True
            ),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        )
        self.deterministic = False
        self.warn_only = False

    def use_deterministic_algorithms(self, enabled: bool, *, warn_only: bool) -> None:
        self.deterministic = bool(enabled)
        self.warn_only = bool(warn_only)

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.deterministic

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return self.warn_only


def test_stage2_environment_sets_backend_flags_then_rechecks_visible_device() -> None:
    fake = _FakeTorch()
    receipt = configure_and_collect_stage2_a100_environment(
        fake,
        nvidia_smi_l_output=(
            "GPU 0: NVIDIA A100 80GB PCIe (UUID: GPU-fixture)\n"
        ),
        environ={
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
        },
    )
    assert receipt["device_count"] == 1
    assert receipt["visible_total_memory_bytes"] > 79 * 1024**3
    assert receipt["deterministic_algorithms"] is True
    assert receipt["cudnn_benchmark"] is False
    assert receipt["cudnn_allow_tf32"] is False


def test_stage2_environment_requires_cublas_before_any_cuda_probe() -> None:
    fake = _FakeTorch()
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        configure_and_collect_stage2_a100_environment(
            fake,
            nvidia_smi_l_output="GPU 0: NVIDIA A100 80GB PCIe\n",
            environ={"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"},
        )
    assert fake.cuda.available_calls == 0
    assert fake.cuda.is_initialized() is False


def _load_existing_instance_audit():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "stage2_existing_instance_audit_test",
        root / "scripts/elice/audit_stage2_2khz_existing_instance.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_instance_missing_typed_artifacts_stops_before_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_existing_instance_audit()
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text("schema: fixture\n", encoding="utf-8")
    expected = "a" * 40

    def fake_run(*args: str) -> str:
        if args[-2:] == ("rev-parse", "HEAD"):
            return expected
        if "status" in args:
            return ""
        raise AssertionError("typed BLOCKED 이전에 nvidia-smi를 호출하면 안 됩니다")

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=256 * 1024**3, used=64 * 1024**3, free=192 * 1024**3
        ),
    )
    monkeypatch.setattr(
        module,
        "audit_stage2_2khz_campaign",
        lambda *_args, **_kwargs: {"status": "BLOCKED"},
    )
    monkeypatch.setattr(
        module,
        "_environment_probe",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("typed BLOCKED 이전에 torch/CUDA를 조회하면 안 됩니다")
        ),
    )
    with pytest.raises(RuntimeError, match="READY_PRETRAIN이 아닙니다"):
        module.audit_existing_instance(
            expected_commit=expected,
            campaign_path=campaign,
            resume_path=None,
            minimum_free_gib=16,
        )


def test_existing_instance_dirty_checkout_stops_before_campaign_or_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_existing_instance_audit()
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text("schema: fixture\n", encoding="utf-8")
    expected = "b" * 40

    def fake_run(*args: str) -> str:
        if args[-2:] == ("rev-parse", "HEAD"):
            return expected
        if "status" in args:
            return "?? untracked-secret.pem"
        raise AssertionError("dirty checkout 뒤 명령을 더 실행하면 안 됩니다")

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module,
        "audit_stage2_2khz_campaign",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dirty checkout 뒤 campaign scan을 하면 안 됩니다")
        ),
    )
    monkeypatch.setattr(
        module,
        "_environment_probe",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("dirty checkout 뒤 CUDA를 조회하면 안 됩니다")
        ),
    )
    with pytest.raises(RuntimeError, match="dirty"):
        module.audit_existing_instance(
            expected_commit=expected,
            campaign_path=campaign,
            resume_path=None,
            minimum_free_gib=16,
        )
