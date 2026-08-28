"""실시간 ANC가 오디오 장치보다 먼저 strict P/S를 검증하는 회귀 테스트."""

from __future__ import annotations

import builtins

import pytest

from deep_anc.config import load_runtime_config
from deep_anc.realtime.plant_contract import (
    RuntimePlantContractError,
    validate_runtime_plant_contract,
)
from deep_anc.realtime.run_realtime import RealtimeANC


def _strict_runtime_cfg() -> dict:
    return load_runtime_config(
        "configs/runtime_tiny.yaml",
        ["digital_reference_lead_samples=115"],
    )


def test_runtime_strict_plant_contract_rejects_the_legacy_109_sample_lead():
    """legacy ONNX와만 맞는 109는 실제 strict P/S(115)에서 즉시 막는다."""

    cfg = load_runtime_config("configs/runtime_tiny.yaml")
    with pytest.raises(RuntimePlantContractError, match="runtime=109, derived=115"):
        validate_runtime_plant_contract(cfg)


def test_runtime_strict_plant_contract_accepts_the_actual_115_sample_contract():
    """현재 strict P/S raw/analysis/level evidence에서 유도한 115만 통과한다."""

    contract = validate_runtime_plant_contract(_strict_runtime_cfg())

    assert contract is not None
    assert contract.capture_id == "5ac1313488c8434bb4d672a36503df59"
    assert contract.timing.digital_reference_lead_samples == 115
    assert contract.timing.primary_zeros_before_fir_samples == 1386
    assert contract.timing.secondary_delay_samples == 1245
    assert contract.timing.handoff_samples == 256


def test_runtime_strict_plant_contract_rejects_a_primary_secondary_channel_swap():
    """같은 NPZ를 P에 끼워 넣는 식의 경로/채널 위조도 시작 전 거부한다."""

    cfg = _strict_runtime_cfg()
    cfg["duct"]["digital_reference"]["primary_path_npz"] = cfg["duct"][
        "secondary_path"
    ]["npz"]

    with pytest.raises(RuntimePlantContractError, match="output_channel"):
        validate_runtime_plant_contract(cfg)


def test_realtime_constructor_checks_the_strict_plant_before_sounddevice_import(monkeypatch):
    """lead가 틀린 legacy config는 입력 probe나 PortAudio import 이전에 중단한다."""

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise AssertionError("strict plant 검증 전에 sounddevice를 import하면 안 됩니다")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    cfg = load_runtime_config("configs/runtime_tiny.yaml")
    with pytest.raises(RuntimePlantContractError, match="runtime=109, derived=115"):
        RealtimeANC(cfg)


def test_realtime_constructor_rejects_legacy_engine_lead_before_sounddevice_import(
    monkeypatch,
):
    """runtime YAML만 115로 덮어도 ONNX의 실제 109 lead가 먼저 거부된다."""

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise AssertionError("engine metadata preflight 전에 sounddevice를 import하면 안 됩니다")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="runtime=115, checkpoint=109"):
        RealtimeANC(_strict_runtime_cfg())
