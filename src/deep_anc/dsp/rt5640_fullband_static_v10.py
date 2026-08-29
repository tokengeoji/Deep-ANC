"""RT5640 기반 full-octave P/S의 무음 static 계약.

이 모듈은 config bytes와 v3 control-band 계약만 검사한다. ALSA, sounddevice,
오디오 장치, 결과 파일을 열지 않는다. 따라서 여기서 반환하는 ``static_gate_pass``는
S32 stream, J511 반대편 앰프 연결, electrical witness, P/S, ANC 감쇠 또는 학습
적격성을 뜻하지 않는다.

v6의 USB/S16 live authority는 hardware path와 actual PCM scale까지 봉인돼 있다.
그것을 APE/RT5640 S32 path에 재표기하면 Q15 ``int16`` 값을 단순 cast했을 때 약
96.3 dB 작아지는 오류가 생긴다. 이 파일은 그 혼동을 config 단계에서 막는다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .control_band_contract import BroadbandFullOctaveContractV3


SCHEMA = "rt5640_fullband_static_v10"
GENERATION = "rt5640_fullband_v3_v10"
DEFAULT_CONFIG_RELATIVE_PATH = "configs/hardware_jetson_rt5640_fullband_v10.yaml"
RESULT_DIRECTORY = "results/rt5640_fullband_v10"
ALLOWED_LIVE_JACK_STATES = ("HP", "HS")
Q15_TO_S32_LEFT_SHIFT = 16

_AUDIO_KEYS = frozenset({"sample_rate", "block_size", "latency", "input", "output"})
_PORT_KEYS = frozenset({"card", "pcm", "channels", "dtype", "route"})
_TOP_LEVEL_KEYS = frozenset(
    {"schema", "audio", "channels", "fullband_v3", "maximum_authority"}
)
_FULLBAND_KEYS = frozenset(
    {
        "generation",
        "control_band_contract_id",
        "control_band_contract_sha256",
        "excitation_lower_hz",
        "excitation_upper_hz",
        "q15_to_s32_left_shift",
        "source_signal_plan",
        "live_jack_allowed_states",
        "result_directory",
        "legacy_v6_relabel_allowed",
    }
)
_AUTHORITY_KEYS = frozenset(
    {
        "static_contract_only",
        "j511_connection_observed",
        "s32_duplex_transport_pass",
        "hardware_frame_identity_pass",
        "electrical_witness_pass",
        "fullband_plant_identification_pass",
        "canonical_training_eligible",
        "deployment_eligible",
    }
)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}는 mapping이어야 합니다")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], *, expected: frozenset[str], label: str
) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{label} key가 exact하지 않습니다: missing={missing}, extra={extra}")


def _require_exact(value: Any, expected: Any, *, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise ValueError(f"{label}가 계약과 다릅니다: expected={expected!r}, got={value!r}")


def _validate_port(value: Any, *, expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    port = _require_mapping(value, label=label)
    _require_exact_keys(port, expected=_PORT_KEYS, label=label)
    for key, expected_value in expected.items():
        _require_exact(port[key], expected_value, label=f"{label}.{key}")
    return dict(port)


def validate_rt5640_fullband_static_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """v10 RT5640 fullband live 이전의 immutable static boundary를 검사한다.

    반환값의 ``status``는 의도적으로 ``BLOCKED``다. 이 gate는 live path의 표본
    식별성을 검증하지 않으며, 현재는 S32 actual PCM plan/duplex backend/electrical
    witness가 아직 없다.
    """

    root = _require_mapping(payload, label="RT5640 fullband config")
    _require_exact_keys(root, expected=_TOP_LEVEL_KEYS, label="RT5640 fullband config")
    _require_exact(root["schema"], SCHEMA, label="schema")

    audio = _require_mapping(root["audio"], label="audio")
    _require_exact_keys(audio, expected=_AUDIO_KEYS, label="audio")
    _require_exact(audio["sample_rate"], 48_000, label="audio.sample_rate")
    _require_exact(audio["block_size"], 256, label="audio.block_size")
    _require_exact(audio["latency"], "low", label="audio.latency")
    input_port = _validate_port(
        audio["input"],
        expected={
            "card": "APE",
            "pcm": 1,
            "channels": 2,
            "dtype": "int32",
            "route": "I2S2_ADMAIF2_ERR_REF",
        },
        label="audio.input",
    )
    output_port = _validate_port(
        audio["output"],
        expected={
            "card": "APE",
            "pcm": 0,
            "channels": 2,
            "dtype": "int32",
            "route": "ADMAIF1_I2S1_RT5640_J511",
        },
        label="audio.output",
    )

    channels = _require_mapping(root["channels"], label="channels")
    expected_channels = {
        "error_mic": 0,
        "reference_mic": 1,
        "noise_out": 0,
        "cancel_out": 1,
    }
    _require_exact_keys(channels, expected=frozenset(expected_channels), label="channels")
    for key, expected_value in expected_channels.items():
        _require_exact(channels[key], expected_value, label=f"channels.{key}")

    contract = BroadbandFullOctaveContractV3.canonical()
    fullband = _require_mapping(root["fullband_v3"], label="fullband_v3")
    _require_exact_keys(fullband, expected=_FULLBAND_KEYS, label="fullband_v3")
    required_fullband = {
        "generation": GENERATION,
        "control_band_contract_id": contract.contract_id,
        "control_band_contract_sha256": contract.digest(),
        "excitation_lower_hz": 80.0,
        "excitation_upper_hz": 11_313.7084989848,
        "q15_to_s32_left_shift": Q15_TO_S32_LEFT_SHIFT,
        "source_signal_plan": "fullband_causal_v5_q15_v3_signal_only",
        "live_jack_allowed_states": list(ALLOWED_LIVE_JACK_STATES),
        "result_directory": RESULT_DIRECTORY,
        "legacy_v6_relabel_allowed": False,
    }
    for key, expected_value in required_fullband.items():
        _require_exact(fullband[key], expected_value, label=f"fullband_v3.{key}")

    maximum = _require_mapping(root["maximum_authority"], label="maximum_authority")
    _require_exact_keys(maximum, expected=_AUTHORITY_KEYS, label="maximum_authority")
    expected_maximum = {
        "static_contract_only": True,
        "j511_connection_observed": False,
        "s32_duplex_transport_pass": False,
        "hardware_frame_identity_pass": False,
        "electrical_witness_pass": False,
        "fullband_plant_identification_pass": False,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
    }
    for key, expected_value in expected_maximum.items():
        _require_exact(maximum[key], expected_value, label=f"maximum_authority.{key}")

    return {
        "schema": "rt5640_fullband_static_receipt_v10",
        "status": "BLOCKED",
        "static_gate_pass": True,
        "audio_opened": False,
        "control_band_contract": contract.model_dump(mode="json"),
        "control_band_contract_sha256": contract.digest(),
        "hardware_audio": {
            "sample_rate": 48_000,
            "block_size": 256,
            "latency": "low",
            "input": input_port,
            "output": output_port,
            "channels": dict(channels),
        },
        "s32_signal_scale": {
            "source": "Q15 actual-int16 signal-only plan",
            "conversion": "exact_signed_left_shift",
            "left_shift_bits": Q15_TO_S32_LEFT_SHIFT,
            "normalized_full_scale_preserved": True,
            "simple_int16_to_int32_cast_allowed": False,
        },
        "live_jack_allowed_states": list(ALLOWED_LIVE_JACK_STATES),
        "next_required_gates": [
            "actual_s32_pcm_signal_plan_and_duplex_transport",
            "j511_connection_state_HP_or_HS_at_live_preflight",
            "independent_electrical_or_fixed_LTI_conditional_clock_witness",
            "fresh_level_meter_and_fullband_P_S_raw",
            "v3_manifest_and_training_contract_rebinding",
        ],
        "authority": dict(expected_maximum),
        "config_payload_sha256": _payload_sha256(dict(root)),
    }


def load_rt5640_fullband_static_contract(path: str | Path) -> dict[str, Any]:
    """YAML을 읽어 static receipt를 반환한다. 파일은 수정하지 않는다."""

    config_path = Path(path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("RT5640 fullband config YAML root는 mapping이어야 합니다")
    receipt = validate_rt5640_fullband_static_contract(loaded)
    raw = config_path.read_bytes()
    return {
        **receipt,
        "config": {
            "path": str(config_path),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
        },
    }


__all__ = [
    "ALLOWED_LIVE_JACK_STATES",
    "DEFAULT_CONFIG_RELATIVE_PATH",
    "GENERATION",
    "Q15_TO_S32_LEFT_SHIFT",
    "RESULT_DIRECTORY",
    "SCHEMA",
    "load_rt5640_fullband_static_contract",
    "validate_rt5640_fullband_static_contract",
]
