"""Stage-2 2 kHz용 RT5640/J511 S32 측정 준비 계약.

이 모듈은 실제 ALSA/PortAudio 장치를 열지 않고, 소리를 내거나 결과 파일을 쓰지
않는다. 목적은 현재 USB AB13X output-master/S16 경로와 **분리된** APE PCM1 입력 /
APE PCM0 출력 공통-clock 후보의 정적 계약과 exact S32 제출 PCM을 만드는 것이다.

Stage-2의 canonical int16 signal-only plan을 ``int64`` 곱셈으로 정확히 16 bit
left-shift한다. 단순 ``int16 -> int32`` cast, USB/AB13X provenance, output-master
split-clock receipt, S16 receipt는 이 경로에 결속될 수 없다. 이 파일의 PASS는
RT5640 S32 transport, J511 물리 연결, P/S, lead, ANC 감쇠 또는 학습 적격성을 뜻하지
않는다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .stage2_2khz_contract import Stage2TwoKilohertzContract
from .stage2_2khz_measurement_v2 import (
    BLOCK_SIZE,
    SAMPLE_RATE,
    build_stage2_v2_live_safe_fallback_plan,
)


SCHEMA = "stage2_2khz_rt5640_s32_static_v1"
STATIC_RECEIPT_SCHEMA = "stage2_2khz_rt5640_s32_static_receipt_v1"
SIGNAL_PLAN_SCHEMA = "stage2_2khz_rt5640_s32_signal_plan_v1"
PROVENANCE_SCHEMA = "stage2_2khz_rt5640_s32_planned_transport_provenance_v1"
DEFAULT_CONFIG_RELATIVE_PATH = "configs/hardware_jetson_rt5640_stage2_2khz_s32.yaml"
RAW_TARGET_RELATIVE_PATH = "results/stage2_2khz_rt5640_s32_v1/raw_capture.npz"
Q15_TO_S32_LEFT_SHIFT = 16
_MULTIPLIER = 1 << Q15_TO_S32_LEFT_SHIFT
_Q15_DTYPE = np.dtype("<i2")
_S32_DTYPE = np.dtype("<i4")
_REPO_ROOT = Path(__file__).resolve().parents[3]

_PORT_KEYS = frozenset({"card", "pcm", "channels", "format", "route"})
_AUDIO_KEYS = frozenset(
    {"sample_rate_hz", "block_size", "latency", "clock_domain", "input", "output"}
)
_STAGE2_KEYS = frozenset(
    {
        "generation",
        "contract_id",
        "contract_sha256",
        "source_int16_plan_schema",
        "source_int16_plan_role",
        "q15_to_s32_left_shift",
        "raw_target_relative_path",
        "usb_ab13x_receipt_reuse_allowed",
        "output_master_receipt_reuse_allowed",
        "s16_receipt_reuse_allowed",
        "live_jack_allowed_states",
    }
)
_AUTHORITY_KEYS = frozenset(
    {
        "static_contract_only",
        "j511_connection_observed",
        "s32_duplex_transport_pass",
        "same_hardware_frame_identity_pass",
        "relative_clock_or_fixed_lti_condition_pass",
        "stage2_ps_identification_pass",
        "canonical_training_eligible",
        "deployment_eligible",
    }
)
_TOP_LEVEL_KEYS = frozenset({"schema", "audio", "channels", "stage2_2khz", "authority"})


class Stage2Rt5640S32ContractError(ValueError):
    """Stage-2 RT5640 S32 contract/provenance가 섞였거나 변조된 경우."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        raise Stage2Rt5640S32ContractError("PCM SHA 입력은 C-contiguous여야 합니다")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage2Rt5640S32ContractError(f"{label}는 mapping이어야 합니다")
    return value


def _require_exact_keys(value: Mapping[str, Any], *, expected: frozenset[str], label: str) -> None:
    observed = frozenset(value)
    if observed != expected:
        raise Stage2Rt5640S32ContractError(
            f"{label} key가 exact하지 않습니다: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _require_exact(value: Any, expected: Any, *, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise Stage2Rt5640S32ContractError(
            f"{label}가 계약과 다릅니다: expected={expected!r}, got={value!r}"
        )


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise Stage2Rt5640S32ContractError(f"{label}는 64자리 SHA-256이어야 합니다")
    try:
        int(value, 16)
    except ValueError as error:
        raise Stage2Rt5640S32ContractError(f"{label}는 hexadecimal SHA-256이어야 합니다") from error
    return value


def _require_q15_stereo(value: np.ndarray, *, label: str, require_block_multiple: bool) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != _Q15_DTYPE:
        raise Stage2Rt5640S32ContractError(f"{label}는 exact little-endian int16이어야 합니다")
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] != 2:
        raise Stage2Rt5640S32ContractError(f"{label}는 nonempty exact [frames,2]이어야 합니다")
    if require_block_multiple and array.shape[0] % BLOCK_SIZE:
        raise Stage2Rt5640S32ContractError("Stage-2 submitted Q15 frame 수는 256의 배수여야 합니다")
    if not array.flags.c_contiguous:
        raise Stage2Rt5640S32ContractError(f"{label}는 C-contiguous여야 합니다")
    return array


def _require_s32_stereo(value: np.ndarray, *, label: str, require_block_multiple: bool) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != _S32_DTYPE:
        raise Stage2Rt5640S32ContractError(f"{label}는 exact little-endian int32이어야 합니다")
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] != 2:
        raise Stage2Rt5640S32ContractError(f"{label}는 nonempty exact [frames,2]이어야 합니다")
    if require_block_multiple and array.shape[0] % BLOCK_SIZE:
        raise Stage2Rt5640S32ContractError("Stage-2 submitted S32 frame 수는 256의 배수여야 합니다")
    if not array.flags.c_contiguous:
        raise Stage2Rt5640S32ContractError(f"{label}는 C-contiguous여야 합니다")
    return array


def q15_to_stage2_rt5640_s32_exact(q15_pcm: np.ndarray) -> np.ndarray:
    """Stage-2 Q15 stereo PCM을 full-scale-equivalent S32_LE로 정확히 변환한다."""

    q15 = _require_q15_stereo(q15_pcm, label="Stage-2 Q15 PCM", require_block_multiple=False)
    wide = q15.astype(np.int64) * _MULTIPLIER
    limits = np.iinfo(np.int32)
    if np.any(wide < limits.min) or np.any(wide > limits.max):
        raise OverflowError("Stage-2 Q15→S32 exact shift가 int32 범위를 벗어났습니다")
    s32 = np.ascontiguousarray(wide.astype(_S32_DTYPE))
    restored = np.right_shift(s32.astype(np.int64), Q15_TO_S32_LEFT_SHIFT)
    if not np.array_equal(restored, q15.astype(np.int64)):
        raise AssertionError("Stage-2 Q15→S32 signed round-trip이 보존되지 않았습니다")
    if np.any(np.bitwise_and(s32.astype(np.int64), _MULTIPLIER - 1)):
        raise AssertionError("Stage-2 S32 submitted PCM의 low 16 bits가 0이 아닙니다")
    return s32


def stage2_rt5640_s32_to_q15_exact(s32_pcm: np.ndarray) -> np.ndarray:
    """exact Q15-left-shift S32 PCM만 역변환한다."""

    s32 = _require_s32_stereo(s32_pcm, label="Stage-2 S32 PCM", require_block_multiple=False)
    if np.any(np.bitwise_and(s32.astype(np.int64), _MULTIPLIER - 1)):
        raise Stage2Rt5640S32ContractError(
            "Stage-2 S32 PCM은 Q15 exact scaling의 low 16 bits가 0이어야 합니다"
        )
    wide = np.right_shift(s32.astype(np.int64), Q15_TO_S32_LEFT_SHIFT)
    limits = np.iinfo(np.int16)
    if np.any(wide < limits.min) or np.any(wide > limits.max):
        raise Stage2Rt5640S32ContractError("Stage-2 S32 inverse가 int16 범위를 벗어났습니다")
    return np.ascontiguousarray(wide.astype(_Q15_DTYPE))


def _validate_port(value: Any, *, expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    port = _require_mapping(value, label=label)
    _require_exact_keys(port, expected=_PORT_KEYS, label=label)
    for key, expected_value in expected.items():
        _require_exact(port[key], expected_value, label=f"{label}.{key}")
    return dict(port)


def validate_stage2_rt5640_s32_static_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Stage-2 전용 RT5640/J511 S32 static config를 fail-closed로 검사한다."""

    root = _require_mapping(payload, label="Stage-2 RT5640 S32 config")
    _require_exact_keys(root, expected=_TOP_LEVEL_KEYS, label="Stage-2 RT5640 S32 config")
    _require_exact(root["schema"], SCHEMA, label="schema")

    audio = _require_mapping(root["audio"], label="audio")
    _require_exact_keys(audio, expected=_AUDIO_KEYS, label="audio")
    _require_exact(audio["sample_rate_hz"], SAMPLE_RATE, label="audio.sample_rate_hz")
    _require_exact(audio["block_size"], BLOCK_SIZE, label="audio.block_size")
    _require_exact(audio["latency"], "low", label="audio.latency")
    _require_exact(audio["clock_domain"], "APE_PLL_A_SHARED", label="audio.clock_domain")
    input_port = _validate_port(
        audio["input"],
        expected={
            "card": "APE",
            "pcm": 1,
            "channels": 2,
            "format": "S32_LE",
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
            "format": "S32_LE",
            "route": "ADMAIF1_I2S1_RT5640_J511",
        },
        label="audio.output",
    )

    channels = _require_mapping(root["channels"], label="channels")
    expected_channels = {"error_mic": 0, "reference_mic": 1, "noise_out": 0, "cancel_out": 1}
    _require_exact_keys(channels, expected=frozenset(expected_channels), label="channels")
    for key, expected_value in expected_channels.items():
        _require_exact(channels[key], expected_value, label=f"channels.{key}")

    contract = Stage2TwoKilohertzContract.canonical()
    stage2 = _require_mapping(root["stage2_2khz"], label="stage2_2khz")
    _require_exact_keys(stage2, expected=_STAGE2_KEYS, label="stage2_2khz")
    expected_stage2 = {
        "generation": "rt5640_stage2_2khz_s32_v1",
        "contract_id": contract.contract_id,
        "contract_sha256": contract.digest(),
        "source_int16_plan_schema": "stage2_2khz_time_separated_lower_guard_dpss_plan_v2",
        "source_int16_plan_role": "signal_only_live_safe_fallback_no_audio_authority",
        "q15_to_s32_left_shift": Q15_TO_S32_LEFT_SHIFT,
        "raw_target_relative_path": RAW_TARGET_RELATIVE_PATH,
        "usb_ab13x_receipt_reuse_allowed": False,
        "output_master_receipt_reuse_allowed": False,
        "s16_receipt_reuse_allowed": False,
        "live_jack_allowed_states": ["HP", "HS"],
    }
    for key, expected_value in expected_stage2.items():
        _require_exact(stage2[key], expected_value, label=f"stage2_2khz.{key}")

    authority = _require_mapping(root["authority"], label="authority")
    _require_exact_keys(authority, expected=_AUTHORITY_KEYS, label="authority")
    expected_authority = {
        "static_contract_only": True,
        "j511_connection_observed": False,
        "s32_duplex_transport_pass": False,
        "same_hardware_frame_identity_pass": False,
        "relative_clock_or_fixed_lti_condition_pass": False,
        "stage2_ps_identification_pass": False,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
    }
    for key, expected_value in expected_authority.items():
        _require_exact(authority[key], expected_value, label=f"authority.{key}")

    return {
        "schema": STATIC_RECEIPT_SCHEMA,
        "status": "BLOCKED_MISSING_RT5640_S32_CAPTURE_ADAPTER_AND_PHYSICAL_RAW",
        "static_gate_pass": True,
        "audio_opened": False,
        "results_written": False,
        "stage2_contract": contract.model_dump(mode="json"),
        "stage2_contract_sha256": contract.digest(),
        "hardware_audio": {
            "sample_rate_hz": SAMPLE_RATE,
            "block_size": BLOCK_SIZE,
            "latency": "low",
            "clock_domain": "APE_PLL_A_SHARED",
            "input": input_port,
            "output": output_port,
            "channels": dict(channels),
        },
        "source_plan_binding": {
            "schema": expected_stage2["source_int16_plan_schema"],
            "role": expected_stage2["source_int16_plan_role"],
            "q15_to_s32_left_shift": Q15_TO_S32_LEFT_SHIFT,
            "simple_int16_to_int32_cast_allowed": False,
            "low_16_bits_must_be_zero": True,
            "signed_right_shift_roundtrip_required": True,
        },
        "forbidden_receipt_origins": {
            "usb_ab13x": True,
            "output_master_split_clock": True,
            "s16_transport": True,
            "legacy_relabel_or_promotion": True,
        },
        "next_required_gates": [
            "j511_HP_or_HS_three_read_only_samples",
            "no_audio_s32_duplex_dry_run",
            "same_card_S32_actual_submitted_and_captured_raw_first",
            "shared_q_or_fixed_LTI_conditional_clock_receipt",
            "stage2_2khz_P_S_analysis_and_plant_binding",
        ],
        "authority": expected_authority,
        "config_payload_sha256": _payload_sha256(dict(root)),
    }


def _default_config_path(repository_root: Path) -> Path:
    return (repository_root / DEFAULT_CONFIG_RELATIVE_PATH).resolve()


def load_stage2_rt5640_s32_static_contract(
    path: str | Path | None = None, *, repository_root: str | Path = _REPO_ROOT
) -> dict[str, Any]:
    """sealed Stage-2 RT5640 config만 read-only로 검증한다."""

    root = Path(repository_root).resolve()
    expected = _default_config_path(root)
    supplied = None if path is None else Path(path)
    config_path = (
        expected
        if supplied is None
        else (root / supplied).resolve()
        if not supplied.is_absolute()
        else supplied.resolve()
    )
    if config_path != expected:
        raise Stage2Rt5640S32ContractError("Stage-2 RT5640 S32은 sealed default config만 허용합니다")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise Stage2Rt5640S32ContractError("Stage-2 RT5640 S32 config YAML root는 mapping이어야 합니다")
    receipt = validate_stage2_rt5640_s32_static_contract(loaded)
    raw = config_path.read_bytes()
    return {
        **receipt,
        "config": {
            "path": str(config_path),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
        },
    }


def build_stage2_rt5640_s32_signal_plan(
    *, raw_target_relative_path: str = RAW_TARGET_RELATIVE_PATH
) -> tuple[dict[str, Any], np.ndarray]:
    """Stage-2 fallback Q15 signal plan을 RT5640 S32 submitted plan으로 정확히 봉인한다."""

    if raw_target_relative_path != RAW_TARGET_RELATIVE_PATH:
        raise Stage2Rt5640S32ContractError("Stage-2 S32 plan은 sealed default raw target만 허용합니다")
    static = load_stage2_rt5640_s32_static_contract(repository_root=_REPO_ROOT)
    if static["status"] != "BLOCKED_MISSING_RT5640_S32_CAPTURE_ADAPTER_AND_PHYSICAL_RAW":
        raise Stage2Rt5640S32ContractError("Stage-2 RT5640 static status가 예상과 다릅니다")
    if static["authority"]["canonical_training_eligible"] is not False:
        raise Stage2Rt5640S32ContractError("static authority가 canonical training으로 누수됐습니다")

    source_plan, source_q15 = build_stage2_v2_live_safe_fallback_plan()
    q15 = _require_q15_stereo(source_q15, label="Stage-2 source Q15 PCM", require_block_multiple=True)
    source_contract = Stage2TwoKilohertzContract.canonical()
    expected_source_schema = "stage2_2khz_time_separated_lower_guard_dpss_plan_v2"
    expected_source_role = "signal_only_live_safe_fallback_no_audio_authority"
    if source_plan.get("schema") != expected_source_schema:
        raise Stage2Rt5640S32ContractError("Stage-2 source signal plan schema가 다릅니다")
    if source_plan.get("role") != expected_source_role:
        raise Stage2Rt5640S32ContractError("Stage-2 source signal plan role이 다릅니다")
    if source_plan.get("contract", {}).get("sha256") != source_contract.digest():
        raise Stage2Rt5640S32ContractError("Stage-2 source signal plan contract SHA가 다릅니다")
    if source_plan.get("actual_submitted_pcm", {}).get("sha256") != _array_sha256(q15):
        raise Stage2Rt5640S32ContractError("Stage-2 source Q15 PCM SHA가 plan과 다릅니다")
    if source_plan.get("live_safety", {}).get("audio_execution_allowed_by_this_plan") is not False:
        raise Stage2Rt5640S32ContractError("source signal plan의 audio authority가 허용돼서는 안 됩니다")
    if source_plan.get("consumer_binding", {}).get("canonical_training_eligible") is not False:
        raise Stage2Rt5640S32ContractError("source signal plan의 training authority가 허용돼서는 안 됩니다")

    s32 = q15_to_stage2_rt5640_s32_exact(q15)
    s32 = _require_s32_stereo(s32, label="Stage-2 planned S32 PCM", require_block_multiple=True)
    restored = stage2_rt5640_s32_to_q15_exact(s32)
    if not np.array_equal(restored, q15):
        raise AssertionError("Stage-2 planned S32의 Q15 inverse가 byte-exact하지 않습니다")
    s32.setflags(write=False)
    s32_abs = np.abs(s32.astype(np.int64))
    plan: dict[str, Any] = {
        "schema": SIGNAL_PLAN_SCHEMA,
        "status": "BLOCKED_MISSING_RT5640_S32_CAPTURE_ADAPTER_AND_PHYSICAL_RAW",
        "role": "signal_only_no_audio_no_training_authority",
        "sample_rate_hz": SAMPLE_RATE,
        "block_size": BLOCK_SIZE,
        "duration_seconds": len(s32) / SAMPLE_RATE,
        "expected_callbacks": len(s32) // BLOCK_SIZE,
        "stage2_contract_sha256": source_contract.digest(),
        "rt5640_static_contract": {
            "schema": static["schema"],
            "config_payload_sha256": static["config_payload_sha256"],
            "config_file_sha256": static["config"]["file_sha256"],
            "hardware_audio": static["hardware_audio"],
            "forbidden_receipt_origins": static["forbidden_receipt_origins"],
        },
        "source_int16_plan": {
            "schema": source_plan["schema"],
            "role": source_plan["role"],
            "canonical_payload_sha256": source_plan["canonical_payload_sha256"],
            "stage2_contract_sha256": source_plan["contract"]["sha256"],
            "actual_submitted_pcm_sha256": _array_sha256(q15),
            "actual_submitted_pcm_shape": list(q15.shape),
            "actual_submitted_pcm_dtype": q15.dtype.str,
            "audio_execution_allowed_by_source_plan": False,
            "canonical_training_eligible_by_source_plan": False,
        },
        "quantization": {
            "source_dtype": _Q15_DTYPE.str,
            "output_dtype": _S32_DTYPE.str,
            "conversion": "int64_multiply_then_int32_range_checked_exact_signed_left_shift",
            "left_shift_bits": Q15_TO_S32_LEFT_SHIFT,
            "multiplier": _MULTIPLIER,
            "simple_int16_to_int32_cast_allowed": False,
            "float_quantization_allowed": False,
            "saturation_or_clipping_allowed": False,
            "low_16_bits_must_be_zero": True,
            "signed_right_shift_roundtrip_required": True,
            "normalized_full_scale_preserved": True,
        },
        "sealed_planned_s32_pcm": {
            "sha256": _array_sha256(s32),
            "shape": list(s32.shape),
            "dtype": s32.dtype.str,
            "bytes": int(s32.nbytes),
            "min_pcm": int(np.min(s32)),
            "max_pcm": int(np.max(s32)),
            "abs_peak_pcm": int(np.max(s32_abs)),
        },
        "future_raw_target": {
            "relative_path": raw_target_relative_path,
            "file_created_by_this_module": False,
            "raw_schema_created_by_this_module": False,
        },
        "authority": {
            "signal_plan_pass": True,
            "s32_duplex_transport_pass": False,
            "same_hardware_frame_identity_pass": False,
            "relative_clock_or_fixed_lti_condition_pass": False,
            "stage2_ps_identification_pass": False,
            "canonical_training_eligible": False,
            "deployment_eligible": False,
        },
    }
    plan["canonical_payload_sha256"] = _payload_sha256(plan)
    return plan, s32


def validate_stage2_rt5640_s32_signal_plan(
    plan: Mapping[str, Any], planned_s32_pcm: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    """Stage-2 source/scale/route SHA가 하나라도 달라지면 plan을 거부한다."""

    if not isinstance(plan, Mapping):
        raise Stage2Rt5640S32ContractError("Stage-2 RT5640 S32 plan은 mapping이어야 합니다")
    source = _require_s32_stereo(planned_s32_pcm, label="Stage-2 planned S32 PCM", require_block_multiple=True)
    expected_plan, expected_pcm = build_stage2_rt5640_s32_signal_plan()
    if dict(plan) != expected_plan:
        raise Stage2Rt5640S32ContractError("Stage-2 RT5640 S32 canonical plan payload가 다릅니다")
    if source.shape != expected_pcm.shape or not np.array_equal(source, expected_pcm):
        raise Stage2Rt5640S32ContractError("Stage-2 RT5640 S32 planned PCM이 canonical plan과 다릅니다")
    if _array_sha256(source) != plan["sealed_planned_s32_pcm"]["sha256"]:
        raise Stage2Rt5640S32ContractError("Stage-2 RT5640 S32 planned PCM SHA가 다릅니다")
    restored = stage2_rt5640_s32_to_q15_exact(source)
    if _array_sha256(restored) != plan["source_int16_plan"]["actual_submitted_pcm_sha256"]:
        raise Stage2Rt5640S32ContractError("Stage-2 S32 inverse source SHA가 다릅니다")
    return expected_plan, expected_pcm


def build_stage2_rt5640_s32_planned_transport_provenance(
    plan: Mapping[str, Any], planned_s32_pcm: np.ndarray
) -> dict[str, Any]:
    """future live adapter가 반드시 보존할 plan-to-transport binding을 만든다.

    이는 actual capture receipt가 아니라 출력 전 무음 provenance이다. 정확한 topology를
    강제해 USB/AB13X, split output-master, S16 receipt가 이후 raw binding에 섞일 수 없게
    한다.
    """

    canonical, pcm = validate_stage2_rt5640_s32_signal_plan(plan, planned_s32_pcm)
    static = canonical["rt5640_static_contract"]
    provenance: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "status": "PLANNED_ONLY_NO_AUDIO_NO_CAPTURE",
        "planned_signal_plan_sha256": canonical["canonical_payload_sha256"],
        "stage2_contract_sha256": canonical["stage2_contract_sha256"],
        "hardware_contract": {
            "config_payload_sha256": static["config_payload_sha256"],
            "config_file_sha256": static["config_file_sha256"],
            "input": static["hardware_audio"]["input"],
            "output": static["hardware_audio"]["output"],
            "sample_rate_hz": SAMPLE_RATE,
            "block_size": BLOCK_SIZE,
            "native_format": "S32_LE",
            "same_card": True,
            "same_clock_domain": "APE_PLL_A_SHARED",
        },
        "source_int16_plan": canonical["source_int16_plan"],
        "planned_s32": {
            **canonical["sealed_planned_s32_pcm"],
            "low_16_bits_zero": True,
            "q15_inverse_sha256": _array_sha256(stage2_rt5640_s32_to_q15_exact(pcm)),
        },
        "prohibited_receipt_lineage": {
            "usb_ab13x": True,
            "output_master_split_clock": True,
            "s16_transport": True,
            "legacy_relabel_or_promotion": True,
        },
        "actual_capture_present": False,
        "actual_audio_output_claimed": False,
        "actual_ps_or_training_authority_claimed": False,
    }
    provenance["canonical_payload_sha256"] = _payload_sha256(provenance)
    return provenance


def validate_stage2_rt5640_s32_planned_transport_provenance(
    provenance: Mapping[str, Any], plan: Mapping[str, Any], planned_s32_pcm: np.ndarray
) -> dict[str, Any]:
    """future adapter가 plan provenance를 다른 transport/receipt로 바꾸지 못하게 한다."""

    if not isinstance(provenance, Mapping):
        raise Stage2Rt5640S32ContractError("Stage-2 RT5640 S32 provenance는 mapping이어야 합니다")
    expected = build_stage2_rt5640_s32_planned_transport_provenance(plan, planned_s32_pcm)
    if dict(provenance) != expected:
        raise Stage2Rt5640S32ContractError(
            "Stage-2 RT5640 S32 provenance가 USB/output-master/S16 또는 다른 plan과 섞였습니다"
        )
    _sha256(provenance.get("canonical_payload_sha256"), label="planned provenance SHA")
    return expected


__all__ = [
    "DEFAULT_CONFIG_RELATIVE_PATH",
    "PROVENANCE_SCHEMA",
    "Q15_TO_S32_LEFT_SHIFT",
    "RAW_TARGET_RELATIVE_PATH",
    "SCHEMA",
    "SIGNAL_PLAN_SCHEMA",
    "Stage2Rt5640S32ContractError",
    "build_stage2_rt5640_s32_planned_transport_provenance",
    "build_stage2_rt5640_s32_signal_plan",
    "load_stage2_rt5640_s32_static_contract",
    "q15_to_stage2_rt5640_s32_exact",
    "stage2_rt5640_s32_to_q15_exact",
    "validate_stage2_rt5640_s32_planned_transport_provenance",
    "validate_stage2_rt5640_s32_signal_plan",
    "validate_stage2_rt5640_s32_static_contract",
]
