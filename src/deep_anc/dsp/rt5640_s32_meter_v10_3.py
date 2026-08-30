"""RT5640 S32 full-octave 측정의 fresh level-meter *recipe*.

이 모듈은 PCM/ALSA/sounddevice를 열지 않는다. 기존 AB13X/S16 level evidence를
RT5640으로 재표기하지 않고, 같은 0.003 Q15 probe를 exact Q15→S32 left-shift로
재생성한다. ``-50.1±2 dBFS``는 150--1600 Hz amplifier-level compatibility
control일 뿐 2/4/8 kHz P/S 또는 ANC authority가 아니다.

full-octave health와 causal P/S authority는 이후 raw-first capture/analysis가 따로
발행해야 한다. 따라서 모든 반환 payload는 training/deployment authority를 false로
고정한다.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..audio_io import float32_to_pcm_int16
from .control_band_contract import BroadbandFullOctaveContractV3
from .interleaved_probe import build_interleaved_probe, crest_factor_db
from .measurement_level import OFFICIAL_MEASUREMENT_LEVEL
from .rt5640_fullband_s32_plan_v10 import q15_to_s32_exact
from .rt5640_fullband_static_v10 import (
    DEFAULT_CONFIG_RELATIVE_PATH,
    Q15_TO_S32_LEFT_SHIFT,
    load_rt5640_fullband_static_contract,
)


SCHEMA = "rt5640_s32_fullband_meter_contract_v10_3"
METER_PLAN_SCHEMA = "rt5640_s32_level_control_plan_v10_3"
METER_RAW_SCHEMA = "rt5640_s32_level_meter_raw_v10_3"
METER_RECEIPT_SCHEMA = "rt5640_s32_level_meter_receipt_v10_3"
DEFAULT_CONFIG_RELATIVE_PATH_V10_3 = "configs/hardware_jetson_rt5640_fullband_s32_v10_3.yaml"
DEFAULT_RAW_TARGET_RELATIVE_PATH = "results/rt5640_fullband_s32_v10_3/level_meter_raw.npz"

SAMPLE_RATE = 48_000
BLOCK_SIZE = 256
METER_SECONDS = 20.0
METER_FRAMES = int(SAMPLE_RATE * METER_SECONDS)
Q15_DTYPE = np.dtype("<i2")
S32_DTYPE = np.dtype("<i4")
_S32_SHIFT = 1 << Q15_TO_S32_LEFT_SHIFT
_REPO_ROOT = Path(__file__).resolve().parents[3]

if METER_FRAMES % BLOCK_SIZE:
    raise AssertionError("20초 meter frame은 256의 배수여야 합니다")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        raise ValueError("PCM SHA array는 C-contiguous여야 합니다")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _exact_mapping(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{label} key 집합이 정확하지 않습니다: {actual}")
    return dict(value)


def _exact(value: object, expected: object, *, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise ValueError(f"{label}가 계약과 다릅니다: expected={expected!r}, got={value!r}")


def _require_s32_stereo(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != S32_DTYPE or array.shape != (METER_FRAMES, 2) or not array.flags.c_contiguous:
        raise ValueError(f"{label}는 exact <i4 [{METER_FRAMES},2] C-contiguous여야 합니다")
    return array


def _s32_to_q15_exact(value: np.ndarray) -> np.ndarray:
    s32 = _require_s32_stereo(value, label="S32 level meter PCM")
    wide = s32.astype(np.int64)
    if np.any(np.bitwise_and(wide, _S32_SHIFT - 1)):
        raise ValueError("S32 level meter PCM의 low 16 bits가 0이 아닙니다")
    restored = np.right_shift(wide, Q15_TO_S32_LEFT_SHIFT)
    limits = np.iinfo(np.int16)
    if np.any(restored < limits.min) or np.any(restored > limits.max):
        raise ValueError("S32 level meter PCM의 Q15 inverse가 int16 범위를 벗어났습니다")
    return np.ascontiguousarray(restored.astype(Q15_DTYPE))


def _meter_q15_pcm() -> tuple[np.ndarray, dict[str, Any]]:
    """공식 low-band probe를 exact Q15 stereo recipe로 재생성한다.

    Level target의 뜻은 output peak 하나가 아니라 signal crest/RMS recipe까지
    동일하다는 것이다. 그래서 old output bytes(S16)를 가져오지 않고 same float
    generator→Q15 quantization만 공유한다.
    """

    level = OFFICIAL_MEASUREMENT_LEVEL
    if (
        level.sample_rate != SAMPLE_RATE
        or level.meter_seconds != METER_SECONDS
        or level.probe_amplitude != 0.003
        or level.design_band_hz != (60.0, 1650.0)
        or level.meter_band_hz != (150.0, 1600.0)
    ):
        raise ValueError("official low-band MeasurementLevelContract가 expected recipe와 다릅니다")
    probe = build_interleaved_probe(
        sample_rate=level.sample_rate,
        period_seconds=level.period_seconds,
        band_hz=level.design_band_hz,
        amplitude=level.probe_amplitude,
        tone_spacing_hz=None,
    )
    repeats = int(math.ceil(METER_FRAMES / probe.period_samples))
    mono = np.tile(np.asarray(probe.noise_signal, dtype=np.float32), repeats)[:METER_FRAMES]
    q15 = np.zeros((METER_FRAMES, 2), dtype=Q15_DTYPE)
    q15[:, 0] = float32_to_pcm_int16(mono)
    if int(np.max(np.abs(q15[:, 0].astype(np.int64)))) != 98:
        raise AssertionError("0.003 level-control probe의 exact Q15 peak가 바뀌었습니다")
    if np.any(q15[:, 1]):
        raise AssertionError("level-control meter ch1은 exact zero여야 합니다")
    recipe = {
        "source": "official_measurement_level_same_float_generator_then_q15",
        "sample_rate_hz": SAMPLE_RATE,
        "meter_seconds": METER_SECONDS,
        "meter_frames": METER_FRAMES,
        "period_seconds": level.period_seconds,
        "period_samples": probe.period_samples,
        "design_band_hz": list(level.design_band_hz),
        "meter_band_hz": list(level.meter_band_hz),
        "probe_peak_normalized": level.probe_amplitude,
        "actual_q15_peak": 98,
        "actual_s32_peak": int(98 * _S32_SHIFT),
        "meter_target_dbfs": level.meter_target_dbfs,
        "meter_tolerance_db": level.meter_tolerance_db,
        "interleaved_err_noise_bin_dbfs": level.interleaved_err_noise_bin_dbfs,
        "interleaved_err_noise_bin_tolerance_db": level.interleaved_err_noise_bin_tolerance_db,
        "noise_out_channel": 0,
        "cancel_out_channel_exact_zero": 1,
        "noise_channel_crest_factor_db": float(crest_factor_db(mono)),
        "q15_pcm_sha256": _array_sha256(q15),
    }
    recipe["payload_sha256"] = _payload_sha256(recipe)
    return q15, recipe


def build_rt5640_s32_level_control_plan_v10_3() -> tuple[dict[str, Any], np.ndarray]:
    """20초 low-band level control S32 PCM을 pure deterministic하게 만든다."""

    q15, recipe = _meter_q15_pcm()
    s32 = q15_to_s32_exact(q15)
    if s32.dtype != S32_DTYPE or s32.shape != (METER_FRAMES, 2):
        raise AssertionError("Q15→S32 meter shape/dtype가 다릅니다")
    if not np.array_equal(_s32_to_q15_exact(s32), q15):
        raise AssertionError("level-control Q15→S32 round-trip이 다릅니다")
    if np.any(s32[:, 1]) or np.any(np.bitwise_and(s32.astype(np.int64), _S32_SHIFT - 1)):
        raise AssertionError("level-control S32 channel/low-bit 계약이 다릅니다")
    contract = BroadbandFullOctaveContractV3.canonical()
    plan: dict[str, Any] = {
        "schema": METER_PLAN_SCHEMA,
        "role": "fresh_rt5640_s32_low_band_level_control_only",
        "sample_rate_hz": SAMPLE_RATE,
        "block_size": BLOCK_SIZE,
        "frames": METER_FRAMES,
        "callbacks": METER_FRAMES // BLOCK_SIZE,
        "duration_seconds": METER_SECONDS,
        "dtype": S32_DTYPE.str,
        "dither_off": True,
        "channel_semantics": {
            "noise_out_channel": 0,
            "cancel_out_channel": 1,
            "cancel_out_exact_zero": True,
        },
        "control_band_contract_sha256": contract.digest(),
        "level_control_recipe": recipe,
        "q15_to_s32": {
            "conversion": "int64_multiply_then_int32_range_checked_exact_signed_left_shift",
            "left_shift_bits": Q15_TO_S32_LEFT_SHIFT,
            "simple_int16_to_int32_cast_allowed": False,
            "float_quantization_allowed": False,
            "low_16_bits_must_be_zero": True,
            "normalized_full_scale_preserved": True,
        },
        "planned_q15_pcm_sha256": _array_sha256(q15),
        "planned_s32_pcm_sha256": _array_sha256(s32),
        "authority": {
            "low_band_level_recipe_pass": True,
            "full_octave_health_pass": False,
            "s32_transport_pass": False,
            "electrical_output_witness_pass": False,
            "fullband_plant_identification_pass": False,
            "canonical_training_eligible": False,
            "deployment_eligible": False,
        },
    }
    plan["canonical_payload_sha256"] = _payload_sha256(plan)
    return plan, s32


def validate_rt5640_s32_level_control_plan_v10_3(
    plan: Mapping[str, Any], planned_pcm_s32: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    """재생성 결과와 다른 level meter plan/PCM은 fail-closed로 거부한다."""

    actual = _require_s32_stereo(planned_pcm_s32, label="planned S32 level meter PCM")
    expected_plan, expected_pcm = build_rt5640_s32_level_control_plan_v10_3()
    if dict(plan) != expected_plan:
        raise ValueError("S32 level-control plan canonical payload가 재생성 결과와 다릅니다")
    if not np.array_equal(actual, expected_pcm):
        raise ValueError("S32 level-control PCM이 canonical recipe와 다릅니다")
    if _array_sha256(actual) != plan["planned_s32_pcm_sha256"]:
        raise ValueError("S32 level-control PCM SHA가 plan과 다릅니다")
    return expected_plan, expected_pcm


def validate_rt5640_s32_meter_static_contract_v10_3(
    payload: Mapping[str, Any], *, repository_root: str | Path = _REPO_ROOT
) -> dict[str, Any]:
    """YAML/recipe만 검사한다. device open·결과 write·audio output은 0회다."""

    root = Path(repository_root).resolve(strict=True)
    config = _exact_mapping(
        payload,
        {"schema", "hardware_static_config", "fullband_v3", "level_meter", "authority"},
        label="RT5640 S32 meter config",
    )
    _exact(config["schema"], SCHEMA, label="schema")
    _exact(
        config["hardware_static_config"],
        DEFAULT_CONFIG_RELATIVE_PATH,
        label="hardware_static_config",
    )
    static = load_rt5640_fullband_static_contract(root / DEFAULT_CONFIG_RELATIVE_PATH)
    if static["static_gate_pass"] is not True or static["audio_opened"] is not False:
        raise ValueError("v10 RT5640 static contract가 expected no-audio state가 아닙니다")
    contract = BroadbandFullOctaveContractV3.canonical()
    fullband = _exact_mapping(
        config["fullband_v3"],
        {"control_band_contract_id", "control_band_contract_sha256", "health_bands_hz"},
        label="fullband_v3",
    )
    _exact(fullband["control_band_contract_id"], contract.contract_id, label="fullband_v3 id")
    _exact(fullband["control_band_contract_sha256"], contract.digest(), label="fullband_v3 SHA")
    expected_health = [list(band) for band in contract.equal_weight_octave_objective_bands_hz]
    _exact(fullband["health_bands_hz"], expected_health, label="fullband_v3 health bands")
    meter = _exact_mapping(
        config["level_meter"],
        {
            "duration_seconds",
            "probe_peak_normalized",
            "control_band_hz",
            "meter_target_dbfs",
            "meter_tolerance_db",
            "freshness_seconds",
            "raw_target_relative_path",
            "disarmed_stream_required",
            "immediate_disconnect_notice_required",
        },
        label="level_meter",
    )
    expected_meter = {
        "duration_seconds": METER_SECONDS,
        "probe_peak_normalized": 0.003,
        "control_band_hz": [150.0, 1600.0],
        "meter_target_dbfs": -50.1,
        "meter_tolerance_db": 2.0,
        "freshness_seconds": 600,
        "raw_target_relative_path": DEFAULT_RAW_TARGET_RELATIVE_PATH,
        "disarmed_stream_required": True,
        "immediate_disconnect_notice_required": True,
    }
    for key, expected in expected_meter.items():
        _exact(meter[key], expected, label=f"level_meter.{key}")
    authority = _exact_mapping(
        config["authority"],
        {
            "static_contract_only",
            "level_control_only",
            "full_octave_health_pass",
            "plant_identification_pass",
            "canonical_training_eligible",
            "deployment_eligible",
        },
        label="authority",
    )
    expected_authority = {
        "static_contract_only": True,
        "level_control_only": True,
        "full_octave_health_pass": False,
        "plant_identification_pass": False,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
    }
    if authority != expected_authority:
        raise ValueError("S32 meter static authority가 level-control-only 경계를 벗어났습니다")
    plan, pcm = build_rt5640_s32_level_control_plan_v10_3()
    return {
        "schema": "rt5640_s32_fullband_meter_static_receipt_v10_3",
        "status": "BLOCKED",
        "static_gate_pass": True,
        "audio_opened": False,
        "results_written": False,
        "control_band_contract_sha256": contract.digest(),
        "hardware_static_config": {
            "path": DEFAULT_CONFIG_RELATIVE_PATH,
            "file_sha256": static["config"]["file_sha256"],
            "payload_sha256": static["config_payload_sha256"],
        },
        "level_control_plan": plan,
        "planned_s32_pcm_sha256": _array_sha256(pcm),
        "raw_target_relative_path": DEFAULT_RAW_TARGET_RELATIVE_PATH,
        "next_required_gates": [
            "read_only_occupancy_route_and_j511_preflight",
            "disarmed_s32_stream_hw_params_admission",
            "fresh_20_second_s32_level_raw_and_receipt",
            "external_electrical_output_witness",
            "full_octave_raw_first_P_S_analysis",
        ],
        "authority": expected_authority,
    }


def load_rt5640_s32_meter_static_contract_v10_3(
    path: str | Path = DEFAULT_CONFIG_RELATIVE_PATH_V10_3,
    *,
    repository_root: str | Path = _REPO_ROOT,
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    expected = root / DEFAULT_CONFIG_RELATIVE_PATH_V10_3
    if config_path.resolve(strict=False) != expected:
        raise ValueError("v10.3 meter static gate는 sealed default config만 허용합니다")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("v10.3 S32 meter config 최상위는 mapping이어야 합니다")
    receipt = validate_rt5640_s32_meter_static_contract_v10_3(
        loaded, repository_root=root
    )
    raw = config_path.read_bytes()
    return {
        **receipt,
        "config": {"path": str(config_path), "file_sha256": hashlib.sha256(raw).hexdigest()},
    }


__all__ = [
    "BLOCK_SIZE",
    "DEFAULT_CONFIG_RELATIVE_PATH_V10_3",
    "DEFAULT_RAW_TARGET_RELATIVE_PATH",
    "METER_FRAMES",
    "METER_PLAN_SCHEMA",
    "METER_RAW_SCHEMA",
    "METER_RECEIPT_SCHEMA",
    "METER_SECONDS",
    "SCHEMA",
    "build_rt5640_s32_level_control_plan_v10_3",
    "load_rt5640_s32_meter_static_contract_v10_3",
    "validate_rt5640_s32_level_control_plan_v10_3",
    "validate_rt5640_s32_meter_static_contract_v10_3",
]
