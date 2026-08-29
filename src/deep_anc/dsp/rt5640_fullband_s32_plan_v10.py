"""RT5640 full-octave P/S용 S32 actual-PCM *signal-only* plan.

이 모듈은 v5의 canonical v3 Q15 waveform만 입력 recipe로 사용한다. v5/v6 live
authority, raw publisher, meter, USB hardware identity는 전혀 승계하지 않는다. Q15를
S32로 단순 cast하면 normalized level이 96.3 dB 낮아지므로 exact signed 16-bit left
shift만 허용한다.

ALSA/sounddevice를 import하거나 PCM을 열거나 결과 파일을 쓰지 않는다. 반환 plan은
오직 future S32 transport가 제출해야 하는 sealed planned PCM bytes를 정의하며, 실제
DAC submission/P/S/electrical witness/training authority는 모두 false다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .control_band_contract import BroadbandFullOctaveContractV3
from .fullband_causal_v5 import BLOCK, FS, build_plan_v5
from .rt5640_fullband_static_v10 import (
    DEFAULT_CONFIG_RELATIVE_PATH,
    Q15_TO_S32_LEFT_SHIFT,
    load_rt5640_fullband_static_contract,
)


SCHEMA = "rt5640_fullband_s32_signal_plan_v10"
QUANTIZATION_SCHEMA = "rt5640_q15_to_s32_exact_quantization_v10"
RAW_TARGET_RELATIVE_PATH = "results/rt5640_fullband_v10/raw_capture.npz"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_Q15_DTYPE = np.dtype("<i2")
_S32_DTYPE = np.dtype("<i4")
_S32_MULTIPLIER = 1 << Q15_TO_S32_LEFT_SHIFT


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


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        raise ValueError("PCM SHA input은 C-contiguous여야 합니다")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_q15_stereo(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != _Q15_DTYPE:
        raise ValueError(f"{label}는 exact little-endian int16이어야 합니다")
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] != 2:
        raise ValueError(f"{label}는 nonempty exact [frames,2]이어야 합니다")
    if not array.flags.c_contiguous:
        raise ValueError(f"{label}는 C-contiguous여야 합니다")
    return array


def _require_s32_stereo(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != _S32_DTYPE:
        raise ValueError(f"{label}는 exact little-endian int32이어야 합니다")
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] != 2:
        raise ValueError(f"{label}는 nonempty exact [frames,2]이어야 합니다")
    if not array.flags.c_contiguous:
        raise ValueError(f"{label}는 C-contiguous여야 합니다")
    return array


def q15_to_s32_exact(q15_pcm: np.ndarray) -> np.ndarray:
    """Q15 stereo PCM을 full-scale-normalized equivalent S32 PCM으로 변환한다.

    ``int16`` native shift와 float/saturation을 사용하지 않는다. wide ``int64``에서
    곱한 뒤 범위를 검사해 sign/scale을 보존한다.
    """

    q15 = _require_q15_stereo(q15_pcm, label="Q15 PCM")
    wide = q15.astype(np.int64) * _S32_MULTIPLIER
    info = np.iinfo(np.int32)
    if np.any(wide < info.min) or np.any(wide > info.max):
        raise OverflowError("Q15→S32 exact quantization이 int32 범위를 벗어났습니다")
    s32 = np.ascontiguousarray(wide.astype(_S32_DTYPE))
    # int32 lower 16 bits must be zero. signed right shift is exact because every
    # value was created as an integer multiple of 2**16.
    restored = np.right_shift(s32.astype(np.int64), Q15_TO_S32_LEFT_SHIFT)
    if not np.array_equal(restored, q15.astype(np.int64)):
        raise AssertionError("Q15→S32 exact round-trip이 보존되지 않았습니다")
    if np.any(np.bitwise_and(s32.astype(np.int64), _S32_MULTIPLIER - 1)):
        raise AssertionError("S32 exact plan의 low 16 bits가 0이 아닙니다")
    return s32


def _s32_to_q15_exact(s32_pcm: np.ndarray) -> np.ndarray:
    """sealed S32 plan이 exact Q15 scaling인지 검증하며 역변환한다."""

    s32 = _require_s32_stereo(s32_pcm, label="S32 PCM")
    if np.any(np.bitwise_and(s32.astype(np.int64), _S32_MULTIPLIER - 1)):
        raise ValueError("S32 PCM은 Q15 exact scaling의 low 16 bits가 0이어야 합니다")
    wide = np.right_shift(s32.astype(np.int64), Q15_TO_S32_LEFT_SHIFT)
    info = np.iinfo(np.int16)
    if np.any(wide < info.min) or np.any(wide > info.max):
        raise ValueError("S32 PCM의 Q15 inverse가 int16 범위를 벗어났습니다")
    return np.ascontiguousarray(wide.astype(_Q15_DTYPE))


def _safe_raw_target(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0] != "results"
        or ".." in path.parts
        or path.suffix != ".npz"
        or path.as_posix() != value
    ):
        raise ValueError("v10 raw target은 results/ 아래 canonical .npz 상대경로여야 합니다")
    return value


def _quantization_receipt() -> dict[str, Any]:
    receipt = {
        "schema": QUANTIZATION_SCHEMA,
        "source_dtype": _Q15_DTYPE.str,
        "output_dtype": _S32_DTYPE.str,
        "source_normalization_divisor": 32_768,
        "output_normalization_divisor": 2_147_483_648,
        "conversion": "int64_multiply_then_int32_range_checked_exact_signed_left_shift",
        "left_shift_bits": Q15_TO_S32_LEFT_SHIFT,
        "multiplier": _S32_MULTIPLIER,
        "float_quantization_allowed": False,
        "simple_int16_to_int32_cast_allowed": False,
        "saturation_or_clipping_allowed": False,
        "low_16_bits_must_be_zero": True,
        "signed_right_shift_roundtrip_required": True,
        "normalized_full_scale_preserved": True,
    }
    return {**receipt, "payload_sha256": _payload_sha256(receipt)}


def build_rt5640_fullband_s32_plan_v10(
    *, raw_target_relative_path: str = RAW_TARGET_RELATIVE_PATH
) -> tuple[dict[str, Any], np.ndarray]:
    """v3 Q15 waveform으로부터 future RT5640 S32 planned PCM을 결정론적으로 만든다."""

    raw_target = _safe_raw_target(raw_target_relative_path)
    if raw_target != RAW_TARGET_RELATIVE_PATH:
        raise ValueError("v10 S32 signal plan은 sealed default raw target만 허용합니다")
    static = load_rt5640_fullband_static_contract(
        _REPO_ROOT / DEFAULT_CONFIG_RELATIVE_PATH
    )
    if static["status"] != "BLOCKED" or static["static_gate_pass"] is not True:
        raise ValueError("v10 static contract가 expected BLOCKED/PASS 상태가 아닙니다")
    if static["authority"]["canonical_training_eligible"] is not False:
        raise ValueError("static contract authority가 training으로 누수됐습니다")

    source_plan, source_q15 = build_plan_v5()
    q15 = _require_q15_stereo(source_q15, label="v5 signal-only Q15 PCM")
    source_contract = BroadbandFullOctaveContractV3.canonical()
    if source_plan.get("schema") != "fullband_causal_time_separated_near_white_v5":
        raise ValueError("v5 source signal schema가 expected value와 다릅니다")
    if source_plan.get("role") != "signal_only_dry_run_no_audio":
        raise ValueError("v5 source는 signal-only role이어야 합니다")
    if source_plan.get("live_authority") is not None or source_plan.get("live_capture_enabled") is not False:
        raise ValueError("v5 source live authority를 v10 S32 plan에 가져올 수 없습니다")
    if source_plan.get("canonical_training_eligible") is not False:
        raise ValueError("v5 source training authority를 가져올 수 없습니다")
    if source_plan.get("control_band_contract_sha256") != source_contract.digest():
        raise ValueError("v5 source v3 control-band digest가 다릅니다")
    if source_plan.get("actual_submitted_pcm_sha256") != _array_sha256(q15):
        raise ValueError("v5 source Q15 PCM SHA가 plan과 다릅니다")

    s32 = q15_to_s32_exact(q15)
    if len(s32) % BLOCK or len(s32) != len(q15):
        raise AssertionError("v10 S32 plan의 block/frame count가 Q15 source와 다릅니다")
    if not np.array_equal(_s32_to_q15_exact(s32), q15):
        raise AssertionError("v10 S32 plan은 Q15 source를 exact 복원해야 합니다")

    source_projection = {
        "source_schema": source_plan["schema"],
        "source_role": source_plan["role"],
        "source_canonical_payload_sha256": source_plan["canonical_payload_sha256"],
        "source_control_band_contract_sha256": source_plan[
            "control_band_contract_sha256"
        ],
        "source_q15_pcm_sha256": _array_sha256(q15),
        "source_q15_shape": list(q15.shape),
        "source_q15_dtype": q15.dtype.str,
        "v5_live_authority_inherited": False,
        "v5_raw_publisher_inherited": False,
        "v5_meter_inherited": False,
    }
    quantization = _quantization_receipt()
    s32_abs = np.abs(s32.astype(np.int64))
    plan: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "BLOCKED_MISSING_S32_DUPLEX_AND_ELECTRICAL_WITNESS",
        "role": "signal_only_dry_run_no_audio",
        "sample_rate": FS,
        "block_size": BLOCK,
        "duration_seconds": len(s32) / FS,
        "expected_callbacks": len(s32) // BLOCK,
        "control_band_contract": source_contract.model_dump(mode="json"),
        "control_band_contract_sha256": source_contract.digest(),
        "static_contract": {
            "schema": static["schema"],
            "config_payload_sha256": static["config_payload_sha256"],
            "config_file_sha256": static["config"]["file_sha256"],
            "static_gate_pass": True,
            "authority": static["authority"],
        },
        "source_q15_signal_only": source_projection,
        "quantization": quantization,
        "sealed_planned_s32_pcm_sha256": _array_sha256(s32),
        "sealed_planned_s32_shape": list(s32.shape),
        "sealed_planned_s32_dtype": s32.dtype.str,
        "sealed_planned_s32_min_pcm": int(np.min(s32)),
        "sealed_planned_s32_max_pcm": int(np.max(s32)),
        "sealed_planned_s32_abs_peak_pcm": int(np.max(s32_abs)),
        "sealed_planned_s32_bytes": int(s32.nbytes),
        "per_channel": {
            "noise_out_primary_ch0": {
                "channel": 0,
                "source_q15_sha256": _array_sha256(np.ascontiguousarray(q15[:, 0])),
                "planned_s32_sha256": _array_sha256(np.ascontiguousarray(s32[:, 0])),
            },
            "cancel_out_secondary_ch1": {
                "channel": 1,
                "source_q15_sha256": _array_sha256(np.ascontiguousarray(q15[:, 1])),
                "planned_s32_sha256": _array_sha256(np.ascontiguousarray(s32[:, 1])),
            },
        },
        "future_raw_target": {
            "relative_path": raw_target,
            "file_created_by_this_module": False,
            "raw_schema_created_by_this_module": False,
        },
        "authority": {
            "s32_signal_plan_pass": True,
            "s32_duplex_transport_pass": False,
            "hardware_frame_identity_pass": False,
            "electrical_witness_pass": False,
            "fullband_plant_identification_pass": False,
            "canonical_training_eligible": False,
            "deployment_eligible": False,
        },
    }
    plan["canonical_payload_sha256"] = _payload_sha256(plan)
    return plan, s32


def validate_rt5640_fullband_s32_plan_v10(
    plan: Mapping[str, Any], planned_s32_pcm: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    """수정/채널 교환/단순 cast된 S32 plan을 fail-closed로 거부한다."""

    if not isinstance(plan, Mapping):
        raise ValueError("v10 S32 plan은 mapping이어야 합니다")
    source = _require_s32_stereo(planned_s32_pcm, label="v10 planned S32 PCM")
    expected_plan, expected_pcm = build_rt5640_fullband_s32_plan_v10(
        raw_target_relative_path=RAW_TARGET_RELATIVE_PATH
    )
    if dict(plan) != expected_plan:
        raise ValueError("v10 S32 plan canonical payload가 재생성 결과와 다릅니다")
    if source.shape != expected_pcm.shape or not np.array_equal(source, expected_pcm):
        raise ValueError("v10 S32 planned PCM이 canonical plan과 다릅니다")
    if _array_sha256(source) != plan.get("sealed_planned_s32_pcm_sha256"):
        raise ValueError("v10 S32 planned PCM SHA가 plan과 다릅니다")
    restored = _s32_to_q15_exact(source)
    source_meta = plan.get("source_q15_signal_only")
    if not isinstance(source_meta, Mapping) or source_meta.get("source_q15_pcm_sha256") != _array_sha256(restored):
        raise ValueError("v10 S32→Q15 exact round-trip source SHA가 다릅니다")
    return expected_plan, expected_pcm


__all__ = [
    "QUANTIZATION_SCHEMA",
    "RAW_TARGET_RELATIVE_PATH",
    "SCHEMA",
    "build_rt5640_fullband_s32_plan_v10",
    "q15_to_s32_exact",
    "validate_rt5640_fullband_s32_plan_v10",
]
