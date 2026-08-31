"""RT5640/J511용 Stage-2 실제 P/S 자극의 무음 준비 계약.

이 모듈은 실제 ALSA/PortAudio backend를 import하거나 PCM을 열지 않는다. 기존 USB
AB13X/S16 및 output-master 진단, 그리고 band-limited fallback plan을 P/S raw로
재표기하지 않기 위해, *time-separated full-PE* Stage-2 source plan의 byte-exact
Q15 자극만 읽어 S32_LE 제출 후보로 변환한다.

여기서 ``actual``은 primary/secondary path 식별에 사용할 시간/채널 역할이 실제
측정용으로 명시됐다는 뜻일 뿐, 물리 출력·P/S 식별·plant binding·학습 권한을 뜻하지
않는다. 이 plan은 후속 same-card S32 raw-first adapter가 반드시 소비해야 할 준비
artifact이며, 그 adapter가 생기기 전에는 모든 authority bit가 false다.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
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
    FIT_ROLES,
    HOLDOUT_ROLE,
    PATH_CHANNEL,
    PATHS,
    SAMPLE_RATE,
    SCHEMA as SOURCE_MEASUREMENT_PLAN_SCHEMA,
    Stage2MeasurementV2Error,
    build_stage2_v2_signal_plan,
    validate_stage2_v2_signal_plan,
)


CONFIG_SCHEMA = "stage2_2khz_rt5640_actual_ps_s32_config_v1"
PLAN_SCHEMA = "stage2_2khz_rt5640_actual_ps_excitation_plan_v1"
PROVENANCE_SCHEMA = "stage2_2khz_rt5640_actual_ps_planned_provenance_v1"
STATIC_RECEIPT_SCHEMA = "stage2_2khz_rt5640_actual_ps_static_receipt_v1"
DEFAULT_CONFIG_RELATIVE_PATH = "configs/hardware_jetson_rt5640_stage2_2khz_actual_ps_s32.yaml"
RAW_TARGET_RELATIVE_PATH = "results/stage2_2khz_rt5640_actual_ps_s32_v1/native_raw_capture.npz"
ANALYSIS_TARGET_RELATIVE_PATH = "results/stage2_2khz_rt5640_actual_ps_s32_v1/analysis_receipt.json"

Q15_TO_S32_LEFT_SHIFT = 16
_Q15_MULTIPLIER = 1 << Q15_TO_S32_LEFT_SHIFT
_Q15_DTYPE = np.dtype("<i2")
_S32_DTYPE = np.dtype("<i4")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

_EXPECTED_AUDIO = {
    "sample_rate_hz": SAMPLE_RATE,
    "block_size": BLOCK_SIZE,
    "latency": "low",
    "clock_domain": "APE_PLL_A_SHARED",
}
_EXPECTED_INPUT = {
    "card": "APE",
    "pcm": 1,
    "channels": 2,
    "format": "S32_LE",
    "route": "I2S2_ADMAIF2_ERR_REF",
}
_EXPECTED_OUTPUT = {
    "card": "APE",
    "pcm": 0,
    "channels": 2,
    "format": "S32_LE",
    "route": "ADMAIF1_I2S1_RT5640_J511",
}
_EXPECTED_CHANNELS = {
    "error_mic": 0,
    "reference_mic": 1,
    "noise_out": 0,
    "cancel_out": 1,
}
_EXPECTED_SOURCE_ROLE = "signal_only_no_audio_no_training_authority"
_EXPECTED_SOURCE_LIVE_STATUS = "BLOCKED_NEAR_NYQUIST_PHYSICAL_SAFETY_NOT_ESTABLISHED"


class Stage2ActualPsPlanError(ValueError):
    """Stage-2 actual P/S preparation contract가 바뀌거나 섞인 경우."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        raise Stage2ActualPsPlanError("PCM SHA 입력은 C-contiguous여야 합니다")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _clone_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage2ActualPsPlanError(f"{label}는 mapping이어야 합니다")
    return value


def _require_exact_keys(value: Mapping[str, Any], *, expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise Stage2ActualPsPlanError(
            f"{label} key가 exact하지 않습니다: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _require_exact(value: Any, expected: Any, *, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise Stage2ActualPsPlanError(
            f"{label}가 계약과 다릅니다: expected={expected!r}, got={value!r}"
        )


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise Stage2ActualPsPlanError(f"{label}는 64자리 SHA-256이어야 합니다")
    try:
        int(value, 16)
    except ValueError as error:
        raise Stage2ActualPsPlanError(f"{label}는 hexadecimal SHA-256이어야 합니다") from error
    return value


def _require_q15_stereo(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != _Q15_DTYPE:
        raise Stage2ActualPsPlanError(f"{label}는 exact little-endian int16이어야 합니다")
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] != 2:
        raise Stage2ActualPsPlanError(f"{label}는 nonempty exact [frames,2]이어야 합니다")
    if array.shape[0] % BLOCK_SIZE:
        raise Stage2ActualPsPlanError("Stage-2 actual P/S Q15 frame 수는 256의 배수여야 합니다")
    if not array.flags.c_contiguous:
        raise Stage2ActualPsPlanError(f"{label}는 C-contiguous여야 합니다")
    return array


def _require_s32_stereo(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != _S32_DTYPE:
        raise Stage2ActualPsPlanError(f"{label}는 exact little-endian int32이어야 합니다")
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] != 2:
        raise Stage2ActualPsPlanError(f"{label}는 nonempty exact [frames,2]이어야 합니다")
    if array.shape[0] % BLOCK_SIZE:
        raise Stage2ActualPsPlanError("Stage-2 actual P/S S32 frame 수는 256의 배수여야 합니다")
    if not array.flags.c_contiguous:
        raise Stage2ActualPsPlanError(f"{label}는 C-contiguous여야 합니다")
    return array


def q15_to_stage2_actual_ps_s32_exact(q15_pcm: np.ndarray) -> np.ndarray:
    """Q15 actual-P/S plan을 full-scale-equivalent S32_LE로 정확히 변환한다."""

    q15 = _require_q15_stereo(q15_pcm, label="Stage-2 actual P/S Q15 PCM")
    wide = q15.astype(np.int64) * _Q15_MULTIPLIER
    limits = np.iinfo(np.int32)
    if np.any(wide < limits.min) or np.any(wide > limits.max):
        raise OverflowError("Stage-2 actual P/S Q15→S32 shift가 int32 범위를 벗어났습니다")
    s32 = np.ascontiguousarray(wide.astype(_S32_DTYPE))
    if np.any(np.bitwise_and(s32.astype(np.int64), _Q15_MULTIPLIER - 1)):
        raise AssertionError("Stage-2 actual P/S S32 low 16 bits가 0이 아닙니다")
    if not np.array_equal(
        np.right_shift(s32.astype(np.int64), Q15_TO_S32_LEFT_SHIFT),
        q15.astype(np.int64),
    ):
        raise AssertionError("Stage-2 actual P/S Q15→S32 signed round-trip이 보존되지 않았습니다")
    return s32


def stage2_actual_ps_s32_to_q15_exact(s32_pcm: np.ndarray) -> np.ndarray:
    """정확한 Q15 left-shift S32만 원래 actual-P/S Q15로 역변환한다."""

    s32 = _require_s32_stereo(s32_pcm, label="Stage-2 actual P/S S32 PCM")
    if np.any(np.bitwise_and(s32.astype(np.int64), _Q15_MULTIPLIER - 1)):
        raise Stage2ActualPsPlanError("Stage-2 actual P/S S32 low 16 bits가 0이 아닙니다")
    wide = np.right_shift(s32.astype(np.int64), Q15_TO_S32_LEFT_SHIFT)
    limits = np.iinfo(np.int16)
    if np.any(wide < limits.min) or np.any(wide > limits.max):
        raise Stage2ActualPsPlanError("Stage-2 actual P/S S32 inverse가 int16 범위를 벗어났습니다")
    return np.ascontiguousarray(wide.astype(_Q15_DTYPE))


def _default_config_path(repository_root: Path) -> Path:
    return (repository_root / DEFAULT_CONFIG_RELATIVE_PATH).resolve()


def validate_stage2_actual_ps_static_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """same-card S32 target만 허용하는 정적 config를 fail-closed로 검사한다."""

    root = _require_mapping(payload, label="Stage-2 actual P/S RT5640 config")
    _require_exact_keys(
        root,
        expected={"schema", "audio", "channels", "stage2_2khz", "authority"},
        label="Stage-2 actual P/S RT5640 config",
    )
    _require_exact(root["schema"], CONFIG_SCHEMA, label="schema")

    audio = _require_mapping(root["audio"], label="audio")
    _require_exact_keys(
        audio,
        expected={"sample_rate_hz", "block_size", "latency", "clock_domain", "input", "output"},
        label="audio",
    )
    for key, expected in _EXPECTED_AUDIO.items():
        _require_exact(audio[key], expected, label=f"audio.{key}")
    for label, expected_port in (("audio.input", _EXPECTED_INPUT), ("audio.output", _EXPECTED_OUTPUT)):
        port = _require_mapping(audio[label.rsplit(".", 1)[1]], label=label)
        _require_exact_keys(port, expected=set(expected_port), label=label)
        for key, expected in expected_port.items():
            _require_exact(port[key], expected, label=f"{label}.{key}")

    channels = _require_mapping(root["channels"], label="channels")
    _require_exact_keys(channels, expected=set(_EXPECTED_CHANNELS), label="channels")
    for key, expected in _EXPECTED_CHANNELS.items():
        _require_exact(channels[key], expected, label=f"channels.{key}")

    contract = Stage2TwoKilohertzContract.canonical()
    stage2 = _require_mapping(root["stage2_2khz"], label="stage2_2khz")
    expected_stage2 = {
        "generation": "rt5640_stage2_2khz_actual_ps_s32_v1",
        "contract_id": contract.contract_id,
        "contract_sha256": contract.digest(),
        "source_measurement_plan_schema": SOURCE_MEASUREMENT_PLAN_SCHEMA,
        "source_measurement_plan_role": _EXPECTED_SOURCE_ROLE,
        "source_transport_inherited": False,
        "q15_to_s32_left_shift": Q15_TO_S32_LEFT_SHIFT,
        "raw_target_relative_path": RAW_TARGET_RELATIVE_PATH,
        "analysis_target_relative_path": ANALYSIS_TARGET_RELATIVE_PATH,
        "usb_ab13x_receipt_reuse_allowed": False,
        "output_master_receipt_reuse_allowed": False,
        "fallback_plan_reuse_allowed": False,
        "s16_transport_receipt_reuse_allowed": False,
        "live_jack_allowed_states": ["HP", "HS"],
    }
    _require_exact_keys(stage2, expected=set(expected_stage2), label="stage2_2khz")
    for key, expected in expected_stage2.items():
        _require_exact(stage2[key], expected, label=f"stage2_2khz.{key}")

    authority = _require_mapping(root["authority"], label="authority")
    expected_authority = {
        "plan_preparation_only": True,
        "j511_connection_observed": False,
        "same_card_s32_transport_pass": False,
        "post_start_hw_params_route_receipt_pass": False,
        "raw_first_capture_published_no_replace": False,
        "shared_clock_or_fixed_lti_condition_pass": False,
        "stage2_ps_identification_pass": False,
        "physical_ps_authority": False,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
    }
    _require_exact_keys(authority, expected=set(expected_authority), label="authority")
    for key, expected in expected_authority.items():
        _require_exact(authority[key], expected, label=f"authority.{key}")

    return {
        "schema": STATIC_RECEIPT_SCHEMA,
        "status": "PLAN_ONLY_ACTUAL_PS_CAPTURE_NOT_AUTHORIZED",
        "audio_opened": False,
        "speaker_output": False,
        "results_written": False,
        "hardware_audio": {
            **_EXPECTED_AUDIO,
            "input": dict(_EXPECTED_INPUT),
            "output": dict(_EXPECTED_OUTPUT),
            "channels": dict(_EXPECTED_CHANNELS),
        },
        "stage2_contract_sha256": contract.digest(),
        "forbidden_source_or_receipt_origins": {
            "usb_ab13x": True,
            "output_master_split_clock": True,
            "bandlimited_fallback": True,
            "s16_transport": True,
            "legacy_relabel_or_promotion": True,
        },
        "authority": expected_authority,
        "config_payload_sha256": _payload_sha256(dict(root)),
    }


def load_stage2_actual_ps_static_config(
    path: str | Path | None = None, *, repository_root: str | Path = _REPOSITORY_ROOT
) -> dict[str, Any]:
    """정확한 default config만 read-only로 검증한다."""

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
        raise Stage2ActualPsPlanError("Stage-2 actual P/S는 sealed default config만 허용합니다")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise Stage2ActualPsPlanError("Stage-2 actual P/S config YAML root는 mapping이어야 합니다")
    receipt = validate_stage2_actual_ps_static_config(loaded)
    return {
        **receipt,
        "config": {
            "path": str(config_path),
            "file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
    }


@lru_cache(maxsize=1)
def _authoritative_source_cache() -> tuple[dict[str, Any], np.ndarray]:
    """fallback 없이 time-separated full-PE source plan을 canonicalize한다."""

    source_plan, source_pcm = build_stage2_v2_signal_plan()
    try:
        canonical_plan, canonical_pcm = validate_stage2_v2_signal_plan(source_plan, source_pcm)
    except Stage2MeasurementV2Error as error:
        raise Stage2ActualPsPlanError("authoritative Stage-2 source plan 검증 실패") from error
    if canonical_plan.get("schema") != SOURCE_MEASUREMENT_PLAN_SCHEMA:
        raise Stage2ActualPsPlanError("actual P/S source schema가 full-PE Stage-2 plan이 아닙니다")
    if canonical_plan.get("role") != _EXPECTED_SOURCE_ROLE:
        raise Stage2ActualPsPlanError("actual P/S source role이 signal-only source 계약과 다릅니다")
    if canonical_plan.get("live_safety", {}).get("status") != _EXPECTED_SOURCE_LIVE_STATUS:
        raise Stage2ActualPsPlanError("actual P/S source의 physical-safety block이 사라졌습니다")
    if canonical_plan.get("live_safety", {}).get("audio_execution_allowed_by_this_plan") is not False:
        raise Stage2ActualPsPlanError("actual P/S source가 출력 authority를 가져서는 안 됩니다")
    if canonical_plan.get("live_safety", {}).get("actual_acoustic_ps_or_clock_authority_claimed") is not False:
        raise Stage2ActualPsPlanError("actual P/S source가 physical authority를 주장합니다")
    q15 = _require_q15_stereo(canonical_pcm, label="authoritative actual P/S Q15 source")
    if canonical_plan.get("actual_submitted_pcm", {}).get("sha256") != _array_sha256(q15):
        raise Stage2ActualPsPlanError("actual P/S source plan/PCM SHA가 다릅니다")
    if canonical_plan.get("contract", {}).get("sha256") != Stage2TwoKilohertzContract.canonical().digest():
        raise Stage2ActualPsPlanError("actual P/S source의 Stage-2 contract SHA가 다릅니다")
    return _clone_mapping(canonical_plan), np.array(q15, copy=True, order="C")


def _authoritative_source() -> tuple[dict[str, Any], np.ndarray]:
    source_plan, q15 = _authoritative_source_cache()
    return deepcopy(source_plan), np.array(q15, copy=True, order="C")


def _slot_projection(source_plan: Mapping[str, Any], q15: np.ndarray, s32: np.ndarray) -> list[dict[str, Any]]:
    rows = [row for row in source_plan["layout"] if row.get("kind") == "pe_slot"]
    expected_order = [(role, path) for role in (*FIT_ROLES, HOLDOUT_ROLE) for path in PATHS]
    observed_order = [(str(row.get("role")), str(row.get("path"))) for row in rows]
    # source builder order is intentionally non-symmetric for PE decorrelation; exact order is a contract.
    source_order = [
        ("fit_a", "primary"),
        ("fit_a", "secondary"),
        ("fit_b", "secondary"),
        ("fit_b", "primary"),
        (HOLDOUT_ROLE, "secondary"),
        (HOLDOUT_ROLE, "primary"),
    ]
    if set(observed_order) != set(expected_order) or observed_order != source_order:
        raise Stage2ActualPsPlanError("actual P/S source PE time-role/path 순서가 다릅니다")
    result: list[dict[str, Any]] = []
    for sequence, row in enumerate(rows):
        role, path = str(row["role"]), str(row["path"])
        output_channel = PATH_CHANNEL[path]
        start, stop = int(row["start_frame"]), int(row["stop_frame"])
        central_start, central_stop = int(row["central_start_frame"]), int(row["central_stop_frame"])
        source_q15_slot = np.ascontiguousarray(q15[start:stop])
        planned_s32_slot = np.ascontiguousarray(s32[start:stop])
        if _array_sha256(source_q15_slot) != row.get("submitted_slot_sha256"):
            raise Stage2ActualPsPlanError("actual P/S source PE slot Q15 SHA가 layout과 다릅니다")
        result.append(
            {
                "sequence": sequence,
                "time_role": role,
                "path": path,
                "stimulus_role": "NS" if path == "primary" else "CS",
                "output_channel": output_channel,
                "opposite_output_channel": 1 - output_channel,
                "source_slot_start_frame": start,
                "source_slot_stop_frame": stop,
                "source_central_start_frame": central_start,
                "source_central_stop_frame": central_stop,
                "source_slot_frames": stop - start,
                "source_central_frames": central_stop - central_start,
                "source_q15_slot_sha256": _array_sha256(source_q15_slot),
                "planned_s32_slot_sha256": _array_sha256(planned_s32_slot),
                "opposite_main_channel_exact_zero_except_disjoint_pilot": True,
                "fit_input_allowed": role in FIT_ROLES,
                "holdout_prediction_only": role == HOLDOUT_ROLE,
            }
        )
    return result


def _source_projection(source_plan: Mapping[str, Any], q15: np.ndarray, slot_projection: list[dict[str, Any]]) -> dict[str, Any]:
    layout_payload = {"pe_slots": slot_projection}
    return {
        "schema": SOURCE_MEASUREMENT_PLAN_SCHEMA,
        "role": _EXPECTED_SOURCE_ROLE,
        "canonical_payload_sha256": source_plan["canonical_payload_sha256"],
        "actual_submitted_pcm_sha256": _array_sha256(q15),
        "actual_submitted_pcm_shape": list(q15.shape),
        "actual_submitted_pcm_dtype": q15.dtype.str,
        "stage2_contract_sha256": source_plan["contract"]["sha256"],
        "source_transport_inherited": False,
        "source_usb_or_s16_receipt_usable": False,
        "source_output_master_receipt_usable": False,
        "source_fallback_plan_usable": False,
        "source_audio_execution_allowed": False,
        "source_physical_ps_authority": False,
        "time_role_channel_mapping_sha256": _payload_sha256(layout_payload),
        "time_role_channel_mapping": slot_projection,
    }


def build_stage2_actual_ps_excitation_plan() -> tuple[dict[str, Any], np.ndarray]:
    """full-PE Stage-2 source를 RT5640/J511 S32 P/S plan으로 준비한다.

    이 함수는 plan과 PCM을 메모리에서만 만들며, live backend/ALSA, raw write, speaker
    output과 무관하다.
    """

    static = load_stage2_actual_ps_static_config(repository_root=_REPOSITORY_ROOT)
    if static["authority"]["canonical_training_eligible"] is not False:
        raise Stage2ActualPsPlanError("actual P/S static config가 training authority로 누수됐습니다")
    source_plan, q15 = _authoritative_source()
    s32 = q15_to_stage2_actual_ps_s32_exact(q15)
    if not np.array_equal(stage2_actual_ps_s32_to_q15_exact(s32), q15):
        raise AssertionError("actual P/S S32 inverse가 authoritative Q15 source와 다릅니다")
    slots = _slot_projection(source_plan, q15, s32)
    source = _source_projection(source_plan, q15, slots)
    contract = Stage2TwoKilohertzContract.canonical()
    s32_abs = np.abs(s32.astype(np.int64))
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "status": "PLAN_ONLY_ACTUAL_PS_CAPTURE_NOT_AUTHORIZED",
        "role": "actual_ps_excitation_preparation_no_audio_no_training_authority",
        "sample_rate_hz": SAMPLE_RATE,
        "block_size": BLOCK_SIZE,
        "duration_seconds": len(s32) / SAMPLE_RATE,
        "expected_callbacks": len(s32) // BLOCK_SIZE,
        "stage2_contract": {
            "id": contract.contract_id,
            "sha256": contract.digest(),
        },
        "rt5640_static_config": {
            "schema": static["schema"],
            "config_payload_sha256": static["config_payload_sha256"],
            "config_file_sha256": static["config"]["file_sha256"],
            "hardware_audio": static["hardware_audio"],
            "forbidden_source_or_receipt_origins": static["forbidden_source_or_receipt_origins"],
        },
        "source_measurement_plan": source,
        "ps_time_role_channel_mapping": {
            "primary": {
                "stimulus_role": "NS",
                "output_channel": 0,
                "response_capture_channels": {"ERR": 0, "REF": 1},
                "fit_time_roles": list(FIT_ROLES),
                "holdout_time_role": HOLDOUT_ROLE,
            },
            "secondary": {
                "stimulus_role": "CS",
                "output_channel": 1,
                "response_capture_channels": {"ERR": 0, "REF": 1},
                "fit_time_roles": list(FIT_ROLES),
                "holdout_time_role": HOLDOUT_ROLE,
            },
            "same_capture_required": True,
            "fit_a_fit_b_independent_crosscheck_required": True,
            "untouched_holdout_refit_or_support_selection_forbidden": True,
        },
        "quantization": {
            "source_dtype": _Q15_DTYPE.str,
            "output_dtype": _S32_DTYPE.str,
            "conversion": "int64_multiply_then_int32_range_checked_exact_signed_left_shift",
            "left_shift_bits": Q15_TO_S32_LEFT_SHIFT,
            "multiplier": _Q15_MULTIPLIER,
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
            "q15_inverse_sha256": _array_sha256(stage2_actual_ps_s32_to_q15_exact(s32)),
        },
        "future_raw_targets": {
            "native_raw_relative_path": RAW_TARGET_RELATIVE_PATH,
            "analysis_receipt_relative_path": ANALYSIS_TARGET_RELATIVE_PATH,
            "file_created_by_this_module": False,
            "raw_schema_created_by_this_module": False,
            "no_replace_required_before_analysis": True,
        },
        "conditional_actual_ps_authority_requirements": {
            "clean_exact_commit_and_default_dry_run_pass": True,
            "rt5640_s32_read_only_preflight_pass": True,
            "j511_HP_or_HS_three_read_only_samples": True,
            "pcm_global_unoccupied_before_open": True,
            "same_card_ape_pcm1_input_and_pcm0_output_S32_48k_256": True,
            "pre_arm_output_exact_zero": True,
            "post_start_hw_params_route_and_occupancy_receipt": True,
            "actual_submitted_s32_sha256_and_q15_inverse_sha256_match": True,
            "native_raw_first_no_replace_publication": True,
            "xrun_clip_callback_status_sample_slip_drop_add_exact_zero": True,
            "shared_clock_or_predeclared_fixed_lti_condition_receipt": True,
            "fit_a_fit_b_holdout_and_stage2_2khz_analysis_pass": True,
            "independent_review_and_explicit_plant_binding": True,
        },
        "authority": {
            "actual_ps_excitation_plan_prepared": True,
            "audio_output_performed": False,
            "same_card_s32_transport_pass": False,
            "physical_raw_present": False,
            "physical_ps_authority": False,
            "canonical_training_eligible": False,
            "deployment_eligible": False,
        },
    }
    plan["canonical_payload_sha256"] = _payload_sha256(plan)
    s32.setflags(write=False)
    return plan, s32


def validate_stage2_actual_ps_excitation_plan(
    plan: Mapping[str, Any], planned_s32_pcm: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    """다른 source/slot/route/quantization이 섞인 actual-P/S plan을 거부한다."""

    if not isinstance(plan, Mapping):
        raise Stage2ActualPsPlanError("actual P/S S32 plan은 mapping이어야 합니다")
    source = _require_s32_stereo(planned_s32_pcm, label="actual P/S planned S32 PCM")
    expected_plan, expected_pcm = build_stage2_actual_ps_excitation_plan()
    if dict(plan) != expected_plan:
        raise Stage2ActualPsPlanError(
            "actual P/S S32 plan이 source/fallback/USB/output-master 또는 mapping과 다릅니다"
        )
    if source.shape != expected_pcm.shape or not np.array_equal(source, expected_pcm):
        raise Stage2ActualPsPlanError("actual P/S planned S32 PCM이 canonical bytes와 다릅니다")
    if _array_sha256(source) != plan["sealed_planned_s32_pcm"]["sha256"]:
        raise Stage2ActualPsPlanError("actual P/S planned S32 SHA가 plan과 다릅니다")
    restored = stage2_actual_ps_s32_to_q15_exact(source)
    if _array_sha256(restored) != plan["source_measurement_plan"]["actual_submitted_pcm_sha256"]:
        raise Stage2ActualPsPlanError("actual P/S S32 inverse source SHA가 다릅니다")
    _require_sha256(plan.get("canonical_payload_sha256"), label="actual P/S plan SHA")
    return expected_plan, expected_pcm


def build_stage2_actual_ps_planned_provenance(
    plan: Mapping[str, Any], planned_s32_pcm: np.ndarray
) -> dict[str, Any]:
    """후속 live adapter가 보존할 plan/source/route raw-first binding을 만든다."""

    canonical, pcm = validate_stage2_actual_ps_excitation_plan(plan, planned_s32_pcm)
    source = canonical["source_measurement_plan"]
    static = canonical["rt5640_static_config"]
    provenance: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "status": "PLANNED_ONLY_ACTUAL_PS_CAPTURE_NOT_AUTHORIZED",
        "actual_ps_plan_sha256": canonical["canonical_payload_sha256"],
        "source_measurement_plan_sha256": source["canonical_payload_sha256"],
        "source_time_role_channel_mapping_sha256": source["time_role_channel_mapping_sha256"],
        "stage2_contract_sha256": canonical["stage2_contract"]["sha256"],
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
        "source": {
            "schema": source["schema"],
            "role": source["role"],
            "actual_submitted_q15_sha256": source["actual_submitted_pcm_sha256"],
            "actual_submitted_q15_shape": source["actual_submitted_pcm_shape"],
            "transport_inherited": False,
            "physical_authority_inherited": False,
        },
        "planned_s32": {
            **canonical["sealed_planned_s32_pcm"],
            "low_16_bits_zero": True,
        },
        "prohibited_receipt_lineage": static["forbidden_source_or_receipt_origins"],
        "actual_capture_present": False,
        "actual_audio_output_claimed": False,
        "actual_ps_or_training_authority_claimed": False,
        "remaining_adapter_work": [
            "PASS-only assert_rt5640_stage2_s32_preflight receipt before any backend import",
            "same-card S32 duplex backend adapter with pre-arm exact-zero output",
            "post-start negotiated hw_params/route/occupancy receipt",
            "native S32 raw-first no-replace publisher and partial-failure preservation",
            "Q15-inverse raw analyzer with shared-clock/fixed-LTI and Stage-2 P/S gates",
            "independent review and explicit plant binding before training admission",
        ],
    }
    provenance["canonical_payload_sha256"] = _payload_sha256(provenance)
    return provenance


def validate_stage2_actual_ps_planned_provenance(
    provenance: Mapping[str, Any], plan: Mapping[str, Any], planned_s32_pcm: np.ndarray
) -> dict[str, Any]:
    """future adapter가 다른 plan/transport/receipt로 바꾸지 못하게 한다."""

    if not isinstance(provenance, Mapping):
        raise Stage2ActualPsPlanError("actual P/S planned provenance는 mapping이어야 합니다")
    expected = build_stage2_actual_ps_planned_provenance(plan, planned_s32_pcm)
    if dict(provenance) != expected:
        raise Stage2ActualPsPlanError(
            "actual P/S provenance가 USB/output-master/fallback/다른 source와 섞였습니다"
        )
    _require_sha256(provenance.get("canonical_payload_sha256"), label="actual P/S provenance SHA")
    return expected


__all__ = [
    "ANALYSIS_TARGET_RELATIVE_PATH",
    "CONFIG_SCHEMA",
    "DEFAULT_CONFIG_RELATIVE_PATH",
    "PLAN_SCHEMA",
    "PROVENANCE_SCHEMA",
    "Q15_TO_S32_LEFT_SHIFT",
    "RAW_TARGET_RELATIVE_PATH",
    "STATIC_RECEIPT_SCHEMA",
    "Stage2ActualPsPlanError",
    "build_stage2_actual_ps_excitation_plan",
    "build_stage2_actual_ps_planned_provenance",
    "load_stage2_actual_ps_static_config",
    "q15_to_stage2_actual_ps_s32_exact",
    "stage2_actual_ps_s32_to_q15_exact",
    "validate_stage2_actual_ps_excitation_plan",
    "validate_stage2_actual_ps_planned_provenance",
    "validate_stage2_actual_ps_static_config",
]
