"""Stage-2 RT5640/J511 S32 P/S capture adapter의 fail-closed scaffold.

이 모듈은 같은 APE card의 ``pcm1``(ERR/REF 입력)과 ``pcm0``(J511 출력)을
48 kHz / 256 frame / S32_LE로 묶을 *후속* capture adapter의 단일 준비 경계다.
현재 source plan은 명시적으로 ``signal_only_no_audio_no_training_authority``이므로
이 모듈은 실제 backend, ALSA, PortAudio를 import하거나 열지 않는다. ``execute``
entry point도 현재는 backend에 닿기 전에 항상 실패한다.

그 경계를 유지하는 이유는 간단하다. USB AB13X/S16 output-master diagnostic을
같은-card S32 P/S로 재표기하면 high-frequency phase claim이 거짓이 된다. 실제
출력은 별도 actual P/S plan, J511 HP/HS read-only 확인, stream-start 뒤 negotiated
hw_params/route/occupancy receipt가 함께 생긴 뒤의 별도 변경에서만 열 수 있다.

``deep_anc.audio_duplex_s32_disarmed_v10_3`` primitive는 그 미래 경로에서만
사용하도록 연결해 두었다. 해당 primitive는 arm 전 callback output을 exact zero로
보장한다. 현재 authority에서는 그 호출점까지 도달할 수 없다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from deep_anc.audio_duplex_s32_disarmed_v10_3 import capture_disarmed_planned_s32_duplex

from .stage2_2khz_rt5640_s32 import (
    PROVENANCE_SCHEMA,
    SIGNAL_PLAN_SCHEMA,
    Stage2Rt5640S32ContractError,
    build_stage2_rt5640_s32_planned_transport_provenance,
    build_stage2_rt5640_s32_signal_plan,
    load_stage2_rt5640_s32_static_contract,
    validate_stage2_rt5640_s32_planned_transport_provenance,
    validate_stage2_rt5640_s32_signal_plan,
)


CAPTURE_SCAFFOLD_SCHEMA = "stage2_2khz_rt5640_s32_capture_scaffold_v1"
RAW_SCHEMA = "stage2_2khz_rt5640_s32_raw_v1"
BLOCKED_STATUS = "BLOCKED_SEPARATE_ACTUAL_PS_AUTHORITY_AND_J511_CHECK_REQUIRED"

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


class Stage2Rt5640S32CaptureBlocked(RuntimeError):
    """현재 signal-only plan으로 live capture를 시도한 경우의 fail-closed error."""


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


def _prepare_stage2_rt5640_s32_capture() -> tuple[dict[str, Any], np.ndarray]:
    """sealed static/signal/provenance를 재검산하고, 무음 receipt와 PCM을 반환한다.

    반환 PCM은 이 함수 안에서만 future disarmed primitive에 전달될 수 있다. 현재
    receipt는 ``live_capture_may_open=False``라서 public execute 함수는 그 호출점에
    도달하지 않는다.
    """

    static = load_stage2_rt5640_s32_static_contract()
    plan, pcm = build_stage2_rt5640_s32_signal_plan()
    canonical_plan, canonical_pcm = validate_stage2_rt5640_s32_signal_plan(plan, pcm)
    provenance = build_stage2_rt5640_s32_planned_transport_provenance(
        canonical_plan, canonical_pcm
    )
    validate_stage2_rt5640_s32_planned_transport_provenance(
        provenance, canonical_plan, canonical_pcm
    )

    hardware = static["hardware_audio"]
    if hardware["input"] != _EXPECTED_INPUT or hardware["output"] != _EXPECTED_OUTPUT:
        raise Stage2Rt5640S32ContractError("Stage-2 capture scaffold의 APE PCM route가 다릅니다")
    if (
        hardware["sample_rate_hz"] != 48_000
        or hardware["block_size"] != 256
        or hardware["latency"] != "low"
        or hardware["clock_domain"] != "APE_PLL_A_SHARED"
    ):
        raise Stage2Rt5640S32ContractError("Stage-2 capture scaffold의 S32 transport 조건이 다릅니다")
    if canonical_plan["schema"] != SIGNAL_PLAN_SCHEMA:
        raise Stage2Rt5640S32ContractError("Stage-2 S32 signal plan schema가 다릅니다")
    if canonical_plan["role"] != "signal_only_no_audio_no_training_authority":
        raise Stage2Rt5640S32ContractError("Stage-2 source plan role이 signal-only가 아닙니다")
    if canonical_plan["authority"] != {
        "signal_plan_pass": True,
        "s32_duplex_transport_pass": False,
        "same_hardware_frame_identity_pass": False,
        "relative_clock_or_fixed_lti_condition_pass": False,
        "stage2_ps_identification_pass": False,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
    }:
        raise Stage2Rt5640S32ContractError("Stage-2 signal plan authority가 예상과 다릅니다")
    if provenance["schema"] != PROVENANCE_SCHEMA:
        raise Stage2Rt5640S32ContractError("Stage-2 planned provenance schema가 다릅니다")
    if provenance["status"] != "PLANNED_ONLY_NO_AUDIO_NO_CAPTURE":
        raise Stage2Rt5640S32ContractError("Stage-2 planned provenance가 capture를 주장합니다")
    if provenance["actual_capture_present"] is not False:
        raise Stage2Rt5640S32ContractError("Stage-2 planned provenance가 actual capture를 주장합니다")
    if provenance["actual_audio_output_claimed"] is not False:
        raise Stage2Rt5640S32ContractError("Stage-2 planned provenance가 audio output을 주장합니다")
    if provenance["actual_ps_or_training_authority_claimed"] is not False:
        raise Stage2Rt5640S32ContractError("Stage-2 planned provenance가 P/S 또는 training authority를 주장합니다")

    core: dict[str, Any] = {
        "schema": CAPTURE_SCAFFOLD_SCHEMA,
        "status": BLOCKED_STATUS,
        "dry_run": True,
        "audio_backend_imported": False,
        "alsa_or_pcm_opened": False,
        "speaker_output": False,
        "raw_written": False,
        "live_capture_may_open": False,
        "hardware_audio": {
            "sample_rate_hz": 48_000,
            "block_size": 256,
            "latency": "low",
            "clock_domain": "APE_PLL_A_SHARED",
            "input": dict(_EXPECTED_INPUT),
            "output": dict(_EXPECTED_OUTPUT),
        },
        "planned_signal": {
            "plan_schema": canonical_plan["schema"],
            "plan_payload_sha256": canonical_plan["canonical_payload_sha256"],
            "planned_s32_pcm_sha256": canonical_plan["sealed_planned_s32_pcm"]["sha256"],
            "planned_s32_shape": canonical_plan["sealed_planned_s32_pcm"]["shape"],
            "planned_s32_dtype": canonical_plan["sealed_planned_s32_pcm"]["dtype"],
            "expected_callbacks": canonical_plan["expected_callbacks"],
            "low_16_bits_must_be_zero": True,
            "source_audio_execution_allowed": False,
            "source_canonical_training_eligible": False,
        },
        "planned_transport_provenance": {
            "schema": provenance["schema"],
            "payload_sha256": provenance["canonical_payload_sha256"],
            "status": provenance["status"],
            "same_card": provenance["hardware_contract"]["same_card"],
            "same_clock_domain": provenance["hardware_contract"]["same_clock_domain"],
            "native_format": provenance["hardware_contract"]["native_format"],
            "prohibited_receipt_lineage": provenance["prohibited_receipt_lineage"],
        },
        "disarmed_primitive": {
            "module": "deep_anc.audio_duplex_s32_disarmed_v10_3",
            "entry_point": "capture_disarmed_planned_s32_duplex",
            "pre_arm_output_exact_zero_required": True,
            "post_start_hw_params_route_j511_occupancy_check_required": True,
            "nonzero_assignment_before_arm_allowed": False,
            "current_plan_can_reach_primitive": False,
        },
        "raw_first_publication_plan": {
            "schema": RAW_SCHEMA,
            "target_relative_path": canonical_plan["future_raw_target"]["relative_path"],
            "dry_run_must_not_write": True,
            "actual_capture_present": False,
            "analysis_before_raw_publication_allowed": False,
        },
        "required_before_separate_live_implementation": [
            "actual_stage2_2khz_P_S_plan_with_explicit_live_authority",
            "J511_HP_or_HS_three_read_only_samples",
            "pre_open_APE_pcm1_pcm0_occupancy_and_route_snapshot",
            "post_start_negotiated_APE_pcm1_pcm0_S32_LE_48000_256_receipt",
            "post_start_J511_route_and_occupancy_receipt_before_arm",
            "raw_first_no_replace_capture_publication",
            "shared_q_or_fixed_LTI_conditional_clock_receipt",
            "stage2_2khz_P_S_analysis_and_plant_binding",
        ],
        "authority": {
            "s32_duplex_transport_pass": False,
            "same_hardware_frame_identity_pass": False,
            "relative_clock_or_fixed_lti_condition_pass": False,
            "stage2_ps_identification_pass": False,
            "canonical_training_eligible": False,
            "deployment_eligible": False,
        },
    }
    return {**core, "canonical_payload_sha256": _payload_sha256(core)}, canonical_pcm


def build_stage2_rt5640_s32_capture_dry_run_receipt() -> dict[str, Any]:
    """무음 Stage-2 S32 capture scaffold receipt를 만든다.

    PCM device/backend/result file은 열거나 쓰지 않는다. 이 receipt의 success는
    physical transport 또는 plant authority가 아니라, 현재 signal-only plan을
    output-capable capture로 오승격하지 않았다는 뜻이다.
    """

    receipt, _ = _prepare_stage2_rt5640_s32_capture()
    return receipt


def assert_stage2_rt5640_s32_live_capture_blocked() -> None:
    """현 signal-only plan으로 live backend 경로를 열지 못하게 한다."""

    receipt = build_stage2_rt5640_s32_capture_dry_run_receipt()
    if receipt["live_capture_may_open"] is not False:
        raise AssertionError("Stage-2 capture scaffold live gate가 fail-closed가 아닙니다")
    raise Stage2Rt5640S32CaptureBlocked(
        f"{BLOCKED_STATUS}: 현재 plan은 signal-only이며 actual P/S authority, "
        "J511 HP/HS 확인, post-start S32 route receipt가 없습니다; "
        "audio backend import/open은 수행하지 않았습니다"
    )


def execute_stage2_rt5640_s32_disarmed_capture(
    backend: Any,
    *,
    input_device: int,
    output_device: int,
    post_start_pre_arm_check: Callable[[], None],
    pre_open_check: Callable[[], None] | None = None,
    on_output_closed: Callable[[bool], None] | None = None,
    watchdog_grace_seconds: float = 2.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """future actual-P/S entry point; 현재는 stream constructor 전에 항상 차단한다.

    함수 signature에는 동일-card S32 disarmed primitive의 exact inputs만 남겼다.
    future separate authority implementation이 이 entry point를 열기 전에는
    :func:`assert_stage2_rt5640_s32_live_capture_blocked`가 backend object을 읽지도
    않고 raise한다. 따라서 fake backend도 stream을 만들 수 없고, real backend도
    import/open될 수 없다.
    """

    # 반드시 backend/stream/PCM validation보다 먼저 실행한다. 현재 plan에는 live
    # audio 권한이 없으므로 불완전한 user input조차 audio-open 시도로 이어지지 않는다.
    assert_stage2_rt5640_s32_live_capture_blocked()

    # 현재는 unreachable이다. 별도 actual P/S authority change가 이 branch를 열 때도
    # Q15→S32 sealed PCM, 48 kHz/256, pre-arm zero guarantee를 우회하지 않게 보존한다.
    _receipt, planned_s32_pcm = _prepare_stage2_rt5640_s32_capture()
    return capture_disarmed_planned_s32_duplex(
        backend,
        planned_pcm=planned_s32_pcm,
        input_device=input_device,
        output_device=output_device,
        post_start_pre_arm_check=post_start_pre_arm_check,
        pre_open_check=pre_open_check,
        on_output_closed=on_output_closed,
        watchdog_grace_seconds=watchdog_grace_seconds,
    )


__all__ = [
    "BLOCKED_STATUS",
    "CAPTURE_SCAFFOLD_SCHEMA",
    "RAW_SCHEMA",
    "Stage2Rt5640S32CaptureBlocked",
    "assert_stage2_rt5640_s32_live_capture_blocked",
    "build_stage2_rt5640_s32_capture_dry_run_receipt",
    "execute_stage2_rt5640_s32_disarmed_capture",
]
