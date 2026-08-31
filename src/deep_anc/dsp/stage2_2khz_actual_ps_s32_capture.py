"""Stage-2 2 kHz actual P/S S32 capture의 순수 fail-closed 준비 경계.

이 모듈은 ``stage2_2khz_actual_ps_plan``의 time-separated full-PE plan과
planned provenance만 받아 후속 same-card S32 raw-first adapter가 지켜야 할
admission receipt를 만든다. PCM/ALSA/PortAudio backend를 import하거나 열지 않고,
파일을 쓰거나 스피커를 출력하지도 않는다.

특히 기존 signal-only/fallback/output-master/USB capture scaffold를 이 경로에
연결하지 않는다. 실제 live adapter는 별도 변경에서 actual-config read-only
preflight와 명시적인 사용자 출력 승인을 통과한 뒤에만 **zero-only** stream을 시작할 수
있고, stream-start 뒤 pre-arm receipt가 PASS하기 전에는 nonzero output arm 또는 raw
publisher를 절대 시작할 수 없다. 이 준비 module에는 backend/raw publisher/live execution이
의도적으로 아직 없다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from .stage2_2khz_actual_ps_plan import (
    PLAN_SCHEMA,
    PROVENANCE_SCHEMA,
    Stage2ActualPsPlanError,
    STATIC_RECEIPT_SCHEMA,
    build_stage2_actual_ps_excitation_plan,
    build_stage2_actual_ps_planned_provenance,
    validate_stage2_actual_ps_excitation_plan,
    validate_stage2_actual_ps_planned_provenance,
)


CAPTURE_SCAFFOLD_SCHEMA = "stage2_2khz_actual_ps_s32_capture_scaffold_v1"
RAW_SCHEMA = "stage2_2khz_actual_ps_s32_native_raw_v1"
ACTUAL_CONFIG_PREFLIGHT_SCHEMA = "rt5640_stage2_s32_read_only_preflight_v1"
ACTUAL_CONFIG_PREFLIGHT_PASS_STATUS = "PASS_READ_ONLY_RT5640_STAGE2_S32_PREFLIGHT"
USER_LIVE_GATE_SCHEMA = "stage2_2khz_actual_ps_s32_explicit_user_live_gate_v1"
POST_START_RECEIPT_SCHEMA = "stage2_2khz_actual_ps_s32_post_start_pre_arm_receipt_v1"
BLOCKED_STATUS = "BLOCKED_ACTUAL_PS_S32_LIVE_ADAPTER_NOT_IMPLEMENTED"

_EXPECTED_AUDIO = {
    "sample_rate_hz": 48_000,
    "block_size": 256,
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
_EXPECTED_ROUTES = {
    "I2S1 Mux": "ADMAIF1",
    "ADMAIF1 Mux": "I2S1",
    "ADMAIF2 Mux": "I2S2",
    "I2S2 Mux": "ADMAIF2",
}
_EXPECTED_FORBIDDEN_ORIGINS = {
    "usb_ab13x": True,
    "output_master_split_clock": True,
    "bandlimited_fallback": True,
    "s16_transport": True,
    "legacy_relabel_or_promotion": True,
}
_EXPECTED_PLAN_ONLY_AUTHORITY = {
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
_EXPECTED_PROHIBITED_TRANSPORTS = {
    "usb_ab13x_selected": False,
    "output_master_split_clock_selected": False,
    "bandlimited_fallback_selected": False,
    "s16_selected": False,
    "contract_forbids_usb_ab13x": True,
    "contract_forbids_output_master_split_clock": True,
    "contract_forbids_bandlimited_fallback": True,
    "contract_forbids_s16": True,
}


class Stage2ActualPsS32CaptureBlocked(RuntimeError):
    """actual P/S live capture가 아직 구현/승인되지 않은 경우의 fail-closed error."""


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


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage2ActualPsS32CaptureBlocked(f"{label}는 mapping이어야 합니다")
    return value


def _require_exact(value: Any, expected: Any, *, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise Stage2ActualPsS32CaptureBlocked(
            f"{label}가 계약과 다릅니다: expected={expected!r}, got={value!r}"
        )


def _require_true(value: Any, *, label: str) -> None:
    _require_exact(value, True, label=label)


def _require_false(value: Any, *, label: str) -> None:
    _require_exact(value, False, label=label)


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise Stage2ActualPsS32CaptureBlocked(f"{label}는 64자리 SHA-256이어야 합니다")
    try:
        int(value, 16)
    except ValueError as error:
        raise Stage2ActualPsS32CaptureBlocked(f"{label}는 hexadecimal SHA-256이어야 합니다") from error
    return value


def _require_empty_sequence(value: Any, *, label: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 0:
        raise Stage2ActualPsS32CaptureBlocked(f"{label}는 빈 list 또는 tuple이어야 합니다")


def _hardware_audio_from_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    static = _require_mapping(plan.get("rt5640_static_config"), label="actual P/S static config")
    hardware = _require_mapping(static.get("hardware_audio"), label="actual P/S hardware_audio")
    expected = {
        **_EXPECTED_AUDIO,
        "input": _EXPECTED_INPUT,
        "output": _EXPECTED_OUTPUT,
        "channels": _EXPECTED_CHANNELS,
    }
    if dict(hardware) != expected:
        raise Stage2ActualPsS32CaptureBlocked("actual P/S plan의 same-card S32 hardware mapping이 다릅니다")
    return expected


def _prepare_stage2_actual_ps_s32_capture(
    plan: Mapping[str, Any], planned_s32_pcm: np.ndarray, provenance: Mapping[str, Any]
) -> tuple[dict[str, Any], np.ndarray]:
    """actual full-PE plan/provenance를 봉인한 순수 dry-run receipt를 만든다."""

    try:
        canonical_plan, canonical_pcm = validate_stage2_actual_ps_excitation_plan(plan, planned_s32_pcm)
        canonical_provenance = validate_stage2_actual_ps_planned_provenance(
            provenance, canonical_plan, canonical_pcm
        )
    except Stage2ActualPsPlanError as error:
        raise Stage2ActualPsS32CaptureBlocked(
            f"actual Stage-2 full-PE plan/provenance가 봉인 검증을 통과하지 못했습니다: {error}"
        ) from error

    hardware_audio = _hardware_audio_from_plan(canonical_plan)
    source = _require_mapping(canonical_plan.get("source_measurement_plan"), label="actual P/S source plan")
    _require_exact(
        source.get("schema"),
        "stage2_2khz_time_separated_full_pe_plan_v2",
        label="actual P/S source schema",
    )
    _require_exact(
        source.get("role"),
        "signal_only_no_audio_no_training_authority",
        label="actual P/S source role",
    )
    for key in (
        "source_transport_inherited",
        "source_usb_or_s16_receipt_usable",
        "source_output_master_receipt_usable",
        "source_fallback_plan_usable",
        "source_audio_execution_allowed",
        "source_physical_ps_authority",
    ):
        _require_false(source.get(key), label=f"actual P/S {key}")

    authority = _require_mapping(canonical_plan.get("authority"), label="actual P/S plan authority")
    for key in (
        "audio_output_performed",
        "same_card_s32_transport_pass",
        "physical_raw_present",
        "physical_ps_authority",
        "canonical_training_eligible",
        "deployment_eligible",
    ):
        _require_false(authority.get(key), label=f"actual P/S plan authority.{key}")
    _require_true(
        authority.get("actual_ps_excitation_plan_prepared"),
        label="actual P/S plan authority.actual_ps_excitation_plan_prepared",
    )

    if canonical_plan.get("schema") != PLAN_SCHEMA:
        raise Stage2ActualPsS32CaptureBlocked("actual P/S excitation plan schema가 다릅니다")
    if canonical_provenance.get("schema") != PROVENANCE_SCHEMA:
        raise Stage2ActualPsS32CaptureBlocked("actual P/S planned provenance schema가 다릅니다")
    _require_false(
        canonical_provenance.get("actual_capture_present"),
        label="actual P/S planned provenance.actual_capture_present",
    )
    _require_false(
        canonical_provenance.get("actual_audio_output_claimed"),
        label="actual P/S planned provenance.actual_audio_output_claimed",
    )
    _require_false(
        canonical_provenance.get("actual_ps_or_training_authority_claimed"),
        label="actual P/S planned provenance.actual_ps_or_training_authority_claimed",
    )

    core: dict[str, Any] = {
        "schema": CAPTURE_SCAFFOLD_SCHEMA,
        "status": BLOCKED_STATUS,
        "dry_run": True,
        "audio_backend_imported": False,
        "alsa_pcm_opened": False,
        "speaker_output": False,
        "raw_written": False,
        "live_capture_may_open": False,
        "raw_publisher_implemented": False,
        "live_execution_implemented": False,
        "hardware_audio": hardware_audio,
        "planned_actual_ps": {
            "plan_schema": canonical_plan["schema"],
            "plan_payload_sha256": canonical_plan["canonical_payload_sha256"],
            "planned_s32_pcm_sha256": canonical_plan["sealed_planned_s32_pcm"]["sha256"],
            "planned_s32_shape": canonical_plan["sealed_planned_s32_pcm"]["shape"],
            "planned_s32_dtype": canonical_plan["sealed_planned_s32_pcm"]["dtype"],
            "expected_callbacks": canonical_plan["expected_callbacks"],
            "duration_seconds": canonical_plan["duration_seconds"],
            "source_schema": source["schema"],
            "source_plan_payload_sha256": source["canonical_payload_sha256"],
            "source_time_role_channel_mapping_sha256": source["time_role_channel_mapping_sha256"],
            "source_transport_inherited": False,
            "source_fallback_plan_usable": False,
            "source_audio_execution_allowed": False,
            "low_16_bits_must_be_zero": True,
        },
        "planned_provenance": {
            "schema": canonical_provenance["schema"],
            "payload_sha256": canonical_provenance["canonical_payload_sha256"],
            "actual_ps_plan_sha256": canonical_provenance["actual_ps_plan_sha256"],
            "source_measurement_plan_sha256": canonical_provenance["source_measurement_plan_sha256"],
            "source_time_role_channel_mapping_sha256": canonical_provenance[
                "source_time_role_channel_mapping_sha256"
            ],
            "stage2_contract_sha256": canonical_provenance["stage2_contract_sha256"],
            "actual_capture_present": False,
            "actual_audio_output_claimed": False,
            "actual_ps_or_training_authority_claimed": False,
        },
        "future_live_admission": {
            "actual_config_preflight_receipt_required": True,
            "explicit_user_facing_live_gate_required": True,
            "post_start_pre_arm_receipt_required": True,
            "pre_arm_output_exact_zero_required": True,
            "preflight_and_user_gate_must_pass_before_backend_import_or_stream_open": True,
            "post_start_receipt_must_pass_before_nonzero_output_arm_or_raw_publisher": True,
            "zero_only_stream_start_may_follow_preflight_and_user_gate": True,
            "post_start_receipt_must_be_collected_after_stream_start_before_arm": True,
            "output_duration_seconds": canonical_plan["duration_seconds"],
            "output_close_notice_required": "출력 종료 — 지금 스피커 분리",
            "automatic_retry_or_reoutput_allowed": False,
        },
        "future_raw_first_publication": {
            "schema": RAW_SCHEMA,
            "target_relative_path": canonical_plan["future_raw_targets"]["native_raw_relative_path"],
            "implemented": False,
            "dry_run_must_not_write": True,
            "no_replace_required_before_analysis": True,
        },
        "authority": {
            "actual_config_read_only_preflight_pass": False,
            "explicit_user_live_gate_pass": False,
            "post_start_pre_arm_receipt_pass": False,
            "same_card_s32_transport_pass": False,
            "physical_raw_present": False,
            "physical_ps_authority": False,
            "canonical_training_eligible": False,
            "deployment_eligible": False,
        },
    }
    return {**core, "canonical_payload_sha256": _payload_sha256(core)}, canonical_pcm


def build_stage2_actual_ps_s32_capture_dry_run_receipt() -> dict[str, Any]:
    """backend/PCM/file 접근 없이 actual P/S capture 준비 receipt를 만든다."""

    plan, planned_s32_pcm = build_stage2_actual_ps_excitation_plan()
    provenance = build_stage2_actual_ps_planned_provenance(plan, planned_s32_pcm)
    receipt, _ = _prepare_stage2_actual_ps_s32_capture(plan, planned_s32_pcm, provenance)
    return receipt


def validate_stage2_actual_ps_s32_preflight_receipt(
    receipt: Mapping[str, Any], plan: Mapping[str, Any], planned_s32_pcm: np.ndarray
) -> dict[str, Any]:
    """actual-config에 묶인 read-only preflight PASS만 future adapter에 허용한다.

    이 검증은 preflight 모듈을 import하지 않는다. 같은 top schema라도
    ``actual_ps_config`` binding이 없는 이전 receipt는 명시적으로 거절한다.
    """

    canonical_plan, _ = validate_stage2_actual_ps_excitation_plan(plan, planned_s32_pcm)
    observed = _require_mapping(receipt, label="actual-config preflight receipt")
    _require_exact(observed.get("schema"), ACTUAL_CONFIG_PREFLIGHT_SCHEMA, label="preflight schema")
    _require_exact(
        observed.get("status"), ACTUAL_CONFIG_PREFLIGHT_PASS_STATUS, label="preflight PASS status"
    )
    _require_true(observed.get("passed"), label="preflight passed")
    for key in (
        "dry_run",
        "audio_backend_imported",
        "alsa_pcm_opened",
        "speaker_output",
        "filesystem_write_performed",
    ):
        expected = True if key == "dry_run" else False
        _require_exact(observed.get(key), expected, label=f"preflight {key}")

    actual_config = _require_mapping(observed.get("actual_ps_config"), label="preflight actual_ps_config")
    static = _require_mapping(canonical_plan.get("rt5640_static_config"), label="actual P/S static config")
    _require_exact(
        actual_config.get("schema"), STATIC_RECEIPT_SCHEMA, label="preflight actual config schema"
    )
    _require_exact(
        actual_config.get("status"),
        "PLAN_ONLY_ACTUAL_PS_CAPTURE_NOT_AUTHORIZED",
        label="preflight actual config status",
    )
    for key in ("audio_opened", "speaker_output", "results_written"):
        _require_false(actual_config.get(key), label=f"preflight actual config {key}")
    _require_exact(
        actual_config.get("config_payload_sha256"),
        static["config_payload_sha256"],
        label="preflight actual config payload SHA",
    )
    _require_exact(
        actual_config.get("config_file_sha256"),
        static["config_file_sha256"],
        label="preflight actual config file SHA",
    )
    _require_exact(
        actual_config.get("stage2_contract_sha256"),
        canonical_plan["stage2_contract"]["sha256"],
        label="preflight Stage-2 contract SHA",
    )
    _require_exact(
        actual_config.get("hardware_audio"), _hardware_audio_from_plan(canonical_plan), label="preflight hardware_audio"
    )
    _require_exact(
        actual_config.get("forbidden_source_or_receipt_origins"),
        _EXPECTED_FORBIDDEN_ORIGINS,
        label="preflight prohibited receipt lineage",
    )
    _require_exact(
        actual_config.get("authority"),
        _EXPECTED_PLAN_ONLY_AUTHORITY,
        label="preflight actual config plan-only authority",
    )
    _require_exact(
        actual_config.get("prohibited_transports"),
        _EXPECTED_PROHIBITED_TRANSPORTS,
        label="preflight prohibited transports",
    )
    config_path = actual_config.get("config_path")
    if type(config_path) is not str or not config_path:
        raise Stage2ActualPsS32CaptureBlocked("preflight actual config path가 비어 있습니다")

    j511 = _require_mapping(observed.get("j511"), label="preflight J511")
    _require_true(
        j511.get("three_identical_connected_samples"), label="preflight J511 three stable connected samples"
    )
    occupancy = _require_mapping(observed.get("pcm_occupancy"), label="preflight PCM occupancy")
    _require_true(occupancy.get("all_pcm_substreams_closed"), label="preflight PCM closed")
    _require_empty_sequence(occupancy.get("owners"), label="preflight PCM owners")
    routes = _require_mapping(observed.get("ape_routes"), label="preflight APE routes")
    for control, expected in _EXPECTED_ROUTES.items():
        row = _require_mapping(routes.get(control), label=f"preflight APE route {control}")
        _require_exact(row.get("observed"), expected, label=f"preflight APE route {control}")
    preflight_authority = _require_mapping(observed.get("authority"), label="preflight authority")
    _require_true(
        preflight_authority.get("same_card_s32_actual_config_provenance_pass"),
        label="preflight actual config provenance pass",
    )
    for key in (
        "actual_s32_stream_opened",
        "same_hardware_frame_identity_pass",
        "clock_or_fixed_lti_witness_pass",
        "stage2_ps_identification_pass",
        "canonical_training_eligible",
        "deployment_eligible",
    ):
        _require_false(preflight_authority.get(key), label=f"preflight authority.{key}")

    return dict(observed)


def validate_stage2_actual_ps_s32_user_live_gate(
    receipt: Mapping[str, Any], plan: Mapping[str, Any], planned_s32_pcm: np.ndarray
) -> dict[str, Any]:
    """명시적인 한 번의 24초 actual P/S 출력 승인만 구조적으로 확인한다."""

    canonical_plan, _ = validate_stage2_actual_ps_excitation_plan(plan, planned_s32_pcm)
    observed = _require_mapping(receipt, label="explicit user-facing live gate")
    _require_exact(observed.get("schema"), USER_LIVE_GATE_SCHEMA, label="user live gate schema")
    _require_true(observed.get("approved"), label="user live gate approved")
    for key in (
        "confirm_speaker",
        "confirm_user_present",
        "confirm_volume_minimum",
        "confirm_routing_and_geometry",
        "one_time_actual_ps_output",
    ):
        _require_true(observed.get(key), label=f"user live gate {key}")
    _require_exact(
        observed.get("actual_ps_plan_sha256"),
        canonical_plan["canonical_payload_sha256"],
        label="user live gate plan SHA",
    )
    _require_exact(
        observed.get("actual_config_payload_sha256"),
        canonical_plan["rt5640_static_config"]["config_payload_sha256"],
        label="user live gate actual config SHA",
    )
    _require_exact(
        observed.get("expected_output_duration_seconds"),
        canonical_plan["duration_seconds"],
        label="user live gate output duration",
    )
    _require_exact(
        observed.get("speaker_disconnect_notice_required"),
        "출력 종료 — 지금 스피커 분리",
        label="user live gate speaker disconnect notice",
    )
    return dict(observed)


def validate_stage2_actual_ps_s32_post_start_receipt(
    receipt: Mapping[str, Any], plan: Mapping[str, Any], planned_s32_pcm: np.ndarray
) -> dict[str, Any]:
    """future stream-start 뒤, arm 전 immutable receipt의 최소 형식을 검증한다.

    현 모듈은 stream을 만들지 않으므로 이 함수가 physical receipt를 생성하지 않는다.
    별도 live adapter가 pre-arm zero 상태에서 실제로 수집한 receipt만 전달해야 한다.
    """

    canonical_plan, _ = validate_stage2_actual_ps_excitation_plan(plan, planned_s32_pcm)
    observed = _require_mapping(receipt, label="post-start pre-arm receipt")
    _require_exact(observed.get("schema"), POST_START_RECEIPT_SCHEMA, label="post-start receipt schema")
    _require_true(observed.get("passed"), label="post-start receipt passed")
    _require_true(observed.get("stream_started"), label="post-start stream_started")
    _require_true(observed.get("checked_before_arm"), label="post-start checked_before_arm")
    _require_true(observed.get("pre_arm_output_exact_zero"), label="post-start pre-arm zero")
    _require_false(observed.get("speaker_output_armed"), label="post-start speaker_output_armed")
    _require_false(observed.get("raw_written"), label="post-start raw_written")
    _require_exact(
        observed.get("actual_ps_plan_sha256"),
        canonical_plan["canonical_payload_sha256"],
        label="post-start actual P/S plan SHA",
    )
    _require_exact(
        observed.get("actual_config_payload_sha256"),
        canonical_plan["rt5640_static_config"]["config_payload_sha256"],
        label="post-start actual config SHA",
    )
    _require_exact(
        observed.get("negotiated_hardware_audio"),
        _hardware_audio_from_plan(canonical_plan),
        label="post-start negotiated hardware_audio",
    )
    for key in ("resolved_input_device", "resolved_output_device"):
        value = observed.get(key)
        if type(value) is not int or value < 0:
            raise Stage2ActualPsS32CaptureBlocked(f"post-start {key}는 nonnegative exact int여야 합니다")
    j511 = _require_mapping(observed.get("j511"), label="post-start J511")
    _require_true(j511.get("three_identical_connected_samples"), label="post-start J511 stable")
    occupancy = _require_mapping(observed.get("pcm_occupancy"), label="post-start PCM occupancy")
    _require_true(
        occupancy.get("only_capture_owned_pcm_nodes"), label="post-start only capture PCM owner"
    )
    _require_empty_sequence(occupancy.get("foreign_owners"), label="post-start foreign PCM owners")
    routes = _require_mapping(observed.get("ape_routes"), label="post-start APE routes")
    for control, expected in _EXPECTED_ROUTES.items():
        row = _require_mapping(routes.get(control), label=f"post-start APE route {control}")
        _require_exact(row.get("observed"), expected, label=f"post-start APE route {control}")
    authority = _require_mapping(observed.get("authority"), label="post-start authority")
    for key in ("physical_ps_authority", "canonical_training_eligible", "deployment_eligible"):
        _require_false(authority.get(key), label=f"post-start authority.{key}")
    return dict(observed)


def assert_stage2_actual_ps_s32_live_capture_blocked(
    backend: Any,
    *,
    actual_config_preflight_receipt: Mapping[str, Any] | None = None,
    explicit_user_live_gate: Mapping[str, Any] | None = None,
    post_start_pre_arm_receipt: Mapping[str, Any] | None = None,
) -> None:
    """전달받은 backend를 읽지 않고 현재 미구현 capture를 차단한다.

    ``backend`` parameter는 후속 adapter가 현재 scaffold의 attribute-access 금지 성질을
    시험하기 위한 자리다. 이 함수는 backend import/factory ordering을 보장한다고 주장하지
    않는다. 실제 adapter는 preflight+user gate 뒤 zero-only stream을 열고, 그 stream에서
    얻은 post-start receipt PASS 뒤에만 nonzero arm/raw publisher를 검토해야 한다. 현
    준비 module은 이를 생성하거나 output을 arm하지 않는다.
    """

    del backend
    plan, planned_s32_pcm = build_stage2_actual_ps_excitation_plan()
    provenance = build_stage2_actual_ps_planned_provenance(plan, planned_s32_pcm)
    _prepare_stage2_actual_ps_s32_capture(plan, planned_s32_pcm, provenance)
    if actual_config_preflight_receipt is None:
        raise Stage2ActualPsS32CaptureBlocked(
            "actual-config read-only preflight receipt가 없어 backend를 읽기 전에 차단했습니다"
        )
    validate_stage2_actual_ps_s32_preflight_receipt(
        actual_config_preflight_receipt, plan, planned_s32_pcm
    )
    if explicit_user_live_gate is None:
        raise Stage2ActualPsS32CaptureBlocked(
            "명시적 사용자 live gate가 없어 backend를 읽기 전에 차단했습니다"
        )
    validate_stage2_actual_ps_s32_user_live_gate(explicit_user_live_gate, plan, planned_s32_pcm)
    if post_start_pre_arm_receipt is None:
        raise Stage2ActualPsS32CaptureBlocked(
            "post-start pre-arm receipt가 없어 backend를 읽기 전에 차단했습니다"
        )
    validate_stage2_actual_ps_s32_post_start_receipt(
        post_start_pre_arm_receipt, plan, planned_s32_pcm
    )
    raise Stage2ActualPsS32CaptureBlocked(
        f"{BLOCKED_STATUS}: raw publisher와 live execution은 아직 구현되지 않았으며 "
        "backend object은 읽지 않았습니다"
    )


def execute_stage2_actual_ps_s32_disarmed_capture(
    backend: Any,
    *,
    actual_config_preflight_receipt: Mapping[str, Any] | None = None,
    explicit_user_live_gate: Mapping[str, Any] | None = None,
    post_start_pre_arm_receipt: Mapping[str, Any] | None = None,
) -> None:
    """future live entry의 이름만 예약하며 현 단계에서는 항상 fail-closed한다."""

    assert_stage2_actual_ps_s32_live_capture_blocked(
        backend,
        actual_config_preflight_receipt=actual_config_preflight_receipt,
        explicit_user_live_gate=explicit_user_live_gate,
        post_start_pre_arm_receipt=post_start_pre_arm_receipt,
    )


__all__ = [
    "ACTUAL_CONFIG_PREFLIGHT_PASS_STATUS",
    "ACTUAL_CONFIG_PREFLIGHT_SCHEMA",
    "BLOCKED_STATUS",
    "CAPTURE_SCAFFOLD_SCHEMA",
    "POST_START_RECEIPT_SCHEMA",
    "RAW_SCHEMA",
    "USER_LIVE_GATE_SCHEMA",
    "Stage2ActualPsS32CaptureBlocked",
    "assert_stage2_actual_ps_s32_live_capture_blocked",
    "build_stage2_actual_ps_s32_capture_dry_run_receipt",
    "execute_stage2_actual_ps_s32_disarmed_capture",
    "validate_stage2_actual_ps_s32_post_start_receipt",
    "validate_stage2_actual_ps_s32_preflight_receipt",
    "validate_stage2_actual_ps_s32_user_live_gate",
]
