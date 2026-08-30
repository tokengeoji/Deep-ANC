"""외부 동기 전기 witness의 *무출력* admission boundary.

이 모듈은 새 ADC를 연결하거나 PCM을 열지 않는다. 현재 Jetson의 APE/INMP441
ERR·REF와 외부 output tap recorder가 서로 다른 clock domain이라는 사실을 config
단계에서 잊지 않게 만드는 최소 경계다.

중요: 여기서 ``topology_contract_met``가 참이어도 실제 배선·raw·P/S·ANC는 증명하지
않는다. 실제 raw adapter가 native raw/analysis SHA, callback, aperiodic S32→tap
frame map을 별도로 결속하기 전에는 electrical/plant/training/deployment authority는
항상 false다.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .control_band_contract import BroadbandFullOctaveContractV3


SCHEMA = "external_electrical_witness_admission_v1"
RECEIPT_SCHEMA = "external_electrical_witness_static_receipt_v1"
DEFAULT_CONFIG_RELATIVE_PATH = "configs/external_electrical_witness_admission_v1.yaml"
SAMPLE_RATE = 48_000
BLOCK_SIZE = 256
RAW_DTYPE = "<i4"
TIMING_RESIDUAL_MAX_SAMPLES = 0.0675518903
ROLE_NAMES = ("ERR", "REF", "NOISE_TAP", "CANCEL_TAP")
ALLOWED_TOPOLOGIES = (
    "single_acquisition_clock_all_four",
    "ape_external_hardware_frame_bridge",
)

_ROOT_KEYS = frozenset(
    {
        "schema",
        "role",
        "control_band_contract",
        "s32_transport_requirement",
        "topology_requirement",
        "raw_evidence_requirement",
        "tap_safety_requirement",
        "spatial_requirement",
        "authority",
    }
)
_CONTRACT_KEYS = frozenset({"id", "sha256", "sample_rate_hz", "block_size"})
_S32_REQUIREMENT_KEYS = frozenset(
    {
        "planned_s32_callback_sha256_required",
        "actual_callback_s32_sha256_required",
        "aperiodic_command_to_tap_frame_map_required",
        "callback_xrun_drop_add_must_be_zero",
        "actual_callback_s32_sha256",
    }
)
_TOPOLOGY_REQUIREMENT_KEYS = frozenset(
    {
        "allowed_topologies",
        "required_roles",
        "minimum_simultaneous_inputs",
        "ape_external_split_requires_hardware_frame_bridge",
        "nominal_rate_or_host_timestamp_only_accepted",
        "selected_topology",
    }
)
_RAW_EVIDENCE_KEYS = frozenset(
    {
        "native_raw_file_sha256_required",
        "canonical_raw_file_sha256_required",
        "analysis_file_sha256_required",
        "no_replace_publication_required",
        "native_raw_file_sha256",
        "canonical_raw_file_sha256",
        "analysis_file_sha256",
    }
)
_TAP_SAFETY_KEYS = frozenset(
    {
        "high_impedance_required",
        "isolated_required",
        "dc_blocked_required",
        "attenuated_required",
        "direct_speaker_terminal_to_adc_allowed",
        "agc_disabled_required",
        "limiter_disabled_required",
        "fixed_gain_required",
        "polarity_channel_test_required",
        "clip_stuck_must_be_zero",
        "pre_amplifier_tap_scope",
    }
)
_SPATIAL_KEYS = frozenset(
    {
        "plant_identification_minimum_inputs",
        "final_quiet_zone_minimum_simultaneous_inputs",
        "final_quiet_zone_err_positions",
        "sequential_err_positions_accepted",
    }
)
_AUTHORITY_KEYS = frozenset(
    {
        "static_requirements_only",
        "topology_contract_met",
        "electrical_witness_pass",
        "fullband_plant_identification_pass",
        "canonical_training_eligible",
        "deployment_eligible",
    }
)
_TOPOLOGY_KEYS = frozenset(
    {
        "kind",
        "input_channels",
        "sample_rate_hz",
        "raw_dtype",
        "role_channels",
        "role_capture_domains",
        "simultaneous_sampling",
        "shared_hardware_sample_clock",
        "continuous_frame_counter",
        "host_timestamp_only",
        "hardware_frame_bridge",
        "tap_safety",
    }
)
_BRIDGE_KEYS = frozenset(
    {
        "bclk_witness",
        "ws_witness",
        "absolute_frame_counter_witness",
        "continuous",
        "software_timestamp_only",
    }
)
_CANDIDATE_TAP_SAFETY_KEYS = frozenset(
    {
        "high_impedance",
        "isolated",
        "dc_blocked",
        "attenuated",
        "direct_speaker_terminal_to_adc",
        "agc_disabled",
        "limiter_disabled",
        "fixed_gain",
        "polarity_channel_test_pass",
        "clip_count",
        "stuck_channel_count",
        "scope",
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


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}는 mapping이어야 합니다")
    return value


def _exact_keys(value: Mapping[str, Any], *, expected: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{label} key가 exact하지 않습니다: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _exact(value: Any, expected: Any, *, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise ValueError(f"{label}가 계약과 다릅니다: expected={expected!r}, got={value!r}")


def _sha_or_none(value: Any, *, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label}는 null 또는 64자리 SHA-256이어야 합니다")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label}는 hexadecimal SHA-256이어야 합니다") from error


def _require_exact_bool(value: Any, *, label: str, expected: bool) -> None:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{label}는 exact {expected!r}여야 합니다")


def _validate_static_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="external electrical witness config")
    _exact_keys(root, expected=_ROOT_KEYS, label="external electrical witness config")
    _exact(root["schema"], SCHEMA, label="schema")
    _exact(
        root["role"],
        "fullband_external_electrical_witness_static_requirements_only",
        label="role",
    )

    contract = _mapping(root["control_band_contract"], label="control_band_contract")
    _exact_keys(contract, expected=_CONTRACT_KEYS, label="control_band_contract")
    expected_contract = BroadbandFullOctaveContractV3.canonical()
    _exact(contract["id"], expected_contract.contract_id, label="control_band_contract.id")
    _exact(contract["sha256"], expected_contract.digest(), label="control_band_contract.sha256")
    _exact(contract["sample_rate_hz"], SAMPLE_RATE, label="control_band_contract.sample_rate_hz")
    _exact(contract["block_size"], BLOCK_SIZE, label="control_band_contract.block_size")

    transport = _mapping(root["s32_transport_requirement"], label="s32_transport_requirement")
    _exact_keys(transport, expected=_S32_REQUIREMENT_KEYS, label="s32_transport_requirement")
    for name in (
        "planned_s32_callback_sha256_required",
        "actual_callback_s32_sha256_required",
        "aperiodic_command_to_tap_frame_map_required",
        "callback_xrun_drop_add_must_be_zero",
    ):
        _require_exact_bool(transport[name], label=f"s32_transport_requirement.{name}", expected=True)
    _sha_or_none(transport["actual_callback_s32_sha256"], label="s32_transport_requirement.actual_callback_s32_sha256")
    if transport["actual_callback_s32_sha256"] is not None:
        raise ValueError("static admission에는 actual callback PCM SHA를 넣을 수 없습니다")

    topology = _mapping(root["topology_requirement"], label="topology_requirement")
    _exact_keys(topology, expected=_TOPOLOGY_REQUIREMENT_KEYS, label="topology_requirement")
    _exact(topology["allowed_topologies"], list(ALLOWED_TOPOLOGIES), label="topology_requirement.allowed_topologies")
    _exact(topology["required_roles"], list(ROLE_NAMES), label="topology_requirement.required_roles")
    _exact(topology["minimum_simultaneous_inputs"], 4, label="topology_requirement.minimum_simultaneous_inputs")
    _require_exact_bool(
        topology["ape_external_split_requires_hardware_frame_bridge"],
        label="topology_requirement.ape_external_split_requires_hardware_frame_bridge",
        expected=True,
    )
    _require_exact_bool(
        topology["nominal_rate_or_host_timestamp_only_accepted"],
        label="topology_requirement.nominal_rate_or_host_timestamp_only_accepted",
        expected=False,
    )
    _exact(topology["selected_topology"], None, label="topology_requirement.selected_topology")

    raw = _mapping(root["raw_evidence_requirement"], label="raw_evidence_requirement")
    _exact_keys(raw, expected=_RAW_EVIDENCE_KEYS, label="raw_evidence_requirement")
    for name in (
        "native_raw_file_sha256_required",
        "canonical_raw_file_sha256_required",
        "analysis_file_sha256_required",
        "no_replace_publication_required",
    ):
        _require_exact_bool(raw[name], label=f"raw_evidence_requirement.{name}", expected=True)
    for name in (
        "native_raw_file_sha256",
        "canonical_raw_file_sha256",
        "analysis_file_sha256",
    ):
        _sha_or_none(raw[name], label=f"raw_evidence_requirement.{name}")
        if raw[name] is not None:
            raise ValueError(f"static admission에는 {name}를 넣을 수 없습니다")

    safety = _mapping(root["tap_safety_requirement"], label="tap_safety_requirement")
    _exact_keys(safety, expected=_TAP_SAFETY_KEYS, label="tap_safety_requirement")
    for name in (
        "high_impedance_required",
        "isolated_required",
        "dc_blocked_required",
        "attenuated_required",
        "agc_disabled_required",
        "limiter_disabled_required",
        "fixed_gain_required",
        "polarity_channel_test_required",
        "clip_stuck_must_be_zero",
    ):
        _require_exact_bool(safety[name], label=f"tap_safety_requirement.{name}", expected=True)
    _require_exact_bool(
        safety["direct_speaker_terminal_to_adc_allowed"],
        label="tap_safety_requirement.direct_speaker_terminal_to_adc_allowed",
        expected=False,
    )
    _exact(
        safety["pre_amplifier_tap_scope"],
        "dac_line_input_witness_only_not_amplifier_speaker_transfer",
        label="tap_safety_requirement.pre_amplifier_tap_scope",
    )

    spatial = _mapping(root["spatial_requirement"], label="spatial_requirement")
    _exact_keys(spatial, expected=_SPATIAL_KEYS, label="spatial_requirement")
    _exact(spatial["plant_identification_minimum_inputs"], 4, label="spatial_requirement.plant_identification_minimum_inputs")
    _exact(spatial["final_quiet_zone_minimum_simultaneous_inputs"], 8, label="spatial_requirement.final_quiet_zone_minimum_simultaneous_inputs")
    _exact(spatial["final_quiet_zone_err_positions"], 5, label="spatial_requirement.final_quiet_zone_err_positions")
    _require_exact_bool(
        spatial["sequential_err_positions_accepted"],
        label="spatial_requirement.sequential_err_positions_accepted",
        expected=False,
    )

    authority = _mapping(root["authority"], label="authority")
    _exact_keys(authority, expected=_AUTHORITY_KEYS, label="authority")
    expected_authority = {
        "static_requirements_only": True,
        "topology_contract_met": False,
        "electrical_witness_pass": False,
        "fullband_plant_identification_pass": False,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
    }
    for name, expected in expected_authority.items():
        _require_exact_bool(authority[name], label=f"authority.{name}", expected=expected)

    return {
        "control_band_contract_sha256": expected_contract.digest(),
        "config_payload_sha256": _payload_sha256(dict(root)),
        "authority": expected_authority,
    }


def _candidate_mapping(value: Any, *, label: str, expected: frozenset[str]) -> dict[str, Any]:
    result = _mapping(value, label=label)
    _exact_keys(result, expected=expected, label=label)
    return dict(result)


def _validate_candidate_tap_safety(value: Any) -> list[str]:
    safety = _candidate_mapping(value, label="candidate.tap_safety", expected=_CANDIDATE_TAP_SAFETY_KEYS)
    reasons: list[str] = []
    required = {
        "high_impedance": True,
        "isolated": True,
        "dc_blocked": True,
        "attenuated": True,
        "direct_speaker_terminal_to_adc": False,
        "agc_disabled": True,
        "limiter_disabled": True,
        "fixed_gain": True,
        "polarity_channel_test_pass": True,
    }
    for name, expected in required.items():
        if type(safety[name]) is not bool or safety[name] is not expected:
            reasons.append(f"tap_safety_{name}")
    for name in ("clip_count", "stuck_channel_count"):
        value = safety[name]
        if type(value) is not int or value != 0:
            reasons.append(f"tap_safety_{name}")
    scope = safety["scope"]
    if scope not in {
        "dac_line_input_witness_only_not_amplifier_speaker_transfer",
        "post_amplifier_isolated_attenuated_witness",
    }:
        reasons.append("tap_safety_scope")
    return reasons


def assess_candidate_topology_v1(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """후보 acquisition topology를 검사한다. 물리적 PASS를 발행하지 않는다.

    이 함수는 장치/파일을 열지 않는다. ``topology_contract_met``는 이후 실제 native
    raw receipt를 받을 *형식적 가능성*만 뜻하며, electrical/plant/training/deployment
    authority는 의도적으로 false다.
    """

    payload = _candidate_mapping(candidate, label="external witness candidate", expected=_TOPOLOGY_KEYS)
    reasons: list[str] = []
    kind = payload["kind"]
    if kind not in ALLOWED_TOPOLOGIES:
        reasons.append("unsupported_topology")
    channels = payload["input_channels"]
    if type(channels) is not int or channels < 4:
        reasons.append("minimum_four_simultaneous_inputs")
    if payload["sample_rate_hz"] != SAMPLE_RATE or type(payload["sample_rate_hz"]) is not int:
        reasons.append("sample_rate_48000")
    if payload["raw_dtype"] != RAW_DTYPE or type(payload["raw_dtype"]) is not str:
        reasons.append("raw_dtype_int32")
    for name in ("simultaneous_sampling", "shared_hardware_sample_clock", "continuous_frame_counter"):
        if type(payload[name]) is not bool or payload[name] is not True:
            reasons.append(name)
    if type(payload["host_timestamp_only"]) is not bool or payload["host_timestamp_only"] is not False:
        reasons.append("host_timestamp_only_rejected")

    role_channels = payload["role_channels"]
    if not isinstance(role_channels, Mapping) or set(role_channels) != set(ROLE_NAMES):
        reasons.append("four_unique_role_map")
    else:
        indexes = list(role_channels.values())
        if (
            any(type(index) is not int or index < 0 or (type(channels) is int and index >= channels) for index in indexes)
            or len(set(indexes)) != len(ROLE_NAMES)
        ):
            reasons.append("four_unique_role_map")

    domains = payload["role_capture_domains"]
    if not isinstance(domains, Mapping) or set(domains) != set(ROLE_NAMES) or not all(
        isinstance(value, str) and value for value in (domains.values() if isinstance(domains, Mapping) else ())
    ):
        reasons.append("role_capture_domains")

    bridge = payload["hardware_frame_bridge"]
    if not isinstance(bridge, Mapping):
        reasons.append("hardware_frame_bridge")
    else:
        try:
            bridge = _candidate_mapping(bridge, label="candidate.hardware_frame_bridge", expected=_BRIDGE_KEYS)
        except ValueError:
            reasons.append("hardware_frame_bridge")
        else:
            bridge_required = kind == "ape_external_hardware_frame_bridge"
            for name in (
                "bclk_witness",
                "ws_witness",
                "absolute_frame_counter_witness",
                "continuous",
            ):
                if bridge_required and (type(bridge[name]) is not bool or bridge[name] is not True):
                    reasons.append(f"hardware_frame_bridge_{name}")
            if type(bridge["software_timestamp_only"]) is not bool or bridge["software_timestamp_only"] is not False:
                reasons.append("hardware_frame_bridge_software_timestamp_only")

    if isinstance(domains, Mapping) and set(domains) == set(ROLE_NAMES):
        all_domains = set(domains.values())
        if kind == "single_acquisition_clock_all_four" and len(all_domains) != 1:
            reasons.append("single_clock_all_roles_same_domain")
        if kind == "ape_external_hardware_frame_bridge":
            if domains["ERR"] != domains["REF"] or domains["NOISE_TAP"] != domains["CANCEL_TAP"]:
                reasons.append("bridge_role_domain_pairing")
            if domains["ERR"] == domains["NOISE_TAP"]:
                reasons.append("bridge_requires_distinct_ape_external_domains")

    try:
        reasons.extend(_validate_candidate_tap_safety(payload["tap_safety"]))
    except ValueError:
        reasons.append("tap_safety")

    reasons = list(dict.fromkeys(reasons))
    return {
        "schema": "external_electrical_witness_topology_assessment_v1",
        "status": "TOPOLOGY_REQUIREMENTS_MET" if not reasons else "BLOCKED",
        "topology_contract_met": not reasons,
        "blocking_reasons": reasons,
        "electrical_witness_pass": False,
        "fullband_plant_identification_pass": False,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
        "timing_residual_max_samples": TIMING_RESIDUAL_MAX_SAMPLES,
    }


def validate_external_electrical_witness_static_admission(payload: Mapping[str, Any]) -> dict[str, Any]:
    """현재 static requirement config를 검증한다. 항상 physical ``BLOCKED``다."""

    core = _validate_static_config(payload)
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "BLOCKED",
        "static_gate_pass": True,
        "audio_opened": False,
        "results_written": False,
        "control_band_contract_sha256": core["control_band_contract_sha256"],
        "timing_residual_max_samples": TIMING_RESIDUAL_MAX_SAMPLES,
        "required_topologies": list(ALLOWED_TOPOLOGIES),
        "required_roles": list(ROLE_NAMES),
        "blocking_requirements": [
            "actual_s32_callback_pcm_sha256",
            "resolved_topology_candidate",
            "native_canonical_analysis_raw_sha256_chain",
            "zero_xrun_drop_add_and_continuous_frame_counter",
            "aperiodic_s32_to_two_tap_frame_identity",
            "interchannel_timing_residual_within_limit",
            "safe_tap_hardware_evidence_and_operator_acknowledgement",
            "raw_first_fullband_P_S_analysis",
        ],
        "authority": core["authority"],
        "config_payload_sha256": core["config_payload_sha256"],
    }


def load_external_electrical_witness_static_admission(path: str | Path) -> dict[str, Any]:
    """YAML requirement file을 읽기만 한다. hardware/PCM/result 파일은 열지 않는다."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("external electrical witness config YAML root는 mapping이어야 합니다")
    receipt = validate_external_electrical_witness_static_admission(payload)
    return {
        **receipt,
        "config": {
            "path": str(config_path),
            "file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
    }


__all__ = [
    "ALLOWED_TOPOLOGIES",
    "BLOCK_SIZE",
    "DEFAULT_CONFIG_RELATIVE_PATH",
    "RAW_DTYPE",
    "RECEIPT_SCHEMA",
    "ROLE_NAMES",
    "SAMPLE_RATE",
    "SCHEMA",
    "TIMING_RESIDUAL_MAX_SAMPLES",
    "assess_candidate_topology_v1",
    "load_external_electrical_witness_static_admission",
    "validate_external_electrical_witness_static_admission",
]
