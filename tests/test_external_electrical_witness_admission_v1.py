from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from deep_anc.dsp.external_electrical_witness_admission_v1 import (
    DEFAULT_CONFIG_RELATIVE_PATH,
    TIMING_RESIDUAL_MAX_SAMPLES,
    assess_candidate_topology_v1,
    load_external_electrical_witness_static_admission,
    validate_external_electrical_witness_static_admission,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / DEFAULT_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8")
    )


def _tap_safety() -> dict:
    return {
        "high_impedance": True,
        "isolated": True,
        "dc_blocked": True,
        "attenuated": True,
        "direct_speaker_terminal_to_adc": False,
        "agc_disabled": True,
        "limiter_disabled": True,
        "fixed_gain": True,
        "polarity_channel_test_pass": True,
        "clip_count": 0,
        "stuck_channel_count": 0,
        "scope": "dac_line_input_witness_only_not_amplifier_speaker_transfer",
    }


def _candidate(kind: str = "single_acquisition_clock_all_four") -> dict:
    domains = {
        "ERR": "external_adc_0",
        "REF": "external_adc_0",
        "NOISE_TAP": "external_adc_0",
        "CANCEL_TAP": "external_adc_0",
    }
    bridge = {
        "bclk_witness": False,
        "ws_witness": False,
        "absolute_frame_counter_witness": False,
        "continuous": False,
        "software_timestamp_only": False,
    }
    if kind == "ape_external_hardware_frame_bridge":
        domains = {
            "ERR": "ape_i2s2",
            "REF": "ape_i2s2",
            "NOISE_TAP": "external_adc_0",
            "CANCEL_TAP": "external_adc_0",
        }
        bridge = {
            "bclk_witness": True,
            "ws_witness": True,
            "absolute_frame_counter_witness": True,
            "continuous": True,
            "software_timestamp_only": False,
        }
    return {
        "kind": kind,
        "input_channels": 4,
        "sample_rate_hz": 48_000,
        "raw_dtype": "<i4",
        "role_channels": {"ERR": 0, "REF": 1, "NOISE_TAP": 2, "CANCEL_TAP": 3},
        "role_capture_domains": domains,
        "simultaneous_sampling": True,
        "shared_hardware_sample_clock": True,
        "continuous_frame_counter": True,
        "host_timestamp_only": False,
        "hardware_frame_bridge": bridge,
        "tap_safety": _tap_safety(),
    }


def test_static_contract_is_explicitly_blocked_and_never_opens_audio() -> None:
    receipt = validate_external_electrical_witness_static_admission(_config())
    assert receipt["status"] == "BLOCKED"
    assert receipt["static_gate_pass"] is True
    assert receipt["audio_opened"] is False
    assert receipt["results_written"] is False
    assert receipt["control_band_contract_sha256"] == (
        "53579b9ff8419ac19fb2458c29a3e8a94ffbb2eeb88cc07f34b76c68033989f2"
    )
    assert receipt["timing_residual_max_samples"] == TIMING_RESIDUAL_MAX_SAMPLES
    assert receipt["authority"]["canonical_training_eligible"] is False
    assert receipt["authority"]["deployment_eligible"] is False


@pytest.mark.parametrize(
    "mutate,match",
    (
        (
            lambda c: c["control_band_contract"].__setitem__("sha256", "0" * 64),
            "sha256",
        ),
        (
            lambda c: c["topology_requirement"].__setitem__("minimum_simultaneous_inputs", 2),
            "minimum_simultaneous_inputs",
        ),
        (
            lambda c: c["topology_requirement"].__setitem__("nominal_rate_or_host_timestamp_only_accepted", True),
            "nominal_rate_or_host_timestamp_only_accepted",
        ),
        (
            lambda c: c["tap_safety_requirement"].__setitem__("direct_speaker_terminal_to_adc_allowed", True),
            "direct_speaker_terminal",
        ),
        (
            lambda c: c["authority"].__setitem__("canonical_training_eligible", True),
            "canonical_training_eligible",
        ),
        (
            lambda c: c["raw_evidence_requirement"].__setitem__("native_raw_file_sha256", "a" * 64),
            "static admission",
        ),
    ),
)
def test_static_contract_rejects_premature_promotion_or_evidence_injection(mutate, match: str) -> None:  # type: ignore[no-untyped-def]
    config = deepcopy(_config())
    mutate(config)
    with pytest.raises(ValueError, match=match):
        validate_external_electrical_witness_static_admission(config)


def test_four_channel_single_clock_topology_is_only_a_future_raw_candidate() -> None:
    assessment = assess_candidate_topology_v1(_candidate())
    assert assessment["status"] == "TOPOLOGY_REQUIREMENTS_MET"
    assert assessment["topology_contract_met"] is True
    assert assessment["electrical_witness_pass"] is False
    assert assessment["fullband_plant_identification_pass"] is False
    assert assessment["canonical_training_eligible"] is False
    assert assessment["deployment_eligible"] is False


def test_ape_to_external_candidate_requires_actual_hardware_frame_bridge() -> None:
    candidate = _candidate("ape_external_hardware_frame_bridge")
    candidate["hardware_frame_bridge"]["absolute_frame_counter_witness"] = False
    candidate["hardware_frame_bridge"]["software_timestamp_only"] = True
    assessment = assess_candidate_topology_v1(candidate)
    assert assessment["status"] == "BLOCKED"
    assert "hardware_frame_bridge_absolute_frame_counter_witness" in assessment["blocking_reasons"]
    assert "hardware_frame_bridge_software_timestamp_only" in assessment["blocking_reasons"]


@pytest.mark.parametrize(
    "mutate,reason",
    (
        (lambda p: p.__setitem__("input_channels", 3), "minimum_four_simultaneous_inputs"),
        (lambda p: p["role_channels"].__setitem__("CANCEL_TAP", 2), "four_unique_role_map"),
        (lambda p: p.__setitem__("host_timestamp_only", True), "host_timestamp_only_rejected"),
        (lambda p: p.__setitem__("raw_dtype", "<i2"), "raw_dtype_int32"),
        (lambda p: p["tap_safety"].__setitem__("agc_disabled", False), "tap_safety_agc_disabled"),
        (lambda p: p["tap_safety"].__setitem__("direct_speaker_terminal_to_adc", True), "tap_safety_direct_speaker_terminal_to_adc"),
    ),
)
def test_candidate_rejects_shortcuts(mutate, reason: str) -> None:  # type: ignore[no-untyped-def]
    candidate = _candidate()
    mutate(candidate)
    assessment = assess_candidate_topology_v1(candidate)
    assert assessment["status"] == "BLOCKED"
    assert reason in assessment["blocking_reasons"]


def test_static_checker_loads_current_requirements_without_hardware_access() -> None:
    receipt = load_external_electrical_witness_static_admission(
        REPO_ROOT / DEFAULT_CONFIG_RELATIVE_PATH
    )
    assert receipt["config"]["path"].endswith(DEFAULT_CONFIG_RELATIVE_PATH)
    assert len(receipt["config"]["file_sha256"]) == 64


def test_static_source_has_no_audio_backend_or_writer() -> None:
    sources = [
        REPO_ROOT / "src/deep_anc/dsp/external_electrical_witness_admission_v1.py",
        REPO_ROOT / "scripts/jetson/check_external_electrical_witness_static.py",
    ]
    tree = ast.parse("\n".join(path.read_text(encoding="utf-8") for path in sources))
    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    assert not {"sounddevice", "subprocess", "torch"} & imported
    assert not {"save", "savez", "write", "write_text", "write_bytes"} & called
