from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from deep_anc.dsp.rt5640_fullband_static_v10 import (
    ALLOWED_LIVE_JACK_STATES,
    DEFAULT_CONFIG_RELATIVE_PATH,
    Q15_TO_S32_LEFT_SHIFT,
    load_rt5640_fullband_static_contract,
    validate_rt5640_fullband_static_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _payload() -> dict[str, object]:
    return yaml.safe_load((REPO_ROOT / DEFAULT_CONFIG_RELATIVE_PATH).read_text())


def test_static_contract_binds_v3_s32_and_explicitly_remains_blocked() -> None:
    receipt = validate_rt5640_fullband_static_contract(_payload())
    assert receipt["status"] == "BLOCKED"
    assert receipt["static_gate_pass"] is True
    assert receipt["audio_opened"] is False
    assert receipt["control_band_contract_sha256"] == (
        "53579b9ff8419ac19fb2458c29a3e8a94ffbb2eeb88cc07f34b76c68033989f2"
    )
    assert receipt["hardware_audio"]["input"]["dtype"] == "int32"
    assert receipt["hardware_audio"]["output"]["dtype"] == "int32"
    assert receipt["s32_signal_scale"] == {
        "source": "Q15 actual-int16 signal-only plan",
        "conversion": "exact_signed_left_shift",
        "left_shift_bits": Q15_TO_S32_LEFT_SHIFT,
        "normalized_full_scale_preserved": True,
        "simple_int16_to_int32_cast_allowed": False,
    }
    assert receipt["live_jack_allowed_states"] == list(ALLOWED_LIVE_JACK_STATES)
    assert receipt["authority"] == {
        "static_contract_only": True,
        "j511_connection_observed": False,
        "s32_duplex_transport_pass": False,
        "hardware_frame_identity_pass": False,
        "electrical_witness_pass": False,
        "fullband_plant_identification_pass": False,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
    }


@pytest.mark.parametrize(
    "mutate,match",
    (
        (lambda p: p.__setitem__("schema", "legacy"), "schema"),
        (lambda p: p["audio"]["output"].__setitem__("dtype", "int16"), "output.dtype"),
        (lambda p: p["audio"]["output"].__setitem__("card", "Audio"), "output.card"),
        (lambda p: p["fullband_v3"].__setitem__("q15_to_s32_left_shift", 0), "left_shift"),
        (lambda p: p["fullband_v3"].__setitem__("excitation_lower_hz", 150.0), "excitation_lower"),
        (lambda p: p["fullband_v3"].__setitem__("excitation_upper_hz", 8000.0), "excitation_upper"),
        (lambda p: p["fullband_v3"].__setitem__("live_jack_allowed_states", ["None"]), "live_jack"),
        (lambda p: p["fullband_v3"].__setitem__("legacy_v6_relabel_allowed", True), "relabel"),
        (lambda p: p["maximum_authority"].__setitem__("canonical_training_eligible", True), "canonical_training"),
        (lambda p: p["channels"].__setitem__("cancel_out", 0), "cancel_out"),
    ),
)
def test_static_contract_rejects_unsafe_or_legacy_mutations(mutate, match: str) -> None:  # type: ignore[no-untyped-def]
    payload = deepcopy(_payload())
    mutate(payload)
    with pytest.raises(ValueError, match=match):
        validate_rt5640_fullband_static_contract(payload)


def test_static_contract_rejects_extra_field_and_preserves_config_file_hash() -> None:
    payload = _payload()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="extra"):
        validate_rt5640_fullband_static_contract(payload)

    receipt = load_rt5640_fullband_static_contract(
        REPO_ROOT / DEFAULT_CONFIG_RELATIVE_PATH
    )
    assert receipt["config"]["path"].endswith(DEFAULT_CONFIG_RELATIVE_PATH)
    assert len(receipt["config"]["file_sha256"]) == 64


def test_static_gate_has_no_audio_backend_or_evidence_writer() -> None:
    """이 세대의 PASS가 device open/write를 뜻하지 않게 source 자체를 고정한다."""

    module_source = (
        REPO_ROOT / "src/deep_anc/dsp/rt5640_fullband_static_v10.py"
    ).read_text(encoding="utf-8")
    script_source = (
        REPO_ROOT / "scripts/jetson/check_rt5640_fullband_static_v10.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(module_source + "\n" + script_source)
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
    assert not {"sounddevice", "subprocess"} & imported
    assert not {"save", "savez", "write", "write_bytes", "write_text"} & called
