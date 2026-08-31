from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest

from deep_anc.dsp.stage2_2khz_actual_ps_plan import (
    build_stage2_actual_ps_excitation_plan,
    load_stage2_actual_ps_static_config,
)
from deep_anc.dsp.stage2_2khz_actual_ps_s32_capture import (
    ACTUAL_CONFIG_PREFLIGHT_PASS_STATUS,
    ACTUAL_CONFIG_PREFLIGHT_SCHEMA,
    BLOCKED_STATUS,
    CAPTURE_SCAFFOLD_SCHEMA,
    POST_START_RECEIPT_SCHEMA,
    USER_LIVE_GATE_SCHEMA,
    Stage2ActualPsS32CaptureBlocked,
    assert_stage2_actual_ps_s32_live_capture_blocked,
    build_stage2_actual_ps_s32_capture_dry_run_receipt,
    execute_stage2_actual_ps_s32_disarmed_capture,
    validate_stage2_actual_ps_s32_post_start_receipt,
    validate_stage2_actual_ps_s32_preflight_receipt,
    validate_stage2_actual_ps_s32_user_live_gate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE = REPO_ROOT / "src/deep_anc/dsp/stage2_2khz_actual_ps_s32_capture.py"

_ROUTES = {
    "I2S1 Mux": "ADMAIF1",
    "ADMAIF1 Mux": "I2S1",
    "ADMAIF2 Mux": "I2S2",
    "I2S2 Mux": "ADMAIF2",
}


def _route_receipt() -> dict[str, dict[str, str]]:
    return {name: {"observed": target} for name, target in _ROUTES.items()}


def _actual_material() -> tuple[dict[str, object], object, dict[str, object]]:
    plan, pcm = build_stage2_actual_ps_excitation_plan()
    static = load_stage2_actual_ps_static_config()
    return plan, pcm, static


def _preflight_receipt(plan: dict[str, object], static: dict[str, object]) -> dict[str, object]:
    return {
        "schema": ACTUAL_CONFIG_PREFLIGHT_SCHEMA,
        "status": ACTUAL_CONFIG_PREFLIGHT_PASS_STATUS,
        "passed": True,
        "dry_run": True,
        "audio_backend_imported": False,
        "alsa_pcm_opened": False,
        "speaker_output": False,
        "filesystem_write_performed": False,
        "actual_ps_config": {
            "schema": static["schema"],
            "status": static["status"],
            "audio_opened": False,
            "speaker_output": False,
            "results_written": False,
            "hardware_audio": static["hardware_audio"],
            "config_path": static["config"]["path"],
            "config_file_sha256": static["config"]["file_sha256"],
            "config_payload_sha256": static["config_payload_sha256"],
            "stage2_contract_sha256": plan["stage2_contract"]["sha256"],
            "forbidden_source_or_receipt_origins": static["forbidden_source_or_receipt_origins"],
            "authority": static["authority"],
            "prohibited_transports": {
                "usb_ab13x_selected": False,
                "output_master_split_clock_selected": False,
                "bandlimited_fallback_selected": False,
                "s16_selected": False,
                "contract_forbids_usb_ab13x": True,
                "contract_forbids_output_master_split_clock": True,
                "contract_forbids_bandlimited_fallback": True,
                "contract_forbids_s16": True,
            },
        },
        "j511": {"three_identical_connected_samples": True},
        "pcm_occupancy": {"all_pcm_substreams_closed": True, "owners": ()},
        "ape_routes": _route_receipt(),
        "authority": {
            "same_card_s32_actual_config_provenance_pass": True,
            "actual_s32_stream_opened": False,
            "same_hardware_frame_identity_pass": False,
            "clock_or_fixed_lti_witness_pass": False,
            "stage2_ps_identification_pass": False,
            "canonical_training_eligible": False,
            "deployment_eligible": False,
        },
    }


def _user_gate(plan: dict[str, object]) -> dict[str, object]:
    return {
        "schema": USER_LIVE_GATE_SCHEMA,
        "approved": True,
        "confirm_speaker": True,
        "confirm_user_present": True,
        "confirm_volume_minimum": True,
        "confirm_routing_and_geometry": True,
        "one_time_actual_ps_output": True,
        "actual_ps_plan_sha256": plan["canonical_payload_sha256"],
        "actual_config_payload_sha256": plan["rt5640_static_config"]["config_payload_sha256"],
        "expected_output_duration_seconds": 24.0,
        "speaker_disconnect_notice_required": "출력 종료 — 지금 스피커 분리",
    }


def _post_start_receipt(plan: dict[str, object], static: dict[str, object]) -> dict[str, object]:
    return {
        "schema": POST_START_RECEIPT_SCHEMA,
        "passed": True,
        "stream_started": True,
        "checked_before_arm": True,
        "pre_arm_output_exact_zero": True,
        "speaker_output_armed": False,
        "raw_written": False,
        "actual_ps_plan_sha256": plan["canonical_payload_sha256"],
        "actual_config_payload_sha256": plan["rt5640_static_config"]["config_payload_sha256"],
        "negotiated_hardware_audio": static["hardware_audio"],
        "resolved_input_device": 1,
        "resolved_output_device": 2,
        "j511": {"three_identical_connected_samples": True},
        "pcm_occupancy": {"only_capture_owned_pcm_nodes": True, "foreign_owners": ()},
        "ape_routes": _route_receipt(),
        "authority": {
            "physical_ps_authority": False,
            "canonical_training_eligible": False,
            "deployment_eligible": False,
        },
    }


def test_dry_run_receipt_binds_actual_full_pe_plan_and_never_claims_audio_or_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    receipt = build_stage2_actual_ps_s32_capture_dry_run_receipt()

    assert receipt["schema"] == CAPTURE_SCAFFOLD_SCHEMA
    assert receipt["status"] == BLOCKED_STATUS
    assert receipt["dry_run"] is True
    assert receipt["audio_backend_imported"] is False
    assert receipt["alsa_pcm_opened"] is False
    assert receipt["speaker_output"] is False
    assert receipt["raw_written"] is False
    assert receipt["live_capture_may_open"] is False
    assert receipt["raw_publisher_implemented"] is False
    assert receipt["live_execution_implemented"] is False
    assert receipt["hardware_audio"]["input"]["pcm"] == 1
    assert receipt["hardware_audio"]["output"]["pcm"] == 0
    assert receipt["hardware_audio"]["channels"] == {
        "error_mic": 0,
        "reference_mic": 1,
        "noise_out": 0,
        "cancel_out": 1,
    }
    assert receipt["planned_actual_ps"] == {
        "plan_schema": "stage2_2khz_rt5640_actual_ps_excitation_plan_v1",
        "plan_payload_sha256": receipt["planned_provenance"]["actual_ps_plan_sha256"],
        "planned_s32_pcm_sha256": receipt["planned_actual_ps"]["planned_s32_pcm_sha256"],
        "planned_s32_shape": [1_152_000, 2],
        "planned_s32_dtype": "<i4",
        "expected_callbacks": 4_500,
        "duration_seconds": 24.0,
        "source_schema": "stage2_2khz_time_separated_full_pe_plan_v2",
        "source_plan_payload_sha256": receipt["planned_provenance"]["source_measurement_plan_sha256"],
        "source_time_role_channel_mapping_sha256": receipt["planned_provenance"]["source_time_role_channel_mapping_sha256"],
        "source_transport_inherited": False,
        "source_fallback_plan_usable": False,
        "source_audio_execution_allowed": False,
        "low_16_bits_must_be_zero": True,
    }
    assert receipt["future_live_admission"]["actual_config_preflight_receipt_required"] is True
    assert receipt["future_live_admission"]["explicit_user_facing_live_gate_required"] is True
    assert receipt["future_live_admission"]["post_start_pre_arm_receipt_required"] is True
    assert receipt["future_live_admission"] == {
        "actual_config_preflight_receipt_required": True,
        "explicit_user_facing_live_gate_required": True,
        "post_start_pre_arm_receipt_required": True,
        "pre_arm_output_exact_zero_required": True,
        "preflight_and_user_gate_must_pass_before_backend_import_or_stream_open": True,
        "post_start_receipt_must_pass_before_nonzero_output_arm_or_raw_publisher": True,
        "zero_only_stream_start_may_follow_preflight_and_user_gate": True,
        "post_start_receipt_must_be_collected_after_stream_start_before_arm": True,
        "output_duration_seconds": 24.0,
        "output_close_notice_required": "출력 종료 — 지금 스피커 분리",
        "automatic_retry_or_reoutput_allowed": False,
    }
    assert receipt["future_live_admission"]["output_duration_seconds"] == 24.0
    assert receipt["authority"] == {
        "actual_config_read_only_preflight_pass": False,
        "explicit_user_live_gate_pass": False,
        "post_start_pre_arm_receipt_pass": False,
        "same_card_s32_transport_pass": False,
        "physical_raw_present": False,
        "physical_ps_authority": False,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
    }
    assert list(tmp_path.iterdir()) == []


def test_actual_config_preflight_requires_new_actual_config_binding_not_old_schema_or_static_receipt() -> None:
    plan, pcm, static = _actual_material()
    receipt = _preflight_receipt(plan, static)

    assert validate_stage2_actual_ps_s32_preflight_receipt(receipt, plan, pcm) == receipt

    old = deepcopy(receipt)
    del old["actual_ps_config"]
    with pytest.raises(Stage2ActualPsS32CaptureBlocked, match="actual_ps_config"):
        validate_stage2_actual_ps_s32_preflight_receipt(old, plan, pcm)

    forged = deepcopy(receipt)
    forged["actual_ps_config"]["config_payload_sha256"] = "0" * 64
    with pytest.raises(Stage2ActualPsS32CaptureBlocked, match="payload SHA"):
        validate_stage2_actual_ps_s32_preflight_receipt(forged, plan, pcm)

    forbidden = deepcopy(receipt)
    forbidden["actual_ps_config"]["prohibited_transports"]["bandlimited_fallback_selected"] = True
    with pytest.raises(Stage2ActualPsS32CaptureBlocked, match="prohibited transports"):
        validate_stage2_actual_ps_s32_preflight_receipt(forbidden, plan, pcm)


def test_user_gate_and_post_start_receipt_are_bound_to_plan_and_pre_arm_state() -> None:
    plan, pcm, static = _actual_material()
    gate = _user_gate(plan)
    post_start = _post_start_receipt(plan, static)

    assert validate_stage2_actual_ps_s32_user_live_gate(gate, plan, pcm) == gate
    assert validate_stage2_actual_ps_s32_post_start_receipt(post_start, plan, pcm) == post_start

    no_approval = deepcopy(gate)
    no_approval["approved"] = False
    with pytest.raises(Stage2ActualPsS32CaptureBlocked, match="approved"):
        validate_stage2_actual_ps_s32_user_live_gate(no_approval, plan, pcm)

    wrong_duration = deepcopy(gate)
    wrong_duration["expected_output_duration_seconds"] = 48.0
    with pytest.raises(Stage2ActualPsS32CaptureBlocked, match="output duration"):
        validate_stage2_actual_ps_s32_user_live_gate(wrong_duration, plan, pcm)

    armed = deepcopy(post_start)
    armed["pre_arm_output_exact_zero"] = False
    with pytest.raises(Stage2ActualPsS32CaptureBlocked, match="pre-arm zero"):
        validate_stage2_actual_ps_s32_post_start_receipt(armed, plan, pcm)

    foreign_owner = deepcopy(post_start)
    foreign_owner["pcm_occupancy"]["foreign_owners"] = ["pid=999"]
    with pytest.raises(Stage2ActualPsS32CaptureBlocked, match="foreign PCM owners"):
        validate_stage2_actual_ps_s32_post_start_receipt(foreign_owner, plan, pcm)


class _UnreadableBackend:
    def __init__(self) -> None:
        object.__setattr__(self, "touched", False)

    def __getattribute__(self, name: str):  # type: ignore[no-untyped-def]
        if name == "touched":
            return object.__getattribute__(self, name)
        object.__setattr__(self, "touched", True)
        raise AssertionError(f"backend must not be read: {name}")


def test_live_entry_rejects_missing_or_even_structurally_complete_gates_before_backend_read() -> None:
    backend = _UnreadableBackend()
    with pytest.raises(Stage2ActualPsS32CaptureBlocked, match="preflight receipt"):
        execute_stage2_actual_ps_s32_disarmed_capture(backend)
    assert backend.touched is False

    plan, pcm, static = _actual_material()
    with pytest.raises(Stage2ActualPsS32CaptureBlocked, match=BLOCKED_STATUS):
        assert_stage2_actual_ps_s32_live_capture_blocked(
            backend,
            actual_config_preflight_receipt=_preflight_receipt(plan, static),
            explicit_user_live_gate=_user_gate(plan),
            post_start_pre_arm_receipt=_post_start_receipt(plan, static),
        )
    assert backend.touched is False


def test_module_imports_only_actual_plan_layer_and_has_no_audio_or_raw_write_calls() -> None:
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
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

    assert "stage2_2khz_actual_ps_plan" in imported
    assert not {
        "sounddevice",
        "subprocess",
        "alsaaudio",
        "pyaudio",
        "portaudio",
        "deep_anc.audio_duplex_s32_disarmed_v10_3",
        "stage2_2khz_rt5640_s32",
        "stage2_2khz_rt5640_s32_capture",
        "stage2_2khz_measurement_v2",
    } & imported
    assert "build_stage2_v2_live_safe_fallback_plan" not in source
    assert "validate_stage2_v2_live_safe_fallback_plan" not in source
    assert "capture_disarmed_planned_s32_duplex" not in source
    assert not {"open", "write", "write_text", "write_bytes", "save", "savez"} & called
