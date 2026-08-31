from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from deep_anc.dsp.stage2_2khz_actual_ps_plan import (
    DEFAULT_CONFIG_RELATIVE_PATH,
    PLAN_SCHEMA,
    Q15_TO_S32_LEFT_SHIFT,
    PROVENANCE_SCHEMA,
    Stage2ActualPsPlanError,
    build_stage2_actual_ps_excitation_plan,
    build_stage2_actual_ps_planned_provenance,
    load_stage2_actual_ps_static_config,
    q15_to_stage2_actual_ps_s32_exact,
    stage2_actual_ps_s32_to_q15_exact,
    validate_stage2_actual_ps_excitation_plan,
    validate_stage2_actual_ps_planned_provenance,
    validate_stage2_actual_ps_static_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE = REPO_ROOT / "src/deep_anc/dsp/stage2_2khz_actual_ps_plan.py"


def _config() -> dict[str, object]:
    return yaml.safe_load((REPO_ROOT / DEFAULT_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8"))


def test_actual_ps_plan_uses_full_pe_source_and_preserves_ps_time_role_channel_mapping() -> None:
    plan, s32 = build_stage2_actual_ps_excitation_plan()

    assert plan["schema"] == PLAN_SCHEMA
    assert plan["status"] == "PLAN_ONLY_ACTUAL_PS_CAPTURE_NOT_AUTHORIZED"
    assert plan["role"] == "actual_ps_excitation_preparation_no_audio_no_training_authority"
    assert plan["duration_seconds"] == 24.0
    assert plan["expected_callbacks"] == 4_500
    assert s32.dtype == np.dtype("<i4")
    assert s32.shape == (1_152_000, 2)
    assert not np.any(s32.astype(np.int64) & ((1 << Q15_TO_S32_LEFT_SHIFT) - 1))

    source = plan["source_measurement_plan"]
    assert source["schema"] == "stage2_2khz_time_separated_full_pe_plan_v2"
    assert source["role"] == "signal_only_no_audio_no_training_authority"
    assert source["source_transport_inherited"] is False
    assert source["source_usb_or_s16_receipt_usable"] is False
    assert source["source_output_master_receipt_usable"] is False
    assert source["source_fallback_plan_usable"] is False
    assert source["source_audio_execution_allowed"] is False
    assert source["source_physical_ps_authority"] is False
    assert len(source["time_role_channel_mapping"]) == 6

    slots = source["time_role_channel_mapping"]
    assert [(row["time_role"], row["path"]) for row in slots] == [
        ("fit_a", "primary"),
        ("fit_a", "secondary"),
        ("fit_b", "secondary"),
        ("fit_b", "primary"),
        ("untouched_holdout", "secondary"),
        ("untouched_holdout", "primary"),
    ]
    assert [row["output_channel"] for row in slots] == [0, 1, 1, 0, 1, 0]
    assert [row["stimulus_role"] for row in slots] == ["NS", "CS", "CS", "NS", "CS", "NS"]
    assert all(row["source_slot_frames"] == 65_536 for row in slots)
    assert all(row["source_central_frames"] == 32_768 for row in slots)
    assert all(row["fit_input_allowed"] is True for row in slots[:4])
    assert all(row["holdout_prediction_only"] is True for row in slots[4:])

    mapping = plan["ps_time_role_channel_mapping"]
    assert mapping["primary"]["stimulus_role"] == "NS"
    assert mapping["primary"]["output_channel"] == 0
    assert mapping["secondary"]["stimulus_role"] == "CS"
    assert mapping["secondary"]["output_channel"] == 1
    assert mapping["primary"]["response_capture_channels"] == {"ERR": 0, "REF": 1}
    assert mapping["secondary"]["response_capture_channels"] == {"ERR": 0, "REF": 1}

    authority = plan["authority"]
    assert authority == {
        "actual_ps_excitation_plan_prepared": True,
        "audio_output_performed": False,
        "same_card_s32_transport_pass": False,
        "physical_raw_present": False,
        "physical_ps_authority": False,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
    }


def test_q15_s32_exact_conversion_extrema_low_bits_and_inverse() -> None:
    q15 = np.zeros((256, 2), dtype="<i2")
    q15[:3] = np.array([[-32768, -1], [0, 1], [32767, -12345]], dtype="<i2")

    s32 = q15_to_stage2_actual_ps_s32_exact(q15)
    assert np.array_equal(
        s32[:3],
        np.array(
            [[-2147483648, -65536], [0, 65536], [2147418112, -809041920]],
            dtype="<i4",
        ),
    )
    assert not np.any(s32.astype(np.int64) & ((1 << 16) - 1))
    assert np.array_equal(stage2_actual_ps_s32_to_q15_exact(s32), q15)

    malformed = s32.copy()
    malformed[0, 0] += 1
    with pytest.raises(Stage2ActualPsPlanError, match="low 16 bits"):
        stage2_actual_ps_s32_to_q15_exact(malformed)


def test_static_config_is_sealed_same_card_s32_and_forbids_relabelled_origins() -> None:
    receipt = validate_stage2_actual_ps_static_config(_config())
    assert receipt["status"] == "PLAN_ONLY_ACTUAL_PS_CAPTURE_NOT_AUTHORIZED"
    assert receipt["audio_opened"] is False
    assert receipt["speaker_output"] is False
    assert receipt["results_written"] is False
    assert receipt["hardware_audio"]["input"] == {
        "card": "APE",
        "pcm": 1,
        "channels": 2,
        "format": "S32_LE",
        "route": "I2S2_ADMAIF2_ERR_REF",
    }
    assert receipt["hardware_audio"]["output"] == {
        "card": "APE",
        "pcm": 0,
        "channels": 2,
        "format": "S32_LE",
        "route": "ADMAIF1_I2S1_RT5640_J511",
    }
    assert receipt["forbidden_source_or_receipt_origins"] == {
        "usb_ab13x": True,
        "output_master_split_clock": True,
        "bandlimited_fallback": True,
        "s16_transport": True,
        "legacy_relabel_or_promotion": True,
    }
    assert receipt["authority"]["physical_ps_authority"] is False
    assert receipt["authority"]["canonical_training_eligible"] is False


@pytest.mark.parametrize(
    "mutate,match",
    (
        (lambda cfg: cfg["audio"]["output"].__setitem__("format", "S16_LE"), "output.format"),
        (lambda cfg: cfg["audio"]["output"].__setitem__("card", "Audio"), "output.card"),
        (lambda cfg: cfg["stage2_2khz"].__setitem__("fallback_plan_reuse_allowed", True), "fallback_plan"),
        (lambda cfg: cfg["stage2_2khz"].__setitem__("source_transport_inherited", True), "source_transport"),
        (lambda cfg: cfg["authority"].__setitem__("physical_ps_authority", True), "physical_ps"),
    ),
)
def test_static_config_rejects_transport_relabel_or_authority_promotion(mutate, match: str) -> None:  # type: ignore[no-untyped-def]
    payload = deepcopy(_config())
    mutate(payload)
    with pytest.raises(Stage2ActualPsPlanError, match=match):
        validate_stage2_actual_ps_static_config(payload)


def test_plan_and_provenance_bind_source_mapping_s32_and_remain_planned_only() -> None:
    plan, s32 = build_stage2_actual_ps_excitation_plan()
    checked, checked_s32 = validate_stage2_actual_ps_excitation_plan(plan, s32)
    assert checked == plan
    assert np.array_equal(checked_s32, s32)

    provenance = build_stage2_actual_ps_planned_provenance(plan, s32)
    assert provenance["schema"] == PROVENANCE_SCHEMA
    assert provenance["actual_ps_plan_sha256"] == plan["canonical_payload_sha256"]
    assert provenance["source_measurement_plan_sha256"] == plan["source_measurement_plan"]["canonical_payload_sha256"]
    assert provenance["source_time_role_channel_mapping_sha256"] == plan["source_measurement_plan"]["time_role_channel_mapping_sha256"]
    assert provenance["hardware_contract"]["native_format"] == "S32_LE"
    assert provenance["hardware_contract"]["same_card"] is True
    assert provenance["actual_capture_present"] is False
    assert provenance["actual_audio_output_claimed"] is False
    assert provenance["actual_ps_or_training_authority_claimed"] is False
    assert validate_stage2_actual_ps_planned_provenance(provenance, plan, s32) == provenance

    forged_plan = deepcopy(plan)
    forged_plan["source_measurement_plan"]["time_role_channel_mapping"][0]["output_channel"] = 1
    with pytest.raises(Stage2ActualPsPlanError, match="mapping"):
        validate_stage2_actual_ps_excitation_plan(forged_plan, s32)

    forged_pcm = s32.copy()
    forged_pcm[0, 0] += 1
    with pytest.raises(Stage2ActualPsPlanError, match="canonical bytes"):
        validate_stage2_actual_ps_excitation_plan(plan, forged_pcm)

    forged_provenance = deepcopy(provenance)
    forged_provenance["hardware_contract"]["native_format"] = "S16_LE"
    with pytest.raises(Stage2ActualPsPlanError, match="USB/output-master/fallback"):
        validate_stage2_actual_ps_planned_provenance(forged_provenance, plan, s32)


def test_plan_only_code_never_imports_audio_backend_or_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert not {"sounddevice", "subprocess", "alsa", "pyaudio"} & imported
    assert "build_stage2_v2_live_safe_fallback_plan" not in source
    assert "capture_disarmed_planned_s32_duplex" not in source
    assert not {"save", "savez", "write", "write_text", "write_bytes", "open"} & called

    monkeypatch.chdir(tmp_path)
    receipt = load_stage2_actual_ps_static_config(repository_root=REPO_ROOT)
    plan, _ = build_stage2_actual_ps_excitation_plan()
    assert receipt["audio_opened"] is False
    assert plan["authority"]["audio_output_performed"] is False
    assert list(tmp_path.iterdir()) == []
