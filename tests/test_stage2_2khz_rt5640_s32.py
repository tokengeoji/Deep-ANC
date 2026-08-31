from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from deep_anc.dsp.stage2_2khz_rt5640_s32 import (
    DEFAULT_CONFIG_RELATIVE_PATH,
    Q15_TO_S32_LEFT_SHIFT,
    Stage2Rt5640S32ContractError,
    build_stage2_rt5640_s32_planned_transport_provenance,
    build_stage2_rt5640_s32_signal_plan,
    load_stage2_rt5640_s32_static_contract,
    q15_to_stage2_rt5640_s32_exact,
    stage2_rt5640_s32_to_q15_exact,
    validate_stage2_rt5640_s32_planned_transport_provenance,
    validate_stage2_rt5640_s32_signal_plan,
    validate_stage2_rt5640_s32_static_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return yaml.safe_load((REPO_ROOT / DEFAULT_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8"))


def test_static_contract_is_ape_same_card_s32_and_explicitly_blocks_legacy_receipts() -> None:
    receipt = validate_stage2_rt5640_s32_static_contract(_config())
    assert receipt["status"] == "BLOCKED_MISSING_RT5640_S32_CAPTURE_ADAPTER_AND_PHYSICAL_RAW"
    assert receipt["static_gate_pass"] is True
    assert receipt["audio_opened"] is False
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
    assert receipt["forbidden_receipt_origins"] == {
        "usb_ab13x": True,
        "output_master_split_clock": True,
        "s16_transport": True,
        "legacy_relabel_or_promotion": True,
    }
    assert receipt["authority"]["canonical_training_eligible"] is False


@pytest.mark.parametrize(
    "mutate,match",
    (
        (lambda cfg: cfg["audio"]["input"].__setitem__("pcm", 0), "input.pcm"),
        (lambda cfg: cfg["audio"]["output"].__setitem__("card", "Audio"), "output.card"),
        (lambda cfg: cfg["audio"]["output"].__setitem__("format", "S16_LE"), "output.format"),
        (lambda cfg: cfg["audio"].__setitem__("clock_domain", "USB_ADAPTIVE"), "clock_domain"),
        (lambda cfg: cfg["stage2_2khz"].__setitem__("usb_ab13x_receipt_reuse_allowed", True), "usb_ab13x"),
        (lambda cfg: cfg["stage2_2khz"].__setitem__("output_master_receipt_reuse_allowed", True), "output_master"),
        (lambda cfg: cfg["stage2_2khz"].__setitem__("s16_receipt_reuse_allowed", True), "s16_receipt"),
        (lambda cfg: cfg["authority"].__setitem__("stage2_ps_identification_pass", True), "stage2_ps"),
    ),
)
def test_static_contract_rejects_other_transport_or_authority(
    mutate, match: str
) -> None:  # type: ignore[no-untyped-def]
    payload = deepcopy(_config())
    mutate(payload)
    with pytest.raises(Stage2Rt5640S32ContractError, match=match):
        validate_stage2_rt5640_s32_static_contract(payload)


def test_q15_s32_extrema_low_bits_and_byte_exact_inverse() -> None:
    q15 = np.array([[-32768, -1], [0, 1], [32767, -12345]], dtype="<i2")
    s32 = q15_to_stage2_rt5640_s32_exact(q15)
    assert s32.dtype == np.dtype("<i4")
    assert np.array_equal(
        s32,
        np.array(
            [[-2147483648, -65536], [0, 65536], [2147418112, -809041920]],
            dtype="<i4",
        ),
    )
    assert not np.any(s32.astype(np.int64) & ((1 << Q15_TO_S32_LEFT_SHIFT) - 1))
    assert np.array_equal(stage2_rt5640_s32_to_q15_exact(s32), q15)

    malformed = s32.copy()
    malformed[0, 0] += 1
    with pytest.raises(Stage2Rt5640S32ContractError, match="low 16 bits"):
        stage2_rt5640_s32_to_q15_exact(malformed)


def test_signal_plan_binds_actual_stage2_int16_plan_scale_and_provenance() -> None:
    plan, pcm = build_stage2_rt5640_s32_signal_plan()
    assert plan["source_int16_plan"]["schema"] == "stage2_2khz_time_separated_lower_guard_dpss_plan_v2"
    assert plan["source_int16_plan"]["role"] == "signal_only_live_safe_fallback_no_audio_authority"
    assert plan["source_int16_plan"]["actual_submitted_pcm_dtype"] == "<i2"
    assert pcm.dtype == np.dtype("<i4")
    assert pcm.shape[1] == 2
    assert pcm.shape[0] % 256 == 0
    assert not np.any(pcm.astype(np.int64) & ((1 << 16) - 1))
    assert np.array_equal(
        stage2_rt5640_s32_to_q15_exact(pcm).astype(np.int64),
        pcm.astype(np.int64) >> Q15_TO_S32_LEFT_SHIFT,
    )
    checked, checked_pcm = validate_stage2_rt5640_s32_signal_plan(plan, pcm)
    assert checked == plan
    assert np.array_equal(checked_pcm, pcm)

    provenance = build_stage2_rt5640_s32_planned_transport_provenance(plan, pcm)
    assert provenance["hardware_contract"]["native_format"] == "S32_LE"
    assert provenance["hardware_contract"]["same_card"] is True
    assert provenance["prohibited_receipt_lineage"]["output_master_split_clock"] is True
    assert provenance["actual_capture_present"] is False
    assert validate_stage2_rt5640_s32_planned_transport_provenance(provenance, plan, pcm) == provenance


def test_plan_or_provenance_mutation_cannot_relabel_usb_s16_or_output_master() -> None:
    plan, pcm = build_stage2_rt5640_s32_signal_plan()
    simple_cast = np.ascontiguousarray(stage2_rt5640_s32_to_q15_exact(pcm).astype("<i4"))
    with pytest.raises(Stage2Rt5640S32ContractError, match="canonical plan"):
        validate_stage2_rt5640_s32_signal_plan(plan, simple_cast)

    provenance = build_stage2_rt5640_s32_planned_transport_provenance(plan, pcm)
    for key, value in (("native_format", "S16_LE"), ("same_card", False), ("same_clock_domain", "USB_ADAPTIVE")):
        forged = deepcopy(provenance)
        forged["hardware_contract"][key] = value
        with pytest.raises(Stage2Rt5640S32ContractError, match="USB/output-master/S16"):
            validate_stage2_rt5640_s32_planned_transport_provenance(forged, plan, pcm)


def test_static_load_is_sealed_to_default_config() -> None:
    receipt = load_stage2_rt5640_s32_static_contract(repository_root=REPO_ROOT)
    assert receipt["config"]["path"].endswith(DEFAULT_CONFIG_RELATIVE_PATH)
    assert len(receipt["config"]["file_sha256"]) == 64
    with pytest.raises(Stage2Rt5640S32ContractError, match="sealed default"):
        load_stage2_rt5640_s32_static_contract(
            REPO_ROOT / "configs/hardware_jetson_rt5640_fullband_v10.yaml",
            repository_root=REPO_ROOT,
        )


def test_cli_is_default_dry_run_and_never_imports_or_opens_audio_backend() -> None:
    source_paths = (
        REPO_ROOT / "src/deep_anc/dsp/stage2_2khz_rt5640_s32.py",
        REPO_ROOT / "scripts/jetson/check_stage2_2khz_rt5640_s32.py",
    )
    tree = ast.parse("\n".join(path.read_text(encoding="utf-8") for path in source_paths))
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
    assert not {"save", "savez", "write", "write_text", "write_bytes", "open"} & called

    result = subprocess.run(
        [sys.executable, "-B", "scripts/jetson/check_stage2_2khz_rt5640_s32.py", "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "audio was not opened" in result.stdout
    assert "speaker_output=0" in result.stdout
