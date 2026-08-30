from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from deep_anc.dsp.rt5640_s32_meter_v10_3 import (
    DEFAULT_CONFIG_RELATIVE_PATH_V10_3,
    METER_FRAMES,
    build_rt5640_s32_level_control_plan_v10_3,
    load_rt5640_s32_meter_static_contract_v10_3,
    validate_rt5640_s32_level_control_plan_v10_3,
    validate_rt5640_s32_meter_static_contract_v10_3,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / DEFAULT_CONFIG_RELATIVE_PATH_V10_3).read_text(encoding="utf-8")
    )


def test_s32_meter_static_contract_is_level_only_and_no_audio() -> None:
    receipt = validate_rt5640_s32_meter_static_contract_v10_3(
        _config(), repository_root=REPO_ROOT
    )
    assert receipt["status"] == "BLOCKED"
    assert receipt["static_gate_pass"] is True
    assert receipt["audio_opened"] is False
    assert receipt["results_written"] is False
    plan = receipt["level_control_plan"]
    assert plan["frames"] == METER_FRAMES == 960_000
    assert plan["callbacks"] == 3750
    assert plan["duration_seconds"] == 20.0
    assert plan["level_control_recipe"]["probe_peak_normalized"] == 0.003
    assert plan["level_control_recipe"]["actual_q15_peak"] == 98
    assert plan["level_control_recipe"]["actual_s32_peak"] == 98 << 16
    assert plan["authority"]["full_octave_health_pass"] is False
    assert plan["authority"]["canonical_training_eligible"] is False


def test_s32_meter_pcm_is_exact_q15_left_shift_and_ch1_zero() -> None:
    plan, pcm = build_rt5640_s32_level_control_plan_v10_3()
    assert pcm.dtype == np.dtype("<i4")
    assert pcm.shape == (METER_FRAMES, 2)
    assert pcm.flags.c_contiguous
    assert not np.any(pcm[:, 1])
    assert not np.any(np.bitwise_and(pcm.astype(np.int64), (1 << 16) - 1))
    expected_plan, expected_pcm = validate_rt5640_s32_level_control_plan_v10_3(plan, pcm)
    assert expected_plan == plan
    assert np.array_equal(expected_pcm, pcm)

    wrong_scale = pcm.copy()
    wrong_scale[:, 0] //= 1 << 16
    with pytest.raises(ValueError, match="canonical recipe"):
        validate_rt5640_s32_level_control_plan_v10_3(plan, wrong_scale)

    wrong_ch1 = pcm.copy()
    wrong_ch1[0, 1] = 1 << 16
    with pytest.raises(ValueError, match="canonical recipe"):
        validate_rt5640_s32_level_control_plan_v10_3(plan, wrong_ch1)


@pytest.mark.parametrize(
    "mutate,match",
    (
        (lambda cfg: cfg.__setitem__("schema", "legacy"), "schema"),
        (
            lambda cfg: cfg["fullband_v3"].__setitem__(
                "control_band_contract_sha256", "0" * 64
            ),
            "SHA",
        ),
        (
            lambda cfg: cfg["level_meter"].__setitem__("probe_peak_normalized", 0.01),
            "probe_peak",
        ),
        (
            lambda cfg: cfg["level_meter"].__setitem__("disarmed_stream_required", False),
            "disarmed",
        ),
        (
            lambda cfg: cfg["authority"].__setitem__("canonical_training_eligible", True),
            "authority",
        ),
    ),
)
def test_s32_meter_rejects_legacy_or_authority_promotion(mutate, match: str) -> None:  # type: ignore[no-untyped-def]
    config = deepcopy(_config())
    mutate(config)
    with pytest.raises(ValueError, match=match):
        validate_rt5640_s32_meter_static_contract_v10_3(
            config, repository_root=REPO_ROOT
        )


def test_s32_meter_config_file_is_sealed_and_hash_bound() -> None:
    receipt = load_rt5640_s32_meter_static_contract_v10_3(repository_root=REPO_ROOT)
    assert receipt["config"]["path"].endswith(DEFAULT_CONFIG_RELATIVE_PATH_V10_3)
    assert len(receipt["config"]["file_sha256"]) == 64
    with pytest.raises(ValueError, match="sealed default config"):
        load_rt5640_s32_meter_static_contract_v10_3(
            REPO_ROOT / "configs/hardware_jetson_rt5640_fullband_v10.yaml",
            repository_root=REPO_ROOT,
        )


def test_s32_meter_static_source_has_no_audio_backend_or_result_writer() -> None:
    sources = [
        REPO_ROOT / "src/deep_anc/dsp/rt5640_s32_meter_v10_3.py",
        REPO_ROOT / "scripts/jetson/check_rt5640_s32_meter_static_v10_3.py",
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
