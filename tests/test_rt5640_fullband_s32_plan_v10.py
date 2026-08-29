from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from deep_anc.dsp.rt5640_fullband_s32_plan_v10 import (
    RAW_TARGET_RELATIVE_PATH,
    build_rt5640_fullband_s32_plan_v10,
    q15_to_s32_exact,
    validate_rt5640_fullband_s32_plan_v10,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_q15_extrema_are_exact_s32_fullscale_equivalents() -> None:
    q15 = np.array(
        [[-32768, -1], [0, 1], [32767, -12345]], dtype="<i2"
    )
    s32 = q15_to_s32_exact(q15)
    assert s32.dtype == np.dtype("<i4")
    assert np.array_equal(
        s32,
        np.array(
            [
                [-2147483648, -65536],
                [0, 65536],
                [2147418112, -809041920],
            ],
            dtype="<i4",
        ),
    )
    assert np.array_equal(s32.astype(np.int64) >> 16, q15.astype(np.int64))
    assert not np.any(s32.astype(np.int64) & ((1 << 16) - 1))


@pytest.mark.parametrize(
    "bad",
    (
        np.zeros((4, 2), dtype="<i4"),
        np.zeros((4,), dtype="<i2"),
        np.zeros((4, 1), dtype="<i2"),
        np.zeros((4, 2), dtype=">i2"),
        np.zeros((4, 4), dtype="<i2")[:, ::2],
    ),
)
def test_q15_to_s32_rejects_wrong_dtype_shape_endian_or_strides(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        q15_to_s32_exact(bad)


def test_s32_plan_is_deterministic_and_binds_static_v3_source() -> None:
    plan_a, pcm_a = build_rt5640_fullband_s32_plan_v10()
    plan_b, pcm_b = build_rt5640_fullband_s32_plan_v10()
    assert plan_a == plan_b
    assert np.array_equal(pcm_a, pcm_b)
    assert plan_a["status"] == "BLOCKED_MISSING_S32_DUPLEX_AND_ELECTRICAL_WITNESS"
    assert plan_a["role"] == "signal_only_dry_run_no_audio"
    assert plan_a["control_band_contract_sha256"] == (
        "53579b9ff8419ac19fb2458c29a3e8a94ffbb2eeb88cc07f34b76c68033989f2"
    )
    assert plan_a["static_contract"]["config_file_sha256"] == (
        "5fe219b4e2026d09fffc276aa5ad7e99a84e46e47bdcdefe08284e7af83ecfa4"
    )
    assert plan_a["source_q15_signal_only"] == {
        "source_schema": "fullband_causal_time_separated_near_white_v5",
        "source_role": "signal_only_dry_run_no_audio",
        "source_canonical_payload_sha256": (
            "32a79b3700b457dc40373dc4dd0969301287baea7100b1ec5edd86ea907ee127"
        ),
        "source_control_band_contract_sha256": (
            "53579b9ff8419ac19fb2458c29a3e8a94ffbb2eeb88cc07f34b76c68033989f2"
        ),
        "source_q15_pcm_sha256": (
            "c18416e4066556479fd317659d908c215e6662d08f5bfa9d50e4ac63971c4aff"
        ),
        "source_q15_shape": [557056, 2],
        "source_q15_dtype": "<i2",
        "v5_live_authority_inherited": False,
        "v5_raw_publisher_inherited": False,
        "v5_meter_inherited": False,
    }
    assert pcm_a.dtype == np.dtype("<i4")
    assert pcm_a.shape == (557056, 2)
    assert pcm_a.nbytes == 4456448
    assert plan_a["expected_callbacks"] == 2176
    assert plan_a["duration_seconds"] == pytest.approx(11.605333333333334)
    assert plan_a["sealed_planned_s32_pcm_sha256"] == (
        "dc897bde69b60e8df81d4d677cd68ae030b375e704d5082b214c1dbacd40c6ce"
    )
    assert plan_a["sealed_planned_s32_min_pcm"] == -4521984
    assert plan_a["sealed_planned_s32_max_pcm"] == 4521984
    assert plan_a["future_raw_target"] == {
        "relative_path": RAW_TARGET_RELATIVE_PATH,
        "file_created_by_this_module": False,
        "raw_schema_created_by_this_module": False,
    }
    assert plan_a["authority"] == {
        "s32_signal_plan_pass": True,
        "s32_duplex_transport_pass": False,
        "hardware_frame_identity_pass": False,
        "electrical_witness_pass": False,
        "fullband_plant_identification_pass": False,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
    }
    checked_plan, checked_pcm = validate_rt5640_fullband_s32_plan_v10(plan_a, pcm_a)
    assert checked_plan == plan_a
    assert np.array_equal(checked_pcm, pcm_a)


def test_plan_rejects_pcm_cast_channel_swap_and_plan_mutation() -> None:
    plan, s32 = build_rt5640_fullband_s32_plan_v10()
    q15 = (s32.astype(np.int64) >> 16).astype("<i2")
    simple_cast = np.ascontiguousarray(q15.astype("<i4"))
    with pytest.raises(ValueError, match="canonical plan"):
        validate_rt5640_fullband_s32_plan_v10(plan, simple_cast)
    with pytest.raises(ValueError, match="canonical plan"):
        validate_rt5640_fullband_s32_plan_v10(plan, np.ascontiguousarray(s32[:, ::-1]))
    changed = deepcopy(plan)
    changed["authority"]["canonical_training_eligible"] = True
    with pytest.raises(ValueError, match="canonical payload"):
        validate_rt5640_fullband_s32_plan_v10(changed, s32)


@pytest.mark.parametrize(
    "raw_path",
    (
        "raw_capture.npz",
        "/tmp/raw_capture.npz",
        "results/../raw_capture.npz",
        "results/rt5640_fullband_v10/raw_capture.wav",
        "results/rt5640_fullband_v10/other_capture.npz",
    ),
)
def test_plan_rejects_noncanonical_raw_targets(raw_path: str) -> None:
    with pytest.raises(ValueError):
        build_rt5640_fullband_s32_plan_v10(raw_target_relative_path=raw_path)


def test_signal_only_plan_imports_no_audio_backend_or_writer() -> None:
    source = (
        REPO_ROOT / "src/deep_anc/dsp/rt5640_fullband_s32_plan_v10.py"
    ).read_text(encoding="utf-8")
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
    assert not {"sounddevice", "subprocess"} & imported
    assert not {"save", "savez", "write", "write_bytes", "write_text"} & called
