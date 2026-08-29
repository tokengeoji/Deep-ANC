from __future__ import annotations

import ast
import hashlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from deep_anc.audio_duplex_s32_disarmed_v10_3 import S32DisarmedDuplexCaptureFailure
from deep_anc.dsp.rt5640_fullband_s32_plan_v10 import (
    build_rt5640_fullband_s32_plan_v10,
)
from deep_anc.dsp.rt5640_s32_capture_admission_v10_3 import (
    BLOCKED_STATUS,
    POST_START_PRE_ARM_RECEIPT_SCHEMA,
    STATUS,
    assess_rt5640_s32_capture_admission_v10_3,
    validate_rt5640_s32_capture_admission_v10_3,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@pytest.fixture(scope="module")
def sealed_plan() -> tuple[dict, np.ndarray]:
    return build_rt5640_fullband_s32_plan_v10()


def _post_start_receipt() -> dict:
    return {
        "schema": POST_START_PRE_ARM_RECEIPT_SCHEMA,
        "passed": True,
        "observed_after_stream_start": True,
        "resolved_input_device": 11,
        "resolved_output_device": 12,
        "input_hw_params": {
            "card": "APE",
            "pcm": 1,
            "format": "S32_LE",
            "sample_rate_hz": 48_000,
            "channels": 2,
            "period_frames": 256,
        },
        "output_hw_params": {
            "card": "APE",
            "pcm": 0,
            "format": "S32_LE",
            "sample_rate_hz": 48_000,
            "channels": 2,
            "period_frames": 256,
        },
        "routes": {
            "input": "I2S2_ADMAIF2_ERR_REF",
            "output": "ADMAIF1_I2S1_RT5640_J511",
            "observed_after_stream_start": True,
        },
        "j511": {"state": "HP", "samples": 3, "all_samples_equal": True},
        "occupancy": {
            "input_owned_by_this_capture": True,
            "output_owned_by_this_capture": True,
            "other_pcm_owners": [],
        },
        "pre_snapshot_sha256": "a" * 64,
        "post_snapshot_sha256": "b" * 64,
    }


def _telemetry(plan: dict, pcm: np.ndarray) -> tuple[np.ndarray, dict]:
    frames = len(pcm)
    callbacks = frames // 256
    sequence = np.arange(callbacks, dtype="<i8")
    starts = sequence * 256
    counts = np.full(callbacks, 256, dtype="<i8")
    status = np.zeros(callbacks, dtype="<u4")
    planned_adc = np.arange(10, 10 + callbacks, dtype="<f8")
    planned_dac = planned_adc + 0.1
    planned_current = planned_adc + 0.2
    prearm_adc = np.array([1.0], dtype="<f8")
    captured = np.zeros((frames, 2), dtype="<i4")
    nominal = frames / 48_000.0
    telemetry = {
        "schema": "rt5640_fullband_s32_disarmed_duplex_telemetry_v10_3",
        "authority": "application_buffer_disarmed_planned_s32_only_no_physical_sample_identity",
        "portaudio_application_buffer_only": True,
        "post_start_pre_arm_check_required": True,
        "post_start_pre_arm_check_passed": True,
        "nonzero_assignment_before_arm_allowed": False,
        "on_stream_started_is_nonzero_admission": False,
        "hardware_sample_slip_authority": False,
        "physical_output_authority": False,
        "electrical_output_authority": False,
        "acoustic_output_authority": False,
        "sample_rate_hz": 48_000,
        "block_size": 256,
        "latency": "low",
        "channels": [2, 2],
        "input_dtype": "<i4",
        "output_dtype": "<i4",
        "dither_off": True,
        "planned_pcm_sha256": plan["sealed_planned_s32_pcm_sha256"],
        "planned_frames": frames,
        "expected_planned_callbacks": callbacks,
        "resolved_input_device": 11,
        "resolved_output_device": 12,
        "pre_open_monotonic_started": 0.0,
        "pre_open_monotonic_completed": 0.1,
        "capture_monotonic_started": 1.0,
        "arm_check_monotonic_started": 1.1,
        "arm_check_monotonic_completed": 1.2,
        "arm_monotonic": 1.3,
        "capture_monotonic_completed": 1.0 + nominal,
        "capture_monotonic_elapsed_seconds": nominal,
        "watchdog_grace_seconds": 2.0,
        "prearm_callback_sequence": np.array([0], dtype="<i8"),
        "prearm_callback_start_frames": np.array([0], dtype="<i8"),
        "prearm_callback_frame_counts": np.array([256], dtype="<i8"),
        "prearm_input_buffer_adc_time": prearm_adc,
        "prearm_output_buffer_dac_time": prearm_adc + 0.1,
        "prearm_callback_current_time": prearm_adc + 0.2,
        "prearm_callback_status_bitmask": np.zeros(1, dtype="<u4"),
        "prearm_callback_count": 1,
        "prearm_output_zero_observed": True,
        "planned_callback_sequence": sequence,
        "planned_callback_start_frames": starts,
        "planned_callback_frame_counts": counts,
        "planned_input_buffer_adc_time": planned_adc,
        "planned_output_buffer_dac_time": planned_dac,
        "planned_callback_current_time": planned_current,
        "planned_callback_status_bitmask": status,
        "xrun_count": 0,
        "status_present_count": 0,
        "captured_frames": frames,
        "submitted_frames": frames,
        "callback_zero_attempt_count": callbacks + 1,
        "callback_zero_confirmed_count": callbacks + 1,
        "callback_planned_assignment_attempt_count": callbacks,
        "callback_planned_assignment_confirmed_count": callbacks,
        "possible_nonzero_output_after_failed_assignment": False,
        "planned_callback_sequence_contiguous": True,
        "planned_callback_start_frames_contiguous": True,
        "planned_callback_frame_counts_exact": True,
        "capture_valid_all_true": True,
        "submitted_valid_all_true": True,
        "full_frame_accounting_valid": True,
        "actual_matches_planned_application_buffer": True,
        "actual_submitted_pcm_hash_eligible": True,
        "actual_submitted_pcm_sha256": _array_sha256(pcm),
        "actual_submitted_pcm": pcm.copy(),
        "capture_valid_mask": np.ones(frames, dtype=np.bool_),
        "submitted_valid_mask": np.ones(frames, dtype=np.bool_),
        "completed": True,
        "callback_error": None,
        "canonical_invalid_reasons": [],
        "stream_constructor_error": None,
        "stream_start_error": None,
        "post_start_pre_arm_check_error": None,
        "watchdog_error": None,
        "stream_stop_error": None,
        "stream_abort_error": None,
        "stream_close_error": None,
        "on_output_closed_error": None,
        "termination_signals": [],
        "stream_start_returned_without_exception": True,
        "stream_stop_attempted": True,
        "stream_stop_returned_without_exception": True,
        "stream_abort_attempted": False,
        "stream_abort_returned_without_exception": False,
        "stream_close_attempted": True,
        "stream_close_returned_without_exception": True,
        "normal_stop_completed": True,
        "faults": [],
    }
    return captured, telemetry


def test_complete_s32_capture_is_transport_only_not_electrical_or_training(
    sealed_plan: tuple[dict, np.ndarray],
) -> None:
    plan, pcm = sealed_plan
    captured, telemetry = _telemetry(plan, pcm)
    receipt = validate_rt5640_s32_capture_admission_v10_3(
        plan=plan,
        planned_pcm_s32=pcm,
        captured_pcm=captured,
        telemetry=telemetry,
        post_start_pre_arm_receipt=_post_start_receipt(),
    )
    assert receipt["status"] == STATUS
    assert receipt["transport_capture_pass"] is True
    assert receipt["actual_submitted_s32_pcm_sha256"] == plan["sealed_planned_s32_pcm_sha256"]
    assert len(receipt["post_start_pre_arm_receipt_payload_sha256"]) == 64
    assert receipt["authority"] == {
        "s32_transport_capture_pass": True,
        "hardware_sample_slip_authority": False,
        "physical_output_authority": False,
        "electrical_witness_bound": False,
        "fullband_plant_identification_pass": False,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
    }


@pytest.mark.parametrize(
    "mutate,match",
    (
        (
            lambda capture, telemetry, post: telemetry.__setitem__("xrun_count", 1),
            "xrun_count",
        ),
        (
            lambda capture, telemetry, post: telemetry["capture_valid_mask"].__setitem__(0, False),
            "valid mask",
        ),
        (
            lambda capture, telemetry, post: telemetry["actual_submitted_pcm"].__setitem__((0, 0), 1),
            "actual submitted S32 PCM",
        ),
        (
            lambda capture, telemetry, post: post["j511"].__setitem__("state", "None"),
            "J511",
        ),
        (
            lambda capture, telemetry, post: post["occupancy"].__setitem__("other_pcm_owners", ["other"]),
            "other_pcm_owners",
        ),
        (
            lambda capture, telemetry, post: post["input_hw_params"].__setitem__("format", "S16_LE"),
            "input_hw_params.format",
        ),
    ),
)
def test_capture_admission_rejects_transport_and_post_start_shortcuts(
    sealed_plan: tuple[dict, np.ndarray], mutate, match: str  # type: ignore[no-untyped-def]
) -> None:
    plan, pcm = sealed_plan
    captured, telemetry = _telemetry(plan, pcm)
    telemetry = deepcopy(telemetry)
    post = _post_start_receipt()
    mutate(captured, telemetry, post)
    with pytest.raises(ValueError, match=match):
        validate_rt5640_s32_capture_admission_v10_3(
            plan=plan,
            planned_pcm_s32=pcm,
            captured_pcm=captured,
            telemetry=telemetry,
            post_start_pre_arm_receipt=post,
        )


def test_partial_disarmed_failure_becomes_explicit_blocked_receipt(
    sealed_plan: tuple[dict, np.ndarray],
) -> None:
    plan, pcm = sealed_plan
    partial = np.zeros_like(pcm)
    failure = S32DisarmedDuplexCaptureFailure(
        "simulated callback failure",
        partial,
        partial.copy(),
        np.zeros(len(pcm), dtype=np.bool_),
        np.zeros(len(pcm), dtype=np.bool_),
        {"schema": "rt5640_fullband_s32_disarmed_duplex_telemetry_v10_3"},
    )
    receipt = assess_rt5640_s32_capture_admission_v10_3(
        plan=plan,
        planned_pcm_s32=pcm,
        capture=failure,
        post_start_pre_arm_receipt=None,
    )
    assert receipt["status"] == BLOCKED_STATUS
    assert receipt["transport_capture_pass"] is False
    assert receipt["authority"]["canonical_training_eligible"] is False


def test_mutated_plan_and_missing_post_start_receipt_are_rejected(
    sealed_plan: tuple[dict, np.ndarray],
) -> None:
    plan, pcm = sealed_plan
    captured, telemetry = _telemetry(plan, pcm)
    forged = deepcopy(plan)
    forged["authority"]["canonical_training_eligible"] = True
    with pytest.raises(ValueError, match="canonical payload"):
        validate_rt5640_s32_capture_admission_v10_3(
            plan=forged,
            planned_pcm_s32=pcm,
            captured_pcm=captured,
            telemetry=telemetry,
            post_start_pre_arm_receipt=_post_start_receipt(),
        )
    with pytest.raises(ValueError, match="post-start"):
        assess_rt5640_s32_capture_admission_v10_3(
            plan=plan,
            planned_pcm_s32=pcm,
            capture=(captured, telemetry),
            post_start_pre_arm_receipt=None,
        )


def test_capture_admission_source_has_no_audio_backend_or_result_writer() -> None:
    source = (
        REPO_ROOT / "src/deep_anc/dsp/rt5640_s32_capture_admission_v10_3.py"
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
    assert not {"sounddevice", "subprocess", "torch"} & imported
    assert not {"save", "savez", "write", "write_text", "write_bytes"} & called
