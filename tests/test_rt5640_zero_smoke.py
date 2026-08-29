from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from deep_anc.realtime.rt5640_zero_smoke import (
    AUTHORITY_CEILING,
    BLOCK_SIZE,
    CHANNELS,
    PCM_DTYPE,
    SAMPLE_RATE_HZ,
    TELEMETRY_SCHEMA,
    build_zero_duplex_plan,
    build_zero_duplex_receipt,
    canonical_json_bytes,
    capture_telemetry_to_contract,
    payload_sha256,
    validate_zero_duplex_plan,
    validate_zero_duplex_receipt,
    validate_zero_duplex_telemetry,
)


def _make_plan() -> tuple[dict[str, object], np.ndarray]:
    return build_zero_duplex_plan(frame_count=4 * BLOCK_SIZE)


def _make_telemetry(plan: dict[str, object], pcm: np.ndarray) -> dict[str, object]:
    callbacks = int(plan["callback_count"])
    frames = int(plan["frame_count"])
    callback_seconds = BLOCK_SIZE / SAMPLE_RATE_HZ
    times = np.arange(callbacks, dtype="<f8") * callback_seconds
    pre_started = 1.0
    pre_completed = 2.0
    started = 3.0
    completed = started + float(plan["nominal_duration_seconds"])
    return {
        "schema": TELEMETRY_SCHEMA,
        "authority": "zero_duplex_transport_only_no_sample_identity",
        "callback_frame_semantics": (
            "software_accounting_only_not_hardware_slip_witness"
        ),
        "output_zero_scope": "portaudio_application_callback_buffer_only",
        "portaudio_application_buffer_only": True,
        "portaudio_timestamp_authority": False,
        "hardware_sample_slip_authority": False,
        "physical_output_zero_authority": False,
        "electrical_output_zero_authority": False,
        "acoustic_output_zero_authority": False,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "block_size": BLOCK_SIZE,
        "latency": "low",
        "channels": [CHANNELS, CHANNELS],
        "input_dtype": PCM_DTYPE.str,
        "output_dtype": PCM_DTYPE.str,
        "dither_off": True,
        "resolved_input_device": 0,
        "resolved_output_device": 1,
        "planned_frames": frames,
        "expected_callbacks": callbacks,
        "pre_open_monotonic_started": pre_started,
        "pre_open_monotonic_completed": pre_completed,
        "capture_monotonic_started": started,
        "capture_monotonic_completed": completed,
        "capture_monotonic_elapsed_seconds": completed - started,
        "watchdog_grace_seconds": 2.0,
        "callback_sequence": np.arange(callbacks, dtype="<i8"),
        "callback_start_frames": np.arange(callbacks, dtype="<i8") * BLOCK_SIZE,
        "callback_frame_counts": np.full(callbacks, BLOCK_SIZE, dtype="<i8"),
        "input_buffer_adc_time": np.ascontiguousarray(100.0 + times, dtype="<f8"),
        "output_buffer_dac_time": np.ascontiguousarray(100.1 + times, dtype="<f8"),
        "callback_current_time": np.ascontiguousarray(100.2 + times, dtype="<f8"),
        "callback_status_bitmask": np.zeros(callbacks, dtype="<u4"),
        "xrun_count": 0,
        "status_present_count": 0,
        "captured_frames": frames,
        "submitted_frames": frames,
        "actual_submitted_nonzero_count": 0,
        "callback_zero_attempt_count": callbacks,
        "callback_zero_confirmed_count": callbacks,
        "callback_sequence_contiguous": True,
        "callback_start_frames_contiguous": True,
        "callback_frame_counts_exact": True,
        "capture_valid_all_true": True,
        "submitted_valid_all_true": True,
        "full_frame_accounting_valid": True,
        "actual_submitted_pcm_hash_eligible": True,
        "application_buffer_zero_submission_complete": True,
        "completed": True,
        "callback_error": None,
        "canonical_invalid_reasons": [],
        "stream_constructor_error": None,
        "stream_start_error": None,
        "watchdog_error": None,
        "stream_stop_error": None,
        "stream_abort_error": None,
        "stream_close_error": None,
        "on_output_closed_error": None,
        "termination_signals": [],
        "termination_signal": None,
        "termination_exit_code": None,
        "stream_start_returned_without_exception": True,
        "stream_stop_attempted": True,
        "stream_stop_returned_without_exception": True,
        "stream_abort_attempted": False,
        "stream_abort_returned_without_exception": False,
        "stream_close_attempted": True,
        "stream_close_returned_without_exception": True,
        "normal_stop_completed": True,
        "faults": [],
        "actual_submitted_pcm": np.array(pcm, dtype=PCM_DTYPE, order="C", copy=True),
        "capture_valid_mask": np.ones(frames, dtype=np.bool_),
        "submitted_valid_mask": np.ones(frames, dtype=np.bool_),
    }


def _resign(payload: dict[str, object]) -> None:
    core = {
        key: value
        for key, value in payload.items()
        if key != "canonical_payload_sha256"
    }
    payload["canonical_payload_sha256"] = payload_sha256(core)


def _valid_receipt() -> dict[str, object]:
    plan, pcm = _make_plan()
    return build_zero_duplex_receipt(
        plan=plan,
        planned_pcm=pcm,
        telemetry=_make_telemetry(plan, pcm),
    )


def test_plan_is_deterministic_read_only_and_bitwise_zero() -> None:
    plan_a, pcm_a = _make_plan()
    plan_b, pcm_b = _make_plan()

    assert plan_a == plan_b
    assert canonical_json_bytes(plan_a) == canonical_json_bytes(plan_b)
    assert pcm_a.dtype == np.dtype("<i4")
    assert pcm_a.shape == (4 * BLOCK_SIZE, 2)
    assert pcm_a.flags.c_contiguous
    assert not pcm_a.flags.writeable
    assert np.all(pcm_a.view(np.uint8) == 0)
    assert np.array_equal(pcm_a, pcm_b)
    assert validate_zero_duplex_plan(plan_a, pcm_a) == plan_a
    assert len(plan_a["planned_pcm_sha256"]) == 64
    assert len(plan_a["zero_payload_sha256"]) == 64
    assert plan_a["planned_pcm_sha256"] != plan_a["zero_payload_sha256"]
    assert plan_a["watchdog_grace_seconds"] == 2.0
    assert plan_a["watchdog_deadline_seconds"] == pytest.approx(
        float(plan_a["nominal_duration_seconds"]) + 2.0
    )


@pytest.mark.parametrize("frame_count", [True, np.int64(256), 0, -256, 257])
def test_plan_builder_rejects_non_exact_or_non_block_frame_count(
    frame_count: object,
) -> None:
    with pytest.raises(ValueError):
        build_zero_duplex_plan(frame_count=frame_count)  # type: ignore[arg-type]


def test_plan_rejects_nonzero_wrong_dtype_and_noncontiguous() -> None:
    plan, pcm = _make_plan()
    nonzero = np.array(pcm, copy=True)
    nonzero[0, 0] = 1
    with pytest.raises(ValueError, match="zero|SHA"):
        validate_zero_duplex_plan(plan, nonzero)
    with pytest.raises(ValueError, match="dtype/shape"):
        validate_zero_duplex_plan(plan, np.zeros(pcm.shape, dtype="<i2"))
    noncontiguous = np.zeros((pcm.shape[0], 4), dtype="<i4")[:, ::2]
    assert noncontiguous.shape == pcm.shape
    assert not noncontiguous.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        validate_zero_duplex_plan(plan, noncontiguous)


def test_plan_rejects_resigned_authority_splice_extra_key_and_bad_sha() -> None:
    plan, pcm = _make_plan()
    spliced = copy.deepcopy(plan)
    spliced["authority_ceiling"] = "SHARED_CLOCK_AUTHORITY_PASS"
    _resign(spliced)
    with pytest.raises(ValueError, match="authority"):
        validate_zero_duplex_plan(spliced, pcm)

    with pytest.raises(ValueError, match="key"):
        validate_zero_duplex_plan({**plan, "shared_clock_pass": True}, pcm)
    malformed = copy.deepcopy(plan)
    malformed["canonical_payload_sha256"] = "A" * 64
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_zero_duplex_plan(malformed, pcm)


def test_valid_live_schema_telemetry_and_receipt_are_transport_only() -> None:
    plan, pcm = _make_plan()
    telemetry = _make_telemetry(plan, pcm)
    telemetry_receipt = validate_zero_duplex_telemetry(
        plan=plan,
        planned_pcm=pcm,
        telemetry=telemetry,
    )
    receipt = build_zero_duplex_receipt(
        plan=plan,
        planned_pcm=pcm,
        telemetry=telemetry,
    )

    assert telemetry_receipt["passed"] is True
    assert telemetry_receipt["all_submitted_buffers_bitwise_zero"] is True
    assert telemetry_receipt["portaudio_status_witness_pass"] is True
    assert telemetry_receipt["output_zero_scope"] == (
        "portaudio_application_callback_buffer_only"
    )
    assert telemetry_receipt["physical_output_zero_authority"] is False
    assert telemetry_receipt["electrical_output_zero_authority"] is False
    assert telemetry_receipt["acoustic_output_zero_authority"] is False
    assert telemetry_receipt["hardware_sample_slip_authority"] is False
    assert telemetry_receipt["portaudio_timestamp_authority"] is False
    assert telemetry_receipt["non_authoritative_observation"] == {
        "callback_deadline_miss_authority": False,
        "callback_deadline_miss_observed": None,
        "fallback_block_authority": False,
        "fallback_block_observed": None,
        "hardware_add_sample_observed": None,
        "hardware_drop_add_authority": False,
        "hardware_drop_sample_observed": None,
    }
    assert "deadline_miss_count" not in telemetry_receipt
    assert "drop_sample_count" not in telemetry_receipt
    assert "add_sample_count" not in telemetry_receipt
    assert "fallback_block_count" not in telemetry_receipt
    assert receipt["status"] == AUTHORITY_CEILING
    assert receipt["authority_ceiling"] == AUTHORITY_CEILING
    assert receipt["authority"]["zero_duplex_transport_smoke_pass"] is True
    assert receipt["authority"]["shared_clock_authority_pass"] is False
    assert receipt["authority"]["sample_identity_pass"] is False
    assert validate_zero_duplex_receipt(receipt) == receipt
    json.dumps(receipt, allow_nan=False, sort_keys=True)


def test_capture_telemetry_adapter_reconstructs_the_sealed_zero_plan() -> None:
    plan, pcm = _make_plan()
    telemetry = _make_telemetry(plan, pcm)
    direct = validate_zero_duplex_telemetry(
        plan=plan,
        planned_pcm=pcm,
        telemetry=telemetry,
    )
    assert capture_telemetry_to_contract(plan, telemetry) == direct


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("xrun_count", 1),
        ("xrun_count", True),
        ("status_present_count", np.int64(0)),
        ("actual_submitted_nonzero_count", 1),
        ("callback_zero_attempt_count", 3),
        ("callback_zero_confirmed_count", 5),
        ("submitted_frames", True),
        ("captured_frames", np.int64(1024)),
        ("expected_callbacks", True),
    ],
)
def test_telemetry_rejects_bad_software_accounting(
    field: str,
    bad_value: object,
) -> None:
    plan, pcm = _make_plan()
    telemetry = _make_telemetry(plan, pcm)
    telemetry[field] = bad_value
    with pytest.raises(ValueError, match=field):
        validate_zero_duplex_telemetry(
            plan=plan,
            planned_pcm=pcm,
            telemetry=telemetry,
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("completed", 1),
        ("normal_stop_completed", False),
        ("stream_close_returned_without_exception", np.bool_(True)),
        ("portaudio_timestamp_authority", True),
        ("hardware_sample_slip_authority", True),
        ("physical_output_zero_authority", True),
        ("portaudio_application_buffer_only", 1),
        ("dither_off", False),
        ("application_buffer_zero_submission_complete", False),
        ("stream_abort_attempted", True),
    ],
)
def test_telemetry_rejects_wrong_exact_bool_or_authority(
    field: str,
    bad_value: object,
) -> None:
    plan, pcm = _make_plan()
    telemetry = _make_telemetry(plan, pcm)
    telemetry[field] = bad_value
    with pytest.raises(ValueError, match=field):
        validate_zero_duplex_telemetry(
            plan=plan,
            planned_pcm=pcm,
            telemetry=telemetry,
        )


@pytest.mark.parametrize(
    ("field", "dtype"),
    [
        ("actual_submitted_pcm", "<i2"),
        ("callback_sequence", "<i4"),
        ("callback_frame_counts", ">i8"),
        ("callback_status_bitmask", "<i4"),
        ("capture_valid_mask", "<u1"),
    ],
)
def test_telemetry_rejects_wrong_array_dtype(field: str, dtype: str) -> None:
    plan, pcm = _make_plan()
    telemetry = _make_telemetry(plan, pcm)
    telemetry[field] = np.asarray(telemetry[field], dtype=dtype)
    with pytest.raises(ValueError, match="dtype/shape"):
        validate_zero_duplex_telemetry(
            plan=plan,
            planned_pcm=pcm,
            telemetry=telemetry,
        )


def test_telemetry_rejects_noncontiguous_or_nonzero_submitted_pcm() -> None:
    plan, pcm = _make_plan()
    telemetry = _make_telemetry(plan, pcm)
    noncontiguous = np.zeros((pcm.shape[0], 4), dtype="<i4")[:, ::2]
    telemetry["actual_submitted_pcm"] = noncontiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        validate_zero_duplex_telemetry(
            plan=plan,
            planned_pcm=pcm,
            telemetry=telemetry,
        )

    telemetry = _make_telemetry(plan, pcm)
    actual = telemetry["actual_submitted_pcm"]
    assert isinstance(actual, np.ndarray)
    actual[BLOCK_SIZE, 1] = -1
    with pytest.raises(ValueError, match="bitwise exact zero"):
        validate_zero_duplex_telemetry(
            plan=plan,
            planned_pcm=pcm,
            telemetry=telemetry,
        )


@pytest.mark.parametrize(
    ("field", "index", "value", "match"),
    [
        ("callback_sequence", 1, 9, "sequence"),
        ("callback_start_frames", 2, 511, "start frame"),
        ("callback_frame_counts", 0, 255, "frame count"),
        ("callback_status_bitmask", 3, 1, "status"),
        ("input_buffer_adc_time", 1, np.nan, "non-finite"),
        ("output_buffer_dac_time", 2, 0.0, "strict-monotonic"),
        ("capture_valid_mask", 2, False, "valid mask"),
        ("submitted_valid_mask", 2, False, "valid mask"),
    ],
)
def test_telemetry_rejects_callback_or_validity_corruption(
    field: str,
    index: int,
    value: object,
    match: str,
) -> None:
    plan, pcm = _make_plan()
    telemetry = _make_telemetry(plan, pcm)
    array = telemetry[field]
    assert isinstance(array, np.ndarray)
    array[index] = value
    with pytest.raises(ValueError, match=match):
        validate_zero_duplex_telemetry(
            plan=plan,
            planned_pcm=pcm,
            telemetry=telemetry,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("pre_open_monotonic_started", np.float64(1.0), "exact finite float"),
        ("pre_open_monotonic_completed", 0.5, "이상"),
        ("capture_monotonic_started", 1.5, "이상"),
        ("capture_monotonic_completed", 2.5, "이상"),
        ("capture_monotonic_elapsed_seconds", 0.5, "재계산"),
        ("watchdog_grace_seconds", 3.0, "sealed plan"),
    ],
)
def test_telemetry_rejects_invalid_watchdog_timing(
    field: str,
    value: object,
    match: str,
) -> None:
    plan, pcm = _make_plan()
    telemetry = _make_telemetry(plan, pcm)
    telemetry[field] = value
    with pytest.raises(ValueError, match=match):
        validate_zero_duplex_telemetry(
            plan=plan,
            planned_pcm=pcm,
            telemetry=telemetry,
        )


def test_telemetry_rejects_outside_watchdog_even_if_self_consistent() -> None:
    plan, pcm = _make_plan()
    telemetry = _make_telemetry(plan, pcm)
    telemetry["capture_monotonic_completed"] = 13.0
    telemetry["capture_monotonic_elapsed_seconds"] = 10.0
    with pytest.raises(ValueError, match="watchdog 범위"):
        validate_zero_duplex_telemetry(
            plan=plan,
            planned_pcm=pcm,
            telemetry=telemetry,
        )


@pytest.mark.parametrize(
    "field",
    [
        "callback_error",
        "stream_constructor_error",
        "stream_start_error",
        "watchdog_error",
        "stream_stop_error",
        "stream_abort_error",
        "stream_close_error",
        "on_output_closed_error",
        "termination_signal",
        "termination_exit_code",
    ],
)
def test_telemetry_rejects_any_transport_error(field: str) -> None:
    plan, pcm = _make_plan()
    telemetry = _make_telemetry(plan, pcm)
    telemetry[field] = "injected failure"
    with pytest.raises(ValueError, match=field):
        validate_zero_duplex_telemetry(
            plan=plan,
            planned_pcm=pcm,
            telemetry=telemetry,
        )


@pytest.mark.parametrize("field", ["canonical_invalid_reasons", "termination_signals", "faults"])
def test_telemetry_rejects_nonempty_failure_lists(field: str) -> None:
    plan, pcm = _make_plan()
    telemetry = _make_telemetry(plan, pcm)
    telemetry[field] = ["injected"]
    with pytest.raises(ValueError, match=field):
        validate_zero_duplex_telemetry(
            plan=plan,
            planned_pcm=pcm,
            telemetry=telemetry,
        )


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "deadline_miss_count",
        "fallback_block_count",
        "drop_sample_count",
        "add_sample_count",
        "shared_clock_authority_pass",
    ],
)
def test_telemetry_rejects_unobservable_counter_or_authority_splice(
    forbidden_field: str,
) -> None:
    plan, pcm = _make_plan()
    telemetry = _make_telemetry(plan, pcm)
    telemetry[forbidden_field] = 0
    with pytest.raises(ValueError, match="extra"):
        validate_zero_duplex_telemetry(
            plan=plan,
            planned_pcm=pcm,
            telemetry=telemetry,
        )


@pytest.mark.parametrize(
    ("authority_field", "bad_value"),
    [
        ("shared_clock_authority_pass", True),
        ("common_clock_topology_pass", True),
        ("hardware_sample_slip_authority", True),
        ("physical_output_route_pass", True),
        ("plant_identification_pass", True),
        ("canonical_training_eligible", True),
        ("zero_duplex_transport_smoke_pass", 1),
    ],
)
def test_resigned_final_receipt_cannot_leak_authority(
    authority_field: str,
    bad_value: object,
) -> None:
    receipt = _valid_receipt()
    authority = receipt["authority"]
    assert isinstance(authority, dict)
    authority[authority_field] = bad_value
    _resign(receipt)
    with pytest.raises(ValueError, match="권위 누수|exact bool"):
        validate_zero_duplex_receipt(receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "SHARED_CLOCK_AUTHORITY_PASS"),
        ("authority_ceiling", "CANONICAL_PASS"),
        ("valid", 1),
    ],
)
def test_resigned_final_receipt_cannot_raise_status_or_bool(
    field: str,
    value: object,
) -> None:
    receipt = _valid_receipt()
    receipt[field] = value
    _resign(receipt)
    with pytest.raises(ValueError, match="transport smoke|exact bool"):
        validate_zero_duplex_receipt(receipt)


def test_resigned_nested_receipt_cannot_leak_or_add_authority() -> None:
    for mutation in ("flip", "extra"):
        receipt = _valid_receipt()
        nested = receipt["telemetry_receipt"]
        assert isinstance(nested, dict)
        if mutation == "flip":
            nested["hardware_sample_slip_authority"] = True
        else:
            nested["shared_clock_authority_pass"] = True
        _resign(nested)
        _resign(receipt)
        with pytest.raises(ValueError, match="authority|key|의미"):
            validate_zero_duplex_receipt(receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("callback_deadline_miss_authority", True),
        ("callback_deadline_miss_observed", 0),
        ("fallback_block_observed", 0),
        ("hardware_drop_add_authority", True),
        ("hardware_drop_sample_observed", 0),
        ("hardware_add_sample_observed", 0),
    ],
)
def test_resigned_receipt_cannot_turn_unobservable_into_zero_observation(
    field: str,
    value: object,
) -> None:
    receipt = _valid_receipt()
    nested = receipt["telemetry_receipt"]
    assert isinstance(nested, dict)
    observation = nested["non_authoritative_observation"]
    assert isinstance(observation, dict)
    observation[field] = value
    _resign(nested)
    _resign(receipt)
    with pytest.raises(ValueError, match="비권위|exact bool"):
        validate_zero_duplex_receipt(receipt)


def test_resigned_nested_receipt_rejects_bad_frame_accounting_or_digest() -> None:
    receipt = _valid_receipt()
    nested = receipt["telemetry_receipt"]
    assert isinstance(nested, dict)
    nested["captured_frames"] = int(nested["captured_frames"]) - 1
    _resign(nested)
    _resign(receipt)
    with pytest.raises(ValueError, match="accounting"):
        validate_zero_duplex_receipt(receipt)

    receipt = _valid_receipt()
    nested = receipt["telemetry_receipt"]
    assert isinstance(nested, dict)
    callback_digests = nested["callback_array_sha256"]
    assert isinstance(callback_digests, dict)
    callback_digests["physical_slip_witness"] = "0" * 64
    _resign(nested)
    _resign(receipt)
    with pytest.raises(ValueError, match="key"):
        validate_zero_duplex_receipt(receipt)


def test_resigned_nested_receipt_rejects_bad_sha_and_zero_digest_splice() -> None:
    receipt = _valid_receipt()
    nested = receipt["telemetry_receipt"]
    assert isinstance(nested, dict)
    nested["actual_submitted_pcm_sha256"] = "G" * 64
    _resign(nested)
    _resign(receipt)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_zero_duplex_receipt(receipt)

    receipt = _valid_receipt()
    nested = receipt["telemetry_receipt"]
    assert isinstance(nested, dict)
    nested["actual_submitted_zero_payload_sha256"] = "0" * 64
    _resign(nested)
    _resign(receipt)
    with pytest.raises(ValueError, match="zero payload SHA"):
        validate_zero_duplex_receipt(receipt)


def test_module_has_no_audio_backend_or_device_import() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "deep_anc"
        / "realtime"
        / "rt5640_zero_smoke.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint(
        {"sounddevice", "alsaaudio", "pyaudio", "soundfile", "portaudio"}
    )
