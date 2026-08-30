from __future__ import annotations

import copy
import datetime as dt
import hashlib
from pathlib import Path

import numpy as np
import pytest

from deep_anc import audio_duplex_v6 as audio
from deep_anc.audio_duplex_v5 import DUPLEX_TELEMETRY_SCHEMA as V5_TELEMETRY_SCHEMA
from deep_anc.audio_io import analyze_int32_input_probe
from deep_anc.dsp import fullband_live_authority_v6 as authority
from deep_anc.dsp import fullband_live_raw_v5 as common
from deep_anc.dsp import fullband_live_raw_v6 as raw
from deep_anc.dsp.fullband_causal_v6 import build_plan_v6


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fixture() -> tuple[dict, dict, dict, dict, dict]:
    _, planned = build_plan_v6()
    preflight_index = np.arange(192, dtype=np.float64)
    preflight = np.column_stack(
        (
            np.rint(2_000_000 * np.sin(2 * np.pi * preflight_index / 31)),
            np.rint(1_500_000 * np.cos(2 * np.pi * preflight_index / 29)),
        )
    ).astype("<i4")
    hardware_identity = _sha("hardware-identity")
    channels = analyze_int32_input_probe(preflight)["channels"]
    preflight_report = {
        "schema": raw.PREFLIGHT_REPORT_SCHEMA,
        "passed": True,
        "identity_sha256": common._preflight_identity_sha256(
            preflight_raw=preflight,
            hardware_identity_sha256=hardware_identity,
            resolved_input_device=3,
            sample_rate_hz=48_000,
            frames=len(preflight),
        ),
        "resolved_input_device": 3,
        "sample_rate_hz": 48_000,
        "frames": len(preflight),
        "channels": channels,
    }
    completed = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    started = completed - dt.timedelta(seconds=len(planned) / 48_000.0)
    session = {
        "schema": raw.SESSION_SCHEMA,
        "capture_id": "0123456789abcdef0123456789abcdef",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "publisher_prepared_at_utc": (completed + dt.timedelta(seconds=0.1)).isoformat(),
        "audio_lock_identity_sha256": _sha("audio-lock"),
        "repository_commit": "a" * 40,
        "repository_branch": "work/test-v6",
        "repository_dirty": False,
        "adapter_path": "scripts/data/measure_paths_fullband_causal_v6.py",
        "adapter_file_sha256": _sha("adapter-v6"),
    }
    evidence_file = _sha("evidence-file")
    bindings = {
        "signal_plan": {
            "schema": authority.PLAN_ENVELOPE_SCHEMA,
            "path": authority.SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
            "file_sha256": authority.EXPECTED_PLAN_ENVELOPE_FILE_SHA256,
            "payload_sha256": authority.EXPECTED_PLAN_PAYLOAD_SHA256,
            "pcm_sha256": authority.EXPECTED_PCM_SHA256,
            "raw_session_relative_path": authority.SEALED_RAW_RELATIVE_PATH,
        },
        "live_capture_authority": {
            "schema": authority.AUTHORITY_SCHEMA,
            "path": authority.SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
            "file_sha256": authority.EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256,
            "payload_sha256": authority.EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256,
            "signal_plan_file_sha256": authority.EXPECTED_PLAN_ENVELOPE_FILE_SHA256,
            "signal_plan_payload_sha256": authority.EXPECTED_PLAN_PAYLOAD_SHA256,
            "signal_pcm_sha256": authority.EXPECTED_PCM_SHA256,
            "hardware_file_sha256": authority.EXPECTED_HARDWARE_FILE_SHA256,
            "raw_session_relative_path": authority.SEALED_RAW_RELATIVE_PATH,
        },
        "meter": {
            "schema": "measurement_level_meter_raw_v1",
            "path": "results/fullband_causal_v6/level_meter/session/meter_raw.npz",
            "receipt_path": "results/fullband_causal_v6/level_meter/session/meter_raw.npz.receipt.json",
            "raw_sha256": _sha("meter-raw"),
            "receipt_sha256": _sha("meter-receipt"),
            "completed_at_utc": (started - dt.timedelta(seconds=60)).isoformat(),
            "identity_sha256": _sha("meter-identity"),
            "followup_contract_sha256": _sha("followup-v6"),
            "live_authority_file_sha256": authority.EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256,
            "level_evidence_file_sha256": evidence_file,
            "hardware_file_sha256": authority.EXPECTED_HARDWARE_FILE_SHA256,
        },
        "level_evidence": {
            "schema": "measurement_level_evidence_v2_bootstrap_pair",
            "path": "assets/measured/measurement_level_evidence.json",
            "file_sha256": evidence_file,
            "identity_sha256": _sha("evidence-identity"),
            "scope": "tracked_historical_attestation_for_fresh_v5_meter_only",
            "preserved_raw_revalidated": False,
        },
        "hardware": {
            "schema": "jetson_measurement_hardware_v1",
            "path": authority.SEALED_HARDWARE_RELATIVE_PATH,
            "file_sha256": authority.EXPECTED_HARDWARE_FILE_SHA256,
            "identity_sha256": hardware_identity,
            "physical_fingerprint_sha256": _sha("physical-fingerprint"),
            "resolved_devices": {"input": 3, "output": 7},
        },
        "preflight": {
            "schema": raw.PREFLIGHT_REPORT_SCHEMA,
            "raw_sha256": common._array_sha256(preflight),
            "report_sha256": hashlib.sha256(
                common._canonical_json_bytes(preflight_report)
            ).hexdigest(),
            "identity_sha256": preflight_report["identity_sha256"],
            "passed": True,
        },
    }
    post = {
        "schema": raw.POST_CAPTURE_BINDING_SCHEMA,
        "valid": True,
        "error": None,
        "refreshed_signal_plan_file_sha256": bindings["signal_plan"]["file_sha256"],
        "refreshed_signal_plan_payload_sha256": bindings["signal_plan"]["payload_sha256"],
        "refreshed_signal_pcm_sha256": bindings["signal_plan"]["pcm_sha256"],
        "refreshed_authority_file_sha256": bindings["live_capture_authority"]["file_sha256"],
        "refreshed_authority_payload_sha256": bindings["live_capture_authority"]["payload_sha256"],
        "refreshed_meter_raw_sha256": bindings["meter"]["raw_sha256"],
        "refreshed_meter_receipt_sha256": bindings["meter"]["receipt_sha256"],
        "refreshed_level_evidence_file_sha256": bindings["level_evidence"]["file_sha256"],
        "refreshed_hardware_file_sha256": bindings["hardware"]["file_sha256"],
        "refreshed_hardware_identity_sha256": bindings["hardware"]["identity_sha256"],
        "refreshed_physical_fingerprint_sha256": bindings["hardware"]["physical_fingerprint_sha256"],
        "refreshed_audio_lock_identity_sha256": session["audio_lock_identity_sha256"],
        "resolved_devices": dict(bindings["hardware"]["resolved_devices"]),
        "raw_target_fresh": True,
    }
    count = len(planned) // 256
    sequence = np.arange(count, dtype="<i8")
    times = np.arange(1, count + 1, dtype="<f8") * 0.01
    arrays = {
        "planned_submitted_pcm": planned,
        "actual_submitted_pcm": planned.copy(),
        "captured_pcm": np.zeros_like(planned, dtype="<i4"),
        "submitted_valid_mask": np.ones(len(planned), dtype=np.bool_),
        "capture_valid_mask": np.ones(len(planned), dtype=np.bool_),
        "preflight_raw_int32": preflight,
        "callback_sequence": sequence,
        "callback_start_frames": sequence * 256,
        "callback_frame_counts": np.full(count, 256, dtype="<i8"),
        "input_buffer_adc_time": times,
        "output_buffer_dac_time": times + 0.001,
        "callback_current_time": times + 0.002,
        "callback_status_bitmask": np.zeros(count, dtype="<u4"),
    }
    telemetry = {
        "schema": audio.DUPLEX_TELEMETRY_SCHEMA,
        "callback_frame_semantics": "software_accounting_only_not_hardware_slip_witness",
        "portaudio_xrun_status_witness": True,
        "hardware_sample_slip_authority": False,
        "watchdog_coverage": "host_wait_until_planned_frames_plus_grace_not_hardware_deadline_witness",
        "sample_rate_hz": 48_000,
        "block_size": 256,
        "latency": "low",
        "channels": [2, 2],
        "input_dtype": "<i4",
        "output_dtype": "<i2",
        "resolved_input_device": 3,
        "resolved_output_device": 7,
        "pre_open_monotonic_started": 99.0,
        "pre_open_monotonic_completed": 100.0,
        "pre_open_monotonic_elapsed_seconds": 1.0,
        "capture_monotonic_started": 100.0,
        "capture_monotonic_completed": 100.0 + len(planned) / 48_000.0,
        "capture_monotonic_elapsed_seconds": len(planned) / 48_000.0,
        "watchdog_grace_seconds": 2.0,
        "xrun_count": 0,
        "status_present_count": 0,
        "captured_frames": len(planned),
        "submitted_frames": len(planned),
        "completed": True,
        "callback_error": None,
        "canonical_invalid_reasons": [],
        "stream_stop_error": None,
        "stream_abort_error": None,
        "stream_close_error": None,
        "termination_signal": None,
        "normal_stop_completed": True,
        "output_stop_confirmed": True,
    }
    return arrays, telemetry, session, bindings, {"report": preflight_report, "post": post}


CONFIRMATIONS = {
    "speaker_output": True,
    "user_present": True,
    "volume_minimum": True,
    "routing_and_geometry": True,
    "same_amplifier_setting": True,
}


def test_v6_profile_builds_only_v6_raw_and_duplex_schemas() -> None:
    arrays, telemetry, session, bindings, extra = _fixture()
    metadata = common._validate_and_build_metadata(
        arrays=arrays,
        telemetry=telemetry,
        capture_exception=None,
        session=session,
        bindings=bindings,
        preflight_report=extra["report"],
        operator_confirmations=CONFIRMATIONS,
        post_capture_binding=extra["post"],
    )
    assert metadata["schema"] == raw.LIVE_RAW_SCHEMA
    assert metadata["duplex_telemetry_schema"] == audio.DUPLEX_TELEMETRY_SCHEMA
    assert metadata["session"]["schema"] == raw.SESSION_SCHEMA
    assert metadata["status"] == "CAPTURE_PASS"


def test_v6_pre_open_timing_survives_canonical_publish_and_rebuild(
    tmp_path: Path,
) -> None:
    arrays, telemetry, session, bindings, extra = _fixture()
    capture_telemetry = {
        **telemetry,
        "actual_submitted_pcm": arrays["actual_submitted_pcm"],
        "capture_valid_mask": arrays["capture_valid_mask"],
        "submitted_valid_mask": arrays["submitted_valid_mask"],
        **{name: arrays[name] for name in common.TELEMETRY_ARRAY_FIELDS},
    }
    session_input = dict(session)
    session_input.pop("publisher_prepared_at_utc")
    target = tmp_path / authority.SEALED_RAW_RELATIVE_PATH
    published = raw.publish_live_raw_v6(
        target,
        repository_root=tmp_path,
        planned_submitted_pcm=arrays["planned_submitted_pcm"],
        capture=(arrays["captured_pcm"], capture_telemetry),
        preflight_raw_int32=arrays["preflight_raw_int32"],
        preflight_report=extra["report"],
        session=session_input,
        bindings=bindings,
        operator_confirmations=CONFIRMATIONS,
        post_capture_binding=extra["post"],
    )
    loaded = raw.load_live_raw_v6(
        target,
        repository_root=tmp_path,
        expected_bindings=bindings,
        expected_raw_file_sha256=published["raw_file_sha256"],
    )
    scalar = loaded["metadata"]["duplex_telemetry_scalars"]
    assert scalar["pre_open_monotonic_elapsed_seconds"] == 1.0
    assert scalar["capture_monotonic_started"] == scalar[
        "pre_open_monotonic_completed"
    ]


def test_v5_telemetry_or_authority_cannot_splice_into_v6_raw() -> None:
    arrays, telemetry, session, bindings, extra = _fixture()
    telemetry["schema"] = V5_TELEMETRY_SCHEMA
    with pytest.raises(ValueError, match="duplex telemetry schema"):
        common._validate_and_build_metadata(
            arrays=arrays,
            telemetry=telemetry,
            capture_exception=None,
            session=session,
            bindings=bindings,
            preflight_report=extra["report"],
            operator_confirmations=CONFIRMATIONS,
            post_capture_binding=extra["post"],
        )

    arrays, telemetry, session, bindings, extra = _fixture()
    changed = copy.deepcopy(bindings)
    changed["signal_plan"]["schema"] = "fullband_causal_signal_plan_envelope_v5"
    with pytest.raises(ValueError):
        common._validate_and_build_metadata(
            arrays=arrays,
            telemetry=telemetry,
            capture_exception=None,
            session=session,
            bindings=changed,
            preflight_report=extra["report"],
            operator_confirmations=CONFIRMATIONS,
            post_capture_binding=extra["post"],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pre_open_monotonic_started", float("nan")),
        ("pre_open_monotonic_completed", 98.0),
        ("pre_open_monotonic_elapsed_seconds", 2.0),
        ("capture_monotonic_started", 99.5),
    ],
)
def test_v6_pre_open_timing_is_exact_and_ordered(field: str, value: float) -> None:
    arrays, telemetry, session, bindings, extra = _fixture()
    telemetry[field] = value
    if field == "capture_monotonic_started":
        telemetry["capture_monotonic_elapsed_seconds"] = (
            telemetry["capture_monotonic_completed"] - value
        )
    with pytest.raises(ValueError, match="pre-open|finite"):
        common._validate_and_build_metadata(
            arrays=arrays,
            telemetry=telemetry,
            capture_exception=None,
            session=session,
            bindings=bindings,
            preflight_report=extra["report"],
            operator_confirmations=CONFIRMATIONS,
            post_capture_binding=extra["post"],
        )


def test_public_v6_wrapper_rejects_v5_bindings_before_common_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal called
        called = True
        raise AssertionError("v5-origin common dispatch가 호출됐습니다")

    monkeypatch.setattr(raw, "publish_live_raw_v5", forbidden)
    monkeypatch.setattr(raw, "load_live_raw_v5", forbidden)
    v5_bindings = {
        "signal_plan": {"schema": "fullband_causal_signal_plan_envelope_v5"}
    }
    with pytest.raises(ValueError, match="exact v6"):
        raw.publish_live_raw_v6("ignored.npz", bindings=v5_bindings)
    with pytest.raises(ValueError, match="exact v6"):
        raw.load_live_raw_v6(
            "ignored.npz",
            repository_root=".",
            expected_bindings=v5_bindings,
            expected_raw_file_sha256="0" * 64,
        )
    assert called is False
