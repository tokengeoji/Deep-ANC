"""광대역 live 경로를 실제 오디오 없이 mock으로 검증한다."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deep_anc.dsp.control_band_contract import ControlBandContract
from scripts.data import measure_paths_broadband_interleaved as broadband
from scripts.data import measure_paths_interleaved as mpi


REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE = REPO_ROOT / "configs" / "hardware_jetson.yaml"
CONFIRMATIONS = {
    "speaker_output": True,
    "user_present": True,
    "volume_minimum": True,
    "routing_and_geometry": True,
    "same_amplifier_setting": True,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _callback_time_info(frames: int, block: int = 8) -> dict[str, np.ndarray]:
    starts = np.arange(0, frames, block, dtype=np.int64)
    counts = np.full(starts.size, block, dtype=np.int64)
    timebase = np.arange(starts.size, dtype=np.float64) * (block / 48_000.0) + 1.0
    return {
        "callback_start_frames": starts,
        "callback_frame_counts": counts,
        "input_buffer_adc_time": timebase,
        "output_buffer_dac_time": timebase + 0.001,
        "callback_current_time": timebase + 0.002,
    }


def test_raw_publisher_records_exact_pcm_and_required_schema(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    planned = np.arange(32, dtype=np.int16).reshape(16, 2)
    pcm_sha = hashlib.sha256(planned.tobytes(order="C")).hexdigest()
    metadata = {
        "plan": {
            "pcm_sha256": pcm_sha,
            "file_sha256": "1" * 64,
            "payload_sha256": "2" * 64,
        },
        "control_band_contract_sha256": "3" * 64,
        "hardware": {"sha256": "4" * 64},
        "meter": {"raw_sha256": "5" * 64},
        "post_capture_binding": {"valid": True, "error": None},
        "invalid_reasons": [],
    }

    result = broadband.publish_broadband_raw_capture(
        session_dir=session,
        metadata=metadata,
        planned_pcm=planned,
        submitted_pcm=planned.copy(),
        input_raw_int32=np.zeros((16, 2), dtype=np.int32),
        preflight_raw_int32=np.zeros((8, 2), dtype=np.int32),
        callback_time_info=_callback_time_info(16),
    )

    assert result["valid"] is True
    assert result["metadata"]["raw_capture_schema"] == (
        broadband.BROADBAND_RAW_CAPTURE_SCHEMA
    )
    assert result["metadata"]["status"] == "PASS"
    with np.load(result["paths"]["raw"], allow_pickle=False) as archive:
        stored = json.loads(str(archive["metadata_json"].item()))
        np.testing.assert_array_equal(
            archive["submitted_output_pcm_int16"], planned
        )
    assert stored["plan"]["file_sha256"] == "1" * 64
    assert stored["control_band_contract_sha256"] == "3" * 64
    assert stored["hardware"]["sha256"] == "4" * 64
    assert stored["meter"]["raw_sha256"] == "5" * 64


def test_raw_publisher_preserves_but_invalidates_submitted_pcm_mismatch(
    tmp_path: Path,
) -> None:
    session = tmp_path / "invalid"
    session.mkdir()
    planned = np.zeros((16, 2), dtype=np.int16)
    submitted = planned.copy()
    submitted[3, 1] = 1
    metadata = {
        "plan": {
            "pcm_sha256": hashlib.sha256(planned.tobytes()).hexdigest(),
        },
        "invalid_reasons": [],
    }

    result = broadband.publish_broadband_raw_capture(
        session_dir=session,
        metadata=metadata,
        planned_pcm=planned,
        submitted_pcm=submitted,
        input_raw_int32=np.zeros((16, 2), dtype=np.int32),
        preflight_raw_int32=np.zeros((8, 2), dtype=np.int32),
        callback_time_info=_callback_time_info(16),
    )

    assert result["valid"] is False
    assert result["metadata"]["status"] == "INVALID"
    assert "submitted_pcm_not_exact_plan" in result["metadata"]["invalid_reasons"]
    assert result["paths"]["raw"].is_file()


def test_execute_live_requires_all_confirmations_before_any_audio() -> None:
    called = False

    def capture(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("capture must not run")

    try:
        broadband.execute_live_capture(
            hardware_path=HARDWARE,
            saved_plan={},
            planned_pcm=np.zeros((1, 2), dtype=np.int16),
            session_dir=REPO_ROOT / "results" / "never-created",
            meter_raw_path="missing.npz",
            level_evidence_path="missing.json",
            operator_confirmations={**CONFIRMATIONS, "user_present": False},
            sounddevice_module=SimpleNamespace(),
            capture_function=capture,
        )
    except ValueError as exc:
        assert "operator confirmation" in str(exc)
    else:
        raise AssertionError("missing confirmation must fail")
    assert called is False


def test_live_authority_rejects_byte_different_semantically_equal_plan(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(broadband, "REPO_ROOT", tmp_path)
    plan_path = tmp_path / "results" / "authority.json"
    plan_path.parent.mkdir(parents=True)
    planned = np.zeros((8, 2), dtype=np.int16)
    pcm_sha = hashlib.sha256(planned.tobytes(order="C")).hexdigest()
    plan = {
        "schema": broadband.BROADBAND_MEASUREMENT_PLAN_SCHEMA,
        "recipe": {
            "fixed_clock_pilot_pcm_exact_across_panels": True,
            "fixed_clock_pilot_pcm_spectrum_sha256": "a" * 64,
            "global_clock_input_domain": (
                "actual_submitted_int16_period_spectrum_not_intended_float"
            ),
            "submitted_pilot_cross_channel_null": {
                "all_panels_passed": True,
                "maximum_absolute_observed": 0.0,
                "maximum_ratio_observed": 0.0,
            },
        },
        "output": {
            "frames": 8,
            "channels": 2,
            "dtype": "int16",
            "pcm_sha256": pcm_sha,
        },
    }
    plan_path.write_text(json.dumps(plan, separators=(",", ":")), encoding="utf-8")
    authority = {
        "path": "results/authority.json",
        "file_sha256": _sha256(plan_path),
        "payload_sha256": broadband._plan_payload_sha256(plan),
        "pcm_sha256": pcm_sha,
    }
    monkeypatch.setattr(broadband, "BROADBAND_LIVE_AUTHORITY", authority)
    saved = {
        "path": plan_path,
        "file_sha256": authority["file_sha256"],
        "payload_sha256": authority["payload_sha256"],
        "payload": plan,
    }
    assert broadband.validate_live_authority_binding(saved, planned) == authority

    # JSON 의미는 같지만 bytes/파일 SHA만 바꾼다. semantic equality로 통과하면 안 된다.
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    changed = {**saved, "file_sha256": _sha256(plan_path)}
    with pytest.raises(ValueError, match="file_sha256"):
        broadband.validate_live_authority_binding(changed, planned)


@pytest.mark.parametrize(
    "mutation",
    ("evidence_mode", "evidence_sha", "hardware_path", "followup_plan_sha"),
)
def test_meter_followup_binding_rejects_cross_invocation_drift(
    tmp_path: Path, monkeypatch, mutation: str
) -> None:
    monkeypatch.setattr(broadband, "REPO_ROOT", tmp_path)
    hardware = tmp_path / "configs" / "hardware.yaml"
    evidence = tmp_path / "assets" / "level.json"
    session = tmp_path / "results" / "session"
    plan_binding = {
        "path": "results/authority.json",
        "file_sha256": "1" * 64,
        "payload_sha256": "2" * 64,
        "pcm_sha256": "3" * 64,
    }
    hardware_binding = {"path": "configs/hardware.yaml", "sha256": "4" * 64}
    evidence_binding = {"path": "assets/level.json", "sha256": "5" * 64}
    metadata = {
        "calibration_evidence": {
            "mode": "verified_existing",
            **evidence_binding,
        },
        "hardware": dict(hardware_binding),
        "followup_contract": {
            "schema": "broadband_meter_followup_v1",
            "mode": "broadband",
            "plan": dict(plan_binding),
            "raw_session_dir": "results/session",
            "hardware": dict(hardware_binding),
            "level_evidence": dict(evidence_binding),
        },
    }
    if mutation == "evidence_mode":
        metadata["calibration_evidence"]["mode"] = "bootstrap_pending"
    elif mutation == "evidence_sha":
        metadata["calibration_evidence"]["sha256"] = "f" * 64
    elif mutation == "hardware_path":
        metadata["hardware"]["path"] = "configs/other.yaml"
    else:
        metadata["followup_contract"]["plan"]["file_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="meter"):
        broadband.validate_meter_followup_binding(
            meter_metadata=metadata,
            plan_binding=plan_binding,
            hardware_path=hardware,
            hardware_sha256="4" * 64,
            level_evidence_path=evidence,
            level_evidence_sha256="5" * 64,
            raw_session_dir=session,
        )


@pytest.mark.parametrize(
    ("partial", "tamper_after_capture"),
    ((False, False), (True, False), (False, True)),
)
def test_execute_live_mock_uses_preflight_gates_and_publishes_raw(
    tmp_path: Path,
    monkeypatch,
    partial: bool,
    tamper_after_capture: bool,
) -> None:
    planned = np.zeros((32, 2), dtype=np.int16)
    planned[4:28, 0] = 42
    planned[4:28, 1] = -41
    pcm_sha = hashlib.sha256(planned.tobytes(order="C")).hexdigest()
    hardware_sha = _sha256(HARDWARE)
    plan = {
        "schema": broadband.BROADBAND_MEASUREMENT_PLAN_SCHEMA,
        "hardware": {"sha256": hardware_sha},
        "control_band_contract_sha256": (
            ControlBandContract.broadband_point_control().digest()
        ),
        "output": {
            "frames": 32,
            "channels": 2,
            "dtype": "int16",
            "pcm_sha256": pcm_sha,
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    saved_plan = {
        "path": plan_path,
        "file_sha256": _sha256(plan_path),
        "payload_sha256": broadband._plan_payload_sha256(plan),
        "payload": plan,
    }
    evidence_path = tmp_path / "level.json"
    evidence_path.write_text("{}", encoding="utf-8")
    session_dir = tmp_path / f"session-{partial}-{tamper_after_capture}"
    authority = {
        "path": "results/test-live-authority.json",
        "file_sha256": saved_plan["file_sha256"],
        "payload_sha256": saved_plan["payload_sha256"],
        "pcm_sha256": pcm_sha,
    }
    relative_paths = {
        HARDWARE.resolve(): "configs/hardware_jetson.yaml",
        evidence_path.resolve(): "assets/measured/test-level.json",
        session_dir.resolve(): f"results/session-{partial}-{tamper_after_capture}",
    }
    monkeypatch.setattr(
        broadband,
        "validate_live_authority_binding",
        lambda observed_plan, observed_pcm: dict(authority),
    )
    monkeypatch.setattr(
        broadband,
        "_repository_relative_path",
        lambda path: relative_paths[Path(path).resolve()],
    )
    identity = {"physical": "mock"}
    now = dt.datetime.now(dt.timezone.utc)
    evidence_sha = _sha256(evidence_path)
    hardware_binding = {
        "path": relative_paths[HARDWARE.resolve()],
        "sha256": hardware_sha,
    }
    evidence_binding = {
        "path": relative_paths[evidence_path.resolve()],
        "sha256": evidence_sha,
    }
    meter_result = {
        "path": tmp_path / "meter.npz",
        "receipt_path": tmp_path / "meter.receipt.json",
        "sha256": "a" * 64,
        "metadata": {
            "resolved_devices": {"input": 7, "output": 9},
            "calibration_evidence": {
                "mode": "verified_existing",
                **evidence_binding,
            },
            "hardware": hardware_binding,
            "followup_contract": {
                "schema": "broadband_meter_followup_v1",
                "mode": "broadband",
                "plan": authority,
                "raw_session_dir": relative_paths[session_dir.resolve()],
                "hardware": hardware_binding,
                "level_evidence": evidence_binding,
            },
        },
        "meter_ch0_dbfs": -50.0,
        "completed_at_utc": now,
    }
    calls = {
        "pcm_gate": 0,
        "preflight": 0,
        "capture": 0,
        "meter": 0,
        "physical": 0,
    }

    monkeypatch.setattr(
        broadband, "validate_fresh_raw_session_target", lambda path: None
    )

    def physical(config):
        calls["physical"] += 1
        return {"fingerprint": "same"}

    monkeypatch.setattr(broadband, "collect_alsa_physical_fingerprint", physical)
    monkeypatch.setattr(
        broadband,
        "measurement_hardware_identity",
        lambda config, physical_fingerprint: identity,
    )
    monkeypatch.setattr(
        broadband,
        "load_measurement_level_evidence",
        lambda path, repository_root: {
                "hardware_identity": identity,
                "_evidence_path": str(evidence_path),
                "_evidence_sha256": evidence_sha,
        },
    )

    def meter(*args, **kwargs):
        calls["meter"] += 1
        return meter_result

    monkeypatch.setattr(broadband, "validate_bootstrap_meter_raw", meter)
    monkeypatch.setattr(
        broadband,
        "assert_live_pcm_clock_preconditions",
        lambda audio: calls.__setitem__("pcm_gate", calls["pcm_gate"] + 1),
    )
    monkeypatch.setattr(
        broadband, "resolve_alsa_portaudio_device", lambda *args, **kwargs: 9
    )

    def preflight(sd, audio, seconds):
        calls["preflight"] += 1
        assert seconds == 2.0
        return (
            np.zeros((16, 2), dtype=np.int32),
            {
                "device": 7,
                "channels": [{"valid": True}, {"valid": True}],
            },
        )

    monkeypatch.setattr(mpi.cw, "_capture_preflight", preflight)

    @contextmanager
    def lock(*args, **kwargs):
        yield {"path": "results/mock.lock", "pid": 1, "uid": 1}

    def capture(sd, **kwargs):
        calls["capture"] += 1
        kwargs["pre_open_check"]()
        if tamper_after_capture:
            evidence_path.write_text("{}\n", encoding="utf-8")
        submitted = mpi.cw.float32_to_pcm_int16(kwargs["output_float"])
        telemetry = {
            "xrun_count": 0,
            "unexpected_status_count": 0,
            "callback_error": None,
            "stream_abort_error": None,
            "stream_close_error": None,
            "output_stop_confirmed": True,
            "completed": not partial,
            "callback_time_info": _callback_time_info(32),
        }
        recorded = np.zeros((32, 2), dtype=np.int32)
        if partial:
            raise mpi.PartialCaptureError(
                RuntimeError("mock partial"),
                recorded_raw=recorded,
                output_pcm=submitted,
                telemetry=telemetry,
            )
        return recorded, submitted, telemetry

    result = broadband.execute_live_capture(
        hardware_path=HARDWARE,
        saved_plan=saved_plan,
        planned_pcm=planned,
        session_dir=session_dir,
        meter_raw_path=tmp_path / "meter.npz",
        level_evidence_path=evidence_path,
        operator_confirmations=CONFIRMATIONS,
        sounddevice_module=SimpleNamespace(),
        capture_function=capture,
        audio_lock_factory=lock,
    )

    assert result["valid"] is (not partial and not tamper_after_capture)
    expected_binding_calls = 2 if tamper_after_capture else 3
    assert calls == {
        "pcm_gate": expected_binding_calls,
        "preflight": 1,
        "capture": 1,
        "meter": expected_binding_calls,
        "physical": expected_binding_calls,
    }
    assert result["metadata"]["analysis_status"] == "NOT_RUN_RAW_FIRST"
    assert result["metadata"]["plan"]["file_sha256"] == saved_plan[
        "file_sha256"
    ]
    assert result["metadata"]["hardware"]["sha256"] == hardware_sha
    assert result["metadata"]["meter"]["raw_sha256"] == "a" * 64
    assert result["paths"]["raw"].is_file()
    if partial:
        assert "capture_incomplete" in result["metadata"]["invalid_reasons"]
        assert result["partial_capture_error"] is not None
    else:
        assert result["partial_capture_error"] is None
    if tamper_after_capture:
        assert "post_capture_binding_invalid" in result["metadata"][
            "invalid_reasons"
        ]
        assert result["metadata"]["post_capture_binding"]["valid"] is False
        assert "level evidence" in result["metadata"]["post_capture_binding"][
            "error"
        ]


def test_cli_execute_live_without_confirmations_does_not_import_sounddevice(
    monkeypatch,
    capsys,
) -> None:
    imported = False
    original = __import__

    def guarded_import(name, *args, **kwargs):
        nonlocal imported
        if name == "sounddevice":
            imported = True
            raise AssertionError("sounddevice import must not happen")
        return original(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    assert broadband.main(["--execute-live"]) == 2
    assert imported is False
    assert "모두 필요" in capsys.readouterr().err


def test_callback_time_info_rejects_missing_or_one_sample_slip():
    with pytest.raises(ValueError, match="mapping"):
        broadband.validate_callback_time_info(None, expected_frames=32)
    slipped = _callback_time_info(32)
    slipped["callback_start_frames"] = slipped["callback_start_frames"].copy()
    slipped["callback_start_frames"][2] += 1
    with pytest.raises(ValueError, match="sample slip"):
        broadband.validate_callback_time_info(slipped, expected_frames=32)

    fractional = _callback_time_info(32)
    fractional["callback_start_frames"] = fractional[
        "callback_start_frames"
    ].astype(np.float64)
    fractional["callback_start_frames"][2] += 0.5
    with pytest.raises(ValueError, match="exact integer"):
        broadband.validate_callback_time_info(fractional, expected_frames=32)

    after_completion = _callback_time_info(32)
    for name, value in tuple(after_completion.items()):
        after_completion[name] = np.append(value, value[-1] + 1)
    after_completion["callback_start_frames"][-1] = 32
    after_completion["callback_frame_counts"][-1] = 8
    with pytest.raises(ValueError, match="capture 완료 뒤"):
        broadband.validate_callback_time_info(after_completion, expected_frames=32)


def test_cli_dry_run_never_imports_sounddevice(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    imported = False
    original = __import__

    def guarded_import(name, *args, **kwargs):
        nonlocal imported
        if name == "sounddevice":
            imported = True
            raise AssertionError("dry-run must not import sounddevice")
        return original(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    monkeypatch.setattr(
        broadband,
        "validate_dry_run_environment",
        lambda **kwargs: {
            "input_alsa": [0, 1],
            "output_alsa": [2, 0],
        },
    )
    output = tmp_path / "plan.json"
    assert broadband.main(
        [
            "--dry-run",
            "--output",
            str(output),
            "--raw-session-dir",
            "results/_pytest_broadband_dry_never",
        ]
    ) == 0
    assert imported is False
    assert output.is_file()
    assert "sounddevice import/open" in capsys.readouterr().err
