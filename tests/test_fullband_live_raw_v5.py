from __future__ import annotations

import copy
import datetime as dt
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from deep_anc import audio_duplex_v5 as audio
from deep_anc.audio_io import analyze_int32_input_probe
from deep_anc.dsp import fullband_live_authority_v5 as authority
from deep_anc.dsp.fullband_causal_v5 import build_plan_v5
from deep_anc.dsp import fullband_live_raw_v5 as raw


_CANONICAL_PLAN, _CANONICAL_PCM = build_plan_v5()
FRAMES = len(_CANONICAL_PCM)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _planned() -> np.ndarray:
    return _CANONICAL_PCM.copy()


def _preflight() -> np.ndarray:
    index = np.arange(192, dtype=np.float64)
    return np.column_stack(
        (
            np.rint(2_000_000 * np.sin(2 * np.pi * index / 31)),
            np.rint(1_500_000 * np.cos(2 * np.pi * index / 29)),
        )
    ).astype("<i4")


def _preflight_report(
    preflight: np.ndarray,
    *,
    hardware_identity: str | None = None,
) -> dict:
    hardware_identity = hardware_identity or _sha("hardware-identity")
    analyzed = analyze_int32_input_probe(preflight)
    return {
        "schema": raw.PREFLIGHT_REPORT_SCHEMA,
        "passed": all(item["valid"] is True for item in analyzed["channels"]),
        "identity_sha256": raw._preflight_identity_sha256(
            preflight_raw=preflight,
            hardware_identity_sha256=hardware_identity,
            resolved_input_device=3,
            sample_rate_hz=48_000,
            frames=len(preflight),
        ),
        "resolved_input_device": 3,
        "sample_rate_hz": 48_000,
        "frames": len(preflight),
        "channels": analyzed["channels"],
    }


def _session() -> dict:
    completed = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    started = completed - dt.timedelta(seconds=FRAMES / 48_000.0)
    return {
        "schema": raw.SESSION_SCHEMA,
        "capture_id": "0123456789abcdef0123456789abcdef",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "audio_lock_identity_sha256": _sha("audio-lock"),
        "repository_commit": "a" * 40,
        "repository_branch": "work/test",
        "repository_dirty": False,
        "adapter_path": "scripts/data/measure_paths_fullband_causal_v5.py",
        "adapter_file_sha256": _sha("adapter"),
    }


def _bindings(
    planned: np.ndarray, preflight: np.ndarray, report: dict | None = None
) -> dict:
    report = report or _preflight_report(preflight)
    plan_file = authority.EXPECTED_PLAN_ENVELOPE_FILE_SHA256
    plan_payload = authority.EXPECTED_PLAN_PAYLOAD_SHA256
    pcm = raw._array_sha256(planned)
    assert pcm == authority.EXPECTED_PCM_SHA256
    hardware_file = authority.EXPECTED_HARDWARE_FILE_SHA256
    authority_file = authority.EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256
    evidence_file = _sha("evidence-file")
    path = authority.SEALED_RAW_RELATIVE_PATH
    hardware_identity = _sha("hardware-identity")
    return {
        "signal_plan": {
            "schema": "fullband_causal_signal_plan_envelope_v5",
            "path": authority.SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
            "file_sha256": plan_file,
            "payload_sha256": plan_payload,
            "pcm_sha256": pcm,
            "raw_session_relative_path": path,
        },
        "live_capture_authority": {
            "schema": "fullband_causal_v5_live_capture_authority_v1",
            "path": authority.SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
            "file_sha256": authority_file,
            "payload_sha256": authority.EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256,
            "signal_plan_file_sha256": plan_file,
            "signal_plan_payload_sha256": plan_payload,
            "signal_pcm_sha256": pcm,
            "hardware_file_sha256": hardware_file,
            "raw_session_relative_path": path,
        },
        "meter": {
            "schema": "measurement_level_meter_raw_v1",
            "path": "results/meter/meter_raw.npz",
            "receipt_path": "results/meter/meter_raw.receipt.json",
            "raw_sha256": _sha("meter-raw"),
            "receipt_sha256": _sha("meter-receipt"),
            "completed_at_utc": (
                dt.datetime.fromisoformat(_session()["started_at_utc"])
                - dt.timedelta(seconds=60)
            ).isoformat(),
            "identity_sha256": _sha("meter-identity"),
            "followup_contract_sha256": _sha("followup"),
            "live_authority_file_sha256": authority_file,
            "level_evidence_file_sha256": evidence_file,
            "hardware_file_sha256": hardware_file,
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
            "path": "configs/hardware_jetson.yaml",
            "file_sha256": hardware_file,
            "identity_sha256": hardware_identity,
            "physical_fingerprint_sha256": _sha("physical-fingerprint"),
            "resolved_devices": {"input": 3, "output": 7},
        },
        "preflight": {
            "schema": raw.PREFLIGHT_REPORT_SCHEMA,
            "raw_sha256": raw._array_sha256(preflight),
            "report_sha256": hashlib.sha256(
                raw._canonical_json_bytes(report)
            ).hexdigest(),
            "identity_sha256": report["identity_sha256"],
            "passed": report["passed"],
        },
    }


CONFIRMATIONS = {
    "speaker_output": True,
    "user_present": True,
    "volume_minimum": True,
    "routing_and_geometry": True,
    "same_amplifier_setting": True,
}
def _post_binding(bindings: dict, session: dict, *, valid: bool = True) -> dict:
    return {
        "schema": raw.POST_CAPTURE_BINDING_SCHEMA,
        "valid": valid,
        "error": None if valid else "refreshed binding invalid",
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


def _telemetry(prefix: int, *, status: int = 0, completed: bool = True) -> dict:
    count = prefix // audio.BLOCK_SIZE
    sequence = np.arange(count, dtype="<i8")
    starts = sequence * audio.BLOCK_SIZE
    times = np.arange(1, count + 1, dtype="<f8") * 0.01
    statuses = np.full(count, status, dtype="<u4")
    status_present = int(np.count_nonzero(statuses & audio.STATUS_PRESENT))
    xrun = int(np.count_nonzero(statuses & audio.STATUS_XRUN_MASK))
    elapsed = FRAMES / 48_000.0
    return {
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
        "capture_monotonic_started": 100.0,
        "capture_monotonic_completed": 100.0 + elapsed,
        "capture_monotonic_elapsed_seconds": elapsed,
        "watchdog_grace_seconds": 2.0,
        "callback_sequence": sequence,
        "callback_start_frames": starts,
        "callback_frame_counts": np.full(count, 256, dtype="<i8"),
        "input_buffer_adc_time": times,
        "output_buffer_dac_time": times + 0.001,
        "callback_current_time": times + 0.002,
        "callback_status_bitmask": statuses,
        "xrun_count": xrun,
        "status_present_count": status_present,
        "captured_frames": prefix,
        "submitted_frames": prefix,
        "completed": completed,
        "callback_error": None,
        "canonical_invalid_reasons": [],
        "stream_stop_error": None,
        "stream_abort_error": None,
        "stream_close_error": None,
        "termination_signal": None,
        "normal_stop_completed": completed and status == 0,
        "output_stop_confirmed": True,
    }


def _success_capture(planned: np.ndarray) -> tuple[np.ndarray, dict]:
    captured = (planned.astype(np.int32) * 10_000).astype("<i4")
    telemetry = _telemetry(FRAMES)
    telemetry.update(
        {
            "actual_submitted_pcm": planned.copy(),
            "capture_valid_mask": np.ones(FRAMES, dtype=np.bool_),
            "submitted_valid_mask": np.ones(FRAMES, dtype=np.bool_),
        }
    )
    return captured, telemetry


def _failure(
    planned: np.ndarray,
    *,
    prefix: int = 256,
    status: int = 0,
    nonzero_tail: bool = False,
    noncontiguous: bool = False,
    silence_unconfirmed: bool = False,
) -> audio.DuplexCaptureFailure:
    actual = np.zeros_like(planned)
    actual[:prefix] = planned[:prefix]
    if nonzero_tail:
        actual[prefix:] = planned[prefix:]
    captured = np.zeros(planned.shape, dtype="<i4")
    captured[:prefix] = planned[:prefix].astype(np.int32) * 10_000
    out_mask = np.zeros(FRAMES, dtype=np.bool_)
    cap_mask = np.zeros(FRAMES, dtype=np.bool_)
    out_mask[:prefix] = True
    cap_mask[:prefix] = True
    if noncontiguous:
        out_mask[prefix + 1] = True
        cap_mask[prefix + 1] = True
    telemetry = _telemetry(prefix, status=status, completed=prefix == FRAMES)
    telemetry["normal_stop_completed"] = False
    if status:
        telemetry["canonical_invalid_reasons"] = ["portaudio_xrun_status"]
    else:
        telemetry["callback_error"] = "ValueError: callback input shape"
    if silence_unconfirmed:
        telemetry["canonical_invalid_reasons"] = [
            "output_silence_not_confirmed_on_callback_failure"
        ]
        telemetry["output_stop_confirmed"] = False
    return audio.DuplexCaptureFailure(
        message="RuntimeError: duplex failed",
        captured_pcm=captured,
        submitted_pcm=actual,
        capture_valid_mask=cap_mask,
        submitted_valid_mask=out_mask,
        telemetry=telemetry,
    )


def _publish(tmp_path: Path, capture, *, bindings: dict | None = None, post=None):
    planned = _planned()
    preflight = _preflight()
    report = _preflight_report(preflight)
    bound = bindings or _bindings(planned, preflight, report)
    session = _session()
    post = post or _post_binding(bound, session)
    target = tmp_path / bound["signal_plan"]["raw_session_relative_path"]
    published = raw.publish_live_raw_v5(
        target,
        repository_root=tmp_path,
        planned_submitted_pcm=planned,
        capture=capture,
        preflight_raw_int32=preflight,
        preflight_report=report,
        session=session,
        bindings=bound,
        operator_confirmations=CONFIRMATIONS,
        post_capture_binding=post,
    )
    return planned, preflight, bound, published


def test_success_exact_schema_roundtrip_but_external_admission_unbound(tmp_path: Path) -> None:
    planned = _planned()
    _, _, bindings, published = _publish(tmp_path, _success_capture(planned))
    assert published["metadata"]["status"] == "CAPTURE_PASS"
    assert published["metadata"]["analysis_admission_eligible"] is False
    assert published["metadata"]["external_post_capture_receipt_bound"] is False
    assert published["metadata"]["post_capture_binding_scope"] == (
        "primitive_self_attestation_not_external_receipt"
    )
    assert published["metadata"]["canonical_training_eligible"] is False
    assert published["metadata"]["hardware_sample_slip_authority"] is False
    with np.load(published["path"], allow_pickle=False) as archive:
        assert set(archive.files) == set(raw.RAW_ARRAY_FIELDS) | {raw.METADATA_MEMBER}
        assert len(raw.RAW_ARRAY_FIELDS) == 13
    loaded = raw.load_live_raw_v5(
        published["path"],
        repository_root=tmp_path,
        expected_bindings=bindings,
        expected_raw_file_sha256=published["raw_file_sha256"],
    )
    assert loaded["metadata"] == published["metadata"]
    assert loaded["metadata"]["session"]["capture_id"] == _session()["capture_id"]
    assert "publisher_prepared_at_utc" in loaded["metadata"]["session"]
    assert loaded["metadata"]["container_writer_contract"]["schema"] == raw.WRITER_CONTRACT_SCHEMA
    assert loaded["metadata"]["container_writer_contract"]["byte_reproducibility_scope"] == (
        "same_python_numpy_runtime_only"
    )
    assert loaded["metadata"]["preflight_report"] == _preflight_report(_preflight())
    assert np.array_equal(loaded["arrays"]["actual_submitted_pcm"], planned)
    with pytest.raises(ValueError, match="analysis admission"):
        raw.admit_live_raw_v5_for_analysis(
            published["path"],
            repository_root=tmp_path,
            expected_bindings=bindings,
            expected_raw_file_sha256=published["raw_file_sha256"],
        )


def test_partial_callback_failure_is_immutable_invalid_with_zero_tail(tmp_path: Path) -> None:
    planned = _planned()
    failure = _failure(planned, silence_unconfirmed=True)
    _, _, bindings, published = _publish(tmp_path, failure)
    metadata = published["metadata"]
    assert metadata["status"] == "INVALID"
    assert metadata["analysis_admission_eligible"] is False
    assert "capture_incomplete" in metadata["invalid_reasons"]
    assert "output_silence_not_confirmed_on_callback_failure" in metadata["invalid_reasons"]
    loaded = raw.load_live_raw_v5(
        published["path"], repository_root=tmp_path,
        expected_bindings=bindings,
        expected_raw_file_sha256=published["raw_file_sha256"],
    )
    assert np.all(loaded["arrays"]["actual_submitted_pcm"][256:] == 0)
    assert np.all(~loaded["arrays"]["submitted_valid_mask"][256:])
    with pytest.raises(ValueError, match="analysis admission"):
        raw.admit_live_raw_v5_for_analysis(
            published["path"], repository_root=tmp_path,
            expected_bindings=bindings,
            expected_raw_file_sha256=published["raw_file_sha256"],
        )


def test_xrun_and_post_binding_failure_are_invalid(tmp_path: Path) -> None:
    planned = _planned()
    status = audio.STATUS_PRESENT | audio.STATUS_INPUT_OVERFLOW
    failure = _failure(planned, prefix=FRAMES, status=status)
    _, _, _, published = _publish(
        tmp_path, failure, post=_post_binding(
            _bindings(_planned(), _preflight()), _session(), valid=False
        )
    )
    assert published["metadata"]["status"] == "INVALID"
    assert "callback_status_nonzero" in published["metadata"]["invalid_reasons"]
    assert "post_capture_binding_invalid" in published["metadata"]["invalid_reasons"]


def test_rejects_noncontiguous_masks_and_planned_tail_impersonation(tmp_path: Path) -> None:
    planned = _planned()
    preflight = _preflight()
    bindings = _bindings(planned, preflight)
    target = tmp_path / bindings["signal_plan"]["raw_session_relative_path"]
    kwargs = dict(
        target=target,
        repository_root=tmp_path,
        planned_submitted_pcm=planned,
        preflight_raw_int32=preflight,
        preflight_report=_preflight_report(preflight),
        session=_session(),
        bindings=bindings,
        operator_confirmations=CONFIRMATIONS,
        post_capture_binding=_post_binding(bindings, _session()),
    )
    with pytest.raises(ValueError, match="contiguous prefix"):
        raw.publish_live_raw_v5(capture=_failure(planned, noncontiguous=True), **kwargs)
    with pytest.raises(ValueError, match="invalid tail"):
        raw.publish_live_raw_v5(capture=_failure(planned, nonzero_tail=True), **kwargs)
    assert not target.exists()


def test_existing_target_and_symlink_parent_are_rejected(tmp_path: Path) -> None:
    planned = _planned()
    _, _, bindings, published = _publish(tmp_path, _success_capture(planned))
    with pytest.raises(FileExistsError):
        raw.publish_live_raw_v5(
            published["path"], repository_root=tmp_path,
            planned_submitted_pcm=planned, capture=_success_capture(planned),
            preflight_raw_int32=_preflight(),
            preflight_report=_preflight_report(_preflight()), session=_session(),
            bindings=bindings,
            operator_confirmations=CONFIRMATIONS,
            post_capture_binding=_post_binding(bindings, _session()),
        )


def test_publish_parent_rename_symlink_swap_cannot_escape_dirfd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planned = _planned()
    preflight = _preflight()
    report = _preflight_report(preflight)
    bindings = _bindings(planned, preflight, report)
    session = _session()
    external = tmp_path / "external"
    external.mkdir()
    real_link = raw.os.link
    swapped = False

    def attacking_link(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            results = tmp_path / "results"
            results.rename(tmp_path / "results-detached")
            results.symlink_to(external, target_is_directory=True)
            swapped = True
        return real_link(*args, **kwargs)

    monkeypatch.setattr(raw.os, "link", attacking_link)
    with pytest.raises(RuntimeError, match="inode/lexical chain"):
        raw.publish_live_raw_v5(
            authority.SEALED_RAW_RELATIVE_PATH,
            repository_root=tmp_path,
            planned_submitted_pcm=planned,
            capture=_success_capture(planned),
            preflight_raw_int32=preflight,
            preflight_report=report,
            session=session,
            bindings=bindings,
            operator_confirmations=CONFIRMATIONS,
            post_capture_binding=_post_binding(bindings, session),
        )
    assert swapped is True
    assert not (external / "fullband_causal_v5" / "raw_capture.npz").exists()
    assert (
        tmp_path
        / "results-detached/fullband_causal_v5/raw_capture.npz"
    ).is_file()


def test_loader_parent_rename_symlink_swap_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planned = _planned()
    _, _, bindings, published = _publish(tmp_path, _success_capture(planned))
    external = tmp_path / "external"
    external.mkdir()
    real_read = raw._read_regular_file_at

    def attacking_read(parent_fd: int, filename: str) -> bytes:
        results = tmp_path / "results"
        results.rename(tmp_path / "results-detached")
        results.symlink_to(external, target_is_directory=True)
        return real_read(parent_fd, filename)

    monkeypatch.setattr(raw, "_read_regular_file_at", attacking_read)
    with pytest.raises(RuntimeError, match="inode/lexical chain"):
        raw.load_live_raw_v5(
            published["path"],
            repository_root=tmp_path,
            expected_bindings=bindings,
            expected_raw_file_sha256=published["raw_file_sha256"],
        )
    assert not (external / "fullband_causal_v5" / "raw_capture.npz").exists()


def test_staging_unlink_failure_after_link_is_warning_not_false_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planned = _planned()
    preflight = _preflight()
    report = _preflight_report(preflight)
    bindings = _bindings(planned, preflight, report)
    session = _session()
    real_unlink = raw.os.unlink
    failed = False

    def failing_unlink(path, *args, **kwargs):
        nonlocal failed
        if not failed and ".staging-" in str(path):
            failed = True
            raise OSError("injected staging unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(raw.os, "unlink", failing_unlink)
    published = raw.publish_live_raw_v5(
        authority.SEALED_RAW_RELATIVE_PATH,
        repository_root=tmp_path,
        planned_submitted_pcm=planned,
        capture=_success_capture(planned),
        preflight_raw_int32=preflight,
        preflight_report=report,
        session=session,
        bindings=bindings,
        operator_confirmations=CONFIRMATIONS,
        post_capture_binding=_post_binding(bindings, session),
    )
    assert failed is True
    assert published["path"].is_file()
    assert published["publication_warnings"]
    loaded = raw.load_live_raw_v5(
        published["path"],
        repository_root=tmp_path,
        expected_bindings=bindings,
        expected_raw_file_sha256=published["raw_file_sha256"],
    )
    assert loaded["metadata"]["status"] == "CAPTURE_PASS"

    symlink_root = tmp_path / "symlink-root"
    symlink_root.mkdir()
    other = symlink_root / "other"
    other.mkdir()
    link = symlink_root / "results"
    link.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="parent symlink"):
        raw.publish_live_raw_v5(
            authority.SEALED_RAW_RELATIVE_PATH, repository_root=symlink_root,
            planned_submitted_pcm=planned, capture=_success_capture(planned),
            preflight_raw_int32=_preflight(),
            preflight_report=_preflight_report(_preflight()), session=_session(),
            bindings=bindings,
            operator_confirmations=CONFIRMATIONS,
            post_capture_binding=_post_binding(bindings, _session()),
        )


def test_loader_rejects_array_tamper_and_authority_tamper(tmp_path: Path) -> None:
    planned = _planned()
    _, _, bindings, published = _publish(tmp_path, _success_capture(planned))
    with pytest.raises(ValueError, match="file SHA"):
        raw.load_live_raw_v5(
            published["path"], repository_root=tmp_path,
            expected_bindings=bindings, expected_raw_file_sha256=_sha("wrong-raw"),
        )
    with np.load(published["path"], allow_pickle=False) as archive:
        members = {name: np.asarray(archive[name]) for name in archive.files}
    members["captured_pcm"] = members["captured_pcm"].copy()
    members["captured_pcm"][0, 0] += 1
    stream = io.BytesIO()
    np.savez(stream, **members)
    published["path"].write_bytes(stream.getvalue())
    changed_sha = hashlib.sha256(stream.getvalue()).hexdigest()
    with pytest.raises(ValueError, match="captured_pcm SHA"):
        raw.load_live_raw_v5(
            published["path"], repository_root=tmp_path,
            expected_bindings=bindings, expected_raw_file_sha256=changed_sha,
        )

    # 새 정상 raw에서 expected authority만 바꿔도 container binding과 일치하지 않는다.
    second_root = tmp_path / "second"
    second_root.mkdir()
    _, _, bindings, published = _publish(second_root, _success_capture(planned))
    forged = copy.deepcopy(bindings)
    forged["level_evidence"]["identity_sha256"] = _sha("forged-evidence-identity")
    with pytest.raises(ValueError, match="authority binding"):
        raw.load_live_raw_v5(
            published["path"], repository_root=second_root,
            expected_bindings=forged,
            expected_raw_file_sha256=published["raw_file_sha256"],
        )


def test_loader_rejects_noncanonical_repack_even_with_new_file_sha(tmp_path: Path) -> None:
    planned = _planned()
    _, _, bindings, published = _publish(tmp_path, _success_capture(planned))
    with np.load(published["path"], allow_pickle=False) as archive:
        members = {name: np.asarray(archive[name]) for name in archive.files}
    stream = io.BytesIO()
    np.savez_compressed(stream, **members)
    repacked = stream.getvalue()
    published["path"].write_bytes(repacked)
    with pytest.raises(ValueError, match="canonical writer bytes"):
        raw.load_live_raw_v5(
            published["path"], repository_root=tmp_path,
            expected_bindings=bindings,
            expected_raw_file_sha256=hashlib.sha256(repacked).hexdigest(),
        )


def test_wrong_plan_pcm_preflight_and_confirmations_fail_before_publish(tmp_path: Path) -> None:
    planned = _planned()
    preflight = _preflight()
    bindings = _bindings(planned, preflight)
    target = tmp_path / bindings["signal_plan"]["raw_session_relative_path"]
    bad = copy.deepcopy(bindings)
    bad["preflight"]["raw_sha256"] = _sha("not-preflight")
    with pytest.raises(ValueError, match="preflight raw SHA"):
        raw.publish_live_raw_v5(
            target, repository_root=tmp_path, planned_submitted_pcm=planned,
            capture=_success_capture(planned), preflight_raw_int32=preflight,
            preflight_report=_preflight_report(preflight), session=_session(),
            bindings=bad, operator_confirmations=CONFIRMATIONS,
            post_capture_binding=_post_binding(bad, _session()),
        )
    missing = dict(CONFIRMATIONS)
    missing["user_present"] = False
    with pytest.raises(ValueError, match="five|다섯"):
        raw.publish_live_raw_v5(
            target, repository_root=tmp_path, planned_submitted_pcm=planned,
            capture=_success_capture(planned), preflight_raw_int32=preflight,
            preflight_report=_preflight_report(preflight), session=_session(),
            bindings=bindings, operator_confirmations=missing,
            post_capture_binding=_post_binding(bindings, _session()),
        )
    assert not target.exists()


def test_session_sha_and_posix_path_types_are_exact(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lowercase"):
        raw._sha256(_sha("upper").upper(), label="test SHA")
    with pytest.raises(ValueError, match="lowercase"):
        raw._sha256(123, label="test SHA")
    with pytest.raises(ValueError, match="POSIX"):
        raw._relative_path("results\\raw.npz", label="test path")

    planned = _planned()
    preflight = _preflight()
    report = _preflight_report(preflight)
    bindings = _bindings(planned, preflight, report)
    kwargs = {
        "target": bindings["signal_plan"]["raw_session_relative_path"],
        "repository_root": tmp_path,
        "planned_submitted_pcm": planned,
        "capture": _success_capture(planned),
        "preflight_raw_int32": preflight,
        "preflight_report": report,
        "bindings": bindings,
        "operator_confirmations": CONFIRMATIONS,
        "post_capture_binding": _post_binding(bindings, _session()),
    }
    bad_id = _session()
    bad_id["capture_id"] = bad_id["capture_id"].upper()
    with pytest.raises(ValueError, match="capture_id"):
        raw.publish_live_raw_v5(session=bad_id, **kwargs)
    for field, value, match in (
        ("repository_commit", "abc", "repository_commit"),
        ("repository_dirty", True, "repository_dirty"),
        ("adapter_path", "scripts/data/other.py", "adapter_path"),
        ("adapter_file_sha256", "bad", "adapter_file_sha256"),
    ):
        bad_execution = _session()
        bad_execution[field] = value
        with pytest.raises(ValueError, match=match):
            raw.publish_live_raw_v5(session=bad_execution, **kwargs)
    reverse = _session()
    reverse["completed_at_utc"] = (
        dt.datetime.fromisoformat(reverse["started_at_utc"])
        - dt.timedelta(seconds=1)
    ).isoformat()
    reverse_root = tmp_path / "reverse"
    reverse_root.mkdir()
    reverse_kwargs = dict(kwargs)
    reverse_kwargs["repository_root"] = reverse_root
    reversed_publish = raw.publish_live_raw_v5(session=reverse, **reverse_kwargs)
    assert reversed_publish["metadata"]["status"] == "INVALID"
    assert "session_duration_zero_or_negative" in reversed_publish["metadata"]["invalid_reasons"]

    # 상대 target은 process cwd가 아니라 repository_root에 결속된다.
    published = raw.publish_live_raw_v5(session=_session(), **kwargs)
    assert published["path"] == tmp_path / bindings["signal_plan"][
        "raw_session_relative_path"
    ]
    loaded = raw.load_live_raw_v5(
        bindings["signal_plan"]["raw_session_relative_path"],
        repository_root=tmp_path,
        expected_bindings=bindings,
        expected_raw_file_sha256=published["raw_file_sha256"],
    )
    assert loaded["path"] == published["path"]


def test_preflight_report_is_self_contained_and_tamper_evident(tmp_path: Path) -> None:
    planned = _planned()
    _, _, bindings, published = _publish(tmp_path, _success_capture(planned))
    with np.load(published["path"], allow_pickle=False) as archive:
        members = {name: np.asarray(archive[name]) for name in archive.files}
    metadata = json.loads(bytes(members[raw.METADATA_MEMBER]).decode("utf-8"))
    metadata["preflight_report"]["channels"][0]["rms_dbfs"] = -55.0
    members[raw.METADATA_MEMBER] = np.frombuffer(
        raw._canonical_json_bytes(metadata), dtype=np.uint8
    ).copy()
    stream = io.BytesIO()
    np.savez(stream, **members)
    forged = stream.getvalue()
    published["path"].write_bytes(forged)
    with pytest.raises(ValueError, match="audio_io 재계산"):
        raw.load_live_raw_v5(
            published["path"], repository_root=tmp_path,
            expected_bindings=bindings,
            expected_raw_file_sha256=hashlib.sha256(forged).hexdigest(),
        )


def test_loader_revalidates_session_time_order(tmp_path: Path) -> None:
    planned = _planned()
    _, _, bindings, published = _publish(tmp_path, _success_capture(planned))
    with np.load(published["path"], allow_pickle=False) as archive:
        members = {name: np.asarray(archive[name]) for name in archive.files}
    metadata = json.loads(bytes(members[raw.METADATA_MEMBER]).decode("utf-8"))
    metadata["session"]["completed_at_utc"] = (
        dt.datetime.fromisoformat(metadata["session"]["started_at_utc"])
        - dt.timedelta(seconds=1)
    ).isoformat()
    members[raw.METADATA_MEMBER] = np.frombuffer(
        raw._canonical_json_bytes(metadata), dtype=np.uint8
    ).copy()
    stream = io.BytesIO()
    np.savez(stream, **members)
    forged = stream.getvalue()
    published["path"].write_bytes(forged)
    with pytest.raises(ValueError, match="metadata가 arrays/telemetry"):
        raw.load_live_raw_v5(
            published["path"], repository_root=tmp_path,
            expected_bindings=bindings,
            expected_raw_file_sha256=hashlib.sha256(forged).hexdigest(),
        )


@pytest.mark.parametrize("kind", ["zero", "low", "clipped", "stuck"])
def test_preflight_bad_signal_cannot_be_capture_pass(tmp_path: Path, kind: str) -> None:
    if kind == "zero":
        preflight = np.zeros((192, 2), dtype="<i4")
    elif kind == "low":
        values = (np.arange(192, dtype=np.int32) % 17) - 8
        preflight = np.column_stack((values, -values)).astype("<i4")
    elif kind == "clipped":
        values = np.where(np.arange(192) % 2, np.iinfo(np.int32).max, np.iinfo(np.int32).min)
        preflight = np.column_stack((values, -values - 1)).astype("<i4")
    else:
        preflight = np.full((192, 2), 2_000_000, dtype="<i4")
    report = _preflight_report(preflight)
    assert report["passed"] is False
    planned = _planned()
    bindings = _bindings(planned, preflight, report)
    session = _session()
    published = raw.publish_live_raw_v5(
        authority.SEALED_RAW_RELATIVE_PATH,
        repository_root=tmp_path,
        planned_submitted_pcm=planned,
        capture=_success_capture(planned),
        preflight_raw_int32=preflight,
        preflight_report=report,
        session=session,
        bindings=bindings,
        operator_confirmations=CONFIRMATIONS,
        post_capture_binding=_post_binding(bindings, session),
    )
    assert published["metadata"]["status"] == "INVALID"
    assert "preflight_invalid" in published["metadata"]["invalid_reasons"]


def test_forged_preflight_stats_and_identity_are_rejected(tmp_path: Path) -> None:
    planned = _planned()
    preflight = np.zeros((192, 2), dtype="<i4")
    report = _preflight_report(preflight)
    report["passed"] = True
    for channel in report["channels"]:
        channel.update(
            valid=True,
            stuck=False,
            rms_dbfs=-40.0,
            peak=0.01,
            clip_ratio=0.0,
            unique_codes=32,
        )
    bindings = _bindings(planned, preflight, report)
    session = _session()
    with pytest.raises(ValueError, match="audio_io 재계산"):
        raw.publish_live_raw_v5(
            authority.SEALED_RAW_RELATIVE_PATH,
            repository_root=tmp_path,
            planned_submitted_pcm=planned,
            capture=_success_capture(planned),
            preflight_raw_int32=preflight,
            preflight_report=report,
            session=session,
            bindings=bindings,
            operator_confirmations=CONFIRMATIONS,
            post_capture_binding=_post_binding(bindings, session),
        )


def test_chronology_device_and_refreshed_binding_fail_closed(tmp_path: Path) -> None:
    planned = _planned()
    preflight = _preflight()
    report = _preflight_report(preflight)
    bindings = _bindings(planned, preflight, report)

    wrong_device = _success_capture(planned)
    wrong_device[1]["resolved_input_device"] = 4
    session = _session()
    with pytest.raises(ValueError, match="resolved input device"):
        raw.publish_live_raw_v5(
            authority.SEALED_RAW_RELATIVE_PATH,
            repository_root=tmp_path / "wrong-device",
            planned_submitted_pcm=planned,
            capture=wrong_device,
            preflight_raw_int32=preflight,
            preflight_report=report,
            session=session,
            bindings=bindings,
            operator_confirmations=CONFIRMATIONS,
            post_capture_binding=_post_binding(bindings, session),
        )

    stale = copy.deepcopy(bindings)
    started = dt.datetime.fromisoformat(session["started_at_utc"])
    stale["meter"]["completed_at_utc"] = (started - dt.timedelta(seconds=601)).isoformat()
    stale_post = _post_binding(stale, session)
    stale_root = tmp_path / "stale"
    stale_root.mkdir()
    stale_publish = raw.publish_live_raw_v5(
        authority.SEALED_RAW_RELATIVE_PATH,
        repository_root=stale_root,
        planned_submitted_pcm=planned,
        capture=_success_capture(planned),
        preflight_raw_int32=preflight,
        preflight_report=report,
        session=session,
        bindings=stale,
        operator_confirmations=CONFIRMATIONS,
        post_capture_binding=stale_post,
    )
    assert "meter_session_age_invalid" in stale_publish["metadata"]["invalid_reasons"]

    forged_post = _post_binding(bindings, session)
    forged_post["refreshed_physical_fingerprint_sha256"] = _sha("changed-fingerprint")
    post_root = tmp_path / "post"
    post_root.mkdir()
    post_publish = raw.publish_live_raw_v5(
        authority.SEALED_RAW_RELATIVE_PATH,
        repository_root=post_root,
        planned_submitted_pcm=planned,
        capture=_success_capture(planned),
        preflight_raw_int32=preflight,
        preflight_report=report,
        session=session,
        bindings=bindings,
        operator_confirmations=CONFIRMATIONS,
        post_capture_binding=forged_post,
    )
    assert post_publish["metadata"]["status"] == "INVALID"
    assert "post_capture_binding_invalid" in post_publish["metadata"]["invalid_reasons"]


def test_caller_utc_cannot_override_actual_monotonic_duration(tmp_path: Path) -> None:
    planned = _planned()
    capture = _success_capture(planned)
    capture[1]["capture_monotonic_started"] = 200.0
    capture[1]["capture_monotonic_completed"] = 200.25
    capture[1]["capture_monotonic_elapsed_seconds"] = 0.25
    _, _, _, published = _publish(tmp_path, capture)
    assert published["metadata"]["status"] == "INVALID"
    assert "capture_monotonic_elapsed_outside_nominal_watchdog" in (
        published["metadata"]["invalid_reasons"]
    )


def test_prepared_utc_is_not_false_durable_freshness_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session()
    completed = dt.datetime.fromisoformat(session["completed_at_utc"])
    monkeypatch.setattr(
        raw,
        "_publisher_prepared_utc_now",
        lambda: (completed + dt.timedelta(hours=1)).isoformat(),
    )
    planned = _planned()
    _, _, _, published = _publish(tmp_path, _success_capture(planned))
    assert published["metadata"]["status"] == "CAPTURE_PASS"
    assert all("publication_stale" not in reason for reason in published["metadata"]["invalid_reasons"])


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("zero", "session_duration_zero_or_negative"),
        ("future", "session_completed_after_publisher_preparation"),
    ],
)
def test_internal_publication_chronology_cannot_capture_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, reason: str
) -> None:
    planned = _planned()
    preflight = _preflight()
    report = _preflight_report(preflight)
    bindings = _bindings(planned, preflight, report)
    base = dt.datetime(2026, 8, 29, 3, 0, 0, tzinfo=dt.timezone.utc)
    session = _session()
    session["started_at_utc"] = (base - dt.timedelta(seconds=FRAMES / 48_000.0)).isoformat()
    session["completed_at_utc"] = base.isoformat()
    prepared_at = base + dt.timedelta(seconds=1)
    if case == "zero":
        session["started_at_utc"] = base.isoformat()
    elif case == "future":
        prepared_at = base - dt.timedelta(seconds=1)
    bindings["meter"]["completed_at_utc"] = (
        dt.datetime.fromisoformat(session["started_at_utc"]) - dt.timedelta(seconds=60)
    ).isoformat()
    monkeypatch.setattr(
        raw, "_publisher_prepared_utc_now", lambda: prepared_at.isoformat()
    )
    published = raw.publish_live_raw_v5(
        authority.SEALED_RAW_RELATIVE_PATH,
        repository_root=tmp_path,
        planned_submitted_pcm=planned,
        capture=_success_capture(planned),
        preflight_raw_int32=preflight,
        preflight_report=report,
        session=session,
        bindings=bindings,
        operator_confirmations=CONFIRMATIONS,
        post_capture_binding=_post_binding(bindings, session),
    )
    assert published["metadata"]["status"] == "INVALID"
    assert reason in published["metadata"]["invalid_reasons"]


class _BackendStop(Exception):
    pass


class _BackendAbort(Exception):
    pass


class _BackendStatus:
    input_overflow = True

    def __bool__(self) -> bool:
        return True


class _LiveBackend:
    CallbackStop = _BackendStop
    CallbackAbort = _BackendAbort

    def __init__(self, mode: str = "success") -> None:
        self.mode = mode
        self.calls: list[object] = []

    def Stream(self, **kwargs):
        outer = self

        class Stream:
            def start(self) -> None:
                outer.calls.append("start")
                for index in range(FRAMES // audio.BLOCK_SIZE):
                    frames = 128 if outer.mode == "callback_failure" and index == 1 else 256
                    input_data = np.full((frames, 2), index + 1, dtype="<i4")
                    output_data = np.full((frames, 2), 99, dtype="<i2")
                    time_info = {
                        "inputBufferAdcTime": index + 0.1,
                        "outputBufferDacTime": index + 0.2,
                        "currentTime": index + 0.3,
                    }
                    status = _BackendStatus() if outer.mode == "xrun" and index == 1 else None
                    try:
                        kwargs["callback"](input_data, output_data, frames, time_info, status)
                    except (_BackendStop, _BackendAbort):
                        break

            def stop(self, *, ignore_errors: bool) -> None:
                outer.calls.append(("stop", ignore_errors))

            def abort(self, *, ignore_errors: bool) -> None:
                outer.calls.append(("abort", ignore_errors))

            def close(self, *, ignore_errors: bool) -> None:
                outer.calls.append(("close", ignore_errors))

        return Stream()


@pytest.mark.parametrize(
    ("mode", "expected_status"),
    [("success", "CAPTURE_PASS"), ("xrun", "INVALID"), ("callback_failure", "INVALID")],
)
def test_fake_backend_capture_publish_load_e2e(
    tmp_path: Path, mode: str, expected_status: str
) -> None:
    planned = _planned()
    backend = _LiveBackend(mode)
    ticks = iter([100.0, 100.0 + FRAMES / 48_000.0])
    try:
        capture = audio.capture_duplex_v5(
            backend,
            submitted_pcm=planned,
            input_device=3,
            output_device=7,
            monotonic=lambda: next(ticks),
        )
    except audio.DuplexCaptureFailure as failure:
        capture = failure
    _, _, bindings, published = _publish(tmp_path, capture)
    assert published["metadata"]["status"] == expected_status
    loaded = raw.load_live_raw_v5(
        published["path"],
        repository_root=tmp_path,
        expected_bindings=bindings,
        expected_raw_file_sha256=published["raw_file_sha256"],
    )
    assert loaded["metadata"]["status"] == expected_status
    if expected_status == "INVALID":
        with pytest.raises(ValueError, match="analysis admission"):
            raw.admit_live_raw_v5_for_analysis(
                published["path"],
                repository_root=tmp_path,
                expected_bindings=bindings,
                expected_raw_file_sha256=published["raw_file_sha256"],
            )


def test_fake_backend_wrong_device_never_reaches_publisher(tmp_path: Path) -> None:
    backend = _LiveBackend()
    with pytest.raises(ValueError, match="device"):
        audio.capture_duplex_v5(
            backend, submitted_pcm=_planned(), input_device=-1, output_device=7
        )
    assert backend.calls == []
    assert not (tmp_path / authority.SEALED_RAW_RELATIVE_PATH).exists()
