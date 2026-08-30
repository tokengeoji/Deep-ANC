from __future__ import annotations

import builtins
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import deep_anc.audio_duplex_v5 as duplex
import deep_anc.audio_io as audio_io
import deep_anc.dsp.fullband_live_post_v5 as post
import deep_anc.dsp.fullband_live_raw_v5 as live_raw
import deep_anc.dsp.measurement_level as measurement
from deep_anc.dsp.fullband_causal_v5 import build_plan_v5


SCRIPT = Path("scripts/data/measure_paths_fullband_causal_v5.py")


def _load_script(name: str = "fullband_v5_adapter_test"):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _binding(preflight: dict) -> dict:
    return {
        "signal_plan": {
            "schema": "fullband_causal_signal_plan_envelope_v5",
            "path": "assets/contracts/fullband_causal_v5_signal_plan.json",
            "file_sha256": "1" * 64,
            "payload_sha256": "2" * 64,
            "pcm_sha256": "3" * 64,
            "raw_session_relative_path": "results/fullband_causal_v5/raw_capture.npz",
        },
        "live_capture_authority": {
            "schema": "fullband_causal_v5_live_capture_authority_v1",
            "path": "assets/contracts/fullband_causal_v5_live_capture_authority.json",
            "file_sha256": "4" * 64,
            "payload_sha256": "5" * 64,
            "signal_plan_file_sha256": "1" * 64,
            "signal_plan_payload_sha256": "2" * 64,
            "signal_pcm_sha256": "3" * 64,
            "hardware_file_sha256": "6" * 64,
            "raw_session_relative_path": "results/fullband_causal_v5/raw_capture.npz",
        },
        "meter": {
            "schema": "measurement_level_meter_raw_v1",
            "path": "results/fullband_causal_v5/level_meter/meter_raw.npz",
            "receipt_path": "results/fullband_causal_v5/level_meter/meter_raw.receipt.json",
            "raw_sha256": "7" * 64,
            "receipt_sha256": "8" * 64,
            "completed_at_utc": "2026-08-29T00:00:00+00:00",
            "identity_sha256": "9" * 64,
            "followup_contract_sha256": "a" * 64,
            "live_authority_file_sha256": "4" * 64,
            "level_evidence_file_sha256": "b" * 64,
            "hardware_file_sha256": "6" * 64,
        },
        "level_evidence": {
            "schema": "measurement_level_evidence_v2_bootstrap_pair",
            "path": "assets/measured/measurement_level_evidence.json",
            "file_sha256": "b" * 64,
            "identity_sha256": "c" * 64,
            "scope": "tracked_historical_attestation_for_fresh_v5_meter_only",
            "preserved_raw_revalidated": False,
        },
        "hardware": {
            "schema": "jetson_measurement_hardware_v1",
            "path": "configs/hardware_jetson.yaml",
            "file_sha256": "6" * 64,
            "identity_sha256": "d" * 64,
            "physical_fingerprint_sha256": "e" * 64,
            "resolved_devices": {"input": 11, "output": 12},
        },
        "preflight": preflight,
    }


def test_repository_audio_lock_file_matches_live_validator_identity(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()

    with measurement.repository_audio_lock(
        tmp_path, purpose="fullband_causal_v5_live_capture"
    ) as audio_lock:
        receipt = post.validate_held_audio_lock(
            tmp_path,
            audio_lock,
            expected_purpose="fullband_causal_v5_live_capture",
        )
        saved = json.loads((tmp_path / audio_lock["path"]).read_text(encoding="utf-8"))

    assert saved == audio_lock
    assert receipt["device"] == audio_lock["device"]
    assert receipt["inode"] == audio_lock["inode"]
    assert receipt["exclusive_lock_observed"] is True
    assert receipt["identity_sha256"] == post.audio_lock_identity_sha256(audio_lock)


def test_repository_audio_lock_rejects_named_inode_replacement(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()

    with measurement.repository_audio_lock(
        tmp_path, purpose="fullband_causal_v5_live_capture"
    ) as audio_lock:
        lock_path = tmp_path / audio_lock["path"]
        detached = lock_path.with_name("detached-held-lock")
        lock_path.rename(detached)
        replacement = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            payload = json.dumps(
                audio_lock,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            assert os.write(replacement, payload) == len(payload)
            fcntl.flock(replacement, fcntl.LOCK_EX | fcntl.LOCK_NB)

            with pytest.raises(RuntimeError, match="inode"):
                post.validate_held_audio_lock(
                    tmp_path,
                    audio_lock,
                    expected_purpose="fullband_causal_v5_live_capture",
                )
        finally:
            fcntl.flock(replacement, fcntl.LOCK_UN)
            os.close(replacement)


def _fake_static(plan: dict) -> dict:
    return {
        "exact_plan": {"envelope": {"signal_plan": plan}},
        "hardware_audio": {
            "sample_rate": 48_000,
            "block_size": 256,
            "latency": "low",
            "input": {"card": "APE", "pcm": 1, "channels": 2},
            "output": {"card": "Audio", "pcm": 0, "channels": 2},
        },
        "hardware_config": {},
        "hardware": {"identity_sha256": "d" * 64},
        "physical_fingerprint": {"sha256": "e" * 64},
        "prevalidated_meter": {
            "hardware": {"resolved_devices": {"input": 11, "output": 12}}
        },
    }


def _live_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        plan_envelope="assets/contracts/fullband_causal_v5_signal_plan.json",
        live_authority="assets/contracts/fullband_causal_v5_live_capture_authority.json",
        meter_raw="results/fullband_causal_v5/level_meter/meter_raw.npz",
        level_evidence="assets/measured/measurement_level_evidence.json",
        hardware="configs/hardware_jetson.yaml",
        raw_target="results/fullband_causal_v5/raw_capture.npz",
        confirm_speaker=True,
        confirm_user_present=True,
        confirm_volume_minimum=True,
        confirm_routing_and_geometry=True,
        confirm_same_amplifier_setting=True,
    )


def _wire_fake_live(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    failure: bool = False,
) -> tuple[object, list[str]]:
    module = _load_script(f"v5_fake_live_{failure}")
    plan, submitted = build_plan_v5()
    events: list[str] = []
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "build_plan_v5", lambda: (plan, submitted.copy()))
    monkeypatch.setattr(
        module,
        "_repository_execution_identity",
        lambda: {
            "repository_commit": "a" * 40,
            "repository_branch": "work/test",
            "repository_dirty": False,
            "adapter_path": module.ADAPTER_REPOSITORY_PATH,
            "adapter_file_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(module, "_static_contract_before_backend_import", lambda _args: _fake_static(plan))
    monkeypatch.setattr(module, "_resolve_devices", lambda _static: {"input": 11, "output": 12})
    backend = object()
    original_import = module.importlib.import_module

    def import_module(name: str):
        if name == "sounddevice":
            events.append("backend_import")
            return backend
        return original_import(name)

    monkeypatch.setattr(module.importlib, "import_module", import_module)
    meter_api = SimpleNamespace(
        validate_fullband_v5_meter_raw=lambda *_a, **_k: {
            "hardware": {"resolved_devices": {"input": 11, "output": 12}}
        }
    )
    monkeypatch.setattr(module, "_meter_contract_module", lambda: meter_api)
    rng = np.random.default_rng(5)
    preflight_raw = rng.integers(-2_000_000, 2_000_000, size=(512, 2), dtype=np.int32)
    analyzed = audio_io.analyze_int32_input_probe(preflight_raw)
    public_report = {
        **analyzed,
        "passed": True,
        "resolved_input_device": 11,
        "sample_rate_hz": 48_000,
        "capture_seconds": 1.5,
        "settle_seconds": 0.5,
        "analyzed_seconds": len(preflight_raw) / 48_000,
    }
    monkeypatch.setattr(
        audio_io,
        "capture_measurement_preflight_raw",
        lambda *_a, **_k: (events.append("input_preflight") or (preflight_raw.copy(), public_report)),
    )

    @contextmanager
    def fake_lock(_root, *, purpose):
        events.append("lock_acquired")
        try:
            yield {
                "path": "results/.live_audio_uid_1000.lock",
                "pid": os.getpid(),
                "uid": os.getuid(),
                "purpose": purpose,
                "device": 1,
                "inode": 2,
            }
        finally:
            events.append("lock_released")

    monkeypatch.setattr(measurement, "repository_audio_lock", fake_lock)
    monkeypatch.setattr(measurement, "assert_live_pcm_clock_preconditions", lambda _h: events.append("readonly_gate"))
    monkeypatch.setattr(measurement, "collect_alsa_physical_fingerprint", lambda _h: {"sha256": "e" * 64})
    monkeypatch.setattr(post, "validate_held_audio_lock", lambda *_a, **_k: events.append("lock_validated") or {})
    monkeypatch.setattr(post, "assert_repository_target_fresh_nofollow", lambda *_a, **_k: events.append("fresh"))

    def collect_bindings(*_args, preflight_binding, **_kwargs):
        events.append("external_bindings")
        return _binding(dict(preflight_binding))

    monkeypatch.setattr(post, "collect_actual_external_bindings_v5", collect_bindings)

    if failure:
        partial = np.zeros_like(submitted, dtype="<i4")
        actual = np.zeros_like(submitted, dtype="<i2")
        mask = np.zeros(len(submitted), dtype=np.bool_)

        def capture(*_args, pre_open_check, **_kwargs):
            pre_open_check()
            events.append("capture_closed")
            _kwargs["on_output_closed"](False)
            raise duplex.DuplexCaptureFailure(
                "fake failure",
                partial,
                actual,
                mask,
                mask.copy(),
                {"output_stop_confirmed": False},
            )

    else:
        def capture(*_args, pre_open_check, **_kwargs):
            pre_open_check()
            events.append("capture_closed")
            _kwargs["on_output_closed"](True)
            return np.zeros_like(submitted, dtype="<i4"), {"output_stop_confirmed": True}

    monkeypatch.setattr(duplex, "capture_duplex_v5", capture)

    def publish(_target, **kwargs):
        events.append("raw_publish")
        assert isinstance(kwargs["capture"], duplex.DuplexCaptureFailure) is failure
        status = "INVALID" if failure else "CAPTURE_PASS"
        return {
            "path": tmp_path / "results/fullband_causal_v5/raw_capture.npz",
            "raw_file_sha256": "f" * 64,
            "metadata": {
                "status": status,
                "session": kwargs["session"],
            },
        }

    monkeypatch.setattr(live_raw, "publish_live_raw_v5", publish)

    def issue(**_kwargs):
        events.append("post_receipt")
        return {
            "relative_path": "results/fullband_causal_v5/raw_capture.npz.post_receipt.json",
            "file_sha256": "0" * 64,
            "receipt": {"valid": not failure},
        }

    monkeypatch.setattr(post, "issue_external_post_capture_receipt_v5", issue)
    monkeypatch.setattr(
        post,
        "issue_invalid_external_post_capture_receipt_v5",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("normal fake issue should not fail")),
    )
    return module, events


def test_execute_live_orders_lock_input_capture_disconnect_and_durable_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, events = _wire_fake_live(monkeypatch, tmp_path)
    import deep_anc.dsp.fullband_live_delay_core as delay_core

    monkeypatch.setattr(
        delay_core,
        "analyze_committed_v5_live_delay",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("capture command가 offline analyzer를 호출했습니다")
        ),
    )
    original_print = builtins.print

    def observed_print(*args, **kwargs):
        if "스피커 출력 종료" in " ".join(str(value) for value in args):
            events.append("disconnect_notice")
        return original_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", observed_print)
    code = module._execute_live(_live_args(tmp_path))
    assert code == 0
    assert events.index("lock_acquired") < events.index("input_preflight")
    assert events.index("capture_closed") < events.index("disconnect_notice")
    assert events.index("disconnect_notice") < events.index("raw_publish")
    assert events.index("raw_publish") < events.index("post_receipt")
    assert events.index("post_receipt") < events.index("lock_released")


def test_input_preflight_failure_after_lock_prints_disconnect_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    module, events = _wire_fake_live(monkeypatch, tmp_path)
    monkeypatch.setattr(
        audio_io,
        "capture_measurement_preflight_raw",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("input failed")),
    )
    assert module._execute_live(_live_args(tmp_path)) == 1
    assert "출력 시작 전 중단" in capsys.readouterr().err
    assert events.index("lock_acquired") < events.index("lock_released")


def test_dirty_checkout_blocks_before_backend_import(tmp_path, monkeypatch):
    module = _load_script("v5_dirty_checkout")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "_repository_execution_identity",
        lambda: (_ for _ in ()).throw(RuntimeError("dirty repository checkout")),
    )
    called = []
    monkeypatch.setattr(module.importlib, "import_module", lambda name: called.append(name))
    assert module._execute_live(_live_args(tmp_path)) == 2
    assert called == []


def test_offline_command_uses_absolute_current_interpreter_and_script(tmp_path):
    module = _load_script("v5_absolute_offline_command")
    command = module._offline_command(
        args=_live_args(tmp_path),
        receipt_relative_path="results/fullband_causal_v5/raw_capture.post_receipt.json",
        receipt_file_sha256="a" * 64,
        capture_id="b" * 32,
    )
    assert command.startswith(str(Path(sys.executable).absolute()))
    assert str(Path(module.__file__).resolve()) in command


def test_post_validation_failure_publishes_invalid_receipt_and_keeps_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, events = _wire_fake_live(monkeypatch, tmp_path)
    monkeypatch.setattr(
        post,
        "issue_external_post_capture_receipt_v5",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("post bytes changed")),
    )

    def invalid_issue(**kwargs):
        events.append("invalid_post_receipt")
        assert kwargs["expected_raw_file_sha256"] == "f" * 64
        assert "post bytes changed" in kwargs["errors"][0]
        return {
            "relative_path": "results/fullband_causal_v5/raw_capture.npz.post_receipt.json",
            "file_sha256": "0" * 64,
            "receipt": {"valid": False},
        }

    monkeypatch.setattr(post, "issue_invalid_external_post_capture_receipt_v5", invalid_issue)
    assert module._execute_live(_live_args(tmp_path)) == 1
    assert events.index("raw_publish") < events.index("invalid_post_receipt")
    assert events.index("invalid_post_receipt") < events.index("lock_released")


def test_duplex_failure_is_preserved_after_unconfirmed_disconnect_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module, events = _wire_fake_live(monkeypatch, tmp_path, failure=True)
    code = module._execute_live(_live_args(tmp_path))
    output = capsys.readouterr()
    assert code == 1
    assert "출력 종료 확인 불가" in output.out
    assert "INVALID" in output.err
    assert events.index("capture_closed") < events.index("raw_publish")
    assert "post_receipt" in events


@pytest.mark.parametrize("reason", ["authority tamper", "meter tamper"])
def test_static_tamper_aborts_before_backend_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    module = _load_script(f"v5_static_tamper_{reason}")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "_repository_execution_identity", lambda: {})
    monkeypatch.setattr(
        module,
        "_static_contract_before_backend_import",
        lambda _args: (_ for _ in ()).throw(ValueError(reason)),
    )
    called = []
    monkeypatch.setattr(module.importlib, "import_module", lambda name: called.append(name))
    assert module._execute_live(_live_args(tmp_path)) == 2
    assert called == []


def test_wrong_resolved_device_aborts_before_audio_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("v5_wrong_device")
    plan, submitted = build_plan_v5()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "_repository_execution_identity", lambda: {})
    monkeypatch.setattr(module, "build_plan_v5", lambda: (plan, submitted.copy()))
    monkeypatch.setattr(module, "_static_contract_before_backend_import", lambda _a: _fake_static(plan))
    monkeypatch.setattr(module, "_resolve_devices", lambda _s: {"input": 11, "output": 12})
    real_import = module.importlib.import_module
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: object() if name == "sounddevice" else real_import(name),
    )
    monkeypatch.setattr(
        module,
        "_meter_contract_module",
        lambda: SimpleNamespace(
            validate_fullband_v5_meter_raw=lambda *_a, **_k: {
                "hardware": {"resolved_devices": {"input": 99, "output": 12}}
            }
        ),
    )
    lock_calls = []
    monkeypatch.setattr(measurement, "repository_audio_lock", lambda *_a, **_k: lock_calls.append(True))
    assert module._execute_live(_live_args(tmp_path)) == 2
    assert lock_calls == []


def test_missing_meter_and_dry_run_do_not_import_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("v5_no_backend")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    calls = []
    real_import = module.importlib.import_module

    def no_sounddevice(name: str):
        if name == "sounddevice":
            calls.append(name)
            raise AssertionError("dry/missing-meter path imported backend")
        return real_import(name)

    monkeypatch.setattr(module.importlib, "import_module", no_sounddevice)
    assert module.main(["--execute-live"]) == 2
    plan, pcm = build_plan_v5()
    monkeypatch.setattr(module, "build_plan_v5", lambda **_k: (plan, pcm))
    monkeypatch.setattr(
        module,
        "exact_condition_audit_v5",
        lambda *_a, **_k: {"joint_fit_condition_number": 9.0},
    )
    assert module.main(["--dry-run"]) == 0
    assert calls == []


def test_external_receipt_offline_rejects_raw_array_splice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = "results/fullband_causal_v5/raw_capture.npz.post_receipt.json"
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    bindings = {name: {} for name in live_raw.BINDING_KEYS}
    metadata = {
        "bindings": bindings,
        "array_sha256": {"captured_pcm": "a" * 64},
        "session": {"capture_id": "1" * 32},
        "status": "CAPTURE_PASS",
        "schema": live_raw.LIVE_RAW_SCHEMA,
        "valid": True,
    }
    core = {
        "schema": post.EXTERNAL_POST_RECEIPT_SCHEMA,
        "status": "POST_CAPTURE_PASS",
        "valid": True,
        "invalid_reasons": [],
        "scope": post.EXTERNAL_POST_RECEIPT_SCOPE,
        "raw": {
            "path": "results/fullband_causal_v5/raw_capture.npz",
            "file_sha256": "2" * 64,
            "metadata_payload_sha256": post._payload_sha256(metadata),
            "bindings_payload_sha256": post._payload_sha256(bindings),
            "array_sha256": {"captured_pcm": "b" * 64},  # deliberate splice
            "capture_id": "1" * 32,
            "status": "CAPTURE_PASS",
            "schema": live_raw.LIVE_RAW_SCHEMA,
        },
        "external_bindings": bindings,
        "operator_confirmations": {},
        "primitive_post_capture_binding": {},
        "audio_lock": {},
        "resolved_devices": {"input": 11, "output": 12},
        "analysis_admission_eligible": True,
        "canonical_training_eligible": False,
        "hardware_sample_slip_authority": False,
    }
    receipt = {**core, "receipt_payload_sha256": post._payload_sha256(core)}
    payload = post._canonical_json_file_bytes(receipt)
    target.write_bytes(payload)
    monkeypatch.setattr(
        post,
        "_collect_offline_external_bindings_without_backend",
        lambda **_kwargs: bindings,
    )
    monkeypatch.setattr(
        post,
        "load_live_raw_v5",
        lambda *_a, **_k: {"metadata": metadata, "arrays": {}},
    )
    with pytest.raises(ValueError, match="array SHA"):
        post.load_external_post_capture_receipt_v5(
            repository_root=tmp_path,
            receipt_relative_path=relative,
            expected_receipt_file_sha256=hashlib.sha256(payload).hexdigest(),
            plan_envelope_path="assets/contracts/fullband_causal_v5_signal_plan.json",
            live_authority_path="assets/contracts/fullband_causal_v5_live_capture_authority.json",
            meter_raw_path="results/fullband_causal_v5/level_meter/meter_raw.npz",
            level_evidence_path="assets/measured/measurement_level_evidence.json",
            hardware_path="configs/hardware_jetson.yaml",
        )


def test_analysis_operator_publish_is_atomic_noreplace_and_keeps_authority_false(
    tmp_path: Path,
) -> None:
    array = np.arange(8, dtype="<f8")
    operator = {
        "primary": array,
        "receipt": {
            "operator_array_sha256": {"primary": post._array_contract_sha256(array)},
            "canonical_training_eligible": False,
            "hardware_sample_slip_authority_available": False,
        },
    }
    analysis = {
        "canonical_training_eligible": False,
        "hardware_slip_authority_available": False,
    }
    published = post.publish_live_delay_analysis_v5(
        repository_root=tmp_path,
        output_directory_relative_path="results/fullband_causal_v5/analysis_test",
        external_receipt_file_sha256="f" * 64,
        analysis=analysis,
        operator=operator,
    )
    assert published["analysis"]["path"].is_file()
    assert published["operator"]["path"].is_file()
    envelope = json.loads(published["analysis"]["path"].read_text())
    assert envelope["canonical_training_eligible"] is False
    assert envelope["hardware_sample_slip_authority"] is False
    with pytest.raises(FileExistsError):
        post.publish_live_delay_analysis_v5(
            repository_root=tmp_path,
            output_directory_relative_path="results/fullband_causal_v5/analysis_test",
            external_receipt_file_sha256="f" * 64,
            analysis=analysis,
            operator=operator,
        )


def test_post_file_primitives_reject_symlink_and_no_replace(tmp_path: Path) -> None:
    published = post._publish_json_noreplace(tmp_path, "results/post.json", {"a": 1})
    assert published["path"].is_file()
    with pytest.raises(FileExistsError):
        post._publish_json_noreplace(tmp_path, "results/post.json", {"a": 2})
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        post.read_repository_file_nofollow(tmp_path, "linked/post.json")
