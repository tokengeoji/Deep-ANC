from __future__ import annotations

import builtins
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import deep_anc.audio_duplex_v6 as duplex
import deep_anc.audio_io as audio_io
import deep_anc.dsp.fullband_live_delay_core_v6 as delay_core
import deep_anc.dsp.fullband_live_post_v6 as post
import deep_anc.dsp.fullband_live_raw_v6 as live_raw
import deep_anc.dsp.fullband_v6_meter as meter
import deep_anc.dsp.measurement_level as measurement
from deep_anc.dsp.fullband_causal_v6 import V6ClockAdmissionError, build_plan_v6


SCRIPT = Path("scripts/data/measure_paths_fullband_causal_v6.py")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        plan_envelope="assets/contracts/fullband_causal_v6_signal_plan.json",
        live_authority="assets/contracts/fullband_causal_v6_live_capture_authority.json",
        meter_raw="results/fullband_causal_v6/level_meter/meter_raw.npz",
        level_evidence="assets/measured/measurement_level_evidence.json",
        hardware="configs/hardware_jetson.yaml",
        raw_target="results/fullband_causal_v6/raw_capture.npz",
        post_receipt="results/fullband_causal_v6/raw_capture.npz.post_receipt.json",
        expected_post_receipt_sha256="a" * 64,
        analysis_output="results/fullband_causal_v6/analysis_test",
        failure_output="results/fullband_causal_v6/failure_test.json",
        confirm_speaker=True,
        confirm_user_present=True,
        confirm_volume_minimum=True,
        confirm_routing_and_geometry=True,
        confirm_same_amplifier_setting=True,
    )


def _binding(preflight: dict) -> dict:
    return {
        "signal_plan": {
            "path": "assets/contracts/fullband_causal_v6_signal_plan.json",
            "file_sha256": "1" * 64,
            "payload_sha256": "2" * 64,
            "pcm_sha256": "3" * 64,
        },
        "live_capture_authority": {
            "path": "assets/contracts/fullband_causal_v6_live_capture_authority.json",
            "file_sha256": "4" * 64,
            "payload_sha256": "5" * 64,
        },
        "meter": {
            "path": "results/fullband_causal_v6/level_meter/meter_raw.npz",
            "raw_sha256": "6" * 64,
            "receipt_sha256": "7" * 64,
        },
        "level_evidence": {
            "path": "assets/measured/measurement_level_evidence.json",
            "file_sha256": "8" * 64,
        },
        "hardware": {
            "path": "configs/hardware_jetson.yaml",
            "file_sha256": "9" * 64,
            "identity_sha256": "a" * 64,
            "physical_fingerprint_sha256": "b" * 64,
            "resolved_devices": {"input": 11, "output": 12},
        },
        "preflight": preflight,
    }


def _wire_live(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _load_script("v6_adapter_live")
    plan, submitted = build_plan_v6()
    events: list[str] = []
    prevalidated_meter = {"hardware": {"resolved_devices": {"input": 11, "output": 12}}}
    static = {
        "exact_plan": {"signal_plan": plan},
        "hardware_audio": {
            "sample_rate": 48_000,
            "block_size": 256,
            "latency": "low",
            "input": {"card": "APE", "pcm": 1, "channels": 2},
            "output": {"card": "Audio", "pcm": 0, "channels": 2},
        },
        "hardware_config": {},
        "hardware": {"identity_sha256": "a" * 64},
        "physical_fingerprint": {"sha256": "b" * 64},
        "prevalidated_meter": prevalidated_meter,
    }
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "build_plan_v6",
        lambda **_kwargs: (plan, submitted.copy()),
    )
    monkeypatch.setattr(
        module,
        "_repository_execution_identity",
        lambda: {
            "repository_commit": "c" * 40,
            "repository_branch": "work/test-v6",
            "repository_dirty": False,
            "adapter_path": module.ADAPTER_REPOSITORY_PATH,
            "adapter_file_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(module, "_static_contract_before_backend_import", lambda _a: static)
    monkeypatch.setattr(module, "_resolve_devices", lambda _s: {"input": 11, "output": 12})
    real_import = module.importlib.import_module

    def import_module(name: str):
        if name == "sounddevice":
            events.append("backend_import")
            return object()
        return real_import(name)

    monkeypatch.setattr(module.importlib, "import_module", import_module)
    monkeypatch.setattr(meter, "validate_fullband_v6_meter_raw", lambda *_a, **_k: prevalidated_meter)
    preflight = np.tile(np.asarray([[1_000_000, -1_000_000]], dtype="<i4"), (256, 1))
    report = {
        "passed": True,
        "resolved_input_device": 11,
        "sample_rate_hz": 48_000,
        "frames": len(preflight),
        "channels": audio_io.analyze_int32_input_probe(preflight)["channels"],
    }
    monkeypatch.setattr(
        audio_io,
        "capture_measurement_preflight_raw",
        lambda *_a, **_k: (events.append("input_preflight") or (preflight.copy(), report)),
    )

    @contextmanager
    def lock(_root, *, purpose):
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

    monkeypatch.setattr(measurement, "repository_audio_lock", lock)
    monkeypatch.setattr(measurement, "assert_live_pcm_clock_preconditions", lambda _h: events.append("read_only_gate"))
    monkeypatch.setattr(measurement, "collect_alsa_physical_fingerprint", lambda _h: {"sha256": "b" * 64})
    monkeypatch.setattr(post, "audio_lock_identity_sha256", lambda _l: "e" * 64)
    monkeypatch.setattr(post, "validate_held_audio_lock", lambda *_a, **_k: events.append("lock_validated") or {})
    monkeypatch.setattr(post, "assert_repository_target_fresh_nofollow", lambda *_a, **_k: events.append("fresh"))

    def collect(*_args, preflight_binding, **_kwargs):
        events.append("external_bindings")
        return _binding(dict(preflight_binding))

    monkeypatch.setattr(post, "collect_actual_external_bindings_v6", collect)

    def capture(*_args, pre_open_check, on_output_closed, **_kwargs):
        pre_open_check()
        events.append("capture_closed")
        on_output_closed(True)
        return np.zeros_like(submitted, dtype="<i4"), {"output_stop_confirmed": True}

    monkeypatch.setattr(duplex, "capture_duplex_v6", capture)

    def publish(_target, **kwargs):
        events.append("raw_publish")
        return {
            "path": tmp_path / "results/fullband_causal_v6/raw_capture.npz",
            "raw_file_sha256": "f" * 64,
            "metadata": {
                "status": "CAPTURE_PASS",
                "session": kwargs["session"],
            },
        }

    monkeypatch.setattr(live_raw, "publish_live_raw_v6", publish)
    monkeypatch.setattr(
        post,
        "issue_external_post_capture_receipt_v6",
        lambda **_kwargs: (
            events.append("post_receipt")
            or {
                "relative_path": "results/fullband_causal_v6/raw_capture.npz.post_receipt.json",
                "file_sha256": "0" * 64,
                "receipt": {"valid": True},
            }
        ),
    )
    monkeypatch.setattr(
        post,
        "issue_invalid_external_post_capture_receipt_v6",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("invalid receipt 경로")),
    )
    return module, events


def test_execute_live_orders_close_notice_before_raw_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, events = _wire_live(monkeypatch, tmp_path)
    original_print = builtins.print

    def observed_print(*values, **kwargs):
        if "스피커 출력 종료" in " ".join(str(value) for value in values):
            events.append("disconnect_notice")
        return original_print(*values, **kwargs)

    monkeypatch.setattr(builtins, "print", observed_print)
    assert module._execute_live(_args()) == 0
    assert events.index("lock_acquired") < events.index("input_preflight")
    assert events.index("capture_closed") < events.index("disconnect_notice")
    assert events.index("disconnect_notice") < events.index("raw_publish")
    assert events.index("raw_publish") < events.index("post_receipt")
    assert events.index("post_receipt") < events.index("lock_released")


def test_dirty_checkout_and_missing_meter_never_import_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("v6_adapter_dirty")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "_repository_execution_identity",
        lambda: (_ for _ in ()).throw(RuntimeError("dirty repository checkout")),
    )
    imports: list[str] = []
    monkeypatch.setattr(module.importlib, "import_module", lambda name: imports.append(name))
    assert module._execute_live(_args()) == 2
    assert imports == []

    imports.clear()
    assert module.main(["--execute-live"]) == 2
    assert imports == []


def _offline_admission() -> dict:
    arrays = {
        "actual_submitted_pcm": np.zeros((256, 2), dtype="<i2"),
        "captured_pcm": np.zeros((256, 2), dtype="<i4"),
        "capture_valid_mask": np.ones(256, dtype=np.bool_),
        "submitted_valid_mask": np.ones(256, dtype=np.bool_),
        "callback_sequence": np.asarray([0], dtype="<i8"),
        "callback_start_frames": np.asarray([0], dtype="<i8"),
        "callback_frame_counts": np.asarray([256], dtype="<i8"),
        "input_buffer_adc_time": np.asarray([0.0]),
        "output_buffer_dac_time": np.asarray([0.0]),
        "callback_current_time": np.asarray([0.0]),
        "callback_status_bitmask": np.asarray([0], dtype="<u4"),
    }
    return {
        "receipt_file_sha256": "a" * 64,
        "raw": {
            "raw_file_sha256": "b" * 64,
            "arrays": arrays,
            "metadata": {
                "duplex_telemetry_scalars": {},
                "session": _execution_identity(),
            },
        },
    }


def _execution_identity() -> dict:
    return {
        "repository_commit": "c" * 40,
        "repository_branch": "work/test-v6",
        "repository_dirty": False,
        "adapter_path": "scripts/data/measure_paths_fullband_causal_v6.py",
        "adapter_file_sha256": "d" * 64,
    }


@pytest.mark.parametrize(
    ("failure", "expected_stage", "optimizer_started"),
    [
        (V6ClockAdmissionError("low snr", stage="pre_snr", optimizer_started=False), "pre_snr", False),
        (ValueError("subband failed"), "post_clock_operator_analysis", True),
    ],
)
def test_offline_expected_failures_are_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    expected_stage: str,
    optimizer_started: bool,
) -> None:
    module = _load_script(f"v6_offline_{expected_stage}")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "_repository_execution_identity", _execution_identity)
    monkeypatch.setattr(module, "committed_plan_envelope_v6", lambda: {"signal_plan": {}})
    monkeypatch.setattr(post, "load_external_post_capture_receipt_v6", lambda **_k: _offline_admission())
    monkeypatch.setattr(delay_core, "validate_committed_v6_plan_and_derive_windows", lambda *_a, **_k: ({}, {}))
    monkeypatch.setattr(delay_core, "validate_duplex_telemetry_v6", lambda *_a, **_k: {})
    monkeypatch.setattr(delay_core, "analyze_committed_v6_live_delay", lambda **_k: (_ for _ in ()).throw(failure))
    observed = []
    monkeypatch.setattr(
        post,
        "publish_live_delay_failure_v6",
        lambda **kwargs: (
            observed.append(kwargs)
            or {"path": tmp_path / "failure.json", "file_sha256": "c" * 64}
        ),
    )
    assert module._offline_analyze(_args()) == 1
    assert observed[0]["failure_stage"] == expected_stage
    assert observed[0]["optimizer_started"] is optimizer_started


def test_offline_dirty_or_capture_commit_mismatch_blocks_before_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("v6_offline_identity_gate")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "_repository_execution_identity",
        lambda: (_ for _ in ()).throw(RuntimeError("dirty checkout")),
    )
    monkeypatch.setattr(
        post,
        "load_external_post_capture_receipt_v6",
        lambda **_k: calls.append("receipt"),
    )
    with pytest.raises(RuntimeError, match="dirty checkout"):
        module._offline_analyze(_args())
    assert calls == []

    monkeypatch.setattr(module, "_repository_execution_identity", _execution_identity)
    mismatched = _offline_admission()
    mismatched["raw"]["metadata"]["session"]["repository_commit"] = "f" * 40
    monkeypatch.setattr(
        post,
        "load_external_post_capture_receipt_v6",
        lambda **_k: (calls.append("receipt") or mismatched),
    )
    monkeypatch.setattr(
        delay_core,
        "analyze_committed_v6_live_delay",
        lambda **_k: calls.append("core"),
    )
    with pytest.raises(ValueError, match="checkout/adapter identity"):
        module._offline_analyze(_args())
    assert calls == ["receipt"]


def test_dry_run_is_signal_only_and_does_not_import_sounddevice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("v6_adapter_dry")
    plan, submitted = build_plan_v6()
    calls: list[str] = []
    real_import = module.importlib.import_module

    def no_backend(name: str):
        if name == "sounddevice":
            calls.append(name)
            raise AssertionError("dry-run이 PortAudio를 import했습니다")
        return real_import(name)

    monkeypatch.setattr(module.importlib, "import_module", no_backend)
    monkeypatch.setattr(module, "build_plan_v6", lambda **_k: (plan, submitted))
    monkeypatch.setattr(
        module,
        "exact_condition_audit_v6",
        lambda *_a, **_k: {"joint_fit_condition_number": 1.96},
    )
    assert module.main(["--dry-run"]) == 0
    assert calls == []
