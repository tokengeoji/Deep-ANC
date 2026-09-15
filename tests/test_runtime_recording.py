"""장치 없이 녹음 메타·원본 배열 호환·덮어쓰기 거부를 검증한다."""

import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from deep_anc.realtime import recording, run_realtime


@pytest.fixture
def recorded_runtime(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace())
    monkeypatch.setattr(run_realtime, "resolve_alsa_portaudio_device", lambda *_args: 0)
    monkeypatch.setattr(
        run_realtime, "build_engine",
        lambda _cfg: SimpleNamespace(digital_reference_lead_samples=0),
    )
    secondary = tmp_path / "secondary.npz"
    np.savez(
        secondary, fir=np.array([1.0, 0.3], dtype=np.float32),
        delay_samples=11, sample_rate=48_000,
        calibration_block_size=256, calibration_latency="low",
    )
    cfg = {
        "reference": "mic", "controller": "fxlms", "hop": 256,
        "digital_reference_lead_samples": 0,
        "hardware": {
            "audio": {
                "sample_rate": 48_000, "block_size": 256, "latency": "low",
                "input": {"card": "fake", "pcm": 0},
                "output": {"card": "fake", "pcm": 0},
            },
            "channels": {
                "error_mic": 0, "reference_mic": 1, "noise_out": 0, "cancel_out": 1,
            },
            "dc_blocker_r": 0.99,
        },
        "duct": {"secondary_path": {"npz": str(secondary), "handoff_extra_samples": 256}},
        "noise": {"enabled": False, "type": "tone", "amplitude": 0.03},
        "safety": {"control_limit": 0.125},
    }
    runtime = run_realtime.RealtimeANC(cfg, record_seconds=513 / 48_000)
    for i, key in enumerate(recording.SIGNAL_KEYS):
        runtime.rec[key][:300] = np.linspace(0.0, 0.05 * (i + 1), 300, dtype=np.float32)
    runtime.rec_pos = 300
    return runtime, secondary


def test_metadata_matches_measured_constructor_and_runtime_health(recorded_runtime):
    runtime, secondary = recorded_runtime
    runtime.xruns = 2
    runtime.in_ring.drops = 256
    runtime.out_ring.drops = 512
    runtime.out_ring.underruns = 3
    runtime.state.fatal_error = RuntimeError("가짜 오류")
    payload = recording.build_recording_payload(runtime)
    meta = json.loads(str(payload["recording_meta_json"]))
    assert meta == {
        "reference": "mic", "controller": "fxlms", "sample_rate": 48_000,
        "block_size": 256, "hop": 256, "latency": "low",
        "digital_reference_lead_samples": 0, "handoff_extra_samples": 256,
        "secondary_sha256": hashlib.sha256(secondary.read_bytes()).hexdigest(),
        "secondary_delay_samples": 11, "secondary_fir_length": 2,
        "control_limit": 0.125, "dc_blocker_r": 0.99,
        "channels": {"error_mic": 0, "reference_mic": 1, "noise_out": 0, "cancel_out": 1},
        "record_requested_samples": 513, "recorded_samples": 300,
        "runtime_health": {
            "xrun_count": 2, "input_ring_drops": 256, "output_ring_drops": 512,
            "output_ring_underruns": 3, "fatal_error": True,
            "scope": "whole_runtime_including_startup_stop",
        },
    }
    assert payload["recording_schema_version"].shape == ()
    assert payload["recording_schema_version"].item() == 1
    assert payload["recording_meta_json"].shape == ()
    assert payload["recording_meta_json"].dtype.kind == "U"


def test_provenance_stays_at_constructor_time_after_secondary_and_config_change(recorded_runtime):
    runtime, secondary = recorded_runtime
    old_hash = hashlib.sha256(secondary.read_bytes()).hexdigest()
    secondary.write_bytes("실험 시작 후 변경된 가짜 측정 파일".encode("utf-8"))
    runtime.cfg["controller"] = "hybrid"
    runtime.cfg["hardware"]["dc_blocker_r"] = 0.5
    runtime.cfg["hardware"]["channels"]["error_mic"] = 1
    runtime.safety.control_limit = 0.9
    meta = json.loads(str(recording.build_recording_payload(runtime)["recording_meta_json"]))
    assert meta["secondary_sha256"] == old_hash
    assert meta["secondary_delay_samples"] == 11
    assert meta["controller"] == "fxlms"
    assert meta["dc_blocker_r"] == 0.99
    assert meta["channels"]["error_mic"] == 0
    assert meta["control_limit"] == 0.125


def test_session_data_stays_trimmed_float32_arrays_and_npz_preserves_them(recorded_runtime, tmp_path):
    runtime, _ = recorded_runtime
    data = runtime.session_data()
    assert set(data) == set(recording.SIGNAL_KEYS)
    assert all(array.shape == (300,) and array.dtype == np.float32 for array in data.values())
    data["err"][0] = 0.25
    assert runtime.rec["err"][0] == 0.0  # 기존 copy 계약
    data = runtime.session_data()
    payload = recording.build_recording_payload(runtime, data)
    target = recording.save_recording(tmp_path / "session.wav", payload)
    assert target == tmp_path / "session.npz"
    with np.load(target, allow_pickle=False) as stored:
        assert set(stored.files) == {*recording.SIGNAL_KEYS, "fs", "recording_schema_version", "recording_meta_json"}
        assert stored["fs"].shape == () and stored["fs"].item() == 48_000
        for key in recording.SIGNAL_KEYS:
            assert stored[key].dtype == np.float32
            np.testing.assert_array_equal(stored[key], data[key])
        assert not json.loads(str(stored["recording_meta_json"]))["runtime_health"]["fatal_error"]


def test_no_recording_does_not_read_secondary_hash(recorded_runtime, monkeypatch):
    runtime, _ = recorded_runtime
    def forbidden(*_args):
        raise AssertionError("녹음 비활성 시 해시 캡처 금지")
    monkeypatch.setattr(run_realtime, "capture_recording_provenance", forbidden)
    no_record = run_realtime.RealtimeANC(runtime.cfg, record_seconds=0)
    assert no_record._recording_provenance is None
    assert no_record.session_data() == {}
    with pytest.raises(ValueError, match="provenance"):
        recording.build_recording_payload(no_record)


@pytest.mark.parametrize("recorded_samples", [0, 300, 513])
def test_recording_length_uses_actual_samples_without_hop_rounding(recorded_runtime, recorded_samples):
    runtime, _ = recorded_runtime
    runtime.rec_pos = recorded_samples
    payload = recording.build_recording_payload(runtime)
    meta = json.loads(str(payload["recording_meta_json"]))
    assert meta["recorded_samples"] == recorded_samples
    assert meta["record_requested_samples"] == 513
    assert all(payload[key].shape == (recorded_samples,) for key in recording.SIGNAL_KEYS)


def test_fake_cli_saves_metadata_after_stop_without_opening_devices(recorded_runtime, tmp_path, monkeypatch):
    runtime, _ = recorded_runtime
    def create(_cfg, record_seconds):
        assert record_seconds == 1.0
        return runtime
    def stop():
        runtime.xruns = 7  # 종료까지의 누적 상태가 저장되어야 한다.
    monkeypatch.setattr(run_realtime, "RealtimeANC", create)
    monkeypatch.setattr(runtime, "start", runtime.state.quit_event.set)
    monkeypatch.setattr(runtime, "stop", stop)
    class Keyboard:
        def __init__(self, _state):
            pass

        @staticmethod
        def help_text():
            return "가짜 키보드"

        def start(self):
            pass

        def stop(self):
            pass
    monkeypatch.setattr(run_realtime, "KeyboardController", Keyboard)
    target = tmp_path / "from_cli.wav"
    assert run_realtime.run_cli(runtime.cfg, 1.0, target) == 0
    with np.load(target.with_suffix(".npz"), allow_pickle=False) as data:
        assert data["recording_schema_version"].item() == 1
        meta = json.loads(str(data["recording_meta_json"]))
        assert meta["recorded_samples"] == 300
        assert meta["runtime_health"]["xrun_count"] == 7


@pytest.mark.parametrize("entry_kind", ["file", "directory", "broken_symlink"])
def test_prepare_refuses_existing_normalized_target(tmp_path, entry_kind):
    target = tmp_path / "session.npz"
    if entry_kind == "file":
        target.write_bytes(b"keep")
    elif entry_kind == "directory":
        target.mkdir()
    else:
        target.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        recording.prepare_recording_path(tmp_path / "session.wav")


def test_run_cli_rejects_existing_file_before_runtime_constructor(monkeypatch, tmp_path):
    target = tmp_path / "session.npz"
    target.write_bytes(b"keep")
    def forbidden(*_args, **_kwargs):
        raise AssertionError("기존 녹음이 있으면 장치 초기화 금지")
    monkeypatch.setattr(run_realtime, "RealtimeANC", forbidden)
    with pytest.raises(FileExistsError):
        run_realtime.run_cli({}, 1.0, str(tmp_path / "session.wav"))
    assert target.read_bytes() == b"keep"


def test_main_rejects_existing_file_before_input_preflight(monkeypatch, tmp_path):
    target = tmp_path / "session.npz"
    target.write_bytes(b"keep")
    monkeypatch.setattr(sys, "argv", ["run_realtime", "--record", str(target), "--run-seconds", "1"])
    monkeypatch.setattr(run_realtime, "load_runtime_config", lambda *_args: {})
    def forbidden(*_args, **_kwargs):
        raise AssertionError("기존 녹음이 있으면 입력 장치 사전점검도 금지")
    monkeypatch.setattr(run_realtime, "input_preflight", forbidden)
    with pytest.raises(SystemExit) as error:
        run_realtime.main()
    assert error.value.code == 2
    assert target.read_bytes() == b"keep"


@pytest.mark.parametrize("seconds", [0.0, -1.0, float("nan"), float("inf")])
def test_run_cli_rejects_invalid_recording_duration_before_constructor(monkeypatch, tmp_path, seconds):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("녹음 시간이 무효이면 장치 초기화 금지")
    monkeypatch.setattr(run_realtime, "RealtimeANC", forbidden)
    with pytest.raises(ValueError, match="run_seconds"):
        run_realtime.run_cli({}, seconds, str(tmp_path / "session.npz"))


def test_save_refuses_file_created_after_initial_check(recorded_runtime, tmp_path):
    runtime, _ = recorded_runtime
    target = recording.prepare_recording_path(tmp_path / "session")
    target.write_bytes(b"other recording")
    with pytest.raises(FileExistsError):
        recording.save_recording(target, recording.build_recording_payload(runtime))
    assert target.read_bytes() == b"other recording"


def test_exclusive_open_refuses_racing_writer(recorded_runtime, tmp_path, monkeypatch):
    runtime, _ = recorded_runtime
    original_prepare = recording.prepare_recording_path
    def race(path):
        target = original_prepare(path)
        target.write_bytes(b"racing recording")
        return target
    monkeypatch.setattr(recording, "prepare_recording_path", race)
    target = tmp_path / "session.npz"
    with pytest.raises(FileExistsError):
        recording.save_recording(target, recording.build_recording_payload(runtime))
    assert target.read_bytes() == b"racing recording"


@pytest.mark.parametrize("fault", ["length", "nan", "gain", "fs", "meta_count", "object"])
def test_save_validates_payload_before_creating_directories(recorded_runtime, tmp_path, fault):
    runtime, _ = recorded_runtime
    payload = recording.build_recording_payload(runtime)
    if fault == "length":
        payload["err"] = payload["err"][:-1]
    elif fault == "nan":
        payload["err"][0] = np.nan
    elif fault == "gain":
        payload["anc_gain"][0] = 2.0
    elif fault == "fs":
        payload["fs"] = np.asarray(48_001)
    elif fault == "object":
        payload["err"] = payload["err"].astype(object)
    else:
        meta = json.loads(str(payload["recording_meta_json"]))
        meta["recorded_samples"] = 301
        payload["recording_meta_json"] = np.asarray(json.dumps(meta))
    target = tmp_path / "not_created" / "session.npz"
    with pytest.raises(ValueError):
        recording.save_recording(target, payload)
    assert not target.parent.exists()


def test_failed_write_keeps_partial_file_and_refuses_reuse(recorded_runtime, tmp_path, monkeypatch):
    runtime, _ = recorded_runtime
    payload = recording.build_recording_payload(runtime)
    def fail(handle, **_payload):
        handle.write(b"partial")
        raise OSError("가짜 저장 실패")
    monkeypatch.setattr(recording.np, "savez_compressed", fail)
    target = tmp_path / "session.npz"
    with pytest.raises(OSError, match="가짜 저장 실패"):
        recording.save_recording(target, payload)
    assert target.read_bytes() == b"partial"
    with pytest.raises(FileExistsError):
        recording.save_recording(target, payload)


def test_helper_import_does_not_load_audio_or_torch():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import deep_anc.realtime.recording; "
         "assert 'sounddevice' not in sys.modules; assert 'torch' not in sys.modules"],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mismatch", [None, "secondary_sha256", "control_limit"])
def test_saved_runtime_payload_binds_to_acoustic_analyzer(tmp_path, monkeypatch, mismatch):
    from deep_anc.config import load_runtime_config
    from deep_anc.eval.acoustic_session import analyze_acoustic_session

    # 8 kHz는 이 무출력 합성 회귀 전용. Jetson의 실제 오디오 조건을 바꾸지 않는다.
    secondary = tmp_path / "synthetic_secondary.npz"
    secondary_data = {
        "fir": np.array([1.0, 0.1], dtype=np.float32), "delay_samples": 11,
        "sample_rate": 8000, "calibration_block_size": 256, "calibration_latency": "low",
        "consistency_band_hz": [100.0, 3000.0], "coherence_median": 0.95,
        "output_channel": "cancel",
    }
    np.savez(secondary, **secondary_data)
    cfg = load_runtime_config("configs/runtime_acoustic.yaml", [
        "hardware.audio.sample_rate=8000", f"duct.secondary_path.npz={secondary}",
    ])
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace())
    monkeypatch.setattr(run_realtime, "resolve_alsa_portaudio_device", lambda *_args: 0)
    monkeypatch.setattr(
        run_realtime, "build_engine",
        lambda _cfg: SimpleNamespace(digital_reference_lead_samples=0),
    )
    runtime = run_realtime.RealtimeANC(cfg, record_seconds=6.0)
    time = np.arange(runtime.record_len) / runtime.fs
    signal = 0.02 * (np.sin(2 * np.pi * 300 * time) + np.sin(2 * np.pi * 1500 * time))
    on = slice(2 * runtime.fs, 4 * runtime.fs)
    runtime.rec["err"][:] = signal
    runtime.rec["err"][on] *= 0.5
    runtime.rec["ref"][:] = signal
    runtime.rec["control"][on] = 0.1 * signal[on]
    runtime.rec["anc_gain"][on] = 1.0
    runtime.rec_pos = runtime.record_len
    saved = recording.save_recording(tmp_path / "synthetic_runtime", recording.build_recording_payload(runtime))
    original = saved.read_bytes()
    if mismatch == "secondary_sha256":
        secondary_data["fir"] = np.array([1.0, 0.2], dtype=np.float32)
        np.savez(secondary, **secondary_data)
    elif mismatch == "control_limit":
        cfg["safety"]["control_limit"] *= 0.5
    kwargs = {"window_seconds": 0.1, "startup_guard_seconds": 0.05,
              "on_warmup_seconds": 0.1, "edge_guard_seconds": 0.05}
    if mismatch:
        with pytest.raises(ValueError, match=mismatch):
            analyze_acoustic_session(saved, cfg, "synthetic_two_tone", **kwargs)
    else:
        report = analyze_acoustic_session(saved, cfg, "synthetic_two_tone", **kwargs)
        assert report["provenance"]["recording_context_verified"]
        assert report["all_cycles_complete"] and report["comparison_available"]
        assert not report["performance_claim_allowed"]
        full = next(row for row in report["metrics"] if row["band"] == "full")
        assert full["observed_err_reduction_db"] == pytest.approx(20 * np.log10(2), abs=1e-5)
    assert saved.read_bytes() == original
