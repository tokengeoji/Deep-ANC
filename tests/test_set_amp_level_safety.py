"""앰프 미터 실기 진입·종료의 fail-closed 회귀 테스트."""

from types import SimpleNamespace
import hashlib
import json
import signal
from contextlib import contextmanager

import numpy as np
import pytest

from deep_anc import audio_io
from deep_anc.dsp.measurement_level import ALSA_PHYSICAL_FINGERPRINT_SCHEMA
from scripts.data import set_amp_level as meter


def _portaudio_available() -> bool:
    """PortAudio가 없는 Elice 학습 노드에서는 실제 stream 테스트를 건너뛴다."""

    try:
        import sounddevice  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


requires_portaudio = pytest.mark.skipif(
    not _portaudio_available(), reason="PortAudio가 없는 학습 노드에서는 실기 meter를 실행하지 않음"
)


def _physical_fingerprint() -> dict:
    def endpoint(card_id: str, pcm: int, stream: str, realpath: str) -> dict:
        return {
            "configured_card_id": card_id,
            "proc_card_id": card_id,
            "pcm_device": pcm,
            "pcm_stream": stream,
            "pcm_info": {
                "device": str(pcm),
                "stream": stream,
                "id": f"{card_id} PCM",
                "name": f"{card_id} codec",
                "subname": "subdevice #0",
                "class": "0",
                "subclass": "0",
                "subdevices_count": "1",
            },
            "sys_device_realpath": realpath,
            "sys_device_uevent": {"DRIVER": f"driver-{card_id.lower()}"},
            "stable_attributes": [],
        }

    payload = {
        "schema": ALSA_PHYSICAL_FINGERPRINT_SCHEMA,
        "input": endpoint("APE", 1, "CAPTURE", "devices/platform/sound"),
        "output": endpoint("Audio", 0, "PLAYBACK", "devices/usb/1-1:1.0"),
    }
    payload["output"]["stable_attributes"] = [
        {
            "sys_relative_path": "devices/usb/1-1",
            "values": {"serial": "dac-test-001"},
        }
    ]
    unsigned = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["sha256"] = hashlib.sha256(unsigned).hexdigest()
    return payload


def _telemetry(**updates):
    base = {
        "completed": True,
        "interrupted": False,
        "output_frames": 960_000,
        "xrun_count": 0,
        "unexpected_status_count": 0,
        "callback_error": None,
        "stream_abort_error": None,
        "stream_close_error": None,
        "output_stop_confirmed": True,
        "meter_drop_count": 0,
    }
    base.update(updates)
    return base


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"interrupted": True}, "operator_interrupt"),
        ({"xrun_count": 1}, "xrun_1"),
        ({"unexpected_status_count": 1}, "unexpected_callback_status_1"),
        ({"completed": False, "output_frames": 10}, "capture_incomplete"),
        ({"stream_close_error": "OSError: close", "output_stop_confirmed": False},
         "stream_close_error"),
    ],
)
def test_meter_capture_faults_are_never_accepted(updates, expected):
    reasons = meter.meter_capture_invalid_reasons(
        _telemetry(**updates),
        [-50.1] * 8,
        expected_frames=960_000,
    )

    assert any(expected in reason for reason in reasons)


def test_meter_nonfinite_and_missing_levels_fail():
    assert "no_meter_levels" in meter.meter_capture_invalid_reasons(
        _telemetry(), [], expected_frames=960_000
    )
    assert "meter_nonfinite" in meter.meter_capture_invalid_reasons(
        _telemetry(), [-50.1, float("nan")], expected_frames=960_000
    )


@pytest.mark.parametrize("bootstrap", [False, True])
def test_meter_followup_command_always_requires_fresh_user_presence(bootstrap):
    command = meter.strict_followup_command(
        "results/meter/raw.npz", capture_id="abcdef012345", bootstrap=bootstrap
    )

    assert "--confirm-user-present" in command
    assert "--confirm-volume-minimum" in command
    assert "--confirm-routing-and-geometry" in command
    assert "--confirm-same-amplifier-setting" in command
    assert ("--bootstrap-level-evidence" in command) is bootstrap


def test_meter_close_failure_prints_physical_disconnect_warning(capsys):
    class FakeSD:
        class CallbackStop(Exception):
            pass

        class CallbackAbort(Exception):
            pass

        class Stream:
            def __init__(self, *, callback, **_kwargs):
                self.callback = callback

            def start(self):
                indata = np.arange(8, dtype=np.int32).reshape(4, 2)
                outdata = np.zeros((4, 2), dtype=np.int16)
                with pytest.raises(FakeSD.CallbackStop):
                    self.callback(indata, outdata, 4, None, None)

            def abort(self):
                return None

            def close(self):
                raise OSError("injected close failure")

    levels, telemetry = meter.capture_meter_stream(
        FakeSD,
        noise=np.zeros(4, dtype=np.float32),
        fs=48_000,
        in_dev=1,
        out_dev=2,
        err_ch=0,
        noise_out_ch=0,
    )

    assert levels == []
    assert telemetry["output_stop_confirmed"] is False
    assert "injected close failure" in telemetry["stream_close_error"]
    assert "정지 확인 불가" in capsys.readouterr().out


def test_meter_raw_capture_preserves_submitted_pcm_and_int_input():
    class FakeSD:
        class CallbackStop(Exception):
            pass

        class CallbackAbort(Exception):
            pass

        class Stream:
            def __init__(self, *, callback, **_kwargs):
                self.callback = callback
                self.active = True

            def start(self):
                indata = np.arange(16, dtype=np.int32).reshape(8, 2)
                outdata = np.zeros((8, 2), dtype=np.int16)
                with pytest.raises(FakeSD.CallbackStop):
                    self.callback(indata, outdata, 8, None, None)

            def abort(self):
                self.active = False

            def close(self):
                return None

    noise = np.linspace(-0.003, 0.003, 8, dtype=np.float32)
    levels, telemetry, submitted, input_raw = meter.capture_meter_stream(
        FakeSD,
        noise=noise,
        fs=48_000,
        in_dev=1,
        out_dev=2,
        err_ch=0,
        noise_out_ch=0,
        include_raw=True,
    )

    assert levels == []
    assert telemetry["completed"] is True
    assert submitted.dtype == np.int16
    assert submitted.shape == (8, 2)
    assert np.count_nonzero(submitted[:, 1]) == 0
    assert input_raw.dtype == np.int32
    assert np.array_equal(input_raw, np.arange(16, dtype=np.int32).reshape(8, 2))


def test_meter_sighup_aborts_closes_and_announces_disconnect(capsys):
    signum = getattr(signal, "SIGHUP", signal.SIGTERM)
    events = []

    class FakeSD:
        class CallbackStop(Exception):
            pass

        class CallbackAbort(Exception):
            pass

        class Stream:
            def __init__(self, **_kwargs):
                self.active = True

            def start(self):
                events.append("start")
                handler = signal.getsignal(signum)
                handler(signum, None)

            def abort(self):
                events.append("abort")
                self.active = False

            def close(self):
                events.append("close")

    levels, telemetry = meter.capture_meter_stream(
        FakeSD,
        noise=np.zeros(8, dtype=np.float32),
        fs=48_000,
        in_dev=1,
        out_dev=2,
        err_ch=0,
        noise_out_ch=0,
    )

    assert levels == []
    assert telemetry["interrupted"] is True
    assert telemetry["termination_signal"] == int(signum)
    assert telemetry["output_stop_confirmed"] is True
    assert events == ["start", "abort", "close"]
    assert "스피커 출력 종료" in capsys.readouterr().out


def test_official_meter_duration_cannot_be_overridden(monkeypatch, capsys):
    monkeypatch.setattr(
        meter,
        "measure",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("invalid duration에서 measure를 호출하면 안 됩니다")
        ),
    )

    assert meter.main(["--confirm-speaker", "--seconds", "19"]) == 2
    assert "정확히 20초" in capsys.readouterr().err


@requires_portaudio
def test_live_meter_stops_before_hardware_when_paired_raw_evidence_is_missing(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        meter,
        "load_measurement_level_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("paired raw missing")
        ),
    )
    monkeypatch.setattr(
        meter,
        "load_yaml",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("evidence 전에 hardware를 읽으면 안 됩니다")
        ),
    )
    args = SimpleNamespace(
        seconds=20.0,
        level_evidence="missing.json",
        hardware="configs/hardware_jetson.yaml",
    )

    assert meter.measure(args) == 2
    assert "paired raw missing" in capsys.readouterr().err


def test_self_test_needs_no_live_evidence(monkeypatch):
    monkeypatch.setattr(
        meter,
        "load_measurement_level_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("self-test는 live evidence를 읽지 않습니다")
        ),
    )

    assert meter.main(["--self-test"]) == 0


def test_bootstrap_meter_requires_explicit_user_and_volume_confirmations(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        meter,
        "measure",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("confirmation 누락 상태에서 measure를 호출하면 안 됩니다")
        ),
    )

    assert meter.main(["--bootstrap-level-evidence", "--confirm-speaker"]) == 2
    message = capsys.readouterr().err
    assert "--confirm-user-present" in message
    assert "--confirm-volume-minimum" in message


def test_normal_live_meter_also_requires_user_and_volume_confirmations(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        meter,
        "measure",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("normal live confirmation 누락 상태에서 measure를 호출하면 안 됩니다")
        ),
    )

    assert meter.main(["--confirm-speaker"]) == 2
    message = capsys.readouterr().err
    assert "모든 live meter" in message
    assert "--confirm-user-present" in message
    assert "--confirm-volume-minimum" in message


def test_explicit_bootstrap_mode_reaches_measure_without_existing_evidence(
    monkeypatch,
):
    observed = {}

    def fake_measure(args):
        observed["bootstrap"] = args.bootstrap_level_evidence
        observed["user"] = args.confirm_user_present
        observed["volume"] = args.confirm_volume_minimum
        return 0

    monkeypatch.setattr(meter, "measure", fake_measure)

    assert meter.main(
        [
            "--bootstrap-level-evidence",
            "--confirm-speaker",
            "--confirm-user-present",
            "--confirm-volume-minimum",
        ]
    ) == 0
    assert observed == {"bootstrap": True, "user": True, "volume": True}


@requires_portaudio
def test_normal_meter_uses_permanent_evidence_but_always_emits_fresh_raw(
    tmp_path, monkeypatch, capsys
):
    events = []
    hardware_config = {
        "audio": {
            "sample_rate": 48_000,
            "block_size": 256,
            "latency": "low",
            "input": {"card": "APE", "pcm": 1, "channels": 2},
            "output": {"card": "Audio", "pcm": 0, "channels": 2},
        },
        "channels": {
            "error_mic": 0,
            "reference_mic": 1,
            "noise_out": 0,
            "cancel_out": 1,
        },
    }
    physical_fingerprint = _physical_fingerprint()
    identity = meter.measurement_hardware_identity(
        hardware_config, physical_fingerprint=physical_fingerprint
    )
    monkeypatch.setattr(
        meter,
        "load_measurement_level_evidence",
        lambda *_a, **_k: {
            "hardware_identity": identity,
            "_evidence_path": "assets/measured/measurement_level_evidence.json",
            "_evidence_sha256": "e" * 64,
        },
    )
    monkeypatch.setattr(meter, "load_yaml", lambda _path: hardware_config)
    monkeypatch.setattr(
        meter,
        "collect_alsa_physical_fingerprint",
        lambda _config: (
            events.append("physical_fingerprint")
            or json.loads(json.dumps(physical_fingerprint))
        ),
    )
    monkeypatch.setattr(
        audio_io,
        "resolve_alsa_portaudio_device",
        lambda *_a, **_k: 1 if _a[2] == "input" else 2,
    )
    monkeypatch.setattr(
        audio_io,
        "assert_measurement_preconditions",
        lambda *_a, **_k: events.append("input_preflight") or [0.0, 0.0],
    )
    monkeypatch.setattr(
        meter,
        "assert_live_pcm_clock_preconditions",
        lambda *_a, **_k: events.append("immediate_pcm_clock"),
    )
    session = tmp_path / "meter-session"
    session.mkdir()
    monkeypatch.setattr(meter, "_bootstrap_meter_session", lambda _args: (session, "cap"))

    @contextmanager
    def fake_lock(*_args, **_kwargs):
        yield {"path": "results/.lock", "pid": 1, "uid": 1000, "purpose": "test"}

    monkeypatch.setattr(meter, "repository_audio_lock", fake_lock)
    frames = 960_000
    monkeypatch.setattr(meter, "probe_signal", lambda _seconds: np.zeros(frames, np.float32))
    observed = {}

    def fake_capture(*_args, **kwargs):
        kwargs["pre_open_check"]()
        events.append("output_stream")
        observed["include_raw"] = kwargs["include_raw"]
        return (
            [-50.1] * 8,
            _telemetry(),
            np.zeros((frames, 2), np.int16),
            np.zeros((frames, 2), np.int32),
        )

    monkeypatch.setattr(meter, "capture_meter_stream", fake_capture)

    def fake_write(_path, **kwargs):
        observed["metadata"] = kwargs["metadata"]
        return {
            "raw": meter.REPO_ROOT / "results/fresh_normal_meter.npz",
            "receipt": meter.REPO_ROOT / "results/fresh_normal_meter.receipt.json",
            "sha256": "a" * 64,
        }

    monkeypatch.setattr(meter, "write_bootstrap_meter_raw_atomic", fake_write)
    args = SimpleNamespace(
        seconds=20.0,
        bootstrap_level_evidence=False,
        level_evidence="assets/measured/measurement_level_evidence.json",
        hardware="configs/hardware_jetson.yaml",
        diagnostics_root="results/calibration_interleaved/level_bootstrap",
        confirm_speaker=True,
        confirm_user_present=True,
        confirm_volume_minimum=True,
    )

    assert meter.measure(args) == 0
    assert observed["include_raw"] is True
    assert observed["metadata"]["calibration_evidence"]["mode"] == "verified_existing"
    assert events == [
        "physical_fingerprint",
        "input_preflight",
        "physical_fingerprint",
        "immediate_pcm_clock",
        "output_stream",
    ]
    output = capsys.readouterr().out
    assert "무출력 입력 preflight 1.5초" in output
    assert "fresh meter immutable raw 저장" in output
    assert "--meter-raw results/fresh_normal_meter.npz" in output
    assert "--confirm-same-amplifier-setting" in output
    assert "--confirm-user-present" in output
    assert "--primary-out assets/measured/primary_path_il_strict_cap.npz" in output
    assert "--secondary-out assets/measured/secondary_path_il_strict_cap.npz" in output
