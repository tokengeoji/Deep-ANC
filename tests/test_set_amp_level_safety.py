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


def test_default_strict_followup_command_remains_byte_exact():
    assert meter.strict_followup_command(
        "results/meter/raw.npz",
        capture_id="abcdef012345",
        bootstrap=False,
    ) == (
        "  .venv/bin/python scripts/data/measure_paths_interleaved.py "
        "--meter-raw results/meter/raw.npz "
        "--confirm-same-amplifier-setting --confirm-user-present "
        "--confirm-volume-minimum --confirm-routing-and-geometry "
        "--primary-out assets/measured/primary_path_il_strict_abcdef01.npz "
        "--secondary-out assets/measured/secondary_path_il_strict_abcdef01.npz"
    )


def test_broadband_followup_command_carries_exact_inputs_and_all_confirmations():
    command = meter.broadband_followup_command(
        "results/meter raw/meter_raw.npz",
        plan="results/plans/live authority.json",
        raw_session_dir="results/calibration_interleaved/broadband/session-01",
        level_evidence="assets/measured/measurement_level_evidence.json",
        hardware="configs/custom hardware.yaml",
    )

    assert "scripts/data/measure_paths_broadband_interleaved.py" in command
    assert "--execute-live" in command
    assert "'results/meter raw/meter_raw.npz'" in command
    assert "'results/plans/live authority.json'" in command
    assert "--hardware 'configs/custom hardware.yaml'" in command
    assert (
        "--raw-session-dir results/calibration_interleaved/broadband/session-01"
        in command
    )
    assert (
        "--level-evidence assets/measured/measurement_level_evidence.json" in command
    )
    for confirmation in (
        "--confirm-speaker",
        "--confirm-user-present",
        "--confirm-volume-minimum",
        "--confirm-routing-and-geometry",
        "--confirm-same-amplifier-setting",
    ):
        assert confirmation in command


def test_fullband_v5_contract_and_command_are_exact_capture_only() -> None:
    args = SimpleNamespace(
        followup_mode="fullband-v5",
        broadband_plan=None,
        broadband_raw_session_dir=None,
        bootstrap_level_evidence=False,
        hardware="configs/hardware_jetson.yaml",
        level_evidence="assets/measured/measurement_level_evidence.json",
    )
    contract = meter.validate_followup_contract(args)
    assert contract["schema"] == "fullband_v5_meter_followup_v1"
    assert contract["status"] == "blocked_until_v5_live_adapter_implementation"
    assert contract["capture_only"] is True
    assert contract["plan_live_capture_enabled"] is False
    assert contract["sealed_raw"] == {
        "path": "results/fullband_causal_v5/raw_capture.npz",
        "fresh": True,
    }
    command = meter.fullband_v5_followup_command(
        "results/meter/raw.npz", contract=contract
    )
    assert "measure_paths_fullband_causal_v5.py --execute-live" in command
    for fragment in (
        "--plan-envelope assets/contracts/fullband_causal_v5_signal_plan.json",
        "--live-authority assets/contracts/fullband_causal_v5_live_capture_authority.json",
        "--meter-raw results/meter/raw.npz",
        "--level-evidence assets/measured/measurement_level_evidence.json",
        "--hardware configs/hardware_jetson.yaml",
        "--raw-target results/fullband_causal_v5/raw_capture.npz",
        "--confirm-speaker",
        "--confirm-user-present",
        "--confirm-volume-minimum",
        "--confirm-routing-and-geometry",
        "--confirm-same-amplifier-setting",
    ):
        assert fragment in command
    assert "measure_paths_interleaved.py" not in command
    assert "measure_paths_broadband_interleaved.py" not in command
    assert "--primary-out" not in command


def test_fullband_v5_rejects_old_broadband_arguments() -> None:
    args = SimpleNamespace(
        followup_mode="fullband-v5",
        broadband_plan="old-v4.json",
        broadband_raw_session_dir=None,
        bootstrap_level_evidence=False,
        hardware="configs/hardware_jetson.yaml",
        level_evidence="assets/measured/measurement_level_evidence.json",
    )
    with pytest.raises(ValueError, match="old --broadband"):
        meter.validate_followup_contract(args)


def test_fullband_v5_authority_tamper_stops_before_sounddevice(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        meter,
        "_validate_fullband_v5_followup_contract",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("authority file SHA tamper")),
    )
    monkeypatch.setattr(
        meter,
        "measure",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("tamper 뒤 sounddevice/meter에 도달하면 안 됩니다")
        ),
    )
    result = meter.main(
        [
            "--followup-mode",
            "fullband-v5",
            "--confirm-speaker",
            "--confirm-user-present",
            "--confirm-volume-minimum",
            "--confirm-routing-and-geometry",
            "--confirm-same-amplifier-setting",
        ]
    )
    assert result == 2
    assert "authority file SHA tamper" in capsys.readouterr().err


def test_fullband_v5_requires_all_five_confirmations_before_meter(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        meter,
        "measure",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("5 confirmations 전에 meter를 열면 안 됩니다")
        ),
    )
    result = meter.main(
        [
            "--followup-mode",
            "fullband-v5",
            "--confirm-speaker",
            "--confirm-user-present",
            "--confirm-volume-minimum",
        ]
    )
    assert result == 2
    assert "5개 confirmation" in capsys.readouterr().err


def test_fullband_v5_meter_signal_duration_peak_and_channel_role() -> None:
    signal = meter.probe_signal(20.0)
    assert signal.shape == (960_000,)
    assert np.max(np.abs(signal)) <= 0.003 + 1.0e-8
    output = np.zeros((len(signal), 2), dtype=np.float32)
    output[:, 0] = signal
    assert np.any(output[:, 0] != 0.0)
    assert np.all(output[:, 1] == 0.0)


def test_broadband_followup_missing_plan_stops_before_meter(monkeypatch, capsys):
    monkeypatch.setattr(
        meter,
        "measure",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("followup preflight 실패 뒤 meter를 열면 안 됩니다")
        ),
    )

    assert meter.main(
        [
            "--followup-mode",
            "broadband",
            "--broadband-raw-session-dir",
            "results/calibration_interleaved/broadband/fresh-session",
            "--confirm-speaker",
            "--confirm-user-present",
            "--confirm-volume-minimum",
        ]
    ) == 2
    assert "--broadband-plan" in capsys.readouterr().err


def test_broadband_followup_rejects_target_outside_results_before_meter(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        meter,
        "measure",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("부적절한 target으로 meter를 열면 안 됩니다")
        ),
    )

    assert meter.main(
        [
            "--followup-mode",
            "broadband",
            "--broadband-plan",
            "results/missing-plan.json",
            "--broadband-raw-session-dir",
            "assets/measured/not-a-raw-session",
            "--confirm-speaker",
            "--confirm-user-present",
            "--confirm-volume-minimum",
        ]
    ) == 2
    assert "results/ 아래" in capsys.readouterr().err


def test_broadband_followup_rejects_existing_target_before_plan_or_meter(
    tmp_path, monkeypatch, capsys
):
    from scripts.data import measure_paths_broadband_interleaved as broadband

    existing = tmp_path / "results" / "existing-session"
    existing.mkdir(parents=True)
    monkeypatch.setattr(meter, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(broadband, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        meter,
        "measure",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("기존 target으로 meter를 열면 안 됩니다")
        ),
    )

    assert meter.main(
        [
            "--followup-mode",
            "broadband",
            "--broadband-plan",
            "results/missing-plan.json",
            "--broadband-raw-session-dir",
            "results/existing-session",
            "--confirm-speaker",
            "--confirm-user-present",
            "--confirm-volume-minimum",
        ]
    ) == 2
    assert "덮어쓰지 않습니다" in capsys.readouterr().err


def test_broadband_followup_rejects_bootstrap_pairing_before_meter(monkeypatch, capsys):
    monkeypatch.setattr(
        meter,
        "measure",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("bootstrap+broadband에서 meter를 열면 안 됩니다")
        ),
    )

    assert meter.main(
        [
            "--bootstrap-level-evidence",
            "--followup-mode",
            "broadband",
            "--broadband-plan",
            "results/missing-plan.json",
            "--broadband-raw-session-dir",
            "results/calibration_interleaved/broadband/fresh-session",
            "--confirm-speaker",
            "--confirm-user-present",
            "--confirm-volume-minimum",
        ]
    ) == 2
    assert "--bootstrap-level-evidence" in capsys.readouterr().err


def test_broadband_followup_revalidation_detects_plan_binding_change(monkeypatch):
    contract = {
        "schema": "broadband_meter_followup_v1",
        "mode": "broadband",
        "plan": {
            "path": "results/authority.json",
            "file_sha256": "1" * 64,
            "payload_sha256": "2" * 64,
            "pcm_sha256": "3" * 64,
        },
        "raw_session_dir": "results/fresh-session",
        "hardware": {"path": "configs/hardware.yaml", "sha256": "4" * 64},
        "level_evidence": {"path": "assets/level.json", "sha256": "5" * 64},
    }
    changed = {
        **contract,
        "plan": {**contract["plan"], "file_sha256": "f" * 64},
    }
    monkeypatch.setattr(
        meter,
        "_validate_broadband_followup_contract",
        lambda **kwargs: changed,
    )

    with pytest.raises(ValueError, match="최초 preflight 이후 변경"):
        meter.revalidate_followup_contract(contract)


def test_broadband_followup_revalidation_detects_target_claim(monkeypatch):
    contract = {
        "schema": "broadband_meter_followup_v1",
        "mode": "broadband",
        "plan": {
            "path": "results/authority.json",
            "file_sha256": "1" * 64,
            "payload_sha256": "2" * 64,
            "pcm_sha256": "3" * 64,
        },
        "raw_session_dir": "results/fresh-session",
        "hardware": {"path": "configs/hardware.yaml", "sha256": "4" * 64},
        "level_evidence": {"path": "assets/level.json", "sha256": "5" * 64},
    }
    monkeypatch.setattr(
        meter,
        "_validate_broadband_followup_contract",
        lambda **kwargs: (_ for _ in ()).throw(
            FileExistsError("raw target claimed during meter")
        ),
    )

    with pytest.raises(FileExistsError, match="target claimed"):
        meter.revalidate_followup_contract(contract)


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


def test_obsolete_mode_and_fullband_v6_are_rejected_before_measure(monkeypatch):
    """historical v6 meter를 활성 CLI 경로로 되살리지 않는다."""

    monkeypatch.setattr(
        meter,
        "measure",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("obsolete meter mode는 measure() 전에 거부해야 합니다")
        ),
    )
    with pytest.raises(SystemExit):
        meter.main(["--mode", "fullband-v6", "--self-test"])
    with pytest.raises(SystemExit):
        meter.main(["--followup-mode", "fullband-v6", "--self-test"])


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
@pytest.mark.parametrize("followup_mode", ("strict", "broadband", "fullband-v5"))
def test_normal_meter_uses_permanent_evidence_but_always_emits_fresh_raw(
    tmp_path, monkeypatch, capsys, followup_mode
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
            "schema": "measurement_level_evidence_v2_bootstrap_pair",
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
    followup = {"mode": "strict"}
    if followup_mode == "broadband":
        followup = {
            "schema": "broadband_meter_followup_v1",
            "mode": "broadband",
            "plan": {
                "path": "results/data_audit/test-authority.json",
                "file_sha256": "1" * 64,
                "payload_sha256": "2" * 64,
                "pcm_sha256": "3" * 64,
            },
            "raw_session_dir": "results/calibration_interleaved/broadband/test-session",
            "hardware": {
                "path": "configs/hardware_jetson.yaml",
                "sha256": meter._sha256_file(
                    meter.REPO_ROOT / "configs/hardware_jetson.yaml"
                ),
            },
            "level_evidence": {
                "path": "assets/measured/measurement_level_evidence.json",
                "sha256": "e" * 64,
            },
        }
    elif followup_mode == "fullband-v5":
        followup = {
            "schema": "fullband_v5_meter_followup_v1",
            "mode": "fullband-v5",
            "status": "blocked_until_v5_live_adapter_implementation",
            "capture_only": True,
            "plan_live_capture_enabled": False,
            "plan_envelope": {
                "path": "assets/contracts/fullband_causal_v5_signal_plan.json",
            },
            "live_capture_authority": {
                "path": "assets/contracts/fullband_causal_v5_live_capture_authority.json",
            },
            "sealed_raw": {
                "path": "results/fullband_causal_v5/raw_capture.npz",
                "fresh": True,
            },
            "hardware": {
                "path": "configs/hardware_jetson.yaml",
                "sha256": meter._sha256_file(
                    meter.REPO_ROOT / "configs/hardware_jetson.yaml"
                ),
            },
            "level_evidence": {
                "path": "assets/measured/measurement_level_evidence.json",
                "sha256": "e" * 64,
                "schema": "measurement_level_evidence_v2_bootstrap_pair",
            },
            "canonical_payload_sha256": "f" * 64,
        }
    revalidations = []
    monkeypatch.setattr(
        meter,
        "revalidate_followup_contract",
        lambda contract: revalidations.append(dict(contract)) or contract,
    )

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
        confirm_routing_and_geometry=True,
        confirm_same_amplifier_setting=True,
        _validated_followup_contract=followup,
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
    if followup_mode == "strict":
        assert "--primary-out assets/measured/primary_path_il_strict_cap.npz" in output
        assert "--secondary-out assets/measured/secondary_path_il_strict_cap.npz" in output
        assert len(revalidations) == 2
        assert "followup_contract" not in observed["metadata"]
    elif followup_mode == "broadband":
        assert "measure_paths_broadband_interleaved.py --execute-live" in output
        assert "--hardware configs/hardware_jetson.yaml" in output
        assert "--plan results/data_audit/test-authority.json" in output
        assert "--primary-out" not in output
        assert len(revalidations) == 4
        assert observed["metadata"]["followup_contract"] == followup
        assert observed["metadata"]["hardware"] == followup["hardware"]
        assert observed["metadata"]["calibration_evidence"] == {
            "mode": "verified_existing",
            **followup["level_evidence"],
        }
    else:
        assert "measure_paths_fullband_causal_v5.py --execute-live" in output
        assert "blocked_until_v5_live_adapter_implementation" in output
        assert "--plan-envelope assets/contracts/fullband_causal_v5_signal_plan.json" in output
        assert "--live-authority assets/contracts/fullband_causal_v5_live_capture_authority.json" in output
        assert "--raw-target results/fullband_causal_v5/raw_capture.npz" in output
        assert len(revalidations) == 4
        assert observed["metadata"]["followup_contract"] == followup
        assert all(observed["metadata"]["fullband_v5_operator_confirmations"].values())
