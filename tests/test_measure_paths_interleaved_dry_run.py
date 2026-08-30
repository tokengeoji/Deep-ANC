"""interleaved 측정의 무출력 사전 검증 경로."""

import hashlib
import json
import datetime as dt
import signal
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from scripts.data import measure_paths_interleaved as mpi
from scripts.data import reanalyse_paths_interleaved as rpi


def _portaudio_available() -> bool:
    """Elice 학습 노드처럼 PortAudio가 없는 환경에서는 실기 경로만 건너뛴다."""

    try:
        import sounddevice  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


requires_portaudio = pytest.mark.skipif(
    not _portaudio_available(), reason="PortAudio가 없는 학습 노드에서는 실기 캡처를 실행하지 않음"
)


def _hardware() -> dict:
    return {
        "audio": {
            "sample_rate": 48_000,
            "block_size": 256,
            "latency": "low",
            "input": {"card": "APE", "pcm": 1, "channels": 2},
            "output": {"card": "Audio", "pcm": 0, "channels": 2},
        },
        "channels": dict(mpi.OFFICIAL_CHANNEL_MAP),
    }


def _physical_fingerprint(*, output_serial: str = "dac-test-001") -> dict:
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
        "schema": mpi.ALSA_PHYSICAL_FINGERPRINT_SCHEMA,
        "input": endpoint("APE", 1, "CAPTURE", "devices/platform/sound"),
        "output": endpoint("Audio", 0, "PLAYBACK", "devices/usb/1-1:1.0"),
    }
    payload["output"]["stable_attributes"] = [
        {
            "sys_relative_path": "devices/usb/1-1",
            "values": {"serial": output_serial},
        }
    ]
    unsigned = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["sha256"] = hashlib.sha256(unsigned).hexdigest()
    return payload


@pytest.fixture(autouse=True)
def _inject_physical_fingerprint(monkeypatch):
    fingerprint = _physical_fingerprint()
    monkeypatch.setattr(
        mpi,
        "collect_alsa_physical_fingerprint",
        lambda _config: json.loads(json.dumps(fingerprint)),
    )


def _install_normal_live_authority(monkeypatch, *, input_device=5, output_device=24):
    identity = mpi.measurement_hardware_identity(
        _hardware(), physical_fingerprint=_physical_fingerprint()
    )
    monkeypatch.setattr(
        mpi,
        "load_measurement_level_evidence",
        lambda *_args, **_kwargs: {
            "hardware_identity": identity,
            "_evidence_path": str(mpi.REPO_ROOT / "assets/measured/evidence.json"),
            "_evidence_sha256": "e" * 64,
        },
    )
    meter = {
        "path": mpi.REPO_ROOT / "results" / "fresh_meter_raw.npz",
        "sha256": "a" * 64,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc),
        "metadata": {
            "resolved_devices": {
                "input": int(input_device),
                "output": int(output_device),
            }
        },
    }
    monkeypatch.setattr(
        mpi, "validate_bootstrap_meter_raw", lambda *_args, **_kwargs: dict(meter)
    )

    @contextmanager
    def fake_lock(*_args, **_kwargs):
        yield {
            "path": "results/.test_audio.lock",
            "pid": 123,
            "uid": 1000,
            "purpose": "test",
        }

    monkeypatch.setattr(mpi, "repository_audio_lock", fake_lock)
    return meter


def test_dry_run_never_opens_audio_or_creates_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mpi, "load_yaml", lambda _path: _hardware())
    monkeypatch.setattr(
        mpi.cw,
        "_repo_path",
        lambda value, require_results=False: tmp_path
        / ("diagnostics" if require_results else Path(value).name),
    )
    monkeypatch.setattr(mpi, "alsa_card_index", lambda card: 1 if card == "APE" else 2)
    monkeypatch.setattr(mpi, "validate_alsa_pcm_mapping", lambda **_kwargs: None)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run에서 오디오/UUID 경로를 호출하면 안 됩니다")

    monkeypatch.setattr(mpi, "assert_live_pcm_clock_preconditions", forbidden)
    monkeypatch.setattr(mpi, "resolve_alsa_portaudio_device", forbidden)
    monkeypatch.setattr(mpi.cw, "_capture_preflight", forbidden)
    monkeypatch.setattr(mpi.cw, "_capture_measurement", forbidden)
    monkeypatch.setattr(mpi.uuid, "uuid4", forbidden)

    result = mpi.main(
        [
            "--dry-run",
            "--primary-out", "p.npz",
            "--secondary-out", "s.npz",
            "--diagnostics-root", "results/dry-run-forbidden",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "DRY-RUN PASS" in output
    assert "input-only preflight=3.00s" in output
    assert "output nominal=12.50s / hard-max=13.50s" in output
    assert "silent lead-in 0.50s + stimulus 12.00s" in output
    assert "level meter target=-50.1dBFS ±2.0dB" in output
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("audio", "sample_rate", 44_100), "sample_rate"),
        (("audio", "block_size", 512), "block_size"),
        (("audio", "latency", "high"), "hardware latency"),
        (("channels", "error_mic", 1), "exact"),
        (("channels", "cancel_out", 0), "exact"),
    ],
)
def test_dry_run_rejects_hardware_identity_before_paths_or_audio(
    tmp_path, monkeypatch, capsys, mutation, message
):
    config = _hardware()
    section, key, value = mutation
    config[section][key] = value
    monkeypatch.setattr(mpi, "load_yaml", lambda _path: config)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid hardware contract 뒤 path/audio를 호출하면 안 됩니다")

    monkeypatch.setattr(mpi.cw, "_repo_path", forbidden)
    monkeypatch.setattr(mpi, "alsa_card_index", forbidden)
    monkeypatch.setattr(mpi, "create_session_directory", forbidden)
    monkeypatch.setattr(mpi.cw, "_capture_preflight", forbidden)

    assert mpi.main(["--dry-run"]) == 2
    assert message in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "confirmation",
    [
        "--confirm-user-present",
        "--confirm-volume-minimum",
        "--confirm-routing-and-geometry",
    ],
)
def test_actual_measurement_requires_all_operator_confirmations_before_config(
    monkeypatch, capsys, confirmation
):
    monkeypatch.setattr(
        mpi,
        "load_yaml",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("operator confirmation 전에 config를 읽으면 안 됩니다")
        ),
    )

    assert mpi.main([confirmation]) == 2
    message = capsys.readouterr().err
    assert "--confirm-user-present" in message
    assert "--confirm-volume-minimum" in message
    assert "--confirm-routing-and-geometry" in message


@pytest.mark.parametrize(
    "argv",
    [
        [
            "--confirm-user-present",
            "--confirm-volume-minimum",
            "--confirm-routing-and-geometry",
        ],
        [
            "--meter-raw",
            "results/fresh_meter.npz",
            "--confirm-user-present",
            "--confirm-volume-minimum",
            "--confirm-routing-and-geometry",
        ],
    ],
)
def test_normal_strict_requires_fresh_meter_and_same_amplifier_before_config(
    monkeypatch, capsys, argv
):
    monkeypatch.setattr(
        mpi,
        "load_yaml",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("fresh meter confirmation 전에 config를 읽으면 안 됩니다")
        ),
    )

    assert mpi.main(argv) == 2
    message = capsys.readouterr().err
    assert "--meter-raw" in message
    assert "--confirm-same-amplifier-setting" in message


@pytest.mark.parametrize("mismatch", ["logical_pcm", "physical_dac_serial"])
def test_normal_strict_rejects_calibration_evidence_hardware_mismatch_before_meter(
    tmp_path, monkeypatch, capsys, mismatch
):
    monkeypatch.setattr(mpi, "load_yaml", lambda _path: _hardware())
    monkeypatch.setattr(
        mpi.cw,
        "_repo_path",
        lambda value, require_results=False: tmp_path
        / ("diagnostics" if require_results else Path(value).name),
    )
    wrong_physical = _physical_fingerprint(
        output_serial=(
            "different-physical-dac"
            if mismatch == "physical_dac_serial"
            else "dac-test-001"
        )
    )
    wrong = mpi.measurement_hardware_identity(
        _hardware(), physical_fingerprint=wrong_physical
    )
    if mismatch == "logical_pcm":
        wrong["output"]["pcm"] += 1
    monkeypatch.setattr(
        mpi,
        "load_measurement_level_evidence",
        lambda *_a, **_k: {"hardware_identity": wrong},
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("evidence identity 실패 뒤 meter/session/audio를 보면 안 됩니다")

    monkeypatch.setattr(mpi, "validate_bootstrap_meter_raw", forbidden)
    monkeypatch.setattr(mpi, "create_session_directory", forbidden)

    result = mpi.main(
        [
            "--meter-raw",
            "results/fresh_meter.npz",
            "--confirm-same-amplifier-setting",
            "--confirm-user-present",
            "--confirm-volume-minimum",
            "--confirm-routing-and-geometry",
            "--primary-out",
            "p.npz",
            "--secondary-out",
            "s.npz",
        ]
    )

    assert result == 2
    assert "hardware identity" in capsys.readouterr().err


def test_actual_measurement_requires_paired_level_raw_before_session_or_audio(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(mpi, "load_yaml", lambda _path: _hardware())
    monkeypatch.setattr(
        mpi.cw,
        "_repo_path",
        lambda value, require_results=False: tmp_path
        / ("diagnostics" if require_results else Path(value).name),
    )
    monkeypatch.setattr(
        mpi,
        "load_measurement_level_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("paired ch0 raw missing")
        ),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("evidence 실패 뒤 session/audio를 시작하면 안 됩니다")

    monkeypatch.setattr(mpi, "create_session_directory", forbidden)
    monkeypatch.setattr(mpi, "assert_live_pcm_clock_preconditions", forbidden)

    result = mpi.main(
        [
            "--meter-raw",
            "results/fresh_meter_raw.npz",
            "--confirm-same-amplifier-setting",
            "--confirm-user-present",
            "--confirm-volume-minimum",
            "--confirm-routing-and-geometry",
            "--primary-out",
            "p.npz",
            "--secondary-out",
            "s.npz",
        ]
    )

    assert result == 2
    assert "paired ch0 raw missing" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@requires_portaudio
def test_strict_rechecks_physical_fingerprint_after_input_preflight_before_output(
    tmp_path, monkeypatch, capsys
):
    diagnostics = tmp_path / "diagnostics"
    monkeypatch.setattr(mpi, "load_yaml", lambda _path: _hardware())
    monkeypatch.setattr(
        mpi.cw,
        "_repo_path",
        lambda value, require_results=False: (
            diagnostics if require_results else tmp_path / Path(value).name
        ),
    )
    _install_normal_live_authority(monkeypatch)
    fingerprints = iter(
        (
            _physical_fingerprint(),
            _physical_fingerprint(output_serial="swapped-dac-before-output"),
        )
    )
    monkeypatch.setattr(
        mpi, "collect_alsa_physical_fingerprint", lambda _config: next(fingerprints)
    )
    monkeypatch.setattr(
        mpi, "assert_live_pcm_clock_preconditions", lambda _hardware: None
    )
    monkeypatch.setattr(mpi, "resolve_alsa_portaudio_device", lambda *_a, **_k: 24)
    raw = np.column_stack(
        (
            np.arange(256, dtype=np.int32) * 10_000,
            np.arange(256, dtype=np.int32) * 7_000,
        )
    )
    report = mpi.cw.analyze_int32_input_probe(raw)
    report.update({"device": 5, "sample_rate": 48_000})
    monkeypatch.setattr(
        mpi.cw, "_capture_preflight", lambda *_a, **_k: (raw, report)
    )

    def forbidden_capture(*_args, **_kwargs):
        _kwargs["pre_open_check"]()
        raise AssertionError("physical fingerprint 변경 뒤 output을 열면 안 됩니다")

    monkeypatch.setattr(
        mpi, "capture_measurement_preserving_partial", forbidden_capture
    )
    result = mpi.main(
        [
            "--meter-raw", "results/fresh_meter_raw.npz",
            "--confirm-same-amplifier-setting",
            "--confirm-user-present",
            "--confirm-volume-minimum",
            "--confirm-routing-and-geometry",
            "--primary-out", "p.npz",
            "--secondary-out", "s.npz",
            "--diagnostics-root", "results/physical-swap",
        ]
    )

    assert result == 1
    assert "physical fingerprint" in capsys.readouterr().err


@pytest.mark.parametrize(
    "missing",
    ["meter", "same_amplifier"],
)
def test_actual_bootstrap_requires_meter_and_same_amplifier_before_config(
    monkeypatch, capsys, missing
):
    monkeypatch.setattr(
        mpi,
        "load_yaml",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("bootstrap confirmation 전에 config를 읽으면 안 됩니다")
        ),
    )
    argv = [
        "--bootstrap-level-evidence",
        "--confirm-user-present",
        "--confirm-volume-minimum",
        "--confirm-routing-and-geometry",
    ]
    if missing != "meter":
        argv += ["--bootstrap-meter-raw", "results/meter_raw.npz"]
    if missing != "same_amplifier":
        argv += ["--confirm-same-amplifier-setting"]

    assert mpi.main(argv) == 2
    output = capsys.readouterr().err
    assert "--meter-raw" in output
    assert "--confirm-same-amplifier-setting" in output


def test_actual_bootstrap_validates_fresh_meter_before_session_or_audio(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(mpi, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mpi, "load_yaml", lambda _path: _hardware())
    monkeypatch.setattr(
        mpi.cw,
        "_repo_path",
        lambda value, require_results=False: tmp_path
        / ("diagnostics" if require_results else Path(value).name),
    )
    monkeypatch.setattr(
        mpi,
        "validate_bootstrap_meter_raw",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("stale meter raw")
        ),
    )
    monkeypatch.setattr(
        mpi,
        "load_measurement_level_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bootstrap에서 기존 evidence loader를 쓰면 안 됩니다")
        ),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("meter 검증 실패 뒤 session/audio를 시작하면 안 됩니다")

    monkeypatch.setattr(mpi, "create_session_directory", forbidden)
    monkeypatch.setattr(mpi, "assert_live_pcm_clock_preconditions", forbidden)

    result = mpi.main(
        [
            "--bootstrap-level-evidence",
            "--level-evidence",
            str(tmp_path / "measurement_level_evidence.json"),
            "--bootstrap-meter-raw",
            "results/meter_raw.npz",
            "--confirm-same-amplifier-setting",
            "--confirm-user-present",
            "--confirm-volume-minimum",
            "--confirm-routing-and-geometry",
            "--primary-out",
            "p.npz",
            "--secondary-out",
            "s.npz",
        ]
    )

    assert result == 2
    assert "stale meter raw" in capsys.readouterr().err


def test_bootstrap_dry_run_opens_no_audio_and_validates_no_raw(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(mpi, "load_yaml", lambda _path: _hardware())
    monkeypatch.setattr(
        mpi.cw,
        "_repo_path",
        lambda value, require_results=False: tmp_path
        / ("diagnostics" if require_results else Path(value).name),
    )
    monkeypatch.setattr(mpi, "alsa_card_index", lambda card: 1 if card == "APE" else 2)
    monkeypatch.setattr(mpi, "validate_alsa_pcm_mapping", lambda **_kwargs: None)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("bootstrap dry-run은 raw/audio/session을 건드리면 안 됩니다")

    monkeypatch.setattr(mpi, "validate_bootstrap_meter_raw", forbidden)
    monkeypatch.setattr(mpi, "assert_live_pcm_clock_preconditions", forbidden)
    monkeypatch.setattr(mpi, "create_session_directory", forbidden)

    result = mpi.main(
        [
            "--dry-run",
            "--bootstrap-level-evidence",
            "--primary-out",
            "p.npz",
            "--secondary-out",
            "s.npz",
            "--diagnostics-root",
            "results/bootstrap-dry-run",
        ]
    )

    assert result == 0
    assert "DRY-RUN PASS" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


@requires_portaudio
def test_bootstrap_evidence_failure_blocks_all_postprocessing_and_official_outputs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(mpi, "REPO_ROOT", tmp_path)
    events = []
    lock_active = {"value": False}
    diagnostics = tmp_path / "diagnostics"
    monkeypatch.setattr(mpi, "load_yaml", lambda _path: _hardware())
    monkeypatch.setattr(
        mpi.cw,
        "_repo_path",
        lambda value, require_results=False: (
            diagnostics if require_results else tmp_path / Path(value).name
        ),
    )
    meter = {
        "path": mpi.REPO_ROOT / "results" / "test_meter_raw.npz",
        "sha256": "a" * 64,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc),
        "metadata": {"resolved_devices": {"input": 5, "output": 24}},
    }
    def validate_meter(*_args, **_kwargs):
        events.append("meter_validate")
        return dict(meter)

    monkeypatch.setattr(mpi, "validate_bootstrap_meter_raw", validate_meter)

    def precondition(*_args, **_kwargs):
        events.append("pcm_clock_precondition")

    monkeypatch.setattr(mpi, "assert_live_pcm_clock_preconditions", precondition)
    monkeypatch.setattr(mpi, "resolve_alsa_portaudio_device", lambda *_a, **_k: 24)
    preflight_index = np.arange(256, dtype=np.float64)
    preflight_raw = np.stack(
        (
            np.rint(900_000 * np.sin(preflight_index * 0.07)),
            np.rint(700_000 * np.cos(preflight_index * 0.05)),
        ),
        axis=1,
    ).astype(np.int32)
    preflight_report = mpi.cw.analyze_int32_input_probe(preflight_raw)
    preflight_report.update({"device": 5, "sample_rate": 48_000})
    def capture_preflight(_sd, _hardware_config, analyzed_seconds):
        # 1초 I2S settle + 2초 analyzed raw = 실제 input-only 총 3초.
        assert analyzed_seconds == pytest.approx(2.0)
        events.append("input_preflight")
        return preflight_raw, preflight_report

    monkeypatch.setattr(mpi.cw, "_capture_preflight", capture_preflight)

    @contextmanager
    def tracked_lock(*_args, **_kwargs):
        events.append("lock_enter")
        lock_active["value"] = True
        try:
            yield {"path": "results/.lock", "pid": 1, "uid": 1000, "purpose": "test"}
        finally:
            lock_active["value"] = False
            events.append("lock_exit")

    monkeypatch.setattr(mpi, "repository_audio_lock", tracked_lock)

    def completed_capture(_sd, **kwargs):
        kwargs["pre_open_check"]()
        assert lock_active["value"] is True
        events.append("stream_capture_and_close")
        output = np.asarray(kwargs["output_float"], dtype=np.float32)
        index = np.arange(output.shape[0], dtype=np.float64)
        recorded = np.stack(
            (
                np.rint(1_000_000 * np.sin(index * 0.011)),
                np.rint(800_000 * np.cos(index * 0.013)),
            ),
            axis=1,
        ).astype(np.int32)
        return (
            recorded,
            mpi.cw.float32_to_pcm_int16(output),
            {
                "xrun_count": 0,
                "unexpected_status_count": 0,
                "callback_error": None,
                "stream_abort_error": None,
                "stream_close_error": None,
                "completed": True,
                "output_stop_confirmed": True,
                "captured_frames": int(output.shape[0]),
            },
        )

    monkeypatch.setattr(
        mpi, "capture_measurement_preserving_partial", completed_capture
    )
    def reject_evidence(*_args, **_kwargs):
        events.append("evidence")
        raise ValueError("injected paired evidence rejection")

    monkeypatch.setattr(
        mpi, "create_measurement_level_evidence_atomic", reject_evidence
    )

    def forbidden_postprocess(*_args, **_kwargs):
        events.append("postprocess")
        raise AssertionError("evidence 실패 뒤 postprocessing을 실행하면 안 됩니다")

    monkeypatch.setattr(mpi, "pcm_int32_to_float32", forbidden_postprocess)
    monkeypatch.setattr(mpi, "write_official_pair_atomic", forbidden_postprocess)

    result = mpi.main(
        [
            "--bootstrap-level-evidence",
            "--level-evidence",
            str(tmp_path / "measurement_level_evidence.json"),
            "--bootstrap-meter-raw",
            "results/meter_raw.npz",
            "--confirm-same-amplifier-setting",
            "--confirm-user-present",
            "--confirm-volume-minimum",
            "--confirm-routing-and-geometry",
            "--primary-out",
            "primary.npz",
            "--secondary-out",
            "secondary.npz",
            "--diagnostics-root",
            "results/bootstrap-evidence-order",
        ]
    )

    assert result == 1
    assert events == [
        "meter_validate",
        "lock_enter",
        "pcm_clock_precondition",
        "input_preflight",
        "meter_validate",
        "pcm_clock_precondition",
        "stream_capture_and_close",
        "lock_exit",
        "evidence",
    ]
    sessions = [path for path in diagnostics.iterdir() if path.is_dir()]
    assert len(sessions) == 1
    assert (sessions[0] / "raw_measurement.npz").is_file()
    assert not (tmp_path / "primary.npz").exists()
    assert not (tmp_path / "secondary.npz").exists()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--consistency-band", "300", "1600"], "official --consistency-band"),
        (["--required-band", "200", "1600"], "official --required-band"),
        (["--fit-band", "150", "20000"], "fit-band"),
        (["--fit-band", "150", "151"], "톤이"),
        (["--band", "200", "1650"], "official --band"),
        (["--period-seconds", "0.005"], "period-seconds"),
        (["--period-seconds", "1"], "period-seconds"),
        (["--warmup-periods", "33"], "warmup-periods"),
        (["--repeats", "65"], "repeats"),
        (["--input-probe-seconds", "4"], "input-probe-seconds"),
        (["--band", "60", "1700"], "official --band"),
        (["--required-band", "100", "1600"], "official --required-band"),
        (["--consistency-band", "100", "1600"], "official --consistency-band"),
        (["--tone-spacing-hz", "-1"], "tone-spacing-hz"),
        (["--pre-roll", "4000"], "pre-roll"),
        (["--fir-length", "256", "--pre-roll", "256"], "fir-length"),
        (["--fir-length", "4000"], "fir-length"),
        (["--max-delay-ms", "1"], "pre-roll"),
        (["--min-alignment-score", "0.94"], "min-alignment-score"),
        (["--min-alignment-score", "1.1"], "min-alignment-score"),
        (["--min-alignment-score", "nan"], "min-alignment-score"),
        (["--max-relative-tau-samples", "3.1"], "max-relative-tau-samples"),
        (["--max-drift-deviation-samples", "2.1"], "max-drift-deviation-samples"),
        (["--max-delay-jitter-ms", "0.1"], "0..3 samples"),
        (["--min-kept-repeats", "7"], "min-kept-repeats"),
        (["--min-kept-repeats", "65"], "min-kept-repeats"),
        (["--warmup-periods", "0"], "warmup-periods"),
        (["--block-size", "0"], "block-size"),
        (["--block-size", "512"], "official block-size"),
        (["--latency", "high"], "official latency"),
        (["--track-window", "0"], "track-window"),
        (["--track-min-peak", "1.1"], "track-window"),
        (["--dewarp"], "dewarp"),
        (["--input-probe-seconds", "0"], "input-probe-seconds"),
        (["--input-probe-seconds", "nan"], "input-probe-seconds"),
        (["--amplitude", "0.004"], "amplitude"),
        (["--amplitude", "0.021"], "amplitude"),
    ],
)
def test_dry_run_rejects_analysis_contract_before_hardware(
    tmp_path, monkeypatch, capsys, extra, message
):
    monkeypatch.setattr(mpi, "load_yaml", lambda _path: _hardware())
    monkeypatch.setattr(
        mpi.cw,
        "_repo_path",
        lambda value, require_results=False: tmp_path
        / ("diagnostics" if require_results else Path(value).name),
    )
    monkeypatch.setattr(
        mpi,
        "alsa_card_index",
        lambda *_args: (_ for _ in ()).throw(AssertionError("hardware까지 가면 안 됩니다")),
    )
    argv = ["--dry-run", "--primary-out", "p.npz", "--secondary-out", "s.npz"] + extra

    assert mpi.main(argv) == 2
    assert message in capsys.readouterr().err


def test_parser_separates_safe_default_from_hard_amplitude_limit():
    args = mpi.build_parser().parse_args(["--dry-run"])

    assert args.amplitude == pytest.approx(
        mpi.OFFICIAL_MEASUREMENT_LEVEL.probe_amplitude
    )
    assert args.amplitude < mpi.cw.MAX_AMPLITUDE
    assert args.warmup_periods == 32
    assert args.repeats == 64


def test_duration_plan_uses_actual_rounded_probe_samples():
    args = mpi.build_parser().parse_args(["--dry-run"])

    durations = mpi.measurement_duration_plan(
        args,
        sample_rate=48_000,
        period_samples=6_001,
    )

    assert durations["lead_in_seconds"] == pytest.approx(0.5)
    assert durations["stimulus_seconds"] == pytest.approx(96 * 6_001 / 48_000)
    assert durations["output_stream_seconds"] == pytest.approx(
        0.5 + 96 * 6_001 / 48_000
    )


def test_capture_release_notice_precedes_analysis(monkeypatch):
    events = []
    monkeypatch.setattr(
        mpi,
        "announce_speaker_disconnect",
        lambda **_kwargs: events.append("disconnect"),
    )

    result = mpi.capture_with_speaker_release_notice(
        lambda: events.append("capture") or "captured"
    )
    events.append("analysis")

    assert result == "captured"
    assert events == ["capture", "disconnect", "analysis"]


def test_capture_failure_still_announces_speaker_release(monkeypatch):
    events = []
    monkeypatch.setattr(
        mpi,
        "announce_speaker_disconnect",
        lambda **_kwargs: events.append("disconnect"),
    )

    def fail_capture():
        events.append("capture")
        raise RuntimeError("injected capture failure")

    with pytest.raises(RuntimeError, match="injected capture failure"):
        mpi.capture_with_speaker_release_notice(fail_capture)

    assert events == ["capture", "disconnect"]


def test_clock_mask_reapplies_drift_envelope_to_final_median_fixed_point():
    # Initial median is 2, so the old one-pass envelope retained all 11 rows.
    # The adjacent gate then removes the two transition rows, moving the median
    # to 0 while four d=4 rows remain outside the final ±2 envelope.  That old
    # mask had 9 rows and could be promoted, then rejected by readiness.
    common = np.asarray([0.0] * 5 + [2.0] + [4.0] * 5 + [np.nan])
    base_valid = np.isfinite(common)
    adjacent = np.full(common.size, np.nan, dtype=np.float64)
    adjacent[1:-1] = np.abs(np.diff(common[:-1]))

    with pytest.raises(ValueError, match="final-median.*5개"):
        mpi._fixed_point_clock_valid_mask(
            base_valid=base_valid,
            common_delay_samples=common,
            adjacent_change_samples=adjacent,
            max_drift_deviation_samples=2.0,
            max_adjacent_change_samples=0.5,
            min_valid_periods=8,
        )


def test_channel_quality_requires_importer_median_snr_gate():
    reasons = mpi.channel_quality(
        consistency=0.999,
        snr_db=np.full(64, 20.0),
        min_consistency=0.9,
    )

    assert not any(reason.startswith("tone_snr_coverage") for reason in reasons)
    assert reasons == ["tone_snr_median_20.00"]


def test_unexpected_callback_status_is_a_raw_capture_defect():
    reasons = mpi.capture_invalid_reasons(
        {
            "xrun_count": 0,
            "unexpected_status_count": 2,
            "completed": True,
        },
        {
            "channels": [
                {"clip_ratio": 0.0, "valid": True},
                {"clip_ratio": 0.0, "valid": True},
            ]
        },
    )

    assert reasons == ["unexpected_callback_status_2"]


@pytest.mark.parametrize(
    ("channels", "expected"),
    [
        ([{"valid": False, "clip_ratio": 0.0}, {"valid": True, "clip_ratio": 0.0}],
         "measurement_invalid_ch0"),
        ([{"valid": True, "clip_ratio": 0.0}],
         "measurement_missing_err_ref_channels"),
    ],
)
def test_dead_or_missing_measurement_mic_is_a_raw_capture_defect(channels, expected):
    reasons = mpi.capture_invalid_reasons(
        {
            "xrun_count": 0,
            "unexpected_status_count": 0,
            "completed": True,
        },
        {"channels": channels},
    )

    assert expected in reasons


def test_interleaved_capture_failure_surfaces_partial_arrays_for_raw_recovery():
    class FakeSD:
        class CallbackStop(Exception):
            pass

        class CallbackAbort(Exception):
            pass

        class Stream:
            def __init__(self, *, callback, **_kwargs):
                self.callback = callback

            def start(self):
                frames = 4
                indata = np.arange(frames * 2, dtype=np.int32).reshape(frames, 2)
                outdata = np.zeros((frames, 2), dtype=np.int16)
                self.callback(indata, outdata, frames, None, None)
                raise OSError("injected stream failure after speaker output")

            def abort(self):
                pass

            def close(self):
                pass

    output = np.linspace(-0.003, 0.003, 16, dtype=np.float32).reshape(8, 2)

    with pytest.raises(mpi.PartialCaptureError, match="injected stream failure") as caught:
        mpi.capture_measurement_preserving_partial(
            FakeSD,
            fs=48_000,
            block_size=4,
            latency="low",
            in_dev=1,
            out_dev=2,
            output_float=output,
        )

    error = caught.value
    assert error.telemetry["captured_frames"] == 4
    assert error.telemetry["completed"] is False
    np.testing.assert_array_equal(
        error.recorded_raw[:4], np.arange(8, dtype=np.int32).reshape(4, 2)
    )
    np.testing.assert_array_equal(
        error.output_pcm, mpi.cw.float32_to_pcm_int16(output)
    )


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_interleaved_signal_aborts_closes_and_announces_disconnect(
    monkeypatch, capsys, signum
):
    events = []

    class FakeSD:
        class CallbackStop(Exception):
            pass

        class CallbackAbort(Exception):
            pass

        class Stream:
            def __init__(self, **_kwargs):
                events.append("construct")

            def start(self):
                events.append("start")
                handler = signal.getsignal(signum)
                assert callable(handler)
                handler(signum, None)

            def abort(self):
                events.append("abort")

            def close(self):
                events.append("close")

    output = np.zeros((8, 2), dtype=np.float32)
    with pytest.raises(mpi.PartialCaptureError) as caught:
        mpi.capture_with_speaker_release_notice(
            lambda: mpi.capture_measurement_preserving_partial(
                FakeSD,
                fs=48_000,
                block_size=8,
                latency="low",
                in_dev=1,
                out_dev=2,
                output_float=output,
            )
        )

    assert events == ["construct", "start", "abort", "close"]
    assert caught.value.telemetry["termination_signal"] == int(signum)
    assert caught.value.telemetry["output_stop_confirmed"] is True
    assert mpi.SPEAKER_DISCONNECT_NOTICE in capsys.readouterr().out


def test_interleaved_close_failure_is_not_swallowed_and_requires_physical_disconnect(
    monkeypatch,
):
    events = []

    class FakeSD:
        class CallbackStop(Exception):
            pass

        class CallbackAbort(Exception):
            pass

        class Stream:
            def __init__(self, *, callback, **_kwargs):
                self.callback = callback

            def start(self):
                frames = 8
                indata = np.arange(frames * 2, dtype=np.int32).reshape(frames, 2)
                outdata = np.zeros((frames, 2), dtype=np.int16)
                with pytest.raises(FakeSD.CallbackStop):
                    self.callback(indata, outdata, frames, None, None)

            def abort(self):
                return None

            def close(self):
                raise OSError("injected close failure")

    monkeypatch.setattr(
        mpi,
        "announce_speaker_disconnect",
        lambda *, output_stop_confirmed=True: events.append(output_stop_confirmed),
    )
    output = np.linspace(-0.003, 0.003, 16, dtype=np.float32).reshape(8, 2)

    with pytest.raises(mpi.PartialCaptureError, match="injected close failure") as caught:
        mpi.capture_with_speaker_release_notice(
            lambda: mpi.capture_measurement_preserving_partial(
                FakeSD,
                fs=48_000,
                block_size=8,
                latency="low",
                in_dev=1,
                out_dev=2,
                output_float=output,
            )
        )

    assert caught.value.telemetry["stream_close_error"].endswith(
        "injected close failure"
    )
    assert caught.value.telemetry["output_stop_confirmed"] is False
    assert events == [False]


@requires_portaudio
def test_main_partial_capture_writes_invalid_immutable_raw_and_reanalysis_rejects(
    tmp_path, monkeypatch
):
    diagnostics = tmp_path / "diagnostics"
    monkeypatch.setattr(mpi, "load_yaml", lambda _path: _hardware())

    def repo_path(value, require_results=False):
        if require_results:
            return diagnostics
        return tmp_path / Path(value).name

    monkeypatch.setattr(mpi.cw, "_repo_path", repo_path)
    monkeypatch.setattr(
        mpi, "assert_live_pcm_clock_preconditions", lambda *_a, **_k: None
    )
    _install_normal_live_authority(monkeypatch)
    monkeypatch.setattr(mpi, "resolve_alsa_portaudio_device", lambda *_a, **_k: 24)
    preflight_index = np.arange(256, dtype=np.float64)
    preflight_raw = np.stack(
        (
            np.rint(900_000 * np.sin(preflight_index * 0.07)),
            np.rint(700_000 * np.cos(preflight_index * 0.05)),
        ),
        axis=1,
    ).astype(np.int32)
    preflight_report = mpi.cw.analyze_int32_input_probe(preflight_raw)
    preflight_report.update({"device": 5, "sample_rate": 48_000})
    monkeypatch.setattr(
        mpi.cw,
        "_capture_preflight",
        lambda *_a, **_k: (preflight_raw, preflight_report),
    )

    def fail_after_output(_sd, **kwargs):
        kwargs["pre_open_check"]()
        output = np.asarray(kwargs["output_float"], dtype=np.float32)
        recorded = np.zeros(output.shape, dtype=np.int32)
        recorded[:4] = np.arange(8, dtype=np.int32).reshape(4, 2)
        raise mpi.PartialCaptureError(
            OSError("injected callback failure after output"),
            recorded_raw=recorded,
            output_pcm=mpi.cw.float32_to_pcm_int16(output),
            telemetry={
                "callback_count": 1,
                "callback_status_count": 0,
                "xrun_count": 0,
                "priming_output_count": 0,
                "unexpected_status_count": 0,
                "statuses": [],
                "callback_error": "OSError: injected callback failure",
                "completed": False,
                "captured_frames": 4,
                "elapsed_seconds": 0.01,
            },
        )

    monkeypatch.setattr(mpi, "capture_measurement_preserving_partial", fail_after_output)

    result = mpi.main(
        [
            "--meter-raw", "results/fresh_meter_raw.npz",
            "--confirm-same-amplifier-setting",
            "--confirm-user-present",
            "--confirm-volume-minimum",
            "--confirm-routing-and-geometry",
            "--primary-out", "primary.npz",
            "--secondary-out", "secondary.npz",
            "--diagnostics-root", "results/partial-e2e",
        ]
    )

    assert result == 1
    sessions = [path for path in diagnostics.iterdir() if path.is_dir()]
    assert len(sessions) == 1
    session = sessions[0]
    raw_path = session / "raw_measurement.npz"
    metadata_path = session / "metadata.json"
    assert raw_path.is_file() and metadata_path.is_file()
    with np.load(raw_path, allow_pickle=False) as data:
        embedded = json.loads(str(data["metadata_json"]))
        assert data["input_raw_int32"].dtype == np.int32
        assert data["output_pcm_int16"].dtype == np.int16
    assert "capture_incomplete" in embedded["invalid_reasons"]
    assert "callback_error" in " ".join(embedded["invalid_reasons"])
    assert json.loads(metadata_path.read_text()) == embedded
    with pytest.raises(ValueError, match="캡처 자체가 결함"):
        rpi.load_capture(session)


@pytest.mark.parametrize(
    ("fault_site", "fault"),
    [
        ("pcm", RuntimeError("injected pcm conversion failure")),
        ("analyze", RuntimeError("injected measurement analysis failure")),
        ("analyze", KeyboardInterrupt()),
    ],
)
@requires_portaudio
def test_completed_capture_is_durable_before_any_postprocessing_fault(
    tmp_path, monkeypatch, fault_site, fault
):
    diagnostics = tmp_path / "diagnostics"
    monkeypatch.setattr(mpi, "load_yaml", lambda _path: _hardware())
    monkeypatch.setattr(
        mpi.cw,
        "_repo_path",
        lambda value, require_results=False: (
            diagnostics if require_results else tmp_path / Path(value).name
        ),
    )
    monkeypatch.setattr(
        mpi, "assert_live_pcm_clock_preconditions", lambda *_a, **_k: None
    )
    _install_normal_live_authority(monkeypatch)
    monkeypatch.setattr(mpi, "resolve_alsa_portaudio_device", lambda *_a, **_k: 24)
    preflight_index = np.arange(256, dtype=np.float64)
    preflight_raw = np.stack(
        (
            np.rint(900_000 * np.sin(preflight_index * 0.07)),
            np.rint(700_000 * np.cos(preflight_index * 0.05)),
        ),
        axis=1,
    ).astype(np.int32)
    preflight_report = mpi.cw.analyze_int32_input_probe(preflight_raw)
    preflight_report.update({"device": 5, "sample_rate": 48_000})
    monkeypatch.setattr(
        mpi.cw,
        "_capture_preflight",
        lambda *_a, **_k: (preflight_raw, preflight_report),
    )

    def completed_capture(_sd, **kwargs):
        kwargs["pre_open_check"]()
        output = np.asarray(kwargs["output_float"], dtype=np.float32)
        index = np.arange(output.shape[0], dtype=np.float64)
        recorded = np.stack(
            (
                np.rint(1_000_000 * np.sin(index * 0.011)),
                np.rint(800_000 * np.cos(index * 0.013)),
            ),
            axis=1,
        ).astype(np.int32)
        return (
            recorded,
            mpi.cw.float32_to_pcm_int16(output),
            {
                "callback_count": 10,
                "callback_status_count": 0,
                "xrun_count": 0,
                "priming_output_count": 0,
                "unexpected_status_count": 0,
                "statuses": [],
                "callback_error": None,
                "completed": True,
                "captured_frames": int(output.shape[0]),
                "elapsed_seconds": 1.0,
            },
        )

    monkeypatch.setattr(mpi, "capture_measurement_preserving_partial", completed_capture)

    def raise_fault(*_args, **_kwargs):
        raise fault

    if fault_site == "pcm":
        monkeypatch.setattr(mpi, "pcm_int32_to_float32", raise_fault)
    else:
        monkeypatch.setattr(mpi.cw, "analyze_int32_input_probe", raise_fault)

    result = mpi.main(
        [
            "--meter-raw", "results/fresh_meter_raw.npz",
            "--confirm-same-amplifier-setting",
            "--confirm-user-present",
            "--confirm-volume-minimum",
            "--confirm-routing-and-geometry",
            "--primary-out", "primary.npz",
            "--secondary-out", "secondary.npz",
            "--diagnostics-root", "results/postprocess-fault",
        ]
    )

    assert result == 1
    sessions = [path for path in diagnostics.iterdir() if path.is_dir()]
    assert len(sessions) == 1
    raw_path = sessions[0] / "raw_measurement.npz"
    assert raw_path.is_file()
    with np.load(raw_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        assert metadata["telemetry"]["completed"] is True
        assert "measurement" not in metadata
        assert data["input_raw_int32"].dtype == np.int32
        assert data["output_pcm_int16"].dtype == np.int16
    assert not (sessions[0] / "analysis_results.npz").exists()
    assert not (tmp_path / "primary.npz").exists()
    assert not (tmp_path / "secondary.npz").exists()


@pytest.mark.parametrize("fail_on_publish", [1, 2])
def test_analysis_write_exception_preserves_immutable_raw_pair(
    tmp_path, monkeypatch, fail_on_publish
):
    raw_path = tmp_path / "raw_measurement.npz"
    raw_metadata_path = tmp_path / "metadata.json"
    np.savez_compressed(raw_path, sentinel=np.arange(8, dtype=np.int16))
    raw_metadata_path.write_text('{"capture_id":"immutable"}\n', encoding="utf-8")
    before = {
        path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.read_bytes())
        for path in (raw_path, raw_metadata_path)
    }

    original_publish = mpi.atomic_publish_noreplace
    calls = 0

    def injected_publish(source, target):
        nonlocal calls
        calls += 1
        if calls == fail_on_publish:
            raise OSError("injected publish failure")
        return original_publish(source, target)

    monkeypatch.setattr(mpi, "atomic_publish_noreplace", injected_publish)
    with pytest.raises(OSError, match="injected publish failure"):
        mpi.write_analysis_outputs_atomic(
            tmp_path,
            metadata={"valid": True},
            arrays={"result": np.arange(4, dtype=np.float64)},
        )

    for path in (raw_path, raw_metadata_path):
        digest, content = before[path.name]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert path.read_bytes() == content
    assert not (tmp_path / "analysis_results.npz").exists()
    assert not (tmp_path / "analysis_metadata.json").exists()
    assert not list(tmp_path.glob("*.partial"))


def test_analysis_outputs_are_separate_and_atomic(tmp_path):
    raw_path = tmp_path / "raw_measurement.npz"
    raw_metadata_path = tmp_path / "metadata.json"
    raw_path.write_bytes(b"raw-never-overwritten")
    raw_metadata_path.write_bytes(b"raw-meta-never-overwritten")

    paths = mpi.write_analysis_outputs_atomic(
        tmp_path,
        metadata={"valid": True, "capture_id": "cap"},
        arrays={"result": np.arange(4, dtype=np.float64)},
    )

    assert raw_path.read_bytes() == b"raw-never-overwritten"
    assert raw_metadata_path.read_bytes() == b"raw-meta-never-overwritten"
    with np.load(paths["results"], allow_pickle=False) as data:
        np.testing.assert_array_equal(data["result"], np.arange(4, dtype=np.float64))
    assert json.loads(paths["metadata"].read_text(encoding="utf-8")) == {
        "capture_id": "cap",
        "valid": True,
    }


def _raw_payload():
    return (
        {"capture_id": "raw-cap", "sample_rate": 48_000},
        {"input_raw_int32": np.arange(16, dtype=np.int32).reshape(8, 2)},
    )


def _assert_no_partial_files(path: Path) -> None:
    assert not list(path.glob("*.partial"))


def test_session_directory_creation_fsyncs_diagnostics_parent(tmp_path, monkeypatch):
    fsynced = []
    monkeypatch.setattr(mpi, "_fsync_directory", lambda path: fsynced.append(Path(path)))

    session = mpi.create_session_directory(tmp_path, "capture-id")

    assert session == tmp_path / "capture-id"
    assert session.is_dir()
    assert fsynced == [tmp_path]


def test_session_directory_durably_creates_each_missing_diagnostics_ancestor(
    tmp_path, monkeypatch
):
    diagnostics_root = tmp_path / "new-results" / "nested-diagnostics"
    fsynced = []
    monkeypatch.setattr(mpi, "_fsync_directory", lambda path: fsynced.append(Path(path)))

    session = mpi.create_session_directory(diagnostics_root, "capture-id")

    assert session.is_dir()
    assert fsynced == [tmp_path, tmp_path / "new-results", diagnostics_root]


def test_raw_capture_is_completed_before_atomic_exposure(tmp_path):
    metadata, arrays = _raw_payload()

    paths = mpi.write_immutable_raw_capture_atomic(
        tmp_path, metadata=metadata, arrays=arrays
    )

    with np.load(paths["raw"], allow_pickle=False) as data:
        assert json.loads(str(data["metadata_json"])) == metadata
        np.testing.assert_array_equal(data["input_raw_int32"], arrays["input_raw_int32"])
    assert json.loads(paths["metadata"].read_text(encoding="utf-8")) == metadata
    _assert_no_partial_files(tmp_path)


def test_raw_temp_truncation_never_exposes_final_capture(tmp_path, monkeypatch):
    metadata, arrays = _raw_payload()

    def truncated_write(handle, **_arrays):
        handle.write(b"truncated-not-an-npz")
        handle.flush()
        raise OSError("injected raw write failure")

    monkeypatch.setattr(mpi.np, "savez_compressed", truncated_write)
    with pytest.raises(OSError, match="injected raw write failure"):
        mpi.write_immutable_raw_capture_atomic(
            tmp_path, metadata=metadata, arrays=arrays
        )

    assert not (tmp_path / "raw_measurement.npz").exists()
    assert not (tmp_path / "metadata.json").exists()
    _assert_no_partial_files(tmp_path)


def test_raw_publish_failure_never_exposes_final_capture(tmp_path, monkeypatch):
    metadata, arrays = _raw_payload()
    monkeypatch.setattr(
        mpi,
        "atomic_publish_noreplace",
        lambda *_args: (_ for _ in ()).throw(OSError("injected raw publish failure")),
    )

    with pytest.raises(OSError, match="injected raw publish failure"):
        mpi.write_immutable_raw_capture_atomic(
            tmp_path, metadata=metadata, arrays=arrays
        )

    assert not (tmp_path / "raw_measurement.npz").exists()
    assert not (tmp_path / "metadata.json").exists()
    _assert_no_partial_files(tmp_path)


@pytest.mark.parametrize("failure", ["write", "publish"])
def test_sidecar_failure_preserves_canonical_raw_for_recovery(
    tmp_path, monkeypatch, failure
):
    metadata, arrays = _raw_payload()
    if failure == "write":
        def failed_dump(_value, handle, **_kwargs):
            handle.write("{truncated")
            handle.flush()
            raise OSError("injected sidecar write failure")

        monkeypatch.setattr(mpi.json, "dump", failed_dump)
        message = "injected sidecar write failure"
    else:
        original_publish = mpi.atomic_publish_noreplace
        calls = 0

        def failed_second_publish(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected sidecar publish failure")
            return original_publish(source, target)

        monkeypatch.setattr(mpi, "atomic_publish_noreplace", failed_second_publish)
        message = "injected sidecar publish failure"

    with pytest.raises(mpi.RawCaptureSidecarError, match=message) as caught:
        mpi.write_immutable_raw_capture_atomic(
            tmp_path, metadata=metadata, arrays=arrays
        )

    raw_path = tmp_path / "raw_measurement.npz"
    assert caught.value.raw_path == raw_path
    assert raw_path.is_file()
    assert not (tmp_path / "metadata.json").exists()
    with np.load(raw_path, allow_pickle=False) as data:
        assert json.loads(str(data["metadata_json"])) == metadata
        np.testing.assert_array_equal(data["input_raw_int32"], arrays["input_raw_int32"])
    _assert_no_partial_files(tmp_path)


def test_existing_raw_capture_is_never_replaced(tmp_path):
    raw_path = tmp_path / "raw_measurement.npz"
    metadata_path = tmp_path / "metadata.json"
    raw_path.write_bytes(b"existing-raw")
    metadata_path.write_bytes(b"existing-metadata")
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (raw_path, metadata_path)
    }
    metadata, arrays = _raw_payload()

    with pytest.raises(FileExistsError, match="덮어쓰지"):
        mpi.write_immutable_raw_capture_atomic(
            tmp_path, metadata=metadata, arrays=arrays
        )

    for path in (raw_path, metadata_path):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before[path.name]
    _assert_no_partial_files(tmp_path)


def test_official_pair_success_message_lists_only_new_strict_paths(tmp_path):
    primary = tmp_path / "assets" / "primary_strict_new.npz"
    secondary = tmp_path / "assets" / "secondary_strict_new.npz"

    message = mpi.official_pair_success_message(
        primary,
        secondary,
        repository_root=tmp_path,
    )

    assert message.strip().splitlines() == [
        "[성공] strict primary NPZ: assets/primary_strict_new.npz",
        "       strict secondary NPZ: assets/secondary_strict_new.npz",
    ]
    assert "d_noise_delay_samples" not in message
    assert "digital_reference_lead_samples" not in message
    assert "duct.yaml" not in message


def test_official_pair_is_exposed_only_after_both_temps_are_complete(tmp_path):
    primary = tmp_path / "p.npz"
    secondary = tmp_path / "s.npz"

    mpi.write_official_pair_atomic(
        primary, {"channel": np.str_("noise")},
        secondary, {"channel": np.str_("cancel")},
    )

    with np.load(primary, allow_pickle=False) as data:
        assert str(data["channel"]) == "noise"
    with np.load(secondary, allow_pickle=False) as data:
        assert str(data["channel"]) == "cancel"
    _assert_no_partial_files(tmp_path)


@pytest.mark.parametrize("fail_on_write", [1, 2])
def test_official_pair_write_failure_exposes_neither_member(
    tmp_path, monkeypatch, fail_on_write
):
    original = mpi.np.savez
    calls = 0

    def injected(handle, **arrays):
        nonlocal calls
        calls += 1
        if calls == fail_on_write:
            handle.write(b"truncated")
            raise OSError("injected official write failure")
        return original(handle, **arrays)

    monkeypatch.setattr(mpi.np, "savez", injected)
    with pytest.raises(OSError, match="injected official write failure"):
        mpi.write_official_pair_atomic(
            tmp_path / "p.npz", {"x": np.arange(3)},
            tmp_path / "s.npz", {"x": np.arange(4)},
        )

    assert not (tmp_path / "p.npz").exists()
    assert not (tmp_path / "s.npz").exists()
    _assert_no_partial_files(tmp_path)


@pytest.mark.parametrize("fail_on_publish", [1, 2])
def test_official_pair_noreplace_failure_removes_promoted_orphan(
    tmp_path, monkeypatch, fail_on_publish
):
    original = mpi.atomic_publish_noreplace
    calls = 0

    def injected(source, target):
        nonlocal calls
        calls += 1
        if calls == fail_on_publish:
            raise OSError("injected official no-replace publish failure")
        return original(source, target)

    monkeypatch.setattr(mpi, "atomic_publish_noreplace", injected)
    with pytest.raises(OSError, match="injected official no-replace publish failure"):
        mpi.write_official_pair_atomic(
            tmp_path / "p.npz", {"x": np.arange(3)},
            tmp_path / "s.npz", {"x": np.arange(4)},
        )

    assert not (tmp_path / "p.npz").exists()
    assert not (tmp_path / "s.npz").exists()
    _assert_no_partial_files(tmp_path)


def test_official_pair_racing_target_is_never_overwritten(tmp_path, monkeypatch):
    primary = tmp_path / "p.npz"
    secondary = tmp_path / "s.npz"
    original_link = mpi.os.link

    def racing_link(source, target):
        target = Path(target)
        if target == primary:
            target.write_bytes(b"racing owner")
        return original_link(source, target)

    monkeypatch.setattr(mpi.os, "link", racing_link)
    with pytest.raises(FileExistsError, match="race-safe publish"):
        mpi.write_official_pair_atomic(
            primary, {"x": np.arange(3)},
            secondary, {"x": np.arange(4)},
        )

    assert primary.read_bytes() == b"racing owner"
    assert not secondary.exists()
    _assert_no_partial_files(tmp_path)
