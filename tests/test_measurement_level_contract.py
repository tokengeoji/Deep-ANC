"""P/S 측정과 앰프 미터의 공용 레벨 계약 회귀 테스트."""

from __future__ import annotations

import hashlib
import json
import datetime as dt
import os
import signal
from pathlib import Path

import numpy as np
import pytest

from deep_anc import audio_io
from deep_anc.audio_io import float32_to_pcm_int16
from deep_anc.dsp.measurement_level import (
    ALSA_PHYSICAL_FINGERPRINT_SCHEMA,
    BOOTSTRAP_METER_RAW_SCHEMA,
    BOOTSTRAP_METER_MAX_AGE_SECONDS,
    LiveAudioTermination,
    MEASUREMENT_LEVEL_EVIDENCE_SCHEMA,
    MIN_MEASUREMENT_CPU_IDLE_FRACTION,
    OFFICIAL_MEASUREMENT_LEVEL,
    assert_measurement_cpu_idle,
    assert_live_pcm_clock_preconditions,
    atomic_publish_noreplace,
    collect_alsa_physical_fingerprint,
    create_measurement_level_evidence_atomic,
    interleaved_err_noise_bin_dbfs,
    load_measurement_level_evidence,
    measurement_hardware_identity,
    measurement_cpu_idle_fraction,
    meter_raw_level_dbfs,
    repository_audio_lock,
    scoped_live_audio_signal_handlers,
    validate_bootstrap_meter_raw,
    write_bootstrap_meter_raw_atomic,
)
from deep_anc.dsp.interleaved_probe import build_interleaved_probe
from scripts.data import measure_paths_interleaved as measure_paths
from scripts.data import set_amp_level


def test_measurement_and_meter_share_the_exact_contract_instance():
    assert measure_paths.OFFICIAL_MEASUREMENT_LEVEL is OFFICIAL_MEASUREMENT_LEVEL
    assert set_amp_level.OFFICIAL_MEASUREMENT_LEVEL is OFFICIAL_MEASUREMENT_LEVEL

    args = measure_paths.build_parser().parse_args(["--dry-run"])
    assert args.amplitude == pytest.approx(0.003)
    assert args.amplitude == pytest.approx(OFFICIAL_MEASUREMENT_LEVEL.probe_amplitude)
    assert args.period_seconds == pytest.approx(
        OFFICIAL_MEASUREMENT_LEVEL.period_seconds
    )
    assert tuple(args.band) == OFFICIAL_MEASUREMENT_LEVEL.design_band_hz
    assert args.warmup_periods == OFFICIAL_MEASUREMENT_LEVEL.warmup_periods == 32
    assert args.repeats == OFFICIAL_MEASUREMENT_LEVEL.analysis_repeats == 64
    assert (
        args.input_probe_seconds
        == OFFICIAL_MEASUREMENT_LEVEL.input_probe_seconds
        == 3.0
    )
    assert OFFICIAL_MEASUREMENT_LEVEL.meter_seconds == 20.0


def test_meter_probe_uses_the_official_interleaved_peak():
    signal = set_amp_level.probe_signal(
        OFFICIAL_MEASUREMENT_LEVEL.period_seconds
    )

    assert signal.dtype == np.float32
    assert signal.size == round(
        OFFICIAL_MEASUREMENT_LEVEL.sample_rate
        * OFFICIAL_MEASUREMENT_LEVEL.period_seconds
    )
    assert float(np.max(np.abs(signal))) == pytest.approx(
        OFFICIAL_MEASUREMENT_LEVEL.probe_amplitude,
        rel=1e-6,
    )


def test_meter_verdict_uses_contract_bounds():
    assert "올리세요" in set_amp_level.verdict(
        OFFICIAL_MEASUREMENT_LEVEL.meter_min_dbfs - 0.1
    )
    assert "맞았습니다" in set_amp_level.verdict(
        OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs
    )
    assert "내리세요" in set_amp_level.verdict(
        OFFICIAL_MEASUREMENT_LEVEL.meter_max_dbfs + 0.1
    )


def test_meter_verdict_exposes_exact_boundary_margin_without_relaxing_gate():
    maximum = OFFICIAL_MEASUREMENT_LEVEL.meter_max_dbfs

    at_boundary = set_amp_level.verdict(maximum)
    just_outside = set_amp_level.verdict(np.nextafter(maximum, np.inf))

    assert "맞았습니다" in at_boundary
    assert "경계 여유 0.0000 dB" in at_boundary
    assert "내리세요" in just_outside
    assert f"상한 {maximum:+.4f}" in just_outside
    assert "목표 대비 +2.0 dB" in just_outside


def _hardware():
    return {
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
        "schema": ALSA_PHYSICAL_FINGERPRINT_SCHEMA,
        "input": endpoint("APE", 1, "CAPTURE", "devices/platform/sound"),
        "output": endpoint("Audio", 0, "PLAYBACK", "devices/usb/1-1:1.0"),
    }
    payload["output"]["stable_attributes"] = [
        {
            "sys_relative_path": "devices/usb/1-1",
            "values": {"serial": output_serial},
        }
    ]
    payload["sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _write_proc_pcm_info(path: Path, *, card: int, device: int, stream: str) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            (
                f"card: {card}",
                f"device: {device}",
                "subdevice: 0",
                f"stream: {stream}",
                "id: test pcm",
                "name: test codec",
                "subname: subdevice #0",
                "class: 0",
                "subclass: 0",
                "subdevices_count: 1",
                "subdevices_avail: 1",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_physical_fingerprint_binds_proc_pcm_sysfs_and_usb_serial(tmp_path):
    proc = tmp_path / "proc/asound"
    sys_root = tmp_path / "sys"
    sys_class = sys_root / "class/sound"
    for index, card_id in ((1, "APE"), (2, "Audio")):
        card = proc / f"card{index}"
        card.mkdir(parents=True)
        (card / "id").write_text(card_id + "\n", encoding="utf-8")
    _write_proc_pcm_info(proc / "card1/pcm1c/info", card=1, device=1, stream="CAPTURE")
    _write_proc_pcm_info(proc / "card2/pcm0p/info", card=2, device=0, stream="PLAYBACK")

    input_device = sys_root / "devices/platform/sound"
    output_parent = sys_root / "devices/usb/1-1"
    output_device = output_parent / "1-1:1.0"
    input_device.mkdir(parents=True)
    output_device.mkdir(parents=True)
    (input_device / "uevent").write_text(
        "DRIVER=tegra-asoc\nOF_FULLNAME=/sound\n", encoding="utf-8"
    )
    (output_device / "uevent").write_text(
        "DRIVER=snd-usb-audio\nPRODUCT=1f/b21/100\nDEVNUM=9\n",
        encoding="utf-8",
    )
    (output_parent / "serial").write_text("physical-dac-A\n", encoding="utf-8")
    for index, target in ((1, input_device), (2, output_device)):
        card_class = sys_class / f"card{index}"
        card_class.mkdir(parents=True)
        (card_class / "device").symlink_to(target)

    first = collect_alsa_physical_fingerprint(
        _hardware(),
        proc_asound_root=proc,
        sys_class_sound_root=sys_class,
        sys_root=sys_root,
    )
    assert first["input"]["sys_device_realpath"] == "devices/platform/sound"
    assert first["output"]["sys_device_uevent"]["PRODUCT"] == "1f/b21/100"
    assert "DEVNUM" not in first["output"]["sys_device_uevent"]
    assert first["output"]["stable_attributes"][0]["values"]["serial"] == "physical-dac-A"

    (output_parent / "serial").write_text("physical-dac-B\n", encoding="utf-8")
    second = collect_alsa_physical_fingerprint(
        _hardware(),
        proc_asound_root=proc,
        sys_class_sound_root=sys_class,
        sys_root=sys_root,
    )
    assert second != first
    assert second["sha256"] != first["sha256"]

    (output_device / "uevent").unlink()
    with pytest.raises(FileNotFoundError, match="uevent"):
        collect_alsa_physical_fingerprint(
            _hardware(),
            proc_asound_root=proc,
            sys_class_sound_root=sys_class,
            sys_root=sys_root,
        )


def test_immediate_live_gate_is_read_only_pcm_then_clock(monkeypatch):
    events = []
    monkeypatch.setattr(
        audio_io,
        "assert_measurement_pcm_unoccupied",
        lambda hardware: events.append(("pcm", hardware)),
    )
    monkeypatch.setattr(
        audio_io,
        "assert_capture_clock_undisturbed",
        lambda card: events.append(("clock", card)),
    )
    monkeypatch.setattr(
        "deep_anc.dsp.measurement_level.assert_measurement_cpu_idle",
        lambda: events.append(("cpu_idle", None)),
    )

    hardware = _hardware()["audio"]
    assert_live_pcm_clock_preconditions(hardware)

    assert events == [
        ("pcm", hardware),
        ("cpu_idle", None),
        ("clock", "APE"),
    ]


def test_cpu_idle_fraction_uses_monotonic_aggregate_jiffies(tmp_path):
    proc_stat = tmp_path / "stat"
    proc_stat.write_text("cpu  10 2 3 70 5 1 1 0 0 0\n", encoding="utf-8")

    def advance(_seconds):
        # total delta=100, true idle delta=60. iowait delta=10 is deliberately
        # not counted as usable measurement CPU time.
        proc_stat.write_text("cpu  20 2 13 130 15 6 6 0 0 0\n", encoding="utf-8")

    observed = measurement_cpu_idle_fraction(
        proc_stat_path=proc_stat,
        sample_seconds=0.01,
        sleep_fn=advance,
    )
    assert observed == pytest.approx(0.60)


def test_cpu_idle_gate_fails_closed_below_contract(monkeypatch):
    monkeypatch.setattr(
        "deep_anc.dsp.measurement_level.measurement_cpu_idle_fraction",
        lambda: MIN_MEASUREMENT_CPU_IDLE_FRACTION - 0.01,
    )
    with pytest.raises(RuntimeError, match="CPU 유휴율이 부족"):
        assert_measurement_cpu_idle()


def _int32(signal):
    return np.rint(np.clip(signal, -1.0, 1.0) * (2**31 - 1)).astype(np.int32)


def _write_real_evidence_pair(tmp_path):
    contract = OFFICIAL_MEASUREMENT_LEVEL
    identity = measurement_hardware_identity(
        _hardware(), physical_fingerprint=_physical_fingerprint()
    )
    now = dt.datetime.now(dt.timezone.utc)

    meter_frames = int(contract.sample_rate * contract.meter_seconds)
    meter_time = np.arange(meter_frames, dtype=np.float64) / contract.sample_rate
    meter_signal = np.sqrt(2.0) * 10 ** (contract.meter_target_dbfs / 20.0) * np.sin(
        2.0 * np.pi * 500.0 * meter_time
    )
    meter_input = np.column_stack((_int32(meter_signal), _int32(meter_signal * 0.8)))
    meter_level = meter_raw_level_dbfs(meter_input, error_channel=0)
    meter_output = np.zeros((meter_frames, 2), dtype=np.int16)
    meter_output[:, 0] = float32_to_pcm_int16(
        set_amp_level.probe_signal(contract.meter_seconds)
    )
    meter_metadata = {
        "schema": BOOTSTRAP_METER_RAW_SCHEMA,
        "capture_id": "meter-test",
        "status": "PASS",
        "passed": True,
        "started_at_utc": (now - dt.timedelta(seconds=55)).isoformat(),
        "completed_at_utc": (now - dt.timedelta(seconds=35)).isoformat(),
        "hardware_identity": identity,
        "resolved_devices": {"input": 1, "output": 2},
        "operator_confirmations": {
            "speaker_output": True,
            "user_present": True,
            "volume_minimum_before_start": True,
        },
        "recipe": {
            "sample_rate": 48_000,
            "block_size": 256,
            "latency": "low",
            "seconds": 20.0,
            "probe_amplitude": 0.003,
            "period_seconds": 0.125,
            "design_band_hz": [60.0, 1650.0],
            "meter_band_hz": [150.0, 1600.0],
            "meter_target_dbfs": -50.1,
            "meter_tolerance_db": 2.0,
            "noise_output_channel": 0,
            "cancel_output_silent": True,
        },
        "meter_ch0_dbfs": meter_level,
        "telemetry": {
            "completed": True,
            "interrupted": False,
            "output_frames": meter_frames,
            "xrun_count": 0,
            "unexpected_status_count": 0,
            "callback_error": None,
            "stream_abort_error": None,
            "stream_close_error": None,
            "output_stop_confirmed": True,
            "meter_drop_count": 0,
            "termination_signal": None,
            "stream_started_at_utc": (now - dt.timedelta(seconds=55)).isoformat(),
            "nominal_output_seconds": 20.0,
            "hard_max_output_seconds": 21.0,
        },
        "invalid_reasons": [],
    }
    meter_paths = write_bootstrap_meter_raw_atomic(
        tmp_path / "meter_raw.npz",
        repository_root=tmp_path,
        metadata=meter_metadata,
        submitted_output_pcm_int16=meter_output,
        input_raw_int32=meter_input,
    )

    probe = build_interleaved_probe(
        sample_rate=48_000,
        period_seconds=0.125,
        band_hz=(60.0, 1650.0),
        amplitude=0.003,
        tone_spacing_hz=None,
    )
    lead = 24_000
    periods = 32 + 64
    total = lead + periods * probe.period_samples
    output_float = np.zeros((total, 2), dtype=np.float32)
    output_float[lead:, 0] = np.tile(probe.noise_signal, periods)
    output_float[lead:, 1] = np.tile(probe.cancel_signal, periods)
    input_float = np.zeros((total, 2), dtype=np.float64)
    base = np.tile(probe.noise_signal.astype(np.float64), periods)
    current_rms = float(np.sqrt(np.mean(probe.noise_signal.astype(np.float64) ** 2)))
    scale = 10 ** (contract.interleaved_err_noise_bin_dbfs / 20.0) / current_rms
    input_float[lead:, 0] = base * scale
    input_float[lead:, 1] = base * scale * 0.8
    # lead-in도 stuck-line 판정을 피할 작은 실제 입력으로 채운다.
    lead_time = np.arange(lead, dtype=np.float64) / contract.sample_rate
    input_float[:lead, 0] = 1e-4 * np.sin(2 * np.pi * 500 * lead_time)
    input_float[:lead, 1] = 8e-5 * np.sin(2 * np.pi * 700 * lead_time)
    strict_input = _int32(input_float)
    strict_level = interleaved_err_noise_bin_dbfs(
        strict_input,
        error_channel=0,
        lead_in_samples=lead,
        warmup_periods=32,
        repeats=64,
    )
    assert strict_level == pytest.approx(contract.interleaved_err_noise_bin_dbfs, abs=0.1)
    strict_metadata = {
        "capture_id": "strict-test",
        "method": "interleaved_multitone",
        "raw_capture_schema": (
            "interleaved_raw_v4_user_present_observed_pcm_preanalysis"
        ),
        # session/preflight 시작 시각은 실제 output stream 시작과 의도적으로 다르다.
        "started_at_utc": (now - dt.timedelta(seconds=5)).isoformat(),
        "sample_rate": 48_000,
        "block_size": 256,
        "latency": "low",
        "hardware_identity": identity,
        "resolved_devices": {"input": 1, "output": 2},
        "channel_map": dict(identity["channel_map"]),
        "operator_confirmations": {
            "user_present": True,
            "volume_minimum": True,
            "routing_and_geometry": True,
        },
        "amplitude": 0.003,
        "period_seconds": 0.125,
        "warmup_periods": 32,
        "repeats": 64,
        "lead_in_samples": lead,
        "telemetry": {
            "completed": True,
            "captured_frames": total,
            "xrun_count": 0,
            "unexpected_status_count": 0,
            "callback_error": None,
            "stream_abort_error": None,
            "stream_close_error": None,
            "output_stop_confirmed": True,
            "termination_signal": None,
            "stream_started_at_utc": (now - dt.timedelta(seconds=30)).isoformat(),
            "nominal_output_seconds": 12.5,
            "hard_max_output_seconds": 13.5,
        },
        "invalid_reasons": [],
        "measurement_level_bootstrap": {
            "enabled": True,
            "same_amplifier_setting_confirmed": True,
        },
    }
    strict_path = tmp_path / "strict_raw.npz"
    np.savez_compressed(
        strict_path,
        metadata_json=np.asarray(json.dumps(strict_metadata, sort_keys=True)),
        output_pcm_int16=float32_to_pcm_int16(output_float),
        input_raw_int32=strict_input,
    )
    evidence_path = tmp_path / "evidence.json"
    create_measurement_level_evidence_atomic(
        evidence_path,
        repository_root=tmp_path,
        meter_raw_path=meter_paths["raw"],
        interleaved_raw_path=strict_path,
        hardware_identity=identity,
    )
    return evidence_path, meter_paths["raw"]


def test_level_evidence_requires_two_preserved_raw_sha(tmp_path):
    path, _meter = _write_real_evidence_pair(tmp_path)

    payload = load_measurement_level_evidence(path, repository_root=tmp_path)

    assert payload["passed"] is True
    # metadata.started_at_utc가 아니라 실제 stream-start telemetry에 결박된다.
    assert payload["capture_gap_seconds"] == pytest.approx(5.0)


def test_level_evidence_fails_closed_when_missing_or_raw_tampered(tmp_path):
    with pytest.raises(FileNotFoundError, match="paired raw 증거가 없습니다"):
        load_measurement_level_evidence(
            tmp_path / "missing.json", repository_root=tmp_path
        )

    path, meter = _write_real_evidence_pair(tmp_path)
    meter.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA 불일치"):
        load_measurement_level_evidence(path, repository_root=tmp_path)


def test_bootstrap_meter_freshness_and_device_identity_are_fail_closed(tmp_path):
    path, meter = _write_real_evidence_pair(tmp_path)
    payload = load_measurement_level_evidence(path, repository_root=tmp_path)
    completed = dt.datetime.fromisoformat(payload["meter_raw"]["completed_at_utc"])

    with pytest.raises(ValueError, match="freshness"):
        validate_bootstrap_meter_raw(
            meter,
            repository_root=tmp_path,
            expected_hardware_identity=payload["hardware_identity"],
            now_utc=completed
            + dt.timedelta(seconds=BOOTSTRAP_METER_MAX_AGE_SECONDS + 1),
        )

    wrong_identity = json.loads(json.dumps(payload["hardware_identity"]))
    wrong_identity["output"]["pcm"] += 1
    with pytest.raises(ValueError, match="device/channel identity"):
        validate_bootstrap_meter_raw(
            meter,
            repository_root=tmp_path,
            expected_hardware_identity=wrong_identity,
            require_fresh=False,
        )


def test_atomic_publish_noreplace_preserves_racing_target(tmp_path, monkeypatch):
    temporary = tmp_path / ".evidence.partial"
    target = tmp_path / "evidence.json"
    temporary.write_bytes(b"validated evidence")
    original_link = os.link

    def racing_link(source, destination):
        Path(destination).write_bytes(b"racing owner")
        return original_link(source, destination)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError, match="race-safe publish"):
        atomic_publish_noreplace(temporary, target)

    assert target.read_bytes() == b"racing owner"
    assert temporary.read_bytes() == b"validated evidence"


def test_repository_uid_audio_lock_rejects_concurrent_holder(tmp_path):
    (tmp_path / "results").mkdir()
    with repository_audio_lock(tmp_path, purpose="first") as held:
        assert held["uid"] == os.getuid()
        with pytest.raises(RuntimeError, match="이미 실행 중"):
            with repository_audio_lock(tmp_path, purpose="second"):
                pass


def test_live_signal_handlers_are_scoped_and_include_hup():
    signum = getattr(signal, "SIGHUP", signal.SIGTERM)
    original = signal.getsignal(signum)

    with pytest.raises(LiveAudioTermination) as caught:
        with scoped_live_audio_signal_handlers():
            handler = signal.getsignal(signum)
            assert callable(handler)
            handler(signum, None)

    assert caught.value.signum == int(signum)
    assert signal.getsignal(signum) == original
