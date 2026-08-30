from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

import deep_anc.data.recording_level_campaign as campaign
import deep_anc.data.recorded_generation as recorded_generation
from deep_anc.dsp.measurement_level import (
    ALSA_PHYSICAL_FINGERPRINT_SCHEMA,
    BOOTSTRAP_METER_RAW_SCHEMA,
    OFFICIAL_MEASUREMENT_LEVEL,
    expected_meter_output_pcm,
    measurement_hardware_identity,
    meter_raw_level_dbfs,
    write_bootstrap_meter_raw_atomic,
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
        "channels": {
            "error_mic": 0,
            "reference_mic": 1,
            "noise_out": 0,
            "cancel_out": 1,
        },
    }


def _physical_fingerprint() -> dict:
    def endpoint(card: str, pcm: int, stream: str, device: str) -> dict:
        return {
            "configured_card_id": card,
            "proc_card_id": card,
            "pcm_device": pcm,
            "pcm_stream": stream,
            "pcm_info": {
                "device": str(pcm),
                "stream": stream,
                "id": f"{card} PCM",
                "name": f"{card} codec",
                "subname": "subdevice #0",
                "class": "0",
                "subclass": "0",
                "subdevices_count": "1",
            },
            "sys_device_realpath": device,
            "sys_device_uevent": {"DRIVER": f"driver-{card.lower()}"},
            "stable_attributes": [],
        }

    value = {
        "schema": ALSA_PHYSICAL_FINGERPRINT_SCHEMA,
        "input": endpoint("APE", 1, "CAPTURE", "devices/platform/sound"),
        "output": endpoint("Audio", 0, "PLAYBACK", "devices/usb/1-1:1.0"),
    }
    value["output"]["stable_attributes"] = [
        {
            "sys_relative_path": "devices/usb/1-1",
            "values": {"serial": "dac-recording-level-test"},
        }
    ]
    value["sha256"] = hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return value


def _int32(values: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(values, -1.0, 1.0) * (2**31 - 1)).astype(np.int32)


def _fixture(root: Path) -> dict[str, object]:
    (root / "results/meter").mkdir(parents=True)
    (root / "configs").mkdir()
    hardware = _hardware()
    hardware_path = root / "configs/hardware.yaml"
    hardware_path.write_text(
        yaml.safe_dump(hardware, sort_keys=True), encoding="utf-8"
    )
    identity = measurement_hardware_identity(
        hardware,
        physical_fingerprint=_physical_fingerprint(),
    )
    completed = dt.datetime(2026, 8, 30, 7, 0, 0, tzinfo=dt.timezone.utc)
    started = completed - dt.timedelta(seconds=20)
    frames = int(
        OFFICIAL_MEASUREMENT_LEVEL.sample_rate
        * OFFICIAL_MEASUREMENT_LEVEL.meter_seconds
    )
    time = np.arange(frames, dtype=np.float64) / 48_000.0
    signal = (
        np.sqrt(2.0)
        * 10.0 ** (OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs / 20.0)
        * np.sin(2.0 * np.pi * 500.0 * time)
    )
    input_raw = np.column_stack((_int32(signal), _int32(0.8 * signal)))
    meter_level = meter_raw_level_dbfs(input_raw, error_channel=0)
    metadata = {
        "schema": BOOTSTRAP_METER_RAW_SCHEMA,
        "capture_id": "0123456789abcdef0123456789abcdef",
        "status": "PASS",
        "passed": True,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
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
            "output_frames": frames,
            "xrun_count": 0,
            "unexpected_status_count": 0,
            "callback_error": None,
            "stream_abort_error": None,
            "stream_close_error": None,
            "output_stop_confirmed": True,
            "meter_drop_count": 0,
            "termination_signal": None,
            "stream_started_at_utc": started.isoformat(),
            "nominal_output_seconds": 20.0,
            "hard_max_output_seconds": 21.0,
        },
        "invalid_reasons": [],
    }
    meter = write_bootstrap_meter_raw_atomic(
        root / "results/meter/meter_raw.npz",
        repository_root=root,
        metadata=metadata,
        submitted_output_pcm_int16=expected_meter_output_pcm(noise_channel=0),
        input_raw_int32=input_raw,
    )
    return {
        "raw": meter["raw"].relative_to(root).as_posix(),
        "receipt": meter["receipt"].relative_to(root).as_posix(),
        "hardware": hardware_path.relative_to(root).as_posix(),
        "completed": completed,
        "now": completed + dt.timedelta(seconds=30),
        "identity": identity,
    }


def _issue(root: Path, fixture: dict[str, object]) -> dict:
    return campaign.issue_recording_level_campaign(
        repo_root=root,
        meter_raw=str(fixture["raw"]),
        meter_receipt=str(fixture["receipt"]),
        hardware_config=str(fixture["hardware"]),
        now_utc=fixture["now"],
    )


def _rendered_source() -> np.ndarray:
    time = np.arange(campaign.RECORDING_LEVEL_SESSION_FRAMES) / 48_000.0
    return (0.06 * np.sin(2.0 * np.pi * 500.0 * time)).astype(np.float32)


def test_issue_validate_and_build_small_session_binding(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    summary = _issue(tmp_path, fixture)
    payload = summary["payload"]
    assert payload["meter"]["probe_peak"] == 0.003
    assert payload["meter"]["target_dbfs"] == -50.1
    assert payload["meter"]["tolerance_db"] == 2.0
    assert payload["meter"]["age_seconds_at_issue"] == 30.0
    assert payload["hardware"]["physical_fingerprint_sha256"] == (
        fixture["identity"]["physical_fingerprint"]["sha256"]
    )
    assert Path(summary["receipt_path"]).parts[-2] == summary["campaign_id"]

    verified = campaign.validate_recording_level_campaign(
        repo_root=tmp_path,
        campaign_receipt=summary["receipt_path"],
        expected_sha256=summary["receipt_sha256"],
        now_utc=fixture["now"],
    )
    rendered = campaign.rendered_source_level_evidence(_rendered_source())
    started = fixture["completed"] + dt.timedelta(seconds=120)
    binding = campaign.build_recording_level_session_binding(
        verified,
        session_started_at_utc=started,
        same_amplifier_setting=True,
        rendered_source=rendered,
    )
    assert binding["meter_age_seconds_at_session_start"] == 120.0
    assert binding["rendered_source"]["peak_linear"] == pytest.approx(0.06)
    assert campaign.validate_recording_level_session_binding(verified, binding) == binding
    assert binding["rendered_source"]["sample_encoding"] == "float32_le"
    assert len(binding["rendered_source"]["sample_sha256"]) == 64


def test_session_binding_rejects_start_before_campaign_issue(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    verified = _issue(tmp_path, fixture)
    rendered = campaign.rendered_source_level_evidence(_rendered_source())
    # meter는 끝났지만 campaign은 completed+30초에 발행됐다. +10초 session을
    # 허용하면 사후 발행한 receipt로 과거 raw를 승격할 수 있다.
    started = fixture["completed"] + dt.timedelta(seconds=10)
    with pytest.raises(
        campaign.RecordingLevelCampaignError,
        match="campaign 발행 시각보다 빠를 수 없습니다",
    ):
        campaign.build_recording_level_session_binding(
            verified,
            session_started_at_utc=started,
            same_amplifier_setting=True,
            rendered_source=rendered,
        )


def test_generation_recomputes_historical_binding_from_actual_source_samples(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    verified = _issue(tmp_path, fixture)
    source = _rendered_source()
    started = fixture["completed"] + dt.timedelta(seconds=120)
    binding = campaign.build_recording_level_session_binding(
        verified,
        session_started_at_utc=started,
        same_amplifier_setting=True,
        rendered_source=campaign.rendered_source_level_evidence(source),
    )
    summary = recorded_generation._validate_recording_level_binding(
        metadata={"recording_level_binding": binding},
        rendered_source=source,
        session_dir=tmp_path / "data/recorded_additions/session-fixture",
        repo_root=tmp_path,
    )
    assert summary["campaign_id"] == verified["campaign_id"]
    assert summary["campaign"]["sha256"] == verified["receipt_sha256"]

    changed = source.copy()
    changed[12345] += np.float32(1e-4)
    with pytest.raises(
        recorded_generation.RecordedGenerationError,
        match="sample SHA",
    ):
        recorded_generation._validate_recording_level_binding(
            metadata={"recording_level_binding": binding},
            rendered_source=changed,
            session_dir=tmp_path / "data/recorded_additions/session-fixture",
            repo_root=tmp_path,
        )


def test_campaign_publish_is_no_replace(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = _issue(tmp_path, fixture)
    before = (tmp_path / first["receipt_path"]).read_bytes()
    with pytest.raises(campaign.RecordingLevelCampaignError, match="덮어쓰지"):
        _issue(tmp_path, fixture)
    assert (tmp_path / first["receipt_path"]).read_bytes() == before


@pytest.mark.parametrize("target", ("raw", "receipt", "hardware", "campaign"))
def test_campaign_rejects_source_or_receipt_tamper(
    tmp_path: Path, target: str
) -> None:
    fixture = _fixture(tmp_path)
    summary = _issue(tmp_path, fixture)
    if target == "campaign":
        path = tmp_path / summary["receipt_path"]
        payload = json.loads(path.read_text())
        payload["meter"]["probe_peak"] = 0.004
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif target == "hardware":
        path = tmp_path / str(fixture["hardware"])
        path.write_text(path.read_text() + "\nextra: changed\n", encoding="utf-8")
    elif target == "receipt":
        path = tmp_path / str(fixture["receipt"])
        path.write_text('{"tampered":true}', encoding="utf-8")
    else:
        path = tmp_path / str(fixture["raw"])
        path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(campaign.RecordingLevelCampaignError):
        campaign.validate_recording_level_campaign(
            repo_root=tmp_path,
            campaign_receipt=summary["receipt_path"],
            now_utc=fixture["now"],
            require_fresh=False,
        )


def test_campaign_and_session_freshness_are_independent_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    stale = fixture["completed"] + dt.timedelta(
        seconds=campaign.RECORDING_LEVEL_MAX_AGE_SECONDS + 1
    )
    with pytest.raises(campaign.RecordingLevelCampaignError, match="freshness"):
        campaign.build_recording_level_campaign_payload(
            repo_root=tmp_path,
            meter_raw=str(fixture["raw"]),
            meter_receipt=str(fixture["receipt"]),
            hardware_config=str(fixture["hardware"]),
            now_utc=stale,
        )

    summary = _issue(tmp_path, fixture)
    rendered = campaign.rendered_source_level_evidence(_rendered_source())
    for started in (
        fixture["completed"] - dt.timedelta(microseconds=1),
        stale,
    ):
        with pytest.raises(campaign.RecordingLevelCampaignError, match="meter age"):
            campaign.build_recording_level_session_binding(
                summary,
                session_started_at_utc=started,
                same_amplifier_setting=True,
                rendered_source=rendered,
            )
    with pytest.raises(campaign.RecordingLevelCampaignError, match="same_amplifier"):
        campaign.build_recording_level_session_binding(
            summary,
            session_started_at_utc=fixture["now"] + dt.timedelta(seconds=1),
            same_amplifier_setting=False,
            rendered_source=rendered,
        )


def test_rendered_source_rejects_nonfinite_and_unsafe_peak(tmp_path: Path) -> None:
    del tmp_path
    source = _rendered_source()
    source[100] = np.nan
    with pytest.raises(campaign.RecordingLevelCampaignError, match="non-finite"):
        campaign.rendered_source_level_evidence(source)
    source = _rendered_source()
    source *= 2.0
    with pytest.raises(campaign.RecordingLevelCampaignError, match="peak"):
        campaign.rendered_source_level_evidence(source)


def test_symlink_inputs_and_campaign_receipt_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    alias = tmp_path / "results/meter/alias_raw.npz"
    alias.symlink_to(tmp_path / str(fixture["raw"]))
    alias_receipt = alias.with_name("alias_raw.receipt.json")
    alias_receipt.symlink_to(tmp_path / str(fixture["receipt"]))
    with pytest.raises(campaign.RecordingLevelCampaignError, match="symlink"):
        campaign.build_recording_level_campaign_payload(
            repo_root=tmp_path,
            meter_raw=alias,
            meter_receipt=alias_receipt,
            hardware_config=str(fixture["hardware"]),
            now_utc=fixture["now"],
        )

    summary = _issue(tmp_path, fixture)
    real = tmp_path / summary["receipt_path"]
    alias_campaign = real.parent / "alias.json"
    alias_campaign.symlink_to(real)
    with pytest.raises(campaign.RecordingLevelCampaignError, match="symlink"):
        campaign.validate_recording_level_campaign(
            repo_root=tmp_path,
            campaign_receipt=alias_campaign,
            now_utc=fixture["now"],
        )


def test_snapshot_race_after_meter_validation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    real = campaign.validate_bootstrap_meter_raw

    def racing_validator(*args, **kwargs):  # noqa: ANN002, ANN003
        result = real(*args, **kwargs)
        hardware = tmp_path / str(fixture["hardware"])
        hardware.write_text(hardware.read_text() + "\nrace: changed\n", encoding="utf-8")
        return result

    monkeypatch.setattr(campaign, "validate_bootstrap_meter_raw", racing_validator)
    with pytest.raises(campaign.RecordingLevelCampaignError, match="변경/retarget"):
        campaign.build_recording_level_campaign_payload(
            repo_root=tmp_path,
            meter_raw=str(fixture["raw"]),
            meter_receipt=str(fixture["receipt"]),
            hardware_config=str(fixture["hardware"]),
            now_utc=fixture["now"],
        )


def test_campaign_issuer_help_does_not_open_audio() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(root / "scripts/data/issue_recording_level_campaign.py"),
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--meter-raw" in result.stdout
    assert "--write" in result.stdout
