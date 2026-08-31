from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from deep_anc.audio_duplex_stage2 import (
    DUPLEX_TELEMETRY_SCHEMA,
    DuplexCaptureFailure,
    capture_duplex_stage2,
)
from deep_anc.dsp.stage2_2khz_analysis import (
    analyse_stage2_raw_bytes,
    estimate_shared_affine_q,
)
from deep_anc.dsp.stage2_2khz_live import (
    load_stage2_raw_bytes,
    publish_meter_raw_no_replace,
    publish_stage2_failure_raw_no_replace,
    publish_stage2_signal_raw_no_replace,
)
from deep_anc.dsp.stage2_2khz_measurement import build_stage2_measurement_plan
from deep_anc.dsp.stage2_2khz_measurement import Stage2MeasurementError
from deep_anc.dsp.measurement_level import expected_meter_output_pcm

from test_audio_duplex_v5 import Backend
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_physical_capture(submitted: np.ndarray) -> np.ndarray:
    source = submitted.astype(np.float64) / 32768.0
    output = np.zeros_like(source)
    for path, delay in enumerate((700, 520)):
        gains = (0.8, 0.35) if path == 0 else (0.28, 0.75)
        for microphone, gain in enumerate(gains):
            impulse = np.zeros(delay + 64, dtype=np.float64)
            impulse[delay] = gain
            impulse[delay + 3] = -0.13 * gain
            impulse[delay + 19] = 0.04 * gain
            output[:, microphone] += lfilter(impulse, [1.0], source[:, path])
    return np.clip(
        np.rint(output * 2147483648.0), -2147483648, 2147483647
    ).astype("<i4")


def _meter_capture() -> np.ndarray:
    frames = 20 * 48_000
    time = np.arange(frames, dtype=np.float64) / 48_000.0
    peak = 10.0 ** (-50.1 / 20.0) * np.sqrt(2.0)
    signal = peak * np.sin(2.0 * np.pi * 300.0 * time)
    raw = np.zeros((frames, 2), dtype="<i4")
    raw[:, 0] = np.rint(signal * 2147483648.0).astype("<i4")
    raw[:, 1] = raw[:, 0] // 2
    return raw


def test_mock_two_stream_cannot_publish_rank_deficient_compact_plant(tmp_path: Path) -> None:
    config = json.loads(
        (ROOT / "configs/stage2_2khz_measurement.json").read_text(encoding="utf-8")
    )
    plan, signal_pcm = build_stage2_measurement_plan(config)
    meter_pcm = expected_meter_output_pcm(noise_channel=0)
    _, meter_telemetry = capture_duplex_stage2(
        Backend(blocks=len(meter_pcm) // 256),
        submitted_pcm=meter_pcm,
        input_device=1,
        output_device=2,
    )
    _, signal_telemetry = capture_duplex_stage2(
        Backend(blocks=len(signal_pcm) // 256),
        submitted_pcm=signal_pcm,
        input_device=1,
        output_device=2,
    )
    assert meter_telemetry["schema"] == DUPLEX_TELEMETRY_SCHEMA
    assert signal_telemetry["schema"] == DUPLEX_TELEMETRY_SCHEMA
    metadata = {
        "capture_id": "stage2-mock-capture",
        "execution_identity": {
            "repository_commit": "a" * 40,
            "repository_dirty": False,
        },
        "clean_exact_commit": True,
        "same_amplifier_setting_meter_to_signal": True,
    }
    meter = publish_meter_raw_no_replace(
        str(tmp_path),
        plan,
        submitted_pcm=meter_pcm,
        captured_pcm=_meter_capture(),
        telemetry=meter_telemetry,
        capture_metadata={**metadata, "phase": "meter"},
    )
    assert meter["admission"]["passed"] is True
    signal = publish_stage2_signal_raw_no_replace(
        str(tmp_path),
        plan,
        submitted_pcm=signal_pcm,
        captured_pcm=_synthetic_physical_capture(signal_pcm),
        telemetry=signal_telemetry,
        capture_metadata={**metadata, "phase": "same_capture_ps"},
        meter_raw_sha256=meter["sha256"],
    )
    native_bytes = (tmp_path / signal["path"]).read_bytes()
    meter_bytes = (tmp_path / meter["path"]).read_bytes()
    loaded = load_stage2_raw_bytes(native_bytes)
    assert loaded["metadata"]["physical_acoustic_capture"] is True
    del meter_bytes
    with pytest.raises(Stage2MeasurementError, match="identifiability 실패.*fullband_PE"):
        analyse_stage2_raw_bytes(plan, native_bytes)


def test_shared_q_rejects_one_sample_insert_drop() -> None:
    config = json.loads(
        (ROOT / "configs/stage2_2khz_measurement.json").read_text(encoding="utf-8")
    )
    _, submitted = build_stage2_measurement_plan(config)
    captured = _synthetic_physical_capture(submitted)
    midpoint = len(captured) // 2
    captured[midpoint + 1 :] = captured[midpoint:-1]
    with np.testing.assert_raises(Stage2MeasurementError):
        estimate_shared_affine_q(submitted, captured)


def test_partial_meter_capture_is_preserved_but_never_promotable(
    tmp_path: Path,
) -> None:
    config = json.loads(
        (ROOT / "configs/stage2_2khz_measurement.json").read_text(encoding="utf-8")
    )
    plan, _ = build_stage2_measurement_plan(config)
    meter_pcm = expected_meter_output_pcm(noise_channel=0)
    with pytest.raises(DuplexCaptureFailure) as caught:
        capture_duplex_stage2(
            Backend(frames=128),
            submitted_pcm=meter_pcm,
            input_device=1,
            output_device=2,
        )
    published = publish_stage2_failure_raw_no_replace(
        str(tmp_path),
        plan,
        phase="meter",
        planned_pcm=meter_pcm,
        failure=caught.value,
        capture_metadata={"capture_id": "partial-meter"},
    )
    with np.load(tmp_path / published["path"], allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
    assert metadata["schema"] == "stage2_2khz_partial_capture_raw_v1"
    assert metadata["partial_capture_never_promotable"] is True
    assert metadata["capture_valid_frames"] == 0
