"""스피커 출력 전에 녹음 source의 timeline/level 필요조건을 검증한다."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from deep_anc.data import recording_source_preflight as preflight


def _continuous_source() -> np.ndarray:
    time = np.arange(preflight.SOURCE_PREFLIGHT_FRAMES, dtype=np.float64) / 48_000.0
    signal = (
        0.022 * np.sin(2.0 * np.pi * 260.0 * time)
        + 0.016 * np.sin(2.0 * np.pi * 720.0 * time + 0.3)
        + 0.012 * np.sin(2.0 * np.pi * 1320.0 * time + 0.7)
    )
    return signal.astype(np.float32)


def test_timeline_feasibility_uses_estimator_source_span_and_passes_continuous():
    evidence = preflight.timeline_source_feasibility(_continuous_source())
    assert evidence["timeline_window_samples"] == 12_000
    assert evidence["timeline_coarse_search_samples"] == 600
    assert evidence["source_span_samples"] == 13_200
    assert evidence["hop_samples"] == 3_000
    assert evidence["total_windows"] == 236
    assert evidence["eligible_windows"] == 236
    assert evidence["eligible_ratio"] == 1.0
    assert evidence["passed"] is True


def test_intermittent_source_is_rejected_before_recording():
    source = _continuous_source()
    source[6 * 48_000 : 9 * 48_000] = 0.0
    timeline = preflight.timeline_source_feasibility(source)
    assert timeline["eligible_ratio"] < preflight.SOURCE_PREFLIGHT_MIN_ELIGIBLE_RATIO
    assert timeline["passed"] is False
    with pytest.raises(
        preflight.RecordingSourcePreflightError,
        match="source preflight 실패",
    ):
        preflight.require_rendered_source_preflight(source, label="intermittent fixture")


def test_continuous_trusted_band_source_passes_full_preflight():
    evidence = preflight.require_rendered_source_preflight(
        _continuous_source(), label="continuous fixture"
    )
    assert evidence["passed"] is True
    assert evidence["timeline_feasibility"]["passed"] is True
    assert evidence["trusted_band_rms_dbfs"] >= evidence[
        "minimum_trusted_band_rms_dbfs"
    ]
    assert evidence["predicted_signal_to_quiet_db"] >= evidence[
        "minimum_predicted_signal_to_quiet_db"
    ]


def test_out_of_trusted_band_only_source_fails_even_when_continuous():
    time = np.arange(preflight.SOURCE_PREFLIGHT_FRAMES, dtype=np.float64) / 48_000.0
    source = (0.05 * np.sin(2.0 * np.pi * 4_000.0 * time)).astype(np.float32)
    evidence = preflight.rendered_source_preflight(source)
    assert evidence["timeline_feasibility"]["passed"] is True
    assert evidence["trusted_band_rms_dbfs"] < evidence[
        "minimum_trusted_band_rms_dbfs"
    ]
    assert evidence["passed"] is False


def test_timeline_receipt_ratio_tamper_is_rejected():
    evidence = preflight.timeline_source_feasibility(_continuous_source())
    forged = copy.deepcopy(evidence)
    forged["eligible_ratio"] = 0.99
    with pytest.raises(
        preflight.RecordingSourcePreflightError,
        match="산술/계약",
    ):
        preflight.validate_timeline_source_feasibility(forged)


def test_full_receipt_predicted_snr_tamper_is_rejected():
    evidence = preflight.rendered_source_preflight(_continuous_source())
    forged = copy.deepcopy(evidence)
    forged["predicted_signal_to_quiet_db"] += 0.1
    with pytest.raises(
        preflight.RecordingSourcePreflightError,
        match="산술/계약",
    ):
        preflight.validate_rendered_source_preflight(forged)
