from __future__ import annotations

import numpy as np
import pytest

from deep_anc.audio_duplex_v6 import (
    DUPLEX_TELEMETRY_SCHEMA,
    DuplexCaptureFailure,
    capture_duplex_v6,
)

from test_audio_duplex_v5 import Backend, pcm


def test_v6_wrapper_preserves_transport_and_uses_v6_schema() -> None:
    backend = Backend()
    captured, telemetry = capture_duplex_v6(
        backend,
        submitted_pcm=pcm(),
        input_device=1,
        output_device=2,
    )
    assert captured.dtype == np.dtype("<i4")
    assert telemetry["schema"] == DUPLEX_TELEMETRY_SCHEMA
    assert telemetry["normal_stop_completed"] is True
    assert set(telemetry) >= {
        "pre_open_monotonic_started",
        "pre_open_monotonic_completed",
        "pre_open_monotonic_elapsed_seconds",
    }


def test_v5_and_v6_telemetry_schemas_are_distinct() -> None:
    from deep_anc.audio_duplex_v5 import DUPLEX_TELEMETRY_SCHEMA as v5_schema

    assert v5_schema != DUPLEX_TELEMETRY_SCHEMA


def test_v6_slow_pre_open_is_separate_from_capture_elapsed() -> None:
    backend = Backend()
    ticks = iter([10.0, 71.0, 71.0, 71.025])
    _, telemetry = capture_duplex_v6(
        backend,
        submitted_pcm=pcm(),
        input_device=1,
        output_device=2,
        pre_open_check=lambda: None,
        monotonic=lambda: next(ticks),
        sleep=lambda _seconds: None,
    )
    assert telemetry["pre_open_monotonic_started"] == 10.0
    assert telemetry["pre_open_monotonic_completed"] == 71.0
    assert telemetry["pre_open_monotonic_elapsed_seconds"] == 61.0
    assert telemetry["capture_monotonic_started"] == 71.0
    assert telemetry["capture_monotonic_elapsed_seconds"] == pytest.approx(0.025)


def test_v6_pre_open_failure_has_no_stream_or_output() -> None:
    backend = Backend()
    ticks = iter([5.0, 66.0])

    def reject() -> None:
        raise RuntimeError("v6 pre-open rejected")

    with pytest.raises(DuplexCaptureFailure, match="v6 pre-open rejected") as caught:
        capture_duplex_v6(
            backend,
            submitted_pcm=pcm(),
            input_device=1,
            output_device=2,
            pre_open_check=reject,
            monotonic=lambda: next(ticks),
        )
    assert backend.calls == []
    assert backend.outputs == []
    assert caught.value.telemetry["captured_frames"] == 0
    assert caught.value.telemetry["pre_open_monotonic_elapsed_seconds"] == 61.0
    assert caught.value.telemetry["capture_monotonic_elapsed_seconds"] == 0.0
