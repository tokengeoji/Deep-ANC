"""v6 clock-checkpoint 측정용 전이중 callback public wrapper.

장치 처리 primitive는 검증된 v5 구현과 공유하지만 artifact schema는 분리한다.
이 모듈 자체는 오디오 backend를 import하거나 장치를 열지 않는다.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from .audio_duplex_v5 import (
    BLOCK_SIZE,
    CHANNELS,
    LATENCY,
    SAMPLE_RATE,
    STATUS_PRESENT,
    STATUS_XRUN_MASK,
    DuplexCaptureFailure,
    _capture_duplex,
    status_bitmask_v5,
)


DUPLEX_TELEMETRY_SCHEMA = "fullband_causal_v6_duplex_telemetry_v2"


def capture_duplex_v6(
    backend: Any,
    *,
    submitted_pcm: np.ndarray,
    input_device: int,
    output_device: int,
    pre_open_check: Callable[[], None] | None = None,
    watchdog_grace_seconds: float = 2.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    on_output_closed: Callable[[bool], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return _capture_duplex(
        backend,
        submitted_pcm=submitted_pcm,
        input_device=input_device,
        output_device=output_device,
        telemetry_schema=DUPLEX_TELEMETRY_SCHEMA,
        include_pre_open_telemetry=True,
        pre_open_check=pre_open_check,
        watchdog_grace_seconds=watchdog_grace_seconds,
        monotonic=monotonic,
        sleep=sleep,
        on_output_closed=on_output_closed,
    )


__all__ = [
    "BLOCK_SIZE",
    "CHANNELS",
    "DUPLEX_TELEMETRY_SCHEMA",
    "DuplexCaptureFailure",
    "LATENCY",
    "SAMPLE_RATE",
    "STATUS_PRESENT",
    "STATUS_XRUN_MASK",
    "capture_duplex_v6",
    "status_bitmask_v5",
]
