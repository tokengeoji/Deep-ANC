"""실제 출력 전에 15초 녹음 소스의 측정 가능성을 검증한다.

이 모듈은 오디오 장치를 열지 않는다. 녹음에 실제로 공급될 ``float32`` 파형만
받아, timeline 추정기가 필요로 하는 source RMS 창과 공식 측정 레벨 계약을 같은
숫자로 재검산한다. 목적은 스피커를 울린 뒤에야 알 수 있었던 무음/저레벨 소스를
selector와 source-plan dry-run 단계에서 미리 차단하는 것이다.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from typing import Any, Mapping

import numpy as np

from deep_anc.data.timeline import TimelineSettings
from deep_anc.dsp.measurement_level import (
    OFFICIAL_MEASUREMENT_CHANNEL_MAP,
    OFFICIAL_MEASUREMENT_LEVEL,
    band_rms_dbfs,
    expected_meter_output_pcm,
)


SOURCE_PREFLIGHT_SCHEMA = "recording_source_preflight/v1"
TIMELINE_FEASIBILITY_SCHEMA = "recording_source_timeline_feasibility/v1"
SOURCE_PREFLIGHT_SAMPLE_RATE = 48_000
SOURCE_PREFLIGHT_FRAMES = 720_000
SOURCE_PREFLIGHT_PLAYBACK_AMPLITUDE = 0.06

# 실제 capture gate의 valid-window 하한은 0.90이다. source-only 필요조건은 물리 경로,
# 품질 점수, rail rejection이 더해지기 전에 이미 만족해야 하므로 5 %p 준비 여유를 둔다.
# 이 값은 실패 결과를 통과시키려고 낮추지 않는다.
SOURCE_PREFLIGHT_MIN_ELIGIBLE_RATIO = 0.95

# 2026-08-30 input-only probe의 가장 높은 quiet ERR(-64.33 dBFS)보다 보수적인 ceiling.
# 공식 meter 재생보다 source trusted-band가 2 dB 넘게 약해지지 않게 하고, capture
# coherence^2=0.90에 필요한 SNR도 독립적으로 요구한다.
SOURCE_PREFLIGHT_CONSERVATIVE_QUIET_FLOOR_DBFS = -64.0
SOURCE_PREFLIGHT_REQUIRED_COHERENCE_SQUARED = 0.90
SOURCE_PREFLIGHT_MAX_DB_BELOW_OFFICIAL_METER_PLAYBACK = 2.0


class RecordingSourcePreflightError(ValueError):
    """렌더된 녹음 소스가 무출력 사전 게이트를 만족하지 않는다."""


def _canonical_json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordingSourcePreflightError(f"{label}은 finite 숫자여야 합니다")
    result = float(value)
    if not math.isfinite(result):
        raise RecordingSourcePreflightError(f"{label}은 finite 숫자여야 합니다")
    return result


@lru_cache(maxsize=1)
def _official_meter_reference() -> tuple[float, float, float, float]:
    meter_pcm = expected_meter_output_pcm(
        noise_channel=OFFICIAL_MEASUREMENT_CHANNEL_MAP["noise_out"]
    )
    meter = np.asarray(meter_pcm[:, 0], dtype=np.float64) / 32768.0
    meter_playback = float(band_rms_dbfs(meter))
    minimum_trusted = float(
        meter_playback - SOURCE_PREFLIGHT_MAX_DB_BELOW_OFFICIAL_METER_PLAYBACK
    )
    required_snr = float(
        10.0
        * math.log10(
            SOURCE_PREFLIGHT_REQUIRED_COHERENCE_SQUARED
            / (1.0 - SOURCE_PREFLIGHT_REQUIRED_COHERENCE_SQUARED)
        )
    )
    return (
        meter_playback,
        minimum_trusted,
        float(OFFICIAL_MEASUREMENT_LEVEL.meter_min_dbfs),
        required_snr,
    )


def timeline_source_feasibility(samples: np.ndarray) -> dict[str, Any]:
    """Timeline estimator의 source-RMS 필요조건을 exact rendered 파형에서 계산한다.

    추정기의 source segment는 ``window + 2*coarse_search`` 길이다. 따라서 단순한
    0.25초 RMS가 아니라 현행 12,000 + 2*600 = 13,200 sample 창을 사용한다.
    """

    values = np.asarray(samples)
    if values.ndim != 1 or values.shape != (SOURCE_PREFLIGHT_FRAMES,):
        raise RecordingSourcePreflightError(
            f"source preflight는 mono exact {SOURCE_PREFLIGHT_FRAMES} frames여야 합니다"
        )
    canonical = np.ascontiguousarray(values, dtype="<f4")
    signal = np.asarray(canonical, dtype=np.float64)
    if not bool(np.isfinite(signal).all()):
        raise RecordingSourcePreflightError("source preflight 파형에 non-finite가 있습니다")

    settings = TimelineSettings(sample_rate=SOURCE_PREFLIGHT_SAMPLE_RATE)
    span = int(settings.window_samples + 2 * settings.coarse_search_samples)
    hop = int(settings.hop_samples)
    starts = np.arange(0, signal.size - span + 1, hop, dtype=np.int64)
    if starts.size < 1:
        raise RecordingSourcePreflightError("source preflight RMS 창을 만들 수 없습니다")
    squared = np.square(signal, dtype=np.float64)
    cumulative = np.empty(signal.size + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(squared, dtype=np.float64, out=cumulative[1:])
    mean_power = (cumulative[starts + span] - cumulative[starts]) / float(span)
    rms = np.sqrt(np.maximum(mean_power, 0.0))
    eligible = rms >= float(settings.min_window_rms)
    total = int(starts.size)
    eligible_count = int(np.count_nonzero(eligible))
    ratio = float(eligible_count / total)

    longest = 0
    current = 0
    for passed in eligible:
        if bool(passed):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    evidence = {
        "schema": TIMELINE_FEASIBILITY_SCHEMA,
        "sample_rate": SOURCE_PREFLIGHT_SAMPLE_RATE,
        "frames": SOURCE_PREFLIGHT_FRAMES,
        "timeline_window_samples": int(settings.window_samples),
        "timeline_coarse_search_samples": int(settings.coarse_search_samples),
        "source_span_samples": span,
        "hop_samples": hop,
        "minimum_window_rms": float(settings.min_window_rms),
        "total_windows": total,
        "eligible_windows": eligible_count,
        "eligible_ratio": ratio,
        "minimum_eligible_ratio": SOURCE_PREFLIGHT_MIN_ELIGIBLE_RATIO,
        "longest_ineligible_run_windows": int(longest),
        "passed": bool(ratio >= SOURCE_PREFLIGHT_MIN_ELIGIBLE_RATIO),
    }
    return validate_timeline_source_feasibility(evidence)


def validate_timeline_source_feasibility(value: Any) -> dict[str, Any]:
    required = {
        "schema",
        "sample_rate",
        "frames",
        "timeline_window_samples",
        "timeline_coarse_search_samples",
        "source_span_samples",
        "hop_samples",
        "minimum_window_rms",
        "total_windows",
        "eligible_windows",
        "eligible_ratio",
        "minimum_eligible_ratio",
        "longest_ineligible_run_windows",
        "passed",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise RecordingSourcePreflightError("timeline source feasibility 필드 집합이 다릅니다")
    settings = TimelineSettings(sample_rate=SOURCE_PREFLIGHT_SAMPLE_RATE)
    span = int(settings.window_samples + 2 * settings.coarse_search_samples)
    total_expected = (
        (SOURCE_PREFLIGHT_FRAMES - span) // int(settings.hop_samples) + 1
    )
    integers: dict[str, int] = {}
    for field in (
        "sample_rate",
        "frames",
        "timeline_window_samples",
        "timeline_coarse_search_samples",
        "source_span_samples",
        "hop_samples",
        "total_windows",
        "eligible_windows",
        "longest_ineligible_run_windows",
    ):
        observed = value.get(field)
        if isinstance(observed, bool) or not isinstance(observed, int):
            raise RecordingSourcePreflightError(f"timeline {field}는 정수여야 합니다")
        integers[field] = int(observed)
    ratio = _finite_number(value.get("eligible_ratio"), label="eligible_ratio")
    minimum_ratio = _finite_number(
        value.get("minimum_eligible_ratio"), label="minimum_eligible_ratio"
    )
    minimum_rms = _finite_number(
        value.get("minimum_window_rms"), label="minimum_window_rms"
    )
    eligible = integers["eligible_windows"]
    total = integers["total_windows"]
    expected_pass = ratio >= SOURCE_PREFLIGHT_MIN_ELIGIBLE_RATIO
    if (
        value.get("schema") != TIMELINE_FEASIBILITY_SCHEMA
        or integers["sample_rate"] != SOURCE_PREFLIGHT_SAMPLE_RATE
        or integers["frames"] != SOURCE_PREFLIGHT_FRAMES
        or integers["timeline_window_samples"] != settings.window_samples
        or integers["timeline_coarse_search_samples"]
        != settings.coarse_search_samples
        or integers["source_span_samples"] != span
        or integers["hop_samples"] != settings.hop_samples
        or total != total_expected
        or not 0 <= eligible <= total
        or not math.isclose(ratio, eligible / total, rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(
            minimum_ratio,
            SOURCE_PREFLIGHT_MIN_ELIGIBLE_RATIO,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not math.isclose(
            minimum_rms,
            settings.min_window_rms,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not 0 <= integers["longest_ineligible_run_windows"] <= total - eligible
        or value.get("passed") is not expected_pass
    ):
        raise RecordingSourcePreflightError("timeline source feasibility 산술/계약 위반")
    return _canonical_json_copy(value)


def rendered_source_preflight(samples: np.ndarray) -> dict[str, Any]:
    """15초 exact rendered source의 timeline+절대 trusted-band 증거를 만든다."""

    values = np.asarray(samples)
    if values.ndim != 1 or values.shape != (SOURCE_PREFLIGHT_FRAMES,):
        raise RecordingSourcePreflightError(
            f"rendered source는 mono exact {SOURCE_PREFLIGHT_FRAMES} frames여야 합니다"
        )
    canonical = np.ascontiguousarray(values, dtype="<f4")
    signal = np.asarray(canonical, dtype=np.float64)
    if not bool(np.isfinite(signal).all()):
        raise RecordingSourcePreflightError("rendered source에 non-finite가 있습니다")
    peak = float(np.max(np.abs(signal)))
    floor = np.finfo(np.float64).tiny
    rms_dbfs = float(
        20.0 * math.log10(math.sqrt(float(np.mean(np.square(signal)))) + floor)
    )
    trusted = float(band_rms_dbfs(signal))
    meter_playback, minimum_trusted, meter_min, required_snr = (
        _official_meter_reference()
    )
    predicted_err = float(meter_min + trusted - meter_playback)
    predicted_snr = float(
        predicted_err - SOURCE_PREFLIGHT_CONSERVATIVE_QUIET_FLOOR_DBFS
    )
    timeline = timeline_source_feasibility(canonical)
    passed = bool(
        peak > 0.0
        and peak <= SOURCE_PREFLIGHT_PLAYBACK_AMPLITUDE + 1.0e-6
        and trusted >= minimum_trusted
        and predicted_snr >= required_snr
        and timeline["passed"] is True
    )
    evidence = {
        "schema": SOURCE_PREFLIGHT_SCHEMA,
        "sample_rate": SOURCE_PREFLIGHT_SAMPLE_RATE,
        "frames": SOURCE_PREFLIGHT_FRAMES,
        "sample_encoding": "float32_le",
        "sample_sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
        "playback_amplitude": SOURCE_PREFLIGHT_PLAYBACK_AMPLITUDE,
        "peak_linear": peak,
        "rms_dbfs": rms_dbfs,
        "trusted_band_hz": list(OFFICIAL_MEASUREMENT_LEVEL.meter_band_hz),
        "trusted_band_rms_dbfs": trusted,
        "official_meter_playback_trusted_band_dbfs": meter_playback,
        "maximum_db_below_official_meter_playback": (
            SOURCE_PREFLIGHT_MAX_DB_BELOW_OFFICIAL_METER_PLAYBACK
        ),
        "minimum_trusted_band_rms_dbfs": minimum_trusted,
        "meter_target_min_dbfs": meter_min,
        "conservative_quiet_floor_dbfs": (
            SOURCE_PREFLIGHT_CONSERVATIVE_QUIET_FLOOR_DBFS
        ),
        "predicted_err_trusted_band_min_dbfs": predicted_err,
        "predicted_signal_to_quiet_db": predicted_snr,
        "required_capture_coherence_squared": (
            SOURCE_PREFLIGHT_REQUIRED_COHERENCE_SQUARED
        ),
        "minimum_predicted_signal_to_quiet_db": required_snr,
        "timeline_feasibility": timeline,
        "passed": passed,
    }
    return validate_rendered_source_preflight(evidence)


def validate_rendered_source_preflight(value: Any) -> dict[str, Any]:
    required = {
        "schema",
        "sample_rate",
        "frames",
        "sample_encoding",
        "sample_sha256",
        "playback_amplitude",
        "peak_linear",
        "rms_dbfs",
        "trusted_band_hz",
        "trusted_band_rms_dbfs",
        "official_meter_playback_trusted_band_dbfs",
        "maximum_db_below_official_meter_playback",
        "minimum_trusted_band_rms_dbfs",
        "meter_target_min_dbfs",
        "conservative_quiet_floor_dbfs",
        "predicted_err_trusted_band_min_dbfs",
        "predicted_signal_to_quiet_db",
        "required_capture_coherence_squared",
        "minimum_predicted_signal_to_quiet_db",
        "timeline_feasibility",
        "passed",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise RecordingSourcePreflightError("rendered source preflight 필드 집합이 다릅니다")
    timeline = validate_timeline_source_feasibility(value.get("timeline_feasibility"))
    sample_sha = value.get("sample_sha256")
    numbers = {
        field: _finite_number(value.get(field), label=field)
        for field in (
            "playback_amplitude",
            "peak_linear",
            "rms_dbfs",
            "trusted_band_rms_dbfs",
            "official_meter_playback_trusted_band_dbfs",
            "maximum_db_below_official_meter_playback",
            "minimum_trusted_band_rms_dbfs",
            "meter_target_min_dbfs",
            "conservative_quiet_floor_dbfs",
            "predicted_err_trusted_band_min_dbfs",
            "predicted_signal_to_quiet_db",
            "required_capture_coherence_squared",
            "minimum_predicted_signal_to_quiet_db",
        )
    }
    meter_playback, minimum_trusted, meter_min, required_snr = (
        _official_meter_reference()
    )
    predicted_err = (
        meter_min + numbers["trusted_band_rms_dbfs"] - meter_playback
    )
    predicted_snr = (
        predicted_err - SOURCE_PREFLIGHT_CONSERVATIVE_QUIET_FLOOR_DBFS
    )
    expected_pass = bool(
        numbers["peak_linear"] > 0.0
        and numbers["peak_linear"]
        <= SOURCE_PREFLIGHT_PLAYBACK_AMPLITUDE + 1.0e-6
        and numbers["trusted_band_rms_dbfs"] >= minimum_trusted
        and predicted_snr >= required_snr
        and timeline["passed"] is True
    )
    exact = {
        "playback_amplitude": SOURCE_PREFLIGHT_PLAYBACK_AMPLITUDE,
        "official_meter_playback_trusted_band_dbfs": meter_playback,
        "maximum_db_below_official_meter_playback": (
            SOURCE_PREFLIGHT_MAX_DB_BELOW_OFFICIAL_METER_PLAYBACK
        ),
        "minimum_trusted_band_rms_dbfs": minimum_trusted,
        "meter_target_min_dbfs": meter_min,
        "conservative_quiet_floor_dbfs": (
            SOURCE_PREFLIGHT_CONSERVATIVE_QUIET_FLOOR_DBFS
        ),
        "required_capture_coherence_squared": (
            SOURCE_PREFLIGHT_REQUIRED_COHERENCE_SQUARED
        ),
        "minimum_predicted_signal_to_quiet_db": required_snr,
    }
    if (
        value.get("schema") != SOURCE_PREFLIGHT_SCHEMA
        or value.get("sample_rate") != SOURCE_PREFLIGHT_SAMPLE_RATE
        or value.get("frames") != SOURCE_PREFLIGHT_FRAMES
        or value.get("sample_encoding") != "float32_le"
        or not isinstance(sample_sha, str)
        or len(sample_sha) != 64
        or any(character not in "0123456789abcdef" for character in sample_sha)
        or value.get("trusted_band_hz")
        != list(OFFICIAL_MEASUREMENT_LEVEL.meter_band_hz)
        or any(
            not math.isclose(
                numbers[field], float(expected), rel_tol=1e-12, abs_tol=1e-12
            )
            for field, expected in exact.items()
        )
        or not math.isclose(
            numbers["predicted_err_trusted_band_min_dbfs"],
            predicted_err,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            numbers["predicted_signal_to_quiet_db"],
            predicted_snr,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or numbers["rms_dbfs"] > 0.0
        or value.get("passed") is not expected_pass
    ):
        raise RecordingSourcePreflightError("rendered source preflight 산술/계약 위반")
    return _canonical_json_copy(value)


def require_rendered_source_preflight(
    samples: np.ndarray, *, label: str
) -> dict[str, Any]:
    evidence = rendered_source_preflight(samples)
    if evidence["passed"] is not True:
        timeline = evidence["timeline_feasibility"]
        raise RecordingSourcePreflightError(
            f"{label} source preflight 실패: timeline="
            f"{float(timeline['eligible_ratio']):.6f} < "
            f"{SOURCE_PREFLIGHT_MIN_ELIGIBLE_RATIO:.2f} 또는 trusted="
            f"{float(evidence['trusted_band_rms_dbfs']):.2f} dBFS < "
            f"{float(evidence['minimum_trusted_band_rms_dbfs']):.2f} dBFS"
        )
    return evidence


__all__ = [
    "RecordingSourcePreflightError",
    "SOURCE_PREFLIGHT_FRAMES",
    "SOURCE_PREFLIGHT_MIN_ELIGIBLE_RATIO",
    "SOURCE_PREFLIGHT_PLAYBACK_AMPLITUDE",
    "SOURCE_PREFLIGHT_SAMPLE_RATE",
    "SOURCE_PREFLIGHT_SCHEMA",
    "TIMELINE_FEASIBILITY_SCHEMA",
    "rendered_source_preflight",
    "require_rendered_source_preflight",
    "timeline_source_feasibility",
    "validate_rendered_source_preflight",
    "validate_timeline_source_feasibility",
]
