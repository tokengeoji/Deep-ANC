#!/usr/bin/env python3
"""단일 PortAudio callback에서 FxLMS를 직접 구동하는 구조화된 실기 평가기.

3-thread 실시간 런타임의 ring handoff/miss와 분리해, 한 callback 안에서
digital reference 생성, 소음 재생, FxLMS 제어, ERR/REF 기록을 순서대로 수행한다.
기본 프로토콜은 저음량 300 Hz에서 OFF 10초 -> ON 30초 -> OFF 5초다.

안전 실행 예::

    .venv/bin/python scripts/demo/evaluate_fxlms_direct.py \
      --confirm-user-present-volume-minimum

이 도구는 ``deep_anc.baselines.fxlms_core``의 로컬 구현만 사용한다. 외부
``anc_project``를 import하거나 그 디렉터리에 파일을 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.baselines.fxlms_core import (  # noqa: E402
    DCBlocker,
    FxLMSController,
    float32_to_pcm_int16,
    load_secondary_path,
    pcm_int32_to_float32,
    resolve_alsa_portaudio_device,
    rms_dbfs,
)
from deep_anc.config import REPO_ROOT, load_yaml  # noqa: E402
from deep_anc.audio_io import assert_measurement_preconditions  # noqa: E402
from deep_anc.dsp.measurement_level import assert_live_pcm_clock_preconditions  # noqa: E402


DEFAULT_SECONDARY_PATH = "assets/measured/secondary_path_legacy_512high.npz"
DEFAULT_AMPLITUDE = 0.005
MAX_AMPLITUDE = 0.02
MAX_INPUT_CLIP_RATIO = 0.0
MIN_PATH_REPEAT_CONSISTENCY = 0.90
# I2S 기동 트랜지언트를 버리는 길이. deep_anc.audio_io.DEFAULT_PROBE_SETTLE_SECONDS 와 같은 근거.
PROBE_SETTLE_SECONDS = 1.0
FULL_GAIN_THRESHOLD = 0.999
CONTROL_DIVERGENCE_MULTIPLIER = 10.0
DEFAULT_MAX_TIMESTAMP_JITTER_MS = 1.0
MICROPHONE_NAMES = ("ERR", "REF")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument("--secondary-path", default=DEFAULT_SECONDARY_PATH)
    parser.add_argument("--frequency", type=float, default=300.0)
    parser.add_argument("--amplitude", type=float, default=DEFAULT_AMPLITUDE)
    parser.add_argument("--noise-delay-ms", type=float, default=70.0)
    parser.add_argument("--mu", type=float, default=0.001)
    parser.add_argument("--control-limit", type=float, default=0.10)
    parser.add_argument("--control-len", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--latency", choices=("low", "high"), default="high")
    parser.add_argument(
        "--max-timestamp-jitter-ms",
        type=float,
        default=DEFAULT_MAX_TIMESTAMP_JITTER_MS,
        help="callback ADC/DAC/current timestamp 진행·offset 허용 jitter (ms)",
    )
    parser.add_argument("--off-seconds", type=float, default=10.0)
    parser.add_argument("--on-seconds", type=float, default=30.0)
    parser.add_argument("--tail-off-seconds", type=float, default=5.0)
    parser.add_argument("--pre-silence-seconds", type=float, default=1.0)
    parser.add_argument("--post-silence-seconds", type=float, default=1.0)
    parser.add_argument("--fade-seconds", type=float, default=0.05)
    parser.add_argument("--analysis-seconds", type=float, default=5.0)
    parser.add_argument("--analysis-guard-seconds", type=float, default=1.0)
    parser.add_argument("--input-probe-seconds", type=float, default=2.0)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="새 결과 디렉터리(results 아래). 생략하면 timestamp로 생성",
    )
    parser.add_argument(
        "--confirm-user-present-volume-minimum",
        action="store_true",
        help="사용자 입회와 물리 앰프 볼륨 최저를 확인",
    )
    parser.add_argument("--confirm-speaker", action="store_true")
    return parser


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Deep_ANC 저장소 밖 경로는 사용할 수 없습니다: {resolved}") from exc
    return resolved


def _new_result_dir(value: str | None) -> Path:
    results_root = (REPO_ROOT / "results").resolve()
    if value is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = results_root / "fxlms_direct" / f"fxlms_direct_{stamp}"
    else:
        path = _repo_path(value)
    try:
        path.relative_to(results_root)
    except ValueError as exc:
        raise ValueError(f"--out-dir은 results 아래여야 합니다: {path}") from exc
    if path.exists():
        raise FileExistsError(f"기존 결과 디렉터리는 덮어쓰지 않습니다: {path}")
    return path


def validate_options(args: argparse.Namespace, sample_rate: int) -> None:
    """장치 접근 전에 자극, 프로토콜, 출력 경로를 검증한다."""
    if sample_rate <= 0:
        raise ValueError("sample_rate는 양수여야 합니다")
    if not 0.0 < float(args.frequency) < sample_rate / 2.0:
        raise ValueError("--frequency는 0보다 크고 Nyquist보다 작아야 합니다")
    if not 0.0 < float(args.amplitude) <= MAX_AMPLITUDE:
        raise ValueError(
            f"--amplitude는 0보다 크고 {MAX_AMPLITUDE:.3f} 이하여야 합니다"
        )
    if float(args.noise_delay_ms) < 0.0:
        raise ValueError("--noise-delay-ms는 0 이상이어야 합니다")
    if not 0.0 < float(args.mu) <= 2.0:
        raise ValueError("--mu는 (0, 2] 범위여야 합니다")
    if not 0.0 < float(args.control_limit) <= 0.10:
        raise ValueError("--control-limit은 0보다 크고 0.10 이하여야 합니다")
    if int(args.control_len) < 1 or int(args.block_size) < 1:
        raise ValueError("--control-len/--block-size는 양수여야 합니다")
    if not np.isfinite(float(args.max_timestamp_jitter_ms)) or float(
        args.max_timestamp_jitter_ms
    ) <= 0.0:
        raise ValueError("--max-timestamp-jitter-ms는 유한한 양수여야 합니다")
    if float(args.pre_silence_seconds) < 0.5 or float(args.post_silence_seconds) < 0.5:
        raise ValueError("시작/끝 무음은 각각 0.5초 이상이어야 합니다")
    if min(float(args.off_seconds), float(args.on_seconds), float(args.tail_off_seconds)) <= 0.0:
        raise ValueError("OFF/ON/후행 OFF 길이는 모두 양수여야 합니다")
    if not 0.0 < float(args.fade_seconds) <= min(
        float(args.on_seconds) / 2.0,
        (float(args.off_seconds) + float(args.on_seconds) + float(args.tail_off_seconds)) / 2.0,
    ):
        raise ValueError("--fade-seconds가 프로토콜 길이에 맞지 않습니다")
    if float(args.analysis_seconds) <= 0.0 or float(args.analysis_guard_seconds) < 0.0:
        raise ValueError("분석 길이는 양수, guard는 0 이상이어야 합니다")
    usable_off = float(args.off_seconds) - 2.0 * float(args.analysis_guard_seconds)
    usable_on = float(args.on_seconds) - 2.0 * float(args.analysis_guard_seconds)
    if min(usable_off, usable_on) < 0.5:
        raise ValueError("OFF/ON 구간이 guard 뒤 최소 0.5초 분석창을 제공해야 합니다")
    if float(args.input_probe_seconds) <= 0.0:
        raise ValueError("--input-probe-seconds는 양수여야 합니다")


def build_program(
    *,
    sample_rate: int,
    frequency: float,
    amplitude: float,
    noise_delay_ms: float,
    off_seconds: float,
    on_seconds: float,
    tail_off_seconds: float,
    pre_silence_seconds: float,
    post_silence_seconds: float,
    fade_seconds: float,
) -> dict[str, Any]:
    """digital-reference, 지연된 noise playback, ON gain을 샘플 단위로 만든다."""

    def frames(seconds: float) -> int:
        return int(round(float(seconds) * sample_rate))

    pre = frames(pre_silence_seconds)
    off = frames(off_seconds)
    on = frames(on_seconds)
    tail = frames(tail_off_seconds)
    post = frames(post_silence_seconds)
    delay = frames(noise_delay_ms / 1000.0)
    fade = max(1, frames(fade_seconds))
    total = pre + off + on + tail + post
    active_start = pre
    on_start = active_start + off
    on_stop = on_start + on
    active_stop = on_stop + tail

    reference = np.zeros(total, dtype=np.float32)
    active_frames = active_stop - active_start
    phase = np.arange(active_frames, dtype=np.float64) / float(sample_rate)
    tone = float(amplitude) * np.sin(2.0 * np.pi * float(frequency) * phase)
    source_fade = min(fade, active_frames // 2)
    if source_fade > 0:
        ramp = np.sin(
            np.linspace(0.0, np.pi / 2.0, source_fade, endpoint=True)
        ) ** 2
        tone[:source_fade] *= ramp
        tone[-source_fade:] *= ramp[::-1]
    reference[active_start:active_stop] = tone.astype(np.float32)

    # reference[t]를 먼저 알고, 실제 noise speaker playback만 지연한다.
    noise_playback = np.zeros_like(reference)
    if delay == 0:
        noise_playback[:] = reference
    elif delay < total:
        noise_playback[delay:] = reference[:-delay]

    scheduled_on = np.zeros(total, dtype=np.bool_)
    scheduled_on[on_start:on_stop] = True
    gain = scheduled_on.astype(np.float32)
    gate_fade = min(fade, on // 2)
    if gate_fade > 0:
        ramp = np.sin(
            np.linspace(0.0, np.pi / 2.0, gate_fade, endpoint=True)
        ) ** 2
        gain[on_start : on_start + gate_fade] = ramp.astype(np.float32)
        gain[on_stop - gate_fade : on_stop] = ramp[::-1].astype(np.float32)

    return {
        "reference": reference,
        "noise_playback": noise_playback,
        "scheduled_on": scheduled_on,
        "scheduled_gain": gain,
        "bounds": {
            "pre_silence": (0, pre),
            "initial_off": (active_start, on_start),
            "on": (on_start, on_stop),
            "tail_off": (on_stop, active_stop),
            "post_silence": (active_stop, total),
        },
        "delay_samples": delay,
        "total_frames": total,
    }


def analyze_raw_probe(raw_samples: np.ndarray) -> dict[str, Any]:
    """ERR/REF raw S32_LE가 동적이며 단 한 샘플도 clipping하지 않는지 검사한다."""
    raw = np.asarray(raw_samples)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] < 2:
        raise ValueError(f"ERR/REF 2채널 raw probe가 필요합니다: {raw.shape}")
    raw = raw[:, :2].astype(np.int32, copy=False)
    normalized = pcm_int32_to_float32(raw)
    channels: list[dict[str, Any]] = []
    for channel in range(2):
        values = raw[:, channel]
        signal = normalized[:, channel]
        unique_codes = int(np.unique(values).size)
        clip_count = int(np.count_nonzero(np.abs(signal.astype(np.float64)) >= 0.99))
        clip_ratio = clip_count / float(signal.size)
        rms = float(rms_dbfs(signal))
        valid = bool(unique_codes >= 8 and rms >= -80.0 and clip_ratio <= MAX_INPUT_CLIP_RATIO)
        channels.append(
            {
                "channel": channel,
                "rms_dbfs": rms,
                "peak": float(np.max(np.abs(signal))),
                "clip_count": clip_count,
                "clip_ratio": clip_ratio,
                "unique_codes": unique_codes,
                "raw_min": int(np.min(values)),
                "raw_max": int(np.max(values)),
                "stuck": unique_codes < 8,
                "valid": valid,
            }
        )
    return {"frames": int(raw.shape[0]), "channels": channels}


def capture_raw_preflight(
    sd: Any,
    *,
    audio: dict[str, Any],
    seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """출력 장치는 열지 않고 APE ERR/REF만 캡처한다.

    앞 1초는 I2S 기동 트랜지언트라 버린다 — 포함하면 무신호 마이크도 -42dBFS 로 보여
    생존 판정을 통과한다(deep_anc.audio_io.capture_input_probe 주석 참조).
    """
    sample_rate = int(audio["sample_rate"])
    input_cfg = audio["input"]
    input_device = resolve_alsa_portaudio_device(
        input_cfg["card"], input_cfg["pcm"], "input", 2
    )
    settle_frames = int(round(PROBE_SETTLE_SECONDS * sample_rate))
    raw = sd.rec(
        settle_frames + int(round(float(seconds) * sample_rate)),
        samplerate=sample_rate,
        channels=2,
        dtype="int32",
        device=input_device,
    )
    sd.wait()
    raw = np.asarray(raw, dtype=np.int32)[settle_frames:]
    report = analyze_raw_probe(raw)
    report.update(
        {
            "device": int(input_device),
            "sample_rate": sample_rate,
            "settle_seconds": PROBE_SETTLE_SECONDS,
        }
    )
    return raw, report


def assess_secondary_path(
    path: Path,
    *,
    model: Any,
    block_size: int,
    latency: str,
    frequency: float,
) -> dict[str, Any]:
    """S(z)의 조건 일치와 공식 ESS 승격 증거를 보수적으로 판정한다.

    구형 512/high 모델은 direct FxLMS 진단에는 사용할 수 있지만 반복 timing
    안정성 증거가 없어 정식 성능 주장에는 사용할 수 없다.
    """
    raw_meta: dict[str, Any] = {}
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            for key in (
                "method",
                "repeats",
                "xrun_count",
                "delay_spread_samples",
                "max_delay_jitter_samples",
                "excitation_band_hz",
            ):
                if key in data:
                    value = data[key]
                    raw_meta[key] = value.item() if value.shape == () else value.tolist()

    reasons: list[str] = []
    if int(model.sample_rate) <= 0:
        reasons.append("secondary_path_invalid_sample_rate")
    if int(model.calibration_block_size) not in (0, int(block_size)):
        reasons.append("secondary_path_block_size_mismatch")
    model_latency = str(model.calibration_latency).lower()
    if model_latency not in {"unknown", "legacy-npy", str(latency).lower()}:
        reasons.append("secondary_path_latency_mismatch")

    method = str(raw_meta.get("method", ""))
    repeats = int(raw_meta.get("repeats", 0))
    coherence = float(model.coherence_median)
    if method != "ess":
        reasons.append("secondary_path_not_official_ess")
    if repeats < 3:
        reasons.append("secondary_path_repeat_evidence_missing")
    if not np.isfinite(coherence) or coherence < MIN_PATH_REPEAT_CONSISTENCY:
        reasons.append("secondary_path_repeat_consistency_below_0.9")
    if int(raw_meta.get("xrun_count", 1)) != 0:
        reasons.append("secondary_path_calibration_xrun_or_unknown")
    spread = raw_meta.get("delay_spread_samples")
    maximum = raw_meta.get("max_delay_jitter_samples")
    if spread is None or maximum is None or int(spread) > int(maximum):
        reasons.append("secondary_path_delay_stability_unverified")
    band = raw_meta.get("excitation_band_hz")
    if not (
        isinstance(band, list)
        and len(band) == 2
        and float(band[0]) <= float(frequency) <= float(band[1])
    ):
        reasons.append("tone_outside_secondary_path_evidence_band")

    return {
        "path": str(path.relative_to(REPO_ROOT.resolve())),
        "sample_rate": int(model.sample_rate),
        "delay_samples": int(model.delay_samples),
        "fir_length": int(model.fir.size),
        "calibration_block_size": int(model.calibration_block_size),
        "calibration_latency": str(model.calibration_latency),
        "fit_improvement_db": float(model.fit_improvement_db),
        "coherence_or_repeat_consistency": coherence,
        "raw_quality_metadata": raw_meta,
        "valid_for_performance_claim": not reasons,
        "invalid_reasons": reasons,
        "interpretation": (
            "공식 ESS 반복/지연 안정성 게이트 통과"
            if not reasons
            else "FxLMS 진단에는 사용 가능하지만 정식 성능 주장에는 사용 금지"
        ),
    }


def _empty_telemetry(total_frames: int) -> dict[str, Any]:
    return {
        "callback_count": 0,
        "callback_status_count": 0,
        "xrun_count": 0,
        "priming_output_count": 0,
        "unexpected_status_count": 0,
        "statuses": [],
        "callback_frame_start": [],
        "callback_frames": [],
        "callback_input_buffer_adc_time": [],
        "callback_current_time": [],
        "callback_output_buffer_dac_time": [],
        "callback_timestamps": {
            "stable": False,
            "invalid_reasons": ["callback_timestamps_not_analyzed"],
        },
        "callback_error": None,
        "runner_error": None,
        "safety_latched": False,
        "safety_reasons": [],
        "anc_forced_off_at_frame": None,
        "input_clip_count": 0,
        "output_clip_count": 0,
        "control_limit_count": 0,
        "adaptation_enabled_frames": 0,
        "adaptation_adapted_frames": 0,
        "adaptation_update_segments": 0,
        "adaptation_skipped_segments": 0,
        "completed": False,
        "terminal": False,
        "stream_started": False,
        "stream_closed": False,
        "captured_frames": 0,
        "total_frames": int(total_frames),
    }


def _time_info_value(time_info: Any, name: str) -> float:
    """PortAudio callback time_info의 속성/매핑 값을 float로 읽는다."""
    try:
        value = getattr(time_info, name)
    except (AttributeError, TypeError):
        try:
            value = time_info[name]
        except (KeyError, TypeError):
            return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarize_callback_timestamps(
    telemetry: dict[str, Any],
    *,
    sample_rate: int,
    max_jitter_seconds: float,
) -> dict[str, Any]:
    """callback frame 진행과 ADC/DAC/current timestamp 안정성을 fail-closed 판정한다."""
    if sample_rate <= 0 or not np.isfinite(max_jitter_seconds) or max_jitter_seconds <= 0.0:
        raise ValueError("timestamp 판정 인자는 유효한 양수여야 합니다")

    frame_start = np.asarray(telemetry.get("callback_frame_start", []), dtype=np.float64)
    frames = np.asarray(telemetry.get("callback_frames", []), dtype=np.float64)
    adc = np.asarray(
        telemetry.get("callback_input_buffer_adc_time", []), dtype=np.float64
    )
    current = np.asarray(telemetry.get("callback_current_time", []), dtype=np.float64)
    dac = np.asarray(
        telemetry.get("callback_output_buffer_dac_time", []), dtype=np.float64
    )
    sizes = {values.size for values in (frame_start, frames, adc, current, dac)}
    same_length = len(sizes) == 1
    count = int(frame_start.size) if same_length else 0
    enough = bool(same_length and count >= 2)
    reported_callback_count = int(telemetry.get("callback_count", count))
    matches_callback_count = bool(enough and count == reported_callback_count)
    all_finite = bool(
        enough
        and all(np.all(np.isfinite(values)) for values in (frame_start, frames, adc, current, dac))
    )
    positive_times = bool(
        all_finite
        and np.all(frames > 0.0)
        and np.all(adc > 0.0)
        and np.all(current > 0.0)
        and np.all(dac > 0.0)
    )

    frame_steps = np.diff(frame_start) if all_finite else np.empty(0, np.float64)
    expected_steps = (
        frame_steps / float(sample_rate) if frame_steps.size else np.empty(0, np.float64)
    )
    frame_sequence_contiguous = bool(
        all_finite
        and frame_steps.size > 0
        and np.all(frame_steps > 0.0)
        and np.allclose(frame_steps, frames[:-1], rtol=0.0, atol=0.0)
    )
    adc_steps = np.diff(adc) if all_finite else np.empty(0, np.float64)
    current_steps = np.diff(current) if all_finite else np.empty(0, np.float64)
    dac_steps = np.diff(dac) if all_finite else np.empty(0, np.float64)
    strictly_progressing = bool(
        all_finite
        and expected_steps.size > 0
        and np.all(adc_steps > 0.0)
        and np.all(current_steps > 0.0)
        and np.all(dac_steps > 0.0)
    )

    progression_errors = {
        "input_buffer_adc_time": float("nan"),
        "current_time": float("nan"),
        "output_buffer_dac_time": float("nan"),
    }
    if all_finite and expected_steps.size:
        progression_errors = {
            "input_buffer_adc_time": float(np.max(np.abs(adc_steps - expected_steps))),
            "current_time": float(np.max(np.abs(current_steps - expected_steps))),
            "output_buffer_dac_time": float(np.max(np.abs(dac_steps - expected_steps))),
        }
    current_time_tolerance = max(
        float(max_jitter_seconds),
        0.5 * float(np.median(expected_steps)) if expected_steps.size else 0.0,
    )
    progression_limits = {
        "input_buffer_adc_time": float(max_jitter_seconds),
        # currentTime은 ADC/DAC 시각이 아니라 callback 호출 시각이라 scheduler
        # 흔들림을 포함한다. transfer-map과 같은 반 callback 상한을 사용한다.
        "current_time": current_time_tolerance,
        "output_buffer_dac_time": float(max_jitter_seconds),
    }
    progression_matches_frames = bool(
        strictly_progressing
        and all(
            np.isfinite(progression_errors[key])
            and progression_errors[key] <= progression_limits[key]
            for key in progression_errors
        )
    )

    dac_minus_adc = dac - adc if all_finite else np.empty(0, np.float64)
    offset_positive_plausible = bool(
        positive_times
        and dac_minus_adc.size > 0
        and np.all(dac_minus_adc > 0.0)
        and np.all(dac_minus_adc < 1.0)
    )
    offset_spread = (
        float(np.ptp(dac_minus_adc)) if dac_minus_adc.size else float("nan")
    )
    offset_stable = bool(
        offset_positive_plausible
        and np.isfinite(offset_spread)
        and offset_spread <= float(max_jitter_seconds)
    )

    reasons: list[str] = []
    if not enough or not all_finite:
        reasons.append("callback_timestamps_missing_or_non_finite")
    if enough and not matches_callback_count:
        reasons.append("callback_timestamp_count_mismatch")
    if all_finite and not positive_times:
        reasons.append("callback_timestamps_non_positive")
    if all_finite and not frame_sequence_contiguous:
        reasons.append("callback_frame_sequence_discontinuous")
    if all_finite and not strictly_progressing:
        reasons.append("callback_timestamps_non_progressing")
    if strictly_progressing and not progression_matches_frames:
        reasons.append("callback_timestamp_progression_jitter_exceeded")
    if positive_times and not offset_positive_plausible:
        reasons.append("callback_dac_adc_offset_implausible")
    if offset_positive_plausible and not offset_stable:
        reasons.append("callback_dac_adc_offset_jitter_exceeded")

    stable = bool(
        enough
        and matches_callback_count
        and all_finite
        and positive_times
        and frame_sequence_contiguous
        and strictly_progressing
        and progression_matches_frames
        and offset_stable
        and not reasons
    )
    return {
        "stable": stable,
        "invalid_reasons": reasons,
        "callback_count": count,
        "reported_callback_count": reported_callback_count,
        "matches_callback_count": matches_callback_count,
        "same_length": same_length,
        "all_finite": all_finite,
        "positive_times": positive_times,
        "frame_sequence_contiguous": frame_sequence_contiguous,
        "strictly_progressing": strictly_progressing,
        "progression_matches_frame_start": progression_matches_frames,
        "maximum_progression_error_seconds": {
            key: value if np.isfinite(value) else None
            for key, value in progression_errors.items()
        },
        "maximum_progression_jitter_seconds": progression_limits,
        "dac_minus_adc_seconds": {
            "minimum": float(np.min(dac_minus_adc)) if dac_minus_adc.size else None,
            "median": float(np.median(dac_minus_adc)) if dac_minus_adc.size else None,
            "maximum": float(np.max(dac_minus_adc)) if dac_minus_adc.size else None,
            "spread": offset_spread if np.isfinite(offset_spread) else None,
        },
        "max_allowed_jitter_seconds": float(max_jitter_seconds),
        "interpretation": (
            "outputBufferDacTime-inputBufferAdcTime은 PortAudio 스케줄 offset이며 "
            "음향 전달 지연과 별도다"
        ),
    }


def _status_snapshot(status: Any) -> dict[str, Any]:
    xrun_names = (
        "input_underflow",
        "input_overflow",
        "output_underflow",
        "output_overflow",
    )
    item = {name: bool(getattr(status, name, False)) for name in xrun_names}
    item["priming_output"] = bool(getattr(status, "priming_output", False))
    item["text"] = str(status).strip()
    item["is_xrun"] = any(item[name] for name in xrun_names)
    item["unexpected"] = bool(status) and not item["is_xrun"] and not item["priming_output"]
    return item


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """bool 마스크를 같은 상태의 연속 구간으로 나눈다."""
    values = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if values.size == 0:
        return []
    edges = np.flatnonzero(values[1:] != values[:-1]) + 1
    points = np.concatenate(([0], edges, [values.size]))
    return [
        (int(start), int(stop), bool(values[int(start)]))
        for start, stop in zip(points[:-1], points[1:])
    ]


def run_direct_session(
    sd: Any,
    *,
    controller: Any,
    sample_rate: int,
    block_size: int,
    latency: str,
    input_device: int,
    output_device: int,
    noise_output_channel: int,
    cancel_output_channel: int,
    program: dict[str, Any],
    control_limit: float,
    dc_block_r: float,
    max_timestamp_jitter_seconds: float,
) -> dict[str, Any]:
    """한 callback에서 FxLMS를 구동하고 모든 샘플/상태를 기록한다."""
    if {int(noise_output_channel), int(cancel_output_channel)} != {0, 1}:
        raise ValueError("noise/cancel 출력은 서로 다른 stereo 채널 0/1이어야 합니다")
    reference = np.asarray(program["reference"], dtype=np.float32)
    noise = np.asarray(program["noise_playback"], dtype=np.float32)
    scheduled_on = np.asarray(program["scheduled_on"], dtype=np.bool_)
    scheduled_gain = np.asarray(program["scheduled_gain"], dtype=np.float32)
    total = int(reference.size)
    if not all(values.size == total for values in (noise, scheduled_on, scheduled_gain)):
        raise ValueError("program 배열 길이가 서로 다릅니다")

    raw_input = np.zeros((total, 2), dtype=np.int32)
    output_pcm = np.zeros((total, 2), dtype=np.int16)
    control = np.zeros(total, dtype=np.float32)
    control_unlimited = np.zeros(total, dtype=np.float32)
    actual_gain = np.zeros(total, dtype=np.float32)
    telemetry = _empty_telemetry(total)
    cursor = {"frames": 0}
    dc_blocker = DCBlocker(float(dc_block_r))

    def latch(reason: str, frame: int) -> None:
        if reason not in telemetry["safety_reasons"]:
            telemetry["safety_reasons"].append(reason)
        telemetry["safety_latched"] = True
        if telemetry["anc_forced_off_at_frame"] is None:
            telemetry["anc_forced_off_at_frame"] = int(frame)

    def callback(indata, outdata, frames, time_info, status):
        # 예외/차단 시 이전 출력이 반복되지 않도록 항상 0부터 시작한다.
        outdata.fill(0)
        telemetry["callback_count"] += 1
        start = int(cursor["frames"])
        # callback hot path에는 숫자 append만 둔다. 요약/배열 변환은 stream 종료 뒤 한다.
        telemetry["callback_frame_start"].append(start)
        telemetry["callback_frames"].append(int(frames))
        telemetry["callback_input_buffer_adc_time"].append(
            _time_info_value(time_info, "inputBufferAdcTime")
        )
        telemetry["callback_current_time"].append(
            _time_info_value(time_info, "currentTime")
        )
        telemetry["callback_output_buffer_dac_time"].append(
            _time_info_value(time_info, "outputBufferDacTime")
        )
        count = min(int(frames), total - start)
        try:
            if status:
                item = _status_snapshot(status)
                telemetry["callback_status_count"] += 1
                telemetry["xrun_count"] += int(item["is_xrun"])
                telemetry["priming_output_count"] += int(item["priming_output"])
                telemetry["unexpected_status_count"] += int(item["unexpected"])
                if len(telemetry["statuses"]) < 64:
                    telemetry["statuses"].append(item)
                if item["is_xrun"]:
                    latch("xrun_detected", start)
                if item["unexpected"]:
                    latch("unexpected_callback_status", start)

            if count <= 0:
                telemetry["completed"] = start >= total
                telemetry["terminal"] = True
                raise sd.CallbackStop

            block_raw = np.asarray(indata[:count, :2], dtype=np.int32)
            raw_input[start : start + count] = block_raw
            block_float = pcm_int32_to_float32(block_raw)
            input_clips = int(
                np.count_nonzero(np.abs(block_float.astype(np.float64)) >= 0.99)
            )
            telemetry["input_clip_count"] += input_clips
            if input_clips:
                latch("input_clipping", start)

            if telemetry["safety_latched"]:
                cursor["frames"] = start + count
                telemetry["captured_frames"] = int(cursor["frames"])
                telemetry["terminal"] = True
                raise sd.CallbackStop

            error_dc = dc_blocker.process(block_float[:, 0])
            y_unlimited = np.zeros(count, dtype=np.float32)
            for local_start, local_stop, adapt_enabled in _mask_runs(
                scheduled_on[start : start + count]
            ):
                source_part = reference[start + local_start : start + local_stop]
                generated = np.asarray(
                    controller.generate_block(source_part), dtype=np.float32
                ).reshape(-1)
                if generated.size != local_stop - local_start or not np.all(
                    np.isfinite(generated)
                ):
                    latch("non_finite_or_wrong_length_control", start + local_start)
                    break
                y_unlimited[local_start:local_stop] = generated
                result = controller.adapt_block(
                    error_dc[local_start:local_stop], enabled=adapt_enabled
                )
                if adapt_enabled:
                    telemetry["adaptation_enabled_frames"] += local_stop - local_start
                    if bool(result.adapted):
                        telemetry["adaptation_adapted_frames"] += (
                            local_stop - local_start
                        )
                        telemetry["adaptation_update_segments"] += 1
                    else:
                        telemetry["adaptation_skipped_segments"] += 1
                if bool(result.weight_limited) or not np.isfinite(float(result.weight_norm)):
                    latch("fxlms_weight_divergence", start + local_start)
                    break

            if (
                telemetry["safety_latched"]
                or not np.all(np.isfinite(controller.w))
                or not np.all(np.isfinite(y_unlimited))
            ):
                if not telemetry["safety_latched"]:
                    latch("fxlms_non_finite_state", start)
                cursor["frames"] = start + count
                telemetry["captured_frames"] = int(cursor["frames"])
                telemetry["terminal"] = True
                raise sd.CallbackStop

            divergence_limit = float(control_limit) * CONTROL_DIVERGENCE_MULTIPLIER
            if y_unlimited.size and float(np.max(np.abs(y_unlimited))) > divergence_limit:
                latch("unlimited_control_divergence", start)
                cursor["frames"] = start + count
                telemetry["captured_frames"] = int(cursor["frames"])
                telemetry["terminal"] = True
                raise sd.CallbackStop

            limited = np.clip(y_unlimited, -float(control_limit), float(control_limit))
            telemetry["control_limit_count"] += int(
                np.count_nonzero(np.abs(y_unlimited) > float(control_limit))
            )
            applied = limited * scheduled_gain[start : start + count]
            block_output = np.zeros((count, 2), dtype=np.float32)
            block_output[:, int(noise_output_channel)] = noise[start : start + count]
            block_output[:, int(cancel_output_channel)] = applied
            output_clips = int(
                np.count_nonzero(np.abs(block_output.astype(np.float64)) >= 1.0)
            )
            telemetry["output_clip_count"] += output_clips
            if output_clips:
                latch("output_clipping", start)
                cursor["frames"] = start + count
                telemetry["captured_frames"] = int(cursor["frames"])
                telemetry["terminal"] = True
                raise sd.CallbackStop

            pcm = float32_to_pcm_int16(block_output)
            outdata[:count, :2] = pcm
            output_pcm[start : start + count] = pcm
            control_unlimited[start : start + count] = y_unlimited
            control[start : start + count] = applied
            actual_gain[start : start + count] = scheduled_gain[start : start + count]
            cursor["frames"] = start + count
            telemetry["captured_frames"] = int(cursor["frames"])
            if cursor["frames"] >= total:
                telemetry["completed"] = True
                telemetry["terminal"] = True
                raise sd.CallbackStop
        except sd.CallbackStop:
            raise
        except Exception as exc:
            outdata.fill(0)
            latch("callback_exception", start)
            telemetry["callback_error"] = f"{type(exc).__name__}: {exc}"
            telemetry["terminal"] = True
            raise sd.CallbackAbort

    stream = None
    started = time.monotonic()
    deadline = started + total / float(sample_rate) + 15.0
    try:
        stream = sd.Stream(
            samplerate=sample_rate,
            blocksize=block_size,
            device=(input_device, output_device),
            channels=(2, 2),
            dtype=("int32", "int16"),
            latency=(latency, latency),
            callback=callback,
            prime_output_buffers_using_stream_callback=True,
        )
        stream.start()
        telemetry["stream_started"] = True
        while not telemetry["terminal"]:
            if time.monotonic() >= deadline:
                latch("callback_timeout", int(cursor["frames"]))
                telemetry["runner_error"] = "오디오 callback 완료 대기 시간 초과"
                break
            time.sleep(0.02)
    except Exception as exc:
        latch("stream_exception", int(cursor["frames"]))
        telemetry["runner_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if stream is not None:
            try:
                stream.abort()
            except Exception as exc:
                telemetry["runner_error"] = telemetry["runner_error"] or (
                    f"stream abort: {type(exc).__name__}: {exc}"
                )
            try:
                stream.close()
                telemetry["stream_closed"] = True
            except Exception as exc:
                telemetry["runner_error"] = telemetry["runner_error"] or (
                    f"stream close: {type(exc).__name__}: {exc}"
                )
    telemetry["elapsed_seconds"] = float(time.monotonic() - started)
    telemetry["callback_timestamps"] = summarize_callback_timestamps(
        telemetry,
        sample_rate=int(sample_rate),
        max_jitter_seconds=float(max_timestamp_jitter_seconds),
    )
    captured = int(telemetry["captured_frames"])
    return {
        "raw_input": raw_input[:captured],
        "output_pcm": output_pcm[:captured],
        "reference": reference[:captured],
        "noise_playback": noise[:captured],
        "control": control[:captured],
        "control_unlimited": control_unlimited[:captured],
        "gain": actual_gain[:captured],
        "scheduled_on": scheduled_on[:captured],
        "weights": np.asarray(controller.w, dtype=np.float32).copy(),
        "telemetry": telemetry,
    }


def flush_output_silence(
    sd: Any,
    *,
    output_device: int,
    sample_rate: int,
    block_size: int,
    latency: str,
    seconds: float = 0.25,
) -> dict[str, Any]:
    """finally에서 양 출력 채널에 여러 블록의 0을 쓰고 닫는다."""
    blocks = max(2, int(math.ceil(seconds * sample_rate / block_size)))
    zeros = np.zeros((block_size, 2), dtype=np.int16)
    report: dict[str, Any] = {
        "attempted": True,
        "zero_blocks_requested": blocks,
        "zero_blocks_written": 0,
        "underflow_blocks": 0,
        "both_channels_zero": True,
        "stream_closed": False,
        "error": None,
    }
    stream = None
    try:
        stream = sd.OutputStream(
            samplerate=sample_rate,
            blocksize=block_size,
            device=output_device,
            channels=2,
            dtype="int16",
            latency=latency,
        )
        stream.start()
        for _ in range(blocks):
            report["underflow_blocks"] += int(bool(stream.write(zeros)))
            report["zero_blocks_written"] += 1
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                try:
                    stream.abort()
                except Exception:
                    pass
            try:
                stream.close()
                report["stream_closed"] = True
            except Exception as exc:
                report["error"] = report["error"] or f"close: {type(exc).__name__}: {exc}"
    return report


def collect_with_final_flush(
    sd: Any,
    *,
    run_kwargs: dict[str, Any],
    flush_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """측정 성공/예외와 관계없이 양 채널 zero flush를 수행한다."""
    try:
        capture = run_direct_session(sd, **run_kwargs)
    finally:
        flush_report = flush_output_silence(sd, **flush_kwargs)
    return capture, flush_report


def _tone_level_dbfs(samples: np.ndarray, frequency: float, sample_rate: int) -> float:
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if values.size < 4:
        return -200.0
    values = values - float(np.mean(values))
    window = np.hanning(values.size)
    scale = float(np.sum(window))
    if scale <= 0.0:
        return -200.0
    phase = np.exp(
        -2.0j * np.pi * float(frequency) * np.arange(values.size) / float(sample_rate)
    )
    amplitude = 2.0 * float(np.abs(np.sum(values * window * phase))) / scale
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        return -200.0
    return max(-200.0, 20.0 * math.log10(amplitude))


def compute_matched_metrics(
    capture: dict[str, Any],
    *,
    bounds: dict[str, tuple[int, int]],
    sample_rate: int,
    frequency: float,
    analysis_seconds: float,
    guard_seconds: float,
) -> dict[str, Any]:
    """동일 길이의 후반 OFF와 후반 ON 창을 비교한다."""
    raw = np.asarray(capture["raw_input"], dtype=np.int32)
    if raw.ndim != 2 or raw.shape[1] < 2:
        return {"available": False, "reason": "ERR/REF capture missing"}
    available_frames = int(raw.shape[0])
    off_start, off_stop = bounds["initial_off"]
    on_start, on_stop = bounds["on"]
    tail_start, tail_stop = bounds["tail_off"]
    guard = int(round(float(guard_seconds) * sample_rate))
    max_frames = min(
        int(round(float(analysis_seconds) * sample_rate)),
        off_stop - off_start - 2 * guard,
        on_stop - on_start - 2 * guard,
        tail_stop - tail_start - 2 * guard,
    )
    if max_frames <= 0 or available_frames < tail_stop:
        return {
            "available": False,
            "reason": "complete matched OFF/ON/OFF windows unavailable",
        }
    off_end = off_stop - guard
    on_end = on_stop - guard
    tail_end = tail_stop - guard
    off_slice = slice(off_end - max_frames, off_end)
    on_slice = slice(on_end - max_frames, on_end)
    tail_slice = slice(tail_end - max_frames, tail_end)
    normalized = pcm_int32_to_float32(raw[:, :2])

    def signal_metrics(values: np.ndarray) -> dict[str, float]:
        centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
        return {
            "rms_dbfs": float(rms_dbfs(centered)),
            "tone_dbfs_peak": float(_tone_level_dbfs(centered, frequency, sample_rate)),
            "peak": float(np.max(np.abs(centered))),
        }

    err_off = signal_metrics(normalized[off_slice, 0])
    err_on = signal_metrics(normalized[on_slice, 0])
    err_tail = signal_metrics(normalized[tail_slice, 0])
    ref_off = signal_metrics(normalized[off_slice, 1])
    ref_on = signal_metrics(normalized[on_slice, 1])
    ref_tail = signal_metrics(normalized[tail_slice, 1])
    source = np.asarray(capture["reference"], dtype=np.float32)
    control = np.asarray(capture["control"], dtype=np.float32)
    rms_vs_initial = float(err_off["rms_dbfs"] - err_on["rms_dbfs"])
    rms_vs_tail = float(err_tail["rms_dbfs"] - err_on["rms_dbfs"])
    tone_vs_initial = float(err_off["tone_dbfs_peak"] - err_on["tone_dbfs_peak"])
    tone_vs_tail = float(err_tail["tone_dbfs_peak"] - err_on["tone_dbfs_peak"])
    return {
        "available": True,
        "window_frames": int(max_frames),
        "window_seconds": max_frames / float(sample_rate),
        "off_indices": [int(off_slice.start), int(off_slice.stop)],
        "on_indices": [int(on_slice.start), int(on_slice.stop)],
        "tail_off_indices": [int(tail_slice.start), int(tail_slice.stop)],
        "error": {
            "off": err_off,
            "on": err_on,
            "tail_off": err_tail,
            "rms_attenuation_vs_initial_db": rms_vs_initial,
            "rms_attenuation_vs_tail_db": rms_vs_tail,
            "rms_attenuation_db": min(rms_vs_initial, rms_vs_tail),
            "tone_attenuation_vs_initial_db": tone_vs_initial,
            "tone_attenuation_vs_tail_db": tone_vs_tail,
            "tone_attenuation_db": min(tone_vs_initial, tone_vs_tail),
            "off_return_rms_change_db": float(
                err_tail["rms_dbfs"] - err_off["rms_dbfs"]
            ),
            "off_return_tone_change_db": float(
                err_tail["tone_dbfs_peak"] - err_off["tone_dbfs_peak"]
            ),
        },
        "reference_mic": {
            "off": ref_off,
            "on": ref_on,
            "tail_off": ref_tail,
            "rms_change_db": float(ref_on["rms_dbfs"] - ref_off["rms_dbfs"]),
            "tone_change_db": float(
                ref_on["tone_dbfs_peak"] - ref_off["tone_dbfs_peak"]
            ),
            "off_return_rms_change_db": float(
                ref_tail["rms_dbfs"] - ref_off["rms_dbfs"]
            ),
            "off_return_tone_change_db": float(
                ref_tail["tone_dbfs_peak"] - ref_off["tone_dbfs_peak"]
            ),
        },
        "digital_reference": {
            "off_rms_dbfs": float(rms_dbfs(source[off_slice])),
            "on_rms_dbfs": float(rms_dbfs(source[on_slice])),
        },
        "control": {
            "off_rms_dbfs": float(rms_dbfs(control[off_slice])),
            "on_rms_dbfs": float(rms_dbfs(control[on_slice])),
            "tail_off_rms_dbfs": float(rms_dbfs(control[tail_slice])),
            "on_peak": float(np.max(np.abs(control[on_slice]))),
        },
    }


def quality_gate(
    *,
    preflight_report: dict[str, Any],
    telemetry: dict[str, Any],
    final_flush: dict[str, Any],
    on_duty: float,
    metrics: dict[str, Any],
    secondary_path: dict[str, Any],
) -> dict[str, Any]:
    """측정 유효성과 정식 성능 주장 가능 여부를 분리해 판정한다."""
    reasons: list[str] = []
    channels = preflight_report.get("channels", [])[:2]
    if len(channels) != 2 or not all(bool(item.get("valid")) for item in channels):
        reasons.append("preflight_both_mics_invalid")
    if int(telemetry.get("xrun_count", 0)) != 0:
        reasons.append("xrun_detected")
    timestamp_quality = telemetry.get("callback_timestamps", {})
    if not bool(timestamp_quality.get("stable", False)):
        reasons.append("callback_timestamps_unstable")
        reasons.extend(str(value) for value in timestamp_quality.get("invalid_reasons", []))
    if int(telemetry.get("input_clip_count", 0)) != 0:
        reasons.append("input_clipping")
    if int(telemetry.get("output_clip_count", 0)) != 0:
        reasons.append("output_clipping")
    if bool(telemetry.get("safety_latched", False)):
        reasons.extend(str(v) for v in telemetry.get("safety_reasons", []))
    if telemetry.get("callback_error") or telemetry.get("runner_error"):
        reasons.append("audio_runtime_error")
    if not bool(telemetry.get("completed", False)):
        reasons.append("measurement_incomplete")
    if int(telemetry.get("captured_frames", -1)) != int(telemetry.get("total_frames", -2)):
        reasons.append("captured_frame_count_mismatch")
    if not bool(telemetry.get("stream_closed", False)):
        reasons.append("stream_not_closed")
    if float(on_duty) < 0.95:
        reasons.append("on_duty_below_95_percent")
    if int(telemetry.get("adaptation_update_segments", 0)) == 0:
        reasons.append("fxlms_never_adapted")
    enabled_frames = int(telemetry.get("adaptation_enabled_frames", 0))
    adapted_frames = int(telemetry.get("adaptation_adapted_frames", 0))
    adapted_duty = adapted_frames / float(enabled_frames) if enabled_frames else 0.0
    if adapted_duty < 0.95:
        reasons.append("adaptation_duty_below_95_percent")
    if int(telemetry.get("control_limit_count", 0)) != 0:
        reasons.append("control_output_hard_limited")
    if not bool(metrics.get("available", False)):
        reasons.append("matched_metrics_unavailable")
    else:
        control_metrics = metrics.get("control", {})
        control_peak = float(control_metrics.get("on_peak", float("nan")))
        if not np.isfinite(control_peak) or control_peak <= 1.0e-7:
            reasons.append("control_output_inactive")
        error_return = abs(
            float(metrics.get("error", {}).get("off_return_tone_change_db", float("inf")))
        )
        ref_return = abs(
            float(
                metrics.get("reference_mic", {}).get(
                    "off_return_tone_change_db", float("inf")
                )
            )
        )
        if error_return > 3.0:
            reasons.append("error_off_baseline_did_not_return")
        if ref_return > 3.0:
            reasons.append("reference_off_baseline_did_not_return")
    flush_ok = bool(
        final_flush.get("attempted")
        and final_flush.get("both_channels_zero")
        and final_flush.get("stream_closed")
        and final_flush.get("error") is None
        and int(final_flush.get("underflow_blocks", 0)) == 0
        and int(final_flush.get("zero_blocks_written", 0))
        == int(final_flush.get("zero_blocks_requested", -1))
    )
    if not flush_ok:
        reasons.append("final_zero_flush_incomplete")
    reasons = list(dict.fromkeys(reasons))
    measurement_valid = not reasons

    claim_reasons = list(reasons)
    if not bool(secondary_path.get("valid_for_performance_claim", False)):
        claim_reasons.append("secondary_path_not_validated_for_performance")
        claim_reasons.extend(
            f"S:{value}" for value in secondary_path.get("invalid_reasons", [])
        )
    claim_reasons = list(dict.fromkeys(claim_reasons))
    claim_allowed = not claim_reasons
    reduction = (
        float(metrics["error"]["rms_attenuation_db"])
        if bool(metrics.get("available", False))
        else float("nan")
    )
    return {
        "measurement_valid": measurement_valid,
        "measurement_invalid_reasons": reasons,
        "performance_claim_allowed": claim_allowed,
        "performance_claim_block_reasons": claim_reasons,
        "fxlms_reduction_observed": bool(np.isfinite(reduction) and reduction > 0.0),
        "performance_success": bool(claim_allowed and np.isfinite(reduction) and reduction > 0.0),
        "success_requirements": {
            "on_duty_at_least_95_percent": float(on_duty) >= 0.95,
            "xrun_zero": int(telemetry.get("xrun_count", 0)) == 0,
            "callback_timestamps_stable": bool(timestamp_quality.get("stable", False)),
            "clip_zero": int(telemetry.get("input_clip_count", 0)) == 0
            and int(telemetry.get("output_clip_count", 0)) == 0,
            "complete_termination": bool(
                telemetry.get("completed") and telemetry.get("stream_closed") and flush_ok
            ),
            "adaptation_duty_at_least_95_percent": adapted_duty >= 0.95,
            "control_output_active": "control_output_inactive" not in reasons,
            "control_hard_limit_zero": int(
                telemetry.get("control_limit_count", 0)
            )
            == 0,
            "off_baselines_returned": not any(
                reason
                in {
                    "error_off_baseline_did_not_return",
                    "reference_off_baseline_did_not_return",
                }
                for reason in reasons
            ),
            "secondary_path_officially_validated": bool(
                secondary_path.get("valid_for_performance_claim", False)
            ),
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_results(
    session_dir: Path,
    *,
    preflight_raw: np.ndarray,
    capture: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """NPZ/JSON/weights를 새 세션 디렉터리에 배타적으로 저장한다."""
    session_dir.mkdir(parents=True, exist_ok=False)
    npz_path = session_dir / "measurement.npz"
    json_path = session_dir / "summary.json"
    weights_path = session_dir / "weights.npy"
    raw = np.asarray(capture["raw_input"], dtype=np.int32)
    normalized = (
        pcm_int32_to_float32(raw)
        if raw.size
        else np.empty((0, 2), dtype=np.float32)
    )
    metadata_json = json.dumps(_json_safe(metadata), ensure_ascii=False, sort_keys=True)
    telemetry = capture.get("telemetry", {})
    with npz_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            preflight_raw_int32=np.asarray(preflight_raw, dtype=np.int32),
            input_raw_int32=raw,
            err_raw_int32=raw[:, 0] if raw.size else np.empty(0, np.int32),
            ref_raw_int32=raw[:, 1] if raw.size else np.empty(0, np.int32),
            err=np.asarray(normalized[:, 0], dtype=np.float32),
            ref=np.asarray(normalized[:, 1], dtype=np.float32),
            source=np.asarray(capture["reference"], dtype=np.float32),
            noise_playback=np.asarray(capture["noise_playback"], dtype=np.float32),
            control=np.asarray(capture["control"], dtype=np.float32),
            control_unlimited=np.asarray(capture["control_unlimited"], dtype=np.float32),
            gain=np.asarray(capture["gain"], dtype=np.float32),
            scheduled_on=np.asarray(capture["scheduled_on"], dtype=np.uint8),
            output_pcm_int16=np.asarray(capture["output_pcm"], dtype=np.int16),
            weights=np.asarray(capture["weights"], dtype=np.float32),
            callback_frame_start=np.asarray(
                telemetry.get("callback_frame_start", []), dtype=np.int64
            ),
            callback_frames=np.asarray(telemetry.get("callback_frames", []), dtype=np.int64),
            callback_input_buffer_adc_time=np.asarray(
                telemetry.get("callback_input_buffer_adc_time", []), dtype=np.float64
            ),
            callback_current_time=np.asarray(
                telemetry.get("callback_current_time", []), dtype=np.float64
            ),
            callback_output_buffer_dac_time=np.asarray(
                telemetry.get("callback_output_buffer_dac_time", []), dtype=np.float64
            ),
            metadata_json=np.asarray(metadata_json, dtype=np.str_),
        )
    with weights_path.open("xb") as handle:
        np.save(handle, np.asarray(capture["weights"], dtype=np.float32))
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(_json_safe(metadata), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return npz_path, json_path, weights_path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # 사용자 확인은 config, sounddevice import, 장치 query보다 먼저 검사한다.
    if not (args.confirm_user_present_volume_minimum and args.confirm_speaker):
        print(
            "[중단] 사용자 입회와 물리 앰프 볼륨 최저를 확인한 뒤 "
            "--confirm-user-present-volume-minimum 및 --confirm-speaker를 지정하세요.",
            file=sys.stderr,
        )
        return 2

    try:
        hardware_path = _repo_path(args.hardware)
        secondary_path = _repo_path(args.secondary_path)
        result_dir = _new_result_dir(args.out_dir)
        hardware = load_yaml(hardware_path)
        audio = hardware["audio"]
        channels = hardware["channels"]
        sample_rate = int(audio["sample_rate"])
        validate_options(args, sample_rate)
        model = load_secondary_path(secondary_path)
        if int(model.sample_rate) != sample_rate:
            raise ValueError(
                f"S(z) {model.sample_rate}Hz != hardware {sample_rate}Hz"
            )
        if int(model.calibration_block_size) not in (0, int(args.block_size)):
            raise ValueError(
                f"S(z) block={model.calibration_block_size} != 실행 block={args.block_size}"
            )
        if str(model.calibration_latency).lower() not in {
            "unknown",
            "legacy-npy",
            str(args.latency).lower(),
        }:
            raise ValueError(
                f"S(z) latency={model.calibration_latency} != 실행 latency={args.latency}"
            )
        secondary_assessment = assess_secondary_path(
            secondary_path,
            model=model,
            block_size=int(args.block_size),
            latency=str(args.latency),
            frequency=float(args.frequency),
        )
        program = build_program(
            sample_rate=sample_rate,
            frequency=float(args.frequency),
            amplitude=float(args.amplitude),
            noise_delay_ms=float(args.noise_delay_ms),
            off_seconds=float(args.off_seconds),
            on_seconds=float(args.on_seconds),
            tail_off_seconds=float(args.tail_off_seconds),
            pre_silence_seconds=float(args.pre_silence_seconds),
            post_silence_seconds=float(args.post_silence_seconds),
            fade_seconds=float(args.fade_seconds),
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, FileExistsError) as exc:
        print(f"[중단] 설정 오류: {exc}", file=sys.stderr)
        return 2

    import sounddevice as sd
    try:
        assert_live_pcm_clock_preconditions(audio)
        assert_measurement_preconditions(sd, audio)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] 오디오 사전점검 실패: {exc}", file=sys.stderr)
        return 2

    # 출력 장치를 열기 전에 ERR/REF 둘 다 raw int32로 통과해야 한다.
    try:
        print("출력 없는 ERR/REF raw preflight 중...")
        preflight_raw, preflight_report = capture_raw_preflight(
            sd, audio=audio, seconds=float(args.input_probe_seconds)
        )
    except (OSError, RuntimeError, ValueError, sd.PortAudioError) as exc:
        print(f"[중단] raw 입력 preflight 실패: {exc}", file=sys.stderr)
        return 2
    for name, item in zip(MICROPHONE_NAMES, preflight_report["channels"]):
        verdict = "PASS" if item["valid"] else "FAIL"
        print(
            f"[{verdict}] {name}: RMS {item['rms_dbfs']:.2f}dBFS, "
            f"peak {item['peak']:.6f}, clip {item['clip_count']}, "
            f"unique {item['unique_codes']}"
        )
    if not all(bool(item["valid"]) for item in preflight_report["channels"][:2]):
        print(
            "[중단] ERR/REF raw preflight가 모두 PASS하지 않아 출력 장치를 열지 않습니다.",
            file=sys.stderr,
        )
        return 2

    try:
        output_cfg = audio["output"]
        output_device = resolve_alsa_portaudio_device(
            output_cfg["card"], output_cfg["pcm"], "output", 2
        )
        input_device = int(preflight_report["device"])
        noise_channel = int(channels["noise_out"])
        cancel_channel = int(channels["cancel_out"])
        controller = FxLMSController(
            model.fir,
            secondary_delay_samples=int(model.delay_samples),
            control_len=int(args.control_len),
            mu=float(args.mu),
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[중단] 장치/FxLMS 초기화 실패: {exc}", file=sys.stderr)
        return 2

    print(
        f"[저음량 direct FxLMS] {args.frequency:.1f}Hz peak={args.amplitude:.4f}, "
        f"digital-ref lead={args.noise_delay_ms:.1f}ms, block={args.block_size}/{args.latency}"
    )
    print(
        f"OFF {args.off_seconds:.1f}s -> ON {args.on_seconds:.1f}s -> "
        f"OFF {args.tail_off_seconds:.1f}s; 시작/끝 무음 포함"
    )
    if not secondary_assessment["valid_for_performance_claim"]:
        print(
            "[주의] 현재 S(z)는 공식 반복/지연 안정성 게이트를 통과하지 않았습니다. "
            "감쇠가 보여도 진단값이며 성능 성공으로 주장하지 않습니다.",
            file=sys.stderr,
        )

    run_kwargs = {
        "controller": controller,
        "sample_rate": sample_rate,
        "block_size": int(args.block_size),
        "latency": str(args.latency),
        "input_device": input_device,
        "output_device": output_device,
        "noise_output_channel": noise_channel,
        "cancel_output_channel": cancel_channel,
        "program": program,
        "control_limit": float(args.control_limit),
        "dc_block_r": float(model.dc_block_r),
        "max_timestamp_jitter_seconds": float(args.max_timestamp_jitter_ms) / 1000.0,
    }
    flush_kwargs = {
        "output_device": output_device,
        "sample_rate": sample_rate,
        "block_size": int(args.block_size),
        "latency": str(args.latency),
    }
    try:
        capture, final_flush = collect_with_final_flush(
            sd, run_kwargs=run_kwargs, flush_kwargs=flush_kwargs
        )
    except BaseException as exc:
        # collect_with_final_flush의 finally가 이미 양 채널 0 flush를 시도했다.
        print(f"[실패] direct callback 실행 예외: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    scheduled_total_on = int(np.count_nonzero(program["scheduled_on"]))
    delivered_full_on = int(
        np.count_nonzero(
            np.asarray(capture["gain"], dtype=np.float32) >= FULL_GAIN_THRESHOLD
        )
    )
    on_duty = delivered_full_on / float(scheduled_total_on) if scheduled_total_on else 0.0
    metrics = compute_matched_metrics(
        capture,
        bounds=program["bounds"],
        sample_rate=sample_rate,
        frequency=float(args.frequency),
        analysis_seconds=float(args.analysis_seconds),
        guard_seconds=float(args.analysis_guard_seconds),
    )
    quality = quality_gate(
        preflight_report=preflight_report,
        telemetry=capture["telemetry"],
        final_flush=final_flush,
        on_duty=on_duty,
        metrics=metrics,
        secondary_path=secondary_assessment,
    )
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "measurement_kind": "direct_callback_digital_reference_fxlms_off_on_off",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hardware_config": str(hardware_path.relative_to(REPO_ROOT.resolve())),
        "controller_implementation": "deep_anc.baselines.fxlms_core.FxLMSController",
        "external_anc_project_used": False,
        "sample_rate": sample_rate,
        "input_device": input_device,
        "output_device": output_device,
        "channels": {
            "error_mic": int(channels["error_mic"]),
            "reference_mic": int(channels["reference_mic"]),
            "noise_out": noise_channel,
            "cancel_out": cancel_channel,
        },
        "stimulus": {
            "type": "tone",
            "frequency_hz": float(args.frequency),
            "amplitude_peak": float(args.amplitude),
            "hard_maximum_amplitude_peak": MAX_AMPLITUDE,
            "digital_reference_lead_noise_delay_ms": float(args.noise_delay_ms),
            "digital_reference_lead_noise_delay_samples": int(program["delay_samples"]),
        },
        "fxlms": {
            "mu": float(args.mu),
            "control_limit": float(args.control_limit),
            "control_length": int(args.control_len),
            "secondary_path": secondary_assessment,
            "adaptation_policy": "scheduled ON frames only",
        },
        "stream": {
            "block_size": int(args.block_size),
            "latency": str(args.latency),
            "max_timestamp_jitter_ms": float(args.max_timestamp_jitter_ms),
            "callback_timestamps": capture["telemetry"]["callback_timestamps"],
        },
        "protocol_seconds": {
            "pre_silence": float(args.pre_silence_seconds),
            "initial_off": float(args.off_seconds),
            "on": float(args.on_seconds),
            "tail_off": float(args.tail_off_seconds),
            "post_silence": float(args.post_silence_seconds),
            "fade": float(args.fade_seconds),
        },
        "raw_preflight": preflight_report,
        "callback_telemetry": capture["telemetry"],
        "final_zero_flush": final_flush,
        "on_duty": on_duty,
        "matched_metrics": metrics,
        "quality": quality,
        "limitations": [
            "300Hz 단일 톤의 direct-callback FxLMS 진단이다.",
            (
                "이 결과 하나로 덕트 P(z)/S(z), 광대역 전달함수 또는 "
                "전체 덕트 구조가 식별되지는 않는다."
            ),
            (
                "S(z)가 공식 반복/지연 안정성 게이트를 통과하지 않으면 "
                "양의 감쇠도 성능 주장에 사용할 수 없다."
            ),
        ],
    }
    try:
        npz_path, json_path, weights_path = save_results(
            result_dir,
            preflight_raw=preflight_raw,
            capture=capture,
            metadata=metadata,
        )
    except (OSError, ValueError, FileExistsError) as exc:
        print(f"[실패] 결과 저장 실패: {exc}", file=sys.stderr)
        return 2

    print(f"ON duty: {100.0 * on_duty:.2f}%")
    if metrics.get("available"):
        print(
            f"matched ERR 감쇠: RMS {metrics['error']['rms_attenuation_db']:+.2f}dB, "
            f"{args.frequency:.1f}Hz {metrics['error']['tone_attenuation_db']:+.2f}dB"
        )
    verdict = "VALID" if quality["measurement_valid"] else "INVALID"
    claim = "가능" if quality["performance_claim_allowed"] else "금지"
    print(f"측정 품질: {verdict} | 정식 성능 주장: {claim}")
    if quality["performance_claim_block_reasons"]:
        print(
            "성능 주장 차단 사유: "
            + ", ".join(quality["performance_claim_block_reasons"]),
            file=sys.stderr,
        )
    print(f"JSON: {json_path}")
    print(f"NPZ: {npz_path}")
    print(f"weights: {weights_path}")
    return 0 if quality["measurement_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
