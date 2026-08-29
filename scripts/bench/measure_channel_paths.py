#!/usr/bin/env python3
"""저레벨 단일 톤으로 출력 2채널과 ERR/REF 마이크의 결합 경로를 확인한다.

이 스크립트는 ``noise_out``과 ``cancel_out``을 한 번에 하나씩 구동하고,
각 구동 전/후 무음과 ERR/REF 원시 S32_LE 입력을 함께 저장한다. 결과는 300 Hz
한 점의 채널/배선 진단일 뿐이며, 덕트 P(z)/S(z) 식별이나 FxLMS 성능 측정이 아니다.

안전 실행 예::

    .venv/bin/python scripts/bench/measure_channel_paths.py \
      --confirm-user-present-volume-minimum

``--dry-run``은 sounddevice/ALSA를 import·open하지 않고, 입력 인자·hardware YAML·
출력 PCM 계획·no-replace 출력 경로만 검사한다. 따라서 물리 승인 flag는 dry-run에서
요구하지 않는다. 반대로 기본 live 경로는 실제 스피커를 구동하므로 두 confirmation
flag가 모두 필수다.
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

from deep_anc.audio_io import (  # noqa: E402
    assert_measurement_preconditions,
    capture_input_probe,
    pcm_int32_to_float32,
    resolve_alsa_portaudio_device,
    rms_dbfs,
)
from deep_anc.config import REPO_ROOT, load_yaml  # noqa: E402
from deep_anc.dsp.measurement_level import assert_live_pcm_clock_preconditions  # noqa: E402


DEFAULT_AMPLITUDE = 0.005
MAX_AMPLITUDE = 0.02
MICROPHONE_NAMES = ("error_mic", "reference_mic")
DEFAULT_FLUSH_SECONDS = 0.25
DRY_RUN_SCHEMA_VERSION = 1


def validate_probe_settings(
    *,
    sample_rate: int,
    frequency: float,
    amplitude: float,
    block_size: int,
    pre_seconds: float,
    tone_seconds: float,
    post_seconds: float,
) -> None:
    """오디오 장치를 열기 전에 자극과 길이의 안전 범위를 검사한다."""
    if sample_rate <= 0:
        raise ValueError("sample_rate는 양수여야 합니다")
    if not math.isfinite(float(frequency)) or not 0.0 < frequency < sample_rate / 2.0:
        raise ValueError("frequency는 0보다 크고 Nyquist보다 작아야 합니다")
    if not math.isfinite(float(amplitude)) or not 0.0 < amplitude <= MAX_AMPLITUDE:
        raise ValueError(
            f"amplitude는 0보다 크고 {MAX_AMPLITUDE:.3f} 이하여야 합니다"
        )
    if block_size <= 0:
        raise ValueError("block-size는 양수여야 합니다")
    if (
        not math.isfinite(float(pre_seconds))
        or not math.isfinite(float(post_seconds))
        or pre_seconds < 0.5
        or post_seconds < 0.5
    ):
        raise ValueError("안전을 위해 pre/post 무음은 각각 0.5초 이상이어야 합니다")
    if not math.isfinite(float(tone_seconds)) or tone_seconds < 1.0:
        raise ValueError("안정적인 저레벨 분석을 위해 tone-seconds는 1초 이상이어야 합니다")


def build_output_program(
    *,
    sample_rate: int,
    frequency: float,
    amplitude: float,
    output_channel: int,
    pre_seconds: float,
    tone_seconds: float,
    post_seconds: float,
    fade_seconds: float = 0.05,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """정확히 한 출력 채널만 포함하는 S16_LE 전후-무음 프로그램을 만든다."""
    if output_channel not in (0, 1):
        raise ValueError("output_channel은 0 또는 1이어야 합니다")
    pre_frames = int(round(pre_seconds * sample_rate))
    tone_frames = int(round(tone_seconds * sample_rate))
    post_frames = int(round(post_seconds * sample_rate))
    if min(pre_frames, tone_frames, post_frames) <= 0:
        raise ValueError("모든 구간은 한 프레임 이상이어야 합니다")

    t = np.arange(tone_frames, dtype=np.float64) / float(sample_rate)
    tone = amplitude * np.sin(2.0 * np.pi * frequency * t)
    fade_frames = min(int(round(fade_seconds * sample_rate)), tone_frames // 2)
    if fade_frames > 0:
        ramp = np.sin(
            np.linspace(0.0, np.pi / 2.0, fade_frames, endpoint=True)
        ) ** 2
        tone[:fade_frames] *= ramp
        tone[-fade_frames:] *= ramp[::-1]

    total = pre_frames + tone_frames + post_frames
    output = np.zeros((total, 2), dtype=np.int16)
    tone_start = pre_frames
    tone_stop = tone_start + tone_frames
    output[tone_start:tone_stop, output_channel] = np.rint(
        np.clip(tone, -1.0, 1.0) * 32767.0
    ).astype(np.int16)
    bounds = {
        "pre": (0, pre_frames),
        "tone": (tone_start, tone_stop),
        "post": (tone_stop, total),
    }
    return output, bounds


def tone_bin_dbfs(samples: np.ndarray, frequency: float, sample_rate: int) -> float:
    """Hann 창 단일 주파수 투영의 peak-amplitude dBFS를 반환한다."""
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if values.size < 4:
        return -200.0
    values = values - float(np.mean(values))
    window = np.hanning(values.size)
    scale = float(np.sum(window))
    if scale <= 0.0:
        return -200.0
    phase = np.exp(
        -2.0j
        * np.pi
        * float(frequency)
        * np.arange(values.size, dtype=np.float64)
        / float(sample_rate)
    )
    amplitude = 2.0 * float(np.abs(np.sum(values * window * phase))) / scale
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        return -200.0
    return max(-200.0, 20.0 * math.log10(amplitude))


def _segment_metrics(
    signal: np.ndarray,
    raw: np.ndarray,
    frequency: float,
    sample_rate: int,
) -> dict[str, Any]:
    values = np.asarray(signal, dtype=np.float32)
    raw_values = np.asarray(raw, dtype=np.int32)
    if values.size == 0:
        raise ValueError("비어 있는 분석 구간입니다")
    return {
        "frames": int(values.size),
        "rms_dbfs": float(rms_dbfs(values)),
        "tone_bin_dbfs_peak": float(tone_bin_dbfs(values, frequency, sample_rate)),
        "peak": float(np.max(np.abs(values))),
        "clip_ratio": float(
            np.mean(np.abs(values.astype(np.float64)) >= 0.99)
        ),
        "raw_min": int(np.min(raw_values)),
        "raw_max": int(np.max(raw_values)),
        "unique_codes": int(np.unique(raw_values).size),
    }


def analyze_path_capture(
    raw_input: np.ndarray,
    *,
    bounds: dict[str, tuple[int, int]],
    frequency: float,
    sample_rate: int,
    analysis_guard_seconds: float = 0.10,
    status_report: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """출력 한 채널을 구동한 녹음에서 ERR/REF 결합 증거를 계산한다."""
    raw = np.asarray(raw_input)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"ERR/REF 2채널 S32_LE 입력이 필요합니다: {raw.shape}")
    raw = raw.astype(np.int32, copy=False)
    floating = pcm_int32_to_float32(raw)
    guard = int(round(analysis_guard_seconds * sample_rate))

    pre_start, pre_stop = bounds["pre"]
    tone_start, tone_stop = bounds["tone"]
    post_start, post_stop = bounds["post"]
    steady_start = min(tone_stop, tone_start + guard)
    steady_stop = max(steady_start, tone_stop - guard)
    if steady_stop <= steady_start:
        steady_start, steady_stop = tone_start, tone_stop

    xrun_blocks = int(
        (status_report or {}).get(
            "xrun_blocks", (status_report or {}).get("status_blocks", 0)
        )
    )
    reports: list[dict[str, Any]] = []
    for channel, name in enumerate(MICROPHONE_NAMES):
        pre = _segment_metrics(
            floating[pre_start:pre_stop, channel],
            raw[pre_start:pre_stop, channel],
            frequency,
            sample_rate,
        )
        tone = _segment_metrics(
            floating[steady_start:steady_stop, channel],
            raw[steady_start:steady_stop, channel],
            frequency,
            sample_rate,
        )
        post = _segment_metrics(
            floating[post_start:post_stop, channel],
            raw[post_start:post_stop, channel],
            frequency,
            sample_rate,
        )
        tone_delta = float(tone["tone_bin_dbfs_peak"] - pre["tone_bin_dbfs_peak"])
        rms_delta = float(tone["rms_dbfs"] - pre["rms_dbfs"])
        coupling_detected = bool(
            xrun_blocks == 0
            and tone["clip_ratio"] <= 0.005
            and tone["tone_bin_dbfs_peak"] > -100.0
            and tone_delta >= 6.0
        )
        reports.append(
            {
                "input_channel": channel,
                "microphone": name,
                "pre_silence": pre,
                "tone_steady": tone,
                "post_silence": post,
                "rms_change_db": rms_delta,
                "tone_bin_change_db": tone_delta,
                "coupling_detected": coupling_detected,
                "interpretation": (
                    f"{frequency:g}Hz 출력-마이크 결합 증거"
                    if coupling_detected
                    else f"현재 SNR에서 {frequency:g}Hz 결합을 확정하지 못함"
                ),
            }
        )
    return reports


def _empty_status_report() -> dict[str, Any]:
    return {
        "callback_blocks": 0,
        "status_blocks": 0,
        "xrun_blocks": 0,
        "input_overflow_blocks": 0,
        "input_underflow_blocks": 0,
        "output_overflow_blocks": 0,
        "output_underflow_blocks": 0,
        "priming_output_blocks": 0,
        "messages": [],
    }


def _record_callback_status(report: dict[str, Any], status: Any) -> None:
    report["callback_blocks"] += 1
    if not bool(status):
        return
    report["status_blocks"] += 1
    xrun = False
    for attribute in (
        "input_overflow",
        "input_underflow",
        "output_overflow",
        "output_underflow",
    ):
        if bool(getattr(status, attribute, False)):
            report[f"{attribute}_blocks"] += 1
            xrun = True
    if xrun:
        report["xrun_blocks"] += 1
    if bool(getattr(status, "priming_output", False)):
        report["priming_output_blocks"] += 1
    message = str(status).strip()
    if message and message not in report["messages"] and len(report["messages"]) < 32:
        report["messages"].append(message)


def run_output_probe(
    sd: Any,
    *,
    input_device: int,
    output_device: int,
    sample_rate: int,
    block_size: int,
    latency: str,
    output_pcm: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], float]:
    """한 출력 프로그램을 전이중 스트림으로 재생하고 원시 입력을 캡처한다."""
    program = np.asarray(output_pcm, dtype=np.int16)
    if program.ndim != 2 or program.shape[1] != 2:
        raise ValueError("output_pcm은 [frames, 2] int16이어야 합니다")
    total = int(program.shape[0])
    recorded = np.zeros((total, 2), dtype=np.int32)
    cursor = {"frames": 0}
    status_report = _empty_status_report()

    def callback(indata, outdata, frames, _time_info, status):
        # 어떤 종료/예외 경로에서도 지정하지 않은 채널과 남는 프레임은 0이다.
        outdata.fill(0)
        _record_callback_status(status_report, status)
        start = cursor["frames"]
        count = min(int(frames), total - start)
        if count > 0:
            recorded[start : start + count] = np.asarray(
                indata[:count, :2], dtype=np.int32
            )
            outdata[:count, :2] = program[start : start + count]
            cursor["frames"] = start + count
        if cursor["frames"] >= total:
            raise sd.CallbackStop

    expected_seconds = total / float(sample_rate)
    deadline = time.monotonic() + expected_seconds + max(5.0, expected_seconds)
    started = time.monotonic()
    with sd.Stream(
        samplerate=sample_rate,
        blocksize=block_size,
        device=(input_device, output_device),
        channels=(2, 2),
        dtype=("int32", "int16"),
        latency=(latency, latency),
        callback=callback,
        prime_output_buffers_using_stream_callback=True,
    ):
        while cursor["frames"] < total:
            if time.monotonic() >= deadline:
                raise RuntimeError("오디오 callback이 제한 시간 안에 완료되지 않았습니다")
            time.sleep(0.02)
    elapsed = time.monotonic() - started
    return recorded, status_report, elapsed


def flush_output_silence(
    sd: Any,
    *,
    output_device: int,
    sample_rate: int,
    block_size: int,
    latency: str,
    seconds: float = DEFAULT_FLUSH_SECONDS,
) -> dict[str, Any]:
    """두 출력 채널에 0을 쓰고 스트림을 닫아 마지막 자극을 제거한다."""
    blocks = max(2, int(math.ceil(seconds * sample_rate / block_size)))
    zeros = np.zeros((block_size, 2), dtype=np.int16)
    underflows = 0
    with sd.OutputStream(
        samplerate=sample_rate,
        blocksize=block_size,
        device=output_device,
        channels=2,
        dtype="int16",
        latency=latency,
    ) as stream:
        for _ in range(blocks):
            underflows += int(bool(stream.write(zeros)))
    return {
        "attempted": True,
        "zero_blocks": blocks,
        "underflow_blocks": underflows,
        "both_channels_zero": True,
        "stream_closed": True,
    }


def collect_channel_paths(
    sd: Any,
    *,
    input_device: int,
    output_device: int,
    sample_rate: int,
    block_size: int,
    latency: str,
    programs: list[tuple[str, int, np.ndarray]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """두 채널을 순차 측정하며 성공/실패와 무관하게 마지막에 양 채널을 mute한다."""
    captures: dict[str, dict[str, Any]] = {}
    silence_report: dict[str, Any] = {"attempted": False}
    try:
        for name, channel, output_pcm in programs:
            raw, status, elapsed = run_output_probe(
                sd,
                input_device=input_device,
                output_device=output_device,
                sample_rate=sample_rate,
                block_size=block_size,
                latency=latency,
                output_pcm=output_pcm,
            )
            captures[name] = {
                "output_channel": int(channel),
                "raw_input": raw,
                "output_pcm": output_pcm,
                "status": status,
                "elapsed_seconds": float(elapsed),
            }
    finally:
        silence_report = flush_output_silence(
            sd,
            output_device=output_device,
            sample_rate=sample_rate,
            block_size=block_size,
            latency=latency,
        )
    return captures, silence_report


def _repo_path(value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Deep_ANC 저장소 밖 경로는 사용할 수 없습니다: {resolved}") from exc
    return resolved


def _channel_paths_output_root() -> Path:
    """진단 산출물의 허용된 **실경로** root를 반환한다.

    ``results/channel_paths`` 아래라는 문자열만 검사하면 ``..`` 또는 중간 symlink로
    ``assets/measured`` 등 다른 역할의 경로에 raw를 남길 수 있다. 결과 역할은
    diagnostic-only이므로 realpath도 Deep_ANC 내부의 이 전용 root 아래여야 한다.
    """

    repository_root = REPO_ROOT.resolve()
    expected_root = repository_root / "results" / "channel_paths"
    root = expected_root.resolve()
    try:
        # root가 repository 안이라는 것만으로는 부족하다. 예를 들어
        # results/channel_paths -> assets/measured symlink라면 raw 역할이 섞인다.
        root.relative_to(expected_root)
    except ValueError as exc:
        raise ValueError(
            "results/channel_paths 자체의 실경로가 전용 diagnostic root 밖을 가리켜 "
            "채널 경로 진단 결과를 저장할 수 없습니다"
        ) from exc
    return root


def _require_channel_paths_output_path(path: Path) -> Path:
    """``path``가 diagnostic output root 아래의 실경로인지 fail-closed로 확인한다."""

    root = _channel_paths_output_root()
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "채널 경로 진단 --out-prefix는 실경로 기준 "
            f"results/channel_paths/ 아래여야 합니다: {resolved}"
        ) from exc
    return resolved


def _output_paths(value: str | None) -> tuple[Path, Path]:
    if value is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = REPO_ROOT / "results" / "channel_paths" / f"channel_paths_{stamp}"
    else:
        prefix = _repo_path(value)
        if prefix.suffix.lower() in {".json", ".npz"}:
            prefix = prefix.with_suffix("")
    # suffix 제거 뒤에도 다시 검사해야 ``escape.npz``가 아닌 ``escape`` 자체가
    # symlink인 경우를 놓치지 않는다. 두 최종 파일도 재확인해 기존 symlink artifact를
    # 덮어쓰는 경로를 만들지 않는다.
    prefix = _require_channel_paths_output_path(prefix)
    npz_path = _require_channel_paths_output_path(prefix.with_suffix(".npz"))
    json_path = _require_channel_paths_output_path(prefix.with_suffix(".json"))
    return npz_path, json_path


def _validate_dry_run_hardware_config(
    hardware_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """장치를 열지 않고 채널 진단에 필요한 YAML 부분만 엄격히 확인한다.

    이 함수는 ``--dry-run`` 전용이다. ALSA card/proc 조회나 PortAudio 장치 열기는
    수행하지 않으므로, 계획 검증이 실제 hardware liveness 증거로 오인될 수 없다.
    """
    if not isinstance(hardware_cfg, dict):
        raise ValueError("hardware YAML 최상위는 mapping이어야 합니다")
    audio = hardware_cfg.get("audio")
    channels = hardware_cfg.get("channels")
    if not isinstance(audio, dict) or not isinstance(channels, dict):
        raise ValueError("hardware YAML에 audio와 channels mapping이 필요합니다")
    input_cfg = audio.get("input")
    output_cfg = audio.get("output")
    if not isinstance(input_cfg, dict) or not isinstance(output_cfg, dict):
        raise ValueError("hardware YAML에 audio.input/audio.output mapping이 필요합니다")

    try:
        sample_rate = int(audio["sample_rate"])
        input_channels = int(input_cfg["channels"])
        output_channels = int(output_cfg["channels"])
        input_pcm = int(input_cfg["pcm"])
        output_pcm = int(output_cfg["pcm"])
        input_card = str(input_cfg["card"]).strip()
        output_card = str(output_cfg["card"]).strip()
        channel_map = {
            name: int(channels[name])
            for name in (
                "error_mic",
                "reference_mic",
                "noise_out",
                "cancel_out",
            )
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"hardware YAML channel/sample-rate 값이 유효하지 않습니다: {exc}") from exc

    if sample_rate <= 0:
        raise ValueError("hardware YAML audio.sample_rate는 양수여야 합니다")
    if not input_card or not output_card or input_pcm < 0 or output_pcm < 0:
        raise ValueError("hardware YAML input/output card와 pcm은 유효해야 합니다")
    if input_channels < 2 or output_channels < 2:
        raise ValueError("채널 경로 진단에는 ERR/REF 입력 2채널과 출력 2채널이 필요합니다")
    if channel_map["error_mic"] != 0 or channel_map["reference_mic"] != 1:
        raise ValueError("ERR/REF channel map은 각각 0/1이어야 합니다")
    if {channel_map["noise_out"], channel_map["cancel_out"]} != {0, 1}:
        raise ValueError("noise_out/cancel_out은 서로 다른 stereo 채널 0/1이어야 합니다")
    if str(input_cfg.get("dtype", "")).lower() != "int32":
        raise ValueError("채널 경로 진단 input dtype은 int32여야 합니다")
    if str(output_cfg.get("dtype", "")).lower() != "int16":
        raise ValueError("채널 경로 진단 output dtype은 int16여야 합니다")
    return audio, channels, sample_rate


def build_channel_probe_plan(
    *,
    sample_rate: int,
    frequency: float,
    amplitude: float,
    block_size: int,
    pre_seconds: float,
    tone_seconds: float,
    post_seconds: float,
    preflight_seconds: float,
    output_channels: tuple[tuple[str, int], ...],
) -> tuple[list[tuple[str, int, np.ndarray]], dict[str, tuple[int, int]], dict[str, Any]]:
    """live와 dry-run이 공유하는 출력 PCM·정확한 시간 계획을 만든다.

    반환 PCM은 memory-only 계획이며, raw microphone capture나 파일을 만들지 않는다.
    """
    validate_probe_settings(
        sample_rate=sample_rate,
        frequency=frequency,
        amplitude=amplitude,
        block_size=block_size,
        pre_seconds=pre_seconds,
        tone_seconds=tone_seconds,
        post_seconds=post_seconds,
    )
    if not math.isfinite(float(preflight_seconds)) or preflight_seconds <= 0.0:
        raise ValueError("preflight-seconds는 양수 finite여야 합니다")
    if tuple(name for name, _channel in output_channels) != ("noise_out", "cancel_out"):
        raise ValueError("출력 계획은 noise_out 뒤 cancel_out 순서여야 합니다")
    if {int(channel) for _name, channel in output_channels} != {0, 1}:
        raise ValueError("출력 계획은 서로 다른 stereo 채널 0/1이어야 합니다")

    programs: list[tuple[str, int, np.ndarray]] = []
    common_bounds: dict[str, tuple[int, int]] | None = None
    stream_plans: list[dict[str, Any]] = []
    for name, channel in output_channels:
        program, bounds = build_output_program(
            sample_rate=sample_rate,
            frequency=frequency,
            amplitude=amplitude,
            output_channel=int(channel),
            pre_seconds=pre_seconds,
            tone_seconds=tone_seconds,
            post_seconds=post_seconds,
        )
        if common_bounds is None:
            common_bounds = bounds
        elif bounds != common_bounds:
            raise RuntimeError("두 출력 채널의 segment bounds가 다릅니다")
        frames = int(program.shape[0])
        stream_plans.append(
            {
                "name": name,
                "output_channel": int(channel),
                "frames": frames,
                "seconds": frames / float(sample_rate),
                "pre_silence_frames": int(bounds["pre"][1] - bounds["pre"][0]),
                "tone_frames": int(bounds["tone"][1] - bounds["tone"][0]),
                "post_silence_frames": int(bounds["post"][1] - bounds["post"][0]),
                "pre_silence_seconds": (
                    bounds["pre"][1] - bounds["pre"][0]
                )
                / float(sample_rate),
                "audible_tone_seconds": (
                    bounds["tone"][1] - bounds["tone"][0]
                )
                / float(sample_rate),
                "post_silence_seconds": (
                    bounds["post"][1] - bounds["post"][0]
                )
                / float(sample_rate),
            }
        )
        programs.append((name, int(channel), program))

    if common_bounds is None:  # pragma: no cover - fixed two-channel contract
        raise RuntimeError("출력 계획이 비어 있습니다")
    flush_blocks = max(
        2,
        int(math.ceil(DEFAULT_FLUSH_SECONDS * sample_rate / block_size)),
    )
    output_stream_seconds = float(sum(item["seconds"] for item in stream_plans))
    audible_seconds = float(sum(item["audible_tone_seconds"] for item in stream_plans))
    flush_stream_seconds = flush_blocks * block_size / float(sample_rate)
    duration_plan: dict[str, Any] = {
        "input_preflight_seconds": float(preflight_seconds),
        "output_stream_seconds": output_stream_seconds,
        "audible_seconds": audible_seconds,
        "flush_requested_seconds": DEFAULT_FLUSH_SECONDS,
        "flush_zero_blocks": flush_blocks,
        "flush_stream_seconds": flush_stream_seconds,
        "total_expected_device_occupancy_seconds": (
            float(preflight_seconds) + output_stream_seconds + flush_stream_seconds
        ),
        "output_streams": stream_plans,
    }
    return programs, common_bounds, duration_plan


def _dry_run_report(
    *,
    hardware_path: Path,
    sample_rate: int,
    frequency: float,
    amplitude: float,
    block_size: int,
    latency: str,
    channels: dict[str, Any],
    duration_plan: dict[str, Any],
    npz_path: Path,
    json_path: Path,
) -> dict[str, Any]:
    """물리 승인 없이 보여 줄, side-effect 없는 계획 receipt를 만든다."""
    return {
        "schema_version": DRY_RUN_SCHEMA_VERSION,
        "measurement_kind": "low_level_output_microphone_channel_path_probe_dry_run",
        "mode": "dry_run",
        "hardware_config": str(hardware_path.relative_to(REPO_ROOT.resolve())),
        "sample_rate": int(sample_rate),
        "frequency_hz": float(frequency),
        "amplitude_peak": float(amplitude),
        "block_size": int(block_size),
        "latency": latency,
        "channels": {
            name: int(channels[name])
            for name in ("error_mic", "reference_mic", "noise_out", "cancel_out")
        },
        "duration_plan": duration_plan,
        "output_paths": {
            "raw_capture_npz": str(npz_path.relative_to(REPO_ROOT.resolve())),
            "summary_json": str(json_path.relative_to(REPO_ROOT.resolve())),
            "no_replace_checked": True,
        },
        "dry_run_guarantees": {
            "sounddevice_imported": False,
            "alsa_or_portaudio_device_opened": False,
            "raw_microphone_capture_created": False,
            "artifact_written": False,
            "speaker_output": False,
        },
        "confirmation_policy": {
            "required_for_dry_run": False,
            "required_for_live": [
                "--confirm-user-present-volume-minimum",
                "--confirm-speaker",
            ],
            "reason": "dry-run은 장치를 열거나 스피커를 출력하지 않으므로 물리 승인을 받지 않는다",
        },
        "limitations": [
            "계획 검증은 실제 ALSA/PortAudio 장치 liveness나 microphone rail을 증명하지 않는다.",
            "실제 live는 input preflight PASS와 두 confirmation flag 후에만 시작된다.",
        ],
    }


def _save_results(
    npz_path: Path,
    json_path: Path,
    summary: dict[str, Any],
    captures: dict[str, dict[str, Any]],
) -> None:
    if npz_path.exists() or json_path.exists():
        raise FileExistsError("기존 측정 결과는 덮어쓰지 않습니다. 다른 --out-prefix를 쓰세요")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(
            json.dumps(summary, ensure_ascii=False, sort_keys=True), dtype=np.str_
        )
    }
    for name, capture in captures.items():
        arrays[f"{name}_input_raw_int32"] = np.asarray(
            capture["raw_input"], dtype=np.int32
        )
        arrays[f"{name}_output_pcm_int16"] = np.asarray(
            capture["output_pcm"], dtype=np.int16
        )

    token = f"{time.time_ns()}"
    npz_temp = npz_path.with_name(f".{npz_path.name}.{token}.tmp")
    json_temp = json_path.with_name(f".{json_path.name}.{token}.tmp")
    try:
        with npz_temp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        json_temp.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        npz_temp.replace(npz_path)
        json_temp.replace(json_path)
    finally:
        # 정상 경로에서는 replace로 사라진다. 실패 시 작은 임시 메타/측정 파일만 정리한다.
        npz_temp.unlink(missing_ok=True)
        json_temp.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument("--frequency", type=float, default=300.0)
    parser.add_argument("--amplitude", type=float, default=DEFAULT_AMPLITUDE)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--latency", choices=("low", "high"), default="high")
    parser.add_argument("--pre-seconds", type=float, default=1.0)
    parser.add_argument("--tone-seconds", type=float, default=2.0)
    parser.add_argument("--post-seconds", type=float, default=1.0)
    parser.add_argument("--preflight-seconds", type=float, default=2.0)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "sounddevice/ALSA를 import·open하지 않고 hardware YAML, PCM 계획, "
            "출력 no-replace 경로와 예상 시간을 JSON으로 확인한다. "
            "물리 confirmation flag는 필요 없다"
        ),
    )
    parser.add_argument(
        "--confirm-user-present-volume-minimum",
        action="store_true",
        help="사용자 입회와 물리 앰프 볼륨 최저를 확인",
    )
    parser.add_argument("--confirm-speaker", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.dry_run and not (
        args.confirm_user_present_volume_minimum and args.confirm_speaker
    ):
        print(
            "[중단] 사용자 입회와 앰프 볼륨 최저를 확인한 뒤 "
            "--confirm-user-present-volume-minimum 및 --confirm-speaker를 지정하세요.",
            file=sys.stderr,
        )
        return 2

    try:
        hardware_path = _repo_path(args.hardware)
        hardware_cfg = load_yaml(hardware_path)
        # dry-run/live 공통으로 장치 import/open 전에 diagnostic raw의 저장 역할을
        # 고정한다. 이후 경로 오류가 발견되면 입력 preflight조차 시작하지 않는다.
        npz_path, json_path = _output_paths(args.out_prefix)
        if npz_path.exists() or json_path.exists():
            raise FileExistsError(
                "기존 측정 결과는 덮어쓰지 않습니다. 다른 --out-prefix를 쓰세요"
            )
        if args.dry_run:
            _audio, channels, sample_rate = _validate_dry_run_hardware_config(
                hardware_cfg
            )
            output_channels = (
                ("noise_out", int(channels["noise_out"])),
                ("cancel_out", int(channels["cancel_out"])),
            )
            _programs, _bounds, duration_plan = build_channel_probe_plan(
                sample_rate=sample_rate,
                frequency=args.frequency,
                amplitude=args.amplitude,
                block_size=args.block_size,
                pre_seconds=args.pre_seconds,
                tone_seconds=args.tone_seconds,
                post_seconds=args.post_seconds,
                preflight_seconds=args.preflight_seconds,
                output_channels=output_channels,
            )
            report = _dry_run_report(
                hardware_path=hardware_path,
                sample_rate=sample_rate,
                frequency=args.frequency,
                amplitude=args.amplitude,
                block_size=args.block_size,
                latency=args.latency,
                channels=channels,
                duration_plan=duration_plan,
                npz_path=npz_path,
                json_path=json_path,
            )
            print("[DRY-RUN PASS] 재생·녹음·raw/artifact 파일 생성 없이 계획만 검증했습니다.")
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        audio = hardware_cfg["audio"]
        import sounddevice as sd
        assert_live_pcm_clock_preconditions(audio)
        assert_measurement_preconditions(sd, audio)
        channels = hardware_cfg["channels"]
        sample_rate = int(audio["sample_rate"])
        validate_probe_settings(
            sample_rate=sample_rate,
            frequency=args.frequency,
            amplitude=args.amplitude,
            block_size=args.block_size,
            pre_seconds=args.pre_seconds,
            tone_seconds=args.tone_seconds,
            post_seconds=args.post_seconds,
        )
        if not math.isfinite(float(args.preflight_seconds)) or args.preflight_seconds <= 0.0:
            raise ValueError("preflight-seconds는 양수여야 합니다")
        output_channels = (int(channels["noise_out"]), int(channels["cancel_out"]))
        if set(output_channels) != {0, 1}:
            raise ValueError(
                "noise_out/cancel_out은 서로 다른 stereo 채널 0/1이어야 합니다: "
                f"{output_channels}"
            )
    except (KeyError, OSError, TypeError, ValueError, FileExistsError) as exc:
        print(f"[중단] 설정 오류: {exc}", file=sys.stderr)
        return 2

    # 출력 장치를 열기 전에 S32_LE 원시 코드 다양성/RMS/클리핑을 검사한다.
    try:
        preflight = capture_input_probe(
            audio,
            seconds=args.preflight_seconds,
            min_rms_dbfs=-80.0,
            max_clip_ratio=0.005,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] raw int32 입력 preflight 실패: {exc}", file=sys.stderr)
        return 2
    if len(preflight.get("channels", [])) < 2 or not all(
        bool(item.get("valid")) for item in preflight["channels"][:2]
    ):
        print(
            "[중단] ERR/REF raw int32 preflight가 모두 PASS하지 않아 출력하지 않습니다.",
            file=sys.stderr,
        )
        return 2

    import sounddevice as sd

    input_device = resolve_alsa_portaudio_device(
        audio["input"]["card"], audio["input"]["pcm"], "input", 2
    )
    output_device = resolve_alsa_portaudio_device(
        audio["output"]["card"], audio["output"]["pcm"], "output", 2
    )
    programs, bounds, _duration_plan = build_channel_probe_plan(
        sample_rate=sample_rate,
        frequency=args.frequency,
        amplitude=args.amplitude,
        block_size=args.block_size,
        pre_seconds=args.pre_seconds,
        tone_seconds=args.tone_seconds,
        post_seconds=args.post_seconds,
        preflight_seconds=args.preflight_seconds,
        output_channels=(
            ("noise_out", int(channels["noise_out"])),
            ("cancel_out", int(channels["cancel_out"])),
        ),
    )

    print(
        f"[저레벨 경로 진단] {args.frequency:.1f}Hz, peak={args.amplitude:.4f}, "
        f"block={args.block_size}, latency={args.latency}"
    )
    print("noise_out과 cancel_out을 한 번에 하나씩 구동합니다. FxLMS는 실행하지 않습니다.")
    try:
        captures, silence_report = collect_channel_paths(
            sd,
            input_device=input_device,
            output_device=output_device,
            sample_rate=sample_rate,
            block_size=args.block_size,
            latency=args.latency,
            programs=programs,
        )
    except (OSError, RuntimeError, ValueError, sd.PortAudioError) as exc:
        print(
            f"[실패] 채널 경로 측정 실패: {exc}. 출력 스트림은 닫고 양 채널 0 flush를 시도했습니다.",
            file=sys.stderr,
        )
        return 1

    path_reports: dict[str, Any] = {}
    matrix_delta: list[list[float]] = []
    matrix_detected: list[list[bool]] = []
    for name, channel, _program in programs:
        capture = captures[name]
        microphones = analyze_path_capture(
            capture["raw_input"],
            bounds=bounds,
            frequency=args.frequency,
            sample_rate=sample_rate,
            status_report=capture["status"],
        )
        path_reports[name] = {
            "output_channel": channel,
            "elapsed_seconds": capture["elapsed_seconds"],
            "callback_status": capture["status"],
            "microphones": microphones,
        }
        matrix_delta.append([float(item["tone_bin_change_db"]) for item in microphones])
        matrix_detected.append([bool(item["coupling_detected"]) for item in microphones])

    summary: dict[str, Any] = {
        "schema_version": 1,
        "measurement_kind": "low_level_output_microphone_channel_path_probe",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hardware_config": str(hardware_path.relative_to(REPO_ROOT.resolve())),
        "sample_rate": sample_rate,
        "frequency_hz": float(args.frequency),
        "amplitude_peak": float(args.amplitude),
        "maximum_allowed_amplitude_peak": MAX_AMPLITUDE,
        "block_size": int(args.block_size),
        "latency": args.latency,
        "segments_seconds": {
            "pre_silence": float(args.pre_seconds),
            "tone": float(args.tone_seconds),
            "post_silence": float(args.post_seconds),
        },
        "input_device": int(input_device),
        "output_device": int(output_device),
        "raw_int32_preflight": preflight,
        "paths": path_reports,
        "matrix_rows": ["noise_out", "cancel_out"],
        "matrix_columns": ["error_mic", "reference_mic"],
        "tone_bin_change_db_matrix": matrix_delta,
        "coupling_detected_matrix": matrix_detected,
        "final_silence_flush": silence_report,
        "raw_capture_npz": str(npz_path.relative_to(REPO_ROOT.resolve())),
        "fxlms_applied": False,
        "performance_claim_allowed": False,
        "duct_identification_complete": False,
        "limitations": [
            "단일 tone의 출력-마이크 결합/채널 배선 진단이다.",
            "P(z), S(z), 광대역 전달함수, 절대 지연을 식별하지 않는다.",
            "ANC 감쇠 또는 FxLMS 성공을 주장하는 데 사용할 수 없다.",
        ],
    }
    try:
        _save_results(npz_path, json_path, summary, captures)
    except (OSError, ValueError, FileExistsError) as exc:
        print(f"[실패] 결과 저장 실패: {exc}", file=sys.stderr)
        return 1

    for name in ("noise_out", "cancel_out"):
        report = path_reports[name]
        print(f"{name} ch{report['output_channel']}:")
        for item in report["microphones"]:
            verdict = "DETECTED" if item["coupling_detected"] else "UNRESOLVED"
            print(
                f"  {item['microphone']}: {verdict}, "
                f"tone-bin {item['tone_bin_change_db']:+.2f}dB, "
                f"RMS {item['rms_change_db']:+.2f}dB"
            )
    print(f"저장: {json_path}")
    print(f"원시 ERR/REF S32_LE 저장: {npz_path}")
    print("이 결과는 채널 경로 진단이며 덕트 식별/FxLMS 성능 결과가 아닙니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
