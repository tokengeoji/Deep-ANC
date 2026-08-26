#!/usr/bin/env python3
"""저레벨 단일 톤으로 출력 2채널과 ERR/REF 마이크의 결합 경로를 확인한다.

이 스크립트는 ``noise_out``과 ``cancel_out``을 한 번에 하나씩 구동하고,
각 구동 전/후 무음과 ERR/REF 원시 S32_LE 입력을 함께 저장한다. 결과는 300 Hz
한 점의 채널/배선 진단일 뿐이며, 덕트 P(z)/S(z) 식별이나 FxLMS 성능 측정이 아니다.

안전 실행 예::

    .venv/bin/python scripts/bench/measure_channel_paths.py \
      --confirm-user-present-volume-minimum
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
    if not 0.0 < frequency < sample_rate / 2.0:
        raise ValueError("frequency는 0보다 크고 Nyquist보다 작아야 합니다")
    if not 0.0 < amplitude <= MAX_AMPLITUDE:
        raise ValueError(
            f"amplitude는 0보다 크고 {MAX_AMPLITUDE:.3f} 이하여야 합니다"
        )
    if block_size <= 0:
        raise ValueError("block-size는 양수여야 합니다")
    if pre_seconds < 0.5 or post_seconds < 0.5:
        raise ValueError("안전을 위해 pre/post 무음은 각각 0.5초 이상이어야 합니다")
    if tone_seconds < 1.0:
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
                    "300Hz 출력-마이크 결합 증거"
                    if coupling_detected
                    else "현재 SNR에서 300Hz 결합을 확정하지 못함"
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
    seconds: float = 0.25,
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


def _output_paths(value: str | None) -> tuple[Path, Path]:
    if value is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = REPO_ROOT / "results" / "channel_paths" / f"channel_paths_{stamp}"
    else:
        prefix = _repo_path(value)
        if prefix.suffix.lower() in {".json", ".npz"}:
            prefix = prefix.with_suffix("")
    return prefix.with_suffix(".npz"), prefix.with_suffix(".json")


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


def main(argv: list[str] | None = None) -> int:
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
        "--confirm-user-present-volume-minimum",
        action="store_true",
        help="사용자 입회와 물리 앰프 볼륨 최저를 확인",
    )
    parser.add_argument("--confirm-speaker", action="store_true")
    args = parser.parse_args(argv)

    if not (args.confirm_user_present_volume_minimum and args.confirm_speaker):
        print(
            "[중단] 사용자 입회와 앰프 볼륨 최저를 확인한 뒤 "
            "--confirm-user-present-volume-minimum 및 --confirm-speaker를 지정하세요.",
            file=sys.stderr,
        )
        return 2

    try:
        hardware_path = _repo_path(args.hardware)
        hardware_cfg = load_yaml(hardware_path)
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
        if args.preflight_seconds <= 0.0:
            raise ValueError("preflight-seconds는 양수여야 합니다")
        npz_path, json_path = _output_paths(args.out_prefix)
        output_channels = (int(channels["noise_out"]), int(channels["cancel_out"]))
        if set(output_channels) != {0, 1}:
            raise ValueError(
                "noise_out/cancel_out은 서로 다른 stereo 채널 0/1이어야 합니다: "
                f"{output_channels}"
            )
        if npz_path.exists() or json_path.exists():
            raise FileExistsError(
                "기존 측정 결과는 덮어쓰지 않습니다. 다른 --out-prefix를 쓰세요"
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
    bounds: dict[str, tuple[int, int]] | None = None
    programs: list[tuple[str, int, np.ndarray]] = []
    for name, config_key in (("noise_out", "noise_out"), ("cancel_out", "cancel_out")):
        channel = int(channels[config_key])
        program, current_bounds = build_output_program(
            sample_rate=sample_rate,
            frequency=args.frequency,
            amplitude=args.amplitude,
            output_channel=channel,
            pre_seconds=args.pre_seconds,
            tone_seconds=args.tone_seconds,
            post_seconds=args.post_seconds,
        )
        if bounds is None:
            bounds = current_bounds
        programs.append((name, channel, program))
    assert bounds is not None

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
            "300Hz 한 점의 출력-마이크 결합/채널 배선 진단이다.",
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
