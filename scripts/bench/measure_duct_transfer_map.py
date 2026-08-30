#!/usr/bin/env python3
"""저음량 단일 스트림으로 덕트의 2×2 시간–주파수 전달맵을 측정한다.

``noise_out``과 ``cancel_out``을 같은 full-duplex PortAudio 스트림에서 시간분할로
한 채널씩 구동하고 ERR/REF를 동시에 수집한다. 네 경로(NS→REF, NS→ERR,
CS→REF, CS→ERR)의 반복별 IR과 주파수응답을 계산한다.

이 도구는 덕트 식별 도구이지 ANC/FxLMS 성능 평가기가 아니다. 모든 안전·일관성
게이트가 통과해야만 ``duct_identification_complete=true``가 되며, 통과하더라도
``anc_performance_claim_allowed``는 항상 false다.

실기 실행은 사용자 입회와 물리 앰프 볼륨 최저 상태에서만 허용한다::

    .venv/bin/python scripts/bench/measure_duct_transfer_map.py \
      --confirm-volume-minimum

출력 peak는 0.02를 넘을 수 없고, 기본값은 0.005다. 결과는 덮어쓰지 않는
NPZ/JSON/Markdown과 선택 PNG로 ``results/duct_transfer_map`` 아래에 저장한다.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import fft, signal

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "src"))

from deep_anc.audio_io import (  # noqa: E402
    analyze_int32_input_probe,
    float32_to_pcm_int16,
    pcm_int32_to_float32,
    resolve_alsa_portaudio_device,
)
from deep_anc.config import REPO_ROOT, load_yaml  # noqa: E402
from deep_anc.audio_io import assert_measurement_preconditions  # noqa: E402
from deep_anc.dsp.measurement_level import assert_live_pcm_clock_preconditions  # noqa: E402
from scripts.bench import measure_channel_paths  # noqa: E402
from scripts.data import calibrate_wideband as calibration  # noqa: E402


DEFAULT_BAND_HZ = (80.0, 1600.0)
DEFAULT_REPORT_BANDS_HZ = (
    (80.0, 125.0),
    (125.0, 250.0),
    (250.0, 500.0),
    (500.0, 1000.0),
    (1000.0, 1600.0),
)
PATH_ORDER = ("ns_to_ref", "ns_to_err", "cs_to_ref", "cs_to_err")
PATH_LABELS = {
    "ns_to_ref": "NS→REF",
    "ns_to_err": "NS→ERR",
    "cs_to_ref": "CS→REF",
    "cs_to_err": "CS→ERR",
}
DRIVE_TO_PREFIX = {"noise_out": "ns", "cancel_out": "cs"}
DEFAULT_AMPLITUDE = 0.005
MAX_AMPLITUDE = 0.02
MIN_REPEATS = calibration.MIN_REPEATS
MIN_REPEAT_CONSISTENCY = calibration.MIN_CONSISTENCY
MIN_BAND_COHERENCE = 0.80
MAX_INPUT_CLIP_RATIO = calibration.MAX_INPUT_CLIP_RATIO
MIN_TDOA_CONFIDENCE = 1.05
MIN_TDOA_BAND_RMS_DBFS = -100.0
MIN_DRIVEN_EXCESS_DB = 6.0
MIN_IR_PEAK_TO_NOISE_DB = 12.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument("--duct", default="configs/duct.yaml")
    parser.add_argument("--excitation", choices=("ess", "multitone"), default="ess")
    parser.add_argument(
        "--band",
        type=float,
        nargs=2,
        default=list(DEFAULT_BAND_HZ),
        help="정식 전달맵은 현재 80 1600으로 고정",
    )
    parser.add_argument("--excitation-seconds", type=float, default=4.0)
    parser.add_argument("--gap-seconds", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=MIN_REPEATS)
    parser.add_argument("--multitone-count", type=int, default=48)
    parser.add_argument("--amplitude", type=float, default=DEFAULT_AMPLITUDE)
    parser.add_argument("--fir-length", type=int, default=4096)
    parser.add_argument("--pre-roll", type=int, default=32)
    parser.add_argument("--max-delay-ms", type=float, default=250.0)
    parser.add_argument("--max-delay-jitter-ms", type=float, default=1.0)
    parser.add_argument("--max-tdoa-ms", type=float, default=20.0)
    parser.add_argument("--max-tdoa-jitter-ms", type=float, default=0.25)
    parser.add_argument("--max-timestamp-jitter-ms", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--latency", choices=("low", "high"), default="high")
    parser.add_argument("--input-probe-seconds", type=float, default=2.0)
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="results/ 아래 결과 prefix(.npz/.json/.md는 자동 부여)",
    )
    parser.add_argument("--plot", action="store_true", help="선택 PNG 요약 그래프 저장")
    parser.add_argument(
        "--confirm-volume-minimum",
        action="store_true",
        help="사용자 입회 및 물리 앰프 볼륨 최저를 확인",
    )
    parser.add_argument("--confirm-speaker", action="store_true")
    parser.add_argument("--confirm-user-present", action="store_true")
    return parser


def _repository_path(value: str | Path, *, require_results: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    root = REPO_ROOT.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Deep_ANC 밖의 경로는 사용할 수 없습니다: {path}") from exc
    if require_results and (not relative.parts or relative.parts[0] != "results"):
        raise ValueError(f"측정 결과는 results/ 아래에만 저장해야 합니다: {path}")
    return path


def output_paths(prefix: str | None, *, include_plot: bool) -> dict[str, Path]:
    if prefix is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = REPO_ROOT / "results" / "duct_transfer_map" / f"transfer_map_{stamp}"
    else:
        base = _repository_path(prefix, require_results=True)
        if base.suffix.lower() in {".npz", ".json", ".md", ".png"}:
            base = base.with_suffix("")
    paths = {
        "npz": base.with_suffix(".npz"),
        "json": base.with_suffix(".json"),
        "markdown": base.with_suffix(".md"),
        # 판정을 뒤집는 것은 항상 **반복별 개별 관측치**(onset, TDOA, coherence)다.
        # 그 값이 중첩 JSON 안에만 있으면 실행 간 비교를 사람이 손으로 해야 한다.
        "paths_csv": base.with_name(base.name + "_paths").with_suffix(".csv"),
        "repeats_csv": base.with_name(base.name + "_repeats").with_suffix(".csv"),
    }
    if include_plot:
        paths["plot"] = base.with_suffix(".png")
    return paths


def ensure_no_overwrite(paths: dict[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("기존 측정 결과는 덮어쓰지 않습니다: " + ", ".join(existing))


def _artifact_display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        # 정상 CLI는 results/ 밖을 거부한다. 테스트/임베딩 호출의 명시적 절대 경로만
        # 진단 metadata에서 안전하게 표현한다.
        return str(path.resolve())


def validate_options(
    args: argparse.Namespace,
    *,
    sample_rate: int,
    hardware_channels: dict[str, Any],
) -> dict[str, int]:
    if sample_rate <= 0:
        raise ValueError("sample_rate는 양수여야 합니다")
    if args.repeats < MIN_REPEATS:
        raise ValueError(f"--repeats는 {MIN_REPEATS} 이상이어야 합니다")
    if not 0.0 < float(args.amplitude) <= MAX_AMPLITUDE:
        raise ValueError(f"--amplitude는 0 초과 {MAX_AMPLITUDE} 이하여야 합니다")
    f_low, f_high = map(float, args.band)
    if not 0.0 < f_low < f_high < sample_rate / 2.0:
        raise ValueError("--band는 0 < 하한 < 상한 < Nyquist여야 합니다")
    if not np.allclose((f_low, f_high), DEFAULT_BAND_HZ, rtol=0.0, atol=1e-9):
        raise ValueError(
            "현재 정식 전달맵/대역 게이트는 --band 80 1600만 지원합니다"
        )
    if float(args.excitation_seconds) <= 0.0 or float(args.gap_seconds) < 0.25:
        raise ValueError("excitation-seconds는 양수, gap-seconds는 0.25초 이상이어야 합니다")
    if int(args.multitone_count) < 4:
        raise ValueError("--multitone-count는 4 이상이어야 합니다")
    if int(args.fir_length) < 16 or int(args.pre_roll) < 0:
        raise ValueError("fir-length는 16 이상이고 pre-roll은 음수가 아니어야 합니다")
    required_quiet_gap = (
        float(args.max_delay_ms) / 1000.0
        + int(args.fir_length) / float(sample_rate)
        + 0.10
    )
    if float(args.gap_seconds) < required_quiet_gap:
        raise ValueError(
            "gap-seconds는 max-delay+FIR tail 뒤 최소 0.10초 noise floor를 "
            f"남겨야 합니다: {required_quiet_gap:.3f}초 이상"
        )
    positive = (
        float(args.max_delay_ms),
        float(args.max_delay_jitter_ms),
        float(args.max_tdoa_ms),
        float(args.max_tdoa_jitter_ms),
        float(args.max_timestamp_jitter_ms),
        float(args.input_probe_seconds),
    )
    if any(value <= 0.0 for value in positive):
        raise ValueError("지연·지터·preflight 시간 인자는 모두 양수여야 합니다")
    if args.block_size is not None and int(args.block_size) <= 0:
        raise ValueError("--block-size는 양수여야 합니다")

    channels = {
        key: int(hardware_channels[key])
        for key in ("error_mic", "reference_mic", "noise_out", "cancel_out")
    }
    if {channels["error_mic"], channels["reference_mic"]} != {0, 1}:
        raise ValueError("ERR/REF 입력은 서로 다른 stereo 채널 0/1이어야 합니다")
    if {channels["noise_out"], channels["cancel_out"]} != {0, 1}:
        raise ValueError("noise_out/cancel_out은 서로 다른 stereo 채널 0/1이어야 합니다")
    return channels


def build_multitone(
    f_low: float,
    f_high: float,
    seconds: float,
    sample_rate: int,
    amplitude: float,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """FFT bin 정렬 Schroeder 위상의 저 crest-factor multitone을 만든다."""
    frames = int(round(seconds * sample_rate))
    if frames < 16:
        raise ValueError("multitone 길이가 너무 짧습니다")
    requested = np.geomspace(float(f_low), float(f_high), int(count))
    bins = np.unique(np.rint(requested * frames / sample_rate).astype(np.int64))
    bins = bins[(bins > 0) & (bins < frames // 2)]
    if bins.size < 4:
        raise ValueError("요청 대역에서 고유 multitone FFT bin이 4개 미만입니다")
    frequencies = bins.astype(np.float64) * sample_rate / frames
    n = np.arange(frames, dtype=np.float64)
    tone_indices = np.arange(bins.size, dtype=np.float64)
    phases = -np.pi * tone_indices * (tone_indices - 1.0) / bins.size
    values = np.sum(
        np.sin(2.0 * np.pi * frequencies[:, None] * n[None, :] / sample_rate + phases[:, None]),
        axis=0,
    )
    peak = float(np.max(np.abs(values)))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("multitone 생성 결과가 유효하지 않습니다")
    values *= float(amplitude) / peak
    fade = min(max(1, int(round(0.05 * sample_rate))), frames // 2)
    ramp = np.sin(np.linspace(0.0, np.pi / 2.0, fade)) ** 2
    values[:fade] *= ramp
    values[-fade:] *= ramp[::-1]
    return values.astype(np.float32), frequencies


def build_excitation(
    *,
    kind: str,
    band_hz: tuple[float, float],
    seconds: float,
    sample_rate: int,
    amplitude: float,
    multitone_count: int,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    if kind == "ess":
        values, inverse = calibration.ess_pair(
            band_hz[0], band_hz[1], seconds, sample_rate, amplitude
        )
        return values, inverse, {"kind": "ess", "tone_frequencies_hz": []}
    if kind == "multitone":
        values, frequencies = build_multitone(
            band_hz[0], band_hz[1], seconds, sample_rate, amplitude, multitone_count
        )
        return values, None, {
            "kind": "multitone",
            "tone_frequencies_hz": frequencies.tolist(),
        }
    raise ValueError(f"지원하지 않는 excitation입니다: {kind}")


def build_time_division_program(
    excitation: np.ndarray,
    *,
    sample_rate: int,
    gap_seconds: float,
    repeats: int,
    noise_channel: int,
    cancel_channel: int,
) -> tuple[np.ndarray, dict[str, tuple[int, int]], int]:
    """한 스트림에서 NS epoch 다음 CS epoch를 재생하는 float32 프로그램."""
    values = np.asarray(excitation, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("excitation은 비어 있지 않은 finite 1-D 배열이어야 합니다")
    if {int(noise_channel), int(cancel_channel)} != {0, 1}:
        raise ValueError("출력 채널은 서로 다른 0/1이어야 합니다")
    gap_samples = int(round(float(gap_seconds) * sample_rate))
    if gap_samples <= 0:
        raise ValueError("gap은 한 sample 이상이어야 합니다")
    one_shot = np.concatenate(
        [np.zeros(gap_samples, np.float32), values, np.zeros(gap_samples, np.float32)]
    )
    epoch = np.tile(one_shot, int(repeats))
    separator = np.zeros(gap_samples, dtype=np.float32)
    total = epoch.size * 2 + separator.size
    output = np.zeros((total, 2), dtype=np.float32)
    ns_bounds = (0, int(epoch.size))
    separator_bounds = (ns_bounds[1], ns_bounds[1] + int(separator.size))
    cs_bounds = (separator_bounds[1], total)
    output[ns_bounds[0] : ns_bounds[1], int(noise_channel)] = epoch
    output[cs_bounds[0] : cs_bounds[1], int(cancel_channel)] = epoch
    bounds = {
        "noise_out": ns_bounds,
        "separator": separator_bounds,
        "cancel_out": cs_bounds,
    }
    return output, bounds, gap_samples


def _time_info_value(time_info: Any, name: str) -> float:
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


def capture_time_division(
    sd: Any,
    *,
    sample_rate: int,
    block_size: int,
    latency: str,
    input_device: int,
    output_device: int,
    output_float: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """단일 full-duplex 스트림 캡처와 callback ADC/DAC timestamp를 보존한다."""
    output = np.asarray(output_float, dtype=np.float32)
    if output.ndim != 2 or output.shape[1] != 2:
        raise ValueError("output_float는 [frames, 2]여야 합니다")
    if output.size and float(np.max(np.abs(output))) > MAX_AMPLITUDE + 1e-9:
        raise ValueError("출력이 안전 amplitude 상한을 초과했습니다")
    output_pcm = float32_to_pcm_int16(output)
    total = int(output.shape[0])
    recorded = np.zeros((total, 2), dtype=np.int32)
    cursor = {"frames": 0}
    telemetry: dict[str, Any] = {
        "callback_count": 0,
        "callback_status_count": 0,
        "xrun_count": 0,
        "priming_output_count": 0,
        "unexpected_status_count": 0,
        "statuses": [],
        "callback_time_info": [],
        "callback_error": None,
        "completed": False,
    }

    def callback(indata, outdata, frames, time_info, status):
        outdata.fill(0)
        try:
            start = int(cursor["frames"])
            telemetry["callback_count"] += 1
            telemetry["callback_time_info"].append(
                {
                    "frame_start": start,
                    "frames": int(frames),
                    "input_buffer_adc_time": _time_info_value(
                        time_info, "inputBufferAdcTime"
                    ),
                    "current_time": _time_info_value(time_info, "currentTime"),
                    "output_buffer_dac_time": _time_info_value(
                        time_info, "outputBufferDacTime"
                    ),
                }
            )
            if status:
                item = calibration._status_snapshot(status)
                telemetry["callback_status_count"] += 1
                telemetry["xrun_count"] += int(item["is_xrun"])
                telemetry["priming_output_count"] += int(item["priming_output"])
                telemetry["unexpected_status_count"] += int(item["unexpected"])
                telemetry["statuses"].append(item)

            count = min(int(frames), total - start)
            if count > 0:
                recorded[start : start + count] = np.asarray(
                    indata[:count, :2], dtype=np.int32
                )
                outdata[:count, :2] = output_pcm[start : start + count]
                cursor["frames"] = start + count
            if cursor["frames"] >= total:
                telemetry["completed"] = True
                raise sd.CallbackStop
        except sd.CallbackStop:
            raise
        except Exception as exc:
            outdata.fill(0)
            telemetry["callback_error"] = f"{type(exc).__name__}: {exc}"
            raise sd.CallbackAbort

    stream = None
    normal_completion = False
    started = time.monotonic()
    deadline = started + total / sample_rate + 15.0
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
        while not telemetry["completed"]:
            if telemetry["callback_error"]:
                raise RuntimeError(f"오디오 콜백 실패: {telemetry['callback_error']}")
            if time.monotonic() >= deadline:
                raise TimeoutError("오디오 콜백 완료 대기 시간이 초과되었습니다")
            time.sleep(0.02)
        # CallbackStop 뒤 PortAudio가 마지막 queue를 정상 drain하여 inactive가 될
        # 때까지 기다린다. 정상 경로에서 abort하면 마지막 자극/입력 tail이 잘릴 수 있다.
        drain_deadline = min(deadline, time.monotonic() + 5.0)
        while bool(getattr(stream, "active", False)):
            if time.monotonic() >= drain_deadline:
                raise TimeoutError("마지막 PortAudio 출력 queue drain이 완료되지 않았습니다")
            time.sleep(0.01)
        normal_completion = True
    finally:
        if stream is not None:
            if normal_completion:
                try:
                    stream.stop()
                except Exception:
                    pass
            else:
                try:
                    stream.abort()
                except Exception:
                    pass
            try:
                stream.close()
            except Exception:
                pass

    telemetry["captured_frames"] = int(cursor["frames"])
    telemetry["elapsed_seconds"] = float(time.monotonic() - started)
    telemetry["normal_drain_completed"] = bool(normal_completion)
    return recorded, output_pcm, telemetry


def _safe_final_mute(
    sd: Any,
    *,
    output_device: int,
    sample_rate: int,
    block_size: int,
    latency: str,
) -> dict[str, Any]:
    try:
        return measure_channel_paths.flush_output_silence(
            sd,
            output_device=output_device,
            sample_rate=sample_rate,
            block_size=block_size,
            latency=latency,
        )
    except Exception as exc:
        return {
            "attempted": True,
            "both_channels_zero": False,
            "stream_closed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def capture_and_mute(
    sd: Any,
    *,
    final_mute_report: dict[str, Any],
    sample_rate: int,
    block_size: int,
    latency: str,
    input_device: int,
    output_device: int,
    output_float: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """캡처 성공/실패와 무관하게 두 출력에 마지막 zero flush를 수행한다."""
    try:
        return capture_time_division(
            sd,
            sample_rate=sample_rate,
            block_size=block_size,
            latency=latency,
            input_device=input_device,
            output_device=output_device,
            output_float=output_float,
        )
    finally:
        final_mute_report.update(
            _safe_final_mute(
                sd,
                output_device=output_device,
                sample_rate=sample_rate,
                block_size=block_size,
                latency=latency,
            )
        )


def summarize_callback_times(
    rows: list[dict[str, Any]],
    *,
    sample_rate: int,
    max_jitter_seconds: float,
    frame_bounds: tuple[int, int] | None = None,
) -> dict[str, Any]:
    selected = []
    for row in rows:
        start = int(row.get("frame_start", 0))
        stop = start + int(row.get("frames", 0))
        if frame_bounds is None or (stop > frame_bounds[0] and start < frame_bounds[1]):
            selected.append(row)
    frame_start = np.asarray(
        [row.get("frame_start", np.nan) for row in selected], dtype=np.float64
    )
    adc = np.asarray(
        [row.get("input_buffer_adc_time", np.nan) for row in selected], dtype=np.float64
    )
    dac = np.asarray(
        [row.get("output_buffer_dac_time", np.nan) for row in selected], dtype=np.float64
    )
    current = np.asarray(
        [row.get("current_time", np.nan) for row in selected], dtype=np.float64
    )
    finite = (
        np.isfinite(frame_start)
        & np.isfinite(adc)
        & np.isfinite(dac)
        & np.isfinite(current)
    )
    all_finite = bool(finite.size >= 2 and np.all(finite))
    frame_start = frame_start[finite]
    adc = adc[finite]
    dac = dac[finite]
    current = current[finite]
    delta = dac - adc
    expected_steps = (
        np.diff(frame_start) / float(sample_rate)
        if frame_start.size >= 2
        else np.empty(0, dtype=np.float64)
    )
    adc_steps = np.diff(adc)
    dac_steps = np.diff(dac)
    current_steps = np.diff(current)
    strictly_progressing = bool(
        expected_steps.size > 0
        and np.all(expected_steps > 0.0)
        and np.all(adc_steps > 0.0)
        and np.all(dac_steps > 0.0)
        and np.all(current_steps > 0.0)
    )
    positive_times = bool(
        adc.size >= 2
        and np.all(adc > 0.0)
        and np.all(dac > 0.0)
        and np.all(current > 0.0)
        and np.all(delta > 0.0)
        and np.all(delta < 1.0)
    )
    adc_residual = adc_steps - expected_steps
    dac_residual = dac_steps - expected_steps
    current_residual = current_steps - expected_steps
    current_tolerance = max(
        float(max_jitter_seconds),
        0.5 * float(np.median(expected_steps)) if expected_steps.size else 0.0,
    )
    adc_progression_error = (
        float(np.max(np.abs(adc_residual))) if adc_residual.size else float("nan")
    )
    dac_progression_error = (
        float(np.max(np.abs(dac_residual))) if dac_residual.size else float("nan")
    )
    current_progression_error = (
        float(np.max(np.abs(current_residual)))
        if current_residual.size
        else float("nan")
    )
    progression_matches_frames = bool(
        np.isfinite(adc_progression_error)
        and np.isfinite(dac_progression_error)
        and np.isfinite(current_progression_error)
        and adc_progression_error <= float(max_jitter_seconds)
        and dac_progression_error <= float(max_jitter_seconds)
        and current_progression_error <= current_tolerance
    )
    spread = float(np.ptp(delta)) if delta.size else float("nan")
    stable = bool(
        all_finite
        and positive_times
        and strictly_progressing
        and progression_matches_frames
        and np.isfinite(spread)
        and spread <= float(max_jitter_seconds)
    )
    return {
        "selected_callback_count": len(selected),
        "valid_timestamp_count": int(adc.size),
        "all_selected_timestamps_finite": all_finite,
        "positive_and_plausible": positive_times,
        "strictly_progressing": strictly_progressing,
        "progression_matches_frame_start": progression_matches_frames,
        "maximum_progression_error_seconds": {
            "input_buffer_adc_time": adc_progression_error
            if np.isfinite(adc_progression_error)
            else None,
            "output_buffer_dac_time": dac_progression_error
            if np.isfinite(dac_progression_error)
            else None,
            "current_time": current_progression_error
            if np.isfinite(current_progression_error)
            else None,
        },
        "current_time_progression_tolerance_seconds": current_tolerance,
        "stable": stable,
        "max_allowed_dac_minus_adc_spread_seconds": float(max_jitter_seconds),
        "dac_minus_adc_seconds": {
            "minimum": float(np.min(delta)) if delta.size else None,
            "median": float(np.median(delta)) if delta.size else None,
            "maximum": float(np.max(delta)) if delta.size else None,
            "spread": spread if np.isfinite(spread) else None,
            "median_samples": float(np.median(delta) * sample_rate) if delta.size else None,
        },
        "interpretation": (
            "outputBufferDacTime-inputBufferAdcTime; 음향 TDOA와 별도인 PortAudio 스케줄 시각"
        ),
    }


def cross_epoch_timestamp_consistency(
    timestamp_results: dict[str, Any],
    *,
    sample_rate: int,
    max_difference_seconds: float,
) -> dict[str, Any]:
    """NS/CS epoch의 PortAudio DAC-ADC 공통 offset이 같은지 검사한다."""
    noise = timestamp_results.get("noise_out", {})
    cancel = timestamp_results.get("cancel_out", {})
    noise_offset = noise.get("dac_minus_adc_seconds", {}).get("median")
    cancel_offset = cancel.get("dac_minus_adc_seconds", {}).get("median")
    difference = (
        float(noise_offset) - float(cancel_offset)
        if noise_offset is not None and cancel_offset is not None
        else None
    )
    stable = bool(
        noise.get("stable", False)
        and cancel.get("stable", False)
        and difference is not None
        and abs(difference) <= float(max_difference_seconds)
    )
    return {
        "stable": stable,
        "noise_out_median_dac_minus_adc_seconds": noise_offset,
        "cancel_out_median_dac_minus_adc_seconds": cancel_offset,
        "noise_minus_cancel_offset_seconds": difference,
        "noise_minus_cancel_offset_samples": difference * sample_rate
        if difference is not None
        else None,
        "maximum_allowed_absolute_difference_seconds": float(max_difference_seconds),
        "reason": None if stable else "cross_epoch_portaudio_offset_changed_or_invalid",
    }


def _regularized_repeat_irs(
    response: np.ndarray,
    excitation: np.ndarray,
    *,
    repeats: int,
    gap_samples: int,
    max_delay_samples: int,
    fir_length: int,
    band_hz: tuple[float, float],
    sample_rate: int,
) -> np.ndarray:
    """multitone용 band-limited 반복 IR(Wiener spectral division)."""
    x = np.concatenate(
        [
            np.zeros(gap_samples, np.float64),
            np.asarray(excitation, dtype=np.float64),
            np.zeros(gap_samples, np.float64),
        ]
    )
    values = np.asarray(response, dtype=np.float64).reshape(-1)
    shot_size = x.size
    wanted = int(max_delay_samples) + int(fir_length) + 4096
    nfft = fft.next_fast_len(2 * shot_size - 1)
    spectrum_x = np.fft.rfft(x, n=nfft)
    power = np.abs(spectrum_x) ** 2
    regularizer = max(float(np.max(power)) * 1e-8, np.finfo(np.float64).eps)
    frequencies = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    band_mask = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    results = []
    for repeat in range(int(repeats)):
        segment = values[repeat * shot_size : (repeat + 1) * shot_size]
        if segment.size != shot_size:
            raise ValueError("multitone 반복 응답 길이가 부족합니다")
        spectrum_y = np.fft.rfft(segment, n=nfft)
        transfer = spectrum_y * np.conj(spectrum_x) / (power + regularizer)
        transfer[~band_mask] = 0.0
        ir = np.fft.irfft(transfer, n=nfft)
        results.append(ir[:wanted])
    return np.stack(results)


def extract_repeat_path(
    response: np.ndarray,
    *,
    excitation: np.ndarray,
    inverse: np.ndarray | None,
    excitation_kind: str,
    repeats: int,
    gap_samples: int,
    max_delay_samples: int,
    fir_length: int,
    pre_roll: int,
    max_delay_jitter_samples: int,
    band_hz: tuple[float, float],
    sample_rate: int,
) -> tuple[dict[str, Any] | None, np.ndarray, float, list[float], str | None]:
    if excitation_kind == "ess":
        if inverse is None:
            raise ValueError("ESS inverse filter가 없습니다")
        return calibration.extract_path_model(
            err=np.asarray(response, dtype=np.float64),
            sweep=np.asarray(excitation, dtype=np.float32),
            inv=np.asarray(inverse, dtype=np.float32),
            repeats=int(repeats),
            gap_samples=int(gap_samples),
            max_delay_samples=int(max_delay_samples),
            fir_length=int(fir_length),
            pre_roll=int(pre_roll),
            max_delay_jitter_samples=int(max_delay_jitter_samples),
        )
    irs = _regularized_repeat_irs(
        response,
        excitation,
        repeats=repeats,
        gap_samples=gap_samples,
        max_delay_samples=max_delay_samples,
        fir_length=fir_length,
        band_hz=band_hz,
        sample_rate=sample_rate,
    )
    model, consistency, correlations, error = calibration._model_from_repeat_irs(
        irs,
        max_delay_samples=max_delay_samples,
        fir_length=fir_length,
        pre_roll=pre_roll,
        max_delay_jitter_samples=max_delay_jitter_samples,
    )
    return model, irs, consistency, correlations, error


def frequency_response_from_repeat_irs(
    repeat_irs: np.ndarray,
    *,
    sample_rate: int,
    band_hz: tuple[float, float],
    report_bands_hz: tuple[tuple[float, float], ...] = DEFAULT_REPORT_BANDS_HZ,
    excited_frequencies_hz: np.ndarray | None = None,
) -> dict[str, Any]:
    values = np.asarray(repeat_irs, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 16:
        raise ValueError("주파수응답에는 [repeat>=2, samples>=16] IR이 필요합니다")
    nfft = max(8192, 1 << int(math.ceil(math.log2(values.shape[1]))))
    transfer = np.fft.rfft(values, n=nfft, axis=1)
    frequencies = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    mean_transfer = np.mean(transfer, axis=0)
    mean_power = np.mean(np.abs(transfer) ** 2, axis=0)
    coherence = np.abs(mean_transfer) ** 2 / np.maximum(
        mean_power, np.finfo(np.float64).tiny
    )
    coherence = np.clip(coherence, 0.0, 1.0)
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(mean_transfer), 1e-15))
    phase_wrapped_degrees = np.rad2deg(np.angle(mean_transfer))
    support = np.ones(frequencies.shape, dtype=bool)
    sparse = excited_frequencies_hz is not None and len(excited_frequencies_hz) > 0
    if sparse:
        support.fill(False)
        for target in np.asarray(excited_frequencies_hz, dtype=np.float64):
            support[int(np.argmin(np.abs(frequencies - target)))] = True
    phase_unwrapped = np.full(frequencies.shape, np.nan, dtype=np.float64)
    group_delay_seconds = np.full(frequencies.shape, np.nan, dtype=np.float64)
    supported_indices = np.flatnonzero(support)
    if supported_indices.size >= 2:
        supported_phase = np.unwrap(np.angle(mean_transfer[supported_indices]))
        supported_omega = 2.0 * np.pi * frequencies[supported_indices]
        phase_unwrapped[supported_indices] = supported_phase
        group_delay_seconds[supported_indices] = -np.gradient(
            supported_phase, supported_omega, edge_order=1
        )

    band_rows = []
    for index, (low, high) in enumerate(report_bands_hz):
        inclusive_high = index == len(report_bands_hz) - 1
        mask = support & (frequencies >= low) & (
            frequencies <= high if inclusive_high else frequencies < high
        )
        trusted = mask & np.isfinite(coherence) & (coherence >= MIN_BAND_COHERENCE)
        center = math.sqrt(low * high)
        candidate_indices = np.flatnonzero(mask)
        center_index = (
            int(candidate_indices[np.argmin(np.abs(frequencies[candidate_indices] - center))])
            if candidate_indices.size
            else int(np.argmin(np.abs(frequencies - center)))
        )
        band_rows.append(
            {
                "low_hz": float(low),
                "high_hz": float(high),
                "bin_count": int(np.count_nonzero(mask)),
                "trusted_bin_count": int(np.count_nonzero(trusted)),
                "magnitude_db_median": float(np.median(magnitude_db[mask]))
                if np.any(mask)
                else None,
                "phase_degrees_at_geometric_center": float(
                    phase_wrapped_degrees[center_index]
                ),
                "coherence_median": float(np.median(coherence[mask]))
                if np.any(mask)
                else None,
                "group_delay_ms_median": float(
                    1000.0 * np.median(group_delay_seconds[trusted])
                )
                if np.any(trusted)
                else None,
                "valid": bool(
                    np.any(mask)
                    and np.median(coherence[mask]) >= MIN_BAND_COHERENCE
                    and np.any(trusted)
                ),
            }
        )
    overall_mask = (
        support & (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    )
    coherence_median = (
        float(np.median(coherence[overall_mask])) if np.any(overall_mask) else 0.0
    )
    return {
        "frequencies_hz": frequencies,
        "repeat_transfer_complex": transfer,
        "mean_transfer_complex": mean_transfer,
        "magnitude_db": magnitude_db,
        "phase_unwrapped_radians": phase_unwrapped,
        "phase_wrapped_degrees": phase_wrapped_degrees,
        "coherence": coherence,
        "group_delay_seconds": group_delay_seconds,
        "coherence_definition": "|mean(H_repeat)|^2 / mean(|H_repeat|^2)",
        "frequency_support": "multitone_bins_only" if sparse else "dense_ess_bins",
        "supported_frequency_bin_count": int(np.count_nonzero(support & overall_mask)),
        "coherence_median_in_measurement_band": coherence_median,
        "band_rows": band_rows,
    }


def gcc_phat_lag(
    later_candidate: np.ndarray,
    reference: np.ndarray,
    *,
    sample_rate: int,
    band_hz: tuple[float, float],
    max_lag_samples: int,
) -> tuple[int, float]:
    """GCC-PHAT lag. 양수면 첫 신호가 두 번째 신호보다 늦게 도착했다."""
    x = np.asarray(later_candidate, dtype=np.float64).reshape(-1)
    y = np.asarray(reference, dtype=np.float64).reshape(-1)
    size = min(x.size, y.size)
    if size < 16 or max_lag_samples < 1:
        raise ValueError("TDOA 입력/최대 lag가 너무 작습니다")
    x = x[:size] - float(np.mean(x[:size]))
    y = y[:size] - float(np.mean(y[:size]))
    nfft = fft.next_fast_len(2 * size - 1)
    cross = np.fft.rfft(x, nfft) * np.conj(np.fft.rfft(y, nfft))
    frequencies = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    mask = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    normalized = np.zeros_like(cross)
    usable = mask & (np.abs(cross) > np.finfo(np.float64).tiny)
    normalized[usable] = cross[usable] / np.abs(cross[usable])
    correlation = np.fft.irfft(normalized, nfft)
    limit = min(int(max_lag_samples), nfft // 2)
    window = np.concatenate([correlation[-limit:], correlation[: limit + 1]])
    lags = np.arange(-limit, limit + 1, dtype=np.int64)
    magnitudes = np.abs(window)
    peak_index = int(np.argmax(magnitudes))
    peak = float(magnitudes[peak_index])
    guard = max(1, int(round(0.00025 * sample_rate)))
    competitors = magnitudes.copy()
    competitors[max(0, peak_index - guard) : peak_index + guard + 1] = 0.0
    second = float(np.max(competitors)) if competitors.size else 0.0
    confidence = peak / max(second, np.finfo(np.float64).eps)
    return int(lags[peak_index]), float(confidence)


def _band_rms_dbfs(
    values: np.ndarray,
    *,
    sample_rate: int,
    band_hz: tuple[float, float],
) -> float:
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    if samples.size < 16 or not np.all(np.isfinite(samples)):
        return -200.0
    samples = samples - float(np.mean(samples))
    spectrum = np.fft.rfft(samples)
    frequencies = np.fft.rfftfreq(samples.size, 1.0 / sample_rate)
    mask = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    if not np.any(mask):
        return -200.0
    # Parseval: real one-sided spectrum의 내부 bin은 음의 주파수 쌍을 포함한다.
    power = np.abs(spectrum[mask]) ** 2
    if power.size:
        power = 2.0 * power
    mean_square = float(np.sum(power)) / float(samples.size * samples.size)
    if not np.isfinite(mean_square) or mean_square <= 0.0:
        return -200.0
    return max(-200.0, 10.0 * math.log10(mean_square))


def driven_response_snr_report(
    response_epoch: np.ndarray,
    repeat_irs: np.ndarray,
    model: dict[str, Any] | None,
    *,
    repeats: int,
    gap_samples: int,
    excitation_samples: int,
    max_delay_samples: int,
    fir_length: int,
    sample_rate: int,
    band_hz: tuple[float, float],
) -> dict[str, Any]:
    """구동 구간이 동일 shot의 무출력 gap보다 실제로 커졌는지 검사한다."""
    values = np.asarray(response_epoch, dtype=np.float64).reshape(-1)
    irs = np.asarray(repeat_irs, dtype=np.float64)
    shot_size = 2 * int(gap_samples) + int(excitation_samples)
    quiet_tail_samples = int(gap_samples) - int(max_delay_samples) - int(fir_length)
    excess_db: list[float] = []
    active_dbfs: list[float] = []
    floor_dbfs: list[float] = []
    window_valid = bool(
        shot_size > 0
        and quiet_tail_samples > 0
        and values.size >= int(repeats) * shot_size
    )
    if window_valid:
        for repeat in range(int(repeats)):
            segment = values[repeat * shot_size : (repeat + 1) * shot_size]
            pre = segment[: int(gap_samples)]
            post = segment[-quiet_tail_samples:]
            active_stop = min(
                shot_size,
                int(gap_samples)
                + int(excitation_samples)
                + int(max_delay_samples)
                + int(fir_length),
            )
            active = segment[int(gap_samples) : active_stop]
            floor = np.concatenate([pre, post])
            active_level = _band_rms_dbfs(
                active, sample_rate=sample_rate, band_hz=band_hz
            )
            floor_level = _band_rms_dbfs(
                floor, sample_rate=sample_rate, band_hz=band_hz
            )
            active_dbfs.append(active_level)
            floor_dbfs.append(floor_level)
            excess_db.append(float(active_level - floor_level))

    onsets = [] if model is None else list(model.get("repeat_onset_samples") or [])
    peak_to_noise_db: list[float] = []
    if irs.ndim == 2 and len(onsets) == irs.shape[0]:
        for ir, onset_value in zip(irs, onsets):
            if onset_value is None:
                peak_to_noise_db.append(float("nan"))
                continue
            onset = max(0, int(onset_value))
            signal_stop = min(ir.size, onset + int(fir_length))
            direct = ir[onset:signal_stop]
            before_stop = max(0, onset - calibration.ONSET_ENERGY_WINDOW_SAMPLES)
            after_start = min(ir.size, signal_stop)
            noise = np.concatenate([ir[:before_stop], ir[after_start:]])
            peak = float(np.max(np.abs(direct))) if direct.size else 0.0
            if noise.size:
                centered = noise - float(np.median(noise))
                robust_noise_rms = 1.4826 * float(np.median(np.abs(centered)))
            else:
                robust_noise_rms = float("nan")
            if peak > 0.0 and np.isfinite(robust_noise_rms):
                ratio = peak / max(robust_noise_rms, np.finfo(np.float64).eps)
                peak_to_noise_db.append(float(20.0 * math.log10(ratio)))
            else:
                peak_to_noise_db.append(float("nan"))

    excess_valid = bool(
        len(excess_db) == int(repeats)
        and all(np.isfinite(value) and value >= MIN_DRIVEN_EXCESS_DB for value in excess_db)
    )
    ir_snr_valid = bool(
        len(peak_to_noise_db) == int(repeats)
        and all(
            np.isfinite(value) and value >= MIN_IR_PEAK_TO_NOISE_DB
            for value in peak_to_noise_db
        )
    )
    return {
        "valid": bool(window_valid and excess_valid and ir_snr_valid),
        "window_valid": window_valid,
        "quiet_tail_samples": quiet_tail_samples,
        "repeat_active_band_rms_dbfs": active_dbfs,
        "repeat_gap_floor_band_rms_dbfs": floor_dbfs,
        "repeat_driven_excess_db": excess_db,
        "minimum_required_driven_excess_db": MIN_DRIVEN_EXCESS_DB,
        "repeat_ir_peak_to_noise_db": peak_to_noise_db,
        "minimum_required_ir_peak_to_noise_db": MIN_IR_PEAK_TO_NOISE_DB,
        "driven_excess_valid": excess_valid,
        "ir_peak_to_noise_valid": ir_snr_valid,
        "interpretation": (
            "출력 프로그램 존재가 아니라 각 repeat의 excitation window가 pre/post silent gap보다 "
            "실제로 상승했는지 검증"
        ),
    }


def relative_tdoa_by_repeat(
    epoch: np.ndarray,
    *,
    error_channel: int,
    reference_channel: int,
    repeats: int,
    shot_size: int,
    sample_rate: int,
    band_hz: tuple[float, float],
    max_lag_samples: int,
    max_jitter_samples: int,
) -> dict[str, Any]:
    values = np.asarray(epoch, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("TDOA에는 ERR/REF 2채널 epoch가 필요합니다")
    lags: list[int] = []
    confidence: list[float] = []
    err_band_rms_dbfs: list[float] = []
    ref_band_rms_dbfs: list[float] = []
    boundary_hits: list[bool] = []
    for repeat in range(int(repeats)):
        segment = values[repeat * shot_size : (repeat + 1) * shot_size]
        if segment.shape[0] != shot_size:
            raise ValueError("TDOA 반복 epoch 길이가 부족합니다")
        lag, score = gcc_phat_lag(
            segment[:, int(error_channel)],
            segment[:, int(reference_channel)],
            sample_rate=sample_rate,
            band_hz=band_hz,
            max_lag_samples=max_lag_samples,
        )
        lags.append(lag)
        confidence.append(score)
        err_band_rms_dbfs.append(
            _band_rms_dbfs(
                segment[:, int(error_channel)],
                sample_rate=sample_rate,
                band_hz=band_hz,
            )
        )
        ref_band_rms_dbfs.append(
            _band_rms_dbfs(
                segment[:, int(reference_channel)],
                sample_rate=sample_rate,
                band_hz=band_hz,
            )
        )
        boundary_hits.append(abs(lag) >= max(0, int(max_lag_samples) - 1))
    spread = int(max(lags) - min(lags)) if lags else 0
    signal_valid = bool(
        err_band_rms_dbfs
        and min(err_band_rms_dbfs) >= MIN_TDOA_BAND_RMS_DBFS
        and min(ref_band_rms_dbfs) >= MIN_TDOA_BAND_RMS_DBFS
    )
    confidence_valid = bool(
        confidence and min(confidence) >= MIN_TDOA_CONFIDENCE
    )
    boundary_valid = bool(boundary_hits and not any(boundary_hits))
    stable = bool(
        lags
        and spread <= int(max_jitter_samples)
        and signal_valid
        and confidence_valid
        and boundary_valid
    )
    return {
        "sign_convention": "lag_err_minus_ref_samples; positive means ERR arrives after REF",
        "buffer_common_property": (
            "ERR/REF는 같은 I2S 입력 클록이므로 출력/USB 공통 버퍼 지연이 상쇄된 상대 TDOA"
        ),
        "repeat_lag_err_minus_ref_samples": lags,
        "repeat_confidence_peak_ratio": confidence,
        "minimum_required_confidence_peak_ratio": MIN_TDOA_CONFIDENCE,
        "repeat_err_band_rms_dbfs": err_band_rms_dbfs,
        "repeat_ref_band_rms_dbfs": ref_band_rms_dbfs,
        "minimum_required_band_rms_dbfs": MIN_TDOA_BAND_RMS_DBFS,
        "search_boundary_hits": boundary_hits,
        "signal_valid": signal_valid,
        "confidence_valid": confidence_valid,
        "search_boundary_valid": boundary_valid,
        "median_lag_err_minus_ref_samples": float(np.median(lags)) if lags else None,
        "median_lag_err_minus_ref_ms": float(1000.0 * np.median(lags) / sample_rate)
        if lags
        else None,
        "spread_samples": spread,
        "max_allowed_spread_samples": int(max_jitter_samples),
        "stable": stable,
    }


def absolute_delay_report(
    model: dict[str, Any] | None,
    *,
    sample_rate: int,
    timestamp_summary: dict[str, Any],
    max_jitter_samples: int,
) -> dict[str, Any]:
    onsets = [] if model is None else list(model.get("repeat_onset_samples") or [])
    compact = [] if model is None else list(model.get("repeat_delay_samples") or [])
    onsets = [int(value) for value in onsets if value is not None]
    compact = [int(value) for value in compact if value is not None]
    onset_spread = int(max(onsets) - min(onsets)) if onsets else None
    delta = timestamp_summary.get("dac_minus_adc_seconds", {}).get("median")
    corrected_ms = (
        [1000.0 * (value / sample_rate - float(delta)) for value in onsets]
        if onsets and delta is not None
        else []
    )
    corrected_spread_ms = (
        float(max(corrected_ms) - min(corrected_ms)) if corrected_ms else None
    )
    corrected_samples = [value * sample_rate / 1000.0 for value in corrected_ms]
    # band-limited ESS의 robust energy onset은 window 중심/전링잉 때문에 실제 direct
    # onset보다 앞에 나타날 수 있다. 이 음수는 0으로 꾸미지 않고 보존하고,
    # 한 window를 넘으면 DAC→ADC 물리 분해만 unresolved로 표시한다. 같은 스트림의
    # callback frame delay 안정성/차동 지연 판정을 이 해석 플래그와 혼합하지 않는다.
    minimum_allowed_corrected_samples = -float(
        calibration.ONSET_ENERGY_WINDOW_SAMPLES
    )
    corrected_physical_plausible = bool(
        corrected_samples
        and min(corrected_samples) >= minimum_allowed_corrected_samples
    )
    stable = bool(
        model is not None
        and model.get("stable_delay", False)
        and len(onsets) >= MIN_REPEATS
        and onset_spread is not None
        and onset_spread <= int(max_jitter_samples)
        and timestamp_summary.get("stable", False)
        and corrected_samples
    )
    return {
        "state": "stable" if stable else "invalid_or_unstable",
        "stable": stable,
        "repeat_callback_frame_onset_samples": onsets,
        "callback_frame_onset_median_samples": float(np.median(onsets))
        if onsets
        else None,
        "callback_frame_onset_spread_samples": onset_spread,
        "repeat_compact_model_delay_samples": compact,
        "compact_model_delay_median_samples": float(np.median(compact))
        if compact
        else None,
        "timestamp_corrected_dac_to_adc_path_ms": corrected_ms,
        "timestamp_corrected_dac_to_adc_path_samples": corrected_samples,
        "timestamp_corrected_median_ms": float(np.median(corrected_ms))
        if corrected_ms
        else None,
        "timestamp_corrected_median_samples": float(np.median(corrected_samples))
        if corrected_samples
        else None,
        "timestamp_corrected_spread_ms": corrected_spread_ms,
        "timestamp_corrected_physical_plausible": corrected_physical_plausible,
        "minimum_allowed_timestamp_corrected_samples": (
            minimum_allowed_corrected_samples
        ),
        "timestamp_corrected_resolution_note": (
            "ESS robust onset window 때문에 0보다 최대 한 onset window 앞선 값은 "
            "측정 해상도 범위로 보존"
        ),
        "max_allowed_callback_delay_spread_samples": int(max_jitter_samples),
        "separation_note": (
            "callback frame lag(버퍼 포함), PortAudio DAC-ADC offset, 같은 I2S ERR-REF TDOA를 "
            "서로 대체하지 않고 별도 기록"
        ),
    }


def differential_delay_report(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    sample_rate: int,
    max_jitter_samples: int,
    sign_convention: str,
) -> dict[str, Any]:
    a = list(left.get("repeat_callback_frame_onset_samples", []))
    b = list(right.get("repeat_callback_frame_onset_samples", []))
    count = min(len(a), len(b))
    differences = [float(a[i]) - float(b[i]) for i in range(count)]
    a_corrected = list(left.get("timestamp_corrected_dac_to_adc_path_samples", []))
    b_corrected = list(right.get("timestamp_corrected_dac_to_adc_path_samples", []))
    corrected_count = min(len(a_corrected), len(b_corrected))
    corrected_differences = [
        float(a_corrected[i]) - float(b_corrected[i])
        for i in range(corrected_count)
    ]
    spread = float(np.ptp(differences)) if differences else None
    corrected_spread = (
        float(np.ptp(corrected_differences)) if corrected_differences else None
    )
    stable = bool(
        left.get("stable", False)
        and right.get("stable", False)
        and count >= MIN_REPEATS
        and corrected_count >= MIN_REPEATS
        and spread is not None
        and spread <= max_jitter_samples
        and corrected_spread is not None
        and corrected_spread <= max_jitter_samples
    )
    return {
        "sign_convention": sign_convention,
        "repeat_difference_samples": differences,
        "median_difference_samples": float(np.median(differences))
        if differences
        else None,
        "median_difference_ms": float(1000.0 * np.median(differences) / sample_rate)
        if differences
        else None,
        "spread_samples": spread,
        "repeat_timestamp_corrected_difference_samples": corrected_differences,
        "timestamp_corrected_median_difference_samples": float(
            np.median(corrected_differences)
        )
        if corrected_differences
        else None,
        "timestamp_corrected_spread_samples": corrected_spread,
        "stable": stable,
        "single_stream_time_division": True,
    }


def calculate_causal_budget(
    *,
    sample_rate: int,
    noise_tdoa: dict[str, Any],
    ns_to_err_absolute: dict[str, Any],
    cs_to_err_absolute: dict[str, Any],
    cross_epoch_timestamps: dict[str, Any],
    processing_samples: int,
) -> dict[str, Any]:
    ref_lead = noise_tdoa.get("median_lag_err_minus_ref_samples")
    ns_err = ns_to_err_absolute.get("callback_frame_onset_median_samples")
    cs_err = cs_to_err_absolute.get("callback_frame_onset_median_samples")
    ns_err_corrected = ns_to_err_absolute.get("timestamp_corrected_median_samples")
    cs_err_corrected = cs_to_err_absolute.get("timestamp_corrected_median_samples")
    acoustic_valid = bool(
        noise_tdoa.get("stable", False)
        and cs_to_err_absolute.get("stable", False)
        and ref_lead is not None
        and cs_err is not None
    )
    digital_valid = bool(
        ns_to_err_absolute.get("stable", False)
        and cs_to_err_absolute.get("stable", False)
        and cross_epoch_timestamps.get("stable", False)
        and ns_err_corrected is not None
        and cs_err_corrected is not None
    )
    processing = int(processing_samples)
    deadline = float(ref_lead) - float(cs_err) if acoustic_valid else None
    acoustic_late = (
        processing + float(cs_err) - float(ref_lead) if acoustic_valid else None
    )
    digital_lead = (
        processing + float(cs_err_corrected) - float(ns_err_corrected)
        if digital_valid
        else None
    )
    raw_digital_lead = (
        processing + float(cs_err) - float(ns_err)
        if ns_err is not None and cs_err is not None
        else None
    )
    return {
        "valid": bool(acoustic_valid and digital_valid),
        "invalid_if_unstable": True,
        "configured_processing_handoff_samples": processing,
        "cross_epoch_portaudio_common_offset": cross_epoch_timestamps,
        "acoustic_reference": {
            "valid": acoustic_valid,
            "reference_lead_samples": float(ref_lead) if ref_lead is not None else None,
            "reference_lead_ms": float(1000.0 * ref_lead / sample_rate)
            if ref_lead is not None
            else None,
            "processing_deadline_samples": deadline,
            "processing_deadline_ms": float(1000.0 * deadline / sample_rate)
            if deadline is not None
            else None,
            "deadline_margin_after_configured_processing_samples": (
                deadline - processing if deadline is not None else None
            ),
            "cancel_arrival_alignment_error_samples": acoustic_late,
            "alignment_sign": "positive means cancel arrives late at ERR",
            "causal_without_prediction": bool(deadline is not None and deadline >= processing),
        },
        "digital_reference": {
            "valid": digital_valid,
            "required_source_lead_samples": digital_lead,
            "required_source_lead_ms": float(1000.0 * digital_lead / sample_rate)
            if digital_lead is not None
            else None,
            "lead_sign": (
                "positive means controller reference must precede noise playback by this amount"
            ),
            "cancel_vs_noise_arrival_error_at_zero_lead_samples": digital_lead,
            "raw_callback_frame_lead_before_offset_correction_samples": raw_digital_lead,
            "uses_timestamp_corrected_path_difference": True,
        },
        "formulae": {
            "acoustic_processing_deadline": "(NS→ERR - NS→REF) - CS→ERR",
            "digital_required_lead": "processing + CS→ERR - NS→ERR",
        },
    }


def routing_topology_gate(
    *,
    positions_m: dict[str, Any],
    speed_of_sound_mps: float,
    sample_rate: int,
    tdoa_results: dict[str, Any],
    path_results: dict[str, Any],
) -> dict[str, Any]:
    """duct geometry와 TDOA/onset 부호로 NS/CS·ERR/REF 라우팅 반전을 검출한다."""
    speed = float(speed_of_sound_mps)
    if speed <= 0.0:
        return {"valid": False, "reason": "invalid_speed_of_sound", "drives": {}}
    definitions = {
        "noise_out": ("noise_speaker", "ns_to_err", "ns_to_ref"),
        "cancel_out": ("cancel_speaker", "cs_to_err", "cs_to_ref"),
    }
    reports: dict[str, Any] = {}
    for drive, (source_key, err_key, ref_key) in definitions.items():
        try:
            source = float(positions_m[source_key])
            error = float(positions_m["error_mic"])
            reference = float(positions_m["reference_mic"])
        except (KeyError, TypeError, ValueError):
            reports[drive] = {"valid": False, "reason": "geometry_position_missing"}
            continue
        expected = (
            abs(error - source) - abs(reference - source)
        ) / speed * sample_rate
        measured = tdoa_results.get(drive, {}).get(
            "median_lag_err_minus_ref_samples"
        )
        err_onset = path_results.get(err_key, {}).get("absolute_delay", {}).get(
            "callback_frame_onset_median_samples"
        )
        ref_onset = path_results.get(ref_key, {}).get("absolute_delay", {}).get(
            "callback_frame_onset_median_samples"
        )
        onset_difference = (
            float(err_onset) - float(ref_onset)
            if err_onset is not None and ref_onset is not None
            else None
        )
        expected_sign = int(np.sign(expected))
        measured_sign = int(np.sign(measured)) if measured is not None else 0
        onset_sign = int(np.sign(onset_difference)) if onset_difference is not None else 0
        minimum_magnitude = max(1.0, 0.20 * abs(expected))
        maximum_magnitude = max(4.0 * abs(expected), abs(expected) + 0.005 * sample_rate)
        magnitude_valid = bool(
            measured is not None
            and minimum_magnitude <= abs(float(measured)) <= maximum_magnitude
        )
        agreement_tolerance = max(0.002 * sample_rate, 0.75 * abs(expected))
        onset_agrees = bool(
            onset_difference is not None
            and measured is not None
            and abs(onset_difference - float(measured)) <= agreement_tolerance
        )
        stable_inputs = bool(
            tdoa_results.get(drive, {}).get("stable", False)
            and path_results.get(err_key, {}).get("absolute_delay", {}).get(
                "stable", False
            )
            and path_results.get(ref_key, {}).get("absolute_delay", {}).get(
                "stable", False
            )
        )
        valid = bool(
            expected_sign != 0
            and measured_sign == expected_sign
            and onset_sign == expected_sign
            and magnitude_valid
            and onset_agrees
            and stable_inputs
        )
        reports[drive] = {
            "valid": valid,
            "source_position_m": source,
            "expected_lag_err_minus_ref_samples": expected,
            "expected_sign": expected_sign,
            "measured_lag_err_minus_ref_samples": measured,
            "measured_sign": measured_sign,
            "absolute_onset_err_minus_ref_samples": onset_difference,
            "absolute_onset_sign": onset_sign,
            "acceptable_measured_magnitude_samples": [
                minimum_magnitude,
                maximum_magnitude,
            ],
            "onset_vs_gcc_tolerance_samples": agreement_tolerance,
            "magnitude_valid": magnitude_valid,
            "onset_agrees_with_gcc": onset_agrees,
            "stable_inputs": stable_inputs,
            "reason": None if valid else "routing_or_microphone_topology_mismatch",
        }
    return {
        "valid": bool(
            len(reports) == len(definitions)
            and all(report.get("valid", False) for report in reports.values())
        ),
        "sign_convention": "positive means ERR arrives after REF",
        "geometry_model": "1-D direct distance; used only as a routing/topology gate",
        "drives": reports,
    }


def analyze_paths(
    recorded_float: np.ndarray,
    *,
    bounds: dict[str, tuple[int, int]],
    callback_rows: list[dict[str, Any]],
    excitation: np.ndarray,
    inverse: np.ndarray | None,
    excitation_kind: str,
    excited_frequencies_hz: np.ndarray | None,
    repeats: int,
    gap_samples: int,
    channels: dict[str, int],
    sample_rate: int,
    band_hz: tuple[float, float],
    fir_length: int,
    pre_roll: int,
    max_delay_samples: int,
    max_delay_jitter_samples: int,
    max_tdoa_samples: int,
    max_tdoa_jitter_samples: int,
    max_timestamp_jitter_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    floating = np.asarray(recorded_float, dtype=np.float32)
    shot_size = 2 * int(gap_samples) + int(excitation.size)
    path_results: dict[str, Any] = {}
    tdoa_results: dict[str, Any] = {}
    timestamp_results: dict[str, Any] = {}
    for drive in ("noise_out", "cancel_out"):
        start, stop = bounds[drive]
        epoch = floating[start:stop]
        timestamp_results[drive] = summarize_callback_times(
            callback_rows,
            sample_rate=sample_rate,
            max_jitter_seconds=max_timestamp_jitter_seconds,
            frame_bounds=(start, stop),
        )
        tdoa_results[drive] = relative_tdoa_by_repeat(
            epoch,
            error_channel=channels["error_mic"],
            reference_channel=channels["reference_mic"],
            repeats=repeats,
            shot_size=shot_size,
            sample_rate=sample_rate,
            band_hz=band_hz,
            max_lag_samples=max_tdoa_samples,
            max_jitter_samples=max_tdoa_jitter_samples,
        )
        prefix = DRIVE_TO_PREFIX[drive]
        for microphone, config_key in (("ref", "reference_mic"), ("err", "error_mic")):
            key = f"{prefix}_to_{microphone}"
            model, irs, consistency, correlations, extraction_error = extract_repeat_path(
                epoch[:, channels[config_key]],
                excitation=excitation,
                inverse=inverse,
                excitation_kind=excitation_kind,
                repeats=repeats,
                gap_samples=gap_samples,
                max_delay_samples=max_delay_samples,
                fir_length=fir_length,
                pre_roll=pre_roll,
                max_delay_jitter_samples=max_delay_jitter_samples,
                band_hz=band_hz,
                sample_rate=sample_rate,
            )
            frequency = frequency_response_from_repeat_irs(
                irs,
                sample_rate=sample_rate,
                band_hz=band_hz,
                excited_frequencies_hz=excited_frequencies_hz,
            )
            driven_response = driven_response_snr_report(
                epoch[:, channels[config_key]],
                irs,
                model,
                repeats=repeats,
                gap_samples=gap_samples,
                excitation_samples=int(excitation.size),
                max_delay_samples=max_delay_samples,
                fir_length=fir_length,
                sample_rate=sample_rate,
                band_hz=band_hz,
            )
            absolute = absolute_delay_report(
                model,
                sample_rate=sample_rate,
                timestamp_summary=timestamp_results[drive],
                max_jitter_samples=max_delay_jitter_samples,
            )
            band_valid = all(row["valid"] for row in frequency["band_rows"])
            path_valid = bool(
                model is not None
                and model.get("stable_delay", False)
                and np.isfinite(consistency)
                and consistency >= MIN_REPEAT_CONSISTENCY
                and frequency["coherence_median_in_measurement_band"]
                >= MIN_BAND_COHERENCE
                and band_valid
                and driven_response["valid"]
                and absolute["stable"]
                and extraction_error is None
            )
            reasons = []
            if model is None or not model.get("stable_delay", False):
                reasons.append("absolute_delay_unstable")
            if not np.isfinite(consistency) or consistency < MIN_REPEAT_CONSISTENCY:
                reasons.append("repeat_consistency_below_0.9")
            if frequency["coherence_median_in_measurement_band"] < MIN_BAND_COHERENCE:
                reasons.append("measurement_band_coherence_below_0.8")
            if not band_valid:
                reasons.append("one_or_more_frequency_bands_invalid")
            if not driven_response["driven_excess_valid"]:
                reasons.append("speaker_driven_excess_below_6db")
            if not driven_response["ir_peak_to_noise_valid"]:
                reasons.append("deconvolved_ir_peak_to_noise_below_12db")
            if not driven_response["window_valid"]:
                reasons.append("quiet_gap_window_unavailable")
            if not absolute["stable"]:
                reasons.append("timestamp_corrected_absolute_delay_invalid")
            if extraction_error:
                reasons.append("path_extraction_error")
            path_results[key] = {
                "label": PATH_LABELS[key],
                "drive": drive,
                "microphone": microphone,
                "model": model,
                "repeat_irs": irs,
                "repeat_consistency": float(consistency),
                "pairwise_correlations": correlations,
                "frequency": frequency,
                "driven_response_snr": driven_response,
                "absolute_delay": absolute,
                "valid": path_valid,
                "invalid_reasons": reasons,
                "extraction_error": extraction_error,
            }
    return path_results, tdoa_results, timestamp_results


def overall_quality_gate(
    *,
    preflight_report: dict[str, Any],
    measurement_report: dict[str, Any],
    output_float: np.ndarray,
    output_pcm: np.ndarray,
    telemetry: dict[str, Any],
    final_mute: dict[str, Any],
    path_results: dict[str, Any],
    tdoa_results: dict[str, Any],
    timestamp_results: dict[str, Any],
    cross_epoch_timestamps: dict[str, Any],
    differential_results: dict[str, Any],
    routing_topology: dict[str, Any],
    causal_budget: dict[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    preflight_channels = preflight_report.get("channels", [])[:2]
    if len(preflight_channels) != 2 or not all(
        bool(item.get("valid")) for item in preflight_channels
    ):
        reasons.append("preflight_both_mics_invalid")
    measured_channels = measurement_report.get("channels", [])[:2]
    if len(measured_channels) != 2:
        reasons.append("measurement_both_mics_missing")
    else:
        if not all(bool(item.get("valid", False)) for item in measured_channels):
            reasons.append("measurement_both_mics_invalid")
        if any(
            float(item.get("clip_ratio", 1.0)) > MAX_INPUT_CLIP_RATIO
            for item in measured_channels
        ):
            reasons.append("input_clipping")
    peak = float(np.max(np.abs(output_float))) if output_float.size else 0.0
    output_clip = (
        float(np.mean(np.abs(output_float.astype(np.float64)) >= 1.0))
        if output_float.size
        else 0.0
    )
    pcm_saturation = (
        float(np.mean(np.abs(output_pcm.astype(np.int32)) >= np.iinfo(np.int16).max))
        if output_pcm.size
        else 0.0
    )
    if peak > MAX_AMPLITUDE + 1e-9 or output_clip > 0.0 or pcm_saturation > 0.0:
        reasons.append("output_clipping_or_amplitude_limit")
    if int(telemetry.get("xrun_count", 0)) > 0:
        reasons.append("xrun_detected")
    if int(telemetry.get("unexpected_status_count", 0)) > 0:
        reasons.append("unexpected_callback_status")
    if telemetry.get("callback_error"):
        reasons.append("callback_error")
    if not bool(telemetry.get("completed", False)):
        reasons.append("measurement_incomplete")
    if not bool(telemetry.get("normal_drain_completed", False)):
        reasons.append("normal_output_drain_incomplete")
    if not (
        final_mute.get("attempted", False)
        and final_mute.get("both_channels_zero", False)
        and final_mute.get("stream_closed", False)
        and int(final_mute.get("underflow_blocks", 0)) == 0
    ):
        reasons.append("final_mute_unverified")
    for key in PATH_ORDER:
        if not path_results.get(key, {}).get("valid", False):
            reasons.append(f"{key}_invalid")
    for drive in ("noise_out", "cancel_out"):
        if not tdoa_results.get(drive, {}).get("stable", False):
            reasons.append(f"{drive}_relative_tdoa_unstable")
        if not timestamp_results.get(drive, {}).get("stable", False):
            reasons.append(f"{drive}_callback_timestamps_unstable")
    if not cross_epoch_timestamps.get("stable", False):
        reasons.append("cross_epoch_portaudio_offset_changed")
    for name, report in differential_results.items():
        if not report.get("stable", False):
            reasons.append(f"{name}_differential_delay_unstable")
    if not causal_budget.get("valid", False):
        reasons.append("causal_budget_invalid_due_to_unstable_delay")
    if not routing_topology.get("valid", False):
        reasons.append("routing_topology_mismatch")
    reasons = list(dict.fromkeys(reasons))
    return not reasons, reasons, {
        "output_peak": peak,
        "output_clip_ratio": output_clip,
        "output_pcm_saturation_ratio": pcm_saturation,
    }


def _path_json_summary(path: dict[str, Any]) -> dict[str, Any]:
    model = path.get("model") or {}
    frequency = path.get("frequency") or {}
    return {
        "label": path.get("label"),
        "drive": path.get("drive"),
        "microphone": path.get("microphone"),
        "valid": bool(path.get("valid", False)),
        "invalid_reasons": list(path.get("invalid_reasons", [])),
        "repeat_consistency": path.get("repeat_consistency"),
        "pairwise_correlations": path.get("pairwise_correlations", []),
        "repeat_onset_samples": model.get("repeat_onset_samples"),
        "repeat_model_delay_samples": model.get("repeat_delay_samples"),
        "delay_spread_samples": model.get("delay_spread_samples"),
        "absolute_delay": path.get("absolute_delay", {}),
        "driven_response_snr": path.get("driven_response_snr", {}),
        "coherence_definition": frequency.get("coherence_definition"),
        "coherence_median_in_measurement_band": frequency.get(
            "coherence_median_in_measurement_band"
        ),
        "frequency_bands": frequency.get("band_rows", []),
        "npz_arrays": {
            "repeat_irs": f"{path.get('key', '')}_repeat_irs",
            "repeat_transfer_complex": f"{path.get('key', '')}_repeat_transfer_complex",
        },
        "extraction_error": path.get("extraction_error"),
    }


def build_npz_arrays(
    *,
    metadata: dict[str, Any],
    output_float: np.ndarray,
    output_pcm: np.ndarray,
    recorded_raw: np.ndarray,
    preflight_raw: np.ndarray,
    excitation: np.ndarray,
    callback_rows: list[dict[str, Any]],
    path_results: dict[str, Any],
    tdoa_results: dict[str, Any],
) -> dict[str, np.ndarray]:
    rows = callback_rows
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(
            json.dumps(calibration._json_safe(metadata), ensure_ascii=False, sort_keys=True)
        ),
        "output_float32": np.asarray(output_float, dtype=np.float32),
        "output_pcm_int16": np.asarray(output_pcm, dtype=np.int16),
        "input_raw_int32": np.asarray(recorded_raw, dtype=np.int32),
        "input_float32": pcm_int32_to_float32(recorded_raw)
        if recorded_raw.size
        else np.empty((0, 2), dtype=np.float32),
        "preflight_raw_int32": np.asarray(preflight_raw, dtype=np.int32),
        "excitation_float32": np.asarray(excitation, dtype=np.float32),
        "callback_frame_start": np.asarray(
            [row.get("frame_start", 0) for row in rows], dtype=np.int64
        ),
        "callback_frames": np.asarray(
            [row.get("frames", 0) for row in rows], dtype=np.int64
        ),
        "callback_input_buffer_adc_time": np.asarray(
            [row.get("input_buffer_adc_time", np.nan) for row in rows], dtype=np.float64
        ),
        "callback_current_time": np.asarray(
            [row.get("current_time", np.nan) for row in rows], dtype=np.float64
        ),
        "callback_output_buffer_dac_time": np.asarray(
            [row.get("output_buffer_dac_time", np.nan) for row in rows], dtype=np.float64
        ),
    }
    for drive, report in tdoa_results.items():
        arrays[f"{drive}_repeat_lag_err_minus_ref_samples"] = np.asarray(
            report.get("repeat_lag_err_minus_ref_samples", []), dtype=np.int64
        )
    for key, path in path_results.items():
        frequency = path["frequency"]
        arrays[f"{key}_repeat_irs"] = np.asarray(path["repeat_irs"], dtype=np.float64)
        arrays[f"{key}_frequencies_hz"] = np.asarray(
            frequency["frequencies_hz"], dtype=np.float64
        )
        arrays[f"{key}_repeat_transfer_complex"] = np.asarray(
            frequency["repeat_transfer_complex"], dtype=np.complex128
        )
        arrays[f"{key}_mean_transfer_complex"] = np.asarray(
            frequency["mean_transfer_complex"], dtype=np.complex128
        )
        for name in (
            "magnitude_db",
            "phase_unwrapped_radians",
            "phase_wrapped_degrees",
            "coherence",
            "group_delay_seconds",
        ):
            arrays[f"{key}_{name}"] = np.asarray(frequency[name], dtype=np.float64)
    return arrays


def _display_number(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(numeric):
        return "—"
    return f"{numeric:.{int(digits)}f}"


def render_markdown(metadata: dict[str, Any]) -> str:
    result = metadata["result"]
    verdict = "PASS" if result["duct_identification_complete"] else "INVALID"
    lines = [
        "# 덕트 시간–주파수 전달맵",
        "",
        f"- 판정: **{verdict}**",
        f"- 생성 시각(UTC): `{metadata['created_at_utc']}`",
        f"- 자극: `{metadata['configuration']['excitation']}`, "
        f"{metadata['configuration']['band_hz'][0]:.0f}–"
        f"{metadata['configuration']['band_hz'][1]:.0f}Hz, peak "
        f"{metadata['configuration']['amplitude_peak']:.4f}",
        "- FxLMS 적용: **아니오** — 이 보고서는 ANC 감쇠 성능을 주장하지 않습니다.",
        "",
        "## 지연 해석 규약",
        "",
        "ERR–REF TDOA는 같은 I²S 입력 클록에서 계산하며 `ERR-REF > 0`이면 ERR가 늦습니다. "
        "output→mic callback frame lag(PortAudio/USB 버퍼 포함), ADC/DAC callback timestamp, "
        "timestamp 보정 DAC→ADC 경로는 서로 다른 값으로 보존합니다.",
        "",
        "## 네 경로 품질",
        "",
        "| 경로 | 반복 일관성 | 절대 onset(samples) | spread | 대역 coherence | driven excess(min dB) | IR P/N(min dB) | 판정 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in PATH_ORDER:
        path = metadata.get("paths", {}).get(key, {})
        absolute = path.get("absolute_delay", {})
        driven = path.get("driven_response_snr", {})
        onset = absolute.get("callback_frame_onset_median_samples")
        spread = absolute.get("callback_frame_onset_spread_samples")
        driven_excess = driven.get("repeat_driven_excess_db", [])
        ir_peak_noise = driven.get("repeat_ir_peak_to_noise_db", [])
        lines.append(
            f"| {path.get('label', PATH_LABELS[key])} | "
            f"{_display_number(path.get('repeat_consistency'))} | "
            f"{onset if onset is not None else '—'} | {spread if spread is not None else '—'} | "
            f"{_display_number(path.get('coherence_median_in_measurement_band'))} | "
            f"{_display_number(min(driven_excess) if driven_excess else None, 2)} | "
            f"{_display_number(min(ir_peak_noise) if ir_peak_noise else None, 2)} | "
            f"{'PASS' if path.get('valid', False) else 'INVALID'} |"
        )
    lines.extend(
        [
            "",
            "## 같은 I²S 클록 ERR–REF 상대 TDOA",
            "",
            "| 구동 | 반복 lag(samples) | 중앙값(ms) | spread(samples) | 판정 |",
            "|---|---|---:|---:|---|",
        ]
    )
    for drive in ("noise_out", "cancel_out"):
        report = metadata.get("relative_tdoa", {}).get(drive, {})
        lines.append(
            f"| {drive} | `{report.get('repeat_lag_err_minus_ref_samples', [])}` | "
            f"{_display_number(report.get('median_lag_err_minus_ref_ms'))} | "
            f"{report.get('spread_samples', '—')} | "
            f"{'PASS' if report.get('stable', False) else 'INVALID'} |"
        )
    topology = metadata.get("routing_topology", {})
    common_offset = metadata.get("callback_timestamps", {}).get(
        "cross_epoch_common_offset_gate", {}
    )
    lines.extend(
        [
            "",
            f"- duct geometry 라우팅/부호 일치: **{topology.get('valid', False)}**",
            f"- NS/CS epoch PortAudio 공통 offset 일치: "
            f"**{common_offset.get('stable', False)}**",
            "",
            "## 80–1600Hz 대역표",
            "",
        ]
    )
    for key in PATH_ORDER:
        path = metadata.get("paths", {}).get(key)
        if path is None:
            continue
        lines.extend(
            [
                f"### {path['label']}",
                "",
                "| 대역(Hz) | magnitude(dB) | phase@center(°) | coherence | group delay(ms) | 판정 |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in path["frequency_bands"]:
            group = row["group_delay_ms_median"]
            lines.append(
                f"| {row['low_hz']:.0f}–{row['high_hz']:.0f} | "
                f"{_display_number(row.get('magnitude_db_median'), 2)} | "
                f"{_display_number(row.get('phase_degrees_at_geometric_center'), 1)} | "
                f"{_display_number(row.get('coherence_median'))} | "
                f"{_display_number(group)} | "
                f"{'PASS' if row['valid'] else 'INVALID'} |"
            )
        lines.append("")
    causal = metadata.get("causal_budget", {})
    acoustic = causal.get("acoustic_reference", {})
    digital = causal.get("digital_reference", {})
    lines.extend(
        [
            "## 인과성 예산",
            "",
            f"- 계산 유효: **{causal.get('valid', False)}** (지연이 불안정하면 계산 자체를 INVALID 처리)",
            f"- REF 선행시간: `{acoustic.get('reference_lead_samples')}` samples",
            f"- acoustic processing deadline: `{acoustic.get('processing_deadline_samples')}` samples",
            f"- configured handoff 후 cancel 도착 오차: "
            f"`{acoustic.get('cancel_arrival_alignment_error_samples')}` samples "
            "(양수=늦음)",
            f"- digital source required lead: `{digital.get('required_source_lead_samples')}` samples",
            "",
            "## 최종 판정",
            "",
            f"- 무효 사유: `{result['invalid_reasons']}`",
            f"- 덕트 식별 완료 주장 가능: **{result['measurement_claim_allowed']}**",
            "- ANC/FxLMS 성능 주장 가능: **False**",
            "",
        ]
    )
    return "\n".join(lines)


def build_path_rows(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """경로(4종)당 한 행 — 판정과 그 판정을 만든 수치를 한 표에 모은다."""

    rows: list[dict[str, Any]] = []
    for key in PATH_ORDER:
        path = (metadata.get("paths") or {}).get(key)
        if not isinstance(path, dict):
            continue
        absolute = path.get("absolute_delay") or {}
        snr = path.get("driven_response_snr") or {}
        rows.append({
            "path": key,
            "label": PATH_LABELS.get(key, key),
            "valid": bool(path.get("valid")),
            "coherence_median": path.get("coherence_median_in_measurement_band"),
            "min_required_coherence": MIN_BAND_COHERENCE,
            "onset_median_samples": absolute.get("callback_frame_onset_median_samples"),
            "onset_spread_samples": absolute.get("callback_frame_onset_spread_samples"),
            "max_allowed_onset_spread_samples": absolute.get(
                "max_allowed_callback_delay_spread_samples"
            ),
            "onset_stable": absolute.get("stable"),
            "timestamp_corrected_median_samples": absolute.get(
                "timestamp_corrected_median_samples"
            ),
            "timestamp_corrected_spread_ms": absolute.get("timestamp_corrected_spread_ms"),
            "driven_snr_valid": snr.get("valid"),
        })
    return rows


def build_repeat_rows(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """반복 × 경로마다 한 행 — 요약값이 아니라 **개별 관측치**를 남긴다.

    coherence 가 낮을 때 원인이 레벨인지 반복 간 지연 흔들림인지는 이 계열을 봐야 갈린다.
    요약만 남기면 그 판별이 불가능해진다.
    """

    rows: list[dict[str, Any]] = []
    for key in PATH_ORDER:
        path = (metadata.get("paths") or {}).get(key)
        if not isinstance(path, dict):
            continue
        onsets = list((path.get("absolute_delay") or {}).get(
            "repeat_callback_frame_onset_samples"
        ) or [])
        models = list(path.get("repeat_model_delay_samples") or [])
        corrected = list((path.get("absolute_delay") or {}).get(
            "timestamp_corrected_dac_to_adc_path_samples"
        ) or [])
        for index in range(max(len(onsets), len(models), len(corrected))):
            rows.append({
                "path": key,
                "repeat": index,
                "onset_samples": onsets[index] if index < len(onsets) else "",
                "model_delay_samples": models[index] if index < len(models) else "",
                "timestamp_corrected_samples": (
                    corrected[index] if index < len(corrected) else ""
                ),
            })
    for key, tdoa in (metadata.get("relative_tdoa") or {}).items():
        if not isinstance(tdoa, dict):
            continue
        lags = list(tdoa.get("repeat_lag_err_minus_ref_samples") or [])
        scores = list(tdoa.get("repeat_confidence_peak_ratio") or [])
        err_rms = list(tdoa.get("repeat_err_band_rms_dbfs") or [])
        ref_rms = list(tdoa.get("repeat_ref_band_rms_dbfs") or [])
        for index, lag in enumerate(lags):
            rows.append({
                "path": f"tdoa_{key}",
                "repeat": index,
                "tdoa_err_minus_ref_samples": lag,
                "gcc_confidence": scores[index] if index < len(scores) else "",
                "min_required_confidence": tdoa.get("minimum_required_confidence_peak_ratio"),
                "err_band_rms_dbfs": err_rms[index] if index < len(err_rms) else "",
                "ref_band_rms_dbfs": ref_rms[index] if index < len(ref_rms) else "",
            })
    return rows


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for name in row:
            if name not in columns:
                columns.append(name)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def save_artifacts(
    paths: dict[str, Path],
    *,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    markdown: str,
) -> None:
    ensure_no_overwrite(paths)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    core = {key: paths[key] for key in ("npz", "json", "markdown")}
    token = f"{os.getpid()}_{time.time_ns()}"
    temporary = {
        key: path.with_name(f".{path.name}.{token}.tmp")
        for key, path in core.items()
    }
    published: list[Path] = []
    try:
        with temporary["npz"].open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        with temporary["json"].open("x", encoding="utf-8") as handle:
            json.dump(
                calibration._json_safe(metadata),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        with temporary["markdown"].open("x", encoding="utf-8") as handle:
            handle.write(markdown)
            handle.flush()
            os.fsync(handle.fileno())

        # hard-link는 목적지가 있으면 실패하므로 원자적인 no-overwrite publish다.
        # 다중 파일 중 하나라도 실패하면 이 호출에서 publish한 파일만 rollback한다.
        for key in ("npz", "json", "markdown"):
            os.link(temporary[key], core[key])
            published.append(core[key])
        # CSV 는 파생물이다. 실패해도 핵심 산출물을 잃지 않도록 publish 뒤에 쓰고,
        # 실패하면 경고만 남긴다 — 측정은 스피커를 다시 울려야 얻는다.
        for key, builder in (("paths_csv", build_path_rows), ("repeats_csv", build_repeat_rows)):
            target = paths.get(key)
            if target is None:
                continue
            try:
                _write_csv_rows(target, builder(metadata))
            except Exception as exc:  # noqa: BLE001
                print(f"[경고] {key} 생성 실패(핵심 산출물은 저장됨): {exc}", file=sys.stderr)
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def save_plot(path: Path, path_results: dict[str, Any], band_hz: tuple[float, float]) -> None:
    if path.exists():
        raise FileExistsError(f"기존 plot을 덮어쓰지 않습니다: {path}")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for key in PATH_ORDER:
        frequency = path_results[key]["frequency"]
        mask = (frequency["frequencies_hz"] >= band_hz[0]) & (
            frequency["frequencies_hz"] <= band_hz[1]
        )
        axes[0].plot(
            frequency["frequencies_hz"][mask],
            frequency["magnitude_db"][mask],
            label=PATH_LABELS[key],
        )
        axes[1].plot(
            frequency["frequencies_hz"][mask],
            frequency["coherence"][mask],
            label=PATH_LABELS[key],
        )
    axes[0].set_ylabel("Magnitude (dB)")
    axes[1].set_ylabel("Repeat coherence")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylim(0.0, 1.05)
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=2)
    figure.tight_layout()
    with path.open("xb") as handle:
        figure.savefig(handle, format="png", dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # 확인 플래그는 YAML/sounddevice/장치 접근보다 먼저 검사한다.
    if not (args.confirm_volume_minimum and args.confirm_speaker and args.confirm_user_present):
        print(
            "[중단] 스피커 연결·사용자 입회·볼륨 최저 확인 플래그가 필요합니다: "
            "--confirm-volume-minimum --confirm-speaker --confirm-user-present",
            file=sys.stderr,
        )
        return 2

    try:
        hardware_path = _repository_path(args.hardware)
        duct_path = _repository_path(args.duct)
        hardware = load_yaml(hardware_path)
        duct = load_yaml(duct_path)
        audio = hardware["audio"]
        import sounddevice as sd
        assert_live_pcm_clock_preconditions(audio)
        assert_measurement_preconditions(sd, audio)
        sample_rate = int(audio["sample_rate"])
        channels = validate_options(
            args,
            sample_rate=sample_rate,
            hardware_channels=hardware["channels"],
        )
        block_size = int(args.block_size or audio["block_size"])
        paths = output_paths(args.out_prefix, include_plot=bool(args.plot))
        ensure_no_overwrite(paths)
        band_hz = (float(args.band[0]), float(args.band[1]))
        max_delay_samples = int(round(float(args.max_delay_ms) * sample_rate / 1000.0))
        max_delay_jitter_samples = int(
            round(float(args.max_delay_jitter_ms) * sample_rate / 1000.0)
        )
        max_tdoa_samples = int(round(float(args.max_tdoa_ms) * sample_rate / 1000.0))
        max_tdoa_jitter_samples = int(
            round(float(args.max_tdoa_jitter_ms) * sample_rate / 1000.0)
        )
        processing_samples = int(duct["secondary_path"]["handoff_extra_samples"])
    except (KeyError, OSError, TypeError, ValueError, FileExistsError) as exc:
        print(f"[중단] 설정 오류: {exc}", file=sys.stderr)
        return 2

    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    preflight_raw = np.empty((0, 2), dtype=np.int32)
    recorded_raw = np.empty((0, 2), dtype=np.int32)
    output_float = np.empty((0, 2), dtype=np.float32)
    output_pcm = np.empty((0, 2), dtype=np.int16)
    excitation = np.empty(0, dtype=np.float32)
    excitation_meta: dict[str, Any] = {"kind": args.excitation}
    bounds: dict[str, tuple[int, int]] = {}
    preflight_report: dict[str, Any] = {"channels": []}
    measurement_report: dict[str, Any] = {"channels": []}
    telemetry: dict[str, Any] = {"completed": False, "callback_time_info": []}
    final_mute: dict[str, Any] = {"attempted": False}
    path_results: dict[str, Any] = {}
    tdoa_results: dict[str, Any] = {}
    timestamp_results: dict[str, Any] = {}
    cross_epoch_timestamps: dict[str, Any] = {"stable": False}
    differential_results: dict[str, Any] = {}
    routing_topology: dict[str, Any] = {"valid": False}
    causal_budget: dict[str, Any] = {"valid": False}
    invalid_reasons: list[str] = []
    output_summary: dict[str, Any] = {}
    error_message: str | None = None
    measurement_exception = False

    try:
        import sounddevice as sd

        print("출력 없는 ERR/REF raw int32 preflight 중...")
        preflight_raw, preflight_report = calibration._capture_preflight(
            sd, audio, float(args.input_probe_seconds)
        )
        if len(preflight_report.get("channels", [])) < 2 or not all(
            bool(item.get("valid")) for item in preflight_report["channels"][:2]
        ):
            invalid_reasons = ["preflight_both_mics_invalid"]
            error_message = "양 마이크 preflight 실패 — 출력 장치를 열지 않았습니다"
        else:
            input_device = int(preflight_report["device"])
            output_device = resolve_alsa_portaudio_device(
                audio["output"]["card"], audio["output"]["pcm"], "output", 2
            )
            excitation, inverse, excitation_meta = build_excitation(
                kind=str(args.excitation),
                band_hz=band_hz,
                seconds=float(args.excitation_seconds),
                sample_rate=sample_rate,
                amplitude=float(args.amplitude),
                multitone_count=int(args.multitone_count),
            )
            output_float, bounds, gap_samples = build_time_division_program(
                excitation,
                sample_rate=sample_rate,
                gap_seconds=float(args.gap_seconds),
                repeats=int(args.repeats),
                noise_channel=channels["noise_out"],
                cancel_channel=channels["cancel_out"],
            )
            print(
                f"단일 스트림 시간분할 {args.excitation.upper()}: "
                f"noise_out→cancel_out, {band_hz[0]:.0f}–{band_hz[1]:.0f}Hz, "
                f"peak {args.amplitude:.4f}, repeats {args.repeats}"
            )
            recorded_raw, output_pcm, telemetry = capture_and_mute(
                sd,
                final_mute_report=final_mute,
                sample_rate=sample_rate,
                block_size=block_size,
                latency=str(args.latency),
                input_device=input_device,
                output_device=output_device,
                output_float=output_float,
            )
            measurement_report = analyze_int32_input_probe(
                recorded_raw,
                min_rms_dbfs=-120.0,
                max_clip_ratio=MAX_INPUT_CLIP_RATIO,
            )
            recorded_float = pcm_int32_to_float32(recorded_raw)
            path_results, tdoa_results, timestamp_results = analyze_paths(
                recorded_float,
                bounds=bounds,
                callback_rows=telemetry.get("callback_time_info", []),
                excitation=excitation,
                inverse=inverse,
                excitation_kind=str(args.excitation),
                excited_frequencies_hz=np.asarray(
                    excitation_meta.get("tone_frequencies_hz", []),
                    dtype=np.float64,
                ),
                repeats=int(args.repeats),
                gap_samples=gap_samples,
                channels=channels,
                sample_rate=sample_rate,
                band_hz=band_hz,
                fir_length=int(args.fir_length),
                pre_roll=int(args.pre_roll),
                max_delay_samples=max_delay_samples,
                max_delay_jitter_samples=max_delay_jitter_samples,
                max_tdoa_samples=max_tdoa_samples,
                max_tdoa_jitter_samples=max_tdoa_jitter_samples,
                max_timestamp_jitter_seconds=float(args.max_timestamp_jitter_ms) / 1000.0,
            )
            cross_epoch_timestamps = cross_epoch_timestamp_consistency(
                timestamp_results,
                sample_rate=sample_rate,
                max_difference_seconds=float(args.max_timestamp_jitter_ms) / 1000.0,
            )
            differential_results = {
                "ns_minus_cs_at_err": differential_delay_report(
                    path_results["ns_to_err"]["absolute_delay"],
                    path_results["cs_to_err"]["absolute_delay"],
                    sample_rate=sample_rate,
                    max_jitter_samples=max_delay_jitter_samples,
                    sign_convention="positive means NS→ERR arrives later than CS→ERR",
                ),
                "ns_minus_cs_at_ref": differential_delay_report(
                    path_results["ns_to_ref"]["absolute_delay"],
                    path_results["cs_to_ref"]["absolute_delay"],
                    sample_rate=sample_rate,
                    max_jitter_samples=max_delay_jitter_samples,
                    sign_convention="positive means NS→REF arrives later than CS→REF",
                ),
            }
            routing_topology = routing_topology_gate(
                positions_m=duct["positions_m"],
                speed_of_sound_mps=float(duct["duct"]["speed_of_sound_mps"]),
                sample_rate=sample_rate,
                tdoa_results=tdoa_results,
                path_results=path_results,
            )
            causal_budget = calculate_causal_budget(
                sample_rate=sample_rate,
                noise_tdoa=tdoa_results["noise_out"],
                ns_to_err_absolute=path_results["ns_to_err"]["absolute_delay"],
                cs_to_err_absolute=path_results["cs_to_err"]["absolute_delay"],
                cross_epoch_timestamps=cross_epoch_timestamps,
                processing_samples=processing_samples,
            )
            valid, invalid_reasons, output_summary = overall_quality_gate(
                preflight_report=preflight_report,
                measurement_report=measurement_report,
                output_float=output_float,
                output_pcm=output_pcm,
                telemetry=telemetry,
                final_mute=final_mute,
                path_results=path_results,
                tdoa_results=tdoa_results,
                timestamp_results=timestamp_results,
                cross_epoch_timestamps=cross_epoch_timestamps,
                differential_results=differential_results,
                routing_topology=routing_topology,
                causal_budget=causal_budget,
            )
            if not valid:
                print(
                    "[INVALID] 덕트 식별 완료로 승격하지 않습니다: "
                    + ", ".join(invalid_reasons),
                    file=sys.stderr,
                )
    except Exception as exc:
        measurement_exception = True
        error_message = f"{type(exc).__name__}: {exc}"
        invalid_reasons = list(dict.fromkeys([*invalid_reasons, "measurement_exception"]))
        print(f"[실패] {error_message}", file=sys.stderr)

    complete = bool(
        not measurement_exception
        and path_results
        and not invalid_reasons
        and causal_budget.get("valid", False)
    )
    for key, path in path_results.items():
        path["key"] = key
    callback_rows = telemetry.get("callback_time_info", [])
    telemetry_summary = {
        key: value
        for key, value in telemetry.items()
        if key != "callback_time_info"
    }
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "measurement_kind": "single_stream_time_frequency_duct_transfer_map",
        "created_at_utc": created_at,
        "configuration": {
            "hardware": str(hardware_path.relative_to(REPO_ROOT.resolve())),
            "duct": str(duct_path.relative_to(REPO_ROOT.resolve())),
            "sample_rate": sample_rate,
            "block_size": block_size,
            "latency": str(args.latency),
            "excitation": str(args.excitation),
            "excitation_metadata": excitation_meta,
            "band_hz": list(band_hz),
            "report_bands_hz": [list(row) for row in DEFAULT_REPORT_BANDS_HZ],
            "amplitude_peak": float(args.amplitude),
            "maximum_allowed_amplitude_peak": MAX_AMPLITUDE,
            "repeats": int(args.repeats),
            "excitation_seconds": float(args.excitation_seconds),
            "gap_seconds": float(args.gap_seconds),
            "fir_length": int(args.fir_length),
            "pre_roll": int(args.pre_roll),
            "channels": channels,
            "single_stream_time_division": True,
            "processing_handoff_samples": processing_samples,
        },
        "preflight": {
            "channels": calibration._probe_summary(preflight_report),
            "passed_both": len(preflight_report.get("channels", [])) >= 2
            and all(bool(item.get("valid")) for item in preflight_report.get("channels", [])[:2]),
        },
        "measurement_channels": calibration._probe_summary(measurement_report),
        "telemetry": telemetry_summary,
        "callback_timestamps": {
            "exact_arrays_in_npz": [
                "callback_frame_start",
                "callback_frames",
                "callback_input_buffer_adc_time",
                "callback_current_time",
                "callback_output_buffer_dac_time",
            ],
            "by_output_epoch": timestamp_results,
            "cross_epoch_common_offset_gate": cross_epoch_timestamps,
        },
        "program_bounds_frames": {key: list(value) for key, value in bounds.items()},
        "final_mute": final_mute,
        "output": output_summary,
        "paths": {key: _path_json_summary(path) for key, path in path_results.items()},
        "relative_tdoa": tdoa_results,
        "routing_topology": routing_topology,
        "differential_output_path_delay": differential_results,
        "causal_budget": causal_budget,
        "result": {
            "duct_identification_complete": complete,
            "measurement_claim_allowed": complete,
            "anc_performance_claim_allowed": False,
            "fxlms_applied": False,
            "invalid_reasons": invalid_reasons,
            "error": error_message,
        },
        "artifacts": {
            key: _artifact_display_path(path) for key, path in paths.items()
        },
    }
    try:
        markdown = render_markdown(metadata)
        arrays = build_npz_arrays(
            metadata=metadata,
            output_float=output_float,
            output_pcm=output_pcm,
            recorded_raw=recorded_raw,
            preflight_raw=preflight_raw,
            excitation=excitation,
            callback_rows=callback_rows,
            path_results=path_results,
            tdoa_results=tdoa_results,
        )
        save_artifacts(paths, arrays=arrays, metadata=metadata, markdown=markdown)
        if args.plot and path_results:
            save_plot(paths["plot"], path_results, band_hz)
    except Exception as exc:
        print(f"[실패] 결과 저장 실패: {exc}", file=sys.stderr)
        return 2

    print(f"NPZ: {paths['npz']}")
    print(f"JSON: {paths['json']}")
    print(f"Markdown: {paths['markdown']}")
    if complete:
        print("[PASS] 네 경로와 인과성 예산 게이트를 통과했습니다.")
        return 0
    print("[INVALID] 원시 결과는 저장했지만 덕트 식별 완료를 주장하지 않습니다.")
    return 2 if measurement_exception else 1


if __name__ == "__main__":
    raise SystemExit(main())
