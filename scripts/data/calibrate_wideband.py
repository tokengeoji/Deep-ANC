#!/usr/bin/env python3
"""ESS로 P(z) 또는 S(z)를 저음량 측정한다.

이 스크립트는 두 출력 경로에 같은 품질 게이트를 적용한다.

* ``--output-channel noise``: 소음 스피커 -> ERR 마이크, P(z)
* ``--output-channel cancel``: 상쇄 스피커 -> ERR 마이크, S(z)

정식 모델 NPZ는 양 마이크 사전 점검, 무 xrun, 입출력 무클리핑, 3회 이상 반복,
반복 일관성 0.9 이상을 모두 만족할 때만 생성된다. 측정 성공 여부와 관계없이
재현에 필요한 원시 출력/ERR/REF와 메타데이터는 ``results/``에 별도로 보존한다.

예시 (사용자 입회, 앰프 볼륨 최저에서만 실행)::

  .venv/bin/python scripts/data/calibrate_wideband.py \
    --confirm-volume-minimum --output-channel cancel \
    --out assets/measured/secondary_path_wb.npz

기존 파일은 덮어쓰지 않는다. 실측 모델의 극성은 그대로 보존하며 후단에서 추가
부호 반전을 하지 않는다(``e = d + S*y`` 규약).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.audio_io import (  # noqa: E402
    DEFAULT_PROBE_SETTLE_SECONDS,
    analyze_int32_input_probe,
    assert_measurement_preconditions,
    float32_to_pcm_int16,
    pcm_int32_to_float32,
    resolve_alsa_portaudio_device,
    rms_dbfs,
)
from deep_anc.config import REPO_ROOT, load_yaml  # noqa: E402


DEFAULT_BAND_HZ = (80.0, 1600.0)
DEFAULT_AMPLITUDE = 0.005
MAX_AMPLITUDE = 0.02
MIN_REPEATS = 3
MIN_CONSISTENCY = 0.9
MAX_INPUT_CLIP_RATIO = 0.005
DEFAULT_MAX_DELAY_JITTER_MS = 1.0
ONSET_ENERGY_WINDOW_SAMPLES = 64
ONSET_NOISE_MULTIPLIER = 4.0
ONSET_PEAK_ENERGY_FRACTION = 0.05


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument("--output-channel", choices=["cancel", "noise"], default="cancel")
    parser.add_argument("--band", type=float, nargs=2, default=list(DEFAULT_BAND_HZ))
    parser.add_argument("--sweep-seconds", type=float, default=4.0)
    parser.add_argument("--repeats", type=int, default=MIN_REPEATS)
    parser.add_argument("--amplitude", type=float, default=DEFAULT_AMPLITUDE)
    parser.add_argument("--fir-length", type=int, default=2048)
    parser.add_argument("--pre-roll", type=int, default=32)
    parser.add_argument("--max-delay-ms", type=float, default=250.0)
    parser.add_argument(
        "--max-delay-jitter-ms",
        type=float,
        default=DEFAULT_MAX_DELAY_JITTER_MS,
        help="반복별 robust 지연의 허용 max-min 지터(ms, 기본 1ms)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="PortAudio 블록 크기(생략 시 hardware YAML 값)",
    )
    parser.add_argument(
        "--latency",
        choices=["low", "high"],
        default="high",
        help="PortAudio latency 모드(기본 high: 측정 중 xrun 위험 최소화)",
    )
    parser.add_argument("--input-probe-seconds", type=float, default=2.0)
    parser.add_argument("--out", default=None, help="품질 게이트 통과 시 생성할 정식 모델 NPZ")
    parser.add_argument(
        "--diagnostics-root",
        default="results/calibration_wideband",
        help="원시 진단 세션을 만들 results/ 하위 디렉터리",
    )
    parser.add_argument(
        "--confirm-volume-minimum",
        action="store_true",
        help="사용자 입회 및 물리 앰프 볼륨 최저 상태를 확인",
    )
    return parser


def ess_pair(f1: float, f2: float, seconds: float, fs: int, amp: float):
    """지수 사인 스윕과 Farina 방식 진폭 보상 역필터를 만든다."""
    n = int(round(seconds * fs))
    if n < 4:
        raise ValueError("스윕 길이가 너무 짧습니다")
    t = np.arange(n, dtype=np.float64) / fs
    ratio_log = np.log(f2 / f1)
    phase = 2 * np.pi * f1 * seconds / ratio_log * (
        np.exp(t * ratio_log / seconds) - 1.0
    )
    sweep = np.sin(phase)
    fade = max(1, min(int(0.05 * fs), n // 2))
    env = np.ones(n, dtype=np.float64)
    ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
    env[:fade] = ramp
    env[-fade:] = ramp[::-1]
    sweep = (amp * sweep * env).astype(np.float32)
    inv = sweep[::-1].astype(np.float64) * np.exp(-t * ratio_log / seconds)
    return sweep, inv.astype(np.float32)


def _repo_path(value: str | Path, *, require_results: bool = False) -> Path:
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
        raise ValueError(f"원시 진단은 results/ 아래에만 저장해야 합니다: {path}")
    return path


def validate_options(args: argparse.Namespace, fs: int) -> tuple[int, Path, Path]:
    if not args.confirm_volume_minimum:
        raise ValueError(
            "사용자 입회와 물리 앰프 볼륨 최저를 확인한 뒤 "
            "--confirm-volume-minimum을 지정하세요"
        )
    if args.repeats < MIN_REPEATS:
        raise ValueError(f"--repeats는 {MIN_REPEATS} 이상이어야 합니다")
    if not (0.0 < args.amplitude <= MAX_AMPLITUDE):
        raise ValueError(f"--amplitude는 0 초과 {MAX_AMPLITUDE} 이하여야 합니다")
    f1, f2 = map(float, args.band)
    if not (0.0 < f1 < f2 < fs / 2.0):
        raise ValueError(f"--band는 0 < 하한 < 상한 < Nyquist({fs / 2:.0f}Hz)여야 합니다")
    if args.sweep_seconds <= 0.0:
        raise ValueError("--sweep-seconds는 양수여야 합니다")
    if args.input_probe_seconds <= 0.0:
        raise ValueError("--input-probe-seconds는 양수여야 합니다")
    if (
        args.fir_length < 1
        or args.pre_roll < 0
        or args.max_delay_ms <= 0.0
        or args.max_delay_jitter_ms < 0.0
    ):
        raise ValueError("FIR/지연 인자가 잘못되었습니다")

    block_size = int(args.block_size) if args.block_size is not None else 0
    if args.block_size is not None and block_size <= 0:
        raise ValueError("--block-size는 양수여야 합니다")

    default_name = (
        "primary_path_wb.npz" if args.output_channel == "noise" else "secondary_path_wb.npz"
    )
    out_path = _repo_path(args.out or f"assets/measured/{default_name}")
    if out_path.suffix.lower() != ".npz":
        raise ValueError("--out은 .npz 파일이어야 합니다")
    if out_path.exists():
        raise FileExistsError(f"기존 정식 모델은 덮어쓰지 않습니다: {out_path}")

    diagnostics_root = _repo_path(args.diagnostics_root, require_results=True)
    return block_size, out_path, diagnostics_root


def _new_diagnostics_dir(root: Path, output_channel: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = root / f"{stamp}_{output_channel}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _capture_preflight(sd: Any, audio: dict, seconds: float) -> tuple[np.ndarray, dict]:
    """출력 장치를 열기 전에 두 마이크의 원시 S32_LE를 캡처한다.

    앞 ``DEFAULT_PROBE_SETTLE_SECONDS`` 는 I2S 기동 트랜지언트라 버린다 — 포함해서 재면
    무신호 마이크도 -42dBFS 로 보여 생존 판정을 통과한다(audio_io.capture_input_probe 주석 참조).
    """
    fs = int(audio["sample_rate"])
    input_cfg = audio["input"]
    in_dev = resolve_alsa_portaudio_device(
        input_cfg["card"], input_cfg["pcm"], "input", 2
    )
    settle_frames = int(round(DEFAULT_PROBE_SETTLE_SECONDS * fs))
    raw = sd.rec(
        settle_frames + int(round(seconds * fs)),
        samplerate=fs,
        channels=2,
        dtype="int32",
        device=in_dev,
    )
    sd.wait()
    raw = np.asarray(raw, dtype=np.int32)[settle_frames:]
    report = analyze_int32_input_probe(
        raw,
        min_rms_dbfs=-80.0,
        max_clip_ratio=MAX_INPUT_CLIP_RATIO,
    )
    report.update(
        {
            "device": int(in_dev),
            "sample_rate": fs,
            "settle_seconds": DEFAULT_PROBE_SETTLE_SECONDS,
        }
    )
    return raw, report


def _probe_summary(report: dict) -> list[dict]:
    return [
        {
            "channel": int(item["channel"]),
            "rms_dbfs": float(item["rms_dbfs"]),
            "peak": float(item["peak"]),
            "clip_ratio": float(item["clip_ratio"]),
            "unique_codes": int(item["unique_codes"]),
            "raw_min": int(item["raw_min"]),
            "raw_max": int(item["raw_max"]),
            "stuck": bool(item["stuck"]),
            "valid": bool(item["valid"]),
        }
        for item in report.get("channels", [])[:2]
    ]


def _status_snapshot(status: Any) -> dict[str, Any]:
    xrun_names = (
        "input_underflow",
        "input_overflow",
        "output_underflow",
        "output_overflow",
    )
    flags = {name: bool(getattr(status, name, False)) for name in xrun_names}
    flags["priming_output"] = bool(getattr(status, "priming_output", False))
    flags["text"] = str(status)
    flags["is_xrun"] = any(flags[name] for name in xrun_names)
    flags["unexpected"] = bool(status) and not flags["is_xrun"] and not flags["priming_output"]
    return flags


def _capture_measurement(
    sd: Any,
    *,
    fs: int,
    block_size: int,
    latency: str,
    in_dev: int,
    out_dev: int,
    output_float: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """전이중 측정. 어떤 종료 경로에서도 스트림을 abort/close한다."""
    if output_float.ndim != 2 or output_float.shape[1] != 2:
        raise ValueError("output_float는 [frames, 2]여야 합니다")
    total = int(output_float.shape[0])
    output_pcm = float32_to_pcm_int16(output_float)
    recorded_raw = np.zeros((total, 2), dtype=np.int32)
    cursor = {"frames": 0}
    telemetry: dict[str, Any] = {
        "callback_count": 0,
        "callback_status_count": 0,
        "xrun_count": 0,
        "priming_output_count": 0,
        "unexpected_status_count": 0,
        "statuses": [],
        "callback_error": None,
        "completed": False,
    }

    def callback(indata, outdata, frames, _time_info, status):
        # 콜백 예외 시 마지막 버퍼가 반복되지 않도록 가장 먼저 무음으로 만든다.
        outdata.fill(0)
        try:
            telemetry["callback_count"] += 1
            if status:
                item = _status_snapshot(status)
                telemetry["callback_status_count"] += 1
                telemetry["xrun_count"] += int(item["is_xrun"])
                telemetry["priming_output_count"] += int(item["priming_output"])
                telemetry["unexpected_status_count"] += int(item["unexpected"])
                telemetry["statuses"].append(item)

            i = int(cursor["frames"])
            n = min(int(frames), total - i)
            if n > 0:
                recorded_raw[i : i + n] = np.asarray(indata[:n, :2], dtype=np.int32)
                outdata[:n] = output_pcm[i : i + n]
                cursor["frames"] = i + n
            if cursor["frames"] >= total:
                telemetry["completed"] = True
                raise sd.CallbackStop
        except sd.CallbackStop:
            raise
        except Exception as exc:  # PortAudio 경계를 넘어 예외가 유실되지 않게 기록
            outdata.fill(0)
            telemetry["callback_error"] = f"{type(exc).__name__}: {exc}"
            raise sd.CallbackAbort

    stream = None
    started = time.monotonic()
    # 예상 길이에 충분한 여유를 주되 장치 정지 시 스피커를 무한히 열어두지 않는다.
    deadline = started + total / fs + 15.0
    try:
        stream = sd.Stream(
            samplerate=fs,
            blocksize=block_size,
            device=(in_dev, out_dev),
            channels=(2, 2),
            dtype=("int32", "int16"),
            latency=(latency, latency),
            callback=callback,
            prime_output_buffers_using_stream_callback=True,
        )
        stream.start()
        while not telemetry["completed"]:
            if telemetry["callback_error"] is not None:
                raise RuntimeError(f"오디오 콜백 실패: {telemetry['callback_error']}")
            if time.monotonic() >= deadline:
                raise TimeoutError("오디오 콜백 완료 대기 시간이 초과되었습니다")
            time.sleep(0.02)
    finally:
        # 닫힌 출력 장치는 무음을 유지한다. 오류 중 abort/close 자체 실패는 원래 오류를
        # 가리지 않도록 삼키되, 두 동작을 모두 시도한다.
        if stream is not None:
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
    return recorded_raw, output_pcm, telemetry


def _pairwise_consistency(irs: np.ndarray) -> tuple[float, list[float]]:
    correlations: list[float] = []
    for left, right in combinations(range(int(irs.shape[0])), 2):
        a = np.asarray(irs[left], dtype=np.float64)
        b = np.asarray(irs[right], dtype=np.float64)
        if np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
            correlations.append(float("nan"))
        else:
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
    finite = [value for value in correlations if np.isfinite(value)]
    consistency = float(np.mean(finite)) if len(finite) == len(correlations) and finite else 0.0
    return consistency, correlations


def _robust_energy_onset(
    ir: np.ndarray,
    *,
    max_delay_samples: int,
    window_samples: int = ONSET_ENERGY_WINDOW_SAMPLES,
) -> dict[str, Any]:
    """단일 peak가 아닌 연속된 두 에너지 구간으로 IR onset을 찾는다.

    서로 겹치지 않는 두 window가 모두 임계값을 넘어야 하므로 잡음의 단일
    sample이 과거의 ``abs(ir) >= 5% peak``처럼 onset으로 채택되지 않는다.
    임계값은 search 구간의 낮은 에너지 25%로 구한 noise floor와 전체 peak
    energy 양쪽에 대해 보수적으로 정한다.
    """
    values = np.asarray(ir, dtype=np.float64).reshape(-1)
    result: dict[str, Any] = {
        "onset_samples": None,
        "noise_rms": None,
        "threshold_rms": None,
        "peak_rms": None,
        "window_samples": None,
    }
    if (
        values.size < 6
        or max_delay_samples <= 0
        or not np.all(np.isfinite(values))
    ):
        return result

    # onset 경계 전후의 서로 겹치지 않는 두 window가 search 범위에 들어가게 한다.
    window = min(
        int(window_samples),
        max(1, int(max_delay_samples) // 3),
        max(1, int(values.size) // 3),
    )
    if window < 2:
        return result
    search_stop = min(values.size, int(max_delay_samples) + window)
    search = values[:search_stop]
    running_rms = np.sqrt(
        np.convolve(search * search, np.ones(window) / window, mode="valid")
    )
    if running_rms.size <= window:
        return result

    block_starts = np.arange(
        0,
        min(int(max_delay_samples), int(running_rms.size)),
        window,
        dtype=np.int64,
    )
    block_rms = running_rms[block_starts]
    if block_rms.size == 0:
        return result
    quiet_count = max(1, int(np.ceil(0.25 * block_rms.size)))
    quiet = np.partition(block_rms, quiet_count - 1)[:quiet_count]
    noise_rms = float(np.median(quiet))
    peak_rms = float(
        np.max(running_rms[: min(int(max_delay_samples), running_rms.size)])
    )
    threshold_rms = max(
        ONSET_NOISE_MULTIPLIER * noise_rms,
        ONSET_PEAK_ENERGY_FRACTION * peak_rms,
        np.finfo(np.float64).eps,
    )

    # i:i+w와 i+w:i+2w가 모두 threshold 이상이어야 한다. 반환점은 두
    # window의 경계이며, 이 지점에서 pre_roll을 빼 compact FIR을 만든다.
    candidate_count = min(
        int(max_delay_samples) - window + 1,
        int(running_rms.size) - window,
    )
    if candidate_count > 0:
        sustained = (
            running_rms[:candidate_count] >= threshold_rms
        ) & (
            running_rms[window : window + candidate_count] >= threshold_rms
        )
        candidates = np.flatnonzero(sustained)
        if candidates.size:
            result["onset_samples"] = int(candidates[0] + window)

    result.update(
        {
            "noise_rms": noise_rms,
            "threshold_rms": threshold_rms,
            "peak_rms": peak_rms,
            "window_samples": int(window),
        }
    )
    return result


def _model_from_repeat_irs(
    irs: np.ndarray,
    *,
    max_delay_samples: int,
    fir_length: int,
    pre_roll: int,
    max_delay_jitter_samples: int,
) -> tuple[dict[str, Any], float, list[float], str | None]:
    """반복별 robust delay가 안정적일 때만 정렬 FIR 평균을 만든다."""
    values = np.asarray(irs, dtype=np.float64)
    raw_consistency, raw_correlations = _pairwise_consistency(values)
    onset_reports = [
        _robust_energy_onset(ir, max_delay_samples=max_delay_samples)
        for ir in values
    ]
    onsets = [report["onset_samples"] for report in onset_reports]
    model: dict[str, Any] = {
        "stable_delay": False,
        "repeat_onset_samples": onsets,
        "repeat_delay_samples": None,
        "delay_spread_samples": None,
        "max_delay_jitter_samples": int(max_delay_jitter_samples),
        "onset_reports": onset_reports,
        "unaligned_consistency": float(raw_consistency),
        "unaligned_pairwise_correlations": raw_correlations,
    }
    if any(value is None for value in onsets):
        return model, raw_consistency, raw_correlations, "반복 IR robust onset을 찾지 못했습니다"

    delays = [max(0, int(value) - int(pre_roll)) for value in onsets]
    spread = int(max(delays) - min(delays))
    model["repeat_delay_samples"] = delays
    model["delay_spread_samples"] = spread
    if spread > int(max_delay_jitter_samples):
        return (
            model,
            raw_consistency,
            raw_correlations,
            "반복 지연 지터가 허용값을 초과했습니다: "
            f"{spread} > {int(max_delay_jitter_samples)} samples",
        )

    # 각 반복을 자체 robust delay에서 잘라 정렬한 뒤 평균한다. 대표 pure
    # delay는 robust delay 중앙값으로 두어 한 반복의 outlier에 끌리지 않는다.
    aligned = []
    for ir, delay in zip(values, delays):
        compact = ir[delay : delay + int(fir_length)]
        if compact.size != int(fir_length) or not np.all(np.isfinite(compact)):
            return model, raw_consistency, raw_correlations, "정렬 FIR 추출 길이가 부족합니다"
        aligned.append(compact)
    aligned_irs = np.stack(aligned)
    consistency, correlations = _pairwise_consistency(aligned_irs)
    median_delay = int(np.floor(float(np.median(delays)) + 0.5))
    fir = np.mean(aligned_irs, axis=0).astype(np.float32)
    model.update(
        {
            "stable_delay": True,
            "fir": fir,
            "delay_samples": median_delay,
        }
    )
    return model, consistency, correlations, None


def extract_path_model(
    *,
    err: np.ndarray,
    sweep: np.ndarray,
    inv: np.ndarray,
    repeats: int,
    gap_samples: int,
    max_delay_samples: int,
    fir_length: int,
    pre_roll: int,
    max_delay_jitter_samples: int = 48,
) -> tuple[dict[str, Any] | None, np.ndarray, float, list[float], str | None]:
    """반복별 IR, robust 지연, 정렬 compact FIR을 계산한다."""
    one_shot_size = gap_samples + sweep.size + gap_samples
    ref = signal.fftconvolve(sweep.astype(np.float64), inv, mode="full")
    ref_peak = float(np.max(np.abs(ref))) if ref.size else 0.0
    if not np.isfinite(ref_peak) or ref_peak <= 1e-12:
        return None, np.empty((0, 0)), 0.0, [], "ESS 자기 디컨볼루션 피크가 없습니다"

    ir_list: list[np.ndarray] = []
    start = gap_samples + sweep.size - 1
    wanted = max_delay_samples + fir_length + 4096
    for repeat in range(repeats):
        segment = err[repeat * one_shot_size : (repeat + 1) * one_shot_size]
        ir_full = signal.fftconvolve(segment, inv, mode="full") / ref_peak
        ir_list.append(ir_full[start : start + wanted])
    if not ir_list or min(map(len, ir_list)) < fir_length:
        return None, np.empty((0, 0)), 0.0, [], "IR 추출 길이가 부족합니다"
    n_min = min(map(len, ir_list))
    irs = np.stack([value[:n_min] for value in ir_list])
    model, consistency, correlations, error = _model_from_repeat_irs(
        irs,
        max_delay_samples=max_delay_samples,
        fir_length=fir_length,
        pre_roll=pre_roll,
        max_delay_jitter_samples=max_delay_jitter_samples,
    )
    return model, irs, consistency, correlations, error


def quality_gate(
    *,
    preflight_report: dict,
    measurement_report: dict,
    output_float: np.ndarray,
    output_pcm: np.ndarray,
    telemetry: dict,
    consistency: float,
    model_available: bool,
    delay_spread_samples: int | None = 0,
    max_delay_jitter_samples: int = 48,
) -> tuple[bool, list[str], dict]:
    """P/S 공통 승격 기준을 적용하는 순수 함수."""
    reasons: list[str] = []
    preflight_channels = preflight_report.get("channels", [])[:2]
    if len(preflight_channels) != 2 or not all(bool(v.get("valid")) for v in preflight_channels):
        reasons.append("preflight_both_mics_invalid")

    measurement_channels = measurement_report.get("channels", [])[:2]
    if len(measurement_channels) != 2:
        reasons.append("measurement_both_mics_missing")
    elif any(float(v.get("clip_ratio", 1.0)) > MAX_INPUT_CLIP_RATIO for v in measurement_channels):
        reasons.append("input_clipping")

    output_peak = float(np.max(np.abs(output_float))) if output_float.size else 0.0
    output_clip_ratio = (
        float(np.mean(np.abs(output_float.astype(np.float64)) >= 1.0))
        if output_float.size
        else 0.0
    )
    pcm_saturation_ratio = (
        float(np.mean(np.abs(output_pcm.astype(np.int32)) >= np.iinfo(np.int16).max))
        if output_pcm.size
        else 0.0
    )
    if output_peak > 1.0 or output_clip_ratio > 0.0 or pcm_saturation_ratio > 0.0:
        reasons.append("output_clipping")
    if int(telemetry.get("xrun_count", 0)) > 0:
        reasons.append("xrun_detected")
    if int(telemetry.get("unexpected_status_count", 0)) > 0:
        reasons.append("unexpected_callback_status")
    if telemetry.get("callback_error"):
        reasons.append("callback_error")
    if not bool(telemetry.get("completed", False)):
        reasons.append("measurement_incomplete")
    if not np.isfinite(consistency) or consistency < MIN_CONSISTENCY:
        reasons.append("repeat_consistency_below_0.9")
    if delay_spread_samples is None:
        reasons.append("repeat_delay_unavailable")
    elif int(delay_spread_samples) > int(max_delay_jitter_samples):
        reasons.append("repeat_delay_spread_exceeds_limit")
    if not model_available:
        reasons.append("path_model_unavailable")

    output_summary = {
        "peak": output_peak,
        "clip_ratio": output_clip_ratio,
        "pcm_saturation_ratio": pcm_saturation_ratio,
    }
    return not reasons, reasons, output_summary


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


def save_diagnostics(
    session_dir: Path,
    *,
    output: np.ndarray,
    output_pcm: np.ndarray,
    recorded_raw: np.ndarray,
    preflight_raw: np.ndarray,
    repeat_irs: np.ndarray,
    metadata: dict,
) -> tuple[Path, Path]:
    """원시 진단 NPZ/JSON을 새 파일로만 저장한다."""
    session_dir.mkdir(parents=True, exist_ok=True)
    npz_path = session_dir / "raw_measurement.npz"
    json_path = session_dir / "metadata.json"
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(f"진단 파일은 덮어쓰지 않습니다: {session_dir}")

    normalized = pcm_int32_to_float32(recorded_raw) if recorded_raw.size else np.empty((0, 2), np.float32)
    err = normalized[:, 0] if normalized.ndim == 2 and normalized.shape[1] >= 2 else np.empty(0, np.float32)
    ref = normalized[:, 1] if normalized.ndim == 2 and normalized.shape[1] >= 2 else np.empty(0, np.float32)
    metadata_json = json.dumps(_json_safe(metadata), ensure_ascii=False, sort_keys=True)
    with npz_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            output=np.asarray(output, dtype=np.float32),
            output_pcm_int16=np.asarray(output_pcm, dtype=np.int16),
            err=np.asarray(err, dtype=np.float32),
            ref=np.asarray(ref, dtype=np.float32),
            input_raw_int32=np.asarray(recorded_raw, dtype=np.int32),
            preflight_raw_int32=np.asarray(preflight_raw, dtype=np.int32),
            repeat_irs=np.asarray(repeat_irs, dtype=np.float64),
            metadata_json=np.asarray(metadata_json),
        )
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(_json_safe(metadata), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return npz_path, json_path


def save_official_model(path: Path, *, valid: bool, arrays: dict[str, Any]) -> bool:
    """품질 게이트 통과 모델만 배타적으로 저장한다."""
    if not valid:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez(handle, **arrays)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # 확인 플래그는 sounddevice import 및 장치 접근보다 먼저 검사한다.
    if not args.confirm_volume_minimum:
        print(
            "[중단] 사용자 입회와 물리 앰프 볼륨 최저를 확인한 뒤 "
            "--confirm-volume-minimum을 지정하세요.",
            file=sys.stderr,
        )
        return 2

    try:
        hardware = load_yaml(REPO_ROOT / args.hardware)["audio"]
        fs = int(hardware["sample_rate"])
        requested_block, out_path, diagnostics_root = validate_options(args, fs)
        block_size = requested_block or int(hardware["block_size"])
        max_delay_jitter_samples = int(
            round(float(args.max_delay_jitter_ms) / 1000.0 * fs)
        )
        if block_size <= 0:
            raise ValueError("block size는 양수여야 합니다")
    except (KeyError, OSError, RuntimeError, ValueError, FileExistsError) as exc:
        print(f"[중단] {exc}", file=sys.stderr)
        return 2

    # 이 지점부터 측정 시도이므로 성공/실패와 관계없이 진단 세션을 남긴다.
    try:
        diagnostics_dir = _new_diagnostics_dir(diagnostics_root, args.output_channel)
    except (OSError, ValueError) as exc:
        print(f"[중단] 진단 디렉터리를 만들 수 없습니다: {exc}", file=sys.stderr)
        return 2

    preflight_raw = np.empty((0, 2), dtype=np.int32)
    recorded_raw = np.empty((0, 2), dtype=np.int32)
    output_float = np.empty((0, 2), dtype=np.float32)
    output_pcm = np.empty((0, 2), dtype=np.int16)
    repeat_irs = np.empty((0, 0), dtype=np.float64)
    preflight_report: dict[str, Any] = {"channels": []}
    measurement_report: dict[str, Any] = {"channels": []}
    telemetry: dict[str, Any] = {
        "xrun_count": 0,
        "unexpected_status_count": 0,
        "completed": False,
    }
    model: dict[str, Any] | None = None
    consistency = 0.0
    correlations: list[float] = []
    invalid_reasons: list[str] = []
    output_summary: dict[str, Any] = {}
    official_saved = False
    error_message: str | None = None
    exit_code = 2
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()

    try:
        import sounddevice as sd

        # 출력 장치를 열기 전 양 마이크 raw probe.
        # 캡처 클록 교란 방지 — PulseAudio 가 같은 APE 카드를 44.1kHz 로 잡으면
        # PLL_A 가 재조정되어 우리 48kHz 캡처의 BCLK 가 세션 중에 이동한다.
        # XRUN 이 아니라서 기존 게이트가 못 잡는다(2026-08-06 I2S 설계 검증에서 발견).
        assert_measurement_preconditions(sd, hardware)
        print("출력 없는 ERR/REF raw preflight 중...")
        preflight_raw, preflight_report = _capture_preflight(
            sd, hardware, args.input_probe_seconds
        )
        for name, item in zip(("ERR", "REF"), _probe_summary(preflight_report)):
            verdict = "PASS" if item["valid"] else "FAIL"
            print(
                f"[{verdict}] {name}: RMS {item['rms_dbfs']:.2f}dBFS, "
                f"peak {item['peak']:.6f}, clip {item['clip_ratio']:.3%}, "
                f"unique {item['unique_codes']}"
            )
        if len(preflight_report.get("channels", [])) < 2 or not all(
            bool(item.get("valid")) for item in preflight_report["channels"][:2]
        ):
            invalid_reasons = ["preflight_both_mics_invalid"]
            error_message = "양 마이크 raw preflight 실패 — 출력 장치를 열지 않았습니다"
            exit_code = 1
        else:
            in_dev = int(preflight_report["device"])
            output_cfg = hardware["output"]
            out_dev = resolve_alsa_portaudio_device(
                output_cfg["card"], output_cfg["pcm"], "output", 2
            )
            out_channel = 1 if args.output_channel == "cancel" else 0
            sweep, inv = ess_pair(
                float(args.band[0]),
                float(args.band[1]),
                float(args.sweep_seconds),
                fs,
                float(args.amplitude),
            )
            gap = np.zeros(fs, dtype=np.float32)
            one_shot = np.concatenate([gap, sweep, gap])
            playback = np.tile(one_shot, int(args.repeats))
            output_float = np.zeros((playback.size, 2), dtype=np.float32)
            output_float[:, out_channel] = playback

            print(
                f"저음량 ESS: {args.band[0]:.0f}–{args.band[1]:.0f}Hz "
                f"x{args.repeats}, {args.output_channel}/ch{out_channel}, "
                f"peak {args.amplitude:.4f}, block {block_size}, latency {args.latency}"
            )
            recorded_raw, output_pcm, telemetry = _capture_measurement(
                sd,
                fs=fs,
                block_size=block_size,
                latency=str(args.latency),
                in_dev=in_dev,
                out_dev=out_dev,
                output_float=output_float,
            )
            normalized = pcm_int32_to_float32(recorded_raw)
            measurement_report = analyze_int32_input_probe(
                recorded_raw,
                min_rms_dbfs=-120.0,
                max_clip_ratio=MAX_INPUT_CLIP_RATIO,
            )
            err = normalized[:, 0].astype(np.float64)
            print(
                f"녹음 완료: ERR {rms_dbfs(err):.2f}dBFS, "
                f"xrun {telemetry.get('xrun_count', 0)}"
            )
            model, repeat_irs, consistency, correlations, extraction_error = extract_path_model(
                err=err,
                sweep=sweep,
                inv=inv,
                repeats=int(args.repeats),
                gap_samples=gap.size,
                max_delay_samples=int(float(args.max_delay_ms) / 1000.0 * fs),
                fir_length=int(args.fir_length),
                pre_roll=int(args.pre_roll),
                max_delay_jitter_samples=max_delay_jitter_samples,
            )
            if extraction_error:
                error_message = extraction_error
            model_available = bool(
                model is not None
                and model.get("stable_delay", False)
                and "fir" in model
                and "delay_samples" in model
            )
            delay_spread_samples = (
                model.get("delay_spread_samples") if model is not None else None
            )
            valid, invalid_reasons, output_summary = quality_gate(
                preflight_report=preflight_report,
                measurement_report=measurement_report,
                output_float=output_float,
                output_pcm=output_pcm,
                telemetry=telemetry,
                consistency=consistency,
                model_available=model_available,
                delay_spread_samples=delay_spread_samples,
                max_delay_jitter_samples=max_delay_jitter_samples,
            )
            if extraction_error and "path_model_unavailable" not in invalid_reasons:
                invalid_reasons.append("path_model_unavailable")
                valid = False

            if model is not None and model.get("repeat_delay_samples") is not None:
                print(
                    "반복 robust delay: "
                    f"{model['repeat_delay_samples']} samples, "
                    f"spread {model['delay_spread_samples']}/"
                    f"{max_delay_jitter_samples} samples"
                )
            if model_available and model is not None:
                print(
                    f"추정: delay {model['delay_samples']} samples "
                    f"({1000 * model['delay_samples'] / fs:.2f}ms), "
                    f"FIR {len(model['fir'])} taps, pairwise consistency {consistency:.3f}"
                )
            if valid and model_available and model is not None:
                arrays = {
                    "fir": model["fir"],
                    "delay_samples": int(model["delay_samples"]),
                    "sample_rate": fs,
                    "dc_block_r": 0.995,
                    "fit_improvement_db": float("nan"),
                    "coherence_median": consistency,
                    "excitation_band_hz": np.asarray(args.band, dtype=np.float64),
                    "calibration_block_size": block_size,
                    "calibration_latency": str(args.latency),
                    "output_channel": args.output_channel,
                    "method": "ess",
                    "repeats": int(args.repeats),
                    "amplitude": float(args.amplitude),
                    "xrun_count": int(telemetry.get("xrun_count", 0)),
                    "repeat_onset_samples": np.asarray(
                        model["repeat_onset_samples"], dtype=np.int64
                    ),
                    "repeat_delay_samples": np.asarray(
                        model["repeat_delay_samples"], dtype=np.int64
                    ),
                    "delay_spread_samples": int(model["delay_spread_samples"]),
                    "max_delay_jitter_samples": max_delay_jitter_samples,
                    "onset_method": "two_window_robust_energy",
                }
                official_saved = save_official_model(out_path, valid=True, arrays=arrays)
                exit_code = 0
                print(f"[PASS] 정식 {args.output_channel} 경로 모델 저장: {out_path}")
            else:
                exit_code = 1
                print(
                    "[INVALID] 정식 모델을 저장하지 않습니다: " + ", ".join(invalid_reasons),
                    file=sys.stderr,
                )
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        invalid_reasons = list(dict.fromkeys([*invalid_reasons, "measurement_exception"]))
        print(f"[실패] {error_message}", file=sys.stderr)
        exit_code = 2
    finally:
        metadata = {
            "schema_version": 1,
            "measurement_kind": "wideband_path_raw_diagnostic",
            "started_at_utc": started_at,
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "configuration": {
                "output_channel": args.output_channel,
                "output_channel_index": 1 if args.output_channel == "cancel" else 0,
                "band_hz": [float(args.band[0]), float(args.band[1])],
                "sweep_seconds": float(args.sweep_seconds),
                "repeats": int(args.repeats),
                "amplitude": float(args.amplitude),
                "sample_rate": fs,
                "block_size": block_size,
                "latency": str(args.latency),
                "fir_length": int(args.fir_length),
                "pre_roll": int(args.pre_roll),
                "max_delay_ms": float(args.max_delay_ms),
                "max_delay_jitter_ms": float(args.max_delay_jitter_ms),
                "max_delay_jitter_samples": max_delay_jitter_samples,
            },
            "preflight": {
                "channels": _probe_summary(preflight_report),
                "passed_both": len(preflight_report.get("channels", [])) >= 2
                and all(bool(item.get("valid")) for item in preflight_report["channels"][:2]),
            },
            "measurement_channels": _probe_summary(measurement_report),
            "telemetry": telemetry,
            "output": output_summary,
            "result": {
                "consistency": float(consistency),
                "pairwise_correlations": correlations,
                "unaligned_consistency": model.get("unaligned_consistency")
                if model is not None
                else None,
                "unaligned_pairwise_correlations": model.get(
                    "unaligned_pairwise_correlations"
                )
                if model is not None
                else None,
                "repeat_onset_samples": model.get("repeat_onset_samples")
                if model is not None
                else None,
                "repeat_delay_samples": model.get("repeat_delay_samples")
                if model is not None
                else None,
                "delay_spread_samples": model.get("delay_spread_samples")
                if model is not None
                else None,
                "max_delay_jitter_samples": max_delay_jitter_samples,
                "stable_delay": bool(model.get("stable_delay", False))
                if model is not None
                else False,
                "onset_reports": model.get("onset_reports")
                if model is not None
                else None,
                "delay_samples": int(model["delay_samples"])
                if model is not None and "delay_samples" in model
                else None,
                "fir_length": int(len(model["fir"]))
                if model is not None and "fir" in model
                else None,
                "valid_for_model": bool(exit_code == 0 and official_saved),
                "invalid_reasons": invalid_reasons,
                "official_model_path": str(out_path.relative_to(REPO_ROOT))
                if official_saved
                else None,
            },
            "error": error_message,
        }
        try:
            npz_path, json_path = save_diagnostics(
                diagnostics_dir,
                output=output_float,
                output_pcm=output_pcm,
                recorded_raw=recorded_raw,
                preflight_raw=preflight_raw,
                repeat_irs=repeat_irs,
                metadata=metadata,
            )
            print(f"원시 진단 저장: {npz_path}")
            print(f"메타데이터 저장: {json_path}")
        except Exception as exc:
            print(f"[실패] 원시 진단 저장 실패: {exc}", file=sys.stderr)
            exit_code = 2

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
