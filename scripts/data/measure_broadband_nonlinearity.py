#!/usr/bin/env python3
"""광대역 THD/IMD 측정의 signal-only/무음 dry-run 계획을 만든다.

이 파일은 의도적으로 오디오 백엔드를 import하거나 장치를 열지 않는다. 실제 출력/캡처
경로는 raw publisher와 분석 gate가 완성된 뒤 별도 변경으로만 열며, 현재는 ``--dry-run``
없이 실행하면 실패한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.config import load_yaml  # noqa: E402


BROADBAND_NONLINEARITY_PLAN_SCHEMA = "broadband_nonlinearity_signal_plan_v1"
SAMPLE_RATE = 48_000
BLOCK_SIZE = 256
LATENCY = "low"
DEFAULT_PEAK = 0.003
RELATIVE_LEVELS_DB = (-24.0, -16.0, -8.0, 0.0)
THD_BAND_HZ = (100.0, 11_300.0)
THD_SLOT_SECONDS = 2.25
THD_ACTIVE_LIMIT_SECONDS = 1.5
IMD_PAIRS_HZ = ((1_800.0, 2_200.0), (3_600.0, 4_400.0), (7_200.0, 8_800.0))
IMD_ACTIVE_SECONDS = 0.75
IMD_GUARD_SECONDS = 0.50
IMD_FADE_SECONDS = 0.01
HARD_MAX_SECONDS = 50.0
THD_GATE_DBC = -30.0
COMPRESSION_GATE_DB = 1.0

DRIVE_PATHS = (
    {
        "plant": "P",
        "speaker": "NS",
        "output_role": "noise_out",
        "output_channel": 0,
    },
    {
        "plant": "S",
        "speaker": "CS",
        "output_role": "cancel_out",
        "output_channel": 1,
    },
)


def _float_to_pcm16(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype(np.int16)


def _peak_normalise(signal: np.ndarray, peak: float) -> np.ndarray:
    signal64 = np.asarray(signal, dtype=np.float64)
    observed = float(np.max(np.abs(signal64), initial=0.0))
    if not math.isfinite(observed) or observed <= 0.0:
        raise ValueError("비선형 probe가 유한한 non-zero 신호가 아닙니다")
    # 0.003은 float32에서 미세하게 위로 반올림된다. 승인 ceiling을 수치상으로도
    # 넘지 않도록 representable lower neighbour를 사용하고 마지막에 다시 clip한다.
    requested_peak = float(peak)
    safe_peak = np.float32(requested_peak)
    if float(safe_peak) > requested_peak:
        safe_peak = np.nextafter(safe_peak, np.float32(0.0))
    result = (signal64 * (float(safe_peak) / observed)).astype(np.float32)
    return np.clip(result, -safe_peak, safe_peak).astype(np.float32, copy=False)


def _build_synchronised_ess(*, sample_rate: int, peak: float) -> tuple[np.ndarray, dict[str, Any]]:
    """100→11.3kHz exponential sweep를 Novak식 정수 L·f1로 만든다."""

    f_start, f_stop = THD_BAND_HZ
    log_ratio = math.log(f_stop / f_start)
    synchronisation_order = int(
        math.floor(THD_ACTIVE_LIMIT_SECONDS * f_start / log_ratio)
    )
    if synchronisation_order < 1:
        raise ValueError("THD ESS synchronisation order가 유효하지 않습니다")
    sweep_constant_seconds = synchronisation_order / f_start
    continuous_duration_seconds = sweep_constant_seconds * log_ratio
    active_frames = int(round(continuous_duration_seconds * sample_rate)) + 1
    time_seconds = np.arange(active_frames, dtype=np.float64) / float(sample_rate)
    phase = (
        2.0
        * math.pi
        * f_start
        * sweep_constant_seconds
        * (np.exp(time_seconds / sweep_constant_seconds) - 1.0)
    )
    sweep = _peak_normalise(np.sin(phase), peak)
    actual_stop_hz = f_start * math.exp(time_seconds[-1] / sweep_constant_seconds)
    return sweep, {
        "construction": "synchronised_exponential_swept_sine",
        "requested_band_hz": [f_start, f_stop],
        "actual_start_hz": f_start,
        "actual_stop_hz": float(actual_stop_hz),
        "synchronisation_order": synchronisation_order,
        "sweep_constant_seconds": float(sweep_constant_seconds),
        "continuous_duration_seconds": float(continuous_duration_seconds),
        "active_frames": active_frames,
        "sample_grid_duration_seconds": float((active_frames - 1) / sample_rate),
        "slot_seconds": THD_SLOT_SECONDS,
    }


def _build_imd_pair(
    *, sample_rate: int, pair_hz: tuple[float, float], peak: float
) -> np.ndarray:
    active_frames = int(round(IMD_ACTIVE_SECONDS * sample_rate))
    time_seconds = np.arange(active_frames, dtype=np.float64) / float(sample_rate)
    signal = np.sin(2.0 * math.pi * pair_hz[0] * time_seconds)
    signal += np.sin(2.0 * math.pi * pair_hz[1] * time_seconds)
    fade_frames = int(round(IMD_FADE_SECONDS * sample_rate))
    if fade_frames * 2 >= active_frames:
        raise ValueError("IMD fade가 active window보다 깁니다")
    # 시작/종료 click이 비선형 산물로 잘못 집계되지 않도록 동일한 cosine ramp를 고정한다.
    phase = np.arange(fade_frames, dtype=np.float64) / float(fade_frames)
    ramp = 0.5 - 0.5 * np.cos(math.pi * phase)
    signal[:fade_frames] *= ramp
    signal[-fade_frames:] *= ramp[::-1]
    return _peak_normalise(signal, peak)


def _validate_hardware(hardware_path: str | Path) -> tuple[Path, dict[str, Any]]:
    hardware_file = Path(hardware_path).expanduser().resolve()
    hardware = load_yaml(hardware_file)
    audio = dict(hardware.get("audio") or {})
    channels = dict(hardware.get("channels") or {})
    actual = (
        int(audio.get("sample_rate", 0)),
        int(audio.get("block_size", 0)),
        str(audio.get("latency", "")),
    )
    if actual != (SAMPLE_RATE, BLOCK_SIZE, LATENCY):
        raise ValueError("광대역 THD/IMD dry-run은 hardware 48kHz/256/low가 필요합니다")
    expected_channels = {
        "error_mic": 0,
        "reference_mic": 1,
        "noise_out": 0,
        "cancel_out": 1,
    }
    if channels != expected_channels:
        raise ValueError(f"광대역 THD/IMD channel map이 다릅니다: {channels!r}")
    return hardware_file, {
        "sample_rate": SAMPLE_RATE,
        "block_size": BLOCK_SIZE,
        "latency": LATENCY,
        "channels": channels,
    }


def build_signal_plan(
    *,
    hardware_path: str | Path,
    peak: float = DEFAULT_PEAK,
    hard_max_seconds: float = HARD_MAX_SECONDS,
) -> tuple[dict[str, Any], np.ndarray]:
    """P/NS와 S/CS를 순차 구동하는 48초 THD/IMD PCM 계획을 만든다."""

    hardware_file, hardware = _validate_hardware(hardware_path)
    peak_value = float(peak)
    if not math.isfinite(peak_value) or not (0.0 < peak_value <= DEFAULT_PEAK):
        raise ValueError("광대역 THD/IMD 합산 peak는 0보다 크고 0.003 이하여야 합니다")

    thd_slot_frames = int(round(THD_SLOT_SECONDS * SAMPLE_RATE))
    imd_active_frames = int(round(IMD_ACTIVE_SECONDS * SAMPLE_RATE))
    imd_guard_frames = int(round(IMD_GUARD_SECONDS * SAMPLE_RATE))
    imd_slot_frames = imd_active_frames + imd_guard_frames
    ess_full, ess_metadata = _build_synchronised_ess(
        sample_rate=SAMPLE_RATE, peak=peak_value
    )
    if ess_full.shape[0] > thd_slot_frames:
        raise ValueError("THD ESS가 고정 slot보다 깁니다")

    segments: list[np.ndarray] = []
    layout: list[dict[str, Any]] = []
    cursor = 0

    def append_slot(
        signal: np.ndarray,
        *,
        slot_frames: int,
        drive: dict[str, Any],
        measurement: str,
        level_db: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        nonlocal cursor
        if signal.shape[0] > slot_frames:
            raise ValueError(f"{measurement} active signal이 slot보다 깁니다")
        stereo = np.zeros((slot_frames, 2), dtype=np.float32)
        stereo[: signal.shape[0], int(drive["output_channel"])] = signal
        start = cursor
        cursor += slot_frames
        row = {
            "kind": "nonlinearity_measurement_slot",
            "measurement": measurement,
            "plant": drive["plant"],
            "speaker": drive["speaker"],
            "output_role": drive["output_role"],
            "output_channel": drive["output_channel"],
            "relative_level_db": float(level_db),
            "start_frame": int(start),
            "active_stop_frame": int(start + signal.shape[0]),
            "stop_frame": int(cursor),
            "active_frames": int(signal.shape[0]),
            "guard_frames": int(slot_frames - signal.shape[0]),
            "slot_frames": int(slot_frames),
            **(extra or {}),
        }
        layout.append(row)
        segments.append(stereo)

    # 한 slot에는 한 speaker만 구동한다. P와 S를 별개 transfer/nonlinearity로 분석한다.
    for drive in DRIVE_PATHS:
        for level_db in RELATIVE_LEVELS_DB:
            level_peak = peak_value * (10.0 ** (level_db / 20.0))
            append_slot(
                _peak_normalise(ess_full, level_peak),
                slot_frames=thd_slot_frames,
                drive=drive,
                measurement="THD_ESS",
                level_db=level_db,
                extra={"requested_band_hz": list(THD_BAND_HZ)},
            )

    imd_signals = {
        pair: _build_imd_pair(sample_rate=SAMPLE_RATE, pair_hz=pair, peak=peak_value)
        for pair in IMD_PAIRS_HZ
    }
    for drive in DRIVE_PATHS:
        for pair_hz in IMD_PAIRS_HZ:
            for level_db in RELATIVE_LEVELS_DB:
                level_peak = peak_value * (10.0 ** (level_db / 20.0))
                append_slot(
                    _peak_normalise(imd_signals[pair_hz], level_peak),
                    slot_frames=imd_slot_frames,
                    drive=drive,
                    measurement="IMD_TWO_TONE",
                    level_db=level_db,
                    extra={"tone_pair_hz": list(pair_hz)},
                )

    output_float = np.concatenate(segments, axis=0)
    padding_frames = (-output_float.shape[0]) % BLOCK_SIZE
    if padding_frames:
        output_float = np.concatenate(
            (output_float, np.zeros((padding_frames, 2), dtype=np.float32)), axis=0
        )
        layout.append(
            {
                "kind": "block_padding_silence",
                "start_frame": int(cursor),
                "active_stop_frame": int(cursor),
                "stop_frame": int(cursor + padding_frames),
                "active_frames": 0,
                "guard_frames": int(padding_frames),
                "slot_frames": int(padding_frames),
            }
        )
        cursor += padding_frames
    if cursor != output_float.shape[0]:
        raise ValueError("THD/IMD layout와 output frame 수가 다릅니다")
    if not np.all(np.isfinite(output_float)):
        raise ValueError("THD/IMD signal plan에 NaN/Inf가 있습니다")
    observed_peak = float(np.max(np.abs(output_float), initial=0.0))
    if observed_peak > peak_value + 1.0e-7 or observed_peak > DEFAULT_PEAK + 1.0e-7:
        raise ValueError("THD/IMD signal plan peak가 0.003을 넘습니다")
    duration_seconds = output_float.shape[0] / SAMPLE_RATE
    if duration_seconds >= float(hard_max_seconds):
        raise ValueError(
            f"THD/IMD plan {duration_seconds:.3f}s가 hard max {float(hard_max_seconds):.3f}s 미만이 아닙니다"
        )

    output_pcm = _float_to_pcm16(output_float)
    plan: dict[str, Any] = {
        "schema": BROADBAND_NONLINEARITY_PLAN_SCHEMA,
        "role": "signal_only_dry_run_no_audio",
        "live_capture_enabled": False,
        "hardware": {"path": str(hardware_file), **hardware},
        "safety": {
            "anc_state": "OFF",
            "maximum_sum_peak": peak_value,
            "approved_peak_ceiling": DEFAULT_PEAK,
            "one_speaker_active_per_slot": True,
            "audio_device_opened": False,
            "hard_max_seconds": float(hard_max_seconds),
        },
        "decision_gates": {
            "thd_or_imd_nonlinearity_threshold_dbc": THD_GATE_DBC,
            "compression_threshold_db": COMPRESSION_GATE_DB,
            "interpretation": (
                "THD 또는 IMD가 -30 dBc보다 크거나 compression이 1 dB 이상이면 "
                "small-signal linear P/S만으로 학습하지 않는다"
            ),
        },
        "protocol": {
            "relative_levels_db": list(RELATIVE_LEVELS_DB),
            "drive_paths": list(DRIVE_PATHS),
            "window_seconds": {
                "thd_all_paths": 2 * len(RELATIVE_LEVELS_DB) * THD_SLOT_SECONDS,
                "imd_all_paths": (
                    2
                    * len(IMD_PAIRS_HZ)
                    * len(RELATIVE_LEVELS_DB)
                    * (IMD_ACTIVE_SECONDS + IMD_GUARD_SECONDS)
                ),
                "all_measurements": 48.0,
                "per_plant": 24.0,
                "active_nonzero_all_paths": float(
                    2
                    * len(RELATIVE_LEVELS_DB)
                    * ess_full.shape[0]
                    / SAMPLE_RATE
                    + 2
                    * len(IMD_PAIRS_HZ)
                    * len(RELATIVE_LEVELS_DB)
                    * IMD_ACTIVE_SECONDS
                ),
            },
            "thd_ess": ess_metadata,
            "imd_two_tone": {
                "pairs_hz": [list(pair) for pair in IMD_PAIRS_HZ],
                "active_seconds": IMD_ACTIVE_SECONDS,
                "guard_seconds": IMD_GUARD_SECONDS,
                "fade_seconds": IMD_FADE_SECONDS,
                "equal_tone_amplitudes": True,
            },
        },
        "layout": layout,
        "output": {
            "frames": int(output_pcm.shape[0]),
            "channels": 2,
            "dtype": str(output_pcm.dtype),
            "duration_seconds": float(duration_seconds),
            "padding_frames": int(padding_frames),
            "peak_float": observed_peak,
            "peak_pcm": int(np.max(np.abs(output_pcm.astype(np.int32)), initial=0)),
            "pcm_sha256": hashlib.sha256(output_pcm.tobytes(order="C")).hexdigest(),
            "channel_pcm_sha256": {
                "P_noise_out_ch0": hashlib.sha256(
                    output_pcm[:, 0].tobytes(order="C")
                ).hexdigest(),
                "S_cancel_out_ch1": hashlib.sha256(
                    output_pcm[:, 1].tobytes(order="C")
                ).hexdigest(),
            },
        },
    }
    return plan, output_pcm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--peak", type=float, default=DEFAULT_PEAK)
    parser.add_argument("--hard-max-seconds", type=float, default=HARD_MAX_SECONDS)
    parser.add_argument(
        "--output",
        type=Path,
        help="signal-only THD/IMD 계획 JSON을 no-replace로 저장합니다. PCM/오디오는 저장하지 않습니다",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run:
        print(
            "[중단] THD/IMD live 출력은 잠겨 있습니다. raw publisher와 분석 gate가 "
            "완성되기 전에는 소리를 내지 않습니다. 현재는 --dry-run만 허용합니다.",
            file=sys.stderr,
        )
        return 2
    try:
        plan, _ = build_signal_plan(
            hardware_path=args.hardware,
            peak=args.peak,
            hard_max_seconds=args.hard_max_seconds,
        )
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("x", encoding="utf-8") as handle:
                handle.write(rendered)
        except FileExistsError:
            print(f"[중단] 기존 계획은 덮어쓰지 않습니다: {output}", file=sys.stderr)
            return 2
        print(f"[saved] {output}", file=sys.stderr)
    print(rendered, end="")
    print(
        "[PASS] signal-only THD/IMD 계획. 오디오 장치와 스피커를 열지 않았습니다.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
