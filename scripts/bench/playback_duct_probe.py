#!/usr/bin/env python3
"""측정 마이크 없이 소음 스피커 ch0로 저레벨 단계 톤을 재생하는 정성 진단.

이 도구는 출력 채널·누설·주관적 공진 확인용이다. 마이크 데이터가 없으므로
P(z)/S(z), 감쇠 dB, 덕트 지연을 산출하거나 configs/duct.yaml을 변경하지 않는다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.audio_io import (  # noqa: E402
    assert_measurement_preconditions,
    float32_to_pcm_int16,
    resolve_alsa_portaudio_device,
)
from deep_anc.dsp.measurement_level import assert_live_pcm_clock_preconditions  # noqa: E402
from deep_anc.config import REPO_ROOT, load_runtime_config  # noqa: E402


def stepped_tone(
    frequency: float,
    sample_rate: int,
    seconds: float,
    amplitude: float,
    fade_seconds: float = 0.05,
) -> np.ndarray:
    """양끝이 0으로 페이드되는 단일 톤 float32 블록."""
    frames = int(round(sample_rate * seconds))
    if frames < 2:
        raise ValueError("톤 길이가 너무 짧습니다")
    t = np.arange(frames, dtype=np.float64) / sample_rate
    signal = amplitude * np.sin(2.0 * np.pi * frequency * t)
    fade = min(int(round(sample_rate * fade_seconds)), frames // 2)
    if fade > 0:
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, fade, endpoint=True)) ** 2
        signal[:fade] *= ramp
        signal[-fade:] *= ramp[::-1]
    return signal.astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--frequencies", type=float, nargs="+", default=None)
    parser.add_argument("--amplitude", type=float, default=0.002)
    parser.add_argument("--tone-seconds", type=float, default=0.75)
    parser.add_argument("--gap-seconds", type=float, default=0.40)
    parser.add_argument("--confirm-volume-minimum", action="store_true")
    parser.add_argument("--confirm-speaker", action="store_true")
    parser.add_argument("--confirm-user-present", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if not (args.confirm_volume_minimum and args.confirm_speaker and args.confirm_user_present):
        print(
            "[중단] 스피커 연결·사용자 입회·볼륨 최저 플래그가 필요합니다.",
            file=sys.stderr,
        )
        return 2
    if not 0.0001 <= args.amplitude <= 0.05:
        raise ValueError("amplitude는 0.0001–0.05 범위여야 합니다")
    if args.tone_seconds <= 0.1 or args.gap_seconds < 0.1:
        raise ValueError("tone-seconds>0.1, gap-seconds>=0.1이어야 합니다")

    cfg = load_runtime_config(args.config)
    audio = cfg["hardware"]["audio"]
    import sounddevice as sd
    assert_live_pcm_clock_preconditions(audio)
    assert_measurement_preconditions(sd, audio)
    channels = cfg["hardware"]["channels"]
    fs = int(audio["sample_rate"])
    output_device = resolve_alsa_portaudio_device(
        audio["output"]["card"], audio["output"]["pcm"], "output", 2
    )
    if args.frequencies is None:
        resonances = cfg["duct"]["acoustics"]["axial_resonances_hz"]
        frequencies = sorted({float(value) for value in [*resonances, 300.0]})
    else:
        frequencies = [float(value) for value in args.frequencies]
    if any(value <= 0.0 or value >= fs / 2.0 for value in frequencies):
        raise ValueError("모든 주파수는 0보다 크고 Nyquist보다 작아야 합니다")

    noise_channel = int(channels["noise_out"])
    gap = np.zeros((int(round(fs * args.gap_seconds)), 2), dtype=np.int16)
    print(
        f"[정성 진단] output device={output_device}, ch{noise_channel}, "
        f"peak={args.amplitude:.4f}; 상쇄 ch{int(channels['cancel_out'])}는 항상 0"
    )
    with sd.OutputStream(
        samplerate=fs,
        blocksize=int(audio["block_size"]),
        device=output_device,
        channels=2,
        dtype="int16",
        latency=str(audio.get("latency", "low")),
    ) as stream:
        for frequency in frequencies:
            print(f"  {frequency:.0f} Hz", flush=True)
            mono = stepped_tone(
                frequency, fs, args.tone_seconds, args.amplitude
            )
            stereo = np.zeros((mono.size, 2), dtype=np.float32)
            stereo[:, noise_channel] = mono
            started = time.perf_counter()
            tone_underflow = bool(stream.write(float32_to_pcm_int16(stereo)))
            gap_underflow = bool(stream.write(gap))
            elapsed = time.perf_counter() - started
            if tone_underflow or gap_underflow:
                raise RuntimeError(f"{frequency:.0f}Hz 출력에서 underflow가 발생했습니다")
            expected = args.tone_seconds + args.gap_seconds
            if elapsed > expected + 1.0:
                raise RuntimeError(
                    f"{frequency:.0f}Hz 출력 시간이 비정상입니다: "
                    f"{elapsed:.2f}s (예상 {expected:.2f}s)"
                )

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (
        Path(args.out)
        if args.out
        else REPO_ROOT / "results" / f"playback_probe_{stamp}.json"
    )
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "measurement_kind": "qualitative_output_only",
        "microphone_data": False,
        "config": args.config,
        "sample_rate": fs,
        "output_device": output_device,
        "noise_channel": noise_channel,
        "cancel_channel_silent": True,
        "frequencies_hz": frequencies,
        "amplitude_peak": args.amplitude,
        "tone_seconds": args.tone_seconds,
        "gap_seconds": args.gap_seconds,
        "performance_claim_allowed": False,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"자극 로그 저장: {out}")
    print("마이크 측정값이 없으므로 duct/P/S/감쇠 수치는 갱신하지 않습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
