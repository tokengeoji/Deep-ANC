#!/usr/bin/env python3
"""I/O 왕복 지연 스윕 — 처프 재생(상쇄 스피커)→에러 마이크 상호상관.

  .venv/bin/python scripts/bench/measure_io_latency.py --blocks 128 256 512 --repeats 3 \
      --out results/latency_sweep.md
⚠ 스피커에서 처프가 재생된다 — 볼륨을 낮춘 상태로 실행할 것.
결과는 캘리브레이션/학습 지연 가정(secondary_path npz + handoff)의 검증 자료다.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.audio_io import (                          # noqa: E402
    assert_measurement_preconditions,
    pcm_int32_to_float32,
    resolve_alsa_portaudio_device,
)
from deep_anc.config import REPO_ROOT, load_yaml         # noqa: E402
from deep_anc.dsp.measurement_level import assert_live_pcm_clock_preconditions  # noqa: E402


def measure_once(sd, in_dev, out_dev, fs, block, latency, seconds=3.0, amp=0.05):
    t = np.arange(int(seconds * fs)) / fs
    chirp = (amp * signal.chirp(t, 150.0, seconds, 1500.0, method="logarithmic")).astype(np.float32)
    fade = int(0.05 * fs)
    chirp[:fade] *= np.linspace(0, 1, fade)
    chirp[-fade:] *= np.linspace(1, 0, fade)
    total = chirp.size + int(1.0 * fs)
    recorded = np.zeros(total, dtype=np.float32)
    cursor = {"i": 0}

    def callback(indata, outdata, frames, _t, status):
        i = cursor["i"]
        n = min(frames, total - i)
        recorded[i : i + n] = pcm_int32_to_float32(indata[:n, :2])[:, 0]
        out = np.zeros((frames, 2), dtype=np.float32)
        m = max(0, min(frames, chirp.size - i))
        if m > 0:
            out[:m, 1] = chirp[i : i + m]          # ch1 = 상쇄 스피커
        outdata[:] = np.rint(np.clip(out, -1, 1) * 32767).astype(np.int16)
        cursor["i"] = i + n
        if cursor["i"] >= total:
            raise sd.CallbackStop

    with sd.Stream(
        samplerate=fs, blocksize=block, device=(in_dev, out_dev),
        channels=(2, 2), dtype=("int32", "int16"), latency=(latency, latency),
        callback=callback, prime_output_buffers_using_stream_callback=True,
    ):
        while cursor["i"] < total:
            time.sleep(0.05)

    corr = signal.fftconvolve(recorded.astype(np.float64), chirp[::-1].astype(np.float64), "full")
    lag = int(np.argmax(np.abs(corr))) - (chirp.size - 1)
    peak = float(np.max(np.abs(corr)) / (np.linalg.norm(chirp) * np.linalg.norm(recorded) + 1e-12))
    return lag, peak


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument("--blocks", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--latencies", nargs="+", default=["low", "high"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="results/latency_sweep.md")
    parser.add_argument("--confirm-speaker", action="store_true")
    parser.add_argument("--confirm-user-present", action="store_true")
    parser.add_argument("--confirm-volume-minimum", action="store_true")
    args = parser.parse_args()

    if not (args.confirm_speaker and args.confirm_user_present and args.confirm_volume_minimum):
        print("[중단] 스피커 연결·사용자 입회·볼륨 최저를 모두 확인해야 합니다.", file=sys.stderr)
        return 2

    import sounddevice as sd

    hw = load_yaml(REPO_ROOT / args.hardware)["audio"]
    try:
        assert_live_pcm_clock_preconditions(hw)
        assert_measurement_preconditions(sd, hw)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] 오디오 사전점검 실패: {exc}", file=sys.stderr)
        return 2
    fs = int(hw["sample_rate"])
    in_dev = resolve_alsa_portaudio_device(hw["input"]["card"], hw["input"]["pcm"], "input", 2)
    out_dev = resolve_alsa_portaudio_device(hw["output"]["card"], hw["output"]["pcm"], "output", 2)

    rows = ["| block | latency | 지연(샘플) | 지연(ms) | 상관 피크 |", "|---|---|---|---|---|"]
    for block in args.blocks:
        for lat in args.latencies:
            lat_val = lat if lat in ("low", "high") else float(lat)
            lags = []
            for _ in range(args.repeats):
                lag, peak = measure_once(sd, in_dev, out_dev, fs, block, lat_val)
                lags.append(lag)
            med = int(np.median(lags))
            print(f"block {block:4d} / {lat:>5}: {med}샘플 = {1000*med/fs:.2f}ms (피크 {peak:.3f})")
            rows.append(f"| {block} | {lat} | {med} | {1000*med/fs:.2f} | {peak:.3f} |")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# I/O 왕복 지연 스윕\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
