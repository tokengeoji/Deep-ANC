#!/usr/bin/env python3
"""재생 클록과 녹음 클록이 실제로 얼마나, 어떤 모양으로 어긋나는지 잰다.

⚠ 스피커가 약 15초간 울린다(peak 0.02, 측정용 최저 레벨). 사용자 입회 하에만 실행한다.

왜 다시 재는가
--------------
2026-08-04 인터리브 측정에서 "1초 창 안에서 100~200샘플 warp"라는 결론이 나왔다. 그런데
그 추정에는 **교란 요인**이 하나 있다. 자극이 Schroeder 위상 멀티톤이고, 이는 사실상
선형 처프다. 짧은 창으로 상호상관을 하면 창마다 **다른 주파수 대역**이 들어가고, 덕트는
공진 때문에 주파수별 군지연이 크게 다르다. 즉 "창을 옮기니 지연이 달라졌다"가
클록 warp 인지 **덕트의 군지연 분산**인지 그 측정으로는 구분할 수 없다.

실제로 관측된 궤적에는 자극 주기(1초 = 창 4개)와 맞물린 듯한 반복 패턴이 있었다.
클록 드리프트라면 자극 주기와 무관해야 한다.

설계
----
* 자극은 **매 주기 완전히 동일한** 대역제한 잡음 버스트다. 처프가 아니므로 자기상관
  첨두가 좁고(대역폭 1.5kHz → 약 32샘플), 주기끼리 비교할 때 주파수 편향이 없다.
* 주기 k 를 **주기 0** 과 통째로(1초) 상관시킨다. 창을 옮기지 않으므로 군지연 분산이
  두 신호에 **똑같이** 실려 상쇄된다. 남는 것은 순수한 시간축 어긋남뿐이다.
* lag(k) 를 k 에 대해 1차 적합한다.
    - 기울기가 유의하고 잔차가 작다  → **고정 rate 오프셋**. 리샘플링으로 정확히 보정된다.
    - 기울기 ≈ 0 인데 잔차가 크다     → 무작위 점프. 버퍼 관리/드롭 의심.
    - 둘 다 작다                      → 클록은 문제가 없고 앞선 결론이 틀렸다.
* 대조군: 같은 주기 안에서 ERR 과 REF 를 비교한다. 둘은 **같은 ADC** 를 쓰므로 여기서
  어긋나면 원인은 클록이 아니라 분석 코드다.

    .venv/bin/python scripts/bench/measure_clock_drift.py --confirm-volume-minimum
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "data"))

import calibrate_wideband as cw  # noqa: E402

from deep_anc.audio_io import (  # noqa: E402
    assert_measurement_preconditions, pcm_int32_to_float32, resolve_alsa_portaudio_device,
)
from deep_anc.config import REPO_ROOT, load_yaml  # noqa: E402
from deep_anc.dsp.measurement_level import assert_live_pcm_clock_preconditions  # noqa: E402


def band_limited_burst(
    *, sample_rate: int, seconds: float, band_hz: tuple[float, float],
    amplitude: float, seed: int,
) -> np.ndarray:
    """대역제한 잡음. 처프가 아니라 위상이 무작위라 자기상관 첨두가 대칭이고 좁다."""

    n = int(round(seconds * sample_rate))
    rng = np.random.default_rng(seed)
    spectrum = np.zeros(n // 2 + 1, dtype=np.complex128)
    lo = int(np.ceil(band_hz[0] * n / sample_rate))
    hi = int(np.floor(band_hz[1] * n / sample_rate))
    spectrum[lo : hi + 1] = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, hi + 1 - lo))
    signal = np.fft.irfft(spectrum, n)
    return (signal / float(np.max(np.abs(signal))) * amplitude).astype(np.float32)


def lag_between(a: np.ndarray, b: np.ndarray, *, search: int) -> tuple[float, float]:
    """b 가 a 보다 얼마나 늦는지(샘플, 소수점까지)와 정규화 상관 첨두."""

    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    corr = np.correlate(y, x, mode="full")
    centre = x.size - 1
    window = corr[centre - search : centre + search + 1]
    index = int(np.argmax(np.abs(window)))
    if 0 < index < window.size - 1:
        y0, y1, y2 = window[index - 1], window[index], window[index + 1]
        denominator = y0 - 2.0 * y1 + y2
        fraction = 0.5 * (y0 - y2) / denominator if denominator != 0.0 else 0.0
    else:
        fraction = 0.0
    norm = float(np.linalg.norm(x) * np.linalg.norm(y))
    peak = float(window[index] / norm) if norm > 0.0 else 0.0
    return float(index - search) + float(fraction), peak


def capture_via_alsa(
    hardware: dict, *, playback: np.ndarray, fs: int, session: Path
) -> tuple[np.ndarray, dict]:
    """``aplay``/``arecord`` 를 각각 띄워 PortAudio 를 경로에서 뺀다.

    PortAudio 는 서로 다른 카드의 전이중 스트림을 한 콜백으로 묶는다. 두 카드가 각자
    클록을 가지면 어느 쪽이든 데이터가 모자라거나 남는 순간이 오고, 그때 프레임을
    버리거나 채워 넣는다. 그 조작은 xrun 으로 보고되지 않으면서 시간축에 계단을 만든다.

    ALSA 를 직접 쓰면 두 스트림이 각자 자기 클록으로 자유롭게 흐른다. 상대 시간은
    **단조롭게만** 어긋난다. 따라서 이 경로에서 계단이 사라지면 범인은 PortAudio 다.
    """

    import shutil
    import subprocess

    import soundfile as sf

    for tool in ("aplay", "arecord"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} 이 없습니다 (alsa-utils)")

    play_path = session / "alsa_playback.wav"
    rec_path = session / "alsa_capture.wav"
    sf.write(str(play_path), playback, fs, subtype="PCM_16")

    in_cfg, out_cfg = hardware["input"], hardware["output"]
    in_dev = f"hw:{in_cfg['card']},{in_cfg['pcm']}"
    out_dev = f"hw:{out_cfg['card']},{out_cfg['pcm']}"
    seconds = playback.shape[0] / fs + 2.0

    recorder = subprocess.Popen(
        ["arecord", "-D", in_dev, "-f", "S32_LE", "-r", str(fs), "-c", "2",
         "-d", str(int(seconds) + 1), "-q", str(rec_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    time.sleep(1.0)   # 녹음이 확실히 흐른 뒤 재생을 시작한다
    player = subprocess.run(
        ["aplay", "-D", out_dev, "-q", str(play_path)],
        capture_output=True, text=True,
    )
    _, rec_err = recorder.communicate(timeout=seconds + 20)

    if player.returncode != 0:
        raise RuntimeError(f"aplay 실패: {player.stderr.strip()[:200]}")
    if recorder.returncode != 0:
        raise RuntimeError(f"arecord 실패: {rec_err.decode(errors='replace')[:200]}")

    data, rate = sf.read(str(rec_path), dtype="int32", always_2d=True)
    if int(rate) != fs:
        raise RuntimeError(f"녹음 샘플레이트 {rate} != {fs}")
    return data.astype(np.int32), {
        "backend": "alsa",
        "xrun_count": 0,
        "completed": True,
        "aplay_stderr": player.stderr.strip()[:200],
        "arecord_stderr": rec_err.decode(errors="replace").strip()[:200],
    }


def locate_playback_start(
    err: np.ndarray, burst: np.ndarray, *, lead_in: int, fs: int
) -> int:
    """녹음에서 첫 버스트가 시작된 인덱스를 찾아 ``lead_in`` 기준으로 되돌린다."""

    probe = np.asarray(burst, dtype=np.float64)
    limit = min(err.size, probe.size * 6)
    segment = err[:limit] - err[:limit].mean()
    corr = np.correlate(segment, probe - probe.mean(), mode="valid")
    start = int(np.argmax(np.abs(corr)))
    return max(0, start - lead_in)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument("--period-seconds", type=float, default=1.0)
    parser.add_argument("--periods", type=int, default=14)
    parser.add_argument("--band", type=float, nargs=2, default=[80.0, 1600.0])
    parser.add_argument("--amplitude", type=float, default=0.02)
    parser.add_argument("--output-channel", choices=["noise", "cancel"], default="noise")
    parser.add_argument("--search", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--latency", choices=["low", "high"], default="high")
    parser.add_argument("--input-probe-seconds", type=float, default=2.0)
    parser.add_argument("--out-root", default="results/clock_drift")
    parser.add_argument(
        "--backend",
        choices=["portaudio", "alsa"],
        default="portaudio",
        help=(
            "portaudio = 지금 측정/실기가 쓰는 경로(전이중 한 콜백). "
            "alsa = aplay/arecord 를 각각 띄워 PortAudio 의 교차 카드 정렬을 배제한다"
        ),
    )
    parser.add_argument("--confirm-volume-minimum", action="store_true")
    parser.add_argument("--confirm-speaker", action="store_true")
    parser.add_argument("--confirm-user-present", action="store_true")
    args = parser.parse_args(argv)

    if not (args.confirm_volume_minimum and args.confirm_speaker and args.confirm_user_present):
        print(
            "[중단] 스피커 연결·사용자 입회·볼륨 최저를 모두 확인해야 합니다.",
            file=sys.stderr,
        )
        return 2
    if not 0.0 < args.amplitude <= cw.MAX_AMPLITUDE:
        print(f"[중단] --amplitude 는 0 초과 {cw.MAX_AMPLITUDE} 이하", file=sys.stderr)
        return 2

    hardware = load_yaml(REPO_ROOT / args.hardware)["audio"]
    try:
        assert_live_pcm_clock_preconditions(hardware)
        import sounddevice as sd
        assert_measurement_preconditions(sd, hardware)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] 오디오 사전점검 실패: {exc}", file=sys.stderr)
        return 2
    fs = int(hardware["sample_rate"])
    block_size = int(args.block_size or hardware["block_size"])
    period = int(round(args.period_seconds * fs))
    out_root = cw._repo_path(args.out_root, require_results=True)
    session = out_root / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=False)

    burst = band_limited_burst(
        sample_rate=fs, seconds=args.period_seconds,
        band_hz=(float(args.band[0]), float(args.band[1])),
        amplitude=float(args.amplitude), seed=20260804,
    )
    lead_in = fs // 2
    total = lead_in + int(args.periods) * period
    playback = np.zeros((total, 2), dtype=np.float32)
    channel = 1 if args.output_channel == "cancel" else 0
    playback[lead_in:, channel] = np.tile(burst, int(args.periods))

    print(
        f"클록 드리프트 측정 — 동일 버스트 {args.periods}회 × {args.period_seconds:.2f}s "
        f"({total / fs:.1f}초 재생)\n"
        f"  대역 {args.band[0]:.0f}-{args.band[1]:.0f}Hz · peak {args.amplitude:.4f} · "
        f"{args.output_channel}/ch{channel} · block {block_size} · latency {args.latency}"
    )

    try:
        import sounddevice as sd

        print("출력 없는 ERR/REF raw preflight 중...")
        preflight_raw, preflight = cw._capture_preflight(sd, hardware, args.input_probe_seconds)
        for name, item in zip(("ERR", "REF"), cw._probe_summary(preflight)):
            print(
                f"[{'PASS' if item['valid'] else 'FAIL'}] {name}: "
                f"RMS {item['rms_dbfs']:.2f}dBFS, peak {item['peak']:.6f}, "
                f"clip {item['clip_ratio']:.3%}"
            )
        channels = preflight.get("channels", [])
        if len(channels) < 2 or not all(bool(c.get("valid")) for c in channels[:2]):
            print("[실패] 양 마이크 preflight 실패", file=sys.stderr)
            return 1

        if args.backend == "alsa":
            recorded_raw, telemetry = capture_via_alsa(
                hardware, playback=playback, fs=fs, session=session
            )
        else:
            output_cfg = hardware["output"]
            recorded_raw, _, telemetry = cw._capture_measurement(
                sd, fs=fs, block_size=block_size, latency=str(args.latency),
                in_dev=int(preflight["device"]),
                out_dev=resolve_alsa_portaudio_device(
                    output_cfg["card"], output_cfg["pcm"], "output", 2
                ),
                output_float=playback,
            )
    except (ImportError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"[실패] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    recorded = pcm_int32_to_float32(recorded_raw)
    if args.backend == "alsa":
        # ALSA 경로는 재생/녹음 시작 시각이 서로 독립이다. 재생 시작 위치를 녹음에서
        # 찾아 lead_in 기준으로 맞춘다 — 이후 분석은 PortAudio 경로와 완전히 같다.
        offset = locate_playback_start(
            recorded[:, 0].astype(np.float64), burst, lead_in=lead_in, fs=fs
        )
        print(f"  ALSA: 재생 시작 정렬 오프셋 {offset} 샘플")
        recorded = recorded[offset:]
        recorded_raw = recorded_raw[offset:]
    err = recorded[:, 0].astype(np.float64)
    ref = recorded[:, 1].astype(np.float64)

    def slice_period(signal: np.ndarray, k: int) -> np.ndarray:
        return signal[lead_in + k * period : lead_in + (k + 1) * period]

    # 첫 주기는 스트림 정착 전이라 버린다.
    first = 1
    base = slice_period(err, first)
    rows = []
    for k in range(first, int(args.periods)):
        segment = slice_period(err, k)
        if segment.size != period:
            break
        lag, peak = lag_between(base, segment, search=args.search)
        ref_lag, ref_peak = lag_between(segment, slice_period(ref, k), search=args.search)
        rows.append(
            {
                "period": k,
                "lag_vs_first": lag,
                "peak": peak,
                "err_ref_lag": ref_lag,
                "err_ref_peak": ref_peak,
            }
        )

    index = np.array([r["period"] - first for r in rows], dtype=np.float64)
    lags = np.array([r["lag_vs_first"] for r in rows], dtype=np.float64)
    peaks = np.array([r["peak"] for r in rows], dtype=np.float64)
    slope, intercept = np.polyfit(index, lags, 1) if index.size >= 2 else (0.0, 0.0)
    residual = lags - (slope * index + intercept)
    ppm = slope / period * 1e6

    print("\n주기  주기0 대비 지연  상관   ERR-REF 지연  상관")
    for row in rows:
        print(
            f"{row['period']:4d}  {row['lag_vs_first']:14.2f}  {row['peak']:.4f}  "
            f"{row['err_ref_lag']:11.2f}  {row['err_ref_peak']:.4f}"
        )

    print(
        f"\n1차 적합: 기울기 {slope:+.3f} 샘플/주기 = **{ppm:+.1f} ppm**\n"
        f"          잔차 rms {float(np.std(residual)):.2f} 샘플 · "
        f"최대 {float(np.max(np.abs(residual))):.2f} 샘플\n"
        f"          상관 첨두 중앙 {float(np.median(peaks)):.4f} "
        f"(최소 {float(np.min(peaks)):.4f})"
    )

    drift_dominant = abs(slope) > 3.0 * max(float(np.std(residual)), 1e-6)
    if float(np.median(peaks)) < 0.5:
        verdict = "상관 자체가 낮다 — 신호가 약하거나 비선형이 크다. 레벨/배선을 먼저 본다"
    elif drift_dominant:
        verdict = (
            f"**고정 rate 오프셋**({ppm:+.1f} ppm)이 지배적이다. "
            "녹음을 이 비율로 리샘플링하면 정확히 보정된다"
        )
    elif float(np.std(residual)) > 10.0:
        verdict = (
            "기울기는 작은데 잔차가 크다 — 무작위 점프다. "
            "버퍼 드롭/중복을 의심하고 ALSA 직접 경로로 재확인한다"
        )
    else:
        verdict = (
            "기울기도 잔차도 작다 — **클록은 문제가 아니다**. "
            "앞선 warp 결론은 멀티톤 처프의 주파수별 군지연을 오인한 것이다"
        )
    print(f"\n판정: {verdict}")

    payload = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sample_rate": fs,
        "period_samples": period,
        "periods": int(args.periods),
        "band_hz": [float(args.band[0]), float(args.band[1])],
        "amplitude": float(args.amplitude),
        "output_channel": args.output_channel,
        "block_size": block_size,
        "latency": args.latency,
        "slope_samples_per_period": float(slope),
        "drift_ppm": float(ppm),
        "residual_rms_samples": float(np.std(residual)),
        "residual_max_samples": float(np.max(np.abs(residual))) if residual.size else 0.0,
        "peak_median": float(np.median(peaks)) if peaks.size else 0.0,
        "verdict": verdict,
        "rows": rows,
        "telemetry": telemetry,
        "preflight": preflight,
    }
    (session / "clock_drift.json").write_text(
        json.dumps(cw._json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        session / "raw.npz",
        err=recorded[:, 0].astype(np.float32),
        ref=recorded[:, 1].astype(np.float32),
        burst=burst,
        preflight_raw_int32=preflight_raw.astype(np.int32),
    )
    print(f"산출물: {session.relative_to(REPO_ROOT)}/{{clock_drift.json, raw.npz}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
