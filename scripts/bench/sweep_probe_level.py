#!/usr/bin/env python3
"""자극 레벨을 바꿔가며 반복 일관성이 어떻게 변하는지 잰다 — G1 원인 판별용.

⚠ 스피커가 레벨당 약 8초씩 울린다. 사용자 입회 하에만 실행한다.

무엇을 가리는가
---------------
G1(실측 P/S)이 막혀 있고, 지금까지 좁혀진 후보는 둘이다.

  (a) **시간축**   재생↔녹음 대응이 흔들려 위상이 재현되지 않는다.
  (b) **레벨/비선형**  앰프 볼륨 최저 + 진폭 0.02 라 class-D 앰프가 크로스오버
      영역에서 동작하고, 그 비선형이 반복마다 다른 응답을 만든다.

두 가설은 **레벨 의존성**에서 갈린다.

  (a) 라면 일관성은 레벨과 거의 무관하다 — 시간축은 진폭을 모른다.
  (b) 라면 레벨을 올릴수록 일관성이 뚜렷하게 좋아진다.

이미 배제된 것: PortAudio(=ALSA 직접 경로도 같은 증상), ADC(=ERR/REF 지연이
13주기 내내 편차 0.5샘플), USB 커널 오류(=dmesg 0건).

정렬은 ``align_repeats`` 로 통일한다. 시간영역 온셋 검출은 대역제한 IR 의 선행 링잉
때문에 흔들려서, 안정적인 측정도 지터가 큰 것처럼 보고한다.

    .venv/bin/python scripts/bench/sweep_probe_level.py --confirm-volume-minimum
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
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
from deep_anc.dsp.interleaved_probe import (  # noqa: E402
    align_repeats, build_interleaved_probe, complex_consistency, estimate_transfer,
)

# 진단 상한. 0.06 은 recorded 80세션을 안전하게 수집한 레벨이라 새 위험이 아니다.
# official 측정의 상한(0.02)과는 별개이며, 이 스크립트의 산출물은 official 이 아니다.
DIAGNOSTIC_MAX_AMPLITUDE = 0.06
DEFAULT_FIT_BAND_HZ = (150.0, 600.0)
REPORT_BANDS = ((80, 150), (150, 300), (300, 600), (600, 1000), (1000, 1600))


def analyse(err: np.ndarray, probe, *, lead_in: int, warmup: int, repeats: int,
            fit_band: tuple[float, float]) -> dict:
    period = probe.period_samples
    out: dict = {}
    for drive in ("noise", "cancel"):
        rows = []
        for k in range(warmup, warmup + repeats):
            segment = err[lead_in + k * period : lead_in + (k + 1) * period]
            if segment.size != period:
                break
            rows.append(estimate_transfer(segment, probe, drive=drive))
        freq = rows[0][0]
        stack = np.stack([H for _, H in rows])
        aligned, taus, scores = align_repeats(
            freq, stack, sample_rate=probe.sample_rate, fit_band_hz=fit_band
        )
        consistency = complex_consistency(aligned)
        bands = {
            f"{lo}-{hi}": complex_consistency(
                aligned[:, (freq >= lo) & (freq <= hi)]
            )
            for lo, hi in REPORT_BANDS
        }
        out[drive] = {
            "raw_consistency": complex_consistency(stack),
            "aligned_consistency": consistency,
            "tau_spread_samples": float(np.max(taus) - np.min(taus)),
            "tau_samples": [float(v) for v in taus],
            "bands": bands,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument("--amplitudes", type=float, nargs="+",
                        default=[0.005, 0.02, 0.06])
    parser.add_argument(
        "--period-seconds", type=float, nargs="+", default=[1.0],
        help="분석 주기(초). 여러 값을 주면 진폭 × 주기 전부를 잰다",
    )
    parser.add_argument("--warmup-periods", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--band", type=float, nargs=2, default=[70.0, 1610.0])
    parser.add_argument("--latency", choices=["low", "high"], default="high")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--input-probe-seconds", type=float, default=2.0)
    parser.add_argument("--fit-band", type=float, nargs=2,
                        default=list(DEFAULT_FIT_BAND_HZ),
                        help="반복 시간이동 τ 를 적합할 대역 — 재현되는 대역만 넣는다")
    parser.add_argument("--out-root", default="results/level_sweep")
    parser.add_argument("--confirm-volume-minimum", action="store_true")
    parser.add_argument("--confirm-speaker", action="store_true")
    parser.add_argument("--confirm-user-present", action="store_true")
    args = parser.parse_args(argv)

    if not (args.confirm_volume_minimum and args.confirm_speaker and args.confirm_user_present):
        print("[중단] 스피커 연결·사용자 입회·볼륨 최저 플래그가 필요합니다.", file=sys.stderr)
        return 2
    amplitudes = [float(a) for a in args.amplitudes]
    if any(not 0.0 < a <= DIAGNOSTIC_MAX_AMPLITUDE for a in amplitudes):
        print(f"[중단] 진폭은 0 초과 {DIAGNOSTIC_MAX_AMPLITUDE} 이하", file=sys.stderr)
        return 2

    fit_band = (float(args.fit_band[0]), float(args.fit_band[1]))
    hardware = load_yaml(REPO_ROOT / args.hardware)["audio"]
    try:
        import sounddevice as sd
        assert_live_pcm_clock_preconditions(hardware)
        assert_measurement_preconditions(sd, hardware)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] 오디오 사전점검 실패: {exc}", file=sys.stderr)
        return 2
    fs = int(hardware["sample_rate"])
    block_size = int(args.block_size or hardware["block_size"])
    session = cw._repo_path(args.out_root, require_results=True) / dt.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    session.mkdir(parents=True, exist_ok=False)

    periods_seconds = [float(v) for v in args.period_seconds]
    total_periods = int(args.warmup_periods) + int(args.repeats)
    plan = [(a, p) for a in amplitudes for p in periods_seconds]
    sound = sum(total_periods * p + 0.5 for _, p in plan)
    print(
        f"스윕 {len(plan)}개 조합 — 진폭 {amplitudes} × 주기 {periods_seconds}s\n"
        f"  조합당 {total_periods} 주기 · 총 재생 약 {sound:.0f}초\n"
        f"  τ 적합 대역 {fit_band[0]:.0f}-{fit_band[1]:.0f}Hz · "
        f"block {block_size} · latency {args.latency}"
    )

    try:
        import sounddevice as sd

        print("출력 없는 ERR/REF raw preflight 중...")
        preflight_raw, preflight = cw._capture_preflight(sd, hardware, args.input_probe_seconds)
        floor_dbfs = float(preflight["channels"][0]["rms_dbfs"])
        for name, item in zip(("ERR", "REF"), cw._probe_summary(preflight)):
            print(
                f"[{'PASS' if item['valid'] else 'FAIL'}] {name}: "
                f"RMS {item['rms_dbfs']:.2f}dBFS, peak {item['peak']:.6f}, "
                f"clip {item['clip_ratio']:.3%}"
            )
        if not all(bool(c.get("valid")) for c in preflight["channels"][:2]):
            print("[실패] 양 마이크 preflight 실패", file=sys.stderr)
            return 1

        in_dev = int(preflight["device"])
        output_cfg = hardware["output"]
        out_dev = resolve_alsa_portaudio_device(
            output_cfg["card"], output_cfg["pcm"], "output", 2
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"[실패] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    lead_in = fs // 2
    rows: list[dict] = []
    for amplitude, period_seconds in plan:
        probe = build_interleaved_probe(
            sample_rate=fs, period_seconds=period_seconds,
            band_hz=(float(args.band[0]), float(args.band[1])), amplitude=amplitude,
        )
        playback = np.zeros(
            (lead_in + total_periods * probe.period_samples, 2), dtype=np.float32
        )
        playback[lead_in:, 0] = np.tile(probe.noise_signal, total_periods)
        playback[lead_in:, 1] = np.tile(probe.cancel_signal, total_periods)
        try:
            recorded_raw, _, telemetry = cw._capture_measurement(
                sd, fs=fs, block_size=block_size, latency=str(args.latency),
                in_dev=in_dev, out_dev=out_dev, output_float=playback,
            )
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            print(f"  [실패] amp {amplitude}: {exc}", file=sys.stderr)
            continue

        recorded = pcm_int32_to_float32(recorded_raw)
        err = recorded[:, 0].astype(np.float64)
        driven = err[lead_in + probe.period_samples :]
        driven_dbfs = 10.0 * np.log10(float(np.mean(driven**2)) + 1e-30)
        clip = float(np.mean(np.abs(recorded[:, :2]) >= 0.99))
        result = analyse(
            err, probe, lead_in=lead_in,
            warmup=int(args.warmup_periods), repeats=int(args.repeats),
            fit_band=fit_band,
        )
        rows.append({
            "amplitude": amplitude,
            "period_seconds": period_seconds,
            "driven_err_dbfs": driven_dbfs,
            "snr_db": driven_dbfs - floor_dbfs,
            "clip_ratio": clip,
            "xrun_count": int(telemetry.get("xrun_count", 0)),
            **{f"{d}_{k}": v for d, r in result.items()
               for k, v in r.items() if k != "tau_samples"},
        })
        print(
            f"\n  amp {amplitude:.3f} · 주기 {period_seconds:.3f}s  "
            f"ERR {driven_dbfs:6.1f} dBFS "
            f"(SNR {driven_dbfs - floor_dbfs:5.1f} dB) · clip {clip:.3%} · "
            f"xrun {telemetry.get('xrun_count', 0)}"
        )
        for drive in ("noise", "cancel"):
            item = result[drive]
            bands = "  ".join(
                f"{k}:{v:.3f}" for k, v in item["bands"].items()
            )
            print(
                f"    {drive:6s} 정렬전 {item['raw_consistency']:.3f} → "
                f"정렬후 **{item['aligned_consistency']:.3f}** · "
                f"τ폭 {item['tau_spread_samples']:6.1f}\n"
                f"           {bands}"
            )

    if not rows:
        print("[실패] 유효한 측정이 없습니다", file=sys.stderr)
        return 1

    print("\n" + "=" * 78)
    print("레벨 대비 정렬후 일관성 (요구 0.90)")
    print(f"{'amp':>7} {'주기s':>7} {'SNR':>7} {'noise':>7} {'cancel':>7}"
          f"   300-600  600-1000  1000-1600 (noise)")
    for row in rows:
        b = row["noise_bands"]
        print(
            f"{row['amplitude']:7.3f} {row['period_seconds']:7.3f} {row['snr_db']:7.1f} "
            f"{row['noise_aligned_consistency']:7.3f} "
            f"{row['cancel_aligned_consistency']:7.3f}   "
            f"{b['300-600']:7.3f} {b['600-1000']:9.3f} {b['1000-1600']:10.3f}"
        )

    by_amp = {}
    for row in rows:
        by_amp.setdefault(row["amplitude"], []).append(row)
    lo, hi = rows[0], rows[-1]
    gain = hi["noise_aligned_consistency"] - lo["noise_aligned_consistency"]
    if len(periods_seconds) > 1:
        first = min(rows, key=lambda r: r["period_seconds"])
        last = max(rows, key=lambda r: r["period_seconds"])
        step = (first["noise_aligned_consistency"]
                - last["noise_aligned_consistency"])
        verdict = (
            f"주기 {last['period_seconds']:.3f}s → {first['period_seconds']:.3f}s 에서 "
            f"일관성 {step:+.3f}. "
            + ("**창을 줄이면 좋아진다 — 주기 내 warp 가 원인**"
               if step > 0.1 else "창 길이에 둔감하다 — 주기 내 warp 가 원인이 아니다")
        )
        print(f"\n판정: {verdict}")
    elif gain > 0.15:
        verdict = (
            f"레벨을 올리자 일관성이 {gain:+.3f} 올랐다 → **레벨/비선형이 지배적**. "
            "official 진폭 상한(0.02)이 측정을 불가능하게 만들고 있다"
        )
    elif abs(gain) <= 0.05:
        verdict = (
            "레벨과 거의 무관하다 → **시간축 문제**. 진폭을 올려도 풀리지 않는다"
        )
    else:
        verdict = f"레벨 의존이 약하게 있다({gain:+.3f}) — 두 요인이 섞여 있을 수 있다"
    print(f"\n판정: {verdict}")

    payload = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sample_rate": fs,
        "period_seconds_list": periods_seconds,
        "warmup_periods": int(args.warmup_periods),
        "repeats": int(args.repeats),
        "design_band_hz": [float(args.band[0]), float(args.band[1])],
        "fit_band_hz": list(fit_band),
        "block_size": block_size,
        "latency": args.latency,
        "noise_floor_dbfs": floor_dbfs,
        "rows": rows,
        "verdict": verdict,
        "preflight": preflight,
    }
    (session / "level_sweep.json").write_text(
        json.dumps(cw._json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"산출물: {session.relative_to(REPO_ROOT)}/level_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
