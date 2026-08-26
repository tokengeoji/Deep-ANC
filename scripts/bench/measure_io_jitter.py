#!/usr/bin/env python3
"""연속 full-duplex 스트림에서 DAC→ADC 도달 지연의 **지터**를 측정한다.

``measure_io_latency.py`` 는 블록/레이턴시 조합별 평균 왕복 지연을 스윕한다. 이 도구는
다른 질문에 답한다 — **끊김 없이 계속 재생하는 동안 그 지연이 얼마나 흔들리는가.**
ANC 는 고정 위상을 전제하므로 평균 지연보다 지터가 성능을 먼저 결정한다.

왜 별도 도구인가
----------------
덕트 전달맵이 INVALID 로 나왔을 때 원인이 레벨인지 배선인지 시간축인지 구분되지 않았다.
자극 진폭을 4배로 올려도 coherence 가 오히려 떨어졌고, 그제서야 시간축이 범인임을 알았다.
그 판별을 재현 가능하게 만든 것이 이 스크립트다.

두 가지 설계가 핵심이다.

1. **ERR−REF 자기검증.** 두 마이크는 같은 ADC 클록이므로 상대 지연은 물리적으로 고정이다.
   따라서 ERR−REF 가 흔들리면 그것은 하드웨어가 아니라 **추정기가 틀린 것**이다. 이를
   유효성 판정에 쓰면 잔향 때문에 생기는 가짜 지터를 걸러낼 수 있다.
2. **대역제한 PHAT.** 전대역 PHAT 는 자극이 없는 대역(여기서는 1.6kHz 이상)의 잡음까지
   백색화해 증폭하므로 피크가 엉뚱한 곳에 잡힌다. 실제로 그렇게 −5408 샘플 같은 값이
   나왔다. 가중을 자극 대역으로 한정해야 한다.

판정
----
ERR−REF 가 안정한데 ERR/REF 가 **함께** 흔들리면 DAC→ADC 경로의 실제 지터다(common-mode).
두 채널이 따로 흔들리면 추정 잡음이므로 측정 조건을 고쳐야 한다.

⚠ 스피커에서 처프가 재생된다 — 사용자 입회 + 볼륨 최소 상태에서만 실행할 것.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.config import load_yaml  # noqa: E402
from deep_anc.audio_io import (  # noqa: E402
    assert_measurement_preconditions, pcm_int32_to_float32,
)
from deep_anc.dsp.measurement_level import assert_live_pcm_clock_preconditions  # noqa: E402

MAX_AMPLITUDE = 0.02
DEFAULT_AMPLITUDE = 0.02


def resolve_devices(hardware: dict):
    import sounddevice as sd

    audio = hardware["audio"]

    def find(card: str, pcm: int, want_input: bool) -> int:
        suffix = f",{pcm}"
        for index, device in enumerate(sd.query_devices()):
            channels = (
                device["max_input_channels"] if want_input else device["max_output_channels"]
            )
            if channels >= 2 and card in device["name"] and device["name"].rstrip(")").endswith(suffix):
                return index
        for index, device in enumerate(sd.query_devices()):
            channels = (
                device["max_input_channels"] if want_input else device["max_output_channels"]
            )
            if channels >= 2 and card in device["name"]:
                return index
        raise SystemExit(f"장치를 찾지 못했습니다: card={card} pcm={pcm}")

    return (
        find(audio["input"]["card"], audio["input"]["pcm"], True),
        find(audio["output"]["card"], audio["output"]["pcm"], False),
    )


def make_chirp(fs: int, period: int, low: float, high: float) -> np.ndarray:
    t = np.arange(period) / fs
    chirp = np.sin(2 * np.pi * (low * t + (high - low) / (2 * t[-1]) * t**2)).astype(np.float32)
    # 주기 경계 불연속은 광대역 클릭을 만들어 추정을 망친다. 양끝을 페이드한다.
    ramp = max(1, int(0.01 * fs))
    window = np.ones(period, dtype=np.float32)
    window[:ramp] = np.linspace(0.0, 1.0, ramp)
    window[-ramp:] = np.linspace(1.0, 0.0, ramp)
    return chirp * window


def band_limited_onset(signal: np.ndarray, reference_fft: np.ndarray, band: np.ndarray,
                       nfft: int, period: int) -> int:
    """자극 대역에만 PHAT 가중을 적용한 상호상관의 피크 위치."""

    spectrum = np.fft.rfft(signal, nfft)
    cross = spectrum * np.conj(reference_fft)
    magnitude = np.abs(cross)
    magnitude[magnitude < 1e-15] = 1e-15
    weighted = np.zeros_like(cross)
    weighted[band] = cross[band] / magnitude[band]
    envelope = np.abs(np.fft.irfft(weighted, nfft)[:period])
    return int(np.argmax(envelope))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument("--amplitude", type=float, default=DEFAULT_AMPLITUDE)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--period-seconds", type=float, default=1.0)
    parser.add_argument("--band", type=float, nargs=2, default=[80.0, 1600.0])
    parser.add_argument("--blocksize", type=int, default=256)
    parser.add_argument("--latency", choices=("low", "high"), default="low")
    parser.add_argument("--max-jitter-ms", type=float, default=1.0)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--confirm-volume-minimum",
        action="store_true",
        help="사용자 입회 + 앰프 볼륨 최소 확인. 없으면 실행하지 않는다",
    )
    parser.add_argument("--confirm-speaker", action="store_true")
    parser.add_argument("--confirm-user-present", action="store_true")
    args = parser.parse_args(argv)

    if not (args.confirm_volume_minimum and args.confirm_speaker and args.confirm_user_present):
        print("[중단] 스피커 연결·사용자 입회·볼륨 최저를 모두 확인해야 합니다.", file=sys.stderr)
        return 2
    if not 0.0 < args.amplitude <= MAX_AMPLITUDE:
        print(f"[중단] --amplitude 는 0 초과 {MAX_AMPLITUDE} 이하여야 합니다", file=sys.stderr)
        return 2

    import sounddevice as sd

    hardware = load_yaml(args.hardware)
    try:
        assert_live_pcm_clock_preconditions(hardware["audio"])
        import sounddevice as sd
        assert_measurement_preconditions(sd, hardware["audio"])
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] 오디오 사전점검 실패: {exc}", file=sys.stderr)
        return 2
    fs = int(hardware["audio"].get("sample_rate", 48000))
    device_in, device_out = resolve_devices(hardware)
    period = int(args.period_seconds * fs)
    count = max(3, int(args.seconds / args.period_seconds))

    reference = make_chirp(fs, period, args.band[0], args.band[1])
    playback = np.zeros((period * count, 2), dtype=np.float32)
    playback[:, 0] = np.tile(reference * args.amplitude, count)  # ch0 = 소음 스피커

    print(f"입력  {sd.query_devices()[device_in]['name']}")
    print(f"출력  {sd.query_devices()[device_out]['name']}")
    print(f"연속 {count}주기 × {args.period_seconds}초 · peak {args.amplitude} · "
          f"blocksize {args.blocksize}/{args.latency}")
    recording = sd.playrec(
        playback, samplerate=fs, channels=2, dtype=("int32", "int16"),
        device=(device_in, device_out), blocking=True,
        latency=args.latency, blocksize=args.blocksize,
    )

    nfft = 1 << int(np.ceil(np.log2(2 * period)))
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    band = (freqs >= args.band[0]) & (freqs <= args.band[1])
    reference_fft = np.fft.rfft(reference.astype(np.float64), nfft)

    err_all, ref_all = [], []
    recording_float = pcm_int32_to_float32(recording)
    error_channel = recording_float[:, 0].astype(np.float64)
    reference_channel = recording_float[:, 1].astype(np.float64)
    for k in range(1, count - 1):
        err_seg = error_channel[k * period : (k + 2) * period]
        ref_seg = reference_channel[k * period : (k + 2) * period]
        if err_seg.size < 2 * period:
            break
        err_all.append(band_limited_onset(err_seg, reference_fft, band, nfft, period))
        ref_all.append(band_limited_onset(ref_seg, reference_fft, band, nfft, period))

    err_all = np.asarray(err_all, dtype=float)
    ref_all = np.asarray(ref_all, dtype=float)
    difference = err_all - ref_all
    # ERR-REF 는 물리적으로 고정이므로, 여기서 벗어난 주기는 추정 실패로 본다.
    median_difference = float(np.median(difference))
    valid = np.abs(difference - median_difference) <= 5
    clip_ratio = float(np.mean(np.abs(recording_float) >= 0.999))

    print(f"\n입력 peak {np.max(np.abs(recording_float)):.4f} · clip {clip_ratio*100:.3f}%")
    print(f"유효 추정 {int(valid.sum())}/{len(valid)} 주기 (ERR−REF 중앙 {median_difference:.0f} samples)")

    if valid.sum() < 3:
        print("[INVALID] 유효 추정이 부족합니다 — 진폭·대역·잔향 조건을 확인하세요.", file=sys.stderr)
        return 1

    # 유효 주기만 보고하면 걸러낸 주기에 나쁜 소식이 숨는다. 실제로 그렇게 해서 게이트를
    # PASS 로 읽었는데 전달맵은 같은 조건에서 수백 샘플 spread 를 보고했다. 전 주기 계열과
    # 선형 드리프트를 항상 함께 낸다 — 드리프트는 지터와 다른 고장이고 대응도 다르다.
    index = np.arange(len(err_all), dtype=float)
    drift_slope = (
        float(np.polyfit(index, err_all, 1)[0]) if len(err_all) >= 3 else float("nan")
    )
    drift_total = drift_slope * (len(err_all) - 1)
    print(
        f"전 주기 ERR 범위 {np.ptp(err_all):.0f} sm ({np.ptp(err_all)/fs*1000:.2f} ms)"
        f" · 드리프트 {drift_slope:+.2f} sm/주기 (총 {drift_total:+.0f} sm,"
        f" {drift_total/fs*1000:+.2f} ms)"
    )
    print(f"  ERR 전 주기: {[int(v) for v in err_all]}")
    print(f"  ERR−REF   : {[int(v) for v in difference]}")

    def summary(name: str, values: np.ndarray) -> dict:
        span = float(values.max() - values.min())
        record = {
            "median_samples": float(np.median(values)),
            "range_samples": span,
            "range_ms": span / fs * 1000.0,
            "std_samples": float(values.std()),
            "std_ms": float(values.std()) / fs * 1000.0,
        }
        print(f"  {name:22s} 중앙 {record['median_samples']:8.1f} sm  "
              f"범위 {span:5.0f} sm ({record['range_ms']:5.2f} ms)  "
              f"std {record['std_samples']:5.1f} sm ({record['std_ms']:.3f} ms)")
        return record

    print("\n[유효 주기]")
    report = {
        "sample_rate": fs,
        "amplitude": args.amplitude,
        "blocksize": args.blocksize,
        "latency": args.latency,
        "band_hz": list(args.band),
        "valid_periods": int(valid.sum()),
        "total_periods": int(len(valid)),
        "input_clip_ratio": clip_ratio,
        "err_dac_to_adc": summary("ERR (DAC→ADC)", err_all[valid]),
        "ref_dac_to_adc": summary("REF (DAC→ADC)", ref_all[valid]),
        "err_minus_ref": summary("ERR−REF (ADC 내부)", difference[valid]),
        # 걸러낸 주기까지 포함한 원자료. 판정은 유효 주기로 하되 근거는 전부 남긴다.
        "all_periods": {
            "err_onset_samples": [float(v) for v in err_all],
            "ref_onset_samples": [float(v) for v in ref_all],
            "err_minus_ref_samples": [float(v) for v in difference],
            "valid_mask": [bool(v) for v in valid],
            "err_range_samples": float(np.ptp(err_all)),
            "err_drift_samples_per_period": drift_slope,
            "err_drift_total_samples": drift_total,
        },
    }

    estimator = report["err_minus_ref"]["std_ms"]
    measured = report["err_dac_to_adc"]["std_ms"]
    report["estimator_noise_ms"] = estimator
    report["common_mode"] = bool(estimator < 0.25 * measured) if measured > 0 else False
    report["jitter_gate_pass"] = bool(report["err_dac_to_adc"]["range_ms"] <= args.max_jitter_ms)

    print(f"\n[판정]")
    print(f"  추정기 자체 잡음(ERR−REF std) : {estimator:.3f} ms")
    print(f"  측정된 DAC→ADC 지터(ERR std)  : {measured:.3f} ms")
    if report["common_mode"]:
        print(f"  => COMMON-MODE ({measured/max(estimator,1e-9):.0f}배) — 실제 출력 경로 지터다.")
    else:
        print("  => 추정 잡음 비중이 크다. 결과를 지터로 해석하지 말 것.")
    print(f"  {args.max_jitter_ms:.1f}ms 게이트: "
          f"{'PASS' if report['jitter_gate_pass'] else 'FAIL'}")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n산출물: {path}")
    return 0 if report["jitter_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
