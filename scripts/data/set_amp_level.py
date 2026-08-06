#!/usr/bin/env python3
"""앰프 볼륨을 **교정된 레벨**에 맞춘다 — 실시간 미터.

    .venv/bin/python scripts/data/set_amp_level.py --self-test        # 소리 없음
    .venv/bin/python scripts/data/set_amp_level.py --confirm-speaker  # 소리 20초

왜 이 도구가 있는가
------------------
2026-08-06: 앰프 볼륨을 눈감고 맞추다가 하루를 날렸다. 이 시스템은 **구동 레벨에
창(window)이 있고 그 창이 좁다.** 실측(분석 구간 ERR 대역 RMS 기준):

    -68.9 dBFS  →  신호가 잡음 바닥(-69)에 묻혀 P/S 측정이 전량 기각
    -48.3 dBFS  →  **정상. P−S = 140 (기하 예측 147과 일치)**   ← 목표
    -37   dBFS  →  P−S = 1 로 붕괴. 두 채널이 결합된다

위쪽이 왜 깨지는가: TPA3116D2 가 Jetson USB-C 에서 어댑터로 전원을 받는다. 이 앰프는
12~24V 용인데 5V 로는 부족해서, 세게 구동하면 전원이 주저앉고 **공유 전원·접지를 통해
두 채널이 결합**한다. 그 상태에서는 P(소음경로)와 S(취소경로)를 구분할 수 없고,
ANC 는 원리적으로 성립하지 않는다 (안티노이즈가 소음 스피커에서도 나온다).

즉 이 레벨은 취향이 아니라 **측정이 성립하는 조건**이다. 그래서 숫자로 맞춘다.

무엇을 재는가
------------
``measure_paths_interleaved.py`` 와 **똑같은 인터리브 멀티톤**의 소음 채널 성분을
ch0 으로만 흘리고, ERR 마이크의 대역 RMS 를 0.25초마다 출력한다.

프로브를 같게 두는 것이 핵심이다 — 자체 신호를 만들면 눈금이 그 측정과 어긋난다.
실제로 밴드 노이즈(크레스트 12dB)로 만들었다가 멀티톤(5.6dB) 대비 6dB 넘게 어긋났다.

⚠ 취소 스피커(ch1)는 무음으로 둔다. 두 채널을 동시에 울리면 결합이 있을 때 레벨이
   부풀어 보여서 맞출 수가 없다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.config import load_yaml  # noqa: E402

FS = 48000
BAND = (150.0, 1600.0)

# 목표 ERR 대역 RMS (dBFS).
#
# 이 값은 **P−S = 140 이 나온 2026-08-04 캡처의 분석 구간 ERR RMS** 다.
# 같은 통계·같은 프로브로 재야 눈금이 맞으므로, 아래 probe_signal() 은
# measure_paths_interleaved.py 와 **동일한 인터리브 멀티톤**을 쓴다.
#
#   8/4 225546 (P−S=140)  분석구간 RMS  -48.32 dBFS   ← 목표
#   8/4 235822 (P−S=140)                -48.45 dBFS
#   8/6 볼륨↓  (측정 실패)               -68.90 dBFS   ← 잡음 바닥(-69)에 묻힘
#   8/6 볼륨↑  (P−S=1)                   약 -37 dBFS   ← +11.3 dB 과다, 채널 결합
#
# ⚠ 프로브를 바꾸면 이 숫자도 다시 잡아야 한다. 밴드 노이즈(크레스트 12dB)와
#   멀티톤(5.6dB)은 같은 peak 에서 RMS 가 6dB 넘게 다르다 — 실제로 그렇게 한 번 틀렸다.
TARGET_DBFS = -48.3
TOLERANCE_DB = 2.0

# 프로브 진폭 — 인터리브 측정의 기본값과 같아야 한다(cw.MAX_AMPLITUDE).
AMPLITUDE = 0.02


def band_rms_dbfs(x: np.ndarray) -> float:
    """대역 안 RMS(dBFS). 대역 밖 험·DC 가 눈금을 흔들지 않게 대역제한한다."""

    n = int(x.size)
    if n < 256:
        return -120.0
    X = np.fft.rfft(x * np.hanning(n))
    f = np.fft.rfftfreq(n, 1.0 / FS)
    m = (f >= BAND[0]) & (f <= BAND[1])
    # Parseval: 창 보정 포함
    power = 2.0 * np.sum(np.abs(X[m]) ** 2) / (n * np.sum(np.hanning(n) ** 2))
    return 10.0 * np.log10(max(power, 1e-24))


def verdict(level: float) -> str:
    if level < TARGET_DBFS - TOLERANCE_DB:
        return f"↑ 올리세요 ({TARGET_DBFS - level:+.1f} dB 부족)"
    if level > TARGET_DBFS + TOLERANCE_DB:
        return f"↓ 내리세요 ({level - TARGET_DBFS:+.1f} dB 초과)"
    return "✅ 맞았습니다 — 여기서 멈추세요"


def probe_signal(seconds: float) -> np.ndarray:
    """``measure_paths_interleaved.py`` 와 **동일한** 자극을 반복해 만든다.

    여기서 자체 신호를 만들면 눈금이 그 측정과 어긋난다 — 크레스트가 다르면 같은 peak
    에서 RMS 가 6dB 넘게 갈린다. 이 저장소를 무너뜨린 발생기 A(같은 물리량을 두 곳에서
    따로 유도)를 계측 도구에서 반복하지 않는다.

    소음 채널(ch0) 성분만 쓴다. 취소 채널까지 함께 울리면 결합이 있을 때 레벨이 부풀어
    보여서 노브를 맞출 수가 없다.
    """

    from deep_anc.dsp.interleaved_probe import build_interleaved_probe

    probe = build_interleaved_probe(
        sample_rate=FS, period_seconds=0.125, band_hz=(60.0, 1650.0),
        amplitude=AMPLITUDE, tone_spacing_hz=None,
    )
    period = np.asarray(probe.noise_signal, dtype=np.float32).reshape(-1)
    repeats = int(np.ceil(seconds * FS / period.size))
    return np.tile(period, repeats)[: int(seconds * FS)].astype(np.float32)


def self_test() -> int:
    """미터와 판정을 소리 없이 검증한다."""

    print("[self-test] 알려진 레벨을 주입해 미터를 검증한다")
    ok = True
    for target in (-60.0, -44.0, -33.0):
        n = FS
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n)
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(n, 1.0 / FS)
        X[(f < BAND[0]) | (f > BAND[1])] = 0.0
        x = np.fft.irfft(X, n)
        x *= 10 ** (target / 20.0) / np.sqrt(np.mean(x**2))
        got = band_rms_dbfs(x)
        good = abs(got - target) < 0.5
        ok &= good
        print(f"  [{'OK' if good else 'FAIL'}] 주입 {target:+7.1f} → 측정 {got:+7.1f} dBFS | {verdict(got)}")
    # 대역 밖 험이 눈금을 흔들지 않는지
    n = FS
    t = np.arange(n) / FS
    hum = 0.05 * np.sin(2 * np.pi * 60.0 * t)
    got = band_rms_dbfs(hum)
    good = got < -40.0
    ok &= good
    print(f"  [{'OK' if good else 'FAIL'}] 60Hz 험 진폭 0.05 → 대역내 {got:+7.1f} dBFS (대역 밖이라 낮아야 한다)")
    print(f"[self-test] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def measure(args) -> int:
    import sounddevice as sd

    from deep_anc.audio_io import pcm_int32_to_float32, resolve_alsa_portaudio_device

    hardware = load_yaml(REPO_ROOT / args.hardware)["audio"]
    in_dev = resolve_alsa_portaudio_device(
        hardware["input"]["card"], hardware["input"]["pcm"], "input", 2
    )
    out_dev = resolve_alsa_portaudio_device(
        hardware["output"]["card"], hardware["output"]["pcm"], "output", 2
    )
    err_ch = int(hardware.get("channels", {}).get("error_mic", 0)) if isinstance(
        hardware.get("channels"), dict
    ) else 0

    noise = probe_signal(args.seconds)

    print(f"목표 {TARGET_DBFS:+.1f} dBFS (허용 ±{TOLERANCE_DB:.0f} dB) · "
          f"프로브 peak {AMPLITUDE:.3f} · ch0(소음 스피커)만 재생")
    print(f"{args.seconds:.0f}초 동안 0.25초마다 표시합니다. 숫자를 보며 노브를 돌리세요.")
    print("Ctrl-C 로 언제든 중단할 수 있습니다.\n", flush=True)

    # 콜백으로 **실제 전달된 블록**만 잰다.
    # sd.playrec 로 미리 만든 배열을 훑으면, 아직 안 채워진 구간을 읽고 조용히 건너뛰어
    # 화면에 아무것도 안 나온다 — 사용자에게는 멈춘 것으로 보인다(2026-08-06 실제 발생).
    import queue as _queue
    import time

    hop = int(0.25 * FS)
    cursor = {"out": 0}
    meter: "_queue.Queue[float]" = _queue.Queue(maxsize=64)
    pending = {"buf": np.zeros(0, dtype=np.float64)}

    def callback(indata, outdata, frames, _time, status):  # noqa: ANN001
        start = cursor["out"]
        end = min(start + frames, noise.size)
        take = end - start
        outdata[:] = 0.0
        if take > 0:
            outdata[:take, 0] = noise[start:end]
        cursor["out"] = end
        if take < frames:
            raise sd.CallbackStop
        buf = np.concatenate([pending["buf"], indata[:, err_ch].astype(np.float64)])
        while buf.size >= hop:
            try:
                meter.put_nowait(band_rms_dbfs(buf[:hop]))
            except _queue.Full:
                pass
            buf = buf[hop:]
        pending["buf"] = buf

    levels: list[float] = []
    stream = sd.Stream(
        samplerate=FS, blocksize=1024, dtype="float32", channels=2,
        device=(in_dev, out_dev), callback=callback,
    )
    deadline = args.seconds + 3.0
    try:
        with stream:
            started = time.monotonic()
            while stream.active and time.monotonic() - started < deadline:
                try:
                    level = meter.get(timeout=0.5)
                except _queue.Empty:
                    continue
                if not levels and level < -100.0:
                    continue  # 스트림 기동 직후의 빈 블록
                levels.append(level)
                filled = max(0, min(40, int((level + 70.0))))
                bar = "█" * filled
                print(f"  {level:+7.1f} dBFS  {bar:<40} {verdict(level)}", flush=True)
    except KeyboardInterrupt:
        print("\n중단했습니다.", flush=True)

    if not levels:
        print("\n[실패] 레벨을 읽지 못했습니다 — 입력 장치를 확인하세요.", file=sys.stderr)
        return 1
    final = float(np.median(levels[-8:] if len(levels) >= 8 else levels))
    print(f"\n마지막 구간 중앙값 {final:+.1f} dBFS — {verdict(final)}")
    if abs(final - TARGET_DBFS) <= TOLERANCE_DB:
        print("다음: measure_paths_interleaved.py 로 P−S 를 확인하세요 (6초)")
        return 0
    print("목표 범위 밖입니다. 노브를 조정하고 다시 실행하세요.", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--self-test", action="store_true", help="소리 없이 미터만 검증")
    parser.add_argument("--confirm-speaker", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.confirm_speaker:
        print(
            f"스피커에서 {args.seconds:.0f}초 동안 소리가 납니다. "
            "--confirm-speaker 를 붙여 실행하세요.",
            file=sys.stderr,
        )
        return 2
    return measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
