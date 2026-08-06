#!/usr/bin/env python3
"""덕트 실측 수집 — 소음 재생(ch0) + 레퍼런스/에러 마이크 동시 녹음 (ANC OFF).

  .venv/bin/python scripts/data/record_duct.py --program tone --frequency 300 --seconds 60
  .venv/bin/python scripts/data/record_duct.py --program band --seconds 120
  .venv/bin/python scripts/data/record_duct.py --program silence --seconds 30   # 암소음 측정

저장: data/recorded/<타임스탬프_프로그램>/
      {mics.wav(2ch PCM_32), source.wav(원본 provenance), source_aligned.wav(학습용), session.json}
시작 시 레퍼런스 마이크(ch1) 자가진단 — 과거 무신호 이력 대응 (docs/02).
상쇄 스피커(ch1 출력)는 전 구간 무음을 유지한다.

시간축 규약 (2026-08-05 결함 2 수정)
-----------------------------------
이 스크립트는 예전에 ``cursor["in"] == cursor["out"]`` 이라는 이유만으로
``source[t]`` 와 ``mics[t]`` 가 **같은 물리 시각**이라고 단언했다. 그것은 DAC→ADC
왕복지연이 0 이라는 가정이고, 실제로는 0 도 아니고 상수도 아니다 — AB13X USB DAC 이
UAC1 ADAPTIVE(full speed, 피드백 엔드포인트 없음)라 장치 PLL 이 주기 4~5 초, 진폭
259~407 샘플로 헌팅한다. 그 결과 80 세션 전부가 ``coh²(source→ERR)=0.02~0.13`` 인
채로 QA 를 통과했다.

지금은 **단언하지 않고 측정한다**:

* ``source.wav`` 는 재생한 그대로 남긴다 (원본 provenance — 절대 덮어쓰지 않는다).
* REF 마이크(ch1)는 ERR 과 **같은 ADC** 를 타므로 재생 신호의 시간축 증인이다.
  이 증인으로 시변 지연 L(t) 를 추정해 ``source_aligned.wav`` 를 만든다.
* 검증은 추정에 쓰지 않은 ERR 채널로 한다(홀드아웃). 기준 미달이면 **저장하지 않는다.**
* PortAudio 콜백 타임스탬프는 provenance 로만 남긴다 — 실측상 ``dac−adc`` 가
  0.010/0.020 s 두 값 사이를 16 샘플 단위로만 튀어 실제 ±130 샘플 변조를 전혀
  보여주지 않는다. **진단용이지 수정 수단이 아니다.**
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.audio_io import (                              # noqa: E402
    pcm_int32_to_float32,
    resolve_alsa_portaudio_device,
    rms_dbfs,
)
from deep_anc.config import REPO_ROOT, load_yaml             # noqa: E402
from deep_anc.data.manifest import (                         # noqa: E402
    validate_group_id,
    validate_session_id,
    validate_source_family,
)
from deep_anc.data.timeline import (                         # noqa: E402
    TimelineSettings,
    align_source_to_adc,
)
from deep_anc.audio_io import (                             # noqa: E402
    MAX_PROBE_CLIP_RATIO,
    input_rail_gate,
)
from deep_anc.realtime.noise_gen import NoiseProgram         # noqa: E402

# 게이트 하한. CLI 는 이 값 **이상**만 받는다(강화 전용).
# 0.90 의 근거: 선형 Wiener 하한 10·log10(1−coh²) 로 −10 dB. 재정렬이 성공한 실측
# 세션은 0.87~0.96 이고 붕괴 세션은 0.02~0.13 이라 그 사이 골짜기가 넓다.
DEFAULT_MIN_TIMELINE_COHERENCE = 0.90
DEFAULT_MIN_VALID_WINDOW_RATIO = 0.90
# 자가진단 상한. QA 의 max_clip_ratio(0.005)와 같은 자리에서 판정하되, 재생 전에 본다.
# 레일 게이트와 임계는 src/deep_anc/audio_io.py 가 단일 출처다.
# 여기 두면 다른 도구가 sys.path 를 조작해 스크립트에서 import 해야 하고,
# 실제로 그 불편함이 "새 도구는 그냥 안 쓴다" 로 이어졌다(2026-08-06).


def timeline_gate(report, *, min_coherence: float, min_valid_window_ratio: float) -> bool:
    """gate: ``recording_timeline_fail_closed`` — 재정렬이 실제로 성공했는가.

    ``coh²(source_aligned→ERR, 150-600Hz)`` 와 유효창 비율을 **저장 전에** 본다.
    실패하면 세션을 쓰지 않는다. 이 판정을 저장 뒤로 미루면 결함 2 가 그대로 재발한다 —
    80 세션이 정확히 그렇게 만들어졌다.
    """

    return bool(
        report.coh2_150_600_after >= float(min_coherence)
        and report.valid_window_ratio >= float(min_valid_window_ratio)
    )


def _summarise_io_timestamps(stamps: np.ndarray, fs: int) -> dict:
    """PortAudio 콜백 타임스탬프 요약 — **provenance 전용, 수정 수단 아님.**

    실측(무음 40초 전이중 프로브, record_duct 와 동일 스트림 설정):
    콜백 7500회 / 프레임 1,920,000 / status 이벤트 **0회**, adc rate +5.0 ppm(잔차
    0.018 ms). 그런데 ``dac − adc`` 는 0.010 s 와 0.020 s 두 값 사이를 16 샘플 단위로만
    튄다. 실제로 일어나고 있는 ±130 샘플 변조를 **전혀 보여주지 않는다** — 이 값들은
    호스트 시계에서 나온 예측치이지 장치가 실제로 소리를 낸 시각이 아니기 때문이다.

    그래서 여기에 남기는 것은 "그때 호스트가 뭐라고 믿었는가" 뿐이고, 정렬은 REF 증인
    재정렬이 담당한다. 이 구분을 흐리면 결함 2 가 그대로 재발한다.
    """

    if stamps.size == 0:
        return {"callbacks": 0, "note": "타임스탬프 없음"}
    adc = stamps[:, 0]
    dac = stamps[:, 1]
    frames = stamps[:, 2]
    elapsed = np.cumsum(np.concatenate([[0.0], frames[:-1]])) / float(fs)
    summary: dict = {
        "callbacks": int(stamps.shape[0]),
        "frames_total": int(frames.sum()),
        "unique_frames": sorted({int(value) for value in frames}),
        "dac_minus_adc_s": {
            "min": float(np.min(dac - adc)),
            "median": float(np.median(dac - adc)),
            "max": float(np.max(dac - adc)),
            "unique_values": int(np.unique(np.round(dac - adc, 6)).size),
        },
        "note": (
            "provenance 전용. 실측상 dac−adc 는 16샘플 단위 계단이라 실제 DAC 헌팅"
            "(4~5초 주기, 259~407샘플)을 보여주지 못한다. 정렬 복원에 쓰지 말 것."
        ),
    }
    if stamps.shape[0] >= 8:
        for name, series in (("adc", adc), ("dac", dac)):
            slope, intercept = np.polyfit(elapsed, series - series[0], 1)
            residual = (series - series[0]) - (slope * elapsed + intercept)
            summary[f"{name}_rate_ppm"] = float((slope - 1.0) * 1.0e6)
            summary[f"{name}_residual_rms_ms"] = float(np.sqrt(np.mean(residual**2)) * 1000.0)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument(
        "--program",
        default="tone",
        choices=["tone", "multitone", "white", "band", "nonlinear", "sweep", "file", "silence"],
    )
    parser.add_argument("--frequency", type=float, default=300.0)
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--band", type=float, nargs=2, default=[80.0, 1000.0])
    parser.add_argument("--file", default=None, help="program=file 재생 wav")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--out-root", default="data/recorded")
    parser.add_argument(
        "--source-family",
        default=None,
        help=(
            "소스 계열 ID(예: speech/music/environment). 생략 시 program 이름을 사용하며, "
            "program=file은 명시를 권장"
        ),
    )
    parser.add_argument(
        "--group-id",
        default=None,
        help=(
            "분할 누수를 막을 상관 그룹 ID(같은 화자/곡/원본/환경의 반복 세션은 같은 값). "
            "생략 시 현재 세션만의 ID 사용"
        ),
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help=(
            "스트림을 연 뒤 무음으로 흘려보내고 버릴 길이. I2S 기동 트랜지언트가 "
            "약 0.5초 지속되므로 여유를 둔 1.0초가 기본값"
        ),
    )
    parser.add_argument("--force", action="store_true", help="ref 마이크 무신호여도 진행")
    parser.add_argument("--ref-check-dbfs", type=float, default=-80.0)
    parser.add_argument(
        "--min-timeline-coherence",
        type=float,
        default=0.90,
        help=(
            "재정렬 후 coh²(source_aligned→ERR, 150-600Hz) 하한. 이 값 미만이면 세션을 "
            "저장하지 않는다(실패-폐쇄). 강화(올리기)만 허용된다"
        ),
    )
    parser.add_argument(
        "--min-valid-window-ratio",
        type=float,
        default=0.90,
        help="지연 추정에 성공한 창의 비율 하한. 강화만 허용된다",
    )
    args = parser.parse_args()

    # 게이트 인자는 **강화 방향으로만** 열려 있다. 완화 값을 주면 그 자리에서 거부한다 —
    # 게이트를 인자로 풀 수 있으면 그것은 게이트가 아니라 제안이다.
    if args.min_timeline_coherence < DEFAULT_MIN_TIMELINE_COHERENCE:
        parser.error(
            f"--min-timeline-coherence 는 {DEFAULT_MIN_TIMELINE_COHERENCE} 이상이어야 "
            f"합니다 (받은 값 {args.min_timeline_coherence}) — 게이트는 강화만 합니다"
        )
    if args.min_valid_window_ratio < DEFAULT_MIN_VALID_WINDOW_RATIO:
        parser.error(
            f"--min-valid-window-ratio 는 {DEFAULT_MIN_VALID_WINDOW_RATIO} 이상이어야 "
            f"합니다 (받은 값 {args.min_valid_window_ratio}) — 게이트는 강화만 합니다"
        )

    try:
        source_family = validate_source_family(args.source_family or args.program)
        requested_group_id = (
            validate_group_id(args.group_id) if args.group_id is not None else None
        )
    except ValueError as exc:
        parser.error(str(exc))

    import sounddevice as sd

    hw = load_yaml(REPO_ROOT / args.hardware)["audio"]
    fs = int(hw["sample_rate"])
    block = int(hw["block_size"])

    in_dev = resolve_alsa_portaudio_device(hw["input"]["card"], hw["input"]["pcm"], "input", 2)
    out_dev = resolve_alsa_portaudio_device(hw["output"]["card"], hw["output"]["pcm"], "output", 2)

    # ----- 1) 레퍼런스 마이크 자가진단 (2초 무음 캡처) -----
    print("레퍼런스 마이크 점검 중 (2초)...")
    # 앞 1초는 기동 트랜지언트라 버린다. 이걸 포함해서 재면 무신호 마이크도 -42dBFS 로
    # 보여 "살아 있다"고 오판한다 — 이 점검의 목적을 정확히 무력화한다.
    probe_settle = int(1.0 * fs)
    probe = sd.rec(
        probe_settle + int(2 * fs), samplerate=fs, channels=2, dtype="int32", device=in_dev
    )
    sd.wait()
    probe_f = pcm_int32_to_float32(probe[probe_settle:])
    err_db = rms_dbfs(probe_f[:, 0])
    ref_db = rms_dbfs(probe_f[:, 1])
    rail_ok, clip_ratio = input_rail_gate(probe_f)
    print(
        f"  ch0(err) {err_db:7.2f} dBFS | ch1(ref) {ref_db:7.2f} dBFS | "
        f"레일 비율 {clip_ratio[0]:.4f}/{clip_ratio[1]:.4f}"
    )
    if ref_db < args.ref_check_dbfs and not args.force:
        print(
            f"[중단] 레퍼런스 마이크(ch1)가 무신호로 보입니다 ({ref_db:.1f} dBFS < "
            f"{args.ref_check_dbfs}). 배선 점검(docs/02_hardware_setup.md) 후 재시도하거나 "
            "--force 로 강행하세요.", file=sys.stderr,
        )
        return 1
    # 판정 근거는 input_rail_gate() 의 docstring 에 있다. 재생 **전에** 막는다.
    if not args.force and not rail_ok:
        print(
            f"[중단] 마이크 입력이 풀스케일에 붙어 있습니다 (레일 비율 "
            f"ch0 {clip_ratio[0]:.4f} / ch1 {clip_ratio[1]:.4f} > {MAX_PROBE_CLIP_RATIO}). "
            "입력단 전원/배선을 확인하세요 — 이 상태의 녹음은 클리핑으로 전량 폐기됩니다. "
            "스피커를 울리기 전에 멈춥니다.",
            file=sys.stderr,
        )
        return 1

    # ----- 2) 프로그램 준비 -----
    prog_cfg = {
        "type": args.program,
        "frequency": args.frequency,
        "amplitude": args.amplitude,
        "band": args.band,
        "file": args.file,
    }
    program = NoiseProgram(prog_cfg, fs)

    # I2S 입력은 스트림을 연 직후 약 0.5초 동안 큰 기동 트랜지언트를 낸다
    # (실측: 0.0-0.5초 -36.3 dBFS peak 0.062 → 0.5초 이후 -67.4 dBFS peak 0.002).
    # 이 구간을 세션에 남기면 (a) 학습 데이터 앞머리가 잡음이 되고 (b) 세션 QA 의
    # peak/RMS 통계가 트랜지언트를 재게 된다. 무음으로 흘려보내고 잘라낸다.
    # 출력과 입력을 같은 길이만큼 버리므로 정렬은 유지된다.
    settle = int(max(0.0, args.settle_seconds) * fs)
    keep = int(args.seconds * fs)
    total = keep + settle
    source = np.zeros(total, dtype=np.float32)
    recorded = np.zeros((total, 2), dtype=np.float32)
    cursor = {"in": 0, "out": 0}
    xrun_state: dict = {"count": 0, "flags": set()}

    fade = np.linspace(0.0, 1.0, int(0.1 * fs), dtype=np.float32)
    # 재생 진폭 포락선을 미리 만들어 둔다. 예전에는 콜백 안에서 `for k in range(frames)`
    # 로 샘플마다 파이썬 분기를 돌았다 — 블록 256 샘플에 파이썬 루프 256회를 5.33 ms
    # 마감 안에서 하는 것은 그 자체가 xrun 위험이다. 여기서 한 번 만들고 콜백은
    # 슬라이스 곱셈만 한다.
    envelope = np.zeros(total + 8 * block, dtype=np.float32)
    envelope[settle : settle + keep] = 1.0
    ramp = min(fade.size, keep // 2)
    if ramp > 0:
        envelope[settle : settle + ramp] = fade[:ramp]
        envelope[settle + keep - ramp : settle + keep] = fade[:ramp][::-1]

    # 콜백 타임스탬프는 **provenance 전용**이다. 실측(무음 40초 프로브): status 0회,
    # adc/cur rate +5.0 ppm, 그런데 dac−adc 는 0.010/0.020 s 두 값 사이를 16 샘플
    # 단위로만 튄다 → 실제 ±130 샘플 변조를 전혀 보여주지 않는다. 이 값으로 정렬을
    # 고치려 들면 안 된다. 정렬은 REF 증인으로 사후 추정한다.
    max_callbacks = total // max(1, block) + 64
    stamps = np.zeros((max_callbacks, 3), dtype=np.float64)
    stamp_count = {"n": 0}

    def callback(indata, outdata, frames, time_info, status):
        if status:
            # 콜백 안에서 print 하면 그 자체가 다음 xrun 을 만든다. 세어만 두고 밖에서 판정한다.
            # xrun 은 source 와 mics 사이에 **영구 오프셋**을 남긴다 — 커서는 frames 만큼
            # 계속 전진하므로 드롭된 블록만큼 두 배열이 세션 끝까지 어긋난다.
            # ⚠ 이 가드는 **필요조건일 뿐 충분조건이 아니다.** status==0 이어도 시간축은
            #   깨진다: 무음 40초 프로브에서 status 0회였고, PortAudio 를 완전히 배제한
            #   aplay+arecord 직결 경로에서도 4~5초 주기 5.4~8.5 ms 변조가 같은 파형으로
            #   재현됐다. 그래서 저장 시점의 REF 증인 재정렬이 진짜 판정이다.
            xrun_state["count"] += 1
            xrun_state["flags"].add(str(status))
        idx = stamp_count["n"]
        if idx < max_callbacks:
            stamps[idx, 0] = time_info.inputBufferAdcTime
            stamps[idx, 1] = time_info.outputBufferDacTime
            stamps[idx, 2] = float(frames)
            stamp_count["n"] = idx + 1

        i = cursor["in"]
        n = min(frames, total - i)
        recorded[i : i + n] = pcm_int32_to_float32(indata[:n, :2])
        cursor["in"] = i + n

        o = cursor["out"]
        blk = program.generate(frames)
        # settle 구간은 무음으로 흘린다 — 프로그램 위상은 그대로 진행시켜 잘라낸 뒤에도
        # source 배열과 재생 샘플이 같은 인덱스를 가리키게 한다.
        # ⚠ 이 인덱스 동일성은 "재생 배열 안에서" 만 유효하다. 재생 샘플이 **언제
        #   공기 중으로 나갔는가**는 여기서 알 수 없다 — 그것이 결함 2 의 본체였다.
        blk *= envelope[o : o + frames]
        m = min(frames, total - o)
        source[o : o + m] = blk[:m]
        out = np.zeros((frames, 2), dtype=np.float32)
        out[:, 0] = blk                             # ch0 = 소음 스피커
        # ch1(상쇄 스피커)은 무음 유지
        outdata[:] = np.rint(np.clip(out, -1, 1) * 32767).astype(np.int16)
        cursor["out"] = o + m
        if cursor["in"] >= total:
            raise sd.CallbackStop

    print(f"녹음 시작: {args.program}, {args.seconds:.0f}초 (ANC 없음, ch1 무음)")
    with sd.Stream(
        samplerate=fs,
        blocksize=block,
        device=(in_dev, out_dev),
        channels=(2, 2),
        dtype=("int32", "int16"),
        latency=("low", "low"),
        callback=callback,
        prime_output_buffers_using_stream_callback=True,
    ):
        while cursor["in"] < total:
            time.sleep(0.1)

    # ----- 3) 저장 -----
    # xrun 이 하나라도 있으면 source↔mics 정렬이 깨졌다. 전달맵은 이미 xrun 을 무효화
    # 사유로 쓰는데(measure_duct_transfer_map) 학습데이터 수집기만 기준이 느슨했다.
    if xrun_state["count"] > 0:
        print(
            f"[중단] 오디오 xrun {xrun_state['count']}회 ({', '.join(sorted(xrun_state['flags']))}) — "
            "source 와 mics 의 정렬이 깨져 학습에 쓸 수 없습니다. 세션을 저장하지 않습니다.",
            file=sys.stderr,
        )
        return 1

    # ----- 3a) 시간축 재정렬 (저장 **전에** 판정한다) -----
    # settle 구간을 양쪽에서 동일하게 잘라낸다. 이 자르기가 보존하는 것은 인덱스
    # 동일성일 뿐 물리 시각 동일성이 아니다 — 그래서 바로 아래에서 실제로 측정한다.
    mics_keep = recorded[settle:]
    source_keep = source[settle:]

    silent_program = args.program == "silence" or source_family == "silence"
    timeline_meta: dict = {}
    aligned = None
    if silent_program:
        # 무음 세션은 재생 신호가 없어 L(t) 를 추정할 수 없다. 추정할 수 없다는 사실을
        # 그대로 기록한다 — "검사하지 않음"을 "통과"로 적으면 그게 결함 2 의 재발이다.
        timeline_meta = {
            "method": "skipped_silent_program",
            "usable_for_digital_reference": False,
            "reason": "무음 프로그램은 재생↔녹음 대응을 측정할 수 없습니다",
        }
        print("[안내] 무음 세션이라 시간축 재정렬을 건너뜁니다 (digital-ref 학습에 쓸 수 없음)")
    else:
        print("시간축 재정렬 중 (REF 마이크를 증인으로 사용)...")
        aligned, report = align_source_to_adc(
            source_keep,
            mics_keep[:, 1],   # 증인 = REF (추정 전용)
            mics_keep[:, 0],   # 홀드아웃 = ERR (검증 전용)
            fs,
            settings=TimelineSettings(sample_rate=fs),
        )
        timeline_meta = report.as_metadata()
        timeline_meta["usable_for_digital_reference"] = True
        print(
            f"  coh²(source→ERR,150-600Hz) {report.coh2_150_600_before:.3f} → "
            f"{report.coh2_150_600_after:.3f} | 600-1600Hz "
            f"{report.coh2_600_1600_before:.3f} → {report.coh2_600_1600_after:.3f}"
        )
        print(
            f"  유효창 {report.valid_window_ratio:.3f} | 원시지연 중앙 "
            f"{report.raw_lag_median_samples:.1f} ptp {report.raw_lag_ptp_samples:.1f} | "
            f"잔여지연 중앙 {report.aligned_lag_median_samples:.2f} "
            f"robust-std {report.aligned_lag_robust_std_samples:.2f} "
            f"p95-p5 {report.aligned_lag_p95_p5_samples:.2f}"
        )
        if not timeline_gate(
            report,
            min_coherence=args.min_timeline_coherence,
            min_valid_window_ratio=args.min_valid_window_ratio,
        ):
            print(
                "[중단] 시간축 재정렬 실패 — 세션을 저장하지 않습니다 "
                f"(coh² {report.coh2_150_600_after:.3f} < {args.min_timeline_coherence}, "
                f"유효창 {report.valid_window_ratio:.3f} < {args.min_valid_window_ratio}). "
                f"음향 대조군 coh²(REF→ERR)={report.coh2_ref_err_150_600:.3f} — 이 값이 "
                "높으면 배선/음향이 아니라 재생-캡처 타임베이스 문제입니다.",
                file=sys.stderr,
            )
            return 1

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = REPO_ROOT / args.out_root / f"{stamp}_{args.program}"
    session_dir.mkdir(parents=True, exist_ok=True)
    sf.write(session_dir / "mics.wav", mics_keep, fs, subtype="PCM_32")
    # source.wav 는 **원본 provenance** 다. 절대 재정렬본으로 덮어쓰지 마라 —
    # 덮어쓰면 워프 알고리즘을 고쳤을 때 되돌릴 방법이 없다.
    sf.write(session_dir / "source.wav", source_keep, fs, subtype="FLOAT")
    if aligned is not None:
        sf.write(session_dir / "source_aligned.wav", aligned, fs, subtype="FLOAT")
    session_id = validate_session_id(session_dir.name)
    group_id = requested_group_id or validate_group_id(session_id)
    meta = {
        "session_id": session_id,
        "program": prog_cfg,
        "source_family": source_family,
        "group_id": group_id,
        "seconds": args.seconds,
        "sample_rate": fs,
        "block_size": block,
        "channels": {"err_mic": 0, "ref_mic": 1, "noise_out": 0, "cancel_out": 1},
        "ref_check_dbfs": {"err": err_db, "ref": ref_db},
        "timestamp": stamp,
        "timeline": timeline_meta,
        "io_timestamps": _summarise_io_timestamps(stamps[: stamp_count["n"]], fs),
    }
    (session_dir / "session.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"저장 완료: {session_dir}")
    print("다음: .venv/bin/python scripts/data/make_recorded_manifest.py 로 manifest 갱신")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
