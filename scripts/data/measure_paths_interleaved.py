#!/usr/bin/env python3
"""P(z)와 S(z)를 **한 번의 재생으로 동시에** 측정한다.

왜 순차 ESS 로는 안 되는가
--------------------------
재생은 USB(AB13X), 녹음은 Tegra APE I²S 다. 클록 도메인이 서로 달라 "출력 샘플 번호 ↔
녹음 샘플 번호" 대응이 시간에 따라 흔들린다(wander). 저장된 측정 4건을 재분석한 결과가
이를 못박는다 — 재생 프로그램 기준 반복 간 coherence 0.08~0.17, 같은 녹음을 ERR/REF
기준으로 보면 0.9915~0.9976, |H| 반복 std 0.08dB. 즉 **깨진 것은 시간축 대응 하나뿐**이다.
자극 진폭을 4배 올려도 개선이 없었다는 사실이 레벨 가설을 직접 반증한다.

``calibrate_wideband.py`` 는 P 와 S 를 별도 실행으로 잰다. 두 측정이 수십 초 떨어지면
그 사이의 wander 가 **두 경로의 상대 지연**에 그대로 실린다. ANC 가 실제로 요구하는 양이
바로 그 상대 지연(``lead = S_delay + handoff − P_delay``)이므로, 순차 측정은 우리가 가장
필요로 하는 숫자를 가장 크게 틀린다.

해법
----
두 출력 채널은 **같은 DAC·같은 스트림**을 지나므로 warp D(t) 가 동일하다. 정확히 같은
시각에 두 경로를 구동하면 D 는 두 경로에 공통으로 실리고 상대 관계에서 상쇄된다.
동시 재생 상태로 두 응답을 분리하기 위해 주파수를 번갈아 나눈다(guard=1).

    ch0(소음 스피커) → 짝수번째 톤,  ch1(상쇄 스피커) → 홀수번째 톤

정수 주기 FFT 라 빈 집합이 정확히 서로소이고 누설이 0 이다. 시뮬레이션 검증 결과
(``tests/test_interleaved_probe.py``) 실측 wander 3.2샘플에서 상대오차 −26.9dB —
같은 조건의 순차 측정은 −4dB 다.

산출물
------
게이트를 통과하면 P/S NPZ 두 개를 **같은 capture_id** 로 함께 저장한다. 같은 capture 에서
나왔다는 사실이 파일에 박혀 있어야 파인튜닝 진입 감사가 "두 경로가 같은 조건"임을
파일만 보고 확인할 수 있다(``finetune_readiness.audit_official_path_model``).

사용자 입회·앰프 볼륨 최저에서만 실행한다::

  .venv/bin/python scripts/data/measure_paths_interleaved.py --confirm-volume-minimum
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import calibrate_wideband as cw  # noqa: E402

from deep_anc.audio_io import (  # noqa: E402
    pcm_int32_to_float32,
    resolve_alsa_portaudio_device,
)
from deep_anc.config import REPO_ROOT, load_yaml  # noqa: E402
from deep_anc.dsp.interleaved_probe import (  # noqa: E402
    DEFAULT_TRACK_WINDOW,
    align_repeats,
    build_interleaved_probe,
    channel_impulse_response,
    complex_consistency,
    dewarp_recording,
    estimate_transfer,
    tone_snr_db,
    track_warp,
)

METHOD = "interleaved_multitone"

# 설계 대역은 필수 대역 [80,1600] 보다 넓게 잡는다. 채널마다 톤이 한 칸씩 어긋나므로
# 딱 맞춰 잡으면 한 채널의 마지막 톤이 상한 안쪽으로 떨어져 대역을 덮지 못한다.
DEFAULT_BAND_HZ = (60.0, 1650.0)
# None = 주파수 분해능 그대로. 그래야 인접 빈이 서로 다른 채널이 되어 guard=1 이 된다.
# guard 를 넓히면 두 경로를 **서로 다른 주파수에서** 보게 되어 동시 측정의 이점을 깎는다.
DEFAULT_TONE_SPACING_HZ = None

# 분석 주기는 이 측정의 **결정적 파라미터**다. 재생↔녹음 대응이 주기 *안에서* 흔들리고,
# 그 위상오차는 2πfτ/fs 로 주파수에 비례한다. 창을 줄이면 창 안의 warp 가 줄어 고역이
# 살아난다. 2026-08-04 실측 스윕(진폭 0.06 고정, 반복 16):
#
#   주기 1.000s → 일관성 P 0.535 / S 0.535   (1000-1600Hz 0.368)
#   주기 0.250s →         P 0.793 / S 0.726  (          0.668)
#   주기 0.125s →         P 0.955 / S 0.925  (          0.887)   ← 게이트 0.90 통과
#
# 레벨은 원인이 아니다 — 같은 스윕에서 SNR 을 13.8→35.0dB 로 21dB 올려도 일관성은
# 개선되지 않았다(오히려 -0.14). PortAudio 도 아니다(ALSA 직접 경로 동일 증상).
DEFAULT_PERIOD_SECONDS = 0.125
DEFAULT_WARMUP_PERIODS = 4      # 순환 정상상태 도달 전 주기는 버린다
DEFAULT_REPEATS = 16
# τ 는 재현되는 대역에서만 적합한다. 재현 안 되는 대역을 넣으면 그 잡음이 τ 를 끌고 간다.
DEFAULT_FIT_BAND_HZ = (150.0, 1200.0)
# 일관성을 **어느 대역에서 쟀는지**가 곧 이 모델을 어느 대역에서 믿을 수 있는가다.
# 그래서 숫자만 저장하지 않고 대역도 함께 저장하고, 게이트가 그 대역이 요구 대역을
# 덮는지 검사한다. 대역이 안 적혀 있으면 0.95 라는 숫자가 무엇에 대한 0.95 인지 모른다.
DEFAULT_CONSISTENCY_BAND_HZ = (150.0, 600.0)

MIN_TONE_SNR_DB = 12.0          # 톤 중앙값 SNR 하한
MIN_TONE_SNR_FRACTION = 0.9     # 이 비율 이상의 톤이 하한을 넘어야 한다
MAX_CREST_DB = 14.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument("--band", type=float, nargs=2, default=list(DEFAULT_BAND_HZ))
    parser.add_argument("--required-band", type=float, nargs=2, default=[80.0, 1600.0])
    parser.add_argument(
        "--tone-spacing-hz",
        type=float,
        default=DEFAULT_TONE_SPACING_HZ,
        help="채널별 톤 간격(Hz). 생략하면 guard=1 이 되는 최소 간격을 쓴다",
    )
    parser.add_argument("--period-seconds", type=float, default=DEFAULT_PERIOD_SECONDS)
    parser.add_argument("--warmup-periods", type=int, default=DEFAULT_WARMUP_PERIODS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--amplitude", type=float, default=cw.MAX_AMPLITUDE)
    parser.add_argument("--fir-length", type=int, default=2048)
    parser.add_argument("--pre-roll", type=int, default=256)
    parser.add_argument("--max-delay-ms", type=float, default=100.0)
    parser.add_argument("--fit-band", type=float, nargs=2,
                        default=list(DEFAULT_FIT_BAND_HZ),
                        help="반복 정렬 τ 와 벌크 지연을 적합할 대역")
    parser.add_argument("--consistency-band", type=float, nargs=2,
                        default=list(DEFAULT_CONSISTENCY_BAND_HZ),
                        help="official coherence_median 을 계산할 대역(아티팩트에 함께 기록)")
    parser.add_argument(
        "--min-alignment-score", type=float, default=0.5,
        help="이 신뢰도 미만인 반복은 τ 탐색 실패로 보고 버린다(개수는 산출물에 기록)",
    )
    parser.add_argument("--min-kept-repeats", type=int, default=8)
    parser.add_argument("--max-delay-jitter-ms", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--latency", choices=["low", "high"], default="high")
    parser.add_argument("--input-probe-seconds", type=float, default=3.0)
    parser.add_argument("--primary-out", default="assets/measured/primary_path_il.npz")
    parser.add_argument("--secondary-out", default="assets/measured/secondary_path_il.npz")
    parser.add_argument("--diagnostics-root", default="results/calibration_interleaved")
    parser.add_argument(
        "--dewarp",
        action="store_true",
        help=(
            "주기 분석 전에 warp 궤적을 추적해 녹음을 재생 타임베이스로 되돌린다. "
            "실측에서 반복 일관성 0.05 → 0.85 (게이트 0.90 에는 아직 미달)"
        ),
    )
    parser.add_argument("--track-window", type=int, default=DEFAULT_TRACK_WINDOW)
    parser.add_argument("--track-min-peak", type=float, default=0.2)
    parser.add_argument("--confirm-volume-minimum", action="store_true")
    return parser


def bulk_delay_samples(
    frequencies_hz: np.ndarray,
    transfer: np.ndarray,
    *,
    sample_rate: int,
    band_hz: tuple[float, float],
    max_delay_samples: int,
) -> float:
    """위상 기울기에서 순수지연을 뽑는다 — 시간영역 온셋 검출을 쓰지 않는다.

    대역제한 IR 은 선행 링잉이 길어 에너지 온셋이 흔들린다. 그 흔들림이 그대로
    "지연 지터"로 보고돼 안정적인 측정도 게이트에서 떨어진다. 위상 기울기는 그 문제가
    없고, 짧은 분석 주기에서 IR 복원 주기가 절대 지연보다 짧아 감기는 문제도 피한다.

    구현은 정합 필터다 — ``|Σ_f H(f) e^{+j2πfτ/fs}|`` 를 최대화하는 τ. 위상 언랩보다
    잡음에 강하다(언랩은 한 번 튀면 그 뒤가 전부 어긋난다).
    """

    freq = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    values = np.asarray(transfer, dtype=np.complex128).reshape(-1)
    mask = (freq >= float(band_hz[0])) & (freq <= float(band_hz[1]))
    if int(mask.sum()) < 8:
        raise ValueError(f"지연 추정 대역 안의 톤이 부족합니다: {int(mask.sum())}개")
    taus = np.arange(0.0, float(max_delay_samples) + 0.25, 0.25)
    scores = np.abs(
        values[mask] @ np.exp(2j * np.pi * np.outer(taus, freq[mask]) / sample_rate).T
    )
    index = int(np.argmax(scores))
    if 0 < index < scores.size - 1:
        y0, y1, y2 = scores[index - 1], scores[index], scores[index + 1]
        denominator = y0 - 2.0 * y1 + y2
        fraction = 0.5 * (y0 - y2) / denominator if denominator != 0.0 else 0.0
    else:
        fraction = 0.0
    return float(taus[index] + fraction * 0.25)


def analyse_channel(
    *,
    err: np.ndarray,
    probe,
    drive: str,
    period_starts: list[int],
    fir_length: int,
    pre_roll: int,
    max_delay_samples: int,
    fit_band_hz: tuple[float, float],
    consistency_band_hz: tuple[float, float],
    min_alignment_score: float,
    min_kept_repeats: int,
) -> dict[str, Any]:
    """주기별 전달함수를 주파수영역에서 정렬·평균한 뒤 순수지연 + compact FIR 로 나눈다.

    반환에 ``taus`` 가 들어 있는 것이 핵심이다. 두 채널은 같은 스트림을 지나므로
    warp 가 공통으로 실린다 — 따라서 **τ 의 차이**(P − S)가 lead 가 의존하는 유일한
    양이고, 절대 τ 의 흔들림은 lead 에서 상쇄된다. 게이트가 판정해야 하는 것도 그 차이다.
    """

    rows = []
    for start in period_starts:
        segment = err[start : start + probe.period_samples]
        rows.append(estimate_transfer(segment, probe, drive=drive))
    frequencies = rows[0][0]
    stack = np.stack([H for _, H in rows])
    aligned, taus, scores = align_repeats(
        frequencies, stack, sample_rate=probe.sample_rate, fit_band_hz=fit_band_hz
    )
    # τ 탐색이 봉우리를 못 찾은 반복은 시간축 정보가 아니라 잡음이다. 기준(첫 반복)은
    # 자기 자신과의 상관이 1 이므로 항상 살아남는다. 판정 근거는 **정렬 신뢰도 하나**이며,
    # "결과가 좋아지는가"로 고르지 않는다 — 그건 게이트 우회다. 몇 개를 왜 버렸는지
    # 산출물에 남겨 검토자가 확인할 수 있게 한다.
    keep = scores >= float(min_alignment_score)
    if int(keep.sum()) < int(min_kept_repeats):
        raise ValueError(
            f"정렬에 성공한 반복이 {int(keep.sum())}개뿐입니다 "
            f"(최소 {int(min_kept_repeats)}, 신뢰도 하한 {min_alignment_score})"
        )
    aligned, taus_kept = aligned[keep], taus[keep]
    band_mask = (frequencies >= float(consistency_band_hz[0])) & (
        frequencies <= float(consistency_band_hz[1])
    )
    if int(band_mask.sum()) < 8:
        raise ValueError(
            f"일관성 대역 안의 톤이 부족합니다: {int(band_mask.sum())}개"
        )
    consistency = complex_consistency(aligned[:, band_mask])
    fullband_consistency = complex_consistency(aligned)
    mean_transfer = aligned.mean(axis=0)

    # 지연 탐색 범위는 **복원 주기 안으로 제한해야 한다.** 이 채널은 bin_step 마다
    # 하나씩만 빈을 가지므로 위상 램프가 period/bin_step 마다 되풀이된다. 범위를 그보다
    # 넓게 잡으면 정합 필터가 τ 와 τ+복원주기 를 구분하지 못한다 — 실측에서 S(z) 가
    # 1339 대신 4339(=1339+3000) 로 나왔고, 값이 그럴듯해 보여 조용히 틀릴 뻔했다.
    unambiguous = probe.period_samples // probe.bin_step(drive)
    delay = bulk_delay_samples(
        frequencies, mean_transfer, sample_rate=probe.sample_rate,
        band_hz=fit_band_hz,
        max_delay_samples=min(int(max_delay_samples), unambiguous - 1),
    )
    integer_delay = int(round(delay))
    if type(pre_roll) is not int or pre_roll < 0:
        raise ValueError("pre_roll은 비음수 정수여야 합니다")
    # pre-roll은 compact FIR의 시간 원점을 뒤로 옮긴다. 이를 별도
    # delay에서도 유지하면 같은 여유를 두 번 더하게 된다.
    # 음수 delay로 선행 샘플을 요구하는 모델은 저장하지 않는다.
    if pre_roll > integer_delay:
        raise ValueError(
            f"pre_roll {pre_roll} > 벌크 지연 {integer_delay}: "
            "비인과 compact 모델이 됩니다. 더 작은 pre-roll로 다시 분석하세요"
        )
    if type(fir_length) is not int or fir_length <= pre_roll:
        raise ValueError("fir_length는 pre_roll보다 큰 정수여야 합니다")
    # 벌크 지연을 빼면 남는 IR 이 짧아져 복원 주기 안에 안전하게 들어간다.
    residual = mean_transfer * np.exp(
        2j * np.pi * frequencies * integer_delay / probe.sample_rate
    )
    ir = channel_impulse_response(probe, residual, drive=drive, pre_roll=pre_roll)
    if ir.size < fir_length:
        raise ValueError(
            f"복원 IR 길이 {ir.size} < FIR {fir_length} — 분석 주기를 늘리세요"
        )
    fir = ir[:fir_length].astype(np.float32)

    return {
        "frequencies_hz": frequencies,
        "repeat_transfers": stack,
        "aligned_transfers": aligned,
        "mean_transfer": mean_transfer,
        "taus": taus_kept,
        "all_taus": taus,
        "alignment_scores": scores,
        "kept_mask": keep,
        "rejected_repeats": int(taus.size - taus_kept.size),
        "consistency": consistency,
        "fullband_consistency": fullband_consistency,
        "consistency_band_hz": (
            float(consistency_band_hz[0]), float(consistency_band_hz[1])
        ),
        "raw_consistency": complex_consistency(stack),
        "absolute_tau_spread": float(np.max(taus_kept) - np.min(taus_kept)),
        "delay_samples": integer_delay - pre_roll,
        "bulk_delay_samples": integer_delay,
        "delay_fractional": delay,
        "time_origin_convention": "bulk_delay_minus_pre_roll_v1",
        "fir": fir,
        "ir": ir,
        "pre_roll": int(pre_roll),
    }


def channel_quality(
    *,
    consistency: float,
    snr_db: np.ndarray,
    min_consistency: float,
) -> list[str]:
    reasons: list[str] = []
    if not np.isfinite(consistency) or consistency < min_consistency:
        reasons.append(f"consistency_{consistency:.4f}")
    finite = snr_db[np.isfinite(snr_db)]
    if finite.size != snr_db.size or finite.size == 0:
        reasons.append("tone_snr_not_finite")
    else:
        good = float(np.mean(finite >= MIN_TONE_SNR_DB))
        if good < MIN_TONE_SNR_FRACTION:
            reasons.append(f"tone_snr_coverage_{good:.3f}")
    return reasons


def _official_arrays(
    *,
    model: dict[str, Any],
    relative_delay_spread: int,
    max_delay_jitter_samples: int,
    fs: int,
    consistency: float,
    band_hz: tuple[float, float],
    amplitude: float,
    block_size: int,
    latency: str,
    output_channel: str,
    repeats: int,
    xrun_count: int,
    capture_id: str,
    probe,
    drive: str,
    snr_db: np.ndarray,
    period_seconds: float,
) -> dict[str, Any]:
    return {
        "fir": np.asarray(model["fir"], dtype=np.float32),
        "delay_samples": np.int64(model["delay_samples"]),
        "bulk_delay_samples": np.int64(model["bulk_delay_samples"]),
        "pre_roll_samples": np.int64(model["pre_roll"]),
        "time_origin_convention": np.str_(model["time_origin_convention"]),
        "sample_rate": np.int64(fs),
        "coherence_median": np.float64(consistency),
        "consistency_band_hz": np.asarray(
            model["consistency_band_hz"], dtype=np.float64
        ),
        "fullband_consistency": np.float64(model["fullband_consistency"]),
        "excitation_band_hz": np.asarray(band_hz, dtype=np.float64),
        "calibration_block_size": np.int64(block_size),
        "calibration_latency": np.str_(latency),
        "output_channel": np.str_(output_channel),
        "method": np.str_(METHOD),
        "repeats": np.int64(repeats),
        "amplitude": np.float64(amplitude),
        "xrun_count": np.int64(xrun_count),
        # 동시 측정에서 게이트가 판정해야 하는 지연 안정도는 **두 경로의 상대값**이다.
        # 두 채널은 같은 DAC·같은 스트림을 지나므로 절대 warp 가 공통으로 실리고,
        # lead = S + handoff − P 에서 상쇄된다. 절대 흔들림은 아래에 따로 남긴다.
        "delay_spread_samples": np.int64(relative_delay_spread),
        "max_delay_jitter_samples": np.int64(max_delay_jitter_samples),
        "absolute_tau_spread_samples": np.float64(model["absolute_tau_spread"]),
        "repeat_tau_samples": np.asarray(model["taus"], dtype=np.float64),
        "raw_consistency": np.float64(model["raw_consistency"]),
        "rejected_repeats": np.int64(model["rejected_repeats"]),
        "alignment_scores": np.asarray(model["alignment_scores"], dtype=np.float64),
        # --- interleaved 전용 (게이트가 method 별로 추가 검사한다) ---
        "capture_id": np.str_(capture_id),
        "interleave_guard_bins": np.int64(probe.guard_bins()),
        "analysis_period_seconds": np.float64(period_seconds),
        "tone_count": np.int64(probe.bins_for(drive).size),
        "tone_snr_median_db": np.float64(float(np.median(snr_db))),
        "tone_snr_min_db": np.float64(float(np.min(snr_db))),
        "tone_frequencies_hz": (
            probe.bins_for(drive) * probe.sample_rate / probe.period_samples
        ).astype(np.float64),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.confirm_volume_minimum:
        print(
            "[중단] 스피커가 울립니다. 사용자 입회와 앰프 볼륨 최저를 확인한 뒤 "
            "--confirm-volume-minimum 을 지정하세요.",
            file=sys.stderr,
        )
        return 2

    try:
        hardware = load_yaml(REPO_ROOT / args.hardware)["audio"]
        fs = int(hardware["sample_rate"])
        block_size = int(args.block_size or hardware["block_size"])
        if not 0.0 < args.amplitude <= cw.MAX_AMPLITUDE:
            raise ValueError(f"--amplitude 는 0 초과 {cw.MAX_AMPLITUDE} 이하여야 합니다")
        if args.repeats < cw.MIN_REPEATS:
            raise ValueError(f"--repeats 는 {cw.MIN_REPEATS} 이상이어야 합니다")
        primary_out = cw._repo_path(args.primary_out)
        secondary_out = cw._repo_path(args.secondary_out)
        for path in (primary_out, secondary_out):
            if path.exists():
                raise FileExistsError(f"기존 정식 모델은 덮어쓰지 않습니다: {path}")
        if primary_out == secondary_out:
            raise ValueError("P 와 S 는 다른 파일이어야 합니다")
        diagnostics_root = cw._repo_path(args.diagnostics_root, require_results=True)
        max_delay = int(round(args.max_delay_ms / 1000.0 * fs))
        max_jitter = int(round(args.max_delay_jitter_ms / 1000.0 * fs))
        probe = build_interleaved_probe(
            sample_rate=fs,
            period_seconds=args.period_seconds,
            band_hz=(float(args.band[0]), float(args.band[1])),
            amplitude=float(args.amplitude),
            tone_spacing_hz=(
                float(args.tone_spacing_hz) if args.tone_spacing_hz else None
            ),
        )
        if probe.guard_bins() != 1:
            raise ValueError(
                f"guard={probe.guard_bins()} bin — 게이트는 1 을 요구합니다. "
                "--tone-spacing-hz 를 지우거나 --period-seconds 를 조정하세요"
            )
    except (KeyError, OSError, ValueError, FileExistsError) as exc:
        print(f"[중단] {exc}", file=sys.stderr)
        return 2

    need_lo, need_hi = float(args.required_band[0]), float(args.required_band[1])
    resolution = fs / probe.period_samples
    channel_band = {}
    for drive in ("noise", "cancel"):
        bins = probe.bins_for(drive)
        low, high = float(bins[0]) * resolution, float(bins[-1]) * resolution
        channel_band[drive] = (low, high)
        if low > need_lo or high < need_hi:
            print(
                f"[중단] {drive} 톤 대역 {low:.1f}-{high:.1f}Hz 가 필수 대역 "
                f"{need_lo:.0f}-{need_hi:.0f}Hz 를 덮지 못합니다. --band 를 넓히세요.",
                file=sys.stderr,
            )
            return 2

    crest_noise, crest_cancel = probe.crest_db()
    if max(crest_noise, crest_cancel) > MAX_CREST_DB:
        print(
            f"[중단] 크레스트 {crest_noise:.1f}/{crest_cancel:.1f} dB 가 "
            f"{MAX_CREST_DB} dB 를 넘습니다 — 같은 피크에서 음향 에너지를 잃습니다.",
            file=sys.stderr,
        )
        return 2

    capture_id = uuid.uuid4().hex
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = diagnostics_root / f"{stamp}_{capture_id[:8]}"
    session_dir.mkdir(parents=True, exist_ok=False)

    print(
        f"동시 인터리브 측정 {args.band[0]:.0f}-{args.band[1]:.0f}Hz · "
        f"톤 간격 {probe.bin_step('noise') * resolution:.2f}Hz · "
        f"주기 {args.period_seconds:.2f}s\n"
        f"  톤 수 noise {probe.noise_bins.size} / cancel {probe.cancel_bins.size} · "
        f"guard {probe.guard_bins()} bin · crest {crest_noise:.1f}/{crest_cancel:.1f} dB\n"
        f"  peak {args.amplitude:.4f} · block {block_size} · latency {args.latency} · "
        f"warmup {args.warmup_periods} + 분석 {args.repeats} 주기 "
        f"({(args.warmup_periods + args.repeats) * args.period_seconds:.0f}초 재생)"
    )

    try:
        import sounddevice as sd

        print("출력 없는 ERR/REF raw preflight 중...")
        preflight_raw, preflight_report = cw._capture_preflight(
            sd, hardware, args.input_probe_seconds
        )
        for name, item in zip(("ERR", "REF"), cw._probe_summary(preflight_report)):
            verdict = "PASS" if item["valid"] else "FAIL"
            print(
                f"[{verdict}] {name}: RMS {item['rms_dbfs']:.2f}dBFS, "
                f"peak {item['peak']:.6f}, clip {item['clip_ratio']:.3%}"
            )
        channels = preflight_report.get("channels", [])
        if len(channels) < 2 or not all(bool(c.get("valid")) for c in channels[:2]):
            print("[실패] 양 마이크 preflight 실패 — 출력 장치를 열지 않았습니다", file=sys.stderr)
            return 1

        in_dev = int(preflight_report["device"])
        output_cfg = hardware["output"]
        out_dev = resolve_alsa_portaudio_device(
            output_cfg["card"], output_cfg["pcm"], "output", 2
        )

        lead_in = fs // 2
        total_periods = int(args.warmup_periods) + int(args.repeats)
        playback = np.zeros((lead_in + total_periods * probe.period_samples, 2), np.float32)
        playback[lead_in:, 0] = np.tile(probe.noise_signal, total_periods)
        playback[lead_in:, 1] = np.tile(probe.cancel_signal, total_periods)

        recorded_raw, output_pcm, telemetry = cw._capture_measurement(
            sd,
            fs=fs,
            block_size=block_size,
            latency=str(args.latency),
            in_dev=in_dev,
            out_dev=out_dev,
            output_float=playback,
        )
    except (ImportError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"[실패] 측정 중단: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    recorded = pcm_int32_to_float32(recorded_raw)
    err = recorded[:, 0].astype(np.float64)
    measurement_report = cw.analyze_int32_input_probe(recorded_raw)

    invalid: list[str] = []
    if int(telemetry.get("xrun_count", 0)) != 0:
        invalid.append(f"xrun_{telemetry['xrun_count']}")
    if not telemetry.get("completed"):
        invalid.append("capture_incomplete")
    for index, item in enumerate(measurement_report.get("channels", [])[:2]):
        if float(item.get("clip_ratio", 1.0)) > cw.MAX_INPUT_CLIP_RATIO:
            invalid.append(f"input_clip_ch{index}_{item['clip_ratio']:.4f}")

    warp_report: dict[str, Any] = {"applied": False}
    if args.dewarp:
        # 재생 두 채널의 합이 곧 스피커가 함께 만든 음향 자극의 시간 구조다.
        # warp 는 그 합에 공통으로 걸리므로 합을 기준으로 추적하는 것이 옳다.
        mono = playback[:, 0].astype(np.float64) + playback[:, 1].astype(np.float64)
        centres, delays, peaks = track_warp(mono, err, window=int(args.track_window))
        err = dewarp_recording(err, centres, delays, peaks, min_peak=float(args.track_min_peak))
        warp_report = {
            "applied": True,
            "window": int(args.track_window),
            "points": int(delays.size),
            "kept_fraction": float(np.mean(peaks >= float(args.track_min_peak))),
            "delay_min": float(np.min(delays)),
            "delay_max": float(np.max(delays)),
            "delay_range": float(np.max(delays) - np.min(delays)),
            "peak_median": float(np.median(peaks)),
        }
        print(
            f"\nwarp 추적: 지연 {warp_report['delay_min']:.0f}~{warp_report['delay_max']:.0f} "
            f"(범위 {warp_report['delay_range']:.0f} 샘플) · 상관 중앙 "
            f"{warp_report['peak_median']:.3f} · 채택 {warp_report['kept_fraction']:.1%}"
        )

    period_starts = [
        lead_in + (int(args.warmup_periods) + k) * probe.period_samples
        for k in range(int(args.repeats))
    ]

    # 배경잡음 스펙트럼은 preflight 를 **같은 길이·같은 FFT** 로 변환해야 분모가 맞는다.
    preflight_err = pcm_int32_to_float32(preflight_raw)[:, 0].astype(np.float64)
    if preflight_err.size < probe.period_samples:
        preflight_err = np.pad(preflight_err, (0, probe.period_samples - preflight_err.size))
    noise_spectrum = np.fft.rfft(preflight_err[-probe.period_samples :])
    signal_spectrum = np.fft.rfft(err[period_starts[0] : period_starts[0] + probe.period_samples])

    fit_band = (float(args.fit_band[0]), float(args.fit_band[1]))
    consistency_band = (
        float(args.consistency_band[0]), float(args.consistency_band[1])
    )
    if consistency_band[0] > need_lo or consistency_band[1] < need_hi:
        print(
            f"[중단] 일관성 대역 {consistency_band} 가 필수 대역 "
            f"({need_lo}, {need_hi}) 를 덮지 못합니다 — 게이트가 거부할 값을 만듭니다.",
            file=sys.stderr,
        )
        return 2
    results: dict[str, dict[str, Any]] = {}
    for drive, output_channel in (("noise", "noise"), ("cancel", "cancel")):
        try:
            model = analyse_channel(
                err=err, probe=probe, drive=drive, period_starts=period_starts,
                fir_length=int(args.fir_length), pre_roll=int(args.pre_roll),
                max_delay_samples=max_delay, fit_band_hz=fit_band,
                consistency_band_hz=consistency_band,
                min_alignment_score=float(args.min_alignment_score),
                min_kept_repeats=int(args.min_kept_repeats),
            )
        except ValueError as exc:
            print(f"[실패] {drive} 분석: {exc}", file=sys.stderr)
            return 1
        snr = tone_snr_db(signal_spectrum, noise_spectrum, probe.bins_for(drive))
        results[drive] = {
            "model": model,
            "snr_db": snr,
            "output_channel": output_channel,
            "reasons": channel_quality(
                consistency=model["consistency"], snr_db=snr,
                min_consistency=cw.MIN_CONSISTENCY,
            ),
        }

    # lead 가 의존하는 유일한 양 — 두 경로의 **상대** 시간이동. 절대 warp 는 두 채널에
    # 공통이므로 여기서 상쇄된다. 이것이 커지면 lead 를 믿을 수 없다.
    both = results["noise"]["model"]["kept_mask"] & results["cancel"]["model"]["kept_mask"]
    if int(both.sum()) < 2:
        print("[실패] 두 채널 모두 정렬에 성공한 반복이 2개 미만입니다", file=sys.stderr)
        return 1
    relative_tau = (
        results["noise"]["model"]["all_taus"][both]
        - results["cancel"]["model"]["all_taus"][both]
    )
    relative_spread = int(round(float(np.max(relative_tau) - np.min(relative_tau))))
    if relative_spread > max_jitter:
        for item in results.values():
            item["reasons"].append(f"relative_delay_spread_{relative_spread}")

    for drive, label in (("noise", "P(z) 소음→ERR"), ("cancel", "S(z) 상쇄→ERR")):
        item = results[drive]
        model, snr = item["model"], item["snr_db"]
        freq = model["frequencies_hz"]
        bands = "  ".join(
            f"{lo}-{hi}:{complex_consistency(model['aligned_transfers'][:, (freq >= lo) & (freq <= hi)]):.3f}"
            for lo, hi in ((80, 150), (150, 300), (300, 600), (600, 1000), (1000, 1600))
        )
        print(
            f"\n=== {label} ===\n"
            f"  분리 delay {model['delay_samples']} 샘플 "
            f"(벌크 {model['delay_fractional']:.2f}, pre-roll {model['pre_roll']}) · "
            f"{model['consistency_band_hz'][0]:.0f}-"
            f"{model['consistency_band_hz'][1]:.0f}Hz 일관성 "
            f"**{model['consistency']:.4f}** (전대역 {model['fullband_consistency']:.4f})\n"
            f"  절대 τ 흔들림 {model['absolute_tau_spread']:.1f} 샘플 "
            f"(상대 {relative_spread}, 허용 {max_jitter}) · "
            f"정렬 실패 반복 {model['rejected_repeats']}/{model['all_taus'].size}\n"
            f"  톤 SNR 중앙 {np.median(snr):.1f} dB · 최소 {np.min(snr):.1f} dB · "
            f"{float(np.mean(snr >= MIN_TONE_SNR_DB)):.1%} 가 {MIN_TONE_SNR_DB:.0f}dB 이상\n"
            f"  대역별 {bands}"
        )
        if item["reasons"]:
            print(f"  [미달] {', '.join(item['reasons'])}")

    valid = not invalid and not results["noise"]["reasons"] and not results["cancel"]["reasons"]

    metadata = {
        "capture_id": capture_id,
        "method": METHOD,
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sample_rate": fs,
        "block_size": block_size,
        "latency": args.latency,
        "amplitude": float(args.amplitude),
        "design_band_hz": [float(args.band[0]), float(args.band[1])],
        "required_band_hz": [need_lo, need_hi],
        "channel_band_hz": {k: list(v) for k, v in channel_band.items()},
        "tone_spacing_hz": float(probe.bin_step("noise") * resolution),
        "period_seconds": float(args.period_seconds),
        "warmup_periods": int(args.warmup_periods),
        "repeats": int(args.repeats),
        "guard_bins": probe.guard_bins(),
        "crest_db": {"noise": crest_noise, "cancel": crest_cancel},
        "warp": warp_report,
        "telemetry": telemetry,
        "preflight": preflight_report,
        "measurement": measurement_report,
        "invalid_reasons": invalid,
        "valid": valid,
        "fit_band_hz": list(fit_band),
        "consistency_band_hz": list(consistency_band),
        "relative_delay_spread_samples": relative_spread,
        "max_delay_jitter_samples": max_jitter,
        "channels": {
            drive: {
                "output_channel": item["output_channel"],
                "consistency": item["model"]["consistency"],
                "fullband_consistency": item["model"]["fullband_consistency"],
                "consistency_band_hz": list(item["model"]["consistency_band_hz"]),
                "raw_consistency": item["model"]["raw_consistency"],
                "delay_samples": item["model"]["delay_samples"],
                "bulk_delay_samples": item["model"]["bulk_delay_samples"],
                "pre_roll_samples": item["model"]["pre_roll"],
                "time_origin_convention": item["model"]["time_origin_convention"],
                "delay_fractional": item["model"]["delay_fractional"],
                "repeat_tau_samples": [float(v) for v in item["model"]["taus"]],
                "absolute_tau_spread_samples": item["model"]["absolute_tau_spread"],
                "rejected_repeats": item["model"]["rejected_repeats"],
                "alignment_scores": [float(v) for v in item["model"]["alignment_scores"]],
                "tone_snr_median_db": float(np.median(item["snr_db"])),
                "tone_snr_min_db": float(np.min(item["snr_db"])),
                "reasons": item["reasons"],
            }
            for drive, item in results.items()
        },
    }

    npz_path = session_dir / "raw_measurement.npz"
    with npz_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            output=playback.astype(np.float32),
            err=recorded[:, 0].astype(np.float32),
            ref=recorded[:, 1].astype(np.float32),
            input_raw_int32=recorded_raw.astype(np.int32),
            preflight_raw_int32=preflight_raw.astype(np.int32),
            noise_transfers=results["noise"]["model"]["repeat_transfers"],
            cancel_transfers=results["cancel"]["model"]["repeat_transfers"],
            noise_ir=results["noise"]["model"]["ir"].astype(np.float64),
            cancel_ir=results["cancel"]["model"]["ir"].astype(np.float64),
            frequencies_hz=results["noise"]["model"]["frequencies_hz"],
            relative_tau_samples=relative_tau,
            noise_snr_db=results["noise"]["snr_db"].astype(np.float64),
            cancel_snr_db=results["cancel"]["snr_db"].astype(np.float64),
            metadata_json=np.asarray(
                json.dumps(cw._json_safe(metadata), ensure_ascii=False, sort_keys=True)
            ),
        )
    with (session_dir / "metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(cw._json_safe(metadata), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    if not valid:
        print(f"\n[실패] 정식 모델을 저장하지 않았습니다. 진단: {session_dir}", file=sys.stderr)
        if invalid:
            print(f"  캡처 결함: {', '.join(invalid)}", file=sys.stderr)
        return 1

    for drive, out_path in (("noise", primary_out), ("cancel", secondary_out)):
        item = results[drive]
        cw.save_official_model(
            out_path,
            valid=True,
            arrays=_official_arrays(
                model=item["model"],
                relative_delay_spread=relative_spread,
                max_delay_jitter_samples=max_jitter,
                fs=fs,
                consistency=item["model"]["consistency"],
                band_hz=channel_band[drive],
                amplitude=float(args.amplitude),
                block_size=block_size,
                latency=str(args.latency),
                output_channel=item["output_channel"],
                repeats=int(args.repeats),
                xrun_count=int(telemetry.get("xrun_count", 0)),
                capture_id=capture_id,
                probe=probe,
                drive=drive,
                snr_db=item["snr_db"],
                period_seconds=float(args.period_seconds),
            ),
        )

    p_delay = int(results["noise"]["model"]["delay_samples"])
    s_delay = int(results["cancel"]["model"]["delay_samples"])
    handoff = 256
    lead = max(0, s_delay + handoff - p_delay)
    print(
        f"\n[성공] P {primary_out.relative_to(REPO_ROOT)}\n"
        f"       S {secondary_out.relative_to(REPO_ROOT)}\n"
        f"       진단 {session_dir.relative_to(REPO_ROOT)}\n\n"
        f"duct.yaml 에 기입할 값:\n"
        f"  digital_reference.primary_path_npz: {primary_out.relative_to(REPO_ROOT)}\n"
        f"  digital_reference.d_noise_delay_samples: {p_delay}\n"
        f"  secondary_path.npz: {secondary_out.relative_to(REPO_ROOT)}\n"
        f"data_sim.yaml digital_reference_lead_samples: "
        f"{lead}  (= S {s_delay} + handoff {handoff} − P {p_delay})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
