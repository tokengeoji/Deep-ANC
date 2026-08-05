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
    relative_tau_outliers,
    timebase_drift,
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
# 워밍업 4주기(0.5s)로는 스트림이 정상상태에 못 든다 — 2026-08-05 실측에서 저장된
# 캡처 9건 **전부** 반복 0(대개 반복 1도)의 국소 타임베이스 드리프트가 정상상태의
# 2~5배였다. 16주기(2.0s)로 늘린다.
DEFAULT_WARMUP_PERIODS = 16
# 게이트가 반복을 버리므로 여유를 둔다. 저장된 캡처에서 유지율은 33~75% 였고
# min_kept_repeats 8 을 확보하려면 32 가 필요하다(16 이면 8/16 로 아슬아슬했다).
# 재생 길이는 0.125s×(16+32) = 6.0초 — 앞선 (4+16)×0.125 = 2.5초보다 길다.
DEFAULT_REPEATS = 32
# τ 는 재현되는 대역에서만 적합한다. 재현 안 되는 대역을 넣으면 그 잡음이 τ 를 끌고 간다.
DEFAULT_FIT_BAND_HZ = (150.0, 1200.0)
# 일관성을 **어느 대역에서 쟀는지**가 곧 이 모델을 어느 대역에서 믿을 수 있는가다.
# 그래서 숫자만 저장하지 않고 대역도 함께 저장하고, 게이트가 그 대역이 요구 대역을
# 덮는지 검사한다. 대역이 안 적혀 있으면 0.95 라는 숫자가 무엇에 대한 0.95 인지 모른다.
#
# 2026-08-05 정정: [150,600] 은 "600Hz 위는 덕트 물리 한계" 라는 **틀린 전제** 위에
# 있었다. 아래 게이트를 켜고 재분석하면 1000-1600Hz 일관성이 P 0.999 / S 0.999 다.
DEFAULT_CONSISTENCY_BAND_HZ = (150.0, 1600.0)

MIN_TONE_SNR_DB = 12.0          # 톤 중앙값 SNR 하한
MIN_TONE_SNR_FRACTION = 0.9     # 이 비율 이상의 톤이 하한을 넘어야 한다
MAX_CREST_DB = 14.0

# --- 2-pass 공동 판정 상수 -------------------------------------------------
#
# 아래 네 값은 저장된 캡처 10건(총 236 반복)을 직접 재분석해 정한 것이다.
# 임의로 완화하면 2026-08-05 결함 1(프레임 슬립 5반복 + 정상상태 미도달 1반복이
# 그대로 official 이 된 사건)이 그대로 재현된다.
DEFAULT_MIN_ALIGNMENT_SCORE = 0.95
"""중앙 앵커 2-pass 규약 기준. 단일 pass(앵커 0) 규약이라면 0.80 이 맞는 값이다.

앵커 0 규약에서는 정상군 최저 0.838 과 오염군 최고 0.836 이 겹쳐 점수만으로는
어떤 임계로도 분리되지 않는다. 2-pass 후에는 유지 반복 0.9845~0.9995 / 오염
반복 최고 0.966 으로 완전히 갈린다.
"""

DEFAULT_MAX_RELATIVE_TAU_SAMPLES = 3.0
"""|rel − median(rel)| 허용치(샘플). 정상 최대 1.987 / 오염 최소 4.316.

두 값 사이가 비어 있어 3.0 이 양쪽에 각각 1.5x / 1.44x 여유를 준다.
MAD 스케일 임계는 쓰지 않는다 — 오염이 과반인 캡처에서 MAD 가 부풀어
(실측 1.346 → 허용 11.98) 슬립 블록을 통째로 통과시킨다.
"""

DEFAULT_MAX_DRIFT_DEVIATION_SAMPLES = 2.0
"""|국소 드리프트 − 중앙| 허용치(샘플/주기). 정상 ≤0.83 / 이상 ≥2.63."""

MAX_KEPT_RELATIVE_TAU_ABS_SAMPLES = 1.0
"""최종 아티팩트 게이트. 2-pass 후 실측 |rel|max 는 0.128~0.417 이다."""

MIN_BAND_CONSISTENCY = 0.90
"""필수 대역 안 **모든** 부대역이 넘어야 한다 — 총계는 에너지 가중이라 약한
대역을 숨긴다(실측: S 전대역 총계 0.9987 인데 80-150Hz 부대역만 보면 0.706)."""

CONSISTENCY_SUB_BANDS_HZ = (
    (80.0, 150.0),
    (150.0, 300.0),
    (300.0, 600.0),
    (600.0, 1000.0),
    (1000.0, 1600.0),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument("--band", type=float, nargs=2, default=list(DEFAULT_BAND_HZ))
    parser.add_argument("--required-band", type=float, nargs=2, default=[150.0, 1600.0])
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
        "--min-alignment-score", type=float, default=DEFAULT_MIN_ALIGNMENT_SCORE,
        help=(
            "이 신뢰도 미만인 반복은 τ 탐색 실패로 보고 버린다(개수는 산출물에 기록). "
            "중앙 앵커 2-pass 규약에서 유지 반복은 0.9845~0.9995, 오염 반복은 최고 0.966"
        ),
    )
    parser.add_argument(
        "--max-relative-tau-samples", type=float,
        default=DEFAULT_MAX_RELATIVE_TAU_SAMPLES,
        help="P−S 상대 τ 의 중앙값 편차 허용(샘플). 정상 최대 1.99 / 오염 최소 4.32",
    )
    parser.add_argument(
        "--max-drift-deviation-samples", type=float,
        default=DEFAULT_MAX_DRIFT_DEVIATION_SAMPLES,
        help="국소 타임베이스 드리프트의 중앙값 편차 허용(샘플/주기). 정상 ≤0.83",
    )
    parser.add_argument("--min-kept-repeats", type=int, default=8)
    parser.add_argument("--max-delay-jitter-ms", type=float, default=0.0625)
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
            "**쓰지 마라** — 2026-08-05 오프라인 재검증에서 전대역 일관성이 "
            "window 1024/2048/4096 에서 0.834/0.879/0.762 로 **떨어졌다**(무보정 0.979). "
            "앞선 '0.05 → 0.85' 개선 주장은 반복 0 오염이 지배하던 시절의 값이다. "
            "진단용으로만 남긴다."
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


def channel_stack(
    *,
    err: np.ndarray,
    probe,
    drive: str,
    period_starts: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """주기별 전달함수를 쌓아 ``(주파수, [반복, 톤] 복소 스택)`` 을 준다."""

    rows = [
        estimate_transfer(err[s : s + probe.period_samples], probe, drive=drive)
        for s in period_starts
    ]
    return rows[0][0], np.stack([H for _, H in rows])


def select_repeats(
    *,
    frequencies: dict[str, np.ndarray],
    stacks: dict[str, np.ndarray],
    sample_rate: int,
    fit_band_hz: tuple[float, float],
    max_relative_tau_samples: float,
    max_drift_deviation_samples: float,
    min_kept_repeats: int = 3,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """두 채널을 **함께** 보고 쓸 반복과 앵커를 정한다.

    반환 ``(keep, anchor, report)``. 판정 근거는 전부 타임베이스 관측이다 —
    "결과가 좋아지는가"로 고르지 않는다. 어떤 반복을 왜 버렸는지 report 에 남는다.

    채널별 독립 판정이면 P 에서만 버린 반복이 S 에 남아 두 경로가 서로 다른 반복
    집합에서 만들어질 수 있다. 그러면 lead 가 물리량이 아니게 된다.
    """

    taus: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    for drive in ("noise", "cancel"):
        _, tau, score = align_repeats(
            frequencies[drive], stacks[drive], sample_rate=sample_rate,
            fit_band_hz=fit_band_hz, anchor=0,
        )
        taus[drive], scores[drive] = tau, score

    # (a) 타임베이스가 정상상태였는가. 워밍업 4주기로는 부족하다는 것이 실측이다.
    common = 0.5 * (taus["noise"] + taus["cancel"])
    drift, drift_median = timebase_drift(common)
    drift_dev = np.abs(drift - drift_median)
    keep = drift_dev <= float(max_drift_deviation_samples)

    # (b) P−S 상대 τ 연속성. 이 측정 방식의 유일한 물리 불변량이다.
    bad_rel, rel_dev, rel_centre = relative_tau_outliers(
        taus["noise"], taus["cancel"],
        tolerance_samples=float(max_relative_tau_samples),
    )
    keep = keep & ~bad_rel

    # (c) 살아남은 무리가 **스트림의 첫 분석 주기와 같은 프레임 정렬**인가.
    #     pass1 앵커(반복 0)에서는 상대 τ 가 구조적으로 0 이다. 중앙값이 0 에서
    #     크게 벗어났다는 것은 살아남은 쪽이 슬립 **이후** 무리라는 뜻이고, 한 캡처
    #     안의 정보만으로는 어느 쪽이 옳은지 가릴 수 없다 — 실패 폐쇄한다.
    #     실측 9건의 |centre| 는 0.07~1.81 로 임계 3.0 안에 넉넉히 들어온다.
    if abs(rel_centre) > float(max_relative_tau_samples):
        raise ValueError(
            f"유지 무리의 P−S 상대 τ 중앙값이 {rel_centre:+.2f} 샘플로 첫 분석 주기와 "
            f"다릅니다 (허용 ±{float(max_relative_tau_samples)}) — 프레임 슬립이 "
            "과반이라 어느 무리가 옳은지 이 캡처만으로는 가릴 수 없습니다"
        )

    kept = np.flatnonzero(keep)
    if kept.size < int(min_kept_repeats):
        raise ValueError(
            f"타임베이스가 안정한 반복이 {kept.size}개뿐입니다 "
            f"(최소 {int(min_kept_repeats)}, 드리프트 중앙 {drift_median:.2f} 샘플/주기)"
        )
    anchor = int(kept[kept.size // 2])   # 중앙 반복 — 드리프트 외삽을 최소화한다
    report = {
        "drift_samples_per_period": float(drift_median),
        "drift_deviation": drift_dev,
        "relative_tau": taus["noise"] - taus["cancel"],
        "relative_tau_centre": rel_centre,
        "relative_tau_deviation": rel_dev,
        "pass1_taus": taus,
        "pass1_scores": scores,
        "drift_rejected": np.flatnonzero(
            drift_dev > float(max_drift_deviation_samples)
        ),
        "relative_tau_rejected": np.flatnonzero(bad_rel),
    }
    return keep, anchor, report


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
    keep: np.ndarray,
    anchor: int,
    stack: np.ndarray | None = None,
) -> dict[str, Any]:
    """주기별 전달함수를 주파수영역에서 정렬·평균한 뒤 순수지연 + compact FIR 로 나눈다.

    ``keep`` 과 ``anchor`` 는 **받는다** — 스스로 정하지 않는다. 두 채널의 τ 를
    함께 봐야만 판정할 수 있는 P−S 상대 τ 연속성이 판정 근거에 들어 있고
    (``select_repeats``), 채널마다 다른 반복 집합을 쓰면 lead 가 물리량이 아니게 된다.
    기본값을 주지 않는 것도 의도적이다 — 빠뜨리면 조용히 통과하는 대신 즉시 실패한다.

    반환에 ``taus`` 가 들어 있는 것이 핵심이다. 두 채널은 같은 스트림을 지나므로
    warp 가 공통으로 실린다 — 따라서 **τ 의 차이**(P − S)가 lead 가 의존하는 유일한
    양이고, 절대 τ 의 흔들림은 lead 에서 상쇄된다. 게이트가 판정해야 하는 것도 그 차이다.
    """

    if stack is None:
        frequencies, stack = channel_stack(
            err=err, probe=probe, drive=drive, period_starts=period_starts
        )
    else:
        stack = np.asarray(stack, dtype=np.complex128)
        frequencies = (
            probe.bins_for(drive) * probe.sample_rate / probe.period_samples
        ).astype(np.float64)
    aligned, taus, scores = align_repeats(
        frequencies, stack, sample_rate=probe.sample_rate,
        fit_band_hz=fit_band_hz, anchor=int(anchor),
    )
    # τ 탐색이 봉우리를 못 찾은 반복은 시간축 정보가 아니라 잡음이다. 판정 근거는
    # **타임베이스 관측(select_repeats)과 정렬 신뢰도** 둘뿐이며, "결과가 좋아지는가"로
    # 고르지 않는다 — 그건 게이트 우회다. 몇 개를 왜 버렸는지 산출물에 남긴다.
    keep = np.asarray(keep, dtype=bool).reshape(-1) & (
        scores >= float(min_alignment_score)
    )
    if keep.size != stack.shape[0]:
        raise ValueError(f"keep 길이가 반복 수와 다릅니다: {keep.size} != {stack.shape[0]}")
    keep[int(anchor)] = True     # 앵커는 자기상관 1 이라 항상 살아남는다
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

    band_consistency = np.asarray(
        [
            complex_consistency(aligned[:, (frequencies >= lo) & (frequencies <= hi)])
            if int(((frequencies >= lo) & (frequencies <= hi)).sum()) >= 4
            else np.nan
            for lo, hi in CONSISTENCY_SUB_BANDS_HZ
        ],
        dtype=np.float64,
    )

    return {
        "frequencies_hz": frequencies,
        "repeat_transfers": stack,
        "aligned_transfers": aligned,
        "mean_transfer": mean_transfer,
        "taus": taus_kept,
        "all_taus": taus,
        "alignment_scores": scores,
        "kept_mask": keep,
        "anchor_repeat": int(anchor),
        "kept_repeat_indices": np.flatnonzero(keep).astype(np.int64),
        "band_consistency": band_consistency,
        "band_consistency_hz": np.asarray(CONSISTENCY_SUB_BANDS_HZ, dtype=np.float64),
        "rejected_repeats": int(taus.size - taus_kept.size),
        "consistency": consistency,
        "fullband_consistency": fullband_consistency,
        "consistency_band_hz": (
            float(consistency_band_hz[0]), float(consistency_band_hz[1])
        ),
        "raw_consistency": complex_consistency(stack),
        "absolute_tau_spread": float(np.max(taus_kept) - np.min(taus_kept)),
        "delay_samples": integer_delay,
        "delay_fractional": delay,
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


def analyse_capture(
    *,
    err: np.ndarray,
    probe,
    period_starts: list[int],
    snr_spectra: tuple[np.ndarray, np.ndarray],
    fir_length: int,
    pre_roll: int,
    max_delay_samples: int,
    fit_band_hz: tuple[float, float],
    consistency_band_hz: tuple[float, float],
    required_band_hz: tuple[float, float],
    min_alignment_score: float,
    min_kept_repeats: int,
    max_relative_tau_samples: float,
    max_drift_deviation_samples: float,
    max_delay_jitter_samples: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """캡처 한 건을 2-pass 공동 분석한다 — **온라인·오프라인이 공유하는 유일한 경로**.

    pass1 은 두 채널의 τ 궤적만 얻어 (a) 타임베이스 드리프트 이상치와 (b) P−S 상대 τ
    연속성 위반을 **함께** 판정한다. pass2 는 살아남은 반복의 중앙을 앵커로 재정렬한다.
    측정 스크립트와 재분석 스크립트가 서로 다른 코드로 갈라지면 재현성이 깨지므로
    두 경로 모두 이 함수만 호출한다.
    """

    signal_spectrum, noise_spectrum = snr_spectra
    need_lo, need_hi = float(required_band_hz[0]), float(required_band_hz[1])

    frequencies: dict[str, np.ndarray] = {}
    stacks: dict[str, np.ndarray] = {}
    for drive in ("noise", "cancel"):
        frequencies[drive], stacks[drive] = channel_stack(
            err=err, probe=probe, drive=drive, period_starts=period_starts
        )

    keep, anchor, report = select_repeats(
        frequencies=frequencies, stacks=stacks, sample_rate=probe.sample_rate,
        fit_band_hz=fit_band_hz,
        max_relative_tau_samples=max_relative_tau_samples,
        max_drift_deviation_samples=max_drift_deviation_samples,
        min_kept_repeats=min_kept_repeats,
    )
    report["drift_ppm"] = (
        1e6 * report["drift_samples_per_period"] / float(probe.period_samples)
    )

    results: dict[str, dict[str, Any]] = {}
    for drive in ("noise", "cancel"):
        model = analyse_channel(
            err=err, probe=probe, drive=drive, period_starts=period_starts,
            fir_length=fir_length, pre_roll=pre_roll,
            max_delay_samples=max_delay_samples, fit_band_hz=fit_band_hz,
            consistency_band_hz=consistency_band_hz,
            min_alignment_score=min_alignment_score,
            min_kept_repeats=min_kept_repeats,
            keep=keep, anchor=anchor, stack=stacks[drive],
        )
        snr = tone_snr_db(signal_spectrum, noise_spectrum, probe.bins_for(drive))
        results[drive] = {
            "model": model,
            "snr_db": snr,
            "output_channel": drive,
            "reasons": channel_quality(
                consistency=model["consistency"], snr_db=snr,
                min_consistency=cw.MIN_CONSISTENCY,
            ),
        }

    # 두 채널의 점수 게이트가 서로 다른 반복을 떨어뜨릴 수 있다. official 은 **두 채널이
    # 같은 반복 집합**에서 나와야 lead 가 물리량이므로 교집합을 최종 유지 집합으로 삼는다.
    final_keep = (
        results["noise"]["model"]["kept_mask"] & results["cancel"]["model"]["kept_mask"]
    )
    if int(final_keep.sum()) < int(min_kept_repeats):
        raise ValueError(
            f"두 채널 모두 통과한 반복이 {int(final_keep.sum())}개뿐입니다 "
            f"(최소 {int(min_kept_repeats)})"
        )

    # lead 가 의존하는 유일한 양 — 두 경로의 **상대** 시간이동. 절대 warp 는 두 채널에
    # 공통이므로 여기서 상쇄된다. 이것이 커지면 lead 를 믿을 수 없다.
    relative_tau = (
        results["noise"]["model"]["all_taus"][final_keep]
        - results["cancel"]["model"]["all_taus"][final_keep]
    )
    relative_spread = int(
        np.ceil(float(np.max(relative_tau) - np.min(relative_tau)) - 1e-9)
    )
    if relative_spread > int(max_delay_jitter_samples):
        for item in results.values():
            item["reasons"].append(f"relative_delay_spread_{relative_spread}")

    # 최종 아티팩트 게이트 — 2-pass 후 상대 τ 는 진짜 0 근처로 수렴해야 한다.
    # 실측 |rel|max 는 캡처 8건에서 0.128~0.417 이다.
    relative_max_abs = float(np.max(np.abs(relative_tau - np.median(relative_tau))))
    if relative_max_abs > MAX_KEPT_RELATIVE_TAU_ABS_SAMPLES:
        for item in results.values():
            item["reasons"].append(f"kept_relative_tau_{relative_max_abs:.2f}")

    # 최악 부대역 게이트 — 총계는 에너지 가중이라 약한 대역을 숨긴다.
    for item in results.values():
        bands = item["model"]["band_consistency"]
        edges = item["model"]["band_consistency_hz"]
        for (lo, hi), value in zip(edges, bands):
            if lo < need_lo or hi > need_hi:
                continue     # 필수 대역 밖은 판정하지 않는다
            if not np.isfinite(value) or value < MIN_BAND_CONSISTENCY:
                item["reasons"].append(
                    f"band_consistency_{lo:.0f}_{hi:.0f}_{value:.3f}"
                )

    report.update(
        {
            "keep": final_keep,
            "anchor": int(anchor),
            "relative_tau_kept": relative_tau,
            "relative_delay_spread_samples": relative_spread,
            "relative_tau_max_abs": relative_max_abs,
        }
    )
    return results, report


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
    drift_samples_per_period: float,
    relative_tau_max_abs: float,
) -> dict[str, Any]:
    return {
        "fir": np.asarray(model["fir"], dtype=np.float32),
        "delay_samples": np.int64(model["delay_samples"]),
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
        # --- 최악 부대역 게이트용. 총계는 에너지 가중이라 약한 대역을 숨긴다. ---
        "band_consistency": np.asarray(model["band_consistency"], dtype=np.float64),
        "band_consistency_hz": np.asarray(
            model["band_consistency_hz"], dtype=np.float64
        ),
        # 절대 지연은 앵커 규약에 의존한다 — 어느 반복을 기준으로 잡았는지 없이는
        # delay_samples 를 재현할 수 없다. 반드시 함께 저장한다.
        "anchor_repeat": np.int64(model["anchor_repeat"]),
        "kept_repeat_indices": np.asarray(
            model["kept_repeat_indices"], dtype=np.int64
        ),
        "drift_samples_per_period": np.float64(drift_samples_per_period),
        "relative_tau_max_abs_samples": np.float64(relative_tau_max_abs),
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
    try:
        results, report = analyse_capture(
            err=err, probe=probe, period_starts=period_starts,
            snr_spectra=(signal_spectrum, noise_spectrum),
            fir_length=int(args.fir_length), pre_roll=int(args.pre_roll),
            max_delay_samples=max_delay, fit_band_hz=fit_band,
            consistency_band_hz=consistency_band,
            required_band_hz=(need_lo, need_hi),
            min_alignment_score=float(args.min_alignment_score),
            min_kept_repeats=int(args.min_kept_repeats),
            max_relative_tau_samples=float(args.max_relative_tau_samples),
            max_drift_deviation_samples=float(args.max_drift_deviation_samples),
            max_delay_jitter_samples=max_jitter,
        )
    except ValueError as exc:
        print(f"[실패] 분석: {exc}", file=sys.stderr)
        return 1
    relative_tau = report["relative_tau_kept"]
    relative_spread = int(report["relative_delay_spread_samples"])
    keep, anchor = report["keep"], int(report["anchor"])

    print(
        f"\n반복 선별: {int(keep.sum())}/{keep.size} 유지 · 앵커 {anchor}\n"
        f"  드리프트 중앙 {report['drift_samples_per_period']:.2f} 샘플/주기 "
        f"({report['drift_ppm']:.0f} ppm) · "
        f"드리프트 기각 {list(map(int, report['drift_rejected']))}\n"
        f"  상대 τ 중앙 {report['relative_tau_centre']:.2f} · "
        f"기각 {list(map(int, report['relative_tau_rejected']))} · "
        f"유지 |rel| 최대 {report['relative_tau_max_abs']:.3f}"
    )

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
            f"  순수지연 {model['delay_samples']} 샘플 "
            f"({model['delay_fractional']:.2f}) · "
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
        # 오프라인 재분석이 period_starts 를 재현하려면 반드시 있어야 한다.
        "lead_in_samples": int(lead_in),
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
        "anchor_repeat": anchor,
        "kept_repeat_indices": [int(v) for v in np.flatnonzero(keep)],
        "drift_samples_per_period": float(report["drift_samples_per_period"]),
        "drift_ppm": float(report["drift_ppm"]),
        "relative_tau_centre_samples": float(report["relative_tau_centre"]),
        "relative_tau_max_abs_samples": float(report["relative_tau_max_abs"]),
        "drift_rejected_repeats": [int(v) for v in report["drift_rejected"]],
        "relative_tau_rejected_repeats": [
            int(v) for v in report["relative_tau_rejected"]
        ],
        "max_relative_tau_samples": float(args.max_relative_tau_samples),
        "max_drift_deviation_samples": float(args.max_drift_deviation_samples),
        "min_alignment_score": float(args.min_alignment_score),
        "channels": {
            drive: {
                "output_channel": item["output_channel"],
                "band_consistency": [
                    float(v) for v in item["model"]["band_consistency"]
                ],
                "band_consistency_hz": [
                    [float(lo), float(hi)]
                    for lo, hi in item["model"]["band_consistency_hz"]
                ],
                "consistency": item["model"]["consistency"],
                "fullband_consistency": item["model"]["fullband_consistency"],
                "consistency_band_hz": list(item["model"]["consistency_band_hz"]),
                "raw_consistency": item["model"]["raw_consistency"],
                "delay_samples": item["model"]["delay_samples"],
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
                # 재생한 주기 수가 아니라 **실제로 평균에 들어간** 반복 수를 신고한다.
                repeats=int(keep.sum()),
                xrun_count=int(telemetry.get("xrun_count", 0)),
                capture_id=capture_id,
                probe=probe,
                drive=drive,
                snr_db=item["snr_db"],
                period_seconds=float(args.period_seconds),
                drift_samples_per_period=float(report["drift_samples_per_period"]),
                relative_tau_max_abs=float(report["relative_tau_max_abs"]),
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
