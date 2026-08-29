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
동시 재생 상태로 두 응답을 분리하기 위해 DAC 명령의 주파수를 번갈아 나눈다(guard=1).

    ch0(소음 스피커) → 짝수번째 톤,  ch1(상쇄 스피커) → 홀수번째 톤

DAC 명령에서만 두 빈 집합이 정확히 서로소다. USB DAC와 Tegra ADC가 비동기이면 ADC가
관측하는 톤은 정수 빈에서 벗어나며, guard=1 정수 FFT는 바로 옆 채널을 오염시킨다. 저장된
캡처의 주기당 약 620 ppm 오차에서는 1--1.6 kHz 교차 성분이 약 15%였다. 따라서 각 반복의
실제 주기 길이를 ERR/REF 원시 시간영역에서 독립 추정하고, 저장된 실제 int16 DAC 명령의
모든 톤을 한꺼번에 넣은 fractional-frequency joint real LS로 P/S를 분리한다. cubic
playback-grid 재표본화+정수 FFT가 전대역과 네 부대역에서 같은 전달함수를 내는지도 독립
교차검증한다. 이 clock witness, LS, 교차검증 중 하나라도 빠지면 official로 승격하지 않는다.

산출물
------
게이트를 통과하면 P/S NPZ 두 개를 **같은 capture_id** 로 함께 저장한다. 같은 capture 에서
나왔다는 사실이 파일에 박혀 있어야 파인튜닝 진입 감사가 "두 경로가 같은 조건"임을
파일만 보고 확인할 수 있다(``finetune_readiness.audit_official_path_model``).

사용자 입회·앰프 볼륨 최저에서만 실행한다::

  .venv/bin/python scripts/data/measure_paths_interleaved.py \
      --confirm-user-present --confirm-volume-minimum --confirm-routing-and-geometry

최초 paired level evidence를 만들 때만, 방금 출력된 meter raw를 10분 안에 넘긴다::

  .venv/bin/python scripts/data/measure_paths_interleaved.py \
      --bootstrap-level-evidence --meter-raw <meter_raw.npz> \
      --confirm-same-amplifier-setting \
      --confirm-user-present --confirm-volume-minimum --confirm-routing-and-geometry \
      --primary-out assets/measured/primary_path_il_strict_<capture-id>.npz \
      --secondary-out assets/measured/secondary_path_il_strict_<capture-id>.npz
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import signal
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import map_coordinates

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import calibrate_wideband as cw  # noqa: E402

from deep_anc.audio_io import (  # noqa: E402
    alsa_card_index,
    pcm_int32_to_float32,
    resolve_alsa_portaudio_device,
)
from deep_anc.config import REPO_ROOT, load_yaml  # noqa: E402
from deep_anc.dsp.measurement_level import (  # noqa: E402
    ALSA_PHYSICAL_FINGERPRINT_SCHEMA,
    BOOTSTRAP_CLOCK_FUTURE_TOLERANCE_SECONDS,
    BOOTSTRAP_METER_MAX_AGE_SECONDS,
    DEFAULT_MEASUREMENT_LEVEL_EVIDENCE_PATH,
    LIVE_WATCHDOG_GRACE_SECONDS,
    LiveAudioTermination,
    OFFICIAL_MEASUREMENT_CHANNEL_MAP,
    OFFICIAL_MEASUREMENT_LEVEL,
    assert_live_pcm_clock_preconditions,
    atomic_publish_noreplace,
    collect_alsa_physical_fingerprint,
    create_measurement_level_evidence_atomic,
    load_measurement_level_evidence,
    measurement_hardware_identity,
    repository_audio_lock,
    scoped_live_audio_signal_handlers,
    validate_bootstrap_meter_raw,
    validate_measurement_hardware_contract,
)
from deep_anc.dsp.interleaved_probe import (  # noqa: E402
    DEFAULT_TRACK_WINDOW,
    align_repeats,
    build_interleaved_probe,
    channel_impulse_response,
    complex_consistency,
    dewarp_recording,
    estimate_repeat_delay,
    estimate_transfer,
    relative_tau_outliers,
    timebase_drift,
    tone_snr_db,
    track_warp,
)

METHOD = "interleaved_multitone"
OUTPUT_PCM_PROVENANCE_OBSERVED = "observed_submitted_int16"
OUTPUT_PCM_PROVENANCE_DERIVED = "derived_not_observed"
RAW_CAPTURE_SCHEMA = "interleaved_raw_v4_user_present_observed_pcm_preanalysis"
OFFICIAL_SAMPLE_RATE = OFFICIAL_MEASUREMENT_LEVEL.sample_rate
OFFICIAL_CHANNEL_MAP = OFFICIAL_MEASUREMENT_CHANNEL_MAP
SEPARATION_ALGORITHM = "fractional_clock_joint_real_ls"
SEPARATION_ALGORITHM_VERSION = 1
CLOCK_MIN_ADJACENT_SCORE = 0.995
CLOCK_MAX_ERR_REF_DELTA_SAMPLES = 0.25
CLOCK_BAND_HZ = OFFICIAL_MEASUREMENT_LEVEL.meter_band_hz
CLOCK_MAX_SUBWINDOW_SPREAD_SAMPLES = 0.35
CLOCK_MAX_ADJACENT_CHANGE_SAMPLES = 0.50
CLOCK_MAX_ABS_PERIOD_DELTA_SAMPLES = 6.0
JOINT_LS_MAX_CONDITION = 1.25
JOINT_LS_MAX_RESIDUAL_P95 = 0.05
SEPARATION_CROSSCHECK_MIN_AGREEMENT = 0.999
SEPARATION_CROSSCHECK_MAX_RELATIVE_ERROR = 0.01

# 설계 대역은 필수 대역 [80,1600] 보다 넓게 잡는다. 채널마다 톤이 한 칸씩 어긋나므로
# 딱 맞춰 잡으면 한 채널의 마지막 톤이 상한 안쪽으로 떨어져 대역을 덮지 못한다.
DEFAULT_BAND_HZ = OFFICIAL_MEASUREMENT_LEVEL.design_band_hz
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
DEFAULT_PERIOD_SECONDS = OFFICIAL_MEASUREMENT_LEVEL.period_seconds
# 워밍업 4주기(0.5s)로는 스트림이 정상상태에 못 든다 — 2026-08-05 실측에서 저장된
# 캡처 9건 **전부** 반복 0(대개 반복 1도)의 국소 타임베이스 드리프트가 정상상태의
# 2~5배였다. 검증 캡처와 같은 32주기(4.0s)를 기본으로 둔다.
DEFAULT_WARMUP_PERIODS = OFFICIAL_MEASUREMENT_LEVEL.warmup_periods
# 게이트가 반복을 버리므로 여유를 둔다. 저장된 캡처에서 유지율은 33~75% 였고
# min_kept_repeats 8 을 넉넉히 확보하고 독립 부대역 평균을 안정화하려 64를 쓴다.
# 재생 길이는 0.125s×(32+64) = 12.0초다.
DEFAULT_REPEATS = OFFICIAL_MEASUREMENT_LEVEL.analysis_repeats
DEFAULT_AMPLITUDE = OFFICIAL_MEASUREMENT_LEVEL.probe_amplitude
"""P/S 측정·레벨 미터 공용 peak. 안전 상한 ``cw.MAX_AMPLITUDE``와 별개다."""
# τ 는 재현되는 대역에서만 적합한다. 재현 안 되는 대역을 넣으면 그 잡음이 τ 를 끌고 간다.
DEFAULT_FIT_BAND_HZ = (150.0, 1200.0)
# 일관성을 **어느 대역에서 쟀는지**가 곧 이 모델을 어느 대역에서 믿을 수 있는가다.
# 그래서 숫자만 저장하지 않고 대역도 함께 저장하고, 게이트가 그 대역이 요구 대역을
# 덮는지 검사한다. 대역이 안 적혀 있으면 0.95 라는 숫자가 무엇에 대한 0.95 인지 모른다.
#
# 2026-08-05 정정: [150,600] 은 "600Hz 위는 덕트 물리 한계" 라는 **틀린 전제** 위에
# 있었다. 아래 게이트를 켜고 재분석하면 1000-1600Hz 일관성이 P 0.999 / S 0.999 다.
DEFAULT_CONSISTENCY_BAND_HZ = OFFICIAL_MEASUREMENT_LEVEL.meter_band_hz

OUTPUT_LEAD_IN_SECONDS = 0.5
SPEAKER_DISCONNECT_NOTICE = (
    "[스피커 출력 종료] 오디오 스트림을 닫았습니다. "
    "지금 스피커/앰프 연결을 즉시 해제하세요. 이후는 무음 저장·분석만 진행합니다."
)
SPEAKER_STOP_UNCONFIRMED_NOTICE = (
    "[스피커 정지 확인 불가] 오디오 스트림 close를 확인하지 못했습니다. "
    "소프트웨어 상태를 기다리지 말고 지금 스피커/앰프를 즉시 물리 분리하세요."
)

MIN_TONE_SNR_DB = 12.0          # 톤 중앙값 SNR 하한
MIN_TONE_SNR_FRACTION = 0.9     # 이 비율 이상의 톤이 하한을 넘어야 한다
MIN_TONE_SNR_MEDIAN_DB = 30.0   # v1/v2 importer가 요구하는 official 중앙값 하한
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

MIN_BAND_CONSISTENCY = 0.95
"""필수 대역 안 **모든** 부대역에 적용하는 interleaved 전용 하한이다."""

MIN_INTERLEAVED_CONSISTENCY = 0.95
"""전대역 반복 일관성 하한. importer/readiness와 같은 0.95를 쓴다."""

OFFICIAL_BLOCK_SIZE = 256
OFFICIAL_LATENCY = "low"

DELAY_SEMANTICS = "effective_zeros_before_compact_fir"
"""official ``delay_samples`` 는 compact FIR 앞에 실제로 붙일 0의 개수다."""

MIN_COMPACT_TRANSFER_AGREEMENT = 0.995
MAX_COMPACT_TRANSFER_RELATIVE_ERROR = 0.10
"""후처리된 FIR+delay와 정렬 평균 복소 전달함수의 출하 직전 round-trip gate.

fresh_rir_A를 올바르게 변환하면 P/S agreement 0.9990/0.9992, 상대오차
0.0438/0.0408이다. 결함이 있던 odd-bin S는 0.580/0.815였으므로 이 한계는 정상
절단 오차에는 2배 이상 여유를 주면서 같은 유형의 후처리 손상을 확실히 거부한다.
"""

COMPACT_TRANSFER_SUB_BANDS_HZ = (
    (150.0, 300.0),
    (300.0, 600.0),
    (600.0, 1000.0),
    (1000.0, 1600.0),
)

SEPARATION_CROSSCHECK_OVERALL_BAND_HZ = (150.0, 1600.0)
"""strict v1 joint-LS/cubic 독립 교차검증의 전대역 기본값."""

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
    parser.add_argument(
        "--required-band",
        type=float,
        nargs=2,
        default=list(OFFICIAL_MEASUREMENT_LEVEL.meter_band_hz),
    )
    parser.add_argument(
        "--tone-spacing-hz",
        type=float,
        default=DEFAULT_TONE_SPACING_HZ,
        help="채널별 톤 간격(Hz). 생략하면 guard=1 이 되는 최소 간격을 쓴다",
    )
    parser.add_argument("--period-seconds", type=float, default=DEFAULT_PERIOD_SECONDS)
    parser.add_argument("--warmup-periods", type=int, default=DEFAULT_WARMUP_PERIODS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--amplitude", type=float, default=DEFAULT_AMPLITUDE)
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
    # 런타임(run_realtime / record_duct)은 'low' 로 **고정**돼 있다
    # (configs/hardware_jetson.yaml: "코드는 'low' 고정"). 기본값이 'high' 였던 탓에
    # 측정하는 플랜트와 실제로 도는 플랜트의 버퍼 구성이 달랐다 — 실측 절대지연이
    # low 1565~1659 / high 2858~2888 로 갈린다. 출하 npz(calibration_latency=low)도
    # low 에서 나왔다. 배포와 다른 모드로 잰 플랜트는 그 자체로 fail-open 이다.
    parser.add_argument("--latency", choices=["low", "high"], default="low")
    parser.add_argument(
        "--input-probe-seconds",
        type=float,
        default=OFFICIAL_MEASUREMENT_LEVEL.input_probe_seconds,
    )
    parser.add_argument(
        "--level-evidence",
        default=str(DEFAULT_MEASUREMENT_LEVEL_EVIDENCE_PATH),
        help=(
            "peak 0.003 meter↔interleaved paired raw 증거 JSON. 실제 오디오 실행은 "
            "이 증거와 raw SHA가 유효할 때만 허용한다"
        ),
    )
    parser.add_argument(
        "--bootstrap-level-evidence",
        action="store_true",
        help=(
            "canonical evidence가 없을 때만 쓰는 1회 paired 흐름. 바로 직전 PASS "
            "meter raw와 이 12.5초 strict raw로 evidence를 원자 생성한다"
        ),
    )
    parser.add_argument(
        "--meter-raw",
        "--bootstrap-meter-raw",
        dest="meter_raw",
        default=None,
        help=(
            "set_amp_level.py가 방금 저장한 fresh PASS meter_raw.npz. "
            "이전 --bootstrap-meter-raw 이름도 호환 alias로만 허용한다"
        ),
    )
    parser.add_argument(
        "--confirm-same-amplifier-setting",
        action="store_true",
        help="meter raw 이후 앰프 노브 설정을 바꾸지 않았음을 운영자가 확인",
    )
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "설정·probe·출력 경로 freshness·입출력 장치 매핑만 검증하고 종료한다. "
            "오디오 스트림, 스피커, 진단 session/파일을 만들지 않는다"
        ),
    )
    parser.add_argument(
        "--confirm-user-present",
        action="store_true",
        help="strict 출력 동안 사용자가 입회함을 명시적으로 확인",
    )
    parser.add_argument("--confirm-volume-minimum", action="store_true")
    parser.add_argument(
        "--confirm-routing-and-geometry",
        action="store_true",
        help=(
            "ERR/REF와 noise/cancel 물리 배선 및 덕트 기하가 hardware YAML의 "
            "0/1 채널 맵과 일치함을 운영자가 직접 확인했다는 필수 확인"
        ),
    )
    return parser


def validate_hardware_contract(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    """official 측정의 48k/256/low 및 논리 채널 맵을 재생 전에 고정한다."""
    return validate_measurement_hardware_contract(config)


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


def compact_transfer_round_trip(
    frequencies_hz: np.ndarray,
    measured_transfer: np.ndarray,
    fir: np.ndarray,
    *,
    effective_delay_samples: int,
    sample_rate: int,
    band_hz: tuple[float, float],
    required_subbands_hz: tuple[
        tuple[float, float], ...
    ] = COMPACT_TRANSFER_SUB_BANDS_HZ,
) -> dict[str, Any]:
    """출하될 ``zeros(delay)+FIR`` 이 측정 복소 전달함수를 재현하는지 검사한다."""

    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    measured = np.asarray(measured_transfer, dtype=np.complex128).reshape(-1)
    taps = np.asarray(fir, dtype=np.float64).reshape(-1)
    if frequencies.size != measured.size or frequencies.size == 0:
        raise ValueError("compact transfer round-trip 입력 길이가 맞지 않습니다")
    if taps.size == 0 or not np.isfinite(taps).all():
        raise ValueError("compact transfer round-trip FIR이 비었거나 non-finite입니다")
    if effective_delay_samples < 0 or sample_rate <= 0:
        raise ValueError("effective delay/sample rate가 잘못되었습니다")
    sample_indices = np.arange(taps.size, dtype=np.float64) + float(
        effective_delay_samples
    )

    def _metrics(bounds: tuple[float, float]) -> dict[str, Any]:
        mask = (frequencies >= float(bounds[0])) & (
            frequencies <= float(bounds[1])
        )
        if int(mask.sum()) < 8:
            raise ValueError(
                f"compact transfer round-trip {bounds[0]:.0f}-{bounds[1]:.0f}Hz "
                f"톤이 부족합니다: {int(mask.sum())}개"
            )
        reconstructed = np.exp(
            -2j
            * np.pi
            * np.outer(frequencies[mask], sample_indices)
            / float(sample_rate)
        ) @ taps
        target = measured[mask]
        target_norm = float(np.linalg.norm(target))
        reconstructed_norm = float(np.linalg.norm(reconstructed))
        if target_norm <= 0.0 or reconstructed_norm <= 0.0:
            raise ValueError("compact transfer round-trip 전달함수 energy가 0입니다")
        agreement = float(
            abs(complex(np.vdot(target, reconstructed)))
            / (target_norm * reconstructed_norm)
        )
        relative_error = float(np.linalg.norm(reconstructed - target) / target_norm)
        return {
            "band_hz": [float(bounds[0]), float(bounds[1])],
            "tone_count": int(mask.sum()),
            "complex_agreement": agreement,
            "relative_error": relative_error,
            "passed": bool(
                np.isfinite(agreement)
                and np.isfinite(relative_error)
                and agreement >= MIN_COMPACT_TRANSFER_AGREEMENT
                and relative_error <= MAX_COMPACT_TRANSFER_RELATIVE_ERROR
            ),
        }

    required_subbands = tuple(
        (float(bounds[0]), float(bounds[1])) for bounds in required_subbands_hz
    )
    if not required_subbands or any(lo >= hi for lo, hi in required_subbands):
        raise ValueError("compact transfer 필수 부대역이 비었거나 잘못되었습니다")

    overall = _metrics((float(band_hz[0]), float(band_hz[1])))
    subbands = [
        _metrics(bounds)
        for bounds in required_subbands
        if bounds[0] >= float(band_hz[0]) and bounds[1] <= float(band_hz[1])
    ]
    if len(subbands) != len(required_subbands):
        raise ValueError(
            f"compact gate 대역 {band_hz}가 필수 부대역 "
            f"{required_subbands} 전체를 덮지 못합니다"
        )
    passed = bool(overall["passed"] and all(item["passed"] for item in subbands))
    return {
        "passed": passed,
        "band_hz": overall["band_hz"],
        "tone_count": overall["tone_count"],
        "complex_agreement": overall["complex_agreement"],
        "relative_error": overall["relative_error"],
        "subbands": subbands,
        "minimum_complex_agreement": MIN_COMPACT_TRANSFER_AGREEMENT,
        "maximum_relative_error": MAX_COMPACT_TRANSFER_RELATIVE_ERROR,
    }


def aligned_transfer_sha256(
    frequencies_hz: np.ndarray,
    transfer: np.ndarray,
) -> str:
    """독립 재감사용 source 배열을 dtype/endianness까지 고정해 digest한다."""

    frequencies = np.ascontiguousarray(
        np.asarray(frequencies_hz, dtype="<f8").reshape(-1)
    )
    values = np.asarray(transfer, dtype=np.complex128).reshape(-1)
    if frequencies.size != values.size or frequencies.size == 0:
        raise ValueError("aligned transfer digest 입력 길이가 맞지 않습니다")
    real = np.ascontiguousarray(values.real.astype("<f8", copy=False))
    imag = np.ascontiguousarray(values.imag.astype("<f8", copy=False))
    digest = hashlib.sha256()
    digest.update(frequencies.tobytes(order="C"))
    digest.update(real.tobytes(order="C"))
    digest.update(imag.tobytes(order="C"))
    return digest.hexdigest()


def validate_official_capture_recipe_args(args: argparse.Namespace) -> None:
    """probe를 만들기 전 duration/level/band recipe를 exact 고정한다."""

    exact_float = {
        "period-seconds": (
            float(args.period_seconds),
            OFFICIAL_MEASUREMENT_LEVEL.period_seconds,
        ),
        "input-probe-seconds": (
            float(args.input_probe_seconds),
            OFFICIAL_MEASUREMENT_LEVEL.input_probe_seconds,
        ),
        "amplitude": (
            float(args.amplitude),
            OFFICIAL_MEASUREMENT_LEVEL.probe_amplitude,
        ),
    }
    for name, (observed, expected) in exact_float.items():
        if not np.isfinite(observed) or not np.isclose(
            observed, expected, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f"official --{name}는 {expected}로 고정입니다: {observed!r}"
            )
    exact_int = {
        "warmup-periods": (
            int(args.warmup_periods),
            OFFICIAL_MEASUREMENT_LEVEL.warmup_periods,
        ),
        "repeats": (
            int(args.repeats),
            OFFICIAL_MEASUREMENT_LEVEL.analysis_repeats,
        ),
    }
    for name, (observed, expected) in exact_int.items():
        if observed != expected:
            raise ValueError(
                f"official --{name}는 {expected}로 고정입니다: {observed!r}"
            )
    exact_bands = {
        "band": OFFICIAL_MEASUREMENT_LEVEL.design_band_hz,
        "required-band": OFFICIAL_MEASUREMENT_LEVEL.meter_band_hz,
        "consistency-band": OFFICIAL_MEASUREMENT_LEVEL.meter_band_hz,
    }
    for name, expected in exact_bands.items():
        observed = np.asarray(getattr(args, name.replace("-", "_")), dtype=np.float64)
        if observed.shape != (2,) or not np.all(np.isfinite(observed)) or not np.allclose(
            observed, expected, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f"official --{name}는 {expected}로 고정입니다: {observed.tolist()!r}"
            )
    if args.tone_spacing_hz is not None:
        raise ValueError("official --tone-spacing-hz는 생략(None)해야 합니다")


def validate_analysis_contract(
    args: argparse.Namespace, probe, *, fs: int, block_size: int
) -> dict[str, tuple[float, float]]:
    """재생 전/dry-run이 공유하는 분석 가능성 검증. 하나라도 불명확하면 실패한다."""

    validate_official_capture_recipe_args(args)
    nyquist = float(fs) / 2.0
    bands: dict[str, tuple[float, float]] = {}
    for name, raw in {
        "band": args.band,
        "required-band": args.required_band,
        "fit-band": args.fit_band,
        "consistency-band": args.consistency_band,
    }.items():
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        if values.size != 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"--{name}는 유한한 [low, high] 두 값이어야 합니다: {raw}")
        lo, hi = float(values[0]), float(values[1])
        if not 0.0 < lo < hi < nyquist:
            raise ValueError(f"--{name}가 유효하지 않습니다: {(lo, hi)}")
        bands[name] = (lo, hi)

    required = bands["required-band"]
    fit = bands["fit-band"]
    consistency = bands["consistency-band"]
    if required[0] > 150.0 or required[1] < 1600.0:
        raise ValueError("--required-band는 official 고정 대역 150-1600Hz를 덮어야 합니다")
    if consistency[0] > required[0] or consistency[1] < required[1]:
        raise ValueError("--consistency-band가 --required-band 전체를 덮어야 합니다")
    if consistency[0] > 150.0 or consistency[1] < 1600.0:
        raise ValueError("--consistency-band는 compact gate의 150-1600Hz를 덮어야 합니다")

    channel_band: dict[str, tuple[float, float]] = {}
    resolution = float(fs) / float(probe.period_samples)
    for drive in ("noise", "cancel"):
        frequencies = probe.bins_for(drive).astype(np.float64) * resolution
        channel_band[drive] = (float(frequencies[0]), float(frequencies[-1]))
        for name, bounds in {
            "required-band": required,
            "fit-band": fit,
            "consistency-band": consistency,
        }.items():
            mask = (frequencies >= bounds[0]) & (frequencies <= bounds[1])
            if frequencies[0] > bounds[0] or frequencies[-1] < bounds[1]:
                raise ValueError(
                    f"{drive} 톤 대역 {channel_band[drive]}가 --{name} {bounds}를 "
                    "덮지 못합니다"
                )
            if int(mask.sum()) < 8:
                raise ValueError(
                    f"{drive} --{name} 안의 톤이 {int(mask.sum())}개뿐입니다"
                )
        for lo, hi in COMPACT_TRANSFER_SUB_BANDS_HZ:
            count = int(((frequencies >= lo) & (frequencies <= hi)).sum())
            if count < 8:
                raise ValueError(
                    f"{drive} compact 부대역 {lo:.0f}-{hi:.0f}Hz 톤이 "
                    f"{count}개뿐입니다"
                )

    unambiguous = min(
        probe.period_samples // probe.bin_step(drive)
        for drive in ("noise", "cancel")
    )
    if int(args.fir_length) <= 0 or int(args.fir_length) > unambiguous:
        raise ValueError(
            f"--fir-length는 1..{unambiguous} 이어야 합니다: {args.fir_length}"
        )
    if int(args.pre_roll) < 0 or int(args.pre_roll) >= unambiguous:
        raise ValueError(
            f"--pre-roll은 0..{unambiguous - 1} 이어야 합니다: {args.pre_roll}"
        )
    if int(args.pre_roll) >= int(args.fir_length):
        raise ValueError("--pre-roll은 --fir-length보다 작아야 합니다")

    finite_positive = {
        "period-seconds": args.period_seconds,
        "max-delay-ms": args.max_delay_ms,
        "input-probe-seconds": args.input_probe_seconds,
        "track-min-peak": args.track_min_peak,
    }
    if args.tone_spacing_hz is not None:
        finite_positive["tone-spacing-hz"] = args.tone_spacing_hz
    for name, value in finite_positive.items():
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"--{name}는 유한한 양수여야 합니다: {value}")
    for name, value in {
        "max-delay-jitter-ms": args.max_delay_jitter_ms,
        "max-relative-tau-samples": args.max_relative_tau_samples,
        "max-drift-deviation-samples": args.max_drift_deviation_samples,
    }.items():
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"--{name}는 유한한 0 이상이어야 합니다: {value}")

    max_delay_samples = int(round(float(args.max_delay_ms) / 1000.0 * fs))
    if max_delay_samples <= int(args.pre_roll):
        raise ValueError(
            f"--max-delay-ms는 pre-roll보다 큰 탐색 범위를 만들어야 합니다: "
            f"{max_delay_samples} <= {args.pre_roll} samples"
        )
    jitter_samples = int(round(float(args.max_delay_jitter_ms) / 1000.0 * fs))
    if not 0 <= jitter_samples <= 3:
        raise ValueError("--max-delay-jitter-ms는 0..3 samples 범위여야 합니다")
    if int(block_size) != OFFICIAL_BLOCK_SIZE:
        raise ValueError(
            f"official block-size는 runtime과 같은 {OFFICIAL_BLOCK_SIZE}여야 합니다: "
            f"{block_size}"
        )
    if str(args.latency) != OFFICIAL_LATENCY:
        raise ValueError(
            f"official latency는 runtime과 같은 {OFFICIAL_LATENCY!r}여야 합니다: "
            f"{args.latency!r}"
        )
    alignment = float(args.min_alignment_score)
    if not np.isfinite(alignment) or not DEFAULT_MIN_ALIGNMENT_SCORE <= alignment <= 1.0:
        raise ValueError(
            f"--min-alignment-score는 {DEFAULT_MIN_ALIGNMENT_SCORE}..1 범위여야 합니다"
        )
    if float(args.max_relative_tau_samples) > DEFAULT_MAX_RELATIVE_TAU_SAMPLES:
        raise ValueError(
            f"--max-relative-tau-samples는 {DEFAULT_MAX_RELATIVE_TAU_SAMPLES} 이하여야 합니다"
        )
    if float(args.max_drift_deviation_samples) > DEFAULT_MAX_DRIFT_DEVIATION_SAMPLES:
        raise ValueError(
            "--max-drift-deviation-samples는 "
            f"{DEFAULT_MAX_DRIFT_DEVIATION_SAMPLES} 이하여야 합니다"
        )
    if int(args.warmup_periods) < 1:
        raise ValueError("--warmup-periods는 1 이상이어야 합니다")
    if int(args.repeats) < cw.MIN_REPEATS:
        raise ValueError(f"--repeats는 {cw.MIN_REPEATS} 이상이어야 합니다")
    if not 8 <= int(args.min_kept_repeats) <= int(args.repeats):
        raise ValueError("--min-kept-repeats는 8 이상이고 --repeats 이하여야 합니다")
    if int(args.track_window) <= 0 or not 0.0 < float(args.track_min_peak) <= 1.0:
        raise ValueError("track-window/track-min-peak가 유효하지 않습니다")
    if args.dewarp:
        raise ValueError("--dewarp는 official 측정에서 금지됩니다")
    return channel_band


def validate_alsa_pcm_mapping(
    *, input_card: int, input_pcm: int, output_card: int, output_pcm: int
) -> None:
    """스트림을 열지 않고 ALSA PCM의 방향별 procfs node 존재만 확인한다."""

    nodes = {
        "input": Path(f"/proc/asound/card{input_card}/pcm{input_pcm}c"),
        "output": Path(f"/proc/asound/card{output_card}/pcm{output_pcm}p"),
    }
    missing = [f"{direction}={path}" for direction, path in nodes.items() if not path.is_dir()]
    if missing:
        raise RuntimeError("ALSA PCM 방향 매핑을 찾지 못했습니다: " + ", ".join(missing))


def _fsync_directory(path: Path) -> None:
    """같은 디렉터리 안 rename 자체도 전원 손실 뒤 남도록 directory를 fsync한다."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_session_directory(diagnostics_root: Path, name: str) -> Path:
    """필요한 diagnostics 조상과 새 세션 entry를 차례로 durable하게 만든다."""

    diagnostics_root = Path(diagnostics_root)
    missing: list[Path] = []
    cursor = diagnostics_root
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    if cursor.exists() and not cursor.is_dir():
        raise NotADirectoryError(f"diagnostics 조상이 디렉터리가 아닙니다: {cursor}")
    for directory in reversed(missing):
        directory.mkdir(exist_ok=False)
        _fsync_directory(directory.parent)

    session_dir = diagnostics_root / str(name)
    session_dir.mkdir(exist_ok=False)
    _fsync_directory(diagnostics_root)
    return session_dir


class RawCaptureSidecarError(OSError):
    """canonical raw NPZ는 보존됐지만 metadata sidecar 승격이 실패한 상태."""

    def __init__(self, raw_path: Path, cause: BaseException):
        self.raw_path = Path(raw_path)
        self.cause = cause
        super().__init__(
            f"metadata sidecar 저장 실패; canonical raw NPZ는 보존됐습니다: "
            f"{self.raw_path}. embedded metadata_json으로 재분석할 수 있습니다: {cause}"
        )


class DeferredTerminationSignal(BaseException):
    """raw commit 동안 받은 SIGINT/SIGTERM을 durable 승격 뒤 전달한다."""

    def __init__(self, signum: int):
        self.signum = int(signum)
        self.exit_code = 128 + self.signum
        super().__init__(f"signal {self.signum} deferred until raw commit completed")


@contextmanager
def defer_termination_signals_during_raw_commit():
    """interactive termination을 raw NPZ/sidecar 승격 뒤까지 잠시 지연한다.

    SIGKILL, 전원 손실, 디스크 오류는 막을 수 없다. 이 gate는 사용자가 캡처 직후
    Ctrl-C를 눌러 유일한 in-memory raw가 압축 중 사라지는 경우만 닫는다.
    """

    watched = tuple(
        value
        for value in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None))
        if value is not None
    )
    previous = {value: signal.getsignal(value) for value in watched}
    pending: list[int] = []

    def remember(signum, _frame):
        pending.append(int(signum))

    for value in watched:
        signal.signal(value, remember)
    active_error = False
    try:
        yield
    except BaseException:
        active_error = True
        raise
    finally:
        for value, handler in previous.items():
            signal.signal(value, handler)
        if pending and not active_error:
            raise DeferredTerminationSignal(pending[0])


class PartialCaptureError(RuntimeError):
    """일부 스피커 출력 뒤 중단되어 raw 보존이 필요한 캡처 실패."""

    def __init__(
        self,
        cause: BaseException,
        *,
        recorded_raw: np.ndarray,
        output_pcm: np.ndarray,
        telemetry: dict[str, Any],
    ):
        self.cause = cause
        self.recorded_raw = recorded_raw
        self.output_pcm = output_pcm
        self.telemetry = telemetry
        super().__init__(f"{type(cause).__name__}: {cause}")


_CaptureResult = TypeVar("_CaptureResult")


def measurement_duration_plan(
    args: Any, *, sample_rate: int, period_samples: int
) -> dict[str, float]:
    """dry-run과 실제 실행이 공유하는 스피커 연결 시간 계산."""

    if sample_rate <= 0 or period_samples <= 0:
        raise ValueError("duration plan의 sample rate/period samples는 양수여야 합니다")
    lead_in_samples = int(round(OUTPUT_LEAD_IN_SECONDS * sample_rate))
    stimulus_seconds = (
        (int(args.warmup_periods) + int(args.repeats))
        * int(period_samples)
        / float(sample_rate)
    )
    return {
        "input_preflight_seconds": float(args.input_probe_seconds),
        "lead_in_seconds": lead_in_samples / float(sample_rate),
        "stimulus_seconds": stimulus_seconds,
        "output_stream_seconds": lead_in_samples / float(sample_rate)
        + stimulus_seconds,
        "output_hard_max_seconds": lead_in_samples / float(sample_rate)
        + stimulus_seconds
        + LIVE_WATCHDOG_GRACE_SECONDS,
        "watchdog_grace_seconds": LIVE_WATCHDOG_GRACE_SECONDS,
    }


def announce_speaker_disconnect(*, output_stop_confirmed: bool = True) -> None:
    """캡처 스트림 종료를 분석보다 먼저 운영자에게 알린다."""

    notice = (
        SPEAKER_DISCONNECT_NOTICE
        if output_stop_confirmed
        else SPEAKER_STOP_UNCONFIRMED_NOTICE
    )
    print(notice, flush=True)


def capture_with_speaker_release_notice(
    capture: Callable[[], _CaptureResult],
) -> _CaptureResult:
    """성공/실패 모두 스트림 반환 직후 분리 안내를 보장한다."""

    try:
        result = capture()
    except BaseException as exc:
        confirmed = not (
            isinstance(exc, PartialCaptureError)
            and exc.telemetry.get("output_stop_confirmed") is False
        )
        announce_speaker_disconnect(output_stop_confirmed=confirmed)
        raise
    announce_speaker_disconnect(output_stop_confirmed=True)
    return result


def capture_measurement_preserving_partial(
    sd: Any,
    *,
    fs: int,
    block_size: int,
    latency: str,
    in_dev: int,
    out_dev: int,
    output_float: np.ndarray,
    meter_completed_at_utc: dt.datetime | None = None,
    pre_open_check: Callable[[], None] | None = None,
    record_callback_time_info: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """interleaved 전용 full-duplex; 실패도 partial arrays/telemetry를 surface한다."""

    if output_float.ndim != 2 or output_float.shape[1] != 2:
        raise ValueError("output_float는 [frames,2]여야 합니다")
    total = int(output_float.shape[0])
    output_pcm = cw.float32_to_pcm_int16(output_float)
    recorded_raw = np.zeros((total, 2), dtype=np.int32)
    cursor = {"frames": 0}
    timing_capacity = int(math.ceil(total / max(1, int(block_size)))) + 4
    timing_arrays = {
        "callback_start_frames": np.full(timing_capacity, -1, dtype=np.int64),
        "callback_frame_counts": np.zeros(timing_capacity, dtype=np.int64),
        "input_buffer_adc_time": np.full(timing_capacity, np.nan, dtype=np.float64),
        "output_buffer_dac_time": np.full(timing_capacity, np.nan, dtype=np.float64),
        "callback_current_time": np.full(timing_capacity, np.nan, dtype=np.float64),
    }
    telemetry: dict[str, Any] = {
        "callback_count": 0,
        "callback_status_count": 0,
        "xrun_count": 0,
        "priming_output_count": 0,
        "unexpected_status_count": 0,
        "statuses": [],
        "callback_error": None,
        "completed": False,
        "stream_started_at_utc": None,
        "termination_signal": None,
        "nominal_output_seconds": total / float(fs),
        "hard_max_output_seconds": (
            total / float(fs) + LIVE_WATCHDOG_GRACE_SECONDS
        ),
    }

    def callback(indata, outdata, frames, _time_info, status):
        outdata.fill(0)
        try:
            telemetry["callback_count"] += 1
            timing_index = int(telemetry["callback_count"]) - 1
            if record_callback_time_info:
                if timing_index >= timing_capacity:
                    raise RuntimeError("callback timing witness capacity를 넘었습니다")

                def time_value(name: str) -> float:
                    if isinstance(_time_info, dict):
                        value = _time_info.get(name, float("nan"))
                    else:
                        value = getattr(_time_info, name, float("nan"))
                    return float(value)

                timing_arrays["callback_start_frames"][timing_index] = int(
                    cursor["frames"]
                )
                timing_arrays["callback_frame_counts"][timing_index] = int(frames)
                timing_arrays["input_buffer_adc_time"][timing_index] = time_value(
                    "inputBufferAdcTime"
                )
                timing_arrays["output_buffer_dac_time"][timing_index] = time_value(
                    "outputBufferDacTime"
                )
                timing_arrays["callback_current_time"][timing_index] = time_value(
                    "currentTime"
                )
            if status:
                item = cw._status_snapshot(status)
                telemetry["callback_status_count"] += 1
                telemetry["xrun_count"] += int(item["is_xrun"])
                telemetry["priming_output_count"] += int(item["priming_output"])
                telemetry["unexpected_status_count"] += int(item["unexpected"])
                telemetry["statuses"].append(item)
            start = int(cursor["frames"])
            count = min(int(frames), total - start)
            if count > 0:
                recorded_raw[start : start + count] = np.asarray(
                    indata[:count, :2], dtype=np.int32
                )
                outdata[:count] = output_pcm[start : start + count]
                cursor["frames"] = start + count
            if cursor["frames"] >= total:
                telemetry["completed"] = True
                raise sd.CallbackStop
        except sd.CallbackStop:
            raise
        except Exception as exc:
            outdata.fill(0)
            telemetry["callback_error"] = f"{type(exc).__name__}: {exc}"
            raise sd.CallbackAbort

    stream = None
    failure: BaseException | None = None
    call_started = time.monotonic()
    stream_started: float | None = None
    deadline: float | None = None
    try:
        with scoped_live_audio_signal_handlers():
            if meter_completed_at_utc is not None:
                now_utc = dt.datetime.now(dt.timezone.utc)
                completed = meter_completed_at_utc.astimezone(dt.timezone.utc)
                age = (now_utc - completed).total_seconds()
                if (
                    age < -BOOTSTRAP_CLOCK_FUTURE_TOLERANCE_SECONDS
                    or age > BOOTSTRAP_METER_MAX_AGE_SECONDS
                ):
                    raise RuntimeError(
                        "stream start 직전 fresh meter age 계약 위반: "
                        f"{age:.1f}s (max {BOOTSTRAP_METER_MAX_AGE_SECONDS}s)"
                    )
            if pre_open_check is not None:
                pre_open_check()
            stream = sd.Stream(
                samplerate=fs,
                blocksize=block_size,
                device=(in_dev, out_dev),
                channels=(2, 2),
                dtype=("int32", "int16"),
                latency=(latency, latency),
                callback=callback,
                prime_output_buffers_using_stream_callback=True,
            )
            stream.start()
            stream_started = time.monotonic()
            telemetry["stream_started_at_utc"] = dt.datetime.now(
                dt.timezone.utc
            ).isoformat()
            deadline = stream_started + total / fs + LIVE_WATCHDOG_GRACE_SECONDS
            while not telemetry["completed"]:
                if telemetry["callback_error"] is not None:
                    raise RuntimeError(f"오디오 콜백 실패: {telemetry['callback_error']}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("오디오 hard-max watchdog을 넘었습니다")
                time.sleep(0.01)
    except LiveAudioTermination as exc:
        telemetry["termination_signal"] = int(exc.signum)
        failure = exc
    except BaseException as exc:
        failure = exc

    abort_error: BaseException | None = None
    close_error: BaseException | None = None
    if stream is not None:
        try:
            stream.abort()
        except BaseException as exc:
            abort_error = exc
        try:
            stream.close()
        except BaseException as exc:
            close_error = exc
    telemetry["stream_abort_error"] = (
        None if abort_error is None else f"{type(abort_error).__name__}: {abort_error}"
    )
    telemetry["stream_close_error"] = (
        None if close_error is None else f"{type(close_error).__name__}: {close_error}"
    )
    # close가 실패하면 abort 성공 여부와 관계없이 PortAudio/driver의 최종 상태를
    # 확인할 수 없다. 성공처럼 안내하거나 telemetry를 삼키지 않는다.
    telemetry["output_stop_confirmed"] = bool(
        stream is None or close_error is None
    )
    telemetry["captured_frames"] = int(cursor["frames"])
    if record_callback_time_info:
        count = int(telemetry["callback_count"])
        telemetry["callback_time_info"] = {
            name: np.asarray(values[:count]).copy()
            for name, values in timing_arrays.items()
        }
    telemetry["elapsed_seconds"] = float(time.monotonic() - call_started)
    telemetry["output_elapsed_seconds"] = (
        None
        if stream_started is None
        else float(time.monotonic() - stream_started)
    )
    cleanup_failures = [
        value for value in (abort_error, close_error) if value is not None
    ]
    if failure is not None or cleanup_failures:
        details: list[str] = []
        if failure is not None:
            details.append(f"capture={type(failure).__name__}: {failure}")
        if abort_error is not None:
            details.append(f"abort={type(abort_error).__name__}: {abort_error}")
        if close_error is not None:
            details.append(f"close={type(close_error).__name__}: {close_error}")
        cause = RuntimeError("; ".join(details))
        raise PartialCaptureError(
            cause,
            recorded_raw=recorded_raw,
            output_pcm=output_pcm,
            telemetry=telemetry,
        ) from failure
    return recorded_raw, output_pcm, telemetry


def write_immutable_raw_capture_atomic(
    session_dir: Path,
    *,
    metadata: dict[str, Any],
    arrays: dict[str, Any],
) -> dict[str, Path]:
    """유일한 캡처를 same-dir temp에서 완성한 뒤 raw NPZ부터 원자 승격한다.

    NPZ 안 ``metadata_json``이 canonical 복구원이다. sidecar 단계가 실패해도 이미
    승격·fsync한 NPZ는 지우지 않으며 ``RawCaptureSidecarError``로 그 상태를 알린다.
    """

    raw_path = session_dir / "raw_measurement.npz"
    metadata_path = session_dir / "metadata.json"
    for path in (raw_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"기존 raw capture는 덮어쓰지 않습니다: {path}")
    if "metadata_json" in arrays:
        raise ValueError("raw arrays에 reserved metadata_json key를 넣을 수 없습니다")
    safe_metadata = cw._json_safe(metadata)
    canonical_json = json.dumps(
        safe_metadata, ensure_ascii=False, sort_keys=True
    )
    token = uuid.uuid4().hex
    raw_temp = session_dir / f".raw_measurement.{token}.partial"
    metadata_temp = session_dir / f".metadata.{token}.partial"

    try:
        with raw_temp.open("xb") as handle:
            np.savez_compressed(
                handle,
                metadata_json=np.asarray(canonical_json),
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        atomic_publish_noreplace(raw_temp, raw_path)
        _fsync_directory(session_dir)
    except BaseException:
        raw_temp.unlink(missing_ok=True)
        # replace 전 실패면 final은 없어야 한다. replace 뒤 directory fsync 실패면
        # canonical raw가 유일한 복구본이므로 절대로 삭제하지 않는다.
        raise

    try:
        with metadata_temp.open("x", encoding="utf-8") as handle:
            json.dump(
                safe_metadata,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        atomic_publish_noreplace(metadata_temp, metadata_path)
        _fsync_directory(session_dir)
    except BaseException as exc:
        metadata_temp.unlink(missing_ok=True)
        # raw_path는 canonical recovery source다. sidecar 실패 때문에 지우면 안 된다.
        raise RawCaptureSidecarError(raw_path, exc) from exc
    return {"raw": raw_path, "metadata": metadata_path}


def write_analysis_outputs_atomic(
    session_dir: Path,
    *,
    metadata: dict[str, Any],
    arrays: dict[str, Any],
    suffix: str = "",
) -> dict[str, Path]:
    """원시 캡처를 건드리지 않고 분석 쌍을 완성본 temp에서만 승격한다."""

    if suffix and (
        not suffix.startswith(".reanalysis_")
        or any(character not in ".abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in suffix)
    ):
        raise ValueError(f"유효하지 않은 analysis suffix: {suffix!r}")
    npz_path = session_dir / f"analysis_results{suffix}.npz"
    metadata_path = session_dir / f"analysis_metadata{suffix}.json"
    for path in (npz_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"기존 분석 산출물은 덮어쓰지 않습니다: {path}")
    token = uuid.uuid4().hex
    npz_temp = session_dir / f".{npz_path.name}.{token}.partial"
    metadata_temp = session_dir / f".{metadata_path.name}.{token}.partial"
    promoted: list[Path] = []
    try:
        with npz_temp.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        with metadata_temp.open("x", encoding="utf-8") as handle:
            json.dump(
                cw._json_safe(metadata),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        atomic_publish_noreplace(npz_temp, npz_path)
        promoted.append(npz_path)
        _fsync_directory(session_dir)
        atomic_publish_noreplace(metadata_temp, metadata_path)
        promoted.append(metadata_path)
        _fsync_directory(session_dir)
    except BaseException:
        # 분석 파일은 다시 만들 수 있지만 raw capture는 다시 만들 수 없다. 분석 쌍이
        # 반만 보이는 상태도 제거하며, raw_measurement.npz/metadata.json은 접근하지 않는다.
        for path in (npz_temp, metadata_temp, *promoted):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return {"results": npz_path, "metadata": metadata_path}


def write_official_pair_atomic(
    primary_path: Path,
    primary_arrays: dict[str, Any],
    secondary_path: Path,
    secondary_arrays: dict[str, Any],
) -> tuple[Path, Path]:
    """P/S temp를 모두 fsync한 뒤 pair 승격하고 실패 시 orphan을 제거한다."""

    targets = (Path(primary_path), Path(secondary_path))
    if targets[0] == targets[1]:
        raise ValueError("P/S official target은 달라야 합니다")
    for target in targets:
        if target.exists():
            raise FileExistsError(f"기존 정식 모델은 덮어쓰지 않습니다: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temps = tuple(
        target.parent / f".{target.name}.{token}.partial" for target in targets
    )
    promoted: list[Path] = []
    try:
        for temp, arrays in zip(temps, (primary_arrays, secondary_arrays)):
            with temp.open("xb") as handle:
                np.savez(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
        for temp, target in zip(temps, targets):
            atomic_publish_noreplace(temp, target)
            promoted.append(target)
    except BaseException:
        for path in (*temps, *promoted):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        for parent in {target.parent for target in targets}:
            try:
                _fsync_directory(parent)
            except OSError:
                pass
        raise
    return targets


def official_pair_success_message(
    primary_path: Path,
    secondary_path: Path,
    *,
    repository_root: Path,
) -> str:
    """수동 timing 숫자 없이 새 strict P/S 경로만 안내한다."""

    root = Path(repository_root).resolve()

    def display(path: Path) -> Path:
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(root)
        except ValueError:
            return resolved

    return (
        f"\n[성공] strict primary NPZ: {display(primary_path)}\n"
        f"       strict secondary NPZ: {display(secondary_path)}"
    )


def _adjacent_cycle_delay_time_domain(
    previous: np.ndarray, current: np.ndarray, *, span_samples: int = 16
) -> tuple[float, float, float]:
    """세 중앙 subwindow의 normalized correlation로 fractional cycle shift를 잰다."""

    left = np.asarray(previous, dtype=np.float64).reshape(-1)
    right = np.asarray(current, dtype=np.float64).reshape(-1)
    if left.size != right.size or left.size < 8 * (span_samples + 1):
        raise ValueError("adjacent cycle correlation 입력 길이가 유효하지 않습니다")
    length = int(left.size)
    windows = (
        (length // 8, 3 * length // 8),
        (3 * length // 8, 5 * length // 8),
        (5 * length // 8, 7 * length // 8),
    )
    lags = np.arange(-int(span_samples), int(span_samples) + 1, dtype=np.int64)
    delays: list[float] = []
    peaks: list[float] = []
    for lo, hi in windows:
        correlations = np.empty(lags.size, dtype=np.float64)
        for position, lag in enumerate(lags):
            start = lo + max(0, -int(lag))
            stop = hi - max(0, int(lag))
            a = left[start:stop]
            b = right[start + int(lag) : stop + int(lag)]
            a = a - float(np.mean(a))
            b = b - float(np.mean(b))
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            correlations[position] = (
                float(np.dot(a, b)) / denominator if denominator > 0.0 else 0.0
            )
        best = int(np.argmax(correlations))
        fraction = 0.0
        peak = float(correlations[best])
        if 0 < best < correlations.size - 1:
            y0, y1, y2 = correlations[best - 1 : best + 2]
            denominator = float(y0 - 2.0 * y1 + y2)
            if denominator != 0.0:
                fraction = float(0.5 * (y0 - y2) / denominator)
            peak = float(y1 - 0.25 * (y0 - y2) * fraction)
        delays.append(float(lags[best]) + fraction)
        peaks.append(peak)
    return (
        float(np.median(delays)),
        float(np.min(peaks)),
        float(np.ptp(delays)),
    )


def _fixed_point_clock_valid_mask(
    *,
    base_valid: np.ndarray,
    common_delay_samples: np.ndarray,
    adjacent_change_samples: np.ndarray,
    max_drift_deviation_samples: float,
    max_adjacent_change_samples: float,
    min_valid_periods: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return the monotone, final-median-stable clock observation mask.

    The first median envelope and the adjacent-change gate can shift the median.
    Reapplying the envelope to a fixed point is therefore part of the official
    separation contract: every returned observation is within the hard bound of
    the *returned* median, not merely the pre-filter median.
    """

    valid = np.asarray(base_valid, dtype=np.bool_).reshape(-1).copy()
    common = np.asarray(common_delay_samples, dtype=np.float64).reshape(-1)
    adjacent = np.asarray(adjacent_change_samples, dtype=np.float64).reshape(-1)
    if common.size != valid.size or adjacent.size != valid.size:
        raise ValueError("clock valid-mask witness 배열 길이가 다릅니다")
    minimum = int(min_valid_periods)
    if minimum <= 0 or int(valid.sum()) < minimum:
        raise ValueError(
            f"clock drift base-valid 주기가 {int(valid.sum())}개뿐입니다 "
            f"(최소 {minimum})"
        )

    initial_median = float(np.median(common[valid]))
    valid &= (
        np.abs(common - initial_median) <= float(max_drift_deviation_samples)
    )
    valid[1:] &= (
        ~np.isfinite(adjacent[1:])
        | (adjacent[1:] <= float(max_adjacent_change_samples))
    )

    # This loop is monotone (only True -> False), so it terminates in at most N
    # iterations and cannot oscillate when the median moves after exclusions.
    while True:
        count = int(valid.sum())
        if count < minimum:
            raise ValueError(
                f"clock drift final-median 고정점 주기가 {count}개뿐입니다 "
                f"(최소 {minimum})"
            )
        median = float(np.median(common[valid]))
        deviation = np.abs(common - median)
        updated = valid & (
            deviation <= float(max_drift_deviation_samples)
        )
        if np.array_equal(updated, valid):
            return valid, median, deviation
        valid = updated


def observe_period_clock_ratios(
    *,
    err: np.ndarray,
    ref: np.ndarray,
    probe,
    period_starts: list[int],
    max_drift_deviation_samples: float,
    min_valid_periods: int,
    clock_band_hz: tuple[float, float] = CLOCK_BAND_HZ,
) -> dict[str, Any]:
    """ERR/REF 인접 주기의 공통 observed-cycle 길이와 ``q=N/(N+d)``를 잰다."""

    starts = np.asarray(period_starts, dtype=np.int64).reshape(-1)
    if starts.size < 3:
        raise ValueError("clock ratio 관측에는 주기 3개 이상이 필요합니다")
    n_period = int(starts.size)
    n = int(probe.period_samples)
    clock_band = (float(clock_band_hz[0]), float(clock_band_hz[1]))
    nyquist = 0.5 * float(probe.sample_rate)
    if not (0.0 <= clock_band[0] < clock_band[1] <= nyquist):
        raise ValueError(
            f"clock 관측 대역이 잘못되었습니다: {clock_band}, Nyquist={nyquist}"
        )
    clock_segments: dict[str, np.ndarray] = {}
    clock_frequencies = np.fft.rfftfreq(n, d=1.0 / float(probe.sample_rate))
    clock_mask = (
        (clock_frequencies >= clock_band[0])
        & (clock_frequencies <= clock_band[1])
    )
    for name, values in (("err", err), ("ref", ref)):
        signal = np.asarray(values, dtype=np.float64).reshape(-1)
        if starts[-1] + n > signal.size:
            raise ValueError(f"{name}가 clock ratio 관측 주기보다 짧습니다")
        rows = []
        for start in starts:
            spectrum = np.fft.rfft(signal[start : start + n])
            spectrum[~clock_mask] = 0.0
            rows.append(np.fft.irfft(spectrum, n=n))
        clock_segments[name] = np.stack(rows)

    delays = {
        name: np.full(n_period, np.nan, dtype=np.float64)
        for name in ("err", "ref")
    }
    scores = {
        name: np.full(n_period, np.nan, dtype=np.float64)
        for name in ("err", "ref")
    }
    spreads = {
        name: np.full(n_period, np.nan, dtype=np.float64)
        for name in ("err", "ref")
    }
    for index in range(n_period - 1):
        for name in ("err", "ref"):
            delay, score, spread = _adjacent_cycle_delay_time_domain(
                clock_segments[name][index], clock_segments[name][index + 1]
            )
            delays[name][index] = delay
            scores[name][index] = score
            spreads[name][index] = spread

    score_sum = scores["err"] + scores["ref"]
    common_delay = (
        delays["err"] * scores["err"] + delays["ref"] * scores["ref"]
    ) / np.maximum(score_sum, np.finfo(np.float64).tiny)
    mic_delta = np.abs(delays["err"] - delays["ref"])
    valid = (
        np.isfinite(common_delay)
        & np.isfinite(mic_delta)
        & np.isfinite(spreads["err"])
        & np.isfinite(spreads["ref"])
        & (scores["err"] >= CLOCK_MIN_ADJACENT_SCORE)
        & (scores["ref"] >= CLOCK_MIN_ADJACENT_SCORE)
        & (scores["err"] <= 1.000001)
        & (scores["ref"] <= 1.000001)
        & (spreads["err"] >= 0.0)
        & (spreads["ref"] >= 0.0)
        & (spreads["err"] <= CLOCK_MAX_SUBWINDOW_SPREAD_SAMPLES)
        & (spreads["ref"] <= CLOCK_MAX_SUBWINDOW_SPREAD_SAMPLES)
        & (mic_delta <= CLOCK_MAX_ERR_REF_DELTA_SAMPLES)
        & (np.abs(common_delay) <= CLOCK_MAX_ABS_PERIOD_DELTA_SAMPLES)
        & ((n + common_delay) > 0.0)
    )
    if int(valid.sum()) < int(min_valid_periods):
        raise ValueError(
            f"ERR/REF 공통 clock ratio가 유효한 주기가 {int(valid.sum())}개뿐입니다 "
            f"(최소 {int(min_valid_periods)}, score>={CLOCK_MIN_ADJACENT_SCORE}, "
            f"|dERR-dREF|<={CLOCK_MAX_ERR_REF_DELTA_SAMPLES})"
        )
    adjacent_change = np.full(n_period, np.nan, dtype=np.float64)
    adjacent_change[1:-1] = np.abs(np.diff(common_delay[:-1]))
    try:
        valid, drift_median, drift_deviation = _fixed_point_clock_valid_mask(
            base_valid=valid,
            common_delay_samples=common_delay,
            adjacent_change_samples=adjacent_change,
            max_drift_deviation_samples=max_drift_deviation_samples,
            max_adjacent_change_samples=CLOCK_MAX_ADJACENT_CHANGE_SAMPLES,
            min_valid_periods=min_valid_periods,
        )
    except ValueError as exc:
        raise ValueError(
            "clock drift 중앙값/인접 변화 hard gate 실패: "
            f"{exc}; 허용 ±{float(max_drift_deviation_samples):.3f} samples"
        ) from exc
    q = np.full(n_period, np.nan, dtype=np.float64)
    q[valid] = n / (n + common_delay[valid])
    return {
        "valid": valid,
        "q": q,
        "common_delay_samples": common_delay,
        "err_delay_samples": delays["err"],
        "ref_delay_samples": delays["ref"],
        "err_score": scores["err"],
        "ref_score": scores["ref"],
        "err_subwindow_spread_samples": spreads["err"],
        "ref_subwindow_spread_samples": spreads["ref"],
        "err_ref_delta_samples": mic_delta,
        "adjacent_change_samples": adjacent_change,
        "drift_deviation_samples": drift_deviation,
        "drift_samples_per_period": drift_median,
        "drift_ppm": 1e6 * drift_median / float(n),
        "clock_band_hz": np.asarray(clock_band, dtype=np.float64),
    }


def _joint_real_basis(
    probe, q: float
) -> tuple[np.ndarray, tuple[np.ndarray, bool], int, float, np.ndarray]:
    """fractional tone real basis와 normal-equation Cholesky/condition을 만든다."""

    union_bins = np.sort(
        np.concatenate((probe.noise_bins, probe.cancel_bins)).astype(np.int64)
    )
    n = np.arange(probe.period_samples, dtype=np.float64)[:, None]
    phase = 2.0 * np.pi * n * (float(q) * union_bins[None, :]) / float(
        probe.period_samples
    )
    basis = np.concatenate((np.cos(phase), np.sin(phase)), axis=1)
    gram = basis.T @ basis
    eigenvalues = np.linalg.eigvalsh(gram)
    largest = float(eigenvalues[-1])
    threshold = largest * max(gram.shape) * np.finfo(np.float64).eps
    rank = int(np.sum(eigenvalues > threshold))
    condition = (
        float(np.sqrt(largest / float(eigenvalues[0])))
        if eigenvalues[0] > 0.0
        else float("inf")
    )
    if rank != gram.shape[0] or not np.isfinite(condition):
        raise ValueError(
            f"fractional joint LS rank 결함: rank={rank}/{gram.shape[0]}, "
            f"condition={condition}"
        )
    factor = cho_factor(gram, lower=True, check_finite=False)
    return basis, factor, rank, condition, union_bins


def fractional_joint_channel_stacks(
    *,
    err: np.ndarray,
    ref: np.ndarray,
    output_pcm_int16: np.ndarray,
    probe,
    period_starts: list[int],
    fit_band_hz: tuple[float, float],
    max_drift_deviation_samples: float,
    min_valid_periods: int,
    clock_band_hz: tuple[float, float] = CLOCK_BAND_HZ,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """clock-corrected real joint LS로 P/S를 동시에 풀고 독립 cubic FFT를 만든다."""

    clock = observe_period_clock_ratios(
        err=err,
        ref=ref,
        probe=probe,
        period_starts=period_starts,
        max_drift_deviation_samples=max_drift_deviation_samples,
        min_valid_periods=min_valid_periods,
        clock_band_hz=clock_band_hz,
    )
    starts = np.asarray(period_starts, dtype=np.int64)
    n_period = int(starts.size)
    n = int(probe.period_samples)
    valid = np.asarray(clock["valid"], dtype=bool)
    median_q = n / (n + float(clock["drift_samples_per_period"]))
    q_for_solve = np.where(valid, clock["q"], median_q)
    signal = np.asarray(err, dtype=np.float64).reshape(-1)
    submitted = np.asarray(output_pcm_int16)
    if submitted.dtype != np.int16 or submitted.ndim != 2 or submitted.shape[1] != 2:
        raise ValueError("output_pcm_int16은 실제 제출된 [frames,2] int16이어야 합니다")
    reference_start = int(starts[0])
    if reference_start + n > submitted.shape[0]:
        raise ValueError("output_pcm_int16이 분석 주기보다 짧습니다")
    submitted_period = submitted[reference_start : reference_start + n].astype(
        np.float64
    ) / float(np.iinfo(np.int16).max)
    input_spectra = {
        drive: np.fft.rfft(submitted_period[:, channel])
        for drive, channel in (("noise", 0), ("cancel", 1))
    }
    frequencies = {
        drive: (
            probe.bins_for(drive)
            * float(probe.sample_rate)
            / float(probe.period_samples)
        ).astype(np.float64)
        for drive in ("noise", "cancel")
    }
    stacks = {
        drive: np.zeros(
            (n_period, probe.bins_for(drive).size), dtype=np.complex128
        )
        for drive in ("noise", "cancel")
    }
    crosscheck = {
        drive: np.full(stacks[drive].shape, np.nan + 1j * np.nan, np.complex128)
        for drive in stacks
    }
    ranks = np.zeros(n_period, dtype=np.int64)
    conditions = np.full(n_period, np.nan, dtype=np.float64)
    residuals = np.full(n_period, np.nan, dtype=np.float64)
    cached_key: bytes | None = None
    cached_prepared: (
        tuple[np.ndarray, tuple[np.ndarray, bool], int, float, np.ndarray] | None
    ) = None
    sample_index = np.arange(n, dtype=np.float64)

    for index, (start, q_value) in enumerate(zip(starts, q_for_solve)):
        if not valid[index]:
            continue
        key = np.float64(q_value).tobytes()
        if cached_key == key and cached_prepared is not None:
            prepared = cached_prepared
        else:
            prepared = _joint_real_basis(probe, float(q_value))
            cached_key, cached_prepared = key, prepared
        basis, factor, rank, condition, union_bins = prepared
        segment = signal[int(start) : int(start) + n]
        coefficients = cho_solve(
            factor, basis.T @ segment, check_finite=False
        )
        count = int(union_bins.size)
        complex_output = (n / 2.0) * (
            coefficients[:count] - 1j * coefficients[count:]
        )
        reconstructed = basis @ coefficients
        denominator = float(np.linalg.norm(segment))
        residuals[index] = (
            float(np.linalg.norm(segment - reconstructed)) / denominator
            if denominator > 0.0
            else float("inf")
        )
        ranks[index] = rank
        conditions[index] = condition
        for drive in ("noise", "cancel"):
            selected = probe.bins_for(drive)
            positions = np.searchsorted(union_bins, selected)
            stacks[drive][index] = complex_output[positions] / input_spectra[drive][
                selected
            ]

        # joint LS와 독립인 playback-grid cubic 재격자화. q<1이면 다음 주기에서 몇
        # 샘플을 더 읽으므로 마지막(애초 q가 없는) 주기는 valid가 아니다.
        coordinates = float(start) + sample_index / float(q_value)
        if coordinates[-1] >= signal.size - 1:
            valid[index] = False
            for drive in crosscheck:
                crosscheck[drive][index].fill(np.nan)
        else:
            resampled = map_coordinates(
                signal,
                [coordinates],
                order=3,
                mode="nearest",
                prefilter=True,
            )
            for drive in crosscheck:
                selected = probe.bins_for(drive)
                crosscheck[drive][index] = (
                    np.fft.rfft(resampled)[selected]
                    / input_spectra[drive][selected]
                )

    if int(valid.sum()) < int(min_valid_periods):
        raise ValueError(
            f"재격자화 가능한 q-valid 주기가 {int(valid.sum())}개뿐입니다 "
            f"(최소 {int(min_valid_periods)})"
        )
    clock["valid"] = valid
    if np.any(ranks[valid] != 2 * np.sort(np.r_[probe.noise_bins, probe.cancel_bins]).size):
        raise ValueError("fractional joint LS가 full rank가 아닙니다")
    if np.any(conditions[valid] > JOINT_LS_MAX_CONDITION):
        raise ValueError(
            f"fractional joint LS condition {float(np.max(conditions[valid])):.6f} > "
            f"{JOINT_LS_MAX_CONDITION:.6f}"
        )
    residual_p95 = float(np.percentile(residuals[valid], 95.0))
    if not np.isfinite(residual_p95) or residual_p95 > JOINT_LS_MAX_RESIDUAL_P95:
        raise ValueError(
            f"fractional joint LS reconstruction residual p95 "
            f"{residual_p95:.6f} > {JOINT_LS_MAX_RESIDUAL_P95:.6f}"
        )
    clock.update(
        {
            "joint_ls_rank": ranks,
            "joint_ls_condition": conditions,
            "joint_ls_reconstruction_relative_error": residuals,
            "joint_ls_reconstruction_relative_error_p95": residual_p95,
            "crosscheck_transfers": crosscheck,
            "submitted_input_spectra": input_spectra,
            "separation_algorithm": SEPARATION_ALGORITHM,
            "separation_algorithm_version": SEPARATION_ALGORITHM_VERSION,
        }
    )
    return frequencies, stacks, clock


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
    initial_keep: np.ndarray | None = None,
    observed_drift_samples: np.ndarray | None = None,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """두 채널을 **함께** 보고 쓸 반복과 앵커를 정한다.

    반환 ``(keep, anchor, report)``. 판정 근거는 전부 타임베이스 관측이다 —
    "결과가 좋아지는가"로 고르지 않는다. 어떤 반복을 왜 버렸는지 report 에 남는다.

    채널별 독립 판정이면 P 에서만 버린 반복이 S 에 남아 두 경로가 서로 다른 반복
    집합에서 만들어질 수 있다. 그러면 lead 가 물리량이 아니게 된다.
    """

    taus: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    if initial_keep is None:
        initial = np.ones(stacks["noise"].shape[0], dtype=bool)
    else:
        initial = np.asarray(initial_keep, dtype=bool).reshape(-1)
        if initial.size != stacks["noise"].shape[0]:
            raise ValueError("initial_keep 길이가 반복 수와 다릅니다")
    candidates = np.flatnonzero(initial)
    if candidates.size < int(min_kept_repeats):
        raise ValueError(
            f"초기 유효 반복이 {candidates.size}개뿐입니다 (최소 {min_kept_repeats})"
        )
    pass1_anchor = int(candidates[0])
    for drive in ("noise", "cancel"):
        _, tau, score = align_repeats(
            frequencies[drive], stacks[drive], sample_rate=sample_rate,
            fit_band_hz=fit_band_hz, anchor=pass1_anchor,
        )
        taus[drive], scores[drive] = tau, score

    # (a) 타임베이스가 정상상태였는가. 워밍업 4주기로는 부족하다는 것이 실측이다.
    common = 0.5 * (taus["noise"] + taus["cancel"])
    if observed_drift_samples is None:
        drift, drift_median = timebase_drift(common)
    else:
        drift = np.asarray(observed_drift_samples, dtype=np.float64).reshape(-1)
        if drift.size != common.size:
            raise ValueError("observed_drift_samples 길이가 반복 수와 다릅니다")
        if not np.all(np.isfinite(drift[initial])):
            raise ValueError("초기 유효 반복의 observed drift에 NaN/Inf가 있습니다")
        drift_median = float(np.median(drift[initial]))
    drift_dev = np.abs(drift - drift_median)
    keep = initial & (drift_dev <= float(max_drift_deviation_samples))

    # (b) P−S 상대 τ 연속성. 이 측정 방식의 유일한 물리 불변량이다.
    relative = taus["noise"] - taus["cancel"]
    centre_candidates = initial.copy()
    centre_candidates[pass1_anchor] = False
    if not np.any(centre_candidates):
        centre_candidates = initial
    rel_centre = float(np.median(relative[centre_candidates]))
    rel_dev = np.abs(relative - rel_centre)
    bad_rel = rel_dev > float(max_relative_tau_samples)
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
        "pass1_anchor": pass1_anchor,
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
    common_alignment_taus: np.ndarray | None = None,
    provisional_taus: np.ndarray | None = None,
    alignment_scores: np.ndarray | None = None,
    consistency_subbands_hz: tuple[
        tuple[float, float], ...
    ] = CONSISTENCY_SUB_BANDS_HZ,
    compact_transfer_subbands_hz: tuple[
        tuple[float, float], ...
    ] = COMPACT_TRANSFER_SUB_BANDS_HZ,
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
    _, measured_taus, measured_scores = align_repeats(
        frequencies, stack, sample_rate=probe.sample_rate,
        fit_band_hz=fit_band_hz, anchor=int(anchor),
    )
    if provisional_taus is not None:
        supplied = np.asarray(provisional_taus, dtype=np.float64).reshape(-1)
        if supplied.size != stack.shape[0]:
            raise ValueError("provisional_taus 길이가 반복 수와 다릅니다")
        measured_taus = supplied
    scores = (
        measured_scores
        if alignment_scores is None
        else np.asarray(alignment_scores, dtype=np.float64).reshape(-1)
    )
    if scores.size != stack.shape[0] or not np.all(np.isfinite(scores)):
        raise ValueError("alignment_scores 길이/finite 계약 위반")
    if common_alignment_taus is None:
        taus = measured_taus
    else:
        taus = np.asarray(common_alignment_taus, dtype=np.float64).reshape(-1)
        if taus.size != stack.shape[0] or not np.all(np.isfinite(taus)):
            raise ValueError("common_alignment_taus 길이/finite 계약 위반")
    aligned = stack * np.exp(
        2j * np.pi * frequencies[None, :] * taus[:, None] / probe.sample_rate
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
    bulk_integer_delay = int(round(delay))
    effective_delay = bulk_integer_delay - int(pre_roll)
    if effective_delay < 0:
        raise ValueError(
            f"{drive} bulk delay {bulk_integer_delay} < pre-roll {pre_roll} — "
            "compact FIR 앞의 effective delay를 만들 수 없습니다"
        )
    # 벌크 지연을 빼면 남는 IR 이 짧아져 복원 주기 안에 안전하게 들어간다.
    residual = mean_transfer * np.exp(
        2j * np.pi * frequencies * bulk_integer_delay / probe.sample_rate
    )
    ir = channel_impulse_response(probe, residual, drive=drive, pre_roll=pre_roll)
    if ir.size < fir_length:
        raise ValueError(
            f"복원 IR 길이 {ir.size} < FIR {fir_length} — 분석 주기를 늘리세요"
        )
    fir = ir[:fir_length].astype(np.float32)
    round_trip = compact_transfer_round_trip(
        frequencies,
        mean_transfer,
        fir,
        effective_delay_samples=effective_delay,
        sample_rate=probe.sample_rate,
        band_hz=consistency_band_hz,
        required_subbands_hz=compact_transfer_subbands_hz,
    )
    if not round_trip["passed"]:
        raise ValueError(
            f"{drive} compact FIR 복소 전달 round-trip 실패: "
            f"agreement={round_trip['complex_agreement']:.6f} "
            f"(최소 {MIN_COMPACT_TRANSFER_AGREEMENT:.6f}), "
            f"relative_error={round_trip['relative_error']:.6f} "
            f"(최대 {MAX_COMPACT_TRANSFER_RELATIVE_ERROR:.6f})"
        )

    consistency_subbands = tuple(
        (float(bounds[0]), float(bounds[1]))
        for bounds in consistency_subbands_hz
    )
    if not consistency_subbands or any(
        lo >= hi for lo, hi in consistency_subbands
    ):
        raise ValueError("일관성 부대역이 비었거나 잘못되었습니다")
    band_consistency = np.asarray(
        [
            complex_consistency(aligned[:, (frequencies >= lo) & (frequencies <= hi)])
            if int(((frequencies >= lo) & (frequencies <= hi)).sum()) >= 4
            else np.nan
            for lo, hi in consistency_subbands
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
        "provisional_taus": measured_taus,
        "provisional_taus_kept": measured_taus[keep],
        "common_alignment_taus": taus,
        "common_alignment_taus_kept": taus_kept,
        "alignment_scores": scores,
        "kept_mask": keep,
        "anchor_repeat": int(anchor),
        "kept_repeat_indices": np.flatnonzero(keep).astype(np.int64),
        "band_consistency": band_consistency,
        "band_consistency_hz": np.asarray(consistency_subbands, dtype=np.float64),
        "rejected_repeats": int(taus.size - taus_kept.size),
        "consistency": consistency,
        "fullband_consistency": fullband_consistency,
        "consistency_band_hz": (
            float(consistency_band_hz[0]), float(consistency_band_hz[1])
        ),
        "raw_consistency": complex_consistency(stack),
        "absolute_tau_spread": float(np.max(taus_kept) - np.min(taus_kept)),
        "delay_samples": effective_delay,
        "delay_fractional": delay - float(pre_roll),
        "bulk_delay_samples": bulk_integer_delay,
        "bulk_delay_fractional_samples": delay,
        "pre_roll_samples": int(pre_roll),
        "delay_semantics": DELAY_SEMANTICS,
        "compact_transfer_round_trip": round_trip,
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
        median = float(np.median(finite))
        if median < MIN_TONE_SNR_MEDIAN_DB:
            reasons.append(f"tone_snr_median_{median:.2f}")
    return reasons


def capture_telemetry_invalid_reasons(telemetry: dict[str, Any]) -> list[str]:
    """분석 호출 없이 callback telemetry만으로 확정되는 raw 결함."""
    reasons: list[str] = []
    if int(telemetry.get("xrun_count", 0)) != 0:
        reasons.append(f"xrun_{telemetry['xrun_count']}")
    if int(telemetry.get("unexpected_status_count", 0)) != 0:
        reasons.append(
            f"unexpected_callback_status_{telemetry['unexpected_status_count']}"
        )
    if telemetry.get("callback_error"):
        reasons.append(f"callback_error_{telemetry['callback_error']}")
    if telemetry.get("stream_abort_error"):
        reasons.append(f"stream_abort_error_{telemetry['stream_abort_error']}")
    if telemetry.get("stream_close_error"):
        reasons.append(f"stream_close_error_{telemetry['stream_close_error']}")
    if telemetry.get("output_stop_confirmed") is False:
        reasons.append("output_stop_unconfirmed")
    if not telemetry.get("completed"):
        reasons.append("capture_incomplete")
    return reasons


def capture_invalid_reasons(
    telemetry: dict[str, Any], measurement_report: dict[str, Any]
) -> list[str]:
    """raw 캡처 자체의 결함을 분석 결과와 무관하게 fail-closed 판정한다."""

    reasons = capture_telemetry_invalid_reasons(telemetry)
    channels = measurement_report.get("channels", [])
    if len(channels) < 2:
        reasons.append("measurement_missing_err_ref_channels")
    for index, item in enumerate(channels[:2]):
        if not bool(item.get("valid")):
            reasons.append(f"measurement_invalid_ch{index}")
        if float(item.get("clip_ratio", 1.0)) > cw.MAX_INPUT_CLIP_RATIO:
            reasons.append(f"input_clip_ch{index}_{item['clip_ratio']:.4f}")
    return reasons


def separation_crosscheck_metrics(
    *,
    frequencies: dict[str, np.ndarray],
    joint_stacks: dict[str, np.ndarray],
    resampled_stacks: dict[str, np.ndarray],
    keep: np.ndarray,
    subbands_hz: tuple[
        tuple[float, float], ...
    ] = COMPACT_TRANSFER_SUB_BANDS_HZ,
    overall_band_hz: tuple[
        float, float
    ] = SEPARATION_CROSSCHECK_OVERALL_BAND_HZ,
) -> dict[str, Any]:
    """joint LS와 독립 cubic playback-grid FFT의 복소 일치를 hard gate한다."""

    selected_rows = np.asarray(keep, dtype=bool).reshape(-1)
    required_subbands = tuple(
        (float(bounds[0]), float(bounds[1])) for bounds in subbands_hz
    )
    overall_band = (float(overall_band_hz[0]), float(overall_band_hz[1]))
    if not required_subbands or any(lo >= hi for lo, hi in required_subbands):
        raise ValueError("separation crosscheck 부대역이 비었거나 잘못되었습니다")
    if overall_band[0] >= overall_band[1]:
        raise ValueError("separation crosscheck 전대역이 잘못되었습니다")
    report: dict[str, Any] = {}
    for drive in ("noise", "cancel"):
        freq = np.asarray(frequencies[drive], dtype=np.float64).reshape(-1)
        joint = np.asarray(joint_stacks[drive], dtype=np.complex128)[selected_rows]
        check = np.asarray(resampled_stacks[drive], dtype=np.complex128)[selected_rows]
        rows = []
        for bounds in (*required_subbands, overall_band):
            mask = (freq >= bounds[0]) & (freq <= bounds[1])
            if int(mask.sum()) < 4:
                raise ValueError(
                    f"{drive} separation crosscheck {bounds} 톤이 부족합니다"
                )
            left = joint[:, mask].reshape(-1)
            right = check[:, mask].reshape(-1)
            if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
                raise ValueError(f"{drive} separation crosscheck에 NaN/Inf가 있습니다")
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            agreement = (
                abs(complex(np.vdot(left, right))) / denominator
                if denominator > 0.0
                else 0.0
            )
            energy = float(np.linalg.norm(left))
            relative_error = (
                float(np.linalg.norm(right - left)) / energy
                if energy > 0.0
                else float("inf")
            )
            passed = (
                agreement >= SEPARATION_CROSSCHECK_MIN_AGREEMENT
                and relative_error <= SEPARATION_CROSSCHECK_MAX_RELATIVE_ERROR
            )
            rows.append(
                {
                    "band_hz": (float(bounds[0]), float(bounds[1])),
                    "tone_count": int(mask.sum()),
                    "complex_agreement": float(agreement),
                    "relative_error": float(relative_error),
                    "passed": bool(passed),
                }
            )
        subbands, overall = rows[:-1], rows[-1]
        if not overall["passed"] or not all(row["passed"] for row in subbands):
            worst_agreement = min(row["complex_agreement"] for row in rows)
            worst_error = max(row["relative_error"] for row in rows)
            raise ValueError(
                f"{drive} fractional separation 독립 crosscheck 실패: "
                f"worst agreement={worst_agreement:.6f} "
                f"(<{SEPARATION_CROSSCHECK_MIN_AGREEMENT}), "
                f"worst relative_error={worst_error:.6f} "
                f"(>{SEPARATION_CROSSCHECK_MAX_RELATIVE_ERROR})"
            )
        report[drive] = {"overall": overall, "subbands": subbands}
    return report


def analyse_capture(
    *,
    err: np.ndarray,
    ref: np.ndarray,
    output_pcm_int16: np.ndarray,
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
    clock_band_hz: tuple[float, float] = CLOCK_BAND_HZ,
    consistency_subbands_hz: tuple[
        tuple[float, float], ...
    ] = CONSISTENCY_SUB_BANDS_HZ,
    compact_transfer_subbands_hz: tuple[
        tuple[float, float], ...
    ] = COMPACT_TRANSFER_SUB_BANDS_HZ,
    crosscheck_subbands_hz: tuple[
        tuple[float, float], ...
    ] = COMPACT_TRANSFER_SUB_BANDS_HZ,
    crosscheck_overall_band_hz: tuple[
        float, float
    ] = SEPARATION_CROSSCHECK_OVERALL_BAND_HZ,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """캡처 한 건을 2-pass 공동 분석한다 — **온라인·오프라인이 공유하는 유일한 경로**.

    pass1 은 두 채널의 τ 궤적만 얻어 (a) 타임베이스 드리프트 이상치와 (b) P−S 상대 τ
    연속성 위반을 **함께** 판정한다. pass2 는 살아남은 반복의 중앙을 앵커로 재정렬한다.
    측정 스크립트와 재분석 스크립트가 서로 다른 코드로 갈라지면 재현성이 깨지므로
    두 경로 모두 이 함수만 호출한다.
    """

    if float(min_alignment_score) < DEFAULT_MIN_ALIGNMENT_SCORE:
        raise ValueError("min_alignment_score는 official hard lower bound보다 낮출 수 없습니다")
    if float(max_relative_tau_samples) > DEFAULT_MAX_RELATIVE_TAU_SAMPLES:
        raise ValueError("max_relative_tau_samples는 official hard bound보다 키울 수 없습니다")
    if float(max_drift_deviation_samples) > DEFAULT_MAX_DRIFT_DEVIATION_SAMPLES:
        raise ValueError("max_drift_deviation_samples는 official hard bound보다 키울 수 없습니다")
    if int(min_kept_repeats) < 8:
        raise ValueError("min_kept_repeats는 official hard lower bound 8보다 낮출 수 없습니다")
    if int(max_delay_jitter_samples) > 3:
        raise ValueError("max_delay_jitter_samples는 official hard bound 3보다 키울 수 없습니다")

    _legacy_signal_spectrum, noise_spectrum = snr_spectra
    need_lo, need_hi = float(required_band_hz[0]), float(required_band_hz[1])

    frequencies, stacks, separation = fractional_joint_channel_stacks(
        err=err,
        ref=ref,
        output_pcm_int16=output_pcm_int16,
        probe=probe,
        period_starts=period_starts,
        fit_band_hz=clock_band_hz,
        max_drift_deviation_samples=max_drift_deviation_samples,
        min_valid_periods=min_kept_repeats,
        clock_band_hz=clock_band_hz,
    )

    keep, anchor, report = select_repeats(
        frequencies=frequencies, stacks=stacks, sample_rate=probe.sample_rate,
        fit_band_hz=fit_band_hz,
        max_relative_tau_samples=max_relative_tau_samples,
        max_drift_deviation_samples=max_drift_deviation_samples,
        min_kept_repeats=min_kept_repeats,
        initial_keep=separation["valid"],
        observed_drift_samples=separation["common_delay_samples"],
    )
    report["drift_ppm"] = float(separation["drift_ppm"])

    provisional_taus: dict[str, np.ndarray] = {}
    alignment_scores: dict[str, np.ndarray] = {}
    for drive in ("noise", "cancel"):
        _, tau, score = align_repeats(
            frequencies[drive],
            stacks[drive],
            sample_rate=probe.sample_rate,
            fit_band_hz=fit_band_hz,
            anchor=anchor,
        )
        provisional_taus[drive] = tau
        alignment_scores[drive] = score

    final_keep = (
        keep
        & (alignment_scores["noise"] >= float(min_alignment_score))
        & (alignment_scores["cancel"] >= float(min_alignment_score))
    )
    final_keep[anchor] = True
    if int(final_keep.sum()) < int(min_kept_repeats):
        raise ValueError(
            f"두 채널 모두 통과한 반복이 {int(final_keep.sum())}개뿐입니다 "
            f"(최소 {int(min_kept_repeats)})"
        )

    score_sum = alignment_scores["noise"] + alignment_scores["cancel"]
    common_taus = (
        provisional_taus["noise"] * alignment_scores["noise"]
        + provisional_taus["cancel"] * alignment_scores["cancel"]
    ) / np.maximum(score_sum, np.finfo(np.float64).tiny)

    crosscheck = separation_crosscheck_metrics(
        frequencies=frequencies,
        joint_stacks=stacks,
        resampled_stacks=separation["crosscheck_transfers"],
        keep=final_keep,
        subbands_hz=crosscheck_subbands_hz,
        overall_band_hz=crosscheck_overall_band_hz,
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
            keep=final_keep, anchor=anchor, stack=stacks[drive],
            common_alignment_taus=common_taus,
            provisional_taus=provisional_taus[drive],
            alignment_scores=alignment_scores[drive],
            consistency_subbands_hz=consistency_subbands_hz,
            compact_transfer_subbands_hz=compact_transfer_subbands_hz,
        )
        if not np.array_equal(model["kept_mask"], final_keep):
            raise ValueError(f"{drive} 공통 반복 집합 재계산이 일치하지 않습니다")
        selected_bins = probe.bins_for(drive)
        captured_tone_magnitude = np.median(
            np.abs(
                stacks[drive][final_keep]
                * separation["submitted_input_spectra"][drive][selected_bins][None, :]
            ),
            axis=0,
        )
        preflight_noise_magnitude = np.abs(
            np.asarray(noise_spectrum).reshape(-1)[selected_bins]
        )
        snr = 20.0 * np.log10(
            (captured_tone_magnitude + 1e-30)
            / (preflight_noise_magnitude + 1e-30)
        )
        results[drive] = {
            "model": model,
            "snr_db": snr,
            "output_channel": drive,
            "reasons": channel_quality(
            consistency=model["consistency"], snr_db=snr,
            min_consistency=MIN_INTERLEAVED_CONSISTENCY,
            ),
        }

    # lead 가 의존하는 유일한 양 — 두 경로의 **상대** 시간이동. 절대 warp 는 두 채널에
    # 공통이므로 여기서 상쇄된다. 이것이 커지면 lead 를 믿을 수 없다.
    relative_tau = (
        provisional_taus["noise"][final_keep]
        - provisional_taus["cancel"][final_keep]
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
            "common_alignment_taus": common_taus,
            "separation": separation,
            "separation_crosscheck": crosscheck,
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
    channel_map: dict[str, int],
    operator_confirmations: dict[str, bool],
    output_channel: str,
    repeats: int,
    xrun_count: int,
    capture_id: str,
    probe,
    drive: str,
    snr_db: np.ndarray,
    period_seconds: float,
    drift_samples_per_period: float,
    max_drift_deviation_samples: float,
    relative_tau_max_abs: float,
    source_raw_npz_path: str,
    source_raw_npz_sha256: str,
    source_analysis_npz_path: str,
    source_analysis_npz_sha256: str,
    output_pcm_provenance: str,
    separation: dict[str, Any],
    separation_crosscheck: dict[str, Any],
) -> dict[str, Any]:
    if int(fs) != OFFICIAL_SAMPLE_RATE:
        raise ValueError(
            f"official sample_rate는 {OFFICIAL_SAMPLE_RATE}여야 합니다: {fs}"
        )
    if int(block_size) != OFFICIAL_BLOCK_SIZE or str(latency) != OFFICIAL_LATENCY:
        raise ValueError(
            "official transport는 block_size=256, latency='low'여야 합니다: "
            f"{block_size}, {latency!r}"
        )
    normalized_channel_map = {
        name: int(channel_map.get(name, -1)) for name in OFFICIAL_CHANNEL_MAP
    }
    if normalized_channel_map != OFFICIAL_CHANNEL_MAP or set(channel_map) != set(
        OFFICIAL_CHANNEL_MAP
    ):
        raise ValueError(f"official channel map 계약 위반: {channel_map!r}")
    required_confirmations = {
        "user_present": True,
        "volume_minimum": True,
        "routing_and_geometry": True,
    }
    if operator_confirmations != required_confirmations:
        raise ValueError(
            f"official operator confirmation 계약 위반: {operator_confirmations!r}"
        )
    if model.get("delay_semantics") != DELAY_SEMANTICS:
        raise ValueError(f"알 수 없는 delay semantics: {model.get('delay_semantics')!r}")
    effective_delay = int(model["delay_samples"])
    bulk_delay = int(model["bulk_delay_samples"])
    pre_roll = int(model["pre_roll_samples"])
    if effective_delay != bulk_delay - pre_roll:
        raise ValueError(
            f"delay 계약 위반: effective={effective_delay}, "
            f"bulk={bulk_delay}, pre_roll={pre_roll}"
        )
    frequencies = np.asarray(model["frequencies_hz"], dtype=np.float64).reshape(-1)
    aligned_mean = np.asarray(model["mean_transfer"], dtype=np.complex128).reshape(-1)
    # analyse_channel가 남긴 score를 신뢰하지 않고, 저장할 source 배열과 FIR에서
    # 출하 직전에 다시 계산한다.
    round_trip = compact_transfer_round_trip(
        frequencies,
        aligned_mean,
        np.asarray(model["fir"], dtype=np.float64),
        effective_delay_samples=effective_delay,
        sample_rate=fs,
        band_hz=tuple(float(v) for v in model["consistency_band_hz"]),
    )
    if not round_trip["passed"]:
        raise ValueError("저장 직전 compact FIR 복소 전달 round-trip gate 실패")
    subbands = round_trip["subbands"]
    source_digest = aligned_transfer_sha256(frequencies, aligned_mean)
    if output_pcm_provenance != OUTPUT_PCM_PROVENANCE_OBSERVED:
        raise ValueError("official에는 observed submitted PCM provenance가 필요합니다")
    if separation.get("separation_algorithm") != SEPARATION_ALGORITHM:
        raise ValueError("official separation algorithm 계약 위반")
    if int(separation.get("separation_algorithm_version", -1)) != (
        SEPARATION_ALGORITHM_VERSION
    ):
        raise ValueError("official separation algorithm version 계약 위반")
    clock_valid = np.asarray(separation["valid"], dtype=bool).reshape(-1)
    clock_indices = np.flatnonzero(clock_valid).astype(np.int64)
    crosscheck = separation_crosscheck[drive]
    crosscheck_subbands = crosscheck["subbands"]
    return {
        "fir": np.asarray(model["fir"], dtype=np.float32),
        "delay_samples": np.int64(effective_delay),
        "bulk_delay_samples": np.int64(bulk_delay),
        "pre_roll_samples": np.int64(pre_roll),
        "delay_semantics": np.str_(DELAY_SEMANTICS),
        "sample_rate": np.int64(fs),
        "coherence_median": np.float64(consistency),
        "consistency_band_hz": np.asarray(
            model["consistency_band_hz"], dtype=np.float64
        ),
        "fullband_consistency": np.float64(model["fullband_consistency"]),
        "excitation_band_hz": np.asarray(band_hz, dtype=np.float64),
        "calibration_block_size": np.int64(block_size),
        "calibration_latency": np.str_(latency),
        "error_mic_channel": np.int64(normalized_channel_map["error_mic"]),
        "reference_mic_channel": np.int64(
            normalized_channel_map["reference_mic"]
        ),
        "noise_output_channel": np.int64(normalized_channel_map["noise_out"]),
        "cancel_output_channel": np.int64(
            normalized_channel_map["cancel_out"]
        ),
        "operator_confirmed_volume_minimum": np.bool_(True),
        "operator_confirmed_user_present": np.bool_(True),
        "operator_confirmed_routing_and_geometry": np.bool_(True),
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
        # channel-specific provisional τ가 실제 P−S 상대지연 witness다. FIR 평균에는
        # 아래 shared common τ를 썼지만 이 배열을 덮으면 readiness 재감사가 구조적으로
        # 0만 보게 된다.
        "repeat_tau_samples": np.asarray(
            model["provisional_taus_kept"], dtype=np.float64
        ),
        "provisional_repeat_tau_samples": np.asarray(
            model["provisional_taus_kept"], dtype=np.float64
        ),
        "common_alignment_tau_samples": np.asarray(
            model["common_alignment_taus_kept"], dtype=np.float64
        ),
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
        "tone_frequencies_hz": frequencies,
        "aligned_mean_transfer_real": aligned_mean.real.astype(np.float64),
        "aligned_mean_transfer_imag": aligned_mean.imag.astype(np.float64),
        "aligned_mean_transfer_sha256": np.str_(source_digest),
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
        "clock_sample_rate": np.int64(fs),
        "clock_max_drift_deviation_samples": np.float64(
            max_drift_deviation_samples
        ),
        "relative_tau_max_abs_samples": np.float64(relative_tau_max_abs),
        "output_pcm_provenance": np.str_(output_pcm_provenance),
        "source_raw_npz_path": np.str_(source_raw_npz_path),
        "source_raw_npz_sha256": np.str_(source_raw_npz_sha256),
        "source_analysis_npz_path": np.str_(source_analysis_npz_path),
        "source_analysis_npz_sha256": np.str_(source_analysis_npz_sha256),
        "separation_algorithm": np.str_(SEPARATION_ALGORITHM),
        "separation_algorithm_version": np.int64(SEPARATION_ALGORITHM_VERSION),
        "clock_estimator": np.str_(
            "adjacent_cycle_time_domain_three_subwindow_parabolic"
        ),
        "clock_band_hz": np.asarray(CLOCK_BAND_HZ, dtype=np.float64),
        "clock_min_adjacent_score": np.float64(CLOCK_MIN_ADJACENT_SCORE),
        "clock_max_err_ref_delta_samples": np.float64(
            CLOCK_MAX_ERR_REF_DELTA_SAMPLES
        ),
        "clock_max_subwindow_spread_samples": np.float64(
            CLOCK_MAX_SUBWINDOW_SPREAD_SAMPLES
        ),
        "clock_max_adjacent_change_samples": np.float64(
            CLOCK_MAX_ADJACENT_CHANGE_SAMPLES
        ),
        "clock_max_abs_period_delta_samples": np.float64(
            CLOCK_MAX_ABS_PERIOD_DELTA_SAMPLES
        ),
        "clock_observation_repeat_indices": clock_indices,
        "clock_period_delta_samples": np.asarray(
            separation["common_delay_samples"], dtype=np.float64
        )[clock_valid],
        "clock_q_ratio": np.asarray(separation["q"], dtype=np.float64)[clock_valid],
        "clock_err_delay_samples": np.asarray(
            separation["err_delay_samples"], dtype=np.float64
        )[clock_valid],
        "clock_ref_delay_samples": np.asarray(
            separation["ref_delay_samples"], dtype=np.float64
        )[clock_valid],
        "clock_err_score": np.asarray(
            separation["err_score"], dtype=np.float64
        )[clock_valid],
        "clock_ref_score": np.asarray(
            separation["ref_score"], dtype=np.float64
        )[clock_valid],
        "clock_err_subwindow_spread_samples": np.asarray(
            separation["err_subwindow_spread_samples"], dtype=np.float64
        )[clock_valid],
        "clock_ref_subwindow_spread_samples": np.asarray(
            separation["ref_subwindow_spread_samples"], dtype=np.float64
        )[clock_valid],
        "clock_err_ref_delta_samples": np.asarray(
            separation["err_ref_delta_samples"], dtype=np.float64
        )[clock_valid],
        "joint_ls_expected_rank": np.int64(
            2 * np.sort(np.r_[probe.noise_bins, probe.cancel_bins]).size
        ),
        "joint_ls_rank": np.asarray(
            separation["joint_ls_rank"], dtype=np.int64
        )[clock_valid],
        "joint_ls_condition": np.asarray(
            separation["joint_ls_condition"], dtype=np.float64
        )[clock_valid],
        "joint_ls_max_condition": np.float64(JOINT_LS_MAX_CONDITION),
        "joint_ls_reconstruction_relative_error": np.asarray(
            separation["joint_ls_reconstruction_relative_error"], dtype=np.float64
        )[clock_valid],
        "joint_ls_reconstruction_relative_error_p95": np.float64(
            separation["joint_ls_reconstruction_relative_error_p95"]
        ),
        "joint_ls_max_reconstruction_relative_error_p95": np.float64(
            JOINT_LS_MAX_RESIDUAL_P95
        ),
        "separation_crosscheck_band_hz": np.asarray(
            crosscheck["overall"]["band_hz"], dtype=np.float64
        ),
        "separation_crosscheck_complex_agreement": np.float64(
            crosscheck["overall"]["complex_agreement"]
        ),
        "separation_crosscheck_relative_error": np.float64(
            crosscheck["overall"]["relative_error"]
        ),
        "separation_crosscheck_subband_hz": np.asarray(
            [row["band_hz"] for row in crosscheck_subbands], dtype=np.float64
        ),
        "separation_crosscheck_subband_complex_agreement": np.asarray(
            [row["complex_agreement"] for row in crosscheck_subbands],
            dtype=np.float64,
        ),
        "separation_crosscheck_subband_relative_error": np.asarray(
            [row["relative_error"] for row in crosscheck_subbands],
            dtype=np.float64,
        ),
        "minimum_separation_crosscheck_agreement": np.float64(
            SEPARATION_CROSSCHECK_MIN_AGREEMENT
        ),
        "maximum_separation_crosscheck_relative_error": np.float64(
            SEPARATION_CROSSCHECK_MAX_RELATIVE_ERROR
        ),
        "compact_transfer_band_hz": np.asarray(
            round_trip["band_hz"], dtype=np.float64
        ),
        "compact_transfer_tone_count": np.int64(round_trip["tone_count"]),
        "compact_transfer_complex_agreement": np.float64(
            round_trip["complex_agreement"]
        ),
        "compact_transfer_relative_error": np.float64(
            round_trip["relative_error"]
        ),
        "compact_transfer_subband_hz": np.asarray(
            [item["band_hz"] for item in subbands], dtype=np.float64
        ),
        "compact_transfer_subband_tone_count": np.asarray(
            [item["tone_count"] for item in subbands], dtype=np.int64
        ),
        "compact_transfer_subband_complex_agreement": np.asarray(
            [item["complex_agreement"] for item in subbands], dtype=np.float64
        ),
        "compact_transfer_subband_relative_error": np.asarray(
            [item["relative_error"] for item in subbands], dtype=np.float64
        ),
        "minimum_compact_transfer_agreement": np.float64(
            round_trip["minimum_complex_agreement"]
        ),
        "maximum_compact_transfer_relative_error": np.float64(
            round_trip["maximum_relative_error"]
        ),
    }


def analysis_provenance_arrays(
    results: dict[str, dict[str, Any]], report: dict[str, Any]
) -> dict[str, np.ndarray]:
    """official separation 증거를 hash된 analysis NPZ에 독립 재감사용으로 보존한다.

    official NPZ의 요약 숫자만 다시 읽으면 exporter가 같은 잘못된 숫자를 여러 필드에
    복사해도 잡을 수 없다. 따라서 full repeat 축의 clock witness, joint-LS 결과와
    cubic crosscheck를 analysis artifact에 함께 넣고 readiness가 official kept subset과
    직접 대조한다.
    """

    separation = report["separation"]
    return {
        "noise_transfers": np.asarray(
            results["noise"]["model"]["repeat_transfers"], dtype=np.complex128
        ),
        "cancel_transfers": np.asarray(
            results["cancel"]["model"]["repeat_transfers"], dtype=np.complex128
        ),
        "noise_frequencies_hz": np.asarray(
            results["noise"]["model"]["frequencies_hz"], dtype=np.float64
        ),
        "cancel_frequencies_hz": np.asarray(
            results["cancel"]["model"]["frequencies_hz"], dtype=np.float64
        ),
        "clock_valid_mask": np.asarray(separation["valid"], dtype=np.bool_),
        "clock_q_ratio": np.asarray(separation["q"], dtype=np.float64),
        "clock_period_delta_samples": np.asarray(
            separation["common_delay_samples"], dtype=np.float64
        ),
        "clock_err_delay_samples": np.asarray(
            separation["err_delay_samples"], dtype=np.float64
        ),
        "clock_ref_delay_samples": np.asarray(
            separation["ref_delay_samples"], dtype=np.float64
        ),
        "clock_err_score": np.asarray(separation["err_score"], dtype=np.float64),
        "clock_ref_score": np.asarray(separation["ref_score"], dtype=np.float64),
        "clock_err_subwindow_spread_samples": np.asarray(
            separation["err_subwindow_spread_samples"], dtype=np.float64
        ),
        "clock_ref_subwindow_spread_samples": np.asarray(
            separation["ref_subwindow_spread_samples"], dtype=np.float64
        ),
        "clock_err_ref_delta_samples": np.asarray(
            separation["err_ref_delta_samples"], dtype=np.float64
        ),
        "joint_ls_rank": np.asarray(separation["joint_ls_rank"], dtype=np.int64),
        "joint_ls_condition": np.asarray(
            separation["joint_ls_condition"], dtype=np.float64
        ),
        "joint_ls_reconstruction_relative_error": np.asarray(
            separation["joint_ls_reconstruction_relative_error"], dtype=np.float64
        ),
        "common_alignment_tau_samples": np.asarray(
            report["common_alignment_taus"], dtype=np.float64
        ),
        "noise_provisional_tau_samples": np.asarray(
            results["noise"]["model"]["provisional_taus"], dtype=np.float64
        ),
        "cancel_provisional_tau_samples": np.asarray(
            results["cancel"]["model"]["provisional_taus"], dtype=np.float64
        ),
        "noise_cubic_crosscheck_transfers": np.asarray(
            separation["crosscheck_transfers"]["noise"], dtype=np.complex128
        ),
        "cancel_cubic_crosscheck_transfers": np.asarray(
            separation["crosscheck_transfers"]["cancel"], dtype=np.complex128
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.dry_run and not (
        args.confirm_user_present
        and args.confirm_volume_minimum
        and args.confirm_routing_and_geometry
    ):
        print(
            "[중단] 스피커가 울립니다. 사용자 입회로 앰프 볼륨 최저와 ERR/REF·"
            "noise/cancel 배선/기하를 확인한 뒤 --confirm-user-present, "
            "--confirm-volume-minimum 및 --confirm-routing-and-geometry 를 모두 "
            "지정하세요.",
            file=sys.stderr,
        )
        return 2
    if not args.dry_run and not (
        args.meter_raw and args.confirm_same_amplifier_setting
    ):
        print(
            "[중단] 모든 strict live는 바로 직전 fresh PASS meter raw와 같은 앰프 "
            "노브 확인이 필수입니다: --meter-raw <meter_raw.npz> "
            "--confirm-same-amplifier-setting",
            file=sys.stderr,
        )
        return 2

    try:
        hardware_config = load_yaml(REPO_ROOT / args.hardware)
        hardware, channel_map = validate_hardware_contract(hardware_config)
        # dry-run도 read-only proc/sys fingerprint까지 검증한다. official identity는
        # 논리 YAML만으로 만들 수 없고 실제 codec/DAC fingerprint가 항상 필수다.
        physical_fingerprint = collect_alsa_physical_fingerprint(hardware_config)
        hardware_identity = measurement_hardware_identity(
            hardware_config,
            physical_fingerprint=physical_fingerprint,
        )
        fs = int(hardware["sample_rate"])
        block_size = (
            int(args.block_size)
            if args.block_size is not None
            else int(hardware["block_size"])
        )
        # 매우 큰 period override가 probe 배열을 만들기 전에 즉시 거부되도록 한다.
        validate_official_capture_recipe_args(args)
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
        channel_band = validate_analysis_contract(
            args, probe, fs=fs, block_size=block_size
        )
    except (KeyError, OSError, RuntimeError, ValueError, FileExistsError) as exc:
        print(f"[중단] {exc}", file=sys.stderr)
        return 2

    need_lo, need_hi = float(args.required_band[0]), float(args.required_band[1])
    resolution = fs / probe.period_samples
    durations = measurement_duration_plan(
        args, sample_rate=fs, period_samples=probe.period_samples
    )
    level_evidence_error: str | None = None
    level_evidence_payload: dict[str, Any] | None = None
    level_evidence_path: Path | None = None
    level_evidence_sha256: str | None = None
    fresh_meter: dict[str, Any] | None = None
    if args.bootstrap_level_evidence:
        if args.dry_run:
            level_evidence_error = (
                "bootstrap dry-run: live에서는 fresh meter raw+receipt와 "
                "same-amplifier confirmation을 검증합니다"
            )
        else:
            try:
                evidence_target = Path(args.level_evidence)
                if not evidence_target.is_absolute():
                    evidence_target = REPO_ROOT / evidence_target
                evidence_target = evidence_target.resolve()
                evidence_target.relative_to(REPO_ROOT.resolve())
                if evidence_target.exists():
                    raise FileExistsError(
                        "canonical level evidence가 이미 있어 bootstrap을 다시 실행할 수 "
                        f"없습니다: {evidence_target}"
                    )
                if not evidence_target.parent.is_dir():
                    raise FileNotFoundError(
                        f"level evidence 상위 디렉터리가 없습니다: {evidence_target.parent}"
                    )
                if not os.access(evidence_target.parent, os.W_OK):
                    raise PermissionError(
                        f"level evidence 상위 디렉터리에 쓸 수 없습니다: {evidence_target.parent}"
                    )
            except (FileNotFoundError, OSError, ValueError) as exc:
                level_evidence_error = str(exc)
    else:
        try:
            level_evidence_payload = load_measurement_level_evidence(
                args.level_evidence,
                repository_root=REPO_ROOT,
            )
            verified_path = level_evidence_payload.get("_evidence_path")
            if verified_path:
                level_evidence_path = Path(str(verified_path)).resolve()
            level_evidence_sha256 = level_evidence_payload.get("_evidence_sha256")
            if level_evidence_payload.get("hardware_identity") != hardware_identity:
                raise ValueError(
                    "영구 calibration evidence의 hardware identity가 현재 live "
                    "hardware/channel 계약과 다릅니다"
                )
        except (FileNotFoundError, OSError, ValueError) as exc:
            level_evidence_error = str(exc)

    if not args.dry_run and level_evidence_error is None:
        try:
            fresh_meter = validate_bootstrap_meter_raw(
                args.meter_raw,
                repository_root=REPO_ROOT,
                expected_hardware_identity=hardware_identity,
                require_fresh=True,
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            level_evidence_error = str(exc)

    crest_noise, crest_cancel = probe.crest_db()
    if max(crest_noise, crest_cancel) > MAX_CREST_DB:
        print(
            f"[중단] 크레스트 {crest_noise:.1f}/{crest_cancel:.1f} dB 가 "
            f"{MAX_CREST_DB} dB 를 넘습니다 — 같은 피크에서 음향 에너지를 잃습니다.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        # 이 경로는 sounddevice를 import하지 않고 ALSA의 read-only 카드 표만 확인한다.
        # PortAudio 조회조차 환경에 따라 장치를 준비/open할 수 있으므로 실제 preflight와
        # 스트림 검증은 명시적 물리 확인 뒤의 측정 경로에만 둔다.
        try:
            input_cfg = hardware["input"]
            output_cfg = hardware["output"]
            if int(input_cfg.get("channels", 0)) < 2:
                raise ValueError("입력 장치가 ERR/REF 2채널로 설정되지 않았습니다")
            if int(output_cfg.get("channels", 0)) < 2:
                raise ValueError("출력 장치가 noise/cancel 2채널로 설정되지 않았습니다")
            input_card = alsa_card_index(str(input_cfg["card"]))
            output_card = alsa_card_index(str(output_cfg["card"]))
            input_pcm = int(input_cfg["pcm"])
            output_pcm = int(output_cfg["pcm"])
            validate_alsa_pcm_mapping(
                input_card=input_card,
                input_pcm=input_pcm,
                output_card=output_card,
                output_pcm=output_pcm,
            )
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            print(f"[DRY-RUN 실패] 장치 매핑: {exc}", file=sys.stderr)
            return 2
        evidence_line = (
            "  live level evidence=PASS\n"
            if level_evidence_error is None
            else f"  live level evidence=BLOCKED ({level_evidence_error})\n"
        )
        print(
            "[DRY-RUN PASS] 재생/녹음/파일 생성 없이 검증했습니다.\n"
            f"  fs={fs}, block={block_size}, amplitude={args.amplitude:.4f}, "
            f"warmup/repeats={args.warmup_periods}/{args.repeats}\n"
            f"  input-only preflight={durations['input_preflight_seconds']:.2f}s, "
            f"output nominal={durations['output_stream_seconds']:.2f}s / "
            f"hard-max={durations['output_hard_max_seconds']:.2f}s "
            f"(silent lead-in {durations['lead_in_seconds']:.2f}s + "
            f"stimulus {durations['stimulus_seconds']:.2f}s)\n"
            f"  level meter target={OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs:+.1f}dBFS "
            f"±{OFFICIAL_MEASUREMENT_LEVEL.meter_tolerance_db:.1f}dB\n"
            + evidence_line
            +
            f"  input=hw:{input_card},{input_pcm} (2ch), "
            f"output=hw:{output_card},{output_pcm} (2ch)\n"
            f"  channels={channel_map}\n"
            f"  official outputs fresh: {primary_out}, {secondary_out}\n"
            "  실제 측정은 세 operator confirmation 없이 시작되지 않습니다."
        )
        return 0

    if level_evidence_error is not None:
        print(
            "[중단] 레벨 계약 offline evidence gate 실패: "
            f"{level_evidence_error}",
            file=sys.stderr,
        )
        return 2

    capture_id = uuid.uuid4().hex
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    # raw capture 쌍에는 이 시각과 캡처 전제만 기록하고 이후 절대 덮어쓰지 않는다.
    started_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    session_dir = create_session_directory(
        diagnostics_root, f"{stamp}_{capture_id[:8]}"
    )

    print(
        f"동시 인터리브 측정 {args.band[0]:.0f}-{args.band[1]:.0f}Hz · "
        f"톤 간격 {probe.bin_step('noise') * resolution:.2f}Hz · "
        f"주기 {args.period_seconds:.2f}s\n"
        f"  톤 수 noise {probe.noise_bins.size} / cancel {probe.cancel_bins.size} · "
        f"guard {probe.guard_bins()} bin · crest {crest_noise:.1f}/{crest_cancel:.1f} dB\n"
        f"  peak {args.amplitude:.4f} · block {block_size} · latency {args.latency} · "
        f"warmup {args.warmup_periods} + 분석 {args.repeats} 주기 "
        f"(자극 {durations['stimulus_seconds']:.2f}초, "
        f"무음 lead-in 포함 nominal {durations['output_stream_seconds']:.2f}초, "
        f"hard-max {durations['output_hard_max_seconds']:.2f}초)"
    )

    audio_lock_manager = None
    audio_lock: dict[str, Any] | None = None

    def _release_audio_lock() -> None:
        nonlocal audio_lock_manager
        if audio_lock_manager is not None:
            manager, audio_lock_manager = audio_lock_manager, None
            manager.__exit__(None, None, None)

    try:
        import sounddevice as sd

        pending_lock_manager = repository_audio_lock(
            REPO_ROOT, purpose="strict_interleaved_paths"
        )
        audio_lock = pending_lock_manager.__enter__()
        audio_lock_manager = pending_lock_manager
        print(
            f"저장소/UID audio lock 획득: {audio_lock['path']} (pid={audio_lock['pid']}). "
            "다른 저장소나 lock 미준수 프로세스는 장치 점유 gate로만 방어합니다."
        )

        # 3초 input-only official preflight를 정확히 한 번만 연다. 1초 I2S settle은
        # 이 총시간 안에 포함하고, 남은 2초 raw로 ERR/REF 생존·rail을 판정한다.
        # PCM/clock은 read-only로 먼저 확인하므로 점유 상태에서 입력도 열지 않는다.
        assert_live_pcm_clock_preconditions(hardware)
        print("출력 없는 ERR/REF raw preflight 중...")
        analyzed_preflight_seconds = (
            float(args.input_probe_seconds)
            - float(cw.DEFAULT_PROBE_SETTLE_SECONDS)
        )
        if analyzed_preflight_seconds <= 0.0:
            raise ValueError("input preflight total은 I2S settle보다 길어야 합니다")
        preflight_raw, preflight_report = cw._capture_preflight(
            sd, hardware, analyzed_preflight_seconds
        )
        for name, item in zip(("ERR", "REF"), cw._probe_summary(preflight_report)):
            verdict = "PASS" if item["valid"] else "FAIL"
            print(
                f"[{verdict}] {name}: RMS {item['rms_dbfs']:.2f}dBFS, "
                f"peak {item['peak']:.6f}, clip {item['clip_ratio']:.3%}"
            )
        channels = preflight_report.get("channels", [])
        required_input_indices = (
            channel_map["error_mic"],
            channel_map["reference_mic"],
        )
        if len(channels) < 2 or not all(
            bool(channels[index].get("valid")) for index in required_input_indices
        ):
            _release_audio_lock()
            print("[실패] 양 마이크 preflight 실패 — 출력 장치를 열지 않았습니다", file=sys.stderr)
            return 1

        in_dev = int(preflight_report["device"])
        output_cfg = hardware["output"]
        out_dev = resolve_alsa_portaudio_device(
            output_cfg["card"], output_cfg["pcm"], "output", 2
        )
        if fresh_meter is not None:
            meter_devices = fresh_meter["metadata"].get("resolved_devices")
            strict_devices = {"input": int(in_dev), "output": int(out_dev)}
            if meter_devices != strict_devices:
                _release_audio_lock()
                print(
                    "[실패] meter 이후 resolved PortAudio device가 달라졌습니다. "
                    f"meter={meter_devices!r}, strict={strict_devices!r}. "
                    "출력 장치는 열지 않았습니다.",
                    file=sys.stderr,
                )
                return 1

        lead_in = int(round(OUTPUT_LEAD_IN_SECONDS * fs))
        total_periods = int(args.warmup_periods) + int(args.repeats)
        playback = np.zeros((lead_in + total_periods * probe.period_samples, 2), np.float32)
        playback[lead_in:, channel_map["noise_out"]] = np.tile(
            probe.noise_signal, total_periods
        )
        playback[lead_in:, channel_map["cancel_out"]] = np.tile(
            probe.cancel_signal, total_periods
        )

        def _capture_metadata(
            captured_telemetry: dict[str, Any],
            raw_invalid_reasons: list[str],
            extra: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """캡처 반환 즉시 쓸 수 있는 분석 비의존 canonical metadata."""

            base: dict[str, Any] = {
                "capture_id": capture_id,
                "method": METHOD,
                "raw_capture_schema": RAW_CAPTURE_SCHEMA,
                "started_at_utc": started_at_utc,
                "sample_rate": fs,
                "block_size": block_size,
                "latency": args.latency,
                "hardware_identity": hardware_identity,
                "audio_lock": dict(audio_lock or {}),
                "resolved_devices": {"input": int(in_dev), "output": int(out_dev)},
                "channel_map": dict(channel_map),
                "operator_confirmations": {
                    "user_present": bool(args.confirm_user_present),
                    "volume_minimum": bool(args.confirm_volume_minimum),
                    "routing_and_geometry": bool(
                        args.confirm_routing_and_geometry
                    ),
                },
                "amplitude": float(args.amplitude),
                "design_band_hz": [float(args.band[0]), float(args.band[1])],
                "required_band_hz": [need_lo, need_hi],
                "channel_band_hz": {k: list(v) for k, v in channel_band.items()},
                "tone_spacing_hz": float(probe.bin_step("noise") * resolution),
                "period_seconds": float(args.period_seconds),
                "warmup_periods": int(args.warmup_periods),
                "repeats": int(args.repeats),
                "lead_in_samples": int(lead_in),
                "guard_bins": probe.guard_bins(),
                "crest_db": {"noise": crest_noise, "cancel": crest_cancel},
                "warp": {"applied": False},
                "telemetry": captured_telemetry,
                "preflight": preflight_report,
                "measurement_level_evidence": {
                    "path": (
                        str(level_evidence_path.relative_to(REPO_ROOT))
                        if level_evidence_path is not None
                        else str(args.level_evidence)
                    ),
                    "sha256": level_evidence_sha256,
                    "payload": {
                        key: value
                        for key, value in (level_evidence_payload or {}).items()
                        if not key.startswith("_")
                    },
                },
                "measurement_level_bootstrap": {
                    "enabled": bool(args.bootstrap_level_evidence),
                    "meter_raw_path": (
                        str(fresh_meter["path"].relative_to(REPO_ROOT))
                        if fresh_meter is not None
                        else None
                    ),
                    "meter_raw_sha256": (
                        fresh_meter.get("sha256")
                        if fresh_meter is not None
                        else None
                    ),
                    "meter_completed_at_utc": (
                        fresh_meter["completed_at_utc"].isoformat()
                        if fresh_meter is not None
                        else None
                    ),
                    "max_meter_age_seconds": BOOTSTRAP_METER_MAX_AGE_SECONDS,
                    "same_amplifier_setting_confirmed": bool(
                        args.confirm_same_amplifier_setting
                    ),
                },
                "invalid_reasons": list(raw_invalid_reasons),
                "analysis_contract": {
                    "fit_band_hz": [float(args.fit_band[0]), float(args.fit_band[1])],
                    "consistency_band_hz": [
                        float(args.consistency_band[0]),
                        float(args.consistency_band[1]),
                    ],
                    "required_band_hz": [need_lo, need_hi],
                    "fir_length": int(args.fir_length),
                    "pre_roll_samples": int(args.pre_roll),
                    "max_delay_samples": int(max_delay),
                    "min_alignment_score": float(args.min_alignment_score),
                    "min_kept_repeats": int(args.min_kept_repeats),
                    "max_relative_tau_samples": float(args.max_relative_tau_samples),
                    "max_drift_deviation_samples": float(
                        args.max_drift_deviation_samples
                    ),
                    "max_delay_jitter_samples": int(max_jitter),
                    "clock_band_hz": list(CLOCK_BAND_HZ),
                    "clock_min_adjacent_score": CLOCK_MIN_ADJACENT_SCORE,
                    "clock_max_err_ref_delta_samples": (
                        CLOCK_MAX_ERR_REF_DELTA_SAMPLES
                    ),
                    "clock_max_subwindow_spread_samples": (
                        CLOCK_MAX_SUBWINDOW_SPREAD_SAMPLES
                    ),
                    "clock_max_adjacent_change_samples": (
                        CLOCK_MAX_ADJACENT_CHANGE_SAMPLES
                    ),
                    "clock_max_abs_period_delta_samples": (
                        CLOCK_MAX_ABS_PERIOD_DELTA_SAMPLES
                    ),
                    "separation_algorithm": SEPARATION_ALGORITHM,
                    "separation_algorithm_version": SEPARATION_ALGORITHM_VERSION,
                },
            }
            if extra:
                base.update(extra)
            return base

        # playback 구성·metadata closure 이후, output stream open에 가장 가까운 지점에서
        # 실제 codec/DAC identity, PCM 점유/clock/입력과 fresh meter raw bytes를
        # 다시 검증한다.
        def _pre_open_check() -> None:
            # output PCM/raw buffer 준비가 끝난 뒤 sd.Stream open 직전 이 순서로
            # fresh bytes → physical codec/DAC → read-only PCM/clock을 재검증한다.
            refreshed_meter = validate_bootstrap_meter_raw(
                args.meter_raw,
                repository_root=REPO_ROOT,
                expected_hardware_identity=hardware_identity,
                require_fresh=True,
            )
            if (
                fresh_meter is None
                or refreshed_meter["sha256"] != fresh_meter["sha256"]
            ):
                raise RuntimeError("preflight 이후 fresh meter raw/receipt가 변경됐습니다")
            refreshed_physical = collect_alsa_physical_fingerprint(hardware_config)
            if refreshed_physical != physical_fingerprint:
                raise RuntimeError(
                    "output 직전 ALSA physical fingerprint가 preflight 이후 변경됐습니다"
                )
            assert_live_pcm_clock_preconditions(hardware)

        recorded_raw, output_pcm, telemetry = capture_with_speaker_release_notice(
            lambda: capture_measurement_preserving_partial(
                sd,
                fs=fs,
                block_size=block_size,
                latency=str(args.latency),
                in_dev=in_dev,
                out_dev=out_dev,
                output_float=playback,
                meter_completed_at_utc=fresh_meter["completed_at_utc"],
                pre_open_check=_pre_open_check,
            )
        )
        _release_audio_lock()
    except PartialCaptureError as exc:
        _release_audio_lock()
        recorded_raw = exc.recorded_raw
        output_pcm = exc.output_pcm
        telemetry = exc.telemetry
        invalid = capture_telemetry_invalid_reasons(telemetry)
        if "capture_incomplete" not in invalid:
            invalid.append("capture_incomplete")
        try:
            with defer_termination_signals_during_raw_commit():
                partial_paths = write_immutable_raw_capture_atomic(
                    session_dir,
                    metadata=_capture_metadata(telemetry, invalid),
                    arrays={
                        "output": playback,
                        "output_pcm_int16": output_pcm,
                        "input_raw_int32": recorded_raw,
                        "preflight_raw_int32": preflight_raw,
                    },
                )
            recovery = f"invalid immutable raw 저장: {partial_paths['raw']}"
        except DeferredTerminationSignal as save_exc:
            raw_path = session_dir / "raw_measurement.npz"
            print(
                f"[중단] signal {save_exc.signum}; partial immutable raw 보존: "
                f"{raw_path}",
                file=sys.stderr,
            )
            return save_exc.exit_code
        except RawCaptureSidecarError as save_exc:
            recovery = str(save_exc)
        except (OSError, ValueError) as save_exc:
            recovery = f"partial raw 저장도 실패: {save_exc}"
        print(f"[실패] 측정 중단: {exc}; {recovery}", file=sys.stderr)
        return 1
    except (ImportError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        _release_audio_lock()
        print(f"[실패] 측정 중단: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # ── 원시 캡처를 **분석보다 먼저** 저장한다 ──────────────────────────────
    # 스피커를 울린 시간은 되돌릴 수 없다(사용자 지시: 스피커 구동 하드웨어 수명).
    # 분석 게이트는 정당하게 실패할 수 있고 실제로 실패한다 —
    # 2026-08-06: 드리프트 중앙 3.95 / −4.03 샘플/주기로 두 번 연속 전량 기각됐는데,
    # 저장이 분석 뒤에 있어서 당시 **6초씩 두 번을 그냥 버렸다.** 현재 기본 자극은
    # 12초이고, 0.5초 무음 lead-in을 포함한 출력 스트림은 12.5초다.
    # reanalyse_paths_interleaved.py 가 저장된 캡처만으로 오프라인 재분석을 하므로
    # (실제로 그렇게 플랜트를 복구했다), 먼저 저장해 두면 실패해도 살릴 수 있다.

    raw_invalid = capture_telemetry_invalid_reasons(telemetry)
    try:
        with defer_termination_signals_during_raw_commit():
            raw_paths = write_immutable_raw_capture_atomic(
                session_dir,
                metadata=_capture_metadata(telemetry, raw_invalid),
                arrays={
                    "output": playback,
                    # callback이 실제 outdata에 복사한 정확한 int16 command다. ideal
                    # float probe만 저장하면 향후 양자화/변환 회귀를 독립 감사할 수 없다.
                    "output_pcm_int16": output_pcm,
                    "input_raw_int32": recorded_raw,
                    "preflight_raw_int32": preflight_raw,
                },
            )
    except DeferredTerminationSignal as exc:
        raw_path = session_dir / "raw_measurement.npz"
        print(
            f"[중단] signal {exc.signum}은 raw commit 뒤 처리했습니다. "
            f"immutable raw 보존: {raw_path}",
            file=sys.stderr,
        )
        return exc.exit_code
    except RawCaptureSidecarError as exc:
        print(f"[실패] {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        raw_path = session_dir / "raw_measurement.npz"
        recovery = (
            f" canonical raw가 남아 있습니다: {raw_path}"
            if raw_path.is_file()
            else " final raw는 노출되지 않았습니다"
        )
        print(f"[실패] raw capture 저장: {exc}.{recovery}", file=sys.stderr)
        return 1
    try:
        raw_display = raw_paths["raw"].relative_to(REPO_ROOT)
    except ValueError:
        raw_display = raw_paths["raw"]
    print(
        f"원시 캡처 저장: {raw_display}"
        " — 분석이 실패해도 reanalyse_paths_interleaved.py 로 되살릴 수 있다"
    )

    # canonical evidence가 아직 없는 최초 1회만, 방금 보존한 strict raw를 paired
    # interleaved half로 쓴다. 별도 audible probe는 절대 추가하지 않는다. 이 함수는
    # meter receipt/SHA/freshness/device/recipe/status/target과 strict raw의 PCM·입력
    # clipping을 전부 temp JSON에서 재검증한 뒤 final evidence를 한 번만 노출한다.
    # 따라서 evidence가 고정되기 전에는 아래 official P/S 분석/NPZ 승격으로 가지 않는다.
    if args.bootstrap_level_evidence:
        assert fresh_meter is not None
        try:
            level_evidence_payload = create_measurement_level_evidence_atomic(
                args.level_evidence,
                repository_root=REPO_ROOT,
                meter_raw_path=fresh_meter["path"],
                interleaved_raw_path=raw_paths["raw"],
                hardware_identity=hardware_identity,
            )
            level_evidence_path = Path(
                str(level_evidence_payload["_evidence_path"])
            ).resolve()
            level_evidence_sha256 = str(
                level_evidence_payload["_evidence_sha256"]
            )
        except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
            print(
                "[실패] paired level evidence 생성/재검증 실패. official 분석/NPZ는 "
                f"승격하지 않습니다: {exc}. immutable strict raw는 보존했습니다: "
                f"{raw_paths['raw']}",
                file=sys.stderr,
            )
            return 1
        print(
            "canonical paired level evidence 원자 승격 PASS: "
            f"{level_evidence_path.relative_to(REPO_ROOT)}\n"
            f"  evidence SHA256 {level_evidence_sha256}"
        )

    try:
        recorded = pcm_int32_to_float32(recorded_raw)
        err = recorded[:, channel_map["error_mic"]].astype(np.float64)
        measurement_report = cw.analyze_int32_input_probe(recorded_raw)
        invalid = capture_invalid_reasons(telemetry, measurement_report)
    except BaseException as exc:
        print(
            f"[실패] raw 저장 후 입력 후처리 중단: {type(exc).__name__}: {exc}. "
            f"immutable raw는 보존했습니다: {raw_paths['raw']}",
            file=sys.stderr,
        )
        return 1

    period_starts = [
        lead_in + (int(args.warmup_periods) + k) * probe.period_samples
        for k in range(int(args.repeats))
    ]

    # 배경잡음 스펙트럼은 preflight 를 **같은 길이·같은 FFT** 로 변환해야 분모가 맞는다.
    preflight_err = pcm_int32_to_float32(preflight_raw)[
        :, channel_map["error_mic"]
    ].astype(np.float64)
    if preflight_err.size < probe.period_samples:
        preflight_err = np.pad(preflight_err, (0, probe.period_samples - preflight_err.size))
    noise_spectrum = np.fft.rfft(preflight_err[-probe.period_samples :])
    signal_spectrum = np.fft.rfft(err[period_starts[0] : period_starts[0] + probe.period_samples])

    fit_band = (float(args.fit_band[0]), float(args.fit_band[1]))
    consistency_band = (
        float(args.consistency_band[0]), float(args.consistency_band[1])
    )
    try:
        results, report = analyse_capture(
            err=err,
            ref=recorded[:, channel_map["reference_mic"]].astype(np.float64),
            output_pcm_int16=output_pcm,
            probe=probe,
            period_starts=period_starts,
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
            f"  effective 지연 {model['delay_samples']} 샘플 "
            f"(bulk {model['bulk_delay_samples']}, pre-roll {model['pre_roll_samples']}) · "
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

    # 분석 metadata는 별도 파일이다. immutable raw metadata와 역할을 섞지 않는다.
    metadata = _capture_metadata(telemetry, invalid, {
        "measurement": measurement_report,
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
                "bulk_delay_samples": item["model"]["bulk_delay_samples"],
                "bulk_delay_fractional_samples": item["model"][
                    "bulk_delay_fractional_samples"
                ],
                "pre_roll_samples": item["model"]["pre_roll_samples"],
                "delay_semantics": item["model"]["delay_semantics"],
                "compact_transfer_round_trip": item["model"][
                    "compact_transfer_round_trip"
                ],
                "aligned_mean_transfer_sha256": aligned_transfer_sha256(
                    item["model"]["frequencies_hz"], item["model"]["mean_transfer"]
                ),
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
    })

    try:
        analysis_paths = write_analysis_outputs_atomic(
            session_dir,
            metadata=metadata,
            arrays={
                **analysis_provenance_arrays(results, report),
                "noise_ir": results["noise"]["model"]["ir"].astype(np.float64),
                "cancel_ir": results["cancel"]["model"]["ir"].astype(np.float64),
                "frequencies_hz": results["noise"]["model"]["frequencies_hz"],
                "relative_tau_samples": relative_tau,
                "noise_snr_db": results["noise"]["snr_db"].astype(np.float64),
                "cancel_snr_db": results["cancel"]["snr_db"].astype(np.float64),
            },
        )
    except (OSError, ValueError) as exc:
        print(
            f"[실패] 분석 산출물 저장: {exc}. immutable raw capture는 보존했습니다: "
            f"{session_dir}",
            file=sys.stderr,
        )
        return 1
    print(
        f"분석 산출물 저장: {analysis_paths['results'].name}, "
        f"{analysis_paths['metadata'].name} (raw capture 불변)"
    )

    source_raw_sha256 = hashlib.sha256(raw_paths["raw"].read_bytes()).hexdigest()
    source_analysis_sha256 = hashlib.sha256(
        analysis_paths["results"].read_bytes()
    ).hexdigest()
    source_raw_path = str(raw_paths["raw"].relative_to(REPO_ROOT))
    source_analysis_path = str(analysis_paths["results"].relative_to(REPO_ROOT))

    if not valid:
        print(f"\n[실패] 정식 모델을 저장하지 않았습니다. 진단: {session_dir}", file=sys.stderr)
        if invalid:
            print(f"  캡처 결함: {', '.join(invalid)}", file=sys.stderr)
        return 1

    official: dict[str, dict[str, Any]] = {}
    for drive in ("noise", "cancel"):
        item = results[drive]
        official[drive] = _official_arrays(
                model=item["model"],
                relative_delay_spread=relative_spread,
                max_delay_jitter_samples=max_jitter,
                fs=fs,
                consistency=item["model"]["consistency"],
                band_hz=channel_band[drive],
                amplitude=float(args.amplitude),
                block_size=block_size,
                latency=str(args.latency),
                channel_map=channel_map,
                operator_confirmations={
                    "user_present": bool(args.confirm_user_present),
                    "volume_minimum": bool(args.confirm_volume_minimum),
                    "routing_and_geometry": bool(
                        args.confirm_routing_and_geometry
                    ),
                },
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
                max_drift_deviation_samples=float(
                    args.max_drift_deviation_samples
                ),
                relative_tau_max_abs=float(report["relative_tau_max_abs"]),
                source_raw_npz_path=source_raw_path,
                source_raw_npz_sha256=source_raw_sha256,
                source_analysis_npz_path=source_analysis_path,
                source_analysis_npz_sha256=source_analysis_sha256,
                output_pcm_provenance=OUTPUT_PCM_PROVENANCE_OBSERVED,
                separation=report["separation"],
                separation_crosscheck=report["separation_crosscheck"],
        )
    try:
        write_official_pair_atomic(
            primary_out,
            official["noise"],
            secondary_out,
            official["cancel"],
        )
    except (OSError, ValueError, FileExistsError) as exc:
        print(
            f"[실패] P/S official pair 저장: {exc}. orphan official은 제거했고 "
            f"raw/analysis는 보존했습니다: {session_dir}",
            file=sys.stderr,
        )
        return 1

    print(
        official_pair_success_message(
            primary_out,
            secondary_out,
            repository_root=REPO_ROOT,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
