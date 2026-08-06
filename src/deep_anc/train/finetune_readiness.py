"""실측 파인튜닝의 진입·완료를 실패 폐쇄 방식으로 판정한다.

이 모듈은 오디오 장치를 열지 않는다. 측정 도구가 품질 게이트를 통과해 만든
P/S NPZ, recorded manifest/파일, 사전학습 checkpoint와 독립 평가 NPZ를 읽기만
한다. 하나라도 검증할 수 없으면 ``ok=False``이며, 파일의 존재만으로 measured
파인튜닝 또는 완료를 인정하지 않는다.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import torch

from ..config import REPO_ROOT
from ..data.manifest import read_manifest
# 지연·lead 부기의 단일 출처. 게이트가 lead 를 **스스로 유도하면** 그것이 두 번째
# 유도가 되고, trainer 와 갈라진 채로 양쪽 다 "통과" 한다 (발생기 A, 커밋 aaeef41).
from ..dsp.invariants import check_lead_agreement
from ..dsp.secondary_path import load_secondary_path
from ..dsp.timing import BandPlan, FrequencyBand, PlantDelays
from ..data.recorded_qa import (
    settings_from_data_config,
    validate_recorded_sessions,
)


DEFAULT_REQUIRED_PATH_BAND_HZ = (80.0, 1600.0)
DEFAULT_REQUIRED_SOURCE_FAMILIES = ("speech", "music", "environment", "machine")

# official P/S 를 만들 수 있는 측정 방식.
#
# ``ess``  — scripts/data/calibrate_wideband.py. 경로마다 따로 실행한다.
# ``interleaved_multitone`` — scripts/data/measure_paths_interleaved.py. 두 경로를 **한 번의
#   재생으로 동시에** 잰다. 재생(USB)과 녹음(I²S)이 다른 클록 도메인이라 두 측정이 떨어져
#   있으면 그 사이의 wander 가 **P 와 S 의 상대 지연**에 그대로 실리는데, ANC 가 실제로
#   요구하는 값이 바로 그 상대 지연(lead)이다. 동시 측정은 warp 를 두 경로에 공통으로
#   실어 상대 관계에서 상쇄시킨다.
#
# 이 방식은 게이트를 넓히는 것이 아니라 **좁힌다**: ESS 에는 없는 아래 항목을 추가로
# 요구하고, 무엇보다 두 파일이 같은 ``capture_id`` 를 갖는지 검사해 "같은 조건"을
# 진폭·블록·latency 값의 우연한 일치가 아니라 **같은 캡처였다는 사실**로 확인한다.
ALLOWED_PATH_METHODS = ("ess", "interleaved_multitone")
INTERLEAVED_REQUIRED_FIELDS = (
    "capture_id",
    "interleave_guard_bins",
    "analysis_period_seconds",
    "tone_count",
    "tone_snr_median_db",
    "tone_snr_min_db",
    # 일관성을 **어느 대역에서 쟀는지**. 이게 없으면 coherence_median 0.95 가 무엇에
    # 대한 0.95 인지 알 수 없고, 좁은 대역에서 잰 값으로 넓은 대역을 주장할 수 있다.
    "consistency_band_hz",
)
INTERLEAVED_MAX_PERIOD_SECONDS = 2.0   # 실측 위상 잔차가 2.26s 에서 2.33rad 로 무너진다
INTERLEAVED_MIN_TONE_COUNT = 64
INTERLEAVED_MIN_TONE_SNR_MEDIAN_DB = 12.0

MAX_RELATIVE_DELAY_SPREAD_SAMPLES = 3
"""P−S 상대 τ 의 유지 반복 내 spread 상한(샘플).

측정 스크립트가 쓴 ``max_delay_jitter_samples`` 를 그대로 믿으면 게이트가
**자기증명**이 된다 — 둘 다 같은 스크립트가 같은 NPZ 에 쓰므로 측정 시
``--max-delay-jitter-ms`` 를 키우면 검사가 조용히 사라진다. 실측 2026-08-05:
32 샘플 프레임 슬립이 허용치 48 을 통과해 형상 기준 50% 틀린 S(z) 로
파인튜닝 50,000 step 이 낭비됐다. 정상 캡처의 실측 spread 는 0.11~0.26 샘플이다.
"""

MIN_KEPT_REPEATS = 8
"""플랜트 아티팩트가 **유지한** 반복 수의 하한.

기각을 많이 한 것은 문제가 아니다 — 오히려 좋은 신호다. 2026-08-05 복구에서 채택
캡처는 48 반복 중 30 을 기각하고 18 을 남겼고, 그 기각이 정확히 옳은 조치였다.
문제가 되는 것은 **남은 것이 적을 때**다. 반복 3개로 평균한 플랜트는 한 번의 이상치가
형상을 지배한다. 그래서 기각 비율이 아니라 유지 개수에 하한을 둔다.

값 8 은 재분석 파라미터 봉투의 ``min_kept_repeats`` 와 같다 — 두 곳이 다르면
"재분석은 통과하는데 게이트는 실패" 같은 해석 불가능한 상태가 생긴다.
이전 값은 3 이었다.
"""

MIN_BAND_CONSISTENCY = 0.90
"""필수 대역 안 **모든** 부대역이 넘어야 하는 값.

총계는 에너지 가중이라 약한 대역을 숨긴다 — 실측에서 S 의 전대역 총계는
0.9984 인데 80-150Hz 부대역만 보면 0.706 이다(저역 에너지 비중 0.1%).
"모든 소리를 제거한다 — 평균이 아니라 최악값" 이 이 프로젝트의 목표다.
"""

ALLOWED_REANALYSIS_ENVELOPE: dict[str, tuple[float | None, float | None]] = {
    # 오프라인 재분석은 "파라미터를 바꿔 결과를 고르는" 유혹을 만든다. 아티팩트에
    # 박힌 파라미터가 이 봉투 안에 있는지 게이트가 **독립적으로** 확인한다.
    "min_alignment_score": (0.95, None),          # 하한만 강제
    "max_relative_tau_samples": (None, 3.0),      # 상한만 강제
    "max_drift_deviation_samples": (None, 2.0),
    "min_kept_repeats": (8, None),
}


def _min_groups_per_family_default() -> int:
    """G4 평가기가 쓰는 그룹 하한을 **그쪽에서 읽어온다** (단일 출처).

    진입 게이트와 G4 평가기가 각자 4 를 들고 있으면 언젠가 한쪽만 바뀐다. 그러면
    "진입은 통과했는데 완료에서 판정 불가" 같은 해석 불가능한 상태가 생기고, 그 상태를
    푸는 가장 쉬운 방법은 언제나 낮은 쪽에 맞추는 것이다 — 이 저장소에서 반복된
    발생기 A(같은 값을 두 곳에서 따로 정하고 대조하지 않는다)와 정확히 같은 모양이라
    여기서 미리 끊는다.

    지연 import 인 이유: ``eval.recorded`` 가 ``train.trainer`` 를 부르므로 모듈 최상단
    에서 가져오면 순환이 된다.
    """

    from ..eval.recorded import MIN_GROUPS_PER_FAMILY

    return int(MIN_GROUPS_PER_FAMILY)


MIN_GROUPS_PER_FAMILY_PER_SPLIT = 4
"""val/test 의 계열당 **독립 그룹** 하한.

⚠ 이 값의 단일 출처는 :data:`deep_anc.eval.recorded.MIN_GROUPS_PER_FAMILY` 다.
설정이 값을 주지 않으면 :func:`_min_groups_per_family_default` 로 그쪽을 읽는다.
아래 문서 문자열은 근거를 남기기 위한 것이고, 두 값이 갈라지면
``test_group_floor_has_a_single_source`` 가 실패한다.

왜 4 인가. G4 는 계열별 평균으로 "최악 계열"을 고르는데, 그 평균의 오차는 세그먼트
수가 아니라 **그룹 수**로 정해진다(같은 그룹 안 세그먼트는 독립이 아니다). 2026-08-05
실측: 계열 내 그룹 간 잔차 SD 1.46 dB → 그룹 2개일 때 평균의 SE 는 1.03 dB 인데
파인튜닝 후 계열 간 전체 폭은 0.92 dB 였다. **폭이 1 SE 보다 작다** — 최악 계열
선택이 동전 던지기라는 뜻이다. 그룹 4개면 SE 가 0.73 dB 로 내려가고 cluster
bootstrap 의 클러스터 수도 CI 를 정의할 수 있는 최소치를 넘는다. 그룹이 1개면
(실측: val machine, test environment, test machine) 오차 추정 자체가 불가능하다.
"""

# ---------------------------------------------------------------------------------
# 달성 가능 상한
# ---------------------------------------------------------------------------------
def achievable_cancellation_ceiling_db(
    gamma_secondary: float, gamma_primary: float | None = None
) -> float:
    """반복 일관성 γ 가 허용하는 **플랜트 불확실성 상한**(dB).

    유도
    ----
    γ 는 반복 전달함수 쌍의 정규화 복소 내적 평균이다. 반복별 추정을
    ``H_i = H + N_i`` (N 은 평균 0, 반복 간 독립) 로 두고 상대 오차 전력을
    ``ρ = E‖N‖²/‖H‖²`` 라 하면::

        ⟨H_i, H_j⟩ ≈ ‖H‖²              (교차항 소거)
        ‖H_i‖·‖H_j‖ ≈ ‖H‖²(1 + ρ)
        ∴ γ ≈ 1/(1+ρ)   →   ρ = (1 − γ)/γ

    모델은 K 반복의 평균으로 학습하지만 **실행 시 마주치는 플랜트는 하나의 실현**이므로
    유효 불일치는 ρ 로 본다(보수적). 최적 제어기는 ``W = −P/Ŝ`` 이고 실제 플랜트가
    ``S = Ŝ(1+δ)`` 이면 잔차는 ``e = P·x·δ`` 이므로 ``|δ|² = ε_S²`` 다. P 도 실측이면
    오차가 독립이므로 ``ε² = ε_S² + ε_P²``::

        상한(dB) = −10·log10( (1−γ_S)/γ_S + (1−γ_P)/γ_P )

    ⚠ 이 값을 "달성 가능한 상쇄량"으로 읽으면 안 된다
    ------------------------------------------------
    이것은 **플랜트를 몰라서 잃는 몫**의 상한일 뿐이다. 실제 달성치는 인과성(lead),
    FIR 길이, 신호 예측 가능성에 의해 훨씬 낮게 묶인다. 복구된 플랜트 실측이 그 증거다::

        γ 기반 상한 (γ_S=0.9990, γ_P=0.9993)   ≈ 28 dB
        주파수영역 정규방정식 직접 계산(M=2048, lead=116, 150-600Hz)  =  6.53 dB

    즉 γ 상한은 4배 이상 낙관적이다. 그래서 이 프로젝트의 게이트는 이 값을 단독으로
    쓰지 않고 ``readiness.measured_design_ceiling_db`` (정규방정식으로 직접 계산한
    설계 상한)와 **작은 쪽**을 취한다. 낙관적인 상한 하나만 믿는 것이 이 저장소에서
    반복된 사고의 형태다.

    또 하나의 한계: γ→ε 사상은 "반복 간 오차가 독립·평균 0" 가정 위에 있다. 프레임
    슬립처럼 계통적 오염이면 γ 가 낮아져 상한이 보수적으로(작게) 나오므로 안전하지만,
    오염이 **모든 반복에 동일하게** 실리면 γ 는 높게 나오고 상한이 낙관적이 된다.
    그 경우는 γ 로 못 잡으므로 P−S 상대 τ 검사와 독립 캡처 재측정이 따로 필요하다.
    """

    for label, value in (("gamma_secondary", gamma_secondary), ("gamma_primary", gamma_primary)):
        if value is None:
            continue
        if not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{label} 는 (0, 1] 안의 유한값이어야 합니다: {value!r}")
    eps2 = (1.0 - float(gamma_secondary)) / float(gamma_secondary)
    if gamma_primary is not None:
        eps2 += (1.0 - float(gamma_primary)) / float(gamma_primary)
    if eps2 <= 0.0:
        # γ 가 정확히 1 이면 상한이 무한대다. 측정으로 그런 값이 나올 수 없으므로
        # 무한대를 돌려주는 대신 "측정 불가"로 다룬다.
        return float("inf")
    return -10.0 * math.log10(eps2)


def required_consistency_for(target_db: float) -> float:
    """목표 상쇄량을 내려면 **경로당** 최소 얼마의 γ 가 필요한가 (단일 경로 기준).

    ``achievable_cancellation_ceiling_db`` 의 단일 경로 역함수다::

        target = −10·log10((1−γ)/γ)  →  γ = 1 / (1 + 10^(−target/10))

    임계를 이 함수로 적어 두면 되돌리기가 눈에 띈다. ``min_path_consistency: 0.9``
    같은 맨숫자는 근거가 없어 보여 조정 압력을 받지만, ``required_consistency_for(12.0)``
    은 "목표를 12 dB 에서 내리겠다"는 선언이 되어 숨길 수 없다.
    """

    if not math.isfinite(float(target_db)):
        raise ValueError(f"target_db 는 유한값이어야 합니다: {target_db!r}")
    return 1.0 / (1.0 + 10.0 ** (-float(target_db) / 10.0))


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: str | Path, *, block_bytes: int = 1024 * 1024) -> str:
    """큰 artifact도 메모리에 올리지 않고 SHA-256을 계산한다."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _npz_scalar(data: Any, key: str) -> Any:
    if key not in data:
        raise ValueError(f"필수 메타데이터 누락: {key}")
    value = data[key]
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{key}는 scalar여야 합니다: shape={array.shape}")
    return array.reshape(-1)[0].item()


def audit_official_path_model(
    path: str | Path,
    *,
    expected_output_channel: str,
    sample_rate: int,
    required_band_hz: tuple[float, float] = DEFAULT_REQUIRED_PATH_BAND_HZ,
    min_consistency: float = 0.9,
) -> dict[str, Any]:
    """``calibrate_wideband.py``의 official P/S artifact를 엄격히 검사한다."""

    model_path = _repo_path(path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"실측 경로 모델이 없습니다: {model_path}")
    if expected_output_channel not in {"noise", "cancel"}:
        raise ValueError(f"잘못된 expected output channel: {expected_output_channel}")
    band_lo, band_hi = map(float, required_band_hz)
    if not 0.0 < band_lo < band_hi < float(sample_rate) / 2.0:
        raise ValueError(f"잘못된 필수 경로 대역: {required_band_hz}")

    with np.load(model_path, allow_pickle=False) as data:
        required = {
            "fir",
            "delay_samples",
            "sample_rate",
            "coherence_median",
            "excitation_band_hz",
            "calibration_block_size",
            "calibration_latency",
            "output_channel",
            "method",
            "repeats",
            "amplitude",
            "xrun_count",
            "delay_spread_samples",
            "max_delay_jitter_samples",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(
                f"official ESS 품질 메타데이터가 없습니다: {', '.join(missing)}"
            )
        fir = np.asarray(data["fir"], dtype=np.float32).reshape(-1)
        delay = int(_npz_scalar(data, "delay_samples"))
        artifact_rate = int(_npz_scalar(data, "sample_rate"))
        consistency = float(_npz_scalar(data, "coherence_median"))
        excitation_band = np.asarray(
            data["excitation_band_hz"], dtype=np.float64
        ).reshape(-1)
        output_channel = str(_npz_scalar(data, "output_channel"))
        method = str(_npz_scalar(data, "method"))
        repeats = int(_npz_scalar(data, "repeats"))
        amplitude = float(_npz_scalar(data, "amplitude"))
        xrun_count = int(_npz_scalar(data, "xrun_count"))
        block_size = int(_npz_scalar(data, "calibration_block_size"))
        latency = str(_npz_scalar(data, "calibration_latency"))
        delay_spread = int(_npz_scalar(data, "delay_spread_samples"))
        max_delay_jitter = int(_npz_scalar(data, "max_delay_jitter_samples"))

        interleaved: dict[str, Any] = {}
        if method == "interleaved_multitone":
            missing_il = sorted(
                set(INTERLEAVED_REQUIRED_FIELDS).difference(data.files)
            )
            if missing_il:
                raise ValueError(
                    "interleaved 측정 메타데이터가 없습니다: " + ", ".join(missing_il)
                )
            consistency_band = np.asarray(
                data["consistency_band_hz"], dtype=np.float64
            ).reshape(-1)
            if "band_consistency" in data.files and "band_consistency_hz" in data.files:
                band_values = np.asarray(
                    data["band_consistency"], dtype=np.float64
                ).reshape(-1)
                band_edges = np.asarray(
                    data["band_consistency_hz"], dtype=np.float64
                ).reshape(-1, 2)
            else:
                band_values = None
                band_edges = None
            reanalysis_params = (
                json.loads(str(_npz_scalar(data, "reanalysis_params_json")))
                if "reanalysis_params_json" in data.files
                else None
            )
            interleaved = {
                "consistency_band_hz": [float(v) for v in consistency_band[:2]],
                "capture_id": str(_npz_scalar(data, "capture_id")),
                "guard_bins": int(_npz_scalar(data, "interleave_guard_bins")),
                "analysis_period_seconds": float(
                    _npz_scalar(data, "analysis_period_seconds")
                ),
                "tone_count": int(_npz_scalar(data, "tone_count")),
                "tone_snr_median_db": float(_npz_scalar(data, "tone_snr_median_db")),
                "tone_snr_min_db": float(_npz_scalar(data, "tone_snr_min_db")),
            }

    errors: list[str] = []
    if fir.size < 1 or not np.all(np.isfinite(fir)) or np.max(np.abs(fir)) <= 0.0:
        errors.append("FIR이 비었거나 NaN/Inf/영값입니다")
    if delay < 0:
        errors.append("delay_samples가 음수입니다")
    if artifact_rate != int(sample_rate):
        errors.append(f"sample_rate {artifact_rate} != {sample_rate}")
    if output_channel != expected_output_channel:
        errors.append(
            f"output_channel={output_channel!r}; expected={expected_output_channel!r}"
        )
    if method not in ALLOWED_PATH_METHODS:
        errors.append(
            f"method={method!r}; 허용 method={ALLOWED_PATH_METHODS}"
        )
    elif method == "interleaved_multitone":
        if interleaved["guard_bins"] != 1:
            errors.append(
                f"interleave_guard_bins={interleaved['guard_bins']}; 1이어야 합니다"
            )
        period = interleaved["analysis_period_seconds"]
        if not math.isfinite(period) or not 0.0 < period <= INTERLEAVED_MAX_PERIOD_SECONDS:
            errors.append(
                f"analysis_period_seconds={period!r}; "
                f"(0, {INTERLEAVED_MAX_PERIOD_SECONDS}] 이어야 합니다"
            )
        if interleaved["tone_count"] < INTERLEAVED_MIN_TONE_COUNT:
            errors.append(
                f"tone_count={interleaved['tone_count']} < {INTERLEAVED_MIN_TONE_COUNT}"
            )
        snr = interleaved["tone_snr_median_db"]
        if not math.isfinite(snr) or snr < INTERLEAVED_MIN_TONE_SNR_MEDIAN_DB:
            errors.append(
                f"tone_snr_median_db={snr!r} < {INTERLEAVED_MIN_TONE_SNR_MEDIAN_DB}"
            )
        if not interleaved["capture_id"]:
            errors.append("capture_id가 비었습니다")
        measured_consistency_band = interleaved["consistency_band_hz"]
        if len(measured_consistency_band) < 2 or not all(
            math.isfinite(v) for v in measured_consistency_band
        ):
            errors.append("consistency_band_hz가 유효하지 않습니다")
        elif (
            measured_consistency_band[0] > band_lo
            or measured_consistency_band[1] < band_hi
        ):
            # 좁은 대역에서 잰 일관성으로 넓은 대역을 주장할 수 없다.
            errors.append(
                f"일관성 측정 대역 {tuple(measured_consistency_band)} 이 "
                f"필수 대역 {required_band_hz} 를 덮지 못합니다"
            )

        # 최악 부대역 게이트 — 총계는 에너지 가중이라 약한 대역을 숨긴다.
        if band_values is None or band_edges is None:
            errors.append(
                "band_consistency/band_consistency_hz 가 없습니다 — "
                "최악 부대역을 검증할 수 없는 아티팩트는 official 이 될 수 없습니다"
            )
        elif band_values.size != band_edges.shape[0]:
            errors.append(
                f"band_consistency 길이 {band_values.size} != "
                f"band_consistency_hz {band_edges.shape[0]}"
            )
        else:
            judged = 0
            for (lo, hi), value in zip(band_edges, band_values):
                if lo < band_lo or hi > band_hi:
                    continue     # 필수 대역 밖은 판정하지 않는다
                judged += 1
                if not math.isfinite(float(value)) or float(value) < MIN_BAND_CONSISTENCY:
                    errors.append(
                        f"부대역 {lo:.0f}-{hi:.0f}Hz 일관성 {float(value):.4f} "
                        f"< {MIN_BAND_CONSISTENCY}"
                    )
            if judged == 0:
                errors.append(
                    f"필수 대역 {required_band_hz} 안에 판정 가능한 부대역이 없습니다"
                )
            interleaved["band_consistency"] = [float(v) for v in band_values]
            interleaved["band_consistency_hz"] = [
                [float(lo), float(hi)] for lo, hi in band_edges
            ]

        # 재분석 아티팩트면 파라미터 봉투를 검사한다. 게이트를 약화한 값으로 다시 푼
        # 결과가 official 이 되면 게이트 전체가 무의미해진다.
        if reanalysis_params is not None:
            for key, (lo_ok, hi_ok) in ALLOWED_REANALYSIS_ENVELOPE.items():
                value = reanalysis_params.get(key)
                if value is None:
                    errors.append(f"재분석 파라미터 {key} 가 없습니다")
                elif not math.isfinite(float(value)):
                    errors.append(f"재분석 {key}={value!r} 가 유한하지 않습니다")
                elif lo_ok is not None and float(value) < lo_ok:
                    errors.append(f"재분석 {key}={value} < {lo_ok}")
                elif hi_ok is not None and float(value) > hi_ok:
                    errors.append(f"재분석 {key}={value} > {hi_ok}")
            interleaved["reanalysis_params"] = reanalysis_params
    if repeats < MIN_KEPT_REPEATS:
        errors.append(
            f"유지된 반복 {repeats}회 < {MIN_KEPT_REPEATS}회 — 평균화가 부족해 "
            "한 번의 이상치가 플랜트 형상을 지배합니다"
        )
    if not math.isfinite(consistency) or consistency < float(min_consistency):
        errors.append(
            f"반복 일관성 {consistency!r} < {float(min_consistency):.3f}"
        )
    if excitation_band.size < 2 or not np.all(np.isfinite(excitation_band[:2])):
        errors.append("excitation_band_hz가 유효하지 않습니다")
        measured_band = (float("nan"), float("nan"))
    else:
        measured_band = (float(excitation_band[0]), float(excitation_band[1]))
        if measured_band[0] > band_lo or measured_band[1] < band_hi:
            errors.append(
                f"측정 대역 {measured_band}가 필수 {required_band_hz}를 덮지 못합니다"
            )
    if not math.isfinite(amplitude) or not 0.0 < amplitude <= 0.02:
        errors.append(f"측정 amplitude가 안전 official 범위 밖입니다: {amplitude!r}")
    if xrun_count != 0:
        errors.append(f"xrun_count={xrun_count}; 0이어야 합니다")
    if block_size <= 0:
        errors.append(f"calibration_block_size={block_size}")
    if latency not in {"low", "high"}:
        errors.append(f"calibration_latency={latency!r}")
    # 허용치를 **아티팩트에서 읽지 않는다**. 측정 스크립트가 자기 허용치를 함께 쓰므로
    # 그것을 믿으면 게이트가 자기증명이 된다(실측: 32 샘플 슬립이 허용 48 을 통과).
    if delay_spread < 0 or max_delay_jitter < 0:
        errors.append(
            f"지연 spread 메타데이터가 음수입니다: "
            f"delay_spread={delay_spread}, max_delay_jitter={max_delay_jitter}"
        )
    elif delay_spread > MAX_RELATIVE_DELAY_SPREAD_SAMPLES:
        errors.append(
            f"P−S 상대 τ spread {delay_spread} > 허용 "
            f"{MAX_RELATIVE_DELAY_SPREAD_SAMPLES} samples "
            f"(아티팩트가 신고한 {max_delay_jitter} 는 참고값일 뿐이다)"
        )
    if errors:
        raise ValueError(f"{model_path}: " + "; ".join(errors))

    return {
        "path": str(model_path),
        "sha256": sha256_file(model_path),
        "method": method,
        "interleaved": interleaved or None,
        "output_channel": output_channel,
        "sample_rate": artifact_rate,
        "delay_samples": delay,
        "fir_length": int(fir.size),
        "consistency": consistency,
        "excitation_band_hz": list(measured_band),
        "amplitude": amplitude,
        "calibration_block_size": block_size,
        "calibration_latency": latency,
        "repeats": repeats,
        "xrun_count": xrun_count,
        "delay_spread_samples": delay_spread,
        "max_delay_jitter_samples": max_delay_jitter,
    }


def _checkpoint_lead(state: dict) -> int:
    cfg = state.get("cfg", {}) or {}
    if "digital_reference_lead_samples" in cfg:
        return int(cfg["digital_reference_lead_samples"])
    return int((cfg.get("data", {}) or {}).get("digital_reference_lead_samples", 0))


def _checkpoint_optimize_band(state: dict) -> FrequencyBand | None:
    """checkpoint 가 **어느 대역에서 개선을 요구받았는지**. 없으면 None.

    trainer 는 ``BandPlan.resolve(...).optimize`` 를 resolved cfg 의
    ``trusted_band_hz`` 로 저장한다. 그것이 이 값의 단일 출처다 — 게이트가 여기서
    다시 유도하지 않는다.
    """

    cfg = state.get("cfg", {}) or {}
    raw = cfg.get("trusted_band_hz")
    if raw is None:
        return None
    return FrequencyBand.parse(raw, name="checkpoint trusted")


def _load_checkpoint_state(path: Path) -> dict:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f"checkpoint를 읽을 수 없습니다: {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint 최상위가 dict가 아닙니다: {path}")
    if not isinstance(state.get("model"), dict) or not state["model"]:
        raise ValueError(f"checkpoint model state가 비었습니다: {path}")
    if not all(isinstance(value, torch.Tensor) for value in state["model"].values()):
        raise ValueError(f"checkpoint model state에 tensor가 아닌 값이 있습니다: {path}")
    if not isinstance(state.get("cfg"), dict):
        raise ValueError(f"checkpoint resolved cfg가 없습니다: {path}")
    return state


def _checkpoint_identity(cfg: dict) -> dict:
    """resume 경로처럼 실행 중 바뀔 수 있는 값은 빼고 run 정체성을 만든다."""

    keys = (
        "stage",
        "model",
        "data",
        "duct",
        "optimizer",
        "schedule",
        "loss",
        "seed",
        "batch_size",
        "recorded_manifest",
        "recorded_ratio",
        "init_ckpt",
        "physics_status",
        "digital_reference_lead_samples",
    )
    return {key: cfg.get(key) for key in keys if key in cfg}


def _model_state_signature(state: dict) -> dict[str, tuple[tuple[int, ...], str]]:
    return {
        str(name): (tuple(int(value) for value in tensor.shape), str(tensor.dtype))
        for name, tensor in state["model"].items()
        if isinstance(tensor, torch.Tensor)
    }


def audit_init_checkpoint(
    path: str | Path,
    *,
    expected_model_cfg: dict,
    expected_lead: int,
    expected_optimize_band: FrequencyBand | None = None,
    max_lead_mismatch_samples: int = 0,
    require_completed: bool = True,
    max_best_metric_db: float = 0.0,
    allowed_physics_statuses: tuple[str, ...] = (
        "secondary_surrogate_representation_pretrain",
    ),
) -> dict[str, Any]:
    """사전학습 best와 같은 run의 완료된 last를 함께 검증한다.

    ``max_lead_mismatch_samples`` 는 **init checkpoint 에만** 적용되는 허용 오차다.
    기본 0(정확히 일치)이며, 늘리려면 설정에 명시적으로 적어야 한다.

    왜 허용 오차가 필요한가. init checkpoint 는 정의상 surrogate 물리로 학습된 것이고
    (physics_status=secondary_surrogate_representation_pretrain), 그때 쓴 lead 는
    잠정값이다. 실측이 끝나면 lead 가 몇 샘플 달라지는 것이 정상이며, 그 차이를 흡수하는
    것이 파인튜닝의 목적이다. 실제로 같은 양을 독립적으로 잰 값이 109/113/116/119 로
    폭 10샘플이었다 — 측정 불확도 자체가 이 정도다.

    이 허용이 게이트를 무르게 만들지 않는 이유: **정확성을 지키는 게이트는 따로 있다.**
    ``path_delay_and_lead`` 가 "fine-tune 설정의 lead == 실측 S+handoff−P" 를 정확히
    요구하고, 여기서 벗어나면 그쪽에서 걸린다. 이 허용은 오직 "어떤 checkpoint 에서
    출발할 수 있는가"에만 관여한다. 과거 사고였던 lead=0 checkpoint 는 113 과 113 샘플
    떨어져 있으므로 어떤 합리적 허용치로도 통과하지 못한다.

    ``expected_optimize_band`` — **대역 축** (2026-08-06 신설, 반증 #13/#17 대응)
    ----------------------------------------------------------------------
    lead 와 달리 대역 차이는 파인튜닝이 흡수하는 종류가 아니다. 좁은 대역으로 학습한
    모델은 벌점이 **없던** 구간을 적극적으로 증폭한다 — 실측: ``[150,800]`` 설정으로
    학습한 모델이 600–1600Hz 를 **+27.01 dB** 키웠다. 그런 checkpoint 에서 출발하면
    파인튜닝은 "고역 증폭기" 를 초기값으로 받는다.

    실제 상태(2026-08-06): ``runs/pretrain_{base,tiny}_corrected/ckpt/best.pt`` 는
    ``cfg.trusted_band_hz = [150, 600]`` / lead 109 인데 현재 설정이 유도하는 값은
    ``[150, 1600]`` / 116 이다. lead 축만 보던 게이트는 이 checkpoint 를 통과시켰다.

    그래서 요구는 하나다: **checkpoint 의 최적화 대역이 파인튜닝의 최적화 대역을
    덮어야 한다.** 넓은 쪽에서 좁은 쪽으로 가는 것은 허용된다(벌점을 받아 본 구간이
    더 넓다). 좁은 쪽에서 넓은 쪽으로 가는 것은 거부한다. 허용치 설정은 두지 않았다 —
    "몇 Hz 까지 봐준다" 가 존재하는 순간 게이트를 통과시키려고 그 값을 키우게 된다.
    """

    checkpoint = _repo_path(path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"init checkpoint가 없습니다: {checkpoint}")
    state = _load_checkpoint_state(checkpoint)
    saved_cfg = state["cfg"]
    if saved_cfg.get("model") != expected_model_cfg:
        raise ValueError("init checkpoint 모델 설정이 fine-tune 모델 설정과 다릅니다")
    physics_status = str(saved_cfg.get("physics_status", ""))
    if physics_status not in set(allowed_physics_statuses):
        raise ValueError(
            "init checkpoint physics_status가 승인된 corrected pretrain이 아닙니다: "
            f"{physics_status!r}; allowed={list(allowed_physics_statuses)}"
        )
    tolerance = int(max_lead_mismatch_samples)
    if tolerance < 0:
        raise ValueError("max_lead_mismatch_samples는 음수일 수 없습니다")
    saved_lead = _checkpoint_lead(state)
    lead_mismatch = abs(saved_lead - int(expected_lead))
    if lead_mismatch > tolerance:
        raise ValueError(
            "init checkpoint digital-reference lead 불일치: "
            f"checkpoint={saved_lead}, fine-tune={int(expected_lead)}, "
            f"차이 {lead_mismatch} > 허용 {tolerance} samples"
        )
    saved_band = _checkpoint_optimize_band(state)
    if expected_optimize_band is not None:
        if saved_band is None:
            raise ValueError(
                "init checkpoint 에 trusted_band_hz 가 없어 **어느 대역에서 학습됐는지** "
                "알 수 없습니다 — 좁은 대역으로 학습된 모델은 벌점이 없던 대역을 "
                "증폭합니다(실측 +27.01 dB). 대역을 기록하는 trainer 로 다시 학습하세요"
            )
        if not saved_band.covers(expected_optimize_band):
            raise ValueError(
                "init checkpoint 학습 대역이 파인튜닝 대역을 덮지 않습니다: "
                f"checkpoint={saved_band.as_tuple()}, fine-tune="
                f"{expected_optimize_band.as_tuple()} — 벌점을 받아 본 적 없는 구간이 "
                "남습니다. 좁은 대역으로 학습한 모델은 그 구간을 적극 증폭합니다 "
                "([150,800] 로 학습한 모델의 600-1600Hz 실측 +27.01 dB). "
                "lead 와 달리 이 차이는 파인튜닝이 흡수하지 않습니다"
            )

    best_metric = float(state.get("best_metric", float("nan")))
    if not math.isfinite(best_metric) or best_metric >= float(max_best_metric_db):
        raise ValueError(
            f"init checkpoint best_metric={best_metric!r}; "
            f"{float(max_best_metric_db):.2f}dB 미만이어야 합니다"
        )

    completion_path = checkpoint.parent / "last.pt"
    completion_step: int | None = None
    completion_target: int | None = None
    if require_completed:
        if not completion_path.is_file():
            raise FileNotFoundError(
                "사전학습 완료를 증명할 같은 ckpt/last.pt가 없습니다: "
                f"{completion_path}"
            )
        last_state = _load_checkpoint_state(completion_path)
        last_cfg = last_state["cfg"]
        if _checkpoint_identity(last_cfg) != _checkpoint_identity(saved_cfg):
            raise ValueError("best.pt와 last.pt의 immutable run 설정이 다릅니다")
        if _model_state_signature(last_state) != _model_state_signature(state):
            raise ValueError("best.pt와 last.pt의 model state 구조가 다릅니다")
        if abs(_checkpoint_lead(last_state) - int(expected_lead)) > tolerance:
            raise ValueError("last.pt의 digital-reference lead가 fine-tune 설정과 다릅니다")
        if _checkpoint_lead(last_state) != saved_lead:
            raise ValueError("best.pt와 last.pt의 lead가 서로 다릅니다")
        last_band = _checkpoint_optimize_band(last_state)
        if (last_band is None) != (saved_band is None) or (
            last_band is not None
            and saved_band is not None
            and last_band.as_tuple() != saved_band.as_tuple()
        ):
            raise ValueError(
                "best.pt와 last.pt의 학습 대역이 서로 다릅니다: "
                f"best={None if saved_band is None else saved_band.as_tuple()}, "
                f"last={None if last_band is None else last_band.as_tuple()}"
            )
        schedule = last_cfg.get("schedule", {}) or {}
        completion_target = int(
            last_cfg.get("run_until_step", schedule.get("total_steps", 0))
        )
        completion_step = int(last_state.get("step", -1))
        if completion_target <= 0 or completion_step < completion_target:
            raise ValueError(
                "사전학습이 완료되지 않았습니다: "
                f"last step={completion_step}, target={completion_target}"
            )

    return {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "step": int(state.get("step", -1)),
        "best_metric_db": best_metric,
        "physics_status": physics_status,
        "digital_reference_lead_samples": saved_lead,
        "trusted_band_hz": None if saved_band is None else list(saved_band.as_tuple()),
        "expected_trusted_band_hz": (
            None
            if expected_optimize_band is None
            else list(expected_optimize_band.as_tuple())
        ),
        "completion_checkpoint": str(completion_path) if require_completed else None,
        "completion_step": completion_step,
        "completion_target_step": completion_target,
    }


class _Audit:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.checks: list[dict[str, Any]] = []

    def pass_(self, check_id: str, message: str, **details: Any) -> None:
        self.checks.append(
            {"id": check_id, "ok": True, "message": message, "details": details}
        )

    def fail(self, check_id: str, message: str, **details: Any) -> None:
        self.checks.append(
            {"id": check_id, "ok": False, "message": message, "details": details}
        )

    def report(self, **extra: Any) -> dict[str, Any]:
        ok = bool(self.checks) and all(bool(item["ok"]) for item in self.checks)
        return {
            "schema_version": 1,
            "kind": self.kind,
            "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "ok": ok,
            "status": "PASS" if ok else "FAIL",
            "checks": self.checks,
            **extra,
        }


def _required_families(readiness_cfg: dict) -> tuple[str, ...]:
    values = readiness_cfg.get(
        "required_source_families", DEFAULT_REQUIRED_SOURCE_FAMILIES
    )
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        raise ValueError("readiness.required_source_families는 문자열 목록이어야 합니다")
    families = tuple(str(value) for value in values)
    if not families or any(not value for value in families):
        raise ValueError("required_source_families가 비었거나 빈 값이 있습니다")
    return families


def _relative_tau_check(primary: dict, secondary: dict):
    """두 아티팩트에 저장된 ``repeat_tau_samples`` 로 P−S 상대 τ 상수성을 본다.

    두 채널은 같은 DAC·같은 출력 스트림의 인터리브이므로 τ_P − τ_S 는 설계 원리상
    상수다. 튀면 출력 버퍼가 한쪽 채널에서만 미끄러진 것이다.

    검사는 :func:`deep_anc.dsp.invariants.check_relative_tau_constancy` 가 한다 —
    측정·재분석·게이트가 **같은 코드**로 같은 판정을 내려야 한다. 궤적을 저장하지
    않는 옛 아티팩트(ESS 등)는 ``None`` 을 돌려주고, 그 경우 기존 스칼라 검사가
    유일한 방어선으로 남는다.
    """

    from ..dsp.invariants import check_relative_tau_constancy

    try:
        with np.load(_repo_path(primary["path"]), allow_pickle=False) as p_data:
            if "repeat_tau_samples" not in p_data.files:
                return None
            tau_primary = np.asarray(p_data["repeat_tau_samples"], dtype=np.float64)
        with np.load(_repo_path(secondary["path"]), allow_pickle=False) as s_data:
            if "repeat_tau_samples" not in s_data.files:
                return None
            tau_secondary = np.asarray(s_data["repeat_tau_samples"], dtype=np.float64)
    except (FileNotFoundError, OSError, KeyError, ValueError):
        return None
    if tau_primary.shape != tau_secondary.shape or tau_primary.size < 2:
        raise ValueError(
            f"P/S repeat_tau_samples 길이가 다릅니다: {tau_primary.shape} != "
            f"{tau_secondary.shape} — 같은 캡처의 두 채널이 아닙니다"
        )
    return check_relative_tau_constancy(
        tau_primary,
        tau_secondary,
        tolerance_samples=float(MAX_RELATIVE_DELAY_SPREAD_SAMPLES),
    )


def _alignment_cfg_keys() -> frozenset[str]:
    """QA 가 받는 정렬 키를 **QA 에서 유도한다.** 손으로 베끼지 않는다.

    ⚠ 2026-08-06 통합 검증이 잡은 결함: 이 목록이 옛 4개 키
    (``max_source_err_delay_std_samples`` 등)로 하드코딩돼 있어서, readiness 에
    새 키(``max_source_err_delay_robust_std_samples`` 등)를 선언하면 **경고 한 줄
    없이 버려지고** QA 는 기본값으로 돌았다. ``deprecated_threshold_notes`` 도 빈
    튜플이 되어 "폐기 키를 쓰고 있다"는 안내문마저 사라진다 — 즉 HANDOFF 가 지시한
    "키 이름을 새 것으로 갈아라"를 그대로 따르면 설정이 조용히 무력화된다.

    목록을 복사하는 것이 원인이므로 복사를 없앤다. 폐기 키까지 함께 넘기는 이유는
    :func:`settings_from_data_config` 가 그것을 받아 "무엇이 무시됐는지" 를
    ``deprecated_threshold_notes`` 로 되돌려 주기 때문이다. 알 수 없는 키는 QA 가
    ``ValueError`` 로 거절한다.
    """

    from ..data.recorded_qa import (
        _ALIGNMENT_OVERRIDE_KEYS,
        _DEPRECATED_ALIGNMENT_DROPPED,
        _DEPRECATED_ALIGNMENT_KEYS,
    )

    return frozenset(
        set(_ALIGNMENT_OVERRIDE_KEYS)
        | set(_DEPRECATED_ALIGNMENT_KEYS)
        | set(_DEPRECATED_ALIGNMENT_DROPPED)
    )


def _alignment_overrides(readiness_cfg: dict) -> dict:
    """readiness 가 선언한 정렬 임계를 QA 로 넘긴다.

    QA 와 게이트가 **같은 임계**를 써야 한다. 두 곳이 각자 기본값을 들고 있으면
    "QA 는 통과했는데 게이트는 실패" 같은 해석 불가능한 상태가 만들어지고, 그 상태를
    푸는 가장 쉬운 방법은 언제나 게이트를 낮추는 것이다.
    """

    overrides: dict = {}
    for key in _alignment_cfg_keys():
        value = readiness_cfg.get(key)
        if value is None:
            continue
        # 값의 형(bool/int/tuple)은 QA 의 RecordedQASettings 가 검증한다. 여기서
        # float() 로 뭉개면 alignment_band_hz 같은 비스칼라 키가 깨진다.
        overrides[key] = tuple(value) if isinstance(value, list) else value
    return overrides


def _audit_recorded_alignment(
    audit: "_Audit",
    readiness_cfg: dict,
    recorded_report: dict | None,
    full_recorded_qa: bool,
) -> None:
    """G2b — 학습 데이터에 **source→ERR 관계가 실제로 존재하는가**.

    결함 2. recorded QA 가 80/80 PASS 였던 이유는 무엇을 봤는가가 아니라 무엇을
    **안 봤는가**다. 이 게이트는 QA 가 새로 측정한 정렬 지표를 판정에 쓴다.

    QA 를 건너뛴 실행(``--skip-recorded-qa``)에서는 통과시키지 않는다. "측정하지
    않았다"와 "측정해서 통과했다"를 같게 취급하는 것이 이 저장소의 반복된 실패다.
    """

    if not full_recorded_qa:
        audit.fail(
            "recorded_alignment_integrity",
            "전수 QA 를 건너뛰면 source→ERR 정렬을 검증할 수 없습니다 — "
            "측정하지 않은 것을 통과로 세지 않습니다",
        )
        return
    if not recorded_report or not recorded_report.get("sessions"):
        audit.fail(
            "recorded_alignment_integrity",
            "recorded QA 리포트가 없어 source→ERR 정렬을 검증할 수 없습니다",
        )
        return

    threshold = float(readiness_cfg.get("min_source_err_coherence", 0.60))
    unmeasured: list[str] = []
    failing: list[tuple[str, float]] = []
    coherences: list[float] = []
    for session in recorded_report["sessions"]:
        alignment = session.get("alignment") or {}
        session_id = str(session.get("session_id", "?"))
        if "source_err_coherence" not in alignment:
            unmeasured.append(session_id)
            continue
        value = float(alignment["source_err_coherence"])
        coherences.append(value)
        if not alignment.get("ok", False) or value < threshold:
            failing.append((session_id, value))

    if unmeasured:
        audit.fail(
            "recorded_alignment_integrity",
            f"정렬을 측정하지 못한 세션 {len(unmeasured)}개 (예: {unmeasured[:5]}) — "
            "측정 불가는 통과가 아닙니다",
            unmeasured_sessions=unmeasured[:20],
        )
        return
    if failing:
        audit.fail(
            "recorded_alignment_integrity",
            f"source→ERR 결맞음/지연 안정성 미달 세션 {len(failing)}개 "
            f"(예: {[f'{sid} coh²={value:.3f}' for sid, value in failing[:5]]}) — "
            "후처리로 구제되지 않습니다. 재녹음이 필요합니다",
            failing_sessions=[sid for sid, _ in failing][:20],
            min_source_err_coherence=threshold,
        )
        return
    audit.pass_(
        "recorded_alignment_integrity",
        f"전 {len(coherences)}개 세션의 source→ERR 시간축이 유효합니다 "
        f"(최소 coh² {min(coherences):.3f} ≥ {threshold:.2f})",
        min_observed_coherence=min(coherences),
        sessions=len(coherences),
    )


def _groups_per_family_per_split(entries: Iterable[dict]) -> list[tuple[str, str, int]]:
    """``(split, source_family, 그룹 수)`` 를 센다. **세션 수가 아니라 그룹 수다.**"""

    buckets: dict[tuple[str, str], set[str]] = {}
    for entry in entries:
        key = (str(entry.get("split", "")), str(entry.get("source_family", "")))
        buckets.setdefault(key, set()).add(str(entry.get("group_id", "")))
    return sorted(
        (split, family, len(groups)) for (split, family), groups in buckets.items()
    )


def _audit_statistical_power(
    audit: "_Audit", readiness_cfg: dict, entries: list[dict]
) -> None:
    """G2c — val/test 의 계열당 그룹이 통계 판정을 지탱하는가 (결함 4 / D3).

    G4 는 계열별 평균으로 최악 계열을 고르는데, 그 평균의 불확도는 세그먼트 수가
    아니라 **그룹 수**가 정한다. 실측: 계열 간 폭 0.92 dB < 그룹 SE 1.03 dB 였다.
    표본을 늘리지 않고는 어떤 통계 처리로도 구제되지 않는 문제이므로, 학습을 시작하기
    전에 데이터 수집 요구사항으로 못 박는다.
    """

    if not entries:
        audit.fail("recorded_statistical_power", "manifest 항목이 없어 검정력을 판정할 수 없습니다")
        return
    minimum = int(
        readiness_cfg.get(
            "min_groups_per_family_per_split", _min_groups_per_family_default()
        )
    )
    weak = [
        (split, family, count)
        for split, family, count in _groups_per_family_per_split(entries)
        if split in ("val", "test") and count < minimum
    ]
    if weak:
        audit.fail(
            "recorded_statistical_power",
            f"val/test 의 계열당 그룹이 부족합니다 (최소 {minimum}): "
            + ", ".join(f"{split}/{family}={count}" for split, family, count in weak)
            + " — 그룹이 1–2개면 cluster bootstrap 의 클러스터 수가 CI 를 정의하지 "
            "못해 G4 판정 자체가 성립하지 않습니다. 같은 그룹 안의 세션을 늘려도 "
            "클러스터 수는 늘지 않습니다",
            weak=[list(item) for item in weak],
            min_groups_per_family_per_split=minimum,
        )
        return
    audit.pass_(
        "recorded_statistical_power",
        f"val/test 의 모든 계열이 그룹 {minimum}개 이상을 갖습니다",
        min_groups_per_family_per_split=minimum,
    )


def _recorded_source_clips(csv_path: Path) -> dict[str, list[str]]:
    """``sources.csv`` 의 ``clips`` 열에서 계열별 원본 클립 목록을 읽는다."""

    import csv

    families: dict[str, list[str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            family = str(row.get("source_family", "")).strip()
            if not family:
                continue
            raw = row.get("clips") or "[]"
            try:
                clips = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{csv_path}: clips 열을 읽을 수 없습니다: {exc}") from exc
            families.setdefault(family, []).extend(str(item) for item in clips)
    return families


def _recorded_session_root(readiness_cfg: dict, entries: list[dict] | None) -> Path | None:
    """실측 세션이 실제로 놓인 디렉터리. **매니페스트 항목에서 유도한다.**

    설정이 ``recorded_session_root`` 를 명시하면 그것이 우선한다. 그 외에는 이미 읽어 둔
    매니페스트 항목의 경로를 따라간다 — 학습이 읽는 것이 매니페스트이므로 그것이 사실이다.

    전역 기본값("data/recorded")을 두면 **tmp_path 픽스처가 개발자의 실제 녹음을 읽는다.**
    2026-08-06 에 실제로 그렇게 테스트 8개가 깨졌다. 알 수 없으면 ``None`` 을 돌려주고,
    호출자는 대조를 건너뛴다 — 없는 것을 지어내지 않는다.
    """

    declared = readiness_cfg.get("recorded_session_root")
    if declared:
        return _repo_path(declared)
    for entry in entries or []:
        raw = str(entry.get("path") or "")
        if not raw:
            continue
        candidate = Path(raw)
        if candidate.is_dir():
            return candidate.parent
    return None


def observed_source_pools(recorded_root: Path | None) -> dict[str, int]:
    """실측 세션이 **실제로 재생한** 소스풀을 세션에서 읽어 센다.

    왜 설정이 아니라 세션인가
    ------------------------
    2026-08-06 감사가 재현한 fail-open: ``readiness.recorded_source_pool_csv`` 는
    ``data/source_pool/sources.csv``(v1) 를 가리키고 있었는데, v1 은 machine 이 8 그룹뿐이라
    분할 하한(9)을 만족할 수 없어 **재녹음은 v2 로 해야 한다**. 그런데 이 키를 안 고치고
    v2 로 녹음하면 누수 게이트가 **v1 클립끼리 비교해 PASS 하면서 v2 누수를 100% 통과**시킨다.
    실제로 같은 검사가 v1 csv 로는 ok, v2 csv 로는 not ok 를 낸다.

    원인은 발생기 A 다 — "실측이 어떤 오디오를 썼는가" 라는 하나의 물리량을 설정과 세션이
    따로 들고 있고 아무도 대조하지 않았다. 세션의 ``program.file`` 이 물리적 사실이므로
    그것을 단일 출처로 삼고, 설정은 **대조 대상**으로만 쓴다.

    반환: ``{"data/source_pool_v2/sources.csv": 세션 수}``.
    """

    counts: dict[str, int] = {}
    if recorded_root is None or not recorded_root.is_dir():
        return counts
    for session_dir in sorted(recorded_root.iterdir()):
        meta = session_dir / "session.json"
        if not session_dir.is_dir() or not meta.is_file():
            continue
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        played = str((payload.get("program") or {}).get("file") or "")
        if not played:
            continue
        # data/source_pool_v2/environment/environment_000.wav → data/source_pool_v2
        parts = PurePosixPath(played).parts
        if len(parts) < 3:
            continue
        pool_csv = str(PurePosixPath(parts[0], parts[1]) / "sources.csv")
        counts[pool_csv] = counts.get(pool_csv, 0) + 1
    return counts


def _synthetic_clip_index(
    manifest_dir: Path, tags: Iterable[str]
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """합성 학습 스트림이 쓰는 원본 파일과 각 파일의 split 을 모은다."""

    by_tag: dict[str, list[str]] = {}
    splits: dict[str, str] = {}
    for tag in tags:
        path = manifest_dir / f"{tag}.jsonl"
        if not path.is_file():
            continue
        rows: list[str] = []
        for entry in read_manifest(path):
            value = str(entry.get("path", ""))
            if not value:
                continue
            rows.append(value)
            splits[value] = str(entry.get("split", ""))
        by_tag[str(tag)] = rows
    return by_tag, splits


def _audit_corpus_leak(
    audit: "_Audit", readiness_cfg: dict, data_cfg: dict, entries: list[dict] | None = None
) -> None:
    """D1 — 합성 학습 스트림과 실측이 **같은 원본 오디오**를 쓰지 않는가.

    2026-08-05 감사: 실측 music 60 트랙이 **100%** 합성 풀과 겹치고 그중 55개(92%)가
    합성 *train* split 에 있었다. 같은 곡에서 두 브랜치가 반대 방향 gradient 를 주고,
    실제로 **music 만 개선되지 않았다**(+0.09 dB vs 나머지 −0.85 ~ −2.05 dB).

    검사 자체는 :func:`deep_anc.dsp.invariants.check_corpus_disjoint` 가 한다. 여기서
    다시 구현하면 그것이 두 번째 정의가 된다.
    """

    from ..dsp.invariants import check_corpus_disjoint

    csv_value = readiness_cfg.get(
        "recorded_source_pool_csv", "data/source_pool_v2/sources.csv"
    )
    manifest_dir_value = data_cfg.get("noise_manifest_dir", "data/manifests")
    tags = sorted((data_cfg.get("source_mix_ratio") or {}).keys())
    try:
        # 실측 세션이 **실제로 재생한** 풀을 먼저 읽는다. 설정은 대조 대상이지
        # 단일 출처가 아니다 — v2 로 녹음하고 이 키를 v1 로 두면 게이트가 엉뚱한
        # 클립끼리 비교해 PASS 하면서 누수를 100% 통과시킨다 (2026-08-06 재현됨).
        # 세션 위치는 **매니페스트가 가리키는 곳**에서 유도한다. 전역 기본값
        # ("data/recorded")을 쓰면 tmp_path 픽스처가 개발자의 실제 녹음을 읽어 버린다 —
        # 2026-08-06 에 실제로 그렇게 테스트 8개가 깨졌다. 매니페스트가 학습이 읽는
        # 세션의 단일 출처이므로 그것을 따라간다.
        observed = observed_source_pools(
            _recorded_session_root(readiness_cfg, entries)
        )
        # 설정은 문자열 하나 또는 목록이다. 재녹음이 두 풀에 걸치는 것은 **정상**이다 —
        # 복구된 47세션(v1)과 신규 33세션(v2)을 합치면 계열별 그룹이 하한을 넘기고
        # 스피커 시간이 93.3분 → 38.5분으로 줄어든다. 위험한 것은 섞임 자체가 아니라
        # **세션이 재생한 풀을 게이트가 모르는 것**이다. 그래서 요구는 포함관계다:
        # 관측된 모든 풀이 선언 안에 있어야 한다. 선언에 여분이 있는 것은 무해하다.
        declared = (
            {str(csv_value)}
            if isinstance(csv_value, str)
            else {str(item) for item in csv_value}
        )
        unknown = sorted(set(observed) - declared)
        if unknown:
            raise ValueError(
                "실측 세션이 **설정에 없는 소스풀**을 재생했습니다 — 그 풀의 클립은 "
                "held-out 에서 빠지고, 누수 게이트가 못 본 채 통과합니다.\n"
                f"  설정 readiness.recorded_source_pool_csv = {csv_value}\n"
                f"  세션이 실제로 재생한 풀 = "
                + ", ".join(f"{k} ({v}세션)" for k, v in sorted(observed.items()))
                + f"\n  선언에 없는 풀: {unknown}\n"
                "  설정에 그 풀을 추가하세요 (목록으로 여러 개를 적을 수 있습니다)."
            )
        csv_paths = [_repo_path(value) for value in sorted(observed or declared)]
        missing_csv = [str(p) for p in csv_paths if not p.is_file()]
        if missing_csv:
            raise FileNotFoundError(
                f"실측 소스 목록이 없습니다: {', '.join(missing_csv)} — 겹침을 검사할 수 "
                "없으면 겹치지 않는다고 주장할 수 없습니다"
            )
        recorded: dict[str, list[str]] = {}
        for path in csv_paths:
            for family, clips in _recorded_source_clips(path).items():
                recorded.setdefault(family, []).extend(clips)
        if not recorded:
            raise ValueError(
                f"{', '.join(str(p) for p in csv_paths)}: 실측 클립 목록이 비었습니다"
            )
        synthetic, splits = _synthetic_clip_index(_repo_path(manifest_dir_value), tags)
        if not synthetic:
            raise FileNotFoundError(
                f"합성 소음 manifest 를 찾지 못했습니다 (dir={manifest_dir_value}, "
                f"tags={tags}) — 합성 풀을 알 수 없으면 누수 여부를 판정할 수 없습니다. "
                "scripts/data/prepare_noise_pool.py 로 manifest 를 만드세요"
            )
        # 태그 **하나라도** manifest 가 없으면 판정 불가다. 2026-08-06 통합 검증에서
        # 실제로 재현된 fail-open: data/manifests 에 esc50.jsonl 하나만 있는 상태에서
        # 이 게이트가 "실측 691개와 합성 1587개가 서로소" 로 PASS 했다. 그런데 D1 이
        # 실제로 찾은 누수는 **music 60/60(100%)** 이고 music.jsonl 이 없어 비교 대상에
        # 아예 들어가지 않았다. 즉 누수가 있는 태그를 못 보고 통과한 것이다.
        #
        # 게다가 없는 태그는 조용히 사라지지 않는다 — synth_dataset 은 manifest 없는
        # 태그를 **합성원으로 자동 폴백**한다(data/manifests/<tag>.jsonl 부재 시).
        # 위 상태로 학습하면 선언한 7종 혼합(esc50 5%)이 실제로는
        # synthetic 95% + esc50 5% 가 되는데 아무 로그도 남지 않는다.
        # 'synthetic' 은 파일 풀이 아니라 생성기이므로 제외한다.
        missing = [
            str(tag)
            for tag in tags
            if str(tag) != "synthetic" and str(tag) not in synthetic
        ]
        if missing:
            raise FileNotFoundError(
                f"source_mix_ratio 가 선언한 태그 중 manifest 가 없는 것이 있습니다: "
                f"{missing} (dir={manifest_dir_value}). 풀을 모르는 태그의 누수는 판정할 "
                "수 없고, 학습기는 그 태그를 조용히 합성원으로 대체해 선언한 혼합비와 "
                "다른 데이터로 돕니다. scripts/data/prepare_noise_pool.py 로 해당 태그의 "
                "manifest 를 만들거나, 쓰지 않는 태그라면 source_mix_ratio 에서 지우세요"
            )
        result = check_corpus_disjoint(recorded, synthetic, synthetic_splits=splits)
    except (FileNotFoundError, OSError, ValueError, KeyError, TypeError) as exc:
        audit.fail("corpus_disjoint", str(exc))
        return
    if result.ok:
        audit.pass_("corpus_disjoint", result.detail, **result.measured)
    else:
        audit.fail("corpus_disjoint", result.detail, **result.measured)


def _audit_measured_source_delay(
    audit: "_Audit",
    readiness_cfg: dict,
    primary: dict | None,
    recorded_report: dict | None,
    data_cfg: dict | None = None,
) -> None:
    """D2 — **두 학습 브랜치가 모델에게 같은 과제를 주는가.**

    합성 브랜치와 실측 브랜치가 x_ref 를 d 보다 **같은 만큼** 앞세워야 한다. 아니면
    같은 모델이 두 브랜치에서 서로 다른 예측 과제를 배운다.

    2026-08-07 재설계 — 옛 판정은 **절대** source→ERR 지연을 P(z) 유도값과 대조했는데,
    그 관측량은 재현되지 않는다(82세션 산포 189 샘플 vs 허용 64). 걱정은 옳았지만
    비교 대상이 틀렸다. 지금은 **총 선행량**을 본다:

        합성 = D_noise + K
        실측 = d_recorded + K'

    ``d_recorded``(정렬 잔여 지연)는 82세션에서 142.0~144.1, std 0.32 로 재현되고
    기하 예측 140 샘플과 2.5 샘플 안에서 일치한다.

    실측(2026-08-06): ``recorded_lead_mode=constant`` 이면 합성 1718 vs 실측 258 로
    **1460 샘플(30.4 ms)** 어긋났다. ``timeline`` 으로 바꾸면 차이가 0.1 샘플이 된다.
    """

    data_cfg = data_cfg or {}

    if primary is None:
        audit.fail(
            "measured_source_delay_agreement",
            "유효한 official P(z)가 없어 실측 지연과 대조할 수 없습니다",
        )
        return
    if not recorded_report or not recorded_report.get("sessions"):
        audit.fail(
            "measured_source_delay_agreement",
            "recorded QA 리포트가 없어 실측 source→ERR 지연을 알 수 없습니다",
        )
        return

    # ⚠ 2026-08-06 통합 검증 — **같은 이름이 두 물리량을 오간다.**
    # QA 는 "학습이 실제로 읽는 파일" 을 재므로 재정렬본(source_aligned.wav)이 있으면
    # 그것을 잰다. 그 값은 재정렬 후 **잔여 음향 지연 142.5 샘플**이지, 이 게이트가
    # P(z) 유도값(1602 + argmax tap = 1849)과 대조하려는 **원본 재생→ERR 지연**이
    # 아니다. 대조하면 1706.5 샘플 어긋나 무조건 FAIL 하고, 그 가짜 실패가 진짜 결함
    # (원본 관측 1672 vs 유도 1849 = 177 샘플, 허용 64 의 2.8배)을 덮는다.
    # 숫자를 비교하기 전에 **무엇을 잰 값인지** 먼저 본다.
    # ⚠ 2026-08-07 재설계 — **비교 대상이 재현되지 않는 양이었다.**
    #
    # 옛 게이트는 실측 세션의 **절대** source→ERR 지연을 P(z) 유도값(bulk + argmax)과
    # 대조했다. 그런데 그 관측량은 재현되지 않는다 — 82세션 실측 산포가
    # 1425.6~1614.2 (폭 189 샘플) 로 허용치 64 의 3배다. 설정 차이가 아니고
    # (둘 다 latency=low, block=256) 추정기 차이도 아니다 (같은 추정기로도 248 샘플).
    # HANDOFF 가 이미 적고 있다: "절대 지연은 재현되지 않는다."
    #
    # 게이트의 **걱정은 옳았다**: 합성 d 와 실측 d 가 모델에게 다른 과제를 주면 안 된다.
    # 그 걱정을 재현되는 양으로 다시 세운다 — **총 선행량**(x_ref 가 d 보다 얼마나
    # 앞서는가)이다. 두 브랜치가 이것만 맞으면 같은 물리를 배운다.
    #
    #     합성:  D_noise + K
    #     실측:  d_recorded + K'      (K' 는 세션마다 timeline 모드가 유도)
    #
    # 재현성 실측(2026-08-06, 82세션): aligned_lag_median 142.0~144.1, std 0.32.
    # 기하 예측(REF 0.10m → ERR 1.10m = 140 샘플)과 2.5 샘플 안에서 일치한다.
    # 절대 지연(폭 189)과 달리 이 양은 재현된다.
    advances: list[float] = []
    for session in recorded_report["sessions"]:
        alignment = session.get("alignment") or {}
        residual = alignment.get("source_err_delay_median_samples")
        if residual is None:
            continue
        value = float(residual)
        if math.isfinite(value):
            advances.append(value)
    minimum_sessions = int(readiness_cfg.get("min_delay_crosscheck_sessions", 8))
    if len(advances) < minimum_sessions:
        audit.fail(
            "measured_source_delay_agreement",
            f"정렬 잔여 지연 표본이 {len(advances)}개로 최소 {minimum_sessions}개에 "
            "미달합니다 — 표본이 없으면 두 브랜치가 같은 과제를 준다고 말할 수 없습니다",
            observations=len(advances),
        )
        return

    residual_median = float(np.median(np.asarray(advances, dtype=np.float64)))
    residual_spread = float(np.max(advances) - np.min(advances))
    lead = int(data_cfg.get("digital_reference_lead_samples", 0))
    d_noise = int(data_cfg.get("d_noise_delay_samples", 0))
    lead_mode = str(data_cfg.get("recorded_lead_mode", "constant"))
    tolerance = float(readiness_cfg.get("max_measured_delay_mismatch_samples", 64.0))

    if d_noise <= 0:
        audit.fail(
            "measured_source_delay_agreement",
            "data.d_noise_delay_samples 가 없습니다 — 합성 브랜치의 총 선행량을 알 수 "
            "없으면 실측 브랜치와 맞출 수 없습니다 (duct.yaml 에서 통과됩니다)",
        )
        return

    synthetic_advance = float(d_noise + lead)
    recorded_lead = (
        float(lead)
        if lead_mode == "constant"
        else max(0.0, synthetic_advance - residual_median)
    )
    recorded_advance = residual_median + recorded_lead
    mismatch = abs(synthetic_advance - recorded_advance)
    if mismatch > tolerance:
        audit.fail(
            "measured_source_delay_agreement",
            "두 학습 브랜치가 모델에게 주는 **총 선행량**이 다릅니다: "
            f"합성 {synthetic_advance:.0f} (D_noise {d_noise} + lead {lead}) vs "
            f"실측 {recorded_advance:.1f} (잔여 {residual_median:.1f} + lead {recorded_lead:.0f}) "
            f"— 차이 {mismatch:.1f} > 허용 {tolerance:.0f} 샘플 "
            f"({mismatch / 48.0:.1f} ms). 같은 모델이 두 브랜치에서 다른 예측 과제를 "
            "배웁니다. data.recorded_lead_mode=timeline 으로 세션마다 lead 를 유도하세요 "
            f"(현재 {lead_mode!r})",
            synthetic_advance_samples=synthetic_advance,
            recorded_advance_samples=recorded_advance,
            recorded_lead_mode=lead_mode,
            residual_median_samples=residual_median,
        )
        return

    audit.pass_(
        "measured_source_delay_agreement",
        f"두 브랜치의 총 선행량이 일치합니다: 합성 {synthetic_advance:.0f} vs 실측 "
        f"{recorded_advance:.1f} (차이 {mismatch:.1f} <= {tolerance:.0f} 샘플). "
        f"정렬 잔여 지연 {residual_median:.1f} 샘플, {len(advances)}세션 산포 "
        f"{residual_spread:.1f} — 절대 지연(산포 189)과 달리 이 양은 재현됩니다",
        synthetic_advance_samples=synthetic_advance,
        recorded_advance_samples=recorded_advance,
        residual_median_samples=residual_median,
        residual_spread_samples=residual_spread,
        recorded_lead_mode=lead_mode,
        sessions=len(advances),
    )
    return

    try:
        with np.load(_repo_path(primary["path"]), allow_pickle=False) as data:
            derived = derive_playback_to_error_delay_samples(
                int(primary["delay_samples"]), data["fir"]
            )
    except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
        audit.fail("measured_source_delay_agreement", f"P(z) 유도 실패: {exc}")
        return

    observed = float(np.median(np.asarray(observations, dtype=np.float64)))
    result = check_measured_delay_agreement(
        observed,
        derived,
        tolerance_samples=float(
            readiness_cfg.get("max_measured_delay_mismatch_samples", 64.0)
        ),
        observation_count=len(observations),
    )
    if result.ok:
        audit.pass_("measured_source_delay_agreement", result.detail, **result.measured)
    else:
        audit.fail("measured_source_delay_agreement", result.detail, **result.measured)


DESIGN_CEILING_TOLERANCE_DB = 0.5
"""선언된 설계 상한과 아티팩트 재계산의 허용 차이.

수치 조건(대역제한 필터 차수, Tikhonov λ, 탭 수)이 조금 달라도 0.2~0.3 dB 는 움직인다.
0.5 는 그 폭보다 크고, 이 저장소에서 실제로 문제가 된 오차(150-600Hz 의 6.53 을
150-1600Hz 에 쓴 것 = 약 2 dB)보다 훨씬 작다.
"""


def _audit_plant_confidence_ceiling(
    audit: "_Audit",
    readiness_cfg: dict,
    secondary: dict | None,
    primary: dict | None,
    lead_samples: int = 0,
) -> None:
    """G1c — 이 플랜트로 **목표를 낼 수 있기는 한가**를 학습 시작 전에 판정한다.

    지금까지 ``min_path_consistency: 0.9`` 는 근거 없는 숫자로 보였지만, 실제로는
    "상쇄 상한 9.54 dB" 를 뜻했다 — 아무도 그렇게 읽지 않았다. 목표를 dB 로 적고
    필요한 γ 를 역산하면 임계가 임의의 숫자가 아니라 **물리로부터 유도된 값**이 된다.

    두 상한 중 **작은 쪽**으로 판정한다
    ----------------------------------
    γ 기반 상한은 플랜트를 몰라서 잃는 몫만 센다. 복구된 플랜트에서 그 값은 약 28 dB
    인데, 정규방정식으로 직접 계산한 설계 상한은 **6.53 dB** (M=2048, lead=116,
    150-600Hz)다. 4배 이상 낙관적이다 — 인과성과 FIR 길이가 진짜 병목이기 때문이다.
    낙관적인 상한 하나만 믿는 것이 이 저장소에서 반복된 사고의 형태이므로, 설정이
    ``measured_design_ceiling_db`` 로 직접 계산한 값을 선언하면 그쪽도 함께 본다.
    """

    target_value = readiness_cfg.get("target_cancellation_db")
    if target_value is None:
        audit.fail(
            "plant_confidence_ceiling",
            "readiness.target_cancellation_db 가 없습니다 — 목표를 선언하지 않으면 "
            "달성 가능 여부를 판정할 수 없고, min_path_consistency 도 근거를 잃습니다",
        )
        return
    if secondary is None or primary is None:
        audit.fail(
            "plant_confidence_ceiling",
            "유효한 official P/S 가 없어 달성 가능 상한을 계산할 수 없습니다",
        )
        return

    target = float(target_value)
    margin = float(readiness_cfg.get("cancellation_ceiling_margin_db", 3.0))
    gamma_s = float(secondary["consistency"])
    gamma_p = float(primary["consistency"])
    try:
        gamma_ceiling = achievable_cancellation_ceiling_db(gamma_s, gamma_p)
    except ValueError as exc:
        audit.fail("plant_confidence_ceiling", f"상한 계산 실패: {exc}")
        return

    design_value = readiness_cfg.get("measured_design_ceiling_db")
    if design_value is None:
        # **선언 생략도 우회다.** 값 날조(=30.0)는 2026-08-06 에 막았지만 생략(=null)은
        # 남아 있었고, 그러면 구속 상한이 플랜트 일관성(27.73 dB)으로 폴백해 통과한다.
        # 실제 인과 FIR 상한은 최악 옥타브에서 2.16 dB 이므로 12배 낙관적인 값으로
        # 통과하는 것이다. "선언 안 하면 검사 안 함" 은 검사가 없는 것과 같다.
        audit.fail(
            "plant_confidence_ceiling",
            "readiness.measured_design_ceiling_db 가 선언되지 않았습니다 — 선언이 없으면 "
            "구속 상한이 플랜트 일관성(약 27.7 dB)으로 폴백해 물리적으로 불가능한 목표도 "
            "통과합니다. 실측 인과 FIR 상한은 최악 옥타브에서 2.16 dB 입니다. "
            "measured_design_ceiling_db 와 measured_design_ceiling_band_hz 를 선언하세요 "
            "(게이트가 아티팩트에서 다시 풀어 대조하므로 날조할 수 없습니다)",
        )
        return
    if design_value is not None:
        # 상한은 **대역이 붙어야 숫자다.** 2026-08-06 통합 검증에서 잡힌 실제 오판정:
        # 설정이 6.53 dB 를 선언했는데 그것은 150-600Hz 에서 푼 값이었고,
        # required_path_band_hz 는 [150, 1600] 이었다. 같은 플랜트를 150-1600Hz 에서
        # 다시 풀면 상한은 4.4~4.7 dB 다(독립 재현 2회). 즉 게이트가 2 dB 낙관적인
        # 숫자로 통과하고 있었고, 그 오판정 방향이 정확히 "고역 방치"(절대목표 1)와
        # 같다. 대역 없는 float 하나에 상한 전체를 거는 것이 발생기 A 그 자체이므로,
        # 이제 선언에 대역을 함께 요구하고 요구 대역을 덮는지 검사한다.
        declared_band = readiness_cfg.get("measured_design_ceiling_band_hz")
        required_band = readiness_cfg.get("required_path_band_hz")
        if declared_band is None:
            audit.fail(
                "plant_confidence_ceiling",
                "readiness.measured_design_ceiling_db 가 선언됐는데 "
                "measured_design_ceiling_band_hz 가 없습니다 — 설계 상한은 대역마다 다르므로"
                "(같은 플랜트: 150-600Hz 6.53 dB vs 150-1600Hz 4.6 dB) 대역 없는 값은 "
                "어느 요구에 대한 상한인지 알 수 없고, 넓은 대역을 좁은 대역의 값으로 "
                "통과시키는 사고가 실제로 있었습니다",
                design_ceiling_db=float(design_value),
            )
            return
        try:
            band_lo, band_hi = (float(v) for v in tuple(declared_band)[:2])
        except (TypeError, ValueError):
            audit.fail(
                "plant_confidence_ceiling",
                "readiness.measured_design_ceiling_band_hz 를 [lo, hi] 로 읽을 수 "
                f"없습니다: {declared_band!r}",
            )
            return
        if required_band is not None:
            req_lo, req_hi = (float(v) for v in tuple(required_band)[:2])
            if band_lo > req_lo + 1e-6 or band_hi < req_hi - 1e-6:
                audit.fail(
                    "plant_confidence_ceiling",
                    f"설계 상한 {float(design_value):.2f} dB 를 잰 대역 "
                    f"[{band_lo:.0f}, {band_hi:.0f}]Hz 가 요구 대역 "
                    f"[{req_lo:.0f}, {req_hi:.0f}]Hz 를 덮지 못합니다 — 좁은 대역에서 푼 "
                    "상한은 넓은 대역에서 반드시 낙관적입니다(상쇄가 어려운 구간이 빠져 "
                    "있다). 요구 대역 전체에서 정규방정식을 다시 풀어 선언하세요",
                    design_ceiling_db=float(design_value),
                    design_ceiling_band_hz=[band_lo, band_hi],
                    required_path_band_hz=[req_lo, req_hi],
                )
                return
    # ---- 선언값을 **아티팩트에서 다시 풀어** 대조한다 ------------------------------
    # 2026-08-06 감사가 재현한 fail-open: 이 값은 사람이 한 번 계산해 설정에 적은
    # 숫자였고 게이트는 그것을 그대로 믿었다. `--set readiness.measured_design_ceiling_db=30.0`
    # 으로 날조해도 PASS 하는 것을 직접 확인했다(100.0 도 마찬가지). 배선을 고쳐 P/S 를
    # 다시 재면 sha 는 바뀌지만 설정의 숫자는 그대로 남아 계속 통과한다 — 게이트가
    # 자기 자신을 증명하는 구조였다.
    recomputed: float | None = None
    worst_octave_hz: float | None = None
    recompute_note = ""
    if design_value is not None:
        try:
            from ..dsp.design_ceiling import (
                cached_design_ceiling_db,
                worst_octave_ceiling_db,
            )

            band_lo, band_hi = (
                float(v) for v in tuple(readiness_cfg["measured_design_ceiling_band_hz"])[:2]
            )
            solved = cached_design_ceiling_db(
                _repo_path(primary["path"]),
                _repo_path(secondary["path"]),
                lead_samples=int(lead_samples),
                band_hz=(band_lo, band_hi),
                sample_rate=float(secondary["sample_rate"]),
            )
            # **옥타브별 최악값이 진짜 구속이다.** 대역평균은 저역의 큰 여유가 중역의
            # 병목을 가린다 — 실측 official 에서 전대역 4.83 dB 인데 옥타브 500 은
            # 2.159 dB 뿐이다. 절대목표 1의 평가(G4)가 옥타브별이므로 진입 게이트도
            # 같은 축에서 판정해야 한다. 평균이 최악값을 가리는 것이 이 저장소가
            # 반복해서 겪은 실패 형태다.
            worst_octave_db, worst_octave_hz = worst_octave_ceiling_db(
                _repo_path(primary["path"]),
                _repo_path(secondary["path"]),
                lead_samples=int(lead_samples),
                band_hz=(band_lo, band_hi),
                sample_rate=float(secondary["sample_rate"]),
            )
            recomputed = float(min(solved.ceiling_db, worst_octave_db))
            if not solved.stable_over_regularisation:
                recompute_note = " (⚠ 정규화에 민감 — 수치를 신뢰하기 어렵다)"
        except Exception as exc:  # noqa: BLE001 — 재계산 실패는 판정 불가다
            audit.fail(
                "plant_confidence_ceiling",
                f"선언된 설계 상한을 아티팩트에서 다시 풀지 못했습니다: "
                f"{type(exc).__name__}: {exc} — 재계산할 수 없으면 선언값이 맞다고 "
                "주장할 수 없습니다",
                design_ceiling_db=float(design_value),
            )
            return
        # 선언이 재계산보다 **낙관적**이면 거부한다. 보수적인 쪽은 허용한다 —
        # 사람이 여유를 더 두는 것은 안전한 방향이다.
        if float(design_value) > recomputed + DESIGN_CEILING_TOLERANCE_DB:
            audit.fail(
                "plant_confidence_ceiling",
                f"선언된 설계 상한 {float(design_value):.2f} dB 가 아티팩트에서 다시 푼 값 "
                f"{recomputed:.2f} dB 보다 낙관적입니다 (허용 오차 "
                f"{DESIGN_CEILING_TOLERANCE_DB:.2f} dB){recompute_note}. 선언값을 재계산값으로 "
                "맞추거나, 계산 조건(대역·탭수·lead)이 다르다면 그 근거를 남기세요 — "
                "설정에 적힌 숫자를 게이트가 그대로 믿으면 물리적으로 불가능한 목표도 "
                "통과합니다",
                design_ceiling_db=float(design_value),
                recomputed_design_ceiling_db=recomputed,
            )
            return

    design_ceiling = float(design_value) if design_value is not None else float("inf")
    ceiling = min(gamma_ceiling, design_ceiling)
    binding = "정규방정식 설계 상한" if design_ceiling <= gamma_ceiling else "플랜트 일관성"
    details = {
        "gamma_ceiling_db": gamma_ceiling,
        "design_ceiling_db": design_ceiling if math.isfinite(design_ceiling) else None,
        "recomputed_ceiling_db": recomputed,
        "worst_octave_hz": worst_octave_hz if recomputed is not None else None,
        "design_ceiling_band_hz": (
            [float(v) for v in tuple(readiness_cfg["measured_design_ceiling_band_hz"])[:2]]
            if design_value is not None
            else None
        ),
        "binding_ceiling_db": ceiling,
        "binding_constraint": binding,
        "target_db": target,
        "margin_db": margin,
        "gamma_secondary": gamma_s,
        "gamma_primary": gamma_p,
    }
    if ceiling < target + margin:
        audit.fail(
            "plant_confidence_ceiling",
            f"달성 가능 상한이 {ceiling:.2f} dB 인데 목표는 {target:.2f} dB 입니다 "
            f"(여유 {margin:.1f} dB 필요; 구속 조건 = {binding}). "
            f"플랜트 일관성 상한 {gamma_ceiling:.2f} dB, 설계 상한 "
            + (
                f"{design_ceiling:.2f} dB. "
                if math.isfinite(design_ceiling)
                else "미선언. "
            )
            + f"필요 γ ≥ {required_consistency_for(target + margin):.4f} "
            f"(실측 S={gamma_s:.4f} P={gamma_p:.4f}). "
            "학습이 아니라 재측정 또는 목표 재설정이 필요합니다",
            **details,
        )
        return
    audit.pass_(
        "plant_confidence_ceiling",
        f"플랜트가 목표 {target:.1f} dB 를 여유 {margin:.1f} dB 로 허용합니다 "
        f"(구속 상한 {ceiling:.2f} dB = {binding})",
        **details,
    )


def audit_finetune_readiness(cfg: dict, *, full_recorded_qa: bool = True) -> dict:
    """resolved train config의 G1–G3 진입 조건을 한 번에 검사한다."""

    audit = _Audit("finetune_readiness")
    readiness_cfg = cfg.get("readiness", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    duct_cfg = cfg.get("duct", {}) or {}

    required_flags = (
        "require_measured_primary_path",
        "require_init_checkpoint",
        "require_recorded_manifest",
    )
    missing_flags = [name for name in required_flags if cfg.get(name) is not True]
    if missing_flags:
        audit.fail(
            "config_fail_closed_flags",
            "필수 fail-closed 설정이 true가 아닙니다",
            missing=missing_flags,
        )
    else:
        audit.pass_("config_fail_closed_flags", "필수 fail-closed 설정 3종이 활성입니다")

    reference_mode = str(data_cfg.get("reference_mode", ""))
    primary_mode = str(data_cfg.get("digital_primary_path_mode", ""))
    if reference_mode != "digital" or primary_mode != "measured":
        audit.fail(
            "measured_primary_mode",
            "fine-tune은 digital reference + measured P(z)여야 합니다",
            reference_mode=reference_mode,
            digital_primary_path_mode=primary_mode,
        )
    else:
        audit.pass_("measured_primary_mode", "digital measured P(z) 모드입니다")

    required_ratio = float(readiness_cfg.get("required_recorded_ratio", 0.7))
    recorded_ratio = float(cfg.get("recorded_ratio", float("nan")))
    if not math.isfinite(recorded_ratio) or not math.isclose(
        recorded_ratio, required_ratio, rel_tol=0.0, abs_tol=1e-9
    ):
        audit.fail(
            "recorded_mix_ratio",
            "실측/합성 혼합비가 승인된 값과 다릅니다",
            recorded_ratio=recorded_ratio,
            required_recorded_ratio=required_ratio,
        )
    else:
        audit.pass_(
            "recorded_mix_ratio",
            "실측/합성 혼합비가 정합합니다",
            recorded_ratio=recorded_ratio,
        )

    sample_rate = int(data_cfg.get("sample_rate", 0))
    raw_band = readiness_cfg.get(
        "required_path_band_hz", list(DEFAULT_REQUIRED_PATH_BAND_HZ)
    )
    required_band = (float(raw_band[0]), float(raw_band[1]))
    min_consistency = float(readiness_cfg.get("min_path_consistency", 0.9))
    secondary_value = duct_cfg.get("secondary_path", {}).get("npz")
    primary_value = duct_cfg.get("digital_reference", {}).get("primary_path_npz")
    secondary = None
    primary = None
    try:
        if not secondary_value:
            raise ValueError("duct.secondary_path.npz가 비었습니다")
        secondary = audit_official_path_model(
            secondary_value,
            expected_output_channel="cancel",
            sample_rate=sample_rate,
            required_band_hz=required_band,
            min_consistency=min_consistency,
        )
        audit.pass_(
            "official_secondary_path",
            "S(z) official ESS 품질·채널·대역 게이트가 통과했습니다",
            secondary=secondary,
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        audit.fail("official_secondary_path", str(exc))

    try:
        if not primary_value:
            raise ValueError("duct.digital_reference.primary_path_npz가 비었습니다")
        primary = audit_official_path_model(
            primary_value,
            expected_output_channel="noise",
            sample_rate=sample_rate,
            required_band_hz=required_band,
            min_consistency=min_consistency,
        )
        audit.pass_(
            "official_primary_path",
            "P(z) official ESS 품질·채널·대역 게이트가 통과했습니다",
            primary=primary,
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        audit.fail("official_primary_path", str(exc))

    try:
        if primary is None or secondary is None:
            raise ValueError("유효한 official P/S가 모두 있어야 측정 조건을 비교할 수 있습니다")
        if primary["path"] == secondary["path"]:
            raise ValueError("P(z)와 S(z)가 같은 파일을 가리킵니다")
        if primary["method"] != secondary["method"]:
            raise ValueError(
                f"P/S 측정 방식 불일치: P={primary['method']!r}, S={secondary['method']!r}"
            )
        if primary["method"] == "interleaved_multitone":
            # 동시 측정의 근거는 값의 우연한 일치가 아니라 **같은 캡처였다는 사실**이다.
            # capture_id 가 다르면 두 파일은 서로 다른 재생에서 나왔고, 그 사이의
            # 클록 wander 가 상대 지연에 그대로 실린다 — lead 가 조용히 틀린다.
            left = primary["interleaved"]["capture_id"]
            right = secondary["interleaved"]["capture_id"]
            if left != right:
                raise ValueError(
                    f"P/S capture_id 불일치: P={left!r}, S={right!r} — 동시 측정이 아닙니다"
                )
        for key in ("amplitude", "calibration_block_size", "calibration_latency"):
            left, right = primary[key], secondary[key]
            equal = (
                math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
                if key == "amplitude"
                else left == right
            )
            if not equal:
                raise ValueError(
                    f"P/S 측정 조건 불일치: {key}: P={left!r}, S={right!r}"
                )
        # P−S 상대 τ 를 **저장된 궤적에서 직접** 본다.
        #
        # 결함 1 의 결정적 증거(repeat_tau_samples)는 2026-08-04 당시에도 두 NPZ 안에
        # 전부 들어 있었다. 게이트는 그중 하나도 열어보지 않고, 측정 스크립트가 요약해
        # 써 넣은 스칼라 ``delay_spread_samples`` 만 봤다. 그 스칼라는 range(max−min)
        # 이라 "11개가 1.2, 5개가 32" 라는 이봉 구조를 32 라는 한 숫자로 뭉갠다.
        # 궤적을 직접 보면 계단이 계단으로 보인다.
        tau_check = _relative_tau_check(primary, secondary)
        if tau_check is not None and not tau_check.ok:
            raise ValueError(tau_check.detail)
        audit.pass_(
            "matched_path_measurement_conditions",
            "P/S official ESS 디지털 gain·block·latency 조건이 정합하고 "
            "P−S 상대 τ 궤적이 상수입니다",
            primary=primary,
            secondary=secondary,
            relative_tau=None if tau_check is None else tau_check.measured,
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        audit.fail("matched_path_measurement_conditions", str(exc))

    configured_lead = int(data_cfg.get("digital_reference_lead_samples", -1))
    configured_primary_delay = duct_cfg.get("digital_reference", {}).get(
        "d_noise_delay_samples"
    )
    if primary is not None and secondary is not None:
        # handoff 도 lead 도 여기서 다시 유도하지 않는다 — trainer / eval 과 **같은
        # 함수**를 부른다. 두 곳이 각자 유도해 109 와 113 으로 갈라졌던 것이
        # 커밋 aaeef41 의 사고이고, 그때 양쪽 다 자기 기준으로는 "통과" 였다.
        delays = PlantDelays.from_config(
            duct_cfg=duct_cfg,
            secondary_delay_samples=int(secondary["delay_samples"]),
            primary_delay_samples=int(primary["delay_samples"]),
            sample_rate=int(sample_rate),
        )
        handoff = int(delays.handoff_samples)
        lead_check = check_lead_agreement(configured_lead, delays)
        expected_lead = int(lead_check.measured["derived_lead_samples"])
        delay_matches = (
            configured_primary_delay is not None
            and int(configured_primary_delay) == int(primary["delay_samples"])
        )
        if not lead_check.ok or not delay_matches:
            audit.fail(
                "path_delay_and_lead",
                "P/S 순수지연과 fine-tune lead 설정이 다릅니다",
                configured_lead=configured_lead,
                expected_lead=expected_lead,
                configured_primary_delay=configured_primary_delay,
                measured_primary_delay=primary["delay_samples"],
            )
        else:
            audit.pass_(
                "path_delay_and_lead",
                "P/S 지연·handoff·digital lead가 정합합니다",
                digital_reference_lead_samples=configured_lead,
                primary_delay_samples=primary["delay_samples"],
                secondary_delay_samples=secondary["delay_samples"],
                handoff_extra_samples=handoff,
            )
    else:
        audit.fail(
            "path_delay_and_lead",
            "유효한 official P/S가 없어 lead를 검증할 수 없습니다",
        )

    # 파인튜닝이 개선을 요구할 대역. trainer 와 **같은 함수**로 유도한다 —
    # 여기서 손으로 교집합을 쓰면 그것이 여섯 번째 복붙이다(발생기 A).
    expected_optimize_band: FrequencyBand | None = None
    if secondary is not None:
        try:
            expected_optimize_band = BandPlan.resolve(
                # trainer 와 **같은 로더·같은 규칙**: consistency_band 가 있으면 그것,
                # 없으면 excitation_band. 여기서 NPZ 키를 직접 읽으면 두 번째 규칙이 된다.
                plant_trusted_band_hz=load_secondary_path(
                    secondary["path"]
                ).trusted_band_hz(),
                duct_cfg=duct_cfg,
                sample_rate=sample_rate,
            ).optimize
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            expected_optimize_band = None

    try:
        init_value = cfg.get("init_ckpt")
        if not init_value:
            raise ValueError("init_ckpt가 비었습니다")
        if expected_optimize_band is None:
            raise ValueError(
                "S(z) 신뢰 대역을 알 수 없어 init checkpoint 의 학습 대역을 검증할 수 "
                "없습니다 — official_secondary_path 를 먼저 통과시키세요"
            )
        init = audit_init_checkpoint(
            init_value,
            expected_model_cfg=cfg.get("model", {}),
            expected_lead=configured_lead,
            expected_optimize_band=expected_optimize_band,
            max_lead_mismatch_samples=int(
                readiness_cfg.get("max_init_lead_mismatch_samples", 0)
            ),
            require_completed=bool(
                readiness_cfg.get("require_completed_init_checkpoint", True)
            ),
            max_best_metric_db=float(
                readiness_cfg.get("max_init_best_metric_db", 0.0)
            ),
            allowed_physics_statuses=tuple(
                str(value)
                for value in readiness_cfg.get(
                    "allowed_init_physics_statuses",
                    ["secondary_surrogate_representation_pretrain"],
                )
            ),
        )
        audit.pass_(
            "completed_init_checkpoint",
            "사전학습 init best와 완료 last가 정합합니다",
            checkpoint=init,
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        audit.fail("completed_init_checkpoint", str(exc))

    manifest_value = cfg.get("recorded_manifest")
    entries: list[dict] = []
    recorded_report: dict | None = None
    try:
        if not manifest_value:
            raise ValueError("recorded_manifest가 비었습니다")
        manifest_path = _repo_path(manifest_value).resolve()
        entries = read_manifest(manifest_path)
        if full_recorded_qa:
            settings = settings_from_data_config(
                data_cfg,
                required_splits=("train", "val", "test"),
                allow_incomplete_family_coverage=False,
                alignment_overrides=_alignment_overrides(readiness_cfg),
            )
            recorded_report = validate_recorded_sessions(
                entries, settings, manifest_path=str(manifest_path)
            )
            if not recorded_report["ok"]:
                messages = [*recorded_report.get("errors", [])]
                for session in recorded_report.get("sessions", []):
                    messages.extend(session.get("errors", []))
                    if len(messages) >= 8:
                        break
                raise ValueError("recorded 전수 QA FAIL: " + "; ".join(messages[:8]))
        else:
            recorded_report = {
                "ok": True,
                "summary": {
                    "sessions": len(entries),
                    "duration_s": sum(float(e.get("duration_s", 0.0)) for e in entries),
                    "source_families": {},
                },
            }
        summary = recorded_report["summary"]
        min_sessions = int(readiness_cfg.get("min_recorded_sessions", 80))
        min_duration = float(
            readiness_cfg.get("min_recorded_duration_seconds", 90.0 * 60.0)
        )
        required_families = _required_families(readiness_cfg)
        observed_families = {
            str(entry.get("source_family", "")) for entry in entries
        }
        missing_families = sorted(set(required_families).difference(observed_families))
        if int(summary.get("sessions", 0)) < min_sessions:
            raise ValueError(
                f"recorded 세션 {summary.get('sessions', 0)}개 < 최소 {min_sessions}개"
            )
        if float(summary.get("duration_s", 0.0)) < min_duration:
            raise ValueError(
                f"recorded 분량 {float(summary.get('duration_s', 0.0)) / 60.0:.1f}분 "
                f"< 최소 {min_duration / 60.0:.1f}분"
            )
        if missing_families:
            raise ValueError(f"필수 source_family 누락: {missing_families}")
        audit.pass_(
            "recorded_dataset_qa",
            "recorded 전수 QA·분할·family·최소 분량이 통과했습니다",
            manifest=str(manifest_path),
            manifest_sha256=sha256_file(manifest_path),
            sessions=int(summary.get("sessions", 0)),
            duration_seconds=float(summary.get("duration_s", 0.0)),
            source_families=sorted(observed_families),
            full_recorded_qa=bool(full_recorded_qa),
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        audit.fail("recorded_dataset_qa", str(exc))

    _audit_recorded_alignment(audit, readiness_cfg, recorded_report, full_recorded_qa)
    _audit_statistical_power(audit, readiness_cfg, entries)
    _audit_corpus_leak(audit, readiness_cfg, data_cfg, entries)
    _audit_measured_source_delay(
        audit, readiness_cfg, primary, recorded_report, data_cfg
    )
    _audit_plant_confidence_ceiling(
        audit,
        readiness_cfg,
        secondary,
        primary,
        # lead 는 path_delay_and_lead 게이트가 이미 실측과 대조한 값이다.
        # 여기서 다시 유도하면 그것이 두 번째 유도가 된다 (발생기 A).
        lead_samples=int(data_cfg.get("digital_reference_lead_samples", 0)),
    )

    return audit.report(
        stage=str(cfg.get("stage", "")),
        ckpt_dir=str(cfg.get("ckpt_dir", "")),
        full_recorded_qa=bool(full_recorded_qa),
    )


def require_finetune_readiness(cfg: dict, *, full_recorded_qa: bool = True) -> dict:
    """준비 감사가 실패하면 학습 시작 전에 단일 예외로 중단한다."""

    report = audit_finetune_readiness(cfg, full_recorded_qa=full_recorded_qa)
    if not report["ok"]:
        failures = [item["message"] for item in report["checks"] if not item["ok"]]
        raise RuntimeError("파인튜닝 준비 게이트 FAIL:\n- " + "\n- ".join(failures))
    return report


def _audit_g4_metrics(
    path: str | Path,
    *,
    expected_split: str,
    checkpoint_sha256: str,
    manifest_sha256: str,
    required_source_families: tuple[str, ...],
) -> dict[str, Any]:
    metrics_path = _repo_path(path).resolve()
    if not metrics_path.is_file():
        raise FileNotFoundError(f"recorded {expected_split} metrics가 없습니다: {metrics_path}")
    with np.load(metrics_path, allow_pickle=False) as data:
        required = {
            "split",
            "physics_status",
            "allow_surrogate",
            "checkpoint_sha256",
            "manifest_sha256",
            "g4_trusted_pass",
            "g4_fullband_pass",
            "g4_pass",
            "source_family",
            "n_sessions",
            "n_segments",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"G4 provenance/판정 필드 누락: {missing}")
        split = str(_npz_scalar(data, "split"))
        physics_status = str(_npz_scalar(data, "physics_status"))
        allow_surrogate = bool(_npz_scalar(data, "allow_surrogate"))
        saved_checkpoint_sha = str(_npz_scalar(data, "checkpoint_sha256"))
        saved_manifest_sha = str(_npz_scalar(data, "manifest_sha256"))
        trusted_pass = bool(_npz_scalar(data, "g4_trusted_pass"))
        fullband_pass = bool(_npz_scalar(data, "g4_fullband_pass"))
        g4_pass = bool(_npz_scalar(data, "g4_pass"))
        # 기능 2(모든 소리 제거)는 소스별 **최악값** 문제다. 이 필드가 없는 metrics.npz 는
        # 최악 소스를 보지 않던 옛 평가기의 산출물이므로 통과시키지 않는다 — 평균만 보면
        # 대화를 6 dB 증폭하는 모델이 PASS 한다.
        if "g4_source_pass" not in data.files:
            raise ValueError(
                "G4 metrics.npz 에 g4_source_pass 가 없습니다 — 최악 source family 판정을 "
                "하지 않는 구버전 평가기의 산출물입니다. evaluate_recorded.py 로 재평가하세요."
            )
        source_pass = bool(_npz_scalar(data, "g4_source_pass"))
        # 2026-08-05 신설 판정. 없는 것은 구버전 산출물이므로 통과시키지 않는다 —
        # 이미 g4_source_pass 에 같은 관례를 쓰고 있고, 그 관례가 옳았다.
        modern = {
            "g4_verdict",
            "g4_do_no_harm_pass",
            "g4_power_pass",
            "g4_ci_pass",
            "plant_fingerprint_json",
        }
        missing_modern = sorted(modern.difference(data.files))
        if missing_modern:
            raise ValueError(
                f"G4 metrics.npz 에 {missing_modern} 가 없습니다 — 대역 밖 do-no-harm"
                "(절대목표 1), 통계적 검정력, 플랜트 지문을 판정하지 않는 구버전 "
                "평가기의 산출물입니다. evaluate_recorded.py 로 재평가하세요."
            )
        verdict = str(_npz_scalar(data, "g4_verdict"))
        do_no_harm_pass = bool(_npz_scalar(data, "g4_do_no_harm_pass"))
        power_pass = bool(_npz_scalar(data, "g4_power_pass"))
        ci_pass = bool(_npz_scalar(data, "g4_ci_pass"))
        fingerprint_json = str(_npz_scalar(data, "plant_fingerprint_json"))
        worst_octave_hz = float(_npz_scalar(data, "g4_worst_octave_center_hz"))
        worst_octave_db = float(_npz_scalar(data, "g4_worst_octave_worst10_db"))
        worst_source_db = float(_npz_scalar(data, "g4_worst_source_trusted_mean_db"))
        worst_source_family = str(_npz_scalar(data, "g4_worst_source_family"))
        families = {str(value) for value in np.asarray(data["source_family"]).tolist()}
        n_sessions = int(_npz_scalar(data, "n_sessions"))
        n_segments = int(_npz_scalar(data, "n_segments"))
    errors: list[str] = []
    if split != expected_split:
        errors.append(f"split={split!r}; expected={expected_split!r}")
    if physics_status != "measured_primary_path" or allow_surrogate:
        errors.append(
            f"물리 상태가 measured가 아닙니다: {physics_status}, allow_surrogate={allow_surrogate}"
        )
    if saved_checkpoint_sha != checkpoint_sha256:
        errors.append("평가 checkpoint SHA-256이 완료 후보와 다릅니다")
    if saved_manifest_sha != manifest_sha256:
        errors.append("평가 manifest SHA-256이 readiness manifest와 다릅니다")
    if not (trusted_pass and fullband_pass and source_pass and g4_pass):
        errors.append(
            "G4 판정을 통과하지 못했습니다: "
            f"trusted={trusted_pass}, fullband={fullband_pass}, "
            f"source(기능2 최악값)={source_pass} "
            f"[최악 {worst_source_family or 'n/a'} {worst_source_db:+.2f} dB], g4={g4_pass}"
        )
    # ``INCONCLUSIVE`` 는 PASS 가 아니다. 표본 부족으로 아무 말도 할 수 없는 상태를
    # 완료로 기록하면 게이트가 있는 것이 없는 것보다 나쁘다.
    if verdict != "PASS":
        errors.append(
            f"G4 판정이 {verdict} 입니다 (PASS 아님) — "
            f"do_no_harm={do_no_harm_pass}, power={power_pass}, ci={ci_pass}"
        )
    if not do_no_harm_pass:
        errors.append(
            "대역 밖 do-no-harm 실패 (절대목표 1): 최악 옥타브 "
            f"{worst_octave_hz:.0f}Hz {worst_octave_db:+.2f} dB — fullband 평균 NMSE 는 "
            "d 에 에너지가 없는 대역의 증폭을 원리적으로 잡지 못한다"
        )
    if not power_pass:
        errors.append(
            "계열당 그룹 수가 부족해 G4 판정이 통계적으로 성립하지 않습니다"
        )
    if not ci_pass:
        errors.append(
            "계열별 cluster bootstrap CI 상단이 0 아래가 아닙니다 — 점추정만으로 "
            "개선을 주장할 수 없습니다"
        )
    missing_families = sorted(set(required_source_families).difference(families))
    if missing_families:
        errors.append(f"G4 source_family 결과 누락: {missing_families}")
    if n_sessions <= 0 or n_segments <= 0:
        errors.append(f"G4 평가 표본이 비었습니다: sessions={n_sessions}, segments={n_segments}")
    if errors:
        raise ValueError(f"{metrics_path}: " + "; ".join(errors))
    return {
        "path": str(metrics_path),
        "sha256": sha256_file(metrics_path),
        "split": split,
        "n_sessions": n_sessions,
        "n_segments": n_segments,
        "source_families": sorted(families),
        "g4_pass": True,
        "g4_verdict": verdict,
        "plant_fingerprint_json": fingerprint_json,
    }


def _audit_plant_identity(audit: "_Audit", fingerprints: dict[str, str]) -> None:
    """G5 — 완료 판정에 쓰인 val/test 결과가 **같은 플랜트**에서 나왔는가.

    2026-08-04 사고: 파인튜닝 전 기준선은 S 지연 1342 / lead 109 / surrogate 물리였고
    후는 1465 / 113 / measured 였다. 서로 다른 물리인데 "1.30 dB 개선"이라고 적혔다.
    비교를 막는 장치가 아무 데도 없었다.

    val 과 test 는 같은 checkpoint·같은 플랜트로 평가돼야 한다. 둘의 지문이 다르면
    두 수치를 나란히 놓는 것 자체가 성립하지 않는다.
    """

    from ..dsp.invariants import check_plant_fingerprint_match
    from ..dsp.timing import PlantFingerprint

    if len(fingerprints) < 2:
        audit.fail(
            "plant_identity_for_comparison",
            "val/test 두 평가의 플랜트 지문을 모두 얻지 못해 비교 가능성을 "
            "판정할 수 없습니다",
            available=sorted(fingerprints),
        )
        return
    try:
        models = {
            split: PlantFingerprint(**json.loads(payload))
            for split, payload in fingerprints.items()
        }
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        audit.fail("plant_identity_for_comparison", f"플랜트 지문을 읽을 수 없습니다: {exc}")
        return
    result = check_plant_fingerprint_match(models["val"], models["test"])
    if result.ok:
        audit.pass_(
            "plant_identity_for_comparison",
            f"val/test 평가가 같은 플랜트에서 나왔습니다 (digest "
            f"{models['val'].digest()[:12]})",
            digest=models["val"].digest(),
        )
    else:
        audit.fail("plant_identity_for_comparison", result.detail, **result.measured)


def audit_finetune_completion(
    cfg: dict,
    *,
    checkpoint: str | Path,
    val_metrics: str | Path,
    test_metrics: str | Path,
    full_recorded_qa: bool = True,
) -> dict:
    """measured checkpoint와 독립 val/test G4를 묶어 완료 여부를 판정한다."""

    readiness = audit_finetune_readiness(cfg, full_recorded_qa=full_recorded_qa)
    audit = _Audit("finetune_completion")
    if readiness["ok"]:
        audit.pass_("readiness", "fine-tune 진입 준비 게이트가 통과했습니다")
    else:
        audit.fail("readiness", "fine-tune 진입 준비 게이트가 통과하지 않았습니다")

    checkpoint_path = _repo_path(checkpoint).resolve()
    candidate_sha: str | None = None
    try:
        state = _load_checkpoint_state(checkpoint_path)
        saved_cfg = state["cfg"]
        if saved_cfg.get("physics_status") != "measured_primary_path":
            raise ValueError(
                "fine-tuned checkpoint physics_status가 measured_primary_path가 아닙니다"
            )
        if saved_cfg.get("model") != cfg.get("model"):
            raise ValueError("fine-tuned checkpoint 모델 설정이 현재 config와 다릅니다")
        if str(saved_cfg.get("stage")) != str(cfg.get("stage")):
            raise ValueError("fine-tuned checkpoint stage가 현재 config와 다릅니다")
        if not math.isclose(
            float(saved_cfg.get("recorded_ratio", float("nan"))),
            float(cfg.get("recorded_ratio", float("nan"))),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("fine-tuned checkpoint recorded_ratio가 현재 config와 다릅니다")
        if _checkpoint_lead(state) != int(
            cfg.get("data", {}).get("digital_reference_lead_samples", -1)
        ):
            raise ValueError("fine-tuned checkpoint lead가 현재 config와 다릅니다")
        companion_last = checkpoint_path.parent / "last.pt"
        last_state = _load_checkpoint_state(companion_last)
        last_cfg = last_state["cfg"]
        if _checkpoint_identity(last_cfg) != _checkpoint_identity(saved_cfg):
            raise ValueError("fine-tune best.pt와 last.pt의 immutable run 설정이 다릅니다")
        if _model_state_signature(last_state) != _model_state_signature(state):
            raise ValueError("fine-tune best.pt와 last.pt의 model state 구조가 다릅니다")
        target = int(
            last_cfg.get(
                "run_until_step", (last_cfg.get("schedule", {}) or {}).get("total_steps", 0)
            )
        )
        step = int(last_state.get("step", -1))
        if target <= 0 or step < target:
            raise ValueError(f"fine-tune 학습 미완료: last step={step}, target={target}")
        candidate_sha = sha256_file(checkpoint_path)
        audit.pass_(
            "measured_finetune_checkpoint",
            "measured fine-tune checkpoint와 완료 last가 정합합니다",
            checkpoint=str(checkpoint_path),
            checkpoint_sha256=candidate_sha,
            last_step=step,
            target_step=target,
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        audit.fail("measured_finetune_checkpoint", str(exc))

    manifest_path = _repo_path(cfg.get("recorded_manifest", "")).resolve()
    try:
        manifest_sha = sha256_file(manifest_path)
    except (FileNotFoundError, OSError) as exc:
        manifest_sha = ""
        audit.fail("recorded_manifest_provenance", str(exc))
    else:
        audit.pass_(
            "recorded_manifest_provenance",
            "완료 판정용 manifest 지문을 계산했습니다",
            manifest=str(manifest_path),
            manifest_sha256=manifest_sha,
        )

    required_families = _required_families(cfg.get("readiness", {}) or {})
    fingerprints: dict[str, str] = {}
    for split, path in (("val", val_metrics), ("test", test_metrics)):
        check_id = f"recorded_{split}_g4"
        if candidate_sha is None or not manifest_sha:
            audit.fail(check_id, "checkpoint/manifest provenance가 없어 G4를 검증할 수 없습니다")
            continue
        try:
            details = _audit_g4_metrics(
                path,
                expected_split=split,
                checkpoint_sha256=candidate_sha,
                manifest_sha256=manifest_sha,
                required_source_families=required_families,
            )
            fingerprints[split] = str(details.pop("plant_fingerprint_json"))
            audit.pass_(check_id, f"독립 recorded {split} G4가 통과했습니다", **details)
        except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            audit.fail(check_id, str(exc))

    _audit_plant_identity(audit, fingerprints)

    return audit.report(
        readiness=readiness,
        checkpoint=str(checkpoint_path),
        fine_tuning_complete=all(bool(item["ok"]) for item in audit.checks),
    )


def render_audit_markdown(report: dict) -> str:
    """readiness/completion JSON과 같은 판정을 간결한 Markdown으로 만든다."""

    title = (
        "파인튜닝 완료 검증"
        if report.get("kind") == "finetune_completion"
        else "파인튜닝 준비 검증"
    )
    lines = [
        f"# {title}",
        "",
        f"- 판정: **{'PASS' if report.get('ok') else 'FAIL'}**",
        f"- 검사 시각(UTC): `{report.get('checked_at_utc', '')}`",
        "",
        "| 게이트 | 판정 | 내용 |",
        "|---|---|---|",
    ]
    for item in report.get("checks", []):
        message = str(item.get("message", "")).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{item.get('id', '')}` | {'PASS' if item.get('ok') else 'FAIL'} | {message} |"
        )
    lines += [
        "",
        "> FAIL이면 학습/완료로 표시하지 않는다. 이 검사는 오디오 장치를 열지 않는다.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "audit_finetune_completion",
    "audit_finetune_readiness",
    "audit_init_checkpoint",
    "audit_official_path_model",
    "render_audit_markdown",
    "require_finetune_readiness",
    "sha256_file",
]
