"""실측 파인튜닝의 진입·완료를 실패 폐쇄 방식으로 판정한다.

이 모듈은 오디오 장치를 열지 않는다. 측정 도구가 품질 게이트를 통과해 만든
P/S NPZ, recorded manifest/파일, 사전학습 checkpoint와 독립 평가 NPZ를 읽기만
한다. 하나라도 검증할 수 없으면 ``ok=False``이며, 파일의 존재만으로 measured
파인튜닝 또는 완료를 인정하지 않는다.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import torch

from ..audio_io import analyze_int32_input_probe
from ..config import (
    CANONICAL_LOSS_PILOT_STEPS,
    REPO_ROOT,
    loss_selection_sha256,
    validate_canonical_training_policy,
)
from ..data.holdout_contract import (
    read_regular_file_snapshot,
    validate_holdout_contract,
)
from ..data.manifest import read_manifest, read_manifest_bytes
from ..data.manifest_contract import validate_manifest_generation
from ..data.recorded_generation_exclusion import (
    RecordedGenerationExclusionError,
    derive_recorded_generation_exclusion,
    find_recorded_generation_overlaps,
)
from ..data.recorded_subband_coverage import (
    CANONICAL_COVERAGE_SPLITS,
    CANONICAL_EDGE_TRIM_SECONDS,
    CANONICAL_MAX_SEGMENTS_PER_SESSION,
    build_recorded_subband_coverage_contract,
    recorded_subband_coverage_report_path,
    validate_recorded_subband_coverage_report,
)
from ..data.transfer_contract import validate_recorded_training_snapshot
# 지연·lead 부기의 단일 출처. 게이트가 lead 를 **스스로 유도하면** 그것이 두 번째
# 유도가 되고, trainer 와 갈라진 채로 양쪽 다 "통과" 한다 (발생기 A, 커밋 aaeef41).
from ..dsp.invariants import (
    ABSOLUTE_OBJECTIVE_BAND_HZ,
    REQUIRED_SOURCE_FAMILIES,
    REQUIRED_SOURCE_FAMILY_MIX_TAGS,
)
from ..dsp.secondary_path import load_secondary_path
from ..dsp.interleaved_probe import build_interleaved_probe
from ..dsp.timing import (
    BandPlan,
    FrequencyBand,
    PlantDelays,
    TrainingTimingContract,
)
from ..data.recorded_qa import (
    settings_from_data_config,
    validate_recorded_sessions,
)
from ..eval.trusted_subbands import (
    STRICT_TRUSTED_SUBBANDS_HZ,
)
from .completion_receipt import validate_completion_receipt
from .evaluation_contract import (
    canonical_test_ledger_event_paths_from_payload,
    canonical_test_ledger_paths_from_payload,
    read_json_snapshot,
    snapshot_regular_file,
    validate_persisted_g4_metrics,
    validate_test_open_selection,
)
from .experiment_contract import validate_embedded_experiment_contract


DEFAULT_REQUIRED_PATH_BAND_HZ = (80.0, 1600.0)
DEFAULT_REQUIRED_SOURCE_FAMILIES = ("speech", "music", "environment", "machine")
_RECORDED_TRANSFER_TRUST_ROLES = frozenset(
    {"canonical_finetune", "measured_probe"}
)

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
    "anchor_repeat",
    "kept_repeat_indices",
    "alignment_scores",
    "band_consistency",
    "band_consistency_hz",
    # compact FIR 변환 후의 self-describing 지연 계약과 독립 재감사 source.
    "bulk_delay_samples",
    "pre_roll_samples",
    "delay_semantics",
    "tone_frequencies_hz",
    "aligned_mean_transfer_real",
    "aligned_mean_transfer_imag",
    "aligned_mean_transfer_sha256",
    "compact_transfer_band_hz",
    "compact_transfer_tone_count",
    "compact_transfer_complex_agreement",
    "compact_transfer_relative_error",
    "minimum_compact_transfer_agreement",
    "maximum_compact_transfer_relative_error",
    "compact_transfer_subband_hz",
    "compact_transfer_subband_tone_count",
    "compact_transfer_subband_complex_agreement",
    "compact_transfer_subband_relative_error",
    # immutable raw/analysis provenance와 guard=1 clock-corrected separation 증거.
    "output_pcm_provenance",
    "source_raw_npz_path",
    "source_raw_npz_sha256",
    "source_analysis_npz_path",
    "source_analysis_npz_sha256",
    "error_mic_channel",
    "reference_mic_channel",
    "noise_output_channel",
    "cancel_output_channel",
    "operator_confirmed_user_present",
    "operator_confirmed_volume_minimum",
    "operator_confirmed_routing_and_geometry",
    "separation_algorithm",
    "separation_algorithm_version",
    "clock_estimator",
    "clock_sample_rate",
    "clock_band_hz",
    "clock_min_adjacent_score",
    "clock_max_err_ref_delta_samples",
    "clock_max_subwindow_spread_samples",
    "clock_max_adjacent_change_samples",
    "clock_max_abs_period_delta_samples",
    "clock_max_drift_deviation_samples",
    "clock_observation_repeat_indices",
    "clock_period_delta_samples",
    "clock_q_ratio",
    "clock_err_delay_samples",
    "clock_ref_delay_samples",
    "clock_err_score",
    "clock_ref_score",
    "clock_err_subwindow_spread_samples",
    "clock_ref_subwindow_spread_samples",
    "clock_err_ref_delta_samples",
    "joint_ls_expected_rank",
    "joint_ls_rank",
    "joint_ls_condition",
    "joint_ls_max_condition",
    "joint_ls_reconstruction_relative_error",
    "joint_ls_reconstruction_relative_error_p95",
    "joint_ls_max_reconstruction_relative_error_p95",
    "separation_crosscheck_band_hz",
    "separation_crosscheck_complex_agreement",
    "separation_crosscheck_relative_error",
    "separation_crosscheck_subband_hz",
    "separation_crosscheck_subband_complex_agreement",
    "separation_crosscheck_subband_relative_error",
    "minimum_separation_crosscheck_agreement",
    "maximum_separation_crosscheck_relative_error",
    "repeat_tau_samples",
    "provisional_repeat_tau_samples",
    "common_alignment_tau_samples",
    "drift_samples_per_period",
    "relative_tau_max_abs_samples",
)
INTERLEAVED_MAX_PERIOD_SECONDS = 2.0   # 실측 위상 잔차가 2.26s 에서 2.33rad 로 무너진다
INTERLEAVED_MIN_TONE_COUNT = 64
INTERLEAVED_MIN_TONE_SNR_MEDIAN_DB = 30.0
INTERLEAVED_DELAY_SEMANTICS = "effective_zeros_before_compact_fir"
INTERLEAVED_MIN_COMPACT_TRANSFER_AGREEMENT = 0.995
INTERLEAVED_MAX_COMPACT_TRANSFER_RELATIVE_ERROR = 0.10
INTERLEAVED_OUTPUT_PCM_PROVENANCE = "observed_submitted_int16"
INTERLEAVED_SEPARATION_ALGORITHM = "fractional_clock_joint_real_ls"
INTERLEAVED_SEPARATION_ALGORITHM_VERSION = 1
INTERLEAVED_CLOCK_ESTIMATOR = "adjacent_cycle_time_domain_three_subwindow_parabolic"
INTERLEAVED_CLOCK_BAND_HZ = (150.0, 1600.0)
INTERLEAVED_CLOCK_MIN_SCORE = 0.995
INTERLEAVED_CLOCK_MAX_ERR_REF_DELTA = 0.25
INTERLEAVED_CLOCK_MAX_SUBWINDOW_SPREAD = 0.35
INTERLEAVED_CLOCK_MAX_ADJACENT_CHANGE = 0.50
INTERLEAVED_CLOCK_MAX_ABS_PERIOD_DELTA = 6.0
INTERLEAVED_JOINT_LS_MAX_CONDITION = 1.25
INTERLEAVED_JOINT_LS_MAX_RESIDUAL_P95 = 0.05
INTERLEAVED_SEPARATION_MIN_AGREEMENT = 0.999
INTERLEAVED_SEPARATION_MAX_RELATIVE_ERROR = 0.01
INTERLEAVED_RAW_CAPTURE_SCHEMA = (
    "interleaved_raw_v4_user_present_observed_pcm_preanalysis"
)
INTERLEAVED_OFFICIAL_SAMPLE_RATE = 48_000
INTERLEAVED_CHANNEL_MAP_FIELDS = {
    "error_mic_channel": ("error_mic", 0),
    "reference_mic_channel": ("reference_mic", 1),
    "noise_output_channel": ("noise_out", 0),
    "cancel_output_channel": ("cancel_out", 1),
}
INTERLEAVED_OPERATOR_CONFIRMATION_FIELDS = (
    "operator_confirmed_user_present",
    "operator_confirmed_volume_minimum",
    "operator_confirmed_routing_and_geometry",
)
INTERLEAVED_MAX_RELATIVE_TAU_ABS_SAMPLES = 1.0
INTERLEAVED_COMPACT_TRANSFER_SUB_BANDS_HZ = np.asarray(
    STRICT_TRUSTED_SUBBANDS_HZ,
    dtype=np.float64,
)

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

MIN_BAND_CONSISTENCY = 0.95
MIN_INTERLEAVED_CONSISTENCY = 0.95
INTERLEAVED_OFFICIAL_BLOCK_SIZE = 256
INTERLEAVED_OFFICIAL_LATENCY = "low"
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


def _canonical_recorded_lineage_snapshot(
    manifest_path: Path, data_cfg: dict, transfer_snapshot: Any = None
) -> tuple[list[dict], str, dict[str, Any]]:
    """holdout provenance가 증명한 regrouped manifest의 동일 bytes만 반환한다."""

    manifest_dir = _repo_path(
        data_cfg.get("noise_manifest_dir", "data/manifests")
    )
    generation_path = manifest_dir / "manifest_generation.json"
    generation_snapshot = read_regular_file_snapshot(
        generation_path,
        root=REPO_ROOT,
        label="canonical manifest generation sidecar",
    )
    assert generation_snapshot.data is not None
    try:
        generation = json.loads(generation_snapshot.data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("canonical manifest generation sidecar가 손상됐습니다") from exc
    if not isinstance(generation, dict):
        raise ValueError("canonical manifest generation sidecar 최상위가 mapping이 아닙니다")
    holdout_value = generation.get("holdout")
    expected_holdout_sha = generation.get("holdout_sha256")
    if holdout_value != "data/manifests/recorded_holdout.json":
        raise ValueError(
            "canonical manifest generation holdout 경로가 고정 계약과 다릅니다: "
            f"{holdout_value!r}"
        )
    if (
        not isinstance(expected_holdout_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_holdout_sha) is None
    ):
        raise ValueError("canonical manifest generation holdout_sha256가 유효하지 않습니다")
    holdout_summary = validate_holdout_contract(
        REPO_ROOT / holdout_value,
        repo_root=REPO_ROOT,
        expected_sha256=expected_holdout_sha,
    )
    lineage = holdout_summary.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("canonical holdout provenance에 lineage summary가 없습니다")
    if (
        transfer_snapshot is not None
        and getattr(transfer_snapshot, "recorded_generation", None) is not None
    ):
        generation_summary = getattr(
            transfer_snapshot, "recorded_generation_summary", None
        )
        recorded_snapshot = getattr(transfer_snapshot, "recorded_manifest", None)
        if not isinstance(generation_summary, dict) or recorded_snapshot is None:
            raise ValueError("validated recorded generation/combined manifest snapshot이 없습니다")
        expected_path = Path(os.path.abspath(recorded_snapshot.path))
        actual_path = Path(os.path.abspath(manifest_path))
        if actual_path != expected_path:
            raise ValueError(
                "학습 recorded_manifest가 transfer-검증된 generation combined manifest와 "
                f"다릅니다: configured={actual_path}, proven={expected_path}"
            )
        combined = generation_summary.get("combined")
        additions = generation_summary.get("additions")
        if (
            not isinstance(combined, dict)
            or not isinstance(combined.get("manifest"), dict)
            or combined["manifest"].get("sha256") != recorded_snapshot.sha256
            or combined.get("session_count") != 99
            or not isinstance(additions, dict)
            or additions.get("expected_session_count") != 17
        ):
            raise ValueError("recorded generation combined/additions summary가 exact 82+17이 아닙니다")
        assert recorded_snapshot.data is not None
        entries = read_manifest_bytes(
            recorded_snapshot.data, manifest_path=actual_path
        )
        if len(entries) != 99:
            raise ValueError(f"recorded generation combined manifest row가 99가 아닙니다: {len(entries)}")
        generation_lineage = dict(lineage)
        generation_lineage.update(
            {
                "regrouped_manifest": combined["manifest"].get("path"),
                "regrouped_manifest_sha256": recorded_snapshot.sha256,
                "regrouped_row_count": len(entries),
                "component_count": int(lineage.get("component_count", 0)) + 17,
                "component_membership_sha256": hashlib.sha256(
                    (
                        str(lineage.get("component_membership_sha256", ""))
                        + str(additions.get("session_aggregate_sha256", ""))
                        + str(additions.get("source_lineage_evidence_sha256", ""))
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
        generation_holdout = dict(holdout_summary)
        generation_holdout["lineage"] = generation_lineage
        return entries, recorded_snapshot.sha256, generation_holdout
    declared_manifest = lineage.get("regrouped_manifest")
    if declared_manifest != "data/manifests/recorded_regrouped.jsonl":
        raise ValueError(
            "canonical lineage regrouped manifest 경로가 고정 계약과 다릅니다: "
            f"{declared_manifest!r}"
        )
    expected_path = Path(
        os.path.abspath(REPO_ROOT / str(declared_manifest))
    )
    actual_path = Path(os.path.abspath(manifest_path))
    if actual_path != expected_path:
        raise ValueError(
            "학습 recorded_manifest가 canonical lineage 증거의 regrouped manifest와 "
            f"다릅니다: configured={actual_path}, proven={expected_path}"
        )
    manifest_snapshot = read_regular_file_snapshot(
        actual_path,
        root=REPO_ROOT,
        label="canonical recorded regrouped manifest",
    )
    expected_manifest_sha = lineage.get("regrouped_manifest_sha256")
    if manifest_snapshot.sha256 != expected_manifest_sha:
        raise ValueError(
            "학습 recorded_manifest bytes가 canonical lineage 증거와 다릅니다: "
            f"actual={manifest_snapshot.sha256}, proven={expected_manifest_sha}"
        )
    assert manifest_snapshot.data is not None
    entries = read_manifest_bytes(
        manifest_snapshot.data, manifest_path=actual_path
    )
    expected_rows = lineage.get("regrouped_row_count")
    if expected_rows != len(entries):
        raise ValueError(
            "canonical lineage regrouped row count가 실제 manifest와 다릅니다: "
            f"actual={len(entries)}, proven={expected_rows}"
        )
    return entries, manifest_snapshot.sha256, holdout_summary


def _npz_scalar(data: Any, key: str) -> Any:
    if key not in data:
        raise ValueError(f"필수 메타데이터 누락: {key}")
    value = data[key]
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{key}는 scalar여야 합니다: shape={array.shape}")
    return array.reshape(-1)[0].item()


def _fixed_point_clock_valid_mask(
    *,
    base_valid: np.ndarray,
    common_delay_samples: np.ndarray,
    adjacent_change_samples: np.ndarray,
    max_drift_deviation_samples: float,
    max_adjacent_change_samples: float,
    min_valid_periods: int,
) -> tuple[np.ndarray, float]:
    """Independently reconstruct the official final-median clock mask.

    This deliberately mirrors, rather than trusts, the measurement producer.
    The mask is monotonically reduced until every survivor is within the hard
    envelope around the median of those same survivors.
    """

    valid = np.asarray(base_valid, dtype=np.bool_).reshape(-1).copy()
    common = np.asarray(common_delay_samples, dtype=np.float64).reshape(-1)
    adjacent = np.asarray(adjacent_change_samples, dtype=np.float64).reshape(-1)
    if common.size != valid.size or adjacent.size != valid.size:
        raise ValueError("source clock witness 배열 길이가 다릅니다")
    minimum = int(min_valid_periods)
    if minimum <= 0 or int(valid.sum()) < minimum:
        raise ValueError("source analysis base clock-valid 반복이 8개 미만입니다")

    initial_median = float(np.median(common[valid]))
    valid &= (
        np.abs(common - initial_median) <= float(max_drift_deviation_samples)
    )
    valid[1:] &= (
        ~np.isfinite(adjacent[1:])
        | (adjacent[1:] <= float(max_adjacent_change_samples))
    )
    while True:
        if int(valid.sum()) < minimum:
            raise ValueError(
                "source analysis final-median fixed-point clock-valid 반복이 8개 미만입니다"
            )
        median = float(np.median(common[valid]))
        updated = valid & (
            np.abs(common - median) <= float(max_drift_deviation_samples)
        )
        if np.array_equal(updated, valid):
            return valid, median
        valid = updated


def _aligned_transfer_sha256(
    frequencies_hz: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
) -> str:
    frequencies = np.ascontiguousarray(
        np.asarray(frequencies_hz, dtype="<f8").reshape(-1)
    )
    real_values = np.ascontiguousarray(np.asarray(real, dtype="<f8").reshape(-1))
    imag_values = np.ascontiguousarray(np.asarray(imag, dtype="<f8").reshape(-1))
    if not (frequencies.size == real_values.size == imag_values.size) or not frequencies.size:
        raise ValueError("aligned mean transfer source 배열 길이가 맞지 않습니다")
    digest = hashlib.sha256()
    digest.update(frequencies.tobytes(order="C"))
    digest.update(real_values.tobytes(order="C"))
    digest.update(imag_values.tobytes(order="C"))
    return digest.hexdigest()


def _compact_transfer_metrics(
    frequencies_hz: np.ndarray,
    measured_transfer: np.ndarray,
    fir: np.ndarray,
    *,
    delay_samples: int,
    sample_rate: int,
    band_hz: tuple[float, float],
) -> dict[str, Any]:
    """NPZ의 source complex transfer를 FIR+effective delay에서 독립 재계산한다."""

    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    measured = np.asarray(measured_transfer, dtype=np.complex128).reshape(-1)
    taps = np.asarray(fir, dtype=np.float64).reshape(-1)
    if frequencies.size != measured.size or frequencies.size == 0:
        raise ValueError("compact transfer source 길이가 맞지 않습니다")
    if delay_samples < 0 or sample_rate <= 0 or taps.size == 0:
        raise ValueError("compact transfer FIR/delay/sample_rate가 유효하지 않습니다")
    if not (
        np.all(np.isfinite(frequencies))
        and np.all(np.isfinite(measured))
        and np.all(np.isfinite(taps))
    ):
        raise ValueError("compact transfer source/FIR에 NaN/Inf가 있습니다")
    indices = np.arange(taps.size, dtype=np.float64) + float(delay_samples)

    def metrics(bounds: tuple[float, float]) -> dict[str, Any]:
        mask = (frequencies >= float(bounds[0])) & (frequencies <= float(bounds[1]))
        if int(mask.sum()) < 8:
            raise ValueError(
                f"compact transfer {bounds[0]:.0f}-{bounds[1]:.0f}Hz 톤이 "
                f"{int(mask.sum())}개뿐입니다"
            )
        reconstructed = np.exp(
            -2j * np.pi * np.outer(frequencies[mask], indices) / float(sample_rate)
        ) @ taps
        target = measured[mask]
        target_norm = float(np.linalg.norm(target))
        model_norm = float(np.linalg.norm(reconstructed))
        if target_norm <= 0.0 or model_norm <= 0.0:
            raise ValueError("compact transfer 대역 energy가 0입니다")
        return {
            "band_hz": [float(bounds[0]), float(bounds[1])],
            "tone_count": int(mask.sum()),
            "complex_agreement": float(
                abs(complex(np.vdot(target, reconstructed)))
                / (target_norm * model_norm)
            ),
            "relative_error": float(np.linalg.norm(reconstructed - target) / target_norm),
        }

    return {
        "overall": metrics((float(band_hz[0]), float(band_hz[1]))),
        "subbands": [metrics(tuple(row)) for row in INTERLEAVED_COMPACT_TRANSFER_SUB_BANDS_HZ],
    }


def _strict_source_path_and_sha256(
    data: Any, *, path_key: str, sha_key: str
) -> tuple[Path, str]:
    stored_path = str(_npz_scalar(data, path_key))
    stored_sha = str(_npz_scalar(data, sha_key))
    if not stored_path:
        raise ValueError(f"{path_key}가 비었습니다")
    if len(stored_sha) != 64 or stored_sha != stored_sha.lower() or any(
        character not in "0123456789abcdef" for character in stored_sha
    ):
        raise ValueError(f"{sha_key}가 canonical SHA256가 아닙니다")
    relative = Path(stored_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{path_key}는 repo-relative results 경로여야 합니다")
    resolved = (REPO_ROOT / relative).resolve()
    results_root = (REPO_ROOT / "results").resolve()
    try:
        resolved.relative_to(results_root)
    except ValueError as exc:
        raise ValueError(f"{path_key}가 REPO_ROOT/results 밖을 가리킵니다") from exc
    if path_key == "source_raw_npz_path" and resolved.name != "raw_measurement.npz":
        raise ValueError("source_raw_npz_path basename은 'raw_measurement.npz'여야 합니다")
    if path_key == "source_analysis_npz_path" and not (
        resolved.name == "analysis_results.npz"
        or re.fullmatch(
            r"analysis_results\.reanalysis_[0-9]{8}T[0-9]{12}Z_[0-9a-f]{8}\.npz",
            resolved.name,
        )
    ):
        raise ValueError(
            "source_analysis_npz_path basename은 canonical/versioned analysis_results여야 합니다"
        )
    if not resolved.is_file():
        raise ValueError(f"{path_key} 원본이 없습니다: {resolved}")
    actual_sha = sha256_file(resolved)
    if actual_sha != stored_sha:
        raise ValueError(
            f"{sha_key} 불일치: stored={stored_sha}, actual={actual_sha}"
        )
    return resolved, stored_sha


def _audit_observed_raw_source(
    data: Any,
    *,
    capture_id: str,
    clock_sample_rate: int,
    period_seconds: float,
    alignment_count: int,
    max_drift_deviation_samples: float,
) -> dict[str, Any]:
    """official이 가리키는 immutable raw가 실제 submitted PCM 캡처인지 확인한다."""

    official_channel_map: dict[str, int] = {}
    for field, (logical_name, expected_index) in INTERLEAVED_CHANNEL_MAP_FIELDS.items():
        raw_value = np.asarray(data[field])
        if raw_value.shape != () or raw_value.dtype.kind not in "iu":
            raise ValueError(f"{field}는 scalar integer여야 합니다")
        value = int(raw_value.item())
        if value != expected_index:
            raise ValueError(f"{field}={value}; official index {expected_index}여야 합니다")
        official_channel_map[logical_name] = value
    for field in INTERLEAVED_OPERATOR_CONFIRMATION_FIELDS:
        raw_value = np.asarray(data[field])
        if raw_value.shape != () or raw_value.dtype.kind != "b" or not bool(
            raw_value.item()
        ):
            raise ValueError(f"{field}는 scalar bool true여야 합니다")

    raw_path, raw_sha = _strict_source_path_and_sha256(
        data,
        path_key="source_raw_npz_path",
        sha_key="source_raw_npz_sha256",
    )
    with np.load(raw_path, allow_pickle=False) as raw:
        required = {
            "metadata_json",
            "output",
            "output_pcm_int16",
            "input_raw_int32",
            "preflight_raw_int32",
        }
        missing = sorted(required.difference(raw.files))
        if missing:
            raise ValueError(
                "source raw observed PCM 필드가 없습니다: " + ", ".join(missing)
            )
        try:
            metadata = json.loads(str(_npz_scalar(raw, "metadata_json")))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"source raw metadata_json이 유효하지 않습니다: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError("source raw metadata_json은 object여야 합니다")
        if metadata.get("method") != "interleaved_multitone":
            raise ValueError("source raw method가 interleaved_multitone이 아닙니다")
        if metadata.get("raw_capture_schema") != INTERLEAVED_RAW_CAPTURE_SCHEMA:
            raise ValueError(
                f"source raw schema={metadata.get('raw_capture_schema')!r}; "
                f"{INTERLEAVED_RAW_CAPTURE_SCHEMA!r} 이어야 합니다"
            )
        if metadata.get("capture_id") != capture_id:
            raise ValueError(
                f"source raw capture_id={metadata.get('capture_id')!r} != {capture_id!r}"
            )
        if metadata.get("channel_map") != official_channel_map:
            raise ValueError(
                f"source raw channel_map={metadata.get('channel_map')!r} != "
                f"official {official_channel_map!r}"
            )
        if metadata.get("operator_confirmations") != {
            "user_present": True,
            "volume_minimum": True,
            "routing_and_geometry": True,
        }:
            raise ValueError("source raw operator confirmations가 모두 true가 아닙니다")
        expected_metadata = {
            "sample_rate": int(clock_sample_rate),
            "block_size": int(_npz_scalar(data, "calibration_block_size")),
            "latency": str(_npz_scalar(data, "calibration_latency")),
            "amplitude": float(_npz_scalar(data, "amplitude")),
            "period_seconds": float(period_seconds),
            "repeats": int(alignment_count),
            "guard_bins": int(_npz_scalar(data, "interleave_guard_bins")),
        }
        for key, expected in expected_metadata.items():
            value = metadata.get(key)
            equal = (
                math.isclose(float(value), expected, rel_tol=0.0, abs_tol=0.0)
                if isinstance(expected, float)
                else value == expected
            )
            if not equal:
                official_label = (
                    "interleave_guard_bins" if key == "guard_bins" else key
                )
                raise ValueError(
                    f"source raw {key}={value!r} != official "
                    f"{official_label}={expected!r}"
                )
        invalid_reasons = metadata.get("invalid_reasons")
        if not isinstance(invalid_reasons, list) or invalid_reasons:
            raise ValueError(f"source raw invalid_reasons={invalid_reasons!r}")
        warp = metadata.get("warp")
        if not isinstance(warp, dict) or bool(warp.get("applied")):
            raise ValueError("source raw dewarp/warp 캡처는 official provenance가 아닙니다")
        telemetry = metadata.get("telemetry")
        if not isinstance(telemetry, dict):
            raise ValueError("source raw telemetry가 없습니다")
        if (
            not bool(telemetry.get("completed"))
            or int(telemetry.get("xrun_count", -1)) != 0
            or int(telemetry.get("unexpected_status_count", -1)) != 0
            or telemetry.get("callback_error") not in (None, "")
        ):
            raise ValueError(f"source raw telemetry 결함: {telemetry!r}")
        contract = metadata.get("analysis_contract")
        if not isinstance(contract, dict):
            raise ValueError("source raw analysis_contract가 없습니다")
        fixed_contract = {
            "clock_band_hz": list(INTERLEAVED_CLOCK_BAND_HZ),
            "clock_min_adjacent_score": INTERLEAVED_CLOCK_MIN_SCORE,
            "clock_max_err_ref_delta_samples": INTERLEAVED_CLOCK_MAX_ERR_REF_DELTA,
            "clock_max_subwindow_spread_samples": INTERLEAVED_CLOCK_MAX_SUBWINDOW_SPREAD,
            "clock_max_adjacent_change_samples": INTERLEAVED_CLOCK_MAX_ADJACENT_CHANGE,
            "clock_max_abs_period_delta_samples": INTERLEAVED_CLOCK_MAX_ABS_PERIOD_DELTA,
            "separation_algorithm": INTERLEAVED_SEPARATION_ALGORITHM,
            "separation_algorithm_version": INTERLEAVED_SEPARATION_ALGORITHM_VERSION,
            "max_drift_deviation_samples": float(max_drift_deviation_samples),
        }
        for key, expected in fixed_contract.items():
            if contract.get(key) != expected:
                raise ValueError(
                    f"source raw analysis_contract {key}={contract.get(key)!r} "
                    f"!= {expected!r}"
                )

        output = np.asarray(raw["output"])
        output_pcm = np.asarray(raw["output_pcm_int16"])
        input_raw = np.asarray(raw["input_raw_int32"])
        preflight_raw = np.asarray(raw["preflight_raw_int32"])
        if output.dtype != np.float32 or output.ndim != 2 or output.shape[1] != 2:
            raise ValueError("source raw output은 [frames,2] float32여야 합니다")
        if output_pcm.dtype != np.int16 or output_pcm.shape != output.shape:
            raise ValueError("source raw output_pcm_int16 dtype/shape가 유효하지 않습니다")
        if input_raw.dtype != np.int32 or input_raw.shape != output.shape:
            raise ValueError("source raw input_raw_int32 dtype/shape가 유효하지 않습니다")
        if (
            preflight_raw.dtype != np.int32
            or preflight_raw.ndim != 2
            or preflight_raw.shape[1] != 2
            or preflight_raw.shape[0] < 1
        ):
            raise ValueError("source raw preflight_raw_int32 dtype/shape가 유효하지 않습니다")
        if not np.all(np.isfinite(output)):
            raise ValueError("source raw output에 NaN/Inf가 있습니다")
        design_band = metadata.get("design_band_hz")
        if not isinstance(design_band, list) or len(design_band) != 2:
            raise ValueError("source raw design_band_hz가 없습니다")
        probe = build_interleaved_probe(
            sample_rate=clock_sample_rate,
            period_seconds=period_seconds,
            band_hz=(float(design_band[0]), float(design_band[1])),
            amplitude=float(expected_metadata["amplitude"]),
            tone_spacing_hz=None,
        )
        if probe.guard_bins() != int(expected_metadata["guard_bins"]):
            raise ValueError("source raw reconstructed probe guard_bins가 다릅니다")
        crest = probe.crest_db()
        stored_crest = metadata.get("crest_db")
        if not isinstance(stored_crest, dict) or any(
            not math.isclose(
                float(got), float(stored_crest.get(name, float("nan"))),
                rel_tol=0.0, abs_tol=1e-6,
            )
            for got, name in zip(crest, ("noise", "cancel"))
        ):
            raise ValueError("source raw reconstructed probe crest_db가 다릅니다")
        reconstructed_channel_band = {
            drive: [
                float(value)
                for value in (
                    probe.bins_for(drive)[[0, -1]]
                    * clock_sample_rate
                    / probe.period_samples
                )
            ]
            for drive in ("noise", "cancel")
        }
        if metadata.get("channel_band_hz") != reconstructed_channel_band:
            raise ValueError("source raw reconstructed channel_band_hz가 다릅니다")
        lead_in = int(metadata.get("lead_in_samples", -1))
        warmup = int(metadata.get("warmup_periods", -1))
        raw_repeats = int(metadata.get("repeats", -1))
        if lead_in < 0 or warmup < 0 or raw_repeats != alignment_count:
            raise ValueError("source raw lead/warmup/repeat 계약이 유효하지 않습니다")
        expected_playback = np.zeros(
            (
                lead_in
                + (warmup + raw_repeats) * probe.period_samples,
                2,
            ),
            dtype=np.float32,
        )
        expected_playback[lead_in:, official_channel_map["noise_out"]] = np.tile(
            probe.noise_signal, warmup + raw_repeats
        )
        expected_playback[lead_in:, official_channel_map["cancel_out"]] = np.tile(
            probe.cancel_signal, warmup + raw_repeats
        )
        if not np.array_equal(output, expected_playback):
            raise ValueError("source raw ideal output이 metadata probe 재구성과 다릅니다")
        expected_pcm = np.rint(
            np.clip(output.astype(np.float32), -1.0, 1.0)
            * np.float32(np.iinfo(np.int16).max)
        ).astype(np.int16)
        if not np.array_equal(output_pcm, expected_pcm):
            raise ValueError(
                "source raw observed output_pcm_int16이 ideal playback 양자화와 다릅니다"
            )
        captured_frames = int(telemetry.get("captured_frames", -1))
        if captured_frames != output.shape[0]:
            raise ValueError(
                f"source raw captured_frames={captured_frames} != {output.shape[0]}"
            )
        expected_frames = lead_in + (warmup + raw_repeats) * int(
            round(period_seconds * clock_sample_rate)
        )
        if lead_in < 0 or warmup < 0 or expected_frames != output.shape[0]:
            raise ValueError(
                f"source raw frame 계약 {output.shape[0]} != expected {expected_frames}"
            )
        recomputed_measurement = analyze_int32_input_probe(input_raw)
        recomputed_preflight = analyze_int32_input_probe(preflight_raw)
        if "measurement" in metadata and json.dumps(
            recomputed_measurement, sort_keys=True
        ) != json.dumps(metadata["measurement"], sort_keys=True):
            raise ValueError("source raw measurement report가 input_raw 재계산과 다릅니다")
        stored_preflight = metadata.get("preflight")
        if not isinstance(stored_preflight, dict):
            raise ValueError("source raw preflight report가 없습니다")
        for key in ("frames", "channels"):
            if json.dumps(recomputed_preflight.get(key), sort_keys=True) != json.dumps(
                stored_preflight.get(key), sort_keys=True
            ):
                raise ValueError(
                    f"source raw preflight {key}가 preflight_raw 재계산과 다릅니다"
                )
        if int(stored_preflight.get("sample_rate", -1)) != clock_sample_rate:
            raise ValueError("source raw preflight sample_rate가 clock rate와 다릅니다")
        for label, report in (
            ("measurement", recomputed_measurement),
            ("preflight", recomputed_preflight),
        ):
            channels = report.get("channels", [])
            input_indices = (
                official_channel_map["error_mic"],
                official_channel_map["reference_mic"],
            )
            if len(channels) < 2 or not all(
                bool(channels[index].get("valid")) for index in input_indices
            ):
                raise ValueError(f"source raw {label} ERR/REF channel이 유효하지 않습니다")
    return {
        "path": str(raw_path),
        "sha256": raw_sha,
        "frames": int(output.shape[0]),
        "schema": INTERLEAVED_RAW_CAPTURE_SCHEMA,
        "channel_map": official_channel_map,
        "operator_confirmations": {
            "user_present": True,
            "volume_minimum": True,
            "routing_and_geometry": True,
        },
        "tone_frequencies_hz": {
            drive: (
                probe.bins_for(drive).astype(np.float64)
                * clock_sample_rate
                / probe.period_samples
            ).tolist()
            for drive in ("noise", "cancel")
        },
    }


def _complex_pair_metrics(
    frequencies_hz: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    band_hz: tuple[float, float],
) -> tuple[float, float]:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    lhs = np.asarray(left, dtype=np.complex128)
    rhs = np.asarray(right, dtype=np.complex128)
    if lhs.ndim != 2 or rhs.shape != lhs.shape or lhs.shape[1] != frequencies.size:
        raise ValueError("separation source transfer shape가 맞지 않습니다")
    mask = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    if int(mask.sum()) < 4:
        raise ValueError(f"separation source {band_hz} 톤이 부족합니다")
    lhs_flat = lhs[:, mask].reshape(-1)
    rhs_flat = rhs[:, mask].reshape(-1)
    if not np.all(np.isfinite(lhs_flat)) or not np.all(np.isfinite(rhs_flat)):
        raise ValueError("separation source transfer에 NaN/Inf가 있습니다")
    lhs_norm = float(np.linalg.norm(lhs_flat))
    rhs_norm = float(np.linalg.norm(rhs_flat))
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        raise ValueError("separation source transfer energy가 0입니다")
    agreement = abs(complex(np.vdot(lhs_flat, rhs_flat))) / (lhs_norm * rhs_norm)
    relative_error = float(np.linalg.norm(rhs_flat - lhs_flat)) / lhs_norm
    return float(agreement), float(relative_error)


def _audit_interleaved_separation(
    data: Any,
    *,
    artifact_rate: int,
    period_seconds: float,
    repeats: int,
    kept_indices: np.ndarray,
    alignment_count: int,
    output_channel: str,
    capture_id: str,
) -> dict[str, Any]:
    """guard=1 official의 clock-corrected demix 증거를 source arrays에서 재감사한다."""

    if int(artifact_rate) != INTERLEAVED_OFFICIAL_SAMPLE_RATE:
        raise ValueError(
            f"interleaved sample_rate={artifact_rate}; "
            f"{INTERLEAVED_OFFICIAL_SAMPLE_RATE}여야 합니다"
        )

    if not math.isfinite(period_seconds) or not (
        0.0 < period_seconds <= INTERLEAVED_MAX_PERIOD_SECONDS
    ):
        raise ValueError(
            f"analysis_period_seconds={period_seconds!r}; "
            f"(0, {INTERLEAVED_MAX_PERIOD_SECONDS}] 이어야 합니다"
        )

    if str(_npz_scalar(data, "output_pcm_provenance")) != INTERLEAVED_OUTPUT_PCM_PROVENANCE:
        raise ValueError("output_pcm_provenance가 observed submitted int16이 아닙니다")
    if str(_npz_scalar(data, "separation_algorithm")) != INTERLEAVED_SEPARATION_ALGORITHM:
        raise ValueError("separation_algorithm이 공식 fractional joint-LS가 아닙니다")
    if int(_npz_scalar(data, "separation_algorithm_version")) != (
        INTERLEAVED_SEPARATION_ALGORITHM_VERSION
    ):
        raise ValueError("separation_algorithm_version이 공식 버전과 다릅니다")
    if str(_npz_scalar(data, "clock_estimator")) != INTERLEAVED_CLOCK_ESTIMATOR:
        raise ValueError("clock_estimator가 독립 time-domain estimator가 아닙니다")
    clock_sample_rate = int(_npz_scalar(data, "clock_sample_rate"))
    if clock_sample_rate <= 0:
        raise ValueError("clock_sample_rate가 유효하지 않습니다")
    if clock_sample_rate != int(artifact_rate):
        raise ValueError(
            f"clock_sample_rate={clock_sample_rate} != artifact sample_rate={artifact_rate}"
        )
    max_drift_deviation = float(
        _npz_scalar(data, "clock_max_drift_deviation_samples")
    )
    if not 0.0 < max_drift_deviation <= 2.0:
        raise ValueError(
            "clock_max_drift_deviation_samples는 (0,2] hard envelope 안이어야 합니다"
        )

    fixed_scalars = {
        "clock_min_adjacent_score": INTERLEAVED_CLOCK_MIN_SCORE,
        "clock_max_err_ref_delta_samples": INTERLEAVED_CLOCK_MAX_ERR_REF_DELTA,
        "clock_max_subwindow_spread_samples": INTERLEAVED_CLOCK_MAX_SUBWINDOW_SPREAD,
        "clock_max_adjacent_change_samples": INTERLEAVED_CLOCK_MAX_ADJACENT_CHANGE,
        "clock_max_abs_period_delta_samples": INTERLEAVED_CLOCK_MAX_ABS_PERIOD_DELTA,
        "joint_ls_max_condition": INTERLEAVED_JOINT_LS_MAX_CONDITION,
        "joint_ls_max_reconstruction_relative_error_p95": (
            INTERLEAVED_JOINT_LS_MAX_RESIDUAL_P95
        ),
        "minimum_separation_crosscheck_agreement": INTERLEAVED_SEPARATION_MIN_AGREEMENT,
        "maximum_separation_crosscheck_relative_error": (
            INTERLEAVED_SEPARATION_MAX_RELATIVE_ERROR
        ),
    }
    for key, expected in fixed_scalars.items():
        value = float(_npz_scalar(data, key))
        if not math.isfinite(value) or value != expected:
            raise ValueError(f"{key}={value!r}; 공식 고정값 {expected!r}와 다릅니다")
    clock_band = np.asarray(data["clock_band_hz"], dtype=np.float64).reshape(-1)
    if not np.array_equal(clock_band, np.asarray(INTERLEAVED_CLOCK_BAND_HZ)):
        raise ValueError("clock_band_hz는 고정 150-1600Hz여야 합니다")

    index_raw = np.asarray(data["clock_observation_repeat_indices"])
    indices = np.asarray(index_raw, dtype=np.int64).reshape(-1)
    names = (
        "clock_period_delta_samples",
        "clock_q_ratio",
        "clock_err_delay_samples",
        "clock_ref_delay_samples",
        "clock_err_score",
        "clock_ref_score",
        "clock_err_subwindow_spread_samples",
        "clock_ref_subwindow_spread_samples",
        "clock_err_ref_delta_samples",
        "joint_ls_rank",
        "joint_ls_condition",
        "joint_ls_reconstruction_relative_error",
    )
    arrays = {name: np.asarray(data[name]).reshape(-1) for name in names}
    if index_raw.ndim != 1 or index_raw.dtype.kind not in "iu":
        raise ValueError("clock_observation_repeat_indices는 1-D integer여야 합니다")
    if indices.size < MIN_KEPT_REPEATS or any(
        values.size != indices.size for values in arrays.values()
    ):
        raise ValueError("clock/joint-LS witness 배열 길이가 맞지 않거나 8개 미만입니다")
    if np.any(indices < 0) or np.any(indices >= alignment_count - 1):
        raise ValueError("clock observation index가 원본 반복 범위 밖입니다")
    if indices.size > 1 and not np.all(np.diff(indices) > 0):
        raise ValueError("clock observation indices는 strictly sorted unique여야 합니다")
    if not set(int(v) for v in kept_indices).issubset(set(int(v) for v in indices)):
        raise ValueError("kept_repeat_indices가 q-valid clock observations의 subset이 아닙니다")

    float_names = tuple(name for name in names if name != "joint_ls_rank")
    if any(
        not np.all(np.isfinite(np.asarray(arrays[name], dtype=np.float64)))
        for name in float_names
    ):
        raise ValueError("clock/joint-LS witness에 NaN/Inf가 있습니다")
    period_delta = np.asarray(arrays["clock_period_delta_samples"], dtype=np.float64)
    q_ratio = np.asarray(arrays["clock_q_ratio"], dtype=np.float64)
    err_delay = np.asarray(arrays["clock_err_delay_samples"], dtype=np.float64)
    ref_delay = np.asarray(arrays["clock_ref_delay_samples"], dtype=np.float64)
    err_score = np.asarray(arrays["clock_err_score"], dtype=np.float64)
    ref_score = np.asarray(arrays["clock_ref_score"], dtype=np.float64)
    err_spread = np.asarray(
        arrays["clock_err_subwindow_spread_samples"], dtype=np.float64
    )
    ref_spread = np.asarray(
        arrays["clock_ref_subwindow_spread_samples"], dtype=np.float64
    )
    err_ref_delta = np.asarray(arrays["clock_err_ref_delta_samples"], dtype=np.float64)
    recomputed_delta = np.abs(err_delay - ref_delay)
    if np.any(err_score < INTERLEAVED_CLOCK_MIN_SCORE) or np.any(
        ref_score < INTERLEAVED_CLOCK_MIN_SCORE
    ):
        raise ValueError("clock adjacent correlation score가 공식 하한 미만입니다")
    if np.any(err_score > 1.000001) or np.any(ref_score > 1.000001):
        raise ValueError("clock adjacent correlation score가 정규화 상한 밖입니다")
    if np.any(err_spread < 0.0) or np.any(ref_spread < 0.0) or np.any(
        err_ref_delta < 0.0
    ):
        raise ValueError("clock spread/delta는 음수일 수 없습니다")
    weighted_delay = (err_delay * err_score + ref_delay * ref_score) / (
        err_score + ref_score
    )
    period_samples_float = float(clock_sample_rate) * float(period_seconds)
    period_samples = int(round(period_samples_float))
    if period_samples <= 0 or not math.isclose(
        period_samples_float, float(period_samples), rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("analysis period가 정수 sample 주기가 아닙니다")
    recomputed_q = period_samples / (period_samples + weighted_delay)
    official_tone_frequencies = np.asarray(
        data["tone_frequencies_hz"], dtype=np.float64
    ).reshape(-1)
    official_tone_bins = official_tone_frequencies * float(period_seconds)
    if (
        official_tone_frequencies.size < INTERLEAVED_MIN_TONE_COUNT
        or not np.all(np.isfinite(official_tone_frequencies))
        or not np.all(np.diff(official_tone_frequencies) > 0.0)
        or not np.allclose(
            official_tone_bins,
            np.rint(official_tone_bins),
            rtol=0.0,
            atol=1e-9,
        )
        or not np.all(np.diff(np.rint(official_tone_bins).astype(np.int64)) == 2)
    ):
        raise ValueError(
            "tone_frequencies_hz가 period DFT의 strictly sorted step=2 bins가 아닙니다"
        )
    if not np.allclose(period_delta, weighted_delay, rtol=0.0, atol=1e-12):
        raise ValueError("clock_period_delta_samples가 ERR/REF weighted witness와 다릅니다")
    if not np.allclose(err_ref_delta, recomputed_delta, rtol=0.0, atol=1e-12):
        raise ValueError("clock_err_ref_delta_samples가 직접 계산과 다릅니다")
    if not np.allclose(q_ratio, recomputed_q, rtol=0.0, atol=1e-12):
        raise ValueError("clock_q_ratio가 N/(N+d)와 다릅니다")
    if np.any(err_ref_delta > INTERLEAVED_CLOCK_MAX_ERR_REF_DELTA):
        raise ValueError("clock ERR/REF period delta가 공식 상한 초과입니다")
    if np.any(err_spread > INTERLEAVED_CLOCK_MAX_SUBWINDOW_SPREAD) or np.any(
        ref_spread > INTERLEAVED_CLOCK_MAX_SUBWINDOW_SPREAD
    ):
        raise ValueError("clock subwindow spread가 공식 상한 초과입니다")
    if np.any(np.abs(period_delta) > INTERLEAVED_CLOCK_MAX_ABS_PERIOD_DELTA):
        raise ValueError("clock absolute period delta가 공식 ±6 samples 밖입니다")
    drift_median = float(np.median(period_delta))
    stored_drift = float(_npz_scalar(data, "drift_samples_per_period"))
    if not math.isclose(stored_drift, drift_median, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("drift_samples_per_period가 q-valid median 재계산과 다릅니다")
    if np.any(np.abs(period_delta - drift_median) > max_drift_deviation):
        raise ValueError("q-valid period delta가 drift median hard envelope 밖입니다")
    consecutive = np.diff(indices) == 1
    if np.any(np.abs(np.diff(period_delta))[consecutive] > INTERLEAVED_CLOCK_MAX_ADJACENT_CHANGE):
        raise ValueError("clock adjacent period change가 공식 상한 초과입니다")

    expected_rank = int(_npz_scalar(data, "joint_ls_expected_rank"))
    ranks_raw = np.asarray(data["joint_ls_rank"])
    ranks = np.asarray(ranks_raw, dtype=np.int64).reshape(-1)
    conditions = np.asarray(arrays["joint_ls_condition"], dtype=np.float64)
    residuals = np.asarray(
        arrays["joint_ls_reconstruction_relative_error"], dtype=np.float64
    )
    if ranks_raw.ndim != 1 or ranks_raw.dtype.kind not in "iu":
        raise ValueError("joint_ls_rank는 1-D integer여야 합니다")
    if expected_rank <= 0 or expected_rank % 2 or np.any(ranks != expected_rank):
        raise ValueError("joint LS rank가 신고된 full rank와 다릅니다")
    if np.any(conditions < 1.0):
        raise ValueError("joint LS condition은 1 미만일 수 없습니다")
    if np.any(conditions > INTERLEAVED_JOINT_LS_MAX_CONDITION):
        raise ValueError("joint LS condition이 공식 상한 초과입니다")
    residual_p95 = float(np.percentile(residuals, 95.0))
    if np.any(residuals < 0.0):
        raise ValueError("joint LS reconstruction residual은 음수일 수 없습니다")
    stored_p95 = float(_npz_scalar(data, "joint_ls_reconstruction_relative_error_p95"))
    if not math.isclose(stored_p95, residual_p95, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("joint LS residual p95 저장값이 배열 재계산과 다릅니다")
    if residual_p95 > INTERLEAVED_JOINT_LS_MAX_RESIDUAL_P95:
        raise ValueError("joint LS reconstruction residual p95가 공식 상한 초과입니다")

    cross_band = np.asarray(data["separation_crosscheck_band_hz"], dtype=np.float64).reshape(-1)
    cross_subbands = np.asarray(
        data["separation_crosscheck_subband_hz"], dtype=np.float64
    ).reshape(-1, 2)
    if not np.array_equal(cross_band, np.asarray(INTERLEAVED_CLOCK_BAND_HZ)):
        raise ValueError("separation crosscheck overall band가 150-1600Hz가 아닙니다")
    if not np.array_equal(cross_subbands, INTERLEAVED_COMPACT_TRANSFER_SUB_BANDS_HZ):
        raise ValueError("separation crosscheck canonical 4 subband schema가 다릅니다")
    stored_agreements = np.r_[
        float(_npz_scalar(data, "separation_crosscheck_complex_agreement")),
        np.asarray(data["separation_crosscheck_subband_complex_agreement"], dtype=np.float64).reshape(-1),
    ]
    stored_errors = np.r_[
        float(_npz_scalar(data, "separation_crosscheck_relative_error")),
        np.asarray(data["separation_crosscheck_subband_relative_error"], dtype=np.float64).reshape(-1),
    ]
    if stored_agreements.size != 5 or stored_errors.size != 5:
        raise ValueError("separation crosscheck metric 배열 길이가 다릅니다")
    if not np.all(np.isfinite(stored_agreements)) or not np.all(np.isfinite(stored_errors)):
        raise ValueError("separation crosscheck metric에 NaN/Inf가 있습니다")
    if np.any(stored_agreements < INTERLEAVED_SEPARATION_MIN_AGREEMENT) or np.any(
        stored_errors > INTERLEAVED_SEPARATION_MAX_RELATIVE_ERROR
    ):
        raise ValueError("separation crosscheck가 공식 agreement/error gate를 못 넘습니다")
    if np.any(stored_agreements > 1.000001) or np.any(stored_errors < 0.0):
        raise ValueError("separation crosscheck agreement/error domain이 유효하지 않습니다")

    repeat_tau = np.asarray(data["repeat_tau_samples"], dtype=np.float64).reshape(-1)
    provisional_tau = np.asarray(
        data["provisional_repeat_tau_samples"], dtype=np.float64
    ).reshape(-1)
    common_tau = np.asarray(data["common_alignment_tau_samples"], dtype=np.float64).reshape(-1)
    if any(values.size != repeats for values in (repeat_tau, provisional_tau, common_tau)):
        raise ValueError(
            f"repeats={repeats} != provisional/common repeat tau 길이 "
            f"{(repeat_tau.size, provisional_tau.size, common_tau.size)}"
        )
    if not all(np.all(np.isfinite(values)) for values in (repeat_tau, provisional_tau, common_tau)):
        raise ValueError("provisional/common repeat tau에 NaN/Inf가 있습니다")
    if not np.array_equal(repeat_tau, provisional_tau):
        raise ValueError("repeat_tau_samples는 channel-specific provisional tau여야 합니다")
    stored_relative_tau_max_abs = float(
        _npz_scalar(data, "relative_tau_max_abs_samples")
    )
    if (
        not math.isfinite(stored_relative_tau_max_abs)
        or stored_relative_tau_max_abs < 0.0
        or stored_relative_tau_max_abs > INTERLEAVED_MAX_RELATIVE_TAU_ABS_SAMPLES
    ):
        raise ValueError("relative_tau_max_abs_samples가 official hard bound 밖입니다")

    raw_source = _audit_observed_raw_source(
        data,
        capture_id=capture_id,
        clock_sample_rate=clock_sample_rate,
        period_seconds=period_seconds,
        alignment_count=alignment_count,
        max_drift_deviation_samples=max_drift_deviation,
    )
    if not np.array_equal(
        official_tone_frequencies,
        np.asarray(raw_source["tone_frequencies_hz"][output_channel]),
    ):
        raise ValueError(
            "official tone_frequencies_hz가 raw reconstructed probe grid와 다릅니다"
        )
    analysis_path, analysis_sha = _strict_source_path_and_sha256(
        data,
        path_key="source_analysis_npz_path",
        sha_key="source_analysis_npz_sha256",
    )
    raw_parent = Path(raw_source["path"]).parent.resolve()
    analysis_parent = analysis_path.parent.resolve()
    if analysis_parent != raw_parent:
        raise ValueError("source raw/analysis가 같은 immutable capture session이 아닙니다")
    with np.load(analysis_path, allow_pickle=False) as source:
        source_required = {
            "clock_valid_mask",
            "clock_q_ratio",
            "clock_period_delta_samples",
            "clock_err_delay_samples",
            "clock_ref_delay_samples",
            "clock_err_score",
            "clock_ref_score",
            "clock_err_subwindow_spread_samples",
            "clock_ref_subwindow_spread_samples",
            "clock_err_ref_delta_samples",
            "joint_ls_rank",
            "joint_ls_condition",
            "joint_ls_reconstruction_relative_error",
            "common_alignment_tau_samples",
            "noise_provisional_tau_samples",
            "cancel_provisional_tau_samples",
            f"{output_channel}_frequencies_hz",
            f"{output_channel}_transfers",
            f"{output_channel}_cubic_crosscheck_transfers",
        }
        missing = sorted(source_required.difference(source.files))
        if missing:
            raise ValueError(
                "source analysis 독립 재감사 배열이 없습니다: " + ", ".join(missing)
            )
        source_valid = np.asarray(source["clock_valid_mask"])
        if source_valid.dtype != np.bool_ or source_valid.ndim != 1:
            raise ValueError("source analysis clock_valid_mask는 1-D bool이어야 합니다")
        if source_valid.size != alignment_count:
            raise ValueError(
                "source analysis clock 축 길이가 alignment_scores 원본 반복 수와 다릅니다"
            )
        source_clock_fields = {
            "clock_q_ratio": q_ratio,
            "clock_period_delta_samples": period_delta,
            "clock_err_delay_samples": err_delay,
            "clock_ref_delay_samples": ref_delay,
            "clock_err_score": err_score,
            "clock_ref_score": ref_score,
            "clock_err_subwindow_spread_samples": err_spread,
            "clock_ref_subwindow_spread_samples": ref_spread,
            "clock_err_ref_delta_samples": err_ref_delta,
            "joint_ls_condition": conditions,
            "joint_ls_reconstruction_relative_error": residuals,
        }
        source_full: dict[str, np.ndarray] = {}
        for key, official_values in source_clock_fields.items():
            full = np.asarray(source[key], dtype=np.float64).reshape(-1)
            source_full[key] = full
            if full.size != source_valid.size or not np.array_equal(
                full[indices], official_values
            ):
                raise ValueError(f"official {key}가 source analysis와 다릅니다")

        full_err_delay = source_full["clock_err_delay_samples"]
        full_ref_delay = source_full["clock_ref_delay_samples"]
        full_err_score = source_full["clock_err_score"]
        full_ref_score = source_full["clock_ref_score"]
        full_err_spread = source_full["clock_err_subwindow_spread_samples"]
        full_ref_spread = source_full["clock_ref_subwindow_spread_samples"]
        full_mic_delta = np.abs(full_err_delay - full_ref_delay)
        score_sum = full_err_score + full_ref_score
        full_common = (
            full_err_delay * full_err_score + full_ref_delay * full_ref_score
        ) / np.maximum(score_sum, np.finfo(np.float64).tiny)
        base_valid = (
            np.isfinite(full_common)
            & np.isfinite(full_mic_delta)
            & np.isfinite(full_err_spread)
            & np.isfinite(full_ref_spread)
            & (full_err_score >= INTERLEAVED_CLOCK_MIN_SCORE)
            & (full_ref_score >= INTERLEAVED_CLOCK_MIN_SCORE)
            & (full_err_score <= 1.000001)
            & (full_ref_score <= 1.000001)
            & (full_err_spread >= 0.0)
            & (full_ref_spread >= 0.0)
            & (full_err_spread <= INTERLEAVED_CLOCK_MAX_SUBWINDOW_SPREAD)
            & (full_ref_spread <= INTERLEAVED_CLOCK_MAX_SUBWINDOW_SPREAD)
            & (full_mic_delta <= INTERLEAVED_CLOCK_MAX_ERR_REF_DELTA)
            & (np.abs(full_common) <= INTERLEAVED_CLOCK_MAX_ABS_PERIOD_DELTA)
            & ((period_samples + full_common) > 0.0)
        )
        adjacent_change = np.full(source_valid.size, np.nan, dtype=np.float64)
        adjacent_change[1:-1] = np.abs(np.diff(full_common[:-1]))
        recomputed_valid, _source_drift_median = _fixed_point_clock_valid_mask(
            base_valid=base_valid,
            common_delay_samples=full_common,
            adjacent_change_samples=adjacent_change,
            max_drift_deviation_samples=max_drift_deviation,
            max_adjacent_change_samples=INTERLEAVED_CLOCK_MAX_ADJACENT_CHANGE,
            min_valid_periods=MIN_KEPT_REPEATS,
        )
        if not np.array_equal(source_valid, recomputed_valid):
            raise ValueError(
                "source analysis clock_valid_mask가 raw witness hard rule 재계산과 다릅니다"
            )
        if not np.array_equal(np.flatnonzero(source_valid), indices):
            raise ValueError("official clock observation indices가 source analysis와 다릅니다")
        source_q = source_full["clock_q_ratio"]
        source_q_expected = period_samples / (
            period_samples + full_common[source_valid]
        )
        if np.any(source_q[source_valid] <= 0.0) or not np.allclose(
            source_q[source_valid], source_q_expected, rtol=0.0, atol=1e-12
        ):
            raise ValueError("source analysis clock_q_ratio가 q=N/(N+d)와 다릅니다")
        source_ranks = np.asarray(source["joint_ls_rank"])
        if (
            source_ranks.dtype.kind not in "iu"
            or source_ranks.ndim != 1
            or source_ranks.size != source_valid.size
            or not np.array_equal(source_ranks[indices], ranks)
        ):
            raise ValueError("official joint_ls_rank가 source analysis와 다릅니다")
        source_common = np.asarray(
            source["common_alignment_tau_samples"], dtype=np.float64
        ).reshape(-1)
        source_provisional = np.asarray(
            source[f"{output_channel}_provisional_tau_samples"], dtype=np.float64
        ).reshape(-1)
        if source_common.size != alignment_count or source_provisional.size != alignment_count:
            raise ValueError("source analysis tau 원본 반복 길이가 alignment_scores와 다릅니다")
        if not np.array_equal(source_common[kept_indices], common_tau):
            raise ValueError("official common_alignment_tau_samples가 source analysis와 다릅니다")
        if not np.array_equal(source_provisional[kept_indices], provisional_tau):
            raise ValueError("official provisional_repeat_tau_samples가 source analysis와 다릅니다")

        frequencies = np.asarray(
            source[f"{output_channel}_frequencies_hz"], dtype=np.float64
        ).reshape(-1)
        if not np.array_equal(frequencies, official_tone_frequencies):
            raise ValueError(
                "official tone_frequencies_hz가 source analysis frequency grid와 다릅니다"
            )
        if (
            frequencies.size != int(_npz_scalar(data, "tone_count"))
            or frequencies.size < INTERLEAVED_MIN_TONE_COUNT
            or not np.all(np.isfinite(frequencies))
            or not np.all(np.diff(frequencies) > 0.0)
            or frequencies[0] <= 0.0
            or frequencies[-1] >= artifact_rate / 2.0
        ):
            raise ValueError("source analysis tone frequencies/tone_count가 유효하지 않습니다")
        joint = np.asarray(source[f"{output_channel}_transfers"], dtype=np.complex128)
        cubic = np.asarray(
            source[f"{output_channel}_cubic_crosscheck_transfers"], dtype=np.complex128
        )
        if joint.ndim != 2 or joint.shape[0] != source_valid.size or cubic.shape != joint.shape:
            raise ValueError("source analysis joint/cubic transfer shape가 유효하지 않습니다")
        selected_joint = joint[kept_indices]
        selected_cubic = cubic[kept_indices]
        recomputed_rows = [
            _complex_pair_metrics(
                frequencies, selected_joint, selected_cubic, tuple(bounds)
            )
            for bounds in (
                tuple(INTERLEAVED_CLOCK_BAND_HZ),
                *[tuple(row) for row in INTERLEAVED_COMPACT_TRANSFER_SUB_BANDS_HZ],
            )
        ]
        recomputed_agreements = np.asarray([row[0] for row in recomputed_rows])
        recomputed_errors = np.asarray([row[1] for row in recomputed_rows])
        if not np.allclose(
            stored_agreements, recomputed_agreements, rtol=0.0, atol=1e-9
        ) or not np.allclose(stored_errors, recomputed_errors, rtol=0.0, atol=1e-9):
            raise ValueError("stored separation crosscheck가 source arrays 재계산과 다릅니다")

    return {
        "output_pcm_provenance": INTERLEAVED_OUTPUT_PCM_PROVENANCE,
        "source_raw": raw_source,
        "channel_map": raw_source["channel_map"],
        "operator_confirmations": raw_source["operator_confirmations"],
        "source_analysis": {"path": str(analysis_path), "sha256": analysis_sha},
        "clock_observation_repeat_indices": indices.tolist(),
        "clock_period_delta_samples": period_delta.tolist(),
        "clock_q_ratio": q_ratio.tolist(),
        "clock_sample_rate": clock_sample_rate,
        "clock_max_drift_deviation_samples": max_drift_deviation,
        "drift_samples_per_period": drift_median,
        "clock_err_delay_samples": err_delay.tolist(),
        "clock_ref_delay_samples": ref_delay.tolist(),
        "clock_err_score": err_score.tolist(),
        "clock_ref_score": ref_score.tolist(),
        "clock_err_subwindow_spread_samples": err_spread.tolist(),
        "clock_ref_subwindow_spread_samples": ref_spread.tolist(),
        "clock_err_ref_delta_samples": err_ref_delta.tolist(),
        "joint_ls_expected_rank": expected_rank,
        "joint_ls_rank": ranks.tolist(),
        "joint_ls_condition": conditions.tolist(),
        "joint_ls_reconstruction_relative_error": residuals.tolist(),
        "joint_ls_reconstruction_relative_error_p95": residual_p95,
        "crosscheck_complex_agreement": stored_agreements.tolist(),
        "crosscheck_relative_error": stored_errors.tolist(),
        "provisional_repeat_tau_samples": provisional_tau.tolist(),
        "common_alignment_tau_samples": common_tau.tolist(),
        "relative_tau_max_abs_samples": stored_relative_tau_max_abs,
        "tone_frequencies_hz": official_tone_frequencies.tolist(),
    }


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
            kept_raw = np.asarray(data["kept_repeat_indices"])
            alignment_raw = np.asarray(data["alignment_scores"])
            kept_indices = np.asarray(kept_raw, dtype=np.int64).reshape(-1)
            alignment_scores = np.asarray(
                alignment_raw, dtype=np.float64
            ).reshape(-1)
            anchor_repeat = int(_npz_scalar(data, "anchor_repeat"))
            band_values = np.asarray(
                data["band_consistency"], dtype=np.float64
            ).reshape(-1)
            band_edges = np.asarray(
                data["band_consistency_hz"], dtype=np.float64
            ).reshape(-1, 2)
            reanalysis_params = (
                json.loads(str(_npz_scalar(data, "reanalysis_params_json")))
                if "reanalysis_params_json" in data.files
                else None
            )
            bulk_delay = int(_npz_scalar(data, "bulk_delay_samples"))
            pre_roll = int(_npz_scalar(data, "pre_roll_samples"))
            delay_semantics = str(_npz_scalar(data, "delay_semantics"))
            tone_frequencies = np.asarray(
                data["tone_frequencies_hz"], dtype=np.float64
            ).reshape(-1)
            aligned_real = np.asarray(
                data["aligned_mean_transfer_real"], dtype=np.float64
            ).reshape(-1)
            aligned_imag = np.asarray(
                data["aligned_mean_transfer_imag"], dtype=np.float64
            ).reshape(-1)
            source_digest = str(_npz_scalar(data, "aligned_mean_transfer_sha256"))
            computed_source_digest = _aligned_transfer_sha256(
                tone_frequencies, aligned_real, aligned_imag
            )
            compact_band = np.asarray(
                data["compact_transfer_band_hz"], dtype=np.float64
            ).reshape(-1)
            if compact_band.size != 2:
                raise ValueError("compact_transfer_band_hz는 [lo, hi]여야 합니다")
            compact_recomputed = _compact_transfer_metrics(
                tone_frequencies,
                aligned_real + 1j * aligned_imag,
                fir,
                delay_samples=delay,
                sample_rate=artifact_rate,
                band_hz=(float(compact_band[0]), float(compact_band[1])),
            )
            compact_stored = {
                "tone_count": int(_npz_scalar(data, "compact_transfer_tone_count")),
                "complex_agreement": float(
                    _npz_scalar(data, "compact_transfer_complex_agreement")
                ),
                "relative_error": float(
                    _npz_scalar(data, "compact_transfer_relative_error")
                ),
                "minimum_agreement": float(
                    _npz_scalar(data, "minimum_compact_transfer_agreement")
                ),
                "maximum_relative_error": float(
                    _npz_scalar(data, "maximum_compact_transfer_relative_error")
                ),
                "subband_hz": np.asarray(
                    data["compact_transfer_subband_hz"], dtype=np.float64
                ).reshape(-1, 2),
                "subband_tone_count": np.asarray(
                    data["compact_transfer_subband_tone_count"], dtype=np.int64
                ).reshape(-1),
                "subband_complex_agreement": np.asarray(
                    data["compact_transfer_subband_complex_agreement"], dtype=np.float64
                ).reshape(-1),
                "subband_relative_error": np.asarray(
                    data["compact_transfer_subband_relative_error"], dtype=np.float64
                ).reshape(-1),
            }
            analysis_period_seconds = float(
                _npz_scalar(data, "analysis_period_seconds")
            )
            capture_id = str(_npz_scalar(data, "capture_id"))
            interleaved = {
                "consistency_band_hz": [float(v) for v in consistency_band[:2]],
                "capture_id": capture_id,
                "guard_bins": int(_npz_scalar(data, "interleave_guard_bins")),
                "analysis_period_seconds": analysis_period_seconds,
                "tone_count": int(_npz_scalar(data, "tone_count")),
                "tone_snr_median_db": float(_npz_scalar(data, "tone_snr_median_db")),
                "tone_snr_min_db": float(_npz_scalar(data, "tone_snr_min_db")),
                "anchor_repeat": anchor_repeat,
                "kept_repeat_indices": kept_indices.tolist(),
                "kept_repeat_indices_is_1d": kept_raw.ndim == 1,
                "kept_repeat_indices_is_integer": kept_raw.dtype.kind in "iu",
                "alignment_scores": alignment_scores.tolist(),
                "alignment_scores_is_1d": alignment_raw.ndim == 1,
                "source_tone_count": int(tone_frequencies.size),
                "delay_semantics": delay_semantics,
                "bulk_delay_samples": bulk_delay,
                "pre_roll_samples": pre_roll,
                "aligned_mean_transfer_sha256": source_digest,
                "computed_aligned_mean_transfer_sha256": computed_source_digest,
                "compact_transfer": compact_recomputed,
                "compact_transfer_stored": compact_stored,
            }
            # separation source를 열기 전에 repeat 축 schema를 먼저 확정한다. 그렇지
            # 않으면 malformed kept 배열이 source 대조의 2차 오류로 가려진다.
            if kept_raw.ndim != 1 or kept_raw.dtype.kind not in "iu":
                raise ValueError("kept_repeat_indices는 1-D integer dtype이어야 합니다")
            if alignment_raw.ndim != 1 or alignment_scores.size == 0:
                raise ValueError("alignment_scores는 비어 있지 않은 1-D여야 합니다")
            if kept_indices.size != repeats:
                raise ValueError(
                    f"repeats={repeats} != kept_repeat_indices 길이 {kept_indices.size}"
                )
            if np.any(kept_indices < 0) or np.any(
                kept_indices >= alignment_scores.size
            ):
                raise ValueError("kept_repeat_indices가 alignment_scores 범위 밖입니다")
            if kept_indices.size > 1 and not np.all(np.diff(kept_indices) > 0):
                raise ValueError("kept_repeat_indices는 strictly sorted unique여야 합니다")
            if not 0 <= anchor_repeat < alignment_scores.size:
                raise ValueError("anchor_repeat가 alignment_scores 범위 밖입니다")
            if anchor_repeat not in set(int(value) for value in kept_indices):
                raise ValueError("anchor_repeat가 kept_repeat_indices에 없습니다")
            interleaved["separation"] = _audit_interleaved_separation(
                data,
                artifact_rate=artifact_rate,
                period_seconds=analysis_period_seconds,
                repeats=repeats,
                kept_indices=kept_indices,
                alignment_count=int(alignment_scores.size),
                output_channel=output_channel,
                capture_id=capture_id,
            )

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
        if block_size != INTERLEAVED_OFFICIAL_BLOCK_SIZE:
            errors.append(
                f"interleaved calibration_block_size={block_size}; "
                f"{INTERLEAVED_OFFICIAL_BLOCK_SIZE} 이어야 합니다"
            )
        if latency != INTERLEAVED_OFFICIAL_LATENCY:
            errors.append(
                f"interleaved calibration_latency={latency!r}; "
                f"{INTERLEAVED_OFFICIAL_LATENCY!r} 이어야 합니다"
            )
        if interleaved["delay_semantics"] != INTERLEAVED_DELAY_SEMANTICS:
            errors.append(
                f"delay_semantics={interleaved['delay_semantics']!r}; "
                f"{INTERLEAVED_DELAY_SEMANTICS!r} 이어야 합니다"
            )
        if delay != interleaved["bulk_delay_samples"] - interleaved["pre_roll_samples"]:
            errors.append(
                f"delay 계약 위반: effective={delay}, "
                f"bulk={interleaved['bulk_delay_samples']}, "
                f"pre_roll={interleaved['pre_roll_samples']}"
            )
        if interleaved["pre_roll_samples"] < 0 or interleaved["bulk_delay_samples"] < 0:
            errors.append("bulk_delay_samples/pre_roll_samples가 음수입니다")
        kept = np.asarray(interleaved["kept_repeat_indices"], dtype=np.int64)
        alignment = np.asarray(interleaved["alignment_scores"], dtype=np.float64)
        if not interleaved["kept_repeat_indices_is_1d"]:
            errors.append("kept_repeat_indices는 1-D여야 합니다")
        if not interleaved["kept_repeat_indices_is_integer"]:
            errors.append("kept_repeat_indices는 integer dtype이어야 합니다")
        if not interleaved["alignment_scores_is_1d"] or alignment.size == 0:
            errors.append("alignment_scores는 비어 있지 않은 1-D여야 합니다")
        elif not np.all(np.isfinite(alignment)):
            errors.append("alignment_scores에 NaN/Inf가 있습니다")
        if kept.size != repeats:
            errors.append(
                f"repeats={repeats} != kept_repeat_indices 길이 {kept.size}"
            )
        if kept.size == 0:
            errors.append("kept_repeat_indices가 비었습니다")
        else:
            if np.any(kept < 0) or np.any(kept >= alignment.size):
                errors.append(
                    f"kept_repeat_indices가 alignment_scores 범위 0.."
                    f"{alignment.size - 1} 밖입니다"
                )
            if kept.size > 1 and not np.all(np.diff(kept) > 0):
                errors.append("kept_repeat_indices는 strictly sorted unique여야 합니다")
            if (
                np.all((kept >= 0) & (kept < alignment.size))
                and np.any(alignment[kept] < 0.95)
            ):
                errors.append("kept repeat alignment_score가 official hard 0.95 미만입니다")
        anchor_repeat = int(interleaved["anchor_repeat"])
        if not 0 <= anchor_repeat < alignment.size:
            errors.append("anchor_repeat가 alignment_scores 범위 밖입니다")
        elif anchor_repeat not in set(int(v) for v in kept):
            errors.append("anchor_repeat가 kept_repeat_indices에 없습니다")
        if (
            interleaved["aligned_mean_transfer_sha256"]
            != interleaved["computed_aligned_mean_transfer_sha256"]
        ):
            errors.append("aligned mean transfer source SHA256가 배열과 일치하지 않습니다")

        compact = interleaved["compact_transfer"]
        stored = interleaved["compact_transfer_stored"]
        overall = compact["overall"]
        compact_lo, compact_hi = overall["band_hz"]
        if compact_lo > 150.0 or compact_hi < 1600.0:
            errors.append(
                f"compact transfer 대역 {overall['band_hz']}이 150-1600Hz를 덮지 않습니다"
            )
        if stored["minimum_agreement"] != INTERLEAVED_MIN_COMPACT_TRANSFER_AGREEMENT:
            errors.append("artifact compact agreement 임계가 공식 고정값과 다릅니다")
        if stored["maximum_relative_error"] != INTERLEAVED_MAX_COMPACT_TRANSFER_RELATIVE_ERROR:
            errors.append("artifact compact relative-error 임계가 공식 고정값과 다릅니다")
        if stored["tone_count"] != overall["tone_count"]:
            errors.append("저장/recomputed compact tone_count가 다릅니다")
        if not math.isclose(
            stored["complex_agreement"], overall["complex_agreement"],
            rel_tol=0.0, abs_tol=1e-9,
        ):
            errors.append("저장/recomputed compact complex agreement가 다릅니다")
        if not math.isclose(
            stored["relative_error"], overall["relative_error"],
            rel_tol=0.0, abs_tol=1e-9,
        ):
            errors.append("저장/recomputed compact relative error가 다릅니다")
        compact_rows = compact["subbands"]
        expected_edges = INTERLEAVED_COMPACT_TRANSFER_SUB_BANDS_HZ
        stored_lengths = {
            stored["subband_hz"].shape[0],
            stored["subband_tone_count"].size,
            stored["subband_complex_agreement"].size,
            stored["subband_relative_error"].size,
        }
        if stored_lengths != {len(compact_rows)} or not np.array_equal(
            stored["subband_hz"], expected_edges
        ):
            errors.append("compact transfer 부대역 schema가 공식 4개 대역과 다릅니다")
        else:
            for index, recomputed in enumerate(compact_rows):
                agreement = float(recomputed["complex_agreement"])
                relative_error = float(recomputed["relative_error"])
                label = f"{recomputed['band_hz'][0]:.0f}-{recomputed['band_hz'][1]:.0f}Hz"
                if int(stored["subband_tone_count"][index]) != recomputed["tone_count"]:
                    errors.append(f"{label} compact tone_count 저장값 불일치")
                if not math.isclose(
                    float(stored["subband_complex_agreement"][index]), agreement,
                    rel_tol=0.0, abs_tol=1e-9,
                ):
                    errors.append(f"{label} compact agreement 저장값 불일치")
                if not math.isclose(
                    float(stored["subband_relative_error"][index]), relative_error,
                    rel_tol=0.0, abs_tol=1e-9,
                ):
                    errors.append(f"{label} compact relative error 저장값 불일치")
                if agreement < INTERLEAVED_MIN_COMPACT_TRANSFER_AGREEMENT:
                    errors.append(
                        f"{label} compact agreement {agreement:.6f} < "
                        f"{INTERLEAVED_MIN_COMPACT_TRANSFER_AGREEMENT:.6f}"
                    )
                if relative_error > INTERLEAVED_MAX_COMPACT_TRANSFER_RELATIVE_ERROR:
                    errors.append(
                        f"{label} compact relative error {relative_error:.6f} > "
                        f"{INTERLEAVED_MAX_COMPACT_TRANSFER_RELATIVE_ERROR:.6f}"
                    )
        if overall["complex_agreement"] < INTERLEAVED_MIN_COMPACT_TRANSFER_AGREEMENT:
            errors.append("전대역 compact complex agreement가 공식 하한 미만입니다")
        if overall["relative_error"] > INTERLEAVED_MAX_COMPACT_TRANSFER_RELATIVE_ERROR:
            errors.append("전대역 compact relative error가 공식 상한 초과입니다")
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
        if interleaved["tone_count"] != interleaved["source_tone_count"]:
            errors.append(
                f"tone_count={interleaved['tone_count']} != source 배열 "
                f"{interleaved['source_tone_count']}"
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

        # 최악 부대역 게이트 — canonical 4개 대역은 required-band 설정과 무관하게
        # 모두 존재하고 모두 통과해야 한다. 임의의 넓은 대역 하나로 대체할 수 없다.
        if band_values.size != band_edges.shape[0]:
            errors.append(
                f"band_consistency 길이 {band_values.size} != "
                f"band_consistency_hz {band_edges.shape[0]}"
            )
        else:
            for expected in INTERLEAVED_COMPACT_TRANSFER_SUB_BANDS_HZ:
                matches = np.all(
                    np.isclose(band_edges, expected[None, :], rtol=0.0, atol=1e-9),
                    axis=1,
                )
                if int(matches.sum()) != 1:
                    errors.append(
                        f"canonical 부대역 {expected[0]:.0f}-{expected[1]:.0f}Hz가 "
                        "정확히 한 번 존재해야 합니다"
                    )
                    continue
                value = float(band_values[np.flatnonzero(matches)[0]])
                if not math.isfinite(value) or value < MIN_BAND_CONSISTENCY:
                    errors.append(
                        f"canonical 부대역 {expected[0]:.0f}-{expected[1]:.0f}Hz "
                        f"일관성 {value:.4f} < {MIN_BAND_CONSISTENCY}"
                    )
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
        # 반환 report는 JSON 직렬화 가능해야 한다.
        interleaved["compact_transfer_stored"] = {
            **stored,
            "subband_hz": stored["subband_hz"].tolist(),
            "subband_tone_count": stored["subband_tone_count"].tolist(),
            "subband_complex_agreement": stored[
                "subband_complex_agreement"
            ].tolist(),
            "subband_relative_error": stored["subband_relative_error"].tolist(),
        }
    if repeats < MIN_KEPT_REPEATS:
        errors.append(
            f"유지된 반복 {repeats}회 < {MIN_KEPT_REPEATS}회 — 평균화가 부족해 "
            "한 번의 이상치가 플랜트 형상을 지배합니다"
        )
    required_consistency = (
        max(float(min_consistency), MIN_INTERLEAVED_CONSISTENCY)
        if method == "interleaved_multitone"
        else float(min_consistency)
    )
    if not math.isfinite(consistency) or consistency < required_consistency:
        errors.append(
            f"반복 일관성 {consistency!r} < {required_consistency:.3f}"
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


def _decode_checkpoint_state(raw: bytes, path: Path) -> dict:
    try:
        state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
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


def _load_checkpoint_state(path: Path) -> dict:
    return _decode_checkpoint_state(path.read_bytes(), path)


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
        "experiment_role",
        "init_eligible",
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
    require_completion_receipt: bool | None = None,
    expected_completion_step: int | None = None,
    max_best_metric_db: float = 0.0,
    allowed_physics_statuses: tuple[str, ...] = (
        "secondary_surrogate_representation_pretrain",
    ),
    required_experiment_role: str | None = None,
    require_init_eligible: bool = False,
    expected_loss_selection_sha256: str | None = None,
) -> dict[str, Any]:
    """init best와 같은 run의 완료된 last/receipt를 역할별로 검증한다.

    ``require_completed``는 sibling ``last.pt``와 완료 step을 요구한다.
    ``require_completion_receipt``는 그와 별개로 immutable ``completion.json``까지
    요구한다. 값이 ``None``이면 기존 호출 호환을 위해 canonical role checkpoint에서만
    receipt를 요구한다. 공식 호출부는 이 값을 명시한다.

    ``expected_completion_step``이 있으면 schedule 전체가 아니라 그 exact operational
    stop과 best/last의 ``run_until_step``을 함께 요구한다. 따라서 20k loss pilot은
    completion receipt 없이 measured probe init이 될 수 있지만 19,999 step 또는
    ``run_until_step``이 다른 checkpoint는 통과하지 못한다. 값이 없으면 기존처럼
    ``schedule.total_steps`` 전체 완료를 요구한다.

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
    experiment_role = str(saved_cfg.get("experiment_role", ""))
    strict_canonical_init = required_experiment_role is not None
    if require_completion_receipt is not None and not isinstance(
        require_completion_receipt, bool
    ):
        raise TypeError("require_completion_receipt는 bool 또는 None이어야 합니다")
    completion_receipt_required = (
        strict_canonical_init
        if require_completion_receipt is None
        else require_completion_receipt
    )
    if completion_receipt_required and not require_completed:
        raise ValueError(
            "completion receipt를 요구하려면 sibling last.pt 완료 검증도 필요합니다"
        )
    explicit_completion_step: int | None = None
    if expected_completion_step is not None:
        if isinstance(expected_completion_step, bool) or not isinstance(
            expected_completion_step, int
        ):
            raise TypeError("expected_completion_step은 양의 정수여야 합니다")
        explicit_completion_step = expected_completion_step
        if explicit_completion_step <= 0:
            raise ValueError("expected_completion_step은 양의 정수여야 합니다")
        if not require_completed:
            raise ValueError(
                "expected_completion_step을 지정하려면 sibling last.pt 검증이 필요합니다"
            )
    if strict_canonical_init:
        validate_embedded_experiment_contract(saved_cfg)
        validate_canonical_training_policy(saved_cfg)
    if (
        required_experiment_role is not None
        and experiment_role != str(required_experiment_role)
    ):
        raise ValueError(
            "init checkpoint experiment_role이 승인된 canonical pretrain이 아닙니다: "
            f"checkpoint={experiment_role!r}, required={required_experiment_role!r}"
        )
    if require_init_eligible and saved_cfg.get("init_eligible") is not True:
        raise ValueError(
            "init checkpoint가 init_eligible=true가 아닙니다 — loss pilot/measured "
            "probe는 공식 fine-tune 초기값으로 사용할 수 없습니다"
        )
    saved_loss_sha = str(saved_cfg.get("loss_selection_sha256", ""))
    if strict_canonical_init:
        recomputed_loss_sha = loss_selection_sha256(saved_cfg.get("loss") or {})
        if saved_loss_sha != recomputed_loss_sha:
            raise ValueError(
                "init checkpoint loss-selection digest가 embedded loss와 다릅니다"
            )
        if (
            expected_loss_selection_sha256 is not None
            and saved_loss_sha != str(expected_loss_selection_sha256)
        ):
            raise ValueError(
                "canonical pretrain과 fine-tune loss selection이 다릅니다: "
                f"pretrain={saved_loss_sha}, "
                f"finetune={expected_loss_selection_sha256}"
            )
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
        if strict_canonical_init:
            validate_embedded_experiment_contract(last_cfg)
        schedule = last_cfg.get("schedule", {}) or {}
        schedule_target = int(schedule.get("total_steps", 0))
        if explicit_completion_step is None:
            completion_target = schedule_target
        else:
            completion_target = explicit_completion_step
            best_run_until = int(saved_cfg.get("run_until_step", -1))
            last_run_until = int(last_cfg.get("run_until_step", -1))
            if (
                schedule_target <= 0
                or completion_target > schedule_target
                or best_run_until != completion_target
                or last_run_until != completion_target
            ):
                raise ValueError(
                    "init checkpoint operational completion 계약이 다릅니다: "
                    f"expected={completion_target}, schedule={schedule_target}, "
                    f"best run_until={best_run_until}, last run_until={last_run_until}"
                )
        completion_step = int(last_state.get("step", -1))
        if completion_target <= 0 or completion_step != completion_target:
            raise ValueError(
                "사전학습이 완료되지 않았습니다: "
                f"last step={completion_step}, target={completion_target}"
            )
        if completion_receipt_required:
            if not strict_canonical_init:
                raise ValueError(
                    "completion receipt 검증에는 required_experiment_role이 필요합니다"
                )
            validate_completion_receipt(
                checkpoint.parent,
                expected_role=str(required_experiment_role),
                expected_init_eligible=True if require_init_eligible else None,
            )

    return {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "step": int(state.get("step", -1)),
        "best_metric_db": best_metric,
        "physics_status": physics_status,
        "experiment_role": experiment_role,
        "init_eligible": saved_cfg.get("init_eligible"),
        "loss_selection_sha256": saved_loss_sha or None,
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
        "completion_receipt": (
            str(checkpoint.parent / "completion.json")
            if require_completed and completion_receipt_required
            else None
        ),
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


def _audit_recorded_subband_coverage(
    audit: "_Audit",
    cfg: dict,
    readiness_cfg: dict,
    *,
    manifest_path: Path | None,
    transfer_snapshot: Any = None,
) -> None:
    """G2d — 학습·val·test target에 strict 고역 판정 에너지가 실제로 있는가."""

    report_dir_value = readiness_cfg.get("recorded_subband_coverage_report_dir")
    if not isinstance(report_dir_value, str) or not report_dir_value.strip():
        audit.fail(
            "recorded_subband_coverage",
            "readiness.recorded_subband_coverage_report_dir가 없습니다",
        )
        return
    if manifest_path is None:
        audit.fail(
            "recorded_subband_coverage",
            "recorded manifest를 검증하지 못해 부대역 coverage report를 결속할 수 없습니다",
        )
        return
    try:
        minimum = int(
            readiness_cfg.get(
                "min_groups_per_family_per_split", _min_groups_per_family_default()
            )
        )
        manifest_snapshot = snapshot_regular_file(manifest_path)
        coverage_contract = build_recorded_subband_coverage_contract(
            manifest_path=manifest_snapshot.path,
            manifest_content=manifest_snapshot.content,
            data_cfg=cfg.get("data", {}) or {},
            model_hop=int((cfg.get("model") or {}).get("hop", 0)),
            splits=CANONICAL_COVERAGE_SPLITS,
            max_segments_per_session=CANONICAL_MAX_SEGMENTS_PER_SESSION,
            edge_trim_seconds=CANONICAL_EDGE_TRIM_SECONDS,
        )
        report_path = recorded_subband_coverage_report_path(
            _repo_path(report_dir_value), coverage_contract
        )
        summary = validate_recorded_subband_coverage_report(
            report_path,
            manifest_path=manifest_path,
            data_cfg=cfg.get("data", {}) or {},
            model_hop=int((cfg.get("model") or {}).get("hop", 0)),
            required_families=_required_families(readiness_cfg),
            configured_min_groups_per_family=minimum,
            splits=CANONICAL_COVERAGE_SPLITS,
            max_segments_per_session=CANONICAL_MAX_SEGMENTS_PER_SESSION,
            edge_trim_seconds=CANONICAL_EDGE_TRIM_SECONDS,
        )
        if str(cfg.get("experiment_role", "")) in _RECORDED_TRANSFER_TRUST_ROLES:
            if transfer_snapshot is None:
                raise ValueError(
                    "canonical/probe coverage는 외부 SHA로 고정한 bootstrap receipt가 "
                    "필요합니다"
                )
            receipt = transfer_snapshot.recorded_subband_coverage_receipt
            receipt_snapshot = transfer_snapshot.recorded_subband_coverage_report
            if not isinstance(receipt, dict) or receipt_snapshot is None:
                raise ValueError(
                    "bootstrap receipt에 recorded subband coverage binding이 없습니다"
                )
            expected_binding = {
                "path": Path(summary["report_path"]).relative_to(REPO_ROOT).as_posix(),
                "sha256": summary["report_sha256"],
                "evidence_sha256": summary["evidence_sha256"],
                "manifest_sha256": summary["manifest_sha256"],
                "training_timing_contract_sha256": coverage_contract[
                    "training_timing_contract_sha256"
                ],
                "coverage_contract_sha256": coverage_contract[
                    "coverage_contract_sha256"
                ],
                "all_requested_splits_pass": summary[
                    "all_requested_splits_pass"
                ],
            }
            if receipt != expected_binding or receipt_snapshot.sha256 != summary["report_sha256"]:
                raise ValueError(
                    "recorded subband coverage report가 외부 bootstrap receipt binding과 다릅니다"
                )
        if not summary["all_requested_splits_pass"]:
            weak = list(summary["weak"])
            examples = ", ".join(
                f"{row['split']}/{row['source_family']}/"
                f"{row['band_hz'][0]:g}-{row['band_hz'][1]:g}Hz="
                f"{row['n_covered_groups']}그룹"
                for row in weak[:8]
            )
            audit.fail(
                "recorded_subband_coverage",
                "strict 150–1600 Hz target coverage가 부족합니다: "
                f"{examples}. 실제 에너지가 없는 대역은 ANC 성능을 학습·판정할 수 "
                "없으므로 해당 family의 독립 원본을 추가 녹음해야 합니다",
                **summary,
            )
            return
        audit.pass_(
            "recorded_subband_coverage",
            "train/val/test 모든 family×strict 부대역에 독립 target 그룹이 충분합니다",
            **summary,
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        audit.fail("recorded_subband_coverage", str(exc))


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
            _assert_clip_record_complete(csv_path, row, clips)
            families.setdefault(family, []).extend(str(item) for item in clips)
    return families


def _assert_clip_record_complete(csv_path: Path, row: dict, clips: list) -> None:
    """재생 기록이 **잘려 있으면 거부한다.**

    누수 판정은 이 열이 유일한 근거다. 잘린 항목은 "재생한 적 없는 것" 이 되어 조용히
    합성 학습셋에 남는다 — 게이트는 그것을 서로소라고 부른다.

    2026-08-07 실측: ``build_recording_sources.py`` 가 ``used[:12]`` 로 잘라
    v1 225개 · v2 31개 = **256 placement 가 어디에도 기록되지 않았다** (절단행 57/160).
    그 결과 실측 **test** 세션이 재생한 원본이 합성 **train** 에 살아 있었다.
    """

    declared = row.get("clip_count")
    if declared in (None, ""):
        return
    try:
        count = int(declared)
    except (TypeError, ValueError):
        raise ValueError(f"{csv_path}: clip_count 를 읽을 수 없습니다: {declared!r}")
    if count > len(clips):
        raise ValueError(
            f"{csv_path}: 재생 기록이 잘려 있습니다 — {row.get('path', '?')} 는 "
            f"clip_count={count} 인데 clips 에는 {len(clips)}개만 있습니다 "
            f"(미기록 {count - len(clips)}). 누수 판정의 근거가 불완전하면 "
            "'서로소' 는 증명이 아니라 무지입니다. sources.csv 를 다시 만드세요"
        )


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
    manifest_dir: Path,
    tags: Iterable[str],
    *,
    validated_entries: dict[str, list[dict]] | None = None,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """합성 학습 스트림이 쓰는 원본 파일과 각 파일의 split 을 모은다."""

    by_tag: dict[str, list[str]] = {}
    splits: dict[str, str] = {}
    for tag in tags:
        path = manifest_dir / f"{tag}.jsonl"
        if not path.is_file():
            continue
        rows: list[str] = []
        entries = (
            validated_entries.get(str(tag), [])
            if validated_entries is not None
            else read_manifest(path)
        )
        for entry in entries:
            value = str(entry.get("path", ""))
            if not value:
                continue
            rows.append(value)
            splits[value] = str(entry.get("split", ""))
        by_tag[str(tag)] = rows
    return by_tag, splits


def _audit_corpus_leak(
    audit: "_Audit",
    readiness_cfg: dict,
    data_cfg: dict,
    entries: list[dict] | None = None,
    transfer_snapshot: Any = None,
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
    tags = sorted(
        str(tag)
        for tag, ratio in (data_cfg.get("source_mix_ratio") or {}).items()
        if float(ratio) > 0.0
    )
    generation: dict[str, Any] | None = None
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
        manifest_dir = _repo_path(manifest_dir_value)
        generation = validate_manifest_generation(
            manifest_dir,
            required_tags=tags,
        )
        synthetic, splits = _synthetic_clip_index(
            manifest_dir,
            tags,
            validated_entries=generation.get("_validated_entries"),
        )
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
        generation_exclusion = generation.get(
            "_validated_recorded_generation_exclusion"
        )
        transferred_generation = (
            getattr(transfer_snapshot, "recorded_generation", None)
            if transfer_snapshot is not None
            else None
        )
        if transferred_generation is not None:
            if not isinstance(generation_exclusion, dict):
                raise ValueError(
                    "schema v2 recorded transfer는 public manifest sidecar에 "
                    "recorded_generation_exclusion을 결속해야 합니다"
                )
            generation_ref = generation_exclusion.get("generation")
            try:
                transferred_path = transferred_generation.path.relative_to(
                    REPO_ROOT
                ).as_posix()
            except ValueError as exc:
                raise ValueError(
                    "transfer recorded generation report가 저장소 밖입니다"
                ) from exc
            expected_ref = {
                "path": transferred_path,
                "sha256": transferred_generation.sha256,
                "size": transferred_generation.size,
            }
            if generation_ref != expected_ref:
                raise ValueError(
                    "public manifest exclusion이 transfer가 검증한 recorded generation과 "
                    f"다릅니다: expected={expected_ref}, actual={generation_ref}"
                )
            generation_summary = getattr(
                transfer_snapshot, "recorded_generation_summary", None
            )
            if not isinstance(generation_summary, dict):
                raise ValueError(
                    "schema v2 transfer에 validated recorded generation summary가 없습니다"
                )
            expected_exclusion = derive_recorded_generation_exclusion(
                generation_summary, repo_root=REPO_ROOT
            )
            if generation_exclusion != expected_exclusion:
                raise ValueError(
                    "public manifest exclusion identity가 transfer generation source plan의 "
                    "source/raw SHA·lineage와 다릅니다"
                )
        generation_overlaps: list[dict[str, Any]] = []
        if isinstance(generation_exclusion, dict):
            generation_overlaps = find_recorded_generation_overlaps(
                generation_exclusion,
                generation.get("_validated_entries") or {},
                repo_root=REPO_ROOT,
            )
            if generation_overlaps:
                raise ValueError(
                    "recorded generation additions와 synthetic 6종 manifest의 "
                    "source/raw SHA·lineage 교집합이 남았습니다: "
                    f"{generation_overlaps[:8]}"
                )
        result = check_corpus_disjoint(recorded, synthetic, synthetic_splits=splits)
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RecordedGenerationExclusionError,
    ) as exc:
        audit.fail("corpus_disjoint", str(exc))
        return
    if result.ok:
        lineage = (generation or {}).get("public_lineage") or {}
        audit.pass_(
            "corpus_disjoint",
            result.detail
            + "; canonical holdout 대비 basename/content SHA/authoritative lineage 교집합 0, "
            "public component split crossing 0",
            **result.measured,
            manifest_build_id=(generation or {}).get("build_id"),
            public_lineage_schema=lineage.get("lineage_schema"),
            public_lineage_component_count=lineage.get(
                "manifest_component_count"
            ),
            public_lineage_membership_sha256=lineage.get(
                "manifest_component_membership_sha256"
            ),
            excluded_by_holdout=lineage.get("excluded_by_tag"),
            recorded_generation_exclusion_sha256=(
                None
                if not isinstance(
                    (generation or {}).get("recorded_generation_exclusion"), dict
                )
                else (generation or {})["recorded_generation_exclusion"].get(
                    "identities_sha256"
                )
            ),
        )
    else:
        audit.fail("corpus_disjoint", result.detail, **result.measured)


def _audit_measured_source_delay(
    audit: "_Audit",
    readiness_cfg: dict,
    primary: dict | None,
    recorded_report: dict | None,
    data_cfg: dict | None = None,
    secondary: dict | None = None,
    duct_cfg: dict | None = None,
    timing_contract: TrainingTimingContract | None = None,
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
    if secondary is None:
        audit.fail(
            "measured_source_delay_agreement",
            "유효한 official S(z)가 없어 PlantDelays/총 선행량을 유도할 수 없습니다",
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
    lead_mode = str(data_cfg.get("recorded_lead_mode", "constant"))
    tolerance = float(readiness_cfg.get("max_measured_delay_mismatch_samples", 64.0))

    if timing_contract is None:
        audit.fail(
            "measured_source_delay_agreement",
            "official TrainingTimingContract가 없어 합성/실측 선행량을 비교할 수 없습니다",
        )
        return
    timing = timing_contract
    lead = int(timing.digital_reference_lead_samples)
    artifact_playback_to_err = float(timing.primary_effective_delay_samples)
    synthetic_advance = float(timing.synthetic_total_advance_samples)

    # ── 실측 세션이 말하는 재생→ERR. 새로 재지 않고 세션 기록의 부기 합이다
    # (source→REF + 정렬 후 잔여→ERR). QA 가 이미 계산해 둔다.
    observed: list[float] = []
    for session in recorded_report["sessions"]:
        value = (session.get("alignment") or {}).get(
            "raw_source_err_delay_median_samples"
        )
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            observed.append(value)
    if len(observed) < minimum_sessions:
        audit.fail(
            "measured_source_delay_agreement",
            f"재생→ERR 실측 표본이 {len(observed)}개로 최소 {minimum_sessions}개에 "
            "미달합니다 — 아티팩트가 실측과 맞는지 확인할 수 없습니다",
            observations=len(observed),
        )
        return
    observed_median = float(np.median(np.asarray(observed, dtype=np.float64)))
    observed_spread = float(np.max(observed) - np.min(observed))

    # ── 검사 1 (물리): 아티팩트가 세션과 같은 덕트를 말하는가.
    plant_gap = abs(artifact_playback_to_err - observed_median)
    if plant_gap > tolerance:
        audit.fail(
            "measured_source_delay_agreement",
            f"official P(z) 가 세션 실측과 다른 덕트를 말합니다: 아티팩트 유도 "
            f"{artifact_playback_to_err:.1f} vs {len(observed)}세션 실측 중앙 "
            f"{observed_median:.1f} — 차이 {plant_gap:.1f} > 허용 {tolerance:.0f} 샘플 "
            f"({plant_gap / 48.0:.2f} ms, 산포 {observed_spread:.1f}). 합성 브랜치는 "
            "이 아티팩트로 d 를 만들고 실측 브랜치는 저 마이크로 d 를 받으므로, "
            "둘이 다르면 어느 쪽으로 맞춰도 나머지 한쪽이 틀립니다. P(z) 재측정 또는 "
            "세션 타임라인 규약을 확정하세요",
            artifact_playback_to_err_samples=artifact_playback_to_err,
            observed_playback_to_err_samples=observed_median,
            observed_spread_samples=observed_spread,
            mismatch_samples=plant_gap,
            observations=len(observed),
        )
        return

    # ── 검사 2 (배선): 실측 브랜치가 실제로 쓰는 총 선행량이 합성과 같은가.
    # ``RecordedANCDataset`` 이 읽는 것과 **같은 설정 키**로 같은 산술을 한다.
    try:
        recorded_advance = timing.recorded_total_advance_samples(
            recorded_delay_samples=residual_median,
            mode=lead_mode,
        )
    except ValueError as exc:
        audit.fail("measured_source_delay_agreement", str(exc))
        return
    branch_gap = abs(synthetic_advance - recorded_advance)
    if branch_gap > tolerance:
        audit.fail(
            "measured_source_delay_agreement",
            "두 학습 브랜치가 모델에게 주는 **총 선행량**이 다릅니다: 합성 "
            f"{synthetic_advance:.1f} (재생→ERR {artifact_playback_to_err:.1f} + lead "
            f"{lead}) vs 실측 {recorded_advance:.1f} — 차이 {branch_gap:.1f} > 허용 "
            f"{tolerance:.0f} 샘플 ({branch_gap / 48.0:.2f} ms; 1600 Hz 에서 "
            f"{branch_gap / 48000.0 * 1600.0:.1f} 주기). 같은 모델이 두 브랜치에서 "
            f"다른 예측 과제를 배웁니다 (현재 recorded_lead_mode={lead_mode!r}). "
            "recorded_lead_mode='timeline'으로 실측 lead를 계약에서 유도해야 하며, "
            "실측 브랜치의 총 선행량은 벌크 지연이 아니라 **P(z) 군지연을 포함한** "
            "재생→ERR 지연에서 와야 합니다",
            synthetic_advance_samples=synthetic_advance,
            recorded_advance_samples=recorded_advance,
            artifact_playback_to_err_samples=artifact_playback_to_err,
            recorded_lead_mode=lead_mode,
            residual_median_samples=residual_median,
            mismatch_samples=branch_gap,
            training_timing_contract=timing.model_dump(),
        )
        return

    audit.pass_(
        "measured_source_delay_agreement",
        f"아티팩트와 실측이 {plant_gap:.1f} 샘플 안에서 같은 덕트를 말하고 "
        f"(유도 {artifact_playback_to_err:.1f} vs 실측 {observed_median:.1f}, "
        f"{len(observed)}세션 산포 {observed_spread:.1f}), 두 브랜치의 총 선행량이 "
        f"{branch_gap:.1f} 샘플 안에서 같습니다 (합성 {synthetic_advance:.1f} vs 실측 "
        f"{recorded_advance:.1f}, 허용 {tolerance:.0f})",
        artifact_playback_to_err_samples=artifact_playback_to_err,
        observed_playback_to_err_samples=observed_median,
        observed_spread_samples=observed_spread,
        plant_mismatch_samples=plant_gap,
        synthetic_advance_samples=synthetic_advance,
        recorded_advance_samples=recorded_advance,
        branch_mismatch_samples=branch_gap,
        residual_median_samples=residual_median,
        residual_spread_samples=residual_spread,
        recorded_lead_mode=lead_mode,
        training_timing_contract=timing.model_dump(),
        sessions=len(observed),
    )


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
    인데, legacy 캡처의 정규방정식 설계 상한은 훨씬 낮았다. 현행 수치는
    ``configs/duct.yaml``의 strict P/S와 lead에서 매번 다시 푼다. 인과성과 FIR 길이를
    포함하지 않은 γ 상한만 믿으면 여전히 낙관적이기 때문이다.
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
        # 실제 인과 FIR 상한은 current P/S에서 다시 풀어야 한다. "선언 안 하면 검사
        # 안 함"은 검사가 없는 것과 같고 γ 상한으로 조용히 폴백하게 된다.
        audit.fail(
            "plant_confidence_ceiling",
            "readiness.measured_design_ceiling_db 가 선언되지 않았습니다 — 선언이 없으면 "
            "구속 상한이 플랜트 일관성(약 27.7 dB)으로 폴백해 물리적으로 불가능한 목표도 "
            "통과합니다. 실측 인과 FIR 상한은 current strict P/S에서 다시 풀어야 합니다. "
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
            # readiness의 계산은 cache hit도 신뢰하지 않는다. 측정 자산을 수정하지
            # 않는 기본 no-write 경로로, 매번 실제 P/S 에서 다시 푼다.
            solved = cached_design_ceiling_db(
                _repo_path(primary["path"]),
                _repo_path(secondary["path"]),
                lead_samples=int(lead_samples),
                band_hz=(band_lo, band_hi),
                sample_rate=float(secondary["sample_rate"]),
            )
            # **옥타브별 최악값이 진짜 구속이다.** 대역평균은 저역의 큰 여유가 중역의
            # 병목을 가릴 수 있다. 절대목표 1의 평가(G4)가 옥타브별이므로 진입
            # 게이트도 current strict P/S를 같은 축에서 다시 풀어 판정한다. 평균이
            # 최악값을 가리는 것이 이 저장소가 반복해서 겪은 실패 형태다.
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


def _audit_absolute_objective_scope(
    audit: "_Audit",
    readiness_cfg: dict,
    data_cfg: dict,
    duct_cfg: dict,
) -> None:
    """절대목표를 **설정으로 낮출 수 없게** 만든다.

    다른 게이트들은 설정끼리 비교한다 — "체크포인트 대역이 학습 대역을 덮는가",
    "선언한 태그에 매니페스트가 있는가". 그래서 목표를 낮추면 목표 달성이 참이 된다.
    2026-08-07 검증에서 ``--set`` 두 개로 14개가 전부 열렸다.

    이 게이트만 **코드 상수**와 비교한다. 실행 인자로 내릴 수 있으면 목표가 아니다.
    """

    gate_id = "absolute_objective_scope"
    problems: list[str] = []
    obj_lo, obj_hi = ABSOLUTE_OBJECTIVE_BAND_HZ

    # ── 절대목표 1: 저·고역 **둘 다**. 목표 대역이 코드가 요구하는 대역을 덮어야 한다.
    bands: dict[str, tuple[float, float] | None] = {
        "readiness.required_path_band_hz": None,
        "duct.acoustics.realistic_target_band_hz": None,
    }
    raw_required = readiness_cfg.get("required_path_band_hz")
    if raw_required is not None:
        bands["readiness.required_path_band_hz"] = (
            float(raw_required[0]),
            float(raw_required[1]),
        )
    raw_target = (duct_cfg.get("acoustics") or {}).get("realistic_target_band_hz")
    if raw_target is not None:
        bands["duct.acoustics.realistic_target_band_hz"] = (
            float(raw_target[0]),
            float(raw_target[1]),
        )
    for name, band in bands.items():
        if band is None:
            continue
        lo, hi = band
        if lo > obj_lo or hi < obj_hi:
            problems.append(
                f"{name}=[{lo:g}, {hi:g}] 가 절대목표 대역 "
                f"[{obj_lo:g}, {obj_hi:g}] 를 덮지 않습니다"
            )

    # ── 절대목표 2: 네 source family **모두**. 혼합비에서 계열을 뺄 수 없다.
    mix = data_cfg.get("source_mix_ratio") or {}
    if isinstance(mix, dict):
        for family in REQUIRED_SOURCE_FAMILIES:
            tags = REQUIRED_SOURCE_FAMILY_MIX_TAGS[family]
            weight = sum(float(mix.get(tag, 0.0) or 0.0) for tag in tags)
            if weight <= 0.0:
                problems.append(
                    f"data.source_mix_ratio 에 {family!r} 계열 비중이 없습니다 "
                    f"(tags={list(tags)}, 합계 {weight:g}) — 절대목표 2 는 "
                    "speech/music/environment/machine 모두를 포함합니다"
                )

    if problems:
        audit.fail(
            gate_id,
            "절대목표가 설정으로 낮춰졌습니다: "
            + " / ".join(problems)
            + ". 목표를 낮춰 게이트를 통과시키는 것은 금지입니다 — 목표는 코드에 "
            "있고(deep_anc.dsp.invariants) 실행 인자로 바꿀 수 없습니다",
            objective_band_hz=[obj_lo, obj_hi],
            required_families=list(REQUIRED_SOURCE_FAMILIES),
            problems=problems,
        )
        return

    audit.pass_(
        gate_id,
        f"목표 대역 [{obj_lo:g}, {obj_hi:g}] Hz 와 필수 계열 "
        f"{list(REQUIRED_SOURCE_FAMILIES)} 가 설정에서 유지됩니다",
        objective_band_hz=[obj_lo, obj_hi],
        required_families=list(REQUIRED_SOURCE_FAMILIES),
    )


def audit_finetune_readiness(cfg: dict, *, full_recorded_qa: bool = True) -> dict:
    """resolved train config의 G1–G3 진입 조건을 한 번에 검사한다."""

    audit = _Audit("finetune_readiness")
    readiness_cfg = cfg.get("readiness", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    duct_cfg = cfg.get("duct", {}) or {}
    recorded_transfer_snapshot: Any = None

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

    if str(cfg.get("experiment_role", "")) in _RECORDED_TRANSFER_TRUST_ROLES:
        try:
            transfer = validate_recorded_training_snapshot(
                data_cfg, repo_root=REPO_ROOT
            )
            recorded_transfer_snapshot = transfer
            audit.pass_(
                "recorded_transfer_snapshot",
                "bootstrap receipt→transfer manifest→recorded bytes snapshot이 정합합니다",
                transfer_manifest_sha256=transfer.transfer_manifest.sha256,
                recorded_aggregate_sha256=transfer.recorded_aggregate_sha256,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            audit.fail("recorded_transfer_snapshot", str(exc))

    _audit_absolute_objective_scope(audit, readiness_cfg, data_cfg, duct_cfg)

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
            left_anchor = primary["interleaved"]["anchor_repeat"]
            right_anchor = secondary["interleaved"]["anchor_repeat"]
            if left_anchor != right_anchor:
                raise ValueError(
                    f"P/S anchor_repeat 불일치: P={left_anchor}, S={right_anchor}"
                )
            left_kept = primary["interleaved"]["kept_repeat_indices"]
            right_kept = secondary["interleaved"]["kept_repeat_indices"]
            if left_kept != right_kept:
                raise ValueError(
                    f"P/S kept_repeat_indices 불일치: P={left_kept}, S={right_kept}"
                )
            left_period = float(primary["interleaved"]["analysis_period_seconds"])
            right_period = float(secondary["interleaved"]["analysis_period_seconds"])
            if left_period != right_period:
                raise ValueError(
                    f"P/S analysis_period_seconds 불일치: P={left_period}, S={right_period}"
                )
            left_separation = primary["interleaved"]["separation"]
            right_separation = secondary["interleaved"]["separation"]
            for source_name in ("source_raw", "source_analysis"):
                left_source = left_separation[source_name]
                right_source = right_separation[source_name]
                if left_source["sha256"] != right_source["sha256"]:
                    raise ValueError(
                        f"P/S {source_name} SHA256 불일치: "
                        f"P={left_source['sha256']}, S={right_source['sha256']}"
                    )
                if Path(left_source["path"]).resolve() != Path(
                    right_source["path"]
                ).resolve():
                    raise ValueError(f"P/S {source_name} path가 다릅니다")
            shared_witness_fields = (
                "channel_map",
                "operator_confirmations",
                "clock_observation_repeat_indices",
                "clock_period_delta_samples",
                "clock_q_ratio",
                "clock_err_delay_samples",
                "clock_ref_delay_samples",
                "clock_err_score",
                "clock_ref_score",
                "clock_err_subwindow_spread_samples",
                "clock_ref_subwindow_spread_samples",
                "clock_err_ref_delta_samples",
                "joint_ls_expected_rank",
                "joint_ls_rank",
                "joint_ls_condition",
                "joint_ls_reconstruction_relative_error",
                "joint_ls_reconstruction_relative_error_p95",
                "clock_sample_rate",
                "clock_max_drift_deviation_samples",
                "drift_samples_per_period",
                "common_alignment_tau_samples",
            )
            for key in shared_witness_fields:
                if left_separation[key] != right_separation[key]:
                    raise ValueError(f"P/S shared separation witness {key} 불일치")
            primary_bins = np.rint(
                np.asarray(
                    left_separation["tone_frequencies_hz"], dtype=np.float64
                )
                * left_period
            ).astype(np.int64)
            secondary_bins = np.rint(
                np.asarray(
                    right_separation["tone_frequencies_hz"], dtype=np.float64
                )
                * right_period
            ).astype(np.int64)
            if np.intersect1d(primary_bins, secondary_bins).size:
                raise ValueError("P/S interleaved tone DFT bins가 서로 겹칩니다")
            union_bins = np.sort(np.r_[primary_bins, secondary_bins])
            if union_bins.size < 2 or not np.all(np.diff(union_bins) == 1):
                raise ValueError("P/S interleaved tone union이 연속 DFT bins가 아닙니다")
            expected_joint_rank = 2 * int(union_bins.size)
            if left_separation["joint_ls_expected_rank"] != expected_joint_rank:
                raise ValueError(
                    f"joint_ls_expected_rank={left_separation['joint_ls_expected_rank']} != "
                    f"2*(P tones+S tones)={expected_joint_rank}"
                )
            kept_array = np.asarray(left_kept, dtype=np.int64)
            primary_scores = np.asarray(
                primary["interleaved"]["alignment_scores"], dtype=np.float64
            )[kept_array]
            secondary_scores = np.asarray(
                secondary["interleaved"]["alignment_scores"], dtype=np.float64
            )[kept_array]
            primary_tau = np.asarray(
                left_separation["provisional_repeat_tau_samples"], dtype=np.float64
            )
            secondary_tau = np.asarray(
                right_separation["provisional_repeat_tau_samples"], dtype=np.float64
            )
            common_tau = np.asarray(
                left_separation["common_alignment_tau_samples"], dtype=np.float64
            )
            recomputed_common = (
                primary_tau * primary_scores + secondary_tau * secondary_scores
            ) / (primary_scores + secondary_scores)
            if not np.allclose(common_tau, recomputed_common, rtol=0.0, atol=1e-12):
                raise ValueError(
                    "P/S common_alignment_tau_samples가 channel score 가중평균과 다릅니다"
                )
            relative_tau = primary_tau - secondary_tau
            relative_centered = relative_tau - float(np.median(relative_tau))
            recomputed_max_abs = float(np.max(np.abs(relative_centered)))
            recomputed_spread = int(
                np.ceil(float(np.ptp(relative_tau)) - 1e-9)
            )
            if recomputed_max_abs > INTERLEAVED_MAX_RELATIVE_TAU_ABS_SAMPLES:
                raise ValueError(
                    f"P/S provisional 상대 tau maxabs {recomputed_max_abs:.6f} > "
                    f"{INTERLEAVED_MAX_RELATIVE_TAU_ABS_SAMPLES:.6f}"
                )
            if recomputed_spread > MAX_RELATIVE_DELAY_SPREAD_SAMPLES:
                raise ValueError(
                    f"P/S provisional 상대 tau spread {recomputed_spread} > "
                    f"{MAX_RELATIVE_DELAY_SPREAD_SAMPLES}"
                )
            for label, artifact, separation in (
                ("P", primary, left_separation),
                ("S", secondary, right_separation),
            ):
                if artifact["delay_spread_samples"] != recomputed_spread:
                    raise ValueError(
                        f"{label} delay_spread_samples 저장값이 provisional tau 재계산과 다릅니다"
                    )
                if not math.isclose(
                    float(separation["relative_tau_max_abs_samples"]),
                    recomputed_max_abs,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"{label} relative_tau_max_abs_samples 저장값이 provisional tau 재계산과 다릅니다"
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

    official_timing: TrainingTimingContract | None = None
    if primary is not None and secondary is not None:
        configured_lead: Any = None
        try:
            primary_artifact = load_secondary_path(primary["path"])
            delays = PlantDelays.from_config(
                duct_cfg=duct_cfg,
                secondary_delay_samples=int(secondary["delay_samples"]),
                primary_delay_samples=int(primary["delay_samples"]),
                sample_rate=int(sample_rate),
            )
            derived_timing = TrainingTimingContract.derive(
                primary_fir=primary_artifact.fir,
                plant_delays=delays,
            )
            configured_lead = data_cfg.get("digital_reference_lead_samples")
            resolved_timing = TrainingTimingContract.from_data_config(data_cfg)
            if resolved_timing != derived_timing:
                raise ValueError(
                    "resolved data.training_timing_contract가 official P/S에서 유도한 "
                    "계약과 다릅니다"
                )
            if configured_lead is not None and int(configured_lead) != int(
                derived_timing.digital_reference_lead_samples
            ):
                raise ValueError(
                    "data.digital_reference_lead_samples가 TrainingTimingContract와 "
                    f"다릅니다: configured={configured_lead}, "
                    f"expected={derived_timing.digital_reference_lead_samples}"
                )
            official_timing = derived_timing
            audit.pass_(
                "path_delay_and_lead",
                "official P/S에서 유도한 TrainingTimingContract가 resolved 계약과 "
                "정확히 같습니다",
                timing_contract=official_timing.model_dump(),
                timing_contract_sha256=official_timing.digest(),
                digital_reference_lead_samples=int(
                    official_timing.digital_reference_lead_samples
                ),
            )
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
            details: dict[str, Any] = {}
            if configured_lead is not None and official_timing is None:
                details = {
                    "configured_lead": int(configured_lead),
                    "expected_lead": int(derived_timing.digital_reference_lead_samples)
                    if "derived_timing" in locals()
                    else None,
                }
            audit.fail(
                "path_delay_and_lead",
                str(exc),
                **details,
            )
    else:
        audit.fail(
            "path_delay_and_lead",
            "유효한 official P/S가 없어 lead를 검증할 수 없습니다",
        )
    expected_lead = (
        int(official_timing.digital_reference_lead_samples)
        if official_timing is not None
        else -1
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
        required_init_role = cfg.get("required_init_experiment_role")
        init = audit_init_checkpoint(
            init_value,
            expected_model_cfg=cfg.get("model", {}),
            expected_lead=expected_lead,
            expected_optimize_band=expected_optimize_band,
            max_lead_mismatch_samples=int(
                readiness_cfg.get("max_init_lead_mismatch_samples", 0)
            ),
            require_completed=bool(
                readiness_cfg.get("require_completed_init_checkpoint", True)
            ),
            require_completion_receipt=(
                bool(cfg.get("require_init_completion_receipt", True))
                if required_init_role is not None
                else False
            ),
            expected_completion_step=(
                CANONICAL_LOSS_PILOT_STEPS
                if str(cfg.get("experiment_role", "")) == "measured_probe"
                else None
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
            required_experiment_role=(
                str(required_init_role)
                if required_init_role is not None
                else None
            ),
            require_init_eligible=bool(cfg.get("require_init_eligible", False)),
            expected_loss_selection_sha256=(
                str(cfg.get("loss_selection_sha256"))
                if cfg.get("loss_selection_sha256") is not None
                else None
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
    manifest_sha256: str | None = None
    lineage_summary: dict[str, Any] | None = None
    recorded_manifest_path: Path | None = None
    try:
        if not manifest_value:
            raise ValueError("recorded_manifest가 비었습니다")
        manifest_path = _repo_path(manifest_value)
        recorded_manifest_path = manifest_path
        if str(cfg.get("experiment_role", "")) in _RECORDED_TRANSFER_TRUST_ROLES:
            entries, manifest_sha256, holdout_summary = (
                _canonical_recorded_lineage_snapshot(manifest_path, data_cfg)
                if recorded_transfer_snapshot is None
                else _canonical_recorded_lineage_snapshot(
                    manifest_path, data_cfg, recorded_transfer_snapshot
                )
            )
            lineage_summary = holdout_summary.get("lineage")
        else:
            manifest_path = manifest_path.resolve()
            entries = read_manifest(manifest_path)
            manifest_sha256 = sha256_file(manifest_path)
        if str(data_cfg.get("recorded_sampling", "uniform_session")) == (
            "family_lineage_session_balanced"
        ):
            invalid_lineage = [
                str(entry.get("session_id") or entry.get("path") or index)
                for index, entry in enumerate(entries)
                if not str(entry.get("source_pool_group_id") or "").strip()
                or str(entry.get("source_pool_group_id") or "").strip()
                == str(entry.get("group_id") or "").strip()
            ]
            if invalid_lineage:
                raise ValueError(
                    "recorded manifest가 lineage-component regroup 계약을 만족하지 "
                    "않습니다(source_pool_group_id + 서로 다른 component group_id "
                    f"필수): {invalid_lineage[:8]}"
                )
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
            manifest_sha256=manifest_sha256,
            lineage_component_membership_sha256=(
                None
                if lineage_summary is None
                else lineage_summary.get("component_membership_sha256")
            ),
            lineage_component_count=(
                None
                if lineage_summary is None
                else lineage_summary.get("component_count")
            ),
            sessions=int(summary.get("sessions", 0)),
            duration_seconds=float(summary.get("duration_s", 0.0)),
            source_families=sorted(observed_families),
            full_recorded_qa=bool(full_recorded_qa),
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        audit.fail("recorded_dataset_qa", str(exc))

    _audit_recorded_alignment(audit, readiness_cfg, recorded_report, full_recorded_qa)
    _audit_statistical_power(audit, readiness_cfg, entries)
    _audit_recorded_subband_coverage(
        audit,
        cfg,
        readiness_cfg,
        manifest_path=recorded_manifest_path,
        transfer_snapshot=recorded_transfer_snapshot,
    )
    _audit_corpus_leak(
        audit,
        readiness_cfg,
        data_cfg,
        entries,
        transfer_snapshot=recorded_transfer_snapshot,
    )
    _audit_measured_source_delay(
        audit,
        readiness_cfg,
        primary,
        recorded_report,
        data_cfg,
        secondary=secondary,
        duct_cfg=duct_cfg,
        timing_contract=official_timing,
    )
    _audit_plant_confidence_ceiling(
        audit,
        readiness_cfg,
        secondary,
        primary,
        lead_samples=(
            int(official_timing.digital_reference_lead_samples)
            if official_timing is not None
            else 0
        ),
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
    manifest_bytes: bytes,
    manifest_path: str | Path,
    checkpoint_sha256: str,
    manifest_sha256: str,
    required_source_families: tuple[str, ...],
    experiment_contract_sha256: str | None,
    selection_sha256: str | None = None,
    test_capability_sha256: str | None = None,
    test_consumed_marker_sha256: str | None = None,
    timing_contract_sha256: str | None = None,
    timing_contract: TrainingTimingContract | None = None,
    checkpoint_cfg: dict[str, Any] | None = None,
    canonical_sampling_binding: bool = True,
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
        if experiment_contract_sha256 is not None:
            required.add("experiment_contract_sha256")
        if timing_contract_sha256 is not None:
            required.update(
                {
                    "segment_recorded_lead_samples",
                    "segment_recorded_delay_samples",
                    "segment_timing_contract_sha256",
                    "segment_source_timeline",
                }
            )
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"G4 provenance/판정 필드 누락: {missing}")
        split = str(_npz_scalar(data, "split"))
        physics_status = str(_npz_scalar(data, "physics_status"))
        allow_surrogate = bool(_npz_scalar(data, "allow_surrogate"))
        saved_checkpoint_sha = str(_npz_scalar(data, "checkpoint_sha256"))
        saved_manifest_sha = str(_npz_scalar(data, "manifest_sha256"))
        saved_contract_sha = (
            str(_npz_scalar(data, "experiment_contract_sha256"))
            if "experiment_contract_sha256" in data.files
            else ""
        )
        saved_selection_sha = (
            str(_npz_scalar(data, "selection_sha256"))
            if "selection_sha256" in data.files
            else ""
        )
        saved_capability_sha = (
            str(_npz_scalar(data, "test_capability_sha256"))
            if "test_capability_sha256" in data.files
            else ""
        )
        saved_consumed_sha = (
            str(_npz_scalar(data, "test_consumed_marker_sha256"))
            if "test_consumed_marker_sha256" in data.files
            else ""
        )
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
        # 선택 경로와 completion이 동일한 authority를 써야 한다. 이 validator는
        # summary scalar/boolean 대신 raw global/family/octave arrays를 재계산하고,
        # selected manifest bytes의 모든 session/family/group와 전단사 결속한다.
        # 여기에서만 strict 부대역을 따로 믿으면 scalar-trust 우회가 다시 생긴다.
        persisted_g4 = validate_persisted_g4_metrics(
            data,
            expected_split=expected_split,
            manifest_bytes=manifest_bytes,
            manifest_path=manifest_path,
            checkpoint_cfg=(checkpoint_cfg if canonical_sampling_binding else None),
            canonical=canonical_sampling_binding,
            min_groups=MIN_GROUPS_PER_FAMILY_PER_SPLIT,
        )
        strict_subband_summary = persisted_g4["strict_subband"]
        if strict_subband_summary is None:  # canonical=True 방어
            raise ValueError("canonical G4 strict subband raw evidence가 없습니다")
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
        timing_shas = (
            np.asarray(data["segment_timing_contract_sha256"]).astype(str).reshape(-1)
            if timing_contract_sha256 is not None
            else np.asarray([], dtype=str)
        )
        source_timelines = (
            np.asarray(data["segment_source_timeline"]).astype(str).reshape(-1)
            if timing_contract_sha256 is not None
            else np.asarray([], dtype=str)
        )
        recorded_leads = (
            np.asarray(data["segment_recorded_lead_samples"], dtype=np.int64).reshape(-1)
            if timing_contract_sha256 is not None
            else np.asarray([], dtype=np.int64)
        )
        recorded_delays = (
            np.asarray(data["segment_recorded_delay_samples"], dtype=np.float64).reshape(-1)
            if timing_contract_sha256 is not None
            else np.asarray([], dtype=np.float64)
        )
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
    if (
        experiment_contract_sha256 is not None
        and saved_contract_sha != experiment_contract_sha256
    ):
        errors.append("평가 experiment contract SHA-256이 완료 후보와 다릅니다")
    for label, saved, expected in (
        ("selection", saved_selection_sha, selection_sha256),
        ("test capability", saved_capability_sha, test_capability_sha256),
        ("test consumed marker", saved_consumed_sha, test_consumed_marker_sha256),
    ):
        if expected is not None and saved != expected:
            errors.append(f"평가 {label} SHA-256이 완료 체인과 다릅니다")
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
    strict_flags = strict_subband_summary["flags"]
    if not strict_flags["g4_trusted_subband_pass"]:
        errors.append(
            "strict trusted 150–1600Hz 부대역 G4를 통과하지 못했습니다: "
            f"coverage={strict_flags['g4_trusted_subband_coverage_pass']}, "
            f"groups={strict_flags['g4_trusted_subband_power_pass']}, "
            f"mean={strict_flags['g4_trusted_subband_mean_pass']}, "
            f"worst10={strict_flags['g4_trusted_subband_worst10_pass']}, "
            f"ci={strict_flags['g4_trusted_subband_ci_pass']}"
        )
    if not strict_flags["g4_upper_trusted_subband_pass"]:
        errors.append(
            "trusted upper 1000–1600Hz가 family별 mean/worst10/coverage/CI를 모두 "
            "통과하지 못했습니다 — 전체 150–1600Hz 평균으로 숨길 수 없습니다"
        )
    missing_families = sorted(set(required_source_families).difference(families))
    if missing_families:
        errors.append(f"G4 source_family 결과 누락: {missing_families}")
    if n_sessions <= 0 or n_segments <= 0:
        errors.append(f"G4 평가 표본이 비었습니다: sessions={n_sessions}, segments={n_segments}")
    if timing_contract_sha256 is not None:
        if timing_shas.size != n_segments or not np.all(
            timing_shas == timing_contract_sha256
        ):
            errors.append("segment timing contract SHA가 checkpoint와 다릅니다")
        if source_timelines.size != n_segments or not np.all(
            source_timelines == "source_aligned.wav"
        ):
            errors.append("공식 G4가 source_aligned.wav ADC 시간축을 쓰지 않았습니다")
        if (
            recorded_leads.size != n_segments
            or np.any(recorded_leads < 0)
            or recorded_delays.size != n_segments
            or not np.all(np.isfinite(recorded_delays))
            or np.any(recorded_delays < 0.0)
        ):
            errors.append("segment별 recorded timeline lead provenance가 유효하지 않습니다")
        elif timing_contract is not None:
            expected_leads = np.asarray(
                [
                    timing_contract.recorded_lead_samples(float(delay))
                    for delay in recorded_delays
                ],
                dtype=np.int64,
            )
            if not np.array_equal(recorded_leads, expected_leads):
                errors.append("segment recorded lead가 timing contract 유도값과 다릅니다")
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
    selection: str | Path | None = None,
    test_capability: str | Path | None = None,
    test_consumed_marker: str | Path | None = None,
    full_recorded_qa: bool = True,
) -> dict:
    """measured checkpoint와 독립 val/test G4를 묶어 완료 여부를 판정한다."""

    readiness = audit_finetune_readiness(cfg, full_recorded_qa=full_recorded_qa)
    audit = _Audit("finetune_completion")
    if readiness["ok"]:
        audit.pass_("readiness", "fine-tune 진입 준비 게이트가 통과했습니다")
    else:
        audit.fail("readiness", "fine-tune 진입 준비 게이트가 통과하지 않았습니다")

    checkpoint_path = _repo_path(checkpoint)
    candidate_sha: str | None = None
    contract_sha: str | None = None
    timing_contract_sha: str | None = None
    timing_contract_model: TrainingTimingContract | None = None
    checkpoint_cfg: dict[str, Any] | None = None
    strict_canonical_completion = (
        str(cfg.get("experiment_role", "")) == "canonical_finetune"
    )
    try:
        checkpoint_snapshot = snapshot_regular_file(checkpoint_path)
        checkpoint_path = checkpoint_snapshot.path
        state = _decode_checkpoint_state(checkpoint_snapshot.content, checkpoint_path)
        saved_cfg = state["cfg"]
        checkpoint_cfg = saved_cfg
        if strict_canonical_completion:
            contract_sha = str(
                validate_embedded_experiment_contract(saved_cfg)["sha256"]
            )
            timing_contract_model = TrainingTimingContract.from_data_config(
                saved_cfg.get("data") or {}
            )
            timing_contract_sha = timing_contract_model.digest()
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
        last_snapshot = snapshot_regular_file(companion_last)
        last_state = _decode_checkpoint_state(last_snapshot.content, companion_last)
        last_cfg = last_state["cfg"]
        if strict_canonical_completion:
            validate_embedded_experiment_contract(last_cfg)
        if _checkpoint_identity(last_cfg) != _checkpoint_identity(saved_cfg):
            raise ValueError("fine-tune best.pt와 last.pt의 immutable run 설정이 다릅니다")
        if _model_state_signature(last_state) != _model_state_signature(state):
            raise ValueError("fine-tune best.pt와 last.pt의 model state 구조가 다릅니다")
        if strict_canonical_completion:
            validate_completion_receipt(
                checkpoint_path.parent,
                expected_role="canonical_finetune",
                expected_init_eligible=False,
            )
        target = int((last_cfg.get("schedule", {}) or {}).get("total_steps", 0))
        step = int(last_state.get("step", -1))
        if target <= 0 or step != target:
            raise ValueError(f"fine-tune 학습 미완료: last step={step}, target={target}")
        candidate_sha = checkpoint_snapshot.sha256
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

    manifest_path = _repo_path(cfg.get("recorded_manifest", ""))
    manifest_bytes: bytes | None = None
    try:
        manifest_snapshot = snapshot_regular_file(manifest_path)
        manifest_sha = manifest_snapshot.sha256
        manifest_bytes = manifest_snapshot.content
        manifest_path = manifest_snapshot.path
    except (FileNotFoundError, OSError, ValueError) as exc:
        manifest_sha = ""
        audit.fail("recorded_manifest_provenance", str(exc))
    else:
        audit.pass_(
            "recorded_manifest_provenance",
            "완료 판정용 manifest 지문을 계산했습니다",
            manifest=str(manifest_path),
            manifest_sha256=manifest_sha,
        )

    chain: dict[str, str] = {}
    if strict_canonical_completion:
      try:
        if selection is None or test_capability is None or test_consumed_marker is None:
            raise ValueError(
                "완료 판정에는 selection/test capability/consumed marker가 모두 필요합니다"
            )
        selection_payload, selection_snapshot = read_json_snapshot(_repo_path(selection))
        # capability 발급 뒤 selection/bundle을 변조해 completion만 우회하는
        # 경로를 막는다. single-seed clear-pass 또는 검증된 cross-seed final을
        # 현재 immutable bytes에서 다시 판정한다.
        validate_test_open_selection(selection_payload)
        expected_capability_path, expected_consumed_path = (
            canonical_test_ledger_paths_from_payload(selection_payload)
        )
        event_paths = canonical_test_ledger_event_paths_from_payload(
            selection_payload
        )
        if Path(os.path.abspath(_repo_path(test_capability))) != expected_capability_path:
            raise ValueError("test capability가 campaign-wide canonical ledger 경로가 아닙니다")
        if Path(os.path.abspath(_repo_path(test_consumed_marker))) != expected_consumed_path:
            raise ValueError(
                "test consumed marker가 campaign-wide canonical ledger 경로가 아닙니다"
            )
        capability_payload, capability_snapshot = read_json_snapshot(
            _repo_path(test_capability)
        )
        consumed_payload, consumed_snapshot = read_json_snapshot(
            _repo_path(test_consumed_marker)
        )
        if event_paths["failed"].exists() or event_paths["failed"].is_symlink():
            raise ValueError("recorded test ledger가 failed 상태입니다")
        completed_payload, completed_snapshot = read_json_snapshot(
            event_paths["completed"]
        )
        selected = selection_payload.get("selected")
        if not isinstance(selected, dict):
            raise ValueError("selection.selected가 없습니다")
        expected_selection = {
            "selection_split": "val",
            "manifest_sha256": manifest_sha,
            "experiment_contract_sha256": contract_sha,
        }
        for key, value in expected_selection.items():
            if selection_payload.get(key) != value:
                raise ValueError(f"selection {key}가 완료 후보와 다릅니다")
        if selected.get("checkpoint_sha256") != candidate_sha:
            raise ValueError("selection checkpoint SHA가 완료 후보와 다릅니다")
        selected_metrics_sha = str(selected.get("metrics_sha256", ""))
        if selected_metrics_sha != snapshot_regular_file(val_metrics).sha256:
            raise ValueError("selection val candidate metrics와 canonical val bytes가 다릅니다")
        expected_capability = {
            "selection_sha256": selection_snapshot.sha256,
            "seed_neutral_campaign_sha256": selection_payload.get(
                "seed_neutral_campaign_sha256"
            ),
            "experiment_contract_sha256": contract_sha,
            "selected_checkpoint_sha256": candidate_sha,
            "manifest_sha256": manifest_sha,
        }
        for key, value in expected_capability.items():
            if capability_payload.get(key) != value:
                raise ValueError(f"test capability {key}가 selection과 다릅니다")
        expected_consumed = {
            **expected_capability,
            "capability_sha256": capability_snapshot.sha256,
        }
        expected_consumed.pop("selection_sha256")
        expected_consumed["selection_sha256"] = selection_snapshot.sha256
        for key, value in expected_consumed.items():
            if consumed_payload.get(key) != value:
                raise ValueError(f"test consumed marker {key}가 capability와 다릅니다")
        if capability_payload.get("phase") != "issued":
            raise ValueError("test capability phase가 issued가 아닙니다")
        if consumed_payload.get("phase") != "running":
            raise ValueError("test consumed marker phase가 running이 아닙니다")
        expected_completed = {
            "phase": "completed",
            "seed_neutral_campaign_sha256": selection_payload.get(
                "seed_neutral_campaign_sha256"
            ),
            "selection_sha256": selection_snapshot.sha256,
            "running_marker_sha256": consumed_snapshot.sha256,
            "experiment_contract_sha256": contract_sha,
            "selected_checkpoint_sha256": candidate_sha,
            "manifest_sha256": manifest_sha,
            "metrics_npz_sha256": snapshot_regular_file(test_metrics).sha256,
        }
        for key, value in expected_completed.items():
            if completed_payload.get(key) != value:
                raise ValueError(f"test completed ledger {key}가 결과와 다릅니다")
        chain = {
            "selection_sha256": selection_snapshot.sha256,
            "test_capability_sha256": capability_snapshot.sha256,
            "test_consumed_marker_sha256": consumed_snapshot.sha256,
            "test_completed_marker_sha256": completed_snapshot.sha256,
        }
        audit.pass_(
            "recorded_selection_test_once_chain",
            "recorded-val 선택과 single-use test capability 체인이 정합합니다",
            **chain,
        )
      except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        audit.fail("recorded_selection_test_once_chain", str(exc))

    required_families = _required_families(cfg.get("readiness", {}) or {})
    fingerprints: dict[str, str] = {}
    for split, path in (("val", val_metrics), ("test", test_metrics)):
        check_id = f"recorded_{split}_g4"
        if candidate_sha is None or checkpoint_cfg is None or not manifest_sha or manifest_bytes is None or (
            strict_canonical_completion and contract_sha is None
        ):
            audit.fail(check_id, "checkpoint/manifest provenance가 없어 G4를 검증할 수 없습니다")
            continue
        try:
            details = _audit_g4_metrics(
                path,
                expected_split=split,
                manifest_bytes=manifest_bytes,
                manifest_path=manifest_path,
                checkpoint_sha256=candidate_sha,
                manifest_sha256=manifest_sha,
                required_source_families=required_families,
                experiment_contract_sha256=contract_sha,
                selection_sha256=(
                    chain.get("selection_sha256") if split == "test" else None
                ),
                test_capability_sha256=(
                    chain.get("test_capability_sha256") if split == "test" else None
                ),
                test_consumed_marker_sha256=(
                    chain.get("test_consumed_marker_sha256")
                    if split == "test"
                    else None
                ),
                timing_contract_sha256=timing_contract_sha,
                timing_contract=timing_contract_model,
                checkpoint_cfg=checkpoint_cfg,
                canonical_sampling_binding=strict_canonical_completion,
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
