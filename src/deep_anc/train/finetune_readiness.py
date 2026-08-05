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
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from ..config import DEFAULT_HANDOFF_SAMPLES, REPO_ROOT
from ..data.manifest import read_manifest
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
    if repeats < 3:
        errors.append(f"ESS 반복 {repeats}회 < 3회")
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
        audit.pass_(
            "matched_path_measurement_conditions",
            "P/S official ESS 디지털 gain·block·latency 조건이 정합합니다",
            primary=primary,
            secondary=secondary,
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        audit.fail("matched_path_measurement_conditions", str(exc))

    configured_lead = int(data_cfg.get("digital_reference_lead_samples", -1))
    configured_primary_delay = duct_cfg.get("digital_reference", {}).get(
        "d_noise_delay_samples"
    )
    if primary is not None and secondary is not None:
        handoff = int(
            duct_cfg.get("secondary_path", {}).get(
                "handoff_extra_samples", DEFAULT_HANDOFF_SAMPLES
            )
        )
        expected_lead = max(
            0,
            int(secondary["delay_samples"])
            + handoff
            - int(primary["delay_samples"]),
        )
        delay_matches = (
            configured_primary_delay is not None
            and int(configured_primary_delay) == int(primary["delay_samples"])
        )
        if configured_lead != expected_lead or not delay_matches:
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

    try:
        init_value = cfg.get("init_ckpt")
        if not init_value:
            raise ValueError("init_ckpt가 비었습니다")
        init = audit_init_checkpoint(
            init_value,
            expected_model_cfg=cfg.get("model", {}),
            expected_lead=configured_lead,
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
    }


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
            audit.pass_(check_id, f"독립 recorded {split} G4가 통과했습니다", **details)
        except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            audit.fail(check_id, str(exc))

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
