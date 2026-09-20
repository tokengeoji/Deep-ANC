"""SFANC와 같은 플랜트·출력 한도로 비교하는 오프라인 cold-start FxNLMS.

기존 FP32 FxLMSController를 재사용하지만 플랜트의 S는 FP64 원본 그대로다.
실제 OMAP 펌웨어의 DC blocker·noise gate·tap clamp·step cap 재현은 아니다.
블록 수집/스케줄/I/O 지연을 몰래 한 hop 넣지 않는다. 추가 지연은 명시된 값만
플랜트와 filtered-x에 각각 한 번 적용하며 실제 오디오 지연을 증명하지 않는다.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from deep_anc.baselines.fxlms_core import FxLMSController


_MAX_SAMPLES = 1_048_576
_MAX_MATRIX_ELEMENTS = 16_777_216


def _integer(value, name: str, minimum: int, maximum: int) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not minimum <= value <= maximum):
        raise ValueError(f"{name}: {minimum}..{maximum} 정수가 필요합니다")
    return int(value)


def _positive(value, name: str, maximum: float) -> float:
    if isinstance(value, (bool, np.bool_, str)):
        raise ValueError(f"{name}: 유한한 양수가 필요합니다")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: 유한한 양수가 필요합니다") from exc
    if not np.isfinite(number) or not 0 < number <= maximum:
        raise ValueError(f"{name}: 0 초과 {maximum} 이하의 유한 값이 필요합니다")
    return number


def _wave(values, name: str, maximum_size: int) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: 유한 실수 1차원 배열이 필요합니다") from exc
    if (array.ndim != 1 or not 1 <= array.size <= maximum_size
            or array.dtype.kind not in "fiu" or not np.isfinite(array).all()
            or np.max(np.abs(array.astype(np.float64))) > 1e6):
        raise ValueError(f"{name}: 크기/유한 실수/수치 범위를 확인하십시오")
    return np.array(array, dtype=np.float64, copy=True)


def run_fxnlms(
    reference, disturbance, secondary, *, additional_delay_samples: int = 0,
    control_length: int = 128, block_samples: int = 32, mu: float = 0.05,
    control_limit: float = 0.2,
) -> dict:
    """인과 block FxNLMS를 초기 계수 0에서 실행하고 실제 제한된 u로 e를 계산한다.

    각 블록은 generate → hard clip → 지연/원본 S → 같은 시간 ERR → adapt 순서다.
    clip이 발생하면 해당 블록부터 마지막 clip의 전체 S 꼬리가 지난 다음 블록까지
    적응을 보류한다. 이 보수적 정책은 clip 미분을 선형 gradient로 쓰지 않기 위함이며
    폐루프 안전 인증이 아니다. block_samples=1이면 samplewise update다.

    기본 mu는 사전 고정 진단 설정이며 최적화된 FxNLMS 성능 상한이 아니다.
    호출마다 controller와 플랜트 상태를 새로 만들어 원천 간 이력을 공유하지 않는다.
    """
    delay = _integer(additional_delay_samples, "additional_delay_samples", 0, _MAX_SAMPLES)
    length = _integer(control_length, "control_length", 1, 4096)
    block = _integer(block_samples, "block_samples", 1, _MAX_SAMPLES)
    step = _positive(mu, "mu", 2.0)
    limit = _positive(control_limit, "control_limit", 1e6)
    x = _wave(reference, "reference", _MAX_SAMPLES)
    d = _wave(disturbance, "disturbance", _MAX_SAMPLES)
    secondary_original = _wave(secondary, "secondary", 8192)
    if x.shape != d.shape:
        raise ValueError("reference와 disturbance 길이가 같아야 합니다")
    if not np.any(secondary_original != 0):
        raise ValueError("secondary는 전부 0일 수 없습니다")
    if min(block, x.size) * length > _MAX_MATRIX_ELEMENTS:
        raise ValueError("block_samples * control_length 행렬이 너무 큽니다")

    controller = FxLMSController(
        secondary_original, secondary_delay_samples=delay, control_len=length,
        mu=step, leakage=0.0, normalization_epsilon=1e-12, weight_norm_limit=20.0,
    )
    raw_control = np.zeros(x.size, dtype=np.float64)
    control = np.zeros_like(raw_control)
    residual = np.zeros_like(raw_control)
    secondary_state = np.zeros(secondary_original.size - 1, dtype=np.float64)
    delay_state = np.zeros(delay, dtype=np.float64)
    guard_until = 0
    limited_samples = adapted_blocks = clipped_guard_blocks = weight_limited_blocks = 0

    for start in range(0, x.size, block):
        end = min(start + block, x.size)
        raw = np.asarray(controller.generate_block(x[start:end]), dtype=np.float64)
        if not np.isfinite(raw).all():
            raise FloatingPointError("FxNLMS 제한 전 출력이 유한하지 않습니다")
        clipped = np.abs(raw) > limit
        limited_samples += int(np.count_nonzero(clipped))
        if np.any(clipped):
            # exclusive 끝 + 추가 지연 + (S 길이 - 1); 마지막 clip의 전체 꼬리까지.
            guard_until = max(guard_until, end + delay + secondary_original.size - 1)
        command = np.clip(raw, -limit, limit)
        if delay:
            joined = np.concatenate((delay_state, command))
            delayed = joined[:end - start]
            delay_state = joined[end - start:].copy()
        else:
            delayed = command
        acoustic, secondary_state = signal.lfilter(
            secondary_original, [1.0], delayed, zi=secondary_state,
        )
        error = d[start:end] + acoustic
        if not np.isfinite(error).all():
            raise FloatingPointError("FxNLMS 플랜트 잔차가 유한하지 않습니다")
        guarded = start < guard_until
        result = controller.adapt_block(error, enabled=not guarded)
        diagnostics = (result.filtered_reference_power, result.gradient_norm, result.weight_norm)
        if not np.isfinite(diagnostics).all():
            raise FloatingPointError("FxNLMS 적응 진단이 유한하지 않습니다")
        raw_control[start:end] = raw
        control[start:end] = command
        residual[start:end] = error
        adapted_blocks += int(result.adapted)
        clipped_guard_blocks += int(guarded)
        weight_limited_blocks += int(result.weight_limited)

    return {
        "control": control, "raw_control": raw_control, "residual": residual,
        "limited_samples": limited_samples, "adapted_blocks": adapted_blocks,
        "clipped_guard_blocks": clipped_guard_blocks,
        "weight_limited_blocks": weight_limited_blocks,
        "weight_norm": float(np.linalg.norm(controller.w)),
        "final_weights": controller.w.astype(np.float64),
        "settings": {
            "additional_delay_samples": delay, "control_length": length,
            "block_samples": block, "mu": step, "control_limit": limit,
            "leakage": 0.0, "normalization_epsilon": 1e-12, "weight_norm_limit": 20.0,
            "adaptation_dtype": "float32", "plant_dtype": "float64",
            "secondary_taps": int(secondary_original.size), "sign": "e=d+S*u",
            "initial_weights": "zero", "clipping_policy": "hard_clip_then_freeze_through_secondary_tail",
        },
        "claims": {
            "firmware_equivalent": False, "physical_latency_measured": False,
            "implicit_block_delay_added": False, "real_time_claim_allowed": False,
            "deployment_allowed": False, "hyperparameters_tuned_on_test": False,
        },
    }
