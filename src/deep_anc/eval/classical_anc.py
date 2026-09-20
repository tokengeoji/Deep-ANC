"""명시적으로 구분한 plain FxLMS/FxNLMS의 인과 오프라인 수치 기준선.

FP64 플랜트와 적응 연산을 사용한다. 기존 FP32 FxLMSController/FxNLMS 진단을
대체하거나 성공한 OMAP 펌웨어를 재현하지 않는다. 블록 수집·실기 지연은 별도다.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy import signal

from .sfanc_fxnlms import (
    _integer, _positive, _wave, _MAX_SAMPLES, _MAX_MATRIX_ELEMENTS,
)


class _Path:
    """선행 FIR 탭을 변경하지 않는 FP64 지연 + 상태 보존 경로."""

    def __init__(self, coefficients: np.ndarray, delay: int):
        self.coefficients = coefficients.copy()
        self.delay = np.zeros(delay, dtype=np.float64)
        self.state = np.zeros(coefficients.size - 1, dtype=np.float64)

    def process(self, values: np.ndarray) -> np.ndarray:
        if self.delay.size:
            joined = np.concatenate((self.delay, values))
            delayed = joined[:values.size]
            self.delay = joined[values.size:].copy()
        else:
            delayed = values
        output, self.state = signal.lfilter(
            self.coefficients, [1.0], delayed, zi=self.state,
        )
        return output


def _taps(values: np.ndarray, history: np.ndarray, length: int):
    extended = np.concatenate((history, values))
    matrix = np.lib.stride_tricks.sliding_window_view(extended, length)[:, ::-1]
    return matrix, extended[-(length - 1):].copy() if length > 1 else np.empty(0)


def _hash(coefficients: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(coefficients, dtype="<f8").tobytes()).hexdigest()


def run_classical_anc(
    reference, disturbance, secondary, *, algorithm: str,
    additional_delay_samples: int = 0, control_length: int = 128,
    block_samples: int = 1, mu: float = 0.05, control_limit: float = 0.2,
    normalization_epsilon: float = 1e-12, secondary_estimate=None,
    estimate_delay_samples: int | None = None, weight_norm_limit: float = 20.0,
) -> dict:
    """계수 0에서 시작해 실제 제한 명령으로 e=d+S*delay(u)를 계산한다.

    한 블록의 모든 출력은 갱신 전 w로 만든다. 같은 시각의 ERR 블록을 받은 뒤:
      FxLMS:  w ← w - mu * (X_f.T @ e)
      FxNLMS: w ← w - mu * (X_f.T @ e) / (sum(X_f**2) + epsilon)
    따라서 block_samples=1일 때 각각 -mu*e*xf와 정규화된 샘플별 갱신이다.
    더 큰 블록은 합산 gradient/블록 전체 정규화이며, 블록 길이로 나누거나 샘플마다
    갱신한 것과 같지 않다. step size의 의미가 cadence에 종속됨을 설정에 기록한다.

    S_hat은 별도로 제공할 수 있고 기본은 실제 S의 복사본이다. 추정 추가 지연의
    기본은 실제 추가 지연이다. 이는 완전한 모델 지식을 준 합성 조건이지 실측 인증이
    아니다. 두 경로에서 추가 지연을 각 한 번만 적용하며 1 hop을 암묵 가산하지 않는다.

    hard clip 이후 전체 S 꼬리까지 해당 블록의 적응을 보류하고, 갱신 후 계수 norm을
    명시 한도로 투영한다. 실패·제한 횟수를 숨기지 않는다. 원본 데이터·S는 변경하지
    않으며, 호출마다 모든 상태를 초기화한다. hyperparameter 선택/학습 실행기는 아니다.
    """
    if not isinstance(algorithm, str) or algorithm not in ("fxlms", "fxnlms"):
        raise ValueError("algorithm은 fxlms 또는 fxnlms여야 합니다")
    normalized = algorithm == "fxnlms"
    delay = _integer(additional_delay_samples, "additional_delay_samples", 0, _MAX_SAMPLES)
    estimate_delay = (delay if estimate_delay_samples is None else
                      _integer(estimate_delay_samples, "estimate_delay_samples", 0, _MAX_SAMPLES))
    length = _integer(control_length, "control_length", 1, 4096)
    block = _integer(block_samples, "block_samples", 1, _MAX_SAMPLES)
    step = _positive(mu, "mu", 1e6)
    limit = _positive(control_limit, "control_limit", 1e6)
    epsilon = _positive(normalization_epsilon, "normalization_epsilon", 1e6)
    norm_limit = _positive(weight_norm_limit, "weight_norm_limit", 1e6)
    x = _wave(reference, "reference", _MAX_SAMPLES)
    d = _wave(disturbance, "disturbance", _MAX_SAMPLES)
    actual_s = _wave(secondary, "secondary", 8192)
    estimate_s = (actual_s.copy() if secondary_estimate is None else
                  _wave(secondary_estimate, "secondary_estimate", 8192))
    if x.shape != d.shape:
        raise ValueError("reference와 disturbance의 길이가 같아야 합니다")
    if not np.any(actual_s) or not np.any(estimate_s):
        raise ValueError("secondary와 secondary_estimate는 전부 0일 수 없습니다")
    if min(block, x.size) * length > _MAX_MATRIX_ELEMENTS:
        raise ValueError("block_samples * control_length 행렬이 너무 큽니다")

    plant, filtered_reference = _Path(actual_s, delay), _Path(estimate_s, estimate_delay)
    weights = np.zeros(length, dtype=np.float64)
    history = np.zeros(length - 1, dtype=np.float64)
    filtered_history = np.zeros_like(history)
    raw_control, control, residual = (np.zeros_like(x) for _ in range(3))
    adapted_blocks = limited_samples = clipped_guard_blocks = weight_limited_blocks = 0
    zero_gradient_blocks = 0
    guard_until = 0

    for start in range(0, x.size, block):
        end = min(start + block, x.size)
        x_matrix, history = _taps(x[start:end], history, length)
        raw = x_matrix @ weights
        xf = filtered_reference.process(x[start:end])
        xf_matrix, filtered_history = _taps(xf, filtered_history, length)
        if not np.isfinite(raw).all() or not np.isfinite(xf_matrix).all():
            raise FloatingPointError("제한 전 출력 또는 filtered-x가 유한하지 않습니다")
        clipped = np.abs(raw) > limit
        limited_samples += int(np.count_nonzero(clipped))
        if np.any(clipped):
            guard_until = max(guard_until, end + delay + actual_s.size - 1)
        command = np.clip(raw, -limit, limit)
        error = d[start:end] + plant.process(command)
        if not np.isfinite(error).all():
            raise FloatingPointError("플랜트 잔차가 유한하지 않습니다")
        guarded = start < guard_until
        clipped_guard_blocks += int(guarded)
        if not guarded:
            power = float(np.sum(xf_matrix * xf_matrix))
            gradient = xf_matrix.T @ error
            if not np.isfinite(power) or not np.isfinite(gradient).all():
                raise FloatingPointError("적응 gradient 또는 정규화 파워가 유한하지 않습니다")
            # plain 모드에 epsilon 기반 적응 문턱을 몰래 적용하지 않는다.
            if not np.any(gradient):
                zero_gradient_blocks += 1
            else:
                denominator = power + epsilon if normalized else 1.0
                weights -= step * gradient / denominator
                if not np.isfinite(weights).all():
                    raise FloatingPointError("적응 계수가 유한하지 않습니다")
                norm = float(np.linalg.norm(weights))
                if not np.isfinite(norm):
                    raise FloatingPointError("적응 계수 norm이 유한하지 않습니다")
                if norm > norm_limit:
                    weights *= norm_limit / norm
                    weight_limited_blocks += 1
                adapted_blocks += 1
        raw_control[start:end], control[start:end], residual[start:end] = raw, command, error

    return {
        "control": control, "raw_control": raw_control, "residual": residual,
        "final_weights": weights.copy(), "weight_norm": float(np.linalg.norm(weights)),
        "adapted_blocks": adapted_blocks, "limited_samples": limited_samples,
        "clipped_guard_blocks": clipped_guard_blocks,
        "weight_limited_blocks": weight_limited_blocks, "zero_gradient_blocks": zero_gradient_blocks,
        "settings": {
            "algorithm": algorithm, "normalized": normalized, "dtype": "float64",
            "update_convention": "samplewise" if block == 1 else "block_sum_gradient",
            "gradient": "X_f.T @ e (sum, not mean)",
            "normalization": "sum(X_f**2)+epsilon" if normalized else "none",
            "normalization_epsilon": epsilon if normalized else None,
            "block_samples": block, "control_length": length, "mu": step,
            "control_limit": limit, "weight_norm_limit": norm_limit,
            "initial_weights": "zero", "leakage": 0.0, "sign": "e=d+S*u",
            "additional_delay_samples": delay, "estimate_delay_samples": estimate_delay,
            "secondary_taps": int(actual_s.size), "secondary_sha256_f64le": _hash(actual_s),
            "estimate_taps": int(estimate_s.size), "estimate_sha256_f64le": _hash(estimate_s),
            "estimate_source": "plant_copy" if secondary_estimate is None else "explicit_argument",
            "estimate_matches_plant": bool(np.array_equal(actual_s, estimate_s)),
            "estimate_delay_matches_plant": estimate_delay == delay,
            "clipping_policy": "hard_clip_then_freeze_through_secondary_tail",
        },
        "claims": {
            "firmware_equivalent": False, "strongly_tuned_baseline": False,
            "physical_performance_claim_allowed": False, "physical_latency_measured": False,
            "implicit_block_delay_added": False, "real_time_claim_allowed": False,
            "deployment_allowed": False,
        },
    }
