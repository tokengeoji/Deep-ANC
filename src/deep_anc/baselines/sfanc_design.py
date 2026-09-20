"""SFANC 후보 FIR의 인과 FP64 최소제곱 설계와 독립 고정 필터 채점.

물리 규약은 실제 DAC 명령 y=W*x, e=d+S*y다. S의 이득·부호·선행 탭을
변경하지 않으며 추가 지연은 한 번만 적용한다. 입력이 실측인지 합성인지는
이 수치 API가 판정하지 않는다. 추론·장치·오디오·펌웨어를 열지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.signal import fftconvolve


MAX_SAMPLES = 1_000_000
MAX_CONTROL_LENGTH = 4096
MAX_DESIGN_ELEMENTS = 16_777_216
MAX_SCORE_ELEMENTS = 16_777_216


@dataclass(frozen=True)
class ControlFIRFit:
    """coefficients는 읽기 전용 FP64[L], diagnostics는 직렬화 가능한 수치다."""

    coefficients: np.ndarray
    diagnostics: dict


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name}: {minimum} 이상 정수가 필요합니다")
    return int(value)


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name}: 유한한 실수가 필요합니다")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"{name}: 유한한 {'양수' if positive else '비음수'}가 필요합니다")
    return number


def _vector(value, name):
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: 실수 벡터가 필요합니다") from exc
    if raw.ndim != 1 or not raw.size or raw.size > MAX_SAMPLES or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name}: 길이 1~{MAX_SAMPLES}인 실수 1차원 벡터가 필요합니다")
    values = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: 비유한 값은 허용하지 않습니다")
    return values


def _inputs(reference, disturbance, secondary, secondary_delay_samples, warmup_samples):
    x, d, s = (_vector(value, name) for value, name in (
        (reference, "reference"), (disturbance, "disturbance"), (secondary, "secondary")))
    if x.size != d.size:
        raise ValueError("reference와 disturbance의 길이가 같아야 합니다")
    if not np.any(s):
        raise ValueError("secondary는 모두 0일 수 없습니다")
    delay = _integer(secondary_delay_samples, "secondary_delay_samples")
    warmup = _integer(warmup_samples, "warmup_samples")
    if warmup >= x.size:
        raise ValueError("warmup_samples 뒤에 평가할 샘플이 필요합니다")
    return x, d, s, delay, warmup


def _filtered(values, fir, delay=0):
    """제로 초기 상태의 full 선형 컨볼루션을 입력 길이까지만 반환한다."""
    if delay >= values.size or not np.any(values) or not np.any(fir):
        return np.zeros(values.size, dtype=np.float64)
    onset = int(np.argmax(values != 0)) + int(np.argmax(fir != 0)) + delay
    if onset >= values.size:
        return np.zeros(values.size, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        filtered = fftconvolve(values, fir, mode="full")[:values.size]
    if not np.isfinite(filtered).all():
        raise ValueError("인과 컨볼루션의 FP64 범위를 벗어났습니다")
    if delay:
        shifted = np.zeros_like(filtered)
        shifted[delay:] = filtered[:-delay]
        filtered = shifted
    # 수학적으로 정확히 0인 선행 구간에 FFT 반올림 잡음을 남기지 않는다.
    # S의 선행 탭을 자르거나 이동하는 동작이 아니다.
    filtered[:onset] = 0.0
    return filtered


def _taps(values, length):
    padded = np.pad(values, (length - 1, 0))
    return np.lib.stride_tricks.sliding_window_view(padded, length)[:, ::-1].copy()


def _mse(values, axis=None):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.mean(np.square(values), axis=axis)
    if not np.isfinite(result).all():
        raise ValueError("mean-square 계산의 FP64 범위를 벗어났습니다")
    return result


def score_control_filters(reference, disturbance, secondary, coefficients, *,
                          secondary_delay_samples=0, warmup_samples=0, effort_penalty=0.0):
    """고정 FIR들에 대해 y=W*x, e=d+delay(S*y)를 계산한다.

    coefficients는 [L] 또는 [K,L]이고 출력 파형은 항상 [K,N]이다.
    control/secondary_output/residual은 워밍업을 포함한 전체 길이다.
    residual_mse/control_mse/objective[K]는 플랜트 적용 **후** warmup 이후를
    채점한다. objective=residual_mse+effort_penalty*control_mse이며 계수 ridge는
    포함하지 않는다. baseline_mse는 같은 구간의 d 제곱평균이다.
    """
    x, d, s, delay, warmup = _inputs(
        reference, disturbance, secondary, secondary_delay_samples, warmup_samples)
    effort = _number(effort_penalty, "effort_penalty")
    try:
        weights = np.asarray(coefficients)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("coefficients: [L] 또는 [K,L] 실수 배열이 필요합니다") from exc
    if weights.ndim == 1:
        weights = weights[None, :]
    if (weights.ndim != 2 or not weights.shape[0] or not weights.shape[1]
            or weights.shape[1] > MAX_CONTROL_LENGTH or weights.dtype.kind not in "fiu"
            or weights.shape[0] * x.size > MAX_SCORE_ELEMENTS):
        raise ValueError("coefficients: shape 또는 채점 계산 크기 제한 위반")
    weights = np.array(weights, dtype=np.float64, copy=True)
    if not np.isfinite(weights).all():
        raise ValueError("coefficients: 비유한 값은 허용하지 않습니다")
    control = np.stack([_filtered(x, row) for row in weights])
    secondary_output = np.stack([_filtered(row, s, delay) for row in control])
    with np.errstate(over="ignore", invalid="ignore"):
        residual = d[None, :] + secondary_output
    residual_mse = _mse(residual[:, warmup:], axis=1)
    control_mse = _mse(control[:, warmup:], axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        objective = residual_mse + effort * control_mse
    if not np.isfinite(objective).all():
        raise ValueError("objective의 FP64 범위를 벗어났습니다")
    return {"control": control, "secondary_output": secondary_output, "residual": residual,
            "residual_mse": residual_mse, "control_mse": control_mse, "objective": objective,
            "baseline_mse": float(_mse(d[warmup:])), "warmup_samples": warmup,
            "evaluated_samples": int(x.size - warmup)}


def fit_control_fir(reference, disturbance, secondary, *, control_length,
                    secondary_delay_samples=0, warmup_samples,
                    regularization=1e-6, effort_penalty=0.0, max_coefficient_norm=None):
    """FP64 ridge 최소제곱으로 실제 출력 명령 FIR W를 설계한다.

    J(w)=mean((d+A w)^2)+effort_penalty*mean((X w)^2)+regularization*||w||².
    A는 delay(S*x), X는 x의 인과 Toeplitz다. 평균은 warmup 이후 M개 행에
    대해서만 취한다. 제로 초기 상태로 전체 플랜트를 먼저 적용하므로 워밍업
    이전 출력이 이후 잔차에 주는 영향도 보존한다. sqrt(M)로 스케일한
    augmented least-squares를 SVD로 풀어 normal-equation의 조건수 제곱을 피한다.
    norm 한도를 넘으면 실패하며 계수를 자르거나 정규화하지 않는다.
    """
    x, d, s, delay, warmup = _inputs(
        reference, disturbance, secondary, secondary_delay_samples, warmup_samples)
    length = _integer(control_length, "control_length", minimum=1)
    regularization = _number(regularization, "regularization", positive=True)
    effort = _number(effort_penalty, "effort_penalty")
    maximum = None if max_coefficient_norm is None else _number(
        max_coefficient_norm, "max_coefficient_norm", positive=True)
    count = x.size - warmup
    if length > MAX_CONTROL_LENGTH or x.size * length > MAX_DESIGN_ELEMENTS:
        raise ValueError("FIR 설계 계산 크기 제한 위반")
    if count < length:
        raise ValueError("워밍업 이후 녹음이 control_length보다 짧습니다")
    if not np.any(x):
        raise ValueError("무신호 reference로 필터를 학습할 수 없습니다")
    if delay >= x.size:
        raise ValueError("녹음 안에 secondary 지연 이후 제어 영향이 없습니다")
    filtered_reference = _filtered(x, s, delay)
    design = _taps(filtered_reference, length)[warmup:]
    control_design = _taps(x, length)[warmup:]
    if not np.any(design):
        raise ValueError("채점 구간에 filtered reference가 없습니다")
    scale = math.sqrt(count)
    matrices = [design / scale, math.sqrt(regularization) * np.eye(length)]
    targets = [-d[warmup:] / scale, np.zeros(length)]
    if effort:
        matrices.append(math.sqrt(effort) * control_design / scale)
        targets.append(np.zeros(count))
    matrix, target = np.concatenate(matrices), np.concatenate(targets)
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("정규화 설계 행렬의 FP64 범위를 벗어났습니다")
    try:
        coefficients, _, rank, singular = np.linalg.lstsq(matrix, target, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError("FIR 최소제곱 해를 계산할 수 없습니다") from exc
    norm = float(np.linalg.norm(coefficients))
    if not np.isfinite(coefficients).all() or not math.isfinite(norm):
        raise ValueError("FIR 해가 유한하지 않습니다")
    if maximum is not None and norm > maximum:
        raise ValueError(f"계수 norm {norm:.8g}이 제한 {maximum:.8g}을 넘습니다; 계수 변경 없음")
    score = score_control_filters(x, d, s, coefficients, secondary_delay_samples=delay,
                                  warmup_samples=warmup, effort_penalty=effort)
    gradient = 2 * (design.T @ (d[warmup:] + design @ coefficients) / count
                    + effort * (control_design.T @ (control_design @ coefficients)) / count
                    + regularization * coefficients)
    ridge = float(regularization * np.dot(coefficients, coefficients))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    diagnostics = {
        "solver": "fp64_augmented_lstsq", "sample_count": int(x.size),
        "evaluated_samples": int(count), "warmup_samples": warmup,
        "control_length": length, "secondary_delay_samples": delay,
        "regularization": regularization, "effort_penalty": effort,
        "baseline_mse": score["baseline_mse"], "residual_mse": float(score["residual_mse"][0]),
        "control_mse": float(score["control_mse"][0]), "ridge_term": ridge,
        "objective": float(score["objective"][0] + ridge), "coefficient_norm": norm,
        "normal_equation_gradient_norm": float(np.linalg.norm(gradient)),
        "augmented_rank": int(rank), "augmented_condition_number": condition,
        "objective_definition": "mean(e**2)+effort_penalty*mean(y**2)+regularization*sum(w**2)",
    }
    if any(isinstance(value, float) and not math.isfinite(value) for value in diagnostics.values()):
        raise ValueError("FIR 진단의 FP64 범위를 벗어났습니다")
    coefficients = np.array(coefficients, dtype=np.float64, copy=True)
    coefficients.setflags(write=False)
    return ControlFIRFit(coefficients, diagnostics)
