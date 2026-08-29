"""다중 panel 광대역 전달함수의 공통 위상 stitch와 compact FIR 적합.

각 panel의 fractional joint-LS는 P/S를 같은 ADC↔DAC 시간축에서 분리해야 한다. 이
모듈은 그 결과의 **동일 drive·동일 FFT bin overlap**만 사용해 panel 사이 공통 fractional
delay를 찾는다. P와 S에 서로 다른 보정을 허용하지 않으므로, 상대위상을 결과에 맞춰
임의로 고칠 수 없다.

panel별 sparse IFFT는 pilot↔고역 gap과 제한된 대역 때문에 금지한다. 모든 panel을 stitch한
뒤 한 번만 real-valued compact FIR을 적합하고, 측정 복소 전달과 round-trip을 재검산한다.
오디오 장치는 열지 않는다.
"""

from __future__ import annotations

import math
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar


PANEL_STITCH_MIN_COMPLEX_AGREEMENT = 0.999
PANEL_STITCH_MAX_RELATIVE_ERROR = 0.05
PANEL_STITCH_MAX_CONSTANT_PHASE_RADIANS = 0.05
PANEL_STITCH_MAX_ABS_DELAY_SAMPLES = 16.0
COMPACT_IDENTIFIABILITY_SCHEMA = "compact_fir_identifiability_diagnostic_v1"
COMPACT_MAX_CONDITION_NUMBER = 1.0e8


def compact_fir_identifiability_receipt(
    frequencies_hz: np.ndarray,
    *,
    effective_delay_samples: int,
    fir_length: int,
    sample_rate: int = 48_000,
    ridge_relative: float = 1.0e-8,
) -> dict[str, Any]:
    """ridge 적합 전에 unregularized real design의 식별 가능성을 진단한다.

    ridge는 수치 solve를 가능하게 할 뿐 미측정 null-space를 물리 정보로 만들지
    않는다. 따라서 real design이 full-column-rank가 아니거나 condition gate를
    넘으면 compact FIR은 measured-tone roundtrip이 좋아도 학습 plant가 될 수 없다.
    """

    frequency = _finite_vector(
        frequencies_hz, complex_values=False, label="compact identifiability frequency"
    )
    length = int(fir_length)
    delay = int(effective_delay_samples)
    rate = int(sample_rate)
    ridge = float(ridge_relative)
    if (
        length <= 0
        or delay < 0
        or rate <= 0
        or np.any(np.diff(frequency) <= 0.0)
        or not math.isfinite(ridge)
        or not 0.0 <= ridge <= 1.0e-2
    ):
        raise ValueError("compact identifiability 입력이 잘못됐습니다")
    sample_index = delay + np.arange(length, dtype=np.float64)
    basis = np.exp(
        -2j * math.pi * np.outer(frequency, sample_index) / float(rate)
    )
    design = np.concatenate((basis.real, basis.imag), axis=0)
    singular = np.linalg.svd(design, compute_uv=False)
    tolerance = float(
        max(design.shape) * np.finfo(np.float64).eps * float(singular[0])
    )
    numeric_rank = int(np.count_nonzero(singular > tolerance))
    minimum = float(singular[-1])
    condition = float(singular[0] / minimum) if minimum > 0.0 else math.inf
    full_rank = numeric_rank == length
    condition_passed = math.isfinite(condition) and condition <= (
        COMPACT_MAX_CONDITION_NUMBER
    )
    payload: dict[str, Any] = {
        "schema_version": COMPACT_IDENTIFIABILITY_SCHEMA,
        "compact_role": "diagnostic_only",
        "compact_training_eligible": bool(full_rank and condition_passed),
        "tone_count": int(frequency.size),
        "real_equation_count": int(design.shape[0]),
        "fir_length": length,
        "effective_delay_samples": delay,
        "sample_rate": rate,
        "ridge_relative": ridge,
        "numeric_rank": numeric_rank,
        "rank_tolerance": tolerance,
        "full_column_rank": full_rank,
        "condition_number": condition if math.isfinite(condition) else None,
        "maximum_condition_number": COMPACT_MAX_CONDITION_NUMBER,
        "condition_passed": condition_passed,
        "ridge_does_not_restore_identifiability": True,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _finite_vector(
    value: np.ndarray | Sequence[complex] | Sequence[float],
    *,
    complex_values: bool,
    label: str,
) -> np.ndarray:
    dtype = np.complex128 if complex_values else np.float64
    result = np.asarray(value, dtype=dtype).reshape(-1)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{label}이 비었거나 NaN/Inf를 포함합니다")
    return result


def _exact_overlap(
    reference_frequencies_hz: np.ndarray,
    reference_transfer: np.ndarray,
    current_frequencies_hz: np.ndarray,
    current_transfer: np.ndarray,
    *,
    overlap_band_hz: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_f = _finite_vector(
        reference_frequencies_hz, complex_values=False, label="reference frequency"
    )
    cur_f = _finite_vector(
        current_frequencies_hz, complex_values=False, label="current frequency"
    )
    ref_h = _finite_vector(reference_transfer, complex_values=True, label="reference H")
    cur_h = _finite_vector(current_transfer, complex_values=True, label="current H")
    if ref_f.size != ref_h.size or cur_f.size != cur_h.size:
        raise ValueError("frequency/transfer 길이가 다릅니다")
    if np.any(np.diff(ref_f) <= 0.0) or np.any(np.diff(cur_f) <= 0.0):
        raise ValueError("frequency는 strict 증가해야 합니다")
    lo, hi = (float(value) for value in overlap_band_hz)
    if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo < hi):
        raise ValueError("overlap band가 유효하지 않습니다")

    # 모든 probe가 같은 48k/6000 grid를 사용한다. exact intersection이 없으면 설계 또는
    # artifact가 틀린 것이므로 complex interpolation으로 조용히 메우지 않는다.
    common, ref_index, cur_index = np.intersect1d(
        ref_f, cur_f, assume_unique=True, return_indices=True
    )
    mask = (common >= lo) & (common <= hi)
    if int(mask.sum()) < 8:
        raise ValueError(
            f"동일-bin overlap tone이 {int(mask.sum())}개뿐입니다: {overlap_band_hz}"
        )
    return common[mask], ref_h[ref_index[mask]], cur_h[cur_index[mask]]


def fit_shared_panel_delay(
    *,
    reference_frequencies: Mapping[str, np.ndarray],
    reference_transfers: Mapping[str, np.ndarray],
    current_frequencies: Mapping[str, np.ndarray],
    current_transfers: Mapping[str, np.ndarray],
    overlap_band_hz: tuple[float, float],
    sample_rate: int = 48_000,
    max_abs_delay_samples: float = PANEL_STITCH_MAX_ABS_DELAY_SAMPLES,
) -> dict[str, Any]:
    """P/S 두 drive에 하나의 fractional delay만 허용해 현재 panel을 이전 panel에 맞춘다."""

    rate = int(sample_rate)
    limit = float(max_abs_delay_samples)
    if rate <= 0 or not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("sample_rate/max_abs_delay_samples가 잘못됐습니다")
    required_drives = ("noise", "cancel")
    rows: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for drive in required_drives:
        try:
            rows[drive] = _exact_overlap(
                reference_frequencies[drive],
                reference_transfers[drive],
                current_frequencies[drive],
                current_transfers[drive],
                overlap_band_hz=overlap_band_hz,
            )
        except KeyError as exc:
            raise ValueError(f"panel stitch에 {drive} drive가 없습니다") from exc

    normalized_reference: list[np.ndarray] = []
    normalized_current: list[np.ndarray] = []
    omega: list[np.ndarray] = []
    for frequency, reference, current in rows.values():
        ref_norm = float(np.linalg.norm(reference))
        cur_norm = float(np.linalg.norm(current))
        if ref_norm <= 0.0 or cur_norm <= 0.0:
            raise ValueError("panel overlap transfer energy가 0입니다")
        normalized_reference.append(reference / ref_norm)
        normalized_current.append(current / cur_norm)
        omega.append(2.0 * math.pi * frequency / rate)
    ref_all = np.concatenate(normalized_reference)
    cur_all = np.concatenate(normalized_current)
    omega_all = np.concatenate(omega)

    def objective(delay: float) -> float:
        aligned = cur_all * np.exp(1j * omega_all * float(delay))
        return -float(abs(complex(np.vdot(ref_all, aligned))))

    coarse = np.linspace(-limit, limit, int(round(2.0 * limit / 0.02)) + 1)
    scores = np.asarray([-objective(value) for value in coarse], dtype=np.float64)
    best = int(np.argmax(scores))
    lower = float(coarse[max(0, best - 2)])
    upper = float(coarse[min(coarse.size - 1, best + 2)])
    if lower == upper:
        delay = float(coarse[best])
    else:
        optimized = minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1.0e-8},
        )
        if not optimized.success or not math.isfinite(float(optimized.x)):
            raise ValueError("panel shared fractional delay 최적화가 실패했습니다")
        delay = float(optimized.x)

    drive_reports: dict[str, Any] = {}
    combined_ref: list[np.ndarray] = []
    combined_aligned: list[np.ndarray] = []
    for drive, (frequency, reference, current) in rows.items():
        aligned = current * np.exp(
            2j * math.pi * frequency * delay / float(rate)
        )
        denominator = float(np.linalg.norm(reference) * np.linalg.norm(aligned))
        agreement = (
            float(abs(complex(np.vdot(reference, aligned))) / denominator)
            if denominator > 0.0
            else 0.0
        )
        relative_error = float(
            np.linalg.norm(aligned - reference) / np.linalg.norm(reference)
        )
        phase = float(np.angle(complex(np.vdot(reference, aligned))))
        drive_reports[drive] = {
            "tone_count": int(frequency.size),
            "complex_agreement": agreement,
            "relative_error": relative_error,
            "constant_phase_error_radians": phase,
        }
        combined_ref.append(reference)
        combined_aligned.append(aligned)

    reference_all = np.concatenate(combined_ref)
    aligned_all = np.concatenate(combined_aligned)
    denominator = float(np.linalg.norm(reference_all) * np.linalg.norm(aligned_all))
    overall_agreement = float(
        abs(complex(np.vdot(reference_all, aligned_all))) / denominator
    )
    overall_error = float(
        np.linalg.norm(aligned_all - reference_all) / np.linalg.norm(reference_all)
    )
    max_phase = max(
        abs(float(item["constant_phase_error_radians"]))
        for item in drive_reports.values()
    )
    passed = bool(
        overall_agreement >= PANEL_STITCH_MIN_COMPLEX_AGREEMENT
        and overall_error <= PANEL_STITCH_MAX_RELATIVE_ERROR
        and max_phase <= PANEL_STITCH_MAX_CONSTANT_PHASE_RADIANS
        and all(
            item["complex_agreement"] >= PANEL_STITCH_MIN_COMPLEX_AGREEMENT
            and item["relative_error"] <= PANEL_STITCH_MAX_RELATIVE_ERROR
            for item in drive_reports.values()
        )
    )
    return {
        "passed": passed,
        "shared_delay_samples": delay,
        "overlap_band_hz": [float(value) for value in overlap_band_hz],
        "overall_complex_agreement": overall_agreement,
        "overall_relative_error": overall_error,
        "maximum_constant_phase_error_radians": max_phase,
        "drives": drive_reports,
        "thresholds": {
            "minimum_complex_agreement": PANEL_STITCH_MIN_COMPLEX_AGREEMENT,
            "maximum_relative_error": PANEL_STITCH_MAX_RELATIVE_ERROR,
            "maximum_constant_phase_error_radians": (
                PANEL_STITCH_MAX_CONSTANT_PHASE_RADIANS
            ),
            "maximum_abs_delay_samples": limit,
        },
    }


def apply_shared_panel_delay(
    frequencies: Mapping[str, np.ndarray],
    transfers: Mapping[str, np.ndarray],
    *,
    delay_samples: float,
    sample_rate: int = 48_000,
) -> dict[str, np.ndarray]:
    """fit 결과의 공통 phase ramp를 두 drive에 동일하게 적용한다."""

    result: dict[str, np.ndarray] = {}
    for drive in ("noise", "cancel"):
        frequency = _finite_vector(
            frequencies[drive], complex_values=False, label=f"{drive} frequency"
        )
        transfer = _finite_vector(
            transfers[drive], complex_values=True, label=f"{drive} transfer"
        )
        if frequency.size != transfer.size:
            raise ValueError(f"{drive} frequency/transfer 길이가 다릅니다")
        result[drive] = transfer * np.exp(
            2j * math.pi * frequency * float(delay_samples) / float(sample_rate)
        )
    return result


def merge_stitched_panels(
    panels: Sequence[Mapping[str, Any]],
    *,
    drive: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """요청 panel 안 tone만 합치고 exact duplicate는 복소 평균한다.

    반환은 ``frequency, mean transfer, observations_per_frequency``다. clock pilot은 첫
    panel의 requested band에 속할 때만 plant fit에 들어간다.
    """

    frequencies: list[np.ndarray] = []
    transfers: list[np.ndarray] = []
    for index, panel in enumerate(panels):
        band = tuple(float(value) for value in panel["panel_band_hz"])
        frequency = _finite_vector(
            panel["frequencies"][drive],
            complex_values=False,
            label=f"panel {index} {drive} frequency",
        )
        transfer = _finite_vector(
            panel["transfers"][drive],
            complex_values=True,
            label=f"panel {index} {drive} transfer",
        )
        if frequency.size != transfer.size:
            raise ValueError(f"panel {index} {drive} 길이가 다릅니다")
        mask = (frequency >= band[0]) & (frequency <= band[1])
        if int(mask.sum()) < 8:
            raise ValueError(f"panel {index} {drive} requested-band tone이 부족합니다")
        frequencies.append(frequency[mask])
        transfers.append(transfer[mask])
    all_frequency = np.concatenate(frequencies)
    all_transfer = np.concatenate(transfers)
    unique, inverse, counts = np.unique(
        all_frequency, return_inverse=True, return_counts=True
    )
    sums = np.zeros(unique.size, dtype=np.complex128)
    np.add.at(sums, inverse, all_transfer)
    return unique, sums / counts, counts.astype(np.int64)


def estimate_bulk_delay_samples(
    frequencies_hz: np.ndarray,
    measured_transfer: np.ndarray,
    *,
    sample_rate: int = 48_000,
    minimum_delay_samples: float = 0.0,
    maximum_delay_samples: float = 4_800.0,
    coarse_resolution_samples: float = 0.25,
) -> float:
    """광대역 복소 전달의 matched phase slope를 메모리 제한 chunk로 찾는다."""

    frequency = _finite_vector(
        frequencies_hz, complex_values=False, label="bulk-delay frequency"
    )
    measured = _finite_vector(
        measured_transfer, complex_values=True, label="bulk-delay transfer"
    )
    if frequency.size != measured.size:
        raise ValueError("bulk-delay frequency/transfer 길이가 다릅니다")
    rate = int(sample_rate)
    lower = float(minimum_delay_samples)
    upper = float(maximum_delay_samples)
    step = float(coarse_resolution_samples)
    if not (
        rate > 0
        and math.isfinite(lower)
        and math.isfinite(upper)
        and math.isfinite(step)
        and 0.0 <= lower < upper
        and 0.0 < step <= 1.0
    ):
        raise ValueError("bulk-delay 탐색 범위/해상도가 잘못됐습니다")
    # 깊은 전달함수 notch의 불안정한 위상이 전체 점수를 지배하지 않게 amplitude를
    # p10..p90로 clip한 가중 phase vector를 사용한다.
    magnitude = np.abs(measured)
    positive = magnitude[magnitude > np.finfo(np.float64).tiny]
    if positive.size < 8:
        raise ValueError("bulk-delay에 유효한 complex tone이 부족합니다")
    low_weight, high_weight = np.percentile(positive, (10.0, 90.0))
    phase_vector = measured / np.maximum(magnitude, np.finfo(np.float64).tiny)
    weighted = phase_vector * np.clip(magnitude, low_weight, high_weight)
    omega = 2.0 * math.pi * frequency / float(rate)
    candidates = np.arange(lower, upper + 0.5 * step, step, dtype=np.float64)
    best_delay = float(candidates[0])
    best_score = -math.inf
    for begin in range(0, candidates.size, 256):
        chunk = candidates[begin : begin + 256]
        scores = np.abs(np.exp(1j * np.outer(chunk, omega)) @ weighted)
        index = int(np.argmax(scores))
        if float(scores[index]) > best_score:
            best_score = float(scores[index])
            best_delay = float(chunk[index])

    def objective(delay: float) -> float:
        return -float(abs(complex(np.sum(weighted * np.exp(1j * omega * delay)))))

    refine_lo = max(lower, best_delay - step)
    refine_hi = min(upper, best_delay + step)
    optimized = minimize_scalar(
        objective,
        bounds=(refine_lo, refine_hi),
        method="bounded",
        options={"xatol": 1.0e-8},
    )
    if not optimized.success or not math.isfinite(float(optimized.x)):
        raise ValueError("bulk-delay fractional refinement가 실패했습니다")
    return float(optimized.x)


def fit_real_compact_fir(
    frequencies_hz: np.ndarray,
    measured_transfer: np.ndarray,
    *,
    effective_delay_samples: int,
    fir_length: int,
    sample_rate: int = 48_000,
    ridge_relative: float = 1.0e-8,
) -> dict[str, Any]:
    """irregular broadband tone grid에서 하나의 real compact FIR을 적합한다."""

    frequency = _finite_vector(
        frequencies_hz, complex_values=False, label="compact frequency"
    )
    measured = _finite_vector(
        measured_transfer, complex_values=True, label="compact measured transfer"
    )
    length = int(fir_length)
    delay = int(effective_delay_samples)
    rate = int(sample_rate)
    ridge = float(ridge_relative)
    if frequency.size != measured.size or np.any(np.diff(frequency) <= 0.0):
        raise ValueError("compact frequency/transfer 길이 또는 정렬이 잘못됐습니다")
    if length <= 0 or delay < 0 or rate <= 0:
        raise ValueError("compact FIR length/delay/sample rate가 잘못됐습니다")
    if 2 * frequency.size < length:
        raise ValueError(
            f"real FIR {length} taps를 식별할 독립 복소 tone이 부족합니다: {frequency.size}"
        )
    if not math.isfinite(ridge) or not 0.0 <= ridge <= 1.0e-2:
        raise ValueError("ridge_relative가 유효하지 않습니다")
    sample_index = delay + np.arange(length, dtype=np.float64)
    complex_basis = np.exp(
        -2j * math.pi * np.outer(frequency, sample_index) / float(rate)
    )
    design = np.concatenate((complex_basis.real, complex_basis.imag), axis=0)
    target = np.concatenate((measured.real, measured.imag), axis=0)
    gram = design.T @ design
    scale = float(np.trace(gram) / length)
    regularization = ridge * scale
    coefficients = np.linalg.solve(
        gram + regularization * np.eye(length, dtype=np.float64),
        design.T @ target,
    )
    reconstructed = complex_basis @ coefficients
    denominator = float(np.linalg.norm(measured) * np.linalg.norm(reconstructed))
    agreement = float(
        abs(complex(np.vdot(measured, reconstructed))) / denominator
    )
    relative_error = float(
        np.linalg.norm(reconstructed - measured) / np.linalg.norm(measured)
    )
    return {
        "fir": coefficients.astype(np.float32),
        "reconstructed_transfer": reconstructed,
        "complex_agreement": agreement,
        "relative_error": relative_error,
        "tone_count": int(frequency.size),
        "fir_length": length,
        "effective_delay_samples": delay,
        "ridge_relative": ridge,
        "ridge_absolute": regularization,
        "passed": bool(agreement >= 0.995 and relative_error <= 0.10),
    }


def band_roundtrip_metrics(
    frequencies_hz: np.ndarray,
    measured_transfer: np.ndarray,
    reconstructed_transfer: np.ndarray,
    *,
    subbands_hz: Sequence[Sequence[float]],
    minimum_complex_agreement: float = 0.995,
    maximum_relative_error: float = 0.10,
) -> tuple[dict[str, Any], ...]:
    """모든 제어 subband를 독립적으로 검사해 평균이 실패를 숨기지 못하게 한다."""

    frequency = _finite_vector(
        frequencies_hz, complex_values=False, label="roundtrip frequency"
    )
    measured = _finite_vector(
        measured_transfer, complex_values=True, label="roundtrip measured"
    )
    reconstructed = _finite_vector(
        reconstructed_transfer, complex_values=True, label="roundtrip reconstructed"
    )
    if not (frequency.size == measured.size == reconstructed.size):
        raise ValueError("roundtrip 배열 길이가 다릅니다")
    minimum_agreement = float(minimum_complex_agreement)
    maximum_error = float(maximum_relative_error)
    if not (
        math.isfinite(minimum_agreement)
        and math.isfinite(maximum_error)
        and 0.0 < minimum_agreement <= 1.0
        and 0.0 <= maximum_error < 1.0
    ):
        raise ValueError("roundtrip agreement/error threshold가 잘못됐습니다")
    rows: list[dict[str, Any]] = []
    for raw in subbands_hz:
        lo, hi = (float(value) for value in raw)
        mask = (frequency >= lo) & (frequency <= hi)
        if int(mask.sum()) < 8:
            raise ValueError(f"roundtrip {lo:g}-{hi:g}Hz tone이 부족합니다")
        target = measured[mask]
        estimate = reconstructed[mask]
        denominator = float(np.linalg.norm(target) * np.linalg.norm(estimate))
        agreement = float(abs(complex(np.vdot(target, estimate))) / denominator)
        error = float(np.linalg.norm(estimate - target) / np.linalg.norm(target))
        rows.append(
            {
                "band_hz": [lo, hi],
                "tone_count": int(mask.sum()),
                "complex_agreement": agreement,
                "relative_error": error,
                "passed": bool(
                    agreement >= minimum_agreement and error <= maximum_error
                ),
            }
        )
    return tuple(rows)


__all__ = [
    "COMPACT_IDENTIFIABILITY_SCHEMA",
    "COMPACT_MAX_CONDITION_NUMBER",
    "PANEL_STITCH_MAX_ABS_DELAY_SAMPLES",
    "PANEL_STITCH_MAX_CONSTANT_PHASE_RADIANS",
    "PANEL_STITCH_MAX_RELATIVE_ERROR",
    "PANEL_STITCH_MIN_COMPLEX_AGREEMENT",
    "apply_shared_panel_delay",
    "band_roundtrip_metrics",
    "compact_fir_identifiability_receipt",
    "fit_real_compact_fir",
    "fit_shared_panel_delay",
    "estimate_bulk_delay_samples",
    "merge_stitched_panels",
]
