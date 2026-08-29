"""v5를 변경하지 않는 time-separated 고-SNR clock/PE signal-only v6.

이 모듈은 오디오 장치와 파일을 열지 않는다. actual-int16 신호, immutable plan 및
이미 메모리에 있는 capture의 clock 증거만 계산한다. v6 raw publisher/live authority는
별도 단계이며 이 모듈의 PASS만으로 물리 P/S 또는 학습 authority가 생기지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve

from .fullband_causal_v5 import (
    BLOCK,
    CONDITION_AUDIT_SUPPORT,
    CYCLIC_PREFIX,
    CYCLIC_SUFFIX,
    FS,
    PATH_CHANNEL,
    PERIOD,
    ROLES,
    SEEDS,
    SLOT_FRAMES,
    _exact_condition_audit_with_shifts_v5,
    near_white_period,
)
from .measurement_level import expected_meter_output_pcm
from .fullband_live_raw_v5 import LIVE_RAW_SCHEMA_V6


SCHEMA = "fullband_causal_time_separated_clock_v6"
RAW_DEFAULT = "results/fullband_causal_v6/raw_capture.npz"
CLOCK_BINS = (109, 137, 181, 233, 277, 314, 359, 401)
_CLOCK_PHASES = (
    1.84327308785262,
    6.223928570162518,
    0.7215530936159469,
    3.730785893035909,
    1.2305070031650789,
    6.211696881149294,
    4.022162803325311,
    6.101133374883352,
)
_CLOCK_FLOAT_SCALE = 53.18
CLOCK_PREFIX = PERIOD // 2
CLOCK_SUFFIX = PERIOD // 2
CLOCK_REPEATS = 2
CLOCK_BLOCK_FRAMES = CLOCK_PREFIX + CLOCK_REPEATS * PERIOD + CLOCK_SUFFIX
CLOCK_EPOCHS = ("fit_pre_0", "fit_pre_1", "fit_pre_2", "terminal_post_holdout")
TOTAL_FRAMES = 36 * PERIOD
PE_PEAK_PCM = 49
CLOCK_PEAK_LIMIT_PCM = 98
METER_POWER_MIN_DB = -0.25
METER_POWER_MAX_DB = 0.0
MIN_PREOPTIMIZER_LINE_SNR_DB = 20.0
MIN_REPEAT_LINE_SNR_DB = 20.0
MIN_REPEAT_COMPLEX_AGREEMENT = 0.995
MAX_CLOCK_PPM = 1_000.0
GLOBAL_GRID_STEP_PPM = 1.0
MIN_UNIQUE_BASIN_OBJECTIVE_RATIO = 4.0
MAX_VIEW_ENDPOINT_DISAGREEMENT_SAMPLES = 0.05
MAX_TERMINAL_PHASE_ERROR_SAMPLES = 0.06755189029558946
MAX_CUBIC_LINEAR_ENDPOINT_DISAGREEMENT_SAMPLES = 0.06755189029558945
PLAN_LOCAL_GUARD_OFFSETS = (-6, -5, -4, 4, 5, 6)
PREOPTIMIZER_TARGET_OFFSETS = (-2, -1, 0, 1, 2)
PREOPTIMIZER_GUARD_OFFSETS = (
    -10,
    -9,
    -8,
    -7,
    -6,
    6,
    7,
    8,
    9,
    10,
)


class V6ClockAdmissionError(ValueError):
    """실패 단계와 이미 계산된 증거를 보존하는 v6 clock fail-closed 예외."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        optimizer_started: bool,
        available_receipt: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = str(stage)
        self.optimizer_started = bool(optimizer_started)
        self.available_receipt = dict(available_receipt or {})


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _safe_raw_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.suffix != ".npz"
        or not path.parts
        or path.parts[0] != "results"
    ):
        raise ValueError("v6 raw path는 results/ 아래 lexical .npz 상대경로여야 합니다")
    return path.as_posix()


def _clock_period() -> np.ndarray:
    """고정 8-line을 가진 actual-int16 full-period clipped multisine."""

    sample = np.arange(PERIOD, dtype=np.float64)[:, None]
    bins = np.asarray(CLOCK_BINS, dtype=np.float64)[None, :]
    phases = np.asarray(_CLOCK_PHASES, dtype=np.float64)[None, :]
    waveform = np.sum(
        np.cos(2.0 * np.pi * sample * bins / PERIOD + phases), axis=1
    )
    waveform /= math.sqrt(float(np.mean(waveform**2)))
    actual = np.rint(
        np.clip(
            waveform * _CLOCK_FLOAT_SCALE,
            -CLOCK_PEAK_LIMIT_PCM,
            CLOCK_PEAK_LIMIT_PCM,
        )
    ).astype(np.int16)
    return actual


def _fundamental_period(value: np.ndarray) -> int:
    signal = np.asarray(value)
    for candidate in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
        if np.array_equal(signal, np.roll(signal, candidate)):
            return candidate
    return PERIOD


def build_plan_v6(
    *, raw_session_relative_path: str = RAW_DEFAULT
) -> tuple[dict[str, Any], np.ndarray]:
    """v5와 side-by-side인 exact 24.576초 signal-only plan을 만든다."""

    raw_path = _safe_raw_path(raw_session_relative_path)
    pilot = _clock_period()
    spectrum = np.fft.rfft(pilot.astype(np.float64))
    bins = np.asarray(CLOCK_BINS, dtype=np.int64)
    meter = expected_meter_output_pcm(noise_channel=0).astype(np.float64)
    meter_power = float(np.sum(np.mean((meter / 32768.0) ** 2, axis=0)))
    pilot_power = float(np.mean((pilot.astype(np.float64) / 32768.0) ** 2))
    relative_db = 10.0 * math.log10(pilot_power / meter_power)
    peak = int(np.max(np.abs(pilot.astype(np.int32))))
    if peak > CLOCK_PEAK_LIMIT_PCM:
        raise AssertionError("v6 clock pilot peak가 98 PCM을 초과합니다")
    if not METER_POWER_MIN_DB <= relative_db <= METER_POWER_MAX_DB:
        raise AssertionError("v6 path-only pilot power가 meter-relative 계약 밖입니다")
    if len(CLOCK_BINS) != 8 or not all(152.0 <= b * FS / PERIOD <= 600.0 for b in CLOCK_BINS):
        raise AssertionError("v6 clock line은 exact 8개 152..600 Hz여야 합니다")
    gcd_bins = math.gcd(PERIOD, *CLOCK_BINS)
    gcd_differences = math.gcd(*np.diff(bins).tolist())
    fundamental = _fundamental_period(pilot)
    if gcd_bins != 1 or gcd_differences != 1 or fundamental != PERIOD:
        raise AssertionError("v6 clock pilot가 full-period/no-short-comb가 아닙니다")

    guard_bins = np.asarray(
        [b + offset for b in CLOCK_BINS for offset in PLAN_LOCAL_GUARD_OFFSETS],
        dtype=np.int64,
    )
    line_to_guard_db = 20.0 * math.log10(
        float(np.min(np.abs(spectrum[bins])))
        / max(float(np.max(np.abs(spectrum[guard_bins]))), np.finfo(np.float64).tiny)
    )

    parts: list[np.ndarray] = []
    layout: list[dict[str, Any]] = []
    cursor = 0

    def add_clock_epoch(epoch: str, *, terminal: bool) -> None:
        nonlocal cursor
        for path in ("primary", "secondary"):
            channel = PATH_CHANNEL[path]
            block = np.zeros((CLOCK_BLOCK_FRAMES, 2), dtype=np.int16)
            block[:CLOCK_PREFIX, channel] = pilot[-CLOCK_PREFIX:]
            first = CLOCK_PREFIX
            block[first : first + PERIOD, channel] = pilot
            second = first + PERIOD
            block[second : second + PERIOD, channel] = pilot
            block[second + PERIOD :, channel] = pilot[:CLOCK_SUFFIX]
            parts.append(block)
            layout.append(
                {
                    "kind": "clock_block",
                    "epoch": epoch,
                    "stage": "terminal_validation" if terminal else "preterminal_fit",
                    "path": path,
                    "active_channel": channel,
                    "opposite_channel_exact_zero": True,
                    "opposite_channel_actual_max_abs_dft": 0.0,
                    "start_frame": cursor,
                    "stop_frame": cursor + CLOCK_BLOCK_FRAMES,
                    "prefix_samples": CLOCK_PREFIX,
                    "central_repeat_starts": [cursor + first, cursor + second],
                    "central_repeat_samples": PERIOD,
                    "suffix_samples": CLOCK_SUFFIX,
                    "actual_path_only_pcm_sha256": _array_sha256(block),
                }
            )
            cursor += CLOCK_BLOCK_FRAMES

    def add_pe_role(role: str) -> None:
        nonlocal cursor
        for path in ("primary", "secondary"):
            channel = PATH_CHANNEL[path]
            pe = near_white_period(SEEDS[(path, role)])
            slot = np.zeros((SLOT_FRAMES, 2), dtype=np.int16)
            slot[:CYCLIC_PREFIX, channel] = pe[-CYCLIC_PREFIX:]
            central = CYCLIC_PREFIX
            slot[central : central + PERIOD, channel] = pe
            slot[central + PERIOD :, channel] = pe[:CYCLIC_SUFFIX]
            slot_power = float(
                np.sum(np.mean((slot.astype(np.float64) / 32768.0) ** 2, axis=0))
            )
            slot_vs_meter_db = 10.0 * math.log10(slot_power / meter_power)
            if slot_power > meter_power:
                raise AssertionError("v6 PE slot actual total power가 official meter를 초과합니다")
            parts.append(slot)
            layout.append(
                {
                    "kind": "near_white_pe_slot",
                    "role": role,
                    "path": path,
                    "active_channel": channel,
                    "opposite_channel_exact_zero": True,
                    "start_frame": cursor,
                    "stop_frame": cursor + SLOT_FRAMES,
                    "central_start_frame": cursor + central,
                    "central_stop_frame": cursor + central + PERIOD,
                    "pre_boundary_exclusion_samples": CYCLIC_PREFIX,
                    "post_boundary_exclusion_samples": CYCLIC_SUFFIX,
                    "peak_pcm": PE_PEAK_PCM,
                    "continuous_clock_pilot_present": False,
                    "payload_pcm_sha256": _array_sha256(pe),
                    "actual_submitted_total_power": slot_power,
                    "official_meter_total_power": meter_power,
                    "actual_submitted_vs_meter_db": slot_vs_meter_db,
                    "actual_submitted_not_above_meter": True,
                }
            )
            cursor += SLOT_FRAMES

    add_clock_epoch(CLOCK_EPOCHS[0], terminal=False)
    add_pe_role("fit_a")
    add_clock_epoch(CLOCK_EPOCHS[1], terminal=False)
    add_pe_role("fit_b")
    add_clock_epoch(CLOCK_EPOCHS[2], terminal=False)
    add_pe_role("holdout")
    add_clock_epoch(CLOCK_EPOCHS[3], terminal=True)

    submitted = np.concatenate(parts, axis=0)
    if cursor != TOTAL_FRAMES or submitted.shape != (TOTAL_FRAMES, 2):
        raise AssertionError("v6 plan은 exact 1,179,648 frames여야 합니다")
    if len(submitted) % BLOCK:
        raise AssertionError("v6 plan은 256-frame aligned여야 합니다")
    if int(np.max(np.abs(submitted.astype(np.int32)))) > CLOCK_PEAK_LIMIT_PCM:
        raise AssertionError("v6 submitted peak가 계약 밖입니다")
    holdout_rows = [
        [int(row["start_frame"]), int(row["stop_frame"])]
        for row in layout
        if row.get("role") == "holdout"
    ]
    clock_fit_rows = [
        row["epoch"]
        for row in layout
        if row["kind"] == "clock_block" and row["stage"] == "preterminal_fit"
    ]
    terminal_rows = [
        row["epoch"]
        for row in layout
        if row["kind"] == "clock_block" and row["stage"] == "terminal_validation"
    ]
    pe_power_rows = [
        row for row in layout if row["kind"] == "near_white_pe_slot"
    ]
    worst_pe_vs_meter_db = max(
        float(row["actual_submitted_vs_meter_db"]) for row in pe_power_rows
    )
    plan: dict[str, Any] = {
        "schema": SCHEMA,
        "role": "pure_signal_only_no_audio_no_publisher",
        "sample_rate": FS,
        "block_size": BLOCK,
        "period_samples": PERIOD,
        "duration_seconds": len(submitted) / FS,
        "raw_session_relative_path": raw_path,
        "publisher_contract": {
            "raw_session_relative_path": raw_path,
            "raw_npz_schema": LIVE_RAW_SCHEMA_V6,
            "must_not_exist_before_capture": True,
            "capture_writer_bound": False,
            "live_capture_authority": None,
            "role": "signal_only_no_raw_writer_in_this_module",
        },
        "v5_builder_schema_and_default_raw_preserved": True,
        "live_authority": None,
        "canonical_training_eligible": False,
        "clock_excitation": {
            "path_activation": "time_separated_one_path_only",
            "fixed_line_bins": list(CLOCK_BINS),
            "fixed_line_frequencies_hz": [b * FS / PERIOD for b in CLOCK_BINS],
            "line_selection": "predeclared_from_pre_v6_observability_not_v6_raw",
            "line_count": 8,
            "bin_gcd_with_period": gcd_bins,
            "bin_difference_gcd": gcd_differences,
            "actual_fundamental_period_samples": fundamental,
            "actual_selected_line_min_abs_dft": float(np.min(np.abs(spectrum[bins]))),
            "actual_local_guard_max_abs_dft": float(np.max(np.abs(spectrum[guard_bins]))),
            "actual_line_to_guard_minimum_db": line_to_guard_db,
            "actual_int16_period_pcm_sha256": _array_sha256(pilot),
            "peak_pcm": peak,
            "actual_path_only_total_power": pilot_power,
            "official_meter_total_power": meter_power,
            "path_only_vs_meter_db": relative_db,
            "meter_relative_allowed_db": [METER_POWER_MIN_DB, METER_POWER_MAX_DB],
            "opposite_path_exact_zero_required": True,
        },
        "clock_estimator_contract": {
            "search_bounds_ppm": [-MAX_CLOCK_PPM, MAX_CLOCK_PPM],
            "global_grid_max_step_ppm": GLOBAL_GRID_STEP_PPM,
            "all_local_basins_refined": True,
            "boundary_minimum_rejected": True,
            "minimum_unique_basin_objective_ratio": MIN_UNIQUE_BASIN_OBJECTIVE_RATIO,
            "preoptimizer_fixed_line_snr_min_db": MIN_PREOPTIMIZER_LINE_SNR_DB,
            "post_q_repeat_line_snr_min_db": MIN_REPEAT_LINE_SNR_DB,
            "post_q_repeat_complex_agreement_min": MIN_REPEAT_COMPLEX_AGREEMENT,
            "view_endpoint_disagreement_max_samples": MAX_VIEW_ENDPOINT_DISAGREEMENT_SAMPLES,
            "terminal_phase_error_max_samples": MAX_TERMINAL_PHASE_ERROR_SAMPLES,
            "cubic_linear_endpoint_disagreement_max_samples": (
                MAX_CUBIC_LINEAR_ENDPOINT_DISAGREEMENT_SAMPLES
            ),
        },
        "active_block_power_contract": {
            "power_definition": "sum_channel_mean_square_actual_submitted_int16_normalized_by_32768",
            "official_meter_total_power": meter_power,
            "clock_path_only_vs_meter_db": relative_db,
            "worst_pe_slot_vs_meter_db": worst_pe_vs_meter_db,
            "pe_slot_count": len(pe_power_rows),
            "all_active_blocks_not_above_meter": bool(
                relative_db <= 0.0
                and all(row["actual_submitted_not_above_meter"] for row in pe_power_rows)
            ),
        },
        "holdout_access_policy": {
            "clock_fit_epochs": list(CLOCK_EPOCHS[:3]),
            "clock_terminal_validation_epoch": CLOCK_EPOCHS[3],
            "operator_holdout_frame_ranges": holdout_rows,
            "operator_holdout_used_for_clock_fit_snr_basin_or_selection": False,
            "terminal_clock_block_is_disjoint_from_operator_holdout": True,
            "clock_fit_epoch_rows": clock_fit_rows,
            "terminal_epoch_rows": terminal_rows,
        },
        "layout": layout,
        "actual_submitted_shape": list(submitted.shape),
        "actual_submitted_dtype": submitted.dtype.str,
        "actual_submitted_peak_pcm": int(np.max(np.abs(submitted.astype(np.int32)))),
        "actual_submitted_pcm_sha256": _array_sha256(submitted),
    }
    plan["canonical_payload_sha256"] = _payload_sha256(plan)
    return plan, submitted


def preoptimizer_spectral_line_admission_v6(
    repeats: np.ndarray,
    *,
    fixed_bins: Sequence[int] = CLOCK_BINS,
    minimum_snr_db: float = MIN_PREOPTIMIZER_LINE_SNR_DB,
) -> dict[str, Any]:
    """q 추정 전에 ±2 target와 고정 local guard로 보수적 SNR을 검사한다."""

    value = np.asarray(repeats, dtype=np.float64)
    if value.ndim != 4 or value.shape[1:] != (2, PERIOD, 2):
        raise ValueError("pre-optimizer repeats는 [block,2,PERIOD,ERR_REF=2]여야 합니다")
    if not np.all(np.isfinite(value)):
        raise ValueError("pre-optimizer repeats에 non-finite가 있습니다")
    bins = np.asarray(tuple(int(v) for v in fixed_bins), dtype=np.int64)
    if bins.shape != (8,):
        raise ValueError("pre-optimizer fixed line은 exact 8개여야 합니다")
    if np.any(bins <= 10) or np.any(bins >= PERIOD // 2 - 10):
        raise ValueError("pre-optimizer fixed line의 target/guard support가 유효하지 않습니다")
    window = np.hanning(PERIOD).astype(np.float64)
    spectra = np.fft.rfft(value * window[None, None, :, None], axis=2)
    rows: list[dict[str, Any]] = []
    for block in range(value.shape[0]):
        for repeat in range(2):
            for mic in range(2):
                for line in bins:
                    target = np.asarray(
                        [line + offset for offset in PREOPTIMIZER_TARGET_OFFSETS],
                        dtype=np.int64,
                    )
                    guard = np.asarray(
                        [line + offset for offset in PREOPTIMIZER_GUARD_OFFSETS],
                        dtype=np.int64,
                    )
                    target_power = float(np.mean(np.abs(spectra[block, repeat, target, mic]) ** 2))
                    background_power = float(np.mean(np.abs(spectra[block, repeat, guard, mic]) ** 2))
                    snr_db = 10.0 * math.log10(
                        max(target_power, np.finfo(np.float64).tiny)
                        / max(background_power, np.finfo(np.float64).tiny)
                    )
                    rows.append(
                        {
                            "block": block,
                            "repeat": repeat,
                            "microphone": mic,
                            "line_bin": int(line),
                            "snr_db": snr_db,
                            "passed": bool(snr_db >= minimum_snr_db),
                        }
                    )
    passed = bool(rows and all(row["passed"] for row in rows))
    receipt = {
        "schema": "fullband_v6_preoptimizer_spectral_line_admission_v1",
        "fixed_line_bins": bins.tolist(),
        "window": "full_P_symmetric_Hann_numpy_hanning",
        "window_samples": PERIOD,
        "target_bin_offsets": list(PREOPTIMIZER_TARGET_OFFSETS),
        "local_guard_offsets": list(PREOPTIMIZER_GUARD_OFFSETS),
        "background_semantics": "conservative_local_energy_includes_actual_pcm_quantization_leakage",
        "minimum_snr_db": float(minimum_snr_db),
        "minimum_observed_snr_db": min(row["snr_db"] for row in rows),
        "rows": rows,
        "optimizer_was_run": False,
        "passed": passed,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    if not passed:
        raise V6ClockAdmissionError(
            "pre-optimizer clock SNR admission 실패",
            stage="preoptimizer_snr_admission",
            optimizer_started=False,
            available_receipt=receipt,
        )
    return receipt


def exact_condition_audit_v6(
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    *,
    support: int = CONDITION_AUDIT_SUPPORT,
) -> dict[str, Any]:
    """exact v6 builder/PCM의 pure time-separated PE 1024 Gram을 계산한다."""

    raw_path = str(plan.get("raw_session_relative_path", RAW_DEFAULT))
    expected_plan, expected_submitted = build_plan_v6(
        raw_session_relative_path=raw_path
    )
    submitted = np.asarray(submitted_pcm)
    if dict(plan) != expected_plan or not np.array_equal(submitted, expected_submitted):
        raise ValueError("condition audit 입력이 exact v6 plan/PCM이 아닙니다")
    receipt = _exact_condition_audit_with_shifts_v5(
        expected_plan,
        expected_submitted,
        support=support,
        zeros_by_path=(0, 0),
        schema="fullband_causal_exact_gram_condition_v6",
    )
    if receipt["signal_plan_payload_sha256"] != expected_plan["canonical_payload_sha256"]:
        raise AssertionError("v6 exact Gram receipt가 signal plan SHA와 결속되지 않았습니다")
    return _canonicalize_condition_receipt_v6(receipt)


def _canonicalize_condition_receipt_v6(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """BLAS reduction 순서와 무관한 v6 exact-condition receipt bytes를 만든다.

    gate 해상도(조건수 20, quadratic error 1e-10)보다 충분히 촘촘한 13 significant
    digits를 유지하되 1e-10 미만 crosscheck 오차는 1e-12 absolute grid로 고정한다.
    원 Gram/eigensolve의 PASS 판정이 끝난 뒤 receipt 직렬화만 canonicalize한다.
    """

    def canonical(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): canonical(item) for key, item in value.items()}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        if isinstance(value, tuple):
            return [canonical(item) for item in value]
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("v6 condition receipt에 non-finite float가 있습니다")
            if abs(value) < 1.0e-10:
                return float(round(value, 12))
            return float(format(value, ".13g"))
        return value

    payload = {
        key: value for key, value in receipt.items() if key != "canonical_payload_sha256"
    }
    result = canonical(payload)
    result["numeric_canonicalization"] = {
        "schema": "fullband_v6_condition_numeric_canonicalization_v1",
        "non_tiny_float_significant_digits": 13,
        "absolute_grid_decimals_below_1e_minus_10": 12,
        "applied_after_unrounded_gate_decision": True,
        "purpose": "BLAS_reduction_order_independent_receipt_bytes",
    }
    result["canonical_payload_sha256"] = _payload_sha256(result)
    return result


def global_grid_basin_search_v6(
    objective: Callable[[float], float],
    *,
    grid_step_ppm: float = GLOBAL_GRID_STEP_PPM,
    require_unique: bool = True,
) -> tuple[float, dict[str, Any]]:
    """전체 ±1000 ppm grid 후 모든 interior basin을 deterministic refine한다."""

    step = float(grid_step_ppm)
    if not 0.0 < step <= GLOBAL_GRID_STEP_PPM:
        raise ValueError("global clock grid step은 0<step<=1 ppm이어야 합니다")
    count = int(round(2.0 * MAX_CLOCK_PPM / step))
    ppm_grid = np.linspace(-MAX_CLOCK_PPM, MAX_CLOCK_PPM, count + 1)
    ratio_grid = 1.0 + ppm_grid * 1.0e-6
    values = np.asarray([objective(float(q)) for q in ratio_grid], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("global clock objective는 finite non-negative여야 합니다")
    if int(np.argmin(values)) in {0, len(values) - 1}:
        raise ValueError("global clock optimum이 search boundary입니다")
    local = np.flatnonzero(
        (values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:])
    ) + 1
    if local.size == 0:
        raise ValueError("global clock objective에 interior basin이 없습니다")
    basins: list[dict[str, Any]] = []
    for index in local:
        refined = minimize_scalar(
            objective,
            bounds=(float(ratio_grid[index - 1]), float(ratio_grid[index + 1])),
            method="bounded",
            options={"xatol": 1.0e-12},
        )
        if not refined.success or not np.isfinite(refined.fun) or refined.fun < 0.0:
            raise ValueError("global clock basin refinement가 실패했습니다")
        basins.append(
            {
                "grid_index": int(index),
                "grid_ppm": float(ppm_grid[index]),
                "refined_ratio": float(refined.x),
                "refined_ppm": (float(refined.x) - 1.0) * 1.0e6,
                "objective": float(refined.fun),
            }
        )
    basins.sort(key=lambda row: row["objective"])
    best = basins[0]
    if min(float(values[0]), float(values[-1])) <= best["objective"]:
        raise ValueError("global clock optimum이 search boundary입니다")
    runner = basins[1]["objective"] if len(basins) > 1 else None
    ratio = (
        float(runner) / max(best["objective"], np.finfo(np.float64).tiny)
        if runner is not None
        else math.inf
    )
    unique = bool(ratio >= MIN_UNIQUE_BASIN_OBJECTIVE_RATIO)
    if require_unique and not unique:
        raise ValueError("global clock objective가 multimodal ambiguous입니다")
    receipt = {
        "schema": "fullband_v6_global_grid_all_basin_refinement_v1",
        "bounds_ppm": [-MAX_CLOCK_PPM, MAX_CLOCK_PPM],
        "grid_step_ppm": step,
        "grid_points": len(ratio_grid),
        "all_interior_basins_refined": True,
        "boundary_minimum_rejected": True,
        "basins": basins,
        "selected_ratio": best["refined_ratio"],
        "runner_up_to_best_objective_ratio": ratio if runner is not None else None,
        "minimum_unique_basin_objective_ratio": MIN_UNIQUE_BASIN_OBJECTIVE_RATIO,
        "unique_basin_passed": unique,
    }
    return float(best["refined_ratio"]), receipt


def _clock_rows(plan: Mapping[str, Any], *, terminal: bool) -> list[Mapping[str, Any]]:
    stage = "terminal_validation" if terminal else "preterminal_fit"
    return [
        row
        for row in plan["layout"]
        if row.get("kind") == "clock_block" and row.get("stage") == stage
    ]


def _nominal_repeats(captured: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.stack(
        [
            np.stack(
                [captured[start : start + PERIOD] for start in row["central_repeat_starts"]]
            )
            for row in rows
        ]
    )


def _phase_observations(
    captured: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    pilot_spectrum: np.ndarray,
) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    bins = np.asarray(CLOCK_BINS, dtype=np.int64)
    result: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        path: {} for path in PATH_CHANNEL
    }
    for path in PATH_CHANNEL:
        selected = [row for row in rows if row["path"] == path]
        starts = np.asarray(
            [start for row in selected for start in row["central_repeat_starts"]],
            dtype=np.float64,
        )
        centers = starts + (PERIOD - 1.0) / 2.0
        transfer_by_mic = []
        for start in starts.astype(np.int64):
            spectrum = np.fft.rfft(captured[start : start + PERIOD], axis=0)
            transfer_by_mic.append(spectrum[bins] / pilot_spectrum[bins, None])
        transfer = np.asarray(transfer_by_mic)
        for mic in range(2):
            result[path][mic] = (transfer[:, :, mic], centers)
    return result


def _circular_phase_objective(
    observations: Sequence[tuple[np.ndarray, np.ndarray]], ratio: float
) -> float:
    bins = np.asarray(CLOCK_BINS, dtype=np.float64)
    total = 0.0
    for transfer, centers in observations:
        unit = transfer / np.maximum(np.abs(transfer), np.finfo(np.float64).tiny)
        correction = np.exp(
            -2j
            * np.pi
            * centers[:, None]
            * bins[None, :]
            * (float(ratio) - 1.0)
            / PERIOD
        )
        corrected = unit * correction
        mean = np.mean(corrected, axis=0, keepdims=True)
        mean /= np.maximum(np.abs(mean), np.finfo(np.float64).tiny)
        total += float(np.mean(np.abs(corrected - mean) ** 2))
    return total


def _bounded_interpolators(
    captured: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    kind: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    """각 clock block CP/CS 안에서만 owned local interpolator를 만든다."""

    if kind not in {"cubic", "linear"}:
        raise ValueError("clock interpolation은 cubic/linear만 허용합니다")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    q_min = 1.0 - MAX_CLOCK_PPM * 1.0e-6
    q_max = 1.0 + MAX_CLOCK_PPM * 1.0e-6
    for row in rows:
        first = int(row["central_repeat_starts"][0])
        stop = int(row["central_repeat_starts"][1]) + PERIOD
        queries = (first / q_min, first / q_max, (stop - 1) / q_min, (stop - 1) / q_max)
        margin = 3 if kind == "cubic" else 1
        lower = max(int(row["start_frame"]), int(math.floor(min(queries))) - margin)
        upper = min(int(row["stop_frame"]), int(math.ceil(max(queries))) + margin + 1)
        local = np.array(captured[lower:upper], dtype=np.float64, copy=True, order="C")
        grid = np.arange(lower, upper, dtype=np.float64)
        if kind == "cubic":
            interpolators: list[Callable[[np.ndarray], np.ndarray]] = [
                CubicSpline(grid, local[:, mic], extrapolate=False) for mic in range(2)
            ]
        else:
            interpolators = [
                lambda query, mic=mic, grid=grid, local=local: np.interp(
                    query, grid, local[:, mic]
                )
                for mic in range(2)
            ]
        result[(str(row["epoch"]), str(row["path"]))] = {
            "interpolators": interpolators,
            "lower": lower,
            "upper": upper,
            "owned_local_capture_sha256": _array_sha256(local),
        }
    return result


def _corrected_transfers(
    accessors: Mapping[tuple[str, str], Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    ratio: float,
    pilot_spectrum: np.ndarray,
) -> dict[str, np.ndarray]:
    bins = np.asarray(CLOCK_BINS, dtype=np.int64)
    result: dict[str, np.ndarray] = {}
    for path in PATH_CHANNEL:
        block_values = []
        for row in rows:
            if row["path"] != path:
                continue
            accessor = accessors[(str(row["epoch"]), str(row["path"]))]
            interpolators = accessor["interpolators"]
            repeats = []
            for start in row["central_repeat_starts"]:
                query = np.arange(start, start + PERIOD, dtype=np.float64) / float(ratio)
                if query[0] < int(accessor["lower"]) or query[-1] > int(accessor["upper"]) - 1:
                    raise ValueError("q-corrected clock repeat가 capture support 밖입니다")
                values = np.column_stack([interpolator(query) for interpolator in interpolators])
                repeats.append(
                    np.fft.rfft(values, axis=0)[bins] / pilot_spectrum[bins, None]
                )
            block_values.append(np.stack(repeats))
        result[path] = np.stack(block_values)
    return result


def _transfer_consistency_objective(transfers: Mapping[str, np.ndarray]) -> float:
    total = 0.0
    for value in transfers.values():
        flattened = value.reshape(-1, len(CLOCK_BINS), 2)
        unit = flattened / np.maximum(np.abs(flattened), np.finfo(np.float64).tiny)
        mean = np.mean(unit, axis=0, keepdims=True)
        mean /= np.maximum(np.abs(mean), np.finfo(np.float64).tiny)
        total += float(np.mean(np.abs(unit - mean) ** 2))
    return total


def _post_q_repeat_receipt(
    transfers: Mapping[str, np.ndarray], *, stage: str
) -> dict[str, Any]:
    rows = []
    for path, blocks in transfers.items():
        for block in range(blocks.shape[0]):
            for mic in range(2):
                first = blocks[block, 0, :, mic]
                second = blocks[block, 1, :, mic]
                mean = (first + second) / 2.0
                half_difference = (first - second) / 2.0
                snr = 10.0 * np.log10(
                    np.maximum(np.abs(mean) ** 2, np.finfo(np.float64).tiny)
                    / np.maximum(np.abs(half_difference) ** 2, np.finfo(np.float64).tiny)
                )
                agreement = float(
                    abs(np.vdot(first, second))
                    / max(
                        float(np.linalg.norm(first) * np.linalg.norm(second)),
                        np.finfo(np.float64).tiny,
                    )
                )
                passed = bool(
                    np.all(snr >= MIN_REPEAT_LINE_SNR_DB)
                    and agreement >= MIN_REPEAT_COMPLEX_AGREEMENT
                )
                rows.append(
                    {
                        "path": path,
                        "block": block,
                        "microphone": mic,
                        "line_snr_db": snr.tolist(),
                        "minimum_line_snr_db": float(np.min(snr)),
                        "complex_agreement": agreement,
                        "passed": passed,
                    }
                )
    if not rows or not all(row["passed"] for row in rows):
        raise ValueError(f"post-q {stage} repeat SNR/agreement gate 실패")
    receipt = {
        "schema": "fullband_v6_post_q_repeat_mean_half_difference_v1",
        "stage": stage,
        "fixed_line_bins": list(CLOCK_BINS),
        "signal_definition": "complex_repeat_mean",
        "noise_definition": "complex_repeat_half_difference",
        "minimum_line_snr_db": MIN_REPEAT_LINE_SNR_DB,
        "minimum_complex_agreement": MIN_REPEAT_COMPLEX_AGREEMENT,
        "rows": rows,
        "passed": True,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt


def estimate_common_clock_v6(
    *,
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    captured_pcm: np.ndarray,
) -> dict[str, Any]:
    """고정 v6 checkpoint만으로 fail-closed 공통 DAC/ADC q를 계산한다."""

    expected_plan, expected_submitted = build_plan_v6(
        raw_session_relative_path=str(plan.get("raw_session_relative_path", RAW_DEFAULT))
    )
    submitted = np.asarray(submitted_pcm)
    captured = np.asarray(captured_pcm)
    if dict(plan) != expected_plan or not np.array_equal(submitted, expected_submitted):
        raise ValueError("clock estimator 입력이 exact v6 plan/PCM이 아닙니다")
    if captured.shape != submitted.shape or not np.all(np.isfinite(captured)):
        raise ValueError("v6 captured PCM은 submitted와 같은 finite [frame,2]여야 합니다")
    pre_rows = _clock_rows(plan, terminal=False)
    terminal_rows = _clock_rows(plan, terminal=True)
    if len(pre_rows) != 6 or len(terminal_rows) != 2:
        raise ValueError("v6 clock row 수가 3 preterminal epoch+1 terminal epoch가 아닙니다")
    holdout_ranges = [tuple(value) for value in plan["holdout_access_policy"]["operator_holdout_frame_ranges"]]
    accessed = [
        (int(start), int(start) + PERIOD)
        for row in pre_rows + terminal_rows
        for start in row["central_repeat_starts"]
    ]
    if any(max(a, c) < min(b, d) for a, b in accessed for c, d in holdout_ranges):
        raise ValueError("clock estimator가 operator holdout waveform을 접근했습니다")

    pre_repeats = _nominal_repeats(captured, pre_rows)
    terminal_repeats = _nominal_repeats(captured, terminal_rows)
    try:
        pre_snr = preoptimizer_spectral_line_admission_v6(pre_repeats)
    except V6ClockAdmissionError as exc:
        raise V6ClockAdmissionError(
            str(exc),
            stage="preterminal_preoptimizer_snr_admission",
            optimizer_started=False,
            available_receipt=exc.available_receipt,
        ) from exc
    try:
        terminal_pre_snr = preoptimizer_spectral_line_admission_v6(terminal_repeats)
    except V6ClockAdmissionError as exc:
        raise V6ClockAdmissionError(
            str(exc),
            stage="terminal_preoptimizer_snr_admission",
            optimizer_started=False,
            available_receipt={
                "preterminal_preoptimizer_snr_admission": pre_snr,
                "terminal_preoptimizer_snr_admission": exc.available_receipt,
            },
        ) from exc
    pilot = _clock_period().astype(np.float64)
    pilot_spectrum = np.fft.rfft(pilot)
    observations = _phase_observations(captured, pre_rows, pilot_spectrum)
    combined_observations = [value for path in PATH_CHANNEL for value in observations[path].values()]
    try:
        analytic_ratio, global_receipt = global_grid_basin_search_v6(
            lambda q: _circular_phase_objective(combined_observations, q)
        )
    except ValueError as exc:
        raise V6ClockAdmissionError(
            str(exc),
            stage="global_grid_basin_search",
            optimizer_started=True,
            available_receipt={
                "preterminal_preoptimizer_snr_admission": pre_snr,
                "terminal_preoptimizer_snr_admission": terminal_pre_snr,
            },
        ) from exc
    view_ratios: dict[str, float] = {}
    view_receipts: dict[str, Any] = {}
    for path in PATH_CHANNEL:
        for mic, label in enumerate(("ERR", "REF")):
            name = f"{path}_{label}"
            try:
                view_ratios[name], view_receipts[name] = global_grid_basin_search_v6(
                    lambda q, value=observations[path][mic]: _circular_phase_objective([value], q)
                )
            except ValueError as exc:
                raise V6ClockAdmissionError(
                    str(exc),
                    stage=f"view_global_grid_basin_search/{name}",
                    optimizer_started=True,
                    available_receipt={
                        "global_search": global_receipt,
                        "completed_view_searches": view_receipts,
                    },
                ) from exc
    view_endpoint = (max(view_ratios.values()) - min(view_ratios.values())) * len(submitted)
    if view_endpoint > MAX_VIEW_ENDPOINT_DISAGREEMENT_SAMPLES:
        raise V6ClockAdmissionError(
            "v6 clock path/mic view consensus gate 실패",
            stage="view_consensus",
            optimizer_started=True,
            available_receipt={
                "global_search": global_receipt,
                "view_global_search": view_receipts,
                "view_rate_ratios": view_ratios,
                "maximum_view_endpoint_disagreement_samples": float(view_endpoint),
            },
        )

    selected: dict[str, float] = {}
    interpolation_receipts: dict[str, Any] = {}
    transfer_banks: dict[str, dict[str, np.ndarray]] = {}
    for kind in ("cubic", "linear"):
        pre_accessors = _bounded_interpolators(captured, pre_rows, kind)
        terminal_accessors = _bounded_interpolators(captured, terminal_rows, kind)
        lower = max(1.0 - MAX_CLOCK_PPM * 1e-6, analytic_ratio - GLOBAL_GRID_STEP_PPM * 1e-6)
        upper = min(1.0 + MAX_CLOCK_PPM * 1e-6, analytic_ratio + GLOBAL_GRID_STEP_PPM * 1e-6)
        fit = minimize_scalar(
            lambda q: _transfer_consistency_objective(
                _corrected_transfers(
                    pre_accessors,
                    pre_rows,
                    ratio=float(q),
                    pilot_spectrum=pilot_spectrum,
                )
            ),
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1.0e-12},
        )
        if not fit.success or not np.isfinite(fit.fun):
            raise V6ClockAdmissionError(
                "v6 interpolation-specific clock refinement 실패",
                stage=f"{kind}_interpolation_refinement",
                optimizer_started=True,
                available_receipt={
                    "global_search": global_receipt,
                    "view_global_search": view_receipts,
                    "completed_interpolation": interpolation_receipts,
                },
            )
        selected[kind] = float(fit.x)
        pre_transfers = _corrected_transfers(
            pre_accessors,
            pre_rows,
            ratio=selected[kind],
            pilot_spectrum=pilot_spectrum,
        )
        terminal_transfers = _corrected_transfers(
            terminal_accessors,
            terminal_rows,
            ratio=selected[kind],
            pilot_spectrum=pilot_spectrum,
        )
        try:
            pre_repeat = _post_q_repeat_receipt(
                pre_transfers, stage=f"{kind}_preterminal"
            )
            terminal_repeat = _post_q_repeat_receipt(
                terminal_transfers, stage=f"{kind}_terminal"
            )
        except ValueError as exc:
            raise V6ClockAdmissionError(
                str(exc),
                stage=f"{kind}_post_q_repeat_validation",
                optimizer_started=True,
                available_receipt={
                    "global_search": global_receipt,
                    "view_global_search": view_receipts,
                    "completed_interpolation": interpolation_receipts,
                },
            ) from exc
        transfer_banks[kind] = {
            "preterminal": pre_transfers,
            "terminal": terminal_transfers,
        }
        interpolation_receipts[kind] = {
            "selected_ratio": selected[kind],
            "selected_ppm": (selected[kind] - 1.0) * 1.0e6,
            "local_refinement_bounds": [lower, upper],
            "objective": float(fit.fun),
            "preterminal_repeat": pre_repeat,
            "terminal_repeat": terminal_repeat,
            "bounded_owned_capture": {
                "preterminal": {
                    f"{epoch}/{path}": {
                        key: value
                        for key, value in accessor.items()
                        if key != "interpolators"
                    }
                    for (epoch, path), accessor in pre_accessors.items()
                },
                "terminal": {
                    f"{epoch}/{path}": {
                        key: value
                        for key, value in accessor.items()
                        if key != "interpolators"
                    }
                    for (epoch, path), accessor in terminal_accessors.items()
                },
            },
        }
    endpoint = abs(selected["cubic"] - selected["linear"]) * len(submitted)
    if endpoint > MAX_CUBIC_LINEAR_ENDPOINT_DISAGREEMENT_SAMPLES:
        raise V6ClockAdmissionError(
            "v6 cubic-linear endpoint crosscheck 실패",
            stage="cubic_linear_endpoint_crosscheck",
            optimizer_started=True,
            available_receipt={
                "interpolation": interpolation_receipts,
                "cubic_linear_endpoint_disagreement_samples": float(endpoint),
            },
        )

    bins = np.asarray(CLOCK_BINS, dtype=np.float64)
    maximum_terminal = 0.0
    for kind, banks in transfer_banks.items():
        for path in PATH_CHANNEL:
            fit_mean = np.mean(banks["preterminal"][path], axis=(0, 1))
            terminal_mean = np.mean(banks["terminal"][path], axis=(0, 1))
            phase = np.angle(terminal_mean * np.conj(fit_mean))
            sample_error = np.abs(phase) / (2.0 * np.pi * bins[:, None] / PERIOD)
            maximum_terminal = max(maximum_terminal, float(np.max(sample_error)))
    if maximum_terminal > MAX_TERMINAL_PHASE_ERROR_SAMPLES:
        raise V6ClockAdmissionError(
            "v6 terminal clock phase validation 실패",
            stage="terminal_clock_validation",
            optimizer_started=True,
            available_receipt={
                "interpolation": interpolation_receipts,
                "maximum_terminal_phase_error_samples": float(maximum_terminal),
            },
        )

    receipt = {
        "schema": "fullband_causal_common_clock_v6",
        "signal_plan_payload_sha256": plan["canonical_payload_sha256"],
        "actual_submitted_pcm_sha256": plan["actual_submitted_pcm_sha256"],
        "fixed_line_bins": list(CLOCK_BINS),
        "preoptimizer_snr_admission": pre_snr,
        "terminal_preoptimizer_snr_admission": terminal_pre_snr,
        "global_search": global_receipt,
        "view_global_search": view_receipts,
        "view_rate_ratios": view_ratios,
        "maximum_view_endpoint_disagreement_samples": float(view_endpoint),
        "interpolation": interpolation_receipts,
        "selected_rate_ratio": selected["cubic"],
        "selected_ppm": (selected["cubic"] - 1.0) * 1.0e6,
        "cubic_linear_endpoint_disagreement_samples": float(endpoint),
        "maximum_terminal_phase_error_samples": maximum_terminal,
        "clock_fit_epochs": list(CLOCK_EPOCHS[:3]),
        "clock_terminal_validation_epoch": CLOCK_EPOCHS[3],
        "operator_holdout_accessed": False,
        "accessed_clock_frame_ranges": [list(value) for value in accessed],
        "captured_full_sha256_computed": False,
        "passed": True,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt


def synthesize_affine_capture_v6(
    submitted_pcm: np.ndarray,
    *,
    primary_fir_by_mic: np.ndarray,
    secondary_fir_by_mic: np.ndarray,
    rate_ratio: float,
    noise_rms: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """오디오 없이 fixed two-path FIR와 affine ADC clock fixture를 만든다."""

    submitted = np.asarray(submitted_pcm, dtype=np.float64)
    primary = np.asarray(primary_fir_by_mic, dtype=np.float64)
    secondary = np.asarray(secondary_fir_by_mic, dtype=np.float64)
    if primary.shape != secondary.shape or primary.ndim != 2 or primary.shape[0] != 2:
        raise ValueError("v6 synthetic P/S FIR shape이 잘못됐습니다")
    full = []
    for mic in range(2):
        value = fftconvolve(submitted[:, 0], primary[mic])
        value += fftconvolve(submitted[:, 1], secondary[mic])
        full.append(value)
    dac = np.column_stack(full)
    query = np.arange(len(submitted), dtype=np.float64) * float(rate_ratio)
    captured = np.column_stack(
        [CubicSpline(np.arange(len(dac)), dac[:, mic])(query) for mic in range(2)]
    )
    if noise_rms:
        captured += np.random.default_rng(seed).normal(0.0, float(noise_rms), captured.shape)
    return captured


__all__ = [
    "CLOCK_BINS",
    "CLOCK_BLOCK_FRAMES",
    "CLOCK_EPOCHS",
    "CLOCK_PREFIX",
    "CLOCK_SUFFIX",
    "RAW_DEFAULT",
    "SCHEMA",
    "TOTAL_FRAMES",
    "V6ClockAdmissionError",
    "build_plan_v6",
    "estimate_common_clock_v6",
    "exact_condition_audit_v6",
    "global_grid_basin_search_v6",
    "preoptimizer_spectral_line_admission_v6",
    "synthesize_affine_capture_v6",
]
