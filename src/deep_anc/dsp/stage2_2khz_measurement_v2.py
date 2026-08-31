"""Stage-2 2 kHz P/S용 24초 actual-int16 signal/식별성 계약.

이 모듈은 audio backend를 import하지 않으며 장치·GPU·network를 열지 않는다. 기존
88--2828 Hz 전용 aperiodic code가 1024-tap causal FIR 전체를 식별하지 못한 문제를
해결하기 위해, objective/holdout 평가는 그대로 125--2 kHz에 두되 P/S fitting 입력은
actual-int16 near-white PE로 만든다. continuous dual-disjoint pilot만 두 playback channel에
동시에 존재하고, 주 P/S PE는 시간 분리된다.

signal-only plan과 Gram PASS는 실제 acoustic P/S, clock, THD 또는 training authority가
아니다. 특히 near-Nyquist 에너지를 실제 speaker/amp에 출력하는 live safety는 별도 물리
preflight가 생길 때까지 BLOCKED다.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.linalg import eigh, toeplitz
from scipy.signal.windows import dpss

from .broadband_plant_analysis import exact_two_input_periodic_gram_audit
from .fullband_causal_v4 import continuous_pilot_period
from .fullband_causal_v5 import near_white_period
from .measurement_level import expected_meter_output_pcm
from .stage2_2khz_contract import Stage2TwoKilohertzContract
from .stage2_2khz_level_contract import canonical_stage2_operating_level_contract


SCHEMA = "stage2_2khz_time_separated_full_pe_plan_v2"
GRAM_SCHEMA = "stage2_2khz_actual_int16_shifted_gram_v2"
ACCESS_LEDGER_SCHEMA = "stage2_2khz_holdout_access_ledger_v1"
SAMPLE_RATE = 48_000
BLOCK_SIZE = 256
TOTAL_FRAMES = 1_152_000
SIGNAL_SECONDS = 24.0
PERIOD = 32_768
PREFIX = 16_384
SUFFIX = 16_384
SLOT_FRAMES = PREFIX + PERIOD + SUFFIX
SUPPORT_SAMPLES = 1_024
GRAM_DIMENSION = 2 * SUPPORT_SAMPLES
MAX_GRAM_CONDITION = 20.0
MAX_SUBMITTED_PEAK_PCM = 98
MAX_MODEL_ACTUATOR_ABS = MAX_SUBMITTED_PEAK_PCM / 32768.0
PE_PEAK_PCM = 49
PILOT_PEAK_PCM = 20
DIAGNOSTIC_LEVELS_PCM = (49, 98)
PRE_ZERO_FRAMES = 32_768
PILOT_LEAD_FRAMES = 65_536
PILOT_TAIL_FRAMES = 98_304
POST_ZERO_FRAMES = 37_888
FIT_ROLES = ("fit_a", "fit_b")
HOLDOUT_ROLE = "untouched_holdout"
PATHS = ("primary", "secondary")
PATH_CHANNEL = {"primary": 0, "secondary": 1}
PE_SLOT_ORDER = (
    ("fit_a", "primary"),
    ("fit_a", "secondary"),
    ("fit_b", "secondary"),
    ("fit_b", "primary"),
    (HOLDOUT_ROLE, "secondary"),
    (HOLDOUT_ROLE, "primary"),
)
PE_SEEDS = {
    ("fit_a", "primary"): 710_001,
    ("fit_a", "secondary"): 710_003,
    ("fit_b", "primary"): 710_021,
    ("fit_b", "secondary"): 710_023,
    (HOLDOUT_ROLE, "primary"): 710_041,
    (HOLDOUT_ROLE, "secondary"): 710_043,
}
DIAGNOSTIC_TONE_FREQUENCIES_HZ = ((752, 1_248), (1_800, 2_200))
DIAGNOSTIC_LEAD_FRAMES = 16_384
DIAGNOSTIC_ACTIVE_FRAMES = 36_000
DIAGNOSTIC_TAIL_FRAMES = 13_152
DIAGNOSTIC_FADE_FRAMES = 480
DIAGNOSTIC_ANALYSIS_OFFSET_FRAMES = 4_800
DIAGNOSTIC_ANALYSIS_FRAMES = 24_000
LIVE_SAFETY_STATUS = "BLOCKED_NEAR_NYQUIST_PHYSICAL_SAFETY_NOT_ESTABLISHED"
LIVE_SAFE_FALLBACK_STATUS = (
    "SIGNAL_ONLY_BANDLIMITED_DPSS_DESIGN_PHYSICAL_PREFLIGHT_STILL_REQUIRED"
)
LIVE_SAFE_BAND_HZ = (80.0, 2828.4271247462)
DPSS_HALF_BANDWIDTH_HZ = (LIVE_SAFE_BAND_HZ[1] - LIVE_SAFE_BAND_HZ[0]) / 2.0
DPSS_CENTER_HZ = (LIVE_SAFE_BAND_HZ[1] + LIVE_SAFE_BAND_HZ[0]) / 2.0
DPSS_TIME_HALF_BANDWIDTH = SUPPORT_SAMPLES * DPSS_HALF_BANDWIDTH_HZ / SAMPLE_RATE
DPSS_COMPLEX_TAPER_COUNT = 56
DPSS_REAL_DOF_PER_PATH = 2 * DPSS_COMPLEX_TAPER_COUNT


class Stage2MeasurementV2Error(ValueError):
    """Stage-2 v2 signal/수치 계약 위반."""


class Stage2HoldoutAccessError(RuntimeError):
    """untouched holdout 실행 순서 또는 범위 위반."""


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
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


def build_stage2_bandlimited_dpss_basis() -> tuple[np.ndarray, dict[str, Any]]:
    """80--2828 Hz authority용 deterministic 1024x112 real DPSS basis를 만든다."""

    tapers = np.asarray(
        dpss(
            SUPPORT_SAMPLES,
            DPSS_TIME_HALF_BANDWIDTH,
            Kmax=DPSS_COMPLEX_TAPER_COUNT,
            sym=True,
            norm=2,
        ),
        dtype=np.float64,
    )
    sample = np.arange(SUPPORT_SAMPLES, dtype=np.float64)
    carrier = np.exp(2j * math.pi * DPSS_CENTER_HZ * sample / SAMPLE_RATE)
    modulated = tapers * carrier[None, :]
    candidates = np.column_stack(
        (
            math.sqrt(2.0) * modulated.real.T,
            math.sqrt(2.0) * modulated.imag.T,
        )
    )
    basis, _ = np.linalg.qr(candidates, mode="reduced")
    # QR column sign은 수학적으로 임의다. 최대 절댓값 sample이 양수가 되게 canonicalize한다.
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    orthogonality_error = float(
        np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1])))
    )
    if basis.shape != (SUPPORT_SAMPLES, DPSS_REAL_DOF_PER_PATH):
        raise AssertionError("Stage-2 DPSS basis shape가 잘못됐습니다")
    if orthogonality_error > 1.0e-12:
        raise AssertionError("Stage-2 DPSS QR basis가 orthonormal하지 않습니다")
    basis = np.ascontiguousarray(basis, dtype=np.float64)
    receipt = {
        "schema": "stage2_2khz_bandlimited_dpss_basis_v1",
        "support_samples": SUPPORT_SAMPLES,
        "sample_rate_hz": SAMPLE_RATE,
        "authority_band_hz": list(LIVE_SAFE_BAND_HZ),
        "center_hz": DPSS_CENTER_HZ,
        "half_bandwidth_hz": DPSS_HALF_BANDWIDTH_HZ,
        "time_half_bandwidth": DPSS_TIME_HALF_BANDWIDTH,
        "complex_taper_count": DPSS_COMPLEX_TAPER_COUNT,
        "real_dof_per_path": DPSS_REAL_DOF_PER_PATH,
        "basis_shape": list(basis.shape),
        "basis_dtype": basis.dtype.str,
        "basis_sha256": _array_sha256(basis),
        "orthogonality_max_abs_error": orthogonality_error,
        "coarse_delay_is_external_not_absorbed": True,
        "ridge_or_minimum_norm_nullspace_fill_allowed": False,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    basis.setflags(write=False)
    return basis, receipt


def audit_stage2_bandlimited_dpss_projected_gram(
    role_period_rows: Mapping[str, tuple[np.ndarray, ...]],
    *,
    zeros_by_path: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    """live-safe bandlimited PE의 DPSS-subspace Gram을 actual int16에서 계산한다.

    이 receipt는 unrestricted 1024-tap 식별 가능성을 주장하지 않는다. basis coefficient
    116개/path만 식별하고 1024 causal FIR로 복원한다. training/eval consumer가 basis SHA와
    authority band를 실제로 결속하기 전에는 numerical PASS여도 canonical admission은 false다.
    """

    roles = FIT_ROLES
    rows: dict[str, tuple[np.ndarray, ...]] = {}
    period_samples: int | None = None
    for role in roles:
        source_rows = tuple(np.asarray(value) for value in role_period_rows.get(role, ()))
        if not source_rows:
            raise Stage2MeasurementV2Error(f"DPSS projected audit {role} row가 없습니다")
        owned: list[np.ndarray] = []
        for row in source_rows:
            if row.dtype != np.int16 or row.ndim != 2 or row.shape[1] != 2:
                raise Stage2MeasurementV2Error("DPSS projected audit는 actual int16 [P,2]만 허용합니다")
            if period_samples is None:
                period_samples = int(row.shape[0])
            if row.shape != (period_samples, 2):
                raise Stage2MeasurementV2Error("DPSS projected audit period shape가 다릅니다")
            owned.append(np.array(row, copy=True, order="C"))
        rows[role] = tuple(owned)
    assert period_samples is not None
    raw_zeros = tuple(zeros_by_path)
    if len(raw_zeros) != 2 or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in raw_zeros
    ):
        raise Stage2MeasurementV2Error("DPSS projected shift는 integer pair여야 합니다")
    zeros = tuple(int(value) for value in raw_zeros)
    if any(value < 0 or value >= period_samples for value in zeros):
        raise Stage2MeasurementV2Error("DPSS projected shift 범위가 잘못됐습니다")
    basis, basis_receipt = build_stage2_bandlimited_dpss_basis()
    dof = DPSS_REAL_DOF_PER_PATH
    dimension = 2 * dof
    offsets = np.arange(SUPPORT_SAMPLES)

    def shifted(row: np.ndarray) -> np.ndarray:
        value = row.astype(np.float64)
        return np.column_stack(
            [np.roll(value[:, channel], zeros[channel]) for channel in range(2)]
        )

    def projected_block(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        correlation = np.fft.ifft(
            np.conj(np.fft.fft(left)) * np.fft.fft(right)
        ).real
        block = toeplitz(
            correlation[offsets % period_samples],
            correlation[(-offsets) % period_samples],
        )
        return basis.T @ block @ basis

    gram = np.zeros((dimension, dimension), dtype=np.float64)
    role_conditions: dict[str, float | None] = {}
    for role in roles:
        role_gram = np.zeros_like(gram)
        for row in rows[role]:
            value = shifted(row)
            for left in range(2):
                for right in range(2):
                    role_gram[
                        left * dof : (left + 1) * dof,
                        right * dof : (right + 1) * dof,
                    ] += projected_block(value[:, left], value[:, right])
        role_gram = (role_gram + role_gram.T) * 0.5
        bounds = eigh(
            role_gram,
            subset_by_index=[0, dimension - 1],
            eigvals_only=True,
        )
        lo, hi = float(bounds[0]), float(bounds[-1])
        role_conditions[role] = hi / lo if lo > 0.0 and math.isfinite(hi) else None
        gram += role_gram
    gram = (gram + gram.T) * 0.5
    eigenvalues = np.asarray(eigh(gram, eigvals_only=True), dtype=np.float64)
    lo, hi = float(eigenvalues[0]), float(eigenvalues[-1])
    rank_tolerance = float(dimension * np.finfo(np.float64).eps * max(abs(hi), 1.0))
    rank = int(np.count_nonzero(eigenvalues > rank_tolerance))
    condition = hi / lo if lo > 0.0 and math.isfinite(hi) else None

    probe_errors: list[dict[str, Any]] = []
    grid = np.arange(dimension, dtype=np.float64)
    probes = (
        np.sin(0.071 * grid) + 0.17 * np.cos(0.023 * grid),
        np.where((grid.astype(np.int64) % 11) < 5, 1.0, -1.0),
    )
    for index, probe in enumerate(probes):
        fir = np.stack((basis @ probe[:dof], basis @ probe[dof:]), axis=0)
        transfer = np.fft.rfft(fir, n=period_samples, axis=1)
        direct_energy = 0.0
        direct_normal_full = np.zeros((2, SUPPORT_SAMPLES), dtype=np.float64)
        for role in roles:
            for row in rows[role]:
                value = shifted(row)
                prediction = np.fft.irfft(
                    np.sum(np.fft.rfft(value, axis=0) * transfer.T, axis=1),
                    n=period_samples,
                )
                direct_energy += float(np.dot(prediction, prediction))
                prediction_fft = np.fft.rfft(prediction)
                for channel in range(2):
                    adjoint = np.fft.irfft(
                        np.conj(np.fft.rfft(value[:, channel])) * prediction_fft,
                        n=period_samples,
                    )
                    direct_normal_full[channel] += adjoint[:SUPPORT_SAMPLES]
        direct_normal = np.concatenate(
            (basis.T @ direct_normal_full[0], basis.T @ direct_normal_full[1])
        )
        gram_energy = float(probe @ gram @ probe)
        gram_normal = gram @ probe
        energy_error = abs(direct_energy - gram_energy) / max(
            abs(direct_energy), abs(gram_energy), 1.0
        )
        normal_error = float(
            np.linalg.norm(direct_normal - gram_normal)
            / max(np.linalg.norm(direct_normal), np.linalg.norm(gram_normal), 1.0)
        )
        probe_errors.append(
            {
                "probe_index": index,
                "quadratic_form_relative_error": energy_error,
                "normal_vector_relative_error": normal_error,
            }
        )
    crosscheck_error = max(
        max(row["quadratic_form_relative_error"], row["normal_vector_relative_error"])
        for row in probe_errors
    )
    numerical_passed = bool(
        rank == dimension
        and condition is not None
        and condition <= MAX_GRAM_CONDITION
        and all(
            value is not None and value <= MAX_GRAM_CONDITION
            for value in role_conditions.values()
        )
        and crosscheck_error <= 1.0e-10
    )
    receipt: dict[str, Any] = {
        "schema": "stage2_2khz_bandlimited_dpss_projected_gram_v1",
        "actual_int16_input_required": True,
        "authority_band_hz": list(LIVE_SAFE_BAND_HZ),
        "unrestricted_1024tap_authority_claimed": False,
        "basis_definition_sha256": basis_receipt["canonical_payload_sha256"],
        "basis_array_sha256": basis_receipt["basis_sha256"],
        "support_samples": SUPPORT_SAMPLES,
        "real_dof_per_path": dof,
        "gram_dimension": dimension,
        "numeric_rank": rank,
        "full_numeric_rank": rank == dimension,
        "rank_tolerance": rank_tolerance,
        "minimum_eigenvalue": lo,
        "maximum_eigenvalue": hi,
        "projected_normal_matrix_condition_number": condition,
        "role_condition_numbers": role_conditions,
        "maximum_condition_number": MAX_GRAM_CONDITION,
        "quadratic_crosscheck_receipts": probe_errors,
        "quadratic_crosscheck_error": crosscheck_error,
        "numerical_subspace_passed": numerical_passed,
        "training_eval_consumer_basis_binding_implemented": False,
        "canonical_training_eligible": False,
        "live_physical_authority_claimed": False,
        "remaining_blocker": (
            "training/eval plant consumer가 동일 basis SHA와 80--2828 Hz authority를 "
            "소비하고 physical raw projected fit/untouched holdout을 통과해야 한다"
        ),
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt


def _spectral_floor_receipt(pe: np.ndarray) -> dict[str, Any]:
    signal = np.asarray(pe)
    if signal.dtype != np.int16 or signal.shape != (PERIOD,):
        raise Stage2MeasurementV2Error("PE spectral audit는 exact int16 period여야 합니다")
    magnitude = np.abs(np.fft.rfft(signal.astype(np.float64)))
    frequency = np.fft.rfftfreq(PERIOD, 1.0 / SAMPLE_RATE)
    selected = magnitude[(frequency >= 20.0) & (frequency < SAMPLE_RATE / 2.0)]
    median = float(np.median(selected))
    minimum = float(np.min(selected))
    ratio = minimum / median if median > 0.0 else 0.0
    nonzero = float(np.mean(selected > 0.0))
    if not math.isfinite(ratio) or ratio < 1.0e-3 or nonzero != 1.0:
        raise Stage2MeasurementV2Error("actual-int16 PE fullband spectral floor가 부족합니다")
    return {
        "band_hz": [20.0, 24_000.0],
        "nyquist_bin_excluded": True,
        "nonzero_bin_fraction": nonzero,
        "minimum_to_median_magnitude_ratio": ratio,
        "minimum_required_ratio": 1.0e-3,
        "passed": True,
    }


def _pilot_bin_receipt(pilot: np.ndarray) -> dict[str, Any]:
    spectrum = np.fft.rfft(pilot.astype(np.float64), axis=0)
    frequency = np.fft.rfftfreq(PERIOD, 1.0 / SAMPLE_RATE)
    band = (frequency >= 152.0) & (frequency <= 600.0)
    primary = np.flatnonzero(
        band
        & (np.abs(spectrum[:, 0]) > 1.0e3)
        & (np.abs(spectrum[:, 1]) <= 1.0e-8)
    )
    secondary = np.flatnonzero(
        band
        & (np.abs(spectrum[:, 1]) > 1.0e3)
        & (np.abs(spectrum[:, 0]) <= 1.0e-8)
    )
    if min(primary.size, secondary.size) < 8 or np.intersect1d(primary, secondary).size:
        raise Stage2MeasurementV2Error("dual pilot exact-null bin 분리가 부족합니다")
    return {
        "period_samples": PERIOD,
        "selection_band_hz": [152.0, 600.0],
        "primary_bins": primary.tolist(),
        "secondary_bins": secondary.tolist(),
        "disjoint": True,
        "opposite_channel_exact_null": True,
    }


def _cyclic_payload_slot(payload: np.ndarray, *, channel: int) -> np.ndarray:
    values = np.asarray(payload)
    if values.dtype != np.int16 or values.shape != (PERIOD,) or channel not in (0, 1):
        raise Stage2MeasurementV2Error("cyclic payload 입력이 잘못됐습니다")
    slot = np.zeros((SLOT_FRAMES, 2), dtype=np.int16)
    slot[:PREFIX, channel] = values[-PREFIX:]
    slot[PREFIX : PREFIX + PERIOD, channel] = values
    slot[PREFIX + PERIOD :, channel] = values[:SUFFIX]
    return slot


def _diagnostic_tone_active(pair_index: int, level_pcm: int) -> np.ndarray:
    if pair_index not in (0, 1) or level_pcm not in DIAGNOSTIC_LEVELS_PCM:
        raise Stage2MeasurementV2Error("nonlinearity tone pair/level이 잘못됐습니다")
    first, second = DIAGNOSTIC_TONE_FREQUENCIES_HZ[pair_index]
    sample = np.arange(DIAGNOSTIC_ACTIVE_FRAMES, dtype=np.float64)
    wave = 0.5 * float(DIAGNOSTIC_LEVELS_PCM[0]) * (
        np.cos(2.0 * math.pi * first * sample / SAMPLE_RATE)
        + np.cos(2.0 * math.pi * second * sample / SAMPLE_RATE)
    )
    fade = np.ones(DIAGNOSTIC_ACTIVE_FRAMES, dtype=np.float64)
    ramp = 0.5 - 0.5 * np.cos(
        math.pi * np.arange(DIAGNOSTIC_FADE_FRAMES, dtype=np.float64)
        / float(DIAGNOSTIC_FADE_FRAMES)
    )
    fade[:DIAGNOSTIC_FADE_FRAMES] = ramp
    fade[-DIAGNOSTIC_FADE_FRAMES:] = ramp[::-1]
    low = np.rint(wave * fade).astype(np.int16)
    result = low if level_pcm == DIAGNOSTIC_LEVELS_PCM[0] else (2 * low).astype(np.int16)
    if int(np.max(np.abs(result.astype(np.int32)))) != level_pcm:
        raise AssertionError("diagnostic tone actual peak가 level과 다릅니다")
    return result


def build_stage2_v2_signal_plan() -> tuple[dict[str, Any], np.ndarray]:
    """exact 24초 signal-only plan과 submitted actual-int16 PCM을 생성한다."""

    pilot_period = continuous_pilot_period()
    if pilot_period.dtype != np.int16 or pilot_period.shape != (PERIOD, 2):
        raise AssertionError("v4 continuous pilot period 계약이 바뀌었습니다")
    if int(np.max(np.abs(pilot_period.astype(np.int32)))) != PILOT_PEAK_PCM:
        raise AssertionError("continuous pilot peak 계약이 바뀌었습니다")
    pilot_bins = _pilot_bin_receipt(pilot_period)
    parts: list[np.ndarray] = [np.zeros((PRE_ZERO_FRAMES, 2), dtype=np.int16)]
    layout: list[dict[str, Any]] = [
        {
            "kind": "zero_pre",
            "start_frame": 0,
            "stop_frame": PRE_ZERO_FRAMES,
            "frames": PRE_ZERO_FRAMES,
        }
    ]
    cursor = PRE_ZERO_FRAMES

    lead = np.tile(pilot_period, (PILOT_LEAD_FRAMES // PERIOD, 1))
    parts.append(lead)
    layout.append(
        {
            "kind": "pilot_only_lead",
            "start_frame": cursor,
            "stop_frame": cursor + PILOT_LEAD_FRAMES,
            "frames": PILOT_LEAD_FRAMES,
        }
    )
    continuous_pilot_start = cursor
    cursor += PILOT_LEAD_FRAMES

    payload_receipts: dict[str, Any] = {}
    for role, path in PE_SLOT_ORDER:
        seed = PE_SEEDS[(role, path)]
        payload = near_white_period(seed)
        active = PATH_CHANNEL[path]
        main = _cyclic_payload_slot(payload, channel=active)
        pilot = np.tile(pilot_period, (SLOT_FRAMES // PERIOD, 1))
        # SLOT_FRAMES=2*PERIOD이므로 pilot phase가 prefix와 central 경계에서도 이어진다.
        submitted32 = main.astype(np.int32) + pilot.astype(np.int32)
        if int(np.max(np.abs(submitted32))) > MAX_SUBMITTED_PEAK_PCM:
            raise AssertionError("PE slot peak가 Stage-2 안전 상한을 넘었습니다")
        slot = submitted32.astype(np.int16)
        name = f"{role}_{path}"
        central_start = cursor + PREFIX
        layout.append(
            {
                "kind": "pe_slot",
                "name": name,
                "role": role,
                "path": path,
                "active_main_channel": active,
                "opposite_main_channel_exact_zero_except_disjoint_pilot": True,
                "start_frame": cursor,
                "stop_frame": cursor + SLOT_FRAMES,
                "central_start_frame": central_start,
                "central_stop_frame": central_start + PERIOD,
                "frames": SLOT_FRAMES,
                "cyclic_prefix_frames": PREFIX,
                "cyclic_suffix_frames": SUFFIX,
                "seed": seed,
                "payload_sha256": _array_sha256(payload),
                "submitted_slot_sha256": _array_sha256(slot),
                "untouched_until_fit_frozen": role == HOLDOUT_ROLE,
            }
        )
        payload_receipts[name] = {
            "seed": seed,
            "generator": "deterministic_rademacher_actual_int16",
            "peak_pcm": PE_PEAK_PCM,
            "sha256": _array_sha256(payload),
            "spectral_floor": _spectral_floor_receipt(payload),
        }
        parts.append(slot)
        cursor += SLOT_FRAMES

    main_fit_pilot_stop = cursor
    diagnostic_receipts: list[dict[str, Any]] = []
    for path in PATHS:
        for pair_index in range(len(DIAGNOSTIC_TONE_FREQUENCIES_HZ)):
            for level in DIAGNOSTIC_LEVELS_PCM:
                tone = _diagnostic_tone_active(pair_index, level)
                main = np.zeros((SLOT_FRAMES, 2), dtype=np.int16)
                active_start = DIAGNOSTIC_LEAD_FRAMES
                active_stop = active_start + DIAGNOSTIC_ACTIVE_FRAMES
                main[active_start:active_stop, PATH_CHANNEL[path]] = tone
                slot32 = main.astype(np.int32)
                peak = int(np.max(np.abs(slot32)))
                if peak > MAX_SUBMITTED_PEAK_PCM:
                    raise AssertionError("nonlinearity diagnostic peak가 안전 상한을 넘었습니다")
                slot = slot32.astype(np.int16)
                first, second = DIAGNOSTIC_TONE_FREQUENCIES_HZ[pair_index]
                analysis_start = cursor + active_start + DIAGNOSTIC_ANALYSIS_OFFSET_FRAMES
                analysis_stop = analysis_start + DIAGNOSTIC_ANALYSIS_FRAMES
                local_analysis_start = active_start + DIAGNOSTIC_ANALYSIS_OFFSET_FRAMES
                analysis_pcm = slot[
                    local_analysis_start : local_analysis_start
                    + DIAGNOSTIC_ANALYSIS_FRAMES,
                    PATH_CHANNEL[path],
                ]
                analysis_spectrum = np.fft.rfft(analysis_pcm.astype(np.float64))
                fundamental_bins = [first // 2, second // 2]
                detection_bins = {
                    "difference": abs(second - first) // 2,
                    "sum": (first + second) // 2,
                    "second_harmonics": [first, second],
                }
                flat_detection_bins = [
                    detection_bins["difference"],
                    detection_bins["sum"],
                    *detection_bins["second_harmonics"],
                ]
                receipt = {
                    "kind": "multilevel_two_tone_diagnostic_slot",
                    "path": path,
                    "pair_index": pair_index,
                    "level_pcm": level,
                    "fundamental_frequencies_hz": [first, second],
                    "analysis_fft_bin_spacing_hz": 2.0,
                    "fundamental_analysis_fft_bins": fundamental_bins,
                    "detection_bins": detection_bins,
                    "start_frame": cursor,
                    "stop_frame": cursor + SLOT_FRAMES,
                    "guard_lead_start_frame": cursor,
                    "guard_lead_stop_frame": cursor + DIAGNOSTIC_LEAD_FRAMES,
                    "active_start_frame": cursor + active_start,
                    "active_stop_frame": cursor + active_stop,
                    "guard_tail_start_frame": cursor + active_stop,
                    "guard_tail_stop_frame": cursor + SLOT_FRAMES,
                    "fade_frames_each_edge": DIAGNOSTIC_FADE_FRAMES,
                    "analysis_start_frame": analysis_start,
                    "analysis_stop_frame": analysis_stop,
                    "analysis_frames": DIAGNOSTIC_ANALYSIS_FRAMES,
                    "analysis_window_excludes_fades_and_guards": True,
                    "submitted_analysis_window_sha256": _array_sha256(analysis_pcm),
                    "submitted_fundamental_magnitudes_pcm": [
                        float(abs(analysis_spectrum[index]))
                        for index in fundamental_bins
                    ],
                    "submitted_detection_magnitudes_pcm": [
                        float(abs(analysis_spectrum[index]))
                        for index in flat_detection_bins
                    ],
                    "submitted_detection_line_baseline_must_be_used": True,
                    "frames": SLOT_FRAMES,
                    "actual_active_tone_sha256": _array_sha256(tone),
                    "submitted_slot_sha256": _array_sha256(slot),
                    "submitted_peak_pcm": peak,
                    "inactive_output_exact_zero": True,
                    "clock_pilot_present": False,
                    "diagnostic_only_no_training_authority": True,
                }
                layout.append(receipt)
                diagnostic_receipts.append(receipt)
                parts.append(slot)
                cursor += SLOT_FRAMES

    tail = np.tile(pilot_period, (PILOT_TAIL_FRAMES // PERIOD, 1))
    parts.append(tail)
    layout.append(
        {
            "kind": "pilot_only_tail",
            "start_frame": cursor,
            "stop_frame": cursor + PILOT_TAIL_FRAMES,
            "frames": PILOT_TAIL_FRAMES,
        }
    )
    cursor += PILOT_TAIL_FRAMES
    terminal_pilot_stop = cursor
    parts.append(np.zeros((POST_ZERO_FRAMES, 2), dtype=np.int16))
    layout.append(
        {
            "kind": "zero_post",
            "start_frame": cursor,
            "stop_frame": cursor + POST_ZERO_FRAMES,
            "frames": POST_ZERO_FRAMES,
        }
    )
    cursor += POST_ZERO_FRAMES
    if cursor != TOTAL_FRAMES:
        raise AssertionError(f"Stage-2 v2 layout {cursor} != {TOTAL_FRAMES}")
    submitted = np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.int16)
    peak = int(np.max(np.abs(submitted.astype(np.int32))))
    if submitted.shape != (TOTAL_FRAMES, 2) or peak > MAX_SUBMITTED_PEAK_PCM:
        raise AssertionError("Stage-2 v2 submitted PCM shape/peak가 잘못됐습니다")

    meter = expected_meter_output_pcm(noise_channel=0)
    meter_power = float(
        np.sum(np.mean((meter.astype(np.float64) / 32768.0) ** 2, axis=0))
    )
    slot_powers = []
    for row in layout:
        if row["kind"] in {"pe_slot", "multilevel_two_tone_diagnostic_slot"}:
            slot = submitted[row["start_frame"] : row["stop_frame"]].astype(np.float64)
            slot_powers.append(float(np.sum(np.mean((slot / 32768.0) ** 2, axis=0))))
    worst_power = max(slot_powers)
    if worst_power > meter_power:
        raise AssertionError("Stage-2 v2 active slot power가 official meter를 초과합니다")

    contract = Stage2TwoKilohertzContract.canonical()
    operating_level = canonical_stage2_operating_level_contract()
    plan: dict[str, Any] = {
        "schema": SCHEMA,
        "role": "signal_only_no_audio_no_training_authority",
        "sample_rate_hz": SAMPLE_RATE,
        "block_size": BLOCK_SIZE,
        "signal_seconds": SIGNAL_SECONDS,
        "signal_frames": TOTAL_FRAMES,
        "actual_submitted_pcm": {
            "dtype": submitted.dtype.str,
            "shape": list(submitted.shape),
            "sha256": _array_sha256(submitted),
            "peak_pcm": peak,
            "maximum_peak_pcm": MAX_SUBMITTED_PEAK_PCM,
            "maximum_normalized_abs": MAX_MODEL_ACTUATOR_ABS,
        },
        "contract": {
            "id": contract.contract_id,
            "sha256": contract.digest(),
            "objective_octaves_hz": list(contract.octave_objective_centers_hz),
        },
        "operating_level_plan": operating_level,
        "layout_constants": {
            "zero_pre_frames": PRE_ZERO_FRAMES,
            "pilot_lead_frames": PILOT_LEAD_FRAMES,
            "pe_slot_count": len(PE_SLOT_ORDER),
            "pe_slot_frames": SLOT_FRAMES,
            "diagnostic_slot_count": len(diagnostic_receipts),
            "diagnostic_slot_frames": SLOT_FRAMES,
            "pilot_tail_frames": PILOT_TAIL_FRAMES,
            "zero_post_frames": POST_ZERO_FRAMES,
        },
        "layout": layout,
        "pe_payloads": payload_receipts,
        "pilot": {
            **pilot_bins,
            "peak_pcm": PILOT_PEAK_PCM,
            "period_sha256": _array_sha256(pilot_period),
            "main_fit_epoch_start_frame": continuous_pilot_start,
            "main_fit_epoch_stop_frame": main_fit_pilot_stop,
            "terminal_epoch_start_frame": terminal_pilot_stop - PILOT_TAIL_FRAMES,
            "terminal_epoch_stop_frame": terminal_pilot_stop,
            "diagnostic_gap_contains_pilot": False,
            "common_q_identification_scope": "relative_P/S_only",
            "time_separated_main_pe_with_continuous_dual_disjoint_pilot": True,
        },
        "nonlinearity_diagnostics": {
            "levels_pcm": list(DIAGNOSTIC_LEVELS_PCM),
            "expected_linear_amplitude_ratio": 2.0,
            "slots": diagnostic_receipts,
            "diagnostic_only": True,
            "thd_or_linearity_pass_claimed": False,
        },
        "gram_contract": {
            "support_samples": SUPPORT_SAMPLES,
            "dimension": GRAM_DIMENSION,
            "actual_int16_required": True,
            "maximum_normal_matrix_condition_number": MAX_GRAM_CONDITION,
            "quadratic_crosscheck_maximum": 1.0e-10,
            "fit_roles": list(FIT_ROLES),
            "holdout_excluded": True,
        },
        "holdout_policy": {
            "role": HOLDOUT_ROLE,
            "fit_access_forbidden": True,
            "support_or_threshold_selection_forbidden": True,
            "bounded_accessor_and_ledger_required": True,
            "refit_after_first_holdout_access_forbidden": True,
        },
        "meter_power_gate": {
            "official_meter_pcm_sha256": _array_sha256(meter),
            "official_meter_total_power": meter_power,
            "worst_active_slot_total_power": worst_power,
            "passed": True,
        },
        "live_safety": {
            "status": LIVE_SAFETY_STATUS,
            "audio_execution_allowed_by_this_plan": False,
            "near_nyquist_dac_amp_speaker_safety_verified": False,
            "actual_acoustic_ps_or_clock_authority_claimed": False,
        },
    }
    plan["canonical_payload_sha256"] = _payload_sha256(plan)
    submitted.setflags(write=False)
    return plan, submitted


def _live_safe_equal_band_period(seed: int) -> np.ndarray:
    frequency = np.fft.rfftfreq(PERIOD, 1.0 / SAMPLE_RATE)
    spectrum = np.zeros(frequency.size, dtype=np.complex128)
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    # DPSS의 가장자리 taper까지 식별 가능한 여유를 확보하려면 octave마다 동일
    # total energy를 주어 bin 밀도를 불연속으로 만드는 방식이 아니라 authority band의
    # 모든 DFT bin에 동일 magnitude를 주어야 한다. 이 선택은 fit weighting이며 최종
    # objective/holdout의 octave별 판정 가중치를 바꾸지 않는다.
    bins = np.flatnonzero(
        (frequency >= LIVE_SAFE_BAND_HZ[0]) & (frequency < LIVE_SAFE_BAND_HZ[1])
    )
    phase = generator.uniform(0.0, 2.0 * math.pi, size=bins.size)
    spectrum[bins] = np.exp(1j * phase)
    value = np.fft.irfft(spectrum, n=PERIOD)
    value /= float(np.sqrt(np.mean(value**2)))
    value = np.clip(value, -2.15, 2.15)
    value /= float(np.sqrt(np.mean(value**2)))
    actual = np.rint(value * (PE_PEAK_PCM / float(np.max(np.abs(value))))).astype(
        np.int16
    )
    if int(np.max(np.abs(actual.astype(np.int32)))) != PE_PEAK_PCM:
        raise AssertionError("live-safe equal-band actual peak가 잘못됐습니다")
    return actual


def build_stage2_v2_live_safe_fallback_plan() -> tuple[dict[str, Any], np.ndarray]:
    """near-Nyquist PE 대신 80--2828 Hz PE+DPSS subspace를 쓰는 24초 fallback."""

    source_plan, source_pcm = build_stage2_v2_signal_plan()
    plan = json.loads(json.dumps(source_plan, ensure_ascii=False, allow_nan=False))
    pcm = np.array(source_pcm, copy=True, order="C")
    pilot_period = continuous_pilot_period()
    for row in plan["layout"]:
        if row.get("kind") != "pe_slot":
            continue
        role, path = str(row["role"]), str(row["path"])
        payload = _live_safe_equal_band_period(PE_SEEDS[(role, path)])
        main = _cyclic_payload_slot(payload, channel=PATH_CHANNEL[path])
        pilot = np.tile(pilot_period, (SLOT_FRAMES // PERIOD, 1))
        slot = (main.astype(np.int32) + pilot.astype(np.int32)).astype(np.int16)
        start, stop = int(row["start_frame"]), int(row["stop_frame"])
        pcm[start:stop] = slot
        name = str(row["name"])
        row["payload_sha256"] = _array_sha256(payload)
        row["submitted_slot_sha256"] = _array_sha256(slot)
        spectrum = np.abs(np.fft.rfft(payload.astype(np.float64))) ** 2
        frequency = np.fft.rfftfreq(PERIOD, 1.0 / SAMPLE_RATE)
        in_band = (frequency >= LIVE_SAFE_BAND_HZ[0]) & (
            frequency < LIVE_SAFE_BAND_HZ[1]
        )
        total_energy = float(np.sum(spectrum))
        out_fraction = float(np.sum(spectrum[~in_band]) / total_energy)
        plan["pe_payloads"][name] = {
            "seed": PE_SEEDS[(role, path)],
            "generator": "flat_dft_bin_energy_80_2828_actual_int16",
            "peak_pcm": PE_PEAK_PCM,
            "sha256": _array_sha256(payload),
            "authority_band_hz": list(LIVE_SAFE_BAND_HZ),
            "actual_int16_out_of_band_energy_fraction": out_fraction,
            "unrestricted_1024tap_fit_allowed": False,
        }
    # live phase-1에서 두 path의 49/98 PCM diagnostic raw를 먼저 봉인·판정한 뒤에만
    # phase-2 P/S PE를 열 수 있도록 fallback의 segment 순서를 바꾼다. 프레임 수/각
    # segment bytes는 바꾸지 않는다.
    old_pcm = np.array(pcm, copy=True, order="C")
    old_layout = list(plan["layout"])
    ordered_layout = (
        [row for row in old_layout if row["kind"] == "zero_pre"]
        + [
            row
            for row in old_layout
            if row["kind"] == "multilevel_two_tone_diagnostic_slot"
        ]
        + [row for row in old_layout if row["kind"] == "pilot_only_lead"]
        + [row for row in old_layout if row["kind"] == "pe_slot"]
        + [row for row in old_layout if row["kind"] == "pilot_only_tail"]
        + [row for row in old_layout if row["kind"] == "zero_post"]
    )
    if len(ordered_layout) != len(old_layout):
        raise AssertionError("live-safe fallback reorder가 layout row를 잃었습니다")
    frame_keys = (
        "start_frame",
        "stop_frame",
        "central_start_frame",
        "central_stop_frame",
        "guard_lead_start_frame",
        "guard_lead_stop_frame",
        "active_start_frame",
        "active_stop_frame",
        "guard_tail_start_frame",
        "guard_tail_stop_frame",
        "analysis_start_frame",
        "analysis_stop_frame",
    )
    rendered: list[np.ndarray] = []
    cursor = 0
    for row in ordered_layout:
        old_start, old_stop = int(row["start_frame"]), int(row["stop_frame"])
        rendered.append(old_pcm[old_start:old_stop])
        delta = cursor - old_start
        for key in frame_keys:
            if key in row:
                row[key] = int(row[key]) + delta
        cursor += old_stop - old_start
    if cursor != TOTAL_FRAMES:
        raise AssertionError("live-safe fallback reorder frame 합이 24초가 아닙니다")
    pcm = np.ascontiguousarray(np.concatenate(rendered, axis=0), dtype=np.int16)
    plan["layout"] = ordered_layout
    diagnostic_rows = [
        row
        for row in ordered_layout
        if row["kind"] == "multilevel_two_tone_diagnostic_slot"
    ]
    plan["nonlinearity_diagnostics"]["slots"] = json.loads(
        json.dumps(diagnostic_rows, ensure_ascii=False, allow_nan=False)
    )
    lead_row = next(row for row in ordered_layout if row["kind"] == "pilot_only_lead")
    tail_row = next(row for row in ordered_layout if row["kind"] == "pilot_only_tail")
    pe_rows = [row for row in ordered_layout if row["kind"] == "pe_slot"]
    plan["pilot"].update(
        {
            "main_fit_epoch_start_frame": int(lead_row["start_frame"]),
            "main_fit_epoch_stop_frame": max(int(row["stop_frame"]) for row in pe_rows),
            "terminal_epoch_start_frame": int(tail_row["start_frame"]),
            "terminal_epoch_stop_frame": int(tail_row["stop_frame"]),
            "diagnostic_gap_contains_pilot": False,
        }
    )
    diagnostic_phase_stop = max(int(row["stop_frame"]) for row in diagnostic_rows)
    plan["live_phase_contract"] = {
        "meter_phase_frames": 960_000,
        "meter_phase_seconds": 20.0,
        "diagnostic_phase_start_frame": 0,
        "diagnostic_phase_stop_frame": diagnostic_phase_stop,
        "diagnostic_phase_seconds": diagnostic_phase_stop / SAMPLE_RATE,
        "ps_phase_start_frame": diagnostic_phase_stop,
        "ps_phase_stop_frame": TOTAL_FRAMES,
        "ps_phase_seconds": (TOTAL_FRAMES - diagnostic_phase_stop) / SAMPLE_RATE,
        "total_signal_seconds": SIGNAL_SECONDS,
        "maximum_total_output_seconds": 44.0,
        "diagnostic_raw_publish_and_pass_required_before_ps_phase": True,
        "stream_count": 2,
        "diagnostic_stream": {
            "stream_index": 1,
            "local_start_frame": 0,
            "local_stop_frame": diagnostic_phase_stop,
            "logical_plan_start_frame": 0,
            "logical_plan_stop_frame": diagnostic_phase_stop,
        },
        "ps_stream": {
            "stream_index": 2,
            "local_start_frame": 0,
            "local_stop_frame": TOTAL_FRAMES - diagnostic_phase_stop,
            "logical_plan_start_frame": diagnostic_phase_stop,
            "logical_plan_stop_frame": TOTAL_FRAMES,
        },
        "adc_clock_is_restarted_between_streams": True,
        "common_clock_scope": "ps_stream_only_local_coordinates",
        "concatenated_single_stream_clock_claim_forbidden": True,
        "automatic_retry_allowed": False,
    }
    plan["schema"] = "stage2_2khz_time_separated_bandlimited_dpss_plan_v1"
    plan["role"] = "signal_only_live_safe_fallback_no_audio_authority"
    plan["actual_submitted_pcm"]["sha256"] = _array_sha256(pcm)
    plan["actual_submitted_pcm"]["peak_pcm"] = int(
        np.max(np.abs(pcm.astype(np.int32)))
    )
    active_powers = []
    for row in plan["layout"]:
        if row["kind"] in {"pe_slot", "multilevel_two_tone_diagnostic_slot"}:
            slot = pcm[row["start_frame"] : row["stop_frame"]].astype(np.float64)
            active_powers.append(
                float(np.sum(np.mean((slot / 32768.0) ** 2, axis=0)))
            )
    worst_power = max(active_powers)
    meter_power = float(plan["meter_power_gate"]["official_meter_total_power"])
    if worst_power > meter_power:
        raise AssertionError("live-safe fallback active slot power가 meter를 초과합니다")
    plan["meter_power_gate"]["worst_active_slot_total_power"] = worst_power
    plan["meter_power_gate"]["passed"] = True
    basis, basis_receipt = build_stage2_bandlimited_dpss_basis()
    del basis
    plan["gram_contract"] = {
        "support_samples": SUPPORT_SAMPLES,
        "representation": "real_bandpass_modulated_dpss_subspace",
        "real_dof_per_path": DPSS_REAL_DOF_PER_PATH,
        "projected_gram_dimension": 2 * DPSS_REAL_DOF_PER_PATH,
        "basis_definition_sha256": basis_receipt["canonical_payload_sha256"],
        "basis_array_sha256": basis_receipt["basis_sha256"],
        "authority_band_hz": list(LIVE_SAFE_BAND_HZ),
        "maximum_normal_matrix_condition_number": MAX_GRAM_CONDITION,
        "unrestricted_1024tap_fit_allowed": False,
        "ridge_or_minimum_norm_nullspace_fill_allowed": False,
        "holdout_excluded": True,
    }
    plan["live_safety"] = {
        "status": LIVE_SAFE_FALLBACK_STATUS,
        "audio_execution_allowed_by_this_plan": False,
        "excitation_design_band_hz": list(LIVE_SAFE_BAND_HZ),
        "near_nyquist_targeted_pe_present": False,
        "actual_int16_quantization_out_of_band_energy_is_explicit_per_payload": True,
        "physical_level_route_and_thd_preflight_required": True,
        "actual_acoustic_ps_or_clock_authority_claimed": False,
    }
    plan["consumer_binding"] = {
        "training_eval_basis_sha_consumed": False,
        "authority_band_enforced": False,
        "canonical_training_eligible": False,
    }
    meter = expected_meter_output_pcm(noise_channel=0)
    plan["meter_submitted_pcm"] = {
        "dtype": meter.dtype.str,
        "shape": list(meter.shape),
        "sha256": _array_sha256(meter),
        "peak_pcm": int(np.max(np.abs(meter.astype(np.int32)))),
    }
    plan["artifacts"] = {
        "meter_raw": "results/stage2_2khz_ps_v2/meter_raw.npz",
        "diagnostic_phase_raw": "results/stage2_2khz_ps_v2/diagnostic_phase_raw.npz",
        "diagnostic_analysis_receipt": "results/stage2_2khz_ps_v2/diagnostic_analysis_receipt.json",
        "ps_phase_raw": "results/stage2_2khz_ps_v2/ps_phase_raw.npz",
        "analysis_receipt": "results/stage2_2khz_ps_v2/analysis_receipt.json",
        "analysis_arrays": "results/stage2_2khz_ps_v2/analysis_arrays.npz",
        "source_operating_level": "results/stage2_2khz_ps_v2/source_operating_level.json",
        "primary_candidate": "results/stage2_2khz_ps_v2/primary_path_candidate.npz",
        "secondary_candidate": "results/stage2_2khz_ps_v2/secondary_path_candidate.npz",
        "plant_binding": "results/stage2_2khz_ps_v2/plant_binding.json",
    }
    plan.pop("canonical_payload_sha256", None)
    plan["canonical_payload_sha256"] = _payload_sha256(plan)
    pcm.setflags(write=False)
    return plan, pcm


def validate_stage2_v2_live_safe_fallback_plan(
    plan: Mapping[str, Any], submitted_pcm: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    expected_plan, expected_pcm = build_stage2_v2_live_safe_fallback_plan()
    source = np.asarray(submitted_pcm)
    if dict(plan) != expected_plan:
        raise Stage2MeasurementV2Error("Stage-2 live-safe fallback plan이 canonical과 다릅니다")
    if (
        source.dtype != np.int16
        or source.shape != expected_pcm.shape
        or not np.array_equal(source, expected_pcm)
    ):
        raise Stage2MeasurementV2Error("Stage-2 live-safe fallback actual PCM bytes가 다릅니다")
    return expected_plan, expected_pcm


def audit_stage2_v2_live_safe_dpss_gram(
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    *,
    zeros_by_path: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    owned_plan, owned_pcm = validate_stage2_v2_live_safe_fallback_plan(
        plan, submitted_pcm
    )
    rows = {
        role: tuple(
            _central_period(owned_plan, owned_pcm, role=role, path=path)
            for path in PATHS
        )
        for role in FIT_ROLES
    }
    receipt = audit_stage2_bandlimited_dpss_projected_gram(
        rows, zeros_by_path=zeros_by_path
    )
    receipt.pop("canonical_payload_sha256", None)
    receipt["signal_plan_sha256"] = owned_plan["canonical_payload_sha256"]
    receipt["actual_submitted_pcm_sha256"] = owned_plan["actual_submitted_pcm"][
        "sha256"
    ]
    receipt["holdout_accessed"] = False
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt


def validate_stage2_v2_signal_plan(
    plan: Mapping[str, Any], submitted_pcm: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    """외부 plan/PCM이 canonical deterministic bytes인지 exact 비교한다."""

    expected_plan, expected_pcm = build_stage2_v2_signal_plan()
    source = np.asarray(submitted_pcm)
    if dict(plan) != expected_plan:
        raise Stage2MeasurementV2Error("Stage-2 v2 plan이 canonical payload와 다릅니다")
    if (
        source.dtype != np.int16
        or source.shape != expected_pcm.shape
        or not np.array_equal(source, expected_pcm)
        or _array_sha256(source) != plan["actual_submitted_pcm"]["sha256"]
    ):
        raise Stage2MeasurementV2Error("Stage-2 v2 submitted actual-int16 bytes가 다릅니다")
    return expected_plan, expected_pcm


def _central_period(
    plan: Mapping[str, Any], pcm: np.ndarray, *, role: str, path: str
) -> np.ndarray:
    matches = [
        row
        for row in plan["layout"]
        if row.get("kind") == "pe_slot"
        and row.get("role") == role
        and row.get("path") == path
    ]
    if len(matches) != 1:
        raise Stage2MeasurementV2Error(f"{role}/{path} PE slot이 정확히 하나가 아닙니다")
    row = matches[0]
    return np.ascontiguousarray(
        pcm[row["central_start_frame"] : row["central_stop_frame"]],
        dtype=np.int16,
    )


def audit_stage2_v2_shifted_gram(
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    *,
    zeros_by_path: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    """fit-a/fit-b actual submitted periods의 2048x2048 shifted Gram을 감사한다."""

    owned_plan, owned_pcm = validate_stage2_v2_signal_plan(plan, submitted_pcm)
    rows = {
        role: tuple(
            _central_period(owned_plan, owned_pcm, role=role, path=path)
            for path in PATHS
        )
        for role in FIT_ROLES
    }
    audit = exact_two_input_periodic_gram_audit(
        rows,
        role_order=FIT_ROLES,
        support_samples=SUPPORT_SAMPLES,
        zeros_by_input=zeros_by_path,
        maximum_condition_number=MAX_GRAM_CONDITION,
        crosscheck_tolerance=1.0e-10,
    )
    receipt = {
        "schema": GRAM_SCHEMA,
        "signal_plan_sha256": owned_plan["canonical_payload_sha256"],
        "actual_submitted_pcm_sha256": owned_plan["actual_submitted_pcm"]["sha256"],
        "holdout_accessed": False,
        "holdout_used_for_fit_or_support_selection": False,
        **audit,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt


@dataclass(frozen=True)
class _AccessEvent:
    sequence: int
    operation: str
    role: str | None
    path: str | None
    start_frame: int | None
    stop_frame: int | None
    value_sha256: str | None


class Stage2HoldoutAccessLedger:
    """fit freeze 이전 holdout 접근과 holdout 뒤 refit을 구조적으로 거부한다.

    이 객체는 접근 순서만 증명하며 물리 측정 PASS를 발행하지 않는다. capture는 생성 시
    private copy로 고정되고 caller가 임의 범위를 요청할 API를 제공하지 않는다.
    """

    def __init__(self, plan: Mapping[str, Any], capture: np.ndarray) -> None:
        if plan.get("schema") == SCHEMA:
            canonical, _ = validate_stage2_v2_signal_plan(
                plan, build_stage2_v2_signal_plan()[1]
            )
        elif plan.get("schema") == "stage2_2khz_time_separated_bandlimited_dpss_plan_v1":
            canonical, _ = validate_stage2_v2_live_safe_fallback_plan(
                plan, build_stage2_v2_live_safe_fallback_plan()[1]
            )
        else:
            raise Stage2HoldoutAccessError("지원하지 않는 Stage-2 v2 plan schema입니다")
        source = np.asarray(capture)
        if source.ndim != 2 or source.shape[0] != TOTAL_FRAMES or source.shape[1] < 1:
            raise Stage2HoldoutAccessError("capture는 exact Stage-2 frame axis를 가져야 합니다")
        if not np.issubdtype(source.dtype, np.number) or not np.all(np.isfinite(source)):
            raise Stage2HoldoutAccessError("capture가 숫자형 finite array가 아닙니다")
        self._plan = canonical
        self._capture = np.array(source, copy=True, order="C")
        self._capture.setflags(write=False)
        self._events: list[_AccessEvent] = []
        self._fit_reads: set[tuple[str, str]] = set()
        self._fit_frozen = False
        self._holdout_started = False

    def _record(self, operation: str, role: str | None, path: str | None, value: np.ndarray | None, start: int | None, stop: int | None) -> None:
        self._events.append(
            _AccessEvent(
                sequence=len(self._events),
                operation=operation,
                role=role,
                path=path,
                start_frame=start,
                stop_frame=stop,
                value_sha256=None if value is None else _array_sha256(value),
            )
        )

    def _bounded_period(self, role: str, path: str) -> tuple[np.ndarray, int, int]:
        matches = [
            row
            for row in self._plan["layout"]
            if row.get("kind") == "pe_slot"
            and row.get("role") == role
            and row.get("path") == path
        ]
        if len(matches) != 1:
            raise Stage2HoldoutAccessError("plan bounded period가 유일하지 않습니다")
        row = matches[0]
        start, stop = int(row["central_start_frame"]), int(row["central_stop_frame"])
        value = np.array(self._capture[start:stop], copy=True, order="C")
        value.setflags(write=False)
        return value, start, stop

    def read_fit_period(self, *, role: str, path: str) -> np.ndarray:
        if role not in FIT_ROLES or path not in PATHS:
            raise Stage2HoldoutAccessError("fit accessor는 fit_a/fit_b P/S만 허용합니다")
        if self._fit_frozen or self._holdout_started:
            raise Stage2HoldoutAccessError("fit freeze/holdout 뒤에는 refit 입력을 읽을 수 없습니다")
        value, start, stop = self._bounded_period(role, path)
        self._fit_reads.add((role, path))
        self._record("read_fit_period", role, path, value, start, stop)
        return value

    def freeze_fit(self) -> None:
        required = {(role, path) for role in FIT_ROLES for path in PATHS}
        if self._fit_frozen or self._holdout_started:
            raise Stage2HoldoutAccessError("fit state는 한 번만 freeze할 수 있습니다")
        if self._fit_reads != required:
            raise Stage2HoldoutAccessError("모든 fit-a/fit-b P/S bounded period를 읽기 전입니다")
        self._fit_frozen = True
        self._record("freeze_fit_state", None, None, None, None, None)

    def read_holdout_period(self, *, path: str) -> np.ndarray:
        if path not in PATHS:
            raise Stage2HoldoutAccessError("holdout accessor path가 잘못됐습니다")
        if not self._fit_frozen:
            raise Stage2HoldoutAccessError("fit freeze 전 holdout 접근은 금지됩니다")
        self._holdout_started = True
        value, start, stop = self._bounded_period(HOLDOUT_ROLE, path)
        self._record("read_untouched_holdout_period", HOLDOUT_ROLE, path, value, start, stop)
        return value

    def receipt(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": ACCESS_LEDGER_SCHEMA,
            "role": "access_order_evidence_only_no_physical_or_training_authority",
            "signal_plan_sha256": self._plan["canonical_payload_sha256"],
            "capture_snapshot_sha256": _array_sha256(self._capture),
            "fit_frozen": self._fit_frozen,
            "holdout_access_started": self._holdout_started,
            "refit_after_holdout_allowed": False,
            "authority_pass_claimed": False,
            "events": [event.__dict__.copy() for event in self._events],
        }
        payload["canonical_payload_sha256"] = _payload_sha256(payload)
        return payload


__all__ = [
    "ACCESS_LEDGER_SCHEMA",
    "DIAGNOSTIC_LEVELS_PCM",
    "GRAM_DIMENSION",
    "GRAM_SCHEMA",
    "LIVE_SAFETY_STATUS",
    "LIVE_SAFE_FALLBACK_STATUS",
    "MAX_GRAM_CONDITION",
    "MAX_MODEL_ACTUATOR_ABS",
    "MAX_SUBMITTED_PEAK_PCM",
    "SCHEMA",
    "SIGNAL_SECONDS",
    "Stage2HoldoutAccessError",
    "Stage2HoldoutAccessLedger",
    "Stage2MeasurementV2Error",
    "TOTAL_FRAMES",
    "audit_stage2_bandlimited_dpss_projected_gram",
    "audit_stage2_v2_live_safe_dpss_gram",
    "audit_stage2_v2_shifted_gram",
    "build_stage2_bandlimited_dpss_basis",
    "build_stage2_v2_live_safe_fallback_plan",
    "build_stage2_v2_signal_plan",
    "validate_stage2_v2_live_safe_fallback_plan",
    "validate_stage2_v2_signal_plan",
]
