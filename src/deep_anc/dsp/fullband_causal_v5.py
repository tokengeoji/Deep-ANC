"""v5 광대역 causal P/S 식별용 신호 전용 계약.

이 모듈은 오디오 장치를 열지 않는다. v4의 comb-null PE를 재사용하지 않고, P와 S를
시간 분리한 actual-int16 near-white 입력으로 1024-sample joint causal FIR의 지속 여기
조건을 검사한다. 실제 raw가 없으므로 live authority는 의도적으로 ``None``이다.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Any, Mapping

import numpy as np
from scipy.linalg import eigh, toeplitz
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline

from .control_band_contract import BroadbandFullOctaveContractV3
from .fullband_causal_v4 import continuous_pilot_period
from .measurement_level import expected_meter_output_pcm

FS = 48_000
BLOCK = 256
PERIOD = 32_768
CYCLIC_PREFIX = 16_384
CYCLIC_SUFFIX = 16_384
SLOT_FRAMES = CYCLIC_PREFIX + PERIOD + CYCLIC_SUFFIX
SUPPORTS = (1_024, 2_048, 4_096, 8_192)
CONDITION_AUDIT_SUPPORT = 1_024
MAX_CONDITION = 20.0
CLOCK_HARD_MAX_RESIDUAL_SAMPLES = 0.06755189029558946
CLOCK_VIEW_MAX_ENDPOINT_DISAGREEMENT_SAMPLES = 0.05
PE_PEAK_PCM = 49
SUBMITTED_PEAK_LIMIT_PCM = 98
EXCITATION_QUALIFICATION_BAND_HZ = (80.0, 11_313.7084989848)
SUBBAND_MAX_RELATIVE_RESIDUAL = 0.10
SUBBAND_MIN_COMPLEX_AGREEMENT = 0.995
SUBBAND_MIN_RESPONSE_TO_NOISE_DB = 20.0
ROLES = ("fit_a", "fit_b", "holdout")
PATH_CHANNEL = {"primary": 0, "secondary": 1}
SEEDS = {
    ("primary", "fit_a"): 710_001,
    ("secondary", "fit_a"): 710_003,
    ("primary", "fit_b"): 710_021,
    ("secondary", "fit_b"): 710_023,
    ("primary", "holdout"): 710_041,
    ("secondary", "holdout"): 710_043,
}
LIVE_AUTHORITY = None
CANONICAL_BLOCKER = (
    "signal-only 설계와 합성 fixture는 실제 raw P/S, 8대역 SNR/agreement/residual, "
    "stationarity/change-point 증거를 대신하지 않는다"
)


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
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


def _safe_raw_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".npz":
        raise ValueError("raw publisher path는 저장소 내부의 lexical .npz 상대경로여야 합니다")
    if not path.parts or path.parts[0] != "results":
        raise ValueError("raw publisher path는 results/ 아래여야 합니다")
    return path.as_posix()


@lru_cache(maxsize=None)
def _near_white_period_cached(seed: int) -> np.ndarray:
    """DC/Nyquist까지 실제 int16로 존재하는 결정론적 Rademacher PE."""

    rng = np.random.default_rng(int(seed))
    bits = rng.integers(0, 2, size=PERIOD, dtype=np.int8)
    signal = np.where(bits == 0, -PE_PEAK_PCM, PE_PEAK_PCM).astype(np.int16)
    signal.setflags(write=False)
    return signal


def near_white_period(seed: int) -> np.ndarray:
    return _near_white_period_cached(int(seed)).copy()


def _central_role_period(
    submitted: np.ndarray, layout: list[dict[str, Any]], path: str, role: str
) -> np.ndarray:
    match = [row for row in layout if row.get("path") == path and row.get("role") == role]
    if len(match) != 1:
        raise ValueError(f"{path}/{role} 중앙 period가 정확히 하나가 아닙니다")
    row = match[0]
    return np.asarray(
        submitted[row["central_start_frame"] : row["central_stop_frame"]],
        dtype=np.float64,
    )


def build_plan_v5(
    *, raw_session_relative_path: str = "results/fullband_causal_v5/raw_capture.npz"
) -> tuple[dict[str, Any], np.ndarray]:
    """actual submitted PCM과 immutable signal-only plan을 생성한다."""

    raw_path = _safe_raw_relative_path(raw_session_relative_path)
    contract = BroadbandFullOctaveContractV3.canonical()
    contract_payload = contract.model_dump(mode="json")
    contract_sha = _payload_sha256(contract_payload)

    pilot_period = continuous_pilot_period()
    pilot_spectrum = np.fft.rfft(pilot_period.astype(np.float64), axis=0)
    pilot_frequency = np.fft.rfftfreq(PERIOD, 1.0 / FS)
    pilot_band = (pilot_frequency >= 152.0) & (pilot_frequency <= 600.0)
    primary_pilot_bins = np.flatnonzero(
        pilot_band
        & (np.abs(pilot_spectrum[:, 0]) > 1.0e3)
        & (np.abs(pilot_spectrum[:, 1]) <= 1.0e-8)
    )
    secondary_pilot_bins = np.flatnonzero(
        pilot_band
        & (np.abs(pilot_spectrum[:, 1]) > 1.0e3)
        & (np.abs(pilot_spectrum[:, 0]) <= 1.0e-8)
    )
    if min(primary_pilot_bins.size, secondary_pilot_bins.size) < 8:
        raise AssertionError("actual pilot의 경로별 exact-null line이 부족합니다")
    parts: list[np.ndarray] = [np.zeros((2 * PERIOD, 2), dtype=np.int16)]
    layout: list[dict[str, Any]] = [
        {
            "kind": "pilot_only_lead",
            "start_frame": 0,
            "stop_frame": 2 * PERIOD,
            "frames": 2 * PERIOD,
        }
    ]
    cursor = 2 * PERIOD
    payload_meta: dict[str, Any] = {}

    for role in ROLES:
        for path in ("primary", "secondary"):
            seed = SEEDS[(path, role)]
            pe = near_white_period(seed)
            channel = PATH_CHANNEL[path]
            slot = np.zeros((SLOT_FRAMES, 2), dtype=np.int16)
            slot[:CYCLIC_PREFIX, channel] = pe[-CYCLIC_PREFIX:]
            slot[CYCLIC_PREFIX : CYCLIC_PREFIX + PERIOD, channel] = pe
            slot[CYCLIC_PREFIX + PERIOD :, channel] = pe[:CYCLIC_SUFFIX]
            central = cursor + CYCLIC_PREFIX
            parts.append(slot)
            name = f"{path}_{role}"
            layout.append(
                {
                    "kind": f"{name}_slot",
                    "path": path,
                    "role": role,
                    "start_frame": cursor,
                    "stop_frame": cursor + SLOT_FRAMES,
                    "central_start_frame": central,
                    "central_stop_frame": central + PERIOD,
                    "pre_boundary_exclusion_samples": CYCLIC_PREFIX,
                    "post_boundary_exclusion_samples": CYCLIC_SUFFIX,
                    "payload_pcm_sha256": _array_sha256(pe),
                }
            )
            spectrum = np.fft.rfft(pe.astype(np.float64))
            frequency = np.fft.rfftfreq(PERIOD, 1.0 / FS)
            qualified = (frequency >= EXCITATION_QUALIFICATION_BAND_HZ[0]) & (
                frequency <= EXCITATION_QUALIFICATION_BAND_HZ[1]
            )
            payload_meta[name] = {
                "seed": seed,
                "distribution": "deterministic_rademacher_actual_int16",
                "peak_pcm": PE_PEAK_PCM,
                "rms_pcm": PE_PEAK_PCM,
                "pcm_sha256": _array_sha256(pe),
                "qualified_bin_nonzero_fraction": float(
                    np.mean(np.abs(spectrum[qualified]) > 0.0)
                ),
            }
            cursor += SLOT_FRAMES

    parts.append(np.zeros((3 * PERIOD, 2), dtype=np.int16))
    layout.append(
        {
            "kind": "pilot_only_tail",
            "start_frame": cursor,
            "stop_frame": cursor + 3 * PERIOD,
            "frames": 3 * PERIOD,
        }
    )
    highband = np.concatenate(parts, axis=0)
    repeats = math.ceil(len(highband) / PERIOD)
    pilot = np.tile(pilot_period, (repeats, 1))[: len(highband)]
    submitted32 = highband.astype(np.int32) + pilot.astype(np.int32)
    peak = int(np.max(np.abs(submitted32)))
    if peak > SUBMITTED_PEAK_LIMIT_PCM:
        raise AssertionError(f"actual submitted peak {peak} > {SUBMITTED_PEAK_LIMIT_PCM}")
    submitted = submitted32.astype(np.int16)
    if len(submitted) % BLOCK:
        raise AssertionError("v5 signal은 256-frame block aligned여야 합니다")
    active_total_powers = []
    for row in layout:
        if row.get("role") in ROLES:
            slot = submitted[row["start_frame"] : row["stop_frame"]].astype(np.float64) / 32768.0
            active_total_powers.append(float(np.sum(np.mean(slot**2, axis=0))))
    worst_total_power = max(active_total_powers)
    meter_pcm = expected_meter_output_pcm(noise_channel=0)
    meter_total_power = float(
        np.sum(np.mean((meter_pcm.astype(np.float64) / 32768.0) ** 2, axis=0))
    )
    if worst_total_power > meter_total_power:
        raise AssertionError("v5 active-slot total power가 official meter recipe를 초과합니다")

    physical_bands = contract_payload["physical_identification_subbands_hz"]
    plan: dict[str, Any] = {
        "schema": "fullband_causal_time_separated_near_white_v5",
        "role": "signal_only_dry_run_no_audio",
        "sample_rate": FS,
        "block_size": BLOCK,
        "duration_seconds": len(submitted) / FS,
        "live_authority": LIVE_AUTHORITY,
        "live_capture_enabled": False,
        "canonical_training_eligible": False,
        "canonical_blocker": CANONICAL_BLOCKER,
        "control_band_contract": contract_payload,
        "control_band_contract_sha256": contract_sha,
        "excitation": {
            "kind": "actual_int16_near_white_time_separated_by_path",
            "qualification_band_hz": list(EXCITATION_QUALIFICATION_BAND_HZ),
            "actual_spectral_extent_hz": [0.0, FS / 2.0],
            "p_and_s_main_pe_simultaneously_active": False,
            "payloads": payload_meta,
        },
        "layout": layout,
        "clock_contract": {
            "method": "matching_path_actual_submitted_two_input_spectral_fit",
            "pilot_present_for_every_submitted_frame": True,
            "actual_submitted_denominator_includes_near_white_pe": True,
            "pe_contamination_ignored": False,
            "opposite_path_actual_dft_null_required_at_selected_pilot_lines": True,
            "err_ref_primary_secondary_one_common_affine_map_required": True,
            "fit_roles": ["fit_a", "fit_b"],
            "holdout_used_for_fit_or_selection": False,
            "piecewise_or_change_point_must_reject": True,
            "primary_pilot_bins": primary_pilot_bins.tolist(),
            "secondary_pilot_bins": secondary_pilot_bins.tolist(),
            "pilot_frequencies_are_adc_observed_rate_witness": True,
        },
        "plant_identification": {
            "operator": "joint_two_input_finite_causal_fir_actual_full_input",
            "candidate_support_samples": list(SUPPORTS),
            "selected_support_before_live_evidence": CONDITION_AUDIT_SUPPORT,
            "support_1024_exact_condition_required_max": MAX_CONDITION,
            "support_2048_4096_8192_condition_status": "NOT_AUDITED_NO_CLAIM",
            "fit_roles": ["fit_a", "fit_b"],
            "terminal_holdout_role": "holdout",
            "holdout_used_for_generation_or_selection": False,
            "physical_identification_subbands_hz": physical_bands,
            "each_path_mic_band_requires_snr_residual_complex_agreement": True,
            "independent_coherence_available": False,
            "global_residual_alone_sufficient": False,
            "finite_memory_proved_by_finite_capture": False,
        },
        "boundary_contract": {
            "path_switch_and_period_boundary_both_sides_excluded": True,
            "pre_exclusion_samples": CYCLIC_PREFIX,
            "post_exclusion_samples": CYCLIC_SUFFIX,
            "support_1024_fits_inside_each_exclusion": True,
        },
        "publisher_contract": {
            "raw_session_relative_path": raw_path,
            "raw_npz_schema": "fullband_causal_raw_capture_v5",
            "lexical_parent_symlink_rejected": True,
            "atomic_sibling_staging_hardlink_noreplace": True,
            "flush_and_fsync_file_and_parent": True,
            "required_arrays": ["submitted_pcm", "captured_pcm", "callback_frames"],
            "role": "fixture_only_raw_container_v5",
            "callback_semantics": "frame_accounting_only",
            "live_xrun_slip_authority": False,
            "live_authority_at_plan_time": None,
        },
        "actual_submitted_pcm_sha256": _array_sha256(submitted),
        "actual_submitted_shape": list(submitted.shape),
        "actual_submitted_dtype": submitted.dtype.str,
        "actual_submitted_peak_pcm": peak,
        "measurement_level_safety": {
            "official_meter_output_pcm_sha256": _array_sha256(meter_pcm),
            "official_meter_reference_total_power": meter_total_power,
            "worst_active_slot_total_power": worst_total_power,
            "normalization_divisor": 32768,
            "scope": "whole_active_slot_including_prefix_central_suffix",
            "worst_active_slot_vs_meter_db": 10.0
            * math.log10(worst_total_power / meter_total_power),
            "meter_total_power_not_exceeded": True,
        },
    }
    plan["canonical_payload_sha256"] = _payload_sha256(plan)
    return plan, submitted


def _validate_canonical_plan_and_pcm_v5(
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    *,
    allow_configured_raw_path: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    """허용된 raw 경로로 재생성한 canonical builder 결과만 반환한다."""

    if allow_configured_raw_path:
        publisher = plan.get("publisher_contract")
        if not isinstance(publisher, Mapping):
            raise ValueError("canonical v5 plan publisher contract가 없습니다")
        raw_path = publisher.get("raw_session_relative_path")
        if not isinstance(raw_path, str):
            raise ValueError("canonical v5 plan raw path가 문자열이 아닙니다")
        expected_plan, expected_pcm = build_plan_v5(
            raw_session_relative_path=raw_path
        )
    else:
        expected_plan, expected_pcm = build_plan_v5()
    submitted = np.asarray(submitted_pcm)
    if (
        dict(plan) != expected_plan
        or submitted.dtype != np.int16
        or submitted.shape != expected_pcm.shape
        or not np.array_equal(submitted, expected_pcm)
    ):
        raise ValueError("audit 입력이 canonical v5 plan/layout/payload/PCM이 아닙니다")
    return expected_plan, expected_pcm


def _periodic_cross_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.fft.ifft(
        np.conj(np.fft.fft(left.astype(np.float64)))
        * np.fft.fft(right.astype(np.float64))
    ).real


def _toeplitz_gram_block(correlation: np.ndarray, support: int) -> np.ndarray:
    # G[i,j] = sum_n x[n-i] y[n-j] = r_xy[i-j].
    first_col = correlation[np.arange(support) % PERIOD]
    first_row = correlation[(-np.arange(support)) % PERIOD]
    return toeplitz(first_col, first_row)


def _exact_condition_audit_with_shifts_v5(
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    *,
    support: int,
    zeros_by_path: tuple[int, int],
    schema: str,
) -> dict[str, Any]:
    if int(support) != CONDITION_AUDIT_SUPPORT:
        raise ValueError("v5는 support 1024만 exact audit했다; 더 긴 support를 추정하지 않습니다")
    raw_zeros = tuple(zeros_by_path)
    if len(raw_zeros) != 2 or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw_zeros
    ):
        raise ValueError("P/S shifted condition zeros는 exact integer pair여야 합니다")
    zeros = tuple(int(value) for value in raw_zeros)
    if any(value < 0 or value >= PERIOD for value in zeros):
        raise ValueError("P/S shifted condition zeros가 유효하지 않습니다")
    submitted = np.asarray(submitted_pcm)
    if submitted.dtype != np.int16 or _array_sha256(submitted) != plan.get(
        "actual_submitted_pcm_sha256"
    ):
        raise ValueError("plan과 exact actual-int16 submitted PCM이 일치하지 않습니다")
    gram = np.zeros((2 * support, 2 * support), dtype=np.float64)
    role_conditions: dict[str, float] = {}
    for role in ("fit_a", "fit_b"):
        role_gram = np.zeros_like(gram)
        for path in ("primary", "secondary"):
            period = _central_role_period(submitted, list(plan["layout"]), path, role)
            shifted_period = np.column_stack(
                [np.roll(period[:, channel], zeros[channel]) for channel in range(2)]
            )
            for left in range(2):
                for right in range(2):
                    correlation = _periodic_cross_correlation(
                        shifted_period[:, left], shifted_period[:, right]
                    )
                    block = _toeplitz_gram_block(correlation, support)
                    role_gram[
                        left * support : (left + 1) * support,
                        right * support : (right + 1) * support,
                    ] += block
        role_gram = (role_gram + role_gram.T) * 0.5
        lo = float(eigh(role_gram, subset_by_index=[0, 0], eigvals_only=True)[0])
        hi = float(
            eigh(
                role_gram,
                subset_by_index=[2 * support - 1, 2 * support - 1],
                eigvals_only=True,
            )[0]
        )
        if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0:
            raise ValueError("exact shifted Gram 고유값이 finite positive가 아닙니다")
        role_conditions[role] = hi / lo
        gram += role_gram
    gram = (gram + gram.T) * 0.5
    probe_errors = []
    grid = np.arange(2 * support, dtype=np.float64)
    for probe_index, probe in enumerate((
        np.sin(grid * 0.017) + 0.25 * np.cos(grid * 0.031),
        np.cos(grid * 0.011) - 0.31 * np.sin(grid * 0.043),
        np.where((grid.astype(np.int64) % 17) < 8, 1.0, -1.0),
        np.exp(-grid / max(float(support), 1.0)) * np.cos(grid * 0.071),
    )):
        probe_fir = probe.reshape(2, support)
        probe_transfer = np.fft.rfft(probe_fir, n=PERIOD, axis=1)
        direct_prediction_energy = 0.0
        direct_normal_vector = np.zeros(2 * support, dtype=np.float64)
        for role in ("fit_a", "fit_b"):
            for path in ("primary", "secondary"):
                period = _central_role_period(submitted, list(plan["layout"]), path, role)
                shifted_period = np.column_stack(
                    [np.roll(period[:, channel], zeros[channel]) for channel in range(2)]
                )
                prediction = np.fft.irfft(
                    np.sum(np.fft.rfft(shifted_period, axis=0) * probe_transfer.T, axis=1),
                    n=PERIOD,
                )
                direct_prediction_energy += float(np.dot(prediction, prediction))
                prediction_fft = np.fft.rfft(prediction)
                for channel in range(2):
                    adjoint_period = np.fft.irfft(
                        np.conj(np.fft.rfft(shifted_period[:, channel]))
                        * prediction_fft,
                        n=PERIOD,
                    )
                    direct_normal_vector[
                        channel * support:(channel + 1) * support
                    ] += adjoint_period[:support]
        gram_prediction_energy = float(probe @ gram @ probe)
        gram_normal_vector = gram @ probe
        vector_relative_error = float(
            np.linalg.norm(direct_normal_vector - gram_normal_vector)
            / max(np.linalg.norm(direct_normal_vector), np.linalg.norm(gram_normal_vector), 1.0)
        )
        relative_error = abs(direct_prediction_energy - gram_prediction_energy) / max(
            abs(direct_prediction_energy), abs(gram_prediction_energy), 1.0
        )
        probe_errors.append({
            "probe_index": probe_index,
            "quadratic_form_relative_error": relative_error,
            "normal_vector_relative_error": vector_relative_error,
        })
    quadratic_relative_error = max(row["quadratic_form_relative_error"] for row in probe_errors)
    normal_vector_relative_error = max(row["normal_vector_relative_error"] for row in probe_errors)
    if max(quadratic_relative_error, normal_vector_relative_error) > 1.0e-10:
        raise ValueError("exact shifted Gram과 실제 circular operator가 일치하지 않습니다")
    lo = float(eigh(gram, subset_by_index=[0, 0], eigvals_only=True)[0])
    hi = float(
        eigh(
            gram,
            subset_by_index=[2 * support - 1, 2 * support - 1],
            eigvals_only=True,
        )[0]
    )
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0:
        raise ValueError("joint exact shifted Gram 고유값이 finite positive가 아닙니다")
    # 여기서 보고하는 condition은 kappa_2(A)가 아니라
    # periodic normal-matrix Gram의 kappa_2(G)=lambda_max/lambda_min=kappa_2(A)^2다.
    condition = hi / lo
    operator_definition = {
        "operator": "two_row_two_input_periodic_convolution_finite_support",
        "period_samples": PERIOD,
        "support_samples": support,
        "row_order": ["primary_slot", "secondary_slot"],
        "coefficient_path_order": ["primary", "secondary"],
        "zeros_before_fir_samples": {
            "primary": zeros[0],
            "secondary": zeros[1],
        },
        "shift_convention": "x_shifted[n]=x[(n-zeros_before_fir) mod period]",
        "equivalent_frequency_phase": "exp(-j*2*pi*k*zeros/period)",
    }
    receipt = {
        "schema": schema,
        "support_samples": support,
        "signal_plan_payload_sha256": plan.get("canonical_payload_sha256"),
        "actual_submitted_pcm_sha256": _array_sha256(submitted),
        "operator_definition": operator_definition,
        "operator_definition_sha256": _payload_sha256(operator_definition),
        "operator_quadratic_form_relative_error": quadratic_relative_error,
        "operator_quadratic_form_probe_receipts": probe_errors,
        "operator_normal_vector_relative_error": normal_vector_relative_error,
        "operator_quadratic_form_maximum_allowed": 1.0e-10,
        "operator_quadratic_form_crosscheck_passed": True,
        "zeros_before_fir_samples": [zeros[0], zeros[1]],
        "fit_roles": ["fit_a", "fit_b"],
        "role_condition_numbers": role_conditions,
        "periodic_normal_matrix_gram_condition_number": condition,
        "joint_fit_condition_number": condition,
        "condition_scope": "exact finite-support periodic normal matrix X^T X; not an acoustic transfer condition",
        "minimum_eigenvalue": lo,
        "maximum_eigenvalue": hi,
        "maximum_allowed": MAX_CONDITION,
        "passed": bool(condition <= MAX_CONDITION and all(v <= MAX_CONDITION for v in role_conditions.values())),
        "longer_supports": {
            str(value): "NOT_AUDITED_NO_CLAIM" for value in SUPPORTS if value != support
        },
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt


def exact_condition_audit_v5(
    plan: Mapping[str, Any], submitted_pcm: np.ndarray, *, support: int = 1_024
) -> dict[str, Any]:
    """canonical unshifted actual PCM의 exact periodic Gram kappa를 계산한다."""

    owned_plan, owned_pcm = _validate_canonical_plan_and_pcm_v5(
        plan, submitted_pcm, allow_configured_raw_path=True
    )
    return _exact_condition_audit_with_shifts_v5(
        owned_plan,
        owned_pcm,
        support=support,
        zeros_by_path=(0, 0),
        schema="fullband_causal_exact_gram_condition_v5",
    )


def exact_shifted_condition_audit_v5(
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    *,
    zeros_by_path: tuple[int, int],
    support: int = 1_024,
) -> dict[str, Any]:
    """실제 compact P/S zeros를 적용한 operator의 exact Gram을 계산한다."""

    source_plan_bytes = json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    source_pcm = np.asarray(submitted_pcm)
    source_pcm_sha = _array_sha256(source_pcm)
    owned_plan = json.loads(source_plan_bytes.decode("utf-8"))
    owned_pcm = np.array(source_pcm, copy=True, order="C")
    if source_plan_bytes != json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") or source_pcm_sha != _array_sha256(source_pcm):
        raise ValueError("shifted condition audit input TOCTOU mutation")
    owned_plan, owned_pcm = _validate_canonical_plan_and_pcm_v5(
        owned_plan, owned_pcm, allow_configured_raw_path=False
    )
    receipt = _exact_condition_audit_with_shifts_v5(
        owned_plan,
        owned_pcm,
        support=support,
        zeros_by_path=zeros_by_path,
        schema="fullband_causal_shifted_exact_gram_condition_v5",
    )
    exit_plan_bytes = json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    exit_pcm_sha = _array_sha256(np.asarray(submitted_pcm))
    entry_exit_equal = bool(
        source_plan_bytes == exit_plan_bytes and source_pcm_sha == exit_pcm_sha
    )
    if not entry_exit_equal:
        raise ValueError("shifted condition heavy audit 중 source plan/PCM TOCTOU mutation")
    receipt.pop("canonical_payload_sha256", None)
    receipt["owned_input_receipt"] = {
        "canonical_plan_exact": True,
        "source_plan_sha256": hashlib.sha256(source_plan_bytes).hexdigest(),
        "owned_plan_sha256": hashlib.sha256(source_plan_bytes).hexdigest(),
        "source_pcm_entry_sha256": source_pcm_sha,
        "source_pcm_exit_sha256": exit_pcm_sha,
        "owned_pcm_sha256": _array_sha256(owned_pcm),
        "source_plan_exit_sha256": hashlib.sha256(exit_plan_bytes).hexdigest(),
        "toctou_entry_exit_equal": entry_exit_equal,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt


def estimate_common_affine_clock_v5(
    *,
    row_times: np.ndarray,
    phase_radians: np.ndarray,
    frequencies_hz: np.ndarray,
) -> dict[str, Any]:
    """actual-input transfer phase의 시간 기울기에서 공통 affine rate를 추정한다.

    입력 phase는 반드시 captured FFT / matching-path actual submitted FFT로 얻어야 한다.
    이 함수는 amplitude/고정 LTI phase를 view별 intercept로 profile out한다.
    """

    times = np.asarray(row_times, dtype=np.float64)
    phase = np.asarray(phase_radians, dtype=np.float64)
    frequency_input = np.asarray(frequencies_hz, dtype=np.float64)
    if phase.ndim != 3 or phase.shape[0] != times.size:
        raise ValueError("phase shape은 [row, view, pilot_bin]이어야 합니다")
    if frequency_input.ndim == 1:
        frequency = np.broadcast_to(frequency_input[None, :], phase.shape[1:])
    elif frequency_input.shape == phase.shape[1:]:
        frequency = frequency_input
    else:
        raise ValueError("frequency는 [pilot_bin] 또는 [view,pilot_bin]이어야 합니다")
    if np.any(frequency <= 0.0):
        raise ValueError("clock pilot frequency는 양수여야 합니다")
    unwrapped = np.unwrap(phase, axis=0)

    def objective(ppm: float) -> float:
        slope = 2.0 * np.pi * frequency * (ppm * 1.0e-6)
        corrected = unwrapped - times[:, None, None] * slope[None, :, :]
        centered = corrected - np.mean(corrected, axis=0, keepdims=True)
        return float(np.mean(centered**2))

    fit = minimize_scalar(objective, bounds=(-1_000.0, 1_000.0), method="bounded")
    ppm = float(fit.x)
    corrected = unwrapped - times[:, None, None] * (
        2.0 * np.pi * frequency * ppm * 1.0e-6
    )[None, :, :]
    residual = corrected - np.mean(corrected, axis=0, keepdims=True)
    residual_samples = residual / (2.0 * np.pi * frequency[None, :, :]) * FS
    row_rms_samples = np.sqrt(np.mean(residual_samples**2, axis=(1, 2)))
    maximum_residual = float(np.max(np.abs(residual_samples)))
    change = (
        float(np.max(np.abs(np.diff(residual_samples, axis=0))))
        if len(residual_samples) > 1
        else 0.0
    )
    view_ppm: list[float] = []
    for view in range(phase.shape[1]):
        view_phase = unwrapped[:, view : view + 1, :]

        def view_objective(candidate_ppm: float) -> float:
            slope = 2.0 * np.pi * frequency[view] * (candidate_ppm * 1.0e-6)
            adjusted = view_phase - times[:, None, None] * slope[None, None, :]
            adjusted -= np.mean(adjusted, axis=0, keepdims=True)
            return float(np.mean(adjusted**2))

        view_ppm.append(
            float(
                minimize_scalar(
                    view_objective,
                    bounds=(-1_000.0, 1_000.0),
                    method="bounded",
                ).x
            )
        )
    elapsed = float(np.max(times) - np.min(times))
    view_disagreement = (
        (max(view_ppm) - min(view_ppm)) * 1.0e-6 * elapsed * FS
        if view_ppm
        else math.inf
    )
    passed = bool(
        maximum_residual <= CLOCK_HARD_MAX_RESIDUAL_SAMPLES
        and change <= CLOCK_HARD_MAX_RESIDUAL_SAMPLES
        and view_disagreement <= CLOCK_VIEW_MAX_ENDPOINT_DISAGREEMENT_SAMPLES
    )
    receipt = {
        "schema": "fullband_causal_common_affine_clock_fit_v5",
        "estimated_ppm": ppm,
        "objective": float(fit.fun),
        "view_estimated_ppm": view_ppm,
        "row_residual_rms_samples": row_rms_samples.tolist(),
        "maximum_residual_samples": maximum_residual,
        "maximum_change_point_samples": change,
        "maximum_view_endpoint_disagreement_samples": float(view_disagreement),
        "hard_max_residual_samples": CLOCK_HARD_MAX_RESIDUAL_SAMPLES,
        "view_max_endpoint_disagreement_samples": (
            CLOCK_VIEW_MAX_ENDPOINT_DISAGREEMENT_SAMPLES
        ),
        "passed": passed,
        "actual_submitted_denominator_required": True,
        "fixed_lti_phase_profiled_as_intercept": True,
    }
    return receipt


def _clock_waveform_rows_v5(
    plan: Mapping[str, Any],
    path: str,
    *,
    validation_policy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layout = list(plan["layout"])
    lead = next(row for row in layout if row["kind"] == "pilot_only_lead")
    tail = next(row for row in layout if row["kind"] == "pilot_only_tail")
    fit = [
        {"name": "lead", "start": int(lead["start_frame"]) + PERIOD, "role": "fit_reference"}
    ]
    for role in ("fit_a", "fit_b"):
        row = next(row for row in layout if row.get("path") == path and row.get("role") == role)
        fit.append({"name": role, "start": int(row["central_start_frame"]), "role": role})
    tail_validation = {
        "name": "tail",
        "start": int(tail["start_frame"]) + PERIOD,
        "role": "pilot_only_tail",
    }
    if validation_policy == "holdout_and_tail_legacy":
        holdout = next(
            row
            for row in layout
            if row.get("path") == path and row.get("role") == "holdout"
        )
        validation = [
            {
                "name": "holdout",
                "start": int(holdout["central_start_frame"]),
                "role": "operator_holdout",
            },
            tail_validation,
        ]
    elif validation_policy == "pilot_tail_only_pre_operator_holdout":
        validation = [tail_validation]
    else:
        raise ValueError("알 수 없는 clock validation policy입니다")
    return fit, validation


def _waveform_transfer_bank_v5(
    *,
    plan: Mapping[str, Any],
    submitted: np.ndarray,
    rate_ratio: float,
    path: str,
    rows: list[dict[str, Any]],
    accessors: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    channel = PATH_CHANNEL[path]
    bins = np.asarray(plan["clock_contract"][f"{path}_pilot_bins"], dtype=np.int64)
    transfers: list[np.ndarray] = []
    for row in rows:
        start = int(row["start"])
        dac_q = np.arange(start, start + PERIOD, dtype=np.float64)
        adc_q = dac_q / float(rate_ratio)
        accessor = accessors[row["name"]]
        lower = int(accessor["lower_adc_frame"])
        upper = int(accessor["upper_adc_frame_exclusive"])
        if adc_q[0] < lower or adc_q[-1] > upper - 1:
            raise ValueError("candidate q가 captured raw 범위를 벗어납니다")
        captured_period = np.column_stack(
            [interpolator(adc_q) for interpolator in accessor["interpolators"]]
        )
        captured_spectrum = np.fft.rfft(captured_period, axis=0)
        submitted_period = submitted[start : start + PERIOD].astype(np.float64)
        submitted_spectrum = np.fft.rfft(submitted_period, axis=0)
        opposite = 1 - channel
        if float(np.max(np.abs(submitted_spectrum[bins, opposite]))) > 1.0e-8:
            raise ValueError("matching-path pilot line에서 반대 DAC actual DFT가 exact zero가 아닙니다")
        denominator = submitted_spectrum[bins, channel]
        if float(np.min(np.abs(denominator))) <= 1.0:
            raise ValueError("actual submitted clock denominator가 너무 작습니다")
        transfers.append((captured_spectrum[bins, :] / denominator[:, None]).T)
    return np.stack(transfers, axis=0)


def _clock_local_accessor_v5(
    captured_adc: np.ndarray,
    *,
    row: Mapping[str, Any],
    interpolation_kind: str,
) -> dict[str, Any]:
    """한 clock row의 ±1000 ppm query support만 owned-copy한다."""

    start = int(row["start"])
    stop = start + PERIOD
    q_min = 1.0 - 1_000.0e-6
    q_max = 1.0 + 1_000.0e-6
    query_bounds = (
        start / q_min,
        start / q_max,
        (stop - 1) / q_min,
        (stop - 1) / q_max,
    )
    margin = 3 if interpolation_kind == "cubic" else 1
    lower = max(0, int(math.floor(min(query_bounds))) - margin)
    upper = min(len(captured_adc), int(math.ceil(max(query_bounds))) + margin + 1)
    if upper - lower < 4:
        raise ValueError("clock bounded local accessor support가 부족합니다")
    local = np.array(captured_adc[lower:upper], dtype=np.float64, copy=True, order="C")
    if local.ndim != 2 or local.shape[1] != 2 or not np.all(np.isfinite(local)):
        raise ValueError("clock bounded local capture가 finite [frame,2]가 아닙니다")
    grid = np.arange(lower, upper, dtype=np.float64)
    if interpolation_kind == "cubic":
        interpolators = [
            CubicSpline(grid, local[:, mic], extrapolate=False) for mic in range(2)
        ]
    elif interpolation_kind == "linear":
        interpolators = [
            (lambda query, mic=mic: np.interp(query, grid, local[:, mic]))
            for mic in range(2)
        ]
    else:
        raise ValueError("clock interpolation은 linear/cubic만 허용합니다")
    return {
        "interpolators": interpolators,
        "lower_adc_frame": lower,
        "upper_adc_frame_exclusive": upper,
        "receipt": {
            "name": row["name"],
            "role": row["role"],
            "logical_dac_start_frame": start,
            "logical_dac_stop_frame": stop,
            "owned_adc_start_frame": lower,
            "owned_adc_stop_frame": upper,
            "owned_local_capture_sha256": _array_sha256(local),
            "interpolation_kind": interpolation_kind,
        },
    }


def estimate_common_clock_from_waveforms_v5(
    *,
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    captured_adc_pcm: np.ndarray,
    interpolation_kind: str = "cubic",
    validation_policy: str = "holdout_and_tail_legacy",
) -> dict[str, Any]:
    """raw waveform에서 actual two-input denominator를 직접 사용해 공통 q를 추정한다."""

    submitted = np.asarray(submitted_pcm)
    captured = np.asarray(captured_adc_pcm)
    if submitted.dtype != np.int16 or _array_sha256(submitted) != plan.get("actual_submitted_pcm_sha256"):
        raise ValueError("signal plan SHA에 결속된 actual submitted PCM이 아닙니다")
    if captured.ndim != 2 or captured.shape[1] != 2:
        raise ValueError("captured ADC raw는 [frame,ERR_REF=2]여야 합니다")
    if plan.get("canonical_payload_sha256") != _payload_sha256(
        {key: value for key, value in plan.items() if key != "canonical_payload_sha256"}
    ):
        raise ValueError("signal plan canonical payload SHA가 유효하지 않습니다")

    path_rows = {
        path: _clock_waveform_rows_v5(
            plan,
            path,
            validation_policy=validation_policy,
        )
        for path in PATH_CHANNEL
    }
    if interpolation_kind not in {"cubic", "linear"}:
        raise ValueError("clock interpolation은 linear/cubic만 허용합니다")
    path_accessors = {
        path: tuple(
            {
                row["name"]: _clock_local_accessor_v5(
                    captured,
                    row=row,
                    interpolation_kind=interpolation_kind,
                )
                for row in rows
            }
            for rows in path_rows[path]
        )
        for path in PATH_CHANNEL
    }

    def banks(candidate_ratio: float, validation: bool = False) -> dict[str, np.ndarray]:
        index = 1 if validation else 0
        return {
            path: _waveform_transfer_bank_v5(
                plan=plan,
                submitted=submitted,
                rate_ratio=candidate_ratio,
                path=path,
                rows=path_rows[path][index],
                accessors=path_accessors[path][index],
            )
            for path in PATH_CHANNEL
        }

    def bank_objective(bank: np.ndarray) -> float:
        normalized = bank / np.maximum(np.abs(bank), np.finfo(np.float64).tiny)
        mean = np.mean(normalized, axis=0, keepdims=True)
        mean /= np.maximum(np.abs(mean), np.finfo(np.float64).tiny)
        return float(np.mean(np.abs(normalized - mean) ** 2))

    def objective_ratio(candidate_ratio: float) -> float:
        return float(sum(bank_objective(bank) for bank in banks(candidate_ratio).values()))

    fit = minimize_scalar(
        objective_ratio,
        bounds=(1.0 - 1_000.0e-6, 1.0 + 1_000.0e-6),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    ratio = float(fit.x)
    view_ratios: dict[str, float] = {}
    for path in PATH_CHANNEL:
        for mic in range(2):
            def view_objective(candidate_ratio: float, *, selected_path: str = path, selected_mic: int = mic) -> float:
                bank = banks(candidate_ratio)[selected_path][:, selected_mic : selected_mic + 1, :]
                return bank_objective(bank)

            view_fit = minimize_scalar(
                view_objective,
                bounds=(1.0 - 1_000.0e-6, 1.0 + 1_000.0e-6),
                method="bounded",
                options={"xatol": 1.0e-12},
            )
            view_ratios[f"{path}_{'ERR' if mic == 0 else 'REF'}"] = float(view_fit.x)
    elapsed = max(
        row["start"] for pair in path_rows.values() for rows in pair for row in rows
    ) - min(row["start"] for pair in path_rows.values() for rows in pair for row in rows)
    view_disagreement = (
        max(view_ratios.values()) - min(view_ratios.values())
    ) * elapsed

    fit_banks = banks(ratio)
    validation_banks = banks(ratio, validation=True)
    maximum_validation_samples = 0.0
    for path in PATH_CHANNEL:
        reference = np.mean(fit_banks[path], axis=0)
        frequency = (
            np.asarray(plan["clock_contract"][f"{path}_pilot_bins"], dtype=np.float64)
            * FS
            / PERIOD
        )
        for validation in validation_banks[path]:
            phase = np.angle(validation * np.conj(reference))
            sample_error = np.abs(phase) / (2.0 * np.pi * frequency[None, :]) * FS
            maximum_validation_samples = max(
                maximum_validation_samples, float(np.max(sample_error))
            )
    passed = bool(
        view_disagreement <= CLOCK_VIEW_MAX_ENDPOINT_DISAGREEMENT_SAMPLES
        and maximum_validation_samples <= CLOCK_HARD_MAX_RESIDUAL_SAMPLES
    )
    receipt = {
        "schema": "fullband_causal_raw_waveform_common_clock_v5",
        "signal_plan_payload_sha256": plan["canonical_payload_sha256"],
        "actual_submitted_pcm_sha256": plan["actual_submitted_pcm_sha256"],
        "captured_adc_full_sha256_computed": False,
        "accessed_waveform_receipts_by_path": {
            path: {
                "fit": [
                    accessor["receipt"]
                    for accessor in path_accessors[path][0].values()
                ],
                "validation": [
                    accessor["receipt"]
                    for accessor in path_accessors[path][1].values()
                ],
            }
            for path in PATH_CHANNEL
        },
        "estimated_rate_ratio": ratio,
        "interpolation_kind": interpolation_kind,
        "estimated_ppm": (ratio - 1.0) * 1.0e6,
        "view_rate_ratios": view_ratios,
        "maximum_view_endpoint_disagreement_samples": float(view_disagreement),
        "maximum_validation_phase_error_samples": maximum_validation_samples,
        "actual_submitted_denominator_includes_pe": True,
        "highband_result_based_phase_repair_samples": 0.0,
        "holdout_used_for_fit_or_selection": False,
        "validation_policy": validation_policy,
        "clock_fit_rows_by_path": {
            path: [row["name"] for row in path_rows[path][0]]
            for path in PATH_CHANNEL
        },
        "clock_validation_rows_by_path": {
            path: [row["name"] for row in path_rows[path][1]]
            for path in PATH_CHANNEL
        },
        "operator_holdout_used_for_clock_validation": bool(
            validation_policy == "holdout_and_tail_legacy"
        ),
        "operator_holdout_first_open_reserved_for_post_fit_scoring": bool(
            validation_policy == "pilot_tail_only_pre_operator_holdout"
        ),
        "passed": passed,
    }
    receipt["accessed_waveform_bundle_sha256"] = _payload_sha256(
        receipt["accessed_waveform_receipts_by_path"]
    )
    return receipt


def synthesize_affine_capture_v5(
    submitted_pcm: np.ndarray,
    *,
    primary_fir_by_mic: np.ndarray,
    secondary_fir_by_mic: np.ndarray,
    rate_ratio: float,
    piecewise_ratio_after_half: float | None = None,
) -> np.ndarray:
    """장치를 열지 않고 fixed two-path FIR와 affine ADC clock raw를 합성한다."""

    submitted = np.asarray(submitted_pcm, dtype=np.float64)
    p_fir = np.asarray(primary_fir_by_mic, dtype=np.float64)
    s_fir = np.asarray(secondary_fir_by_mic, dtype=np.float64)
    if p_fir.shape != s_fir.shape or p_fir.ndim != 2 or p_fir.shape[0] != 2:
        raise ValueError("fixture P/S FIR shape이 잘못됐습니다")
    dac = np.zeros((len(submitted), 2), dtype=np.float64)
    for mic in range(2):
        dac[:, mic] = np.convolve(submitted[:, 0], p_fir[mic], mode="full")[: len(dac)]
        dac[:, mic] += np.convolve(submitted[:, 1], s_fir[mic], mode="full")[: len(dac)]
    count = int(math.ceil(len(dac) / min(rate_ratio, piecewise_ratio_after_half or rate_ratio))) + 8
    adc = np.arange(count, dtype=np.float64)
    dac_q = adc * float(rate_ratio)
    if piecewise_ratio_after_half is not None:
        boundary = count // 2
        anchor = dac_q[boundary]
        dac_q[boundary:] = anchor + (adc[boundary:] - adc[boundary]) * float(piecewise_ratio_after_half)
    valid = dac_q <= len(dac) - 1
    adc = adc[valid]
    dac_q = dac_q[valid]
    return np.column_stack(
        [CubicSpline(np.arange(len(dac)), dac[:, mic])(dac_q) for mic in range(2)]
    )


def score_candidate_on_role_v5(
    *,
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    captured_pcm: np.ndarray,
    primary_fir_by_mic: np.ndarray,
    secondary_fir_by_mic: np.ndarray,
    role: str,
) -> dict[str, Any]:
    """P/S×ERR/REF×8대역의 raw-derived causal FIR 점수를 계산한다.

    ``captured_pcm``은 공통 q로 DAC grid에 이미 재표본화된 전체 capture다. noise floor는
    pilot-only lead/tail에서 두 입력 DFT가 모두 exact zero인 bin만 사용한다.
    """

    if role not in ROLES:
        raise ValueError("role은 fit_a/fit_b/holdout 중 하나여야 합니다")
    submitted = np.asarray(submitted_pcm)
    captured = np.asarray(captured_pcm, dtype=np.float64)
    if submitted.shape != captured.shape or submitted.shape[1] != 2:
        raise ValueError("submitted/captured shape은 동일한 [frame,2]여야 합니다")
    if _array_sha256(submitted) != plan.get("actual_submitted_pcm_sha256"):
        raise ValueError("plan과 submitted PCM SHA가 다릅니다")
    p_fir = np.asarray(primary_fir_by_mic, dtype=np.float64)
    s_fir = np.asarray(secondary_fir_by_mic, dtype=np.float64)
    if p_fir.ndim != 2 or s_fir.shape != p_fir.shape or p_fir.shape[0] != 2:
        raise ValueError("P/S FIR은 [ERR_REF=2, support]이고 shape이 같아야 합니다")
    if p_fir.shape[1] != CONDITION_AUDIT_SUPPORT:
        raise ValueError("v5 authority 후보는 exact-audited support 1024만 허용합니다")

    frequency = np.fft.rfftfreq(PERIOD, 1.0 / FS)
    p_transfer = np.fft.rfft(p_fir, n=PERIOD, axis=1)
    s_transfer = np.fft.rfft(s_fir, n=PERIOD, axis=1)
    lead = captured[PERIOD : 2 * PERIOD]
    tail_row = next(row for row in plan["layout"] if row["kind"] == "pilot_only_tail")
    tail_start = int(tail_row["start_frame"]) + PERIOD
    tail = captured[tail_start : tail_start + PERIOD]
    pilot_input = submitted[PERIOD : 2 * PERIOD].astype(np.float64)
    pilot_spectrum = np.fft.rfft(pilot_input, axis=0)
    exact_zero = np.all(np.abs(pilot_spectrum) <= 1.0e-8, axis=1)
    noise_spectra = (
        np.fft.rfft(lead, axis=0),
        np.fft.rfft(tail, axis=0),
    )
    bands = plan["control_band_contract"]["physical_identification_subbands_hz"]
    rows: list[dict[str, Any]] = []
    for path in ("primary", "secondary"):
        period = _central_role_period(submitted, list(plan["layout"]), path, role)
        source = np.fft.rfft(period, axis=0)
        observed_row = next(
            row
            for row in plan["layout"]
            if row.get("path") == path and row.get("role") == role
        )
        start = int(observed_row["central_start_frame"])
        observed = np.fft.rfft(captured[start : start + PERIOD], axis=0)
        for mic in range(2):
            predicted = source[:, 0] * p_transfer[mic] + source[:, 1] * s_transfer[mic]
            for band_index, (lower, upper) in enumerate(bands):
                mask = (frequency >= float(lower)) & (frequency <= float(upper))
                target = observed[mask, mic]
                estimate = predicted[mask]
                noise_mask = mask & exact_zero
                if int(np.sum(mask)) < 8 or int(np.sum(noise_mask)) < 8:
                    raise ValueError("8대역 score에 필요한 response/noise bin이 부족합니다")
                target_power = float(np.mean(np.abs(target) ** 2))
                residual_power = float(np.mean(np.abs(target - estimate) ** 2))
                noise_power = max(
                    float(
                        np.mean(
                            [
                                np.mean(np.abs(noise[noise_mask, mic]) ** 2)
                                for noise in noise_spectra
                            ]
                        )
                    ),
                    np.finfo(np.float64).tiny,
                )
                denominator = math.sqrt(
                    max(target_power, np.finfo(np.float64).tiny)
                    * max(float(np.mean(np.abs(estimate) ** 2)), np.finfo(np.float64).tiny)
                )
                agreement = float(abs(np.vdot(target, estimate)) / (len(target) * denominator))
                relative = math.sqrt(
                    residual_power / max(target_power, np.finfo(np.float64).tiny)
                )
                snr_db = 10.0 * math.log10(
                    max(target_power, np.finfo(np.float64).tiny) / noise_power
                )
                passed = bool(
                    relative <= SUBBAND_MAX_RELATIVE_RESIDUAL
                    and agreement >= SUBBAND_MIN_COMPLEX_AGREEMENT
                    and snr_db >= SUBBAND_MIN_RESPONSE_TO_NOISE_DB
                )
                rows.append(
                    {
                        "path": path,
                        "microphone": "ERR" if mic == 0 else "REF",
                        "band_index": band_index,
                        "band_hz": [float(lower), float(upper)],
                        "response_bins": int(np.sum(mask)),
                        "exact_zero_noise_bins": int(np.sum(noise_mask)),
                        "noise_conditioned_relative_residual": relative,
                        "complex_agreement": agreement,
                        "response_to_noise_db": snr_db,
                        "passed": passed,
                    }
                )
    receipt: dict[str, Any] = {
        "schema": "fullband_causal_band_score_receipt_v5",
        "role": role,
        "control_band_contract_sha256": plan["control_band_contract_sha256"],
        "actual_submitted_pcm_sha256": plan["actual_submitted_pcm_sha256"],
        "support_samples": CONDITION_AUDIT_SUPPORT,
        "rows": rows,
        "expected_row_count": 2 * 2 * len(bands),
        "all_paths_microphones_subbands_passed": bool(
            len(rows) == 2 * 2 * len(bands) and all(row["passed"] for row in rows)
        ),
        "holdout_used_for_fit_or_selection": False,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt
