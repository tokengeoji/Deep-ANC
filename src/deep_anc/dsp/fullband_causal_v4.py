"""연속 pilot을 쓰는 full-band causal P/S 식별 signal-only 실험.

이 모듈은 두 질문을 분리한다. fixed-LTI acoustic path의 상수 진폭/위상을 nuisance로
취급해도 DAC/ADC rate를 식별할 수 있는가, acoustic evidence만으로 ADC-clock 이동과 공통
time-varying plant delay를 구분할 수 있는가이다.

첫 질문은 affine synthetic fixture에서 PASS했다. 둘째는 선언한 fixed-LTI 모델 클래스 밖에서
불가능하다. 두 가설이 byte-identical raw를 만들 수 있기 때문이다. 따라서 조건부 canonical
경로는 raw-derived stationarity/change-point, 독립 PE, terminal holdout gate를 모두 요구한다.
``LIVE_AUTHORITY``가 ``None``인 이유는 현재 physical raw/publisher가 없기 때문이며 electrical
loopback이 무조건 필수이기 때문은 아니다.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.linalg import eigvalsh
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.signal import butter, fftconvolve, sosfiltfilt
from scipy.sparse.linalg import LinearOperator, eigsh, lsmr

from .control_band_contract import (
    BROADBAND_POINT_CONTROL_SUBBANDS_HZ,
    ControlBandContract,
    max_timing_error_samples_for_attenuation,
)
from .interleaved_probe import schroeder_phases


FS = 48_000
BLOCK = 256
PERIOD = 32_768
CYCLIC_PREFIX = 16_384
CYCLIC_SUFFIX = 16_384
SLOT_FRAMES = CYCLIC_PREFIX + PERIOD + CYCLIC_SUFFIX
LEAD_FRAMES = 2 * PERIOD
# One period clears the final high-band response, one is analysed, and one
# remains after it so zero-phase pilot isolation never uses a capture edge.
TAIL_FRAMES = 3 * PERIOD

TARGET_BAND = (100.0, 11_400.0)
PILOT_BAND = (152.0, 600.0)
PILOT_PEAK_PCM = 20
PE_MAX_PEAK_PCM = 78
PE_COMB_SHIFT = PERIOD // 4

MAX_DELAY = 4_800
SUPPORTS = (1_024, 2_048, 4_096, 8_192)
MAX_HISTORY = MAX_DELAY + max(SUPPORTS)
MAX_CONDITION = 20.0
CONDITION_AUDIT_SUPPORT = min(SUPPORTS)

CLOCK_MAX_ABS_PPM = 1_000.0
CLOCK_VIEW_DISAGREEMENT_MAX = 0.050
CLOCK_LEAVEOUT_MAX = 0.050
CLOCK_CUBIC_MAX = 0.006
CLOCK_COMBINED_MAX = 0.056
CLOCK_HARD_MAX = 0.06755189029558946
CLOCK_MIN_COHERENCE = 0.995

FIT_RESIDUAL_MAX = 0.03
CROSS_FIT_RESIDUAL_MAX = 0.05
FIT_TAP_DISAGREEMENT_MAX = 0.10
HOLDOUT_RESIDUAL_MAX = 0.05
SUBBAND_MAX_RELATIVE_ERROR = 0.10
SUBBAND_MIN_COMPLEX_AGREEMENT = 0.995
SUBBAND_MIN_TARGET_TO_NOISE_DB = 20.0
SUBBAND_MIN_INPUT_RMS_DBFS = -90.0
SUBBAND_MIN_TARGET_RMS_DBFS = -90.0
SUBBAND_MIN_EXACT_ZERO_NOISE_BINS = 8
SUBBAND_MIN_RESPONSE_BINS = 8
SUBBAND_PHASE_BIN_MIN_SNR_DB = 12.0
EXACT_ZERO_DFT_MAX = 1.0e-8

ROLES = (
    ("fit_a", 710_001, 710_003),
    ("fit_b", 810_013, 810_017),
    ("holdout", 910_019, 910_021),
)
MARKERS = (("primary", 610_001), ("secondary", 610_003))
PATH_CHANNEL = {"primary": 0, "secondary": 1}

LIVE_AUTHORITY: None = None
CANONICAL_BLOCKER = (
    "live_raw_and_joint_clock_fir_stationarity_holdout_evidence_absent"
)
TRAINING_AUTHORITY_SCHEMA = "fullband_causal_joint_fir_training_plant_v4"
TRAINING_AUTHORITY_ENVELOPE_SCHEMA = (
    "fullband_causal_training_authority_envelope_v4"
)
OPERATOR_NPZ_SCHEMA = "fullband_causal_joint_fir_operator_npz_v4"
OPERATOR_REFERENCE_SCHEMA = "fullband_causal_joint_fir_operator_reference_v4"
IMMUTABLE_JSON_ARTIFACT_REFERENCE_SCHEMA = "immutable_json_artifact_reference_v4"
IDENTIFIABILITY_LIMITATION = (
    "a_common_time_varying_plant_delay_can_be_observationally_equivalent_to_"
    "adc_clock_motion_outside_the_declared_fixed_lti_model_class"
)


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pilot_bin_sets() -> dict[str, np.ndarray]:
    frequency = np.fft.rfftfreq(PERIOD, 1.0 / FS)
    primary = np.asarray(
        [
            index
            for index, value in enumerate(frequency)
            if PILOT_BAND[0] <= value <= PILOT_BAND[1] and index % 8 == 0
        ],
        dtype=np.int64,
    )
    secondary = np.asarray(
        [
            index
            for index, value in enumerate(frequency)
            if PILOT_BAND[0] <= value <= PILOT_BAND[1] and index % 8 == 4
        ],
        dtype=np.int64,
    )
    if primary.size < 8 or secondary.size < 8:
        raise AssertionError("continuous pilot has too few independent line bins")
    return {"primary": primary, "secondary": secondary}


@lru_cache(maxsize=1)
def _continuous_pilot_period_cached() -> np.ndarray:
    """Return two exact-int16, frequency-disjoint continuous pilots.

    Primary is an integer 4096-sample word repeated eight times.  Secondary is
    an integer 4096-sample word followed by its exact negative and repeated
    four times.  Therefore their 32768-point DFT supports are respectively
    ``k % 8 == 0`` and ``k % 8 == 4`` even after quantisation.  The opposite
    channel null is algebraic, not an intended-float approximation.
    """

    bins = _pilot_bin_sets()

    primary_base_bins = bins["primary"] // 8
    primary_spectrum = np.zeros(4_096 // 2 + 1, dtype=np.complex128)
    primary_spectrum[primary_base_bins] = np.exp(
        1j * schroeder_phases(primary_base_bins.size)
    )
    primary_float = np.fft.irfft(primary_spectrum, n=4_096)
    primary_word = np.rint(
        primary_float / np.max(np.abs(primary_float)) * PILOT_PEAK_PCM
    ).astype(np.int16)
    primary = np.tile(primary_word, 8)

    # k_full = 4 * (2j+1).  A half-integer number of cycles in the first
    # 4096 samples, followed by an exact sign flip, preserves this support
    # after integer rounding.
    odd_indices = bins["secondary"] // 4
    sample = np.arange(4_096, dtype=np.float64)
    phase = schroeder_phases(odd_indices.size)
    secondary_float = np.sum(
        np.cos(
            2.0
            * np.pi
            * (odd_indices[:, None] / 2.0)
            * sample[None, :]
            / 4_096.0
            + phase[:, None]
        ),
        axis=0,
    )
    secondary_word = np.rint(
        secondary_float / np.max(np.abs(secondary_float)) * PILOT_PEAK_PCM
    ).astype(np.int16)
    secondary = np.tile(np.concatenate((secondary_word, -secondary_word)), 4)

    pilot = np.column_stack((primary, secondary)).astype(np.int16, copy=False)
    spectra = np.fft.rfft(pilot.astype(np.float64), axis=0)
    if float(np.max(np.abs(spectra[bins["primary"], 1]))) > 1e-8:
        raise AssertionError("secondary actual-int16 pilot leaks into primary lines")
    if float(np.max(np.abs(spectra[bins["secondary"], 0]))) > 1e-8:
        raise AssertionError("primary actual-int16 pilot leaks into secondary lines")
    if float(np.min(np.abs(spectra[bins["primary"], 0]))) <= 1e3:
        raise AssertionError("primary actual-int16 pilot denominator is too small")
    if float(np.min(np.abs(spectra[bins["secondary"], 1]))) <= 1e3:
        raise AssertionError("secondary actual-int16 pilot denominator is too small")
    pilot.setflags(write=False)
    return pilot


def continuous_pilot_period() -> np.ndarray:
    return _continuous_pilot_period_cached().copy()


@lru_cache(maxsize=None)
def _aperiodic_payload_cached(
    seed: int,
    excitation_lower_hz: float = TARGET_BAND[0],
    excitation_upper_hz: float = TARGET_BAND[1],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create one actual-int16 PE record with algebraic pilot-line nulls."""

    frequency = np.fft.rfftfreq(PERIOD, 1.0 / FS)
    lower = float(excitation_lower_hz)
    upper = float(excitation_upper_hz)
    if not (0.0 < lower < upper < FS / 2.0):
        raise ValueError("aperiodic excitation band가 잘못됐습니다")
    active = (frequency >= lower) & (frequency <= upper)
    active[0] = False
    active[-1] = False
    active_indices = np.flatnonzero(active)
    base_phase = schroeder_phases(active_indices.size)
    random_phase = np.random.default_rng(seed).uniform(
        0.0, 2.0 * np.pi, active_indices.size
    )
    best: tuple[float, np.ndarray] | None = None
    for blend in np.linspace(0.0, 0.30, 31):
        spectrum = np.zeros(frequency.size, dtype=np.complex128)
        spectrum[active_indices] = np.exp(
            1j * ((1.0 - blend) * base_phase + blend * random_phase)
        )
        base = np.fft.irfft(spectrum, n=PERIOD)
        base /= np.max(np.abs(base))
        for base_peak in range(1, 101):
            integer_base = np.rint(base * base_peak).astype(np.int16)
            # Integer comb: X[k] * (1-exp(-j2*pi*k/4)).  Every k % 4 == 0
            # is exactly zero, including both disjoint pilot line sets.
            payload32 = integer_base.astype(np.int32) - np.roll(
                integer_base.astype(np.int32), PE_COMB_SHIFT
            )
            peak = int(np.max(np.abs(payload32)))
            if peak > PE_MAX_PEAK_PCM:
                break
            payload = payload32.astype(np.int16)
            rms = float(np.sqrt(np.mean(payload.astype(np.float64) ** 2)))
            if best is None or rms > best[0]:
                best = (rms, payload)
    if best is None:
        raise AssertionError("could not construct bounded aperiodic PE payload")
    payload = np.asarray(best[1], dtype=np.int16)
    bins = _pilot_bin_sets()
    pilot_lines = np.concatenate((bins["primary"], bins["secondary"]))
    payload_spectrum = np.fft.rfft(payload.astype(np.float64))
    pilot_null = float(np.max(np.abs(payload_spectrum[pilot_lines])))
    if pilot_null > 1e-8:
        raise AssertionError(f"actual-int16 PE pilot-line null failed: {pilot_null}")
    decoded = payload.astype(np.float64) / 32767.0
    nonzero_target = np.abs(payload_spectrum[active]) > 1e-8
    meta = {
        "seed": int(seed),
        "frames": PERIOD,
        "target_band_hz": [lower, upper],
        "peak_pcm": int(np.max(np.abs(payload.astype(np.int32)))),
        "rms": float(np.sqrt(np.mean(decoded**2))),
        "actual_int16_pilot_line_max_abs_dft": pilot_null,
        "actual_target_bin_nonzero_fraction": float(np.mean(nonzero_target)),
        "comb_shift_samples": PE_COMB_SHIFT,
        "comb_zero_bin_rule": "k_mod_4_equals_0",
        "pcm_sha256": _sha256_array(payload),
    }
    payload.setflags(write=False)
    return payload, meta


def _aperiodic_payload(
    seed: int,
    *,
    excitation_lower_hz: float = TARGET_BAND[0],
    excitation_upper_hz: float = TARGET_BAND[1],
) -> tuple[np.ndarray, dict[str, Any]]:
    payload, metadata = _aperiodic_payload_cached(
        int(seed), float(excitation_lower_hz), float(excitation_upper_hz)
    )
    return payload.copy(), dict(metadata)


def _clock_row(
    *, name: str, start: int, purpose: str, role: str | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "start_frame": int(start),
        "stop_frame": int(start + PERIOD),
        "frames": PERIOD,
        "purpose": purpose,
        "role": role,
        "uses_full_period_after_bilateral_boundary_exclusion": True,
    }


@lru_cache(maxsize=8)
def _build_plan_cached(
    control_band_contract_json: str,
    excitation_lower_hz: float,
    excitation_upper_hz: float,
) -> tuple[dict[str, Any], np.ndarray]:
    contract_payload = json.loads(control_band_contract_json)
    contract_subbands = tuple(
        tuple(float(value) for value in band)
        for band in contract_payload["point_control_subbands_hz"]
    )
    contract_digest = hashlib.sha256(
        json.dumps(
            contract_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    payloads: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    payloads.update(
        {
            f"marker_{path}": _aperiodic_payload(
                seed,
                excitation_lower_hz=excitation_lower_hz,
                excitation_upper_hz=excitation_upper_hz,
            )
            for path, seed in MARKERS
        }
    )
    for role, primary_seed, secondary_seed in ROLES:
        payloads[f"primary_{role}"] = _aperiodic_payload(
            primary_seed,
            excitation_lower_hz=excitation_lower_hz,
            excitation_upper_hz=excitation_upper_hz,
        )
        payloads[f"secondary_{role}"] = _aperiodic_payload(
            secondary_seed,
            excitation_lower_hz=excitation_lower_hz,
            excitation_upper_hz=excitation_upper_hz,
        )

    highband_parts: list[np.ndarray] = []
    layout: list[dict[str, Any]] = []
    clock_rows: list[dict[str, Any]] = []
    cursor = 0

    def add(kind: str, value: np.ndarray, **extra: Any) -> None:
        nonlocal cursor
        highband_parts.append(value)
        layout.append(
            {
                "kind": kind,
                "start_frame": cursor,
                "stop_frame": cursor + len(value),
                "frames": len(value),
                **extra,
            }
        )
        cursor += len(value)

    add(
        "continuous_pilot_lead",
        np.zeros((LEAD_FRAMES, 2), dtype=np.int16),
        highband_component_exact_zero=True,
    )
    clock_rows.append(
        _clock_row(
            name="lead_reference",
            start=PERIOD,
            purpose="fit",
            role="fit_reference",
        )
    )

    ordered_slots: list[tuple[str, str, str | None]] = [
        ("primary", "marker_primary", None),
        ("secondary", "marker_secondary", None),
    ]
    for role, _, _ in ROLES:
        ordered_slots.extend(
            (("primary", f"primary_{role}", role), ("secondary", f"secondary_{role}", role))
        )

    for path, payload_name, role in ordered_slots:
        payload, payload_meta = payloads[payload_name]
        channel = PATH_CHANNEL[path]
        slot = np.zeros((SLOT_FRAMES, 2), dtype=np.int16)
        slot[:CYCLIC_PREFIX, channel] = payload[-CYCLIC_PREFIX:]
        slot[CYCLIC_PREFIX : CYCLIC_PREFIX + PERIOD, channel] = payload
        slot[CYCLIC_PREFIX + PERIOD :, channel] = payload[:CYCLIC_SUFFIX]
        central_start = cursor + CYCLIC_PREFIX
        purpose = "validation" if role in (None, "holdout") else "fit"
        add(
            f"{payload_name}_slot",
            slot,
            path=path,
            role=role,
            output_channel=channel,
            payload_pcm_sha256=payload_meta["pcm_sha256"],
            central_start_frame=central_start,
            central_stop_frame=central_start + PERIOD,
            pre_boundary_exclusion_samples=CYCLIC_PREFIX,
            post_boundary_exclusion_samples=CYCLIC_SUFFIX,
            candidate_data=role in ("fit_a", "fit_b", "holdout"),
        )
        clock_rows.append(
            _clock_row(
                name=payload_name,
                start=central_start,
                purpose=purpose,
                role=role if role is not None else "marker_validation",
            )
        )

    tail_start = cursor
    add(
        "continuous_pilot_tail",
        np.zeros((TAIL_FRAMES, 2), dtype=np.int16),
        highband_component_exact_zero=True,
    )
    clock_rows.append(
        _clock_row(
            name="tail_validation",
            start=tail_start + PERIOD,
            purpose="validation",
            role="tail_validation",
        )
    )

    highband = np.concatenate(highband_parts, axis=0)
    if len(highband) % BLOCK:
        raise AssertionError("v4 layout must be block aligned before pilot overlay")
    pilot_period = _continuous_pilot_period_cached()
    repeats = int(math.ceil(len(highband) / PERIOD))
    pilot = np.tile(pilot_period, (repeats, 1))[: len(highband)]
    submitted32 = highband.astype(np.int32) + pilot.astype(np.int32)
    peak = int(np.max(np.abs(submitted32)))
    if peak > 98:
        raise AssertionError(f"v4 submitted peak {peak} exceeds PCM 98")
    submitted = submitted32.astype(np.int16)

    bins = _pilot_bin_sets()
    pilot_spectrum = np.fft.rfft(pilot_period.astype(np.float64), axis=0)
    pilot_metadata = {
        "period_samples": PERIOD,
        "band_hz": list(PILOT_BAND),
        "primary_bins": bins["primary"].tolist(),
        "secondary_bins": bins["secondary"].tolist(),
        "primary_frequencies_hz": np.fft.rfftfreq(PERIOD, 1.0 / FS)[
            bins["primary"]
        ].tolist(),
        "secondary_frequencies_hz": np.fft.rfftfreq(PERIOD, 1.0 / FS)[
            bins["secondary"]
        ].tolist(),
        "primary_opposite_actual_int16_max_abs_dft": float(
            np.max(np.abs(pilot_spectrum[bins["primary"], 1]))
        ),
        "secondary_opposite_actual_int16_max_abs_dft": float(
            np.max(np.abs(pilot_spectrum[bins["secondary"], 0]))
        ),
        "actual_int16_period_pcm_sha256": _sha256_array(pilot_period),
        "overlay_present_for_every_submitted_frame": True,
        "highband_clock_fit_forbidden": True,
        "unknown_fixed_lti_gain_and_phase_profiled_out": True,
    }

    plan: dict[str, Any] = {
        "schema": "fullband_causal_continuous_reserved_pilot_v4",
        "role": "signal_only_dry_run_no_audio",
        "sample_rate": FS,
        "block_size": BLOCK,
        "control_band_contract": contract_payload,
        "control_band_contract_sha256": contract_digest,
        "live_capture_enabled": False,
        "live_authority": None,
        "canonical_training_eligible": False,
        "canonical_blocker": CANONICAL_BLOCKER,
        "layout": layout,
        "clock_rows": clock_rows,
        "continuous_pilot": pilot_metadata,
        "aperiodic_payloads": {
            name: metadata for name, (_, metadata) in payloads.items()
        },
        "boundary_contract": {
            "pre_exclusion_samples": CYCLIC_PREFIX,
            "post_exclusion_samples": CYCLIC_SUFFIX,
            "maximum_delay_plus_support_samples": MAX_HISTORY,
            "pre_and_post_both_exceed_maximum_history": bool(
                CYCLIC_PREFIX > MAX_HISTORY and CYCLIC_SUFFIX > MAX_HISTORY
            ),
            "path_switch_and_period_boundary_both_sides_excluded": True,
        },
        "clock_contract": {
            "fit_rows": [
                row["name"] for row in clock_rows if row["purpose"] == "fit"
            ],
            "validation_rows": [
                row["name"]
                for row in clock_rows
                if row["purpose"] == "validation"
            ],
            "holdout_used_for_fit_or_selection": False,
            "err_ref_primary_secondary_common_map_required": True,
            "actual_submitted_int16_denominator_required": True,
            "callback_role": "monotonic_and_sample_count_slip_witness_only",
            "maximum_abs_ppm": CLOCK_MAX_ABS_PPM,
            "view_disagreement_max_samples": CLOCK_VIEW_DISAGREEMENT_MAX,
            "leaveout_max_samples": CLOCK_LEAVEOUT_MAX,
            "cubic_max_samples": CLOCK_CUBIC_MAX,
            "combined_max_samples": CLOCK_COMBINED_MAX,
            "hard_max_samples": CLOCK_HARD_MAX,
        },
        "plant_identification": {
            "conditional_training_authority_schema": TRAINING_AUTHORITY_SCHEMA,
            "control_band_contract_sha256": contract_digest,
            "operator": "joint_two_input_finite_causal_fir",
            "actual_full_input_including_continuous_pilot": True,
            "pilot_low_band_is_part_of_joint_plant_fit": True,
            "opposite_path_continuous_pilot_is_jointly_fit_not_subtracted": True,
            "candidate_support_samples": list(SUPPORTS),
            "maximum_condition_number": MAX_CONDITION,
            "condition_audit_support_samples": CONDITION_AUDIT_SUPPORT,
            "condition_audit_uses_exact_actual_int16_cp_gram": True,
            "longer_support_condition_cannot_improve_by_interlacing": True,
            "fit_roles": ["fit_a", "fit_b"],
            "support_selection_uses_holdout": False,
            "holdout_is_terminal_validation_only": True,
            "finite_memory_proved_by_finite_capture": False,
            "finite_memory_scope": (
                "only the predeclared candidate supports are falsified or retained; "
                "an infinite or later echo is not proved absent"
            ),
            "subband_authority": {
                "bands_hz": [
                    [float(lo), float(hi)]
                    for lo, hi in contract_subbands
                ],
                "each_primary_secondary_microphone_band_must_pass": True,
                "global_residual_is_diagnostic_not_sufficient": True,
                "maximum_noise_conditioned_relative_residual": (
                    SUBBAND_MAX_RELATIVE_ERROR
                ),
                "minimum_complex_agreement": SUBBAND_MIN_COMPLEX_AGREEMENT,
                "minimum_target_to_exact_zero_noise_db": (
                    SUBBAND_MIN_TARGET_TO_NOISE_DB
                ),
                "minimum_input_rms_dbfs": SUBBAND_MIN_INPUT_RMS_DBFS,
                "minimum_target_rms_dbfs": SUBBAND_MIN_TARGET_RMS_DBFS,
                "minimum_exact_zero_noise_bins_per_reference_row": (
                    SUBBAND_MIN_EXACT_ZERO_NOISE_BINS
                ),
                "minimum_response_bins": SUBBAND_MIN_RESPONSE_BINS,
                "exact_zero_dft_max_absolute": EXACT_ZERO_DFT_MAX,
                "noise_reference_rows": ["lead_reference", "tail_validation"],
                "noise_power_estimator": (
                    "p95_abs_rfft_squared_across_actual_both_input_exact_zero_bins"
                ),
                "timing_resolution_attenuation_db": 20.0,
            },
            "excitation_band_hz": [
                float(excitation_lower_hz),
                float(excitation_upper_hz),
            ],
            "125hz_octave_lower_hz": 125.0 / math.sqrt(2.0),
            "125hz_octave_fully_covered": bool(
                float(contract_payload["point_control_target_hz"][0])
                <= 125.0 / math.sqrt(2.0)
                and float(excitation_lower_hz) <= 125.0 / math.sqrt(2.0)
            ),
            "125hz_octave_contract_blocker": (
                None
                if float(contract_payload["point_control_target_hz"][0])
                <= 125.0 / math.sqrt(2.0)
                and float(excitation_lower_hz) <= 125.0 / math.sqrt(2.0)
                else "control_contract_or_excitation_does_not_cover_full_125hz_octave"
            ),
        },
        "identifiability": {
            "fixed_lti_clock_rate_identifiable_from_line_phase_slope": True,
            "fixed_lti_constant_phase_and_amplitude_are_nuisance": True,
            "common_time_varying_plant_delay_is_clock_equivalent": True,
            "limitation": IDENTIFIABILITY_LIMITATION,
            "conditional_canonical_path_exists_under_fixed_lti": True,
            "required_raw_rejections": [
                "clock_change_point",
                "fit_a_fit_b_fir_change_point",
                "holdout_phase_or_residual_failure",
                "err_ref_primary_secondary_map_disagreement",
            ],
            "independent_electrical_continuous_witness_required_for_canonical": False,
            "independent_electrical_continuous_witness_recommended": True,
        },
        "output": {
            "frames": int(len(submitted)),
            "duration_seconds": float(len(submitted) / FS),
            "active_highband_slot_seconds": float(
                len(ordered_slots) * SLOT_FRAMES / FS
            ),
            "peak_pcm": peak,
            "pcm_sha256": _sha256_array(submitted),
        },
    }
    plan["canonical_payload_sha256"] = _json_sha256(plan)
    submitted.setflags(write=False)
    return plan, submitted


def build_plan(
    *,
    control_band_contract: ControlBandContract | None = None,
    excitation_lower_hz: float | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """contract가 선언한 전 대역으로 plan을 만들되 live authority는 열지 않는다."""

    contract = (
        ControlBandContract.broadband_point_control()
        if control_band_contract is None
        else control_band_contract
    )
    if contract.role != "broadband_point_control":
        raise ValueError("v4에는 broadband point-control contract가 필요합니다")
    lower = TARGET_BAND[0] if excitation_lower_hz is None else float(
        excitation_lower_hz
    )
    upper = max(TARGET_BAND[1], float(contract.required_excitation_upper_hz))
    if lower > float(contract.point_control_target_hz[0]):
        raise ValueError("excitation lower가 control target lower보다 높습니다")
    contract_json = json.dumps(
        contract.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    plan, submitted = _build_plan_cached(contract_json, lower, upper)
    # JSON round-trip gives callers an independent nested metadata structure.
    return json.loads(json.dumps(plan)), submitted.copy()


def _validate_callbacks(
    callback_time_info: Mapping[str, np.ndarray], expected_frames: int
) -> dict[str, Any]:
    required = ("frame_index", "frame_count", "input_adc_time", "output_dac_time")
    if any(key not in callback_time_info for key in required):
        raise ValueError("callback time_info raw arrays missing")
    frame = np.asarray(callback_time_info["frame_index"], dtype=np.int64)
    count = np.asarray(callback_time_info["frame_count"], dtype=np.int64)
    adc = np.asarray(callback_time_info["input_adc_time"], dtype=np.float64)
    dac = np.asarray(callback_time_info["output_dac_time"], dtype=np.float64)
    if not (len(frame) == len(count) == len(adc) == len(dac) and len(frame) >= 2):
        raise ValueError("callback witness shape mismatch")
    if (
        frame[0] != 0
        or np.any(count <= 0)
        or np.any(np.diff(frame) != count[:-1])
        or np.any(np.diff(adc) <= 0.0)
        or np.any(np.diff(dac) <= 0.0)
        or int(frame[-1] + count[-1]) < int(expected_frames)
    ):
        raise ValueError("callback slip/non-monotonic/coverage")
    payload = {
        "frame_index_sha256": _sha256_array(frame),
        "frame_count_sha256": _sha256_array(count),
        "input_adc_time_sha256": _sha256_array(adc),
        "output_dac_time_sha256": _sha256_array(dac),
        "role": "monotonic_and_sample_count_slip_witness_only",
    }
    payload["sha256"] = _json_sha256(payload)
    return payload


def _interpolate(
    signal: np.ndarray,
    coordinates: np.ndarray,
    *,
    method: str,
    spline: CubicSpline | None = None,
) -> np.ndarray:
    q = np.asarray(coordinates, dtype=np.float64)
    if q.size == 0 or q[0] < 0.0 or q[-1] > len(signal) - 1:
        raise ValueError("clock interpolation support missing")
    if method == "linear":
        left = np.floor(q).astype(np.int64)
        right = np.minimum(left + 1, len(signal) - 1)
        fraction = q - left
        return signal[left] * (1.0 - fraction) + signal[right] * fraction
    if method == "cubic":
        if spline is None:
            spline = CubicSpline(
                np.arange(len(signal), dtype=np.float64), signal, extrapolate=False
            )
        values = np.asarray(spline(q), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("cubic clock interpolation produced non-finite values")
        return values
    raise ValueError(f"unknown interpolation method: {method}")


def _clock_rows_for(plan: Mapping[str, Any], purpose: str) -> list[Mapping[str, Any]]:
    rows = [row for row in plan["clock_rows"] if row["purpose"] == purpose]
    if not rows:
        raise ValueError(f"clock {purpose} rows are empty")
    return rows


def _row_transfer(
    *,
    signal: np.ndarray,
    submitted: np.ndarray,
    row: Mapping[str, Any],
    path: str,
    rate_ratio: float,
    method: str,
    spline: CubicSpline | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    start = int(row["start_frame"])
    stop = int(row["stop_frame"])
    if stop - start != PERIOD:
        raise ValueError("clock row is not one exact pilot period")
    dac_q = np.arange(start, stop, dtype=np.float64)
    response = _interpolate(
        signal, dac_q / float(rate_ratio), method=method, spline=spline
    )
    bins = _pilot_bin_sets()[path]
    channel = PATH_CHANNEL[path]
    active = np.fft.rfft(submitted[start:stop, channel].astype(np.float64))[bins]
    opposite = np.fft.rfft(
        submitted[start:stop, 1 - channel].astype(np.float64)
    )[bins]
    if float(np.min(np.abs(active))) <= 1e3:
        raise ValueError("actual submitted int16 pilot denominator missing")
    if float(np.max(np.abs(opposite))) > 1e-8:
        raise ValueError("actual submitted opposite-channel pilot null failed")
    frequency = np.fft.rfftfreq(PERIOD, 1.0 / FS)[bins]
    transfer = np.fft.rfft(response)[bins] / active
    return transfer, frequency, _sha256_array(active)


def _transfer_bank(
    *,
    plan: Mapping[str, Any],
    submitted: np.ndarray,
    signals: Mapping[str, np.ndarray],
    rate_ratio: float,
    method: str,
    purposes: Iterable[str],
    paths: Iterable[str] | None = None,
) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, str]]:
    allowed = set(purposes)
    selected_paths = tuple(PATH_CHANNEL if paths is None else paths)
    if not selected_paths or any(path not in PATH_CHANNEL for path in selected_paths):
        raise ValueError("continuous pilot path selection is invalid")
    rows = [row for row in plan["clock_rows"] if row["purpose"] in allowed]
    bank: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, str]] = {}
    splines: dict[str, CubicSpline] = {}
    if method == "cubic":
        splines = {
            name: CubicSpline(
                np.arange(len(signal), dtype=np.float64),
                signal,
                extrapolate=False,
            )
            for name, signal in signals.items()
        }
    for microphone, signal in signals.items():
        for path in selected_paths:
            for row in rows:
                bank[(microphone, path, str(row["name"]))] = _row_transfer(
                    signal=signal,
                    submitted=submitted,
                    row=row,
                    path=path,
                    rate_ratio=rate_ratio,
                    method=method,
                    spline=splines.get(microphone),
                )
    return bank


def _variance_objective(
    bank: Mapping[tuple[str, str, str], tuple[np.ndarray, np.ndarray, str]],
    *,
    fit_names: Sequence[str],
    views: Sequence[tuple[str, str]],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for microphone, path in views:
        values = np.stack(
            [bank[(microphone, path, name)][0] for name in fit_names], axis=0
        )
        mean = np.mean(values, axis=0)
        numerator += float(np.sum(np.abs(values - mean[None, :]) ** 2))
        denominator += float(np.sum(np.abs(values) ** 2))
    if denominator <= 1e-30:
        return math.inf
    return numerator / denominator


def _estimate_rate_ratio(
    *,
    plan: Mapping[str, Any],
    submitted: np.ndarray,
    signals: Mapping[str, np.ndarray],
    method: str,
    views: Sequence[tuple[str, str]],
) -> tuple[float, float]:
    fit_rows = _clock_rows_for(plan, "fit")
    fit_names = [str(row["name"]) for row in fit_rows]
    selected_paths = tuple(sorted({path for _, path in views}))
    lo = 1.0 - CLOCK_MAX_ABS_PPM * 1e-6
    hi = 1.0 + CLOCK_MAX_ABS_PPM * 1e-6

    def objective(value: float) -> float:
        try:
            bank = _transfer_bank(
                plan=plan,
                submitted=submitted,
                signals=signals,
                rate_ratio=float(value),
                method=method,
                purposes=("fit",),
                paths=selected_paths,
            )
        except ValueError:
            return math.inf
        return _variance_objective(bank, fit_names=fit_names, views=views)

    grid = np.linspace(lo, hi, 17)
    scores = np.asarray([objective(float(value)) for value in grid])
    if not np.all(np.isfinite(scores)):
        raise ValueError("continuous pilot clock objective is non-finite")
    best = int(np.argmin(scores))
    if best in (0, len(grid) - 1):
        raise ValueError("continuous pilot clock rate hit the search boundary")
    result = minimize_scalar(
        objective,
        bounds=(float(grid[best - 1]), float(grid[best + 1])),
        method="bounded",
        options={"xatol": 1e-13, "maxiter": 100},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise ValueError("continuous pilot clock optimisation failed")
    return float(result.x), float(result.fun)


def _fractional_delay(
    ratio: np.ndarray, frequency: np.ndarray, *, width: float = 1.0
) -> tuple[float, float]:
    valid = np.isfinite(ratio) & (np.abs(ratio) > 0.0)
    phase_ratio = ratio[valid] / np.abs(ratio[valid])
    selected_frequency = frequency[valid]
    if phase_ratio.size < 8:
        raise ValueError("pilot line SNR/bin count insufficient")

    def objective(delay: float) -> float:
        return -float(
            np.abs(
                np.sum(
                    phase_ratio
                    * np.exp(2j * np.pi * selected_frequency * delay / FS)
                )
            )
            / phase_ratio.size
        )

    result = minimize_scalar(
        objective,
        bounds=(-float(width), float(width)),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(result.x), float(-result.fun)


def _validate_clock_rows(
    *,
    plan: Mapping[str, Any],
    submitted: np.ndarray,
    signals: Mapping[str, np.ndarray],
    rate_ratio: float,
    method: str,
    views: Sequence[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    selected_views = (
        tuple((microphone, path) for microphone in signals for path in PATH_CHANNEL)
        if views is None
        else tuple(views)
    )
    if not selected_views:
        raise ValueError("continuous pilot validation views are empty")
    if any(
        microphone not in signals or path not in PATH_CHANNEL
        for microphone, path in selected_views
    ):
        raise ValueError("continuous pilot validation view is invalid")
    selected_paths = tuple(sorted({path for _, path in selected_views}))
    bank = _transfer_bank(
        plan=plan,
        submitted=submitted,
        signals=signals,
        rate_ratio=rate_ratio,
        method=method,
        purposes=("fit", "validation"),
        paths=selected_paths,
    )
    fit_names = [str(row["name"]) for row in _clock_rows_for(plan, "fit")]
    validation_rows = _clock_rows_for(plan, "validation")
    residuals: list[float] = []
    coherences: list[float] = []
    spectra: list[tuple[str, str, str, str]] = []
    for microphone, path in selected_views:
            fit = [bank[(microphone, path, name)][0] for name in fit_names]
            reference = np.mean(np.stack(fit), axis=0)
            frequency = bank[(microphone, path, fit_names[0])][1]
            for row in validation_rows:
                name = str(row["name"])
                candidate, _, digest = bank[(microphone, path, name)]
                floor = max(
                    float(np.max(np.abs(reference))),
                    float(np.max(np.abs(candidate))),
                ) * 1e-8
                valid = (np.abs(reference) > floor) & (np.abs(candidate) > floor)
                if int(np.count_nonzero(valid)) < 8:
                    raise ValueError("pilot line low-SNR validation")
                delay, phase_coherence = _fractional_delay(
                    candidate[valid] / reference[valid], frequency[valid]
                )
                corrected = candidate[valid] * np.exp(
                    2j * np.pi * frequency[valid] * delay / FS
                )
                complex_coherence = float(
                    np.abs(np.vdot(reference[valid], corrected))
                    / (
                        np.linalg.norm(reference[valid])
                        * np.linalg.norm(corrected)
                        + 1e-30
                    )
                )
                coherence = min(float(phase_coherence), complex_coherence)
                if coherence < CLOCK_MIN_COHERENCE:
                    raise ValueError(
                        f"continuous pilot low coherence {coherence:.9f}"
                    )
                residuals.append(abs(float(delay)))
                coherences.append(coherence)
                spectra.append((microphone, path, name, digest))
    return {
        "maximum_leaveout_residual_samples": float(max(residuals)),
        "minimum_transfer_coherence": float(min(coherences)),
        "submitted_pilot_spectra_sha256": _json_sha256(spectra),
        "holdout_used_for_fit_or_selection": False,
    }


def absolute_dac_q_timewarp_v4(
    *,
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    raw_err: np.ndarray,
    raw_ref: np.ndarray,
    callback_time_info: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """fixed-LTI 가정 아래 affine acoustic-pilot DAC-q map을 유도한다.

    현재 반환값은 live raw publisher가 없으므로 diagnostic이다. 추후 독립 PE,
    stationarity/change-point, untouched holdout을 모두 통과한 raw에 한해 conditional
    canonical envelope가 별도로 발행될 수 있다. fixed-LTI 범위 밖의 한계는
    ``acoustic_clock_plant_confounding_counterexample``가 보존한다.
    """

    submitted = np.asarray(submitted_pcm)
    if submitted.dtype != np.int16 or submitted.ndim != 2 or submitted.shape[1] != 2:
        raise ValueError("actual submitted PCM must be exact [frames,2] int16")
    submitted = np.ascontiguousarray(submitted)
    if plan.get("schema") != "fullband_causal_continuous_reserved_pilot_v4":
        raise ValueError("v4 plan schema mismatch")
    if _sha256_array(submitted) != plan["output"]["pcm_sha256"]:
        raise ValueError("v4 plan/PCM lineage mismatch")
    err = np.asarray(raw_err, dtype=np.float64).reshape(-1)
    ref = np.asarray(raw_ref, dtype=np.float64).reshape(-1)
    if min(len(err), len(ref)) < len(submitted):
        raise ValueError("raw shorter than submitted plan")
    callback = _validate_callbacks(callback_time_info, min(len(err), len(ref)))
    sos = butter(12, (120.0, 680.0), btype="bandpass", fs=FS, output="sos")
    signals = {
        "err": sosfiltfilt(sos, err),
        "ref": sosfiltfilt(sos, ref),
    }
    views = [(mic, path) for mic in signals for path in PATH_CHANNEL]
    view_ratios: dict[str, float] = {}
    view_objectives: dict[str, float] = {}
    for microphone, path in views:
        ratio, objective = _estimate_rate_ratio(
            plan=plan,
            submitted=submitted,
            signals={microphone: signals[microphone]},
            method="linear",
            views=((microphone, path),),
        )
        view_ratios[f"{microphone}_{path}"] = ratio
        view_objectives[f"{microphone}_{path}"] = objective
    view_disagreement = (
        max(view_ratios.values()) - min(view_ratios.values())
    ) * len(submitted)
    if view_disagreement > CLOCK_VIEW_DISAGREEMENT_MAX:
        raise ValueError(
            f"ERR/REF/P/S continuous-pilot maps disagree: {view_disagreement}"
        )
    linear_ratio, linear_objective = _estimate_rate_ratio(
        plan=plan,
        submitted=submitted,
        signals=signals,
        method="linear",
        views=views,
    )
    cubic_ratio, cubic_objective = _estimate_rate_ratio(
        plan=plan,
        submitted=submitted,
        signals=signals,
        method="cubic",
        views=views,
    )
    cubic_difference = abs(linear_ratio - cubic_ratio) * len(submitted)
    linear_validation = _validate_clock_rows(
        plan=plan,
        submitted=submitted,
        signals=signals,
        rate_ratio=linear_ratio,
        method="linear",
    )
    cubic_validation = _validate_clock_rows(
        plan=plan,
        submitted=submitted,
        signals=signals,
        rate_ratio=linear_ratio,
        method="cubic",
    )
    leaveout = float(linear_validation["maximum_leaveout_residual_samples"])
    cubic = max(
        float(cubic_difference),
        abs(
            float(cubic_validation["maximum_leaveout_residual_samples"])
            - leaveout
        ),
    )
    combined = leaveout + cubic
    if leaveout > CLOCK_LEAVEOUT_MAX:
        raise ValueError(f"continuous pilot leaveout residual {leaveout}")
    if (
        cubic > CLOCK_CUBIC_MAX
        or combined > CLOCK_COMBINED_MAX
        or combined > CLOCK_HARD_MAX
    ):
        raise ValueError(
            f"20dB continuous-pilot clock budget cubic={cubic} combined={combined}"
        )

    adc_knots = np.asarray(
        [0.0, (len(submitted) - 1) / linear_ratio], dtype=np.float64
    )
    dac_knots = np.asarray([0.0, len(submitted) - 1], dtype=np.float64)
    payload = {
        "schema": "absolute_dac_q_timewarp_continuous_pilot_v4_diagnostic",
        "rate_ratio_dac_q_per_adc_sample": linear_ratio,
        "slope": linear_ratio - 1.0,
        "intercept": 0.0,
        "intercept_semantics": "marker_required_for_live_coarse_branch",
        "adc_knots_sha256": _sha256_array(adc_knots),
        "dac_knots_sha256": _sha256_array(dac_knots),
        "fixed_lti_hypothesis_required": True,
        "highband_used_for_clock_fit": False,
        "canonical_training_eligible": False,
        "canonical_blocker": CANONICAL_BLOCKER,
    }
    payload["map_sha256"] = _json_sha256(payload)
    receipt = {
        **payload,
        "adc_knots": adc_knots,
        "dac_knots": dac_knots,
        "linear_objective": linear_objective,
        "cubic_objective": cubic_objective,
        "view_rate_ratios": view_ratios,
        "view_objectives": view_objectives,
        "view_end_to_end_disagreement_samples": float(view_disagreement),
        "leaveout_max_samples": leaveout,
        "cubic_max_samples": cubic,
        "combined_max_samples": combined,
        "minimum_transfer_coherence": min(
            float(linear_validation["minimum_transfer_coherence"]),
            float(cubic_validation["minimum_transfer_coherence"]),
        ),
        "submitted_pilot_spectra_sha256": linear_validation[
            "submitted_pilot_spectra_sha256"
        ],
        "holdout_used_for_fit_or_selection": False,
        "callback_witness": callback,
        "submitted_pcm_sha256": _sha256_array(submitted),
        "raw_err_sha256": _sha256_array(err),
        "raw_ref_sha256": _sha256_array(ref),
        "passed_under_fixed_lti_hypothesis": True,
        "canonical_passed": False,
    }
    receipt["receipt_sha256"] = _json_sha256(
        {
            key: value
            for key, value in receipt.items()
            if not isinstance(value, np.ndarray)
        }
    )
    return receipt


def marker_branch_v4(
    *, marker: np.ndarray, response_search: np.ndarray
) -> dict[str, Any]:
    """Require one unique 0..MAX_DELAY aperiodic marker branch."""

    source = np.asarray(marker, dtype=np.float64).reshape(-1)
    search = np.asarray(response_search, dtype=np.float64).reshape(-1)
    if search.size < source.size + MAX_DELAY:
        raise ValueError("marker response search is too short")
    correlation = fftconvolve(search, source[::-1], mode="valid")
    allowed = correlation[: MAX_DELAY + 1]
    order = np.argsort(np.abs(allowed))[::-1]
    peak = int(order[0])
    aliases = [
        int(index)
        for index in order[1:]
        if abs(int(index) - peak) >= 32
        and abs(allowed[index]) >= 0.98 * abs(allowed[peak])
    ]
    if aliases:
        raise ValueError("marker coarse-delay branch is not unique")
    return {
        "delay_samples": peak,
        "peak_abs_correlation": float(abs(allowed[peak])),
        "marker_sha256": _sha256_array(source),
        "search_sha256": _sha256_array(search),
    }


def acoustic_clock_plant_confounding_counterexample(
    *, acoustic_signal: np.ndarray, trajectory_samples: np.ndarray
) -> dict[str, Any]:
    """Construct the byte-identical acoustic clock/plant-delay counterexample.

    Hypothesis A has a fixed plant and samples it at ``q=n+delta[n]``.
    Hypothesis B has an ideal ADC clock but a common time-varying plant whose
    output at ``n`` equals the fixed-plant output at ``n+delta[n]``.  No
    acoustic-only statistic can distinguish the two because their raw arrays
    are identical by construction.
    """

    signal = np.asarray(acoustic_signal, dtype=np.float64).reshape(-1)
    trajectory = np.asarray(trajectory_samples, dtype=np.float64).reshape(-1)
    if signal.size != trajectory.size or signal.size < 8:
        raise ValueError("counterexample signal/trajectory shape mismatch")
    coordinate = np.arange(signal.size, dtype=np.float64) + trajectory
    if coordinate[0] < 0.0 or coordinate[-1] > signal.size - 1:
        raise ValueError("counterexample interpolation support missing")
    fixed_plant_async_adc = CubicSpline(
        np.arange(signal.size, dtype=np.float64), signal, extrapolate=False
    )(coordinate)
    ideal_adc_common_time_varying_plant = fixed_plant_async_adc.copy()
    byte_identical = bool(
        np.array_equal(fixed_plant_async_adc, ideal_adc_common_time_varying_plant)
    )
    if not byte_identical:
        raise AssertionError("clock/plant counterexample must be byte-identical")
    return {
        "schema": "acoustic_clock_common_plant_delay_counterexample_v1",
        "hypothesis_a": "fixed_lti_plant_plus_time_varying_adc_q",
        "hypothesis_b": "ideal_adc_plus_common_time_varying_plant_delay",
        "raw_byte_identical": True,
        "raw_sha256": _sha256_array(fixed_plant_async_adc),
        "trajectory_sha256": _sha256_array(trajectory),
        "acoustic_only_decision_possible": False,
        "required_disambiguating_witness": (
            "continuous electrical loopback outside P/S acoustic transfer"
        ),
        "model_scope_limitation": IDENTIFIABILITY_LIMITATION,
        "fixed_lti_stationarity_gates_can_define_conditional_canonical_scope": True,
    }


def _joint_role_indices(plan: Mapping[str, Any], source_role: str) -> np.ndarray:
    selected: list[np.ndarray] = []
    for row in plan["layout"]:
        if row.get("role") != source_role or not bool(row.get("candidate_data")):
            continue
        selected.append(
            np.arange(
                int(row["central_start_frame"]),
                int(row["central_stop_frame"]),
                dtype=np.int64,
            )
        )
    if len(selected) != 2:
        raise ValueError("joint P/S role must contain exactly two central payloads")
    return np.concatenate(selected)


def _joint_fir_operator(
    *,
    submitted: np.ndarray,
    selected_indices: np.ndarray,
    delays: tuple[int, int],
    support: int,
) -> LinearOperator:
    """Matrix-free exact linear-convolution operator for both DAC inputs."""

    source = np.asarray(submitted, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("joint operator requires [frames,2] input")
    indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0 or indices[0] < 0 or indices[-1] >= len(source):
        raise ValueError("joint operator selected indices are invalid")
    shifted: list[np.ndarray] = []
    for channel, delay in enumerate(delays):
        if not 0 <= int(delay) <= MAX_DELAY:
            raise ValueError("joint operator delay outside predeclared branch")
        value = np.zeros(len(source), dtype=np.float64)
        if int(delay) < len(source):
            value[int(delay) :] = source[: len(source) - int(delay), channel]
        shifted.append(value)

    def matvec(taps: np.ndarray) -> np.ndarray:
        vector = np.asarray(taps, dtype=np.float64).reshape(2, support)
        predicted = np.zeros(len(source), dtype=np.float64)
        for channel in range(2):
            predicted += fftconvolve(shifted[channel], vector[channel], mode="full")[
                : len(source)
            ]
        return predicted[indices]

    def rmatvec(residual: np.ndarray) -> np.ndarray:
        embedded = np.zeros(len(source), dtype=np.float64)
        embedded[indices] = np.asarray(residual, dtype=np.float64).reshape(-1)
        gradients: list[np.ndarray] = []
        for channel in range(2):
            correlation = fftconvolve(
                embedded, shifted[channel][::-1], mode="full"
            )
            gradients.append(
                correlation[len(source) - 1 : len(source) - 1 + support]
            )
        return np.concatenate(gradients)

    return LinearOperator(
        shape=(indices.size, 2 * support),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.float64,
    )


def _joint_periodic_role_operator_v4(
    *,
    plan: Mapping[str, Any],
    submitted: np.ndarray,
    source_role: str,
    delays: tuple[int, int],
    support: int,
) -> LinearOperator:
    """양쪽 CP로 보장된 central period의 exact circular-convolution operator."""

    source = np.asarray(submitted, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("periodic joint operator는 [frames,2] input이 필요합니다")
    if int(support) not in SUPPORTS or int(support) + max(delays) > CYCLIC_PREFIX:
        raise ValueError("periodic operator history가 bilateral CP를 넘습니다")
    rows = _joint_role_rows(plan, source_role)
    input_spectra = np.empty((2, 2, PERIOD // 2 + 1), dtype=np.complex128)
    for row_index, row in enumerate(rows):
        start = int(row["central_start_frame"])
        for channel, delay in enumerate(delays):
            if not 0 <= int(delay) <= MAX_DELAY:
                raise ValueError("periodic operator delay가 predeclared branch 밖입니다")
            segment_start = start - int(delay)
            segment_stop = segment_start + PERIOD
            if segment_start < 0 or segment_stop > len(source):
                raise ValueError("periodic operator input history coverage가 없습니다")
            input_spectra[row_index, channel] = np.fft.rfft(
                source[segment_start:segment_stop, channel]
            )

    def matvec(taps: np.ndarray) -> np.ndarray:
        vector = np.asarray(taps, dtype=np.float64).reshape(2, support)
        transfer = np.stack(
            (
                np.fft.rfft(vector[0], n=PERIOD),
                np.fft.rfft(vector[1], n=PERIOD),
            )
        )
        predicted = [
            np.fft.irfft(
                input_spectra[row_index, 0] * transfer[0]
                + input_spectra[row_index, 1] * transfer[1],
                n=PERIOD,
            )
            for row_index in range(2)
        ]
        return np.concatenate(predicted)

    def rmatvec(residual: np.ndarray) -> np.ndarray:
        value = np.asarray(residual, dtype=np.float64).reshape(2, PERIOD)
        residual_spectra = np.fft.rfft(value, axis=1)
        gradients: list[np.ndarray] = []
        for channel in range(2):
            correlation_spectrum = np.sum(
                np.conj(input_spectra[:, channel]) * residual_spectra,
                axis=0,
            )
            gradients.append(
                np.fft.irfft(correlation_spectrum, n=PERIOD)[:support]
            )
        return np.concatenate(gradients)

    return LinearOperator(
        shape=(2 * PERIOD, 2 * support),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.float64,
    )


def periodic_excitation_condition_audit_v4(
    *,
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    source_role: str,
    delays: tuple[int, int],
) -> dict[str, Any]:
    """최단 support의 exact Gram condition으로 모든 긴 support 하한을 증명한다."""

    if source_role not in ("fit_a", "fit_b"):
        raise ValueError("condition audit은 fit_a/fit_b만 사용합니다")
    submitted = np.asarray(submitted_pcm)
    if (
        submitted.dtype != np.int16
        or _sha256_array(submitted) != plan["output"]["pcm_sha256"]
    ):
        raise ValueError("condition audit actual-int16 lineage mismatch")
    support = CONDITION_AUDIT_SUPPORT
    source = submitted.astype(np.float64) / 32767.0
    rows = _joint_role_rows(plan, source_role)
    spectra = np.empty((2, 2, PERIOD // 2 + 1), dtype=np.complex128)
    for row_index, row in enumerate(rows):
        start = int(row["central_start_frame"])
        for channel, delay in enumerate(delays):
            if not 0 <= int(delay) <= MAX_DELAY:
                raise ValueError("condition audit delay가 predeclared branch 밖입니다")
            segment_start = start - int(delay)
            segment_stop = segment_start + PERIOD
            spectra[row_index, channel] = np.fft.rfft(
                source[segment_start:segment_stop, channel]
            )
    indices = np.arange(support, dtype=np.int64)
    circular_difference = (indices[:, None] - indices[None, :]) % PERIOD
    blocks: list[list[np.ndarray]] = []
    for left_channel in range(2):
        block_row: list[np.ndarray] = []
        for right_channel in range(2):
            kernel = np.fft.irfft(
                np.sum(
                    np.conj(spectra[:, left_channel])
                    * spectra[:, right_channel],
                    axis=0,
                ),
                n=PERIOD,
            )
            block_row.append(kernel[circular_difference])
        blocks.append(block_row)
    gram = np.block(blocks)
    gram = 0.5 * (gram + gram.T)
    dimension = int(gram.shape[0])
    smallest = float(
        eigvalsh(
            gram,
            subset_by_index=[0, 0],
            driver="evr",
            check_finite=True,
        )[0]
    )
    largest = float(
        eigvalsh(
            gram,
            subset_by_index=[dimension - 1, dimension - 1],
            driver="evr",
            check_finite=True,
        )[0]
    )
    single_path_conditions: dict[str, float] = {}
    for channel, path in enumerate(("primary", "secondary")):
        start = channel * support
        stop = start + support
        block = gram[start:stop, start:stop]
        block_smallest = float(
            eigvalsh(
                block,
                subset_by_index=[0, 0],
                driver="evr",
                check_finite=True,
            )[0]
        )
        block_largest = float(
            eigvalsh(
                block,
                subset_by_index=[support - 1, support - 1],
                driver="evr",
                check_finite=True,
            )[0]
        )
        single_path_conditions[path] = (
            float(math.sqrt(block_largest / block_smallest))
            if block_smallest > 0.0
            else float(np.finfo(np.float64).max)
        )
    condition = (
        float(math.sqrt(largest / smallest))
        if smallest > 0.0
        else float(np.finfo(np.float64).max)
    )
    passed = bool(math.isfinite(condition) and condition <= MAX_CONDITION)
    receipt: dict[str, Any] = {
        "schema": "actual_int16_periodic_excitation_condition_audit_v4",
        "source_role": source_role,
        "submitted_pcm_sha256": _sha256_array(submitted),
        "delays_samples": [int(value) for value in delays],
        "audit_support_samples": support,
        "gram_dimension": dimension,
        "smallest_eigenvalue": smallest,
        "largest_eigenvalue": largest,
        "condition_number": condition,
        "delay_independent_single_path_condition_numbers": (
            single_path_conditions
        ),
        "delay_independent_joint_condition_lower_bound": float(
            max(single_path_conditions.values())
        ),
        "maximum_condition_number": MAX_CONDITION,
        "passed": passed,
        "longer_support_condition_cannot_improve": True,
        "proof": (
            "A_s^T A_s is a principal submatrix of every longer-support Gram; "
            "Cauchy interlacing makes lambda_max nondecreasing and lambda_min "
            "nonincreasing"
        ),
        "all_predeclared_supports_blocked": bool(not passed),
        "canonical_training_eligible": False,
    }
    receipt["receipt_sha256"] = _json_sha256(receipt)
    return receipt


def _operator_condition(operator: LinearOperator) -> float:
    normal = LinearOperator(
        (operator.shape[1], operator.shape[1]),
        matvec=lambda value: operator.rmatvec(operator.matvec(value)),
        dtype=np.float64,
    )
    initial = np.full(
        operator.shape[1], 1.0 / math.sqrt(operator.shape[1]), dtype=np.float64
    )
    largest = float(
        eigsh(
            normal,
            k=1,
            which="LA",
            return_eigenvectors=False,
            v0=initial,
            tol=1e-6,
            maxiter=1_000,
        )[0]
    )
    smallest = float(
        eigsh(
            normal,
            k=1,
            which="SA",
            return_eigenvectors=False,
            v0=initial,
            tol=1e-6,
            maxiter=1_000,
        )[0]
    )
    if smallest <= 0.0:
        return math.inf
    return float(math.sqrt(largest / smallest))


def _joint_role_rows(
    plan: Mapping[str, Any], source_role: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    rows = tuple(
        row
        for row in plan["layout"]
        if row.get("role") == source_role and bool(row.get("candidate_data"))
    )
    if len(rows) != 2 or tuple(str(row.get("path")) for row in rows) != (
        "primary",
        "secondary",
    ):
        raise ValueError("joint P/S role에는 primary/secondary central row가 각각 하나여야 합니다")
    if any(
        int(row["central_stop_frame"]) - int(row["central_start_frame"])
        != PERIOD
        for row in rows
    ):
        raise ValueError("joint P/S central row 길이가 PERIOD와 다릅니다")
    return rows  # type: ignore[return-value]


def _plan_subbands_v4(
    plan: Mapping[str, Any],
) -> tuple[tuple[float, float], ...]:
    try:
        bands = tuple(
            tuple(float(value) for value in band)
            for band in plan["plant_identification"]["subband_authority"][
                "bands_hz"
            ]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("plan control subband contract가 없습니다") from error
    if not bands or any(
        len(band) != 2
        or not all(math.isfinite(value) for value in band)
        or not 0.0 < band[0] < band[1] < FS / 2.0
        for band in bands
    ):
        raise ValueError("plan control subband contract가 잘못됐습니다")
    if any(
        not math.isclose(bands[index - 1][1], bands[index][0], abs_tol=1e-9)
        for index in range(1, len(bands))
    ):
        raise ValueError("plan control subband에 gap/overlap이 있습니다")
    return bands


def _band_mask_v4(
    frequency_hz: np.ndarray,
    band_index: int,
    subbands_hz: Sequence[Sequence[float]],
) -> np.ndarray:
    lo, hi = subbands_hz[int(band_index)]
    if int(band_index) == len(subbands_hz) - 1:
        return (frequency_hz >= float(lo)) & (frequency_hz <= float(hi))
    return (frequency_hz >= float(lo)) & (frequency_hz < float(hi))


def _spectral_rms_dbfs_v4(spectrum: np.ndarray) -> float:
    energy = float(np.sum(np.abs(np.asarray(spectrum)) ** 2))
    rms = math.sqrt(max(0.0, 2.0 * energy)) / PERIOD
    return float(20.0 * math.log10(max(rms, 1.0e-300)))


def _candidate_identity_v4(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("candidate_sha256", candidate.get("freeze_sha256"))
    if not isinstance(value, str) or not value:
        raise ValueError("candidate/frozen identity SHA가 없습니다")
    return value


def _exact_zero_noise_floor_v4(
    *,
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    response_dac_q: np.ndarray,
    microphone_role: str,
) -> dict[str, Any]:
    """실제 제출 PCM이 두 입력 모두 exact-zero인 bin에서 noise floor를 구한다."""

    submitted = np.asarray(submitted_pcm)
    response = np.asarray(response_dac_q, dtype=np.float64).reshape(-1)
    frequency = np.fft.rfftfreq(PERIOD, 1.0 / FS)
    subbands = _plan_subbands_v4(plan)
    named_rows = {str(row["name"]): row for row in plan["clock_rows"]}
    reference_names = ("lead_reference", "tail_validation")
    rows: list[dict[str, Any]] = []
    for band_index, (lo, hi) in enumerate(subbands):
        band = _band_mask_v4(frequency, band_index, subbands)
        powers: list[np.ndarray] = []
        row_counts: list[int] = []
        zero_indices: list[np.ndarray] = []
        for name in reference_names:
            if name not in named_rows:
                raise ValueError(f"noise reference clock row가 없습니다: {name}")
            row = named_rows[name]
            start = int(row["start_frame"])
            stop = int(row["stop_frame"])
            if stop - start != PERIOD or stop > len(submitted) or stop > response.size:
                raise ValueError("noise reference row의 길이/coverage가 잘못됐습니다")
            actual_spectrum = np.fft.rfft(
                submitted[start:stop].astype(np.float64), axis=0
            )
            exact_zero = (
                band
                & (np.abs(actual_spectrum[:, 0]) <= EXACT_ZERO_DFT_MAX)
                & (np.abs(actual_spectrum[:, 1]) <= EXACT_ZERO_DFT_MAX)
            )
            indices = np.flatnonzero(exact_zero).astype(np.int64)
            count = int(indices.size)
            if count < SUBBAND_MIN_EXACT_ZERO_NOISE_BINS:
                raise ValueError(
                    f"{name} {lo:g}-{hi:g}Hz actual exact-zero noise bin 부족: {count}"
                )
            response_spectrum = np.fft.rfft(response[start:stop])
            powers.append(np.abs(response_spectrum[indices]) ** 2)
            row_counts.append(count)
            zero_indices.append(indices)
        combined_power = np.concatenate(powers)
        # 평균이나 최솟값으로 transient/noise를 숨기지 않도록 live raw를 보기 전에
        # 고정한 보수적 per-bin p95를 사용한다.
        noise_power = float(np.percentile(combined_power, 95.0))
        row = {
            "band_index": int(band_index),
            "band_hz": [float(lo), float(hi)],
            "reference_rows": list(reference_names),
            "exact_zero_bin_count_by_reference_row": row_counts,
            "minimum_exact_zero_bin_count": int(min(row_counts)),
            "combined_exact_zero_bin_count": int(combined_power.size),
            "exact_zero_bin_indices_sha256": _json_sha256(
                [indices.tolist() for indices in zero_indices]
            ),
            "noise_power_per_bin": noise_power,
            "noise_estimator": "p95_abs_rfft_squared",
            "passed": bool(
                min(row_counts) >= SUBBAND_MIN_EXACT_ZERO_NOISE_BINS
                and math.isfinite(noise_power)
                and noise_power >= 0.0
            ),
        }
        rows.append(row)
    receipt: dict[str, Any] = {
        "schema": "actual_exact_zero_noise_floor_v4",
        "microphone_role": microphone_role,
        "submitted_pcm_sha256": _sha256_array(submitted),
        "response_sha256": _sha256_array(response),
        "actual_both_input_exact_zero_required": True,
        "reference_rows": list(reference_names),
        "subbands": rows,
        "passed": bool(all(row["passed"] for row in rows)),
    }
    receipt["receipt_sha256"] = _json_sha256(receipt)
    return receipt


def _score_path_subbands_v4(
    *,
    path: str,
    actual_input_period: np.ndarray,
    target_period: np.ndarray,
    predicted_period: np.ndarray,
    noise_floor: Mapping[str, Any],
    subbands_hz: Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    """한 path의 contract 전 대역을 독립 판정한다. global 평균은 사용하지 않는다."""

    channel = PATH_CHANNEL[path]
    other = 1 - channel
    actual_spectrum = np.fft.rfft(
        np.asarray(actual_input_period, dtype=np.float64), axis=0
    )
    normalized_spectrum = actual_spectrum / 32767.0
    target_spectrum = np.fft.rfft(np.asarray(target_period, dtype=np.float64))
    predicted_spectrum = np.fft.rfft(
        np.asarray(predicted_period, dtype=np.float64)
    )
    frequency = np.fft.rfftfreq(PERIOD, 1.0 / FS)
    floor_by_index = {
        int(row["band_index"]): row for row in noise_floor["subbands"]
    }
    reports: list[dict[str, Any]] = []
    for band_index, (lo, hi) in enumerate(subbands_hz):
        # active path input이 실제 int16 DFT에서 nonzero이고 반대 path는 exact-zero인
        # bin만 사용한다. 저역 pilot도 이 규칙으로 plant 식별에 직접 포함된다.
        isolated = (
            _band_mask_v4(frequency, band_index, subbands_hz)
            & (np.abs(actual_spectrum[:, channel]) > EXACT_ZERO_DFT_MAX)
            & (np.abs(actual_spectrum[:, other]) <= EXACT_ZERO_DFT_MAX)
        )
        indices = np.flatnonzero(isolated)
        response_bin_count = int(indices.size)
        if response_bin_count < SUBBAND_MIN_RESPONSE_BINS:
            raise ValueError(
                f"{path} {lo:g}-{hi:g}Hz isolated response bin 부족: {response_bin_count}"
            )
        target = target_spectrum[indices]
        predicted = predicted_spectrum[indices]
        target_energy = float(np.sum(np.abs(target) ** 2))
        residual_energy = float(np.sum(np.abs(predicted - target) ** 2))
        noise_row = floor_by_index[band_index]
        noise_power = float(noise_row["noise_power_per_bin"])
        raw_noise_energy = noise_power * response_bin_count
        # 수치적으로 exact-zero인 synthetic fixture에서도 finite JSON이 되도록
        # target energy의 1e-30을 분석 floor로 둔다. 이는 SNR을 최대 300 dB로
        # cap할 뿐 PASS 임계(20 dB)를 완화하지 않는다.
        noise_energy = max(raw_noise_energy, target_energy * 1.0e-30, 1.0e-300)
        conditioned_target_energy = max(target_energy - raw_noise_energy, 0.0)
        target_to_noise_db = float(
            10.0
            * math.log10(
                max(conditioned_target_energy, 1.0e-300) / noise_energy
            )
        )
        normalized_residual = float(
            math.sqrt(residual_energy)
            / max(math.sqrt(conditioned_target_energy), 1.0e-300)
        )
        denominator = float(np.linalg.norm(target) * np.linalg.norm(predicted))
        complex_agreement = (
            float(abs(complex(np.vdot(target, predicted))) / denominator)
            if denominator > 0.0
            else 0.0
        )
        phase_amplitude_floor = math.sqrt(max(noise_power, 0.0)) * (
            10.0 ** (SUBBAND_PHASE_BIN_MIN_SNR_DB / 20.0)
        )
        phase_valid = (
            (np.abs(target) > phase_amplitude_floor)
            & (np.abs(predicted) > 0.0)
        )
        phase_bin_count = int(np.count_nonzero(phase_valid))
        if phase_bin_count >= SUBBAND_MIN_RESPONSE_BINS:
            timing_limit = max_timing_error_samples_for_attenuation(
                20.0, float(hi), FS
            )
            phase_delay, phase_coherence = _fractional_delay(
                predicted[phase_valid] / target[phase_valid],
                frequency[indices][phase_valid],
                width=max(1.0, 1.25 * timing_limit),
            )
        else:
            timing_limit = max_timing_error_samples_for_attenuation(
                20.0, float(hi), FS
            )
            phase_delay, phase_coherence = max(1.0, 2.0 * timing_limit), 0.0
        input_rms_dbfs = _spectral_rms_dbfs_v4(
            normalized_spectrum[indices, channel]
        )
        target_rms_dbfs = _spectral_rms_dbfs_v4(target)
        passed = bool(
            int(noise_row["minimum_exact_zero_bin_count"])
            >= SUBBAND_MIN_EXACT_ZERO_NOISE_BINS
            and response_bin_count >= SUBBAND_MIN_RESPONSE_BINS
            and phase_bin_count >= SUBBAND_MIN_RESPONSE_BINS
            and input_rms_dbfs >= SUBBAND_MIN_INPUT_RMS_DBFS
            and target_rms_dbfs >= SUBBAND_MIN_TARGET_RMS_DBFS
            and target_to_noise_db >= SUBBAND_MIN_TARGET_TO_NOISE_DB
            and normalized_residual <= SUBBAND_MAX_RELATIVE_ERROR
            and complex_agreement >= SUBBAND_MIN_COMPLEX_AGREEMENT
            and phase_coherence >= SUBBAND_MIN_COMPLEX_AGREEMENT
            and abs(float(phase_delay)) <= timing_limit
        )
        reports.append(
            {
                "path": path,
                "band_index": int(band_index),
                "band_hz": [float(lo), float(hi)],
                "isolated_response_bin_count": response_bin_count,
                "phase_bin_count": phase_bin_count,
                "isolated_input_bin_indices_sha256": _sha256_array(
                    indices.astype(np.int64)
                ),
                "input_rms_dbfs": input_rms_dbfs,
                "target_rms_dbfs": target_rms_dbfs,
                "target_energy": target_energy,
                "conditioned_target_energy": conditioned_target_energy,
                "exact_zero_noise_bin_count": int(
                    noise_row["minimum_exact_zero_bin_count"]
                ),
                "noise_power_per_bin": noise_power,
                "target_to_noise_db": target_to_noise_db,
                "noise_conditioned_relative_residual": normalized_residual,
                "complex_agreement": complex_agreement,
                "phase_coherence": float(phase_coherence),
                "phase_delay_samples": float(phase_delay),
                "max_abs_phase_delay_samples": timing_limit,
                "passed": passed,
            }
        )
    return reports


def generate_fit_candidate_v4(
    *,
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    response_dac_q: np.ndarray,
    microphone_role: str,
    source_role: str,
    delays: tuple[int, int],
    fractional_delays_samples: tuple[float, float],
    support: int,
) -> dict[str, Any]:
    """Fit one joint P/S FIR from fit_a or fit_b only.

    The exact full submitted stereo input is used.  In particular, the
    continuous low-band pilots and the opposite-path pilot remain in the
    operator; neither is subtracted or replaced by an intended float signal.
    """

    if microphone_role not in ("err", "ref"):
        raise ValueError("microphone_role은 err/ref여야 합니다")
    if source_role not in ("fit_a", "fit_b"):
        raise ValueError("fit_a/fit_b만 candidate 생성에 사용할 수 있습니다")
    if int(support) not in SUPPORTS:
        raise ValueError("candidate support is not predeclared")
    fractional = tuple(float(value) for value in fractional_delays_samples)
    if len(fractional) != 2 or any(
        not math.isfinite(value) or not -0.5 <= value < 0.5
        for value in fractional
    ):
        raise ValueError("fractional delay residual must be finite in [-0.5,0.5)")
    submitted = np.asarray(submitted_pcm)
    if submitted.dtype != np.int16 or _sha256_array(submitted) != plan["output"][
        "pcm_sha256"
    ]:
        raise ValueError("candidate actual submitted int16 lineage mismatch")
    response = np.asarray(response_dac_q, dtype=np.float64).reshape(-1)
    if response.size < len(submitted):
        raise ValueError("candidate response is shorter than submitted input")
    source = submitted.astype(np.float64) / 32767.0
    indices = _joint_role_indices(plan, source_role)
    condition_audit = periodic_excitation_condition_audit_v4(
        plan=plan,
        submitted_pcm=submitted,
        source_role=source_role,
        delays=(int(delays[0]), int(delays[1])),
    )
    if bool(condition_audit["all_predeclared_supports_blocked"]):
        raise ValueError(
            "actual-int16 excitation condition이 predeclared 20을 넘고 "
            "principal-submatrix interlacing상 모든 support가 BLOCKED입니다: "
            f"{condition_audit['condition_number']}"
        )
    operator = _joint_periodic_role_operator_v4(
        plan=plan,
        submitted=source,
        source_role=source_role,
        delays=(int(delays[0]), int(delays[1])),
        support=int(support),
    )
    condition = (
        float(condition_audit["condition_number"])
        if int(support) == CONDITION_AUDIT_SUPPORT
        else _operator_condition(operator)
    )
    if not math.isfinite(condition) or condition > MAX_CONDITION:
        raise ValueError(f"joint actual-input condition {condition} > {MAX_CONDITION}")
    target = response[indices]
    solution = lsmr(
        operator,
        target,
        atol=1e-11,
        btol=1e-11,
        conlim=MAX_CONDITION,
        maxiter=max(2_000, 4 * operator.shape[1]),
    )
    taps = np.asarray(solution[0], dtype=np.float64).reshape(2, int(support))
    residual = operator.matvec(taps.reshape(-1)) - target
    residual_ratio = float(
        np.linalg.norm(residual) / max(np.linalg.norm(target), 1e-30)
    )
    payload: dict[str, Any] = {
        "schema": "joint_actual_input_fit_candidate_v4",
        "microphone_role": microphone_role,
        "source_role": source_role,
        "control_band_contract_sha256": plan["control_band_contract_sha256"],
        "control_subbands_hz": [
            [float(value) for value in band] for band in _plan_subbands_v4(plan)
        ],
        "submitted_pcm_sha256": _sha256_array(submitted),
        "response_sha256": _sha256_array(response),
        "selected_indices_sha256": _sha256_array(indices),
        "delays_samples": [int(delays[0]), int(delays[1])],
        "coarse_integer_delays_samples": [int(delays[0]), int(delays[1])],
        "fractional_delay_residual_samples": list(fractional),
        "bulk_delay_samples_fractional": [
            float(int(delays[0]) + fractional[0]),
            float(int(delays[1]) + fractional[1]),
        ],
        "support_samples": int(support),
        "condition_number": condition,
        "condition_audit_receipt_sha256": condition_audit["receipt_sha256"],
        "condition_method": (
            "matrix_free_extremal_eigenvalues_of_exact_cp_periodic_A_transpose_A"
        ),
        "solver": "matrix_free_lsmr_no_ridge",
        "ridge": 0.0,
        "solver_stop_code": int(solution[1]),
        "solver_iterations": int(solution[2]),
        "fit_residual_ratio": residual_ratio,
        "actual_full_input_including_continuous_pilot": True,
        "pilot_low_band_included_in_joint_fit": True,
        "fractional_delay_encoded_in_post_onset_fir": True,
        "post_onset_peak_index": [
            int(np.argmax(np.abs(taps[0]))),
            int(np.argmax(np.abs(taps[1]))),
        ],
        "primary_post_onset_fir": taps[0].tolist(),
        "secondary_post_onset_fir": taps[1].tolist(),
        "holdout_used_for_generation_or_selection": False,
        "canonical_training_eligible": False,
    }
    payload["candidate_sha256"] = _json_sha256(payload)
    return payload


def score_candidate_on_role_v4(
    *,
    candidate: Mapping[str, Any],
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    response_dac_q: np.ndarray,
    microphone_role: str,
    target_role: str,
) -> dict[str, Any]:
    """candidate를 global 평균과 별개로 P/S×contract 전 대역에서 판정한다."""

    if microphone_role not in ("err", "ref"):
        raise ValueError("microphone_role은 err/ref여야 합니다")
    if target_role not in ("fit_a", "fit_b", "holdout"):
        raise ValueError("unknown score target role")
    support = int(candidate["support_samples"])
    taps = np.stack(
        (
            np.asarray(candidate["primary_post_onset_fir"], dtype=np.float64),
            np.asarray(candidate["secondary_post_onset_fir"], dtype=np.float64),
        )
    )
    if taps.shape != (2, support):
        raise ValueError("candidate tap shape mismatch")
    source = np.asarray(submitted_pcm)
    if (
        source.dtype != np.int16
        or _sha256_array(source) != candidate["submitted_pcm_sha256"]
        or _sha256_array(source) != plan["output"]["pcm_sha256"]
    ):
        raise ValueError("candidate score input lineage mismatch")
    candidate_microphone = candidate.get("microphone_role")
    if candidate_microphone is not None and candidate_microphone != microphone_role:
        raise ValueError("candidate/score microphone role mismatch")
    plan_subbands = _plan_subbands_v4(plan)
    candidate_subbands = tuple(
        tuple(float(value) for value in band)
        for band in candidate.get("control_subbands_hz", ())
    )
    if (
        candidate.get("control_band_contract_sha256")
        != plan.get("control_band_contract_sha256")
        or candidate_subbands != plan_subbands
    ):
        raise ValueError("candidate/score control-band contract mismatch")
    indices = _joint_role_indices(plan, target_role)
    operator = _joint_periodic_role_operator_v4(
        plan=plan,
        submitted=source.astype(np.float64) / 32767.0,
        source_role=target_role,
        delays=tuple(int(value) for value in candidate["delays_samples"]),
        support=support,
    )
    response = np.asarray(response_dac_q, dtype=np.float64).reshape(-1)
    if response.size < len(source):
        raise ValueError("candidate score response가 submitted capture보다 짧습니다")
    target = response[indices]
    predicted = operator.matvec(taps.reshape(-1))
    residual = predicted - target
    global_residual = float(
        np.linalg.norm(residual) / max(np.linalg.norm(target), 1e-30)
    )
    noise_floor = _exact_zero_noise_floor_v4(
        plan=plan,
        submitted_pcm=source,
        response_dac_q=response,
        microphone_role=microphone_role,
    )
    subbands = plan_subbands
    rows = _joint_role_rows(plan, target_role)
    subband_rows: list[dict[str, Any]] = []
    offset = 0
    for row in rows:
        start = int(row["central_start_frame"])
        stop = int(row["central_stop_frame"])
        count = stop - start
        subband_rows.extend(
            _score_path_subbands_v4(
                path=str(row["path"]),
                actual_input_period=source[start:stop],
                target_period=target[offset : offset + count],
                predicted_period=predicted[offset : offset + count],
                noise_floor=noise_floor,
                subbands_hz=subbands,
            )
        )
        offset += count
    if offset != predicted.size or len(subband_rows) != 2 * len(subbands):
        raise AssertionError("P/S×7 subband score shape가 잘못됐습니다")
    candidate_source_role = str(candidate.get("source_role", "frozen_fit_only"))
    global_limit = (
        FIT_RESIDUAL_MAX
        if target_role in ("fit_a", "fit_b")
        and candidate_source_role == target_role
        else CROSS_FIT_RESIDUAL_MAX
    )
    all_subbands_passed = bool(
        noise_floor["passed"] and all(row["passed"] for row in subband_rows)
    )
    receipt: dict[str, Any] = {
        "schema": "joint_actual_input_role_score_v4",
        "candidate_sha256": _candidate_identity_v4(candidate),
        "candidate_source_role": candidate_source_role,
        "microphone_role": microphone_role,
        "target_role": target_role,
        "submitted_pcm_sha256": _sha256_array(source),
        "response_sha256": _sha256_array(response),
        "control_band_contract_sha256": (
            plan["control_band_contract_sha256"]
        ),
        "global_residual_ratio": global_residual,
        "global_residual_threshold": global_limit,
        "global_residual_is_not_sufficient": True,
        "noise_floor_receipt": noise_floor,
        "subband_rows": subband_rows,
        "all_subbands_passed": all_subbands_passed,
        "passed": bool(global_residual <= global_limit and all_subbands_passed),
        "thresholds": {
            "maximum_noise_conditioned_relative_residual": (
                SUBBAND_MAX_RELATIVE_ERROR
            ),
            "minimum_complex_agreement": SUBBAND_MIN_COMPLEX_AGREEMENT,
            "minimum_target_to_noise_db": SUBBAND_MIN_TARGET_TO_NOISE_DB,
            "minimum_input_rms_dbfs": SUBBAND_MIN_INPUT_RMS_DBFS,
            "minimum_target_rms_dbfs": SUBBAND_MIN_TARGET_RMS_DBFS,
            "minimum_exact_zero_noise_bins": (
                SUBBAND_MIN_EXACT_ZERO_NOISE_BINS
            ),
            "minimum_response_bins": SUBBAND_MIN_RESPONSE_BINS,
            "phase_bin_min_snr_db": SUBBAND_PHASE_BIN_MIN_SNR_DB,
            "timing_resolution_attenuation_db": 20.0,
        },
        "holdout_used_for_generation_or_selection": False,
    }
    receipt["receipt_sha256"] = _json_sha256(receipt)
    return receipt


def _fit_transfer_stationarity_v4(
    fit_a: Mapping[str, Any], fit_b: Mapping[str, Any]
) -> dict[str, Any]:
    """fit_a/fit_b FIR transfer가 P/S 각 contract 대역에서 같은지 검사한다."""

    if (
        fit_a.get("control_band_contract_sha256")
        != fit_b.get("control_band_contract_sha256")
        or fit_a.get("control_subbands_hz") != fit_b.get("control_subbands_hz")
    ):
        raise ValueError("fit_a/fit_b control-band contract가 다릅니다")
    try:
        subbands = tuple(
            tuple(float(value) for value in band)
            for band in fit_a["control_subbands_hz"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("fit candidate control subband가 없습니다") from error
    frequency = np.fft.rfftfreq(PERIOD, 1.0 / FS)
    reports: list[dict[str, Any]] = []
    for path, key in (
        ("primary", "primary_post_onset_fir"),
        ("secondary", "secondary_post_onset_fir"),
    ):
        taps_a = np.asarray(fit_a[key], dtype=np.float64)
        taps_b = np.asarray(fit_b[key], dtype=np.float64)
        if taps_a.shape != taps_b.shape or taps_a.ndim != 1 or taps_a.size > PERIOD:
            raise ValueError("fit_a/fit_b stationarity FIR shape가 다릅니다")
        transfer_a = np.fft.rfft(taps_a, n=PERIOD)
        transfer_b = np.fft.rfft(taps_b, n=PERIOD)
        for band_index, (lo, hi) in enumerate(subbands):
            mask = _band_mask_v4(frequency, band_index, subbands)
            a = transfer_a[mask]
            b = transfer_b[mask]
            if a.size < SUBBAND_MIN_RESPONSE_BINS:
                raise ValueError("fit transfer stationarity subband bin이 부족합니다")
            norm_a = float(np.linalg.norm(a))
            norm_b = float(np.linalg.norm(b))
            denominator = norm_a * norm_b
            agreement = (
                float(abs(complex(np.vdot(a, b))) / denominator)
                if denominator > 0.0
                else 0.0
            )
            delta = float(np.linalg.norm(b - a))
            symmetric_error = max(
                delta / max(norm_a, 1.0e-300),
                delta / max(norm_b, 1.0e-300),
            )
            amplitude_floor = max(
                float(np.max(np.abs(a))), float(np.max(np.abs(b)))
            ) * 1.0e-8
            valid = (np.abs(a) > amplitude_floor) & (np.abs(b) > amplitude_floor)
            valid_count = int(np.count_nonzero(valid))
            timing_limit = max_timing_error_samples_for_attenuation(
                20.0, float(hi), FS
            )
            if valid_count >= SUBBAND_MIN_RESPONSE_BINS:
                delay, phase_coherence = _fractional_delay(
                    b[valid] / a[valid],
                    frequency[mask][valid],
                    width=max(1.0, 1.25 * timing_limit),
                )
            else:
                delay, phase_coherence = max(1.0, 2.0 * timing_limit), 0.0
            passed = bool(
                valid_count >= SUBBAND_MIN_RESPONSE_BINS
                and symmetric_error <= SUBBAND_MAX_RELATIVE_ERROR
                and agreement >= SUBBAND_MIN_COMPLEX_AGREEMENT
                and phase_coherence >= SUBBAND_MIN_COMPLEX_AGREEMENT
                and abs(float(delay)) <= timing_limit
            )
            reports.append(
                {
                    "path": path,
                    "band_index": int(band_index),
                    "band_hz": [float(lo), float(hi)],
                    "transfer_bin_count": int(a.size),
                    "phase_bin_count": valid_count,
                    "symmetric_relative_error": symmetric_error,
                    "complex_agreement": agreement,
                    "phase_coherence": float(phase_coherence),
                    "phase_delay_samples": float(delay),
                    "max_abs_phase_delay_samples": timing_limit,
                    "passed": passed,
                }
            )
    receipt: dict[str, Any] = {
        "schema": "fit_a_fit_b_transfer_stationarity_v4",
        "fit_a_candidate_sha256": _candidate_identity_v4(fit_a),
        "fit_b_candidate_sha256": _candidate_identity_v4(fit_b),
        "control_band_contract_sha256": fit_a[
            "control_band_contract_sha256"
        ],
        "subband_rows": reports,
        "all_subbands_passed": bool(all(row["passed"] for row in reports)),
    }
    receipt["passed"] = receipt["all_subbands_passed"]
    receipt["receipt_sha256"] = _json_sha256(receipt)
    return receipt


def _validate_score_receipt_v4(
    score: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    target_role: str,
    microphone_role: str,
    require_pass: bool = True,
) -> None:
    expected_sha = _candidate_identity_v4(candidate)
    try:
        subbands = tuple(
            tuple(float(value) for value in band)
            for band in candidate["control_subbands_hz"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("candidate control subband가 없습니다") from error
    if (
        score.get("schema") != "joint_actual_input_role_score_v4"
        or score.get("candidate_sha256") != expected_sha
        or score.get("target_role") != target_role
        or score.get("microphone_role") != microphone_role
        or score.get("control_band_contract_sha256")
        != candidate.get("control_band_contract_sha256")
    ):
        raise ValueError("score receipt identity/role가 잘못됐습니다")
    if require_pass and (
        not bool(score.get("all_subbands_passed"))
        or not bool(score.get("passed"))
    ):
        raise ValueError("fit score receipt가 PASS가 아닙니다")
    rows = score.get("subband_rows")
    if not isinstance(rows, list) or len(rows) != 2 * len(subbands):
        raise ValueError("fit score에는 P/S×contract 전 subband row가 있어야 합니다")
    expected = {
        (path, band_index)
        for path in PATH_CHANNEL
        for band_index in range(len(subbands))
    }
    observed = {
        (str(row.get("path")), int(row.get("band_index", -1))) for row in rows
    }
    if observed != expected or (
        require_pass and any(not bool(row.get("passed")) for row in rows)
    ):
        raise ValueError("fit score P/S×contract subband 중 누락/실패가 있습니다")
    if any(
        tuple(float(value) for value in row.get("band_hz", ()))
        != subbands[int(row["band_index"])]
        for row in rows
    ):
        raise ValueError("fit score subband edge가 contract와 다릅니다")
    payload = dict(score)
    claimed_sha = payload.pop("receipt_sha256", None)
    if claimed_sha != _json_sha256(payload):
        raise ValueError("fit score receipt SHA가 잘못됐습니다")


def select_and_freeze_fit_support_v4(
    fit_receipts_by_support: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    """Select the shortest support using fit_a/fit_b evidence only."""

    if any("holdout" in receipt for receipt in fit_receipts_by_support.values()):
        raise ValueError("holdout may not enter support selection")
    considered: list[dict[str, Any]] = []
    selected: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]] | None = None
    for support in SUPPORTS:
        if support not in fit_receipts_by_support:
            continue
        receipt = fit_receipts_by_support[support]
        if set(receipt) != {
            "fit_a",
            "fit_b",
            "fit_a_score",
            "fit_b_score",
            "fit_a_on_fit_b_score",
            "fit_b_on_fit_a_score",
        }:
            raise ValueError("fit-only support receipt fields are incomplete")
        fit_a = receipt["fit_a"]
        fit_b = receipt["fit_b"]
        if (
            fit_a.get("source_role") != "fit_a"
            or fit_b.get("source_role") != "fit_b"
            or int(fit_a.get("support_samples", -1)) != support
            or int(fit_b.get("support_samples", -1)) != support
        ):
            raise ValueError("fit candidate role/support mismatch")
        if (
            fit_a.get("delays_samples") != fit_b.get("delays_samples")
            or fit_a.get("fractional_delay_residual_samples")
            != fit_b.get("fractional_delay_residual_samples")
            or fit_a.get("microphone_role") != fit_b.get("microphone_role")
            or fit_a.get("submitted_pcm_sha256")
            != fit_b.get("submitted_pcm_sha256")
        ):
            raise ValueError("fit candidate mic/integer/fractional timing mismatch")
        microphone_role = str(fit_a.get("microphone_role"))
        if microphone_role not in ("err", "ref"):
            raise ValueError("fit candidate microphone role이 없습니다")
        _validate_score_receipt_v4(
            receipt["fit_a_score"],
            candidate=fit_a,
            target_role="fit_a",
            microphone_role=microphone_role,
        )
        _validate_score_receipt_v4(
            receipt["fit_b_score"],
            candidate=fit_b,
            target_role="fit_b",
            microphone_role=microphone_role,
        )
        _validate_score_receipt_v4(
            receipt["fit_a_on_fit_b_score"],
            candidate=fit_a,
            target_role="fit_b",
            microphone_role=microphone_role,
        )
        _validate_score_receipt_v4(
            receipt["fit_b_on_fit_a_score"],
            candidate=fit_b,
            target_role="fit_a",
            microphone_role=microphone_role,
        )
        primary_a = np.asarray(fit_a["primary_post_onset_fir"], dtype=np.float64)
        primary_b = np.asarray(fit_b["primary_post_onset_fir"], dtype=np.float64)
        secondary_a = np.asarray(fit_a["secondary_post_onset_fir"], dtype=np.float64)
        secondary_b = np.asarray(fit_b["secondary_post_onset_fir"], dtype=np.float64)
        disagreement = max(
            float(
                np.linalg.norm(primary_a - primary_b)
                / max(np.linalg.norm(primary_a), 1e-30)
            ),
            float(
                np.linalg.norm(secondary_a - secondary_b)
                / max(np.linalg.norm(secondary_a), 1e-30)
            ),
        )
        stationarity = _fit_transfer_stationarity_v4(fit_a, fit_b)
        passed = bool(
            float(fit_a["condition_number"]) <= MAX_CONDITION
            and float(fit_b["condition_number"]) <= MAX_CONDITION
            and float(fit_a["fit_residual_ratio"]) <= FIT_RESIDUAL_MAX
            and float(fit_b["fit_residual_ratio"]) <= FIT_RESIDUAL_MAX
            and float(receipt["fit_a_score"]["global_residual_ratio"])
            <= FIT_RESIDUAL_MAX
            and float(receipt["fit_b_score"]["global_residual_ratio"])
            <= FIT_RESIDUAL_MAX
            and float(receipt["fit_a_on_fit_b_score"]["global_residual_ratio"])
            <= CROSS_FIT_RESIDUAL_MAX
            and float(receipt["fit_b_on_fit_a_score"]["global_residual_ratio"])
            <= CROSS_FIT_RESIDUAL_MAX
            and disagreement <= FIT_TAP_DISAGREEMENT_MAX
            and bool(stationarity["passed"])
        )
        row = {
            "support_samples": support,
            "passed": passed,
            "fit_tap_relative_disagreement": disagreement,
            "fit_transfer_stationarity_receipt_sha256": stationarity[
                "receipt_sha256"
            ],
            "fit_transfer_stationarity": stationarity,
            "score_receipt_sha256": {
                name: receipt[name]["receipt_sha256"]
                for name in (
                    "fit_a_score",
                    "fit_b_score",
                    "fit_a_on_fit_b_score",
                    "fit_b_on_fit_a_score",
                )
            },
        }
        considered.append(row)
        if passed and selected is None:
            selected = (fit_a, fit_b, receipt)
    if selected is None:
        raise ValueError("no predeclared support passed fit-only selection")
    fit_a, fit_b, _ = selected
    primary = 0.5 * (
        np.asarray(fit_a["primary_post_onset_fir"], dtype=np.float64)
        + np.asarray(fit_b["primary_post_onset_fir"], dtype=np.float64)
    )
    secondary = 0.5 * (
        np.asarray(fit_a["secondary_post_onset_fir"], dtype=np.float64)
        + np.asarray(fit_b["secondary_post_onset_fir"], dtype=np.float64)
    )
    frozen: dict[str, Any] = {
        "schema": "frozen_fit_only_joint_causal_candidate_v4",
        "source_role": "frozen_fit_only",
        "microphone_role": fit_a["microphone_role"],
        "submitted_pcm_sha256": fit_a["submitted_pcm_sha256"],
        "control_band_contract_sha256": fit_a[
            "control_band_contract_sha256"
        ],
        "control_subbands_hz": fit_a["control_subbands_hz"],
        "fit_candidate_sha256": [
            fit_a["candidate_sha256"],
            fit_b["candidate_sha256"],
        ],
        "delays_samples": list(fit_a["delays_samples"]),
        "coarse_integer_delays_samples": list(fit_a["delays_samples"]),
        "fractional_delay_residual_samples": list(
            fit_a["fractional_delay_residual_samples"]
        ),
        "bulk_delay_samples_fractional": list(
            fit_a["bulk_delay_samples_fractional"]
        ),
        "support_samples": int(fit_a["support_samples"]),
        "primary_post_onset_fir": primary.tolist(),
        "secondary_post_onset_fir": secondary.tolist(),
        "considered_supports": considered,
        "selected_fit_transfer_stationarity": next(
            row["fit_transfer_stationarity"]
            for row in considered
            if row["support_samples"] == int(fit_a["support_samples"])
        ),
        "holdout_used_for_generation_or_selection": False,
        "holdout_can_change_selected_support": False,
        "canonical_training_eligible": False,
        "canonical_blocker": CANONICAL_BLOCKER,
    }
    frozen["freeze_sha256"] = _json_sha256(frozen)
    return frozen


def terminal_holdout_receipt_v4(
    *, frozen: Mapping[str, Any], holdout_score: Mapping[str, Any]
) -> dict[str, Any]:
    microphone_role = str(frozen.get("microphone_role"))
    _validate_score_receipt_v4(
        holdout_score,
        candidate=frozen,
        target_role="holdout",
        microphone_role=microphone_role,
        require_pass=False,
    )
    residual = float(holdout_score["global_residual_ratio"])
    payload = {
        "schema": "terminal_holdout_validation_v4",
        "freeze_sha256": frozen["freeze_sha256"],
        "microphone_role": microphone_role,
        "selected_support_samples": int(frozen["support_samples"]),
        "holdout_residual_ratio": residual,
        "threshold": HOLDOUT_RESIDUAL_MAX,
        "holdout_score_receipt_sha256": holdout_score["receipt_sha256"],
        "subband_rows": holdout_score["subband_rows"],
        "all_subbands_passed": bool(holdout_score["all_subbands_passed"]),
        "passed": bool(
            residual <= HOLDOUT_RESIDUAL_MAX
            and bool(holdout_score["all_subbands_passed"])
            and bool(holdout_score["passed"])
        ),
        "support_reselection_after_holdout_forbidden": True,
        "canonical_training_eligible": False,
        "canonical_blocker": CANONICAL_BLOCKER,
    }
    payload["receipt_sha256"] = _json_sha256(payload)
    return payload


__all__ = [
    "CANONICAL_BLOCKER",
    "CLOCK_COMBINED_MAX",
    "CLOCK_CUBIC_MAX",
    "CLOCK_HARD_MAX",
    "CLOCK_LEAVEOUT_MAX",
    "CLOCK_VIEW_DISAGREEMENT_MAX",
    "CONDITION_AUDIT_SUPPORT",
    "FS",
    "EXACT_ZERO_DFT_MAX",
    "IMMUTABLE_JSON_ARTIFACT_REFERENCE_SCHEMA",
    "IDENTIFIABILITY_LIMITATION",
    "LIVE_AUTHORITY",
    "MAX_CONDITION",
    "MAX_DELAY",
    "OPERATOR_NPZ_SCHEMA",
    "OPERATOR_REFERENCE_SCHEMA",
    "PERIOD",
    "PILOT_BAND",
    "SUPPORTS",
    "SUBBAND_MAX_RELATIVE_ERROR",
    "SUBBAND_MIN_COMPLEX_AGREEMENT",
    "SUBBAND_MIN_EXACT_ZERO_NOISE_BINS",
    "SUBBAND_MIN_INPUT_RMS_DBFS",
    "SUBBAND_MIN_RESPONSE_BINS",
    "SUBBAND_MIN_TARGET_RMS_DBFS",
    "SUBBAND_MIN_TARGET_TO_NOISE_DB",
    "TRAINING_AUTHORITY_SCHEMA",
    "TRAINING_AUTHORITY_ENVELOPE_SCHEMA",
    "absolute_dac_q_timewarp_v4",
    "acoustic_clock_plant_confounding_counterexample",
    "build_plan",
    "continuous_pilot_period",
    "generate_fit_candidate_v4",
    "marker_branch_v4",
    "periodic_excitation_condition_audit_v4",
    "score_candidate_on_role_v4",
    "select_and_freeze_fit_support_v4",
    "terminal_holdout_receipt_v4",
]
