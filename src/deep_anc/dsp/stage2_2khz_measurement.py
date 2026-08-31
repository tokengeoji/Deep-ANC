"""Stage-2 2 kHz same-capture P/S 측정의 signal-only 계약.

이 모듈은 오디오 backend를 import하지 않고 장치를 열지 않는다. 24초 actual-int16
NS/CS 자극, fit-a/fit-b/untouched-holdout 분리, immutable lineage 및 측정 결과의
fail-closed 수치 검증만 담당한다. 현재 Jetson의 USB DAC 2-out/1-in + APE 2-in
비동기 topology는 absolute hardware-frame clock authority를 만들 수 없다. 다만 이
Stage-2의 목적은 single-point P/S 상대 plant와 lead이므로, submitted aperiodic code의
shared-q cross-fit과 nonaffine/change-point reject를 통과하면 조건부 relative authority는
물리적으로 가능하다. 실제 live adapter와 offline analyzer는 별도 모듈에서 이 plan만
소비하며, plan builder 자체는 계속 무음이다.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from deep_anc.data.repository_fd import publish_repository_bytes_noreplace
from deep_anc.dsp.measurement_level import expected_meter_output_pcm
from deep_anc.dsp.stage2_2khz_contract import (
    STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ,
    Stage2TwoKilohertzContract,
)


STAGE2_MEASUREMENT_CONFIG_SCHEMA = "stage2_2khz_same_capture_config_v1"
STAGE2_MEASUREMENT_PLAN_SCHEMA = "stage2_2khz_same_capture_ps_plan_v1"
STAGE2_MEASUREMENT_RESULT_SCHEMA = "stage2_2khz_same_capture_ps_result_v1"
STAGE2_RAW_SCHEMA = "stage2_2khz_same_capture_ps_raw_v1"

SAMPLE_RATE = 48_000
BLOCK_SIZE = 256
SIGNAL_SECONDS = 24.0
METER_SECONDS = 20.0
TOTAL_AUDIBLE_SECONDS = SIGNAL_SECONDS + METER_SECONDS
ROLE_SECONDS = 8.0
ROLE_NAMES = ("fit_a", "fit_b", "untouched_holdout")
PLAYBACK_ROLES = ("NS", "CS")
ROLE_SEEDS = {
    "fit_a": {"NS": 17_401, "CS": 17_402},
    "fit_b": {"NS": 27_401, "CS": 27_402},
    "untouched_holdout": {"NS": 37_401, "CS": 37_402},
}
SIGNAL_PEAK_PCM = 79
CREST_CLIP_RMS = 2.15
MIN_SUBBAND_CONSISTENCY = 0.95
MIN_RESPONSE_TO_NOISE_DB = 20.0
MAX_TIMING_RESIDUAL_SAMPLES = 0.270208
METER_RELATIVE_POWER_MIN_DB = -0.25
METER_RELATIVE_POWER_MAX_DB = 0.0

CURRENT_TOPOLOGY_STATUS = "CONDITIONAL_RELATIVE_PS_PHYSICALLY_IDENTIFIABLE"
CURRENT_LIVE_BLOCK_CODE = "BLOCKED_UNTIL_CLEAN_COMMIT_AND_EXPLICIT_CONFIRMATIONS"
CURRENT_LIVE_BLOCK_REASON = (
    "reviewed adapter는 있으나 clean exact commit, fresh no-replace targets, 무점유/마이크 "
    "preflight 및 세 물리 확인이 모두 PASS하기 전에는 audio backend를 import하지 않는다."
)
ABSOLUTE_CLOCK_LIMITATION = (
    "AB13X mono async capture는 APE ERR/REF와 같은 frame clock이 아니므로 absolute "
    "DAC/ADC frame identity, callback-before-start 및 hardware-counter slip=0은 주장할 "
    "수 없다. 이는 shared-q cross-fit으로 얻는 single-point P/S 상대지연과 분리한다."
)
MINIMAL_HARDWARE_ALTERNATIVES = (
    {
        "id": "ape_duplex_shared_frame_identity",
        "requirement": (
            "NS/CS 출력을 APE I2S1/RT5640으로 보내고 APE I2S2 ERR/REF capture와 "
            "동일 hardware frame/clock임을 독립 증거로 봉인한다"
        ),
        "additional_capture_channels": 0,
    },
    {
        "id": "synchronous_third_electrical_tap",
        "requirement": (
            "USB DAC 출력을 유지하되 ERR/REF와 같은 capture clock의 세 번째 "
            "전기 playback-clock tap을 추가한다"
        ),
        "additional_capture_channels": 1,
    },
)


class Stage2MeasurementError(ValueError):
    """Stage-2 측정 계약 위반."""


class Stage2TopologyBlockedError(RuntimeError):
    """현재 물리 topology로 canonical authority를 만들 수 없음."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _safe_relative_path(value: Any, *, suffix: str, label: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise Stage2MeasurementError(f"{label}는 repository 상대경로여야 합니다")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0] != "results"
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
        or path.suffix != suffix
    ):
        raise Stage2MeasurementError(
            f"{label}는 results/ 아래 canonical {suffix} 상대경로여야 합니다"
        )
    return value


def validate_measurement_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """현재 read-only 하드웨어 inventory를 보수적으로 검증한다."""

    if not isinstance(config, Mapping):
        raise Stage2MeasurementError("Stage-2 config root는 mapping이어야 합니다")
    parsed = json.loads(json.dumps(config, ensure_ascii=False, allow_nan=False))
    if parsed.get("schema") != STAGE2_MEASUREMENT_CONFIG_SCHEMA:
        raise Stage2MeasurementError("Stage-2 measurement config schema가 다릅니다")
    if parsed.get("sample_rate_hz") != SAMPLE_RATE or parsed.get("block_size") != BLOCK_SIZE:
        raise Stage2MeasurementError("Stage-2는 exact 48 kHz/256 frame이어야 합니다")
    duration = parsed.get("duration_seconds")
    if duration != {"meter": METER_SECONDS, "signal": SIGNAL_SECONDS}:
        raise Stage2MeasurementError("Stage-2 audible time은 meter20s+signal24s exact여야 합니다")

    playback = parsed.get("playback")
    capture = parsed.get("capture")
    usb_capture = parsed.get("usb_capture")
    if not isinstance(playback, dict) or (
        playback.get("card"),
        playback.get("device"),
        playback.get("channels"),
        playback.get("format"),
        playback.get("clock_mode"),
    ) != ("Audio", 0, 2, "S16_LE", "adaptive"):
        raise Stage2MeasurementError("현행 AB13X 2ch adaptive playback inventory와 다릅니다")
    if not isinstance(capture, dict) or (
        capture.get("card"),
        capture.get("device"),
        capture.get("channels"),
        capture.get("format"),
        tuple(capture.get("roles", ())),
    ) != ("APE", 1, 2, "S32_LE", ("ERR", "REF")):
        raise Stage2MeasurementError("현행 APE ERR/REF 2ch capture inventory와 다릅니다")
    if not isinstance(usb_capture, dict) or (
        usb_capture.get("card"),
        usb_capture.get("device"),
        usb_capture.get("channels"),
        usb_capture.get("format"),
        usb_capture.get("clock_mode"),
    ) != ("Audio", 0, 1, "S16_LE", "async"):
        raise Stage2MeasurementError("현행 AB13X mono async capture inventory와 다릅니다")

    evidence = parsed.get("independent_clock_evidence")
    if not isinstance(evidence, dict):
        raise Stage2MeasurementError("independent clock evidence 선언이 없습니다")
    if evidence != {
        "capture_witness_channels": 0,
        "shared_hardware_frame_identity_proven": False,
        "synchronous_electrical_tap_present": False,
    }:
        raise Stage2MeasurementError(
            "현재 topology config에서 존재하지 않는 clock witness를 선언할 수 없습니다"
        )
    if parsed.get("live_capture_enabled") is not False:
        raise Stage2MeasurementError("invalid v1 signal의 Stage-2 live config는 false여야 합니다")
    if parsed.get("canonical_training_eligible") is not False:
        raise Stage2MeasurementError("현재 topology는 canonical training eligible이 아닙니다")
    if parsed.get("canonical_training_ineligible_reason") != (
        "stage2_v2_physical_integration_missing"
    ):
        raise Stage2MeasurementError(
            "training BLOCK 이유는 absolute clock 부재가 아니라 측정 artifact 미발행이어야 합니다"
        )
    if parsed.get("authority_scope") != {
        "absolute_hardware_frame_clock": False,
        "single_point_relative_ps_lead_physically_identifiable": True,
        "spatial_quiet_zone": False,
    }:
        raise Stage2MeasurementError("Stage-2 authority scope가 상대 P/S 범위를 벗어났습니다")

    artifacts = parsed.get("artifacts")
    if not isinstance(artifacts, dict):
        raise Stage2MeasurementError("Stage-2 artifact 경로가 없습니다")
    _safe_relative_path(artifacts.get("plan"), suffix=".json", label="plan path")
    _safe_relative_path(artifacts.get("meter_raw"), suffix=".npz", label="meter raw path")
    _safe_relative_path(
        artifacts.get("native_raw_capture"), suffix=".npz", label="native raw path"
    )
    _safe_relative_path(artifacts.get("raw_capture"), suffix=".npz", label="canonical raw path")
    _safe_relative_path(artifacts.get("analysis"), suffix=".npz", label="analysis path")
    _safe_relative_path(
        artifacts.get("analysis_receipt"), suffix=".json", label="analysis receipt path"
    )
    _safe_relative_path(
        artifacts.get("analysis_arrays"), suffix=".npz", label="analysis arrays path"
    )
    _safe_relative_path(
        artifacts.get("relative_clock_receipt"),
        suffix=".json",
        label="relative clock receipt path",
    )
    _safe_relative_path(
        artifacts.get("primary_candidate"), suffix=".npz", label="primary candidate path"
    )
    _safe_relative_path(
        artifacts.get("secondary_candidate"), suffix=".npz", label="secondary candidate path"
    )
    _safe_relative_path(
        artifacts.get("measurement_level_evidence"),
        suffix=".json",
        label="measurement level evidence path",
    )
    _safe_relative_path(
        artifacts.get("plant_binding"), suffix=".json", label="plant binding path"
    )
    return parsed


def topology_assessment(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_measurement_config(config)
    return {
        "status": CURRENT_TOPOLOGY_STATUS,
        "relative_ps_lead_authority_physically_possible": True,
        "fixed_lti_assumption_required": True,
        "common_time_gauge_cancels_in_relative_delay": True,
        "playback_to_err_acoustic_delay_must_remain_in_each_plant": True,
        "absolute_hardware_frame_clock_authority_available": False,
        "absolute_clock_limitation": ABSOLUTE_CLOCK_LIMITATION,
        "live_capture_adapter_available": True,
        "live_status": CURRENT_LIVE_BLOCK_CODE,
        "live_block_reason": CURRENT_LIVE_BLOCK_REASON,
        "canonical_live_authority_available": False,
        "acoustic_shared_q_allowed_for_relative_ps_only": True,
        "legacy_v3_v5_v6_promotion_allowed": False,
        "diagnostic_raw_promotion_allowed": False,
        "optional_absolute_clock_upgrade_paths": list(MINIMAL_HARDWARE_ALTERNATIVES),
    }


def _band_equalized_aperiodic_code(seed: int, frames: int) -> np.ndarray:
    """반복 슬롯 없이 6구간 에너지를 균등화한 deterministic code를 만든다."""

    frequencies = np.fft.rfftfreq(frames, 1.0 / SAMPLE_RATE)
    spectrum = np.zeros(frequencies.size, dtype=np.complex128)
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    excitation_bands = (
        (80.0, STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ[0][1]),
        *STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ[1:],
    )
    for lower, upper in excitation_bands:
        indices = np.flatnonzero((frequencies >= lower) & (frequencies < upper))
        if indices.size == 0:
            raise AssertionError("Stage-2 code band에 FFT bin이 없습니다")
        phases = generator.uniform(0.0, 2.0 * np.pi, size=indices.size)
        spectrum[indices] = np.exp(1j * phases) / math.sqrt(float(indices.size))
    signal = np.fft.irfft(spectrum, n=frames)
    rms = float(np.sqrt(np.mean(signal**2)))
    if not math.isfinite(rms) or rms <= 0.0:
        raise AssertionError("Stage-2 aperiodic code RMS가 유효하지 않습니다")
    signal = signal / rms
    signal = np.clip(signal, -CREST_CLIP_RMS, CREST_CLIP_RMS)
    signal = signal / float(np.sqrt(np.mean(signal**2)))
    return signal


def _role_spectral_receipt(value: np.ndarray) -> dict[str, Any]:
    signal = np.asarray(value, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    frequencies = np.fft.rfftfreq(signal.size, 1.0 / SAMPLE_RATE)
    energies: list[float] = []
    for lower, upper in STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ:
        mask = (frequencies >= lower) & (frequencies < upper)
        energies.append(float(np.sum(spectrum[mask])))
    total = float(sum(energies))
    if total <= 0.0 or not all(math.isfinite(value) and value > 0.0 for value in energies):
        raise AssertionError("Stage-2 code의 6구간 energy가 모두 양수여야 합니다")
    fractions = [value / total for value in energies]
    if min(fractions) < 0.12:
        raise AssertionError("Stage-2 code가 6구간 중 하나를 충분히 자극하지 않습니다")
    excitation_mask = (frequencies >= 80.0) & (frequencies < 2828.4271247462)
    lower_edge_mask = (frequencies >= 80.0) & (frequencies < 88.3883476483)
    excitation_energy = float(np.sum(spectrum[excitation_mask]))
    lower_edge_fraction = float(np.sum(spectrum[lower_edge_mask]) / excitation_energy)
    if not math.isfinite(lower_edge_fraction) or lower_edge_fraction < 0.005:
        raise AssertionError("Stage-2 code가 required excitation 80..88.388 Hz를 자극하지 않습니다")
    lags = (256, 1_024, 4_096, 48_000)
    correlations: dict[str, float] = {}
    for lag in lags:
        left = signal[:-lag]
        right = signal[lag:]
        denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
        correlations[str(lag)] = float(np.dot(left, right) / denominator)
    if max(abs(value) for value in correlations.values()) >= 0.10:
        raise AssertionError("Stage-2 code에 짧은 반복/comb 상관이 있습니다")
    return {
        "subband_energy_fraction": fractions,
        "required_excitation_80_88_388_energy_fraction": lower_edge_fraction,
        "selected_lag_normalized_autocorrelation": correlations,
        "repeated_slot_count": 0,
    }


@lru_cache(maxsize=1)
def _canonical_signal_and_roles() -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    role_frames = int(round(ROLE_SECONDS * SAMPLE_RATE))
    float_roles: list[tuple[str, np.ndarray]] = []
    global_peak = 0.0
    for role in ROLE_NAMES:
        channels = np.column_stack(
            [
                _band_equalized_aperiodic_code(ROLE_SEEDS[role][name], role_frames)
                for name in PLAYBACK_ROLES
            ]
        )
        float_roles.append((role, channels))
        global_peak = max(global_peak, float(np.max(np.abs(channels))))
    if global_peak <= 0.0:
        raise AssertionError("Stage-2 signal peak가 0입니다")

    rendered: list[np.ndarray] = []
    layout: list[dict[str, Any]] = []
    cursor = 0
    for role, channels in float_roles:
        pcm = np.rint(channels * (SIGNAL_PEAK_PCM / global_peak)).astype(np.int16)
        if pcm.shape != (role_frames, 2):
            raise AssertionError("Stage-2 role shape가 잘못됐습니다")
        if any(np.count_nonzero(pcm[:, index]) < int(0.98 * role_frames) for index in (0, 1)):
            raise AssertionError("Stage-2 NS/CS code가 capture 전체에 연속하지 않습니다")
        cross = float(np.corrcoef(pcm[:, 0].astype(np.float64), pcm[:, 1].astype(np.float64))[0, 1])
        if not math.isfinite(cross) or abs(cross) >= 0.02:
            raise AssertionError("Stage-2 NS/CS code가 독립적이지 않습니다")
        channel_receipts: dict[str, Any] = {}
        for channel_index, channel_name in enumerate(PLAYBACK_ROLES):
            channel = pcm[:, channel_index]
            channel_receipts[channel_name] = {
                "seed": ROLE_SEEDS[role][channel_name],
                "actual_int16_sha256": _array_sha256(channel),
                "spectral_audit": _role_spectral_receipt(channel),
            }
        stop = cursor + role_frames
        layout.append(
            {
                "role": role,
                "start_frame": cursor,
                "stop_frame": stop,
                "frames": role_frames,
                "used_for_fit": role != "untouched_holdout",
                "used_for_support_selection": role != "untouched_holdout",
                "used_for_threshold_selection": False,
                "untouched_until_final_prediction": role == "untouched_holdout",
                "ns_cs_zero_lag_correlation": cross,
                "channels": channel_receipts,
                "interleaved_actual_int16_sha256": _array_sha256(pcm),
            }
        )
        rendered.append(pcm)
        cursor = stop
    submitted = np.ascontiguousarray(np.concatenate(rendered, axis=0), dtype=np.int16)
    submitted.setflags(write=False)
    return submitted, tuple(layout)


def build_stage2_measurement_plan(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    """현행 topology의 immutable signal plan과 actual submitted PCM을 만든다."""

    parsed = validate_measurement_config(config)
    contract = Stage2TwoKilohertzContract.canonical()
    signal_readonly, cached_role_layout = _canonical_signal_and_roles()
    signal_pcm = signal_readonly.copy()
    signal_frames = int(round(SIGNAL_SECONDS * SAMPLE_RATE))
    if signal_pcm.dtype != np.int16 or signal_pcm.shape != (signal_frames, 2):
        raise AssertionError("Stage-2 submitted PCM은 exact [1152000,2] int16이어야 합니다")
    if signal_frames % BLOCK_SIZE:
        raise AssertionError("Stage-2 signal은 256-frame 정렬이어야 합니다")
    signal_peak = int(np.max(np.abs(signal_pcm.astype(np.int32))))
    if signal_peak != SIGNAL_PEAK_PCM or signal_peak > 98:
        raise AssertionError("Stage-2 signal peak가 exact 79 PCM 또는 안전 상한 밖입니다")

    meter_pcm = expected_meter_output_pcm(noise_channel=0)
    meter_frames = int(round(METER_SECONDS * SAMPLE_RATE))
    if meter_pcm.dtype != np.int16 or meter_pcm.shape != (meter_frames, 2):
        raise AssertionError("공식 meter PCM이 exact [960000,2] int16이 아닙니다")
    # Meter는 level을 확인한 뒤에만 P/S 자극을 허용하기 위해 별도 stream/raw로
    # 보존한다. actual submitted P/S lineage는 연속 24초 signal 자체다.
    submitted = np.ascontiguousarray(signal_pcm, dtype=np.int16)
    total_output_frames = meter_frames + signal_frames
    if total_output_frames % BLOCK_SIZE:
        raise AssertionError("Stage-2 meter+signal output budget이 256 정렬이 아닙니다")
    actual_peak = int(np.max(np.abs(submitted.astype(np.int32))))

    meter = meter_pcm.astype(np.float64)
    meter_power = float(np.sum(np.mean((meter / 32768.0) ** 2, axis=0)))
    signal_power = float(np.sum(np.mean((signal_pcm.astype(np.float64) / 32768.0) ** 2, axis=0)))
    relative_db = 10.0 * math.log10(signal_power / meter_power)
    if not METER_RELATIVE_POWER_MIN_DB <= relative_db <= METER_RELATIVE_POWER_MAX_DB:
        raise AssertionError("Stage-2 signal power가 공식 meter 대비 -0.25..0 dB 밖입니다")

    assessment = topology_assessment(parsed)
    plan_without_digest: dict[str, Any] = {
        "schema": STAGE2_MEASUREMENT_PLAN_SCHEMA,
        "contract": {
            "id": contract.contract_id,
            "sha256": contract.digest(),
            "payload": contract.model_dump(mode="json"),
        },
        "measurement_config_sha256": _payload_sha256(parsed),
        "sample_rate_hz": SAMPLE_RATE,
        "block_size": BLOCK_SIZE,
        "signal_seconds": SIGNAL_SECONDS,
        "signal_frames": signal_frames,
        "meter_frames": meter_frames,
        "total_output_frames_across_two_streams": total_output_frames,
        "meter_seconds": METER_SECONDS,
        "maximum_total_audible_seconds": TOTAL_AUDIBLE_SECONDS,
        "playback_channels": list(PLAYBACK_ROLES),
        "signal_recipe": {
            "schema": "stage2_equal_subband_aperiodic_actual_int16_v1",
            "generator": "numpy_pcg64_independent_phase_irfft",
            "role_seconds": ROLE_SECONDS,
            "roles": list(ROLE_NAMES),
            "crest_clip_rms": CREST_CLIP_RMS,
            "actual_peak_pcm": SIGNAL_PEAK_PCM,
            "subbands_hz": [list(row) for row in STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ],
            "excitation_band_hz": [80.0, 2828.4271247462],
            "no_repeated_period_or_slot": True,
            "both_codes_present_for_entire_signal": True,
        },
        "role_layout": [
            {
                **json.loads(json.dumps(row, allow_nan=False)),
                "start_frame_in_capture": int(row["start_frame"]),
                "stop_frame_in_capture": int(row["stop_frame"]),
            }
            for row in cached_role_layout
        ],
        "meter_submitted_pcm": {
            "dtype": meter_pcm.dtype.str,
            "shape": list(meter_pcm.shape),
            "sha256": _array_sha256(meter_pcm),
            "peak_pcm": int(np.max(np.abs(meter_pcm.astype(np.int32)))),
            "stream_role": "official_level_gate_before_stage2_signal",
        },
        "actual_submitted_pcm": {
            "dtype": submitted.dtype.str,
            "shape": list(submitted.shape),
            "sha256": _array_sha256(submitted),
            "peak_pcm": actual_peak,
            "meter_relative_total_power_db": relative_db,
            "coverage": "exact_continuous_stage2_signal24s_after_separate_meter_pass",
        },
        "fit_policy": {
            "fit_a": "fit_only",
            "fit_b": "independent_refit_crosscheck",
            "untouched_holdout": "prediction_only_after_all_fit_and_support_selection",
            "holdout_transfer_fit_or_support_selection_allowed": False,
            "holdout_allowed_for_predeclared_shared_q_nuisance_likelihood": True,
        },
        "robustness_epoch_policy": {
            "epoch_frames": 48000,
            "total_nonoverlapping_epochs": 24,
            "minimum_kept_epochs_per_role": 8,
            "epochs_are_distinct_aperiodic_observations": True,
            "legacy_periodic_repeat_indices_applicable": False,
            "fake_repeat_index_synthesis_allowed": False,
        },
        "clock_witness_policy": {
            "required_coverage_frames": signal_frames,
            "continuous_no_gap": True,
            "kind": "submitted_aperiodic_shared_q_acoustic_likelihood",
            "selection_input": "full_capture_low_band_known_stereo_codes_to_err_ref",
            "selection_band_hz": [88.3883476483, 600.0],
            "single_shared_q_for_all_p_s_err_ref_views": True,
            "q_model": "single_affine_only",
            "q_search_bound_ppm": [-1000.0, 1000.0],
            "high_band_residual_may_repair_q": False,
            "independent_electrical_required_for_relative_ps_lead": False,
            "independent_electrical_required_for_absolute_frame_claim": True,
            "absolute_frame_claim_allowed": False,
            "maximum_timing_residual_samples": MAX_TIMING_RESIDUAL_SAMPLES,
        },
        "nonaffine_rejection_policy": {
            "transport_callback_contiguity_every_256_frames": True,
            "transport_semantics": "software_accounting_not_absolute_hardware_slip",
            "acoustic_q_epoch_frames": 96000,
            "test_every_acoustic_q_epoch_boundary": True,
            "claim_acoustic_q_at_every_256_boundary": False,
            "reject_any_change_point": True,
            "reject_any_one_sample_insert_drop": True,
            "reject_any_view_specific_q": True,
            "reject_search_boundary_optimum": True,
            "simplest_affine_only": True,
            "holdout_failure_may_refit": False,
        },
        "admission_thresholds": {
            "subbands_hz": [list(row) for row in STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ],
            "minimum_consistency_each_subband": MIN_SUBBAND_CONSISTENCY,
            "minimum_response_to_noise_db_each_subband": MIN_RESPONSE_TO_NOISE_DB,
            "maximum_timing_residual_samples": MAX_TIMING_RESIDUAL_SAMPLES,
            "xrun_clip_status_slip_drop_add_exact": 0,
            "threshold_relaxation_allowed": False,
        },
        "raw_policy": {
            "schema": STAGE2_RAW_SCHEMA,
            "publish": "held_dirfd_no_replace",
            "replace_allowed": False,
            "legacy_v3_v5_v6_or_diagnostic_promotion_allowed": False,
        },
        "artifacts": parsed["artifacts"],
        "topology_assessment": assessment,
    }
    plan = dict(plan_without_digest)
    plan["plan_sha256"] = _payload_sha256(plan_without_digest)
    return plan, submitted


def validate_submitted_pcm(plan: Mapping[str, Any], submitted_pcm: np.ndarray) -> None:
    submitted = np.asarray(submitted_pcm)
    lineage = plan.get("actual_submitted_pcm")
    if not isinstance(lineage, Mapping):
        raise Stage2MeasurementError("plan에 actual submitted PCM lineage가 없습니다")
    if submitted.dtype != np.int16 or list(submitted.shape) != lineage.get("shape"):
        raise Stage2MeasurementError("submitted PCM dtype/shape가 plan과 다릅니다")
    if _array_sha256(submitted) != lineage.get("sha256"):
        raise Stage2MeasurementError("submitted PCM bytes가 plan lineage와 다릅니다")
    plan_without_digest = dict(plan)
    digest = plan_without_digest.pop("plan_sha256", None)
    if digest != _payload_sha256(plan_without_digest):
        raise Stage2MeasurementError("Stage-2 plan SHA가 payload와 다릅니다")


def _exact_zero_counters(value: Any, *, label: str) -> None:
    expected = {
        "xrun": 0,
        "clip": 0,
        "callback_status": 0,
        "sample_slip": 0,
        "sample_drop": 0,
        "sample_add": 0,
    }
    if value != expected:
        raise Stage2MeasurementError(f"{label} counters는 모두 exact 0이어야 합니다")


def validate_stage2_metric_receipt(
    plan: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """이미 메모리에 있는 분석 receipt의 수치 계약만 검증한다.

    이 PASS는 현재 topology의 authority PASS가 아니다. topology admission은 별도이며
    이 함수만의 PASS는 self-attested 수치 PASS일 뿐이다. 실제 승격은 immutable native
    raw를 다시 읽어 계산한 analyzer 결과와 typed admission을 모두 요구한다.
    """

    if not isinstance(receipt, Mapping) or receipt.get("schema") != STAGE2_MEASUREMENT_RESULT_SCHEMA:
        raise Stage2MeasurementError("Stage-2 metric receipt schema가 다릅니다")
    if receipt.get("plan_sha256") != plan.get("plan_sha256"):
        raise Stage2MeasurementError("metric receipt가 exact plan에 결박되지 않았습니다")
    if receipt.get("actual_submitted_pcm_sha256") != plan.get("actual_submitted_pcm", {}).get("sha256"):
        raise Stage2MeasurementError("metric receipt의 actual submitted PCM SHA가 다릅니다")
    if receipt.get("same_capture_ps") is not True:
        raise Stage2MeasurementError("P/S는 같은 native capture여야 합니다")
    if receipt.get("raw_publish_mode") != "no_replace" or receipt.get("analysis_publish_mode") != "no_replace":
        raise Stage2MeasurementError("raw/analysis는 모두 no-replace여야 합니다")
    raw_sha = receipt.get("raw_capture_sha256")
    if type(raw_sha) is not str or len(raw_sha) != 64:
        raise Stage2MeasurementError("immutable raw capture SHA-256이 없습니다")
    _exact_zero_counters(receipt.get("counters"), label="capture")

    holdout = receipt.get("holdout_policy")
    if holdout != {
        "used_for_fit": False,
        "used_for_support_selection": False,
        "used_for_threshold_selection": False,
        "used_for_predeclared_shared_q_nuisance_likelihood": True,
        "evaluated_after_fit_frozen": True,
    }:
        raise Stage2MeasurementError("untouched holdout가 fit/selection에 사용됐습니다")

    witness = receipt.get("clock_witness")
    if not isinstance(witness, Mapping):
        raise Stage2MeasurementError("continuous clock witness가 없습니다")
    if witness.get("kind") != "submitted_aperiodic_shared_q_acoustic_likelihood":
        raise Stage2MeasurementError("Stage-2 relative P/S는 predeclared acoustic shared-q만 허용합니다")
    if witness.get("continuous_frames") != plan.get("signal_frames") or witness.get("gap_frames") != 0:
        raise Stage2MeasurementError("clock witness가 capture 전체에 연속하지 않습니다")
    if witness.get("single_shared_q_for_all_p_s_err_ref_views") is not True:
        raise Stage2MeasurementError("P/S×ERR/REF에 view별 q를 허용하지 않습니다")
    if witness.get("selection_input") != "full_capture_low_band_known_stereo_codes_to_err_ref":
        raise Stage2MeasurementError("shared q selection input이 봉인된 low-band likelihood와 다릅니다")
    if witness.get("selection_band_hz") != [88.3883476483, 600.0]:
        raise Stage2MeasurementError("shared q는 exact 88.388..600 Hz에서만 선선택해야 합니다")
    if witness.get("q_model") != "single_affine" or witness.get("search_boundary_optimum") is not False:
        raise Stage2MeasurementError("shared q는 내부 optimum인 single affine이어야 합니다")
    ambiguity = witness.get("ambiguity_envelope_validation_samples")
    if (
        type(ambiguity) not in {int, float}
        or not math.isfinite(float(ambiguity))
        or float(ambiguity) > MAX_TIMING_RESIDUAL_SAMPLES
    ):
        raise Stage2MeasurementError("shared q ambiguity envelope가 0.270208 sample을 초과합니다")
    residual = witness.get("maximum_timing_residual_samples")
    if type(residual) not in {int, float} or not math.isfinite(float(residual)) or float(residual) > MAX_TIMING_RESIDUAL_SAMPLES:
        raise Stage2MeasurementError("timing residual이 0.270208 sample을 초과합니다")
    _exact_zero_counters(witness.get("counters"), label="clock witness")

    absolute_claims = receipt.get("absolute_transport_claims")
    if absolute_claims != {
        "absolute_hardware_frame_identity_claimed": False,
        "callback_before_start_drop_observed_claimed": False,
        "hardware_counter_slip_zero_claimed": False,
        "relative_ps_lead_only": True,
    }:
        raise Stage2MeasurementError("독립 전기 증거 없는 absolute transport 주장을 허용하지 않습니다")

    nonaffine = receipt.get("nonaffine_change_point_audit")
    if nonaffine != {
        "transport_256_callback_contiguity_tested": True,
        "transport_semantics": "software_accounting_not_absolute_hardware_slip",
        "acoustic_q_epoch_frames": 96000,
        "all_acoustic_q_epoch_boundaries_tested": True,
        "all_256_frame_acoustic_q_boundaries_tested": False,
        "change_point_detected": False,
        "nonaffine_drift_detected": False,
        "one_sample_insert_drop_detected": False,
        "view_specific_q_detected": False,
        "affine_model_frozen_before_holdout": True,
        "holdout_failure_refit_performed": False,
    }:
        raise Stage2MeasurementError("nonaffine/change-point/drop audit가 fail-closed PASS가 아닙니다")

    relative_delay = receipt.get("relative_delay_scope")
    if relative_delay != {
        "playback_to_err_acoustic_delay_included_in_primary": True,
        "playback_to_err_acoustic_delay_included_in_secondary": True,
        "common_intercept_claimed_separately": False,
        "common_time_gauge_cancels_in_p_minus_s": True,
        "manual_lead_allowed": False,
    }:
        raise Stage2MeasurementError("P/S 상대 delay gauge 또는 acoustic onset scope가 잘못됐습니다")

    rows = receipt.get("path_subbands")
    if not isinstance(rows, list) or len(rows) != 24:
        raise Stage2MeasurementError("P/S × ERR/REF × 6구간 exact 24개 metric row가 필요합니다")
    expected_keys = {
        (path, microphone, index)
        for path in ("primary", "secondary")
        for microphone in ("ERR", "REF")
        for index in range(len(STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ))
    }
    observed_keys: set[tuple[str, str, int]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise Stage2MeasurementError("subband metric row는 mapping이어야 합니다")
        key = (
            str(row.get("path")),
            str(row.get("microphone")),
            int(row.get("subband_index", -1)),
        )
        if key not in expected_keys or key in observed_keys:
            raise Stage2MeasurementError("P/S × 6구간 metric row key가 중복/누락됐습니다")
        observed_keys.add(key)
        expected_band = list(STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ[key[2]])
        if row.get("band_hz") != expected_band:
            raise Stage2MeasurementError("subband 경계가 Stage-2 exact 6구간과 다릅니다")
        for field in ("fit_a_fit_b_consistency", "untouched_holdout_consistency"):
            value = row.get(field)
            if type(value) not in {int, float} or not math.isfinite(float(value)) or float(value) < MIN_SUBBAND_CONSISTENCY:
                raise Stage2MeasurementError(f"{field}가 0.95 미만입니다")
        snr = row.get("response_to_noise_db")
        if type(snr) not in {int, float} or not math.isfinite(float(snr)) or float(snr) < MIN_RESPONSE_TO_NOISE_DB:
            raise Stage2MeasurementError("subband response-to-noise가 20 dB 미만입니다")
    if observed_keys != expected_keys:
        raise Stage2MeasurementError("P/S × 6구간 metric row가 완전하지 않습니다")
    if receipt.get("thresholds_relaxed") is not False:
        raise Stage2MeasurementError("Stage-2 threshold 완화는 금지됩니다")
    return {
        "metric_contract_pass": True,
        "conditional_relative_ps_measurement_pass": True,
        "absolute_hardware_frame_clock_authority_pass": False,
        "authority_scope": "single_point_relative_ps_lead_only",
    }


def admit_stage2_relative_ps_candidate(
    plan: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """reviewed live adapter 결과를 relative P/S training 후보로 typed admission한다.

    electrical absolute-frame 증거를 요구하지 않는 대신 synthetic/diagnostic receipt의
    승격과 범위 확대를 명시적으로 막는다. caller가 만든 self-attested receipt는 production
    entrypoint가 아니며, offline analyzer는 immutable raw arrays에서 이 payload를 재계산한다.
    """

    metric = validate_stage2_metric_receipt(plan, receipt)
    generation = receipt.get("capture_generation")
    if generation != {
        "adapter_schema": "stage2_2khz_live_capture_adapter_v1",
        "reviewed_live_adapter_implemented": True,
        "physical_acoustic_capture": True,
        "synthetic_or_diagnostic": False,
        "clean_exact_commit": True,
        "native_raw_published_no_replace": True,
    }:
        raise Stage2MeasurementError(
            "reviewed live adapter의 immutable physical raw가 아니므로 relative P/S 승격 금지"
        )
    return {
        **metric,
        "relative_ps_training_plant_candidate": True,
        "absolute_hardware_frame_clock_authority_pass": False,
        "spatial_quiet_zone_claim_allowed": False,
        "automatic_training_config_update_allowed": False,
        "required_next_step": "independent_review_then_explicit_stage2_duct_binding",
    }


def publish_plan_no_replace(
    repository_root: str, relative_path: str, plan: Mapping[str, Any]
) -> dict[str, Any]:
    target = _safe_relative_path(relative_path, suffix=".json", label="plan path")
    if target != plan.get("artifacts", {}).get("plan"):
        raise Stage2MeasurementError("plan publish target이 sealed artifact path와 다릅니다")
    return publish_repository_bytes_noreplace(
        repository_root,
        target,
        _canonical_json_bytes(dict(plan)),
        mode=0o600,
        recovery_tag="stage2_plan",
    )
__all__ = [
    "BLOCK_SIZE",
    "ABSOLUTE_CLOCK_LIMITATION",
    "CURRENT_LIVE_BLOCK_CODE",
    "CURRENT_LIVE_BLOCK_REASON",
    "CURRENT_TOPOLOGY_STATUS",
    "MAX_TIMING_RESIDUAL_SAMPLES",
    "METER_SECONDS",
    "MINIMAL_HARDWARE_ALTERNATIVES",
    "MIN_RESPONSE_TO_NOISE_DB",
    "MIN_SUBBAND_CONSISTENCY",
    "SAMPLE_RATE",
    "SIGNAL_SECONDS",
    "STAGE2_MEASUREMENT_CONFIG_SCHEMA",
    "STAGE2_MEASUREMENT_PLAN_SCHEMA",
    "STAGE2_MEASUREMENT_RESULT_SCHEMA",
    "STAGE2_RAW_SCHEMA",
    "Stage2MeasurementError",
    "Stage2TopologyBlockedError",
    "build_stage2_measurement_plan",
    "admit_stage2_relative_ps_candidate",
    "publish_plan_no_replace",
    "topology_assessment",
    "validate_measurement_config",
    "validate_stage2_metric_receipt",
    "validate_submitted_pcm",
]
