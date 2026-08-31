"""물리 gain cap에 맞춘 Stage-1 녹음 source 파생 계약.

이 모듈은 오디오 장치를 열지 않는다. 원본 source를 수정하지 않고 exact 15초
coverage-training stimulus를 파생한 뒤, 실제 ``NoiseProgram`` renderer와 strict P를
사용해 source preflight/SNR/peak/RMS를 다시 계산한다.

파생물은 자연음을 그대로 평가한 증거가 아니다. 최종 unseen/natural 평가는 반드시
원본 미가공 source와 별도 physical OFF/ON raw로 수행해야 한다.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import signal

from deep_anc.data.recording_source_preflight import (
    SOURCE_PREFLIGHT_CONSERVATIVE_QUIET_FLOOR_DBFS,
    SOURCE_PREFLIGHT_FRAMES,
    SOURCE_PREFLIGHT_REQUIRED_COHERENCE_SQUARED,
    SOURCE_PREFLIGHT_SAMPLE_RATE,
    rendered_source_preflight,
    validate_timeline_source_feasibility,
)
from deep_anc.realtime.noise_gen import NoiseProgram, render_recording_file_window


CONDITIONING_RECIPE_SCHEMA = "recording_source_conditioning_recipe/v1_cap_aware_4band"
CONDITIONING_RECEIPT_SCHEMA = "recording_source_conditioning_receipt/v1"
CONDITIONING_CAMPAIGN_SCHEMA = "recording_source_conditioning_campaign/v1"
CONDITIONING_ROLE = (
    "coverage_training_stimulus_not_natural_unprocessed_evaluation_evidence"
)
CONDITIONING_ALGORITHM_VERSION = "cap-aware-active100ms-fourband-fir513-leveler-v1"
CAP_AWARE_PREFLIGHT_SCHEMA = "recording_source_cap_aware_preflight/v1"

CONDITIONING_BANDS_HZ = (
    (150.0, 300.0),
    (300.0, 600.0),
    (600.0, 1000.0),
    (1000.0, 1600.0),
)
STRICT_PRIMARY_REQUIRED_BANDS_HZ = ((150.0, 600.0), (600.0, 1600.0))

PCM16_LSB = 1.0 / 32768.0
MINIMUM_ORIGIN_BAND_RMS_LINEAR = 16.0 * PCM16_LSB
MINIMUM_ORIGIN_BAND_RMS_DBFS = 20.0 * math.log10(
    MINIMUM_ORIGIN_BAND_RMS_LINEAR
)
ACTIVE_BLOCK_SAMPLES = 4_800
ACTIVE_RELATIVE_FLOOR_DB = -36.0
EQ_FIR_TAPS = 513
EQ_MAX_GAIN_DB = 6.0
LEVELER_MAX_DRIVE = 6.0
OUTPUT_PEAK = 0.98
ADC_PEAK_HARD_CEILING = 0.5
ADC_RMS_HARD_CEILING = 0.5

# 비용과 과처리를 동시에 제한하는 결정론적 최소 후보 집합이다. 먼저 원본을
# 검사하고, 실패할 때만 active compaction/leveling, 마지막에 작은 4-band EQ를 쓴다.
_LEVELER_DRIVES = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)
_EQ_GAIN_CANDIDATES_DB = (
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 3.0, 3.0),
    (0.0, 0.0, 3.0, 6.0),
    (0.0, 0.0, 6.0, 3.0),
    (0.0, 0.0, 6.0, 6.0),
    (3.0, 3.0, 0.0, 0.0),
)


class RecordingSourceConditioningError(ValueError):
    """cap-aware 파생 source 계약 위반."""


@dataclass(frozen=True)
class ConditionedSourceResult:
    """JSON receipt와 exact 파생 WAV bytes를 함께 보존한다."""

    receipt: dict[str, Any]
    wav_bytes: bytes | None


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seal(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordingSourceConditioningError(f"{label}는 finite 숫자여야 합니다")
    result = float(value)
    if not math.isfinite(result):
        raise RecordingSourceConditioningError(f"{label}는 finite 숫자여야 합니다")
    return result


def _decode_source_window(
    source_bytes: bytes, *, start_seconds: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """NoiseProgram과 같은 whole-file peak 기준으로 exact 15초를 만든다."""

    start = _finite(start_seconds, label="source start_seconds")
    if start < 0.0:
        raise RecordingSourceConditioningError("source start_seconds는 0 이상이어야 합니다")
    import soundfile as sf

    try:
        values, sample_rate = sf.read(
            io.BytesIO(source_bytes), dtype="float32", always_2d=True
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RecordingSourceConditioningError(f"source audio decode 실패: {exc}") from exc
    mono = np.asarray(values.mean(axis=1), dtype=np.float64)
    if sample_rate != SOURCE_PREFLIGHT_SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), SOURCE_PREFLIGHT_SAMPLE_RATE)
        mono = signal.resample_poly(
            mono,
            SOURCE_PREFLIGHT_SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        )
    if mono.size < 1 or not bool(np.isfinite(mono).all()):
        raise RecordingSourceConditioningError("source audio가 비었거나 non-finite입니다")
    whole_peak = float(np.max(np.abs(mono)))
    if whole_peak < PCM16_LSB:
        raise RecordingSourceConditioningError("source 전체 peak가 PCM16 1 LSB 미만입니다")
    start_frame = int(round(start * SOURCE_PREFLIGHT_SAMPLE_RATE))
    if start_frame >= mono.size:
        raise RecordingSourceConditioningError("source start가 파일 길이 밖입니다")
    tail = mono[start_frame:] / whole_peak
    repeats = int(math.ceil(SOURCE_PREFLIGHT_FRAMES / tail.size))
    window = np.tile(tail, repeats)[:SOURCE_PREFLIGHT_FRAMES]
    canonical = np.ascontiguousarray(window, dtype="<f4")
    return canonical, {
        "decoded_sample_rate": int(sample_rate),
        "canonical_sample_rate": SOURCE_PREFLIGHT_SAMPLE_RATE,
        "whole_file_peak_linear": whole_peak,
        "start_seconds": start,
        "tail_frames_after_start": int(tail.size),
        "tail_repeat_count": repeats,
        "frames": SOURCE_PREFLIGHT_FRAMES,
        "float32_le_sha256": _sha256_bytes(canonical.tobytes()),
    }


def _band_rms_dbfs(
    samples: np.ndarray, bands: Sequence[tuple[float, float]]
) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    window = np.hanning(values.size)
    spectrum = np.fft.rfft(values * window)
    frequencies = np.fft.rfftfreq(
        values.size, 1.0 / SOURCE_PREFLIGHT_SAMPLE_RATE
    )
    denominator = values.size * float(np.sum(window**2))
    result: dict[str, float] = {}
    for low, high in bands:
        selected = (frequencies >= low) & (frequencies <= high)
        power = 2.0 * float(np.sum(np.abs(spectrum[selected]) ** 2)) / denominator
        result[f"{int(low)}_{int(high)}"] = 10.0 * math.log10(max(power, 1e-24))
    return result


def origin_band_admission(samples: np.ndarray) -> dict[str, Any]:
    """원본 4-band가 quantization 근처인지 EQ 전에 판정한다."""

    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    peak = float(np.max(np.abs(values)))
    if values.size != SOURCE_PREFLIGHT_FRAMES or not peak > 0.0:
        raise RecordingSourceConditioningError("origin window shape/peak 계약 위반")
    normalized = values / peak
    levels = _band_rms_dbfs(normalized, CONDITIONING_BANDS_HZ)
    failed = [
        band for band, level in levels.items() if level < MINIMUM_ORIGIN_BAND_RMS_DBFS
    ]
    return {
        "schema": "recording_source_origin_band_admission/v1",
        "bands_hz": [list(value) for value in CONDITIONING_BANDS_HZ],
        "metric": "hann_rfft_rms_after_origin_window_peak_normalization_v1",
        "minimum_band_rms_dbfs": MINIMUM_ORIGIN_BAND_RMS_DBFS,
        "minimum_definition": "16_pcm16_lsb_rms_after_window_peak_normalization",
        "band_rms_dbfs": levels,
        "failed_bands": failed,
        "passed": not failed,
    }


def _active_compact(samples: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(samples, dtype=np.float64)
    blocks = values.reshape(-1, ACTIVE_BLOCK_SAMPLES)
    rms = np.sqrt(np.mean(np.square(blocks), axis=1))
    threshold = max(
        MINIMUM_ORIGIN_BAND_RMS_LINEAR,
        float(np.max(rms)) * 10.0 ** (ACTIVE_RELATIVE_FLOOR_DB / 20.0),
    )
    selected_indices = np.flatnonzero(rms >= threshold)
    if selected_indices.size < 1:
        raise RecordingSourceConditioningError("active block가 하나도 없습니다")
    selected = blocks[selected_indices].reshape(-1)
    repeats = int(math.ceil(SOURCE_PREFLIGHT_FRAMES / selected.size))
    compacted = np.tile(selected, repeats)[:SOURCE_PREFLIGHT_FRAMES]
    return np.ascontiguousarray(compacted, dtype=np.float64), {
        "block_samples": ACTIVE_BLOCK_SAMPLES,
        "block_seconds": ACTIVE_BLOCK_SAMPLES / SOURCE_PREFLIGHT_SAMPLE_RATE,
        "relative_floor_db": ACTIVE_RELATIVE_FLOOR_DB,
        "absolute_floor_linear": MINIMUM_ORIGIN_BAND_RMS_LINEAR,
        "threshold_linear": threshold,
        "total_blocks": int(blocks.shape[0]),
        "selected_blocks": int(selected_indices.size),
        "selected_block_indices": [int(value) for value in selected_indices],
        "repeat_count": repeats,
    }


def _four_band_components(samples: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(samples, dtype=np.float64)
    result: list[np.ndarray] = []
    for low, high in CONDITIONING_BANDS_HZ:
        taps = signal.firwin(
            EQ_FIR_TAPS,
            [low, high],
            pass_zero=False,
            fs=SOURCE_PREFLIGHT_SAMPLE_RATE,
            window=("kaiser", 8.0),
        )
        result.append(signal.fftconvolve(values, taps, mode="same"))
    return result


def _apply_eq(
    samples: np.ndarray, components: Sequence[np.ndarray], gains_db: Sequence[float]
) -> np.ndarray:
    if len(gains_db) != len(CONDITIONING_BANDS_HZ):
        raise RecordingSourceConditioningError("4-band gain 개수가 다릅니다")
    result = np.asarray(samples, dtype=np.float64).copy()
    for component, raw_gain in zip(components, gains_db):
        gain = _finite(raw_gain, label="4-band gain")
        if not 0.0 <= gain <= EQ_MAX_GAIN_DB:
            raise RecordingSourceConditioningError("4-band gain이 0..6 dB 밖입니다")
        result += (10.0 ** (gain / 20.0) - 1.0) * np.asarray(component)
    return result


def _apply_leveler(samples: np.ndarray, drive: float) -> np.ndarray:
    value = _finite(drive, label="leveler drive")
    if not 0.0 <= value <= LEVELER_MAX_DRIVE:
        raise RecordingSourceConditioningError("leveler drive가 0..6 밖입니다")
    result = np.asarray(samples, dtype=np.float64)
    peak = float(np.max(np.abs(result)))
    if not peak > 0.0:
        raise RecordingSourceConditioningError("leveler 입력 peak가 0입니다")
    normalized = result / peak
    if value > 0.0:
        normalized = np.tanh(value * normalized) / math.tanh(value)
    final_peak = float(np.max(np.abs(normalized)))
    return normalized / final_peak * OUTPUT_PEAK


def _pcm16_wav_bytes(samples: np.ndarray) -> bytes:
    values = np.asarray(samples, dtype=np.float64)
    pcm = np.clip(np.rint(values * 32767.0), -32768, 32767).astype("<i2")
    handle = io.BytesIO()
    with wave.open(handle, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SOURCE_PREFLIGHT_SAMPLE_RATE)
        writer.setnframes(int(pcm.size))
        writer.writeframes(pcm.tobytes())
    return handle.getvalue()


def _strict_primary_prediction(samples: np.ndarray, fir: np.ndarray) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    taps = np.asarray(fir, dtype=np.float64).reshape(-1)
    if taps.size < 2 or not bool(np.isfinite(taps).all()):
        raise RecordingSourceConditioningError("strict primary FIR이 유효하지 않습니다")
    predicted = signal.fftconvolve(values, taps, mode="full")
    peak = float(np.max(np.abs(predicted)))
    rms = math.sqrt(float(np.sum(np.square(predicted))) / values.size)
    return {
        "frames_in": int(values.size),
        "frames_full_convolution": int(predicted.size),
        "peak_linear": peak,
        "rms_linear": rms,
        "band_metric": "hann_full_convolution_rfft_v1",
        "band_rms_dbfs": _band_rms_dbfs(
            predicted, STRICT_PRIMARY_REQUIRED_BANDS_HZ
        ),
    }


def validate_cap_aware_source_preflight(value: Any) -> dict[str, Any]:
    """0.06 legacy field 없이 exact commanded cap을 산술 검증한다."""

    required = {
        "schema",
        "sample_rate",
        "frames",
        "sample_encoding",
        "sample_sha256",
        "commanded_amplitude_millionths",
        "commanded_amplitude_linear",
        "peak_command_tolerance_linear",
        "peak_linear",
        "rms_dbfs",
        "trusted_band_hz",
        "trusted_band_rms_dbfs",
        "official_meter_playback_trusted_band_dbfs",
        "maximum_db_below_official_meter_playback",
        "minimum_trusted_band_rms_dbfs",
        "meter_target_min_dbfs",
        "conservative_quiet_floor_dbfs",
        "predicted_err_trusted_band_min_dbfs",
        "predicted_signal_to_quiet_db",
        "required_capture_coherence_squared",
        "minimum_predicted_signal_to_quiet_db",
        "timeline_feasibility",
        "gates",
        "passed",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise RecordingSourceConditioningError(
            "cap-aware rendered source preflight 필드 집합이 다릅니다"
        )
    millionths = value.get("commanded_amplitude_millionths")
    if (
        isinstance(millionths, bool)
        or not isinstance(millionths, int)
        or not 1 <= millionths <= 6_000
    ):
        raise RecordingSourceConditioningError(
            "cap-aware commanded amplitude millionths가 유효하지 않습니다"
        )
    amplitude = _finite(
        value.get("commanded_amplitude_linear"),
        label="cap-aware commanded amplitude",
    )
    tolerance = _finite(
        value.get("peak_command_tolerance_linear"),
        label="cap-aware peak command tolerance",
    )
    peak = _finite(value.get("peak_linear"), label="cap-aware peak")
    trusted = _finite(
        value.get("trusted_band_rms_dbfs"), label="cap-aware trusted band RMS"
    )
    minimum_trusted = _finite(
        value.get("minimum_trusted_band_rms_dbfs"),
        label="cap-aware minimum trusted band RMS",
    )
    predicted_snr = _finite(
        value.get("predicted_signal_to_quiet_db"),
        label="cap-aware predicted signal-to-quiet",
    )
    minimum_snr = _finite(
        value.get("minimum_predicted_signal_to_quiet_db"),
        label="cap-aware minimum predicted signal-to-quiet",
    )
    timeline = validate_timeline_source_feasibility(value.get("timeline_feasibility"))
    expected_gates = {
        "nonzero_rendered_peak": peak > 0.0,
        "rendered_peak_within_exact_command": peak <= amplitude + tolerance,
        "trusted_band_level": trusted >= minimum_trusted,
        "predicted_signal_to_quiet": predicted_snr >= minimum_snr,
        "timeline_feasibility": timeline["passed"] is True,
    }
    gates = value.get("gates")
    if (
        value.get("schema") != CAP_AWARE_PREFLIGHT_SCHEMA
        or value.get("sample_rate") != SOURCE_PREFLIGHT_SAMPLE_RATE
        or value.get("frames") != SOURCE_PREFLIGHT_FRAMES
        or value.get("sample_encoding") != "float32_le"
        or not isinstance(value.get("sample_sha256"), str)
        or len(value["sample_sha256"]) != 64
        or not math.isclose(
            amplitude, millionths / 1_000_000.0, rel_tol=0.0, abs_tol=0.0
        )
        or not math.isclose(tolerance, 1.0e-6, rel_tol=0.0, abs_tol=0.0)
        or gates != expected_gates
        or value.get("passed") is not all(expected_gates.values())
    ):
        raise RecordingSourceConditioningError(
            "cap-aware rendered source preflight 산술/command 계약 위반"
        )
    return json.loads(json.dumps(dict(value), sort_keys=True, ensure_ascii=False))


def _cap_aware_source_preflight(
    rendered: np.ndarray, *, amplitude_millionths: int
) -> dict[str, Any]:
    """v1의 metric을 재사용하되 0.06 playback authority는 receipt에 전달하지 않는다."""

    legacy_metric = rendered_source_preflight(rendered)
    amplitude = amplitude_millionths / 1_000_000.0
    excluded = {"schema", "playback_amplitude", "passed"}
    evidence = {
        key: value for key, value in legacy_metric.items() if key not in excluded
    }
    peak = float(evidence["peak_linear"])
    gates = {
        "nonzero_rendered_peak": peak > 0.0,
        "rendered_peak_within_exact_command": peak <= amplitude + 1.0e-6,
        "trusted_band_level": float(evidence["trusted_band_rms_dbfs"])
        >= float(evidence["minimum_trusted_band_rms_dbfs"]),
        "predicted_signal_to_quiet": float(
            evidence["predicted_signal_to_quiet_db"]
        )
        >= float(evidence["minimum_predicted_signal_to_quiet_db"]),
        "timeline_feasibility": evidence["timeline_feasibility"]["passed"] is True,
    }
    current = {
        "schema": CAP_AWARE_PREFLIGHT_SCHEMA,
        **evidence,
        "commanded_amplitude_millionths": amplitude_millionths,
        "commanded_amplitude_linear": amplitude,
        "peak_command_tolerance_linear": 1.0e-6,
        "gates": gates,
        "passed": all(gates.values()),
    }
    return validate_cap_aware_source_preflight(current)


def audit_derived_wav_at_cap(
    wav_bytes: bytes, *, strict_primary_fir: np.ndarray, amplitude_millionths: int
) -> dict[str, Any]:
    """실제 file renderer까지 포함해 기존 source/strict-P gate를 그대로 계산한다."""

    if (
        isinstance(amplitude_millionths, bool)
        or not isinstance(amplitude_millionths, int)
        or not 1 <= amplitude_millionths <= 6_000
    ):
        raise RecordingSourceConditioningError("amplitude cap은 1..6000 millionths입니다")
    amplitude = amplitude_millionths / 1_000_000.0
    program = NoiseProgram(
        {
            "type": "file",
            "file": "sealed-derived-source.wav",
            "file_start_seconds": 0.0,
            "amplitude": amplitude,
        },
        SOURCE_PREFLIGHT_SAMPLE_RATE,
        file_bytes=wav_bytes,
    )
    rendered = render_recording_file_window(
        program,
        SOURCE_PREFLIGHT_FRAMES,
        sample_rate=SOURCE_PREFLIGHT_SAMPLE_RATE,
    )
    preflight = _cap_aware_source_preflight(
        rendered, amplitude_millionths=amplitude_millionths
    )
    prediction = _strict_primary_prediction(rendered, strict_primary_fir)
    required_snr = 10.0 * math.log10(
        SOURCE_PREFLIGHT_REQUIRED_COHERENCE_SQUARED
        / (1.0 - SOURCE_PREFLIGHT_REQUIRED_COHERENCE_SQUARED)
    )
    snr = {
        band: float(level - SOURCE_PREFLIGHT_CONSERVATIVE_QUIET_FLOOR_DBFS)
        for band, level in prediction["band_rms_dbfs"].items()
    }
    gates = {
        "cap_aware_rendered_source_preflight": preflight["passed"] is True,
        "strict_primary_two_band_snr": all(
            value >= required_snr for value in snr.values()
        ),
        "strict_primary_peak": prediction["peak_linear"]
        <= ADC_PEAK_HARD_CEILING,
        "strict_primary_rms": prediction["rms_linear"] <= ADC_RMS_HARD_CEILING,
    }
    return {
        "amplitude_millionths": amplitude_millionths,
        "cap_aware_rendered_source_preflight": preflight,
        "strict_primary_prediction": prediction,
        "strict_primary_snr_db": snr,
        "minimum_strict_primary_snr_db": required_snr,
        "adc_peak_hard_ceiling": ADC_PEAK_HARD_CEILING,
        "adc_rms_hard_ceiling": ADC_RMS_HARD_CEILING,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _candidate_recipes() -> list[tuple[bool, tuple[float, ...], float]]:
    recipes: list[tuple[bool, tuple[float, ...], float]] = []
    # 자연 window exact identity를 항상 가장 먼저 검사한다.
    recipes.append((False, _EQ_GAIN_CANDIDATES_DB[0], 0.0))
    for drive in _LEVELER_DRIVES:
        recipes.append((True, _EQ_GAIN_CANDIDATES_DB[0], drive))
    for gains in _EQ_GAIN_CANDIDATES_DB[1:]:
        for drive in (2.0, 4.0, 6.0):
            recipes.append((True, gains, drive))
    return recipes


def condition_source_at_cap(
    *,
    source_bytes: bytes,
    source_path: str,
    start_seconds: float,
    strict_primary_fir: np.ndarray,
    strict_primary_path: str,
    strict_primary_sha256: str,
    amplitude_millionths: int,
    lineage: Mapping[str, Any],
) -> ConditionedSourceResult:
    """한 source를 보수적 후보 집합에서 파생하고 exact gate로 판정한다."""

    source_sha = _sha256_bytes(source_bytes)
    origin, decode = _decode_source_window(
        source_bytes, start_seconds=start_seconds
    )
    admission = origin_band_admission(origin)
    common: dict[str, Any] = {
        "schema": CONDITIONING_RECEIPT_SCHEMA,
        "role": CONDITIONING_ROLE,
        "algorithm_version": CONDITIONING_ALGORITHM_VERSION,
        "origin": {
            "path": str(source_path),
            "size": len(source_bytes),
            "sha256": source_sha,
            "decode": decode,
        },
        "lineage": json.loads(
            json.dumps(dict(lineage), sort_keys=True, ensure_ascii=False)
        ),
        "strict_primary": {
            "path": str(strict_primary_path),
            "sha256": str(strict_primary_sha256),
        },
        "amplitude_millionths": amplitude_millionths,
        "origin_band_admission": admission,
        "thresholds_relaxed": False,
        "natural_unprocessed_evaluation_eligible": False,
    }
    if admission["passed"] is not True:
        receipt = {
            **common,
            "status": "BLOCKED_REPLACEMENT_REQUIRED",
            "blocker_reasons": [
                "origin_band_energy_at_or_below_quantization_guard"
            ],
            "minimum_replacement_requirement": {
                "same_source_family": True,
                "lineage_disjoint": True,
                "all_four_origin_bands_above_dbfs": MINIMUM_ORIGIN_BAND_RMS_DBFS,
            },
            "selected_recipe": None,
            "derived_wav": None,
            "exact_cap_audit": None,
        }
        receipt["receipt_sha256"] = _seal(receipt)
        return ConditionedSourceResult(receipt=receipt, wav_bytes=None)

    origin64 = np.asarray(origin, dtype=np.float64)
    compacted, compaction_evidence = _active_compact(origin64)
    components_by_compaction: dict[bool, list[np.ndarray]] = {}
    attempts: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any], bytes, dict[str, Any]] | None = None
    for compact, gains, drive in _candidate_recipes():
        base = compacted if compact else origin64
        if any(value != 0.0 for value in gains):
            components = components_by_compaction.get(compact)
            if components is None:
                components = _four_band_components(base)
                components_by_compaction[compact] = components
            shaped = _apply_eq(base, components, gains)
        else:
            shaped = base
        conditioned = _apply_leveler(shaped, drive)
        wav_bytes = _pcm16_wav_bytes(conditioned)
        audit = audit_derived_wav_at_cap(
            wav_bytes,
            strict_primary_fir=strict_primary_fir,
            amplitude_millionths=amplitude_millionths,
        )
        recipe = {
            "schema": CONDITIONING_RECIPE_SCHEMA,
            "algorithm_version": CONDITIONING_ALGORITHM_VERSION,
            "active_frame_compaction": compact,
            "active_frame_contract": compaction_evidence if compact else None,
            "four_band_hz": [list(value) for value in CONDITIONING_BANDS_HZ],
            "four_band_gain_db": list(gains),
            "eq_fir_taps": EQ_FIR_TAPS,
            "eq_max_gain_db": EQ_MAX_GAIN_DB,
            "leveler": "peak_normalize_then_tanh_soft_knee_then_pcm16",
            "leveler_drive": drive,
            "leveler_max_drive": LEVELER_MAX_DRIVE,
            "output_peak": OUTPUT_PEAK,
        }
        recipe["recipe_sha256"] = _seal(recipe)
        attempts.append(
            {
                "recipe_sha256": recipe["recipe_sha256"],
                "active_frame_compaction": compact,
                "four_band_gain_db": list(gains),
                "leveler_drive": drive,
                "passed": audit["passed"],
                "minimum_strict_primary_snr_db_observed": min(
                    audit["strict_primary_snr_db"].values()
                ),
                "timeline_eligible_ratio": audit[
                    "cap_aware_rendered_source_preflight"
                ]["timeline_feasibility"]["eligible_ratio"],
            }
        )
        score = min(audit["strict_primary_snr_db"].values())
        if best is None or score > best[0]:
            best = (score, audit, wav_bytes, recipe)
        if audit["passed"] is True:
            derived = {
                "encoding": "wav_pcm16_le_mono_48000",
                "frames": SOURCE_PREFLIGHT_FRAMES,
                "size": len(wav_bytes),
                "sha256": _sha256_bytes(wav_bytes),
            }
            receipt = {
                **common,
                "status": "PASS_COVERAGE_TRAINING_ONLY",
                "blocker_reasons": [],
                "minimum_replacement_requirement": None,
                "selected_recipe": recipe,
                "derived_wav": derived,
                "exact_cap_audit": audit,
                "attempt_count": len(attempts),
                "attempts": attempts,
            }
            receipt["receipt_sha256"] = _seal(receipt)
            return ConditionedSourceResult(receipt=receipt, wav_bytes=wav_bytes)

    assert best is not None
    _score, best_audit, _best_wav, best_recipe = best
    failed_gates = [
        name for name, passed in best_audit["gates"].items() if passed is not True
    ]
    receipt = {
        **common,
        "status": "BLOCKED_REPLACEMENT_REQUIRED",
        "blocker_reasons": failed_gates,
        "minimum_replacement_requirement": {
            "same_source_family": True,
            "lineage_disjoint": True,
            "must_pass_exact_cap_without_exceeding_eq_or_leveler_limits": True,
            "eq_max_gain_db": EQ_MAX_GAIN_DB,
            "leveler_max_drive": LEVELER_MAX_DRIVE,
        },
        "selected_recipe": None,
        "best_rejected_recipe": best_recipe,
        "derived_wav": None,
        "exact_cap_audit": best_audit,
        "attempt_count": len(attempts),
        "attempts": attempts,
    }
    receipt["receipt_sha256"] = _seal(receipt)
    return ConditionedSourceResult(receipt=receipt, wav_bytes=None)


def publish_no_replace(path: str | Path, raw: bytes) -> None:
    """동일 bytes 재검증만 허용하는 no-replace 발행."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not destination.is_file() or destination.read_bytes() != raw:
            raise RecordingSourceConditioningError(
                f"기존 파생 artifact를 덮어쓰지 않습니다: {destination}"
            )
        return
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


__all__ = [
    "CAP_AWARE_PREFLIGHT_SCHEMA",
    "CONDITIONING_ALGORITHM_VERSION",
    "CONDITIONING_BANDS_HZ",
    "CONDITIONING_CAMPAIGN_SCHEMA",
    "CONDITIONING_RECEIPT_SCHEMA",
    "CONDITIONING_RECIPE_SCHEMA",
    "CONDITIONING_ROLE",
    "ConditionedSourceResult",
    "MINIMUM_ORIGIN_BAND_RMS_DBFS",
    "RecordingSourceConditioningError",
    "audit_derived_wav_at_cap",
    "condition_source_at_cap",
    "origin_band_admission",
    "publish_no_replace",
    "validate_cap_aware_source_preflight",
]
