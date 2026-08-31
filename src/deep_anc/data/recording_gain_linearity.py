"""녹음 source gain v2의 bounded physical probe와 immutable receipt.

이 모듈은 signal/분석만 담당하며 오디오 백엔드를 import하거나 장치를 열지 않는다.
live publisher는 ``scripts/data/measure_recording_gain_linearity.py``에 있고, 여기서
만든 exact PCM과 raw를 다시 계산한 PASS receipt만 canonical source-gain plan을 열 수
있다.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from scipy import linalg, signal

from deep_anc.audio_io import (
    analyze_int32_input_probe,
    input_rail_gate,
    pcm_int32_to_float32,
)
from deep_anc.data.holdout_contract import read_regular_file_snapshot
from deep_anc.data.repository_fd import repository_execution_identity


GAIN_LINEARITY_PLAN_SCHEMA = "recording_gain_linearity_plan/v2"
GAIN_LINEARITY_RAW_SCHEMA = "recording_gain_linearity_raw/v2"
GAIN_LINEARITY_RECEIPT_SCHEMA = "recording_gain_linearity_receipt/v2"
SAMPLE_RATE = 48_000
BLOCK_SIZE = 256
LATENCY = "low"
LEVELS_MILLIONTHS = (3_000, 6_000, 9_000, 12_000)
ESS_BAND_HZ = (80.0, 12_000.0)
ESS_SLOT_SECONDS = 2.25
ESS_ACTIVE_LIMIT_SECONDS = 1.5
IMD_PAIRS_HZ = ((1_800.0, 2_200.0), (3_600.0, 4_400.0), (7_200.0, 8_800.0))
IMD_ACTIVE_SECONDS = 0.75
IMD_GUARD_SECONDS = 0.50
IMD_FADE_SECONDS = 0.01
IMD_ANALYSIS_START_SECONDS = 0.10
IMD_ANALYSIS_SECONDS = 0.50
ADC_CERTIFICATION_PEAK = 0.40
ADC_ABSOLUTE_PEAK_CEILING = 0.50
PREDICTIVE_STOP_PEAK = 0.45
PREDICTIVE_UNCERTAINTY_FACTOR = 1.25
COMPRESSION_GATE_DB = 1.0
THD_IMD_GATE_DBC = -30.0
MAX_DELAY_SAMPLES = 4_800
OPERATOR_FIR_LENGTH = 2_048
OPERATOR_FIT_LEVELS_MILLIONTHS = (3_000, 6_000, 9_000)
OPERATOR_HOLDOUT_LEVEL_MILLIONTHS = 12_000
OPERATOR_SUBBANDS_HZ = (
    (80.0, 150.0),
    (150.0, 1_600.0),
    (1_600.0, 4_000.0),
    (4_000.0, 8_000.0),
    (8_000.0, 12_000.0),
)
OPERATOR_MIN_COMPLEX_AGREEMENT = 0.995
OPERATOR_MAX_RELATIVE_ERROR = 0.10
OPERATOR_RIDGE_RELATIVE = 1.0e-8
INPUT_PREFLIGHT_SECONDS = 3.0
STREAM_WATCHDOG_GRACE_SECONDS = 1.0
STREAM_TRANSITION_BUDGET_SECONDS = 1.0
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
GAIN_LINEARITY_SCRIPT_PATH = "scripts/data/measure_recording_gain_linearity.py"
EXACT_OPERATOR_CONFIRMATIONS = {
    "speaker_output": True,
    "user_present": True,
    "volume_minimum": True,
    "routing_and_geometry": True,
    "same_amplifier_setting": True,
    "bounded_gain_probe": True,
}
_RAW_METADATA_KEYS = frozenset(
    {
        "raw_capture_schema",
        "status",
        "source_commit",
        "repository_execution",
        "hardware",
        "plan",
        "operator_confirmations",
        "preflight",
        "segment_telemetry",
        "safety_stop",
        "invalid_reasons",
        "analysis_status",
        "capture_exception",
    }
)
_EXECUTION_KEYS = frozenset(
    {
        "repository_commit",
        "repository_branch",
        "repository_dirty",
        "script_path",
        "script_file_sha256",
    }
)


class RecordingGainLinearityError(ValueError):
    """v2 gain/linearity plan, raw 또는 receipt 계약 위반."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seal(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _require_sha(value: Any, *, label: str) -> str:
    text = str(value).lower()
    if _SHA_RE.fullmatch(text) is None:
        raise RecordingGainLinearityError(f"{label}는 SHA-256이어야 합니다")
    return text


def _require_commit(value: Any) -> str:
    text = str(value).lower()
    if _COMMIT_RE.fullmatch(text) is None:
        raise RecordingGainLinearityError("source_commit은 exact 40자리 SHA여야 합니다")
    return text


def _relative(value: str | Path, *, label: str) -> str:
    text = Path(value).as_posix()
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise RecordingGainLinearityError(f"{label}는 저장소 상대경로여야 합니다")
    return text


def _snapshot(root: Path, relative: str, *, label: str):
    relative = _relative(relative, label=label)
    return read_regular_file_snapshot(
        root / relative, root=root, label=label, capture_bytes=True
    )


def _file_ref(relative: str, snapshot: Any) -> dict[str, Any]:
    return {
        "path": relative,
        "size": int(snapshot.size),
        "sha256": str(snapshot.sha256),
    }


def _safe_peak(value: float) -> np.float32:
    requested = float(value)
    peak = np.float32(requested)
    if float(peak) > requested:
        peak = np.nextafter(peak, np.float32(0.0))
    return peak


def _peak_normalise(values: np.ndarray, peak: float) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    observed = float(np.max(np.abs(source), initial=0.0))
    if not math.isfinite(observed) or observed <= 0.0:
        raise RecordingGainLinearityError("probe가 finite non-zero가 아닙니다")
    safe = _safe_peak(peak)
    result = (source * (float(safe) / observed)).astype(np.float32)
    return np.clip(result, -safe, safe).astype(np.float32, copy=False)


def _synchronised_ess(peak: float) -> tuple[np.ndarray, dict[str, Any]]:
    low, high = ESS_BAND_HZ
    log_ratio = math.log(high / low)
    order = int(math.floor(ESS_ACTIVE_LIMIT_SECONDS * low / log_ratio))
    constant = order / low
    duration = constant * log_ratio
    frames = int(round(duration * SAMPLE_RATE)) + 1
    time = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    phase = 2.0 * math.pi * low * constant * (np.exp(time / constant) - 1.0)
    return _peak_normalise(np.sin(phase), peak), {
        "band_hz": [low, high],
        "synchronisation_order": order,
        "sweep_constant_seconds": constant,
        "active_frames": frames,
        "active_seconds": frames / SAMPLE_RATE,
    }


def _imd(pair: tuple[float, float], peak: float) -> np.ndarray:
    frames = int(round(IMD_ACTIVE_SECONDS * SAMPLE_RATE))
    time = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    values = np.sin(2.0 * math.pi * pair[0] * time)
    values += np.sin(2.0 * math.pi * pair[1] * time)
    fade = int(round(IMD_FADE_SECONDS * SAMPLE_RATE))
    phase = np.arange(fade, dtype=np.float64) / fade
    ramp = 0.5 - 0.5 * np.cos(math.pi * phase)
    values[:fade] *= ramp
    values[-fade:] *= ramp[::-1]
    return _peak_normalise(values, peak)


def _float_to_pcm16(values: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(values, -1.0, 1.0) * 32767.0).astype(np.int16)


def _hardware_contract(
    *, repo_root: Path, hardware_path: str, physical_fingerprint: Mapping[str, Any]
) -> dict[str, Any]:
    relative = _relative(hardware_path, label="hardware path")
    snapshot = _snapshot(repo_root, relative, label="hardware YAML")
    assert snapshot.data is not None
    try:
        hardware = yaml.safe_load(snapshot.data)
    except yaml.YAMLError as exc:
        raise RecordingGainLinearityError("hardware YAML을 읽을 수 없습니다") from exc
    if not isinstance(hardware, dict):
        raise RecordingGainLinearityError("hardware YAML은 mapping이어야 합니다")
    audio = dict(hardware.get("audio") or {})
    channels = dict(hardware.get("channels") or {})
    actual = (
        int(audio.get("sample_rate", 0)),
        int(audio.get("block_size", 0)),
        str(audio.get("latency", "")),
    )
    expected_channels = {
        "error_mic": 0,
        "reference_mic": 1,
        "noise_out": 0,
        "cancel_out": 1,
    }
    if actual != (SAMPLE_RATE, BLOCK_SIZE, LATENCY) or channels != expected_channels:
        raise RecordingGainLinearityError(
            "gain-linearity probe는 48kHz/256/low와 official 0/1 channel map이 필요합니다"
        )
    fingerprint = json.loads(
        json.dumps(dict(physical_fingerprint), sort_keys=True, allow_nan=False)
    )
    return {
        **_file_ref(relative, snapshot),
        "sample_rate": SAMPLE_RATE,
        "block_size": BLOCK_SIZE,
        "latency": LATENCY,
        "channels": expected_channels,
        "physical_fingerprint": fingerprint,
        "physical_fingerprint_sha256": _seal(fingerprint),
    }


def build_gain_linearity_plan(
    *,
    repo_root: str | Path,
    hardware_path: str,
    source_commit: str,
    physical_fingerprint: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    """네 absolute level의 NS-only ESS+IMD exact PCM plan을 만든다."""

    root = Path(repo_root).resolve()
    hardware = _hardware_contract(
        repo_root=root,
        hardware_path=hardware_path,
        physical_fingerprint=physical_fingerprint,
    )
    commit = _require_commit(source_commit)
    ess_slot_frames = int(round(ESS_SLOT_SECONDS * SAMPLE_RATE))
    imd_slot_frames = int(round((IMD_ACTIVE_SECONDS + IMD_GUARD_SECONDS) * SAMPLE_RATE))
    segments: list[np.ndarray] = []
    layout: list[dict[str, Any]] = []
    capture_groups: list[dict[str, Any]] = []
    cursor = 0
    ess_metadata: dict[str, Any] | None = None

    for level in LEVELS_MILLIONTHS:
        group_start = cursor
        first_layout_index = len(layout)
        peak = level / 1_000_000.0
        ess, metadata = _synchronised_ess(peak)
        ess_metadata = metadata
        stereo = np.zeros((ess_slot_frames, 2), dtype=np.float32)
        stereo[: ess.size, 0] = ess
        layout.append(
            {
                "kind": "ESS",
                "level_millionths": level,
                "start_frame": cursor,
                "active_stop_frame": cursor + int(ess.size),
                "stop_frame": cursor + ess_slot_frames,
                "active_frames": int(ess.size),
            }
        )
        segments.append(stereo)
        cursor += ess_slot_frames
        for pair in IMD_PAIRS_HZ:
            tone = _imd(pair, peak)
            stereo = np.zeros((imd_slot_frames, 2), dtype=np.float32)
            stereo[: tone.size, 0] = tone
            layout.append(
                {
                    "kind": "IMD",
                    "level_millionths": level,
                    "pair_hz": [pair[0], pair[1]],
                    "start_frame": cursor,
                    "active_stop_frame": cursor + int(tone.size),
                    "stop_frame": cursor + imd_slot_frames,
                    "active_frames": int(tone.size),
                }
            )
            segments.append(stereo)
            cursor += imd_slot_frames
        capture_groups.append(
            {
                "level_millionths": level,
                "start_frame": group_start,
                "stop_frame": cursor,
                "first_layout_index": first_layout_index,
                "layout_count": 1 + len(IMD_PAIRS_HZ),
            }
        )

    output_float = np.concatenate(segments, axis=0)
    if output_float.shape != (24 * SAMPLE_RATE, 2):
        raise RecordingGainLinearityError("gain-linearity output duration이 24초가 아닙니다")
    if np.any(output_float[:, 1] != 0.0):
        raise RecordingGainLinearityError("CS ch1은 exact zero여야 합니다")
    pcm = _float_to_pcm16(output_float)
    if np.any(pcm[:, 1] != 0):
        raise RecordingGainLinearityError("submitted CS ch1 PCM은 exact zero여야 합니다")
    assert ess_metadata is not None
    active_seconds = (
        len(LEVELS_MILLIONTHS) * ess_metadata["active_seconds"]
        + len(LEVELS_MILLIONTHS) * len(IMD_PAIRS_HZ) * IMD_ACTIVE_SECONDS
    )
    payload: dict[str, Any] = {
        "schema": GAIN_LINEARITY_PLAN_SCHEMA,
        "role": "bounded_gain_linearity_exact_pcm_no_audio",
        "source_commit": commit,
        "hardware": hardware,
        "contract": {
            "levels_millionths": list(LEVELS_MILLIONTHS),
            "drive": "NS_noise_out_ch0_only",
            "cancel_output_exact_zero": True,
            "ess": ess_metadata,
            "ess_slot_seconds": ESS_SLOT_SECONDS,
            "imd_pairs_hz": [list(pair) for pair in IMD_PAIRS_HZ],
            "imd_active_seconds": IMD_ACTIVE_SECONDS,
            "imd_guard_seconds": IMD_GUARD_SECONDS,
            "adc_certification_peak": ADC_CERTIFICATION_PEAK,
            "adc_absolute_peak_ceiling": ADC_ABSOLUTE_PEAK_CEILING,
            "predictive_stop_peak": PREDICTIVE_STOP_PEAK,
            "predictive_uncertainty_factor": PREDICTIVE_UNCERTAINTY_FACTOR,
            "compression_gate_db": COMPRESSION_GATE_DB,
            "thd_imd_gate_dbc": THD_IMD_GATE_DBC,
            "safety_operator": {
                "role": "source_gain_prediction_only_not_anc_plant_authority",
                "band_hz": list(ESS_BAND_HZ),
                "fir_length": OPERATOR_FIR_LENGTH,
                "fit_levels_millionths": list(OPERATOR_FIT_LEVELS_MILLIONTHS),
                "holdout_level_millionths": OPERATOR_HOLDOUT_LEVEL_MILLIONTHS,
                "subbands_hz": [list(value) for value in OPERATOR_SUBBANDS_HZ],
                "minimum_complex_agreement": OPERATOR_MIN_COMPLEX_AGREEMENT,
                "maximum_relative_error": OPERATOR_MAX_RELATIVE_ERROR,
                "residual_uncertainty_factor": PREDICTIVE_UNCERTAINTY_FACTOR,
            },
        },
        "layout": layout,
        "capture_groups": capture_groups,
        "duration": {
            "audible_nonzero_seconds": active_seconds,
            "output_open_seconds": output_float.shape[0] / SAMPLE_RATE,
            "input_preflight_seconds": INPUT_PREFLIGHT_SECONDS,
            # level당 한 6초 stream + 1초 watchdog/1초 fingerprint·전환 예산.
            # 16 slot별 open/close가 아니라 정확히 네 stream만 연다.
            "stream_open_count": len(capture_groups),
            "per_stream_watchdog_grace_seconds": STREAM_WATCHDOG_GRACE_SECONDS,
            "per_stream_transition_budget_seconds": STREAM_TRANSITION_BUDGET_SECONDS,
            "connected_upper_seconds": (
                INPUT_PREFLIGHT_SECONDS
                + output_float.shape[0] / SAMPLE_RATE
                + len(capture_groups)
                * (
                    STREAM_WATCHDOG_GRACE_SECONDS
                    + STREAM_TRANSITION_BUDGET_SECONDS
                )
            ),
        },
        "output": {
            "frames": int(pcm.shape[0]),
            "channels": 2,
            "dtype": "int16",
            "pcm_sha256": _sha256_bytes(pcm.tobytes(order="C")),
            "noise_ch0_pcm_sha256": _sha256_bytes(pcm[:, 0].tobytes(order="C")),
            "cancel_ch1_pcm_sha256": _sha256_bytes(pcm[:, 1].tobytes(order="C")),
            "peak_pcm": int(np.max(np.abs(pcm.astype(np.int32)))),
        },
    }
    payload["plan_payload_sha256"] = _seal(payload)
    return payload, pcm


def validate_gain_linearity_plan_payload(value: Any) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise RecordingGainLinearityError("gain-linearity plan이 mapping이 아닙니다")
    payload = dict(value)
    seal = payload.pop("plan_payload_sha256", None)
    if (
        not isinstance(seal, str)
        or _SHA_RE.fullmatch(seal) is None
        or seal != _seal(payload)
    ):
        raise RecordingGainLinearityError("gain-linearity plan self-seal 불일치")
    if value.get("schema") != GAIN_LINEARITY_PLAN_SCHEMA:
        raise RecordingGainLinearityError("gain-linearity plan schema 불일치")
    contract = value.get("contract")
    output = value.get("output")
    layout = value.get("layout")
    capture_groups = value.get("capture_groups")
    if (
        not isinstance(contract, Mapping)
        or contract.get("levels_millionths") != list(LEVELS_MILLIONTHS)
        or contract.get("drive") != "NS_noise_out_ch0_only"
        or contract.get("cancel_output_exact_zero") is not True
        or not isinstance(output, Mapping)
        or not isinstance(layout, list)
        or len(layout) != len(LEVELS_MILLIONTHS) * (1 + len(IMD_PAIRS_HZ))
        or not isinstance(capture_groups, list)
        or len(capture_groups) != len(LEVELS_MILLIONTHS)
    ):
        raise RecordingGainLinearityError("gain-linearity plan 고정 계약 불일치")
    segments: list[np.ndarray] = []
    for row in layout:
        level = int(row["level_millionths"])
        peak = level / 1_000_000.0
        frames = int(row["stop_frame"]) - int(row["start_frame"])
        stereo = np.zeros((frames, 2), dtype=np.float32)
        if row["kind"] == "ESS":
            source, _ = _synchronised_ess(peak)
        elif row["kind"] == "IMD":
            source = _imd(tuple(float(v) for v in row["pair_hz"]), peak)
        else:
            raise RecordingGainLinearityError("gain-linearity layout kind 불일치")
        if int(row["active_frames"]) != source.size:
            raise RecordingGainLinearityError("gain-linearity active frame 불일치")
        stereo[: source.size, 0] = source
        segments.append(stereo)
    pcm = _float_to_pcm16(np.concatenate(segments, axis=0))
    if (
        pcm.shape != (int(output.get("frames", -1)), 2)
        or _sha256_bytes(pcm.tobytes(order="C")) != output.get("pcm_sha256")
        or np.any(pcm[:, 1] != 0)
    ):
        raise RecordingGainLinearityError("gain-linearity plan PCM 재구성 불일치")
    expected_groups = []
    for group_index, level in enumerate(LEVELS_MILLIONTHS):
        rows = layout[
            group_index * (1 + len(IMD_PAIRS_HZ)) :
            (group_index + 1) * (1 + len(IMD_PAIRS_HZ))
        ]
        expected_groups.append(
            {
                "level_millionths": level,
                "start_frame": int(rows[0]["start_frame"]),
                "stop_frame": int(rows[-1]["stop_frame"]),
                "first_layout_index": group_index * (1 + len(IMD_PAIRS_HZ)),
                "layout_count": 1 + len(IMD_PAIRS_HZ),
            }
        )
    if capture_groups != expected_groups:
        raise RecordingGainLinearityError("gain-linearity capture group 계약 불일치")
    _require_commit(value.get("source_commit"))
    return pcm


def load_gain_linearity_plan(
    *, repo_root: str | Path, plan_path: str, expected_sha256: str
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    relative = _relative(plan_path, label="gain-linearity plan path")
    snapshot = _snapshot(root, relative, label="gain-linearity plan")
    if snapshot.sha256 != _require_sha(expected_sha256, label="plan expected SHA"):
        raise RecordingGainLinearityError("gain-linearity plan 외부 SHA 불일치")
    assert snapshot.data is not None
    try:
        payload = json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordingGainLinearityError(f"gain-linearity plan JSON 오류: {exc}") from exc
    pcm = validate_gain_linearity_plan_payload(payload)
    hardware = payload.get("hardware")
    if not isinstance(hardware, Mapping):
        raise RecordingGainLinearityError("gain-linearity hardware binding이 없습니다")
    rebuilt, rebuilt_pcm = build_gain_linearity_plan(
        repo_root=root,
        hardware_path=str(hardware.get("path", "")),
        source_commit=str(payload.get("source_commit", "")),
        physical_fingerprint=hardware.get("physical_fingerprint") or {},
    )
    if payload != rebuilt or not np.array_equal(pcm, rebuilt_pcm):
        raise RecordingGainLinearityError(
            "gain-linearity plan이 current hardware bytes/recipe에서 재유도되지 않습니다"
        )
    return {"payload": payload, "pcm": pcm, "file": _file_ref(relative, snapshot)}


def next_level_stop_decision(
    *, observed_peak: float, current_millionths: int, next_millionths: int | None
) -> dict[str, Any]:
    peak = float(observed_peak)
    if not math.isfinite(peak) or peak < 0.0:
        raise RecordingGainLinearityError("observed peak가 finite 0 이상이어야 합니다")
    hard = peak >= ADC_ABSOLUTE_PEAK_CEILING
    certification = peak >= ADC_CERTIFICATION_PEAK
    predicted = None
    predictive = False
    if next_millionths is not None:
        predicted = (
            peak
            * float(next_millionths)
            / float(current_millionths)
            * PREDICTIVE_UNCERTAINTY_FACTOR
        )
        predictive = predicted >= PREDICTIVE_STOP_PEAK
    reasons = []
    if hard:
        reasons.append("adc_absolute_peak_ceiling")
    if certification:
        reasons.append("adc_certification_peak")
    if predictive:
        reasons.append("predictive_next_level_peak")
    return {
        "stop": bool(reasons),
        "reasons": reasons,
        "observed_peak": peak,
        "predicted_next_peak": predicted,
    }


def _delay_samples(source: np.ndarray, target: np.ndarray) -> int:
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    correlation = signal.fftconvolve(y, x[::-1], mode="full")
    start = x.size - 1
    window = correlation[start : start + MAX_DELAY_SAMPLES + 1]
    if window.size != MAX_DELAY_SAMPLES + 1:
        raise RecordingGainLinearityError("ESS delay 탐색 window가 짧습니다")
    return int(np.argmax(np.abs(window)))


def _bin_power(spectrum: np.ndarray, bins: set[int]) -> float:
    selected: set[int] = set()
    for index in bins:
        for offset in (-1, 0, 1):
            candidate = index + offset
            if 0 < candidate < spectrum.size:
                selected.add(candidate)
    if not selected:
        return 0.0
    return float(np.sum(np.abs(spectrum[sorted(selected)]) ** 2))


def _distortion_metrics(values: np.ndarray, pair: tuple[float, float]) -> dict[str, float]:
    samples = np.asarray(values, dtype=np.float64)
    window = np.hanning(samples.size)
    spectrum = np.fft.rfft(samples * window)
    resolution = SAMPLE_RATE / samples.size

    def index(frequency: float) -> int:
        return int(round(float(frequency) / resolution))

    fundamentals = {index(pair[0]), index(pair[1])}
    harmonics = {
        index(multiplier * tone)
        for tone in pair
        for multiplier in (2, 3)
        if multiplier * tone < SAMPLE_RATE / 2
    }
    products_hz = {
        abs(pair[1] - pair[0]),
        pair[0] + pair[1],
        abs(2 * pair[0] - pair[1]),
        abs(2 * pair[1] - pair[0]),
        2 * pair[0] + pair[1],
        pair[0] + 2 * pair[1],
    }
    products = {
        index(value)
        for value in products_hz
        if 0.0 < value < SAMPLE_RATE / 2
    }
    harmonics -= fundamentals
    products -= fundamentals | harmonics
    fundamental_power = _bin_power(spectrum, fundamentals)
    floor = np.finfo(np.float64).tiny
    thd = 10.0 * math.log10(max(_bin_power(spectrum, harmonics), floor) / max(fundamental_power, floor))
    imd = 10.0 * math.log10(max(_bin_power(spectrum, products), floor) / max(fundamental_power, floor))
    return {"thd_dbc": float(thd), "imd_dbc": float(imd)}


def _operator_target(
    *,
    input_float: np.ndarray,
    source_float: np.ndarray,
    row: Mapping[str, Any],
    channel: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """한 ESS slot에서 source와 delay-aligned causal response를 돌려준다."""

    start = int(row["start_frame"])
    stop = int(row["stop_frame"])
    active = int(row["active_frames"])
    source = np.asarray(source_float[start : start + active], dtype=np.float64)
    slot = np.asarray(input_float[start:stop, channel], dtype=np.float64)
    delay = _delay_samples(source, slot)
    target_frames = active + OPERATOR_FIR_LENGTH - 1
    if delay + target_frames > slot.size:
        raise RecordingGainLinearityError(
            "ESS slot guard가 compact safety operator tail보다 짧습니다"
        )
    return source, slot[delay : delay + target_frames], delay


def _fit_compact_operator(
    examples: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """여러 ESS level의 causal convolution normal equation을 한 번에 푼다."""

    if not examples:
        raise RecordingGainLinearityError("safety operator fit example이 없습니다")
    autocorrelation = np.zeros(OPERATOR_FIR_LENGTH, dtype=np.float64)
    cross = np.zeros(OPERATOR_FIR_LENGTH, dtype=np.float64)
    for source, target in examples:
        if target.size != source.size + OPERATOR_FIR_LENGTH - 1:
            raise RecordingGainLinearityError("operator fit source/target 길이가 다릅니다")
        # full convolution design X에 대해 X.T@X는 source autocorrelation Toeplitz,
        # X.T@y는 correlate(target, source)의 non-negative lag 구간이다.
        source_correlation = signal.fftconvolve(
            source, source[::-1], mode="full"
        )
        centre = source.size - 1
        autocorrelation += source_correlation[
            centre : centre + OPERATOR_FIR_LENGTH
        ]
        source_target = signal.fftconvolve(target, source[::-1], mode="full")
        cross += source_target[centre : centre + OPERATOR_FIR_LENGTH]
    ridge = max(float(autocorrelation[0]) * OPERATOR_RIDGE_RELATIVE, 1.0e-18)
    first = autocorrelation.copy()
    first[0] += ridge
    try:
        fitted = linalg.solve_toeplitz((first, first), cross, check_finite=True)
    except (ValueError, linalg.LinAlgError) as exc:
        raise RecordingGainLinearityError(
            f"compact safety operator LS 실패: {exc}"
        ) from exc
    fitted = np.asarray(fitted, dtype=np.float64)
    if fitted.shape != (OPERATOR_FIR_LENGTH,) or not bool(np.isfinite(fitted).all()):
        raise RecordingGainLinearityError("compact safety operator가 finite FIR이 아닙니다")
    return fitted


def _complex_roundtrip(
    target: np.ndarray, predicted: np.ndarray, bounds: tuple[float, float]
) -> dict[str, Any]:
    if target.shape != predicted.shape or target.ndim != 1:
        raise RecordingGainLinearityError("operator round-trip shape 불일치")
    target_spectrum = np.fft.rfft(np.asarray(target, dtype=np.float64))
    predicted_spectrum = np.fft.rfft(np.asarray(predicted, dtype=np.float64))
    frequency = np.fft.rfftfreq(target.size, 1.0 / SAMPLE_RATE)
    selected = (frequency >= bounds[0]) & (frequency <= bounds[1])
    truth = target_spectrum[selected]
    estimate = predicted_spectrum[selected]
    truth_norm = float(np.linalg.norm(truth))
    estimate_norm = float(np.linalg.norm(estimate))
    if truth.size < 2 or truth_norm <= 0.0 or estimate_norm <= 0.0:
        agreement = 0.0
        # Receipt JSON은 NaN/Inf를 허용하지 않는다. 1e300은 명시적 finite FAIL
        # sentinel이며 threshold 0.10보다 충분히 크다.
        relative_error = 1.0e300
    else:
        agreement = float(
            abs(np.vdot(estimate, truth)) / (estimate_norm * truth_norm)
        )
        relative_error = float(np.linalg.norm(estimate - truth) / truth_norm)
    passed = bool(
        math.isfinite(agreement)
        and math.isfinite(relative_error)
        and agreement >= OPERATOR_MIN_COMPLEX_AGREEMENT
        and relative_error <= OPERATOR_MAX_RELATIVE_ERROR
    )
    return {
        "band_hz": [float(bounds[0]), float(bounds[1])],
        "complex_agreement": agreement,
        "relative_error": relative_error,
        "passed": passed,
    }


def _operator_example_metrics(
    *, source: np.ndarray, target: np.ndarray, fir: np.ndarray
) -> dict[str, Any]:
    predicted = signal.fftconvolve(source, fir, mode="full")
    rows = [
        _complex_roundtrip(target, predicted, ESS_BAND_HZ),
        *[
            _complex_roundtrip(target, predicted, bounds)
            for bounds in OPERATOR_SUBBANDS_HZ
        ],
    ]
    residual = np.asarray(target - predicted, dtype=np.float64)
    source_peak = float(np.max(np.abs(source)))
    source_rms = math.sqrt(float(np.mean(np.square(source))))
    if source_peak <= 0.0 or source_rms <= 0.0:
        raise RecordingGainLinearityError("operator source level이 0입니다")
    return {
        "roundtrip": rows,
        "passed": all(row["passed"] is True for row in rows),
        "residual_peak_ratio": float(
            np.max(np.abs(residual)) / source_peak
        ),
        "residual_rms_ratio": float(
            math.sqrt(float(np.mean(np.square(residual)))) / source_rms
        ),
    }


def _build_safety_operators(
    *, payload: Mapping[str, Any], source_float: np.ndarray, input_float: np.ndarray
) -> tuple[dict[str, Any], list[str]]:
    """독립 12k holdout을 포함한 ERR/REF source-gain 전용 operator를 만든다."""

    ess_rows = {
        int(row["level_millionths"]): row
        for row in payload["layout"]
        if row["kind"] == "ESS"
    }
    if set(ess_rows) != set(LEVELS_MILLIONTHS):
        raise RecordingGainLinearityError("ESS level row 집합이 exact하지 않습니다")
    result: dict[str, Any] = {
        "schema": "recording_gain_safety_operator/v2",
        "role": "source_gain_prediction_only_not_anc_plant_authority",
        "band_hz": list(ESS_BAND_HZ),
        "fit_levels_millionths": list(OPERATOR_FIT_LEVELS_MILLIONTHS),
        "holdout_level_millionths": OPERATOR_HOLDOUT_LEVEL_MILLIONTHS,
        "fir_length": OPERATOR_FIR_LENGTH,
        "minimum_complex_agreement": OPERATOR_MIN_COMPLEX_AGREEMENT,
        "maximum_relative_error": OPERATOR_MAX_RELATIVE_ERROR,
        "channels": {},
    }
    reasons: list[str] = []
    for name, channel in (("err", 0), ("ref", 1)):
        prepared: dict[int, tuple[np.ndarray, np.ndarray, int]] = {
            level: _operator_target(
                input_float=input_float,
                source_float=source_float,
                row=ess_rows[level],
                channel=channel,
            )
            for level in LEVELS_MILLIONTHS
        }
        fir = _fit_compact_operator(
            [(prepared[level][0], prepared[level][1]) for level in OPERATOR_FIT_LEVELS_MILLIONTHS]
        )
        metrics = {
            level: _operator_example_metrics(
                source=prepared[level][0], target=prepared[level][1], fir=fir
            )
            for level in LEVELS_MILLIONTHS
        }
        fit_passed = all(
            metrics[level]["passed"] is True
            for level in OPERATOR_FIT_LEVELS_MILLIONTHS
        )
        holdout = metrics[OPERATOR_HOLDOUT_LEVEL_MILLIONTHS]
        if not fit_passed:
            reasons.append(f"{name}_operator_fit_roundtrip")
        if holdout["passed"] is not True:
            reasons.append(f"{name}_operator_holdout_roundtrip")
        # ESS residual/source peak ratio는 arbitrary waveform의 induced bound가
        # 아니다. 각 measured level에서 독립 FIR을 다시 풀어 main FIR과의 차이
        # ||delta_h||_1을 얻는다. Young inequality로 임의 source에 대해
        # ||delta_h*x||_inf <= ||delta_h||_1||x||_inf,
        # ||delta_h*x||_2 <= ||delta_h||_1||x||_2를 보장한다. 독립 FIR로도
        # 설명되지 않는 measured residual은 source-independent absolute margin으로
        # 더한다. 이 authority는 max measured level 안에서만 쓸 수 있다.
        delta_l1: list[float] = []
        unexplained_peak: list[float] = []
        unexplained_rms: list[float] = []
        for level in LEVELS_MILLIONTHS:
            source, target, _delay = prepared[level]
            level_fir = _fit_compact_operator([(source, target)])
            delta_l1.append(float(np.sum(np.abs(level_fir - fir))))
            reconstructed = signal.fftconvolve(source, level_fir, mode="full")
            unexplained = np.asarray(target - reconstructed, dtype=np.float64)
            unexplained_peak.append(float(np.max(np.abs(unexplained))))
            unexplained_rms.append(
                float(math.sqrt(float(np.mean(np.square(unexplained)))))
            )
        induced_l1_upper = max(delta_l1) * PREDICTIVE_UNCERTAINTY_FACTOR
        unexplained_peak_upper = (
            max(unexplained_peak) * PREDICTIVE_UNCERTAINTY_FACTOR
        )
        unexplained_rms_upper = (
            max(unexplained_rms) * PREDICTIVE_UNCERTAINTY_FACTOR
        )
        canonical_fir = np.ascontiguousarray(fir, dtype="<f4")
        result["channels"][name] = {
            "delay_samples_by_level": {
                str(level): int(prepared[level][2]) for level in LEVELS_MILLIONTHS
            },
            "fir_encoding": "float32_le",
            "fir": [float(value) for value in canonical_fir],
            "fir_sha256": _sha256_bytes(canonical_fir.tobytes()),
            "fit": [
                {"level_millionths": level, **metrics[level]}
                for level in OPERATOR_FIT_LEVELS_MILLIONTHS
            ],
            "holdout": {
                "level_millionths": OPERATOR_HOLDOUT_LEVEL_MILLIONTHS,
                **holdout,
            },
            "residual_bound": {
                "definition": (
                    "young_l1_induced_plus_measured_absolute_with_uncertainty_v1"
                ),
                "valid_through_amplitude_millionths": LEVELS_MILLIONTHS[-1],
                "induced_fir_l1_upper": float(induced_l1_upper),
                "unexplained_peak_absolute_upper": float(
                    unexplained_peak_upper
                ),
                "unexplained_rms_absolute_upper": float(unexplained_rms_upper),
                "uncertainty_factor": PREDICTIVE_UNCERTAINTY_FACTOR,
            },
            "passed": bool(fit_passed and holdout["passed"] is True),
        }
    result["operator_sha256"] = _seal(result)
    return result, reasons


def _load_raw(
    *, repo_root: Path, raw_path: str, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    relative = _relative(raw_path, label="gain-linearity raw path")
    snapshot = _snapshot(repo_root, relative, label="gain-linearity raw")
    if snapshot.sha256 != _require_sha(expected_sha256, label="raw expected SHA"):
        raise RecordingGainLinearityError("gain-linearity raw 외부 SHA 불일치")
    assert snapshot.data is not None
    try:
        with np.load(io.BytesIO(snapshot.data), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
            arrays = {name: np.asarray(archive[name]) for name in archive.files if name != "metadata_json"}
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RecordingGainLinearityError(f"gain-linearity raw NPZ 오류: {exc}") from exc
    return metadata, arrays, _file_ref(relative, snapshot)


def _validate_raw_authority_metadata(
    *,
    repo_root: Path,
    metadata: Mapping[str, Any],
    preflight_raw: np.ndarray | None,
    source_commit: str,
) -> list[str]:
    """Raw metadata를 현재 clean checkout 및 저장된 input-only raw에서 재유도한다.

    저장된 ``preflight``의 ``valid`` boolean을 신뢰하지 않는다. exact int32 bytes에서
    생존/RMS/코드 다양성/clip과 별도의 0.999 rail gate를 다시 계산한다. 이 함수의
    실패는 receipt를 FAIL로 만들며 물리 probe authority를 발급하지 않는다.
    """

    reasons: list[str] = []
    if set(metadata) != _RAW_METADATA_KEYS:
        reasons.append("raw_metadata_keys")
    if metadata.get("capture_exception") is not None:
        reasons.append("raw_capture_exception")
    if metadata.get("operator_confirmations") != EXACT_OPERATOR_CONFIRMATIONS:
        reasons.append("operator_confirmations")

    saved_execution = metadata.get("repository_execution")
    if not isinstance(saved_execution, Mapping) or set(saved_execution) != _EXECUTION_KEYS:
        reasons.append("repository_execution_schema")
    else:
        if (
            saved_execution.get("repository_commit") != source_commit
            or saved_execution.get("repository_dirty") is not False
            or saved_execution.get("script_path") != GAIN_LINEARITY_SCRIPT_PATH
        ):
            reasons.append("repository_execution_identity")
        try:
            current_execution = repository_execution_identity(
                repo_root, GAIN_LINEARITY_SCRIPT_PATH
            )
        except (OSError, RuntimeError, ValueError):
            reasons.append("repository_execution_current")
        else:
            if dict(saved_execution) != current_execution:
                reasons.append("repository_execution_current")

    expected_frames = int(
        round((INPUT_PREFLIGHT_SECONDS - 0.5) * SAMPLE_RATE)
    )
    if (
        preflight_raw is None
        or preflight_raw.dtype != np.int32
        or preflight_raw.shape != (expected_frames, 2)
    ):
        reasons.append("preflight_raw_shape_or_dtype")
        return reasons
    try:
        recomputed = analyze_int32_input_probe(
            preflight_raw,
            min_rms_dbfs=-80.0,
            max_clip_ratio=0.005,
        )
        rail_ok, rail_ratios = input_rail_gate(
            pcm_int32_to_float32(preflight_raw)
        )
    except (TypeError, ValueError):
        reasons.append("preflight_recompute")
        return reasons
    if not rail_ok or any(
        not bool(channel.get("valid")) for channel in recomputed["channels"][:2]
    ):
        reasons.append("preflight_channel_invalid")
    stored = metadata.get("preflight")
    if (
        not isinstance(stored, Mapping)
        or set(stored) != {"frames", "channels", "device", "sample_rate", "settle_seconds"}
        or stored.get("frames") != recomputed["frames"]
        or stored.get("channels") != recomputed["channels"]
        or type(stored.get("device")) is not int
        or stored.get("sample_rate") != SAMPLE_RATE
        or float(stored.get("settle_seconds", float("nan"))) != 0.5
    ):
        reasons.append("preflight_report_not_exact")
    # rail은 legacy stored report에 필드가 없으므로 raw에서만 독립 판정한다. receipt에는
    # 재계산 결과를 남겨 downstream이 saved boolean 대신 수치를 감사할 수 있게 한다.
    metadata_rail = metadata.get("preflight")
    if isinstance(metadata_rail, Mapping) and any(
        not math.isfinite(float(value)) for value in rail_ratios
    ):
        reasons.append("preflight_rail_nonfinite")
    return reasons


def build_gain_linearity_receipt(
    *,
    repo_root: str | Path,
    raw_path: str,
    expected_raw_sha256: str,
    plan_path: str,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    """Raw/plan을 독립 재계산하고 PASS/FAIL receipt payload를 만든다."""

    root = Path(repo_root).resolve()
    plan = load_gain_linearity_plan(
        repo_root=root, plan_path=plan_path, expected_sha256=expected_plan_sha256
    )
    metadata, arrays, raw_ref = _load_raw(
        repo_root=root, raw_path=raw_path, expected_sha256=expected_raw_sha256
    )
    payload = plan["payload"]
    pcm = plan["pcm"]
    reasons: list[str] = []
    if metadata.get("raw_capture_schema") != GAIN_LINEARITY_RAW_SCHEMA:
        reasons.append("raw_schema")
    expected_plan_ref = {
        "path": plan["file"]["path"],
        "sha256": plan["file"]["sha256"],
        "payload_sha256": payload["plan_payload_sha256"],
        "pcm_sha256": payload["output"]["pcm_sha256"],
    }
    if metadata.get("plan") != expected_plan_ref:
        reasons.append("raw_plan_binding")
    if metadata.get("source_commit") != payload["source_commit"]:
        reasons.append("source_commit")
    if metadata.get("hardware") != payload["hardware"]:
        reasons.append("hardware_fingerprint")
    if metadata.get("status") != "RAW_COMPLETE_NOT_ANALYSED":
        reasons.append("raw_status")
    if metadata.get("analysis_status") != "NOT_RUN_RAW_FIRST":
        reasons.append("raw_analysis_status")
    if metadata.get("safety_stop") is not None:
        reasons.append("raw_safety_stop")
    if metadata.get("invalid_reasons") != []:
        reasons.append("raw_invalid_reasons")
    submitted = arrays.get("submitted_output_pcm_int16")
    recorded = arrays.get("input_raw_int32")
    preflight = arrays.get("preflight_raw_int32")
    reasons.extend(
        _validate_raw_authority_metadata(
            repo_root=root,
            metadata=metadata,
            preflight_raw=preflight,
            source_commit=str(payload["source_commit"]),
        )
    )
    if (
        submitted is None
        or submitted.dtype != np.int16
        or submitted.shape != pcm.shape
        or not np.array_equal(submitted, pcm)
    ):
        reasons.append("submitted_pcm_not_exact")
    if recorded is None or recorded.dtype != np.int32 or recorded.shape != pcm.shape:
        reasons.append("input_raw_shape_or_dtype")
    telemetry = metadata.get("segment_telemetry")
    if not isinstance(telemetry, list) or len(telemetry) != len(payload["capture_groups"]):
        reasons.append("segment_telemetry_count")
    else:
        for item, group in zip(telemetry, payload["capture_groups"], strict=True):
            if (
                item.get("level_millionths") != group["level_millionths"]
                or item.get("start_frame") != group["start_frame"]
                or item.get("stop_frame") != group["stop_frame"]
                or item.get("completed") is not True
                or int(item.get("xrun_count", -1)) != 0
                or int(item.get("unexpected_status_count", -1)) != 0
                or int(item.get("callback_status_count", -1)) != 0
                or item.get("callback_error") is not None
                or item.get("stream_abort_error") is not None
                or item.get("stream_close_error") is not None
                or item.get("output_stop_confirmed") is not True
            ):
                reasons.append("segment_telemetry_invalid")
                break
    if reasons or submitted is None or recorded is None:
        analysis: dict[str, Any] = {"rows": [], "failure_before_metrics": True}
    else:
        source_float = submitted[:, 0].astype(np.float64) / 32767.0
        input_float = recorded.astype(np.float64) / float(2**31)
        first_ess = payload["layout"][0]
        x0 = source_float[first_ess["start_frame"] : first_ess["active_stop_frame"]]
        delays = {
            name: _delay_samples(
                x0,
                input_float[
                    first_ess["start_frame"] : first_ess["stop_frame"], channel
                ],
            )
            for name, channel in (("err", 0), ("ref", 1))
        }
        try:
            safety_operator, operator_reasons = _build_safety_operators(
                payload=payload,
                source_float=source_float,
                input_float=input_float,
            )
            reasons.extend(operator_reasons)
        except RecordingGainLinearityError as exc:
            safety_operator = None
            reasons.append(f"safety_operator_build:{exc}")
        rows: list[dict[str, Any]] = []
        ess_gain: dict[str, list[tuple[int, float]]] = {"err": [], "ref": []}
        peak_ratios: dict[str, list[float]] = {"err": [], "ref": []}
        for row in payload["layout"]:
            start, stop = int(row["start_frame"]), int(row["stop_frame"])
            level = int(row["level_millionths"])
            result: dict[str, Any] = {
                "kind": row["kind"],
                "level_millionths": level,
                "pair_hz": row.get("pair_hz"),
                "channels": {},
            }
            for name, channel in (("err", 0), ("ref", 1)):
                values = input_float[start:stop, channel]
                peak = float(np.max(np.abs(values)))
                clip_ratio = float(np.mean(np.abs(values) >= 0.999))
                peak_ratios[name].append(peak / (level / 1_000_000.0))
                item: dict[str, Any] = {
                    "peak_linear": peak,
                    "rms_dbfs": float(
                        20.0 * math.log10(math.sqrt(float(np.mean(values**2))) + 1e-30)
                    ),
                    "clip_ratio": clip_ratio,
                }
                if peak >= ADC_ABSOLUTE_PEAK_CEILING:
                    reasons.append(f"{name}_absolute_peak_{level}")
                if peak >= ADC_CERTIFICATION_PEAK:
                    reasons.append(f"{name}_certification_peak_{level}")
                if clip_ratio != 0.0:
                    reasons.append(f"{name}_clip_{level}")
                active = int(row["active_frames"])
                drive = source_float[start : start + active]
                delay = _delay_samples(drive, input_float[start:stop, channel])
                item["delay_samples"] = delay
                if row["kind"] == "ESS":
                    response = input_float[start + delay : start + delay + active, channel]
                    rms = math.sqrt(float(np.mean(response**2)))
                    normalized = 20.0 * math.log10(max(rms, 1e-30)) - 20.0 * math.log10(
                        level / 1_000_000.0
                    )
                    item["normalised_gain_db"] = float(normalized)
                    ess_gain[name].append((level, float(normalized)))
                else:
                    analysis_start = start + delay + int(
                        round(IMD_ANALYSIS_START_SECONDS * SAMPLE_RATE)
                    )
                    analysis_stop = analysis_start + int(
                        round(IMD_ANALYSIS_SECONDS * SAMPLE_RATE)
                    )
                    distortion = _distortion_metrics(
                        input_float[analysis_start:analysis_stop, channel],
                        tuple(float(v) for v in row["pair_hz"]),
                    )
                    item.update(distortion)
                    if distortion["thd_dbc"] > THD_IMD_GATE_DBC:
                        reasons.append(f"{name}_thd_{level}_{row['pair_hz']}")
                    if distortion["imd_dbc"] > THD_IMD_GATE_DBC:
                        reasons.append(f"{name}_imd_{level}_{row['pair_hz']}")
                result["channels"][name] = item
            rows.append(result)
        compression: dict[str, Any] = {}
        for name, values in ess_gain.items():
            baseline = values[0][1]
            deviations = [gain - baseline for _, gain in values]
            maximum = float(max(abs(value) for value in deviations))
            compression[name] = {
                "baseline_level_millionths": values[0][0],
                "deviation_db": deviations,
                "maximum_abs_deviation_db": maximum,
            }
            if maximum > COMPRESSION_GATE_DB:
                reasons.append(f"{name}_compression")
        ratio_upper = {
            name: float(max(values) * PREDICTIVE_UNCERTAINTY_FACTOR)
            for name, values in peak_ratios.items()
        }
        empirical_upper = {
            name: min(
                LEVELS_MILLIONTHS[-1],
                int(math.floor(ADC_CERTIFICATION_PEAK / value * 1_000_000.0)),
            )
            for name, value in ratio_upper.items()
        }
        analysis = {
            "rows": rows,
            "delay_samples": delays,
            "compression": compression,
            "peak_gain_upper_with_uncertainty": ratio_upper,
            "empirical_upper_amplitude_millionths": empirical_upper,
            "supported_max_amplitude_millionths": min(empirical_upper.values()),
            "safety_operator": safety_operator,
            "safety_operator_is_anc_plant_authority": False,
            "failure_before_metrics": False,
        }
    reasons = list(dict.fromkeys(reasons))
    receipt: dict[str, Any] = {
        "schema": GAIN_LINEARITY_RECEIPT_SCHEMA,
        "status": "PASS" if not reasons else "FAIL",
        "source_commit": payload["source_commit"],
        "plan": plan["file"],
        "plan_payload_sha256": payload["plan_payload_sha256"],
        "raw": raw_ref,
        "hardware": payload["hardware"],
        "contract": payload["contract"],
        "analysis": analysis,
        "failure_reasons": reasons,
    }
    receipt["evidence_sha256"] = _seal(receipt)
    return receipt


def issue_gain_linearity_receipt(
    *,
    repo_root: str | Path,
    output_path: str,
    raw_path: str,
    expected_raw_sha256: str,
    plan_path: str,
    expected_plan_sha256: str,
) -> tuple[Path, str, dict[str, Any]]:
    root = Path(repo_root).resolve()
    relative = _relative(output_path, label="gain-linearity receipt path")
    payload = build_gain_linearity_receipt(
        repo_root=root,
        raw_path=raw_path,
        expected_raw_sha256=expected_raw_sha256,
        plan_path=plan_path,
        expected_plan_sha256=expected_plan_sha256,
    )
    data = _pretty_json_bytes(payload)
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
    except FileExistsError as exc:
        raise RecordingGainLinearityError(f"receipt는 no-replace입니다: {relative}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return target, _sha256_bytes(data), payload


def validate_gain_linearity_receipt(
    *, repo_root: str | Path, receipt_path: str, expected_sha256: str
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    relative = _relative(receipt_path, label="gain-linearity receipt path")
    snapshot = _snapshot(root, relative, label="gain-linearity receipt")
    if snapshot.sha256 != _require_sha(expected_sha256, label="receipt expected SHA"):
        raise RecordingGainLinearityError("gain-linearity receipt 외부 SHA 불일치")
    assert snapshot.data is not None
    try:
        payload = json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordingGainLinearityError(f"gain-linearity receipt JSON 오류: {exc}") from exc
    if not isinstance(payload, dict):
        raise RecordingGainLinearityError("gain-linearity receipt가 mapping이 아닙니다")
    seal = payload.get("evidence_sha256")
    unsealed = dict(payload)
    unsealed.pop("evidence_sha256", None)
    if not isinstance(seal, str) or seal != _seal(unsealed):
        raise RecordingGainLinearityError("gain-linearity receipt self-seal 불일치")
    rebuilt = build_gain_linearity_receipt(
        repo_root=root,
        raw_path=str(payload["raw"]["path"]),
        expected_raw_sha256=str(payload["raw"]["sha256"]),
        plan_path=str(payload["plan"]["path"]),
        expected_plan_sha256=str(payload["plan"]["sha256"]),
    )
    if payload != rebuilt:
        raise RecordingGainLinearityError("gain-linearity receipt 독립 재계산 불일치")
    return {
        "receipt_path": relative,
        "receipt_sha256": str(snapshot.sha256),
        "payload": payload,
        "passed": payload.get("status") == "PASS",
    }


__all__ = [
    "ADC_ABSOLUTE_PEAK_CEILING",
    "ADC_CERTIFICATION_PEAK",
    "GAIN_LINEARITY_PLAN_SCHEMA",
    "GAIN_LINEARITY_RAW_SCHEMA",
    "GAIN_LINEARITY_RECEIPT_SCHEMA",
    "LEVELS_MILLIONTHS",
    "PREDICTIVE_STOP_PEAK",
    "RecordingGainLinearityError",
    "build_gain_linearity_plan",
    "build_gain_linearity_receipt",
    "issue_gain_linearity_receipt",
    "load_gain_linearity_plan",
    "next_level_stop_decision",
    "validate_gain_linearity_plan_payload",
    "validate_gain_linearity_receipt",
]
