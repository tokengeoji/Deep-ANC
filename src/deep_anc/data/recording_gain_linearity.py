"""녹음 source gain v3의 bounded physical probe와 immutable receipt.

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
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from scipy import interpolate, linalg, signal

from deep_anc.audio_io import (
    analyze_int32_input_probe,
    input_rail_gate,
    pcm_int32_to_float32,
)
from deep_anc.data.holdout_contract import read_regular_file_snapshot
from deep_anc.data.recorded_qa import (
    CAPTURE_MIN_LOW_BAND_COHERENCE,
    MIN_REF_ERR_COHERENCE,
)
from deep_anc.data.repository_fd import repository_execution_identity
from deep_anc.data.timeline import (
    TIMELINE_METHOD,
    TimelineSettings,
    align_source_to_adc,
    estimate_lag_track,
)
from deep_anc.dsp.invariants import MIN_STREAM_COHERENCE, MIN_STREAM_DELAY_VALID_WINDOWS


GAIN_LINEARITY_PLAN_SCHEMA = "recording_gain_linearity_plan/v3_gainprobe006"
GAIN_LINEARITY_RAW_SCHEMA = "recording_gain_linearity_raw/v3_gainprobe006"
GAIN_LINEARITY_PUBLICATION_SCHEMA = (
    "recording_gain_linearity_capture_publication/v1_gainprobe006"
)
GAIN_LINEARITY_RECEIPT_SCHEMA = "recording_gain_linearity_receipt/v4_gainprobe006"
GAIN_LINEARITY_REANALYSIS_RECEIPT_SCHEMA = (
    "recording_gain_linearity_receipt/v5_gainprobe006_ref_witness_reanalysis"
)
GAIN_LINEARITY_AUTHORITY_SCOPE = (
    "tested_range_adc_peak_safety_only_not_global_nonlinearity"
)
GAIN_LINEARITY_PEAK_ENVELOPE_SCHEMA = (
    "recording_gain_peak_safety_envelope/v1_gainprobe006"
)
SAMPLE_RATE = 48_000
BLOCK_SIZE = 256
LATENCY = "low"
LEVELS_MILLIONTHS = (3_000, 4_000, 5_000, 6_000)
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
DISTORTION_MIN_FUNDAMENTAL_SNR_DB = 40.0
DISTORTION_MIN_NOISE_MARGIN_DB = 10.0
DISTORTION_COHERENT_SUBWINDOW_SECONDS = 0.125
MAX_DELAY_SAMPLES = 4_800
OPERATOR_FIR_LENGTH = 2_048
OPERATOR_PEAK_PRE_ROLL_SAMPLES = 256
OPERATOR_FIT_LEVELS_MILLIONTHS = (3_000, 4_000, 5_000)
OPERATOR_HOLDOUT_LEVEL_MILLIONTHS = 6_000
OPERATOR_SUBBANDS_HZ = (
    (80.0, 150.0),
    (150.0, 1_600.0),
    (1_600.0, 4_000.0),
    (4_000.0, 8_000.0),
    (8_000.0, 12_000.0),
)
OPERATOR_MIN_COMPLEX_AGREEMENT = 0.995
OPERATOR_MAX_RELATIVE_ERROR = 0.10
OPERATOR_RELATIVE_SUBBAND_MIN_NORM_RATIO = 0.01
OPERATOR_RIDGE_RELATIVE = 1.0e-8
INPUT_PREFLIGHT_SECONDS = 3.0
STREAM_WATCHDOG_GRACE_SECONDS = 1.0
STREAM_TRANSITION_BUDGET_SECONDS = 1.0
GROUP_SETTLE_SECONDS = 0.5
CLOCK_PILOT_FRAMES = 2_048
CLOCK_PILOT_OFFSETS_SECONDS = (0.0, 2.0, 3.25, 4.5, 5.65)
CLOCK_PILOT_MIN_NORMALISED_CORRELATION = 0.90
CLOCK_MAX_ABS_PPM = 1_000.0
CLOCK_CHANNEL_MAX_DIFFERENCE_PPM = 100.0
CLOCK_TRAJECTORY_MAX_RESIDUAL_SAMPLES = 0.25
RELATIVE_DELAY_MAX_SPREAD_SAMPLES = 3.0
CALLBACK_TIME_INFO_FIELDS = (
    "callback_start_frames",
    "callback_frame_counts",
    "input_buffer_adc_time",
    "output_buffer_dac_time",
    "callback_current_time",
)
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
GAIN_LINEARITY_SCRIPT_PATH = "scripts/data/measure_recording_gain_linearity.py"
GAIN_LINEARITY_ANALYZER_PATH = "src/deep_anc/data/recording_gain_linearity.py"
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
_PUBLICATION_REF_KEYS = frozenset(
    {"path", "size", "sha256", "capture_device", "capture_inode"}
)
_PUBLICATION_KEYS = frozenset(
    {
        "schema",
        "role",
        "canonical_session_path",
        "source_commit",
        "repository_execution",
        "plan",
        "raw",
        "metadata",
        "publication_payload_sha256",
    }
)


class RecordingGainLinearityError(ValueError):
    """v3 gain/linearity plan, raw 또는 receipt 계약 위반."""


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


def _verify_historical_capture_execution(
    repo_root: Path, execution: Mapping[str, Any]
) -> None:
    """저장된 capture script SHA를 해당 역사 commit의 Git blob과 대조한다.

    재분석 checkout은 capture commit과 달라도 된다. 그렇다고 raw가 선언한 옛
    ``script_file_sha256``를 자체 진술로만 믿지는 않는다. 원 commit의 blob을 직접
    읽어 SHA를 비교하며 replace/graft가 없는 clean 현재 checkout 확인은 별도의
    analyzer execution identity가 담당한다.
    """

    commit = _require_commit(execution.get("repository_commit"))
    script = _relative(
        str(execution.get("script_path", "")), label="historical capture script path"
    )
    expected = _require_sha(
        execution.get("script_file_sha256"), label="historical capture script SHA"
    )
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "blob", f"{commit}:{script}"],
            check=True,
            capture_output=True,
            # ``git cat-file <commit>:<path>`` normally honours refs/replace.
            # The current analyzer identity rejects replacement refs before this
            # function is reached, but the historical lookup is also made exact
            # in isolation so a future caller cannot silently read replacement
            # bytes instead of the capture commit's real tree.
            env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RecordingGainLinearityError(
            "historical capture commit/script blob을 검증할 수 없습니다"
        ) from exc
    if _sha256_bytes(completed.stdout) != expected:
        raise RecordingGainLinearityError(
            "historical capture script SHA가 선언 commit의 blob과 다릅니다"
        )


def _analysis_execution_identity(repo_root: Path) -> dict[str, Any]:
    """현행 clean analyzer module을 capture 실행과 분리해 결속한다."""

    execution = repository_execution_identity(repo_root, GAIN_LINEARITY_ANALYZER_PATH)
    if (
        not isinstance(execution, Mapping)
        or set(execution) != _EXECUTION_KEYS
        or execution.get("repository_dirty") is not False
        or execution.get("script_path") != GAIN_LINEARITY_ANALYZER_PATH
        or _COMMIT_RE.fullmatch(str(execution.get("repository_commit", ""))) is None
        or _SHA_RE.fullmatch(str(execution.get("script_file_sha256", ""))) is None
    ):
        raise RecordingGainLinearityError("gain-linearity analyzer execution identity 불일치")
    return dict(execution)


def _portable_publication_ref(
    value: Mapping[str, Any], *, expected_path: str, label: str
) -> dict[str, Any]:
    """Held-dir publisher ref를 portable capture witness로 정규화한다.

    ``capture_device``/``capture_inode``는 Jetson에서 같은 held descriptor로 발행했다는
    provenance다. 전송 뒤 inode가 달라지는 것은 정상이라 offline 현재 inode와 비교하지
    않고, path/size/SHA와 외부 publication SHA를 다시 검증한다.
    """

    if not isinstance(value, Mapping):
        raise RecordingGainLinearityError(f"{label} publication ref가 mapping이 아닙니다")
    path = _relative(str(value.get("path", "")), label=f"{label} publication path")
    size = value.get("size")
    device = value.get("device", value.get("capture_device"))
    inode = value.get("inode", value.get("capture_inode"))
    if (
        path != expected_path
        or type(size) is not int
        or size < 0
        or _SHA_RE.fullmatch(str(value.get("sha256", "")).lower()) is None
        or type(device) is not int
        or device < 0
        or type(inode) is not int
        or inode <= 0
    ):
        raise RecordingGainLinearityError(f"{label} publication ref 계약 불일치")
    return {
        "path": path,
        "size": int(size),
        "sha256": str(value["sha256"]).lower(),
        "capture_device": int(device),
        "capture_inode": int(inode),
    }


def build_gain_linearity_capture_publication_payload(
    *,
    canonical_session_path: str,
    raw_ref: Mapping[str, Any],
    metadata_ref: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Raw+sidecar의 held-dir no-replace 발행 결과를 self-seal한다."""

    session = _relative(
        canonical_session_path, label="gain-linearity canonical session path"
    )
    session_parts = Path(session).parts
    if len(session_parts) < 2 or session_parts[0] != "results":
        raise RecordingGainLinearityError(
            "gain-linearity canonical session은 results/ 아래 directory여야 합니다"
        )
    raw_path = f"{session}/raw_measurement.npz"
    metadata_path = f"{session}/metadata.json"
    execution = metadata.get("repository_execution")
    plan = metadata.get("plan")
    source_commit = metadata.get("source_commit")
    if (
        not isinstance(execution, Mapping)
        or set(execution) != _EXECUTION_KEYS
        or not isinstance(plan, Mapping)
        or _COMMIT_RE.fullmatch(str(source_commit).lower()) is None
    ):
        raise RecordingGainLinearityError(
            "capture publication의 raw metadata authority가 불완전합니다"
        )
    payload: dict[str, Any] = {
        "schema": GAIN_LINEARITY_PUBLICATION_SCHEMA,
        "role": "held_directory_noreplace_raw_sidecar_binding",
        "canonical_session_path": session,
        "source_commit": str(source_commit).lower(),
        "repository_execution": json.loads(
            json.dumps(dict(execution), sort_keys=True, allow_nan=False)
        ),
        "plan": json.loads(json.dumps(dict(plan), sort_keys=True, allow_nan=False)),
        "raw": _portable_publication_ref(
            raw_ref, expected_path=raw_path, label="raw"
        ),
        "metadata": _portable_publication_ref(
            metadata_ref, expected_path=metadata_path, label="metadata"
        ),
    }
    payload["publication_payload_sha256"] = _seal(payload)
    return payload


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


def _clock_pilot(peak: float) -> np.ndarray:
    """그룹 처음/끝에 똑같이 넣는 deterministic wideband clock pilot."""

    time = np.arange(CLOCK_PILOT_FRAMES, dtype=np.float64) / SAMPLE_RATE
    duration = CLOCK_PILOT_FRAMES / SAMPLE_RATE
    phase = 2.0 * math.pi * (
        300.0 * time
        + 0.5 * (9_000.0 - 300.0) / duration * np.square(time)
    )
    values = np.sin(phase)
    fade = min(256, CLOCK_PILOT_FRAMES // 8)
    ramp = np.sin(
        0.5 * math.pi * np.arange(fade, dtype=np.float64) / max(1, fade)
    ) ** 2
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
    layout: list[dict[str, Any]] = []
    clock_pilot_layout: list[dict[str, Any]] = []
    capture_groups: list[dict[str, Any]] = []
    group_stimulus_frames = 6 * SAMPLE_RATE
    group_settle_frames = int(round(GROUP_SETTLE_SECONDS * SAMPLE_RATE))
    group_frames = group_settle_frames + group_stimulus_frames
    output_float = np.zeros(
        (len(LEVELS_MILLIONTHS) * group_frames, 2), dtype=np.float32
    )
    ess_metadata: dict[str, Any] | None = None

    for group_index, level in enumerate(LEVELS_MILLIONTHS):
        group_start = group_index * group_frames
        stimulus_start = group_start + group_settle_frames
        group_stop = group_start + group_frames
        first_layout_index = len(layout)
        peak = level / 1_000_000.0
        pilot = _clock_pilot(peak)
        pilot_starts = [
            stimulus_start + int(round(offset * SAMPLE_RATE))
            for offset in CLOCK_PILOT_OFFSETS_SECONDS
        ]
        pilot_pcm = _float_to_pcm16(pilot)
        for pilot_repeat_index, pilot_start in enumerate(pilot_starts):
            pilot_stop = pilot_start + pilot.size
            if pilot_start < stimulus_start or pilot_stop > group_stop:
                raise RecordingGainLinearityError("clock pilot가 group 밖입니다")
            output_float[pilot_start:pilot_stop, 0] = pilot
            clock_pilot_layout.append(
                {
                    "kind": "CLOCK_PILOT",
                    "level_millionths": level,
                    "repeat_index": pilot_repeat_index,
                    "start_frame": pilot_start,
                    "stop_frame": pilot_stop,
                    "active_frames": CLOCK_PILOT_FRAMES,
                    "noise_ch0_pcm_sha256": _sha256_bytes(
                        pilot_pcm.tobytes(order="C")
                    ),
                }
            )

        ess, metadata = _synchronised_ess(peak)
        ess_metadata = metadata
        ess_start = stimulus_start + int(round(0.25 * SAMPLE_RATE))
        ess_stop = stimulus_start + ess_slot_frames
        if ess_start + ess.size > ess_stop:
            raise RecordingGainLinearityError("ESS가 clock pilot guard를 침범합니다")
        output_float[ess_start : ess_start + ess.size, 0] = ess
        layout.append(
            {
                "kind": "ESS",
                "level_millionths": level,
                "start_frame": ess_start,
                "active_stop_frame": ess_start + int(ess.size),
                "stop_frame": ess_stop,
                "active_frames": int(ess.size),
            }
        )
        rotated_pairs = tuple(
            IMD_PAIRS_HZ[(group_index + offset) % len(IMD_PAIRS_HZ)]
            for offset in range(len(IMD_PAIRS_HZ))
        )
        for pair_index, pair in enumerate(rotated_pairs):
            tone = _imd(pair, peak)
            tone_start = stimulus_start + ess_slot_frames + pair_index * imd_slot_frames
            tone_stop = tone_start + imd_slot_frames
            output_float[tone_start : tone_start + tone.size, 0] = tone
            layout.append(
                {
                    "kind": "IMD",
                    "level_millionths": level,
                    "pair_hz": [pair[0], pair[1]],
                    "start_frame": tone_start,
                    "active_stop_frame": tone_start + int(tone.size),
                    "stop_frame": tone_stop,
                    "active_frames": int(tone.size),
                }
            )
        capture_groups.append(
            {
                "level_millionths": level,
                "start_frame": group_start,
                "settle_frames": group_settle_frames,
                "stimulus_start_frame": stimulus_start,
                "stop_frame": group_stop,
                "clock_pilot_start_frames": pilot_starts,
                "clock_pilot_active_frames": CLOCK_PILOT_FRAMES,
                "imd_pair_order_hz": [list(pair) for pair in rotated_pairs],
                "first_layout_index": first_layout_index,
                "layout_count": 1 + len(IMD_PAIRS_HZ),
            }
        )

    if output_float.shape != (26 * SAMPLE_RATE, 2):
        raise RecordingGainLinearityError("gain-linearity output duration이 26초가 아닙니다")
    if np.any(output_float[:, 1] != 0.0):
        raise RecordingGainLinearityError("CS ch1은 exact zero여야 합니다")
    pcm = _float_to_pcm16(output_float)
    if np.any(pcm[:, 1] != 0):
        raise RecordingGainLinearityError("submitted CS ch1 PCM은 exact zero여야 합니다")
    assert ess_metadata is not None
    active_seconds = (
        len(LEVELS_MILLIONTHS) * ess_metadata["active_seconds"]
        + len(LEVELS_MILLIONTHS) * len(IMD_PAIRS_HZ) * IMD_ACTIVE_SECONDS
        + len(LEVELS_MILLIONTHS)
        * len(CLOCK_PILOT_OFFSETS_SECONDS)
        * CLOCK_PILOT_FRAMES
        / SAMPLE_RATE
    )
    exact_nonzero_pcm_seconds = float(np.count_nonzero(pcm[:, 0])) / SAMPLE_RATE
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
            "imd_pair_order": "cyclic_rotation_by_level_index",
            "imd_active_seconds": IMD_ACTIVE_SECONDS,
            "imd_guard_seconds": IMD_GUARD_SECONDS,
            "group_settle_seconds": GROUP_SETTLE_SECONDS,
            "group_settle_exact_zero": True,
            "clock_pilot": {
                "role": "repeated_group_affine_clock_and_common_offset_witness",
                "active_frames": CLOCK_PILOT_FRAMES,
                "offset_seconds": list(CLOCK_PILOT_OFFSETS_SECONDS),
                "minimum_repeat_response_normalised_correlation": (
                    CLOCK_PILOT_MIN_NORMALISED_CORRELATION
                ),
                "maximum_abs_ppm": CLOCK_MAX_ABS_PPM,
                "maximum_channel_difference_ppm": (
                    CLOCK_CHANNEL_MAX_DIFFERENCE_PPM
                ),
                "maximum_trajectory_residual_samples": (
                    CLOCK_TRAJECTORY_MAX_RESIDUAL_SAMPLES
                ),
                "maximum_relative_delay_spread_samples": (
                    RELATIVE_DELAY_MAX_SPREAD_SAMPLES
                ),
            },
            "adc_certification_peak": ADC_CERTIFICATION_PEAK,
            "adc_absolute_peak_ceiling": ADC_ABSOLUTE_PEAK_CEILING,
            "predictive_stop_peak": PREDICTIVE_STOP_PEAK,
            "predictive_uncertainty_factor": PREDICTIVE_UNCERTAINTY_FACTOR,
            "compression_gate_db": COMPRESSION_GATE_DB,
            "thd_imd_gate_dbc": THD_IMD_GATE_DBC,
            "distortion_observability": {
                "coherent_subwindow_seconds": (
                    DISTORTION_COHERENT_SUBWINDOW_SECONDS
                ),
                "minimum_fundamental_snr_db": (
                    DISTORTION_MIN_FUNDAMENTAL_SNR_DB
                ),
                "minimum_noise_margin_below_gate_db": (
                    DISTORTION_MIN_NOISE_MARGIN_DB
                ),
            },
            "safety_operator": {
                "role": "source_gain_prediction_only_not_anc_plant_authority",
                "band_hz": list(ESS_BAND_HZ),
                "fir_length": OPERATOR_FIR_LENGTH,
                "peak_pre_roll_samples": OPERATOR_PEAK_PRE_ROLL_SAMPLES,
                "fit_levels_millionths": list(OPERATOR_FIT_LEVELS_MILLIONTHS),
                "holdout_level_millionths": OPERATOR_HOLDOUT_LEVEL_MILLIONTHS,
                "subbands_hz": [list(value) for value in OPERATOR_SUBBANDS_HZ],
                "minimum_complex_agreement": OPERATOR_MIN_COMPLEX_AGREEMENT,
                "maximum_relative_error": OPERATOR_MAX_RELATIVE_ERROR,
                "relative_subband_minimum_target_norm_ratio": (
                    OPERATOR_RELATIVE_SUBBAND_MIN_NORM_RATIO
                ),
                "residual_uncertainty_factor": PREDICTIVE_UNCERTAINTY_FACTOR,
            },
        },
        "layout": layout,
        "clock_pilot_layout": clock_pilot_layout,
        "capture_groups": capture_groups,
        "duration": {
            "nominal_active_seconds": active_seconds,
            "exact_nonzero_pcm_seconds": exact_nonzero_pcm_seconds,
            "output_open_seconds": output_float.shape[0] / SAMPLE_RATE,
            "input_preflight_seconds": INPUT_PREFLIGHT_SECONDS,
            # level당 한 6.5초 stream + 1초 watchdog/1초 fingerprint·전환 예산.
            # 16 slot별 open/close가 아니라 정확히 네 stream만 연다.
            "stream_open_count": len(capture_groups),
            "per_stream_watchdog_grace_seconds": STREAM_WATCHDOG_GRACE_SECONDS,
            "per_stream_transition_budget_seconds": STREAM_TRANSITION_BUDGET_SECONDS,
            "live_campaign_hard_deadline_seconds": (
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
    clock_pilot_layout = value.get("clock_pilot_layout")
    capture_groups = value.get("capture_groups")
    if (
        not isinstance(contract, Mapping)
        or contract.get("levels_millionths") != list(LEVELS_MILLIONTHS)
        or contract.get("drive") != "NS_noise_out_ch0_only"
        or contract.get("cancel_output_exact_zero") is not True
        or not isinstance(output, Mapping)
        or not isinstance(layout, list)
        or len(layout) != len(LEVELS_MILLIONTHS) * (1 + len(IMD_PAIRS_HZ))
        or not isinstance(clock_pilot_layout, list)
        or len(clock_pilot_layout)
        != len(LEVELS_MILLIONTHS) * len(CLOCK_PILOT_OFFSETS_SECONDS)
        or not isinstance(capture_groups, list)
        or len(capture_groups) != len(LEVELS_MILLIONTHS)
    ):
        raise RecordingGainLinearityError("gain-linearity plan 고정 계약 불일치")
    frames_total = int(output.get("frames", -1))
    if frames_total != 26 * SAMPLE_RATE:
        raise RecordingGainLinearityError("gain-linearity v3 frame 수 불일치")
    reconstructed = np.zeros((frames_total, 2), dtype=np.float32)
    for row in layout:
        level = int(row["level_millionths"])
        peak = level / 1_000_000.0
        if row["kind"] == "ESS":
            source, _ = _synchronised_ess(peak)
        elif row["kind"] == "IMD":
            source = _imd(tuple(float(v) for v in row["pair_hz"]), peak)
        else:
            raise RecordingGainLinearityError("gain-linearity layout kind 불일치")
        if int(row["active_frames"]) != source.size:
            raise RecordingGainLinearityError("gain-linearity active frame 불일치")
        start = int(row["start_frame"])
        reconstructed[start : start + source.size, 0] = source
    for group_index, group in enumerate(capture_groups):
        level = int(group["level_millionths"])
        pilot = _clock_pilot(level / 1_000_000.0)
        expected_order = [
            list(IMD_PAIRS_HZ[(group_index + offset) % len(IMD_PAIRS_HZ)])
            for offset in range(len(IMD_PAIRS_HZ))
        ]
        if (
            group.get("settle_frames")
            != int(round(GROUP_SETTLE_SECONDS * SAMPLE_RATE))
            or group.get("stimulus_start_frame")
            != int(group["start_frame"]) + int(group["settle_frames"])
            or group.get("clock_pilot_active_frames") != CLOCK_PILOT_FRAMES
            or group.get("imd_pair_order_hz") != expected_order
        ):
            raise RecordingGainLinearityError("gain-linearity group settle/rotation 불일치")
        settle = reconstructed[
            int(group["start_frame"]) : int(group["stimulus_start_frame"])
        ]
        if np.any(settle != 0.0):
            raise RecordingGainLinearityError("gain-linearity group settle이 exact zero가 아닙니다")
        pilot_count = len(CLOCK_PILOT_OFFSETS_SECONDS)
        group_pilot_rows = clock_pilot_layout[
            group_index * pilot_count : (group_index + 1) * pilot_count
        ]
        if [row.get("start_frame") for row in group_pilot_rows] != group.get(
            "clock_pilot_start_frames"
        ):
            raise RecordingGainLinearityError("clock pilot layout/group 불일치")
        active_intervals = [
            (
                int(row["start_frame"]),
                int(row["active_stop_frame"]),
                str(row["kind"]),
            )
            for row in layout[
                group_index * (1 + len(IMD_PAIRS_HZ)) :
                (group_index + 1) * (1 + len(IMD_PAIRS_HZ))
            ]
        ]
        for pilot_row in group_pilot_rows:
            pilot_start = int(pilot_row["start_frame"])
            pilot_stop = int(pilot_row["stop_frame"])
            expected_pilot_pcm = _float_to_pcm16(pilot)
            if (
                pilot_row.get("kind") != "CLOCK_PILOT"
                or pilot_row.get("level_millionths") != level
                or pilot_row.get("active_frames") != CLOCK_PILOT_FRAMES
                or pilot_stop - pilot_start != CLOCK_PILOT_FRAMES
                or pilot_row.get("noise_ch0_pcm_sha256")
                != _sha256_bytes(expected_pilot_pcm.tobytes(order="C"))
                or any(
                    pilot_start < active_stop and active_start < pilot_stop
                    for active_start, active_stop, _kind in active_intervals
                )
            ):
                raise RecordingGainLinearityError("clock pilot overlap/SHA 계약 불일치")
        for pilot_start in group.get("clock_pilot_start_frames", []):
            start = int(pilot_start)
            reconstructed[start : start + pilot.size, 0] = pilot
        response_intervals = [
            *active_intervals,
            *[
                (int(row["start_frame"]), int(row["stop_frame"]), "CLOCK_PILOT")
                for row in group_pilot_rows
            ],
        ]
        response_intervals.sort(key=lambda item: item[0])
        for current, following in zip(response_intervals, response_intervals[1:]):
            if current[1] + MAX_DELAY_SAMPLES > following[0]:
                raise RecordingGainLinearityError(
                    f"{current[2]} response guard가 {following[2]}를 침범합니다"
                )
        if response_intervals[-1][1] + MAX_DELAY_SAMPLES > int(group["stop_frame"]):
            raise RecordingGainLinearityError("마지막 clock pilot response guard가 부족합니다")
    pcm = _float_to_pcm16(reconstructed)
    if (
        pcm.shape != (int(output.get("frames", -1)), 2)
        or _sha256_bytes(pcm.tobytes(order="C")) != output.get("pcm_sha256")
        or np.any(pcm[:, 1] != 0)
    ):
        raise RecordingGainLinearityError("gain-linearity plan PCM 재구성 불일치")
    expected_groups = []
    group_frames = int(round((6.0 + GROUP_SETTLE_SECONDS) * SAMPLE_RATE))
    for group_index, level in enumerate(LEVELS_MILLIONTHS):
        rows = layout[
            group_index * (1 + len(IMD_PAIRS_HZ)) :
            (group_index + 1) * (1 + len(IMD_PAIRS_HZ))
        ]
        expected_groups.append(
            {
                "level_millionths": level,
                "start_frame": group_index * group_frames,
                "settle_frames": int(round(GROUP_SETTLE_SECONDS * SAMPLE_RATE)),
                "stimulus_start_frame": (
                    group_index * group_frames
                    + int(round(GROUP_SETTLE_SECONDS * SAMPLE_RATE))
                ),
                "stop_frame": (group_index + 1) * group_frames,
                "clock_pilot_start_frames": [
                    group_index * group_frames
                    + int(round(GROUP_SETTLE_SECONDS * SAMPLE_RATE))
                    + int(round(offset * SAMPLE_RATE))
                    for offset in CLOCK_PILOT_OFFSETS_SECONDS
                ],
                "clock_pilot_active_frames": CLOCK_PILOT_FRAMES,
                "imd_pair_order_hz": [
                    list(IMD_PAIRS_HZ[(group_index + offset) % len(IMD_PAIRS_HZ)])
                    for offset in range(len(IMD_PAIRS_HZ))
                ],
                "first_layout_index": group_index * (1 + len(IMD_PAIRS_HZ)),
                "layout_count": 1 + len(IMD_PAIRS_HZ),
            }
        )
    if capture_groups != expected_groups:
        raise RecordingGainLinearityError("gain-linearity capture group 계약 불일치")
    nominal_active_seconds = (
        len(LEVELS_MILLIONTHS) * float(contract["ess"]["active_seconds"])
        + len(LEVELS_MILLIONTHS) * len(IMD_PAIRS_HZ) * IMD_ACTIVE_SECONDS
        + len(LEVELS_MILLIONTHS)
        * len(CLOCK_PILOT_OFFSETS_SECONDS)
        * CLOCK_PILOT_FRAMES
        / SAMPLE_RATE
    )
    expected_duration = {
        "nominal_active_seconds": nominal_active_seconds,
        "exact_nonzero_pcm_seconds": float(np.count_nonzero(pcm[:, 0]))
        / SAMPLE_RATE,
        "output_open_seconds": pcm.shape[0] / SAMPLE_RATE,
        "input_preflight_seconds": INPUT_PREFLIGHT_SECONDS,
        "stream_open_count": len(capture_groups),
        "per_stream_watchdog_grace_seconds": STREAM_WATCHDOG_GRACE_SECONDS,
        "per_stream_transition_budget_seconds": STREAM_TRANSITION_BUDGET_SECONDS,
        "live_campaign_hard_deadline_seconds": (
            INPUT_PREFLIGHT_SECONDS
            + pcm.shape[0] / SAMPLE_RATE
            + len(capture_groups)
            * (
                STREAM_WATCHDOG_GRACE_SECONDS
                + STREAM_TRANSITION_BUDGET_SECONDS
            )
        ),
    }
    if value.get("duration") != expected_duration:
        raise RecordingGainLinearityError("gain-linearity duration exact 계약 불일치")
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


def _fractional_delay_and_correlation(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    """Bounded correlation peak를 quadratic 보간해 fractional delay를 낸다."""

    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    correlation = signal.fftconvolve(y, x[::-1], mode="full")
    start = x.size - 1
    window = correlation[start : start + MAX_DELAY_SAMPLES + 1]
    if window.size != MAX_DELAY_SAMPLES + 1:
        raise RecordingGainLinearityError("clock pilot delay 탐색 window가 짧습니다")
    magnitude = np.abs(window)
    index = int(np.argmax(magnitude))
    fraction = 0.0
    if 0 < index < magnitude.size - 1:
        left, centre, right = (
            float(magnitude[index - 1]),
            float(magnitude[index]),
            float(magnitude[index + 1]),
        )
        denominator = left - 2.0 * centre + right
        if denominator != 0.0:
            fraction = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    delay = float(index + fraction)
    count = min(x.size, y.size - index)
    if count <= 0:
        return delay, 0.0
    observed = y[index : index + count]
    reference = x[:count]
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(observed))
    score = abs(float(np.dot(reference, observed))) / denominator if denominator else 0.0
    return delay, score


def _fractional_signed_lag_and_correlation(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    maximum_abs_lag: int,
) -> tuple[float, float]:
    """같은 plant를 지난 repeat response 사이 signed lag와 관측 score를 낸다."""

    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 3:
        raise RecordingGainLinearityError("clock pilot repeat response shape 불일치")
    bound = int(maximum_abs_lag)
    if bound <= 0 or bound >= x.size - 1:
        raise RecordingGainLinearityError("clock pilot signed-lag bound가 유효하지 않습니다")
    correlation = signal.fftconvolve(y, x[::-1], mode="full")
    lags = np.arange(-x.size + 1, y.size, dtype=np.int64)
    selected = np.flatnonzero((lags >= -bound) & (lags <= bound))
    magnitude = np.abs(correlation[selected])
    local_index = int(np.argmax(magnitude))
    correlation_index = int(selected[local_index])
    integer_lag = int(lags[correlation_index])
    fraction = 0.0
    if 0 < correlation_index < correlation.size - 1:
        left = float(abs(correlation[correlation_index - 1]))
        centre = float(abs(correlation[correlation_index]))
        right = float(abs(correlation[correlation_index + 1]))
        denominator = left - 2.0 * centre + right
        if denominator != 0.0:
            fraction = float(
                np.clip(0.5 * (left - right) / denominator, -0.5, 0.5)
            )
    if integer_lag >= 0:
        observed = y[integer_lag:]
        expected = x[: observed.size]
    else:
        expected = x[-integer_lag:]
        observed = y[: expected.size]
    denominator = float(np.linalg.norm(expected) * np.linalg.norm(observed))
    score = (
        abs(float(np.dot(expected, observed))) / denominator
        if denominator > 0.0
        else 0.0
    )
    return float(integer_lag + fraction), score


def _callback_time_witness(
    value: Mapping[str, np.ndarray], *, expected_frames: int
) -> dict[str, Any]:
    """Callback sample coverage와 ADC/DAC monotonic timebase를 raw 배열에서 재검산한다.

    PortAudio ``time_info``는 callback이 본 host-side buffer timestamp다. 특히
    ``outputBufferDacTime``은 driver queue 예측이 섞여 callback별 step이 일정하지 않으므로
    이 값의 회귀 기울기를 ADC/DAC clock-q authority로 쓰지 않는다. 실제 q는 캡처 PCM에
    삽입한 다섯 clock pilot response의 trajectory로 별도 판정한다.
    """

    if not isinstance(value, Mapping):
        raise RecordingGainLinearityError("callback time witness가 mapping이 아닙니다")
    arrays = {
        name: np.asarray(value.get(name)).reshape(-1)
        for name in CALLBACK_TIME_INFO_FIELDS
    }
    sizes = {array.size for array in arrays.values()}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) < 2:
        raise RecordingGainLinearityError("callback time witness 길이 불일치")
    starts = arrays["callback_start_frames"]
    counts = arrays["callback_frame_counts"]
    if starts.dtype.kind not in "iu" or counts.dtype.kind not in "iu":
        raise RecordingGainLinearityError("callback frame witness가 integer가 아닙니다")
    starts = starts.astype(np.int64)
    counts = counts.astype(np.int64)
    if (
        starts[0] != 0
        or np.any(counts <= 0)
        or np.any(counts != BLOCK_SIZE)
        or np.any(starts < 0)
        or np.any(np.diff(starts) != np.minimum(counts[:-1], expected_frames - starts[:-1]))
        or int(starts[-1]) + min(int(counts[-1]), expected_frames - int(starts[-1]))
        != expected_frames
    ):
        raise RecordingGainLinearityError("callback frame coverage/slip witness 위반")
    informational_time_rates: dict[str, float] = {}
    for name in (
        "input_buffer_adc_time",
        "output_buffer_dac_time",
        "callback_current_time",
    ):
        times = arrays[name].astype(np.float64)
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise RecordingGainLinearityError(f"{name}가 finite monotonic이 아닙니다")
        slope = float(np.polyfit(starts.astype(np.float64), times, 1)[0])
        rate_ratio = slope * SAMPLE_RATE
        ppm = (rate_ratio - 1.0) * 1.0e6
        if not math.isfinite(ppm):
            raise RecordingGainLinearityError(f"{name} callback 회귀 기울기가 유한하지 않습니다")
        informational_time_rates[name] = ppm
    return {
        "valid": True,
        "callback_count": int(starts.size),
        "software_frame_gap_count": 0,
        "hardware_sample_slip_authority": False,
        "portaudio_time_rate_authority": False,
        "informational_fit_rate_ppm": informational_time_rates,
        "role": "callback_monotonic_and_sample_coverage_witness_not_clock_q_authority",
    }


def callback_time_info_evidence(
    value: Mapping[str, np.ndarray], *, group_index: int, expected_frames: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Live publisher가 JSON metadata와 NPZ raw array를 분리해 봉인하게 한다."""

    witness = _callback_time_witness(value, expected_frames=expected_frames)
    arrays: dict[str, np.ndarray] = {}
    refs: dict[str, Any] = {}
    for field in CALLBACK_TIME_INFO_FIELDS:
        dtype = "<i8" if field in {"callback_start_frames", "callback_frame_counts"} else "<f8"
        array = np.ascontiguousarray(np.asarray(value[field]), dtype=dtype)
        key = f"callback_group_{int(group_index)}_{field}"
        arrays[key] = array
        refs[field] = {
            "array_key": key,
            "dtype": dtype,
            "shape": [int(array.size)],
            "sha256": _sha256_bytes(array.tobytes(order="C")),
        }
    return {"summary": witness, "arrays": refs}, arrays


def _load_callback_time_info(
    evidence: Any,
    arrays: Mapping[str, np.ndarray],
    *,
    expected_frames: int,
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping) or set(evidence) != {"summary", "arrays"}:
        raise RecordingGainLinearityError("callback time evidence schema 불일치")
    refs = evidence.get("arrays")
    if not isinstance(refs, Mapping) or set(refs) != set(CALLBACK_TIME_INFO_FIELDS):
        raise RecordingGainLinearityError("callback time raw ref 집합 불일치")
    loaded: dict[str, np.ndarray] = {}
    for field in CALLBACK_TIME_INFO_FIELDS:
        ref = refs[field]
        dtype = "<i8" if field in {"callback_start_frames", "callback_frame_counts"} else "<f8"
        if not isinstance(ref, Mapping) or set(ref) != {"array_key", "dtype", "shape", "sha256"}:
            raise RecordingGainLinearityError("callback time raw ref schema 불일치")
        key = ref.get("array_key")
        array = arrays.get(key) if isinstance(key, str) else None
        if (
            array is None
            or ref.get("dtype") != dtype
            or ref.get("shape") != [int(np.asarray(array).size)]
        ):
            raise RecordingGainLinearityError("callback time raw dtype/shape 불일치")
        canonical = np.ascontiguousarray(np.asarray(array), dtype=dtype)
        if _sha256_bytes(canonical.tobytes(order="C")) != ref.get("sha256"):
            raise RecordingGainLinearityError("callback time raw SHA 불일치")
        loaded[field] = canonical
    rebuilt = _callback_time_witness(loaded, expected_frames=expected_frames)
    if evidence.get("summary") != rebuilt:
        raise RecordingGainLinearityError("callback time summary 독립 재계산 불일치")
    return rebuilt


def _group_clock_alignment(
    *,
    payload: Mapping[str, Any],
    source_float: np.ndarray,
    input_float: np.ndarray,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    """Plant-colored repeat response로 group별 affine q와 공통 offset을 관측한다.

    source pilot와 mic response의 직접 correlation은 덕트의 coloration 때문에 낮을 수
    있으므로 관측 gate로 쓰지 않는다. 첫 response와 뒤 repeat response들을 서로 비교해
    clock trajectory를 얻고, source→첫 response peak는 absolute offset에만 사용한다.
    """

    result: dict[int, dict[str, Any]] = {}
    reasons: list[str] = []
    relative_values: list[float] = []
    for group_index, group in enumerate(payload["capture_groups"]):
        pilot_starts = [int(value) for value in group["clock_pilot_start_frames"]]
        if len(pilot_starts) != len(CLOCK_PILOT_OFFSETS_SECONDS):
            raise RecordingGainLinearityError("clock pilot start 개수가 exact 계약과 다릅니다")
        pilot = np.asarray(
            source_float[
                pilot_starts[0] : pilot_starts[0] + CLOCK_PILOT_FRAMES
            ],
            dtype=np.float64,
        )
        pilot_axis = np.asarray(pilot_starts, dtype=np.float64) - float(
            pilot_starts[0]
        )
        maximum_repeat_lag = int(
            math.ceil(
                float(pilot_axis[-1]) * CLOCK_MAX_ABS_PPM / 1.0e6
            )
        ) + 16
        channels: dict[str, Any] = {}
        for name, channel in (("err", 0), ("ref", 1)):
            response_windows: list[np.ndarray] = []
            for pilot_start in pilot_starts:
                stop = pilot_start + CLOCK_PILOT_FRAMES + MAX_DELAY_SAMPLES
                if stop > int(group["stop_frame"]):
                    raise RecordingGainLinearityError("clock pilot response guard가 부족합니다")
                response_windows.append(
                    np.asarray(input_float[pilot_start:stop, channel], dtype=np.float64)
                )
            absolute_delay, source_template_correlation = (
                _fractional_delay_and_correlation(pilot, response_windows[0])
            )
            relative_lags = [0.0]
            repeat_correlations = [1.0]
            for response in response_windows[1:]:
                lag, correlation = _fractional_signed_lag_and_correlation(
                    response_windows[0],
                    response,
                    maximum_abs_lag=maximum_repeat_lag,
                )
                relative_lags.append(lag)
                repeat_correlations.append(correlation)
            relative_array = np.asarray(relative_lags, dtype=np.float64)
            slope, intercept = np.polyfit(pilot_axis, relative_array, 1)
            trajectory_residual = relative_array - (
                slope * pilot_axis + intercept
            )
            delays = absolute_delay + relative_array
            q = 1.0 + float(slope)
            ppm = (q - 1.0) * 1.0e6
            if min(repeat_correlations) < CLOCK_PILOT_MIN_NORMALISED_CORRELATION:
                reasons.append(f"group_{group_index}_{name}_pilot_correlation")
            maximum_residual = float(np.max(np.abs(trajectory_residual)))
            if (
                not math.isfinite(maximum_residual)
                or maximum_residual > CLOCK_TRAJECTORY_MAX_RESIDUAL_SAMPLES
            ):
                reasons.append(f"group_{group_index}_{name}_clock_trajectory")
            if not math.isfinite(ppm) or abs(ppm) > CLOCK_MAX_ABS_PPM:
                reasons.append(f"group_{group_index}_{name}_clock_q")
            channels[name] = {
                "pilot_delay_samples": [float(value) for value in delays],
                "repeat_response_lag_samples": relative_lags,
                "repeat_response_normalised_correlation": repeat_correlations,
                "source_template_first_delay_samples": absolute_delay,
                "source_template_first_normalised_correlation_informational": (
                    source_template_correlation
                ),
                "trajectory_residual_samples": [
                    float(value) for value in trajectory_residual
                ],
                "maximum_abs_trajectory_residual_samples": maximum_residual,
                "q_ratio": q,
                "rate_ppm": ppm,
            }
        q_values = [channels[name]["q_ratio"] for name in ("err", "ref")]
        if abs(q_values[0] - q_values[1]) * 1.0e6 > CLOCK_CHANNEL_MAX_DIFFERENCE_PPM:
            reasons.append(f"group_{group_index}_channel_clock_q_disagreement")
        relative_delays = [
            float(err_value - ref_value)
            for err_value, ref_value in zip(
                channels["err"]["pilot_delay_samples"],
                channels["ref"]["pilot_delay_samples"],
                strict=True,
            )
        ]
        relative_values.extend(relative_delays)
        observed_common_response_offset = min(
            channels["err"]["pilot_delay_samples"][0],
            channels["ref"]["pilot_delay_samples"][0],
        )
        result[group_index] = {
            "group_index": group_index,
            "level_millionths": int(group["level_millionths"]),
            "capture_timeline_offset_samples": 0.0,
            "observed_common_response_offset_samples": (
                observed_common_response_offset
            ),
            "common_q_ratio": float(np.median(q_values)),
            "relative_delay_samples": relative_delays,
            "channels": channels,
        }
    relative_spread = float(np.ptp(relative_values)) if relative_values else math.inf
    if not math.isfinite(relative_spread) or relative_spread > RELATIVE_DELAY_MAX_SPREAD_SAMPLES:
        reasons.append("err_ref_relative_delay_spread")
    for item in result.values():
        item["all_groups_relative_delay_spread_samples"] = relative_spread
    return result, reasons


def _group_ref_witness_alignment(
    *,
    payload: Mapping[str, Any],
    source_float: np.ndarray,
    input_float: np.ndarray,
) -> tuple[np.ndarray, dict[int, dict[str, Any]], list[str]]:
    """REF witness로 각 reopened stream의 비선형 USB clock hunting을 보정한다.

    affine ``q``를 다시 맞추지 않는다. :mod:`deep_anc.data.timeline`의
    ``REF witness -> non-affine warp -> ERR holdout`` 경로만 사용한다. 이 probe는
    의도적으로 sparse하므로 전체 창 대비 0.90 비율을 요구하는 15초 자연음 수집
    계약을 잘못 적용하지 않는다. 대신 canonical 최소 유효창 8개와 같은 저/고역 및
    REF-ERR coherence 하한을 그대로 적용한다.
    """

    aligned_source = np.asarray(source_float, dtype=np.float64).copy()
    result: dict[int, dict[str, Any]] = {}
    reasons: list[str] = []
    settings = TimelineSettings(sample_rate=SAMPLE_RATE)
    for group_index, group in enumerate(payload["capture_groups"]):
        start = int(group["start_frame"])
        stop = int(group["stop_frame"])
        source = np.asarray(source_float[start:stop], dtype=np.float64)
        ref = np.asarray(input_float[start:stop, 1], dtype=np.float64)
        err = np.asarray(input_float[start:stop, 0], dtype=np.float64)
        try:
            track = estimate_lag_track(source, ref, settings)
            aligned, report = align_source_to_adc(
                source, ref, err, SAMPLE_RATE, settings=settings
            )
        except (TypeError, ValueError) as exc:
            raise RecordingGainLinearityError(
                f"group {group_index} REF witness timeline 실패: {exc}"
            ) from exc
        canonical = np.ascontiguousarray(aligned, dtype="<f4")
        aligned_source[start:stop] = canonical.astype(np.float64)
        if track.valid_count < MIN_STREAM_DELAY_VALID_WINDOWS:
            reasons.append(f"group_{group_index}_timeline_valid_windows")
        if report.coh2_150_600_after < CAPTURE_MIN_LOW_BAND_COHERENCE:
            reasons.append(f"group_{group_index}_timeline_lowband_coherence")
        if report.coh2_600_1600_after < MIN_STREAM_COHERENCE:
            reasons.append(f"group_{group_index}_timeline_highband_coherence")
        if report.coh2_ref_err_150_600 < MIN_REF_ERR_COHERENCE:
            reasons.append(f"group_{group_index}_timeline_ref_err_coherence")
        result[group_index] = {
            "group_index": group_index,
            "level_millionths": int(group["level_millionths"]),
            "method": TIMELINE_METHOD,
            "witness_channel": 1,
            "holdout_channel": 0,
            "affine_q_used": False,
            "valid_windows": int(track.valid_count),
            "track": track.summary(),
            "report": report.as_metadata(),
            "aligned_source_encoding": "float32_le",
            "aligned_source_sha256": _sha256_bytes(canonical.tobytes(order="C")),
            # Existing metric extraction consumes a channel-view mapping. Once the
            # digital source has moved to the ADC timebase, the two ADC channels
            # must remain untouched and therefore use an exact identity view.
            "common_q_ratio": 1.0,
        }
    return aligned_source, result, reasons


def _build_peak_safety_envelope(
    *,
    peak_gain_upper: Mapping[str, float],
    supported_max_amplitude_millionths: int,
) -> dict[str, Any]:
    """Bad complex-FIR fit을 peak-safety authority로 오인하지 않는 envelope.

    기존 0.995/0.10 complex round-trip 임계는 그대로 보존되고 diagnostic FIR에
    적용된다. 이 별도 authority는 네 measured level의 실제 group peak와 1.25배
    uncertainty에서만 유도되며 tested range 밖 외삽, THD 또는 ANC plant 권한을 주지
    않는다.
    """

    channels: dict[str, Any] = {}
    for name in ("err", "ref"):
        upper = float(peak_gain_upper[name])
        if not math.isfinite(upper) or upper <= 0.0:
            raise RecordingGainLinearityError("peak safety envelope gain이 유효하지 않습니다")
        channels[name] = {
            "peak_gain_upper_with_uncertainty": upper,
            "uncertainty_factor": PREDICTIVE_UNCERTAINTY_FACTOR,
            "valid_through_amplitude_millionths": int(
                LEVELS_MILLIONTHS[-1]
            ),
            "prediction": "upper_peak=gain_upper*rendered_source_peak",
        }
    envelope: dict[str, Any] = {
        "schema": GAIN_LINEARITY_PEAK_ENVELOPE_SCHEMA,
        "role": "source_gain_peak_envelope_only_not_anc_plant_authority",
        "fit_levels_millionths": list(LEVELS_MILLIONTHS[:-1]),
        "independent_holdout_level_millionths": LEVELS_MILLIONTHS[-1],
        "tested_max_amplitude_millionths": LEVELS_MILLIONTHS[-1],
        "supported_max_amplitude_millionths": int(
            supported_max_amplitude_millionths
        ),
        "complex_operator_thresholds_relaxed": False,
        "complex_operator_used_as_authority": False,
        "channels": channels,
    }
    envelope["operator_sha256"] = _seal(envelope)
    return envelope


def _aligned_channel_view(
    values: np.ndarray,
    *,
    group: Mapping[str, Any],
    alignment: Mapping[str, Any],
    output_start: int,
    output_stop: int,
) -> np.ndarray:
    """공통 q/offset만 적용해 두 마이크 사이 상대 지연은 보존한다."""

    start = int(group["start_frame"])
    stop = int(group["stop_frame"])
    source_axis = np.arange(start, stop, dtype=np.float64)
    query_output = np.arange(output_start, output_stop, dtype=np.float64)
    # submitted/recorded arrays는 같은 duplex callback cursor로 시작한다. Pilot에서
    # 보이는 source→mic offset은 plant acoustic delay이며 timeline offset이 아니다.
    # 이를 여기서 제거하면 colored FIR의 peak 이전 에너지가 잘려 safety operator가
    # 비인과적으로 된다. Clock-rate correction만 group stream 시작점에 anchor한다.
    query = float(start) + float(alignment["common_q_ratio"]) * (
        query_output - float(start)
    )
    if query.size == 0 or query[0] < source_axis[0] or query[-1] > source_axis[-1]:
        raise RecordingGainLinearityError("clock-corrected interpolation support가 부족합니다")
    spline = interpolate.CubicSpline(
        source_axis,
        np.asarray(values[start:stop], dtype=np.float64),
        extrapolate=False,
    )
    result = np.asarray(spline(query), dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise RecordingGainLinearityError("clock-corrected cubic interpolation이 non-finite입니다")
    return result


def _expanded_bin_indices(spectrum_size: int, bins: set[int]) -> set[int]:
    selected: set[int] = set()
    for index in bins:
        for offset in (-1, 0, 1):
            candidate = index + offset
            if 0 < candidate < spectrum_size:
                selected.add(candidate)
    return selected


def _bin_power(spectrum: np.ndarray, bins: set[int]) -> float:
    selected = _expanded_bin_indices(spectrum.size, bins)
    if not selected:
        return 0.0
    return float(np.sum(np.abs(spectrum[sorted(selected)]) ** 2))


def _coherent_average_power(values: np.ndarray) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64)
    frames = int(round(DISTORTION_COHERENT_SUBWINDOW_SECONDS * SAMPLE_RATE))
    if frames <= 0 or samples.size < frames or samples.size % frames != 0:
        raise RecordingGainLinearityError("distortion coherent subwindow 길이 불일치")
    windows = samples.reshape(-1, frames)
    spectra = np.fft.rfft(windows, axis=1)
    return np.mean(np.square(np.abs(spectra)), axis=0)


def _distortion_metrics(
    values: np.ndarray,
    pair: tuple[float, float],
    *,
    preflight_values: np.ndarray,
) -> dict[str, Any]:
    spectrum = np.sqrt(_coherent_average_power(values)).astype(np.complex128)
    preflight = np.sqrt(
        _coherent_average_power(
            np.asarray(preflight_values, dtype=np.float64)[
                : int(round(IMD_ANALYSIS_SECONDS * SAMPLE_RATE))
            ]
        )
    ).astype(np.complex128)
    frames = int(round(DISTORTION_COHERENT_SUBWINDOW_SECONDS * SAMPLE_RATE))
    resolution = SAMPLE_RATE / frames

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
    signal_bins = fundamentals | harmonics | products
    expanded_signal_bins = _expanded_bin_indices(spectrum.size, signal_bins)
    neighbourhood: set[int] = set()
    for signal_bin in signal_bins:
        neighbourhood.update(
            candidate
            for candidate in range(signal_bin - 12, signal_bin + 13)
            if 0 < candidate < spectrum.size
            and candidate not in expanded_signal_bins
        )
    capture_noise_per_bin = (
        float(
            np.quantile(
                np.abs(spectrum[sorted(neighbourhood)]) ** 2,
                0.95,
            )
        )
        if neighbourhood
        else 0.0
    )

    def matched_noise_upper(bins: set[int]) -> tuple[float, int]:
        selected = _expanded_bin_indices(spectrum.size, bins)
        capture_upper = capture_noise_per_bin * len(selected)
        preflight_upper = (
            float(np.sum(np.abs(preflight[sorted(selected)]) ** 2))
            if selected
            else 0.0
        )
        return max(capture_upper, preflight_upper), len(selected)

    fundamental_noise, fundamental_noise_bin_count = matched_noise_upper(fundamentals)
    thd_noise, thd_noise_bin_count = matched_noise_upper(harmonics)
    imd_noise, imd_noise_bin_count = matched_noise_upper(products)
    floor = np.finfo(np.float64).tiny
    denominator = max(fundamental_power, floor)
    thd = 10.0 * math.log10(
        max(_bin_power(spectrum, harmonics), floor) / denominator
    )
    imd = 10.0 * math.log10(
        max(_bin_power(spectrum, products), floor) / denominator
    )
    fundamental_snr = 10.0 * math.log10(
        denominator / max(fundamental_noise, floor)
    )
    thd_noise_dbc = 10.0 * math.log10(max(thd_noise, floor) / denominator)
    imd_noise_dbc = 10.0 * math.log10(max(imd_noise, floor) / denominator)
    thd_noise_margin = THD_IMD_GATE_DBC - thd_noise_dbc
    imd_noise_margin = THD_IMD_GATE_DBC - imd_noise_dbc
    observable = bool(
        math.isfinite(fundamental_snr)
        and fundamental_snr >= DISTORTION_MIN_FUNDAMENTAL_SNR_DB
        and thd_noise_margin >= DISTORTION_MIN_NOISE_MARGIN_DB
        and imd_noise_margin >= DISTORTION_MIN_NOISE_MARGIN_DB
    )
    verdict = (
        "INCONCLUSIVE"
        if not observable
        else ("PASS" if thd <= THD_IMD_GATE_DBC and imd <= THD_IMD_GATE_DBC else "FAIL")
    )
    return {
        "thd_dbc": float(thd),
        "imd_dbc": float(imd),
        "fundamental_snr_db": float(fundamental_snr),
        "capture_noise_per_bin_power_95th": float(capture_noise_per_bin),
        "fundamental_noise_bin_count": fundamental_noise_bin_count,
        "thd_noise_bin_count": thd_noise_bin_count,
        "imd_noise_bin_count": imd_noise_bin_count,
        "thd_matched_noise_dbc": float(thd_noise_dbc),
        "imd_matched_noise_dbc": float(imd_noise_dbc),
        "thd_noise_margin_below_gate_db": float(thd_noise_margin),
        "imd_noise_margin_below_gate_db": float(imd_noise_margin),
        "coherent_subwindow_seconds": DISTORTION_COHERENT_SUBWINDOW_SECONDS,
        "observable": observable,
        "verdict": verdict,
    }


def _operator_target(
    *,
    input_float: np.ndarray,
    source_float: np.ndarray,
    row: Mapping[str, Any],
    channel: int,
    group: Mapping[str, Any],
    alignment: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, int]:
    """한 ESS slot에서 source와 delay-aligned causal response를 돌려준다."""

    start = int(row["start_frame"])
    stop = int(row["stop_frame"])
    active = int(row["active_frames"])
    source = np.asarray(source_float[start : start + active], dtype=np.float64)
    slot = _aligned_channel_view(
        input_float[:, channel],
        group=group,
        alignment=alignment,
        output_start=start,
        output_stop=stop,
    )
    correlation_peak_delay = _delay_samples(source, slot)
    # A colored duct response의 correlation argmax는 acoustic onset가 아니라 FIR의
    # dominant peak다. Peak부터 자르면 그 이전 에너지가 non-causal loss가 된다.
    # Compact plant와 같은 256-sample pre-roll을 보존해 causal safety FIR을 맞춘다.
    delay = max(0, correlation_peak_delay - OPERATOR_PEAK_PRE_ROLL_SAMPLES)
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
    target: np.ndarray,
    predicted: np.ndarray,
    bounds: tuple[float, float],
    *,
    reference_target_norm: float | None = None,
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
    target_norm_ratio = (
        1.0
        if reference_target_norm is None
        else truth_norm / max(float(reference_target_norm), np.finfo(np.float64).tiny)
    )
    relative_gate_applicable = bool(
        reference_target_norm is None
        or target_norm_ratio >= OPERATOR_RELATIVE_SUBBAND_MIN_NORM_RATIO
    )
    relative_gate_passed = bool(
        math.isfinite(agreement)
        and math.isfinite(relative_error)
        and agreement >= OPERATOR_MIN_COMPLEX_AGREEMENT
        and relative_error <= OPERATOR_MAX_RELATIVE_ERROR
    )
    passed = bool(relative_gate_passed or not relative_gate_applicable)
    return {
        "band_hz": [float(bounds[0]), float(bounds[1])],
        "target_spectrum_norm": truth_norm,
        "target_norm_ratio_to_fullband": float(target_norm_ratio),
        "complex_agreement": agreement,
        "relative_error": relative_error,
        "relative_gate_applicable": relative_gate_applicable,
        "relative_gate_passed": relative_gate_passed,
        "absolute_residual_bound_role": not relative_gate_applicable,
        "passed": passed,
    }


def _operator_example_metrics(
    *, source: np.ndarray, target: np.ndarray, fir: np.ndarray
) -> dict[str, Any]:
    predicted = signal.fftconvolve(source, fir, mode="full")
    fullband = _complex_roundtrip(target, predicted, ESS_BAND_HZ)
    rows = [
        fullband,
        *[
            _complex_roundtrip(
                target,
                predicted,
                bounds,
                reference_target_norm=float(fullband["target_spectrum_norm"]),
            )
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
    *,
    payload: Mapping[str, Any],
    source_float: np.ndarray,
    input_float: np.ndarray,
    clock_alignment: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """독립 6k holdout을 포함한 ERR/REF source-gain 전용 operator를 만든다."""

    ess_rows = {
        int(row["level_millionths"]): row
        for row in payload["layout"]
        if row["kind"] == "ESS"
    }
    if set(ess_rows) != set(LEVELS_MILLIONTHS):
        raise RecordingGainLinearityError("ESS level row 집합이 exact하지 않습니다")
    result: dict[str, Any] = {
        "schema": "recording_gain_safety_operator/v3_gainprobe006",
        "role": "source_gain_prediction_only_not_anc_plant_authority",
        "band_hz": list(ESS_BAND_HZ),
        "fit_levels_millionths": list(OPERATOR_FIT_LEVELS_MILLIONTHS),
        "holdout_level_millionths": OPERATOR_HOLDOUT_LEVEL_MILLIONTHS,
        "fir_length": OPERATOR_FIR_LENGTH,
        "peak_pre_roll_samples": OPERATOR_PEAK_PRE_ROLL_SAMPLES,
        "minimum_complex_agreement": OPERATOR_MIN_COMPLEX_AGREEMENT,
        "maximum_relative_error": OPERATOR_MAX_RELATIVE_ERROR,
        "relative_subband_minimum_target_norm_ratio": (
            OPERATOR_RELATIVE_SUBBAND_MIN_NORM_RATIO
        ),
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
                group=payload["capture_groups"][LEVELS_MILLIONTHS.index(level)],
                alignment=clock_alignment[LEVELS_MILLIONTHS.index(level)],
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


def _load_capture_publication(
    *,
    repo_root: Path,
    publication_path: str,
    expected_sha256: str,
    raw_ref: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """외부 SHA와 sidecar bytes를 포함해 3-leaf 발행을 독립 검증한다."""

    relative = _relative(
        publication_path, label="gain-linearity capture publication path"
    )
    snapshot = _snapshot(
        repo_root, relative, label="gain-linearity capture publication"
    )
    if snapshot.sha256 != _require_sha(
        expected_sha256, label="capture publication expected SHA"
    ):
        raise RecordingGainLinearityError("capture publication 외부 SHA 불일치")
    assert snapshot.data is not None
    try:
        value = json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordingGainLinearityError(
            f"capture publication JSON 오류: {exc}"
        ) from exc
    if not isinstance(value, Mapping) or set(value) != _PUBLICATION_KEYS:
        raise RecordingGainLinearityError("capture publication schema keys 불일치")
    publication = dict(value)
    seal = publication.pop("publication_payload_sha256", None)
    if (
        not isinstance(seal, str)
        or _SHA_RE.fullmatch(seal) is None
        or seal != _seal(publication)
        or value.get("schema") != GAIN_LINEARITY_PUBLICATION_SCHEMA
        or value.get("role")
        != "held_directory_noreplace_raw_sidecar_binding"
    ):
        raise RecordingGainLinearityError("capture publication self-seal/schema 불일치")
    session = _relative(
        str(value.get("canonical_session_path", "")),
        label="capture publication canonical session",
    )
    publication_expected_path = f"{session}/capture_publication.json"
    if relative != publication_expected_path:
        raise RecordingGainLinearityError("capture publication canonical sibling path 불일치")
    raw_expected_path = f"{session}/raw_measurement.npz"
    metadata_expected_path = f"{session}/metadata.json"
    published_raw = value.get("raw")
    published_metadata = value.get("metadata")
    if (
        not isinstance(published_raw, Mapping)
        or set(published_raw) != _PUBLICATION_REF_KEYS
        or not isinstance(published_metadata, Mapping)
        or set(published_metadata) != _PUBLICATION_REF_KEYS
    ):
        raise RecordingGainLinearityError("capture publication file ref schema 불일치")
    # Published inode는 원 장비 witness이고 현재 copy inode와 같을 필요가 없다.
    normalized_raw = _portable_publication_ref(
        published_raw, expected_path=raw_expected_path, label="published raw"
    )
    normalized_metadata = _portable_publication_ref(
        published_metadata,
        expected_path=metadata_expected_path,
        label="published metadata",
    )
    if any(
        normalized_raw[key] != raw_ref.get(key)
        for key in ("path", "size", "sha256")
    ):
        raise RecordingGainLinearityError("capture publication raw binding 불일치")
    metadata_snapshot = _snapshot(
        repo_root, metadata_expected_path, label="gain-linearity metadata sidecar"
    )
    if (
        int(metadata_snapshot.size) != normalized_metadata["size"]
        or str(metadata_snapshot.sha256) != normalized_metadata["sha256"]
    ):
        raise RecordingGainLinearityError("capture publication metadata SHA/size 불일치")
    assert metadata_snapshot.data is not None
    try:
        sidecar = json.loads(metadata_snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordingGainLinearityError(
            f"gain-linearity metadata sidecar JSON 오류: {exc}"
        ) from exc
    if sidecar != dict(metadata):
        raise RecordingGainLinearityError("metadata sidecar와 embedded metadata 불일치")
    if (
        value.get("source_commit") != metadata.get("source_commit")
        or value.get("repository_execution") != metadata.get("repository_execution")
        or value.get("plan") != metadata.get("plan")
    ):
        raise RecordingGainLinearityError("capture publication authority binding 불일치")
    return {
        "file": _file_ref(relative, snapshot),
        "metadata_file": {
            "path": normalized_metadata["path"],
            "size": normalized_metadata["size"],
            "sha256": normalized_metadata["sha256"],
        },
        "payload": dict(value),
    }


def _validate_raw_authority_metadata(
    *,
    repo_root: Path,
    metadata: Mapping[str, Any],
    preflight_raw: np.ndarray | None,
    source_commit: str,
    require_current_capture_execution: bool = True,
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
            or not isinstance(saved_execution.get("repository_branch"), str)
            or not saved_execution.get("repository_branch")
        ):
            reasons.append("repository_execution_identity")
        if require_current_capture_execution:
            try:
                current_execution = repository_execution_identity(
                    repo_root, GAIN_LINEARITY_SCRIPT_PATH
                )
            except (OSError, RuntimeError, ValueError):
                reasons.append("repository_execution_current")
            else:
                # Raw는 Jetson의 named branch에서 캡처한 뒤 같은 exact commit의
                # detached Elice checkout에서 독립 검증한다. branch 이름은 provenance
                # 기록일 뿐 executable bytes의 identity가 아니므로 portable 비교에서
                # 제외하고 commit/clean/script path+blob hash만 exact하게 묶는다.
                portable_keys = _EXECUTION_KEYS - {"repository_branch"}
                if set(current_execution) != _EXECUTION_KEYS or any(
                    saved_execution.get(key) != current_execution.get(key)
                    for key in portable_keys
                ):
                    reasons.append("repository_execution_current")
        else:
            try:
                _verify_historical_capture_execution(repo_root, saved_execution)
            except (OSError, RecordingGainLinearityError, RuntimeError, ValueError):
                reasons.append("historical_capture_execution")

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


def _finalize_gain_linearity_receipt(
    *,
    payload: Mapping[str, Any],
    plan_ref: Mapping[str, Any],
    raw_ref: Mapping[str, Any],
    publication_ref: Mapping[str, Any] | None,
    analysis: Mapping[str, Any],
    reasons: list[str],
    capture_provenance: Mapping[str, Any] | None = None,
    analysis_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    unique_reasons = list(dict.fromkeys(reasons))
    reanalysis = analysis_provenance is not None
    if reanalysis and capture_provenance is None:
        raise RecordingGainLinearityError("재분석 receipt의 capture provenance가 없습니다")
    receipt: dict[str, Any] = {
        "schema": (
            GAIN_LINEARITY_REANALYSIS_RECEIPT_SCHEMA
            if reanalysis
            else GAIN_LINEARITY_RECEIPT_SCHEMA
        ),
        "status": "PASS" if not unique_reasons else "FAIL",
        "source_commit": (
            analysis_provenance["repository_commit"]
            if reanalysis
            else payload["source_commit"]
        ),
        "plan": dict(plan_ref),
        "plan_payload_sha256": payload["plan_payload_sha256"],
        "raw": dict(raw_ref),
        "capture_publication": (
            None if publication_ref is None else dict(publication_ref)
        ),
        "hardware": payload["hardware"],
        "contract": payload["contract"],
        "analysis": dict(analysis),
        "failure_reasons": unique_reasons,
    }
    if reanalysis:
        receipt["capture_provenance"] = json.loads(
            json.dumps(dict(capture_provenance), sort_keys=True, allow_nan=False)
        )
        receipt["analysis_provenance"] = json.loads(
            json.dumps(dict(analysis_provenance), sort_keys=True, allow_nan=False)
        )
    receipt["evidence_sha256"] = _seal(receipt)
    return receipt


def build_gain_linearity_receipt(
    *,
    repo_root: str | Path,
    raw_path: str,
    expected_raw_sha256: str,
    plan_path: str,
    expected_plan_sha256: str,
    publication_path: str | None = None,
    expected_publication_sha256: str | None = None,
    ref_witness_reanalysis: bool = False,
) -> dict[str, Any]:
    """Raw/plan/publication을 독립 재계산하고 PASS/FAIL receipt를 만든다.

    Publication anchor가 없어도 forensic metrics는 계산하지만 canonical PASS는 절대
    발급하지 않는다. 정상 CLI는 path와 외부 SHA를 모두 필수로 전달한다.
    """

    if type(ref_witness_reanalysis) is not bool:
        raise RecordingGainLinearityError("ref_witness_reanalysis는 bool이어야 합니다")
    root = Path(repo_root).resolve()
    analysis_provenance: dict[str, Any] | None = None
    if ref_witness_reanalysis:
        analysis_provenance = _analysis_execution_identity(root)
    plan = load_gain_linearity_plan(
        repo_root=root, plan_path=plan_path, expected_sha256=expected_plan_sha256
    )
    metadata, arrays, raw_ref = _load_raw(
        repo_root=root, raw_path=raw_path, expected_sha256=expected_raw_sha256
    )
    payload = plan["payload"]
    pcm = plan["pcm"]
    capture_provenance = {
        "source_commit": payload["source_commit"],
        "repository_execution": metadata.get("repository_execution"),
    }
    reasons: list[str] = []
    authority_reasons: list[str] = []
    publication_ref: dict[str, Any] | None = None
    if publication_path is None and expected_publication_sha256 is None:
        authority_reasons.append("capture_publication_missing")
    elif publication_path is None or expected_publication_sha256 is None:
        authority_reasons.append("capture_publication_anchor_incomplete")
    else:
        try:
            publication = _load_capture_publication(
                repo_root=root,
                publication_path=publication_path,
                expected_sha256=expected_publication_sha256,
                raw_ref=raw_ref,
                metadata=metadata,
            )
        except (OSError, RecordingGainLinearityError) as exc:
            authority_reasons.append(f"capture_publication_invalid:{exc}")
        else:
            publication_ref = publication["file"]
    if str(raw_ref.get("path", "")).startswith(
        ".deep_anc_live_recovery_"
    ):
        authority_reasons.append("raw_recovery_only")
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
            require_current_capture_execution=not ref_witness_reanalysis,
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
        previous_campaign_elapsed = -math.inf
        for item, group in zip(telemetry, payload["capture_groups"], strict=True):
            expected_group_frames = int(group["stop_frame"]) - int(
                group["start_frame"]
            )
            campaign_elapsed = item.get("live_campaign_elapsed_seconds")
            output_elapsed = item.get("output_elapsed_seconds")
            absolute_deadline = item.get("absolute_deadline_monotonic")
            nominal_output = item.get("nominal_output_seconds")
            hard_max_output = item.get("hard_max_output_seconds")
            expected_nominal_seconds = expected_group_frames / float(SAMPLE_RATE)
            expected_hard_seconds = expected_nominal_seconds + float(
                payload["duration"]["per_stream_watchdog_grace_seconds"]
            )
            if (
                item.get("level_millionths") != group["level_millionths"]
                or item.get("start_frame") != group["start_frame"]
                or item.get("stop_frame") != group["stop_frame"]
                or item.get("completed") is not True
                or int(item.get("xrun_count", -1)) != 0
                or int(item.get("unexpected_status_count", -1)) != 0
                or int(item.get("callback_status_count", -1)) != 0
                or int(item.get("priming_output_count", -1)) != 0
                or item.get("statuses") != []
                or item.get("termination_signal") is not None
                or int(item.get("captured_frames", -1)) != expected_group_frames
                or isinstance(campaign_elapsed, bool)
                or not isinstance(campaign_elapsed, (int, float))
                or not math.isfinite(float(campaign_elapsed))
                or not previous_campaign_elapsed < float(campaign_elapsed)
                or float(campaign_elapsed)
                > float(payload["duration"]["live_campaign_hard_deadline_seconds"])
                or item.get("absolute_deadline_exceeded") is not False
                or item.get("absolute_deadline_abort_error") is not None
                or isinstance(absolute_deadline, bool)
                or not isinstance(absolute_deadline, (int, float))
                or not math.isfinite(float(absolute_deadline))
                or isinstance(output_elapsed, bool)
                or not isinstance(output_elapsed, (int, float))
                or not math.isfinite(float(output_elapsed))
                or float(output_elapsed) < 0.0
                or float(output_elapsed) > expected_hard_seconds
                or isinstance(nominal_output, bool)
                or not isinstance(nominal_output, (int, float))
                or not math.isfinite(float(nominal_output))
                or not math.isclose(
                    float(nominal_output),
                    expected_nominal_seconds,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or isinstance(hard_max_output, bool)
                or not isinstance(hard_max_output, (int, float))
                or not math.isfinite(float(hard_max_output))
                or not math.isclose(
                    float(hard_max_output),
                    expected_hard_seconds,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or item.get("callback_error") is not None
                or item.get("stream_abort_error") is not None
                or item.get("stream_close_error") is not None
                or item.get("output_stop_confirmed") is not True
            ):
                reasons.append("segment_telemetry_invalid")
                break
            try:
                callback_summary = _load_callback_time_info(
                    item.get("callback_time_info"),
                    arrays,
                    expected_frames=expected_group_frames,
                )
            except RecordingGainLinearityError:
                reasons.append("callback_timebase_witness_invalid")
                break
            if int(item.get("callback_count", -1)) != int(
                callback_summary["callback_count"]
            ):
                reasons.append("callback_metadata_raw_count_mismatch")
                break
            previous_campaign_elapsed = float(campaign_elapsed)
    if reasons or submitted is None or recorded is None:
        analysis: dict[str, Any] = {"rows": [], "failure_before_metrics": True}
    else:
        source_float = submitted[:, 0].astype(np.float64) / 32767.0
        input_float = recorded.astype(np.float64) / float(2**31)
        try:
            if ref_witness_reanalysis:
                source_float, clock_alignment, clock_reasons = (
                    _group_ref_witness_alignment(
                        payload=payload,
                        source_float=source_float,
                        input_float=input_float,
                    )
                )
            else:
                clock_alignment, clock_reasons = _group_clock_alignment(
                    payload=payload,
                    source_float=source_float,
                    input_float=input_float,
                )
            reasons.extend(clock_reasons)
        except RecordingGainLinearityError as exc:
            reasons.append(f"clock_alignment_build:{exc}")
            return _finalize_gain_linearity_receipt(
                payload=payload,
                plan_ref=plan["file"],
                raw_ref=raw_ref,
                publication_ref=publication_ref,
                analysis={
                    "rows": [],
                    "clock_alignment": [],
                    "safety_operator": None,
                    "failure_before_metrics": True,
                },
                reasons=[*authority_reasons, *reasons],
                capture_provenance=(
                    capture_provenance if ref_witness_reanalysis else None
                ),
                analysis_provenance=analysis_provenance,
            )
        safety_operator: dict[str, Any] | None = None
        operator_diagnostic: dict[str, Any] | None = None
        if not ref_witness_reanalysis:
            try:
                safety_operator, operator_reasons = _build_safety_operators(
                    payload=payload,
                    source_float=source_float,
                    input_float=input_float,
                    clock_alignment=clock_alignment,
                )
                reasons.extend(operator_reasons)
            except RecordingGainLinearityError as exc:
                reasons.append(f"safety_operator_build:{exc}")
                return _finalize_gain_linearity_receipt(
                    payload=payload,
                    plan_ref=plan["file"],
                    raw_ref=raw_ref,
                    publication_ref=publication_ref,
                    analysis={
                        "rows": [],
                        "clock_alignment": [
                            clock_alignment[index]
                            for index in sorted(clock_alignment)
                        ],
                        "safety_operator": None,
                        "failure_before_metrics": True,
                    },
                    reasons=[*authority_reasons, *reasons],
                )
        else:
            # 기존 complex-FIR 임계는 낮추지 않는다. 재분석에서는 그 operator를
            # authority로 사용하지 않고 결과만 diagnostic으로 보존한다.
            try:
                operator_diagnostic, diagnostic_reasons = _build_safety_operators(
                    payload=payload,
                    source_float=source_float,
                    input_float=input_float,
                    clock_alignment=clock_alignment,
                )
            except RecordingGainLinearityError as exc:
                operator_diagnostic = {"build_error": str(exc)}
            else:
                operator_diagnostic = {
                    "operator": operator_diagnostic,
                    "threshold_failure_reasons": diagnostic_reasons,
                    "authority": False,
                }
        rows: list[dict[str, Any]] = []
        ess_gain: dict[str, list[tuple[int, float]]] = {"err": [], "ref": []}
        peak_ratios: dict[str, list[float]] = {"err": [], "ref": []}
        group_peaks: list[dict[str, Any]] = []
        for group in payload["capture_groups"]:
            group_start = int(group["start_frame"])
            group_stop = int(group["stop_frame"])
            level = int(group["level_millionths"])
            item = {"level_millionths": level, "channels": {}}
            for name, channel in (("err", 0), ("ref", 1)):
                values = input_float[group_start:group_stop, channel]
                peak = float(np.max(np.abs(values)))
                clip_ratio = float(np.mean(np.abs(values) >= 0.999))
                peak_ratios[name].append(peak / (level / 1_000_000.0))
                item["channels"][name] = {
                    "peak_linear": peak,
                    "clip_ratio": clip_ratio,
                }
                if peak >= ADC_ABSOLUTE_PEAK_CEILING:
                    reasons.append(f"{name}_group_absolute_peak_{level}")
                if peak >= ADC_CERTIFICATION_PEAK:
                    reasons.append(f"{name}_group_certification_peak_{level}")
                if clip_ratio != 0.0:
                    reasons.append(f"{name}_group_clip_{level}")
            group_peaks.append(item)
        group_by_level = {
            int(group["level_millionths"]): (index, group)
            for index, group in enumerate(payload["capture_groups"])
        }
        preflight_float = np.asarray(preflight, dtype=np.float64) / float(2**31)
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
                group_index, group = group_by_level[level]
                values = input_float[start:stop, channel]
                peak = float(np.max(np.abs(values)))
                clip_ratio = float(np.mean(np.abs(values) >= 0.999))
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
                aligned_slot = _aligned_channel_view(
                    input_float[:, channel],
                    group=group,
                    alignment=clock_alignment[group_index],
                    output_start=start,
                    output_stop=min(stop, start + active + MAX_DELAY_SAMPLES),
                )
                delay = _delay_samples(drive, aligned_slot)
                item["delay_samples"] = delay
                if row["kind"] == "ESS":
                    response = aligned_slot[delay : delay + active]
                    rms = math.sqrt(float(np.mean(response**2)))
                    normalized = 20.0 * math.log10(max(rms, 1e-30)) - 20.0 * math.log10(
                        level / 1_000_000.0
                    )
                    item["normalised_gain_db"] = float(normalized)
                    ess_gain[name].append((level, float(normalized)))
                else:
                    analysis_start = delay + int(
                        round(IMD_ANALYSIS_START_SECONDS * SAMPLE_RATE)
                    )
                    analysis_stop = analysis_start + int(
                        round(IMD_ANALYSIS_SECONDS * SAMPLE_RATE)
                    )
                    distortion = _distortion_metrics(
                        aligned_slot[analysis_start:analysis_stop],
                        tuple(float(v) for v in row["pair_hz"]),
                        preflight_values=preflight_float[:, channel],
                    )
                    item.update(distortion)
                    # Noise floor에서 distortion을 분리하지 못한 row의 raw
                    # THD/IMD 숫자는 판정 가능한 물리값이 아니다. 그런 row는
                    # INCONCLUSIVE/distortion_certified=false로만 남기고, 실제로
                    # observable한 row에만 -30 dBc gate를 적용한다.
                    if distortion["observable"] is True:
                        if distortion["thd_dbc"] > THD_IMD_GATE_DBC:
                            reasons.append(
                                f"{name}_thd_{level}_{row['pair_hz']}"
                            )
                        if distortion["imd_dbc"] > THD_IMD_GATE_DBC:
                            reasons.append(
                                f"{name}_imd_{level}_{row['pair_hz']}"
                            )
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
        supported_max = min(LEVELS_MILLIONTHS[-1], *empirical_upper.values())
        if ref_witness_reanalysis:
            safety_operator = _build_peak_safety_envelope(
                peak_gain_upper=ratio_upper,
                supported_max_amplitude_millionths=supported_max,
            )
        distortion_channels = [
            channel
            for row in rows
            if row["kind"] == "IMD"
            for channel in row["channels"].values()
        ]
        observable_distortion_rows = sum(
            channel.get("observable") is True for channel in distortion_channels
        )
        analysis = {
            "rows": rows,
            "group_peak_safety": group_peaks,
            "clock_alignment": [
                clock_alignment[index] for index in sorted(clock_alignment)
            ],
            "clock_alignment_method": (
                TIMELINE_METHOD if ref_witness_reanalysis else "affine_repeat_response_v1"
            ),
            "compression": compression,
            "peak_gain_upper_with_uncertainty": ratio_upper,
            "empirical_upper_amplitude_millionths": empirical_upper,
            "tested_max_amplitude_millionths": LEVELS_MILLIONTHS[-1],
            "supported_max_amplitude_millionths": supported_max,
            "safety_operator": safety_operator,
            "complex_operator_diagnostic": operator_diagnostic,
            "safety_operator_is_anc_plant_authority": False,
            # 이 probe의 낮은 acoustic level에서는 THD/IMD를 noise로부터 항상
            # 분리할 수 없다. INCONCLUSIVE를 THD PASS로 승격하지 않는다. 대신
            # independent 0.006 holdout과 induced/absolute residual을 포함한 operator가
            # tested 범위의 ADC peak safety만 bound한다.
            "distortion_certified": False,
            "distortion_observability": {
                "channel_row_count": len(distortion_channels),
                "observable_channel_row_count": observable_distortion_rows,
                "inconclusive_channel_row_count": (
                    len(distortion_channels) - observable_distortion_rows
                ),
                "observable_rows_still_require_minus_30_dbc": True,
                "inconclusive_is_not_thd_pass": True,
            },
            "physical_authority_scope": GAIN_LINEARITY_AUTHORITY_SCOPE,
            "failure_before_metrics": False,
        }
    return _finalize_gain_linearity_receipt(
        payload=payload,
        plan_ref=plan["file"],
        raw_ref=raw_ref,
        publication_ref=publication_ref,
        analysis=analysis,
        reasons=[*authority_reasons, *reasons],
        capture_provenance=(capture_provenance if ref_witness_reanalysis else None),
        analysis_provenance=analysis_provenance,
    )


def build_gain_linearity_reanalysis_receipt(
    *,
    repo_root: str | Path,
    raw_path: str,
    expected_raw_sha256: str,
    plan_path: str,
    expected_plan_sha256: str,
    publication_path: str,
    expected_publication_sha256: str,
) -> dict[str, Any]:
    """Immutable old raw를 현행 clean analyzer로 REF-witness 재분석한다."""

    return build_gain_linearity_receipt(
        repo_root=repo_root,
        raw_path=raw_path,
        expected_raw_sha256=expected_raw_sha256,
        plan_path=plan_path,
        expected_plan_sha256=expected_plan_sha256,
        publication_path=publication_path,
        expected_publication_sha256=expected_publication_sha256,
        ref_witness_reanalysis=True,
    )


def issue_gain_linearity_receipt(
    *,
    repo_root: str | Path,
    output_path: str,
    raw_path: str,
    expected_raw_sha256: str,
    plan_path: str,
    expected_plan_sha256: str,
    publication_path: str | None = None,
    expected_publication_sha256: str | None = None,
) -> tuple[Path, str, dict[str, Any]]:
    root = Path(repo_root).resolve()
    relative = _relative(output_path, label="gain-linearity receipt path")
    payload = build_gain_linearity_receipt(
        repo_root=root,
        raw_path=raw_path,
        expected_raw_sha256=expected_raw_sha256,
        plan_path=plan_path,
        expected_plan_sha256=expected_plan_sha256,
        publication_path=publication_path,
        expected_publication_sha256=expected_publication_sha256,
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


def issue_gain_linearity_reanalysis_receipt(
    *,
    repo_root: str | Path,
    output_path: str,
    raw_path: str,
    expected_raw_sha256: str,
    plan_path: str,
    expected_plan_sha256: str,
    publication_path: str,
    expected_publication_sha256: str,
) -> tuple[Path, str, dict[str, Any]]:
    """새 경로에만 no-replace REF-witness 재분석 receipt를 발행한다."""

    root = Path(repo_root).resolve()
    relative = _relative(output_path, label="gain-linearity reanalysis receipt path")
    payload = build_gain_linearity_reanalysis_receipt(
        repo_root=root,
        raw_path=raw_path,
        expected_raw_sha256=expected_raw_sha256,
        plan_path=plan_path,
        expected_plan_sha256=expected_plan_sha256,
        publication_path=publication_path,
        expected_publication_sha256=expected_publication_sha256,
    )
    data = _pretty_json_bytes(payload)
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
    except FileExistsError as exc:
        raise RecordingGainLinearityError(
            f"재분석 receipt는 no-replace입니다: {relative}"
        ) from exc
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
    publication_ref = payload.get("capture_publication")
    if publication_ref is not None and not isinstance(publication_ref, Mapping):
        raise RecordingGainLinearityError("receipt capture_publication schema 불일치")
    schema = payload.get("schema")
    kwargs = {
        "repo_root": root,
        "raw_path": str(payload["raw"]["path"]),
        "expected_raw_sha256": str(payload["raw"]["sha256"]),
        "plan_path": str(payload["plan"]["path"]),
        "expected_plan_sha256": str(payload["plan"]["sha256"]),
        "publication_path": (
            None if publication_ref is None else str(publication_ref["path"])
        ),
        "expected_publication_sha256": (
            None if publication_ref is None else str(publication_ref["sha256"])
        ),
    }
    if schema == GAIN_LINEARITY_REANALYSIS_RECEIPT_SCHEMA:
        if publication_ref is None:
            raise RecordingGainLinearityError(
                "REF-witness 재분석 receipt에는 capture publication이 필요합니다"
            )
        capture_provenance = payload.get("capture_provenance")
        analysis_provenance = payload.get("analysis_provenance")
        if (
            not isinstance(capture_provenance, Mapping)
            or not isinstance(analysis_provenance, Mapping)
            or set(analysis_provenance) != _EXECUTION_KEYS
            or payload.get("source_commit")
            != analysis_provenance.get("repository_commit")
        ):
            raise RecordingGainLinearityError("재분석 capture/analysis provenance 불일치")
        rebuilt = build_gain_linearity_receipt(
            **kwargs, ref_witness_reanalysis=True
        )
    elif schema == GAIN_LINEARITY_RECEIPT_SCHEMA:
        rebuilt = build_gain_linearity_receipt(**kwargs)
    else:
        raise RecordingGainLinearityError("gain-linearity receipt schema 불일치")
    if payload != rebuilt:
        raise RecordingGainLinearityError("gain-linearity receipt 독립 재계산 불일치")
    capture_publication = None
    capture_metadata = None
    if publication_ref is not None:
        metadata, _arrays, raw_ref = _load_raw(
            repo_root=root,
            raw_path=str(payload["raw"]["path"]),
            expected_sha256=str(payload["raw"]["sha256"]),
        )
        publication = _load_capture_publication(
            repo_root=root,
            publication_path=str(publication_ref["path"]),
            expected_sha256=str(publication_ref["sha256"]),
            raw_ref=raw_ref,
            metadata=metadata,
        )
        capture_publication = dict(publication["file"])
        capture_metadata = dict(publication["metadata_file"])
        if capture_publication != dict(publication_ref):
            raise RecordingGainLinearityError(
                "receipt/publication transfer ref가 독립 재검증과 다릅니다"
            )
    return {
        "receipt_path": relative,
        "receipt_sha256": str(snapshot.sha256),
        "payload": payload,
        "passed": payload.get("status") == "PASS",
        "capture_publication": capture_publication,
        "capture_metadata": capture_metadata,
    }


__all__ = [
    "ADC_ABSOLUTE_PEAK_CEILING",
    "ADC_CERTIFICATION_PEAK",
    "GAIN_LINEARITY_PLAN_SCHEMA",
    "GAIN_LINEARITY_PEAK_ENVELOPE_SCHEMA",
    "GAIN_LINEARITY_PUBLICATION_SCHEMA",
    "GAIN_LINEARITY_RAW_SCHEMA",
    "GAIN_LINEARITY_REANALYSIS_RECEIPT_SCHEMA",
    "GAIN_LINEARITY_RECEIPT_SCHEMA",
    "LEVELS_MILLIONTHS",
    "PREDICTIVE_STOP_PEAK",
    "RecordingGainLinearityError",
    "build_gain_linearity_capture_publication_payload",
    "build_gain_linearity_plan",
    "build_gain_linearity_reanalysis_receipt",
    "build_gain_linearity_receipt",
    "issue_gain_linearity_reanalysis_receipt",
    "issue_gain_linearity_receipt",
    "load_gain_linearity_plan",
    "next_level_stop_decision",
    "validate_gain_linearity_plan_payload",
    "validate_gain_linearity_receipt",
]
