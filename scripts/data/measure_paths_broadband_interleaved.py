#!/usr/bin/env python3
"""광대역 P/S multi-panel 측정 계획과 immutable raw-first 캡처 경로.

``--dry-run``은 sounddevice를 import/open하지 않는다. 실제 출력은 저장된 exact plan,
fresh meter raw, paired level evidence와 모든 운영자 확인을 갖춘 ``--execute-live``에서만
열린다. 캡처가 시작된 뒤 실패하더라도 partial raw를 분석보다 먼저 보존한다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, ContextManager

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.audio_io import (  # noqa: E402
    alsa_card_index,
    resolve_alsa_portaudio_device,
)
from deep_anc.config import load_yaml  # noqa: E402
from deep_anc.dsp.control_band_contract import (  # noqa: E402
    ControlBandContract,
    max_timing_error_samples_for_attenuation,
    phase_error_degrees,
)
from deep_anc.dsp.broadband_interleaved import (  # noqa: E402
    BROADBAND_CLOCK_PILOT_BAND_HZ,
    BROADBAND_MARKER_GUARD_SECONDS,
    BROADBAND_MARKER_SECONDS,
    SUBMITTED_PILOT_CROSS_CHANNEL_MAX_ABSOLUTE,
    SUBMITTED_PILOT_CROSS_CHANNEL_MAX_RATIO,
    build_clock_piloted_panel_probe,
    build_nonperiodic_timing_markers,
    fixed_clock_pilot_complex_spectrum,
    fixed_clock_pilot_sha256,
    validate_submitted_pilot_cross_channel_null,
)
from deep_anc.dsp.measurement_level import (  # noqa: E402
    BOOTSTRAP_METER_MAX_AGE_SECONDS,
    DEFAULT_MEASUREMENT_LEVEL_EVIDENCE_PATH,
    assert_live_pcm_clock_preconditions,
    collect_alsa_physical_fingerprint,
    load_measurement_level_evidence,
    measurement_hardware_identity,
    repository_audio_lock,
    validate_bootstrap_meter_raw,
    validate_measurement_hardware_contract,
)
from scripts.data import measure_paths_interleaved as mpi  # noqa: E402


BROADBAND_MEASUREMENT_PLAN_SCHEMA = (
    "broadband_interleaved_measurement_plan_v5_global_clock"
)
BROADBAND_RAW_CAPTURE_SCHEMA = "broadband_interleaved_raw_v5_global_clock_raw_first"
BROADBAND_METHOD = "broadband_interleaved_multitone_panels_global_clock_v5"
DEFAULT_AMPLITUDE = 0.003
DEFAULT_PERIOD_SECONDS = 0.125
DEFAULT_WARMUP_SECONDS = 4.0
DEFAULT_REPEATS_PER_PANEL = 63
DEFAULT_TRANSITION_SECONDS = 1.25
DEFAULT_LEAD_IN_SECONDS = 0.5
DEFAULT_HARD_MAX_SECONDS = 50.0
MAX_CREST_DB = 14.0
DEFAULT_INPUT_PREFLIGHT_SECONDS = 3.0
DEFAULT_DIAGNOSTICS_ROOT = Path("results/calibration_interleaved/broadband")
BROADBAND_LIVE_AUTHORITY_V4_DIAGNOSTIC = {
    "path": (
        "results/data_audit/"
        "broadband_measurement_signal_plan_live_authority_v4_20260828.json"
    ),
    "file_sha256": "3c71098cbb2d928c22cede751d567f9f9e5c5d2e1a2674bd9228a6f16cd26a58",
    "payload_sha256": "37abc313f4559f1a5dc3c7d9b0131ee29ca40dffda67c67bb0496336f917d063",
    "pcm_sha256": "2e5d1b6031d55ef65610a0e7a86c7cc434d10a09ac042b5af949b78ceaef9b26",
}
# v5 plan과 dry-run/test가 완성된 뒤 root 검토로 exact bytes/semantic/PCM SHA를
# 고정하기 전에는 live를 열지 않는다. v4는 진단 기록으로만 보존한다.
BROADBAND_LIVE_AUTHORITY: dict[str, str] | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _plan_payload_sha256(plan: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(plan)).hexdigest()


def _float_to_pcm16(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype(np.int16)


def build_signal_plan(
    *,
    hardware_path: str | Path,
    amplitude: float = DEFAULT_AMPLITUDE,
    period_seconds: float = DEFAULT_PERIOD_SECONDS,
    warmup_seconds: float = DEFAULT_WARMUP_SECONDS,
    repeats_per_panel: int = DEFAULT_REPEATS_PER_PANEL,
    transition_seconds: float = DEFAULT_TRANSITION_SECONDS,
    lead_in_seconds: float = DEFAULT_LEAD_IN_SECONDS,
    hard_max_seconds: float = DEFAULT_HARD_MAX_SECONDS,
) -> tuple[dict[str, Any], np.ndarray]:
    """전체 output PCM을 메모리에서 만들고 duration/crest/band를 검증한다."""

    contract = ControlBandContract.broadband_point_control()
    hardware_file = Path(hardware_path).expanduser().resolve()
    hardware = load_yaml(hardware_file)
    audio = dict(hardware.get("audio") or {})
    channels = dict(hardware.get("channels") or {})
    sample_rate = int(audio.get("sample_rate", 0))
    block_size = int(audio.get("block_size", 0))
    latency = str(audio.get("latency", ""))
    if (sample_rate, block_size, latency) != (48_000, 256, "low"):
        raise ValueError("광대역 P/S dry-run은 hardware 48kHz/256/low가 필요합니다")
    expected_channels = {
        "error_mic": 0,
        "reference_mic": 1,
        "noise_out": 0,
        "cancel_out": 1,
    }
    if channels != expected_channels:
        raise ValueError(f"광대역 P/S channel map이 다릅니다: {channels!r}")
    amp = float(amplitude)
    if not math.isclose(amp, DEFAULT_AMPLITUDE, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("광대역 P/S probe peak는 paired level 계약 0.003으로 고정입니다")
    if not math.isclose(
        float(period_seconds), DEFAULT_PERIOD_SECONDS, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("광대역 P/S period는 clock 계약 0.125초로 고정입니다")
    if int(repeats_per_panel) != DEFAULT_REPEATS_PER_PANEL:
        raise ValueError("광대역 P/S는 panel마다 63 repeats가 필요합니다")

    probes = []
    panel_metadata: list[dict[str, Any]] = []
    for panel_index, panel in enumerate(contract.measurement_panels_hz):
        probe = build_clock_piloted_panel_probe(
            sample_rate=sample_rate,
            period_seconds=float(period_seconds),
            panel_band_hz=panel,
            amplitude=amp,
        )
        if probe.guard_bins() != 1:
            raise ValueError(f"panel #{panel_index} guard가 1 bin이 아닙니다")
        noise_crest, cancel_crest = probe.crest_db()
        if max(noise_crest, cancel_crest) > MAX_CREST_DB:
            raise ValueError(f"panel #{panel_index} crest가 {MAX_CREST_DB:g}dB를 넘습니다")
        resolution = sample_rate / probe.period_samples
        noise_frequencies = probe.noise_bins.astype(np.float64) * resolution
        cancel_frequencies = probe.cancel_bins.astype(np.float64) * resolution
        noise_band = (float(noise_frequencies[0]), float(noise_frequencies[-1]))
        cancel_band = (float(cancel_frequencies[0]), float(cancel_frequencies[-1]))
        panel_lo, panel_hi = (float(value) for value in panel)
        noise_panel = noise_frequencies[
            (noise_frequencies >= panel_lo) & (noise_frequencies <= panel_hi)
        ]
        cancel_panel = cancel_frequencies[
            (cancel_frequencies >= panel_lo) & (cancel_frequencies <= panel_hi)
        ]
        pilot_lo, pilot_hi = BROADBAND_CLOCK_PILOT_BAND_HZ
        noise_pilot = noise_frequencies[
            (noise_frequencies >= pilot_lo) & (noise_frequencies <= pilot_hi)
        ]
        cancel_pilot = cancel_frequencies[
            (cancel_frequencies >= pilot_lo) & (cancel_frequencies <= pilot_hi)
        ]
        if min(
            noise_panel.size,
            cancel_panel.size,
            noise_pilot.size,
            cancel_pilot.size,
        ) < 8:
            raise ValueError(f"panel #{panel_index}의 분석 tone/clock pilot이 부족합니다")
        probes.append(probe)
        panel_metadata.append(
            {
                "index": panel_index,
                "requested_band_hz": list(panel),
                "clock_pilot_band_hz": list(BROADBAND_CLOCK_PILOT_BAND_HZ),
                "noise_actual_band_hz": list(noise_band),
                "cancel_actual_band_hz": list(cancel_band),
                "noise_panel_actual_band_hz": [
                    float(noise_panel[0]),
                    float(noise_panel[-1]),
                ],
                "cancel_panel_actual_band_hz": [
                    float(cancel_panel[0]),
                    float(cancel_panel[-1]),
                ],
                "noise_clock_pilot_tone_count": int(noise_pilot.size),
                "cancel_clock_pilot_tone_count": int(cancel_pilot.size),
                "noise_tone_count": int(probe.noise_bins.size),
                "cancel_tone_count": int(probe.cancel_bins.size),
                "guard_bins": probe.guard_bins(),
                "noise_crest_db": float(noise_crest),
                "cancel_crest_db": float(cancel_crest),
            }
        )
    for drive in ("noise_actual_band_hz", "cancel_actual_band_hz"):
        if panel_metadata[-1][drive][1] < contract.required_excitation_upper_hz:
            raise ValueError(f"마지막 panel의 {drive}가 8k octave 상단을 덮지 않습니다")

    def zeros(seconds: float) -> np.ndarray:
        frames = int(round(float(seconds) * sample_rate))
        return np.zeros((frames, 2), dtype=np.float32)

    warmup_periods_float = float(warmup_seconds) / float(period_seconds)
    warmup_periods = int(round(warmup_periods_float))
    if not math.isclose(
        warmup_periods_float, float(warmup_periods), rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError("warmup은 0.125초 period의 정수배여야 합니다")
    first_period = np.stack((probes[0].noise_signal, probes[0].cancel_signal), axis=1)
    timing_markers, marker_metadata = build_nonperiodic_timing_markers(
        sample_rate=sample_rate,
        duration_seconds=BROADBAND_MARKER_SECONDS,
        amplitude=amp,
    )
    marker_frames = int(timing_markers.shape[0])
    guard_frames = int(round(BROADBAND_MARKER_GUARD_SECONDS * sample_rate))
    primary_marker = np.zeros_like(timing_markers)
    primary_marker[:, 0] = timing_markers[:, 0]
    secondary_marker = np.zeros_like(timing_markers)
    secondary_marker[:, 1] = timing_markers[:, 1]
    segments: list[np.ndarray] = [
        zeros(lead_in_seconds),
        primary_marker,
        np.zeros((guard_frames, 2), dtype=np.float32),
        secondary_marker,
        np.zeros((guard_frames, 2), dtype=np.float32),
        np.tile(first_period, (warmup_periods, 1)),
    ]
    anchor_periods_float = float(transition_seconds) / float(period_seconds)
    anchor_periods = int(round(anchor_periods_float))
    if not math.isclose(
        anchor_periods_float, float(anchor_periods), rel_tol=0.0, abs_tol=1.0e-9
    ) or anchor_periods != 10:
        raise ValueError(
            "panel 사이 lowband clock anchor는 guard 1 + analysis 9, 총 10 period여야 합니다"
        )
    for index, probe in enumerate(probes):
        one_period = np.stack((probe.noise_signal, probe.cancel_signal), axis=1)
        segments.append(np.tile(one_period, (int(repeats_per_panel), 1)))
        if index + 1 < len(probes):
            # 고역 dense tone이 lowband clock witness를 0/64로 만든 결함을 반복하지 않는다.
            # 첫 period는 row-boundary tail guard로 버리고 뒤 9개에서 8 adjacent
            # witness를 얻도록 첫 panel의 저역 주기를 총 10번 삽입한다.
            segments.append(np.tile(first_period, (anchor_periods, 1)))
    output_float = np.concatenate(segments, axis=0)
    padding_frames = (-output_float.shape[0]) % block_size
    if padding_frames:
        output_float = np.concatenate(
            (output_float, np.zeros((padding_frames, 2), dtype=np.float32)), axis=0
        )
    if not np.all(np.isfinite(output_float)):
        raise ValueError("signal plan에 NaN/Inf가 있습니다")
    if float(np.max(np.abs(output_float))) > amp + 1.0e-7:
        raise ValueError("signal plan peak가 0.003을 넘습니다")
    duration_seconds = output_float.shape[0] / sample_rate
    if duration_seconds > float(hard_max_seconds):
        raise ValueError(
            f"signal plan {duration_seconds:.3f}s가 hard max {float(hard_max_seconds):.3f}s를 넘습니다"
        )
    output_pcm = _float_to_pcm16(output_float)
    pcm_sha = hashlib.sha256(output_pcm.tobytes(order="C")).hexdigest()
    fixed_pilot = fixed_clock_pilot_complex_spectrum(
        sample_rate=sample_rate, period_seconds=float(period_seconds)
    )
    cross_channel_null_receipts: list[dict[str, Any]] = []
    cross_channel_null_digest = hashlib.sha256()
    cross_channel_null_max_absolute = 0.0
    cross_channel_null_max_ratio = 0.0
    for panel_index, probe in enumerate(probes):
        period_pcm = _float_to_pcm16(
            np.stack((probe.noise_signal, probe.cancel_signal), axis=1)
        )
        receipt = validate_submitted_pilot_cross_channel_null(
            period_pcm,
            sample_rate=sample_rate,
            period_seconds=float(period_seconds),
        )
        receipt_payload = {"panel_index": panel_index, **receipt}
        cross_channel_null_receipts.append(receipt_payload)
        cross_channel_null_digest.update(
            _canonical_json_bytes(receipt_payload)
        )
        for drive in ("noise", "cancel"):
            drive_row = receipt["drives"][drive]
            cross_channel_null_max_absolute = max(
                cross_channel_null_max_absolute,
                float(drive_row["cross_channel_max_magnitude"]),
            )
            cross_channel_null_max_ratio = max(
                cross_channel_null_max_ratio,
                float(drive_row["cross_to_main_max_ratio"]),
            )
    pilot_pcm_spectrum_sha: dict[str, list[str]] = {"noise": [], "cancel": []}
    pilot_pcm_max_panel_delta: dict[str, float] = {}
    for drive, channel in (("noise", 0), ("cancel", 1)):
        pilot_bins = np.asarray(fixed_pilot[drive][0], dtype=np.int64)
        spectra: list[np.ndarray] = []
        for probe in probes:
            period_float = np.stack(
                (probe.noise_signal, probe.cancel_signal), axis=1
            )
            period_pcm = _float_to_pcm16(period_float)
            spectrum = np.asarray(
                np.fft.rfft(period_pcm[:, channel].astype(np.float64))[pilot_bins],
                dtype=np.complex128,
            )
            spectra.append(spectrum)
            digest = hashlib.sha256()
            digest.update(np.asarray(spectrum.real, dtype="<f8").tobytes())
            digest.update(np.asarray(spectrum.imag, dtype="<f8").tobytes())
            pilot_pcm_spectrum_sha[drive].append(digest.hexdigest())
        pilot_pcm_max_panel_delta[drive] = float(
            max(np.max(np.abs(spectra[0] - value)) for value in spectra[1:])
        )
    pilot_pcm_exact = all(
        len(set(values)) == 1 for values in pilot_pcm_spectrum_sha.values()
    )
    common_pilot_pcm_sha = (
        hashlib.sha256(
            "".join(
                pilot_pcm_spectrum_sha[drive][0]
                for drive in ("noise", "cancel")
            ).encode("ascii")
        ).hexdigest()
        if pilot_pcm_exact
        else None
    )

    layout: list[dict[str, Any]] = []
    cursor = 0

    def layout_row(kind: str, frames: int, **extra: Any) -> None:
        nonlocal cursor
        row = {
            "kind": kind,
            "start_frame": int(cursor),
            "stop_frame": int(cursor + frames),
            "frames": int(frames),
            **extra,
        }
        layout.append(row)
        cursor += frames

    layout_row("lead_in_silence", int(round(float(lead_in_seconds) * sample_rate)))
    layout_row(
        "primary_nonperiodic_timing_marker",
        marker_frames,
        output_channel=0,
        other_channel_silent=True,
    )
    layout_row("primary_marker_tail_guard", guard_frames)
    layout_row(
        "secondary_nonperiodic_timing_marker",
        marker_frames,
        output_channel=1,
        other_channel_silent=True,
    )
    layout_row("secondary_marker_tail_guard", guard_frames)
    layout_row("warmup_panel_0", warmup_periods * probes[0].period_samples, panel_index=0)
    for index, probe in enumerate(probes):
        layout_row(
            "analysis_panel",
            int(repeats_per_panel) * probe.period_samples,
            panel_index=index,
        )
        if index + 1 < len(probes):
            layout_row(
                "lowband_clock_anchor",
                anchor_periods * probes[0].period_samples,
                before_panel_index=index + 1,
            )
    if padding_frames:
        layout_row("block_padding_silence", padding_frames)
    if cursor != output_pcm.shape[0]:
        raise ValueError(f"layout/output frame 불일치: {cursor} != {output_pcm.shape[0]}")

    phase_budgets = {}
    for center in (2000.0, 4000.0, 8000.0):
        phase_budgets[str(int(center))] = {
            "degrees_per_sample": phase_error_degrees(1.0, center, sample_rate),
            "max_error_samples_10db": max_timing_error_samples_for_attenuation(
                10.0, center, sample_rate
            ),
            "max_error_samples_20db": max_timing_error_samples_for_attenuation(
                20.0, center, sample_rate
            ),
        }
    plan = {
        "schema": BROADBAND_MEASUREMENT_PLAN_SCHEMA,
        "role": "signal_only_dry_run_no_audio",
        "live_capture_enabled": False,
        "control_band_contract": contract.model_dump(mode="json"),
        "control_band_contract_sha256": contract.digest(),
        "hardware": {
            "path": str(hardware_file),
            "sha256": _sha256_file(hardware_file),
            "sample_rate": sample_rate,
            "block_size": block_size,
            "latency": latency,
            "channels": channels,
        },
        "recipe": {
            "amplitude": amp,
            "period_seconds": float(period_seconds),
            "warmup_seconds": float(warmup_seconds),
            "repeats_per_panel": int(repeats_per_panel),
            "transition_seconds": float(transition_seconds),
            "transition_kind": "lowband_clock_anchor",
            "transition_periods": int(anchor_periods),
            "transition_guard_periods": 1,
            "transition_valid_adjacent_periods": 8,
            "lead_in_seconds": float(lead_in_seconds),
            "hard_max_seconds": float(hard_max_seconds),
            "in_panel_clock_pilot_band_hz": list(BROADBAND_CLOCK_PILOT_BAND_HZ),
            "panel_compact_fir_forbidden": True,
            "required_post_analysis": (
                "global_clock_map_then_fractional_joint_ls_then_overlap_validation_only_"
                "then_measured_complex_response_with_diagnostic_only_compact_fir"
            ),
            "fixed_clock_pilot_sha256": fixed_clock_pilot_sha256(
                sample_rate=sample_rate, period_seconds=float(period_seconds)
            ),
            "fixed_clock_pilot_sha256_domain": "intended_float_spectrum",
            "fixed_clock_pilot_pcm_exact_across_panels": pilot_pcm_exact,
            "fixed_clock_pilot_pcm_spectrum_sha256": common_pilot_pcm_sha,
            "fixed_clock_pilot_pcm_panel_sha256": pilot_pcm_spectrum_sha,
            "fixed_clock_pilot_pcm_max_panel_delta": pilot_pcm_max_panel_delta,
            "global_clock_input_domain": (
                "actual_submitted_int16_period_spectrum_not_intended_float"
            ),
            "submitted_pilot_cross_channel_null": {
                "schema": "submitted_int16_pilot_cross_channel_null_plan_v1",
                "sha256": cross_channel_null_digest.hexdigest(),
                "maximum_absolute_allowed": (
                    SUBMITTED_PILOT_CROSS_CHANNEL_MAX_ABSOLUTE
                ),
                "maximum_ratio_allowed": SUBMITTED_PILOT_CROSS_CHANNEL_MAX_RATIO,
                "maximum_absolute_observed": cross_channel_null_max_absolute,
                "maximum_ratio_observed": cross_channel_null_max_ratio,
                "all_panels_passed": True,
                "panel_receipts": cross_channel_null_receipts,
            },
            "timing_marker_seconds": BROADBAND_MARKER_SECONDS,
            "timing_marker_guard_seconds": BROADBAND_MARKER_GUARD_SECONDS,
            "timing_marker_search_samples": [0, 4_800],
            "timing_marker_max_branch_width_samples": 2_999.0,
        },
        "timing_markers": marker_metadata,
        "panels": panel_metadata,
        "layout": layout,
        "output": {
            "frames": int(output_pcm.shape[0]),
            "channels": 2,
            "dtype": str(output_pcm.dtype),
            "duration_seconds": float(duration_seconds),
            "padding_frames": int(padding_frames),
            "peak_pcm": int(np.max(np.abs(output_pcm.astype(np.int32)))),
            "pcm_sha256": pcm_sha,
        },
        "phase_budgets": phase_budgets,
    }
    return plan, output_pcm


def _repository_path(value: str | Path, *, require_results: bool = False) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"광대역 측정 경로는 저장소 안에 있어야 합니다: {resolved}") from exc
    if require_results and (not relative.parts or relative.parts[0] != "results"):
        raise ValueError(f"광대역 raw session은 results/ 아래여야 합니다: {resolved}")
    return resolved


def _repository_relative_path(value: str | Path) -> str:
    return _repository_path(value).relative_to(REPO_ROOT.resolve()).as_posix()


def raw_session_path_for_plan(
    plan: dict[str, Any], requested: str | Path | None = None
) -> Path:
    if requested is None:
        requested = DEFAULT_DIAGNOSTICS_ROOT / _plan_payload_sha256(plan)[:16]
    return _repository_path(requested, require_results=True)


def validate_fresh_raw_session_target(path: Path) -> None:
    target = _repository_path(path, require_results=True)
    if target.exists():
        raise FileExistsError(f"기존 광대역 raw session은 덮어쓰지 않습니다: {target}")
    cursor = target.parent
    while not cursor.exists():
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if cursor.exists() and not cursor.is_dir():
        raise NotADirectoryError(f"광대역 raw session 조상이 디렉터리가 아닙니다: {cursor}")


def validate_plan_pcm_exact(
    plan: dict[str, Any], planned_pcm: np.ndarray
) -> np.ndarray:
    pcm = np.asarray(planned_pcm)
    output = plan.get("output")
    if not isinstance(output, dict):
        raise ValueError("광대역 plan output mapping이 필요합니다")
    expected_shape = (int(output.get("frames", -1)), int(output.get("channels", -1)))
    if pcm.dtype != np.int16 or pcm.shape != expected_shape:
        raise ValueError(
            f"plan PCM shape/dtype 불일치: {pcm.shape}/{pcm.dtype}, "
            f"expected={expected_shape}/int16"
        )
    actual_sha = hashlib.sha256(pcm.tobytes(order="C")).hexdigest()
    if actual_sha != output.get("pcm_sha256"):
        raise ValueError(
            "plan PCM SHA 불일치: "
            f"stored={output.get('pcm_sha256')!r}, actual={actual_sha}"
        )
    output_float = pcm.astype(np.float32) / np.float32(np.iinfo(np.int16).max)
    reconverted = mpi.cw.float32_to_pcm_int16(output_float)
    if not np.array_equal(reconverted, pcm):
        raise ValueError("plan int16 PCM을 capture float로 exact 왕복할 수 없습니다")
    return output_float


def load_exact_saved_plan(
    path: str | Path,
    *,
    expected_plan: dict[str, Any],
    expected_pcm: np.ndarray,
) -> dict[str, Any]:
    plan_path = _repository_path(path)
    if not plan_path.is_file():
        raise FileNotFoundError(f"저장된 광대역 dry-run plan이 없습니다: {plan_path}")
    payload = plan_path.read_bytes()
    try:
        observed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"광대역 plan JSON을 읽을 수 없습니다: {plan_path}: {exc}") from exc
    if observed != expected_plan:
        raise ValueError("저장된 광대역 plan이 현재 hardware/contract/recipe와 다릅니다")
    validate_plan_pcm_exact(observed, expected_pcm)
    return {
        "path": plan_path,
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_sha256": _plan_payload_sha256(observed),
        "payload": observed,
    }


def validate_live_authority_binding(
    saved_plan: dict[str, Any], planned_pcm: np.ndarray
) -> dict[str, str]:
    """실기 광대역 plan을 현재 승인된 exact bytes/semantic/PCM에 결속한다."""

    if BROADBAND_LIVE_AUTHORITY is None:
        raise ValueError(
            "광대역 v5 live authority SHA가 아직 고정되지 않았습니다; "
            "v4 authority는 diagnostic-only입니다"
        )
    authority = dict(BROADBAND_LIVE_AUTHORITY)
    authority_path = _repository_path(authority["path"])
    plan_path = Path(saved_plan.get("path", "")).resolve()
    if plan_path != authority_path:
        raise ValueError(
            "광대역 live plan path가 현재 authority와 다릅니다: "
            f"{str(plan_path)!r} != {authority['path']!r}"
        )
    exact_fields = {
        "file_sha256": str(saved_plan.get("file_sha256", "")),
        "payload_sha256": str(saved_plan.get("payload_sha256", "")),
        "pcm_sha256": str(
            (saved_plan.get("payload") or {}).get("output", {}).get("pcm_sha256", "")
        ),
    }
    for field, observed in exact_fields.items():
        if observed != authority[field]:
            raise ValueError(
                f"광대역 live authority {field} 불일치: "
                f"{observed!r} != {authority[field]!r}"
            )
    if _sha256_file(plan_path) != authority["file_sha256"]:
        raise ValueError("광대역 live authority file bytes SHA가 다릅니다")
    payload = saved_plan.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("광대역 live authority payload mapping이 필요합니다")
    recipe = payload.get("recipe")
    if not isinstance(recipe, dict) or recipe.get("global_clock_input_domain") != (
        "actual_submitted_int16_period_spectrum_not_intended_float"
    ):
        raise ValueError(
            "광대역 v5 global clock은 실제 submitted int16 period spectrum을 "
            "입력으로 사용해야 합니다"
        )
    null_receipt = recipe.get("submitted_pilot_cross_channel_null")
    if not isinstance(null_receipt, dict) or not bool(
        null_receipt.get("all_panels_passed", False)
    ):
        raise ValueError("광대역 v5 plan의 submitted pilot 반대 channel null 증거가 없습니다")
    if not 0.0 <= float(null_receipt.get("maximum_absolute_observed", math.inf)) <= (
        SUBMITTED_PILOT_CROSS_CHANNEL_MAX_ABSOLUTE
    ):
        raise ValueError("광대역 v5 plan의 submitted pilot absolute null이 깨졌습니다")
    if not 0.0 <= float(null_receipt.get("maximum_ratio_observed", math.inf)) <= (
        SUBMITTED_PILOT_CROSS_CHANNEL_MAX_RATIO
    ):
        raise ValueError("광대역 v5 plan의 submitted pilot relative null이 깨졌습니다")
    if _plan_payload_sha256(payload) != authority["payload_sha256"]:
        raise ValueError("광대역 live authority canonical payload SHA가 다릅니다")
    validate_plan_pcm_exact(payload, planned_pcm)
    actual_pcm_sha = hashlib.sha256(
        np.asarray(planned_pcm).tobytes(order="C")
    ).hexdigest()
    if actual_pcm_sha != authority["pcm_sha256"]:
        raise ValueError("광대역 live authority PCM SHA가 다릅니다")
    return authority


def validate_meter_followup_binding(
    *,
    meter_metadata: dict[str, Any],
    plan_binding: dict[str, str],
    hardware_path: Path,
    hardware_sha256: str,
    level_evidence_path: Path,
    level_evidence_sha256: str,
    raw_session_dir: Path,
) -> dict[str, Any]:
    """fresh meter가 바로 이 광대역 live invocation을 승인했는지 대조한다."""

    expected_evidence = {
        "mode": "verified_existing",
        "path": _repository_relative_path(level_evidence_path),
        "sha256": str(level_evidence_sha256),
    }
    if meter_metadata.get("calibration_evidence") != expected_evidence:
        raise ValueError(
            "fresh meter calibration evidence 결속이 live와 다릅니다: "
            f"{meter_metadata.get('calibration_evidence')!r} != {expected_evidence!r}"
        )
    expected_hardware = {
        "path": _repository_relative_path(hardware_path),
        "sha256": str(hardware_sha256),
    }
    if meter_metadata.get("hardware") != expected_hardware:
        raise ValueError(
            "fresh meter hardware 결속이 live와 다릅니다: "
            f"{meter_metadata.get('hardware')!r} != {expected_hardware!r}"
        )
    expected_followup = {
        "schema": "broadband_meter_followup_v1",
        "mode": "broadband",
        "plan": {
            "path": str(plan_binding["path"]),
            "file_sha256": str(plan_binding["file_sha256"]),
            "payload_sha256": str(plan_binding["payload_sha256"]),
            "pcm_sha256": str(plan_binding["pcm_sha256"]),
        },
        "raw_session_dir": _repository_relative_path(raw_session_dir),
        "hardware": expected_hardware,
        "level_evidence": {
            "path": expected_evidence["path"],
            "sha256": expected_evidence["sha256"],
        },
    }
    if meter_metadata.get("followup_contract") != expected_followup:
        raise ValueError("fresh meter followup contract가 현재 broadband live와 다릅니다")
    return expected_followup


def validate_dry_run_environment(
    *,
    hardware_path: str | Path,
    raw_session_dir: Path,
) -> dict[str, Any]:
    """sounddevice 없이 ALSA mapping/physical identity와 raw freshness를 검사한다."""

    hardware_file = _repository_path(hardware_path)
    hardware_config = load_yaml(hardware_file)
    audio, channel_map = validate_measurement_hardware_contract(hardware_config)
    input_card = alsa_card_index(str(audio["input"]["card"]))
    output_card = alsa_card_index(str(audio["output"]["card"]))
    input_pcm = int(audio["input"]["pcm"])
    output_pcm = int(audio["output"]["pcm"])
    mpi.validate_alsa_pcm_mapping(
        input_card=input_card,
        input_pcm=input_pcm,
        output_card=output_card,
        output_pcm=output_pcm,
    )
    fingerprint = collect_alsa_physical_fingerprint(hardware_config)
    identity = measurement_hardware_identity(
        hardware_config, physical_fingerprint=fingerprint
    )
    validate_fresh_raw_session_target(raw_session_dir)
    return {
        "hardware_path": hardware_file,
        "hardware_sha256": _sha256_file(hardware_file),
        "audio": audio,
        "channel_map": channel_map,
        "hardware_identity": identity,
        "input_alsa": [input_card, input_pcm],
        "output_alsa": [output_card, output_pcm],
        "raw_session_dir": raw_session_dir,
    }


def publish_broadband_raw_capture(
    *,
    session_dir: Path,
    metadata: dict[str, Any],
    planned_pcm: np.ndarray,
    submitted_pcm: np.ndarray,
    input_raw_int32: np.ndarray,
    preflight_raw_int32: np.ndarray,
    callback_time_info: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """submitted PCM을 plan과 대조하고 broadband raw를 분석보다 먼저 승격한다."""

    planned = np.asarray(planned_pcm)
    submitted = np.asarray(submitted_pcm)
    recorded = np.asarray(input_raw_int32)
    preflight = np.asarray(preflight_raw_int32)
    invalid = list(metadata.get("invalid_reasons") or [])
    if metadata.get("post_capture_binding") != {"valid": True, "error": None}:
        invalid.append("post_capture_binding_invalid")
    if planned.dtype != np.int16 or submitted.dtype != np.int16:
        invalid.append("submitted_pcm_dtype_mismatch")
    if planned.shape != submitted.shape or not np.array_equal(planned, submitted):
        invalid.append("submitted_pcm_not_exact_plan")
    planned_sha = hashlib.sha256(planned.tobytes(order="C")).hexdigest()
    submitted_sha = hashlib.sha256(submitted.tobytes(order="C")).hexdigest()
    expected_sha = str(metadata.get("plan", {}).get("pcm_sha256", ""))
    if planned_sha != expected_sha or submitted_sha != expected_sha:
        invalid.append("submitted_pcm_sha_mismatch")
    if recorded.dtype != np.int32 or recorded.shape != (planned.shape[0], 2):
        invalid.append("input_raw_shape_or_dtype_mismatch")
    if preflight.dtype != np.int32 or preflight.ndim != 2 or preflight.shape[1] != 2:
        invalid.append("preflight_raw_shape_or_dtype_mismatch")
    callback_arrays: dict[str, np.ndarray] = {}
    try:
        callback_summary = validate_callback_time_info(
            callback_time_info, expected_frames=int(planned.shape[0])
        )
        assert callback_time_info is not None
        callback_arrays = {
            name: np.asarray(callback_time_info[name]).copy()
            for name in CALLBACK_TIME_INFO_FIELDS
        }
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        callback_summary = {
            "valid": False,
            "sample_slip_count": -1,
            "error": str(exc),
        }
        invalid.append("callback_time_info_invalid")
    invalid = list(dict.fromkeys(invalid))
    stored_metadata = {
        **metadata,
        "raw_capture_schema": BROADBAND_RAW_CAPTURE_SCHEMA,
        "method": BROADBAND_METHOD,
        "status": "PASS" if not invalid else "INVALID",
        "valid": not invalid,
        "invalid_reasons": invalid,
        "submitted_pcm_sha256": submitted_sha,
        "callback_timing": callback_summary,
    }
    paths = mpi.write_immutable_raw_capture_atomic(
        Path(session_dir),
        metadata=stored_metadata,
        arrays={
            "submitted_output_pcm_int16": submitted,
            "input_raw_int32": recorded,
            "preflight_raw_int32": preflight,
            **callback_arrays,
        },
    )
    return {"paths": paths, "metadata": stored_metadata, "valid": not invalid}


CALLBACK_TIME_INFO_FIELDS = (
    "callback_start_frames",
    "callback_frame_counts",
    "input_buffer_adc_time",
    "output_buffer_dac_time",
    "callback_current_time",
)


def validate_callback_time_info(
    value: dict[str, np.ndarray] | None, *, expected_frames: int
) -> dict[str, Any]:
    """PortAudio time_info를 monotonic/slip witness로만 검증한다."""

    if not isinstance(value, dict):
        raise ValueError("callback time_info mapping이 필요합니다")
    expected = int(expected_frames)
    if expected <= 0:
        raise ValueError("callback expected_frames는 양수여야 합니다")
    arrays = {
        name: np.asarray(value.get(name)).reshape(-1)
        for name in CALLBACK_TIME_INFO_FIELDS
    }
    lengths = {array.size for array in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
        raise ValueError("callback time_info vector 길이가 같고 1개 이상이어야 합니다")
    raw_starts = arrays["callback_start_frames"]
    raw_counts = arrays["callback_frame_counts"]
    if raw_starts.dtype.kind not in "iu" or raw_counts.dtype.kind not in "iu":
        raise ValueError("callback frame start/count는 exact integer 배열이어야 합니다")
    starts = raw_starts.astype(np.int64)
    counts = raw_counts.astype(np.int64)
    if starts[0] != 0 or np.any(counts <= 0) or np.any(starts < 0):
        raise ValueError("callback frame witness 시작/count가 잘못됐습니다")
    if np.any(starts >= expected):
        raise ValueError("callback frame witness가 capture 완료 뒤에도 존재합니다")
    expected_starts = [0]
    for index in range(starts.size - 1):
        consumed = min(int(counts[index]), expected - int(starts[index]))
        expected_starts.append(int(starts[index]) + max(0, consumed))
    expected_start_array = np.asarray(expected_starts, dtype=np.int64)
    sample_slips = int(np.count_nonzero(starts != expected_start_array))
    final_consumed = min(int(counts[-1]), expected - int(starts[-1]))
    if int(starts[-1]) + max(0, final_consumed) != expected:
        sample_slips += 1
    for name in (
        "input_buffer_adc_time",
        "output_buffer_dac_time",
        "callback_current_time",
    ):
        times = arrays[name].astype(np.float64)
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise ValueError(f"{name}가 finite strict-monotonic이 아닙니다")
    if sample_slips != 0:
        raise ValueError(f"callback frame sample slip이 {sample_slips}개입니다")
    return {
        "valid": True,
        "callback_count": int(starts.size),
        "sample_slip_count": 0,
        "first_start_frame": int(starts[0]),
        "last_start_frame": int(starts[-1]),
    }


def execute_live_capture(
    *,
    hardware_path: str | Path,
    saved_plan: dict[str, Any],
    planned_pcm: np.ndarray,
    session_dir: Path,
    meter_raw_path: str | Path,
    level_evidence_path: str | Path,
    operator_confirmations: dict[str, bool],
    sounddevice_module: Any | None = None,
    capture_function: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]] = (
        mpi.capture_measurement_preserving_partial
    ),
    audio_lock_factory: Callable[..., ContextManager[Any]] = repository_audio_lock,
) -> dict[str, Any]:
    """명시적 권한·fresh evidence 뒤 한 번 캡처하고 immutable raw만 발행한다."""

    required_confirmations = {
        "speaker_output": True,
        "user_present": True,
        "volume_minimum": True,
        "routing_and_geometry": True,
        "same_amplifier_setting": True,
    }
    if operator_confirmations != required_confirmations:
        raise ValueError(f"광대역 live operator confirmation 계약 위반: {operator_confirmations!r}")
    plan = saved_plan.get("payload")
    if not isinstance(plan, dict) or plan.get("schema") != BROADBAND_MEASUREMENT_PLAN_SCHEMA:
        raise ValueError("exact saved broadband plan authority가 필요합니다")
    validate_plan_pcm_exact(plan, planned_pcm)
    authority = validate_live_authority_binding(saved_plan, planned_pcm)
    plan_path = Path(saved_plan["path"]).resolve()
    plan_file_sha = str(saved_plan["file_sha256"])
    plan_payload_sha = str(saved_plan["payload_sha256"])
    if _sha256_file(plan_path) != plan_file_sha or _plan_payload_sha256(plan) != plan_payload_sha:
        raise ValueError("live 진입 전 saved plan SHA가 다릅니다")
    validate_fresh_raw_session_target(session_dir)

    hardware_file = _repository_path(hardware_path)
    hardware_sha = _sha256_file(hardware_file)
    if plan.get("hardware", {}).get("sha256") != hardware_sha:
        raise ValueError("plan hardware SHA가 현재 YAML과 다릅니다")
    contract = ControlBandContract.broadband_point_control()
    if plan.get("control_band_contract_sha256") != contract.digest():
        raise ValueError("plan control-band contract SHA가 현재 코드와 다릅니다")
    hardware_config = load_yaml(hardware_file)
    audio, channel_map = validate_measurement_hardware_contract(hardware_config)
    physical_fingerprint = collect_alsa_physical_fingerprint(hardware_config)
    hardware_identity = measurement_hardware_identity(
        hardware_config, physical_fingerprint=physical_fingerprint
    )
    level_evidence = load_measurement_level_evidence(
        level_evidence_path, repository_root=REPO_ROOT
    )
    if level_evidence.get("hardware_identity") != hardware_identity:
        raise ValueError("paired level evidence hardware identity가 현재 장치와 다릅니다")
    evidence_path = Path(str(level_evidence["_evidence_path"])).resolve()
    evidence_sha = str(level_evidence["_evidence_sha256"])
    meter = validate_bootstrap_meter_raw(
        meter_raw_path,
        repository_root=REPO_ROOT,
        expected_hardware_identity=hardware_identity,
        require_fresh=True,
    )
    plan_binding = {
        "path": str(authority["path"]),
        "file_sha256": plan_file_sha,
        "payload_sha256": plan_payload_sha,
        "pcm_sha256": str(plan["output"]["pcm_sha256"]),
    }
    validate_meter_followup_binding(
        meter_metadata=meter["metadata"],
        plan_binding=plan_binding,
        hardware_path=hardware_file,
        hardware_sha256=hardware_sha,
        level_evidence_path=evidence_path,
        level_evidence_sha256=evidence_sha,
        raw_session_dir=session_dir,
    )
    output_float = validate_plan_pcm_exact(plan, planned_pcm)

    if sounddevice_module is None:
        import importlib

        sounddevice_module = importlib.import_module("sounddevice")
    sd = sounddevice_module
    preflight_raw: np.ndarray | None = None
    preflight_report: dict[str, Any] | None = None
    in_dev: int | None = None
    out_dev: int | None = None
    audio_lock: dict[str, Any] = {}
    capture_id = uuid.uuid4().hex
    started_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    partial_error: mpi.PartialCaptureError | None = None
    post_capture_binding_error: str | None = None

    with audio_lock_factory(REPO_ROOT, purpose="broadband_interleaved_paths") as lock:
        audio_lock = dict(lock)
        assert_live_pcm_clock_preconditions(audio)
        analysed_seconds = DEFAULT_INPUT_PREFLIGHT_SECONDS - float(
            mpi.cw.DEFAULT_PROBE_SETTLE_SECONDS
        )
        if analysed_seconds <= 0.0:
            raise ValueError("광대역 input preflight total은 I2S settle보다 길어야 합니다")
        preflight_raw, preflight_report = mpi.cw._capture_preflight(
            sd, audio, analysed_seconds
        )
        channels = preflight_report.get("channels", [])
        required_indices = (channel_map["error_mic"], channel_map["reference_mic"])
        if len(channels) < 2 or not all(
            bool(channels[index].get("valid")) for index in required_indices
        ):
            raise RuntimeError("양 마이크 3초 input-only preflight가 PASS하지 않았습니다")
        in_dev = int(preflight_report["device"])
        out_dev = int(
            resolve_alsa_portaudio_device(
                audio["output"]["card"], audio["output"]["pcm"], "output", 2
            )
        )
        meter_devices = meter["metadata"].get("resolved_devices")
        if meter_devices != {"input": in_dev, "output": out_dev}:
            raise RuntimeError(
                "fresh meter 이후 PortAudio device mapping이 달라졌습니다: "
                f"meter={meter_devices!r}, capture={{'input': {in_dev}, 'output': {out_dev}}}"
            )
        # 모든 input-only/evidence/device gate를 통과한 뒤, output open 직전에만
        # immutable raw 목적지를 no-replace로 예약한다.
        created_session = mpi.create_session_directory(
            session_dir.parent, session_dir.name
        )

        def pre_open_check() -> None:
            try:
                validate_live_authority_binding(saved_plan, planned_pcm)
            except ValueError as exc:
                raise RuntimeError(
                    f"preflight 이후 saved plan authority가 변경됐습니다: {exc}"
                ) from exc
            if _sha256_file(hardware_file) != hardware_sha:
                raise RuntimeError("preflight 이후 hardware YAML이 변경됐습니다")
            if _sha256_file(evidence_path) != evidence_sha:
                raise RuntimeError("preflight 이후 level evidence가 변경됐습니다")
            refreshed_meter = validate_bootstrap_meter_raw(
                meter_raw_path,
                repository_root=REPO_ROOT,
                expected_hardware_identity=hardware_identity,
                require_fresh=True,
            )
            if refreshed_meter["sha256"] != meter["sha256"]:
                raise RuntimeError("preflight 이후 fresh meter raw가 변경됐습니다")
            validate_meter_followup_binding(
                meter_metadata=refreshed_meter["metadata"],
                plan_binding=plan_binding,
                hardware_path=hardware_file,
                hardware_sha256=hardware_sha,
                level_evidence_path=evidence_path,
                level_evidence_sha256=evidence_sha,
                raw_session_dir=session_dir,
            )
            if collect_alsa_physical_fingerprint(hardware_config) != physical_fingerprint:
                raise RuntimeError("preflight 이후 ALSA physical fingerprint가 변경됐습니다")
            for name in ("raw_measurement.npz", "metadata.json"):
                if (created_session / name).exists():
                    raise RuntimeError(f"output 직전 raw target이 이미 존재합니다: {name}")
            assert_live_pcm_clock_preconditions(audio)

        try:
            recorded_raw, submitted_pcm, telemetry = mpi.capture_with_speaker_release_notice(
                lambda: capture_function(
                    sd,
                    fs=int(audio["sample_rate"]),
                    block_size=int(audio["block_size"]),
                    latency=str(audio["latency"]),
                    in_dev=in_dev,
                    out_dev=out_dev,
                    output_float=output_float,
                    meter_completed_at_utc=meter["completed_at_utc"],
                    pre_open_check=pre_open_check,
                    record_callback_time_info=True,
                )
            )
        except mpi.PartialCaptureError as exc:
            partial_error = exc
            recorded_raw = exc.recorded_raw
            submitted_pcm = exc.output_pcm
            telemetry = exc.telemetry
        # stream close와 즉시 분리 안내가 끝난 뒤, raw를 쓰기 전에 같은 exact
        # authority/evidence/device binding을 한 번 더 읽는다. 캡처 중 TOCTOU가
        # 발견돼도 raw를 버리지 않고 아래에서 INVALID immutable raw로 보존한다.
        try:
            pre_open_check()
        except Exception as exc:  # raw-first: 실패도 캡처와 함께 보존
            post_capture_binding_error = f"{type(exc).__name__}: {exc}"

    assert preflight_raw is not None and preflight_report is not None
    callback_time_info = telemetry.pop("callback_time_info", None)
    invalid_reasons = mpi.capture_telemetry_invalid_reasons(telemetry)
    if partial_error is not None and "capture_incomplete" not in invalid_reasons:
        invalid_reasons.append("capture_incomplete")
    if post_capture_binding_error is not None:
        invalid_reasons.append("post_capture_binding_invalid")
    metadata = {
        "capture_id": capture_id,
        "started_at_utc": started_at_utc,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sample_rate": int(audio["sample_rate"]),
        "block_size": int(audio["block_size"]),
        "latency": str(audio["latency"]),
        "channel_map": dict(channel_map),
        "operator_confirmations": dict(operator_confirmations),
        "hardware_identity": hardware_identity,
        "hardware": {"path": str(hardware_file), "sha256": hardware_sha},
        "audio_lock": audio_lock,
        "resolved_devices": {"input": int(in_dev), "output": int(out_dev)},
        "input_preflight_seconds": DEFAULT_INPUT_PREFLIGHT_SECONDS,
        "preflight": preflight_report,
        "telemetry": telemetry,
        "plan": {
            "path": str(authority["path"]),
            "file_sha256": plan_file_sha,
            "payload_sha256": plan_payload_sha,
            "pcm_sha256": plan["output"]["pcm_sha256"],
            "schema": BROADBAND_MEASUREMENT_PLAN_SCHEMA,
        },
        "control_band_contract_sha256": contract.digest(),
        "meter": {
            "path": str(meter["path"]),
            "receipt_path": str(meter["receipt_path"]),
            "raw_sha256": meter["sha256"],
            "completed_at_utc": meter["completed_at_utc"].isoformat(),
            "meter_ch0_dbfs": float(meter["meter_ch0_dbfs"]),
            "freshness_max_seconds": BOOTSTRAP_METER_MAX_AGE_SECONDS,
        },
        "level_evidence": {"path": str(evidence_path), "sha256": evidence_sha},
        "analysis_status": "NOT_RUN_RAW_FIRST",
        "post_capture_binding": {
            "valid": post_capture_binding_error is None,
            "error": post_capture_binding_error,
        },
        "invalid_reasons": invalid_reasons,
    }
    published = publish_broadband_raw_capture(
        session_dir=created_session,
        metadata=metadata,
        planned_pcm=planned_pcm,
        submitted_pcm=submitted_pcm,
        input_raw_int32=recorded_raw,
        preflight_raw_int32=preflight_raw,
        callback_time_info=callback_time_info,
    )
    published["partial_capture_error"] = (
        None if partial_error is None else str(partial_error)
    )
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-live", action="store_true")
    parser.add_argument("--amplitude", type=float, default=DEFAULT_AMPLITUDE)
    parser.add_argument("--period-seconds", type=float, default=DEFAULT_PERIOD_SECONDS)
    parser.add_argument("--warmup-seconds", type=float, default=DEFAULT_WARMUP_SECONDS)
    parser.add_argument("--repeats-per-panel", type=int, default=DEFAULT_REPEATS_PER_PANEL)
    parser.add_argument("--transition-seconds", type=float, default=DEFAULT_TRANSITION_SECONDS)
    parser.add_argument("--lead-in-seconds", type=float, default=DEFAULT_LEAD_IN_SECONDS)
    parser.add_argument("--hard-max-seconds", type=float, default=DEFAULT_HARD_MAX_SECONDS)
    parser.add_argument(
        "--output",
        type=Path,
        help="signal-only 계획 JSON을 no-replace로 저장합니다. PCM이나 오디오는 저장하지 않습니다",
    )
    parser.add_argument("--plan", type=Path, help="live에서 실행할 저장된 exact dry-run plan JSON")
    parser.add_argument(
        "--raw-session-dir",
        type=Path,
        help="no-replace broadband raw session 경로. 생략하면 plan SHA로 results/ 아래 유도",
    )
    parser.add_argument("--meter-raw", type=Path)
    parser.add_argument(
        "--level-evidence",
        type=Path,
        default=DEFAULT_MEASUREMENT_LEVEL_EVIDENCE_PATH,
    )
    parser.add_argument("--confirm-speaker", action="store_true")
    parser.add_argument("--confirm-user-present", action="store_true")
    parser.add_argument("--confirm-volume-minimum", action="store_true")
    parser.add_argument("--confirm-routing-and-geometry", action="store_true")
    parser.add_argument("--confirm-same-amplifier-setting", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run and not args.execute_live:
        print(
            "[중단] 광대역 live 출력은 아직 잠겨 있습니다. --dry-run으로 exact plan을 "
            "저장한 뒤, 별도 --execute-live와 모든 확인·fresh evidence를 명시해야만 열립니다.",
            file=sys.stderr,
        )
        return 2
    if args.execute_live:
        confirmations = {
            "speaker_output": bool(args.confirm_speaker),
            "user_present": bool(args.confirm_user_present),
            "volume_minimum": bool(args.confirm_volume_minimum),
            "routing_and_geometry": bool(args.confirm_routing_and_geometry),
            "same_amplifier_setting": bool(args.confirm_same_amplifier_setting),
        }
        if not all(confirmations.values()):
            print(
                "[중단] 광대역 live는 --confirm-speaker, --confirm-user-present, "
                "--confirm-volume-minimum, --confirm-routing-and-geometry, "
                "--confirm-same-amplifier-setting이 모두 필요합니다.",
                file=sys.stderr,
            )
            return 2
        if args.plan is None or args.meter_raw is None:
            print(
                "[중단] 광대역 live는 --plan <dry-run.json>과 바로 직전 fresh PASS "
                "--meter-raw <meter_raw.npz>가 모두 필요합니다.",
                file=sys.stderr,
            )
            return 2
    try:
        plan, planned_pcm = build_signal_plan(
            hardware_path=args.hardware,
            amplitude=args.amplitude,
            period_seconds=args.period_seconds,
            warmup_seconds=args.warmup_seconds,
            repeats_per_panel=args.repeats_per_panel,
            transition_seconds=args.transition_seconds,
            lead_in_seconds=args.lead_in_seconds,
            hard_max_seconds=args.hard_max_seconds,
        )
        raw_session = raw_session_path_for_plan(plan, args.raw_session_dir)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.dry_run:
        try:
            if args.output is not None and args.output.expanduser().resolve().exists():
                raise FileExistsError(
                    f"기존 계획은 덮어쓰지 않습니다: {args.output.expanduser().resolve()}"
                )
            environment = validate_dry_run_environment(
                hardware_path=args.hardware,
                raw_session_dir=raw_session,
            )
        except (FileExistsError, KeyError, OSError, RuntimeError, ValueError) as exc:
            print(f"[DRY-RUN 중단] {exc}", file=sys.stderr)
            return 2
        if args.output is not None:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                with output.open("x", encoding="utf-8") as handle:
                    handle.write(rendered)
            except FileExistsError:
                print(f"[중단] 기존 계획은 덮어쓰지 않습니다: {output}", file=sys.stderr)
                return 2
            print(f"[saved] {output}", file=sys.stderr)
        print(rendered, end="")
        print(
            "[PASS] signal-only 광대역 측정 계획. sounddevice import/open과 스피커 출력 0회.\n"
            f"  ALSA input=hw:{environment['input_alsa'][0]},{environment['input_alsa'][1]} "
            f"output=hw:{environment['output_alsa'][0]},{environment['output_alsa'][1]}\n"
            f"  fresh raw target={raw_session}",
            file=sys.stderr,
        )
        return 0

    try:
        saved_plan = load_exact_saved_plan(
            args.plan, expected_plan=plan, expected_pcm=planned_pcm
        )
        print(
            "[LIVE 실행 직전] noise speaker ch0 + cancel speaker ch1이 함께 동작합니다.\n"
            f"  nominal output={plan['output']['duration_seconds']:.3f}초, "
            f"stream watchdog deadline="
            f"{plan['output']['duration_seconds'] + mpi.LIVE_WATCHDOG_GRACE_SECONDS:.3f}초\n"
            "  앰프 볼륨 최저·사용자 입회 상태를 유지하고 출력 종료 안내 즉시 분리하세요.",
            flush=True,
        )
        published = execute_live_capture(
            hardware_path=args.hardware,
            saved_plan=saved_plan,
            planned_pcm=planned_pcm,
            session_dir=raw_session,
            meter_raw_path=args.meter_raw,
            level_evidence_path=args.level_evidence,
            operator_confirmations=confirmations,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TimeoutError,
        ValueError,
    ) as exc:
        print(f"[실패] 광대역 live/raw publisher: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    raw_path = published["paths"]["raw"]
    if not published["valid"]:
        print(
            f"[INVALID] 광대역 raw는 보존됐지만 승격 불가: {raw_path}; "
            f"reasons={published['metadata']['invalid_reasons']}",
            file=sys.stderr,
        )
        return 1
    print(
        f"[PASS] 광대역 immutable raw-first 저장: {raw_path}. "
        "분석은 아직 실행하지 않았습니다.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
