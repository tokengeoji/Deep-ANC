"""Full-Nyquist causal P/S 식별용 결정적 저-crest probe.

이 모듈은 파형과 수치 영수증만 만든다. 오디오 장치를 import하거나 열지 않는다.
기존 100--11.314 kHz panel 측정은 제어 대역 복소 전달함수의 authority이고,
여기서 만드는 두 파형은 causal history/support/tail을 식별하기 위한 별도 authority다.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .interleaved_probe import schroeder_phases


SAMPLE_RATE = 48_000
BLOCK_SIZE = 256
FIT_PERIOD_SAMPLES = 32_768
HOLDOUT_PERIOD_SAMPLES = 30_720
FIT_ANALYSIS_REPEATS = 16
HOLDOUT_ANALYSIS_REPEATS = 8
WARMUP_REPEATS = 1
LEAD_SILENCE_SAMPLES = 24_000
GUARD_SAMPLES = 24_000
MAXIMUM_DELAY_SAMPLES = 4_800
CANDIDATE_SUPPORT_SAMPLES = (1_024, 2_048, 4_096)
TAIL_SCAN_SAMPLES = 16_384
TARGET_PEAK_PCM = 98
MIN_RMS = 0.00140
MAX_RMS = 0.00143
MIN_CREST_DB = 6.3
MAX_CREST_DB = 6.7
MAX_DESIGN_CONDITION = 1.10

FIT_PHASE_SEED = 20_293_596
HOLDOUT_PHASE_SEED = 20_291_548
PHASE_BLEND_GRID = np.linspace(0.0, 0.30, 301, dtype=np.float64)
TARGET_CREST_DB = 6.50


@dataclass(frozen=True)
class ProbePeriod:
    pcm_int16: np.ndarray
    metadata: dict[str, object]


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value).tobytes(order="C")).hexdigest()


def _crest_db(value: np.ndarray) -> float:
    signal = np.asarray(value, dtype=np.float64).reshape(-1)
    rms = float(np.sqrt(np.mean(signal * signal)))
    peak = float(np.max(np.abs(signal)))
    if rms <= 0.0 or peak <= 0.0:
        raise ValueError("probe가 무음입니다")
    return float(20.0 * np.log10(peak / rms))


def _all_bin_period(*, frames: int, seed: int) -> ProbePeriod:
    """모든 real-FFT bin의 intended magnitude가 같은 결정적 파형을 만든다."""

    length = int(frames)
    if length <= MAXIMUM_DELAY_SAMPLES + TAIL_SCAN_SAMPLES:
        raise ValueError("period가 maximum delay+tail scan보다 길어야 합니다")
    bins = length // 2 + 1
    base = schroeder_phases(bins)
    random_phase = np.random.default_rng(int(seed)).uniform(0.0, 2.0 * np.pi, bins)
    best: tuple[float, np.ndarray, float] | None = None
    for blend in PHASE_BLEND_GRID:
        phase = (1.0 - float(blend)) * base + float(blend) * random_phase
        spectrum = np.exp(1j * phase)
        # real rFFT의 DC/Nyquist는 real이어야 하며 magnitude도 1로 유지한다.
        spectrum[0] = 1.0
        spectrum[-1] = 1.0
        candidate = np.fft.irfft(spectrum, n=length)
        crest = _crest_db(candidate)
        score = abs(crest - TARGET_CREST_DB)
        if best is None or score < best[0]:
            best = (score, candidate, float(blend))
    assert best is not None
    candidate = best[1]
    scaled = candidate / float(np.max(np.abs(candidate))) * 0.003
    pcm = np.rint(np.clip(scaled, -1.0, 1.0) * 32_767.0).astype(np.int16)
    decoded = pcm.astype(np.float64) / 32_767.0
    peak_pcm = int(np.max(np.abs(pcm.astype(np.int32))))
    rms = float(np.sqrt(np.mean(decoded * decoded)))
    crest = _crest_db(decoded)
    spectrum_pcm = np.fft.rfft(pcm.astype(np.float64))
    magnitudes = np.abs(spectrum_pcm)
    if np.any(magnitudes <= 0.0):
        raise ValueError("actual submitted int16 spectrum에 zero bin이 있습니다")
    condition = float(np.max(magnitudes) / np.min(magnitudes))
    if peak_pcm != TARGET_PEAK_PCM:
        raise ValueError(f"actual peak PCM이 {TARGET_PEAK_PCM}이 아닙니다: {peak_pcm}")
    if not MIN_RMS <= rms <= MAX_RMS:
        raise ValueError(f"actual PCM RMS가 계약 밖입니다: {rms}")
    if not MIN_CREST_DB <= crest <= MAX_CREST_DB:
        raise ValueError(f"actual PCM crest가 계약 밖입니다: {crest}")
    if condition > MAX_DESIGN_CONDITION:
        raise ValueError(f"actual PCM design condition이 너무 큽니다: {condition}")
    metadata: dict[str, object] = {
        "frames": length,
        "sample_rate": SAMPLE_RATE,
        "block_aligned": length % BLOCK_SIZE == 0,
        "frequency_resolution_hz": SAMPLE_RATE / length,
        "delay_alias_period_samples": length,
        "all_rfft_bins_nonzero": True,
        "numeric_rank": length,
        "design_condition_number": condition,
        "peak_pcm": peak_pcm,
        "rms": rms,
        "crest_db": crest,
        "phase_seed": int(seed),
        "phase_blend": best[2],
        "pcm_sha256": _sha256_array(pcm),
    }
    return ProbePeriod(pcm_int16=pcm, metadata=metadata)


def build_fit_period() -> ProbePeriod:
    return _all_bin_period(frames=FIT_PERIOD_SAMPLES, seed=FIT_PHASE_SEED)


def build_holdout_period() -> ProbePeriod:
    return _all_bin_period(
        frames=HOLDOUT_PERIOD_SAMPLES, seed=HOLDOUT_PHASE_SEED
    )


def off_grid_holdout_bins(
    *, low_hz: float = 100.0, high_hz: float = 11_313.708498984761
) -> np.ndarray:
    """fit 1.4648 Hz grid와 panel 8 Hz grid에 없는 holdout bin."""

    resolution = SAMPLE_RATE / HOLDOUT_PERIOD_SAMPLES
    first = int(math.ceil(float(low_hz) / resolution))
    last = int(math.floor(float(high_hz) / resolution))
    bins = np.arange(first, last + 1, dtype=np.int64)
    # df_B/df_A=16/15이므로 B bin%15==0만 A grid와 겹친다.
    # df_B=25/16 Hz이므로 B bin%128==0만 8 Hz grid와 겹친다.
    return bins[(bins % 15 != 0) & (bins % 128 != 0)]


def build_signal_plan() -> tuple[dict[str, object], np.ndarray]:
    fit = build_fit_period()
    holdout = build_holdout_period()
    rows: list[np.ndarray] = []
    layout: list[dict[str, object]] = []
    cursor = 0

    def append(kind: str, value: np.ndarray, **metadata: object) -> None:
        nonlocal cursor
        array = np.asarray(value, dtype=np.int16)
        rows.append(array)
        layout.append(
            {
                "kind": kind,
                "start_frame": cursor,
                "stop_frame": cursor + int(array.shape[0]),
                "frames": int(array.shape[0]),
                **metadata,
            }
        )
        cursor += int(array.shape[0])

    def silence(frames: int) -> np.ndarray:
        return np.zeros((int(frames), 2), dtype=np.int16)

    append("lead_in_silence", silence(LEAD_SILENCE_SAMPLES))
    for path, channel in (("primary", 0), ("secondary", 1)):
        fit_block = np.zeros(
            ((WARMUP_REPEATS + FIT_ANALYSIS_REPEATS) * FIT_PERIOD_SAMPLES, 2),
            dtype=np.int16,
        )
        fit_block[:, channel] = np.tile(
            fit.pcm_int16, WARMUP_REPEATS + FIT_ANALYSIS_REPEATS
        )
        append(
            f"{path}_fit",
            fit_block,
            path=path,
            output_channel=channel,
            other_channel_silent=True,
            period_samples=FIT_PERIOD_SAMPLES,
            warmup_repeats=WARMUP_REPEATS,
            analysis_repeats=FIT_ANALYSIS_REPEATS,
        )
        append(f"{path}_fit_tail_guard", silence(GUARD_SAMPLES), path=path)
        holdout_block = np.zeros(
            (
                (WARMUP_REPEATS + HOLDOUT_ANALYSIS_REPEATS)
                * HOLDOUT_PERIOD_SAMPLES,
                2,
            ),
            dtype=np.int16,
        )
        holdout_block[:, channel] = np.tile(
            holdout.pcm_int16, WARMUP_REPEATS + HOLDOUT_ANALYSIS_REPEATS
        )
        append(
            f"{path}_holdout",
            holdout_block,
            path=path,
            output_channel=channel,
            other_channel_silent=True,
            period_samples=HOLDOUT_PERIOD_SAMPLES,
            warmup_repeats=WARMUP_REPEATS,
            analysis_repeats=HOLDOUT_ANALYSIS_REPEATS,
        )
        append(f"{path}_holdout_tail_guard", silence(GUARD_SAMPLES), path=path)
    output = np.concatenate(rows, axis=0)
    padding = (-output.shape[0]) % BLOCK_SIZE
    if padding:
        append("block_padding_silence", silence(padding))
        output = np.concatenate(rows, axis=0)
    if int(output.shape[0]) != cursor:
        raise AssertionError("layout cursor가 output과 다릅니다")
    idle_violations = 0
    for row in layout:
        if row.get("other_channel_silent"):
            start, stop = int(row["start_frame"]), int(row["stop_frame"])
            idle = 1 - int(row["output_channel"])
            idle_violations += int(np.count_nonzero(output[start:stop, idle]))
    off_grid = off_grid_holdout_bins()
    active_frames = sum(
        int(row["frames"])
        for row in layout
        if str(row["kind"]).endswith(("_fit", "_holdout"))
    )
    plan: dict[str, object] = {
        "schema": "fullband_causal_ps_signal_plan_v1",
        "role": "signal_only_dry_run_no_audio",
        "live_capture_enabled": False,
        "sample_rate": SAMPLE_RATE,
        "block_size": BLOCK_SIZE,
        "latency": "low",
        "fit": fit.metadata,
        "holdout": holdout.metadata,
        "layout": layout,
        "support_contract": {
            "maximum_delay_samples": MAXIMUM_DELAY_SAMPLES,
            "candidate_support_samples": list(CANDIDATE_SUPPORT_SAMPLES),
            "tail_scan_samples": TAIL_SCAN_SAMPLES,
            "compact_partial_band_promotion_forbidden": True,
            "handoff_baked_into_secondary_fir": False,
        },
        "off_grid_holdout": {
            "low_hz": 100.0,
            "high_hz": 11_313.708498984761,
            "bin_count": int(off_grid.size),
            "first_hz": float(off_grid[0] * SAMPLE_RATE / HOLDOUT_PERIOD_SAMPLES),
            "last_hz": float(off_grid[-1] * SAMPLE_RATE / HOLDOUT_PERIOD_SAMPLES),
            "excludes_fit_grid": True,
            "excludes_panel_8hz_grid": True,
        },
        "output": {
            "frames": int(output.shape[0]),
            "channels": 2,
            "dtype": "int16",
            "duration_seconds": float(output.shape[0] / SAMPLE_RATE),
            "active_slot_duration_seconds": float(active_frames / SAMPLE_RATE),
            "peak_pcm": int(np.max(np.abs(output.astype(np.int32)))),
            "idle_channel_nonzero_count": idle_violations,
            "padding_frames": int(padding),
            "pcm_sha256": _sha256_array(output),
        },
    }
    return plan, output


__all__ = [
    "BLOCK_SIZE",
    "CANDIDATE_SUPPORT_SAMPLES",
    "FIT_ANALYSIS_REPEATS",
    "FIT_PERIOD_SAMPLES",
    "GUARD_SAMPLES",
    "HOLDOUT_ANALYSIS_REPEATS",
    "HOLDOUT_PERIOD_SAMPLES",
    "MAXIMUM_DELAY_SAMPLES",
    "MAX_DESIGN_CONDITION",
    "SAMPLE_RATE",
    "TAIL_SCAN_SAMPLES",
    "build_fit_period",
    "build_holdout_period",
    "build_signal_plan",
    "off_grid_holdout_bins",
]
