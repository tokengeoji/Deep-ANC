"""Canonical 후보용 단발 aperiodic burst + 직접 관측 zero-tail 계획."""

from __future__ import annotations

import hashlib
import math

import numpy as np

from .interleaved_probe import schroeder_phases

FS = 48_000
BLOCK = 256
BURSTS = (("fit_a", 131_072, 310_001), ("fit_b", 130_816, 410_009), ("holdout", 130_560, 510_031))
GUARD = 65_536
LEAD = 24_000
TARGET_BAND = (100.0, 11_313.708498984761)
STABILITY_BANDS = ((64.0, 100.0), (11_313.708498984761, 14_000.0))
MAX_DELAY = 4_800
SUPPORTS = (1_024, 2_048, 4_096)
MAX_OBSERVED_POST_ONSET = 32_768


def _burst(length: int, seed: int) -> tuple[np.ndarray, dict[str, object]]:
    df = FS / length
    freq = np.fft.rfftfreq(length, 1.0 / FS)
    magnitude = np.zeros(freq.size, dtype=np.float64)
    target = (freq >= TARGET_BAND[0]) & (freq <= TARGET_BAND[1])
    stability = ((freq >= 64.0) & (freq < TARGET_BAND[0])) | (
        (freq > TARGET_BAND[1]) & (freq <= 14_000.0)
    )
    magnitude[target] = 1.0
    magnitude[stability] = 0.1
    magnitude[0] = 0.0
    magnitude[-1] = 0.0
    active = np.flatnonzero(magnitude > 0.0)
    base = schroeder_phases(active.size)
    random_phase = np.random.default_rng(seed).uniform(0.0, 2 * np.pi, active.size)
    best = None
    for blend in np.linspace(0.0, 0.30, 61):
        spectrum = np.zeros(freq.size, dtype=np.complex128)
        spectrum[active] = magnitude[active] * np.exp(
            1j * ((1.0 - blend) * base + blend * random_phase)
        )
        signal = np.fft.irfft(spectrum, n=length)
        crest = 20 * np.log10(np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)))
        score = abs(crest - 6.5)
        if best is None or score < best[0]:
            best = (score, signal, float(blend))
    assert best is not None
    signal = best[1] / np.max(np.abs(best[1])) * 0.003
    pcm = np.rint(signal * 32767.0).astype(np.int16)
    decoded = pcm.astype(np.float64) / 32767.0
    rms = float(np.sqrt(np.mean(decoded**2)))
    crest = float(20 * np.log10(np.max(np.abs(decoded)) / rms))
    spectrum = np.fft.rfft(decoded)
    target_mag = np.abs(spectrum[target])
    condition = float(target_mag.max() / target_mag.min())
    if int(np.max(np.abs(pcm.astype(np.int32)))) != 98:
        raise ValueError("aperiodic burst peak PCM이 98이 아닙니다")
    if not 0.00140 <= rms <= 0.00143 or not 6.3 <= crest <= 6.7:
        raise ValueError(f"aperiodic burst RMS/crest가 열 예산 밖입니다: {rms}/{crest}")
    if condition > 1.10:
        raise ValueError(f"target-band actual PCM condition이 너무 큽니다: {condition}")
    return pcm, {
        "frames": length,
        "seed": seed,
        "frequency_resolution_hz": df,
        "target_band_hz": list(TARGET_BAND),
        "stability_bands_hz": [list(x) for x in STABILITY_BANDS],
        "dc_and_nyquist_canonical_identification": False,
        "peak_pcm": 98,
        "rms": rms,
        "crest_db": crest,
        "target_band_condition_number": condition,
        "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
    }


def build_aperiodic_plan() -> tuple[dict[str, object], np.ndarray]:
    probes = {name: _burst(length, seed) for name, length, seed in BURSTS}
    arrays: list[np.ndarray] = []
    layout: list[dict[str, object]] = []
    cursor = 0

    def add(kind: str, array: np.ndarray, **extra: object) -> None:
        nonlocal cursor
        arrays.append(array)
        layout.append({"kind": kind, "start_frame": cursor, "stop_frame": cursor + len(array), "frames": len(array), **extra})
        cursor += len(array)

    add("lead_silence", np.zeros((LEAD, 2), np.int16))
    for path, channel in (("primary", 0), ("secondary", 1)):
        for role, _, _ in BURSTS:
            period = probes[role][0]
            slot = np.zeros((period.size, 2), np.int16)
            slot[:, channel] = period
            add(f"{path}_{role}_burst", slot, path=path, role=role, output_channel=channel, other_channel_silent=True)
            add(f"{path}_{role}_zero_guard", np.zeros((GUARD, 2), np.int16), path=path, role=role)
    output = np.concatenate(arrays)
    padding = (-len(output)) % BLOCK
    if padding:
        add("block_padding_silence", np.zeros((padding, 2), np.int16))
        output = np.concatenate(arrays)
    active = 2 * sum(length for _, length, _ in BURSTS)
    plan = {
        "schema": "fullband_causal_aperiodic_signal_plan_v2",
        "role": "signal_only_dry_run_no_audio",
        "live_capture_enabled": False,
        "sample_rate": FS,
        "block_size": BLOCK,
        "bursts": {name: meta for name, (_, meta) in probes.items()},
        "layout": layout,
        "tail_contract": {
            "zero_guard_samples": GUARD,
            "zero_guard_seconds": GUARD / FS,
            "maximum_delay_samples": MAX_DELAY,
            "candidate_post_onset_support_samples": list(SUPPORTS),
            "direct_observation_post_onset_samples": MAX_OBSERVED_POST_ONSET,
            "geometric_tail_extrapolation_forbidden": True,
        },
        "output": {
            "frames": len(output),
            "duration_seconds": len(output) / FS,
            "active_duration_seconds": active / FS,
            "zero_guard_duration_seconds": 6 * GUARD / FS,
            "peak_pcm": int(np.max(np.abs(output.astype(np.int32)))),
            "padding_frames": padding,
            "pcm_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        },
    }
    return plan, output


def linear_fft_convolve(x: np.ndarray, h: np.ndarray, *, n_fft: int | None = None) -> np.ndarray:
    x = np.asarray(x, np.float64).reshape(-1)
    h = np.asarray(h, np.float64).reshape(-1)
    required = x.size + h.size - 1
    if n_fft is None:
        n_fft = 1 << (required - 1).bit_length()
    if int(n_fft) < required:
        raise ValueError(f"linear FFT convolution n_fft={n_fft} < {required}")
    return np.fft.irfft(np.fft.rfft(x, int(n_fft)) * np.fft.rfft(h, int(n_fft)), int(n_fft))[:required]


__all__ = ["FS", "GUARD", "MAX_DELAY", "SUPPORTS", "TARGET_BAND", "build_aperiodic_plan", "linear_fft_convolve"]
