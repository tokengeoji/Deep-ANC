"""합성 소음원 생성기 — 물리적으로 상쇄 가능한 주기성/준정상 잡음이 1차 목표.

종류: 톤+고조파, AM/FM 기계음, 협대역 잡음, 느린 처프, 멀티톤.
덕트 공진(70/210/350/489/629Hz) 근방 기본주파수가 자주 뽑히도록 가중한다.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

DUCT_RESONANCES = np.array([70.0, 210.0, 350.0, 489.0, 629.0])

KINDS = ("tone_harmonics", "machine", "narrowband", "chirp", "multitone")


class SyntheticNoise:
    def __init__(
        self, sample_rate: int, seed: int | None = None, *,
        target_band_hz: tuple[float, float] | list[float] | None = None,
        target_probability: float = 0.0,
    ) -> None:
        self.fs = int(sample_rate)
        self.rng = np.random.default_rng(seed)
        self.target_band = None
        if isinstance(target_probability, (bool, str, np.bool_)):
            raise ValueError("target_probability는 [0,1]의 유한 수여야 합니다")
        try:
            probability = float(target_probability)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("target_probability는 [0,1]의 유한 수여야 합니다") from exc
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("target_probability는 [0,1]의 유한 수여야 합니다")
        if target_band_hz is not None:
            band = np.asarray(target_band_hz)
            if (band.shape != (2,) or band.dtype.kind not in "fiu"
                    or not np.isfinite(band).all() or not 0 < band[0] < band[1] < 0.45 * self.fs):
                raise ValueError("target_band_hz는 0 < low < high < 0.45*sample_rate여야 합니다")
            self.target_band = tuple(float(value) for value in band)
        elif probability != 0:
            raise ValueError("target_probability > 0에는 target_band_hz가 필요합니다")
        self.target_probability = probability

    def _use_target(self) -> bool:
        # opt-in이 아니면 난수도 추가 소비하지 않아 기존 seed/파형을 그대로 보존한다.
        return self.target_probability > 0 and self.rng.random() < self.target_probability

    def _pick_f0(self) -> float:
        if self._use_target():
            return float(self.rng.uniform(*self.target_band))
        if self.rng.random() < 0.4:
            base = float(self.rng.choice(DUCT_RESONANCES))
            return base * float(self.rng.uniform(0.95, 1.05))
        return float(self.rng.uniform(50.0, 1000.0))

    def tone_harmonics(self, n: int) -> np.ndarray:
        f0 = self._pick_f0()
        n_harm = int(self.rng.integers(1, 7))
        rolloff = float(self.rng.uniform(3.0, 12.0))  # dB/고조파
        t = np.arange(n) / self.fs
        out = np.zeros(n)
        for k in range(1, n_harm + 1):
            f = f0 * k
            if f >= self.fs / 2 * 0.9:
                break
            amp = 10.0 ** (-rolloff * (k - 1) / 20.0)
            out += amp * np.sin(2 * np.pi * f * t + self.rng.uniform(0, 2 * np.pi))
        return out.astype(np.float32)

    def machine(self, n: int) -> np.ndarray:
        """AM/FM 변조 회전기계 근사: 캐리어 + 저주파 변조 + 광대역 바닥."""
        f0 = self._pick_f0()
        t = np.arange(n) / self.fs
        am_rate = float(self.rng.uniform(0.5, 8.0))
        am_depth = float(self.rng.uniform(0.1, 0.5))
        fm_dev = f0 * float(self.rng.uniform(0.0, 0.02))
        fm_rate = float(self.rng.uniform(0.2, 4.0))
        phase = 2 * np.pi * (f0 * t + (fm_dev / max(fm_rate, 1e-6)) * np.sin(2 * np.pi * fm_rate * t))
        carrier = np.sin(phase) * (1.0 + am_depth * np.sin(2 * np.pi * am_rate * t))
        floor = self._filtered_noise(n, f0 * 0.5, min(f0 * 4.0, self.fs / 2 * 0.9))
        mix = carrier + float(self.rng.uniform(0.05, 0.3)) * floor
        return mix.astype(np.float32)

    def _filtered_noise(self, n: int, lo: float, hi: float) -> np.ndarray:
        lo = max(20.0, lo)
        hi = min(self.fs / 2 * 0.95, max(hi, lo * 1.2))
        sos = signal.butter(4, [lo, hi], btype="bandpass", fs=self.fs, output="sos")
        x = signal.sosfilt(sos, self.rng.standard_normal(n + 2048))[2048:]
        return x / (np.std(x) + 1e-9)

    def narrowband(self, n: int) -> np.ndarray:
        center = self._pick_f0()
        bw = center * float(self.rng.uniform(0.05, 0.3))
        return self._filtered_noise(n, center - bw / 2, center + bw / 2).astype(np.float32)

    def chirp(self, n: int) -> np.ndarray:
        if self._use_target():
            f_start, f_end = (float(value) for value in self.rng.uniform(*self.target_band, size=2))
        else:
            f_start = float(self.rng.uniform(60.0, 500.0))
            f_end = f_start * float(self.rng.uniform(0.5, 2.0))
        t = np.arange(n) / self.fs
        return signal.chirp(t, f_start, t[-1], f_end, method="logarithmic").astype(np.float32)

    def multitone(self, n: int) -> np.ndarray:
        n_tones = int(self.rng.integers(2, 5))
        t = np.arange(n) / self.fs
        out = np.zeros(n)
        for _ in range(n_tones):
            f = self._pick_f0()
            out += float(self.rng.uniform(0.3, 1.0)) * np.sin(
                2 * np.pi * f * t + self.rng.uniform(0, 2 * np.pi)
            )
        return out.astype(np.float32)

    def generate(self, n: int, kind: str | None = None) -> np.ndarray:
        if kind is None:
            kind = str(self.rng.choice(KINDS))
        out = getattr(self, kind)(n)
        rms = float(np.sqrt(np.mean(out**2)) + 1e-9)
        return (out / rms).astype(np.float32)
