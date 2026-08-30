"""소음 스피커(ch0) 재생 프로그램 — 스트리밍 블록 생성기.

anc_project/main_realtime_anc.py 의 NoiseGenerator 를 확장:
tone / multitone / white / band / nonlinear(고조파+소프트클립) / sweep / file / silence.
데모 시나리오(configs/eval.yaml)와 실측 수집(record_duct.py), 런타임이 공용한다.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from scipy import signal


class DigitalReferenceBuffer:
    """자기생성 소스를 ``lead_samples`` 만큼 늦춰 미래 레퍼런스를 제공한다.

    호출자가 넘기는 ``future_block`` 은 지금 생성한(게이트까지 적용된) 신호다.
    레퍼런스에는 이 블록을 즉시 공급하고, 실제 소음 스피커 출력은 내부 FIFO를
    거쳐 ``lead_samples`` 뒤에 공급한다. 따라서 정보를 추정하거나 오라클 미래를
    읽지 않으면서도 ``ref[t] == playback[t + lead]`` 가 성립한다.

    ``lead_samples=0`` 은 완전한 no-op이다. 마지막 축을 시간으로 간주하므로 신호와
    게이트를 여러 채널로 묶어 함께 지연할 수도 있다.
    """

    def __init__(self, lead_samples: int = 0) -> None:
        self.lead_samples = int(lead_samples)
        if self.lead_samples < 0:
            raise ValueError("digital reference lead_samples는 0 이상이어야 합니다")
        self._pending: np.ndarray | None = None

    def process(self, future_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(현재 재생 블록, lead만큼 앞선 레퍼런스 블록)`` 을 반환한다."""
        block = np.asarray(future_block)
        if block.ndim < 1 or block.shape[-1] <= 0:
            raise ValueError("future_block의 마지막 축에는 1개 이상의 샘플이 필요합니다")

        if self.lead_samples == 0:
            # 반환 배열을 분리해 호출자의 후속 in-place 처리가 서로 전파되지 않게 한다.
            return block.copy(), block.copy()

        prefix_shape = block.shape[:-1]
        if self._pending is None:
            self._pending = np.zeros(
                (*prefix_shape, self.lead_samples), dtype=block.dtype
            )
        elif self._pending.shape[:-1] != prefix_shape:
            raise ValueError(
                "DigitalReferenceBuffer 호출 사이에 채널/배치 shape를 바꿀 수 없습니다"
            )

        frames = block.shape[-1]
        timeline = np.concatenate([self._pending, block], axis=-1)
        playback = timeline[..., :frames].copy()
        self._pending = timeline[..., frames:].copy()
        return playback, block.copy()


class NoiseProgram:
    def __init__(
        self, cfg: dict, sample_rate: int, *, file_bytes: bytes | None = None
    ) -> None:
        self.fs = int(sample_rate)
        self.kind = str(cfg.get("type", "tone"))
        self.amplitude = float(cfg.get("amplitude", 0.05))
        self.rng = np.random.default_rng(int(cfg.get("seed", 20260801)))
        self.phase = 0.0
        self._phases: np.ndarray | None = None
        self._sos = None
        self._zi = None
        self._band_scale = 1.0
        self._file_data: np.ndarray | None = None
        self._file_pos = 0
        self._t = 0  # sweep 용 누적 샘플

        if self.kind == "tone":
            self.frequency = float(cfg.get("frequency", 300.0))
        elif self.kind == "multitone":
            freqs = cfg.get("frequencies", [120.0, 300.0, 750.0])
            self.frequencies = np.asarray([float(f) for f in freqs])
            self._phases = self.rng.uniform(0, 2 * np.pi, size=self.frequencies.size)
        elif self.kind == "nonlinear":
            # 기본파 + 3·5차 고조파 + 소프트클립 → 비선형 시나리오 (S4)
            self.frequency = float(cfg.get("frequency", 210.0))
        elif self.kind == "band":
            band = cfg.get("band", [80.0, 1000.0])
            self._sos = signal.butter(
                4, [float(band[0]), float(band[1])], btype="bandpass", fs=self.fs, output="sos"
            )
            probe = self.rng.standard_normal(max(65536, self.fs * 2))
            filtered = signal.sosfilt(self._sos, probe)
            rms = float(np.sqrt(np.mean(filtered[4096:] ** 2)))
            self._band_scale = self.amplitude / max(rms, 1e-12)
            self._zi = np.zeros((self._sos.shape[0], 2))
        elif self.kind == "sweep":
            self.sweep_lo = float(cfg.get("band", [80.0, 1000.0])[0])
            self.sweep_hi = float(cfg.get("band", [80.0, 1000.0])[1])
            self.sweep_period = float(cfg.get("period_s", 8.0))
        elif self.kind == "file":
            path = cfg.get("file")
            if not path:
                raise ValueError("file 프로그램에는 file 경로가 필요합니다")
            import soundfile as sf

            data, sr = sf.read(
                io.BytesIO(file_bytes)
                if file_bytes is not None
                else str(Path(path).expanduser()),
                dtype="float32",
                always_2d=True,
            )
            mono = data.mean(axis=1)
            if sr != self.fs:
                from math import gcd

                g = gcd(int(sr), self.fs)
                mono = signal.resample_poly(mono, self.fs // g, int(sr) // g)
            # offset이 달라도 원본 전체 파일의 gain 기준은 바뀌지 않는다. 선택 window만
            # 다시 peak-normalize하면 조용한 후보가 예기치 않게 과증폭된다.
            peak = float(np.max(np.abs(mono)) + 1e-9)
            start_seconds = float(cfg.get("file_start_seconds", 0.0))
            if not np.isfinite(start_seconds) or start_seconds < 0.0:
                raise ValueError("file_start_seconds는 0 이상 finite여야 합니다")
            start_frame = int(round(start_seconds * self.fs))
            if start_frame >= mono.size:
                raise ValueError(
                    f"file_start_seconds={start_seconds}가 재생 파일 길이를 벗어납니다"
                )
            mono = mono[start_frame:]
            self._file_data = (mono / peak * self.amplitude).astype(np.float32)
        elif self.kind in ("white", "silence"):
            pass
        else:
            raise ValueError(f"알 수 없는 노이즈 프로그램: {self.kind}")

    def generate(self, frames: int) -> np.ndarray:
        if self.kind == "silence":
            return np.zeros(frames, dtype=np.float32)

        if self.kind == "tone":
            inc = 2 * np.pi * self.frequency / self.fs
            ph = self.phase + inc * np.arange(frames)
            self.phase = float((self.phase + inc * frames) % (2 * np.pi))
            return (self.amplitude * np.sin(ph)).astype(np.float32)

        if self.kind == "multitone":
            t = np.arange(frames)
            out = np.zeros(frames)
            for i, f in enumerate(self.frequencies):
                inc = 2 * np.pi * f / self.fs
                out += np.sin(self._phases[i] + inc * t)
                self._phases[i] = float((self._phases[i] + inc * frames) % (2 * np.pi))
            out *= self.amplitude / max(1, self.frequencies.size) * 1.5
            return out.astype(np.float32)

        if self.kind == "nonlinear":
            inc = 2 * np.pi * self.frequency / self.fs
            ph = self.phase + inc * np.arange(frames)
            self.phase = float((self.phase + inc * frames) % (2 * np.pi))
            base = np.sin(ph) + 0.35 * np.sin(3 * ph) + 0.15 * np.sin(5 * ph)
            clipped = np.tanh(2.5 * base) / 2.5    # 스피커 과구동 근사
            return (self.amplitude * clipped).astype(np.float32)

        if self.kind == "white":
            out = self.rng.standard_normal(frames) * self.amplitude
            return np.clip(out, -4 * self.amplitude, 4 * self.amplitude).astype(np.float32)

        if self.kind == "band":
            white = self.rng.standard_normal(frames)
            filtered, self._zi = signal.sosfilt(self._sos, white, zi=self._zi)
            out = np.clip(filtered * self._band_scale, -4 * self.amplitude, 4 * self.amplitude)
            return out.astype(np.float32)

        if self.kind == "sweep":
            t = (self._t + np.arange(frames)) / self.fs
            self._t += frames
            cycle = np.mod(t, self.sweep_period) / self.sweep_period
            freq = self.sweep_lo * (self.sweep_hi / self.sweep_lo) ** cycle
            # 위상 연속 근사 적분
            phase = 2 * np.pi * np.cumsum(freq) / self.fs + self.phase
            self.phase = float(phase[-1] % (2 * np.pi))
            return (self.amplitude * np.sin(phase)).astype(np.float32)

        if self.kind == "file":
            assert self._file_data is not None
            out = np.empty(frames, dtype=np.float32)
            pos = self._file_pos
            data = self._file_data
            for i in range(frames):
                out[i] = data[pos]
                pos = (pos + 1) % data.size
            self._file_pos = pos
            return out

        raise RuntimeError(self.kind)
