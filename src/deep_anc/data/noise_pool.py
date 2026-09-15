"""노이즈 wav 풀 — manifest 기반 랜덤 세그먼트 샘플러."""

from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import soundfile as sf
from scipy import signal

from .manifest import read_manifest


class NoisePool:
    def __init__(
        self,
        manifest_paths: list[str | Path],
        split: str,
        sample_rate: int,
        seed: int | None = None,
        strict: bool = False,
    ) -> None:
        self.strict = strict
        self.sample_rate = int(sample_rate)
        self.rng = np.random.default_rng(seed)
        self.entries: list[dict] = []
        for mp in manifest_paths:
            self.entries.extend(read_manifest(mp, split=split))
        if not self.entries:
            raise ValueError(f"'{split}' split 에 해당하는 노이즈 파일이 없습니다: {manifest_paths}")
        durations = np.array([max(0.1, float(e["duration_s"])) for e in self.entries])
        self.weights = durations / durations.sum()
        self._active_weights = self.weights.copy()

    def __len__(self) -> int:
        return len(self.entries)

    def sample_segment(self, n_samples: int) -> np.ndarray:
        """길이 가중 랜덤 파일에서 무작위 구간을 읽어 48kHz 모노로 반환.

        manifest 생성 때 헤더를 읽었더라도 MP3의 중간 프레임이 손상됐을 수 있다.
        디코딩 실패 파일은 이 worker의 풀에서 제외하고 다른 파일로 재시도해 장기
        학습 전체가 단일 손상 파일 때문에 중단되지 않게 한다.
        """
        last_error: Exception | None = None
        max_attempts = min(16, len(self.entries))
        for _ in range(max_attempts):
            active_total = float(self._active_weights.sum())
            if active_total <= 0.0:
                break
            probabilities = self._active_weights / active_total
            index = int(self.rng.choice(len(self.entries), p=probabilities))
            entry = self.entries[index]
            path = entry["path"]
            try:
                if self.strict:
                    digest = hashlib.sha256()
                    with Path(path).open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    if digest.hexdigest() != entry.get("sha256"):
                        raise ValueError(f"strict 음원 SHA-256 불일치: {path}")
                file_sr = int(entry.get("sample_rate", self.sample_rate))
                if file_sr <= 0:
                    raise ValueError(f"잘못된 sample rate: {file_sr}")
                need_src = int(np.ceil(n_samples * file_sr / self.sample_rate)) + 16

                info = sf.info(path)
                total = int(info.frames)
                if total <= need_src:
                    data, _ = sf.read(path, dtype="float32", always_2d=True)
                else:
                    start = int(self.rng.integers(0, total - need_src))
                    data, _ = sf.read(
                        path,
                        start=start,
                        stop=start + need_src,
                        dtype="float32",
                        always_2d=True,
                    )
                mono = data.mean(axis=1)
                if mono.size == 0 or not np.isfinite(mono).all():
                    raise RuntimeError("비어 있거나 유한하지 않은 오디오")

                if file_sr != self.sample_rate:
                    from math import gcd

                    g = gcd(file_sr, self.sample_rate)
                    mono = signal.resample_poly(
                        mono, self.sample_rate // g, file_sr // g
                    )

                if mono.size < n_samples:
                    reps = int(np.ceil(n_samples / mono.size))
                    mono = np.tile(mono, reps)
                return mono[:n_samples].astype(np.float32)
            except (OSError, RuntimeError, ValueError) as exc:
                if self.strict:
                    raise RuntimeError(f"strict 음원 읽기 실패: {path}") from exc
                last_error = exc
                self._active_weights[index] = 0.0

        raise RuntimeError(
            f"오디오 디코딩 재시도 실패({max_attempts}회): {last_error}"
        ) from last_error
