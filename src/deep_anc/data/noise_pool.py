"""노이즈 wav 풀 — manifest 기반 랜덤 세그먼트 샘플러."""

from __future__ import annotations

import os
from pathlib import Path
import stat

import numpy as np
import soundfile as sf
from scipy import signal

from .holdout_contract import reject_symlink_components
from .manifest import read_manifest


class NoisePool:
    def __init__(
        self,
        manifest_paths: list[str | Path],
        split: str,
        sample_rate: int,
        seed: int | None = None,
        *,
        validated_entries: list[dict] | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.rng = np.random.default_rng(seed)
        self.entries: list[dict] = []
        if validated_entries is not None:
            self.entries.extend(
                dict(entry)
                for entry in validated_entries
                if entry.get("split") == split
            )
        else:
            for mp in manifest_paths:
                self.entries.extend(read_manifest(mp, split=split))
        if not self.entries:
            raise ValueError(f"'{split}' split 에 해당하는 노이즈 파일이 없습니다: {manifest_paths}")
        durations = np.array([max(0.1, float(e["duration_s"])) for e in self.entries])
        self.weights = durations / durations.sum()
        self._active_weights = self.weights.copy()

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _read_validated_audio(
        entry: dict,
        *,
        start: int | None,
        stop: int | None,
    ) -> tuple[np.ndarray, int, int]:
        """validator가 고정한 inode/stat과 같은 O_NOFOLLOW fd 하나에서 decode한다."""

        snapshot = entry.get("_validated_file_snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError("validated raw snapshot metadata가 없습니다")
        path = Path(str(entry["path"]))
        raw_root = entry.get("_validated_raw_root")
        if not isinstance(raw_root, str) or not raw_root:
            raise ValueError("validated raw root metadata가 없습니다")
        reject_symlink_components(path, root=raw_root)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            current = os.fstat(descriptor)
            expected = (
                int(snapshot["device"]),
                int(snapshot["inode"]),
                int(snapshot["size"]),
                int(snapshot["mtime_ns"]),
                int(snapshot["ctime_ns"]),
            )
            actual = (
                int(current.st_dev),
                int(current.st_ino),
                int(current.st_size),
                int(current.st_mtime_ns),
                int(current.st_ctime_ns),
            )
            if not stat.S_ISREG(current.st_mode) or actual != expected:
                raise ValueError(f"검증 후 raw audio가 변경/retarget됐습니다: {path}")
            with sf.SoundFile(descriptor, mode="r", closefd=False) as handle:
                total = int(handle.frames)
                rate = int(handle.samplerate)
                if start is not None:
                    handle.seek(start)
                frames = -1 if stop is None or start is None else max(0, stop - start)
                data = handle.read(frames=frames, dtype="float32", always_2d=True)
            return data, rate, total
        finally:
            os.close(descriptor)

    def sample_segment(
        self,
        n_samples: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """길이 가중 랜덤 파일에서 무작위 구간을 읽어 48kHz 모노로 반환.

        manifest 생성 때 헤더를 읽었더라도 MP3의 중간 프레임이 손상됐을 수 있다.
        디코딩 실패 파일은 이 worker의 풀에서 제외하고 다른 파일로 재시도해 장기
        학습 전체가 단일 손상 파일 때문에 중단되지 않게 한다.
        """
        draw_rng = self.rng if rng is None else rng
        # global-index 학습 경로는 한 item이 과거 item의 decode 성공/실패 상태에
        # 의존하면 안 된다. 외부 indexed RNG를 받았을 때는 실패 마스크도 호출
        # 로컬로 만들어 K+resume이 uninterrupted와 같은 표본을 재생하게 한다.
        active_weights = (
            self._active_weights if rng is None else self.weights.copy()
        )
        last_error: Exception | None = None
        max_attempts = min(16, len(self.entries))
        for _ in range(max_attempts):
            active_total = float(active_weights.sum())
            if active_total <= 0.0:
                break
            probabilities = active_weights / active_total
            index = int(draw_rng.choice(len(self.entries), p=probabilities))
            entry = self.entries[index]
            path = entry["path"]
            try:
                file_sr = int(entry.get("sample_rate", self.sample_rate))
                if file_sr <= 0:
                    raise ValueError(f"잘못된 sample rate: {file_sr}")
                need_src = int(np.ceil(n_samples * file_sr / self.sample_rate)) + 16

                validated_snapshot = entry.get("_validated_file_snapshot")
                if isinstance(validated_snapshot, dict):
                    # 먼저 header/stat를 읽고, 선택된 구간도 같은 검증 경로에서 다시
                    # O_NOFOLLOW fd로 읽는다. 두 open 사이 retarget은 stat contract가 막는다.
                    _probe, verified_rate, total = self._read_validated_audio(
                        entry, start=0, stop=0
                    )
                    if verified_rate != file_sr:
                        raise ValueError(
                            f"manifest/sample-rate header 불일치: {verified_rate} != {file_sr}"
                        )
                    if total <= need_src:
                        data, _rate, _total = self._read_validated_audio(
                            entry, start=None, stop=None
                        )
                    else:
                        start = int(draw_rng.integers(0, total - need_src))
                        data, _rate, _total = self._read_validated_audio(
                            entry, start=start, stop=start + need_src
                        )
                else:
                    info = sf.info(path)
                    total = int(info.frames)
                    if total <= need_src:
                        data, _ = sf.read(path, dtype="float32", always_2d=True)
                    else:
                        start = int(draw_rng.integers(0, total - need_src))
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
                last_error = exc
                active_weights[index] = 0.0

        raise RuntimeError(
            f"오디오 디코딩 재시도 실패({max_attempts}회): {last_error}"
        ) from last_error
