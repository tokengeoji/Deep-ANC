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

# soundfile/libmpg123 may emit a decoder warning yet return a numerically corrupt
# MP3 segment.  Such a segment can be normalized to an enormous value and make one
# training batch dominate the loss.  PCM decoded from the public corpus is expected
# to lie near [-1, 1]; keep a small headroom and reject silent/corrupt blocks.
MAX_DECODED_PCM_ABS = 2.0
MIN_DECODED_RMS = 1e-8

# audit_decoder_eligibility의 segment_grid는 파일 전체를 성기게(기본 8개
# 65536-frame 창) 표본화한다 — 실측(dns_fullband/c5RJ2TPfSEM.wav)으로 grid
# 지점 하나가 이미 rms=6.4e-05로 threshold 근접 통과함을 확인했다. 그 창보다
# 짧거나 다르게 걸친 학습 시점 랜덤 위치가 진짜 무음(rms<=MIN_DECODED_RMS)
# 서브구간에 걸리는 건 decoder drift가 아니라 grid가 놓친 자연스러운 조용한
# 구간일 수 있다. 같은 파일 안에서만 몇 번 더 뽑아보고, 그래도 무음이면
# (파일이 아니라 여전히 같은 파일) 기존과 같이 즉시 중단한다.
MAX_SAME_FILE_SILENCE_RETRIES = 4


class _DecodedSilenceError(RuntimeError):
    """decoder segment RMS가 무음 임계값 이하일 때만 쓰는 전용 예외.

    canonical_decoder_audited 모드에서 이 예외만 같은 파일 안 다른 위치로
    제한된 횟수만큼 재시도한다. 그 외 decode 실패(OSError/rate 불일치/PCM
    범위 초과/frame 수 불일치 등)는 여전히 첫 시도에 즉시 중단한다 — 그건
    "grid가 놓친 조용한 구간"이 아니라 진짜 decoder drift다."""


class NoisePool:
    """Manifest 기반 랜덤 세그먼트 샘플러.

    ``canonical_decoder_audited``는 schema-v4 manifest contract가 raw content와
    decoder audit을 모두 검증한 뒤에만 켜는 fail-closed 경계다. 이 모드에서는
    audit 이후의 decoder 오류·rate drift·비정상 PCM을 다른 entry로 바꿔 숨기지
    않는다. 반대로 legacy/diagnostic manifest의 기본값은 기존처럼 손상 entry를
    이번 pool에서 제외하고 재시도한다.
    """

    def __init__(
        self,
        manifest_paths: list[str | Path],
        split: str,
        sample_rate: int,
        seed: int | None = None,
        *,
        validated_entries: list[dict] | None = None,
        canonical_decoder_audited: bool = False,
    ) -> None:
        if not isinstance(canonical_decoder_audited, bool):
            raise TypeError("canonical_decoder_audited는 bool이어야 합니다")
        self.sample_rate = int(sample_rate)
        self.rng = np.random.default_rng(seed)
        self.canonical_decoder_audited = canonical_decoder_audited
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

    @staticmethod
    def _canonical_decode_failure(entry: dict, exc: Exception) -> RuntimeError:
        """Audit-bound pool의 decode drift를 provenance와 함께 fail-closed한다."""

        path = str(entry.get("path", "<unknown>"))
        return RuntimeError(
            "canonical decoder-audited NoisePool decode 계약 위반 — "
            "audit 통과 뒤에는 다른 파일로 재시도하지 않고 학습을 중단합니다: "
            f"{path}; {type(exc).__name__}: {exc}"
        )

    def sample_segment(
        self,
        n_samples: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """길이 가중 랜덤 파일에서 무작위 구간을 읽어 48kHz 모노로 반환.

        manifest 생성 때 헤더를 읽었더라도 MP3의 중간 프레임이 손상됐을 수 있다.
        legacy/diagnostic pool은 디코딩 실패 파일을 이 worker의 풀에서 제외하고 다른
        파일로 재시도한다. 반면 ``canonical_decoder_audited=True``는 이미 전수 audit을
        통과한 분포라는 계약이므로 같은 상황을 decoder drift로 보고 즉시 중단한다.
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

                def _decode_once() -> np.ndarray:
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
                            data, decoded_rate, _total = self._read_validated_audio(
                                entry, start=None, stop=None
                            )
                        else:
                            start = int(draw_rng.integers(0, total - need_src))
                            data, decoded_rate, _total = self._read_validated_audio(
                                entry, start=start, stop=start + need_src
                            )
                        if decoded_rate != file_sr:
                            raise ValueError(
                                "manifest/decoder read sample-rate 불일치: "
                                f"{decoded_rate} != {file_sr}"
                            )
                    else:
                        info = sf.info(path)
                        total = int(info.frames)
                        info_rate = int(info.samplerate)
                        if info_rate != file_sr:
                            raise ValueError(
                                "manifest/decoder header sample-rate 불일치: "
                                f"{info_rate} != {file_sr}"
                            )
                        if total <= need_src:
                            data, decoded_rate = sf.read(
                                path, dtype="float32", always_2d=True
                            )
                        else:
                            start = int(draw_rng.integers(0, total - need_src))
                            data, decoded_rate = sf.read(
                                path,
                                start=start,
                                stop=start + need_src,
                                dtype="float32",
                                always_2d=True,
                            )
                        if int(decoded_rate) != file_sr:
                            raise ValueError(
                                "manifest/decoder read sample-rate 불일치: "
                                f"{decoded_rate} != {file_sr}"
                            )
                    mono = data.mean(axis=1)
                    if mono.size == 0 or not np.isfinite(mono).all():
                        raise RuntimeError("비어 있거나 유한하지 않은 오디오")
                    decoded_peak = float(np.max(np.abs(mono)))
                    if decoded_peak > MAX_DECODED_PCM_ABS:
                        raise RuntimeError(
                            "decoder가 PCM 범위를 벗어난 값을 반환했습니다: "
                            f"peak={decoded_peak:.6g} > {MAX_DECODED_PCM_ABS}"
                        )
                    decoded_rms = float(np.sqrt(np.mean(mono * mono)))
                    if decoded_rms <= MIN_DECODED_RMS:
                        raise _DecodedSilenceError(
                            "decoder segment가 무음/퇴화했습니다: "
                            f"rms={decoded_rms:.6g} <= {MIN_DECODED_RMS}"
                        )

                    if file_sr != self.sample_rate:
                        from math import gcd

                        g = gcd(file_sr, self.sample_rate)
                        mono = signal.resample_poly(
                            mono, self.sample_rate // g, file_sr // g
                        )

                    if not np.isfinite(mono).all():
                        raise RuntimeError("resample 뒤 오디오가 유한하지 않습니다")

                    if mono.size < n_samples:
                        reps = int(np.ceil(n_samples / mono.size))
                        mono = np.tile(mono, reps)
                    return mono[:n_samples].astype(np.float32)

                silence_retries = (
                    MAX_SAME_FILE_SILENCE_RETRIES
                    if self.canonical_decoder_audited
                    else 0
                )
                for silence_attempt in range(silence_retries + 1):
                    try:
                        return _decode_once()
                    except _DecodedSilenceError:
                        if silence_attempt == silence_retries:
                            raise
                        continue
                raise AssertionError("unreachable")
            except (OSError, RuntimeError, ValueError) as exc:
                if self.canonical_decoder_audited:
                    raise self._canonical_decode_failure(entry, exc) from exc
                last_error = exc
                active_weights[index] = 0.0

        raise RuntimeError(
            f"오디오 디코딩 재시도 실패({max_attempts}회): {last_error}"
        ) from last_error
