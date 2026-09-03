"""Canonical decoder-audited NoisePool의 fail-closed 경계 회귀 테스트."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deep_anc.data.manifest import write_manifest
from deep_anc.data.noise_pool import NoisePool


FS = 48_000


class _DeterministicRng:
    def __init__(self) -> None:
        self._indices = iter((0, 1))

    def choice(self, *_args, **_kwargs):
        return next(self._indices)

    def integers(self, *_args, **_kwargs):
        return 0


def _pool(tmp_path: Path, *, canonical_decoder_audited: bool) -> tuple[NoisePool, Path, Path]:
    corrupt = tmp_path / "corrupt.mp3"
    healthy = tmp_path / "healthy.mp3"
    manifest = tmp_path / "music.jsonl"
    write_manifest(
        [
            {
                "path": str(path),
                "duration_s": 1.0,
                "sample_rate": FS,
                "channels": 1,
                "tag": "music",
                "split": "train",
            }
            for path in (corrupt, healthy)
        ],
        manifest,
    )
    pool = NoisePool(
        [manifest],
        split="train",
        sample_rate=FS,
        seed=1,
        canonical_decoder_audited=canonical_decoder_audited,
    )
    pool.rng = _DeterministicRng()
    return pool, corrupt, healthy


@pytest.mark.parametrize(
    "fault",
    (
        "read_error",
        "nonfinite",
        "peak",
        "header_rate_drift",
        "read_rate_drift",
        "silence",
    ),
)
def test_canonical_audited_pool_does_not_hide_decode_contract_failure(
    tmp_path, monkeypatch, fault
):
    """Audit 뒤 한 파일의 decode 이상은 healthy entry로 치환하면 안 된다."""

    pool, corrupt, healthy = _pool(tmp_path, canonical_decoder_audited=True)
    before = pool._active_weights.copy()
    accessed: list[tuple[str, Path]] = []

    def fake_info(path):
        candidate = Path(path)
        accessed.append(("info", candidate))
        if candidate == corrupt and fault == "read_error":
            raise RuntimeError("damaged MP3 frame")
        return SimpleNamespace(
            frames=128,
            samplerate=(44_100 if candidate == corrupt and fault == "header_rate_drift" else FS),
            channels=1,
        )

    def fake_read(path, **_kwargs):
        candidate = Path(path)
        accessed.append(("read", candidate))
        if candidate == corrupt:
            if fault == "nonfinite":
                return np.full((128, 1), np.nan, dtype=np.float32), FS
            if fault == "peak":
                return np.full((128, 1), 4.0, dtype=np.float32), FS
            if fault == "read_rate_drift":
                return np.ones((128, 1), dtype=np.float32), 44_100
            if fault == "silence":
                return np.full((128, 1), 1e-10, dtype=np.float32), FS
        return np.ones((128, 1), dtype=np.float32), FS

    monkeypatch.setattr("deep_anc.data.noise_pool.sf.info", fake_info)
    monkeypatch.setattr("deep_anc.data.noise_pool.sf.read", fake_read)

    with pytest.raises(
        RuntimeError, match="canonical decoder-audited NoisePool decode 계약 위반.*재시도하지 않고"
    ):
        pool.sample_segment(64)

    assert all(path != healthy for _operation, path in accessed)
    assert np.array_equal(pool._active_weights, before)


def test_legacy_pool_retries_header_rate_drift(tmp_path, monkeypatch):
    """기존 diagnostic/legacy manifest의 재시도 동작은 유지한다."""

    pool, corrupt, healthy = _pool(tmp_path, canonical_decoder_audited=False)

    def fake_info(path):
        candidate = Path(path)
        return SimpleNamespace(
            frames=128,
            samplerate=(44_100 if candidate == corrupt else FS),
            channels=1,
        )

    def fake_read(path, **_kwargs):
        assert Path(path) == healthy
        return np.ones((128, 1), dtype=np.float32), FS

    monkeypatch.setattr("deep_anc.data.noise_pool.sf.info", fake_info)
    monkeypatch.setattr("deep_anc.data.noise_pool.sf.read", fake_read)

    segment = pool.sample_segment(64)

    assert np.all(segment == 1.0)
    assert pool._active_weights[0] == 0.0


def test_canonical_audited_pool_retries_transient_silence_within_same_file(
    tmp_path, monkeypatch
):
    """audit의 segment_grid는 파일 전체를 성기게 표본화하므로, grid가 놓친
    짧은 자연스러운 무음 구간에 학습 시점 랜덤 위치가 걸릴 수 있다(실측:
    dns_fullband/c5RJ2TPfSEM.wav). 이건 decoder drift가 아니므로 같은 파일
    안에서 몇 번 더 시도해 살아나야 하고, healthy entry로 바꿔치기하면 안
    되며 active_weights도 건드리면 안 된다."""

    pool, corrupt, healthy = _pool(tmp_path, canonical_decoder_audited=True)
    before = pool._active_weights.copy()
    accessed: list[Path] = []
    read_calls = {"n": 0}

    def fake_info(path):
        return SimpleNamespace(frames=128, samplerate=FS, channels=1)

    def fake_read(path, **_kwargs):
        candidate = Path(path)
        accessed.append(candidate)
        if candidate == corrupt:
            read_calls["n"] += 1
            if read_calls["n"] <= 2:
                return np.full((128, 1), 1e-10, dtype=np.float32), FS
            return np.full((128, 1), 0.5, dtype=np.float32), FS
        return np.ones((128, 1), dtype=np.float32), FS

    monkeypatch.setattr("deep_anc.data.noise_pool.sf.info", fake_info)
    monkeypatch.setattr("deep_anc.data.noise_pool.sf.read", fake_read)

    segment = pool.sample_segment(64)

    assert np.all(segment == 0.5)
    assert read_calls["n"] == 3
    assert all(path != healthy for path in accessed)
    assert np.array_equal(pool._active_weights, before)


def test_legacy_pool_switches_file_immediately_on_silence(tmp_path, monkeypatch):
    """legacy/diagnostic pool은 무음도 기존처럼 같은 위치를 다시 시도하지
    않고 바로 다른 entry로 넘어간다 — canonical 전용 재시도가 새어들면 안
    된다."""

    pool, corrupt, healthy = _pool(tmp_path, canonical_decoder_audited=False)

    def fake_info(path):
        return SimpleNamespace(frames=128, samplerate=FS, channels=1)

    def fake_read(path, **_kwargs):
        if Path(path) == corrupt:
            return np.full((128, 1), 1e-10, dtype=np.float32), FS
        return np.ones((128, 1), dtype=np.float32), FS

    monkeypatch.setattr("deep_anc.data.noise_pool.sf.info", fake_info)
    monkeypatch.setattr("deep_anc.data.noise_pool.sf.read", fake_read)

    segment = pool.sample_segment(64)

    assert np.all(segment == 1.0)
    assert pool._active_weights[0] == 0.0
