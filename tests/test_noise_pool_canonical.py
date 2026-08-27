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
    ("read_error", "nonfinite", "peak", "header_rate_drift", "read_rate_drift"),
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
