"""데이터 파이프라인 검증 — shape/NaN/분할 누수/지연 물리."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from deep_anc.config import REPO_ROOT, default_d_noise_delay
from deep_anc.data.manifest import assign_splits, scan_wavs, write_manifest
from deep_anc.data.noise_pool import NoisePool
from deep_anc.data.recorded_dataset import RecordedANCDataset
from deep_anc.data.synth_dataset import SynthANCDataset
from deep_anc.data.synthetic_signals import KINDS, SyntheticNoise
from deep_anc.dsp.duct_sim import build_rir_bank


@pytest.fixture(scope="module")
def cfgs():
    with open(REPO_ROOT / "configs" / "data_sim.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    with open(REPO_ROOT / "configs" / "duct.yaml", encoding="utf-8") as f:
        duct = yaml.safe_load(f)
    data = dict(data)
    data["segment_seconds"] = 0.5                 # 테스트 고속화
    data["source_mix_ratio"] = {"synthetic": 1.0}
    return data, duct


@pytest.fixture(scope="module")
def rir_bank(cfgs):
    _, duct = cfgs
    return build_rir_bank(duct, 48000, n_variants=12, ir_len=4096)


def test_synthetic_kinds():
    synth = SyntheticNoise(48000, seed=0)
    for kind in KINDS:
        x = synth.generate(4800, kind)
        assert x.shape == (4800,)
        assert np.all(np.isfinite(x))
        assert 0.5 < np.sqrt(np.mean(x**2)) < 2.0  # RMS 정규화


@pytest.mark.parametrize("mode", ["digital", "acoustic"])
def test_dataset_items(cfgs, rir_bank, mode):
    data, duct = cfgs
    data = dict(data)
    data["reference_mode"] = mode
    if mode == "acoustic":
        data["digital_reference_lead_samples"] = 0
        # acoustic 혼합비는 demand/dns_fullband/machine 을 선언하는데 이 저장소에는
        # 그 원본이 없다. 이 테스트가 보는 것은 데이터셋 기계장치(모양·지연·시드)이지
        # 소스 분포가 아니므로 진단 탈출구를 **명시적으로** 켠다.
        # 학습 설정에 이 키를 넣으면 절대목표 2의 소스 게이트가 사라진다.
        data["allow_missing_source_manifests"] = True
    ds = SynthANCDataset(data, duct, split="train", seed=1, rir_bank=rir_bank)
    assert ds.segment % 256 == 0                  # 런타임 블록 배수 요건
    it = iter(ds)
    for _ in range(3):
        item = next(it)
        assert item["x"].shape == (2, ds.segment)
        assert item["d"].shape == (1, ds.segment)
        assert torch.isfinite(item["x"]).all()
        assert torch.isfinite(item["d"]).all()


def test_rir_split_no_leak(cfgs, rir_bank):
    data, duct = cfgs
    splits = {}
    for split in ("train", "val", "test"):
        ds = SynthANCDataset(data, duct, split=split, seed=1, rir_bank=rir_bank)
        splits[split] = set(ds.rir_indices.tolist())
    assert not (splits["train"] & splits["val"])
    assert not (splits["train"] & splits["test"])
    assert not (splits["val"] & splits["test"])


def test_manifest_split_assignment():
    entries = [{"path": f"f{i}.wav", "duration_s": 1.0} for i in range(100)]
    out = assign_splits(entries, {"train": 0.9, "val": 0.05}, seed=1)
    counts = {"train": 0, "val": 0, "test": 0}
    for e in out:
        counts[e["split"]] += 1
    assert counts["train"] == 90 and counts["val"] == 5 and counts["test"] == 5


def test_noise_pool_consumes_validated_entry_snapshot_without_manifest_reread(tmp_path):
    manifest = tmp_path / "music.jsonl"
    validated = [
        {
            "path": str(tmp_path / "validated.wav"),
            "duration_s": 1.0,
            "sample_rate": 48_000,
            "channels": 1,
            "tag": "music",
            "split": "train",
            "content_sha256": "a" * 64,
        }
    ]
    write_manifest(
        [{**validated[0], "path": str(tmp_path / "forged.wav")}], manifest
    )

    pool = NoisePool(
        [manifest],
        split="train",
        sample_rate=48_000,
        validated_entries=validated,
    )

    assert [entry["path"] for entry in pool.entries] == [str(tmp_path / "validated.wav")]


def test_scan_wavs_supports_mp3_and_skips_invalid_files(tmp_path, monkeypatch):
    nested = tmp_path / "nested"
    nested.mkdir()
    for name in ("a.wav", "b.FLAC", "c.Mp3", "broken.mp3", "ignored.ogg"):
        (nested / name).touch()

    inspected = []

    def fake_info(path):
        name = Path(path).name
        inspected.append(name)
        if name == "broken.mp3":
            raise RuntimeError("decode failed")
        return SimpleNamespace(frames=96_000, samplerate=48_000, channels=2)

    monkeypatch.setattr("deep_anc.data.manifest.sf.info", fake_info)

    entries = scan_wavs(tmp_path, tag="noise")

    assert [Path(entry["path"]).name for entry in entries] == ["a.wav", "b.FLAC", "c.Mp3"]
    assert set(inspected) == {"a.wav", "b.FLAC", "c.Mp3", "broken.mp3"}
    assert all(entry["duration_s"] == 2.0 for entry in entries)
    assert all(entry["sample_rate"] == 48_000 for entry in entries)
    assert all(entry["channels"] == 2 and entry["tag"] == "noise" for entry in entries)


def test_noise_pool_excludes_decode_failure_and_retries(tmp_path, monkeypatch):
    manifest = tmp_path / "music.jsonl"
    broken = tmp_path / "broken.mp3"
    healthy = tmp_path / "healthy.mp3"
    write_manifest(
        [
            {
                "path": str(broken),
                "duration_s": 1.0,
                "sample_rate": 48_000,
                "channels": 1,
                "tag": "music",
                "split": "train",
            },
            {
                "path": str(healthy),
                "duration_s": 1.0,
                "sample_rate": 48_000,
                "channels": 1,
                "tag": "music",
                "split": "train",
            },
        ],
        manifest,
    )

    class DeterministicRng:
        def __init__(self):
            self.indices = iter((0, 1))

        def choice(self, *_args, **_kwargs):
            return next(self.indices)

        def integers(self, *_args, **_kwargs):
            return 0

    def fake_info(path):
        if Path(path) == broken:
            raise RuntimeError("damaged MP3 frame")
        return SimpleNamespace(frames=128, samplerate=48_000, channels=1)

    def fake_read(path, **_kwargs):
        assert Path(path) == healthy
        return np.ones((128, 1), dtype=np.float32), 48_000

    monkeypatch.setattr("deep_anc.data.noise_pool.sf.info", fake_info)
    monkeypatch.setattr("deep_anc.data.noise_pool.sf.read", fake_read)

    pool = NoisePool([manifest], split="train", sample_rate=48_000, seed=1)
    pool.rng = DeterministicRng()
    segment = pool.sample_segment(64)

    assert segment.shape == (64,)
    assert np.all(segment == 1.0)
    assert pool._active_weights[0] == 0.0


def test_noise_pool_rejects_corrupt_out_of_range_pcm_and_retries(tmp_path, monkeypatch):
    """decoder 경고 없이 비정상 peak를 돌려주는 MP3도 학습을 오염시키지 않는다."""

    manifest = tmp_path / "music.jsonl"
    corrupt = tmp_path / "corrupt.mp3"
    healthy = tmp_path / "healthy.mp3"
    write_manifest(
        [
            {
                "path": str(path),
                "duration_s": 1.0,
                "sample_rate": 48_000,
                "channels": 1,
                "tag": "music",
                "split": "train",
            }
            for path in (corrupt, healthy)
        ],
        manifest,
    )

    class DeterministicRng:
        def __init__(self):
            self.indices = iter((0, 1))

        def choice(self, *_args, **_kwargs):
            return next(self.indices)

        def integers(self, *_args, **_kwargs):
            return 0

    monkeypatch.setattr(
        "deep_anc.data.noise_pool.sf.info",
        lambda _path: SimpleNamespace(frames=128, samplerate=48_000, channels=1),
    )

    def fake_read(path, **_kwargs):
        if Path(path) == corrupt:
            return np.full((128, 1), 20.0, dtype=np.float32), 48_000
        return np.ones((128, 1), dtype=np.float32), 48_000

    monkeypatch.setattr("deep_anc.data.noise_pool.sf.read", fake_read)
    pool = NoisePool([manifest], split="train", sample_rate=48_000, seed=1)
    pool.rng = DeterministicRng()
    segment = pool.sample_segment(64)

    assert np.all(segment == 1.0)
    assert pool._active_weights[0] == 0.0


def test_indexed_noise_pool_decode_retry_does_not_leak_state_between_items(
    tmp_path, monkeypatch
):
    """global-index RNG 경로의 표본은 이전 item decode 이력과 무관해야 한다."""

    manifest = tmp_path / "music.jsonl"
    broken = tmp_path / "broken.mp3"
    healthy = tmp_path / "healthy.mp3"
    write_manifest(
        [
            {
                "path": str(path),
                "duration_s": 1.0,
                "sample_rate": 48_000,
                "channels": 1,
                "tag": "music",
                "split": "train",
            }
            for path in (broken, healthy)
        ],
        manifest,
    )

    class IndexedDraw:
        def __init__(self):
            self.indices = iter((0, 1))

        def choice(self, *_args, **_kwargs):
            return next(self.indices)

        def integers(self, *_args, **_kwargs):
            return 0

    monkeypatch.setattr(
        "deep_anc.data.noise_pool.sf.info",
        lambda path: (
            (_ for _ in ()).throw(RuntimeError("damaged"))
            if Path(path) == broken
            else SimpleNamespace(frames=128, samplerate=48_000, channels=1)
        ),
    )
    monkeypatch.setattr(
        "deep_anc.data.noise_pool.sf.read",
        lambda *_args, **_kwargs: (np.ones((128, 1), dtype=np.float32), 48_000),
    )

    pool = NoisePool([manifest], split="train", sample_rate=48_000, seed=1)
    before = pool._active_weights.copy()
    assert np.all(pool.sample_segment(64, rng=IndexedDraw()) == 1.0)
    assert np.array_equal(pool._active_weights, before)


def test_d_noise_default_geometry(cfgs):
    """digital-ref 기본 지연 = s_delay − t(CS→ERR) + t(NS→ERR) [C2]."""
    _, duct = cfgs
    fs = 48000
    d = default_d_noise_delay(duct, fs, s_path_delay=1342)
    # CS(1.050)→ERR(1.100)=7샘플, NS(0)→ERR(1.100)=154샘플 → 1342-7+154=1489
    assert d == 1342 - 7 + 154


def test_d_noise_no_double_count(cfgs, rir_bank):
    """리뷰 결함 #1 회귀: RIR 에 t(NS→ERR) 온셋이 포함되므로 dataset 추가 지연은
    총지연 − t(NS→ERR) 이어야 하고, 결과 d 의 온셋은 총지연과 일치해야 한다.

    총지연 값을 여기에 박아두지 않는다. 실측이 끝나면 duct.yaml 이 값을 갖고, 그때
    상수를 박아둔 테스트는 검사하려는 불변식(이중 계상 없음)과 무관하게 깨진다.
    """
    from deep_anc.config import duct_distance_samples
    from deep_anc.dsp.secondary_path import load_secondary_path
    from deep_anc.data.synth_dataset import _delay_np
    from deep_anc.dsp.filters import fft_filter

    data, duct = cfgs
    data = dict(data)
    data["digital_primary_path_mode"] = "rir_surrogate"
    fs = 48000
    ds = SynthANCDataset(data, duct, split="train", seed=1, rir_bank=rir_bank)
    configured = duct["digital_reference"].get("d_noise_delay_samples")
    if configured is not None:
        total = int(configured)
    else:
        secondary = load_secondary_path(REPO_ROOT / duct["secondary_path"]["npz"])
        total = default_d_noise_delay(duct, fs, s_path_delay=secondary.delay_samples)
    t_ns_err = duct_distance_samples(duct, "noise_speaker", "error_mic", fs)
    assert ds.d_noise_total == total
    assert ds.d_noise_delay == total - t_ns_err

    imp = np.zeros(ds.segment, dtype=np.float32)
    imp[0] = 1.0
    d = _delay_np(fft_filter(imp, rir_bank["p_err"][0]), ds.d_noise_delay)
    onset = int(np.flatnonzero(np.abs(d) > np.max(np.abs(d)) * 0.05)[0])
    # RIR 위치 지터(±1cm)·저역통과 전이 여유 포함
    assert abs(onset - total) <= 24, f"d 온셋 {onset} vs 총지연 {total}"


def test_synth_digital_reference_lead_uses_continuous_future(cfgs, rir_bank):
    """K lead는 tail zero-padding이 아니라 같은 연속 source의 t+K여야 한다."""
    data, duct = cfgs
    data = dict(data)
    # 현행 strict P/S artifact에서 TrainingTimingContract가 유도한 lead를
    # fixture에 주입한다. 물리 지연을 116처럼 오래된 숫자로 고정하지 않는다.
    derived_data = dict(data)
    derived_data.pop("digital_reference_lead_samples", None)
    derived_data.update(
        {
            "reference_mode": "digital",
            "level_dbfs": [0.0, 0.0],
            "snr_mic_noise_db": [300.0, 300.0],
            "dc_hum_prob": 0.0,
        }
    )
    derived_ds = SynthANCDataset(
        derived_data, duct, split="train", seed=1, rir_bank=rir_bank
    )
    lead = int(derived_ds.digital_reference_lead)
    data.update(
        {
            "reference_mode": "digital",
            "digital_reference_lead_samples": lead,
            "level_dbfs": [0.0, 0.0],
            "snr_mic_noise_db": [300.0, 300.0],
            "dc_hum_prob": 0.0,
        }
    )
    ds = SynthANCDataset(data, duct, split="train", seed=1, rir_bank=rir_bank)

    class DeterministicItemRng:
        def __init__(self):
            self.uniform_values = iter((0.0, 300.0))

        def uniform(self, *_args, **_kwargs):
            return next(self.uniform_values)

        def choice(self, values, **_kwargs):
            return np.asarray(values).reshape(-1)[0]

        def integers(self, low, *_args, **_kwargs):
            return int(low)

        def standard_normal(self, size):
            return np.zeros(size, dtype=np.float32)

        def random(self):
            return 0.5

    requested = []

    def deterministic_source(_rng, _synth, n_samples=None):
        requested.append(n_samples)
        return np.arange(n_samples, dtype=np.float32)

    ds._sample_source = deterministic_source
    item = ds._make_item(DeterministicItemRng(), SyntheticNoise(ds.fs, seed=0))

    assert requested == [ds.segment + lead]
    np.testing.assert_array_equal(
        item["x"][0].numpy(), np.arange(lead, lead + ds.segment, dtype=np.float32)
    )


def test_reference_lead_is_rejected_for_acoustic_mode(cfgs, rir_bank):
    data, duct = cfgs
    data = dict(data)
    data.update({"reference_mode": "acoustic", "digital_reference_lead_samples": 109})

    with pytest.raises(ValueError, match="reference_mode=digital"):
        SynthANCDataset(data, duct, split="train", seed=1, rir_bank=rir_bank)


def test_recorded_digital_reference_uses_same_lead_alignment(tmp_path):
    manifest = tmp_path / "recorded.jsonl"
    session = tmp_path / "session"
    session.mkdir()
    write_manifest(
        [{"path": str(session), "split": "train", "duration_s": 1.0}], manifest
    )
    data = {
        "sample_rate": 48_000,
        "segment_seconds": 256 / 48_000,
        "reference_mode": "digital",
        "digital_reference_lead_samples": 7,
        "closed_loop": {"feedback_delay_samples": [1, 2]},
    }
    ds = RecordedANCDataset(manifest, data, split="train", seed=1)
    source = np.arange(512, dtype=np.float32)
    ds._load_session = lambda _entry: (source.copy(), np.zeros_like(source), source.copy())

    item = next(iter(ds))

    np.testing.assert_array_equal(
        item["x"][0].numpy(), item["d"][0].numpy() + 7.0
    )


def test_recorded_digital_reference_requires_source_wav(tmp_path):
    manifest = tmp_path / "recorded.jsonl"
    session = tmp_path / "session"
    session.mkdir()
    sf.write(session / "mics.wav", np.zeros((512, 2), dtype=np.float32), 48_000)
    write_manifest(
        [{"path": str(session), "split": "train", "duration_s": 512 / 48_000}],
        manifest,
    )
    data = {
        "sample_rate": 48_000,
        "segment_seconds": 256 / 48_000,
        "reference_mode": "digital",
        "digital_reference_lead_samples": 7,
        "closed_loop": {"feedback_delay_samples": [1, 2]},
    }

    with pytest.raises(FileNotFoundError, match="source.wav"):
        next(iter(RecordedANCDataset(manifest, data, split="train", seed=1)))
