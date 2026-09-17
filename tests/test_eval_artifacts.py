"""측정 산출물 표준(`deep_anc.eval.artifacts`)의 계약을 고정한다.

이 모듈이 지키는 약속은 두 가지이고, 둘 다 실기 사고에서 나왔다.

1. **부분 기록된 파일이 관측되지 않는다.** 측정은 스피커를 다시 울려야 얻으므로,
   쓰다 만 CSV 를 다음 실행이 읽고 "측정했다"고 판단하면 안 된다.
2. **OFF/ON WAV 는 같은 배율로 쓴다.** 파일마다 정규화하면 크기 차이가 사라져
   상쇄 효과가 귀에서 없어진다 — 들어서 확인하는 목적 자체가 무너진다.
"""

from __future__ import annotations

import csv
import copy
import importlib.util
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deep_anc.eval.artifacts import (
    atomic_write_text,
    markdown_table,
    run_directory,
    write_csv,
    write_json,
    write_series_csv,
    write_wav,
    write_wav_pair,
)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_run_directory_prepares_raw_and_wav(tmp_path):
    run_dir = run_directory(tmp_path, "session", "20260804_120000")
    assert run_dir.name == "session_20260804_120000"
    assert (run_dir / "raw").is_dir()
    assert (run_dir / "wav").is_dir()


def test_run_directory_is_idempotent(tmp_path):
    first = run_directory(tmp_path, "session", "stamp")
    (first / "raw" / "keep.npz").write_bytes(b"x")
    second = run_directory(tmp_path, "session", "stamp")
    assert first == second
    assert (second / "raw" / "keep.npz").exists(), "재호출이 기존 산출물을 지우면 안 된다"


def test_atomic_write_leaves_no_partial_file(tmp_path):
    target = tmp_path / "deep" / "summary.md"
    atomic_write_text(target, "본문")
    assert target.read_text(encoding="utf-8") == "본문"
    assert not list(tmp_path.rglob("*.tmp")), "임시 파일이 남으면 안 된다"


def test_write_csv_keeps_first_row_column_order(tmp_path):
    path = write_csv(tmp_path / "m.csv", [
        {"scenario": "tone300", "att_db": 6.26},
        {"scenario": "band", "att_db": 5.14, "extra": 1},
    ])
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().strip()
    assert header == "scenario,att_db,extra"


def test_write_csv_rejects_empty_rows(tmp_path):
    # 빈 CSV 를 쓰면 "측정했는데 결과가 없다"와 "측정에 실패했다"를 구분할 수 없게 된다.
    with pytest.raises(ValueError):
        write_csv(tmp_path / "m.csv", [])


def test_write_csv_normalises_numpy_and_bool(tmp_path):
    path = write_csv(tmp_path / "m.csv", [
        {"a": np.float64(1.5), "b": np.int64(3), "c": np.bool_(True), "d": [1, 2]},
    ])
    row = read_csv(path)[0]
    assert row["a"] == "1.5"
    assert row["b"] == "3"
    assert row["c"] == "1"
    assert json.loads(row["d"]) == [1, 2]


def test_write_series_csv_is_long_format(tmp_path):
    path = write_series_csv(tmp_path / "s.csv", {
        "onset": [690, 674, 687],
        "valid": [True, True, False],
    }, index_name="repeat")
    rows = read_csv(path)
    assert [r["repeat"] for r in rows] == ["0", "1", "2"]
    assert [r["onset"] for r in rows] == ["690", "674", "687"]


def test_write_series_csv_rejects_ragged_series(tmp_path):
    with pytest.raises(ValueError):
        write_series_csv(tmp_path / "s.csv", {"a": [1, 2, 3], "b": [1, 2]})


def test_write_json_handles_numpy(tmp_path):
    path = write_json(tmp_path / "r.json", {"x": np.float32(2.5), "y": np.arange(3)})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["x"] == pytest.approx(2.5)
    assert payload["y"] == [0, 1, 2]


def test_write_wav_is_16bit_mono_at_given_rate(tmp_path):
    path = write_wav(tmp_path / "a.wav", np.zeros(480, dtype=np.float32), 48000)
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 48000
        assert handle.getnframes() == 480


def test_write_wav_clips_instead_of_wrapping(tmp_path):
    # 오버플로가 랩어라운드하면 조용한 신호가 최대 진폭 잡음으로 들린다.
    path = write_wav(tmp_path / "a.wav", np.array([2.0, -2.0], dtype=np.float32), 48000)
    with wave.open(str(path), "rb") as handle:
        pcm = np.frombuffer(handle.readframes(2), dtype="<i2")
    assert pcm.tolist() == [32767, -32767]


def test_wav_pair_uses_one_shared_scale(tmp_path):
    """ON 이 OFF 보다 조용하면 파일에서도 조용해야 한다 — 이것이 이 모듈의 핵심 계약이다."""

    rng = np.random.default_rng(0)
    off = rng.normal(0.0, 0.10, 48000)
    on = off * 0.25                      # 정확히 12dB 감쇠
    paths = write_wav_pair(tmp_path, "band", off, on, 48000)

    def rms(path):
        with wave.open(str(path), "rb") as handle:
            pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        return float(np.sqrt(np.mean((pcm / 32767.0) ** 2)))

    ratio_db = 20.0 * np.log10(rms(paths["off"]) / rms(paths["on"]))
    assert ratio_db == pytest.approx(12.0, abs=0.3)


def test_wav_pair_ab_file_is_off_then_on(tmp_path):
    off = np.full(2000, 0.5)
    on = np.full(2000, 0.05)
    paths = write_wav_pair(tmp_path, "x", off, on, 48000)
    with wave.open(str(paths["ab"]), "rb") as handle:
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    half = pcm.size // 2
    assert np.abs(pcm[:half]).mean() > 5 * np.abs(pcm[half:]).mean()


def test_wav_pair_survives_silent_off_segment(tmp_path):
    # 무음 OFF 에서 0 나눗셈으로 죽으면, 측정은 끝났는데 산출물을 잃는다.
    paths = write_wav_pair(tmp_path, "silent", np.zeros(480), np.zeros(480), 48000)
    assert all(path.exists() for path in paths.values())


def test_markdown_table_matches_csv_columns(tmp_path):
    rows = [{"scenario": "tone300", "att_db": 6.26}]
    columns = ["scenario", "att_db"]
    write_csv(tmp_path / "m.csv", rows, columns=columns)
    lines = list(markdown_table(rows, columns))
    assert lines[0] == "| scenario | att_db |"
    assert read_csv(tmp_path / "m.csv")[0]["scenario"] == "tone300"


@pytest.fixture
def offline_eval():
    path = Path(__file__).resolve().parents[1] / "scripts/eval/evaluate_offline.py"
    spec = importlib.util.spec_from_file_location("offline_source_policy_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def acoustic_source_policy():
    return {
        "sample_rate": 8000,
        "reference_mode": "acoustic",
        "train_only_source_families": ["machine"],
        "source_mix_ratio": {"synthetic": 0.4, "esc50": 0.4, "machine": 0.2},
        "source_mix_ratio_acoustic": {"synthetic": 0.3, "speech": 0.5, "machine": 0.2},
    }


def test_offline_policy_reports_actual_test_denominator(offline_eval, acoustic_source_policy):
    cfg = acoustic_source_policy
    before = copy.deepcopy(cfg)
    mix = offline_eval.source_mix_for_split(cfg, "test")
    policy = offline_eval.source_evaluation_policy(cfg, mix)
    assert cfg == before
    assert policy["configured_training_source_mix"] == cfg["source_mix_ratio_acoustic"]
    assert policy["excluded_train_only_source_families"] == ["machine"]
    assert policy["retained_training_weight_sum"] == pytest.approx(0.8)
    assert policy["training_weight_sum"] == pytest.approx(1.0)
    assert policy["effective_evaluation_source_mix"] == pytest.approx({"synthetic": 0.375, "speech": 0.625})
    assert policy["per_source_requested"] == ["synthetic", "speech"]
    assert policy["diagnostic_only"] is True
    assert policy["performance_claim_allowed"] is False
    json.dumps(policy, allow_nan=False)


@pytest.mark.parametrize("tag", ["synthetic", "speech"])
def test_offline_per_source_also_overrides_acoustic_mix(offline_eval, acoustic_source_policy, tag):
    cfg = acoustic_source_policy
    before = copy.deepcopy(cfg)
    active = offline_eval.source_mix_for_split(cfg, "test")
    isolated = offline_eval.per_source_config(cfg, tag, active)
    assert cfg == before
    assert isolated["source_mix_ratio"] == {tag: 1.0}
    assert isolated["source_mix_ratio_acoustic"] == {tag: 1.0}
    assert isolated["train_only_source_families"] == ["machine"]
    assert offline_eval.source_mix_for_split(isolated, "test") == {tag: 1.0}


@pytest.mark.parametrize("tag", ["machine", "esc50", "music"])
def test_offline_per_source_rejects_excluded_or_inactive(offline_eval, acoustic_source_policy, tag):
    with pytest.raises(ValueError, match="허용하지 않는 소스"):
        offline_eval.per_source_config(
            acoustic_source_policy, tag,
            offline_eval.source_mix_for_split(acoustic_source_policy, "test"),
        )


def test_offline_does_not_force_zero_weight_synthetic(offline_eval):
    cfg = {"source_mix_ratio": {"synthetic": 0.0, "speech": 1.0}}
    policy = offline_eval.source_evaluation_policy(cfg, cfg["source_mix_ratio"])
    assert policy["per_source_requested"] == ["speech"]
    assert policy["excluded_train_only_source_families"] == []
    assert policy["effective_evaluation_source_mix"] == {"speech": 1.0}
    assert offline_eval.per_source_config(cfg, "speech", cfg["source_mix_ratio"])["source_mix_ratio"] == {"speech": 1.0}


@pytest.mark.parametrize("mix", [{}, {"synthetic": 0.0}, {"machine": 1.0}])
def test_offline_report_rejects_empty_or_train_only_evaluation(offline_eval, acoustic_source_policy, mix):
    with pytest.raises(ValueError):
        offline_eval.source_evaluation_policy(acoustic_source_policy, mix)


@pytest.mark.parametrize("fail_speech", [False, True])
def test_offline_main_records_exclusions_and_source_failures(
    offline_eval, acoustic_source_policy, monkeypatch, tmp_path, fail_speech,
):
    """합성 tensor/가짜 플랜트만 사용한다. 원본 준비·실기 성능 통과 검사가 아니다."""
    torch = offline_eval.torch
    cfg = acoustic_source_policy
    manifest = tmp_path / "fixture_manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    seen = []

    class Dataset:
        def __init__(self, data_cfg, duct_cfg, *, split, seed):
            assert split == "test"
            self.mix_ratio = offline_eval.source_mix_for_split(data_cfg, split)
            self.pools = {tag: [manifest] for tag in self.mix_ratio if tag != "synthetic"}
            seen.append(copy.deepcopy(data_cfg))

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    class Plant(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, y, params):
            return y

    def batch(dataset, n_items, seed):
        assert "machine" not in dataset.mix_ratio
        if fail_speech and dataset.mix_ratio == {"speech": 1.0}:
            raise ValueError("합성 fixture의 speech 실패")
        t = torch.arange(512, dtype=torch.float32) / cfg["sample_rate"]
        d = torch.sin(2 * torch.pi * 1000 * t).repeat(n_items, 1).unsqueeze(1)
        return {"x": d.clone(), "d": d}

    state = {"cfg": {"data": cfg, "duct": {
        "secondary_path": {"npz": "unused_fixture.npz", "handoff_extra_samples": 0},
        "acoustics": {"realistic_target_band_hz": [100, 2000]},
    }, "model": {}}, "model": {}}
    monkeypatch.setattr(offline_eval.torch, "load", lambda *a, **k: state)
    monkeypatch.setattr(offline_eval.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(offline_eval, "load_yaml", lambda path: {"octave_bands_hz": [1000]})
    monkeypatch.setattr(offline_eval, "build_model", lambda model_cfg: Model())
    monkeypatch.setattr(offline_eval, "load_secondary_path", lambda path: SimpleNamespace(
        sample_rate=8000, trusted_band_hz=lambda: (100, 2000),
    ))
    monkeypatch.setattr(offline_eval, "DifferentiableSecondaryPath", Plant)
    monkeypatch.setattr(offline_eval, "SynthANCDataset", Dataset)
    monkeypatch.setattr(offline_eval, "make_eval_batch", batch)
    for name in ("spectrogram_pair", "psd_overlay", "band_bar"):
        monkeypatch.setattr(offline_eval, name, lambda *a, **k: None)
    out = tmp_path / "eval"
    monkeypatch.setattr(offline_eval.sys, "argv", [
        "evaluate_offline.py", "--ckpt", "unused.pt", "--n-items", "2", "--out", str(out),
    ])
    assert offline_eval.main() == 0
    assert len(seen) == 3
    assert seen[1]["source_mix_ratio_acoustic"] == {"synthetic": 1.0}
    assert seen[2]["source_mix_ratio_acoustic"] == {"speech": 1.0}
    assert seen[1]["source_mix_ratio"] == {"synthetic": 1.0}
    assert seen[2]["source_mix_ratio"] == {"speech": 1.0}
    with np.load(out / "metrics.npz", allow_pickle=False) as metrics:
        policy = json.loads(str(metrics["source_evaluation_policy_json"]))
        assert metrics["per_item_fullband_db"].shape == (2,)
    assert policy["per_source_requested"] == ["synthetic", "speech"]
    assert policy["per_source_evaluated"] == (["synthetic"] if fail_speech else ["synthetic", "speech"])
    assert policy["per_source_skipped"] == ([{
        "source_family": "speech", "reason": "합성 fixture의 speech 실패",
    }] if fail_speech else [])
    report = (out / "metrics.md").read_text(encoding="utf-8")
    assert "독립 평가 제외(train-only): machine" in report
    assert "남은 학습 가중치 합 0.8" in report
    assert '"speech": 0.625' in report
    assert "실기 감쇠" in report
    if fail_speech:
        assert "| speech | 미평가 | 미평가 |" in report
    assert "| machine |" not in report
