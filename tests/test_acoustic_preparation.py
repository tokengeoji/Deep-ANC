"""합성 PCM fixture만 쓰는 학습 준비 계약. 실제 음향 성능 검증이 아니다."""

import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from deep_anc.config import REPO_ROOT, load_train_config
from deep_anc.data.synth_dataset import SynthANCDataset, make_eval_batch
from deep_anc.data.synthetic_signals import KINDS, SyntheticNoise
from deep_anc.dsp.secondary_path import DifferentiableSecondaryPath, load_secondary_path


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def prepared(tmp_path):
    cfg = load_train_config("configs/train_acoustic_prepared.yaml")
    raw, manifests = tmp_path / "raw", tmp_path / "manifests"
    raw.mkdir()
    manifests.mkdir()
    cfg["data"].update(sample_rate=8000, segment_seconds=0.256, noise_manifest_dir=str(manifests))
    all_rows, manifest_hashes, counts = [], {}, {}
    families = [name for name in cfg["data"]["source_mix_ratio"] if name != "synthetic"]
    for family_index, family in enumerate(families):
        rows = []
        splits = ("train",) if family == "machine" else ("train", "val", "test")
        for index, split in enumerate(splits):
            path = raw / f"{family}_{split}.wav"
            time = np.arange(4096) / 8000
            tone = 0.08 * np.sin(2 * np.pi * (300 + 170 * family_index + 20 * index) * time)
            sf.write(path, tone, 8000, subtype="FLOAT")
            rows.append({
                "path": f"../raw/{path.name}", "path_base": "manifest", "duration_s": 4096 / 8000,
                "sample_rate": 8000, "channels": 1, "frames": 4096, "sha256": _hash(path),
                "group_id": f"{family}-{split}", "source_family": family, "split": split,
                "tag": family, "qa_valid": True,
            })
            if family == "machine":
                rows[-1].update(group_id="machine:mimii-dg-unresolved", metadata={
                    "usage_policy": "train_only_auxiliary", "independent_evaluation_allowed": False,
                })
        manifest_path = manifests / f"{family}.jsonl"
        _write_rows(manifest_path, rows)
        all_rows.extend(rows)
        manifest_hashes[manifest_path.name] = _hash(manifest_path)
        counts[family] = {split: int(split in splits) for split in ("train", "val", "test")}
    _write_rows(manifests / "inventory.jsonl", all_rows)
    (raw / "sources.json").write_text("{}", encoding="utf-8")
    qa = {
        "schema_version": 1, "data_ready": True, "diagnostic_only": True,
        "performance_claim_allowed": False, "raw_root": str(raw), "families": families,
        "train_only_source_families": ["machine"],
        "manifest_sha256": manifest_hashes, "inventory_sha256": _hash(manifests / "inventory.jsonl"),
        "source_metadata_sha256": {"sources.json": _hash(raw / "sources.json")}, "split_counts": counts,
    }
    (manifests / "qa.json").write_text(json.dumps(qa), encoding="utf-8")
    rirs = {key: np.zeros((6, 12), dtype=np.float32) for key in ("p_ref", "p_err", "f_fb")}
    for index in range(6):
        for key, delay in (("p_ref", 2), ("p_err", 5), ("f_fb", 3)):
            rirs[key][index, delay] = 0.5 + index * 0.01
    bank = tmp_path / "bank.npz"
    np.savez(bank, **rirs, sample_rate=8000)
    cfg["data"]["rir_bank"] = str(bank)
    secondary = tmp_path / "secondary.npz"
    np.savez(secondary, fir=[-0.5], delay_samples=7, sample_rate=8000,
             excitation_band_hz=[80, 1600], consistency_band_hz=[150, 600], coherence_median=0.95)
    cfg["duct"]["secondary_path"].update(npz=str(secondary), handoff_extra_samples=256)
    return cfg


def _dataset(cfg, split="train", **kwargs):
    return SynthANCDataset(cfg["data"], cfg["duct"], split=split, seed=1, **kwargs)


def _rewrite_manifest(cfg, family, transform):
    directory = Path(cfg["data"]["noise_manifest_dir"])
    path = directory / f"{family}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    transform(rows)
    _write_rows(path, rows)
    qa = json.loads((directory / "qa.json").read_text())
    qa["manifest_sha256"][path.name] = _hash(path)
    qa["split_counts"][family] = {split: sum(row["split"] == split for row in rows)
                                   for split in ("train", "val", "test")}
    all_rows = []
    for name in qa["families"]:
        all_rows.extend(json.loads(line) for line in (directory / f"{name}.jsonl").read_text().splitlines())
    _write_rows(directory / "inventory.jsonl", all_rows)
    qa["inventory_sha256"] = _hash(directory / "inventory.jsonl")
    (directory / "qa.json").write_text(json.dumps(qa))


def test_strict_family_split_smoke_and_rir_isolation(prepared):
    indices = []
    for split in ("train", "val", "test"):
        ds = _dataset(prepared, split)
        indices.append(set(ds.rir_indices))
        for family in list(ds.mix_ratio):
            ds.mix_ratio = {family: 1.0}
            item = next(iter(ds))
            assert item["x"].shape == (2, 2048) and item["d"].shape == (1, 2048)
            assert torch.isfinite(item["x"]).all() and torch.isfinite(item["d"]).all()
        assert ds.prepared_data_metadata["data_ready"] is True
        assert ds.prepared_data_metadata["performance_claim_allowed"] is False
    assert all(not indices[i] & indices[j] for i in range(3) for j in range(i))


@pytest.mark.parametrize("split", ["val", "test"])
def test_train_only_machine_is_excluded_and_remaining_mix_renormalized(prepared, split):
    from deep_anc.data.synth_dataset import source_mix_for_split

    original = dict(prepared["data"]["source_mix_ratio"])
    train = _dataset(prepared)
    evaluated = _dataset(prepared, split)
    assert train.mix_ratio == original
    assert train.prepared_data_metadata == evaluated.prepared_data_metadata
    assert "machine" in train.pools and "machine" not in evaluated.pools
    assert "machine" not in evaluated.mix_ratio
    assert sum(evaluated.mix_ratio.values()) == pytest.approx(1.0)
    for family, weight in evaluated.mix_ratio.items():
        assert weight == pytest.approx(original[family] / (1 - original["machine"]))
    assert prepared["data"]["source_mix_ratio"] == original
    assert source_mix_for_split(prepared["data"], split) == evaluated.mix_ratio


@pytest.mark.parametrize("split", ["val", "test"])
def test_forced_machine_eval_is_rejected_even_if_pool_cached_or_fallback_available(prepared, split):
    ds = _dataset(prepared, split)
    ds._pool_objs["machine"] = object()  # cached pool도 policy를 우회하지 못한다.
    with pytest.raises(ValueError, match="학습 보조 전용"):
        ds._pool("machine", np.random.default_rng(2))
    ds.mix_ratio = {"machine": 1.0}
    with pytest.raises(ValueError, match="학습 보조 전용"):
        make_eval_batch(ds, 1)


def test_training_dataset_cannot_be_used_as_independent_eval(prepared):
    with pytest.raises(ValueError, match="val/test dataset"):
        make_eval_batch(_dataset(prepared), 1)


@pytest.mark.parametrize("policy", [None, [], "machine", ["machine", "machine"], ["speech"], [True]])
def test_prepared_policy_cannot_be_removed_or_reassigned(prepared, policy):
    prepared["data"]["train_only_source_families"] = policy
    with pytest.raises(ValueError, match="train_only_source_families"):
        _dataset(prepared)


@pytest.mark.parametrize("policy", [None, [], ["speech"], ["machine", "machine"]])
def test_qa_train_only_policy_must_match_config(prepared, policy):
    path = Path(prepared["data"]["noise_manifest_dir"]) / "qa.json"
    qa = json.loads(path.read_text())
    qa["train_only_source_families"] = policy
    path.write_text(json.dumps(qa))
    with pytest.raises(ValueError, match="train_only_source_families"):
        _dataset(prepared)


@pytest.mark.parametrize("fault", ["val", "test", "group", "role", "evaluation_allowed", "metadata_missing", "empty"])
def test_machine_manifest_cannot_claim_independent_eval_with_fresh_hashes(prepared, fault):
    def change(rows):
        if fault in ("val", "test"):
            rows[0]["split"] = fault
        elif fault == "group":
            rows[0]["group_id"] = "machine:section_00"
        elif fault == "role":
            rows[0]["metadata"]["usage_policy"] = "evaluation"
        elif fault == "evaluation_allowed":
            rows[0]["metadata"]["independent_evaluation_allowed"] = True
        elif fault == "metadata_missing":
            rows[0].pop("metadata")
        else:
            rows.clear()
    _rewrite_manifest(prepared, "machine", change)
    with pytest.raises(ValueError, match="학습"):
        _dataset(prepared)


def test_train_only_mix_without_evaluation_sources_fails(prepared):
    prepared["data"]["source_mix_ratio"] = {"machine": 1.0}
    with pytest.raises(ValueError, match="평가 가능한 source"):
        _dataset(prepared)


def test_split_policy_respects_acoustic_mix_override(prepared):
    from deep_anc.data.synth_dataset import source_mix_for_split

    prepared["data"]["source_mix_ratio_acoustic"] = {"machine": 0.25, "speech": 0.75}
    assert source_mix_for_split(prepared["data"], "train") == {"machine": 0.25, "speech": 0.75}
    assert source_mix_for_split(prepared["data"], "val") == {"speech": 1.0}
    assert source_mix_for_split(prepared["data"], "test") == {"speech": 1.0}


def test_legacy_without_policy_retains_machine_eval_mix(prepared):
    from deep_anc.data.synth_dataset import source_mix_for_split

    prepared["data"]["require_prepared_data"] = False
    prepared["data"].pop("train_only_source_families")
    assert source_mix_for_split(prepared["data"], "test") == prepared["data"]["source_mix_ratio"]


@pytest.mark.parametrize("relative", ["qa.json", "speech.jsonl", "inventory.jsonl"])
def test_missing_preparation_artifact_never_falls_back(prepared, relative):
    (Path(prepared["data"]["noise_manifest_dir"]) / relative).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        _dataset(prepared)


@pytest.mark.parametrize("target", ["manifest", "inventory", "metadata", "raw"])
def test_changed_preparation_content_is_rejected(prepared, target):
    directory = Path(prepared["data"]["noise_manifest_dir"])
    path = {"manifest": directory / "speech.jsonl", "inventory": directory / "inventory.jsonl",
            "metadata": directory.parent / "raw/sources.json", "raw": directory.parent / "raw/speech_train.wav"}[target]
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        _dataset(prepared)


@pytest.mark.parametrize("target", ["qa", "inventory", "metadata", "manifest", "raw"])
def test_change_during_prepared_validation_is_rejected(prepared, monkeypatch, target):
    from deep_anc.data import synth_dataset

    directory = Path(prepared["data"]["noise_manifest_dir"])
    paths = {
        "qa": directory / "qa.json", "inventory": directory / "inventory.jsonl",
        "metadata": directory.parent / "raw/sources.json", "manifest": directory / "speech.jsonl",
        "raw": directory.parent / "raw/speech_train.wav",
    }
    changed_path = paths[target]
    original_load = synth_dataset.np.load
    mutated = False
    def load_after_mutation(path, *args, **kwargs):
        nonlocal mutated
        # RIR 로드 시점에는 모든 metadata/manifest/PCM의 첫 검사가 끝난 상태다.
        if Path(path) == Path(prepared["data"]["rir_bank"]) and not mutated:
            mutated = True
            if target == "qa":
                qa = json.loads(changed_path.read_text())
                qa["data_ready"] = False
                changed_path.write_text(json.dumps(qa))
            else:
                with changed_path.open("ab") as handle:
                    handle.write(b"changed during validation")
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(synth_dataset.np, "load", load_after_mutation)
    with pytest.raises(ValueError, match="SHA-256"):
        synth_dataset.validate_prepared_data(prepared["data"])
    assert mutated


def test_prepared_qa_digest_is_bound_to_parsed_bytes(prepared):
    from deep_anc.data.synth_dataset import validate_prepared_data
    path = Path(prepared["data"]["noise_manifest_dir"]) / "qa.json"
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata, _ = validate_prepared_data(prepared["data"])
    assert metadata["qa_sha256"] == expected


@pytest.mark.parametrize("handoff", [-1, -2000, 256.0, 256.5, True, False, "256", None, float("nan"), float("inf")])
def test_prepared_handoff_requires_nonnegative_integer(prepared, handoff):
    prepared["duct"]["secondary_path"]["handoff_extra_samples"] = handoff
    with pytest.raises(ValueError, match="handoff_extra_samples"):
        _dataset(prepared)
    namespace = runpy.run_path(str(REPO_ROOT / "scripts/bench/check_acoustic_training_data.py"))
    with pytest.raises(ValueError, match="handoff_extra_samples"):
        namespace["check_acoustic_training_data"](prepared)


def test_prepared_handoff_zero_and_default_are_valid(prepared):
    from deep_anc.config import DEFAULT_HANDOFF_SAMPLES
    from deep_anc.data.synth_dataset import validate_prepared_handoff
    prepared["duct"]["secondary_path"]["handoff_extra_samples"] = 0
    assert validate_prepared_handoff(prepared["duct"]) == 0
    prepared["duct"]["secondary_path"].pop("handoff_extra_samples")
    assert validate_prepared_handoff(prepared["duct"]) == DEFAULT_HANDOFF_SAMPLES


def test_missing_raw_is_not_a_synthetic_replacement(prepared):
    directory = Path(prepared["data"]["noise_manifest_dir"])
    (directory.parent / "raw/speech_train.wav").unlink()
    with pytest.raises(FileNotFoundError):
        _dataset(prepared)


def test_empty_split_is_rejected_even_when_hashes_are_current(prepared):
    _rewrite_manifest(prepared, "speech", lambda rows: rows.pop())
    with pytest.raises(ValueError, match="비어 있는"):
        _dataset(prepared)


def test_cross_split_group_is_rejected(prepared):
    _rewrite_manifest(prepared, "speech", lambda rows: rows[1].update(group_id=rows[0]["group_id"]))
    with pytest.raises(ValueError, match="group_id"):
        _dataset(prepared)


def test_cross_split_raw_hash_is_rejected(prepared):
    def duplicate(rows):
        rows[1].update(path=rows[0]["path"], sha256=rows[0]["sha256"])
    _rewrite_manifest(prepared, "speech", duplicate)
    with pytest.raises(ValueError, match="SHA-256 중복"):
        _dataset(prepared)


@pytest.mark.parametrize("field,value", [("data_ready", False), ("schema_version", True),
                                        ("performance_claim_allowed", True), ("manifest_sha256", None)])
def test_nonready_or_invalid_qa_cannot_be_promoted(prepared, field, value):
    path = Path(prepared["data"]["noise_manifest_dir"]) / "qa.json"
    qa = json.loads(path.read_text())
    qa[field] = value
    path.write_text(json.dumps(qa))
    with pytest.raises(ValueError):
        _dataset(prepared)


def test_inventory_disagreement_is_rejected_even_with_fresh_file_hash(prepared):
    directory = Path(prepared["data"]["noise_manifest_dir"])
    path = directory / "inventory.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["qa_valid"] = False
    _write_rows(path, rows)
    qa = json.loads((directory / "qa.json").read_text())
    qa["inventory_sha256"] = _hash(path)
    (directory / "qa.json").write_text(json.dumps(qa))
    with pytest.raises(ValueError, match="inventory"):
        _dataset(prepared)


def test_wrong_raw_header_is_not_hidden_by_matching_hash(prepared):
    _rewrite_manifest(prepared, "speech", lambda rows: rows[0].update(sample_rate=16000))
    with pytest.raises(ValueError, match="header"):
        _dataset(prepared)


def test_pool_manifest_disappearance_after_initial_validation_fails(prepared):
    ds = _dataset(prepared)
    (Path(prepared["data"]["noise_manifest_dir"]) / "speech.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        ds._pool("speech", np.random.default_rng(1))


def test_pool_manifest_mutation_after_initial_validation_fails(prepared):
    ds = _dataset(prepared)
    path = Path(prepared["data"]["noise_manifest_dir"]) / "speech.jsonl"
    with path.open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="SHA-256"):
        ds._pool("speech", np.random.default_rng(1))


def test_sample_rate_mismatch_with_secondary_is_rejected(prepared):
    path = Path(prepared["duct"]["secondary_path"]["npz"])
    with np.load(path) as archive:
        values = {key: archive[key] for key in archive.files}
    values["sample_rate"] = 16000
    np.savez(path, **values)
    with pytest.raises(ValueError, match="prepared S sample_rate"):
        _dataset(prepared)


@pytest.mark.parametrize("fault", ["missing", "empty", "nan", "shape", "fs", "duplicate", "metadata", "overflow"])
def test_invalid_rir_never_falls_back(prepared, fault):
    path = Path(prepared["data"]["rir_bank"])
    with np.load(path) as archive:
        values = {key: archive[key] for key in archive.files}
    if fault == "missing":
        path.unlink()
    else:
        if fault == "empty": values["p_ref"] = np.zeros((6, 0))
        if fault == "nan": values["p_ref"][0, 0] = np.nan
        if fault == "shape": values["p_ref"] = np.ones((3, 12))
        if fault == "fs": values["sample_rate"] = 16000
        if fault == "metadata": values.pop("sample_rate")
        if fault == "overflow": values["p_ref"] = np.full((6, 12), 1e300)
        if fault == "duplicate":
            for key in ("p_ref", "p_err", "f_fb"): values[key][1] = values[key][0]
        np.savez(path, **values)
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        _dataset(prepared)


def test_strict_mode_cannot_use_in_memory_rir_bypass(prepared):
    with pytest.raises(ValueError, match="우회"):
        _dataset(prepared, rir_bank={})


def test_legacy_missing_manifest_fallback_is_preserved(prepared):
    prepared["data"].update(require_prepared_data=False, noise_manifest_dir="/missing_fixture_manifests")
    ds = _dataset(prepared)
    ds.mix_ratio = {"speech": 1.0}
    assert torch.isfinite(next(iter(ds))["x"]).all()


def test_strict_noise_pool_checks_hash_again_before_sampling(prepared):
    ds = _dataset(prepared)
    pool = ds._pool("speech", np.random.default_rng(1))
    with Path(pool.entries[0]["path"]).open("ab") as handle:
        handle.write(b"changed after dataset construction")
    with pytest.raises(RuntimeError, match="strict"):
        pool.sample_segment(1024)
    assert np.all(pool._active_weights > 0)


def test_strict_noise_pool_does_not_retry_decode_failure(prepared, monkeypatch):
    ds = _dataset(prepared)
    pool = ds._pool("speech", np.random.default_rng(1))
    def fail(*args, **kwargs): raise RuntimeError("손상 fixture")
    monkeypatch.setattr("deep_anc.data.noise_pool.sf.read", fail)
    with pytest.raises(RuntimeError, match="strict"):
        pool.sample_segment(1024)
    assert np.all(pool._active_weights > 0)


def test_acoustic_reference_does_not_add_digital_or_secondary_delay(prepared, monkeypatch):
    ds = _dataset(prepared)
    impulse = np.zeros(ds.segment, dtype=np.float32)
    impulse[0] = 1
    monkeypatch.setattr(ds, "_sample_source", lambda *args: impulse.copy())
    ds.snr_range, ds.level_range, ds.dc_hum_prob = (300, 300), (0, 0), 0
    class Rng:
        def uniform(self, low, high): return low
        def integers(self, low, high): return low
        def choice(self, values): return values[0]
        def standard_normal(self, size): return np.zeros(size)
        def random(self): return 0.9
    item = ds._make_item(Rng(), ds.synthetic_generator(1))
    assert int(torch.argmax(item["x"][0].abs())) == 2
    assert int(torch.argmax(item["d"][0].abs())) == 5
    secondary = load_secondary_path(prepared["duct"]["secondary_path"]["npz"])
    plant = DifferentiableSecondaryPath(secondary, handoff_extra_samples=256)
    assert plant.base_delay == 7 + 256


def test_prepared_config_metadata_is_preserved_without_physical_claims(prepared):
    from deep_anc.train.trainer import cfg_snapshot
    snapshot = cfg_snapshot(prepared)
    assert snapshot["physics_status"] == "acoustic_rir_training"
    assert snapshot["digital_reference_lead_samples"] == 0
    assert snapshot["data"]["source_metadata"]["performance_claim_allowed"] is False
    assert snapshot["data"]["source_metadata"]["primary_gain_calibration_validated"] is False
    assert prepared["resume"] is None and prepared["num_workers"] == 0
    assert prepared["run_until_step"] == 2


def test_future_source_does_not_change_acoustic_rir_prefix(prepared, monkeypatch):
    ds = _dataset(prepared)
    ds.snr_range, ds.level_range, ds.dc_hum_prob = (300, 300), (0, 0), 0
    class Rng:
        def uniform(self, low, high): return low
        def integers(self, low, high): return low
        def choice(self, values): return values[0]
        def standard_normal(self, size): return np.zeros(size)
        def random(self): return 0.9
    source = np.random.default_rng(1).normal(0, 0.1, ds.segment).astype(np.float32)
    monkeypatch.setattr(ds, "_sample_source", lambda *args: source.copy())
    first = ds._make_item(Rng(), ds.synthetic_generator(1))
    source[512:] *= -3
    second = ds._make_item(Rng(), ds.synthetic_generator(1))
    torch.testing.assert_close(first["x"][:, :512], second["x"][:, :512], atol=1e-7, rtol=0)
    torch.testing.assert_close(first["d"][:, :512], second["d"][:, :512], atol=1e-7, rtol=0)


def test_invalid_target_config_fails_before_iteration(prepared):
    prepared["data"]["synthetic_target_band_hz"] = [800, 8000]
    with pytest.raises(ValueError, match="target_band_hz"):
        _dataset(prepared)


def test_disabled_target_preserves_seeded_default_waveforms():
    default = SyntheticNoise(8000, seed=10)
    disabled = SyntheticNoise(8000, seed=10, target_band_hz=[800, 1600], target_probability=0)
    for kind in KINDS:
        np.testing.assert_array_equal(default.generate(8000, kind), disabled.generate(8000, kind))


def test_target_mix_keeps_low_fundamentals_and_adds_above_1khz():
    generator = SyntheticNoise(8000, seed=10, target_band_hz=[800, 1600], target_probability=0.5)
    frequencies = np.array([generator._pick_f0() for _ in range(1000)])
    assert np.count_nonzero(frequencies < 800) > 200
    assert np.count_nonzero(frequencies >= 1000) > 200
    assert np.all(frequencies < 1600)


@pytest.mark.parametrize("kind", KINDS)
def test_each_target_kind_has_target_energy_and_finite_samples(kind):
    generator = SyntheticNoise(8000, seed=10, target_band_hz=[800, 1600], target_probability=1)
    values = generator.generate(8000, kind)
    power = np.abs(np.fft.rfft(values)) ** 2
    frequencies = np.fft.rfftfreq(len(values), 1 / 8000)
    assert np.isfinite(values).all()
    assert power[(frequencies >= 800) & (frequencies < 1600)].sum() / power.sum() > 0.3


def test_target_harmonics_stop_before_aliasing(monkeypatch):
    generator = SyntheticNoise(8000, seed=10, target_band_hz=[800, 1600], target_probability=1)
    monkeypatch.setattr(generator, "_pick_f0", lambda: 1500.0)
    spectrum = np.abs(np.fft.rfft(generator.generate(8000, "tone_harmonics"))) ** 2
    assert spectrum[1500] > 1
    assert spectrum[3500] < 1e-7


@pytest.mark.parametrize("changes", [
    {"target_probability": -0.1}, {"target_probability": 1.1}, {"target_probability": np.nan},
    {"target_probability": 0.5}, {"target_band_hz": [800, 4000]},
    {"target_band_hz": [1600, 800]}, {"target_band_hz": [800, np.nan]},
])
def test_invalid_target_generation_settings_fail(changes):
    with pytest.raises(ValueError):
        SyntheticNoise(8000, **changes)


def test_eval_uses_same_opt_in_generator_as_training(prepared):
    ds = _dataset(prepared, "val")
    generator = ds.synthetic_generator(12)
    assert generator.target_band == (1000, 1600) and generator.target_probability == 0.5
    batch = make_eval_batch(ds, 2, seed=12)
    assert batch["x"].shape == (2, 2, 2048)


def test_checker_smoke_is_local_only_and_not_training(prepared):
    namespace = runpy.run_path(str(REPO_ROOT / "scripts/bench/check_acoustic_training_data.py"))
    report = namespace["check_acoustic_training_data"](prepared)
    assert len(report["smoke"]) == 16 and len(report["synthetic_coverage"]) == 5
    assert [row["split"] for row in report["smoke"] if row["source_family"] == "machine"] == ["train"]
    assert report["drive_inventory_ready"] is None and report["training_launched"] is False
    assert report["performance_claim_allowed"] is False
    assert report["loss_plant_delay_samples"] == 263
    assert report["secondary_consistency_band_hz"] == (150, 600)
    assert report["target_band_hz"] == [1000, 1600]
    assert report["target_high_endpoint_included"] is False
    assert report["historical_target_band_hz"] == [800, 1600]
    assert all(row["mean_below_target_power_fraction"] >= row["mean_below_800_power_fraction"]
               for row in report["synthetic_coverage"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("frequency,expected", [(800, 0), (900, 0), (1000, 1), (1599, 1), (1600, 0)])
def test_training_qa_1k_target_edges_are_explicit(frequency, expected):
    namespace = runpy.run_path(str(REPO_ROOT / "scripts/bench/check_acoustic_training_data.py"))
    fraction = namespace["_band_fraction"]
    values = np.sin(2 * np.pi * frequency * np.arange(8000) / 8000)
    assert fraction(values, 8000, 1000, 1600) == pytest.approx(expected, abs=1e-12)


def test_training_qa_explicit_historical_target_is_not_relabelled(prepared):
    namespace = runpy.run_path(str(REPO_ROOT / "scripts/bench/check_acoustic_training_data.py"))
    prepared["data"]["synthetic_target_band_hz"] = [800, 1600]
    report = namespace["check_acoustic_training_data"](prepared)
    assert report["target_band_hz"] == [800, 1600]
    assert report["secondary_consistency_band_hz"] == (150, 600)
    assert report["performance_claim_allowed"] is False


def test_prepared_1k_generation_retains_low_band(prepared):
    generator = _dataset(prepared).synthetic_generator(10)
    frequencies = np.array([generator._pick_f0() for _ in range(1000)])
    assert generator.target_band == (1000, 1600)
    assert np.count_nonzero(frequencies < 800) > 200
    assert np.count_nonzero((frequencies >= 1000) & (frequencies < 1600)) > 300


def test_training_qa_missing_target_cannot_report_default_band(prepared):
    namespace = runpy.run_path(str(REPO_ROOT / "scripts/bench/check_acoustic_training_data.py"))
    prepared["data"].pop("synthetic_target_band_hz")
    prepared["data"]["synthetic_target_probability"] = 0
    with pytest.raises(ValueError, match="명시적인 synthetic_target_band_hz"):
        namespace["check_acoustic_training_data"](prepared)


def test_checker_rejects_different_valid_snapshots_between_splits(prepared, monkeypatch):
    namespace = runpy.run_path(str(REPO_ROOT / "scripts/bench/check_acoustic_training_data.py"))
    check = namespace["check_acoustic_training_data"]
    original_dataset = check.__globals__["SynthANCDataset"]
    constructors = 0
    def dataset_with_updated_qa(*args, **kwargs):
        nonlocal constructors
        constructors += 1
        if constructors == 2:
            path = Path(prepared["data"]["noise_manifest_dir"]) / "qa.json"
            qa = json.loads(path.read_text())
            qa["fixture_changed_after_train_split"] = True
            path.write_text(json.dumps(qa))
        return original_dataset(*args, **kwargs)
    monkeypatch.setitem(check.__globals__, "SynthANCDataset", dataset_with_updated_qa)
    with pytest.raises(ValueError, match="QA 도중"):
        check(prepared)


def test_checker_cli_missing_local_stage_fails_without_creating_data(tmp_path):
    missing = tmp_path / "not_staged"
    run = subprocess.run([
        sys.executable, str(REPO_ROOT / "scripts/bench/check_acoustic_training_data.py"),
        "--json", "--set", f"data.noise_manifest_dir={missing}",
    ], cwd=REPO_ROOT, text=True, capture_output=True, timeout=30)
    assert run.returncode == 1, run.stderr
    report = json.loads(run.stdout)
    assert report["data_ready"] is False and report["drive_inventory_ready"] is None
    assert not missing.exists()


def test_cpu_tiny_fixture_trainer_two_steps_preserves_acoustic_metadata(prepared, tmp_path, monkeypatch):
    """독립 합성 fixture의 CPU 모델/손실/저장 연결 검사이며 실제 corpus 학습이 아니다."""
    from deep_anc.train.trainer import Trainer
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)
    prepared["ckpt_dir"] = str(tmp_path / "fixture_training_only")
    trainer = Trainer(prepared)
    assert trainer.device.type == "cpu"
    assert trainer.criterion.plant.base_delay == 7 + 256
    trainer.train()
    assert trainer.step == 2
    assert all(torch.isfinite(parameter).all() for parameter in trainer.model.parameters())
    saved = torch.load(Path(prepared["ckpt_dir"]) / "ckpt/last.pt", map_location="cpu", weights_only=False)
    assert saved["step"] == 2
    assert saved["cfg"]["physics_status"] == "acoustic_rir_training"
    assert saved["cfg"]["digital_reference_lead_samples"] == 0
    assert saved["cfg"]["data"]["source_metadata"]["performance_claim_allowed"] is False
    snapshot = saved["cfg"]["data"]["source_metadata"]["prepared_data_snapshot"]
    assert snapshot == prepared["data"]["source_metadata"]["prepared_data_snapshot"]
    assert snapshot["data_ready"] is True
    assert snapshot["train_only_source_families"] == ["machine"]
    assert "machine" not in snapshot["effective_source_mix_by_split"]["val"]
    assert "machine" not in snapshot["effective_source_mix_by_split"]["test"]
    assert saved["cfg"]["trusted_band_hz"] == [150, 600]
    if trainer.writer:
        trainer.writer.close()
    if trainer.loss_log:
        trainer.loss_log.close()
