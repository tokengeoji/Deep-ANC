"""실측 ERR 광대역 batch 자격/결정성의 fail-closed 테스트."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from torch.utils.data import DataLoader

from deep_anc.data.broadband_batch_sampler import (
    BroadbandQualifiedBatchPlanner,
    MIN_VALID_ITEMS_PER_BAND,
    QUALIFIED_SAMPLING_MODE,
    REQUIRED_FAMILIES,
    build_broadband_batch_receipt,
    target_d_density_ratios,
)
from deep_anc.data.recorded_dataset import (
    RecordedANCDataset,
    apply_same_fir,
    make_recorded_eval_batch,
)
from deep_anc.data.synth_dataset import (
    BROADBAND_SYNTH_PRIMARY_GENERATOR_SCHEMA,
    SynthANCDataset,
)
from deep_anc.dsp.timing import PlantDelays, TrainingTimingContract
from deep_anc.config import REPO_ROOT, load_yaml


FS = 48_000
SEGMENT = 1024
BATCH = 16


def _population(
    tmp_path: Path, *, components_per_family: int = 4, split: str = "train"
):
    entries = []
    rng = np.random.default_rng(20260828)
    for family in REQUIRED_FAMILIES:
        for component_index in range(components_per_family):
            session_id = f"{family}-{component_index}"
            directory = tmp_path / "recorded" / session_id
            directory.mkdir(parents=True)
            # 4개 segment 모두 broadband target이다. 자격 판정은 이 ERR만 사용한다.
            source = (0.05 * rng.standard_normal(SEGMENT * 4)).astype(np.float32)
            err = source.copy()
            ref = np.roll(err, -1)
            sf.write(directory / "mics.wav", np.stack((err, ref), axis=1), FS, subtype="FLOAT")
            sf.write(directory / "source_aligned.wav", source, FS, subtype="FLOAT")
            sf.write(directory / "source.wav", source, FS, subtype="FLOAT")
            entries.append(
                {
                    "path": str(directory),
                    "split": split,
                    "session_id": session_id,
                    "source_family": family,
                    "group_id": f"{family}-component-{component_index}",
                    "source_pool_group_id": f"{family}-pool-{component_index}",
                }
            )
    manifest = tmp_path / "recorded.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(entry, sort_keys=True) for entry in entries) + "\n",
        encoding="utf-8",
    )
    return manifest, entries


def _receipt(
    tmp_path: Path, *, components_per_family: int = 4, split: str = "train"
):
    manifest, entries = _population(
        tmp_path, components_per_family=components_per_family, split=split
    )
    receipt = build_broadband_batch_receipt(
        manifest_path=manifest,
        entries=entries,
        sample_rate=FS,
        segment_samples=SEGMENT,
        batch_size=BATCH,
        valid_prefix_samples=256,
        split=split,
        edge_trim_samples=256,
        max_segments_per_session=4,
    )
    return manifest, entries, receipt


def test_receipt_blocks_when_family_band_has_fewer_than_four_components(tmp_path):
    _, _, receipt = _receipt(tmp_path, components_per_family=3)
    assert receipt["summary"]["status"] == "BLOCKED"
    assert any("eligible components 3 < 4" in row for row in receipt["summary"]["blockers"])
    with pytest.raises(ValueError, match="BLOCKED"):
        BroadbandQualifiedBatchPlanner(receipt)


def test_receipt_binds_split_and_requires_edge_trim_at_least_prefix(tmp_path):
    manifest, entries = _population(tmp_path, split="val")
    with pytest.raises(ValueError, match="segment/batch/trim"):
        build_broadband_batch_receipt(
            manifest_path=manifest,
            entries=entries,
            sample_rate=FS,
            segment_samples=SEGMENT,
            batch_size=BATCH,
            valid_prefix_samples=256,
            split="val",
            edge_trim_samples=255,
        )
    receipt = build_broadband_batch_receipt(
        manifest_path=manifest,
        entries=entries,
        sample_rate=FS,
        segment_samples=SEGMENT,
        batch_size=BATCH,
        valid_prefix_samples=256,
        split="val",
        edge_trim_samples=256,
        max_segments_per_session=4,
    )
    assert receipt["summary"]["status"] == "PASS"
    planner = BroadbandQualifiedBatchPlanner(
        receipt, expected_split="val", expected_valid_prefix_samples=256
    )
    assert planner.split == "val"
    with pytest.raises(ValueError, match="split"):
        BroadbandQualifiedBatchPlanner(receipt, expected_split="train")
    with pytest.raises(ValueError, match="prefix"):
        BroadbandQualifiedBatchPlanner(
            receipt, expected_split="val", expected_valid_prefix_samples=512
        )


def test_batch_is_family_balanced_and_each_band_has_at_least_four_valid_items(tmp_path):
    _, _, receipt = _receipt(tmp_path)
    assert receipt["summary"]["status"] == "PASS"
    planner = BroadbandQualifiedBatchPlanner(receipt)
    left = planner.batch(73, seed=20260803)
    right = planner.batch(73, seed=20260803)
    resumed = tuple(
        planner.item(73 * BATCH + offset, seed=20260803) for offset in range(BATCH)
    )
    assert left == right == resumed
    assert {
        family: sum(row.source_family == family for row in left)
        for family in REQUIRED_FAMILIES
    } == {family: BATCH // 4 for family in REQUIRED_FAMILIES}
    assert all(
        sum(row.valid_bands[index] for row in left) >= MIN_VALID_ITEMS_PER_BAND
        for index in range(7)
    )
    assert len({(row.session_id, row.start_frame) for row in left}) == BATCH


def test_raw_err_change_invalidates_the_receipt(tmp_path):
    _, entries, receipt = _receipt(tmp_path)
    mics = Path(entries[0]["path"]) / "mics.wav"
    values, fs = sf.read(mics, dtype="float32", always_2d=True)
    values[0, 0] += 0.01
    sf.write(mics, values, fs, subtype="FLOAT")
    with pytest.raises(ValueError, match="mics.wav bytes"):
        BroadbandQualifiedBatchPlanner(receipt)


def test_dataset_rechecks_common_eq_after_exact_plan_and_rejects_mix(tmp_path):
    manifest, _, receipt = _receipt(tmp_path)
    receipt_path = tmp_path / "batch_receipt.json"
    receipt_bytes = json.dumps(receipt, sort_keys=True).encode("utf-8")
    receipt_path.write_bytes(receipt_bytes)
    cfg = {
        "sample_rate": FS,
        "segment_seconds": SEGMENT / FS,
        "reference_mode": "digital",
        "digital_reference_lead_samples": 0,
        "recorded_sampling": QUALIFIED_SAMPLING_MODE,
        "recorded_broadband_batch_receipt": str(receipt_path),
        "recorded_broadband_batch_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "broadband_channel_dropout": {
            "reference_probability": 0.0,
            "error_probability": 0.0,
        },
        "recorded_augment": {
            "enabled": True,
            "level_db_range": [-12.0, 6.0],
            "polarity_flip": True,
            "mic_noise_snr_db": [12.0, 40.0],
            "eq_tilt_db": 6.0,
            "eq_band_db": 4.0,
            "eq_bands_hz": [100.0, 300.0, 700.0, 1400.0],
            "mix_probability": 0.0,
            "mix_weight_range": [0.0, 0.7],
            "lead_jitter_samples": 0.0,
        },
        "closed_loop": {"feedback_delay_samples": [1, 2]},
    }
    left = RecordedANCDataset(
        manifest, cfg, split="train", seed=99, training_batch_size=BATCH,
        broadband_valid_prefix_samples=256,
    )
    right = RecordedANCDataset(
        manifest, cfg, split="train", seed=99, training_batch_size=BATCH,
        broadband_valid_prefix_samples=256,
    )
    left_items = iter(left)
    right_items = iter(right)
    for _ in range(BATCH):
        a, b = next(left_items), next(right_items)
        assert np.array_equal(a["x"].numpy(), b["x"].numpy())
        assert np.array_equal(a["d"].numpy(), b["d"].numpy())

    batch = next(iter(DataLoader(left, batch_size=BATCH, num_workers=0)))
    valid_counts = np.zeros(7, dtype=np.int64)
    for target in batch["d"][:, 0].numpy():
        valid_counts += np.asarray(
            [value >= 0.25 for value in target_d_density_ratios(
                target,
                sample_rate=FS,
                bands_hz=left._broadband_batch_planner.bands,
            )],
            dtype=np.int64,
        )
    assert np.all(valid_counts >= MIN_VALID_ITEMS_PER_BAND)

    broken = dict(cfg)
    broken["recorded_augment"] = {
        "enabled": True,
        "eq_tilt_db": 0.0,
        "eq_band_db": 0.0,
        "mix_probability": 0.1,
    }
    with pytest.raises(ValueError, match="mix"):
        RecordedANCDataset(
            manifest, broken, split="train", training_batch_size=BATCH,
            broadband_valid_prefix_samples=256,
        )


def test_multiworker_dataloader_batches_and_resume_are_global_index_equivalent(tmp_path):
    manifest, _, receipt = _receipt(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    receipt_bytes = json.dumps(receipt, sort_keys=True).encode("utf-8")
    receipt_path.write_bytes(receipt_bytes)
    cfg = {
        "sample_rate": FS,
        "segment_seconds": SEGMENT / FS,
        "reference_mode": "digital",
        "digital_reference_lead_samples": 0,
        "recorded_sampling": QUALIFIED_SAMPLING_MODE,
        "recorded_broadband_batch_receipt": str(receipt_path),
        "recorded_broadband_batch_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "broadband_channel_dropout": {
            "reference_probability": 0.0,
            "error_probability": 0.0,
        },
        "recorded_augment": {"enabled": True},
        "closed_loop": {"feedback_delay_samples": [1, 2]},
    }
    single = RecordedANCDataset(
        manifest, cfg, split="train", seed=77, training_batch_size=BATCH,
        broadband_valid_prefix_samples=256,
    )
    multi = RecordedANCDataset(
        manifest, cfg, split="train", seed=77, training_batch_size=BATCH,
        broadband_valid_prefix_samples=256,
    )
    single_iter = iter(DataLoader(single, batch_size=BATCH, num_workers=0))
    multi_iter = iter(DataLoader(multi, batch_size=BATCH, num_workers=2))
    single_batches = [next(single_iter) for _ in range(3)]
    multi_batches = [next(multi_iter) for _ in range(3)]
    for expected, actual in zip(single_batches, multi_batches, strict=True):
        assert np.array_equal(expected["x"].numpy(), actual["x"].numpy())
        assert np.array_equal(expected["d"].numpy(), actual["d"].numpy())

    resumed = RecordedANCDataset(
        manifest,
        cfg,
        split="train",
        seed=77,
        training_batch_size=BATCH,
        resume_batch_index=2,
        broadband_valid_prefix_samples=256,
    )
    resumed_batch = next(iter(DataLoader(resumed, batch_size=BATCH, num_workers=2)))
    assert np.array_equal(single_batches[2]["x"].numpy(), resumed_batch["x"].numpy())
    assert np.array_equal(single_batches[2]["d"].numpy(), resumed_batch["d"].numpy())


def test_recorded_error_dropout_is_exact_and_deterministic(tmp_path):
    manifest, _, receipt = _receipt(tmp_path)
    receipt_path = tmp_path / "dropout_receipt.json"
    receipt_bytes = json.dumps(receipt, sort_keys=True).encode("utf-8")
    receipt_path.write_bytes(receipt_bytes)
    cfg = {
        "sample_rate": FS,
        "segment_seconds": SEGMENT / FS,
        "reference_mode": "digital",
        "digital_reference_lead_samples": 0,
        "recorded_sampling": QUALIFIED_SAMPLING_MODE,
        "recorded_broadband_batch_receipt": str(receipt_path),
        "recorded_broadband_batch_receipt_sha256": hashlib.sha256(
            receipt_bytes
        ).hexdigest(),
        "broadband_channel_dropout": {
            "reference_probability": 0.0,
            "error_probability": 1.0,
        },
        "recorded_augment": {"enabled": False},
        "closed_loop": {"feedback_delay_samples": [1, 2]},
    }
    left = RecordedANCDataset(
        manifest,
        cfg,
        split="train",
        seed=71,
        training_batch_size=BATCH,
        broadband_valid_prefix_samples=256,
    )
    right = RecordedANCDataset(
        manifest,
        cfg,
        split="train",
        seed=71,
        training_batch_size=BATCH,
        broadband_valid_prefix_samples=256,
    )
    left_batch = next(iter(DataLoader(left, batch_size=BATCH, num_workers=0)))
    right_batch = next(iter(DataLoader(right, batch_size=BATCH, num_workers=0)))
    assert np.array_equal(left_batch["x"].numpy(), right_batch["x"].numpy())
    assert np.any(left_batch["x"][:, 0].numpy() != 0.0)
    assert np.all(left_batch["x"][:, 1].numpy() == 0.0)


def test_recorded_common_eq_reads_right_session_history_before_valid_crop(tmp_path):
    manifest, entries, receipt = _receipt(tmp_path)
    receipt_path = tmp_path / "eq_receipt.json"
    receipt_bytes = json.dumps(receipt, sort_keys=True).encode("utf-8")
    receipt_path.write_bytes(receipt_bytes)
    cfg = {
        "sample_rate": FS,
        "segment_seconds": SEGMENT / FS,
        "reference_mode": "digital",
        "digital_reference_lead_samples": 0,
        "recorded_sampling": QUALIFIED_SAMPLING_MODE,
        "recorded_broadband_batch_receipt": str(receipt_path),
        "recorded_broadband_batch_receipt_sha256": hashlib.sha256(
            receipt_bytes
        ).hexdigest(),
        "broadband_channel_dropout": {
            "reference_probability": 0.0,
            "error_probability": 0.0,
        },
        "recorded_augment": {
            "enabled": True,
            "eq_tilt_db": 1.0,
            "eq_band_db": 0.0,
            "mix_probability": 0.0,
        },
        "closed_loop": {"feedback_delay_samples": [1, 2]},
    }
    dataset = RecordedANCDataset(
        manifest,
        cfg,
        split="train",
        seed=9,
        training_batch_size=BATCH,
        broadband_valid_prefix_samples=256,
    )
    planned = dataset._broadband_batch_planner.item(0, seed=9)
    session_index = dataset._qualified_session_indices[planned.session_id]
    pair = dataset._draw_pair(
        session_index,
        np.random.default_rng(0),
        exact_start=planned.start_frame,
    )
    assert pair is not None
    assert pair[1].size == 256 + SEGMENT + 64
    kernel = np.hanning(129).astype(np.float32)
    kernel /= np.sum(kernel)
    local = apply_same_fir(pair[1], kernel)[:-64]
    err, _, _ = dataset._session(session_index)
    full = apply_same_fir(err, kernel)
    expected = full[planned.start_frame - 256 : planned.start_frame + SEGMENT]
    assert np.allclose(local[256:], expected[256:], rtol=1e-6, atol=1e-7)


def test_recorded_val_batch_uses_val_receipt_and_is_fixed(tmp_path):
    manifest, _, receipt = _receipt(tmp_path, split="val")
    receipt_path = tmp_path / "val_receipt.json"
    receipt_bytes = json.dumps(receipt, sort_keys=True).encode("utf-8")
    receipt_path.write_bytes(receipt_bytes)
    cfg = {
        "sample_rate": FS,
        "segment_seconds": SEGMENT / FS,
        "reference_mode": "digital",
        "digital_reference_lead_samples": 0,
        "recorded_sampling": QUALIFIED_SAMPLING_MODE,
        "recorded_broadband_val_batch_receipt": str(receipt_path),
        "recorded_broadband_val_batch_receipt_sha256": hashlib.sha256(
            receipt_bytes
        ).hexdigest(),
        "broadband_channel_dropout": {
            "reference_probability": 0.0,
            "error_probability": 0.0,
        },
        "recorded_augment": {"enabled": False},
        "closed_loop": {"feedback_delay_samples": [1, 2]},
    }
    left = RecordedANCDataset(
        manifest,
        cfg,
        split="val",
        seed=1234,
        training_batch_size=BATCH,
        broadband_valid_prefix_samples=256,
    )
    right = RecordedANCDataset(
        manifest,
        cfg,
        split="val",
        seed=1234,
        training_batch_size=BATCH,
        broadband_valid_prefix_samples=256,
    )
    first = make_recorded_eval_batch(left, BATCH)
    second = make_recorded_eval_batch(right, BATCH)
    assert np.array_equal(first["x"].numpy(), second["x"].numpy())
    assert np.array_equal(first["d"].numpy(), second["d"].numpy())
    assert np.all(first["valid_start_sample"].numpy() == 256)


def _identity_timing() -> TrainingTimingContract:
    return TrainingTimingContract.derive(
        primary_fir=[1.0],
        plant_delays=PlantDelays(
            primary_delay_samples=0,
            secondary_delay_samples=0,
            handoff_samples=0,
            sample_rate=FS,
        ),
    )


def _synthetic_fixture(
    *, resume_batch_index: int = 0, batch_size: int = BATCH
) -> SynthANCDataset:
    data = load_yaml(REPO_ROOT / "configs/data_sim.yaml")
    duct = load_yaml(REPO_ROOT / "configs/duct.yaml")
    data.update(
        {
            "segment_seconds": 4096 / FS,
            "source_mix_ratio": {"synthetic": 1.0},
            "digital_primary_path_mode": "rir_surrogate",
            "digital_reference_lead_samples": 0,
            "level_dbfs": [-30.0, -30.0],
            "snr_mic_noise_db": [300.0, 300.0],
            "dc_hum_prob": 0.0,
        }
    )
    identity = np.ones((12, 1), dtype=np.float32)
    timing = _identity_timing()
    dataset = SynthANCDataset(
        data,
        duct,
        split="train",
        seed=123,
        rir_bank={"p_ref": identity, "p_err": identity, "f_fb": identity},
        training_batch_size=batch_size,
        resume_batch_index=resume_batch_index,
        broadband_batch_qualified=True,
        broadband_primary_operator=lambda values: values.copy(),
        broadband_primary_generator_schema=BROADBAND_SYNTH_PRIMARY_GENERATOR_SCHEMA,
        broadband_primary_history_samples=0,
        broadband_valid_prefix_samples=256,
        broadband_timing_contract=timing,
    )

    def white_source(rng, _synth, n_samples=None):
        size = dataset.segment if n_samples is None else int(n_samples)
        return rng.standard_normal(size).astype(np.float32)

    dataset._sample_source = white_source
    return dataset


def test_synthetic_actual_primary_path_target_has_four_valid_items_per_band():
    dataset = _synthetic_fixture()
    batch = next(iter(DataLoader(dataset, batch_size=BATCH, num_workers=0)))
    valid_counts = np.zeros(7, dtype=np.int64)
    for target in batch["d"][:, 0].numpy():
        valid_counts += np.asarray(
            [
                value >= 0.25
                for value in target_d_density_ratios(
                    target,
                    sample_rate=FS,
                    bands_hz=dataset._broadband_bands,
                )
            ],
            dtype=np.int64,
        )
    assert np.all(valid_counts >= MIN_VALID_ITEMS_PER_BAND)


def test_synthetic_multiworker_and_resume_are_global_index_equivalent():
    single = _synthetic_fixture()
    multi = _synthetic_fixture()
    one = iter(DataLoader(single, batch_size=BATCH, num_workers=0))
    two = iter(DataLoader(multi, batch_size=BATCH, num_workers=2))
    expected = [next(one) for _ in range(3)]
    actual = [next(two) for _ in range(3)]
    for left, right in zip(expected, actual, strict=True):
        assert np.array_equal(left["x"].numpy(), right["x"].numpy())
        assert np.array_equal(left["d"].numpy(), right["d"].numpy())
    resumed = next(
        iter(DataLoader(_synthetic_fixture(resume_batch_index=2), batch_size=BATCH, num_workers=2))
    )
    assert np.array_equal(expected[2]["x"].numpy(), resumed["x"].numpy())
    assert np.array_equal(expected[2]["d"].numpy(), resumed["d"].numpy())


def test_synthetic_density_retry_exhaustion_fails_closed():
    dataset = _synthetic_fixture()

    def silent_source(_rng, _synth, n_samples=None):
        size = dataset.segment if n_samples is None else int(n_samples)
        return np.zeros(size, dtype=np.float32)

    dataset._sample_source = silent_source
    with pytest.raises(RuntimeError, match="128회"):
        next(iter(dataset))


def test_synthetic_batch_four_is_rejected_instead_of_requiring_all_seven_bands():
    with pytest.raises(ValueError, match="5 이상"):
        _synthetic_fixture(batch_size=4)


def test_synthetic_broadband_rejects_missing_manifest_escape_hatch():
    data = load_yaml(REPO_ROOT / "configs/data_sim.yaml")
    duct = load_yaml(REPO_ROOT / "configs/duct.yaml")
    data["allow_missing_source_manifests"] = True
    with pytest.raises(ValueError, match="누락 manifest"):
        SynthANCDataset(
            data,
            duct,
            split="train",
            training_batch_size=BATCH,
            broadband_batch_qualified=True,
            broadband_primary_operator=lambda values: values.copy(),
            broadband_primary_generator_schema=BROADBAND_SYNTH_PRIMARY_GENERATOR_SCHEMA,
            broadband_primary_history_samples=0,
            broadband_valid_prefix_samples=256,
            broadband_timing_contract=_identity_timing(),
        )


def test_synthetic_broadband_never_promotes_compact_primary_generator():
    data = load_yaml(REPO_ROOT / "configs/data_sim.yaml")
    duct = load_yaml(REPO_ROOT / "configs/duct.yaml")
    data["source_mix_ratio"] = {"synthetic": 1.0}
    with pytest.raises(ValueError, match="BLOCKED_COMPACT_PRIMARY_GENERATOR"):
        SynthANCDataset(
            data,
            duct,
            split="train",
            training_batch_size=BATCH,
            broadband_batch_qualified=True,
        )
