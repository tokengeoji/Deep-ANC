"""``RecordedANCDataset`` 의 재정렬본 우선·lead 부기·증강 테스트.

증강 하나하나에 **선형성 보존** 테스트를 붙인다. 덕트 플랜트를 LTI 로 보면 유효한
증강은 "플랜트와 교환 가능한 연산"뿐이고, 그렇지 않은 증강은 조용히 플랜트를 바꾼다 —
그건 증강이 아니라 오염이고, 결함 2 와 같은 종류의 사고다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from deep_anc.config import REPO_ROOT, load_yaml
from deep_anc.data.recorded_dataset import (
    RecordedANCDataset,
    RecordedAugmentConfig,
    RecordedLeadPlan,
    apply_same_fir,
    common_eq_kernel,
)
from deep_anc.dsp.timing import PlantDelays, TrainingTimingContract

FS = 48000
SEGMENT_SECONDS = 0.25
PLANT = np.array([0.0] * 40 + [1.0, -0.45, 0.2, -0.08], dtype=np.float64)


def _write_session(
    root: Path,
    session_id: str,
    *,
    seconds: float = 4.0,
    aligned: bool = True,
    timeline: dict | None = None,
    seed: int = 3,
) -> dict:
    """d = PLANT ∗ source 인 세션을 만든다. 정답 관계가 알려진 데이터다."""

    directory = root / session_id
    directory.mkdir(parents=True, exist_ok=True)
    n = int(seconds * FS)
    rng = np.random.default_rng(seed)
    source = rng.standard_normal(n).astype(np.float32) * 0.06
    err = np.convolve(source, PLANT, mode="same").astype(np.float32)
    ref = np.roll(err, -142).astype(np.float32)
    sf.write(directory / "mics.wav", np.stack([err, ref], axis=1), FS, subtype="PCM_32")
    sf.write(directory / "source.wav", source, FS, subtype="FLOAT")
    if aligned:
        sf.write(directory / "source_aligned.wav", source, FS, subtype="FLOAT")
    meta = {
        "session_id": session_id,
        "source_family": "environment",
        "group_id": "environment-test",
        "seconds": seconds,
        "sample_rate": FS,
    }
    if timeline is not None:
        meta["timeline"] = timeline
    (directory / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    return {
        "path": str(directory),
        "split": "train",
        "session_id": session_id,
        "group_id": "environment-test",
        "source_family": "environment",
        "tag": "recorded",
        "duration_s": seconds,
        "sample_rate": FS,
        "channels": 2,
    }


def _manifest(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "recorded_train.jsonl"
    path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )
    return path


def _cfg(**overrides) -> dict:
    cfg = {
        "sample_rate": FS,
        "segment_seconds": SEGMENT_SECONDS,
        "reference_mode": "digital",
        "digital_reference_lead_samples": 116,
        "closed_loop": {"feedback_delay_samples": [512, 1024]},
    }
    cfg.update(overrides)
    if cfg.get("recorded_lead_mode") == "timeline" and "d_noise_delay_samples" in cfg:
        # 실제 measured P fixture처럼 compact FIR의 최대 탭이 bulk 뒤 247에 있다.
        fir = np.zeros(256, dtype=np.float32)
        fir[247] = 1.0
        primary_delay = int(cfg["d_noise_delay_samples"])
        lead = int(cfg["digital_reference_lead_samples"])
        cfg["training_timing_contract"] = TrainingTimingContract.derive(
            primary_fir=fir,
            plant_delays=PlantDelays(
                primary_delay_samples=primary_delay,
                secondary_delay_samples=primary_delay + lead,
                handoff_samples=0,
                sample_rate=FS,
            ),
        ).model_dump()
    return cfg


# ------------------------------------------------------------------ 재정렬본 우선
def test_aligned_source_is_preferred_over_the_raw_playback_array(tmp_path):
    """``source_aligned.wav`` 가 있으면 그것을 읽는다.

    ``source.wav`` 는 재생 배열이지 방출 시각이 아니다 — 그것을 학습이 쓰는 것이
    결함 2 였다.
    """

    root = tmp_path / "recorded"
    entry = _write_session(root, "20260806_000000_file")
    # 재정렬본을 알아볼 수 있게 표시한다 (원본과 다른 값).
    marker = np.full(int(4.0 * FS), 0.5, dtype=np.float32)
    sf.write(root / "20260806_000000_file" / "source_aligned.wav", marker, FS, subtype="FLOAT")

    dataset = RecordedANCDataset(_manifest(tmp_path, [entry]), _cfg(), split="train")
    _, _, source = dataset._session(0)
    assert np.allclose(source[:1000], 0.5)


def test_require_aligned_source_fails_closed(tmp_path):
    """negative fixture: 재정렬본이 없는데 요구하면 **거부**해야 한다."""

    root = tmp_path / "recorded"
    entry = _write_session(root, "20260806_000001_file", aligned=False)
    dataset = RecordedANCDataset(
        _manifest(tmp_path, [entry]), _cfg(require_aligned_source=True), split="train"
    )
    with pytest.raises(FileNotFoundError, match="source_aligned.wav"):
        dataset._session(0)


# ------------------------------------------------------------------ lead 부기
def test_lead_cannot_be_hand_written_against_the_derivation():
    """``K' = (D_noise + K) − d_recorded`` 를 어기는 lead 는 만들 수 없다."""

    plan = RecordedLeadPlan.from_timeline(
        total_advance_samples=1718,
        recorded_delay_samples=142.7,
        constant_lead_samples=116,
    )
    assert plan.lead_samples == 1575
    with pytest.raises(ValueError, match="lead 유도 관계 위반"):
        RecordedLeadPlan(
            mode="timeline",
            lead_samples=113,            # ← 손으로 쓴 값
            constant_lead_samples=116,
            total_advance_samples=1718,
            recorded_delay_samples=142.7,
        )


def test_timeline_lead_mode_refuses_sessions_without_measured_timeline(tmp_path):
    """negative fixture: 측정값이 없으면 lead 를 추측으로 채우지 않고 거부한다."""

    root = tmp_path / "recorded"
    entry = _write_session(root, "20260806_000002_file")
    dataset = RecordedANCDataset(
        _manifest(tmp_path, [entry]),
        _cfg(
            recorded_lead_mode="timeline",
            d_noise_delay_samples=1602,
            recorded_augment={"enabled": True, "lead_jitter_samples": 0.0},
        ),
        split="train",
    )
    with pytest.raises(ValueError, match="aligned_lag_median_samples"):
        dataset.lead_plan(0)


def test_timeline_lead_mode_derives_the_lead_from_the_session(tmp_path):
    root = tmp_path / "recorded"
    entry = _write_session(
        root,
        "20260806_000003_file",
        timeline={
            "aligned_lag_median_samples": 142.7,
            "aligned_lag_robust_std_samples": 2.2,
        },
    )
    dataset = RecordedANCDataset(
        _manifest(tmp_path, [entry]),
        _cfg(recorded_lead_mode="timeline", d_noise_delay_samples=1602),
        split="train",
    )
    plan = dataset.lead_plan(0)
    assert plan.mode == "timeline"
    assert plan.lead_samples == 1602 + 247 + 116 - 143  # = 1822
    assert plan.recorded_delay_samples == pytest.approx(142.7)
    assert plan.jitter_sigma_samples == 0.0
    assert dataset.total_advance_samples == 1_965


def test_default_lead_mode_keeps_the_configured_constant(tmp_path):
    root = tmp_path / "recorded"
    entry = _write_session(root, "20260806_000004_file")
    dataset = RecordedANCDataset(_manifest(tmp_path, [entry]), _cfg(), split="train")
    plan = dataset.lead_plan(0)
    assert plan.mode == "constant"
    assert plan.lead_samples == 116


# ------------------------------------------------------------------ 증강 선형성
def test_common_eq_is_applied_to_both_channels_so_the_plant_survives():
    """``H·(P·x) = P·(H·x)`` — 공통 EQ 는 플랜트와 교환 가능해야 한다.

    한쪽만 필터링하면 플랜트가 바뀐다. 그게 '유효한 증강' 과 '오염' 을 가르는 선이다.
    """

    rng = np.random.default_rng(0)
    cfg = RecordedAugmentConfig(enabled=True)
    kernel = common_eq_kernel(rng, cfg, FS)
    assert kernel is not None

    x = rng.standard_normal(20000).astype(np.float32)
    d = np.convolve(x, PLANT, mode="same").astype(np.float32)
    x_eq = apply_same_fir(x, kernel)
    d_eq = apply_same_fir(d, kernel)
    core = slice(2000, 18000)
    predicted = np.convolve(x_eq, PLANT, mode="same")
    error = np.linalg.norm(d_eq[core] - predicted[core]) / np.linalg.norm(d_eq[core])
    assert error < 1e-5


def test_augmentation_preserves_the_plant_relation_end_to_end(tmp_path):
    """레벨·극성·EQ 를 다 걸고도 ``d = PLANT ∗ x_ref`` 가 유지돼야 한다.

    ``_augment`` 는 입력에만 마이크 잡음을 넣으므로, 잡음을 끈 설정에서 관계를 잰다.
    """

    root = tmp_path / "recorded"
    entry = _write_session(root, "20260806_000005_file")
    cfg = _cfg(
        digital_reference_lead_samples=0,
        recorded_augment={
            "enabled": True,
            "mic_noise_snr_db": (200.0, 200.0),   # 사실상 잡음 없음
        },
    )
    dataset = RecordedANCDataset(_manifest(tmp_path, [entry]), cfg, split="train")
    rng = np.random.default_rng(11)
    err, _, source = dataset._session(0)
    start = 5000
    segment = dataset.segment
    x_ref = source[start : start + segment].copy()
    d = err[start : start + segment].copy()

    x_aug, err_aug, d_aug = dataset._augment(x_ref, d, rng)
    core = slice(1000, segment - 1000)
    predicted = np.convolve(x_aug, PLANT, mode="same")
    error = np.linalg.norm(d_aug[core] - predicted[core]) / np.linalg.norm(d_aug[core])
    assert error < 0.02
    # 잡음은 입력에만 — d(타깃)는 깨끗해야 한다.
    assert np.allclose(err_aug, d_aug, atol=1e-3)


def test_mic_noise_goes_to_the_input_only_never_to_the_target(tmp_path):
    """negative fixture 성격: 타깃에 잡음이 들어가면 '배울 수 없는 것' 을 요구하게 된다."""

    root = tmp_path / "recorded"
    entry = _write_session(root, "20260806_000006_file")
    cfg = _cfg(
        recorded_augment={"enabled": True, "mic_noise_snr_db": (6.0, 6.0), "eq_tilt_db": 0.0,
                          "eq_band_db": 0.0, "level_db_range": (0.0, 0.0),
                          "polarity_flip": False},
    )
    dataset = RecordedANCDataset(_manifest(tmp_path, [entry]), cfg, split="train")
    err, _, source = dataset._session(0)
    segment = dataset.segment
    x_ref = source[:segment].copy()
    d = err[:segment].copy()
    x_aug, err_aug, d_aug = dataset._augment(x_ref, d, np.random.default_rng(2))
    assert np.allclose(d_aug, d)                      # 타깃은 원본 그대로
    assert not np.allclose(err_aug, d_aug)            # 입력에는 잡음이 들어갔다
    assert not np.allclose(x_aug, x_ref)


def test_augmentation_is_off_by_default(tmp_path):
    """기본이 off 여야 기존 학습/테스트가 조용히 바뀌지 않는다."""

    root = tmp_path / "recorded"
    entry = _write_session(root, "20260806_000007_file")
    dataset = RecordedANCDataset(_manifest(tmp_path, [entry]), _cfg(), split="train")
    assert dataset.augment.enabled is False
    batch = next(iter(dataset))
    assert batch["x"].shape == (2, dataset.segment)
    assert batch["d"].shape == (1, dataset.segment)


@pytest.mark.parametrize(
    "payload",
    [
        {"level_db_range": (6.0, -12.0)},
        {"mic_noise_snr_db": (0.0, 30.0)},
        {"mix_probability": 1.5},
        {"mix_weight_range": (0.8, 0.2)},
        {"lead_jitter_samples": -1.0},
    ],
)
def test_invalid_augment_config_is_rejected_at_construction(payload):
    with pytest.raises(ValueError):
        RecordedAugmentConfig(**payload)


def test_session_cache_is_bounded(tmp_path):
    """워커마다 전 세션을 메모리에 올리면 Jetson 에서 OOM 이다 (실측 ~3.4 GB/worker)."""

    root = tmp_path / "recorded"
    entries = [
        _write_session(root, f"20260806_00001{i}_file", seconds=1.0, seed=i) for i in range(5)
    ]
    dataset = RecordedANCDataset(
        _manifest(tmp_path, entries), _cfg(recorded_session_cache=2), split="train"
    )
    for index in range(5):
        dataset._session(index)
    assert len(dataset._cache) == 2


def test_family_group_session_sampler_balances_each_hierarchy_level(tmp_path):
    """세션 수가 9:1이어도 family가 9:1로 뽑히지 않는다."""

    entries: list[dict] = []
    for index in range(9):
        entries.append(
            {
                "path": str(tmp_path / f"machine-{index}"),
                "split": "train",
                "session_id": f"machine-{index}",
                "group_id": "machine-large" if index < 8 else "machine-small",
                "source_family": "machine",
            }
        )
    entries.append(
        {
            "path": str(tmp_path / "speech-0"),
            "split": "train",
            "session_id": "speech-0",
            "group_id": "speech-only",
            "source_family": "speech",
        }
    )
    dataset = RecordedANCDataset(
        _manifest(tmp_path, entries),
        _cfg(recorded_sampling="family_group_session_balanced"),
        split="train",
        seed=20260803,
    )
    rng = dataset._worker_rng(0)
    counts = {"machine": 0, "speech": 0}
    machine_groups = {"machine-large": 0, "machine-small": 0}
    for _ in range(20_000):
        entry = dataset.entries[dataset._sample_session_index(rng)]
        counts[entry["source_family"]] += 1
        if entry["source_family"] == "machine":
            machine_groups[entry["group_id"]] += 1
    assert counts["machine"] / sum(counts.values()) == pytest.approx(0.5, abs=0.02)
    assert machine_groups["machine-large"] / sum(machine_groups.values()) == pytest.approx(
        0.5, abs=0.02
    )


def test_balanced_sampler_requires_family_and_group_and_is_worker_deterministic(tmp_path):
    entry = {
        "path": str(tmp_path / "session"),
        "split": "train",
        "session_id": "session",
        "group_id": "group",
        "source_family": "speech",
    }
    manifest = _manifest(tmp_path, [entry])
    cfg = _cfg(recorded_sampling="family_group_session_balanced")
    left = RecordedANCDataset(manifest, cfg, split="train", seed=77)
    right = RecordedANCDataset(manifest, cfg, split="train", seed=77)
    for worker_id in (0, 1, 3):
        left_rng = left._worker_rng(worker_id)
        right_rng = right._worker_rng(worker_id)
        assert [left._sample_session_index(left_rng) for _ in range(20)] == [
            right._sample_session_index(right_rng) for _ in range(20)
        ]

    broken = dict(entry)
    broken.pop("group_id")
    (tmp_path / "broken").mkdir()
    with pytest.raises(ValueError, match="source_family/group_id"):
        RecordedANCDataset(
            _manifest(tmp_path / "broken", [broken]), cfg, split="train"
        )


def test_lineage_sampler_rejects_legacy_group_id_and_accepts_regrouped_manifest(tmp_path):
    legacy = {
        "path": str(tmp_path / "session"),
        "split": "train",
        "session_id": "session",
        "group_id": "original-pool-group",
        "source_family": "speech",
    }
    cfg = _cfg(recorded_sampling="family_lineage_session_balanced")
    with pytest.raises(ValueError, match="source_pool_group_id"):
        RecordedANCDataset(_manifest(tmp_path, [legacy]), cfg, split="train")

    regrouped = {
        **legacy,
        "source_pool_group_id": "original-pool-group",
        "group_id": "speech-lineage-component-0001",
    }
    accepted_dir = tmp_path / "accepted"
    accepted_dir.mkdir()
    dataset = RecordedANCDataset(
        _manifest(accepted_dir, [regrouped]), cfg, split="train"
    )
    assert dataset.sampling_mode == "family_lineage_session_balanced"


def test_shipped_recorded_augmentation_contract_is_safe_first_stage():
    data = load_yaml(REPO_ROOT / "configs/data_sim.yaml")
    assert data["recorded_sampling"] == "family_lineage_session_balanced"
    augment = RecordedAugmentConfig.from_data_config(data)
    assert augment.enabled
    assert augment.polarity_flip
    assert augment.eq_tilt_db > 0.0 and augment.eq_band_db > 0.0
    assert augment.mic_noise_snr_db == (12.0, 40.0)
    assert augment.mix_probability == 0.0
    assert augment.lead_jitter_samples == 0.0
