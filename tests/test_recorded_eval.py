import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from deep_anc.data.manifest import write_manifest
from deep_anc.dsp.secondary_path import DifferentiableSecondaryPath, SecondaryPathData
from deep_anc.eval.recorded import (
    RecordedEvalContext,
    RecordedSegment,
    assert_comparable_metrics,
    cluster_bootstrap_ci,
    deterministic_segment_starts,
    evaluate_recorded_segments,
    iter_recorded_segments,
    load_and_audit_recorded_manifest,
    load_recorded_eval_context,
    resolve_warmup_samples,
    validate_resolved_checkpoint,
    write_recorded_metrics,
)


FS = 8_000


def _path_data(path: Path, *, delay: int) -> None:
    np.savez_compressed(
        path,
        fir=np.asarray([1.0], dtype=np.float32),
        delay_samples=np.asarray(delay, dtype=np.int64),
        sample_rate=np.asarray(FS, dtype=np.int64),
        fit_improvement_db=np.asarray(20.0),
        coherence_median=np.asarray(0.99),
        excitation_band_hz=np.asarray([100.0, 1_000.0]),
    )


def _session(
    root: Path,
    name: str,
    *,
    group: str,
    family: str,
    samples: int = 64,
) -> Path:
    path = root / name
    path.mkdir()
    source = np.arange(samples, dtype=np.float32) / samples
    mics = np.stack([source, np.zeros_like(source)], axis=1)
    sf.write(path / "mics.wav", mics, FS, subtype="FLOAT")
    sf.write(path / "source.wav", source, FS, subtype="FLOAT")
    (path / "session.json").write_text(
        json.dumps(
            {
                "group_id": group,
                "source_family": family,
                "sample_rate": FS,
            }
        ),
        encoding="utf-8",
    )
    return path


def _entry(path: Path, split: str, session: str, group: str, family: str) -> dict:
    return {
        "path": str(path),
        "split": split,
        "duration_s": 1.0,
        "session_id": session,
        "group_id": group,
        "source_family": family,
    }


def _secondary(delay: int = 0) -> SecondaryPathData:
    return SecondaryPathData(
        fir=np.asarray([1.0], dtype=np.float32),
        delay_samples=delay,
        sample_rate=FS,
        fit_improvement_db=20.0,
        coherence_median=0.99,
        excitation_band_hz=(100.0, 1_000.0),
        source_path="test_secondary.npz",
    )


class AdvanceCancelModel(torch.nn.Module):
    def forward(self, x):
        return -x[:, :1]


def test_resolved_checkpoint_rejects_surrogate_and_lead_alias_mismatch():
    cfg = {
        "model": {},
        "data": {
            "reference_mode": "digital",
            "digital_primary_path_mode": "secondary_surrogate",
            "digital_reference_lead_samples": 3,
        },
        "duct": {},
        "physics_status": "secondary_surrogate_representation_pretrain",
        "trusted_band_hz": [100, 1_000],
        "digital_reference_lead_samples": 3,
    }
    state = {"cfg": cfg, "model": {}}

    with pytest.raises(ValueError, match="measured_primary_path"):
        validate_resolved_checkpoint(state)
    assert validate_resolved_checkpoint(state, allow_surrogate=True)[1] == 3

    cfg["digital_reference_lead_samples"] = 4
    with pytest.raises(ValueError, match="alias 불일치"):
        validate_resolved_checkpoint(state, allow_surrogate=True)


def test_context_rejects_lead_inconsistent_with_measured_primary(tmp_path):
    secondary = tmp_path / "secondary.npz"
    primary = tmp_path / "primary.npz"
    checkpoint = tmp_path / "model.pt"
    _path_data(secondary, delay=3)
    _path_data(primary, delay=4)
    cfg = {
        "model": {},
        "data": {
            "sample_rate": FS,
            "reference_mode": "digital",
            "digital_primary_path_mode": "measured",
            # expected = (S 3 + handoff 2) - P 4 = 1
            "digital_reference_lead_samples": 2,
        },
        "duct": {
            "secondary_path": {
                "npz": str(secondary),
                "handoff_extra_samples": 2,
            },
            "digital_reference": {
                "primary_path_npz": str(primary),
                "d_noise_delay_samples": 4,
            },
            "acoustics": {"realistic_target_band_hz": [100, 1_000]},
        },
        "physics_status": "measured_primary_path",
        "trusted_band_hz": [100, 1_000],
        "digital_reference_lead_samples": 2,
    }
    torch.save({"cfg": cfg, "model": {}}, checkpoint)

    with pytest.raises(ValueError, match="expected=1"):
        load_recorded_eval_context(checkpoint, device="cpu")


def test_manifest_audit_rejects_group_leakage_and_duplicate_paths(tmp_path):
    session = _session(tmp_path, "s1", group="g1", family="speech")
    leaking = tmp_path / "leaking.jsonl"
    leaking.write_text(
        "\n".join(
            json.dumps(entry)
            for entry in (
                _entry(session, "train", "s1", "g1", "speech"),
                _entry(session, "test", "s2", "g1", "speech"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="여러 split"):
        load_and_audit_recorded_manifest(leaking, "test")

    duplicate = tmp_path / "duplicate.jsonl"
    write_manifest(
        [
            _entry(session, "test", "s1", "g1", "speech"),
            _entry(session, "test", "s1", "g1", "speech"),
        ],
        duplicate,
    )
    with pytest.raises(ValueError, match="중복 session path"):
        load_and_audit_recorded_manifest(duplicate, "test")


def test_recorded_segments_are_finite_deterministic_and_apply_lead(tmp_path):
    session = _session(tmp_path, "s1", group="g1", family="speech")
    entry = _entry(session, "test", "s1", "g1", "speech")
    data = {
        "sample_rate": FS,
        "segment_seconds": 8 / FS,
        "reference_mode": "digital",
        "digital_reference_lead_samples": 2,
        "closed_loop": {"feedback_delay_samples": [1, 1]},
    }

    first = list(
        iter_recorded_segments(
            [entry],
            data,
            model_hop=4,
            max_segments_per_session=2,
            edge_trim_seconds=0.0,
        )
    )
    second = list(
        iter_recorded_segments(
            [entry],
            data,
            model_hop=4,
            max_segments_per_session=2,
            edge_trim_seconds=0.0,
        )
    )

    assert [segment.start_sample for segment in first] == [0, 48]
    assert [segment.start_sample for segment in second] == [0, 48]
    np.testing.assert_array_equal(first[0].x, second[0].x)
    np.testing.assert_allclose(first[0].x[0], first[0].d + 2 / 64)
    np.testing.assert_allclose(first[0].x[1], np.r_[0.0, first[0].d[:-1]])
    assert deterministic_segment_starts(62, 8, 2) == [0, 48]
    assert deterministic_segment_starts(62, 8, 2, edge_trim_samples=8) == [8, 40]
    assert resolve_warmup_samples(data, FS) == 2_000


def test_recorded_segments_default_edge_trim_skips_session_boundaries(tmp_path):
    session = _session(
        tmp_path, "long", group="g1", family="environment", samples=5_000
    )
    data = {
        "sample_rate": FS,
        "segment_seconds": 8 / FS,
        "reference_mode": "digital",
        "digital_reference_lead_samples": 0,
        "closed_loop": {"feedback_delay_samples": [1, 1]},
    }

    segments = list(
        iter_recorded_segments(
            [_entry(session, "test", "long", "g1", "environment")],
            data,
            model_hop=4,
            max_segments_per_session=1,
        )
    )

    # 기본 0.25 s = 2,000 samples를 session 시작과 끝에서 모두 제외한다.
    assert [segment.start_sample for segment in segments] == [2_000]


def test_evaluation_applies_secondary_before_warmup_cut():
    samples = 256
    time = np.arange(samples, dtype=np.float32) / FS
    d = np.cos(2 * np.pi * 500 * time).astype(np.float32)
    # S(z)의 1 sample handoff 뒤 y(t-1)=-d(t)가 되도록 reference를 선행시킨다.
    reference = np.r_[d[1:], 0.0].astype(np.float32)
    segment = RecordedSegment(
        x=np.stack([reference, np.zeros_like(reference)]),
        d=d,
        session_id="s1",
        group_id="g1",
        source_family="speech",
        start_sample=0,
    )
    plant = DifferentiableSecondaryPath(
        _secondary(delay=0), handoff_extra_samples=1
    )

    result = evaluate_recorded_segments(
        AdvanceCancelModel(),
        plant,
        [segment],
        sample_rate=FS,
        trusted_band_hz=(100.0, 1_000.0),
        octave_bands_hz=[500.0],
        batch_size=1,
        warmup_samples=1,
    )

    assert result["fullband"]["mean_db"] < -100.0
    assert result["trusted"]["mean_db"] < -100.0
    assert result["source_rows"][0]["source_family"] == "speech"
    assert result["octave_rows"][0]["attenuation_mean_db"] > 100.0

    # warmup을 쓰지 않으면 S(z) 지연 때문에 첫 샘플의 e=d가 남는다. 위 결과가
    # 이보다 훨씬 좋아야 플랜트를 먼저 적용하고 나중에 자른 순서가 보장된다.
    untrimmed = evaluate_recorded_segments(
        AdvanceCancelModel(),
        plant,
        [segment],
        sample_rate=FS,
        trusted_band_hz=(100.0, 1_000.0),
        octave_bands_hz=[500.0],
        batch_size=1,
        warmup_samples=0,
    )
    assert untrimmed["fullband"]["mean_db"] > -30.0




# ======================================================================================
# G4 판정 — 절대목표 1·2 정렬 + 통계적 검정력
#
# 이 절의 모든 게이트는 **negative fixture 와 짝**을 이룬다. 게이트가 통과하는 것을
# 한 번 본 것은 게이트가 작동한다는 증거가 아니다 — 실측으로 확인된 사실이다:
# 게이트 9개가 전부 PASS 였는데 전부 무용지물이었다.
# ======================================================================================
GROUPS_PER_FAMILY = 4
"""계열당 그룹 수. MIN_GROUPS_PER_FAMILY 와 같아야 CI 가 정의된다.

이 상수를 3 으로 내리면 아래 PASS 테스트가 INCONCLUSIVE 로 바뀐다 — 즉 검정력
게이트가 살아 있다는 것을 픽스처 자체가 증명한다.
"""


def _tone(samples: int, hz: float, *, amplitude: float = 1.0) -> np.ndarray:
    time = np.arange(samples, dtype=np.float32) / FS
    return (amplitude * np.sin(2 * np.pi * hz * time)).astype(np.float32)


def _powered_segments(
    samples: int = 256,
    *,
    families: tuple[str, ...] = ("speech", "music"),
    groups_per_family: int = GROUPS_PER_FAMILY,
    gain: float = 1.0,
) -> list[RecordedSegment]:
    """계열당 그룹이 충분한 세그먼트 집합.

    ``AdvanceCancelModel`` 이 ``-x[:, :1]`` 을 내므로 ``x`` 채널 0 에 ``d`` 를 그대로
    넣으면 상쇄가 일어난다. 그룹마다 진폭을 조금씩 흔들어 그룹 간 분산이 0 이 아니게
    한다 — 분산이 정확히 0 이면 부트스트랩 CI 가 한 점으로 붕괴해 검사가 무의미해진다.
    """

    segments: list[RecordedSegment] = []
    for family_index, family in enumerate(families):
        for group_index in range(groups_per_family):
            d = _tone(samples, 500.0, amplitude=1.0 + 0.05 * group_index)
            segments.append(
                RecordedSegment(
                    x=np.stack([gain * d, np.zeros_like(d)]),
                    d=d,
                    session_id=f"{family}_session_{group_index}",
                    group_id=f"{family}_group_{group_index}",
                    source_family=family,
                    start_sample=family_index,
                )
            )
    return segments


def _context(secondary: SecondaryPathData, plant: DifferentiableSecondaryPath, model):
    return RecordedEvalContext(
        model=model,
        plant=plant,
        cfg={"model": {"hop": 4}, "data": {}, "duct": {}},
        device=torch.device("cpu"),
        sample_rate=FS,
        trusted_band_hz=(100.0, 1_000.0),
        physics_status="measured_primary_path",
        reference_mode="digital",
        digital_reference_lead_samples=1,
        expected_digital_reference_lead_samples=1,
        primary_delay_samples=0,
        secondary_path=secondary,
        secondary_handoff_samples=0,
    )


def _write(tmp_path, result, context, name="out"):
    return write_recorded_metrics(
        result,
        tmp_path / name,
        checkpoint=tmp_path / "best.pt",
        manifest=tmp_path / "recorded.jsonl",
        split="test",
        context=context,
        feedback_delay_samples=1,
        allow_surrogate=False,
        edge_trim_samples=2_000,
        warmup_samples=0,
    )


def _evaluate(segments, *, octave_bands_hz=(500.0, 2_000.0), model=None, secondary=None):
    secondary = secondary or _secondary()
    plant = DifferentiableSecondaryPath(secondary)
    model = model or AdvanceCancelModel()
    result = evaluate_recorded_segments(
        model,
        plant,
        segments,
        sample_rate=FS,
        trusted_band_hz=(100.0, 1_000.0),
        octave_bands_hz=list(octave_bands_hz),
        batch_size=4,
        warmup_samples=0,
    )
    return result, _context(secondary, plant, model)


def test_metrics_markdown_and_npz_include_source_octave_and_worst10(tmp_path):
    """통계적 근거가 갖춰진 정상 결과는 PASS 해야 한다 (오기각 방지)."""

    result, context = _evaluate(_powered_segments())
    markdown, archive = _write(tmp_path, result, context)

    report = markdown.read_text(encoding="utf-8")
    assert "최악 10%" in report
    assert "speech" in report and "music" in report
    assert "500 Hz" in report
    assert "G4 종합: PASS" in report
    assert "S(z) 적용 후 절단" in report
    assert "계열별 그룹 부트스트랩 신뢰구간" in report
    with np.load(archive, allow_pickle=False) as saved:
        assert set(saved["source_family"].tolist()) == {"music", "speech"}
        assert saved["octave_center_hz"].tolist() == [500.0, 2_000.0]
        assert str(saved["physics_status"]) == "measured_primary_path"
        assert bool(saved["g4_pass"])
        assert str(saved["g4_verdict"]) == "PASS"
        assert bool(saved["g4_do_no_harm_pass"])
        assert bool(saved["g4_power_pass"])
        assert bool(saved["g4_ci_pass"])
        assert int(saved["edge_trim_samples"]) == 2_000
        assert int(saved["warmup_samples"]) == 0
        # CI 가 실제로 정의됐고 상단이 0 아래여야 한다.
        assert np.all(np.isfinite(saved["source_trusted_ci_hi_db"]))
        assert np.all(saved["source_trusted_ci_hi_db"] < 0.0)
        # 플랜트 지문이 남아야 이후 전후 비교가 가능하다.
        assert str(saved["plant_fingerprint_digest"])
        assert json.loads(str(saved["plant_fingerprint_json"]))["physics_status"] == (
            "measured_primary_path"
        )


def test_g4_rejects_out_of_band_amplifier(tmp_path):
    """신뢰 대역은 줄이면서 **대역 밖을 크게 키우는** 모델은 FAIL 해야 한다.

    실측 반증(results/session_20260804_0939): tone300 이 fullband **+5.95 dB** 로
    판정 기준을 만족하면서 8 kHz 를 **−21.56 dB** 증폭했다. fullband 평균 NMSE 는
    `d` 에 에너지가 없는 대역의 증폭을 원리적으로 잡지 못한다. 이 테스트가 그
    구멍이 막혔는지 확인한다.
    """

    samples = 512

    class OutOfBandAmplifier(torch.nn.Module):
        """제어 대역(500Hz)은 상쇄하고 대역 밖(2kHz)에는 강한 톤을 더한다."""

        def forward(self, x):
            batch, _, length = x.shape
            time = torch.arange(length, dtype=x.dtype, device=x.device) / FS
            harm = 8.0 * torch.sin(2 * torch.pi * 2_000.0 * time)
            return -x[:, :1] + harm.view(1, 1, -1).expand(batch, 1, length)

    result, context = _evaluate(
        _powered_segments(samples), model=OutOfBandAmplifier()
    )
    markdown, archive = _write(tmp_path, result, context)

    report = markdown.read_text(encoding="utf-8")
    assert "G4 종합: FAIL" in report
    assert "대역 밖 증폭" in report
    with np.load(archive, allow_pickle=False) as saved:
        assert not bool(saved["g4_pass"])
        assert str(saved["g4_verdict"]) == "FAIL"
        assert not bool(saved["g4_do_no_harm_pass"])
        # 어느 옥타브가 문제였는지 기계 판독 가능해야 한다.
        assert float(saved["g4_worst_octave_center_hz"]) == 2_000.0
        assert float(saved["g4_worst_octave_worst10_db"]) < -1.0


def test_g4_is_inconclusive_when_a_family_has_too_few_groups(tmp_path):
    """그룹이 1–2개인 계열이 있으면 **PASS 가 아니라 판정 불가**여야 한다.

    실측 상태: val machine 1그룹, test environment 1그룹, test machine 1그룹.
    클러스터가 1개면 cluster bootstrap CI 가 수학적으로 정의되지 않는데, 그 상태를
    PASS 로 흘려보내면 게이트가 있는 것이 없는 것보다 나쁘다.
    """

    segments = _powered_segments(families=("speech",), groups_per_family=4)
    segments += _powered_segments(families=("machine",), groups_per_family=1)
    result, context = _evaluate(segments)
    markdown, archive = _write(tmp_path, result, context)

    report = markdown.read_text(encoding="utf-8")
    assert "G4 종합: INCONCLUSIVE" in report
    assert "계열당 그룹 부족" in report
    assert "machine=1" in report
    with np.load(archive, allow_pickle=False) as saved:
        # 판정 불가는 통과가 아니다.
        assert not bool(saved["g4_pass"])
        assert str(saved["g4_verdict"]) == "INCONCLUSIVE"
        assert not bool(saved["g4_power_pass"])
        assert saved["g4_underpowered_families"].tolist() == ["machine"]
        # 그룹이 1개인 계열의 CI 는 지어내지 않고 nan 으로 남아야 한다.
        families = saved["source_family"].tolist()
        machine_index = families.index("machine")
        assert not np.isfinite(saved["source_trusted_ci_hi_db"][machine_index])


def test_g4_reports_demonstrated_harm_as_fail_not_inconclusive(tmp_path):
    """증명된 악화는 표본이 부족하더라도 FAIL 이다 — 두 상태를 섞지 않는다."""

    class Amplifier(torch.nn.Module):
        def forward(self, x):
            return x[:, :1]

    segments = _powered_segments(families=("speech",), groups_per_family=1)
    result, context = _evaluate(segments, model=Amplifier())
    markdown, _ = _write(tmp_path, result, context)

    report = markdown.read_text(encoding="utf-8")
    assert "G4 종합: FAIL" in report
    assert "INCONCLUSIVE" not in report.split("G4 종합")[1].split("\n")[0]


def test_metrics_comparison_rejects_different_plants(tmp_path):
    """서로 다른 플랜트에서 나온 두 metrics 의 비교는 거부돼야 한다 (결함 5).

    2026-08-04 사고: 전 = S 지연 1342 / lead 109 / surrogate, 후 = 1465 / 113 /
    measured 를 비교해 "1.30 dB 개선" 이라고 적었다. 서로 다른 물리다.
    """

    segments = _powered_segments()

    before_secondary = _secondary(delay=1_342)
    result_a, context_a = _evaluate(segments, secondary=before_secondary)
    _, archive_a = _write(tmp_path, result_a, context_a, name="before")

    after_secondary = _secondary(delay=1_465)
    result_b, context_b = _evaluate(segments, secondary=after_secondary)
    _, archive_b = _write(tmp_path, result_b, context_b, name="after")

    with np.load(archive_a, allow_pickle=False) as before, np.load(
        archive_b, allow_pickle=False
    ) as after:
        with pytest.raises(ValueError, match="서로 다른 플랜트"):
            assert_comparable_metrics(before, after)
        # 같은 플랜트끼리는 통과해야 한다 (오기각 방지).
        assert_comparable_metrics(before, before)


def test_metrics_comparison_refuses_artifacts_without_a_fingerprint(tmp_path):
    """지문이 없는 구버전 산출물은 "비교 가능"으로 간주하지 않는다 (실패 폐쇄)."""

    legacy = tmp_path / "legacy.npz"
    np.savez_compressed(legacy, g4_pass=np.asarray(True))
    result, context = _evaluate(_powered_segments())
    _, archive = _write(tmp_path, result, context)

    with np.load(legacy, allow_pickle=False) as before, np.load(
        archive, allow_pickle=False
    ) as after:
        with pytest.raises(ValueError, match="plant_fingerprint_json"):
            assert_comparable_metrics(before, after)


def test_cluster_bootstrap_ci_is_undefined_below_the_group_floor():
    """클러스터가 부족하면 CI 를 **지어내지 않는다**."""

    values = np.asarray([-1.0, -1.2, -0.8, -1.1], dtype=np.float64)
    single = np.asarray(["g0", "g0", "g0", "g0"])
    lo, hi, n_groups = cluster_bootstrap_ci(values, single)
    assert n_groups == 1
    assert not np.isfinite(lo) and not np.isfinite(hi)

    four = np.asarray(["g0", "g1", "g2", "g3"])
    lo, hi, n_groups = cluster_bootstrap_ci(values, four)
    assert n_groups == 4
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi < 0.0


def test_cluster_bootstrap_ci_is_wider_than_naive_segment_resampling():
    """그룹 내 세그먼트를 독립으로 세면 CI 가 실제보다 좁아진다는 것을 못 박는다.

    이것이 D3 의 핵심이다 — 같은 음원에서 잘라낸 조각을 독립 표본으로 착각하면
    "개선했다"는 결론이 표본 수만큼 손쉬워진다.
    """

    rng = np.random.default_rng(3)
    # 그룹 4개, 그룹마다 25 세그먼트. 그룹 간 편차가 그룹 내 편차보다 크다.
    group_means = np.asarray([-2.0, -0.5, -1.5, 0.5])
    values = np.concatenate(
        [mean + 0.05 * rng.standard_normal(25) for mean in group_means]
    )
    groups = np.repeat([f"g{index}" for index in range(4)], 25)

    lo, hi, _ = cluster_bootstrap_ci(values, groups)
    # 세그먼트를 전부 독립으로 본 순진한 CI
    naive_half_width = 1.96 * float(values.std(ddof=1)) / np.sqrt(values.size)
    assert (hi - lo) > 4.0 * naive_half_width
