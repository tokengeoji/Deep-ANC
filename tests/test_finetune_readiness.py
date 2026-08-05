"""실측 fine-tune 진입/완료 실패-폐쇄 게이트 회귀 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from deep_anc.config import REPO_ROOT, load_yaml
from deep_anc.data.manifest import write_manifest
from deep_anc.train.finetune_readiness import (
    audit_finetune_completion,
    audit_finetune_readiness,
    audit_official_path_model,
    require_finetune_readiness,
    sha256_file,
)


FS = 8_000
FAMILIES = ("speech", "music", "environment", "machine")


def _official_path(
    path: Path,
    *,
    channel: str,
    delay: int,
    consistency: float = 0.97,
    amplitude: float = 0.005,
    method: str = "ess",
    interleaved: dict | None = None,
) -> None:
    extra: dict = {}
    if method == "interleaved_multitone":
        defaults = {
            "capture_id": np.asarray("cap-1"),
            "interleave_guard_bins": np.asarray(1, dtype=np.int64),
            "analysis_period_seconds": np.asarray(1.0),
            "tone_count": np.asarray(771, dtype=np.int64),
            "tone_snr_median_db": np.asarray(24.0),
            "tone_snr_min_db": np.asarray(14.0),
            "consistency_band_hz": np.asarray([100.0, 1_000.0]),
            # 최악 부대역 게이트가 판정할 배열. 총계 하나로는 약한 대역을 숨길 수
            # 있으므로 게이트가 요구 대역 안 모든 부대역을 따로 본다.
            "band_consistency": np.asarray([0.99, 0.98, 0.97]),
            "band_consistency_hz": np.asarray(
                [[100.0, 300.0], [300.0, 600.0], [600.0, 1_000.0]]
            ),
        }
        extra = defaults
    arrays = {
        "fir": np.asarray([0.5, -0.1, 0.02], dtype=np.float32),
        "delay_samples": np.asarray(delay, dtype=np.int64),
        "sample_rate": np.asarray(FS, dtype=np.int64),
        "fit_improvement_db": np.asarray(np.nan),
        "coherence_median": np.asarray(consistency),
        "excitation_band_hz": np.asarray([100.0, 1_000.0]),
        "calibration_block_size": np.asarray(256, dtype=np.int64),
        "calibration_latency": np.asarray("high"),
        "output_channel": np.asarray(channel),
        "method": np.asarray(method),
        "repeats": np.asarray(3, dtype=np.int64),
        "amplitude": np.asarray(amplitude),
        "xrun_count": np.asarray(0, dtype=np.int64),
        "delay_spread_samples": np.asarray(1, dtype=np.int64),
        "max_delay_jitter_samples": np.asarray(8, dtype=np.int64),
        **extra,
    }
    # ``interleaved`` 는 interleaved 전용 필드뿐 아니라 공통 필드도 덮어쓸 수 있어야
    # 한다 — 게이트가 "아티팩트가 신고한 허용치"를 믿지 않는지 시험하려면 필요하다.
    arrays.update(
        {key: np.asarray(value) for key, value in (interleaved or {}).items()}
    )
    np.savez(path, **arrays)


def _checkpoint(
    path: Path,
    *,
    cfg: dict,
    step: int,
    best_metric: float = -3.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {"weight": torch.ones(1)},
            "cfg": cfg,
            "step": step,
            "best_metric": best_metric,
        },
        path,
    )


def _recorded_manifest(root: Path, *, frames: int = 512) -> Path:
    manifest = root / "manifests" / "recorded.jsonl"
    entries = []
    t = np.arange(frames, dtype=np.float64) / FS
    for family_index, family in enumerate(FAMILIES):
        for split_index, split in enumerate(("train", "val", "test")):
            session_id = f"{family}-{split}"
            session = root / "recorded" / session_id
            session.mkdir(parents=True)
            source = (0.05 * np.sin(2 * np.pi * (250 + family_index * 40) * t)).astype(
                np.float32
            )
            mics = np.stack([0.7 * source, 0.4 * source], axis=1)
            sf.write(session / "mics.wav", mics, FS, subtype="FLOAT")
            sf.write(session / "source.wav", source, FS, subtype="FLOAT")
            group = f"group-{family}-{split_index}"
            (session / "session.json").write_text(
                json.dumps(
                    {
                        "session_id": session_id,
                        "group_id": group,
                        "source_family": family,
                        "sample_rate": FS,
                        "seconds": frames / FS,
                        "program": {"type": "file"},
                    }
                ),
                encoding="utf-8",
            )
            entries.append(
                {
                    "path": str(session),
                    "duration_s": frames / FS,
                    "sample_rate": FS,
                    "channels": 2,
                    "tag": "recorded",
                    "session_id": session_id,
                    "group_id": group,
                    "source_family": family,
                    "split": split,
                }
            )
    write_manifest(entries, manifest)
    return manifest


def _ready_config(tmp_path: Path) -> dict:
    primary = tmp_path / "primary.npz"
    secondary = tmp_path / "secondary.npz"
    _official_path(primary, channel="noise", delay=4)
    _official_path(secondary, channel="cancel", delay=5)
    manifest = _recorded_manifest(tmp_path / "data")
    model_cfg = {"name": "test-model", "hop": 4}
    pretrain_cfg = {
        "model": model_cfg,
        "data": {"digital_reference_lead_samples": 3},
        "digital_reference_lead_samples": 3,
        "physics_status": "secondary_surrogate_representation_pretrain",
        "schedule": {"total_steps": 10},
    }
    init_best = tmp_path / "pretrain" / "ckpt" / "best.pt"
    _checkpoint(init_best, cfg=pretrain_cfg, step=8)
    _checkpoint(init_best.parent / "last.pt", cfg=pretrain_cfg, step=10)

    return {
        "stage": "open_loop",
        "model": model_cfg,
        "data": {
            "sample_rate": FS,
            "segment_seconds": 0.01,
            "reference_mode": "digital",
            "digital_primary_path_mode": "measured",
            "digital_reference_lead_samples": 3,
            "closed_loop": {
                "feedback_delay_samples": [4, 8],
                "warmup_seconds": 0.0,
            },
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
        },
        "require_measured_primary_path": True,
        "require_init_checkpoint": True,
        "require_recorded_manifest": True,
        "init_ckpt": str(init_best),
        "recorded_manifest": str(manifest),
        "recorded_ratio": 0.7,
        "schedule": {"total_steps": 6},
        "ckpt_dir": str(tmp_path / "finetune"),
        "readiness": {
            "required_path_band_hz": [100, 1_000],
            "min_path_consistency": 0.9,
            "required_recorded_ratio": 0.7,
            "min_recorded_sessions": 12,
            "min_recorded_duration_seconds": 0.1,
            "required_source_families": list(FAMILIES),
            "require_completed_init_checkpoint": True,
            "max_init_best_metric_db": 0.0,
        },
    }


def _g4_metrics(
    path: Path,
    *,
    split: str,
    checkpoint: Path,
    manifest: Path,
    source_pass: bool = True,
    worst_source_db: float = -4.0,
    include_source_fields: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": np.asarray(split),
        "physics_status": np.asarray("measured_primary_path"),
        "allow_surrogate": np.asarray(False),
        "checkpoint_sha256": np.asarray(sha256_file(checkpoint)),
        "manifest_sha256": np.asarray(sha256_file(manifest)),
        "g4_trusted_pass": np.asarray(True),
        "g4_fullband_pass": np.asarray(True),
        "g4_pass": np.asarray(bool(source_pass)),
        "source_family": np.asarray(FAMILIES),
        "n_sessions": np.asarray(4, dtype=np.int64),
        "n_segments": np.asarray(16, dtype=np.int64),
    }
    if include_source_fields:
        # 기능 2(모든 소리 제거)는 소스별 최악값 판정이다 — 평균만 담은 옛 형식은
        # 게이트가 거부해야 한다(include_source_fields=False 로 그 회귀를 검사한다).
        payload.update(
            g4_source_pass=np.asarray(bool(source_pass)),
            g4_worst_source_trusted_mean_db=np.asarray(float(worst_source_db)),
            g4_worst_source_trusted_worst10_db=np.asarray(float(worst_source_db) + 1.0),
            g4_worst_source_family=np.asarray("speech"),
        )
    np.savez_compressed(path, **payload)


def test_official_path_gate_rejects_wrong_channel_and_low_consistency(tmp_path):
    wrong = tmp_path / "wrong.npz"
    _official_path(wrong, channel="cancel", delay=4, consistency=0.8)

    try:
        audit_official_path_model(
            wrong,
            expected_output_channel="noise",
            sample_rate=FS,
            required_band_hz=(100, 1_000),
        )
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - 명시적 실패 메시지를 보기 위한 방어
        raise AssertionError("invalid official path가 통과했습니다")

    assert "output_channel" in message
    assert "일관성" in message


def test_readiness_passes_only_with_official_paths_completed_init_and_full_recorded_qa(
    tmp_path,
):
    cfg = _ready_config(tmp_path)
    report = audit_finetune_readiness(cfg)

    assert report["ok"], report
    assert {item["id"] for item in report["checks"]} == {
        "config_fail_closed_flags",
        "measured_primary_mode",
        "recorded_mix_ratio",
        "official_secondary_path",
        "official_primary_path",
        "matched_path_measurement_conditions",
        "path_delay_and_lead",
        "completed_init_checkpoint",
        "recorded_dataset_qa",
    }
    assert require_finetune_readiness(cfg)["ok"]

    cfg["duct"]["digital_reference"]["d_noise_delay_samples"] = 5
    failed = audit_finetune_readiness(cfg)
    assert not failed["ok"]
    assert not next(
        item for item in failed["checks"] if item["id"] == "path_delay_and_lead"
    )["ok"]


def test_readiness_rejects_timing_invalid_or_legacy_path_metadata(tmp_path):
    cfg = _ready_config(tmp_path)
    legacy = tmp_path / "legacy_secondary.npz"
    np.savez(
        legacy,
        fir=np.ones(4, dtype=np.float32),
        delay_samples=5,
        sample_rate=FS,
        coherence_median=0.4,
        excitation_band_hz=np.asarray([150.0, 600.0]),
    )
    cfg["duct"]["secondary_path"]["npz"] = str(legacy)

    report = audit_finetune_readiness(cfg)
    path_check = next(
        item for item in report["checks"] if item["id"] == "official_secondary_path"
    )

    assert not report["ok"]
    assert not path_check["ok"]
    assert "official ESS 품질 메타데이터" in path_check["message"]


def test_completion_requires_same_checkpoint_and_manifest_sha_for_val_and_test(tmp_path):
    cfg = _ready_config(tmp_path)
    run = tmp_path / "finetune"
    best = run / "ckpt" / "best.pt"
    saved_cfg = {
        **cfg,
        "physics_status": "measured_primary_path",
        "digital_reference_lead_samples": 3,
    }
    _checkpoint(best, cfg=saved_cfg, step=4)
    _checkpoint(best.parent / "last.pt", cfg=saved_cfg, step=6)
    manifest = Path(cfg["recorded_manifest"])
    val_metrics = run / "eval_recorded_val" / "metrics.npz"
    test_metrics = run / "eval_recorded_test" / "metrics.npz"
    _g4_metrics(val_metrics, split="val", checkpoint=best, manifest=manifest)
    _g4_metrics(test_metrics, split="test", checkpoint=best, manifest=manifest)

    passed = audit_finetune_completion(
        cfg,
        checkpoint=best,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    assert passed["ok"], passed
    assert passed["fine_tuning_complete"]

    with np.load(test_metrics, allow_pickle=False) as current:
        arrays = {key: current[key] for key in current.files}
    arrays["checkpoint_sha256"] = np.asarray("0" * 64)
    np.savez_compressed(test_metrics, **arrays)
    failed = audit_finetune_completion(
        cfg,
        checkpoint=best,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    assert not failed["ok"]
    assert not failed["fine_tuning_complete"]
    assert "SHA-256" in next(
        item for item in failed["checks"] if item["id"] == "recorded_test_g4"
    )["message"]


def _completion_setup(tmp_path):
    """완료 감사에 필요한 최소 구성. 위 테스트와 같은 뼈대를 재사용한다."""

    cfg = _ready_config(tmp_path)
    run = Path(cfg["ckpt_dir"])
    best = run / "ckpt" / "best.pt"
    saved_cfg = {
        **cfg,
        "physics_status": "measured_primary_path",
        "digital_reference_lead_samples": 3,
    }
    _checkpoint(best, cfg=saved_cfg, step=4)
    _checkpoint(best.parent / "last.pt", cfg=saved_cfg, step=6)
    return cfg, best, Path(cfg["recorded_manifest"]), run


def test_g4_rejects_model_that_amplifies_one_source_family(tmp_path):
    """기능 2 회귀 방어 — 평균이 좋아도 **최악 소스**가 증폭이면 완료가 아니다.

    이 게이트가 없으면 machine −8 dB / speech +6 dB 가 섞여 평균 −1.75 dB 로 통과한다.
    즉 대화를 6 dB 키우는 모델이 `[COMPLETE] G4 PASS` 로 배포 후보가 된다.
    """

    cfg, best, manifest, run = _completion_setup(tmp_path)
    val_metrics = run / "eval_recorded_val" / "metrics.npz"
    test_metrics = run / "eval_recorded_test" / "metrics.npz"
    _g4_metrics(val_metrics, split="val", checkpoint=best, manifest=manifest)
    # test 쪽만 최악 소스가 증폭(+6 dB)인 상황
    _g4_metrics(
        test_metrics, split="test", checkpoint=best, manifest=manifest,
        source_pass=False, worst_source_db=6.0,
    )

    result = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )
    assert not result["ok"]
    assert not result["fine_tuning_complete"]
    message = next(
        item for item in result["checks"] if item["id"] == "recorded_test_g4"
    )["message"]
    assert "기능2" in message and "speech" in message and "+6.00" in message


def test_g4_rejects_metrics_without_worst_source_fields(tmp_path):
    """최악 소스를 판정하지 않던 **구버전 평가기의 산출물**은 통과시키지 않는다.

    필드가 없으면 조용히 통과시키는 것이 가장 위험하다 — 게이트가 있다고 믿으면서
    실제로는 평균만 보게 된다.
    """

    cfg, best, manifest, run = _completion_setup(tmp_path)
    val_metrics = run / "eval_recorded_val" / "metrics.npz"
    test_metrics = run / "eval_recorded_test" / "metrics.npz"
    _g4_metrics(val_metrics, split="val", checkpoint=best, manifest=manifest)
    _g4_metrics(
        test_metrics, split="test", checkpoint=best, manifest=manifest,
        include_source_fields=False,
    )

    result = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )
    assert not result["ok"]
    assert "g4_source_pass" in next(
        item for item in result["checks"] if item["id"] == "recorded_test_g4"
    )["message"]


# ---------------------------------------------------------------------------
# interleaved_multitone — P/S 동시 측정 방식
#
# 이 방식을 허용하는 것은 게이트를 넓히는 것이 아니다. ESS 가 요구하지 않는 항목
# (guard=1, 분석창 길이, 톤 수, 톤 SNR)을 추가로 요구하고, 무엇보다 두 파일이 같은
# capture 에서 나왔는지를 **파일에 박힌 capture_id 로** 확인한다. 진폭·블록·latency 가
# 우연히 같은 서로 다른 두 측정은 여기서 걸린다.
# ---------------------------------------------------------------------------


def _check(report: dict, name: str) -> dict:
    return next(item for item in report["checks"] if item["id"] == name)


def test_interleaved_method_is_accepted_with_full_metadata(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4, method="interleaved_multitone")
    report = audit_official_path_model(
        path, expected_output_channel="noise", sample_rate=FS,
        required_band_hz=(100.0, 1_000.0),
    )
    assert report["method"] == "interleaved_multitone"
    assert report["interleaved"]["capture_id"] == "cap-1"


def test_interleaved_without_extra_metadata_is_rejected(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4)  # ess 필드만
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["method"] = np.asarray("interleaved_multitone")
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="interleaved 측정 메타데이터가 없습니다"):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=FS,
            required_band_hz=(100.0, 1_000.0),
        )


@pytest.mark.parametrize(
    "override, pattern",
    [
        ({"interleave_guard_bins": 2}, "interleave_guard_bins"),
        ({"analysis_period_seconds": 3.7}, "analysis_period_seconds"),
        ({"tone_count": 32}, "tone_count"),
        ({"tone_snr_median_db": 6.0}, "tone_snr_median_db"),
        ({"capture_id": ""}, "capture_id"),
        ({"consistency_band_hz": [300.0, 500.0]}, "일관성 측정 대역"),
    ],
)
def test_interleaved_quality_fields_are_enforced(tmp_path, override, pattern):
    path = tmp_path / "primary.npz"
    _official_path(
        path, channel="noise", delay=4,
        method="interleaved_multitone", interleaved=override,
    )
    with pytest.raises(ValueError, match=pattern):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=FS,
            required_band_hz=(100.0, 1_000.0),
        )


# ---------------------------------------------------------------------------
# 2026-08-05 결함 1 회귀 — 아래 세 게이트가 없으면 33%(형상 기준 50%) 틀린 S(z) 가
# 다시 파인튜닝까지 들어간다. 셋 다 실제로 통과했던 값으로 테스트한다.
# ---------------------------------------------------------------------------


def test_artifact_cannot_declare_its_own_delay_jitter_allowance(tmp_path):
    """허용치를 아티팩트에서 읽으면 게이트가 자기증명이 된다.

    출하본은 delay_spread 32 를 **같은 NPZ 안의** max_delay_jitter 48 과 비교해
    통과했다. 측정 시 --max-delay-jitter-ms 를 키우면 검사가 사라지는 구조였다.
    """

    path = tmp_path / "primary.npz"
    _official_path(
        path, channel="noise", delay=4, method="interleaved_multitone",
        interleaved={"delay_spread_samples": 32, "max_delay_jitter_samples": 48},
    )
    with pytest.raises(ValueError, match="상대 τ spread 32 > 허용 3"):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_missing_sub_band_consistency_is_rejected(tmp_path):
    """최악 부대역을 검증할 수 없는 아티팩트는 official 이 될 수 없다."""

    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4, method="interleaved_multitone")
    with np.load(path) as data:
        arrays = {
            key: data[key] for key in data.files
            if key not in {"band_consistency", "band_consistency_hz"}
        }
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="band_consistency"):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_weak_sub_band_is_rejected_even_when_the_total_passes(tmp_path):
    """총계는 에너지 가중이라 약한 대역을 숨긴다 — 최악값이 판정 기준이다."""

    path = tmp_path / "primary.npz"
    _official_path(
        path, channel="noise", delay=4, consistency=0.99,
        method="interleaved_multitone",
        # 총계 0.99 는 통과하지만 600-1000Hz 부대역만 0.73 이다(출하본 실측값).
        interleaved={"band_consistency": [0.99, 0.99, 0.73]},
    )
    with pytest.raises(ValueError, match="부대역 600-1000Hz 일관성 0.7300"):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_weak_sub_band_outside_the_required_band_is_not_judged(tmp_path):
    """필수 대역 밖은 판정하지 않는다 — 그래야 하한 150Hz 규약이 영구 FAIL 을 안 만든다."""

    path = tmp_path / "primary.npz"
    _official_path(
        path, channel="noise", delay=4, method="interleaved_multitone",
        interleaved={"band_consistency": [0.99, 0.99, 0.40]},
    )
    report = audit_official_path_model(
        path, expected_output_channel="noise", sample_rate=FS,
        required_band_hz=(100.0, 600.0),   # 600-1000Hz 부대역이 요구 대역 밖
    )
    assert report["interleaved"]["band_consistency"][-1] == pytest.approx(0.40)


@pytest.mark.parametrize(
    "params, pattern",
    [
        ({"min_alignment_score": 0.5}, "min_alignment_score=0.5 < 0.95"),
        ({"max_relative_tau_samples": 48.0}, "max_relative_tau_samples=48.0 > 3.0"),
        ({"max_drift_deviation_samples": 25.0}, "max_drift_deviation_samples"),
        ({"min_kept_repeats": 3}, "min_kept_repeats=3 < 8"),
        ({"min_alignment_score": None}, "재분석 파라미터 min_alignment_score 가 없습니다"),
    ],
)
def test_reanalysis_parameter_envelope_is_enforced(tmp_path, params, pattern):
    """재분석 도구가 게이트를 약화한 값으로 푼 결과는 official 이 될 수 없다."""

    envelope = {
        "min_alignment_score": 0.95,
        "max_relative_tau_samples": 3.0,
        "max_drift_deviation_samples": 2.0,
        "min_kept_repeats": 8,
    }
    for key, value in params.items():
        if value is None:
            envelope.pop(key)
        else:
            envelope[key] = value
    path = tmp_path / "primary.npz"
    _official_path(
        path, channel="noise", delay=4, method="interleaved_multitone",
        interleaved={"reanalysis_params_json": json.dumps(envelope, sort_keys=True)},
    )
    with pytest.raises(ValueError, match=pattern):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_reanalysis_inside_the_envelope_is_accepted(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(
        path, channel="noise", delay=4, method="interleaved_multitone",
        interleaved={
            "reanalysis_params_json": json.dumps(
                {
                    "min_alignment_score": 0.95,
                    "max_relative_tau_samples": 3.0,
                    "max_drift_deviation_samples": 2.0,
                    "min_kept_repeats": 8,
                },
                sort_keys=True,
            )
        },
    )
    report = audit_official_path_model(
        path, expected_output_channel="noise", sample_rate=FS,
        required_band_hz=(100.0, 1_000.0),
    )
    assert report["interleaved"]["reanalysis_params"]["min_kept_repeats"] == 8


def test_shipped_official_artifacts_pass_the_new_gates():
    """저장소의 현행 P/S 가 실제로 새 게이트를 통과하는지 — 설정과 아티팩트의 정합."""

    duct = load_yaml(REPO_ROOT / "configs/duct.yaml")
    finetune = load_yaml(REPO_ROOT / "configs/train_finetune.yaml")
    band = tuple(float(v) for v in finetune["readiness"]["required_path_band_hz"])
    reports = {}
    for key, channel in (
        (duct["digital_reference"]["primary_path_npz"], "noise"),
        (duct["secondary_path"]["npz"], "cancel"),
    ):
        reports[channel] = audit_official_path_model(
            REPO_ROOT / key, expected_output_channel=channel,
            sample_rate=48_000, required_band_hz=band,
            min_consistency=float(finetune["readiness"]["min_path_consistency"]),
        )
    # 같은 캡처·같은 반복 집합에서 나왔어야 lead 가 물리량이다.
    assert (
        reports["noise"]["interleaved"]["capture_id"]
        == reports["cancel"]["interleaved"]["capture_id"]
    )
    p_delay = reports["noise"]["delay_samples"]
    s_delay = reports["cancel"]["delay_samples"]
    # P−S 는 이 측정 방식의 유일한 물리 불변량 — 유효 캡처 9건에서 139~141 이다.
    assert 139 <= p_delay - s_delay <= 141
    assert int(duct["digital_reference"]["d_noise_delay_samples"]) == p_delay
    handoff = int(duct["secondary_path"]["handoff_extra_samples"])
    data_sim = load_yaml(REPO_ROOT / "configs/data_sim.yaml")
    assert int(data_sim["digital_reference_lead_samples"]) == (
        s_delay + handoff - p_delay
    )


def test_unknown_method_is_still_rejected(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4, method="white_noise")
    with pytest.raises(ValueError, match="허용 method"):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_interleaved_pair_from_different_captures_fails_matched_conditions(tmp_path):
    """조건 값이 전부 같아도 **다른 capture** 면 통과하면 안 된다.

    다른 재생에서 나왔다면 그 사이의 클록 wander 가 두 경로의 상대 지연에 실린다.
    lead = S_delay + handoff − P_delay 가 바로 그 값이므로, 이걸 놓치면 파인튜닝이
    틀린 lead 로 조용히 진행된다.
    """

    cfg = _ready_config(tmp_path)
    primary = Path(cfg["duct"]["digital_reference"]["primary_path_npz"])
    secondary = Path(cfg["duct"]["secondary_path"]["npz"])
    primary.unlink()
    secondary.unlink()
    _official_path(
        primary, channel="noise", delay=4, method="interleaved_multitone",
        interleaved={"capture_id": "cap-A"},
    )
    _official_path(
        secondary, channel="cancel", delay=5, method="interleaved_multitone",
        interleaved={"capture_id": "cap-B"},
    )
    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    matched = _check(report, "matched_path_measurement_conditions")
    assert not matched["ok"]
    assert "capture_id 불일치" in matched["message"]


def test_interleaved_pair_from_one_capture_passes_matched_conditions(tmp_path):
    cfg = _ready_config(tmp_path)
    primary = Path(cfg["duct"]["digital_reference"]["primary_path_npz"])
    secondary = Path(cfg["duct"]["secondary_path"]["npz"])
    primary.unlink()
    secondary.unlink()
    _official_path(primary, channel="noise", delay=4, method="interleaved_multitone")
    _official_path(secondary, channel="cancel", delay=5, method="interleaved_multitone")
    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    matched = _check(report, "matched_path_measurement_conditions")
    assert matched["ok"], matched["message"]


def test_mixed_methods_are_rejected(tmp_path):
    cfg = _ready_config(tmp_path)
    secondary = Path(cfg["duct"]["secondary_path"]["npz"])
    secondary.unlink()
    _official_path(secondary, channel="cancel", delay=5, method="interleaved_multitone")
    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    matched = _check(report, "matched_path_measurement_conditions")
    assert not matched["ok"]
    assert "측정 방식 불일치" in matched["message"]


def test_init_lead_mismatch_is_rejected_by_default(tmp_path):
    """기본값은 정확히 일치다 — 허용치는 설정에 명시해야만 열린다."""

    cfg = _ready_config(tmp_path)
    cfg["data"]["digital_reference_lead_samples"] = 4      # checkpoint 는 3
    cfg["duct"]["secondary_path"]["handoff_extra_samples"] = 3
    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    check = _check(report, "completed_init_checkpoint")
    assert not check["ok"]
    assert "lead 불일치" in check["message"]


def test_init_lead_mismatch_within_declared_tolerance_passes(tmp_path):
    cfg = _ready_config(tmp_path)
    cfg["data"]["digital_reference_lead_samples"] = 4
    cfg["duct"]["secondary_path"]["handoff_extra_samples"] = 3
    cfg["readiness"]["max_init_lead_mismatch_samples"] = 2
    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    assert _check(report, "completed_init_checkpoint")["ok"]


def test_gross_lead_mismatch_is_rejected_despite_tolerance(tmp_path):
    """허용치는 유계다 — 실물 규모의 사고(lead 113 vs 0)는 통과하면 안 된다.

    픽스처의 lead 는 3 이라 checkpoint 를 0 으로 두면 차이가 3 뿐이다. 허용치 16 을
    넘는 차이를 만들어야 "유계성"을 실제로 검사한다.
    """

    cfg = _ready_config(tmp_path)
    init = Path(cfg["init_ckpt"])
    pretrain_cfg = {
        "model": cfg["model"],
        "data": {"digital_reference_lead_samples": 50},
        "digital_reference_lead_samples": 50,
        "physics_status": "secondary_surrogate_representation_pretrain",
        "schedule": {"total_steps": 10},
    }
    _checkpoint(init, cfg=pretrain_cfg, step=8)
    _checkpoint(init.parent / "last.pt", cfg=pretrain_cfg, step=10)
    cfg["readiness"]["max_init_lead_mismatch_samples"] = 16
    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    check = _check(report, "completed_init_checkpoint")
    assert not check["ok"]
    assert "lead 불일치" in check["message"]


def test_trainer_and_gate_share_one_lead_tolerance():
    """같은 규칙을 두 곳에 구현하면 갈라진다 — 둘 다 같은 설정 키를 읽어야 한다.

    실제로 readiness 만 고쳤다가 trainer 의 별도 검사에서 학습이 막혔다.
    """

    import inspect

    from deep_anc.train import trainer as trainer_module

    source = inspect.getsource(trainer_module.Trainer.__init__)
    assert "max_init_lead_mismatch_samples" in source, (
        "trainer 가 readiness 와 다른 기준으로 init lead 를 검사하고 있다"
    )
