"""실측 fine-tune 진입/완료 실패-폐쇄 게이트 회귀 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from deep_anc.config import REPO_ROOT, load_yaml
from deep_anc.data.manifest import read_manifest, write_manifest
from deep_anc.train.finetune_readiness import (
    MAX_RELATIVE_DELAY_SPREAD_SAMPLES,
    achievable_cancellation_ceiling_db,
    audit_finetune_completion,
    audit_finetune_readiness,
    audit_official_path_model,
    require_finetune_readiness,
    required_consistency_for,
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
        # 유지 반복 수. MIN_KEPT_REPEATS=8 이 하한이다 — 기각을 많이 한 것은 문제가
        # 아니지만(복구 캡처는 48 중 30 을 기각했다) 남은 것이 적으면 한 번의
        # 이상치가 플랜트 형상을 지배한다.
        "repeats": np.asarray(12, dtype=np.int64),
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


GROUPS_PER_FAMILY_PER_SPLIT = 4
"""계열·split 당 그룹 수. ``readiness.min_groups_per_family_per_split`` 과 같아야 한다.

3 으로 내리면 ``recorded_statistical_power`` 가 FAIL 한다 — 즉 이 상수 하나로 검정력
게이트가 살아 있는지 확인할 수 있다.
"""

# P(z) 픽스처의 FIR 은 ``[0.5, -0.1, 0.02]`` 라 최대 tap 이 0 이다. 따라서 유도되는
# 재생→ERR 지연은 ``벌크지연 + 0`` 이고, 실측 세션도 같은 지연을 갖도록 만들어야
# D2 교차검증이 성립한다. 두 값을 **같은 곳에서 유도**하는 것이 이 상수의 목적이다.
PRIMARY_DELAY_SAMPLES = 4
ERR_MIC_DELAY_SAMPLES = PRIMARY_DELAY_SAMPLES
REF_MIC_DELAY_SAMPLES = 2

PRETRAIN_TRUSTED_BAND_HZ = (100.0, 1_000.0)
"""픽스처 init checkpoint 가 학습된 대역.

실제 사고와 같은 축이다: ``runs/pretrain_{base,tiny}_corrected`` 는 [150,600] 으로
학습됐는데 파인튜닝이 유도하는 대역은 [150,1600] 이다. 여기서는 픽스처 S(z) 의
신뢰대역( ``excitation_band_hz`` [100,1000] )과 기본 목표대역 [80,1000] 의 교집합이
파인튜닝 대역이므로 checkpoint 도 같은 [100,1000] 이어야 통과한다.
"""


def _recorded_band_noise(frames: int, seed: int) -> np.ndarray:
    """정렬 검사가 통하는 광대역 소스.

    옛 픽스처는 순음이었다. 순음은 상호상관 최대점이 주기마다 반복돼 지연이 다중값이
    되고, 무엇보다 ERR/REF 두 채널을 **같은 신호의 상수배**로 만들어 시간 관계를
    시험하지 않는다. 실측 프로그램 소재는 광대역이므로 픽스처도 광대역이어야 한다.
    """

    from scipy.signal import butter, lfilter

    rng = np.random.default_rng(seed)
    b, a = butter(4, [120.0 / (FS / 2), 1_200.0 / (FS / 2)], btype="band")
    filtered = lfilter(b, a, rng.standard_normal(frames + 2048))[2048:]
    peak = float(np.max(np.abs(filtered))) or 1.0
    return 0.3 * filtered / peak


def _recorded_manifest(
    root: Path,
    *,
    frames: int = 4_096,
    groups_per_family: int = GROUPS_PER_FAMILY_PER_SPLIT,
    collapse_alignment: bool = False,
    source_delay_samples: int = ERR_MIC_DELAY_SAMPLES,
) -> Path:
    """실측 manifest 픽스처.

    ``collapse_alignment`` 는 2026-08-04 사고(재생/캡처 타임베이스 붕괴)를 주입한다.
    ``source_delay_samples`` 는 D2(실측 지연 vs P(z) 유도값) 교차검증을 흔들 때 쓴다.
    """

    manifest = root / "manifests" / "recorded.jsonl"
    entries = []
    for family_index, family in enumerate(FAMILIES):
        for split_index, split in enumerate(("train", "val", "test")):
            for group_index in range(groups_per_family):
                session_id = f"{family}-{split}-{group_index}"
                session = root / "recorded" / session_id
                session.mkdir(parents=True)
                source = _recorded_band_noise(
                    frames, 1_000 * family_index + 10 * split_index + group_index
                )
                err = 0.7 * np.roll(source, int(source_delay_samples))
                ref = 0.4 * np.roll(source, REF_MIC_DELAY_SAMPLES)
                if collapse_alignment:
                    # ERR/REF 마이크는 그대로 두고 source 만 시간축을 깬다 —
                    # 실측에서 무너진 것이 음향이 아니라 소프트웨어였기 때문이다.
                    rng = np.random.default_rng(7 + group_index)
                    block = max(64, frames // 8)
                    broken = np.zeros(frames, dtype=np.float64)
                    for start in range(0, frames, block):
                        stop = min(frames, start + block)
                        jump = int(rng.integers(-1_000, 1_000))
                        broken[start:stop] = np.roll(source, jump)[start:stop]
                    source = broken
                mics = np.stack([err, ref], axis=1).astype(np.float32)
                sf.write(session / "mics.wav", mics, FS, subtype="FLOAT")
                sf.write(
                    session / "source.wav", source.astype(np.float32), FS, subtype="FLOAT"
                )
                group = f"group-{family}-{split}-{group_index}"
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


def _corpus_fixture(root: Path, *, leak: bool = False) -> tuple[Path, Path]:
    """실측 소스 목록(``sources.csv``)과 합성 소음 manifest 를 만든다.

    ``leak=True`` 면 실측 music 이 쓴 원본이 합성 train 에도 들어간다 — 2026-08-05
    감사가 실측에서 찾은 상태(music 60/60 겹침, 그중 55개가 합성 train)의 축소판이다.
    """

    pool_dir = root / "source_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pool_dir / "sources.csv"
    recorded_clips = {
        "speech": ["spk1-a.flac", "spk1-b.flac"],
        "music": ["music-0001.mp3", "music-0002.mp3"],
        "environment": ["env-1.wav", "env-2.wav"],
        "machine": ["mach-1.wav", "mach-2.wav"],
    }
    rows = ["source_family,session_index,group_id,path,seconds,clips"]
    for family, clips in recorded_clips.items():
        payload = json.dumps(clips).replace('"', '""')
        rows.append(f'{family},0,grp-{family},{family}_000.wav,70.0,"{payload}"')
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    manifest_dir = root / "synth_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    speech_entries = [
        {
            "path": str(root / "raw" / "speech" / "other-1.flac"),
            "duration_s": 5.0,
            "sample_rate": FS,
            "channels": 1,
            "tag": "speech",
            "split": "train",
        }
    ]
    music_paths = ["fresh-a.mp3", "fresh-b.mp3"]
    if leak:
        # 실측이 재생한 바로 그 원본이 합성 train 에도 있다.
        music_paths = ["music-0001.mp3", "fresh-b.mp3"]
    music_entries = [
        {
            "path": str(root / "raw" / "music" / "000" / name),
            "duration_s": 30.0,
            "sample_rate": FS,
            "channels": 2,
            "tag": "music",
            "split": "train",
        }
        for name in music_paths
    ]
    write_manifest(speech_entries, manifest_dir / "speech.jsonl")
    write_manifest(music_entries, manifest_dir / "music.jsonl")
    return csv_path, manifest_dir


def _ready_config(tmp_path: Path, *, manifest: Path | None = None, leak: bool = False) -> dict:
    primary = tmp_path / "primary.npz"
    secondary = tmp_path / "secondary.npz"
    _official_path(primary, channel="noise", delay=PRIMARY_DELAY_SAMPLES)
    _official_path(secondary, channel="cancel", delay=5)
    if manifest is None:
        manifest = _recorded_manifest(tmp_path / "data")
    source_csv, synth_dir = _corpus_fixture(tmp_path / "data", leak=leak)
    model_cfg = {"name": "test-model", "hop": 4}
    pretrain_cfg = {
        "model": model_cfg,
        "data": {"digital_reference_lead_samples": 3},
        "digital_reference_lead_samples": 3,
        "physics_status": "secondary_surrogate_representation_pretrain",
        "schedule": {"total_steps": 10},
        # 어느 대역에서 벌점을 받았는가. trainer 가 resolved cfg 에 저장하는 값이고
        # ``completed_init_checkpoint`` 게이트가 파인튜닝 대역과 대조한다.
        # 픽스처 S(z) 의 신뢰대역 [100,1000] ∩ 기본 목표대역 [80,1000] = [100,1000].
        "trusted_band_hz": list(PRETRAIN_TRUSTED_BAND_HZ),
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
            # 코퍼스 누수 게이트(D1)가 읽는 합성 스트림 구성.
            "noise_manifest_dir": str(synth_dir),
            "source_mix_ratio": {"speech": 0.5, "music": 0.5},
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
            # --- 신규 게이트가 소비하는 선언 ---
            # 픽스처 일관성 0.97 두 경로 → 상한 12.1 dB. 목표 3 + 여유 3 = 6 이므로 통과.
            "target_cancellation_db": 3.0,
            "cancellation_ceiling_margin_db": 3.0,
            "min_groups_per_family_per_split": GROUPS_PER_FAMILY_PER_SPLIT,
            "recorded_source_pool_csv": str(source_csv),
            "min_delay_crosscheck_sessions": 8,
            "max_measured_delay_mismatch_samples": 8.0,
        },
    }


def _plant_fingerprint_payload(**overrides) -> str:
    """metrics.npz 에 박히는 플랜트 지문. 기본값은 val/test 가 같은 플랜트다."""

    payload = {
        "primary_delay_samples": 4,
        "secondary_delay_samples": 5,
        "handoff_samples": 2,
        "lead_samples": 3,
        "sample_rate": FS,
        "physics_status": "measured_primary_path",
        "optimize_band_hz": [100.0, 1_000.0],
        "secondary_sha256": None,
        "primary_sha256": None,
        "capture_id": None,
        "configured_lead_samples": 3,
    }
    payload.update(overrides)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _g4_metrics(
    path: Path,
    *,
    split: str,
    checkpoint: Path,
    manifest: Path,
    source_pass: bool = True,
    worst_source_db: float = -4.0,
    include_source_fields: bool = True,
    include_modern_fields: bool = True,
    verdict: str = "PASS",
    do_no_harm_pass: bool = True,
    power_pass: bool = True,
    ci_pass: bool = True,
    worst_octave_center_hz: float = 500.0,
    worst_octave_worst10_db: float = 3.0,
    fingerprint: str | None = None,
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
    if include_modern_fields:
        # 2026-08-05 신설 판정. 이 필드들이 없는 산출물은 게이트가 거부해야 한다
        # (include_modern_fields=False 로 그 회귀를 검사한다).
        payload.update(
            g4_verdict=np.asarray(verdict),
            g4_do_no_harm_pass=np.asarray(bool(do_no_harm_pass)),
            g4_power_pass=np.asarray(bool(power_pass)),
            g4_ci_pass=np.asarray(bool(ci_pass)),
            g4_worst_octave_center_hz=np.asarray(float(worst_octave_center_hz)),
            g4_worst_octave_worst10_db=np.asarray(float(worst_octave_worst10_db)),
            plant_fingerprint_json=np.asarray(
                fingerprint if fingerprint is not None else _plant_fingerprint_payload()
            ),
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
        # --- 2026-08-05 신설 ---
        # 각각 실측으로 확인된 결함 하나씩에 대응한다. 이 집합에서 항목을 빼면
        # 그 결함을 다시 통과시키는 것이므로 테스트가 즉시 깨진다.
        "recorded_alignment_integrity",   # 결함 2: source→ERR 관계가 존재하는가
        "recorded_statistical_power",     # 결함 4 / D3: 계열당 그룹 ≥ 하한
        "corpus_disjoint",                # D1: 합성 ∩ 실측 원본 = ∅
        "measured_source_delay_agreement",  # D2: 실측 지연 == P(z) 유도값
        "plant_confidence_ceiling",       # G1c: 목표가 달성 가능 상한 안인가
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
        "trusted_band_hz": list(PRETRAIN_TRUSTED_BAND_HZ),
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


# ======================================================================================
# 신설 게이트 5종의 negative fixture
#
# 규칙: **게이트마다 그것을 FAIL 시키는 테스트가 짝으로 있어야 한다.** 이것이 없어서
# 게이트 9개가 전부 PASS 인 채로 무용지물이라는 것을 아무도 몰랐다.
# ======================================================================================
def _gate(report: dict, gate_id: str) -> dict:
    for item in report["checks"]:
        if item["id"] == gate_id:
            return item
    raise AssertionError(f"게이트를 찾을 수 없습니다: {gate_id} ({[c['id'] for c in report['checks']]})")


# ---- G2b 학습 데이터 정렬 (결함 2) ---------------------------------------------------
def test_readiness_rejects_collapsed_source_err_timebase(tmp_path):
    """재생/캡처 타임베이스가 깨진 실측 데이터로는 학습을 시작할 수 없다.

    2026-08-04 실측: coh²(source→ERR) 0.021~0.126 인데 QA 는 80/80 PASS 였다.
    """

    manifest = _recorded_manifest(tmp_path / "data", collapse_alignment=True)
    cfg = _ready_config(tmp_path, manifest=manifest)

    report = audit_finetune_readiness(cfg)

    assert not report["ok"]
    gate = _gate(report, "recorded_alignment_integrity")
    assert not gate["ok"]
    assert "결맞음" in gate["message"] or "재녹음" in gate["message"]


def test_readiness_does_not_count_a_skipped_qa_as_alignment_evidence(tmp_path):
    """QA 를 건너뛴 실행은 정렬 게이트를 **통과하지 못한다**.

    "측정하지 않았다"를 "측정해서 통과했다"와 같게 취급하는 것이 이 저장소에서
    반복된 실패다.
    """

    cfg = _ready_config(tmp_path)
    report = audit_finetune_readiness(cfg, full_recorded_qa=False)

    gate = _gate(report, "recorded_alignment_integrity")
    assert not gate["ok"]
    assert "측정하지 않은" in gate["message"]


# ---- G2c 통계적 검정력 (결함 4 / D3) -------------------------------------------------
def test_readiness_rejects_underpowered_val_and_test_groups(tmp_path):
    """계열당 그룹이 1–2개면 G4 판정이 성립하지 않으므로 진입을 막는다.

    실측 상태: val machine 1그룹, test environment 1그룹, test machine 1그룹.
    """

    manifest = _recorded_manifest(tmp_path / "data", groups_per_family=2)
    cfg = _ready_config(tmp_path, manifest=manifest)

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "recorded_statistical_power")
    assert not gate["ok"]
    assert "cluster bootstrap" in gate["message"]
    # 어느 split/계열이 부족한지 기계 판독 가능해야 한다.
    weak = {(item[0], item[1]) for item in gate["details"]["weak"]}
    assert ("val", "machine") in weak and ("test", "speech") in weak


# ---- D1 코퍼스 누수 -------------------------------------------------------------------
def test_readiness_rejects_corpus_leak_between_synthetic_and_recorded(tmp_path):
    """합성 학습 스트림과 실측이 같은 원본 오디오를 쓰면 FAIL 한다.

    2026-08-05 실측: music 60/60(100%)이 겹치고 55개가 합성 train 에 있었다.
    같은 곡에서 두 브랜치가 반대 방향 gradient 를 주고, **music 만 개선되지 않았다.**
    """

    cfg = _ready_config(tmp_path, leak=True)

    report = audit_finetune_readiness(cfg)

    assert not report["ok"]
    gate = _gate(report, "corpus_disjoint")
    assert not gate["ok"]
    assert "music" in gate["message"]
    families = gate["details"]["families"]
    assert families["music"]["overlap_clips"] == 1
    assert families["speech"]["overlap_clips"] == 0
    # 겹친 원본이 합성의 어느 split 에 있었는지까지 나와야 한다.
    assert families["music"]["overlap_by_synthetic_split"] == {"train": 1}


def test_readiness_refuses_to_claim_disjoint_corpora_it_cannot_see(tmp_path):
    """합성 manifest 가 없으면 "겹치지 않는다"고 주장하지 않는다 (실패 폐쇄)."""

    cfg = _ready_config(tmp_path)
    cfg["data"]["noise_manifest_dir"] = str(tmp_path / "nonexistent")

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "corpus_disjoint")
    assert not gate["ok"]
    assert "판정할 수 없습니다" in gate["message"]


def test_readiness_refuses_when_only_some_declared_tags_have_a_manifest(tmp_path):
    """태그 **일부만** manifest 가 있으면 "서로소" 라고 말하지 않는다.

    2026-08-06 통합 검증에서 실제로 재현된 fail-open 이다. data/manifests 에
    esc50.jsonl 하나만 있는 상태에서 이 게이트가 "실측 691개와 합성 1587개가 서로소"
    로 PASS 했는데, D1 이 실제로 찾은 누수는 music 60/60(100%)이고 music.jsonl 이
    없어 비교 대상에 아예 들어가지 않았다. 없는 태그는 조용히 사라지지 않고
    synth_dataset 이 합성원으로 폴백하므로, 선언한 혼합비와 다른 데이터로 돌기까지 한다.
    """

    cfg = _ready_config(tmp_path)
    manifest_dir = Path(cfg["data"]["noise_manifest_dir"])
    # speech 만 남기고 music manifest 를 지운다 — 선언은 그대로 speech/music 이다.
    (manifest_dir / "music.jsonl").unlink()

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "corpus_disjoint")
    assert not gate["ok"]
    assert "music" in gate["message"]
    assert "manifest 가 없는" in gate["message"]


# ---- D2 실측-측정 지연 교차검증 --------------------------------------------------------
def test_readiness_rejects_recorded_delay_disagreeing_with_the_primary_path(tmp_path):
    """실측 세션의 source→ERR 지연이 P(z) 유도값과 다르면 FAIL 한다.

    2026-08-05 감사: 독립 세 방법이 ~1670 으로 일치하는데 유도값은 ~1850~1950 이었다.
    차이 180~280 샘플(4~6 ms), 비용은 계열별 +0.71 ~ +2.39 dB. 검사하는 게이트가 없었다.
    """

    # P(z) 는 지연 4 를 신고하는데 실측 세션은 60 샘플 지연으로 녹음됐다.
    manifest = _recorded_manifest(tmp_path / "data", source_delay_samples=60)
    cfg = _ready_config(tmp_path, manifest=manifest)

    report = audit_finetune_readiness(cfg)

    assert not report["ok"]
    gate = _gate(report, "measured_source_delay_agreement")
    assert not gate["ok"]
    assert "두 방법으로 잰 값이 다릅니다" in gate["message"]
    assert gate["details"]["mismatch_samples"] > 8.0


def test_readiness_refuses_a_delay_crosscheck_with_too_few_sessions(tmp_path):
    """표본 하나짜리 중앙값으로 지연 부기를 승인하지 않는다."""

    cfg = _ready_config(tmp_path)
    cfg["readiness"]["min_delay_crosscheck_sessions"] = 10_000

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "measured_source_delay_agreement")
    assert not gate["ok"]
    assert "자기증명" in gate["message"]


# ---- G1c 달성 가능 상한 ---------------------------------------------------------------
def test_readiness_rejects_a_target_above_the_achievable_ceiling(tmp_path):
    """플랜트가 허용하는 상한을 넘는 목표는 **학습 시작 전에** 막는다."""

    cfg = _ready_config(tmp_path)
    cfg["readiness"]["target_cancellation_db"] = 20.0

    report = audit_finetune_readiness(cfg)

    assert not report["ok"]
    gate = _gate(report, "plant_confidence_ceiling")
    assert not gate["ok"]
    assert "재측정" in gate["message"]
    assert gate["details"]["binding_constraint"] == "플랜트 일관성"


def test_readiness_uses_the_tighter_of_the_two_ceilings(tmp_path):
    """γ 상한이 낙관적일 때 **정규방정식 설계 상한**이 구속해야 한다.

    복구된 플랜트에서 γ 상한은 약 28 dB 인데 직접 계산한 설계 상한은 6.53 dB 다.
    낙관적인 상한 하나만 믿는 것이 이 저장소에서 반복된 사고의 형태이므로, 두 값 중
    **작은 쪽**으로 판정하는지 못 박아 둔다.
    """

    cfg = _ready_config(tmp_path)
    cfg["readiness"]["target_cancellation_db"] = 3.0
    cfg["readiness"]["measured_design_ceiling_db"] = 4.0  # γ 상한(12.1)보다 훨씬 작다
    # 상한은 대역이 붙어야 숫자다. 요구 대역 [100, 1000] 을 덮게 선언한다.
    cfg["readiness"]["measured_design_ceiling_band_hz"] = [100, 1_000]

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "plant_confidence_ceiling")
    assert not gate["ok"]
    assert gate["details"]["binding_constraint"] == "정규방정식 설계 상한"
    assert gate["details"]["binding_ceiling_db"] == 4.0
    assert gate["details"]["gamma_ceiling_db"] > 10.0


def test_readiness_rejects_a_design_ceiling_measured_on_a_narrower_band(tmp_path):
    """좁은 대역에서 푼 상한으로 넓은 대역 요구를 통과시키지 못한다.

    실제로 있었던 오판정이다: 설정이 6.53 dB 를 선언했고 그것은 150-600Hz 값인데
    required_path_band_hz 는 [150, 1600] 이었다. 같은 플랜트를 요구 대역 전체에서
    다시 풀면 4.6 dB 라 목표 3.0 + 여유 3.0 을 통과할 수 없었는데, 대역 표시가 없는
    float 하나여서 아무도 대조하지 않았다. 상한이 낙관적인 방향으로 틀리면 그 오판정은
    항상 "어려운 대역을 방치한다" 쪽으로 나온다 — 절대목표 1 과 정면으로 충돌한다.
    """

    cfg = _ready_config(tmp_path)
    cfg["readiness"]["required_path_band_hz"] = [100, 1_000]
    cfg["readiness"]["target_cancellation_db"] = 3.0
    # 상한 자체는 넉넉하다. 그런데 그것을 잰 대역이 요구 대역의 절반뿐이다.
    cfg["readiness"]["measured_design_ceiling_db"] = 30.0
    cfg["readiness"]["measured_design_ceiling_band_hz"] = [100, 500]

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "plant_confidence_ceiling")
    assert not gate["ok"]
    assert "덮지 못합니다" in gate["message"]
    assert gate["details"]["design_ceiling_band_hz"] == [100.0, 500.0]
    assert gate["details"]["required_path_band_hz"] == [100.0, 1_000.0]


def test_readiness_rejects_a_design_ceiling_declared_without_its_band(tmp_path):
    """대역 없는 상한 선언은 그 자체로 거부한다 (짝 fixture)."""

    cfg = _ready_config(tmp_path)
    cfg["readiness"]["target_cancellation_db"] = 3.0
    cfg["readiness"]["measured_design_ceiling_db"] = 30.0
    cfg["readiness"].pop("measured_design_ceiling_band_hz", None)

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "plant_confidence_ceiling")
    assert not gate["ok"]
    assert "measured_design_ceiling_band_hz" in gate["message"]


def test_readiness_requires_the_target_to_be_declared_at_all(tmp_path):
    """목표를 선언하지 않으면 min_path_consistency 가 근거를 잃는다 — 그것도 FAIL 이다."""

    cfg = _ready_config(tmp_path)
    del cfg["readiness"]["target_cancellation_db"]

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "plant_confidence_ceiling")
    assert not gate["ok"]
    assert "target_cancellation_db" in gate["message"]


# ---- 상한 계산기 자체의 검증 -------------------------------------------------------------
def test_achievable_ceiling_matches_the_measured_plants():
    """유도식이 실측값을 재현하는지 확인한다. 주석의 숫자가 곧 회귀 기준이다."""

    # 출하본(오염된 S(z)) — 150-600Hz 와 전대역
    assert round(achievable_cancellation_ceiling_db(0.9556, 0.9730), 2) == 11.30
    assert round(achievable_cancellation_ceiling_db(0.7812, 0.9200), 2) == 4.35
    # 클린 재분석본
    assert round(achievable_cancellation_ceiling_db(0.993, 0.993), 2) == 18.51
    assert round(achievable_cancellation_ceiling_db(0.968, 0.979), 2) == 12.64
    # 단일 경로 기준 역함수
    assert round(required_consistency_for(12.0), 4) == 0.9406
    # 게이트 임계 0.90 이 암묵적으로 뜻하던 상한
    assert round(achievable_cancellation_ceiling_db(0.90), 2) == 9.54


def test_achievable_ceiling_is_monotonic_and_rejects_impossible_gamma():
    """γ 가 좋아지면 상한도 좋아지고, 물리적으로 불가능한 γ 는 거부된다."""

    values = [achievable_cancellation_ceiling_db(g) for g in (0.6, 0.8, 0.95, 0.999)]
    assert values == sorted(values)
    for bad in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            achievable_cancellation_ceiling_db(bad)


def test_required_consistency_round_trips_through_the_ceiling():
    """``required_consistency_for`` 는 단일 경로 상한의 역함수여야 한다."""

    for target in (3.0, 6.0, 12.0, 20.0):
        gamma = required_consistency_for(target)
        assert abs(achievable_cancellation_ceiling_db(gamma) - target) < 1e-9


# ======================================================================================
# 완료 게이트 — G4 강화(결함 3·4) 와 G5 플랜트 동일성(결함 5)의 negative fixture
# ======================================================================================
def _g4_completion_setup(tmp_path: Path, **metrics_kwargs):
    """완료 판정에 필요한 checkpoint/metrics 한 벌을 만든다."""

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
    val_kwargs = dict(metrics_kwargs)
    test_kwargs = dict(metrics_kwargs)
    test_kwargs.pop("val_only", None)
    for key in list(val_kwargs):
        if key.startswith("test_"):
            test_kwargs[key[len("test_"):]] = val_kwargs.pop(key)
    test_kwargs = {k: v for k, v in test_kwargs.items() if not k.startswith("test_")}
    _g4_metrics(val_metrics, split="val", checkpoint=best, manifest=manifest, **val_kwargs)
    _g4_metrics(
        test_metrics, split="test", checkpoint=best, manifest=manifest, **test_kwargs
    )
    return cfg, best, val_metrics, test_metrics


def test_completion_rejects_metrics_from_an_evaluator_without_do_no_harm(tmp_path):
    """대역 밖 do-no-harm·검정력·지문을 판정하지 않는 구버전 산출물은 거부된다.

    이 관례는 ``g4_source_pass`` 에서 이미 쓰였고 옳았다 — 옛 형식을 통과시키면
    새 게이트가 있으나 마나다.
    """

    cfg, best, val_metrics, test_metrics = _g4_completion_setup(
        tmp_path, include_modern_fields=False
    )

    report = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )

    assert not report["ok"]
    gate = _gate(report, "recorded_val_g4")
    assert not gate["ok"]
    assert "구버전" in gate["message"]
    assert "g4_do_no_harm_pass" in gate["message"]


def test_completion_rejects_out_of_band_amplification(tmp_path):
    """신뢰 대역이 좋아도 대역 밖을 키웠다면 완료가 아니다 (절대목표 1).

    실측 반증: tone300 이 fullband +5.95 dB 로 기준을 만족하면서 8 kHz 를
    −21.56 dB 증폭했다.
    """

    cfg, best, val_metrics, test_metrics = _g4_completion_setup(
        tmp_path,
        verdict="FAIL",
        do_no_harm_pass=False,
        worst_octave_center_hz=8_000.0,
        worst_octave_worst10_db=-21.56,
    )

    report = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )

    assert not report["ok"]
    gate = _gate(report, "recorded_val_g4")
    assert "do-no-harm 실패" in gate["message"]
    assert "8000Hz" in gate["message"].replace(" ", "")


def test_completion_does_not_accept_an_inconclusive_g4(tmp_path):
    """판정 불가는 완료가 아니다. 표본 부족을 PASS 로 흘려보내지 않는다."""

    cfg, best, val_metrics, test_metrics = _g4_completion_setup(
        tmp_path, verdict="INCONCLUSIVE", power_pass=False, ci_pass=False
    )

    report = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )

    assert not report["ok"]
    gate = _gate(report, "recorded_val_g4")
    assert "INCONCLUSIVE" in gate["message"]
    assert "통계적으로 성립하지 않습니다" in gate["message"]


def test_completion_rejects_val_and_test_from_different_plants(tmp_path):
    """val 과 test 가 서로 다른 플랜트에서 평가됐다면 나란히 놓을 수 없다 (G5).

    2026-08-04 사고: 전 = S 지연 1342 / lead 109 / surrogate, 후 = 1465 / 113 /
    measured 를 비교해 "1.30 dB 개선" 이라고 적었다.
    """

    cfg, best, val_metrics, test_metrics = _g4_completion_setup(
        tmp_path,
        test_fingerprint=_plant_fingerprint_payload(
            secondary_delay_samples=1_465, lead_samples=113, configured_lead_samples=113
        ),
    )

    report = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )

    assert not report["ok"]
    gate = _gate(report, "plant_identity_for_comparison")
    assert not gate["ok"]
    assert "서로 다른 플랜트" in gate["message"]
    assert "secondary_delay_samples" in gate["message"]


def test_completion_accepts_val_and_test_from_the_same_plant(tmp_path):
    """같은 플랜트끼리는 통과해야 한다 — 게이트가 무조건 거부하는 게 아님을 증명."""

    cfg, best, val_metrics, test_metrics = _g4_completion_setup(tmp_path)

    report = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )

    assert report["ok"], report
    gate = _gate(report, "plant_identity_for_comparison")
    assert gate["ok"]
    assert gate["details"]["digest"]


# ---- 플랜트 아티팩트 강화: 유지 반복 수와 P−S 상대 τ 궤적 ------------------------------
def test_official_path_gate_rejects_too_few_kept_repeats(tmp_path):
    """반복을 적게 남긴 플랜트는 official 이 될 수 없다.

    기각을 많이 한 것은 문제가 아니다 — 2026-08-05 복구 캡처는 48 중 30 을 기각했고
    그것이 옳은 조치였다. 문제는 **남은 것이 적을 때**다. 이전 하한 3 은 한 번의
    이상치가 플랜트 형상을 지배하는 것을 허용했다.
    """

    thin = tmp_path / "thin.npz"
    _official_path(thin, channel="cancel", delay=5)
    with np.load(thin, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["repeats"] = np.asarray(3, dtype=np.int64)
    np.savez(thin, **arrays)

    with pytest.raises(ValueError, match="유지된 반복"):
        audit_official_path_model(
            thin,
            expected_output_channel="cancel",
            sample_rate=FS,
            required_band_hz=(100, 1_000),
        )


def test_matched_conditions_reads_the_stored_relative_tau_trajectory(tmp_path):
    """게이트가 **저장돼 있던 궤적을 실제로 읽는지** 확인한다.

    2026-08-04 사고의 핵심은 증거가 없었던 게 아니라 게이트가 파일 안의 증거를 한 번도
    열어보지 않았다는 것이다. 여기서는 그 궤적에 실제 사고 값을 넣고 거부되는지 본다.

    실측 P−S 상대 τ (출하 아티팩트): 반복 0–10 은 ~1.2, 반복 11–15 는 ~32.
    스칼라 delay_spread_samples 는 range 라 이 이봉 구조를 32 라는 한 숫자로 뭉갠다.
    """

    shipped_relative = np.asarray(
        [0.0, 1.20, 1.13, 1.09, 1.09, 1.29, 1.41, 1.47, 1.14, 1.48, 1.36,
         32.11, 32.18, 31.75, 30.26, 29.06]
    )
    tau_secondary = np.zeros_like(shipped_relative)

    cfg = _ready_config(tmp_path)
    for key, tau in (
        ("digital_reference", shipped_relative),
        ("secondary_path", tau_secondary),
    ):
        path = (
            cfg["duct"]["digital_reference"]["primary_path_npz"]
            if key == "digital_reference"
            else cfg["duct"]["secondary_path"]["npz"]
        )
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["repeat_tau_samples"] = tau
        # 스칼라 요약은 **정상값으로 남겨 둔다** — 궤적을 읽지 않으면 통과해야 한다.
        arrays["delay_spread_samples"] = np.asarray(1, dtype=np.int64)
        np.savez(path, **arrays)

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "matched_path_measurement_conditions")
    assert not gate["ok"], "스칼라 요약만 보면 통과하지만 궤적을 보면 프레임 슬립이다"
    assert "프레임 슬립" in gate["message"]
    assert "[11, 12, 13, 14, 15]" in gate["message"]


def test_matched_conditions_accepts_a_constant_relative_tau(tmp_path):
    """상수 궤적은 통과해야 한다 — 무조건 거부하는 검사가 아님을 증명."""

    cfg = _ready_config(tmp_path)
    drift = np.linspace(0.0, 40.0, 12)  # 두 채널에 **공통**으로 실린 드리프트
    for path, offset in (
        (cfg["duct"]["digital_reference"]["primary_path_npz"], 1.4),
        (cfg["duct"]["secondary_path"]["npz"], 0.0),
    ):
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["repeat_tau_samples"] = drift + offset
        np.savez(path, **arrays)

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "matched_path_measurement_conditions")
    assert gate["ok"], gate
    # 공통 드리프트는 상대 τ 에서 상쇄된다 — 그것이 동시 측정의 요점이다.
    assert gate["details"]["relative_tau"]["max_deviation_samples"] < 3.0


def test_group_floor_has_a_single_source():
    """계열당 그룹 하한이 **한 곳에서만** 정의되는지 강제한다 (발생기 A).

    진입 게이트와 G4 평가기가 각자 숫자를 들고 있으면 언젠가 한쪽만 바뀌고, 그러면
    "진입은 통과했는데 완료는 판정 불가" 라는 해석 불가능한 상태가 된다. 이 저장소의
    사고는 대부분 그 모양이었다(lead 109 vs 113 등).
    """

    from deep_anc.eval.recorded import MIN_GROUPS_PER_FAMILY
    from deep_anc.train.finetune_readiness import (
        MIN_GROUPS_PER_FAMILY_PER_SPLIT,
        _min_groups_per_family_default,
    )

    assert _min_groups_per_family_default() == MIN_GROUPS_PER_FAMILY
    assert MIN_GROUPS_PER_FAMILY_PER_SPLIT == MIN_GROUPS_PER_FAMILY
    # 설정이 값을 주지 않아도 게이트가 같은 하한을 쓴다.
    assert load_yaml(REPO_ROOT / "configs/train_finetune.yaml")["readiness"][
        "min_groups_per_family_per_split"
    ] == MIN_GROUPS_PER_FAMILY


# ======================================================================================
# 오발동 반증 — **정상 산출물을 경계까지 몰아도 게이트가 울리지 않는가** (군집 B 나머지 절반)
#
# 2026-08-06 반증 #13: 이 저장소의 메타 테스트는 "발동시키는 fixture 가 있는가" 만
# 강제했다. 모든 게이트의 반응은 차단(학습 중단 / mute = 상쇄 0 dB)인데, "정상 입력에서
# 안 울리는가" 를 운용 범위 끝까지 몰아본 게이트가 하나도 없었다. 그래서 실제로
# 재정렬에 성공한 세션 9개 중 4개(44%)를 QA 게이트가 떨어뜨리고 있었는데도 아무도
# 몰랐다. 아래 테스트들이 그 반쪽이다 — 전부 **한계 위의 정상값**을 넣는다.
# ======================================================================================
def _boundary_config(tmp_path: Path) -> dict:
    """모든 진입 게이트를 **한계에 붙여** 통과시키는 설정.

    여유를 주지 않는다. 여기서 하나라도 한 눈금 더 나빠지면 그 게이트가 FAIL 해야
    하고(그 짝은 각 negative fixture 가 본다), 지금 이대로는 전부 PASS 해야 한다.
    """

    # D2 교차검증을 허용 오차 8.0 샘플의 **90% 지점**(7 샘플 어긋남)에서 돌린다.
    manifest = _recorded_manifest(
        tmp_path / "data", source_delay_samples=ERR_MIC_DELAY_SAMPLES + 7
    )
    cfg = _ready_config(tmp_path, manifest=manifest)
    entries = read_manifest(Path(cfg["recorded_manifest"]))
    sessions = len(entries)
    duration = sum(float(item["duration_s"]) for item in entries)

    # 픽스처 아티팩트의 실제 값. 게이트 임계를 여기에 **정확히** 맞춘다.
    with np.load(tmp_path / "secondary.npz", allow_pickle=False) as data:
        consistency = float(data["coherence_median"])
        excitation = [float(v) for v in np.asarray(data["excitation_band_hz"]).reshape(-1)]

    # 상대 τ spread 를 허용 최대값에 정확히 맞춘다 (3 = 코드 상수).
    for name in ("primary.npz", "secondary.npz"):
        path = tmp_path / name
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["delay_spread_samples"] = np.asarray(
            MAX_RELATIVE_DELAY_SPREAD_SAMPLES, dtype=np.int64
        )
        np.savez(path, **arrays)

    ceiling = achievable_cancellation_ceiling_db(consistency, consistency)
    margin = 3.0
    cfg["readiness"].update(
        {
            # 대역: 아티팩트가 실제로 구동한 **양 끝**을 그대로 요구한다.
            "required_path_band_hz": excitation,
            "min_path_consistency": consistency,          # 여유 0
            "required_recorded_ratio": float(cfg["recorded_ratio"]),
            "min_recorded_sessions": sessions,            # 여유 0
            "min_recorded_duration_seconds": duration,    # 여유 0
            "min_groups_per_family_per_split": GROUPS_PER_FAMILY_PER_SPLIT,
            "min_delay_crosscheck_sessions": sessions,    # 여유 0
            "max_measured_delay_mismatch_samples": 8.0,   # 실제 어긋남 7 = 한계의 90%
            # 목표 + 여유가 상한의 90% 지점에 오도록 잡는다.
            "target_cancellation_db": 0.9 * ceiling - margin,
            "cancellation_ceiling_margin_db": margin,
        }
    )
    return cfg


def test_every_entry_gate_passes_at_its_declared_boundary(tmp_path):
    """진입 게이트 전부가 **한계에 붙은 정상 데이터**에서 PASS 한다.

    몰아본 경계 (전부 여유 0 또는 한계의 90%):
      · 세션 수 = 최소 세션 수 (48)             · 분량 = 최소 분량
      · 반복 일관성 = min_path_consistency      · 요구 대역 = 구동 대역 양 끝 100/1000Hz
      · P−S 상대 τ spread = 허용 최대 3 샘플    · 계열당 그룹 = 하한 4
      · source→ERR 지연 어긋남 7 = 허용 8.0 의 90%
      · 목표 + 여유 = 달성 가능 상한의 90%
    """

    cfg = _boundary_config(tmp_path)

    report = audit_finetune_readiness(cfg)

    failed = [item for item in report["checks"] if not item["ok"]]
    assert failed == [], failed
    assert report["ok"], report
    # 그리고 이것이 "게이트가 없어서 통과" 가 아님을 못박는다.
    assert len(report["checks"]) >= 14


def test_measured_delay_agreement_is_not_a_free_pass_one_sample_further(tmp_path):
    """한계의 90% 에서 PASS 하는 그 게이트가 한계를 넘으면 FAIL 하는가.

    오기각 방지 테스트가 "게이트가 꺼져 있어서 통과" 를 증명하는 사고를 막는다.
    """

    cfg = _boundary_config(tmp_path)
    manifest = _recorded_manifest(
        tmp_path / "beyond", source_delay_samples=ERR_MIC_DELAY_SAMPLES + 20
    )
    cfg["recorded_manifest"] = str(manifest)

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "measured_source_delay_agreement")
    assert not gate["ok"], gate


def test_completion_gates_pass_at_the_minimum_sample_boundary(tmp_path):
    """완료 게이트 전부가 **최소 표본**의 정상 산출물에서 PASS 한다.

    몰아본 경계: 세션 1개 / 세그먼트 1개(0 이면 FAIL 하는 하한 바로 위),
    최악 계열 −0.01 dB(개선이라 말할 수 있는 최소값), 최악 옥타브 0.01 dB.
    """

    cfg, best, val_metrics, test_metrics = _g4_completion_setup(
        tmp_path,
        worst_source_db=-0.01,
        worst_octave_worst10_db=0.01,
    )
    for path in (val_metrics, test_metrics):
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["n_sessions"] = np.asarray(1, dtype=np.int64)
        arrays["n_segments"] = np.asarray(1, dtype=np.int64)
        np.savez_compressed(path, **arrays)

    report = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )

    failed = [item for item in report["checks"] if not item["ok"]]
    assert failed == [], failed
    assert report["fine_tuning_complete"]


def test_official_path_delay_spread_passes_at_the_allowed_maximum(tmp_path):
    """허용 최대 spread(3 샘플) 정확히 위에서는 통과한다 — 한 샘플 더 가면 FAIL."""

    cfg = _ready_config(tmp_path)
    for name in ("primary.npz", "secondary.npz"):
        path = tmp_path / name
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["delay_spread_samples"] = np.asarray(
            MAX_RELATIVE_DELAY_SPREAD_SAMPLES, dtype=np.int64
        )
        np.savez(path, **arrays)

    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    assert _check(report, "official_secondary_path")["ok"]
    assert _check(report, "official_primary_path")["ok"]

    # 한 샘플 더: 같은 게이트가 거부한다.
    path = tmp_path / "secondary.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["delay_spread_samples"] = np.asarray(
        MAX_RELATIVE_DELAY_SPREAD_SAMPLES + 1, dtype=np.int64
    )
    np.savez(path, **arrays)
    worse = audit_finetune_readiness(cfg, full_recorded_qa=False)
    assert not _check(worse, "official_secondary_path")["ok"]


# ---- 결함: init checkpoint 의 **대역 축** (2026-08-06) ---------------------------------
def test_init_checkpoint_trained_on_a_narrower_band_is_rejected(tmp_path):
    """좁은 대역으로 학습한 checkpoint 는 고역 증폭기다 — 게이트가 대역 축을 본다.

    실제 상태: ``runs/pretrain_{base,tiny}_corrected`` 는 cfg.trusted_band_hz
    [150, 600] 인데 현재 설정이 유도하는 대역은 [150, 1600] 이다. 벌점이 없던
    600-1600Hz 를 그 모델은 적극 증폭한다([150,800] 설정 실측 +27.01 dB).
    lead 축만 보던 게이트는 이것을 통과시켰다.
    """

    cfg = _ready_config(tmp_path)
    init = Path(cfg["init_ckpt"])
    narrow = {
        "model": cfg["model"],
        "data": {"digital_reference_lead_samples": 3},
        "digital_reference_lead_samples": 3,
        "physics_status": "secondary_surrogate_representation_pretrain",
        "schedule": {"total_steps": 10},
        # 위쪽 절반을 못 본 채로 학습됐다.
        "trusted_band_hz": [PRETRAIN_TRUSTED_BAND_HZ[0], 600.0],
    }
    _checkpoint(init, cfg=narrow, step=8)
    _checkpoint(init.parent / "last.pt", cfg=narrow, step=10)

    report = audit_finetune_readiness(cfg, full_recorded_qa=False)

    check = _check(report, "completed_init_checkpoint")
    assert not check["ok"]
    assert "대역" in check["message"]
    assert "+27.01 dB" in check["message"]


def test_init_checkpoint_without_a_recorded_band_is_rejected(tmp_path):
    """대역을 기록하지 않은 checkpoint 는 '모른다' 이지 '괜찮다' 가 아니다."""

    cfg = _ready_config(tmp_path)
    init = Path(cfg["init_ckpt"])
    unlabelled = {
        "model": cfg["model"],
        "data": {"digital_reference_lead_samples": 3},
        "digital_reference_lead_samples": 3,
        "physics_status": "secondary_surrogate_representation_pretrain",
        "schedule": {"total_steps": 10},
    }
    _checkpoint(init, cfg=unlabelled, step=8)
    _checkpoint(init.parent / "last.pt", cfg=unlabelled, step=10)

    report = audit_finetune_readiness(cfg, full_recorded_qa=False)

    check = _check(report, "completed_init_checkpoint")
    assert not check["ok"]
    assert "trusted_band_hz" in check["message"]


def test_init_checkpoint_band_exactly_equal_to_the_finetune_band_passes(tmp_path):
    """경계: checkpoint 대역이 파인튜닝 대역과 **정확히 같을 때** 통과한다.

    100.0/1000.0 Hz 양 끝이 한 눈금도 어긋나지 않은 상태 — 여유 0 이다.
    더 넓은 대역에서 온 것도 통과해야 한다(벌점을 받아 본 구간이 더 넓다).
    """

    cfg = _ready_config(tmp_path)
    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    check = _check(report, "completed_init_checkpoint")
    assert check["ok"], check
    assert check["details"]["checkpoint"]["trusted_band_hz"] == [100.0, 1_000.0]
    assert check["details"]["checkpoint"]["expected_trusted_band_hz"] == [100.0, 1_000.0]

    init = Path(cfg["init_ckpt"])
    wider = {
        "model": cfg["model"],
        "data": {"digital_reference_lead_samples": 3},
        "digital_reference_lead_samples": 3,
        "physics_status": "secondary_surrogate_representation_pretrain",
        "schedule": {"total_steps": 10},
        "trusted_band_hz": [80.0, 1_600.0],
    }
    _checkpoint(init, cfg=wider, step=8)
    _checkpoint(init.parent / "last.pt", cfg=wider, step=10)
    assert _check(
        audit_finetune_readiness(cfg, full_recorded_qa=False),
        "completed_init_checkpoint",
    )["ok"]


def test_the_shipped_init_checkpoints_are_high_frequency_amplifiers():
    """출하 상태 확인 — 배포 후보 init checkpoint 가 실제로 이 게이트에 걸린다.

    runs/ 는 .gitignore 대상이라 이 기기에만 있다. 없으면 건너뛴다(다른 환경에서
    빨간불이 되지 않게). 있으면 **[150,600] 으로 학습된 사실**을 못박는다.
    """

    from deep_anc.dsp.timing import FrequencyBand
    from deep_anc.train.finetune_readiness import _checkpoint_optimize_band

    found = 0
    for name in ("pretrain_base_corrected", "pretrain_tiny_corrected"):
        path = REPO_ROOT / "runs" / name / "ckpt" / "best.pt"
        if not path.is_file():
            continue
        found += 1
        state = torch.load(path, map_location="cpu", weights_only=False)
        band = _checkpoint_optimize_band(state)
        assert band is not None
        assert band.as_tuple() == (150.0, 600.0), (name, band)
        # 현재 설정이 요구하는 대역을 덮지 못한다 = 게이트가 거부해야 한다.
        assert not band.covers(FrequencyBand(lo_hz=150.0, hi_hz=1_600.0))
    if found == 0:
        pytest.skip("runs/ 산출물이 없는 트리입니다 (.gitignore 대상)")


def test_readiness_alignment_thresholds_actually_reach_the_qa():
    """readiness 가 선언한 정렬 임계가 **실제로 QA 에 도달하는지** 못 박는다.

    ⚠ 2026-08-06 통합 검증이 잡은 결함의 회귀 테스트. 키 목록이 옛 4개로
    하드코딩돼 있어서 새 키를 선언하면 경고 한 줄 없이 버려졌다. HANDOFF 가
    지시한 "키 이름을 새 것으로 갈아라"를 따르는 순간 설정이 무력화되고,
    그 사실을 알리던 폐기 안내문마저 사라지는 조합이었다.

    목록을 손으로 베끼는 것이 원인이므로 QA 에서 유도하는지도 함께 본다.
    """

    from deep_anc.data.recorded_qa import (
        _ALIGNMENT_OVERRIDE_KEYS,
        settings_from_data_config,
    )
    from deep_anc.train.finetune_readiness import (
        _alignment_cfg_keys,
        _alignment_overrides,
    )

    # QA 가 받는 키는 전부 통로를 지나갈 수 있어야 한다 (복사본이 아니라 유도).
    assert set(_ALIGNMENT_OVERRIDE_KEYS) <= _alignment_cfg_keys()

    data_cfg = {
        "sample_rate": 48_000,
        "segment_seconds": 1.0,
        "digital_reference_lead_samples": 116,
    }

    # 1) 새 키가 실제로 반영된다 — 이것이 이전에 조용히 버려지던 경로다.
    # 값은 **강화 방향**이어야 통과한다. 2026-08-06 에 지터 상한이 8.0 에서
    # 대역 유도값 3.41 로 강화되면서 옛 4.0 은 이제 완화 방향이 됐다.
    declared = {
        "max_source_err_delay_robust_std_samples": 3.0,
        "max_source_err_delay_p95_p5_samples": 24.0,
        "min_source_err_delay_window_ratio": 0.90,
    }
    settings = settings_from_data_config(
        data_cfg, alignment_overrides=_alignment_overrides(declared)
    )
    assert settings.max_source_err_delay_robust_std_samples == 3.0
    assert settings.max_source_err_delay_p95_p5_samples == 24.0
    assert settings.min_source_err_delay_window_ratio == 0.90

    # 2) 폐기 키는 조용히 무시되지 않고 안내로 남는다.
    legacy = settings_from_data_config(
        data_cfg,
        alignment_overrides=_alignment_overrides(
            {
                "max_source_err_delay_std_samples": 64,
                "max_source_err_delay_range_samples": 256,
            }
        ),
    )
    assert len(legacy.deprecated_threshold_notes) == 2
    assert all("폐기" in note for note in legacy.deprecated_threshold_notes)

    # 3) 완화 방향은 통로를 지나가되 QA 가 거절한다 (조용히 통과하지 않는다).
    with pytest.raises(ValueError, match="강화 방향"):
        settings_from_data_config(
            data_cfg,
            alignment_overrides=_alignment_overrides(
                {"min_source_err_delay_window_ratio": 0.50}
            ),
        )


def test_delay_crosscheck_refuses_a_realigned_reference_instead_of_comparing_it():
    """D2 — **같은 이름이 두 물리량을 오가는 것**을 숫자 비교 전에 잡는다.

    ⚠ 2026-08-06 통합 검증. QA 는 "학습이 실제로 읽는 파일" 을 재므로 재정렬본이
    있으면 그것을 잰다. 그 값은 정렬 후 **잔여** 음향 지연(약 142 샘플)이고, 이
    게이트가 P(z) 유도값(1849)과 대조하려는 것은 **원본 재생→ERR 총지연**(관측 약
    1672)이다. 그대로 비교하면 1706 샘플짜리 가짜 실패가 나오고, 그 가짜 실패가
    진짜 결함(1672 vs 1849 = 177 샘플, 허용 64 의 2.8배)을 덮는다.

    두 경우 다 FAIL 이지만 **이유가 달라야** 한다 — 그것이 이 테스트의 요점이다.
    """

    from deep_anc.train.finetune_readiness import _Audit, _audit_measured_source_delay

    primary = {
        "path": "assets/measured/primary_path_il.npz",
        "delay_samples": 1_602,
    }

    def _run(reference: str, observed: float) -> dict:
        audit = _Audit("t")
        report = {
            "sessions": [
                {
                    "alignment_reference": reference,
                    "alignment": {"source_err_delay_median_samples": observed},
                }
                for _ in range(10)
            ]
        }
        _audit_measured_source_delay(audit, {}, primary, report)
        return next(
            item
            for item in audit.report()["checks"]
            if item["id"] == "measured_source_delay_agreement"
        )

    # (a) 재정렬본 — 숫자를 비교하지 않고 "비교할 수 없다" 고 말해야 한다.
    realigned = _run("source_aligned.wav", 142.5)
    assert not realigned["ok"]
    assert "재정렬본" in realigned["message"]
    # 가짜 숫자(1706)를 근거로 들지 않는다.
    assert "차이 1706" not in realigned["message"]

    # (b) 원본 — 진짜 결함이 그대로 드러나야 한다 (가려지면 안 된다).
    raw = _run("source.wav", 1_672.0)
    assert not raw["ok"]
    assert "177" in raw["message"] and "허용 64" in raw["message"]

    # (c) 짝: 원본 기준에서 유도값과 맞으면 통과한다 (게이트가 항상 실패하지 않는다).
    agreeing = _run("source.wav", 1_849.0)
    assert agreeing["ok"]


def test_readiness_rejects_a_pool_the_sessions_did_not_actually_play(tmp_path):
    """설정이 선언한 소스풀과 **세션이 실제로 재생한 풀**이 다르면 FAIL 한다.

    2026-08-06 감사가 재현한 fail-open. ``recorded_source_pool_csv`` 가 v1 을 가리키는데
    재녹음을 v2 로 하면, 누수 게이트가 **v1 클립끼리 비교해 PASS 하면서 v2 누수를 100%
    통과**시킨다. v1 은 machine 이 8 그룹뿐이라 분할 하한(9)을 만족할 수 없어 재녹음은
    v2 로 할 수밖에 없으므로, 이 상태는 우연이 아니라 **예정된 경로**였다.
    """

    cfg = _ready_config(tmp_path)
    sessions = tmp_path / "recorded_sessions"
    for name, played in (
        ("s0", "data/source_pool_v2/music/music_000.wav"),
        ("s1", "data/source_pool_v2/speech/speech_000.wav"),
    ):
        d = sessions / name
        d.mkdir(parents=True)
        (d / "session.json").write_text(
            json.dumps({"session_id": name, "program": {"type": "file", "file": played}}),
            encoding="utf-8",
        )
    cfg["readiness"]["recorded_session_root"] = str(sessions)

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "corpus_disjoint")
    assert not gate["ok"]
    assert "세션이 실제로 재생한 풀" in gate["message"]
    assert "source_pool_v2" in gate["message"]
