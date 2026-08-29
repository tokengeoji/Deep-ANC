"""fine-tune pipeline CLI 회귀 테스트.

지키려는 불변식 3가지:

1. **NOT READY 면 ``runs/`` 아래에 아무것도 만들지 않는다.** 학습 디렉터리의 존재가
   "학습이 실제로 시작됐다"는 의미를 유지해야 한다. 구버전은 감사 리포트를
   ``runs/<run>/audit/`` 에 써서 이 의미를 깨뜨렸다.
2. **중복 실행이 구분된다.** exit 4 는 pipeline 중복(무시 가능), exit 3 은 다른 경로로
   이미 학습이 돌고 있음(조사 대상). 구버전은 둘 다 1로 뭉갰다.
3. **status.json 은 advisory 다.** 위조해도 게이트 판정이 바뀌지 않는다.

여기서 쓰는 config 는 저장소의 실제 ``configs/train_finetune.yaml`` 이지만, 경로 해석을
tmp 로 완전히 격리해 recorded/checkpoint 가 없는 상태를 **구성**한다. 저장소가 실제로
READY 인지와 무관하게 NOT READY 경로가 검사되어야 하기 때문이다 — 예전에는 저장소가
우연히 NOT READY 라서 통과하던 테스트였고, 실측이 끝나자 전부 깨졌다.
GPU 도 실데이터도 필요 없다.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "train"))

from deep_anc.train.pipeline_status import (  # noqa: E402
    EXIT_CONFIG,
    EXIT_NOT_READY,
    EXIT_OK,
    EXIT_PIPELINE_LOCKED,
    PipelineStatus,
    atomic_write_text,
    read_status,
    stable_view,
)
from deep_anc.train.process_lock import (  # noqa: E402
    LockHeldError,
    ProcessLock,
    autostart_state_dir,
    finetune_run_key,
    resolve_run_dir,
)

import run_finetune_pipeline as pipeline  # noqa: E402
import train as train_entry  # noqa: E402
from deep_anc.eval.trusted_subbands import (  # noqa: E402
    MIN_SUBBAND_SOURCE_ENERGY_DENSITY_RATIO,
    STRICT_TRUSTED_SUBBAND_SCHEMA,
    STRICT_TRUSTED_SUBBANDS_HZ,
)
from deep_anc.eval.recorded_sampling import (  # noqa: E402
    CANONICAL_EDGE_TRIM_SECONDS,
    CANONICAL_MAX_SEGMENTS_PER_SESSION,
    CANONICAL_SEGMENT_SECONDS,
    RECORDED_SAMPLING_CONTRACT_SCHEMA,
    effective_segment_samples,
)
from deep_anc.dsp.timing import PlantDelays, TrainingTimingContract  # noqa: E402
from deep_anc.train.evaluation_contract import (  # noqa: E402
    canonical_test_ledger_paths,
    canonical_test_ledger_event_paths_from_payload,
    complete_test_evaluation,
    consume_test_capability,
    fail_test_evaluation,
    issue_test_capability,
    publish_directory_noreplace,
    write_json_exclusive,
)
from deep_anc.train.experiment_contract import stamp_experiment_contract  # noqa: E402

ARGS = [
    "--check-only",
    "--config",
    "configs/train_finetune.yaml",
    "--set",
    "data.digital_primary_path_mode=measured",
]


def _canonical_sampling_checkpoint_cfg(**updates) -> dict:
    timing = TrainingTimingContract.derive(
        primary_fir=np.asarray([1.0], dtype=np.float32),
        plant_delays=PlantDelays(
            primary_delay_samples=0,
            secondary_delay_samples=0,
            handoff_samples=0,
            sample_rate=48_000,
        ),
    )
    cfg = {
        "model": {"name": "toy", "hop": 128},
        "data": {
            "sample_rate": 48_000,
            "segment_seconds": CANONICAL_SEGMENT_SECONDS,
            "reference_mode": "digital",
            "recorded_lead_mode": "timeline",
            "digital_reference_lead_samples": 0,
            "training_timing_contract": timing.model_dump(),
            "closed_loop": {
                "feedback_delay_samples": [0, 0],
                "warmup_seconds": 0.0,
            },
        },
        "digital_reference_lead_samples": 0,
        "loss_start_sample": 0,
    }
    cfg.update(updates)
    return cfg


def _write_canonical_recorded_manifest(
    manifest: Path,
    *,
    splits: tuple[str, ...],
    source_families: tuple[str, ...] = (
        "speech",
        "music",
        "environment",
        "machine",
    ),
    groups_per_family: int = 4,
    duration_s: float = 2.0,
    aligned_lag_median_samples: float = 0.0,
) -> None:
    """raw fixture layout과 같은 selected session population을 JSONL로 고정한다."""

    families = tuple(sorted(source_families))
    rows: list[str] = []
    for split in splits:
        for family in families:
            for index in range(groups_per_family):
                session = f"{family}-{split}-s{index}"
                session_dir = manifest.parent / "sessions" / session
                session_dir.mkdir(parents=True, exist_ok=True)
                (session_dir / "session.json").write_text(
                    json.dumps(
                        {
                            "timeline": {
                                "aligned_lag_median_samples": aligned_lag_median_samples
                            }
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                rows.append(
                    json.dumps(
                        {
                            "path": str(session_dir),
                            "duration_s": duration_s,
                            "sample_rate": 48_000,
                            "split": split,
                            "session_id": session,
                            "source_family": family,
                            "group_id": f"{family}-{split}-g{index}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    manifest.write_text("".join(rows), encoding="utf-8")


def _recorded_val_metric_payload(
    *,
    checkpoint: Path,
    manifest: Path,
    contract_sha: str,
    margin_db: float,
    split: str = "val",
    manifest_splits: tuple[str, ...] | None = ("val",),
    selection_sha256: str = "",
    test_capability_sha256: str = "",
    test_consumed_marker_sha256: str = "",
    source_families: tuple[str, ...] = (
        "speech",
        "music",
        "environment",
        "machine",
    ),
    groups_per_family: int = 4,
) -> dict:
    """shared persisted-G4 raw fixture.

    공식 selection/completion은 scalar만으로 만들 수 없다. 이 fixture도 실제
    evaluator가 남기는 raw segment/global/family/octave evidence와 immutable manifest
    binding을 모두 만든다. 개별 negative test는 이 뒤 한 필드만 변조한다.
    """
    value = -float(margin_db)
    families = np.asarray(sorted(source_families), dtype=np.str_)
    # 각 family의 네 독립 group이 모든 strict 부대역을 포함하는 canonical valid fixture.
    segment_family = np.repeat(families, groups_per_family)
    segment_group = np.asarray(
        [
            f"{family}-{split}-g{index}"
            for family in families
            for index in range(groups_per_family)
        ],
        dtype=np.str_,
    )
    segment_session = np.asarray(
        [
            f"{family}-{split}-s{index}"
            for family in families
            for index in range(groups_per_family)
        ],
        dtype=np.str_,
    )
    if manifest_splits is not None:
        _write_canonical_recorded_manifest(
            manifest,
            splits=manifest_splits,
            source_families=source_families,
            groups_per_family=groups_per_family,
        )
    strict_values = np.full(
        (segment_family.size, len(STRICT_TRUSTED_SUBBANDS_HZ)),
        value,
        dtype=np.float64,
    )
    strict_shape = (families.size, len(STRICT_TRUSTED_SUBBANDS_HZ))
    strict_flags = np.ones(strict_shape, dtype=np.bool_)
    octave_centers = np.asarray(
        (125.0, 250.0, 500.0, 1000.0, 1600.0, 2000.0, 4000.0, 8000.0),
        dtype=np.float64,
    )
    octave_values = np.full((segment_family.size, octave_centers.size), 1.0, dtype=np.float64)
    source_counts = np.full(families.size, groups_per_family, dtype=np.int64)
    source_values = np.full(families.size, value, dtype=np.float64)
    power_pass = groups_per_family >= 4
    source_ci_values = (
        source_values
        if power_pass
        else np.full(families.size, np.nan, dtype=np.float64)
    )
    model_hop = 128
    segment_samples = effective_segment_samples(
        sample_rate=48_000,
        model_hop=model_hop,
        segment_seconds=CANONICAL_SEGMENT_SECONDS,
    )
    edge_trim_samples = int(round(CANONICAL_EDGE_TRIM_SECONDS * 48_000))
    timing = TrainingTimingContract.from_data_config(
        _canonical_sampling_checkpoint_cfg()["data"]
    )
    return {
        "g4_metric_scope": np.asarray("canonical_recorded_g4"),
        "physics_status": np.asarray("measured_primary_path"),
        "allow_surrogate": np.asarray(False),
        "sample_rate": np.asarray(48_000, dtype=np.int64),
        "recorded_sampling_contract_schema": np.asarray(
            RECORDED_SAMPLING_CONTRACT_SCHEMA
        ),
        "recorded_sampling_canonical": np.asarray(True),
        "recorded_sampling_model_hop": np.asarray(model_hop, dtype=np.int64),
        "recorded_sampling_max_segments_per_session": np.asarray(
            CANONICAL_MAX_SEGMENTS_PER_SESSION, dtype=np.int64
        ),
        "recorded_sampling_segment_seconds": np.asarray(
            CANONICAL_SEGMENT_SECONDS, dtype=np.float64
        ),
        "recorded_sampling_plant_settle_samples": np.asarray(0, dtype=np.int64),
        "segment_samples": np.asarray(segment_samples, dtype=np.int64),
        "metric_samples_per_segment": np.asarray(segment_samples, dtype=np.int64),
        "edge_trim_samples": np.asarray(edge_trim_samples, dtype=np.int64),
        "warmup_samples": np.asarray(0, dtype=np.int64),
        "feedback_delay_samples": np.asarray(0, dtype=np.int64),
        "digital_reference_lead_samples": np.asarray(0, dtype=np.int64),
        "primary_delay_samples": np.asarray(0, dtype=np.int64),
        "secondary_delay_samples": np.asarray(0, dtype=np.int64),
        "secondary_handoff_samples": np.asarray(0, dtype=np.int64),
        "trusted_band_hz": np.asarray((150.0, 1600.0), dtype=np.float64),
        "nmse_trusted_worst10_mean_db": np.asarray(value),
        "nmse_trusted_mean_db": np.asarray(value),
        "nmse_trusted_median_db": np.asarray(value),
        "nmse_fullband_mean_db": np.asarray(value),
        "nmse_fullband_median_db": np.asarray(value),
        "nmse_fullband_worst10_mean_db": np.asarray(value),
        "nmse_gap_trusted_minus_fullband_mean_db": np.asarray(0.0),
        "g4_worst_source_trusted_mean_db": np.asarray(value),
        "g4_worst_source_trusted_worst10_db": np.asarray(value),
        "g4_worst_source_family": np.asarray(str(families[0])),
        "g4_worst_octave_center_hz": np.asarray(125.0),
        "g4_worst_octave_worst10_db": np.asarray(1.0),
        "g4_max_out_of_band_amplification_db": np.asarray(1.0),
        "source_trusted_ci_lo_db": source_ci_values,
        "source_trusted_ci_hi_db": source_ci_values,
        "g4_trusted_pass": np.asarray(True),
        "g4_fullband_pass": np.asarray(True),
        "g4_source_pass": np.asarray(True),
        "g4_do_no_harm_pass": np.asarray(True),
        "g4_power_pass": np.asarray(power_pass),
        "g4_ci_pass": np.asarray(power_pass),
        "g4_min_groups_per_family": np.asarray(4, dtype=np.int64),
        "g4_underpowered_families": np.asarray(
            [] if power_pass else families, dtype=np.str_
        ),
        "g4_pass": np.asarray(power_pass),
        "g4_verdict": np.asarray("PASS" if power_pass else "INCONCLUSIVE"),
        "strict_trusted_subband_schema": np.asarray(STRICT_TRUSTED_SUBBAND_SCHEMA),
        "strict_trusted_subband_min_source_energy_density_ratio": np.asarray(
            MIN_SUBBAND_SOURCE_ENERGY_DENSITY_RATIO
        ),
        "trusted_subband_hz": np.asarray(
            STRICT_TRUSTED_SUBBANDS_HZ, dtype=np.float64
        ),
        "source_family": families,
        "n_segments": np.asarray(segment_family.size, dtype=np.int64),
        "n_sessions": np.asarray(segment_session.size, dtype=np.int64),
        "n_groups": np.asarray(segment_group.size, dtype=np.int64),
        "segment_session_id": segment_session,
        "segment_source_family": segment_family,
        "segment_group_id": segment_group,
        "segment_start_sample": np.full(
            segment_family.size, edge_trim_samples, dtype=np.int64
        ),
        "segment_recorded_lead_samples": np.zeros(
            segment_family.size, dtype=np.int64
        ),
        "segment_recorded_delay_samples": np.zeros(
            segment_family.size, dtype=np.float64
        ),
        "segment_timing_contract_sha256": np.asarray(
            [timing.digest()] * segment_family.size, dtype=np.str_
        ),
        "segment_source_timeline": np.asarray(
            ["source_aligned.wav"] * segment_family.size, dtype=np.str_
        ),
        "per_segment_trusted_db": np.full(segment_family.size, value, dtype=np.float64),
        "per_segment_fullband_db": np.full(segment_family.size, value, dtype=np.float64),
        "per_segment_gap_db": np.zeros(segment_family.size, dtype=np.float64),
        "per_segment_octave_attenuation_db": octave_values,
        "octave_center_hz": octave_centers,
        "octave_attenuation_mean_db": np.full(octave_centers.size, 1.0, dtype=np.float64),
        "octave_attenuation_median_db": np.full(octave_centers.size, 1.0, dtype=np.float64),
        "octave_attenuation_worst10_mean_db": np.full(octave_centers.size, 1.0, dtype=np.float64),
        "octave_trusted": np.asarray(
            (False, True, True, True, True, False, False, False), dtype=np.bool_
        ),
        "source_n_segments": source_counts,
        "source_n_sessions": source_counts,
        "source_n_groups": source_counts,
        "source_nmse_trusted_mean_db": source_values,
        "source_nmse_trusted_worst10_mean_db": source_values,
        "source_nmse_fullband_mean_db": source_values,
        "source_nmse_fullband_worst10_mean_db": source_values,
        "source_gap_trusted_minus_fullband_mean_db": np.zeros(families.size, dtype=np.float64),
        "per_segment_trusted_subband_nmse_db": strict_values,
        "per_segment_trusted_subband_coverage": np.ones_like(
            strict_values, dtype=np.bool_
        ),
        "per_segment_trusted_subband_source_energy_density_ratio": np.ones_like(
            strict_values, dtype=np.float64
        ),
        "source_trusted_subband_n_segments": np.full(
            strict_shape, groups_per_family, dtype=np.int64
        ),
        "source_trusted_subband_n_groups": np.full(
            strict_shape, groups_per_family, dtype=np.int64
        ),
        "source_trusted_subband_coverage_fraction": np.ones(
            strict_shape, dtype=np.float64
        ),
        "source_trusted_subband_source_energy_density_ratio_mean": np.ones(
            strict_shape, dtype=np.float64
        ),
        "source_trusted_subband_nmse_mean_db": np.full(
            strict_shape, value, dtype=np.float64
        ),
        "source_trusted_subband_nmse_worst10_mean_db": np.full(
            strict_shape, value, dtype=np.float64
        ),
        "source_trusted_subband_ci_lo_db": np.full(
            strict_shape, value if power_pass else np.nan, dtype=np.float64
        ),
        "source_trusted_subband_ci_hi_db": np.full(
            strict_shape, value if power_pass else np.nan, dtype=np.float64
        ),
        "source_trusted_subband_coverage_pass": strict_flags,
        "source_trusted_subband_power_pass": np.full(
            strict_shape, power_pass, dtype=np.bool_
        ),
        "source_trusted_subband_mean_pass": strict_flags,
        "source_trusted_subband_worst10_pass": strict_flags,
        "source_trusted_subband_ci_pass": np.full(
            strict_shape, power_pass, dtype=np.bool_
        ),
        "source_trusted_subband_pass": np.full(
            strict_shape, power_pass, dtype=np.bool_
        ),
        "g4_trusted_subband_schema_pass": np.asarray(True),
        "g4_trusted_subband_coverage_pass": np.asarray(True),
        "g4_trusted_subband_power_pass": np.asarray(power_pass),
        "g4_trusted_subband_mean_pass": np.asarray(True),
        "g4_trusted_subband_worst10_pass": np.asarray(True),
        "g4_trusted_subband_ci_pass": np.asarray(power_pass),
        "g4_trusted_subband_pass": np.asarray(power_pass),
        "g4_upper_trusted_subband_pass": np.asarray(power_pass),
        "split": np.asarray(split),
        "checkpoint_sha256": np.asarray(pipeline._sha256_file(checkpoint)),
        "manifest_sha256": np.asarray(pipeline._sha256_file(manifest)),
        "experiment_contract_sha256": np.asarray(contract_sha),
        "selection_sha256": np.asarray(selection_sha256),
        "test_capability_sha256": np.asarray(test_capability_sha256),
        "test_consumed_marker_sha256": np.asarray(
            test_consumed_marker_sha256
        ),
    }


def _make_val_selection(
    tmp_path: Path,
    *,
    seed: int = 20260803,
    margin_db: float = 1.0,
    name: str = "run",
    manifest: Path | None = None,
    manifest_splits: tuple[str, ...] = ("val",),
    source_families: tuple[str, ...] = (
        "speech",
        "music",
        "environment",
        "machine",
    ),
    groups_per_family: int = 4,
) -> tuple[Path, dict, Path]:
    if manifest is None:
        manifest = tmp_path / "recorded.jsonl"
    _write_canonical_recorded_manifest(
        manifest,
        splits=manifest_splits,
        source_families=source_families,
        groups_per_family=groups_per_family,
    )
    stamped = stamp_experiment_contract(
        _canonical_sampling_checkpoint_cfg(**{
            "experiment_role": "selection_test",
            "seed": seed,
            "recorded_manifest": str(manifest.absolute()),
        }),
        repo_root=tmp_path,
    )
    checkpoint = tmp_path / f"{name}-{seed}.pt"
    torch.save({"cfg": stamped, "model": {"weight": torch.ones(1)}}, checkpoint)
    evaluation = tmp_path / f"val-{name}-{seed}"
    evaluation.mkdir()
    np.savez_compressed(
        evaluation / "metrics.npz",
        **_recorded_val_metric_payload(
            checkpoint=checkpoint,
            manifest=manifest,
            contract_sha=stamped["experiment_contract_sha256"],
            margin_db=margin_db,
            manifest_splits=None,
            source_families=source_families,
            groups_per_family=groups_per_family,
        ),
    )
    selection_path = tmp_path / f"selection-{name}-{seed}.json"
    selection = pipeline.freeze_recorded_val_selection(
        candidates=[(checkpoint, evaluation)],
        selection_path=selection_path,
        manifest_path=manifest,
        experiment_contract_sha256=stamped["experiment_contract_sha256"],
    )
    return selection_path, selection, manifest


def _make_running_test_evaluation(
    tmp_path: Path,
) -> tuple[Path, dict, Path, Path, Path, Path, dict]:
    """canonical val selection을 소비하고 valid test output 직전 상태를 만든다."""

    selection_path, selection, manifest = _make_val_selection(
        tmp_path,
        name="completion",
        manifest_splits=("val", "test"),
    )
    capability, consumed = canonical_test_ledger_paths(
        selection_path, repo_root=tmp_path
    )
    token = issue_test_capability(
        selection_path=selection_path,
        capability_path=capability,
        repo_root=tmp_path,
    )
    checkpoint = Path(selection["selected"]["checkpoint"])
    consume_test_capability(
        selection_path=selection_path,
        capability_path=capability,
        consumed_marker_path=consumed,
        token=token,
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        repo_root=tmp_path,
    )
    output = tmp_path / "eval-recorded-test"
    output.mkdir()
    (output / "metrics.md").write_text("ok\n", encoding="utf-8")
    metrics_payload = _recorded_val_metric_payload(
        checkpoint=checkpoint,
        manifest=manifest,
        contract_sha=selection["experiment_contract_sha256"],
        margin_db=1.0,
        split="test",
        manifest_splits=None,
        selection_sha256=pipeline._sha256_file(selection_path),
        test_capability_sha256=pipeline._sha256_file(capability),
        test_consumed_marker_sha256=pipeline._sha256_file(consumed),
    )
    return (
        selection_path,
        selection,
        manifest,
        capability,
        consumed,
        output,
        metrics_payload,
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """저장소를 tmp 로 복제하고 **REPO_ROOT 를 들고 있는 모든 모듈**을 교체한다.

    ``from ..config import REPO_ROOT`` 는 값을 복사해 가므로 config 모듈만 바꿔서는
    소용이 없다. 하나라도 빠지면 그 모듈은 진짜 저장소를 읽고, 부모가 보는 state dir 과
    자식이 train.lock 을 쓰는 dir 이 어긋난다 — 실제로 고쳤던 결함이다.

    특히 ``finetune_readiness`` 가 빠져 있었다. 그래서 이 테스트는 tmp 를 본다고 믿으면서
    실제로는 저장소의 data/ 와 runs/ 를 읽고 있었고, 저장소가 NOT READY 인 동안에만
    우연히 통과했다.
    """

    shutil.copytree(REPO_ROOT / "configs", tmp_path / "configs")
    shutil.copytree(REPO_ROOT / "assets", tmp_path / "assets")
    data_config = tmp_path / "configs" / "data_sim.yaml"
    data_config.write_text(
        data_config.read_text(encoding="utf-8").replace(
            "bootstrap_receipt_sha256: null",
            f"bootstrap_receipt_sha256: {'a' * 64}",
        ),
        encoding="utf-8",
    )
    rir_bank = tmp_path / "data" / "rir_bank" / "duct_rirs_v1.npz"
    rir_bank.parent.mkdir(parents=True)
    fixture_rirs = np.zeros((2, 64), dtype=np.float32)
    fixture_rirs[:, 0] = 1.0
    np.savez(
        rir_bank,
        p_ref=fixture_rirs,
        p_err=fixture_rirs,
        f_fb=fixture_rirs,
    )
    (tmp_path / ".gitignore").write_text("/results/\n/runs/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "add", "configs", "data", ".gitignore"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    import deep_anc.config as config_module
    import deep_anc.data.transfer_contract as transfer_contract_module
    import deep_anc.train.finetune_readiness as readiness_module
    import deep_anc.train.process_lock as lock_module

    for module in (config_module, lock_module, readiness_module, pipeline):
        monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        transfer_contract_module,
        "bind_recorded_transfer_config",
        lambda data, repo_root: data.update(
            transfer_manifest="data/manifests/elice_transfer_manifest.json",
            transfer_manifest_sha256="b" * 64,
            recorded_transfer_aggregate_sha256="c" * 64,
        ),
    )
    monkeypatch.setattr(
        readiness_module,
        "validate_recorded_training_snapshot",
        lambda data, repo_root: (_ for _ in ()).throw(
            ValueError("fixture에는 recorded transfer bytes가 없습니다")
        ),
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _state_dir(repo_root: Path) -> Path:
    return autostart_state_dir(_resolved_run_dir(repo_root))


def _resolved_run_dir(repo_root: Path) -> Path:
    from deep_anc.config import load_train_config

    cfg = load_train_config(
        repo_root / "configs" / "train_finetune.yaml",
        ["data.digital_primary_path_mode=measured"],
    )
    return resolve_run_dir(cfg["ckpt_dir"])


# ---------------------------------------------------------------------------
# 불변식 1 — NOT READY 는 runs/ 를 만들지 않는다
# ---------------------------------------------------------------------------


def test_not_ready_never_creates_run_dir_and_writes_only_state_dir(repo):
    assert pipeline.main(ARGS) == EXIT_NOT_READY

    assert not _resolved_run_dir(repo).exists(), (
        "NOT READY 인데 학습 디렉터리가 생겼다 — 구버전 결함이 되살아났다"
    )
    state = _state_dir(repo)
    assert (state / "audit" / "readiness.json").is_file()
    assert (state / "audit" / "readiness.md").is_file()
    assert (state / "status.json").is_file()
    assert not list(state.rglob("*.tmp"))

    status = read_status(state / "status.json")
    assert status["phase"] == "not_ready"
    assert status["exit_code"] == EXIT_NOT_READY
    assert status["advisory"] is True
    assert status["run_dir_created_before_ready"] is False
    # fixture 가 data/ 와 runs/ 를 복제하지 않으므로 이 둘은 반드시 실패한다.
    # 특정 검사 이름을 고를 때는 **fixture 가 보장하는 것**을 골라야 한다 —
    # 예전에는 official_primary_path 를 골랐는데, 그건 저장소가 아직 실측을 안 했다는
    # 일시적 사실에 기댄 것이라 실측이 끝나자 깨졌다.
    failed = status["readiness"]["failed_checks"]
    assert "recorded_dataset_qa" in failed
    assert "completed_init_checkpoint" in failed


def test_audit_lands_in_autostart_state_dir_not_run_dir(repo):
    pipeline.main(ARGS)
    assert (_state_dir(repo) / "audit" / "readiness.json").is_file()
    assert not (_resolved_run_dir(repo) / "audit").exists()


# ---------------------------------------------------------------------------
# --state-dir
# ---------------------------------------------------------------------------


def test_state_dir_relocates_audit_but_never_the_lock(repo):
    """lock 위치까지 옮겨지면 --state-dir 하나로 상호배제를 우회할 수 있다."""

    custom = repo / "results" / "custom_audit"
    assert pipeline.main([*ARGS, "--state-dir", str(custom)]) == EXIT_NOT_READY
    assert (custom / "audit" / "readiness.json").is_file()
    assert (custom / "status.json").is_file()
    # lock 은 canonical 경로에 그대로 있어야 한다.
    assert (_state_dir(repo) / "pipeline.lock").exists()
    assert not (custom / "pipeline.lock").exists()


def test_state_dir_outside_repo_is_rejected(repo, tmp_path):
    outside = tmp_path.parent / "outside_state"
    assert pipeline.main([*ARGS, "--state-dir", str(outside)]) == EXIT_CONFIG
    assert not outside.exists()


# ---------------------------------------------------------------------------
# 불변식 2 — 중복 실행 구분
# ---------------------------------------------------------------------------


def test_second_pipeline_exits_4_and_leaves_state_untouched(repo, monkeypatch):
    pipeline.main(ARGS)
    state = _state_dir(repo)
    status_path = state / "status.json"
    before = status_path.read_bytes()

    calls = []
    monkeypatch.setattr(
        pipeline, "audit_finetune_readiness", lambda cfg: calls.append(cfg) or {}
    )
    held = ProcessLock(state / "pipeline.lock", role="fine-tune pipeline").acquire()
    try:
        assert pipeline.main(ARGS) == EXIT_PIPELINE_LOCKED
    finally:
        held.release()

    # 패자는 흔적을 남기지 않는다 — 무거운 QA 도 돌리지 않는다.
    assert calls == []
    assert status_path.read_bytes() == before


def test_check_only_is_also_protected_by_the_pipeline_lock(repo):
    state = _state_dir(repo)
    state.mkdir(parents=True, exist_ok=True)
    held = ProcessLock(state / "pipeline.lock", role="fine-tune pipeline").acquire()
    try:
        assert pipeline.main(ARGS) == EXIT_PIPELINE_LOCKED
    finally:
        held.release()


def test_pipeline_lock_and_train_lock_are_distinct_paths(repo):
    """같은 파일이면 부모가 잡은 flock 때문에 자식 rank0 이 항상 탈락한다(자기 데드락)."""

    state = _state_dir(repo)
    assert (state / "pipeline.lock") != (state / "train.lock")
    pipeline.main(ARGS)
    assert (state / "pipeline.lock").exists()
    assert not (state / "train.lock").exists()


def test_status_flag_reads_without_taking_the_lock(repo, capsys):
    pipeline.main(ARGS)
    state = _state_dir(repo)
    held = ProcessLock(state / "pipeline.lock", role="fine-tune pipeline").acquire()
    try:
        assert pipeline.main(["--status", "--config", "configs/train_finetune.yaml"]) == EXIT_OK
    finally:
        held.release()
    output = capsys.readouterr().out
    assert "phase=not_ready" in output
    assert "pipeline.lock 보유중=True" in output


# ---------------------------------------------------------------------------
# 불변식 3 — status 는 advisory
# ---------------------------------------------------------------------------


def test_forged_status_does_not_weaken_the_gates(repo):
    """status.json 을 '완료'로 위조해도 여전히 NOT READY 여야 한다."""

    pipeline.main(ARGS)
    status_path = _state_dir(repo) / "status.json"
    atomic_write_text(
        status_path,
        json.dumps({"phase": "done", "exit_code": 0, "completion": {"ok": True}}) + "\n",
    )
    assert pipeline.main(ARGS) == EXIT_NOT_READY
    assert read_status(status_path)["phase"] == "not_ready"


def test_repeated_runs_are_idempotent(repo):
    """재실행이 상태를 파괴하지 않고 같은 판정으로 수렴한다."""

    assert pipeline.main(ARGS) == EXIT_NOT_READY
    status_path = _state_dir(repo) / "status.json"
    first_status = stable_view(read_status(status_path))
    first_audit = json.loads(
        (_state_dir(repo) / "audit" / "readiness.json").read_text(encoding="utf-8")
    )

    assert pipeline.main(ARGS) == EXIT_NOT_READY
    second_status = stable_view(read_status(status_path))
    second_audit = json.loads(
        (_state_dir(repo) / "audit" / "readiness.json").read_text(encoding="utf-8")
    )

    assert first_status == second_status
    first_audit.pop("checked_at_utc", None)
    second_audit.pop("checked_at_utc", None)
    assert first_audit == second_audit
    assert not _resolved_run_dir(repo).exists()
    assert not list(_state_dir(repo).rglob("*.tmp"))


# ---------------------------------------------------------------------------
# process_lock 규약
# ---------------------------------------------------------------------------


def test_run_key_is_stable_and_path_sensitive(repo):
    key = finetune_run_key(repo / "runs" / "finetune_tiny")
    assert key == finetune_run_key("runs/finetune_tiny")
    assert key != finetune_run_key("runs/finetune_other")
    assert all(c.isalnum() or c in "-_" for c in key)


def test_pipeline_and_train_agree_on_the_state_dir(repo):
    """train.py 는 autostart_state_dir(resolve_run_dir(cfg['ckpt_dir'])) 를 쓴다."""

    from deep_anc.config import load_train_config

    cfg = load_train_config(
        repo / "configs" / "train_finetune.yaml",
        ["data.digital_primary_path_mode=measured"],
    )
    assert autostart_state_dir(resolve_run_dir(cfg["ckpt_dir"])) == _state_dir(repo)


def test_canonical_pretrain_also_requires_the_same_run_lock():
    assert train_entry.requires_same_run_lock(
        {"experiment_role": "canonical_pretrain"}
    )
    assert train_entry.requires_same_run_lock(
        {"experiment_role": "canonical_finetune"}
    )
    assert train_entry.requires_same_run_lock(
        {"experiment_role": "a100_pretrain_smoke"}
    )
    assert train_entry.requires_same_run_lock({"experiment_role": "loss_pilot"})
    assert train_entry.requires_same_run_lock({"experiment_role": "measured_probe"})
    assert not train_entry.requires_same_run_lock(
        {"experiment_role": "diagnostic_overfit"}
    )


def test_existing_last_is_never_auto_resumed_and_requires_explicit_last(repo):
    ckpt_dir = repo / "runs" / "finetune_tiny" / "ckpt"
    ckpt_dir.mkdir(parents=True)
    last = ckpt_dir / "last.pt"
    last.write_bytes(b"legacy")

    with pytest.raises(ValueError, match="--resume"):
        pipeline._resolve_explicit_resume(None, ckpt_dir)
    assert pipeline._resolve_explicit_resume(str(last), ckpt_dir) == last

    best = ckpt_dir / "best.pt"
    best.write_bytes(b"best")
    with pytest.raises(ValueError, match="last.pt"):
        pipeline._resolve_explicit_resume(str(best), ckpt_dir)


def test_recorded_val_selection_is_frozen_before_test_and_test_opens_once(tmp_path):
    manifest = tmp_path / "recorded.jsonl"
    _write_canonical_recorded_manifest(manifest, splits=("val",))
    candidates = []
    stamped = stamp_experiment_contract(
        _canonical_sampling_checkpoint_cfg(**{
            "experiment_role": "selection_test",
            "seed": 20260803,
            "recorded_manifest": str(manifest.absolute()),
        }),
        repo_root=tmp_path,
    )
    contract_sha = stamped["experiment_contract_sha256"]
    for name, metric in (("best", -2.0), ("last", -4.0)):
        checkpoint = tmp_path / f"{name}.pt"
        torch.save({"cfg": stamped, "model": {"weight": torch.ones(1)}}, checkpoint)
        evaluation = tmp_path / f"val-{name}"
        evaluation.mkdir()
        np.savez_compressed(
            evaluation / "metrics.npz",
            **_recorded_val_metric_payload(
                checkpoint=checkpoint,
                manifest=manifest,
                contract_sha=contract_sha,
                margin_db=-metric,
                manifest_splits=None,
            ),
        )
        candidates.append((checkpoint, evaluation))

    selection_path = tmp_path / "selection.json"
    selection = pipeline.freeze_recorded_val_selection(
        candidates=candidates,
        selection_path=selection_path,
        manifest_path=manifest,
        experiment_contract_sha256=contract_sha,
    )
    assert Path(selection["selected"]["checkpoint"]).name == "last.pt"
    assert selection["selection_split"] == "val"
    with pytest.raises(FileExistsError, match="이미 있어"):
        pipeline.freeze_recorded_val_selection(
            candidates=candidates,
            selection_path=selection_path,
            manifest_path=manifest,
            experiment_contract_sha256=contract_sha,
        )

    capability, consumed = canonical_test_ledger_paths(
        selection_path, repo_root=tmp_path
    )
    with pytest.raises(ValueError, match="canonical ledger"):
        issue_test_capability(
            selection_path=selection_path,
            capability_path=tmp_path / "alternate-capability.json",
            repo_root=tmp_path,
        )
    token = issue_test_capability(
        selection_path=selection_path, capability_path=capability, repo_root=tmp_path
    )
    consume_test_capability(
        selection_path=selection_path,
        capability_path=capability,
        consumed_marker_path=consumed,
        token=token,
        checkpoint_path=selection["selected"]["checkpoint"],
        manifest_path=manifest,
        repo_root=tmp_path,
    )
    assert consumed.is_file()
    selection_copy = tmp_path / "selection-copy.json"
    from deep_anc.train.evaluation_contract import write_json_exclusive

    write_json_exclusive(selection_copy, selection)
    with pytest.raises(FileExistsError, match="발급/소비"):
        issue_test_capability(
            selection_path=selection_copy,
            capability_path=capability,
            repo_root=tmp_path,
        )
    with pytest.raises(FileExistsError):
        consume_test_capability(
            selection_path=selection_path,
            capability_path=capability,
            consumed_marker_path=consumed,
            token=token,
            checkpoint_path=selection["selected"]["checkpoint"],
            manifest_path=manifest,
            repo_root=tmp_path,
        )


def test_test_capability_rejects_borderline_and_uses_campaign_wide_ledger(tmp_path):
    borderline_path, borderline, manifest = _make_val_selection(
        tmp_path, margin_db=0.1, name="borderline"
    )
    assert borderline["decision"]["status"] == "borderline"
    capability, _ = canonical_test_ledger_paths(
        borderline_path, repo_root=tmp_path
    )
    with pytest.raises(ValueError, match="clear-pass"):
        issue_test_capability(
            selection_path=borderline_path,
            capability_path=capability,
            repo_root=tmp_path,
        )

    alternate = dict(borderline)
    alternate["experiment_contract_sha256"] = "d" * 64
    alternate["selected"] = dict(borderline["selected"])
    alternate["selected"]["checkpoint_sha256"] = "e" * 64
    alternate_path = tmp_path / "alternate-selection.json"
    from deep_anc.train.evaluation_contract import write_json_exclusive

    write_json_exclusive(alternate_path, alternate)
    assert canonical_test_ledger_paths(
        alternate_path, repo_root=tmp_path
    ) == canonical_test_ledger_paths(borderline_path, repo_root=tmp_path)


def test_test_ledger_phases_and_atomic_directory_publication(tmp_path):
    selection_path, selection, manifest = _make_val_selection(
        tmp_path,
        name="phase",
        manifest_splits=("val", "test"),
    )
    capability, consumed = canonical_test_ledger_paths(
        selection_path, repo_root=tmp_path
    )
    token = issue_test_capability(
        selection_path=selection_path,
        capability_path=capability,
        repo_root=tmp_path,
    )
    consume_test_capability(
        selection_path=selection_path,
        capability_path=capability,
        consumed_marker_path=consumed,
        token=token,
        checkpoint_path=selection["selected"]["checkpoint"],
        manifest_path=manifest,
        repo_root=tmp_path,
    )
    selection_sha = pipeline._sha256_file(selection_path)
    capability_sha = pipeline._sha256_file(capability)
    consumed_sha = pipeline._sha256_file(consumed)
    staging = tmp_path / ".eval.staging"
    staging.mkdir()
    (staging / "metrics.md").write_text("ok\n")
    np.savez_compressed(
        staging / "metrics.npz",
        **_recorded_val_metric_payload(
            checkpoint=Path(selection["selected"]["checkpoint"]),
            manifest=manifest,
            contract_sha=selection["experiment_contract_sha256"],
            margin_db=1.0,
            split="test",
            manifest_splits=None,
            selection_sha256=selection_sha,
            test_capability_sha256=capability_sha,
            test_consumed_marker_sha256=consumed_sha,
        ),
    )
    target = tmp_path / "eval_recorded_test"
    publish_directory_noreplace(staging, target)
    assert target.is_dir() and not staging.exists()
    completed = complete_test_evaluation(
        selection_path=selection_path,
        capability_path=capability,
        consumed_marker_path=consumed,
        output_dir=target,
        repo_root=tmp_path,
    )
    assert completed.is_file()
    event_paths = canonical_test_ledger_event_paths_from_payload(
        selection, repo_root=tmp_path
    )
    assert completed == event_paths["completed"]
    assert not event_paths["failed"].exists()
    completed_payload = json.loads(completed.read_text(encoding="utf-8"))
    assert completed_payload["phase"] == "completed"
    assert completed_payload["g4_verdict"] == "PASS"
    second_staging = tmp_path / ".eval.second.staging"
    second_staging.mkdir()
    with pytest.raises(FileExistsError, match="덮어쓸 수 없습니다"):
        publish_directory_noreplace(second_staging, target)
    assert second_staging.is_dir()
    with pytest.raises(FileExistsError, match="phase"):
        consume_test_capability(
            selection_path=selection_path,
            capability_path=capability,
            consumed_marker_path=consumed,
            token=token,
            checkpoint_path=selection["selected"]["checkpoint"],
            manifest_path=manifest,
            repo_root=tmp_path,
        )


@pytest.mark.parametrize(
    "provenance_field",
    [
        "checkpoint_sha256",
        "experiment_contract_sha256",
        "selection_sha256",
        "test_capability_sha256",
        "test_consumed_marker_sha256",
    ],
)
def test_test_completion_rejects_each_tampered_metrics_provenance(
    tmp_path, provenance_field
):
    """NPZ가 one-shot ledger의 다섯 identity 중 하나라도 바꾸면 완료하지 않는다."""

    (
        selection_path,
        selection,
        _manifest,
        capability,
        consumed,
        output,
        metrics_payload,
    ) = _make_running_test_evaluation(tmp_path)
    metrics_payload[provenance_field] = np.asarray("f" * 64)
    np.savez_compressed(output / "metrics.npz", **metrics_payload)
    completed = canonical_test_ledger_event_paths_from_payload(
        selection, repo_root=tmp_path
    )["completed"]

    with pytest.raises(ValueError, match=rf"metrics {provenance_field}.*one-shot"):
        complete_test_evaluation(
            selection_path=selection_path,
            capability_path=capability,
            consumed_marker_path=consumed,
            output_dir=output,
            repo_root=tmp_path,
        )
    assert not completed.exists()


@pytest.mark.parametrize(
    ("running_field", "metrics_field"),
    [
        ("seed_neutral_campaign_sha256", None),
        ("experiment_contract_sha256", "experiment_contract_sha256"),
        ("selected_checkpoint_sha256", "checkpoint_sha256"),
        ("manifest_sha256", "manifest_sha256"),
    ],
)
def test_test_completion_rejects_running_marker_synced_with_metrics(
    tmp_path, running_field, metrics_field
):
    """running과 NPZ를 함께 위조해도 immutable selection을 우회할 수 없다."""

    (
        selection_path,
        selection,
        _manifest,
        capability,
        consumed,
        output,
        metrics_payload,
    ) = _make_running_test_evaluation(tmp_path)
    running = json.loads(consumed.read_text(encoding="utf-8"))
    forged_identity = "f" * 64
    running[running_field] = forged_identity
    consumed.write_text(
        json.dumps(
            running,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    if metrics_field is not None:
        metrics_payload[metrics_field] = np.asarray(forged_identity)
    metrics_payload["test_consumed_marker_sha256"] = np.asarray(
        pipeline._sha256_file(consumed)
    )
    np.savez_compressed(output / "metrics.npz", **metrics_payload)
    completed = canonical_test_ledger_event_paths_from_payload(
        selection, repo_root=tmp_path
    )["completed"]

    with pytest.raises(
        ValueError,
        match=rf"running marker {running_field}.*immutable selection",
    ):
        complete_test_evaluation(
            selection_path=selection_path,
            capability_path=capability,
            consumed_marker_path=consumed,
            output_dir=output,
            repo_root=tmp_path,
        )
    assert not completed.exists()


def test_val_boundary_within_point_three_db_never_becomes_clear_pass(tmp_path):
    checkpoint = tmp_path / "best.pt"
    manifest = tmp_path / "recorded.jsonl"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text("{}\n")
    payload = _recorded_val_metric_payload(
        checkpoint=checkpoint,
        manifest=manifest,
        contract_sha="a" * 64,
        margin_db=0.3,
    )
    buffer = tmp_path / "metrics.npz"
    np.savez_compressed(buffer, **payload)
    decision = pipeline.classify_recorded_val_metrics(
        buffer.read_bytes(),
        manifest_bytes=manifest.read_bytes(),
        manifest_path=manifest,
        checkpoint_cfg=_canonical_sampling_checkpoint_cfg(),
    )
    assert decision["status"] == "borderline"
    assert decision["minimum_margin_db"] == pytest.approx(0.3)


def _set_numeric_strict_subband_failure(
    payload: dict[str, np.ndarray], *, value_db: float
) -> None:
    """coverage/group 수는 충분한 한 부대역을 실제 dB 실패로 바꾼다."""

    family_index = 0
    band_index = len(STRICT_TRUSTED_SUBBANDS_HZ) - 1
    family = str(np.asarray(payload["source_family"])[family_index])
    family_mask = np.asarray(payload["segment_source_family"]).astype(str) == family
    payload["per_segment_trusted_subband_nmse_db"][family_mask, band_index] = value_db
    for key in (
        "source_trusted_subband_nmse_mean_db",
        "source_trusted_subband_nmse_worst10_mean_db",
        "source_trusted_subband_ci_lo_db",
        "source_trusted_subband_ci_hi_db",
    ):
        payload[key][family_index, band_index] = value_db
    for key in (
        "source_trusted_subband_mean_pass",
        "source_trusted_subband_worst10_pass",
        "source_trusted_subband_ci_pass",
        "source_trusted_subband_pass",
    ):
        # fixture의 초기 PASS boolean 배열은 여러 key가 같은 ndarray를 공유한다.
        # 한 summary를 바꾸며 coverage/power까지 우연히 바꾸지 않도록 분리한다.
        updated = np.array(payload[key], copy=True)
        updated[family_index, band_index] = False
        payload[key] = updated
    for key in (
        "g4_trusted_subband_mean_pass",
        "g4_trusted_subband_worst10_pass",
        "g4_trusted_subband_ci_pass",
        "g4_trusted_subband_pass",
        "g4_upper_trusted_subband_pass",
    ):
        payload[key] = np.asarray(False)
    payload["g4_pass"] = np.asarray(False)
    payload["g4_verdict"] = np.asarray("FAIL")


def _set_strict_subband_inconclusive(payload: dict[str, np.ndarray]) -> None:
    """한 family/subband의 실제 coverage를 제거해 valid INCONCLUSIVE를 만든다."""

    family_index = 0
    band_index = len(STRICT_TRUSTED_SUBBANDS_HZ) - 1
    family = str(np.asarray(payload["source_family"])[family_index])
    family_mask = np.asarray(payload["segment_source_family"]).astype(str) == family
    coverage = np.array(
        payload["per_segment_trusted_subband_coverage"], copy=True
    )
    coverage[family_mask, band_index] = False
    payload["per_segment_trusted_subband_coverage"] = coverage
    density = np.array(
        payload["per_segment_trusted_subband_source_energy_density_ratio"],
        copy=True,
    )
    density[family_mask, band_index] = 0.0
    payload["per_segment_trusted_subband_source_energy_density_ratio"] = density
    for key in (
        "source_trusted_subband_n_segments",
        "source_trusted_subband_n_groups",
    ):
        updated = np.array(payload[key], copy=True)
        updated[family_index, band_index] = 0
        payload[key] = updated
    for key in (
        "source_trusted_subband_coverage_fraction",
        "source_trusted_subband_source_energy_density_ratio_mean",
    ):
        updated = np.array(payload[key], copy=True)
        updated[family_index, band_index] = 0.0
        payload[key] = updated
    for key in (
        "source_trusted_subband_nmse_mean_db",
        "source_trusted_subband_nmse_worst10_mean_db",
        "source_trusted_subband_ci_lo_db",
        "source_trusted_subband_ci_hi_db",
    ):
        updated = np.array(payload[key], copy=True)
        updated[family_index, band_index] = np.nan
        payload[key] = updated
    for key in (
        "source_trusted_subband_coverage_pass",
        "source_trusted_subband_power_pass",
        "source_trusted_subband_mean_pass",
        "source_trusted_subband_worst10_pass",
        "source_trusted_subband_ci_pass",
        "source_trusted_subband_pass",
    ):
        updated = np.array(payload[key], copy=True)
        updated[family_index, band_index] = False
        payload[key] = updated
    for key in (
        "g4_trusted_subband_coverage_pass",
        "g4_trusted_subband_power_pass",
        "g4_trusted_subband_mean_pass",
        "g4_trusted_subband_worst10_pass",
        "g4_trusted_subband_ci_pass",
        "g4_trusted_subband_pass",
        "g4_upper_trusted_subband_pass",
        "g4_pass",
    ):
        payload[key] = np.asarray(False)
    payload["g4_verdict"] = np.asarray("INCONCLUSIVE")


@pytest.mark.parametrize("verdict", ["FAIL", "INCONCLUSIVE"])
def test_valid_nonpass_test_g4_is_terminal_rejection_and_raw_is_immutable(
    tmp_path, verdict
):
    """valid non-PASS raw는 보존하지만 completed/readiness를 절대 열지 않는다."""

    (
        selection_path,
        selection,
        _manifest,
        capability,
        consumed,
        output,
        metrics_payload,
    ) = _make_running_test_evaluation(tmp_path)
    if verdict == "FAIL":
        _set_numeric_strict_subband_failure(metrics_payload, value_db=1.0)
    else:
        _set_strict_subband_inconclusive(metrics_payload)
    metrics_path = output / "metrics.npz"
    np.savez_compressed(metrics_path, **metrics_payload)
    metrics_sha = pipeline._sha256_file(metrics_path)
    paths = canonical_test_ledger_event_paths_from_payload(
        selection, repo_root=tmp_path
    )

    terminal = complete_test_evaluation(
        selection_path=selection_path,
        capability_path=capability,
        consumed_marker_path=consumed,
        output_dir=output,
        repo_root=tmp_path,
    )

    assert terminal == paths["failed"]
    assert paths["failed"].is_file()
    assert not paths["completed"].exists()
    marker = json.loads(paths["failed"].read_text(encoding="utf-8"))
    assert marker["phase"] == "failed"
    assert marker["g4_verdict"] == verdict
    assert marker["error_type"] == f"G4_{verdict}"
    assert marker["failure_class"] == "valid_g4_terminal_rejection"
    assert marker["metrics_npz_sha256"] == metrics_sha
    marker_sha = pipeline._sha256_file(paths["failed"])
    assert pipeline._sha256_file(metrics_path) == metrics_sha

    with pytest.raises(FileExistsError):
        complete_test_evaluation(
            selection_path=selection_path,
            capability_path=capability,
            consumed_marker_path=consumed,
            output_dir=output,
            repo_root=tmp_path,
        )
    assert pipeline._sha256_file(paths["failed"]) == marker_sha
    assert pipeline._sha256_file(metrics_path) == metrics_sha
    assert not paths["completed"].exists()


def test_numeric_strict_subband_failure_within_margin_is_borderline(tmp_path):
    checkpoint = tmp_path / "best.pt"
    manifest = tmp_path / "recorded.jsonl"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text("{}\n")
    payload = _recorded_val_metric_payload(
        checkpoint=checkpoint,
        manifest=manifest,
        contract_sha="a" * 64,
        margin_db=1.0,
    )
    _set_numeric_strict_subband_failure(payload, value_db=0.1)
    metrics = tmp_path / "strict-borderline.npz"
    np.savez_compressed(metrics, **payload)

    decision = pipeline.classify_recorded_val_metrics(
        metrics.read_bytes(),
        manifest_bytes=manifest.read_bytes(),
        manifest_path=manifest,
        checkpoint_cfg=_canonical_sampling_checkpoint_cfg(),
    )
    assert decision["status"] == "borderline"
    assert decision["minimum_margin_db"] == pytest.approx(-0.1)
    assert decision["margins_db"]["strict_subband_mean_db"] == pytest.approx(-0.1)


def test_unrelated_boundary_does_not_hide_a_clear_required_gate_failure(tmp_path):
    checkpoint = tmp_path / "best.pt"
    manifest = tmp_path / "recorded.jsonl"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text("{}\n")
    payload = _recorded_val_metric_payload(
        checkpoint=checkpoint,
        manifest=manifest,
        contract_sha="a" * 64,
        margin_db=1.0,
    )
    _set_numeric_strict_subband_failure(payload, value_db=0.1)

    # strict 한 구간은 -0.1 dB margin이지만 trusted/source 필수 gate는
    # -2 dB로 명확히 실패한다. 이때 second seed로 우회하면 안 된다.
    trusted_failure_db = 2.0
    fullband_db = -1.0
    gap_db = trusted_failure_db - fullband_db
    payload["per_segment_trusted_db"][:] = trusted_failure_db
    payload["per_segment_gap_db"][:] = gap_db
    for key in (
        "nmse_trusted_worst10_mean_db",
        "nmse_trusted_mean_db",
        "nmse_trusted_median_db",
        "g4_worst_source_trusted_mean_db",
        "g4_worst_source_trusted_worst10_db",
    ):
        payload[key] = np.asarray(trusted_failure_db)
    payload["nmse_gap_trusted_minus_fullband_mean_db"] = np.asarray(gap_db)
    for key in (
        "source_nmse_trusted_mean_db",
        "source_nmse_trusted_worst10_mean_db",
        "source_trusted_ci_lo_db",
        "source_trusted_ci_hi_db",
    ):
        payload[key] = np.full(
            np.asarray(payload[key]).shape,
            trusted_failure_db,
            dtype=np.float64,
        )
    payload["source_gap_trusted_minus_fullband_mean_db"][:] = gap_db
    for key in ("g4_trusted_pass", "g4_source_pass", "g4_ci_pass", "g4_pass"):
        payload[key] = np.asarray(False)
    payload["g4_verdict"] = np.asarray("FAIL")
    metrics = tmp_path / "clear-fail-with-boundary.npz"
    np.savez_compressed(metrics, **payload)

    decision = pipeline.classify_recorded_val_metrics(
        metrics.read_bytes(),
        manifest_bytes=manifest.read_bytes(),
        manifest_path=manifest,
        checkpoint_cfg=_canonical_sampling_checkpoint_cfg(),
    )
    assert decision["status"] == "clear_fail"
    assert decision["minimum_margin_db"] == pytest.approx(-2.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "recorded_sampling_max_segments_per_session",
            np.asarray(1, dtype=np.int64),
            "max_segments_per_session.*64",
        ),
        (
            "recorded_sampling_segment_seconds",
            np.asarray(1.0, dtype=np.float64),
            "segment_seconds",
        ),
        (
            "edge_trim_samples",
            np.asarray(0, dtype=np.int64),
            "edge_trim_samples.*0.25",
        ),
    ),
)
def test_canonical_val_classifier_rejects_relaxed_sampling_contract(
    tmp_path, field, value, message
):
    """좋은 구간만 고르는 max/길이/trim 완화는 val selection을 열 수 없다."""

    checkpoint = tmp_path / "best.pt"
    manifest = tmp_path / "recorded.jsonl"
    checkpoint.write_bytes(b"checkpoint")
    payload = _recorded_val_metric_payload(
        checkpoint=checkpoint,
        manifest=manifest,
        contract_sha="a" * 64,
        margin_db=1.0,
    )
    payload[field] = value
    metrics = tmp_path / "metrics.npz"
    np.savez_compressed(metrics, **payload)

    with pytest.raises(ValueError, match=message):
        pipeline.classify_recorded_val_metrics(
            metrics.read_bytes(),
            manifest_bytes=manifest.read_bytes(),
            manifest_path=manifest,
            checkpoint_cfg=_canonical_sampling_checkpoint_cfg(),
        )


def test_canonical_val_classifier_rejects_forged_deterministic_start_set(tmp_path):
    checkpoint = tmp_path / "best.pt"
    manifest = tmp_path / "recorded.jsonl"
    checkpoint.write_bytes(b"checkpoint")
    payload = _recorded_val_metric_payload(
        checkpoint=checkpoint,
        manifest=manifest,
        contract_sha="a" * 64,
        margin_db=1.0,
    )
    payload["segment_start_sample"] = np.asarray(
        payload["segment_start_sample"], dtype=np.int64
    ).copy()
    payload["segment_start_sample"][0] += 128
    metrics = tmp_path / "metrics.npz"
    np.savez_compressed(metrics, **payload)

    with pytest.raises(ValueError, match="expected deterministic start exact set"):
        pipeline.classify_recorded_val_metrics(
            metrics.read_bytes(),
            manifest_bytes=manifest.read_bytes(),
            manifest_path=manifest,
            checkpoint_cfg=_canonical_sampling_checkpoint_cfg(),
        )


def test_canonical_val_rejects_synchronized_population_reduction_via_forged_lead(
    tmp_path,
):
    """raw/summary를 일관되게 줄여도 checkpoint timing 밖 lead는 authority가 아니다."""

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "recorded.jsonl"
    _write_canonical_recorded_manifest(
        manifest,
        splits=("val",),
        duration_s=4.0,
        aligned_lag_median_samples=96_000.0,
    )
    # Payload 자체는 family/group/global summary까지 세션당 segment 1개에 맞춘
    # 일관된 artifact다. 공격자는 lead를 크게 자기진술해 4초 세션이 실제로도
    # 1 segment만 가능한 것처럼 deterministic start 검사를 속이려 한다.
    payload = _recorded_val_metric_payload(
        checkpoint=checkpoint,
        manifest=manifest,
        contract_sha="a" * 64,
        margin_db=1.0,
        manifest_splits=None,
    )
    attack_timing = TrainingTimingContract.derive(
        primary_fir=np.asarray([1.0], dtype=np.float32),
        plant_delays=PlantDelays(
            primary_delay_samples=96_000,
            secondary_delay_samples=0,
            handoff_samples=0,
            sample_rate=48_000,
        ),
    )
    attack_cfg = _canonical_sampling_checkpoint_cfg()
    attack_cfg["data"]["training_timing_contract"] = attack_timing.model_dump()
    payload["primary_delay_samples"] = np.asarray(96_000, dtype=np.int64)
    payload["segment_recorded_delay_samples"] = np.zeros(
        int(payload["n_segments"]), dtype=np.float64
    )
    payload["segment_recorded_lead_samples"] = np.full(
        int(payload["n_segments"]), 96_000, dtype=np.int64
    )
    payload["segment_timing_contract_sha256"] = np.asarray(
        [attack_timing.digest()] * int(payload["n_segments"]), dtype=np.str_
    )
    metrics = tmp_path / "metrics.npz"
    np.savez_compressed(metrics, **payload)

    with pytest.raises(ValueError, match="recorded delay.*session.json timeline"):
        pipeline.classify_recorded_val_metrics(
            metrics.read_bytes(),
            manifest_bytes=manifest.read_bytes(),
            manifest_path=manifest,
            checkpoint_cfg=attack_cfg,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda payload: payload.update(
                feedback_delay_samples=np.asarray(1, dtype=np.int64)
            ),
            "feedback delay.*checkpoint",
        ),
        (
            lambda payload: payload.update(
                warmup_samples=np.asarray(71_520, dtype=np.int64),
                metric_samples_per_segment=np.asarray(416, dtype=np.int64),
            ),
            "warmup.*checkpoint",
        ),
    ),
)
def test_canonical_val_rejects_feedback_and_warmup_cherry_pick(
    tmp_path, mutate, message
):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "recorded.jsonl"
    payload = _recorded_val_metric_payload(
        checkpoint=checkpoint,
        manifest=manifest,
        contract_sha="a" * 64,
        margin_db=1.0,
    )
    mutate(payload)
    metrics = tmp_path / "metrics.npz"
    np.savez_compressed(metrics, **payload)
    with pytest.raises(ValueError, match=message):
        pipeline.classify_recorded_val_metrics(
            metrics.read_bytes(),
            manifest_bytes=manifest.read_bytes(),
            manifest_path=manifest,
            checkpoint_cfg=_canonical_sampling_checkpoint_cfg(),
        )


def test_canonical_val_rejects_checkpoint_hop_and_manifest_rate_mismatch(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "recorded.jsonl"
    payload = _recorded_val_metric_payload(
        checkpoint=checkpoint,
        manifest=manifest,
        contract_sha="a" * 64,
        margin_db=1.0,
    )
    metrics = tmp_path / "metrics.npz"
    np.savez_compressed(metrics, **payload)
    wrong_hop = _canonical_sampling_checkpoint_cfg()
    wrong_hop["model"]["hop"] = 64
    with pytest.raises(ValueError, match="model_hop.*checkpoint"):
        pipeline.classify_recorded_val_metrics(
            metrics.read_bytes(),
            manifest_bytes=manifest.read_bytes(),
            manifest_path=manifest,
            checkpoint_cfg=wrong_hop,
        )

    entries = [json.loads(line) for line in manifest.read_text().splitlines()]
    for entry in entries:
        entry["sample_rate"] = 44_100
    manifest.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    payload["manifest_sha256"] = np.asarray(pipeline._sha256_file(manifest))
    np.savez_compressed(metrics, **payload)
    with pytest.raises(ValueError, match="sample_rate.*evaluator"):
        pipeline.classify_recorded_val_metrics(
            metrics.read_bytes(),
            manifest_bytes=manifest.read_bytes(),
            manifest_path=manifest,
            checkpoint_cfg=_canonical_sampling_checkpoint_cfg(),
        )


def test_test_completion_rejects_resealed_reduced_sampling_population(tmp_path):
    (
        selection_path,
        selection,
        _manifest,
        capability,
        consumed,
        output,
        metrics_payload,
    ) = _make_running_test_evaluation(tmp_path)
    metrics_payload["recorded_sampling_max_segments_per_session"] = np.asarray(
        1, dtype=np.int64
    )
    np.savez_compressed(output / "metrics.npz", **metrics_payload)
    completed = canonical_test_ledger_event_paths_from_payload(
        selection, repo_root=tmp_path
    )["completed"]

    with pytest.raises(ValueError, match="max_segments_per_session.*64"):
        complete_test_evaluation(
            selection_path=selection_path,
            capability_path=capability,
            consumed_marker_path=consumed,
            output_dir=output,
            repo_root=tmp_path,
        )
    assert not completed.exists()


def test_test_completion_rejects_session_timeline_delay_forgery(tmp_path):
    (
        selection_path,
        selection,
        _manifest,
        capability,
        consumed,
        output,
        metrics_payload,
    ) = _make_running_test_evaluation(tmp_path)
    metrics_payload["segment_recorded_delay_samples"] = np.ones(
        int(metrics_payload["n_segments"]), dtype=np.float64
    )
    np.savez_compressed(output / "metrics.npz", **metrics_payload)
    completed = canonical_test_ledger_event_paths_from_payload(
        selection, repo_root=tmp_path
    )["completed"]

    with pytest.raises(ValueError, match="recorded delay.*session.json timeline"):
        complete_test_evaluation(
            selection_path=selection_path,
            capability_path=capability,
            consumed_marker_path=consumed,
            output_dir=output,
            repo_root=tmp_path,
        )
    assert not completed.exists()


def test_recorded_val_classifier_rejects_tampered_strict_density_threshold(tmp_path):
    """metrics 내부 scalar를 낮춰 high-band coverage를 넓히는 우회를 막는다."""

    checkpoint = tmp_path / "best.pt"
    manifest = tmp_path / "recorded.jsonl"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text("{}\n")
    payload = _recorded_val_metric_payload(
        checkpoint=checkpoint,
        manifest=manifest,
        contract_sha="a" * 64,
        margin_db=1.0,
    )
    payload["strict_trusted_subband_min_source_energy_density_ratio"] = np.asarray(
        0.01
    )
    metrics = tmp_path / "metrics.npz"
    np.savez_compressed(metrics, **payload)

    with pytest.raises(ValueError, match="canonical 정책"):
        pipeline.classify_recorded_val_metrics(
            metrics.read_bytes(),
            manifest_bytes=manifest.read_bytes(),
            manifest_path=manifest,
            checkpoint_cfg=_canonical_sampling_checkpoint_cfg(),
        )


def test_recorded_val_classifier_rejects_unlisted_raw_source_family(tmp_path):
    """summary source_family가 raw segment family 하나를 숨길 수 없다."""

    checkpoint = tmp_path / "best.pt"
    manifest = tmp_path / "recorded.jsonl"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text("{}\n")
    payload = _recorded_val_metric_payload(
        checkpoint=checkpoint,
        manifest=manifest,
        contract_sha="a" * 64,
        margin_db=1.0,
    )
    payload["source_family"] = np.asarray(["speech"], dtype=np.str_)
    metrics = tmp_path / "metrics.npz"
    np.savez_compressed(metrics, **payload)

    with pytest.raises(ValueError, match="segment_source_family"):
        pipeline.classify_recorded_val_metrics(
            metrics.read_bytes(),
            manifest_bytes=manifest.read_bytes(),
            manifest_path=manifest,
            checkpoint_cfg=_canonical_sampling_checkpoint_cfg(),
        )


def test_canonical_val_classifier_rejects_missing_required_source_family(tmp_path):
    """selected val split은 필수 네 source family를 모두 포함해야 한다."""

    checkpoint = tmp_path / "best.pt"
    manifest = tmp_path / "recorded.jsonl"
    checkpoint.write_bytes(b"checkpoint")
    payload = _recorded_val_metric_payload(
        checkpoint=checkpoint,
        manifest=manifest,
        contract_sha="a" * 64,
        margin_db=1.0,
        source_families=("speech", "music", "environment"),
    )
    metrics = tmp_path / "metrics.npz"
    np.savez_compressed(metrics, **payload)

    with pytest.raises(ValueError, match="source family.*정확히 같아야"):
        pipeline.classify_recorded_val_metrics(
            metrics.read_bytes(),
            manifest_bytes=manifest.read_bytes(),
            manifest_path=manifest,
            checkpoint_cfg=_canonical_sampling_checkpoint_cfg(),
        )


def test_test_capability_rejects_selection_missing_required_source_family(tmp_path):
    """clear-pass 문구를 위조해도 불완전한 selected population으로 test를 열지 않는다."""

    manifest = tmp_path / "recorded.jsonl"
    families = ("speech", "music", "environment")
    _write_canonical_recorded_manifest(
        manifest, splits=("val",), source_families=families
    )
    stamped = stamp_experiment_contract(
        _canonical_sampling_checkpoint_cfg(**{
            "experiment_role": "selection_test",
            "seed": 20260803,
            "recorded_manifest": str(manifest.absolute()),
        }),
        repo_root=tmp_path,
    )
    checkpoint = tmp_path / "best.pt"
    torch.save({"cfg": stamped, "model": {"weight": torch.ones(1)}}, checkpoint)
    evaluation = tmp_path / "val-best"
    evaluation.mkdir()
    metrics = evaluation / "metrics.npz"
    np.savez_compressed(
        metrics,
        **_recorded_val_metric_payload(
            checkpoint=checkpoint,
            manifest=manifest,
            contract_sha=stamped["experiment_contract_sha256"],
            margin_db=1.0,
            manifest_splits=None,
            source_families=families,
        ),
    )
    forged_decision = {"status": "clear_pass"}
    campaign_sha = pipeline.seed_neutral_campaign_sha256(stamped)
    selection = {
        "schema_version": 2,
        "selection_split": "val",
        "manifest": str(manifest.absolute()),
        "manifest_sha256": pipeline._sha256_file(manifest),
        "experiment_contract_sha256": stamped["experiment_contract_sha256"],
        "seed_neutral_campaign_sha256": campaign_sha,
        "seed": 20260803,
        "decision": forged_decision,
        "selected": {
            "checkpoint": str(checkpoint.absolute()),
            "checkpoint_sha256": pipeline._sha256_file(checkpoint),
            "evaluation_dir": str(evaluation.absolute()),
            "metrics_sha256": pipeline._sha256_file(metrics),
            "seed": 20260803,
            "seed_neutral_campaign_sha256": campaign_sha,
            "decision": forged_decision,
        },
    }
    selection_path = tmp_path / "selection.json"
    write_json_exclusive(selection_path, selection)
    capability, _ = canonical_test_ledger_paths(selection_path, repo_root=tmp_path)

    with pytest.raises(ValueError, match="source family.*정확히 같아야"):
        issue_test_capability(
            selection_path=selection_path,
            capability_path=capability,
            repo_root=tmp_path,
        )
    assert not capability.exists()


def test_strict_coverage_gap_is_inconclusive_data_and_cannot_open_test(tmp_path):
    """부대역 target coverage 부족은 seed borderline이 아니라 추가 녹음 대상이다."""

    manifest = tmp_path / "recorded.jsonl"
    _write_canonical_recorded_manifest(manifest, splits=("val",))
    stamped = stamp_experiment_contract(
        _canonical_sampling_checkpoint_cfg(**{
            "experiment_role": "selection_test",
            "seed": 20260803,
            "recorded_manifest": str(manifest.absolute()),
        }),
        repo_root=tmp_path,
    )
    checkpoint = tmp_path / "best.pt"
    torch.save({"cfg": stamped, "model": {"weight": torch.ones(1)}}, checkpoint)
    payload = _recorded_val_metric_payload(
        checkpoint=checkpoint,
        manifest=manifest,
        contract_sha=stamped["experiment_contract_sha256"],
        margin_db=1.0,
        manifest_splits=None,
    )
    family_index = 0
    band_index = len(STRICT_TRUSTED_SUBBANDS_HZ) - 1
    family = str(np.asarray(payload["source_family"])[family_index])
    family_mask = np.asarray(payload["segment_source_family"]).astype(str) == family
    payload["per_segment_trusted_subband_coverage"][family_mask, band_index] = False
    payload["per_segment_trusted_subband_source_energy_density_ratio"][
        family_mask, band_index
    ] = 0.0
    for key in (
        "source_trusted_subband_n_segments",
        "source_trusted_subband_n_groups",
    ):
        payload[key][family_index, band_index] = 0
    for key in (
        "source_trusted_subband_coverage_fraction",
        "source_trusted_subband_source_energy_density_ratio_mean",
    ):
        payload[key][family_index, band_index] = 0.0
    for key in (
        "source_trusted_subband_nmse_mean_db",
        "source_trusted_subband_nmse_worst10_mean_db",
        "source_trusted_subband_ci_lo_db",
        "source_trusted_subband_ci_hi_db",
    ):
        payload[key][family_index, band_index] = np.nan
    for key in (
        "source_trusted_subband_coverage_pass",
        "source_trusted_subband_power_pass",
        "source_trusted_subband_mean_pass",
        "source_trusted_subband_worst10_pass",
        "source_trusted_subband_ci_pass",
        "source_trusted_subband_pass",
    ):
        payload[key][family_index, band_index] = False
    for key in (
        "g4_trusted_subband_coverage_pass",
        "g4_trusted_subband_power_pass",
        "g4_trusted_subband_mean_pass",
        "g4_trusted_subband_worst10_pass",
        "g4_trusted_subband_ci_pass",
        "g4_trusted_subband_pass",
        "g4_upper_trusted_subband_pass",
    ):
        payload[key] = np.asarray(False)
    payload["g4_pass"] = np.asarray(False)
    payload["g4_verdict"] = np.asarray("INCONCLUSIVE")
    evaluation = tmp_path / "val-underpowered"
    evaluation.mkdir()
    np.savez_compressed(evaluation / "metrics.npz", **payload)
    selection_path = tmp_path / "selection-underpowered.json"
    selection = pipeline.freeze_recorded_val_selection(
        candidates=[(checkpoint, evaluation)],
        selection_path=selection_path,
        manifest_path=manifest,
        experiment_contract_sha256=stamped["experiment_contract_sha256"],
    )
    assert selection["decision"]["status"] == "inconclusive_data"
    capability, _ = canonical_test_ledger_paths(selection_path, repo_root=tmp_path)

    with pytest.raises(ValueError, match="clear-pass"):
        issue_test_capability(
            selection_path=selection_path,
            capability_path=capability,
            repo_root=tmp_path,
        )
    assert not capability.exists()


def test_cross_seed_finalize_chooses_g4_pass_with_largest_minimum_margin(tmp_path):
    manifest = tmp_path / "recorded.jsonl"
    _write_canonical_recorded_manifest(manifest, splits=("val",))
    paths = []
    for seed, margin in ((20260803, 0.1), (20260903, 0.8)):
        stamped = stamp_experiment_contract(
            _canonical_sampling_checkpoint_cfg(**{
                "experiment_role": "selection_test",
                "seed": seed,
                "recorded_manifest": str(manifest.absolute()),
            }),
            repo_root=tmp_path,
        )
        checkpoint = tmp_path / f"{seed}.pt"
        torch.save({"cfg": stamped, "model": {"weight": torch.ones(1)}}, checkpoint)
        evaluation = tmp_path / f"val-{seed}"
        evaluation.mkdir()
        np.savez_compressed(
            evaluation / "metrics.npz",
            **_recorded_val_metric_payload(
                checkpoint=checkpoint,
                manifest=manifest,
                contract_sha=stamped["experiment_contract_sha256"],
                margin_db=margin,
                manifest_splits=None,
            ),
        )
        path = tmp_path / f"selection-{seed}.json"
        pipeline.freeze_recorded_val_selection(
            candidates=[(checkpoint, evaluation)],
            selection_path=path,
            manifest_path=manifest,
            experiment_contract_sha256=stamped["experiment_contract_sha256"],
        )
        paths.append(path)
    final = pipeline.finalize_cross_seed_selection(
        selection_paths=paths,
        final_selection_path=tmp_path / "selection-final.json",
    )
    assert final["seed"] == 20260903
    assert final["decision"]["status"] == "cross_seed_final"
    assert len(final["seed_selections"]) == 2
    assert pipeline.finalize_cross_seed_selection(
        selection_paths=paths,
        final_selection_path=tmp_path / "selection-final.json",
    ) == final


def test_selected_val_directory_is_atomic_no_replace_and_exact_reentry(tmp_path):
    source = tmp_path / "source-val"
    source.mkdir()
    metrics = source / "metrics.npz"
    metrics.write_bytes(b"immutable-metrics")
    expected = pipeline._sha256_file(metrics)
    target = tmp_path / "canonical-val"

    assert pipeline._publish_selected_val_directory(
        source, target, metrics_sha256=expected
    ) == target
    assert not list(tmp_path.glob(".canonical-val.*.staging"))
    assert pipeline._publish_selected_val_directory(
        source, target, metrics_sha256=expected
    ) == target

    (target / "metrics.npz").write_bytes(b"forged")
    with pytest.raises(FileExistsError, match="selection과 다릅니다"):
        pipeline._publish_selected_val_directory(
            source, target, metrics_sha256=expected
        )


@pytest.mark.parametrize("mutated", ["checkpoint", "manifest"])
def test_recorded_test_refuses_bytes_changed_after_val_selection(tmp_path, mutated):
    manifest = tmp_path / "recorded.jsonl"
    _write_canonical_recorded_manifest(manifest, splits=("val",))
    checkpoint = tmp_path / "best.pt"
    stamped = stamp_experiment_contract(
        _canonical_sampling_checkpoint_cfg(**{
            "experiment_role": "selection_test",
            "seed": 20260803,
            "recorded_manifest": str(manifest.absolute()),
        }),
        repo_root=tmp_path,
    )
    contract_sha = stamped["experiment_contract_sha256"]
    torch.save({"cfg": stamped, "model": {"weight": torch.ones(1)}}, checkpoint)
    evaluation = tmp_path / "val-best"
    evaluation.mkdir()
    np.savez_compressed(
        evaluation / "metrics.npz",
        **_recorded_val_metric_payload(
            checkpoint=checkpoint,
            manifest=manifest,
            contract_sha=contract_sha,
            margin_db=3.0,
            manifest_splits=None,
        ),
    )
    selection = pipeline.freeze_recorded_val_selection(
        candidates=[(checkpoint, evaluation)],
        selection_path=tmp_path / "selection.json",
        manifest_path=manifest,
        experiment_contract_sha256=contract_sha,
    )
    selection_path = tmp_path / "selection.json"
    capability, consumed = canonical_test_ledger_paths(
        selection_path, repo_root=tmp_path
    )
    token = issue_test_capability(
        selection_path=selection_path, capability_path=capability, repo_root=tmp_path
    )
    target = checkpoint if mutated == "checkpoint" else manifest
    target.write_bytes(target.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="bytes가 바뀌었습니다"):
        consume_test_capability(
            selection_path=selection_path,
            capability_path=capability,
            consumed_marker_path=consumed,
            token=token,
            checkpoint_path=checkpoint,
            manifest_path=manifest,
            repo_root=tmp_path,
        )
    assert not consumed.exists()


@pytest.mark.parametrize("phase", ["freeze", "test_open"])
def test_selection_rejects_manifest_path_different_from_checkpoint_contract(
    tmp_path, phase
):
    """같은 bytes를 복사해도 checkpoint가 선언하지 않은 manifest path는 거부한다."""

    contract_manifest = tmp_path / "contract-recorded.jsonl"
    selected_manifest = tmp_path / "other-recorded.jsonl"
    _write_canonical_recorded_manifest(contract_manifest, splits=("val",))
    selected_manifest.write_bytes(contract_manifest.read_bytes())

    if phase == "test_open":
        selection_path, selection, _ = _make_val_selection(
            tmp_path,
            name="manifest-path",
            manifest=contract_manifest,
        )
        selected_manifest.write_bytes(contract_manifest.read_bytes())
        selection["manifest"] = str(selected_manifest.absolute())
        selection_path.write_text(
            json.dumps(selection, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        capability, _ = canonical_test_ledger_paths(
            selection_path, repo_root=tmp_path
        )
        with pytest.raises(ValueError, match="recorded_manifest와 다릅니다"):
            issue_test_capability(
                selection_path=selection_path,
                capability_path=capability,
                repo_root=tmp_path,
            )
        assert not capability.exists()
        return

    stamped = stamp_experiment_contract(
        _canonical_sampling_checkpoint_cfg(**{
            "experiment_role": "selection_test",
            "seed": 20260803,
            "recorded_manifest": str(contract_manifest.absolute()),
        }),
        repo_root=tmp_path,
    )
    checkpoint = tmp_path / "best.pt"
    torch.save({"cfg": stamped, "model": {"weight": torch.ones(1)}}, checkpoint)
    evaluation = tmp_path / "val-best"
    evaluation.mkdir()
    np.savez_compressed(
        evaluation / "metrics.npz",
        **_recorded_val_metric_payload(
            checkpoint=checkpoint,
            manifest=selected_manifest,
            contract_sha=stamped["experiment_contract_sha256"],
            margin_db=1.0,
            manifest_splits=None,
        ),
    )
    with pytest.raises(ValueError, match="recorded_manifest와 다릅니다"):
        pipeline.freeze_recorded_val_selection(
            candidates=[(checkpoint, evaluation)],
            selection_path=tmp_path / "selection.json",
            manifest_path=selected_manifest,
            experiment_contract_sha256=stamped["experiment_contract_sha256"],
        )


def test_test_open_rejects_arbitrary_selection_campaign_sha(tmp_path):
    """형식만 맞는 campaign SHA로 별도 one-shot ledger를 열 수 없다."""

    selection_path, selection, _manifest = _make_val_selection(
        tmp_path, name="campaign-forgery"
    )
    actual = selection["seed_neutral_campaign_sha256"]
    selection["seed_neutral_campaign_sha256"] = (
        "f" * 64 if actual != "f" * 64 else "e" * 64
    )
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    capability, _ = canonical_test_ledger_paths(selection_path, repo_root=tmp_path)

    with pytest.raises(ValueError, match="campaign digest.*checkpoint"):
        issue_test_capability(
            selection_path=selection_path,
            capability_path=capability,
            repo_root=tmp_path,
        )
    assert not capability.exists()


def test_recorded_val_selection_rejects_candidate_embedded_contract_mismatch(tmp_path):
    manifest = tmp_path / "recorded.jsonl"
    _write_canonical_recorded_manifest(manifest, splits=("val",))
    stamped = stamp_experiment_contract(
        _canonical_sampling_checkpoint_cfg(**{
            "experiment_role": "selection_test",
            "seed": 20260803,
            "recorded_manifest": str(manifest.absolute()),
        }),
        repo_root=tmp_path,
    )
    checkpoint = tmp_path / "best.pt"
    torch.save({"cfg": stamped, "model": {"weight": torch.ones(1)}}, checkpoint)
    evaluation = tmp_path / "val-best"
    evaluation.mkdir()
    np.savez_compressed(
        evaluation / "metrics.npz",
        **_recorded_val_metric_payload(
            checkpoint=checkpoint,
            manifest=manifest,
            contract_sha="f" * 64,
            margin_db=3.0,
            manifest_splits=None,
        ),
    )
    with pytest.raises(ValueError, match="embedded experiment contract"):
        pipeline.freeze_recorded_val_selection(
            candidates=[(checkpoint, evaluation)],
            selection_path=tmp_path / "selection.json",
            manifest_path=manifest,
            experiment_contract_sha256="f" * 64,
        )


def test_test_capability_wrong_token_and_symlink_do_not_create_consumed_marker(
    tmp_path,
):
    selection_path, selection, manifest = _make_val_selection(
        tmp_path, name="wrong-token"
    )
    checkpoint = Path(selection["selected"]["checkpoint"])
    capability, consumed = canonical_test_ledger_paths(
        selection_path, repo_root=tmp_path
    )
    token = issue_test_capability(
        selection_path=selection_path, capability_path=capability, repo_root=tmp_path
    )
    with pytest.raises(ValueError, match="token_sha256"):
        consume_test_capability(
            selection_path=selection_path,
            capability_path=capability,
            consumed_marker_path=consumed,
            token=token + "forged",
            checkpoint_path=checkpoint,
            manifest_path=manifest,
            repo_root=tmp_path,
        )
    assert not consumed.exists()

    link = tmp_path / "selection-link.json"
    link.symlink_to(selection_path)
    with pytest.raises(ValueError, match="regular-file snapshot"):
        consume_test_capability(
            selection_path=link,
            capability_path=capability,
            consumed_marker_path=consumed,
            token=token,
            checkpoint_path=checkpoint,
            manifest_path=manifest,
            repo_root=tmp_path,
        )
    assert not consumed.exists()


def test_json_capability_publication_is_atomic_no_replace_on_failure(
    tmp_path, monkeypatch
):
    import deep_anc.train.evaluation_contract as contract_module

    target = tmp_path / "selection.json"

    def fail_link(*args, **kwargs):
        raise OSError("injected hard-link failure")

    monkeypatch.setattr(contract_module.os, "link", fail_link)
    with pytest.raises(OSError, match="injected"):
        contract_module.write_json_exclusive(target, {"schema_version": 1})
    assert not target.exists()
    assert not list(tmp_path.glob(".selection.json.*.tmp"))


def test_direct_test_evaluator_bypass_is_rejected_before_opening_inputs(tmp_path):
    out = tmp_path / "eval-test"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/eval/evaluate_recorded.py"),
            "--ckpt",
            str(tmp_path / "missing.pt"),
            "--manifest",
            str(tmp_path / "missing.jsonl"),
            "--split",
            "test",
            "--out",
            str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "single-use capability" in completed.stderr
    assert not out.exists()


def test_evaluator_requires_explicit_diagnostic_flag_for_reduced_population(tmp_path):
    base = [
        sys.executable,
        str(REPO_ROOT / "scripts/eval/evaluate_recorded.py"),
        "--ckpt",
        str(tmp_path / "missing.pt"),
        "--manifest",
        str(tmp_path / "missing.jsonl"),
        "--split",
        "val",
        "--max-segments-per-session",
        "1",
    ]
    rejected = subprocess.run(
        base,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "canonical recorded 평가는 max-segments=64" in rejected.stderr

    diagnostic = subprocess.run(
        [*base, "--diagnostic-sampling-override"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert diagnostic.returncode == 2
    assert "canonical recorded 평가는 max-segments=64" not in diagnostic.stderr
    assert "regular-file snapshot" in diagnostic.stderr


@pytest.mark.parametrize(
    "override",
    (
        ["--warmup-seconds", "1.49"],
        ["--feedback-delay-samples", "1"],
    ),
)
def test_evaluator_rejects_metric_window_override_without_diagnostic_flag(
    tmp_path, override
):
    base = [
        sys.executable,
        str(REPO_ROOT / "scripts/eval/evaluate_recorded.py"),
        "--ckpt",
        str(tmp_path / "missing.pt"),
        "--manifest",
        str(tmp_path / "missing.jsonl"),
        "--split",
        "val",
        *override,
    ]
    rejected = subprocess.run(
        base, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert rejected.returncode == 2
    assert "override는 --diagnostic-sampling-override" in rejected.stderr

    diagnostic = subprocess.run(
        [*base, "--diagnostic-sampling-override"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert diagnostic.returncode == 2
    assert "override는 --diagnostic-sampling-override" not in diagnostic.stderr
    assert "regular-file snapshot" in diagnostic.stderr


def test_test_evaluator_never_replaces_an_existing_output_directory(tmp_path):
    out = tmp_path / "eval-test"
    out.mkdir()
    sentinel = out / "metrics.npz"
    sentinel.write_bytes(b"original")
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/eval/evaluate_recorded.py"),
            "--ckpt",
            str(tmp_path / "missing.pt"),
            "--manifest",
            str(tmp_path / "missing.jsonl"),
            "--split",
            "test",
            "--out",
            str(out),
            "--selection",
            str(tmp_path / "selection.json"),
            "--test-capability",
            str(tmp_path / "capability.json"),
            "--test-consumed-marker",
            str(tmp_path / "consumed.json"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "덮어쓸 수 없습니다" in completed.stderr
    assert sentinel.read_bytes() == b"original"
    assert not (tmp_path / "consumed.json").exists()


def test_second_acquire_raises_and_release_allows_reacquire(tmp_path):
    path = tmp_path / "pipeline.lock"
    first = ProcessLock(path, role="pipeline").acquire()
    with pytest.raises(LockHeldError, match="owner="):
        ProcessLock(path, role="pipeline").acquire()
    first.release()
    ProcessLock(path, role="pipeline").acquire().release()
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_kernel_releases_lock_when_holder_dies(tmp_path):
    """flock 은 프로세스 종료 시 커널이 해제한다 — stale 파일은 그대로 재사용 가능."""

    path = tmp_path / "pipeline.lock"
    script = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT / 'src')!r});"
        "from deep_anc.train.process_lock import ProcessLock;"
        f"ProcessLock({str(path)!r}, role='dead').acquire(); print('held', flush=True)"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0 and "held" in done.stdout
    assert path.exists()
    ProcessLock(path, role="fresh").acquire().release()


# ---------------------------------------------------------------------------
# 상태 스키마
# ---------------------------------------------------------------------------


def test_status_rejects_unknown_phase(tmp_path):
    status = PipelineStatus(
        tmp_path / "status.json", mode="pipeline", run_key="k", run_dir=tmp_path,
        state_dir=tmp_path, lock_path=tmp_path / "l", config_path=tmp_path / "c",
        config_sha256="sha256:0", overrides=[], fingerprint="sha256:0",
    )
    with pytest.raises(ValueError, match="알 수 없는 phase"):
        status.update("bogus_phase")


def test_stable_view_drops_volatile_fields():
    payload = {
        "phase": "not_ready", "pid": 1, "hostname": "a", "started_at_utc": "t",
        "updated_at_utc": "t", "duration_seconds": 1.0,
        "readiness": {"ok": False, "checked_at_utc": "t"},
        "steps": [{"name": "train", "returncode": 0, "duration_seconds": 2.0}],
    }
    view = stable_view(payload)
    assert view == {
        "phase": "not_ready",
        "readiness": {"ok": False},
        "steps": [{"name": "train", "returncode": 0}],
    }


def test_child_uses_the_running_interpreter_not_its_symlink_target(repo, monkeypatch):
    """venv 의 bin/python 은 시스템 인터프리터로의 심볼릭 링크다.

    경로를 resolve 하면 링크를 따라가 venv 의 site-packages 를 잃고, 자식이
    ModuleNotFoundError 로 죽는다. 실제로 원격에서 이렇게 실패했다.
    """

    captured: list[list[str]] = []

    def fake_run(command, **kwargs):
        captured.append(list(command))
        raise SystemExit(0)

    monkeypatch.setattr(pipeline, "_run", fake_run)
    try:
        pipeline.main([a for a in ARGS if a != "--check-only"])
    except SystemExit:
        pass
    if captured:
        assert captured[0][0] == sys.executable, (
            f"자식이 {captured[0][0]} 로 떴다 — sys.executable({sys.executable}) 이어야 한다"
        )
