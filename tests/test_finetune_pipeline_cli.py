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
from deep_anc.train.evaluation_contract import (  # noqa: E402
    canonical_test_ledger_paths,
    canonical_test_ledger_event_paths_from_payload,
    complete_test_evaluation,
    consume_test_capability,
    fail_test_evaluation,
    issue_test_capability,
    publish_directory_noreplace,
)
from deep_anc.train.experiment_contract import stamp_experiment_contract  # noqa: E402

ARGS = [
    "--check-only",
    "--config",
    "configs/train_finetune.yaml",
    "--set",
    "data.digital_primary_path_mode=measured",
]


def _recorded_val_metric_payload(
    *, checkpoint: Path, manifest: Path, contract_sha: str, margin_db: float
) -> dict:
    value = -float(margin_db)
    return {
        "nmse_trusted_worst10_mean_db": np.asarray(value),
        "nmse_trusted_mean_db": np.asarray(value),
        "nmse_fullband_mean_db": np.asarray(value),
        "g4_worst_source_trusted_mean_db": np.asarray(value),
        "g4_worst_source_trusted_worst10_db": np.asarray(value),
        "g4_worst_octave_worst10_db": np.asarray(1.0),
        "g4_max_out_of_band_amplification_db": np.asarray(1.0),
        "source_trusted_ci_hi_db": np.asarray([value, value]),
        "g4_trusted_pass": np.asarray(True),
        "g4_fullband_pass": np.asarray(True),
        "g4_source_pass": np.asarray(True),
        "g4_do_no_harm_pass": np.asarray(True),
        "g4_power_pass": np.asarray(True),
        "g4_ci_pass": np.asarray(True),
        "g4_pass": np.asarray(True),
        "g4_verdict": np.asarray("PASS"),
        "split": np.asarray("val"),
        "checkpoint_sha256": np.asarray(pipeline._sha256_file(checkpoint)),
        "manifest_sha256": np.asarray(pipeline._sha256_file(manifest)),
        "experiment_contract_sha256": np.asarray(contract_sha),
    }


def _make_val_selection(
    tmp_path: Path,
    *,
    seed: int = 20260803,
    margin_db: float = 1.0,
    name: str = "run",
    manifest: Path | None = None,
) -> tuple[Path, dict, Path]:
    if manifest is None:
        manifest = tmp_path / "recorded.jsonl"
        if not manifest.exists():
            manifest.write_text('{"split":"val"}\n')
    stamped = stamp_experiment_contract(
        {"experiment_role": "selection_test", "model": {"name": "toy"}, "seed": seed}
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
    (tmp_path / ".gitignore").write_text("/results/\n/runs/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "configs", ".gitignore"], cwd=tmp_path, check=True)
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
    manifest.write_text('{"split":"val"}\n')
    candidates = []
    stamped = stamp_experiment_contract(
        {
            "experiment_role": "selection_test",
            "model": {"name": "toy"},
            "seed": 20260803,
        }
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
    selection_path, selection, manifest = _make_val_selection(tmp_path, name="phase")
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
    staging = tmp_path / ".eval.staging"
    staging.mkdir()
    (staging / "metrics.md").write_text("ok\n")
    (staging / "metrics.npz").write_bytes(b"npz")
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
    decision = pipeline.classify_recorded_val_metrics(buffer.read_bytes())
    assert decision["status"] == "borderline"
    assert decision["minimum_margin_db"] == pytest.approx(0.3)


def test_cross_seed_finalize_chooses_g4_pass_with_largest_minimum_margin(tmp_path):
    manifest = tmp_path / "recorded.jsonl"
    manifest.write_text('{"split":"val"}\n')
    paths = []
    for seed, margin in ((20260803, 0.1), (20260903, 0.8)):
        stamped = stamp_experiment_contract(
            {
                "experiment_role": "selection_test",
                "model": {"name": "toy"},
                "seed": seed,
            }
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
    manifest.write_text('{"split":"val"}\n')
    checkpoint = tmp_path / "best.pt"
    stamped = stamp_experiment_contract(
        {
            "experiment_role": "selection_test",
            "model": {"name": "toy"},
            "seed": 20260803,
        }
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

def test_recorded_val_selection_rejects_candidate_embedded_contract_mismatch(tmp_path):
    manifest = tmp_path / "recorded.jsonl"
    manifest.write_text('{"split":"val"}\n')
    stamped = stamp_experiment_contract(
        {
            "experiment_role": "selection_test",
            "model": {"name": "toy"},
            "seed": 20260803,
        }
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
    manifest = tmp_path / "recorded.jsonl"
    manifest.write_text('{"split":"val"}\n')
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    evaluation = tmp_path / "val-best"
    evaluation.mkdir()
    metrics = evaluation / "metrics.npz"
    np.savez_compressed(
        metrics,
        **_recorded_val_metric_payload(
            checkpoint=checkpoint,
            manifest=manifest,
            contract_sha="a" * 64,
            margin_db=1.0,
        ),
    )
    decision = pipeline.classify_recorded_val_metrics(metrics.read_bytes())
    selection = {
        "schema_version": 2,
        "selection_split": "val",
        "manifest": str(manifest),
        "manifest_sha256": pipeline._sha256_file(manifest),
        "experiment_contract_sha256": "a" * 64,
        "seed_neutral_campaign_sha256": "c" * 64,
        "seed": 20260803,
        "decision": decision,
        "selected": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": pipeline._sha256_file(checkpoint),
            "evaluation_dir": str(evaluation),
            "metrics_sha256": pipeline._sha256_file(metrics),
            "seed": 20260803,
            "decision": decision,
        },
    }
    selection_path = tmp_path / "selection.json"
    from deep_anc.train.evaluation_contract import write_json_exclusive

    write_json_exclusive(selection_path, selection)
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
