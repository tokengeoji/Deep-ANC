"""Canonical Elice one-step campaign entrypoint의 fail-closed 경계."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/train/run_canonical_campaign.py"


def _module():
    spec = importlib.util.spec_from_file_location("canonical_campaign_entrypoint", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses가 __module__ annotation을 해석할 때 실제 module registry가 필요하다.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


campaign = _module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _repo(tmp_path: Path, *, transfer_schema: int = 2) -> tuple[Path, str, str, str]:
    repo = tmp_path / "repo"
    (repo / "data/manifests").mkdir(parents=True)
    (repo / "scripts/elice").mkdir(parents=True)
    (repo / "scripts/elice/bootstrap_all.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (repo / ".gitignore").write_text(
        "/results/\n/data/manifests/elice_bootstrap_receipt.json\n",
        encoding="utf-8",
    )
    holdout = repo / campaign.HOLDOUT_MANIFEST
    holdout.write_text('{"holdout":true}\n', encoding="utf-8")
    transfer = repo / campaign.TRANSFER_MANIFEST
    transfer.write_text(
        json.dumps({"schema_version": transfer_schema}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    return repo, _git(repo, "rev-parse", "HEAD"), _sha(holdout), _sha(transfer)


def _contract_payload(commit: str, holdout: str, transfer: str) -> dict:
    return {
        "schema_version": 2,
        "expected_commit": commit,
        "expected_holdout_sha256": holdout,
        "expected_transfer_manifest_sha256": transfer,
        "campaign": {"seed": 20260803, "second_seed": None},
        "bootstrap": {
            "raw_hash_workers": 8,
            "cublas_workspace_config": ":4096:8",
            "decoder_audit": {
                "expected_audit_sha256": "a" * 64,
                "expected_file_sha256": "b" * 64,
            },
        },
        "candidates": [
            {"alpha": "0.7", "lambda_dnh": "0.00075"},
            {"alpha": "1.0", "lambda_dnh": "0.00075"},
        ],
    }


def _write_contract(path: Path, payload: dict) -> str:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    return _sha(path)


def _write_bootstrap_receipt(
    repo: Path, *, commit: str, holdout_sha: str, transfer_sha: str
) -> Path:
    path = repo / campaign.BOOTSTRAP_RECEIPT
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "expected_commit": commit,
                "canonical_holdout": {
                    "path": campaign.HOLDOUT_MANIFEST,
                    "sha256": holdout_sha,
                },
                "transfer_manifest": {
                    "path": campaign.TRANSFER_MANIFEST,
                    "sha256": transfer_sha,
                },
                "recorded_aggregate_sha256": "1" * 64,
                "recorded_subband_coverage": {},
                "environment": {},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_contract_is_external_sha_anchored_and_has_canonical_candidate_order(tmp_path):
    repo, commit, holdout, transfer = _repo(tmp_path)
    path = tmp_path / "campaign.json"
    digest = _write_contract(path, _contract_payload(commit, holdout, transfer))
    contract = campaign.load_contract(path, digest, repo_root=repo)

    assert contract.expected_commit == commit
    assert [candidate.alpha_text for candidate in contract.candidates] == ["0.7", "1.0"]
    assert contract.candidates[0].key != contract.candidates[1].key

    with pytest.raises(campaign.CampaignError, match="SHA가 외부 anchor"):
        campaign.load_contract(path, "0" * 64, repo_root=repo)

    inside = repo / "campaign.json"
    inside.write_bytes(path.read_bytes())
    with pytest.raises(campaign.CampaignError, match="저장소 밖"):
        campaign.load_contract(inside, _sha(inside), repo_root=repo)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["candidates"][0].update(alpha=0.7), "decimal string"),
        (
            lambda value: value["candidates"][0].update(lambda_dnh="0.000750"),
            "canonical이 아닙니다",
        ),
        (
            lambda value: value["candidates"].insert(
                1, {"alpha": "0.85", "lambda_dnh": "0.00075"}
            ),
            None,
        ),
        (lambda value: value.update(extra=True), "key 집합"),
    ],
)
def test_contract_rejects_ambiguous_numbers_and_unknown_keys(tmp_path, mutate, message):
    repo, commit, holdout, transfer = _repo(tmp_path)
    payload = _contract_payload(commit, holdout, transfer)
    mutate(payload)
    path = tmp_path / "campaign.json"
    digest = _write_contract(path, payload)
    if message is None:
        contract = campaign.load_contract(path, digest, repo_root=repo)
        assert [row.alpha_text for row in contract.candidates] == ["0.7", "0.85", "1.0"]
    else:
        with pytest.raises(campaign.CampaignError, match=message):
            campaign.load_contract(path, digest, repo_root=repo)


def test_schema_v2_secondary_contract_requires_exact_external_primary_link(tmp_path):
    repo, commit, holdout, transfer = _repo(tmp_path)
    primary_path = tmp_path / "primary.json"
    primary_payload = _contract_payload(commit, holdout, transfer)
    primary_sha = _write_contract(primary_path, primary_payload)

    secondary_path = tmp_path / "secondary.json"
    secondary_payload = _contract_payload(commit, holdout, transfer)
    secondary_payload["campaign"] = {
        "seed": 20260903,
        "second_seed": {
            "primary_contract_path": str(primary_path.absolute()),
            "primary_contract_sha256": primary_sha,
            "primary_selection_sha256": "c" * 64,
            "seed_neutral_campaign_sha256": "d" * 64,
        },
    }
    secondary_sha = _write_contract(secondary_path, secondary_payload)
    contract = campaign.load_contract(
        secondary_path, secondary_sha, repo_root=repo
    )

    assert contract.seed == 20260903
    assert contract.second_seed is not None
    assert contract.second_seed.primary_contract_sha256 == primary_sha

    secondary_payload["expected_transfer_manifest_sha256"] = "e" * 64
    changed_sha = _write_contract(secondary_path, secondary_payload)
    with pytest.raises(campaign.CampaignError, match="primary sealed campaign"):
        campaign.load_contract(secondary_path, changed_sha, repo_root=repo)


def test_primary_contract_cannot_claim_second_seed_link(tmp_path):
    repo, commit, holdout, transfer = _repo(tmp_path)
    path = tmp_path / "campaign.json"
    payload = _contract_payload(commit, holdout, transfer)
    payload["campaign"]["second_seed"] = {
        "primary_contract_path": str((tmp_path / "other.json").absolute()),
        "primary_contract_sha256": "a" * 64,
        "primary_selection_sha256": "b" * 64,
        "seed_neutral_campaign_sha256": "c" * 64,
    }
    digest = _write_contract(path, payload)

    with pytest.raises(campaign.CampaignError, match="primary seed contract"):
        campaign.load_contract(path, digest, repo_root=repo)


def test_schema_v1_transfer_blocks_before_bootstrap_or_gpu(tmp_path):
    repo, commit, holdout, transfer = _repo(tmp_path, transfer_schema=1)
    path = tmp_path / "campaign.json"
    digest = _write_contract(path, _contract_payload(commit, holdout, transfer))
    contract = campaign.load_contract(path, digest, repo_root=repo)

    state = campaign.inspect_campaign(contract, repo_root=repo)

    assert state.status == "BLOCKED"
    assert state.next_action == "LOCAL_TRANSFER_SCHEMA_V2_REQUIRED"
    assert state.command is None


def test_missing_bootstrap_receipt_yields_only_exact_bootstrap_all_command(tmp_path):
    repo, commit, holdout, transfer = _repo(tmp_path, transfer_schema=2)
    audit = repo / campaign.DECODER_AUDIT_REPORT
    audit.parent.mkdir(parents=True)
    audit.write_text(
        json.dumps({"audit_sha256": "a" * 64}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path = tmp_path / "campaign.json"
    payload = _contract_payload(commit, holdout, transfer)
    payload["bootstrap"]["decoder_audit"]["expected_file_sha256"] = _sha(audit)
    digest = _write_contract(path, payload)
    contract = campaign.load_contract(path, digest, repo_root=repo)

    state = campaign.inspect_campaign(contract, repo_root=repo)

    assert state.status == "READY_TO_EXECUTE"
    assert state.next_action == "bootstrap"
    assert state.command[:2] == ["bash", str((repo / "scripts/elice/bootstrap_all.sh").absolute())]
    assert "--expected-commit" in state.command
    assert "--expected-transfer-manifest-sha256" in state.command
    assert "--reuse-decoder-audit" in state.command
    assert "--no-update" in state.command
    command_text = " ".join(state.command)
    assert not any(token in command_text for token in campaign._LEGACY_TOKENS)


def test_reuse_contract_blocks_before_bootstrap_when_decoder_cache_is_missing(tmp_path):
    repo, commit, holdout, transfer = _repo(tmp_path, transfer_schema=2)
    path = tmp_path / "campaign.json"
    digest = _write_contract(path, _contract_payload(commit, holdout, transfer))
    contract = campaign.load_contract(path, digest, repo_root=repo)

    state = campaign.inspect_campaign(contract, repo_root=repo)

    assert state.status == "BLOCKED"
    assert state.next_action == "DECODER_AUDIT_CACHE_MISSING"
    assert state.command is None


def test_same_commit_v1_receipt_is_replaced_only_by_exact_v2_bootstrap(tmp_path):
    repo, commit, holdout, transfer = _repo(tmp_path, transfer_schema=2)
    old_transfer = "e" * 64
    old_receipt = _write_bootstrap_receipt(
        repo,
        commit=commit,
        holdout_sha=holdout,
        transfer_sha=old_transfer,
    )
    path = tmp_path / "campaign.json"
    payload = _contract_payload(commit, holdout, transfer)
    payload["bootstrap"]["decoder_audit"] = None
    digest = _write_contract(path, payload)
    contract = campaign.load_contract(path, digest, repo_root=repo)

    state = campaign.inspect_campaign(contract, repo_root=repo)

    assert state.status == "READY_TO_EXECUTE"
    assert state.next_action == "bootstrap"
    assert state.command == campaign.build_bootstrap_command(contract, repo)
    previous = state.details["replaces_previous_bootstrap"]
    assert previous["path"] == str(old_receipt)
    assert previous["sha256"] == _sha(old_receipt)
    assert previous["transfer_manifest_sha256"] == old_transfer
    assert previous["replacement_transfer_manifest_sha256"] == transfer


def test_bootstrap_receipt_from_different_commit_never_auto_replaces(tmp_path):
    repo, commit, holdout, transfer = _repo(tmp_path, transfer_schema=2)
    _write_bootstrap_receipt(
        repo,
        commit="f" * 40,
        holdout_sha=holdout,
        transfer_sha="e" * 64,
    )
    path = tmp_path / "campaign.json"
    payload = _contract_payload(commit, holdout, transfer)
    payload["bootstrap"]["decoder_audit"] = None
    digest = _write_contract(path, payload)
    contract = campaign.load_contract(path, digest, repo_root=repo)

    state = campaign.inspect_campaign(contract, repo_root=repo)

    assert state.status == "BLOCKED"
    assert state.next_action == "LOCAL_ADMISSION_FAILED"
    assert state.command is None


def test_stage_order_and_existing_cli_argv_are_explicit_and_legacy_free(tmp_path):
    repo = tmp_path / "repo"
    for path in (
        repo / ".venv/bin/python",
        repo / "scripts/train/train.py",
        repo / "scripts/eval/evaluate_recorded.py",
        repo / "scripts/bench/diagnose_training_overfit.py",
        repo / "scripts/train/measure_gradient_budget.py",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o755)
    candidate = campaign.Candidate("0.7", "0.00075")
    contract = SimpleNamespace(expected_commit="c" * 40)
    g0 = campaign.build_g0_command(contract, candidate, "d" * 64, repo)
    pilot_overrides = campaign._candidate_overrides(
        candidate, "d" * 64, role="loss_pilot"
    )
    pilot = campaign.build_train_command(
        repo,
        config=campaign.CANONICAL_PRETRAIN_CONFIG,
        overrides=pilot_overrides,
    )
    pilot_val = campaign.build_recorded_val_command(
        repo,
        checkpoint=repo / "runs/pilot/ckpt/best.pt",
        manifest=repo / "data/recorded.jsonl",
        output=repo / "runs/pilot/eval_recorded_val",
        allow_surrogate=True,
    )
    probe_overrides = campaign._candidate_overrides(
        candidate,
        "d" * 64,
        role="measured_probe",
        init_ckpt="runs/pilot/ckpt/best.pt",
    )

    assert campaign.CANONICAL_STAGE_ORDER == (
        "bootstrap",
        "pre_g0_readiness",
        "g0_all_candidates",
        "prepilot_gradient_all_candidates",
        "loss_pilot_20k_each",
        "loss_pilot_recorded_val_each",
        "measured_probe_5k_each",
        "measured_probe_recorded_val_each",
        "raw_winner_selection",
        "selected20k_gradient",
        "resume_smoke",
        "issue_campaign_ledger",
        "issue_second_seed_prerequisite",
        "canonical_pretrain_100k",
        "finetune_readiness_17_of_17",
        "canonical_finetune_50k",
        "cross_seed_finalize_if_required",
    )
    assert "scripts/bench/diagnose_training_overfit.py" in " ".join(g0)
    assert "experiment_role=loss_pilot" in pilot
    assert "run_until_step=20000" in pilot
    assert "--allow-surrogate" in pilot_val
    assert "experiment_role=measured_probe" in probe_overrides
    assert "data.digital_primary_path_mode=measured" in probe_overrides
    assert "run_until_step=5000" in probe_overrides
    assert not any(
        token in " ".join([*g0, *pilot, *pilot_val, *probe_overrides])
        for token in campaign._LEGACY_TOKENS
    )


def test_partial_run_never_resumes_without_exact_path_and_external_sha(tmp_path):
    last = tmp_path / "runs/open_loop/ckpt/last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"checkpoint")
    base = ["python", "train.py"]

    blocked = campaign._explicit_resume(
        action="canonical_pretrain_resume",
        phase="canonical_pretrain",
        base_command=base,
        expected_last=last,
        resume_path=None,
        resume_sha256=None,
        details={},
    )
    assert blocked.next_action == "EXPLICIT_RESUME_REQUIRED"
    assert blocked.command is None

    ready = campaign._explicit_resume(
        action="canonical_pretrain_resume",
        phase="canonical_pretrain",
        base_command=base,
        expected_last=last,
        resume_path=last,
        resume_sha256=_sha(last),
        details={},
    )
    assert ready.status == "READY_TO_EXECUTE"
    assert ready.command == [*base, "--resume", str(last.absolute())]


def test_secondary_finetune_uses_secondary_100k_best_and_cross_finalize_has_no_init_override(
    tmp_path,
):
    repo = tmp_path / "repo"
    for path in (
        repo / ".venv/bin/python",
        repo / "scripts/train/run_finetune_pipeline.py",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o755)
    primary_init = repo / "runs/primary-pretrain/ckpt/best.pt"
    secondary_init = repo / "runs/secondary-pretrain/ckpt/best.pt"
    command = campaign._finetune_pipeline_command(
        repo,
        bootstrap_sha="a" * 64,
        winner=campaign.Candidate("0.85", "0.00075"),
        init_checkpoint=secondary_init,
        seed=20260903,
    )
    cross = campaign._cross_seed_finalize_command(
        repo,
        primary_selection=repo / "primary-selection.json",
        secondary_selection=repo / "secondary-selection.json",
        final_selection=repo / "cross-selection.json",
    )

    assert f"init_ckpt={json.dumps(str(secondary_init.absolute()))}" in command
    assert "seed=20260903" in command
    assert str(primary_init) not in " ".join(command)
    assert "--set" not in cross
    assert str(primary_init) not in " ".join(cross)
    assert str(secondary_init) not in " ".join(cross)


def test_current_contract_requires_embedded_top_level_and_run_dir_sha():
    digest = "a" * 64
    cfg = {
        "experiment_contract": {"sha256": digest},
        "experiment_contract_sha256": digest,
        "resolved_contract_run_dir": {"experiment_contract_sha256": digest},
    }
    modules = {
        "validate_embedded_experiment_contract": lambda value: value[
            "experiment_contract"
        ]
    }

    assert (
        campaign._current_experiment_contract_sha(modules, cfg, label="fixture")
        == digest
    )
    cfg["resolved_contract_run_dir"] = {"experiment_contract_sha256": "b" * 64}
    with pytest.raises(campaign.CampaignError, match="run-directory contract SHA"):
        campaign._current_experiment_contract_sha(modules, cfg, label="fixture")


@pytest.mark.parametrize(
    ("role", "init_eligible"),
    [("canonical_pretrain", True), ("canonical_finetune", False)],
)
def test_completion_receipt_from_other_canonical_contract_is_rejected(
    tmp_path, role, init_eligible
):
    modules = {
        "validate_completion_receipt": lambda *_args, **_kwargs: {
            "experiment_role": role,
            "init_eligible": init_eligible,
            "experiment_contract_sha256": "b" * 64,
        }
    }

    with pytest.raises(campaign.CampaignError, match="현재 resolved contract와 다릅니다"):
        campaign._validated_completion_receipt(
            modules,
            tmp_path / "copied-run/ckpt",
            expected_role=role,
            expected_init_eligible=init_eligible,
            expected_contract_sha256="a" * 64,
            repo_root=tmp_path,
        )


def test_forged_done_status_without_raw_eval_authority_is_blocked(tmp_path):
    repo = tmp_path / "repo"
    finetune_dir = repo / "runs/finetune-contract"
    state_root = repo / "results/finetune_autostart/finetune-contract"
    state_root.mkdir(parents=True)
    (state_root / "status.json").write_text(
        json.dumps({"phase": "done", "exit_code": 0}) + "\n", encoding="utf-8"
    )
    digest = "a" * 64
    modules = {
        # 이 회귀의 시작점은 valid 50k completion receipt다. 그 뒤 raw
        # selection/eval/test authority가 없어도 forged status가 열 수 있는지 본다.
        "validate_completion_receipt": lambda *_args, **_kwargs: {
            "experiment_role": "canonical_finetune",
            "init_eligible": False,
            "experiment_contract_sha256": digest,
        },
        "canonical_test_ledger_paths_from_payload": lambda *_args, **_kwargs: (
            repo / "capability.json",
            repo / "consumed.json",
        ),
        "audit_finetune_completion": lambda *_args, **_kwargs: pytest.fail(
            "raw selection이 없으면 completion audit까지 도달하면 안 됩니다"
        ),
    }

    state = campaign._inspect_finetune_terminal_authority(
        modules,
        repo_root=repo,
        finetune_cfg={},
        finetune_dir=finetune_dir,
        state_root=state_root,
        expected_contract_sha256=digest,
        pretrain_dir=repo / "runs/pretrain-contract",
        winner_detail={"candidate": "fixture"},
        run_detail={"last_step": 50_000},
    )

    assert state.status == "BLOCKED"
    assert state.next_action == "FINETUNE_TERMINAL_AUTHORITY_INVALID"
    observation = state.details["pipeline_status_observation"]
    assert observation["advisory_only"] is True
    assert observation["phase"] == "done"
    assert observation["exit_code"] == 0
    assert "recorded_val_selection.json" in state.blockers[0]["message"]


def test_copied_finetune_receipt_with_other_contract_returns_blocked(tmp_path):
    repo = tmp_path / "repo"
    state_root = repo / "results/state"
    state_root.mkdir(parents=True)
    modules = {
        "validate_completion_receipt": lambda *_args, **_kwargs: {
            "experiment_role": "canonical_finetune",
            "init_eligible": False,
            "experiment_contract_sha256": "b" * 64,
        }
    }

    state = campaign._inspect_finetune_terminal_authority(
        modules,
        repo_root=repo,
        finetune_cfg={},
        finetune_dir=repo / "runs/current-contract",
        state_root=state_root,
        expected_contract_sha256="a" * 64,
        pretrain_dir=repo / "runs/pretrain",
        winner_detail={},
        run_detail={"last_step": 50_000},
    )

    assert state.status == "BLOCKED"
    assert state.next_action == "FINETUNE_TERMINAL_AUTHORITY_INVALID"
    assert "현재 resolved contract와 다릅니다" in state.blockers[0]["message"]


def test_execute_next_writes_dry_run_before_exactly_one_child(monkeypatch, tmp_path):
    contract = campaign.CampaignContract(
        path=tmp_path / "contract.json",
        sha256="a" * 64,
        expected_commit="b" * 40,
        expected_holdout_sha256="c" * 64,
        expected_transfer_manifest_sha256="d" * 64,
        raw_hash_workers=8,
        cublas_workspace_config=":4096:8",
        decoder_audit=None,
        candidates=(campaign.Candidate("0.7", "0.00075"), campaign.Candidate("1.0", "0.00075")),
        seed=20260803,
        second_seed=None,
    )
    ready = campaign.Inspection(
        phase="g0",
        status="READY_TO_EXECUTE",
        next_action="g0",
        command=["exact-python", "exact-g0.py"],
        blockers=[],
        details={},
    )
    complete = campaign.Inspection(
        phase="prepilot_gradient",
        status="READY_TO_EXECUTE",
        next_action="prepilot_gradient",
        command=["exact-python", "exact-gradient.py"],
        blockers=[],
        details={},
    )
    order: list[str] = []
    inspections = iter((ready, complete))
    monkeypatch.setattr(campaign, "load_contract", lambda *_args, **_kwargs: contract)
    monkeypatch.setattr(
        campaign, "inspect_campaign", lambda *_args, **_kwargs: next(inspections)
    )
    monkeypatch.setattr(
        campaign,
        "atomic_write_state",
        lambda *_args, **_kwargs: order.append("state"),
    )
    monkeypatch.setattr(
        campaign,
        "build_execution_seal",
        lambda *_args, **_kwargs: {"fixture": "seal"},
    )
    monkeypatch.setattr(
        campaign, "verify_pre_execution_authority", lambda *_args, **_kwargs: None
    )

    def run(command, **kwargs):
        order.append("child")
        assert command == ["exact-python", "exact-g0.py"]
        assert kwargs["check"] is False
        assert "shell" not in kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign.subprocess, "run", run)
    state_out = tmp_path / "state.json"
    result = campaign.main(
        [
            "--contract",
            str(contract.path),
            "--expected-contract-sha256",
            contract.sha256,
            "--state-out",
            str(state_out),
            "--execute-next",
        ]
    )

    assert result == 0
    assert order == ["state", "child", "state"]


def test_bootstrap_success_preserves_child_exit_and_requires_venv_reinvoke(
    monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    repo.mkdir()
    contract = campaign.CampaignContract(
        path=tmp_path / "contract.json",
        sha256="a" * 64,
        expected_commit="b" * 40,
        expected_holdout_sha256="c" * 64,
        expected_transfer_manifest_sha256="d" * 64,
        raw_hash_workers=8,
        cublas_workspace_config=":4096:8",
        decoder_audit=None,
        candidates=(
            campaign.Candidate("0.7", "0.00075"),
            campaign.Candidate("1.0", "0.00075"),
        ),
        seed=20260803,
        second_seed=None,
    )
    ready = campaign.Inspection(
        phase="bootstrap",
        status="READY_TO_EXECUTE",
        next_action="bootstrap",
        command=["bash", "scripts/elice/bootstrap_all.sh"],
        blockers=[],
        details={},
    )
    inspections = 0
    states: list[dict] = []

    def inspect(*_args, **_kwargs):
        nonlocal inspections
        inspections += 1
        if inspections > 1:
            raise AssertionError("system interpreter에서 post-bootstrap import 금지")
        return ready

    monkeypatch.setattr(campaign, "REPO_ROOT", repo)
    monkeypatch.setattr(campaign.sys, "prefix", str(tmp_path / "system-python"))
    monkeypatch.setattr(campaign, "load_contract", lambda *_args, **_kwargs: contract)
    monkeypatch.setattr(campaign, "inspect_campaign", inspect)
    monkeypatch.setattr(
        campaign,
        "atomic_write_state",
        lambda _path, payload, **_kwargs: states.append(payload),
    )
    monkeypatch.setattr(
        campaign.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        campaign,
        "build_execution_seal",
        lambda *_args, **_kwargs: {"fixture": "seal"},
    )
    monkeypatch.setattr(
        campaign, "verify_pre_execution_authority", lambda *_args, **_kwargs: None
    )

    result = campaign.main(
        [
            "--contract",
            str(contract.path),
            "--expected-contract-sha256",
            contract.sha256,
            "--state-out",
            str(tmp_path / "state.json"),
            "--execute-next",
        ]
    )

    assert result == 0
    assert inspections == 1
    assert len(states) == 2
    assert states[0]["next_action"] == "bootstrap"
    assert states[1]["phase"] == "bootstrap_transition"
    assert states[1]["next_action"] == "REINVOKE_WITH_EXACT_VENV_REQUIRED"
    assert states[1]["execution"]["returncode"] == 0
    assert states[1]["details"]["bootstrap_child_returncode"] == 0


def test_post_execution_inspection_failure_never_masks_child_returncode(
    monkeypatch, tmp_path
):
    contract = campaign.CampaignContract(
        path=tmp_path / "contract.json",
        sha256="a" * 64,
        expected_commit="b" * 40,
        expected_holdout_sha256="c" * 64,
        expected_transfer_manifest_sha256="d" * 64,
        raw_hash_workers=8,
        cublas_workspace_config=":4096:8",
        decoder_audit=None,
        candidates=(
            campaign.Candidate("0.7", "0.00075"),
            campaign.Candidate("1.0", "0.00075"),
        ),
        seed=20260803,
        second_seed=None,
    )
    ready = campaign.Inspection(
        phase="g0",
        status="READY_TO_EXECUTE",
        next_action="g0",
        command=["exact-python", "exact-g0.py"],
        blockers=[],
        details={},
    )
    inspections = iter((ready, RuntimeError("post inspection failed")))
    states: list[dict] = []

    def inspect(*_args, **_kwargs):
        value = next(inspections)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(campaign, "load_contract", lambda *_args, **_kwargs: contract)
    monkeypatch.setattr(campaign, "inspect_campaign", inspect)
    monkeypatch.setattr(
        campaign,
        "atomic_write_state",
        lambda _path, payload, **_kwargs: states.append(payload),
    )
    monkeypatch.setattr(
        campaign.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=7),
    )
    monkeypatch.setattr(
        campaign,
        "build_execution_seal",
        lambda *_args, **_kwargs: {"fixture": "seal"},
    )
    monkeypatch.setattr(
        campaign, "verify_pre_execution_authority", lambda *_args, **_kwargs: None
    )

    result = campaign.main(
        [
            "--contract",
            str(contract.path),
            "--expected-contract-sha256",
            contract.sha256,
            "--state-out",
            str(tmp_path / "state.json"),
            "--execute-next",
        ]
    )

    assert result == 7
    assert len(states) == 1
    assert states[0]["next_action"] == "g0"


def test_pre_execution_revalidation_blocks_script_mutation_after_state_write(
    monkeypatch, tmp_path
):
    repo, _commit, holdout, transfer = _repo(tmp_path)
    entrypoint = repo / "scripts/train/run_canonical_campaign.py"
    target = repo / "scripts/train/train.py"
    python = repo / ".venv/bin/python"
    for path, content in (
        (entrypoint, "#!/usr/bin/env python3\n"),
        (target, "#!/usr/bin/env python3\n"),
        (python, "#!/bin/sh\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "canonical entry fixture")
    commit = _git(repo, "rev-parse", "HEAD")
    contract_path = tmp_path / "campaign.json"
    contract_sha = _write_contract(
        contract_path, _contract_payload(commit, holdout, transfer)
    )
    contract = campaign.load_contract(contract_path, contract_sha, repo_root=repo)
    ready = campaign.Inspection(
        phase="canonical_pretrain",
        status="READY_TO_EXECUTE",
        next_action="canonical_pretrain",
        command=[str(python.absolute()), str(target.absolute())],
        blockers=[],
        details={},
    )
    states: list[dict] = []

    def mutate_after_first_state(_path, payload, **_kwargs):
        states.append(payload)
        if len(states) == 1:
            target.write_text("#!/usr/bin/env python3\n# mutated\n", encoding="utf-8")

    monkeypatch.setattr(campaign, "REPO_ROOT", repo)
    monkeypatch.setattr(campaign, "__file__", str(entrypoint))
    monkeypatch.setattr(campaign, "load_contract", lambda *_args, **_kwargs: contract)
    monkeypatch.setattr(campaign, "inspect_campaign", lambda *_args, **_kwargs: ready)
    monkeypatch.setattr(campaign, "atomic_write_state", mutate_after_first_state)
    original_run = subprocess.run

    def no_child(command, **kwargs):
        if command and command[0] == "git":
            return original_run(command, **kwargs)
        pytest.fail("authority mutation 뒤 child 실행 금지")

    monkeypatch.setattr(
        campaign.subprocess,
        "run",
        no_child,
    )

    result = campaign.main(
        [
            "--contract",
            str(contract_path),
            "--expected-contract-sha256",
            contract_sha,
            "--state-out",
            str(tmp_path / "state.json"),
            "--execute-next",
        ]
    )

    assert result == 2
    assert len(states) == 2
    assert states[-1]["next_action"] == "PRE_EXECUTION_AUTHORITY_CHANGED"
    assert "exact source" in states[-1]["blockers"][0]["message"]


def test_state_path_must_stay_outside_clean_checkout(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "state.json"
    payload = {"ok": True}
    campaign.atomic_write_state(outside, payload, repo_root=repo)
    assert json.loads(outside.read_text(encoding="utf-8")) == payload
    with pytest.raises(campaign.CampaignError, match="저장소 밖"):
        campaign.atomic_write_state(repo / "state.json", payload, repo_root=repo)
