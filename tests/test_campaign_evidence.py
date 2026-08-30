"""campaign v7 후보별 G0+gradient+20k+5k raw-evidence chain의 fail-closed 경계."""

from __future__ import annotations

import io
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import deep_anc.train.campaign_evidence as campaign_evidence_module

from deep_anc.train.campaign_evidence import (
    DNH_GRADIENT_DOMAIN,
    DNH_GRADIENT_NORM,
    DNH_GRADIENT_RECOMMENDATION_RULE,
    DNH_GRADIENT_SHARE_MAX,
    DNH_GRADIENT_SHARE_MIN,
    DNH_GRADIENT_TARGET,
    EVIDENCE_SCHEMA_VERSION,
    FAILED_G0_GRADIENT_RECEIPT_KIND,
    FAILED_G0_RECEIPT_KIND,
    G0_DETERMINISM_ENVIRONMENT_KIND,
    G0_RECEIPT_KIND,
    GRADIENT_RECEIPT_KIND,
    GRADIENT_RECEIPT_SCHEMA_VERSION,
    MEASURED_PROBE_SELECTION_SCORE,
    PILOT_TIE_MARGIN_DB,
    calibrate_dnh_output_gradient,
    configure_g0_evidence_determinism,
    publish_failed_g0_evidence,
    publish_failed_g0_gradient_recommendation,
    publish_g0_evidence,
    publish_gradient_budget_evidence,
    publish_prepilot_gradient_evidence,
    select_loss_pilot,
    snapshot_reference,
    _validate_recorded_metrics,
    validate_g0_receipt,
    validate_gradient_budget_receipt,
)
from deep_anc.train.evaluation_contract import snapshot_regular_file


ISSUER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train"
    / "issue_canonical_pretrain_prerequisite.py"
)


@pytest.mark.parametrize("role", ["loss_pilot", "measured_probe"])
def test_campaign_candidate_rejects_missing_world1_cuda_rng_before_scoring(
    tmp_path, monkeypatch, role
):
    """campaign issuer가 pilot/probe last.pt의 rng.cuda=None을 승인하지 않는다."""

    refs = {
        "best": SimpleNamespace(path=tmp_path / "best.pt", sha256="a" * 64),
        "last": SimpleNamespace(path=tmp_path / "last.pt", sha256="b" * 64),
        "metrics": SimpleNamespace(path=tmp_path / "metrics.npz", sha256="c" * 64),
        "manifest": SimpleNamespace(path=tmp_path / "manifest", sha256="d" * 64),
    }
    monkeypatch.setattr(
        campaign_evidence_module,
        "snapshot_from_reference",
        lambda _root, ref, **_kwargs: refs[ref],
    )
    monkeypatch.setattr(
        campaign_evidence_module,
        "canonical_recorded_manifest_for_data",
        lambda _data: "manifest",
    )
    monkeypatch.setattr(
        campaign_evidence_module,
        "repo_path",
        lambda root, _value, **_kwargs: root / "manifest",
    )
    state = {
        "cfg": {"experiment_role": role},
        "model": {},
        "rng": {"cuda": None},
    }
    monkeypatch.setattr(
        campaign_evidence_module,
        "_load_checkpoint",
        lambda *_args, **_kwargs: state,
    )

    with pytest.raises(ValueError, match="CUDA RNG state가 정확히 하나"):
        campaign_evidence_module._validate_pilot_checkpoint_pair(
            {
                "best_checkpoint": "best",
                "last_checkpoint": "last",
                "metrics": "metrics",
                "manifest": "manifest",
            },
            repo_root=tmp_path,
            canonical_cfg={"data": {}},
            canonical_contract={},
            expected_role=role,
            expected_primary_mode=(
                "measured" if role == "measured_probe" else "secondary_surrogate"
            ),
            expected_steps=5_000 if role == "measured_probe" else 20_000,
            expected_identity=(0.7, 0.5, 0.001),
            label=role,
        )
GRADIENT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train"
    / "measure_gradient_budget.py"
)


@pytest.fixture
def g0_deterministic_backend(monkeypatch: pytest.MonkeyPatch):
    """publisher unit test도 공식 G0와 같은 live backend에서 실행한다."""

    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment = configure_g0_evidence_determinism()
    try:
        yield environment
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.benchmark = previous[2]
        torch.backends.cudnn.deterministic = previous[3]


def _issuer_module():
    spec = importlib.util.spec_from_file_location("_campaign_issuer", ISSUER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _gradient_module():
    spec = importlib.util.spec_from_file_location("_gradient_budget_cli", GRADIENT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gradient_cli_separates_failed_g0_diagnostic_from_approved_prepilot():
    parser = _gradient_module().build_parser()
    failed = parser.parse_args(
        ["--failed-g0-receipt", "failed.json", "--out-dir", "diagnostic"]
    )
    assert failed.failed_g0_receipt == "failed.json"
    assert failed.g0_receipt is None
    selected = parser.parse_args(
        [
            "--checkpoint",
            "best.pt",
            "--authoritative-g0-receipt",
            "g0.json",
            "--out-dir",
            "selected",
        ]
    )
    assert selected.authoritative_g0_receipt == "g0.json"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--g0-receipt",
                "passed.json",
                "--failed-g0-receipt",
                "failed.json",
                "--out-dir",
                "ambiguous",
            ]
        )


@pytest.mark.parametrize(
    ("alpha", "literal"),
    [(0.7, "0.7"), (0.85, "0.85"), (1.0, "1.0")],
)
def test_issuer_resolves_canonical_config_with_the_selected_alpha(
    monkeypatch: pytest.MonkeyPatch, alpha: float, literal: str
) -> None:
    """winner가 기본 0.7이 아니어도 issuer/100k 계약이 같은 alpha를 쓴다."""

    module = _issuer_module()
    captured: list[str] = []

    def fake_load(_config, overrides):
        captured.extend(overrides)
        return {"fixture": True}

    monkeypatch.setattr(module, "load_train_config", fake_load)
    module._canonical_cfg(
        "configs/train_pretrain_tiny.yaml",
        "a" * 64,
        "b" * 64,
        loss_alpha=alpha,
        loss_lambda_dnh=0.000375,
    )
    assert f"loss.nmse_cvar_alpha={literal}" in captured
    assert "loss.lambda_dnh=0.000375" in captured


def test_g0_publisher_keeps_raw_model_batch_and_live_determinism_not_manual_metric(
    tmp_path, g0_deterministic_backend
):
    model = torch.nn.Linear(2, 1)
    receipt_path = publish_g0_evidence(
        repo_root=tmp_path,
        output_dir="results/evidence/g0",
        cfg={"fixture": True},
        model_state=model.state_dict(),
        batch={"x": torch.ones(4, 2, 8), "d": torch.zeros(4, 1, 8)},
        steps=500,
        mode="nominal",
        primary_mode="secondary_surrogate",
        require_nmse_db=-6.0,
        nmse_only=False,
        disable_loss_terms=[],
        determinism_environment=g0_deterministic_backend,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert set(receipt) == {
        "schema_version",
        "kind",
        "checkpoint",
        "batch",
        "environment",
    }
    assert receipt["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert receipt["kind"] == G0_RECEIPT_KIND
    assert not {"nmse_trusted_db", "all_finite", "score_db", "winner"} & set(receipt)
    for reference in (
        receipt["checkpoint"],
        receipt["batch"],
        receipt["environment"],
    ):
        snapshot = snapshot_regular_file(tmp_path / reference["path"])
        assert snapshot.sha256 == reference["sha256"]
    environment = json.loads(
        (tmp_path / receipt["environment"]["path"]).read_text(encoding="utf-8")
    )
    assert environment["kind"] == G0_DETERMINISM_ENVIRONMENT_KIND
    assert environment["torch_use_deterministic_algorithms"] is True
    assert environment["cudnn_benchmark"] is False
    assert environment["cudnn_deterministic"] is True


def test_failed_g0_is_sealed_as_recommendation_only_and_cannot_become_prepilot(
    tmp_path, g0_deterministic_backend
):
    failed_path = publish_failed_g0_evidence(
        repo_root=tmp_path,
        output_dir="results/evidence/g0-failed",
        cfg={"fixture": True},
        model_state=torch.nn.Linear(2, 1).state_dict(),
        batch={"x": torch.ones(4, 2, 8), "d": torch.zeros(4, 1, 8)},
        steps=500,
        mode="nominal",
        primary_mode="secondary_surrogate",
        require_nmse_db=-6.0,
        nmse_only=False,
        disable_loss_terms=[],
        determinism_environment=g0_deterministic_backend,
    )
    failed = json.loads(failed_path.read_text(encoding="utf-8"))
    assert failed["kind"] == FAILED_G0_RECEIPT_KIND
    with pytest.raises(ValueError, match="canonical G0 receipt가 아닙니다"):
        publish_prepilot_gradient_evidence(
            repo_root=tmp_path,
            output_dir="results/evidence/prepilot-from-failure",
            g0_receipt=failed_path,
        )
    assert not (tmp_path / "results/evidence/prepilot-from-failure").exists()

    calibration = {
        "gradient_domain": DNH_GRADIENT_DOMAIN,
        "gradient_norm": DNH_GRADIENT_NORM,
        "accepted_share_min": DNH_GRADIENT_SHARE_MIN,
        "accepted_share_max": DNH_GRADIENT_SHARE_MAX,
        "target_share": DNH_GRADIENT_TARGET,
        "recommendation_rule": DNH_GRADIENT_RECOMMENDATION_RULE,
        "approved": False,
        "current_lambda_dnh": 0.00075,
        "current_share": 0.6,
        "recommended_lambda_dnh": 0.000375,
        "recommended_share": 0.3,
        "current_budget": {"nmse": 1.0, "dnh": 0.6},
        "recommended_budget": {"nmse": 1.0, "dnh": 0.3},
    }
    recommendation_path = publish_failed_g0_gradient_recommendation(
        repo_root=tmp_path,
        output_dir="results/evidence/g0-failed-gradient",
        failed_g0_receipt=failed_path,
        calibration=calibration,
    )
    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
    assert recommendation["kind"] == FAILED_G0_GRADIENT_RECEIPT_KIND
    assert recommendation["campaign_eligible"] is False
    assert recommendation["required_next_action"] == "fresh_g0_from_scratch"
    assert recommendation["calibration_claim"] == calibration
    with pytest.raises(FileExistsError, match="덮어쓸 수 없습니다"):
        publish_failed_g0_gradient_recommendation(
            repo_root=tmp_path,
            output_dir="results/evidence/g0-failed-gradient",
            failed_g0_receipt=failed_path,
            calibration=calibration,
        )


def test_g0_publisher_rejects_inactive_or_forged_start_backend(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        forged = {
            "schema_version": 1,
            "kind": G0_DETERMINISM_ENVIRONMENT_KIND,
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_use_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": ":4096:8",
        }
        with pytest.raises(ValueError, match="publish determinism.*canonical"):
            publish_g0_evidence(
                repo_root=tmp_path,
                output_dir="results/evidence/g0-forged",
                cfg={"fixture": True},
                model_state=torch.nn.Linear(2, 1).state_dict(),
                batch={"x": torch.ones(4, 2, 8), "d": torch.zeros(4, 1, 8)},
                steps=500,
                mode="nominal",
                primary_mode="secondary_surrogate",
                require_nmse_db=-6.0,
                nmse_only=False,
                disable_loss_terms=[],
                determinism_environment=forged,
            )
        assert not (tmp_path / "results/evidence/g0-forged").exists()
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.benchmark = previous[2]
        torch.backends.cudnn.deterministic = previous[3]


def test_g0_evidence_determinism_requires_preconfigured_cublas_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="프로세스 시작 전 CUBLAS_WORKSPACE_CONFIG"):
        configure_g0_evidence_determinism()


def test_g0_validator_rejects_resealed_false_determinism_environment(
    tmp_path, g0_deterministic_backend
):
    receipt_path = publish_g0_evidence(
        repo_root=tmp_path,
        output_dir="results/evidence/g0-tampered",
        cfg={"fixture": True},
        model_state=torch.nn.Linear(2, 1).state_dict(),
        batch={"x": torch.ones(4, 2, 8), "d": torch.zeros(4, 1, 8)},
        steps=500,
        mode="nominal",
        primary_mode="secondary_surrogate",
        require_nmse_db=-6.0,
        nmse_only=False,
        disable_loss_terms=[],
        determinism_environment=g0_deterministic_backend,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    environment_path = tmp_path / receipt["environment"]["path"]
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["torch_use_deterministic_algorithms"] = False
    environment_path.write_text(
        json.dumps(environment, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt["environment"] = snapshot_reference(
        tmp_path, environment_path, label="tampered G0 environment"
    )
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="persisted determinism.*canonical"):
        validate_g0_receipt(
            snapshot_reference(tmp_path, receipt_path, label="tampered G0 receipt"),
            repo_root=tmp_path,
            canonical_cfg={},
            canonical_contract={"fixture": True},
        )


def test_gradient_publisher_is_no_replace_and_binds_external_checkpoint_bytes(tmp_path):
    checkpoint = tmp_path / "runs/pilot/ckpt/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture checkpoint bytes\n")
    batch_artifact = tmp_path / "results/evidence/g0/batch.pt"
    batch_artifact.parent.mkdir(parents=True)
    torch.save(
        {"x": torch.ones(4, 2, 8), "d": torch.zeros(4, 1, 8)},
        batch_artifact,
    )
    receipt_path = publish_gradient_budget_evidence(
        repo_root=tmp_path,
        output_dir="results/evidence/gradient",
        checkpoint=checkpoint,
        batch_artifact=batch_artifact,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["kind"] == GRADIENT_RECEIPT_KIND
    assert receipt["schema_version"] == GRADIENT_RECEIPT_SCHEMA_VERSION
    assert receipt["checkpoint"] == snapshot_reference(
        tmp_path, checkpoint, label="fixture checkpoint"
    )
    assert receipt["batch"] == snapshot_reference(
        tmp_path, batch_artifact, label="authoritative G0 batch"
    )
    assert sorted(path.name for path in receipt_path.parent.iterdir()) == ["receipt.json"]
    assert receipt["calibration_policy"] == {
        "gradient_domain": DNH_GRADIENT_DOMAIN,
        "gradient_norm": DNH_GRADIENT_NORM,
        "accepted_share_min": DNH_GRADIENT_SHARE_MIN,
        "accepted_share_max": DNH_GRADIENT_SHARE_MAX,
        "target_share": DNH_GRADIENT_TARGET,
        "recommendation_rule": DNH_GRADIENT_RECOMMENDATION_RULE,
    }
    with pytest.raises(FileExistsError, match="덮어쓸 수 없습니다"):
        publish_gradient_budget_evidence(
            repo_root=tmp_path,
            output_dir="results/evidence/gradient",
            checkpoint=checkpoint,
            batch_artifact=batch_artifact,
        )


def test_selected_gradient_rejects_same_bytes_copied_to_non_authoritative_batch_path(
    tmp_path,
):
    checkpoint = tmp_path / "runs/pilot/ckpt/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture checkpoint bytes\n")
    authoritative = tmp_path / "results/evidence/g0/batch.pt"
    copied = tmp_path / "results/evidence/copied/batch.pt"
    authoritative.parent.mkdir(parents=True)
    copied.parent.mkdir(parents=True)
    authoritative.write_bytes(b"identical fixed batch bytes")
    copied.write_bytes(authoritative.read_bytes())
    assert snapshot_regular_file(authoritative).sha256 == snapshot_regular_file(copied).sha256

    receipt_path = publish_gradient_budget_evidence(
        repo_root=tmp_path,
        output_dir="results/evidence/gradient-copied-batch",
        checkpoint=checkpoint,
        batch_artifact=copied,
    )
    with pytest.raises(ValueError, match="authoritative G0 fixed batch path/SHA"):
        validate_gradient_budget_receipt(
            snapshot_reference(tmp_path, receipt_path, label="gradient receipt"),
            repo_root=tmp_path,
            canonical_cfg={},
            canonical_contract={"fixture": True},
            expected_checkpoint_sha256=snapshot_regular_file(checkpoint).sha256,
            expected_identity=(0.7, 0.0, 0.00075),
            expected_batch_path=authoritative,
            expected_batch_sha256=snapshot_regular_file(authoritative).sha256,
        )


def test_prepilot_gradient_receipt_binds_the_exact_g0_receipt_checkpoint_and_batch(
    tmp_path, g0_deterministic_backend
):
    g0_path = publish_g0_evidence(
        repo_root=tmp_path,
        output_dir="results/evidence/g0-alpha07",
        cfg={"fixture": True},
        model_state=torch.nn.Linear(2, 1).state_dict(),
        batch={"x": torch.ones(4, 2, 8), "d": torch.zeros(4, 1, 8)},
        steps=500,
        mode="nominal",
        primary_mode="secondary_surrogate",
        require_nmse_db=-6.0,
        nmse_only=False,
        disable_loss_terms=[],
        determinism_environment=g0_deterministic_backend,
    )
    gradient_path = publish_prepilot_gradient_evidence(
        repo_root=tmp_path,
        output_dir="results/evidence/prepilot-alpha07",
        g0_receipt=g0_path,
    )
    g0 = json.loads(g0_path.read_text(encoding="utf-8"))
    gradient = json.loads(gradient_path.read_text(encoding="utf-8"))
    assert gradient["g0_receipt"] == snapshot_reference(
        tmp_path, g0_path, label="G0"
    )
    assert gradient["checkpoint"] == g0["checkpoint"]
    assert gradient["batch"] == g0["batch"]
    assert gradient["kind"] == "campaign_prepilot_dnh_output_gradient"
    with pytest.raises(FileExistsError, match="덮어쓸 수 없습니다"):
        publish_prepilot_gradient_evidence(
            repo_root=tmp_path,
            output_dir="results/evidence/prepilot-alpha07",
            g0_receipt=g0_path,
        )


def test_dnh_calibration_keeps_an_already_approved_lambda_without_target_chasing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[float | None] = []

    def recompute(_cfg, _state, _batch, *, lambda_dnh_override=None, **_kwargs):
        calls.append(lambda_dnh_override)
        return {"nmse": 1.0, "dnh": 0.27}

    monkeypatch.setattr(campaign_evidence_module, "_recompute_gradient_budget", recompute)
    result = calibrate_dnh_output_gradient(
        {"loss": {"lambda_dnh": 0.00075}},
        {"fixture": torch.ones(1)},
        {"x": torch.ones(4, 2, 8), "d": torch.ones(4, 1, 8)},
        repo_root=tmp_path,
    )
    assert result["gradient_domain"] == "model_output_y"
    assert result["approved"] is True
    assert result["current_lambda_dnh"] == pytest.approx(0.00075)
    assert result["recommended_lambda_dnh"] == pytest.approx(0.00075)
    assert result["recommended_share"] == pytest.approx(0.27)
    assert calls == [None]


def test_dnh_calibration_scales_out_of_range_lambda_and_actually_recomputes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[float | None] = []

    def recompute(_cfg, _state, _batch, *, lambda_dnh_override=None, **_kwargs):
        calls.append(lambda_dnh_override)
        effective = 0.00075 if lambda_dnh_override is None else lambda_dnh_override
        return {"nmse": 1.0, "dnh": 0.6 * effective / 0.00075}

    monkeypatch.setattr(campaign_evidence_module, "_recompute_gradient_budget", recompute)
    result = calibrate_dnh_output_gradient(
        {"loss": {"lambda_dnh": 0.00075}},
        {"fixture": torch.ones(1)},
        {"x": torch.ones(4, 2, 8), "d": torch.ones(4, 1, 8)},
        repo_root=tmp_path,
    )
    assert result["approved"] is False
    assert result["current_share"] == pytest.approx(0.6)
    assert result["recommended_lambda_dnh"] == pytest.approx(0.000375)
    assert result["recommended_share"] == pytest.approx(0.3)
    assert calls == [None, pytest.approx(0.000375)]


def test_dnh_calibration_rejects_an_unverified_analytic_recommendation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    calls = 0

    def recompute(_cfg, _state, _batch, *, lambda_dnh_override=None, **_kwargs):
        nonlocal calls
        calls += 1
        return {"nmse": 1.0, "dnh": 0.6 if calls == 1 else 0.41}

    monkeypatch.setattr(campaign_evidence_module, "_recompute_gradient_budget", recompute)
    with pytest.raises(ValueError, match="추천 λ 실제 재계산 share"):
        calibrate_dnh_output_gradient(
            {"loss": {"lambda_dnh": 0.00075}},
            {"fixture": torch.ones(1)},
            {"x": torch.ones(4, 2, 8), "d": torch.ones(4, 1, 8)},
            repo_root=tmp_path,
        )


@pytest.mark.parametrize(
    "validator, kind",
    [
        (validate_g0_receipt, G0_RECEIPT_KIND),
        (validate_gradient_budget_receipt, GRADIENT_RECEIPT_KIND),
    ],
)
def test_arbitrary_verified_json_cannot_act_as_campaign_evidence(
    tmp_path, validator, kind
):
    bogus = tmp_path / "results/evidence/verified.json"
    bogus.parent.mkdir(parents=True)
    bogus.write_text('{"verified":true,"nmse_trusted_db":-99.0}\n', encoding="utf-8")
    reference = snapshot_reference(tmp_path, bogus, label="bogus evidence")
    kwargs = {
        "repo_root": tmp_path,
        "canonical_cfg": {},
        # receipt schema는 canonical target semantics보다 먼저 확인돼야 한다.
        "canonical_contract": {"fixture": True},
    }
    if validator is validate_gradient_budget_receipt:
        kwargs.update(
            expected_checkpoint_sha256="a" * 64,
            expected_identity=(0.7, 0.0, 0.00075),
            expected_batch_path=tmp_path / "authoritative-batch.pt",
            expected_batch_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="key 집합|schema/kind"):
        validator(reference, **kwargs)


def _campaign_artifact(path: Path) -> dict:
    snapshot = snapshot_regular_file(path)
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size_bytes": len(snapshot.content),
        "sha256": snapshot.sha256,
    }


def _campaign_contract(artifact: dict, *, name: str = "rir_bank") -> dict:
    return {
        "source": {"git_commit": "1" * 40, "source_tree_sha256": "2" * 64},
        "input_generation": {
            "bootstrap_receipt_sha256": "3" * 64,
            "transfer_manifest_sha256": "4" * 64,
            "recorded_transfer_aggregate_sha256": "5" * 64,
        },
        "artifacts": {name: artifact},
    }


@pytest.mark.parametrize("artifact_name", ["rir_bank", "source_manifest_generation"])
def test_campaign_rejects_rir_or_manifest_generation_drift_from_canonical(
    tmp_path, monkeypatch, artifact_name
):
    canonical_rir = tmp_path / "canonical-rir.npz"
    changed_rir = tmp_path / "changed-rir.npz"
    canonical_rir.write_bytes(b"canonical rir bytes")
    changed_rir.write_bytes(b"changed rir bytes")
    canonical = _campaign_contract(
        _campaign_artifact(canonical_rir), name=artifact_name
    )
    derivative = _campaign_contract(
        _campaign_artifact(changed_rir), name=artifact_name
    )
    monkeypatch.setattr(
        campaign_evidence_module,
        "validate_embedded_experiment_contract",
        lambda _cfg: derivative,
    )
    with pytest.raises(ValueError, match="공통 학습 artifact.*canonical"):
        campaign_evidence_module._validate_source_and_artifacts(
            {},
            canonical_cfg={},
            canonical_contract=canonical,
            root=tmp_path,
            label="fixture derivative",
            embedded=True,
        )


def test_campaign_reopens_current_artifact_bytes_instead_of_trusting_stamp(
    tmp_path, monkeypatch
):
    rir = tmp_path / "rir.npz"
    rir.write_bytes(b"stamped rir bytes")
    contract = _campaign_contract(_campaign_artifact(rir))
    rir.write_bytes(b"mutated after stamp")
    monkeypatch.setattr(
        campaign_evidence_module,
        "validate_embedded_experiment_contract",
        lambda _cfg: contract,
    )
    with pytest.raises(ValueError, match="rir_bank.*embedded contract"):
        campaign_evidence_module._validate_source_and_artifacts(
            {},
            canonical_cfg={},
            canonical_contract=contract,
            root=tmp_path,
            label="fixture derivative",
            embedded=True,
        )


def _candidate(alpha: float, score: float, lambda_dnh: float | None = None) -> dict:
    if lambda_dnh is None:
        lambda_dnh = {0.7: 0.00075, 0.85: 0.0005, 1.0: 0.000375}[alpha]
    return {
        "identity": (alpha, 0.0, lambda_dnh),
        "score_db": score,
        "selection_score_source": MEASURED_PROBE_SELECTION_SCORE,
        "best_snapshot": SimpleNamespace(sha256=("a" if alpha == 0.7 else "b") * 64),
    }


def test_loss_selection_uses_measured_probe_margin_and_never_accepts_manual_winner():
    with pytest.raises(ValueError, match="alpha=0.85"):
        select_loss_pilot([_candidate(0.7, -4.0), _candidate(1.0, -3.9)])

    result = select_loss_pilot(
        [
            _candidate(0.7, -4.0),
            _candidate(1.0, -3.9),
            _candidate(0.85, -4.05),
        ]
    )
    # winner is derived from raw scores + predeclared tie rule; no ledger-supplied
    # `winner` field participates.
    assert result["winner_identity"] == (0.7, 0.0, 0.00075)


def test_loss_selection_allows_alpha_specific_lambda_but_rejects_duplicate_alpha():
    result = select_loss_pilot([_candidate(0.7, -4.0), _candidate(1.0, -2.0)])
    assert result["winner_identity"][2] == pytest.approx(0.00075)
    with pytest.raises(ValueError, match="alpha 중복"):
        select_loss_pilot(
            [
                _candidate(0.7, -4.0, 0.00075),
                _candidate(0.7, -3.0, 0.0005),
                _candidate(1.0, -2.0),
            ]
        )


def test_loss_selection_rejects_surrogate_pilot_score_source():
    candidate = _candidate(0.7, -99.0)
    candidate["selection_score_source"] = "surrogate_pilot_recorded_val"
    with pytest.raises(ValueError, match="measured-probe recorded-val"):
        select_loss_pilot([candidate, _candidate(1.0, -2.0)])


def test_issuer_pairs_every_pilot_and_probe_by_exact_cli_order():
    module = _issuer_module()
    args = SimpleNamespace(
        g0_receipt=["g0-07", "g0-10"],
        prepilot_gradient_receipt=["gradient-07", "gradient-10"],
        pilot_best=["pilot07-best", "pilot10-best"],
        pilot_last=["pilot07-last", "pilot10-last"],
        pilot_metrics=["pilot07-metrics", "pilot10-metrics"],
        pilot_manifest=None,
        probe_best=["probe07-best", "probe10-best"],
        probe_last=["probe07-last", "probe10-last"],
        probe_metrics=["probe07-metrics", "probe10-metrics"],
        probe_manifest=None,
        probe_init_checkpoint=["pilot07-best", "pilot10-best"],
    )
    generation_manifest = (
        "data/manifests/recorded_generations/highband-coverage-v1/recorded.jsonl"
    )
    chains = module._ordered_candidate_inputs(
        args, default_manifest=generation_manifest
    )
    assert len(chains) == 2
    assert chains[0][0] == "g0-07"
    assert chains[0][1] == "gradient-07"
    assert chains[0][2] == "pilot07-best"
    assert chains[0][6] == "probe07-best"
    assert chains[0][10] == "pilot07-best"
    assert chains[0][5] == generation_manifest
    assert chains[0][9] == generation_manifest
    assert chains[1][0] == "g0-10"
    assert chains[1][1] == "gradient-10"
    assert chains[1][2] == "pilot10-best"
    assert chains[1][6] == "probe10-best"
    assert chains[1][10] == "pilot10-best"

    args.probe_metrics = ["probe07-metrics"]
    with pytest.raises(ValueError, match="같은 개수와 순서"):
        module._ordered_candidate_inputs(args, default_manifest=generation_manifest)


def test_measured_probe_policy_uses_current_generation_manifest():
    generation_path = (
        "data/manifests/recorded_generations/highband-coverage-v1/generation.json"
    )
    generation_manifest = (
        "data/manifests/recorded_generations/highband-coverage-v1/recorded.jsonl"
    )
    canonical_cfg = {
        "data": {
            "recorded_generation": generation_path,
            "recorded_generation_sha256": "a" * 64,
        }
    }
    probe_cfg = deepcopy(campaign_evidence_module.CANONICAL_MEASURED_PROBE_POLICY)
    probe_cfg["recorded_manifest"] = generation_manifest

    assert len(probe_cfg["readiness"]) == 21
    assert probe_cfg["readiness"]["recorded_subband_coverage_report_dir"] == (
        "results/data_audit/recorded_subband_coverage"
    )
    assert probe_cfg["readiness"]["recorded_source_pool_csv"] == [
        "data/source_pool_v2/sources.csv",
        "data/source_pool/sources.csv",
    ]

    campaign_evidence_module._validate_measured_probe_distribution_policy(
        probe_cfg, canonical_cfg, label="fixture probe"
    )

    probe_cfg["recorded_manifest"] = (
        campaign_evidence_module.CANONICAL_RECORDED_VAL_MANIFEST
    )
    with pytest.raises(ValueError, match="recorded_manifest"):
        campaign_evidence_module._validate_measured_probe_distribution_policy(
            probe_cfg, canonical_cfg, label="fixture probe"
        )


@pytest.mark.parametrize(
    "missing_key",
    ["recorded_subband_coverage_report_dir", "recorded_source_pool_csv"],
)
def test_campaign_rejects_measured_probe_without_shared_readiness_gate(missing_key):
    canonical_cfg = {"data": {}}
    probe_cfg = deepcopy(campaign_evidence_module.CANONICAL_MEASURED_PROBE_POLICY)
    probe_cfg["readiness"].pop(missing_key)

    with pytest.raises(ValueError, match="readiness"):
        campaign_evidence_module._validate_measured_probe_distribution_policy(
            probe_cfg, canonical_cfg, label="fixture probe"
        )


def test_pilot_score_is_recomputed_from_raw_segment_metrics_not_npz_summary(tmp_path):
    from tests.test_finetune_pipeline_cli import (
        _canonical_sampling_checkpoint_cfg,
        _recorded_val_metric_payload,
    )

    checkpoint = tmp_path / "runs/pilot/best.pt"
    manifest = tmp_path / "data/manifests/recorded_regrouped.jsonl"
    checkpoint.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    torch.save(
        {
            "cfg": _canonical_sampling_checkpoint_cfg(),
            "model": {"weight": torch.ones(1)},
        },
        checkpoint,
    )
    checkpoint_snapshot = snapshot_regular_file(checkpoint)
    contract_sha = "c" * 64
    payload = _recorded_val_metric_payload(
        checkpoint=checkpoint,
        manifest=manifest,
        contract_sha=contract_sha,
        margin_db=1.0,
    )
    payload.update(
        g4_metric_scope=np.asarray("diagnostic_noncanonical"),
        allow_surrogate=np.asarray(True),
        physics_status=np.asarray(
            "secondary_surrogate_representation_pretrain"
        ),
        primary_path_sha256=np.asarray("p" * 64),
        secondary_path_sha256=np.asarray("s" * 64),
    )
    manifest_snapshot = snapshot_regular_file(manifest)
    expected_worst10 = -1.0
    metrics = tmp_path / "results/pilot/metrics.npz"
    metrics.parent.mkdir(parents=True)
    np.savez(metrics, **payload)
    canonical_contract = {
        "artifacts": {
            "primary_path": {"sha256": "p" * 64},
            "secondary_path": {"sha256": "s" * 64},
        }
    }
    assert _validate_recorded_metrics(
        snapshot_regular_file(metrics),
        label="fixture pilot",
        checkpoint_snapshot=checkpoint_snapshot,
        checkpoint_contract_sha256=contract_sha,
        manifest_snapshot=manifest_snapshot,
        canonical_contract=canonical_contract,
        expected_surrogate=True,
        expected_physics="secondary_surrogate_representation_pretrain",
    ) == pytest.approx(expected_worst10)

    # 같은 raw per-segment 배열에 사람이 선호하는 score scalar만 덮어쓰면 ledger는
    # 발급되지 않는다.
    payload["nmse_trusted_worst10_mean_db"] = np.asarray(-99.0)
    np.savez(metrics, **payload)
    with pytest.raises(ValueError, match="raw segment 재계산값"):
        _validate_recorded_metrics(
            snapshot_regular_file(metrics),
            label="fixture pilot",
            checkpoint_snapshot=checkpoint_snapshot,
            checkpoint_contract_sha256=contract_sha,
            manifest_snapshot=manifest_snapshot,
            canonical_contract=canonical_contract,
            expected_surrogate=True,
            expected_physics="secondary_surrogate_representation_pretrain",
        )
