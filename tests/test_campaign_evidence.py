"""campaign v5 raw-evidence receipt의 no-manual-claim 경계."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deep_anc.train.campaign_evidence import (
    EVIDENCE_SCHEMA_VERSION,
    G0_RECEIPT_KIND,
    GRADIENT_RECEIPT_KIND,
    PILOT_TIE_MARGIN_DB,
    publish_g0_evidence,
    publish_gradient_budget_evidence,
    select_loss_pilot,
    snapshot_reference,
    _validate_recorded_metrics,
    validate_g0_receipt,
    validate_gradient_budget_receipt,
)
from deep_anc.train.evaluation_contract import snapshot_regular_file


def test_g0_publisher_keeps_only_raw_model_and_batch_not_manual_metric(tmp_path):
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
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert set(receipt) == {"schema_version", "kind", "checkpoint", "batch"}
    assert receipt["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert receipt["kind"] == G0_RECEIPT_KIND
    assert not {"nmse_trusted_db", "all_finite", "score_db", "winner"} & set(receipt)
    for reference in (receipt["checkpoint"], receipt["batch"]):
        snapshot = snapshot_regular_file(tmp_path / reference["path"])
        assert snapshot.sha256 == reference["sha256"]


def test_gradient_publisher_is_no_replace_and_binds_external_checkpoint_bytes(tmp_path):
    checkpoint = tmp_path / "runs/pilot/ckpt/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture checkpoint bytes\n")
    receipt_path = publish_gradient_budget_evidence(
        repo_root=tmp_path,
        output_dir="results/evidence/gradient",
        checkpoint=checkpoint,
        batch={"x": torch.ones(4, 2, 8), "d": torch.zeros(4, 1, 8)},
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["kind"] == GRADIENT_RECEIPT_KIND
    assert receipt["checkpoint"] == snapshot_reference(
        tmp_path, checkpoint, label="fixture checkpoint"
    )
    with pytest.raises(FileExistsError, match="덮어쓸 수 없습니다"):
        publish_gradient_budget_evidence(
            repo_root=tmp_path,
            output_dir="results/evidence/gradient",
            checkpoint=checkpoint,
            batch={"x": torch.ones(4, 2, 8), "d": torch.zeros(4, 1, 8)},
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
            expected_pair=(0.7, 0.0),
        )
    with pytest.raises(ValueError, match="key 집합|schema/kind"):
        validator(reference, **kwargs)


def _candidate(alpha: float, score: float) -> dict:
    return {
        "pair": (alpha, 0.0),
        "score_db": score,
        "best_snapshot": SimpleNamespace(sha256=("a" if alpha == 0.7 else "b") * 64),
    }


def test_loss_selection_derives_margin_rule_and_never_accepts_manual_winner():
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
    assert result["winner_pair"] == (0.7, 0.0)


def _recorded_metrics_bytes(
    *,
    checkpoint_sha256: str,
    manifest_sha256: str,
    contract_sha256: str,
    trusted_worst10: float,
) -> bytes:
    """실 evaluator가 저장하는 selection 핵심 필드만 가진 raw NPZ fixture."""

    trusted = np.asarray([-4.0, -3.0, -2.0, -1.0, -0.5, -0.3, -0.2, -0.1, 0.0, 1.0])
    fullband = np.asarray([-3.0, -2.0, -1.0, -0.5, -0.2, 0.0, 0.1, 0.2, 0.3, 0.4])
    payload = io.BytesIO()
    np.savez(
        payload,
        split=np.asarray("val"),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        manifest_sha256=np.asarray(manifest_sha256),
        experiment_contract_sha256=np.asarray(contract_sha256),
        allow_surrogate=np.asarray(True),
        physics_status=np.asarray("secondary_surrogate_representation_pretrain"),
        primary_path_sha256=np.asarray("p" * 64),
        secondary_path_sha256=np.asarray("s" * 64),
        n_sessions=np.asarray(1),
        n_groups=np.asarray(1),
        n_segments=np.asarray(trusted.size),
        per_segment_trusted_db=trusted,
        per_segment_fullband_db=fullband,
        nmse_trusted_mean_db=np.asarray(float(np.mean(trusted))),
        nmse_trusted_worst10_mean_db=np.asarray(trusted_worst10),
        nmse_fullband_mean_db=np.asarray(float(np.mean(fullband))),
        nmse_fullband_worst10_mean_db=np.asarray(float(np.max(fullband))),
        g4_worst_source_trusted_mean_db=np.asarray(1.0),
        g4_worst_source_trusted_worst10_db=np.asarray(1.0),
        g4_worst_octave_worst10_db=np.asarray(1.0),
    )
    return payload.getvalue()


def test_pilot_score_is_recomputed_from_raw_segment_metrics_not_npz_summary(tmp_path):
    checkpoint = tmp_path / "runs/pilot/best.pt"
    manifest = tmp_path / "data/manifests/recorded_regrouped.jsonl"
    checkpoint.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"raw checkpoint bytes\n")
    manifest.write_text('{"session_id":"fixture"}\n', encoding="utf-8")
    checkpoint_snapshot = snapshot_regular_file(checkpoint)
    manifest_snapshot = snapshot_regular_file(manifest)
    contract_sha = "c" * 64
    expected_worst10 = 1.0  # 10 segments의 worst 10% = max(=1.0)
    metrics = tmp_path / "results/pilot/metrics.npz"
    metrics.parent.mkdir(parents=True)
    metrics.write_bytes(
        _recorded_metrics_bytes(
            checkpoint_sha256=checkpoint_snapshot.sha256,
            manifest_sha256=manifest_snapshot.sha256,
            contract_sha256=contract_sha,
            trusted_worst10=expected_worst10,
        )
    )
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
    metrics.write_bytes(
        _recorded_metrics_bytes(
            checkpoint_sha256=checkpoint_snapshot.sha256,
            manifest_sha256=manifest_snapshot.sha256,
            contract_sha256=contract_sha,
            trusted_worst10=-99.0,
        )
    )
    with pytest.raises(ValueError, match="summary scalar.*raw segment"):
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
