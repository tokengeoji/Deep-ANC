from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from deep_anc.config import load_train_config
from deep_anc.train import second_seed_prerequisite as second


ISSUER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/train/issue_second_seed_prerequisite.py"
)


def _issuer_module():
    spec = importlib.util.spec_from_file_location("_second_seed_issuer_test", ISSUER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _ref(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    root = tmp_path / "repo"
    root.mkdir()
    campaign_sha = "c" * 64
    pretrain_neutral = "n" * 64
    bootstrap_sha = "b" * 64
    loss_sha = "d" * 64
    smoke_target = "e" * 64

    ledger = _write(
        root / second.CANONICAL_CAMPAIGN_PREREQUISITE, b"immutable-v7-ledger\n"
    )
    pretrain_checkpoint = _write(root / "runs/primary-pretrain/ckpt/best.pt", b"pretrain")
    fine_checkpoint = _write(root / "runs/primary-finetune/ckpt/best.pt", b"fine")

    decision = {"status": "borderline", "minimum_margin_db": 0.1}
    selection_payload = {
        "schema_version": 2,
        "selection_split": "val",
        "seed": second.PRIMARY_SEED,
        "seed_neutral_campaign_sha256": campaign_sha,
        "decision": decision,
        "selected": {
            "checkpoint": fine_checkpoint.relative_to(root).as_posix(),
            "checkpoint_sha256": hashlib.sha256(fine_checkpoint.read_bytes()).hexdigest(),
        },
    }
    selection = _write(
        root / "results/primary/audit/recorded_val_selection.json",
        (json.dumps(selection_payload, sort_keys=True) + "\n").encode(),
    )

    target_root = root / second.SMOKE_ROOT / smoke_target
    environment = _write(target_root / "environment_receipt.json", b"{}\n")
    telemetry = _write(target_root / "telemetry.json", b"{}\n")
    environment_ref = _ref(root, environment)
    telemetry_ref = _ref(root, telemetry)
    receipt = _write(
        target_root / "receipt.json",
        (
            json.dumps(
                {
                    "environment_receipt": environment_ref,
                    "telemetry": telemetry_ref,
                },
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )

    primary_pretrain_cfg = {
        "contract": "primary-pretrain-contract",
        "experiment_role": "canonical_pretrain",
        "seed": second.PRIMARY_SEED,
        "init_eligible": True,
        "campaign_prerequisite": second.CANONICAL_CAMPAIGN_PREREQUISITE,
        "campaign_prerequisite_sha256": _ref(root, ledger)["sha256"],
        "data": {"bootstrap_receipt_sha256": bootstrap_sha},
        "loss_selection_sha256": loss_sha,
    }
    primary_fine_cfg = {
        "contract": "primary-fine-contract",
        "experiment_role": "canonical_finetune",
        "seed": second.PRIMARY_SEED,
        "init_eligible": False,
        "init_ckpt": pretrain_checkpoint.relative_to(root).as_posix(),
        "loss_selection_sha256": loss_sha,
    }
    secondary_cfg = {
        "contract": "secondary-pretrain-contract",
        "experiment_role": "canonical_pretrain",
        "seed": second.SECONDARY_SEED,
        "init_eligible": True,
        "campaign_prerequisite": second.CANONICAL_CAMPAIGN_PREREQUISITE,
        "campaign_prerequisite_sha256": _ref(root, ledger)["sha256"],
        "data": {"bootstrap_receipt_sha256": bootstrap_sha},
        "loss_selection_sha256": loss_sha,
    }

    def checkpoint_cfg(snapshot, *, label):
        del label
        if snapshot.content == b"fine":
            return primary_fine_cfg
        if snapshot.content == b"pretrain":
            return primary_pretrain_cfg
        raise AssertionError(snapshot.path)

    monkeypatch.setattr(second, "_checkpoint_cfg", checkpoint_cfg)
    monkeypatch.setattr(
        second,
        "validate_recorded_val_selection",
        lambda payload, *, repo_root: decision,
    )
    monkeypatch.setattr(second, "validate_canonical_training_policy", lambda cfg: None)
    monkeypatch.setattr(
        second,
        "validate_embedded_experiment_contract",
        lambda cfg: {"sha256": cfg["contract"]},
    )

    def completion(_path, *, expected_role, expected_init_eligible, repo_root):
        assert repo_root == root
        assert expected_init_eligible is (expected_role == "canonical_pretrain")
        return {
            "experiment_contract_sha256": (
                "primary-fine-contract"
                if expected_role == "canonical_finetune"
                else "primary-pretrain-contract"
            )
        }

    monkeypatch.setattr(second, "validate_completion_receipt", completion)
    primary_validations: list[dict] = []
    monkeypatch.setattr(
        second,
        "validate_canonical_pretrain_prerequisites",
        lambda cfg, *, repo_root: primary_validations.append(cfg),
    )
    monkeypatch.setattr(
        second,
        "seed_neutral_campaign_sha256",
        lambda cfg: pretrain_neutral,
    )
    monkeypatch.setattr(
        second,
        "build_a100_pretrain_smoke_target",
        lambda cfg, *, repo_root: {"sha256": smoke_target},
    )
    smoke_validations: list[tuple[dict, str]] = []
    monkeypatch.setattr(
        second,
        "validate_a100_pretrain_smoke_receipt",
        lambda payload, *, repo_root, expected_smoke_target_sha256: smoke_validations.append(
            (payload, expected_smoke_target_sha256)
        ),
    )
    event_paths = {
        name: root / f"results/recorded_test_ledger/campaign/{name}.json"
        for name in ("issued", "running", "completed", "failed")
    }
    monkeypatch.setattr(
        second,
        "canonical_test_ledger_event_paths_from_payload",
        lambda payload, *, repo_root: event_paths,
    )

    payload = {
        "schema_version": second.SCHEMA_VERSION,
        "kind": second.KIND,
        "shared": {
            "campaign_prerequisite": _ref(root, ledger),
            "bootstrap_receipt_sha256": bootstrap_sha,
            "loss_selection_sha256": loss_sha,
        },
        "primary": {
            "seed": second.PRIMARY_SEED,
            "recorded_val_selection": _ref(root, selection),
            "seed_neutral_campaign_sha256": campaign_sha,
        },
        "secondary": {
            "seed": second.SECONDARY_SEED,
            "a100_smoke_resume": {
                "target_sha256": smoke_target,
                "evidence": _ref(root, receipt),
                "environment_receipt": environment_ref,
                "telemetry": telemetry_ref,
            },
        },
    }
    return {
        "root": root,
        "campaign_sha": campaign_sha,
        "secondary_cfg": secondary_cfg,
        "payload": payload,
        "selection": selection,
        "event_paths": event_paths,
        "primary_validations": primary_validations,
        "smoke_validations": smoke_validations,
    }


def test_validates_primary_raw_chain_secondary_smoke_and_fixed_published_path(
    tmp_path, monkeypatch
):
    case = _fixture(tmp_path, monkeypatch)
    cfg = case["secondary_cfg"]
    payload = case["payload"]
    root = case["root"]

    assert second.validate_second_seed_prerequisite_payload(
        cfg, payload, repo_root=root
    ) == payload
    assert [row["seed"] for row in case["primary_validations"]] == [
        second.PRIMARY_SEED
    ]
    assert case["smoke_validations"][0][1] == "e" * 64

    destination = second.second_seed_prerequisite_path(
        case["campaign_sha"], repo_root=root
    )
    _write(destination, second.prerequisite_json_bytes(payload))
    cfg[second.CONFIG_PATH_KEY] = destination.relative_to(root).as_posix()
    cfg[second.CONFIG_SHA256_KEY] = hashlib.sha256(destination.read_bytes()).hexdigest()
    assert second.validate_second_seed_prerequisites(cfg, repo_root=root) == payload


@pytest.mark.parametrize("event", ["issued", "running", "completed", "failed"])
def test_rejects_any_primary_test_ledger_event(tmp_path, monkeypatch, event):
    case = _fixture(tmp_path, monkeypatch)
    _write(case["event_paths"][event], b"touched\n")
    with pytest.raises(ValueError, match="test ledger가 이미 열렸습니다"):
        second.validate_second_seed_prerequisite_payload(
            case["secondary_cfg"], case["payload"], repo_root=case["root"]
        )


def test_legitimate_completed_cross_seed_ledger_keeps_prerequisite_valid(
    tmp_path, monkeypatch
):
    case = _fixture(tmp_path, monkeypatch)
    for event in ("issued", "running", "completed"):
        _write(case["event_paths"][event], b"immutable\n")
    selection_sha = hashlib.sha256(case["selection"].read_bytes()).hexdigest()
    final = (
        case["root"]
        / second.CROSS_SEED_ROOT
        / case["campaign_sha"]
        / "recorded_val_selection.json"
    )
    _write(
        final,
        (
            json.dumps(
                {
                    "seed_selections": [
                        {
                            "seed": second.PRIMARY_SEED,
                            "path": str(case["selection"].absolute()),
                            "sha256": selection_sha,
                        }
                    ]
                },
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )
    monkeypatch.setattr(
        second,
        "validate_test_open_selection",
        lambda _payload, *, repo_root: None,
    )

    assert second.validate_second_seed_prerequisite_payload(
        case["secondary_cfg"], case["payload"], repo_root=case["root"]
    ) == case["payload"]


def test_rejects_primary_secondary_pretrain_neutral_digest_mismatch(
    tmp_path, monkeypatch
):
    case = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        second,
        "seed_neutral_campaign_sha256",
        lambda cfg: "a" * 64 if cfg["seed"] == second.PRIMARY_SEED else "f" * 64,
    )
    with pytest.raises(ValueError, match="pretrain seed-neutral digest가 다릅니다"):
        second.validate_second_seed_prerequisite_payload(
            case["secondary_cfg"], case["payload"], repo_root=case["root"]
        )


def test_rejects_symlinked_primary_selection_before_checkpoint_validation(
    tmp_path, monkeypatch
):
    case = _fixture(tmp_path, monkeypatch)
    link = case["root"] / "results/primary/audit/selection-link.json"
    link.symlink_to(case["selection"].name)
    case["payload"]["primary"]["recorded_val_selection"] = {
        "path": link.relative_to(case["root"]).as_posix(),
        "sha256": hashlib.sha256(case["selection"].read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="symlink"):
        second.validate_second_seed_prerequisite_payload(
            case["secondary_cfg"], case["payload"], repo_root=case["root"]
        )


def test_fixed_path_rejects_non_sha_campaign(tmp_path):
    with pytest.raises(ValueError, match="64자리"):
        second.second_seed_prerequisite_path("not-a-sha", repo_root=tmp_path)


def test_bare_secondary_canonical_pretrain_is_rejected_before_artifact_binding():
    with pytest.raises(ValueError, match="second_seed_prerequisite"):
        load_train_config(
            "configs/train_pretrain_tiny.yaml",
            [
                "seed=20260903",
                f"data.bootstrap_receipt_sha256={'a' * 64}",
                f"campaign_prerequisite_sha256={'b' * 64}",
            ],
        )


def test_issuer_reloads_published_cfg_before_source_trust(tmp_path, monkeypatch):
    issuer = _issuer_module()
    root = tmp_path / "repo"
    root.mkdir()
    selection = _write(
        root / "primary-selection.json",
        (
            json.dumps({"seed_neutral_campaign_sha256": "c" * 64}) + "\n"
        ).encode(),
    )
    payload = {
        "schema_version": 1,
        "kind": "fixture",
        "published": True,
    }
    destination = second.second_seed_prerequisite_path("c" * 64, repo_root=root)
    loads: list[dict] = []
    source_trust: list[dict] = []
    published_validation: list[dict] = []

    def load(_config, overrides):
        cfg = {
            "artifact_present_at_resolve": destination.is_file(),
            "overrides": list(overrides),
        }
        loads.append(cfg)
        return cfg

    monkeypatch.setattr(issuer, "REPO_ROOT", root)
    monkeypatch.setattr(issuer, "load_train_config", load)
    monkeypatch.setattr(
        issuer,
        "build_second_seed_prerequisite_payload",
        lambda *_args, **_kwargs: payload,
    )
    monkeypatch.setattr(
        issuer,
        "validate_second_seed_prerequisite_payload",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        issuer,
        "require_exact_source_trust",
        lambda cfg, **_kwargs: source_trust.append(cfg),
    )
    monkeypatch.setattr(
        issuer,
        "validate_second_seed_prerequisites",
        lambda cfg, **_kwargs: published_validation.append(cfg),
    )

    result = issuer.main(
        [
            "--config",
            str(root / "configs/train_pretrain_tiny.yaml"),
            "--bootstrap-receipt-sha256",
            "a" * 64,
            "--campaign-prerequisite-sha256",
            "b" * 64,
            "--loss-alpha",
            "0.7",
            "--loss-lambda-dnh",
            "0.00075",
            "--primary-selection",
            str(selection),
            "--secondary-smoke-receipt",
            str(root / "smoke/receipt.json"),
            "--secondary-smoke-environment-receipt",
            str(root / "smoke/environment_receipt.json"),
            "--secondary-smoke-telemetry",
            str(root / "smoke/telemetry.json"),
            "--out",
            str(destination),
        ]
    )

    assert result == 0
    assert [row["artifact_present_at_resolve"] for row in loads] == [
        False,
        False,
        True,
    ]
    assert source_trust == [loads[-1]]
    assert published_validation == [loads[-1]]
    assert destination.is_file()
    assert loads[-1]["overrides"][-1].endswith(
        json.dumps(hashlib.sha256(destination.read_bytes()).hexdigest())
    )
