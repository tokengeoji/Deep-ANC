from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from deep_anc.train.stage2_2khz_git_authority import (
    verify_source_commit_ancestor,
    verify_tracked_head_authority,
)
from deep_anc.train.stage2_2khz_runner import _preverify_campaign_checkout


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _clean_authority_repository(root: Path) -> tuple[dict, str, str]:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "stage2-test@example.invalid")
    _git(root, "config", "user.name", "Stage2 Test")
    (root / "source.txt").write_text("artifact source\n", encoding="utf-8")
    _git(root, "add", "source.txt")
    _git(root, "commit", "-q", "-m", "source")
    source_commit = _git(root, "rev-parse", "HEAD")

    external_sha = _write_json(
        root / "artifacts" / "external.json",
        {"artifact_source_commit_sha": source_commit},
    )
    campaign = {
        "external_contracts": {
            "canonical_pretrain": {
                "path": "artifacts/external.json",
                "sha256": external_sha,
            }
        }
    }
    campaign_path = root / "configs" / "stage2_2khz_campaign.yaml"
    campaign_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_path.write_text(yaml.safe_dump(campaign, sort_keys=True), encoding="utf-8")
    _write_json(
        root / "authority" / "stage2_2khz_pretrain.json",
        {
            "schema": "stage2_2khz_pretrain_git_authority_v1",
            "artifact_source_commit_sha": source_commit,
        },
    )
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "review authority")
    head = _git(root, "rev-parse", "HEAD")
    _git(root, "update-ref", "refs/remotes/origin/dev", head)
    return campaign, source_commit, head


def test_clean_tracked_authority_has_no_current_head_self_reference_cycle(
    tmp_path: Path,
) -> None:
    campaign, source_commit, head = _clean_authority_repository(tmp_path)
    payload, _, observed_head = verify_tracked_head_authority(
        tmp_path, "authority/stage2_2khz_pretrain.json"
    )
    assert observed_head == head
    assert payload["artifact_source_commit_sha"] == source_commit
    verify_source_commit_ancestor(tmp_path, source_commit, head=head)
    assert _preverify_campaign_checkout(tmp_path, campaign) == head


def test_forged_in_memory_campaign_and_dirty_anchor_are_rejected(tmp_path: Path) -> None:
    campaign, _, _ = _clean_authority_repository(tmp_path)
    forged = json.loads(json.dumps(campaign))
    forged["external_contracts"]["canonical_pretrain"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="in-memory.*tracked HEAD bytes"):
        _preverify_campaign_checkout(tmp_path, forged)

    (tmp_path / "authority" / "stage2_2khz_pretrain.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="clean하지"):
        verify_tracked_head_authority(
            tmp_path, "authority/stage2_2khz_pretrain.json"
        )
