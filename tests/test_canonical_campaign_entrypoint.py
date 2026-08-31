"""Canonical Elice one-step campaign entrypoint의 fail-closed 경계."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
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


@pytest.mark.parametrize(
    "module_name", ("deep_anc.config", "deep_anc.data.transfer_contract")
)
def test_project_import_boundary_reorders_current_source_and_rejects_foreign_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_name: str
):
    monkeypatch.setattr(campaign, "_PROJECT_IMPORT_BOUNDARY_INITIALIZED", True)
    stale = tmp_path / "stale-src"
    stale.mkdir()
    monkeypatch.syspath_prepend(str(stale))

    source = campaign._pin_project_import_boundary(REPO_ROOT)  # noqa: SLF001

    assert Path(sys.path[0]) == source
    foreign_path = stale.joinpath(*module_name.split(".")).with_suffix(".py")
    foreign_path.parent.mkdir(parents=True)
    foreign_path.write_text("# malicious cached module\n", encoding="utf-8")
    foreign = types.ModuleType(module_name)
    foreign.__file__ = str(foreign_path)
    monkeypatch.setitem(sys.modules, module_name, foreign)
    with pytest.raises(campaign.CampaignError, match="foreign cached project module"):
        campaign._pin_project_import_boundary(REPO_ROOT)  # noqa: SLF001


def test_project_import_boundary_rejects_cached_module_that_spoofs_repo_file(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(campaign, "_PROJECT_IMPORT_BOUNDARY_INITIALIZED", True)
    spoof = types.ModuleType("deep_anc.config")
    spoof.__file__ = str(REPO_ROOT / "src/deep_anc/config.py")
    # A pathname string is not import authority.  In particular, a preloaded
    # ModuleType must not pass merely by copying the trusted source path.
    spoof.__spec__ = None
    spoof.__loader__ = None
    monkeypatch.setitem(sys.modules, "deep_anc.config", spoof)

    with pytest.raises(campaign.CampaignError, match="loader/spec authority"):
        campaign._pin_project_import_boundary(REPO_ROOT)  # noqa: SLF001


def test_first_import_boundary_rejects_same_path_custom_loader_spoof(
    monkeypatch: pytest.MonkeyPatch,
):
    trusted_path = REPO_ROOT / "src/deep_anc/config.py"

    class SamePathSpoofLoader:
        def get_filename(self, _fullname: str) -> str:
            return str(trusted_path)

        def get_data(self, _filename: str) -> bytes:
            return trusted_path.read_bytes()

    loader = SamePathSpoofLoader()
    spoof = types.ModuleType("deep_anc.config")
    spoof.__file__ = str(trusted_path)
    spoof.__loader__ = loader
    spoof.__spec__ = SimpleNamespace(origin=str(trusted_path), loader=loader)
    spoof.preexecuted_untrusted_payload = True
    monkeypatch.setattr(campaign, "_PROJECT_IMPORT_BOUNDARY_INITIALIZED", False)
    monkeypatch.setitem(sys.modules, "deep_anc.config", spoof)

    with pytest.raises(campaign.CampaignError, match="preloaded deep_anc module"):
        campaign._pin_project_import_boundary(REPO_ROOT)  # noqa: SLF001


def test_imported_main_rejects_nonisolated_interpreter(capsys: pytest.CaptureFixture[str]):
    assert not sys.flags.isolated

    assert campaign.main(["--help"]) == 2
    assert "-I -B" in capsys.readouterr().err


def test_campaign_cli_requires_isolated_no_bytecode_interpreter():
    rejected = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert "-I -B" in rejected.stderr

    accepted = subprocess.run(
        [sys.executable, "-I", "-B", str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0


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
    (repo / "scripts/elice/public_archive_cache.py").write_text(
        "# fixture archive cache entry\n", encoding="utf-8"
    )
    (repo / "scripts/elice/pget.py").write_text(
        "# fixture pget entry\n", encoding="utf-8"
    )
    (repo / ".gitignore").write_text(
        "/results/\n/data/raw/\n/.venv/\n/data/manifests/elice_bootstrap_receipt.json\n",
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
        "schema_version": 3,
        "expected_commit": commit,
        "expected_holdout_sha256": holdout,
        "expected_transfer_manifest_sha256": transfer,
        "campaign": {"seed": 20260803, "second_seed": None},
        "bootstrap": {
            "raw_hash_workers": 8,
            "cublas_workspace_config": ":4096:8",
            "archive_cache": None,
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


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _write_non_null_cache_chain(
    repo: Path, commit: str, external_root: Path
) -> tuple[dict[str, object], object]:
    archive_ids = (
        "dns_noise_000",
        "dns_noise_001",
        "dns_speech_000",
        "demand_dkitchen",
        "demand_dwashing",
        "demand_ooffice",
        "demand_ohallway",
        "demand_tmetro",
        "demand_tcar",
        "mimii_fan",
    )
    archive_origin = {
        "dns_noise_000": (5_364_611_964, "data/raw/noise/shard000.tar.bz2"),
        "dns_noise_001": (5_357_916_291, "data/raw/noise/shard001.tar.bz2"),
        "dns_speech_000": (4_664_045_287, "data/raw/noise/speech000.tar.bz2"),
        "demand_dkitchen": (336_992_458, "data/raw/noise/demand/DKITCHEN_48k.zip"),
        "demand_dwashing": (306_101_499, "data/raw/noise/demand/DWASHING_48k.zip"),
        "demand_ooffice": (277_643_831, "data/raw/noise/demand/OOFFICE_48k.zip"),
        "demand_ohallway": (252_905_617, "data/raw/noise/demand/OHALLWAY_48k.zip"),
        "demand_tmetro": (367_513_573, "data/raw/noise/demand/TMETRO_48k.zip"),
        "demand_tcar": (373_520_251, "data/raw/noise/demand/TCAR_48k.zip"),
        "mimii_fan": (928_511_244, "data/raw/noise/mimii_fan.zip"),
    }
    archive_output = {
        "dns_noise_000": ("data/raw/noise/dns_fullband/noise000", 8_000, None),
        "dns_noise_001": ("data/raw/noise/dns_fullband/noise001", 8_000, None),
        "dns_speech_000": ("data/raw/noise/speech/speech000", 8_065, 8_000_834_860),
        "demand_dkitchen": ("data/raw/noise/demand/DKITCHEN", 16, 460_806_848),
        "demand_dwashing": ("data/raw/noise/demand/DWASHING", 16, 460_806_848),
        "demand_ooffice": ("data/raw/noise/demand/OOFFICE", 16, 460_806_848),
        "demand_ohallway": ("data/raw/noise/demand/OHALLWAY", 16, 460_806_848),
        "demand_tmetro": ("data/raw/noise/demand/TMETRO", 16, 460_806_848),
        "demand_tcar": ("data/raw/noise/demand/TCAR", 16, 460_806_848),
        "mimii_fan": ("data/raw/noise/machine/fan", 3_600, 1_152_158_400),
    }
    rows: list[dict[str, object]] = []
    for archive_id in archive_ids:
        prefix, count, declared_bytes = archive_output[archive_id]
        total = count if declared_bytes is None else declared_bytes
        base_size, remainder = divmod(total, count)
        for index in range(count):
            size = base_size + (1 if index < remainder else 0)
            relative = f"{prefix}/{index:05d}.wav"
            rows.append(
                {
                    "archive_id": archive_id,
                    "path": relative,
                    "sha256": hashlib.sha256(
                        f"{archive_id}:{index}:{size}".encode()
                    ).hexdigest(),
                    "size": size,
                }
            )
    rows.sort(key=lambda row: str(row["path"]))
    output_bytes = sum(int(row["size"]) for row in rows)
    content_projection = hashlib.sha256()
    path_size_projection = hashlib.sha256()
    for row in rows:
        content_projection.update(
            _canonical(
                {
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "size": row["size"],
                }
            )
        )
        path_size_projection.update(
            _canonical({"path": row["path"], "size": row["size"]})
        )
    script_sha = _sha(repo / "scripts/elice/public_archive_cache.py")
    pget_sha = _sha(repo / "scripts/elice/pget.py")
    archive_digests = {
        archive_id: hashlib.sha256(f"archive:{archive_id}".encode()).hexdigest()
        for archive_id in archive_ids
    }
    output_digests: dict[str, str] = {}
    for archive_id in archive_ids:
        digest = hashlib.sha256()
        for row in rows:
            if row["archive_id"] == archive_id:
                digest.update(
                    _canonical(
                        {
                            "path": row["path"],
                            "sha256": row["sha256"],
                            "size": row["size"],
                        }
                    )
                )
        output_digests[archive_id] = digest.hexdigest()
    entries: list[dict[str, object]] = []
    for archive_id in archive_ids:
        size, canonical_target = archive_origin[archive_id]
        filename = f"fixture-{archive_id}.archive"
        archive_rows = [row for row in rows if row["archive_id"] == archive_id]
        archive_bytes = sum(int(row["size"]) for row in archive_rows)
        archive_sha = archive_digests[archive_id]
        entries.append(
            {
                "archive_format": "fixture",
                "archive_id": archive_id,
                "archive_sha256": archive_sha,
                "archive_size": size,
                "cache_path": (
                    f"archives/v1/{archive_id}/bytes_{size}/"
                    f"sha256_{archive_sha}/{filename}"
                ),
                "canonical_target": canonical_target,
                "corpus": "fixture",
                "filename": filename,
                "member_inventory_sha256": hashlib.sha256(
                    f"members:{archive_id}".encode()
                ).hexdigest(),
                "member_content_inventory_sha256": hashlib.sha256(
                    f"member-content:{archive_id}".encode()
                ).hexdigest(),
                "member_prefix": "fixture/",
                "provider_checksum": None,
                "provider_checksum_kind": "none",
                "provider_etag": None,
                "regular_file_bytes": archive_bytes,
                "regular_file_count": len(archive_rows),
                "source_url": f"https://example.invalid/{archive_id}",
                "output_content_inventory_sha256": output_digests[archive_id],
                "wav_bytes": archive_bytes,
                "wav_count": len(archive_rows),
            }
        )
    external_root.mkdir(parents=True)
    manifest_path = external_root / "manifest.json"
    manifest_path.write_bytes(
        _canonical(
            {
                "archive_count": 10,
                "archives": entries,
                "authority": "transport_acceleration_only_not_raw_or_training_authority",
                "excluded_corpora": [
                    "esc50",
                    "fma_small",
                    "fma_metadata",
                    "librispeech",
                ],
                "kind": "deep_anc_public_archive_cache",
                "publisher_commit": commit,
                "publisher_entry_script_sha256": script_sha,
                "publisher_pget_sha256": pget_sha,
                "schema_version": 1,
            }
        )
    )
    manifest_sha = _sha(manifest_path)
    stem = f"{manifest_sha}.{commit}"
    prefix = "data/raw/noise/.archive_cache_consumptions"
    intent_path = f"{prefix}/consume_intent.{stem}.json"
    inventory_path = f"{prefix}/consume_inventory.{stem}.json"
    completion_path = f"{prefix}/consume_complete.{stem}.json"
    origin_path = (
        "data/raw/noise/.archive_cache_origins/"
        f"archive_cache_origin.{stem}.json"
    )
    intent = {
        "archive_count": 10,
        "archive_manifest_sha256": manifest_sha,
        "authority": "cache_transport_state_only_requires_exact_raw_and_decoder_authority",
        "expected_output_bytes": output_bytes,
        "expected_output_count": len(rows),
        "expected_output_path_size_inventory_sha256": path_size_projection.hexdigest(),
        "kind": "deep_anc_archive_cache_consumption_intent",
        "publisher_commit": commit,
        "restorer_entry_script_sha256": script_sha,
        "restorer_pget_sha256": pget_sha,
        "schema_version": 1,
        "state": "in_progress_or_completed_requires_matching_external_anchors",
    }
    inventory = {
        "archive_manifest_sha256": manifest_sha,
        "authority": "cache_transport_state_only_requires_exact_raw_and_decoder_authority",
        "kind": "deep_anc_archive_cache_consumed_member_inventory",
        "output_bytes": output_bytes,
        "output_count": len(rows),
        "publisher_commit": commit,
        "rows": rows,
        "schema_version": 1,
    }
    origin = {
        "archives": [
            {
                "archive_id": archive_id,
                "archive_sha256": archive_digests[archive_id],
                "archive_size": archive_origin[archive_id][0],
                "canonical_target": archive_origin[archive_id][1],
            }
            for archive_id in archive_ids
        ],
        "authority": "cache_origin_only_not_official_raw_or_training_authority",
        "kind": "deep_anc_archive_cache_origin_receipt",
        "manifest_sha256": manifest_sha,
        "publisher_commit": commit,
        "restorer_entry_script_sha256": script_sha,
        "restorer_pget_sha256": pget_sha,
        "schema_version": 1,
    }
    for relative, value in (
        (intent_path, intent),
        (inventory_path, inventory),
        (origin_path, origin),
    ):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_canonical(value))
    completion = {
        "archive_manifest_sha256": manifest_sha,
        "authority": "cache_transport_state_only_requires_exact_raw_and_decoder_authority",
        "intent_path": intent_path,
        "intent_sha256": _sha(repo / intent_path),
        "kind": "deep_anc_archive_cache_consumption_completion",
        "member_inventory_path": inventory_path,
        "member_inventory_sha256": _sha(repo / inventory_path),
        "origin_receipt_path": origin_path,
        "origin_receipt_sha256": _sha(repo / origin_path),
        "output_bytes": output_bytes,
        "output_count": len(rows),
        "output_path_size_sha256_inventory_sha256": content_projection.hexdigest(),
        "publisher_commit": commit,
        "schema_version": 1,
        "state": "held_fd_consume_complete_pending_exact_raw_and_decoder_authority",
    }
    (repo / completion_path).write_bytes(_canonical(completion))
    decoder = {
        "inventory": [
            {
                "content_sha256": row["sha256"],
                "content_size": row["size"],
                "relative_path": row["path"],
            }
            for row in rows
        ],
        "schema_version": 1,
        "status": "complete",
    }
    decoder["audit_sha256"] = hashlib.sha256(
        json.dumps(
            decoder,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    decoder_path = repo / campaign.DECODER_AUDIT_REPORT
    decoder_path.parent.mkdir(parents=True, exist_ok=True)
    decoder_path.write_bytes(_canonical(decoder))
    binding = {
        "archive_manifest_sha256": manifest_sha,
        "authority": "cache_transport_state_bound_to_exact_decoder_inventory",
        "completion_path": completion_path,
        "completion_sha256": _sha(repo / completion_path),
        "member_inventory_path": inventory_path,
        "member_inventory_sha256": _sha(repo / inventory_path),
        "output_path_size_sha256_inventory_sha256": content_projection.hexdigest(),
        "decoder_audit_path": campaign.DECODER_AUDIT_REPORT,
        "decoder_audit_file_sha256": _sha(decoder_path),
        "decoder_audit_semantic_sha256": decoder["audit_sha256"],
        "decoder_cache_projection_sha256": content_projection.hexdigest(),
    }
    return binding, campaign.ArchiveCacheContract(
        root=external_root.absolute(),
        manifest=manifest_path.absolute(),
        expected_manifest_sha256=manifest_sha,
    )


def test_bootstrap_receipt_schema_v3_distinguishes_no_cache_marker_residue(
    tmp_path: Path,
):
    repo, commit, holdout, transfer = _repo(tmp_path)
    receipt = _write_bootstrap_receipt(
        repo, commit=commit, holdout_sha=holdout, transfer_sha=transfer
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["schema_version"] = 3
    payload["archive_cache_consumption"] = None
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    loaded = campaign._load_bootstrap_receipt(  # noqa: SLF001
        SimpleNamespace(
            expected_commit=commit,
            expected_holdout_sha256=holdout,
        ),
        repo,
    )
    assert loaded is not None and loaded[1]["schema_version"] == 3

    (repo / "data/raw/noise/.archive_cache_consumptions").mkdir(parents=True)
    with pytest.raises(campaign.CampaignError, match="no-cache bootstrap receipt"):
        campaign._load_bootstrap_receipt(  # noqa: SLF001
            SimpleNamespace(
                expected_commit=commit,
                expected_holdout_sha256=holdout,
            ),
            repo,
        )


def test_bootstrap_receipt_schema_v3_non_null_cache_chain_enters_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, commit, holdout, transfer = _repo(tmp_path)
    receipt = _write_bootstrap_receipt(
        repo, commit=commit, holdout_sha=holdout, transfer_sha=transfer
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["schema_version"] = 3
    binding, cache_contract = _write_non_null_cache_chain(
        repo, commit, tmp_path / "external-cache"
    )
    payload["archive_cache_consumption"] = binding
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    loaded = campaign._load_bootstrap_receipt(  # noqa: SLF001
        SimpleNamespace(
            expected_commit=commit,
            expected_holdout_sha256=holdout,
            archive_cache=cache_contract,
            decoder_audit=None,
        ),
        repo,
    )
    assert loaded is not None
    assert loaded[1]["archive_cache_consumption"]["completion_path"].endswith(
        f".{commit}.json"
    )

    contract_payload = _contract_payload(commit, holdout, transfer)
    contract_payload["bootstrap"]["archive_cache"] = {
        "root": str(cache_contract.root),
        "manifest": str(cache_contract.manifest),
        "expected_manifest_sha256": cache_contract.expected_manifest_sha256,
    }
    contract_payload["bootstrap"]["decoder_audit"] = {
        "expected_audit_sha256": payload["archive_cache_consumption"][
            "decoder_audit_semantic_sha256"
        ],
        "expected_file_sha256": payload["archive_cache_consumption"][
            "decoder_audit_file_sha256"
        ],
    }
    contract_path = tmp_path / "cache-campaign.json"
    contract_sha = _write_contract(contract_path, contract_payload)
    contract = campaign.load_contract(contract_path, contract_sha, repo_root=repo)

    def load_train_config(_path, overrides):
        digest = hashlib.sha256("\0".join(overrides).encode()).hexdigest()[:12]
        return {
            "ckpt_dir": f"runs/fixture-{digest}",
            "data": {"recorded_manifest": "data/manifests/recorded_regrouped_101.jsonl"},
        }

    monkeypatch.setattr(
        campaign,
        "_lazy_imports",
        lambda _repo: {
            "load_train_config": load_train_config,
            "canonical_recorded_manifest_for_data": lambda data: data[
                "recorded_manifest"
            ],
            "autostart_state_dir": lambda run_dir: run_dir / ".autostart",
        },
    )
    monkeypatch.setattr(campaign, "REPO_ROOT", repo)
    fixture_python = repo / ".venv/bin/python"
    fixture_python.parent.mkdir(parents=True, exist_ok=True)
    fixture_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fixture_python.chmod(0o755)
    inspection = campaign.inspect_campaign(contract, repo_root=repo)
    assert inspection.status == "READY_TO_EXECUTE", inspection.blockers
    assert inspection.phase == "pre_g0_readiness"
    assert inspection.next_action == "pre_g0_readiness"


def test_campaign_rejects_resealed_extra_cache_root_decoder_wav(
    tmp_path: Path,
):
    repo, commit, holdout, transfer = _repo(tmp_path)
    receipt = _write_bootstrap_receipt(
        repo, commit=commit, holdout_sha=holdout, transfer_sha=transfer
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["schema_version"] = 3
    binding, cache_contract = _write_non_null_cache_chain(
        repo, commit, tmp_path / "external-cache-extra-decoder"
    )
    decoder_path = repo / campaign.DECODER_AUDIT_REPORT
    decoder = json.loads(decoder_path.read_text(encoding="utf-8"))
    decoder["inventory"].append(
        {
            "content_sha256": "f" * 64,
            "content_size": 17,
            "relative_path": "data/raw/noise/demand/DKITCHEN/unexpected.wav",
        }
    )
    decoder["audit_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in decoder.items() if key != "audit_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    decoder_path.write_bytes(_canonical(decoder))
    binding["decoder_audit_file_sha256"] = _sha(decoder_path)
    binding["decoder_audit_semantic_sha256"] = decoder["audit_sha256"]
    payload["archive_cache_consumption"] = binding
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(campaign.CampaignError, match="cache raw exact-set"):
        campaign._load_bootstrap_receipt(  # noqa: SLF001
            SimpleNamespace(
                expected_commit=commit,
                expected_holdout_sha256=holdout,
                archive_cache=cache_contract,
                decoder_audit=None,
            ),
            repo,
        )

    decoder = repo / campaign.DECODER_AUDIT_REPORT
    decoder.write_bytes(decoder.read_bytes().replace(b'"status":"complete"', b'"status":"tampered"'))
    with pytest.raises(campaign.CampaignError, match="decoder audit SHA"):
        campaign._load_bootstrap_receipt(  # noqa: SLF001
            SimpleNamespace(
                expected_commit=commit,
                expected_holdout_sha256=holdout,
                archive_cache=cache_contract,
                decoder_audit=None,
            ),
            repo,
        )


def test_schema_v3_rejects_self_consistent_cache_chain_not_in_trusted_manifest(
    tmp_path: Path,
):
    repo, commit, holdout, transfer = _repo(tmp_path)
    receipt = _write_bootstrap_receipt(
        repo, commit=commit, holdout_sha=holdout, transfer_sha=transfer
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["schema_version"] = 3
    binding, cache_contract = _write_non_null_cache_chain(
        repo, commit, tmp_path / "external-cache"
    )

    inventory_path = repo / str(binding["member_inventory_path"])
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    changed_path = inventory["rows"][0]["path"]
    inventory["rows"][0]["sha256"] = "f" * 64
    inventory_path.write_bytes(_canonical(inventory))
    projection = hashlib.sha256()
    for row in inventory["rows"]:
        projection.update(
            _canonical(
                {"path": row["path"], "sha256": row["sha256"], "size": row["size"]}
            )
        )

    completion_path = repo / str(binding["completion_path"])
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["member_inventory_sha256"] = _sha(inventory_path)
    completion["output_path_size_sha256_inventory_sha256"] = projection.hexdigest()
    completion_path.write_bytes(_canonical(completion))

    decoder_path = repo / campaign.DECODER_AUDIT_REPORT
    decoder = json.loads(decoder_path.read_text(encoding="utf-8"))
    for row in decoder["inventory"]:
        if row["relative_path"] == changed_path:
            row["content_sha256"] = "f" * 64
            break
    decoder.pop("audit_sha256")
    decoder["audit_sha256"] = hashlib.sha256(
        json.dumps(
            decoder,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    decoder_path.write_bytes(_canonical(decoder))

    binding["completion_sha256"] = _sha(completion_path)
    binding["member_inventory_sha256"] = _sha(inventory_path)
    binding["output_path_size_sha256_inventory_sha256"] = projection.hexdigest()
    binding["decoder_cache_projection_sha256"] = projection.hexdigest()
    binding["decoder_audit_file_sha256"] = _sha(decoder_path)
    binding["decoder_audit_semantic_sha256"] = decoder["audit_sha256"]
    payload["archive_cache_consumption"] = binding
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(campaign.CampaignError, match="external archive manifest output projection"):
        campaign._load_bootstrap_receipt(  # noqa: SLF001
            SimpleNamespace(
                expected_commit=commit,
                expected_holdout_sha256=holdout,
                archive_cache=cache_contract,
                decoder_audit=None,
            ),
            repo,
        )


def test_existing_receipt_must_match_campaign_decoder_external_sha(tmp_path: Path):
    repo, commit, holdout, transfer = _repo(tmp_path)
    receipt = _write_bootstrap_receipt(
        repo, commit=commit, holdout_sha=holdout, transfer_sha=transfer
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["schema_version"] = 3
    payload["archive_cache_consumption"] = None
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    decoder_path = repo / campaign.DECODER_AUDIT_REPORT
    decoder_path.parent.mkdir(parents=True)
    decoder_path.write_bytes(_canonical({"audit_sha256": "a" * 64, "inventory": []}))

    with pytest.raises(campaign.CampaignError, match="campaign external SHA anchor"):
        campaign._load_bootstrap_receipt(  # noqa: SLF001
            SimpleNamespace(
                expected_commit=commit,
                expected_holdout_sha256=holdout,
                archive_cache=None,
                decoder_audit={
                    "expected_audit_sha256": "b" * 64,
                    "expected_file_sha256": _sha(decoder_path),
                },
            ),
            repo,
        )


def test_schema_v3_local_validator_rejects_nan_intent_despite_foreign_cached_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, commit, holdout, transfer = _repo(tmp_path)
    receipt = _write_bootstrap_receipt(
        repo, commit=commit, holdout_sha=holdout, transfer_sha=transfer
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["schema_version"] = 3
    binding, cache_contract = _write_non_null_cache_chain(
        repo, commit, tmp_path / "external-cache"
    )
    payload["archive_cache_consumption"] = binding
    completion_path = repo / str(binding["completion_path"])
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    intent_path = repo / str(completion["intent_path"])
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    intent["expected_output_bytes"] = float("nan")
    intent_path.write_text(
        json.dumps(intent, sort_keys=True, separators=(",", ":"), allow_nan=True)
        + "\n",
        encoding="utf-8",
    )
    completion["intent_sha256"] = _sha(intent_path)
    completion_path.write_bytes(_canonical(completion))
    binding["completion_sha256"] = _sha(completion_path)
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    foreign = types.ModuleType("deep_anc.data.transfer_contract")
    foreign.__file__ = str(tmp_path / "foreign/transfer_contract.py")
    foreign.TransferContractError = ValueError
    foreign._validate_archive_cache_bootstrap_binding = lambda *_args: binding
    monkeypatch.setitem(sys.modules, "deep_anc.data.transfer_contract", foreign)

    with pytest.raises(campaign.CampaignError, match="non-finite JSON number"):
        campaign._load_bootstrap_receipt(  # noqa: SLF001
            SimpleNamespace(
                expected_commit=commit,
                expected_holdout_sha256=holdout,
                archive_cache=cache_contract,
                decoder_audit=None,
            ),
            repo,
        )


def test_schema_v3_local_validator_rejects_duplicate_completion_key(
    tmp_path: Path,
):
    repo, commit, holdout, transfer = _repo(tmp_path)
    receipt = _write_bootstrap_receipt(
        repo, commit=commit, holdout_sha=holdout, transfer_sha=transfer
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["schema_version"] = 3
    binding, cache_contract = _write_non_null_cache_chain(
        repo, commit, tmp_path / "external-cache"
    )
    payload["archive_cache_consumption"] = binding
    completion_path = repo / str(binding["completion_path"])
    raw = completion_path.read_bytes()
    completion_path.write_bytes(b'{"authority":"shadow",' + raw[1:])
    binding["completion_sha256"] = _sha(completion_path)
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(campaign.CampaignError, match="duplicate JSON key"):
        campaign._load_bootstrap_receipt(  # noqa: SLF001
            SimpleNamespace(
                expected_commit=commit,
                expected_holdout_sha256=holdout,
                archive_cache=cache_contract,
                decoder_audit=None,
            ),
            repo,
        )


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
    contract = SimpleNamespace(expected_commit="c" * 40, archive_cache=None)
    g0 = campaign.build_g0_command(contract, candidate, "d" * 64, repo)
    pilot_overrides = campaign._candidate_overrides(
        contract, candidate, "d" * 64, role="loss_pilot"
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
        contract,
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
        contract=SimpleNamespace(archive_cache=None),
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
    result = campaign._main_impl(  # noqa: SLF001
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

    result = campaign._main_impl(  # noqa: SLF001
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

    result = campaign._main_impl(  # noqa: SLF001
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

    result = campaign._main_impl(  # noqa: SLF001
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
