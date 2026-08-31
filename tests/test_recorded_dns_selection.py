"""외부 DNS speech selection receipt의 fail-closed 회귀."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import py_compile
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from deep_anc.data import public_lineage
from deep_anc.data import recorded_dns_selection as selection
from deep_anc.data.source_trust import (
    SourceTrustError,
    exact_clean_source_evidence,
)
from deep_anc.realtime.noise_gen import NoiseProgram, render_recording_file_window


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_selector_script():
    path = REPO_ROOT / "scripts/data/select_recorded_dns_speech.py"
    spec = importlib.util.spec_from_file_location("_dns_selector_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_source_plan_builder():
    path = REPO_ROOT / "scripts/data/build_recorded_additions_plan.py"
    spec = importlib.util.spec_from_file_location("_dns_source_plan_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _init_source_trust_repo(root: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".gitignore").write_text(
        "/data/\n/results/\n/runs/\n/.venv/\n__pycache__/\n*.pyc\n"
    )
    script = root / "scripts/selector.py"
    script.parent.mkdir(parents=True)
    script.write_text("VALUE = 1\n")
    subprocess.run(
        ["git", "add", ".gitignore", "scripts/selector.py"], cwd=root, check=True
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
            "source fixture",
        ],
        cwd=root,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_receipt_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    commit = "a" * 40
    monkeypatch.setattr(selection, "_git_head", lambda _root: commit)
    clean_source = {
        "schema": "exact_clean_git_source/v1",
        "commit": commit,
        "head_tree_object_id": "b" * 40,
        "git_object_format": "sha1",
        "tracked_file_count": 1,
        "tracked_inventory_sha256": "c" * 64,
        "policy": {
            "tracked_worktree": "exact_HEAD_blob_and_mode",
            "index": "exact_HEAD_tree_no_hidden_flags",
            "nonignored_untracked": "forbidden",
            "protected_ignored_roots": ["src", "scripts", "configs"],
            "protected_runtime_bytecode": "forbidden",
            "ignored_artifacts_outside_protected_roots": "allowed",
            "replace_refs_and_grafts": "forbidden",
        },
    }
    trusted_clean_source = json.loads(json.dumps(clean_source))
    monkeypatch.setattr(
        selection,
        "exact_clean_source_evidence",
        lambda _root, **_kwargs: json.loads(json.dumps(trusted_clean_source)),
    )
    parent = {
        "holdout": {
            "path": "data/manifests/recorded_holdout.json",
            "sha256": "1" * 64,
            "size": 1,
        },
        "speech_lineage_keys": [
            "conservative_speech_book_numeric:999999",
            "conservative_speech_reader_numeric:999999",
        ],
        "speech_lineage_keys_sha256": selection._canonical_json_sha256(
            [
                "conservative_speech_book_numeric:999999",
                "conservative_speech_reader_numeric:999999",
            ]
        ),
        "numeric_aliases": [
            "conservative_speech_book_numeric:999999",
            "conservative_speech_reader_numeric:999999",
        ],
        "numeric_aliases_sha256": selection._canonical_json_sha256(
            [
                "conservative_speech_book_numeric:999999",
                "conservative_speech_reader_numeric:999999",
            ]
        ),
    }
    monkeypatch.setattr(selection, "_parent_speech_authority", lambda _root: parent)

    strict_path = tmp_path / "assets/measured/strict-primary.npz"
    strict_path.parent.mkdir(parents=True)
    np.savez(
        strict_path,
        fir=np.asarray([1.0], dtype=np.float32),
        sample_rate=np.asarray(48_000),
        delay_samples=np.asarray(100),
        consistency_band_hz=np.asarray([150.0, 1600.0]),
    )
    freeze_path = tmp_path / ".venv/environment-freeze.txt"
    freeze_path.parent.mkdir(parents=True)
    freeze_path.write_text(
        "cffi==1.0\n"
        "-e git+https://github.com/Roka-jsj/Deep-ANC.git@"
        f"{commit}#egg=deep_anc\n"
        "numpy==1.26.4\nscipy==1.11.4\nsoundfile==0.14.0\n"
    )
    freeze_sha = _sha(freeze_path.read_bytes())
    bootstrap_path = tmp_path / "data/manifests/elice_bootstrap_receipt.json"
    bootstrap_path.parent.mkdir(parents=True)
    bootstrap_path.write_text(
        json.dumps(
            {
                "expected_commit": commit,
                "environment": {
                    "freeze_receipt": ".venv/environment-freeze.txt",
                    "freeze_receipt_sha256": freeze_sha,
                    "torch_version": "2.5.1+cu121",
                    "torch_cuda": "12.1",
                },
            }
        )
        + "\n"
    )

    bundle = tmp_path / selection.DNS_SELECTION_BUNDLE_ROOT
    bootstrap_copy = bundle / "inputs/elice_bootstrap_receipt.selection-parent.json"
    bootstrap_copy.parent.mkdir(parents=True)
    bootstrap_copy.write_bytes(bootstrap_path.read_bytes())
    freeze_copy = bundle / "inputs/environment-freeze.selection-parent.txt"
    freeze_copy.write_bytes(freeze_path.read_bytes())
    manifest_path = bundle / "inputs/speech.selection-parent.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # public split은 recorded 배정과 독립이다. selector는 전체 후보를 먼저
    # ranking하고 recorded train2/val1/test2를 후단에서 고정 배정한다.
    splits = ("train", "train", "train", "train", "train")
    rows = []
    raw_payloads = []
    composite_payloads = []
    scans = []
    for index, split in enumerate(splits):
        filename = f"book_{100 + index}_chp_1_reader_{200 + index}.wav"
        keys = list(
            public_lineage.conservative_cross_corpus_speech_lineage_keys(
                public_lineage.dns_speech_lineage_keys(filename)
            )
        )
        source_sha = hashlib.sha256(f"source-{index}".encode()).hexdigest()
        group = "public-lineage-" + public_lineage.canonical_json_sha256(
            {"lineage_keys": keys, "content_sha256": [source_sha]}
        )
        row = {
            "path": f"/home/elicer/Deep_ANC/data/raw/noise/speech/{filename}",
            "duration_s": 20.0,
            "sample_rate": 48_000,
            "channels": 1,
            "tag": "speech",
            "split": split,
            "content_sha256": source_sha,
            "content_size": 1000 + index,
            "lineage_schema": public_lineage.PUBLIC_LINEAGE_SCHEMA,
            "lineage_keys": keys,
            "group_id": group,
        }
        rows.append(row)
        time = np.arange(selection.DNS_RAW_FRAMES, dtype=np.float64) / 48_000.0
        values = sum(
            (0.035 + index * 0.001)
            * np.sin(2.0 * np.pi * frequency * time + index * 0.17)
            for frequency in (220.0, 440.0, 800.0, 1250.0)
        )
        raw = selection._pcm16_wav_bytes(values)
        composite = selection.dns_composite_bytes_from_raw(raw)
        source_preflight = selection._rendered_source_preflight(composite)
        assert source_preflight["passed"] is True
        densities, covered = selection._band_density(
            selection._decode_source(raw, label="fixture raw"),
            np.asarray([1.0], dtype=np.float64),
        )
        assert covered == 4
        raw_payloads.append(raw)
        composite_payloads.append(composite)
        scans.append(
            {
                "manifest_index": index,
                "manifest_row_sha256": selection._canonical_json_sha256(row),
                "path": row["path"],
                "content_sha256": source_sha,
                "group_id": group,
                "public_source_split": split,
                "lineage_keys": keys,
                "status": "eligible",
                "reason": "",
                "coverage_scan": {
                    "density_ratios": densities,
                    "covered_subband_count": covered,
                },
                "source_preflight": source_preflight,
                "selected_window_start_frame": 0,
            }
        )
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_raw = manifest_path.read_bytes()
    manifest_sha = _sha(manifest_raw)
    lineage = public_lineage.validate_public_manifest_lineage({"speech": rows})

    selected = []
    for order, scan in enumerate(selection._select_results(scans)):
        index = int(scan["manifest_index"])
        group = str(scan["group_id"])
        suffix = group.removeprefix("public-lineage-")[:12]
        raw_relative = f"sources/speech-dns-{order + 1:02d}-{suffix}-raw.wav"
        composite_relative = (
            f"sources/speech-dns-{order + 1:02d}-{suffix}-repeat-trim.wav"
        )
        raw_path = bundle / raw_relative
        composite_path = bundle / composite_relative
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw_payloads[index])
        composite_path.write_bytes(composite_payloads[index])
        selected.append(
            {
                "order": order,
                "manifest_index": index,
                "manifest_row": rows[index],
                "manifest_row_sha256": selection._canonical_json_sha256(rows[index]),
                "public_group_id": group,
                "public_source_split": scan["public_source_split"],
                "recorded_split": selection.DNS_RECORDED_SPLIT_ASSIGNMENT[order],
                "lineage_keys": scan["lineage_keys"],
                "source_content_sha256": scan["content_sha256"],
                "source_window_start_frame": 0,
                "coverage_scan": scan["coverage_scan"],
                "source_preflight": scan["source_preflight"],
                "raw_output": {
                    "path": raw_path.relative_to(tmp_path).as_posix(),
                    "sha256": _sha(raw_payloads[index]),
                    "size": len(raw_payloads[index]),
                    "sample_rate": 48_000,
                    "channels": 1,
                    "frames": selection.DNS_RAW_FRAMES,
                    "subtype": "PCM_16",
                },
                "composite_output": {
                    "path": composite_path.relative_to(tmp_path).as_posix(),
                    "sha256": _sha(composite_payloads[index]),
                    "size": len(composite_payloads[index]),
                    "sample_rate": 48_000,
                    "channels": 1,
                    "frames": selection.DNS_COMPOSITE_FRAMES,
                    "subtype": "PCM_16",
                    "transform": selection.DNS_TRANSFORM,
                    "repeat_count": selection.DNS_REPEAT_COUNT,
                },
            }
        )
    manifest_ref = {
        "path": manifest_path.relative_to(tmp_path).as_posix(),
        "sha256": manifest_sha,
        "size": len(manifest_raw),
        "row_count": 5,
    }
    issue_root = Path("/home/elicer/Deep_ANC")
    venv_site = issue_root / ".venv/lib/python3.10/site-packages"
    module_versions = {
        "numpy": "1.26.4",
        "numpy.fft": None,
        "numpy.fft._pocketfft": None,
        "numpy.fft._pocketfft_internal": None,
        "numpy.core._multiarray_umath": None,
        "soundfile": "0.14.0",
        "_soundfile": None,
        "_cffi_backend": None,
        "scipy": "1.11.4",
        "scipy.signal": None,
    }
    selector_runtime = {
        "schema": selection.SELECTOR_RUNTIME_SCHEMA,
        "python_executable": str(issue_root / ".venv/bin/python"),
        "python_executable_realpath": "/usr/bin/python3.10",
        "python_executable_sha256": "7" * 64,
        "python_executable_size": 1024,
        "python_base_prefix": "/usr",
        "python_version": "3.10.12",
        "flags": {
            "isolated": 1,
            "ignore_environment": 1,
            "no_user_site": 1,
            "no_site": 1,
            "dont_write_bytecode": 1,
        },
        "pycache_prefix": "/dev/null/deep-anc-selector",
        "sys_path": [
            str(issue_root / "src"),
            "/usr/lib/python310.zip",
            "/usr/lib/python3.10",
            "/usr/lib/python3.10/lib-dynload",
            str(venv_site),
        ],
        "environment_freeze_sha256": freeze_sha,
        "modules": {
            name: {
                "name": name,
                "path": str(
                    venv_site
                    / (
                        name.replace(".", "/")
                        + (
                            ".cpython-310-x86_64-linux-gnu.so"
                            if name
                            in {
                                "numpy.fft._pocketfft_internal",
                                "numpy.core._multiarray_umath",
                                "_cffi_backend",
                            }
                            else ".py"
                        )
                    )
                ),
                "sha256": hashlib.sha256(name.encode()).hexdigest(),
                "size": len(name),
                "version": version,
                "loader": (
                    "ExtensionFileLoader"
                    if name
                    in {
                        "numpy.fft._pocketfft_internal",
                        "numpy.core._multiarray_umath",
                        "_cffi_backend",
                    }
                    else "SourceFileLoader"
                ),
                "origin_kind": (
                    "native_extension"
                    if name
                    in {
                        "numpy.fft._pocketfft_internal",
                        "numpy.core._multiarray_umath",
                        "_cffi_backend",
                    }
                    else "source"
                ),
                "cached_path": (
                    None
                    if name
                    in {
                        "numpy.fft._pocketfft_internal",
                        "numpy.core._multiarray_umath",
                        "_cffi_backend",
                    }
                    else "/dev/null/deep-anc-selector/"
                    + str(venv_site).lstrip("/")
                    + "/"
                    + name.replace(".", "/")
                    + ".cpython-310.pyc"
                ),
            }
            for name, version in module_versions.items()
        },
        "libsndfile": {
            "path": str(venv_site / "_soundfile_data/libsndfile_x86_64.so"),
            "sha256": "8" * 64,
            "size": 2048,
            "version": "1.2.2",
        },
        "scipy_policy": "provenance_recorded_never_called_by_dns_numpy_power2_fft",
    }
    payload = {
        "schema_version": selection.DNS_SELECTION_SCHEMA_VERSION,
        "kind": selection.DNS_SELECTION_KIND,
        "generation_id": selection.DNS_SELECTION_GENERATION_ID,
        "source_commit": commit,
        "clean_source": clean_source,
        "bootstrap_receipt_origin": {
            "path": bootstrap_path.relative_to(tmp_path).as_posix(),
            "sha256": _sha(bootstrap_path.read_bytes()),
            "size": bootstrap_path.stat().st_size,
        },
        "bootstrap_receipt": {
            "path": bootstrap_copy.relative_to(tmp_path).as_posix(),
            "sha256": _sha(bootstrap_copy.read_bytes()),
            "size": bootstrap_copy.stat().st_size,
        },
        "environment_freeze_origin": {
            "path": freeze_path.relative_to(tmp_path).as_posix(),
            "sha256": freeze_sha,
            "size": freeze_path.stat().st_size,
        },
        "environment_freeze": {
            "path": freeze_copy.relative_to(tmp_path).as_posix(),
            "sha256": freeze_sha,
            "size": freeze_copy.stat().st_size,
        },
        "selector_runtime": selector_runtime,
        "public_manifest_origin": {
            **manifest_ref,
            "path": "data/manifests/canonical_v4/speech.jsonl",
        },
        "public_manifest": manifest_ref,
        "public_lineage": {
            "schema": public_lineage.PUBLIC_LINEAGE_SCHEMA,
            "component_count": lineage["component_count"],
            "component_membership_sha256": lineage["component_membership_sha256"],
            "crosswalk_policy_sha256": selection._canonical_json_sha256(
                public_lineage.PUBLIC_CROSSWALK_POLICY
            ),
        },
        "parent82": parent,
        "strict_primary": {
            "path": strict_path.relative_to(tmp_path).as_posix(),
            "sha256": _sha(strict_path.read_bytes()),
            "size": strict_path.stat().st_size,
            "sample_rate": 48_000,
            "delay_samples": 100,
            "fir_taps": 1,
            "consistency_band_hz": [150.0, 1600.0],
        },
        "algorithm": selection.DNS_SCAN_ALGORITHM,
        "scan_results": scans,
        "scan_results_sha256": selection._canonical_json_sha256(scans),
        "selected": selected,
        "selected_sha256": selection._canonical_json_sha256(selected),
    }
    payload["evidence_sha256"] = selection._canonical_json_sha256(payload)
    receipt = bundle / "selection_receipt.json"
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return receipt, payload, parent


def _reseal(receipt: Path, payload: dict) -> None:
    payload["scan_results_sha256"] = selection._canonical_json_sha256(
        payload["scan_results"]
    )
    payload["selected_sha256"] = selection._canonical_json_sha256(payload["selected"])
    payload["evidence_sha256"] = selection._canonical_json_sha256(
        selection._without_evidence_sha(payload)
    )
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_missing_dns_receipt_is_explicitly_blocked(tmp_path):
    with pytest.raises(selection.DNSSelectionBlocked, match="BLOCKED"):
        selection.validate_dns_selection_receipt(
            repo_root=tmp_path, verify_current_commit=False
        )


def test_clean_source_allows_ignored_data_receipt(tmp_path):
    commit = _init_source_trust_repo(tmp_path)
    receipt = tmp_path / "data/manifests/selection_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n")
    evidence = exact_clean_source_evidence(tmp_path, expected_commit=commit)
    assert evidence["commit"] == commit
    assert evidence["tracked_file_count"] == 2


@pytest.mark.parametrize(
    "mutation",
    (
        "tracked_dirty",
        "staged_dirty",
        "untracked_script",
        "ignored_script",
        "ignored_native_extension",
    ),
)
def test_clean_source_rejects_worktree_index_and_code_injection(
    tmp_path, mutation
):
    commit = _init_source_trust_repo(tmp_path)
    tracked = tmp_path / "scripts/selector.py"
    if mutation == "tracked_dirty":
        tracked.write_text("VALUE = 2\n")
    elif mutation == "staged_dirty":
        tracked.write_text("VALUE = 2\n")
        subprocess.run(["git", "add", "scripts/selector.py"], cwd=tmp_path, check=True)
    elif mutation == "untracked_script":
        (tmp_path / "scripts/injected.py").write_text("raise SystemExit\n")
    elif mutation == "ignored_script":
        with (tmp_path / ".git/info/exclude").open("a", encoding="utf-8") as handle:
            handle.write("/scripts/injected.py\n")
        (tmp_path / "scripts/injected.py").write_text("raise SystemExit\n")
    else:
        with (tmp_path / ".git/info/exclude").open("a", encoding="utf-8") as handle:
            handle.write("/src/injected.so\n")
        injected = tmp_path / "src/injected.so"
        injected.parent.mkdir()
        injected.write_bytes(b"forged native module")
    with pytest.raises(SourceTrustError):
        exact_clean_source_evidence(tmp_path, expected_commit=commit)


def test_clean_source_ignores_ambient_git_context_and_checks_actual_root(
    tmp_path, monkeypatch
):
    target = tmp_path / "target"
    decoy = tmp_path / "decoy"
    target.mkdir()
    decoy.mkdir()
    commit = _init_source_trust_repo(target)
    _init_source_trust_repo(decoy)
    (target / "scripts/selector.py").write_text("VALUE = 99\n")
    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(decoy))
    monkeypatch.setenv("GIT_INDEX_FILE", str(decoy / ".git/index"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(decoy / ".git/objects"))
    subprocess.run(
        ["git", "config", "core.worktree", str(decoy)], cwd=target, check=True
    )
    with pytest.raises(SourceTrustError):
        exact_clean_source_evidence(target, expected_commit=commit)


def test_clean_source_runtime_cache_policy_is_split_for_validator_and_issuer(
    tmp_path,
):
    commit = _init_source_trust_repo(tmp_path)
    cache = tmp_path / "scripts/__pycache__/selector.cpython-310.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"runtime cache fixture")
    with (tmp_path / ".git/info/exclude").open("a", encoding="utf-8") as handle:
        handle.write("__pycache__/\n*.pyc\n")
    relaxed = exact_clean_source_evidence(tmp_path, expected_commit=commit)
    assert relaxed["policy"]["protected_runtime_bytecode"].startswith("allowed_only")
    with pytest.raises(SourceTrustError, match="ignored untracked injection"):
        exact_clean_source_evidence(
            tmp_path,
            expected_commit=commit,
            reject_runtime_bytecode=True,
        )


def test_preimport_quarantines_retained_cache_and_supports_restore_repeat(
    tmp_path,
):
    selector = _load_selector_script()
    commit = _init_source_trust_repo(tmp_path)
    cache = tmp_path / "scripts/__pycache__/selector.cpython-310.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"retained Elice cache bytes")

    before, directories, files = selector._preimport_exact_source(
        tmp_path, expected_commit=commit
    )
    assert [entry["source_path"] for entry in files] == [
        "scripts/__pycache__/selector.cpython-310.pyc"
    ]
    first = selector._quarantine_cache_directories(
        tmp_path,
        evidence=before,
        cache_directories=directories,
        cache_files=files,
    )
    transaction = Path(str(first["path"]))
    assert not cache.parent.exists()
    assert (
        transaction / "caches/scripts/__pycache__/selector.cpython-310.pyc"
    ).read_bytes() == b"retained Elice cache bytes"
    after, directories, files = selector._preimport_exact_source(
        tmp_path, expected_commit=commit
    )
    assert after == before
    assert directories == [] and files == []
    assert exact_clean_source_evidence(
        tmp_path, expected_commit=commit, reject_runtime_bytecode=True
    ) == after

    restored = selector._restore_cache_quarantine(
        tmp_path, expected_commit=commit, transaction=transaction
    )
    assert restored["status"] == "restored"
    assert cache.read_bytes() == b"retained Elice cache bytes"
    with pytest.raises(FileExistsError, match="overwrite"):
        selector._restore_cache_quarantine(
            tmp_path, expected_commit=commit, transaction=transaction
        )

    repeated_evidence, directories, files = selector._preimport_exact_source(
        tmp_path, expected_commit=commit
    )
    second = selector._quarantine_cache_directories(
        tmp_path,
        evidence=repeated_evidence,
        cache_directories=directories,
        cache_files=files,
    )
    assert second["status"] == "quarantined"
    assert second["transaction_id"] != first["transaction_id"]
    assert not cache.parent.exists()


def test_preimport_resumes_interrupted_quarantine_without_overwrite(tmp_path):
    selector = _load_selector_script()
    commit = _init_source_trust_repo(tmp_path)
    cache = tmp_path / "scripts/__pycache__/selector.cpython-310.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"resume cache")
    evidence, directories, files = selector._preimport_exact_source(
        tmp_path, expected_commit=commit
    )
    complete = selector._quarantine_cache_directories(
        tmp_path,
        evidence=evidence,
        cache_directories=directories,
        cache_files=files,
    )
    final = Path(str(complete["path"]))
    staging = final.parent / f".building-{complete['transaction_id']}"
    selector._rename_noreplace(final, staging)
    selector._fsync_directory(final.parent)

    resumed = selector._quarantine_cache_directories(
        tmp_path,
        evidence=evidence,
        cache_directories=[],
        cache_files=[],
    )
    assert resumed["status"] == "resumed"
    assert final.is_dir() and not staging.exists()
    with pytest.raises(FileExistsError):
        selector._rename_noreplace(final, final)


@pytest.mark.parametrize("kind", ("symlink", "fifo", "non_pyc"))
def test_preimport_rejects_malicious_or_special_cache_members(tmp_path, kind):
    selector = _load_selector_script()
    commit = _init_source_trust_repo(tmp_path)
    cache = tmp_path / "scripts/__pycache__"
    cache.mkdir()
    member = cache / (
        "selector.cpython-310.pyc" if kind != "non_pyc" else "injected.py"
    )
    if kind == "symlink":
        member.symlink_to("/etc/passwd")
    elif kind == "fifo":
        os.mkfifo(member)
    else:
        member.write_text("raise SystemExit\n")
    with pytest.raises(RuntimeError, match=r"regular(?: file| \.pyc)"):
        selector._preimport_exact_source(tmp_path, expected_commit=commit)


@pytest.mark.parametrize("flag", ("assume-unchanged", "skip-worktree"))
def test_preimport_rejects_hidden_index_flag_and_tracked_mutation(tmp_path, flag):
    selector = _load_selector_script()
    commit = _init_source_trust_repo(tmp_path)
    option = "--assume-unchanged" if flag == "assume-unchanged" else "--skip-worktree"
    subprocess.run(
        ["git", "update-index", option, "scripts/selector.py"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "scripts/selector.py").write_text("VALUE = 999\n")
    with pytest.raises(RuntimeError, match="flag"):
        selector._preimport_exact_source(tmp_path, expected_commit=commit)


def test_selector_rejects_nonisolated_invocation_before_repo_imports():
    result = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            str(REPO_ROOT / "scripts/data/select_recorded_dns_speech.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "-I -S -B" in result.stderr


def test_isolated_selector_ignores_ambient_pythonpath_before_scan(tmp_path):
    forged = tmp_path / "forged"
    forged.mkdir()
    marker = tmp_path / "ambient-imported"
    (forged / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n"
    )
    scipy = forged / "scipy"
    scipy.mkdir()
    (scipy / "__init__.py").write_text("raise RuntimeError('forged scipy imported')\n")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(forged)
    result = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            "-I",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=/dev/null/deep-anc-selector",
            str(REPO_ROOT / "scripts/data/select_recorded_dns_speech.py"),
            "--expected-commit",
            "0" * 40,
            "--help",
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    # 개발 worktree는 dirty라 early exact-source preflight에서 차단될 수 있다.
    # 중요한 점은 Python startup/sitecustomize/scipy injection이 실행되지 않는 것이다.
    assert result.returncode != 0
    assert not marker.exists()
    assert "forged scipy imported" not in (result.stdout + result.stderr)


@pytest.mark.parametrize("retained_cache", (False, True))
def test_actual_isolated_selector_cli_accepts_clean_checkout_and_retained_cache(
    tmp_path, retained_cache
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copytree(
        REPO_ROOT / "src/deep_anc",
        checkout / "src/deep_anc",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    selector_target = checkout / "scripts/data/select_recorded_dns_speech.py"
    selector_target.parent.mkdir(parents=True)
    shutil.copy2(
        REPO_ROOT / "scripts/data/select_recorded_dns_speech.py", selector_target
    )
    (checkout / ".gitignore").write_text(
        "/.venv/\n__pycache__/\n*.pyc\n/.deep_anc_source_cache_quarantine/\n"
    )
    (checkout / ".venv").symlink_to(REPO_ROOT / ".venv", target_is_directory=True)
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "clean selector checkout",
        ],
        cwd=checkout,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if retained_cache:
        cache = checkout / "src/deep_anc/data/__pycache__/retained.cpython-310.pyc"
        cache.parent.mkdir()
        cache.write_bytes(b"retained cache fixture")

    result = subprocess.run(
        [
            str(checkout / ".venv/bin/python"),
            "-I",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=/dev/null/deep-anc-selector",
            str(selector_target),
            "--expected-commit",
            commit,
            "--help",
        ],
        cwd=checkout,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--verify-existing" in result.stdout
    assert not list(checkout.rglob("*.pyc"))


def test_canonical_pycache_prefix_ignores_forged_adjacent_unchecked_pyc(tmp_path):
    module = tmp_path / "probe.py"
    module.write_text("VALUE = 'forged'\n")
    py_compile.compile(
        str(module),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    module.write_text("VALUE = 'source'\n")
    command = (
        "import sys;"
        f"sys.path.insert(0, {str(tmp_path)!r});"
        "import probe;print(probe.VALUE)"
    )
    adjacent = subprocess.run(
        [sys.executable, "-S", "-B", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    isolated = subprocess.run(
        [
            sys.executable,
            "-S",
            "-B",
            "-X",
            "pycache_prefix=/dev/null/deep-anc-selector",
            "-c",
            command,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert adjacent.stdout.strip() == "forged"
    assert isolated.stdout.strip() == "source"


def test_actual_isolated_runtime_binds_numpy_fft_and_libsndfile(tmp_path):
    import importlib.metadata

    freeze = tmp_path / "environment-freeze.txt"
    freeze.write_text(
        "\n".join(
            [
                f"numpy=={importlib.metadata.version('numpy')}",
                f"scipy=={importlib.metadata.version('scipy')}",
                f"soundfile=={importlib.metadata.version('soundfile')}",
            ]
        )
        + "\n"
    )
    freeze_sha = _sha(freeze.read_bytes())
    code = """
import json, os, sys, sysconfig
from pathlib import Path
root = Path(sys.argv[1])
version = f"python{sys.version_info.major}.{sys.version_info.minor}"
stdlib = Path(sysconfig.get_path("stdlib")).resolve()
candidates = [
    root / "src",
    stdlib.parent / f"python{sys.version_info.major}{sys.version_info.minor}.zip",
    stdlib,
    stdlib / "lib-dynload",
    root / ".venv/lib" / version / "site-packages",
    Path(sys.base_prefix) / "local/lib" / version / "dist-packages",
    Path(sys.base_prefix) / "lib" / version / "dist-packages",
    Path(sys.base_prefix) / "lib/python3/dist-packages",
]
sys.path[:] = [os.path.abspath(path) for index, path in enumerate(candidates) if index == 1 or path.is_dir()]
from deep_anc.data.source_trust import exact_selector_runtime_evidence
value = exact_selector_runtime_evidence(
    root, freeze_receipt=sys.argv[2], expected_freeze_sha256=sys.argv[3]
)
print(json.dumps(value, sort_keys=True))
"""
    result = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            "-I",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=/dev/null/deep-anc-selector",
            "-c",
            code,
            str(REPO_ROOT),
            str(freeze),
            freeze_sha,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    runtime = json.loads(result.stdout)
    assert runtime["pycache_prefix"] == "/dev/null/deep-anc-selector"
    assert runtime["libsndfile"]["version"]
    assert Path(runtime["libsndfile"]["path"]).name.startswith("libsndfile_")
    assert any(
        name.startswith("numpy.fft.")
        and reference["origin_kind"] == "native_extension"
        for name, reference in runtime["modules"].items()
    )
    assert all(
        reference["cached_path"] is None
        or reference["cached_path"].startswith(
            "/dev/null/deep-anc-selector/"
        )
        for reference in runtime["modules"].values()
    )


def test_dns_receipt_revalidates_manifest_lineage_raw_composite_and_strict_p(
    tmp_path, monkeypatch
):
    receipt, _payload, _parent = _write_receipt_fixture(tmp_path, monkeypatch)
    summary = selection.validate_dns_selection_receipt(
        repo_root=tmp_path,
        receipt_path=receipt.relative_to(tmp_path).as_posix(),
        expected_receipt_sha256=_sha(receipt.read_bytes()),
        require_source_files=True,
    )
    assert len(summary["selected_group_ids"]) == 5
    assert {item["public_source_split"] for item in summary["selected"]} == {"train"}
    assert [item["recorded_split"] for item in summary["selected"]] == list(
        selection.DNS_RECORDED_SPLIT_ASSIGNMENT
    )


def test_dns_receipt_rejects_composite_byte_replacement(tmp_path, monkeypatch):
    receipt, payload, _parent = _write_receipt_fixture(tmp_path, monkeypatch)
    target = tmp_path / payload["selected"][0]["composite_output"]["path"]
    target.write_bytes(target.read_bytes()[:-2] + b"xx")
    with pytest.raises(selection.DNSSelectionError, match="path/SHA/size"):
        selection.validate_dns_selection_receipt(
            repo_root=tmp_path,
            receipt_path=receipt.relative_to(tmp_path).as_posix(),
        )


def test_dns_receipt_rejects_parent_numeric_alias_overlap_even_if_resealed(
    tmp_path, monkeypatch
):
    receipt, payload, parent = _write_receipt_fixture(tmp_path, monkeypatch)
    colliding = payload["selected"][0]["lineage_keys"][0]
    forged_parent = dict(parent)
    forged_parent["speech_lineage_keys"] = sorted(
        set(parent["speech_lineage_keys"]) | {colliding}
    )
    forged_parent["speech_lineage_keys_sha256"] = selection._canonical_json_sha256(
        forged_parent["speech_lineage_keys"]
    )
    payload["parent82"] = forged_parent
    monkeypatch.setattr(
        selection, "_parent_speech_authority", lambda _root: forged_parent
    )
    _reseal(receipt, payload)
    with pytest.raises(selection.DNSSelectionError, match="parent82"):
        selection.validate_dns_selection_receipt(
            repo_root=tmp_path,
            receipt_path=receipt.relative_to(tmp_path).as_posix(),
        )


def test_dns_receipt_rejects_scan_manifest_row_rebinding_even_if_resealed(
    tmp_path, monkeypatch
):
    receipt, payload, _parent = _write_receipt_fixture(tmp_path, monkeypatch)
    payload["scan_results"][0]["group_id"] = "public-lineage-" + "f" * 64
    _reseal(receipt, payload)
    with pytest.raises(selection.DNSSelectionError, match="immutable manifest row"):
        selection.validate_dns_selection_receipt(
            repo_root=tmp_path,
            receipt_path=receipt.relative_to(tmp_path).as_posix(),
        )


def test_dns_receipt_rejects_selected_window_rebinding_even_if_resealed(
    tmp_path, monkeypatch
):
    receipt, payload, _parent = _write_receipt_fixture(tmp_path, monkeypatch)
    payload["selected"][0]["source_window_start_frame"] += 1
    _reseal(receipt, payload)
    with pytest.raises(selection.DNSSelectionError, match="public manifest row"):
        selection.validate_dns_selection_receipt(
            repo_root=tmp_path,
            receipt_path=receipt.relative_to(tmp_path).as_posix(),
        )


def test_dns_receipt_rejects_forged_source_preflight_even_if_fully_resealed(
    tmp_path, monkeypatch
):
    receipt, payload, _parent = _write_receipt_fixture(tmp_path, monkeypatch)
    selected = payload["selected"][-1]
    index = int(selected["manifest_index"])
    forged = copy.deepcopy(selected["source_preflight"])
    timeline = forged["timeline_feasibility"]
    timeline["eligible_windows"] = timeline["total_windows"] - 1
    timeline["eligible_ratio"] = timeline["eligible_windows"] / timeline["total_windows"]
    timeline["longest_ineligible_run_windows"] = 1
    selected["source_preflight"] = forged
    payload["scan_results"][index]["source_preflight"] = copy.deepcopy(forged)
    _reseal(receipt, payload)
    with pytest.raises(selection.DNSSelectionError, match="rendered source preflight"):
        selection.validate_dns_selection_receipt(
            repo_root=tmp_path,
            receipt_path=receipt.relative_to(tmp_path).as_posix(),
        )


def test_dns_receipt_rejects_clean_source_rebinding_even_if_resealed(
    tmp_path, monkeypatch
):
    receipt, payload, _parent = _write_receipt_fixture(tmp_path, monkeypatch)
    payload["clean_source"]["tracked_inventory_sha256"] = "9" * 64
    _reseal(receipt, payload)
    with pytest.raises(selection.DNSSelectionError, match="clean_source evidence"):
        selection.validate_dns_selection_receipt(
            repo_root=tmp_path,
            receipt_path=receipt.relative_to(tmp_path).as_posix(),
        )


def test_dns_receipt_rejects_runtime_or_freeze_rebinding_even_if_resealed(
    tmp_path, monkeypatch
):
    runtime_root = tmp_path / "runtime"
    receipt, payload, _parent = _write_receipt_fixture(runtime_root, monkeypatch)
    payload["selector_runtime"]["scipy_policy"] = "ambient_scipy_allowed"
    _reseal(receipt, payload)
    with pytest.raises(selection.DNSSelectionError, match="isolated runtime"):
        selection.validate_dns_selection_receipt(
            repo_root=runtime_root,
            receipt_path=receipt.relative_to(runtime_root).as_posix(),
        )

    freeze_root = tmp_path / "freeze"
    receipt, payload, _parent = _write_receipt_fixture(freeze_root, monkeypatch)
    freeze = freeze_root / payload["environment_freeze"]["path"]
    freeze.write_bytes(freeze.read_bytes() + b"forged==1\n")
    with pytest.raises(selection.DNSSelectionError, match="path/SHA/size"):
        selection.validate_dns_selection_receipt(
            repo_root=freeze_root,
            receipt_path=receipt.relative_to(freeze_root).as_posix(),
        )


def test_dns_receipt_rejects_fully_resealed_stale_editable_freeze(
    tmp_path, monkeypatch
):
    receipt, payload, _parent = _write_receipt_fixture(tmp_path, monkeypatch)
    current = str(payload["source_commit"])
    stale = ("0" if current[0] != "0" else "1") + current[1:]
    freeze = tmp_path / payload["environment_freeze"]["path"]
    freeze.write_bytes(freeze.read_bytes().replace(current.encode(), stale.encode()))
    freeze_sha = _sha(freeze.read_bytes())
    for key in ("environment_freeze_origin", "environment_freeze"):
        payload[key]["sha256"] = freeze_sha
        payload[key]["size"] = freeze.stat().st_size
    payload["selector_runtime"]["environment_freeze_sha256"] = freeze_sha

    bootstrap = tmp_path / payload["bootstrap_receipt"]["path"]
    bootstrap_payload = json.loads(bootstrap.read_text(encoding="utf-8"))
    bootstrap_payload["environment"]["freeze_receipt_sha256"] = freeze_sha
    bootstrap.write_text(json.dumps(bootstrap_payload) + "\n", encoding="utf-8")
    bootstrap_sha = _sha(bootstrap.read_bytes())
    for key in ("bootstrap_receipt_origin", "bootstrap_receipt"):
        payload[key]["sha256"] = bootstrap_sha
        payload[key]["size"] = bootstrap.stat().st_size
    _reseal(receipt, payload)

    with pytest.raises(selection.DNSSelectionError, match="source 결속 실패"):
        selection.validate_dns_selection_receipt(
            repo_root=tmp_path,
            receipt_path=receipt.relative_to(tmp_path).as_posix(),
        )


def test_dns_receipt_rejects_forged_bytecode_or_libsndfile_backend_even_if_resealed(
    tmp_path, monkeypatch
):
    bytecode_root = tmp_path / "bytecode"
    receipt, payload, _parent = _write_receipt_fixture(bytecode_root, monkeypatch)
    payload["selector_runtime"]["modules"]["numpy"]["cached_path"] = (
        "/home/elicer/Deep_ANC/.venv/lib/python3.10/site-packages/"
        "numpy/__pycache__/__init__.cpython-310.pyc"
    )
    _reseal(receipt, payload)
    with pytest.raises(selection.DNSSelectionError, match="bytecode isolation"):
        selection.validate_dns_selection_receipt(
            repo_root=bytecode_root,
            receipt_path=receipt.relative_to(bytecode_root).as_posix(),
        )

    backend_root = tmp_path / "backend"
    receipt, payload, _parent = _write_receipt_fixture(backend_root, monkeypatch)
    payload["selector_runtime"]["libsndfile"]["path"] = (
        "/tmp/forged/libsndfile_x86_64.so"
    )
    _reseal(receipt, payload)
    with pytest.raises(selection.DNSSelectionError, match="libsndfile"):
        selection.validate_dns_selection_receipt(
            repo_root=backend_root,
            receipt_path=receipt.relative_to(backend_root).as_posix(),
        )


def test_pcm16_scan_values_equal_actual_wav_decoder():
    base = np.asarray(
        [-2.0, -1.0, -0.5, -1.0 / 32767.0, 0.0, 0.5, 1.0, 2.0],
        dtype=np.float64,
    )
    values = np.resize(base, selection.DNS_RAW_FRAMES)
    raw = selection._pcm16_wav_bytes(values)
    decoded = selection._decode_source(raw, label="PCM16 scale fixture")
    assert np.array_equal(decoded, selection._pcm16_decoded_values(values))


def test_dns_source_preflight_matches_exact_rendered_composite():
    time = np.arange(selection.DNS_RAW_FRAMES, dtype=np.float64) / 48_000.0
    values = sum(
        0.04 * np.sin(2.0 * np.pi * frequency * time)
        for frequency in (220.0, 440.0, 800.0, 1250.0)
    )
    raw = selection._pcm16_wav_bytes(values)
    composite = selection.dns_composite_bytes_from_raw(raw)
    bytes_rendered = selection._render_dns_composite_playback(composite)
    scan_rendered = selection._render_dns_window_playback_fast(values)
    observed = selection._rendered_source_preflight(composite)

    program = NoiseProgram(
        {
            "type": "file",
            "file": "fixture.wav",
            "file_start_seconds": 0.0,
            "amplitude": selection.DNS_PLAYBACK_AMPLITUDE,
        },
        selection.DNS_RAW_SAMPLE_RATE,
        file_bytes=composite,
    )
    exact_rendered = np.asarray(
        render_recording_file_window(
            program,
            selection.DNS_COMPOSITE_FRAMES,
            sample_rate=selection.DNS_RAW_SAMPLE_RATE,
            fade_seconds=selection.RECORDING_FILE_FADE_SECONDS,
        ),
        dtype=np.float32,
    )
    assert np.array_equal(bytes_rendered, exact_rendered)
    assert np.array_equal(scan_rendered, bytes_rendered)
    rendered = np.asarray(exact_rendered, dtype=np.float64)
    starts = np.arange(
        0,
        rendered.size - selection.DNS_TIMELINE_SOURCE_SPAN_FRAMES + 1,
        selection.DNS_TIMELINE_HOP_FRAMES,
    )
    rms = np.asarray(
        [
            np.sqrt(
                np.mean(
                    np.square(
                        rendered[
                            start : start
                            + selection.DNS_TIMELINE_SOURCE_SPAN_FRAMES
                        ]
                    )
                )
            )
            for start in starts
        ]
    )
    expected_count = int(np.count_nonzero(rms >= selection.DNS_TIMELINE_MIN_WINDOW_RMS))
    timeline = observed["timeline_feasibility"]
    assert timeline["source_span_samples"] == 13_200
    assert timeline["hop_samples"] == 3_000
    assert timeline["total_windows"] == starts.size == 236
    assert timeline["eligible_windows"] == expected_count
    assert timeline["eligible_ratio"] == pytest.approx(
        expected_count / starts.size, abs=0.0
    )
    assert observed["passed"] is True
    assert observed["trusted_band_rms_dbfs"] >= observed[
        "minimum_trusted_band_rms_dbfs"
    ]


def test_dns_window_source_preflight_avoids_wav_materialization(monkeypatch):
    time = np.arange(selection.DNS_RAW_FRAMES, dtype=np.float64) / 48_000.0
    values = sum(
        0.04 * np.sin(2.0 * np.pi * frequency * time + 0.13)
        for frequency in (220.0, 440.0, 800.0, 1250.0)
    )
    raw = selection._pcm16_wav_bytes(values)
    composite = selection.dns_composite_bytes_from_raw(raw)
    expected = selection._rendered_source_preflight(composite)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("full DNS scan must not encode/decode WAV per window")

    monkeypatch.setattr(selection, "_pcm16_wav_bytes", forbidden)
    monkeypatch.setattr(selection, "dns_composite_bytes_from_raw", forbidden)
    assert selection._window_source_preflight(values) == expected


def test_dns_source_preflight_rejects_intermittent_composite():
    time = np.arange(selection.DNS_RAW_FRAMES, dtype=np.float64) / 48_000.0
    values = np.zeros(selection.DNS_RAW_FRAMES, dtype=np.float64)
    active = 24_000
    values[:active] = sum(
        0.04 * np.sin(2.0 * np.pi * frequency * time[:active])
        for frequency in (220.0, 440.0, 800.0, 1250.0)
    )
    raw = selection._pcm16_wav_bytes(values)
    composite = selection.dns_composite_bytes_from_raw(raw)
    evidence = selection._rendered_source_preflight(composite)
    assert evidence["timeline_feasibility"]["eligible_ratio"] < 0.95
    assert evidence["passed"] is False


def test_dns_source_preflight_rejects_continuous_out_of_band_composite():
    time = np.arange(selection.DNS_RAW_FRAMES, dtype=np.float64) / 48_000.0
    values = 0.05 * np.sin(2.0 * np.pi * 4_000.0 * time)
    raw = selection._pcm16_wav_bytes(values)
    composite = selection.dns_composite_bytes_from_raw(raw)

    evidence = selection._rendered_source_preflight(composite)
    assert evidence["timeline_feasibility"]["passed"] is True
    assert evidence["trusted_band_rms_dbfs"] < evidence[
        "minimum_trusted_band_rms_dbfs"
    ]
    assert evidence["passed"] is False


def test_dns_density_uses_repo_numpy_fft_not_ambient_scipy(monkeypatch):
    values = np.linspace(-0.2, 0.2, 8192, dtype=np.float64)
    fir = np.asarray([0.25, 0.5, 0.25], dtype=np.float64)
    baseline = selection._band_density(values, fir)

    def forged_convolution(*_args, **_kwargs):
        raise AssertionError("ambient scipy.signal must never be called")

    monkeypatch.setitem(
        sys.modules,
        "scipy",
        types.SimpleNamespace(
            signal=types.SimpleNamespace(fftconvolve=forged_convolution)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "scipy.signal",
        types.SimpleNamespace(fftconvolve=forged_convolution),
    )
    assert selection._band_density(values, fir) == baseline


def test_full_scan_keeps_metrics_only_and_rematerializes_selected_bytes(tmp_path):
    rows = []
    time = np.arange(selection.DNS_RAW_FRAMES, dtype=np.float64) / 48_000.0
    for index in range(6):
        filename = f"book_{700 + index}_chp_1_reader_{800 + index}.wav"
        keys = list(
            public_lineage.conservative_cross_corpus_speech_lineage_keys(
                public_lineage.dns_speech_lineage_keys(filename)
            )
        )
        values = sum(
            0.04
            * np.sin(2.0 * np.pi * frequency * time + index * 0.11)
            for frequency in (220.0, 440.0, 800.0, 1250.0)
        )
        if index == 5:
            intermittent = np.zeros_like(values)
            intermittent[:24_000] = values[:24_000]
            values = intermittent
        raw = selection._pcm16_wav_bytes(values)
        source = tmp_path / "data/raw/noise/speech" / filename
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(raw)
        content = _sha(raw)
        group = "public-lineage-" + public_lineage.canonical_json_sha256(
            {"lineage_keys": keys, "content_sha256": [content]}
        )
        rows.append(
            {
                "path": source.relative_to(tmp_path).as_posix(),
                "duration_s": selection.DNS_RAW_SECONDS,
                "sample_rate": 48_000,
                "channels": 1,
                "tag": "speech",
                "split": "train",
                "content_sha256": content,
                "content_size": len(raw),
                "lineage_schema": public_lineage.PUBLIC_LINEAGE_SCHEMA,
                "lineage_keys": keys,
                "group_id": group,
            }
        )
    lineage = public_lineage.validate_public_manifest_lineage({"speech": rows})
    results = selection._scan_manifest_rows(
        repo_root=tmp_path,
        rows=rows,
        lineage_summary=lineage,
        parent_keys={
            "conservative_speech_book_numeric:999999",
            "conservative_speech_reader_numeric:999999",
        },
        fir=np.asarray([1.0], dtype=np.float64),
    )
    assert len(results) == 6
    assert all(set(result) == selection._SCAN_RESULT_KEYS for result in results)
    assert all(result["coverage_scan"]["covered_subband_count"] == 4 for result in results)
    assert all(result["source_preflight"]["passed"] for result in results[:5])
    assert results[5]["status"] == "ineligible"
    assert results[5]["reason"] == "rendered_source_preflight_below_minimum"
    assert results[5]["source_preflight"]["passed"] is False
    chosen = selection._select_results(results)
    assert len(chosen) == 5
    raw, composite = selection._materialize_selected_bytes(
        repo_root=tmp_path,
        row=rows[int(chosen[0]["manifest_index"])],
        result=chosen[0],
    )
    assert len(selection._decode_source(raw, label="selected raw")) == selection.DNS_RAW_FRAMES
    assert selection.dns_composite_bytes_from_raw(raw) == composite


def test_selector_bundle_publish_is_atomic_no_replace(tmp_path, monkeypatch):
    selector = _load_selector_script()
    bundle = "data/source_plans/recorded_additions/dns-probe"
    (tmp_path / "data/source_plans/recorded_additions").mkdir(parents=True)
    monkeypatch.setattr(selector, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(selector, "DNS_SELECTION_BUNDLE_ROOT", bundle)
    monkeypatch.setattr(
        selector,
        "DNS_SELECTION_RECEIPT",
        f"{bundle}/selection_receipt.json",
    )
    publication = selector._write_bundle_no_replace(
        {"schema_version": 1, "evidence_sha256": "a" * 64},
        {"sources/probe.wav": b"immutable"},
    )
    destination = tmp_path / bundle
    assert (destination / "sources/probe.wav").read_bytes() == b"immutable"
    publication.close()
    assert not list(destination.parent.glob(".dns-selection-*"))
    with pytest.raises(FileExistsError):
        selector._write_bundle_no_replace(
            {"schema_version": 1, "evidence_sha256": "b" * 64},
            {"sources/probe.wav": b"replacement"},
        )
    assert (destination / "sources/probe.wav").read_bytes() == b"immutable"


@pytest.mark.parametrize(
    "failure_stage", ("post_publish_trust", "post_publish_receipt_validation")
)
def test_post_publish_failure_quarantines_owned_bundle_and_retry_succeeds(
    tmp_path, monkeypatch, failure_stage
):
    selector = _load_selector_script()
    bundle = "data/source_plans/recorded_additions/dns-post-failure"
    destination = tmp_path / bundle
    destination.parent.mkdir(parents=True)
    monkeypatch.setattr(selector, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(selector, "DNS_SELECTION_BUNDLE_ROOT", bundle)
    monkeypatch.setattr(
        selector,
        "DNS_SELECTION_RECEIPT",
        f"{bundle}/selection_receipt.json",
    )
    payload = {
        "schema_version": 2,
        "source_commit": "a" * 40,
        "evidence_sha256": "b" * 64,
    }
    files = {"sources/probe.wav": b"immutable"}
    trust_calls = 0

    def trust(_payload):
        nonlocal trust_calls
        trust_calls += 1
        if failure_stage == "post_publish_trust" and trust_calls == 2:
            raise selection.DNSSelectionError("post trust injected failure")

    def validate(**_kwargs):
        if failure_stage == "post_publish_receipt_validation":
            raise selection.DNSSelectionError("receipt injected failure")
        return {"status": "valid"}

    monkeypatch.setattr(selector, "_assert_publish_trust", trust)
    monkeypatch.setattr(selector, "validate_dns_selection_receipt", validate)
    with pytest.raises(selection.DNSSelectionError, match="writer-owned bundle"):
        selector._publish_verified_bundle(payload, files)
    assert not destination.exists()
    failures = list((destination.parent / ".publish-failures").iterdir())
    assert len(failures) == 1
    receipt = json.loads(
        (failures[0] / "publish_failure_receipt.json").read_text(encoding="utf-8")
    )
    evidence_sha = receipt.pop("evidence_sha256")
    assert evidence_sha == selector._canonical_json_sha256(receipt)
    assert receipt["failure_stage"] == failure_stage
    assert (failures[0] / "sources/probe.wav").read_bytes() == b"immutable"

    monkeypatch.setattr(selector, "_assert_publish_trust", lambda _payload: None)
    monkeypatch.setattr(
        selector,
        "validate_dns_selection_receipt",
        lambda **_kwargs: {"status": "valid"},
    )
    assert selector._publish_verified_bundle(payload, files) == {"status": "valid"}
    assert (destination / "sources/probe.wav").read_bytes() == b"immutable"


def test_publish_loser_never_quarantines_existing_winner(tmp_path, monkeypatch):
    selector = _load_selector_script()
    bundle = "data/source_plans/recorded_additions/dns-winner"
    destination = tmp_path / bundle
    destination.mkdir(parents=True)
    (destination / "winner.txt").write_bytes(b"winner")
    monkeypatch.setattr(selector, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(selector, "DNS_SELECTION_BUNDLE_ROOT", bundle)
    monkeypatch.setattr(
        selector,
        "DNS_SELECTION_RECEIPT",
        f"{bundle}/selection_receipt.json",
    )
    monkeypatch.setattr(selector, "_assert_publish_trust", lambda _payload: None)
    with pytest.raises(FileExistsError):
        selector._publish_verified_bundle(
            {"source_commit": "a" * 40, "evidence_sha256": "b" * 64},
            {"sources/probe.wav": b"loser"},
        )
    assert (destination / "winner.txt").read_bytes() == b"winner"
    assert not (destination.parent / ".publish-failures").exists()


def test_failed_publication_quarantine_refuses_replaced_identity(
    tmp_path, monkeypatch
):
    selector = _load_selector_script()
    bundle = "data/source_plans/recorded_additions/dns-owned"
    destination = tmp_path / bundle
    destination.parent.mkdir(parents=True)
    monkeypatch.setattr(selector, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(selector, "DNS_SELECTION_BUNDLE_ROOT", bundle)
    monkeypatch.setattr(
        selector,
        "DNS_SELECTION_RECEIPT",
        f"{bundle}/selection_receipt.json",
    )
    payload = {"source_commit": "a" * 40, "evidence_sha256": "b" * 64}
    publication = selector._write_bundle_no_replace(
        payload, {"sources/probe.wav": b"owned"}
    )
    displaced = destination.parent / "owned-displaced"
    destination.rename(displaced)
    destination.mkdir()
    (destination / "winner.txt").write_bytes(b"other process")
    try:
        with pytest.raises(selection.DNSSelectionError, match="identity"):
            selector._quarantine_failed_publication(
                publication,
                payload=payload,
                stage="post_publish_trust",
                failure=RuntimeError("injected"),
            )
    finally:
        publication.close()
    assert (destination / "winner.txt").read_bytes() == b"other process"
    assert (displaced / "sources/probe.wav").read_bytes() == b"owned"


def test_verify_existing_requires_and_checks_all_external_anchors(
    monkeypatch,
):
    selector = _load_selector_script()
    commit = "a" * 40
    bootstrap_sha = "b" * 64
    receipt_sha = "c" * 64
    observed = {}

    def validate(**kwargs):
        observed["expected_receipt_sha256"] = kwargs["expected_receipt_sha256"]
        return {
            "source_commit": commit,
            "bootstrap_receipt_sha256": bootstrap_sha,
            "receipt_sha256": receipt_sha,
        }

    monkeypatch.setattr(selector, "validate_dns_selection_receipt", validate)
    base = [
        "--expected-commit",
        commit,
        "--bootstrap-receipt-sha256",
        bootstrap_sha,
        "--verify-existing",
    ]
    assert selector.main([*base, "--receipt-sha256", receipt_sha]) == 0
    assert observed["expected_receipt_sha256"] == receipt_sha
    assert selector.main(
        [
            "--expected-commit",
            "d" * 40,
            "--bootstrap-receipt-sha256",
            bootstrap_sha,
            "--verify-existing",
            "--receipt-sha256",
            receipt_sha,
        ]
    ) == 1
    assert selector.main(
        [
            "--expected-commit",
            commit,
            "--bootstrap-receipt-sha256",
            "e" * 64,
            "--verify-existing",
            "--receipt-sha256",
            receipt_sha,
        ]
    ) == 1
    assert selector.main(base) == 1


def test_receipt_sha_option_is_rejected_outside_verify_mode(monkeypatch):
    selector = _load_selector_script()
    assert selector.main(
        [
            "--expected-commit",
            "a" * 40,
            "--bootstrap-receipt-sha256",
            "b" * 64,
            "--receipt-sha256",
            "c" * 64,
            "--check-only",
        ]
    ) == 1


def test_selector_publish_boundary_rechecks_same_source_and_runtime(monkeypatch):
    selector = _load_selector_script()
    source = {"commit": "a" * 40, "inventory": "b" * 64}
    runtime = {"schema": "isolated", "modules": {"numpy": "c" * 64}}
    payload = {
        "source_commit": "a" * 40,
        "clean_source": source,
        "environment_freeze_origin": {
            "path": ".venv/environment-freeze.txt",
            "sha256": "d" * 64,
        },
        "selector_runtime": runtime,
    }
    monkeypatch.setattr(
        selector,
        "exact_clean_source_evidence",
        lambda *_args, **_kwargs: json.loads(json.dumps(source)),
    )
    monkeypatch.setattr(
        selector,
        "exact_selector_runtime_evidence",
        lambda *_args, **_kwargs: json.loads(json.dumps(runtime)),
    )
    selector._assert_publish_trust(payload)
    monkeypatch.setattr(
        selector,
        "exact_selector_runtime_evidence",
        lambda *_args, **_kwargs: {"schema": "forged"},
    )
    with pytest.raises(selection.DNSSelectionError, match="runtime이 변경"):
        selector._assert_publish_trust(payload)


def test_source_plan_requires_and_preserves_external_receipt_sha(monkeypatch):
    builder = _load_source_plan_builder()
    with pytest.raises(SystemExit):
        builder._parser().parse_args(["--check-only"])

    receipt_sha = "d" * 64
    demand_receipt_sha = "4" * 64
    observed = {}
    monkeypatch.setattr(builder, "CANONICAL_SOURCE_POOL_ADDITIONS", {})
    monkeypatch.setattr(builder, "CANONICAL_EXTERNAL_ESC_MACHINE_FILES", {})
    monkeypatch.setattr(builder, "_canonical_source_lineage", lambda _root: {})
    monkeypatch.setattr(
        builder,
        "_canonical_source_selection_evidence",
        lambda _root, _authority: {"evidence_sha256": "e" * 64},
    )

    def fake_receipt(**kwargs):
        observed["expected"] = kwargs["expected_receipt_sha256"]
        return {
            "receipt_path": selection.DNS_SELECTION_RECEIPT,
            "receipt_sha256": receipt_sha,
            "public_manifest_sha256": "f" * 64,
            "selected": [
                {
                    "public_group_id": "public-lineage-" + "1" * 64,
                    "recorded_split": "val",
                    "raw_output": {
                        "path": "data/source_plans/recorded_additions/raw.wav",
                        "sha256": "2" * 64,
                    },
                    "composite_output": {
                        "path": "data/source_plans/recorded_additions/composite.wav",
                        "sha256": "3" * 64,
                    },
                }
            ],
        }

    monkeypatch.setattr(builder, "validate_dns_selection_receipt", fake_receipt)
    def fake_demand_receipt(**kwargs):
        observed["demand_expected"] = kwargs["expected_receipt_sha256"]
        return {
            "receipt_path": builder.DEMAND_SELECTION_RECEIPT,
            "receipt_sha256": demand_receipt_sha,
            "public_manifest_sha256": "5" * 64,
            "selected": {
                "bundle_source": {"sha256": "6" * 64},
                "origin_bundle_source": {"sha256": "7" * 64},
            },
        }

    monkeypatch.setattr(
        builder,
        "validate_demand_selection_receipt",
        fake_demand_receipt,
    )
    rows = builder.build_rows(
        builder.CANONICAL_GENERATION_ID,
        dns_selection_receipt_sha256=receipt_sha,
        demand_selection_receipt_sha256=demand_receipt_sha,
    )
    assert observed["expected"] == receipt_sha
    assert observed["demand_expected"] == demand_receipt_sha
    dns_row = next(
        row
        for row in rows
        if row["source_kind"] == builder.SOURCE_KIND_EXTERNAL_DNS_SPEECH
    )
    assert dns_row["inventory_sha256"] == receipt_sha
    demand_row = next(
        row
        for row in rows
        if row["source_kind"]
        == builder.SOURCE_KIND_EXTERNAL_DEMAND_ENVIRONMENT
    )
    assert demand_row == {
        **{field: "" for field in builder.SOURCE_PLAN_FIELDS},
        "source_kind": builder.SOURCE_KIND_EXTERNAL_DEMAND_ENVIRONMENT,
        "path": builder.DEMAND_SELECTION_SOURCE,
        "seconds": str(builder.DEMAND_WINDOW_SECONDS),
        "start_seconds": str(builder.DEMAND_WINDOW_START_SECONDS),
        "source_family": "environment",
        "group_id": builder.DEMAND_PUBLIC_GROUP_ID,
        "lineage_key": builder.DEMAND_LINEAGE_KEY,
        "split": builder.DEMAND_RECORDED_SPLIT,
        "source_file_sha256": "6" * 64,
        "raw_member_path": builder.DEMAND_SELECTION_ORIGIN_SOURCE,
        "raw_member_sha256": "7" * 64,
        "raw_member_lineage_key": builder.DEMAND_PUBLIC_GROUP_ID,
        "authority_metadata_sha256": "5" * 64,
        "inventory_path": builder.DEMAND_SELECTION_RECEIPT,
        "inventory_sha256": demand_receipt_sha,
        "transform": builder.DEMAND_TRANSFORM,
        "transform_repeat_count": "1",
    }


def test_source_plan_rejects_stale_fixed_gain_generation_before_receipt_reuse():
    builder = _load_source_plan_builder()
    assert builder.CANONICAL_GENERATION_ID == "stage1-coverage-v3-gain012"
    assert selection.DNS_SELECTION_GENERATION_ID == builder.CANONICAL_GENERATION_ID
    with pytest.raises(ValueError, match="현행 exact source plan generation-id"):
        builder.build_rows(
            "stage1-coverage-v2",
            dns_selection_receipt_sha256="a" * 64,
            demand_selection_receipt_sha256="b" * 64,
        )


def test_selector_requires_five_distinct_full_coverage_groups():
    scans = []
    for index, split in enumerate(("train", "train", "val", "test", "test")):
        scans.append(
            {
                "manifest_index": index,
                "group_id": "same" if index < 2 else f"group-{index}",
                "public_source_split": split,
                "status": "eligible",
                "coverage_scan": {
                    "density_ratios": [1.0, 1.0, 1.0, 1.0],
                    "covered_subband_count": 4,
                },
                "source_preflight": {
                    "timeline_feasibility": {"eligible_ratio": 1.0},
                    "passed": True,
                },
            }
        )
    with pytest.raises(selection.DNSSelectionBlocked, match="독립 public"):
        selection._select_results(scans)
