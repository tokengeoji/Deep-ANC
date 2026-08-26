"""source-pool provenance 복구가 추정이나 부분 성공으로 열리지 않는지 검사한다."""

from __future__ import annotations

import csv
import argparse
import hashlib
import importlib.util
import io
import json
import os
import stat
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def provenance():
    return _load(
        "repair_source_pool_provenance",
        "scripts/data/repair_source_pool_provenance.py",
    )


def _plan(
    module,
    *,
    pool: str = "source_pool",
    family: str = "machine",
    index: int = 0,
    clips=None,
    wav=None,
):
    clips = tuple(clips or ("a.wav", "b.wav", "c.wav"))
    return module.RowPlan(
        pool_name=pool,
        builder_commit=module.V1_COMMIT,
        source_family=family,
        session_index=index,
        group_id="machine-engine" if family == "machine" else f"{family}-fixture",
        wav_path=wav or f"data/{pool}/{family}/{family}_{index:03d}.wav",
        source_paths=tuple(f"/raw/{item}" for item in clips),
        clips=clips,
    )


def _csv_bytes(clips: list[str]) -> bytes:
    output = io.StringIO(newline="")
    fields = [
        "source_family",
        "session_index",
        "group_id",
        "path",
        "seconds",
        "sample_rate_hz",
        "clip_count",
        "crest_factor_db",
        "rms_at_unit_peak",
        "clips",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerow(
        {
            "source_family": "machine",
            "session_index": 0,
            "group_id": "machine-engine",
            "path": "data/source_pool/machine/machine_000.wav",
            "seconds": 70.0,
            "sample_rate_hz": 48000,
            "clip_count": 3,
            "crest_factor_db": 9.7,
            "rms_at_unit_peak": 0.3,
            "clips": json.dumps(clips),
        }
    )
    return output.getvalue().encode()


def test_historical_git_blobs_are_pinned(provenance) -> None:
    for commit, expected in provenance.BUILDER_SHA256.items():
        source = provenance._git_builder_source(REPO_ROOT, commit)
        assert provenance.sha256_bytes(source.encode()) == expected


def test_truncated_csv_is_only_accepted_when_it_is_an_exact_prefix(
    provenance, tmp_path: Path
) -> None:
    path = tmp_path / "sources.csv"
    path.write_bytes(_csv_bytes(["a.wav", "b.wav"]))
    audit, _fields, _rows, _raw = provenance.audit_csv_prefix(
        path, [_plan(provenance)]
    )
    assert audit["status"] == "PASS"
    assert audit["missing_clip_placements"] == 1

    path.write_bytes(_csv_bytes(["a.wav", "wrong.wav"]))
    audit, *_ = provenance.audit_csv_prefix(path, [_plan(provenance)])
    assert audit["status"] == "FAIL"
    assert "clips_prefix" in "\n".join(audit["issues"])


def test_repair_changes_only_clips_column(provenance, tmp_path: Path) -> None:
    path = tmp_path / "sources.csv"
    path.write_bytes(_csv_bytes(["a.wav", "b.wav"]))
    fields, rows, _ = provenance.read_csv_rows(path)
    before = dict(rows[0])
    repaired = provenance.render_repaired_csv(fields, rows, [_plan(provenance)])
    repaired_path = tmp_path / "repaired.csv"
    repaired_path.write_bytes(repaired)
    _fields, after_rows, _raw = provenance.read_csv_rows(repaired_path)
    after = after_rows[0]
    assert json.loads(after.pop("clips")) == ["a.wav", "b.wav", "c.wav"]
    before.pop("clips")
    assert after == before


def test_transaction_refuses_any_partial_pool_failure(
    provenance, tmp_path: Path
) -> None:
    path = tmp_path / "sources.csv"
    original = _csv_bytes(["a.wav", "b.wav"])
    path.write_bytes(original)
    fields, rows, raw = provenance.read_csv_rows(path)
    with pytest.raises(provenance.ProvenanceError, match="전체 provenance PASS"):
        provenance.repair_csvs_transactionally(
            {"source_pool": {"status": "FAIL"}},
            {"source_pool": (path, fields, rows, raw)},
            {"source_pool": [_plan(provenance)]},
        )
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.bak")) == []


def test_passed_transaction_keeps_backup_and_fills_full_list(
    provenance, tmp_path: Path
) -> None:
    path = tmp_path / "sources.csv"
    original = _csv_bytes(["a.wav", "b.wav"])
    path.write_bytes(original)
    fields, rows, raw = provenance.read_csv_rows(path)
    result = provenance.repair_csvs_transactionally(
        {"source_pool": {"status": "PASS"}},
        {"source_pool": (path, fields, rows, raw)},
        {"source_pool": [_plan(provenance)]},
    )
    assert result["source_pool"]["changed"] is True
    _fields, repaired, _raw = provenance.read_csv_rows(path)
    assert json.loads(repaired[0]["clips"]) == ["a.wav", "b.wav", "c.wav"]
    backup = Path(result["source_pool"]["backup"])
    assert backup.read_bytes() == original


def test_csv_transaction_rolls_back_replace_when_directory_fsync_fails(
    provenance, tmp_path: Path, monkeypatch
) -> None:
    inputs = {}
    plans = {}
    audits = {}
    originals = {}
    for pool in ("source_pool", "source_pool_v2"):
        path = tmp_path / pool / "sources.csv"
        path.parent.mkdir()
        original = _csv_bytes(["a.wav", "b.wav"]).replace(
            b"data/source_pool/", f"data/{pool}/".encode()
        )
        path.write_bytes(original)
        fields, rows, raw = provenance.read_csv_rows(path)
        inputs[pool] = (path, fields, rows, raw)
        plans[pool] = [_plan(provenance, pool=pool)]
        audits[pool] = {"status": "PASS"}
        originals[pool] = original

    second_target = inputs["source_pool_v2"][0]
    real_replace = provenance.os.replace
    real_fsync = provenance.os.fsync
    state = {"directory_fsync_armed": False, "raised": False}

    def replace_then_arm(source, target):
        real_replace(source, target)
        if Path(target) == second_target and not state["raised"]:
            state["directory_fsync_armed"] = True

    def fail_first_armed_directory_fsync(descriptor):
        if (
            state["directory_fsync_armed"]
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
            and not state["raised"]
        ):
            state["raised"] = True
            state["directory_fsync_armed"] = False
            raise OSError("injected directory fsync failure after replace")
        return real_fsync(descriptor)

    monkeypatch.setattr(provenance.os, "replace", replace_then_arm)
    monkeypatch.setattr(provenance.os, "fsync", fail_first_armed_directory_fsync)

    with pytest.raises(OSError, match="directory fsync failure after replace"):
        provenance.repair_csvs_transactionally(audits, inputs, plans)

    assert state["raised"] is True
    for pool, (path, *_rest) in inputs.items():
        assert path.read_bytes() == originals[pool]


def test_float_wav_requires_exact_but_integer_pcm_allows_one_lsb(provenance) -> None:
    expected = np.array([0.0, 0.25, -0.5], dtype=np.float32)
    exact = provenance.compare_pcm(expected, expected.copy(), "FLOAT")
    assert exact["status"] == "PASS" and exact["accepted_by"] == "exact"

    one_lsb = expected.copy()
    one_lsb[0] += np.float32(1.0 / 32768.0)
    assert provenance.compare_pcm(expected, one_lsb, "PCM_16")["status"] == "PASS"
    assert provenance.compare_pcm(expected, one_lsb, "FLOAT")["status"] == "FAIL"


def test_source_wav_header_pcm_and_sha_use_one_bytes_snapshot(
    provenance, tmp_path: Path, monkeypatch
) -> None:
    wav = tmp_path / "data/source_pool/machine/machine_000.wav"
    wav.parent.mkdir(parents=True)
    expected = np.zeros(32, dtype=np.float32)
    provenance.sf.write(wav, expected, provenance.TARGET_RATE, subtype="FLOAT")
    fake_builder = types.SimpleNamespace(
        load_audio=lambda _path: (expected.copy(), provenance.TARGET_RATE),
        to_target_rate=lambda signal, _rate: signal,
        concatenate=lambda _clips, _length: expected.copy(),
    )
    plan = provenance.RowPlan(
        pool_name="source_pool",
        builder_commit=provenance.V1_COMMIT,
        source_family="machine",
        session_index=0,
        group_id="machine-engine",
        wav_path="data/source_pool/machine/machine_000.wav",
        source_paths=("/raw/only.wav",),
        clips=("only.wav",),
    )
    provenance._WORKER_MODULES[provenance.V1_COMMIT] = fake_builder
    real_info = provenance.sf.info
    real_read = provenance.sf.read
    observed: list[str] = []

    def info_from_snapshot(value, *args, **kwargs):
        assert isinstance(value, io.BytesIO)
        observed.append("info")
        return real_info(value, *args, **kwargs)

    def read_from_snapshot(value, *args, **kwargs):
        assert isinstance(value, io.BytesIO)
        observed.append("read")
        return real_read(value, *args, **kwargs)

    monkeypatch.setattr(provenance.sf, "info", info_from_snapshot)
    monkeypatch.setattr(provenance.sf, "read", read_from_snapshot)
    result = provenance.verify_wav_plan(plan, repo_root=tmp_path)
    assert result["status"] == "PASS"
    assert observed == ["info", "read"]
    assert result["wav_sha256"] == hashlib.sha256(wav.read_bytes()).hexdigest()


def test_active_holdout_uses_only_wavs_observed_in_sessions(provenance) -> None:
    active = _plan(provenance, index=0, clips=("active-a.wav", "active-b.wav"))
    unused = _plan(provenance, index=1, clips=("unused.wav",))
    sessions = [
        {
            "session_id": "s0",
            "source_family": "machine",
            "source_wav": active.wav_path,
        }
    ]
    payload, used = provenance.build_active_holdout(
        sessions,
        {"source_pool": [active, unused], "source_pool_v2": []},
        csv_hashes={"source_pool": "abc"},
        report_path="report.json",
    )
    assert payload["active_source_row_count"] == 1
    assert payload["families"]["machine"] == ["active-a.wav", "active-b.wav"]
    assert "unused.wav" not in json.dumps(payload)
    assert set(used) == {active.wav_path}


def test_holdout_is_bound_only_after_fixed_report_bytes_exist(
    provenance, tmp_path: Path
) -> None:
    report = tmp_path / "results/provenance/report.json"
    report.parent.mkdir(parents=True)
    report.write_bytes(b'{"status":"PASS"}\n')
    holdout = {"provenance_report": "results/provenance/report.json"}
    bound = provenance.bind_holdout_to_fixed_report(
        holdout, report_path=report, repo_root=tmp_path
    )

    assert "provenance_report_sha256" not in holdout
    assert bound["provenance_report_sha256"] == provenance.sha256_file(report)
    before = bound["provenance_report_sha256"]
    report.write_bytes(b'{"status":"FAIL"}\n')
    assert provenance.sha256_file(report) != before


def test_invalid_candidate_never_overwrites_existing_canonical_holdout(
    provenance, tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "data/manifests/recorded_holdout.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old-canonical\n")

    def reject(_path, *, repo_root):
        raise provenance.HoldoutContractError("injected candidate rejection")

    monkeypatch.setattr(provenance, "validate_holdout_contract", reject)
    with pytest.raises(provenance.HoldoutContractError, match="candidate rejection"):
        provenance.validate_then_atomic_write_holdout(
            target, b"new-invalid\n", repo_root=tmp_path
        )
    assert target.read_bytes() == b"old-canonical\n"


def test_content_addressed_report_publish_is_immutable_and_audit_never_overwrites_canonical(
    provenance, tmp_path: Path
) -> None:
    report_dir = tmp_path / "results/provenance"
    report_dir.mkdir(parents=True)
    canonical_bytes = b'{"mode":"repair","status":"PASS"}\n'
    canonical, canonical_sha = provenance.publish_immutable_report(
        report_dir,
        canonical_bytes,
        canonical=True,
        repo_root=tmp_path,
    )
    audit, audit_sha = provenance.publish_immutable_report(
        report_dir,
        b'{"mode":"audit_only","status":"FAIL"}\n',
        canonical=False,
        repo_root=tmp_path,
    )

    assert canonical.name == f"source_pool_provenance_report.{canonical_sha}.json"
    assert audit.name == f"source_pool_provenance_audit.{audit_sha}.json"
    assert canonical.read_bytes() == canonical_bytes
    assert canonical != audit
    repeated, _ = provenance.publish_immutable_report(
        report_dir,
        canonical_bytes,
        canonical=True,
        repo_root=tmp_path,
    )
    assert repeated == canonical


def test_canonical_report_is_deterministic_and_excludes_diagnostic_downstream_state(
    provenance, tmp_path: Path,
) -> None:
    recorded_root = tmp_path / "data/recorded"
    recorded_root.mkdir(parents=True)
    (recorded_root / "evidence.wav").write_bytes(b"immutable-recorded-fixture")
    recorded_snapshot = provenance.snapshot_recorded_tree(
        recorded_root, repo_root=tmp_path
    )
    plans = {
        "source_pool": [_plan(provenance, pool="source_pool", index=0)],
        "source_pool_v2": [_plan(provenance, pool="source_pool_v2", index=0)],
    }
    pcm = {
        name: {
            "pcm": {
                "status": "PASS",
                "passed_rows": 1,
                "expected_rows": 1,
                "rows": [{"path": rows[0].wav_path, "status": "PASS"}],
            },
            # 첫 repair와 재감사에서 달라질 수 있는 pre-repair 진단은 권위 payload가
            # 소비하지 않아야 한다.
            "csv": {"missing_clip_placements": 99},
        }
        for name, rows in plans.items()
    }
    kwargs = dict(
        plans_by_pool=plans,
        pool_audits=pcm,
        selection_evidence={"seed": provenance.SEED},
        csv_hashes={"source_pool": "a" * 64, "source_pool_v2": "b" * 64},
        active_holdout_gate={
            "status": "PASS",
            "active_session_count": 82,
            "active_source_row_count": 80,
            "total_clips": 100,
            "clip_lineage_sha256": "d" * 64,
            "clip_lineage_metadata": {
                "librispeech_chapters": {"sha256": "e" * 64},
                "fma_tracks": {"sha256": provenance.FMA_TRACKS_CSV_SHA256},
                "esc50": {"sha256": "f" * 64},
            },
        },
        recorded_before=recorded_snapshot,
        recorded_after=recorded_snapshot,
        component_plan={
            "status": "READY",
            "tracks_csv_sha256": provenance.FMA_TRACKS_CSV_SHA256,
            "librispeech_chapters_sha256": "e" * 64,
            "esc50_metadata_sha256": "f" * 64,
            "active_session_count": 2,
            "component_count": 1,
            "component_count_by_family": {"machine": 1},
            "components": {"machine-lineage-fixture": ["s1", "s2"]},
        },
        regrouped_manifest={
            "status": "PASS",
            "sha256": "c" * 64,
            "session_count": 2,
            "component_count": 1,
            "groups_by_family_split": {
                "machine": {"train": 1, "val": 0, "test": 0}
            },
            "lineage_cross_split_count": 0,
        },
        recorded_clip_split={"status": "PASS", "cross_split_clip_count": 0},
    )
    first = provenance.build_canonical_provenance_report(**kwargs)
    pcm["source_pool"]["csv"]["missing_clip_placements"] = 0
    second = provenance.build_canonical_provenance_report(**kwargs)

    assert provenance._report_bytes(first) == provenance._report_bytes(second)
    assert "created_at" not in first and "environment" not in first
    assert set(first["downstream_gates"]) == {"active_holdout"}
    assert "synthetic_corpus" not in json.dumps(first)
    tree = first["recorded_tree_protection"]
    assert tree["before_sha256"] == tree["after_sha256"] == recorded_snapshot.sha256
    assert (
        tree["before_content_sha256"]
        == tree["after_content_sha256"]
        == recorded_snapshot.content_sha256
    )
    assert tree["file_count"] == 1
    assert "transfer manifest" in tree["content_integrity_boundary"]


def test_recorded_tree_snapshot_detects_metadata_change_and_rejects_symlink(
    provenance, tmp_path: Path
) -> None:
    recorded_root = tmp_path / "data/recorded"
    recorded_root.mkdir(parents=True)
    first_path = recorded_root / "session-b/audio.wav"
    first_path.parent.mkdir()
    first_path.write_bytes(b"before")
    (recorded_root / "session-a").mkdir()
    (recorded_root / "session-a/session.json").write_bytes(b"{}")
    before = provenance.snapshot_recorded_tree(recorded_root, repo_root=tmp_path)
    assert [entry[0] for entry in before.entries] == [
        "session-a/session.json",
        "session-b/audio.wav",
    ]

    original_mtime_ns = next(
        mtime_ns for relative, _size, mtime_ns in before.entries
        if relative == "session-b/audio.wav"
    )
    # 공격자가 같은 size로 내용을 바꾸고 mtime까지 원복해 metadata digest를
    # 보존해도 same-FD content aggregate가 반드시 달라져야 한다.
    first_path.write_bytes(b"aftore")
    os.utime(first_path, ns=(original_mtime_ns, original_mtime_ns))
    after = provenance.snapshot_recorded_tree(recorded_root, repo_root=tmp_path)
    evidence = provenance.recorded_tree_protection_evidence(before, after)
    assert evidence["status"] == "FAIL"
    assert evidence["before_sha256"] == evidence["after_sha256"]
    assert evidence["before_content_sha256"] != evidence["after_content_sha256"]

    first_path.unlink()
    first_path.symlink_to(recorded_root / "session-a/session.json")
    with pytest.raises(provenance.HoldoutContractError, match="symlink|regular file"):
        provenance.snapshot_recorded_tree(recorded_root, repo_root=tmp_path)


def test_report_publish_then_candidate_failure_keeps_old_bundle_recoverable(
    provenance, tmp_path: Path, monkeypatch
) -> None:
    report_dir = tmp_path / "results/provenance"
    report_dir.mkdir(parents=True)
    old_report, _ = provenance.publish_immutable_report(
        report_dir, b'{"generation":1}\n', canonical=True, repo_root=tmp_path
    )
    new_report, _ = provenance.publish_immutable_report(
        report_dir, b'{"generation":2}\n', canonical=True, repo_root=tmp_path
    )
    target = tmp_path / "data/manifests/recorded_holdout.json"
    target.parent.mkdir(parents=True)
    old_holdout = json.dumps({"provenance_report": str(old_report)}).encode()
    target.write_bytes(old_holdout)

    monkeypatch.setattr(
        provenance,
        "validate_holdout_contract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            provenance.HoldoutContractError("injected crash boundary")
        ),
    )
    with pytest.raises(provenance.HoldoutContractError, match="crash boundary"):
        provenance.validate_then_atomic_write_holdout(
            target,
            json.dumps({"provenance_report": str(new_report)}).encode(),
            repo_root=tmp_path,
        )
    assert target.read_bytes() == old_holdout
    assert old_report.is_file() and new_report.is_file()


def test_authoritative_output_paths_are_exact_and_symlink_parents_are_rejected(
    provenance, tmp_path: Path
) -> None:
    for directory in (
        "data/source_pool",
        "data/source_pool_v2",
        "data/manifests",
        "results/provenance",
    ):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    base = dict(
        v1_csv="data/source_pool/sources.csv",
        v2_csv="data/source_pool_v2/sources.csv",
        report="results/provenance",
        active_holdout="data/manifests/recorded_holdout.json",
        regrouped_manifest="data/manifests/recorded_regrouped.jsonl",
    )
    provenance.validate_cli_output_contract(argparse.Namespace(**base), repo_root=tmp_path)

    escaped = dict(base, v1_csv="results/forged.csv")
    with pytest.raises(provenance.ProvenanceError, match="sources.csv"):
        provenance.validate_cli_output_contract(
            argparse.Namespace(**escaped), repo_root=tmp_path
        )

    (tmp_path / "results/provenance").rmdir()
    (tmp_path / "results/provenance").symlink_to("../outside", target_is_directory=True)
    with pytest.raises(provenance.ProvenanceError, match="symlink"):
        provenance.validate_cli_output_contract(argparse.Namespace(**base), repo_root=tmp_path)


def test_main_rejects_output_escape_before_historical_reconstruction(
    provenance, tmp_path: Path, monkeypatch
) -> None:
    for directory in (
        "data/source_pool",
        "data/source_pool_v2",
        "data/manifests",
        "results/provenance",
    ):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    called = False

    def must_not_reconstruct(_root):
        nonlocal called
        called = True
        raise AssertionError("expensive reconstruction must not start")

    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(provenance, "reconstruct_plans", must_not_reconstruct)
    assert provenance.main(["--report", "results/escaped"]) == 2
    assert called is False


def test_regular_file_snapshot_detects_path_retarget_during_single_fd_read(
    tmp_path: Path, monkeypatch
) -> None:
    import deep_anc.data.holdout_contract as contract

    target = tmp_path / "evidence.json"
    replacement = tmp_path / "replacement.json"
    target.write_bytes(b'{"generation":1}\n')
    replacement.write_bytes(b'{"generation":2}\n')
    real_read = contract.os.read
    replaced = False

    def retarget_after_open(descriptor, count):
        nonlocal replaced
        if not replaced:
            replaced = True
            os.replace(replacement, target)
        return real_read(descriptor, count)

    monkeypatch.setattr(contract.os, "read", retarget_after_open)
    with pytest.raises(contract.HoldoutContractError, match="retarget"):
        contract.read_regular_file_snapshot(
            target, root=tmp_path, label="fixture evidence"
        )


def test_shared_original_clip_across_recorded_splits_is_a_failure(
    provenance, tmp_path: Path
) -> None:
    left = _plan(provenance, index=0, clips=("same.wav", "left.wav"))
    right = _plan(provenance, index=1, clips=("same.wav", "right.wav"))
    sessions = [
        {"session_id": "train-s", "source_family": "machine", "source_wav": left.wav_path},
        {"session_id": "test-s", "source_family": "machine", "source_wav": right.wav_path},
    ]
    manifest = tmp_path / "recorded.jsonl"
    manifest.write_text(
        json.dumps({"session_id": "train-s", "split": "train"})
        + "\n"
        + json.dumps({"session_id": "test-s", "split": "test"})
        + "\n",
        encoding="utf-8",
    )
    audit = provenance.audit_recorded_clip_split_leak(
        sessions, {left.wav_path: left, right.wav_path: right}, manifest
    )
    assert audit["status"] == "FAIL"
    assert audit["cross_split_clip_count"] == 1
    assert audit["cross_split_clips"][0]["clip"] == "same.wav"


def test_legacy_split_is_diagnostic_but_canonical_split_and_corpus_are_required(
    provenance,
) -> None:
    recovered = {
        "active_holdout": {"status": "PASS"},
        "legacy_recorded_clip_split": {"status": "FAIL"},
        "lineage_components": {"status": "READY"},
        "regrouped_manifest": {"status": "PASS"},
        "recorded_clip_split": {"status": "PASS"},
        "synthetic_corpus": {"status": "PASS"},
    }
    assert provenance.downstream_is_blocked(recovered) is False

    recovered["synthetic_corpus"] = {"status": "BLOCKED"}
    assert provenance.downstream_is_blocked(recovered) is True


def test_fma_regroup_is_blocked_without_tracks_and_components_shared_artist(
    provenance, tmp_path: Path
) -> None:
    first = _plan(
        provenance,
        index=0,
        clips=("000001.mp3",),
        wav="data/source_pool/music/music_000.wav",
    )
    second = _plan(
        provenance,
        index=1,
        clips=("000002.mp3",),
        wav="data/source_pool/music/music_001.wav",
    )
    # fixture plan의 family만 music으로 바꾼다.
    first = provenance.RowPlan(**{**first.__dict__, "source_family": "music", "group_id": "music-x"})
    second = provenance.RowPlan(**{**second.__dict__, "source_family": "music", "group_id": "music-y"})
    sessions = [
        {"session_id": "m0", "source_family": "music", "source_wav": first.wav_path},
        {"session_id": "m1", "source_family": "music", "source_wav": second.wav_path},
    ]
    missing = provenance.audit_fma_regroup_gate(
        sessions, {first.wav_path: first, second.wav_path: second}, tmp_path / "missing.csv"
    )
    assert missing["status"] == "BLOCKED"

    tracks = tmp_path / "tracks.csv"
    tracks.write_text(
        ",artist,album\n"
        "track_id,id,id\n"
        "1,artist-10,album-20\n"
        "2,artist-10,album-21\n",
        encoding="utf-8",
    )
    ready = provenance.audit_fma_regroup_gate(
        sessions, {first.wav_path: first, second.wav_path: second}, tracks
    )
    assert ready["status"] == "READY"
    assert ready["component_count"] == 1
    assert set(next(iter(ready["components"].values()))) == {"m0", "m1"}


def test_all_lineage_rules_feed_one_transitive_component_graph(
    provenance, tmp_path: Path
) -> None:
    tracks = tmp_path / "tracks.csv"
    tracks.write_text(
        ",artist,album\n"
        "track_id,id,id\n"
        "1,artist-10,album-20\n"
        "2,artist-10,album-21\n",
        encoding="utf-8",
    )
    chapters = tmp_path / "CHAPTERS.TXT"
    chapters.write_text(
        "20 | 10 | 1.0 | train | project | 500\n"
        "21 | 10 | 1.0 | train | project | 501\n",
        encoding="utf-8",
    )
    esc50 = tmp_path / "esc50.csv"
    esc50.write_text(
        "filename,fold,target,category,esc10,src_file,take\n"
        "shared.wav,1,1,fixture,True,source-1,1\n"
        "other.wav,1,1,fixture,True,source-2,1\n",
        encoding="utf-8",
    )
    plans = [
        _plan(provenance, family="environment", index=0, clips=("shared.wav",)),
        _plan(provenance, family="environment", index=1, clips=("shared.wav", "other.wav")),
        _plan(provenance, family="speech", index=0, clips=("10-20-0001.flac",)),
        _plan(provenance, family="speech", index=1, clips=("10-21-0002.flac",)),
        _plan(provenance, family="music", index=0, clips=("000001.mp3",)),
        _plan(provenance, family="music", index=1, clips=("000002.mp3",)),
    ]
    sessions = [
        {
            "session_id": f"s{index}",
            "source_family": plan.source_family,
            "source_wav": plan.wav_path,
        }
        for index, plan in enumerate(plans)
    ]
    result = provenance.build_lineage_component_plan(
        sessions,
        {plan.wav_path: plan for plan in plans},
        tracks,
        chapters,
        esc50,
        repo_root=tmp_path,
    )
    assert result["status"] == "READY"
    assert result["component_count_by_family"] == {
        "environment": 1,
        "music": 1,
        "speech": 1,
    }
    for left, right in (("s0", "s1"), ("s2", "s3"), ("s4", "s5")):
        assert result["session_to_component"][left] == result["session_to_component"][right]


def test_regrouped_manifest_is_deterministic_component_stratified_and_new_file_only(
    provenance, tmp_path: Path
) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    source = manifest_dir / "recorded_train.jsonl"
    output = manifest_dir / "recorded_regrouped.jsonl"
    sessions = []
    rows = []
    components = {}
    session_to_component = {}
    lineage = {}
    for index in range(9):
        session_id = f"s{index}"
        component = f"machine-lineage-{index:02d}"
        session_dir = tmp_path / "recorded" / session_id
        sessions.append(
            {
                "session_id": session_id,
                "source_family": "machine",
                "source_wav": f"pool/{index}.wav",
                "session_dir": str(session_dir),
            }
        )
        rows.append(
            {
                "path": f"../recorded/{session_id}",
                "path_base": "manifest",
                "duration_s": 70.0,
                "sample_rate": 48000,
                "channels": 2,
                "tag": "recorded",
                "session_id": session_id,
                "group_id": f"old-{index}",
                "source_family": "machine",
                "metadata_inferred": [],
                "split": "train",
            }
        )
        components[component] = [session_id]
        session_to_component[session_id] = component
        lineage[session_id] = [{"kind": "clip", "id": f"clip-{index}"}]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    plan = {
        "status": "READY",
        "components": components,
        "session_to_component": session_to_component,
        "lineage_by_session": lineage,
    }
    first, audit = provenance.build_regrouped_manifest_entries(
        plan, sessions, input_manifest=source, output_manifest=output, seed=7
    )
    second, _ = provenance.build_regrouped_manifest_entries(
        plan, sessions, input_manifest=source, output_manifest=output, seed=7
    )
    assert first == second
    assert audit["groups_by_family_split"]["machine"] == {
        "train": 1,
        "val": 4,
        "test": 4,
    }
    written = provenance.write_regrouped_manifest(
        plan, sessions, input_manifest=source, output_manifest=output, seed=7
    )
    assert written["status"] == "PASS"
    assert output.is_file()
    assert source.read_text(encoding="utf-8") == "".join(
        json.dumps(row) + "\n" for row in rows
    )
    repeated = provenance.write_regrouped_manifest(
        plan, sessions, input_manifest=source, output_manifest=output, seed=7
    )
    assert repeated["status"] == "PASS"
    assert repeated["changed"] is False

    output.write_bytes(output.read_bytes() + b"{}\n")
    with pytest.raises(provenance.ProvenanceError, match="bytes"):
        provenance.write_regrouped_manifest(
            plan, sessions, input_manifest=source, output_manifest=output, seed=7
        )


def test_missing_fma_gate_never_writes_regrouped_manifest(
    provenance, tmp_path: Path
) -> None:
    output = tmp_path / "recorded_regrouped.jsonl"
    with pytest.raises(provenance.ProvenanceError, match="manifest를 쓰지 않습니다"):
        provenance.write_regrouped_manifest(
            {"status": "BLOCKED"},
            [],
            input_manifest=tmp_path / "recorded_train.jsonl",
            output_manifest=output,
        )
    assert not output.exists()


def test_equal_partition_identifier_is_explicitly_non_authoritative(tmp_path: Path) -> None:
    identify = _load("identify_pool_clips", "scripts/data/identify_pool_clips.py")
    assert identify.AUTHORITATIVE_PROVENANCE is False
    common = [
        "--pool",
        str(tmp_path),
        "--manifest",
        str(tmp_path / "manifest.jsonl"),
        "--out",
        str(tmp_path / "out.diagnostic.json"),
    ]
    with pytest.raises(SystemExit) as stopped:
        identify.main(common)
    assert stopped.value.code == 2

    common[common.index(str(tmp_path / "out.diagnostic.json"))] = str(
        tmp_path / "sources.csv"
    )
    with pytest.raises(SystemExit) as stopped:
        identify.main(["--diagnostic-only", *common])
    assert stopped.value.code == 2


def test_real_v1_v2_selection_matches_every_shipped_csv_prefix(provenance) -> None:
    """raw corpus가 있는 개발 머신에서는 역사적 선택 160행을 실제 CSV와 대조한다."""

    required = [
        REPO_ROOT / "data/source_pool/sources.csv",
        REPO_ROOT / "data/source_pool_v2/sources.csv",
        REPO_ROOT / "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv",
        REPO_ROOT / "data/raw/speech",
        REPO_ROOT / "data/raw/music",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("historical source-pool raw corpus가 없는 환경")
    plans, evidence = provenance.reconstruct_plans(REPO_ROOT)
    assert evidence["counts"]["v1_placements"] == 983
    assert evidence["counts"]["v2_placements"] == 416
    for name, path in (
        ("source_pool", required[0]),
        ("source_pool_v2", required[1]),
    ):
        audit, *_ = provenance.audit_csv_prefix(path, plans[name])
        assert audit["status"] == "PASS", audit["issues"]
