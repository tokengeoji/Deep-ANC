"""public raw decoder eligibility audit의 독립 계약 테스트."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import warnings

import numpy as np
import pytest
import soundfile as sf

import deep_anc.data.decoder_audit as decoder_audit
from deep_anc.data.decoder_audit import (
    DEFAULT_SEQUENTIAL_CHUNK_FRAMES,
    audit_audio_tree,
    canonical_json_bytes,
    canonical_json_sha256,
    validate_audit_report_self_digest,
    write_audit_report,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/data/audit_decoder_eligibility.py"
REUSE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/data/verify_decoder_audit_reuse.py"
FS = 48_000


def _write_audio(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, FS, subtype="FLOAT")


def _load_script():
    spec = importlib.util.spec_from_file_location("_decoder_audit_cli_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_reuse_script():
    spec = importlib.util.spec_from_file_location("_decoder_audit_reuse_cli_probe", REUSE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_full_decode_inventory_is_ordered_deterministic_and_raw_is_unchanged(tmp_path):
    root = tmp_path / "raw"
    samples = np.arange(FS, dtype=np.float32) / FS
    _write_audio(root / "z" / "tone.wav", 0.05 * np.sin(2 * np.pi * 440 * samples))
    _write_audio(root / "a" / "tone.wav", 0.04 * np.sin(2 * np.pi * 880 * samples))
    before = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*.wav")
    }

    first = audit_audio_tree(root, root_label="fixture-raw")
    second = audit_audio_tree(root, root_label="fixture-raw")

    assert first == second
    assert first["schema_version"] == 1
    assert first["status"] == "complete"
    assert first["summary"] == {
        "candidate_count": 2,
        "accepted_count": 2,
        "rejected_count": 0,
    }
    assert first["inventory_sha256"] == canonical_json_sha256(first["inventory"])
    accepted = [
        {
            "relative_path": item["relative_path"],
            "content_sha256": item["content_sha256"],
            "content_size": item["content_size"],
        }
        for item in first["inventory"]
        if item["decision"] == "accept"
    ]
    assert first["accepted_inventory_sha256"] == canonical_json_sha256(accepted)
    digest_basis = dict(first)
    assert first["audit_sha256"] == canonical_json_sha256(
        {key: value for key, value in digest_basis.items() if key != "audit_sha256"}
    )
    assert validate_audit_report_self_digest(first) == first["audit_sha256"]
    assert [item["relative_path"] for item in first["inventory"]] == [
        "a/tone.wav",
        "z/tone.wav",
    ]
    for item in first["inventory"]:
        assert item["decision"] == "accept"
        assert item["content_sha256"]
        assert item["content_size"] > 0
        assert item["header"]["sample_rate"] == FS
        assert [scan["chunk_frames"] for scan in item["scan"]["sequential"]] == list(
            DEFAULT_SEQUENTIAL_CHUNK_FRAMES
        )
        assert all(scan["frames_read"] == FS for scan in item["scan"]["sequential"])
        assert item["scan"]["segment_grid"]
    after = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*.wav")
    }
    assert after == before


def test_silent_hot_and_broken_candidates_are_rejected_with_inventory_evidence(tmp_path):
    root = tmp_path / "raw"
    _write_audio(root / "good.wav", np.full(FS, 0.05, dtype=np.float32))
    _write_audio(root / "silent.wav", np.zeros(FS, dtype=np.float32))
    _write_audio(root / "hot.wav", np.full(FS, 2.5, dtype=np.float32))
    (root / "broken.wav").write_bytes(b"not a wav file")

    report = audit_audio_tree(root, root_label="fixture-raw")
    by_path = {item["relative_path"]: item for item in report["inventory"]}

    assert by_path["good.wav"]["decision"] == "accept"
    assert by_path["silent.wav"]["decision"] == "reject"
    assert any(
        finding["code"] == "pcm_rms_below_limit"
        for finding in by_path["silent.wav"]["findings"]
    )
    assert by_path["hot.wav"]["decision"] == "reject"
    assert any(
        finding["code"] == "pcm_peak_exceeds_limit"
        for finding in by_path["hot.wav"]["findings"]
    )
    assert by_path["broken.wav"]["decision"] == "reject"
    assert any(
        finding["code"] in {"header_decode_error", "decode_error"}
        for finding in by_path["broken.wav"]["findings"]
    )
    assert report["summary"] == {
        "candidate_count": 4,
        "accepted_count": 1,
        "rejected_count": 3,
    }


def test_python_decoder_warning_is_a_hard_rejection(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    _write_audio(root / "clip.wav", np.full(FS, 0.05, dtype=np.float32))
    original_info = decoder_audit.sf.info

    def warning_info(*args, **kwargs):
        warnings.warn("fixture decoder warning", RuntimeWarning)
        return original_info(*args, **kwargs)

    monkeypatch.setattr(decoder_audit.sf, "info", warning_info)
    report = audit_audio_tree(root, root_label="fixture-raw")

    item = report["inventory"][0]
    assert item["decision"] == "reject"
    assert any(
        finding["code"] == "decoder_warning" and finding["phase"] == "header"
        for finding in item["findings"]
    )


def test_canonical_writer_and_cli_dry_run_do_not_write_raw_or_report(tmp_path, monkeypatch, capsys):
    root = tmp_path / "raw"
    _write_audio(root / "clip.wav", np.full(FS, 0.02, dtype=np.float32))
    report = audit_audio_tree(root, root_label="fixture-raw")
    report_path = tmp_path / "result" / "audit.json"
    write_audit_report(report, report_path)
    assert report_path.read_bytes() == canonical_json_bytes(report) + b"\n"
    assert json.loads(report_path.read_text(encoding="utf-8"))["audit_sha256"] == report[
        "audit_sha256"
    ]

    cli = _load_script()
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    dry_out = tmp_path / "must_not_exist.json"
    before = hashlib.sha256((root / "clip.wav").read_bytes()).hexdigest()
    result = cli.main(
        [
            "--root",
            ".",
            "--scan-root",
            "raw",
            "--out",
            dry_out.name,
            "--dry-run",
        ]
    )
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "dry_run"
    assert payload["relative_paths"] == ["raw/clip.wav"]
    assert not dry_out.exists()
    assert hashlib.sha256((root / "clip.wav").read_bytes()).hexdigest() == before


def test_canonical_audit_refuses_to_drop_required_full_sequential_scans(tmp_path):
    root = tmp_path / "raw"
    _write_audio(root / "clip.wav", np.full(FS, 0.05, dtype=np.float32))

    with pytest.raises(ValueError, match="65536.*262144"):
        audit_audio_tree(root, sequential_chunk_frames=(65_536,))


def test_reuse_cli_requires_external_file_and_semantic_sha_then_rehashes_raw(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "repo"
    samples = np.arange(FS, dtype=np.float32) / FS
    audio = root / "data/raw/fixture.wav"
    _write_audio(audio, 0.05 * np.sin(2 * np.pi * 440 * samples))
    report = audit_audio_tree(root, root_label=".")
    report_path = root / "results/provenance/decoder_audit.json"
    write_audit_report(report, report_path)
    report_file_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    raw_before = hashlib.sha256(audio.read_bytes()).hexdigest()
    report_before = hashlib.sha256(report_path.read_bytes()).hexdigest()

    cli = _load_reuse_script()
    monkeypatch.setattr(cli, "REPO_ROOT", root)
    result = cli.main(
        [
            "--root",
            ".",
            "--audit",
            "results/provenance/decoder_audit.json",
            "--scan-root",
            "data/raw",
            "--expected-audit-sha256",
            report["audit_sha256"],
            "--expected-file-sha256",
            report_file_sha,
            "--hash-workers",
            "2",
        ]
    )
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert payload["audit_sha256"] == report["audit_sha256"]
    assert payload["file_sha256"] == report_file_sha
    assert hashlib.sha256(audio.read_bytes()).hexdigest() == raw_before
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == report_before

    audio.write_bytes(audio.read_bytes() + b"\0")
    assert (
        cli.main(
            [
                "--root",
                ".",
                "--audit",
                "results/provenance/decoder_audit.json",
                "--scan-root",
                "data/raw",
                "--expected-audit-sha256",
                report["audit_sha256"],
                "--expected-file-sha256",
                report_file_sha,
                "--hash-workers",
                "2",
            ]
        )
        == 1
    )
    assert "raw inventory" in capsys.readouterr().err


def test_reuse_cli_rejects_out_of_range_parallel_hash_worker_count(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "repo"
    _write_audio(root / "data/raw/fixture.wav", np.full(FS, 0.05, dtype=np.float32))
    report = audit_audio_tree(root, root_label=".")
    report_path = root / "results/provenance/decoder_audit.json"
    write_audit_report(report, report_path)

    cli = _load_reuse_script()
    monkeypatch.setattr(cli, "REPO_ROOT", root)
    assert (
        cli.main(
            [
                "--root",
                ".",
                "--audit",
                "results/provenance/decoder_audit.json",
                "--scan-root",
                "data/raw",
                "--expected-audit-sha256",
                report["audit_sha256"],
                "--expected-file-sha256",
                hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "--hash-workers",
                "33",
            ]
        )
        == 1
    )
    assert "--hash-workers" in capsys.readouterr().err


def test_reuse_cli_rejects_invalid_report_self_digest_even_when_file_sha_is_anchored(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "repo"
    _write_audio(root / "data/raw/fixture.wav", np.full(FS, 0.05, dtype=np.float32))
    report = audit_audio_tree(root, root_label=".")
    report_path = root / "results/provenance/decoder_audit.json"
    write_audit_report(report, report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["audit_sha256"] = "0" * 64
    report_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    corrupt_file_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()

    cli = _load_reuse_script()
    monkeypatch.setattr(cli, "REPO_ROOT", root)
    assert (
        cli.main(
            [
                "--root",
                ".",
                "--audit",
                "results/provenance/decoder_audit.json",
                "--scan-root",
                "data/raw",
                "--expected-audit-sha256",
                report["audit_sha256"],
                "--expected-file-sha256",
                corrupt_file_sha,
            ]
        )
        == 1
    )
    assert "audit_sha256" in capsys.readouterr().err
