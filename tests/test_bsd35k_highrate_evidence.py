from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from deep_anc.data import bsd35k_highrate_evidence as highrate
from deep_anc.data.decoder_audit import audit_audio_paths, write_audit_report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - fixture mirrors official MD5 contract.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection() -> dict:
    entries = []
    sound_id = 1
    for split in ("train", "val", "test"):
        for uploader_index in range(4):
            uploader = f"uploader-{split}-{uploader_index}"
            entries.append(
                {
                    "sound_id": sound_id,
                    "split": split,
                    "uploader": uploader,
                    "lineage_group": f"bsd35k_uploader:{uploader}",
                    "archive_member": f"audio/{sound_id}.wav",
                }
            )
            sound_id += 1
    return {"entries": entries, "selection_plan_sha256": "a" * 64}


def _write_fixture(
    root: Path, *, sample_rate: int = 44_100
) -> tuple[Path, Path, Path, Path, Path]:
    selection = _selection()
    plan_path = root / "data/provenance/selection.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps(selection, sort_keys=True), encoding="utf-8")
    metadata_path = root / "data/raw/bsd35k/BSD35k-CS_metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text("fixture metadata only\n", encoding="utf-8")
    raw_root = root / "data/raw/bsd35k_fx_m"
    archive_path = root / "data/raw/bsd35k_audio.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260829)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for entry in selection["entries"]:
            wav = raw_root / str(entry["archive_member"])
            wav.parent.mkdir(parents=True, exist_ok=True)
            # White-noise fixture는 natural-source PASS가 아니라 7-band PSD/lineage
            # validator의 최소 deterministic input이다.
            signal = rng.normal(0.0, 0.03, 16_384)
            sf.write(wav, signal, sample_rate, subtype="PCM_16", format="WAV")
            archive.write(wav, arcname=str(entry["archive_member"]))
    audit_path = root / "results/provenance/decoder_audit.json"
    report = audit_audio_paths(
        sorted(raw_root.rglob("*.wav")),
        root=root,
        root_label=".",
    )
    write_audit_report(report, audit_path)
    return plan_path, metadata_path, raw_root, archive_path, audit_path


@pytest.fixture()
def patched_official_bsd_contract(monkeypatch: pytest.MonkeyPatch):
    # Official CSV-derived selection validator는 BSD35k의 1,323 rows를 의도적으로
    # 요구한다. 이 focused test는 그 validator 자체가 아니라 high-rate raw/PSD gate를
    # 검사하므로 small deterministic selection만 fixture로 허용한다.
    monkeypatch.setattr(highrate.bsd, "validate_bsd35k_machine_selection", lambda _: None)
    monkeypatch.setattr(
        highrate.bsd,
        "verify_bsd35k_machine_selection_against_metadata",
        lambda _plan, _metadata: None,
    )


def test_highrate_source_evidence_binds_archive_lineage_decoder_and_native_psd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_official_bsd_contract
) -> None:
    plan, metadata, raw_root, archive, decoder_audit = _write_fixture(tmp_path)
    monkeypatch.setattr(highrate.bsd, "OFFICIAL_AUDIO_ZIP_SIZE", archive.stat().st_size)
    monkeypatch.setattr(highrate.bsd, "OFFICIAL_AUDIO_ZIP_MD5", _md5_file(archive))

    evidence = highrate.build_bsd35k_highrate_machine_evidence(
        repository_root=tmp_path,
        selection_plan_path=plan,
        metadata_csv_path=metadata,
        selected_raw_root=raw_root,
        audio_archive_path=archive,
        decoder_audit_path=decoder_audit,
    )

    assert evidence["status"] == "PASS"
    assert evidence["authority"]["source_prepopulation_eligible"] is True
    assert evidence["authority"]["canonical_training_eligible"] is False
    assert len(evidence["entries"]) == 12
    assert all(row["passed"] for row in evidence["coverage"])

    output = tmp_path / "results/provenance/highrate.json"
    path, file_sha = highrate.write_bsd35k_highrate_machine_evidence_exclusive(output, evidence)
    result = highrate.load_and_validate_bsd35k_highrate_machine_evidence(
        path,
        repository_root=tmp_path,
        expected_sha256=file_sha,
    )
    assert result["status"] == "PASS"
    assert result["selected_file_count"] == 12
    assert result["canonical_training_eligible"] is False


def test_highrate_evidence_rejects_native_16khz_before_psd_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_official_bsd_contract
) -> None:
    plan, metadata, raw_root, archive, decoder_audit = _write_fixture(tmp_path, sample_rate=16_000)
    monkeypatch.setattr(highrate.bsd, "OFFICIAL_AUDIO_ZIP_SIZE", archive.stat().st_size)
    monkeypatch.setattr(highrate.bsd, "OFFICIAL_AUDIO_ZIP_MD5", _md5_file(archive))

    with pytest.raises(highrate.BSD35kHighRateEvidenceError, match="sample rate"):
        highrate.build_bsd35k_highrate_machine_evidence(
            repository_root=tmp_path,
            selection_plan_path=plan,
            metadata_csv_path=metadata,
            selected_raw_root=raw_root,
            audio_archive_path=archive,
            decoder_audit_path=decoder_audit,
        )


def test_revalidation_detects_tampered_local_selected_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_official_bsd_contract
) -> None:
    plan, metadata, raw_root, archive, decoder_audit = _write_fixture(tmp_path)
    monkeypatch.setattr(highrate.bsd, "OFFICIAL_AUDIO_ZIP_SIZE", archive.stat().st_size)
    monkeypatch.setattr(highrate.bsd, "OFFICIAL_AUDIO_ZIP_MD5", _md5_file(archive))
    evidence = highrate.build_bsd35k_highrate_machine_evidence(
        repository_root=tmp_path,
        selection_plan_path=plan,
        metadata_csv_path=metadata,
        selected_raw_root=raw_root,
        audio_archive_path=archive,
        decoder_audit_path=decoder_audit,
    )
    output = tmp_path / "results/provenance/highrate.json"
    path, _file_sha = highrate.write_bsd35k_highrate_machine_evidence_exclusive(output, evidence)
    target = raw_root / "audio/1.wav"
    target.write_bytes(target.read_bytes() + b"tamper")

    with pytest.raises(highrate.BSD35kHighRateEvidenceError, match="raw"):
        highrate.load_and_validate_bsd35k_highrate_machine_evidence(
            path,
            repository_root=tmp_path,
            expected_sha256=_sha256_file(path),
        )


def test_revalidation_requires_same_official_metadata_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_official_bsd_contract
) -> None:
    plan, metadata, raw_root, archive, decoder_audit = _write_fixture(tmp_path)
    monkeypatch.setattr(highrate.bsd, "OFFICIAL_AUDIO_ZIP_SIZE", archive.stat().st_size)
    monkeypatch.setattr(highrate.bsd, "OFFICIAL_AUDIO_ZIP_MD5", _md5_file(archive))
    evidence = highrate.build_bsd35k_highrate_machine_evidence(
        repository_root=tmp_path,
        selection_plan_path=plan,
        metadata_csv_path=metadata,
        selected_raw_root=raw_root,
        audio_archive_path=archive,
        decoder_audit_path=decoder_audit,
    )
    output = tmp_path / "results/provenance/highrate.json"
    path, _file_sha = highrate.write_bsd35k_highrate_machine_evidence_exclusive(output, evidence)
    metadata.write_text("different fixture metadata\n", encoding="utf-8")

    with pytest.raises(highrate.BSD35kHighRateEvidenceError, match="metadata"):
        highrate.load_and_validate_bsd35k_highrate_machine_evidence(
            path,
            repository_root=tmp_path,
            expected_sha256=_sha256_file(path),
        )
