from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from deep_anc.data.broadband_population_availability_v3 import (
    AvailabilityInputV3,
    PopulationAvailabilityReportV3,
    audit_population_v3_availability,
)
from deep_anc.data.broadband_population_contract_v3 import (
    CausalPrimaryOperatorV3,
    LocalFileReferenceV3,
    PopulationCoverageContractV3,
)
from deep_anc.dsp.control_band_contract import BroadbandFullOctaveContractV3


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_audio(path: Path, *, sample_rate: int, frequency: float = 500.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(4096, sample_rate // 4)
    time = np.arange(frames, dtype=np.float64) / sample_rate
    signal = np.asarray(0.05 * np.sin(2.0 * np.pi * frequency * time), dtype=np.float32)
    sf.write(path, signal, sample_rate, subtype="PCM_16")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_source_pool(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_family",
        "group_id",
        "path",
        "sample_rate_hz",
        "clips",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_actual_files_metadata_lineage_and_unreferenced_scan_are_mapping_only(
    tmp_path: Path,
) -> None:
    root = tmp_path
    chapters = root / "data/raw/speech/LibriSpeech/CHAPTERS.TXT"
    chapters.parent.mkdir(parents=True)
    chapters.write_text("2 | 1 | chapter | 0.25 | subset | 9\n", encoding="utf-8")
    referenced = root / "data/raw/speech/LibriSpeech/dev-clean/1/2/1-2-3.flac"
    extra = root / "data/raw/speech/LibriSpeech/dev-clean/1/2/1-2-4.flac"
    composite = root / "data/source_pool/speech/speech_000.wav"
    _write_audio(referenced, sample_rate=48_000)
    _write_audio(extra, sample_rate=48_000, frequency=700.0)
    _write_audio(composite, sample_rate=48_000, frequency=900.0)

    public = root / "data/manifests/speech.jsonl"
    _write_jsonl(
        public,
        [
            {
                "path": referenced.relative_to(root).as_posix(),
                "sample_rate": 48_000,
                "tag": "speech",
                "split": "train",
                "sha256": _sha(referenced),
            }
        ],
    )
    pool = root / "data/source_pool/sources.csv"
    _write_source_pool(
        pool,
        [
            {
                "source_family": "speech",
                "group_id": "speech-reader-1",
                "path": composite.relative_to(root).as_posix(),
                "sample_rate_hz": 48_000,
                "clips": json.dumps([referenced.name]),
            }
        ],
    )

    report = audit_population_v3_availability(
        repository_root=root,
        inputs=(
            AvailabilityInputV3(kind="public_jsonl", path="data/manifests/speech.jsonl"),
            AvailabilityInputV3(kind="source_pool_csv", path="data/source_pool/sources.csv"),
            AvailabilityInputV3(
                kind="unreferenced_audio_tree",
                path="data/raw/speech",
                source_family="speech",
            ),
        ),
    )

    assert report.status == "BLOCKED"
    assert report.authority is None
    assert report.canonical_population_manifest_issued is False
    assert report.density_recomputed is False
    assert report.causal_primary.status == "MISSING"
    assert report.control_band_contract_sha256 == BroadbandFullOctaveContractV3.canonical().digest()
    assert report.population_contract_sha256 == PopulationCoverageContractV3.canonical().digest()
    assert report.summary.manifest_entries_total == 2
    assert report.summary.candidates_reported == 3
    assert report.summary.files_present == 3
    assert report.summary.files_missing == 0
    assert report.summary.decoder_probe_pass == 3
    assert report.summary.direct_native_present == 2
    assert report.summary.direct_native_full_target_nyquist == 2
    assert report.summary.canonical_population_candidates == 0
    assert report.summary.canonical_physical_component_band_deficit == 384
    assert report.summary.canonical_objective_component_octave_deficit == 336

    public_row, pool_row, scan_row = report.candidates
    assert public_row.availability_status == "MAPPING_CANDIDATE"
    assert pool_row.availability_status == "PARTIAL_MAPPING_CANDIDATE"
    assert scan_row.availability_status == "PARTIAL_MAPPING_CANDIDATE"
    assert public_row.actual_sha256 == _sha(referenced)
    assert public_row.declared_sha_matches is True
    assert public_row.actual_sample_rate_hz == 48_000
    assert public_row.full_target_native_nyquist is True
    assert public_row.qualification_limitations == ()
    assert set(public_row.semantic_lineage_keys) == {
        "gutenberg_book:9",
        "librivox_reader:1",
    }
    # public clip, source-pool composite와 unreferenced clip은 metadata relation으로 묶인다.
    assert len(
        {
            public_row.mapping_component_id,
            pool_row.mapping_component_id,
            scan_row.mapping_component_id,
        }
    ) == 1
    assert report.inputs[2].entries_seen == 2
    assert report.inputs[2].entries_emitted == 1
    assert report.metadata_evidence[0].role == "librispeech"
    assert report.metadata_evidence[0].status == "PASS"

    # 저장 후 다시 model_validate해도 evidence seal과 fail-closed schema가 유지된다.
    restored = PopulationAvailabilityReportV3.model_validate(
        report.model_dump(mode="json")
    )
    assert restored.evidence_sha256 == report.evidence_sha256


def test_missing_bad_decoder_sha_and_sample_rate_mismatches_are_exactly_counted(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "audio/valid.wav"
    bad = tmp_path / "audio/bad.wav"
    _write_audio(valid, sample_rate=16_000)
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not an audio file")
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "path": "audio/valid.wav",
                "sample_rate": 48_000,
                "source_family": "speech",
                "split": "train",
                "group_id": "valid-group",
                "sha256": "0" * 64,
            },
            {
                "path": "audio/missing.wav",
                "sample_rate": 48_000,
                "source_family": "music",
                "split": "val",
                "group_id": "missing-group",
            },
            {
                "path": "audio/bad.wav",
                "sample_rate": 48_000,
                "source_family": "machine",
                "split": "test",
                "group_id": "bad-group",
            },
        ],
    )

    report = audit_population_v3_availability(
        repository_root=tmp_path,
        inputs=(AvailabilityInputV3(kind="public_jsonl", path="manifest.jsonl"),),
    )

    assert report.summary.candidates_reported == 3
    assert report.summary.files_present == 2
    assert report.summary.files_missing == 1
    assert report.summary.unsafe_paths == 0
    assert report.summary.decoder_probe_pass == 1
    assert report.summary.decoder_probe_fail == 1
    assert report.summary.declared_sha_present == 1
    assert report.summary.declared_sha_match == 0
    assert report.summary.declared_sha_mismatch == 1
    valid_row, missing_row, bad_row = report.candidates
    assert valid_row.availability_status == "PARTIAL_MAPPING_CANDIDATE"
    assert valid_row.actual_header_nyquist_hz == 8_000.0
    assert valid_row.native_nyquist_verified is True
    assert valid_row.native_nyquist_hz == 8_000.0
    assert valid_row.full_target_native_nyquist is False
    assert valid_row.qualification_limitations == (
        "native_nyquist_partial_objective_octave_coverage",
        "native_nyquist_partial_physical_band_coverage",
    )
    assert "native_nyquist_does_not_cover_full_v3_target" not in valid_row.blockers
    assert valid_row.declared_sample_rate_matches is False
    assert valid_row.declared_sha_matches is False
    assert missing_row.availability_status == "UNAVAILABLE"
    assert missing_row.file_status == "MISSING"
    assert bad_row.availability_status == "UNAVAILABLE"
    assert bad_row.file_status == "PRESENT"
    assert bad_row.decoder_probe_status == "FAIL"


def test_partial_native_nyquist_contributes_only_to_covered_mapping_bands(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "speech/partial.flac"
    _write_audio(audio, sample_rate=16_000)
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "path": "speech/partial.flac",
                "sample_rate": 16_000,
                "source_family": "speech",
                "split": "train",
                "group_id": "speech-partial-native-a",
                "sha256": _sha(audio),
            }
        ],
    )

    report = audit_population_v3_availability(
        repository_root=tmp_path,
        inputs=(AvailabilityInputV3(kind="public_jsonl", path="manifest.jsonl"),),
    )

    row = report.candidates[0]
    assert row.availability_status == "MAPPING_CANDIDATE"
    assert row.full_target_native_nyquist is False
    assert row.qualification_limitations == (
        "native_nyquist_partial_objective_octave_coverage",
        "native_nyquist_partial_physical_band_coverage",
    )
    assert not any("full_v3_target" in blocker for blocker in row.blockers)
    control = BroadbandFullOctaveContractV3.canonical()
    expected_physical = tuple(upper <= 8_000.0 for _, upper in control.physical_identification_subbands_hz)
    expected_objective = tuple(upper <= 8_000.0 for _, upper in control.equal_weight_octave_objective_bands_hz)
    assert row.native_physical_nyquist_coverage == expected_physical
    assert row.native_objective_octave_nyquist_coverage == expected_objective
    cell = next(
        item
        for item in report.cells
        if item.split == "train" and item.source_family == "speech"
    )
    assert cell.mapping_native_components_per_physical_band == tuple(
        int(value) for value in expected_physical
    )
    assert cell.mapping_native_components_per_objective_octave == tuple(
        int(value) for value in expected_objective
    )
    assert cell.canonical_qualified_components_per_physical_band == (0,) * 8


def test_recorded_jsonl_resolves_source_wav_but_marks_composite_origin(
    tmp_path: Path,
) -> None:
    source = tmp_path / "data/recorded/session-a/source.wav"
    _write_audio(source, sample_rate=48_000)
    manifest = tmp_path / "data/manifests/recorded.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "path": "../recorded/session-a",
                "path_base": "manifest",
                "sample_rate": 48_000,
                "source_family": "machine",
                "split": "test",
                "group_id": "machine-lineage-a",
                "source_pool_group_id": "machine-a",
                "lineage_schema": "test/v2",
            }
        ],
    )
    report = audit_population_v3_availability(
        repository_root=tmp_path,
        inputs=(
            AvailabilityInputV3(
                kind="recorded_jsonl",
                path="data/manifests/recorded.jsonl",
            ),
        ),
    )

    row = report.candidates[0]
    assert row.resolved_audio_path == "data/recorded/session-a/source.wav"
    assert row.origin_role == "recorded_playback_composite"
    assert row.actual_header_nyquist_hz == 24_000.0
    assert row.native_nyquist_verified is False
    assert row.native_nyquist_hz is None
    assert row.full_target_native_nyquist is False
    assert row.availability_status == "MAPPING_CANDIDATE"
    assert "immutable_native_origin_not_bound" in row.blockers
    assert row.mapping_component_authoritative is False
    assert report.cells[-1].split == "test"


def test_optional_causal_p_payload_can_only_be_structural(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    _write_audio(audio, sample_rate=48_000)
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "path": "audio.wav",
                "sample_rate": 48_000,
                "source_family": "environment",
                "split": "train",
                "group_id": "environment-a",
            }
        ],
    )
    fir = np.asarray([1.0, 0.5], dtype="<f4")
    fir_path = tmp_path / "plant/p.f32"
    fir_path.parent.mkdir(parents=True)
    fir_path.write_bytes(fir.tobytes())
    control = BroadbandFullOctaveContractV3.canonical()
    operator = CausalPrimaryOperatorV3(
        control_band_contract_sha256=control.digest(),
        fir_file=LocalFileReferenceV3(
            path="plant/p.f32",
            size_bytes=fir_path.stat().st_size,
            sha256=_sha(fir_path),
        ),
        delay_samples=3,
        verified_lower_hz=80.0,
        verified_upper_hz=11_400.0,
        operator_receipt_sha256=hashlib.sha256(b"operator receipt").hexdigest(),
    )
    operator_path = tmp_path / "operator.json"
    operator_path.write_text(
        json.dumps(operator.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )

    report = audit_population_v3_availability(
        repository_root=tmp_path,
        inputs=(AvailabilityInputV3(kind="public_jsonl", path="manifest.jsonl"),),
        causal_p_authority_path="operator.json",
    )

    assert report.causal_primary.status == "STRUCTURAL_ONLY"
    assert report.causal_primary.fullband_causal_p_authority is False
    assert report.causal_primary.authority is None
    assert report.causal_primary.operator == operator
    assert report.density_recomputed is False
    assert report.status == "BLOCKED"


def test_legacy_v2_payload_is_never_promoted(tmp_path: Path) -> None:
    manifest = tmp_path / "legacy.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "schema_version": "broadband_source_manifest_v2",
                "path": "missing.wav",
                "sample_rate": 48_000,
                "source_family": "speech",
                "split": "train",
                "lineage_component_id": "legacy-component",
            }
        ],
    )
    report = audit_population_v3_availability(
        repository_root=tmp_path,
        inputs=(AvailabilityInputV3(kind="public_jsonl", path="legacy.jsonl"),),
    )

    assert report.legacy_v1_v2_automatic_promotion is False
    assert report.candidates[0].legacy_automatic_promotion is False
    assert report.candidates[0].mapping_only is True
    assert report.canonical_population_manifest_issued is False


def test_missing_input_and_audio_tree_are_reported_without_fabricating_rows(
    tmp_path: Path,
) -> None:
    report = audit_population_v3_availability(
        repository_root=tmp_path,
        inputs=(
            AvailabilityInputV3(kind="public_jsonl", path="missing.jsonl"),
            AvailabilityInputV3(
                kind="unreferenced_audio_tree",
                path="missing-machine-tree",
                source_family="machine",
            ),
        ),
    )

    assert [item.status for item in report.inputs] == ["MISSING", "MISSING"]
    assert report.summary.manifest_entries_total == 0
    assert report.summary.candidates_reported == 0
    assert report.summary.files_missing == 0
    assert len([item for item in report.blockers if "is MISSING" in item]) == 2


def test_cli_writes_no_replace_report_and_returns_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.data.audit_broadband_population_v3_availability as cli

    audio = tmp_path / "audio.wav"
    _write_audio(audio, sample_rate=48_000)
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "path": "audio.wav",
                "sample_rate": 48_000,
                "source_family": "speech",
                "split": "train",
                "group_id": "speech-a",
            }
        ],
    )
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    result = cli.main(
        [
            "--input",
            "public_jsonl=manifest.jsonl",
            "--output",
            "result.json",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert result == 2
    assert output["status"] == "BLOCKED"
    assert output["summary"]["files_present"] == 1
    assert (tmp_path / "result.json").is_file()
    with pytest.raises(FileExistsError):
        cli.main(
            [
                "--input",
                "public_jsonl=manifest.jsonl",
                "--output",
                "result.json",
            ]
        )
