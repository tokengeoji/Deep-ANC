from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from deep_anc.data import manifest_contract, public_lineage
import deep_anc.data.recorded_demand_selection as demand


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pcm16_wav(values: np.ndarray) -> bytes:
    quantized = np.rint(
        np.clip(np.asarray(values, dtype=np.float64), -1.0, 1.0) * 32767.0
    ).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(demand.DEMAND_SAMPLE_RATE)
        handle.writeframes(quantized.tobytes())
    return output.getvalue()


def _group(lineage_key: str, contents: list[str]) -> str:
    basis = {
        "lineage_keys": [lineage_key],
        "content_sha256": sorted(contents),
    }
    return "public-lineage-" + public_lineage.canonical_json_sha256(basis)


def _generation_bytes() -> bytes:
    payload = {
        "schema_version": demand.DEMAND_MANIFEST_SCHEMA_VERSION,
        "training_eligible": True,
        "fixture": "pre-exclusion-demand-generation",
        "created_at": "2026-08-30T00:00:00Z",
    }
    basis = {
        key: value
        for key, value in payload.items()
        if key not in {"build_id", "created_at"}
    }
    payload["build_id"] = demand._canonical_json_sha256(basis)
    return (json.dumps(payload, sort_keys=True) + "\n").encode()


def _fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    source = root / demand.DEMAND_SOURCE_ORIGIN
    source.parent.mkdir(parents=True)
    rng = np.random.default_rng(20260830)
    sf.write(
        source,
        rng.normal(0.0, 0.02, 31 * 48_000).astype(np.float32),
        48_000,
        subtype="PCM_16",
    )
    source_raw = source.read_bytes()
    source_sha = _sha(source_raw)

    primary = root / "assets/measured/strict.npz"
    primary.parent.mkdir(parents=True)
    np.savez(
        primary,
        fir=np.asarray([1.0], dtype=np.float64),
        sample_rate=np.asarray([48_000], dtype=np.int64),
        delay_samples=np.asarray([0], dtype=np.int64),
        consistency_band_hz=np.asarray([150.0, 1600.0], dtype=np.float64),
    )

    environments = (
        "DKITCHEN",
        "DWASHING",
        "OOFFICE",
        "OHALLWAY",
        "TMETRO",
        "TCAR",
    )
    rows: list[dict] = []
    selected_group = ""
    for environment in environments:
        contents = [
            source_sha
            if environment == "DKITCHEN" and index == 1
            else _sha(f"{environment}-{index}".encode())
            for index in range(1, 17)
        ]
        lineage_key = f"demand_environment:{environment}"
        group = _group(lineage_key, contents)
        if environment == "DKITCHEN":
            selected_group = group
        for index, content in enumerate(contents, start=1):
            rows.append(
                {
                    "path": (
                        f"/fixture/data/raw/noise/demand/{environment}/"
                        f"ch{index:02d}.wav"
                    ),
                    "duration_s": 300.004,
                    "sample_rate": 48_000,
                    "channels": 1,
                    "tag": "demand",
                    "content_sha256": content,
                    "content_size": (
                        len(source_raw)
                        if environment == "DKITCHEN" and index == 1
                        else 123
                    ),
                    "lineage_schema": public_lineage.PUBLIC_LINEAGE_SCHEMA,
                    "lineage_keys": [lineage_key],
                    "group_id": group,
                    "split": "train",
                }
            )
    manifest = root / demand.DEMAND_PREEXCLUSION_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    generation = root / demand.DEMAND_MANIFEST_GENERATION
    generation.write_bytes(_generation_bytes())
    generation_payload = json.loads(generation.read_text())

    commit = "a" * 40
    freeze = root / demand.DEMAND_ENVIRONMENT_FREEZE
    freeze.parent.mkdir(parents=True)
    freeze.write_text(
        "-e git+https://example.invalid/Deep-ANC.git@"
        f"{commit}#egg=deep-anc\n",
        encoding="utf-8",
    )
    bootstrap = root / demand.DEMAND_BOOTSTRAP_RECEIPT
    bootstrap.parent.mkdir(parents=True, exist_ok=True)
    bootstrap.write_text(
        json.dumps(
            {
                "expected_commit": commit,
                "environment": {
                    "freeze_receipt": demand.DEMAND_ENVIRONMENT_FREEZE,
                    "freeze_receipt_sha256": _sha(freeze.read_bytes()),
                    "torch_version": "2.5.1+cu121",
                    "torch_cuda": "12.1",
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    holdout = root / demand.DEMAND_PARENT82_HOLDOUT
    holdout.parent.mkdir(parents=True, exist_ok=True)
    holdout.write_text(
        json.dumps(
            {
                "families": {
                    "speech": [],
                    "music": [],
                    "environment": [],
                    "machine": [],
                },
                "clip_lineage": {
                    "clips": [],
                    "clips_sha256": "b" * 64,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    parent_rows = [
        {
            "clip": f"parent-{index:03d}.wav",
            "content_sha256": _sha(f"parent-{index}".encode()),
            "lineage_keys": [f"parent_lineage:{index}"],
        }
        for index in range(demand.DEMAND_PARENT82_CLIP_COUNT)
    ]

    clean_source = {
        "schema": "exact_clean_git_source/v1",
        "commit": commit,
        "head_tree_object_id": "c" * 40,
        "git_object_format": "sha1",
        "tracked_file_count": 10,
        "tracked_inventory_sha256": "d" * 64,
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

    monkeypatch.setattr(demand, "DEMAND_SOURCE_SHA256", source_sha)
    monkeypatch.setattr(demand, "DEMAND_SOURCE_SIZE", len(source_raw))
    # production origin start=185.6s이지만 unit fixture를 200초로 부풀리지 않는다.
    # transform과 validator가 같은 explicit origin frame을 쓰는지 자체가 계약이다.
    monkeypatch.setattr(demand, "DEMAND_ORIGIN_WINDOW_START_SECONDS", 15.0)
    monkeypatch.setattr(demand, "DEMAND_ORIGIN_WINDOW_START_FRAME", 15 * 48_000)
    monkeypatch.setattr(
        demand,
        "DEMAND_PREEXCLUSION_MANIFEST_SHA256",
        _sha(manifest.read_bytes()),
    )
    monkeypatch.setattr(demand, "DEMAND_PUBLIC_GROUP_ID", selected_group)
    monkeypatch.setattr(
        demand,
        "DEMAND_LINEAGE_KEY",
        "environment-demand-lineage-"
        + hashlib.sha256(selected_group.encode()).hexdigest()[:12],
    )
    monkeypatch.setattr(
        demand,
        "DEMAND_SELECTION_STRICT_PRIMARY_PATH",
        primary.relative_to(root).as_posix(),
    )
    monkeypatch.setattr(
        demand,
        "DEMAND_SELECTION_STRICT_PRIMARY_SHA256",
        _sha(primary.read_bytes()),
    )
    monkeypatch.setattr(demand, "_git_head", lambda _root: commit)
    monkeypatch.setattr(
        demand,
        "exact_clean_source_evidence",
        lambda _root, **_kwargs: json.loads(json.dumps(clean_source)),
    )
    monkeypatch.setattr(
        demand,
        "validate_environment_freeze_source_commit",
        lambda _raw, *, expected_commit: expected_commit,
    )
    monkeypatch.setattr(
        demand.public_lineage,
        "validate_recorded_clip_lineage",
        lambda _lineage, *, families: json.loads(json.dumps(parent_rows)),
    )

    def fake_holdout(path, *, repo_root, expected_sha256=None):
        raw = Path(path).read_bytes()
        assert expected_sha256 in {None, _sha(raw)}
        return {"_validated_holdout_bytes": raw}

    monkeypatch.setattr(demand, "validate_holdout_contract", fake_holdout)

    def fake_generation(_manifest_dir, *, required_tags, repo_root):
        assert set(required_tags) == {"demand"}
        return {
            **generation_payload,
            "_validated_sidecar_bytes": generation.read_bytes(),
            "_validated_manifest_bytes": {"demand": manifest.read_bytes()},
            "_validated_recorded_generation_exclusion": None,
        }

    monkeypatch.setattr(
        manifest_contract, "validate_manifest_generation", fake_generation
    )
    return {
        "manifest": manifest,
        "selected_group": selected_group,
        "commit": commit,
        "bootstrap_sha256": _sha(bootstrap.read_bytes()),
        "generation_sha256": _sha(generation.read_bytes()),
    }


def _build(root: Path, fixture: dict[str, object]):
    return demand.build_demand_selection_payload(
        repo_root=root,
        bootstrap_receipt=demand.DEMAND_BOOTSTRAP_RECEIPT,
        bootstrap_receipt_sha256=str(fixture["bootstrap_sha256"]),
        expected_commit=str(fixture["commit"]),
        expected_manifest_generation_sha256=str(fixture["generation_sha256"]),
    )


def _publish(root: Path, payload: dict, files: dict[str, bytes]) -> None:
    bundle = root / demand.DEMAND_SELECTION_BUNDLE_ROOT
    for relative, raw in files.items():
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    receipt = root / demand.DEMAND_SELECTION_RECEIPT
    receipt.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _reseal(root: Path, payload: dict) -> None:
    payload["evidence_sha256"] = demand._canonical_json_sha256(
        demand._without_evidence_sha(payload)
    )
    (root / demand.DEMAND_SELECTION_RECEIPT).write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _render_jsonl(rows: list[dict]) -> bytes:
    return "".join(
        json.dumps(row, separators=(",", ":")) + "\n" for row in rows
    ).encode()


def test_immutable_demand_bundle_survives_live_manifest_republish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    live_manifest = fixture["manifest"]
    assert isinstance(live_manifest, Path)
    selected_group = str(fixture["selected_group"])
    payload, files = _build(tmp_path, fixture)
    _publish(tmp_path, payload, files)

    selected = payload["selected"]
    assert selected["recorded_split"] == "test"
    assert selected["public_group_member_count"] == 16
    assert selected["origin_window_start_seconds"] == 15.0
    assert selected["window_start_seconds"] == 0.0
    assert selected["window_seconds"] == 15.0
    assert selected["strict_p_coverage"]["covered_subband_count"] == 4
    assert selected["stationarity"]["spectral_flatness"] >= 0.75
    assert selected["stationarity"]["spectral_entropy"] >= 0.94
    assert selected["stationarity"]["one_second_rms_peak_to_peak_db"] <= 6.0
    assert "one_second_rms_p90_p10_db" not in selected["stationarity"]
    assert Path(selected["bundle_source"]["path"]).name != "ch01.wav"
    assert selected["bundle_source"]["frames"] == 15 * 48_000
    assert (
        selected["origin_bundle_source"]["sha256"]
        == selected["origin_source"]["sha256"]
    )
    level = selected["rendered_level"]
    assert level["passed"] is True
    assert level["peak_linear"] <= demand.DEMAND_PLAYBACK_AMPLITUDE
    assert (
        level["trusted_band_rms_dbfs"]
        >= level["thresholds"]["minimum_trusted_band_rms_dbfs"]
    )
    assert (
        level["predicted_signal_to_quiet_db"]
        >= level["thresholds"]["minimum_predicted_signal_to_quiet_db"]
    )

    # recorded-generation exclusion 재발행으로 live manifest가 96→80행이 되어도
    # validator는 immutable selection-parent와 source copy만 읽어야 한다.
    rows = [json.loads(line) for line in live_manifest.read_text().splitlines()]
    retained = [row for row in rows if row["group_id"] != selected_group]
    assert len(rows) == 96
    assert len(retained) == 80
    live_manifest.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in retained),
        encoding="utf-8",
    )
    summary = demand.validate_demand_selection_receipt(repo_root=tmp_path)
    assert summary["selected"]["public_group_id"] == selected_group
    assert len(summary["bundle_files"]) == 8
    assert len({item["path"] for item in summary["bundle_files"]}) == 8


def test_demand_bundle_blocks_pretrain_until_exclusion_sidecar_is_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    selected_group = str(fixture["selected_group"])
    manifest = fixture["manifest"]
    assert isinstance(manifest, Path)
    parent_rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    retained = [row for row in parent_rows if row["group_id"] != selected_group]
    assert len(retained) == 80
    payload, files = _build(tmp_path, fixture)
    _publish(tmp_path, payload, files)
    generation = {
        "_validated_entries": {
            "demand": [{"group_id": selected_group}],
        },
        "_validated_manifest_bytes": {"demand": _render_jsonl(parent_rows)},
        "_validated_recorded_generation_exclusion": None,
    }
    with pytest.raises(demand.DemandSelectionBlocked, match="pretrain"):
        demand.require_demand_selection_excluded_from_manifest_generation(
            generation, repo_root=tmp_path
        )
    generation["_validated_recorded_generation_exclusion"] = {"validated": True}
    with pytest.raises(demand.DemandSelectionBlocked, match="16개"):
        demand.require_demand_selection_excluded_from_manifest_generation(
            generation, repo_root=tmp_path
        )
    generation["_validated_entries"]["demand"] = retained
    generation["_validated_manifest_bytes"]["demand"] = _render_jsonl(retained)
    demand.require_demand_selection_excluded_from_manifest_generation(
        generation, repo_root=tmp_path
    )

    generation["_validated_entries"]["demand"] = retained[:-1]
    generation["_validated_manifest_bytes"]["demand"] = _render_jsonl(retained[:-1])
    with pytest.raises(demand.DemandSelectionBlocked, match="exact 80"):
        demand.require_demand_selection_excluded_from_manifest_generation(
            generation, repo_root=tmp_path
        )

    modified = json.loads(json.dumps(retained))
    modified[0]["content_size"] += 1
    generation["_validated_entries"]["demand"] = modified
    generation["_validated_manifest_bytes"]["demand"] = _render_jsonl(modified)
    with pytest.raises(demand.DemandSelectionBlocked, match="exact 80"):
        demand.require_demand_selection_excluded_from_manifest_generation(
            generation, repo_root=tmp_path
        )


def test_demand_bundle_rejects_source_byte_or_resealed_window_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    payload, files = _build(tmp_path, fixture)
    _publish(tmp_path, payload, files)
    source = tmp_path / demand.DEMAND_SELECTION_SOURCE
    raw = bytearray(source.read_bytes())
    raw[-1] ^= 1
    source.write_bytes(raw)
    with pytest.raises(demand.DemandSelectionError, match="SHA/size"):
        demand.validate_demand_selection_receipt(repo_root=tmp_path)

    source.write_bytes(
        files[f"sources/{Path(demand.DEMAND_SELECTION_SOURCE).name}"]
    )
    origin = tmp_path / demand.DEMAND_SELECTION_ORIGIN_SOURCE
    origin_raw = bytearray(origin.read_bytes())
    origin_raw[-1] ^= 1
    origin.write_bytes(origin_raw)
    with pytest.raises(demand.DemandSelectionError, match="origin source SHA/size"):
        demand.validate_demand_selection_receipt(repo_root=tmp_path)
    origin.write_bytes(
        files[f"sources/{Path(demand.DEMAND_SELECTION_ORIGIN_SOURCE).name}"]
    )
    payload["selected"]["origin_window_start_seconds"] = 14.0
    _reseal(tmp_path, payload)
    with pytest.raises(demand.DemandSelectionError, match="origin/playback"):
        demand.validate_demand_selection_receipt(repo_root=tmp_path)


def test_rendered_level_contract_rejects_weak_trusted_band_and_resealed_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 4 kHz 1초 burst는 peak/RMS는 충분하지만 150--1600 Hz excitation이
    # 없다. 따라서 디지털 amplitude=.06만 맞는 source를 통과시키지 않는다.
    frames = demand.DEMAND_WINDOW_FRAMES
    values = np.zeros(frames, dtype=np.float64)
    phase = 2.0 * np.pi * 4_000.0 * np.arange(48_000) / 48_000.0
    values[:48_000] = np.sin(phase)
    with pytest.raises(demand.DemandSelectionError, match="absolute playback"):
        demand._rendered_level_metrics(_pcm16_wav(values))

    fixture = _fixture(tmp_path, monkeypatch)
    payload, files = _build(tmp_path, fixture)
    _publish(tmp_path, payload, files)
    payload["selected"]["rendered_level"]["official_meter_playback_trusted_band_dbfs"] += 1.0
    _reseal(tmp_path, payload)
    with pytest.raises(demand.DemandSelectionError, match="rendered_level"):
        demand.validate_demand_selection_receipt(
            repo_root=tmp_path,
            require_source_files=False,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("source_commit", "clean_source"),
        ("bootstrap_receipt", "bootstrap"),
        ("environment_freeze", "freeze"),
        ("manifest_generation", "manifest_generation"),
        ("parent82", "parent82"),
    ),
)
def test_demand_bundle_rejects_resealed_provenance_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    payload, files = _build(tmp_path, fixture)
    _publish(tmp_path, payload, files)
    if field == "source_commit":
        payload[field] = "f" * 40
    elif field == "parent82":
        payload[field]["clip_count"] -= 1
    else:
        payload[field]["sha256"] = "f" * 64
    _reseal(tmp_path, payload)
    with pytest.raises(demand.DemandSelectionError, match=message):
        demand.validate_demand_selection_receipt(repo_root=tmp_path)


def test_actual_dkitchen_composite_level_and_stationarity_are_stable() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / demand.DEMAND_SOURCE_ORIGIN
    primary = root / demand.DEMAND_SELECTION_STRICT_PRIMARY_PATH
    if not source.is_file() or not primary.is_file():
        pytest.skip("production DEMAND/strict-P artifacts are intentionally external")
    source_raw = source.read_bytes()
    if _sha(source_raw) != demand.DEMAND_SOURCE_SHA256:
        pytest.skip("local DEMAND source is not the audited external artifact")
    origin_values, _audio = demand._decode_source(source_raw)
    composite_raw = demand._canonical_composite_bytes(origin_values)
    values, composite_audio = demand._decode_source(composite_raw)
    assert composite_audio["frames"] == 720_000
    assert _sha(composite_raw) == (
        "bc193c07d161b7ab99d022119ae50f25d2deba7ae6360994e46d04bb51705dd8"
    )
    _snapshot, fir, _metadata = demand._strict_primary(
        root, demand.DEMAND_SELECTION_STRICT_PRIMARY_PATH
    )
    metrics = demand._selection_metrics(values, fir)
    coverage = metrics["strict_p_coverage"]
    stationarity = metrics["stationarity"]
    expected_density = (
        4.883855081844097,
        1.209867120017137,
        0.5078379110284694,
        0.25221072884476115,
    )
    assert all(
        math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
        for observed, expected in zip(
            coverage["density_ratios"], expected_density, strict=True
        )
    )
    assert math.isclose(
        stationarity["spectral_flatness"],
        0.7556982285478725,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert math.isclose(
        stationarity["spectral_entropy"],
        0.9508895515057949,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert math.isclose(
        stationarity["one_second_rms_peak_to_peak_db"],
        3.5869057310283345,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    level = demand._rendered_level_metrics(composite_raw)
    assert level["passed"] is True
    assert level["peak_dbfs"] == pytest.approx(-24.43697518647189, abs=1e-10)
    assert level["rms_dbfs"] == pytest.approx(-43.48975791157436, abs=1e-10)
    assert level["trusted_band_rms_dbfs"] == pytest.approx(
        -56.6646471405205, abs=1e-10
    )
    assert level["predicted_signal_to_quiet_db"] == pytest.approx(
        11.722667586302308, abs=1e-10
    )


def test_demand_selector_help_works_in_documented_isolated_mode() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=/dev/null/deep-anc-selector",
            str(root / "scripts/data/select_recorded_demand_environment.py"),
            "--help",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--expected-manifest-generation-sha256" in result.stdout
