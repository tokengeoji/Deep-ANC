from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from deep_anc.data import recording_source_gain as gain


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo(tmp_path: Path, *, strict_gain: float = 10.0) -> tuple[Path, str, str]:
    source = tmp_path / "data/source.wav"
    source.parent.mkdir(parents=True)
    sample_rate = 48_000
    time = np.arange(sample_rate * 15, dtype=np.float64) / sample_rate
    # low/high capture band가 모두 독립 에너지를 갖도록 한다. NoiseProgram이 전체
    # 파일 peak를 reference amplitude에 맞춘다.
    values = 0.55 * np.sin(2.0 * np.pi * 300.0 * time)
    values += 0.45 * np.sin(2.0 * np.pi * 1000.0 * time + 0.17)
    sf.write(source, values.astype(np.float32), sample_rate, subtype="FLOAT")

    source_plan = tmp_path / "data/source_plan.csv"
    fields = (
        "path",
        "seconds",
        "start_seconds",
        "source_file_sha256",
        "source_family",
        "group_id",
        "lineage_key",
        "split",
    )
    with source_plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "path": "data/source.wav",
                "seconds": "15.0",
                "start_seconds": "0.0",
                "source_file_sha256": _sha256(source),
                "source_family": "music",
                "group_id": "music-test-group",
                "lineage_key": "music-test-lineage",
                "split": "train",
            }
        )

    strict = tmp_path / "assets/strict_primary.npz"
    strict.parent.mkdir(parents=True)
    np.savez(
        strict,
        fir=np.asarray([strict_gain, 0.0], dtype=np.float32),
        sample_rate=np.int64(sample_rate),
        delay_samples=np.int64(100),
        consistency_band_hz=np.asarray([150.0, 1600.0], dtype=np.float64),
        capture_id=np.str_("fixture-capture"),
        output_channel=np.str_("noise"),
        amplitude=np.float64(0.003),
        xrun_count=np.int64(0),
    )
    return tmp_path, _sha256(source_plan), _sha256(strict)


def _build(root: Path, plan_sha: str, strict_sha: str) -> dict:
    return gain.build_recording_source_gain_plan(
        repo_root=root,
        source_plan="data/source_plan.csv",
        expected_source_plan_sha256=plan_sha,
        strict_primary="assets/strict_primary.npz",
        expected_strict_primary_sha256=strict_sha,
    )


def _mock_strict_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gain,
        "_validate_strict_primary_authority",
        lambda *_args, **_kwargs: {
            "capture_id": "fixture-capture",
            "raw_measurement_sha256": "d" * 64,
            "analysis_sha256": "e" * 64,
            "derived_lead_samples": 115,
        },
    )


def test_strict_p_peak_selects_source_specific_micro_amplitude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mock_strict_authority(monkeypatch)
    root, plan_sha, strict_sha = _repo(tmp_path, strict_gain=10.0)
    payload = _build(root, plan_sha, strict_sha)

    assert payload["schema"] == gain.RECORDING_SOURCE_GAIN_SCHEMA
    assert payload["canonical_live_eligible"] is False
    assert payload["blocker_reasons"] == list(gain.GAIN_PLAN_BLOCKERS)
    row = payload["rows"][0]
    assert 49_990 <= row["selected_amplitude_millionths"] <= 50_000
    assert row["selected_predicted_err"]["peak_linear"] <= 0.5
    assert row["selected_predicted_err"]["rms_linear"] <= 0.5
    assert all(value >= 10.0 * np.log10(9.0) for value in row[
        "selected_predicted_err_snr_db"
    ].values())
    assert row["selected_source_preflight"]["timeline_feasibility"]["passed"] is True


def test_infeasible_interval_fails_closed_instead_of_lowering_snr_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mock_strict_authority(monkeypatch)
    root, plan_sha, strict_sha = _repo(tmp_path, strict_gain=1000.0)
    with pytest.raises(gain.RecordingSourceGainError, match="feasible interval"):
        _build(root, plan_sha, strict_sha)


def test_no_replace_plan_recomputes_source_and_strict_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mock_strict_authority(monkeypatch)
    root, plan_sha, strict_sha = _repo(tmp_path, strict_gain=10.0)
    summary = gain.issue_recording_source_gain_plan(
        repo_root=root,
        output_path="results/source_gain.json",
        source_plan="data/source_plan.csv",
        expected_source_plan_sha256=plan_sha,
        strict_primary="assets/strict_primary.npz",
        expected_strict_primary_sha256=strict_sha,
    )
    assert summary["canonical_live_eligible"] is False
    with pytest.raises(gain.RecordingSourceGainError, match="no-replace"):
        gain.issue_recording_source_gain_plan(
            repo_root=root,
            output_path="results/source_gain.json",
            source_plan="data/source_plan.csv",
            expected_source_plan_sha256=plan_sha,
            strict_primary="assets/strict_primary.npz",
            expected_strict_primary_sha256=strict_sha,
        )

    path = root / "results/source_gain.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["selected_amplitude_millionths"] -= 1
    unsealed = dict(payload)
    unsealed.pop("evidence_sha256")
    payload["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            unsealed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(gain.RecordingSourceGainError, match="독립 재계산"):
        gain.validate_recording_source_gain_plan(
            repo_root=root,
            plan_path="results/source_gain.json",
            expected_sha256=_sha256(path),
        )


def _load_batch_module():
    name = "record_session_batch_source_gain_test"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/data/record_session_batch.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_record_duct_module():
    name = "record_duct_source_gain_hardware_test"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/data/record_duct.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_additions_builder_module():
    name = "build_recorded_additions_plan_gain_gate_test"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/data/build_recorded_additions_plan.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_canonical_batch_requires_gain_plan_and_keeps_v1_live_blocked(monkeypatch):
    batch = _load_batch_module()
    missing = argparse.Namespace(
        source_gain_plan=None,
        source_gain_plan_sha256=None,
    )
    with pytest.raises(gain.RecordingSourceGainError, match="path/SHA"):
        batch._validate_batch_source_gain_plan(
            missing, required=True, source_list_sha256="a" * 64
        )

    supplied = argparse.Namespace(
        source_gain_plan="results/source_gain.json",
        source_gain_plan_sha256="b" * 64,
    )
    monkeypatch.setattr(
        batch,
        "validate_recording_source_gain_plan",
        lambda **_kwargs: {
            "plan_sha256": "b" * 64,
            "canonical_live_eligible": False,
            "payload": {
                "source_plan": {"sha256": "a" * 64},
                "blocker_reasons": list(gain.GAIN_PLAN_BLOCKERS),
            },
        },
    )
    with pytest.raises(gain.RecordingSourceGainError, match="ERR-only"):
        batch._validate_batch_source_gain_plan(
            supplied, required=True, source_list_sha256="a" * 64
        )


def _physical_receipt_summary(root: Path, receipt: Path) -> dict:
    channels = {}
    for name, tap in (("err", 1.0), ("ref", 2.0)):
        fir = np.zeros(gain.OPERATOR_FIR_LENGTH, dtype=np.float32)
        fir[0] = tap
        channels[name] = {
            "passed": True,
            "fir_encoding": "float32_le",
            "fir": [float(value) for value in fir],
            "fir_sha256": hashlib.sha256(
                np.ascontiguousarray(fir, dtype="<f4").tobytes()
            ).hexdigest(),
            "residual_bound": {
                "definition": (
                    "young_l1_induced_plus_measured_absolute_with_uncertainty_v1"
                ),
                "valid_through_amplitude_millionths": 6_000,
                "induced_fir_l1_upper": 0.0,
                "unexplained_peak_absolute_upper": 0.0,
                "unexplained_rms_absolute_upper": 0.0,
                "uncertainty_factor": 1.25,
            },
        }
    operator = {
        "schema": "recording_gain_safety_operator/v3_gainprobe006",
        "role": "source_gain_prediction_only_not_anc_plant_authority",
        "fir_length": gain.OPERATOR_FIR_LENGTH,
        "peak_pre_roll_samples": gain.OPERATOR_PEAK_PRE_ROLL_SAMPLES,
        "relative_subband_minimum_target_norm_ratio": (
            gain.OPERATOR_RELATIVE_SUBBAND_MIN_NORM_RATIO
        ),
        "channels": channels,
    }
    operator["operator_sha256"] = gain._seal(operator)
    hardware_path = root / "configs/hardware.yaml"
    hardware_path.parent.mkdir(parents=True, exist_ok=True)
    hardware_path.write_text("fixture: true\n", encoding="utf-8")
    fingerprint = {"schema": "fixture", "cards": ["APE", "Audio"]}
    payload = {
        "source_commit": "a" * 40,
        "hardware": {
            "path": "configs/hardware.yaml",
            "size": hardware_path.stat().st_size,
            "sha256": _sha256(hardware_path),
            "sample_rate": 48_000,
            "block_size": 256,
            "latency": "low",
            "channels": {
                "error_mic": 0,
                "reference_mic": 1,
                "noise_out": 0,
                "cancel_out": 1,
            },
            "physical_fingerprint": fingerprint,
            "physical_fingerprint_sha256": gain._seal(fingerprint),
        },
        "analysis": {
            "safety_operator": operator,
            "safety_operator_is_anc_plant_authority": False,
            "tested_max_amplitude_millionths": 6_000,
            "supported_max_amplitude_millionths": 6_000,
            "distortion_certified": False,
            "physical_authority_scope": gain.GAIN_LINEARITY_AUTHORITY_SCOPE,
        },
    }
    return {"passed": True, "payload": payload}


def test_v3_caps_selected_gain_at_measured_006_and_builds_exact_session_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mock_strict_authority(monkeypatch)
    root, plan_sha, strict_sha = _repo(tmp_path, strict_gain=10.0)
    receipt = root / "results/linearity_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")
    receipt_sha = _sha256(receipt)
    summary = _physical_receipt_summary(root, receipt)
    monkeypatch.setattr(gain, "validate_gain_linearity_receipt", lambda **_kwargs: summary)

    payload = gain.build_recording_source_gain_plan(
        repo_root=root,
        source_plan="data/source_plan.csv",
        expected_source_plan_sha256=plan_sha,
        strict_primary="assets/strict_primary.npz",
        expected_strict_primary_sha256=strict_sha,
        gain_linearity_receipt="results/linearity_receipt.json",
        expected_gain_linearity_receipt_sha256=receipt_sha,
    )
    assert payload["canonical_live_eligible"] is True
    assert payload["contract"]["reference_amplitude_millionths"] == 6_000
    assert payload["contract"]["source_reference_amplitude"] == 0.006
    assert payload["contract"]["legacy_schema_v1_source_reference_amplitude"] is None
    row = payload["rows"][0]
    assert row["reference_amplitude_millionths"] == 6_000
    assert row["selected_amplitude_millionths"] == 6_000
    assert row["bounds"]["upper_constraints"]["physical_probe_supported_max"] == 6_000
    assert row["selected_physical_prediction"]["ref"]["upper_peak_linear"] <= 0.4
    assert payload["contract"]["safety_operator_is_anc_plant_authority"] is False

    plan = root / "results/source_gain_v2.json"
    plan.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    validated = gain.validate_recording_source_gain_plan(
        repo_root=root,
        plan_path="results/source_gain_v2.json",
        expected_sha256=_sha256(plan),
    )
    binding = gain.build_recording_source_gain_session_binding(
        validated, source_row_number=2, expected_source_commit="a" * 40
    )
    assert binding["amplitude_millionths"] == 6_000
    assert binding["supported_max_amplitude_millionths"] == 6_000
    assert binding["safety_operator_is_anc_plant_authority"] is False
    assert gain.validate_recording_source_gain_session_binding(validated, binding) == binding


def test_v3_uses_receipt_dynamic_cap_below_tested_6k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mock_strict_authority(monkeypatch)
    root, plan_sha, strict_sha = _repo(tmp_path, strict_gain=10.0)
    receipt = root / "results/linearity_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")
    summary = _physical_receipt_summary(root, receipt)
    summary["payload"]["analysis"]["supported_max_amplitude_millionths"] = 5_000
    monkeypatch.setattr(
        gain, "validate_gain_linearity_receipt", lambda **_kwargs: summary
    )
    payload = gain.build_recording_source_gain_plan(
        repo_root=root,
        source_plan="data/source_plan.csv",
        expected_source_plan_sha256=plan_sha,
        strict_primary="assets/strict_primary.npz",
        expected_strict_primary_sha256=strict_sha,
        gain_linearity_receipt="results/linearity_receipt.json",
        expected_gain_linearity_receipt_sha256=_sha256(receipt),
    )
    assert payload["contract"]["physical_probe_tested_max_amplitude_millionths"] == 6_000
    assert payload["contract"]["reference_amplitude_millionths"] == 5_000
    assert payload["rows"][0]["selected_amplitude_millionths"] <= 5_000


def test_v3_rejects_residual_bound_not_valid_through_tested_holdout(
    tmp_path: Path
):
    receipt = tmp_path / "receipt.json"
    summary = _physical_receipt_summary(tmp_path, receipt)
    analysis = summary["payload"]["analysis"]
    operator = analysis["safety_operator"]
    operator["channels"]["err"]["residual_bound"][
        "valid_through_amplitude_millionths"
    ] = 5_000
    operator.pop("operator_sha256")
    operator["operator_sha256"] = gain._seal(operator)
    with pytest.raises(gain.RecordingSourceGainError, match="FIR/margin"):
        gain._physical_operator_contract(analysis)


def test_v3_rejects_receipt_that_claims_distortion_certification(tmp_path: Path):
    summary = _physical_receipt_summary(tmp_path, tmp_path / "receipt.json")
    analysis = summary["payload"]["analysis"]
    analysis["distortion_certified"] = True
    with pytest.raises(gain.RecordingSourceGainError, match="tested ADC peak safety"):
        gain._physical_operator_contract(analysis)


def test_measured_cap_audit_and_deterministic_window_selector_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _mock_strict_authority(monkeypatch)
    root, _plan_sha, strict_sha = _repo(tmp_path, strict_gain=10.0)
    source = root / "data/source.wav"
    sample_rate = 48_000
    time = np.arange(sample_rate * 15, dtype=np.float64) / sample_rate
    active = 0.55 * np.sin(2.0 * np.pi * 300.0 * time)
    active += 0.45 * np.sin(2.0 * np.pi * 1000.0 * time + 0.17)
    sf.write(
        source,
        np.concatenate([np.zeros_like(active), active]).astype(np.float32),
        sample_rate,
        subtype="FLOAT",
    )
    plan = root / "data/source_plan.csv"
    rows = list(csv.DictReader(plan.read_text(encoding="utf-8").splitlines()))
    rows[0]["source_file_sha256"] = _sha256(source)
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plan_sha = _sha256(plan)

    blocked = gain.audit_source_plan_at_measured_cap(
        repo_root=root,
        source_plan="data/source_plan.csv",
        expected_source_plan_sha256=plan_sha,
        strict_primary="assets/strict_primary.npz",
        expected_strict_primary_sha256=strict_sha,
        amplitude_millionths=6_000,
    )
    assert blocked["all_rows_feasible"] is False
    assert blocked["feasible_row_count"] == 0
    assert "rendered_source_preflight" in blocked["blockers"][0]["reasons"]

    _source_ref, source_rows = gain._read_source_rows(
        root, "data/source_plan.csv", expected_sha256=plan_sha
    )
    _strict_ref, fir = gain._load_strict_primary(
        root, "assets/strict_primary.npz", expected_sha256=strict_sha
    )
    selected = gain.select_best_feasible_source_window(
        row=source_rows[0],
        strict_primary_fir=fir,
        candidate_start_seconds=[0.0, 15.0],
    )
    assert selected["candidate_count"] == 2
    assert selected["feasible_count"] == 1
    assert selected["selected_start_seconds"] == 15.0
    with pytest.raises(gain.RecordingSourceGainError, match="feasible source window"):
        gain.select_best_feasible_source_window(
            row=source_rows[0],
            strict_primary_fir=fir,
            candidate_start_seconds=[0.0],
        )


def test_batch_uses_exact_v3_integer_gain_per_row_and_dynamic_cap():
    batch = _load_batch_module()
    entries = [{"source_row_number": 2}, {"source_row_number": 3}]
    summary = {
        "canonical_live_eligible": True,
        "payload": {
            "contract": {"reference_amplitude_millionths": 5_000},
            "rows": [
                {
                    "source_row_number": 2,
                    "selected_amplitude_millionths": 5_000,
                    "feasible": True,
                },
                {
                    "source_row_number": 3,
                    "selected_amplitude_millionths": 4_001,
                    "feasible": True,
                },
            ]
        },
    }
    assert batch._canonical_source_gain_by_row(summary, entries) == {
        2: 0.005,
        3: 0.004001,
    }
    summary["payload"]["rows"][0]["selected_amplitude_millionths"] = 5_001
    with pytest.raises(gain.RecordingSourceGainError, match="row/amplitude"):
        batch._canonical_source_gain_by_row(summary, entries)


def test_additions_builder_consumes_receipt_cap_and_requires_19_of_19(monkeypatch):
    builder = _load_additions_builder_module()
    monkeypatch.setattr(
        builder,
        "validate_gain_linearity_receipt",
        lambda **_kwargs: {
            "passed": True,
            "payload": {"analysis": {"supported_max_amplitude_millionths": 6_000}},
        },
    )
    assert builder._measured_cap_from_receipt(
        receipt_path="results/receipt.json", expected_receipt_sha256="a" * 64
    ) == 6_000

    monkeypatch.setattr(
        builder,
        "audit_source_plan_at_measured_cap",
        lambda **_kwargs: {
            "row_count": 19,
            "feasible_row_count": 18,
            "blockers": [
                {
                    "source_row_number": 2,
                    "path": "data/source.wav",
                    "reasons": ["rendered_source_preflight"],
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="19/19 feasible"):
        builder._require_all_rows_feasible(
            relative_plan="data/plan.csv",
            plan_sha256="b" * 64,
            amplitude_millionths=6_000,
        )


def test_additions_verify_existing_rejects_stale_generation_before_receipt_read(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    builder = _load_additions_builder_module()
    calls = []
    monkeypatch.setattr(
        builder,
        "_measured_cap_from_receipt",
        lambda **_kwargs: calls.append("receipt") or 6_000,
    )
    result = builder.main(
        [
            "--generation-id",
            "stage1-coverage-v3-gain012",
            "--dns-selection-receipt-sha256",
            "a" * 64,
            "--demand-selection-receipt-sha256",
            "b" * 64,
            "--gain-linearity-receipt",
            "results/stale.json",
            "--gain-linearity-receipt-sha256",
            "c" * 64,
            "--verify-existing",
        ]
    )
    assert result == 1
    assert calls == []
    assert "현행 exact source plan generation-id" in capsys.readouterr().err


def test_strict_primary_authority_accepts_current_full_pair_and_rejects_legacy():
    canonical = ROOT / "assets/measured/primary_path_il_strict_5dc06fdd.npz"
    authority = gain._validate_strict_primary_authority(
        ROOT, canonical.relative_to(ROOT).as_posix(), _sha256(canonical)
    )
    assert authority["capture_id"] == "5ac1313488c8434bb4d672a36503df59"
    assert authority["derived_lead_samples"] == 115
    assert len(authority["raw_measurement_sha256"]) == 64
    assert len(authority["analysis_sha256"]) == 64

    legacy = ROOT / "assets/measured/primary_path_il.npz"
    with pytest.raises(gain.RecordingSourceGainError, match="canonical configs/duct.yaml"):
        gain._validate_strict_primary_authority(
            ROOT, legacy.relative_to(ROOT).as_posix(), _sha256(legacy)
        )


def test_environment_006_42s_old_012_pass_is_not_reused_at_new_006_cap():
    path = "data/source_pool/environment/environment_006.wav"
    source = ROOT / path
    source_sha = _sha256(source)
    strict_path = ROOT / "assets/measured/primary_path_il_strict_5dc06fdd.npz"
    _strict_ref, fir = gain._load_strict_primary(
        ROOT,
        strict_path.relative_to(ROOT).as_posix(),
        expected_sha256=_sha256(strict_path),
    )
    base = {
        "source_row_number": 2,
        "path": path,
        "seconds": 15.0,
        "source_file": {
            "path": path,
            "size": source.stat().st_size,
            "sha256": source_sha,
        },
        "source_bytes": source.read_bytes(),
    }
    stale = dict(base, start_seconds=25.75, source_identity_sha256="a" * 64)
    current = dict(base, start_seconds=42.0, source_identity_sha256="b" * 64)
    stale_evidence = gain._source_cap_evidence(
        stale, fir, amplitude_millionths=6_000
    )
    current_evidence = gain._source_cap_evidence(
        current, fir, amplitude_millionths=6_000
    )
    old_cap_evidence = gain._source_cap_evidence(
        current, fir, amplitude_millionths=12_000
    )
    assert stale_evidence["feasible"] is False
    assert "rendered_source_preflight" in stale_evidence["blocker_reasons"]
    assert old_cap_evidence["feasible"] is True
    assert old_cap_evidence["rendered_source_preflight"]["timeline_feasibility"][
        "eligible_ratio"
    ] >= 0.95
    assert current_evidence["feasible"] is False
    assert any(
        reason.startswith("strict_primary_snr")
        for reason in current_evidence["blocker_reasons"]
    )


def test_synthetic_minimal_primary_cannot_bypass_canonical_authority(tmp_path: Path):
    root, _plan_sha, strict_sha = _repo(tmp_path)
    with pytest.raises((gain.RecordingSourceGainError, ValueError)):
        gain._load_strict_primary(
            root, "assets/strict_primary.npz", expected_sha256=strict_sha
        )


def test_strict_authority_held_guard_rejects_mid_validation_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "configs").mkdir()
    (tmp_path / "assets").mkdir()
    duct = tmp_path / gain.CANONICAL_DUCT_CONFIG
    hardware = tmp_path / gain.CANONICAL_HARDWARE_CONFIG
    primary = tmp_path / "assets/p.npz"
    secondary = tmp_path / "assets/s.npz"
    level = tmp_path / "assets/level.json"
    duct.write_text(
        "digital_reference:\n  primary_path_npz: assets/p.npz\n"
        "secondary_path:\n  npz: assets/s.npz\n"
        "strict_measurement:\n  measurement_level_evidence: assets/level.json\n",
        encoding="utf-8",
    )
    hardware.write_text("audio:\n  block_size: 256\n", encoding="utf-8")
    primary.write_bytes(b"primary")
    secondary.write_bytes(b"secondary")
    level.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        gain,
        "load_secondary_path",
        lambda path: SimpleNamespace(
            fir=np.asarray([1.0, 0.0]), delay_samples=1, sample_rate=48_000
        ),
    )
    monkeypatch.setattr(
        gain,
        "PlantDelays",
        SimpleNamespace(from_config=lambda **_kwargs: SimpleNamespace()),
    )
    monkeypatch.setattr(
        gain,
        "TrainingTimingContract",
        SimpleNamespace(
            derive=lambda **_kwargs: SimpleNamespace(
                digital_reference_lead_samples=115
            )
        ),
    )

    def mutate_during_validator(_config):
        duct.write_text(duct.read_text(encoding="utf-8") + "# mutated\n", encoding="utf-8")
        return SimpleNamespace(
            primary_path_sha256=_sha256(primary),
            secondary_path_sha256=_sha256(secondary),
            measurement_level_evidence_sha256=_sha256(level),
            capture_id="fixture",
            raw_measurement_sha256="a" * 64,
            analysis_sha256="b" * 64,
        )

    monkeypatch.setattr(gain, "validate_runtime_plant_contract", mutate_during_validator)
    with pytest.raises(gain.RecordingSourceGainError, match="변경"):
        gain._validate_strict_primary_authority(
            tmp_path, "assets/p.npz", _sha256(primary)
        )


def test_record_duct_blocks_wrong_physical_fingerprint_before_audio_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    record_duct = _load_record_duct_module()
    hardware = tmp_path / "configs/hardware.yaml"
    hardware.parent.mkdir(parents=True)
    hardware.write_text("fixture: true\n", encoding="utf-8")
    expected_fingerprint = {"schema": "fixture", "device": "expected"}
    gain_hardware = {
        "path": "configs/hardware.yaml",
        "sha256": _sha256(hardware),
        "physical_fingerprint": expected_fingerprint,
        "physical_fingerprint_sha256": gain._seal(expected_fingerprint),
    }
    summary = {
        "canonical_live_eligible": True,
        "payload": {
            "source_plan": {"sha256": "a" * 64},
            "contract": {"gain_linearity_source_commit": "b" * 40},
        },
    }
    binding = {
        "gain_linearity_hardware": gain_hardware,
        "source_file": {"sha256": "c" * 64},
        "amplitude_millionths": 6_000,
        "amplitude": 0.006,
    }
    monkeypatch.setattr(record_duct, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        record_duct, "validate_recording_source_gain_plan", lambda **_kwargs: summary
    )
    monkeypatch.setattr(
        record_duct,
        "build_recording_source_gain_session_binding",
        lambda *_args, **_kwargs: binding,
    )
    monkeypatch.setattr(
        record_duct,
        "collect_alsa_physical_fingerprint",
        lambda *_args: {"schema": "fixture", "device": "wrong"},
    )
    monkeypatch.setattr(
        record_duct,
        "_sha256",
        lambda path: (
            "a" * 64
            if Path(path).name == "plan.csv"
            else (
                "c" * 64
                if Path(path).name == "source.wav"
                else _sha256(Path(path))
            )
        ),
    )
    args = argparse.Namespace(
        source_gain_plan="results/source_gain.json",
        source_gain_plan_sha256="d" * 64,
        require_source_gain_plan=True,
        dry_run=False,
        hardware="configs/hardware.yaml",
        file="data/source.wav",
        amplitude=0.006,
    )
    collection = {
        "status": "exact",
        "source_list": "data/plan.csv",
        "source_list_sha256": "a" * 64,
        "source_row_number": 2,
        "source_file_sha256": "c" * 64,
    }
    with pytest.raises(gain.RecordingSourceGainError, match="physical fingerprint"):
        record_duct._validate_source_gain_authority(
            args, collection_plan=collection, require_clean_execution=False
        )

    monkeypatch.setattr(
        record_duct,
        "collect_alsa_physical_fingerprint",
        lambda *_args: expected_fingerprint,
    )
    args.amplitude = 0.006001
    with pytest.raises(gain.RecordingSourceGainError, match="CLI amplitude"):
        record_duct._validate_source_gain_authority(
            args, collection_plan=collection, require_clean_execution=False
        )
