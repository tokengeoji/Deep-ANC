from __future__ import annotations

import hashlib
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from deep_anc.dsp import fullband_live_authority_v6 as v6
from deep_anc.dsp import fullband_v6_meter as meter


def test_v6_design_geometry_is_exact():
    contract = v6.design_contract_v6()
    assert v6.CLOCK_BLOCK_FRAMES * v6.CLOCK_BLOCKS == 24 * v6.PERIOD
    assert v6.PE_BLOCK_FRAMES * v6.PE_BLOCKS == 12 * v6.PERIOD
    assert contract["total_frames"] == 1_179_648
    assert contract["duration_seconds"] == 24.576
    assert contract["clock"]["fixed_bins"] == [109, 137, 181, 233, 277, 314, 359, 401]
    assert contract["submitted_peak_limit_pcm"] == 98


def test_actual_builder_assets_and_pinned_loaders(tmp_path: Path):
    envelope = v6.committed_plan_envelope_v6()
    authority = v6.build_live_capture_authority_v6(plan_envelope_file_sha256=v6.EXPECTED_PLAN_ENVELOPE_FILE_SHA256, hardware_file_sha256=v6.EXPECTED_HARDWARE_FILE_SHA256)
    assert envelope["actual_submitted_pcm_sha256"] == v6.EXPECTED_PCM_SHA256
    receipt = envelope["support_1024_condition_receipt"]
    assert receipt["canonical_payload_sha256"] == v6.EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
    assert receipt["joint_fit_condition_number"] == pytest.approx(1.9649539063087111)
    assert v6.validate_live_capture_authority_v6(authority) == authority
    assert authority["canonical_training_eligible"] is False
    assets = v6.asset_payloads_v6(hardware_file_sha256=v6.EXPECTED_HARDWARE_FILE_SHA256)
    assert set(assets) == {
        v6.SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
        v6.SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
    }
    for relative, raw in assets.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    plan = v6.load_exact_saved_plan_v6(tmp_path / v6.SEALED_PLAN_ENVELOPE_RELATIVE_PATH, repository_root=tmp_path, expected_file_sha256=v6.EXPECTED_PLAN_ENVELOPE_FILE_SHA256)
    loaded = v6.load_exact_saved_live_capture_authority_v6(tmp_path / v6.SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH, repository_root=tmp_path, expected_file_sha256=v6.EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256, expected_payload_sha256=v6.EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256)
    assert plan["pcm_sha256"] == v6.EXPECTED_PCM_SHA256
    assert plan["condition_receipt_payload_sha256"] == v6.EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
    assert loaded["authority"]["signal_plan_envelope"]["condition_receipt_payload_sha256"] == v6.EXPECTED_CONDITION_RECEIPT_PAYLOAD_SHA256
    assert loaded["authority"] == authority


def test_v6_pinned_mutation_and_v5_splice_rejected(tmp_path: Path):
    assets = v6.asset_payloads_v6(hardware_file_sha256=v6.EXPECTED_HARDWARE_FILE_SHA256)
    target = tmp_path / v6.SEALED_PLAN_ENVELOPE_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(assets[v6.SEALED_PLAN_ENVELOPE_RELATIVE_PATH] + b" ")
    with pytest.raises(ValueError, match="file SHA"):
        v6.load_exact_saved_plan_v6(target, repository_root=tmp_path, expected_file_sha256=v6.EXPECTED_PLAN_ENVELOPE_FILE_SHA256)

    authority = v6.build_live_capture_authority_v6(
        plan_envelope_file_sha256=v6.EXPECTED_PLAN_ENVELOPE_FILE_SHA256,
        hardware_file_sha256=v6.EXPECTED_HARDWARE_FILE_SHA256,
    )
    for mutate in (
        lambda item: item.__setitem__("scope", "attacker_scope"),
        lambda item: item.__setitem__("hardware_sample_slip_authority", True),
        lambda item: item["signal_plan_envelope"].__setitem__("path", "assets/contracts/fullband_causal_v5_signal_plan.json"),
        lambda item: item["hardware"].__setitem__("path", "../outside.yaml"),
    ):
        changed = copy.deepcopy(authority)
        mutate(changed)
        core = {key: value for key, value in changed.items() if key != "authority_sha256"}
        changed["authority_sha256"] = v6.payload_sha256(core)
        with pytest.raises(ValueError, match="exact pinned|pinned authority"):
            v6.validate_live_capture_authority_v6(changed)
    mutated = v6.committed_plan_envelope_v6()
    mutated["support_1024_condition_receipt"]["joint_fit_condition_number"] = 1.0
    target.write_bytes(v6.canonical_json_bytes(mutated, pretty=True))
    with pytest.raises(ValueError, match="file SHA"):
        v6.load_exact_saved_plan_v6(target, repository_root=tmp_path, expected_file_sha256=v6.EXPECTED_PLAN_ENVELOPE_FILE_SHA256)
    with pytest.raises(ValueError):
        v6.build_live_capture_authority_v6(plan_envelope_file_sha256="1" * 64, hardware_file_sha256=v6.EXPECTED_HARDWARE_FILE_SHA256)
    target.write_bytes(Path("assets/contracts/fullband_causal_v5_signal_plan.json").read_bytes())
    with pytest.raises(ValueError, match="file SHA"):
        v6.load_exact_saved_plan_v6(target, repository_root=tmp_path, expected_file_sha256=v6.EXPECTED_PLAN_ENVELOPE_FILE_SHA256)


def test_v6_meter_namespace_is_side_by_side():
    confirmations = {key: True for key in meter.CONFIRMATION_KEYS}
    contract = {
        "plan": {"path": meter.DEFAULT_PLAN_ENVELOPE_PATH},
        "live_capture_authority": {"path": meter.DEFAULT_LIVE_AUTHORITY_PATH},
        "hardware": {"path": "configs/hardware_jetson.yaml"},
        "level_evidence": {"path": "assets/measured/measurement_level_evidence.json"},
        "sealed_raw": {"path": meter.DEFAULT_RAW_TARGET_PATH, "must_not_exist_before_capture": True},
    }
    followup = meter.build_fullband_v6_followup(contract, resolved_devices={"input": 1, "output": 2}, confirmations=confirmations)
    assert followup["schema"] == meter.FOLLOWUP_SCHEMA
    assert "v6" in followup["signal_plan"]["path"]
    with pytest.raises(ValueError):
        meter.build_fullband_v6_followup(contract, resolved_devices={"input": 1, "output": 2}, confirmations={**confirmations, "speaker_output": False})
