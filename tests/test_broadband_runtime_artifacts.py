from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import deep_anc.eval.broadband_runtime as runtime
from deep_anc.dsp.control_band_contract import ControlBandContract
from deep_anc.realtime.clock_telemetry import payload_sha256, sha256_file


FS = 48_000
BLOCK = 256
CALLBACK_COUNT = 5_625


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _time_domain() -> dict[str, object]:
    return {
        "finite_count": CALLBACK_COUNT,
        "missing_or_nonfinite_count": 0,
        "strict_monotonic_violation_count": 0,
        "frame_step_violation_count": 0,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical_json(value) + "\n", encoding="utf-8")


def _write_physical_receipt(
    path: Path,
    *,
    plan: Path,
    session: Path,
    clock: Path,
    hardware_sha256: str,
    clock_sha256: str | None = None,
) -> None:
    # ``independent_clock_authority_pass=true``를 일부러 넣는다. 현재 schema에는
    # 독립 electrical raw가 없으므로 producer가 이 self-claim을 무시하고 false로
    # 유도하는지가 이 통합 회귀의 핵심이다.
    evidence = {
        "schema_version": "runtime_physical_clock_witness_v1",
        "status": "CONDITIONAL_PASS",
        "conditional_physical_timing_pass": True,
        "independent_clock_authority_pass": True,
        "synthetic_fixture": False,
        "plan_path": str(plan.resolve()),
        "plan_file_sha256": sha256_file(plan),
        "session_npz_path": str(session.resolve()),
        "clock_receipt_path": str(clock.resolve()),
        "session_npz_sha256": sha256_file(session),
        "clock_receipt_file_sha256": clock_sha256 or sha256_file(clock),
        "hardware_fingerprint_sha256": hardware_sha256,
        "clock_fit": {"segment_stationarity": {"sample_slip_count": 0}},
    }
    evidence_sha = _sha256_bytes(
        (_canonical_json(evidence) + "\n").encode("utf-8")
    )
    evidence["evidence_sha256"] = evidence_sha
    _write_json(
        path,
        {
            "schema_version": "runtime_physical_clock_witness_bundle_v1",
            "status": "CONDITIONAL_PASS",
            "evidence_sha256": evidence_sha,
            "evidence": evidence,
        },
    )


def _artifact_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    checkpoint = tmp_path / "model.pt"
    artifact = tmp_path / "model.onnx"
    metadata = tmp_path / "model.json"
    checkpoint.write_bytes(b"checkpoint-fixture")
    artifact.write_bytes(b"onnx-fixture")
    metadata.write_bytes(b"metadata-fixture")

    config = {
        "controller": "dl",
        "reference": "digital",
        "digital_reference_lead_samples": 115,
        "engine": {
            "type": "ort",
            "ckpt": str(checkpoint),
            "onnx": str(artifact),
        },
    }
    plant = SimpleNamespace(
        timing=SimpleNamespace(
            handoff_samples=BLOCK,
            digital_reference_lead_samples=115,
        ),
        primary_path_sha256="6" * 64,
        secondary_path_sha256="7" * 64,
    )
    contract = ControlBandContract.broadband_point_control()
    identity = runtime.RuntimeDeploymentIdentity(
        model_name="hybrid_anc_tiny",
        engine="ort",
        experiment_contract_sha256="1" * 64,
        control_band_contract_sha256=contract.digest(),
        training_timing_contract_sha256="2" * 64,
        checkpoint_path=str(checkpoint.resolve()),
        checkpoint_sha256=sha256_file(checkpoint),
        checkpoint_lead_samples=115,
        deployment_artifact_path=str(artifact.resolve()),
        deployment_artifact_sha256=sha256_file(artifact),
        deployment_metadata_path=str(metadata.resolve()),
        deployment_metadata_sha256=sha256_file(metadata),
        deployment_lead_samples=115,
        primary_path_sha256=plant.primary_path_sha256,
        secondary_path_sha256=plant.secondary_path_sha256,
    )
    monkeypatch.setattr(
        runtime, "validate_runtime_plant_contract", lambda _cfg: plant
    )
    monkeypatch.setattr(
        runtime,
        "verify_runtime_deployment_identity",
        lambda **_kwargs: identity,
    )

    deployment_snapshot = runtime.snapshot_runtime_deployment_files(
        runtime_cfg=config,
        plant=plant,
        repo_root=tmp_path,
    )
    fingerprint_unsigned = {
        "schema": "alsa_physical_hardware_fingerprint_v1",
        "input": {"fixture": "APE:1"},
        "output": {"fixture": "Audio:0"},
    }
    hardware_sha = _sha256_bytes(
        _canonical_json(fingerprint_unsigned).encode("utf-8")
    )
    fingerprint = {**fingerprint_unsigned, "sha256": hardware_sha}

    telemetry = {
        "schema_version": "realtime_clock_telemetry_v1",
        "authority_status": "INCONCLUSIVE",
        "structural_status": "PASS",
        "sample_rate": FS,
        "block_size": BLOCK,
        "callback_summary": {
            "callback_count": CALLBACK_COUNT,
            "completed_callback_count": CALLBACK_COUNT,
            "incomplete_callback_count": 0,
            "pending_callback_count": 0,
            "portaudio_status_callback_count": 0,
            "callback_host_deadline_miss_count": 0,
            "omitted_callback_record_count": 0,
            "stored_callback_record_count": CALLBACK_COUNT,
            "application_observed_frames": CALLBACK_COUNT * BLOCK,
            "application_observed_seconds": 30.0,
        },
        "runtime_counters_final": {
            "xrun_count": 0,
            "deadline_miss_count": 0,
            "engine_error_blocks": 0,
            "input_ring_drop_samples": 0,
            "output_ring_drop_samples": 0,
            "input_ring_overrun_blocks": 0,
            "output_ring_overrun_blocks": 0,
            "input_ring_underrun_blocks": 0,
            "output_ring_underrun_blocks": 0,
            "ring_add_samples": 0,
            "fallback_silence_blocks": 0,
            "watchdog_trip_counts": {},
        },
        "maximum_input_backlog_samples": BLOCK,
        "maximum_output_backlog_samples": BLOCK,
        "allowed_input_backlog_samples": BLOCK,
        "allowed_output_backlog_samples": BLOCK,
        "time_domains": {
            "input_buffer_adc_time": _time_domain(),
            "output_buffer_dac_time": _time_domain(),
            "callback_current_time": _time_domain(),
        },
        "issue_counts": {},
        "callbacks": [{"completed": True} for _ in range(CALLBACK_COUNT)],
    }
    telemetry_sha = payload_sha256(telemetry)
    session = tmp_path / "runtime.npz"
    snapshot_json = _canonical_json(deployment_snapshot)
    fingerprint_json = _canonical_json(fingerprint)
    with session.open("xb") as handle:
        np.savez_compressed(
            handle,
            fs=np.asarray(FS),
            runtime_clock_telemetry_sha256=np.asarray(telemetry_sha),
            runtime_clock_authority_status=np.asarray("INCONCLUSIVE"),
            inference_step_times_ms=np.full(
                CALLBACK_COUNT - 1, 1.25, dtype=np.float64
            ),
            inference_step_count=np.asarray(CALLBACK_COUNT - 1),
            intentional_startup_prime_blocks=np.asarray(1),
            runtime_deployment_snapshot_json=np.asarray(snapshot_json),
            runtime_deployment_snapshot_sha256=np.asarray(
                deployment_snapshot["snapshot_sha256"]
            ),
            runtime_physical_fingerprint_json=np.asarray(fingerprint_json),
            runtime_physical_fingerprint_sha256=np.asarray(hardware_sha),
        )
    clock = tmp_path / "runtime.runtime_clock.json"
    _write_json(
        clock,
        {
            "schema_version": "realtime_clock_receipt_bundle_v1",
            "authority_status": "INCONCLUSIVE",
            "runtime_clock_telemetry_sha256": telemetry_sha,
            "recording_npz": str(session.resolve()),
            "recording_npz_sha256": sha256_file(session),
            "runtime_clock_telemetry": telemetry,
        },
    )
    plan = tmp_path / "runtime.physical_clock.plan.json"
    _write_json(plan, {"fixture": "predeclared-before-capture"})
    physical = tmp_path / "runtime.physical_clock.json"
    _write_physical_receipt(
        physical,
        plan=plan,
        session=session,
        clock=clock,
        hardware_sha256=hardware_sha,
    )
    original_physical_evidence = json.loads(
        physical.read_text(encoding="utf-8")
    )["evidence"]
    monkeypatch.setattr(
        runtime,
        "audit_runtime_physical_clock_witness",
        lambda **_kwargs: dict(original_physical_evidence),
    )
    return {
        "artifact": artifact,
        "clock": clock,
        "config": config,
        "contract": contract,
        "physical": physical,
        "plan": plan,
        "plant": plant,
        "session": session,
        "hardware_sha256": hardware_sha,
    }


def _build(chain: dict[str, object]) -> runtime.BroadbandRuntimeEvidence:
    return runtime.build_broadband_runtime_evidence_from_artifacts(
        contract=chain["contract"],
        runtime_cfg=chain["config"],
        session_npz_path=chain["session"],
        clock_receipt_path=chain["clock"],
        physical_witness_receipt_path=chain["physical"],
        power_mode="MAXN",
        repo_root=Path(chain["session"]).parent,
    )


def test_valid_artifact_chain_reaches_only_independent_clock_blocker(
    tmp_path, monkeypatch
):
    chain = _artifact_chain(tmp_path, monkeypatch)

    evidence = _build(chain)
    audit = runtime.audit_broadband_runtime_evidence(
        chain["contract"],
        evidence,
        expected_plant_lead_samples=115,
    )

    assert evidence.conditional_physical_timing_pass is True
    assert evidence.independent_clock_authority_pass is False
    assert evidence.callback_count == CALLBACK_COUNT
    assert evidence.inference_max_ms == 1.25
    assert not audit.ok
    assert len(audit.reasons) == 1
    assert "electrical clock" in audit.reasons[0]


def test_session_replacement_is_rejected_by_clock_bound_sha(tmp_path, monkeypatch):
    chain = _artifact_chain(tmp_path, monkeypatch)
    with Path(chain["session"]).open("ab") as handle:
        handle.write(b"replacement")

    with pytest.raises(ValueError, match="clock receipt.*session SHA"):
        _build(chain)


def test_resigned_physical_receipt_cannot_point_to_other_clock(tmp_path, monkeypatch):
    chain = _artifact_chain(tmp_path, monkeypatch)
    _write_physical_receipt(
        chain["physical"],
        plan=chain["plan"],
        session=chain["session"],
        clock=chain["clock"],
        hardware_sha256=chain["hardware_sha256"],
        clock_sha256="f" * 64,
    )

    with pytest.raises(ValueError, match="physical witness clock receipt SHA"):
        _build(chain)


def test_deployment_artifact_replacement_is_rejected_by_session_snapshot(
    tmp_path, monkeypatch
):
    chain = _artifact_chain(tmp_path, monkeypatch)
    Path(chain["artifact"]).write_bytes(b"replaced-onnx-fixture")

    with pytest.raises(ValueError, match="deployment start/end/current snapshot"):
        _build(chain)


def test_session_path_swap_between_hash_and_npz_parse_is_rejected(
    tmp_path, monkeypatch
):
    chain = _artifact_chain(tmp_path, monkeypatch)
    real_load = runtime.np.load
    attacked = False

    def replace_session_then_load(source, *args, **kwargs):
        nonlocal attacked
        if not attacked:
            attacked = True
            replacement = tmp_path / "attacker-session.npz"
            replacement.write_bytes(b"different-session-bytes")
            replacement.replace(chain["session"])
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(runtime.np, "load", replace_session_then_load)
    with pytest.raises(ValueError, match="runtime session NPZ path/bytes"):
        _build(chain)


def test_clock_path_swap_after_json_parse_is_rejected(tmp_path, monkeypatch):
    chain = _artifact_chain(tmp_path, monkeypatch)
    real_payload_sha256 = runtime.payload_sha256
    attacked = False

    def replace_clock_then_hash(payload):
        nonlocal attacked
        if not attacked:
            attacked = True
            replacement = tmp_path / "attacker-clock.json"
            _write_json(replacement, {"attacker": True})
            replacement.replace(chain["clock"])
        return real_payload_sha256(payload)

    monkeypatch.setattr(runtime, "payload_sha256", replace_clock_then_hash)
    with pytest.raises(ValueError, match="runtime clock receipt path/bytes"):
        _build(chain)


def test_physical_receipt_path_swap_after_parse_is_rejected(
    tmp_path, monkeypatch
):
    chain = _artifact_chain(tmp_path, monkeypatch)
    recompute = runtime.audit_runtime_physical_clock_witness

    def replace_physical_then_recompute(**kwargs):
        replacement = tmp_path / "attacker-physical.json"
        _write_json(replacement, {"attacker": True})
        replacement.replace(chain["physical"])
        return recompute(**kwargs)

    monkeypatch.setattr(
        runtime,
        "audit_runtime_physical_clock_witness",
        replace_physical_then_recompute,
    )
    with pytest.raises(
        ValueError, match="runtime physical clock witness receipt path/bytes"
    ):
        _build(chain)


def test_resigned_physical_self_attestation_must_match_raw_recomputation(
    tmp_path, monkeypatch
):
    chain = _artifact_chain(tmp_path, monkeypatch)
    bundle = json.loads(Path(chain["physical"]).read_text(encoding="utf-8"))
    evidence = dict(bundle["evidence"])
    evidence["clock_fit"] = {
        "segment_stationarity": {"sample_slip_count": 1}
    }
    unsigned = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    evidence_sha = _sha256_bytes(
        (_canonical_json(unsigned) + "\n").encode("utf-8")
    )
    evidence["evidence_sha256"] = evidence_sha
    bundle["evidence"] = evidence
    bundle["evidence_sha256"] = evidence_sha
    _write_json(Path(chain["physical"]), bundle)

    with pytest.raises(ValueError, match="raw session/clock/plan 재계산"):
        _build(chain)
