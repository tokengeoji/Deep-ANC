from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from deep_anc.dsp import fullband_live_post_v6 as post
from deep_anc.dsp import fullband_live_delay_core_v6 as core
from deep_anc.dsp.fullband_live_authority_v6 import SEALED_RAW_RELATIVE_PATH


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_archival_loader_is_separate_and_can_never_return_analysis_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def fake_common(**kwargs):  # noqa: ANN003, ANN202
        calls.append(kwargs.pop("require_current_meter_execution"))
        assert kwargs
        return {
            "receipt_file_sha256": "a" * 64,
            "receipt": {
                "schema": post.EXTERNAL_POST_RECEIPT_SCHEMA,
                "status": "POST_CAPTURE_PASS",
                "valid": True,
                "receipt_payload_sha256": "b" * 64,
            },
            "raw": {"metadata": {}},
        }

    monkeypatch.setattr(post, "_load_external_post_capture_receipt_v6", fake_common)
    values = {
        "repository_root": "/tmp/repository",
        "receipt_relative_path": "results/raw.npz.post_receipt.json",
        "expected_receipt_file_sha256": "a" * 64,
        "plan_envelope_path": "assets/plan.json",
        "live_authority_path": "assets/authority.json",
        "meter_raw_path": "results/meter.npz",
        "level_evidence_path": "assets/evidence.json",
        "hardware_path": "configs/hardware.yaml",
    }
    normal = post.load_external_post_capture_receipt_v6(**values)
    assert normal["receipt"]["valid"] is True
    archival = post._load_external_post_capture_receipt_v6_archival_forensics(**values)
    assert calls == [True, False]
    assert set(archival) == {
        "schema",
        "scope",
        "analysis_admission_eligible",
        "canonical_training_eligible",
        "source_receipt_evidence",
        "forensic_raw_snapshot",
    }
    assert "receipt" not in archival and "raw" not in archival
    assert archival["analysis_admission_eligible"] is False
    assert archival["canonical_training_eligible"] is False
    assert archival["scope"] == (
        "archival_forensics_only_no_analysis_no_plant_no_training_authority"
    )


def _bound_files(root: Path, *, schema: str = post.EXTERNAL_POST_RECEIPT_SCHEMA):
    raw = root / SEALED_RAW_RELATIVE_PATH
    raw.parent.mkdir(parents=True)
    raw_bytes = b"immutable-v6-raw"
    raw.write_bytes(raw_bytes)
    receipt_relative = post.external_post_receipt_relative_path(SEALED_RAW_RELATIVE_PATH)
    receipt = root / receipt_relative
    receipt_value = {
        "schema": schema,
        "raw": {"path": SEALED_RAW_RELATIVE_PATH, "file_sha256": _sha(raw_bytes)},
    }
    receipt.write_text(json.dumps(receipt_value), encoding="utf-8")
    return raw, receipt, _sha(raw_bytes), _sha(receipt.read_bytes())


def _publish(root: Path, **overrides):
    raw, receipt, raw_sha, receipt_sha = _bound_files(root)
    values = dict(
        repository_root=root,
        failure_relative_path="results/fullband_causal_v6/run/failure.json",
        raw_relative_path=SEALED_RAW_RELATIVE_PATH,
        raw_file_sha256=raw_sha,
        external_receipt_relative_path=receipt.relative_to(root).as_posix(),
        external_receipt_file_sha256=receipt_sha,
        failure_stage="joint_delay_fit",
        optimizer_started=True,
        error="optimizer did not converge",
        available_snr_receipt={"schema": "snr_receipt_v1", "snr_db": 12.5},
    )
    values.update(overrides)
    return post.publish_live_delay_failure_v6(**values)


def test_failure_is_sha_bound_immutable_and_emits_no_analysis(tmp_path: Path):
    result = _publish(tmp_path)
    value = json.loads(result["path"].read_text(encoding="utf-8"))
    assert value["schema"] == post.FAILURE_SCHEMA
    assert value["analysis_published"] is False
    assert value["operator_published"] is False
    assert not (result["path"].parent / "analysis.json").exists()
    assert not (result["path"].parent / "operator.npz").exists()
    with pytest.raises(FileExistsError):
        post.publish_live_delay_failure_v6(
            repository_root=tmp_path,
            failure_relative_path=result["relative_path"],
            raw_relative_path=value["raw"]["path"],
            raw_file_sha256=value["raw"]["file_sha256"],
            external_receipt_relative_path=value["external_post_receipt"]["path"],
            external_receipt_file_sha256=value["external_post_receipt"]["file_sha256"],
            failure_stage="again", optimizer_started=False, error="again",
        )


def test_wrong_sha_and_v5_splice_are_rejected_without_output(tmp_path: Path):
    with pytest.raises(ValueError, match="raw SHA"):
        _publish(tmp_path, raw_file_sha256="0" * 64)
    assert not (tmp_path / "results/fullband_causal_v6/run/failure.json").exists()

    other = tmp_path / "splice"
    with pytest.raises(ValueError, match="receipt schema"):
        raw, receipt, raw_sha, receipt_sha = _bound_files(
            other, schema="fullband_causal_v5_external_post_capture_receipt_v1"
        )
        post.publish_live_delay_failure_v6(
            repository_root=other,
            failure_relative_path="results/fullband_causal_v6/run/failure.json",
            raw_relative_path=SEALED_RAW_RELATIVE_PATH,
            raw_file_sha256=raw_sha,
            external_receipt_relative_path=receipt.relative_to(other).as_posix(),
            external_receipt_file_sha256=receipt_sha,
            failure_stage="admission", optimizer_started=False, error="splice",
        )


def test_symlink_parent_and_outside_path_are_rejected(tmp_path: Path):
    raw, receipt, raw_sha, receipt_sha = _bound_files(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "results/fullband_causal_v6/out").symlink_to(outside, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        post.publish_live_delay_failure_v6(
            repository_root=tmp_path,
            failure_relative_path="results/fullband_causal_v6/out/run/failure.json",
            raw_relative_path=SEALED_RAW_RELATIVE_PATH,
            raw_file_sha256=raw_sha,
            external_receipt_relative_path=receipt.relative_to(tmp_path).as_posix(),
            external_receipt_file_sha256=receipt_sha,
            failure_stage="publish", optimizer_started=False, error="symlink",
        )
    assert not (outside / "run/failure.json").exists()
    with pytest.raises(ValueError):
        post.external_post_receipt_relative_path("../v6_raw.npz")


def _success_publish_values(root: Path) -> dict:
    receipt_sha = "a" * 64
    captured_sha = "b" * 64
    submitted_sha = "c" * 64
    operator = {
        "primary_compact_fir_by_mic": np.zeros((2, 1_024), dtype="<f8"),
        "secondary_compact_fir_by_mic": np.zeros((2, 1_024), dtype="<f8"),
        "primary_zeros_before_fir": np.asarray(244, dtype="<i8"),
        "secondary_zeros_before_fir": np.asarray(164, dtype="<i8"),
        "support_samples": np.asarray(1_024, dtype="<i8"),
        "separate_fractional_phase_applications": np.asarray(0, dtype="<i8"),
    }
    operator_receipt = {
        "schema": core.OPERATOR_RECEIPT_SCHEMA,
        "canonical_training_eligible": False,
        "hardware_sample_slip_authority_available": False,
        "captured_adc_pcm_sha256": captured_sha,
        "actual_submitted_pcm_sha256": submitted_sha,
        "operator_array_sha256": {
            name: core._array_sha256(value) for name, value in sorted(operator.items())
        },
    }
    operator_receipt["canonical_payload_sha256"] = core._payload_sha256(
        operator_receipt
    )
    operator["receipt"] = operator_receipt
    analysis = {
        "schema": core.ANALYSIS_SCHEMA,
        "status": "OFFLINE_MATH_PASS_RAW_PUBLISHER_AUTHORITY_UNBOUND",
        "canonical_training_eligible": False,
        "hardware_slip_authority_available": False,
        "captured_raw_binding": {
            "captured_adc_pcm_sha256": captured_sha,
            "actual_submitted_pcm_sha256": submitted_sha,
        },
        "final_fixed_average": {"operator_receipt": operator_receipt},
    }
    analysis["analysis_sha256"] = core._payload_sha256(analysis)
    execution = {
        "repository_commit": "d" * 40,
        "repository_branch": "work/test-v6",
        "repository_dirty": False,
        "adapter_path": "scripts/data/measure_paths_fullband_causal_v6.py",
        "adapter_file_sha256": "e" * 64,
    }
    return {
        "repository_root": root,
        "output_directory_relative_path": "results/fullband_causal_v6/success",
        "external_receipt_relative_path": post.external_post_receipt_relative_path(
            SEALED_RAW_RELATIVE_PATH
        ),
        "external_receipt_file_sha256": receipt_sha,
        "plan_envelope_path": "assets/contracts/fullband_causal_v6_signal_plan.json",
        "live_authority_path": "assets/contracts/fullband_causal_v6_live_capture_authority.json",
        "meter_raw_path": "results/fullband_causal_v6/level_meter/meter_raw.npz",
        "level_evidence_path": "assets/measured/measurement_level_evidence.json",
        "hardware_path": "configs/hardware_jetson.yaml",
        "analysis_execution_identity": execution,
        "analysis": analysis,
        "operator": operator,
    }


def _install_success_recompute_stubs(
    monkeypatch: pytest.MonkeyPatch,
    values: dict,
    *,
    recomputed_analysis: dict | None = None,
    recomputed_operator: dict | None = None,
) -> list[dict]:
    called: list[dict] = []
    execution = values["analysis_execution_identity"]
    callback_count = 1
    arrays = {
        "actual_submitted_pcm": np.zeros((256, 2), dtype="<i2"),
        "captured_pcm": np.zeros((256, 2), dtype="<i4"),
        "capture_valid_mask": np.ones(256, dtype=np.bool_),
        "submitted_valid_mask": np.ones(256, dtype=np.bool_),
        "callback_sequence": np.arange(callback_count, dtype="<i8"),
        "callback_start_frames": np.arange(callback_count, dtype="<i8") * 256,
        "callback_frame_counts": np.full(callback_count, 256, dtype="<i8"),
        "input_buffer_adc_time": np.arange(callback_count, dtype="<f8"),
        "output_buffer_dac_time": np.arange(callback_count, dtype="<f8"),
        "callback_current_time": np.arange(callback_count, dtype="<f8"),
        "callback_status_bitmask": np.zeros(callback_count, dtype="<u4"),
    }

    def admitted(**kwargs):  # noqa: ANN003, ANN202
        called.append(kwargs)
        return {
            "receipt_file_sha256": values["external_receipt_file_sha256"],
            "raw": {
                "metadata": {
                    "array_sha256": {
                        "captured_pcm": "b" * 64,
                        "actual_submitted_pcm": "c" * 64,
                    },
                    "session": dict(execution),
                    "duplex_telemetry_scalars": {},
                },
                "arrays": arrays,
            },
        }

    monkeypatch.setattr(post, "load_external_post_capture_receipt_v6", admitted)
    monkeypatch.setattr(
        post,
        "repository_execution_identity",
        lambda _root, script: {
            "repository_commit": execution["repository_commit"],
            "repository_branch": execution["repository_branch"],
            "repository_dirty": False,
            "script_path": script,
            "script_file_sha256": execution["adapter_file_sha256"],
        },
    )
    monkeypatch.setattr(
        post,
        "committed_plan_envelope_v6",
        lambda: {"signal_plan": {"fixture": "committed-v6"}},
    )
    expected_analysis = (
        copy.deepcopy(values["analysis"])
        if recomputed_analysis is None
        else recomputed_analysis
    )
    expected_operator = (
        copy.deepcopy(values["operator"])
        if recomputed_operator is None
        else recomputed_operator
    )

    def recompute(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["plan"] == {"fixture": "committed-v6"}
        assert kwargs["submitted_pcm"] is arrays["actual_submitted_pcm"]
        assert kwargs["captured_adc_pcm"] is arrays["captured_pcm"]
        assert kwargs["duplex_telemetry"]["callback_sequence"] is arrays["callback_sequence"]
        return expected_analysis, expected_operator

    monkeypatch.setattr(post, "analyze_committed_v6_live_delay", recompute)
    # 이 fixture의 축약 analysis는 core exact schema 전체가 아니므로, 이 파일은
    # publisher의 admission/recompute 경계만 격리한다.
    monkeypatch.setattr(
        post,
        "validate_analysis_operator_v6",
        lambda *_args, **_kwargs: {"passed": True},
    )
    return called


def test_success_publisher_requires_receipt_raw_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _success_publish_values(tmp_path)
    called = _install_success_recompute_stubs(monkeypatch, values)
    result = post.publish_live_delay_analysis_v6(**values)
    assert called and result["analysis"]["path"].is_file()
    assert result["operator"]["path"].is_file()


def test_success_publisher_rejects_recomputed_analysis_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _success_publish_values(tmp_path)
    recomputed_analysis = copy.deepcopy(values["analysis"])
    recomputed_analysis["status"] = "RECOMPUTED_DIFFERENT"
    recomputed_analysis["analysis_sha256"] = core._payload_sha256(
        {
            key: value
            for key, value in recomputed_analysis.items()
            if key != "analysis_sha256"
        }
    )
    _install_success_recompute_stubs(
        monkeypatch,
        values,
        recomputed_analysis=recomputed_analysis,
        recomputed_operator=copy.deepcopy(values["operator"]),
    )
    with pytest.raises(ValueError, match="독립 재계산"):
        post.publish_live_delay_analysis_v6(**values)
    assert not (tmp_path / values["output_directory_relative_path"]).exists()


def test_success_publisher_rejects_self_consistent_fir_splice_after_recompute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _success_publish_values(tmp_path)
    recomputed_analysis = copy.deepcopy(values["analysis"])
    recomputed_operator = copy.deepcopy(values["operator"])

    values["operator"]["primary_compact_fir_by_mic"][0, 0] = 1.0
    receipt = values["operator"]["receipt"]
    receipt["operator_array_sha256"]["primary_compact_fir_by_mic"] = (
        core._array_sha256(values["operator"]["primary_compact_fir_by_mic"])
    )
    receipt["canonical_payload_sha256"] = core._payload_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "canonical_payload_sha256"
        }
    )
    values["analysis"]["final_fixed_average"]["operator_receipt"] = copy.deepcopy(
        receipt
    )
    values["analysis"]["analysis_sha256"] = core._payload_sha256(
        {
            key: value
            for key, value in values["analysis"].items()
            if key != "analysis_sha256"
        }
    )
    _install_success_recompute_stubs(
        monkeypatch,
        values,
        recomputed_analysis=recomputed_analysis,
        recomputed_operator=recomputed_operator,
    )
    with pytest.raises(ValueError, match="독립 재계산"):
        post.publish_live_delay_analysis_v6(**values)
    assert not (tmp_path / values["output_directory_relative_path"]).exists()


def test_success_publisher_requires_current_clean_exact_adapter_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _success_publish_values(tmp_path)
    _install_success_recompute_stubs(monkeypatch, values)
    monkeypatch.setattr(
        post,
        "repository_execution_identity",
        lambda _root, script: {
            "repository_commit": "f" * 40,
            "repository_branch": values["analysis_execution_identity"][
                "repository_branch"
            ],
            "repository_dirty": False,
            "script_path": script,
            "script_file_sha256": values["analysis_execution_identity"][
                "adapter_file_sha256"
            ],
        },
    )
    with pytest.raises(ValueError, match="current clean exact v6 adapter checkout"):
        post.publish_live_delay_analysis_v6(**values)
    assert not (tmp_path / values["output_directory_relative_path"]).exists()


def test_success_publisher_writes_nothing_when_receipt_admission_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _success_publish_values(tmp_path)

    def rejected(**_kwargs):  # noqa: ANN003, ANN202
        raise ValueError("receipt rejected")

    monkeypatch.setattr(post, "load_external_post_capture_receipt_v6", rejected)
    with pytest.raises(ValueError, match="receipt rejected"):
        post.publish_live_delay_analysis_v6(**values)
    assert not (tmp_path / values["output_directory_relative_path"]).exists()


def _receipt_witness_fixture() -> tuple[dict, dict]:
    confirmations = {
        "speaker_output": True,
        "user_present": True,
        "volume_minimum": True,
        "routing_and_geometry": True,
        "same_amplifier_setting": True,
    }
    primitive = {"schema": "fullband_causal_v6_post_capture_binding_v1", "valid": True}
    devices = {"input": 3, "output": 7}
    lock_identity = {
        "path": "results/.live_audio_uid_1000.lock",
        "pid": 123,
        "uid": 1000,
        "purpose": "fullband_causal_v6_live_capture",
        "device": 9,
        "inode": 10,
    }
    lock_sha = post.audio_lock_identity_sha256(lock_identity)
    receipt = {
        "operator_confirmations": confirmations,
        "primitive_post_capture_binding": primitive,
        "resolved_devices": devices,
        "audio_lock": {
            **lock_identity,
            "identity_sha256": lock_sha,
            "exclusive_lock_observed": True,
        },
    }
    metadata = {
        "operator_confirmations": confirmations,
        "post_capture_binding": primitive,
        "bindings": {"hardware": {"resolved_devices": devices}},
        "duplex_telemetry_scalars": {
            "resolved_input_device": 3,
            "resolved_output_device": 7,
        },
        "session": {"audio_lock_identity_sha256": lock_sha},
    }
    return receipt, metadata


def test_receipt_witness_must_match_raw_session_and_telemetry() -> None:
    receipt, metadata = _receipt_witness_fixture()
    post._validate_receipt_witness_against_raw_metadata(receipt, metadata)

    mutations = (
        lambda item: item["operator_confirmations"].__setitem__("speaker_output", False),
        lambda item: item.__setitem__("primitive_post_capture_binding", {"valid": False}),
        lambda item: item.__setitem__("resolved_devices", {"input": 4, "output": 7}),
        lambda item: item["audio_lock"].__setitem__("purpose", "v5"),
        lambda item: item["audio_lock"].__setitem__("exclusive_lock_observed", False),
        lambda item: item["audio_lock"].__setitem__("identity_sha256", "0" * 64),
    )
    for mutate in mutations:
        changed = json.loads(json.dumps(receipt))
        mutate(changed)
        with pytest.raises(ValueError):
            post._validate_receipt_witness_against_raw_metadata(changed, metadata)


def _meter_repository_execution() -> dict[str, object]:
    return {
        "repository_commit": "1" * 40,
        "repository_branch": "work/v6-clock-checkpoints",
        "repository_dirty": False,
        "script_path": "scripts/data/set_amp_level.py",
        "script_file_sha256": "2" * 64,
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("repository_commit", "3" * 40),
        ("repository_branch", "tampered-branch"),
        ("repository_dirty", True),
        ("script_path", "scripts/data/not_set_amp_level.py"),
        ("script_file_sha256", "4" * 64),
    ],
)
def test_offline_meter_repository_execution_mutation_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    current = _meter_repository_execution()
    saved = dict(current)
    saved[field] = replacement

    def current_identity(repository_root: Path, script_path: str) -> dict[str, object]:
        assert repository_root == tmp_path
        assert script_path == "scripts/data/set_amp_level.py"
        return dict(current)

    monkeypatch.setattr(post, "repository_execution_identity", current_identity)

    with pytest.raises(ValueError, match="current clean checkout"):
        post._validate_offline_meter_repository_execution_v6(
            {"repository_execution": saved}, repository_root=tmp_path
        )


def test_offline_meter_repository_execution_requires_exact_present_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = _meter_repository_execution()
    monkeypatch.setattr(
        post,
        "repository_execution_identity",
        lambda root, script: dict(current),
    )

    assert post._validate_offline_meter_repository_execution_v6(
        {"repository_execution": dict(current)}, repository_root=tmp_path
    ) == current
    with pytest.raises(ValueError, match="current clean checkout"):
        post._validate_offline_meter_repository_execution_v6(
            {}, repository_root=tmp_path
        )

    with_extra = dict(current)
    with_extra["unexpected"] = "splice"
    with pytest.raises(ValueError, match="current clean checkout"):
        post._validate_offline_meter_repository_execution_v6(
            {"repository_execution": with_extra}, repository_root=tmp_path
        )
