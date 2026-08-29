from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from deep_anc.data.broadband_coverage_receipt import (
    _validate_recorded_v2_session_binding,
)

from deep_anc.data.recorded_v2_capture import (
    CLOCK_FIT_BAND_HZ,
    FULLBAND_CAUSAL_PLANT_SCHEMA,
    RECORDED_V2_LIVE_AUTHORITY,
    RECORDED_V2_TIMEWARP_SCHEMA,
    SOURCE_FRAMES,
    RecordedV2Blocked,
    capture_contract,
    file_reference,
    publish_raw_capture_noreplace,
    publish_aligned_session_noreplace,
    recompute_actual_err_coverage,
    render_submitted_pcm,
    sha256_file,
    submitted_pcm_evidence,
    validate_fullband_causal_plant,
    validate_raw_capture_bundle,
    validate_stored_actual_err_coverage,
    validate_timewarp_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _seal(payload: dict) -> dict:
    value = copy.deepcopy(payload)
    value.pop("evidence_sha256", None)
    value["evidence_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return value


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _occupancy(tmp_path: Path, label: str) -> dict:
    payload = _seal(
        {
            "schema": "recorded_v2_audio_occupancy_witness_v1",
            "checks": [
                {
                    "stage": stage,
                    "proc_pcm_status": [
                        {
                            "path": "/proc/asound/card1/pcm1c/sub0/status",
                            "status": "closed",
                        }
                    ],
                    "fuser_pcm_owners": [],
                }
                for stage in ("before_input_probe", "before_output_open")
            ],
            "all_pcm_closed": True,
        }
    )
    path = tmp_path / f"occupancy-{label}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {
        "checked_before_input_probe": True,
        "checked_before_output_open": True,
        "all_pcm_closed": True,
        "evidence_file": file_reference(path, repository_root=tmp_path),
    }


def _load_script(name: str):
    path = REPO_ROOT / "scripts/data/record_broadband_v2.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_capture_contract_is_exact_15_seconds_and_live_stays_locked():
    contract = capture_contract()
    assert contract["source_seconds"] == 15.0
    assert contract["source_frames"] == 720_000 == SOURCE_FRAMES
    assert contract["clock"]["fit_band_hz"] == list(CLOCK_FIT_BAND_HZ)
    assert contract["clock"]["highband_used_for_clock_fit"] is False
    assert contract["clock"]["highband_phase_repair_samples"] == 0.0
    assert RECORDED_V2_LIVE_AUTHORITY is None


def test_dry_run_without_verified_source_and_plant_is_blocked_without_audio_or_writes(
    tmp_path, monkeypatch, capsys
):
    module = _load_script("record_broadband_v2_missing_test")
    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise AssertionError("dry-run이 sounddevice를 import했습니다")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    out_root = tmp_path / "must_not_exist"
    result = module.main(
        [
            "--dry-run",
            "--source-plan",
            str(tmp_path / "missing_plan.json"),
            "--plant-evidence",
            str(tmp_path / "missing_plant.json"),
            "--expected-source-plan-sha256",
            _sha("plan"),
            "--expected-plant-sha256",
            _sha("plant"),
            "--out-root",
            str(out_root),
        ]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert "[BLOCKED]" in captured.err
    assert "sounddevice import/open 0회" in captured.err
    assert not out_root.exists()


def test_execute_live_requires_confirmations_before_any_device_check(tmp_path, monkeypatch):
    module = _load_script("record_broadband_v2_confirm_test")
    monkeypatch.setattr(
        module,
        "_assert_read_only_audio_unoccupied",
        lambda: (_ for _ in ()).throw(AssertionError("device gate까지 진행함")),
    )
    assert module.main(["--execute-live"]) == 2


def test_actual_submitted_int16_pcm_sha_is_exact_and_source_sensitive(tmp_path):
    rate = 48_000
    frames = SOURCE_FRAMES + 128
    time = np.arange(frames, dtype=np.float64) / rate
    source = 0.25 * np.sin(2.0 * np.pi * 997.0 * time)
    wav = tmp_path / "source.wav"
    sf.write(wav, source, rate, subtype="FLOAT")
    mono, stereo = render_submitted_pcm(wav, start_frame=64, gain_q15=16_384)
    evidence = submitted_pcm_evidence(mono, stereo)
    assert mono.shape == (720_000,)
    assert stereo.shape == (720_000, 2)
    assert stereo.dtype == np.dtype("<i2")
    assert np.all(stereo[:, 1] == 0)
    assert evidence["mono_pcm_sha256"] == hashlib.sha256(
        mono.astype("<i2", copy=False).tobytes()
    ).hexdigest()
    changed = source.copy()
    changed[64 + 20_000] += 0.02
    sf.write(wav, changed, rate, subtype="FLOAT")
    changed_mono, changed_stereo = render_submitted_pcm(
        wav, start_frame=64, gain_q15=16_384
    )
    changed_evidence = submitted_pcm_evidence(changed_mono, changed_stereo)
    assert changed_evidence["mono_pcm_sha256"] != evidence["mono_pcm_sha256"]
    assert changed_evidence["stereo_interleaved_pcm_sha256"] != evidence[
        "stereo_interleaved_pcm_sha256"
    ]
    unsafe_mono, unsafe_stereo = render_submitted_pcm(
        wav, start_frame=64, gain_q15=32_767
    )
    with pytest.raises(ValueError, match="안전 상한"):
        submitted_pcm_evidence(unsafe_mono, unsafe_stereo)


def test_diagnostic_or_synthetic_plant_cannot_open_recorded_v2(tmp_path):
    plant = _seal(
        {
            "schema": FULLBAND_CAUSAL_PLANT_SCHEMA,
            "role": "diagnostic_only",
            "status": "BLOCKED",
            "canonical_training_eligible": False,
            "synthetic_fixture": True,
            "control_band_contract_sha256": _sha("control"),
            "sample_rate_hz": 48_000,
            "block_size": 256,
            "excitation_band_hz": [150.0, 11_400.0],
            "persistently_exciting_causal_history": False,
            "same_capture_ps": False,
            "raw_capture": {"path": "raw", "size_bytes": 1, "sha256": _sha("raw")},
            "analysis": {"path": "analysis", "size_bytes": 1, "sha256": _sha("analysis")},
            "primary_path": {"path": "p", "size_bytes": 1, "sha256": _sha("p")},
            "secondary_path": {"path": "s", "size_bytes": 1, "sha256": _sha("s")},
            "absolute_dac_q_timewarp": {
                "path": "warp",
                "size_bytes": 1,
                "sha256": _sha("warp"),
            },
            "training_timing_contract_sha256": _sha("timing"),
        }
    )
    path = tmp_path / "plant.json"
    path.write_text(json.dumps(plant, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(RecordedV2Blocked, match="canonical 자격"):
        validate_fullband_causal_plant(
            path,
            expected_file_sha256=sha256_file(path),
            repository_root=tmp_path,
        )


def _timewarp_receipt(
    tmp_path: Path,
    *,
    anchors: dict[str, str] | None = None,
    maximum_q: float = 204_000.0,
) -> tuple[dict, dict[str, str]]:
    adc = np.linspace(0.0, maximum_q, 18, dtype=np.float64).astype("<f8")
    dac = adc * (1.0 if anchors is not None else 1.0 + 20.0e-6) + (
        0.0 if anchors is not None else 37.25
    )
    knots = tmp_path / "knots.npz"
    np.savez(knots, adc_frame_knots=adc, dac_q_knots=dac)
    adc_sha = hashlib.sha256(adc.tobytes()).hexdigest()
    dac_sha = hashlib.sha256(dac.tobytes()).hexdigest()
    map_payload = {
        "method": "monotone_piecewise_cubic_absolute_adc_to_dac_q_v1",
        "knots_file_sha256": sha256_file(knots),
        "adc_frame_knots_sha256": adc_sha,
        "dac_q_knots_sha256": dac_sha,
    }
    map_sha = hashlib.sha256(_canonical(map_payload)).hexdigest()
    anchors = anchors or {
        "raw": _sha("raw-capture"),
        "submitted": _sha("submitted-pcm"),
        "mics": _sha("mics-pcm"),
    }
    receipt = _seal(
        {
            "schema": RECORDED_V2_TIMEWARP_SCHEMA,
            "role": "physical_raw_offline_alignment",
            "raw_capture_sha256": anchors["raw"],
            "submitted_pcm_sha256": anchors["submitted"],
            "mics_pcm_sha256": anchors["mics"],
            "fit_band_hz": list(CLOCK_FIT_BAND_HZ),
            "highband_used_for_clock_fit": False,
            "highband_phase_repair_samples": 0.0,
            "fit_window_parity": "even",
            "holdout_window_parity": "odd",
            "holdout_used_for_fit_or_selection": False,
            "common_map": {
                "method": map_payload["method"],
                "knots_file": {
                    "path": "knots.npz",
                    "size_bytes": knots.stat().st_size,
                    "sha256": sha256_file(knots),
                },
                "adc_frame_knots_sha256": adc_sha,
                "dac_q_knots_sha256": dac_sha,
                "map_sha256": map_sha,
                "leaveout_max_samples": 0.02,
                "cubic_crosscheck_max_samples": 0.004,
                "combined_max_samples": 0.024,
            },
            "witnesses": [
                {
                    "role": role,
                    "common_map_sha256": map_sha,
                    "fit_windows": 9,
                    "holdout_windows": 9,
                    "maximum_residual_samples": 0.02,
                    "minimum_score": 0.999,
                }
                for role in ("P_submitted_playback", "ERR_ch0", "REF_ch1")
            ],
            "callback_witness": {
                "role": "monotonic_and_slip_witness_only",
                "monotonic": True,
                "sample_slip_count": 0,
            },
            "sample_slip_count": 0,
            "xrun_count": 0,
        }
    )
    return receipt, anchors


def test_timewarp_requires_one_lowband_only_common_p_err_ref_map(tmp_path):
    receipt, anchors = _timewarp_receipt(tmp_path)
    result = validate_timewarp_receipt(
        receipt,
        repository_root=tmp_path,
        expected_raw_capture_sha256=anchors["raw"],
        expected_submitted_pcm_sha256=anchors["submitted"],
        expected_mics_pcm_sha256=anchors["mics"],
    )
    assert result["status"] == "PASS_STRUCTURAL_OFFLINE_ONLY"
    assert result["n_knots"] == 18
    assert RECORDED_V2_LIVE_AUTHORITY is None

    highband = copy.deepcopy(receipt)
    highband["highband_used_for_clock_fit"] = True
    highband = _seal(highband)
    with pytest.raises(ValueError, match="고역"):
        validate_timewarp_receipt(
            highband,
            repository_root=tmp_path,
            expected_raw_capture_sha256=anchors["raw"],
            expected_submitted_pcm_sha256=anchors["submitted"],
            expected_mics_pcm_sha256=anchors["mics"],
        )

    split_map = copy.deepcopy(receipt)
    split_map["witnesses"][1]["common_map_sha256"] = _sha("other-map")
    split_map = _seal(split_map)
    with pytest.raises(ValueError, match="같은 absolute DAC-q map"):
        validate_timewarp_receipt(
            split_map,
            repository_root=tmp_path,
            expected_raw_capture_sha256=anchors["raw"],
            expected_submitted_pcm_sha256=anchors["submitted"],
            expected_mics_pcm_sha256=anchors["mics"],
        )


def test_actual_err_seven_band_coverage_is_recomputed_from_arrays_and_tamper_fails():
    rng = np.random.default_rng(20260828)
    source = rng.normal(0.0, 0.02, SOURCE_FRAMES).astype(np.float32)
    mics = np.column_stack(
        [source, rng.normal(0.0, 0.001, SOURCE_FRAMES).astype(np.float32)]
    )
    rows = recompute_actual_err_coverage(source, mics)
    result = validate_stored_actual_err_coverage(
        copy.deepcopy(rows), source_aligned=source, mics=mics
    )
    assert result["segments"] == 9
    assert result["all_seven_band_joint_pass_segments"] == 9
    assert result["status"] == "PASS"
    tampered = copy.deepcopy(rows)
    tampered[0]["target_density_ratio"][-1] += 0.1
    with pytest.raises(ValueError, match="actual WAV 재계산"):
        validate_stored_actual_err_coverage(
            tampered, source_aligned=source, mics=mics
        )


def test_raw_capture_is_published_before_analysis_and_never_replaced(tmp_path):
    submitted = np.zeros((SOURCE_FRAMES, 2), dtype="<i2")
    mics = np.zeros((SOURCE_FRAMES, 2), dtype="<i4")
    callback = np.array(
        [
            [0.0, 0.01, 256.0, 0.0, 0.0, 0.0],
            [0.01, 0.02, 256.0, 0.0, 256.0, 256.0],
        ],
        dtype="<f8",
    )
    plan = tmp_path / "plan.json"
    plant = tmp_path / "plant.json"
    plan.write_text("{}\n", encoding="utf-8")
    plant.write_text("{}\n", encoding="utf-8")
    source_region = submitted[:SOURCE_FRAMES]
    metadata = {
        "capture_id": "capture-1",
        "capture_contract_sha256": capture_contract()["contract_sha256"],
        "source_plan": file_reference(plan, repository_root=tmp_path),
        "source_plan_row_sha256": _sha("plan-row"),
        "plant_evidence": file_reference(plant, repository_root=tmp_path),
        "sample_rate_hz": 48_000,
        "block_size": 256,
        "source_output_start_frame": 0,
        "submitted_source_pcm_sha256": hashlib.sha256(
            source_region[:, 0].tobytes()
        ).hexdigest(),
        "submitted_source_stereo_pcm_sha256": hashlib.sha256(
            source_region.tobytes()
        ).hexdigest(),
        "xrun_count": 0,
        "clip_count": 0,
        "sample_slip_count": 0,
        "safety_confirmations": {
            "user_present": True,
            "volume_minimum": True,
            "routing_and_geometry": True,
        },
        "device_occupancy_witness": _occupancy(tmp_path, "capture"),
    }
    target = tmp_path / "raw" / "capture-1"
    result = publish_raw_capture_noreplace(
        repository_root=tmp_path,
        target_directory=target,
        submitted_output_pcm=submitted,
        mics_raw_pcm=mics,
        callback_time_info=callback,
        receipt_metadata=metadata,
    )
    assert result == target
    receipt = json.loads((target / "raw_receipt.json").read_text(encoding="utf-8"))
    assert receipt["role"] == "immutable_raw_before_analysis"
    assert receipt["raw_published_before_analysis"] is True
    assert receipt["analysis_started"] is False
    validated = validate_raw_capture_bundle(
        target, repository_root=tmp_path, require_valid_for_analysis=True
    )
    assert validated["status"] == "PASS"
    with pytest.raises(FileExistsError, match="덮어쓰지"):
        publish_raw_capture_noreplace(
            repository_root=tmp_path,
            target_directory=target,
            submitted_output_pcm=submitted,
            mics_raw_pcm=mics,
            callback_time_info=callback,
            receipt_metadata=metadata,
        )
    bad_callback = callback.copy()
    bad_callback[1, 3] = 1.0
    bad_metadata = copy.deepcopy(metadata)
    bad_metadata["capture_id"] = "capture-xrun-lie"
    with pytest.raises(ValueError, match="재계산과 다릅니다"):
        publish_raw_capture_noreplace(
            repository_root=tmp_path,
            target_directory=tmp_path / "raw" / "capture-xrun-lie",
            submitted_output_pcm=submitted,
            mics_raw_pcm=mics,
            callback_time_info=bad_callback,
            receipt_metadata=bad_metadata,
        )


def test_aligned_session_is_derived_after_raw_and_coverage_is_recomputed(tmp_path):
    rng = np.random.default_rng(280828)
    guard = 2_048
    total = SOURCE_FRAMES + 2 * guard
    submitted = np.zeros((total, 2), dtype="<i2")
    mono = rng.integers(-2_000, 2_001, SOURCE_FRAMES, dtype=np.int16)
    submitted[guard : guard + SOURCE_FRAMES, 0] = mono
    mics = np.zeros((total, 2), dtype="<i4")
    mics_values = mono.astype(np.int32) * np.int32(65_536)
    mics[guard : guard + SOURCE_FRAMES, 0] = mics_values
    mics[guard : guard + SOURCE_FRAMES, 1] = mics_values
    callback = np.array(
        [
            [0.0, 0.01, 256.0, 0.0, 0.0, 0.0],
            [15.0, 15.01, 256.0, 0.0, 256.0, 256.0],
        ],
        dtype="<f8",
    )
    plan = tmp_path / "plan.json"
    plant = tmp_path / "plant.json"
    plan.write_text("{}\n", encoding="utf-8")
    plant.write_text("{}\n", encoding="utf-8")
    plan_row_sha = _sha("session-plan-row")
    raw_target = tmp_path / "raw" / "capture-session"
    raw_metadata = {
        "capture_id": "capture-session",
        "capture_contract_sha256": capture_contract()["contract_sha256"],
        "source_plan": file_reference(plan, repository_root=tmp_path),
        "source_plan_row_sha256": plan_row_sha,
        "plant_evidence": file_reference(plant, repository_root=tmp_path),
        "sample_rate_hz": 48_000,
        "block_size": 256,
        "source_output_start_frame": guard,
        "submitted_source_pcm_sha256": hashlib.sha256(mono.tobytes()).hexdigest(),
        "submitted_source_stereo_pcm_sha256": hashlib.sha256(
            submitted[guard : guard + SOURCE_FRAMES].tobytes()
        ).hexdigest(),
        "xrun_count": 0,
        "clip_count": 0,
        "sample_slip_count": 0,
        "safety_confirmations": {
            "user_present": True,
            "volume_minimum": True,
            "routing_and_geometry": True,
        },
        "device_occupancy_witness": _occupancy(tmp_path, "session"),
    }
    publish_raw_capture_noreplace(
        repository_root=tmp_path,
        target_directory=raw_target,
        submitted_output_pcm=submitted,
        mics_raw_pcm=mics,
        callback_time_info=callback,
        receipt_metadata=raw_metadata,
    )
    raw_bundle = validate_raw_capture_bundle(
        raw_target, repository_root=tmp_path, require_valid_for_analysis=True
    )
    anchors = {
        "raw": raw_bundle["raw_capture_sha256"],
        "submitted": raw_bundle["submitted_output_pcm_sha256"],
        "mics": raw_bundle["mics_raw_pcm_sha256"],
    }
    warp, _ = _timewarp_receipt(
        tmp_path, anchors=anchors, maximum_q=float(total - 1)
    )
    warp_path = tmp_path / "timewarp.json"
    warp_path.write_text(json.dumps(warp, ensure_ascii=False), encoding="utf-8")
    session_target = tmp_path / "sessions" / "session-v2-001"
    identity = {
        "session_id": "session-v2-001",
        "split": "train",
        "source_family": "speech",
        "group_id": "speech-group-v2-001",
        "lineage_id": "speech-lineage-v2-001",
        "source_plan_row_sha256": plan_row_sha,
        "native_source_sha256": _sha("native-source"),
        "processed_source_sha256": _sha("processed-source"),
        "transform_receipt_sha256": _sha("transform-receipt"),
    }
    published = publish_aligned_session_noreplace(
        repository_root=tmp_path,
        target_directory=session_target,
        raw_capture_directory=raw_target,
        timewarp_receipt_path=warp_path,
        session_identity=identity,
    )
    assert published == session_target
    session = json.loads((session_target / "session.json").read_text(encoding="utf-8"))
    coverage = json.loads((session_target / "coverage.json").read_text(encoding="utf-8"))
    assert session["role"] == "canonical_aligned_after_immutable_raw"
    assert coverage["summary"]["status"] == "PASS"
    assert coverage["summary"]["all_seven_band_joint_pass_segments"] >= 8
    rebound = _validate_recorded_v2_session_binding(
        file_reference(session_target / "session.json", repository_root=tmp_path),
        root=tmp_path,
        session_id=identity["session_id"],
        split=identity["split"],
        family=identity["source_family"],
        group=identity["group_id"],
        lineage=identity["lineage_id"],
        raw_native_sha256=identity["native_source_sha256"],
        processed_sha256=identity["processed_source_sha256"],
        transform_receipt_sha256=identity["transform_receipt_sha256"],
        mics_reference=file_reference(
            session_target / "mics.wav", repository_root=tmp_path
        ),
    )
    assert len(rebound) == 7
    assert all(len(rows) == 9 for rows in rebound)
    with pytest.raises(FileExistsError, match="덮어쓰지"):
        publish_aligned_session_noreplace(
            repository_root=tmp_path,
            target_directory=session_target,
            raw_capture_directory=raw_target,
            timewarp_receipt_path=warp_path,
            session_identity=identity,
        )
