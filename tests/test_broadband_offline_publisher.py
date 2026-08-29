"""광대역 raw-first 오프라인 분석/발행을 오디오 없이 검증한다."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

from deep_anc.audio_io import analyze_int32_input_probe
from deep_anc.dsp.control_band_contract import ControlBandContract
from deep_anc.dsp.measured_band_path import load_measured_band_path
from deep_anc.train.criterion_factory import admit_criterion_config
from scripts.data import analyse_broadband_interleaved as analyzer
from scripts.data import measure_paths_broadband_interleaved as measurement
from scripts.data import measure_paths_interleaved as mpi


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rewrite_raw_metadata(session: Path, mutate) -> None:  # noqa: ANN001
    raw_path = session / "raw_measurement.npz"
    with np.load(raw_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
            if name != "metadata_json"
        }
    mutate(metadata)
    encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    with raw_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(encoded),
            **arrays,
        )
    (session / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _raw_pattern(frames: int, offset: int = 0) -> np.ndarray:
    index = np.arange(frames, dtype=np.int64)
    first = ((index + offset) % 31 - 15) * 100_000
    second = ((index * 3 + offset + 5) % 37 - 18) * 90_000
    return np.stack((first, second), axis=1).astype(np.int32)


def _write_mock_raw_session(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tamper_submitted: bool = False,
) -> dict[str, object]:
    session = root / "results" / "broadband" / "capture"
    assets = root / "assets" / "measured"
    configs = root / "configs"
    evidence_dir = root / "results" / "evidence"
    for path in (session, assets, configs, evidence_dir):
        path.mkdir(parents=True, exist_ok=True)

    hardware = configs / "hardware.yaml"
    hardware.write_text(
        """audio:
  sample_rate: 48000
  block_size: 256
  latency: low
  input: {card: mock-input, pcm: 1, channels: 2}
  output: {card: mock-output, pcm: 0, channels: 2}
channels:
  error_mic: 0
  reference_mic: 1
  noise_out: 0
  cancel_out: 1
""",
        encoding="utf-8",
    )
    contract = ControlBandContract.broadband_point_control()
    repeats = 9
    period_samples = 4
    frames_per_panel = repeats * period_samples
    layout = []
    cursor = 0
    for index in range(5):
        layout.append(
            {
                "kind": "analysis_panel",
                "start_frame": cursor,
                "stop_frame": cursor + frames_per_panel,
                "frames": frames_per_panel,
                "panel_index": index,
            }
        )
        cursor += frames_per_panel
    planned = np.zeros((cursor, 2), dtype=np.int16)
    pcm_sha = hashlib.sha256(planned.tobytes(order="C")).hexdigest()
    plan = {
        "schema": measurement.BROADBAND_MEASUREMENT_PLAN_SCHEMA,
        "role": "signal_only_dry_run_no_audio",
        "live_capture_enabled": False,
        "control_band_contract": contract.model_dump(mode="json"),
        "control_band_contract_sha256": contract.digest(),
        "hardware": {
            "path": str(hardware.resolve()),
            "sha256": _sha(hardware),
            "sample_rate": 48_000,
            "block_size": 256,
            "latency": "low",
            "channels": {
                "error_mic": 0,
                "reference_mic": 1,
                "noise_out": 0,
                "cancel_out": 1,
            },
        },
        "recipe": {
            "amplitude": 0.003,
            "period_seconds": 0.125,
            "repeats_per_panel": repeats,
        },
        "panels": [],
        "layout": layout,
        "output": {
            "frames": cursor,
            "channels": 2,
            "dtype": "int16",
            "duration_seconds": cursor / 48_000.0,
            "padding_frames": 0,
            "peak_pcm": 0,
            "pcm_sha256": pcm_sha,
        },
    }
    plan_path = root / "results" / "broadband" / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    level_meter = evidence_dir / "level_meter.npz"
    level_interleaved = evidence_dir / "level_interleaved.npz"
    level_meter.write_bytes(b"immutable-level-meter")
    level_interleaved.write_bytes(b"immutable-level-interleaved")
    level_path = evidence_dir / "level.json"
    level_path.write_text("{}\n", encoding="utf-8")
    meter_path = evidence_dir / "fresh_meter.npz"
    meter_path.write_bytes(b"immutable-fresh-meter")
    meter_receipt = evidence_dir / "fresh_meter.receipt.json"
    meter_receipt.write_text("{}\n", encoding="utf-8")

    identity = {
        "physical_fingerprint": {
            "input": "mock-input",
            "output": "mock-output",
        },
        "channel_map": {
            "error_mic": 0,
            "reference_mic": 1,
            "noise_out": 0,
            "cancel_out": 1,
        },
    }
    completed = dt.datetime(2026, 8, 28, tzinfo=dt.timezone.utc)
    capture_started = completed + dt.timedelta(seconds=10)
    stream_started = completed + dt.timedelta(seconds=13)
    capture_completed = completed + dt.timedelta(seconds=14)
    input_raw = _raw_pattern(cursor)
    preflight_raw = _raw_pattern(96_000, offset=11)
    preflight = analyze_int32_input_probe(preflight_raw)
    preflight.update({"device": 7, "sample_rate": 48_000, "settle_seconds": 1.0})
    submitted = planned.copy()
    if tamper_submitted:
        submitted[3, 1] = 1
    callback_starts = np.arange(0, cursor, 36, dtype=np.int64)
    callback_counts = np.full(callback_starts.size, 36, dtype=np.int64)
    callback_times = 1.0 + np.arange(callback_starts.size, dtype=np.float64) * 0.001
    callback_time_info = {
        "callback_start_frames": callback_starts,
        "callback_frame_counts": callback_counts,
        "input_buffer_adc_time": callback_times,
        "output_buffer_dac_time": callback_times + 0.0001,
        "callback_current_time": callback_times + 0.0002,
    }
    callback_summary = measurement.validate_callback_time_info(
        callback_time_info, expected_frames=cursor
    )
    metadata = {
        "capture_id": "mock-broadband-capture",
        "started_at_utc": capture_started.isoformat(),
        "completed_at_utc": capture_completed.isoformat(),
        "raw_capture_schema": measurement.BROADBAND_RAW_CAPTURE_SCHEMA,
        "method": measurement.BROADBAND_METHOD,
        "status": "PASS",
        "valid": True,
        "invalid_reasons": [],
        "analysis_status": "NOT_RUN_RAW_FIRST",
        "post_capture_binding": {"valid": True, "error": None},
        "sample_rate": 48_000,
        "block_size": 256,
        "latency": "low",
        "channel_map": {
            "error_mic": 0,
            "reference_mic": 1,
            "noise_out": 0,
            "cancel_out": 1,
        },
        "operator_confirmations": {
            "speaker_output": True,
            "user_present": True,
            "volume_minimum": True,
            "routing_and_geometry": True,
            "same_amplifier_setting": True,
        },
        "hardware_identity": identity,
        "hardware": {"path": str(hardware.resolve()), "sha256": _sha(hardware)},
        "resolved_devices": {"input": 7, "output": 9},
        "input_preflight_seconds": 3.0,
        "preflight": preflight,
        "telemetry": {
            "xrun_count": 0,
            "unexpected_status_count": 0,
            "callback_error": None,
            "stream_abort_error": None,
            "stream_close_error": None,
            "output_stop_confirmed": True,
            "completed": True,
            "captured_frames": cursor,
            "termination_signal": None,
            "stream_started_at_utc": stream_started.isoformat(),
            "nominal_output_seconds": cursor / 48_000.0,
            "hard_max_output_seconds": (
                cursor / 48_000.0 + mpi.LIVE_WATCHDOG_GRACE_SECONDS
            ),
        },
        "plan": {
            "path": str(plan_path.resolve()),
            "file_sha256": _sha(plan_path),
            "payload_sha256": _canonical_sha(plan),
            "pcm_sha256": pcm_sha,
            "schema": measurement.BROADBAND_MEASUREMENT_PLAN_SCHEMA,
        },
        "control_band_contract_sha256": contract.digest(),
        "meter": {
            "path": str(meter_path.resolve()),
            "receipt_path": str(meter_receipt.resolve()),
            "raw_sha256": _sha(meter_path),
            "completed_at_utc": completed.isoformat(),
            "meter_ch0_dbfs": -50.0,
            "freshness_max_seconds": measurement.BOOTSTRAP_METER_MAX_AGE_SECONDS,
        },
        "level_evidence": {
            "path": str(level_path.resolve()),
            "sha256": _sha(level_path),
        },
        "submitted_pcm_sha256": hashlib.sha256(
            submitted.tobytes(order="C")
        ).hexdigest(),
        "callback_timing": callback_summary,
    }
    encoded_metadata = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    with (session / "raw_measurement.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(encoded_metadata),
            submitted_output_pcm_int16=submitted,
            input_raw_int32=input_raw,
            preflight_raw_int32=preflight_raw,
            **callback_time_info,
        )
    (session / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        analyzer.broadband_measure,
        "build_signal_plan",
        lambda **kwargs: (plan, planned.copy()),
    )
    monkeypatch.setattr(
        analyzer.broadband_measure,
        "validate_live_authority_binding",
        lambda saved_plan, submitted_pcm: {
            "path": str(plan_path.relative_to(root)),
            "file_sha256": _sha(plan_path),
            "payload_sha256": _canonical_sha(plan),
            "pcm_sha256": pcm_sha,
        }
        if (
            Path(saved_plan["path"]).resolve() == plan_path.resolve()
            and saved_plan["file_sha256"] == _sha(plan_path)
            and saved_plan["payload_sha256"] == _canonical_sha(plan)
            and saved_plan["payload"] == plan
            and np.array_equal(submitted_pcm, planned)
        )
        else (_ for _ in ()).throw(ValueError("mock live authority mismatch")),
    )
    monkeypatch.setattr(
        analyzer,
        "load_measurement_level_evidence",
        lambda path, repository_root: {
            "_evidence_sha256": _sha(level_path),
            "hardware_identity": identity,
            "meter_raw": {
                "path": str(level_meter.relative_to(root)),
                "sha256": _sha(level_meter),
            },
            "interleaved_raw": {
                "path": str(level_interleaved.relative_to(root)),
                "sha256": _sha(level_interleaved),
            },
        },
    )
    monkeypatch.setattr(
        analyzer,
        "validate_bootstrap_meter_raw",
        lambda *args, **kwargs: {
            "path": meter_path,
            "receipt_path": meter_receipt,
            "sha256": _sha(meter_path),
            "metadata": {"resolved_devices": {"input": 7, "output": 9}},
            "meter_ch0_dbfs": -50.0,
            "completed_at_utc": completed,
        },
    )

    def validate_meter_followup(**kwargs):
        assert kwargs["plan_binding"] == {
            "path": str(plan_path.resolve()),
            "file_sha256": _sha(plan_path),
            "payload_sha256": _canonical_sha(plan),
            "pcm_sha256": pcm_sha,
        }
        assert Path(kwargs["hardware_path"]).resolve() == hardware.resolve()
        assert kwargs["hardware_sha256"] == _sha(hardware)
        assert Path(kwargs["level_evidence_path"]).resolve() == level_path.resolve()
        assert kwargs["level_evidence_sha256"] == _sha(level_path)
        assert Path(kwargs["raw_session_dir"]).resolve() == session.resolve()
        return {"status": "PASS"}

    monkeypatch.setattr(
        analyzer.broadband_measure,
        "validate_meter_followup_binding",
        validate_meter_followup,
    )
    return {
        "root": root,
        "session": session,
        "assets": assets,
        "hardware": hardware,
        "plan_path": plan_path,
        "plan": plan,
        "planned": planned,
    }


class _FakeProbe:
    period_samples = 4


def _mock_panel(panel_band: tuple[float, float]) -> dict[str, object]:
    repeats = 9
    valid = np.asarray([True] * 8 + [False])
    scores = np.asarray([0.999] * 8 + [np.nan])
    frequencies = {
        "noise": np.linspace(panel_band[0], panel_band[1], 8),
        "cancel": np.linspace(panel_band[0] + 1.0, panel_band[1] - 1.0, 8),
    }
    transfers = {
        drive: np.ones(8, dtype=np.complex128) for drive in ("noise", "cancel")
    }
    stacks = {
        drive: np.ones((8, 8), dtype=np.complex128)
        for drive in ("noise", "cancel")
    }
    return {
        "panel_band_hz": panel_band,
        "frequencies": frequencies,
        "transfers": transfers,
        "crosscheck_transfers": {
            drive: value.copy() for drive, value in transfers.items()
        },
        "aligned_stacks": stacks,
        "aligned_crosscheck_stacks": {
            drive: value.copy() for drive, value in stacks.items()
        },
        "panel_consistency": {"noise": 0.999, "cancel": 0.999},
        "relative_tau_max_abs_samples": 0.0,
        "separation": {
            "valid": valid,
            "q": np.ones(repeats),
            "common_delay_samples": np.zeros(repeats),
            "err_delay_samples": np.zeros(repeats),
            "ref_delay_samples": np.zeros(repeats),
            "err_score": scores,
            "ref_score": scores,
            "err_subwindow_spread_samples": np.zeros(repeats),
            "ref_subwindow_spread_samples": np.zeros(repeats),
            "err_ref_delta_samples": np.zeros(repeats),
            "adjacent_change_samples": np.zeros(repeats),
            "drift_deviation_samples": np.zeros(repeats),
            "drift_samples_per_period": 0.0,
            "drift_ppm": 0.0,
        },
        "selection": {
            "keep": valid,
            "anchor": 0,
            "common_alignment_taus": np.zeros(repeats),
            "relative_tau_kept": np.zeros(8),
        },
    }


def _mock_stitched() -> dict[str, object]:
    contract = ControlBandContract.broadband_point_control()
    rows = tuple(
        {
            "band_hz": list(band),
            "tone_count": 16,
            "complex_agreement": 0.9999,
            "relative_error": 0.001,
            "passed": True,
        }
        for band in contract.point_control_subbands_hz
    )
    frequency = np.arange(104.0, 11_400.0 + 0.1, 16.0, dtype=np.float64)
    drive_results = {}
    for drive, bulk in (("noise", 500), ("cancel", 450)):
        fractional_bulk = float(bulk) + 0.125
        measured = np.exp(-2j * np.pi * frequency * fractional_bulk / 48_000.0)
        fir = np.zeros(1024, dtype=np.float32)
        fir[0] = 1.0
        identifiability = {
            "schema_version": "compact_fir_identifiability_diagnostic_v1",
            "compact_role": "diagnostic_only",
            "compact_training_eligible": False,
            "numeric_rank": 523,
            "condition_number": 1.0e15,
            "reason": "design_matrix_not_identifiable",
        }
        drive_results[drive] = {
            "frequencies_hz": frequency,
            "mean_transfer": measured,
            "observations_per_frequency": np.ones(frequency.size, dtype=np.int64),
            "bulk_delay_fractional_samples": fractional_bulk,
            "bulk_delay_samples": bulk,
            "effective_delay_samples": bulk - 256,
            "fractional_effective_delay_samples": fractional_bulk - 256.0,
            "pre_roll_samples": 256,
            "fir": fir,
            "compact": {
                "passed": True,
                "reconstructed_transfer": measured.copy(),
            },
            "compact_subbands": rows,
            "compact_role": "diagnostic_only",
            "compact_training_eligible": False,
            "compact_identifiability": identifiability,
            "compact_identifiability_sha256": hashlib.sha256(
                json.dumps(
                    identifiability,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "measured_interpolation_holdout": {
                "schema_version": "every_other_tone_seven_subband_holdout_v1",
                "passed": True,
                "rows": list(rows),
            },
            "measured_interpolation_subbands": rows,
            "separation_subbands": rows,
        }
    return {
        "status": "PASS",
        "control_band_contract_sha256": contract.digest(),
        "panel_stitch": [
            {"passed": True, "shared_delay_samples": 0.0} for _ in range(4)
        ],
        "drives": drive_results,
        "primary_consistency": (0.999,) * 7,
        "secondary_consistency": (0.999,) * 7,
        "relative_phase_jitter_samples": (0.0,) * 7,
        "panel_bulk_delay_fractional_samples": {
            "noise": (500.125,) * 5,
            "cancel": (450.125,) * 5,
        },
        "panel_primary_minus_secondary_bulk_delay_samples": (50.0,) * 5,
        "panel_relative_delay_deviation_samples": (0.0,) * 5,
    }


def _load_and_analyse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    fixture = _write_mock_raw_session(tmp_path, monkeypatch)
    capture = analyzer.load_broadband_raw_capture(
        session_dir=fixture["session"],
        plan_path=fixture["plan_path"],
        hardware_path=fixture["hardware"],
        repository_root=fixture["root"],
    )
    contract = ControlBandContract.broadband_point_control()
    panels = [_mock_panel(tuple(band)) for band in contract.measurement_panels_hz]
    starts = [np.arange(index * 36, index * 36 + 36, 4) for index in range(5)]
    global_starts = np.arange(45, dtype=np.int64) * 4
    global_offsets = np.zeros(global_starts.size, dtype=np.float64)
    analysis = {
        "panels": panels,
        "period_starts_by_panel": starts,
        "panel_summaries": [],
        "stitched": _mock_stitched(),
        "clock_valid_repeats": (8,) * 5,
        "clock_min_adjacent_score_observed": (0.999,) * 5,
        "separation_crosscheck_agreement": (0.9999,) * 7,
        "separation_crosscheck_relative_error": (0.001,) * 7,
        "compact_roundtrip_agreement": (0.9999,) * 7,
        "compact_roundtrip_relative_error": (0.001,) * 7,
        "measured_interpolation_agreement": (0.9999,) * 7,
        "measured_interpolation_relative_error": (0.001,) * 7,
        "timing_markers": {
            "noise": {
                "coarse_delay_samples": 500.0,
                "search_width_samples": 2000.0,
                "alias_candidate_count": 1,
            },
            "cancel": {
                "coarse_delay_samples": 450.0,
                "search_width_samples": 1450.0,
                "alias_candidate_count": 1,
            },
        },
        "timing_marker_pcm_sha256": "3" * 64,
        "fixed_clock_pilot_sha256": "4" * 64,
        "intended_float_pilot_sha256": "7" * 64,
        "submitted_pilot_validation": {
            "sha256": "6" * 64,
            "submitted_pilot_spectra_sha256": "4" * 64,
            "cross_channel_null_sha256": "8" * 64,
            "cross_channel_null_maximum_absolute_observed": 0.0,
            "cross_channel_null_maximum_ratio_observed": 0.0,
            "pairwise_trajectory_agreement_samples": 0.0,
            "highband_phase_used_for_map": False,
        },
        "global_clock_map_sha256": "5" * 64,
        "global_clock_map": {
            "period_starts": global_starts,
            "period_offsets_samples": global_offsets,
            "residual_samples": global_offsets,
            "slope_samples_per_sample": 0.0,
            "intercept_samples": 0.0,
            "maximum_residual_samples": 0.0,
        },
        "panel_clock_offsets_samples": (0.0,) * 5,
        "transition_anchor_valid_counts": (8, 8, 8, 8),
        "clock_trajectory_agreement_samples": 0.0,
        "applied_per_drive_phase_repair_samples": (0.0,) * 10,
    }
    return fixture, capture, analysis


def test_raw_loader_accepts_exact_broadband_schema_and_pcm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write_mock_raw_session(tmp_path, monkeypatch)
    capture = analyzer.load_broadband_raw_capture(
        session_dir=fixture["session"],
        plan_path=fixture["plan_path"],
        hardware_path=fixture["hardware"],
        repository_root=fixture["root"],
    )
    assert capture["xrun_count"] == 0
    assert capture["clip_count"] == 0
    assert capture["plan_pcm_sha256"] == hashlib.sha256(
        fixture["planned"].tobytes(order="C")
    ).hexdigest()


def test_raw_loader_rejects_one_code_submitted_pcm_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write_mock_raw_session(
        tmp_path, monkeypatch, tamper_submitted=True
    )
    with pytest.raises(ValueError, match="exact plan PCM"):
        analyzer.load_broadband_raw_capture(
            session_dir=fixture["session"],
            plan_path=fixture["plan_path"],
            hardware_path=fixture["hardware"],
            repository_root=fixture["root"],
        )


def test_raw_loader_rejects_missing_post_capture_toctou_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write_mock_raw_session(tmp_path, monkeypatch)

    def invalidate(metadata):
        metadata["post_capture_binding"] = {
            "valid": False,
            "error": "injected plan SHA change",
        }

    _rewrite_raw_metadata(fixture["session"], invalidate)
    with pytest.raises(ValueError, match="post-capture TOCTOU"):
        analyzer.load_broadband_raw_capture(
            session_dir=fixture["session"],
            plan_path=fixture["plan_path"],
            hardware_path=fixture["hardware"],
            repository_root=fixture["root"],
        )


@pytest.mark.parametrize(
    ("validator_name", "message"),
    (
        ("validate_live_authority_binding", "authority plan mismatch"),
        ("validate_meter_followup_binding", "meter followup mismatch"),
    ),
)
def test_raw_loader_rejects_unbound_authority_or_meter_followup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validator_name: str,
    message: str,
) -> None:
    fixture = _write_mock_raw_session(tmp_path, monkeypatch)

    def reject(*args, **kwargs):  # noqa: ANN002, ANN003
        raise ValueError(message)

    monkeypatch.setattr(analyzer.broadband_measure, validator_name, reject)
    with pytest.raises(ValueError, match=message):
        analyzer.load_broadband_raw_capture(
            session_dir=fixture["session"],
            plan_path=fixture["plan_path"],
            hardware_path=fixture["hardware"],
            repository_root=fixture["root"],
        )


def test_five_panel_raw_analysis_publishes_only_after_broadband_audit_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, capture, analysis = _load_and_analyse(tmp_path, monkeypatch)
    primary = fixture["assets"] / "primary_broadband.npz"
    secondary = fixture["assets"] / "secondary_broadband.npz"
    result = analyzer.publish_broadband_analysis(
        capture=capture,
        analysis=analysis,
        analysis_npz_path=fixture["session"] / "analysis.npz",
        analysis_json_path=fixture["session"] / "analysis.json",
        primary_path=primary,
        secondary_path=secondary,
        publish=True,
    )
    assert result["audit"].status == "PASS"
    assert all(path.is_file() for path in result["paths"].values())
    with np.load(primary, allow_pickle=False) as archive:
        assert str(archive["schema_version"].item()) == (
            analyzer.BROADBAND_PLANT_ARTIFACT_SCHEMA
        )
        assert str(archive["plant_role"].item()) == "primary"
        assert str(archive["compact_role"].item()) == "diagnostic_only"
        assert bool(archive["compact_training_eligible"].item()) is False
        assert archive["measured_frequencies_hz"].shape == archive[
            "measured_transfer_real"
        ].shape
        assert str(archive["source_analysis_npz_sha256"].item()) == result[
            "analysis_sha256"
        ]
        np.testing.assert_array_equal(archive["clock_valid_repeats"], [8] * 5)
        assert archive["verified_subbands_hz"].shape == (7, 2)
        embedded = str(archive["broadband_plant_evidence_json"].item())
        embedded_sha = hashlib.sha256(embedded.encode("utf-8")).hexdigest()
        assert str(archive["broadband_plant_evidence_sha256"].item()) == embedded_sha
        embedded_payload = json.loads(embedded)
        assert embedded_payload["exact_plan_file_sha256"] == capture[
            "plan_file_sha256"
        ]
        assert embedded_payload["exact_plan_payload_sha256"] == capture[
            "plan_payload_sha256"
        ]
        assert embedded_payload["exact_plan_pcm_sha256"] == capture[
            "plan_pcm_sha256"
        ]
        assert embedded_payload["fresh_meter_raw_sha256"] == capture[
            "meter_sha256"
        ]
        assert embedded_payload["fresh_meter_receipt_sha256"] == capture[
            "meter_receipt_sha256"
        ]
        assert json.dumps(
            embedded_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) == embedded
        assert int(archive["derived_lead_samples"].item()) == 206
        np.testing.assert_allclose(
            archive["panel_primary_minus_secondary_bulk_delay_samples"],
            [50.0] * 5,
        )
    with np.load(secondary, allow_pickle=False) as archive:
        assert str(archive["plant_role"].item()) == "secondary"
        assert str(archive["source_raw_npz_sha256"].item()) == capture[
            "raw_sha256"
        ]
        assert str(archive["source_analysis_npz_sha256"].item()) == result[
            "analysis_sha256"
        ]
        assert str(archive["measurement_level_evidence_sha256"].item()) == (
            capture["level_evidence_sha256"]
        )
        assert str(archive["fresh_meter_raw_path"].item()) == str(
            Path(capture["meter_path"]).relative_to(fixture["root"])
        )
        assert str(archive["fresh_meter_raw_sha256"].item()) == capture[
            "meter_sha256"
        ]
        assert str(archive["fresh_meter_receipt_sha256"].item()) == capture[
            "meter_receipt_sha256"
        ]
        np.testing.assert_array_equal(
            archive["band_consistency"], result["evidence"].secondary_consistency
        )
    contract = ControlBandContract.broadband_point_control()
    loaded = load_measured_band_path(
        secondary,
        role="secondary",
        valid_band_hz=contract.point_control_target_hz,
        subbands_hz=contract.point_control_subbands_hz,
    )
    assert loaded.holdout_receipt["passed"] is True
    metadata = json.loads(result["paths"]["analysis_json"].read_text())
    assert metadata["status"] == "PASS"
    assert metadata["same_capture_for_primary_secondary"] is True
    assert metadata["broadband_plant_evidence"]["clip_count"] == 0


def test_published_secondary_is_accepted_by_broadband_criterion_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, capture, analysis = _load_and_analyse(tmp_path, monkeypatch)
    primary = fixture["assets"] / "admission-primary.npz"
    secondary = fixture["assets"] / "admission-secondary.npz"
    result = analyzer.publish_broadband_analysis(
        capture=capture,
        analysis=analysis,
        analysis_npz_path=fixture["session"] / "admission-analysis.npz",
        analysis_json_path=fixture["session"] / "admission-analysis.json",
        primary_path=primary,
        secondary_path=secondary,
        publish=True,
    )
    with np.load(secondary, allow_pickle=False) as archive:
        evidence_sha = str(archive["broadband_plant_evidence_sha256"].item())
    cfg = {
        "seed": 17,
        "loss": {
                "schema_version": "broadband_equal_subband_loss_v3",
            "lambda_dnh": 0.01,
        },
            "data": {
                "sample_rate": 48_000,
                "digital_primary_path_mode": "measured",
            },
        "duct": {
            "secondary_path": {
                "npz": str(secondary.relative_to(fixture["root"])),
                "handoff_extra_samples": 256,
            },
            "digital_reference": {
                "primary_path_npz": str(primary.relative_to(fixture["root"])),
            },
            "acoustics": {
                "plane_wave_cutoff_hz": 1633.0,
                "realistic_target_band_hz": [150.0, 11_313.708498984761],
            },
        },
        "control_band_contract_sha256": (
            ControlBandContract.broadband_point_control().digest()
        ),
        "broadband_plant_evidence_sha256": evidence_sha,
    }

    admission = admit_criterion_config(
        cfg, repo_root=fixture["root"], require_bound=False
    )

    assert admission.role == "broadband_point_control"
    assert admission.broadband_plant_evidence_sha256 == evidence_sha
    assert admission.broadband_derived_lead_samples == 206
    assert admission.broadband_source_analysis_path == result["paths"][
        "analysis_npz"
    ]


def test_audit_failure_leaves_no_derived_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, capture, analysis = _load_and_analyse(tmp_path, monkeypatch)
    analysis = dict(analysis)
    analysis["clock_valid_repeats"] = (8, 8, 8, 8, 7)
    paths = {
        "analysis_npz": fixture["session"] / "blocked-analysis.npz",
        "analysis_json": fixture["session"] / "blocked-analysis.json",
        "primary": fixture["assets"] / "blocked-primary.npz",
        "secondary": fixture["assets"] / "blocked-secondary.npz",
    }
    with pytest.raises(ValueError, match="BLOCKED"):
        analyzer.publish_broadband_analysis(
            capture=capture,
            analysis=analysis,
            analysis_npz_path=paths["analysis_npz"],
            analysis_json_path=paths["analysis_json"],
            primary_path=paths["primary"],
            secondary_path=paths["secondary"],
            publish=True,
        )
    assert not any(path.exists() for path in paths.values())


def test_existing_target_blocks_transaction_without_replacing_or_orphans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, capture, analysis = _load_and_analyse(tmp_path, monkeypatch)
    primary = fixture["assets"] / "existing-primary.npz"
    primary.write_bytes(b"keep-me")
    paths = {
        "analysis_npz": fixture["session"] / "never-analysis.npz",
        "analysis_json": fixture["session"] / "never-analysis.json",
        "secondary": fixture["assets"] / "never-secondary.npz",
    }
    with pytest.raises(FileExistsError, match="덮어쓰지"):
        analyzer.publish_broadband_analysis(
            capture=capture,
            analysis=analysis,
            analysis_npz_path=paths["analysis_npz"],
            analysis_json_path=paths["analysis_json"],
            primary_path=primary,
            secondary_path=paths["secondary"],
            publish=True,
        )
    assert primary.read_bytes() == b"keep-me"
    assert not any(path.exists() for path in paths.values())


def test_noreplace_race_after_first_promotion_removes_derived_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, capture, analysis = _load_and_analyse(tmp_path, monkeypatch)
    paths = {
        "analysis_npz": fixture["session"] / "race-analysis.npz",
        "analysis_json": fixture["session"] / "race-analysis.json",
        "primary": fixture["assets"] / "race-primary.npz",
        "secondary": fixture["assets"] / "race-secondary.npz",
    }
    original = analyzer.atomic_publish_noreplace
    calls = 0

    def fail_second(temporary, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FileExistsError("injected no-replace race")
        return original(temporary, target)

    monkeypatch.setattr(analyzer, "atomic_publish_noreplace", fail_second)
    with pytest.raises(FileExistsError, match="injected"):
        analyzer.publish_broadband_analysis(
            capture=capture,
            analysis=analysis,
            analysis_npz_path=paths["analysis_npz"],
            analysis_json_path=paths["analysis_json"],
            primary_path=paths["primary"],
            secondary_path=paths["secondary"],
            publish=True,
        )
    assert calls == 2
    assert not any(path.exists() for path in paths.values())
    assert not list(fixture["session"].glob(".*.partial"))
    assert not list(fixture["assets"].glob(".*.partial"))


def test_bound_plan_tamper_after_analysis_blocks_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, capture, analysis = _load_and_analyse(tmp_path, monkeypatch)
    fixture["plan_path"].write_text("{}\n", encoding="utf-8")
    paths = {
        "analysis_npz": fixture["session"] / "tamper-analysis.npz",
        "analysis_json": fixture["session"] / "tamper-analysis.json",
        "primary": fixture["assets"] / "tamper-primary.npz",
        "secondary": fixture["assets"] / "tamper-secondary.npz",
    }
    with pytest.raises(ValueError, match="bytes가 변경"):
        analyzer.publish_broadband_analysis(
            capture=capture,
            analysis=analysis,
            analysis_npz_path=paths["analysis_npz"],
            analysis_json_path=paths["analysis_json"],
            primary_path=paths["primary"],
            secondary_path=paths["secondary"],
            publish=True,
        )
    assert not any(path.exists() for path in paths.values())


def test_cli_failure_path_never_imports_sounddevice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported = False
    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        nonlocal imported
        if name == "sounddevice":
            imported = True
            raise AssertionError("offline analyzer must not import sounddevice")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    assert analyzer.main(
        [str(tmp_path / "missing"), "--plan", str(tmp_path / "missing.json")]
    ) == 1
    assert imported is False


def test_raw_loader_rejects_forged_meter_to_stream_freshness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write_mock_raw_session(tmp_path, monkeypatch)

    def make_stale(metadata):
        meter_completed = dt.datetime.fromisoformat(
            metadata["meter"]["completed_at_utc"]
        )
        stream_started = meter_completed + dt.timedelta(
            seconds=measurement.BOOTSTRAP_METER_MAX_AGE_SECONDS + 1
        )
        metadata["started_at_utc"] = (
            stream_started - dt.timedelta(seconds=3)
        ).isoformat()
        metadata["telemetry"]["stream_started_at_utc"] = stream_started.isoformat()
        metadata["completed_at_utc"] = (
            stream_started + dt.timedelta(seconds=1)
        ).isoformat()

    _rewrite_raw_metadata(fixture["session"], make_stale)
    with pytest.raises(ValueError, match="freshness 위반"):
        analyzer.load_broadband_raw_capture(
            session_dir=fixture["session"],
            plan_path=fixture["plan_path"],
            hardware_path=fixture["hardware"],
            repository_root=fixture["root"],
        )


def test_publication_rejects_repository_escape_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, capture, analysis = _load_and_analyse(tmp_path, monkeypatch)
    outside = tmp_path.parent / f"outside-{tmp_path.name}.npz"
    secondary = fixture["assets"] / "inside-secondary.npz"
    with pytest.raises(ValueError, match="저장소 안"):
        analyzer.publish_broadband_analysis(
            capture=capture,
            analysis=analysis,
            analysis_npz_path=fixture["session"] / "escape-analysis.npz",
            analysis_json_path=fixture["session"] / "escape-analysis.json",
            primary_path=outside,
            secondary_path=secondary,
            publish=True,
        )
    assert not outside.exists()
    assert not secondary.exists()


def test_rollback_never_deletes_rival_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, capture, analysis = _load_and_analyse(tmp_path, monkeypatch)
    paths = {
        "analysis_npz": fixture["session"] / "rival-analysis.npz",
        "analysis_json": fixture["session"] / "rival-analysis.json",
        "primary": fixture["assets"] / "rival-primary.npz",
        "secondary": fixture["assets"] / "rival-secondary.npz",
    }
    original = analyzer.atomic_publish_noreplace
    calls = 0

    def replace_first_then_fail_second(temporary, target):
        nonlocal calls
        calls += 1
        if calls == 1:
            original(temporary, target)
            Path(target).unlink()
            Path(target).write_bytes(b"rival-owned")
            return Path(target)
        raise FileExistsError("injected rival race")

    monkeypatch.setattr(
        analyzer, "atomic_publish_noreplace", replace_first_then_fail_second
    )
    with pytest.raises(FileExistsError, match="rival race"):
        analyzer.publish_broadband_analysis(
            capture=capture,
            analysis=analysis,
            analysis_npz_path=paths["analysis_npz"],
            analysis_json_path=paths["analysis_json"],
            primary_path=paths["primary"],
            secondary_path=paths["secondary"],
            publish=True,
        )
    assert paths["analysis_npz"].read_bytes() == b"rival-owned"
    assert not paths["analysis_json"].exists()
    assert not paths["primary"].exists()
    assert not paths["secondary"].exists()
