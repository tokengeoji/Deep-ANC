from __future__ import annotations

import hashlib
import json
import numpy as np
import pytest

from deep_anc.dsp.fullband_causal_v6 import (
    FS,
    build_plan_v6,
    synthesize_affine_capture_v6,
)
from deep_anc.dsp.fullband_causal_v6_forensics import (
    _deterministic_modes,
    _short_time_line_rate_ppm,
    diagnose_short_time_clock_v6,
    replay_affine_clock_admission_v6,
    validate_failure_binding_v6,
)


def test_short_time_line_rate_uses_absolute_phase_and_recovers_scale() -> None:
    frequency = 401.0 * FS / 32_768.0
    truth_ppm = 3_146.25
    start = 123_456
    sample = start + np.arange(65_536, dtype=np.float64)
    signal = np.cos(
        2.0 * np.pi * frequency * (1.0 + truth_ppm * 1.0e-6) * sample / FS
        + 0.37
    )
    rates, midpoints = _short_time_line_rate_ppm(
        signal,
        absolute_start_frame=start,
        frequencies_hz=np.asarray([frequency]),
    )
    assert rates.shape == (1, 56)
    assert midpoints.shape == (56,)
    assert float(np.median(rates)) == pytest.approx(truth_ppm, abs=0.01)
    assert float(np.max(rates) - np.min(rates)) < 0.02


def test_diagnostic_mode_summary_is_deterministic_and_not_a_gate() -> None:
    values = np.concatenate(
        (
            np.linspace(-4_900.0, -4_600.0, 40),
            np.linspace(-950.0, -750.0, 50),
            np.linspace(3_000.0, 3_300.0, 60),
            np.asarray([-3_000.0, 250.0, 1_800.0]),
        )
    )
    first = _deterministic_modes(values)
    second = _deterministic_modes(values.copy())
    assert first == second
    assert len(first) == 3
    assert [row["membership_median_ppm"] for row in first] == pytest.approx(
        [-4_750.0, -850.0, 3_150.0], abs=1.0
    )


def test_affine_fixture_yields_one_diagnostic_mode_without_authority() -> None:
    plan, submitted = build_plan_v6()
    primary = np.asarray([[7_000.0], [5_000.0]], dtype=np.float64)
    secondary = np.asarray([[-6_000.0], [4_500.0]], dtype=np.float64)
    truth_ppm = 173.25
    captured = synthesize_affine_capture_v6(
        submitted,
        primary_fir_by_mic=primary,
        secondary_fir_by_mic=secondary,
        rate_ratio=1.0 + truth_ppm * 1.0e-6,
        noise_rms=0.1,
        seed=20260829,
    )
    replay = replay_affine_clock_admission_v6(
        plan=plan, submitted_pcm=submitted, captured_pcm=captured
    )
    assert replay["passed"] is True
    captured = np.ascontiguousarray(np.rint(captured).astype(np.int32))
    report = diagnose_short_time_clock_v6(
        plan=plan, submitted_pcm=submitted, captured_pcm=captured
    )
    assert report["authority"] == "diagnostic_only_no_clock_no_plant_no_training_authority"
    assert report["analysis_admission_eligible"] is False
    assert report["clock_estimate_authority"] is False
    assert report["canonical_training_eligible"] is False
    assert report["deployment_eligible"] is False
    assert report["attenuation_assessed"] is False
    assert report["plant_identification_assessed"] is False
    assert report["summary"]["step_count"] == 448
    assert report["summary"]["rate_ppm_median"] == pytest.approx(truth_ppm, abs=25.0)
    assert report["summary"]["rate_ppm_p95"] - report["summary"]["rate_ppm_p05"] < 100.0
    assert report["summary"]["declared_affine_search_bounds_member_fraction"] == 1.0
    assert len(report["summary"]["diagnostic_modes"]) == 1
    assert len(report["blocks"]) == 8


def test_forensics_rejects_nonexact_submitted_pcm() -> None:
    plan, submitted = build_plan_v6()
    changed = submitted.copy()
    changed[0, 0] += 1
    with pytest.raises(ValueError, match="exact v6 plan/PCM"):
        diagnose_short_time_clock_v6(
            plan=plan,
            submitted_pcm=changed,
            captured_pcm=np.zeros_like(changed, dtype=np.float64),
        )

    float_submitted = submitted.astype(np.float64)
    with pytest.raises(ValueError, match="exact v6 plan/PCM"):
        diagnose_short_time_clock_v6(
            plan=plan,
            submitted_pcm=float_submitted,
            captured_pcm=np.zeros_like(submitted, dtype=np.int32),
        )

    with pytest.raises(ValueError, match="C-contiguous int32"):
        diagnose_short_time_clock_v6(
            plan=plan,
            submitted_pcm=submitted,
            captured_pcm=np.zeros(submitted.shape, dtype=">i4"),
        )


def test_short_time_helper_rejects_invalid_frequency_and_zero_phasor() -> None:
    signal = np.ones(16_384, dtype=np.float64)
    with pytest.raises(ValueError, match="frequency"):
        _short_time_line_rate_ppm(
            signal,
            absolute_start_frame=0,
            frequencies_hz=np.asarray([np.nan]),
        )
    with pytest.raises(ValueError, match="phasor"):
        _short_time_line_rate_ppm(
            np.zeros_like(signal),
            absolute_start_frame=0,
            frequencies_hz=np.asarray([1_000.0]),
        )


def test_failure_binding_requires_self_hash_raw_receipt_and_false_authority() -> None:
    raw_sha = "1" * 64
    receipt_sha = "2" * 64
    core = {
        "schema": "fullband_causal_v6_live_delay_failure_v1",
        "status": "FAILED",
        "raw": {"path": "results/fullband_causal_v6/raw_capture.npz", "file_sha256": raw_sha},
        "external_post_receipt": {
            "path": "results/fullband_causal_v6/raw_capture.npz.post_receipt.json",
            "file_sha256": receipt_sha,
        },
        "failure_stage": "global_grid_basin_search",
        "optimizer_started": True,
        "error": "V6ClockAdmissionError: global clock objective가 multimodal ambiguous입니다",
        "available_snr_receipt": {"passed": True},
        "analysis_published": False,
        "operator_published": False,
        "canonical_training_eligible": False,
        "hardware_sample_slip_authority": False,
    }
    payload = json.dumps(
        core,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    failure = {
        **core,
        "failure_payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert validate_failure_binding_v6(
        failure,
        raw_relative_path=core["raw"]["path"],
        raw_file_sha256=raw_sha,
        receipt_relative_path=core["external_post_receipt"]["path"],
        receipt_file_sha256=receipt_sha,
    ) == failure

    changed = dict(failure)
    changed["canonical_training_eligible"] = True
    with pytest.raises(ValueError, match="authority"):
        validate_failure_binding_v6(
            changed,
            raw_relative_path=core["raw"]["path"],
            raw_file_sha256=raw_sha,
            receipt_relative_path=core["external_post_receipt"]["path"],
            receipt_file_sha256=receipt_sha,
        )

    with pytest.raises(ValueError, match="raw binding"):
        validate_failure_binding_v6(
            failure,
            raw_relative_path=core["raw"]["path"],
            raw_file_sha256="3" * 64,
            receipt_relative_path=core["external_post_receipt"]["path"],
            receipt_file_sha256=receipt_sha,
        )
