from __future__ import annotations

import importlib.util
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import fftconvolve
import deep_anc.dsp.fullband_causal_v5 as causal_v5

from deep_anc.dsp.fullband_causal_v5 import (
    CONDITION_AUDIT_SUPPORT,
    LIVE_AUTHORITY,
    PERIOD,
    ROLES,
    build_plan_v5,
    estimate_common_affine_clock_v5,
    exact_condition_audit_v5,
    exact_shifted_condition_audit_v5,
    score_candidate_on_role_v5,
    estimate_common_clock_from_waveforms_v5,
    synthesize_affine_capture_v5,
)


def test_v5_plan_binds_exact_v3_contract_actual_pcm_and_live_lock() -> None:
    plan, pcm = build_plan_v5()
    assert LIVE_AUTHORITY is None
    assert plan["live_authority"] is None
    assert plan["canonical_training_eligible"] is False
    assert plan["control_band_contract_sha256"] == (
        "53579b9ff8419ac19fb2458c29a3e8a94ffbb2eeb88cc07f34b76c68033989f2"
    )
    assert plan["canonical_payload_sha256"] == (
        "32a79b3700b457dc40373dc4dd0969301287baea7100b1ec5edd86ea907ee127"
    )
    assert plan["actual_submitted_pcm_sha256"] == (
        "c18416e4066556479fd317659d908c215e6662d08f5bfa9d50e4ac63971c4aff"
    )
    assert plan["measurement_level_safety"]["official_meter_output_pcm_sha256"] == (
        "95a7f97b19d1ea26203ae05d72309326eacd3f0f9a5c54f8e60f5041b817f596"
    )
    assert plan["measurement_level_safety"]["meter_total_power_not_exceeded"] is True
    bands = plan["control_band_contract"]["physical_identification_subbands_hz"]
    assert len(bands) == 8 and bands[0][0] == pytest.approx(88.3883476483)
    assert bands[-1][1] == pytest.approx(11313.7084989848)
    assert plan["excitation"]["qualification_band_hz"] == [80.0, 11313.7084989848]
    assert plan["actual_submitted_peak_pcm"] <= 98
    assert plan["duration_seconds"] < 50.0
    assert pcm.dtype == np.int16 and pcm.shape[1] == 2
    assert plan["publisher_contract"]["raw_session_relative_path"].endswith(".npz")
    for role in ROLES:
        assert sum(row.get("role") == role for row in plan["layout"]) == 2


def test_v5_time_separation_and_actual_pilot_denominator_include_pe() -> None:
    plan, pcm = build_plan_v5()
    bins_by_path = {
        "primary": np.asarray(plan["clock_contract"]["primary_pilot_bins"]),
        "secondary": np.asarray(plan["clock_contract"]["secondary_pilot_bins"]),
    }
    lead_spectrum = np.fft.rfft(pcm[PERIOD : 2 * PERIOD].astype(np.float64), axis=0)
    for row in plan["layout"]:
        if row.get("role") not in ROLES:
            continue
        path = row["path"]
        active = 0 if path == "primary" else 1
        opposite = 1 - active
        central = pcm[row["central_start_frame"] : row["central_stop_frame"]]
        spectrum = np.fft.rfft(central.astype(np.float64), axis=0)
        bins = bins_by_path[path]
        assert np.max(np.abs(spectrum[bins, opposite])) <= 1.0e-8
        # near-white PE의 pilot-line 성분까지 실제 분모에 포함됨을 확인한다.
        assert np.max(np.abs(spectrum[bins, active] - lead_spectrum[bins, active])) > 1.0
        assert np.all(central[:, opposite] == pcm[PERIOD : 2 * PERIOD, opposite])


def test_v5_exact_support_1024_condition_passes_and_longer_are_not_claimed() -> None:
    plan, pcm = build_plan_v5()
    receipt = exact_condition_audit_v5(plan, pcm)
    assert receipt["passed"] is True
    assert receipt["joint_fit_condition_number"] == pytest.approx(9.05803353091781)
    assert receipt["role_condition_numbers"]["fit_a"] == pytest.approx(11.571714021472099)
    assert receipt["role_condition_numbers"]["fit_b"] == pytest.approx(12.575291092522646)
    assert receipt["zeros_before_fir_samples"] == [0, 0]
    assert receipt["operator_definition"]["zeros_before_fir_samples"] == {
        "primary": 0,
        "secondary": 0,
    }
    assert len(receipt["operator_definition_sha256"]) == 64
    assert receipt["operator_quadratic_form_crosscheck_passed"] is True
    assert len(receipt["operator_quadratic_form_probe_receipts"]) == 4
    assert (
        receipt["operator_quadratic_form_relative_error"]
        <= receipt["operator_quadratic_form_maximum_allowed"]
    )
    assert set(receipt["longer_supports"].values()) == {"NOT_AUDITED_NO_CLAIM"}
    with pytest.raises(ValueError, match="1024"):
        exact_condition_audit_v5(plan, pcm, support=2048)
    with pytest.raises(ValueError, match="exact integer"):
        exact_shifted_condition_audit_v5(
            plan,
            pcm,
            zeros_by_path=(1386.5, 1245),  # type: ignore[arg-type]
        )

    shifted = exact_shifted_condition_audit_v5(
        plan, pcm, zeros_by_path=(1386, 1245)
    )
    assert shifted["owned_input_receipt"]["canonical_plan_exact"] is True
    assert shifted["owned_input_receipt"]["toctou_entry_exit_equal"] is True
    changed = copy.deepcopy(plan)
    changed["layout"][1]["central_start_frame"] += 1
    with pytest.raises(ValueError, match="canonical v5"):
        exact_shifted_condition_audit_v5(
            changed, pcm, zeros_by_path=(1386, 1245)
        )


@pytest.mark.parametrize("audit", ["unshifted", "shifted"])
@pytest.mark.parametrize("mutation", ["layout", "payload"])
def test_public_exact_audits_reject_rehashed_plan_tampering_before_heavy_math(
    monkeypatch: pytest.MonkeyPatch, audit: str, mutation: str
) -> None:
    plan, pcm = build_plan_v5()
    changed = copy.deepcopy(plan)
    if mutation == "layout":
        changed["layout"][1]["central_start_frame"] += 1
    else:
        changed["excitation"]["payloads"]["primary_fit_a"]["seed"] += 2
    unsigned = {
        key: value for key, value in changed.items()
        if key != "canonical_payload_sha256"
    }
    changed["canonical_payload_sha256"] = causal_v5._payload_sha256(unsigned)

    def forbidden_heavy_audit(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("canonical 변조가 heavy Gram 계산 전에 거부되지 않았습니다")

    monkeypatch.setattr(
        causal_v5, "_exact_condition_audit_with_shifts_v5", forbidden_heavy_audit
    )
    with pytest.raises(ValueError, match="canonical v5 plan/layout/payload/PCM"):
        if audit == "unshifted":
            exact_condition_audit_v5(changed, pcm)
        else:
            exact_shifted_condition_audit_v5(
                changed, pcm, zeros_by_path=(1386, 1245)
            )


@pytest.mark.parametrize("mutation", ["pcm", "plan"])
def test_shifted_heavy_audit_detects_source_toctou(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    plan, pcm = build_plan_v5()
    original = causal_v5._exact_condition_audit_with_shifts_v5

    def mutate_after_owned_audit(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        result = original(*args, **kwargs)
        if mutation == "pcm":
            pcm[0, 0] ^= np.int16(1)
        else:
            plan["layout"][0]["kind"] = "mutated_after_owned_copy"
        return result

    monkeypatch.setattr(causal_v5, "_exact_condition_audit_with_shifts_v5", mutate_after_owned_audit)
    with pytest.raises(ValueError, match="TOCTOU mutation"):
        exact_shifted_condition_audit_v5(plan, pcm, zeros_by_path=(1386, 1245))


@pytest.mark.parametrize("ppm", [-413.931, 0.0, 413.931])
def test_v5_common_clock_recovers_affine_drift_with_path_specific_lines(ppm: float) -> None:
    plan, _ = build_plan_v5()
    primary = np.asarray(plan["clock_contract"]["primary_pilot_bins"][:38]) * 48_000 / PERIOD
    secondary = np.asarray(plan["clock_contract"]["secondary_pilot_bins"][:38]) * 48_000 / PERIOD
    frequency = np.stack((primary, primary, secondary, secondary))
    times = np.arange(7, dtype=np.float64) * 1.5
    intercept = np.arange(frequency.size, dtype=np.float64).reshape(frequency.shape) * 0.013
    phase = intercept[None] + (
        2.0 * np.pi * frequency[None] * ppm * 1.0e-6 * times[:, None, None]
    )
    receipt = estimate_common_affine_clock_v5(
        row_times=times, phase_radians=phase, frequencies_hz=frequency
    )
    assert receipt["estimated_ppm"] == pytest.approx(ppm, abs=1.0e-6)
    assert receipt["passed"] is True


def test_v5_common_clock_rejects_piecewise_change_point() -> None:
    frequency = np.tile(np.linspace(160.0, 590.0, 20), (4, 1))
    times = np.arange(7, dtype=np.float64) * 1.5
    phase = 2.0 * np.pi * frequency[None] * 100.0e-6 * times[:, None, None]
    phase = np.broadcast_to(phase, (7, 4, 20)).copy()
    phase[4:] += (
        2.0
        * np.pi
        * frequency[None]
        * 300.0e-6
        * (times[4:, None, None] - times[3])
    )
    receipt = estimate_common_affine_clock_v5(
        row_times=times, phase_radians=phase, frequencies_hz=frequency
    )
    assert receipt["passed"] is False
    assert receipt["maximum_change_point_samples"] > receipt["hard_max_residual_samples"]


def _fixture_firs() -> tuple[np.ndarray, np.ndarray]:
    primary = np.zeros((2, CONDITION_AUDIT_SUPPORT), dtype=np.float64)
    secondary = np.zeros_like(primary)
    primary[:, 10] = [0.40, 0.30]
    primary[:, 30] = [0.08, -0.04]
    secondary[:, 15] = [0.35, 0.28]
    secondary[:, 40] = [-0.07, 0.05]
    return primary, secondary


@pytest.mark.parametrize("ppm", [-413.931, 413.931])
def test_v5_raw_waveform_clock_uses_actual_two_input_denominator(ppm: float) -> None:
    plan, submitted = build_plan_v5()
    primary, secondary = _fixture_firs()
    captured = synthesize_affine_capture_v5(
        submitted,
        primary_fir_by_mic=primary,
        secondary_fir_by_mic=secondary,
        rate_ratio=1.0 + ppm * 1.0e-6,
    )
    receipt = estimate_common_clock_from_waveforms_v5(
        plan=plan, submitted_pcm=submitted, captured_adc_pcm=captured
    )
    assert receipt["estimated_ppm"] == pytest.approx(ppm, abs=0.01)
    assert receipt["actual_submitted_denominator_includes_pe"] is True
    assert receipt["highband_result_based_phase_repair_samples"] == 0.0
    assert receipt["validation_policy"] == "holdout_and_tail_legacy"
    assert receipt["operator_holdout_used_for_clock_validation"] is True
    assert receipt["passed"] is True

    # pilot-only 또는 float 의도 신호를 분모로 바꾸면 plan PCM SHA에서 먼저 닫힌다.
    wrong_denominator = submitted.copy()
    active = next(row for row in plan["layout"] if row.get("role") == "fit_a")
    active_channel = 0 if active["path"] == "primary" else 1
    wrong_denominator[
        active["central_start_frame"] : active["central_stop_frame"], active_channel
    ] = 0
    with pytest.raises(ValueError, match="actual submitted PCM"):
        estimate_common_clock_from_waveforms_v5(
            plan=plan,
            submitted_pcm=wrong_denominator,
            captured_adc_pcm=captured,
        )


def test_v5_raw_waveform_clock_rejects_piecewise_affine_capture() -> None:
    plan, submitted = build_plan_v5()
    primary, secondary = _fixture_firs()
    captured = synthesize_affine_capture_v5(
        submitted,
        primary_fir_by_mic=primary,
        secondary_fir_by_mic=secondary,
        rate_ratio=1.0 - 100.0e-6,
        piecewise_ratio_after_half=1.0 + 300.0e-6,
    )
    receipt = estimate_common_clock_from_waveforms_v5(
        plan=plan, submitted_pcm=submitted, captured_adc_pcm=captured
    )
    assert receipt["passed"] is False
    assert receipt["maximum_validation_phase_error_samples"] > 0.06755189029558946


def test_v5_band_score_requires_every_path_mic_band_and_holdout() -> None:
    plan, submitted = build_plan_v5()
    primary = np.zeros((2, CONDITION_AUDIT_SUPPORT), dtype=np.float64)
    secondary = np.zeros_like(primary)
    primary[0, :3] = [0.42, -0.08, 0.025]
    primary[1, :3] = [0.31, 0.05, -0.018]
    secondary[0, :3] = [0.37, 0.07, -0.021]
    secondary[1, :3] = [0.29, -0.04, 0.014]
    captured = np.zeros_like(submitted, dtype=np.float64)
    for mic in range(2):
        captured[:, mic] = fftconvolve(submitted[:, 0], primary[mic], mode="full")[: len(submitted)]
        captured[:, mic] += fftconvolve(submitted[:, 1], secondary[mic], mode="full")[: len(submitted)]
    for role in ROLES:
        receipt = score_candidate_on_role_v5(
            plan=plan,
            submitted_pcm=submitted,
            captured_pcm=captured,
            primary_fir_by_mic=primary,
            secondary_fir_by_mic=secondary,
            role=role,
        )
        assert len(receipt["rows"]) == 32
        assert receipt["all_paths_microphones_subbands_passed"] is True
        assert all(row["response_to_noise_db"] >= 20.0 for row in receipt["rows"])


def test_v5_cli_no_replace_and_symlink_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path("scripts/data/measure_paths_fullband_causal_v5.py")
    spec = importlib.util.spec_from_file_location("v5_cli_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    target = tmp_path / "results" / "plan.json"
    module._write_json_no_replace({"a": 1}, target)
    with pytest.raises(FileExistsError):
        module._write_json_no_replace({"a": 2}, target)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module._write_json_no_replace({"a": 1}, link / "plan.json")
    assert json.loads(target.read_text()) == {"a": 1}


def test_v5_offline_raw_publisher_exact_path_no_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path("scripts/data/measure_paths_fullband_causal_v5.py")
    spec = importlib.util.spec_from_file_location("v5_raw_cli_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    plan, submitted = build_plan_v5(
        raw_session_relative_path="results/v5-test/raw.npz"
    )
    captured = np.zeros(submitted.shape, dtype=np.int32)
    callbacks = np.full(math.ceil(len(submitted) / 256), 256, dtype=np.int64)
    target = module._publish_raw_no_replace(
        plan=plan,
        submitted_pcm=submitted,
        captured_pcm=captured,
        callback_frames=callbacks,
    )
    assert target == tmp_path / "results/v5-test/raw.npz"
    with np.load(target, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "submitted_pcm",
            "captured_pcm",
            "callback_frames",
            "metadata_json_utf8",
        }
        metadata = json.loads(bytes(archive["metadata_json_utf8"]).decode("utf-8"))
        assert metadata["signal_plan_payload_sha256"] == plan["canonical_payload_sha256"]
    with pytest.raises(FileExistsError):
        module._publish_raw_no_replace(
            plan=plan,
            submitted_pcm=submitted,
            captured_pcm=captured,
            callback_frames=callbacks,
        )
    assert not list((tmp_path / "results/v5-test").glob(".*.staging-*"))
    with pytest.raises(ValueError, match="<i4"):
        other_plan, other_submitted = build_plan_v5(
            raw_session_relative_path="results/v5-test/raw-float.npz"
        )
        module._publish_raw_no_replace(
            plan=other_plan,
            submitted_pcm=other_submitted,
            captured_pcm=np.zeros(other_submitted.shape, dtype=np.float64),
            callback_frames=callbacks,
        )
