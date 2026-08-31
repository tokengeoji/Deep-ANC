from __future__ import annotations

import builtins
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from deep_anc.dsp.stage2_2khz_contract import (
    STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ,
)
from deep_anc.dsp.stage2_2khz_measurement import (
    CURRENT_LIVE_BLOCK_CODE,
    CURRENT_TOPOLOGY_STATUS,
    Stage2MeasurementError,
    admit_stage2_relative_ps_candidate,
    build_stage2_measurement_plan,
    publish_plan_no_replace,
    validate_stage2_metric_receipt,
    validate_submitted_pcm,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "stage2_2khz_measurement.json"
SCRIPT_PATH = ROOT / "scripts" / "data" / "measure_paths_stage2_2khz.py"


@pytest.fixture(scope="module")
def plan_and_pcm() -> tuple[dict, np.ndarray]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_stage2_measurement_plan(config)


def _zero_counters() -> dict[str, int]:
    return {
        "xrun": 0,
        "clip": 0,
        "callback_status": 0,
        "sample_slip": 0,
        "sample_drop": 0,
        "sample_add": 0,
    }


def _passing_receipt(plan: dict) -> dict:
    rows = []
    for path in ("primary", "secondary"):
        for microphone in ("ERR", "REF"):
            for index, band in enumerate(
                STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ
            ):
                rows.append(
                    {
                        "path": path,
                        "microphone": microphone,
                        "subband_index": index,
                        "band_hz": list(band),
                        "fit_a_fit_b_consistency": 0.951,
                        "untouched_holdout_consistency": 0.952,
                        "response_to_noise_db": 20.1,
                    }
                )
    return {
        "schema": "stage2_2khz_same_capture_ps_result_v1",
        "plan_sha256": plan["plan_sha256"],
        "actual_submitted_pcm_sha256": plan["actual_submitted_pcm"]["sha256"],
        "same_capture_ps": True,
        "raw_publish_mode": "no_replace",
        "analysis_publish_mode": "no_replace",
        "raw_capture_sha256": "a" * 64,
        "counters": _zero_counters(),
        "holdout_policy": {
            "used_for_fit": False,
            "used_for_support_selection": False,
            "used_for_threshold_selection": False,
            "used_for_predeclared_shared_q_nuisance_likelihood": True,
            "evaluated_after_fit_frozen": True,
        },
        "clock_witness": {
            "kind": "submitted_aperiodic_shared_q_acoustic_likelihood",
            "continuous_frames": plan["signal_frames"],
            "gap_frames": 0,
            "single_shared_q_for_all_p_s_err_ref_views": True,
            "selection_input": "full_capture_low_band_known_stereo_codes_to_err_ref",
            "selection_band_hz": [88.3883476483, 600.0],
            "q_model": "single_affine",
            "search_boundary_optimum": False,
            "ambiguity_envelope_validation_samples": 0.049,
            "maximum_timing_residual_samples": 0.270208,
            "counters": _zero_counters(),
        },
        "absolute_transport_claims": {
            "absolute_hardware_frame_identity_claimed": False,
            "callback_before_start_drop_observed_claimed": False,
            "hardware_counter_slip_zero_claimed": False,
            "relative_ps_lead_only": True,
        },
        "nonaffine_change_point_audit": {
            "transport_256_callback_contiguity_tested": True,
            "transport_semantics": "software_accounting_not_absolute_hardware_slip",
            "acoustic_q_epoch_frames": 96000,
            "all_acoustic_q_epoch_boundaries_tested": True,
            "all_256_frame_acoustic_q_boundaries_tested": False,
            "change_point_detected": False,
            "nonaffine_drift_detected": False,
            "one_sample_insert_drop_detected": False,
            "view_specific_q_detected": False,
            "affine_model_frozen_before_holdout": True,
            "holdout_failure_refit_performed": False,
        },
        "relative_delay_scope": {
            "playback_to_err_acoustic_delay_included_in_primary": True,
            "playback_to_err_acoustic_delay_included_in_secondary": True,
            "common_intercept_claimed_separately": False,
            "common_time_gauge_cancels_in_p_minus_s": True,
            "manual_lead_allowed": False,
        },
        "path_subbands": rows,
        "thresholds_relaxed": False,
        "capture_generation": {
            "adapter_schema": "stage2_2khz_live_capture_adapter_v1",
            "reviewed_live_adapter_implemented": True,
            "physical_acoustic_capture": True,
            "synthetic_or_diagnostic": False,
            "clean_exact_commit": True,
            "native_raw_published_no_replace": True,
        },
    }


def test_plan_has_exact_44_second_actual_int16_lineage_and_independent_codes(
    plan_and_pcm: tuple[dict, np.ndarray],
) -> None:
    plan, pcm = plan_and_pcm
    assert plan["schema"] == "stage2_2khz_same_capture_ps_plan_v1"
    assert plan["signal_seconds"] == 24.0
    assert plan["meter_seconds"] == 20.0
    assert plan["maximum_total_audible_seconds"] == 44.0
    assert pcm.dtype == np.int16
    # meter20s는 level PASS 뒤에만 signal을 허용하기 위한 별도 official stream/raw다.
    # P/S actual submitted lineage는 연속 24s signal이며 총 audible budget은 44s다.
    assert pcm.shape == (1_152_000, 2)
    assert plan["meter_frames"] == 960_000
    assert plan["signal_frames"] == 1_152_000
    assert plan["total_output_frames_across_two_streams"] == 2_112_000
    assert plan["meter_submitted_pcm"]["shape"] == [960_000, 2]
    assert plan["actual_submitted_pcm"]["shape"] == [1_152_000, 2]
    assert plan["actual_submitted_pcm"]["coverage"] == "exact_continuous_stage2_signal24s_after_separate_meter_pass"
    assert plan["actual_submitted_pcm"]["peak_pcm"] == 79
    assert -0.25 <= plan["actual_submitted_pcm"]["meter_relative_total_power_db"] <= 0.0
    assert [item["role"] for item in plan["role_layout"]] == [
        "fit_a",
        "fit_b",
        "untouched_holdout",
    ]
    assert all(abs(item["ns_cs_zero_lag_correlation"]) < 0.02 for item in plan["role_layout"])
    for item in plan["role_layout"]:
        assert item["frames"] == 384_000
        assert item["start_frame_in_capture"] == item["start_frame"]
        assert item["channels"]["NS"]["actual_int16_sha256"] != item["channels"]["CS"]["actual_int16_sha256"]
        for channel in item["channels"].values():
            fractions = channel["spectral_audit"]["subband_energy_fraction"]
            assert len(fractions) == 6
            assert min(fractions) >= 0.12
            assert channel["spectral_audit"]["required_excitation_80_88_388_energy_fraction"] >= 0.005
            assert channel["spectral_audit"]["repeated_slot_count"] == 0
    validate_submitted_pcm(plan, pcm)
    altered = pcm.copy()
    altered[0, 0] ^= np.int16(1)
    with pytest.raises(Stage2MeasurementError, match="bytes"):
        validate_submitted_pcm(plan, altered)


def test_current_hardware_scope_separates_relative_lead_from_absolute_clock(
    plan_and_pcm: tuple[dict, np.ndarray],
) -> None:
    plan, _ = plan_and_pcm
    assessment = plan["topology_assessment"]
    assert assessment["status"] == CURRENT_TOPOLOGY_STATUS
    assert assessment["relative_ps_lead_authority_physically_possible"] is True
    assert assessment["common_time_gauge_cancels_in_relative_delay"] is True
    assert assessment["playback_to_err_acoustic_delay_must_remain_in_each_plant"] is True
    assert assessment["absolute_hardware_frame_clock_authority_available"] is False
    assert assessment["live_capture_adapter_available"] is True
    assert assessment["live_status"] == CURRENT_LIVE_BLOCK_CODE
    policy = plan["clock_witness_policy"]
    assert policy["single_shared_q_for_all_p_s_err_ref_views"] is True
    assert policy["selection_band_hz"] == [88.3883476483, 600.0]
    assert policy["independent_electrical_required_for_relative_ps_lead"] is False
    assert policy["independent_electrical_required_for_absolute_frame_claim"] is True


def test_plan_publish_is_no_replace(
    tmp_path: Path, plan_and_pcm: tuple[dict, np.ndarray]
) -> None:
    plan, _ = plan_and_pcm
    relative = plan["artifacts"]["plan"]
    published = publish_plan_no_replace(str(tmp_path), relative, plan)
    assert published["path"] == relative
    assert (tmp_path / relative).is_file()
    with pytest.raises(FileExistsError, match="덮어쓰지"):
        publish_plan_no_replace(str(tmp_path), relative, plan)


def test_metric_and_future_typed_relative_admission_require_24_rows(
    plan_and_pcm: tuple[dict, np.ndarray],
) -> None:
    plan, _ = plan_and_pcm
    receipt = _passing_receipt(plan)
    result = validate_stage2_metric_receipt(plan, receipt)
    assert result["conditional_relative_ps_measurement_pass"] is True
    assert result["absolute_hardware_frame_clock_authority_pass"] is False
    admission = admit_stage2_relative_ps_candidate(plan, receipt)
    assert admission["relative_ps_training_plant_candidate"] is True
    assert admission["automatic_training_config_update_allowed"] is False
    assert len(receipt["path_subbands"]) == 24


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value["path_subbands"][0].update(fit_a_fit_b_consistency=0.949999), "0.95 미만"),
        (lambda value: value["clock_witness"].update(maximum_timing_residual_samples=0.270209), "0.270208"),
        (lambda value: value["clock_witness"].update(ambiguity_envelope_validation_samples=0.270209), "0.270208 sample"),
        (lambda value: value["nonaffine_change_point_audit"].update(change_point_detected=True), "nonaffine/change-point"),
        (lambda value: value["holdout_policy"].update(used_for_fit=True), "holdout"),
        (lambda value: value["absolute_transport_claims"].update(hardware_counter_slip_zero_claimed=True), "absolute transport"),
    ],
)
def test_metric_receipt_fails_closed(
    plan_and_pcm: tuple[dict, np.ndarray], mutation, match: str
) -> None:
    plan, _ = plan_and_pcm
    receipt = deepcopy(_passing_receipt(plan))
    mutation(receipt)
    with pytest.raises(Stage2MeasurementError, match=match):
        validate_stage2_metric_receipt(plan, receipt)


def test_typed_admission_rejects_synthetic_or_unreviewed_capture(
    plan_and_pcm: tuple[dict, np.ndarray],
) -> None:
    plan, _ = plan_and_pcm
    receipt = _passing_receipt(plan)
    receipt["capture_generation"]["synthetic_or_diagnostic"] = True
    with pytest.raises(Stage2MeasurementError, match="immutable physical raw"):
        admit_stage2_relative_ps_candidate(plan, receipt)


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (["--dry-run"], 0),
        (["--execute-live"], 2),
        (
            [
                "--execute-live",
                "--confirm-user-present",
                "--confirm-volume-fixed",
                "--confirm-routing-and-geometry",
            ],
            2,
        ),
    ],
)
def test_cli_never_imports_or_calls_sounddevice(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    expected: int,
) -> None:
    import_count = 0
    backend_calls = 0
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal import_count
        if name == "sounddevice" or name.startswith("sounddevice."):
            import_count += 1
        return real_import(name, globals, locals, fromlist, level)

    fake = types.ModuleType("sounddevice")

    def backend_call(*_args, **_kwargs):
        nonlocal backend_calls
        backend_calls += 1
        raise AssertionError("sounddevice backend를 호출하면 안 됩니다")

    fake.query_devices = backend_call
    fake.Stream = backend_call
    fake.playrec = backend_call
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    spec = importlib.util.spec_from_file_location("stage2_measure_cli_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--config", str(CONFIG_PATH), *arguments]) == expected
    assert import_count == 0
    assert backend_calls == 0


def test_direct_execute_live_rechecks_confirmations_before_repository_or_audio(
    monkeypatch: pytest.MonkeyPatch, plan_and_pcm: tuple[dict, np.ndarray]
) -> None:
    plan, pcm = plan_and_pcm
    spec = importlib.util.spec_from_file_location("stage2_measure_direct_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "repository_execution_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("confirmation gate 전에 repository/audio 경로를 열면 안 됩니다")
        ),
    )
    assert (
        module._execute_live(
            config={},
            plan=plan,
            submitted_pcm=pcm,
            hardware_path="configs/hardware_jetson.yaml",
            confirmations={
                "user_present": True,
                "volume_fixed_after_meter_adjustment": False,
                "routing_and_geometry": True,
            },
        )
        == 2
    )
