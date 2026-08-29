from __future__ import annotations

from pathlib import Path

import pytest

from deep_anc.dsp.control_band_contract import OCTAVE_8K_UPPER_HZ
from scripts.data.measure_paths_broadband_interleaved import (
    BROADBAND_LIVE_AUTHORITY,
    build_signal_plan,
    main,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE = REPO_ROOT / "configs" / "hardware_jetson.yaml"


def test_broadband_signal_only_plan_covers_8khz_octave_within_50_seconds():
    plan, pcm = build_signal_plan(hardware_path=HARDWARE)

    assert plan["live_capture_enabled"] is False
    assert plan["role"] == "signal_only_dry_run_no_audio"
    assert len(plan["panels"]) == 5
    assert plan["recipe"]["in_panel_clock_pilot_band_hz"] == [150.0, 600.0]
    assert plan["recipe"]["panel_compact_fir_forbidden"] is True
    assert BROADBAND_LIVE_AUTHORITY is None
    assert plan["recipe"]["global_clock_input_domain"] == (
        "actual_submitted_int16_period_spectrum_not_intended_float"
    )
    # panel high tone을 더한 뒤 int16 양자화한 pilot은 panel마다 다르므로
    # intended float spectrum을 clock 분모로 대체하면 안 된다.
    assert plan["recipe"]["fixed_clock_pilot_pcm_exact_across_panels"] is False
    assert all(
        value > 0.0
        for value in plan["recipe"]["fixed_clock_pilot_pcm_max_panel_delta"].values()
    )
    null_receipt = plan["recipe"]["submitted_pilot_cross_channel_null"]
    assert null_receipt["all_panels_passed"] is True
    assert null_receipt["maximum_absolute_observed"] <= 1.0e-8
    assert null_receipt["maximum_ratio_observed"] <= 1.0e-12
    assert len(null_receipt["panel_receipts"]) == 5
    assert all(
        panel["noise_clock_pilot_tone_count"] >= 8
        and panel["cancel_clock_pilot_tone_count"] >= 8
        for panel in plan["panels"]
    )
    assert plan["panels"][-1]["noise_actual_band_hz"][1] >= OCTAVE_8K_UPPER_HZ
    assert plan["panels"][-1]["cancel_actual_band_hz"][1] >= OCTAVE_8K_UPPER_HZ
    assert plan["output"]["duration_seconds"] <= 50.0
    assert plan["output"]["frames"] % 256 == 0
    assert plan["layout"][0]["kind"] == "lead_in_silence"
    assert plan["layout"][1]["kind"] == "primary_nonperiodic_timing_marker"
    assert plan["layout"][1]["frames"] == 12_000
    assert plan["layout"][2]["kind"] == "primary_marker_tail_guard"
    assert plan["layout"][3]["kind"] == "secondary_nonperiodic_timing_marker"
    assert plan["layout"][4]["kind"] == "secondary_marker_tail_guard"
    assert plan["timing_markers"]["frequency_gcd_hz"] == 4.0
    assert plan["timing_markers"]["delay_alias_period_samples"] == 12_000
    assert sum(row["kind"] == "analysis_panel" for row in plan["layout"]) == 5
    assert sum(row["kind"] == "lowband_clock_anchor" for row in plan["layout"]) == 4
    assert plan["layout"][-1]["stop_frame"] == plan["output"]["frames"]
    assert pcm.shape == (plan["output"]["frames"], 2)
    assert str(pcm.dtype) == "int16"
    assert plan["output"]["peak_pcm"] <= 99
    assert plan["phase_budgets"]["8000"]["degrees_per_sample"] == pytest.approx(60.0)


def test_broadband_measurement_live_path_is_fail_closed(capsys):
    assert main([]) == 2
    captured = capsys.readouterr()
    assert "live 출력은 아직 잠겨" in captured.err


def test_broadband_signal_plan_rejects_amplitude_override():
    with pytest.raises(ValueError, match="0.003"):
        build_signal_plan(hardware_path=HARDWARE, amplitude=0.004)
