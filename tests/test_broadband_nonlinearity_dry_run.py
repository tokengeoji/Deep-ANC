from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.data.measure_broadband_nonlinearity import (
    DEFAULT_PEAK,
    IMD_PAIRS_HZ,
    RELATIVE_LEVELS_DB,
    build_signal_plan,
    main,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE = REPO_ROOT / "configs" / "hardware_jetson.yaml"


def test_nonlinearity_plan_is_deterministic_safe_and_under_50_seconds():
    first, first_pcm = build_signal_plan(hardware_path=HARDWARE)
    second, second_pcm = build_signal_plan(hardware_path=HARDWARE)

    assert first == second
    assert np.array_equal(first_pcm, second_pcm)
    assert first["role"] == "signal_only_dry_run_no_audio"
    assert first["live_capture_enabled"] is False
    assert first["safety"]["audio_device_opened"] is False
    assert first["hardware"]["sample_rate"] == 48_000
    assert first["hardware"]["block_size"] == 256
    assert first["hardware"]["latency"] == "low"
    assert first["output"]["duration_seconds"] == pytest.approx(48.0)
    assert first["output"]["duration_seconds"] < 50.0
    assert first["protocol"]["window_seconds"]["thd_all_paths"] == pytest.approx(18.0)
    assert first["protocol"]["window_seconds"]["imd_all_paths"] == pytest.approx(30.0)
    assert first["protocol"]["window_seconds"]["per_plant"] == pytest.approx(24.0)
    assert first["output"]["frames"] % 256 == 0
    assert first["output"]["peak_float"] <= DEFAULT_PEAK
    assert first["output"]["peak_pcm"] <= 99
    assert first_pcm.shape == (first["output"]["frames"], 2)
    assert first_pcm.dtype == np.int16
    assert len(first["output"]["pcm_sha256"]) == 64


def test_nonlinearity_plan_has_four_levels_for_separate_p_and_s_thd_imd():
    plan, pcm = build_signal_plan(hardware_path=HARDWARE)
    rows = [row for row in plan["layout"] if row["kind"] == "nonlinearity_measurement_slot"]

    for plant, channel in (("P", 0), ("S", 1)):
        thd = [row for row in rows if row["plant"] == plant and row["measurement"] == "THD_ESS"]
        imd = [row for row in rows if row["plant"] == plant and row["measurement"] == "IMD_TWO_TONE"]
        assert sorted(row["relative_level_db"] for row in thd) == list(RELATIVE_LEVELS_DB)
        assert len(imd) == len(IMD_PAIRS_HZ) * len(RELATIVE_LEVELS_DB)
        assert {tuple(row["tone_pair_hz"]) for row in imd} == set(IMD_PAIRS_HZ)
        for pair in IMD_PAIRS_HZ:
            pair_rows = [row for row in imd if tuple(row["tone_pair_hz"]) == pair]
            assert sorted(row["relative_level_db"] for row in pair_rows) == list(
                RELATIVE_LEVELS_DB
            )
        for row in thd + imd:
            other_channel = 1 - channel
            assert not np.any(pcm[row["start_frame"] : row["stop_frame"], other_channel])


def test_nonlinearity_live_path_is_fail_closed(capsys):
    assert main([]) == 2
    captured = capsys.readouterr()
    assert "live 출력은 잠겨" in captured.err


def test_nonlinearity_plan_rejects_peak_over_approved_ceiling():
    with pytest.raises(ValueError, match="0.003"):
        build_signal_plan(hardware_path=HARDWARE, peak=0.0031)


def test_nonlinearity_json_publish_is_no_replace(tmp_path, capsys):
    output = tmp_path / "plan.json"
    argv = ["--dry-run", "--hardware", str(HARDWARE), "--output", str(output)]

    assert main(argv) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "broadband_nonlinearity_signal_plan_v1"
    assert main(argv) == 2
    assert "덮어쓰지 않습니다" in capsys.readouterr().err
