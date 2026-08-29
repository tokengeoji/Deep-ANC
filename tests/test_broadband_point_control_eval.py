from __future__ import annotations

import numpy as np
import pytest

from deep_anc.dsp.control_band_contract import ControlBandContract
from deep_anc.eval.broadband_point_control import (
    BroadbandControlSegment,
    evaluate_broadband_point_control_segments,
)


FS = 48_000
N = 8192
FREQUENCIES = (225.0, 450.0, 800.0, 1300.0, 2200.0, 4000.0, 8000.0)
BAND_WIDTHS = (150.0, 300.0, 400.0, 600.0, 1228.42712474619, 2828.42712474619, 5656.85424949238)


def _signals(dl_scale=None, fx_scale=None, *, omit_band: int | None = None):
    dl_scale = np.full(7, 0.40) if dl_scale is None else np.asarray(dl_scale)
    fx_scale = np.full(7, 0.65) if fx_scale is None else np.asarray(fx_scale)
    time = np.arange(N, dtype=np.float64) / FS
    tones = []
    for index, frequency in enumerate(FREQUENCIES):
        # 각 band 전력을 폭에 비례시켜 flat power spectral density를 만든다.
        amplitude = 0.0 if index == omit_band else 0.003 * np.sqrt(BAND_WIDTHS[index] / BAND_WIDTHS[0])
        tones.append(amplitude * np.sin(2.0 * np.pi * frequency * time + index * 0.2))
    tones = np.asarray(tones)
    return tones.sum(axis=0), (dl_scale[:, None] * tones).sum(axis=0), (
        fx_scale[:, None] * tones
    ).sum(axis=0)


def _campaign(
    *,
    dl_scale=None,
    fx_scale=None,
    omit_band: int | None = None,
    positions=("center",),
):
    d, dl, fx = _signals(dl_scale, fx_scale, omit_band=omit_band)
    rows = []
    for position in positions:
        for family in ("speech", "music", "environment", "machine"):
            for group in range(4):
                rows.append(
                    BroadbandControlSegment(
                        session_id=f"{position}-{family}-{group}",
                        source_family=family,
                        group_id=f"{family}-{group}",
                        error_position_id=position,
                        sample_rate=FS,
                        disturbance_off=d,
                        error_deep_anc=dl,
                        error_fxlms=fx,
                    )
                )
    return rows


def _evaluate(rows, *, spatial=False):
    return evaluate_broadband_point_control_segments(
        rows,
        contract=ControlBandContract.broadband_point_control(),
        require_spatial=spatial,
        n_resamples=200,
    )


def test_every_low_and_high_band_passes_with_matched_fxlms_superiority():
    result = _evaluate(_campaign())
    assert result["status"] == "PASS"
    assert len(result["cells"]) == 4 * 7
    assert all(cell["positive_attenuation_pass"] for cell in result["cells"])
    assert all(cell["passed"] for cell in result["cells"])


def test_highband_failure_cannot_be_hidden_by_good_lowband_average():
    scales = np.full(7, 0.20)
    scales[-1] = 1.20
    result = _evaluate(_campaign(dl_scale=scales))
    assert result["status"] == "BLOCKED"
    failed = [cell for cell in result["cells"] if not cell["passed"]]
    assert failed
    assert all(cell["band_hz"][0] >= 5656.0 for cell in failed)


def test_lowband_failure_cannot_be_hidden_by_good_highband_average():
    scales = np.full(7, 0.20)
    scales[0] = 1.20
    result = _evaluate(_campaign(dl_scale=scales))
    assert result["status"] == "BLOCKED"
    assert any(cell["band_hz"] == [150.0, 300.0] and not cell["passed"] for cell in result["cells"])


def test_positive_highband_attenuation_is_not_enough_when_fxlms_is_better():
    dl = np.full(7, 0.40)
    fx = np.full(7, 0.65)
    fx[4:] = 0.20
    result = _evaluate(_campaign(dl_scale=dl, fx_scale=fx))
    assert result["status"] == "BLOCKED"
    high = [cell for cell in result["cells"] if cell["band_hz"][0] >= 1600.0]
    assert all(cell["positive_attenuation_pass"] for cell in high)
    assert all(not cell["matched_fxlms_superiority_pass"] for cell in high)


def test_missing_target_energy_blocks_only_the_uncovered_band_cells():
    result = _evaluate(_campaign(omit_band=6))
    assert result["status"] == "BLOCKED"
    high_last = [cell for cell in result["cells"] if cell["band_hz"][0] >= 5656.0]
    assert high_last
    assert all(not cell["coverage_pass"] for cell in high_last)


def test_multiple_positions_cannot_be_averaged_as_single_point():
    rows = _campaign(positions=("center", "y_plus"))
    with pytest.raises(ValueError, match="ERR 위치 하나"):
        _evaluate(rows)


def test_spatial_claim_requires_five_positions_and_each_position_passes():
    four = _campaign(positions=("center", "y_plus", "y_minus", "z_plus"))
    blocked = _evaluate(four, spatial=True)
    assert blocked["status"] == "BLOCKED"
    assert not blocked["spatial_position_count_pass"]

    five = _campaign(
        positions=("center", "y_plus", "y_minus", "z_plus", "z_minus")
    )
    passed = _evaluate(five, spatial=True)
    assert passed["status"] == "PASS"
    assert passed["spatial_position_count_pass"]


def test_group_floor_cannot_be_weakened():
    with pytest.raises(ValueError, match="4"):
        evaluate_broadband_point_control_segments(
            _campaign(),
            contract=ControlBandContract.broadband_point_control(),
            minimum_groups=3,
        )
