from __future__ import annotations

import dataclasses
import hashlib
import math

import numpy as np
import pytest

from deep_anc.dsp.stage2_2khz_contract import Stage2TwoKilohertzContract
from deep_anc.eval.stage2_2khz import (
    ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ,
    STAGE2_2KHZ_RAW_EVAL_DOMAIN,
    Stage2TwoKilohertzRawSegment,
    evaluate_stage2_2khz_raw_segments,
)


FS = 48_000
SAMPLES = 32_768


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _band_components(*, omit_objective: int | None = None) -> np.ndarray:
    contract = Stage2TwoKilohertzContract.canonical()
    bands = (
        *contract.octave_objective_bands_hz,
        *contract.do_no_harm_octave_bands_hz,
    )
    frequencies = np.fft.rfftfreq(SAMPLES, d=1.0 / FS)
    components: list[np.ndarray] = []
    for index, (lower, upper) in enumerate(bands):
        spectrum = np.zeros(frequencies.shape, dtype=np.complex128)
        include_upper = index == len(bands) - 1
        mask = (frequencies >= lower) & (
            frequencies <= upper if include_upper else frequencies < upper
        )
        if omit_objective != index:
            phase = 0.001 * np.flatnonzero(mask) + 0.17 * index
            spectrum[mask] = 0.02 * np.exp(1j * phase)
        components.append(np.fft.irfft(spectrum, n=SAMPLES, norm="ortho"))
    return np.asarray(components)


def _campaign(
    *,
    attenuation_db: tuple[float, ...] = (8.0, 8.0, 8.0, 8.0, 8.0, 0.0, 0.0),
    omit_objective: int | None = None,
    groups: int = 4,
    families: tuple[str, ...] = ("speech", "music", "environment", "machine"),
    position: str = "err_center",
) -> list[Stage2TwoKilohertzRawSegment]:
    assert len(attenuation_db) == 7
    contract = Stage2TwoKilohertzContract.canonical()
    components = _band_components(omit_objective=omit_objective)
    disturbance = np.sum(components, axis=0)
    scales = np.asarray(
        [10.0 ** (-float(value) / 20.0) for value in attenuation_db],
        dtype=np.float64,
    )
    error_on = np.sum(scales[:, None] * components, axis=0)
    segments: list[Stage2TwoKilohertzRawSegment] = []
    for family in families:
        for group in range(groups):
            identity = f"{position}-{family}-{group}"
            segments.append(
                Stage2TwoKilohertzRawSegment(
                    session_id=identity,
                    source_family=family,
                    group_id=f"{family}-component-{group}",
                    error_position_id=position,
                    sample_rate=FS,
                    disturbance_off=disturbance,
                    error_on=error_on,
                    raw_artifact_sha256=_sha(f"raw-{identity}"),
                    model_artifact_sha256=_sha("model"),
                    stage2_plant_binding_sha256=_sha("stage2-plant"),
                    control_band_contract_sha256=contract.digest(),
                )
            )
    return segments


def _evaluate(segments: list[Stage2TwoKilohertzRawSegment]):
    return evaluate_stage2_2khz_raw_segments(
        segments,
        contract=Stage2TwoKilohertzContract.canonical(),
        n_resamples=200,
    )


def test_all_low_octaves_two_khz_minimum_and_dnh_pass() -> None:
    result = _evaluate(_campaign())

    assert result["status"] == "PASS"
    assert result["single_point_only"] is True
    assert result["spatial_quiet_zone_claim"] is False
    assert result["full_octave_v3_claim"] is False
    assert result["deployment_runtime_claim"] is False
    assert len(result["objective_cells"]) == 4 * 5
    assert len(result["one_point_six_khz_sentinel_cells"]) == 4
    assert len(result["do_no_harm_cells"]) == 4 * 2
    assert result["all_source_density_and_group_gates_pass"] is True
    assert result["all_attenuation_objectives_pass"] is True
    assert result["all_one_point_six_khz_sentinel_gates_pass"] is True
    assert result["all_do_no_harm_gates_pass"] is True
    assert float(result["minimum_frequency_gate_margin_db"]) > 0.9
    assert float(result["two_khz_family_equal_mean_attenuation_db"]) > 3.9
    selection = result["checkpoint_selection_policy"]
    assert selection["two_khz_positive_is_secondary_diagnostic"] is True
    assert selection["one_point_six_khz_minimum_attenuation_db"] == 6.0
    assert selection["eligibility_requires_external_physical_runtime_latency_gate_pass"] is True
    assert selection["eligibility_requires_one_point_six_khz_sentinel_pass"] is True
    assert selection["one_point_six_khz_sentinel_runtime_exact_zero_required"] is True
    assert selection["eligible_from_this_frequency_report_alone"] is False
    assert selection["primary_order"] == "maximize_minimum_frequency_gate_margin_db"
    assert selection["secondary_order"] == (
        "maximize_two_khz_family_equal_mean_attenuation_db"
    )
    two_khz = [
        cell for cell in result["objective_cells"] if cell["octave_center_hz"] == 2000.0
    ]
    assert len(two_khz) == 4
    assert all(cell["attenuation_threshold_db"] == 0.0 for cell in two_khz)
    assert all(cell["attenuation_threshold_comparator"] == ">" for cell in two_khz)
    assert all(float(cell["attenuation_worst10_mean_db"]) > 3.9 for cell in two_khz)


def test_one_point_six_khz_near_zero_is_blocked_even_if_two_khz_octave_passes() -> None:
    rows = _campaign(
        attenuation_db=(8.0, 8.0, 8.0, 8.0, 12.0, 0.0, 0.0)
    )
    frequencies = np.fft.rfftfreq(SAMPLES, d=1.0 / FS)
    lower, upper = ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ
    selected = (frequencies >= lower) & (frequencies < upper)
    for index, row in enumerate(rows):
        disturbance_spectrum = np.fft.rfft(row.disturbance_off)
        error_spectrum = np.fft.rfft(row.error_on)
        error_spectrum[selected] = disturbance_spectrum[selected]
        rows[index] = dataclasses.replace(
            row,
            error_on=np.fft.irfft(error_spectrum, n=SAMPLES),
        )

    result = _evaluate(rows)
    two_khz = [
        cell for cell in result["objective_cells"]
        if cell["octave_center_hz"] == 2000.0
    ]
    sentinel = result["one_point_six_khz_sentinel_cells"]
    assert all(cell["passed"] for cell in two_khz)
    assert all(not cell["passed"] for cell in sentinel)
    assert result["all_one_point_six_khz_sentinel_gates_pass"] is False
    assert result["status"] == "BLOCKED"


def test_one_point_six_khz_six_db_is_the_hard_floor() -> None:
    result = _evaluate(
        _campaign(attenuation_db=(8.0, 8.0, 8.0, 8.0, 5.99, 0.0, 0.0))
    )

    assert result["status"] == "BLOCKED"
    sentinel = result["one_point_six_khz_sentinel_cells"]
    assert all(cell["attenuation_threshold_db"] == 6.0 for cell in sentinel)
    assert all(cell["attenuation_threshold_comparator"] == ">=" for cell in sentinel)
    assert all(not cell["passed"] for cell in sentinel)


def test_two_khz_source_density_missing_blocks_even_when_other_bands_are_good() -> None:
    result = _evaluate(_campaign(omit_objective=4))

    assert result["status"] == "BLOCKED"
    two_khz = [
        cell for cell in result["objective_cells"] if cell["octave_center_hz"] == 2000.0
    ]
    assert all(not cell["source_density_and_group_coverage_pass"] for cell in two_khz)
    low = [
        cell for cell in result["objective_cells"] if cell["octave_center_hz"] < 2000.0
    ]
    assert all(cell["passed"] for cell in low)


def test_good_two_khz_cannot_hide_low_octave_amplification() -> None:
    result = _evaluate(
        _campaign(attenuation_db=(-1.0, 8.0, 8.0, 8.0, 8.0, 0.0, 0.0))
    )

    assert result["status"] == "BLOCKED"
    failed = [cell for cell in result["objective_cells"] if not cell["passed"]]
    assert failed
    assert {cell["octave_center_hz"] for cell in failed} == {125.0}


def test_two_khz_below_old_three_db_is_allowed_when_positive() -> None:
    result = _evaluate(
        _campaign(attenuation_db=(8.0, 8.0, 8.0, 8.0, 2.9, 0.0, 0.0))
    )

    assert result["status"] == "BLOCKED"
    two_khz = [
        cell for cell in result["objective_cells"] if cell["octave_center_hz"] == 2000.0
    ]
    assert all(cell["attenuation_mean_pass"] for cell in two_khz)
    assert all(cell["attenuation_worst10_pass"] for cell in two_khz)
    assert all(cell["attenuation_ci_lower_pass"] for cell in two_khz)
    assert all(cell["passed"] for cell in two_khz)
    assert not result["all_one_point_six_khz_sentinel_gates_pass"]


def test_four_or_eight_khz_one_db_amplification_limit_is_strict() -> None:
    failed = _evaluate(
        _campaign(attenuation_db=(8.0, 8.0, 8.0, 8.0, 8.0, -1.1, 0.0))
    )
    assert failed["status"] == "BLOCKED"
    four_khz = [
        cell for cell in failed["do_no_harm_cells"] if cell["octave_center_hz"] == 4000.0
    ]
    assert all(not cell["do_no_harm_pass"] for cell in four_khz)
    assert all(float(cell["worst10_amplification_db"]) > 1.0 for cell in four_khz)

    passed = _evaluate(
        _campaign(attenuation_db=(8.0, 8.0, 8.0, 8.0, 8.0, -0.9, -0.9))
    )
    assert passed["status"] == "PASS"


def test_family_octave_group_floor_and_missing_family_fail_closed() -> None:
    underpowered = _evaluate(_campaign(groups=3))
    assert underpowered["status"] == "BLOCKED"
    assert all(
        not cell["source_density_and_group_coverage_pass"]
        for cell in underpowered["objective_cells"]
    )

    missing_family = _evaluate(
        _campaign(families=("speech", "music", "environment"))
    )
    assert missing_family["status"] == "BLOCKED"
    machine = [
        cell
        for cell in missing_family["objective_cells"]
        if cell["source_family"] == "machine"
    ]
    assert machine and all(not cell["passed"] for cell in machine)


def test_group_floor_argument_cannot_be_relaxed() -> None:
    with pytest.raises(ValueError, match="하한 4"):
        evaluate_stage2_2khz_raw_segments(
            _campaign(),
            contract=Stage2TwoKilohertzContract.canonical(),
            minimum_groups=3,
        )


def test_multi_position_cannot_be_promoted_to_single_point() -> None:
    rows = _campaign(position="err_center") + _campaign(position="err_offset")
    with pytest.raises(ValueError, match="ERR 위치 하나"):
        _evaluate(rows)


def test_legacy_domain_and_wrong_contract_sha_are_rejected() -> None:
    rows = _campaign()
    rows[0] = dataclasses.replace(rows[0], evaluation_domain="legacy_diagnostic")
    with pytest.raises(ValueError, match="legacy/surrogate"):
        _evaluate(rows)

    rows = _campaign()
    rows[0] = dataclasses.replace(rows[0], control_band_contract_sha256="0" * 64)
    with pytest.raises(ValueError, match="contract SHA"):
        _evaluate(rows)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("model_artifact_sha256", _sha("other-model"), "model artifact"),
        ("stage2_plant_binding_sha256", _sha("other-plant"), "P/S binding"),
    ],
)
def test_mixed_model_or_plant_campaign_is_rejected(
    field: str, value: str, match: str
) -> None:
    rows = _campaign()
    rows[0] = dataclasses.replace(rows[0], **{field: value})
    with pytest.raises(ValueError, match=match):
        _evaluate(rows)


def test_same_raw_artifact_cannot_count_as_two_independent_groups() -> None:
    rows = _campaign()
    rows[1] = dataclasses.replace(
        rows[1], raw_artifact_sha256=rows[0].raw_artifact_sha256
    )
    with pytest.raises(ValueError, match="서로 다른 독립 group"):
        _evaluate(rows)


def test_invalid_raw_shape_and_nonfinite_values_are_rejected() -> None:
    rows = _campaign()
    rows[0] = dataclasses.replace(rows[0], error_on=rows[0].error_on[:-256])
    with pytest.raises(ValueError, match="길이가 다릅니다"):
        _evaluate(rows)

    rows = _campaign()
    invalid = rows[0].error_on.copy()
    invalid[0] = math.nan
    rows[0] = dataclasses.replace(rows[0], error_on=invalid)
    with pytest.raises(ValueError, match="NaN/Inf"):
        _evaluate(rows)
