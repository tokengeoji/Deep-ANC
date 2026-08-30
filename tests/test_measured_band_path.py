"""측정대역 전용 P/S response의 보간·외삽 차단 계약."""

from __future__ import annotations

import math
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from deep_anc.dsp.control_band_contract import ControlBandContract
from deep_anc.dsp.measured_band_path import (
    MEASURED_BAND_INTERPOLATION_SCHEMA,
    MeasuredBandPath,
    MeasuredBandPathData,
    load_measured_band_path,
)


FS = 48_000
CONTRACT = ControlBandContract.broadband_point_control()


def _known_transfer(frequency: np.ndarray, *, delay: float) -> np.ndarray:
    # Delay 제거 뒤에는 짧고 매끄러운 residual이다. wrapped phase 자체를 선형
    # 보간하면 16 Hz마다 약 pi가 회전해 이 fixture도 즉시 실패한다.
    omega = 2.0 * np.pi * frequency / FS
    residual = 0.7 - 0.12 * np.exp(-1j * omega) + 0.04 * np.exp(-2j * omega)
    return residual * np.exp(-1j * omega * delay)


def _data(*, role: str = "secondary", delay: float = 1501.375) -> MeasuredBandPathData:
    frequency = np.arange(104.0, 11_400.0 + 0.1, 16.0, dtype=np.float64)
    bulk_integer = int(round(delay))
    pre_roll = 256
    return MeasuredBandPathData.from_arrays(
        role=role,
        sample_rate=FS,
        frequencies_hz=frequency,
        transfer=_known_transfer(frequency, delay=delay),
        bulk_delay_samples=bulk_integer,
        bulk_delay_fractional_samples=delay,
        pre_roll_samples=pre_roll,
        effective_delay_samples=bulk_integer - pre_roll,
        fractional_effective_delay_samples=delay - pre_roll,
        delay_semantics="effective_zeros_before_compact_fir",
        valid_band_hz=CONTRACT.point_control_target_hz,
        control_band_contract_sha256=CONTRACT.digest(),
        source_analysis_sha256="a" * 64,
        plant_evidence_sha256="b" * 64,
        subbands_hz=CONTRACT.point_control_subbands_hz,
        source_path="fixture.npz",
    )


def test_delay_removed_linear_interpolation_recovers_known_response() -> None:
    data = _data()
    assert data.holdout_receipt["passed"] is True
    assert data.holdout_receipt["interpolation_schema"] == (
        MEASURED_BAND_INTERPOLATION_SCHEMA
    )
    assert len(data.holdout_receipt["rows"]) == 7

    path = MeasuredBandPath(data)
    query = torch.tensor([150.0, 333.0, 1999.0, 7999.0, 11_313.0])
    actual = path.response_at(query).detach().numpy()
    expected = _known_transfer(query.numpy(), delay=data.bulk_delay_fractional_samples)
    assert actual == pytest.approx(expected, rel=2.0e-4, abs=8.0e-5)


@pytest.mark.parametrize("query", ([149.0, 200.0], [11_000.0, 11_314.0]))
def test_response_refuses_any_query_outside_valid_band(query: list[float]) -> None:
    with pytest.raises(ValueError, match="valid band 밖"):
        MeasuredBandPath(_data()).response_at(torch.tensor(query))


def test_valid_band_cannot_extend_outside_measured_convex_hull() -> None:
    frequency = np.arange(200.0, 11_000.0, 16.0)
    with pytest.raises(ValueError, match="convex hull 밖"):
        MeasuredBandPathData.from_arrays(
            role="secondary",
            sample_rate=FS,
            frequencies_hz=frequency,
            transfer=_known_transfer(frequency, delay=100.0),
            bulk_delay_samples=100,
            bulk_delay_fractional_samples=100.0,
            pre_roll_samples=64,
            effective_delay_samples=36,
            fractional_effective_delay_samples=36.0,
            delay_semantics="effective_zeros_before_compact_fir",
            valid_band_hz=CONTRACT.point_control_target_hz,
            control_band_contract_sha256=CONTRACT.digest(),
            source_analysis_sha256="a" * 64,
            plant_evidence_sha256="b" * 64,
            subbands_hz=CONTRACT.point_control_subbands_hz,
            source_path="fixture.npz",
        )


def test_time_domain_forward_is_structurally_unavailable() -> None:
    path = MeasuredBandPath(_data())
    with pytest.raises(RuntimeError, match="시간영역 convolution"):
        path(torch.zeros(1, 1, 1024))


def test_response_identity_changes_with_role_or_source_sha() -> None:
    secondary = _data(role="secondary")
    primary = _data(role="primary")
    assert secondary.response_sha256 != primary.response_sha256
    assert len(secondary.response_sha256) == 64
    with pytest.raises(ValueError, match="read-only"):
        secondary.frequencies_hz[0] = 0.0


def test_extra_handoff_delay_is_applied_only_as_phase() -> None:
    data = _data()
    query = torch.tensor([2000.0], dtype=torch.float64)
    base = MeasuredBandPath(data).response_at(query)
    shifted = MeasuredBandPath(data, extra_delay_samples=256).response_at(query)
    expected = torch.exp(
        torch.tensor(-2j * math.pi * 2000.0 * 256.0 / FS, dtype=torch.complex128)
    )
    assert shifted == pytest.approx(base * expected, rel=1.0e-12, abs=1.0e-12)


def test_response_uses_full_bulk_then_adds_handoff_once_not_effective_delay() -> None:
    data = _data(delay=1501.375)
    query = torch.tensor([1999.0, 7999.0], dtype=torch.float64)
    actual = MeasuredBandPath(data, extra_delay_samples=256).response_at(query)
    expected = torch.from_numpy(
        _known_transfer(query.numpy(), delay=data.bulk_delay_fractional_samples)
    ) * torch.exp(-2j * math.pi * query * 256.0 / FS)
    wrongly_double_counted = torch.from_numpy(
        _known_transfer(query.numpy(), delay=data.bulk_delay_fractional_samples)
    ) * torch.exp(
        -2j
        * math.pi
        * query
        * (data.fractional_effective_delay_samples + 256.0)
        / FS
    )
    assert actual == pytest.approx(expected, rel=2.0e-4, abs=8.0e-5)
    assert not torch.allclose(actual, wrongly_double_counted, rtol=1.0e-3, atol=1.0e-3)


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"bulk_delay_samples": 1500}, "bulk integer/fractional"),
        ({"effective_delay_samples": 1000}, "bulk integer - pre-roll"),
        ({"fractional_effective_delay_samples": 1000.0}, "bulk fractional"),
        ({"delay_semantics": "unknown"}, "delay_semantics"),
    ),
)
def test_delay_metadata_relations_are_fail_closed(
    override: dict[str, object], message: str
) -> None:
    frequency = np.arange(104.0, 11_400.0 + 0.1, 16.0, dtype=np.float64)
    kwargs: dict[str, object] = {
        "role": "secondary",
        "sample_rate": FS,
        "frequencies_hz": frequency,
        "transfer": _known_transfer(frequency, delay=1501.375),
        "bulk_delay_samples": 1501,
        "bulk_delay_fractional_samples": 1501.375,
        "pre_roll_samples": 256,
        "effective_delay_samples": 1245,
        "fractional_effective_delay_samples": 1245.375,
        "delay_semantics": "effective_zeros_before_compact_fir",
        "valid_band_hz": CONTRACT.point_control_target_hz,
        "control_band_contract_sha256": CONTRACT.digest(),
        "source_analysis_sha256": "a" * 64,
        "plant_evidence_sha256": "b" * 64,
        "subbands_hz": CONTRACT.point_control_subbands_hz,
        "source_path": "fixture.npz",
    }
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        MeasuredBandPathData.from_arrays(**kwargs)


def test_loader_requires_canonical_measured_fields_and_diagnostic_compact(
    tmp_path: Path,
) -> None:
    frequency = np.arange(104.0, 11_400.0 + 0.1, 16.0, dtype=np.float64)
    transfer = _known_transfer(frequency, delay=1501.375)
    path = tmp_path / "secondary.npz"
    np.savez(
        path,
        schema_version=np.asarray("broadband_measured_band_plant_v2_raw_derived"),
        plant_role=np.asarray("secondary"),
        sample_rate=np.asarray(FS),
        measured_frequencies_hz=frequency,
        measured_transfer_real=transfer.real,
        measured_transfer_imag=transfer.imag,
        aligned_mean_transfer_sha256=np.asarray(
            hashlib.sha256(
                np.asarray(transfer, dtype=np.complex128).tobytes(order="C")
            ).hexdigest()
        ),
        bulk_delay_samples=np.asarray(1501),
        bulk_delay_fractional_samples=np.asarray(1501.375),
        pre_roll_samples=np.asarray(256),
        effective_delay_samples=np.asarray(1245),
        fractional_effective_delay_samples=np.asarray(1245.375),
        delay_semantics=np.asarray("effective_zeros_before_compact_fir"),
        compact_role=np.asarray("diagnostic_only"),
        compact_training_eligible=np.asarray(False),
        control_band_contract_sha256=np.asarray(CONTRACT.digest()),
        source_analysis_npz_sha256=np.asarray("a" * 64),
        broadband_plant_evidence_sha256=np.asarray("b" * 64),
    )
    loaded = load_measured_band_path(
        path,
        role="secondary",
        valid_band_hz=CONTRACT.point_control_target_hz,
        subbands_hz=CONTRACT.point_control_subbands_hz,
    )
    assert loaded.fractional_effective_delay_samples == pytest.approx(1245.375)
    assert loaded.delay_semantics == "effective_zeros_before_compact_fir"
