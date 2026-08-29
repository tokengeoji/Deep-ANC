"""P/S 분석 대역 파라미터의 strict-v1 기본값과 광대역 확장 경로를 검증한다.

실제 오디오 장치나 저장 artifact는 열지 않는다. 모든 신호는 메모리에서 합성한다.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.data import measure_paths_interleaved as mpi


FS = 48_000


def _exact_transfer(
    frequencies_hz: np.ndarray,
    fir: np.ndarray,
    *,
    delay_samples: int,
) -> np.ndarray:
    indices = np.arange(fir.size, dtype=np.float64) + float(delay_samples)
    return np.exp(
        -2j * np.pi * np.outer(frequencies_hz, indices) / float(FS)
    ) @ fir


def test_compact_round_trip_default_is_exact_strict_v1_contract() -> None:
    frequencies = np.arange(100.0, 3000.0, 10.0, dtype=np.float64)
    fir = np.asarray((0.7, -0.2, 0.1), dtype=np.float64)
    measured = _exact_transfer(frequencies, fir, delay_samples=11)

    implicit = mpi.compact_transfer_round_trip(
        frequencies,
        measured,
        fir,
        effective_delay_samples=11,
        sample_rate=FS,
        band_hz=(150.0, 1600.0),
    )
    explicit = mpi.compact_transfer_round_trip(
        frequencies,
        measured,
        fir,
        effective_delay_samples=11,
        sample_rate=FS,
        band_hz=(150.0, 1600.0),
        required_subbands_hz=mpi.COMPACT_TRANSFER_SUB_BANDS_HZ,
    )

    assert implicit == explicit
    assert tuple(tuple(row["band_hz"]) for row in implicit["subbands"]) == (
        mpi.COMPACT_TRANSFER_SUB_BANDS_HZ
    )


def test_compact_round_trip_accepts_explicit_panel_subbands() -> None:
    frequencies = np.arange(1200.0, 3300.0, 10.0, dtype=np.float64)
    fir = np.asarray((0.6, -0.15, 0.04), dtype=np.float64)
    measured = _exact_transfer(frequencies, fir, delay_samples=7)
    panel_subbands = ((1600.0, 2100.0), (2100.0, 2800.0))

    result = mpi.compact_transfer_round_trip(
        frequencies,
        measured,
        fir,
        effective_delay_samples=7,
        sample_rate=FS,
        band_hz=(1500.0, 2900.0),
        required_subbands_hz=panel_subbands,
    )

    assert result["passed"] is True
    assert tuple(tuple(row["band_hz"]) for row in result["subbands"]) == (
        panel_subbands
    )


def _crosscheck_fixture() -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray
]:
    frequencies = np.arange(100.0, 3300.0, 10.0, dtype=np.float64)
    rng = np.random.default_rng(20260828)
    base = rng.normal(size=(3, frequencies.size)) + 1j * rng.normal(
        size=(3, frequencies.size)
    )
    stacks = {"noise": base.copy(), "cancel": (0.8 - 0.2j) * base}
    return (
        {"noise": frequencies.copy(), "cancel": frequencies.copy()},
        stacks,
        np.ones(base.shape[0], dtype=bool),
    )


def test_crosscheck_default_is_exact_strict_v1_contract() -> None:
    frequencies, stacks, keep = _crosscheck_fixture()
    crosscheck = {name: values.copy() for name, values in stacks.items()}

    implicit = mpi.separation_crosscheck_metrics(
        frequencies=frequencies,
        joint_stacks=stacks,
        resampled_stacks=crosscheck,
        keep=keep,
    )
    explicit = mpi.separation_crosscheck_metrics(
        frequencies=frequencies,
        joint_stacks=stacks,
        resampled_stacks=crosscheck,
        keep=keep,
        subbands_hz=mpi.COMPACT_TRANSFER_SUB_BANDS_HZ,
        overall_band_hz=mpi.SEPARATION_CROSSCHECK_OVERALL_BAND_HZ,
    )

    assert implicit == explicit


def test_crosscheck_accepts_explicit_panel_subbands() -> None:
    frequencies, stacks, keep = _crosscheck_fixture()
    crosscheck = {name: values.copy() for name, values in stacks.items()}
    panel_subbands = ((1800.0, 2200.0), (2200.0, 2800.0))

    result = mpi.separation_crosscheck_metrics(
        frequencies=frequencies,
        joint_stacks=stacks,
        resampled_stacks=crosscheck,
        keep=keep,
        subbands_hz=panel_subbands,
        overall_band_hz=(1700.0, 2900.0),
    )

    for drive in ("noise", "cancel"):
        assert tuple(row["band_hz"] for row in result[drive]["subbands"]) == (
            panel_subbands
        )
        assert result[drive]["overall"]["band_hz"] == (1700.0, 2900.0)


def _periodic_band_signal(
    *,
    period_samples: int,
    low_shift_per_period: int,
    high_shift_per_period: int,
    periods: int,
) -> np.ndarray:
    rng = np.random.default_rng(6114)
    grid = np.fft.rfftfreq(period_samples, d=1.0 / FS)

    def random_band(low: float, high: float) -> np.ndarray:
        spectrum = np.zeros(grid.size, dtype=np.complex128)
        selected = np.flatnonzero((grid >= low) & (grid <= high))[::7]
        spectrum[selected] = rng.normal(size=selected.size) + 1j * rng.normal(
            size=selected.size
        )
        return np.fft.irfft(spectrum, n=period_samples)

    low = random_band(200.0, 1500.0)
    high = random_band(6000.0, 10_000.0)
    rows = [
        np.roll(low, index * low_shift_per_period)
        + np.roll(high, index * high_shift_per_period)
        for index in range(periods)
    ]
    return np.concatenate(rows)


def test_clock_observer_uses_explicit_band_instead_of_v1_global() -> None:
    period_samples = 6000
    periods = 10
    signal = _periodic_band_signal(
        period_samples=period_samples,
        low_shift_per_period=3,
        high_shift_per_period=1,
        periods=periods,
    )
    probe = SimpleNamespace(sample_rate=FS, period_samples=period_samples)
    starts = [index * period_samples for index in range(periods)]

    low = mpi.observe_period_clock_ratios(
        err=signal,
        ref=signal,
        probe=probe,
        period_starts=starts,
        max_drift_deviation_samples=2.0,
        min_valid_periods=8,
    )
    high_band = (6000.0, 10_000.0)
    high = mpi.observe_period_clock_ratios(
        err=signal,
        ref=signal,
        probe=probe,
        period_starts=starts,
        max_drift_deviation_samples=2.0,
        min_valid_periods=8,
        clock_band_hz=high_band,
    )

    np.testing.assert_array_equal(low["clock_band_hz"], mpi.CLOCK_BAND_HZ)
    np.testing.assert_array_equal(high["clock_band_hz"], high_band)
    assert abs(float(low["drift_samples_per_period"])) == pytest.approx(
        3.0, abs=0.15
    )
    assert abs(float(high["drift_samples_per_period"])) == pytest.approx(
        1.0, abs=0.15
    )


@pytest.mark.parametrize(
    "call",
    (
        lambda frequencies, measured, fir: mpi.compact_transfer_round_trip(
            frequencies,
            measured,
            fir,
            effective_delay_samples=0,
            sample_rate=FS,
            band_hz=(150.0, 1600.0),
            required_subbands_hz=(),
        ),
        lambda frequencies, measured, fir: mpi.separation_crosscheck_metrics(
            frequencies={"noise": frequencies, "cancel": frequencies},
            joint_stacks={
                "noise": measured[None, :],
                "cancel": measured[None, :],
            },
            resampled_stacks={
                "noise": measured[None, :],
                "cancel": measured[None, :],
            },
            keep=np.ones(1, dtype=bool),
            subbands_hz=(),
        ),
    ),
)
def test_empty_required_band_sets_fail_closed(call) -> None:
    frequencies = np.arange(100.0, 2000.0, 10.0, dtype=np.float64)
    fir = np.asarray((1.0,), dtype=np.float64)
    measured = _exact_transfer(frequencies, fir, delay_samples=0)

    with pytest.raises(ValueError, match="비었거나"):
        call(frequencies, measured, fir)
