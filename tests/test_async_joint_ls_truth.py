"""비동기 DAC/ADC clock에서 guard-1 joint-LS의 물리 truth matrix.

실제 audio device나 저장 artifact를 사용하지 않는다. 실제 측정과 같은 48 kHz,
0.003 peak의 submitted int16 probe를 서로 다른 P/S FIR에 통과시키고 ADC clock
drift, 느린 wander, 독립 noise를 합성한다. public 측정 함수가 전체/4개 필수
부대역의 복소 전달을 복원하는지, nominal-grid guard-1 방식과 q/crosscheck
변조는 실패하는지를 함께 고정한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from deep_anc.dsp.interleaved_probe import align_repeats, build_interleaved_probe
from scripts.data import measure_paths_interleaved as mpi


FS = 48_000
# 3840 samples gives exactly 64 tones per channel over the physical band while
# keeping the truth matrix quick enough for Jetson CI.
PERIOD_SECONDS = 0.08
AMPLITUDE = 0.003
REPEATS = 10
EXTRA_PERIODS = 2
FIT_BAND_HZ = (150.0, 1600.0)
TRUTH_BANDS_HZ = (
    FIT_BAND_HZ,
    (150.0, 300.0),
    (300.0, 600.0),
    (600.0, 1000.0),
    (1000.0, 1600.0),
)
DRIFT_PPM = (364, 729, -729)
MIN_TRUTH_AGREEMENT = 0.995
MAX_TRUTH_RELATIVE_ERROR = 0.10


def _probe():
    return build_interleaved_probe(
        sample_rate=FS,
        period_seconds=PERIOD_SECONDS,
        band_hz=(60.0, 1650.0),
        amplitude=AMPLITUDE,
        tone_spacing_hz=None,
    )


def _physical_paths(period_samples: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """서로 다른 gain/delay와 고역 notch를 가진 P/S truth를 만든다."""

    primary = np.zeros(80, dtype=np.float64)
    primary[17] = 0.72
    primary[23] = -0.18
    primary[51] = 0.08

    # 반경 0.6의 conjugate-zero 쌍을 20-sample 간격으로 배치한다. 따라서
    # secondary는 primary와 gain/형상/지연이 다르고 1.1~1.3 kHz에 뚜렷한
    # notch를 가지지만 truth metric의 분모가 0이 될 정도의 완전 영점은 아니다.
    notch_delay = 20
    notch_radius = 0.6
    notch_angle = 2.0 * np.pi * 1100.0 / FS * notch_delay
    secondary = np.zeros(80, dtype=np.float64)
    secondary[[9, 9 + notch_delay, 9 + 2 * notch_delay]] = -0.35 * np.asarray(
        [
            1.0,
            -2.0 * notch_radius * np.cos(notch_angle),
            notch_radius**2,
        ]
    )

    impulse_responses = {"noise": primary, "cancel": secondary}
    transfers = {
        role: np.fft.rfft(impulse, n=period_samples)
        for role, impulse in impulse_responses.items()
    }
    return impulse_responses, transfers


def _complex_metrics(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    estimate_norm = float(np.linalg.norm(estimate))
    truth_norm = float(np.linalg.norm(truth))
    assert estimate_norm > 0.0 and truth_norm > 0.0
    agreement = abs(complex(np.vdot(truth, estimate))) / (
        truth_norm * estimate_norm
    )
    relative_error = float(np.linalg.norm(estimate - truth)) / truth_norm
    return float(agreement), float(relative_error)


def _synthesise_case(ppm: int) -> dict[str, Any]:
    probe = _probe()
    n = int(probe.period_samples)
    assert n == int(round(FS * PERIOD_SECONDS))

    ideal_period = np.stack(
        (probe.noise_signal, probe.cancel_signal), axis=1
    ).astype(np.float32)
    submitted_period = np.rint(
        np.clip(ideal_period, -1.0, 1.0) * np.float32(np.iinfo(np.int16).max)
    ).astype(np.int16)
    submitted_float = submitted_period.astype(np.float64) / float(
        np.iinfo(np.int16).max
    )

    impulse_responses, truth_full = _physical_paths(n)
    error_period = np.fft.irfft(
        np.fft.rfft(submitted_float[:, 0]) * truth_full["noise"]
        + np.fft.rfft(submitted_float[:, 1]) * truth_full["cancel"],
        n=n,
    )
    # REF도 같은 ADC clock을 보되 ERR과 다른 acoustic mixture를 갖게 한다.
    reference_period = submitted_float[:, 0] + 0.7 * submitted_float[:, 1]

    period_delta = n * float(ppm) * 1.0e-6
    q = n / (n + period_delta)
    total_periods = REPEATS + EXTRA_PERIODS
    total_samples = total_periods * n
    sample_index = np.arange(total_samples, dtype=np.float64)
    wander = (
        0.08 * np.sin(2.0 * np.pi * sample_index / (0.8 * total_samples))
        + 0.03 * np.sin(2.0 * np.pi * sample_index / (3.7 * n))
    )
    playback_coordinates = np.mod(q * sample_index + wander, n)
    err = map_coordinates(
        error_period,
        [playback_coordinates],
        order=3,
        mode="grid-wrap",
        prefilter=True,
    )
    ref = map_coordinates(
        reference_period,
        [playback_coordinates],
        order=3,
        mode="grid-wrap",
        prefilter=True,
    )
    rng = np.random.default_rng(10_000 + int(ppm))
    err += rng.normal(0.0, 2.0e-7, total_samples)
    ref += rng.normal(0.0, 2.0e-7, total_samples)

    submitted = np.tile(submitted_period, (total_periods, 1))
    starts = [index * n for index in range(REPEATS)]
    frequencies, stacks, separation = mpi.fractional_joint_channel_stacks(
        err=err,
        ref=ref,
        output_pcm_int16=submitted,
        probe=probe,
        period_starts=starts,
        fit_band_hz=FIT_BAND_HZ,
        max_drift_deviation_samples=2.0,
        min_valid_periods=8,
    )
    valid = np.asarray(separation["valid"], dtype=bool)
    crosscheck = mpi.separation_crosscheck_metrics(
        frequencies=frequencies,
        joint_stacks=stacks,
        resampled_stacks=separation["crosscheck_transfers"],
        keep=valid,
    )

    truth_metrics: dict[str, list[dict[str, Any]]] = {}
    naive_metrics: dict[str, tuple[float, float]] = {}
    for role in ("noise", "cancel"):
        role_frequencies = np.asarray(frequencies[role], dtype=np.float64)
        aligned, _, _ = align_repeats(
            role_frequencies,
            np.asarray(stacks[role])[valid],
            sample_rate=FS,
            fit_band_hz=FIT_BAND_HZ,
            anchor=0,
        )
        estimate = np.mean(aligned, axis=0)
        truth = truth_full[role][probe.bins_for(role)]
        rows = []
        for low, high in TRUTH_BANDS_HZ:
            mask = (role_frequencies >= low) & (role_frequencies <= high)
            agreement, relative_error = _complex_metrics(
                estimate[mask], truth[mask]
            )
            rows.append(
                {
                    "band_hz": (low, high),
                    "agreement": agreement,
                    "relative_error": relative_error,
                }
            )
        truth_metrics[role] = rows

        naive_frequencies, naive_stack = mpi.channel_stack(
            err=err,
            probe=probe,
            drive=role,
            period_starts=starts,
        )
        naive_aligned, _, _ = align_repeats(
            naive_frequencies,
            naive_stack,
            sample_rate=FS,
            fit_band_hz=FIT_BAND_HZ,
            anchor=0,
        )
        high_band = (naive_frequencies >= 1000.0) & (
            naive_frequencies <= 1600.0
        )
        naive_metrics[role] = _complex_metrics(
            np.mean(naive_aligned, axis=0)[high_band], truth[high_band]
        )

    return {
        "ppm": ppm,
        "probe": probe,
        "ideal_period": ideal_period,
        "submitted_period": submitted_period,
        "submitted": submitted,
        "err": err,
        "ref": ref,
        "starts": starts,
        "impulse_responses": impulse_responses,
        "truth_full": truth_full,
        "frequencies": frequencies,
        "stacks": stacks,
        "separation": separation,
        "crosscheck": crosscheck,
        "truth_metrics": truth_metrics,
        "naive_metrics": naive_metrics,
    }


@pytest.fixture(scope="module")
def async_truth_matrix() -> dict[int, dict[str, Any]]:
    return {ppm: _synthesise_case(ppm) for ppm in DRIFT_PPM}


def test_submitted_probe_is_exact_low_level_int16(
    async_truth_matrix: dict[int, dict[str, Any]],
) -> None:
    case = async_truth_matrix[364]
    submitted = case["submitted_period"]
    expected = np.rint(
        np.clip(case["ideal_period"], -1.0, 1.0)
        * np.float32(np.iinfo(np.int16).max)
    ).astype(np.int16)

    assert submitted.dtype == np.int16
    assert int(np.max(np.abs(submitted.astype(np.int32)))) == 98
    np.testing.assert_array_equal(submitted, expected)
    assert np.any(
        submitted.astype(np.float64) / float(np.iinfo(np.int16).max)
        != case["ideal_period"].astype(np.float64)
    )


def test_truth_paths_have_different_gain_shape_and_secondary_notch(
    async_truth_matrix: dict[int, dict[str, Any]],
) -> None:
    case = async_truth_matrix[364]
    primary = case["impulse_responses"]["noise"]
    secondary = case["impulse_responses"]["cancel"]
    assert not np.array_equal(primary, secondary)
    assert float(np.max(np.abs(primary))) != pytest.approx(
        float(np.max(np.abs(secondary)))
    )

    secondary_transfer = case["truth_full"]["cancel"]
    grid = np.fft.rfftfreq(case["probe"].period_samples, 1.0 / FS)

    def magnitude_at(frequency: float) -> float:
        return float(np.abs(secondary_transfer[np.argmin(np.abs(grid - frequency))]))

    assert magnitude_at(1100.0) < 0.5 * magnitude_at(500.0)


@pytest.mark.parametrize("ppm", DRIFT_PPM)
def test_fractional_joint_ls_recovers_truth_in_overall_and_four_subbands(
    async_truth_matrix: dict[int, dict[str, Any]], ppm: int
) -> None:
    case = async_truth_matrix[ppm]
    observed_ppm = float(case["separation"]["drift_ppm"])
    assert observed_ppm == pytest.approx(float(ppm), abs=3.0)
    assert int(np.asarray(case["separation"]["valid"]).sum()) >= 8

    for role in ("noise", "cancel"):
        rows = case["truth_metrics"][role]
        assert tuple(row["band_hz"] for row in rows) == TRUTH_BANDS_HZ
        for row in rows:
            assert row["agreement"] >= MIN_TRUTH_AGREEMENT, (ppm, role, row)
            assert row["relative_error"] <= MAX_TRUTH_RELATIVE_ERROR, (
                ppm,
                role,
                row,
            )

        crosscheck_rows = [
            case["crosscheck"][role]["overall"],
            *case["crosscheck"][role]["subbands"],
        ]
        assert len(crosscheck_rows) == 5
        assert all(row["passed"] for row in crosscheck_rows)


@pytest.mark.parametrize("ppm", DRIFT_PPM)
def test_naive_nominal_grid_guard1_fails_high_band_truth(
    async_truth_matrix: dict[int, dict[str, Any]], ppm: int
) -> None:
    for role, (agreement, relative_error) in async_truth_matrix[ppm][
        "naive_metrics"
    ].items():
        assert (
            agreement < MIN_TRUTH_AGREEMENT
            or relative_error > MAX_TRUTH_RELATIVE_ERROR
        ), (ppm, role, agreement, relative_error)


def test_crosscheck_tamper_fails_closed(
    async_truth_matrix: dict[int, dict[str, Any]],
) -> None:
    case = async_truth_matrix[729]
    tampered = {
        role: np.asarray(values).copy()
        for role, values in case["separation"]["crosscheck_transfers"].items()
    }
    tampered["cancel"] *= 1.02

    with pytest.raises(ValueError, match="crosscheck 실패"):
        mpi.separation_crosscheck_metrics(
            frequencies=case["frequencies"],
            joint_stacks=case["stacks"],
            resampled_stacks=tampered,
            keep=case["separation"]["valid"],
        )


def test_biased_q_fails_physical_truth_gate(
    async_truth_matrix: dict[int, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    case = async_truth_matrix[729]
    original_observer = mpi.observe_period_clock_ratios

    def biased_observer(**kwargs):
        observed = dict(original_observer(**kwargs))
        valid = np.asarray(observed["valid"], dtype=bool)
        biased_q = np.asarray(observed["q"], dtype=np.float64).copy()
        biased_q[valid] *= 1.0 + 200.0e-6
        observed["q"] = biased_q
        median_q = float(np.median(biased_q[valid]))
        observed["drift_samples_per_period"] = (
            case["probe"].period_samples / median_q
            - case["probe"].period_samples
        )
        return observed

    monkeypatch.setattr(mpi, "observe_period_clock_ratios", biased_observer)
    frequencies, stacks, separation = mpi.fractional_joint_channel_stacks(
        err=case["err"],
        ref=case["ref"],
        output_pcm_int16=case["submitted"],
        probe=case["probe"],
        period_starts=case["starts"],
        fit_band_hz=FIT_BAND_HZ,
        max_drift_deviation_samples=2.0,
        min_valid_periods=8,
    )

    failed_rows = []
    valid = np.asarray(separation["valid"], dtype=bool)
    for role in ("noise", "cancel"):
        aligned, _, _ = align_repeats(
            frequencies[role],
            stacks[role][valid],
            sample_rate=FS,
            fit_band_hz=FIT_BAND_HZ,
            anchor=0,
        )
        estimate = np.mean(aligned, axis=0)
        truth = case["truth_full"][role][case["probe"].bins_for(role)]
        for low, high in TRUTH_BANDS_HZ:
            mask = (frequencies[role] >= low) & (frequencies[role] <= high)
            agreement, relative_error = _complex_metrics(
                estimate[mask], truth[mask]
            )
            if (
                agreement < MIN_TRUTH_AGREEMENT
                or relative_error > MAX_TRUTH_RELATIVE_ERROR
            ):
                failed_rows.append((role, low, high, agreement, relative_error))

    assert failed_rows, "200 ppm q bias가 모든 physical truth gate를 통과했습니다"
