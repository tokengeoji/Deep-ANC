from __future__ import annotations

import numpy as np

from deep_anc.data.recorded_broadband_coverage import (
    measure_broadband_session,
    summarize_broadband_sessions,
)
from deep_anc.dsp.control_band_contract import ControlBandContract


def test_broadband_session_requires_both_target_energy_and_source_coherence():
    contract = ControlBandContract.broadband_point_control()
    rng = np.random.default_rng(20260828)
    target = rng.normal(0.0, 0.01, 65_536)
    unrelated = rng.normal(0.0, 0.01, 65_536)

    related_metrics = measure_broadband_session(
        target,
        target,
        sample_rate=48_000,
        subbands_hz=contract.point_control_subbands_hz,
        nperseg=4096,
        noverlap=2048,
    )
    unrelated_metrics = measure_broadband_session(
        unrelated,
        target,
        sample_rate=48_000,
        subbands_hz=contract.point_control_subbands_hz,
        nperseg=4096,
        noverlap=2048,
    )

    assert all(related_metrics["target_energy_density_pass"])
    assert all(related_metrics["coherence_pass"])
    assert all(related_metrics["joint_pass"])
    assert all(unrelated_metrics["target_energy_density_pass"])
    assert not any(unrelated_metrics["coherence_pass"])
    assert not any(unrelated_metrics["joint_pass"])


def test_broadband_summary_does_not_hide_one_failed_family_or_subband():
    contract = ControlBandContract.broadband_point_control()
    n = len(contract.point_control_subbands_hz)

    def row(family: str, group: str, passed: tuple[bool, ...]):
        return {
            "session_id": group,
            "source_family": family,
            "split": "test",
            "group_id": group,
            "coherence": [0.9 if value else 0.1 for value in passed],
            "target_energy_density_ratio": [1.0] * n,
            "coherence_pass": list(passed),
            "target_energy_density_pass": [True] * n,
            "joint_pass": list(passed),
        }

    all_pass = (True,) * n
    speech_last_band_fail = (*((True,) * (n - 1)), False)
    rows = [
        row("music", "music-1", all_pass),
        row("speech", "speech-1", speech_last_band_fail),
    ]

    summary = summarize_broadband_sessions(
        rows, subbands_hz=contract.point_control_subbands_hz
    )

    assert summary["all"]["subbands"][-1]["joint_pass_fraction"] == 0.5
    assert summary["by_family"]["music"]["subbands"][-1]["joint_pass_fraction"] == 1.0
    assert summary["by_family"]["speech"]["subbands"][-1]["joint_pass_fraction"] == 0.0
    assert (
        summary["by_split_family"]["test"]["speech"]["subbands"][-1][
            "joint_pass_independent_groups"
        ]
        == 0
    )
