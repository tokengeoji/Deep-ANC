from __future__ import annotations

import hashlib
import json

import numpy as np

from deep_anc.dsp.fullband_causal_aperiodic import build_aperiodic_plan, linear_fft_convolve
from scripts.data.analyse_fullband_causal_aperiodic import load_panel_raw_provenance, validate_candidate


def _candidate(delay: int, short: np.ndarray) -> dict[str, object]:
    taps = np.zeros(1024)
    taps[: len(short)] = short
    return {"integer_delay_samples": delay, "post_onset_fir": taps}


def _responses(pcm: np.ndarray, kernels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    for path, channel in (("primary", 0), ("secondary", 1)):
        y = linear_fft_convolve(pcm[:, channel] / 32767.0, kernels[path])
        result[path] = y[: len(pcm)]
    return result


def _evidence(noise: float = 0.0):
    return {path: {role: {"subband_snr_db": [80.0] * 7, "subband_coherence": [1.0] * 7, "input_only_noise_rms": noise} for role in ("fit_a", "fit_b", "holdout")} for path in ("primary", "secondary")}


def test_v2_duration_thermal_budget_and_independent_bursts() -> None:
    plan, pcm = build_aperiodic_plan()
    assert pcm.shape == (1_202_176, 2)
    assert plan["output"]["duration_seconds"] == 25.045333333333332
    assert plan["output"]["active_duration_seconds"] == 16.352
    assert plan["output"]["zero_guard_duration_seconds"] == 8.192
    assert plan["output"]["peak_pcm"] == 98
    assert len({plan["bursts"][role]["seed"] for role in plan["bursts"]}) == 3
    assert len({plan["bursts"][role]["frames"] for role in plan["bursts"]}) == 3
    for meta in plan["bursts"].values():
        assert 0.00140 <= meta["rms"] <= 0.00143
        assert meta["target_band_condition_number"] <= 1.10
        assert meta["dc_and_nyquist_canonical_identification"] is False


def test_linear_fft_padding_rejects_circular_wrap() -> None:
    x = np.ones(100)
    h = np.zeros(80)
    h[-1] = 1.0
    try:
        linear_fft_convolve(x, h, n_fft=128)
    except ValueError as exc:
        assert "n_fft" in str(exc)
    else:
        raise AssertionError("circular convolution 크기가 허용됐습니다")


def test_known_fir_passes_but_never_becomes_canonical() -> None:
    plan, pcm = build_aperiodic_plan()
    short_p = np.array([0.8, -0.12, 0.04])
    short_s = np.array([-0.5, 0.08, -0.02])
    candidates = {"primary": _candidate(173, short_p), "secondary": _candidate(311, short_s)}
    kernels = {path: np.concatenate((np.zeros(int(c["integer_delay_samples"])), np.asarray(c["post_onset_fir"]))) for path, c in candidates.items()}
    receipt = validate_candidate(plan=plan, submitted_pcm=pcm, responses=_responses(pcm, kernels), candidates=candidates, response_evidence=_evidence(), synthetic_fixture=True)
    assert receipt["status"] == "PASS"
    assert receipt["canonical_training_eligible"] is False


def test_fractional_phase_is_in_post_onset_taps_not_added_to_delay() -> None:
    plan, pcm = build_aperiodic_plan()
    n = np.arange(97, dtype=np.float64)
    fractional = np.sinc(n - 32.35) * np.hanning(97)
    fractional /= np.sum(fractional)
    candidates = {
        "primary": _candidate(173, fractional),
        "secondary": _candidate(311, -0.7 * fractional),
    }
    kernels = {
        path: np.concatenate(
            (np.zeros(int(candidate["integer_delay_samples"])), np.asarray(candidate["post_onset_fir"]))
        )
        for path, candidate in candidates.items()
    }
    receipt = validate_candidate(
        plan=plan,
        submitted_pcm=pcm,
        responses=_responses(pcm, kernels),
        candidates=candidates,
        response_evidence=_evidence(),
        synthetic_fixture=True,
    )
    assert receipt["status"] == "PASS"
    assert all(
        row["artifact_semantics"] == "integer_delay_plus_post_onset_fir_only"
        for row in receipt["paths"].values()
    )


def test_delayed_echo_beyond_16k_and_double_delay_are_blocked() -> None:
    plan, pcm = build_aperiodic_plan()
    base = _candidate(173, np.array([0.8, -0.1]))
    candidates = {"primary": base, "secondary": _candidate(311, np.array([-0.5, 0.05]))}
    kernels = {}
    for path, candidate in candidates.items():
        kernel = np.concatenate((np.zeros(int(candidate["integer_delay_samples"])), np.asarray(candidate["post_onset_fir"])))
        if path == "primary":
            kernel = np.pad(kernel, (0, 20_500 - len(kernel)))
            kernel[20_000] = 0.2
        kernels[path] = kernel
    delayed = validate_candidate(plan=plan, submitted_pcm=pcm, responses=_responses(pcm, kernels), candidates=candidates, response_evidence=_evidence(), synthetic_fixture=True)
    assert delayed["status"] == "BLOCKED"
    double = {"primary": _candidate(173, np.concatenate((np.zeros(173), [0.8, -0.1]))), "secondary": candidates["secondary"]}
    doubled = validate_candidate(plan=plan, submitted_pcm=pcm, responses=_responses(pcm, {"primary": np.concatenate((np.zeros(173), [0.8, -0.1])), "secondary": np.concatenate((np.zeros(311), [-0.5, 0.05]))}), candidates=double, response_evidence=_evidence(), synthetic_fixture=True)
    assert doubled["status"] == "BLOCKED"


def test_panel_loader_reads_actual_bytes_and_rejects_fake_sha(tmp_path) -> None:
    identity = {"capture_id": "c1", "hardware_identity": "h1", "geometry_id": "g1", "level_evidence_sha256": "l1"}
    raw = tmp_path / "raw.json"; analysis = tmp_path / "analysis.json"; hardware = tmp_path / "hw.yml"; level = tmp_path / "level.json"
    raw.write_text(json.dumps(identity)); analysis.write_text(json.dumps({**identity, "minimum_repeat_consistency": 0.99})); hardware.write_text("audio: ok\n"); level.write_text("{}")
    def item(path): return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt = tmp_path / "receipt.json"
    payload = {"schema": "panel_raw_same_geometry_binding_v1", "raw": item(raw), "analysis": item(analysis), "hardware": item(hardware), "level": item(level)}
    receipt.write_text(json.dumps(payload))
    assert load_panel_raw_provenance(receipt)["raw"]["capture_id"] == "c1"
    payload["raw"]["sha256"] = "0" * 64; receipt.write_text(json.dumps(payload))
    try:
        load_panel_raw_provenance(receipt)
    except ValueError as exc:
        assert "SHA" in str(exc)
    else:
        raise AssertionError("fake panel SHA가 통과했습니다")
