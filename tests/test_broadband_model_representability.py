from __future__ import annotations

import copy

import numpy as np
import torch
import yaml

from deep_anc.config import REPO_ROOT
from deep_anc.dsp.control_band_contract import (
    BROADBAND_GLOBAL_CLOCK_MAX_RESIDUAL_SAMPLES,
)
from deep_anc.models.broadband_representability import (
    BROADBAND_G0_GATE_SCHEMA,
    BROADBAND_UPPER_HZ,
    broadband_g0_gate_spec,
    checkpoint_polyphase_report,
    output_lattice_contract,
    tone_limiter_feasibility_report,
)


def _tiny_cfg() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "configs/model_tiny.yaml").read_text(encoding="utf-8")
    )


def _identity_synthesis_state(hop: int = 4) -> dict[str, torch.Tensor]:
    # 실제 모델처럼 head_channels=2*core를 쓰되 첫 hop 채널로 W0=I,
    # W1=W2=0을 만들면 모든 sample-frequency의 polyphase target을 exact 생성한다.
    head = torch.zeros(2 * hop, hop, 1)
    head[:hop, :, 0] = torch.eye(hop)
    decoder = torch.zeros(2 * hop, 1, hop * 3)
    for phase in range(hop):
        decoder[phase, 0, phase] = 1.0
    return {
        "head.weight": head,
        "decoder.weight": decoder,
        "head_act.weight": torch.tensor([0.25]),
    }


def _small_cfg(hop: int = 4) -> dict:
    cfg = copy.deepcopy(_tiny_cfg())
    cfg["hop"] = hop
    cfg["win"] = 3 * hop
    cfg["encoder"]["channels"] = hop
    return cfg


def test_hop_is_polyphase_not_output_sample_rate_reduction() -> None:
    receipt = output_lattice_contract(_tiny_cfg())
    assert receipt["structural_passed"] is True
    assert receipt["frames_per_runtime_block"] == 2
    assert receipt["decoder_polyphase_phases"] == 128
    assert receipt["nyquist_hz"] == 24_000.0
    assert receipt["sample_rate_covers_broadband"] is True
    assert receipt["broadband_upper_hz"] == BROADBAND_UPPER_HZ
    # 기존 hop-boundary causality test가 감추는 sample-level dependency를 명시한다.
    assert receipt["maximum_intra_hop_future_dependency_samples"] == 127
    assert receipt["sample_causal_without_runtime_handoff"] is False
    assert receipt["runtime_handoff_makes_dependency_implementable"] is True
    assert receipt["glstm_present"] is True
    assert receipt["past_receptive_field"] == "unbounded_state"


def test_identity_polyphase_generates_low_and_high_tones_exactly() -> None:
    cfg = _small_cfg()
    report = checkpoint_polyphase_report(
        _identity_synthesis_state(),
        cfg,
        frequencies_hz=(100.0, 2_000.0, 8_000.0, BROADBAND_UPPER_HZ),
    )
    assert report["minimum_algebraic_rank"] == 4
    assert report["algebraic_probe_passed"] is True
    assert report["fp16_weight_probe_passed"] is True
    assert report["bf16_weight_probe_passed"] is True
    assert report["maximum_fp16_weight_steering_relative_error"] < 1.0e-12


def test_rank_deficient_output_head_is_blocked() -> None:
    cfg = _small_cfg()
    state = _identity_synthesis_state()
    state["head.weight"][:, 3] = state["head.weight"][:, 2]
    report = checkpoint_polyphase_report(
        state, cfg, frequencies_hz=(2_000.0, 8_000.0)
    )
    assert report["minimum_algebraic_rank"] < 4
    assert report["algebraic_probe_passed"] is False
    assert report["canonical_training_admitted"] is False


def test_tone_limiter_margin_is_fail_closed() -> None:
    primary = np.asarray([1.0 + 0j, 2.0 + 0j])
    secondary = np.asarray([1.0 + 0j, 0.2 + 0j])
    passed = tone_limiter_feasibility_report(
        primary, secondary, source_peak=0.01, limiter_limit=0.2
    )
    failed = tone_limiter_feasibility_report(
        primary, secondary, source_peak=0.03, limiter_limit=0.2
    )
    assert passed["passed"] is True
    assert failed["passed"] is False
    assert passed["canonical_training_admitted"] is False


def test_broadband_g0_requires_all_phases_prefix_precision_and_families() -> None:
    spec = broadband_g0_gate_spec()
    assert spec["schema_version"] == BROADBAND_G0_GATE_SCHEMA
    assert spec["required_source_families"] == [
        "speech",
        "music",
        "environment",
        "machine",
    ]
    assert spec["sample_phase_coverage"]["impulse_residues_mod_hop"] == list(
        range(128)
    )
    assert spec["sample_phase_coverage"]["simultaneous_low_high_required"] is True
    assert spec["artifact_requirements"]["exact_operator_prefix_or_state"] is True
    assert (
        spec["boundary_contract"][
            "random_recorded_crop_requires_real_prefix_or_serialized_state"
        ]
        is True
    )
    assert spec["oracle_feasibility"]["maximum_abs_control"] == 0.18
    assert spec["precision_and_runtime_gate"][
        "upper_band_timing_residual_samples_at_most"
    ] == BROADBAND_GLOBAL_CLOCK_MAX_RESIDUAL_SAMPLES
    assert len(spec["sha256"]) == 64
