from __future__ import annotations

import torch
import yaml

from deep_anc.config import REPO_ROOT
from deep_anc.models.broadband_deterministic_g0 import (
    BROADBAND_DETERMINISTIC_G0_SCHEMA,
    build_deterministic_g0_fixture,
)


def _tiny_cfg() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "configs/model_tiny.yaml").read_text(encoding="utf-8")
    )


def test_fixture_uses_plantdelays_lead_and_all_sample_phases() -> None:
    model_input, primary, receipt = build_deterministic_g0_fixture(
        _tiny_cfg(),
        primary_delay_samples=1386,
        secondary_delay_samples=1245,
        handoff_samples=256,
    )
    assert receipt["schema_version"] == BROADBAND_DETERMINISTIC_G0_SCHEMA
    assert receipt["derived_lead_samples"] == 115
    assert receipt["removed_common_delay_samples"] == 1245
    assert receipt["normalised_delay_fixture"] == {
        "primary_delay_samples": 141,
        "secondary_delay_samples": 0,
        "handoff_samples": 256,
        "sample_rate": 48000,
    }
    assert receipt["impulse_residues_mod_hop"] == list(range(128))
    assert receipt["impulse_count"] == 128
    assert model_input.shape == primary.shape[:1] + (2, primary.shape[-1])
    assert torch.count_nonzero(primary[receipt["impulse_index_start"] :]) == 128


def test_fixture_contains_requested_low_high_and_seven_band_probes() -> None:
    _, _, receipt = build_deterministic_g0_fixture(
        _tiny_cfg(),
        primary_delay_samples=1386,
        secondary_delay_samples=1245,
        handoff_samples=256,
    )
    tones = receipt["tone_frequencies_hz"]
    for required in (100.0, 125.0, 250.0, 500.0, 1000.0, 1600.0, 2000.0, 4000.0, 8000.0):
        assert required in tones
    assert len(receipt["subband_probe_frequencies_hz"]) == 7
    assert max(tones) > 11_300.0
