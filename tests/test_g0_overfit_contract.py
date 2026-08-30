"""G0 고정-batch 진단의 fail-closed CLI 계약."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from deep_anc.config import load_train_config

from deep_anc.train.trainer import (
    validate_finite_batch,
    validate_finite_gradients,
    validate_finite_loss_metrics,
    validate_finite_output,
    validate_finite_parameters,
    validate_g0_nmse,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bench/diagnose_training_overfit.py"


def _module():
    spec = importlib.util.spec_from_file_location("_g0_overfit_contract", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_g0_uses_config_lead_and_enforces_minus_six_db_by_default():
    args = _module().build_parser().parse_args([])

    assert args.lead_samples is None
    assert args.require_nmse_db == -6.0


def test_g0_diagnostic_escapes_are_explicit():
    args = _module().build_parser().parse_args(
        ["--lead-samples", "117", "--no-require-nmse"]
    )

    assert args.lead_samples == 117
    assert args.require_nmse_db is None


def test_g0_canonical_input_is_resolved_as_noneligible_diagnostic(tmp_path):
    module = _module()
    args = module.build_parser().parse_args(
        ["--config", "configs/train_pretrain_tiny.yaml", "--steps", "3"]
    )
    overrides = module.build_diagnostic_overrides(args, str(tmp_path / "run"))
    cfg = load_train_config(args.config, overrides)

    assert cfg["experiment_role"] == "diagnostic_overfit"
    assert cfg["init_eligible"] is False
    assert cfg["contract_run_dir"] is False
    assert cfg["schedule"]["total_steps"] == 3
    assert "experiment_contract_sha256" not in cfg


def test_g0_minus_six_db_boundary_is_a_failure():
    with pytest.raises(ValueError, match=">= 엄격 합격선"):
        validate_g0_nmse({"nmse_trusted_db": -6.0})
    assert validate_g0_nmse({"nmse_trusted_db": -6.000001}) < -6.0


@pytest.mark.parametrize("location", ["input", "output", "loss", "nmse", "gradient", "parameter"])
def test_g0_every_numeric_boundary_fails_closed_on_nonfinite(location):
    model = torch.nn.Linear(1, 1)
    if location == "input":
        with pytest.raises(FloatingPointError, match="input.x"):
            validate_finite_batch({"x": torch.tensor([float("nan")])})
    elif location == "output":
        with pytest.raises(FloatingPointError, match="model.output"):
            validate_finite_output(torch.tensor([float("inf")]))
    elif location == "loss":
        with pytest.raises(FloatingPointError, match="loss"):
            validate_finite_loss_metrics(
                torch.tensor(float("nan")), {"nmse_trusted_db": -7.0}
            )
    elif location == "nmse":
        with pytest.raises(FloatingPointError, match="metric.nmse_trusted_db"):
            validate_finite_loss_metrics(
                torch.tensor(0.0), {"nmse_trusted_db": float("nan")}
            )
    elif location == "gradient":
        model.weight.grad = torch.full_like(model.weight, float("inf"))
        with pytest.raises(FloatingPointError, match="gradient.weight"):
            validate_finite_gradients(model)
    else:
        with torch.no_grad():
            model.bias.fill_(float("nan"))
        with pytest.raises(FloatingPointError, match="parameter.bias"):
            validate_finite_parameters(model)
