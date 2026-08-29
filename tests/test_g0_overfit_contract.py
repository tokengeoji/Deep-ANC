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

    assert args.config == "configs/train_pretrain_tiny.yaml"
    assert args.lead_samples is None
    assert args.require_nmse_db == -6.0
    assert args.loss_alpha is None
    assert args.loss_lambda_dnh is None


def test_g0_diagnostic_escapes_are_explicit():
    args = _module().build_parser().parse_args(
        ["--lead-samples", "117", "--no-require-nmse"]
    )

    assert args.lead_samples == 117
    assert args.require_nmse_db is None


def test_g0_can_ablate_one_auxiliary_loss_without_claiming_nmse_only():
    module = _module()
    args = module.build_parser().parse_args(
        ["--disable-loss-term", "mrstft", "--disable-loss-term", "frame"]
    )
    overrides = module.build_diagnostic_overrides(args, "/tmp/diagnostic")

    assert args.nmse_only is False
    assert "loss.lambda_mrstft=0.0" in overrides
    assert "loss.lambda_frame=0.0" in overrides


def test_g0_alpha_specific_loss_identity_is_materialized_for_pre_pilot_receipt():
    module = _module()
    args = module.build_parser().parse_args(
        ["--loss-alpha", "1.0", "--loss-lambda-dnh", "0.000375"]
    )
    overrides = module.build_diagnostic_overrides(args, "/tmp/diagnostic")
    assert "loss.nmse_cvar_alpha=1.0" in overrides
    assert "loss.lambda_dnh=0.000375" in overrides


def test_g0_contract_and_fixed_batch_hashes_are_stable_and_content_addressed():
    module = _module()
    first = {"x": torch.tensor([[1.0, 2.0]]), "d": torch.tensor([[3.0, 4.0]])}
    reordered = {"d": torch.tensor([[3.0, 4.0]]), "x": torch.tensor([[1.0, 2.0]])}
    changed = {"x": torch.tensor([[1.0, 2.0]]), "d": torch.tensor([[3.0, 4.1]])}

    assert module.fixed_batch_sha256(first) == module.fixed_batch_sha256(reordered)
    assert module.fixed_batch_sha256(first) != module.fixed_batch_sha256(changed)
    assert module.resolved_config_sha256({"seed": 1, "loss": {"a": 2}}) == (
        module.resolved_config_sha256({"loss": {"a": 2}, "seed": 1})
    )


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


def test_finite_parameter_checks_sync_once_per_device_on_success(monkeypatch):
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.Linear(8, 2),
    )
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    original_item = torch.Tensor.item
    item_devices = []

    def counted_item(value, *args, **kwargs):
        item_devices.append(value.device)
        return original_item(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)

    validate_finite_gradients(model)
    assert item_devices == [torch.device("cpu")]

    item_devices.clear()
    validate_finite_parameters(model)
    assert item_devices == [torch.device("cpu")]


def test_finite_gradient_check_ignores_none_and_rejects_empty_tensor():
    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = None
    validate_finite_gradients(model)

    empty_model = torch.nn.Module()
    empty_model.register_parameter("empty", torch.nn.Parameter(torch.empty(0)))
    empty_model.empty.grad = torch.empty(0)
    with pytest.raises(FloatingPointError, match=r"gradient\.empty"):
        validate_finite_gradients(empty_model)
    with pytest.raises(FloatingPointError, match=r"parameter\.empty"):
        validate_finite_parameters(empty_model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device가 없습니다")
def test_finite_parameter_checks_sync_once_for_each_mixed_device(monkeypatch):
    model = torch.nn.Module()
    model.register_parameter("cpu_weight", torch.nn.Parameter(torch.ones(2)))
    model.register_parameter(
        "cuda_weight", torch.nn.Parameter(torch.ones(2, device="cuda"))
    )
    model.cpu_weight.grad = torch.ones_like(model.cpu_weight)
    model.cuda_weight.grad = torch.ones_like(model.cuda_weight)

    original_item = torch.Tensor.item
    item_devices = []

    def counted_item(value, *args, **kwargs):
        item_devices.append(value.device)
        return original_item(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)

    validate_finite_gradients(model)
    expected_devices = [model.cpu_weight.device, model.cuda_weight.device]
    assert item_devices == expected_devices

    item_devices.clear()
    validate_finite_parameters(model)
    assert item_devices == expected_devices
