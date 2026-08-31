"""A100 smoke runner가 loss-pilot 선택을 semantic target에 전달하는지 검사한다."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from deep_anc.train.a100_pretrain_smoke import A100_MIN_USABLE_MEMORY_BYTES


RUNNER = Path(__file__).resolve().parents[1] / "scripts/train/run_a100_pretrain_smoke.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("_a100_smoke_runner_probe", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_loss_identity_override_is_explicit_and_preserves_all_required_smoke_overrides():
    runner = _load_runner()
    base = runner._overrides(
        bootstrap_sha256="a" * 64,
        label="uninterrupted",
        run_until_step=500,
    )
    selected = runner._overrides(
        bootstrap_sha256="a" * 64,
        label="uninterrupted",
        run_until_step=500,
        loss_alpha=1.0,
        loss_lambda_dnh=0.000375,
    )

    assert "loss.nmse_cvar_alpha=1.0" not in base
    assert "loss.nmse_cvar_alpha=1.0" in selected
    assert "loss.lambda_dnh=0.000375" in selected
    assert set(base).issubset(set(selected))
    assert selected.count("loss.nmse_cvar_alpha=1.0") == 1
    assert selected.count("loss.lambda_dnh=0.000375") == 1


def test_runner_requires_selected_alpha_and_lambda_and_rejects_nonpositive_lambda():
    runner = _load_runner()
    parser = runner.build_parser()
    alpha = next(item for item in parser._actions if item.dest == "loss_alpha")
    lambda_dnh = next(
        item for item in parser._actions if item.dest == "loss_lambda_dnh"
    )
    assert alpha.required is True
    assert lambda_dnh.required is True
    import pytest

    with pytest.raises(ValueError, match="finite 양수"):
        runner._overrides(
            bootstrap_sha256="a" * 64,
            label="uninterrupted",
            run_until_step=500,
            loss_alpha=0.7,
            loss_lambda_dnh=0.0,
        )


def test_runner_only_admits_the_approved_loss_alpha_grid():
    runner = _load_runner()
    action = next(
        item for item in runner.build_parser()._actions if item.dest == "loss_alpha"
    )
    assert tuple(action.choices) == (0.7, 0.85, 1.0)


def test_secondary_seed_smoke_is_fresh_and_never_consumes_its_own_prerequisite():
    runner = _load_runner()
    overrides = runner._overrides(
        bootstrap_sha256="a" * 64,
        label="uninterrupted",
        run_until_step=500,
        loss_alpha=0.7,
        loss_lambda_dnh=0.00075,
        seed=20260903,
    )
    seed_action = next(
        item for item in runner.build_parser()._actions if item.dest == "seed"
    )

    assert tuple(seed_action.choices) == (20260803, 20260903)
    assert "seed=20260903" in overrides
    assert "second_seed_prerequisite=null" in overrides
    assert "second_seed_prerequisite_sha256=null" in overrides


def test_runner_uses_the_same_nominal_a100_usable_memory_floor_as_receipt_validator():
    runner = _load_runner()
    assert runner.A100_MIN_USABLE_MEMORY_BYTES == A100_MIN_USABLE_MEMORY_BYTES
    assert A100_MIN_USABLE_MEMORY_BYTES == 79 * 1024**3
