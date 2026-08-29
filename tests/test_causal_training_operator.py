"""광대역 continuous-prefix causal P/S 연산자 회귀."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.signal import fftconvolve

from deep_anc.dsp.causal_training_operator import (
    CausalTrainingAuthorityUnavailable,
    load_causal_training_authority,
    load_joint_causal_operator_npz,
    operator_npz_internal_sha256,
)
from deep_anc.dsp.control_band_contract import ControlBandContract
from deep_anc.dsp import fullband_causal_v4 as v4
from deep_anc.dsp.fullband_causal_v4 import OPERATOR_NPZ_SCHEMA
from deep_anc.train.checkpoint import _validate_training_state_preview

from deep_anc.losses.broadband_loss import (
    BROADBAND_CAUSAL_CONVOLUTION_SCHEMA,
    BROADBAND_CAUSAL_INTERPOLATION_SCHEMA,
    BROADBAND_CAUSAL_PATH_SCHEMA,
    BroadbandANCLoss,
    CausalFIRPath,
    CausalFIRPathData,
)


FS = 48_000


def _path(
    *,
    role: str = "secondary",
    fir: np.ndarray | None = None,
    delay: int = 3,
    handoff: int = 0,
    fractional: float = 0.25,
) -> CausalFIRPath:
    taps = np.ascontiguousarray(
        np.asarray([0.25, -0.5, 0.75] if fir is None else fir, dtype="<f8")
    )
    digest = hashlib.sha256(taps.tobytes(order="C")).hexdigest()
    return CausalFIRPath(
        CausalFIRPathData(
            role=role,
            post_onset_fir=taps,
            coarse_delay_samples=delay,
            fractional_delay_samples=fractional,
            support_samples=taps.size,
            sample_rate=FS,
            handoff_extra_samples=handoff,
            operator_file_sha256="a" * 64,
            operator_internal_sha256="b" * 64,
            fir_sha256=digest,
            authority_sha256="c" * 64,
            source_path="fixture-only-not-canonical",
        )
    )


def _loss_cfg() -> dict[str, object]:
    return {
        "plant_representation_schema": BROADBAND_CAUSAL_PATH_SCHEMA,
        "interpolation_schema": BROADBAND_CAUSAL_INTERPOLATION_SCHEMA,
        "linear_spectral_schema": BROADBAND_CAUSAL_CONVOLUTION_SCHEMA,
        "lambda_mrstft": 0.0,
        "lambda_frame": 0.0,
        "lambda_dnh": 0.01,
        "lambda_pow": 0.0,
        "lambda_sat": 0.0,
        "nmse_cvar_alpha": 0.0,
        "mrstft_ffts": [256],
    }


def _broadband_target(samples: int, batch: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260828)
    values = torch.randn(samples, generator=generator)
    return values.view(1, 1, -1).repeat(batch, 1, 1) * 0.01


def test_numpy_and_torch_paths_equal_full_linear_causal_convolution() -> None:
    path = _path()
    source = np.random.default_rng(11).standard_normal(2048).astype(np.float32)
    expected = fftconvolve(source.astype(np.float64), path.data.post_onset_fir)[
        : source.size
    ]
    expected = np.pad(expected, (path.base_delay, 0))[: source.size]

    actual_numpy = path.filter_numpy(source)
    actual_torch = path(torch.from_numpy(source).view(1, 1, -1)).numpy()[0, 0]
    assert np.allclose(actual_numpy, expected, rtol=2e-6, atol=2e-6)
    assert np.allclose(actual_torch, expected, rtol=2e-6, atol=2e-6)


def test_handoff_is_applied_once_and_never_to_primary() -> None:
    secondary = _path(fir=np.asarray([1.0]), delay=2, handoff=256, fractional=0.0)
    impulse = torch.zeros(1, 1, 512)
    impulse[..., 0] = 1.0
    response = secondary(impulse)
    assert int(response[0, 0].abs().argmax()) == 258
    assert secondary.history_samples == 259

    with pytest.raises(ValueError, match="secondary"):
        _path(
            role="primary",
            fir=np.asarray([1.0]),
            delay=2,
            handoff=256,
            fractional=0.0,
        )


def test_causal_loss_requires_real_prefix_and_crops_only_valid_target() -> None:
    plant = _path(fir=np.asarray([1.0]), delay=0, fractional=0.0)
    criterion = BroadbandANCLoss(plant, _loss_cfg(), FS).eval()
    prefix = 256
    d = _broadband_target(prefix + 4096)
    y = (-d).detach().requires_grad_(True)
    loss, metrics = criterion(y, d, loss_start_sample=prefix)
    loss.backward()
    assert math.isfinite(float(loss))
    assert metrics["causal_prefix_samples"] == prefix
    assert metrics["nmse_subband_worst_db"] < -60.0
    assert y.grad is not None and torch.isfinite(y.grad).all()

    delayed = _path(fir=np.ones(8), delay=16, fractional=0.0)
    delayed_loss = BroadbandANCLoss(delayed, _loss_cfg(), FS).eval()
    with pytest.raises(ValueError, match="연속 prefix가 부족"):
        delayed_loss(y.detach(), d, loss_start_sample=delayed.history_samples - 1)


def test_fractional_delay_is_metadata_bound_but_not_double_applied() -> None:
    path = _path(
        fir=np.asarray([0.0, 1.0]), delay=7, handoff=0, fractional=-0.375
    )
    impulse = torch.zeros(1, 1, 64)
    impulse[..., 0] = 1.0
    # fractional residual은 이미 FIR 위상/탭 안에 있다. forward가 추가 fractional
    # shifter를 적용했다면 이 exact integer 위치가 바뀐다.
    assert int(path(impulse)[0, 0].abs().argmax()) == 8


def test_fir_byte_sha_and_fractional_encoding_fail_closed() -> None:
    taps = np.asarray([1.0], dtype="<f8")
    common = dict(
        role="secondary",
        post_onset_fir=taps,
        coarse_delay_samples=0,
        fractional_delay_samples=0.0,
        support_samples=1,
        sample_rate=FS,
        handoff_extra_samples=0,
        operator_file_sha256="a" * 64,
        operator_internal_sha256="b" * 64,
        fir_sha256="0" * 64,
        authority_sha256="c" * 64,
        source_path="fixture",
    )
    with pytest.raises(ValueError, match="FIR bytes"):
        CausalFIRPathData(**common)
    common["fir_sha256"] = hashlib.sha256(taps.tobytes()).hexdigest()
    common["fractional_delay_encoded_in_post_onset_fir"] = False
    with pytest.raises(ValueError, match="분수 지연"):
        CausalFIRPathData(**common)


def _utf8(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()


def _joint_arrays() -> dict[str, np.ndarray]:
    return {
        "schema": _utf8(OPERATOR_NPZ_SCHEMA),
        "primary_post_onset_fir": np.asarray([0.5, -0.25], dtype="<f8"),
        "secondary_post_onset_fir": np.asarray([0.25, 0.75], dtype="<f8"),
        "primary_coarse_delay_samples": np.asarray(10, dtype="<i8"),
        "secondary_coarse_delay_samples": np.asarray(20, dtype="<i8"),
        "primary_fractional_delay_samples": np.asarray(0.125, dtype="<f8"),
        "secondary_fractional_delay_samples": np.asarray(-0.25, dtype="<f8"),
        "support_samples": np.asarray(2, dtype="<i8"),
        "sample_rate_hz": np.asarray(FS, dtype="<i8"),
        "source_submitted_pcm_sha256": _utf8("1" * 64),
        "source_raw_sha256": _utf8("2" * 64),
        "fit_freeze_sha256": _utf8("3" * 64),
    }


def test_joint_npz_is_the_single_exact_primary_secondary_byte_source(
    tmp_path: Path,
) -> None:
    arrays = _joint_arrays()
    path = tmp_path / "joint.npz"
    np.savez(path, **arrays)
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    internal_sha = operator_npz_internal_sha256(arrays)
    loaded = load_joint_causal_operator_npz(
        path,
        expected_file_sha256=file_sha,
        expected_internal_sha256=internal_sha,
    )
    assert np.array_equal(
        loaded.primary_post_onset_fir, arrays["primary_post_onset_fir"]
    )
    assert np.array_equal(
        loaded.secondary_post_onset_fir, arrays["secondary_post_onset_fir"]
    )
    assert loaded.path == path

    arrays["diagnostic_duplicate_primary"] = arrays[
        "primary_post_onset_fir"
    ].copy()
    polluted = tmp_path / "polluted.npz"
    np.savez(polluted, **arrays)
    with pytest.raises(ValueError):
        load_joint_causal_operator_npz(
            polluted,
            expected_file_sha256=hashlib.sha256(polluted.read_bytes()).hexdigest(),
            expected_internal_sha256="0" * 64,
        )


def test_frozen_causal_plant_resume_has_explicit_no_rng_schema() -> None:
    nonlinear = np.random.default_rng(17).bit_generator.state
    _validate_training_state_preview(
        {
            "schema_version": 2,
            "plant_rng_kind": "not_applicable_frozen_causal_fir",
            "plant_rng": None,
            "nonlinear_rng": nonlinear,
        }
    )
    with pytest.raises(ValueError, match="marker"):
        _validate_training_state_preview(
            {
                "schema_version": 2,
                "plant_rng_kind": "random_plant",
                "plant_rng": None,
                "nonlinear_rng": nonlinear,
            }
        )


def test_live_schema_string_cannot_bypass_missing_exact_clock_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "schema": v4.TRAINING_AUTHORITY_ENVELOPE_SCHEMA,
        "authority": v4.TRAINING_AUTHORITY_SCHEMA,
        "status": "PASS",
        "canonical_training_eligible": True,
        "synthetic_fixture": False,
        "control_band_contract_sha256": (
            ControlBandContract.broadband_point_control().digest()
        ),
        "sample_rate_hz": FS,
        "block_size": 256,
        "latency": "low",
        "handoff_extra_samples": 256,
        "capture_id": "fixture-must-never-open-training",
        "operator": {},
        "clock": {},
        "fit": {},
        "holdout": {},
        "stationarity": {},
        "provenance": {},
    }
    encoded_body = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["evidence_sha256"] = hashlib.sha256(encoded_body).hexdigest()
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    path = tmp_path / "authority.json"
    path.write_bytes(encoded)
    monkeypatch.setattr(v4, "LIVE_AUTHORITY", v4.TRAINING_AUTHORITY_SCHEMA)
    with pytest.raises(
        CausalTrainingAuthorityUnavailable,
        match="BLOCKED_MISSING_EXACT_CLOCK_CHANGE_POINT_VALIDATOR",
    ):
        load_causal_training_authority(
            path,
            expected_file_sha256=hashlib.sha256(encoded).hexdigest(),
            expected_evidence_sha256=payload["evidence_sha256"],
            require_live_authority=True,
        )
