"""Stage-1/광대역 criterion admission과 factory 회귀."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from deep_anc.config import REPO_ROOT, _finalize_training_metadata, load_yaml
from deep_anc.dsp.control_band_contract import (
    BroadbandPlantEvidence,
    ControlBandContract,
    max_timing_error_samples_for_attenuation,
)
from deep_anc.dsp.measurement_level import meter_receipt_path
from deep_anc.losses import ANCLoss, BroadbandANCLoss
from deep_anc.train import trainer as trainer_module
from deep_anc.train.campaign_evidence import _build_criterion
from deep_anc.train.criterion_factory import (
    BROADBAND_CRITERION_ROLE,
    STAGE1_CRITERION_ROLE,
    admit_criterion_config,
    bind_criterion_contract,
    build_criterion_from_config,
)
from deep_anc.train.experiment_contract import stamp_experiment_contract


FS = 48_000


def test_validation_source_keeps_stage1_and_requires_recorded_val_for_broadband() -> None:
    assert trainer_module.resolve_validation_source(
        criterion_role=STAGE1_CRITERION_ROLE,
        has_recorded_stream=True,
        recorded_ratio=0.7,
    ) == "synthetic_val"
    assert trainer_module.resolve_validation_source(
        criterion_role=BROADBAND_CRITERION_ROLE,
        has_recorded_stream=True,
        recorded_ratio=0.7,
    ) == "recorded_val_only"
    with pytest.raises(FileNotFoundError, match="recorded-val-only"):
        trainer_module.resolve_validation_source(
            criterion_role=BROADBAND_CRITERION_ROLE,
            has_recorded_stream=False,
            recorded_ratio=0.7,
        )


def _sha(character: str) -> str:
    return character * 64


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _primary_path(secondary_path: Path) -> Path:
    return secondary_path.with_name(f"{secondary_path.stem}.primary.npz")


def _valid_evidence(
    contract: ControlBandContract,
    **overrides: object,
) -> BroadbandPlantEvidence:
    n_bands = len(contract.point_control_subbands_hz)
    n_panels = len(contract.measurement_panels_hz)
    phase_limits = tuple(
        max_timing_error_samples_for_attenuation(
            contract.measurement_resolution_attenuation_db,
            band[1],
            contract.sample_rate,
        )
        for band in contract.point_control_subbands_hz
    )
    payload: dict[str, object] = {
        "control_band_contract_sha256": contract.digest(),
        "primary_capture_id": "toy-capture-v2",
        "secondary_capture_id": "toy-capture-v2",
        "primary_raw_sha256": _sha("a"),
        "secondary_raw_sha256": _sha("a"),
        "primary_analysis_sha256": _sha("b"),
        "secondary_analysis_sha256": _sha("b"),
        "measurement_level_evidence_sha256": _sha("c"),
        "exact_plan_file_sha256": _sha("d"),
        "exact_plan_payload_sha256": _sha("e"),
        "exact_plan_pcm_sha256": _sha("f"),
        "fresh_meter_raw_sha256": _sha("0"),
        "fresh_meter_receipt_sha256": _sha("1"),
        "timing_marker_pcm_sha256": _sha("2"),
        "fixed_clock_pilot_sha256": _sha("3"),
        "submitted_pilot_validation_sha256": _sha("7"),
        "submitted_pilot_cross_channel_null_sha256": _sha("8"),
        "submitted_pilot_cross_channel_max_absolute": 0.0,
        "submitted_pilot_cross_channel_max_ratio": 0.0,
        "global_clock_input_domain": (
            "actual_submitted_int16_period_spectrum_not_intended_float"
        ),
        "global_clock_map_sha256": _sha("4"),
        "global_clock_slope_samples_per_sample": 2.4835888604 / 6000.0,
        "global_clock_intercept_samples": 0.0,
        "global_clock_max_residual_samples": 0.05,
        "clock_trajectory_agreement_samples": 0.05,
        "transition_anchor_valid_counts": (8, 8, 8, 8),
        "callback_timing_valid": True,
        "callback_sample_slip_count": 0,
        "panel_clock_offsets_samples": (0.0,) * n_panels,
        "applied_per_drive_phase_repair_samples": (0.0,) * (2 * n_panels),
        "primary_marker_delay_samples": 500.0,
        "secondary_marker_delay_samples": 450.0,
        "primary_marker_branch_width_samples": 2000.0,
        "secondary_marker_branch_width_samples": 1450.0,
        "primary_marker_alias_candidate_count": 1,
        "secondary_marker_alias_candidate_count": 1,
        "primary_bulk_delay_fractional_samples": 400.0,
        "secondary_bulk_delay_fractional_samples": 300.0,
        "primary_bulk_delay_samples": 400,
        "secondary_bulk_delay_samples": 300,
        "primary_effective_delay_samples": 336,
        "secondary_effective_delay_samples": 236,
        "pre_roll_samples": 64,
        "handoff_extra_samples": 256,
        "derived_lead_samples": 156,
        "panel_primary_minus_secondary_bulk_delay_samples": (100.0,) * n_panels,
        "panel_relative_delay_deviation_samples": (0.0,) * n_panels,
        "sample_rate": FS,
        "block_size": 256,
        "latency": "low",
        "observed_submitted_pcm": True,
        "excitation_panels_hz": contract.measurement_panels_hz,
        "verified_subbands_hz": contract.point_control_subbands_hz,
        "primary_consistency": (0.951,) * n_bands,
        "secondary_consistency": (0.951,) * n_bands,
        "clock_valid_repeats": (8,) * n_panels,
        "clock_min_adjacent_score_observed": (0.995,) * n_panels,
        "relative_phase_jitter_samples": tuple(value * 0.99 for value in phase_limits),
        "separation_crosscheck_agreement": (0.999,) * n_bands,
        "separation_crosscheck_relative_error": (0.01,) * n_bands,
        "measured_interpolation_agreement": (0.995,) * n_bands,
        "measured_interpolation_relative_error": (0.10,) * n_bands,
        "primary_compact_role": "diagnostic_only",
        "secondary_compact_role": "diagnostic_only",
        "primary_compact_training_eligible": False,
        "secondary_compact_training_eligible": False,
        "primary_compact_identifiability_sha256": _sha("5"),
        "secondary_compact_identifiability_sha256": _sha("6"),
        "compact_roundtrip_agreement": (0.995,) * n_bands,
        "compact_roundtrip_relative_error": (0.10,) * n_bands,
        "xrun_count": 0,
        "clip_count": 0,
    }
    payload.update(overrides)
    return BroadbandPlantEvidence.model_validate(payload)


def _write_secondary(
    path: Path,
    *,
    broadband: bool,
    artifact_sha: str | None = None,
    excitation: tuple[float, float] | None = None,
    consistency: tuple[float, float] | None = None,
    verified: np.ndarray | None = None,
    evidence_overrides: dict[str, object] | None = None,
    npz_secondary_consistency: np.ndarray | None = None,
    npz_secondary_raw_sha256: str | None = None,
    npz_timing_overrides: dict[str, object] | None = None,
    npz_evidence_sha256: str | None = None,
) -> None:
    contract = ControlBandContract.broadband_point_control()
    default_band = tuple(contract.point_control_target_hz) if broadband else (150.0, 1600.0)
    payload: dict[str, object] = {
        "fir": np.asarray([1.0], dtype=np.float32),
        "delay_samples": np.asarray(0, dtype=np.int64),
        "sample_rate": np.asarray(FS, dtype=np.int64),
        "fit_improvement_db": np.asarray(100.0, dtype=np.float64),
        "coherence_median": np.asarray(1.0, dtype=np.float64),
        "excitation_band_hz": np.asarray(excitation or default_band, dtype=np.float64),
        "consistency_band_hz": np.asarray(consistency or default_band, dtype=np.float64),
    }
    if broadband:
        raw_path = path.with_suffix(".raw.npz")
        analysis_path = path.with_suffix(".analysis.npz")
        level_path = path.with_suffix(".level.json")
        plan_path = path.with_suffix(".plan.json")
        meter_path = path.with_suffix(".meter.npz")
        meter_receipt = meter_receipt_path(meter_path)
        raw_path.write_bytes(b"immutable broadband raw")
        analysis_path.write_bytes(b"immutable broadband analysis")
        level_path.write_bytes(b'{"level":"immutable"}\n')
        plan_payload = {"output": {"pcm_sha256": _sha("f")}}
        plan_path.write_text(
            json.dumps(plan_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        meter_path.write_bytes(b"immutable broadband fresh meter")
        meter_receipt.write_bytes(b'{"meter_receipt":"immutable"}\n')
        provenance = {
            "primary_raw_sha256": _file_sha256(raw_path),
            "secondary_raw_sha256": _file_sha256(raw_path),
            "primary_analysis_sha256": _file_sha256(analysis_path),
            "secondary_analysis_sha256": _file_sha256(analysis_path),
            "measurement_level_evidence_sha256": _file_sha256(level_path),
            "exact_plan_file_sha256": _file_sha256(plan_path),
            "exact_plan_payload_sha256": hashlib.sha256(
                json.dumps(
                    plan_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            "exact_plan_pcm_sha256": _sha("f"),
            "fresh_meter_raw_sha256": _file_sha256(meter_path),
            "fresh_meter_receipt_sha256": _file_sha256(meter_receipt),
        }
        provenance.update(evidence_overrides or {})
        evidence = _valid_evidence(contract, **provenance)
        evidence_json = json.dumps(
            evidence.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        payload["schema_version"] = np.asarray(
            "broadband_measured_band_plant_v2_raw_derived"
        )
        payload["plant_role"] = np.asarray("secondary")
        payload["delay_samples"] = np.asarray(
            evidence.secondary_effective_delay_samples, dtype=np.int64
        )
        for field in (
            "primary_bulk_delay_fractional_samples",
            "secondary_bulk_delay_fractional_samples",
            "primary_bulk_delay_samples",
            "secondary_bulk_delay_samples",
            "primary_effective_delay_samples",
            "secondary_effective_delay_samples",
            "pre_roll_samples",
            "handoff_extra_samples",
            "derived_lead_samples",
            "panel_primary_minus_secondary_bulk_delay_samples",
            "panel_relative_delay_deviation_samples",
        ):
            payload[field] = np.asarray(
                (npz_timing_overrides or {}).get(field, getattr(evidence, field))
            )
        payload["control_band_contract_sha256"] = np.asarray(
            artifact_sha or contract.digest()
        )
        payload["broadband_plant_evidence_json"] = np.asarray(evidence_json)
        payload["broadband_plant_evidence_sha256"] = np.asarray(
            npz_evidence_sha256
            or hashlib.sha256(evidence_json.encode("utf-8")).hexdigest()
        )
        frequencies = np.arange(104.0, 11_400.0 + 0.1, 16.0, dtype=np.float64)
        secondary_phase = np.exp(
            -2j * np.pi * frequencies * evidence.secondary_bulk_delay_fractional_samples / FS
        )
        primary_phase = np.exp(
            -2j * np.pi * frequencies * evidence.primary_bulk_delay_fractional_samples / FS
        )
        payload["measured_frequencies_hz"] = frequencies
        payload["measured_transfer_real"] = secondary_phase.real
        payload["measured_transfer_imag"] = secondary_phase.imag
        payload["aligned_mean_transfer_sha256"] = np.asarray(
            hashlib.sha256(
                np.asarray(secondary_phase, dtype=np.complex128).tobytes(order="C")
            ).hexdigest()
        )
        payload["bulk_delay_samples"] = np.asarray(
            evidence.secondary_bulk_delay_samples, dtype=np.int64
        )
        payload["bulk_delay_fractional_samples"] = np.asarray(
            evidence.secondary_bulk_delay_fractional_samples, dtype=np.float64
        )
        payload["effective_delay_samples"] = np.asarray(
            evidence.secondary_effective_delay_samples, dtype=np.int64
        )
        payload["fractional_effective_delay_samples"] = np.asarray(
            evidence.secondary_bulk_delay_fractional_samples
            - evidence.pre_roll_samples,
            dtype=np.float64,
        )
        payload["delay_semantics"] = np.asarray(
            "effective_zeros_before_compact_fir"
        )
        payload["compact_role"] = np.asarray("diagnostic_only")
        payload["compact_training_eligible"] = np.asarray(False)
        payload["compact_identifiability_sha256"] = np.asarray(
            evidence.secondary_compact_identifiability_sha256
        )
        payload["measured_interpolation_agreement"] = np.asarray(
            evidence.measured_interpolation_agreement, dtype=np.float64
        )
        payload["measured_interpolation_relative_error"] = np.asarray(
            evidence.measured_interpolation_relative_error, dtype=np.float64
        )
        payload["source_raw_npz_path"] = np.asarray(raw_path.name)
        payload["source_raw_npz_sha256"] = np.asarray(
            npz_secondary_raw_sha256 or evidence.secondary_raw_sha256
        )
        payload["source_analysis_npz_path"] = np.asarray(analysis_path.name)
        payload["source_analysis_npz_sha256"] = np.asarray(
            evidence.secondary_analysis_sha256
        )
        payload["measurement_level_evidence_path"] = np.asarray(level_path.name)
        payload["measurement_level_evidence_sha256"] = np.asarray(
            evidence.measurement_level_evidence_sha256
        )
        payload["source_plan_path"] = np.asarray(plan_path.name)
        payload["fresh_meter_raw_path"] = np.asarray(meter_path.name)
        for artifact_field, evidence_field in (
            ("source_plan_file_sha256", "exact_plan_file_sha256"),
            ("source_plan_payload_sha256", "exact_plan_payload_sha256"),
            ("source_plan_pcm_sha256", "exact_plan_pcm_sha256"),
            ("fresh_meter_raw_sha256", "fresh_meter_raw_sha256"),
            ("fresh_meter_receipt_sha256", "fresh_meter_receipt_sha256"),
        ):
            payload[artifact_field] = np.asarray(getattr(evidence, evidence_field))
        payload["verified_subbands_hz"] = np.asarray(
            contract.point_control_subbands_hz if verified is None else verified,
            dtype=np.float64,
        )
        payload["band_consistency"] = np.asarray(
            evidence.secondary_consistency
            if npz_secondary_consistency is None
            else npz_secondary_consistency,
            dtype=np.float64,
        )
        primary_payload = dict(payload)
        primary_payload["plant_role"] = np.asarray("primary")
        primary_payload["delay_samples"] = np.asarray(
            evidence.primary_effective_delay_samples, dtype=np.int64
        )
        primary_payload["effective_delay_samples"] = np.asarray(
            evidence.primary_effective_delay_samples, dtype=np.int64
        )
        primary_payload["bulk_delay_samples"] = np.asarray(
            evidence.primary_bulk_delay_samples, dtype=np.int64
        )
        primary_payload["bulk_delay_fractional_samples"] = np.asarray(
            evidence.primary_bulk_delay_fractional_samples, dtype=np.float64
        )
        primary_payload["fractional_effective_delay_samples"] = np.asarray(
            evidence.primary_bulk_delay_fractional_samples - evidence.pre_roll_samples,
            dtype=np.float64,
        )
        primary_payload["measured_transfer_real"] = primary_phase.real
        primary_payload["measured_transfer_imag"] = primary_phase.imag
        primary_payload["aligned_mean_transfer_sha256"] = np.asarray(
            hashlib.sha256(
                np.asarray(primary_phase, dtype=np.complex128).tobytes(order="C")
            ).hexdigest()
        )
        primary_payload["compact_identifiability_sha256"] = np.asarray(
            evidence.primary_compact_identifiability_sha256
        )
        primary_payload["band_consistency"] = np.asarray(
            evidence.primary_consistency, dtype=np.float64
        )
        np.savez(_primary_path(path), **primary_payload)
    np.savez(path, **payload)


def _stage1_loss() -> dict[str, object]:
    return {
        "nmse_objective": "trusted_band",
        "mrstft_ffts": [256],
        "lambda_mrstft": 0.0,
        "lambda_pow": 0.0,
        "lambda_dnh": 0.0,
        "lambda_frame": 0.0,
        "lambda_sat": 0.0,
        "band_weight": "trusted_only",
    }


def _cfg(path: Path, *, broadband: bool) -> dict[str, object]:
    contract = ControlBandContract.broadband_point_control()
    loss = (
        {
            "schema_version": "broadband_equal_subband_loss_v3",
            "mrstft_ffts": [256],
            "lambda_mrstft": 0.0,
            "lambda_pow": 0.0,
            "lambda_dnh": 0.01,
            "lambda_frame": 0.0,
            "lambda_sat": 0.0,
            "band_weight": "trusted_only",
        }
        if broadband
        else _stage1_loss()
    )
    cfg: dict[str, object] = {
        "seed": 17,
        "loss": loss,
        "data": {
            "sample_rate": FS,
            "digital_primary_path_mode": "measured" if broadband else "secondary_surrogate",
            "plant_perturbation": {
                "delay_jitter_range": [0, 0],
                "gain_db": [0.0, 0.0],
                "gain_tilt_db_per_octave": [0.0, 0.0],
                "allpass_perturb": False,
            },
            "nonlinear": {
                "sef_eta_choices": [10.0],
                "drive_range": [1.0, 1.0],
                "hardclip_prob": 0.0,
            },
        },
        "duct": {
            "secondary_path": {
                "npz": path.name,
                "handoff_extra_samples": 256,
            },
            "acoustics": {
                "plane_wave_cutoff_hz": 1633.0,
                "realistic_target_band_hz": [80.0, 1600.0],
            },
        },
    }
    if broadband:
        cfg["duct"]["digital_reference"] = {
            "primary_path_npz": _primary_path(path).name,
        }
        cfg["control_band_contract_sha256"] = contract.digest()
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                if "broadband_plant_evidence_json" in archive:
                    evidence_json = str(
                        np.asarray(archive["broadband_plant_evidence_json"]).item()
                    )
                    cfg["broadband_plant_evidence_sha256"] = hashlib.sha256(
                        evidence_json.encode("utf-8")
                    ).hexdigest()
    return cfg


def test_stage1_config_keeps_existing_ancloss_path_and_resolved_bytes(tmp_path: Path) -> None:
    plant = tmp_path / "stage1.npz"
    _write_secondary(plant, broadband=False)
    cfg = _cfg(plant, broadband=False)
    before = copy.deepcopy(cfg)

    admission = bind_criterion_contract(cfg, repo_root=tmp_path)
    bundle = build_criterion_from_config(
        cfg,
        repo_root=tmp_path,
        limiter_limit=0.2,
        device=torch.device("cpu"),
        admission=admission,
    )

    assert cfg == before
    assert admission.role == STAGE1_CRITERION_ROLE
    assert type(bundle.criterion) is ANCLoss
    assert admission.trusted_band_hz == (150.0, 1600.0)


def test_valid_broadband_artifact_selects_broadband_loss_and_binds_identity(
    tmp_path: Path,
) -> None:
    plant = tmp_path / "broadband.npz"
    _write_secondary(plant, broadband=True)
    cfg = _cfg(plant, broadband=True)

    admission = bind_criterion_contract(cfg, repo_root=tmp_path)
    contract = ControlBandContract.broadband_point_control()
    assert admission.role == BROADBAND_CRITERION_ROLE
    assert admission.primary_path == _primary_path(plant)
    assert admission.measured_band_contract_sha256 is not None
    assert cfg["criterion_role"] == BROADBAND_CRITERION_ROLE
    assert cfg["loss"]["schema_version"] == "broadband_equal_subband_loss_v3"
    assert cfg["broadband_measured_band_contract"]["segment_boundary_status"] == (
        "BLOCKED_MISSING_PREFIX_OR_STATE"
    )
    assert cfg["broadband_measured_band_contract"][
        "synthetic_primary_generator_status"
    ] == "BLOCKED_COMPACT_PRIMARY_GENERATOR"
    assert cfg["control_band_contract_sha256"] == contract.digest()
    assert cfg["data"]["digital_reference_lead_samples"] == 156
    assert admission.trusted_band_hz == contract.point_control_target_hz
    with np.load(plant, allow_pickle=False) as archive:
        evidence = json.loads(str(archive["broadband_plant_evidence_json"].item()))
    assert evidence["schema_version"] == "broadband_interleaved_plant_evidence_v4"
    with pytest.raises(ValueError, match="admission BLOCKED"):
        build_criterion_from_config(
            cfg,
            repo_root=tmp_path,
            limiter_limit=0.2,
            device=torch.device("cpu"),
            admission=admission,
        )


@pytest.mark.parametrize(
    ("dropout", "message"),
    (
        (None, "dropout 확률을 exact mapping"),
        (
            {"reference_probability": 0.1, "error_probability": 0.0},
            "x_ref dropout은 exact 0",
        ),
    ),
)
def test_causal_broadband_admission_binds_explicit_input_channel_policy(
    tmp_path: Path,
    dropout: dict[str, float] | None,
    message: str,
) -> None:
    plant = tmp_path / "unused-legacy.npz"
    _write_secondary(plant, broadband=True)
    cfg = _cfg(plant, broadband=True)
    cfg["data"]["digital_primary_path_mode"] = "causal_joint_v4"
    cfg["loss"].update(
        {
            "plant_representation_schema": "fullband_causal_joint_fir_operator_npz_v4",
            "interpolation_schema": "not_applicable_frozen_causal_fir_v1",
            "linear_spectral_schema": (
                "full_linear_causal_convolution_continuous_prefix_valid_crop_v1"
            ),
        }
    )
    if dropout is not None:
        cfg["data"]["broadband_channel_dropout"] = dropout
    cfg["broadband_causal_training_authority"] = {
        "schema": "fullband_causal_training_authority_config_v1",
        "path": "does-not-exist.json",
        "file_sha256": "1" * 64,
        "evidence_sha256": "2" * 64,
    }
    with pytest.raises(ValueError, match=message):
        admit_criterion_config(cfg, repo_root=tmp_path, require_bound=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("wrong_top_sha", "resolved config control-band contract SHA"),
        ("wrong_artifact_sha", "S NPZ control-band contract SHA"),
        ("short_excitation", "excitation band"),
        ("short_consistency", "consistency band"),
        ("wrong_subbands", "verified_subbands_hz"),
        ("forged_consistency", "consistency vector"),
        ("wrong_raw_sha", "raw SHA"),
        ("wrong_timing_metadata", "derived_lead_samples"),
        ("wrong_config_evidence_sha", "config plant evidence SHA"),
        ("wrong_npz_evidence_sha", "embedded plant evidence SHA"),
    ),
)
def test_broadband_admission_rejects_wrong_or_incomplete_plant(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    contract = ControlBandContract.broadband_point_control()
    plant = tmp_path / f"{mutation}.npz"
    kwargs: dict[str, object] = {"broadband": True}
    if mutation == "wrong_artifact_sha":
        kwargs["artifact_sha"] = "0" * 64
    elif mutation == "short_excitation":
        kwargs["excitation"] = (150.0, 1600.0)
    elif mutation == "short_consistency":
        kwargs["consistency"] = (150.0, 1600.0)
    elif mutation == "wrong_subbands":
        rows = np.asarray(contract.point_control_subbands_hz, dtype=np.float64)
        rows[-1, 1] = 8000.0
        kwargs["verified"] = rows
    elif mutation == "forged_consistency":
        kwargs["npz_secondary_consistency"] = np.asarray([0.999] * 7)
    elif mutation == "wrong_raw_sha":
        kwargs["npz_secondary_raw_sha256"] = _sha("d")
    elif mutation == "wrong_timing_metadata":
        kwargs["npz_timing_overrides"] = {"derived_lead_samples": 155}
    elif mutation == "wrong_npz_evidence_sha":
        kwargs["npz_evidence_sha256"] = _sha("e")
    _write_secondary(plant, **kwargs)
    cfg = _cfg(plant, broadband=True)
    if mutation == "wrong_top_sha":
        cfg["control_band_contract_sha256"] = "f" * 64
    elif mutation == "wrong_config_evidence_sha":
        cfg["broadband_plant_evidence_sha256"] = _sha("e")

    with pytest.raises(ValueError, match=message):
        bind_criterion_contract(cfg, repo_root=tmp_path)


def test_current_strict_v1_secondary_cannot_be_renamed_broadband() -> None:
    duct = load_yaml(REPO_ROOT / "configs/duct.yaml")
    cfg: dict[str, object] = {
        "loss": {
            "schema_version": "broadband_equal_subband_loss_v3",
            "lambda_dnh": 0.01,
            "band_weight": "trusted_only",
        },
        "data": {"sample_rate": FS, "digital_primary_path_mode": "measured"},
        "duct": duct,
        "control_band_contract_sha256": (
            ControlBandContract.broadband_point_control().digest()
        ),
        "broadband_plant_evidence_sha256": "0" * 64,
    }

    with pytest.raises(ValueError, match="control_band_contract_sha256 metadata"):
        bind_criterion_contract(cfg, repo_root=REPO_ROOT)


def test_broadband_rejects_legacy_or_secondary_surrogate_primary_generator(
    tmp_path: Path,
) -> None:
    plant = tmp_path / "compact-primary-bypass.npz"
    _write_secondary(plant, broadband=True)
    cfg = _cfg(plant, broadband=True)
    cfg["data"]["digital_primary_path_mode"] = "secondary_surrogate"
    with pytest.raises(ValueError, match="compact P 우회"):
        bind_criterion_contract(cfg, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("secondary_as_primary", "schema/role"),
        ("legacy_primary", "control_band_contract_sha256 metadata"),
        ("different_capture", "config plant evidence SHA"),
    ),
)
def test_broadband_admission_rejects_wrong_configured_primary(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    plant = tmp_path / "secondary.npz"
    _write_secondary(plant, broadband=True)
    cfg = _cfg(plant, broadband=True)
    if mutation == "secondary_as_primary":
        cfg["duct"]["digital_reference"]["primary_path_npz"] = plant.name
    elif mutation == "legacy_primary":
        _write_secondary(_primary_path(plant), broadband=False)
    else:
        other = tmp_path / "other-secondary.npz"
        _write_secondary(
            other,
            broadband=True,
            evidence_overrides={
                "primary_capture_id": "other-capture-v3",
                "secondary_capture_id": "other-capture-v3",
            },
        )
        cfg["duct"]["digital_reference"]["primary_path_npz"] = (
            _primary_path(other).name
        )

    with pytest.raises(ValueError, match=message):
        bind_criterion_contract(cfg, repo_root=tmp_path)


def test_invalid_broadband_plant_fails_before_cuda_or_model_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plant = tmp_path / "strict-renamed.npz"
    _write_secondary(plant, broadband=False)
    cfg = _cfg(plant, broadband=True)
    cfg["criterion_role"] = BROADBAND_CRITERION_ROLE
    cfg["broadband_plant_evidence_sha256"] = "0" * 64

    def forbidden(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("admission 실패 전에 CUDA/model을 건드렸습니다")

    monkeypatch.setattr(trainer_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(trainer_module.torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(trainer_module, "build_model", forbidden)
    with pytest.raises(FileNotFoundError, match="criterion primary path"):
        trainer_module.Trainer(cfg)


def test_broadband_admission_rejects_evidence_with_one_failed_subband(
    tmp_path: Path,
) -> None:
    plant = tmp_path / "failed-consistency.npz"
    _write_secondary(
        plant,
        broadband=True,
        evidence_overrides={
            "secondary_consistency": (0.951,) * 6 + (0.949,),
        },
    )
    cfg = _cfg(plant, broadband=True)

    with pytest.raises(ValueError, match="S consistency"):
        bind_criterion_contract(cfg, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("config_handoff", "evidence handoff"),
        ("config_lead", "digital-reference lead"),
    ),
)
def test_broadband_admission_rejects_timing_config_mismatch(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    plant = tmp_path / f"{mutation}.npz"
    _write_secondary(plant, broadband=True)
    cfg = _cfg(plant, broadband=True)
    if mutation == "config_handoff":
        cfg["duct"]["secondary_path"]["handoff_extra_samples"] = 257
    else:
        cfg["data"]["digital_reference_lead_samples"] = 155

    with pytest.raises(ValueError, match=message):
        bind_criterion_contract(cfg, repo_root=tmp_path)


def test_broadband_admission_rejects_tampered_external_evidence_bytes(
    tmp_path: Path,
) -> None:
    plant = tmp_path / "tampered.npz"
    _write_secondary(plant, broadband=True)
    cfg = _cfg(plant, broadband=True)
    analysis = plant.with_suffix(".analysis.npz")
    analysis.write_bytes(analysis.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="실제 파일 bytes"):
        bind_criterion_contract(cfg, repo_root=tmp_path)


@pytest.mark.parametrize("target", ("plan", "meter", "meter_receipt"))
def test_broadband_admission_rejects_missing_external_authority(
    tmp_path: Path,
    target: str,
) -> None:
    plant = tmp_path / f"missing-{target}.npz"
    _write_secondary(plant, broadband=True)
    cfg = _cfg(plant, broadband=True)
    paths = {
        "plan": plant.with_suffix(".plan.json"),
        "meter": plant.with_suffix(".meter.npz"),
        "meter_receipt": meter_receipt_path(plant.with_suffix(".meter.npz")),
    }
    paths[target].unlink()

    with pytest.raises(FileNotFoundError, match="broadband"):
        bind_criterion_contract(cfg, repo_root=tmp_path)


def test_finalize_and_campaign_factory_share_broadband_resolved_contract(
    tmp_path: Path,
) -> None:
    plant = tmp_path / "broadband.npz"
    _write_secondary(plant, broadband=True)
    cfg = _cfg(plant, broadband=True)

    # load_train_config의 experiment stamp 직전 경계와 같은 materialization.
    _finalize_training_metadata(cfg, repo_root=tmp_path)
    assert cfg["criterion_role"] == BROADBAND_CRITERION_ROLE
    assert cfg["best_metric_key"] == "nmse_subband_guard_cvar_db"
    assert cfg["trusted_band_hz"] == pytest.approx(
        ControlBandContract.broadband_point_control().point_control_target_hz
    )
    stamped = stamp_experiment_contract(cfg, repo_root=tmp_path)
    artifacts = stamped["experiment_contract"]["artifacts"]
    assert artifacts["broadband_source_raw"]["sha256"] == _file_sha256(
        plant.with_suffix(".raw.npz")
    )
    assert artifacts["broadband_source_analysis"]["sha256"] == _file_sha256(
        plant.with_suffix(".analysis.npz")
    )
    assert artifacts["broadband_measurement_level_evidence"][
        "sha256"
    ] == _file_sha256(plant.with_suffix(".level.json"))
    assert artifacts["broadband_source_plan"]["sha256"] == _file_sha256(
        plant.with_suffix(".plan.json")
    )
    assert artifacts["broadband_fresh_meter_raw"]["sha256"] == _file_sha256(
        plant.with_suffix(".meter.npz")
    )
    assert artifacts["broadband_fresh_meter_receipt"][
        "sha256"
    ] == _file_sha256(meter_receipt_path(plant.with_suffix(".meter.npz")))

    class Model:
        limit = 0.2

    with pytest.raises(ValueError, match="admission BLOCKED"):
        admit_criterion_config(
            cfg,
            repo_root=tmp_path,
            require_bound=True,
        )
