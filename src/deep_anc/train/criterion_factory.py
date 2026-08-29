"""Stage-1/광대역 ANC criterion의 단일 admission·construction 경계.

광대역 loss schema 문자열만 바꾸어 현 strict-v1 S(z)를 11.314 kHz plant처럼 쓰는
우회를 막는다. 광대역 역할은 모델/CUDA/DataLoader를 만들기 전에 resolved config와
실제 S NPZ를 함께 검증해야 하며, Trainer와 campaign evidence는 반드시 이 모듈의
같은 factory를 소비한다.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from ..dsp.control_band_contract import (
    BroadbandPlantEvidence,
    ControlBandContract,
    audit_broadband_plant_evidence,
)
from ..dsp.causal_training_operator import (
    CausalTrainingAuthorityData,
    load_causal_training_authority,
)
from ..dsp.measurement_level import meter_receipt_path
from ..dsp.measured_band_path import (
    MEASURED_BAND_HOLDOUT_SCHEMA,
    MEASURED_BAND_INTERPOLATION_SCHEMA,
    MEASURED_BAND_PATH_SCHEMA_VERSION,
    MeasuredBandPath,
    MeasuredBandPathData,
    load_measured_band_path,
)
from ..dsp.nonlinear import RandomNonlinear
from ..dsp.secondary_path import (
    DifferentiableSecondaryPath,
    SecondaryPathData,
    load_secondary_path,
)
from ..dsp.timing import BandPlan, handoff_samples_from_config
from ..models.broadband_representability import output_lattice_contract
from ..losses import ANCLoss, BroadbandANCLoss, BroadbandLossConfig, LossConfig
from ..losses.broadband_loss import (
    BROADBAND_CAUSAL_CONVOLUTION_SCHEMA,
    BROADBAND_CAUSAL_INTERPOLATION_SCHEMA,
    BROADBAND_CAUSAL_PATH_SCHEMA,
    BROADBAND_DNH_CALIBRATION_PLACEHOLDER,
    BROADBAND_DNH_DOMAIN,
    BROADBAND_DNH_SCHEMA_VERSION,
    BROADBAND_LINEAR_SPECTRAL_SCHEMA,
    BROADBAND_LOSS_SCHEMA_VERSION,
    CausalFIRPath,
    CausalFIRPathData,
)


STAGE1_CRITERION_ROLE = "stage1_trusted_band"
BROADBAND_CRITERION_ROLE = "broadband_point_control"
BROADBAND_PLANT_ARTIFACT_SCHEMA = "broadband_measured_band_plant_v2_raw_derived"
BROADBAND_SEGMENT_BOUNDARY_SCHEMA = (
    "blocked_stateless_random_segment_missing_linear_history_v1"
)
BROADBAND_SEGMENT_BOUNDARY_STATUS = "BLOCKED_MISSING_PREFIX_OR_STATE"
BROADBAND_SYNTH_PRIMARY_STATUS = "BLOCKED_COMPACT_PRIMARY_GENERATOR"
BROADBAND_CAUSAL_AUTHORITY_CONFIG_SCHEMA = (
    "fullband_causal_training_authority_config_v1"
)
BROADBAND_CAUSAL_PREFIX_SCHEMA = (
    "continuous_session_prefix_model_and_joint_ps_history_v1"
)


@dataclass(frozen=True)
class CriterionAdmission:
    role: Literal["stage1_trusted_band", "broadband_point_control"]
    loss_schema_version: str | None
    control_band_contract_sha256: str | None
    primary_path: Path | None
    secondary_path: Path
    broadband_plant_evidence_sha256: str | None
    broadband_source_raw_path: Path | None
    broadband_source_analysis_path: Path | None
    measurement_level_evidence_path: Path | None
    broadband_source_plan_path: Path | None
    broadband_fresh_meter_raw_path: Path | None
    broadband_fresh_meter_receipt_path: Path | None
    broadband_derived_lead_samples: int | None
    secondary: SecondaryPathData | None
    primary_measured_band: MeasuredBandPathData | None
    secondary_measured_band: MeasuredBandPathData | None
    measured_band_contract_sha256: str | None
    target_band_hz: tuple[float, float]
    trusted_band_hz: tuple[float, float]
    band_plan: BandPlan | None
    causal_authority: CausalTrainingAuthorityData | None = None
    primary_causal: CausalFIRPathData | None = None
    secondary_causal: CausalFIRPathData | None = None
    broadband_valid_prefix_samples: int | None = None
    broadband_causal_operator_contract: dict[str, Any] | None = None


@dataclass(frozen=True)
class CriterionBundle:
    criterion: ANCLoss
    admission: CriterionAdmission


def _repo_file(root: str | Path, value: str | Path, *, label: str) -> Path:
    """저장소 root 안의 symlink 없는 regular file만 허용한다."""

    base = Path(os.path.abspath(Path(root).expanduser()))
    if base.is_symlink() or not base.is_dir():
        raise ValueError(f"{label} 저장소 root가 유효하지 않습니다: {base}")
    raw = Path(value).expanduser()
    path = Path(os.path.abspath(raw if raw.is_absolute() else base / raw))
    try:
        relative = path.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"{label}는 저장소 내부여야 합니다: {path}") from exc
    cursor = base
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise ValueError(f"{label} 경로에 symlink가 있습니다: {cursor}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} 파일이 없습니다: {path}")
    return path


def _npz_scalar(data: Any, key: str) -> Any:
    if key not in data:
        raise ValueError(f"광대역 S NPZ에 {key} metadata가 없습니다")
    value = data[key]
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"광대역 S NPZ의 {key}가 scalar가 아닙니다")
    return array.reshape(-1)[0].item()


def _sha256(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label}는 lowercase SHA-256이어야 합니다")
    return text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _measured_band_contract_payload(
    primary: MeasuredBandPathData, secondary: MeasuredBandPathData
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "broadband_measured_band_criterion_v1",
        "path_schema_version": MEASURED_BAND_PATH_SCHEMA_VERSION,
        "interpolation_schema": MEASURED_BAND_INTERPOLATION_SCHEMA,
        "holdout_schema": MEASURED_BAND_HOLDOUT_SCHEMA,
        "linear_spectral_schema": BROADBAND_LINEAR_SPECTRAL_SCHEMA,
        "dnh_domain": BROADBAND_DNH_DOMAIN,
        "dnh_schema_version": BROADBAND_DNH_SCHEMA_VERSION,
        "dnh_calibration_status": BROADBAND_DNH_CALIBRATION_PLACEHOLDER,
        "primary_response_sha256": primary.response_sha256,
        "secondary_response_sha256": secondary.response_sha256,
        "primary_holdout_passed": bool(primary.holdout_receipt.get("passed")),
        "secondary_holdout_passed": bool(secondary.holdout_receipt.get("passed")),
        "primary_bulk_delay_fractional_samples": (
            primary.bulk_delay_fractional_samples
        ),
        "secondary_bulk_delay_fractional_samples": (
            secondary.bulk_delay_fractional_samples
        ),
        "primary_fractional_effective_delay_samples": (
            primary.fractional_effective_delay_samples
        ),
        "secondary_fractional_effective_delay_samples": (
            secondary.fractional_effective_delay_samples
        ),
        "primary_pre_roll_samples": primary.pre_roll_samples,
        "secondary_pre_roll_samples": secondary.pre_roll_samples,
        "delay_semantics": "full_bulk_measured_response_plus_handoff_once",
        "segment_boundary_schema": BROADBAND_SEGMENT_BOUNDARY_SCHEMA,
        "segment_boundary_status": BROADBAND_SEGMENT_BOUNDARY_STATUS,
        "synthetic_primary_generator_representation": (
            "resolve_digital_primary_path_compact_fir"
        ),
        "synthetic_primary_generator_status": BROADBAND_SYNTH_PRIMARY_STATUS,
        "extrapolation": "forbid",
        "time_domain_compact_fir": "forbidden",
    }
    payload["sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def _band(value: object, *, label: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 2 or not np.all(np.isfinite(array)) or not 0.0 <= array[0] < array[1]:
        raise ValueError(f"{label}가 유효한 [lo, hi]가 아닙니다")
    return float(array[0]), float(array[1])


def _covers(actual: tuple[float, float], required: tuple[float, float]) -> bool:
    return actual[0] <= required[0] + 1.0e-9 and actual[1] >= required[1] - 1.0e-9


def _validate_broadband_plant_npz(
    path: Path,
    *,
    plant: SecondaryPathData,
    role: Literal["primary", "secondary"],
    contract: ControlBandContract,
    repo_root: str | Path,
    configured_evidence_sha256: str,
) -> tuple[BroadbandPlantEvidence, str, Path, Path, Path, Path, Path, Path]:
    """P/S NPZ와 embedded evidence/외부 immutable 입력을 모두 재검산한다.

    최종 analysis JSON은 P/S SHA를 포함하므로 그 JSON SHA를 P/S에 다시 넣으면
    순환한다. 따라서 publisher가 S NPZ에 넣은 canonical evidence payload의 digest를
    config authority로 쓰고, payload가 가리키는 raw/analysis/level 파일 bytes를 직접
    확인한다. NPZ 이름이나 metadata label만 바꾸어 통과할 수 없다.
    """

    plant_label = "P" if role == "primary" else "S"
    with np.load(path, allow_pickle=False) as data:
        artifact_sha = _sha256(
            _npz_scalar(data, "control_band_contract_sha256"),
            label="광대역 S NPZ control-band contract SHA",
        )
        sample_rate = int(_npz_scalar(data, "sample_rate"))
        artifact_schema = str(_npz_scalar(data, "schema_version"))
        artifact_role = str(_npz_scalar(data, "plant_role"))
        evidence_json = str(_npz_scalar(data, "broadband_plant_evidence_json"))
        artifact_evidence_sha = _sha256(
            _npz_scalar(data, "broadband_plant_evidence_sha256"),
            label="광대역 S NPZ embedded plant evidence SHA",
        )
        raw_path_value = str(_npz_scalar(data, "source_raw_npz_path"))
        artifact_raw_sha = _sha256(
            _npz_scalar(data, "source_raw_npz_sha256"),
            label="광대역 S NPZ source raw SHA",
        )
        analysis_path_value = str(_npz_scalar(data, "source_analysis_npz_path"))
        artifact_analysis_sha = _sha256(
            _npz_scalar(data, "source_analysis_npz_sha256"),
            label="광대역 S NPZ source analysis SHA",
        )
        level_path_value = str(_npz_scalar(data, "measurement_level_evidence_path"))
        plan_path_value = str(_npz_scalar(data, "source_plan_path"))
        meter_raw_path_value = str(_npz_scalar(data, "fresh_meter_raw_path"))
        artifact_level_sha = _sha256(
            _npz_scalar(data, "measurement_level_evidence_sha256"),
            label="광대역 S NPZ level evidence SHA",
        )
        artifact_authority_sha = {
            evidence_field: _sha256(
                _npz_scalar(data, artifact_field),
                label=f"광대역 plant {artifact_field}",
            )
            for artifact_field, evidence_field in (
                ("source_plan_file_sha256", "exact_plan_file_sha256"),
                ("source_plan_payload_sha256", "exact_plan_payload_sha256"),
                ("source_plan_pcm_sha256", "exact_plan_pcm_sha256"),
                ("fresh_meter_raw_sha256", "fresh_meter_raw_sha256"),
                ("fresh_meter_receipt_sha256", "fresh_meter_receipt_sha256"),
            )
        }
        if "excitation_band_hz" not in data:
            raise ValueError("광대역 S NPZ에 excitation_band_hz metadata가 없습니다")
        if "consistency_band_hz" not in data:
            raise ValueError("광대역 S NPZ에 consistency_band_hz metadata가 없습니다")
        if "verified_subbands_hz" not in data:
            raise ValueError("광대역 S NPZ에 verified_subbands_hz metadata가 없습니다")
        excitation = _band(data["excitation_band_hz"], label="광대역 S excitation band")
        consistency = _band(
            data["consistency_band_hz"], label="광대역 S consistency band"
        )
        verified = np.asarray(data["verified_subbands_hz"], dtype=np.float64)
        if "band_consistency" not in data:
            raise ValueError("광대역 S NPZ에 band_consistency metadata가 없습니다")
        plant_consistency = np.asarray(
            data["band_consistency"], dtype=np.float64
        ).reshape(-1)
        artifact_timing_scalars = {
            field: _npz_scalar(data, field)
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
            )
        }
        artifact_compact_role = str(_npz_scalar(data, "compact_role"))
        artifact_compact_training_eligible = bool(
            _npz_scalar(data, "compact_training_eligible")
        )
        artifact_compact_identifiability_sha = _sha256(
            _npz_scalar(data, "compact_identifiability_sha256"),
            label="광대역 compact identifiability SHA",
        )
        artifact_interpolation_agreement = np.asarray(
            data["measured_interpolation_agreement"], dtype=np.float64
        ).reshape(-1)
        artifact_interpolation_error = np.asarray(
            data["measured_interpolation_relative_error"], dtype=np.float64
        ).reshape(-1)
        artifact_panel_relative = np.asarray(
            data["panel_primary_minus_secondary_bulk_delay_samples"],
            dtype=np.float64,
        ).reshape(-1) if "panel_primary_minus_secondary_bulk_delay_samples" in data else None
        artifact_panel_deviation = np.asarray(
            data["panel_relative_delay_deviation_samples"], dtype=np.float64
        ).reshape(-1) if "panel_relative_delay_deviation_samples" in data else None

    if artifact_schema != BROADBAND_PLANT_ARTIFACT_SCHEMA or artifact_role != role:
        raise ValueError(
            f"광대역 {plant_label} NPZ schema/role이 publisher 계약과 다릅니다: "
            f"schema={artifact_schema!r}, role={artifact_role!r}"
        )
    try:
        evidence_payload = json.loads(evidence_json)
        evidence = BroadbandPlantEvidence.model_validate(evidence_payload)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"광대역 {plant_label} embedded plant evidence가 유효하지 않습니다: {exc}"
        ) from exc
    canonical_evidence = _canonical_json(evidence.model_dump(mode="json"))
    if evidence_json.encode("utf-8") != canonical_evidence:
        raise ValueError(
            f"광대역 {plant_label} embedded plant evidence가 canonical JSON이 아닙니다"
        )
    evidence_sha = hashlib.sha256(canonical_evidence).hexdigest()
    if artifact_evidence_sha != evidence_sha:
        raise ValueError(
            f"광대역 {plant_label} NPZ embedded plant evidence SHA가 "
            "canonical payload와 다릅니다"
        )
    if configured_evidence_sha256 != evidence_sha:
        raise ValueError(
            f"resolved config plant evidence SHA가 {plant_label} NPZ embedded "
            "evidence와 다릅니다"
        )
    audit = audit_broadband_plant_evidence(contract, evidence)
    if not audit.ok:
        raise ValueError(
            "broadband plant evidence가 threshold를 통과하지 못했습니다: "
            + "; ".join(audit.reasons)
        )
    for field, artifact_value in artifact_authority_sha.items():
        if artifact_value != getattr(evidence, field):
            raise ValueError(
                f"광대역 {plant_label} NPZ {field}가 embedded plant evidence와 다릅니다"
            )
    for field, artifact_value in artifact_timing_scalars.items():
        evidence_value = getattr(evidence, field)
        if artifact_value != evidence_value:
            raise ValueError(
                f"광대역 {plant_label} NPZ {field}가 embedded plant evidence와 다릅니다"
            )
    expected_compact_role = getattr(evidence, f"{role}_compact_role")
    expected_compact_eligible = getattr(
        evidence, f"{role}_compact_training_eligible"
    )
    expected_compact_sha = getattr(
        evidence, f"{role}_compact_identifiability_sha256"
    )
    if (
        artifact_compact_role != expected_compact_role
        or artifact_compact_training_eligible != expected_compact_eligible
        or artifact_compact_identifiability_sha != expected_compact_sha
        or artifact_compact_role != "diagnostic_only"
        or artifact_compact_training_eligible
    ):
        raise ValueError(
            f"광대역 {plant_label} compact artifact가 diagnostic-only evidence와 다릅니다"
        )
    for label, artifact_values, evidence_values in (
        (
            "measured interpolation agreement",
            artifact_interpolation_agreement,
            evidence.measured_interpolation_agreement,
        ),
        (
            "measured interpolation relative error",
            artifact_interpolation_error,
            evidence.measured_interpolation_relative_error,
        ),
    ):
        expected_values = np.asarray(evidence_values, dtype=np.float64)
        if not np.array_equal(artifact_values, expected_values):
            raise ValueError(
                f"광대역 {plant_label} {label}가 embedded evidence와 다릅니다"
            )
    for field, artifact_value in (
        (
            "panel_primary_minus_secondary_bulk_delay_samples",
            artifact_panel_relative,
        ),
        ("panel_relative_delay_deviation_samples", artifact_panel_deviation),
    ):
        if artifact_value is None:
            raise ValueError(
                f"광대역 {plant_label} NPZ에 {field} metadata가 없습니다"
            )
        evidence_value = np.asarray(getattr(evidence, field), dtype=np.float64)
        if not np.array_equal(artifact_value, evidence_value):
            raise ValueError(
                f"광대역 {plant_label} NPZ {field}가 embedded plant evidence와 다릅니다"
            )
    raw_path = _repo_file(repo_root, raw_path_value, label="broadband source raw")
    analysis_path = _repo_file(
        repo_root, analysis_path_value, label="broadband source analysis"
    )
    level_path = _repo_file(
        repo_root, level_path_value, label="broadband measurement-level evidence"
    )
    plan_path = _repo_file(repo_root, plan_path_value, label="broadband exact plan")
    meter_raw_path = _repo_file(
        repo_root, meter_raw_path_value, label="broadband fresh meter raw"
    )
    meter_receipt = meter_receipt_path(meter_raw_path)
    meter_receipt = _repo_file(
        repo_root, meter_receipt, label="broadband fresh meter receipt"
    )
    for label, external_path, expected_sha in (
        ("source raw", raw_path, artifact_raw_sha),
        ("source analysis", analysis_path, artifact_analysis_sha),
        ("measurement-level evidence", level_path, artifact_level_sha),
    ):
        actual_sha = _file_sha256(external_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"광대역 {plant_label} {label} SHA가 실제 파일 bytes와 다릅니다: "
                f"declared={expected_sha}, actual={actual_sha}"
            )
    try:
        plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"광대역 exact plan JSON을 검증할 수 없습니다: {exc}") from exc
    plan_payload_sha = hashlib.sha256(_canonical_json(plan_payload)).hexdigest()
    if plan_payload_sha != artifact_authority_sha["exact_plan_payload_sha256"]:
        raise ValueError(
            "광대역 exact plan canonical payload SHA가 embedded evidence와 다릅니다"
        )
    output = plan_payload.get("output") if isinstance(plan_payload, dict) else None
    plan_pcm_sha = output.get("pcm_sha256") if isinstance(output, dict) else None
    plan_pcm_sha = _sha256(plan_pcm_sha, label="광대역 exact plan output PCM SHA")
    if plan_pcm_sha != artifact_authority_sha["exact_plan_pcm_sha256"]:
        raise ValueError(
            "광대역 exact plan output PCM SHA가 embedded evidence와 다릅니다"
        )
    for label, external_path, expected_sha in (
        ("exact plan", plan_path, artifact_authority_sha["exact_plan_file_sha256"]),
        ("fresh meter raw", meter_raw_path, artifact_authority_sha["fresh_meter_raw_sha256"]),
        (
            "fresh meter receipt",
            meter_receipt,
            artifact_authority_sha["fresh_meter_receipt_sha256"],
        ),
    ):
        actual_sha = _file_sha256(external_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"광대역 {plant_label} {label} SHA가 실제 파일 bytes와 다릅니다: "
                f"declared={expected_sha}, actual={actual_sha}"
            )

    required_sha = contract.digest()
    if artifact_sha != required_sha:
        raise ValueError(
            f"광대역 {plant_label} NPZ control-band contract SHA가 현재 계약과 다릅니다: "
            f"artifact={artifact_sha}, required={required_sha}"
        )
    evidence_raw_sha = str(getattr(evidence, f"{role}_raw_sha256"))
    evidence_analysis_sha = str(getattr(evidence, f"{role}_analysis_sha256"))
    if artifact_raw_sha != evidence_raw_sha:
        raise ValueError(
            f"광대역 {plant_label} NPZ raw SHA가 plant evidence와 다릅니다"
        )
    if artifact_analysis_sha != evidence_analysis_sha:
        raise ValueError(
            f"광대역 {plant_label} NPZ analysis SHA가 plant evidence와 다릅니다"
        )
    if artifact_level_sha != evidence.measurement_level_evidence_sha256:
        raise ValueError(
            f"광대역 {plant_label} NPZ level evidence SHA가 plant evidence와 다릅니다"
        )
    if sample_rate != contract.sample_rate or int(plant.sample_rate) != contract.sample_rate:
        raise ValueError(
            f"광대역 {plant_label} NPZ sample rate가 control-band 계약과 다릅니다: "
            f"npz={sample_rate}, loaded={plant.sample_rate}, contract={contract.sample_rate}"
        )
    expected_delay = int(getattr(evidence, f"{role}_effective_delay_samples"))
    if int(plant.delay_samples) != expected_delay:
        raise ValueError(
            f"광대역 {plant_label} NPZ delay_samples가 evidence {role} "
            "effective delay와 다릅니다"
        )
    required_band = tuple(contract.point_control_target_hz)
    if not _covers(excitation, required_band):
        raise ValueError(
            f"광대역 {plant_label} excitation band가 150–11.314kHz를 덮지 않습니다: "
            f"actual={excitation}, required={required_band}"
        )
    if not _covers(consistency, required_band):
        raise ValueError(
            f"광대역 {plant_label} consistency band가 150–11.314kHz를 덮지 않습니다: "
            f"actual={consistency}, required={required_band}"
        )
    expected = np.asarray(contract.point_control_subbands_hz, dtype=np.float64)
    if verified.shape != expected.shape or not np.array_equal(verified, expected):
        raise ValueError(
            f"광대역 {plant_label} verified_subbands_hz가 7개 control subband와 정확히 다릅니다"
        )
    evidence_verified = np.asarray(evidence.verified_subbands_hz, dtype=np.float64)
    evidence_panels = np.asarray(evidence.excitation_panels_hz, dtype=np.float64)
    expected_panels = np.asarray(contract.measurement_panels_hz, dtype=np.float64)
    if not np.array_equal(evidence_verified, expected):
        raise ValueError("광대역 embedded evidence subband가 control 계약과 exact하지 않습니다")
    if not np.array_equal(evidence_panels, expected_panels):
        raise ValueError("광대역 embedded evidence panel이 control 계약과 exact하지 않습니다")
    evidence_consistency = np.asarray(
        getattr(evidence, f"{role}_consistency"), dtype=np.float64
    )
    if plant_consistency.shape != (len(expected),) or not np.all(
        np.isfinite(plant_consistency)
    ):
        raise ValueError(
            f"광대역 {plant_label} consistency vector가 7개 유한값이 아닙니다"
        )
    if not np.array_equal(plant_consistency, evidence_consistency):
        raise ValueError(
            f"광대역 {plant_label} consistency vector가 raw plant evidence와 다릅니다"
        )
    if np.any(plant_consistency < 0.95):
        raise ValueError(
            f"광대역 {plant_label} consistency가 한 subband라도 0.95 미만입니다"
        )
    loaded_excitation = tuple(float(value) for value in plant.excitation_band_hz)
    loaded_consistency = tuple(float(value) for value in plant.trusted_band_hz())
    if loaded_excitation != excitation or loaded_consistency != consistency:
        raise ValueError(
            f"광대역 {plant_label} NPZ raw metadata와 SecondaryPathData 해석이 다릅니다"
        )
    return (
        evidence,
        evidence_sha,
        raw_path,
        analysis_path,
        level_path,
        plan_path,
        meter_raw_path,
        meter_receipt,
    )


def _causal_prefix_samples(
    cfg: dict[str, Any], authority: CausalTrainingAuthorityData
) -> tuple[int, dict[str, Any]]:
    """plant/model/feedback/EQ history를 덮는 256-aligned real prefix를 유도한다."""

    data = cfg["data"]
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("광대역 causal prefix 유도에 model mapping이 필요합니다")
    lattice = output_lattice_contract(model_cfg)
    if lattice.get("structural_passed") is not True:
        raise ValueError(
            "광대역 model lattice 구조 gate 실패: "
            + "; ".join(str(value) for value in lattice.get("reasons", ()))
        )
    closed_loop = data.get("closed_loop") or {}
    feedback = closed_loop.get("feedback_delay_samples", [512, 1024])
    try:
        feedback_history = max(int(value) for value in feedback)
    except (TypeError, ValueError) as exc:
        raise ValueError("광대역 feedback delay 계약이 유효하지 않습니다") from exc
    if feedback_history < 0:
        raise ValueError("광대역 feedback delay는 0 이상이어야 합니다")
    augment = data.get("recorded_augment") or {}
    eq_enabled = bool(augment.get("enabled", False)) and (
        float(augment.get("eq_tilt_db", 6.0)) > 0.0
        or float(augment.get("eq_band_db", 4.0)) > 0.0
    )
    # recorded common EQ는 129 taps linear-phase이며, valid target 시작에
    # zero boundary가 닿지 않도록 half-history 64를 feedback 앞에 더한다.
    eq_half_history = 64 if eq_enabled else 0
    components = {
        "secondary_causal_history_samples": int(authority.secondary_history_samples),
        "model_finite_branch_input_span_samples": int(
            lattice["finite_branch_input_span_samples"]
        ),
        "feedback_plus_eq_history_samples": int(
            feedback_history + eq_half_history
        ),
    }
    required = max(components.values())
    derived = ((required + 255) // 256) * 256
    payload: dict[str, Any] = {
        "schema": BROADBAND_CAUSAL_PREFIX_SCHEMA,
        "required_components": components,
        "alignment_samples": 256,
        "valid_prefix_samples": int(derived),
        "model_state_initialization": "real_continuous_prefix_from_zero_stream_state",
        "plant_state_initialization": "full_linear_causal_history_then_valid_crop",
    }
    payload["sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return int(derived), payload


def _admit_causal_broadband(
    cfg: dict[str, Any],
    *,
    repo_root: str | Path,
    require_bound: bool,
    contract: ControlBandContract,
) -> CriterionAdmission:
    """v4 joint P/S authority를 단일 source로 causal 광대역 경로를 승인."""

    data = cfg["data"]
    duct = cfg["duct"]
    loss = cfg["loss"]
    raw = cfg.get("broadband_causal_training_authority")
    expected_keys = {
        "schema", "path", "file_sha256", "evidence_sha256"
    }
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        raise ValueError(
            "broadband_causal_training_authority는 schema/path/file/evidence SHA "
            "exact mapping이어야 합니다"
        )
    if raw["schema"] != BROADBAND_CAUSAL_AUTHORITY_CONFIG_SCHEMA:
        raise ValueError("광대역 causal authority config schema가 다릅니다")
    if str(data.get("digital_primary_path_mode")) != "causal_joint_v4":
        raise ValueError(
            "causal joint P view는 data.digital_primary_path_mode=causal_joint_v4로 "
            "tone-only measured/compact key와 분리해야 합니다"
        )
    if (
        loss.get("plant_representation_schema") != BROADBAND_CAUSAL_PATH_SCHEMA
        or loss.get("interpolation_schema") != BROADBAND_CAUSAL_INTERPOLATION_SCHEMA
        or loss.get("linear_spectral_schema") != BROADBAND_CAUSAL_CONVOLUTION_SCHEMA
    ):
        raise ValueError("광대역 loss가 v4 causal FIR/prefix convolution schema가 아닙니다")
    dropout = data.get("broadband_channel_dropout")
    if not isinstance(dropout, dict) or set(dropout) != {
        "reference_probability", "error_probability"
    }:
        raise ValueError(
            "causal broadband은 reference/error dropout 확률을 exact mapping으로 "
            "명시해야 합니다"
        )
    reference_dropout = float(dropout["reference_probability"])
    error_dropout = float(dropout["error_probability"])
    if reference_dropout != 0.0 or not 0.0 <= error_dropout <= 1.0:
        raise ValueError(
            "canonical digital-reference의 x_ref dropout은 exact 0이어야 하고 "
            "error dropout은 [0,1]이어야 합니다"
        )
    authority_path = _repo_file(repo_root, str(raw["path"]), label="causal authority")
    authority = load_causal_training_authority(
        authority_path,
        expected_file_sha256=str(raw["file_sha256"]),
        expected_evidence_sha256=str(raw["evidence_sha256"]),
        require_live_authority=True,
    )
    if authority.payload["control_band_contract_sha256"] != contract.digest():
        raise ValueError("causal authority control-band contract SHA가 현재 계약과 다릅니다")
    configured_handoff = handoff_samples_from_config(duct)
    if configured_handoff != authority.plant_delays.handoff_samples:
        raise ValueError("causal authority/config handoff가 다릅니다")
    if int(data.get("sample_rate", 0)) != authority.operator.sample_rate_hz:
        raise ValueError("causal authority/data sample rate가 다릅니다")
    prefix, prefix_contract = _causal_prefix_samples(cfg, authority)
    configured_prefix = data.get("broadband_model_prefix_samples")
    configured_loss_start = cfg.get("loss_start_sample")
    configured_timing = data.get("training_timing_contract")
    configured_operator_contract = cfg.get("broadband_causal_operator_contract")
    operator_contract: dict[str, Any] = {
        "schema": "joint_causal_training_operator_binding_v1",
        "authority_file_sha256": authority.authority_file_sha256,
        "authority_evidence_sha256": authority.authority_evidence_sha256,
        "operator_file_sha256": authority.operator.file_sha256,
        "operator_internal_sha256": authority.operator.internal_sha256,
        "primary_fir_sha256": authority.operator.primary_fir_sha256,
        "secondary_fir_sha256": authority.operator.secondary_fir_sha256,
        "timing_contract_sha256": authority.timing_contract.digest(),
        "prefix_contract": prefix_contract,
        "input_channel_contract": {
            "schema": "digital_reference_error_context_policy_v1",
            "reference_dropout_probability": reference_dropout,
            "error_dropout_probability": error_dropout,
            "training_input_mode": (
                "digital_reference_only_err_zero"
                if error_dropout == 1.0
                else "digital_reference_plus_error_context"
            ),
            "output_clock_master_runtime_eligible": False,
            "runtime_blocker": (
                "requires_same_absolute_g0_and_val_gates_with_err_forced_zero_or_"
                "validated_asrc_clock_bridge"
            ),
        },
        "inline_receipt_sha256": dict(authority.inline_receipt_sha256),
    }
    operator_contract["sha256"] = hashlib.sha256(
        _canonical_json(operator_contract)
    ).hexdigest()
    declared_lead = data.get("digital_reference_lead_samples")
    expected_lead = int(authority.timing_contract.digital_reference_lead_samples)
    if declared_lead is not None and int(declared_lead) != expected_lead:
        raise ValueError("configured digital-reference lead가 causal authority와 다릅니다")
    for label, configured, expected in (
        ("broadband model prefix", configured_prefix, prefix),
        ("loss start", configured_loss_start, prefix),
        ("training timing", configured_timing, authority.timing_contract.model_dump()),
        ("causal operator contract", configured_operator_contract, operator_contract),
    ):
        if configured is not None and configured != expected:
            raise ValueError(f"configured {label}가 causal authority 유도값과 다릅니다")
    if require_bound and (
        int(configured_prefix if configured_prefix is not None else -1) != prefix
        or int(configured_loss_start if configured_loss_start is not None else -1)
        != prefix
        or configured_timing != authority.timing_contract.model_dump()
        or configured_operator_contract != operator_contract
    ):
        raise ValueError(
            "resolved config의 causal timing/prefix/operator binding이 authority와 다릅니다"
        )
    op = authority.operator
    common = {
        "support_samples": op.support_samples,
        "sample_rate": op.sample_rate_hz,
        "operator_file_sha256": op.file_sha256,
        "operator_internal_sha256": op.internal_sha256,
        "authority_sha256": authority.authority_evidence_sha256,
        "source_path": str(op.path),
    }
    primary_data = CausalFIRPathData(
        role="primary",
        post_onset_fir=op.primary_post_onset_fir,
        coarse_delay_samples=op.primary_coarse_delay_samples,
        fractional_delay_samples=op.primary_fractional_delay_samples,
        handoff_extra_samples=0,
        fir_sha256=op.primary_fir_sha256,
        **common,
    )
    secondary_data = CausalFIRPathData(
        role="secondary",
        post_onset_fir=op.secondary_post_onset_fir,
        coarse_delay_samples=op.secondary_coarse_delay_samples,
        fractional_delay_samples=op.secondary_fractional_delay_samples,
        handoff_extra_samples=configured_handoff,
        fir_sha256=op.secondary_fir_sha256,
        **common,
    )
    target = tuple(contract.point_control_target_hz)
    return CriterionAdmission(
        role=BROADBAND_CRITERION_ROLE,
        loss_schema_version=BROADBAND_LOSS_SCHEMA_VERSION,
        control_band_contract_sha256=contract.digest(),
        primary_path=op.path,
        secondary_path=op.path,
        broadband_plant_evidence_sha256=None,
        broadband_source_raw_path=None,
        broadband_source_analysis_path=authority.authority_path,
        measurement_level_evidence_path=None,
        broadband_source_plan_path=None,
        broadband_fresh_meter_raw_path=None,
        broadband_fresh_meter_receipt_path=None,
        broadband_derived_lead_samples=int(
            authority.timing_contract.digital_reference_lead_samples
        ),
        secondary=None,
        primary_measured_band=None,
        secondary_measured_band=None,
        measured_band_contract_sha256=None,
        target_band_hz=target,
        trusted_band_hz=target,
        band_plan=None,
        causal_authority=authority,
        primary_causal=primary_data,
        secondary_causal=secondary_data,
        broadband_valid_prefix_samples=prefix,
        broadband_causal_operator_contract=operator_contract,
    )


def admit_criterion_config(
    cfg: dict[str, Any],
    *,
    repo_root: str | Path,
    require_bound: bool,
) -> CriterionAdmission:
    """resolved config와 S NPZ를 모델/CUDA/DataLoader 전에 승인한다."""

    data = cfg.get("data")
    duct = cfg.get("duct")
    loss = cfg.get("loss")
    if not isinstance(data, dict) or not isinstance(duct, dict) or not isinstance(loss, dict):
        raise ValueError("criterion admission에는 resolved data/duct/loss mapping이 필요합니다")
    schema = loss.get("schema_version")
    declared_role = cfg.get("criterion_role")
    declared_sha = cfg.get("control_band_contract_sha256")
    if schema == BROADBAND_LOSS_SCHEMA_VERSION and cfg.get(
        "broadband_causal_training_authority"
    ) is not None:
        BroadbandLossConfig.parse(loss)
        contract = ControlBandContract.broadband_point_control()
        configured_sha = _sha256(
            declared_sha, label="resolved config control-band contract SHA"
        )
        if configured_sha != contract.digest():
            raise ValueError("resolved config control-band contract SHA가 광대역 v2와 다릅니다")
        if require_bound and declared_role != BROADBAND_CRITERION_ROLE:
            raise ValueError("광대역 checkpoint/resolved config에 criterion_role binding이 없습니다")
        if declared_role not in (None, BROADBAND_CRITERION_ROLE):
            raise ValueError("광대역 loss schema와 criterion_role이 다릅니다")
        return _admit_causal_broadband(
            cfg,
            repo_root=repo_root,
            require_bound=require_bound,
            contract=contract,
        )
    secondary_cfg = duct.get("secondary_path")
    if not isinstance(secondary_cfg, dict) or not secondary_cfg.get("npz"):
        raise ValueError("criterion admission에 duct.secondary_path.npz가 없습니다")
    secondary_path = _repo_file(
        repo_root,
        str(secondary_cfg["npz"]),
        label="criterion secondary path",
    )
    secondary = load_secondary_path(secondary_path)
    sample_rate = int(data.get("sample_rate", 0))
    if sample_rate <= 0 or int(secondary.sample_rate) != sample_rate:
        raise ValueError(
            "criterion S(z)/data sample rate가 다릅니다: "
            f"S={secondary.sample_rate}, data={sample_rate}"
        )

    if schema is None:
        # Stage-1 기본 경로는 기존 config bytes/의미를 그대로 유지한다. 광대역 표식
        # 일부만 주입한 모호한 상태는 이름 바꾸기 우회이므로 거부한다.
        if declared_role is not None or declared_sha is not None:
            raise ValueError(
                "loss.schema_version 없는 Stage-1 설정에 criterion_role/control-band "
                "SHA를 부분 주입할 수 없습니다"
            )
        LossConfig.parse(loss)
        band_plan = BandPlan.resolve(
            plant_trusted_band_hz=secondary.trusted_band_hz(),
            duct_cfg=duct,
            sample_rate=sample_rate,
        )
        return CriterionAdmission(
            role=STAGE1_CRITERION_ROLE,
            loss_schema_version=None,
            control_band_contract_sha256=None,
            primary_path=None,
            secondary_path=secondary_path,
            broadband_plant_evidence_sha256=None,
            broadband_source_raw_path=None,
            broadband_source_analysis_path=None,
            measurement_level_evidence_path=None,
            broadband_source_plan_path=None,
            broadband_fresh_meter_raw_path=None,
            broadband_fresh_meter_receipt_path=None,
            broadband_derived_lead_samples=None,
            secondary=secondary,
            primary_measured_band=None,
            secondary_measured_band=None,
            measured_band_contract_sha256=None,
            target_band_hz=band_plan.target.as_tuple(),
            trusted_band_hz=band_plan.optimize.as_tuple(),
            band_plan=band_plan,
        )
    if schema != BROADBAND_LOSS_SCHEMA_VERSION:
        raise ValueError(f"알 수 없는 loss.schema_version입니다: {schema!r}")

    BroadbandLossConfig.parse(loss)
    primary_mode = str(data.get("digital_primary_path_mode", "rir_surrogate"))
    if primary_mode != "measured":
        raise ValueError(
            "광대역 criterion은 legacy/secondary-surrogate compact P 우회를 "
            "허용하지 않습니다; data.digital_primary_path_mode=measured가 필요합니다"
        )
    contract = ControlBandContract.broadband_point_control()
    required_sha = contract.digest()
    configured_sha = _sha256(
        declared_sha, label="resolved config control-band contract SHA"
    )
    if configured_sha != required_sha:
        raise ValueError(
            "resolved config control-band contract SHA가 광대역 v2와 다릅니다: "
            f"configured={configured_sha}, required={required_sha}"
        )
    if require_bound and declared_role != BROADBAND_CRITERION_ROLE:
        raise ValueError(
            "광대역 checkpoint/resolved config에 criterion_role binding이 없습니다"
        )
    if declared_role not in (None, BROADBAND_CRITERION_ROLE):
        raise ValueError(
            "광대역 loss schema와 criterion_role이 다릅니다: "
            f"{declared_role!r}"
        )
    configured_evidence_sha = _sha256(
        cfg.get("broadband_plant_evidence_sha256"),
        label="resolved config embedded broadband plant evidence SHA",
    )
    digital_cfg = duct.get("digital_reference")
    primary_value = (
        digital_cfg.get("primary_path_npz")
        if isinstance(digital_cfg, dict)
        else None
    )
    if not primary_value:
        raise ValueError(
            "광대역 criterion에는 duct.digital_reference.primary_path_npz가 필요합니다"
        )
    primary_path = _repo_file(
        repo_root,
        str(primary_value),
        label="criterion primary path",
    )
    primary = load_secondary_path(primary_path)
    if int(primary.sample_rate) != sample_rate:
        raise ValueError(
            "criterion P(z)/data sample rate가 다릅니다: "
            f"P={primary.sample_rate}, data={sample_rate}"
        )
    (
        evidence,
        evidence_sha,
        raw_path,
        analysis_path,
        level_path,
        plan_path,
        meter_raw_path,
        meter_receipt,
    ) = _validate_broadband_plant_npz(
        secondary_path,
        plant=secondary,
        role="secondary",
        contract=contract,
        repo_root=repo_root,
        configured_evidence_sha256=configured_evidence_sha,
    )
    (
        primary_evidence,
        primary_evidence_sha,
        primary_raw_path,
        primary_analysis_path,
        primary_level_path,
        primary_plan_path,
        primary_meter_raw_path,
        primary_meter_receipt,
    ) = _validate_broadband_plant_npz(
        primary_path,
        plant=primary,
        role="primary",
        contract=contract,
        repo_root=repo_root,
        configured_evidence_sha256=configured_evidence_sha,
    )
    if (
        primary_evidence_sha != evidence_sha
        or primary_evidence != evidence
        or primary_raw_path != raw_path
        or primary_analysis_path != analysis_path
        or primary_level_path != level_path
        or primary_plan_path != plan_path
        or primary_meter_raw_path != meter_raw_path
        or primary_meter_receipt != meter_receipt
    ):
        raise ValueError(
            "resolved broadband P/S가 같은 embedded evidence와 immutable "
            "raw/analysis/level capture를 가리키지 않습니다"
        )
    configured_handoff = handoff_samples_from_config(duct)
    if int(evidence.handoff_extra_samples) != int(configured_handoff):
        raise ValueError(
            "광대역 plant evidence handoff가 resolved duct config와 다릅니다: "
            f"evidence={evidence.handoff_extra_samples}, config={configured_handoff}"
        )
    declared_lead = data.get("digital_reference_lead_samples")
    if declared_lead is not None and int(declared_lead) != int(
        evidence.derived_lead_samples
    ):
        raise ValueError(
            "광대역 data digital-reference lead가 PlantDelays evidence와 다릅니다: "
            f"configured={declared_lead}, evidence={evidence.derived_lead_samples}"
        )
    if require_bound and declared_lead is None:
        raise ValueError(
            "광대역 resolved config에 evidence-derived digital-reference lead가 없습니다"
        )
    primary_measured = load_measured_band_path(
        primary_path,
        role="primary",
        valid_band_hz=contract.point_control_target_hz,
        subbands_hz=contract.point_control_subbands_hz,
    )
    secondary_measured = load_measured_band_path(
        secondary_path,
        role="secondary",
        valid_band_hz=contract.point_control_target_hz,
        subbands_hz=contract.point_control_subbands_hz,
    )
    if (
        primary_measured.control_band_contract_sha256 != required_sha
        or secondary_measured.control_band_contract_sha256 != required_sha
        or primary_measured.plant_evidence_sha256 != evidence_sha
        or secondary_measured.plant_evidence_sha256 != evidence_sha
    ):
        raise ValueError(
            "광대역 measured P/S response가 control/evidence SHA와 다릅니다"
        )
    measured_contract = _measured_band_contract_payload(
        primary_measured, secondary_measured
    )
    configured_measured_contract = cfg.get("broadband_measured_band_contract")
    if configured_measured_contract is not None and configured_measured_contract != (
        measured_contract
    ):
        raise ValueError(
            "resolved config measured-band criterion fingerprint가 현재 P/S와 다릅니다"
        )
    if require_bound:
        if configured_measured_contract is None:
            raise ValueError(
                "광대역 resolved config에 measured-band criterion fingerprint가 없습니다"
            )
        # 현재 Trainer는 plant prefix/state 없는 stateless random segment를 만들고
        # compact-FIR settle crop을 가정한다. measured H(f)Y(f)는 full linear history가
        # 없으면 segment 왼쪽 경계를 물리적으로 정의할 수 없으므로 학습을 막는다.
        raise ValueError(
            "광대역 학습 admission BLOCKED: stateless random segment에 measured-band "
            "linear-convolution prefix/state가 없고 synthetic d generator가 여전히 "
            "compact P FIR을 사용합니다"
        )
    return CriterionAdmission(
        role=BROADBAND_CRITERION_ROLE,
        loss_schema_version=BROADBAND_LOSS_SCHEMA_VERSION,
        control_band_contract_sha256=required_sha,
        primary_path=primary_path,
        secondary_path=secondary_path,
        broadband_plant_evidence_sha256=evidence_sha,
        broadband_source_raw_path=raw_path,
        broadband_source_analysis_path=analysis_path,
        measurement_level_evidence_path=level_path,
        broadband_source_plan_path=plan_path,
        broadband_fresh_meter_raw_path=meter_raw_path,
        broadband_fresh_meter_receipt_path=meter_receipt,
        broadband_derived_lead_samples=int(evidence.derived_lead_samples),
        secondary=secondary,
        primary_measured_band=primary_measured,
        secondary_measured_band=secondary_measured,
        measured_band_contract_sha256=str(measured_contract["sha256"]),
        target_band_hz=tuple(contract.point_control_target_hz),
        trusted_band_hz=tuple(contract.point_control_target_hz),
        band_plan=None,
    )


def bind_criterion_contract(
    cfg: dict[str, Any], *, repo_root: str | Path
) -> CriterionAdmission:
    """experiment-contract stamp 전에 광대역 role을 resolved cfg에 결속한다."""

    admission = admit_criterion_config(cfg, repo_root=repo_root, require_bound=False)
    if admission.role == BROADBAND_CRITERION_ROLE:
        cfg["criterion_role"] = BROADBAND_CRITERION_ROLE
        assert admission.broadband_derived_lead_samples is not None
        cfg["data"]["digital_reference_lead_samples"] = int(
            admission.broadband_derived_lead_samples
        )
        if admission.causal_authority is not None:
            assert admission.broadband_valid_prefix_samples is not None
            assert admission.broadband_causal_operator_contract is not None
            cfg["data"]["training_timing_contract"] = (
                admission.causal_authority.timing_contract.model_dump()
            )
            cfg["data"]["broadband_model_prefix_samples"] = int(
                admission.broadband_valid_prefix_samples
            )
            cfg["loss_start_sample"] = int(
                admission.broadband_valid_prefix_samples
            )
            cfg["broadband_causal_operator_contract"] = dict(
                admission.broadband_causal_operator_contract
            )
        else:
            assert admission.primary_measured_band is not None
            assert admission.secondary_measured_band is not None
            cfg["broadband_measured_band_contract"] = _measured_band_contract_payload(
                admission.primary_measured_band,
                admission.secondary_measured_band,
            )
        # schema/evidence/control SHA와 PlantDelays lead를 resolved cfg에 함께 남겨
        # experiment contract/checkpoint가 손실 역할과 시간축을 분리할 수 없게 한다.
        if cfg["loss"].get("schema_version") != admission.loss_schema_version:
            raise RuntimeError("광대역 loss schema binding이 admission과 달라졌습니다")
        if cfg.get("control_band_contract_sha256") != (
            admission.control_band_contract_sha256
        ):
            raise RuntimeError("광대역 control-band SHA binding이 admission과 달라졌습니다")
    return admission


def build_criterion_from_config(
    cfg: dict[str, Any],
    *,
    repo_root: str | Path,
    limiter_limit: float,
    device: torch.device | str | None,
    admission: CriterionAdmission | None = None,
) -> CriterionBundle:
    """승인된 resolved config에서 Stage-1 또는 광대역 criterion을 만든다."""

    # artifact가 preflight 뒤 바뀌는 TOCTOU를 막기 위해 construction 경계에서 다시
    # 읽고, 전달된 admission은 immutable identity와만 대조한다.
    current = admit_criterion_config(cfg, repo_root=repo_root, require_bound=True)
    if admission is not None:
        expected_identity = (
            admission.role,
            admission.loss_schema_version,
            admission.control_band_contract_sha256,
            admission.primary_path,
            admission.secondary_path,
            admission.broadband_plant_evidence_sha256,
            admission.broadband_source_raw_path,
            admission.broadband_source_analysis_path,
            admission.measurement_level_evidence_path,
            admission.broadband_source_plan_path,
            admission.broadband_fresh_meter_raw_path,
            admission.broadband_fresh_meter_receipt_path,
            admission.broadband_derived_lead_samples,
            admission.measured_band_contract_sha256,
            (
                None
                if admission.causal_authority is None
                else admission.causal_authority.authority_file_sha256
            ),
            (
                None
                if admission.causal_authority is None
                else admission.causal_authority.authority_evidence_sha256
            ),
            admission.broadband_valid_prefix_samples,
            admission.broadband_causal_operator_contract,
            admission.target_band_hz,
            admission.trusted_band_hz,
        )
        current_identity = (
            current.role,
            current.loss_schema_version,
            current.control_band_contract_sha256,
            current.primary_path,
            current.secondary_path,
            current.broadband_plant_evidence_sha256,
            current.broadband_source_raw_path,
            current.broadband_source_analysis_path,
            current.measurement_level_evidence_path,
            current.broadband_source_plan_path,
            current.broadband_fresh_meter_raw_path,
            current.broadband_fresh_meter_receipt_path,
            current.broadband_derived_lead_samples,
            current.measured_band_contract_sha256,
            (
                None
                if current.causal_authority is None
                else current.causal_authority.authority_file_sha256
            ),
            (
                None
                if current.causal_authority is None
                else current.causal_authority.authority_evidence_sha256
            ),
            current.broadband_valid_prefix_samples,
            current.broadband_causal_operator_contract,
            current.target_band_hz,
            current.trusted_band_hz,
        )
        if current_identity != expected_identity:
            raise ValueError("criterion admission이 현재 resolved config/S NPZ와 다릅니다")
    checked = current

    data = cfg["data"]
    duct = cfg["duct"]
    perturb = data.get("plant_perturbation") or {}
    if checked.role == BROADBAND_CRITERION_ROLE:
        if checked.secondary_causal is not None:
            plant: DifferentiableSecondaryPath | MeasuredBandPath | CausalFIRPath = (
                CausalFIRPath(checked.secondary_causal)
            )
        else:
            assert checked.secondary_measured_band is not None
            plant = MeasuredBandPath(
                checked.secondary_measured_band,
                extra_delay_samples=handoff_samples_from_config(duct),
            )
    else:
        plant = DifferentiableSecondaryPath(
            checked.secondary,
            handoff_extra_samples=handoff_samples_from_config(duct),
            delay_jitter_range=tuple(perturb.get("delay_jitter_range", [0, 0])),
            gain_db_range=tuple(perturb.get("gain_db", [0.0, 0.0])),
            tilt_db_per_octave_range=tuple(
                perturb.get("gain_tilt_db_per_octave", [0.0, 0.0])
            ),
            allpass_perturb=bool(perturb.get("allpass_perturb", False)),
            seed=int(cfg.get("seed", 0)) + 17,
        )
    nonlinear_cfg = data.get("nonlinear") or {}
    nonlinear = RandomNonlinear(
        nonlinear_cfg.get("sef_eta_choices", [10.0]),
        tuple(nonlinear_cfg.get("drive_range", [1.0, 1.0])),
        hardclip_prob=float(nonlinear_cfg.get("hardclip_prob", 0.0)),
        seed=int(cfg.get("seed", 0)) + 29,
    )
    if checked.role == BROADBAND_CRITERION_ROLE:
        criterion: ANCLoss = BroadbandANCLoss(
            plant,
            cfg["loss"],
            int(data["sample_rate"]),
            nonlinear=nonlinear,
            limiter_limit=float(limiter_limit),
            control_band_contract=ControlBandContract.broadband_point_control(),
        )
    else:
        criterion = ANCLoss(
            plant,
            cfg["loss"],
            int(data["sample_rate"]),
            nonlinear=nonlinear,
            cutoff_hz=float(
                (duct.get("acoustics") or {}).get("plane_wave_cutoff_hz", 1633.0)
            ),
            target_band_hz=checked.target_band_hz,
            trusted_band_hz=checked.trusted_band_hz,
            limiter_limit=float(limiter_limit),
        )
    if device is not None:
        criterion = criterion.to(device)
    return CriterionBundle(criterion=criterion, admission=checked)


__all__ = [
    "BROADBAND_CAUSAL_AUTHORITY_CONFIG_SCHEMA",
    "BROADBAND_CAUSAL_PREFIX_SCHEMA",
    "BROADBAND_CRITERION_ROLE",
    "BROADBAND_SEGMENT_BOUNDARY_SCHEMA",
    "BROADBAND_SEGMENT_BOUNDARY_STATUS",
    "BROADBAND_SYNTH_PRIMARY_STATUS",
    "CriterionAdmission",
    "CriterionBundle",
    "STAGE1_CRITERION_ROLE",
    "admit_criterion_config",
    "bind_criterion_contract",
    "build_criterion_from_config",
]
