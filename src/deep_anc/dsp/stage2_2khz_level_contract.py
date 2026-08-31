"""Stage-2 source/control operating level의 단일 typed 계약.

측정 analyzer의 caller가 임의로 작은 source amplitude를 넣어 actuator feasibility를
우회할 수 없게 한다. 학습·평가도 같은 payload SHA를 checkpoint 외부 계약에 포함해야
하며, decoded source는 peak-normalize 뒤 이 계약의 gain 분포와 cap을 적용해야 한다.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


SCHEMA = "stage2_2khz_operating_level_contract_v1"
PHYSICAL_EVIDENCE_SCHEMA = "stage2_2khz_physical_operating_level_evidence_v1"
CONTRACT_ID = "stage2_2khz_source49_control98_v1"
SOURCE_OPERATING_PEAK_PCM = 49
ACTUATOR_LIMIT_PEAK_PCM = 98
PCM_DENOMINATOR = 32_768
TRAINING_GAIN_DB_MIN = -6.0
TRAINING_GAIN_DB_MAX = 0.0
TARGET_ATTENUATION_DB = 3.0
MINIMUM_ACTUATOR_HEADROOM_DB = 3.0


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def canonical_stage2_operating_level_contract() -> dict[str, Any]:
    source_abs = SOURCE_OPERATING_PEAK_PCM / PCM_DENOMINATOR
    actuator_abs = ACTUATOR_LIMIT_PEAK_PCM / PCM_DENOMINATOR
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "contract_id": CONTRACT_ID,
        "status": "PLANNED_VALUES_NOT_PHYSICAL_AUTHORITY",
        "canonical_training_eligible": False,
        "sample_format": {
            "sample_rate_hz": 48_000,
            "encoding": "signed_pcm_normalized_by_32768",
            "full_scale_abs": 1.0,
            "pcm_denominator": PCM_DENOMINATOR,
        },
        "source_operating_peak_pcm": SOURCE_OPERATING_PEAK_PCM,
        "source_operating_peak_abs": source_abs,
        "source_operating_peak_dbfs": 20.0 * math.log10(source_abs),
        "actuator_limit_peak_pcm": ACTUATOR_LIMIT_PEAK_PCM,
        "actuator_limit_abs": actuator_abs,
        "actuator_limit_dbfs": 20.0 * math.log10(actuator_abs),
        "augmentation_gain_db": {
            "distribution": "stateless_uniform_by_global_sample_index",
            "minimum": TRAINING_GAIN_DB_MIN,
            "maximum": TRAINING_GAIN_DB_MAX,
            "per_sample_peak_normalization_before_gain_required": True,
            "post_gain_hard_peak_cap_abs": source_abs,
        },
        "feasibility_policy": {
            "target_attenuation_db": TARGET_ATTENUATION_DB,
            "minimum_additional_headroom_db": MINIMUM_ACTUATOR_HEADROOM_DB,
            "per_frequency_ps_ratio_is_necessary_not_sufficient": True,
            "broadband_time_domain_peak_gate_required": True,
            "training_or_evaluation_performance_claimed": False,
        },
        "physical_authority_requires": {
            "diagnostic_phase_raw_sha256": True,
            "diagnostic_analysis_receipt_sha256": True,
            "meter_raw_and_receipt_sha256": True,
            "ps_phase_raw_boundary_and_sha256": True,
            "no_replace_evidence_artifact": True,
        },
    }
    payload["canonical_payload_sha256"] = _digest(payload)
    return payload


def validate_stage2_operating_level_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = canonical_stage2_operating_level_contract()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError("Stage-2 operating level contract payload/SHA가 canonical과 다릅니다")
    return expected


def _artifact_ref(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} artifact ref key가 exact하지 않습니다")
    path, digest = value.get("path"), value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} artifact ref가 canonical repository path/SHA가 아닙니다")
    return {"path": path, "sha256": digest}


def build_stage2_physical_operating_level_evidence(
    *,
    signal_plan_sha256: str,
    capture_id: str,
    hardware_identity: Mapping[str, Any],
    meter_raw_artifact: Mapping[str, Any],
    meter_receipt_artifact: Mapping[str, Any],
    calibration_evidence_artifact: Mapping[str, Any],
    diagnostic_raw_artifact: Mapping[str, Any],
    diagnostic_authorization_artifact: Mapping[str, Any],
    ps_raw_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """검증된 physical bytes가 가리키는 고정 49/98 PCM level evidence를 만든다.

    이 함수 자체는 파일을 열지 않는 pure serializer다. repository-aware publisher/loader가
    모든 ref를 다시 열어 SHA와 diagnostic authorization chain을 검증해야만 이 payload를
    analyzer에 넘길 수 있다. 숫자는 인자로 받지 않고 tracked canonical 계약에서만 유도한다.
    """

    if (
        not isinstance(signal_plan_sha256, str)
        or len(signal_plan_sha256) != 64
        or any(character not in "0123456789abcdef" for character in signal_plan_sha256)
        or not isinstance(capture_id, str)
        or not capture_id
        or not isinstance(hardware_identity, Mapping)
        or not hardware_identity
    ):
        raise ValueError("Stage-2 physical level plan/capture/hardware identity가 유효하지 않습니다")
    planned = canonical_stage2_operating_level_contract()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_EVIDENCE_SCHEMA,
        "contract_id": planned["contract_id"],
        "status": "PHYSICAL_DIAGNOSTIC_AND_PS_RAW_BOUND_ANALYSIS_REQUIRED",
        "canonical_training_eligible": False,
        "signal_plan_sha256": signal_plan_sha256,
        "planned_level_contract_sha256": planned["canonical_payload_sha256"],
        "capture_id": capture_id,
        "hardware_identity": dict(hardware_identity),
        "sample_format": dict(planned["sample_format"]),
        "source_operating_peak_pcm": planned["source_operating_peak_pcm"],
        "source_operating_peak_abs": planned["source_operating_peak_abs"],
        "actuator_limit_peak_pcm": planned["actuator_limit_peak_pcm"],
        "actuator_limit_abs": planned["actuator_limit_abs"],
        "augmentation_gain_db": dict(planned["augmentation_gain_db"]),
        "feasibility_policy": dict(planned["feasibility_policy"]),
        "artifact_lineage": {
            "meter_raw": _artifact_ref(meter_raw_artifact, label="meter raw"),
            "meter_receipt": _artifact_ref(
                meter_receipt_artifact, label="meter receipt"
            ),
            "calibration_evidence": _artifact_ref(
                calibration_evidence_artifact, label="calibration evidence"
            ),
            "diagnostic_phase_raw": _artifact_ref(
                diagnostic_raw_artifact, label="diagnostic raw"
            ),
            "diagnostic_authorization": _artifact_ref(
                diagnostic_authorization_artifact,
                label="diagnostic authorization",
            ),
            "ps_phase_raw": _artifact_ref(ps_raw_artifact, label="PS raw"),
        },
        "authority_scope": {
            "physical_49_98_pcm_linearity_and_route": True,
            "actuator_feasibility_analysis_completed": False,
            "plant_binding_published": False,
            "training_or_evaluation_authority": False,
        },
    }
    payload["canonical_payload_sha256"] = _digest(payload)
    return payload


def validate_stage2_physical_operating_level_evidence(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Stage-2 physical operating level evidence가 mapping이 아닙니다")
    digest = value.get("canonical_payload_sha256")
    unsigned = {key: item for key, item in value.items() if key != "canonical_payload_sha256"}
    if digest != _digest(unsigned):
        raise ValueError("Stage-2 physical operating level evidence SHA가 다릅니다")
    lineage = value.get("artifact_lineage")
    expected_keys = {
        "meter_raw",
        "meter_receipt",
        "calibration_evidence",
        "diagnostic_phase_raw",
        "diagnostic_authorization",
        "ps_phase_raw",
    }
    if not isinstance(lineage, Mapping) or set(lineage) != expected_keys:
        raise ValueError("Stage-2 physical operating level lineage가 exact하지 않습니다")
    rebuilt = build_stage2_physical_operating_level_evidence(
        signal_plan_sha256=value.get("signal_plan_sha256"),
        capture_id=value.get("capture_id"),
        hardware_identity=value.get("hardware_identity"),
        meter_raw_artifact=lineage["meter_raw"],
        meter_receipt_artifact=lineage["meter_receipt"],
        calibration_evidence_artifact=lineage["calibration_evidence"],
        diagnostic_raw_artifact=lineage["diagnostic_phase_raw"],
        diagnostic_authorization_artifact=lineage["diagnostic_authorization"],
        ps_raw_artifact=lineage["ps_phase_raw"],
    )
    if dict(value) != rebuilt:
        raise ValueError("Stage-2 physical operating level evidence 의미/상수가 다릅니다")
    return rebuilt


__all__ = [
    "ACTUATOR_LIMIT_PEAK_PCM",
    "CONTRACT_ID",
    "MINIMUM_ACTUATOR_HEADROOM_DB",
    "PHYSICAL_EVIDENCE_SCHEMA",
    "SCHEMA",
    "SOURCE_OPERATING_PEAK_PCM",
    "TARGET_ATTENUATION_DB",
    "TRAINING_GAIN_DB_MAX",
    "TRAINING_GAIN_DB_MIN",
    "canonical_stage2_operating_level_contract",
    "build_stage2_physical_operating_level_evidence",
    "validate_stage2_physical_operating_level_evidence",
    "validate_stage2_operating_level_contract",
]
