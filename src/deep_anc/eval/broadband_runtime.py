"""2/4/8 kHz 광대역 ANC의 Jetson realtime timing 증거 계약.

평균 inference 시간만 빠른 것은 고역 위상 증거가 아니다. 한 번의 sample slip,
deadline miss, fallback silence 또는 ring drop도 8 kHz에서 큰 위상 불연속을 만들 수
있다. 이 모듈은 acoustic 성능과 분리된 runtime evidence를 fail-closed로 판정한다.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from ..dsp.control_band_contract import ControlBandContract


BROADBAND_RUNTIME_EVIDENCE_SCHEMA = "broadband_runtime_evidence_v2"
MAX_P99_INFERENCE_MS = 3.0
MIN_OBSERVED_SECONDS = 30.0
_HEX = frozenset("0123456789abcdef")


class BroadbandRuntimeEvidence(BaseModel):
    """한 고정 모델/plant/timing의 raw runtime log 요약."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["broadband_runtime_evidence_v2"] = (
        BROADBAND_RUNTIME_EVIDENCE_SCHEMA
    )
    control_band_contract_sha256: str
    experiment_contract_sha256: str
    training_timing_contract_sha256: str
    checkpoint_sha256: str
    deployment_artifact_sha256: str
    runtime_log_sha256: str
    primary_path_sha256: str
    secondary_path_sha256: str
    hardware_fingerprint_sha256: str
    model_name: str
    engine: Literal["ort", "trt"]
    power_mode: str
    sample_rate: int
    block_size: int
    handoff_extra_samples: int
    plant_lead_samples: int
    checkpoint_lead_samples: int
    deployment_lead_samples: int
    runtime_lead_samples: int
    observed_seconds: float
    callback_count: int
    inference_p50_ms: float
    inference_p95_ms: float
    inference_p99_ms: float
    inference_max_ms: float
    deadline_miss_count: int
    xrun_count: int
    input_ring_drop_samples: int
    output_ring_drop_samples: int
    ring_add_samples: int
    maximum_input_backlog_samples: int
    maximum_output_backlog_samples: int
    allowed_input_backlog_samples: int
    allowed_output_backlog_samples: int
    maximum_excess_input_backlog_samples: int
    maximum_excess_output_backlog_samples: int
    fallback_silence_blocks: int
    watchdog_trip_count: int
    sample_slip_count: int

    @model_validator(mode="after")
    def _validate_shape(self) -> "BroadbandRuntimeEvidence":
        sha_fields = (
            "control_band_contract_sha256",
            "experiment_contract_sha256",
            "training_timing_contract_sha256",
            "checkpoint_sha256",
            "deployment_artifact_sha256",
            "runtime_log_sha256",
            "primary_path_sha256",
            "secondary_path_sha256",
            "hardware_fingerprint_sha256",
        )
        for field in sha_fields:
            value = str(getattr(self, field))
            if len(value) != 64 or any(character not in _HEX for character in value):
                raise ValueError(f"{field}는 lowercase SHA-256이어야 합니다")
        if not self.model_name.strip() or not self.power_mode.strip():
            raise ValueError("runtime model_name/power_mode가 비었습니다")
        numeric_times = (
            self.observed_seconds,
            self.inference_p50_ms,
            self.inference_p95_ms,
            self.inference_p99_ms,
            self.inference_max_ms,
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in numeric_times):
            raise ValueError("runtime duration/latency가 유한한 0 이상이어야 합니다")
        counters = (
            self.callback_count,
            self.deadline_miss_count,
            self.xrun_count,
            self.input_ring_drop_samples,
            self.output_ring_drop_samples,
            self.ring_add_samples,
            self.maximum_input_backlog_samples,
            self.maximum_output_backlog_samples,
            self.allowed_input_backlog_samples,
            self.allowed_output_backlog_samples,
            self.maximum_excess_input_backlog_samples,
            self.maximum_excess_output_backlog_samples,
            self.fallback_silence_blocks,
            self.watchdog_trip_count,
            self.sample_slip_count,
        )
        if any(int(value) < 0 for value in counters):
            raise ValueError("runtime counter는 0 이상이어야 합니다")
        if (
            self.allowed_input_backlog_samples != self.handoff_extra_samples
            or self.allowed_output_backlog_samples != self.handoff_extra_samples
        ):
            raise ValueError(
                "허용 ring backlog는 runtime handoff와 exact하게 같아야 합니다"
            )
        expected_input_excess = max(
            0,
            self.maximum_input_backlog_samples
            - self.allowed_input_backlog_samples,
        )
        expected_output_excess = max(
            0,
            self.maximum_output_backlog_samples
            - self.allowed_output_backlog_samples,
        )
        if (
            self.maximum_excess_input_backlog_samples != expected_input_excess
            or self.maximum_excess_output_backlog_samples
            != expected_output_excess
        ):
            raise ValueError(
                "maximum excess backlog은 absolute maximum과 allowed backlog에서 "
                "exact하게 유도되어야 합니다"
            )
        return self


class BroadbandRuntimeAudit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["PASS", "BLOCKED"]
    reasons: tuple[str, ...]
    block_deadline_ms: float
    degrees_per_sample_8khz: float

    @property
    def ok(self) -> bool:
        return self.status == "PASS"


def audit_broadband_runtime_evidence(
    contract: ControlBandContract,
    evidence: BroadbandRuntimeEvidence,
) -> BroadbandRuntimeAudit:
    """고정 lead와 zero-slip/deadline runtime을 독립 판정한다."""

    reasons: list[str] = []
    if contract.role != "broadband_point_control":
        reasons.append("Stage-1 runtime은 광대역 runtime 증거가 아닙니다")
    if evidence.control_band_contract_sha256 != contract.digest():
        reasons.append("control-band contract SHA가 다릅니다")
    if (evidence.sample_rate, evidence.block_size) != (48_000, 256):
        reasons.append("광대역 realtime은 48kHz/256 samples여야 합니다")
    if evidence.handoff_extra_samples != 256:
        reasons.append("runtime handoff가 256 samples 계약과 다릅니다")
    leads = {
        evidence.plant_lead_samples,
        evidence.checkpoint_lead_samples,
        evidence.deployment_lead_samples,
        evidence.runtime_lead_samples,
    }
    if len(leads) != 1:
        reasons.append("plant/checkpoint/deployment/runtime lead가 exact하게 같지 않습니다")
    if evidence.observed_seconds < MIN_OBSERVED_SECONDS:
        reasons.append(f"runtime 관측 시간이 {MIN_OBSERVED_SECONDS:g}초 미만입니다")
    expected_callbacks = math.floor(
        evidence.observed_seconds * evidence.sample_rate / evidence.block_size
    )
    if evidence.callback_count < expected_callbacks:
        reasons.append("관측 시간 대비 callback count가 부족합니다")
    times = (
        evidence.inference_p50_ms,
        evidence.inference_p95_ms,
        evidence.inference_p99_ms,
        evidence.inference_max_ms,
    )
    if not all(left <= right for left, right in zip(times, times[1:])):
        reasons.append("P50/P95/P99/max latency 순서가 일관되지 않습니다")
    deadline_ms = 1000.0 * evidence.block_size / evidence.sample_rate
    if evidence.inference_p99_ms >= MAX_P99_INFERENCE_MS:
        reasons.append(f"P99 inference가 {MAX_P99_INFERENCE_MS:g}ms 미만이 아닙니다")
    if evidence.inference_max_ms >= deadline_ms:
        reasons.append("max inference가 256-sample deadline 미만이 아닙니다")
    zero_counters = {
        "deadline miss": evidence.deadline_miss_count,
        "xrun": evidence.xrun_count,
        "input ring drop": evidence.input_ring_drop_samples,
        "output ring drop": evidence.output_ring_drop_samples,
        "ring add": evidence.ring_add_samples,
        "input excess backlog": evidence.maximum_excess_input_backlog_samples,
        "output excess backlog": evidence.maximum_excess_output_backlog_samples,
        "fallback silence": evidence.fallback_silence_blocks,
        "watchdog trip": evidence.watchdog_trip_count,
        "sample slip": evidence.sample_slip_count,
    }
    nonzero = [f"{name}={value}" for name, value in zero_counters.items() if value != 0]
    if nonzero:
        reasons.append("runtime discontinuity counter가 0이 아닙니다: " + ", ".join(nonzero))
    backlog_violations = []
    if evidence.maximum_input_backlog_samples > evidence.allowed_input_backlog_samples:
        backlog_violations.append(
            "input absolute/allowed="
            f"{evidence.maximum_input_backlog_samples}/"
            f"{evidence.allowed_input_backlog_samples}"
        )
    if evidence.maximum_output_backlog_samples > evidence.allowed_output_backlog_samples:
        backlog_violations.append(
            "output absolute/allowed="
            f"{evidence.maximum_output_backlog_samples}/"
            f"{evidence.allowed_output_backlog_samples}"
        )
    if backlog_violations:
        reasons.append(
            "runtime absolute backlog이 허용 한도를 넘습니다: "
            + ", ".join(backlog_violations)
        )
    return BroadbandRuntimeAudit(
        status="PASS" if not reasons else "BLOCKED",
        reasons=tuple(reasons),
        block_deadline_ms=deadline_ms,
        degrees_per_sample_8khz=60.0,
    )


__all__ = [
    "BROADBAND_RUNTIME_EVIDENCE_SCHEMA",
    "BroadbandRuntimeAudit",
    "BroadbandRuntimeEvidence",
    "MAX_P99_INFERENCE_MS",
    "MIN_OBSERVED_SECONDS",
    "audit_broadband_runtime_evidence",
]
