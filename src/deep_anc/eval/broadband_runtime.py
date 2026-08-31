"""2/4/8 kHz 광대역 ANC의 Jetson realtime timing 증거 계약.

평균 inference 시간만 빠른 것은 고역 위상 증거가 아니다. 한 번의 sample slip,
deadline miss, fallback silence 또는 ring drop도 8 kHz에서 큰 위상 불연속을 만들 수
있다. 이 모듈은 acoustic 성능과 분리된 runtime evidence를 fail-closed로 판정한다.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from pydantic import BaseModel, ConfigDict, model_validator

from ..dsp.control_band_contract import ControlBandContract
from ..dsp.timing import TrainingTimingContract
from ..realtime.clock_telemetry import payload_sha256
from ..realtime.engines import checkpoint_digital_reference_lead_samples
from ..realtime.physical_clock_witness import (
    audit_runtime_physical_clock_witness,
)
from ..realtime.plant_contract import (
    RuntimePlantContract,
    validate_runtime_plant_contract,
)
from ..train.experiment_contract import validate_embedded_experiment_contract


BROADBAND_RUNTIME_EVIDENCE_SCHEMA = "broadband_runtime_evidence_v2"
RUNTIME_DEPLOYMENT_METADATA_SCHEMA = "broadband_runtime_deployment_metadata_v1"
MAX_P99_INFERENCE_MS = 3.0
MIN_OBSERVED_SECONDS = 30.0
_HEX = frozenset("0123456789abcdef")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_bytes(raw: bytes, *, path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} JSON을 읽을 수 없습니다: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON 최상위는 mapping이어야 합니다")
    return value


def _path(value: str | Path, *, root: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _authority_path(value: str | Path, *, root: Path) -> Path:
    """Authority file의 lexical absolute path를 보존한다.

    ``resolve()``로 final symlink를 먼저 따라가면 config가 가리킨 directory entry를
    audit 중 교체해도 알아낼 수 없다. held-FD 검증 대상은 원래 entry여야 한다.
    """

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return Path(os.path.abspath(candidate))


def _file_signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _assert_held_regular_path(
    *,
    path: Path,
    held: os.stat_result,
    initial: os.stat_result | None,
    label: str,
) -> None:
    if not stat.S_ISREG(held.st_mode):
        raise ValueError(f"{label}는 regular file이어야 합니다: {path}")
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} path identity를 확인할 수 없습니다: {path}") from exc
    if not stat.S_ISREG(current.st_mode):
        raise ValueError(f"{label}는 symlink가 아닌 regular file이어야 합니다: {path}")
    if (int(current.st_dev), int(current.st_ino)) != (
        int(held.st_dev),
        int(held.st_ino),
    ):
        raise ValueError(f"{label} path가 audit 중 교체됐습니다: {path}")
    if initial is not None and _file_signature(held) != _file_signature(initial):
        raise ValueError(f"{label} bytes가 audit 중 변경됐습니다: {path}")


def _open_held_regular(path: Path, *, label: str):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError(
            f"{label}는 symlink가 아닌 readable regular file이어야 합니다: {path}"
        ) from exc
    handle = os.fdopen(descriptor, "rb")
    initial = os.fstat(handle.fileno())
    try:
        _assert_held_regular_path(
            path=path, held=initial, initial=None, label=label
        )
    except BaseException:
        handle.close()
        raise
    return handle, initial


def _sha256_handle(handle) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _stable_file_snapshot(path: Path, *, label: str) -> dict[str, int | str]:
    handle, initial = _open_held_regular(path, label=label)
    with handle:
        digest = _sha256_handle(handle)
        final = os.fstat(handle.fileno())
        _assert_held_regular_path(
            path=path, held=final, initial=initial, label=label
        )
    return {
        "device": int(initial.st_dev),
        "inode": int(initial.st_ino),
        "size": int(initial.st_size),
        "sha256": digest,
    }


def _stable_file_bytes(
    path: Path, *, label: str
) -> tuple[bytes, dict[str, int | str]]:
    handle, initial = _open_held_regular(path, label=label)
    with handle:
        value = handle.read()
        final = os.fstat(handle.fileno())
        _assert_held_regular_path(
            path=path, held=final, initial=initial, label=label
        )
    return value, {
        "device": int(initial.st_dev),
        "inode": int(initial.st_ino),
        "size": int(initial.st_size),
        "sha256": hashlib.sha256(value).hexdigest(),
    }


def _require_same_snapshot(
    before: Mapping[str, int | str],
    after: Mapping[str, int | str],
    *,
    label: str,
) -> None:
    if dict(before) != dict(after):
        raise ValueError(f"{label} path/bytes가 audit 중 교체됐습니다")


def _load_stable_checkpoint(
    path: Path,
) -> tuple[dict[str, Any], dict[str, int | str]]:
    handle, initial = _open_held_regular(path, label="runtime checkpoint")
    try:
        import torch

        with handle:
            digest_before = _sha256_handle(handle)
            handle.seek(0)
            state = torch.load(handle, map_location="cpu", weights_only=False)
            digest_after = _sha256_handle(handle)
            final = os.fstat(handle.fileno())
            _assert_held_regular_path(
                path=path,
                held=final,
                initial=initial,
                label="runtime checkpoint",
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"runtime checkpoint를 읽을 수 없습니다: {path}") from exc
    if digest_before != digest_after:
        raise ValueError("runtime checkpoint bytes가 load 중 변경됐습니다")
    if not isinstance(state, dict):
        raise ValueError("runtime checkpoint 최상위는 mapping이어야 합니다")
    return state, {
        "device": int(initial.st_dev),
        "inode": int(initial.st_ino),
        "size": int(initial.st_size),
        "sha256": digest_before,
    }


def _require_sha(value: Any, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in _HEX for character in digest):
        raise ValueError(f"{label}는 lowercase SHA-256이어야 합니다")
    return digest


class RuntimeDeploymentIdentity(BaseModel):
    """실제 checkpoint/export/strict plant bytes에서 다시 계산한 배포 identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["broadband_runtime_deployment_metadata_v1"] = (
        RUNTIME_DEPLOYMENT_METADATA_SCHEMA
    )
    model_name: str
    engine: Literal["ort", "trt"]
    experiment_contract_sha256: str
    control_band_contract_sha256: str
    training_timing_contract_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_lead_samples: int
    deployment_artifact_path: str
    deployment_artifact_sha256: str
    deployment_metadata_path: str
    deployment_metadata_sha256: str
    deployment_lead_samples: int
    primary_path_sha256: str
    secondary_path_sha256: str


def snapshot_runtime_deployment_files(
    *,
    runtime_cfg: Mapping[str, Any],
    plant: RuntimePlantContract,
    repo_root: str | Path,
) -> dict[str, Any]:
    """녹음 시작/종료에 비교할 DL runtime file bytes snapshot."""

    root = Path(repo_root).resolve()
    if str(runtime_cfg.get("controller", "dl")) != "dl":
        raise ValueError("broadband DL deployment snapshot은 controller=dl만 허용합니다")
    engine_cfg = runtime_cfg.get("engine")
    if not isinstance(engine_cfg, Mapping):
        raise ValueError("runtime engine config mapping이 필요합니다")
    engine = str(engine_cfg.get("type", ""))
    if engine not in {"ort", "trt"}:
        raise ValueError("broadband deployment snapshot은 ORT/TRT만 허용합니다")
    checkpoint = _authority_path(str(engine_cfg.get("ckpt", "")), root=root)
    artifact = _authority_path(
        str(engine_cfg.get("onnx" if engine == "ort" else "plan", "")),
        root=root,
    )
    if engine == "ort":
        metadata = artifact.with_suffix(".json")
    else:
        metadata = (
            _authority_path(str(engine_cfg["onnx_meta"]), root=root)
            if engine_cfg.get("onnx_meta")
            else artifact.with_suffix(".json")
        )
    files = {
        "checkpoint": checkpoint,
        "deployment_artifact": artifact,
        "deployment_metadata": metadata,
    }
    snapshot: dict[str, Any] = {
        "schema_version": "runtime_deployment_file_snapshot_v1",
        "engine": engine,
        "runtime_lead_samples": int(runtime_cfg.get("digital_reference_lead_samples", -1)),
        "plant_lead_samples": int(plant.timing.digital_reference_lead_samples),
        "primary_path_sha256": plant.primary_path_sha256,
        "secondary_path_sha256": plant.secondary_path_sha256,
        "files": {},
    }
    held_snapshots: dict[str, dict[str, int | str]] = {}
    for label, path in files.items():
        held_snapshots[label] = _stable_file_snapshot(
            path, label=f"runtime {label}"
        )
        snapshot["files"][label] = {
            "path": str(path),
            "size": int(held_snapshots[label]["size"]),
            "sha256": str(held_snapshots[label]["sha256"]),
        }
    # 세 leaf를 순서대로 읽는 동안 앞서 읽은 path가 바뀌는 것도 거부한다.
    for label, path in files.items():
        _require_same_snapshot(
            held_snapshots[label],
            _stable_file_snapshot(path, label=f"runtime {label}"),
            label=f"runtime {label}",
        )
    snapshot["snapshot_sha256"] = hashlib.sha256(
        _canonical_json(snapshot).encode("utf-8")
    ).hexdigest()
    return snapshot


def verify_runtime_deployment_identity(
    *,
    contract: ControlBandContract,
    runtime_cfg: Mapping[str, Any],
    plant: RuntimePlantContract,
    repo_root: str | Path,
) -> RuntimeDeploymentIdentity:
    """checkpoint, export sidecar, engine bytes를 열어 self-attestation을 제거한다.

    sidecar의 SHA 문자열만 믿지 않는다. raw checkpoint의 embedded experiment/timing
    contract를 재검산하고, 실제 checkpoint/export/P/S file SHA와 sidecar를
    모두 대조한다.
    """

    root = Path(repo_root).resolve()
    engine_cfg = runtime_cfg.get("engine")
    if not isinstance(engine_cfg, Mapping):
        raise ValueError("runtime engine config mapping이 필요합니다")
    engine = str(engine_cfg.get("type", ""))
    if engine not in {"ort", "trt"}:
        raise ValueError("canonical broadband runtime evidence는 ORT/TRT만 허용합니다")
    checkpoint = _authority_path(str(engine_cfg.get("ckpt", "")), root=root)
    artifact_key = "onnx" if engine == "ort" else "plan"
    artifact = _authority_path(str(engine_cfg.get(artifact_key, "")), root=root)
    if engine == "ort":
        metadata = artifact.with_suffix(".json")
    else:
        metadata_value = engine_cfg.get("onnx_meta")
        metadata = (
            _authority_path(str(metadata_value), root=root)
            if metadata_value
            else artifact.with_suffix(".json")
        )
    for label, path in (
        ("checkpoint", checkpoint),
        ("deployment artifact", artifact),
        ("deployment metadata", metadata),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"runtime {label}가 없습니다: {path}")

    state, checkpoint_snapshot = _load_stable_checkpoint(checkpoint)
    if not isinstance(state, dict) or not isinstance(state.get("cfg"), dict):
        raise ValueError("runtime checkpoint에 resolved cfg가 없습니다")
    checkpoint_cfg = state["cfg"]
    embedded_contract = validate_embedded_experiment_contract(checkpoint_cfg)
    experiment_sha = _require_sha(
        embedded_contract.get("sha256"), label="checkpoint experiment contract SHA"
    )
    timing = TrainingTimingContract.from_data_config(checkpoint_cfg.get("data") or {})
    timing_sha = timing.digest()
    control_sha = _require_sha(
        checkpoint_cfg.get("control_band_contract_sha256"),
        label="checkpoint control-band contract SHA",
    )
    if control_sha != contract.digest():
        raise ValueError("checkpoint control-band contract가 현재 runtime 계약과 다릅니다")
    checkpoint_lead = int(checkpoint_digital_reference_lead_samples(state))
    if checkpoint_lead != int(timing.digital_reference_lead_samples):
        raise ValueError("checkpoint lead가 embedded training timing contract와 다릅니다")
    model_cfg = checkpoint_cfg.get("model") or {}
    model_name = str(model_cfg.get("name", "")).strip()
    if not model_name:
        raise ValueError("checkpoint model.name이 비었습니다")

    metadata_raw, metadata_snapshot = _stable_file_bytes(
        metadata, label="deployment metadata"
    )
    try:
        metadata_payload = json.loads(metadata_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"deployment metadata JSON을 읽을 수 없습니다: {metadata}"
        ) from exc
    if not isinstance(metadata_payload, dict):
        raise ValueError("deployment metadata JSON 최상위는 mapping이어야 합니다")
    required = {
        "schema_version",
        "model_name",
        "engine",
        "experiment_contract_sha256",
        "control_band_contract_sha256",
        "training_timing_contract_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "digital_reference_lead_samples",
        "deployment_artifact_path",
        "deployment_artifact_sha256",
        "primary_path_sha256",
        "secondary_path_sha256",
    }
    missing = sorted(required - set(metadata_payload))
    if missing:
        raise ValueError(f"deployment metadata 필수 필드가 없습니다: {missing}")
    if metadata_payload.get("schema_version") != RUNTIME_DEPLOYMENT_METADATA_SCHEMA:
        raise ValueError("deployment metadata schema가 canonical runtime 계약과 다릅니다")

    artifact_snapshot = _stable_file_snapshot(
        artifact, label="deployment artifact"
    )
    checkpoint_sha = str(checkpoint_snapshot["sha256"])
    artifact_sha = str(artifact_snapshot["sha256"])
    metadata_sha = str(metadata_snapshot["sha256"])
    expected = {
        "model_name": model_name,
        "engine": engine,
        "experiment_contract_sha256": experiment_sha,
        "control_band_contract_sha256": control_sha,
        "training_timing_contract_sha256": timing_sha,
        "checkpoint_sha256": checkpoint_sha,
        "digital_reference_lead_samples": checkpoint_lead,
        "deployment_artifact_sha256": artifact_sha,
        "primary_path_sha256": plant.primary_path_sha256,
        "secondary_path_sha256": plant.secondary_path_sha256,
    }
    for key, value in expected.items():
        if metadata_payload.get(key) != value:
            raise ValueError(
                f"deployment metadata {key}가 실제 checkpoint/export/plant와 다릅니다"
            )
    if _authority_path(metadata_payload["checkpoint_path"], root=root) != checkpoint:
        raise ValueError("deployment metadata checkpoint path가 runtime과 다릅니다")
    if _authority_path(
        metadata_payload["deployment_artifact_path"], root=root
    ) != artifact:
        raise ValueError("deployment metadata artifact path가 runtime과 다릅니다")

    # checkpoint load와 sidecar parse가 끝날 때까지 세 실제 leaf가 그대로였는지
    # 다시 연다. 같은 path에 같은 크기의 다른 inode를 바꿔 끼우는 것도 거부한다.
    for label, path, before in (
        ("runtime checkpoint", checkpoint, checkpoint_snapshot),
        ("deployment artifact", artifact, artifact_snapshot),
        ("deployment metadata", metadata, metadata_snapshot),
    ):
        _require_same_snapshot(
            before,
            _stable_file_snapshot(path, label=label),
            label=label,
        )

    return RuntimeDeploymentIdentity(
        model_name=model_name,
        engine=engine,
        experiment_contract_sha256=experiment_sha,
        control_band_contract_sha256=control_sha,
        training_timing_contract_sha256=timing_sha,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=checkpoint_sha,
        checkpoint_lead_samples=checkpoint_lead,
        deployment_artifact_path=str(artifact),
        deployment_artifact_sha256=artifact_sha,
        deployment_metadata_path=str(metadata),
        deployment_metadata_sha256=metadata_sha,
        deployment_lead_samples=int(metadata_payload["digital_reference_lead_samples"]),
        primary_path_sha256=plant.primary_path_sha256,
        secondary_path_sha256=plant.secondary_path_sha256,
    )


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
    deployment_metadata_sha256: str
    runtime_session_sha256: str
    runtime_log_sha256: str
    primary_path_sha256: str
    secondary_path_sha256: str
    hardware_fingerprint_sha256: str
    runtime_physical_witness_file_sha256: str
    runtime_physical_witness_evidence_sha256: str
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
    engine_error_blocks: int
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
    conditional_physical_timing_pass: bool
    independent_clock_authority_pass: bool

    @model_validator(mode="after")
    def _validate_shape(self) -> "BroadbandRuntimeEvidence":
        sha_fields = (
            "control_band_contract_sha256",
            "experiment_contract_sha256",
            "training_timing_contract_sha256",
            "checkpoint_sha256",
            "deployment_artifact_sha256",
            "deployment_metadata_sha256",
            "runtime_session_sha256",
            "runtime_log_sha256",
            "primary_path_sha256",
            "secondary_path_sha256",
            "hardware_fingerprint_sha256",
            "runtime_physical_witness_file_sha256",
            "runtime_physical_witness_evidence_sha256",
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
            self.engine_error_blocks,
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
    *,
    expected_plant_lead_samples: int,
) -> BroadbandRuntimeAudit:
    """실측 plant lead와 zero-slip/deadline runtime을 독립 판정한다.

    ``expected_plant_lead_samples``는 runtime evidence가 스스로 선언한 숫자가
    아니라, 감사자가 strict P/S의 ``PlantDelays.lead()``에서 유도해 넘겨야
    한다. 그렇지 않으면 plant/checkpoint/deployment/runtime이 모두 같은
    잘못된 값을 적어도 PASS하는 self-attestation이 된다.
    """

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
    expected_lead = int(expected_plant_lead_samples)
    if expected_lead < 0:
        raise ValueError("expected strict plant lead는 0 이상이어야 합니다")
    if any(value != expected_lead for value in leads):
        reasons.append(
            "plant/checkpoint/deployment/runtime lead가 strict P/S 유도값과 "
            f"다릅니다: expected={expected_lead}, observed={sorted(leads)}"
        )
    if evidence.observed_seconds < MIN_OBSERVED_SECONDS:
        reasons.append(f"runtime 관측 시간이 {MIN_OBSERVED_SECONDS:g}초 미만입니다")
    if not evidence.conditional_physical_timing_pass:
        reasons.append("acoustic continuous-pilot physical timing witness가 PASS가 아닙니다")
    if not evidence.independent_clock_authority_pass:
        reasons.append(
            "callback 전 silent period drop 0건을 증명할 독립 electrical clock "
            "authority가 없습니다"
        )
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
        "engine error block": evidence.engine_error_blocks,
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


def _npz_scalar(archive: Any, key: str) -> Any:
    if key not in archive.files:
        raise ValueError(f"runtime session NPZ에 {key}가 없습니다")
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"runtime session {key}는 scalar여야 합니다")
    item = value.reshape(-1)[0]
    return item.item() if hasattr(item, "item") else item


def _verified_runtime_fingerprint(raw_json: str, declared_sha: str) -> str:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError("runtime physical fingerprint JSON이 손상됐습니다") from exc
    if not isinstance(payload, dict) or payload.get("schema") != (
        "alsa_physical_hardware_fingerprint_v1"
    ):
        raise ValueError("runtime ALSA physical fingerprint schema가 다릅니다")
    if not isinstance(payload.get("input"), dict) or not isinstance(
        payload.get("output"), dict
    ):
        raise ValueError("runtime ALSA physical fingerprint endpoint가 누락됐습니다")
    embedded = _require_sha(payload.get("sha256"), label="runtime hardware fingerprint")
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    actual = hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
    if embedded != actual or declared_sha != actual:
        raise ValueError("runtime ALSA physical fingerprint SHA가 내용과 다릅니다")
    return actual


def build_broadband_runtime_evidence_from_artifacts(
    *,
    contract: ControlBandContract,
    runtime_cfg: Mapping[str, Any],
    session_npz_path: str | Path,
    clock_receipt_path: str | Path,
    physical_witness_receipt_path: str | Path,
    power_mode: str,
    repo_root: str | Path,
) -> BroadbandRuntimeEvidence:
    """실제 runtime NPZ/clock raw/checkpoint/export/P/S에서 evidence를 재구성한다.

    이 함수가 canonical producer다. 저장된 percentile은 받지 않고 모든
    ``inference_step_times_ms`` raw에서 다시 계산한다. clock receipt가 결속한
    session bytes, stream 전/후 deployment snapshot, ALSA 물리 fingerprint, strict
    ``PlantDelays.lead()``를 한 번에 대조한다.
    """

    mode = str(power_mode).strip()
    if not mode:
        raise ValueError("runtime power_mode가 비었습니다")
    root = Path(repo_root).resolve()
    # Authority leaf를 resolve()/Path.read_*()/sha256_file()/np.load(path)로
    # 반복 재개방하면, rename race 중 clock이 결속한 session과
    # 실제로 분석한 session이 달라질 수 있다. 세 leaf를 모두
    # O_NOFOLLOW regular-file bytes로 고정하고 이후 분석은 그 bytes만 쓴다.
    session = _authority_path(session_npz_path, root=root)
    clock = _authority_path(clock_receipt_path, root=root)
    physical_witness = _authority_path(physical_witness_receipt_path, root=root)
    try:
        session_raw, session_snapshot = _stable_file_bytes(
            session, label="runtime session NPZ"
        )
        clock_raw, clock_snapshot = _stable_file_bytes(
            clock, label="runtime clock receipt"
        )
        physical_raw, physical_snapshot = _stable_file_bytes(
            physical_witness, label="runtime physical clock witness receipt"
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "runtime session NPZ/clock receipt/physical witness receipt가 모두 필요합니다"
        ) from exc
    session_sha = str(session_snapshot["sha256"])
    clock_file_sha = str(clock_snapshot["sha256"])
    physical_file_sha = str(physical_snapshot["sha256"])

    plant = validate_runtime_plant_contract(dict(runtime_cfg))
    if plant is None:
        raise ValueError("broadband runtime evidence는 strict digital-reference DL plant가 필요합니다")
    identity = verify_runtime_deployment_identity(
        contract=contract,
        runtime_cfg=runtime_cfg,
        plant=plant,
        repo_root=root,
    )
    current_snapshot = snapshot_runtime_deployment_files(
        runtime_cfg=runtime_cfg,
        plant=plant,
        repo_root=root,
    )

    bundle = _json_bytes(clock_raw, path=clock, label="runtime clock receipt")
    if bundle.get("schema_version") != "realtime_clock_receipt_bundle_v1":
        raise ValueError("runtime clock receipt bundle schema가 다릅니다")
    if bundle.get("authority_status") != "INCONCLUSIVE":
        raise ValueError("runtime clock receipt가 structural PASS/INCONCLUSIVE가 아닙니다")
    if _path(str(bundle.get("recording_npz", "")), root=root) != session:
        raise ValueError("clock receipt가 다른 runtime session을 가리킵니다")
    if _require_sha(
        bundle.get("recording_npz_sha256"), label="clock-bound session SHA"
    ) != session_sha:
        raise ValueError("clock receipt가 결속한 runtime session SHA가 다릅니다")
    telemetry = bundle.get("runtime_clock_telemetry")
    if not isinstance(telemetry, dict):
        raise ValueError("runtime clock telemetry payload가 누락됐습니다")
    telemetry_sha = payload_sha256(telemetry)
    if _require_sha(
        bundle.get("runtime_clock_telemetry_sha256"),
        label="runtime clock telemetry SHA",
    ) != telemetry_sha:
        raise ValueError("runtime clock telemetry payload SHA가 다릅니다")
    if (
        telemetry.get("schema_version") != "realtime_clock_telemetry_v1"
        or telemetry.get("authority_status") != "INCONCLUSIVE"
        or telemetry.get("structural_status") != "PASS"
    ):
        raise ValueError("runtime clock telemetry가 structural PASS/INCONCLUSIVE가 아닙니다")

    summary = telemetry.get("callback_summary")
    counters = telemetry.get("runtime_counters_final")
    callbacks = telemetry.get("callbacks")
    if not isinstance(summary, dict) or not isinstance(counters, dict):
        raise ValueError("runtime callback summary/counter가 누락됐습니다")
    if not isinstance(callbacks, list) or not callbacks:
        raise ValueError("runtime callback raw row가 누락됐습니다")
    callback_count = int(summary.get("callback_count", -1))
    if (
        callback_count != len(callbacks)
        or int(summary.get("completed_callback_count", -1)) != callback_count
        or int(summary.get("stored_callback_record_count", -1)) != callback_count
        or any(row.get("completed") is not True for row in callbacks)
    ):
        raise ValueError("runtime callback raw가 전수 완료·저장되지 않았습니다")
    for field in (
        "incomplete_callback_count",
        "pending_callback_count",
        "portaudio_status_callback_count",
        "callback_host_deadline_miss_count",
        "omitted_callback_record_count",
    ):
        if int(summary.get(field, -1)) != 0:
            raise ValueError(f"runtime callback {field}가 0이 아닙니다")
    if telemetry.get("issue_counts") not in ({}, None):
        raise ValueError("runtime clock telemetry issue_counts가 비어 있지 않습니다")

    with np.load(io.BytesIO(session_raw), allow_pickle=False) as archive:
        if str(_npz_scalar(archive, "runtime_clock_telemetry_sha256")) != telemetry_sha:
            raise ValueError("runtime NPZ embedded clock telemetry SHA가 다릅니다")
        if str(_npz_scalar(archive, "runtime_clock_authority_status")) != "INCONCLUSIVE":
            raise ValueError("runtime NPZ embedded clock authority가 INCONCLUSIVE가 아닙니다")
        sample_rate = int(_npz_scalar(archive, "fs"))
        times = np.asarray(archive["inference_step_times_ms"], dtype=np.float64)
        inference_count = int(_npz_scalar(archive, "inference_step_count"))
        prime_blocks = int(_npz_scalar(archive, "intentional_startup_prime_blocks"))
        snapshot_json = str(_npz_scalar(archive, "runtime_deployment_snapshot_json"))
        snapshot_sha = str(
            _npz_scalar(archive, "runtime_deployment_snapshot_sha256")
        )
        fingerprint_json = str(
            _npz_scalar(archive, "runtime_physical_fingerprint_json")
        )
        fingerprint_sha = str(
            _npz_scalar(archive, "runtime_physical_fingerprint_sha256")
        )
    if times.ndim != 1 or times.size == 0 or np.any(~np.isfinite(times)) or np.any(times < 0):
        raise ValueError("runtime inference latency raw가 유한한 1-D 양수가 아닙니다")
    if inference_count != int(times.size):
        raise ValueError("runtime inference step count와 latency raw 수가 다릅니다")
    if prime_blocks != 1:
        raise ValueError("runtime startup output handoff prime은 exact 1 block이어야 합니다")
    try:
        embedded_snapshot = json.loads(snapshot_json)
    except json.JSONDecodeError as exc:
        raise ValueError("runtime deployment snapshot JSON이 손상됐습니다") from exc
    if not isinstance(embedded_snapshot, dict):
        raise ValueError("runtime deployment snapshot은 mapping이어야 합니다")
    unsigned_snapshot = {
        key: value
        for key, value in embedded_snapshot.items()
        if key != "snapshot_sha256"
    }
    actual_snapshot_sha = hashlib.sha256(
        _canonical_json(unsigned_snapshot).encode("utf-8")
    ).hexdigest()
    if (
        _require_sha(snapshot_sha, label="runtime deployment snapshot SHA")
        != actual_snapshot_sha
        or embedded_snapshot.get("snapshot_sha256") != actual_snapshot_sha
        or embedded_snapshot != current_snapshot
    ):
        raise ValueError("runtime deployment start/end/current snapshot이 다릅니다")
    hardware_sha = _verified_runtime_fingerprint(fingerprint_json, fingerprint_sha)

    physical_bundle = _json_bytes(
        physical_raw,
        path=physical_witness,
        label="runtime physical clock witness receipt",
    )
    if physical_bundle.get("schema_version") != (
        "runtime_physical_clock_witness_bundle_v1"
    ):
        raise ValueError("runtime physical witness bundle schema가 다릅니다")
    physical_evidence = physical_bundle.get("evidence")
    if not isinstance(physical_evidence, dict):
        raise ValueError("runtime physical witness embedded evidence가 누락됐습니다")
    if physical_evidence.get("schema_version") != (
        "runtime_physical_clock_witness_v1"
    ):
        raise ValueError("runtime physical witness evidence schema가 다릅니다")
    physical_evidence_sha = _require_sha(
        physical_bundle.get("evidence_sha256"), label="runtime physical evidence SHA"
    )
    unsigned_physical = {
        key: value
        for key, value in physical_evidence.items()
        if key != "evidence_sha256"
    }
    recomputed_physical_sha = hashlib.sha256(
        (_canonical_json(unsigned_physical) + "\n").encode("utf-8")
    ).hexdigest()
    if (
        physical_evidence.get("evidence_sha256") != physical_evidence_sha
        or recomputed_physical_sha != physical_evidence_sha
    ):
        raise ValueError("runtime physical witness evidence SHA가 내용과 다릅니다")
    if physical_bundle.get("status") != physical_evidence.get("status"):
        raise ValueError("runtime physical witness bundle/evidence status가 다릅니다")
    if bool(physical_evidence.get("synthetic_fixture")):
        raise ValueError("synthetic runtime physical witness는 실기 latency 증거가 아닙니다")
    if physical_evidence.get("status") != "CONDITIONAL_PASS":
        raise ValueError("runtime physical witness가 CONDITIONAL_PASS가 아닙니다")
    if physical_evidence.get("conditional_physical_timing_pass") is not True:
        raise ValueError("runtime physical timing conditional pass가 false입니다")
    if _path(str(physical_evidence.get("session_npz_path", "")), root=root) != session:
        raise ValueError("runtime physical witness가 다른 session NPZ를 가리킵니다")
    if _path(str(physical_evidence.get("clock_receipt_path", "")), root=root) != clock:
        raise ValueError("runtime physical witness가 다른 clock receipt를 가리킵니다")
    if physical_evidence.get("session_npz_sha256") != session_sha:
        raise ValueError("runtime physical witness session SHA가 다릅니다")
    if physical_evidence.get("clock_receipt_file_sha256") != clock_file_sha:
        raise ValueError("runtime physical witness clock receipt SHA가 다릅니다")
    if physical_evidence.get("hardware_fingerprint_sha256") != hardware_sha:
        raise ValueError("runtime physical witness hardware fingerprint가 session과 다릅니다")

    # Embedded SHA는 작성자가 임의 JSON을 다시 hash하면 맞출 수 있어
    # physical 증거의 trust root가 아니다. predeclared plan + 같은 raw
    # session/clock으로 producer를 다시 실행하고 exact evidence를 비교한다.
    plan_value = str(physical_evidence.get("plan_path", "")).strip()
    if not plan_value:
        raise ValueError("runtime physical witness predeclared plan path가 누락됐습니다")
    plan_path = _authority_path(plan_value, root=root)
    try:
        _plan_raw, plan_snapshot = _stable_file_bytes(
            plan_path, label="runtime physical clock witness plan"
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"runtime physical clock witness plan이 없습니다: {plan_path}"
        ) from exc
    if physical_evidence.get("plan_file_sha256") != str(plan_snapshot["sha256"]):
        raise ValueError("runtime physical witness plan file SHA가 실제 bytes와 다릅니다")
    recomputed_physical = audit_runtime_physical_clock_witness(
        plan_path=plan_path,
        session_npz_path=session,
        clock_receipt_path=clock,
    )
    if not isinstance(recomputed_physical, dict) or (
        _canonical_json(recomputed_physical) != _canonical_json(physical_evidence)
    ):
        raise ValueError(
            "runtime physical witness가 raw session/clock/plan 재계산 결과와 다릅니다"
        )
    physical_slips = int(
        ((physical_evidence.get("clock_fit") or {}).get("segment_stationarity") or {}).get(
            "sample_slip_count", -1
        )
    )
    if physical_slips < 0:
        raise ValueError("runtime physical witness sample-slip result가 누락됐습니다")

    observed_seconds = float(summary.get("application_observed_seconds", -1.0))
    block_size = int(telemetry.get("block_size", -1))
    if sample_rate != int(telemetry.get("sample_rate", -1)):
        raise ValueError("runtime NPZ/clock sample rate가 다릅니다")
    minimum_steps = max(0, callback_count - 1)
    if not (minimum_steps <= inference_count <= callback_count):
        raise ValueError(
            "zero-fallback 1-hop runtime의 callback/inference step 수 관계가 다릅니다"
        )

    time_domains = telemetry.get("time_domains")
    if not isinstance(time_domains, dict):
        raise ValueError("runtime PortAudio time-domain raw summary가 누락됐습니다")
    sample_slips = 0
    for name in (
        "input_buffer_adc_time",
        "output_buffer_dac_time",
        "callback_current_time",
    ):
        row = time_domains.get(name)
        if not isinstance(row, dict):
            raise ValueError(f"runtime {name} time-domain summary가 누락됐습니다")
        if int(row.get("finite_count", -1)) != callback_count:
            raise ValueError(f"runtime {name} finite count가 callback과 다릅니다")
        for field in (
            "missing_or_nonfinite_count",
            "strict_monotonic_violation_count",
        ):
            if int(row.get(field, -1)) != 0:
                raise ValueError(f"runtime {name} {field}가 0이 아닙니다")
        sample_slips += int(row.get("frame_step_violation_count", -1))
    if sample_slips < 0:
        raise ValueError("runtime sample-slip counter가 누락됐습니다")

    zero_only = (
        "input_ring_overrun_blocks",
        "output_ring_overrun_blocks",
        "input_ring_underrun_blocks",
        "output_ring_underrun_blocks",
    )
    for field in zero_only:
        if int(counters.get(field, -1)) != 0:
            raise ValueError(f"runtime counter {field}가 0이 아닙니다")
    watchdogs = counters.get("watchdog_trip_counts")
    if not isinstance(watchdogs, dict):
        raise ValueError("runtime watchdog counter mapping이 누락됐습니다")
    watchdog_count = sum(int(value) for value in watchdogs.values())
    maximum_input = int(telemetry.get("maximum_input_backlog_samples", -1))
    maximum_output = int(telemetry.get("maximum_output_backlog_samples", -1))
    allowed_input = int(telemetry.get("allowed_input_backlog_samples", -1))
    allowed_output = int(telemetry.get("allowed_output_backlog_samples", -1))
    quantiles = np.percentile(times, [50.0, 95.0, 99.0])

    # 오래 걸리는 physical clock 재계산 도중 authority directory entry나
    # same-inode bytes가 변경되지 않았음을 반환 직전 다시 증명한다.
    for path, before, label in (
        (session, session_snapshot, "runtime session NPZ"),
        (clock, clock_snapshot, "runtime clock receipt"),
        (
            physical_witness,
            physical_snapshot,
            "runtime physical clock witness receipt",
        ),
        (plan_path, plan_snapshot, "runtime physical clock witness plan"),
    ):
        _require_same_snapshot(
            before,
            _stable_file_snapshot(path, label=label),
            label=label,
        )

    evidence = BroadbandRuntimeEvidence(
        control_band_contract_sha256=identity.control_band_contract_sha256,
        experiment_contract_sha256=identity.experiment_contract_sha256,
        training_timing_contract_sha256=identity.training_timing_contract_sha256,
        checkpoint_sha256=identity.checkpoint_sha256,
        deployment_artifact_sha256=identity.deployment_artifact_sha256,
        deployment_metadata_sha256=identity.deployment_metadata_sha256,
        runtime_session_sha256=session_sha,
        runtime_log_sha256=clock_file_sha,
        primary_path_sha256=identity.primary_path_sha256,
        secondary_path_sha256=identity.secondary_path_sha256,
        hardware_fingerprint_sha256=hardware_sha,
        runtime_physical_witness_file_sha256=physical_file_sha,
        runtime_physical_witness_evidence_sha256=physical_evidence_sha,
        model_name=identity.model_name,
        engine=identity.engine,
        power_mode=mode,
        sample_rate=sample_rate,
        block_size=block_size,
        handoff_extra_samples=int(plant.timing.handoff_samples),
        plant_lead_samples=int(plant.timing.digital_reference_lead_samples),
        checkpoint_lead_samples=identity.checkpoint_lead_samples,
        deployment_lead_samples=identity.deployment_lead_samples,
        runtime_lead_samples=int(runtime_cfg.get("digital_reference_lead_samples", -1)),
        observed_seconds=observed_seconds,
        callback_count=callback_count,
        inference_p50_ms=float(quantiles[0]),
        inference_p95_ms=float(quantiles[1]),
        inference_p99_ms=float(quantiles[2]),
        inference_max_ms=float(np.max(times)),
        deadline_miss_count=int(counters.get("deadline_miss_count", -1)),
        engine_error_blocks=int(counters.get("engine_error_blocks", -1)),
        xrun_count=int(counters.get("xrun_count", -1)),
        input_ring_drop_samples=int(counters.get("input_ring_drop_samples", -1)),
        output_ring_drop_samples=int(counters.get("output_ring_drop_samples", -1)),
        ring_add_samples=int(counters.get("ring_add_samples", -1)),
        maximum_input_backlog_samples=maximum_input,
        maximum_output_backlog_samples=maximum_output,
        allowed_input_backlog_samples=allowed_input,
        allowed_output_backlog_samples=allowed_output,
        maximum_excess_input_backlog_samples=max(0, maximum_input - allowed_input),
        maximum_excess_output_backlog_samples=max(0, maximum_output - allowed_output),
        fallback_silence_blocks=int(counters.get("fallback_silence_blocks", -1)),
        watchdog_trip_count=watchdog_count,
        sample_slip_count=sample_slips + physical_slips,
        conditional_physical_timing_pass=bool(
            physical_evidence["conditional_physical_timing_pass"]
        ),
        # Acoustic capture와 같은 저장소가 스스로 서명한 JSON은 callback 진입
        # 이전의 silent period drop을 독립적으로 증명할 수 없다. 현재
        # physical-witness schema는 외부 electrical raw/검증자 결속을 갖지
        # 않으므로, receipt에 true를 써도 canonical authority로 승격하지 않는다.
        # 독립 electrical witness schema와 raw 재검산기가 추가될 때에만 이
        # 값을 외부 증거에서 유도해야 한다.
        independent_clock_authority_pass=False,
    )
    # 이 함수는 BLOCKED raw도 버리지 않는다. 호출자는 반드시 외부
    # strict P/S lead를 받는 ``audit_broadband_runtime_evidence``를 이어서
    # 호출하고, PASS가 아니면 deployment를 막아야 한다.
    return evidence


__all__ = [
    "BROADBAND_RUNTIME_EVIDENCE_SCHEMA",
    "RUNTIME_DEPLOYMENT_METADATA_SCHEMA",
    "BroadbandRuntimeAudit",
    "BroadbandRuntimeEvidence",
    "RuntimeDeploymentIdentity",
    "MAX_P99_INFERENCE_MS",
    "MIN_OBSERVED_SECONDS",
    "audit_broadband_runtime_evidence",
    "build_broadband_runtime_evidence_from_artifacts",
    "snapshot_runtime_deployment_files",
    "verify_runtime_deployment_identity",
]
