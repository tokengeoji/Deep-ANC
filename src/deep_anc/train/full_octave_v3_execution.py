"""125 Hz--8 kHz v3 raw-bound execution envelope의 읽기 전용 preflight.

``full_octave_v3_admission``은 의도적으로 admission-only checklist다. 이 모듈은
그와 별개인 future execution envelope의 **선언 SHA 구조만** 검사한다. raw capture,
analysis, electrical witness, P/S operator, population, sampler, DNH, causal binding,
exact training YAML 및 execution receipt가 하나의 nonce와 실제 file bytes SHA에 모두
결속되어도, 현 저장소에는 그 선언을 독립적으로 검증하는 typed live publisher/validator가
없다. 따라서 이 모듈은 어떤 입력에도 ``READY``를 발행하지 않는다.

이 모듈은 Trainer, DataLoader, GPU, 오디오 장치, subprocess 및 run directory를
만들지 않는다. ``declared_sha_structure_valid``는 receipt 무결성 선언의 형식 검사일
뿐이며 canonical 학습 권한, provenance 또는 실제 물리 측정의 증거가 아니다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ..dsp.control_band_contract import BroadbandFullOctaveContractV3
from .full_octave_v3_admission import (
    ADMISSION_CONFIG_SCHEMA,
    ADMISSION_ROLE,
    audit_full_octave_v3_admission,
)


FULL_OCTAVE_V3_EXECUTION_CONFIG_SCHEMA = "full_octave_v3_execution_config_v1"
FULL_OCTAVE_V3_EXECUTION_RESULT_SCHEMA = "full_octave_v3_execution_preflight_v1"
FULL_OCTAVE_V3_EXECUTION_ROLE = "raw_bound_preflight_no_trainer_no_audio"
FULL_OCTAVE_V3_EXECUTION_RECEIPT_SCHEMA = "full_octave_v3_execution_receipt_v1"
FULL_OCTAVE_V3_RAW_BOUND_BINDING_SCHEMA = "full_octave_v3_raw_bound_binding_receipt_v1"
FULL_OCTAVE_V3_TRAIN_CONFIG_BINDING_SCHEMA = "full_octave_v3_train_config_binding_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXECUTION_STAGES = frozenset({"canonical_pretrain", "canonical_finetune"})
_ADMISSION_ARTIFACT_ROLES = (
    "fullband_raw_capture",
    "causal_plant_authority",
    "population_authority",
    "family_balanced_batch_receipt",
    "dnh_gradient_calibration",
)
_EXECUTION_ARTIFACT_ROLES = (
    "fullband_analysis",
    "electrical_witness_receipt",
    "primary_causal_operator",
    "secondary_causal_operator",
    "causal_plant_binding",
    "canonical_training_config",
    "execution_receipt",
)
_ARTIFACT_ROLES = (*_ADMISSION_ARTIFACT_ROLES, *_EXECUTION_ARTIFACT_ROLES)
_INPUT_ARTIFACT_ROLES = tuple(
    role for role in _ARTIFACT_ROLES if role != "execution_receipt"
)
_ADMISSION_REQUIRED_CHECKS = frozenset(
    {
        "canonical_v3_contract",
        "fullband_raw_capture",
        "causal_plant_authority",
        "population_authority",
        "family_balanced_batch_receipt",
        "dnh_gradient_calibration",
        "causal_plant_authority_metadata",
        "population_authority_metadata",
        "family_balanced_batch_metadata",
        "dnh_gradient_calibration_metadata",
        "v3_trainer_evaluator_consumer_wiring",
    }
)

# 현재는 JSON/YAML/NPZ bytes의 self-attested SHA 관계만 읽을 수 있다. 아래 항목을
# 실제로 검증할 typed authority가 생기기 전에는 complete-looking chain도 절대 training
# permission으로 승격하지 않는다. 이 목록은 future adapter/trust root 구현을 뜻하지
# 않으며, 현재 checker가 **하지 못하는 일**을 report에 보존한다.
UNATTESTED_EXECUTION_PROVENANCE_STATUS = "BLOCKED_UNATTESTED_EXECUTION_PROVENANCE"
UNATTESTED_EXECUTION_PROVENANCE_BLOCKERS = (
    "typed_primary_secondary_operator_raw_analysis_validators_and_exact_timing_crosslinks",
    "typed_fullband_raw_analysis_electrical_witness_validator",
    "actual_submitted_pcm_callback_telemetry_and_native_canonical_recipe_equality",
    "capture_adapter_o_excl_receipt_bound_to_plan_nonce_device_and_monotonic_session",
    "stage_specific_training_schema_with_canonical_finetune_init_checkpoint_contract_and_recorded_selection",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value or "")
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{label}는 lowercase SHA-256이어야 합니다")
    return text


def _require_exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} key 집합이 정확하지 않습니다: {actual}")
    return value


def _check(condition: bool, *, check_id: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": "PASS" if condition else "BLOCKED",
        "detail": detail,
        **extra,
    }


def _inside_repository(root: Path, raw_path: object, *, label: str) -> Path:
    text = str(raw_path or "")
    if not text:
        raise ValueError(f"{label} path가 비었습니다")
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label}는 repository 내부 상대 경로여야 합니다")
    target = root / candidate
    cursor = root
    for part in candidate.parts:
        cursor /= part
        try:
            node = cursor.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(node.st_mode):
            raise ValueError(f"{label} 경로에 symlink가 있습니다: {cursor}")
    return target


def _snapshot_regular_file(path: Path) -> tuple[bytes, str]:
    """O_NOFOLLOW와 fstat 전후 비교를 쓰는 immutable file snapshot."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError(f"artifact를 열 수 없습니다: {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"regular file만 허용합니다: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in stable):
        raise ValueError(f"artifact snapshot 도중 파일이 바뀌었습니다: {path}")
    if stat.S_ISLNK(path.lstat().st_mode):
        raise ValueError(f"artifact symlink는 허용하지 않습니다: {path}")
    content = b"".join(chunks)
    if len(content) != int(after.st_size):
        raise ValueError(f"artifact byte 수가 파일 크기와 다릅니다: {path}")
    return content, _sha256_bytes(content)


def _artifact_intent(
    artifacts: Mapping[str, Any], role: str
) -> tuple[str | None, str | None]:
    entry = _require_exact_keys(
        artifacts[role], {"path", "sha256"}, label=f"artifacts.{role}"
    )
    path = entry["path"]
    sha = entry["sha256"]
    if (path is None) != (sha is None):
        raise ValueError(f"artifacts.{role} path와 sha256은 함께 선언하거나 함께 null이어야 합니다")
    if path is None:
        return None, None
    if not isinstance(path, str):
        raise ValueError(f"artifacts.{role}.path는 string 또는 null이어야 합니다")
    return path, _require_sha256(sha, label=f"artifacts.{role}.sha256")


def _read_json_artifact(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}는 UTF-8 JSON artifact여야 합니다") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON 최상위는 mapping이어야 합니다")
    return payload


def _read_yaml_artifact(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(content.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"{label}는 UTF-8 YAML mapping이어야 합니다") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} YAML 최상위는 mapping이어야 합니다")
    return payload


def _require_exact_physical_bands(value: object, *, label: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{label}는 exact v3 physical subband list여야 합니다")
    try:
        actual = tuple(tuple(float(item) for item in band) for band in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} subband가 숫자가 아닙니다") from exc
    expected = tuple(
        tuple(float(item) for item in band)
        for band in BroadbandFullOctaveContractV3.canonical().physical_identification_subbands_hz
    )
    if actual != expected:
        raise ValueError(f"{label}가 canonical v3 physical subband 전체와 다릅니다")


def _require_false(value: object, *, label: str) -> None:
    if value is not False:
        raise ValueError(f"{label}=false가 필요합니다")


def _require_true(value: object, *, label: str) -> None:
    if value is not True:
        raise ValueError(f"{label}=true가 필요합니다")


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label}는 0 이상 bool 아닌 int여야 합니다")
    return int(value)


def _read_config(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, tuple[str | None, str | None]]]:
    root = _require_exact_keys(
        payload,
        {
            "schema",
            "role",
            "execution_stage",
            "execution_nonce_sha256",
            "control_band_contract",
            "artifacts",
        },
        label="full-octave v3 execution config",
    )
    if root["schema"] != FULL_OCTAVE_V3_EXECUTION_CONFIG_SCHEMA:
        raise ValueError("full-octave v3 execution config schema가 다릅니다")
    if root["role"] != FULL_OCTAVE_V3_EXECUTION_ROLE:
        raise ValueError("execution config role이 raw-bound preflight가 아닙니다")
    if root["execution_stage"] not in _EXECUTION_STAGES:
        raise ValueError("execution_stage는 canonical_pretrain 또는 canonical_finetune이어야 합니다")
    nonce = root["execution_nonce_sha256"]
    if nonce is not None:
        _require_sha256(nonce, label="execution_nonce_sha256")
    contract = _require_exact_keys(
        root["control_band_contract"], {"id", "sha256"}, label="control_band_contract"
    )
    canonical = BroadbandFullOctaveContractV3.canonical()
    if contract["id"] != canonical.contract_id or contract["sha256"] != canonical.digest():
        raise ValueError("execution config가 exact canonical v3 contract를 가리키지 않습니다")
    artifacts = _require_exact_keys(
        root["artifacts"], set(_ARTIFACT_ROLES), label="execution artifacts"
    )
    intents = {role: _artifact_intent(artifacts, role) for role in _ARTIFACT_ROLES}
    return root, intents


def is_full_octave_v3_execution_config(payload: Mapping[str, Any]) -> bool:
    """``train.py``가 generic training config 이전에 fail-close할 schema probe."""

    return (
        isinstance(payload, Mapping)
        and payload.get("schema") == FULL_OCTAVE_V3_EXECUTION_CONFIG_SCHEMA
    )


def _admission_payload_from_execution(
    root: Mapping[str, Any], intents: Mapping[str, tuple[str | None, str | None]]
) -> dict[str, Any]:
    return {
        "schema": ADMISSION_CONFIG_SCHEMA,
        "role": ADMISSION_ROLE,
        "control_band_contract": root["control_band_contract"],
        "artifacts": {
            role: {"path": intents[role][0], "sha256": intents[role][1]}
            for role in _ADMISSION_ARTIFACT_ROLES
        },
    }


def _admission_chain_ready(report: Mapping[str, Any]) -> bool:
    statuses = {
        str(check.get("id")): str(check.get("status"))
        for check in report.get("checks", [])
        if isinstance(check, Mapping)
    }
    return all(statuses.get(check_id) == "PASS" for check_id in _ADMISSION_REQUIRED_CHECKS)


def _execution_input_payload(
    root: Mapping[str, Any],
    intents: Mapping[str, tuple[str | None, str | None]],
    snapshots: Mapping[str, tuple[bytes, str]],
) -> dict[str, object]:
    if root["execution_nonce_sha256"] is None:
        raise ValueError("execution nonce가 없어 execution input을 계산할 수 없습니다")
    bindings: dict[str, dict[str, str]] = {}
    for role in _INPUT_ARTIFACT_ROLES:
        path, declared_sha = intents[role]
        snapshot = snapshots.get(role)
        if path is None or declared_sha is None or snapshot is None:
            raise ValueError(f"execution input에 필요한 {role} artifact가 없습니다")
        if snapshot[1] != declared_sha:
            raise ValueError(f"execution input {role} SHA snapshot이 config와 다릅니다")
        bindings[role] = {"path": path, "sha256": snapshot[1]}
    return {
        "schema": "full_octave_v3_execution_input_v1",
        "role": root["role"],
        "execution_stage": root["execution_stage"],
        "execution_nonce_sha256": root["execution_nonce_sha256"],
        "control_band_contract": root["control_band_contract"],
        "artifacts": bindings,
    }


def _validate_raw_bound_binding(
    payload: Mapping[str, Any],
    *,
    contract: BroadbandFullOctaveContractV3,
    raw_sha256: str,
    analysis_sha256: str,
    plant_authority_sha256: str,
    timing_sha256: str,
    electrical_witness_sha256: str,
    primary_operator_sha256: str,
    secondary_operator_sha256: str,
) -> None:
    expected = {
        "schema",
        "binding_schema",
        "control_band_contract_sha256",
        "raw_capture_sha256",
        "analysis_sha256",
        "causal_plant_authority_sha256",
        "training_timing_contract_sha256",
        "electrical_witness_receipt_sha256",
        "primary_operator_file_sha256",
        "secondary_operator_file_sha256",
        "sample_rate_hz",
        "block_size",
        "verified_physical_subbands_hz",
        "err_channel_index",
        "reference_channel_index",
        "fixture_only",
        "canonical_training_eligible",
        "publisher",
    }
    entry = _require_exact_keys(payload, expected, label="raw-bound causal plant binding")
    if entry["schema"] != FULL_OCTAVE_V3_RAW_BOUND_BINDING_SCHEMA:
        raise ValueError("raw-bound causal plant binding schema가 다릅니다")
    if entry["binding_schema"] != "full_octave_causal_plant_binding_v4":
        raise ValueError("raw-bound causal plant binding이 v4 causal binding이 아닙니다")
    if entry["control_band_contract_sha256"] != contract.digest():
        raise ValueError("raw-bound causal plant binding v3 contract SHA가 다릅니다")
    expected_refs = {
        "raw_capture_sha256": raw_sha256,
        "analysis_sha256": analysis_sha256,
        "causal_plant_authority_sha256": plant_authority_sha256,
        "training_timing_contract_sha256": timing_sha256,
        "electrical_witness_receipt_sha256": electrical_witness_sha256,
        "primary_operator_file_sha256": primary_operator_sha256,
        "secondary_operator_file_sha256": secondary_operator_sha256,
    }
    for key, expected_sha in expected_refs.items():
        observed = _require_sha256(entry[key], label=f"raw-bound causal plant binding.{key}")
        if observed != expected_sha:
            raise ValueError(f"raw-bound causal plant binding {key}가 configured bytes와 다릅니다")
    if entry["sample_rate_hz"] != 48_000 or entry["block_size"] != 256:
        raise ValueError("raw-bound causal plant binding sample rate/block size가 다릅니다")
    _require_exact_physical_bands(
        entry["verified_physical_subbands_hz"], label="raw-bound causal plant binding"
    )
    _require_nonnegative_int(entry["err_channel_index"], label="ERR channel index")
    _require_nonnegative_int(entry["reference_channel_index"], label="reference channel index")
    _require_false(entry["fixture_only"], label="raw-bound causal plant binding.fixture_only")
    _require_true(
        entry["canonical_training_eligible"],
        label="raw-bound causal plant binding.canonical_training_eligible",
    )
    if entry["publisher"] != "full_octave_v3_raw_bound_physical_publisher_v1":
        raise ValueError("raw-bound causal plant binding publisher가 physical raw publisher가 아닙니다")


def _validate_training_config_binding(
    payload: Mapping[str, Any],
    *,
    contract: BroadbandFullOctaveContractV3,
    execution_stage: str,
    nonce_sha256: str,
    binding_sha256: str,
    timing_sha256: str,
) -> None:
    if payload.get("experiment_role") != execution_stage:
        raise ValueError("canonical training config experiment_role과 execution_stage가 다릅니다")
    marker = _require_exact_keys(
        payload.get("full_octave_v3_execution"),
        {
            "schema",
            "execution_stage",
            "control_band_contract_sha256",
            "execution_nonce_sha256",
            "causal_plant_binding_sha256",
            "training_timing_contract_sha256",
            "fixture_only",
            "canonical_training_eligible",
            "requires_execution_preflight",
        },
        label="canonical training config.full_octave_v3_execution",
    )
    if marker["schema"] != FULL_OCTAVE_V3_TRAIN_CONFIG_BINDING_SCHEMA:
        raise ValueError("canonical training config v3 binding schema가 다릅니다")
    if marker["execution_stage"] != execution_stage:
        raise ValueError("canonical training config v3 binding stage가 다릅니다")
    if marker["control_band_contract_sha256"] != contract.digest():
        raise ValueError("canonical training config v3 contract SHA가 다릅니다")
    for key, expected_sha in (
        ("execution_nonce_sha256", nonce_sha256),
        ("causal_plant_binding_sha256", binding_sha256),
        ("training_timing_contract_sha256", timing_sha256),
    ):
        observed = _require_sha256(marker[key], label=f"canonical training config.{key}")
        if observed != expected_sha:
            raise ValueError(f"canonical training config {key}가 execution authority와 다릅니다")
    _require_false(marker["fixture_only"], label="canonical training config.fixture_only")
    _require_true(
        marker["canonical_training_eligible"],
        label="canonical training config.canonical_training_eligible",
    )
    _require_true(
        marker["requires_execution_preflight"],
        label="canonical training config.requires_execution_preflight",
    )


def _validate_execution_receipt(
    payload: Mapping[str, Any],
    *,
    contract: BroadbandFullOctaveContractV3,
    execution_input_sha256: str,
    execution_stage: str,
    nonce_sha256: str,
    artifact_shas: Mapping[str, str],
    timing_sha256: str,
) -> None:
    expected = {
        "schema",
        "preflight_role",
        "control_band_contract_sha256",
        "execution_input_sha256",
        "execution_stage",
        "execution_nonce_sha256",
        "fullband_raw_capture_sha256",
        "fullband_analysis_sha256",
        "causal_plant_authority_sha256",
        "electrical_witness_receipt_sha256",
        "primary_causal_operator_sha256",
        "secondary_causal_operator_sha256",
        "causal_plant_binding_sha256",
        "canonical_training_config_sha256",
        "population_authority_sha256",
        "family_balanced_batch_receipt_sha256",
        "dnh_gradient_calibration_sha256",
        "training_timing_contract_sha256",
        "fixture_only",
        "canonical_training_eligible",
    }
    entry = _require_exact_keys(payload, expected, label="full-octave v3 execution receipt")
    if entry["schema"] != FULL_OCTAVE_V3_EXECUTION_RECEIPT_SCHEMA:
        raise ValueError("full-octave v3 execution receipt schema가 다릅니다")
    if entry["preflight_role"] != "raw_bound_execution_receipt":
        raise ValueError("execution receipt preflight role이 다릅니다")
    if entry["control_band_contract_sha256"] != contract.digest():
        raise ValueError("execution receipt v3 contract SHA가 다릅니다")
    if entry["execution_input_sha256"] != execution_input_sha256:
        raise ValueError("execution receipt가 exact execution input bytes에 결속되지 않았습니다")
    if entry["execution_stage"] != execution_stage:
        raise ValueError("execution receipt stage가 execution config와 다릅니다")
    if entry["execution_nonce_sha256"] != nonce_sha256:
        raise ValueError("execution receipt nonce가 execution config와 다릅니다")
    receipt_to_artifact = {
        "fullband_raw_capture_sha256": "fullband_raw_capture",
        "fullband_analysis_sha256": "fullband_analysis",
        "causal_plant_authority_sha256": "causal_plant_authority",
        "electrical_witness_receipt_sha256": "electrical_witness_receipt",
        "primary_causal_operator_sha256": "primary_causal_operator",
        "secondary_causal_operator_sha256": "secondary_causal_operator",
        "causal_plant_binding_sha256": "causal_plant_binding",
        "canonical_training_config_sha256": "canonical_training_config",
        "population_authority_sha256": "population_authority",
        "family_balanced_batch_receipt_sha256": "family_balanced_batch_receipt",
        "dnh_gradient_calibration_sha256": "dnh_gradient_calibration",
    }
    for receipt_key, artifact_role in receipt_to_artifact.items():
        observed = _require_sha256(entry[receipt_key], label=f"execution receipt.{receipt_key}")
        if observed != artifact_shas[artifact_role]:
            raise ValueError(f"execution receipt {receipt_key}가 configured {artifact_role} bytes와 다릅니다")
    observed_timing = _require_sha256(
        entry["training_timing_contract_sha256"], label="execution receipt timing SHA"
    )
    if observed_timing != timing_sha256:
        raise ValueError("execution receipt timing SHA가 causal plant authority와 다릅니다")
    _require_false(entry["fixture_only"], label="execution receipt.fixture_only")
    _require_true(
        entry["canonical_training_eligible"],
        label="execution receipt.canonical_training_eligible",
    )


def _resnapshot_unchanged(
    *,
    targets: Mapping[str, Path],
    snapshots: Mapping[str, tuple[bytes, str]],
) -> None:
    for role, target in targets.items():
        _, after_sha = _snapshot_regular_file(target)
        before = snapshots.get(role)
        if before is None or before[1] != after_sha:
            raise ValueError(f"execution preflight 도중 {role} artifact bytes가 바뀌었습니다")


def audit_full_octave_v3_execution(
    payload: Mapping[str, Any], *, repo_root: str | Path
) -> dict[str, Any]:
    """future canonical execution receipt를 파일 bytes 기준으로 fail-close 검사한다.

    기본 null config는 예외가 아니라 structured ``BLOCKED``다. 반면 half declaration,
    SHA mismatch, symlink, malformed future receipt는 허위 authority로 이어질 수 있으므로
    ``ValueError``다.
    """

    root_payload, intents = _read_config(payload)
    root = Path(repo_root).resolve(strict=True)
    if root.is_symlink():
        raise ValueError("repository root가 symlink일 수 없습니다")
    contract = BroadbandFullOctaveContractV3.canonical()
    checks: list[dict[str, Any]] = [
        _check(
            True,
            check_id="canonical_v3_contract",
            detail="125 Hz--8 kHz canonical v3 contract와 digest가 execution config에 결속됐습니다.",
            control_band_contract_sha256=contract.digest(),
        )
    ]
    targets: dict[str, Path] = {}
    snapshots: dict[str, tuple[bytes, str]] = {}
    for role in _ARTIFACT_ROLES:
        path_text, expected_sha = intents[role]
        check_id = f"artifact:{role}"
        if path_text is None:
            checks.append(_check(False, check_id=check_id, detail="artifact가 아직 선언되지 않았습니다."))
            continue
        target = _inside_repository(root, path_text, label=f"artifacts.{role}")
        if not target.exists():
            checks.append(
                _check(
                    False,
                    check_id=check_id,
                    detail="declared artifact가 repository에 없습니다.",
                    path=path_text,
                )
            )
            continue
        content, observed_sha = _snapshot_regular_file(target)
        if observed_sha != expected_sha:
            raise ValueError(
                f"artifacts.{role} bytes SHA가 config와 다릅니다: "
                f"expected={expected_sha}, observed={observed_sha}"
            )
        targets[role] = target
        snapshots[role] = (content, observed_sha)
        checks.append(
            _check(
                True,
                check_id=check_id,
                detail="regular-file bytes SHA가 config와 일치합니다.",
                path=path_text,
                sha256=observed_sha,
            )
        )

    admission_report = audit_full_octave_v3_admission(
        _admission_payload_from_execution(root_payload, intents), repo_root=root
    )
    admission_ready = _admission_chain_ready(admission_report)
    checks.append(
        _check(
            admission_ready,
            check_id="v3_admission_artifact_chain",
            detail=(
                "fullband raw/P-S/population/family sampler/DNH artifact chain이 admission schema를 통과했습니다."
                if admission_ready
                else "admission artifact chain이 아직 complete/valid하지 않습니다."
            ),
        )
    )

    nonce = root_payload["execution_nonce_sha256"]
    nonce_ready = nonce is not None
    checks.append(
        _check(
            nonce_ready,
            check_id="execution_nonce",
            detail=(
                "execution nonce가 receipt replay를 막도록 config에 결속됐습니다."
                if nonce_ready
                else "execution nonce가 아직 선언되지 않았습니다."
            ),
        )
    )

    input_ready = nonce_ready and all(role in snapshots for role in _INPUT_ARTIFACT_ROLES)
    execution_input_sha256: str | None = None
    if input_ready:
        input_payload = _execution_input_payload(root_payload, intents, snapshots)
        execution_input_sha256 = _sha256_bytes(_canonical_json(input_payload))
        checks.append(
            _check(
                True,
                check_id="execution_input_binding",
                detail="receipt 전의 모든 configured execution input bytes/path/nonce가 하나의 digest로 고정됐습니다.",
                sha256=execution_input_sha256,
            )
        )
    else:
        checks.append(
            _check(
                False,
                check_id="execution_input_binding",
                detail="nonce 또는 receipt 이외 execution input artifact가 없어 digest를 만들 수 없습니다.",
            )
        )

    binding_ready = False
    binding_sha256 = snapshots.get("causal_plant_binding", (b"", ""))[1]
    timing_sha256: str | None = None
    plant_payload: dict[str, Any] | None = None
    binding_dependencies = {
        "fullband_raw_capture",
        "fullband_analysis",
        "causal_plant_authority",
        "electrical_witness_receipt",
        "primary_causal_operator",
        "secondary_causal_operator",
        "causal_plant_binding",
    }
    if admission_ready and binding_dependencies.issubset(snapshots):
        plant_payload = _read_json_artifact(
            snapshots["causal_plant_authority"][0], label="causal_plant_authority"
        )
        timing_sha256 = _require_sha256(
            plant_payload.get("training_timing_contract_sha256"),
            label="causal plant authority timing SHA",
        )
        analysis_sha256 = _require_sha256(
            plant_payload.get("analysis_sha256"), label="causal plant authority analysis SHA"
        )
        if analysis_sha256 != snapshots["fullband_analysis"][1]:
            raise ValueError("causal plant authority analysis SHA가 configured fullband_analysis bytes와 다릅니다")
        _validate_raw_bound_binding(
            _read_json_artifact(
                snapshots["causal_plant_binding"][0], label="causal_plant_binding"
            ),
            contract=contract,
            raw_sha256=snapshots["fullband_raw_capture"][1],
            analysis_sha256=analysis_sha256,
            plant_authority_sha256=snapshots["causal_plant_authority"][1],
            timing_sha256=timing_sha256,
            electrical_witness_sha256=snapshots["electrical_witness_receipt"][1],
            primary_operator_sha256=snapshots["primary_causal_operator"][1],
            secondary_operator_sha256=snapshots["secondary_causal_operator"][1],
        )
        binding_ready = True
        checks.append(
            _check(
                True,
                check_id="raw_bound_nonfixture_causal_binding",
                detail="non-fixture v4 causal binding receipt가 raw/analysis/witness/operator/timing bytes와 결속됐습니다.",
                sha256=binding_sha256,
            )
        )
    else:
        checks.append(
            _check(
                False,
                check_id="raw_bound_nonfixture_causal_binding",
                detail="admission artifact chain 또는 raw-bound causal binding dependencies가 아직 없습니다.",
            )
        )

    training_config_ready = False
    if binding_ready and nonce_ready and "canonical_training_config" in snapshots and timing_sha256:
        _validate_training_config_binding(
            _read_yaml_artifact(
                snapshots["canonical_training_config"][0], label="canonical_training_config"
            ),
            contract=contract,
            execution_stage=str(root_payload["execution_stage"]),
            nonce_sha256=str(nonce),
            binding_sha256=binding_sha256,
            timing_sha256=timing_sha256,
        )
        training_config_ready = True
        checks.append(
            _check(
                True,
                check_id="canonical_training_config_binding",
                detail="actual training YAML이 stage/nonce/non-fixture binding/timing authority와 결속됐습니다.",
                sha256=snapshots["canonical_training_config"][1],
            )
        )
    else:
        checks.append(
            _check(
                False,
                check_id="canonical_training_config_binding",
                detail="non-fixture causal binding, timing SHA 또는 canonical training YAML이 아직 없습니다.",
            )
        )

    receipt_ready = False
    if (
        input_ready
        and execution_input_sha256 is not None
        and binding_ready
        and training_config_ready
        and timing_sha256 is not None
        and "execution_receipt" in snapshots
    ):
        artifact_shas = {role: snapshots[role][1] for role in _INPUT_ARTIFACT_ROLES}
        _validate_execution_receipt(
            _read_json_artifact(
                snapshots["execution_receipt"][0], label="execution_receipt"
            ),
            contract=contract,
            execution_input_sha256=execution_input_sha256,
            execution_stage=str(root_payload["execution_stage"]),
            nonce_sha256=str(nonce),
            artifact_shas=artifact_shas,
            timing_sha256=timing_sha256,
        )
        receipt_ready = True
        checks.append(
            _check(
                True,
                check_id="execution_receipt_binding",
                detail="execution receipt가 exact input digest, nonce, physical evidence 및 training YAML bytes에 결속됐습니다.",
                sha256=snapshots["execution_receipt"][1],
            )
        )
    else:
        checks.append(
            _check(
                False,
                check_id="execution_receipt_binding",
                detail="complete non-fixture execution inputs와 receipt가 아직 함께 존재하지 않습니다.",
            )
        )

    # 이 bool은 이름 그대로 self-declared bytes/SHA structure만 뜻한다. admission과
    # binding receipt의 `canonical_training_eligible=true` 역시 현재 checker 입장에서는
    # 검증된 authority가 아니라 artifact 저자가 적은 문자열이다.
    declared_sha_structure_valid = (
        admission_ready
        and nonce_ready
        and binding_ready
        and training_config_ready
        and receipt_ready
    )
    checks.append(
        _check(
            declared_sha_structure_valid,
            check_id="declared_sha_structure",
            detail=(
                "self-attested execution config/receipt의 exact SHA·nonce·non-fixture field 구조가 서로 일치합니다. "
                "이는 typed provenance 검증이나 학습 권한이 아닙니다."
                if declared_sha_structure_valid
                else "raw-bound execution config/receipt의 선언 SHA 구조가 아직 complete·non-fixture·cross-bound 상태가 아닙니다."
            ),
        )
    )
    # trusted publisher, raw/analysis parser, electrical witness decoder 또는 exact
    # operator/timing cross-link verifier가 아직 없으므로 이 항목은 intentionally
    # false다. 값을 true로 만드는 local schema/JSON은 추가하지 않는다.
    checks.append(
        _check(
            False,
            check_id="typed_execution_provenance",
            detail=(
                "현 checker는 self-attested artifact SHA만 읽습니다. typed P/S operator·raw·analysis "
                "validator, electrical witness, submitted PCM telemetry, O_EXCL adapter receipt 및 "
                "stage-specific canonical training authority가 없습니다."
            ),
            required_authorities=list(UNATTESTED_EXECUTION_PROVENANCE_BLOCKERS),
        )
    )
    # 기존 consumer guardrail ID를 유지하되, SHA-only chain이 complete여도 canonical
    # execution config로 승격되지 않도록 항상 blocked로 남긴다.
    checks.append(
        _check(
            False,
            check_id="v3_raw_bound_execution_config",
            detail=(
                "declared SHA 구조는 complete하지만 independent typed provenance가 없어 "
                "canonical raw-bound execution config가 아닙니다."
                if declared_sha_structure_valid
                else "raw-bound execution config의 선언 구조 및 independent typed provenance가 아직 없습니다."
            ),
        )
    )
    # 가장 마지막에 다시 열어 first snapshot과 비교한다. file 하나만 atomic하게
    # 읽는 것뿐 아니라 receipt 검증 중 교체도 fail-close한다.
    _resnapshot_unchanged(targets=targets, snapshots=snapshots)
    blockers = tuple(check["id"] for check in checks if check["status"] != "PASS")
    return {
        "schema": FULL_OCTAVE_V3_EXECUTION_RESULT_SCHEMA,
        "status": (
            UNATTESTED_EXECUTION_PROVENANCE_STATUS
            if declared_sha_structure_valid
            else "BLOCKED"
        ),
        # Backward-compatible field name, but a SHA declaration must never make it true.
        "preflight_ready": False,
        "declared_sha_structure_valid": declared_sha_structure_valid,
        "typed_execution_provenance_attested": False,
        "self_attested_artifacts_only": declared_sha_structure_valid,
        "canonical_training_eligible": False,
        "deployment_eligible": False,
        # 이 receipt preflight는 physical G4, checkpoint, runtime deployment permission이
        # 아니다. current generic Trainer는 v3 raw-bound binding loader를 아직 소비하지
        # 않으므로 train.py도 여기서 실행을 멈춘다.
        "trainer_release": False,
        "audio_opened": False,
        "trainer_constructed": False,
        "gpu_initialized": False,
        "runs_directory_created": False,
        "control_band_contract": contract.model_dump(mode="json"),
        "control_band_contract_sha256": contract.digest(),
        "execution_input_sha256": execution_input_sha256,
        "checks": checks,
        "blockers": blockers,
        "blocking_requirements": list(UNATTESTED_EXECUTION_PROVENANCE_BLOCKERS),
        "next_required_implementation": (
            "현재 self-attested SHA chain을 trusted live authority로 승격하는 구현은 하지 않습니다. "
            "먼저 typed P/S operator·raw·analysis validator와 timing exact crosslink, typed raw/analysis/"
            "electrical witness, actual submitted PCM/callback telemetry 및 native↔canonical recipe equality, "
            "capture adapter O_EXCL receipt, canonical finetune init checkpoint·contract·recorded selection을 "
            "포함한 stage-specific training schema가 독립 검증돼야 합니다. 그 뒤에도 raw-bound loader와 "
            "Trainer/DataLoader consumer를 별도 review로 연결해야 합니다."
        ),
    }


def render_full_octave_v3_execution_markdown(report: Mapping[str, Any]) -> str:
    """CLI stdout/immutable 기록용 사람이 읽는 preflight summary."""

    lines = [
        "# Full-octave v3 raw-bound execution preflight",
        "",
        f"- 상태: `{report['status']}`",
        f"- preflight 준비: `{report['preflight_ready']}`",
        f"- 선언 SHA 구조만 유효: `{report['declared_sha_structure_valid']}`",
        f"- typed execution provenance attested: `{report['typed_execution_provenance_attested']}`",
        f"- Trainer release: `{report['trainer_release']}`",
        f"- 오디오/Trainer/GPU/run 생성: `{report['audio_opened']}`/"
        f"`{report['trainer_constructed']}`/`{report['gpu_initialized']}`/"
        f"`{report['runs_directory_created']}`",
        f"- v3 contract SHA: `{report['control_band_contract_sha256']}`",
        "",
        "## 검사",
        "",
    ]
    for check in report["checks"]:
        lines.append(f"- [{check['status']}] `{check['id']}` — {check['detail']}")
    lines.extend(["", "## 다음 구현", "", str(report["next_required_implementation"]), ""])
    return "\n".join(lines)


def write_execution_preflight_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> None:
    """선택적 report 기록용 O_EXCL writer. 기본 preflight는 이를 호출하지 않는다."""

    target = Path(path)
    data = _canonical_json(dict(payload)) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("execution preflight report write가 진행되지 않았습니다")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "FULL_OCTAVE_V3_EXECUTION_CONFIG_SCHEMA",
    "FULL_OCTAVE_V3_EXECUTION_RESULT_SCHEMA",
    "FULL_OCTAVE_V3_EXECUTION_ROLE",
    "FULL_OCTAVE_V3_EXECUTION_RECEIPT_SCHEMA",
    "FULL_OCTAVE_V3_RAW_BOUND_BINDING_SCHEMA",
    "FULL_OCTAVE_V3_TRAIN_CONFIG_BINDING_SCHEMA",
    "UNATTESTED_EXECUTION_PROVENANCE_BLOCKERS",
    "UNATTESTED_EXECUTION_PROVENANCE_STATUS",
    "audit_full_octave_v3_execution",
    "is_full_octave_v3_execution_config",
    "render_full_octave_v3_execution_markdown",
    "write_execution_preflight_json_exclusive",
]
