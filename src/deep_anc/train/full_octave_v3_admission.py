"""125 Hz--8 kHz v3 학습 전용의 읽기 전용 admission preflight.

이 모듈은 학습기를 만들거나 GPU/오디오 장치를 열지 않는다. 목적은 아직 없는
full-octave P/S, 계보, sampler, DNH gradient 증거를 기존 150--1600 Hz 자료로
대체하는 실수를 막는 것이다. v3의 causal prefix/loss/FxLMS consumer 자체는
구현됐지만, 이 YAML은 여전히 *admission-only* checklist다. raw-bound execution
config와 physical evidence가 없는 한 이 모듈은 학습을 허용하지 않는다.

개별 implementation check는 ``PASS``일 수 있어도, 이 admission-only 경계의
``eligible``은 항상 false다. raw-first artifact publisher와 별도 raw-bound execution
envelope/receipt가 함께 검증되기 전에는 여기서 학습을 열 수 없다.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..dsp.control_band_contract import BroadbandFullOctaveContractV3


ADMISSION_CONFIG_SCHEMA = "full_octave_v3_admission_config_v1"
ADMISSION_RESULT_SCHEMA = "full_octave_v3_admission_audit_v1"
ADMISSION_ROLE = "admission_only_no_trainer_no_audio"

# 아래 True는 future artifact만 보고 낙관적으로 올린 값이 아니다.
#
# - train/full_octave_v3_consumers.py:
#   CausalSecondaryPrefixAdapterV1 -> P*n + S*y ->
#   BroadbandFullOctaveLossPrimitiveV3 실제 tensor consumer
# - eval/full_octave_v3.py:
#   같은 binding/reference/prefix/block의 matched FxLMS 7-octave evaluator
#
# 두 경로의 prefix crop, 7-octave loss, FxLMS matched binding은 dedicated CPU
# regression으로 검증한다. 그러나 현재 config의 role은 admission-only이므로 이
# boolean은 canonical training 또는 physical G4 허용을 뜻하지 않는다.
V3_TRAINER_EVALUATOR_CONSUMERS_IMPLEMENTED = True

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_FAMILIES = ("speech", "music", "environment", "machine")
_REQUIRED_SPLITS = ("train", "val", "test")
_ARTIFACT_ROLES = (
    "fullband_raw_capture",
    "causal_plant_authority",
    "population_authority",
    "family_balanced_batch_receipt",
    "dnh_gradient_calibration",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} key 집합이 정확하지 않습니다: {actual}")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value or "")
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{label}는 lowercase SHA-256이어야 합니다")
    return text


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
    # parent symlink를 통한 root 탈출도 허용하지 않는다. 존재하지 않는 target은 이
    # 함수가 아니라 caller가 structured BLOCKED로 보고한다.
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
    """TOCTOU를 줄인 immutable regular-file snapshot이다."""

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
    raw = artifacts[role]
    entry = _require_exact_keys(raw, {"path", "sha256"}, label=f"artifacts.{role}")
    path = entry["path"]
    sha = entry["sha256"]
    if (path is None) != (sha is None):
        raise ValueError(f"artifacts.{role} path와 sha256은 함께 선언하거나 함께 null이어야 합니다")
    if path is None:
        return None, None
    if not isinstance(path, str):
        raise ValueError(f"artifacts.{role}.path는 string 또는 null이어야 합니다")
    return path, _require_sha256(sha, label=f"artifacts.{role}.sha256")


def _read_json_artifact(content: bytes, *, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role}는 UTF-8 JSON artifact여야 합니다") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{role} JSON 최상위는 mapping이어야 합니다")
    return payload


def _require_bool(payload: Mapping[str, Any], key: str, *, label: str) -> None:
    if payload.get(key) is not True:
        raise ValueError(f"{label}.{key}=true가 필요합니다")


def _require_number_between(
    payload: Mapping[str, Any], key: str, lo: float, hi: float, *, label: str
) -> None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}.{key}는 숫자여야 합니다")
    numeric = float(value)
    if not math.isfinite(numeric) or not lo <= numeric <= hi:
        raise ValueError(f"{label}.{key}는 [{lo}, {hi}] 범위여야 합니다")


def _validate_operator(
    operator: object,
    *,
    role: str,
    contract: BroadbandFullOctaveContractV3,
) -> None:
    entry = _require_exact_keys(
        operator,
        {
            "role",
            "sample_rate_hz",
            "causal",
            "physical_measurement",
            "operator_file_sha256",
            "verified_lower_hz",
            "verified_upper_hz",
        },
        label=f"causal plant {role} operator",
    )
    if entry["role"] != role or entry["sample_rate_hz"] != 48_000:
        raise ValueError(f"causal plant {role} operator role/sample_rate가 다릅니다")
    if entry["causal"] is not True or entry["physical_measurement"] is not True:
        raise ValueError(f"causal plant {role} operator는 physical causal이어야 합니다")
    _require_sha256(entry["operator_file_sha256"], label=f"{role} operator file SHA")
    lower = entry["verified_lower_hz"]
    upper = entry["verified_upper_hz"]
    if isinstance(lower, bool) or isinstance(upper, bool):
        raise ValueError(f"causal plant {role} verified band가 숫자가 아닙니다")
    if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
        raise ValueError(f"causal plant {role} verified band가 숫자가 아닙니다")
    if (
        not math.isfinite(float(lower))
        or not math.isfinite(float(upper))
        or float(lower) > contract.physical_identification_subbands_hz[0][0]
        or float(upper) < contract.physical_identification_subbands_hz[-1][1]
    ):
        raise ValueError(f"causal plant {role}가 v3 88.388--11313.708 Hz를 모두 덮지 않습니다")


def _validate_causal_plant_authority(
    payload: Mapping[str, Any],
    *,
    contract: BroadbandFullOctaveContractV3,
    raw_capture_sha256: str,
) -> None:
    expected = {
        "schema",
        "control_band_contract_sha256",
        "raw_capture_sha256",
        "analysis_sha256",
        "training_timing_contract_sha256",
        "sample_rate_hz",
        "block_size",
        "plant_identification_pass",
        "electrical_output_witness_pass",
        "shared_clock_authority_pass",
        "hardware_sample_identity_pass",
        "canonical_training_eligible",
        "timing_contract",
        "primary_operator",
        "secondary_operator",
    }
    entry = _require_exact_keys(payload, expected, label="causal plant authority")
    if entry["schema"] != "full_octave_causal_plant_authority_v3":
        raise ValueError("causal plant authority schema가 다릅니다")
    if entry["control_band_contract_sha256"] != contract.digest():
        raise ValueError("causal plant authority v3 contract SHA가 다릅니다")
    if entry["raw_capture_sha256"] != raw_capture_sha256:
        raise ValueError("causal plant authority가 configured fullband raw bytes에 결속되지 않았습니다")
    _require_sha256(entry["analysis_sha256"], label="causal plant analysis SHA")
    _require_sha256(
        entry["training_timing_contract_sha256"], label="training timing contract SHA"
    )
    if entry["sample_rate_hz"] != 48_000 or entry["block_size"] != 256:
        raise ValueError("causal plant authority sample rate/block size가 다릅니다")
    for key in (
        "plant_identification_pass",
        "electrical_output_witness_pass",
        "shared_clock_authority_pass",
        "hardware_sample_identity_pass",
        "canonical_training_eligible",
    ):
        _require_bool(entry, key, label="causal plant authority")
    timing = _require_exact_keys(
        entry["timing_contract"],
        {
            "schema",
            "plant_delays_lead_derived",
            "manual_lead_allowed",
            "lead_samples",
        },
        label="causal plant timing_contract",
    )
    if (
        timing["schema"] != "training_timing_contract_v1"
        or timing["plant_delays_lead_derived"] is not True
        or timing["manual_lead_allowed"] is not False
        or isinstance(timing["lead_samples"], bool)
        or not isinstance(timing["lead_samples"], int)
        or timing["lead_samples"] < 0
    ):
        raise ValueError("causal plant timing contract가 PlantDelays.lead() 기반이 아닙니다")
    _validate_operator(entry["primary_operator"], role="primary", contract=contract)
    _validate_operator(entry["secondary_operator"], role="secondary", contract=contract)


def _validate_population_authority(
    payload: Mapping[str, Any], *, contract: BroadbandFullOctaveContractV3
) -> None:
    expected = {
        "schema",
        "control_band_contract_sha256",
        "population_audit_sha256",
        "manifest_bundle_sha256",
        "canonical_training_eligible",
        "external_manifest_authority_bound",
        "connected_component_authority_bound",
        "interval_alias_authority_bound",
        "actual_raw_manifest_authority_bound",
        "recorded_synthetic_lineage_intersections_zero",
        "required_source_families",
        "required_splits",
        "minimum_independent_components_per_split_family_band",
    }
    entry = _require_exact_keys(payload, expected, label="population authority")
    if entry["schema"] != "full_octave_population_authority_v3":
        raise ValueError("population authority schema가 다릅니다")
    if entry["control_band_contract_sha256"] != contract.digest():
        raise ValueError("population authority v3 contract SHA가 다릅니다")
    _require_sha256(entry["population_audit_sha256"], label="population audit SHA")
    _require_sha256(entry["manifest_bundle_sha256"], label="manifest bundle SHA")
    for key in (
        "canonical_training_eligible",
        "external_manifest_authority_bound",
        "connected_component_authority_bound",
        "interval_alias_authority_bound",
        "actual_raw_manifest_authority_bound",
        "recorded_synthetic_lineage_intersections_zero",
    ):
        _require_bool(entry, key, label="population authority")
    if tuple(entry["required_source_families"]) != _REQUIRED_FAMILIES:
        raise ValueError("population authority source family 순서/집합이 다릅니다")
    if tuple(entry["required_splits"]) != _REQUIRED_SPLITS:
        raise ValueError("population authority split 순서/집합이 다릅니다")
    count = entry["minimum_independent_components_per_split_family_band"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 4:
        raise ValueError("population authority의 independent component 하한은 4 이상이어야 합니다")


def _validate_batch_receipt(
    payload: Mapping[str, Any],
    *,
    contract: BroadbandFullOctaveContractV3,
    population_authority_sha256: str,
) -> None:
    expected = {
        "schema",
        "control_band_contract_sha256",
        "population_authority_sha256",
        "canonical_training_eligible",
        "actual_family_balanced_batch_receipt_consumed",
        "global_sample_index_deterministic",
        "component_uniform_long_run_sampler_proven",
        "batch_size",
        "family_counts",
    }
    entry = _require_exact_keys(payload, expected, label="family-balanced batch receipt")
    if entry["schema"] != "full_octave_family_balanced_batch_receipt_v3":
        raise ValueError("family-balanced batch receipt schema가 다릅니다")
    if entry["control_band_contract_sha256"] != contract.digest():
        raise ValueError("batch receipt v3 contract SHA가 다릅니다")
    if entry["population_authority_sha256"] != population_authority_sha256:
        raise ValueError("batch receipt가 configured population authority bytes와 다릅니다")
    for key in (
        "canonical_training_eligible",
        "actual_family_balanced_batch_receipt_consumed",
        "global_sample_index_deterministic",
        "component_uniform_long_run_sampler_proven",
    ):
        _require_bool(entry, key, label="family-balanced batch receipt")
    size = entry["batch_size"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 4 or size % 4:
        raise ValueError("batch receipt batch_size는 4보다 크고 4의 배수여야 합니다")
    counts = entry["family_counts"]
    # JSON canonicalization은 key를 정렬할 수 있으므로 mapping insertion order를
    # authority로 쓰지 않는다. 네 required family가 정확히 한 번씩 있는지를 본다.
    if not isinstance(counts, Mapping) or frozenset(counts) != frozenset(_REQUIRED_FAMILIES):
        raise ValueError("batch receipt family_counts는 정확한 네 canonical family mapping이어야 합니다")
    expected_count = size // 4
    if any(value != expected_count for value in counts.values()):
        raise ValueError("batch receipt의 네 family quota가 같지 않습니다")


def _validate_gradient_calibration(
    payload: Mapping[str, Any],
    *,
    contract: BroadbandFullOctaveContractV3,
    batch_receipt_sha256: str,
) -> None:
    expected = {
        "schema",
        "control_band_contract_sha256",
        "batch_receipt_sha256",
        "canonical_training_eligible",
        "calibration_pass",
        "actual_causal_secondary_output",
        "output_y_gradient_share",
    }
    entry = _require_exact_keys(payload, expected, label="DNH gradient calibration")
    if entry["schema"] != "full_octave_dnh_gradient_calibration_v3":
        raise ValueError("DNH gradient calibration schema가 다릅니다")
    if entry["control_band_contract_sha256"] != contract.digest():
        raise ValueError("DNH calibration v3 contract SHA가 다릅니다")
    if entry["batch_receipt_sha256"] != batch_receipt_sha256:
        raise ValueError("DNH calibration이 configured batch receipt bytes와 다릅니다")
    for key in (
        "canonical_training_eligible",
        "calibration_pass",
        "actual_causal_secondary_output",
    ):
        _require_bool(entry, key, label="DNH gradient calibration")
    _require_number_between(
        entry,
        "output_y_gradient_share",
        0.2,
        0.4,
        label="DNH gradient calibration",
    )


def _read_config(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _require_exact_keys(
        payload,
        {"schema", "role", "control_band_contract", "artifacts"},
        label="full-octave v3 admission config",
    )
    if root["schema"] != ADMISSION_CONFIG_SCHEMA:
        raise ValueError("full-octave v3 admission config schema가 다릅니다")
    if root["role"] != ADMISSION_ROLE:
        raise ValueError("이 config는 admission-only role이어야 합니다")
    contract = _require_exact_keys(
        root["control_band_contract"], {"id", "sha256"}, label="control_band_contract"
    )
    canonical = BroadbandFullOctaveContractV3.canonical()
    if contract["id"] != canonical.contract_id or contract["sha256"] != canonical.digest():
        raise ValueError("admission config가 exact canonical v3 contract를 가리키지 않습니다")
    artifacts = _require_exact_keys(
        root["artifacts"], set(_ARTIFACT_ROLES), label="artifacts"
    )
    # 모든 role을 여기서 먼저 정규화해 path/sha half-declaration을 fail-closed한다.
    intents = {role: _artifact_intent(artifacts, role) for role in _ARTIFACT_ROLES}
    return root, intents


def is_full_octave_v3_admission_config(payload: Mapping[str, Any]) -> bool:
    """``train.py``가 admission-only config를 조기 거부할 때 쓰는 non-throwing probe."""

    return isinstance(payload, Mapping) and payload.get("schema") == ADMISSION_CONFIG_SCHEMA


def audit_full_octave_v3_admission(
    payload: Mapping[str, Any], *, repo_root: str | Path
) -> dict[str, Any]:
    """artifact bytes를 직접 snapshot해 full-octave admission blockers를 발행한다.

    누락 artifact는 정상적인 ``BLOCKED`` 결과다. malformed config, SHA mismatch,
    symlink 또는 잘못된 future schema는 caller가 무시하지 못하도록 ``ValueError``다.
    """

    _, intents = _read_config(payload)
    root = Path(repo_root).resolve(strict=True)
    if root.is_symlink():
        raise ValueError("repository root가 symlink일 수 없습니다")
    contract = BroadbandFullOctaveContractV3.canonical()
    checks: list[dict[str, Any]] = [
        _check(
            True,
            check_id="canonical_v3_contract",
            detail="125 Hz--8 kHz canonical v3 contract와 digest가 config에 결속됐습니다.",
            control_band_contract_sha256=contract.digest(),
        )
    ]
    snapshots: dict[str, tuple[bytes, str]] = {}
    for role in _ARTIFACT_ROLES:
        path_text, expected_sha = intents[role]
        if path_text is None:
            checks.append(
                _check(
                    False,
                    check_id=role,
                    detail="artifact가 아직 선언되지 않았습니다.",
                )
            )
            continue
        target = _inside_repository(root, path_text, label=f"artifacts.{role}")
        if not target.exists():
            checks.append(
                _check(
                    False,
                    check_id=role,
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
        snapshots[role] = (content, observed_sha)
        checks.append(
            _check(
                True,
                check_id=role,
                detail="regular-file bytes SHA가 config와 일치합니다.",
                path=path_text,
                sha256=observed_sha,
            )
        )

    # raw-first: authority JSON이 raw bytes보다 먼저 또는 별도로 서술될 수 없다.
    raw = snapshots.get("fullband_raw_capture")
    plant = snapshots.get("causal_plant_authority")
    if raw is not None and plant is not None:
        try:
            _validate_causal_plant_authority(
                _read_json_artifact(plant[0], role="causal_plant_authority"),
                contract=contract,
                raw_capture_sha256=raw[1],
            )
        except ValueError as exc:
            raise ValueError(f"causal plant authority 검증 실패: {exc}") from exc
        checks.append(
            _check(
                True,
                check_id="causal_plant_authority_metadata",
                detail="raw-first fullband causal P/S authority metadata가 v3와 결속됐습니다.",
            )
        )
    elif plant is not None:
        checks.append(
            _check(
                False,
                check_id="causal_plant_authority_metadata",
                detail="causal plant authority는 있으나 configured raw capture가 없습니다.",
            )
        )

    population = snapshots.get("population_authority")
    if population is not None:
        try:
            _validate_population_authority(
                _read_json_artifact(population[0], role="population_authority"),
                contract=contract,
            )
        except ValueError as exc:
            raise ValueError(f"population authority 검증 실패: {exc}") from exc
        checks.append(
            _check(
                True,
                check_id="population_authority_metadata",
                detail="external raw manifest·component·lineage zero-intersection metadata가 결속됐습니다.",
            )
        )

    batch = snapshots.get("family_balanced_batch_receipt")
    if batch is not None and population is not None:
        try:
            _validate_batch_receipt(
                _read_json_artifact(batch[0], role="family_balanced_batch_receipt"),
                contract=contract,
                population_authority_sha256=population[1],
            )
        except ValueError as exc:
            raise ValueError(f"family-balanced batch receipt 검증 실패: {exc}") from exc
        checks.append(
            _check(
                True,
                check_id="family_balanced_batch_metadata",
                detail="global-index/component-uniform family-balanced batch가 population bytes에 결속됐습니다.",
            )
        )
    elif batch is not None:
        checks.append(
            _check(
                False,
                check_id="family_balanced_batch_metadata",
                detail="batch receipt는 있으나 configured population authority가 없습니다.",
            )
        )

    calibration = snapshots.get("dnh_gradient_calibration")
    if calibration is not None and batch is not None:
        try:
            _validate_gradient_calibration(
                _read_json_artifact(calibration[0], role="dnh_gradient_calibration"),
                contract=contract,
                batch_receipt_sha256=batch[1],
            )
        except ValueError as exc:
            raise ValueError(f"DNH gradient calibration 검증 실패: {exc}") from exc
        checks.append(
            _check(
                True,
                check_id="dnh_gradient_calibration_metadata",
                detail="actual causal S*y batch의 output-y gradient share가 0.2--0.4에 결속됐습니다.",
            )
        )
    elif calibration is not None:
        checks.append(
            _check(
                False,
                check_id="dnh_gradient_calibration_metadata",
                detail="DNH calibration은 있으나 configured batch receipt가 없습니다.",
            )
        )

    checks.append(
        _check(
            V3_TRAINER_EVALUATOR_CONSUMERS_IMPLEMENTED,
            check_id="v3_trainer_evaluator_consumer_wiring",
            detail=(
                "causal S*y prefix -> v3 7-octave loss -> 같은 P/S/prefix/block의 "
                "matched FxLMS surrogate evaluator가 구현·회귀 검증됐습니다."
                if V3_TRAINER_EVALUATOR_CONSUMERS_IMPLEMENTED
                else "v3 trainer/evaluator consumer가 아직 구현되지 않았습니다."
            ),
        )
    )
    # 이 파일의 정확한 role은 admission-only다. consumer가 구현됐어도 여기서
    # Trainer/DataLoader/GPU를 만들거나 fixture JSON만으로 canonical training을
    # 열 수 없다. 별도 execution envelope preflight는 존재하지만, 이 YAML 자체가
    # 그 receipt를 대체하지 않는다.
    checks.append(
        _check(
            False,
            check_id="v3_raw_bound_execution_config",
            detail=(
                "현재 YAML은 admission-only입니다. 별도 execution envelope의 "
                "non-fixture binding/actual batch receipt/nonce receipt가 없으므로 "
                "canonical execution을 열 수 없습니다."
            ),
        )
    )
    blockers = tuple(check["id"] for check in checks if check["status"] != "PASS")
    return {
        "schema": ADMISSION_RESULT_SCHEMA,
        "status": "READY" if not blockers else "BLOCKED",
        "eligible": False,
        "admission_only": True,
        "audio_opened": False,
        "trainer_constructed": False,
        "gpu_initialized": False,
        "runs_directory_created": False,
        "control_band_contract": contract.model_dump(mode="json"),
        "control_band_contract_sha256": contract.digest(),
        "checks": checks,
        "blockers": blockers,
        "next_required_implementation": (
            "새 full-octave causal P/S·population·batch·DNH artifact가 모두 raw-first로 "
            "발행된 뒤, non-fixture binding과 canonical execution envelope/receipt를 별도 "
            "authority로 발행하고 physical raw G4를 수행해야 합니다."
        ),
    }


def render_full_octave_v3_admission_markdown(report: Mapping[str, Any]) -> str:
    """CLI stdout/기록에 쓸 사람이 읽는 blocked summary."""

    lines = [
        "# Full-octave v3 학습 admission 감사",
        "",
        f"- 상태: `{report['status']}`",
        f"- 학습 허용: `{report['eligible']}`",
        f"- 오디오 open: `{report['audio_opened']}`",
        f"- Trainer/GPU/run 생성: `{report['trainer_constructed']}`/"
        f"`{report['gpu_initialized']}`/`{report['runs_directory_created']}`",
        f"- v3 contract SHA: `{report['control_band_contract_sha256']}`",
        "",
        "## 검사",
        "",
    ]
    for check in report["checks"]:
        lines.append(f"- [{check['status']}] `{check['id']}` — {check['detail']}")
    lines.extend(["", "## 다음 구현", "", str(report["next_required_implementation"]), ""])
    return "\n".join(lines)


__all__ = [
    "ADMISSION_CONFIG_SCHEMA",
    "ADMISSION_RESULT_SCHEMA",
    "ADMISSION_ROLE",
    "V3_TRAINER_EVALUATOR_CONSUMERS_IMPLEMENTED",
    "audit_full_octave_v3_admission",
    "is_full_octave_v3_admission_config",
    "render_full_octave_v3_admission_markdown",
]
