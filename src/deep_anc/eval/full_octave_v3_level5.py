"""Full-octave v3 Level-5 unseen physical challenge의 read-only lifecycle.

이 모듈은 capture/evaluator가 아니다. 오디오 장치, GPU, 네트워크, subprocess를 열거나
파일을 쓰지 않는다. 역할은 다음 future artifact를 *읽기만* 하여 하나의 immutable
identity로 묶는 것이다.

* model-selection 뒤 고정된 checkpoint/controller/experiment contract,
* train/validation/test union과 완전히 분리된 네 family Level-5 source
  (model selection은 validation manifest SHA alias),
* 그 source와 고정 controller를 가리키는 8-input physical raw bundle,
* self-declared single-use capability -> consumed marker -> terminal receipt의 한계.

특히 이 파일은 raw ANC OFF/ON 또는 octave metric을 재계산하지 않는다. 따라서 model
lock, manifest, physical bundle report, terminal ``PASS``가 모두 형식상 맞아도 그 조합은
``BLOCKED_UNATTESTED_*``다. 어떤 입력도 ``canonical_generalization_pass=True`` 또는
independent physical PASS를 만들 수 없다. 실제 evaluator가 raw, spatial quiet-zone,
full-octave/FxLMS metric을 별도로 검증해야 한다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

from ..data.full_octave_v3_physical_bundle import (
    REQUIRED_ROLES as OFFICIAL_PHYSICAL_BUNDLE_REQUIRED_ROLES,
    UNATTESTED_STRUCTURAL_RAW_STATUS,
    load_full_octave_v3_physical_session_bundle,
)
from ..dsp.control_band_contract import (
    BroadbandFullOctaveContractV3,
    REQUIRED_SOURCE_FAMILIES,
)


FULL_OCTAVE_V3_LEVEL5_CONFIG_SCHEMA = "full_octave_v3_level5_lifecycle_config_v1"
FULL_OCTAVE_V3_LEVEL5_REPORT_SCHEMA = "full_octave_v3_level5_lifecycle_report_v1"
FULL_OCTAVE_V3_LEVEL5_ROLE = "post_lock_level5_unseen_physical_challenge_read_only"
FULL_OCTAVE_V3_LEVEL5_MODEL_LOCK_SCHEMA = "full_octave_v3_level5_model_lock_v1"
FULL_OCTAVE_V3_LEVEL5_RAW_MANIFEST_SCHEMA = (
    "full_octave_v3_level5_raw_source_manifest_v1"
)
FULL_OCTAVE_V3_LEVEL5_PHYSICAL_BUNDLE_LOCK_SCHEMA = (
    "full_octave_v3_level5_physical_bundle_lock_v1"
)
FULL_OCTAVE_V3_LEVEL5_CAPABILITY_SCHEMA = (
    "full_octave_v3_level5_one_shot_capability_v1"
)
FULL_OCTAVE_V3_LEVEL5_CONSUMED_SCHEMA = (
    "full_octave_v3_level5_one_shot_consumed_v1"
)
FULL_OCTAVE_V3_LEVEL5_RECEIPT_SCHEMA = (
    "full_octave_v3_level5_one_shot_receipt_v1"
)

# 이 read-only lifecycle은 trust root가 아니다. 특히 JSON의 ``fixture_only=false``,
# self-sealed SHA, O_EXCL 선언은 발행자 자신이 쓴 주장일 뿐 물리 실험 authority를 만들 수
# 없다. independent evaluator가 아직 없으므로 이 모듈/CLI에는 public success predicate도
# 없다. 훗날 opaque verified receipt를 직접 검증하는 별도 evaluator만 success exit을 가질 수
# 있다.
BLOCKED_UNATTESTED_MISSING_AUTHORITY = "BLOCKED_UNATTESTED_MISSING_AUTHORITY"
BLOCKED_UNATTESTED_INVALID_DECLARATION = "BLOCKED_UNATTESTED_INVALID_DECLARATION"
BLOCKED_UNATTESTED_SELF_DECLARED_CHAIN = "BLOCKED_UNATTESTED_SELF_DECLARED_CHAIN"
BLOCKED_UNATTESTED_TERMINAL_RECEIPT = "BLOCKED_UNATTESTED_TERMINAL_RECEIPT"

DEFAULT_CONFIG_RELATIVE_PATH = "configs/full_octave_v3_level5_lifecycle.yaml"
LEVEL5_LEDGER_ROOT = Path("results/full_octave_v3_level5_ledger")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PARTITIONS = ("training", "validation", "test", "selection", "challenge")
_TRAIN_VAL_TEST_PARTITIONS = ("training", "validation", "test")
_CHALLENGE_EXCLUSION_PARTITIONS = _TRAIN_VAL_TEST_PARTITIONS
_FAMILIES = tuple(REQUIRED_SOURCE_FAMILIES)
_PHYSICAL_REPORT_SCHEMA = "full_octave_v3_physical_session_bundle_report_v1"
_PHYSICAL_PLAN_SCHEMA = "full_octave_v3_physical_session_plan_v1"
_PHYSICAL_ROLES = (
    "REF",
    "NOISE_TAP",
    "CANCEL_TAP",
    "ERR_0",
    "ERR_1",
    "ERR_2",
    "ERR_3",
    "ERR_4",
)
_ARTIFACT_ROLES = (
    "model_lock",
    "training_raw_manifest",
    "validation_raw_manifest",
    "test_raw_manifest",
    "selection_raw_manifest",
    "challenge_raw_manifest",
    "physical_bundle_lock",
    "one_shot_capability",
    "one_shot_consumed_marker",
    "one_shot_receipt",
)
_PRIMARY_ARTIFACT_ROLES = _ARTIFACT_ROLES[:7]
_MANIFEST_ROLE_TO_PARTITION = {
    "training_raw_manifest": "training",
    "validation_raw_manifest": "validation",
    "test_raw_manifest": "test",
    "challenge_raw_manifest": "challenge",
}

# 모든 상태에서 남아 있어야 하는 독립 authority blocker다. 아래 JSON schema/bytes 검사는
# 구조적 self-consistency만 볼 수 있으므로, 이 항목 중 하나라도 현재 lifecycle 자체에서
# PASS가 될 수 없다.
_INDEPENDENT_AUTHORITY_BLOCKERS: tuple[tuple[str, str], ...] = (
    (
        "official_physical_bundle_revalidation_8ch_raw_sidecar",
        "physical bundle config SHA를 입력으로 official validator를 다시 실행하고, REF/NOISE_TAP/"
        "CANCEL_TAP/ERR_0..ERR_4 8ch와 native/canonical raw 및 sidecar bytes를 독립 재검산해야 합니다.",
    ),
    (
        "immutable_lineage_inventory_population_reservation",
        "self-declared manifest가 아니라 immutable lineage inventory, population snapshot 및 challenge "
        "reservation을 독립적으로 대조해야 합니다.",
    ),
    (
        "completed_canonical_checkpoint_contract_selection_export_provenance",
        "완료된 canonical checkpoint 내부 experiment contract, 검증된 validation selection 및 export provenance를 "
        "독립적으로 확인해야 합니다.",
    ),
    (
        "submitted_pcm_controller_ps_timing_lead_exact_binding",
        "submitted PCM, controller config/artifact, fullband P/S, timing contract 및 lead의 exact binding을 "
        "raw session마다 독립 대조해야 합니다.",
    ),
    (
        "all_four_family_matched_off_dl_fxlms_campaigns",
        "speech/music/environment/machine 네 family 모두에 대해 동일 조건 OFF/DL/FxLMS matched campaign raw를 "
        "독립 검증해야 합니다.",
    ),
    (
        "dirfd_o_excl_one_shot_issuer_terminal_mutual_exclusion",
        "dirfd 기반 O_EXCL issuer와 terminal issuer의 mutual exclusion/no-replace 역사를 독립적으로 증명해야 합니다; "
        "사후 JSON 선언만으로는 충분하지 않습니다.",
    ),
    (
        "independent_raw_evaluator_receipt",
        "독립 raw evaluator가 OFF/DL/FxLMS raw, five-ERR quiet-zone, full-octave, runtime telemetry를 직접 "
        "재계산한 verified receipt가 필요합니다.",
    ),
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


def _require_exact_keys(
    value: object, expected: set[str] | frozenset[str], *, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{label} key 집합이 정확하지 않습니다: {actual}")
    return dict(value)


def _require_true(value: object, *, label: str) -> None:
    if value is not True:
        raise ValueError(f"{label}=true가 필요합니다")


def _require_false(value: object, *, label: str) -> None:
    if value is not False:
        raise ValueError(f"{label}=false가 필요합니다")


def _require_identity_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 256:
        raise ValueError(f"{label}는 비어 있지 않은 256자 이하 문자열이어야 합니다")
    if any(character in value for character in ("/", "\\", "\x00", "\n", "\r")):
        raise ValueError(f"{label}에 경로 구분자/제어 문자를 사용할 수 없습니다")
    return value


def _inside_repository(root: Path, raw_path: object, *, label: str) -> tuple[str, Path]:
    text = str(raw_path or "")
    candidate = Path(text)
    if not text or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label}.path는 저장소 내부 상대경로여야 합니다")
    target = root / candidate
    cursor = root
    for part in candidate.parts:
        cursor /= part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label}.path에 symlink가 있습니다: {cursor}")
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label}.path가 저장소 밖을 가리킵니다") from exc
    return candidate.as_posix(), target


def _snapshot_regular_file(path: Path, *, label: str) -> tuple[bytes, str, int]:
    """O_NOFOLLOW+fstat 전후 비교로 immutable regular file snapshot을 만든다."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError(f"{label}를 열 수 없습니다: {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label}는 regular file이어야 합니다: {path}")
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
        raise ValueError(f"{label} snapshot 중 파일이 바뀌었습니다: {path}")
    if stat.S_ISLNK(path.lstat().st_mode):
        raise ValueError(f"{label} symlink는 허용하지 않습니다: {path}")
    content = b"".join(chunks)
    if len(content) != int(after.st_size):
        raise ValueError(f"{label} byte 수와 file size가 다릅니다: {path}")
    return content, _sha256_bytes(content), int(after.st_size)


def _load_json(content: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} JSON duplicate key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(content.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}는 UTF-8 JSON object여야 합니다") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON root는 object여야 합니다")
    return payload


def _sealed_payload(payload: Mapping[str, Any], *, evidence_key: str, label: str) -> dict[str, Any]:
    entry = dict(payload)
    evidence = _require_sha256(entry.get(evidence_key), label=f"{label}.{evidence_key}")
    body = {key: value for key, value in entry.items() if key != evidence_key}
    if _sha256_bytes(_canonical_json(body)) != evidence:
        raise ValueError(f"{label}.{evidence_key}가 canonical payload와 다릅니다")
    return entry


def _artifact_intent(
    artifacts: Mapping[str, Any], role: str
) -> tuple[str | None, str | None]:
    entry = _require_exact_keys(
        artifacts[role], {"path", "sha256"}, label=f"artifacts.{role}"
    )
    path, digest = entry["path"], entry["sha256"]
    if (path is None) != (digest is None):
        raise ValueError(f"artifacts.{role}.path와 sha256은 함께 null이거나 함께 필요합니다")
    if path is None:
        return None, None
    if not isinstance(path, str):
        raise ValueError(f"artifacts.{role}.path는 string 또는 null이어야 합니다")
    return path, _require_sha256(digest, label=f"artifacts.{role}.sha256")


def _snapshot_declared_artifact(
    *, root: Path, path: str, expected_sha256: str, label: str
) -> tuple[str, bytes, str, int]:
    relative, target = _inside_repository(root, path, label=label)
    content, observed_sha, size = _snapshot_regular_file(target, label=label)
    if observed_sha != expected_sha256:
        raise ValueError(
            f"{label} bytes SHA가 config와 다릅니다: expected={expected_sha256}, observed={observed_sha}"
        )
    return relative, content, observed_sha, size


def _verify_file_reference(
    value: object, *, root: Path, label: str
) -> dict[str, Any]:
    entry = _require_exact_keys(value, {"path", "size_bytes", "sha256"}, label=label)
    if isinstance(entry["size_bytes"], bool) or not isinstance(entry["size_bytes"], int):
        raise ValueError(f"{label}.size_bytes는 bool 아닌 int여야 합니다")
    if int(entry["size_bytes"]) <= 0:
        raise ValueError(f"{label}.size_bytes는 양수여야 합니다")
    expected_sha = _require_sha256(entry["sha256"], label=f"{label}.sha256")
    relative, target = _inside_repository(root, entry["path"], label=label)
    _content, observed_sha, size = _snapshot_regular_file(target, label=label)
    if observed_sha != expected_sha or size != int(entry["size_bytes"]):
        raise ValueError(f"{label}의 path/size/SHA가 실제 bytes와 다릅니다")
    return {"path": relative, "size_bytes": size, "sha256": observed_sha}


def _read_config(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, tuple[str | None, str | None]]]:
    root = _require_exact_keys(
        payload,
        {"schema", "role", "control_band_contract", "artifacts"},
        label="full-octave v3 Level-5 lifecycle config",
    )
    if root["schema"] != FULL_OCTAVE_V3_LEVEL5_CONFIG_SCHEMA:
        raise ValueError("Level-5 lifecycle config schema가 다릅니다")
    if root["role"] != FULL_OCTAVE_V3_LEVEL5_ROLE:
        raise ValueError("Level-5 lifecycle config role이 다릅니다")
    contract = _require_exact_keys(
        root["control_band_contract"], {"id", "sha256"}, label="control_band_contract"
    )
    canonical = BroadbandFullOctaveContractV3.canonical()
    if contract["id"] != canonical.contract_id or contract["sha256"] != canonical.digest():
        raise ValueError("Level-5 lifecycle은 exact canonical full-octave v3 contract만 허용합니다")
    artifacts = _require_exact_keys(
        root["artifacts"], set(_ARTIFACT_ROLES), label="Level-5 lifecycle artifacts"
    )
    return root, {role: _artifact_intent(artifacts, role) for role in _ARTIFACT_ROLES}


def _validate_experiment_contract(
    content: bytes, *, expected_sha256: str, label: str
) -> dict[str, Any]:
    contract = _load_json(content, label=label)
    if set(contract) != {
        "schema_version",
        "config_sha256",
        "source",
        "input_generation",
        "artifacts",
        "sha256",
    }:
        raise ValueError(f"{label} key 집합이 현재 experiment contract와 다릅니다")
    if contract.get("schema_version") != 2:
        raise ValueError(f"{label}.schema_version=2가 필요합니다")
    body = {key: value for key, value in contract.items() if key != "sha256"}
    embedded = _require_sha256(contract.get("sha256"), label=f"{label}.sha256")
    if embedded != _sha256_bytes(_canonical_json(body)) or embedded != expected_sha256:
        raise ValueError(f"{label} embedded/declared experiment contract SHA가 다릅니다")
    source = contract.get("source")
    if not isinstance(source, Mapping):
        raise ValueError(f"{label}.source가 없습니다")
    _require_true(source.get("verifiable"), label=f"{label}.source.verifiable")
    _require_true(source.get("clean_exact_commit"), label=f"{label}.source.clean_exact_commit")
    _require_sha256(source.get("source_tree_sha256"), label=f"{label}.source.source_tree_sha256")
    if not isinstance(source.get("git_commit"), str) or not source["git_commit"].strip():
        raise ValueError(f"{label}.source.git_commit이 없습니다")
    return contract


def _validate_model_lock(
    payload: Mapping[str, Any], *, root: Path, validation_manifest_sha256: str
) -> dict[str, Any]:
    expected = {
        "schema",
        "role",
        "fixture_only",
        "control_band_contract_sha256",
        "checkpoint",
        "controller_artifact",
        "experiment_contract",
        "experiment_contract_sha256",
        "selection_manifest_sha256",
        "canonical_model_frozen",
        "selection_finalized",
        "model_lock_evidence_sha256",
    }
    entry = _sealed_payload(
        _require_exact_keys(payload, expected, label="Level-5 model lock"),
        evidence_key="model_lock_evidence_sha256",
        label="Level-5 model lock",
    )
    if entry["schema"] != FULL_OCTAVE_V3_LEVEL5_MODEL_LOCK_SCHEMA:
        raise ValueError("Level-5 model lock schema가 다릅니다")
    if entry["role"] != "frozen_canonical_model_after_val_selection":
        raise ValueError("Level-5 model lock role이 val-selection 이후 freeze가 아닙니다")
    _require_false(entry["fixture_only"], label="Level-5 model lock.fixture_only")
    contract = BroadbandFullOctaveContractV3.canonical()
    if entry["control_band_contract_sha256"] != contract.digest():
        raise ValueError("Level-5 model lock v3 contract SHA가 다릅니다")
    checkpoint = _verify_file_reference(entry["checkpoint"], root=root, label="model_lock.checkpoint")
    controller = _verify_file_reference(
        entry["controller_artifact"], root=root, label="model_lock.controller_artifact"
    )
    experiment = _verify_file_reference(
        entry["experiment_contract"], root=root, label="model_lock.experiment_contract"
    )
    experiment_sha = _require_sha256(
        entry["experiment_contract_sha256"], label="model_lock.experiment_contract_sha256"
    )
    _validate_experiment_contract(
        _snapshot_regular_file(root / experiment["path"], label="model_lock.experiment_contract")[0],
        expected_sha256=experiment_sha,
        label="model_lock.experiment_contract",
    )
    if entry["selection_manifest_sha256"] != validation_manifest_sha256:
        raise ValueError("model lock selection manifest SHA가 validation manifest SHA alias와 다릅니다")
    _require_true(entry["canonical_model_frozen"], label="model_lock.canonical_model_frozen")
    _require_true(entry["selection_finalized"], label="model_lock.selection_finalized")
    return {
        "checkpoint": checkpoint,
        "controller_artifact": controller,
        "experiment_contract": experiment,
        "experiment_contract_sha256": experiment_sha,
    }


def _validate_source_record(
    value: object, *, root: Path, label: str
) -> dict[str, Any]:
    entry = _require_exact_keys(
        value,
        {
            "record_id",
            "source_family",
            "source_ids",
            "lineage_component_id",
            "lineage_keys",
            "native_source",
            "decoded_pcm",
        },
        label=label,
    )
    record_id = _require_identity_text(entry["record_id"], label=f"{label}.record_id")
    family = _require_identity_text(entry["source_family"], label=f"{label}.source_family")
    if family not in _FAMILIES:
        raise ValueError(f"{label}.source_family가 required family가 아닙니다: {family!r}")
    component = _require_identity_text(
        entry["lineage_component_id"], label=f"{label}.lineage_component_id"
    )
    source_values = entry["source_ids"]
    lineage_values = entry["lineage_keys"]
    if not isinstance(source_values, Sequence) or isinstance(source_values, (str, bytes)):
        raise ValueError(f"{label}.source_ids는 비어 있지 않은 string list여야 합니다")
    if not isinstance(lineage_values, Sequence) or isinstance(lineage_values, (str, bytes)):
        raise ValueError(f"{label}.lineage_keys는 비어 있지 않은 string list여야 합니다")
    source_ids = tuple(
        _require_identity_text(item, label=f"{label}.source_ids[{index}]")
        for index, item in enumerate(source_values)
    )
    lineage_keys = tuple(
        _require_identity_text(item, label=f"{label}.lineage_keys[{index}]")
        for index, item in enumerate(lineage_values)
    )
    if not source_ids or not lineage_keys or len(set(source_ids)) != len(source_ids) or len(set(lineage_keys)) != len(lineage_keys):
        raise ValueError(f"{label} source_ids/lineage_keys는 비어 있지 않고 중복이 없어야 합니다")
    return {
        "record_id": record_id,
        "source_family": family,
        "source_ids": source_ids,
        "lineage_component_id": component,
        "lineage_keys": lineage_keys,
        "native_source": _verify_file_reference(
            entry["native_source"], root=root, label=f"{label}.native_source"
        ),
        "decoded_pcm": _verify_file_reference(
            entry["decoded_pcm"], root=root, label=f"{label}.decoded_pcm"
        ),
    }


def _validate_raw_source_manifest(
    payload: Mapping[str, Any], *, root: Path, partition: str, model_lock_sha256: str | None
) -> list[dict[str, Any]]:
    expected = {
        "schema",
        "role",
        "partition",
        "fixture_only",
        "control_band_contract_sha256",
        "raw_manifest_authority",
        "artifact_bytes_verified",
        "source_identity_complete",
        "model_lock_sha256",
        "records",
        "manifest_evidence_sha256",
    }
    entry = _sealed_payload(
        _require_exact_keys(payload, expected, label=f"Level-5 {partition} raw manifest"),
        evidence_key="manifest_evidence_sha256",
        label=f"Level-5 {partition} raw manifest",
    )
    if entry["schema"] != FULL_OCTAVE_V3_LEVEL5_RAW_MANIFEST_SCHEMA:
        raise ValueError(f"Level-5 {partition} raw manifest schema가 다릅니다")
    if entry["role"] != "immutable_raw_source_identity_manifest":
        raise ValueError(f"Level-5 {partition} raw manifest role이 다릅니다")
    if entry["partition"] != partition:
        raise ValueError(f"Level-5 raw manifest partition이 config role과 다릅니다")
    _require_false(entry["fixture_only"], label=f"Level-5 {partition} raw manifest.fixture_only")
    if entry["control_band_contract_sha256"] != BroadbandFullOctaveContractV3.canonical().digest():
        raise ValueError(f"Level-5 {partition} raw manifest v3 contract SHA가 다릅니다")
    for key in ("raw_manifest_authority", "artifact_bytes_verified", "source_identity_complete"):
        _require_true(entry[key], label=f"Level-5 {partition} raw manifest.{key}")
    declared_lock = entry["model_lock_sha256"]
    if partition == "challenge":
        if model_lock_sha256 is None or declared_lock != model_lock_sha256:
            raise ValueError("Level-5 challenge raw manifest는 model lock bytes SHA에 결속돼야 합니다")
    elif declared_lock is not None:
        raise ValueError(f"Level-5 {partition} raw manifest에는 사후 model lock SHA를 넣을 수 없습니다")
    records_value = entry["records"]
    if not isinstance(records_value, list) or not records_value:
        raise ValueError(f"Level-5 {partition} raw manifest.records가 비었습니다")
    records = [
        _validate_source_record(item, root=root, label=f"Level-5 {partition} record[{index}]")
        for index, item in enumerate(records_value)
    ]
    if len(records) != len({item["record_id"] for item in records}):
        raise ValueError(f"Level-5 {partition} raw manifest record_id가 중복됐습니다")
    families = {item["source_family"] for item in records}
    if families != set(_FAMILIES):
        raise ValueError(f"Level-5 {partition} raw manifest가 네 family를 모두 포함하지 않습니다")
    return records


def _all_identity_sets(records: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    return {
        "source_ids": {source_id for item in records for source_id in item["source_ids"]},
        "lineage_component_ids": {str(item["lineage_component_id"]) for item in records},
        "lineage_keys": {lineage for item in records for lineage in item["lineage_keys"]},
        "native_source_sha256": {str(item["native_source"]["sha256"]) for item in records},
        "decoded_pcm_sha256": {str(item["decoded_pcm"]["sha256"]) for item in records},
    }


def _validate_global_identity_consistency(
    partition_records: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    """한 identity를 다른 family로 위장하는 source alias를 거부한다."""

    ownership: dict[tuple[str, str], str] = {}
    for partition, records in partition_records.items():
        for item in records:
            family = str(item["source_family"])
            values: dict[str, Sequence[str]] = {
                "source_id": item["source_ids"],
                "lineage_component_id": (str(item["lineage_component_id"]),),
                "lineage_key": item["lineage_keys"],
                "native_source_sha256": (str(item["native_source"]["sha256"]),),
                "decoded_pcm_sha256": (str(item["decoded_pcm"]["sha256"]),),
            }
            for role, members in values.items():
                for member in members:
                    key = (role, member)
                    previous = ownership.setdefault(key, family)
                    if previous != family:
                        raise ValueError(
                            f"{role}={member!r}가 서로 다른 family에 재사용됐습니다: "
                            f"{previous!r} vs {family!r} ({partition})"
                        )


def _validate_selection_validation_alias(
    snapshots: Mapping[str, tuple[str, bytes, str, int]]
) -> dict[str, str | bool]:
    """model selection은 별도 split이 아니라 recorded validation의 exact SHA alias다."""

    validation = snapshots["validation_raw_manifest"]
    selection = snapshots["selection_raw_manifest"]
    if selection[2] != validation[2]:
        raise ValueError(
            "selection_raw_manifest는 validation_raw_manifest와 exact same SHA여야 합니다; "
            "별도 selection split은 허용하지 않습니다"
        )
    return {
        "policy": "selection_is_exact_validation_manifest_sha_alias",
        "validation_path": validation[0],
        "selection_path": selection[0],
        "validation_sha256": validation[2],
        "selection_sha256": selection[2],
        "exact_sha_match": True,
    }


def _level5_exclusion(
    partition_records: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, dict[str, dict[str, int | bool]]]:
    challenge = partition_records["challenge"]
    result: dict[str, dict[str, dict[str, int | bool]]] = {}
    for family in _FAMILIES:
        family_records = [item for item in challenge if item["source_family"] == family]
        if not family_records:
            raise ValueError(f"Level-5 challenge에 {family} family source가 없습니다")
        candidate = _all_identity_sets(family_records)
        per_partition: dict[str, dict[str, int | bool]] = {}
        for partition in _CHALLENGE_EXCLUSION_PARTITIONS:
            known = _all_identity_sets(partition_records[partition])
            counts = {
                identity: len(candidate[identity] & known[identity])
                for identity in candidate
            }
            passed = not any(counts.values())
            if not passed:
                raise ValueError(
                    f"Level-5 {family} challenge가 {partition} source와 lineage/bytes/source ID가 겹칩니다: {counts}"
                )
            per_partition[partition] = {**counts, "passed": True}
        result[family] = per_partition
    return result


def _base_split_pairwise_leakage(
    partition_records: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, dict[str, int | bool]]:
    """학습/검증/테스트의 identity 교집합을 직접 계산한다.

    challenge exclusion만 0이라고 train/val/test leakage가 사라지는 것은 아니다. selection은
    validation SHA alias이므로 별도 pair가 아니다. 이 값은
    declared bytes를 기준으로 한 구조 검사일 뿐, 독립 lineage inventory authority는
    아니다. 교집합이 있으면 lifecycle은 즉시 거부한다.
    """

    result: dict[str, dict[str, int | bool]] = {}
    for left, right in combinations(_TRAIN_VAL_TEST_PARTITIONS, 2):
        left_identity = _all_identity_sets(partition_records[left])
        right_identity = _all_identity_sets(partition_records[right])
        counts = {
            identity: len(left_identity[identity] & right_identity[identity])
            for identity in left_identity
        }
        passed = not any(counts.values())
        pair = f"{left}__{right}"
        result[pair] = {**counts, "passed": passed}
        if not passed:
            raise ValueError(
                "Level-5 base split pairwise leakage가 있습니다: "
                f"{pair}={counts}"
            )
    return result


def _challenge_preregistration_limitations() -> dict[str, object]:
    """현재 checker가 소급해 증명할 수 없는 Level-5 예약 한계를 명시한다."""

    return {
        "independent_reservation_verified": False,
        "challenge_created_before_result_access_proven": False,
        "challenge_absent_from_immutable_global_inventory_proven": False,
        "limitation": (
            "현재 manifest/model-lock SHA는 분석 시점의 self-attested bytes만 결속합니다. "
            "challenge가 결과 접근 전 독립적으로 예약됐는지, 전체 immutable lineage inventory와 "
            "population에서 실제 제외됐는지는 별도 authority 없이는 소급 증명할 수 없습니다."
        ),
    }


def _unattested_authority_checks() -> list[dict[str, Any]]:
    """self-attested chain으로 닫을 수 없는 필수 authority를 항상 BLOCKED로 기록한다."""

    return [
        _check(False, check_id=check_id, detail=detail)
        for check_id, detail in _INDEPENDENT_AUTHORITY_BLOCKERS
    ]


def _rerun_official_physical_bundle_validator(
    report: Mapping[str, Any],
    *,
    root: Path,
    report_canonical_raw_sha256: str,
    report_sidecar_sha256: str,
) -> dict[str, str]:
    """report 자체가 아니라 config SHA로 official 8-input checker를 다시 실행한다.

    이것도 physical authority가 아니라 self-consistency를 한 단계 더 확인하는 것뿐이다.
    그러나 최소 report JSON을 손으로 만들어 Level-5 model lock에 연결하는 우회는 여기서
    차단한다. 이 helper는 file read만 수행하며 audio/GPU/network를 열지 않는다.
    """

    config = _require_exact_keys(
        report.get("config"), {"path", "file_sha256"}, label="physical bundle report.config"
    )
    config_text = str(config["path"] or "")
    config_candidate = Path(config_text)
    if not config_text or ".." in config_candidate.parts:
        raise ValueError("physical bundle report.config.path가 안전한 repository path가 아닙니다")
    config_target = config_candidate if config_candidate.is_absolute() else root / config_candidate
    try:
        config_target.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise ValueError("physical bundle report.config.path가 repository 밖을 가리킵니다") from exc
    _config_content, config_sha, _config_size = _snapshot_regular_file(
        config_target, label="physical bundle report.config"
    )
    if config_sha != _require_sha256(
        config["file_sha256"], label="physical bundle report.config.file_sha256"
    ):
        raise ValueError("physical bundle report.config SHA가 actual config bytes와 다릅니다")
    try:
        official = load_full_octave_v3_physical_session_bundle(
            config_target, repository_root=root
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"official physical bundle validator 재실행이 실패했습니다: {exc}") from exc
    if (
        official.get("status") != UNATTESTED_STRUCTURAL_RAW_STATUS
        or official.get("declared_sha_structure_valid") is not True
        or official.get("physical_raw_provenance_attested") is not False
    ):
        raise ValueError(
            "official physical bundle validator가 unattested 8-input raw의 declared SHA structure를 "
            "재검산하지 못했습니다"
        )
    if tuple(OFFICIAL_PHYSICAL_BUNDLE_REQUIRED_ROLES) != _PHYSICAL_ROLES:
        raise ValueError("official physical bundle required-role contract가 Level-5 contract와 다릅니다")
    if tuple(official.get("required_roles", ())) != _PHYSICAL_ROLES:
        raise ValueError("official physical bundle validator의 required roles가 8-input contract와 다릅니다")
    official_config = official.get("config")
    if not isinstance(official_config, Mapping) or official_config.get("file_sha256") != config_sha:
        raise ValueError("official physical bundle validator가 같은 config SHA를 재확인하지 못했습니다")
    official_raw = official.get("canonical_raw")
    official_sidecar = official.get("session_sidecar")
    if not isinstance(official_raw, Mapping) or official_raw.get("sha256") != report_canonical_raw_sha256:
        raise ValueError("official physical bundle validator canonical raw SHA가 Level-5 report와 다릅니다")
    if not isinstance(official_sidecar, Mapping) or official_sidecar.get("file_sha256") != report_sidecar_sha256:
        raise ValueError("official physical bundle validator sidecar SHA가 Level-5 report와 다릅니다")
    return {
        "config_sha256": config_sha,
        "canonical_raw_sha256": report_canonical_raw_sha256,
        "session_sidecar_sha256": report_sidecar_sha256,
    }


def _validate_physical_bundle_lock(
    payload: Mapping[str, Any],
    *,
    root: Path,
    model_lock_sha256: str,
    model_controller_sha256: str,
    experiment_contract_sha256: str,
    challenge_manifest_sha256: str,
) -> dict[str, Any]:
    expected = {
        "schema",
        "role",
        "fixture_only",
        "control_band_contract_sha256",
        "model_lock_sha256",
        "physical_bundle_report",
        "required_roles",
        "physical_bundle_lock_evidence_sha256",
    }
    entry = _sealed_payload(
        _require_exact_keys(payload, expected, label="Level-5 physical bundle lock"),
        evidence_key="physical_bundle_lock_evidence_sha256",
        label="Level-5 physical bundle lock",
    )
    if entry["schema"] != FULL_OCTAVE_V3_LEVEL5_PHYSICAL_BUNDLE_LOCK_SCHEMA:
        raise ValueError("Level-5 physical bundle lock schema가 다릅니다")
    if entry["role"] != "frozen_eight_input_bundle_for_level5_one_shot":
        raise ValueError("Level-5 physical bundle lock role이 다릅니다")
    _require_false(entry["fixture_only"], label="Level-5 physical bundle lock.fixture_only")
    if entry["control_band_contract_sha256"] != BroadbandFullOctaveContractV3.canonical().digest():
        raise ValueError("Level-5 physical bundle lock v3 contract SHA가 다릅니다")
    if entry["model_lock_sha256"] != model_lock_sha256:
        raise ValueError("Level-5 physical bundle lock model lock SHA가 다릅니다")
    if tuple(entry["required_roles"]) != _PHYSICAL_ROLES:
        raise ValueError("Level-5 physical bundle lock eight-input role 순서가 다릅니다")
    report_ref = _verify_file_reference(
        entry["physical_bundle_report"], root=root, label="physical_bundle_lock.physical_bundle_report"
    )
    report_content, report_sha, _size = _snapshot_regular_file(
        root / report_ref["path"], label="physical_bundle_lock.physical_bundle_report"
    )
    if report_sha != report_ref["sha256"]:
        raise ValueError("physical bundle report SHA가 바뀌었습니다")
    report = _load_json(report_content, label="physical_bundle_lock.physical_bundle_report")
    if report.get("schema") != _PHYSICAL_REPORT_SCHEMA:
        raise ValueError("physical bundle report schema가 다릅니다")
    if (
        report.get("status") != UNATTESTED_STRUCTURAL_RAW_STATUS
        or report.get("declared_sha_structure_valid") is not True
        or report.get("physical_raw_provenance_attested") is not False
    ):
        raise ValueError(
            "Level-5 physical bundle은 physical authority가 아닌 unattested declared SHA structure여야 합니다"
        )
    if report.get("fixture_only_evidence") is True:
        raise ValueError("fixture-only physical bundle은 Level-5에 사용할 수 없습니다")
    if report.get("control_band_contract_sha256") != BroadbandFullOctaveContractV3.canonical().digest():
        raise ValueError("physical bundle report v3 contract SHA가 다릅니다")
    if tuple(report.get("required_roles", ())) != _PHYSICAL_ROLES:
        raise ValueError("physical bundle report required roles가 eight-input contract와 다릅니다")
    if report.get("canonical_training_eligible") is not False or report.get("deployment_eligible") is not False:
        raise ValueError("physical bundle structural report는 canonical/deployment authority를 주장할 수 없습니다")
    capture = report.get("capture_plan")
    if not isinstance(capture, Mapping):
        raise ValueError("physical bundle report capture_plan이 없습니다")
    plan_path = capture.get("path")
    plan_sha = _require_sha256(capture.get("file_sha256"), label="physical bundle capture plan SHA")
    _relative, plan_target = _inside_repository(root, plan_path, label="physical bundle capture plan")
    plan_content, observed_plan_sha, _plan_size = _snapshot_regular_file(
        plan_target, label="physical bundle capture plan"
    )
    if observed_plan_sha != plan_sha:
        raise ValueError("physical bundle capture plan SHA가 report와 다릅니다")
    plan = _load_json(plan_content, label="physical bundle capture plan")
    if plan.get("schema") != _PHYSICAL_PLAN_SCHEMA or plan.get("fixture_only") is not False:
        raise ValueError("Level-5 physical capture plan은 non-fixture exact plan이어야 합니다")
    if plan.get("control_band_contract_sha256") != BroadbandFullOctaveContractV3.canonical().digest():
        raise ValueError("physical capture plan v3 contract SHA가 다릅니다")
    identity = plan.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("physical capture plan identity가 없습니다")
    source = identity.get("source")
    controller = identity.get("controller")
    if not isinstance(source, Mapping) or not isinstance(controller, Mapping):
        raise ValueError("physical capture plan source/controller identity가 없습니다")
    if source.get("source_manifest_sha256") != challenge_manifest_sha256:
        raise ValueError("physical capture plan이 exact Level-5 challenge raw manifest를 가리키지 않습니다")
    if controller.get("controller_artifact_sha256") != model_controller_sha256:
        raise ValueError("physical capture plan controller artifact가 frozen model과 다릅니다")
    canonical_raw = _verify_file_reference(
        report.get("canonical_raw"), root=root, label="physical bundle canonical_raw"
    )
    sidecar_entry = _require_exact_keys(
        report.get("session_sidecar"),
        {"path", "file_sha256", "evidence_sha256"},
        label="physical bundle session_sidecar",
    )
    sidecar_relative, sidecar_target = _inside_repository(
        root, sidecar_entry["path"], label="physical bundle session_sidecar"
    )
    _sidecar_content, sidecar_sha, _sidecar_size = _snapshot_regular_file(
        sidecar_target, label="physical bundle session_sidecar"
    )
    if sidecar_sha != _require_sha256(
        sidecar_entry["file_sha256"], label="physical bundle sidecar SHA"
    ):
        raise ValueError("physical bundle session sidecar SHA가 report와 다릅니다")
    _require_sha256(
        sidecar_entry["evidence_sha256"], label="physical bundle sidecar evidence SHA"
    )
    official_bundle = _rerun_official_physical_bundle_validator(
        report,
        root=root,
        report_canonical_raw_sha256=canonical_raw["sha256"],
        report_sidecar_sha256=sidecar_sha,
    )
    # physical plan config SHA may represent a runtime config rather than the experiment
    # contract itself. The Level-5 lock still binds both contract and controller through
    # its model lock; require the contract digest to be present in the plan source metadata
    # only when the publisher exposes it is intentionally not assumed here.
    _require_sha256(experiment_contract_sha256, label="frozen experiment contract SHA")
    return {
        "report": report_ref,
        "official_bundle_config_sha256": official_bundle["config_sha256"],
        "capture_plan_sha256": observed_plan_sha,
        "canonical_raw_sha256": canonical_raw["sha256"],
        "session_sidecar_sha256": sidecar_sha,
        "session_sidecar_path": sidecar_relative,
    }


def _challenge_input_payload(
    *,
    root_payload: Mapping[str, Any],
    artifact_paths: Mapping[str, str],
    artifact_sha256: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": "full_octave_v3_level5_challenge_input_v1",
        "role": FULL_OCTAVE_V3_LEVEL5_ROLE,
        "control_band_contract": dict(root_payload["control_band_contract"]),
        "model_lock": {
            "path": artifact_paths["model_lock"],
            "sha256": artifact_sha256["model_lock"],
        },
        "source_manifests": {
            partition: {
                "path": artifact_paths[f"{partition}_raw_manifest"],
                "sha256": artifact_sha256[f"{partition}_raw_manifest"],
            }
            for partition in _PARTITIONS
        },
        "physical_bundle_lock": {
            "path": artifact_paths["physical_bundle_lock"],
            "sha256": artifact_sha256["physical_bundle_lock"],
        },
    }


def canonical_level5_ledger_paths(
    challenge_input_sha256: str, *, repo_root: str | Path
) -> dict[str, Path]:
    """Level-5 input digest 하나에만 허용되는 one-shot ledger 경로."""

    digest = _require_sha256(challenge_input_sha256, label="Level-5 challenge input SHA")
    root = Path(repo_root).resolve(strict=True)
    directory = root / LEVEL5_LEDGER_ROOT / digest
    return {
        "issued": directory / "capability.json",
        "running": directory / "consumed.json",
        "completed": directory / "completed.json",
        "failed": directory / "failed.json",
    }


def _require_expected_ledger_path(
    actual_relative: str, expected: Path, *, root: Path, label: str
) -> None:
    _relative, target = _inside_repository(root, actual_relative, label=label)
    if target != expected:
        raise ValueError(f"{label}는 canonical Level-5 ledger 경로만 허용합니다: expected={expected}")


def _validate_capability(
    payload: Mapping[str, Any],
    *,
    challenge_input_sha256: str,
    model_lock_sha256: str,
    physical_bundle_lock_sha256: str,
    manifests: Mapping[str, str],
) -> str:
    expected = {
        "schema",
        "role",
        "phase",
        "fixture_only",
        "control_band_contract_sha256",
        "challenge_input_sha256",
        "model_lock_sha256",
        "physical_bundle_lock_sha256",
        "source_manifest_sha256",
        "token_sha256",
        "no_replace_declared",
        "capability_evidence_sha256",
    }
    entry = _sealed_payload(
        _require_exact_keys(payload, expected, label="Level-5 one-shot capability"),
        evidence_key="capability_evidence_sha256",
        label="Level-5 one-shot capability",
    )
    if (
        entry["schema"] != FULL_OCTAVE_V3_LEVEL5_CAPABILITY_SCHEMA
        or entry["role"] != "issued_level5_physical_evaluation_capability"
        or entry["phase"] != "issued"
    ):
        raise ValueError("Level-5 one-shot capability schema/role/phase가 다릅니다")
    _require_false(entry["fixture_only"], label="Level-5 one-shot capability.fixture_only")
    _require_true(entry["no_replace_declared"], label="Level-5 one-shot capability.no_replace_declared")
    if entry["control_band_contract_sha256"] != BroadbandFullOctaveContractV3.canonical().digest():
        raise ValueError("Level-5 one-shot capability v3 contract SHA가 다릅니다")
    for key, expected_sha in (
        ("challenge_input_sha256", challenge_input_sha256),
        ("model_lock_sha256", model_lock_sha256),
        ("physical_bundle_lock_sha256", physical_bundle_lock_sha256),
    ):
        if entry[key] != expected_sha:
            raise ValueError(f"Level-5 one-shot capability {key}가 frozen input과 다릅니다")
    if not isinstance(entry["source_manifest_sha256"], Mapping):
        raise ValueError("Level-5 one-shot capability source manifest map이 없습니다")
    if dict(entry["source_manifest_sha256"]) != dict(manifests):
        raise ValueError("Level-5 one-shot capability source manifest SHA map이 다릅니다")
    return _require_sha256(entry["token_sha256"], label="Level-5 one-shot capability.token_sha256")


def _validate_consumed_marker(
    payload: Mapping[str, Any],
    *,
    capability_sha256: str,
    challenge_input_sha256: str,
    model_lock_sha256: str,
    physical_bundle_lock_sha256: str,
) -> None:
    expected = {
        "schema",
        "role",
        "phase",
        "fixture_only",
        "capability_sha256",
        "challenge_input_sha256",
        "model_lock_sha256",
        "physical_bundle_lock_sha256",
        "no_replace_declared",
        "consumed_evidence_sha256",
    }
    entry = _sealed_payload(
        _require_exact_keys(payload, expected, label="Level-5 one-shot consumed marker"),
        evidence_key="consumed_evidence_sha256",
        label="Level-5 one-shot consumed marker",
    )
    if (
        entry["schema"] != FULL_OCTAVE_V3_LEVEL5_CONSUMED_SCHEMA
        or entry["role"] != "consumed_level5_physical_evaluation_capability"
        or entry["phase"] != "running"
    ):
        raise ValueError("Level-5 consumed marker schema/role/phase가 다릅니다")
    _require_false(entry["fixture_only"], label="Level-5 consumed marker.fixture_only")
    _require_true(entry["no_replace_declared"], label="Level-5 consumed marker.no_replace_declared")
    for key, expected_sha in (
        ("capability_sha256", capability_sha256),
        ("challenge_input_sha256", challenge_input_sha256),
        ("model_lock_sha256", model_lock_sha256),
        ("physical_bundle_lock_sha256", physical_bundle_lock_sha256),
    ):
        if entry[key] != expected_sha:
            raise ValueError(f"Level-5 consumed marker {key}가 immutable ledger와 다릅니다")


def _validate_terminal_receipt(
    payload: Mapping[str, Any],
    *,
    root: Path,
    capability_sha256: str,
    consumed_marker_sha256: str,
    challenge_input_sha256: str,
    model_lock_sha256: str,
    physical_bundle_lock_sha256: str,
) -> str:
    expected = {
        "schema",
        "role",
        "phase",
        "fixture_only",
        "evaluation_domain",
        "verdict",
        "capability_sha256",
        "consumed_marker_sha256",
        "challenge_input_sha256",
        "model_lock_sha256",
        "physical_bundle_lock_sha256",
        "raw_evaluation_bundle",
        "metrics",
        "evaluator_receipt",
        "no_replace_declared",
        "receipt_evidence_sha256",
    }
    entry = _sealed_payload(
        _require_exact_keys(payload, expected, label="Level-5 one-shot terminal receipt"),
        evidence_key="receipt_evidence_sha256",
        label="Level-5 one-shot terminal receipt",
    )
    if entry["schema"] != FULL_OCTAVE_V3_LEVEL5_RECEIPT_SCHEMA:
        raise ValueError("Level-5 terminal receipt schema가 다릅니다")
    if entry["role"] != "terminal_level5_physical_evaluation_receipt":
        raise ValueError("Level-5 terminal receipt role이 다릅니다")
    _require_false(entry["fixture_only"], label="Level-5 terminal receipt.fixture_only")
    _require_true(entry["no_replace_declared"], label="Level-5 terminal receipt.no_replace_declared")
    if entry["evaluation_domain"] != "physical_duct_level5_one_shot":
        raise ValueError("Level-5 terminal receipt evaluation domain이 physical duct one-shot이 아닙니다")
    verdict = entry["verdict"]
    if verdict not in {"PASS", "FAIL", "INCONCLUSIVE"}:
        raise ValueError("Level-5 terminal receipt verdict가 승인 enum이 아닙니다")
    expected_phase = "completed" if verdict == "PASS" else "failed"
    if entry["phase"] != expected_phase:
        raise ValueError("Level-5 terminal receipt phase와 verdict가 모순됩니다")
    for key, expected_sha in (
        ("capability_sha256", capability_sha256),
        ("consumed_marker_sha256", consumed_marker_sha256),
        ("challenge_input_sha256", challenge_input_sha256),
        ("model_lock_sha256", model_lock_sha256),
        ("physical_bundle_lock_sha256", physical_bundle_lock_sha256),
    ):
        if entry[key] != expected_sha:
            raise ValueError(f"Level-5 terminal receipt {key}가 immutable ledger와 다릅니다")
    for field in ("raw_evaluation_bundle", "metrics", "evaluator_receipt"):
        _verify_file_reference(entry[field], root=root, label=f"Level-5 terminal receipt.{field}")
    return str(verdict)


def _check(passed: bool, *, check_id: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": "PASS" if passed else "BLOCKED",
        "detail": detail,
        **extra,
    }


def audit_full_octave_v3_level5_lifecycle(
    payload: Mapping[str, Any], *, repo_root: str | Path
) -> dict[str, Any]:
    """Level-5 declaration을 읽되, self-attested chain을 authority로 승격하지 않는다.

    예전 구현은 fake checkpoint/controller, minimal physical report, 자체 봉인 manifest만
    맞춰도 준비 완료처럼 보이는 상태를 냈다. 이는 capture 발급 권한처럼 읽힐 수 있는
    false-ready 경로였다. 이 함수는 구조 오류를 진단할 수는 있어도 별도 independent
    evaluator authority가 없는 현재/미래 JSON 조합을 항상
    ``BLOCKED_UNATTESTED_*``로 반환한다.
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
            detail="125 Hz--8 kHz canonical v3 contract와 digest가 lifecycle config에 결속됐습니다.",
            control_band_contract_sha256=contract.digest(),
        )
    ]
    snapshots: dict[str, tuple[str, bytes, str, int]] = {}
    declaration_errors: list[str] = []
    for role in _ARTIFACT_ROLES:
        path, expected_sha = intents[role]
        if path is None:
            checks.append(
                _check(False, check_id=f"artifact:{role}", detail="artifact가 아직 선언되지 않았습니다.")
            )
            continue
        try:
            relative, content, observed_sha, size = _snapshot_declared_artifact(
                root=root,
                path=path,
                expected_sha256=expected_sha or "",
                label=f"artifacts.{role}",
            )
        except (OSError, ValueError) as exc:
            declaration_errors.append(f"{role}: {exc}")
            checks.append(
                _check(
                    False,
                    check_id=f"artifact:{role}",
                    detail=f"self-declared artifact를 authority로 사용할 수 없습니다: {exc}",
                )
            )
            continue
        snapshots[role] = (relative, content, observed_sha, size)
        checks.append(
            _check(
                True,
                check_id=f"artifact:{role}",
                detail="regular-file bytes SHA가 config와 일치합니다. 이는 독립 physical authority는 아닙니다.",
                path=relative,
                sha256=observed_sha,
            )
        )

    preregistration = _challenge_preregistration_limitations()
    primary_present = all(role in snapshots for role in _PRIMARY_ARTIFACT_ROLES)
    if declaration_errors:
        checks.append(
            _check(
                False,
                check_id="self_attested_declaration_integrity",
                detail="선언된 artifact 중 read-only byte/SHA/경로 검사를 통과하지 못한 것이 있습니다.",
                errors=declaration_errors,
            )
        )
        checks.extend(_unattested_authority_checks())
        blockers = tuple(check["id"] for check in checks if check["status"] != "PASS")
        return _report(
            status=BLOCKED_UNATTESTED_INVALID_DECLARATION,
            checks=checks,
            blockers=blockers,
            challenge_input_sha256=None,
            exclusion=None,
            base_pairwise_leakage=None,
            selection_validation_alias=None,
            challenge_preregistration=preregistration,
            self_declared_terminal_verdict=None,
        )

    if not primary_present:
        checks.extend(
            (
                _check(False, check_id="frozen_model_contract_lock", detail="model lock 또는 source manifest가 아직 없습니다."),
                _check(False, check_id="base_train_val_test_pairwise_leakage", detail="base train/val/test split의 actual identity inventory가 없습니다."),
                _check(False, check_id="selection_validation_exact_sha_alias", detail="model selection이 validation manifest의 exact SHA alias인지 아직 확인할 수 없습니다."),
                _check(False, check_id="level5_raw_manifest_exclusion", detail="네 family raw source manifest authority가 아직 complete하지 않습니다."),
                _check(False, check_id="challenge_preregistration_limit", detail="challenge reservation을 독립적으로 증명할 artifact가 없습니다."),
                _check(False, check_id="frozen_physical_bundle_identity", detail="frozen eight-input physical bundle lock이 아직 없습니다."),
                _check(False, check_id="self_attested_one_shot_ledger", detail="issuer/consumed/terminal ledger가 없고, 이 checker는 발급 권한도 없습니다."),
            )
        )
        checks.extend(_unattested_authority_checks())
        blockers = tuple(check["id"] for check in checks if check["status"] != "PASS")
        return _report(
            status=BLOCKED_UNATTESTED_MISSING_AUTHORITY,
            checks=checks,
            blockers=blockers,
            challenge_input_sha256=None,
            exclusion=None,
            base_pairwise_leakage=None,
            selection_validation_alias=None,
            challenge_preregistration=preregistration,
            self_declared_terminal_verdict=None,
        )

    try:
        manifests: dict[str, list[dict[str, Any]]] = {}
        manifest_sha256: dict[str, str] = {}
        for role, partition in _MANIFEST_ROLE_TO_PARTITION.items():
            manifests[partition] = (
                _validate_raw_source_manifest(
                    _load_json(snapshots[role][1], label=f"artifacts.{role}"),
                    root=root,
                    partition=partition,
                    model_lock_sha256=None,
                )
                if partition != "challenge"
                else []
            )
            manifest_sha256[partition] = snapshots[role][2]

        selection_validation_alias = _validate_selection_validation_alias(snapshots)
        manifest_sha256["selection"] = snapshots["selection_raw_manifest"][2]
        model_lock_sha = snapshots["model_lock"][2]
        model = _validate_model_lock(
            _load_json(snapshots["model_lock"][1], label="artifacts.model_lock"),
            root=root,
            validation_manifest_sha256=manifest_sha256["validation"],
        )
        manifests["challenge"] = _validate_raw_source_manifest(
            _load_json(snapshots["challenge_raw_manifest"][1], label="artifacts.challenge_raw_manifest"),
            root=root,
            partition="challenge",
            model_lock_sha256=model_lock_sha,
        )
        _validate_global_identity_consistency(manifests)
        base_pairwise_leakage = _base_split_pairwise_leakage(manifests)
        exclusion = _level5_exclusion(manifests)
        physical = _validate_physical_bundle_lock(
            _load_json(snapshots["physical_bundle_lock"][1], label="artifacts.physical_bundle_lock"),
            root=root,
            model_lock_sha256=model_lock_sha,
            model_controller_sha256=model["controller_artifact"]["sha256"],
            experiment_contract_sha256=model["experiment_contract_sha256"],
            challenge_manifest_sha256=manifest_sha256["challenge"],
        )
    except (OSError, ValueError) as exc:
        checks.append(
            _check(
                False,
                check_id="self_attested_structural_chain",
                detail=f"self-declared Level-5 chain의 구조 검사가 실패했습니다: {exc}",
            )
        )
        checks.extend(_unattested_authority_checks())
        blockers = tuple(check["id"] for check in checks if check["status"] != "PASS")
        return _report(
            status=BLOCKED_UNATTESTED_INVALID_DECLARATION,
            checks=checks,
            blockers=blockers,
            challenge_input_sha256=None,
            exclusion=None,
            base_pairwise_leakage=None,
            selection_validation_alias=None,
            challenge_preregistration=preregistration,
            self_declared_terminal_verdict=None,
        )

    physical_lock_sha = snapshots["physical_bundle_lock"][2]
    checks.extend(
        (
            _check(
                True,
                check_id="frozen_model_contract_lock",
                detail="model lock JSON이 declared checkpoint/controller/experiment-contract bytes와 structural SHA로 결속됐습니다. canonical completion/selection/export provenance는 독립 확인 전입니다.",
                model_lock_sha256=model_lock_sha,
                checkpoint_sha256=model["checkpoint"]["sha256"],
                experiment_contract_sha256=model["experiment_contract_sha256"],
            ),
            _check(
                True,
                check_id="base_train_val_test_pairwise_leakage",
                detail="declared base train/val/test manifest bytes에서는 pairwise identity 교집합이 0입니다. selection은 validation exact-SHA alias이며 immutable inventory authority는 별도입니다.",
                pairwise=base_pairwise_leakage,
            ),
            _check(
                True,
                check_id="selection_validation_exact_sha_alias",
                detail="model selection은 별도 split이 아니라 validation raw manifest의 exact SHA alias입니다.",
                alias=selection_validation_alias,
            ),
            _check(
                True,
                check_id="level5_raw_manifest_exclusion",
                detail="declared challenge bytes가 각 family별 training∪validation∪test union identity와 0 교집합입니다. selection은 validation alias이므로 같은 union에 포함됩니다. 이는 self-attested structural result입니다.",
                exclusion=exclusion,
            ),
            _check(
                False,
                check_id="challenge_preregistration_limit",
                detail=str(preregistration["limitation"]),
            ),
            _check(
                True,
                check_id="frozen_physical_bundle_identity",
                detail="physical bundle config SHA로 official 8-input validator를 재실행해 report/plan/raw/sidecar structural binding을 다시 확인했습니다. 이는 여전히 independent physical authority가 아닙니다.",
                physical_bundle_lock_sha256=physical_lock_sha,
                official_bundle_config_sha256=physical["official_bundle_config_sha256"],
                capture_plan_sha256=physical["capture_plan_sha256"],
                canonical_raw_sha256=physical["canonical_raw_sha256"],
            ),
        )
    )

    artifact_paths = {role: snapshots[role][0] for role in _PRIMARY_ARTIFACT_ROLES}
    artifact_sha256 = {role: snapshots[role][2] for role in _PRIMARY_ARTIFACT_ROLES}
    challenge_input = _challenge_input_payload(
        root_payload=root_payload,
        artifact_paths=artifact_paths,
        artifact_sha256=artifact_sha256,
    )
    challenge_input_sha = _sha256_bytes(_canonical_json(challenge_input))
    ledger_roles = (
        "one_shot_capability",
        "one_shot_consumed_marker",
        "one_shot_receipt",
    )
    declared_ledger = [role for role in ledger_roles if role in snapshots]
    self_declared_terminal_verdict: str | None = None
    if "one_shot_receipt" in snapshots:
        try:
            terminal_payload = _load_json(
                snapshots["one_shot_receipt"][1], label="artifacts.one_shot_receipt"
            )
            declared = terminal_payload.get("verdict")
            self_declared_terminal_verdict = (
                str(declared) if isinstance(declared, str) else "UNREADABLE_SELF_DECLARATION"
            )
        except ValueError:
            self_declared_terminal_verdict = "UNREADABLE_SELF_DECLARATION"
    checks.append(
        _check(
            False,
            check_id="self_attested_one_shot_ledger",
            detail=(
                "capability/consumed/terminal JSON의 선언 또는 SHA 연결은 dirfd O_EXCL 발급·terminal "
                "mutual exclusion·independent raw evaluation을 소급 증명하지 못합니다."
            ),
            declared_artifacts=declared_ledger,
            self_declared_terminal_verdict=self_declared_terminal_verdict,
        )
    )
    checks.extend(_unattested_authority_checks())
    blockers = tuple(check["id"] for check in checks if check["status"] != "PASS")
    status = (
        BLOCKED_UNATTESTED_TERMINAL_RECEIPT
        if self_declared_terminal_verdict is not None
        else BLOCKED_UNATTESTED_SELF_DECLARED_CHAIN
    )
    return _report(
        status=status,
        checks=checks,
        blockers=blockers,
        challenge_input_sha256=challenge_input_sha,
        exclusion=exclusion,
        base_pairwise_leakage=base_pairwise_leakage,
        selection_validation_alias=selection_validation_alias,
        challenge_preregistration=preregistration,
        self_declared_terminal_verdict=self_declared_terminal_verdict,
    )


def _report(
    *,
    status: str,
    checks: Sequence[Mapping[str, Any]],
    blockers: Sequence[str],
    challenge_input_sha256: str | None,
    exclusion: Mapping[str, Any] | None,
    base_pairwise_leakage: Mapping[str, Any] | None,
    selection_validation_alias: Mapping[str, Any] | None,
    challenge_preregistration: Mapping[str, Any],
    self_declared_terminal_verdict: str | None,
) -> dict[str, Any]:
    """모든 lifecycle 상태에서 self-attested authority를 명시적으로 false로 고정."""

    return {
        "schema": FULL_OCTAVE_V3_LEVEL5_REPORT_SCHEMA,
        "status": status,
        "control_band_contract_sha256": BroadbandFullOctaveContractV3.canonical().digest(),
        "challenge_input_sha256": challenge_input_sha256,
        "checks": [dict(item) for item in checks],
        "blockers": list(blockers),
        "level5_exclusion": exclusion,
        # base train/val/test은 challenge와 별개로 pairwise zero intersection이어야 한다.
        # 이 값은 declared files의 구조 계산이며 immutable lineage inventory 검증은 아니다.
        "base_train_val_test_pairwise_leakage": base_pairwise_leakage,
        "selection_validation_manifest_alias": (
            dict(selection_validation_alias) if selection_validation_alias is not None else None
        ),
        "challenge_preregistration_limitations": dict(challenge_preregistration),
        "self_declared_terminal_verdict": self_declared_terminal_verdict,
        "audio_opened": False,
        "alsa_opened": False,
        "gpu_initialized": False,
        "network_opened": False,
        "files_written": False,
        "actual_capture_started": False,
        "actual_evaluation_started": False,
        "fixture_authority_allowed": False,
        # post-hoc read-only 검사는 O_EXCL 선언과 ledger 경로를 확인할 수 있을 뿐,
        # kernel 수준의 no-replace history 자체를 소급 증명할 수는 없다.
        "no_replace_history_proven": False,
        "self_attested_chain_only": True,
        "independent_raw_evaluator_receipt_verified": False,
        "canonical_generalization_pass": False,
        "physical_generalization_authority": False,
        "next_required_authority": (
            "physical bundle config SHA를 사용한 official 8ch/raw/sidecar validator 재실행, immutable "
            "lineage inventory/population/reservation, completed canonical checkpoint/embedded contract/"
            "val selection/export provenance, submitted PCM/controller/P-S/timing/lead exact binding, 네 "
            "family matched OFF/DL/FxLMS campaign, dirfd O_EXCL one-shot mutual exclusion, 그리고 독립 "
            "raw evaluator receipt가 모두 필요합니다. 이 lifecycle report만으로 Level-5 PASS 또는 "
            "deployment를 선언할 수 없습니다."
        ),
    }


def load_full_octave_v3_level5_lifecycle(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    """YAML 및 declared artifact를 read-only로 검사한다."""

    root = Path(repository_root).resolve(strict=True)
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Level-5 lifecycle config를 읽을 수 없습니다: {config_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Level-5 lifecycle YAML root는 mapping이어야 합니다")
    report = audit_full_octave_v3_level5_lifecycle(payload, repo_root=root)
    config_content, config_sha, _config_size = _snapshot_regular_file(
        config_path, label="Level-5 lifecycle config"
    )
    # config bytes를 다시 해석하지 않고 snapshot SHA만 report에 더한다. 결과를 쓰지 않는다.
    del config_content
    return {**report, "config": {"path": str(config_path), "file_sha256": config_sha}}


def render_full_octave_v3_level5_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Full-octave v3 Level-5 unseen physical lifecycle",
        "",
        f"- 상태: `{report['status']}`",
        f"- v3 contract SHA: `{report['control_band_contract_sha256']}`",
        f"- challenge input SHA: `{report['challenge_input_sha256']}`",
        f"- canonical/generalization PASS: `{report['canonical_generalization_pass']}`/"
        f"`{report['physical_generalization_authority']}`",
        f"- self-attested chain only / independent evaluator receipt verified: "
        f"`{report['self_attested_chain_only']}`/"
        f"`{report['independent_raw_evaluator_receipt_verified']}`",
        f"- 오디오/ALSA/GPU/network/쓰기: `{report['audio_opened']}`/"
        f"`{report['alsa_opened']}`/`{report['gpu_initialized']}`/"
        f"`{report['network_opened']}`/`{report['files_written']}`",
        "",
        "## 검사",
        "",
    ]
    for check in report["checks"]:
        lines.append(f"- [{check['status']}] `{check['id']}` — {check['detail']}")
    lines.extend(
        [
            "",
            "## Split 및 challenge 예약 한계",
            "",
            "- base train/val/test pairwise leakage (declared bytes): "
            f"`{report.get('base_train_val_test_pairwise_leakage')}`",
            "- selection == validation exact-SHA alias: "
            f"`{report.get('selection_validation_manifest_alias')}`",
            "- challenge preregistration limitation: "
            f"`{report.get('challenge_preregistration_limitations')}`",
        ]
    )
    if report.get("blockers"):
        lines.extend(["", "## 차단 항목", ""])
        lines.extend(f"- `{blocker}`" for blocker in report["blockers"])
    lines.extend(["", "## 다음 authority", "", str(report["next_required_authority"]), ""])
    return "\n".join(lines)


__all__ = [
    "BLOCKED_UNATTESTED_INVALID_DECLARATION",
    "BLOCKED_UNATTESTED_MISSING_AUTHORITY",
    "BLOCKED_UNATTESTED_SELF_DECLARED_CHAIN",
    "BLOCKED_UNATTESTED_TERMINAL_RECEIPT",
    "DEFAULT_CONFIG_RELATIVE_PATH",
    "FULL_OCTAVE_V3_LEVEL5_CAPABILITY_SCHEMA",
    "FULL_OCTAVE_V3_LEVEL5_CONFIG_SCHEMA",
    "FULL_OCTAVE_V3_LEVEL5_CONSUMED_SCHEMA",
    "FULL_OCTAVE_V3_LEVEL5_MODEL_LOCK_SCHEMA",
    "FULL_OCTAVE_V3_LEVEL5_PHYSICAL_BUNDLE_LOCK_SCHEMA",
    "FULL_OCTAVE_V3_LEVEL5_RAW_MANIFEST_SCHEMA",
    "FULL_OCTAVE_V3_LEVEL5_RECEIPT_SCHEMA",
    "FULL_OCTAVE_V3_LEVEL5_REPORT_SCHEMA",
    "FULL_OCTAVE_V3_LEVEL5_ROLE",
    "LEVEL5_LEDGER_ROOT",
    "audit_full_octave_v3_level5_lifecycle",
    "canonical_level5_ledger_paths",
    "load_full_octave_v3_level5_lifecycle",
    "render_full_octave_v3_level5_markdown",
]
