"""노트북 작업 상태를 Drive의 append-only receipt로 교환한다.

대용량 public corpus는 Git에 넣지 않는다. 노트북은 작은 검증 receipt를 이 모듈로
content-addressed 경로에 먼저 올리고, 마지막에 상태 JSON을 ``--immutable``로
발행한다. Jetson은 ``read`` 경로에서 ``rclone cat/lsjson``만 사용한다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from ..dsp.stage2_2khz_contract import (
    STAGE2_2KHZ_CONTRACT_ID,
    STAGE2_2KHZ_OBJECTIVE_BANDS_HZ,
    STAGE2_2KHZ_SOURCE_FAMILIES,
    Stage2TwoKilohertzContract,
)


STATUS_SCHEMA = "deep_anc_notebook_stage2_status_v2"
SUMMARY_SCHEMA = "deep_anc_notebook_stage2_summary_v2"
CAMPAIGN = "stage2_2khz_v1"
DEFAULT_REMOTE_ROOT = "gdrive:DeepANC/notebook_exchange/stage2_2khz_v1"
PHASES = (
    "preflight",
    "drive_partial_restore",
    "public_archive_cache",
    "decoder_qa",
    "lineage_manifest",
    "frequency_coverage",
    "bundle_publish",
)
STATES = ("BLOCKED", "IN_PROGRESS", "FAIL", "PASS")

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_REMOTE = re.compile(r"^[A-Za-z0-9_.-]+:[A-Za-z0-9_.\-/]+$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_SAFE_KIND = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_STATUS_NAME = re.compile(
    r"^(\d{8}T\d{6}Z)_([0-9a-f]{12})_([a-z0-9_]+)_([0-9a-f]{16})\.json$"
)
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_ALLOWED_RECEIPT_SUFFIXES = {".json", ".jsonl", ".txt", ".log", ".md", ".csv"}
_MAX_RECEIPT_BYTES = 64 * 1024 * 1024

_PHASE_REQUIRED_ARTIFACT_KINDS = {
    "preflight": frozenset({"checkout_audit"}),
    "drive_partial_restore": frozenset({"restore_receipt"}),
    "public_archive_cache": frozenset({"archive_cache_receipt"}),
    "decoder_qa": frozenset({"decoder_qa_receipt"}),
    "lineage_manifest": frozenset({"manifest_bundle", "lineage_receipt"}),
    # coverage가 주장하는 qualified component를 실제 manifest row에 다시 결속한다.
    "frequency_coverage": frozenset(
        {"manifest_bundle", "frequency_coverage_receipt"}
    ),
    "bundle_publish": frozenset(
        {
            "manifest_bundle",
            "lineage_receipt",
            "frequency_coverage_receipt",
            "transfer_bootstrap_receipt",
        }
    ),
}
_ARTIFACT_SCHEMAS = {
    "checkout_audit": "deep_anc_notebook_checkout_audit_v1",
    "restore_receipt": "stage2_drive_local_restore_receipt_v1",
    # 다음 두 schema는 canonical 학습 authority가 아니라, 실제 production
    # manifest/audit을 exact code로 읽어 만든 작은 notebook advisory projection이다.
    "archive_cache_receipt": "deep_anc_notebook_archive_cache_readback_v1",
    "decoder_qa_receipt": "deep_anc_notebook_full_decoder_receipt_v2",
    "manifest_bundle": "stage2_2khz_public_manifest_bundle_v1",
    "lineage_receipt": "stage2_2khz_public_lineage_receipt_v2",
    "frequency_coverage_receipt": "stage2_2khz_public_frequency_coverage_v2",
    "transfer_bootstrap_receipt": "stage2_2khz_transfer_bootstrap_receipt_v1",
}

_SPLITS = ("train", "val", "test")
_FAMILIES = tuple(STAGE2_2KHZ_SOURCE_FAMILIES)
_RECORDED_HOLDOUT_PATH = "data/manifests/recorded_holdout.json"
_RECORDED_SYNTHETIC_INTERSECTION_ALGORITHM = (
    "transitive_basename_content_sha256_lineage_keys_v1"
)
_MIN_FREE_BYTES = 32 * 1024**3
_RESTORE_FILE_COUNT = 12_819
_RESTORE_BYTE_COUNT = 9_480_223_737
_ARCHIVE_SIZES = {
    "dns_noise_000": 5_364_611_964,
    "dns_noise_001": 5_357_916_291,
    "dns_speech_000": 4_664_045_287,
    "demand_dkitchen": 336_992_458,
    "demand_dwashing": 306_101_499,
    "demand_ooffice": 277_643_831,
    "demand_ohallway": 252_905_617,
    "demand_tmetro": 367_513_573,
    "demand_tcar": 373_520_251,
    "mimii_fan": 928_511_244,
}
_ARCHIVE_TOTAL_BYTES = 18_229_762_015
_DECODER_COHORT_COUNTS = {
    "dns_fullband": 16_000,
    "speech": 8_065,
    "music": 8_000,
    "demand": 96,
    "machine": 3_600,
    "esc50": 2_000,
}
_DECODER_TOTAL_COUNT = 37_761
_DECODER_ACCEPTED_COHORT_COUNTS = {
    "dns_fullband": 15_553,
    "speech": 7_971,
    "music": 7_941,
    "demand": 96,
    "machine": 3_600,
    "esc50": 1_707,
}
_DECODER_REJECTED_COHORT_COUNTS = {
    name: _DECODER_COHORT_COUNTS[name] - accepted
    for name, accepted in _DECODER_ACCEPTED_COHORT_COUNTS.items()
}
_DECODER_ACCEPTED_COUNT = 36_868
_DECODER_REJECTED_COUNT = 893
_SENTINEL_BAND_HZ = [1425.437949, 1795.939277]
_RESTORE_BLOCKERS = [
    "DNS_DEMAND_MIMII_FIXED_ARCHIVE_CACHE_OR_OFFICIAL_DOWNLOAD_REQUIRED",
    "STAGE2_LINEAGE_AND_FREQUENCY_MANIFEST_BUNDLE_REQUIRED",
    "LOCAL_DECODER_AND_SOURCE_DENSITY_AUDIT_REQUIRED",
]


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(payload) != expected:
        raise NotebookExchangeError(f"{label} key 집합이 exact하지 않습니다")


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if not _HEX64.fullmatch(text):
        raise NotebookExchangeError(f"{label}가 lowercase SHA-256이 아닙니다")
    return text


def _require_commit(value: object, *, label: str) -> str:
    text = str(value)
    if not _HEX40.fullmatch(text):
        raise NotebookExchangeError(f"{label}가 lowercase 40자리 commit이 아닙니다")
    return text


def _require_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NotebookExchangeError(f"{label}가 양의 정수가 아닙니다")
    return value


def _contract_entry() -> dict[str, str]:
    contract = Stage2TwoKilohertzContract.canonical()
    return {"id": STAGE2_2KHZ_CONTRACT_ID, "sha256": contract.digest()}


def _validate_checkout(payload: Mapping[str, Any], *, expected_commit: str | None) -> None:
    _require_exact_keys(
        payload,
        {
            "schema",
            "status",
            "source_commit",
            "repository_clean_exact",
            "work_root_outside_repository",
            "work_root_free_bytes",
            "rclone_executable_name",
            "rclone_version",
            "secrets_recorded",
        },
        label="checkout audit",
    )
    commit = _require_commit(payload["source_commit"], label="checkout source_commit")
    if expected_commit is not None and commit != expected_commit:
        raise NotebookExchangeError("checkout audit commit이 status commit과 다릅니다")
    if (
        payload["status"] != "PASS"
        or payload["repository_clean_exact"] is not True
        or payload["work_root_outside_repository"] is not True
        or payload["secrets_recorded"] is not False
    ):
        raise NotebookExchangeError("checkout clean/outside/secrets 의미가 PASS가 아닙니다")
    free = _require_positive_int(payload["work_root_free_bytes"], label="checkout free bytes")
    if free < _MIN_FREE_BYTES:
        raise NotebookExchangeError("checkout work root 가용공간이 32 GiB 미만입니다")
    executable = str(payload["rclone_executable_name"])
    version = str(payload["rclone_version"])
    if not _SAFE_NAME.fullmatch(executable) or not version.startswith("rclone v"):
        raise NotebookExchangeError("checkout rclone executable/version evidence가 없습니다")


def _validate_restore(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(
        payload,
        {
            "schema",
            "authority",
            "status",
            "anchor_file_sha256",
            "anchor_evidence_sha256",
            "snapshot_manifest_file_sha256",
            "restore_root",
            "file_count",
            "byte_count",
            "extension_counts",
            "path_size_sha256_projection_sha256",
            "stage2_public_pretrain_ready",
            "remaining_blockers",
            "evidence_sha256",
        },
        label="partial restore receipt",
    )
    if (
        payload["authority"]
        != "local_partial_restore_content_verified_not_training_authority"
        or payload["status"] != "PASS_PARTIAL_RESTORE_ONLY"
        or payload["stage2_public_pretrain_ready"] is not False
        or payload["remaining_blockers"] != _RESTORE_BLOCKERS
        or payload["file_count"] != _RESTORE_FILE_COUNT
        or payload["byte_count"] != _RESTORE_BYTE_COUNT
    ):
        raise NotebookExchangeError("partial restore exact 12,819/9,480,223,737 계약이 아닙니다")
    for key in (
        "anchor_file_sha256",
        "anchor_evidence_sha256",
        "snapshot_manifest_file_sha256",
        "path_size_sha256_projection_sha256",
    ):
        _require_sha256(payload[key], label=f"restore {key}")
    root = str(payload["restore_root"])
    if not root.startswith("/") or "\x00" in root:
        raise NotebookExchangeError("restore_root가 절대경로가 아닙니다")
    extensions = payload["extension_counts"]
    if (
        not isinstance(extensions, dict)
        or not extensions
        or any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in extensions.items()
        )
        or sum(extensions.values()) != _RESTORE_FILE_COUNT
    ):
        raise NotebookExchangeError("restore extension projection이 file_count와 다릅니다")
    evidence = _require_sha256(payload["evidence_sha256"], label="restore evidence SHA")
    unsigned = {key: value for key, value in payload.items() if key != "evidence_sha256"}
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != evidence:
        raise NotebookExchangeError("restore evidence self digest가 다릅니다")


def _validate_archive(payload: Mapping[str, Any], *, expected_commit: str | None) -> None:
    _require_exact_keys(
        payload,
        {
            "schema",
            "status",
            "authority",
            "source_commit",
            "production_manifest_sha256",
            "production_manifest_remote_path",
            "archive_count",
            "archive_total_bytes",
            "archive_sizes_by_id",
            "archive_sha256_by_id",
            "immutable_content_addressed_archive_paths",
            "publisher_source_sha256_verified",
            "publisher_archive_and_manifest_readback_enforced",
            "canonical_training_authority",
            "evidence_sha256",
        },
        label="archive cache advisory receipt",
    )
    commit = _require_commit(payload["source_commit"], label="archive source_commit")
    if expected_commit is not None and commit != expected_commit:
        raise NotebookExchangeError("archive publisher commit이 status commit과 다릅니다")
    manifest_sha = _require_sha256(
        payload["production_manifest_sha256"], label="archive production manifest SHA"
    )
    if payload["production_manifest_remote_path"] != (
        f"manifests/v1/sha256_{manifest_sha}/archive_cache_manifest.json"
    ):
        raise NotebookExchangeError("archive production manifest remote path가 content-addressed가 아닙니다")
    sha_by_id = payload["archive_sha256_by_id"]
    if (
        payload["status"] != "PASS"
        or payload["authority"]
        != "transport_readback_advisory_not_raw_or_training_authority"
        or payload["archive_count"] != len(_ARCHIVE_SIZES)
        or payload["archive_total_bytes"] != _ARCHIVE_TOTAL_BYTES
        or payload["archive_sizes_by_id"] != _ARCHIVE_SIZES
        or not isinstance(sha_by_id, dict)
        or set(sha_by_id) != set(_ARCHIVE_SIZES)
        or any(not _HEX64.fullmatch(str(value)) for value in sha_by_id.values())
        or payload["immutable_content_addressed_archive_paths"] is not True
        or payload["publisher_source_sha256_verified"] is not True
        or payload["publisher_archive_and_manifest_readback_enforced"] is not True
        or payload["canonical_training_authority"] is not False
    ):
        raise NotebookExchangeError("archive fixed 10/18,229,762,015/readback 계약이 아닙니다")
    evidence = _require_sha256(payload["evidence_sha256"], label="archive evidence SHA")
    unsigned = {key: value for key, value in payload.items() if key != "evidence_sha256"}
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != evidence:
        raise NotebookExchangeError("archive evidence self digest가 다릅니다")


def _validate_decoder(payload: Mapping[str, Any], *, expected_commit: str | None) -> None:
    _require_exact_keys(
        payload,
        {
            "schema",
            "status",
            "authority",
            "source_commit",
            "decoder_audit_file_sha256",
            "decoder_audit_semantic_sha256",
            "inventory_sha256",
            "accepted_inventory_sha256",
            "candidate_count",
            "accepted_count",
            "rejected_count",
            "cohort_counts",
            "accepted_cohort_counts",
            "rejected_cohort_counts",
            "full_sequential_chunk_frames",
            "full_inventory_rows_consumed",
            "partial_only",
            "canonical_training_authority",
            "evidence_sha256",
        },
        label="full decoder advisory receipt",
    )
    commit = _require_commit(payload["source_commit"], label="decoder source_commit")
    if expected_commit is not None and commit != expected_commit:
        raise NotebookExchangeError("decoder receipt commit이 status commit과 다릅니다")
    for key in (
        "decoder_audit_file_sha256",
        "decoder_audit_semantic_sha256",
        "inventory_sha256",
        "accepted_inventory_sha256",
    ):
        _require_sha256(payload[key], label=f"decoder {key}")
    if (
        payload["status"] != "PASS"
        or payload["authority"] != "full_decoder_projection_not_training_authority"
        or payload["candidate_count"] != _DECODER_TOTAL_COUNT
        or payload["accepted_count"] != _DECODER_ACCEPTED_COUNT
        or payload["rejected_count"] != _DECODER_REJECTED_COUNT
        or payload["cohort_counts"] != _DECODER_COHORT_COUNTS
        or payload["accepted_cohort_counts"] != _DECODER_ACCEPTED_COHORT_COUNTS
        or payload["rejected_cohort_counts"] != _DECODER_REJECTED_COHORT_COUNTS
        or payload["full_sequential_chunk_frames"] != [65_536, 262_144]
        or payload["full_inventory_rows_consumed"] is not True
        or payload["partial_only"] is not False
        or payload["canonical_training_authority"] is not False
    ):
        raise NotebookExchangeError(
            "decoder full 37,761 = accept 36,868 + reject 893 partition 계약이 아닙니다"
        )
    evidence = _require_sha256(payload["evidence_sha256"], label="decoder evidence SHA")
    unsigned = {key: value for key, value in payload.items() if key != "evidence_sha256"}
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != evidence:
        raise NotebookExchangeError("decoder evidence self digest가 다릅니다")


def _manifest_rows(
    payload: Mapping[str, Any], *, expected_commit: str | None
) -> dict[int, dict[str, Any]]:
    _require_exact_keys(
        payload,
        {
            "schema",
            "status",
            "canonical_pretrain_eligible",
            "control_band_contract",
            "required_source_families",
            "required_splits",
            "recorded_artifacts_required_for_pretrain",
            "test_split_for_checkpoint_selection_allowed",
            "source_inventory_commit_sha",
            "items",
        },
        label="Stage-2 manifest bundle",
    )
    source_commit = _require_commit(
        payload["source_inventory_commit_sha"], label="manifest source inventory commit"
    )
    if expected_commit is not None and source_commit != expected_commit:
        raise NotebookExchangeError("manifest source inventory commit이 status commit과 다릅니다")
    if (
        payload["status"] != "PASS"
        or payload["canonical_pretrain_eligible"] is not True
        or payload["control_band_contract"] != _contract_entry()
        or payload["required_source_families"] != list(_FAMILIES)
        or payload["required_splits"] != list(_SPLITS)
        or payload["recorded_artifacts_required_for_pretrain"] is not False
        or payload["test_split_for_checkpoint_selection_allowed"] is not False
    ):
        raise NotebookExchangeError("manifest Stage-2 canonical 의미가 다릅니다")
    items = payload["items"]
    if not isinstance(items, list) or not items:
        raise NotebookExchangeError("manifest items가 비었습니다")
    by_index: dict[int, dict[str, Any]] = {}
    paths: set[str] = set()
    component_splits: dict[str, set[str]] = {}
    sha_splits: dict[str, set[str]] = {}
    sha_components: dict[str, set[str]] = {}
    lineage_splits: dict[str, set[str]] = {}
    cells: dict[tuple[str, str], set[str]] = {
        (split, family): set() for split in _SPLITS for family in _FAMILIES
    }
    row_keys = {
        "dataset_index",
        "source_family",
        "component_id",
        "split",
        "path",
        "content_sha256",
        "content_size",
        "native_sample_rate",
        "native_nyquist_hz",
        "lineage_keys",
    }
    for raw in items:
        if not isinstance(raw, dict) or set(raw) != row_keys:
            raise NotebookExchangeError("manifest row key 집합이 exact하지 않습니다")
        index = raw["dataset_index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index in by_index
        ):
            raise NotebookExchangeError("manifest dataset_index가 중복/음수입니다")
        split = str(raw["split"])
        family = str(raw["source_family"])
        component = str(raw["component_id"])
        if split not in _SPLITS or family not in _FAMILIES or not component:
            raise NotebookExchangeError("manifest split/family/component가 canonical이 아닙니다")
        path = str(raw["path"])
        pure = PurePosixPath(path)
        if (
            not path
            or "\\" in path
            or pure.is_absolute()
            or any(part in {"", ".", ".."} for part in pure.parts)
            or path in paths
        ):
            raise NotebookExchangeError("manifest path가 중복/비정상 상대경로입니다")
        paths.add(path)
        digest = _require_sha256(raw["content_sha256"], label="manifest source SHA")
        _require_positive_int(raw["content_size"], label="manifest content_size")
        rate = _require_positive_int(raw["native_sample_rate"], label="manifest sample rate")
        nyquist = raw["native_nyquist_hz"]
        if (
            isinstance(nyquist, bool)
            or not isinstance(nyquist, (int, float))
            or float(nyquist) < 2828.4271247462
            or abs(float(nyquist) - rate / 2.0) > 1.0e-9
        ):
            raise NotebookExchangeError("manifest source Nyquist가 2 kHz octave를 덮지 않습니다")
        lineage = raw["lineage_keys"]
        if (
            not isinstance(lineage, list)
            or not lineage
            or lineage != sorted(set(lineage))
            or any(not isinstance(value, str) or not value for value in lineage)
        ):
            raise NotebookExchangeError("manifest lineage_keys가 nonempty sorted unique가 아닙니다")
        component_splits.setdefault(component, set()).add(split)
        sha_splits.setdefault(digest, set()).add(split)
        sha_components.setdefault(digest, set()).add(component)
        for value in lineage:
            lineage_splits.setdefault(value, set()).add(split)
        cells[(split, family)].add(component)
        by_index[index] = dict(raw)
    for label, projection in (
        ("component", component_splits),
        ("source SHA", sha_splits),
        ("original lineage", lineage_splits),
    ):
        if any(len(splits) != 1 for splits in projection.values()):
            raise NotebookExchangeError(f"manifest {label}가 split을 가로지릅니다")
    if any(len(components) != 1 for components in sha_components.values()):
        raise NotebookExchangeError("manifest 동일 source SHA가 여러 component로 세탁됐습니다")
    if any(len(components) < 4 for components in cells.values()):
        raise NotebookExchangeError("manifest 3 split×4 family distinct component가 4개 미만입니다")
    return by_index


def _validate_lineage(payload: Mapping[str, Any], *, expected_commit: str | None) -> None:
    _require_exact_keys(
        payload,
        {
            "schema",
            "status",
            "canonical_pretrain_eligible",
            "control_band_contract_sha256",
            "manifest_bundle_sha256",
            "verified_item_count",
            "component_cross_split_count",
            "source_sha_cross_split_count",
            "original_lineage_cross_split_count",
            "recorded_synthetic_lineage_intersection_count",
            "actual_manifest_rows_consumed",
            "recorded_holdout",
            "recorded_clip_count",
            "recorded_clip_lineage_sha256",
            "recorded_synthetic_intersection_algorithm",
            "actual_recorded_holdout_bytes_consumed",
            "source_inventory_commit_sha",
        },
        label="lineage receipt",
    )
    commit = _require_commit(
        payload["source_inventory_commit_sha"], label="lineage source inventory commit"
    )
    if expected_commit is not None and commit != expected_commit:
        raise NotebookExchangeError("lineage source commit이 status commit과 다릅니다")
    _require_sha256(payload["manifest_bundle_sha256"], label="lineage manifest SHA")
    recorded_holdout = payload["recorded_holdout"]
    if (
        not isinstance(recorded_holdout, dict)
        or set(recorded_holdout) != {"path", "sha256"}
        or recorded_holdout["path"] != _RECORDED_HOLDOUT_PATH
    ):
        raise NotebookExchangeError("lineage recorded_holdout ref가 canonical exact가 아닙니다")
    _require_sha256(recorded_holdout["sha256"], label="lineage recorded holdout SHA")
    _require_sha256(
        payload["recorded_clip_lineage_sha256"],
        label="lineage recorded clip semantic SHA",
    )
    if (
        payload["status"] != "PASS"
        or payload["canonical_pretrain_eligible"] is not True
        or payload["control_band_contract_sha256"] != _contract_entry()["sha256"]
        or _require_positive_int(payload["verified_item_count"], label="lineage item count") < 1
        or payload["component_cross_split_count"] != 0
        or payload["source_sha_cross_split_count"] != 0
        or payload["original_lineage_cross_split_count"] != 0
        or payload["recorded_synthetic_lineage_intersection_count"] != 0
        or payload["actual_manifest_rows_consumed"] is not True
        or _require_positive_int(
            payload["recorded_clip_count"], label="lineage recorded clip count"
        )
        < 1
        or payload["recorded_synthetic_intersection_algorithm"]
        != _RECORDED_SYNTHETIC_INTERSECTION_ALGORITHM
        or payload["actual_recorded_holdout_bytes_consumed"] is not True
    ):
        raise NotebookExchangeError("lineage zero-intersection/actual-row 계약이 아닙니다")


def _qualified_entry(value: object, *, label: str) -> dict[str, Any]:
    keys = {"dataset_index", "component_id", "path", "content_sha256"}
    if not isinstance(value, dict) or set(value) != keys:
        raise NotebookExchangeError(f"{label} qualified source entry가 exact하지 않습니다")
    index = value["dataset_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise NotebookExchangeError(f"{label} dataset_index가 잘못됐습니다")
    component = str(value["component_id"])
    path = str(value["path"])
    if not component or not path:
        raise NotebookExchangeError(f"{label} component/path가 비었습니다")
    _require_sha256(value["content_sha256"], label=f"{label} content SHA")
    return dict(value)


def _qualified_cells(
    value: object, *, octave: bool, label: str
) -> dict[tuple[str, str], list[list[dict[str, Any]]] | list[dict[str, Any]]]:
    if not isinstance(value, dict) or set(value) != set(_SPLITS):
        raise NotebookExchangeError(f"{label} split 집합이 exact하지 않습니다")
    result: dict[tuple[str, str], list[list[dict[str, Any]]] | list[dict[str, Any]]] = {}
    for split in _SPLITS:
        families = value[split]
        if not isinstance(families, dict) or set(families) != set(_FAMILIES):
            raise NotebookExchangeError(f"{label} family 집합이 exact하지 않습니다")
        for family in _FAMILIES:
            raw_cell = families[family]
            if octave:
                if not isinstance(raw_cell, list) or len(raw_cell) != 5:
                    raise NotebookExchangeError(f"{label} octave list는 정확히 5개여야 합니다")
                bands: list[list[dict[str, Any]]] = []
                for octave_index, raw_entries in enumerate(raw_cell):
                    if not isinstance(raw_entries, list) or len(raw_entries) < 4:
                        raise NotebookExchangeError(f"{label} octave qualified source가 4개 미만입니다")
                    entries = [
                        _qualified_entry(
                            item,
                            label=f"{label}.{split}.{family}.octave{octave_index}",
                        )
                        for item in raw_entries
                    ]
                    identities = {
                        (item["dataset_index"], item["component_id"]) for item in entries
                    }
                    if len(identities) != len(entries) or len(
                        {item["component_id"] for item in entries}
                    ) < 4:
                        raise NotebookExchangeError(f"{label} octave component가 중복됩니다")
                    bands.append(entries)
                result[(split, family)] = bands
            else:
                if not isinstance(raw_cell, list) or len(raw_cell) < 4:
                    raise NotebookExchangeError(f"{label} sentinel qualified source가 4개 미만입니다")
                entries = [
                    _qualified_entry(item, label=f"{label}.{split}.{family}.sentinel")
                    for item in raw_cell
                ]
                identities = {
                    (item["dataset_index"], item["component_id"]) for item in entries
                }
                if len(identities) != len(entries) or len(
                    {item["component_id"] for item in entries}
                ) < 4:
                    raise NotebookExchangeError(f"{label} sentinel component가 중복됩니다")
                result[(split, family)] = entries
    return result


def _validate_coverage(payload: Mapping[str, Any], *, expected_commit: str | None) -> None:
    _require_exact_keys(
        payload,
        {
            "schema",
            "status",
            "canonical_pretrain_eligible",
            "control_band_contract_sha256",
            "manifest_bundle_sha256",
            "actual_source_bytes_recomputed",
            "plant_binding_file_sha256",
            "source_density_algorithm",
            "octave_objective_bands_hz",
            "minimum_source_density_ratio",
            "minimum_independent_components_per_family_octave",
            "qualified_sources_by_split_family_octave",
            "one_point_six_khz_sentinel_band_hz",
            "qualified_sources_by_split_family_one_point_six_khz_sentinel",
            "source_inventory_commit_sha",
        },
        label="frequency coverage receipt",
    )
    commit = _require_commit(
        payload["source_inventory_commit_sha"], label="coverage source inventory commit"
    )
    if expected_commit is not None and commit != expected_commit:
        raise NotebookExchangeError("coverage source commit이 status commit과 다릅니다")
    _require_sha256(payload["manifest_bundle_sha256"], label="coverage manifest SHA")
    _require_sha256(payload["plant_binding_file_sha256"], label="coverage plant binding SHA")
    if (
        payload["status"] != "PASS"
        or payload["canonical_pretrain_eligible"] is not True
        or payload["control_band_contract_sha256"] != _contract_entry()["sha256"]
        or payload["actual_source_bytes_recomputed"] is not True
        or payload["source_density_algorithm"]
        != "mono_mean_welch_nperseg8192_noverlap4096_detrend_false_v1"
        or payload["octave_objective_bands_hz"]
        != [list(values) for values in STAGE2_2KHZ_OBJECTIVE_BANDS_HZ]
        or payload["minimum_source_density_ratio"] != 0.25
        or payload["minimum_independent_components_per_family_octave"] != 4
        or payload["one_point_six_khz_sentinel_band_hz"] != _SENTINEL_BAND_HZ
    ):
        raise NotebookExchangeError("coverage Stage-2 density/sentinel 의미가 다릅니다")
    _qualified_cells(
        payload["qualified_sources_by_split_family_octave"],
        octave=True,
        label="coverage",
    )
    _qualified_cells(
        payload["qualified_sources_by_split_family_one_point_six_khz_sentinel"],
        octave=False,
        label="coverage",
    )


def _validate_transfer(payload: Mapping[str, Any], *, expected_commit: str | None) -> None:
    _require_exact_keys(
        payload,
        {
            "schema",
            "status",
            "canonical_pretrain_eligible",
            "control_band_contract_sha256",
            "manifest_bundle_sha256",
            "existing_instance_cache_reused",
            "all_declared_source_bytes_rehashed",
            "stale_run_or_checkpoint_auto_resume_allowed",
            "scratch_new_run_directory_required",
            "source_inventory_commit_sha",
        },
        label="transfer/bootstrap receipt",
    )
    commit = _require_commit(
        payload["source_inventory_commit_sha"], label="transfer source inventory commit"
    )
    if expected_commit is not None and commit != expected_commit:
        raise NotebookExchangeError("transfer source commit이 status commit과 다릅니다")
    _require_sha256(payload["manifest_bundle_sha256"], label="transfer manifest SHA")
    if (
        payload["status"] != "PASS"
        or payload["canonical_pretrain_eligible"] is not True
        or payload["control_band_contract_sha256"] != _contract_entry()["sha256"]
        or payload["existing_instance_cache_reused"] is not True
        or payload["all_declared_source_bytes_rehashed"] is not True
        or payload["stale_run_or_checkpoint_auto_resume_allowed"] is not False
        or payload["scratch_new_run_directory_required"] is not True
    ):
        raise NotebookExchangeError("transfer scratch/re-hash/auto-resume 계약이 다릅니다")


def _validate_typed_receipt_payload(
    kind: str, payload: object, *, expected_commit: str | None = None
) -> str:
    expected_schema = _ARTIFACT_SCHEMAS.get(kind)
    if expected_schema is None:
        raise NotebookExchangeError(f"허용되지 않은 artifact kind입니다: {kind}")
    if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
        raise NotebookExchangeError(f"{kind} receipt schema가 exact하지 않습니다")
    if kind == "checkout_audit":
        _validate_checkout(payload, expected_commit=expected_commit)
    elif kind == "restore_receipt":
        _validate_restore(payload)
    elif kind == "archive_cache_receipt":
        _validate_archive(payload, expected_commit=expected_commit)
    elif kind == "decoder_qa_receipt":
        _validate_decoder(payload, expected_commit=expected_commit)
    elif kind == "manifest_bundle":
        _manifest_rows(payload, expected_commit=expected_commit)
    elif kind == "lineage_receipt":
        _validate_lineage(payload, expected_commit=expected_commit)
    elif kind == "frequency_coverage_receipt":
        _validate_coverage(payload, expected_commit=expected_commit)
    elif kind == "transfer_bootstrap_receipt":
        _validate_transfer(payload, expected_commit=expected_commit)
    else:  # pragma: no cover - kind allowlist 위에서 완전 열거된다.
        raise NotebookExchangeError(f"검증기가 없는 artifact kind입니다: {kind}")
    return expected_schema


class NotebookExchangeError(ValueError):
    """교환 schema, source 또는 remote가 안전 계약을 어길 때 발생한다."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "payload_sha256"}
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_archive_cache_advisory_receipt(
    *,
    manifest: object,
    manifest_file_sha256: str,
    expected_commit: str,
    publisher_script_sha256: str,
) -> dict[str, Any]:
    """exact production publisher manifest를 작은 advisory receipt로 투영한다.

    이 projection은 archive transport/readback 완료만 뜻한다. extracted raw,
    decoder, lineage 또는 canonical training authority가 아니다.
    """

    if not isinstance(manifest, dict):
        raise NotebookExchangeError("archive production manifest root가 object가 아닙니다")
    _require_exact_keys(
        manifest,
        {
            "archive_count",
            "archives",
            "authority",
            "excluded_corpora",
            "kind",
            "publisher_commit",
            "publisher_entry_script_sha256",
            "publisher_pget_sha256",
            "schema_version",
        },
        label="archive production manifest",
    )
    commit = _require_commit(expected_commit, label="archive expected commit")
    manifest_sha = _require_sha256(
        manifest_file_sha256, label="archive production manifest file SHA"
    )
    script_sha = _require_sha256(
        publisher_script_sha256, label="archive publisher script SHA"
    )
    _require_sha256(manifest["publisher_pget_sha256"], label="archive publisher pget SHA")
    archives = manifest["archives"]
    if (
        manifest["schema_version"] != 1
        or manifest["kind"] != "deep_anc_public_archive_cache"
        or manifest["authority"]
        != "transport_acceleration_only_not_raw_or_training_authority"
        or manifest["publisher_commit"] != commit
        or manifest["publisher_entry_script_sha256"] != script_sha
        or manifest["excluded_corpora"]
        != ["esc50", "fma_small", "fma_metadata", "librispeech"]
        or manifest["archive_count"] != len(_ARCHIVE_SIZES)
        or not isinstance(archives, list)
        or len(archives) != len(_ARCHIVE_SIZES)
    ):
        raise NotebookExchangeError("archive production manifest fixed publisher 계약이 다릅니다")
    sizes: dict[str, int] = {}
    shas: dict[str, str] = {}
    for raw in archives:
        if not isinstance(raw, dict):
            raise NotebookExchangeError("archive production entry가 object가 아닙니다")
        archive_id = str(raw.get("archive_id", ""))
        if archive_id not in _ARCHIVE_SIZES or archive_id in sizes:
            raise NotebookExchangeError("archive production allowlist ID가 중복/누락됐습니다")
        size = raw.get("archive_size")
        digest = _require_sha256(raw.get("archive_sha256"), label=f"archive {archive_id} SHA")
        if size != _ARCHIVE_SIZES[archive_id]:
            raise NotebookExchangeError(f"archive {archive_id} size가 fixed allowlist와 다릅니다")
        expected_fragment = f"/{archive_id}/bytes_{size}/sha256_{digest}/"
        cache_path = f"/{str(raw.get('cache_path', ''))}"
        if expected_fragment not in cache_path:
            raise NotebookExchangeError(f"archive {archive_id} cache path가 content-addressed가 아닙니다")
        sizes[archive_id] = int(size)
        shas[archive_id] = digest
    if sizes != _ARCHIVE_SIZES or sum(sizes.values()) != _ARCHIVE_TOTAL_BYTES:
        raise NotebookExchangeError("archive production manifest 10개 aggregate가 다릅니다")
    payload: dict[str, Any] = {
        "schema": _ARTIFACT_SCHEMAS["archive_cache_receipt"],
        "status": "PASS",
        "authority": "transport_readback_advisory_not_raw_or_training_authority",
        "source_commit": commit,
        "production_manifest_sha256": manifest_sha,
        "production_manifest_remote_path": (
            f"manifests/v1/sha256_{manifest_sha}/archive_cache_manifest.json"
        ),
        "archive_count": len(sizes),
        "archive_total_bytes": sum(sizes.values()),
        "archive_sizes_by_id": dict(sorted(sizes.items())),
        "archive_sha256_by_id": dict(sorted(shas.items())),
        "immutable_content_addressed_archive_paths": True,
        # exact tracked publisher는 각 archive와 manifest 모두 rclone check --download
        # 및 cat SHA readback을 끝낸 뒤에만 이 manifest를 반환한다.
        "publisher_source_sha256_verified": True,
        "publisher_archive_and_manifest_readback_enforced": True,
        "canonical_training_authority": False,
    }
    payload["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    _validate_archive(payload, expected_commit=commit)
    return payload


def _decoder_cohort(relative_path: str) -> str | None:
    prefixes = (
        ("data/raw/noise/dns_fullband/", "dns_fullband"),
        ("data/raw/noise/speech/", "speech"),
        ("data/raw/music/fma_small/", "music"),
        ("data/raw/noise/demand/", "demand"),
        ("data/raw/noise/machine/", "machine"),
        ("data/raw/noise/esc50/ESC-50-master/audio/", "esc50"),
    )
    for prefix, name in prefixes:
        if relative_path.startswith(prefix):
            return name
    return None


def build_full_decoder_advisory_receipt(
    *,
    report: object,
    report_file_sha256: str,
    expected_commit: str,
) -> dict[str, Any]:
    """실제 decoder audit 37,761 rows를 전수 검사해 작은 advisory를 만든다."""

    if not isinstance(report, dict):
        raise NotebookExchangeError("decoder audit root가 object가 아닙니다")
    _require_exact_keys(
        report,
        {
            "schema_version",
            "status",
            "root_label",
            "audit_policy",
            "decoder_fingerprint",
            "decoder_fingerprint_sha256",
            "inventory",
            "inventory_sha256",
            "accepted_inventory_sha256",
            "summary",
            "audit_sha256",
        },
        label="decoder audit",
    )
    commit = _require_commit(expected_commit, label="decoder expected commit")
    file_sha = _require_sha256(report_file_sha256, label="decoder audit file SHA")
    if report["schema_version"] != 1 or report["status"] != "complete":
        raise NotebookExchangeError("decoder audit schema/status가 complete v1이 아닙니다")
    fingerprint = report["decoder_fingerprint"]
    if hashlib.sha256(canonical_json_bytes(fingerprint)).hexdigest() != report[
        "decoder_fingerprint_sha256"
    ]:
        raise NotebookExchangeError("decoder fingerprint SHA가 다릅니다")
    policy = report["audit_policy"]
    if not isinstance(policy, dict):
        raise NotebookExchangeError("decoder audit policy가 object가 아닙니다")
    chunks = policy.get("sequential_chunk_frames")
    if (
        not isinstance(chunks, list)
        or not {65_536, 262_144}.issubset(chunks)
        or not isinstance(policy.get("segment_frames"), int)
        or int(policy["segment_frames"]) <= 0
        or not isinstance(policy.get("segment_grid_denominator"), int)
        or int(policy["segment_grid_denominator"]) <= 0
    ):
        raise NotebookExchangeError("decoder audit가 full sequential+seek-grid policy가 아닙니다")
    inventory = report["inventory"]
    if not isinstance(inventory, list) or len(inventory) != _DECODER_TOTAL_COUNT:
        raise NotebookExchangeError("decoder audit가 full 37,761 inventory가 아닙니다")
    if hashlib.sha256(canonical_json_bytes(inventory)).hexdigest() != report["inventory_sha256"]:
        raise NotebookExchangeError("decoder inventory SHA가 actual rows와 다릅니다")
    seen_paths: set[str] = set()
    cohort_counts = {name: 0 for name in _DECODER_COHORT_COUNTS}
    accepted_cohort_counts = {name: 0 for name in _DECODER_COHORT_COUNTS}
    rejected_cohort_counts = {name: 0 for name in _DECODER_COHORT_COUNTS}
    accepted_projection: list[dict[str, Any]] = []
    row_keys = {
        "relative_path",
        "content_sha256",
        "content_size",
        "header",
        "scan",
        "findings",
        "decision",
    }
    for row in inventory:
        if not isinstance(row, dict) or set(row) != row_keys:
            raise NotebookExchangeError("decoder inventory row key 집합이 exact하지 않습니다")
        relative = str(row["relative_path"])
        cohort = _decoder_cohort(relative)
        if cohort is None or relative in seen_paths:
            raise NotebookExchangeError("decoder inventory path가 unknown/duplicate cohort입니다")
        seen_paths.add(relative)
        cohort_counts[cohort] += 1
        digest = _require_sha256(row["content_sha256"], label="decoder raw content SHA")
        size = _require_positive_int(row["content_size"], label="decoder raw content_size")
        scan = row["scan"]
        sequential = scan.get("sequential") if isinstance(scan, dict) else None
        segments = scan.get("segment_grid") if isinstance(scan, dict) else None
        if not isinstance(row["findings"], list):
            raise NotebookExchangeError("decoder inventory findings 형식이 잘못됐습니다")
        if row["decision"] == "accept":
            if (
                row["findings"] != []
                or not isinstance(row["header"], dict)
                or not isinstance(sequential, list)
                or not isinstance(segments, list)
                or not segments
            ):
                raise NotebookExchangeError("decoder accept row가 full accepted decode가 아닙니다")
            by_phase = {
                str(item.get("phase")): item
                for item in sequential
                if isinstance(item, dict)
            }
            for required_chunk in (65_536, 262_144):
                scan_row = by_phase.get(f"sequential_{required_chunk}")
                if (
                    not isinstance(scan_row, dict)
                    or scan_row.get("chunk_frames") != required_chunk
                    or scan_row.get("start_frame") is not None
                    or scan_row.get("requested_frames") is not None
                    or not isinstance(scan_row.get("frames_read"), int)
                    or int(scan_row["frames_read"]) <= 0
                    or scan_row.get("expected_frames") != scan_row.get("frames_read")
                    or not isinstance(scan_row.get("chunks"), int)
                    or int(scan_row["chunks"]) <= 0
                    or not isinstance(scan_row.get("rms"), (int, float))
                    or float(scan_row["rms"]) <= 0.0
                    or any(key in scan_row for key in ("stderr", "warnings", "error"))
                ):
                    raise NotebookExchangeError(
                        "decoder accept row가 필수 full sequential scan을 통과하지 않았습니다"
                    )
            accepted_cohort_counts[cohort] += 1
            accepted_projection.append(
                {
                    "relative_path": relative,
                    "content_sha256": digest,
                    "content_size": size,
                }
            )
        elif row["decision"] == "reject":
            if not row["findings"]:
                raise NotebookExchangeError("decoder reject row에 실제 finding이 없습니다")
            if row["header"] is not None and not isinstance(row["header"], dict):
                raise NotebookExchangeError("decoder reject row header 형식이 잘못됐습니다")
            rejected_cohort_counts[cohort] += 1
        else:
            raise NotebookExchangeError("decoder decision은 accept/reject만 허용합니다")
    if cohort_counts != _DECODER_COHORT_COUNTS:
        raise NotebookExchangeError("decoder cohort별 full count가 37,761 계약과 다릅니다")
    if (
        accepted_cohort_counts != _DECODER_ACCEPTED_COHORT_COUNTS
        or rejected_cohort_counts != _DECODER_REJECTED_COHORT_COUNTS
    ):
        raise NotebookExchangeError("decoder cohort별 accept/reject exact partition이 다릅니다")
    accepted_sha = hashlib.sha256(canonical_json_bytes(accepted_projection)).hexdigest()
    if accepted_sha != report["accepted_inventory_sha256"]:
        raise NotebookExchangeError("decoder accepted inventory SHA가 actual accept rows와 다릅니다")
    expected_summary = {
        "candidate_count": _DECODER_TOTAL_COUNT,
        "accepted_count": _DECODER_ACCEPTED_COUNT,
        "rejected_count": _DECODER_REJECTED_COUNT,
    }
    if report["summary"] != expected_summary:
        raise NotebookExchangeError("decoder summary가 full/no-rejection 계약과 다릅니다")
    semantic_sha = _require_sha256(report["audit_sha256"], label="decoder semantic SHA")
    unsigned = {key: value for key, value in report.items() if key != "audit_sha256"}
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != semantic_sha:
        raise NotebookExchangeError("decoder audit semantic self digest가 다릅니다")
    payload: dict[str, Any] = {
        "schema": _ARTIFACT_SCHEMAS["decoder_qa_receipt"],
        "status": "PASS",
        "authority": "full_decoder_projection_not_training_authority",
        "source_commit": commit,
        "decoder_audit_file_sha256": file_sha,
        "decoder_audit_semantic_sha256": semantic_sha,
        "inventory_sha256": report["inventory_sha256"],
        "accepted_inventory_sha256": report["accepted_inventory_sha256"],
        **expected_summary,
        "cohort_counts": cohort_counts,
        "accepted_cohort_counts": accepted_cohort_counts,
        "rejected_cohort_counts": rejected_cohort_counts,
        "full_sequential_chunk_frames": [65_536, 262_144],
        "full_inventory_rows_consumed": True,
        "partial_only": False,
        "canonical_training_authority": False,
    }
    payload["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    _validate_decoder(payload, expected_commit=commit)
    return payload


def validate_remote_root(value: str) -> str:
    if not _SAFE_REMOTE.fullmatch(value) or value.endswith(":"):
        raise NotebookExchangeError("안전한 rclone remote:path 형식이 아닙니다")
    _remote, raw_path = value.split(":", 1)
    parts = PurePosixPath(raw_path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise NotebookExchangeError("remote root에 traversal/empty component가 있습니다")
    return value.rstrip("/")


def _validate_receipt(
    kind: str, path: str | Path, *, expected_commit: str | None = None
) -> dict[str, Any]:
    if not _SAFE_KIND.fullmatch(kind):
        raise NotebookExchangeError(f"artifact kind가 안전하지 않습니다: {kind!r}")
    receipt = Path(path)
    try:
        info = receipt.lstat()
    except OSError as exc:
        raise NotebookExchangeError(f"receipt를 읽을 수 없습니다: {receipt}") from exc
    if not stat.S_ISREG(info.st_mode) or receipt.is_symlink():
        raise NotebookExchangeError(f"receipt는 regular non-symlink 파일이어야 합니다: {receipt}")
    if info.st_size < 1 or info.st_size > _MAX_RECEIPT_BYTES:
        raise NotebookExchangeError(
            f"receipt 크기는 1..{_MAX_RECEIPT_BYTES} bytes여야 합니다: {receipt}"
        )
    if receipt.suffix.lower() not in _ALLOWED_RECEIPT_SUFFIXES:
        raise NotebookExchangeError(f"허용되지 않은 receipt 확장자입니다: {receipt.suffix}")
    if not _SAFE_NAME.fullmatch(receipt.name):
        raise NotebookExchangeError(f"receipt basename이 안전하지 않습니다: {receipt.name!r}")
    if receipt.suffix.lower() != ".json":
        raise NotebookExchangeError("PASS typed artifact는 schema가 있는 JSON receipt여야 합니다")
    try:
        content = receipt.read_bytes()
        payload = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NotebookExchangeError(f"receipt JSON을 검증할 수 없습니다: {receipt}") from exc
    schema = _validate_typed_receipt_payload(
        kind, payload, expected_commit=expected_commit
    )
    if len(content) != int(info.st_size):
        raise NotebookExchangeError("receipt bytes가 lstat 이후 바뀌었습니다")
    return {
        "kind": kind,
        "name": receipt.name,
        "size_bytes": int(info.st_size),
        "sha256": hashlib.sha256(content).hexdigest(),
        "schema": schema,
        "local_path": str(receipt.resolve()),
        "_payload": payload,
    }


def _validate_phase_artifact_payloads(
    *,
    phase: str,
    artifacts: Sequence[Mapping[str, Any]],
    expected_commit: str,
) -> None:
    """한 PASS status의 실제 receipt bytes 사이 결속을 다시 계산한다."""

    if phase not in PHASES:
        raise NotebookExchangeError(f"알 수 없는 phase입니다: {phase}")
    by_kind = {str(item.get("kind")): item for item in artifacts}
    required = _PHASE_REQUIRED_ARTIFACT_KINDS[phase]
    if set(by_kind) != set(required):
        raise NotebookExchangeError(
            f"{phase} PASS artifact kind 집합이 exact하지 않습니다: "
            f"expected={sorted(required)}, actual={sorted(by_kind)}"
        )
    for kind, item in by_kind.items():
        payload = item.get("_payload")
        schema = _validate_typed_receipt_payload(
            kind, payload, expected_commit=expected_commit
        )
        if schema != item.get("schema"):
            raise NotebookExchangeError(f"{kind} schema가 artifact descriptor와 다릅니다")

    manifest_item = by_kind.get("manifest_bundle")
    if manifest_item is None:
        return
    manifest_payload = manifest_item["_payload"]
    assert isinstance(manifest_payload, dict)
    rows = _manifest_rows(manifest_payload, expected_commit=expected_commit)
    manifest_sha = str(manifest_item["sha256"])

    lineage_item = by_kind.get("lineage_receipt")
    if lineage_item is not None:
        lineage = lineage_item["_payload"]
        assert isinstance(lineage, dict)
        if (
            lineage["manifest_bundle_sha256"] != manifest_sha
            or lineage["verified_item_count"] != len(rows)
            or lineage["source_inventory_commit_sha"]
            != manifest_payload["source_inventory_commit_sha"]
        ):
            raise NotebookExchangeError("lineage receipt가 실제 manifest SHA/items/commit과 다릅니다")

    coverage_item = by_kind.get("frequency_coverage_receipt")
    if coverage_item is not None:
        coverage = coverage_item["_payload"]
        assert isinstance(coverage, dict)
        if (
            coverage["manifest_bundle_sha256"] != manifest_sha
            or coverage["source_inventory_commit_sha"]
            != manifest_payload["source_inventory_commit_sha"]
        ):
            raise NotebookExchangeError("coverage가 실제 manifest SHA/commit과 다릅니다")
        octave_cells = _qualified_cells(
            coverage["qualified_sources_by_split_family_octave"],
            octave=True,
            label="coverage",
        )
        sentinel_cells = _qualified_cells(
            coverage[
                "qualified_sources_by_split_family_one_point_six_khz_sentinel"
            ],
            octave=False,
            label="coverage",
        )
        for (split, family), raw_bands in octave_cells.items():
            assert isinstance(raw_bands, list)
            bands = raw_bands
            for entries in bands:
                assert isinstance(entries, list)
                for entry in entries:
                    row = rows.get(int(entry["dataset_index"]))
                    projection = (
                        None
                        if row is None
                        else {
                            "dataset_index": row["dataset_index"],
                            "component_id": row["component_id"],
                            "path": row["path"],
                            "content_sha256": row["content_sha256"],
                        }
                    )
                    if (
                        entry != projection
                        or row is None
                        or row["split"] != split
                        or row["source_family"] != family
                    ):
                        raise NotebookExchangeError(
                            "coverage octave qualified source가 manifest actual row와 다릅니다"
                        )
        for (split, family), raw_entries in sentinel_cells.items():
            assert isinstance(raw_entries, list)
            for entry in raw_entries:
                row = rows.get(int(entry["dataset_index"]))
                projection = (
                    None
                    if row is None
                    else {
                        "dataset_index": row["dataset_index"],
                        "component_id": row["component_id"],
                        "path": row["path"],
                        "content_sha256": row["content_sha256"],
                    }
                )
                if (
                    entry != projection
                    or row is None
                    or row["split"] != split
                    or row["source_family"] != family
                ):
                    raise NotebookExchangeError(
                        "coverage 1.6 kHz qualified source가 manifest actual row와 다릅니다"
                    )

    transfer_item = by_kind.get("transfer_bootstrap_receipt")
    if transfer_item is not None:
        transfer = transfer_item["_payload"]
        assert isinstance(transfer, dict)
        if (
            transfer["manifest_bundle_sha256"] != manifest_sha
            or transfer["source_inventory_commit_sha"]
            != manifest_payload["source_inventory_commit_sha"]
        ):
            raise NotebookExchangeError("transfer receipt가 실제 manifest SHA/commit과 다릅니다")
        # lineage/coverage/transfer의 실제 file SHA는 status artifact_bundle_sha256가
        # 한 번 더 묶는다. 각 receipt의 manifest SHA까지 위에서 동일함을 확인하므로
        # bundle_publish는 네 독립 bytes가 같은 manifest를 가리키는 cross-SHA graph다.


def build_status(
    *,
    source_commit: str,
    phase: str,
    state: str,
    message: str,
    artifacts: Sequence[Mapping[str, Any]],
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    if not _HEX40.fullmatch(source_commit):
        raise NotebookExchangeError("source_commit은 소문자 40자리 SHA여야 합니다")
    if phase not in PHASES:
        raise NotebookExchangeError(f"알 수 없는 phase입니다: {phase}")
    if state not in STATES:
        raise NotebookExchangeError(f"알 수 없는 state입니다: {state}")
    clean_message = str(message).strip()
    if not clean_message or len(clean_message) > 1000 or "\x00" in clean_message:
        raise NotebookExchangeError("message는 1..1000자의 일반 텍스트여야 합니다")
    created = created_at_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not _UTC_TIMESTAMP.fullmatch(created):
        raise NotebookExchangeError("created_at_utc가 canonical UTC 초 단위 형식이 아닙니다")
    try:
        datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise NotebookExchangeError("created_at_utc가 유효한 UTC 시각이 아닙니다") from exc
    normalized_artifacts: list[dict[str, Any]] = []
    for artifact in artifacts:
        kind = str(artifact.get("kind", ""))
        name = str(artifact.get("name", ""))
        digest = str(artifact.get("sha256", ""))
        remote_path = str(artifact.get("remote_path", ""))
        schema = str(artifact.get("schema", ""))
        size = artifact.get("size_bytes")
        if (
            not _SAFE_KIND.fullmatch(kind)
            or not _SAFE_NAME.fullmatch(name)
            or not _SAFE_NAME.fullmatch(schema)
            or not _HEX64.fullmatch(digest)
        ):
            raise NotebookExchangeError("artifact kind/name/SHA 형식이 잘못됐습니다")
        expected_schema = _ARTIFACT_SCHEMAS.get(kind)
        if expected_schema is None or schema != expected_schema:
            raise NotebookExchangeError("artifact kind/schema 조합이 allowlist와 다릅니다")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise NotebookExchangeError("artifact size가 양의 정수가 아닙니다")
        expected_remote = f"receipts/sha256_{digest}/{name}"
        if remote_path != expected_remote:
            raise NotebookExchangeError("artifact remote path가 content-addressed 형식이 아닙니다")
        normalized_artifacts.append(
            {
                "kind": kind,
                "name": name,
                "schema": schema,
                "remote_path": remote_path,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    normalized_artifacts.sort(
        key=lambda item: (item["kind"], item["sha256"], item["name"])
    )
    if len({item["remote_path"] for item in normalized_artifacts}) != len(
        normalized_artifacts
    ):
        raise NotebookExchangeError("artifact remote path가 중복됩니다")
    if len({item["kind"] for item in normalized_artifacts}) != len(
        normalized_artifacts
    ):
        raise NotebookExchangeError("한 status 안에서 artifact kind는 중복될 수 없습니다")
    if state == "PASS":
        required_kinds = _PHASE_REQUIRED_ARTIFACT_KINDS[phase]
        actual_kinds = {item["kind"] for item in normalized_artifacts}
        if actual_kinds != set(required_kinds):
            raise NotebookExchangeError(
                f"{phase} PASS artifact kind 집합이 exact하지 않습니다: "
                f"expected={sorted(required_kinds)}, actual={sorted(actual_kinds)}"
            )
    artifact_bundle_sha256 = hashlib.sha256(
        canonical_json_bytes(normalized_artifacts)
    ).hexdigest()
    payload: dict[str, Any] = {
        "schema": STATUS_SCHEMA,
        "campaign": CAMPAIGN,
        "source_commit": source_commit,
        "created_at_utc": created,
        "phase": phase,
        "state": state,
        "message": clean_message,
        "artifacts": normalized_artifacts,
        "artifact_bundle_sha256": artifact_bundle_sha256,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def validate_status(value: object, *, expected_commit: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NotebookExchangeError("status 최상위는 object여야 합니다")
    required = {
        "schema",
        "campaign",
        "source_commit",
        "created_at_utc",
        "phase",
        "state",
        "message",
        "artifacts",
        "artifact_bundle_sha256",
        "payload_sha256",
    }
    if set(value) != required:
        raise NotebookExchangeError("status key 집합이 schema와 다릅니다")
    source_commit = str(value["source_commit"])
    if expected_commit is not None and source_commit != expected_commit:
        raise NotebookExchangeError("status source commit이 expected commit과 다릅니다")
    rebuilt = build_status(
        source_commit=source_commit,
        phase=str(value["phase"]),
        state=str(value["state"]),
        message=str(value["message"]),
        artifacts=value["artifacts"] if isinstance(value["artifacts"], list) else (),
        created_at_utc=str(value["created_at_utc"]),
    )
    if value.get("schema") != STATUS_SCHEMA or value.get("campaign") != CAMPAIGN:
        raise NotebookExchangeError("status schema/campaign이 다릅니다")
    if value.get("artifact_bundle_sha256") != rebuilt["artifact_bundle_sha256"]:
        raise NotebookExchangeError("status artifact bundle SHA가 다릅니다")
    if str(value.get("payload_sha256")) != rebuilt["payload_sha256"]:
        raise NotebookExchangeError("status payload SHA가 다릅니다")
    return rebuilt


def summarize_statuses(
    statuses: Sequence[Mapping[str, Any]], *, expected_commit: str
) -> dict[str, Any]:
    if not _HEX40.fullmatch(expected_commit):
        raise NotebookExchangeError("expected commit은 소문자 40자리 SHA여야 합니다")
    latest: dict[str, dict[str, Any]] = {}
    for raw in statuses:
        status = validate_status(dict(raw), expected_commit=expected_commit)
        phase = str(status["phase"])
        previous = latest.get(phase)
        if (
            previous is not None
            and status["created_at_utc"] == previous["created_at_utc"]
            and status["payload_sha256"] != previous["payload_sha256"]
        ):
            raise NotebookExchangeError(
                f"{phase}에 동일 초의 서로 다른 append-only status가 있어 latest가 모호합니다"
            )
        if previous is None or str(status["created_at_utc"]) > str(
            previous["created_at_utc"]
        ):
            latest[phase] = status
    structural_pass = all(
        latest.get(phase, {}).get("state") == "PASS" for phase in PHASES
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "campaign": CAMPAIGN,
        "expected_commit": expected_commit,
        "completion_scope": "INCOMPLETE",
        "advisory_complete": False,
        "all_phase_statuses_structurally_pass": structural_pass,
        "semantic_receipts_verified": False,
        # Notebook receipt는 transport/선행 계산이다. tracked Git authority, held
        # source bytes, strict P/S 및 A100 smoke를 대체하지 못한다.
        "canonical_pretrain_ready": False,
        "all_required_phases_pass": False,
        "missing_phases": [phase for phase in PHASES if phase not in latest],
        "latest_by_phase": {phase: latest[phase] for phase in PHASES if phase in latest},
    }


def _mark_summary_semantically_verified(summary: Mapping[str, Any]) -> dict[str, Any]:
    """read_remote_statuses의 actual receipt loop 뒤에서만 advisory 완료를 연다."""

    if summary.get("schema") != SUMMARY_SCHEMA:
        raise NotebookExchangeError("notebook summary schema가 다릅니다")
    structural_pass = summary.get("all_phase_statuses_structurally_pass") is True
    return {
        **summary,
        "completion_scope": "ADVISORY_COMPLETE" if structural_pass else "INCOMPLETE",
        "advisory_complete": structural_pass,
        "semantic_receipts_verified": True,
        "all_required_phases_pass": structural_pass,
        "canonical_pretrain_ready": False,
    }


def assert_exact_checkout(*, repository_root: str | Path, expected_commit: str) -> None:
    if not _HEX40.fullmatch(expected_commit):
        raise NotebookExchangeError("expected commit은 소문자 40자리 SHA여야 합니다")
    root = Path(repository_root).resolve()
    env = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"}
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()
    if head != expected_commit:
        raise NotebookExchangeError(f"checkout HEAD 불일치: {head} != {expected_commit}")
    symbolic = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if symbolic.returncode == 0:
        raise NotebookExchangeError(
            f"notebook checkout은 detached HEAD여야 합니다: {symbolic.stdout.strip()}"
        )
    if symbolic.returncode != 1:
        raise NotebookExchangeError("checkout detached HEAD 상태를 검증할 수 없습니다")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout
    if dirty:
        raise NotebookExchangeError("checkout이 clean하지 않습니다")


def _run_rclone(args: Sequence[str], *, timeout_seconds: int) -> bytes:
    completed = subprocess.run(
        list(args),
        check=True,
        capture_output=True,
        timeout=timeout_seconds,
    )
    return completed.stdout


def publish_status(
    *,
    remote_root: str,
    repository_root: str | Path,
    expected_commit: str,
    phase: str,
    state: str,
    message: str,
    receipt_paths: Sequence[tuple[str, str | Path]],
    rclone_executable: str = "rclone",
    timeout_seconds: int = 3600,
) -> dict[str, Any]:
    root = validate_remote_root(remote_root)
    assert_exact_checkout(repository_root=repository_root, expected_commit=expected_commit)
    local_artifacts = [
        _validate_receipt(kind, path, expected_commit=expected_commit)
        for kind, path in receipt_paths
    ]
    if state == "PASS":
        _validate_phase_artifact_payloads(
            phase=phase,
            artifacts=local_artifacts,
            expected_commit=expected_commit,
        )
    artifacts: list[dict[str, Any]] = []
    for item in local_artifacts:
        relative = f"receipts/sha256_{item['sha256']}/{item['name']}"
        _run_rclone(
            [
                rclone_executable,
                "copyto",
                item["local_path"],
                f"{root}/{relative}",
                "--immutable",
                "--transfers",
                "1",
                "--checkers",
                "1",
            ],
            timeout_seconds=timeout_seconds,
        )
        artifacts.append(
            {
                "kind": item["kind"],
                "name": item["name"],
                "schema": item["schema"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
                "remote_path": relative,
            }
        )
    status = build_status(
        source_commit=expected_commit,
        phase=phase,
        state=state,
        message=message,
        artifacts=artifacts,
    )
    stamp = str(status["created_at_utc"]).replace("-", "").replace(":", "")
    name = (
        f"{stamp}_{expected_commit[:12]}_{phase}_"
        f"{status['payload_sha256'][:16]}.json"
    )
    raw = canonical_json_bytes(status) + b"\n"
    with tempfile.NamedTemporaryFile(prefix="deep_anc_notebook_", suffix=".json") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
        _run_rclone(
            [
                rclone_executable,
                "copyto",
                stream.name,
                f"{root}/status/{name}",
                "--immutable",
                "--transfers",
                "1",
                "--checkers",
                "1",
            ],
            timeout_seconds=timeout_seconds,
        )
    return {**status, "remote_status_path": f"status/{name}"}


def read_remote_statuses(
    *,
    remote_root: str,
    expected_commit: str,
    rclone_executable: str = "rclone",
    timeout_seconds: int = 300,
    maximum_status_files: int = 256,
) -> dict[str, Any]:
    root = validate_remote_root(remote_root)
    if maximum_status_files < len(PHASES) or maximum_status_files > 4096:
        raise NotebookExchangeError("maximum_status_files 범위가 잘못됐습니다")
    listing_raw = _run_rclone(
        [rclone_executable, "lsjson", f"{root}/status", "--files-only"],
        timeout_seconds=timeout_seconds,
    )
    try:
        listing = json.loads(listing_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NotebookExchangeError("rclone status listing이 UTF-8 JSON이 아닙니다") from exc
    if not isinstance(listing, list):
        raise NotebookExchangeError("rclone status listing이 list가 아닙니다")
    names: list[str] = []
    for row in listing:
        if not isinstance(row, dict) or bool(row.get("IsDir")):
            continue
        name = str(row.get("Path") or "")
        match = _STATUS_NAME.fullmatch(name)
        if match and match.group(2) == expected_commit[:12]:
            names.append(name)
    names = sorted(set(names))[-maximum_status_files:]
    statuses: list[dict[str, Any]] = []
    for name in names:
        raw = _run_rclone(
            [rclone_executable, "cat", f"{root}/status/{name}"],
            timeout_seconds=timeout_seconds,
        )
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NotebookExchangeError(f"status JSON이 손상됐습니다: {name}") from exc
        statuses.append(validate_status(value, expected_commit=expected_commit))
    verified_artifacts: dict[str, dict[str, Any]] = {}
    for status in statuses:
        if status["state"] != "PASS":
            continue
        semantic_artifacts: list[dict[str, Any]] = []
        for artifact in status["artifacts"]:
            remote_path = str(artifact["remote_path"])
            if remote_path in verified_artifacts:
                semantic_artifacts.append(
                    {**artifact, "_payload": verified_artifacts[remote_path]}
                )
                continue
            content = _run_rclone(
                [rclone_executable, "cat", f"{root}/{remote_path}"],
                timeout_seconds=timeout_seconds,
            )
            if len(content) != int(artifact["size_bytes"]):
                raise NotebookExchangeError("remote receipt size가 status와 다릅니다")
            if hashlib.sha256(content).hexdigest() != artifact["sha256"]:
                raise NotebookExchangeError("remote receipt SHA가 status와 다릅니다")
            try:
                receipt = json.loads(content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise NotebookExchangeError("remote receipt가 UTF-8 JSON이 아닙니다") from exc
            schema = _validate_typed_receipt_payload(
                str(artifact["kind"]),
                receipt,
                expected_commit=expected_commit,
            )
            if schema != artifact["schema"]:
                raise NotebookExchangeError("remote receipt schema가 status와 다릅니다")
            verified_artifacts[remote_path] = receipt
            semantic_artifacts.append({**artifact, "_payload": receipt})
        _validate_phase_artifact_payloads(
            phase=str(status["phase"]),
            artifacts=semantic_artifacts,
            expected_commit=expected_commit,
        )
    summary = _mark_summary_semantically_verified(
        summarize_statuses(statuses, expected_commit=expected_commit)
    )
    summary["remote_root"] = root
    summary["status_files_read"] = len(names)
    summary["remote_artifacts_verified"] = len(verified_artifacts)
    return summary


__all__ = [
    "CAMPAIGN",
    "DEFAULT_REMOTE_ROOT",
    "NotebookExchangeError",
    "PHASES",
    "STATES",
    "assert_exact_checkout",
    "build_archive_cache_advisory_receipt",
    "build_full_decoder_advisory_receipt",
    "build_status",
    "canonical_json_bytes",
    "payload_sha256",
    "publish_status",
    "read_remote_statuses",
    "sha256_file",
    "summarize_statuses",
    "validate_status",
]
