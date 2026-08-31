"""Stage-2 공개 사전학습 입력 4종의 raw-bound production issuer.

Elice ``bootstrap_all.sh``가 만든 canonical-v4 공개 manifest 세대를 실제 원본
bytes와 함께 다시 검증한 뒤, Stage-2 typed admission이 소비하는 manifest/lineage/
frequency-coverage/transfer receipt를 만든다. 출력은 기존 경로를 덮어쓰지 않으며,
실패한 부분 출력도 자동 삭제하지 않는다.

이 모듈은 Git authority를 발행하지 않는다. 생성 결과는 별도 human review 후
``authority/stage2_2khz_public_data.json``과 campaign config에 결속돼야 한다.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import welch

from ..dsp.stage2_2khz_contract import (
    STAGE2_2KHZ_SOURCE_FAMILIES,
    Stage2TwoKilohertzContract,
)
from .manifest_contract import MANIFEST_GENERATION_FILE, validate_manifest_generation
from .public_lineage import PUBLIC_LINEAGE_SCHEMA, validate_recorded_clip_lineage
from .transfer_contract import (
    TransferContractError,
    _validate_archive_cache_bootstrap_binding,
    require_stage2_2khz_physical_transfer_manifest,
)


STAGE2_PRETRAIN_DATA_ISSUER_SCHEMA = "stage2_2khz_public_data_issuer_v1"
STAGE2_PUBLIC_MANIFEST_BUNDLE_SCHEMA = "stage2_2khz_public_manifest_bundle_v1"
STAGE2_PUBLIC_LINEAGE_RECEIPT_SCHEMA = "stage2_2khz_public_lineage_receipt_v2"
STAGE2_PUBLIC_COVERAGE_RECEIPT_SCHEMA = "stage2_2khz_public_frequency_coverage_v2"
STAGE2_TRANSFER_BOOTSTRAP_RECEIPT_SCHEMA = (
    "stage2_2khz_transfer_bootstrap_receipt_v1"
)
STAGE2_PRETRAIN_PUBLICATION_INTENT_SCHEMA = (
    "stage2_2khz_public_data_publication_intent_v1"
)
STAGE2_PRETRAIN_PUBLICATION_COMPLETE_SCHEMA = (
    "stage2_2khz_public_data_publication_complete_v1"
)

REQUIRED_CANONICAL_TAGS = (
    "demand",
    "dns_fullband",
    "esc50",
    "machine",
    "music",
    "speech",
)
TAG_TO_FAMILY = {
    "demand": "environment",
    "dns_fullband": "environment",
    "esc50": "environment",
    "machine": "machine",
    "music": "music",
    "speech": "speech",
}
SPLITS = ("train", "val", "test")
RECORDED_HOLDOUT_PATH = "data/manifests/recorded_holdout.json"
ELICE_BOOTSTRAP_RECEIPT_PATH = "data/manifests/elice_bootstrap_receipt.json"
ELICE_TRANSFER_MANIFEST_PATH = "data/manifests/elice_transfer_manifest.json"
SOURCE_DENSITY_ALGORITHM = (
    "mono_mean_welch_nperseg8192_noverlap4096_detrend_false_v1"
)
RECORDED_SYNTHETIC_INTERSECTION_ALGORITHM = (
    "transitive_basename_content_sha256_lineage_keys_v1"
)
ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ = (1425.437949, 1795.939277)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class Stage2PretrainDataIssueError(ValueError):
    """실제 공개 source/receipt가 발행 계약을 만족하지 않음."""


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def snapshot_regular(path: Path) -> tuple[bytes, str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Stage2PretrainDataIssueError(f"regular file만 허용합니다: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise Stage2PretrainDataIssueError(f"snapshot 중 파일이 바뀌었습니다: {path}")
    content = b"".join(chunks)
    if len(content) != int(after.st_size):
        raise Stage2PretrainDataIssueError(f"snapshot 크기가 바뀌었습니다: {path}")
    return content, sha256_bytes(content), len(content)


def repository_file(root: Path, raw: object, *, label: str) -> Path:
    candidate = Path(str(raw or ""))
    if candidate.is_absolute():
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise Stage2PretrainDataIssueError(f"{label}를 찾을 수 없습니다") from exc
    else:
        if not candidate.parts or ".." in candidate.parts:
            raise Stage2PretrainDataIssueError(f"{label} 경로가 안전하지 않습니다")
        try:
            resolved = (root / candidate).resolve(strict=True)
        except OSError as exc:
            raise Stage2PretrainDataIssueError(f"{label}를 찾을 수 없습니다") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise Stage2PretrainDataIssueError(f"{label}가 repository 밖입니다") from exc
    cursor = root
    for part in resolved.relative_to(root).parts:
        cursor /= part
        if cursor.is_symlink():
            raise Stage2PretrainDataIssueError(f"{label} 경로에 symlink가 있습니다")
    return resolved


def _json_object(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2PretrainDataIssueError(f"{label}가 UTF-8 JSON이 아닙니다") from exc
    if not isinstance(value, dict):
        raise Stage2PretrainDataIssueError(f"{label} root가 object가 아닙니다")
    return value


def validate_elice_bootstrap_inputs(
    root: Path,
    *,
    source_inventory_commit_sha: str,
    bootstrap_receipt_path: str = ELICE_BOOTSTRAP_RECEIPT_PATH,
    expected_bootstrap_receipt_sha256: str | None = None,
    require_stage2_physical_transfer: bool = False,
) -> dict[str, Any]:
    """bootstrap receipt와 그것이 지시하는 holdout/transfer/freeze를 다시 읽는다.

    schema-v3의 archive-cache 상태도 여기서 transfer-contract verifier로 재검증한다.
    따라서 plain no-cache bootstrap은 cache를 재사용했다고 발행할 수 없고, cache
    bootstrap은 completion/inventory/decoder binding을 다시 통과해야 한다.
    """

    if not _HEX40.fullmatch(source_inventory_commit_sha):
        raise Stage2PretrainDataIssueError("source inventory commit은 40자리 SHA여야 합니다")
    if (
        expected_bootstrap_receipt_sha256 is not None
        and not _HEX64.fullmatch(expected_bootstrap_receipt_sha256)
    ):
        raise Stage2PretrainDataIssueError(
            "expected Elice bootstrap receipt SHA는 64자리 SHA여야 합니다"
        )
    bootstrap_path = repository_file(root, bootstrap_receipt_path, label="Elice bootstrap receipt")
    bootstrap_bytes, bootstrap_sha, _ = snapshot_regular(bootstrap_path)
    if (
        expected_bootstrap_receipt_sha256 is not None
        and bootstrap_sha != expected_bootstrap_receipt_sha256
    ):
        raise Stage2PretrainDataIssueError(
            "Elice bootstrap receipt SHA가 transfer receipt ref와 다릅니다"
        )
    payload = _json_object(bootstrap_bytes, label="Elice bootstrap receipt")
    keys = {
        "schema_version",
        "expected_commit",
        "canonical_holdout",
        "transfer_manifest",
        "recorded_aggregate_sha256",
        "archive_cache_consumption",
        "recorded_subband_coverage",
        "environment",
    }
    if set(payload) != keys or payload.get("schema_version") != 3:
        raise Stage2PretrainDataIssueError("Elice bootstrap receipt schema-v3 key 집합이 다릅니다")
    if payload.get("expected_commit") != source_inventory_commit_sha:
        raise Stage2PretrainDataIssueError("bootstrap receipt commit이 source inventory commit과 다릅니다")
    try:
        archive_cache_binding = _validate_archive_cache_bootstrap_binding(root, payload)
    except TransferContractError as exc:
        raise Stage2PretrainDataIssueError(
            "bootstrap archive-cache provenance 검증에 실패했습니다"
        ) from exc

    holdout_ref = payload.get("canonical_holdout")
    transfer_ref = payload.get("transfer_manifest")
    environment = payload.get("environment")
    if not isinstance(holdout_ref, dict) or set(holdout_ref) != {"path", "sha256"}:
        raise Stage2PretrainDataIssueError("bootstrap holdout ref가 exact하지 않습니다")
    if not isinstance(transfer_ref, dict) or set(transfer_ref) != {"path", "sha256"}:
        raise Stage2PretrainDataIssueError("bootstrap transfer ref가 exact하지 않습니다")
    if holdout_ref.get("path") != RECORDED_HOLDOUT_PATH:
        raise Stage2PretrainDataIssueError("bootstrap holdout 경로가 canonical이 아닙니다")
    if transfer_ref.get("path") != ELICE_TRANSFER_MANIFEST_PATH:
        raise Stage2PretrainDataIssueError("bootstrap transfer 경로가 canonical이 아닙니다")
    if not isinstance(environment, dict) or set(environment) != {
        "freeze_receipt",
        "freeze_receipt_sha256",
        "torch_version",
        "torch_cuda",
    }:
        raise Stage2PretrainDataIssueError("bootstrap environment ref가 exact하지 않습니다")
    if environment.get("torch_version") != "2.5.1+cu121" or environment.get("torch_cuda") != "12.1":
        raise Stage2PretrainDataIssueError("bootstrap torch/CUDA 계약이 다릅니다")

    verified: dict[str, dict[str, object]] = {}
    for label, ref in (("recorded_holdout", holdout_ref), ("transfer_manifest", transfer_ref)):
        path = repository_file(root, ref["path"], label=label)
        _content, digest, size = snapshot_regular(path)
        if digest != ref.get("sha256") or not _HEX64.fullmatch(digest):
            raise Stage2PretrainDataIssueError(f"bootstrap {label} SHA가 actual bytes와 다릅니다")
        verified[label] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": digest,
            "size": size,
        }
    stage2_physical_transfer: dict[str, dict[str, str]] | None = None
    if require_stage2_physical_transfer:
        try:
            stage2_physical_transfer = require_stage2_2khz_physical_transfer_manifest(
                root / str(transfer_ref["path"]),
                repo_root=root,
                expected_sha256=str(transfer_ref["sha256"]),
            )
        except (OSError, TransferContractError) as exc:
            raise Stage2PretrainDataIssueError(
                "Stage-2 canonical admission에는 actual physical P/S schema-v3 transfer가 필요합니다"
            ) from exc
    freeze_path = repository_file(root, environment["freeze_receipt"], label="environment freeze")
    _freeze, freeze_sha, freeze_size = snapshot_regular(freeze_path)
    if freeze_sha != environment.get("freeze_receipt_sha256"):
        raise Stage2PretrainDataIssueError("environment freeze SHA가 bootstrap receipt와 다릅니다")
    verified["environment_freeze"] = {
        "path": freeze_path.relative_to(root).as_posix(),
        "sha256": freeze_sha,
        "size": freeze_size,
    }
    return {
        "bootstrap_receipt": {
            "path": bootstrap_path.relative_to(root).as_posix(),
            "sha256": bootstrap_sha,
            "size": len(bootstrap_bytes),
        },
        "archive_cache_reused": archive_cache_binding is not None,
        **(
            {"stage2_2khz_physical_transfer": stage2_physical_transfer}
            if stage2_physical_transfer is not None
            else {}
        ),
        **verified,
    }


def build_transfer_bootstrap_receipt(
    *,
    manifest_bundle_sha256: str,
    source_inventory_commit_sha: str,
    bootstrap_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """actual Elice bootstrap ref/cache state를 Stage-2 transfer receipt에 봉인한다."""

    if not _HEX64.fullmatch(manifest_bundle_sha256):
        raise Stage2PretrainDataIssueError("manifest bundle SHA는 64자리 SHA여야 합니다")
    if not _HEX40.fullmatch(source_inventory_commit_sha):
        raise Stage2PretrainDataIssueError("source inventory commit은 40자리 SHA여야 합니다")
    origin = bootstrap_inputs.get("bootstrap_receipt")
    if (
        not isinstance(origin, Mapping)
        or set(origin) != {"path", "sha256", "size"}
        or not isinstance(origin.get("path"), str)
        or not origin["path"]
        or not _HEX64.fullmatch(str(origin.get("sha256")))
        or type(origin.get("size")) is not int
        or int(origin["size"]) < 1
    ):
        raise Stage2PretrainDataIssueError("Elice bootstrap receipt ref가 exact하지 않습니다")
    cache_reused = bootstrap_inputs.get("archive_cache_reused")
    if type(cache_reused) is not bool:
        raise Stage2PretrainDataIssueError("Elice bootstrap cache 재사용 상태가 bool이 아닙니다")
    contract = Stage2TwoKilohertzContract.canonical()
    return {
        "schema": STAGE2_TRANSFER_BOOTSTRAP_RECEIPT_SCHEMA,
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract_sha256": contract.digest(),
        "manifest_bundle_sha256": manifest_bundle_sha256,
        "elice_bootstrap_receipt": {
            "path": str(origin["path"]),
            "sha256": str(origin["sha256"]),
        },
        "existing_instance_cache_reused": cache_reused,
        "all_declared_source_bytes_rehashed": True,
        "stale_run_or_checkpoint_auto_resume_allowed": False,
        "scratch_new_run_directory_required": True,
        "source_inventory_commit_sha": source_inventory_commit_sha,
    }


def build_manifest_items(
    root: Path,
    validated_entries: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    if tuple(sorted(validated_entries)) != REQUIRED_CANONICAL_TAGS:
        raise Stage2PretrainDataIssueError(
            "canonical-v4 manifest tag는 DNS/speech/music/DEMAND/MIMII/ESC-50 exact 6종이어야 합니다"
        )
    staged: list[dict[str, Any]] = []
    for tag in REQUIRED_CANONICAL_TAGS:
        family = TAG_TO_FAMILY[tag]
        rows = validated_entries[tag]
        if not rows:
            raise Stage2PretrainDataIssueError(f"canonical manifest가 비었습니다: {tag}")
        for source in rows:
            path = repository_file(root, source.get("path"), label=f"{tag} source")
            _content, actual_sha, actual_size = snapshot_regular(path)
            declared_sha = str(source.get("content_sha256") or "")
            declared_size = source.get("content_size")
            if actual_sha != declared_sha or actual_size != declared_size:
                raise Stage2PretrainDataIssueError(f"{tag} source SHA/size가 actual bytes와 다릅니다")
            split = str(source.get("split") or "")
            component = str(source.get("group_id") or "")
            lineage = source.get("lineage_keys")
            rate = source.get("sample_rate")
            if (
                split not in SPLITS
                or not component.startswith("public-lineage-")
                or source.get("lineage_schema") != PUBLIC_LINEAGE_SCHEMA
                or not isinstance(lineage, list)
                or not lineage
                or lineage != sorted(set(lineage))
                or isinstance(rate, bool)
                or not isinstance(rate, int)
                or rate < 5657
            ):
                raise Stage2PretrainDataIssueError(f"{tag} source의 split/lineage/rate가 유효하지 않습니다")
            staged.append(
                {
                    "source_family": family,
                    "component_id": component,
                    "split": split,
                    "path": path.relative_to(root).as_posix(),
                    "content_sha256": actual_sha,
                    "content_size": actual_size,
                    "native_sample_rate": int(rate),
                    "native_nyquist_hz": float(rate) / 2.0,
                    "lineage_keys": list(lineage),
                }
            )
    staged.sort(
        key=lambda row: (
            row["source_family"],
            row["split"],
            row["component_id"],
            row["path"],
            row["content_sha256"],
        )
    )
    items = [{"dataset_index": index, **row} for index, row in enumerate(staged)]
    component_splits: dict[str, set[str]] = defaultdict(set)
    component_families: dict[str, set[str]] = defaultdict(set)
    content_components: dict[str, set[str]] = defaultdict(set)
    lineage_components: dict[str, set[str]] = defaultdict(set)
    for row in items:
        component_splits[row["component_id"]].add(row["split"])
        component_families[row["component_id"]].add(row["source_family"])
        content_components[row["content_sha256"]].add(row["component_id"])
        for key in row["lineage_keys"]:
            lineage_components[key].add(row["component_id"])
    if any(len(values) != 1 for values in component_splits.values()):
        raise Stage2PretrainDataIssueError("public component가 split을 가로지릅니다")
    if any(len(values) != 1 for values in component_families.values()):
        raise Stage2PretrainDataIssueError("public component가 source family를 가로지릅니다")
    if any(len(values) != 1 for values in content_components.values()):
        raise Stage2PretrainDataIssueError("동일 source SHA가 여러 component에 있습니다")
    if any(len(values) != 1 for values in lineage_components.values()):
        raise Stage2PretrainDataIssueError("동일 original lineage key가 여러 component에 있습니다")
    for split in SPLITS:
        for family in STAGE2_2KHZ_SOURCE_FAMILIES:
            components = {
                row["component_id"]
                for row in items
                if row["split"] == split and row["source_family"] == family
            }
            if len(components) < 4:
                raise Stage2PretrainDataIssueError(
                    f"split×family independent component가 4개 미만입니다: {split}/{family}"
                )
    return items


def stage2_recorded_public_intersection(
    *,
    recorded_rows: Sequence[Mapping[str, Any]],
    public_items: Sequence[Mapping[str, Any]],
) -> int:
    """generic basename 중복끼리는 합치지 않고 recorded와의 실제 교집합만 센다.

    canonical public ``component_id``가 content/lineage closure를 이미 봉인한다.
    basename은 recorded↔public 직접 일치 증거로 사용하되, DEMAND의 ``ch01.wav``처럼
    서로 무관한 public source를 basename만으로 하나의 component로 합치지 않는다.
    """

    recorded_basenames = {str(row["clip"]).casefold() for row in recorded_rows}
    recorded_content = {str(row["content_sha256"]) for row in recorded_rows}
    recorded_lineage = {
        str(key) for row in recorded_rows for key in row["lineage_keys"]
    }
    # 이 helper의 저수준 negative fixture는 full manifest schema 검증 전 단일 row로
    # 호출될 수 있다. 그 경우에만 각 row를 독립 component로 취급한다. production
    # bundle은 앞선 validator가 non-empty component_id를 강제한다.
    component_ids = [
        str(row.get("component_id") or f"__unbound_row_{index}")
        for index, row in enumerate(public_items)
    ]
    overlapping_components = {
        component_ids[index]
        for index, row in enumerate(public_items)
        if (
            Path(str(row["path"])).name.casefold() in recorded_basenames
            or str(row["content_sha256"]) in recorded_content
            or bool(set(str(key) for key in row["lineage_keys"]) & recorded_lineage)
        )
    }
    return sum(
        1 for component in component_ids if component in overlapping_components
    )


def build_lineage_receipt(
    root: Path,
    *,
    items: Sequence[Mapping[str, Any]],
    manifest_bundle_sha256: str,
    source_inventory_commit_sha: str,
) -> dict[str, Any]:
    holdout_path = repository_file(root, RECORDED_HOLDOUT_PATH, label="recorded holdout")
    holdout_bytes, holdout_sha, _ = snapshot_regular(holdout_path)
    holdout = _json_object(holdout_bytes, label="recorded holdout")
    lineage = holdout.get("clip_lineage")
    families = holdout.get("families")
    if not isinstance(lineage, dict) or not isinstance(families, dict):
        raise Stage2PretrainDataIssueError("recorded holdout clip_lineage/families가 없습니다")
    recorded = validate_recorded_clip_lineage(lineage, families=families)
    intersection = stage2_recorded_public_intersection(
        recorded_rows=recorded,
        public_items=items,
    )
    if intersection != 0:
        raise Stage2PretrainDataIssueError(
            f"recorded/public transitive lineage 교집합이 0이 아닙니다: {intersection}"
        )
    return {
        "schema": STAGE2_PUBLIC_LINEAGE_RECEIPT_SCHEMA,
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract_sha256": Stage2TwoKilohertzContract.canonical().digest(),
        "manifest_bundle_sha256": manifest_bundle_sha256,
        "verified_item_count": len(items),
        "component_cross_split_count": 0,
        "source_sha_cross_split_count": 0,
        "original_lineage_cross_split_count": 0,
        "recorded_synthetic_lineage_intersection_count": 0,
        "actual_manifest_rows_consumed": True,
        "recorded_holdout": {"path": RECORDED_HOLDOUT_PATH, "sha256": holdout_sha},
        "recorded_clip_count": len(recorded),
        "recorded_clip_lineage_sha256": str(lineage["clips_sha256"]),
        "recorded_synthetic_intersection_algorithm": RECORDED_SYNTHETIC_INTERSECTION_ALGORITHM,
        "actual_recorded_holdout_bytes_consumed": True,
        "source_inventory_commit_sha": source_inventory_commit_sha,
    }


def _density(content: bytes, *, sample_rate: int, contract: Stage2TwoKilohertzContract) -> tuple[list[float], float]:
    try:
        audio, actual_rate = sf.read(io.BytesIO(content), dtype="float64", always_2d=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise Stage2PretrainDataIssueError("public source bytes를 decode할 수 없습니다") from exc
    values = np.asarray(audio, dtype=np.float64)
    if int(actual_rate) != sample_rate or values.ndim != 2 or values.shape[0] < 256 or not np.all(np.isfinite(values)):
        raise Stage2PretrainDataIssueError("public source PCM/rate가 coverage 계산에 부적격합니다")
    mono = np.mean(values, axis=1, dtype=np.float64)
    nperseg = min(8192, int(mono.size))
    noverlap = min(4096, nperseg - 1)
    frequency, psd = welch(mono, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, detrend=False)
    baseline_mask = (frequency >= contract.octave_objective_bands_hz[0][0]) & (
        frequency < contract.octave_objective_bands_hz[-1][1]
    )
    baseline = float(np.mean(psd[baseline_mask]))
    ratios: list[float] = []
    for lower, upper in contract.octave_objective_bands_hz:
        mask = (frequency >= lower) & (frequency < upper)
        density = float(np.mean(psd[mask]))
        ratios.append(0.0 if baseline <= np.finfo(np.float64).tiny else density / baseline)
    lower, upper = ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ
    sentinel_density = float(np.mean(psd[(frequency >= lower) & (frequency < upper)]))
    sentinel = 0.0 if baseline <= np.finfo(np.float64).tiny else sentinel_density / baseline
    if not math.isfinite(baseline) or any(not math.isfinite(value) or value < 0.0 for value in (*ratios, sentinel)):
        raise Stage2PretrainDataIssueError("public source density가 finite한 0 이상이 아닙니다")
    return ratios, sentinel


def build_coverage_receipt(
    root: Path,
    *,
    items: Sequence[Mapping[str, Any]],
    manifest_bundle_sha256: str,
    plant_binding_file_sha256: str,
    source_inventory_commit_sha: str,
    workers: int | None = None,
) -> dict[str, Any]:
    if not _HEX64.fullmatch(plant_binding_file_sha256):
        raise Stage2PretrainDataIssueError("plant binding file SHA가 유효하지 않습니다")
    contract = Stage2TwoKilohertzContract.canonical()
    octave = {
        split: {family: [[] for _ in contract.octave_objective_bands_hz] for family in STAGE2_2KHZ_SOURCE_FAMILIES}
        for split in SPLITS
    }
    sentinel = {
        split: {family: [] for family in STAGE2_2KHZ_SOURCE_FAMILIES}
        for split in SPLITS
    }
    if workers is None:
        worker_count = min(16, max(1, int(os.cpu_count() or 1)))
    elif (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 16
    ):
        raise Stage2PretrainDataIssueError(
            "coverage workers는 1..16 integer여야 합니다"
        )
    else:
        worker_count = workers

    def analyze(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[float], float]:
        path = repository_file(root, row["path"], label="coverage source")
        content, digest, size = snapshot_regular(path)
        if digest != row["content_sha256"] or size != row["content_size"]:
            raise Stage2PretrainDataIssueError("coverage source가 manifest bytes와 다릅니다")
        ratios, sentinel_ratio = _density(
            content,
            sample_rate=int(row["native_sample_rate"]),
            contract=contract,
        )
        return row, ratios, sentinel_ratio

    # executor.map은 입력 순서로 결과/예외를 surface한다. worker scheduling과
    # 무관하게 qualified list와 receipt bytes가 동일하다.
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="stage2-density",
    ) as executor:
        analyzed = executor.map(analyze, items)
        for row, ratios, sentinel_ratio in analyzed:
            entry = {
                "dataset_index": int(row["dataset_index"]),
                "component_id": str(row["component_id"]),
                "path": str(row["path"]),
                "content_sha256": str(row["content_sha256"]),
            }
            for index, ratio in enumerate(ratios):
                if ratio >= contract.minimum_source_density_ratio:
                    octave[row["split"]][row["source_family"]][index].append(entry)
            if sentinel_ratio >= contract.minimum_source_density_ratio:
                sentinel[row["split"]][row["source_family"]].append(entry)
    minimum = int(contract.minimum_groups_per_family_octave)
    for split in SPLITS:
        for family in STAGE2_2KHZ_SOURCE_FAMILIES:
            for index, entries in enumerate(octave[split][family]):
                if len({row["component_id"] for row in entries}) < minimum:
                    raise Stage2PretrainDataIssueError(
                        f"source density component가 4개 미만입니다: {split}/{family}/octave{index}"
                    )
            if len({row["component_id"] for row in sentinel[split][family]}) < minimum:
                raise Stage2PretrainDataIssueError(
                    f"1.6kHz sentinel component가 4개 미만입니다: {split}/{family}"
                )
    return {
        "schema": STAGE2_PUBLIC_COVERAGE_RECEIPT_SCHEMA,
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract_sha256": contract.digest(),
        "manifest_bundle_sha256": manifest_bundle_sha256,
        "actual_source_bytes_recomputed": True,
        "plant_binding_file_sha256": plant_binding_file_sha256,
        "source_density_algorithm": SOURCE_DENSITY_ALGORITHM,
        "octave_objective_bands_hz": [[float(a), float(b)] for a, b in contract.octave_objective_bands_hz],
        "minimum_source_density_ratio": 0.25,
        "minimum_independent_components_per_family_octave": 4,
        "qualified_sources_by_split_family_octave": octave,
        "one_point_six_khz_sentinel_band_hz": list(ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ),
        "qualified_sources_by_split_family_one_point_six_khz_sentinel": sentinel,
        "source_inventory_commit_sha": source_inventory_commit_sha,
    }


def build_stage2_pretrain_data_payloads(
    root: Path,
    *,
    manifest_dir: Path,
    plant_binding_path: Path,
    expected_plant_binding_sha256: str,
    source_inventory_commit_sha: str,
    bootstrap_receipt_path: str = ELICE_BOOTSTRAP_RECEIPT_PATH,
    workers: int | None = None,
) -> dict[str, dict[str, Any]]:
    """canonical generation과 actual source를 읽고 발행할 payload를 만든다."""

    # production P/S loader는 tracked human authority부터 확인하고 P/S/raw/analysis/
    # level/clock actual bytes를 전부 다시 검증한다. JSON의 PASS scalar만으로는 issuer를
    # 열지 않는다. local import는 data manifest helper의 import cycle을 피한다.
    from ..train.stage2_2khz_binding import (  # noqa: PLC0415
        STAGE2_2KHZ_PHYSICAL_AUTHORITY_PATH,
        load_stage2_2khz_plant_binding,
    )

    plant_bytes, plant_sha, plant_size = snapshot_regular(plant_binding_path)
    if plant_sha != expected_plant_binding_sha256 or not _HEX64.fullmatch(plant_sha):
        raise Stage2PretrainDataIssueError("plant binding external SHA가 actual bytes와 다릅니다")
    try:
        plant_payload = json.loads(plant_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2PretrainDataIssueError("plant binding이 JSON이 아닙니다") from exc
    if not isinstance(plant_payload, dict) or plant_payload.get("status") != "PASS":
        raise Stage2PretrainDataIssueError("plant binding status가 PASS가 아닙니다")
    try:
        plant_binding = load_stage2_2khz_plant_binding(
            plant_binding_path,
            repository_root=root,
            expected_binding_file_sha256=plant_sha,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise Stage2PretrainDataIssueError(
            "plant binding physical Git authority/actual bytes 검증에 실패했습니다"
        ) from exc
    if plant_binding.fixture_only:
        raise Stage2PretrainDataIssueError(
            "fixture/diagnostic plant binding은 production issuer에 사용할 수 없습니다"
        )
    authority_path = repository_file(
        root,
        STAGE2_2KHZ_PHYSICAL_AUTHORITY_PATH,
        label="Stage-2 physical Git authority",
    )
    _authority_bytes, authority_sha, authority_size = snapshot_regular(authority_path)

    bootstrap = validate_elice_bootstrap_inputs(
        root,
        source_inventory_commit_sha=source_inventory_commit_sha,
        bootstrap_receipt_path=bootstrap_receipt_path,
        require_stage2_physical_transfer=True,
    )
    transferred_physical = bootstrap.get("stage2_2khz_physical_transfer")
    expected_transferred_physical = {
        "plant_binding": {
            "path": plant_binding_path.relative_to(root).as_posix(),
            "sha256": plant_sha,
        },
        "physical_authority": {
            "path": authority_path.relative_to(root).as_posix(),
            "sha256": authority_sha,
        },
    }
    if transferred_physical != expected_transferred_physical:
        raise Stage2PretrainDataIssueError(
            "Stage-2 plant binding/physical authority가 validated transfer typed role과 다릅니다"
        )
    generation = validate_manifest_generation(
        manifest_dir,
        required_tags=REQUIRED_CANONICAL_TAGS,
        repo_root=root,
    )
    entries = generation.get("_validated_entries")
    if not isinstance(entries, dict):
        raise Stage2PretrainDataIssueError("validated canonical generation entries가 없습니다")
    items = build_manifest_items(root, entries)
    contract = Stage2TwoKilohertzContract.canonical()
    manifest = {
        "schema": STAGE2_PUBLIC_MANIFEST_BUNDLE_SCHEMA,
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract": {"id": contract.contract_id, "sha256": contract.digest()},
        "required_source_families": list(STAGE2_2KHZ_SOURCE_FAMILIES),
        "required_splits": list(SPLITS),
        "recorded_artifacts_required_for_pretrain": False,
        "test_split_for_checkpoint_selection_allowed": False,
        "source_inventory_commit_sha": source_inventory_commit_sha,
        "items": items,
    }
    manifest_sha = sha256_bytes(canonical_json_bytes(manifest))
    lineage = build_lineage_receipt(
        root,
        items=items,
        manifest_bundle_sha256=manifest_sha,
        source_inventory_commit_sha=source_inventory_commit_sha,
    )
    coverage = build_coverage_receipt(
        root,
        items=items,
        manifest_bundle_sha256=manifest_sha,
        plant_binding_file_sha256=plant_sha,
        source_inventory_commit_sha=source_inventory_commit_sha,
        workers=workers,
    )
    transfer = build_transfer_bootstrap_receipt(
        manifest_bundle_sha256=manifest_sha,
        source_inventory_commit_sha=source_inventory_commit_sha,
        bootstrap_inputs=bootstrap,
    )
    generation_path = manifest_dir / MANIFEST_GENERATION_FILE
    _generation_bytes, generation_sha, generation_size = snapshot_regular(generation_path)
    issuer = {
        "schema": STAGE2_PRETRAIN_DATA_ISSUER_SCHEMA,
        "status": "PASS_CANDIDATE_REQUIRES_GIT_AUTHORITY",
        "source_inventory_commit_sha": source_inventory_commit_sha,
        "plant_binding": {
            "path": plant_binding_path.relative_to(root).as_posix(),
            "sha256": plant_sha,
            "size": plant_size,
            "runtime_sha256": plant_binding.digest(),
            "physical_git_authority": {
                "path": authority_path.relative_to(root).as_posix(),
                "sha256": authority_sha,
                "size": authority_size,
            },
        },
        "canonical_manifest_generation": {
            "path": generation_path.relative_to(root).as_posix(),
            "sha256": generation_sha,
            "size": generation_size,
            "build_id": generation["build_id"],
        },
        "bootstrap_inputs": bootstrap,
        "artifact_sha256": {
            "manifest_bundle": manifest_sha,
            "lineage_receipt": sha256_bytes(canonical_json_bytes(lineage)),
            "frequency_coverage_receipt": sha256_bytes(canonical_json_bytes(coverage)),
            "transfer_bootstrap_receipt": sha256_bytes(canonical_json_bytes(transfer)),
        },
        "source_item_count": len(items),
        "actual_source_bytes_rehashed_and_decoded": True,
    }
    return {
        "manifest_bundle.json": manifest,
        "lineage_receipt.json": lineage,
        "frequency_coverage_receipt.json": coverage,
        "transfer_bootstrap_receipt.json": transfer,
        "issuer_receipt.json": issuer,
    }


_PUBLICATION_INTENT_NAME = "publication_intent.json"
_PUBLICATION_COMPLETE_NAME = "publication_complete.json"
_PRODUCTION_PAYLOAD_NAMES = (
    "manifest_bundle.json",
    "lineage_receipt.json",
    "frequency_coverage_receipt.json",
    "transfer_bootstrap_receipt.json",
    "issuer_receipt.json",
)


def _write_bytes_exclusive(path: Path, content: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"publication write가 진행되지 않습니다: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_payloads_noreplace(
    output_dir: Path,
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """새 directory에 intent-first로 쓰고 기존/부분 결과를 보존한다.

    모든 payload를 먼저 canonical serialize한 다음 directory를 선점한다. 중간에
    process가 종료되면 intent만 있고 completion이 없는 forensic generation이 남는다.
    이 generation은 재사용·덮어쓰기·권위 승격이 불가능하다.
    """

    serialized: dict[str, bytes] = {}
    for name, payload in payloads.items():
        if (
            Path(name).name != name
            or not name.endswith(".json")
            or name in {_PUBLICATION_INTENT_NAME, _PUBLICATION_COMPLETE_NAME}
        ):
            raise Stage2PretrainDataIssueError(
                f"output 이름이 안전하지 않습니다: {name}"
            )
        serialized[name] = canonical_json_bytes(payload)
    if not serialized:
        raise Stage2PretrainDataIssueError("publication payload가 비었습니다")
    digests = {
        name: sha256_bytes(content) for name, content in sorted(serialized.items())
    }
    issuer = payloads.get("issuer_receipt.json")
    source_commit = (
        str(issuer.get("source_inventory_commit_sha") or "")
        if isinstance(issuer, Mapping)
        else None
    )
    intent = {
        "schema": STAGE2_PRETRAIN_PUBLICATION_INTENT_SCHEMA,
        "status": "WRITE_IN_PROGRESS_NOT_AUTHORITY",
        "source_inventory_commit_sha": source_commit,
        "expected_artifact_sha256": digests,
        "expected_artifact_count": len(digests),
        "no_replace": True,
        "completion_required_for_candidate_use": True,
    }
    intent_bytes = canonical_json_bytes(intent)

    try:
        output_dir.mkdir(parents=False, mode=0o755)
    except FileExistsError as exc:
        raise Stage2PretrainDataIssueError(
            f"output directory가 이미 있습니다: {output_dir}"
        ) from exc
    _write_bytes_exclusive(output_dir / _PUBLICATION_INTENT_NAME, intent_bytes)
    _fsync_directory(output_dir)
    ordered_names = [name for name in _PRODUCTION_PAYLOAD_NAMES if name in serialized]
    ordered_names.extend(sorted(set(serialized) - set(ordered_names)))
    for name in ordered_names:
        _write_bytes_exclusive(output_dir / name, serialized[name])
    _fsync_directory(output_dir)
    return digests


def seal_published_payloads_noreplace(
    output_dir: Path,
    *,
    artifact_sha256: Mapping[str, str],
    validated_record_count: int,
) -> tuple[dict[str, Any], str]:
    """독립 candidate revalidation 후에만 completion을 O_EXCL로 발행한다."""

    if isinstance(validated_record_count, bool) or validated_record_count <= 0:
        raise Stage2PretrainDataIssueError(
            "validated record count가 0보다 커야 합니다"
        )
    intent_path = output_dir / _PUBLICATION_INTENT_NAME
    intent_bytes, intent_sha, _ = snapshot_regular(intent_path)
    intent = _json_object(intent_bytes, label="publication intent")
    expected_intent_keys = {
        "schema",
        "status",
        "source_inventory_commit_sha",
        "expected_artifact_sha256",
        "expected_artifact_count",
        "no_replace",
        "completion_required_for_candidate_use",
    }
    if (
        set(intent) != expected_intent_keys
        or intent.get("schema") != STAGE2_PRETRAIN_PUBLICATION_INTENT_SCHEMA
        or intent.get("status") != "WRITE_IN_PROGRESS_NOT_AUTHORITY"
        or intent.get("no_replace") is not True
        or intent.get("completion_required_for_candidate_use") is not True
    ):
        raise Stage2PretrainDataIssueError(
            "publication intent가 exact하지 않습니다"
        )
    declared = intent.get("expected_artifact_sha256")
    actual_expected = dict(
        sorted((str(key), str(value)) for key, value in artifact_sha256.items())
    )
    if (
        set(actual_expected) != set(_PRODUCTION_PAYLOAD_NAMES)
        or declared != actual_expected
        or intent.get("expected_artifact_count") != len(actual_expected)
    ):
        raise Stage2PretrainDataIssueError(
            "publication intent/artifact 집합이 exact하지 않습니다"
        )
    allowed = {_PUBLICATION_INTENT_NAME, *actual_expected}
    try:
        actual_names = {entry.name for entry in output_dir.iterdir()}
    except OSError as exc:
        raise Stage2PretrainDataIssueError(
            "publication directory를 읽을 수 없습니다"
        ) from exc
    if actual_names != allowed:
        raise Stage2PretrainDataIssueError(
            "publication directory에 missing/extra residue가 있습니다"
        )
    for name, expected_sha in actual_expected.items():
        _content, actual_sha, _size = snapshot_regular(output_dir / name)
        if actual_sha != expected_sha or not _HEX64.fullmatch(actual_sha):
            raise Stage2PretrainDataIssueError(
                f"publication artifact bytes가 intent와 다릅니다: {name}"
            )
    completion = {
        "schema": STAGE2_PRETRAIN_PUBLICATION_COMPLETE_SCHEMA,
        "status": "PASS_CANDIDATE_REVALIDATED_REQUIRES_GIT_AUTHORITY",
        "publication_intent_sha256": intent_sha,
        "artifact_sha256": actual_expected,
        "validated_record_count": int(validated_record_count),
        "actual_artifact_bytes_rehashed": True,
        "independent_candidate_revalidation_complete": True,
        "git_training_authority_granted": False,
    }
    completion_bytes = canonical_json_bytes(completion)
    _write_bytes_exclusive(output_dir / _PUBLICATION_COMPLETE_NAME, completion_bytes)
    _fsync_directory(output_dir)
    return completion, sha256_bytes(completion_bytes)


__all__ = [
    "REQUIRED_CANONICAL_TAGS",
    "Stage2PretrainDataIssueError",
    "build_transfer_bootstrap_receipt",
    "build_stage2_pretrain_data_payloads",
    "publish_payloads_noreplace",
    "seal_published_payloads_noreplace",
    "stage2_recorded_public_intersection",
    "validate_elice_bootstrap_inputs",
]
