"""합성 코퍼스 manifest 한 세대의 무결성 계약.

JSONL 파일 하나씩의 존재만 확인하면 중단된 빌드가 서로 다른 holdout/config 세대를
섞어도 학습이 시작된다. ``prepare_noise_pool.py``가 남기는 sidecar를 다시 계산하여
필수 태그, 각 파일 SHA, holdout/config SHA와 build identity를 함께 검증한다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable

from ..config import REPO_ROOT
from .decoder_audit import (
    DEFAULT_AUDIO_EXTENSIONS,
    DEFAULT_SEGMENT_FRAMES,
    DEFAULT_SEGMENT_GRID_DENOMINATOR,
    DEFAULT_SEQUENTIAL_CHUNK_FRAMES,
    MAX_DECODED_PCM_ABS,
    MIN_DECODED_RMS,
    decoder_fingerprint,
    discover_audio_files,
)
from .holdout_contract import read_regular_file_snapshot, reject_symlink_components
from .manifest import read_manifest_bytes
from .public_lineage import (
    PUBLIC_LINEAGE_SCHEMA,
    canonical_json_sha256,
    validate_public_manifest_lineage,
    validate_recorded_clip_lineage,
)


MANIFEST_GENERATION_FILE = "manifest_generation.json"
CANONICAL_MANIFEST_SCHEMA_VERSION = 4
DECODER_AUDIT_FILE = "decoder_audit.json"
DECODER_AUDIT_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: dict) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _contract_path(value: object, *, field: str, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest generation {field} 경로가 비었습니다")
    path = Path(value)
    candidate = path if path.is_absolute() else repo_root / path
    return Path(os.path.abspath(candidate))


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} SHA-256가 유효하지 않습니다")
    return value


def _decoder_audit_relative_path(value: object, *, field: str) -> str:
    """audit가 raw 파일을 가리키는 이식 가능한 저장소 상대 POSIX 경로를 검증한다."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}가 비었거나 문자열이 아닙니다")
    if "\\" in value or PureWindowsPath(value).is_absolute():
        raise ValueError(f"{field}는 Windows/절대 경로가 아닌 POSIX 상대 경로여야 합니다")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field}에 절대/상위/빈 경로 component가 있습니다: {value!r}")
    normalised = path.as_posix()
    if normalised == "." or normalised != value:
        raise ValueError(f"{field}는 정규화된 POSIX 상대 경로여야 합니다: {value!r}")
    return normalised


def _canonical_json_digest(value: object, *, field: str) -> str:
    try:
        return canonical_json_sha256(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}가 canonical JSON으로 직렬화되지 않습니다") from exc


def _accepted_decoder_audit_projection(
    inventory: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """decoder audit가 정의한 accepted inventory identity만으로 digest를 재계산한다.

    ``findings``/scan 통계는 forensic evidence라서 accept set의 동일성에는 포함하지
    않는다. 대신 path와 raw bytes identity는 빠짐없이 포함한다.
    """

    return [
        {
            "relative_path": str(row["relative_path"]),
            "content_sha256": str(row["content_sha256"]),
            "content_size": int(row["content_size"]),
        }
        for row in inventory
        if row["decision"] == "accept"
    ]


def _validate_decoder_audit_policy(payload: dict[str, Any], *, label: str) -> None:
    """schema-v1 audit가 실제 canonical scan recipe를 선언하는지 확인한다.

    inventory/SHA만 있으면 얕은 ``sf.info()`` 검사도 full-decode audit인 것처럼
    결속될 수 있다. schema-v1은 full sequential 두 방식과 deterministic seek grid,
    수치 거부 한계를 정확히 고정한다. 다음 recipe 변경은 schema v2로 명시해야 한다.
    """

    policy = payload.get("audit_policy")
    expected_keys = {
        "audio_extensions",
        "sequential_chunk_frames",
        "segment_frames",
        "segment_grid_denominator",
        "max_decoded_pcm_abs",
        "min_decoded_rms",
    }
    if not isinstance(policy, dict) or set(policy) != expected_keys:
        raise ValueError(f"{label} audit_policy schema-v1 필드가 정확하지 않습니다")
    extensions = policy.get("audio_extensions")
    if not isinstance(extensions, list) or extensions != sorted(DEFAULT_AUDIO_EXTENSIONS):
        raise ValueError(f"{label} audit_policy audio_extensions가 canonical 값이 아닙니다")
    chunks = policy.get("sequential_chunk_frames")
    if (
        not isinstance(chunks, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in chunks)
        or chunks != sorted(DEFAULT_SEQUENTIAL_CHUNK_FRAMES)
    ):
        raise ValueError(
            f"{label} audit_policy에는 65536/262144 full sequential scan이 정확히 필요합니다"
        )
    if policy.get("segment_frames") != DEFAULT_SEGMENT_FRAMES:
        raise ValueError(f"{label} audit_policy segment_frames가 canonical 값이 아닙니다")
    if policy.get("segment_grid_denominator") != DEFAULT_SEGMENT_GRID_DENOMINATOR:
        raise ValueError(
            f"{label} audit_policy segment_grid_denominator가 canonical 값이 아닙니다"
        )
    if policy.get("max_decoded_pcm_abs") != MAX_DECODED_PCM_ABS:
        raise ValueError(
            f"{label} audit_policy max_decoded_pcm_abs가 canonical 값이 아닙니다"
        )
    if policy.get("min_decoded_rms") != MIN_DECODED_RMS:
        raise ValueError(f"{label} audit_policy min_decoded_rms가 canonical 값이 아닙니다")


def read_decoder_audit(
    path: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
    label: str = "decoder audit",
) -> dict[str, Any]:
    """완료된 decoder audit JSON 한 개를 같은 FD snapshot으로 읽고 검증한다.

    Audit은 raw 파일 자체를 바꾸지 않는 진단 결과다. 하지만 canonical manifest가
    이를 신뢰하려면 inventory의 정렬·digest·decoder fingerprint까지 모두 확인해야
    한다. 이 함수는 raw bytes를 다시 해시하지 않는다. 그 일은 manifest 행을 실제
    raw snapshot과 결속하는 호출 지점에서 한다.
    """

    root = Path(os.path.abspath(repo_root))
    snapshot = read_regular_file_snapshot(path, root=root, label=label)
    assert snapshot.data is not None
    try:
        payload = json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} JSON 오류: {snapshot.path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 최상위는 JSON object여야 합니다")
    if payload.get("schema_version") != DECODER_AUDIT_SCHEMA_VERSION:
        raise ValueError(
            f"{label} schema_version은 {DECODER_AUDIT_SCHEMA_VERSION}이어야 합니다"
        )
    if payload.get("status") != "complete":
        raise ValueError(f"{label} status=complete 증거가 없습니다")
    _validate_decoder_audit_policy(payload, label=label)

    fingerprint = payload.get("decoder_fingerprint")
    if not isinstance(fingerprint, dict) or not fingerprint:
        raise ValueError(f"{label} decoder_fingerprint가 비었거나 object가 아닙니다")
    fingerprint_sha = _require_sha256(
        payload.get("decoder_fingerprint_sha256"),
        field=f"{label} decoder_fingerprint",
    )
    actual_fingerprint_sha = _canonical_json_digest(
        fingerprint, field=f"{label} decoder_fingerprint"
    )
    if fingerprint_sha != actual_fingerprint_sha:
        raise ValueError(
            f"{label} decoder_fingerprint SHA 불일치: "
            f"expected={fingerprint_sha}, actual={actual_fingerprint_sha}"
        )
    current_fingerprint = decoder_fingerprint()
    if fingerprint != current_fingerprint:
        raise ValueError(
            f"{label} decoder_fingerprint가 현재 runtime과 다릅니다; "
            "다른 SoundFile/libsndfile/libmpg123/Python 환경의 audit은 canonical 학습에 "
            "재사용할 수 없습니다"
        )

    inventory = payload.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError(f"{label} inventory가 비었거나 목록이 아닙니다")
    previous_path: str | None = None
    seen_paths: set[str] = set()
    for index, row in enumerate(inventory):
        if not isinstance(row, dict):
            raise ValueError(f"{label} inventory #{index}가 object가 아닙니다")
        required = {"relative_path", "content_sha256", "content_size", "decision"}
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"{label} inventory #{index} 필수 필드 누락: {missing}")
        relative = _decoder_audit_relative_path(
            row.get("relative_path"), field=f"{label} inventory #{index}.relative_path"
        )
        if relative in seen_paths:
            raise ValueError(
                f"{label} inventory에 중복 relative_path가 있습니다: {relative}"
            )
        if previous_path is not None and relative <= previous_path:
            raise ValueError(f"{label} inventory는 relative_path 오름차순이어야 합니다")
        previous_path = relative
        seen_paths.add(relative)
        _require_sha256(
            row.get("content_sha256"),
            field=f"{label} inventory #{index}.content_sha256",
        )
        size = row.get("content_size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(
                f"{label} inventory #{index}.content_size는 양수 정수여야 합니다"
            )
        if row.get("decision") not in {"accept", "reject"}:
            raise ValueError(
                f"{label} inventory #{index}.decision은 accept/reject여야 합니다"
            )

    inventory_sha = _require_sha256(
        payload.get("inventory_sha256"), field=f"{label} inventory"
    )
    actual_inventory_sha = _canonical_json_digest(inventory, field=f"{label} inventory")
    if inventory_sha != actual_inventory_sha:
        raise ValueError(
            f"{label} inventory SHA 불일치: "
            f"expected={inventory_sha}, actual={actual_inventory_sha}"
        )
    accepted = _accepted_decoder_audit_projection(inventory)
    accepted_sha = _require_sha256(
        payload.get("accepted_inventory_sha256"),
        field=f"{label} accepted inventory",
    )
    actual_accepted_sha = _canonical_json_digest(
        accepted, field=f"{label} accepted inventory"
    )
    if accepted_sha != actual_accepted_sha:
        raise ValueError(
            f"{label} accepted inventory SHA 불일치: "
            f"expected={accepted_sha}, actual={actual_accepted_sha}"
        )

    result = dict(payload)
    result["_snapshot"] = snapshot
    result["_inventory_by_relative_path"] = {
        str(row["relative_path"]): row for row in inventory
    }
    result["_rejected_content_sha256"] = {
        str(row["content_sha256"])
        for row in inventory
        if row["decision"] == "reject"
    }
    return result


def decoder_audit_binding(audit: dict[str, Any]) -> dict[str, Any]:
    """transaction sidecar에 기록할 audit file/identity evidence를 만든다."""

    snapshot = audit.get("_snapshot")
    if snapshot is None or not hasattr(snapshot, "sha256") or not hasattr(snapshot, "size"):
        raise ValueError("decoder audit source snapshot이 없습니다")
    return {
        "schema_version": DECODER_AUDIT_SCHEMA_VERSION,
        "file": DECODER_AUDIT_FILE,
        "sha256": str(snapshot.sha256),
        "size": int(snapshot.size),
        "inventory_sha256": str(audit["inventory_sha256"]),
        "accepted_inventory_sha256": str(audit["accepted_inventory_sha256"]),
        "decoder_fingerprint": audit["decoder_fingerprint"],
        "decoder_fingerprint_sha256": str(audit["decoder_fingerprint_sha256"]),
    }


def _decoder_audit_index(
    audit: dict[str, Any],
    *,
    repo_root: Path,
    raw_roots: Iterable[Path],
    label: str,
) -> dict[str, dict[str, Any]]:
    """audit 상대경로를 raw root 안의 lexical absolute path로 고정한다."""

    roots = [Path(os.path.abspath(root)) for root in raw_roots]
    if not roots:
        raise ValueError(f"{label} raw root가 비었습니다")
    result: dict[str, dict[str, Any]] = {}
    rows = audit.get("inventory")
    if not isinstance(rows, list):  # read_decoder_audit가 보장하지만 API 경계도 닫는다.
        raise ValueError(f"{label} inventory가 없습니다")
    for index, row in enumerate(rows):
        assert isinstance(row, dict)
        relative = _decoder_audit_relative_path(
            row.get("relative_path"), field=f"{label} inventory #{index}.relative_path"
        )
        absolute = Path(os.path.abspath(repo_root / Path(relative)))
        allowed = next(
            (
                root
                for root in roots
                if absolute == root or root in absolute.parents
            ),
            None,
        )
        if allowed is None:
            raise ValueError(
                f"{label} inventory raw path가 declared raw root 밖입니다: {relative}"
            )
        try:
            reject_symlink_components(absolute, root=allowed)
        except Exception as exc:
            raise ValueError(
                f"{label} inventory raw path 경로 계약 위반: {relative}: {exc}"
            ) from exc
        key = str(absolute)
        if key in result:  # relative collision/alias는 canonical 입력으로 금지한다.
            raise ValueError(f"{label} inventory raw path alias가 있습니다: {relative}")
        result[key] = row
    return result


def _validate_decoder_audit_raw_inventory(
    audit: dict[str, Any],
    *,
    repo_root: Path,
    raw_roots: Iterable[Path],
    label: str,
) -> dict[str, dict[str, Any]]:
    """audit inventory가 현재 선언 raw tree 전체와 여전히 같은지 검증한다.

    manifest에 들어가지 않은 reject 파일도 raw tree의 일부다. 그것을 빼면 audit 뒤
    raw가 추가·교체돼도 accepted manifest만 우연히 맞는 상태가 생긴다. 따라서 후보
    경로 집합과 모든 후보의 SHA/size를 audit snapshot과 대조한다.
    """

    roots = [Path(os.path.abspath(root)) for root in raw_roots]
    index = _decoder_audit_index(
        audit,
        repo_root=repo_root,
        raw_roots=roots,
        label=label,
    )
    current_paths = discover_audio_files(roots)
    current_by_path = {str(Path(os.path.abspath(path))): Path(path) for path in current_paths}
    missing = sorted(set(index).difference(current_by_path))
    added = sorted(set(current_by_path).difference(index))
    if missing or added:
        detail: list[str] = []
        if missing:
            detail.append(f"audit 뒤 사라진 raw {len(missing)}개 ({missing[0]})")
        if added:
            detail.append(f"audit 뒤 추가된 raw {len(added)}개 ({added[0]})")
        raise ValueError(f"{label} raw inventory가 audit와 다릅니다: {'; '.join(detail)}")
    for absolute_text, path in sorted(current_by_path.items()):
        row = index[absolute_text]
        allowed = next(
            (root for root in roots if Path(absolute_text) == root or root in Path(absolute_text).parents),
            None,
        )
        assert allowed is not None
        try:
            snapshot = read_regular_file_snapshot(
                path,
                root=allowed,
                label=f"{label} raw inventory {row['relative_path']}",
                capture_bytes=False,
            )
        except Exception as exc:
            raise ValueError(
                f"{label} raw inventory를 안전하게 읽지 못했습니다: {row['relative_path']}: {exc}"
            ) from exc
        if snapshot.sha256 != row.get("content_sha256") or snapshot.size != row.get(
            "content_size"
        ):
            raise ValueError(
                f"{label} raw inventory SHA/size가 audit와 다릅니다: {row['relative_path']}"
            )
    return index


def validate_decoder_audit_binding(
    binding: object,
    *,
    manifest_dir: str | Path,
    repo_root: str | Path,
    raw_roots: Iterable[Path],
) -> dict[str, Any]:
    """generation sidecar가 transaction으로 복사한 audit와 정확히 결속됐는지 확인한다."""

    expected_keys = {
        "schema_version",
        "file",
        "sha256",
        "size",
        "inventory_sha256",
        "accepted_inventory_sha256",
        "decoder_fingerprint",
        "decoder_fingerprint_sha256",
    }
    if not isinstance(binding, dict) or set(binding) != expected_keys:
        raise ValueError("manifest generation decoder_audit binding 필드가 정확하지 않습니다")
    if binding.get("schema_version") != DECODER_AUDIT_SCHEMA_VERSION:
        raise ValueError("manifest generation decoder_audit schema_version이 다릅니다")
    if binding.get("file") != DECODER_AUDIT_FILE:
        raise ValueError(
            f"manifest generation decoder_audit file은 {DECODER_AUDIT_FILE!r}여야 합니다"
        )
    expected_sha = _require_sha256(binding.get("sha256"), field="decoder_audit binding")
    expected_size = binding.get("size")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
    ):
        raise ValueError("manifest generation decoder_audit size는 양수 정수여야 합니다")
    root = Path(os.path.abspath(repo_root))
    manifest_root = Path(os.path.abspath(manifest_dir))
    audit = read_decoder_audit(
        manifest_root / DECODER_AUDIT_FILE,
        repo_root=root,
        label="canonical decoder audit",
    )
    snapshot = audit["_snapshot"]
    if snapshot.sha256 != expected_sha or snapshot.size != expected_size:
        raise ValueError("manifest generation decoder_audit file SHA/size 불일치")
    for field in (
        "inventory_sha256",
        "accepted_inventory_sha256",
        "decoder_fingerprint_sha256",
    ):
        if binding.get(field) != audit.get(field):
            raise ValueError(f"manifest generation decoder_audit {field} 불일치")
    if binding.get("decoder_fingerprint") != audit.get("decoder_fingerprint"):
        raise ValueError("manifest generation decoder_audit decoder_fingerprint 불일치")
    audit["_index_by_raw_path"] = _validate_decoder_audit_raw_inventory(
        audit,
        repo_root=root,
        raw_roots=raw_roots,
        label="canonical decoder audit",
    )
    return audit


def validate_decoder_audit_manifest_entry(
    audit: dict[str, Any],
    entry: dict[str, Any],
    *,
    repo_root: str | Path,
    raw_roots: Iterable[Path],
    label: str,
) -> dict[str, Any]:
    """manifest 행이 audit에서 승인된 정확히 같은 raw file인지 fail-closed 검증한다."""

    root = Path(os.path.abspath(repo_root))
    index = audit.get("_index_by_raw_path")
    if not isinstance(index, dict):
        index = _decoder_audit_index(
            audit,
            repo_root=root,
            raw_roots=raw_roots,
            label=label,
        )
    absolute = Path(os.path.abspath(Path(str(entry.get("path") or ""))))
    row = index.get(str(absolute))
    if not isinstance(row, dict):
        try:
            relative = absolute.relative_to(root).as_posix()
        except ValueError:
            relative = str(absolute)
        raise ValueError(f"{label} raw audio가 decoder audit inventory에 없습니다: {relative}")
    if row.get("decision") != "accept":
        raise ValueError(
            f"{label} raw audio는 decoder audit에서 reject됐습니다: {row['relative_path']}"
        )
    expected_content = _require_sha256(
        entry.get("content_sha256"), field=f"{label} manifest content"
    )
    content_size = entry.get("content_size")
    if (
        isinstance(content_size, bool)
        or not isinstance(content_size, int)
        or content_size <= 0
    ):
        raise ValueError(f"{label} manifest content_size가 유효하지 않습니다")
    rejected_content = audit.get("_rejected_content_sha256")
    if not isinstance(rejected_content, set):
        rejected_content = {
            str(item["content_sha256"])
            for item in audit.get("inventory", [])
            if isinstance(item, dict) and item.get("decision") == "reject"
        }
    if expected_content in rejected_content:
        raise ValueError(
            f"{label} raw content SHA가 decoder audit reject 행과 중복됩니다: "
            f"{row['relative_path']}"
        )
    if (
        row.get("content_sha256") != expected_content
        or row.get("content_size") != content_size
    ):
        raise ValueError(
            f"{label} manifest/audit raw SHA·size가 다릅니다: {row['relative_path']}"
        )
    return row


def validate_manifest_generation(
    manifest_dir: str | Path,
    *,
    required_tags: Iterable[str],
    repo_root: str | Path = REPO_ROOT,
) -> dict:
    """sidecar와 모든 필수 JSONL을 byte 단위로 검증하고 payload를 반환한다."""

    root = Path(os.path.abspath(manifest_dir))
    configured_root = Path(os.path.abspath(repo_root))
    # 실제 저장소 안 공식 세대는 REPO_ROOT 밖을 절대 허용하지 않는다. 단위시험의
    # 완전 격리 tree는 absolute config/holdout의 공통 루트를 계약 root로 유도한다.
    try:
        root.relative_to(configured_root)
        contract_root = configured_root
    except ValueError:
        contract_root = Path(
            os.path.commonpath(
                [
                    str(root),
                    str(Path(manifest_dir).parent),
                ]
            )
        )
    sidecar = root / MANIFEST_GENERATION_FILE
    try:
        try:
            sidecar.lstat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "합성 manifest 세대 sidecar가 없어 코퍼스 무결성을 판정할 수 없습니다: "
                f"{sidecar}"
            ) from exc
        sidecar_snapshot = read_regular_file_snapshot(
            sidecar,
            root=contract_root,
            label="manifest generation sidecar",
        )
        assert sidecar_snapshot.data is not None
        sidecar_bytes = sidecar_snapshot.data
        payload = json.loads(sidecar_bytes.decode("utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"합성 manifest 세대 sidecar가 없어 코퍼스 무결성을 판정할 수 없습니다: "
            f"{sidecar}"
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"합성 manifest 세대 sidecar JSON 오류: {sidecar}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != CANONICAL_MANIFEST_SCHEMA_VERSION:
        legacy = payload.get("schema_version") if isinstance(payload, dict) else None
        if legacy == 3:
            raise ValueError(
                "schema_version=3 manifest는 decoder audit 결속 전 diagnostic-only 세대입니다; "
                "canonical 학습에는 schema_version=4가 필요합니다"
            )
        raise ValueError(
            "학습용 manifest generation schema_version은 public lineage와 decoder audit을 "
            f"결속하는 {CANONICAL_MANIFEST_SCHEMA_VERSION}이어야 합니다"
        )
    if payload.get("training_eligible") is not True:
        raise ValueError("manifest generation이 training_eligible=true가 아닙니다")

    # 격리 fixture에서만 contract 파일들의 공통 상위로 root를 넓힌다. 공식 저장소
    # 경로에서는 위 configured_root가 그대로 고정돼 sidecar가 경계를 선택할 수 없다.
    if contract_root != configured_root:
        candidates = [root]
        for field in ("data_config", "holdout"):
            value = payload.get(field)
            if isinstance(value, str) and Path(value).is_absolute():
                candidates.append(Path(value))
        contract_root = Path(os.path.commonpath([str(item) for item in candidates]))
        reject_symlink_components(root, root=contract_root)

    raw_root_values = payload.get("raw_roots")
    if not isinstance(raw_root_values, list) or not raw_root_values:
        raise ValueError("manifest generation raw_roots가 비었습니다")
    if (
        not all(isinstance(value, str) and value for value in raw_root_values)
        or raw_root_values != sorted(set(raw_root_values))
    ):
        raise ValueError("manifest generation raw_roots는 중복 없는 정렬 문자열 배열이어야 합니다")
    raw_roots: list[Path] = []
    for index, value in enumerate(raw_root_values):
        raw_root = _contract_path(value, field=f"raw_roots[{index}]", repo_root=contract_root)
        reject_symlink_components(raw_root, root=contract_root)
        if not raw_root.is_dir():
            raise ValueError(f"manifest generation raw root가 directory가 아닙니다: {raw_root}")
        raw_roots.append(raw_root)

    decoder_audit = validate_decoder_audit_binding(
        payload.get("decoder_audit"),
        manifest_dir=root,
        repo_root=contract_root,
        raw_roots=raw_roots,
    )

    manifests = payload.get("manifests")
    if (
        not isinstance(manifests, dict)
        or not manifests
        or not all(isinstance(tag, str) and tag for tag in manifests)
    ):
        raise ValueError("manifest generation manifests가 비었습니다")
    required = {str(tag) for tag in required_tags if str(tag) != "synthetic"}
    missing = sorted(required.difference(manifests))
    if missing:
        raise ValueError(f"manifest generation 필수 태그 누락: {missing}")

    validated_entries: dict[str, list[dict]] = {}
    validated_bytes: dict[str, bytes] = {}
    for tag in sorted(manifests):
        metadata = manifests[tag]
        if not isinstance(metadata, dict):
            raise ValueError(f"manifest generation {tag} metadata가 mapping이 아닙니다")
        expected_name = f"{tag}.jsonl"
        if metadata.get("file") != expected_name:
            raise ValueError(
                f"manifest generation {tag} file은 {expected_name!r}여야 합니다"
            )
        path = root / expected_name
        try:
            path.lstat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"manifest generation에 선언된 {tag} manifest 가 없는 상태입니다: {path}"
            ) from exc
        expected_sha = metadata.get("sha256")
        if not isinstance(expected_sha, str) or _SHA256_RE.fullmatch(expected_sha) is None:
            raise ValueError(f"manifest generation {tag} SHA-256가 유효하지 않습니다")
        # 이 bytes 하나로 hash, entry count, loader snapshot을 모두 만든다. hash 뒤
        # path.read_text()로 다시 열던 판은 두 read 사이 교체를 허용했다.
        manifest_snapshot = read_regular_file_snapshot(
            path,
            root=contract_root,
            label=f"manifest generation {tag} manifest",
        )
        assert manifest_snapshot.data is not None
        raw = manifest_snapshot.data
        actual_sha = manifest_snapshot.sha256
        if actual_sha != expected_sha:
            raise ValueError(
                f"manifest generation {tag} SHA 불일치: expected={expected_sha}, "
                f"actual={actual_sha}"
            )
        expected_entries = metadata.get("entries")
        if not isinstance(expected_entries, int) or expected_entries <= 0:
            raise ValueError(f"manifest generation {tag} entries는 양수 정수여야 합니다")
        entries = read_manifest_bytes(raw, manifest_path=path)
        actual_entries = len(entries)
        if actual_entries != expected_entries:
            raise ValueError(
                f"manifest generation {tag} entry 수 불일치: "
                f"expected={expected_entries}, actual={actual_entries}"
            )
        seen_paths: set[str] = set()
        for index, entry in enumerate(entries):
            if entry.get("tag") != tag:
                raise ValueError(
                    f"manifest generation {tag} entry #{index} tag가 파일 이름과 다릅니다"
                )
            audio_path = Path(str(entry.get("path") or ""))
            expected_content = entry.get("content_sha256")
            if (
                not isinstance(expected_content, str)
                or _SHA256_RE.fullmatch(expected_content) is None
            ):
                raise ValueError(
                    f"manifest generation {tag} entry #{index} content_sha256가 유효하지 않습니다"
                )
            absolute_audio = Path(os.path.abspath(audio_path))
            allowed_root = next(
                (
                    raw_root
                    for raw_root in raw_roots
                    if absolute_audio == raw_root or raw_root in absolute_audio.parents
                ),
                None,
            )
            if allowed_root is None:
                raise ValueError(
                    f"manifest generation {tag} raw path가 declared root 밖입니다: {audio_path}"
                )
            key = str(absolute_audio)
            if key in seen_paths:
                raise ValueError(f"manifest generation {tag}에 중복 raw path가 있습니다: {audio_path}")
            seen_paths.add(key)
            audio_snapshot = read_regular_file_snapshot(
                absolute_audio,
                root=allowed_root,
                label=f"manifest generation {tag} raw audio #{index}",
                capture_bytes=False,
            )
            if audio_snapshot.sha256 != expected_content:
                raise ValueError(
                    f"manifest generation {tag} raw content SHA 불일치: {audio_path}; "
                    f"expected={expected_content}, actual={audio_snapshot.sha256}"
                )
            content_size = entry.get("content_size")
            if content_size is not None and content_size != audio_snapshot.size:
                raise ValueError(
                    f"manifest generation {tag} raw content size 불일치: {audio_path}"
                )
            if content_size is None:
                raise ValueError(
                    f"manifest generation {tag} raw content_size 증거가 없습니다: {audio_path}"
                )
            validate_decoder_audit_manifest_entry(
                decoder_audit,
                entry,
                repo_root=contract_root,
                raw_roots=raw_roots,
                label=f"manifest generation {tag} raw audio #{index}",
            )
            entry["_validated_file_snapshot"] = audio_snapshot.stat_contract()
            entry["_validated_raw_root"] = str(allowed_root)
        validated_entries[tag] = entries
        validated_bytes[tag] = raw

    holdout_payload: dict | None = None
    for field in ("data_config", "holdout"):
        path = _contract_path(payload.get(field), field=field, repo_root=contract_root)
        expected_sha = payload.get(f"{field}_sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(f"manifest generation {field}_sha256가 유효하지 않습니다")
        contract_snapshot = read_regular_file_snapshot(
            path,
            root=contract_root,
            label=f"manifest generation {field}",
        )
        actual_sha = contract_snapshot.sha256
        if actual_sha != expected_sha:
            raise ValueError(
                f"manifest generation {field} SHA 불일치: expected={expected_sha}, "
                f"actual={actual_sha}"
            )
        if field == "holdout":
            assert contract_snapshot.data is not None
            try:
                decoded = json.loads(contract_snapshot.data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"manifest generation holdout JSON 오류: {exc}") from exc
            if not isinstance(decoded, dict):
                raise ValueError("manifest generation holdout는 JSON object여야 합니다")
            holdout_payload = decoded

    lineage = payload.get("public_lineage")
    if not isinstance(lineage, dict) or lineage.get("schema_version") != 1:
        raise ValueError("manifest generation public_lineage schema_version=1 증거가 없습니다")
    if lineage.get("lineage_schema") != PUBLIC_LINEAGE_SCHEMA:
        raise ValueError("manifest generation public_lineage schema가 canonical 값이 아닙니다")
    metadata = lineage.get("metadata")
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("manifest generation public_lineage metadata가 비었습니다")
    for name, evidence in sorted(metadata.items()):
        if (
            not isinstance(name, str)
            or not isinstance(evidence, dict)
            or set(evidence) != {"path", "sha256", "size"}
            or not isinstance(evidence.get("path"), str)
            or _SHA256_RE.fullmatch(str(evidence.get("sha256") or "")) is None
            or not isinstance(evidence.get("size"), int)
            or evidence.get("size") <= 0
        ):
            raise ValueError(f"public_lineage metadata.{name} 증거가 유효하지 않습니다")
        metadata_path = _contract_path(
            evidence["path"], field=f"public_lineage.metadata.{name}.path", repo_root=contract_root
        )
        metadata_snapshot = read_regular_file_snapshot(
            metadata_path,
            root=contract_root,
            label=f"public lineage metadata {name}",
            capture_bytes=False,
        )
        if (
            metadata_snapshot.sha256 != evidence["sha256"]
            or metadata_snapshot.size != evidence["size"]
        ):
            raise ValueError(f"public lineage metadata.{name} path/SHA/size 불일치")

    components = lineage.get("components")
    if not isinstance(components, dict):
        raise ValueError("public_lineage components가 mapping이 아닙니다")
    if lineage.get("component_count") != len(components):
        raise ValueError("public_lineage component_count 불일치")
    if lineage.get("component_membership_sha256") != canonical_json_sha256(
        {key: components[key] for key in sorted(components)}
    ):
        raise ValueError("public_lineage 전체 component membership digest 불일치")

    manifest_lineage = validate_public_manifest_lineage(validated_entries)
    if (
        lineage.get("manifest_component_count")
        != manifest_lineage["component_count"]
        or lineage.get("manifest_component_membership_sha256")
        != manifest_lineage["component_membership_sha256"]
    ):
        raise ValueError("public_lineage manifest component 증거 불일치")
    if holdout_payload is None:
        raise ValueError("public_lineage 검증에 필요한 holdout snapshot이 없습니다")
    holdout_lineage = holdout_payload.get("clip_lineage")
    holdout_families = holdout_payload.get("families")
    if not isinstance(holdout_families, dict) or not holdout_families:
        raise ValueError("manifest generation holdout families가 비었습니다")
    try:
        holdout_rows = validate_recorded_clip_lineage(
            holdout_lineage if isinstance(holdout_lineage, dict) else {},
            families=holdout_families,
        )
    except ValueError as exc:
        raise ValueError(f"manifest generation holdout clip_lineage 오류: {exc}") from exc
    if lineage.get("holdout_clips_sha256") != holdout_lineage.get("clips_sha256"):
        raise ValueError("public_lineage와 holdout clip_lineage digest가 다릅니다")
    holdout_names = {row["clip"] for row in holdout_rows}
    holdout_content = {row["content_sha256"] for row in holdout_rows}
    holdout_keys = {key for row in holdout_rows for key in row["lineage_keys"]}
    leaked: list[str] = []
    for tag, entries in sorted(validated_entries.items()):
        for entry in entries:
            basename = Path(str(entry["path"])).name.casefold()
            if (
                basename in holdout_names
                or entry["content_sha256"] in holdout_content
                or set(entry["lineage_keys"]).intersection(holdout_keys)
            ):
                leaked.append(f"{tag}:{basename}")
    if leaked:
        raise ValueError(
            "recorded holdout와 synthetic manifest의 basename/content/lineage 누수: "
            f"{leaked[:8]}"
        )

    build_id = payload.get("build_id")
    if not isinstance(build_id, str) or _SHA256_RE.fullmatch(build_id) is None:
        raise ValueError("manifest generation build_id가 canonical SHA-256가 아닙니다")
    basis = {key: value for key, value in payload.items() if key not in {"build_id", "created_at"}}
    expected_build_id = hashlib.sha256(_canonical_json_bytes(basis)).hexdigest()
    if build_id != expected_build_id:
        raise ValueError(
            f"manifest generation build_id 불일치: expected={expected_build_id}, "
            f"actual={build_id}"
        )
    snapshot = dict(payload)
    snapshot["_validated_entries"] = validated_entries
    snapshot["_validated_manifest_bytes"] = validated_bytes
    snapshot["_validated_sidecar_bytes"] = sidecar_bytes
    snapshot["_validated_decoder_audit"] = decoder_audit
    # JSONL 행을 변형하지 않고, canonical v4 audit을 통과한 generation이라는 사실을
    # dataset/NoisePool 경계에 전달한다. NoisePool은 이 marker가 있을 때 재시도 대신
    # 비정상 decode를 hard fail로 취급해야 한다.
    snapshot["_canonical_decoder_audited"] = True
    return snapshot


__all__ = [
    "CANONICAL_MANIFEST_SCHEMA_VERSION",
    "DECODER_AUDIT_FILE",
    "DECODER_AUDIT_SCHEMA_VERSION",
    "MANIFEST_GENERATION_FILE",
    "decoder_audit_binding",
    "read_decoder_audit",
    "sha256_file",
    "validate_decoder_audit_binding",
    "validate_decoder_audit_manifest_entry",
    "validate_manifest_generation",
]
