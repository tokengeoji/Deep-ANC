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
from pathlib import Path
from typing import Iterable

from ..config import REPO_ROOT
from .holdout_contract import read_regular_file_snapshot, reject_symlink_components
from .manifest import read_manifest_bytes
from .public_lineage import (
    PUBLIC_LINEAGE_SCHEMA,
    canonical_json_sha256,
    validate_public_manifest_lineage,
    validate_recorded_clip_lineage,
)


MANIFEST_GENERATION_FILE = "manifest_generation.json"
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
    if not isinstance(payload, dict) or payload.get("schema_version") != 3:
        raise ValueError(
            "학습용 manifest generation schema_version은 public lineage까지 결속하는 3이어야 합니다"
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
    return snapshot


__all__ = ["MANIFEST_GENERATION_FILE", "sha256_file", "validate_manifest_generation"]
