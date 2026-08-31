"""Stage-2 public manifest와 recorded holdout 계보 교집합의 byte 감사.

legacy public JSONL은 lineage 필드를 갖지 않으므로 그 상태를 그대로 PASS시키지
않는다. 대신 holdout에 결속된 권위 metadata bytes에서 각 row의 semantic key를
다시 유도하고, basename/content/semantic key의 transitive component를 계산한다.
이 보고서는 exclusion 대상과 재생성 필요성을 증명할 뿐 canonical manifest를
자동 생성하거나 승격하지 않는다. 네트워크, 오디오 장치와 GPU를 사용하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .holdout_contract import validate_holdout_contract
from .public_lineage import (
    PUBLIC_LINEAGE_SCHEMA,
    esc50_lineage_keys,
    fma_lineage_keys,
    librispeech_lineage_keys,
    parse_esc50_metadata_bytes,
    parse_fma_tracks_bytes,
    parse_librispeech_chapters_bytes,
    validate_recorded_clip_lineage,
)


SCHEMA = "stage2_public_recorded_lineage_overlap_audit_v1"
SPLITS = ("train", "val", "test")
_HEX = frozenset("0123456789abcdef")


class Stage2PublicRecordedLineageError(ValueError):
    """입력 bytes/schema가 fail-closed 감사를 허용하지 않는다."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(path: Path, *, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise Stage2PublicRecordedLineageError(
            f"repository 밖 파일은 감사할 수 없습니다: {resolved}"
        ) from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise Stage2PublicRecordedLineageError(
            f"regular non-symlink file이 아닙니다: {resolved}"
        )
    return {
        "path": relative.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise Stage2PublicRecordedLineageError(f"JSON duplicate key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage2PublicRecordedLineageError(f"JSON을 읽을 수 없습니다: {path}") from exc
    if not isinstance(value, dict):
        raise Stage2PublicRecordedLineageError(f"JSON root가 object가 아닙니다: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise Stage2PublicRecordedLineageError(f"JSONL을 읽을 수 없습니다: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_pairs)
        except json.JSONDecodeError as exc:
            raise Stage2PublicRecordedLineageError(
                f"JSONL 오류: {path}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise Stage2PublicRecordedLineageError(
                f"JSONL row가 object가 아닙니다: {path}:{line_number}"
            )
        rows.append(row)
    if not rows:
        raise Stage2PublicRecordedLineageError(f"JSONL이 비었습니다: {path}")
    return rows


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def find(self, value: str) -> str:
        if value not in self.parent:
            self.parent[value] = value
            self.size[value] = 1
            return value
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        current = value
        while current != root:
            parent = self.parent[current]
            self.parent[current] = root
            current = parent
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.size[a] < self.size[b] or (
            self.size[a] == self.size[b] and a > b
        ):
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def _identity_nodes(row: Mapping[str, Any]) -> tuple[str, ...]:
    basename = str(row.get("basename") or "").casefold()
    content = str(row.get("content_sha256") or "")
    keys = row.get("lineage_keys")
    if (
        not basename
        or Path(basename).name != basename
        or not isinstance(keys, Sequence)
        or isinstance(keys, (str, bytes))
        or not keys
        or any(not isinstance(value, str) or not value for value in keys)
        or (content and (len(content) != 64 or any(value not in _HEX for value in content)))
    ):
        raise Stage2PublicRecordedLineageError("lineage row identity가 불완전합니다")
    nodes = [f"basename:{basename}"]
    if content:
        nodes.append(f"content_sha256:{content}")
    nodes.extend(f"semantic:{value}" for value in sorted(set(keys)))
    return tuple(nodes)


def audit_lineage_rows(
    *,
    recorded_rows: Sequence[Mapping[str, Any]],
    public_rows_by_manifest: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """이미 metadata에서 유도한 identity row의 exact/transitive 교집합을 센다."""

    if not recorded_rows or not public_rows_by_manifest:
        raise Stage2PublicRecordedLineageError("recorded/public lineage rows가 비었습니다")
    dsu = _DisjointSet()
    recorded_nodes: set[str] = set()
    recorded_basenames: set[str] = set()
    for row in recorded_rows:
        nodes = _identity_nodes(row)
        recorded_nodes.update(nodes)
        recorded_basenames.add(str(row["basename"]).casefold())
        for node in nodes[1:]:
            dsu.union(nodes[0], node)

    normalized_public: dict[str, list[dict[str, Any]]] = {}
    for manifest, rows in sorted(public_rows_by_manifest.items()):
        if not manifest or not rows:
            raise Stage2PublicRecordedLineageError("public manifest identity/rows가 비었습니다")
        normalized: list[dict[str, Any]] = []
        seen_basenames: set[str] = set()
        for source_index, row in enumerate(rows):
            split = str(row.get("split") or "")
            if split not in SPLITS:
                raise Stage2PublicRecordedLineageError(
                    f"public split이 canonical 값이 아닙니다: {manifest}:{source_index}"
                )
            nodes = _identity_nodes(row)
            basename = str(row["basename"]).casefold()
            if basename in seen_basenames:
                raise Stage2PublicRecordedLineageError(
                    f"public manifest basename이 중복됩니다: {manifest}:{basename}"
                )
            seen_basenames.add(basename)
            for node in nodes[1:]:
                dsu.union(nodes[0], node)
            normalized.append(
                {
                    "split": split,
                    "basename": basename,
                    "nodes": nodes,
                    "exact_basename": basename in recorded_basenames,
                    "direct_identity": bool(set(nodes) & recorded_nodes),
                }
            )
        normalized_public[manifest] = normalized

    recorded_roots = {dsu.find(node) for node in recorded_nodes}
    inventories: list[dict[str, Any]] = []
    all_overlap = 0
    for manifest, rows in normalized_public.items():
        component_splits: dict[str, set[str]] = defaultdict(set)
        exact: Counter[str] = Counter()
        direct: Counter[str] = Counter()
        connected: Counter[str] = Counter()
        for row in rows:
            root = dsu.find(row["nodes"][0])
            component_splits[root].add(row["split"])
            exact[row["split"]] += int(row["exact_basename"])
            direct[row["split"]] += int(row["direct_identity"])
            connected[row["split"]] += int(root in recorded_roots)
        crossing = {root for root, splits in component_splits.items() if len(splits) > 1}
        rows_in_crossing = sum(
            dsu.find(row["nodes"][0]) in crossing for row in rows
        )
        overlap_count = sum(connected.values())
        all_overlap += overlap_count
        inventories.append(
            {
                "manifest": manifest,
                "rows": len(rows),
                "exact_recorded_basename_rows_by_split": {
                    split: exact[split] for split in SPLITS
                },
                "direct_recorded_identity_rows_by_split": {
                    split: direct[split] for split in SPLITS
                },
                "transitive_recorded_component_rows_by_split": {
                    split: connected[split] for split in SPLITS
                },
                "exact_recorded_basename_rows": sum(exact.values()),
                "direct_recorded_identity_rows": sum(direct.values()),
                "transitive_recorded_component_rows": overlap_count,
                "semantic_components": len(component_splits),
                "components_crossing_public_splits": len(crossing),
                "public_rows_in_cross_split_components": rows_in_crossing,
                "status": "BLOCKED" if overlap_count or crossing else "PASS",
            }
        )
    return {
        "status": "BLOCKED" if all_overlap else "PASS",
        "inventories": inventories,
        "total_transitive_recorded_component_rows": all_overlap,
        "component_semantics": (
            "basename, available content SHA, and metadata-derived lineage keys are "
            "unioned transitively; a connected component is indivisible"
        ),
    }


def _derive_public_rows(
    *,
    path: Path,
    parser: Callable[[str], Sequence[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in _read_jsonl(path):
        basename = Path(str(source.get("path") or "")).name.casefold()
        rows.append(
            {
                "split": source.get("split"),
                "basename": basename,
                "content_sha256": str(source.get("content_sha256") or ""),
                "lineage_keys": list(parser(basename)),
            }
        )
    return rows


def audit_stage2_public_recorded_lineage(
    *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(os.path.abspath(repository_root)).resolve(strict=True)
    holdout_path = root / "data/manifests/recorded_holdout.json"
    holdout_payload = _read_json(holdout_path)
    holdout_summary = validate_holdout_contract(holdout_path, repo_root=root)
    if int(holdout_summary["active_session_count"]) != 82:
        raise Stage2PublicRecordedLineageError("canonical holdout은 정확히 82 sessions여야 합니다")
    recorded = validate_recorded_clip_lineage(
        holdout_payload.get("clip_lineage", {}),
        families=holdout_payload.get("families"),
    )
    metadata = holdout_payload["clip_lineage"]["metadata"]
    metadata_payloads: dict[str, bytes] = {}
    metadata_evidence: dict[str, dict[str, Any]] = {}
    for name, authority in sorted(metadata.items()):
        path = root / str(authority["path"])
        snapshot = _snapshot(path, root=root)
        if (
            snapshot["sha256"] != authority["sha256"]
            or snapshot["size_bytes"] != authority["size"]
        ):
            raise Stage2PublicRecordedLineageError(
                f"holdout authority와 현재 metadata bytes가 다릅니다: {name}"
            )
        metadata_payloads[name] = path.read_bytes()
        metadata_evidence[name] = snapshot

    chapters = parse_librispeech_chapters_bytes(
        metadata_payloads["librispeech_chapters"]
    )
    tracks = parse_fma_tracks_bytes(metadata_payloads["fma_tracks"])
    esc50 = parse_esc50_metadata_bytes(metadata_payloads["esc50"])
    manifest_specs = {
        "data/manifests/speech.jsonl": lambda value: librispeech_lineage_keys(
            value, chapters
        ),
        "data/manifests/music.jsonl": lambda value: fma_lineage_keys(value, tracks),
        "data/manifests/esc50.jsonl": lambda value: esc50_lineage_keys(value, esc50),
    }
    public: dict[str, list[dict[str, Any]]] = {}
    manifest_evidence: dict[str, dict[str, Any]] = {}
    lineage_fields_complete: dict[str, bool] = {}
    for relative, parser in manifest_specs.items():
        path = root / relative
        source_rows = _read_jsonl(path)
        required = {
            "content_sha256",
            "content_size",
            "lineage_schema",
            "lineage_keys",
            "group_id",
        }
        lineage_fields_complete[relative] = all(
            required.issubset(row) and row.get("lineage_schema") == PUBLIC_LINEAGE_SCHEMA
            for row in source_rows
        )
        public[relative] = _derive_public_rows(path=path, parser=parser)
        manifest_evidence[relative] = _snapshot(path, root=root)

    recorded_rows = [
        {
            "basename": row["clip"],
            "content_sha256": row["content_sha256"],
            "lineage_keys": row["lineage_keys"],
        }
        for row in recorded
    ]
    overlap = audit_lineage_rows(
        recorded_rows=recorded_rows,
        public_rows_by_manifest=public,
    )
    blockers = [
        "PUBLIC_RECORDED_TRANSITIVE_COMPONENT_OVERLAP",
        "PUBLIC_MANIFESTS_REQUIRE_REGENERATION_AFTER_COMPONENT_EXCLUSION",
    ]
    if not all(lineage_fields_complete.values()):
        blockers.append("LEGACY_PUBLIC_MANIFEST_LINEAGE_FIELDS_INCOMPLETE")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "BLOCKED",
        "canonical_stage2_public_manifest_eligible": False,
        "legacy_automatic_promotion": False,
        "inputs": {
            "recorded_holdout": _snapshot(holdout_path, root=root),
            "recorded_holdout_active_sessions": 82,
            "recorded_clip_rows": len(recorded_rows),
            "metadata": metadata_evidence,
            "public_manifests": manifest_evidence,
        },
        "legacy_public_lineage_fields_complete": lineage_fields_complete,
        "overlap": overlap,
        "blockers": sorted(blockers),
        "required_action": (
            "raw public corpus에서 recorded component 전체를 제외한 뒤 canonical public "
            "manifest를 재생성하고, source-density 및 decoder audit를 다시 수행한다"
        ),
    }
    payload["evidence_sha256"] = _digest(payload)
    return payload


def write_json_no_replace(payload: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"기존 audit를 덮어쓰지 않습니다: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_canonical_json(dict(payload)) + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = audit_stage2_public_recorded_lineage(
        repository_root=args.repo_root
    )
    write_json_no_replace(payload, args.output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": args.output,
                "evidence_sha256": payload["evidence_sha256"],
                "overlap": payload["overlap"][
                    "total_transitive_recorded_component_rows"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
