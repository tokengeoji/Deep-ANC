"""82세션 parent를 보존하며 추가 녹음을 canonical generation으로 승격한다.

이 계약은 기존 ``data/recorded``를 확장하지 않는다. 기존 82세션은 holdout과
recorded-tree 증거에 계속 결속되고, 추가 세션은 별도 source plan과 별도 root에
수집된다. 두 집합을 합친 manifest는 세대별 새 경로에만 발행한다.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import wave
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from deep_anc.data.holdout_contract import (
    FileSnapshot,
    HoldoutContractError,
    read_regular_file_snapshot,
    reject_symlink_components,
    snapshot_regular_tree_metadata,
    validate_holdout_contract,
)
from deep_anc.data.manifest import (
    VALID_SPLITS,
    read_manifest_bytes,
    validate_group_id,
    validate_source_family,
)
from deep_anc.data.timeline import TIMELINE_METHOD
from deep_anc.dsp.invariants import REQUIRED_SOURCE_FAMILIES
from deep_anc.data import public_lineage
from deep_anc.data.recorded_dns_selection import (
    DNS_COMPOSITE_SECONDS,
    DNS_REPEAT_COUNT,
    DNS_SELECTION_RECEIPT,
    DNS_SOURCE_KIND,
    DNS_TRANSFORM,
    DNSSelectionError,
    validate_dns_selection_receipt,
)


RECORDED_GENERATION_SCHEMA_VERSION = 1
PARENT_SESSION_COUNT = 82
ADDITION_SESSION_COUNT = 17
COMBINED_SESSION_COUNT = PARENT_SESSION_COUNT + ADDITION_SESSION_COUNT
EXPECTED_ADDITION_FAMILY_COUNTS = {
    "speech": 5,
    "music": 4,
    "environment": 4,
    "machine": 4,
}
EXPECTED_ADDITION_FAMILY_SPLIT_COUNTS = {
    ("speech", "train"): 2,
    ("speech", "val"): 1,
    ("speech", "test"): 2,
    ("music", "train"): 0,
    ("music", "val"): 2,
    ("music", "test"): 2,
    ("environment", "train"): 0,
    ("environment", "val"): 1,
    ("environment", "test"): 3,
    ("machine", "train"): 1,
    ("machine", "val"): 1,
    ("machine", "test"): 2,
}
EXPECTED_ADDITION_FAMILY_KIND_COUNTS = {
    ("speech", "external_dns_speech_composite"): 5,
    ("music", "source_pool_row"): 4,
    ("environment", "source_pool_row"): 4,
    ("machine", "external_exact_composite"): 4,
}
PARENT_ROOT = "data/recorded"
PARENT_MANIFEST = "data/manifests/recorded_regrouped.jsonl"
PARENT_HOLDOUT = "data/manifests/recorded_holdout.json"
GENERATION_ROOT = "data/manifests/recorded_generations"
ADDITIONS_ROOT = "data/recorded_additions"
SOURCE_PLAN_ROOT = "data/source_plans/recorded_additions"
SOURCE_PLAN_FIELDS = (
    "source_kind",
    "path",
    "seconds",
    "start_seconds",
    "source_family",
    "group_id",
    "lineage_key",
    "split",
    "source_file_sha256",
    "raw_member_path",
    "raw_member_sha256",
    "raw_member_lineage_key",
    "authority_metadata_sha256",
    "inventory_path",
    "inventory_sha256",
    "transform",
    "transform_repeat_count",
)
SOURCE_KIND_POOL = "source_pool_row"
SOURCE_KIND_EXTERNAL = "external_exact_composite"
SOURCE_KIND_EXTERNAL_LIBRISPEECH = "external_librispeech_file"
SOURCE_KIND_EXTERNAL_DNS_SPEECH = DNS_SOURCE_KIND
EXTERNAL_TRANSFORM = "mono_polyphase_kaiser5_resample_48000_pcm16_repeat/v1"
EXTERNAL_REPEAT_COUNT = 3
EXTERNAL_RAW_SECONDS = 5.0
EXTERNAL_OUTPUT_SECONDS = EXTERNAL_RAW_SECONDS * EXTERNAL_REPEAT_COUNT
EXTERNAL_LIBRISPEECH_TRANSFORM = "identity_window/v1"
EXTERNAL_LIBRISPEECH_SECONDS = 15.0
CANONICAL_ADDITION_SECONDS = 15.0
CANONICAL_ADDITION_SECONDS_BY_KIND = {
    SOURCE_KIND_POOL: CANONICAL_ADDITION_SECONDS,
    SOURCE_KIND_EXTERNAL: CANONICAL_ADDITION_SECONDS,
    SOURCE_KIND_EXTERNAL_LIBRISPEECH: CANONICAL_ADDITION_SECONDS,
    SOURCE_KIND_EXTERNAL_DNS_SPEECH: CANONICAL_ADDITION_SECONDS,
}
CANONICAL_SOURCE_POOL_ADDITIONS = {
    "data/source_pool/environment/environment_008.wav": ("environment", 54.1, "test"),
    "data/source_pool_v2/environment/environment_012.wav": ("environment", 3.0, "test"),
    "data/source_pool_v2/environment/environment_004.wav": ("environment", 5.9, "test"),
    "data/source_pool_v2/environment/environment_017.wav": ("environment", 26.2, "val"),
    "data/source_pool/music/music_007.wav": ("music", 54.8, "test"),
    "data/source_pool_v2/music/music_007.wav": ("music", 12.8, "test"),
    "data/source_pool_v2/music/music_012.wav": ("music", 17.1, "val"),
    "data/source_pool_v2/music/music_017.wav": ("music", 20.1, "val"),
}
# 이 다섯 source-pool speech는 PSD 진단에서 고역 후보였지만 full CHAPTERS
# authority closure에서 parent82와 겹친다. canonical plan에는 넣지 않고 receipt의
# rejected evidence로만 유지한다.
REJECTED_SOURCE_POOL_SPEECH_ADDITIONS = {
    "data/source_pool/speech/speech_002.wav": ("speech", 51.0, "test"),
    "data/source_pool/speech/speech_013.wav": ("speech", 13.75, "test"),
    "data/source_pool_v2/speech/speech_002.wav": ("speech", 0.75, "train"),
    "data/source_pool_v2/speech/speech_016.wav": ("speech", 51.75, "val"),
    "data/source_pool_v2/speech/speech_019.wav": ("speech", 50.0, "train"),
}
# 이 네 파일은 파일명상의 reader/book 직접 비교에서는 free처럼 보였지만,
# 권위 CHAPTERS.TXT 전체 reader<->book 전이 DSU에서 parent82 active component와
# 모두 연결된다. exact source plan에는 절대 포함하지 않으며 탈락 증거로만 보존한다.
CANONICAL_EXTERNAL_LIBRISPEECH_FILES = {
    "data/raw/speech/LibriSpeech/dev-clean/2035/152373/2035-152373-0013.flac": (
        3.0,
        "train",
    ),
    "data/raw/speech/LibriSpeech/dev-clean/1272/128104/1272-128104-0004.flac": (
        0.75,
        "train",
    ),
    "data/raw/speech/LibriSpeech/dev-clean/6241/61943/6241-61943-0027.flac": (
        0.5,
        "test",
    ),
    "data/raw/speech/LibriSpeech/dev-clean/2412/153948/2412-153948-0006.flac": (
        0.25,
        "test",
    ),
}
REJECTED_EXTERNAL_LIBRISPEECH_COMPONENT = "speech-librispeech-lineage-d697786cc484"

# 후보 선택 감사 증거. source bytes SHA는 source plan CSV가 별도로 exact 결속하고,
# 여기서는 선택 알고리즘/strict P/결정론적 window 결과와 탈락 이유를 봉인한다.
SOURCE_SELECTION_CONTRACT_SCHEMA = "recorded_highband_source_selection/v1"
CANONICAL_SOURCE_PLAN_BLOCKER = (
    "BLOCKED: external DNS speech 5개의 no-replace selection receipt와 exact raw/"
    "composite 검증이 완료되지 않아 source plan을 발행할 수 없습니다"
)
SOURCE_SELECTION_STRICT_PRIMARY_PATH = (
    "assets/measured/primary_path_il_strict_5dc06fdd.npz"
)
SOURCE_SELECTION_STRICT_PRIMARY_SHA256 = (
    "23fa43f1ec46d5bca6bdad53938b81bb2d2c85afc4eee35e83c555b6c4f0c598"
)
CANONICAL_SPEECH_SELECTION_EVIDENCE = {
    "data/source_pool/speech/speech_002.wav": {
        "start_seconds": 51.0,
        "split": "test",
        "component": "speech-source-lineage-5e6747e2fe8e",
        "covered_segment_counts": [9, 9, 7, 1],
        "max_density_ratios": [6.020, 2.395, 0.805, 0.2613584148552767],
    },
    "data/source_pool/speech/speech_013.wav": {
        "start_seconds": 13.75,
        "split": "test",
        "component": "speech-source-lineage-12018c875756",
        "covered_segment_counts": [9, 9, 2, 1],
        "max_density_ratios": [6.869, 3.550, 0.501, 0.26340568165926725],
    },
    "data/source_pool_v2/speech/speech_002.wav": {
        "start_seconds": 0.75,
        "split": "train",
        "component": "speech-source-lineage-40228101fb7e",
        "covered_segment_counts": [9, 9, 1, 1],
        "max_density_ratios": [7.175, 2.580, 0.379, 0.2807645547193211],
    },
    "data/source_pool_v2/speech/speech_016.wav": {
        "start_seconds": 51.75,
        "split": "val",
        "component": "speech-source-lineage-f0a7305ec56a",
        "covered_segment_counts": [9, 9, 9, 5],
        "max_density_ratios": [1.280, 3.916, 1.105, 0.4121062719781847],
    },
    "data/source_pool_v2/speech/speech_019.wav": {
        "start_seconds": 50.0,
        "split": "train",
        "component": "speech-source-lineage-4d567a68408e",
        "covered_segment_counts": [9, 9, 2, 1],
        "max_density_ratios": [6.702, 4.321, 0.451, 0.28715328085255387],
    },
}
CANONICAL_EXTERNAL_ESC_MACHINE_FILES = {
    "1-28808-A-43.wav": ("28808", "car_horn"),
    "5-235507-A-44.wav": ("235507", "engine"),
    "4-102871-A-42.wav": ("102871", "siren"),
    "5-222524-A-41.wav": ("222524", "chainsaw"),
}
CANONICAL_EXTERNAL_ESC_SPLITS = {
    "1-28808-A-43.wav": "train",
    "5-235507-A-44.wav": "val",
    "4-102871-A-42.wav": "test",
    "5-222524-A-41.wav": "test",
}
CANONICAL_EXTERNAL_ESC_OUTPUT_NAMES = {
    "1-28808-A-43.wav": "machine-28808-repeat3.wav",
    "5-235507-A-44.wav": "machine-235507-repeat3.wav",
    "4-102871-A-42.wav": "machine-102871-repeat3.wav",
    "5-222524-A-41.wav": "machine-222524-repeat3.wav",
}
ADDITION_LINEAGE_SCHEMA = "exact_collection_plan/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GENERATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


class RecordedGenerationError(ValueError):
    """Recorded generation 또는 수집 증거가 계약을 위반했다."""


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecordedGenerationError(f"JSON 중복 키 금지: {key}")
        result[key] = value
    return result


def _canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _generation_id(value: object) -> str:
    if not isinstance(value, str) or _GENERATION_ID_RE.fullmatch(value) is None:
        raise RecordedGenerationError(
            "generation_id는 3~64자의 소문자 영숫자/하이픈이어야 합니다"
        )
    return value


def validate_generation_id(value: object) -> str:
    """CLI와 builder가 공유하는 recorded generation 식별자 검증."""

    return _generation_id(value)


def _relative(value: object, *, field: str, prefix: str | None = None) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RecordedGenerationError(f"{field}는 비어 있지 않은 문자열이어야 합니다")
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise RecordedGenerationError(f"{field}는 canonical 저장소 상대 POSIX 경로여야 합니다")
    result = path.as_posix()
    if prefix is not None and not result.startswith(prefix.rstrip("/") + "/"):
        raise RecordedGenerationError(f"{field}는 {prefix}/ 아래여야 합니다: {result}")
    return result


def _snapshot(
    repo_root: Path, relative: str, *, label: str, capture_bytes: bool = False
) -> FileSnapshot:
    try:
        return read_regular_file_snapshot(
            repo_root / relative,
            root=repo_root,
            label=label,
            capture_bytes=capture_bytes,
        )
    except HoldoutContractError as exc:
        raise RecordedGenerationError(str(exc)) from exc


def _file_ref(snapshot: FileSnapshot, *, repo_root: Path) -> dict[str, object]:
    return {
        "path": snapshot.path.relative_to(repo_root).as_posix(),
        "sha256": snapshot.sha256,
        "size": snapshot.size,
    }


def _tree_summary(tree: Any) -> dict[str, object]:
    return {
        "file_count": tree.file_count,
        "total_bytes": sum(int(item[1]) for item in tree.content_entries),
        "metadata_snapshot_sha256": tree.sha256,
        "content_snapshot_sha256": tree.content_sha256,
    }


def _manifest_entries(snapshot: FileSnapshot) -> list[dict[str, Any]]:
    assert snapshot.data is not None
    try:
        text = snapshot.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecordedGenerationError(f"manifest UTF-8 오류: {snapshot.path}") from exc
    raw_entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_pairs_without_duplicates)
        except json.JSONDecodeError as exc:
            raise RecordedGenerationError(
                f"manifest JSON 오류: {snapshot.path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise RecordedGenerationError(
                f"manifest row는 object여야 합니다: {snapshot.path}:{line_number}"
            )
        raw_entries.append(value)
    try:
        read_manifest_bytes(snapshot.data, manifest_path=snapshot.path)
    except (ValueError, json.JSONDecodeError) as exc:
        raise RecordedGenerationError(str(exc)) from exc
    return raw_entries


def _source_row_identity_keys(
    *,
    family: str,
    clips: list[str],
    fma_tracks: dict[int, tuple[str, str]],
    librispeech_chapters: dict[int, tuple[int, int]],
    esc50_metadata: dict[str, str],
) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for clip in clips:
        item: list[tuple[str, str]] = [("clip", str(clip).casefold())]
        try:
            if family == "music":
                item.extend(
                    ("music", key)
                    for key in public_lineage.fma_lineage_keys(clip, fma_tracks)
                )
            elif family == "speech":
                item.extend(
                    ("speech", key)
                    for key in public_lineage.librispeech_lineage_keys(
                        clip, librispeech_chapters
                    )
                )
            elif family in {"environment", "machine"}:
                item.extend(
                    ("esc50", key)
                    for key in public_lineage.esc50_lineage_keys(clip, esc50_metadata)
                )
        except ValueError as exc:
            raise RecordedGenerationError(
                f"source-pool metadata lineage를 유도할 수 없습니다: {clip}: {exc}"
            ) from exc
        keys.extend(item)
    return keys


def _derive_source_component_map(
    rows: dict[str, dict[str, Any]],
    *,
    fma_tracks: dict[int, tuple[str, str]],
    librispeech_chapters: dict[int, tuple[int, int]],
    esc50_metadata: dict[str, str],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """source-pool 160행의 metadata identity transitive component를 재유도한다."""

    parent = {path: path for path in rows}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    owner: dict[tuple[str, str], str] = {}
    for path, row in sorted(rows.items()):
        family = str(row["source_family"])
        keys = _source_row_identity_keys(
            family=family,
            clips=[str(value) for value in row["clips"]],
            fma_tracks=fma_tracks,
            librispeech_chapters=librispeech_chapters,
            esc50_metadata=esc50_metadata,
        )
        for key in keys:
            previous = owner.setdefault(key, path)
            union(previous, path)
    raw_components: dict[str, list[str]] = defaultdict(list)
    for path in sorted(rows):
        raw_components[find(path)].append(path)
    components: dict[str, list[str]] = {}
    by_path: dict[str, str] = {}
    for members in sorted(raw_components.values()):
        families = {str(rows[path]["source_family"]) for path in members}
        if len(families) != 1:
            raise RecordedGenerationError(
                f"source-pool lineage component가 family를 가로지릅니다: {members[:5]}"
            )
        family = next(iter(families))
        component = (
            f"{family}-source-lineage-"
            + hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()[:12]
        )
        components[component] = members
        for path in members:
            by_path[path] = component
    return by_path, components


def _derive_librispeech_identity_component_map(
    chapters: dict[int, tuple[int, int]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """CHAPTERS 전체 reader/book 그래프의 transitive component를 재유도한다.

    한 파일의 reader와 book 두 키만 active 집합과 직접 비교하면, 같은 reader가 읽은
    다른 책이나 같은 책을 읽은 다른 reader를 경유하는 누수를 놓친다. 따라서 권위
    CHAPTERS 전체를 bipartite graph로 union한 뒤 component 단위로만 신규 소스를
    승인한다.
    """

    parent: dict[str, str] = {}

    def add(value: str) -> None:
        parent.setdefault(value, value)

    def find(value: str) -> str:
        root = value
        while parent[root] != root:
            root = parent[root]
        while value != root:
            previous = parent[value]
            parent[value] = root
            value = previous
        return root

    def union(left: str, right: str) -> None:
        add(left)
        add(right)
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for chapter, pair in sorted(chapters.items()):
        if (
            isinstance(chapter, bool)
            or not isinstance(chapter, int)
            or chapter < 0
            or not isinstance(pair, tuple)
            or len(pair) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in pair)
        ):
            raise RecordedGenerationError("LibriSpeech CHAPTERS parsed mapping이 유효하지 않습니다")
        reader, book = pair
        union(f"librivox_reader:{reader}", f"gutenberg_book:{book}")
    if not parent:
        raise RecordedGenerationError("LibriSpeech CHAPTERS component mapping이 비었습니다")

    raw_components: dict[str, list[str]] = defaultdict(list)
    for key in sorted(parent):
        raw_components[find(key)].append(key)
    components: dict[str, list[str]] = {}
    by_identity: dict[str, str] = {}
    for members in sorted(raw_components.values()):
        component = (
            "speech-librispeech-lineage-"
            + hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()[:12]
        )
        components[component] = members
        for key in members:
            by_identity[key] = component
    return by_identity, components


def _parse_esc50_authority_bytes(raw: bytes) -> dict[str, tuple[str, str]]:
    """ESC-50 inventory에서 filename별 src_file/category를 exact 보존한다."""

    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
        rows = list(reader)
    except UnicodeDecodeError as exc:
        raise RecordedGenerationError("ESC-50 metadata UTF-8 오류") from exc
    required = {"filename", "src_file", "category"}
    if not required.issubset(reader.fieldnames or ()):
        raise RecordedGenerationError("ESC-50 metadata filename/src_file/category 열 누락")
    result: dict[str, tuple[str, str]] = {}
    for number, row in enumerate(rows, start=2):
        filename = str(row.get("filename") or "")
        source = str(row.get("src_file") or "")
        category = str(row.get("category") or "")
        if (
            not filename
            or Path(filename).name != filename
            or not source
            or not category
            or filename in result
        ):
            raise RecordedGenerationError(
                f"ESC-50 metadata inventory row가 유효하지 않습니다: {number}"
            )
        result[filename] = (source, category)
    if not result:
        raise RecordedGenerationError("ESC-50 metadata inventory가 비었습니다")
    return result


def _canonical_source_lineage(repo_root: Path) -> dict[str, Any]:
    """검증된 holdout/metadata에서 160 source-row component와 active 교집합을 만든다."""

    try:
        holdout = validate_holdout_contract(
            repo_root / PARENT_HOLDOUT,
            repo_root=repo_root,
        )
    except (OSError, HoldoutContractError) as exc:
        raise RecordedGenerationError(f"source lineage holdout 검증 실패: {exc}") from exc
    source_paths = holdout.get("sources_csv")
    source_hashes = holdout.get("sources_csv_sha256")
    lineage = holdout.get("lineage")
    holdout_bytes = holdout.get("_validated_holdout_bytes")
    if (
        source_paths != ["data/source_pool/sources.csv", "data/source_pool_v2/sources.csv"]
        or not isinstance(source_hashes, dict)
        or not isinstance(lineage, dict)
        or not isinstance(holdout_bytes, bytes)
    ):
        raise RecordedGenerationError("holdout source CSV/lineage raw anchor가 불완전합니다")
    try:
        raw_holdout = json.loads(
            holdout_bytes.decode("utf-8"), object_pairs_hook=_pairs_without_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordedGenerationError("holdout raw source_rows를 읽을 수 없습니다") from exc
    active_rows = raw_holdout.get("source_rows") if isinstance(raw_holdout, dict) else None
    if (
        not isinstance(active_rows, list)
        or len(active_rows) != PARENT_SESSION_COUNT
        or not all(isinstance(value, str) for value in active_rows)
    ):
        raise RecordedGenerationError("holdout active source_rows가 exact 82가 아닙니다")

    rows: dict[str, dict[str, Any]] = {}
    source_evidence: list[dict[str, Any]] = []
    for index, relative in enumerate(source_paths):
        snapshot = _snapshot(
            repo_root, relative, label=f"source-pool CSV generation {index}", capture_bytes=True
        )
        expected = source_hashes.get("source_pool" if index == 0 else "source_pool_v2")
        if snapshot.sha256 != expected:
            raise RecordedGenerationError(f"source-pool CSV SHA가 holdout과 다릅니다: {relative}")
        assert snapshot.data is not None
        try:
            reader = csv.DictReader(io.StringIO(snapshot.data.decode("utf-8"), newline=""))
            csv_rows = list(reader)
        except UnicodeDecodeError as exc:
            raise RecordedGenerationError(f"source-pool CSV UTF-8 오류: {relative}") from exc
        if len(csv_rows) != 80:
            raise RecordedGenerationError(f"source-pool CSV는 세대별 정확히 80행이어야 합니다: {relative}")
        required = {"source_family", "group_id", "path", "clips"}
        if not required.issubset(reader.fieldnames or ()):
            raise RecordedGenerationError(f"source-pool CSV 필수 열 누락: {relative}")
        for number, raw in enumerate(csv_rows, start=2):
            path = _relative(raw.get("path"), field=f"{relative}:{number}.path")
            if path in rows:
                raise RecordedGenerationError(f"source-pool row path 중복: {path}")
            try:
                family = validate_source_family(raw.get("source_family"))
                group_id = validate_group_id(raw.get("group_id"))
                clips = json.loads(raw.get("clips") or "")
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RecordedGenerationError(f"source-pool row 오류: {relative}:{number}: {exc}") from exc
            if (
                family not in REQUIRED_SOURCE_FAMILIES
                or not isinstance(clips, list)
                or not clips
                or not all(isinstance(value, str) and value for value in clips)
            ):
                raise RecordedGenerationError(f"source-pool row family/clips 오류: {relative}:{number}")
            rows[path] = {
                "source_family": family,
                "group_id": group_id,
                "clips": clips,
            }
        source_evidence.append(_file_ref(snapshot, repo_root=repo_root))
    if len(rows) != 160 or not set(active_rows).issubset(rows):
        raise RecordedGenerationError("source-pool union 160행 또는 active 82 mapping이 불완전합니다")

    metadata_specs = (
        (
            public_lineage.FMA_TRACKS,
            lineage.get("tracks_csv_sha256"),
            public_lineage.parse_fma_tracks_bytes,
        ),
        (
            str(lineage.get("librispeech_chapters_path")),
            lineage.get("librispeech_chapters_sha256"),
            public_lineage.parse_librispeech_chapters_bytes,
        ),
        (
            str(lineage.get("esc50_metadata_path")),
            lineage.get("esc50_metadata_sha256"),
            public_lineage.parse_esc50_metadata_bytes,
        ),
    )
    parsed_metadata: list[Any] = []
    metadata_raw: list[bytes] = []
    metadata_evidence: list[dict[str, Any]] = []
    for relative, expected_sha, parser in metadata_specs:
        if _SHA256_RE.fullmatch(str(expected_sha)) is None:
            raise RecordedGenerationError("holdout lineage metadata SHA가 유효하지 않습니다")
        snapshot = _snapshot(repo_root, relative, label=f"source lineage metadata {relative}", capture_bytes=True)
        if snapshot.sha256 != expected_sha:
            raise RecordedGenerationError(f"source lineage metadata SHA 불일치: {relative}")
        assert snapshot.data is not None
        try:
            parsed_metadata.append(parser(snapshot.data))
        except ValueError as exc:
            raise RecordedGenerationError(f"source lineage metadata 파싱 실패: {relative}: {exc}") from exc
        metadata_raw.append(snapshot.data)
        metadata_evidence.append(_file_ref(snapshot, repo_root=repo_root))
    esc50_authority = _parse_esc50_authority_bytes(metadata_raw[2])
    by_path, components = _derive_source_component_map(
        rows,
        fma_tracks=parsed_metadata[0],
        librispeech_chapters=parsed_metadata[1],
        esc50_metadata=parsed_metadata[2],
    )
    libri_component_by_identity, libri_components = (
        _derive_librispeech_identity_component_map(parsed_metadata[1])
    )
    active_components = {by_path[path] for path in active_rows}
    active_identity_keys: set[tuple[str, str]] = set()
    for path in active_rows:
        row = rows[path]
        active_identity_keys.update(
            _source_row_identity_keys(
                family=str(row["source_family"]),
                clips=[str(value) for value in row["clips"]],
                fma_tracks=parsed_metadata[0],
                librispeech_chapters=parsed_metadata[1],
                esc50_metadata=parsed_metadata[2],
            )
        )
    active_librispeech_components = {
        libri_component_by_identity[key]
        for namespace, key in active_identity_keys
        if namespace == "speech" and key in libri_component_by_identity
    }
    authority_tokens_by_path: dict[str, set[str]] = {}
    for path, row in rows.items():
        tokens = {f"source_pool_component:{by_path[path]}"}
        identity_keys = _source_row_identity_keys(
            family=str(row["source_family"]),
            clips=[str(value) for value in row["clips"]],
            fma_tracks=parsed_metadata[0],
            librispeech_chapters=parsed_metadata[1],
            esc50_metadata=parsed_metadata[2],
        )
        for namespace, key in identity_keys:
            if namespace == "speech":
                component = libri_component_by_identity.get(key)
                if component is None:
                    raise RecordedGenerationError(
                        f"source-pool speech identity가 CHAPTERS component에 없습니다: {path}: {key}"
                    )
                tokens.add(f"librispeech_component:{component}")
            elif namespace == "esc50":
                tokens.add(f"esc50_identity:{key}")
            elif namespace == "music":
                tokens.add(f"fma_identity:{key}")
            elif namespace == "clip":
                tokens.add(f"clip_identity:{key}")
        authority_tokens_by_path[path] = tokens
    evidence = {
        "source_csv": source_evidence,
        "metadata": metadata_evidence,
        "components": components,
        "librispeech_components": libri_components,
        "active_source_rows": sorted(active_rows),
        "active_components": sorted(active_components),
        "active_librispeech_components": sorted(active_librispeech_components),
        "active_identity_keys": [list(value) for value in sorted(active_identity_keys)],
        "authority_tokens_by_path": {
            path: sorted(tokens)
            for path, tokens in sorted(authority_tokens_by_path.items())
        },
    }
    return {
        "rows": rows,
        "component_by_path": by_path,
        "active_components": active_components,
        "active_identity_keys": active_identity_keys,
        "librispeech_chapters": parsed_metadata[1],
        "librispeech_component_by_identity": libri_component_by_identity,
        "active_librispeech_components": active_librispeech_components,
        "authority_tokens_by_path": authority_tokens_by_path,
        "librispeech_chapters_sha256": metadata_evidence[1]["sha256"],
        "librispeech_chapters_path": metadata_evidence[1]["path"],
        "esc50_metadata": parsed_metadata[2],
        "esc50_metadata_sha256": metadata_evidence[2]["sha256"],
        "esc50_metadata_path": metadata_evidence[2]["path"],
        "esc50_authority": esc50_authority,
        "evidence_sha256": _canonical_json_sha256(evidence),
    }


def _canonical_source_selection_evidence(
    repo_root: Path, source_lineage: dict[str, Any]
) -> dict[str, Any]:
    """현행 17행 선택의 물리/계보/외부 DNS receipt를 exact 결속한다."""

    primary = _snapshot(
        repo_root,
        SOURCE_SELECTION_STRICT_PRIMARY_PATH,
        label="recorded high-band source selection strict primary",
    )
    if primary.sha256 != SOURCE_SELECTION_STRICT_PRIMARY_SHA256:
        raise RecordedGenerationError(
            "high-band source selection strict P SHA가 현행 권위와 다릅니다"
        )
    selected: list[dict[str, Any]] = []
    used_authority_tokens: dict[str, str] = {}
    component_by_path = source_lineage["component_by_path"]
    active_components = source_lineage["active_components"]
    for path, (family, start, split) in sorted(
        CANONICAL_SOURCE_POOL_ADDITIONS.items()
    ):
        component = component_by_path.get(path)
        if not isinstance(component, str) or component in active_components:
            raise RecordedGenerationError(
                f"high-band selected source가 authority에서 free가 아닙니다: {path}"
            )
        row: dict[str, Any] = {
            "path": path,
            "source_family": family,
            "start_seconds": start,
            "seconds": CANONICAL_ADDITION_SECONDS,
            "split": split,
            "authority_component": component,
        }
        authority_tokens = set(
            source_lineage["authority_tokens_by_path"].get(path, set())
        )
        overlap = sorted(authority_tokens & set(used_authority_tokens))
        if overlap:
            first = used_authority_tokens[overlap[0]]
            raise RecordedGenerationError(
                f"{CANONICAL_SOURCE_PLAN_BLOCKER}; overlap={overlap[0]}, "
                f"paths={first},{path}"
            )
        for token in authority_tokens:
            used_authority_tokens[token] = path
        selected.append(row)

    rejected_pool_speech: list[dict[str, Any]] = []
    for path, (_family, start, split) in sorted(
        REJECTED_SOURCE_POOL_SPEECH_ADDITIONS.items()
    ):
        component = component_by_path.get(path)
        expected = CANONICAL_SPEECH_SELECTION_EVIDENCE.get(path)
        tokens = sorted(source_lineage["authority_tokens_by_path"].get(path, set()))
        libri_components = sorted(
            token.removeprefix("librispeech_component:")
            for token in tokens
            if token.startswith("librispeech_component:")
        )
        if (
            not isinstance(component, str)
            or not isinstance(expected, dict)
            or expected.get("start_seconds") != start
            or expected.get("split") != split
            or expected.get("component") != component
            or len(libri_components) != 1
        ):
            raise RecordedGenerationError(
                f"rejected source-pool speech 진단/authority가 예상과 다릅니다: {path}"
            )
        rejected_pool_speech.append(
            {
                "path": path,
                "previous_start_seconds": start,
                "previous_split": split,
                "source_pool_component": component,
                "librispeech_component": libri_components[0],
                "overlaps_parent82": libri_components[0]
                in source_lineage["active_librispeech_components"],
                "diagnostic_scan_only": {
                    "covered_segment_counts": expected["covered_segment_counts"],
                    "max_density_ratios": expected["max_density_ratios"],
                },
                "reason": (
                    "source_pool_speech_candidate_set_not_five_independent_"
                    "full_authority_components"
                ),
            }
        )

    rejected: list[dict[str, Any]] = []
    for path, (start, split) in sorted(
        CANONICAL_EXTERNAL_LIBRISPEECH_FILES.items()
    ):
        identities = public_lineage.librispeech_lineage_keys(
            Path(path).name, source_lineage["librispeech_chapters"]
        )
        components = {
            source_lineage["librispeech_component_by_identity"].get(key)
            for key in identities
        }
        if components != {REJECTED_EXTERNAL_LIBRISPEECH_COMPONENT} or not (
            components & source_lineage["active_librispeech_components"]
        ):
            raise RecordedGenerationError(
                f"rejected LibriSpeech authority component가 예상과 다릅니다: {path}"
            )
        rejected.append(
            {
                "path": path,
                "previous_start_seconds": start,
                "previous_split": split,
                "authority_component": REJECTED_EXTERNAL_LIBRISPEECH_COMPONENT,
                "reason": "full_CHAPTERS_transitive_component_overlaps_parent82",
            }
        )

    try:
        dns_selection = validate_dns_selection_receipt(
            repo_root=repo_root,
            receipt_path=DNS_SELECTION_RECEIPT,
            require_source_files=True,
        )
    except DNSSelectionError as exc:
        raise RecordedGenerationError(f"{CANONICAL_SOURCE_PLAN_BLOCKER}: {exc}") from exc
    if (
        dns_selection["strict_primary_path"]
        != SOURCE_SELECTION_STRICT_PRIMARY_PATH
        or dns_selection["strict_primary_sha256"] != primary.sha256
    ):
        raise RecordedGenerationError(
            "external DNS selection receipt가 현행 strict P path/SHA와 다릅니다"
        )
    dns_bundle_files = [
        {
            "path": dns_selection["receipt_path"],
            "sha256": dns_selection["receipt_sha256"],
            "size": dns_selection["receipt_size"],
        },
        {
            "path": dns_selection["public_manifest_path"],
            "sha256": dns_selection["public_manifest_sha256"],
            "size": dns_selection["public_manifest_size"],
        },
        {
            "path": dns_selection["bootstrap_receipt_path"],
            "sha256": dns_selection["bootstrap_receipt_sha256"],
            "size": dns_selection["bootstrap_receipt_size"],
        },
        {
            "path": dns_selection["environment_freeze_path"],
            "sha256": dns_selection["environment_freeze_sha256"],
            "size": dns_selection["environment_freeze_size"],
        },
    ]
    for item in dns_selection["selected"]:
        for field in ("raw_output", "composite_output"):
            ref = item[field]
            dns_bundle_files.append(
                {
                    "path": ref["path"],
                    "sha256": ref["sha256"],
                    "size": ref["size"],
                }
            )
    dns_bundle_files.sort(key=lambda item: str(item["path"]))
    if len(dns_bundle_files) != 14 or len(
        {str(item["path"]) for item in dns_bundle_files}
    ) != 14:
        raise RecordedGenerationError(
            "external DNS selection bundle은 receipt+bootstrap+freeze+manifest+raw5+"
            "composite5 exact 14개여야 합니다"
        )

    payload = {
        "schema": SOURCE_SELECTION_CONTRACT_SCHEMA,
        "generation_id": "highband-coverage-v1",
        "strict_primary": _file_ref(primary, repo_root=repo_root),
        "algorithm": {
            "source_population": "full_source_pool_160_authority_DSU",
            "segment_seconds": 1.5,
            "edge_trim_seconds": 0.25,
            "segment_start_rule": "start+0.25+1.5*k_for_k=0..8",
            "strict_subbands_hz": [
                [150.0, 300.0],
                [300.0, 600.0],
                [600.0, 1000.0],
                [1000.0, 1600.0],
            ],
            "minimum_density_ratio": 0.25,
            "lineage_rule": "metadata_transitive_component_disjoint_from_parent82_and_each_other",
        },
        "selected_source_pool": selected,
        "rejected_source_pool_speech": rejected_pool_speech,
        "rejected_external_librispeech": rejected,
        "external_dns_speech_selection": {
            "receipt_path": dns_selection["receipt_path"],
            "receipt_sha256": dns_selection["receipt_sha256"],
            "evidence_sha256": dns_selection["evidence_sha256"],
            "public_manifest_sha256": dns_selection["public_manifest_sha256"],
            "strict_primary_sha256": dns_selection["strict_primary_sha256"],
            "selected_group_ids": dns_selection["selected_group_ids"],
            "bundle_files": dns_bundle_files,
        },
        "external_machine_policy": "exact_ESC50_raw_identity_and_repeat3_composite",
    }
    payload["evidence_sha256"] = _canonical_json_sha256(payload)
    return payload


def _resolved_manifest_session(entry: dict[str, Any], manifest_path: Path) -> Path:
    path = entry.get("path")
    if not isinstance(path, str):
        raise RecordedGenerationError("recorded manifest path가 문자열이 아닙니다")
    if entry.get("path_base") != "manifest":
        raise RecordedGenerationError("recorded generation은 path_base='manifest'만 허용합니다")
    return Path(os.path.abspath(manifest_path.parent / path))


def _canonical_external_composite_bytes(
    raw_path: Path, *, raw_bytes: bytes | None = None
) -> bytes:
    """5초 mono raw를 48kHz로 고정 resample한 뒤 3회 반복한 canonical PCM16 WAV."""

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    try:
        values, sample_rate = sf.read(
            io.BytesIO(raw_bytes) if raw_bytes is not None else str(raw_path),
            dtype="float64",
            always_2d=True,
        )
    except (OSError, RuntimeError) as exc:
        raise RecordedGenerationError(f"external raw member decode 실패: {raw_path}: {exc}") from exc
    if values.shape[1] != 1 or sample_rate <= 0 or not bool(np.isfinite(values).all()):
        raise RecordedGenerationError("external raw member는 유한 mono audio여야 합니다")
    duration = values.shape[0] / float(sample_rate)
    if not math.isclose(duration, EXTERNAL_RAW_SECONDS, rel_tol=0.0, abs_tol=1e-9):
        raise RecordedGenerationError(
            f"external raw member는 정확히 {EXTERNAL_RAW_SECONDS:.1f}초여야 합니다: {duration}"
        )
    common = math.gcd(int(sample_rate), 48_000)
    resampled = resample_poly(
        values[:, 0],
        48_000 // common,
        int(sample_rate) // common,
        window=("kaiser", 5.0),
        padtype="constant",
    )
    expected_frames = int(EXTERNAL_RAW_SECONDS * 48_000)
    if resampled.shape != (expected_frames,):
        raise RecordedGenerationError(
            f"external resample frame 수 불일치: {resampled.shape} != {(expected_frames,)}"
        )
    repeated = np.tile(resampled, EXTERNAL_REPEAT_COUNT)
    # WAV bytes 자체까지 재현 가능하도록 libsndfile writer 대신 stdlib wave와
    # 명시적 signed PCM16 양자화를 쓴다.
    quantized = np.rint(np.clip(repeated, -1.0, 1.0) * 32767.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.setnframes(int(quantized.size))
        handle.writeframes(quantized.tobytes(order="C"))
    return output.getvalue()


def _validate_addition_population(rows: list[dict[str, Any]]) -> None:
    observed_family_counts = {
        family: sum(row["source_family"] == family for row in rows)
        for family in EXPECTED_ADDITION_FAMILY_COUNTS
    }
    if observed_family_counts != EXPECTED_ADDITION_FAMILY_COUNTS:
        raise RecordedGenerationError(
            "recorded additions family 구성은 speech 5 + music/environment/machine 각 4로 "
            f"고정됩니다: actual={observed_family_counts}"
        )
    observed_family_split_counts = {
        key: sum(
            row["source_family"] == key[0] and row["split"] == key[1]
            for row in rows
        )
        for key in EXPECTED_ADDITION_FAMILY_SPLIT_COUNTS
    }
    if observed_family_split_counts != EXPECTED_ADDITION_FAMILY_SPLIT_COUNTS:
        raise RecordedGenerationError(
            "recorded additions family×split 구성은 coverage 결손을 복구하는 exact matrix로 "
            f"고정됩니다: actual={observed_family_split_counts}"
        )
    observed_family_kind_counts = {
        key: sum(
            row["source_family"] == key[0] and row["source_kind"] == key[1]
            for row in rows
        )
        for key in EXPECTED_ADDITION_FAMILY_KIND_COUNTS
    }
    observed_nonzero_family_kind_counts = {
        (str(row["source_family"]), str(row["source_kind"]))
        for row in rows
    }
    if (
        observed_family_kind_counts != EXPECTED_ADDITION_FAMILY_KIND_COUNTS
        or observed_nonzero_family_kind_counts
        != set(EXPECTED_ADDITION_FAMILY_KIND_COUNTS)
    ):
        raise RecordedGenerationError(
            "recorded additions source_kind 구성은 source-pool 8 + external DNS speech5 "
            "+ external ESC4로 "
            f"고정됩니다: actual={observed_family_kind_counts}"
        )


def _read_source_plan(
    *, repo_root: Path, relative: str, require_source_files: bool
) -> tuple[FileSnapshot, list[dict[str, Any]], str, dict[str, Any]]:
    snapshot = _snapshot(repo_root, relative, label="recorded additions source plan", capture_bytes=True)
    assert snapshot.data is not None
    try:
        handle = io.StringIO(snapshot.data.decode("utf-8"), newline="")
    except UnicodeDecodeError as exc:
        raise RecordedGenerationError("recorded additions source plan UTF-8 오류") from exc
    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_PLAN_FIELDS:
            raise RecordedGenerationError(
                "recorded additions source plan header는 exact 계약이어야 합니다: "
                f"expected={SOURCE_PLAN_FIELDS}, actual={tuple(reader.fieldnames or ())}"
            )
        raw_rows = list(reader)
    if len(raw_rows) != ADDITION_SESSION_COUNT:
        raise RecordedGenerationError(
            f"recorded additions source plan은 정확히 {ADDITION_SESSION_COUNT}행이어야 합니다: "
            f"{len(raw_rows)}"
        )

    source_lineage = _canonical_source_lineage(repo_root)
    selection_evidence = _canonical_source_selection_evidence(
        repo_root, source_lineage
    )
    dns_selection: dict[str, Any] | None = None
    dns_items_by_composite: dict[str, dict[str, Any]] = {}
    if any(
        str(row.get("source_kind")) == SOURCE_KIND_EXTERNAL_DNS_SPEECH
        for row in raw_rows
    ):
        try:
            dns_selection = validate_dns_selection_receipt(
                repo_root=repo_root,
                receipt_path=DNS_SELECTION_RECEIPT,
                require_source_files=True,
            )
        except DNSSelectionError as exc:
            raise RecordedGenerationError(f"{CANONICAL_SOURCE_PLAN_BLOCKER}: {exc}") from exc
        dns_items_by_composite = {
            str(item["composite_output"]["path"]): item
            for item in dns_selection["selected"]
        }
    canonical_rows = source_lineage["rows"]
    component_by_path = source_lineage["component_by_path"]
    active_components = source_lineage["active_components"]
    rows: list[dict[str, Any]] = []
    lineages: set[str] = set()
    source_hashes: set[str] = set()
    used_authority_tokens: set[str] = set()
    generation_id = Path(relative).stem
    for offset, raw in enumerate(raw_rows, start=2):
        try:
            source_kind = str(raw["source_kind"])
            path = _relative(raw["path"], field=f"source plan row {offset}.path")
            seconds = float(raw["seconds"])
            start = float(raw["start_seconds"])
            family = validate_source_family(raw["source_family"])
            group = validate_group_id(raw["group_id"])
            lineage = validate_group_id(raw["lineage_key"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RecordedGenerationError(f"source plan row {offset} 오류: {exc}") from exc
        if family not in REQUIRED_SOURCE_FAMILIES:
            raise RecordedGenerationError(
                f"source plan row {offset} source_family이 canonical 네 family 밖입니다: {family}"
            )
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise RecordedGenerationError(f"source plan row {offset} seconds는 양수 finite여야 합니다")
        expected_seconds = CANONICAL_ADDITION_SECONDS_BY_KIND.get(source_kind)
        if expected_seconds is not None and not math.isclose(
            seconds, expected_seconds, rel_tol=0.0, abs_tol=1e-9
        ):
            raise RecordedGenerationError(
                f"source plan row {offset} seconds는 canonical "
                f"{expected_seconds:.1f}초여야 합니다"
            )
        if not math.isfinite(start) or start < 0.0:
            raise RecordedGenerationError(
                f"source plan row {offset} start_seconds는 0 이상 finite여야 합니다"
            )
        split = raw["split"]
        if split not in VALID_SPLITS:
            raise RecordedGenerationError(
                f"source plan row {offset} split은 {VALID_SPLITS} 중 하나여야 합니다"
            )
        digest = raw["source_file_sha256"]
        if _SHA256_RE.fullmatch(digest or "") is None:
            raise RecordedGenerationError(
                f"source plan row {offset} source_file_sha256가 유효하지 않습니다"
            )
        raw_member_path = (raw.get("raw_member_path") or "").strip()
        raw_member_sha256 = (raw.get("raw_member_sha256") or "").strip()
        raw_member_lineage_key = (raw.get("raw_member_lineage_key") or "").strip()
        authority_metadata_sha256 = (raw.get("authority_metadata_sha256") or "").strip()
        inventory_path = (raw.get("inventory_path") or "").strip()
        inventory_sha256 = (raw.get("inventory_sha256") or "").strip()
        transform = (raw.get("transform") or "").strip()
        repeat_text = (raw.get("transform_repeat_count") or "").strip()
        source_snapshot: FileSnapshot | None = None
        row_authority_tokens: set[str]
        if source_kind == SOURCE_KIND_POOL:
            canonical_row = canonical_rows.get(path)
            planned_pool = CANONICAL_SOURCE_POOL_ADDITIONS.get(path)
            if not isinstance(canonical_row, dict):
                raise RecordedGenerationError(
                    f"source plan row {offset}가 canonical source-pool 160행 밖입니다: {path}"
                )
            derived_lineage = component_by_path.get(path)
            if planned_pool != (family, start, split):
                raise RecordedGenerationError(
                    f"source plan row {offset}는 exact high-band source-pool plan과 다릅니다: "
                    f"path={path}, family/start/split={(family, start, split)}"
                )
            if (
                canonical_row.get("source_family") != family
                or canonical_row.get("group_id") != group
                or lineage != derived_lineage
            ):
                raise RecordedGenerationError(
                    f"source plan row {offset} family/group/lineage가 권위 metadata DSU와 다릅니다: "
                    f"declared=({family},{group},{lineage}), "
                    f"derived=({canonical_row.get('source_family')},{canonical_row.get('group_id')},"
                    f"{derived_lineage})"
                )
            if derived_lineage in active_components:
                raise RecordedGenerationError(
                    f"source plan row {offset}는 parent 82 active component와 겹칩니다: "
                    f"{path} -> {derived_lineage}"
                )
            row_authority_tokens = set(
                source_lineage["authority_tokens_by_path"].get(path, set())
            )
            if not row_authority_tokens:
                raise RecordedGenerationError(
                    f"source_pool_row {offset} authority component가 없습니다"
                )
            if (
                raw_member_path
                or raw_member_sha256
                or raw_member_lineage_key
                or authority_metadata_sha256
                or inventory_path
                or inventory_sha256
                or transform != "identity"
                or repeat_text != "1"
            ):
                raise RecordedGenerationError(
                    f"source_pool_row {offset}의 raw-member/authority/transform 필드는 "
                    "empty/identity/1이어야 합니다"
                )
        elif source_kind == SOURCE_KIND_EXTERNAL:
            if family != "machine":
                raise RecordedGenerationError(
                    "external_exact_composite는 exact machine 보충 4개만 허용합니다"
                )
            raw_member_path = _relative(
                raw_member_path,
                field=f"source plan row {offset}.raw_member_path",
                prefix="data/raw/noise/esc50/ESC-50-master/audio",
            )
            if _SHA256_RE.fullmatch(raw_member_sha256) is None:
                raise RecordedGenerationError(
                    f"source plan row {offset} raw_member_sha256가 유효하지 않습니다"
                )
            if authority_metadata_sha256 != source_lineage["esc50_metadata_sha256"]:
                raise RecordedGenerationError(
                    f"external row {offset} ESC-50 metadata SHA가 holdout authority와 다릅니다"
                )
            if (
                inventory_path != source_lineage["esc50_metadata_path"]
                or inventory_sha256 != source_lineage["esc50_metadata_sha256"]
            ):
                raise RecordedGenerationError(
                    f"external row {offset} ESC-50 inventory path/SHA가 authority와 다릅니다"
                )
            raw_filename = Path(raw_member_path).name
            expected_raw_path = (
                "data/raw/noise/esc50/ESC-50-master/audio/" + raw_filename
            )
            expected_inventory = CANONICAL_EXTERNAL_ESC_MACHINE_FILES.get(raw_filename)
            observed_inventory = source_lineage["esc50_authority"].get(raw_filename)
            if (
                raw_member_path != expected_raw_path
                or expected_inventory is None
                or observed_inventory != expected_inventory
                or split != CANONICAL_EXTERNAL_ESC_SPLITS.get(raw_filename)
            ):
                raise RecordedGenerationError(
                    f"external row {offset}는 canonical ESC-50 machine 4개 flat inventory 밖입니다: "
                    f"path={raw_member_path}, metadata={observed_inventory}"
                )
            try:
                derived_raw_lineage = public_lineage.esc50_lineage_keys(
                    Path(raw_member_path).name, source_lineage["esc50_metadata"]
                )[0]
            except ValueError as exc:
                raise RecordedGenerationError(
                    f"source plan row {offset} ESC-50 raw lineage 재유도 실패: {exc}"
                ) from exc
            derived_group = (
                f"{family}-esc50-source-"
                + hashlib.sha256(derived_raw_lineage.encode("utf-8")).hexdigest()[:12]
            )
            derived_lineage = (
                f"{family}-external-lineage-"
                + hashlib.sha256(derived_raw_lineage.encode("utf-8")).hexdigest()[:12]
            )
            row_authority_tokens = {
                f"esc50_identity:{derived_raw_lineage}",
                f"clip_identity:{Path(raw_member_path).name.casefold()}",
            }
            if (
                raw_member_lineage_key != derived_raw_lineage
                or ("esc50", derived_raw_lineage) in source_lineage["active_identity_keys"]
                or group != derived_group
                or lineage != derived_lineage
            ):
                raise RecordedGenerationError(
                    f"external row {offset} raw/group/lineage가 ESC metadata 또는 active82 disjoint "
                    "계약과 다릅니다"
                )
            expected_prefix = (
                f"{SOURCE_PLAN_ROOT}/{generation_id}_sources/"
            )
            expected_output_path = (
                expected_prefix + CANONICAL_EXTERNAL_ESC_OUTPUT_NAMES[raw_filename]
            )
            if path != expected_output_path:
                raise RecordedGenerationError(
                    "external composite output path가 exact inventory와 다릅니다: "
                    f"expected={expected_output_path}, actual={path}"
                )
            if (
                transform != EXTERNAL_TRANSFORM
                or repeat_text != str(EXTERNAL_REPEAT_COUNT)
                or not math.isclose(seconds, EXTERNAL_OUTPUT_SECONDS, rel_tol=0.0, abs_tol=1e-9)
                or not math.isclose(start, 0.0, rel_tol=0.0, abs_tol=1e-9)
            ):
                raise RecordedGenerationError(
                    f"external row {offset} transform/seconds/start 계약이 다릅니다"
                )
            if require_source_files:
                raw_snapshot = _snapshot(
                    repo_root,
                    raw_member_path,
                    label=f"external raw member row {offset}",
                    capture_bytes=True,
                )
                if raw_snapshot.sha256 != raw_member_sha256:
                    raise RecordedGenerationError(
                        f"external row {offset} raw member SHA 불일치"
                    )
                expected_output = _canonical_external_composite_bytes(
                    repo_root / raw_member_path,
                    raw_bytes=raw_snapshot.data,
                )
                if _sha256_bytes(expected_output) != digest:
                    raise RecordedGenerationError(
                        f"external row {offset} canonical transform output SHA 불일치"
                    )
        elif source_kind == SOURCE_KIND_EXTERNAL_DNS_SPEECH:
            assert dns_selection is not None
            if family != "speech":
                raise RecordedGenerationError(
                    "external_dns_speech_composite는 speech family만 허용합니다"
                )
            item = dns_items_by_composite.get(path)
            if not isinstance(item, dict):
                raise RecordedGenerationError(
                    f"DNS speech row {offset}가 validated receipt selected 5개 밖입니다: {path}"
                )
            raw_ref = item["raw_output"]
            composite_ref = item["composite_output"]
            derived_group = str(item["public_group_id"])
            digest12 = hashlib.sha256(derived_group.encode("utf-8")).hexdigest()[:12]
            derived_lineage = f"speech-dns-lineage-{digest12}"
            row_authority_tokens = {
                f"public_lineage_key:{key}" for key in item["lineage_keys"]
            }
            # set comprehension과 고정 token을 합쳐 receipt source/raw bytes도
            # generation exclusion authority로 전달한다.
            row_authority_tokens.update(
                {
                    f"public_group:{derived_group}",
                    f"content_sha256:{item['source_content_sha256']}",
                    f"raw_content_sha256:{raw_ref['sha256']}",
                }
            )
            if (
                group != derived_group
                or lineage != derived_lineage
                or split != item["recorded_split"]
                or not math.isclose(start, 0.0, rel_tol=0.0, abs_tol=1e-9)
                or not math.isclose(
                    seconds, DNS_COMPOSITE_SECONDS, rel_tol=0.0, abs_tol=1e-9
                )
                or digest != composite_ref["sha256"]
                or raw_member_path != raw_ref["path"]
                or raw_member_sha256 != raw_ref["sha256"]
                or raw_member_lineage_key != derived_group
                or authority_metadata_sha256
                != dns_selection["public_manifest_sha256"]
                or inventory_path != dns_selection["receipt_path"]
                or inventory_sha256 != dns_selection["receipt_sha256"]
                or transform != DNS_TRANSFORM
                or repeat_text != str(DNS_REPEAT_COUNT)
            ):
                raise RecordedGenerationError(
                    f"DNS speech row {offset}가 receipt/group/split/SHA/transform과 다릅니다"
                )
            if require_source_files:
                source_snapshot = _snapshot(
                    repo_root,
                    path,
                    label=f"DNS speech composite row {offset}",
                    capture_bytes=True,
                )
                if source_snapshot.sha256 != digest:
                    raise RecordedGenerationError(
                        f"DNS speech row {offset} composite SHA 불일치"
                    )
        elif source_kind == SOURCE_KIND_EXTERNAL_LIBRISPEECH:
            if family != "speech":
                raise RecordedGenerationError(
                    "external_librispeech_file은 speech family만 허용합니다"
                )
            path = _relative(
                path,
                field=f"source plan row {offset}.path",
                prefix="data/raw/speech/LibriSpeech",
            )
            raw_member_path = _relative(
                raw_member_path,
                field=f"source plan row {offset}.raw_member_path",
                prefix="data/raw/speech/LibriSpeech",
            )
            if path != raw_member_path or Path(path).suffix.casefold() != ".flac":
                raise RecordedGenerationError(
                    f"external LibriSpeech row {offset}는 동일한 untouched FLAC path를 "
                    "path/raw_member_path에 선언해야 합니다"
                )
            relative_libri = PurePosixPath(path).relative_to(
                PurePosixPath("data/raw/speech/LibriSpeech")
            )
            planned_libri = CANONICAL_EXTERNAL_LIBRISPEECH_FILES.get(path)
            if planned_libri != (start, split):
                raise RecordedGenerationError(
                    f"external LibriSpeech row {offset}는 exact high-band file/start/split "
                    f"plan 밖입니다: path={path}, start={start}, split={split}"
                )
            if len(relative_libri.parts) != 4 or relative_libri.parts[0] != "dev-clean":
                raise RecordedGenerationError(
                    f"external LibriSpeech row {offset}는 canonical dev-clean "
                    "subset/speaker/chapter/file 경로여야 합니다"
                )
            subset, speaker_dir, chapter_dir, filename = relative_libri.parts
            stem_fields = Path(filename).stem.split("-")
            if (
                len(stem_fields) < 3
                or speaker_dir != stem_fields[0]
                or chapter_dir != stem_fields[1]
            ):
                raise RecordedGenerationError(
                    f"external LibriSpeech row {offset} speaker/chapter directory와 filename이 다릅니다"
                )
            if raw_member_sha256 != digest or _SHA256_RE.fullmatch(raw_member_sha256) is None:
                raise RecordedGenerationError(
                    f"external LibriSpeech row {offset} raw/source SHA가 다릅니다"
                )
            if authority_metadata_sha256 != source_lineage["librispeech_chapters_sha256"]:
                raise RecordedGenerationError(
                    f"external LibriSpeech row {offset} CHAPTERS SHA가 holdout authority와 다릅니다"
                )
            expected_inventory_path = (
                f"data/raw/speech/LibriSpeech/{subset}/{speaker_dir}/{chapter_dir}/"
                f"{speaker_dir}-{chapter_dir}.trans.txt"
            )
            if (
                inventory_path != expected_inventory_path
                or _SHA256_RE.fullmatch(inventory_sha256) is None
            ):
                raise RecordedGenerationError(
                    f"external LibriSpeech row {offset} transcript inventory path/SHA가 유효하지 않습니다"
                )
            try:
                raw_identity_keys = public_lineage.librispeech_lineage_keys(
                    Path(path).name, source_lineage["librispeech_chapters"]
                )
            except ValueError as exc:
                raise RecordedGenerationError(
                    f"external LibriSpeech row {offset} reader/book 재유도 실패: {exc}"
                ) from exc
            identity_components = {
                source_lineage["librispeech_component_by_identity"].get(key)
                for key in raw_identity_keys
            }
            if None in identity_components or len(identity_components) != 1:
                raise RecordedGenerationError(
                    f"external LibriSpeech row {offset} CHAPTERS transitive component가 불완전합니다"
                )
            derived_raw_lineage = next(iter(identity_components))
            assert isinstance(derived_raw_lineage, str)
            digest12 = hashlib.sha256(
                derived_raw_lineage.encode("utf-8")
            ).hexdigest()[:12]
            derived_group = f"speech-librispeech-source-{digest12}"
            derived_lineage = f"speech-external-lineage-{digest12}"
            row_authority_tokens = {
                f"librispeech_component:{derived_raw_lineage}",
                f"clip_identity:{Path(path).name.casefold()}",
            }
            if (
                derived_raw_lineage in source_lineage["active_librispeech_components"]
                or raw_member_lineage_key != derived_raw_lineage
                or group != derived_group
                or lineage != derived_lineage
            ):
                raise RecordedGenerationError(
                    f"external LibriSpeech row {offset} reader/book transitive component가 "
                    "active82와 겹치거나 선언 lineage가 재유도값과 다릅니다"
                )
            if (
                transform != EXTERNAL_LIBRISPEECH_TRANSFORM
                or repeat_text != "1"
                or not math.isclose(
                    seconds,
                    EXTERNAL_LIBRISPEECH_SECONDS,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise RecordedGenerationError(
                    f"external LibriSpeech row {offset} transform/seconds 계약이 다릅니다"
                )
            if require_source_files:
                source_snapshot = _snapshot(
                    repo_root,
                    path,
                    label=f"external LibriSpeech source row {offset}",
                    capture_bytes=True,
                )
                if source_snapshot.sha256 != digest:
                    raise RecordedGenerationError(
                        f"external LibriSpeech row {offset} source SHA 불일치"
                    )
                assert source_snapshot.data is not None
                inventory_snapshot = _snapshot(
                    repo_root,
                    inventory_path,
                    label=f"external LibriSpeech transcript row {offset}",
                    capture_bytes=True,
                )
                if inventory_snapshot.sha256 != inventory_sha256:
                    raise RecordedGenerationError(
                        f"external LibriSpeech row {offset} transcript inventory SHA 불일치"
                    )
                assert inventory_snapshot.data is not None
                try:
                    transcript_ids = [
                        line.split(maxsplit=1)[0]
                        for line in inventory_snapshot.data.decode("utf-8").splitlines()
                        if line.strip()
                    ]
                except UnicodeDecodeError as exc:
                    raise RecordedGenerationError(
                        f"external LibriSpeech row {offset} transcript UTF-8 오류"
                    ) from exc
                if transcript_ids.count(Path(filename).stem) != 1:
                    raise RecordedGenerationError(
                        f"external LibriSpeech row {offset} utterance가 transcript inventory에 "
                        "정확히 한 번 존재하지 않습니다"
                    )
                import soundfile as sf

                try:
                    info = sf.info(io.BytesIO(source_snapshot.data))
                except RuntimeError as exc:
                    raise RecordedGenerationError(
                        f"external LibriSpeech row {offset} FLAC decode 실패: {exc}"
                    ) from exc
                duration = float(info.frames) / float(info.samplerate)
                if (
                    info.format != "FLAC"
                    or info.channels != 1
                    or info.samplerate <= 0
                    or start + seconds > duration + 1e-9
                ):
                    raise RecordedGenerationError(
                        f"external LibriSpeech row {offset}는 mono FLAC이며 "
                        f"duration >= start+{seconds:.1f}s여야 합니다: "
                        f"format={info.format}, channels={info.channels}, duration={duration}"
                    )
        else:
            raise RecordedGenerationError(
                f"source plan row {offset} source_kind가 허용되지 않습니다: {source_kind!r}"
            )
        if lineage in lineages:
            raise RecordedGenerationError(
                f"추가 녹음 lineage_key는 독립적이어야 합니다: {lineage}"
            )
        if digest in source_hashes:
            raise RecordedGenerationError(
                f"추가 녹음 원본 SHA가 중복됩니다: {digest}"
            )
        authority_overlap = sorted(used_authority_tokens & row_authority_tokens)
        if authority_overlap:
            raise RecordedGenerationError(
                "추가 source들이 서로 같은 권위 metadata component를 공유합니다: "
                f"row={offset}, overlap={authority_overlap[:5]}"
            )
        lineages.add(lineage)
        source_hashes.add(digest)
        used_authority_tokens.update(row_authority_tokens)
        if require_source_files:
            source_snapshot = source_snapshot or _snapshot(
                repo_root,
                path,
                label=f"source plan row {offset} source",
                capture_bytes=True,
            )
            if source_snapshot.sha256 != digest:
                raise RecordedGenerationError(
                    f"source plan row {offset} 원본 SHA 불일치: "
                    f"declared={digest}, actual={source_snapshot.sha256}"
                )
            assert source_snapshot.data is not None
            import soundfile as sf

            try:
                source_info = sf.info(io.BytesIO(source_snapshot.data))
            except RuntimeError as exc:
                raise RecordedGenerationError(
                    f"source plan row {offset} audio decode 실패: {exc}"
                ) from exc
            if source_info.samplerate <= 0 or source_info.channels < 1:
                raise RecordedGenerationError(
                    f"source plan row {offset} source format이 유효하지 않습니다"
                )
            source_duration = float(source_info.frames) / float(source_info.samplerate)
            if start + seconds > source_duration + 1e-9:
                raise RecordedGenerationError(
                    f"source plan row {offset} source window가 file duration을 넘습니다: "
                    f"start={start}, seconds={seconds}, duration={source_duration}"
                )
        rows.append(
            {
                "source_row_number": offset,
                "source_kind": source_kind,
                "path": path,
                "seconds": seconds,
                "start_seconds": start,
                "source_family": family,
                "group_id": group,
                "lineage_key": lineage,
                "split": split,
                "source_file_sha256": digest,
                "raw_member_path": raw_member_path,
                "raw_member_sha256": raw_member_sha256,
                "raw_member_lineage_key": raw_member_lineage_key,
                "authority_metadata_sha256": authority_metadata_sha256,
                "inventory_path": inventory_path,
                "inventory_sha256": inventory_sha256,
                "authority_components": sorted(row_authority_tokens),
                "transform": transform,
                "transform_repeat_count": int(repeat_text),
                "_source_bytes": (
                    source_snapshot.data
                    if source_snapshot is not None and require_source_files
                    else None
                ),
            }
        )
    _validate_addition_population(rows)
    return (
        snapshot,
        rows,
        str(source_lineage["evidence_sha256"]),
        selection_evidence,
    )


def _session_file_evidence(session_dir: Path, *, repo_root: Path) -> dict[str, object]:
    tree = snapshot_regular_tree_metadata(
        session_dir, repo_root=repo_root, label=f"recorded addition {session_dir.name}"
    )
    entries = [
        {"path": path, "size": size, "sha256": digest}
        for path, size, digest in tree.content_entries
    ]
    return {
        "file_count": tree.file_count,
        "total_bytes": sum(int(item[1]) for item in tree.content_entries),
        "aggregate_sha256": _canonical_json_sha256(entries),
    }


def _validate_session_artifacts(
    *,
    session_dir: Path,
    metadata: dict[str, Any],
    row: dict[str, Any],
    expected_seconds: float,
    repo_root: Path,
) -> None:
    """record_duct가 발행한 exact 3 WAV + session.json과 self evidence를 검증한다."""

    import numpy as np
    import soundfile as sf

    expected_names = {"mics.wav", "source.wav", "source_aligned.wav", "session.json"}
    observed_names = {path.name for path in session_dir.iterdir()}
    if observed_names != expected_names:
        raise RecordedGenerationError(
            f"추가 session artifact exact 집합이 다릅니다: {session_dir}; "
            f"missing={sorted(expected_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_names)}"
        )
    snapshots: dict[str, FileSnapshot] = {}
    for name in sorted(expected_names - {"session.json"}):
        relative = (session_dir / name).relative_to(repo_root).as_posix()
        snapshots[name] = _snapshot(
            repo_root,
            relative,
            label=f"recorded addition {session_dir.name}/{name}",
            capture_bytes=True,
        )
    expected_artifacts = [
        {
            "path": name,
            "size_bytes": snapshots[name].size,
            "sha256": snapshots[name].sha256,
        }
        for name in ("mics.wav", "source.wav", "source_aligned.wav")
    ]
    if metadata.get("artifacts") != expected_artifacts:
        raise RecordedGenerationError(
            f"추가 session artifact SHA/size evidence가 current bytes와 다릅니다: {session_dir}"
        )

    expected_frames_float = expected_seconds * 48_000.0
    expected_frames = int(round(expected_frames_float))
    if not math.isclose(
        expected_frames_float, float(expected_frames), rel_tol=0.0, abs_tol=1e-9
    ):
        raise RecordedGenerationError(
            f"추가 session seconds가 48kHz 정수 frame으로 표현되지 않습니다: {session_dir}"
        )
    expected_audio = {
        "mics.wav": (2, "PCM_32"),
        "source.wav": (1, "FLOAT"),
        "source_aligned.wav": (1, "FLOAT"),
    }
    decoded: dict[str, Any] = {}
    for name, (channels, subtype) in expected_audio.items():
        snapshot = snapshots[name]
        assert snapshot.data is not None
        try:
            info = sf.info(io.BytesIO(snapshot.data))
        except RuntimeError as exc:
            raise RecordedGenerationError(
                f"추가 session WAV decode 실패: {session_dir / name}: {exc}"
            ) from exc
        if (
            info.format != "WAV"
            or info.samplerate != 48_000
            or info.frames != expected_frames
            or info.channels != channels
            or info.subtype != subtype
        ):
            raise RecordedGenerationError(
                f"추가 session WAV shape/format 계약 불일치: {session_dir / name}; "
                f"format={info.format}, fs={info.samplerate}, frames={info.frames}, "
                f"channels={info.channels}, subtype={info.subtype}"
            )
        try:
            values, decoded_rate = sf.read(
                io.BytesIO(snapshot.data), dtype="float32", always_2d=True
            )
        except RuntimeError as exc:
            raise RecordedGenerationError(
                f"추가 session WAV content decode 실패: {session_dir / name}: {exc}"
            ) from exc
        if (
            decoded_rate != 48_000
            or not bool(np.isfinite(values).all())
            or float(np.max(np.abs(values))) <= 1e-7
        ):
            raise RecordedGenerationError(
                f"추가 session WAV content가 finite/non-silent가 아닙니다: {session_dir / name}"
            )
        decoded[name] = values

    source_bytes = row.get("_source_bytes")
    if source_bytes is not None:
        if not isinstance(source_bytes, bytes):
            raise RecordedGenerationError("source plan verified bytes가 유효하지 않습니다")
        program = metadata.get("program")
        assert isinstance(program, dict)
        from deep_anc.realtime.noise_gen import (
            NoiseProgram,
            render_recording_file_window,
        )

        try:
            expected_program = NoiseProgram(
                {
                    "type": "file",
                    "file": row["path"],
                    "file_start_seconds": row["start_seconds"],
                    "amplitude": float(program["amplitude"]),
                },
                48_000,
                file_bytes=source_bytes,
            )
            expected_source = render_recording_file_window(
                expected_program,
                expected_frames,
                sample_rate=48_000,
            )
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            raise RecordedGenerationError(
                f"추가 session deterministic playback을 source plan에서 재유도할 수 없습니다: "
                f"{session_dir}: {exc}"
            ) from exc
        observed_source = np.asarray(decoded["source.wav"][:, 0], dtype=np.float32)
        if not np.array_equal(observed_source, expected_source):
            max_error = float(np.max(np.abs(observed_source - expected_source)))
            raise RecordedGenerationError(
                "추가 session source.wav가 source plan bytes/start/amplitude에서 재유도한 "
                f"playback과 다릅니다: {session_dir}; max_abs_error={max_error}"
            )


def _read_session_metadata(session_dir: Path, *, repo_root: Path) -> dict[str, Any]:
    snapshot = _snapshot(
        repo_root,
        (session_dir / "session.json").relative_to(repo_root).as_posix(),
        label=f"recorded addition metadata {session_dir.name}",
        capture_bytes=True,
    )
    assert snapshot.data is not None
    try:
        value = json.loads(snapshot.data.decode("utf-8"), object_pairs_hook=_pairs_without_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordedGenerationError(f"session.json 오류: {session_dir}") from exc
    if not isinstance(value, dict):
        raise RecordedGenerationError(f"session.json 최상위가 object가 아닙니다: {session_dir}")
    return value


def _canonical_plan_path(value: object, *, repo_root: Path) -> str:
    if not isinstance(value, str) or not value:
        raise RecordedGenerationError("collection_plan.source_list가 없습니다")
    candidate = Path(value).expanduser()
    candidate = candidate if candidate.is_absolute() else repo_root / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        return candidate.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise RecordedGenerationError("collection_plan.source_list가 저장소 밖입니다") from exc


def _expected_addition_entry(
    *, row: dict[str, Any], metadata: dict[str, Any], session_dir: Path, manifest_path: Path
) -> dict[str, Any]:
    try:
        sample_rate = int(metadata["sample_rate"])
        seconds = float(metadata["seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RecordedGenerationError(f"추가 session sample_rate/seconds 오류: {session_dir}") from exc
    if sample_rate != 48_000 or not math.isclose(seconds, row["seconds"], rel_tol=0.0, abs_tol=1e-9):
        raise RecordedGenerationError(
            f"추가 session sample_rate/seconds가 source plan과 다릅니다: {session_dir}"
        )
    relative_path = Path(os.path.relpath(session_dir, manifest_path.parent)).as_posix()
    return {
        "path": relative_path,
        "path_base": "manifest",
        "duration_s": seconds,
        "sample_rate": sample_rate,
        "channels": 2,
        "tag": "recorded",
        "session_id": session_dir.name,
        "group_id": row["lineage_key"],
        "source_family": row["source_family"],
        "metadata_inferred": [],
        "source_pool_group_id": row["group_id"],
        "lineage_schema": ADDITION_LINEAGE_SCHEMA,
        "split": row["split"],
    }


def _validate_additions(
    *,
    repo_root: Path,
    generation_id: str,
    additions_root: str,
    source_plan: str,
    combined_manifest_path: Path,
    require_source_files: bool,
) -> tuple[dict[str, object], list[dict[str, Any]]]:
    expected_root = f"{ADDITIONS_ROOT}/{generation_id}"
    expected_plan = f"{SOURCE_PLAN_ROOT}/{generation_id}.csv"
    if additions_root != expected_root or source_plan != expected_plan:
        raise RecordedGenerationError(
            "generation additions root/source plan 경로가 generation_id와 다릅니다: "
            f"root={additions_root}, plan={source_plan}"
        )
    (
        plan_snapshot,
        rows,
        source_lineage_sha256,
        source_selection_evidence,
    ) = _read_source_plan(
        repo_root=repo_root, relative=source_plan, require_source_files=require_source_files
    )
    root = reject_symlink_components(repo_root / additions_root, root=repo_root)
    if not root.is_dir():
        raise RecordedGenerationError(f"recorded additions root가 directory가 아닙니다: {root}")
    children = sorted(root.iterdir(), key=lambda item: item.name)
    session_dirs: list[Path] = []
    root_files: list[str] = []
    for child in children:
        info = child.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise RecordedGenerationError(f"recorded additions symlink 금지: {child}")
        if stat.S_ISDIR(info.st_mode):
            session_dirs.append(child)
        elif stat.S_ISREG(info.st_mode) and child.name == "batch_progress.csv":
            root_files.append(child.name)
        else:
            raise RecordedGenerationError(
                f"recorded additions root에는 session directory와 batch_progress.csv만 허용합니다: {child}"
            )
    if len(session_dirs) != ADDITION_SESSION_COUNT or root_files != ["batch_progress.csv"]:
        raise RecordedGenerationError(
            f"recorded additions는 {ADDITION_SESSION_COUNT} session과 batch_progress.csv가 필요합니다: "
            f"sessions={len(session_dirs)}, root_files={root_files}"
        )

    row_by_number = {int(row["source_row_number"]): row for row in rows}
    observed_rows: set[int] = set()
    sessions: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    for session_dir in session_dirs:
        metadata = _read_session_metadata(session_dir, repo_root=repo_root)
        if metadata.get("session_id") != session_dir.name:
            raise RecordedGenerationError(f"session_id와 directory가 다릅니다: {session_dir}")
        plan = metadata.get("collection_plan")
        required_plan_keys = {
            "status",
            "source_list",
            "source_list_sha256",
            "source_row_number",
            "lineage_key",
            "preassigned_split",
            "split_source",
            "source_file_sha256",
            "start_seconds",
        }
        if not isinstance(plan, dict) or set(plan) != required_plan_keys or plan.get("status") != "exact":
            raise RecordedGenerationError(f"추가 session collection_plan이 exact가 아닙니다: {session_dir}")
        if _canonical_plan_path(plan.get("source_list"), repo_root=repo_root) != source_plan:
            raise RecordedGenerationError(f"추가 session source_list 경로 불일치: {session_dir}")
        if plan.get("source_list_sha256") != plan_snapshot.sha256:
            raise RecordedGenerationError(f"추가 session source_list SHA 불일치: {session_dir}")
        row_number = plan.get("source_row_number")
        if isinstance(row_number, bool) or not isinstance(row_number, int) or row_number not in row_by_number:
            raise RecordedGenerationError(f"추가 session source row가 유효하지 않습니다: {session_dir}")
        if row_number in observed_rows:
            raise RecordedGenerationError(f"source plan row가 여러 session에 사용됐습니다: {row_number}")
        observed_rows.add(row_number)
        row = row_by_number[row_number]
        expected_plan = {
            "status": "exact",
            "source_list": plan["source_list"],
            "source_list_sha256": plan_snapshot.sha256,
            "source_row_number": row_number,
            "lineage_key": row["lineage_key"],
            "preassigned_split": row["split"],
            "split_source": "csv",
            "source_file_sha256": row["source_file_sha256"],
            "start_seconds": row["start_seconds"],
        }
        if plan != expected_plan:
            raise RecordedGenerationError(f"추가 session collection_plan/CSV 불일치: {session_dir}")
        program = metadata.get("program")
        required_program_keys = {
            "type",
            "frequency",
            "amplitude",
            "band",
            "file",
            "file_start_seconds",
        }
        amplitude = program.get("amplitude") if isinstance(program, dict) else None
        file_start = (
            program.get("file_start_seconds") if isinstance(program, dict) else None
        )
        frequency = program.get("frequency") if isinstance(program, dict) else None
        band = program.get("band") if isinstance(program, dict) else None
        if (
            not isinstance(program, dict)
            or set(program) != required_program_keys
            or program.get("type") != "file"
            or program.get("file") != row["path"]
            or isinstance(amplitude, bool)
            or not isinstance(amplitude, (int, float))
            or not math.isfinite(float(amplitude))
            or not (0.0 < float(amplitude) <= 0.15)
            or isinstance(file_start, bool)
            or not isinstance(file_start, (int, float))
            or not math.isclose(
                float(file_start), row["start_seconds"], rel_tol=0.0, abs_tol=1e-9
            )
            or isinstance(frequency, bool)
            or not isinstance(frequency, (int, float))
            or not math.isfinite(float(frequency))
            or not isinstance(band, list)
            or len(band) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in band
            )
            or metadata.get("source_family") != row["source_family"]
            or metadata.get("group_id") != row["group_id"]
            or metadata.get("preassigned_split") != row["split"]
        ):
            raise RecordedGenerationError(f"추가 session top-level metadata/CSV 불일치: {session_dir}")
        channels = metadata.get("channels")
        confirmations = metadata.get("safety_confirmations")
        timeline = metadata.get("timeline")
        if (
            metadata.get("block_size") != 256
            or channels
            != {"err_mic": 0, "ref_mic": 1, "noise_out": 0, "cancel_out": 1}
            or confirmations
            != {
                "user_present": True,
                "volume_minimum": True,
                "routing_and_geometry": True,
            }
            or not isinstance(timeline, dict)
            or timeline.get("method") != TIMELINE_METHOD
            or timeline.get("witness_channel") != 1
            or timeline.get("usable_for_digital_reference") is not True
        ):
            raise RecordedGenerationError(
                f"추가 session block/channel/safety/timeline provenance가 canonical이 아닙니다: "
                f"{session_dir}"
            )
        timeline_numbers = {
            "valid_window_ratio": 0.90,
            "aligned_lag_median_samples": None,
            "aligned_lag_robust_std_samples": 0.0,
            "coh2_150_600_after": 0.90,
        }
        for key, minimum in timeline_numbers.items():
            value = timeline.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (minimum is not None and float(value) < minimum)
            ):
                raise RecordedGenerationError(
                    f"추가 session timeline.{key}가 유효하지 않습니다: {session_dir}"
                )
        _validate_session_artifacts(
            session_dir=session_dir,
            metadata=metadata,
            row=row,
            expected_seconds=float(row["seconds"]),
            repo_root=repo_root,
        )
        evidence = _session_file_evidence(session_dir, repo_root=repo_root)
        sessions.append(
            {
                "session_id": session_dir.name,
                "path": session_dir.relative_to(repo_root).as_posix(),
                "source_row_number": row_number,
                "source_kind": row["source_kind"],
                "source_path": row["path"],
                "source_file_sha256": row["source_file_sha256"],
                "raw_member_path": row["raw_member_path"],
                "raw_member_sha256": row["raw_member_sha256"],
                "raw_member_lineage_key": row["raw_member_lineage_key"],
                "authority_metadata_sha256": row["authority_metadata_sha256"],
                "authority_components": row["authority_components"],
                "transform": row["transform"],
                "transform_repeat_count": row["transform_repeat_count"],
                "source_family": row["source_family"],
                "group_id": row["group_id"],
                "lineage_key": row["lineage_key"],
                "split": row["split"],
                **evidence,
            }
        )
        manifest_entries.append(
            _expected_addition_entry(
                row=row,
                metadata=metadata,
                session_dir=session_dir,
                manifest_path=combined_manifest_path,
            )
        )
    if observed_rows != set(row_by_number):
        raise RecordedGenerationError("source plan의 일부 행이 session에 정확히 대응하지 않습니다")

    progress_snapshot = _snapshot(
        repo_root,
        f"{additions_root}/batch_progress.csv",
        label="recorded additions batch progress",
        capture_bytes=True,
    )
    assert progress_snapshot.data is not None
    try:
        progress = list(csv.DictReader(io.StringIO(progress_snapshot.data.decode("utf-8"))))
    except UnicodeDecodeError as exc:
        raise RecordedGenerationError("batch_progress.csv UTF-8 오류") from exc
    successful = [item for item in progress if item.get("verdict") == "ok"]
    progress_pairs: set[tuple[int, str | None]] = set()
    row_by_number = {int(row["source_row_number"]): row for row in rows}
    for item in successful:
        row_text = item.get("source_row_number", "")
        try:
            duration = float(item.get("seconds", ""))
        except (TypeError, ValueError) as exc:
            raise RecordedGenerationError(
                "batch_progress.csv PASS 행 seconds가 없습니다"
            ) from exc
        if not row_text.isdigit():
            raise RecordedGenerationError(
                "batch_progress.csv PASS 행 source_row_number가 유효하지 않습니다"
            )
        row_number = int(row_text)
        planned = row_by_number.get(row_number)
        if (
            planned is None
            or not math.isfinite(duration)
            or duration != float(planned["seconds"])
        ):
            raise RecordedGenerationError(
                "batch_progress.csv PASS duration이 source plan과 다릅니다"
            )
        progress_pairs.add((row_number, item.get("session_id")))
    expected_pairs = {(int(item["source_row_number"]), str(item["session_id"])) for item in sessions}
    if progress_pairs != expected_pairs or len(successful) != ADDITION_SESSION_COUNT:
        raise RecordedGenerationError("batch_progress.csv의 PASS 행과 추가 session exact 집합이 다릅니다")

    additions_tree = snapshot_regular_tree_metadata(
        root, repo_root=repo_root, label="recorded additions generation tree"
    )
    sessions.sort(key=lambda item: int(item["source_row_number"]))
    manifest_entries.sort(key=lambda item: str(item["session_id"]))
    summary = {
        "root": additions_root,
        "expected_session_count": ADDITION_SESSION_COUNT,
        "source_plan": _file_ref(plan_snapshot, repo_root=repo_root),
        "source_lineage_evidence_sha256": source_lineage_sha256,
        "source_selection": source_selection_evidence,
        "tree": _tree_summary(additions_tree),
        "sessions": sessions,
        "session_aggregate_sha256": _canonical_json_sha256(sessions),
    }
    return summary, manifest_entries


def _parent_summary(
    *, repo_root: Path, expected_holdout_sha256: str, combined_manifest_path: Path
) -> tuple[dict[str, object], list[dict[str, Any]]]:
    if _SHA256_RE.fullmatch(expected_holdout_sha256 or "") is None:
        raise RecordedGenerationError("parent holdout SHA-256이 유효하지 않습니다")
    try:
        holdout = validate_holdout_contract(
            repo_root / PARENT_HOLDOUT,
            repo_root=repo_root,
            expected_sha256=expected_holdout_sha256,
        )
    except (OSError, HoldoutContractError) as exc:
        raise RecordedGenerationError(f"parent holdout 검증 실패: {exc}") from exc
    lineage = holdout.get("lineage")
    tree_anchor = holdout.get("recorded_tree")
    if not isinstance(lineage, dict) or not isinstance(tree_anchor, dict):
        raise RecordedGenerationError("parent holdout lineage/recorded_tree anchor가 없습니다")
    if lineage.get("regrouped_manifest") != PARENT_MANIFEST:
        raise RecordedGenerationError("parent holdout regrouped manifest 경로가 다릅니다")
    manifest_snapshot = _snapshot(
        repo_root, PARENT_MANIFEST, label="parent recorded manifest", capture_bytes=True
    )
    if manifest_snapshot.sha256 != lineage.get("regrouped_manifest_sha256"):
        raise RecordedGenerationError("parent recorded manifest SHA가 holdout anchor와 다릅니다")
    entries = _manifest_entries(manifest_snapshot)
    if len(entries) != PARENT_SESSION_COUNT or lineage.get("regrouped_row_count") != len(entries):
        raise RecordedGenerationError(f"parent manifest는 정확히 {PARENT_SESSION_COUNT}행이어야 합니다")
    parent_root = repo_root / PARENT_ROOT
    parent_tree = snapshot_regular_tree_metadata(
        parent_root, repo_root=repo_root, label="parent recorded immutable tree"
    )
    actual_tree = _tree_summary(parent_tree)
    anchored_tree = {
        "file_count": tree_anchor.get("file_count"),
        "metadata_snapshot_sha256": tree_anchor.get("metadata_snapshot_sha256"),
        "content_snapshot_sha256": tree_anchor.get("content_snapshot_sha256"),
    }
    if {key: actual_tree[key] for key in anchored_tree} != anchored_tree:
        raise RecordedGenerationError("parent 82 recorded tree가 holdout anchor 이후 변경됐습니다")

    session_ids: set[str] = set()
    combined_entries: list[dict[str, Any]] = []
    for entry in entries:
        session_id = entry.get("session_id")
        if not isinstance(session_id, str) or session_id in session_ids:
            raise RecordedGenerationError("parent manifest session_id 누락/중복")
        session_ids.add(session_id)
        resolved = _resolved_manifest_session(entry, manifest_snapshot.path)
        expected = Path(os.path.abspath(parent_root / session_id))
        if resolved != expected or not (expected / "session.json").is_file():
            raise RecordedGenerationError(f"parent manifest session path 불일치: {session_id}")
        copied = dict(entry)
        copied["path"] = Path(os.path.relpath(expected, combined_manifest_path.parent)).as_posix()
        combined_entries.append(copied)
    provenance_path = holdout.get("provenance_report")
    provenance_sha = holdout.get("provenance_report_sha256")
    if not isinstance(provenance_path, str) or not isinstance(provenance_sha, str):
        raise RecordedGenerationError("parent provenance report pointer가 불완전합니다")
    provenance_snapshot = _snapshot(repo_root, provenance_path, label="parent provenance report")
    if provenance_snapshot.sha256 != provenance_sha:
        raise RecordedGenerationError("parent provenance report SHA가 holdout anchor와 다릅니다")
    holdout_snapshot = _snapshot(repo_root, PARENT_HOLDOUT, label="parent holdout")
    summary = {
        "root": PARENT_ROOT,
        "session_count": PARENT_SESSION_COUNT,
        "manifest": _file_ref(manifest_snapshot, repo_root=repo_root),
        "holdout": _file_ref(holdout_snapshot, repo_root=repo_root),
        "provenance_report": _file_ref(provenance_snapshot, repo_root=repo_root),
        "tree": actual_tree,
        "session_ids": sorted(session_ids),
        "session_aggregate_sha256": _canonical_json_sha256(sorted(session_ids)),
    }
    return summary, combined_entries


def build_recorded_generation_payload(
    *,
    repo_root: str | Path,
    generation_id: str,
    expected_holdout_sha256: str,
    require_source_files: bool = True,
) -> dict[str, Any]:
    """이미 no-replace 발행된 combined manifest로 generation payload를 유도한다."""

    root = Path(os.path.abspath(repo_root))
    generation = _generation_id(generation_id)
    directory = f"{GENERATION_ROOT}/{generation}"
    combined_relative = f"{directory}/recorded.jsonl"
    combined_path = root / combined_relative
    parent, expected_parent_entries = _parent_summary(
        repo_root=root,
        expected_holdout_sha256=expected_holdout_sha256,
        combined_manifest_path=combined_path,
    )
    additions, expected_addition_entries = _validate_additions(
        repo_root=root,
        generation_id=generation,
        additions_root=f"{ADDITIONS_ROOT}/{generation}",
        source_plan=f"{SOURCE_PLAN_ROOT}/{generation}.csv",
        combined_manifest_path=combined_path,
        require_source_files=require_source_files,
    )
    parent_groups = {str(item["group_id"]) for item in expected_parent_entries}
    addition_groups = {str(item["group_id"]) for item in expected_addition_entries}
    overlap = sorted(parent_groups & addition_groups)
    if overlap:
        raise RecordedGenerationError(
            f"추가 lineage가 parent component와 겹칩니다(독립 group 아님): {overlap[:5]}"
        )
    parent_ids = {str(item["session_id"]) for item in expected_parent_entries}
    addition_ids = {str(item["session_id"]) for item in expected_addition_entries}
    if parent_ids & addition_ids:
        raise RecordedGenerationError("parent/additions session_id가 겹칩니다")

    combined_snapshot = _snapshot(
        root, combined_relative, label="combined recorded generation manifest", capture_bytes=True
    )
    actual_entries = _manifest_entries(combined_snapshot)
    expected_by_id = {
        str(item["session_id"]): item
        for item in [*expected_parent_entries, *expected_addition_entries]
    }
    actual_by_id: dict[str, dict[str, Any]] = {}
    for entry in actual_entries:
        session_id = entry.get("session_id")
        if not isinstance(session_id, str) or session_id in actual_by_id:
            raise RecordedGenerationError("combined manifest session_id 누락/중복")
        actual_by_id[session_id] = entry
    if len(actual_entries) != COMBINED_SESSION_COUNT or actual_by_id != expected_by_id:
        missing = sorted(set(expected_by_id) - set(actual_by_id))[:5]
        extra = sorted(set(actual_by_id) - set(expected_by_id))[:5]
        changed = sorted(
            key for key in set(actual_by_id) & set(expected_by_id)
            if actual_by_id[key] != expected_by_id[key]
        )[:5]
        raise RecordedGenerationError(
            "combined manifest가 parent 82 + exact additions 17과 다릅니다: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    for entry in actual_entries:
        resolved = _resolved_manifest_session(entry, combined_snapshot.path)
        if not (resolved / "session.json").is_file():
            raise RecordedGenerationError(
                f"combined manifest session directory가 없습니다: {entry.get('session_id')}"
            )
    combined = {
        "manifest": _file_ref(combined_snapshot, repo_root=root),
        "session_count": COMBINED_SESSION_COUNT,
        "parent_session_count": PARENT_SESSION_COUNT,
        "addition_session_count": ADDITION_SESSION_COUNT,
        "session_ids": sorted(actual_by_id),
        "session_aggregate_sha256": _canonical_json_sha256(sorted(actual_by_id)),
    }
    payload: dict[str, Any] = {
        "schema_version": RECORDED_GENERATION_SCHEMA_VERSION,
        "generation_id": generation,
        "parent": parent,
        "additions": additions,
        "combined": combined,
    }
    payload["evidence_sha256"] = _canonical_json_sha256(payload)
    return payload


def build_combined_manifest_bytes(
    *,
    repo_root: str | Path,
    generation_id: str,
    expected_holdout_sha256: str,
    require_source_files: bool = True,
) -> bytes:
    """parent와 additions 증거에서만 combined JSONL bytes를 만든다."""

    root = Path(os.path.abspath(repo_root))
    generation = _generation_id(generation_id)
    combined_path = root / GENERATION_ROOT / generation / "recorded.jsonl"
    _parent, parent_entries = _parent_summary(
        repo_root=root,
        expected_holdout_sha256=expected_holdout_sha256,
        combined_manifest_path=combined_path,
    )
    _additions, addition_entries = _validate_additions(
        repo_root=root,
        generation_id=generation,
        additions_root=f"{ADDITIONS_ROOT}/{generation}",
        source_plan=f"{SOURCE_PLAN_ROOT}/{generation}.csv",
        combined_manifest_path=combined_path,
        require_source_files=require_source_files,
    )
    parent_groups = {str(item["group_id"]) for item in parent_entries}
    addition_groups = {str(item["group_id"]) for item in addition_entries}
    if parent_groups & addition_groups:
        raise RecordedGenerationError("추가 lineage가 parent component와 겹칩니다")
    entries = sorted([*parent_entries, *addition_entries], key=lambda item: str(item["session_id"]))
    if len(entries) != COMBINED_SESSION_COUNT:
        raise RecordedGenerationError("combined manifest session 수가 99가 아닙니다")
    return b"".join(
        (json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for entry in entries
    )


def validate_recorded_generation(
    path: str | Path,
    *,
    repo_root: str | Path,
    expected_sha256: str | None = None,
    require_source_files: bool = True,
) -> dict[str, Any]:
    """generation report와 현재 parent/additions/combined bytes를 전부 재유도한다."""

    root = Path(os.path.abspath(repo_root))
    try:
        snapshot = read_regular_file_snapshot(
            path, root=root, label="recorded generation report", capture_bytes=True
        )
    except HoldoutContractError as exc:
        raise RecordedGenerationError(str(exc)) from exc
    if expected_sha256 is not None:
        if _SHA256_RE.fullmatch(expected_sha256 or "") is None or snapshot.sha256 != expected_sha256:
            raise RecordedGenerationError("recorded generation report 외부 SHA-256 불일치")
    assert snapshot.data is not None
    try:
        payload = json.loads(
            snapshot.data.decode("utf-8"), object_pairs_hook=_pairs_without_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordedGenerationError(f"recorded generation JSON 오류: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "generation_id", "parent", "additions", "combined", "evidence_sha256"
    }:
        raise RecordedGenerationError("recorded generation 최상위 schema가 불완전합니다")
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != RECORDED_GENERATION_SCHEMA_VERSION
    ):
        raise RecordedGenerationError("recorded generation schema_version은 1이어야 합니다")
    generation = _generation_id(payload.get("generation_id"))
    expected_path = root / GENERATION_ROOT / generation / "generation.json"
    if Path(os.path.abspath(snapshot.path)) != Path(os.path.abspath(expected_path)):
        raise RecordedGenerationError(
            f"recorded generation report 경로가 generation_id와 다릅니다: {snapshot.path}"
        )
    parent = payload.get("parent")
    if not isinstance(parent, dict):
        raise RecordedGenerationError("recorded generation parent가 object가 아닙니다")
    holdout_ref = parent.get("holdout")
    if (
        not isinstance(holdout_ref, dict)
        or set(holdout_ref) != {"path", "sha256", "size"}
        or holdout_ref.get("path") != PARENT_HOLDOUT
        or not isinstance(holdout_ref.get("sha256"), str)
    ):
        raise RecordedGenerationError("recorded generation parent holdout ref가 불완전합니다")
    derived = build_recorded_generation_payload(
        repo_root=root,
        generation_id=generation,
        expected_holdout_sha256=str(holdout_ref["sha256"]),
        require_source_files=require_source_files,
    )
    if payload != derived:
        raise RecordedGenerationError(
            "recorded generation report가 현재 immutable parent/additions/combined 증거와 다릅니다"
        )
    evidence = dict(payload)
    declared = evidence.pop("evidence_sha256")
    if declared != _canonical_json_sha256(evidence):
        raise RecordedGenerationError("recorded generation evidence_sha256 불일치")
    return {
        "generation_sha256": snapshot.sha256,
        "generation_id": generation,
        "parent_session_count": PARENT_SESSION_COUNT,
        "addition_session_count": ADDITION_SESSION_COUNT,
        "recorded_session_count": COMBINED_SESSION_COUNT,
        "recorded_manifest": derived["combined"]["manifest"],
        "parent": derived["parent"],
        "additions": derived["additions"],
        "combined": derived["combined"],
        "_validated_generation_snapshot": snapshot,
    }
