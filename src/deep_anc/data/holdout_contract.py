"""Canonical recorded holdout의 무결성 계약.

이 모듈은 Elice venv가 만들어지기 *전*에도 실행할 수 있도록 Python 표준 라이브러리만
사용한다. 부트스트랩은 사용자가 별도 채널에서 전달한 holdout SHA-256과 이 구조 계약을
모두 확인하기 전에는 환경 설치나 데이터 다운로드를 시작하면 안 된다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import stat
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


REQUIRED_FAMILIES = frozenset({"environment", "machine", "music", "speech"})
PROVENANCE_AUTHORITY = "historical_builder_reproduction_plus_pcm_validation"
EXPECTED_HISTORICAL_BUILDERS = {
    "v1": {
        "commit": "7c7800fa94a8c5e156e049be896fd0b9586d983f",
        "path": "scripts/data/build_recording_sources.py",
        "source_sha256": "26d7fa6987310d6fd58f68a117a67a5e9397453aa96b61d1713838fc37452140",
    },
    "v2": {
        "commit": "0cb13b14e36c334783953aedd47aa0bc13d0fb6a",
        "path": "scripts/data/build_recording_sources.py",
        "source_sha256": "fc0f5fa428be4897291bcd486793ce1c08d2f5faa30306c463b8a04560fe71bc",
    },
}
EXPECTED_SOURCES_CSV = (
    "data/source_pool/sources.csv",
    "data/source_pool_v2/sources.csv",
)
EXPECTED_POOL_ROW_COUNT = 80
EXPECTED_POOL_FAMILY_ROWS = 20
EXPECTED_SAMPLE_RATE = 48_000
EXPECTED_FRAMES = 3_360_000
FMA_TRACKS_CSV = "data/raw/music/fma_metadata/tracks.csv"
FMA_TRACKS_CSV_SHA256 = "f73260fd112b8cd42bcd4f7c8918fc66b19d9d4c7b97f4faedce524b59e95d6b"
LIBRISPEECH_CHAPTERS = "data/raw/speech/LibriSpeech/CHAPTERS.TXT"
ESC50_METADATA = "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv"
RECORDED_REGROUPED_MANIFEST = "data/manifests/recorded_regrouped.jsonl"
RECORDED_TREE_SNAPSHOT_ENCODING = (
    "sha256(canonical-json-v1:sorted[{relative_path,size,mtime_ns}])"
)
RECORDED_TREE_CONTENT_SNAPSHOT_ENCODING = (
    "sha256(canonical-json-v1:sorted[{relative_path,size,sha256}])"
)
RECORDED_CONTENT_INTEGRITY_BOUNDARY = (
    "provenance repair는 모든 recorded regular file을 same-FD SHA-256으로 전후 보호하고; "
    "Elice transfer manifest는 같은 파일을 per-file size/SHA-256으로 다시 보호한다"
)
EXPECTED_SELECTION_COUNTS = {
    "v1_rows": 80,
    "v1_placements": 983,
    "v1_unique": 854,
    "v1_historical_exclusion_unique": 691,
    "v2_rows": 80,
    "v2_placements": 416,
}
EXPECTED_INVOCATIONS = {
    "v1_invocations": [["environment", "machine"], ["speech"], ["music"]],
    "v2_invocations": [["environment", "machine", "speech", "music"]],
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class HoldoutContractError(ValueError):
    """Canonical holdout 또는 그 provenance 증거가 계약을 위반했다."""


def _public_lineage_module():
    """package import와 ``python holdout_contract.py`` CLI를 모두 지원한다."""

    try:
        from . import public_lineage
    except ImportError:  # direct script bootstrap preflight
        source_root = str(Path(__file__).resolve().parents[2])
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        from deep_anc.data import public_lineage
    return public_lineage


@dataclass(frozen=True)
class FileSnapshot:
    """한 regular-file descriptor에서 얻은 bytes/hash/stat 증거."""

    path: Path
    data: bytes | None
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    def stat_contract(self) -> dict[str, int | str]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


@dataclass(frozen=True)
class TreeMetadataSnapshot:
    """regular-file tree의 metadata와 same-FD content canonical snapshot."""

    root: Path
    entries: tuple[tuple[str, int, int], ...]
    sha256: str
    content_entries: tuple[tuple[str, int, str], ...]
    content_sha256: str

    @property
    def file_count(self) -> int:
        return len(self.entries)


def _tree_metadata_bytes(entries: tuple[tuple[str, int, int], ...]) -> bytes:
    payload = [
        {"relative_path": relative, "size": size, "mtime_ns": mtime_ns}
        for relative, size, mtime_ns in entries
    ]
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _tree_content_bytes(entries: tuple[tuple[str, int, str], ...]) -> bytes:
    payload = [
        {"relative_path": relative, "size": size, "sha256": digest}
        for relative, size, digest in entries
    ]
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def snapshot_regular_tree_metadata(
    tree_root: str | Path,
    *,
    repo_root: str | Path,
    label: str,
) -> TreeMetadataSnapshot:
    """symlink 없이 metadata와 모든 file content의 same-FD snapshot을 만든다."""

    boundary = _lexical_absolute(Path(repo_root))
    root = reject_symlink_components(tree_root, root=boundary)
    try:
        root_info = root.lstat()
    except FileNotFoundError as exc:  # reject_symlink_components와 경합한 경우
        raise HoldoutContractError(f"{label} root가 사라졌습니다: {root}") from exc
    if not stat.S_ISDIR(root_info.st_mode):
        raise HoldoutContractError(f"{label} root는 directory여야 합니다: {root}")

    values: list[tuple[str, int, int]] = []
    content_values: list[tuple[str, int, str]] = []
    for current_text, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_text)
        current_info = current.lstat()
        if stat.S_ISLNK(current_info.st_mode) or not stat.S_ISDIR(current_info.st_mode):
            raise HoldoutContractError(f"{label} directory가 안전하지 않습니다: {current}")
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            child = current / name
            child_info = child.lstat()
            if stat.S_ISLNK(child_info.st_mode) or not stat.S_ISDIR(child_info.st_mode):
                raise HoldoutContractError(
                    f"{label}에는 symlink/비-directory 하위 경로를 둘 수 없습니다: {child}"
                )
        for name in file_names:
            child = current / name
            file_snapshot = read_regular_file_snapshot(
                child,
                root=boundary,
                label=f"{label} content",
                capture_bytes=False,
            )
            relative = child.relative_to(root).as_posix()
            values.append(
                (
                    relative,
                    file_snapshot.size,
                    file_snapshot.mtime_ns,
                )
            )
            content_values.append(
                (relative, file_snapshot.size, file_snapshot.sha256)
            )

    entries = tuple(sorted(values, key=lambda item: item[0]))
    content_entries = tuple(sorted(content_values, key=lambda item: item[0]))
    return TreeMetadataSnapshot(
        root=root,
        entries=entries,
        sha256=hashlib.sha256(_tree_metadata_bytes(entries)).hexdigest(),
        content_entries=content_entries,
        content_sha256=hashlib.sha256(_tree_content_bytes(content_entries)).hexdigest(),
    )


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def reject_symlink_components(
    path: str | Path,
    *,
    root: str | Path,
    allow_missing_leaf: bool = False,
) -> Path:
    """resolve로 symlink를 따라가기 전에 lexical containment와 모든 component를 검사한다."""

    candidate = _lexical_absolute(Path(path))
    boundary = _lexical_absolute(Path(root))
    try:
        relative = candidate.relative_to(boundary)
    except ValueError as exc:
        raise HoldoutContractError(f"경로가 허용 루트 밖입니다: {candidate} (root={boundary})") from exc
    current = boundary
    components = (Path("."), *[Path(part) for part in relative.parts])
    for index, part in enumerate(components):
        if part != Path("."):
            current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if allow_missing_leaf and index == len(components) - 1:
                return candidate
            raise HoldoutContractError(f"경로 component가 없습니다: {current}")
        if stat.S_ISLNK(info.st_mode):
            raise HoldoutContractError(f"symlink 경로 component는 허용하지 않습니다: {current}")
    return candidate


def read_regular_file_snapshot(
    path: str | Path,
    *,
    root: str | Path,
    label: str,
    capture_bytes: bool = True,
) -> FileSnapshot:
    """O_NOFOLLOW fd 하나로 hash/bytes/stat를 만들고 pathname retarget도 대조한다."""

    candidate = reject_symlink_components(path, root=root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise HoldoutContractError(f"{label}을 안전하게 열 수 없습니다: {candidate}: {exc}") from exc
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if capture_bytes else None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise HoldoutContractError(f"{label}은 regular file이어야 합니다: {candidate}")
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise HoldoutContractError(f"{label}을 읽는 동안 파일이 변경됐습니다: {candidate}")
    if total != int(after.st_size):
        raise HoldoutContractError(f"{label} byte 수가 fstat size와 다릅니다: {candidate}")
    try:
        pathname = candidate.lstat()
    except FileNotFoundError as exc:
        raise HoldoutContractError(f"{label} pathname이 읽는 동안 사라졌습니다: {candidate}") from exc
    if stat.S_ISLNK(pathname.st_mode) or (pathname.st_dev, pathname.st_ino) != (
        after.st_dev,
        after.st_ino,
    ):
        raise HoldoutContractError(f"{label} pathname이 읽는 동안 retarget됐습니다: {candidate}")
    return FileSnapshot(
        path=candidate,
        data=b"".join(chunks) if chunks is not None else None,
        sha256=digest.hexdigest(),
        device=int(after.st_dev),
        inode=int(after.st_ino),
        size=int(after.st_size),
        mtime_ns=int(after.st_mtime_ns),
        ctime_ns=int(after.st_ctime_ns),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HoldoutContractError(f"JSON 중복 키는 허용되지 않습니다: {key}")
        result[key] = value
    return result


def _load_json_bytes(raw: bytes, *, path: Path, label: str) -> dict[str, Any]:
    if not raw.strip():
        raise HoldoutContractError(f"{label}이 비어 있습니다: {path}")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_object_without_duplicate_keys
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HoldoutContractError(f"{label} JSON이 잘렸거나 손상됐습니다: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HoldoutContractError(f"{label} 최상위 값은 JSON object여야 합니다: {path}")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HoldoutContractError(f"{field}는 양의 정수여야 합니다")
    return value


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise HoldoutContractError(f"{field}는 앞뒤 공백 없는 비어 있지 않은 문자열이어야 합니다")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise HoldoutContractError(f"{field}는 소문자 64자리 SHA-256이어야 합니다")
    return value


def _repo_relative_file(repo_root: Path, value: Any, *, field: str) -> Path:
    relative = Path(_nonempty_string(value, field=field))
    if relative.is_absolute():
        raise HoldoutContractError(f"{field}는 저장소 상대경로여야 합니다")
    root = _lexical_absolute(repo_root)
    resolved = _lexical_absolute(root / relative)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HoldoutContractError(f"{field}가 저장소 밖을 가리킵니다: {relative}") from exc
    try:
        resolved.lstat()
    except FileNotFoundError as exc:
        raise HoldoutContractError(f"{field} 파일이 없습니다: {relative}") from exc
    return reject_symlink_components(resolved, root=root)


def _validate_families(payload: dict[str, Any]) -> tuple[int, dict[str, tuple[str, ...]]]:
    families = payload.get("families")
    if not isinstance(families, dict) or set(families) != REQUIRED_FAMILIES:
        raise HoldoutContractError(
            "families는 environment/machine/music/speech 네 계열을 정확히 포함해야 합니다"
        )

    global_clips: dict[str, str] = {}
    total = 0
    normalised: dict[str, tuple[str, ...]] = {}
    for family in sorted(REQUIRED_FAMILIES):
        clips = families[family]
        if not isinstance(clips, list) or not clips:
            raise HoldoutContractError(f"families.{family}는 비어 있지 않은 배열이어야 합니다")
        local: set[str] = set()
        for index, raw in enumerate(clips):
            clip = _nonempty_string(raw, field=f"families.{family}[{index}]")
            if Path(clip).name != clip or clip in {".", ".."}:
                raise HoldoutContractError(
                    f"families.{family}[{index}]는 경로가 아닌 원본 clip basename이어야 합니다"
                )
            key = clip.casefold()
            if key in local:
                raise HoldoutContractError(f"families.{family}에 중복 clip이 있습니다: {clip}")
            if key in global_clips:
                raise HoldoutContractError(
                    f"서로 다른 family에 같은 clip이 있습니다: {clip} "
                    f"({global_clips[key]}, {family})"
                )
            local.add(key)
            global_clips[key] = family
        canonical = tuple(sorted(local))
        if tuple(value.casefold() for value in clips) != canonical:
            raise HoldoutContractError(
                f"families.{family}는 casefold 기준으로 정렬된 canonical 배열이어야 합니다"
            )
        normalised[family] = canonical
        total += len(clips)

    declared = payload.get("total_clips")
    if isinstance(declared, bool) or not isinstance(declared, int) or declared != total:
        raise HoldoutContractError(
            f"total_clips 불일치: declared={declared!r}, families 합계={total}"
        )
    return total, normalised


def _repo_relative_posix(value: Any, *, field: str) -> str:
    raw = _nonempty_string(value, field=field)
    path = PurePosixPath(raw)
    if path.is_absolute() or "\\" in raw or any(part in {"", ".", ".."} for part in path.parts):
        raise HoldoutContractError(f"{field}는 정규화된 저장소 상대 POSIX 경로여야 합니다")
    return path.as_posix()


def _parse_source_csv_bytes(
    raw: bytes, *, relative_path: str
) -> dict[str, tuple[str, int, tuple[str, ...]]]:
    """한 source CSV 전체를 엄격히 읽어 row path -> (family, clips)를 만든다."""

    try:
        with io.StringIO(raw.decode("utf-8-sig"), newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "source_family",
                "session_index",
                "path",
                "sample_rate_hz",
                "clip_count",
                "clips",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise HoldoutContractError(
                    f"{relative_path} header가 canonical source-row 감사에 불완전합니다"
                )
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise HoldoutContractError(f"{relative_path} CSV를 읽을 수 없습니다: {exc}") from exc

    if len(rows) != EXPECTED_POOL_ROW_COUNT:
        raise HoldoutContractError(
            f"{relative_path} row 수 불일치: {len(rows)} != {EXPECTED_POOL_ROW_COUNT}"
        )
    expected_pool = PurePosixPath(relative_path).parent.name
    result: dict[str, tuple[str, int, tuple[str, ...]]] = {}
    identity: set[tuple[str, int]] = set()
    family_counts: Counter[str] = Counter()
    for number, row in enumerate(rows, start=2):
        field = f"{relative_path}:{number}"
        family = _nonempty_string(row.get("source_family"), field=f"{field}.source_family")
        if family not in REQUIRED_FAMILIES:
            raise HoldoutContractError(f"{field}.source_family가 canonical 네 계열 밖입니다")
        try:
            session_index = int(row.get("session_index", ""))
            sample_rate = int(row.get("sample_rate_hz", ""))
            clip_count = int(row.get("clip_count", ""))
        except (TypeError, ValueError) as exc:
            raise HoldoutContractError(f"{field} 정수 필드가 손상됐습니다") from exc
        if not 0 <= session_index < EXPECTED_POOL_FAMILY_ROWS:
            raise HoldoutContractError(f"{field}.session_index 범위가 0..19가 아닙니다")
        if sample_rate != EXPECTED_SAMPLE_RATE:
            raise HoldoutContractError(f"{field}.sample_rate_hz가 48000이 아닙니다")
        row_identity = (family, session_index)
        if row_identity in identity:
            raise HoldoutContractError(f"{relative_path}에 중복 row가 있습니다: {row_identity}")
        identity.add(row_identity)
        family_counts[family] += 1

        row_path = _repo_relative_posix(row.get("path"), field=f"{field}.path")
        expected_prefix = f"data/{expected_pool}/{family}/"
        if not row_path.startswith(expected_prefix):
            raise HoldoutContractError(
                f"{field}.path가 pool/family와 일치하지 않습니다: {row_path}"
            )
        if row_path in result:
            raise HoldoutContractError(f"source CSV에 중복 path가 있습니다: {row_path}")
        try:
            clips_value = json.loads(row.get("clips") or "")
        except json.JSONDecodeError as exc:
            raise HoldoutContractError(f"{field}.clips JSON이 손상됐습니다") from exc
        if (
            not isinstance(clips_value, list)
            or not clips_value
            or not all(isinstance(item, str) for item in clips_value)
        ):
            raise HoldoutContractError(f"{field}.clips는 비어 있지 않은 문자열 배열이어야 합니다")
        clips: list[str] = []
        for index, item in enumerate(clips_value):
            clip = _nonempty_string(item, field=f"{field}.clips[{index}]")
            if Path(clip).name != clip or clip in {".", ".."}:
                raise HoldoutContractError(f"{field}.clips[{index}]는 basename이어야 합니다")
            clips.append(clip.casefold())
        if len(clips) != clip_count:
            raise HoldoutContractError(
                f"{field}.clip_count 불일치: {clip_count} != {len(clips)}"
            )
        if len(set(clips)) != len(clips):
            raise HoldoutContractError(f"{field}.clips에 중복이 있습니다")
        result[row_path] = (family, session_index, tuple(clips))

    expected_families = {family: EXPECTED_POOL_FAMILY_ROWS for family in REQUIRED_FAMILIES}
    if dict(family_counts) != expected_families:
        raise HoldoutContractError(
            f"{relative_path} family별 row 수가 20개씩이 아닙니다: {dict(family_counts)}"
        )
    return result


def _validate_source_row_union(
    payload: dict[str, Any],
    *,
    csv_paths: list[str],
    parsed_csvs: dict[str, dict[str, tuple[str, int, tuple[str, ...]]]],
    declared_families: dict[str, tuple[str, ...]],
) -> tuple[
    dict[str, set[tuple[str, int, str]]],
    dict[str, tuple[str, int, tuple[str, ...]]],
]:
    all_rows: dict[str, tuple[str, int, tuple[str, ...]]] = {}
    identities_by_pool: dict[str, set[tuple[str, int, str]]] = {}
    for relative in csv_paths:
        parsed = parsed_csvs[relative]
        duplicate = set(all_rows) & set(parsed)
        if duplicate:
            raise HoldoutContractError(f"두 source CSV 사이에 중복 path가 있습니다: {sorted(duplicate)[:3]}")
        all_rows.update(parsed)
        pool_name = PurePosixPath(relative).parent.name
        identities_by_pool[pool_name] = {
            (family, session_index, row_path)
            for row_path, (family, session_index, _clips) in parsed.items()
        }

    source_rows = payload["source_rows"]
    selected: list[str] = [
        _repo_relative_posix(value, field=f"source_rows[{index}]")
        for index, value in enumerate(source_rows)
    ]
    if selected != sorted(selected) or len(set(selected)) != len(selected):
        raise HoldoutContractError("source_rows는 중복 없는 정렬된 canonical 배열이어야 합니다")
    unknown = [value for value in selected if value not in all_rows]
    if unknown:
        raise HoldoutContractError(
            f"source_rows가 두 canonical CSV에 없는 행을 참조합니다: {unknown[:3]}"
        )
    derived: dict[str, set[str]] = defaultdict(set)
    clip_owner: dict[str, str] = {}
    for row_path in selected:
        family, _session_index, clips = all_rows[row_path]
        for clip in clips:
            owner = clip_owner.setdefault(clip, family)
            if owner != family:
                raise HoldoutContractError(
                    f"선택 source_rows의 동일 clip이 서로 다른 family에 있습니다: {clip}"
                )
            derived[family].add(clip)
    derived_canonical = {
        family: tuple(sorted(derived.get(family, set()))) for family in sorted(REQUIRED_FAMILIES)
    }
    if derived_canonical != declared_families:
        raise HoldoutContractError(
            "families가 두 sources.csv의 source_rows→clips exact 합집합과 다릅니다"
        )
    return identities_by_pool, all_rows


def _validate_report_row_evidence(
    report: dict[str, Any],
    *,
    csv_hashes: dict[str, str],
    source_csv_identities: dict[str, set[tuple[str, int, str]]],
) -> None:
    """최소 PASS report가 통과하지 못하도록 160개 행의 full 증거를 감사한다."""

    selection = report.get("selection")
    if not isinstance(selection, dict):
        raise HoldoutContractError("provenance report selection 증거가 없습니다")
    if selection.get("seed") != 20260804:
        raise HoldoutContractError("provenance report selection.seed가 canonical 값이 아닙니다")
    for field, expected in EXPECTED_INVOCATIONS.items():
        if selection.get(field) != expected:
            raise HoldoutContractError(f"provenance report selection.{field}가 canonical 값이 아닙니다")
    if selection.get("v2_exclusion_semantics") != "historical v1 full plan의 used[:12] unique set":
        raise HoldoutContractError("provenance report v2 exclusion semantics가 canonical 값이 아닙니다")
    counts = selection.get("counts")
    declared_expected = selection.get("expected")
    if not isinstance(counts, dict) or not isinstance(declared_expected, dict):
        raise HoldoutContractError("provenance report selection count 증거가 없습니다")
    for field, expected in EXPECTED_SELECTION_COUNTS.items():
        if counts.get(field) != expected or declared_expected.get(field) != expected:
            raise HoldoutContractError(f"provenance report selection.{field} 기준선이 다릅니다")

    pools = report.get("pools")
    if not isinstance(pools, dict) or set(pools) != {"source_pool", "source_pool_v2"}:
        raise HoldoutContractError("provenance report pools는 source_pool/v2를 정확히 포함해야 합니다")
    for pool_name, builder_name in (("source_pool", "v1"), ("source_pool_v2", "v2")):
        pool = pools[pool_name]
        if not isinstance(pool, dict) or pool.get("status") != "PASS":
            raise HoldoutContractError(f"provenance report pools.{pool_name} 전체 PASS 증거가 없습니다")
        csv_audit = pool.get("csv")
        pcm_audit = pool.get("pcm")
        if not isinstance(csv_audit, dict) or not isinstance(pcm_audit, dict):
            raise HoldoutContractError(f"provenance report pools.{pool_name} row 증거가 없습니다")
        csv_rows = csv_audit.get("rows")
        pcm_rows = pcm_audit.get("rows")
        if (
            csv_audit.get("status") != "PASS"
            or csv_audit.get("row_count") != EXPECTED_POOL_ROW_COUNT
            or csv_audit.get("expected_row_count") != EXPECTED_POOL_ROW_COUNT
            or csv_audit.get("issues") != []
            or not isinstance(csv_rows, list)
            or len(csv_rows) != EXPECTED_POOL_ROW_COUNT
        ):
            raise HoldoutContractError(f"provenance report pools.{pool_name}.csv full audit가 불완전합니다")
        if (
            pcm_audit.get("status") != "PASS"
            or pcm_audit.get("passed_rows") != EXPECTED_POOL_ROW_COUNT
            or pcm_audit.get("expected_rows") != EXPECTED_POOL_ROW_COUNT
            or not isinstance(pcm_rows, list)
            or len(pcm_rows) != EXPECTED_POOL_ROW_COUNT
        ):
            raise HoldoutContractError(f"provenance report pools.{pool_name}.pcm full audit가 불완전합니다")

        expected_commit = EXPECTED_HISTORICAL_BUILDERS[builder_name]["commit"]
        csv_identities: set[tuple[str, int, str]] = set()
        pcm_identities: set[tuple[str, int, str]] = set()
        for index, row in enumerate(csv_rows):
            if not isinstance(row, dict):
                raise HoldoutContractError(f"pools.{pool_name}.csv.rows[{index}]가 object가 아닙니다")
            session_value = row.get("session_index")
            clip_numbers = (
                row.get("declared_clips"),
                row.get("reconstructed_clips"),
                row.get("missing_clips"),
            )
            try:
                identity = (
                    str(row["family"]),
                    int(session_value),
                    _repo_relative_posix(row["path"], field=f"pools.{pool_name}.csv.rows[{index}].path"),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise HoldoutContractError(f"pools.{pool_name}.csv.rows[{index}] identity 오류") from exc
            if (
                identity[0] not in REQUIRED_FAMILIES
                or isinstance(session_value, bool)
                or not isinstance(session_value, int)
                or not 0 <= identity[1] < EXPECTED_POOL_FAMILY_ROWS
                or row.get("prefix_pass") is not True
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in clip_numbers
                )
                or row["declared_clips"] + row["missing_clips"] != row["reconstructed_clips"]
            ):
                raise HoldoutContractError(f"pools.{pool_name}.csv.rows[{index}] audit가 불완전합니다")
            csv_identities.add(identity)
        for index, row in enumerate(pcm_rows):
            if not isinstance(row, dict):
                raise HoldoutContractError(f"pools.{pool_name}.pcm.rows[{index}]가 object가 아닙니다")
            session_value = row.get("session_index")
            try:
                identity = (
                    str(row["family"]),
                    int(session_value),
                    _repo_relative_posix(row["path"], field=f"pools.{pool_name}.pcm.rows[{index}].path"),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise HoldoutContractError(f"pools.{pool_name}.pcm.rows[{index}] identity 오류") from exc
            pcm = row.get("pcm")
            if (
                row.get("status") != "PASS"
                or isinstance(session_value, bool)
                or not isinstance(session_value, int)
                or row.get("builder_commit") != expected_commit
                or row.get("sample_rate_hz") != EXPECTED_SAMPLE_RATE
                or isinstance(row.get("channels"), bool)
                or row.get("channels") != 1
                or row.get("frames") != EXPECTED_FRAMES
                or not isinstance(row.get("wav_sha256"), str)
                or not _SHA256_RE.fullmatch(row["wav_sha256"])
                or not isinstance(pcm, dict)
                or pcm.get("status") != "PASS"
                or pcm.get("shape_match") is not True
                or pcm.get("expected_shape") != [EXPECTED_FRAMES]
                or pcm.get("actual_shape") != [EXPECTED_FRAMES]
            ):
                raise HoldoutContractError(f"pools.{pool_name}.pcm.rows[{index}] 증거가 불완전합니다")
            pcm_identities.add(identity)
        if len(csv_identities) != EXPECTED_POOL_ROW_COUNT or csv_identities != pcm_identities:
            raise HoldoutContractError(f"pools.{pool_name} CSV/PCM row identity 집합이 다릅니다")
        if csv_identities != source_csv_identities.get(pool_name):
            raise HoldoutContractError(
                f"pools.{pool_name} full row 증거가 실제 sources.csv identity와 다릅니다"
            )
        identities_by_family = Counter(item[0] for item in csv_identities)
        if identities_by_family != Counter(
            {family: EXPECTED_POOL_FAMILY_ROWS for family in REQUIRED_FAMILIES}
        ):
            raise HoldoutContractError(f"pools.{pool_name} family별 full row 증거가 20개씩이 아닙니다")

        repair = report.get("repair")
        repair_files = repair.get("files") if isinstance(repair, dict) else None
        repair_item = repair_files.get(pool_name) if isinstance(repair_files, dict) else None
        if (
            not isinstance(repair, dict)
            or not isinstance(repair_files, dict)
            or report.get("mode") != "repair"
            or repair.get("requested") is not True
            or repair.get("performed") is not True
            or not isinstance(repair_item, dict)
            or repair_item.get("after_sha256") != csv_hashes[pool_name]
        ):
            raise HoldoutContractError(f"provenance report repair.{pool_name} 증거가 불완전합니다")


def _parse_fma_lineage_bytes(raw: bytes) -> dict[int, tuple[str, str]]:
    try:
        handle = io.StringIO(raw.decode("utf-8"), newline="")
    except UnicodeDecodeError as exc:
        raise HoldoutContractError("canonical FMA tracks.csv UTF-8 오류") from exc
    with handle:
        reader = csv.reader(handle)
        try:
            level0 = next(reader)
            level1 = next(reader)
        except StopIteration as exc:
            raise HoldoutContractError("canonical FMA tracks.csv header가 불완전합니다") from exc
        width = max(len(level0), len(level1))
        level0 += [""] * (width - len(level0))
        level1 += [""] * (width - len(level1))

        def column(group: str, field: str) -> int:
            hits = [
                index
                for index, (left, right) in enumerate(zip(level0, level1))
                if left.strip().casefold() == group and right.strip().casefold() == field
            ]
            if len(hits) != 1:
                raise HoldoutContractError(
                    f"canonical FMA tracks.csv {group}/{field} column이 유일하지 않습니다"
                )
            return hits[0]

        artist_col = column("artist", "id")
        album_col = column("album", "id")
        mapping: dict[int, tuple[str, str]] = {}
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue
            if max(artist_col, album_col) >= len(row):
                raise HoldoutContractError("canonical FMA tracks.csv row column 누락")
            artist, album = row[artist_col].strip(), row[album_col].strip()
            if not artist or not album:
                raise HoldoutContractError("canonical FMA tracks.csv artist/album ID 누락")
            mapping[int(row[0])] = (artist, album)
    return mapping


def _session_source_wav(value: Any, *, field: str) -> str:
    raw = _nonempty_string(value, field=field).replace("\\", "/")
    for prefix in ("data/source_pool/", "data/source_pool_v2/"):
        position = raw.find(prefix)
        if position >= 0:
            return PurePosixPath(raw[position:]).as_posix()
    return _repo_relative_posix(raw, field=field)


def _derive_lineage_components(
    *,
    repo_root: Path,
    source_rows_by_path: dict[str, tuple[str, int, tuple[str, ...]]],
    selected_source_rows: set[str],
    fma_tracks: dict[int, tuple[str, str]],
    librispeech_chapters: dict[int, tuple[int, int]],
    esc50_metadata: dict[str, str],
    active_session_count: int,
) -> dict[str, list[str]]:
    # module import cycle을 피하기 위해 holdout module 초기화가 끝난 뒤 가져온다.
    lineage_api = _public_lineage_module()

    recorded_root = repo_root / "data/recorded"
    reject_symlink_components(recorded_root, root=repo_root)
    if not recorded_root.is_dir():
        raise HoldoutContractError(f"active recorded root가 없습니다: {recorded_root}")
    sessions: dict[str, tuple[str, tuple[str, ...]]] = {}
    observed_source_rows: set[str] = set()
    for directory in sorted(recorded_root.iterdir()):
        info = directory.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise HoldoutContractError(f"recorded session symlink 금지: {directory}")
        if not stat.S_ISDIR(info.st_mode):
            if directory.name == "batch_progress.csv" and stat.S_ISREG(info.st_mode):
                continue
            raise HoldoutContractError(
                f"recorded root에는 session directory와 batch_progress.csv만 허용합니다: {directory}"
            )
        metadata_snapshot = read_regular_file_snapshot(
            directory / "session.json",
            root=repo_root,
            label=f"recorded session metadata {directory.name}",
        )
        assert metadata_snapshot.data is not None
        metadata = _load_json_bytes(
            metadata_snapshot.data,
            path=metadata_snapshot.path,
            label="recorded session metadata",
        )
        session_id = _nonempty_string(
            metadata.get("session_id") or directory.name,
            field=f"{directory.name}.session_id",
        )
        if session_id in sessions:
            raise HoldoutContractError(f"active recorded session_id 중복: {session_id}")
        program = metadata.get("program")
        if not isinstance(program, dict) or program.get("type") != "file":
            raise HoldoutContractError(f"active session program.file 누락: {session_id}")
        source_wav = _session_source_wav(
            program.get("file"), field=f"{session_id}.program.file"
        )
        if source_wav not in selected_source_rows or source_wav not in source_rows_by_path:
            raise HoldoutContractError(
                f"active session source가 canonical selected source_rows 밖입니다: {session_id} -> {source_wav}"
            )
        family, _session_index, clips = source_rows_by_path[source_wav]
        declared_family = metadata.get("source_family")
        if declared_family not in (None, "", family):
            raise HoldoutContractError(f"active session source_family 불일치: {session_id}")
        sessions[session_id] = (family, clips)
        observed_source_rows.add(source_wav)
    if len(sessions) != active_session_count or observed_source_rows != selected_source_rows:
        raise HoldoutContractError(
            "active recorded sessions/source_rows exact 집합이 holdout count와 다릅니다"
        )

    parent = {session: session for session in sessions}

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
    for session_id, (family, clips) in sorted(sessions.items()):
        for clip in clips:
            keys: list[tuple[str, str]] = [("clip", clip.casefold())]
            if family == "music":
                stem = Path(clip).stem
                if not stem.isdigit() or int(stem) not in fma_tracks:
                    raise HoldoutContractError(
                        f"active music clip의 FMA metadata mapping 누락: {clip}"
                    )
                artist, album = fma_tracks[int(stem)]
                keys.extend((("music_artist", artist), ("music_album", album)))
            elif family == "speech":
                try:
                    reader_key, book_key = lineage_api.librispeech_lineage_keys(
                        clip, librispeech_chapters
                    )
                except ValueError as exc:
                    raise HoldoutContractError(str(exc)) from exc
                keys.extend(
                    (
                        ("speech_speaker", reader_key.split(":", 1)[1]),
                        ("speech_book", book_key.split(":", 1)[1]),
                    )
                )
            elif family in {"environment", "machine"}:
                try:
                    source_key = lineage_api.esc50_lineage_keys(clip, esc50_metadata)[0]
                except ValueError as exc:
                    raise HoldoutContractError(str(exc)) from exc
                keys.append(("esc50_src_file", source_key.split(":", 1)[1]))
            for key in keys:
                previous = owner.setdefault(key, session_id)
                union(previous, session_id)
    raw_components: dict[str, list[str]] = defaultdict(list)
    for session_id in sorted(sessions):
        raw_components[find(session_id)].append(session_id)
    components: dict[str, list[str]] = {}
    for members in sorted(raw_components.values()):
        families = {sessions[session][0] for session in members}
        if len(families) != 1:
            raise HoldoutContractError("derived lineage component가 family를 가로지릅니다")
        family = next(iter(families))
        component = (
            f"{family}-lineage-"
            + hashlib.sha256("\n".join(members).encode()).hexdigest()[:12]
        )
        components[component] = members
    return components


def _validate_lineage_contract(
    report: dict[str, Any],
    *,
    repo_root: Path,
    active_session_count: int,
    source_rows_by_path: dict[str, tuple[str, int, tuple[str, ...]]],
    selected_source_rows: set[str],
) -> dict[str, Any]:
    lineage = report.get("lineage_contract")
    required_keys = {
        "schema_version",
        "tracks_csv",
        "tracks_csv_sha256",
        "librispeech_chapters_path",
        "librispeech_chapters_sha256",
        "esc50_metadata_path",
        "esc50_metadata_sha256",
        "holdout_clip_lineage_sha256",
        "active_session_count",
        "component_count",
        "component_count_by_family",
        "components",
        "component_membership_sha256",
        "regrouped_manifest",
        "regrouped_manifest_sha256",
        "regrouped_row_count",
        "regrouped_component_count",
        "groups_by_family_split",
        "lineage_cross_split_count",
        "source_clip_cross_split_count",
    }
    if not isinstance(lineage, dict) or set(lineage) != required_keys:
        raise HoldoutContractError("provenance report lineage_contract 구조가 불완전합니다")
    if lineage.get("schema_version") != 2:
        raise HoldoutContractError("lineage_contract schema_version은 2여야 합니다")
    if (
        lineage.get("tracks_csv") != FMA_TRACKS_CSV
        or lineage.get("tracks_csv_sha256") != FMA_TRACKS_CSV_SHA256
    ):
        raise HoldoutContractError("lineage_contract FMA tracks.csv path/SHA가 canonical 값이 아닙니다")
    tracks = read_regular_file_snapshot(
        repo_root / FMA_TRACKS_CSV,
        root=repo_root,
        label="canonical FMA tracks.csv",
    )
    if tracks.sha256 != FMA_TRACKS_CSV_SHA256:
        raise HoldoutContractError(
            f"canonical FMA tracks.csv SHA 불일치: {tracks.sha256}"
        )
    assert tracks.data is not None
    if lineage.get("librispeech_chapters_path") != LIBRISPEECH_CHAPTERS:
        raise HoldoutContractError("lineage_contract LibriSpeech CHAPTERS 경로가 canonical 값이 아닙니다")
    chapters = read_regular_file_snapshot(
        repo_root / LIBRISPEECH_CHAPTERS,
        root=repo_root,
        label="canonical LibriSpeech CHAPTERS.TXT",
    )
    if chapters.sha256 != _sha256(
        lineage.get("librispeech_chapters_sha256"),
        field="lineage_contract.librispeech_chapters_sha256",
    ):
        raise HoldoutContractError("canonical LibriSpeech CHAPTERS.TXT SHA 불일치")
    if lineage.get("esc50_metadata_path") != ESC50_METADATA:
        raise HoldoutContractError("lineage_contract ESC-50 metadata 경로가 canonical 값이 아닙니다")
    esc50 = read_regular_file_snapshot(
        repo_root / ESC50_METADATA,
        root=repo_root,
        label="canonical ESC-50 metadata",
    )
    if esc50.sha256 != _sha256(
        lineage.get("esc50_metadata_sha256"),
        field="lineage_contract.esc50_metadata_sha256",
    ):
        raise HoldoutContractError("canonical ESC-50 metadata SHA 불일치")
    assert chapters.data is not None
    assert esc50.data is not None
    try:
        lineage_api = _public_lineage_module()
        parsed_chapters = lineage_api.parse_librispeech_chapters_bytes(chapters.data)
        parsed_esc50 = lineage_api.parse_esc50_metadata_bytes(esc50.data)
    except ValueError as exc:
        raise HoldoutContractError(str(exc)) from exc

    components = lineage.get("components")
    if not isinstance(components, dict) or not components:
        raise HoldoutContractError("lineage_contract components가 비었습니다")
    normalised_components: dict[str, list[str]] = {}
    session_to_component: dict[str, str] = {}
    family_counts: Counter[str] = Counter()
    for component, members in sorted(components.items()):
        if (
            not isinstance(component, str)
            or "-lineage-" not in component
            or not isinstance(members, list)
            or not members
            or members != sorted(members)
            or len(set(members)) != len(members)
            or not all(isinstance(value, str) and value for value in members)
        ):
            raise HoldoutContractError(f"lineage component가 canonical 형식이 아닙니다: {component!r}")
        family = component.split("-lineage-", 1)[0]
        if family not in REQUIRED_FAMILIES:
            raise HoldoutContractError(f"lineage component family가 canonical 네 계열 밖입니다: {component}")
        family_counts[family] += 1
        normalised_components[component] = list(members)
        for session in members:
            if session in session_to_component:
                raise HoldoutContractError(f"lineage session이 여러 component에 있습니다: {session}")
            session_to_component[session] = component
    membership_bytes = json.dumps(
        normalised_components,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    membership_sha = hashlib.sha256(membership_bytes).hexdigest()
    if lineage.get("component_membership_sha256") != membership_sha:
        raise HoldoutContractError("lineage component membership digest 불일치")
    derived_components = _derive_lineage_components(
        repo_root=repo_root,
        source_rows_by_path=source_rows_by_path,
        selected_source_rows=selected_source_rows,
        fma_tracks=_parse_fma_lineage_bytes(tracks.data),
        librispeech_chapters=parsed_chapters,
        esc50_metadata=parsed_esc50,
        active_session_count=active_session_count,
    )
    if normalised_components != derived_components:
        raise HoldoutContractError(
            "lineage component membership이 active sessions + source CSV + FMA metadata "
            "재계산과 다릅니다"
        )
    if (
        lineage.get("active_session_count") != active_session_count
        or len(session_to_component) != active_session_count
        or lineage.get("component_count") != len(normalised_components)
        or lineage.get("component_count_by_family") != dict(family_counts)
    ):
        raise HoldoutContractError("lineage component/session/count 증거가 실제 membership과 다릅니다")

    if lineage.get("regrouped_manifest") != RECORDED_REGROUPED_MANIFEST:
        raise HoldoutContractError("lineage regrouped_manifest 경로가 canonical 값이 아닙니다")
    declared_manifest_sha = _sha256(
        lineage.get("regrouped_manifest_sha256"),
        field="lineage_contract.regrouped_manifest_sha256",
    )
    regrouped = read_regular_file_snapshot(
        repo_root / RECORDED_REGROUPED_MANIFEST,
        root=repo_root,
        label="canonical recorded_regrouped manifest",
    )
    if regrouped.sha256 != declared_manifest_sha:
        raise HoldoutContractError("canonical recorded_regrouped manifest SHA 불일치")
    assert regrouped.data is not None
    try:
        lines = regrouped.data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise HoldoutContractError("recorded_regrouped manifest UTF-8 오류") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_object_without_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise HoldoutContractError(
                f"recorded_regrouped manifest JSON 오류 line={number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise HoldoutContractError(f"recorded_regrouped row #{number}가 object가 아닙니다")
        rows.append(row)
    if lineage.get("regrouped_row_count") != len(rows) or len(rows) != active_session_count:
        raise HoldoutContractError("recorded_regrouped row count가 active session과 다릅니다")
    seen_sessions: set[str] = set()
    component_splits: dict[str, set[str]] = defaultdict(set)
    computed_groups: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for index, row in enumerate(rows):
        session = row.get("session_id")
        component = row.get("group_id")
        family = row.get("source_family")
        split = row.get("split")
        source_group = row.get("source_pool_group_id")
        if (
            not isinstance(session, str)
            or session in seen_sessions
            or session_to_component.get(session) != component
            or not isinstance(component, str)
            or not isinstance(source_group, str)
            or not source_group
            or source_group == component
            or family != component.split("-lineage-", 1)[0]
            or split not in {"train", "val", "test"}
            or row.get("lineage_schema")
            != "shared_clip+music_artist_album+speech_reader_gutenberg_book/v2"
        ):
            raise HoldoutContractError(
                f"recorded_regrouped row #{index}가 canonical component mapping과 다릅니다"
            )
        seen_sessions.add(session)
        component_splits[component].add(split)
        computed_groups[str(family)][str(split)].add(component)
    if seen_sessions != set(session_to_component):
        raise HoldoutContractError("recorded_regrouped session exact 집합이 component membership과 다릅니다")
    crossings = [name for name, splits in component_splits.items() if len(splits) != 1]
    if crossings or lineage.get("lineage_cross_split_count") != 0:
        raise HoldoutContractError(f"lineage component가 split을 가로지릅니다: {crossings[:5]}")
    groups_summary = {
        family: {
            split: len(computed_groups.get(family, {}).get(split, set()))
            for split in ("train", "val", "test")
        }
        for family in sorted(family_counts)
    }
    if lineage.get("groups_by_family_split") != groups_summary:
        raise HoldoutContractError("lineage groups_by_family_split 재계산 불일치")
    if (
        lineage.get("regrouped_component_count") != len(component_splits)
        or lineage.get("source_clip_cross_split_count") != 0
    ):
        raise HoldoutContractError("lineage regrouped component/crossing 증거 불일치")
    return {
        "tracks_csv_sha256": tracks.sha256,
        "tracks_csv_size": tracks.size,
        "librispeech_chapters_path": LIBRISPEECH_CHAPTERS,
        "librispeech_chapters_sha256": chapters.sha256,
        "librispeech_chapters_size": chapters.size,
        "esc50_metadata_path": ESC50_METADATA,
        "esc50_metadata_sha256": esc50.sha256,
        "esc50_metadata_size": esc50.size,
        "holdout_clip_lineage_sha256": _sha256(
            lineage.get("holdout_clip_lineage_sha256"),
            field="lineage_contract.holdout_clip_lineage_sha256",
        ),
        "component_membership_sha256": membership_sha,
        "component_count": len(normalised_components),
        "regrouped_manifest": RECORDED_REGROUPED_MANIFEST,
        "regrouped_manifest_sha256": regrouped.sha256,
        "regrouped_row_count": len(rows),
    }


def _validate_build_provenance(
    payload: dict[str, Any],
    *,
    repo_root: Path,
    total_clips: int,
    declared_families: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    if payload.get("scope") != "active_sessions_only":
        raise HoldoutContractError("scope는 active_sessions_only여야 합니다")
    _nonempty_string(payload.get("purpose"), field="purpose")
    session_count = _positive_int(
        payload.get("active_session_count"), field="active_session_count"
    )
    row_count = _positive_int(
        payload.get("active_source_row_count"), field="active_source_row_count"
    )
    source_rows = payload.get("source_rows")
    if not isinstance(source_rows, list) or len(source_rows) != row_count:
        raise HoldoutContractError(
            "source_rows 길이는 active_source_row_count와 같아야 합니다"
        )
    normalised_rows = [
        _repo_relative_posix(item, field=f"source_rows[{index}]")
        for index, item in enumerate(source_rows)
    ]
    if len(set(normalised_rows)) != len(normalised_rows):
        raise HoldoutContractError("source_rows에 중복이 있습니다")

    sources_csv = payload.get("sources_csv")
    if not isinstance(sources_csv, list) or len(sources_csv) != 2:
        raise HoldoutContractError("sources_csv는 v1/v2 두 경로를 포함해야 합니다")
    csv_paths = [
        _repo_relative_posix(item, field=f"sources_csv[{index}]")
        for index, item in enumerate(sources_csv)
    ]
    if tuple(csv_paths) != EXPECTED_SOURCES_CSV:
        raise HoldoutContractError(
            "sources_csv는 canonical v1/v2 두 경로를 고정 순서로 포함해야 합니다"
        )
    expected_csv_keys = {Path(value).parent.name for value in csv_paths}
    csv_hashes = payload.get("sources_csv_sha256")
    if not isinstance(csv_hashes, dict) or set(csv_hashes) != expected_csv_keys:
        raise HoldoutContractError(
            "sources_csv_sha256 키는 sources_csv의 두 source-pool 이름과 일치해야 합니다"
        )
    for key, value in csv_hashes.items():
        _sha256(value, field=f"sources_csv_sha256.{key}")
    parsed_csvs: dict[str, dict[str, tuple[str, int, tuple[str, ...]]]] = {}
    for index, value in enumerate(csv_paths):
        csv_path = _repo_relative_file(repo_root, value, field=f"sources_csv[{index}]")
        key = Path(value).parent.name
        csv_snapshot = read_regular_file_snapshot(
            csv_path,
            root=repo_root,
            label=f"sources_csv[{index}]",
        )
        if csv_snapshot.sha256 != csv_hashes[key]:
            raise HoldoutContractError(
                f"{value} SHA-256 불일치: "
                f"declared={csv_hashes[key]}, actual={csv_snapshot.sha256}"
            )
        assert csv_snapshot.data is not None
        parsed_csvs[value] = _parse_source_csv_bytes(
            csv_snapshot.data, relative_path=value
        )

    source_csv_identities, source_rows_by_path = _validate_source_row_union(
        payload,
        csv_paths=csv_paths,
        parsed_csvs=parsed_csvs,
        declared_families=declared_families,
    )

    declared_report_sha256 = _sha256(
        payload.get("provenance_report_sha256"), field="provenance_report_sha256"
    )
    report_relative = _repo_relative_posix(
        payload.get("provenance_report"), field="provenance_report"
    )
    expected_report = (
        "results/provenance/"
        f"source_pool_provenance_report.{declared_report_sha256}.json"
    )
    if report_relative != expected_report:
        raise HoldoutContractError(
            "provenance_report는 report bytes SHA가 파일명에 들어간 canonical immutable "
            f"경로여야 합니다: expected={expected_report!r}, actual={report_relative!r}"
        )
    report_path = _repo_relative_file(
        repo_root, report_relative, field="provenance_report"
    )
    report_snapshot = read_regular_file_snapshot(
        report_path,
        root=repo_root,
        label="provenance report",
    )
    if report_snapshot.sha256 != declared_report_sha256:
        raise HoldoutContractError(
            "provenance report SHA-256 불일치: "
            f"declared={declared_report_sha256}, actual={report_snapshot.sha256}"
        )
    assert report_snapshot.data is not None
    report = _load_json_bytes(
        report_snapshot.data,
        path=report_path,
        label="provenance report",
    )
    if report.get("schema_version") != 1:
        raise HoldoutContractError("provenance report schema_version은 1이어야 합니다")
    if report.get("authority") != PROVENANCE_AUTHORITY:
        raise HoldoutContractError("provenance report authority가 canonical 값이 아닙니다")
    if report.get("status") != "PASS":
        raise HoldoutContractError("provenance report 전체 status가 PASS가 아닙니다")
    _validate_report_row_evidence(
        report,
        csv_hashes=csv_hashes,
        source_csv_identities=source_csv_identities,
    )
    tree = report.get("recorded_tree_protection")
    expected_tree_keys = {
        "schema_version",
        "status",
        "root",
        "file_count",
        "snapshot_encoding",
        "before_sha256",
        "after_sha256",
        "content_snapshot_encoding",
        "before_content_sha256",
        "after_content_sha256",
        "unchanged",
        "content_integrity_boundary",
    }
    if (
        not isinstance(tree, dict)
        or set(tree) != expected_tree_keys
        or tree.get("schema_version") != 1
        or tree.get("status") != "PASS"
        or tree.get("root") != "data/recorded"
        or tree.get("snapshot_encoding") != RECORDED_TREE_SNAPSHOT_ENCODING
        or tree.get("content_snapshot_encoding")
        != RECORDED_TREE_CONTENT_SNAPSHOT_ENCODING
        or tree.get("content_integrity_boundary")
        != RECORDED_CONTENT_INTEGRITY_BOUNDARY
        or tree.get("unchanged") is not True
    ):
        raise HoldoutContractError("recorded_tree_protection 증거가 PASS가 아닙니다")
    tree_file_count = _positive_int(
        tree.get("file_count"), field="recorded_tree_protection.file_count"
    )
    tree_before_sha = _sha256(
        tree.get("before_sha256"),
        field="recorded_tree_protection.before_sha256",
    )
    tree_after_sha = _sha256(
        tree.get("after_sha256"),
        field="recorded_tree_protection.after_sha256",
    )
    if tree_before_sha != tree_after_sha:
        raise HoldoutContractError(
            "recorded_tree_protection before/after metadata digest가 다릅니다"
        )
    tree_before_content_sha = _sha256(
        tree.get("before_content_sha256"),
        field="recorded_tree_protection.before_content_sha256",
    )
    tree_after_content_sha = _sha256(
        tree.get("after_content_sha256"),
        field="recorded_tree_protection.after_content_sha256",
    )
    if tree_before_content_sha != tree_after_content_sha:
        raise HoldoutContractError(
            "recorded_tree_protection before/after content digest가 다릅니다"
        )

    builders = report.get("historical_builders")
    if not isinstance(builders, dict) or set(builders) != {"v1", "v2"}:
        raise HoldoutContractError("historical_builders는 v1/v2를 정확히 포함해야 합니다")
    for name in ("v1", "v2"):
        builder = builders[name]
        if not isinstance(builder, dict):
            raise HoldoutContractError(f"historical_builders.{name}가 object가 아닙니다")
        commit = builder.get("commit")
        if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
            raise HoldoutContractError(
                f"historical_builders.{name}.commit은 소문자 40자리 SHA여야 합니다"
            )
        _sha256(
            builder.get("source_sha256"),
            field=f"historical_builders.{name}.source_sha256",
        )
        _nonempty_string(builder.get("path"), field=f"historical_builders.{name}.path")
        expected_builder = EXPECTED_HISTORICAL_BUILDERS[name]
        for field, expected in expected_builder.items():
            if builder.get(field) != expected:
                raise HoldoutContractError(
                    f"historical_builders.{name}.{field}가 고정된 canonical 값과 다릅니다"
                )

    post_hashes = report.get("post_repair_csv_sha256")
    if post_hashes != csv_hashes:
        raise HoldoutContractError(
            "holdout sources_csv_sha256와 provenance report post_repair_csv_sha256가 다릅니다"
        )
    gates = report.get("downstream_gates")
    active = gates.get("active_holdout") if isinstance(gates, dict) else None
    if not isinstance(active, dict) or active.get("status") != "PASS":
        raise HoldoutContractError("provenance report active_holdout gate가 PASS가 아닙니다")
    expected_active = {
        "active_session_count": session_count,
        "active_source_row_count": row_count,
        "total_clips": total_clips,
    }
    for field, expected in expected_active.items():
        if active.get(field) != expected:
            raise HoldoutContractError(
                f"provenance report active_holdout.{field} 불일치: "
                f"report={active.get(field)!r}, holdout={expected!r}"
            )
    clip_lineage = payload.get("clip_lineage")
    try:
        canonical_clip_rows = _public_lineage_module().validate_recorded_clip_lineage(
            clip_lineage if isinstance(clip_lineage, dict) else {},
            families=declared_families,
        )
    except ValueError as exc:
        raise HoldoutContractError(str(exc)) from exc
    if (
        active.get("clip_lineage_sha256") != clip_lineage.get("clips_sha256")
        or active.get("clip_lineage_metadata") != clip_lineage.get("metadata")
    ):
        raise HoldoutContractError(
            "holdout clip_lineage와 provenance active_holdout 증거가 다릅니다"
        )
    lineage_summary = _validate_lineage_contract(
        report,
        repo_root=repo_root,
        active_session_count=session_count,
        source_rows_by_path=source_rows_by_path,
        selected_source_rows=set(normalised_rows),
    )
    metadata = clip_lineage["metadata"]
    if (
        lineage_summary["holdout_clip_lineage_sha256"]
        != clip_lineage["clips_sha256"]
        or metadata["fma_tracks"]["sha256"] != lineage_summary["tracks_csv_sha256"]
        or metadata["fma_tracks"]["size"] != lineage_summary["tracks_csv_size"]
        or metadata["librispeech_chapters"]["sha256"]
        != lineage_summary["librispeech_chapters_sha256"]
        or metadata["librispeech_chapters"]["size"]
        != lineage_summary["librispeech_chapters_size"]
        or metadata["esc50"]["sha256"]
        != lineage_summary["esc50_metadata_sha256"]
        or metadata["esc50"]["size"] != lineage_summary["esc50_metadata_size"]
    ):
        raise HoldoutContractError(
            "holdout clip_lineage metadata/SHA와 lineage_contract가 다릅니다"
        )
    return {
        "lineage": lineage_summary,
        "clip_lineage": {
            "schema_version": 1,
            "metadata": json.loads(json.dumps(metadata)),
            "clips": canonical_clip_rows,
            "clips_sha256": clip_lineage["clips_sha256"],
        },
        "recorded_tree": {
            "file_count": tree_file_count,
            "metadata_snapshot_sha256": tree_before_sha,
            "content_snapshot_sha256": tree_before_content_sha,
            "snapshot_encoding": RECORDED_TREE_SNAPSHOT_ENCODING,
            "content_snapshot_encoding": RECORDED_TREE_CONTENT_SNAPSHOT_ENCODING,
            "content_integrity_boundary": RECORDED_CONTENT_INTEGRITY_BOUNDARY,
        },
    }


def validate_holdout_contract(
    path: str | Path,
    *,
    repo_root: str | Path,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Canonical holdout의 byte hash, schema, provenance/build 증거를 검증한다."""

    contract_root = _lexical_absolute(Path(repo_root))
    holdout_snapshot = read_regular_file_snapshot(
        path,
        root=contract_root,
        label="canonical recorded holdout",
    )
    holdout_path = holdout_snapshot.path
    actual_sha256 = holdout_snapshot.sha256
    if expected_sha256 is not None:
        expected = _sha256(expected_sha256, field="expected_sha256")
        if actual_sha256 != expected:
            raise HoldoutContractError(
                f"canonical recorded holdout SHA-256 불일치: "
                f"expected={expected}, actual={actual_sha256}"
            )
    assert holdout_snapshot.data is not None
    payload = _load_json_bytes(
        holdout_snapshot.data,
        path=holdout_path,
        label="canonical recorded holdout",
    )
    total_clips, declared_families = _validate_families(payload)
    provenance_summary = _validate_build_provenance(
        payload,
        repo_root=contract_root,
        total_clips=total_clips,
        declared_families=declared_families,
    )
    return {
        "sha256": actual_sha256,
        "active_session_count": payload["active_session_count"],
        "active_source_row_count": payload["active_source_row_count"],
        "total_clips": total_clips,
        "family_counts": {
            name: len(payload["families"][name]) for name in sorted(REQUIRED_FAMILIES)
        },
        # downstream exclusion은 pathname을 다시 열지 않고 이 검증된 동일 bytes
        # snapshot만 소비해야 한다.
        "families": {
            name: list(payload["families"][name]) for name in sorted(REQUIRED_FAMILIES)
        },
        "_validated_holdout_bytes": holdout_snapshot.data,
        "_validated_holdout_stat": holdout_snapshot.stat_contract(),
        "provenance_report": payload["provenance_report"],
        "provenance_report_sha256": payload["provenance_report_sha256"],
        "sources_csv": list(payload["sources_csv"]),
        "sources_csv_sha256": dict(payload["sources_csv_sha256"]),
        "lineage": provenance_summary["lineage"],
        "clip_lineage": provenance_summary["clip_lineage"],
        "recorded_tree": provenance_summary["recorded_tree"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = validate_holdout_contract(
            args.path,
            repo_root=args.repo_root,
            expected_sha256=args.expected_sha256.lower(),
        )
    except HoldoutContractError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 1
    print(
        "[holdout] canonical 계약 확인: "
        f"sha256={summary['sha256']}, "
        f"sessions={summary['active_session_count']}, "
        f"source_rows={summary['active_source_row_count']}, "
        f"clips={summary['total_clips']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
