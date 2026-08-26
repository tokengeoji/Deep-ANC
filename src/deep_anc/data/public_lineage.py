"""Public corpus 원본 계보와 component 원자 분할 계약.

경로 basename만 비교하면 같은 원본의 rename/copy와 같은 speaker/book, artist/album,
Freesound source의 다른 take를 놓친다. 이 모듈은 코퍼스가 제공한 권위 metadata와 raw
content SHA를 한 번에 결속하고, 그 키들의 transitive closure를 ``group_id``로 만든다.

DNS ``read_speech``의 파일명에는 reader/book 식별자가 있지만 그 ID를 LibriSpeech의
LibriVox/Gutenberg ID로 연결하는 공식 crosswalk는 현재 저장소에 없다. 두 namespace를
같다고 추측하지 않는다. recorded speech holdout과 DNS speech를 함께 쓰는 학습 세대는
crosswalk가 권위 자료로 추가되기 전까지 :class:`PublicLineageBlocked`로 차단한다.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .holdout_contract import FileSnapshot, read_regular_file_snapshot


PUBLIC_LINEAGE_SCHEMA = "public-corpus-lineage/v1"
LIBRISPEECH_CHAPTERS = "data/raw/speech/LibriSpeech/CHAPTERS.TXT"
FMA_TRACKS = "data/raw/music/fma_metadata/tracks.csv"
ESC50_METADATA = "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv"
DNS_SPEECH_MARKER = "data/raw/noise/speech000.tar.bz2.extracted"
DNS_NOISE_MARKERS = (
    "data/raw/noise/shard000.tar.bz2.extracted",
    "data/raw/noise/shard001.tar.bz2.extracted",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GROUP_ID_RE = re.compile(r"^public-lineage-[0-9a-f]{64}$")
_DNS_SPEECH_RE = re.compile(
    r"^book_(?P<book>\d+)_chp_(?P<chapter>\d+)_reader_(?P<reader>\d+)"
    r"(?:_[A-Za-z0-9]+)*$",
    re.IGNORECASE,
)
_AUDIOSET_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_MIMII_ID_RE = re.compile(r"^id_(\d+)$", re.IGNORECASE)
_DEMAND_ENVIRONMENTS = frozenset(
    {"DKITCHEN", "DWASHING", "OOFFICE", "OHALLWAY", "TMETRO", "TCAR"}
)


class PublicLineageError(ValueError):
    """계보 metadata가 손상되었거나 raw tree와 불일치한다."""


class PublicLineageBlocked(PublicLineageError):
    """권위 metadata/crosswalk가 없어 무누수를 증명할 수 없다."""


@dataclass(frozen=True)
class PublicLineageBuild:
    entries_by_tag: dict[str, list[dict[str, Any]]]
    excluded_by_tag: dict[str, int]
    evidence: dict[str, Any]


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {str(value): str(value) for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def canonical_json_sha256(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _snapshot_metadata(
    repo_root: Path, relative: str, *, label: str
) -> FileSnapshot:
    try:
        return read_regular_file_snapshot(
            repo_root / relative,
            root=repo_root,
            label=label,
        )
    except (OSError, ValueError) as exc:
        raise PublicLineageBlocked(f"{label} 권위 metadata가 없습니다/유효하지 않습니다: {exc}") from exc


def _metadata_evidence(relative: str, snapshot: FileSnapshot) -> dict[str, Any]:
    return {
        "path": relative,
        "sha256": snapshot.sha256,
        "size": int(snapshot.size),
    }


def parse_librispeech_chapters_bytes(
    raw: bytes,
) -> dict[int, tuple[int, int]]:
    """``chapter_id -> (reader_id, Gutenberg book_id)``를 엄격히 읽는다."""

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PublicLineageError("LibriSpeech CHAPTERS.TXT UTF-8 오류") from exc
    result: dict[int, tuple[int, int]] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        fields = [item.strip() for item in line.split("|")]
        if not fields or not fields[0].isdigit():
            continue
        if len(fields) < 6 or not fields[1].isdigit() or not fields[5].isdigit():
            raise PublicLineageError(
                f"LibriSpeech CHAPTERS.TXT line {number} reader/book 필드 오류"
            )
        chapter, reader, book = int(fields[0]), int(fields[1]), int(fields[5])
        previous = result.setdefault(chapter, (reader, book))
        if previous != (reader, book):
            raise PublicLineageError(f"LibriSpeech chapter_id 중복 충돌: {chapter}")
    if not result:
        raise PublicLineageError("LibriSpeech CHAPTERS.TXT mapping이 비었습니다")
    return result


def librispeech_lineage_keys(
    clip: str, chapters: Mapping[int, tuple[int, int]]
) -> tuple[str, ...]:
    stem = Path(str(clip)).stem
    fields = stem.split("-")
    if len(fields) < 3 or not fields[0].isdigit() or not fields[1].isdigit():
        raise PublicLineageError(
            f"LibriSpeech speaker-chapter-utterance 이름이 아닙니다: {clip}"
        )
    speaker, chapter = int(fields[0]), int(fields[1])
    try:
        reader, book = chapters[chapter]
    except KeyError as exc:
        raise PublicLineageError(
            f"LibriSpeech CHAPTERS.TXT에 chapter가 없습니다: {clip}"
        ) from exc
    if speaker != reader:
        raise PublicLineageError(
            f"LibriSpeech filename speaker와 CHAPTERS reader가 다릅니다: "
            f"{clip}: {speaker} != {reader}"
        )
    return (f"librivox_reader:{reader}", f"gutenberg_book:{book}")


def parse_fma_tracks_bytes(raw: bytes) -> dict[int, tuple[str, str]]:
    """FMA two-row header에서 ``track_id -> (artist.id, album.id)``를 읽는다."""

    try:
        handle = io.StringIO(raw.decode("utf-8-sig"), newline="")
    except UnicodeDecodeError as exc:
        raise PublicLineageError("FMA tracks.csv UTF-8 오류") from exc
    with handle:
        reader = csv.reader(handle)
        try:
            level0 = next(reader)
            level1 = next(reader)
        except StopIteration as exc:
            raise PublicLineageError("FMA tracks.csv header가 불완전합니다") from exc
        width = max(len(level0), len(level1))
        level0 += [""] * (width - len(level0))
        level1 += [""] * (width - len(level1))

        def column(group: str, field: str) -> int:
            hits = [
                index
                for index, (left, right) in enumerate(zip(level0, level1))
                if left.strip().casefold() == group
                and right.strip().casefold() == field
            ]
            if len(hits) != 1:
                raise PublicLineageError(
                    f"FMA tracks.csv ({group}, {field}) column이 정확히 하나가 아닙니다"
                )
            return hits[0]

        artist_column = column("artist", "id")
        album_column = column("album", "id")
        result: dict[int, tuple[str, str]] = {}
        for number, row in enumerate(reader, start=3):
            if not row or not row[0].strip().isdigit():
                continue
            if max(artist_column, album_column) >= len(row):
                raise PublicLineageError(f"FMA tracks.csv line {number}가 잘렸습니다")
            track = int(row[0].strip())
            artist, album = row[artist_column].strip(), row[album_column].strip()
            if not artist or not album:
                raise PublicLineageError(
                    f"FMA tracks.csv line {number} artist/album ID가 비었습니다"
                )
            previous = result.setdefault(track, (artist, album))
            if previous != (artist, album):
                raise PublicLineageError(f"FMA track ID 중복 충돌: {track}")
    if not result:
        raise PublicLineageError("FMA tracks.csv mapping이 비었습니다")
    return result


def fma_lineage_keys(clip: str, tracks: Mapping[int, tuple[str, str]]) -> tuple[str, ...]:
    stem = Path(str(clip)).stem
    if not stem.isdigit() or int(stem) not in tracks:
        raise PublicLineageError(f"FMA track ID를 metadata에 매핑할 수 없습니다: {clip}")
    artist, album = tracks[int(stem)]
    return (f"fma_artist:{artist}", f"fma_album:{album}")


def parse_esc50_metadata_bytes(raw: bytes) -> dict[str, str]:
    """ESC-50 filename(casefold) -> original Freesound ``src_file``."""

    try:
        handle = io.StringIO(raw.decode("utf-8-sig"), newline="")
    except UnicodeDecodeError as exc:
        raise PublicLineageError("ESC-50 metadata UTF-8 오류") from exc
    result: dict[str, str] = {}
    with handle:
        reader = csv.DictReader(handle)
        required = {"filename", "fold", "target", "category", "src_file", "take"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise PublicLineageError("ESC-50 metadata header가 불완전합니다")
        for number, row in enumerate(reader, start=2):
            filename = str(row.get("filename") or "").strip()
            source = str(row.get("src_file") or "").strip()
            if not filename or Path(filename).name != filename or not source:
                raise PublicLineageError(f"ESC-50 metadata line {number} identity 오류")
            key = filename.casefold()
            previous = result.setdefault(key, source)
            if previous != source:
                raise PublicLineageError(f"ESC-50 filename 중복 충돌: {filename}")
    if not result:
        raise PublicLineageError("ESC-50 metadata mapping이 비었습니다")
    return result


def esc50_lineage_keys(clip: str, metadata: Mapping[str, str]) -> tuple[str, ...]:
    key = Path(str(clip)).name.casefold()
    if key not in metadata:
        raise PublicLineageError(f"ESC-50 metadata에 filename이 없습니다: {clip}")
    return (f"esc50_src:{metadata[key]}",)


def dns_speech_lineage_keys(clip: str) -> tuple[str, ...]:
    match = _DNS_SPEECH_RE.fullmatch(Path(str(clip)).stem)
    if match is None:
        raise PublicLineageBlocked(
            "DNS read_speech filename의 reader/book 계보를 엄격히 파싱할 수 없습니다: "
            f"{clip}"
        )
    reader = int(match.group("reader"))
    book = int(match.group("book"))
    return (f"dns_reader:{reader}", f"dns_book:{book}")


def dns_audioset_lineage_keys(clip: str) -> tuple[str, ...]:
    stem = Path(str(clip)).stem
    if _AUDIOSET_ID_RE.fullmatch(stem) is None:
        raise PublicLineageBlocked(
            "DNS AudioSet filename에서 official 11-char source ID를 파싱할 수 없습니다: "
            f"{clip}"
        )
    return (f"audioset_video:{stem}",)


def demand_lineage_keys(path: str | Path, *, tag_root: Path) -> tuple[str, ...]:
    absolute = Path(os.path.abspath(Path(path)))
    root = Path(os.path.abspath(tag_root))
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise PublicLineageError(f"DEMAND raw가 tag root 밖입니다: {path}") from exc
    if len(relative.parts) < 2 or relative.parts[0] not in _DEMAND_ENVIRONMENTS:
        raise PublicLineageBlocked(
            f"DEMAND environment/channel official 경로를 파싱할 수 없습니다: {relative}"
        )
    return (f"demand_environment:{relative.parts[0]}",)


def mimii_lineage_keys(path: str | Path, *, tag_root: Path) -> tuple[str, ...]:
    absolute = Path(os.path.abspath(Path(path)))
    root = Path(os.path.abspath(tag_root))
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise PublicLineageError(f"MIMII raw가 tag root 밖입니다: {path}") from exc
    identifiers = [
        match.group(1)
        for part in relative.parts
        if (match := _MIMII_ID_RE.fullmatch(part)) is not None
    ]
    if len(set(identifiers)) != 1:
        raise PublicLineageBlocked(
            f"MIMII fan physical machine id를 official path에서 파싱할 수 없습니다: {relative}"
        )
    # 같은 physical machine의 SNR/normal/abnormal/take를 전부 한 component로 묶는다.
    return (f"mimii_fan_machine:{identifiers[0]}",)


def _marker_members(snapshot: FileSnapshot) -> set[str]:
    assert snapshot.data is not None
    try:
        lines = snapshot.data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise PublicLineageError(f"DNS official member marker UTF-8 오류: {snapshot.path}") from exc
    members: set[str] = set()
    for number, raw in enumerate(lines, start=1):
        value = raw.strip().replace("\\", "/")
        if not value:
            continue
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise PublicLineageError(
                f"DNS marker line {number}가 canonical member path가 아닙니다: {value!r}"
            )
        member = path.as_posix()
        if member in members:
            raise PublicLineageError(f"DNS marker에 member path 중복이 있습니다: {member}")
        members.add(member)
    if not members:
        raise PublicLineageError(f"DNS marker가 비었습니다: {snapshot.path}")
    return members


def _find_tag_root(path: Path, roots: Sequence[Path]) -> Path:
    absolute = Path(os.path.abspath(path))
    candidates = [
        Path(os.path.abspath(root))
        for root in roots
        if absolute == Path(os.path.abspath(root))
        or Path(os.path.abspath(root)) in absolute.parents
    ]
    if not candidates:
        raise PublicLineageError(f"raw path의 tag root를 찾을 수 없습니다: {path}")
    return max(candidates, key=lambda item: len(item.parts))


def build_recorded_clip_lineage(
    families: Mapping[str, Sequence[str]], *, repo_root: str | Path
) -> dict[str, Any]:
    """active holdout clip의 content SHA와 authoritative lineage key를 만든다.

    source repair가 이 결과를 holdout/provenance에 기록하고 validator가 같은 metadata로
    재계산한다. 원본 audio/metadata 하나라도 없으면 canonical holdout를 만들지 않는다.
    """

    root = Path(os.path.abspath(Path(repo_root)))
    chapters_snapshot = _snapshot_metadata(
        root, LIBRISPEECH_CHAPTERS, label="LibriSpeech CHAPTERS.TXT"
    )
    fma_snapshot = _snapshot_metadata(root, FMA_TRACKS, label="FMA tracks.csv")
    esc_snapshot = _snapshot_metadata(root, ESC50_METADATA, label="ESC-50 metadata")
    assert chapters_snapshot.data is not None
    assert fma_snapshot.data is not None
    assert esc_snapshot.data is not None
    chapters = parse_librispeech_chapters_bytes(chapters_snapshot.data)
    tracks = parse_fma_tracks_bytes(fma_snapshot.data)
    esc = parse_esc50_metadata_bytes(esc_snapshot.data)

    fma_root = root / "data/raw/music/fma_small"
    speech_root = root / "data/raw/speech/LibriSpeech"
    esc_root = root / "data/raw/noise/esc50/ESC-50-master/audio"
    records: list[dict[str, Any]] = []
    for family in sorted(families):
        for raw_clip in sorted(str(item).casefold() for item in families[family]):
            clip = Path(raw_clip).name
            if clip != raw_clip:
                raise PublicLineageError(f"holdout clip은 basename이어야 합니다: {raw_clip}")
            if family == "music":
                keys = fma_lineage_keys(clip, tracks)
                candidates = list(fma_root.glob(f"*/{Path(clip).stem}.mp3"))
            elif family == "speech":
                keys = librispeech_lineage_keys(clip, chapters)
                candidates = list(speech_root.rglob(clip))
            elif family in {"environment", "machine"}:
                keys = esc50_lineage_keys(clip, esc)
                # holdout JSON은 basename을 casefold해 보존한다. Linux raw tree의
                # ESC-50 filename에는 대문자가 있을 수 있으므로 case-sensitive 경로를
                # 추측하지 않고 directory의 exact basename mapping으로 되찾는다.
                candidates = [
                    candidate
                    for candidate in esc_root.iterdir()
                    if candidate.name.casefold() == clip
                ] if esc_root.is_dir() else []
            else:
                raise PublicLineageError(f"지원하지 않는 recorded family: {family}")
            candidates = [candidate for candidate in candidates if candidate.is_file()]
            if len(candidates) != 1:
                raise PublicLineageBlocked(
                    f"recorded 원본 clip을 exact 하나로 해석할 수 없습니다: {clip}: {candidates}"
                )
            snapshot = read_regular_file_snapshot(
                candidates[0], root=root, label=f"recorded source clip {clip}", capture_bytes=False
            )
            records.append(
                {
                    "family": str(family),
                    "clip": clip,
                    "content_sha256": snapshot.sha256,
                    "lineage_keys": sorted(keys),
                }
            )
    records.sort(key=lambda item: (item["family"], item["clip"]))
    return {
        "schema_version": 1,
        "metadata": {
            "librispeech_chapters": _metadata_evidence(
                LIBRISPEECH_CHAPTERS, chapters_snapshot
            ),
            "fma_tracks": _metadata_evidence(FMA_TRACKS, fma_snapshot),
            "esc50": _metadata_evidence(ESC50_METADATA, esc_snapshot),
        },
        "clips": records,
        "clips_sha256": canonical_json_sha256(records),
    }


def validate_recorded_clip_lineage(
    holdout_lineage: Mapping[str, Any],
    *,
    families: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """holdout에 고정된 clip lineage를 canonical 행으로 검증한다.

    원본 audio를 다시 열어 SHA를 계산하는 일은 producer의 historical repair 경계에서만
    가능하다. 소비자는 immutable provenance report와 transfer manifest에 결속된 이
    canonical 행/metadata digest를 사용한다.
    """

    if holdout_lineage.get("schema_version") != 1:
        raise PublicLineageBlocked("holdout clip_lineage schema_version=1 증거가 없습니다")
    metadata = holdout_lineage.get("metadata")
    if not isinstance(metadata, dict) or set(metadata) != {
        "librispeech_chapters",
        "fma_tracks",
        "esc50",
    }:
        raise PublicLineageBlocked("holdout clip_lineage metadata 세 종류가 완전하지 않습니다")
    expected_paths = {
        "librispeech_chapters": LIBRISPEECH_CHAPTERS,
        "fma_tracks": FMA_TRACKS,
        "esc50": ESC50_METADATA,
    }
    for name, expected_path in expected_paths.items():
        item = metadata.get(name)
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256", "size"}
            or item.get("path") != expected_path
            or _SHA256_RE.fullmatch(str(item.get("sha256") or "")) is None
            or not isinstance(item.get("size"), int)
            or item.get("size") <= 0
        ):
            raise PublicLineageError(f"holdout clip_lineage metadata.{name}가 유효하지 않습니다")
    clips = holdout_lineage.get("clips")
    if not isinstance(clips, list) or not clips:
        raise PublicLineageBlocked("holdout clip_lineage.clips가 비었습니다")
    canonical: list[dict[str, Any]] = []
    for index, row in enumerate(clips):
        if not isinstance(row, dict):
            raise PublicLineageError(f"holdout clip_lineage #{index}가 mapping이 아닙니다")
        content = str(row.get("content_sha256") or "")
        keys = row.get("lineage_keys")
        clip = str(row.get("clip") or "")
        family = str(row.get("family") or "")
        if (
            _SHA256_RE.fullmatch(content) is None
            or not clip
            or Path(clip).name != clip
            or not family
            or not isinstance(keys, list)
            or not keys
            or keys != sorted(set(str(item) for item in keys))
        ):
            raise PublicLineageError(f"holdout clip_lineage #{index}가 유효하지 않습니다")
        canonical.append(
            {
                "family": family,
                "clip": clip.casefold(),
                "content_sha256": content,
                "lineage_keys": list(keys),
            }
        )
    canonical.sort(key=lambda item: (item["family"], item["clip"]))
    if len({(item["family"], item["clip"]) for item in canonical}) != len(canonical):
        raise PublicLineageError("holdout clip_lineage family/clip이 중복됩니다")
    if holdout_lineage.get("clips_sha256") != canonical_json_sha256(canonical):
        raise PublicLineageError("holdout clip_lineage digest가 실제 행과 다릅니다")
    if families is not None:
        expected = sorted(
            (str(family), Path(str(clip)).name.casefold())
            for family, values in families.items()
            for clip in values
        )
        actual = [(item["family"], item["clip"]) for item in canonical]
        if actual != expected:
            raise PublicLineageError("holdout families와 clip_lineage exact clip 집합이 다릅니다")
    return canonical


def build_public_lineage(
    entries_by_tag: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    tag_roots: Mapping[str, Sequence[str | Path]],
    repo_root: str | Path,
    holdout_lineage: Mapping[str, Any],
) -> PublicLineageBuild:
    """모든 public tag를 한 DSU로 묶고 holdout component 전체를 제외한다."""

    root = Path(os.path.abspath(Path(repo_root)))
    holdout = validate_recorded_clip_lineage(holdout_lineage)
    holdout_basenames = {item["clip"] for item in holdout}
    holdout_content = {item["content_sha256"] for item in holdout}
    holdout_keys = {
        str(key) for item in holdout for key in item.get("lineage_keys", [])
    }
    has_recorded_speech = any(item["family"] == "speech" for item in holdout)

    metadata: dict[str, dict[str, Any]] = {}
    chapters: dict[int, tuple[int, int]] | None = None
    tracks: dict[int, tuple[str, str]] | None = None
    esc: dict[str, str] | None = None

    def load_chapters() -> dict[int, tuple[int, int]]:
        nonlocal chapters
        if chapters is None:
            snapshot = _snapshot_metadata(
                root, LIBRISPEECH_CHAPTERS, label="LibriSpeech CHAPTERS.TXT"
            )
            assert snapshot.data is not None
            chapters = parse_librispeech_chapters_bytes(snapshot.data)
            metadata["librispeech_chapters"] = _metadata_evidence(
                LIBRISPEECH_CHAPTERS, snapshot
            )
        return chapters

    def load_tracks() -> dict[int, tuple[str, str]]:
        nonlocal tracks
        if tracks is None:
            snapshot = _snapshot_metadata(root, FMA_TRACKS, label="FMA tracks.csv")
            assert snapshot.data is not None
            tracks = parse_fma_tracks_bytes(snapshot.data)
            metadata["fma_tracks"] = _metadata_evidence(FMA_TRACKS, snapshot)
        return tracks

    def load_esc() -> dict[str, str]:
        nonlocal esc
        if esc is None:
            snapshot = _snapshot_metadata(root, ESC50_METADATA, label="ESC-50 metadata")
            assert snapshot.data is not None
            esc = parse_esc50_metadata_bytes(snapshot.data)
            metadata["esc50"] = _metadata_evidence(ESC50_METADATA, snapshot)
        return esc

    dns_members: dict[str, set[str]] = {}
    if entries_by_tag.get("speech"):
        marker = _snapshot_metadata(root, DNS_SPEECH_MARKER, label="DNS speech member marker")
        metadata["dns_speech_members"] = _metadata_evidence(DNS_SPEECH_MARKER, marker)
        dns_members["speech"] = _marker_members(marker)
    if entries_by_tag.get("dns_fullband"):
        combined: set[str] = set()
        for index, relative in enumerate(DNS_NOISE_MARKERS):
            marker = _snapshot_metadata(root, relative, label=f"DNS noise member marker {index}")
            metadata[f"dns_noise_members_{index}"] = _metadata_evidence(relative, marker)
            shard_members = _marker_members(marker)
            overlap = combined.intersection(shard_members)
            if overlap:
                raise PublicLineageError(
                    f"DNS noise shards에 중복 basename이 있습니다: {sorted(overlap)[:5]}"
                )
            combined.update(shard_members)
        dns_members["dns_fullband"] = combined

    nodes: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    tag_for_node: dict[str, str] = {}
    for tag in sorted(entries_by_tag):
        roots = [Path(os.path.abspath(Path(value))) for value in tag_roots.get(tag, ())]
        if not roots:
            raise PublicLineageError(f"tag root가 없습니다: {tag}")
        observed_dns_members: set[str] = set()
        for index, source in enumerate(entries_by_tag[tag]):
            item = dict(source)
            path = Path(str(item.get("path") or ""))
            content = str(item.get("content_sha256") or "")
            if _SHA256_RE.fullmatch(content) is None:
                raise PublicLineageError(f"{tag} entry #{index} content_sha256가 없습니다")
            tag_root = _find_tag_root(path, roots)
            basename = path.name.casefold()
            if tag == "music":
                keys = fma_lineage_keys(path.name, load_tracks())
            elif tag == "esc50":
                keys = esc50_lineage_keys(path.name, load_esc())
            elif tag == "demand":
                keys = demand_lineage_keys(path, tag_root=tag_root)
            elif tag == "machine":
                keys = mimii_lineage_keys(path, tag_root=tag_root)
            elif tag == "speech":
                # DNS shard의 read_speech와 로컬 LibriSpeech 진단 tree를 구분한다.
                if _DNS_SPEECH_RE.fullmatch(path.stem):
                    keys = dns_speech_lineage_keys(path.name)
                    if has_recorded_speech:
                        raise PublicLineageBlocked(
                            "DNS read_speech reader/book ID와 recorded LibriSpeech의 "
                            "LibriVox/Gutenberg ID를 연결하는 공식 crosswalk가 없습니다"
                        )
                else:
                    keys = librispeech_lineage_keys(path.name, load_chapters())
                observed_dns_members.add(path.relative_to(tag_root).as_posix())
            elif tag == "dns_fullband":
                keys = dns_audioset_lineage_keys(path.name)
                observed_dns_members.add(path.relative_to(tag_root).as_posix())
            else:
                raise PublicLineageBlocked(f"public lineage parser가 없는 tag입니다: {tag}")
            node = f"{tag}:{index}"
            nodes.append(node)
            tag_for_node[node] = tag
            item["lineage_schema"] = PUBLIC_LINEAGE_SCHEMA
            item["lineage_keys"] = sorted(set(keys))
            rows[node] = item
        if tag in dns_members and observed_dns_members != dns_members[tag]:
            missing = sorted(dns_members[tag] - observed_dns_members)
            extra = sorted(observed_dns_members - dns_members[tag])
            raise PublicLineageError(
                f"DNS {tag} raw tree와 official archive member marker가 다릅니다: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

    dsu = _DisjointSet(nodes)
    owners: dict[str, str] = {}
    for node in nodes:
        item = rows[node]
        identities = [
            f"content:{item['content_sha256']}",
            *(f"lineage:{key}" for key in item["lineage_keys"]),
        ]
        for identity in identities:
            previous = owners.setdefault(identity, node)
            dsu.union(previous, node)

    components: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        components[dsu.find(node)].append(node)
    component_records: dict[str, dict[str, Any]] = {}
    excluded_nodes: set[str] = set()
    for members in components.values():
        members.sort()
        keys = sorted({key for node in members for key in rows[node]["lineage_keys"]})
        contents = sorted({str(rows[node]["content_sha256"]) for node in members})
        basenames = sorted({Path(str(rows[node]["path"])).name.casefold() for node in members})
        basis = {"lineage_keys": keys, "content_sha256": contents}
        group_id = "public-lineage-" + canonical_json_sha256(basis)
        overlap = {
            "basename": sorted(set(basenames).intersection(holdout_basenames)),
            "content_sha256": sorted(set(contents).intersection(holdout_content)),
            "lineage_keys": sorted(set(keys).intersection(holdout_keys)),
        }
        excluded = any(overlap.values())
        for node in members:
            rows[node]["group_id"] = group_id
            if excluded:
                excluded_nodes.add(node)
        component_records[group_id] = {
            "members": members,
            "tags": sorted({tag_for_node[node] for node in members}),
            "lineage_keys": keys,
            "content_sha256": contents,
            "excluded_by_holdout": excluded,
            "overlap": overlap,
        }

    output: dict[str, list[dict[str, Any]]] = {tag: [] for tag in entries_by_tag}
    excluded_by_tag: dict[str, int] = {tag: 0 for tag in entries_by_tag}
    for node in nodes:
        tag = tag_for_node[node]
        if node in excluded_nodes:
            excluded_by_tag[tag] += 1
        else:
            output[tag].append(rows[node])
    for tag in output:
        output[tag].sort(key=lambda item: str(item["path"]))
        if entries_by_tag[tag] and not output[tag]:
            raise PublicLineageBlocked(
                f"{tag}의 모든 component가 recorded holdout와 겹쳐 학습 항목이 없습니다"
            )

    canonical_components = {
        key: component_records[key] for key in sorted(component_records)
    }
    evidence = {
        "schema_version": 1,
        "lineage_schema": PUBLIC_LINEAGE_SCHEMA,
        "metadata": {key: metadata[key] for key in sorted(metadata)},
        "component_count": len(canonical_components),
        "component_membership_sha256": canonical_json_sha256(canonical_components),
        "components": canonical_components,
        "holdout_clips_sha256": holdout_lineage.get("clips_sha256"),
        "excluded_by_tag": dict(sorted(excluded_by_tag.items())),
    }
    return PublicLineageBuild(output, excluded_by_tag, evidence)


def validate_public_manifest_lineage(entries_by_tag: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """이미 생성된 manifest에서 component가 split/tag 경계를 가르는지 독립 검사."""

    by_group: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"splits": set(), "keys": set(), "contents": set(), "tags": set()}
    )
    owner_by_key: dict[str, str] = {}
    owner_by_content: dict[str, str] = {}
    errors: list[str] = []
    for tag, entries in sorted(entries_by_tag.items()):
        for index, entry in enumerate(entries):
            group = str(entry.get("group_id") or "")
            keys = entry.get("lineage_keys")
            content = str(entry.get("content_sha256") or "")
            split = str(entry.get("split") or "")
            if (
                _GROUP_ID_RE.fullmatch(group) is None
                or entry.get("lineage_schema") != PUBLIC_LINEAGE_SCHEMA
                or not isinstance(keys, list)
                or not keys
                or keys != sorted(set(str(item) for item in keys))
                or _SHA256_RE.fullmatch(content) is None
                or split not in {"train", "val", "test"}
            ):
                errors.append(f"{tag}[{index}] lineage 필드가 불완전합니다")
                continue
            item = by_group[group]
            item["splits"].add(split)
            item["keys"].update(str(value) for value in keys)
            item["contents"].add(content)
            item["tags"].add(tag)
            for key in keys:
                previous = owner_by_key.setdefault(str(key), group)
                if previous != group:
                    errors.append(f"lineage key가 여러 component에 있습니다: {key}")
            previous_content = owner_by_content.setdefault(content, group)
            if previous_content != group:
                errors.append(f"content SHA가 여러 component에 있습니다: {content}")
    crossings = sorted(group for group, item in by_group.items() if len(item["splits"]) != 1)
    if crossings:
        errors.append(f"public lineage component가 split을 가로지릅니다: {crossings[:5]}")
    for group, item in sorted(by_group.items()):
        expected_group = "public-lineage-" + canonical_json_sha256(
            {
                "lineage_keys": sorted(item["keys"]),
                "content_sha256": sorted(item["contents"]),
            }
        )
        if group != expected_group:
            errors.append(f"public lineage group_id가 실제 identity digest와 다릅니다: {group}")
    if errors:
        raise PublicLineageError("; ".join(errors))
    summary = {
        group: {
            "split": next(iter(item["splits"])),
            "tags": sorted(item["tags"]),
            "lineage_keys": sorted(item["keys"]),
            "content_sha256": sorted(item["contents"]),
        }
        for group, item in sorted(by_group.items())
    }
    return {
        "component_count": len(summary),
        "component_membership_sha256": canonical_json_sha256(summary),
        "components": summary,
    }


__all__ = [
    "DNS_NOISE_MARKERS",
    "DNS_SPEECH_MARKER",
    "ESC50_METADATA",
    "FMA_TRACKS",
    "LIBRISPEECH_CHAPTERS",
    "PUBLIC_LINEAGE_SCHEMA",
    "PublicLineageBlocked",
    "PublicLineageBuild",
    "PublicLineageError",
    "build_public_lineage",
    "build_recorded_clip_lineage",
    "canonical_json_sha256",
    "demand_lineage_keys",
    "dns_audioset_lineage_keys",
    "dns_speech_lineage_keys",
    "esc50_lineage_keys",
    "fma_lineage_keys",
    "librispeech_lineage_keys",
    "mimii_lineage_keys",
    "parse_esc50_metadata_bytes",
    "parse_fma_tracks_bytes",
    "parse_librispeech_chapters_bytes",
    "validate_recorded_clip_lineage",
    "validate_public_manifest_lineage",
]
