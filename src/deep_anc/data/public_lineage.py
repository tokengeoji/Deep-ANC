"""Public corpus 원본 계보와 component 원자 분할 계약.

경로 basename만 비교하면 같은 원본의 rename/copy와 같은 speaker/book, artist/album,
Freesound source의 다른 take를 놓친다. 이 모듈은 코퍼스가 제공한 권위 metadata와 raw
content SHA를 한 번에 결속하고, 그 키들의 transitive closure를 ``group_id``로 만든다.

DNS ``read_speech``의 파일명에는 reader/book 식별자가 있지만 DNS 배포본에는 이를
LibriSpeech의 LibriVox/Gutenberg ID로 변환하는 crosswalk가 포함되어 있지 않다. 두
namespace를 같은 것으로 추측하지 않도록 DNS 키에는 ``dns_`` 접두사를 유지한다. 따라서
DNS와 recorded 사이의 lineage join은 하지 않으며, 양쪽을 가로지르는 중복은 공통
content SHA와 basename으로만 차단한다. 이 정책은 오디오 바이트 수준의 중복을 엄격히
제거하면서도, 존재하지 않는 crosswalk를 만들어 잘못된 component를 만드는 것을 막는다.
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

from .holdout_contract import (
    FileSnapshot,
    read_regular_file_snapshot,
    reject_symlink_components,
)


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
_MIMII_DG_FILE_RE = re.compile(
    r"^section_(?P<section>\d+)_(?P<domain>source|target)_"
    r"(?P<split>train|test)_(?P<condition>normal|anomaly)_"
    r"(?P<index>\d+)(?:_(?P<attribute>[A-Za-z0-9_-]+))?$",
    re.IGNORECASE,
)
_DEMAND_ENVIRONMENTS = frozenset(
    {"DKITCHEN", "DWASHING", "OOFFICE", "OHALLWAY", "TMETRO", "TCAR"}
)
_DNS_MARKER_TAGS = ("dns_fullband", "speech")
_DNS_MARKER_PARTITION_SCHEMA_VERSION = 1
# ``speech`` tag에는 DNS read_speech와 local LibriSpeech source tree가 함께 있을 수
# 있다. archive member marker는 전자에만 적용된다. tag 이름만으로 두 root를 섞으면
# LibriSpeech decoder reject가 DNS marker의 fake extra member가 된다.
DNS_MARKER_TAG_ROOTS = {
    "dns_fullband": "data/raw/noise/dns_fullband",
    "speech": "data/raw/noise/speech",
}


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
    """계보 component용 결정론적·비재귀 union-find.

    이전 구현은 root 이름의 lexical 최소값만 부모로 골랐다. component의 의미는
    맞지만, identity가 역순으로 이어지는 입력에서는 parent chain이 길어지고
    재귀 ``find``가 Python recursion limit에 걸릴 수 있었다. root 이름은 공개
    artifact에 포함되지 않으며 members/identity digest가 component를 정의한다.
    따라서 size 우선(동률만 lexical tie-break)과 반복 path compression으로
    의미·결정성을 유지하면서 최악 입력도 안전하게 처리한다.
    """

    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {str(value): str(value) for value in values}
        self.size = {str(value): 1 for value in values}

    def find(self, value: str) -> str:
        current = str(value)
        root = current
        while self.parent[root] != root:
            root = self.parent[root]
        # recursion을 쓰지 않아 adversarial component chain에서도 stack을 소비하지
        # 않는다. 모든 방문 node를 같은 root에 직접 연결한다.
        while current != root:
            parent = self.parent[current]
            self.parent[current] = root
            current = parent
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        # union-by-size는 find depth를 log 수준으로 제한한다. 같은 size일 때만
        # lexical 순서를 사용하므로 같은 입력의 parent forest도 결정론적이다.
        if self.size[a] < self.size[b] or (self.size[a] == self.size[b] and a > b):
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


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
    if len(set(identifiers)) == 1:
        # 기존 MIMII 경로(id_00/normal 등)는 같은 physical machine의
        # SNR/normal/abnormal/take를 전부 한 component로 묶는다.
        return (f"mimii_fan_machine:{identifiers[0]}",)

    # MIMII-DG official fan.zip은 id_XX 디렉터리를 사용하지 않고,
    # section/domain/condition/attribute를 파일명에 기록한다. 여기의
    # ``m-n_W`` 같은 suffix는 domain-shift attribute이지 physical machine
    # ID가 아니다. 공식 파일명에 physical machine crosswalk가 없으므로
    # 누수를 놓치지 않도록 같은 section의 source/target/train/test 및
    # normal/anomaly를 하나의 보수적 component로 묶는다.
    match = _MIMII_DG_FILE_RE.fullmatch(relative.stem)
    if match is not None:
        return (f"mimii_dg_fan_section:{int(match.group('section')):02d}",)

    raise PublicLineageBlocked(
        f"MIMII fan physical machine/official DG section을 파싱할 수 없습니다: {relative}"
    )


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


def _canonical_dns_member(value: object, *, label: str) -> str:
    """DNS archive marker와 비교할 root-relative POSIX member를 엄격히 정규화한다.

    decoder audit의 raw ``relative_path``와 달리 이 값은 tag root 기준이다. basename
    만으로 비교하면 같은 이름의 다른 directory를 marker에 끼워 넣을 수 있으므로,
    path 전체를 그대로 보존한다. Windows separator/case alias도 canonical marker
    membership의 두 번째 표현이 될 수 없게 막는다.
    """

    if not isinstance(value, str) or not value:
        raise PublicLineageError(f"{label}가 비었거나 문자열이 아닙니다")
    if "\\" in value:
        raise PublicLineageError(f"{label}는 POSIX separator만 써야 합니다: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PublicLineageError(f"{label}에 절대/상위/빈 path component가 있습니다: {value!r}")
    normalised = path.as_posix()
    if normalised != value:
        raise PublicLineageError(f"{label}는 정규화된 POSIX path여야 합니다: {value!r}")
    return normalised


def _normalise_decoder_marker_members(
    value: object,
    *,
    tag: str,
    kind: str,
) -> tuple[str, ...]:
    """audit accept/reject marker projection을 결정론적·case-safe tuple로 검증한다."""

    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PublicLineageError(
            f"decoder {kind} member {tag}는 정렬된 POSIX path 목록이어야 합니다"
        )
    members = tuple(
        _canonical_dns_member(item, label=f"decoder {kind} member {tag}#{index}")
        for index, item in enumerate(value)
    )
    if members != tuple(sorted(members)) or len(set(members)) != len(members):
        raise PublicLineageError(
            f"decoder {kind} member {tag}는 중복 없는 오름차순 목록이어야 합니다"
        )
    folded: dict[str, str] = {}
    for member in members:
        previous = folded.setdefault(member.casefold(), member)
        if previous != member:
            raise PublicLineageError(
                f"decoder {kind} member {tag}에 case-variant alias가 있습니다: "
                f"{previous!r}, {member!r}"
            )
    return members


def _canonical_dns_tag_roots(
    tag_roots: Mapping[str, Sequence[str | Path]],
    *,
    repo_root: Path,
    active_tags: Sequence[str],
) -> dict[str, tuple[Path, ...]]:
    """DNS marker tag root를 repository-relative, non-symlink boundary로 고정한다."""

    result: dict[str, tuple[Path, ...]] = {}
    for tag in active_tags:
        raw_roots = tag_roots.get(tag, ())
        if isinstance(raw_roots, (str, bytes)) or not isinstance(raw_roots, Sequence):
            raise PublicLineageError(f"DNS {tag} tag root 목록이 유효하지 않습니다")
        roots: list[Path] = []
        relative_roots: list[str] = []
        for index, raw_root in enumerate(raw_roots):
            try:
                candidate = Path(raw_root)
            except TypeError as exc:
                raise PublicLineageError(
                    f"DNS {tag} tag root #{index}가 path가 아닙니다"
                ) from exc
            absolute = Path(
                os.path.abspath(candidate if candidate.is_absolute() else repo_root / candidate)
            )
            try:
                relative = absolute.relative_to(repo_root).as_posix()
            except ValueError as exc:
                raise PublicLineageError(
                    f"DNS {tag} tag root가 repository 밖입니다: {absolute}"
                ) from exc
            try:
                reject_symlink_components(absolute, root=repo_root)
            except Exception as exc:
                raise PublicLineageError(
                    f"DNS {tag} tag root 경로 계약 위반: {absolute}: {exc}"
                ) from exc
            if not absolute.is_dir():
                raise PublicLineageError(f"DNS {tag} tag root가 directory가 아닙니다: {absolute}")
            roots.append(absolute)
            relative_roots.append(relative)
        expected = Path(os.path.abspath(repo_root / DNS_MARKER_TAG_ROOTS[tag]))
        marker_roots = [candidate for candidate in roots if candidate == expected]
        if len(marker_roots) != 1:
            raise PublicLineageError(
                f"DNS {tag} marker partition은 canonical archive tag root "
                f"{DNS_MARKER_TAG_ROOTS[tag]!r}를 정확히 하나 포함해야 합니다: "
                f"{relative_roots}"
            )
        # 같은 public tag의 다른 source tree(예: LibriSpeech)는 lineage/DSU에는
        # 계속 참여하지만, DNS archive marker membership에는 관여하지 않는다.
        result[tag] = tuple(marker_roots)
    return result


def _dns_marker_members_and_metadata(
    root: Path,
    *,
    tag: str,
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """DNS official archive marker와 그 same-FD provenance를 읽는다."""

    if tag == "speech":
        marker = _snapshot_metadata(
            root, DNS_SPEECH_MARKER, label="DNS speech member marker"
        )
        return _marker_members(marker), {
            "dns_speech_members": _metadata_evidence(DNS_SPEECH_MARKER, marker)
        }
    if tag == "dns_fullband":
        combined: set[str] = set()
        metadata: dict[str, dict[str, Any]] = {}
        for index, relative in enumerate(DNS_NOISE_MARKERS):
            marker = _snapshot_metadata(
                root, relative, label=f"DNS noise member marker {index}"
            )
            metadata[f"dns_noise_members_{index}"] = _metadata_evidence(
                relative, marker
            )
            shard_members = _marker_members(marker)
            overlap = combined.intersection(shard_members)
            if overlap:
                raise PublicLineageError(
                    f"DNS noise shards에 중복 basename이 있습니다: {sorted(overlap)[:5]}"
                )
            combined.update(shard_members)
        return combined, metadata
    raise PublicLineageError(f"DNS marker가 없는 tag입니다: {tag}")


def validate_dns_marker_partition(
    entries_by_tag: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    tag_roots: Mapping[str, Sequence[str | Path]],
    repo_root: str | Path,
    decoder_rejected_members_by_tag: Mapping[str, Sequence[str]] | None = None,
    decoder_accepted_members_by_tag: Mapping[str, Sequence[str]] | None = None,
    decoder_audit_inventory_sha256: str | None = None,
) -> dict[str, Any] | None:
    """DNS marker가 accepted와 decoder-audit reject의 exact partition인지 검증한다.

    읽을 수 없는 WAV가 ``scan_wavs``에서 빠지는 것은 정상적인 audit 결과일 수 있다.
    그렇더라도 archive marker의 member가 사라지면 raw tree/manifest가 불완전해진다.
    따라서 canonical build에서는 다음만 허용한다.

    ``official marker == accepted manifest members ∪ audit-rejected members``
    ``accepted manifest members ∩ audit-rejected members == ∅``

    reject는 학습/DSU component에 넣지 않는다. 이 함수는 marker completeness의
    evidence만 다루므로 rejected bytes가 accepted lineage identity를 바꾸지 않는다.
    """

    root = Path(os.path.abspath(Path(repo_root)))
    if (decoder_rejected_members_by_tag is None) != (
        decoder_audit_inventory_sha256 is None
    ):
        raise PublicLineageError(
            "decoder reject member projection과 audit inventory SHA는 함께 있어야 합니다"
        )
    if decoder_accepted_members_by_tag is not None and (
        decoder_rejected_members_by_tag is None
        or decoder_audit_inventory_sha256 is None
    ):
        raise PublicLineageError(
            "decoder accept member projection은 audit-bound reject projection과 함께 있어야 합니다"
        )
    audit_bound = decoder_rejected_members_by_tag is not None
    rejected_by_tag: Mapping[str, Sequence[str]]
    if decoder_rejected_members_by_tag is None:
        rejected_by_tag = {}
    else:
        if not isinstance(decoder_rejected_members_by_tag, Mapping):
            raise PublicLineageError("decoder reject member projection은 mapping이어야 합니다")
        if _SHA256_RE.fullmatch(str(decoder_audit_inventory_sha256 or "")) is None:
            raise PublicLineageError("decoder reject member projection의 audit inventory SHA가 유효하지 않습니다")
        extra_tags = sorted(
            str(tag)
            for tag in decoder_rejected_members_by_tag
            if str(tag) not in _DNS_MARKER_TAGS
        )
        if extra_tags:
            raise PublicLineageError(
                f"DNS marker partition에 지원하지 않는 tag가 있습니다: {extra_tags}"
            )
        rejected_by_tag = decoder_rejected_members_by_tag

    accepted_by_tag: Mapping[str, Sequence[str]]
    if decoder_accepted_members_by_tag is None:
        accepted_by_tag = {}
    else:
        if not isinstance(decoder_accepted_members_by_tag, Mapping):
            raise PublicLineageError("decoder accept member projection은 mapping이어야 합니다")
        extra_tags = sorted(
            str(tag)
            for tag in decoder_accepted_members_by_tag
            if str(tag) not in _DNS_MARKER_TAGS
        )
        if extra_tags:
            raise PublicLineageError(
                f"DNS marker partition accept projection에 지원하지 않는 tag가 있습니다: {extra_tags}"
            )
        accepted_by_tag = decoder_accepted_members_by_tag

    active_tags: list[str] = []
    for tag in _DNS_MARKER_TAGS:
        raw_roots = tag_roots.get(tag, ())
        if isinstance(raw_roots, (str, bytes)) or not isinstance(raw_roots, Sequence):
            raise PublicLineageError(f"DNS {tag} tag root 목록이 유효하지 않습니다")
        expected_root = Path(os.path.abspath(root / DNS_MARKER_TAG_ROOTS[tag]))
        has_marker_root = False
        for index, value in enumerate(raw_roots):
            try:
                candidate = Path(value)
            except TypeError as exc:
                raise PublicLineageError(
                    f"DNS {tag} tag root #{index}가 path가 아닙니다"
                ) from exc
            absolute = Path(
                os.path.abspath(candidate if candidate.is_absolute() else root / candidate)
            )
            if absolute == expected_root:
                has_marker_root = True
                break
        if has_marker_root or tag in rejected_by_tag or tag in accepted_by_tag:
            active_tags.append(tag)
    if not active_tags:
        return None
    roots_by_tag = _canonical_dns_tag_roots(
        tag_roots, repo_root=root, active_tags=active_tags
    )
    evidence_tags: dict[str, dict[str, Any]] = {}
    marker_metadata: dict[str, dict[str, Any]] = {}
    for tag in active_tags:
        roots = roots_by_tag[tag]
        marker_members, metadata = _dns_marker_members_and_metadata(root, tag=tag)
        marker_metadata.update(metadata)
        rejected = _normalise_decoder_marker_members(
            rejected_by_tag.get(tag), tag=tag, kind="reject"
        )
        rejected_set = set(rejected)
        observed: set[str] = set()
        observed_folded: dict[str, str] = {}
        for index, source in enumerate(entries_by_tag.get(tag, ())):
            path = Path(str(source.get("path") or ""))
            absolute_path = Path(os.path.abspath(path))
            # speech tag의 non-DNS source (LibriSpeech)는 DNS official archive
            # marker와 독립이다. marker root에 속한 accepted member만 partition의
            # accepted 집합으로 보며, 나머지는 아래 public DSU가 처리한다.
            if roots[0] not in absolute_path.parents:
                continue
            tag_root = _find_tag_root(path, roots)
            try:
                member = _canonical_dns_member(
                    absolute_path.relative_to(tag_root).as_posix(),
                    label=f"DNS {tag} accepted member #{index}",
                )
            except ValueError as exc:
                raise PublicLineageError(
                    f"DNS {tag} accepted member가 tag root 밖입니다: {path}"
                ) from exc
            if member in observed:
                raise PublicLineageError(f"DNS {tag} accepted member가 중복됩니다: {member}")
            previous = observed_folded.setdefault(member.casefold(), member)
            if previous != member:
                raise PublicLineageError(
                    f"DNS {tag} accepted member에 case-variant alias가 있습니다: "
                    f"{previous!r}, {member!r}"
                )
            observed.add(member)
        if decoder_accepted_members_by_tag is None:
            accepted = observed
        else:
            accepted = set(
                _normalise_decoder_marker_members(
                    accepted_by_tag.get(tag), tag=tag, kind="accept"
                )
            )
            not_audited = sorted(observed.difference(accepted))
            if not_audited:
                raise PublicLineageError(
                    f"DNS {tag} manifest accepted member가 decoder audit accept projection에 "
                    f"없습니다: {not_audited[:5]}"
                )
        overlap = sorted(accepted.intersection(rejected_set))
        missing = sorted(marker_members.difference(accepted).difference(rejected_set))
        extra_accepted = sorted(accepted.difference(marker_members))
        extra_rejected = sorted(rejected_set.difference(marker_members))
        if overlap or missing or extra_accepted or extra_rejected:
            detail: list[str] = []
            if overlap:
                detail.append(f"accepted/reject overlap={overlap[:5]}")
            if missing:
                detail.append(f"marker missing={missing[:5]}")
            if extra_accepted:
                detail.append(f"accepted outside marker={extra_accepted[:5]}")
            if extra_rejected:
                detail.append(f"reject outside marker={extra_rejected[:5]}")
            raise PublicLineageError(
                f"DNS {tag} official marker exact partition 위반: {'; '.join(detail)}"
            )
        relative_roots = [item.relative_to(root).as_posix() for item in roots]
        evidence_tags[tag] = {
            "tag_roots": relative_roots,
            "accepted_members": sorted(accepted),
            "accepted_member_count": len(accepted),
            "accepted_members_sha256": canonical_json_sha256(sorted(accepted)),
            "rejected_members": list(rejected),
            "rejected_member_count": len(rejected),
            "rejected_members_sha256": canonical_json_sha256(list(rejected)),
        }
    return {
        "schema_version": _DNS_MARKER_PARTITION_SCHEMA_VERSION,
        "decoder_audit_inventory_sha256": (
            str(decoder_audit_inventory_sha256) if audit_bound else None
        ),
        "marker_metadata": {key: marker_metadata[key] for key in sorted(marker_metadata)},
        "tags": {key: evidence_tags[key] for key in sorted(evidence_tags)},
    }


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
    extra_excluded_basenames: Iterable[str] = (),
    decoder_rejected_members_by_tag: Mapping[str, Sequence[str]] | None = None,
    decoder_audit_inventory_sha256: str | None = None,
) -> PublicLineageBuild:
    """모든 public tag를 한 DSU로 묶고 제외 component 전체를 제외한다.

    ``holdout_lineage``는 recorded 평가 holdout의 권위 계보다. 그러나 실제 녹음에
    사용한 source-pool CSV에는 holdout 세션 외의 예약/활성 clip도 남을 수 있다. 그
    clip이 합성 manifest에 들어가면 readiness가 basename 교집합을 검출한다. 따라서
    producer가 source-pool 전체 basename을 추가 exclusion으로 전달할 수 있게 하되,
    lineage component 단위 원자성은 그대로 유지한다.
    """

    root = Path(os.path.abspath(Path(repo_root)))
    holdout = validate_recorded_clip_lineage(holdout_lineage)
    holdout_basenames = {item["clip"] for item in holdout}
    extra_basenames = {
        str(item).strip().replace("\\", "/").rsplit("/", 1)[-1].casefold()
        for item in extra_excluded_basenames
        if str(item).strip()
    }
    excluded_basenames = holdout_basenames | extra_basenames
    holdout_content = {item["content_sha256"] for item in holdout}
    holdout_keys = {
        str(key) for item in holdout for key in item.get("lineage_keys", [])
    }
    metadata: dict[str, dict[str, Any]] = {}
    chapters: dict[int, tuple[int, int]] | None = None
    tracks: dict[int, tuple[str, str]] | None = None
    esc: dict[str, str] | None = None

    # DNS marker는 scan 가능한 accepted entry만으로는 완전성을 증명할 수 없다.
    # decoder audit가 거부한 broken file도 marker의 한 member일 수 있으므로, DSU를
    # 만들기 전에 exact partition을 먼저 고정한다. rejected row 자체는 아래
    # accepted-only ``rows``에 절대 들어가지 않는다.
    dns_marker_partition = validate_dns_marker_partition(
        entries_by_tag,
        tag_roots=tag_roots,
        repo_root=root,
        decoder_rejected_members_by_tag=decoder_rejected_members_by_tag,
        decoder_audit_inventory_sha256=decoder_audit_inventory_sha256,
    )
    if dns_marker_partition is not None:
        marker_metadata = dns_marker_partition["marker_metadata"]
        assert isinstance(marker_metadata, dict)
        metadata.update(marker_metadata)

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

    nodes: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    tag_for_node: dict[str, str] = {}
    for tag in sorted(entries_by_tag):
        roots = [Path(os.path.abspath(Path(value))) for value in tag_roots.get(tag, ())]
        if not roots:
            raise PublicLineageError(f"tag root가 없습니다: {tag}")
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
                else:
                    keys = librispeech_lineage_keys(path.name, load_chapters())
            elif tag == "dns_fullband":
                keys = dns_audioset_lineage_keys(path.name)
            else:
                raise PublicLineageBlocked(f"public lineage parser가 없는 tag입니다: {tag}")
            node = f"{tag}:{index}"
            nodes.append(node)
            tag_for_node[node] = tag
            item["lineage_schema"] = PUBLIC_LINEAGE_SCHEMA
            item["lineage_keys"] = sorted(set(keys))
            rows[node] = item
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
            "basename": sorted(set(basenames).intersection(excluded_basenames)),
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
        "crosswalk_policy": {
            "dns_read_speech_to_librispeech": "namespace_disjoint_no_official_crosswalk",
            "cross_namespace_overlap_checks": ["content_sha256", "basename"],
        },
    }
    if dns_marker_partition is not None:
        evidence["decoder_rejected_marker_partition"] = dns_marker_partition
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
    "DNS_MARKER_TAG_ROOTS",
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
    "validate_dns_marker_partition",
    "validate_recorded_clip_lineage",
    "validate_public_manifest_lineage",
]
