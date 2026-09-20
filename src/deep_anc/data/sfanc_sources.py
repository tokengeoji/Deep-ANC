"""읽기 전용 LibriSpeech 음성 원천: SFANC 시뮬레이션 사전학습 전용.

동기 REF/ERR 실측 녹음이 아니다. 파일 쓰기·다운로드·재생·리샘플링·정규화는
하지 않으며, 반환 manifest의 저장은 호출자가 새 출력 경로에서 책임진다.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Iterator, Mapping

import numpy as np
import soundfile as sf


SCHEMA = "sfanc_librispeech_sources.v1"
SPLITS = ("train", "validation", "test")
LICENSE_ID = "CC-BY-4.0"
METADATA_FILES = ("LICENSE.TXT", "SPEAKERS.TXT", "CHAPTERS.TXT")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def _source_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part in (".", "..") for part in path.parts):
        raise ValueError("원천 경로는 corpus 내부 상대경로여야 합니다")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("원천 파일/상위 경로의 symlink는 허용하지 않습니다")
    if not current.is_file() or not current.resolve().is_relative_to(root):
        raise ValueError(f"원천 파일이 없거나 corpus 밖입니다: {relative}")
    return current


def _table(text: str, minimum: int, filename: str) -> dict[str, list[str]]:
    rows = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith(";"):
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) < minimum or not fields[0].isdigit() or fields[0] in rows:
            raise ValueError(f"{filename} 메타데이터 형식/중복 ID 오류")
        rows[fields[0]] = fields
    return rows


def _metadata(root: Path):
    paths = {name: _source_path(root, name) for name in METADATA_FILES}
    texts = {name: path.read_text(encoding="utf-8-sig") for name, path in paths.items()}
    if not re.search(r"Creative\s+Commons\s+Attribution\s+4\.0\s+International\s+License",
                     texts["LICENSE.TXT"], flags=re.IGNORECASE):
        raise ValueError("LibriSpeech CC BY 4.0 license 확인이 필요합니다")
    return (_table(texts["SPEAKERS.TXT"], 5, "SPEAKERS.TXT"),
            _table(texts["CHAPTERS.TXT"], 8, "CHAPTERS.TXT"),
            {name: _sha(path) for name, path in paths.items()})


def _identity(relative: str, speakers, chapters):
    parts = Path(relative).parts
    match = re.fullmatch(r"(\d+)-(\d+)-(\d+)\.flac", parts[-1]) if parts else None
    if len(parts) != 4 or parts[0] != "dev-clean" or match is None:
        raise ValueError(f"dev-clean utterance 경로 형식 오류: {relative}")
    speaker, chapter, _ = match.groups()
    if parts[1:3] != (speaker, chapter) or speaker not in speakers or chapter not in chapters:
        raise ValueError("utterance 경로와 speaker/chapter 메타데이터가 불일치합니다")
    speaker_row, chapter_row = speakers[speaker], chapters[chapter]
    if speaker_row[2] != "dev-clean" or chapter_row[3] != "dev-clean" or chapter_row[1] != speaker:
        raise ValueError("speaker/chapter의 reader 또는 공식 subset이 불일치합니다")
    book = chapter_row[5]
    if not book.isdigit() or int(book) <= 0:
        raise ValueError("유효한 Gutenberg book ID가 필요합니다")
    return speaker, chapter, book


def prepare_librispeech_manifest(
    root: str | Path, *, seed: int = 20260920,
    max_files_per_split: Mapping[str, int] | None = None,
    crop_samples: int = 8192, max_crops_per_file: int = 2,
) -> dict:
    """전체 화자/책 연결요소를 먼저 분할하고 제한된 파일만 SHA/구간 선택한다.

    root는 LICENSE.TXT/SPEAKERS.TXT/CHAPTERS.TXT/dev-clean을 포함한 LibriSpeech
    디렉터리다. 기존 corpus manifest는 읽거나 수정하지 않는다. 80/10/10에 가까운
    그룹 수 분할이며 각 split에 최소 한 그룹을 둔다. 공식 ASR split 평가는 아니다.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed는 비음수 정수여야 합니다")
    if type(crop_samples) is not int or not 1 <= crop_samples <= 8192:
        raise ValueError("crop_samples는 1~8192 정수여야 합니다")
    if type(max_crops_per_file) is not int or not 1 <= max_crops_per_file <= 2:
        raise ValueError("max_crops_per_file은 1 또는 2여야 합니다")
    limits = {"train": 60, "validation": 20, "test": 20}
    if max_files_per_split is not None:
        if set(max_files_per_split) - set(SPLITS):
            raise ValueError("알 수 없는 split 제한")
        limits.update(max_files_per_split)
    if any(type(value) is not int or value < 1 for value in limits.values()):
        raise ValueError("split별 파일 제한은 양의 정수여야 합니다")
    root = Path(root).resolve()
    speakers, chapters, metadata_hashes = _metadata(root)
    parents = {}

    def find(value):
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]

    records = []
    for candidate in sorted((root / "dev-clean").rglob("*.flac")):
        relative = candidate.relative_to(root).as_posix()
        path = _source_path(root, relative)
        speaker, chapter, book = _identity(relative, speakers, chapters)
        parents[find("speaker:" + speaker)] = find("book:" + book)
        info = sf.info(path)
        if info.samplerate != 16000 or info.channels != 1 or info.frames <= 0 or info.format != "FLAC":
            raise ValueError(f"16 kHz mono FLAC만 허용합니다: {relative}")
        records.append({"source_id": "librispeech:dev-clean:" + path.stem,
                        "path": relative, "speaker": speaker, "chapter": chapter, "book": book,
                        "frames": info.frames, "duration_s": info.frames / 16000,
                        "sample_rate": 16000, "channels": 1, "license": LICENSE_ID,
                        "attribution_reader": speakers[speaker][4], "official_subset": "dev-clean"})
    if not records:
        raise ValueError("dev-clean FLAC가 없습니다")
    members = defaultdict(list)
    for node in parents:
        members[find(node)].append(node)
    group_ids = {component: "librispeech-group-" + _key(sorted(nodes))[:24]
                 for component, nodes in members.items()}
    groups = defaultdict(list)
    short_files = 0
    for record in records:
        record["group_id"] = group_ids[find("speaker:" + record["speaker"])]
        # 짧은 파일도 위 연결요소에는 포함: 제외 파일을 매개로 한 누출을 막는다.
        if record["frames"] < crop_samples:
            short_files += 1
            continue
        groups[record["group_id"]].append(record)
    ordered = sorted(groups, key=lambda group: _key(seed, "group", group))
    if len(ordered) < 3:
        raise ValueError("train/validation/test에 최소 3개 독립 speaker/book 그룹이 필요합니다")
    heldout_count = max(1, round(len(ordered) * 0.1))
    assigned = {"train": ordered[:-2 * heldout_count],
                "validation": ordered[-2 * heldout_count:-heldout_count],
                "test": ordered[-heldout_count:]}
    splits, hashes = {}, set()
    for split in SPLITS:
        queues = [sorted(groups[group], key=lambda row: _key(seed, "file", row["path"]))
                  for group in assigned[split]]
        # 큰 그룹 하나가 파일 상한을 독점하지 않도록 round-robin으로 선별한다.
        selected = [queue[index] for index in range(max(map(len, queues)))
                    for queue in queues if index < len(queue)][:limits[split]]
        for record in selected:
            digest = _sha(_source_path(root, record["path"]))
            if digest in hashes:
                raise ValueError("선택 원천의 중복 SHA가 발견됐습니다; 중복을 임의 제거하지 않습니다")
            hashes.add(digest)
            record["sha256"] = digest
            record["split"] = split
            generator = np.random.default_rng(int(_key(seed, record["source_id"], digest)[:16], 16))
            count = min(max_crops_per_file, record["frames"] // crop_samples)
            boundaries = np.linspace(0, record["frames"], count + 1, dtype=np.int64)
            record["crops"] = [{"start": int(generator.integers(int(left), int(right) - crop_samples + 1)),
                                "frames": crop_samples}
                               for left, right in zip(boundaries[:-1], boundaries[1:])]
        splits[split] = selected
    return {
        "schema": SCHEMA, "root": str(root), "seed": seed, "sample_rate": 16000,
        "source_corpus": "LibriSpeech/dev-clean", "license": LICENSE_ID,
        "simulation_pretraining_only": True, "measured_ref_err": False,
        "physical_claim_allowed": False, "deployment_allowed": False,
        "amplitude_domain": "decoded_source_audio_not_calibrated_adc_dac",
        "resampling": False, "normalization": False,
        "split_policy": "speaker_and_gutenberg_book_connected_components_then_group_round_robin",
        "official_asr_split_evaluation": False, "crop_samples": crop_samples,
        "max_crops_per_file": max_crops_per_file, "max_files_per_split": limits,
        "inventory": {"utterances": len(records), "components": len(members),
                      "eligible_components": len(groups), "too_short_utterances": short_files,
                      "group_counts": {split: len(values) for split, values in assigned.items()},
                      "selected_group_counts": {split: len({row["group_id"] for row in values})
                                                for split, values in splits.items()}},
        "provenance": {"source_reference": "https://www.openslr.org/12/",
                       "metadata_sha256": metadata_hashes,
                       "official_archive_checksum_verified": False,
                       "pcm_validation_scope": "selected_crops_only_on_iteration",
                       "byte_hash_duplicate_check_scope": "selected_files_only",
                       "network_or_download_performed": False},
        "splits": splits,
    }


def iter_librispeech_crops(manifest: dict, split: str, *, root: str | Path | None = None
                          ) -> Iterator[tuple[np.ndarray, dict]]:
    """각 crop만 decode하여 (float32 1D audio, provenance)를 반환한다."""
    if manifest.get("schema") != SCHEMA or split not in SPLITS:
        raise ValueError("SFANC 원천 manifest schema/split 오류")
    if manifest.get("simulation_pretraining_only") is not True or manifest.get("measured_ref_err") is not False:
        raise ValueError("시뮬레이션 음성 원천 전용 manifest가 필요합니다")
    root = Path(manifest["root"] if root is None else root).resolve()
    speakers, chapters, metadata_hashes = _metadata(root)
    if metadata_hashes != manifest["provenance"]["metadata_sha256"]:
        raise ValueError("license/speaker/chapter 메타데이터 SHA가 변경됐습니다")
    seen = {field: {} for field in ("source_id", "group_id", "speaker", "book")}
    for name in SPLITS:
        if not manifest["splits"].get(name):
            raise ValueError("모든 split에 최소 한 원천이 필요합니다")
        for record in manifest["splits"][name]:
            if record["split"] != name:
                raise ValueError("manifest 내부 split 불일치")
            for field, locations in seen.items():
                value = record[field]
                if value in locations and locations[value] != name:
                    raise ValueError(f"split 간 {field} 누출")
                locations[value] = name
    for record in manifest["splits"][split]:
        path = _source_path(root, record["path"])
        identity = _identity(record["path"], speakers, chapters)
        if identity != (record["speaker"], record["chapter"], record["book"]):
            raise ValueError("manifest 원천 identity 불일치")
        if _sha(path) != record["sha256"]:
            raise ValueError("원천 오디오 SHA가 변경됐습니다")
        with sf.SoundFile(path, mode="r") as stream:
            if stream.samplerate != 16000 or stream.channels != 1 or len(stream) != record["frames"]:
                raise ValueError("원천 오디오 header 불일치")
            crops = record["crops"]
            if not 1 <= len(crops) <= 2:
                raise ValueError("utterance당 crop은 최대 2개여야 합니다")
            previous_end = 0
            for index, crop in enumerate(crops):
                start, frames = crop["start"], crop["frames"]
                if (type(start) is not int or type(frames) is not int or not 1 <= frames <= 8192
                        or start < previous_end or start + frames > len(stream)):
                    raise ValueError("crop 범위/중복 오류")
                stream.seek(start)
                samples = stream.read(frames, dtype="float32", always_2d=False)
                if samples.shape != (frames,) or not np.isfinite(samples).all():
                    raise ValueError("crop이 잘렸거나 비유한 샘플을 포함합니다")
                previous_end = start + frames
                metadata = {key: value for key, value in record.items() if key != "crops"}
                metadata.update(crop_index=index, crop_start=start, crop_frames=frames,
                                simulation_pretraining_only=True, measured_ref_err=False,
                                amplitude_domain=manifest["amplitude_domain"])
                yield samples, metadata
