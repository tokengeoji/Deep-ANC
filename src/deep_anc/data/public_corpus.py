"""명시적으로 stage한 공개 음원의 그룹 분할·전체 PCM QA (다운로드 없음).

ESC-50/FMA/LibriSpeech는 공식 메타데이터를 읽는다. DEMAND/MIMII는 검토된
source_index.csv와 source_index_meta.json을 요구한다. 후자는 staging 작성자의
목록 완전성 선언이며 외부 출처 진위를 자동으로 인증하는 기능은 아니다.
machine(MIMII)은 원녹음 독립성이 미확인된 학습 보조 전용이다. 공식 분할은
metadata에만 보존하고 ANC split=train 및 보수적인 단일 그룹을 강제한다.
오류가 하나라도 있으면 inventory/QA만 발급하고 학습 manifest는 발급하지 않는다.
새 출력 폴더만 쓰며 --reuse는 내용을 재검증할 뿐 어떤 파일도 변경하지 않는다.
쓰기 중 실패한 새 폴더에는 부분 파일이 남을 수 있으므로 재사용하지 말아야 한다.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import soundfile as sf

from .manifest import assign_splits, manifest_relative_path, validate_group_id

FAMILIES = ("esc50", "music", "speech", "demand", "machine")
SPLITS = ("train", "val", "test")
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3"}
DEFAULT_RATIOS = {"train": 0.9, "val": 0.05, "test": 0.05}
TRAIN_ONLY_FAMILIES = ("machine",)
MACHINE_USAGE_POLICY = "train_only_auxiliary"
MACHINE_GROUP_ID = "machine:mimii-dg-unresolved"
MACHINE_GROUPING_BASIS = "conservative_unresolved_mimii_train_only_pool"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _safe_location(path: str | Path) -> Path:
    candidate = Path(path).absolute()
    # resolve 이전 확인: link/../new 형태의 우회도 거부한다.
    for part in (candidate, *candidate.parents):
        if part.is_symlink():
            raise ValueError(f"symlink 경로는 지원하지 않습니다: {part}")
    return candidate.resolve()


def _relative_source(root: Path, value: str) -> Path:
    rel = PurePosixPath(value)
    if not value or value != value.strip() or "\\" in value or "\x00" in value:
        raise ValueError("path는 비어 있지 않은 POSIX 상대 경로여야 합니다")
    if rel.is_absolute() or any(part in {"..", "."} for part in rel.parts):
        raise ValueError("path 절대 경로/상위 경로는 허용하지 않습니다")
    candidate = _safe_location(root / value)
    candidate.relative_to(root)
    return candidate


class _Audit:
    def __init__(self, raw_root: Path, out: Path):
        self.raw_root = raw_root
        self.out = out
        self.entries: list[dict[str, Any]] = []
        self.issues: list[dict[str, str]] = []
        self.metadata_hashes: dict[str, str] = {}
        self.source_provenance: dict[str, Any] = {}

    def issue(self, family: str, code: str, detail: str, path: Path | None = None) -> None:
        item = {"source_family": family, "code": code, "detail": str(detail)}
        if path is not None:
            item["path"] = str(path.relative_to(self.raw_root))
        self.issues.append(item)

    def metadata(self, path: Path) -> str:
        safe = _safe_location(path)
        safe.relative_to(self.raw_root)
        content = safe.read_bytes()
        # 실제 파싱한 바이트의 digest를 보존한다(종료 시 달라진 metadata 재해시 금지).
        self.metadata_hashes[safe.relative_to(self.raw_root).as_posix()] = hashlib.sha256(content).hexdigest()
        return content.decode("utf-8-sig")

    def one_metadata(self, root: Path, name: str) -> Path:
        matches = sorted(root.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"{name}: 공식 metadata 정확히 1개 필요, 발견 {len(matches)}")
        return matches[0]

    def audio_files(self, root: Path) -> list[Path]:
        result = []
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                self.issue(root.name, "unsafe_symlink", "원본 트리의 링크는 허용하지 않습니다", path)
            elif path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                result.append(path)
        return result

    def add(self, family: str, root: Path, path: Path, group: str,
            metadata: dict[str, Any], split: str | None = None) -> None:
        if family in TRAIN_ONLY_FAMILIES:
            # 이 그룹은 원녹음 ID가 아니라 평가 누수를 피하기 위한 보수적 풀이다.
            # 오류/미등재 행도 val/test 또는 독립 원녹음으로 오해되지 않게 남긴다.
            metadata = dict(metadata)
            recording = metadata.get("official", {}).get("source_recording_id", "")
            if "grouping_basis" in metadata:
                metadata["declared_grouping_basis"] = metadata["grouping_basis"]
            metadata.update({
                "usage_policy": MACHINE_USAGE_POLICY,
                "independent_evaluation_allowed": False,
                "grouping_basis": MACHINE_GROUPING_BASIS,
                "source_recording_id_status": "declared_unverified" if recording.strip() else "unknown",
                "source_recording_independence": "unknown",
            })
            group, split = MACHINE_GROUP_ID, "train"
        entry: dict[str, Any] = {
            "path": manifest_relative_path(path, self.out / f"{family}.jsonl"),
            "path_base": "manifest", "source_family": family, "tag": family,
            "group_id": group, "split": split, "metadata": metadata,
            "qa_valid": False,
        }
        self.entries.append(entry)
        try:
            validate_group_id(group)
            if split is not None and split not in SPLITS:
                raise ValueError(f"official_split가 잘못됐습니다: {split}")
            safe = _safe_location(path)
            safe.relative_to(root)
            if safe.suffix.lower() not in AUDIO_SUFFIXES:
                raise ValueError("지원하지 않는 오디오 확장자")
            entry["sha256"] = sha256_file(safe)
            entry.update(inspect_audio(safe))
            if sha256_file(safe) != entry["sha256"]:
                raise ValueError("PCM 검사 중 원본 파일 내용이 변경됐습니다")
            entry["qa_valid"] = True
        except (ValueError, OSError, RuntimeError, OverflowError) as exc:
            entry["qa_error"] = str(exc)
            self.issue(family, "audio_or_entry_invalid", str(exc), path)

    def compare_inventory(self, family: str, root: Path, expected: list[Path]) -> None:
        seen: set[Path] = set()
        for path in expected:
            if path in seen:
                self.issue(family, "metadata_duplicate_path", "목록에서 같은 파일을 여러 번 지정", path)
            seen.add(path)
        for path in self.audio_files(root):
            if path not in seen:
                self.issue(family, "audio_missing_from_metadata", "공식/검토 목록에 없는 오디오", path)
                self.add(family, root, path, f"{family}:unindexed", {"unindexed": True})


def inspect_audio(path: Path) -> dict[str, Any]:
    """전체 PCM을 streaming 읽기. 밴드 파워는 리샘플 전 mono 평균의 rectangular FFT.

    65536-frame chunk의 Parseval 에너지를 합한다. 분할/창 누설이 있는 진단값이며
    원본의 소음 바닥, 정보량, 실제 ANC 감쇠 또는 유효 신뢰대역을 증명하지 않는다.
    """
    energy = np.zeros(3, dtype=np.float64)
    peak = 0.0
    clipped = 0
    decoded = 0
    with sf.SoundFile(path) as stream:
        frames, sample_rate, channels = len(stream), stream.samplerate, stream.channels
        if frames <= 0 or sample_rate <= 0 or channels <= 0:
            raise ValueError("빈 오디오 또는 잘못된 오디오 헤더")
        for block in stream.blocks(blocksize=65536, dtype="float64", always_2d=True):
            if not np.isfinite(block).all():
                raise ValueError("PCM에 NaN/Inf가 있습니다")
            if block.shape[1] != channels:
                raise ValueError("PCM 채널 수 불일치")
            block_peak = float(np.max(np.abs(block)))
            if block_peak > 1e100:
                raise ValueError("PCM 수치 범위가 진단 파워 계산 한계를 초과합니다")
            peak = max(peak, block_peak)
            clipped += int(np.count_nonzero(np.abs(block) >= 0.999))
            mono = block.mean(axis=1)
            size = len(mono)
            spectrum = np.abs(np.fft.rfft(mono)) ** 2 / size
            weights = np.full(len(spectrum), 2.0)
            weights[0] = 1.0
            if size % 2 == 0:
                weights[-1] = 1.0
            spectrum *= weights
            hz = np.fft.rfftfreq(size, d=1.0 / sample_rate)
            for index, mask in enumerate((hz < 800.0, (hz >= 800.0) & (hz <= 1600.0), hz > 1600.0)):
                energy[index] += float(spectrum[mask].sum())
            decoded += size
        if decoded != frames:
            raise ValueError(f"PCM frame 불일치: header={frames}, decoded={decoded}")
    powers = energy / decoded
    if not np.isfinite(powers).all():
        raise ValueError("PCM 밴드 파워가 유한하지 않습니다")
    total = float(powers.sum())
    if total <= 0.0:
        raise ValueError("완전 무음 파일은 음원 풀로 사용할 수 없습니다")
    return {
        "frames": frames, "duration_s": frames / sample_rate,
        "sample_rate": sample_rate, "channels": channels,
        "pcm_peak_abs": peak, "pcm_near_full_scale_samples": clipped,
        "band_power": {
            "low_0_800_hz": float(powers[0]), "target_800_1600_hz": float(powers[1]),
            "outside_above_1600_hz": float(powers[2]), "total": total,
            "target_supported_by_sample_rate": sample_rate / 2 >= 1600,
            "nyquist_hz": sample_rate / 2,
            "information_above_nyquist": "unavailable",
            "method": "mono_mean_rectangular_fft_65536_frame_chunks_parseval",
            "snr_and_trusted_band": "unknown",
        },
    }


def _csv_rows(text: str, fields: set[str]) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or not fields.issubset(reader.fieldnames):
        raise ValueError(f"metadata 필수 CSV 컬럼 누락: {sorted(fields)}")
    rows = list(reader)
    if not rows:
        raise ValueError("metadata 목록이 비어 있습니다")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV 행의 컬럼 수가 헤더와 다릅니다")
    return rows


def _basename_map(audit: _Audit, root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in audit.audio_files(root):
        if path.name in mapping:
            raise ValueError(f"공식 파일명이 중복되어 경로를 결정할 수 없습니다: {path.name}")
        mapping[path.name] = path
    return mapping


def _esc50(audit: _Audit, root: Path) -> None:
    metadata_path = audit.one_metadata(root, "esc50.csv")
    rows = _csv_rows(audit.metadata(metadata_path), {"filename", "src_file", "fold", "category"})
    audio = _basename_map(audit, root)
    expected = []
    for row in rows:
        filename = row["filename"]
        _relative_source(root, filename)
        if "/" in filename or not row["src_file"].isdigit() or int(row["src_file"]) <= 0:
            raise ValueError("ESC-50 filename/src_file가 잘못됐습니다")
        if row["fold"] not in {"1", "2", "3", "4", "5"} or not row["category"].strip():
            raise ValueError("ESC-50 공식 fold/category가 잘못됐습니다")
        path = audio.get(filename, metadata_path.parent.parent / "audio" / filename)
        expected.append(path)
        audit.add("esc50", root, path, f"esc50:src:{row['src_file']}", {
            "official": row, "grouping_basis": "official_src_file",
            "license": "CC BY-NC 3.0 (ESC-50; 개별 원본 조건은 공식 metadata 참조)",
        })
    audit.compare_inventory("esc50", root, expected)


def _music(audit: _Audit, root: Path) -> None:
    metadata_path = audit.one_metadata(root, "tracks.csv")
    reader = csv.reader(io.StringIO(audit.metadata(metadata_path)))
    try:
        first, second = next(reader), next(reader)
    except StopIteration as exc:
        raise ValueError("FMA tracks.csv multi-index 헤더 누락") from exc
    if len(first) != len(second):
        raise ValueError("FMA 헤더 길이 불일치")
    keys = list(zip(first, second))
    for required in (("artist", "id"), ("set", "subset"), ("track", "license")):
        if required not in keys:
            raise ValueError(f"FMA 공식 metadata 컬럼 누락: {required}")
    audio = _basename_map(audit, root)
    expected = []
    for values in reader:
        if values and values[0] == "track_id":
            continue
        if len(values) != len(keys):
            raise ValueError("FMA 데이터 행 컬럼 수 불일치")
        row = dict(zip(keys, values))
        if row[("set", "subset")] != "small":
            continue
        track_id = int(values[0])
        artist_id = int(row[("artist", "id")])
        if track_id <= 0 or artist_id <= 0 or not row[("track", "license")].strip():
            raise ValueError("FMA track/artist ID 또는 개별 license 누락")
        filename = f"{track_id:06d}.mp3"
        path = audio.get(filename, root / "fma_small" / filename[:3] / filename)
        expected.append(path)
        audit.add("music", root, path, f"music:artist:{artist_id}", {
            "official": {f"{left}.{right}": value for (left, right), value in row.items()},
            "license": row[("track", "license")], "grouping_basis": "official_artist_id",
            "official_split_not_used": row.get(("set", "split"), "unknown"),
        })
    if not expected:
        raise ValueError("FMA small metadata 항목이 없습니다")
    audit.compare_inventory("music", root, expected)


def _speech(audit: _Audit, root: Path) -> None:
    """존재하는 공식 transcript 행을 기대 목록으로 삼아 FLAC와 대조한다.

    transcript만 빠지고 FLAC가 남으면 미등재 오디오로 거부한다. 그러나 chapter
    전체(transcript와 FLAC)가 함께 빠지면 원래 있어야 했다는 별도 완전 목록이
    없어 검출할 수 없다. SPEAKERS.TXT의 metadata/발견된 speaker subset을 검사할
    뿐 공식 archive의 전체 speaker/chapter/utterance 완전성을 인증하지 않는다.
    """
    speakers_file = audit.one_metadata(root, "SPEAKERS.TXT")
    speakers = {}
    for line in audit.metadata(speakers_file).splitlines():
        if not line.strip() or line.lstrip().startswith(";"):
            continue
        fields = [part.strip() for part in line.split("|")]
        if len(fields) < 5 or not fields[0].isdigit():
            raise ValueError("LibriSpeech SPEAKERS.TXT 형식이 잘못됐습니다")
        if fields[0] in speakers:
            raise ValueError("LibriSpeech SPEAKERS.TXT 중복 speaker")
        speakers[fields[0]] = {"sex": fields[1], "subset": fields[2], "minutes": fields[3], "name": fields[4]}
    expected = []
    for subset, split in (("train-clean-100", "train"), ("dev-clean", "val"), ("test-clean", "test")):
        dirs = sorted(path for path in root.rglob(subset) if path.is_dir())
        if len(dirs) != 1:
            audit.issue("speech", "official_subset_missing", f"{subset} 디렉터리 정확히 1개 필요")
            continue
        transcripts = sorted(dirs[0].rglob("*.trans.txt"))
        if not transcripts:
            audit.issue("speech", "official_transcripts_missing", f"{subset}: transcript 없음")
        for transcript in transcripts:
            for line in audit.metadata(transcript).splitlines():
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError("LibriSpeech transcript ID/텍스트 누락")
                utterance, text = parts
                ids = utterance.split("-")
                if len(ids) != 3 or not all(part.isdigit() for part in ids):
                    raise ValueError("LibriSpeech utterance ID가 잘못됐습니다")
                speaker, chapter, _ = ids
                if speaker not in speakers or speakers[speaker]["subset"] != subset:
                    raise ValueError("LibriSpeech speaker 공식 subset 불일치/누락")
                if transcript.parent.name != chapter or transcript.parent.parent.name != speaker:
                    raise ValueError("LibriSpeech speaker/chapter 디렉터리 불일치")
                path = transcript.parent / f"{utterance}.flac"
                expected.append(path)
                audit.add("speech", root, path, f"speech:speaker:{speaker}", {
                    "speaker": speakers[speaker], "utterance_id": utterance,
                    "official_subset": subset, "transcript": text,
                    "grouping_basis": "official_speaker_id", "license": "CC BY 4.0",
                }, split)
    audit.compare_inventory("speech", root, expected)


def _indexed(audit: _Audit, root: Path, family: str) -> None:
    meta = json.loads(audit.metadata(root / "source_index_meta.json"))
    _json_bytes(meta)  # 표준 JSON 외 NaN/Inf를 QA 오류로 남긴다.
    if not isinstance(meta, dict) or meta.get("source_family") != family:
        raise ValueError("source_index_meta source_family 불일치")
    if meta.get("expected_inventory_complete") is not True:
        raise ValueError("expected_inventory_complete=true 선언 없음: 누락 검증 불가")
    for key in ("inventory_origin", "grouping_basis"):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise ValueError(f"source_index_meta {key} 근거 누락")
    audit.source_provenance[family] = {**meta, "external_authenticity_verified": False}
    if family in TRAIN_ONLY_FAMILIES:
        if meta.get("usage_policy") != MACHINE_USAGE_POLICY:
            raise ValueError("machine source_index_meta usage_policy=train_only_auxiliary 명시 필요")
        if "independent_evaluation_allowed" in meta and meta["independent_evaluation_allowed"] is not False:
            raise ValueError("machine independent_evaluation_allowed는 false만 허용합니다")
        audit.source_provenance[family].update({
            "usage_policy": MACHINE_USAGE_POLICY,
            "independent_evaluation_allowed": False,
            "declared_grouping_basis": meta["grouping_basis"],
            "grouping_basis": MACHINE_GROUPING_BASIS,
            "group_id": MACHINE_GROUP_ID,
            "source_recording_independence": "unknown",
        })
    rows = _csv_rows(audit.metadata(root / "source_index.csv"), {
        "path", "group_id", "official_split", "license", "source_recording_id",
    })
    expected = []
    recording_groups = {}
    for row in rows:
        path = _relative_source(root, row["path"])
        group = MACHINE_GROUP_ID if family in TRAIN_ONLY_FAMILIES else f"{family}:{row['group_id']}"
        if not row["license"].strip() or (family not in TRAIN_ONLY_FAMILIES and not row["source_recording_id"].strip()):
            raise ValueError("source_index license/source_recording_id 누락")
        if family in TRAIN_ONLY_FAMILIES and row["official_split"] not in ("", *SPLITS):
            raise ValueError("machine 공식 split metadata가 잘못됐습니다")
        recording = row["source_recording_id"]
        if recording.strip() and recording in recording_groups and recording_groups[recording] != group:
            audit.issue(family, "recording_cross_group", "같은 원본/동시녹음이 여러 group에 있습니다", path)
        if recording.strip():
            recording_groups[recording] = group
        expected.append(path)
        audit.add(family, root, path, group, {
            "official": row, "license": row["license"],
            "grouping_basis": meta["grouping_basis"],
            "inventory_completeness": "staging_author_declaration",
        }, row["official_split"] or None)
    audit.compare_inventory(family, root, expected)


def _audit_corpus(raw_root: Path, out: Path, seed: int, ratios: dict[str, float]) -> tuple[dict[str, Any], dict[str, bytes]]:
    audit = _Audit(raw_root, out)
    for family in FAMILIES:
        root = raw_root / family
        try:
            _safe_location(root)
            if not root.is_dir():
                raise ValueError("source family 디렉터리 없음")
            if family == "esc50":
                _esc50(audit, root)
            elif family == "music":
                _music(audit, root)
            elif family == "speech":
                _speech(audit, root)
            else:
                _indexed(audit, root, family)
        except (ValueError, OSError, RuntimeError, OverflowError) as exc:
            audit.issue(family, "source_metadata_invalid", str(exc))
            # 메타데이터 오류로 중단돼도 나머지 실제 오디오의 읽기 QA를 생략하지 않는다.
            try:
                _safe_location(root)
            except ValueError:
                continue  # root symlink 대상의 디렉터리 목록도 읽지 않는다.
            indexed = {entry["path"] for entry in audit.entries}
            for path in audit.audio_files(root):
                relative = manifest_relative_path(path, out / f"{family}.jsonl")
                if relative not in indexed:
                    audit.add(family, root, path, f"{family}:metadata-unresolved", {"metadata_unresolved": True})

    for path in sorted(raw_root.iterdir()):
        if path.name not in FAMILIES and path.is_dir():
            audit.issue(path.name, "unsupported_source_family", "지원되지 않은 family(DNS 포함)는 준비완료에 포함하지 않음", path)
        elif path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            audit.issue("all", "audio_outside_family", "family 밖의 오디오는 출처 그룹을 결정할 수 없음", path)

    for relative, digest in audit.metadata_hashes.items():
        path = raw_root / relative
        try:
            if sha256_file(_safe_location(path)) != digest:
                raise ValueError("검사 중 metadata 내용이 변경됐습니다")
        except (ValueError, OSError) as exc:
            audit.issue("all", "metadata_changed_during_audit", str(exc), path)

    for family in FAMILIES:
        entries = [entry for entry in audit.entries if entry["source_family"] == family]
        assigned = [entry["split"] is not None for entry in entries]
        if any(assigned) and not all(assigned):
            audit.issue(family, "partial_official_split", "같은 family에서 공식 split 지정/미지정을 섞을 수 없습니다")
        elif entries and not any(assigned):
            try:
                split_entries = assign_splits(entries, ratios, seed=seed)
                for entry, split_entry in zip(entries, split_entries):
                    entry["split"] = split_entry["split"]
            except ValueError as exc:
                audit.issue(family, "group_split_invalid", str(exc))

    groups: dict[str, set[str | None]] = {}
    hashes: dict[str, list[dict[str, Any]]] = {}
    for entry in audit.entries:
        groups.setdefault(entry["group_id"], set()).add(entry["split"])
        if "sha256" in entry:
            hashes.setdefault(entry["sha256"], []).append(entry)
        if entry.get("band_power", {}).get("target_supported_by_sample_rate") is False:
            audit.issue(entry["source_family"], "target_band_unsupported", entry["path"])
    for group, splits in sorted(groups.items()):
        if len(splits) > 1:
            audit.issue("all", "group_cross_split", f"{group}: {sorted(str(split) for split in splits)}")
    for digest, entries in sorted(hashes.items()):
        if len(entries) > 1:
            audit.issue("all", "duplicate_sha256", f"{digest}: " + ", ".join(entry["path"] for entry in entries))
    counts = {family: {split: sum(entry["source_family"] == family and entry["split"] == split and entry["qa_valid"] for entry in audit.entries) for split in SPLITS} for family in FAMILIES}
    for family, splits in counts.items():
        required = ("train",) if family in TRAIN_ONLY_FAMILIES else SPLITS
        for split in required:
            if not splits[split]:
                audit.issue(family, "empty_split", split)
        if family in TRAIN_ONLY_FAMILIES and any(splits[split] for split in ("val", "test")):
            audit.issue(family, "train_only_split_violation", "학습 보조 원본을 독립 평가에 사용할 수 없습니다")
    audit.entries.sort(key=lambda entry: (entry["source_family"], entry["path"]))
    payloads = {"inventory.jsonl": b"".join(_json_bytes(entry) for entry in audit.entries)}
    ready = not audit.issues
    if ready:
        for family in FAMILIES:
            payloads[f"{family}.jsonl"] = b"".join(_json_bytes(entry) for entry in audit.entries if entry["source_family"] == family)
    report = {
        "schema_version": 1, "data_ready": ready, "diagnostic_only": True,
        "performance_claim_allowed": False, "raw_root": str(raw_root),
        "stage_required": True, "network_or_download_performed": False,
        "families": list(FAMILIES), "split_counts": counts, "seed": seed,
        "train_only_source_families": list(TRAIN_ONLY_FAMILIES),
        "split_ratios": ratios, "speech_split_policy": "official_subsets",
        "duplicate_policy": "reject_all_exact_file_hash_duplicates_no_selection",
        "issues": audit.issues, "source_provenance": audit.source_provenance,
        "inventory_completeness": {
            "scope": "staged_metadata_and_present_audio_only",
            "official_archive_completeness_verified": False,
            "speech_expected_files_basis": "present_official_transcript_rows",
            "speech_missing_transcript_with_remaining_audio": "rejected",
            "speech_missing_transcript_and_audio": "unknown_without_external_complete_inventory",
        },
        "source_metadata_sha256": audit.metadata_hashes,
        "inventory_sha256": hashlib.sha256(payloads["inventory.jsonl"]).hexdigest(),
        "manifest_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items() if name != "inventory.jsonl"},
        "limitations": [
            "로컬 stage 자료 검증이며 Drive inventory-ready나 실제 ANC 성능 검증이 아님",
            "대역 파워는 mono downmix/원본 sample rate의 chunk FFT 진단이며 SNR/신뢰대역은 unknown",
            "16 kHz 음원은 8 kHz 이상 정보가 없고 800–1600 Hz 표본화 가능 여부만 검사",
            "원본 전체 해시의 정확한 중복만 검출; 재인코딩·부분복제·출처 그룹 외 누수는 미검증",
            "DEMAND/MIMII 목록 완전성과 그룹 근거는 staging 작성자의 선언; 외부 진위 자동 인증 없음",
            "machine(MIMII)은 단일 미확정 그룹의 train-only 보조 자료이며 원녹음 독립성/평가 성능을 주장하지 않음",
            "LibriSpeech는 존재하는 transcript 기준 대조; transcript와 음원 전체가 함께 빠진 chapter/speaker는 공식 완전 목록 없이는 검출 불가",
        ],
    }
    payloads["qa.json"] = _json_bytes(report)
    return report, payloads


def prepare_public_corpus(raw_root: str | Path, out_dir: str | Path, *, seed: int = 20260915,
                          split_ratios: dict[str, float] | None = None,
                          reuse: bool = False) -> dict[str, Any]:
    """stage된 source tree를 검사하고 새 폴더에 결과를 독점 저장한다.

    data_ready는 관측된 metadata 기준 PCM/그룹 분할 정합이며 공식 archive 전체
    보유의 증명이 아니다. 특히 LibriSpeech의 transcript와 해당 음원이 함께 빠진
    경우는 외부 완전 inventory/검증된 archive 추출 receipt 없이 검출할 수 없다.
    machine(MIMII)은 source_index_meta의 train_only_auxiliary 선언이 필수다.
    machine 원녹음 ID의 빈값은 unknown으로 남기며 공식 split/group은 metadata에
    보존한다. 출력은 모두 train/단일 미확정 그룹이며 독립 평가에 사용할 수 없다.

    반환 data_ready=False는 QA 실패이며 family manifest는 없다. 잘못된 호출/출력
    경로/재사용 불일치는 ValueError/FileExistsError. reuse는 전체 raw PCM까지
    다시 읽고 모든 artifact 바이트를 대조하며 수정·덮어쓰기를 수행하지 않는다.
    """
    raw = _safe_location(raw_root)
    out = _safe_location(out_dir)
    if not raw.is_dir():
        raise ValueError(f"명시적으로 stage한 raw_root 디렉터리가 필요합니다: {raw}")
    if out == raw or raw in out.parents or out in raw.parents:
        raise ValueError("출력 폴더와 raw tree는 서로 포함할 수 없습니다")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed는 0 이상의 정수여야 합니다")
    ratios = dict(DEFAULT_RATIOS if split_ratios is None else split_ratios)
    if set(ratios) != set(SPLITS) or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in ratios.values()):
        raise ValueError("train/val/test 비율은 모두 유한한 양수여야 합니다")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9, rel_tol=0):
        raise ValueError("분할 비율 합은 1이어야 합니다")
    if out.exists() and not reuse:
        raise FileExistsError(f"출력 폴더가 이미 있습니다(덮어쓰기 금지): {out}")
    if reuse and not out.is_dir():
        raise ValueError("--reuse는 기존 결과 폴더가 필요합니다")
    report, payloads = _audit_corpus(raw, out, seed, ratios)
    if reuse:
        actual = {path.name for path in out.iterdir()}
        if actual != set(payloads):
            raise ValueError("재사용 실패: 결과 파일 집합이 현재 QA와 다릅니다")
        for name, content in payloads.items():
            path = _safe_location(out / name)
            if path.read_bytes() != content:
                raise ValueError(f"재사용 실패: raw/metadata/manifest/QA 정합 불일치 ({name})")
        return report
    out.mkdir(parents=True, exist_ok=False)
    for name, content in payloads.items():
        with (out / name).open("xb") as stream:
            stream.write(content)
    return report
