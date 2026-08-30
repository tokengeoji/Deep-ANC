"""BSD35k-CS의 광대역 machine source를 결정적으로 선별한다.

이 모듈은 네트워크나 오디오 장치를 열지 않는다. 공식 metadata CSV의 bytes를 먼저
고정하고, ``fx-m``/CC0 row만 uploader-disjoint train/val/test로 나눈 selection plan을
발행한다. 실제 WAV decode, spectrum, resampling 및 causal-P coverage는 의도적으로
별도 후속 gate다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence


BSD35K_RECORD_ID = 19_187_100
BSD35K_DOI = "10.5281/zenodo.19187100"
BSD35K_MACHINE_SELECTION_SCHEMA = "bsd35k_machine_selection_v1"
BSD35K_MACHINE_SPLIT_ALGORITHM = "largest_first_normalized_l1_then_min4_rebalance_v1"
BSD35K_MACHINE_SPLIT_SEED = "deep-anc-bsd35k-machine-v1"

OFFICIAL_METADATA_CSV_SIZE = 23_570_450
OFFICIAL_METADATA_CSV_SHA256 = (
    "2128c89e39d2024ac015bb7053e07099ec287d89122ea0c084e8a3460ab6d363"
)
OFFICIAL_METADATA_ROW_COUNT = 33_829
OFFICIAL_FX_M_ROW_COUNT = 1_542
OFFICIAL_CC0_FX_M_ROW_COUNT = 1_323
OFFICIAL_CC0_FX_M_UPLOADER_COUNT = 188
OFFICIAL_METADATA_ZIP_SIZE = 4_374_871
OFFICIAL_METADATA_ZIP_MD5 = "9876254ce2ed845691a9a76efe13fe5a"
OFFICIAL_METADATA_ZIP_SHA256 = (
    "b595129c00e65f098bee06aaf442ed454cb30430fadd28461bcdd6628b235a51"
)
OFFICIAL_AUDIO_ZIP_SIZE = 35_091_942_026
OFFICIAL_AUDIO_ZIP_MD5 = "d47968c99ad4e93a081f380b2d273acd"

CC0_LICENSE_URL = "http://creativecommons.org/publicdomain/zero/1.0/"
REQUIRED_COLUMNS = (
    "sound_id",
    "class",
    "class_idx",
    "class_top",
    "confidence",
    "uploader",
    "license",
    "title",
    "tags",
    "description",
)
SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
# 최종 gate의 법적 하한은 4개지만, 4개만 두면 한 uploader가 holdout을 지배해
# cluster-bootstrap이 사실상 무의미해진다. 공개 corpus 단계에서는 16개를 예약한다.
MINIMUM_UPLOADERS_PER_SPLIT = 16
_HEX = frozenset("0123456789abcdef")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in _HEX for character in digest):
        raise ValueError(f"{name}는 lowercase SHA-256이어야 합니다")
    return digest


def _read_metadata(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        if columns != REQUIRED_COLUMNS:
            raise ValueError(
                "BSD35k metadata column 계약이 다릅니다: "
                f"expected={REQUIRED_COLUMNS!r}, actual={columns!r}"
            )
        rows = [dict(row) for row in reader]
    return columns, rows


def _uploader_order_key(uploader: str, count: int) -> tuple[int, str, str]:
    digest = hashlib.sha256(
        f"{BSD35K_MACHINE_SPLIT_SEED}\0{uploader}".encode("utf-8")
    ).hexdigest()
    return (-int(count), digest, uploader)


def _split_assignment(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    uploader_counts: dict[str, int] = {}
    for row in rows:
        uploader = str(row["uploader"])
        uploader_counts[uploader] = uploader_counts.get(uploader, 0) + 1
    if len(uploader_counts) < MINIMUM_UPLOADERS_PER_SPLIT * len(SPLITS):
        raise ValueError("uploader-disjoint 3 split에 필요한 독립 uploader가 부족합니다")

    total = sum(uploader_counts.values())
    targets = {split: fraction * total for split, fraction in SPLIT_FRACTIONS.items()}
    current = {split: 0 for split in SPLITS}
    members: dict[str, list[str]] = {split: [] for split in SPLITS}
    split_priority = {split: index for index, split in enumerate(SPLITS)}

    ordered = sorted(
        uploader_counts.items(),
        key=lambda item: _uploader_order_key(item[0], item[1]),
    )
    for uploader, count in ordered:
        def objective(split: str) -> tuple[float, int]:
            projected = dict(current)
            projected[split] += count
            normalized_l1 = sum(
                abs(projected[name] - targets[name]) / targets[name]
                for name in SPLITS
            )
            return float(normalized_l1), split_priority[split]

        selected = min(SPLITS, key=objective)
        current[selected] += count
        members[selected].append(uploader)

    # 큰 uploader 몇 명만으로 val/test 목표 clip 수가 채워져도 lineage 통계에는 최소
    # 네 component가 필요하다. train의 가장 작은 component를 결정적으로 이동한다.
    for destination in ("val", "test"):
        while len(members[destination]) < MINIMUM_UPLOADERS_PER_SPLIT:
            if len(members["train"]) <= MINIMUM_UPLOADERS_PER_SPLIT:
                raise ValueError("minimum uploader rebalance 후 train component가 부족합니다")

            def rebalance_key(uploader: str) -> tuple[int, str, str]:
                digest = hashlib.sha256(
                    (
                        f"{BSD35K_MACHINE_SPLIT_SEED}\0rebalance\0"
                        f"{destination}\0{uploader}"
                    ).encode("utf-8")
                ).hexdigest()
                return uploader_counts[uploader], digest, uploader

            moved = min(members["train"], key=rebalance_key)
            members["train"].remove(moved)
            members[destination].append(moved)
            current["train"] -= uploader_counts[moved]
            current[destination] += uploader_counts[moved]

    assignment = {
        uploader: split
        for split in SPLITS
        for uploader in members[split]
    }
    if set(assignment) != set(uploader_counts):
        raise AssertionError("uploader split assignment가 전단사가 아닙니다")
    return assignment


def _row_sha256(row: Mapping[str, str]) -> str:
    return _json_sha256({column: str(row[column]) for column in REQUIRED_COLUMNS})


def _build_selection_payload(
    *,
    metadata_size: int,
    metadata_sha256: str,
    rows: Sequence[Mapping[str, str]],
    expected_row_count: int,
    expected_fx_m_count: int,
    expected_cc0_count: int,
    expected_uploader_count: int,
) -> dict[str, Any]:
    if len(rows) != int(expected_row_count):
        raise ValueError(
            f"BSD35k metadata row count 불일치: {len(rows)} != {expected_row_count}"
        )
    identifiers: set[int] = set()
    fx_m_rows: list[Mapping[str, str]] = []
    selected: list[Mapping[str, str]] = []
    for row_index, row in enumerate(rows, start=2):
        if tuple(row) != REQUIRED_COLUMNS:
            raise ValueError(f"metadata row #{row_index} column 계약이 다릅니다")
        raw_identifier = str(row["sound_id"])
        try:
            identifier = int(raw_identifier)
        except ValueError as error:
            raise ValueError(f"sound_id가 정수가 아닙니다: {raw_identifier!r}") from error
        if identifier <= 0 or str(identifier) != raw_identifier:
            raise ValueError(f"sound_id canonical decimal 계약 위반: {raw_identifier!r}")
        if identifier in identifiers:
            raise ValueError(f"중복 sound_id: {identifier}")
        identifiers.add(identifier)
        if str(row["class"]) != "fx-m":
            continue
        fx_m_rows.append(row)
        if str(row["license"]) == CC0_LICENSE_URL:
            if not str(row["uploader"]).strip():
                raise ValueError(f"CC0 fx-m uploader가 비었습니다: sound_id={identifier}")
            selected.append(row)

    if len(fx_m_rows) != int(expected_fx_m_count):
        raise ValueError(f"fx-m row count 불일치: {len(fx_m_rows)} != {expected_fx_m_count}")
    if len(selected) != int(expected_cc0_count):
        raise ValueError(f"CC0 fx-m row count 불일치: {len(selected)} != {expected_cc0_count}")
    uploaders = {str(row["uploader"]) for row in selected}
    if len(uploaders) != int(expected_uploader_count):
        raise ValueError(
            "CC0 fx-m uploader count 불일치: "
            f"{len(uploaders)} != {expected_uploader_count}"
        )

    assignment = _split_assignment(selected)
    entries = []
    for row in sorted(selected, key=lambda item: int(item["sound_id"])):
        sound_id = int(row["sound_id"])
        uploader = str(row["uploader"])
        entries.append(
            {
                "sound_id": sound_id,
                "uploader": uploader,
                "lineage_group": f"bsd35k_uploader:{uploader}",
                "split": assignment[uploader],
                "class": "fx-m",
                "license": CC0_LICENSE_URL,
                "archive_member": f"audio/{sound_id}.wav",
                "metadata_row_sha256": _row_sha256(row),
                "audio_file_sha256": None,
                "decoded_pcm_sha256": None,
                "native_sample_rate": None,
                "native_channels": None,
                "native_sample_width_bits": None,
                "native_octave_density": None,
                "causal_primary_err_octave_density": None,
                "audio_status": "NOT_VERIFIED",
            }
        )

    split_summary = {}
    for split in SPLITS:
        split_entries = [entry for entry in entries if entry["split"] == split]
        split_uploaders = sorted({str(entry["uploader"]) for entry in split_entries})
        if len(split_uploaders) < MINIMUM_UPLOADERS_PER_SPLIT:
            raise AssertionError(f"{split} independent uploader가 부족합니다")
        split_summary[split] = {
            "clip_count": len(split_entries),
            "uploader_count": len(split_uploaders),
            "uploader_set_sha256": _json_sha256(split_uploaders),
        }
    uploader_sets = {
        split: {
            str(entry["uploader"])
            for entry in entries
            if entry["split"] == split
        }
        for split in SPLITS
    }
    if any(
        uploader_sets[left] & uploader_sets[right]
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    ):
        raise AssertionError("uploader lineage가 split을 넘었습니다")

    payload: dict[str, Any] = {
        "schema_version": BSD35K_MACHINE_SELECTION_SCHEMA,
        "record_id": BSD35K_RECORD_ID,
        "doi": BSD35K_DOI,
        "source_record_url": f"https://zenodo.org/records/{BSD35K_RECORD_ID}",
        "metadata_csv": {
            # host absolute path를 selection identity에 넣지 않는다. actual local bytes는
            # size/SHA가 결속하고 verifier가 caller path에서 다시 읽는다.
            "path": "metadata/BSD35k-CS_metadata.csv",
            "size": int(metadata_size),
            "sha256": _require_sha256(metadata_sha256, "metadata CSV SHA-256"),
            "row_count": len(rows),
            "columns": list(REQUIRED_COLUMNS),
        },
        "official_archives": {
            "metadata_zip": {
                "size": OFFICIAL_METADATA_ZIP_SIZE,
                "md5": OFFICIAL_METADATA_ZIP_MD5,
                "sha256": OFFICIAL_METADATA_ZIP_SHA256,
            },
            "audio_zip": {
                "size": OFFICIAL_AUDIO_ZIP_SIZE,
                "md5": OFFICIAL_AUDIO_ZIP_MD5,
                "sha256": None,
                "status": "NOT_DOWNLOADED_OR_VERIFIED",
            },
        },
        "selection": {
            "class": "fx-m",
            "license": CC0_LICENSE_URL,
            "split_algorithm": BSD35K_MACHINE_SPLIT_ALGORITHM,
            "split_seed": BSD35K_MACHINE_SPLIT_SEED,
            "minimum_uploaders_per_split": MINIMUM_UPLOADERS_PER_SPLIT,
            "fx_m_row_count": len(fx_m_rows),
            "selected_clip_count": len(entries),
            "selected_uploader_count": len(uploaders),
            "split_summary": split_summary,
        },
        "entries": entries,
        "authority": {
            "metadata_and_lineage_selection_passed": True,
            "actual_audio_archive_verified": False,
            "actual_wav_decode_passed": False,
            "native_44100_hz_16bit_mono_verified": False,
            "v3_exact_octave_density_passed": False,
            "v3_causal_primary_err_density_passed": False,
            "canonical_source_eligible": False,
            "blockers": [
                "official_audio_zip_checksum_absent",
                "selected_actual_wav_decode_receipts_absent",
                "native_exact_octave_density_absent",
                "v3_causal_primary_operator_and_err_density_absent",
            ],
        },
    }
    payload["selection_plan_sha256"] = _json_sha256(payload)
    return payload


def build_official_bsd35k_machine_selection(
    metadata_csv: str | Path,
) -> dict[str, Any]:
    """공식 record 19187100 metadata bytes에서만 selection plan을 만든다."""

    lexical = Path(metadata_csv).expanduser()
    if lexical.is_symlink():
        raise ValueError("BSD35k metadata symlink를 거부합니다")
    path = lexical.resolve(strict=True)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError("BSD35k metadata는 symlink가 아닌 regular file이어야 합니다")
    if int(info.st_size) != OFFICIAL_METADATA_CSV_SIZE:
        raise ValueError(
            f"공식 metadata CSV size가 다릅니다: {info.st_size} != "
            f"{OFFICIAL_METADATA_CSV_SIZE}"
        )
    digest = _file_sha256(path)
    if digest != OFFICIAL_METADATA_CSV_SHA256:
        raise ValueError(
            "공식 metadata CSV SHA-256이 다릅니다: "
            f"{digest} != {OFFICIAL_METADATA_CSV_SHA256}"
        )
    _, rows = _read_metadata(path)
    return _build_selection_payload(
        metadata_size=int(info.st_size),
        metadata_sha256=digest,
        rows=rows,
        expected_row_count=OFFICIAL_METADATA_ROW_COUNT,
        expected_fx_m_count=OFFICIAL_FX_M_ROW_COUNT,
        expected_cc0_count=OFFICIAL_CC0_FX_M_ROW_COUNT,
        expected_uploader_count=OFFICIAL_CC0_FX_M_UPLOADER_COUNT,
    )


def validate_bsd35k_machine_selection(plan: Mapping[str, Any]) -> None:
    payload = dict(plan)
    if payload.get("schema_version") != BSD35K_MACHINE_SELECTION_SCHEMA:
        raise ValueError("BSD35k machine selection schema가 다릅니다")
    claimed = _require_sha256(
        payload.pop("selection_plan_sha256", None), "selection_plan_sha256"
    )
    if _json_sha256(payload) != claimed:
        raise ValueError("BSD35k machine selection plan SHA가 다릅니다")
    if payload.get("record_id") != BSD35K_RECORD_ID or payload.get("doi") != BSD35K_DOI:
        raise ValueError("BSD35k official record identity가 다릅니다")
    metadata = payload.get("metadata_csv")
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata_csv receipt가 없습니다")
    if (
        int(metadata.get("size", -1)) != OFFICIAL_METADATA_CSV_SIZE
        or metadata.get("sha256") != OFFICIAL_METADATA_CSV_SHA256
        or int(metadata.get("row_count", -1)) != OFFICIAL_METADATA_ROW_COUNT
        or tuple(metadata.get("columns", ())) != REQUIRED_COLUMNS
    ):
        raise ValueError("official metadata receipt가 다릅니다")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != OFFICIAL_CC0_FX_M_ROW_COUNT:
        raise ValueError("selected entry count가 다릅니다")
    sound_ids: set[int] = set()
    uploader_split: dict[str, str] = {}
    split_uploaders = {split: set() for split in SPLITS}
    if [int(entry.get("sound_id", -1)) for entry in entries] != sorted(
        int(entry.get("sound_id", -1)) for entry in entries
    ):
        raise ValueError("selection entry가 sound_id 순서가 아닙니다")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("selection entry가 mapping이 아닙니다")
        identifier = int(entry.get("sound_id", -1))
        uploader = str(entry.get("uploader", ""))
        split = str(entry.get("split", ""))
        if identifier <= 0 or identifier in sound_ids:
            raise ValueError("selection sound_id가 중복/비정상입니다")
        sound_ids.add(identifier)
        if split not in SPLITS or not uploader:
            raise ValueError("selection uploader/split이 잘못됐습니다")
        previous = uploader_split.setdefault(uploader, split)
        if previous != split:
            raise ValueError("한 uploader가 여러 split에 존재합니다")
        split_uploaders[split].add(uploader)
        if (
            entry.get("archive_member") != f"audio/{identifier}.wav"
            or entry.get("class") != "fx-m"
            or entry.get("license") != CC0_LICENSE_URL
        ):
            raise ValueError("selection archive/class/license identity가 다릅니다")
        _require_sha256(entry.get("metadata_row_sha256"), "metadata_row_sha256")
        if entry.get("lineage_group") != f"bsd35k_uploader:{uploader}":
            raise ValueError("selection lineage group이 uploader에서 유도되지 않았습니다")
        if entry.get("audio_status") != "NOT_VERIFIED" or any(
            entry.get(key) is not None
            for key in (
                "audio_file_sha256",
                "decoded_pcm_sha256",
                "native_sample_rate",
                "native_channels",
                "native_sample_width_bits",
                "native_octave_density",
                "causal_primary_err_octave_density",
            )
        ):
            raise ValueError("selection plan은 actual audio PASS를 주장할 수 없습니다")
    if len(uploader_split) != OFFICIAL_CC0_FX_M_UPLOADER_COUNT:
        raise ValueError("selected uploader count가 다릅니다")
    if any(len(split_uploaders[split]) < MINIMUM_UPLOADERS_PER_SPLIT for split in SPLITS):
        raise ValueError("split별 independent uploader가 부족합니다")
    expected_assignment = _split_assignment(
        [
            {"uploader": str(entry["uploader"])}
            for entry in entries
        ]
    )
    if any(
        expected_assignment[str(entry["uploader"])] != entry["split"]
        for entry in entries
    ):
        raise ValueError("selection split이 deterministic algorithm 재계산과 다릅니다")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("selection summary가 없습니다")
    if (
        selection.get("class") != "fx-m"
        or selection.get("license") != CC0_LICENSE_URL
        or selection.get("split_algorithm") != BSD35K_MACHINE_SPLIT_ALGORITHM
        or selection.get("split_seed") != BSD35K_MACHINE_SPLIT_SEED
        or int(selection.get("minimum_uploaders_per_split", -1))
        != MINIMUM_UPLOADERS_PER_SPLIT
        or int(selection.get("fx_m_row_count", -1)) != OFFICIAL_FX_M_ROW_COUNT
        or int(selection.get("selected_clip_count", -1))
        != OFFICIAL_CC0_FX_M_ROW_COUNT
        or int(selection.get("selected_uploader_count", -1))
        != OFFICIAL_CC0_FX_M_UPLOADER_COUNT
    ):
        raise ValueError("selection summary identity/count가 다릅니다")
    summary = selection.get("split_summary")
    if not isinstance(summary, Mapping) or set(summary) != set(SPLITS):
        raise ValueError("split summary shape가 다릅니다")
    for split in SPLITS:
        row = summary[split]
        if not isinstance(row, Mapping):
            raise ValueError("split summary row가 mapping이 아닙니다")
        selected_uploaders = sorted(split_uploaders[split])
        selected_clip_count = sum(entry["split"] == split for entry in entries)
        if (
            int(row.get("clip_count", -1)) != selected_clip_count
            or int(row.get("uploader_count", -1)) != len(selected_uploaders)
            or row.get("uploader_set_sha256") != _json_sha256(selected_uploaders)
        ):
            raise ValueError(f"{split} summary 재계산이 다릅니다")
    authority = payload.get("authority")
    if not isinstance(authority, Mapping) or authority.get("canonical_source_eligible") is not False:
        raise ValueError("metadata-only plan은 canonical source eligible일 수 없습니다")
    if authority.get("metadata_and_lineage_selection_passed") is not True:
        raise ValueError("metadata selection PASS가 없습니다")
    for key in (
        "actual_audio_archive_verified",
        "actual_wav_decode_passed",
        "native_44100_hz_16bit_mono_verified",
        "v3_exact_octave_density_passed",
        "v3_causal_primary_err_density_passed",
    ):
        if authority.get(key) is not False:
            raise ValueError(f"metadata-only authority가 {key}=false가 아닙니다")
    if authority.get("blockers") != [
        "official_audio_zip_checksum_absent",
        "selected_actual_wav_decode_receipts_absent",
        "native_exact_octave_density_absent",
        "v3_causal_primary_operator_and_err_density_absent",
    ]:
        raise ValueError("metadata-only blocker 집합이 다릅니다")


def verify_bsd35k_machine_selection_against_metadata(
    plan: Mapping[str, Any], metadata_csv: str | Path
) -> None:
    """plan을 official metadata에서 다시 만들어 byte-semantic exact 비교한다."""

    validate_bsd35k_machine_selection(plan)
    rebuilt = build_official_bsd35k_machine_selection(metadata_csv)
    if _canonical_json_bytes(rebuilt) != _canonical_json_bytes(dict(plan)):
        raise ValueError("selection plan이 official metadata 재생성과 다릅니다")


def write_bsd35k_machine_selection_exclusive(
    target: str | Path, plan: Mapping[str, Any]
) -> tuple[Path, str]:
    """검증된 canonical JSON을 O_EXCL+fsync로 한 번만 쓴다."""

    validate_bsd35k_machine_selection(plan)
    path = Path(target).expanduser()
    if not path.is_absolute():
        path = Path(os.path.abspath(path))
    cursor = path.parent
    missing: list[Path] = []
    while not cursor.exists():
        if cursor.is_symlink():
            raise ValueError(f"output parent symlink를 거부합니다: {cursor}")
        missing.append(cursor)
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    if cursor.is_symlink():
        raise ValueError(f"output parent symlink를 거부합니다: {cursor}")
    for directory in reversed(missing):
        directory.mkdir()
    cursor = path.parent
    while True:
        if cursor.is_symlink():
            raise ValueError(f"output parent symlink를 거부합니다: {cursor}")
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    if path.is_symlink():
        raise ValueError(f"output target symlink를 거부합니다: {path}")
    data = _canonical_json_bytes(dict(plan))
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path, hashlib.sha256(data).hexdigest()


__all__ = [
    "BSD35K_MACHINE_SELECTION_SCHEMA",
    "OFFICIAL_METADATA_CSV_SHA256",
    "build_official_bsd35k_machine_selection",
    "validate_bsd35k_machine_selection",
    "verify_bsd35k_machine_selection_against_metadata",
    "write_bsd35k_machine_selection_exclusive",
]
