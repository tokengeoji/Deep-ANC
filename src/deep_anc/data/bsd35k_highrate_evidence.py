"""BSD35k ``fx-m``의 full-octave native machine source evidence.

이 모듈은 16 kHz MIMII를 48 kHz로 올려 8 kHz octave source라고 부르는
경로를 막기 위한 *source-stage* gate다. ``BSD35k-CS`` 공식 metadata에서 이미
결정된 CC0 ``fx-m`` selection을 출발점으로 다음 네 증거를 같은 JSON에 묶는다.

* official audio.zip의 exact size/MD5와 selected archive member → extracted WAV bytes
* official metadata CSV에서 재검산한 uploader-disjoint selection plan → split/lineage component
* 현재 decoder runtime의 complete audit → selected WAV의 full-decode accept
* native WAV의 deterministic PSD windows → split×band uploader coverage

이는 physical P, P-applied ERR, population authority 또는 training authority가 아니다.
``canonical_training_eligible``은 항상 ``false``이며, 뒤 단계가 이 evidence만으로
full-octave 학습을 열 수 없다. 반대로 이 evidence가 없으면 Elice full-octave
preflight는 high-rate machine source가 준비됐다고 주장할 수 없다.

오디오 장치·네트워크는 열지 않는다. 파일을 읽고 decoder를 통해 native WAV를 분석할
뿐이다.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

from ..dsp.control_band_contract import BroadbandFullOctaveContractV3
from . import bsd35k_machine as bsd
from .broadband_coverage_receipt import (
    MIN_TARGET_D_DENSITY_RATIO,
)
from .broadband_population_contract_v3 import density_ratios_v3
from .decoder_audit import validate_audit_report_self_digest
from .holdout_contract import (
    FileSnapshot,
    HoldoutContractError,
    read_regular_file_snapshot,
    reject_symlink_components,
)
from .manifest_contract import read_decoder_audit


BSD35K_HIGHRATE_EVIDENCE_SCHEMA = "bsd35k_fx_m_highrate_source_evidence_v1"
BSD35K_HIGHRATE_EVIDENCE_ROLE = "full_octave_native_machine_source_prepopulation"
PSD_RECIPE_SCHEMA = "bsd35k_fx_m_native_psd_windows_v1"
PSD_WINDOW_SELECTION = "fixed_start_mid_end_max_density_v1"
MIN_PSD_WINDOW_FRAMES = 16_384
MAX_PSD_WINDOW_FRAMES = 65_536
MIN_INDEPENDENT_UPLOADERS_PER_SPLIT_BAND = 4
REQUIRED_SPLITS = bsd.SPLITS
_HEX = frozenset("0123456789abcdef")


def _full_octave_contract() -> BroadbandFullOctaveContractV3:
    return BroadbandFullOctaveContractV3.canonical()


def _minimum_native_sample_rate_hz() -> int:
    """8 kHz octave 상단을 native Nyquist가 포함하기 위한 정수 하한 (22628 Hz)."""

    return int(math.ceil(2.0 * _full_octave_contract().required_excitation_upper_hz))


class BSD35kHighRateEvidenceError(ValueError):
    """BSD35k native-source evidence가 full-octave source gate를 통과하지 못함."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object, *, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise BSD35kHighRateEvidenceError(f"{label}는 lowercase SHA-256이어야 합니다")
    return text


def _md5(value: object, *, label: str) -> str:
    text = str(value or "")
    if len(text) != 32 or any(character not in _HEX for character in text):
        raise BSD35kHighRateEvidenceError(f"{label}는 lowercase MD5여야 합니다")
    return text


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BSD35kHighRateEvidenceError(f"{label}는 양의 정수여야 합니다")
    return int(value)


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise BSD35kHighRateEvidenceError(f"{label}는 finite number여야 합니다")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BSD35kHighRateEvidenceError(f"{label}는 finite number여야 합니다") from exc
    if not math.isfinite(number):
        raise BSD35kHighRateEvidenceError(f"{label}는 finite number여야 합니다")
    return number


def _exact(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise BSD35kHighRateEvidenceError(
            f"{label} key 집합이 정확하지 않습니다: {actual}"
        )
    return value


def _relative_posix(value: object, *, label: str) -> str:
    text = str(value or "")
    if not text or "\\" in text or PureWindowsPath(text).is_absolute():
        raise BSD35kHighRateEvidenceError(
            f"{label}는 정규화된 repository-relative POSIX path여야 합니다"
        )
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise BSD35kHighRateEvidenceError(
            f"{label}는 정규화된 repository-relative POSIX path여야 합니다"
        )
    normalised = path.as_posix()
    if normalised != text:
        raise BSD35kHighRateEvidenceError(f"{label} path 정규화가 다릅니다: {text!r}")
    return normalised


def _repo_root(value: str | Path) -> Path:
    root = Path(value).resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise BSD35kHighRateEvidenceError("repository root는 symlink 아닌 directory여야 합니다")
    return root


def _inside_root(root: Path, relative: str, *, label: str) -> Path:
    path = root / Path(_relative_posix(relative, label=label))
    try:
        return reject_symlink_components(path, root=root)
    except HoldoutContractError as exc:
        raise BSD35kHighRateEvidenceError(f"{label} 경로 계약 위반: {exc}") from exc


def _relative_to_root(root: Path, path: Path, *, label: str) -> str:
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(root).as_posix()
    except ValueError as exc:
        raise BSD35kHighRateEvidenceError(f"{label}가 repository 밖입니다: {path}") from exc


def _snapshot_file(root: Path, path: Path, *, label: str) -> FileSnapshot:
    try:
        return read_regular_file_snapshot(
            path,
            root=root,
            label=label,
            capture_bytes=True,
        )
    except HoldoutContractError as exc:
        raise BSD35kHighRateEvidenceError(f"{label} snapshot 실패: {exc}") from exc


def _reference_from_snapshot(root: Path, snapshot: FileSnapshot, *, label: str) -> dict[str, Any]:
    return {
        "path": _relative_to_root(root, snapshot.path, label=label),
        "size_bytes": int(snapshot.size),
        "sha256": snapshot.sha256,
    }


def _validate_file_reference(
    root: Path,
    value: object,
    *,
    label: str,
) -> tuple[dict[str, Any], FileSnapshot]:
    entry = _exact(value, {"path", "size_bytes", "sha256"}, label=label)
    relative = _relative_posix(entry["path"], label=f"{label}.path")
    expected_size = _positive_int(entry["size_bytes"], label=f"{label}.size_bytes")
    expected_sha = _sha256(entry["sha256"], label=f"{label}.sha256")
    path = _inside_root(root, relative, label=f"{label}.path")
    snapshot = _snapshot_file(root, path, label=label)
    if snapshot.size != expected_size or snapshot.sha256 != expected_sha:
        raise BSD35kHighRateEvidenceError(f"{label} local bytes가 reference와 다릅니다")
    return (
        {"path": relative, "size_bytes": expected_size, "sha256": expected_sha},
        snapshot,
    )


def _load_json(snapshot: FileSnapshot, *, label: str) -> dict[str, Any]:
    assert snapshot.data is not None

    def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise BSD35kHighRateEvidenceError(f"{label} JSON duplicate key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(snapshot.data.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BSD35kHighRateEvidenceError(f"{label}는 valid UTF-8 JSON이 아닙니다") from exc
    if not isinstance(payload, dict):
        raise BSD35kHighRateEvidenceError(f"{label} JSON 최상위는 object여야 합니다")
    return payload


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - official Zenodo checksum comparison only.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_selected_members(
    archive_path: Path,
    expected_members: Sequence[str],
) -> dict[str, tuple[int, str]]:
    """선택한 official ZIP member를 same bytes digest로 읽는다.

    extraction directory만 신뢰하면 다른 WAV를 같은 이름으로 놓는 경로가 남는다.
    issuer는 archive의 선택 member와 extracted raw bytes를 직접 대조한다.
    """

    wanted = set(expected_members)
    if len(wanted) != len(expected_members):
        raise BSD35kHighRateEvidenceError("selection archive member가 중복됩니다")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos: dict[str, zipfile.ZipInfo] = {}
            for info in archive.infolist():
                raw = info.filename
                path = PurePosixPath(raw)
                if (
                    not raw
                    or "\\" in raw
                    or path.is_absolute()
                    or any(part in {"", ".", ".."} for part in path.parts)
                ):
                    raise BSD35kHighRateEvidenceError(
                        f"official archive에 unsafe member가 있습니다: {raw!r}"
                    )
                if raw not in wanted:
                    continue
                if raw in infos:
                    raise BSD35kHighRateEvidenceError(
                        f"official archive의 selected member가 중복됩니다: {raw}"
                    )
                if info.is_dir() or info.file_size <= 0:
                    raise BSD35kHighRateEvidenceError(
                        f"official archive selected member가 regular audio가 아닙니다: {raw}"
                    )
                infos[raw] = info
            missing = sorted(wanted.difference(infos))
            if missing:
                raise BSD35kHighRateEvidenceError(
                    f"official archive에 selected member가 없습니다: {missing[:3]}"
                )
            result: dict[str, tuple[int, str]] = {}
            for member in expected_members:
                info = infos[member]
                digest = hashlib.sha256()
                total = 0
                with archive.open(info, "r") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        total += len(chunk)
                        digest.update(chunk)
                if total != info.file_size:
                    raise BSD35kHighRateEvidenceError(
                        f"official archive member 크기가 불안정합니다: {member}"
                    )
                result[member] = (total, digest.hexdigest())
            return result
    except zipfile.BadZipFile as exc:
        raise BSD35kHighRateEvidenceError("BSD35k official audio archive ZIP가 손상됐습니다") from exc


def _window_starts(frames: int, window_frames: int) -> tuple[int, ...]:
    final = int(frames) - int(window_frames)
    return tuple(sorted({0, final // 2, final}))


def _native_header_and_psd(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """native PCM header와 deterministic 3-window PSD density를 재계산한다."""

    try:
        info = sf.info(str(path))
    except RuntimeError as exc:
        raise BSD35kHighRateEvidenceError(f"BSD35k WAV header를 열 수 없습니다: {path}") from exc
    contract = _full_octave_contract()
    minimum_rate = _minimum_native_sample_rate_hz()
    sample_rate = int(info.samplerate)
    frames = int(info.frames)
    channels = int(info.channels)
    if str(info.format or "") != "WAV" or str(info.subtype or "") != "PCM_16":
        raise BSD35kHighRateEvidenceError(
            f"BSD35k native source는 WAV/PCM_16이어야 합니다: {path} "
            f"({info.format}/{info.subtype})"
        )
    if channels != 1:
        raise BSD35kHighRateEvidenceError(
            f"BSD35k native source는 mono여야 합니다: {path} channels={channels}"
        )
    if sample_rate < minimum_rate:
        raise BSD35kHighRateEvidenceError(
            f"BSD35k native sample rate가 full-octave 하한보다 낮습니다: "
            f"{sample_rate} < {minimum_rate}"
        )
    if frames < MIN_PSD_WINDOW_FRAMES:
        raise BSD35kHighRateEvidenceError(
            f"BSD35k source가 deterministic PSD 최소 window보다 짧습니다: "
            f"{frames} < {MIN_PSD_WINDOW_FRAMES}"
        )
    window_frames = min(MAX_PSD_WINDOW_FRAMES, frames)
    starts = _window_starts(frames, window_frames)
    matrices: list[list[float]] = []
    try:
        with sf.SoundFile(str(path), mode="r") as stream:
            for start in starts:
                stream.seek(start)
                values = stream.read(
                    frames=window_frames,
                    dtype="float64",
                    always_2d=False,
                )
                signal = np.asarray(values, dtype=np.float64)
                if signal.shape != (window_frames,) or not np.all(np.isfinite(signal)):
                    raise BSD35kHighRateEvidenceError(
                        f"BSD35k PSD window decode가 mono finite exact frame이 아닙니다: {path}"
                    )
                matrices.append(
                    list(
                        density_ratios_v3(
                            signal,
                            sample_rate=sample_rate,
                            bands_hz=contract.equal_weight_octave_objective_bands_hz,
                        )
                    )
                )
    except RuntimeError as exc:
        raise BSD35kHighRateEvidenceError(f"BSD35k PSD window decode 실패: {path}") from exc
    maxima = [max(row[index] for row in matrices) for index in range(7)]
    valid = [value >= MIN_TARGET_D_DENSITY_RATIO for value in maxima]
    header = {
        "format": "WAV",
        "subtype": "PCM_16",
        "native_sample_rate_hz": sample_rate,
        "native_nyquist_hz": sample_rate / 2.0,
        "channels": channels,
        "frames": frames,
    }
    psd = {
        "schema": PSD_RECIPE_SCHEMA,
        "window_selection": PSD_WINDOW_SELECTION,
        "minimum_window_frames": MIN_PSD_WINDOW_FRAMES,
        "maximum_window_frames": MAX_PSD_WINDOW_FRAMES,
        "window_frames": window_frames,
        "window_starts": list(starts),
        "density_ratios_by_window_7": matrices,
        "max_density_ratios_7": maxima,
        "valid_bands_7": valid,
    }
    return header, psd


def _require_native_header(value: object, *, label: str) -> dict[str, Any]:
    entry = _exact(
        value,
        {
            "format",
            "subtype",
            "native_sample_rate_hz",
            "native_nyquist_hz",
            "channels",
            "frames",
        },
        label=label,
    )
    contract = _full_octave_contract()
    rate = _positive_int(entry["native_sample_rate_hz"], label=f"{label}.rate")
    if rate < _minimum_native_sample_rate_hz():
        raise BSD35kHighRateEvidenceError(f"{label} native sample rate가 22628 Hz 미만입니다")
    nyquist = _finite(entry["native_nyquist_hz"], label=f"{label}.nyquist")
    if not math.isclose(nyquist, rate / 2.0, abs_tol=1.0e-9) or nyquist < float(
        contract.required_excitation_upper_hz
    ):
        raise BSD35kHighRateEvidenceError(f"{label} native Nyquist가 8 kHz octave를 덮지 않습니다")
    if entry["format"] != "WAV" or entry["subtype"] != "PCM_16":
        raise BSD35kHighRateEvidenceError(f"{label}는 WAV/PCM_16이어야 합니다")
    if _positive_int(entry["channels"], label=f"{label}.channels") != 1:
        raise BSD35kHighRateEvidenceError(f"{label} channels는 1이어야 합니다")
    _positive_int(entry["frames"], label=f"{label}.frames")
    return {
        "format": "WAV",
        "subtype": "PCM_16",
        "native_sample_rate_hz": rate,
        "native_nyquist_hz": nyquist,
        "channels": 1,
        "frames": int(entry["frames"]),
    }


def _require_psd(value: object, *, label: str) -> dict[str, Any]:
    entry = _exact(
        value,
        {
            "schema",
            "window_selection",
            "minimum_window_frames",
            "maximum_window_frames",
            "window_frames",
            "window_starts",
            "density_ratios_by_window_7",
            "max_density_ratios_7",
            "valid_bands_7",
        },
        label=label,
    )
    if entry["schema"] != PSD_RECIPE_SCHEMA or entry["window_selection"] != PSD_WINDOW_SELECTION:
        raise BSD35kHighRateEvidenceError(f"{label} PSD recipe가 canonical 값이 아닙니다")
    if (
        entry["minimum_window_frames"] != MIN_PSD_WINDOW_FRAMES
        or entry["maximum_window_frames"] != MAX_PSD_WINDOW_FRAMES
    ):
        raise BSD35kHighRateEvidenceError(f"{label} PSD window 정책이 canonical 값이 아닙니다")
    frames = _positive_int(entry["window_frames"], label=f"{label}.window_frames")
    if not MIN_PSD_WINDOW_FRAMES <= frames <= MAX_PSD_WINDOW_FRAMES:
        raise BSD35kHighRateEvidenceError(f"{label} PSD window frame 범위가 잘못됐습니다")
    starts_raw = entry["window_starts"]
    if not isinstance(starts_raw, list) or not starts_raw:
        raise BSD35kHighRateEvidenceError(f"{label}.window_starts가 비었습니다")
    starts = [_positive_or_zero(value, label=f"{label}.window_start") for value in starts_raw]
    if starts != sorted(set(starts)) or len(starts) > 3:
        raise BSD35kHighRateEvidenceError(f"{label}.window_starts는 canonical 1~3개 정렬값이어야 합니다")
    matrices_raw = entry["density_ratios_by_window_7"]
    if not isinstance(matrices_raw, list) or len(matrices_raw) != len(starts):
        raise BSD35kHighRateEvidenceError(f"{label} PSD row 수가 window start와 다릅니다")
    matrices: list[list[float]] = []
    for row_index, raw in enumerate(matrices_raw):
        if not isinstance(raw, list) or len(raw) != 7:
            raise BSD35kHighRateEvidenceError(f"{label} PSD row #{row_index}는 7 bands여야 합니다")
        row = [_finite(item, label=f"{label} PSD value") for item in raw]
        if any(item < 0.0 for item in row):
            raise BSD35kHighRateEvidenceError(f"{label} PSD density에 음수가 있습니다")
        matrices.append(row)
    maxima_raw = entry["max_density_ratios_7"]
    valid_raw = entry["valid_bands_7"]
    if not isinstance(maxima_raw, list) or len(maxima_raw) != 7:
        raise BSD35kHighRateEvidenceError(f"{label}.max_density_ratios_7는 7 bands여야 합니다")
    if not isinstance(valid_raw, list) or len(valid_raw) != 7 or any(
        type(item) is not bool for item in valid_raw
    ):
        raise BSD35kHighRateEvidenceError(f"{label}.valid_bands_7는 bool 7개여야 합니다")
    maxima = [_finite(item, label=f"{label} max density") for item in maxima_raw]
    if any(item < 0.0 for item in maxima):
        raise BSD35kHighRateEvidenceError(f"{label} max density에 음수가 있습니다")
    expected_maxima = [max(row[index] for row in matrices) for index in range(7)]
    expected_valid = [value >= MIN_TARGET_D_DENSITY_RATIO for value in expected_maxima]
    if not np.allclose(maxima, expected_maxima, rtol=1.0e-12, atol=1.0e-14) or list(
        valid_raw
    ) != expected_valid:
        raise BSD35kHighRateEvidenceError(f"{label} PSD max/valid summary가 matrix와 다릅니다")
    return {
        "schema": PSD_RECIPE_SCHEMA,
        "window_selection": PSD_WINDOW_SELECTION,
        "minimum_window_frames": MIN_PSD_WINDOW_FRAMES,
        "maximum_window_frames": MAX_PSD_WINDOW_FRAMES,
        "window_frames": frames,
        "window_starts": starts,
        "density_ratios_by_window_7": matrices,
        "max_density_ratios_7": maxima,
        "valid_bands_7": expected_valid,
    }


def _positive_or_zero(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BSD35kHighRateEvidenceError(f"{label}는 0 이상의 정수여야 합니다")
    return int(value)


def _coverage_from_entries(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    contract = _full_octave_contract()
    covered: dict[tuple[str, int], set[str]] = defaultdict(set)
    for entry in entries:
        split = str(entry["split"])
        lineage = str(entry["lineage_group"])
        valid = entry["native_psd"]["valid_bands_7"]
        for index, accepted in enumerate(valid):
            if accepted:
                covered[(split, index)].add(lineage)
    rows: list[dict[str, Any]] = []
    for split in REQUIRED_SPLITS:
        for index, band in enumerate(contract.equal_weight_octave_objective_bands_hz):
            components = covered[(split, index)]
            rows.append(
                {
                    "split": split,
                    "band_index": index,
                    "band_hz": [float(band[0]), float(band[1])],
                    "independent_uploader_components": len(components),
                    "minimum_components": MIN_INDEPENDENT_UPLOADERS_PER_SPLIT_BAND,
                    "passed": len(components) >= MIN_INDEPENDENT_UPLOADERS_PER_SPLIT_BAND,
                }
            )
    return rows


def _selection_entries(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise BSD35kHighRateEvidenceError("BSD35k selection entries가 없습니다")
    return [dict(item) for item in entries]


def _selection_plan_from_reference(
    root: Path, value: object
) -> tuple[dict[str, Any], FileSnapshot, dict[str, Any]]:
    entry = _exact(
        value,
        {"file", "selection_plan_sha256", "metadata_csv"},
        label="selection_plan",
    )
    reference, snapshot = _validate_file_reference(root, entry["file"], label="selection_plan.file")
    plan = _load_json(snapshot, label="selection_plan.file")
    try:
        bsd.validate_bsd35k_machine_selection(plan)
    except ValueError as exc:
        raise BSD35kHighRateEvidenceError(f"BSD35k official selection plan 검증 실패: {exc}") from exc
    claimed = _sha256(entry["selection_plan_sha256"], label="selection_plan.selection_plan_sha256")
    if claimed != plan.get("selection_plan_sha256"):
        raise BSD35kHighRateEvidenceError("selection_plan semantic SHA가 plan bytes와 다릅니다")
    _metadata_reference, metadata_snapshot = _validate_file_reference(
        root,
        entry["metadata_csv"],
        label="selection_plan.metadata_csv",
    )
    try:
        bsd.verify_bsd35k_machine_selection_against_metadata(plan, metadata_snapshot.path)
    except ValueError as exc:
        raise BSD35kHighRateEvidenceError(
            f"BSD35k official metadata/selection lineage 재검증 실패: {exc}"
        ) from exc
    metadata_after = _snapshot_file(
        root,
        metadata_snapshot.path,
        label="selection_plan.metadata_csv post-verify",
    )
    if (
        metadata_after.sha256 != metadata_snapshot.sha256
        or metadata_after.size != metadata_snapshot.size
        or metadata_after.device != metadata_snapshot.device
        or metadata_after.inode != metadata_snapshot.inode
        or metadata_after.mtime_ns != metadata_snapshot.mtime_ns
        or metadata_after.ctime_ns != metadata_snapshot.ctime_ns
    ):
        raise BSD35kHighRateEvidenceError("BSD35k official metadata CSV가 lineage 재검증 중 변경됐습니다")
    return reference, snapshot, plan


def _decoder_audit_from_reference(
    root: Path, value: object
) -> tuple[dict[str, Any], dict[str, Any]]:
    entry = _exact(
        value,
        {
            "file",
            "audit_sha256",
            "inventory_sha256",
            "accepted_inventory_sha256",
            "decoder_fingerprint_sha256",
        },
        label="decoder_audit",
    )
    reference, snapshot = _validate_file_reference(root, entry["file"], label="decoder_audit.file")
    try:
        audit = read_decoder_audit(snapshot.path, repo_root=root, label="BSD35k decoder audit")
        semantic = {
            key: item for key, item in audit.items() if not str(key).startswith("_")
        }
        validate_audit_report_self_digest(semantic)
    except (OSError, ValueError) as exc:
        raise BSD35kHighRateEvidenceError(f"BSD35k decoder audit 검증 실패: {exc}") from exc
    for key in (
        "audit_sha256",
        "inventory_sha256",
        "accepted_inventory_sha256",
        "decoder_fingerprint_sha256",
    ):
        claimed = _sha256(entry[key], label=f"decoder_audit.{key}")
        if claimed != audit.get(key):
            raise BSD35kHighRateEvidenceError(f"decoder_audit.{key}가 current audit와 다릅니다")
    return reference, audit


def _archive_reference(root: Path, value: object) -> tuple[dict[str, Any], FileSnapshot | None]:
    entry = _exact(
        value,
        {
            "file",
            "official_size_bytes",
            "official_md5",
            "observed_sha256",
            "verified_during_issue",
        },
        label="audio_archive",
    )
    if entry["official_size_bytes"] != bsd.OFFICIAL_AUDIO_ZIP_SIZE:
        raise BSD35kHighRateEvidenceError("BSD35k official audio archive size 계약이 다릅니다")
    if _md5(entry["official_md5"], label="audio_archive.official_md5") != bsd.OFFICIAL_AUDIO_ZIP_MD5:
        raise BSD35kHighRateEvidenceError("BSD35k official audio archive MD5 계약이 다릅니다")
    if entry["verified_during_issue"] is not True:
        raise BSD35kHighRateEvidenceError("audio archive는 issue 시 직접 검증되어야 합니다")
    reference = _exact(entry["file"], {"path", "size_bytes", "sha256"}, label="audio_archive.file")
    if _positive_int(reference["size_bytes"], label="audio_archive.file.size_bytes") != bsd.OFFICIAL_AUDIO_ZIP_SIZE:
        raise BSD35kHighRateEvidenceError("audio archive file size가 official size와 다릅니다")
    observed = _sha256(entry["observed_sha256"], label="audio_archive.observed_sha256")
    if observed != _sha256(reference["sha256"], label="audio_archive.file.sha256"):
        raise BSD35kHighRateEvidenceError("audio archive observed/file SHA가 다릅니다")
    # archive는 source extraction 뒤 삭제할 수 있다. bootstrap 재검증은 selected raw
    # member bytes와 this issue receipt를 보며 archive file 자체는 요구하지 않는다.
    relative = _relative_posix(reference["path"], label="audio_archive.file.path")
    candidate = root / Path(relative)
    if not candidate.exists():
        return (
            {
                "file": {
                    "path": relative,
                    "size_bytes": bsd.OFFICIAL_AUDIO_ZIP_SIZE,
                    "sha256": observed,
                },
                "official_size_bytes": bsd.OFFICIAL_AUDIO_ZIP_SIZE,
                "official_md5": bsd.OFFICIAL_AUDIO_ZIP_MD5,
                "observed_sha256": observed,
                "verified_during_issue": True,
            },
            None,
        )
    file_reference, snapshot = _validate_file_reference(root, reference, label="audio_archive.file")
    if _md5_file(snapshot.path) != bsd.OFFICIAL_AUDIO_ZIP_MD5:
        raise BSD35kHighRateEvidenceError("local audio archive MD5가 official 값과 다릅니다")
    return (
        {
            "file": file_reference,
            "official_size_bytes": bsd.OFFICIAL_AUDIO_ZIP_SIZE,
            "official_md5": bsd.OFFICIAL_AUDIO_ZIP_MD5,
            "observed_sha256": observed,
            "verified_during_issue": True,
        },
        snapshot,
    )


def _raw_root(root: Path, value: object) -> tuple[str, Path]:
    relative = _relative_posix(value, label="selected_raw_root")
    path = _inside_root(root, relative, label="selected_raw_root")
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise BSD35kHighRateEvidenceError("selected_raw_root가 없습니다") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise BSD35kHighRateEvidenceError("selected_raw_root는 directory여야 합니다")
    return relative, path


def _recorded_entry_from_plan(
    *,
    root: Path,
    raw_root: Path,
    plan_entry: Mapping[str, Any],
    audit: Mapping[str, Any],
    archive_members: Mapping[str, tuple[int, str]] | None,
) -> dict[str, Any]:
    sound_id = _positive_int(plan_entry.get("sound_id"), label="selection sound_id")
    member = _relative_posix(plan_entry.get("archive_member"), label="selection archive_member")
    if member != f"audio/{sound_id}.wav":
        raise BSD35kHighRateEvidenceError("selection archive member/sound_id가 다릅니다")
    split = str(plan_entry.get("split") or "")
    uploader = str(plan_entry.get("uploader") or "")
    lineage_group = str(plan_entry.get("lineage_group") or "")
    if split not in REQUIRED_SPLITS or not uploader or lineage_group != f"bsd35k_uploader:{uploader}":
        raise BSD35kHighRateEvidenceError("selection split/uploader/lineage가 다릅니다")
    raw_path = raw_root / Path(member)
    snapshot = _snapshot_file(root, raw_path, label=f"BSD35k raw {member}")
    if archive_members is not None:
        member_size, member_sha = archive_members[member]
        if snapshot.size != member_size or snapshot.sha256 != member_sha:
            raise BSD35kHighRateEvidenceError(
                f"extracted raw WAV가 official archive member와 다릅니다: {member}"
            )
    relative = _relative_to_root(root, snapshot.path, label=f"BSD35k raw {member}")
    audited = audit.get("_inventory_by_relative_path", {}).get(relative)
    if not isinstance(audited, Mapping):
        raise BSD35kHighRateEvidenceError(f"decoder audit에 selected raw가 없습니다: {relative}")
    if (
        audited.get("decision") != "accept"
        or audited.get("content_sha256") != snapshot.sha256
        or audited.get("content_size") != snapshot.size
    ):
        raise BSD35kHighRateEvidenceError(
            f"decoder audit가 selected raw full decode accept를 증명하지 않습니다: {relative}"
        )
    header, psd = _native_header_and_psd(snapshot.path)
    after = _snapshot_file(root, snapshot.path, label=f"BSD35k raw post-PSD {member}")
    if (
        after.sha256 != snapshot.sha256
        or after.size != snapshot.size
        or after.device != snapshot.device
        or after.inode != snapshot.inode
        or after.mtime_ns != snapshot.mtime_ns
        or after.ctime_ns != snapshot.ctime_ns
    ):
        raise BSD35kHighRateEvidenceError(f"PSD decode 중 BSD35k raw가 변경됐습니다: {member}")
    return {
        "sound_id": sound_id,
        "split": split,
        "uploader": uploader,
        "lineage_group": lineage_group,
        "archive_member": member,
        "raw_file": _reference_from_snapshot(root, snapshot, label=f"BSD35k raw {member}"),
        "decoder_audit_relative_path": relative,
        "native_header": header,
        "native_psd": psd,
    }


def _entry_from_evidence(
    *,
    root: Path,
    raw_root: Path,
    plan_entry: Mapping[str, Any],
    raw: object,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    entry = _exact(
        raw,
        {
            "sound_id",
            "split",
            "uploader",
            "lineage_group",
            "archive_member",
            "raw_file",
            "decoder_audit_relative_path",
            "native_header",
            "native_psd",
        },
        label="high-rate machine entry",
    )
    expected_id = _positive_int(plan_entry.get("sound_id"), label="selection sound_id")
    if _positive_int(entry["sound_id"], label="entry.sound_id") != expected_id:
        raise BSD35kHighRateEvidenceError("evidence sound_id가 selection과 다릅니다")
    for key in ("split", "uploader", "lineage_group", "archive_member"):
        if entry[key] != plan_entry.get(key):
            raise BSD35kHighRateEvidenceError(f"evidence {key}가 selection과 다릅니다")
    raw_reference, snapshot = _validate_file_reference(root, entry["raw_file"], label="entry.raw_file")
    expected_member = _relative_posix(plan_entry["archive_member"], label="selection archive_member")
    expected_path = raw_root / Path(expected_member)
    if snapshot.path != expected_path:
        raise BSD35kHighRateEvidenceError("entry raw file이 selected_raw_root/archive_member와 다릅니다")
    relative = _relative_posix(
        entry["decoder_audit_relative_path"], label="entry.decoder_audit_relative_path"
    )
    if relative != raw_reference["path"]:
        raise BSD35kHighRateEvidenceError("entry decoder audit path가 raw reference와 다릅니다")
    audited = audit.get("_inventory_by_relative_path", {}).get(relative)
    if not isinstance(audited, Mapping) or audited.get("decision") != "accept":
        raise BSD35kHighRateEvidenceError("entry decoder audit accept evidence가 없습니다")
    if audited.get("content_sha256") != raw_reference["sha256"] or audited.get(
        "content_size"
    ) != raw_reference["size_bytes"]:
        raise BSD35kHighRateEvidenceError("entry decoder audit bytes가 raw reference와 다릅니다")
    header = _require_native_header(entry["native_header"], label="entry.native_header")
    psd = _require_psd(entry["native_psd"], label="entry.native_psd")
    actual_header, actual_psd = _native_header_and_psd(snapshot.path)
    if header != actual_header:
        raise BSD35kHighRateEvidenceError("entry native header가 local WAV 재검산과 다릅니다")
    if psd["window_frames"] != actual_psd["window_frames"] or psd["window_starts"] != actual_psd[
        "window_starts"
    ]:
        raise BSD35kHighRateEvidenceError("entry PSD window 선택이 local WAV 재검산과 다릅니다")
    if not np.allclose(
        np.asarray(psd["density_ratios_by_window_7"], dtype=np.float64),
        np.asarray(actual_psd["density_ratios_by_window_7"], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise BSD35kHighRateEvidenceError("entry PSD density가 local WAV 재검산과 다릅니다")
    if not np.allclose(
        psd["max_density_ratios_7"],
        actual_psd["max_density_ratios_7"],
        rtol=1.0e-10,
        atol=1.0e-12,
    ) or psd["valid_bands_7"] != actual_psd["valid_bands_7"]:
        raise BSD35kHighRateEvidenceError("entry PSD summary가 local WAV 재검산과 다릅니다")
    return {
        "sound_id": expected_id,
        "split": str(entry["split"]),
        "uploader": str(entry["uploader"]),
        "lineage_group": str(entry["lineage_group"]),
        "archive_member": expected_member,
        "raw_file": raw_reference,
        "decoder_audit_relative_path": relative,
        "native_header": header,
        "native_psd": psd,
    }


def _expected_authority(*, coverage_pass: bool) -> dict[str, Any]:
    blockers = [
        "requires_full_octave_causal_primary_err_density",
        "requires_external_population_authority",
    ]
    if not coverage_pass:
        blockers.insert(0, "native_psd_split_band_coverage_incomplete")
    return {
        "selection_lineage_verified": True,
        "official_audio_archive_verified": True,
        "selected_archive_members_match_raw": True,
        "decoder_audit_accepts_selected_raw": True,
        "native_high_rate_verified": True,
        "native_psd_split_band_coverage_passed": coverage_pass,
        "source_prepopulation_eligible": coverage_pass,
        "canonical_training_eligible": False,
        "blockers": blockers,
    }


def build_bsd35k_highrate_machine_evidence(
    *,
    repository_root: str | Path,
    selection_plan_path: str | Path,
    metadata_csv_path: str | Path,
    selected_raw_root: str | Path,
    audio_archive_path: str | Path,
    decoder_audit_path: str | Path,
) -> dict[str, Any]:
    """actual BSD35k bytes에서 source-stage evidence를 만든다.

    이 함수는 output을 쓰지 않는다. issuer CLI만 O_EXCL writer를 사용한다. archive와
    extracted WAV가 둘 다 있을 때만 호출할 수 있으며, issue 후 archive 삭제 여부는
    evidence validator가 아닌 retention 정책의 문제다.
    """

    root = _repo_root(repository_root)
    selection_snapshot = _snapshot_file(root, Path(selection_plan_path), label="selection plan")
    plan = _load_json(selection_snapshot, label="selection plan")
    try:
        bsd.validate_bsd35k_machine_selection(plan)
    except ValueError as exc:
        raise BSD35kHighRateEvidenceError(f"BSD35k selection plan 검증 실패: {exc}") from exc
    raw_relative = _relative_to_root(root, Path(selected_raw_root), label="selected_raw_root")
    _, raw_root = _raw_root(root, raw_relative)
    metadata_snapshot = _snapshot_file(root, Path(metadata_csv_path), label="BSD35k metadata CSV")
    try:
        bsd.verify_bsd35k_machine_selection_against_metadata(plan, metadata_snapshot.path)
    except ValueError as exc:
        raise BSD35kHighRateEvidenceError(
            f"BSD35k official metadata/selection lineage 재검증 실패: {exc}"
        ) from exc
    metadata_after = _snapshot_file(
        root,
        metadata_snapshot.path,
        label="BSD35k metadata CSV post-verify",
    )
    if (
        metadata_after.sha256 != metadata_snapshot.sha256
        or metadata_after.size != metadata_snapshot.size
        or metadata_after.device != metadata_snapshot.device
        or metadata_after.inode != metadata_snapshot.inode
        or metadata_after.mtime_ns != metadata_snapshot.mtime_ns
        or metadata_after.ctime_ns != metadata_snapshot.ctime_ns
    ):
        raise BSD35kHighRateEvidenceError("BSD35k official metadata CSV가 issue 중 변경됐습니다")
    archive_snapshot = _snapshot_file(root, Path(audio_archive_path), label="BSD35k audio archive")
    if archive_snapshot.size != bsd.OFFICIAL_AUDIO_ZIP_SIZE or _md5_file(
        archive_snapshot.path
    ) != bsd.OFFICIAL_AUDIO_ZIP_MD5:
        raise BSD35kHighRateEvidenceError("BSD35k official audio archive size/MD5가 다릅니다")
    try:
        audit = read_decoder_audit(
            Path(decoder_audit_path), repo_root=root, label="BSD35k decoder audit"
        )
        validate_audit_report_self_digest(
            {key: value for key, value in audit.items() if not str(key).startswith("_")}
        )
    except (OSError, ValueError) as exc:
        raise BSD35kHighRateEvidenceError(f"BSD35k decoder audit 검증 실패: {exc}") from exc

    plan_entries = _selection_entries(plan)
    members = [str(entry["archive_member"]) for entry in plan_entries]
    archive_members = _zip_selected_members(archive_snapshot.path, members)
    entries = [
        _recorded_entry_from_plan(
            root=root,
            raw_root=raw_root,
            plan_entry=entry,
            audit=audit,
            archive_members=archive_members,
        )
        for entry in plan_entries
    ]
    # selected root에 archive selection 밖 audio가 섞이면 decoder audit의 wide source
    # inventory가 뒤섞여 manifest consumer가 source family를 잘못 받기 쉽다.
    discovered: set[str] = set()
    for current_text, _directories, files in os.walk(raw_root, followlinks=False):
        current = Path(current_text)
        for name in files:
            candidate = current / name
            if candidate.suffix.casefold() in {".wav", ".flac", ".mp3"}:
                discovered.add(candidate.relative_to(raw_root).as_posix())
    if discovered != set(members):
        unexpected = sorted(discovered.symmetric_difference(set(members)))
        raise BSD35kHighRateEvidenceError(
            "selected_raw_root audio inventory가 selection archive member와 다릅니다: "
            f"{unexpected[:3]}"
        )
    coverage = _coverage_from_entries(entries)
    coverage_pass = all(row["passed"] for row in coverage)
    selected_raw_tree = [
        {
            "archive_member": entry["archive_member"],
            "raw_file": entry["raw_file"],
        }
        for entry in entries
    ]
    selection_reference = _reference_from_snapshot(root, selection_snapshot, label="selection plan")
    audit_snapshot = audit["_snapshot"]
    assert isinstance(audit_snapshot, FileSnapshot)
    payload: dict[str, Any] = {
        "schema_version": BSD35K_HIGHRATE_EVIDENCE_SCHEMA,
        "role": BSD35K_HIGHRATE_EVIDENCE_ROLE,
        "status": "PASS" if coverage_pass else "BLOCKED",
        "control_band_contract_sha256": _full_octave_contract().digest(),
        "minimum_native_sample_rate_hz": _minimum_native_sample_rate_hz(),
        "required_native_nyquist_hz": float(
            _full_octave_contract().required_excitation_upper_hz
        ),
        "selection_plan": {
            "file": selection_reference,
            "selection_plan_sha256": plan["selection_plan_sha256"],
            "metadata_csv": _reference_from_snapshot(
                root,
                metadata_snapshot,
                label="BSD35k metadata CSV",
            ),
        },
        "audio_archive": {
            "file": _reference_from_snapshot(root, archive_snapshot, label="audio archive"),
            "official_size_bytes": bsd.OFFICIAL_AUDIO_ZIP_SIZE,
            "official_md5": bsd.OFFICIAL_AUDIO_ZIP_MD5,
            "observed_sha256": archive_snapshot.sha256,
            "verified_during_issue": True,
        },
        "decoder_audit": {
            "file": _reference_from_snapshot(root, audit_snapshot, label="decoder audit"),
            "audit_sha256": audit["audit_sha256"],
            "inventory_sha256": audit["inventory_sha256"],
            "accepted_inventory_sha256": audit["accepted_inventory_sha256"],
            "decoder_fingerprint_sha256": audit["decoder_fingerprint_sha256"],
        },
        "selected_raw_root": raw_relative,
        "selected_raw_tree_sha256": hashlib.sha256(_canonical_json(selected_raw_tree)).hexdigest(),
        "entries": entries,
        "coverage": coverage,
        "authority": _expected_authority(coverage_pass=coverage_pass),
    }
    payload["evidence_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    after_archive = _snapshot_file(root, archive_snapshot.path, label="BSD35k audio archive post-issue")
    if (
        after_archive.sha256 != archive_snapshot.sha256
        or after_archive.size != archive_snapshot.size
        or after_archive.device != archive_snapshot.device
        or after_archive.inode != archive_snapshot.inode
        or after_archive.mtime_ns != archive_snapshot.mtime_ns
        or after_archive.ctime_ns != archive_snapshot.ctime_ns
    ):
        raise BSD35kHighRateEvidenceError("BSD35k audio archive가 issue 중 변경됐습니다")
    return payload


def validate_bsd35k_highrate_machine_evidence(
    evidence: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """persisted source evidence와 live selected raw/decoder bytes를 fail-closed로 재검증."""

    root = _repo_root(repository_root)
    payload = _exact(
        evidence,
        {
            "schema_version",
            "role",
            "status",
            "control_band_contract_sha256",
            "minimum_native_sample_rate_hz",
            "required_native_nyquist_hz",
            "selection_plan",
            "audio_archive",
            "decoder_audit",
            "selected_raw_root",
            "selected_raw_tree_sha256",
            "entries",
            "coverage",
            "authority",
            "evidence_sha256",
        },
        label="BSD35k high-rate evidence",
    )
    if payload["schema_version"] != BSD35K_HIGHRATE_EVIDENCE_SCHEMA:
        raise BSD35kHighRateEvidenceError("BSD35k high-rate evidence schema가 다릅니다")
    if payload["role"] != BSD35K_HIGHRATE_EVIDENCE_ROLE:
        raise BSD35kHighRateEvidenceError("BSD35k high-rate evidence role이 다릅니다")
    contract = _full_octave_contract()
    if payload["control_band_contract_sha256"] != contract.digest():
        raise BSD35kHighRateEvidenceError("BSD35k high-rate evidence control-band SHA가 다릅니다")
    if payload["minimum_native_sample_rate_hz"] != _minimum_native_sample_rate_hz():
        raise BSD35kHighRateEvidenceError("BSD35k high-rate evidence native sample-rate 하한이 다릅니다")
    required_nyquist = _finite(
        payload["required_native_nyquist_hz"], label="required_native_nyquist_hz"
    )
    if not math.isclose(required_nyquist, contract.required_excitation_upper_hz, abs_tol=1.0e-9):
        raise BSD35kHighRateEvidenceError("BSD35k high-rate evidence native Nyquist 하한이 다릅니다")
    claimed_evidence_sha = _sha256(payload["evidence_sha256"], label="evidence_sha256")
    if claimed_evidence_sha != hashlib.sha256(
        _canonical_json({key: value for key, value in payload.items() if key != "evidence_sha256"})
    ).hexdigest():
        raise BSD35kHighRateEvidenceError("BSD35k high-rate evidence self SHA가 다릅니다")

    _selection_reference, _selection_snapshot, plan = _selection_plan_from_reference(
        root, payload["selection_plan"]
    )
    _archive, _archive_snapshot = _archive_reference(root, payload["audio_archive"])
    _audit_reference, audit = _decoder_audit_from_reference(root, payload["decoder_audit"])
    raw_relative, raw_root = _raw_root(root, payload["selected_raw_root"])
    if raw_relative != payload["selected_raw_root"]:
        raise BSD35kHighRateEvidenceError("selected_raw_root 정규화가 다릅니다")

    plan_entries = _selection_entries(plan)
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list) or len(raw_entries) != len(plan_entries):
        raise BSD35kHighRateEvidenceError("high-rate evidence entry 수가 official selection과 다릅니다")
    validated_entries = [
        _entry_from_evidence(
            root=root,
            raw_root=raw_root,
            plan_entry=plan_entry,
            raw=raw,
            audit=audit,
        )
        for plan_entry, raw in zip(plan_entries, raw_entries, strict=True)
    ]
    if [entry["sound_id"] for entry in validated_entries] != sorted(
        entry["sound_id"] for entry in validated_entries
    ):
        raise BSD35kHighRateEvidenceError("high-rate evidence entry는 sound_id 오름차순이어야 합니다")
    if len({entry["raw_file"]["path"] for entry in validated_entries}) != len(validated_entries):
        raise BSD35kHighRateEvidenceError("high-rate evidence raw file path가 중복됩니다")
    selected_raw_tree = [
        {"archive_member": entry["archive_member"], "raw_file": entry["raw_file"]}
        for entry in validated_entries
    ]
    if _sha256(payload["selected_raw_tree_sha256"], label="selected_raw_tree_sha256") != hashlib.sha256(
        _canonical_json(selected_raw_tree)
    ).hexdigest():
        raise BSD35kHighRateEvidenceError("selected raw tree SHA가 actual entry와 다릅니다")
    discovered: set[str] = set()
    for current_text, _directories, files in os.walk(raw_root, followlinks=False):
        current = Path(current_text)
        for name in files:
            candidate = current / name
            if candidate.suffix.casefold() in {".wav", ".flac", ".mp3"}:
                discovered.add(candidate.relative_to(raw_root).as_posix())
    expected_members = {entry["archive_member"] for entry in validated_entries}
    if discovered != expected_members:
        raise BSD35kHighRateEvidenceError("selected_raw_root audio inventory가 evidence selection과 다릅니다")

    coverage = _coverage_from_entries(validated_entries)
    if payload["coverage"] != coverage:
        raise BSD35kHighRateEvidenceError("high-rate evidence coverage가 raw PSD 재집계와 다릅니다")
    coverage_pass = all(row["passed"] for row in coverage)
    expected_status = "PASS" if coverage_pass else "BLOCKED"
    if payload["status"] != expected_status:
        raise BSD35kHighRateEvidenceError("high-rate evidence status가 coverage와 다릅니다")
    if payload["authority"] != _expected_authority(coverage_pass=coverage_pass):
        raise BSD35kHighRateEvidenceError("high-rate evidence authority가 canonical blocker와 다릅니다")
    return {
        "status": expected_status,
        "evidence_sha256": claimed_evidence_sha,
        "selected_raw_root": raw_relative,
        "selected_file_count": len(validated_entries),
        "coverage_pass": coverage_pass,
        "canonical_training_eligible": False,
        "blockers": list(payload["authority"]["blockers"]),
    }


def load_and_validate_bsd35k_highrate_machine_evidence(
    path: str | Path,
    *,
    repository_root: str | Path,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """evidence file bytes SHA와 semantic/live source evidence를 함께 검증한다."""

    root = _repo_root(repository_root)
    snapshot = _snapshot_file(root, Path(path), label="BSD35k high-rate evidence")
    if expected_sha256 is not None and snapshot.sha256 != _sha256(
        expected_sha256, label="expected evidence file SHA"
    ):
        raise BSD35kHighRateEvidenceError(
            "BSD35k high-rate evidence file SHA가 external expected SHA와 다릅니다"
        )
    payload = _load_json(snapshot, label="BSD35k high-rate evidence")
    result = validate_bsd35k_highrate_machine_evidence(payload, repository_root=root)
    result["file_sha256"] = snapshot.sha256
    result["path"] = _relative_to_root(root, snapshot.path, label="BSD35k high-rate evidence")
    return result


def write_bsd35k_highrate_machine_evidence_exclusive(
    target: str | Path, evidence: Mapping[str, Any]
) -> tuple[Path, str]:
    """검증된 evidence를 O_EXCL + fsync로 한 번만 쓴다."""

    # Writer는 standalone temp root에도 쓸 수 있으므로 local raw 재검증은 caller가
    # build/validate 단계에서 끝냈다고 보고 JSON self-sha만 먼저 닫는다.
    payload = dict(evidence)
    claimed = _sha256(payload.get("evidence_sha256"), label="evidence_sha256")
    if claimed != hashlib.sha256(
        _canonical_json({key: value for key, value in payload.items() if key != "evidence_sha256"})
    ).hexdigest():
        raise BSD35kHighRateEvidenceError("writer 입력 evidence self SHA가 다릅니다")
    path = Path(target).expanduser()
    if not path.is_absolute():
        path = Path(os.path.abspath(path))
    cursor = path.parent
    missing: list[Path] = []
    while not cursor.exists():
        if cursor.is_symlink():
            raise BSD35kHighRateEvidenceError(f"output parent symlink를 거부합니다: {cursor}")
        missing.append(cursor)
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    if cursor.is_symlink():
        raise BSD35kHighRateEvidenceError(f"output parent symlink를 거부합니다: {cursor}")
    for directory in reversed(missing):
        directory.mkdir()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"no-replace output이 이미 존재합니다: {path}")
    raw = _canonical_json(payload) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
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
    return path, hashlib.sha256(raw).hexdigest()


__all__ = [
    "BSD35K_HIGHRATE_EVIDENCE_ROLE",
    "BSD35K_HIGHRATE_EVIDENCE_SCHEMA",
    "BSD35kHighRateEvidenceError",
    "build_bsd35k_highrate_machine_evidence",
    "load_and_validate_bsd35k_highrate_machine_evidence",
    "validate_bsd35k_highrate_machine_evidence",
    "write_bsd35k_highrate_machine_evidence_exclusive",
]
