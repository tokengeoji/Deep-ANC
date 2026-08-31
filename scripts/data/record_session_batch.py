#!/usr/bin/env python3
"""사전 계획된 소스 목록을 순서대로 재생해 recorded 세션을 수집한다.

이 도구는 무인 장시간 재생기가 아니다. 실제 실행은 사용자 입회·볼륨 최저·배선/덕트
geometry 확인을 모두 명시해야 하며, 재시도는 기본적으로 하지 않는다. 먼저 ``--dry-run``으로
소스 목록, 계보/split, 예상 audible/connected 시간과 세션별 hard timeout을 검증한다.

소스 CSV 필수 열은 ``path,seconds,source_family,group_id``다. ``lineage_key``가 없으면
``group_id``를 lineage key로 사용한다. split은 각 행의 ``split`` 또는 실행 전에 명시한
``--preassigned-split``에서만 가져오며, 녹음 뒤에 배정하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import datetime
import errno
import hashlib
import json
import os
import secrets
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.manifest import (  # noqa: E402
    VALID_SPLITS,
    validate_group_id,
    validate_source_family,
)
from deep_anc.data.holdout_contract import (  # noqa: E402
    read_regular_file_snapshot,
    reject_symlink_components,
)
from deep_anc.data.recorded_generation import (  # noqa: E402
    ADDITION_SESSION_COUNT,
    ADDITIONS_ROOT,
    CANONICAL_GENERATION_ID,
    CANONICAL_RECORDING_AMPLITUDE,
    RecordedGenerationError,
    SOURCE_PLAN_FIELDS,
    SOURCE_PLAN_ROOT,
    _read_source_plan,
    _read_session_metadata,
    _validate_session_artifacts,
    validate_generation_id,
)
from deep_anc.data.recording_level_campaign import (  # noqa: E402
    RecordingLevelCampaignError,
    validate_recording_level_campaign,
)
from deep_anc.data.recording_source_gain import (  # noqa: E402
    PHYSICAL_SELECTOR_MAX_AMPLITUDE_MILLIONTHS,
    RecordingSourceGainError,
    validate_recording_source_gain_plan,
)
from deep_anc.audio_io import MAX_RECORDING_OUTPUT_PEAK  # noqa: E402
from deep_anc.eval.artifacts import write_csv  # noqa: E402

MAX_CLIP_RATIO = 0.0
MIN_ERR_RMS_DBFS = -60.0
MAX_ERR_PEAK = 0.95
FAILURE_RECEIPT_MARKER = "DEEP_ANC_RECORD_DUCT_FAILURE_JSON="
GRACEFUL_CHILD_STOP_SECONDS = 30.0

# record_duct의 input-only probe(앞 1초 settle + 뒤 2초 판정)와 CPU idle witness.
# 실제 출력 시간과 혼동하지 않도록 dry-run에서 별도 항목으로 표시한다.
INPUT_ONLY_PREFLIGHT_SECONDS = 3.5


@dataclass(frozen=True)
class ChildResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


@dataclass(frozen=True)
class ChildFailureEvidence:
    stage: str
    reason: str
    artifact: Path
    receipt: Path
    receipt_sha256: str
    raw_available: bool


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_batch_recording_level_campaign(
    args: argparse.Namespace, *, required: bool
) -> dict | None:
    """Batch가 child를 만들기 전에 external-SHA campaign을 fail-closed한다."""

    receipt = args.recording_level_campaign
    expected_sha = args.recording_level_campaign_sha256
    if receipt is None and expected_sha is None:
        if required:
            raise RecordingLevelCampaignError(
                "canonical additions에는 recording-level campaign path/SHA가 필요합니다"
            )
        if args.confirm_same_amplifier_setting:
            raise RecordingLevelCampaignError(
                "campaign 없이 same-amplifier 확인만 지정할 수 없습니다"
            )
        return None
    if receipt is None or expected_sha is None:
        raise RecordingLevelCampaignError(
            "--recording-level-campaign과 --recording-level-campaign-sha256은 함께 필요합니다"
        )
    if args.confirm_same_amplifier_setting is not True:
        raise RecordingLevelCampaignError(
            "campaign 사용에는 --confirm-same-amplifier-setting이 필요합니다"
        )
    summary = validate_recording_level_campaign(
        repo_root=REPO_ROOT,
        campaign_receipt=str(receipt),
        expected_sha256=str(expected_sha).lower(),
        now_utc=datetime.datetime.now(datetime.timezone.utc),
        require_fresh=True,
    )
    payload = summary.get("payload")
    hardware = payload.get("hardware") if isinstance(payload, dict) else None
    config_ref = hardware.get("config") if isinstance(hardware, dict) else None
    config_path = config_ref.get("path") if isinstance(config_ref, dict) else None
    if (
        not isinstance(config_path, str)
        or _lexical_repo_path(config_path) != _lexical_repo_path(args.hardware)
    ):
        raise RecordingLevelCampaignError(
            "batch --hardware가 recording-level campaign config와 다릅니다"
        )
    return summary


def _validate_batch_source_gain_plan(
    args: argparse.Namespace,
    *,
    required: bool,
    source_list_sha256: str,
) -> dict | None:
    """Canonical live 전에 source별 gain plan과 외부 SHA를 fail-closed한다.

    schema-v1은 strict-P ERR 예측만 제공하고 REF/다중레벨 선형성 authority가 없으므로
    정상적으로 ``canonical_live_eligible=false``다. 따라서 plan을 공급해도 live는
    차단하며, fixed ``0.06`` batch가 다시 열리지 않는다.
    """

    path = args.source_gain_plan
    expected_sha = args.source_gain_plan_sha256
    if path is None and expected_sha is None:
        if required:
            raise RecordingSourceGainError(
                "canonical additions live에는 source별 gain plan path/SHA가 필요합니다"
            )
        return None
    if path is None or expected_sha is None:
        raise RecordingSourceGainError(
            "--source-gain-plan과 --source-gain-plan-sha256은 함께 필요합니다"
        )
    summary = validate_recording_source_gain_plan(
        repo_root=REPO_ROOT,
        plan_path=str(path),
        expected_sha256=str(expected_sha).lower(),
    )
    payload = summary.get("payload")
    source_ref = payload.get("source_plan") if isinstance(payload, dict) else None
    if (
        not isinstance(source_ref, dict)
        or source_ref.get("sha256") != str(source_list_sha256).lower()
    ):
        raise RecordingSourceGainError(
            "source gain plan이 이번 canonical source-list SHA와 다릅니다"
        )
    if required and summary.get("canonical_live_eligible") is not True:
        blockers = payload.get("blocker_reasons") if isinstance(payload, dict) else None
        raise RecordingSourceGainError(
            "source gain schema-v1은 ERR-only 무출력 계획이며 canonical live authority가 "
            f"아닙니다: blockers={blockers}"
        )
    return summary


def _canonical_source_gain_by_row(summary: dict, entries: list[dict]) -> dict[int, float]:
    """검증된 dynamic payload의 integer-millionths를 exact canonical row에 매핑한다."""

    if summary.get("canonical_live_eligible") is not True:
        raise RecordingSourceGainError("canonical live eligible source-gain plan이 아닙니다")
    payload = summary.get("payload")
    rows = payload.get("rows") if isinstance(payload, dict) else None
    contract = payload.get("contract") if isinstance(payload, dict) else None
    measured_cap = (
        contract.get("reference_amplitude_millionths")
        if isinstance(contract, dict)
        else None
    )
    if (
        isinstance(measured_cap, bool)
        or not isinstance(measured_cap, int)
        or not 1
        <= measured_cap
        <= PHYSICAL_SELECTOR_MAX_AMPLITUDE_MILLIONTHS
    ):
        raise RecordingSourceGainError("source-gain dynamic measured cap 계약 위반")
    if not isinstance(rows, list) or len(rows) != len(entries):
        raise RecordingSourceGainError("source-gain row 수가 canonical source plan과 다릅니다")
    expected = {int(entry["source_row_number"]) for entry in entries}
    result: dict[int, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RecordingSourceGainError("source-gain row가 mapping이 아닙니다")
        row_number = row.get("source_row_number")
        millionths = row.get("selected_amplitude_millionths")
        if (
            isinstance(row_number, bool)
            or not isinstance(row_number, int)
            or row_number not in expected
            or row_number in result
            or isinstance(millionths, bool)
            or not isinstance(millionths, int)
            or not 1 <= millionths <= measured_cap
            or row.get("feasible") is not True
        ):
            raise RecordingSourceGainError("source-gain row/amplitude 계약 위반")
        result[row_number] = millionths / 1_000_000.0
    if set(result) != expected:
        raise RecordingSourceGainError("source-gain row 집합이 canonical plan과 다릅니다")
    return result


def analyse_session(session_dir: Path) -> dict:
    import soundfile as sf

    mics, rate = sf.read(str(session_dir / "mics.wav"), dtype="float32", always_2d=True)
    err = mics[:, 0].astype(np.float64)
    ref = mics[:, 1].astype(np.float64) if mics.shape[1] > 1 else np.zeros_like(err)

    def dbfs(values: np.ndarray) -> float:
        power = float(np.mean(values**2))
        return 10.0 * np.log10(power + 1e-30)

    return {
        "sample_rate_hz": int(rate),
        "seconds": mics.shape[0] / float(rate),
        "err_rms_dbfs": dbfs(err),
        "ref_rms_dbfs": dbfs(ref),
        "err_peak": float(np.max(np.abs(err))),
        "ref_peak": float(np.max(np.abs(ref))),
        "clip_ratio": float(np.mean(np.abs(mics) >= 0.999)),
    }


def qa_verdict(stats: dict) -> tuple[bool, str]:
    if stats["clip_ratio"] > MAX_CLIP_RATIO:
        return False, f"입력 클리핑 {stats['clip_ratio']:.4%}"
    if stats["err_peak"] > MAX_ERR_PEAK:
        return False, f"ERR peak {stats['err_peak']:.3f} 가 한계에 근접"
    if stats["err_rms_dbfs"] < MIN_ERR_RMS_DBFS:
        return False, f"ERR RMS {stats['err_rms_dbfs']:.1f} dBFS — 스피커가 울리지 않았을 수 있음"
    return True, "ok"


def _plan_key(entry: dict, source_list_sha256: str) -> tuple[str, str, int, str, str, float]:
    return (
        str(entry["path"]),
        source_list_sha256,
        int(entry["source_row_number"]),
        str(entry["lineage_key"]),
        str(entry["split"]),
        float(entry["start_seconds"]),
    )


def already_recorded(out_root: Path) -> set[tuple[str, str, int, str, str, float]]:
    """정확히 같은 수집 계획에 결속된 성공 세션만 완료로 센다."""

    done: set[tuple[str, str, int, str, str, float]] = set()
    if not out_root.exists():
        return done
    for meta_path in sorted(out_root.glob("*/session.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            plan = meta["collection_plan"]
            key = (
                str((meta.get("program") or {})["file"]),
                str(plan["source_list_sha256"]),
                int(plan["source_row_number"]),
                str(plan["lineage_key"]),
                str(plan["preassigned_split"]),
                float(plan.get("start_seconds", 0.0)),
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        done.add(key)
    return done


def _lexical_repo_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    candidate = candidate if candidate.is_absolute() else REPO_ROOT / candidate
    return Path(os.path.abspath(candidate))


def _reject_existing_symlink_components(path: Path) -> Path:
    """아직 없는 canonical out-root도 허용하되 기존 component symlink는 거부한다."""

    candidate = _lexical_repo_path(path)
    boundary = _lexical_repo_path(REPO_ROOT)
    try:
        relative = candidate.relative_to(boundary)
    except ValueError as exc:
        raise ValueError(f"canonical additions 경로가 저장소 밖입니다: {candidate}") from exc
    current = boundary
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"canonical additions symlink 경로 component 금지: {current}")
    return candidate


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Linux renameat2(RENAME_NOREPLACE)로 session directory를 원자 발행한다."""

    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"발행할 staged session이 regular directory가 아닙니다: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_existing_symlink_components(destination.parent)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE)를 지원하지 않아 canonical 발행을 중단합니다")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(f"canonical session을 덮어쓰지 않습니다: {destination}")
        raise OSError(code, os.strerror(code), f"{source} -> {destination}")
    _fsync_directory(source.parent)
    _fsync_directory(destination.parent)


def _new_attempt_root(*, generation_id: str, row_number: int, attempt: int) -> Path:
    root = _repo_path(
        Path("results/recording_staging/record_session_batch") / generation_id
    )
    root.mkdir(parents=True, exist_ok=True)
    _reject_existing_symlink_components(root)
    name = (
        f"row{row_number:04d}_attempt{attempt}_{int(time.time())}_"
        f"{secrets.token_hex(8)}"
    )
    candidate = root / name
    candidate.mkdir(mode=0o700)
    return candidate


def _quarantine_staged_session(
    *,
    staged_session: Path,
    failed_root: Path,
    generation_id: str,
    row_number: int,
    attempt: int,
) -> Path:
    """QA 실패 raw를 canonical root 밖의 no-replace evidence로 이동한다."""

    batch_root = (
        failed_root
        / "batch_qa"
        / generation_id
        / f"row{row_number:04d}_attempt{attempt}_{int(time.time())}_{secrets.token_hex(4)}"
    )
    batch_root.mkdir(parents=True, exist_ok=False)
    destination = batch_root / staged_session.name
    _atomic_rename_noreplace(staged_session, destination)
    return destination


def _canonical_already_recorded(
    *,
    out_root: Path,
    rows: list[dict],
    source_list: Path,
    source_list_sha256: str,
) -> set[tuple[str, str, int, str, str, float]]:
    """exact plan·artifact·progress가 모두 맞는 canonical 세션만 resume 완료로 센다."""

    if not out_root.exists():
        return set()
    if not out_root.is_dir():
        raise ValueError(f"canonical additions out-root가 directory가 아닙니다: {out_root}")
    row_by_number = {int(row["source_row_number"]): row for row in rows}
    session_by_row: dict[int, str] = {}
    for child in sorted(out_root.iterdir(), key=lambda item: item.name):
        info = child.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"canonical additions root 내부 symlink 금지: {child}")
        if stat.S_ISREG(info.st_mode) and child.name == "batch_progress.csv":
            continue
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(
                "canonical additions root에는 session directory와 "
                f"batch_progress.csv만 허용합니다: {child}"
            )
        try:
            metadata = _read_session_metadata(child, repo_root=REPO_ROOT)
        except (OSError, RecordedGenerationError, ValueError) as exc:
            raise ValueError(f"canonical 기존 session metadata 검증 실패: {child}: {exc}") from exc
        if metadata.get("session_id") != child.name:
            raise ValueError(f"canonical 기존 session_id/directory 불일치: {child}")
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
        if not isinstance(plan, dict) or set(plan) != required_plan_keys:
            raise ValueError(f"canonical 기존 session collection_plan이 exact가 아닙니다: {child}")
        row_number = plan.get("source_row_number")
        if isinstance(row_number, bool) or not isinstance(row_number, int):
            raise ValueError(f"canonical 기존 session source row가 유효하지 않습니다: {child}")
        row = row_by_number.get(row_number)
        if row is None or row_number in session_by_row:
            raise ValueError(f"canonical 기존 session source row 중복/범위 오류: {child}")
        try:
            declared_source_list = _lexical_repo_path(str(plan["source_list"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"canonical 기존 session source_list 오류: {child}") from exc
        expected_plan = {
            "status": "exact",
            "source_list": plan["source_list"],
            "source_list_sha256": source_list_sha256,
            "source_row_number": row_number,
            "lineage_key": row["lineage_key"],
            "preassigned_split": row["split"],
            "split_source": "csv",
            "source_file_sha256": row["source_file_sha256"],
            "start_seconds": row["start_seconds"],
        }
        program = metadata.get("program")
        if (
            plan != expected_plan
            or declared_source_list != source_list
            or not isinstance(program, dict)
            or program.get("type") != "file"
            or program.get("file") != row["path"]
            or program.get("file_start_seconds") != row["start_seconds"]
            or metadata.get("source_family") != row["source_family"]
            or metadata.get("group_id") != row["group_id"]
            or metadata.get("preassigned_split") != row["split"]
        ):
            raise ValueError(f"canonical 기존 session/source plan 불일치: {child}")
        try:
            _validate_session_artifacts(
                session_dir=child,
                metadata=metadata,
                row=row,
                expected_seconds=float(row["seconds"]),
                repo_root=REPO_ROOT,
            )
        except (OSError, RecordedGenerationError, ValueError) as exc:
            raise ValueError(f"canonical 기존 session artifact 검증 실패: {child}: {exc}") from exc
        session_by_row[row_number] = child.name

    progress_path = out_root / "batch_progress.csv"
    if not session_by_row and not progress_path.exists():
        return set()
    if not progress_path.is_file() or progress_path.is_symlink():
        raise ValueError("canonical 기존 session에는 regular batch_progress.csv가 필요합니다")
    try:
        with progress_path.open(encoding="utf-8", newline="") as handle:
            progress = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"canonical batch_progress.csv 읽기 실패: {exc}") from exc
    successful: list[tuple[int, str]] = []
    for item in progress:
        if item.get("verdict") != "ok":
            continue
        row_text = item.get("source_row_number", "")
        if not row_text.isdigit() or not item.get("session_id"):
            raise ValueError("canonical batch_progress.csv PASS 행이 불완전합니다")
        row_number = int(row_text)
        planned = row_by_number.get(row_number)
        try:
            progress_seconds = float(item.get("seconds", ""))
            progress_start = float(item.get("start_seconds", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "canonical batch_progress.csv PASS 행 seconds가 없습니다"
            ) from exc
        if (
            planned is None
            or not np.isfinite(progress_seconds)
            or progress_seconds != float(planned["seconds"])
            or item.get("source_path") != planned["path"]
            or item.get("source_file_sha256") != planned["source_file_sha256"]
            or item.get("source_list_sha256") != source_list_sha256
            or item.get("lineage_key") != planned["lineage_key"]
            or item.get("preassigned_split") != planned["split"]
            or progress_start != float(planned["start_seconds"])
        ):
            raise ValueError(
                "canonical batch_progress.csv PASS 행이 exact source plan과 다릅니다"
            )
        successful.append((row_number, str(item["session_id"])))
    expected_pairs = sorted(session_by_row.items())
    if sorted(successful) != expected_pairs or len(successful) != len(set(successful)):
        raise ValueError(
            "canonical 기존 session과 batch_progress.csv PASS exact 집합이 다릅니다. "
            "QA 실패/불완전 session을 canonical root에 남긴 경우 자동 삭제하지 말고 "
            "별도 quarantine으로 이동하거나 새 generation-id를 사용하세요"
        )
    return {
        _plan_key(row_by_number[row_number], source_list_sha256)
        for row_number, _session_id in expected_pairs
    }


def _load_plan(
    sources_path: Path, *, families: list[str] | None, default_split: str | None
) -> tuple[list[dict], str]:
    raw = sources_path.read_bytes()
    source_list_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"소스 목록이 UTF-8이 아닙니다: {sources_path}: {exc}") from exc
    reader = csv.DictReader(text.splitlines())
    required = {"path", "seconds", "source_family", "group_id"}
    missing = sorted(required - set(reader.fieldnames or ()))
    if missing:
        raise ValueError(f"소스 목록 필수 열 누락: {missing}")

    family_filter = set(families or ())
    entries: list[dict] = []
    group_splits: dict[str, str] = {}
    for row_number, raw_entry in enumerate(reader, start=2):
        if family_filter and raw_entry["source_family"] not in family_filter:
            continue
        entry = dict(raw_entry)
        try:
            entry["source_family"] = validate_source_family(entry["source_family"])
            entry["group_id"] = validate_group_id(entry["group_id"])
            entry["lineage_key"] = validate_group_id(
                entry.get("lineage_key") or entry["group_id"]
            )
            entry["seconds"] = float(entry["seconds"])
            entry["start_seconds"] = float(entry.get("start_seconds") or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"소스 목록 {row_number}행 오류: {exc}") from exc
        if entry["seconds"] <= 0.0 or not np.isfinite(entry["seconds"]):
            raise ValueError(f"소스 목록 {row_number}행 seconds는 양수 finite여야 합니다")
        if entry["start_seconds"] < 0.0 or not np.isfinite(entry["start_seconds"]):
            raise ValueError(f"소스 목록 {row_number}행 start_seconds는 0 이상 finite여야 합니다")
        split = (entry.get("split") or default_split or "").strip()
        if split not in VALID_SPLITS:
            raise ValueError(
                f"소스 목록 {row_number}행 split이 사전 지정되지 않았습니다. "
                f"행의 split 또는 --preassigned-split {VALID_SPLITS} 중 하나가 필요합니다"
            )
        previous = group_splits.setdefault(entry["lineage_key"], split)
        if previous != split:
            raise ValueError(
                f"lineage_key={entry['lineage_key']!r}가 split {previous!r}/{split!r}에 걸칩니다"
            )
        source_path = _repo_path(entry["path"])
        if not source_path.is_file():
            raise ValueError(f"소스 목록 {row_number}행 파일 없음: {source_path}")
        import soundfile as sf

        try:
            source_info = sf.info(str(source_path))
        except RuntimeError as exc:
            raise ValueError(f"소스 목록 {row_number}행 파일을 열 수 없습니다: {exc}") from exc
        duration = float(source_info.frames) / float(source_info.samplerate)
        if entry["start_seconds"] + entry["seconds"] > duration + 1e-9:
            raise ValueError(
                f"소스 목록 {row_number}행 window가 파일 길이를 넘습니다: "
                f"start={entry['start_seconds']}, seconds={entry['seconds']}, duration={duration}"
            )
        source_digest = _sha256(source_path)
        declared_digest = (entry.get("source_file_sha256") or "").strip()
        if declared_digest:
            if (
                len(declared_digest) != 64
                or any(ch not in "0123456789abcdef" for ch in declared_digest)
            ):
                raise ValueError(
                    f"소스 목록 {row_number}행 source_file_sha256은 64자리 소문자 hex여야 합니다"
                )
            if declared_digest != source_digest:
                raise ValueError(
                    f"소스 목록 {row_number}행 source_file_sha256 불일치: "
                    f"declared={declared_digest}, actual={source_digest}"
                )
        entry["split"] = split
        entry["source_row_number"] = row_number
        entry["source_file_sha256"] = source_digest
        entries.append(entry)
    if not entries:
        raise ValueError("선택된 소스가 없습니다")
    return entries, source_list_sha256


def _pump_stream(stream, console, log_handle, log_lock: threading.Lock, sink: list[str]) -> None:
    try:
        for line in iter(stream.readline, ""):
            sink.append(line)
            console.write(line)
            console.flush()
            with log_lock:
                log_handle.write(line)
                log_handle.flush()
    finally:
        stream.close()


def run_child_live(command: list[str], *, timeout_seconds: float, log_path: Path) -> ChildResult:
    """Timeout 시 SIGTERM partial-raw receipt를 기다린 뒤에만 SIGKILL한다."""

    if timeout_seconds <= 0.0 or not np.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds는 양수 finite여야 합니다")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    with log_path.open("x", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None and process.stderr is not None
        lock = threading.Lock()
        threads = [
            threading.Thread(
                target=_pump_stream,
                args=(process.stdout, sys.stdout, log_handle, lock, stdout_lines),
                daemon=True,
            ),
            threading.Thread(
                target=_pump_stream,
                args=(process.stderr, sys.stderr, log_handle, lock, stderr_lines),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            print(
                f"[중단] 세션 hard timeout {timeout_seconds:.1f}초 초과 — 자식을 종료합니다.",
                file=sys.stderr,
                flush=True,
            )
            process.terminate()
            try:
                returncode = process.wait(timeout=GRACEFUL_CHILD_STOP_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                returncode = 124
        for thread in threads:
            thread.join(timeout=2.0)
    return ChildResult(
        returncode=124 if timed_out else int(returncode),
        stdout="".join(stdout_lines),
        stderr="".join(stderr_lines),
        timed_out=timed_out,
    )


def _read_child_failure_evidence(
    result: ChildResult, *, failed_root: Path
) -> ChildFailureEvidence | None:
    """record_duct가 발행한 durable failure receipt를 검증해 읽는다.

    stderr marker 자체는 위치 힌트일 뿐 권위 자료가 아니다. 해당 경로가 이번 failure
    root 내부의 non-symlink directory인지, ``failure.json``이 regular file인지, marker와
    receipt의 stage/reason이 같은지까지 확인한 경우에만 progress 근거로 채택한다.
    """

    marker_lines = [
        line[len(FAILURE_RECEIPT_MARKER) :]
        for line in result.stderr.splitlines()
        if line.startswith(FAILURE_RECEIPT_MARKER)
    ]
    expected_root = _lexical_repo_path(failed_root)
    filesystem_root = Path(expected_root.anchor)
    try:
        expected_root = reject_symlink_components(
            expected_root,
            root=filesystem_root,
        )
    except (OSError, ValueError):
        return None
    for raw_marker in reversed(marker_lines):
        try:
            marker = json.loads(raw_marker)
            required = {
                "schema_version",
                "failure_stage",
                "failure_reason",
                "failure_artifact",
                "failure_receipt",
            }
            if not isinstance(marker, dict) or not required.issubset(marker):
                continue
            if marker["schema_version"] != 1:
                continue
            stage = marker["failure_stage"]
            reason = marker["failure_reason"]
            if (
                not isinstance(stage, str)
                or not stage
                or not isinstance(reason, str)
                or not reason
            ):
                continue
            if not os.path.isabs(str(marker["failure_artifact"])):
                continue
            if not os.path.isabs(str(marker["failure_receipt"])):
                continue
            artifact = Path(os.path.abspath(str(marker["failure_artifact"])))
            receipt = Path(os.path.abspath(str(marker["failure_receipt"])))
            artifact.relative_to(expected_root)
            artifact = reject_symlink_components(artifact, root=expected_root)
            if not artifact.is_dir():
                continue
            if receipt != artifact / "failure.json":
                continue
            snapshot = read_regular_file_snapshot(
                receipt,
                root=expected_root,
                label="record_duct failure receipt",
                capture_bytes=True,
            )
            assert snapshot.data is not None
            payload = json.loads(snapshot.data.decode("utf-8"))
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version") != 1
                or payload.get("status") != "failed_capture"
                or payload.get("failure_stage") != stage
                or payload.get("failure_reason") != reason
                or type(payload.get("raw_available")) is not bool
            ):
                continue
            return ChildFailureEvidence(
                stage=stage,
                reason=reason,
                artifact=artifact,
                receipt=receipt,
                receipt_sha256=snapshot.sha256,
                raw_available=bool(payload["raw_available"]),
            )
        except (OSError, TypeError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            continue
    return None


def _fallback_child_failure_detail(result: ChildResult) -> str:
    """구형 record_duct/timeout을 위한 사람이 읽는 하위 호환 fallback."""

    detail_lines = [
        line
        for line in (result.stderr + result.stdout).splitlines()
        if line and not line.startswith(FAILURE_RECEIPT_MARKER)
    ]
    return detail_lines[-1] if detail_lines else "자식 산출물 없음"


def _confirm_reconnect(reason: str) -> bool:
    """분리 안내 뒤 다음 출력 전에 현장 사용자의 재연결 확인을 다시 받는다."""

    if not sys.stdin.isatty():
        print(
            f"[중단] {reason}: 다음 출력 전 스피커 재연결 확인에는 대화형 TTY가 필요합니다.",
            file=sys.stderr,
        )
        return False
    input(
        f"{reason}: 스피커를 분리한 상태에서 직전 raw/log를 확인하세요. "
        "다음 출력 직전에 다시 연결하고 볼륨/배선/geometry를 확인한 뒤 Enter: "
    )
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="data/source_pool_v2/sources.csv")
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument("--out-root", default="data/recorded")
    parser.add_argument("--failed-root", default="results/recording_failures/record_duct")
    parser.add_argument("--log-root", default="results/recording_logs/record_session_batch")
    parser.add_argument(
        "--amplitude",
        type=float,
        default=CANONICAL_RECORDING_AMPLITUDE,
        help=(
            "diagnostic/legacy file 재생 digital 진폭. 현행 canonical additions에서는 이 값을 "
            "출력에 쓰지 않고 기본 0.06을 unused legacy sentinel로만 검증한 뒤, "
            "검증된 source-gain plan의 행별 receipt-bound <=0.006 값을 child에 전달합니다"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="이번 실행에서 녹음할 최대 세션 수")
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument(
        "--session-timeout-overhead-seconds",
        type=float,
        default=120.0,
        help="각 행의 seconds+settle에 더할 정렬/저장 hard-timeout 여유",
    )
    parser.add_argument(
        "--retry-once",
        action="store_true",
        help="실패 세션을 딱 한 번 다시 재생한다. 기본은 재시도 없음",
    )
    parser.add_argument("--no-retry", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--preassigned-split", choices=VALID_SPLITS, default=None)
    parser.add_argument(
        "--canonical-additions-generation",
        default=None,
        help=(
            "추가 19세션 canonical 수집 모드. 현행 exact generation-id "
            f"{CANONICAL_GENERATION_ID!r}와 source plan/out-root를 강제합니다"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="파일/오디오를 변경하지 않고 계획만 검증")
    parser.add_argument("--confirm-speaker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--confirm-user-present", action="store_true")
    parser.add_argument("--confirm-volume-minimum", action="store_true")
    parser.add_argument("--confirm-routing-and-geometry", action="store_true")
    parser.add_argument(
        "--recording-level-campaign",
        default=None,
        help="fresh recording-level campaign.json 저장소 상대경로",
    )
    parser.add_argument(
        "--recording-level-campaign-sha256",
        default=None,
        help="campaign.json의 외부 SHA-256 anchor",
    )
    parser.add_argument("--confirm-same-amplifier-setting", action="store_true")
    parser.add_argument(
        "--source-gain-plan",
        default=None,
        help="strict-P 기반 source별 gain plan JSON 저장소 상대경로",
    )
    parser.add_argument(
        "--source-gain-plan-sha256",
        default=None,
        help="source별 gain plan JSON의 외부 SHA-256 anchor",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.no_retry and args.retry_once:
        parser.error("--retry-once와 legacy --no-retry를 함께 쓸 수 없습니다")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit은 양수여야 합니다")
    for name in ("settle_seconds", "session_timeout_overhead_seconds"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')}는 0 이상 finite여야 합니다")
    if (
        not np.isfinite(args.amplitude)
        or args.amplitude <= 0.0
        or args.amplitude > MAX_RECORDING_OUTPUT_PEAK
    ):
        parser.error(
            f"--amplitude는 0 초과 {MAX_RECORDING_OUTPUT_PEAK:.3f} 이하여야 합니다"
        )

    confirmations = {
        "--confirm-user-present": args.confirm_user_present,
        "--confirm-volume-minimum": args.confirm_volume_minimum,
        "--confirm-routing-and-geometry": args.confirm_routing_and_geometry,
    }
    missing_confirmations = [name for name, confirmed in confirmations.items() if not confirmed]
    if missing_confirmations and not args.dry_run:
        print("[중단] 실기 확인 누락: " + ", ".join(missing_confirmations), file=sys.stderr)
        return 2

    sources_path = _repo_path(args.sources)
    if not sources_path.is_file():
        print(f"[중단] 소스 목록이 없습니다: {sources_path}", file=sys.stderr)
        return 2
    try:
        entries, source_list_sha256 = _load_plan(
            sources_path, families=args.families, default_split=args.preassigned_split
        )
    except (OSError, ValueError) as exc:
        print(f"[중단] 수집 계획 오류: {exc}", file=sys.stderr)
        return 2

    out_root = _repo_path(args.out_root)
    canonical_rows: list[dict] | None = None
    if args.canonical_additions_generation is not None:
        try:
            generation_id = validate_generation_id(args.canonical_additions_generation)
        except ValueError as exc:
            parser.error(str(exc))
        if generation_id != CANONICAL_GENERATION_ID:
            parser.error(
                "현행 exact recorded generation-id는 "
                f"{CANONICAL_GENERATION_ID!r}입니다"
            )
        expected_sources = _lexical_repo_path(
            f"{SOURCE_PLAN_ROOT}/{generation_id}.csv"
        )
        expected_out = _lexical_repo_path(f"{ADDITIONS_ROOT}/{generation_id}")
        actual_sources = _lexical_repo_path(sources_path)
        try:
            actual_out = _reject_existing_symlink_components(out_root)
        except ValueError as exc:
            parser.error(str(exc))
        if actual_sources != expected_sources:
            parser.error(
                "canonical additions source plan 경로 불일치: "
                f"expected={expected_sources}, actual={actual_sources}"
            )
        if actual_out != expected_out:
            parser.error(
                "canonical additions out-root 경로 불일치: "
                f"expected={expected_out}, actual={actual_out}"
            )
        if args.families is not None or args.preassigned_split is not None:
            parser.error(
                "canonical additions는 CSV 19행 전체의 family/split을 사용하며 "
                "--families/--preassigned-split override를 허용하지 않습니다"
            )
        if len(entries) != ADDITION_SESSION_COUNT:
            parser.error(
                f"canonical additions source plan은 정확히 {ADDITION_SESSION_COUNT}행이어야 합니다"
            )
        if args.amplitude != CANONICAL_RECORDING_AMPLITUDE:
            parser.error(
                "현행 canonical additions에서 --amplitude는 출력값이 아닌 "
                "unused legacy sentinel이며 "
                f"기본 exact {CANONICAL_RECORDING_AMPLITUDE:.2f}로 두어야 합니다"
            )
        with sources_path.open(encoding="utf-8", newline="") as handle:
            fieldnames = tuple(csv.DictReader(handle).fieldnames or ())
        if fieldnames != SOURCE_PLAN_FIELDS:
            parser.error("canonical additions source plan header가 exact 계약과 다릅니다")
        try:
            (
                canonical_snapshot,
                canonical_rows,
                _lineage_sha,
                _selection_evidence,
            ) = _read_source_plan(
                repo_root=REPO_ROOT,
                relative=sources_path.relative_to(REPO_ROOT).as_posix(),
                require_source_files=True,
            )
        except (OSError, RecordedGenerationError, ValueError) as exc:
            parser.error(f"canonical additions 권위 source-plan 검증 실패: {exc}")
        if (
            canonical_snapshot.sha256 != source_list_sha256
            or len(canonical_rows) != ADDITION_SESSION_COUNT
        ):
            parser.error(
                "canonical additions source-plan snapshot이 initial dry-run snapshot과 다릅니다"
            )
    if canonical_rows is not None:
        if args.retry_once:
            parser.error(
                "canonical additions는 실패 raw를 먼저 분석해야 하므로 --retry-once를 "
                "허용하지 않습니다"
            )
        try:
            done = _canonical_already_recorded(
                out_root=out_root,
                rows=canonical_rows,
                source_list=expected_sources,
                source_list_sha256=source_list_sha256,
            )
        except (OSError, ValueError) as exc:
            parser.error(f"canonical additions 기존 상태 검증 실패: {exc}")
    else:
        done = already_recorded(out_root)

    try:
        level_campaign = _validate_batch_recording_level_campaign(
            args, required=canonical_rows is not None and not args.dry_run
        )
    except (OSError, RecordingLevelCampaignError, ValueError) as exc:
        parser.error(f"recording level campaign 검증 실패: {exc}")
    try:
        source_gain_plan = _validate_batch_source_gain_plan(
            args,
            required=canonical_rows is not None and not args.dry_run,
            source_list_sha256=source_list_sha256,
        )
    except (OSError, RecordingSourceGainError, ValueError) as exc:
        parser.error(f"recording source gain plan 검증 실패: {exc}")
    source_gain_by_row: dict[int, float] = {}
    if canonical_rows is not None and not args.dry_run:
        try:
            assert source_gain_plan is not None
            source_gain_by_row = _canonical_source_gain_by_row(
                source_gain_plan, entries
            )
        except (AssertionError, RecordingSourceGainError, ValueError) as exc:
            parser.error(f"recording source gain row 매핑 실패: {exc}")
    pending = [entry for entry in entries if _plan_key(entry, source_list_sha256) not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        print("정확히 같은 source-list SHA/행/lineage/split의 모든 소스가 이미 녹음되었습니다.")
        return 0

    audible_seconds = sum(float(entry["seconds"]) for entry in pending)
    output_open_seconds = audible_seconds + len(pending) * float(args.settle_seconds)
    connected_upper_seconds = (
        output_open_seconds
        + len(pending) * INPUT_ONLY_PREFLIGHT_SECONDS
        + max(0, len(pending) - 1) * float(args.settle_seconds)
    )
    print(
        f"소스 {len(entries)}개 · exact 완료 {len(done)}개 · 이번 계획 {len(pending)}개\n"
        f"source-list SHA256: {source_list_sha256}\n"
        f"예상 audible: {audible_seconds:.1f}초 ({audible_seconds / 60.0:.2f}분)\n"
        f"예상 output-open: {output_open_seconds:.1f}초\n"
        f"예상 connected 상한(분석/저장 제외): {connected_upper_seconds:.1f}초\n"
        f"재생 amplitude: "
        f"{'receipt-bound dynamic per-row' if source_gain_by_row else f'{args.amplitude:.2f} diagnostic/dry-run'} "
        f"(공용 peak 안전 상한 {MAX_RECORDING_OUTPUT_PEAK:.2f})\n"
        f"세션 hard timeout: 각 seconds + settle + {args.session_timeout_overhead_seconds:.1f}초\n"
        f"자동 재시도: {'실패당 1회(opt-in)' if args.retry_once else '없음'}\n"
        f"recording-level campaign: "
        f"{level_campaign['campaign_id'] if level_campaign is not None else 'diagnostic-unbound'}\n"
        f"source-gain plan: "
        f"{source_gain_plan['plan_sha256'] if source_gain_plan is not None else 'not-supplied'}"
    )
    for index, entry in enumerate(pending, start=1):
        print(
            f"  {index:03d} row={entry['source_row_number']} split={entry['split']} "
            f"lineage={entry['lineage_key']} start={entry['start_seconds']:.3f}s "
            f"seconds={entry['seconds']:.1f} "
            f"amplitude={source_gain_by_row.get(int(entry['source_row_number']), args.amplitude):.6f} "
            f"source={entry['path']} sha256={entry['source_file_sha256']}"
        )
    if args.dry_run:
        if canonical_rows is not None and level_campaign is None:
            print(
                "[DRY-RUN 안내] live canonical batch에는 fresh recording-level "
                "campaign path/SHA와 same-amplifier 확인이 필수입니다"
            )
        print("[DRY-RUN PASS] 파일 생성/수정 및 오디오 장치 open 없음")
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    progress_path = out_root / "batch_progress.csv"
    rows: list[dict] = []
    if progress_path.exists():
        rows = list(csv.DictReader(progress_path.open(encoding="utf-8")))

    consecutive_failures = 0
    run_had_failure = False
    started = time.monotonic()
    for index, entry in enumerate(pending, start=1):
        row_amplitude = source_gain_by_row.get(
            int(entry["source_row_number"]), float(args.amplitude)
        )
        if canonical_rows is not None and not (
            0.0 < row_amplitude
            <= float(source_gain_plan["payload"]["contract"][
                "reference_amplitude_millionths"
            ]) / 1_000_000.0
        ):
            raise RuntimeError(
                "canonical child amplitude가 receipt-bound dynamic cap을 넘었습니다"
            )
        command_prefix = [
            sys.executable,
            str(REPO_ROOT / "scripts/data/record_duct.py"),
            "--program", "file", "--file", entry["path"],
            "--hardware", str(args.hardware),
            "--file-start-seconds", str(entry["start_seconds"]),
            "--seconds", str(entry["seconds"]), "--amplitude", str(row_amplitude),
            "--settle-seconds", str(args.settle_seconds),
            "--source-family", entry["source_family"], "--group-id", entry["group_id"],
            "--source-list", str(sources_path), "--source-list-sha256", source_list_sha256,
            "--source-row-number", str(entry["source_row_number"]),
            "--lineage-key", entry["lineage_key"], "--preassigned-split", entry["split"],
            "--confirm-user-present", "--confirm-volume-minimum", "--confirm-routing-and-geometry",
        ]
        if level_campaign is not None:
            command_prefix.extend(
                [
                    "--recording-level-campaign",
                    str(args.recording_level_campaign),
                    "--recording-level-campaign-sha256",
                    str(args.recording_level_campaign_sha256).lower(),
                    "--confirm-same-amplifier-setting",
                ]
            )
        if canonical_rows is not None:
            command_prefix.extend(
                [
                    "--require-recording-level-campaign",
                    "--source-gain-plan",
                    str(args.source_gain_plan),
                    "--source-gain-plan-sha256",
                    str(args.source_gain_plan_sha256).lower(),
                    "--require-source-gain-plan",
                ]
            )
        timeout_seconds = (
            float(entry["seconds"]) + float(args.settle_seconds)
            + float(args.session_timeout_overhead_seconds)
        )
        elapsed = time.monotonic() - started
        print(
            f"[{index}/{len(pending)}] {entry['source_family']:11s} "
            f"{Path(entry['path']).name} (경과 {elapsed / 60.0:.1f}분, "
            f"timeout {timeout_seconds:.1f}초)"
        )

        attempts = 2 if args.retry_once else 1
        result = ChildResult(1, "", "", False)
        created: list[str] = []
        child_out_root = out_root
        completed_attempt = 1
        for attempt in range(1, attempts + 1):
            completed_attempt = attempt
            child_out_root = (
                _new_attempt_root(
                    generation_id=str(args.canonical_additions_generation),
                    row_number=int(entry["source_row_number"]),
                    attempt=attempt,
                )
                if canonical_rows is not None
                else out_root
            )
            command = [
                *command_prefix,
                "--out-root",
                str(child_out_root),
                "--failed-root",
                str(_repo_path(args.failed_root)),
            ]
            before = {p.parent.name for p in child_out_root.glob("*/session.json")}
            log_name = (
                f"row{entry['source_row_number']:04d}_attempt{attempt}_"
                f"{int(time.time())}_{secrets.token_hex(4)}.log"
            )
            result = run_child_live(
                command,
                timeout_seconds=timeout_seconds,
                log_path=_repo_path(args.log_root) / log_name,
            )
            created = sorted(
                {p.parent.name for p in child_out_root.glob("*/session.json")} - before
            )
            if result.returncode == 0 and len(created) == 1:
                break
            if attempt < attempts:
                print("출력 종료 — 지금 스피커를 분리하세요.", flush=True)
                if not _confirm_reconnect("명시적 --retry-once"):
                    break
                print("[명시적 재시도] --retry-once가 지정되어 같은 세션을 한 번 더 재생합니다.")

        row = {
            "source_family": entry["source_family"], "group_id": entry["group_id"],
            "lineage_key": entry["lineage_key"], "preassigned_split": entry["split"],
            "start_seconds": entry["start_seconds"],
            "seconds": entry["seconds"],
            "amplitude": row_amplitude,
            "source_path": entry["path"], "source_file_sha256": entry["source_file_sha256"],
            "source_list_sha256": source_list_sha256,
            "source_row_number": entry["source_row_number"], "returncode": result.returncode,
            "timed_out": result.timed_out, "session_id": "",
            "raw_available": False,
        }
        if result.returncode != 0 or len(created) != 1:
            row["verdict"] = "record_failed"
            evidence = _read_child_failure_evidence(
                result, failed_root=_repo_path(args.failed_root)
            )
            detail = (
                evidence.reason
                if evidence is not None
                else _fallback_child_failure_detail(result)
            )
            if len(created) > 1:
                detail = f"자식이 session {len(created)}개를 발행함"
            row["detail"] = detail
            if evidence is not None:
                row["failure_stage"] = evidence.stage
                row["failure_artifact"] = str(evidence.artifact)
                row["failure_receipt"] = str(evidence.receipt)
                row["failure_receipt_sha256"] = evidence.receipt_sha256
                row["raw_available"] = evidence.raw_available
            if canonical_rows is not None and created:
                quarantined = []
                for session_id in created:
                    quarantined.append(
                        _quarantine_staged_session(
                            staged_session=child_out_root / session_id,
                            failed_root=_repo_path(args.failed_root),
                            generation_id=str(args.canonical_additions_generation),
                            row_number=int(entry["source_row_number"]),
                            attempt=completed_attempt,
                        )
                    )
                quarantine_text = ";".join(str(path) for path in quarantined)
                if row.get("failure_artifact"):
                    row["failure_artifact"] += ";" + quarantine_text
                else:
                    row["failure_artifact"] = quarantine_text
            consecutive_failures += 1
            run_had_failure = True
            print(f"    [실패] {row['detail']}")
        else:
            session_dir = child_out_root / created[0]
            row["raw_available"] = True
            try:
                stats = analyse_session(session_dir)
                ok, reason = qa_verdict(stats)
                if ok and canonical_rows is not None:
                    canonical_row = next(
                        item
                        for item in canonical_rows
                        if int(item["source_row_number"])
                        == int(entry["source_row_number"])
                    )
                    metadata = _read_session_metadata(session_dir, repo_root=REPO_ROOT)
                    _validate_session_artifacts(
                        session_dir=session_dir,
                        metadata=metadata,
                        row=canonical_row,
                        expected_seconds=float(canonical_row["seconds"]),
                        repo_root=REPO_ROOT,
                    )
            except (OSError, RuntimeError, StopIteration, ValueError, RecordedGenerationError) as exc:
                stats = {}
                ok, reason = False, f"batch QA/exact artifact 검증 실패: {exc}"
            row.update(stats)
            row["verdict"] = "ok" if ok else "qa_failed"
            row["detail"] = reason
            if canonical_rows is not None:
                if ok:
                    destination = out_root / created[0]
                    try:
                        _atomic_rename_noreplace(session_dir, destination)
                    except (OSError, RuntimeError, ValueError) as exc:
                        ok = False
                        row["verdict"] = "publish_failed"
                        row["detail"] = f"canonical atomic publish 실패: {exc}"
                    else:
                        row["session_id"] = created[0]
                if not ok:
                    if session_dir.exists():
                        quarantined = _quarantine_staged_session(
                            staged_session=session_dir,
                            failed_root=_repo_path(args.failed_root),
                            generation_id=str(args.canonical_additions_generation),
                            row_number=int(entry["source_row_number"]),
                            attempt=completed_attempt,
                        )
                        row["failure_artifact"] = str(quarantined)
                    run_had_failure = True
            else:
                row["session_id"] = created[0]
                run_had_failure = run_had_failure or not ok
            consecutive_failures = 0 if ok else consecutive_failures + 1
            if stats:
                print(
                    f"    [{'PASS' if ok else 'QA 실패'}] ERR {stats['err_rms_dbfs']:6.1f} dBFS "
                    f"peak {stats['err_peak']:.3f} · REF {stats['ref_rms_dbfs']:6.1f} dBFS · "
                    f"clip {stats['clip_ratio']:.3%}"
                )
            else:
                print(f"    [QA 실패] {row['detail']}")

        rows.append(row)
        write_csv(progress_path, rows)
        if canonical_rows is not None and row.get("verdict") != "ok":
            print(
                "[중단] canonical 실패 증거는 progress의 raw_available/"
                "failure_receipt로 확인합니다. raw_available=false이면 보존 raw가 "
                "없으며, 근거 없이 보존됐다고 간주하지 않습니다. 오프라인 분석 "
                "없이 다음 소스를 재생하지 않습니다.",
                file=sys.stderr,
            )
            print("출력 종료 — 지금 스피커를 분리하세요.", flush=True)
            return 1
        if consecutive_failures >= args.max_consecutive_failures:
            print(
                f"[중단] {consecutive_failures}회 연속 실패. 오프라인 raw/로그를 먼저 분석하세요.",
                file=sys.stderr,
            )
            print("출력 종료 — 지금 스피커를 분리하세요.", flush=True)
            return 1
        if index < len(pending):
            if not _confirm_reconnect(f"다음 세션 {index + 1}/{len(pending)}"):
                return 2
            time.sleep(args.settle_seconds)

    print("출력 종료 — 지금 스피커를 분리하세요.", flush=True)
    ok_rows = [row for row in rows if row.get("verdict") == "ok"]
    total_seconds = sum(float(row.get("seconds", 0) or 0) for row in ok_rows)
    print(f"완료 세션 {len(ok_rows)}개 · {total_seconds / 60.0:.1f}분")
    print(f"진행 기록: {progress_path}")
    return 1 if run_had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
