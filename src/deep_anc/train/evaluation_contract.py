"""recorded-val 선택과 recorded-test 1회 개봉의 파일 capability 계약."""

from __future__ import annotations

import hashlib
import ctypes
import errno
import io
import json
import math
import os
import secrets
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CAPABILITY_ENV = "DEEP_ANC_RECORDED_TEST_TOKEN"
LEDGER_ROOT = Path("results/recorded_test_ledger")
VAL_BORDERLINE_MARGIN_DB = 0.3
OFFICIAL_FINETUNE_SEEDS = frozenset({20260803, 20260903})


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    content: bytes
    sha256: str


def snapshot_regular_file(path: str | Path) -> FileSnapshot:
    """한 immutable regular-file snapshot으로 bytes/hash/load를 함께 만든다."""

    target = Path(os.path.abspath(Path(path)))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise ValueError(f"regular-file snapshot을 열 수 없습니다: {target}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"regular file만 허용합니다: {target}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in stable):
        raise ValueError(f"regular-file snapshot 도중 파일이 변경됐습니다: {target}")
    if len(content) != int(after.st_size):
        raise ValueError(f"regular-file snapshot byte 수가 size와 다릅니다: {target}")
    try:
        pathname = target.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"regular-file snapshot pathname이 사라졌습니다: {target}") from exc
    if stat.S_ISLNK(pathname.st_mode) or (pathname.st_dev, pathname.st_ino) != (
        after.st_dev,
        after.st_ino,
    ):
        raise ValueError(f"regular-file snapshot pathname이 retarget됐습니다: {target}")
    return FileSnapshot(target, content, hashlib.sha256(content).hexdigest())


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def write_json_exclusive(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(os.path.abspath(Path(path)))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"JSON artifact를 덮어쓸 수 없습니다: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        content = _json_bytes(payload)
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, target, follow_symlinks=False)
        directory_fd = os.open(
            target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_json_snapshot(path: str | Path) -> tuple[dict[str, Any], FileSnapshot]:
    snapshot = snapshot_regular_file(path)
    try:
        payload = json.loads(snapshot.content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON artifact가 손상됐습니다: {snapshot.path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact 최상위가 mapping이 아닙니다: {snapshot.path}")
    return payload, snapshot


def _sha256_identity(value: object, *, name: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"test ledger {name}가 64자리 SHA-256이 아닙니다")
    return text


def _npz_scalar(data: np.lib.npyio.NpzFile, key: str) -> object:
    if key not in data.files:
        raise ValueError(f"recorded-val G4 필드가 없습니다: {key}")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"recorded-val G4 {key}는 scalar여야 합니다")
    return value.reshape(-1)[0].item()


def classify_recorded_val_metrics(metrics_bytes: bytes) -> dict[str, Any]:
    """capability와 pipeline이 공유하는 G4 clear-pass/borderline 분류기."""

    try:
        archive = np.load(io.BytesIO(metrics_bytes), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("recorded-val metrics.npz가 손상됐습니다") from exc
    with archive as data:
        verdict = str(_npz_scalar(data, "g4_verdict"))
        threshold = float(_npz_scalar(data, "g4_max_out_of_band_amplification_db"))
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("G4 do-no-harm threshold가 유효하지 않습니다")
        margins = {
            "trusted_mean_db": -float(_npz_scalar(data, "nmse_trusted_mean_db")),
            "fullband_mean_db": -float(_npz_scalar(data, "nmse_fullband_mean_db")),
            "worst_source_mean_db": -float(
                _npz_scalar(data, "g4_worst_source_trusted_mean_db")
            ),
            "worst_source_worst10_db": -float(
                _npz_scalar(data, "g4_worst_source_trusted_worst10_db")
            ),
            "do_no_harm_db": float(
                _npz_scalar(data, "g4_worst_octave_worst10_db")
            )
            + threshold,
        }
        ci_hi = np.asarray(data["source_trusted_ci_hi_db"], dtype=np.float64)
        if ci_hi.size == 0 or not bool(np.isfinite(ci_hi).all()):
            raise ValueError("recorded-val bootstrap CI 상단이 finite/nonempty가 아닙니다")
        margins["worst_source_ci_hi_db"] = -float(np.max(ci_hi))
        if not all(math.isfinite(value) for value in margins.values()):
            raise ValueError("recorded-val G4 margin에 NaN/Inf가 있습니다")
        flags = {
            key: bool(_npz_scalar(data, key))
            for key in (
                "g4_trusted_pass",
                "g4_fullband_pass",
                "g4_source_pass",
                "g4_do_no_harm_pass",
                "g4_power_pass",
                "g4_ci_pass",
                "g4_pass",
            )
        }
        selection_metric = float(_npz_scalar(data, "nmse_trusted_worst10_mean_db"))
    if not math.isfinite(selection_metric):
        raise ValueError("recorded-val 선택 지표가 non-finite")
    minimum = min(margins.values())
    near_boundary = any(abs(value) <= VAL_BORDERLINE_MARGIN_DB for value in margins.values())
    numeric_pass = all(value >= 0.0 for value in margins.values())
    discrete_pass = all(flags.values()) and verdict == "PASS"
    if verdict == "INCONCLUSIVE" or near_boundary:
        status = "borderline"
    elif numeric_pass and discrete_pass:
        status = "clear_pass"
    else:
        status = "clear_fail"
    return {
        "status": status,
        "boundary_margin_db": VAL_BORDERLINE_MARGIN_DB,
        "minimum_margin_db": minimum,
        "margins_db": margins,
        "g4_verdict": verdict,
        "g4_flags": flags,
        "selection_metric_db": selection_metric,
    }


def _validate_selection_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    selected = payload.get("selected")
    if payload.get("selection_split") != "val" or not isinstance(selected, dict):
        raise ValueError("recorded-val selection이 고정되지 않았습니다")
    checkpoint = snapshot_regular_file(selected.get("checkpoint", ""))
    manifest = snapshot_regular_file(payload.get("manifest", ""))
    metrics = snapshot_regular_file(
        Path(str(selected.get("evaluation_dir", ""))) / "metrics.npz"
    )
    if checkpoint.sha256 != selected.get("checkpoint_sha256"):
        raise ValueError("selection checkpoint bytes가 바뀌었습니다")
    if manifest.sha256 != payload.get("manifest_sha256"):
        raise ValueError("selection manifest bytes가 바뀌었습니다")
    if metrics.sha256 != selected.get("metrics_sha256"):
        raise ValueError("selection val metrics bytes가 바뀌었습니다")
    with np.load(io.BytesIO(metrics.content), allow_pickle=False) as data:
        provenance = {
            "split": str(_npz_scalar(data, "split")),
            "checkpoint_sha256": str(_npz_scalar(data, "checkpoint_sha256")),
            "manifest_sha256": str(_npz_scalar(data, "manifest_sha256")),
            "experiment_contract_sha256": str(
                _npz_scalar(data, "experiment_contract_sha256")
            ),
        }
    if provenance != {
        "split": "val",
        "checkpoint_sha256": checkpoint.sha256,
        "manifest_sha256": manifest.sha256,
        "experiment_contract_sha256": payload.get("experiment_contract_sha256"),
    }:
        raise ValueError("selection val metrics provenance가 checkpoint/manifest/contract와 다릅니다")
    decision = classify_recorded_val_metrics(metrics.content)
    if selected.get("decision") != decision:
        raise ValueError("selection decision이 val metrics 재분류와 다릅니다")
    return decision


def validate_test_open_selection(payload: dict[str, Any]) -> None:
    """single clear-pass 또는 검증된 2-seed final만 test 개봉을 허용한다."""

    campaign = _sha256_identity(
        payload.get("seed_neutral_campaign_sha256"), name="seed-neutral campaign"
    )
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("selection decision이 없습니다")
    if decision.get("status") != "cross_seed_final":
        current = _validate_selection_candidate(payload)
        if payload.get("seed") != 20260803 or current.get("status") != "clear_pass":
            raise ValueError("single-seed test는 seed 20260803 val clear-pass만 허용합니다")
        if decision != current:
            raise ValueError("single-seed top-level decision이 val metrics와 다릅니다")
        return

    records = payload.get("seed_selections")
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("cross-seed final에는 두 seed selection snapshot이 필요합니다")
    bundles: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("cross-seed seed selection record가 mapping이 아닙니다")
        bundle, snapshot = read_json_snapshot(record.get("path", ""))
        if snapshot.sha256 != record.get("sha256"):
            raise ValueError("cross-seed seed selection bytes가 final 뒤 바뀌었습니다")
        seed = record.get("seed")
        if seed not in OFFICIAL_FINETUNE_SEEDS or bundle.get("seed") != seed:
            raise ValueError("cross-seed official seed가 다릅니다")
        if bundle.get("seed_neutral_campaign_sha256") != campaign:
            raise ValueError("cross-seed bundle campaign digest가 다릅니다")
        if bundle.get("manifest_sha256") != payload.get("manifest_sha256"):
            raise ValueError("cross-seed bundle recorded manifest가 다릅니다")
        current = _validate_selection_candidate(bundle)
        if bundle.get("decision") != current:
            raise ValueError("cross-seed bundle top-level decision이 metrics와 다릅니다")
        bundles.append((int(seed), bundle, current))
    if {seed for seed, _, _ in bundles} != set(OFFICIAL_FINETUNE_SEEDS):
        raise ValueError("cross-seed final official seed 집합이 다릅니다")
    first = next(item for item in bundles if item[0] == 20260803)
    if first[2].get("status") != "borderline":
        raise ValueError("첫 seed가 borderline/INCONCLUSIVE가 아니어서 2-seed 자격이 없습니다")
    eligible = [
        item
        for item in bundles
        if item[2].get("g4_verdict") == "PASS"
        and float(item[2].get("minimum_margin_db", float("-inf"))) >= 0.0
    ]
    if not eligible:
        raise ValueError("cross-seed final에 val G4 PASS winner가 없습니다")
    winner = max(eligible, key=lambda item: (float(item[2]["minimum_margin_db"]), -item[0]))
    if payload.get("seed") != winner[0] or payload.get("selected") != winner[1].get("selected"):
        raise ValueError("cross-seed final winner가 margin-max selection과 다릅니다")


def canonical_test_ledger_paths_from_payload(
    selection: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """동일 selection snapshot payload에서 3-SHA ledger 경로를 유도한다."""

    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("test ledger selection.selected가 없습니다")
    identity = {
        # ledger scope는 개별 seed/checkpoint가 아니라 전체 campaign이다. 첫 seed를
        # 잘못 개봉한 뒤 다른 seed winner로 두 번째 ledger를 만드는 우회를 막는다.
        "seed_neutral_campaign_sha256": _sha256_identity(
            selection.get("seed_neutral_campaign_sha256"),
            name="seed-neutral campaign",
        ),
        "manifest_sha256": _sha256_identity(
            selection.get("manifest_sha256"), name="manifest"
        ),
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    ledger_id = hashlib.sha256(encoded).hexdigest()
    if repo_root is None:
        from ..config import REPO_ROOT

        root = Path(REPO_ROOT)
    else:
        root = Path(repo_root)
    directory = Path(os.path.abspath(root)) / LEDGER_ROOT / ledger_id
    return directory / "capability.json", directory / "consumed.json"


def canonical_test_ledger_event_paths_from_payload(
    selection: dict[str, Any], *, repo_root: str | Path | None = None
) -> dict[str, Path]:
    capability, consumed = canonical_test_ledger_paths_from_payload(
        selection, repo_root=repo_root
    )
    return {
        "issued": capability,
        "running": consumed,
        "completed": capability.parent / "completed.json",
        "failed": capability.parent / "failed.json",
    }


def canonical_test_ledger_paths(
    selection_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """seed-neutral campaign+manifest의 저장소 내 유일 test ledger 경로."""

    selection, _ = read_json_snapshot(selection_path)
    return canonical_test_ledger_paths_from_payload(
        selection, repo_root=repo_root
    )


def _require_canonical_ledger_path(
    actual: str | Path, expected: Path, *, label: str
) -> Path:
    target = Path(os.path.abspath(Path(actual)))
    canonical = Path(os.path.abspath(expected))
    if target != canonical:
        raise ValueError(
            f"{label}는 campaign canonical ledger 경로만 허용합니다: "
            f"requested={target}, expected={canonical}"
        )
    return target


def publish_directory_noreplace(staging: str | Path, target: str | Path) -> Path:
    """same-filesystem staging dir을 Linux renameat2(NOREPLACE)로 원자 공개."""

    source = Path(os.path.abspath(Path(staging)))
    destination = Path(os.path.abspath(Path(target)))
    if source.parent != destination.parent:
        raise ValueError("atomic directory publication은 sibling staging만 허용합니다")
    source_stat = source.lstat()
    if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISDIR(source_stat.st_mode):
        raise ValueError("atomic directory staging은 실제 directory여야 합니다")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"평가 산출 디렉터리를 덮어쓸 수 없습니다: {destination}")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("canonical no-replace directory publish에 renameat2가 필요합니다")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(
                f"평가 산출 디렉터리를 덮어쓸 수 없습니다: {destination}"
            )
        raise OSError(error, os.strerror(error), str(destination))
    directory_fd = os.open(
        destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return destination


def issue_test_capability(
    *,
    selection_path: str | Path,
    capability_path: str | Path,
    repo_root: str | Path | None = None,
) -> str:
    selection, selection_snapshot = read_json_snapshot(selection_path)
    selected = selection.get("selected")
    if selection.get("selection_split") != "val" or not isinstance(selected, dict):
        raise ValueError("recorded-val selection이 고정되지 않았습니다")
    validate_test_open_selection(selection)
    expected_capability, expected_consumed = canonical_test_ledger_paths_from_payload(
        selection, repo_root=repo_root
    )
    event_paths = canonical_test_ledger_event_paths_from_payload(
        selection, repo_root=repo_root
    )
    capability_path = _require_canonical_ledger_path(
        capability_path, expected_capability, label="test capability"
    )
    terminal = [
        path for path in event_paths.values() if path.exists() or path.is_symlink()
    ]
    if terminal:
        raise FileExistsError(
            "이 campaign/manifest test ledger는 이미 발급/소비됐습니다: "
            f"{terminal}"
        )
    token = secrets.token_urlsafe(32)
    payload = {
        "schema_version": 1,
        "phase": "issued",
        "seed_neutral_campaign_sha256": selection.get(
            "seed_neutral_campaign_sha256"
        ),
        "selection_sha256": selection_snapshot.sha256,
        "experiment_contract_sha256": selection.get(
            "experiment_contract_sha256"
        ),
        "selected_checkpoint_sha256": selected.get("checkpoint_sha256"),
        "manifest_sha256": selection.get("manifest_sha256"),
        "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        "issued_at_unix_ns": time.time_ns(),
    }
    write_json_exclusive(capability_path, payload)
    return token


def consume_test_capability(
    *,
    selection_path: str | Path,
    capability_path: str | Path,
    consumed_marker_path: str | Path,
    token: str,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    repo_root: str | Path | None = None,
) -> tuple[FileSnapshot, FileSnapshot, dict[str, Any]]:
    """test bytes를 읽기 직전에 capability를 영구 소비하고 snapshot을 반환한다."""

    if not token:
        raise ValueError(f"test capability token이 없습니다 ({CAPABILITY_ENV})")
    selection, selection_snapshot = read_json_snapshot(selection_path)
    validate_test_open_selection(selection)
    expected_capability, expected_consumed = canonical_test_ledger_paths_from_payload(
        selection, repo_root=repo_root
    )
    event_paths = canonical_test_ledger_event_paths_from_payload(
        selection, repo_root=repo_root
    )
    capability_path = _require_canonical_ledger_path(
        capability_path, expected_capability, label="test capability"
    )
    consumed_marker_path = _require_canonical_ledger_path(
        consumed_marker_path, expected_consumed, label="test consumed marker"
    )
    for phase in ("running", "completed", "failed"):
        path = event_paths[phase]
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"test ledger phase가 이미 존재합니다: {phase}={path}")
    capability, capability_snapshot = read_json_snapshot(capability_path)
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("selection.selected가 없습니다")
    expected = {
        "selection_sha256": selection_snapshot.sha256,
        "seed_neutral_campaign_sha256": selection.get(
            "seed_neutral_campaign_sha256"
        ),
        "experiment_contract_sha256": selection.get("experiment_contract_sha256"),
        "selected_checkpoint_sha256": selected.get("checkpoint_sha256"),
        "manifest_sha256": selection.get("manifest_sha256"),
        "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
    }
    for key, value in expected.items():
        if capability.get(key) != value:
            raise ValueError(f"test capability {key}가 selection과 다릅니다")
    if capability.get("phase") != "issued":
        raise ValueError("test capability phase가 issued가 아닙니다")
    checkpoint = snapshot_regular_file(checkpoint_path)
    manifest = snapshot_regular_file(manifest_path)
    if checkpoint.sha256 != selected.get("checkpoint_sha256"):
        raise ValueError("selection 뒤 checkpoint bytes가 바뀌었습니다")
    if manifest.sha256 != selection.get("manifest_sha256"):
        raise ValueError("selection 뒤 manifest bytes가 바뀌었습니다")
    marker = {
        "schema_version": 1,
        "phase": "running",
        "seed_neutral_campaign_sha256": selection.get(
            "seed_neutral_campaign_sha256"
        ),
        "selection_sha256": selection_snapshot.sha256,
        "capability_sha256": capability_snapshot.sha256,
        "experiment_contract_sha256": selection.get("experiment_contract_sha256"),
        "selected_checkpoint_sha256": checkpoint.sha256,
        "manifest_sha256": manifest.sha256,
        "consumed_at_unix_ns": time.time_ns(),
    }
    write_json_exclusive(consumed_marker_path, marker)
    return checkpoint, manifest, marker


def _active_test_ledger(
    *,
    selection_path: str | Path,
    capability_path: str | Path,
    consumed_marker_path: str | Path,
    repo_root: str | Path | None,
) -> tuple[dict[str, Any], FileSnapshot, FileSnapshot, dict[str, Path]]:
    selection, selection_snapshot = read_json_snapshot(selection_path)
    paths = canonical_test_ledger_event_paths_from_payload(
        selection, repo_root=repo_root
    )
    _require_canonical_ledger_path(capability_path, paths["issued"], label="test capability")
    _require_canonical_ledger_path(
        consumed_marker_path, paths["running"], label="test consumed marker"
    )
    capability, capability_snapshot = read_json_snapshot(paths["issued"])
    running, running_snapshot = read_json_snapshot(paths["running"])
    if capability.get("phase") != "issued" or running.get("phase") != "running":
        raise ValueError("test ledger issued/running phase가 손상됐습니다")
    if running.get("selection_sha256") != selection_snapshot.sha256:
        raise ValueError("test running marker selection SHA가 다릅니다")
    if running.get("capability_sha256") != capability_snapshot.sha256:
        raise ValueError("test running marker capability SHA가 다릅니다")
    return running, running_snapshot, selection_snapshot, paths


def complete_test_evaluation(
    *,
    selection_path: str | Path,
    capability_path: str | Path,
    consumed_marker_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path | None = None,
) -> Path:
    running, running_snapshot, selection_snapshot, paths = _active_test_ledger(
        selection_path=selection_path,
        capability_path=capability_path,
        consumed_marker_path=consumed_marker_path,
        repo_root=repo_root,
    )
    if paths["failed"].exists() or paths["failed"].is_symlink():
        raise FileExistsError("failed test ledger는 completed로 승격할 수 없습니다")
    directory = Path(os.path.abspath(Path(output_dir)))
    directory_stat = directory.lstat()
    if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
        raise ValueError("canonical test output은 실제 directory여야 합니다")
    markdown = snapshot_regular_file(directory / "metrics.md")
    metrics = snapshot_regular_file(directory / "metrics.npz")
    payload = {
        "schema_version": 1,
        "phase": "completed",
        "selection_sha256": selection_snapshot.sha256,
        "running_marker_sha256": running_snapshot.sha256,
        "experiment_contract_sha256": running.get("experiment_contract_sha256"),
        "seed_neutral_campaign_sha256": running.get(
            "seed_neutral_campaign_sha256"
        ),
        "selected_checkpoint_sha256": running.get("selected_checkpoint_sha256"),
        "manifest_sha256": running.get("manifest_sha256"),
        "output_dir": str(directory),
        "metrics_markdown_sha256": markdown.sha256,
        "metrics_npz_sha256": metrics.sha256,
        "completed_at_unix_ns": time.time_ns(),
    }
    write_json_exclusive(paths["completed"], payload)
    return paths["completed"]


def fail_test_evaluation(
    *,
    selection_path: str | Path,
    capability_path: str | Path,
    consumed_marker_path: str | Path,
    error_type: str,
    repo_root: str | Path | None = None,
) -> Path:
    running, running_snapshot, selection_snapshot, paths = _active_test_ledger(
        selection_path=selection_path,
        capability_path=capability_path,
        consumed_marker_path=consumed_marker_path,
        repo_root=repo_root,
    )
    if paths["completed"].exists() or paths["completed"].is_symlink():
        raise FileExistsError("completed test ledger를 failed로 바꿀 수 없습니다")
    payload = {
        "schema_version": 1,
        "phase": "failed",
        "selection_sha256": selection_snapshot.sha256,
        "running_marker_sha256": running_snapshot.sha256,
        "experiment_contract_sha256": running.get("experiment_contract_sha256"),
        "seed_neutral_campaign_sha256": running.get(
            "seed_neutral_campaign_sha256"
        ),
        "selected_checkpoint_sha256": running.get("selected_checkpoint_sha256"),
        "manifest_sha256": running.get("manifest_sha256"),
        "error_type": str(error_type),
        "failed_at_unix_ns": time.time_ns(),
    }
    write_json_exclusive(paths["failed"], payload)
    return paths["failed"]


__all__ = [
    "CAPABILITY_ENV",
    "FileSnapshot",
    "consume_test_capability",
    "canonical_test_ledger_paths",
    "canonical_test_ledger_paths_from_payload",
    "canonical_test_ledger_event_paths_from_payload",
    "classify_recorded_val_metrics",
    "complete_test_evaluation",
    "issue_test_capability",
    "fail_test_evaluation",
    "publish_directory_noreplace",
    "read_json_snapshot",
    "snapshot_regular_file",
    "validate_test_open_selection",
    "write_json_exclusive",
]
