"""v6 live raw의 외부 post-capture receipt와 offline admission.

이 모듈은 오디오 장치를 열지 않는다. 캡처가 닫힌 뒤 저장된 immutable raw와
plan/authority/meter/evidence/hardware의 *현재 bytes*를 다시 읽고, raw 내부의
self-attestation과 독립적으로 대조한 receipt만 no-replace로 발행한다.
"""

from __future__ import annotations

import datetime as dt
from contextlib import ExitStack
import ctypes
import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
from typing import Any, Mapping

import numpy as np

from deep_anc.data.repository_fd import (
    RepositoryFileGuard,
    repository_execution_identity,
)

from .fullband_live_authority_v6 import (
    AUTHORITY_SCHEMA,
    PLAN_ENVELOPE_SCHEMA,
    SEALED_HARDWARE_RELATIVE_PATH,
    SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
    SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
    SEALED_RAW_RELATIVE_PATH,
    committed_plan_envelope_v6,
)
from .fullband_live_raw_v6 import (
    TELEMETRY_ARRAY_FIELDS,
    load_live_raw_v6,
)
from .fullband_live_raw_v5 import BINDING_KEYS, HARDWARE_BINDING_SCHEMA
from .fullband_live_delay_core_v6 import (
    OPERATOR_ARRAY_NAMES,
    analyze_committed_v6_live_delay,
    validate_analysis_operator_v6,
)


EXTERNAL_POST_RECEIPT_SCHEMA = (
    "fullband_causal_v6_external_post_capture_receipt_v1"
)
EXTERNAL_POST_RECEIPT_SUFFIX = ".post_receipt.json"
EXTERNAL_POST_RECEIPT_SCOPE = (
    "immutable_raw_plus_fresh_external_bytes_not_hardware_slip_authority"
)
ANALYSIS_ENVELOPE_SCHEMA = "fullband_causal_v6_live_delay_analysis_envelope_v1"
OPERATOR_CONTAINER_SCHEMA = "fullband_causal_v6_live_delay_operator_container_v1"
FAILURE_SCHEMA = "fullband_causal_v6_live_delay_failure_v1"
ANALYSIS_ADAPTER_RELATIVE_PATH = (
    "scripts/data/measure_paths_fullband_causal_v6.py"
)

_RECEIPT_KEYS = {
    "schema",
    "status",
    "valid",
    "invalid_reasons",
    "scope",
    "raw",
    "external_bindings",
    "operator_confirmations",
    "primitive_post_capture_binding",
    "audio_lock",
    "resolved_devices",
    "analysis_admission_eligible",
    "canonical_training_eligible",
    "hardware_sample_slip_authority",
    "receipt_payload_sha256",
}
_RAW_RECEIPT_KEYS = {
    "path",
    "file_sha256",
    "metadata_payload_sha256",
    "bindings_payload_sha256",
    "array_sha256",
    "capture_id",
    "status",
    "schema",
}
_AUDIO_LOCK_RECEIPT_KEYS = {
    "path",
    "identity_sha256",
    "pid",
    "uid",
    "purpose",
    "device",
    "inode",
    "exclusive_lock_observed",
}
_AUDIO_LOCK_IDENTITY_KEYS = {
    "path",
    "pid",
    "uid",
    "purpose",
    "device",
    "inode",
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_file_bytes(value: Any) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label}는 64자리 lowercase SHA-256이어야 합니다")
    return value


def _exact_mapping(value: Any, keys: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} key 집합이 exact하지 않습니다")
    return dict(value)


def _canonical_relative_path(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError(f"{label}는 canonical repository 상대경로여야 합니다")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{label}는 canonical repository 상대경로여야 합니다")
    return value


def _validate_offline_meter_repository_execution_v6(
    metadata: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    """meter 생성 실행과 현재 clean exact script checkout을 결속한다."""

    saved_execution = metadata.get("repository_execution")
    current_execution = repository_execution_identity(
        repository_root, "scripts/data/set_amp_level.py"
    )
    if saved_execution != current_execution:
        raise ValueError(
            "offline v6 meter repository commit/branch/script path/SHA가 "
            "current clean checkout과 다릅니다"
        )
    return dict(current_execution)


def _repository_root(value: str | os.PathLike[str]) -> Path:
    root = Path(os.path.abspath(os.fspath(value)))
    status = os.lstat(root)
    if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode):
        raise ValueError("repository root는 symlink가 아닌 directory여야 합니다")
    return root


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_parent_chain(
    repository_root: Path,
    relative_path: str,
    *,
    create: bool,
) -> tuple[str, list[tuple[Path, int, int, int]]]:
    relative = _canonical_relative_path(relative_path, label="repository file path")
    parts = PurePosixPath(relative).parts
    root_fd = os.open(repository_root, _directory_flags())
    opened: list[tuple[Path, int, int, int]] = []
    try:
        root_status = os.fstat(root_fd)
        opened.append(
            (repository_root, root_fd, root_status.st_dev, root_status.st_ino)
        )
        current_fd = root_fd
        cursor = repository_root
        for component in parts[:-1]:
            cursor = cursor / component
            try:
                child_fd = os.open(component, _directory_flags(), dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, 0o755, dir_fd=current_fd)
                    os.fsync(current_fd)
                except FileExistsError:
                    pass
                child_fd = os.open(component, _directory_flags(), dir_fd=current_fd)
            child_status = os.fstat(child_fd)
            if not stat.S_ISDIR(child_status.st_mode):
                os.close(child_fd)
                raise NotADirectoryError(f"repository parent가 directory가 아닙니다: {cursor}")
            opened.append(
                (cursor, child_fd, child_status.st_dev, child_status.st_ino)
            )
            current_fd = child_fd
        _verify_chain(opened)
        return parts[-1], opened
    except BaseException:
        for _path, descriptor, _dev, _ino in reversed(opened):
            os.close(descriptor)
        if not opened:
            os.close(root_fd)
        raise


def _verify_chain(chain: list[tuple[Path, int, int, int]]) -> None:
    for path, descriptor, expected_dev, expected_ino in chain:
        fd_status = os.fstat(descriptor)
        lexical_status = os.lstat(path)
        if (
            not stat.S_ISDIR(fd_status.st_mode)
            or not stat.S_ISDIR(lexical_status.st_mode)
            or stat.S_ISLNK(lexical_status.st_mode)
            or (fd_status.st_dev, fd_status.st_ino) != (expected_dev, expected_ino)
            or (lexical_status.st_dev, lexical_status.st_ino)
            != (expected_dev, expected_ino)
        ):
            raise RuntimeError(f"repository directory inode가 변경됐습니다: {path}")


def _close_chain(chain: list[tuple[Path, int, int, int]]) -> None:
    for _path, descriptor, _dev, _ino in reversed(chain):
        os.close(descriptor)


def read_repository_file_nofollow(
    repository_root: str | os.PathLike[str], relative_path: str
) -> dict[str, Any]:
    """dirfd/O_NOFOLLOW로 regular file 한 inode의 bytes를 읽는다."""

    root = _repository_root(repository_root)
    relative = _canonical_relative_path(relative_path, label="repository file path")
    filename, chain = _open_parent_chain(root, relative, create=False)
    parent_fd = chain[-1][1]
    descriptor = -1
    try:
        descriptor = os.open(
            filename,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        named = os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ValueError(f"repository file은 symlink 아닌 regular file이어야 합니다: {relative}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named_after = os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
        if (
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            or (named_after.st_dev, named_after.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError(f"repository file이 read 중 변경됐습니다: {relative}")
        _verify_chain(chain)
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise RuntimeError(f"repository file size가 read 결과와 다릅니다: {relative}")
        return {
            "path": relative,
            "bytes": payload,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "device": int(before.st_dev),
            "inode": int(before.st_ino),
        }
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        _close_chain(chain)


def assert_repository_target_fresh_nofollow(
    repository_root: str | os.PathLike[str], relative_path: str
) -> None:
    root = _repository_root(repository_root)
    relative = _canonical_relative_path(relative_path, label="fresh target")
    # target 자체는 만들지 않는다. publication이 쓸 전용 parent만 dirfd 기준으로
    # 미리 만들고 fsync해 이후 pre-open freshness와 같은 inode chain을 검사한다.
    filename, chain = _open_parent_chain(root, relative, create=True)
    try:
        try:
            os.stat(filename, dir_fd=chain[-1][1], follow_symlinks=False)
        except FileNotFoundError:
            _verify_chain(chain)
            return
        raise FileExistsError(f"기존 target/symlink를 덮어쓰지 않습니다: {relative}")
    finally:
        _close_chain(chain)


def _publish_json_noreplace(
    repository_root: str | os.PathLike[str], relative_path: str, value: Mapping[str, Any]
) -> dict[str, Any]:
    root = _repository_root(repository_root)
    relative = _canonical_relative_path(relative_path, label="JSON target")
    payload = _canonical_json_file_bytes(value)
    filename, chain = _open_parent_chain(root, relative, create=True)
    parent_fd = chain[-1][1]
    staging = ""
    descriptor = -1
    linked = False
    try:
        try:
            os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"기존 JSON target을 덮어쓰지 않습니다: {relative}")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        for _ in range(128):
            staging = f".{filename}.staging-{secrets.token_hex(16)}"
            try:
                descriptor = os.open(staging, flags, 0o600, dir_fd=parent_fd)
                break
            except FileExistsError:
                continue
        if descriptor < 0:
            raise RuntimeError("external receipt staging 이름을 확보하지 못했습니다")
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _verify_chain(chain)
        os.link(
            staging,
            filename,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        linked = True
        try:
            os.unlink(staging, dir_fd=parent_fd)
            staging = ""
        finally:
            os.fsync(parent_fd)
        _verify_chain(chain)
    except BaseException:
        if staging and not linked:
            try:
                os.unlink(staging, dir_fd=parent_fd)
            except OSError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        _close_chain(chain)
    return {
        "path": root.joinpath(*PurePosixPath(relative).parts),
        "relative_path": relative,
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }


def external_post_receipt_relative_path(raw_relative_path: str) -> str:
    raw = _canonical_relative_path(raw_relative_path, label="v6 live raw path")
    path = PurePosixPath(raw)
    return path.with_name(path.name + EXTERNAL_POST_RECEIPT_SUFFIX).as_posix()


def audio_lock_identity_sha256(audio_lock: Mapping[str, Any]) -> str:
    lock = _exact_mapping(
        audio_lock, _AUDIO_LOCK_IDENTITY_KEYS, label="audio lock"
    )
    _canonical_relative_path(lock["path"], label="audio lock path")
    if type(lock["pid"]) is not int or lock["pid"] <= 0:
        raise ValueError("audio lock pid는 양의 exact int여야 합니다")
    if type(lock["uid"]) is not int or lock["uid"] < 0:
        raise ValueError("audio lock uid는 음이 아닌 exact int여야 합니다")
    if type(lock["purpose"]) is not str or not lock["purpose"]:
        raise ValueError("audio lock purpose가 필요합니다")
    if type(lock["device"]) is not int or lock["device"] < 0:
        raise ValueError("audio lock device는 음이 아닌 exact int여야 합니다")
    if type(lock["inode"]) is not int or lock["inode"] <= 0:
        raise ValueError("audio lock inode는 양의 exact int여야 합니다")
    return _payload_sha256(lock)


def validate_held_audio_lock(
    repository_root: str | os.PathLike[str],
    audio_lock: Mapping[str, Any],
    *,
    expected_purpose: str,
) -> dict[str, Any]:
    """현재 process의 exact lock bytes와 배타 잠금 보유를 독립 확인한다."""

    lock = _exact_mapping(
        audio_lock, _AUDIO_LOCK_IDENTITY_KEYS, label="audio lock"
    )
    if (
        lock["pid"] != os.getpid()
        or lock["uid"] != os.getuid()
        or lock["purpose"] != expected_purpose
    ):
        raise ValueError("audio lock process/uid/purpose가 현재 capture와 다릅니다")
    snapshot = read_repository_file_nofollow(repository_root, lock["path"])
    if snapshot["bytes"] != _canonical_json_bytes(lock):
        raise ValueError("audio lock file bytes가 context identity와 다릅니다")
    expected_inode = (lock["device"], lock["inode"])
    if (snapshot["device"], snapshot["inode"]) != expected_inode:
        raise RuntimeError("audio lock snapshot inode가 held context와 다릅니다")
    root = _repository_root(repository_root)
    relative = _canonical_relative_path(lock["path"], label="audio lock path")
    filename, chain = _open_parent_chain(root, relative, create=False)
    descriptor = -1
    exclusive_observed = False
    try:
        descriptor = os.open(
            filename,
            os.O_RDWR
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=chain[-1][1],
        )
        opened = os.fstat(descriptor)
        named = os.stat(filename, dir_fd=chain[-1][1], follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or (opened.st_dev, opened.st_ino) != expected_inode
            or (named.st_dev, named.st_ino) != expected_inode
        ):
            raise RuntimeError("audio lock probe inode가 held context와 다릅니다")
        _verify_chain(chain)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            exclusive_observed = True
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        opened_after = os.fstat(descriptor)
        named_after = os.stat(filename, dir_fd=chain[-1][1], follow_symlinks=False)
        if (
            (opened_after.st_dev, opened_after.st_ino) != expected_inode
            or (named_after.st_dev, named_after.st_ino) != expected_inode
        ):
            raise RuntimeError("audio lock inode가 flock 검증 중 변경됐습니다")
        _verify_chain(chain)
        if not exclusive_observed:
            raise RuntimeError("repository audio lock이 배타 보유 상태가 아닙니다")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        _close_chain(chain)
    return {
        "path": relative,
        "identity_sha256": audio_lock_identity_sha256(lock),
        "pid": lock["pid"],
        "uid": lock["uid"],
        "purpose": lock["purpose"],
        "device": lock["device"],
        "inode": lock["inode"],
        "exclusive_lock_observed": True,
    }


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON duplicate key를 거부합니다: {key}")
        result[key] = value
    return result


def _load_canonical_json_file(snapshot: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    raw = snapshot["bytes"]
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label}가 UTF-8 JSON이 아닙니다") from error
    if not isinstance(value, dict) or raw != _canonical_json_file_bytes(value):
        raise ValueError(f"{label}가 canonical JSON file bytes가 아닙니다")
    return value


def _normalize_iso_utc(value: Any, *, label: str) -> str:
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{label}가 UTC timestamp가 아닙니다") from error
    else:
        raise ValueError(f"{label}가 UTC timestamp가 아닙니다")
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError(f"{label}는 timezone-aware UTC여야 합니다")
    return parsed.isoformat()


def _normalize_fullband_meter_validation(
    value: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    """meter validator 공개 반환을 raw binding exact shape로 정규화한다."""

    required = {
        "path",
        "receipt_path",
        "raw_sha256",
        "receipt_sha256",
        "identity_sha256",
        "completed_at_utc",
        "followup_contract_sha256",
        "plan",
        "live_capture_authority",
        "level_evidence",
        "hardware",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"v6 meter validator 반환 필드가 부족합니다: {sorted(missing)}")

    def relative(path_value: Any, label: str) -> str:
        path = Path(path_value)
        if path.is_absolute():
            try:
                return path.relative_to(repository_root).as_posix()
            except ValueError as error:
                raise ValueError(f"{label}가 repository 밖입니다") from error
        return _canonical_relative_path(str(path), label=label)

    plan = value["plan"]
    authority = value["live_capture_authority"]
    evidence = value["level_evidence"]
    hardware = value["hardware"]
    for name, item in (
        ("plan", plan),
        ("authority", authority),
        ("level evidence", evidence),
        ("hardware", hardware),
    ):
        if not isinstance(item, Mapping):
            raise ValueError(f"v6 meter {name} 반환은 mapping이어야 합니다")
    devices = hardware.get("resolved_devices")
    if (
        not isinstance(devices, Mapping)
        or set(devices) != {"input", "output"}
        or any(type(devices[name]) is not int or devices[name] < 0 for name in devices)
    ):
        raise ValueError("v6 meter resolved device가 exact하지 않습니다")
    if (
        evidence.get("scope")
        != "tracked_historical_attestation_for_fresh_v5_meter_only"
        or evidence.get("preserved_raw_revalidated") is not False
    ):
        raise ValueError(
            "v6 level evidence는 tracked historical attestation scope와 "
            "preserved_raw_revalidated=false여야 합니다"
        )
    return {
        "signal_plan": {
            "schema": PLAN_ENVELOPE_SCHEMA,
            "path": relative(plan["path"], "meter plan path"),
            "file_sha256": _require_sha256(plan["file_sha256"], label="meter plan file SHA"),
            "payload_sha256": _require_sha256(plan["payload_sha256"], label="meter plan payload SHA"),
            "pcm_sha256": _require_sha256(plan["pcm_sha256"], label="meter plan PCM SHA"),
            "raw_session_relative_path": _canonical_relative_path(
                plan.get("raw_session_relative_path", SEALED_RAW_RELATIVE_PATH),
                label="meter sealed raw path",
            ),
        },
        "live_capture_authority": {
            "schema": AUTHORITY_SCHEMA,
            "path": relative(authority["path"], "meter authority path"),
            "file_sha256": _require_sha256(authority["file_sha256"], label="meter authority file SHA"),
            "payload_sha256": _require_sha256(authority["payload_sha256"], label="meter authority payload SHA"),
            "signal_plan_file_sha256": _require_sha256(plan["file_sha256"], label="meter authority plan file SHA"),
            "signal_plan_payload_sha256": _require_sha256(plan["payload_sha256"], label="meter authority plan payload SHA"),
            "signal_pcm_sha256": _require_sha256(plan["pcm_sha256"], label="meter authority PCM SHA"),
            "hardware_file_sha256": _require_sha256(hardware["file_sha256"], label="meter authority hardware SHA"),
            "raw_session_relative_path": SEALED_RAW_RELATIVE_PATH,
        },
        "meter": {
            "schema": "measurement_level_meter_raw_v1",
            "path": relative(value["path"], "meter raw path"),
            "receipt_path": relative(value["receipt_path"], "meter receipt path"),
            "raw_sha256": _require_sha256(value["raw_sha256"], label="meter raw SHA"),
            "receipt_sha256": _require_sha256(value["receipt_sha256"], label="meter receipt SHA"),
            "completed_at_utc": _normalize_iso_utc(value["completed_at_utc"], label="meter completion"),
            "identity_sha256": _require_sha256(value["identity_sha256"], label="meter identity SHA"),
            "followup_contract_sha256": _require_sha256(value["followup_contract_sha256"], label="meter followup SHA"),
            "live_authority_file_sha256": _require_sha256(authority["file_sha256"], label="meter authority file SHA"),
            "level_evidence_file_sha256": _require_sha256(evidence["file_sha256"], label="meter evidence file SHA"),
            "hardware_file_sha256": _require_sha256(hardware["file_sha256"], label="meter hardware file SHA"),
        },
        "level_evidence": {
            "schema": "measurement_level_evidence_v2_bootstrap_pair",
            "path": relative(evidence["path"], "level evidence path"),
            "file_sha256": _require_sha256(evidence["file_sha256"], label="evidence file SHA"),
            "identity_sha256": _require_sha256(evidence["identity_sha256"], label="evidence identity SHA"),
            "scope": str(evidence["scope"]),
            "preserved_raw_revalidated": evidence["preserved_raw_revalidated"],
        },
        "hardware": {
            "schema": HARDWARE_BINDING_SCHEMA,
            "path": relative(hardware["path"], "hardware path"),
            "file_sha256": _require_sha256(hardware["file_sha256"], label="hardware file SHA"),
            "identity_sha256": _require_sha256(hardware["identity_sha256"], label="hardware identity SHA"),
            "physical_fingerprint_sha256": _require_sha256(
                hardware["physical_fingerprint_sha256"], label="hardware fingerprint SHA"
            ),
            "resolved_devices": {"input": devices["input"], "output": devices["output"]},
        },
    }


def collect_actual_external_bindings_v6(
    *,
    repository_root: str | os.PathLike[str],
    plan_envelope_path: str,
    live_authority_path: str,
    meter_raw_path: str,
    level_evidence_path: str,
    hardware_path: str,
    preflight_binding: Mapping[str, Any],
    require_meter_fresh: bool,
    require_sealed_raw_fresh: bool,
) -> dict[str, Any]:
    """actual files를 loader로 다시 읽어 live raw binding을 독립 재구성한다."""

    root = _repository_root(repository_root)
    supplied = {
        "plan": _canonical_relative_path(plan_envelope_path, label="plan envelope path"),
        "authority": _canonical_relative_path(live_authority_path, label="live authority path"),
        "meter": _canonical_relative_path(meter_raw_path, label="meter raw path"),
        "evidence": _canonical_relative_path(level_evidence_path, label="level evidence path"),
        "hardware": _canonical_relative_path(hardware_path, label="hardware path"),
    }
    exact_paths = {
        "plan": SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
        "authority": SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
        "evidence": "assets/measured/measurement_level_evidence.json",
        "hardware": SEALED_HARDWARE_RELATIVE_PATH,
    }
    for name, expected in exact_paths.items():
        if supplied[name] != expected:
            raise ValueError(f"{name} path가 pinned repository path와 다릅니다")

    from . import fullband_v6_meter

    validator = getattr(fullband_v6_meter, "validate_fullband_v6_meter_raw", None)
    if validator is None:
        raise RuntimeError("validate_fullband_v6_meter_raw public API가 없습니다")
    meter_path = PurePosixPath(supplied["meter"])
    anticipated_receipt = meter_path.with_name(
        f"{meter_path.stem}.receipt.json"
    ).as_posix()
    guarded_paths = {
        **supplied,
        "meter_receipt": anticipated_receipt,
    }
    with ExitStack() as stack:
        guards = {
            name: stack.enter_context(
                RepositoryFileGuard(root, relative, label=f"live external {name}")
            )
            for name, relative in guarded_paths.items()
        }
        meter = validator(
            root / supplied["meter"],
            repository_root=root,
            now_utc=None,
            require_fresh=require_meter_fresh,
            require_sealed_raw_fresh=require_sealed_raw_fresh,
        )
        normalized = _normalize_fullband_meter_validation(
            meter, repository_root=root
        )
        if normalized["meter"]["receipt_path"] != anticipated_receipt:
            raise ValueError("meter receipt path가 anticipated canonical sibling과 다릅니다")
        for guard in guards.values():
            guard.verify()
        source_snapshots = {
            name: guard.snapshot() for name, guard in guards.items()
        }
    snapshot_checks = (
        (source_snapshots["plan"]["sha256"], normalized["signal_plan"]["file_sha256"], "plan"),
        (source_snapshots["authority"]["sha256"], normalized["live_capture_authority"]["file_sha256"], "authority"),
        (source_snapshots["meter"]["sha256"], normalized["meter"]["raw_sha256"], "meter raw"),
        (source_snapshots["meter_receipt"]["sha256"], normalized["meter"]["receipt_sha256"], "meter receipt"),
        (source_snapshots["evidence"]["sha256"], normalized["level_evidence"]["file_sha256"], "level evidence"),
        (source_snapshots["hardware"]["sha256"], normalized["hardware"]["file_sha256"], "hardware"),
    )
    for observed, expected, label in snapshot_checks:
        if observed != expected:
            raise ValueError(f"{label}가 validator와 dirfd snapshot 사이에 변경됐습니다")
    if normalized["signal_plan"]["path"] != supplied["plan"]:
        raise ValueError("meter가 결속한 plan path가 CLI plan과 다릅니다")
    if normalized["live_capture_authority"]["path"] != supplied["authority"]:
        raise ValueError("meter가 결속한 authority path가 CLI authority와 다릅니다")
    if normalized["meter"]["path"] != supplied["meter"]:
        raise ValueError("meter validator raw path가 CLI meter와 다릅니다")
    if normalized["level_evidence"]["path"] != supplied["evidence"]:
        raise ValueError("meter가 결속한 evidence path가 CLI evidence와 다릅니다")
    if normalized["hardware"]["path"] != supplied["hardware"]:
        raise ValueError("meter가 결속한 hardware path가 CLI hardware와 다릅니다")

    preflight = _exact_mapping(
        preflight_binding, BINDING_KEYS["preflight"], label="preflight binding"
    )
    return {**normalized, "preflight": preflight}


def _collect_offline_external_bindings_without_backend(
    *,
    repository_root: Path,
    expected_bindings: Mapping[str, Any],
    plan_envelope_path: str,
    live_authority_path: str,
    meter_raw_path: str,
    level_evidence_path: str,
    hardware_path: str,
    require_current_meter_execution: bool = True,
) -> dict[str, Any]:
    """offline에서는 PortAudio query 없이 receipt가 봉인한 devices를 재검증한다."""

    if type(require_current_meter_execution) is not bool:
        raise TypeError("require_current_meter_execution은 exact bool이어야 합니다")

    from . import fullband_v6_meter
    from .measurement_level import (
        validate_bootstrap_meter_raw,
    )

    bindings = _exact_mapping(
        expected_bindings, set(BINDING_KEYS), label="offline expected bindings"
    )
    supplied = {
        "plan": _canonical_relative_path(plan_envelope_path, label="plan path"),
        "authority": _canonical_relative_path(live_authority_path, label="authority path"),
        "meter": _canonical_relative_path(meter_raw_path, label="meter path"),
        "evidence": _canonical_relative_path(level_evidence_path, label="evidence path"),
        "hardware": _canonical_relative_path(hardware_path, label="hardware path"),
    }
    if supplied != {
        "plan": bindings["signal_plan"]["path"],
        "authority": bindings["live_capture_authority"]["path"],
        "meter": bindings["meter"]["path"],
        "evidence": bindings["level_evidence"]["path"],
        "hardware": bindings["hardware"]["path"],
    }:
        raise ValueError("offline CLI source path가 external receipt binding과 다릅니다")

    meter_path = PurePosixPath(supplied["meter"])
    anticipated_receipt = meter_path.with_name(
        f"{meter_path.stem}.receipt.json"
    ).as_posix()
    if bindings["meter"]["receipt_path"] != anticipated_receipt:
        raise ValueError("offline meter receipt가 canonical raw sibling이 아닙니다")
    guarded_paths = {
        **supplied,
        "meter_receipt": anticipated_receipt,
    }
    with ExitStack() as stack:
        guards = {
            name: stack.enter_context(
                RepositoryFileGuard(
                    repository_root, relative, label=f"offline external {name}"
                )
            )
            for name, relative in guarded_paths.items()
        }
        contract = fullband_v6_meter.validate_fullband_v6_static_contract(
            repository_root=repository_root,
            plan_envelope_path=supplied["plan"],
            live_authority_path=supplied["authority"],
            hardware_path=supplied["hardware"],
            level_evidence_path=supplied["evidence"],
            raw_target_path=SEALED_RAW_RELATIVE_PATH,
            require_sealed_raw_fresh=False,
        )
        contract_checks = (
            (contract["plan"]["path"], bindings["signal_plan"]["path"], "plan path"),
            (contract["plan"]["file_sha256"], bindings["signal_plan"]["file_sha256"], "plan file SHA"),
            (contract["plan"]["payload_sha256"], bindings["signal_plan"]["payload_sha256"], "plan payload SHA"),
            (contract["plan"]["pcm_sha256"], bindings["signal_plan"]["pcm_sha256"], "plan PCM SHA"),
            (contract["live_capture_authority"]["path"], bindings["live_capture_authority"]["path"], "authority path"),
            (contract["live_capture_authority"]["file_sha256"], bindings["live_capture_authority"]["file_sha256"], "authority file SHA"),
            (contract["live_capture_authority"]["payload_sha256"], bindings["live_capture_authority"]["payload_sha256"], "authority payload SHA"),
            (contract["hardware"]["file_sha256"], bindings["hardware"]["file_sha256"], "hardware file SHA"),
            (contract["hardware"]["identity_sha256"], bindings["hardware"]["identity_sha256"], "hardware identity SHA"),
            (contract["hardware"]["physical_fingerprint_sha256"], bindings["hardware"]["physical_fingerprint_sha256"], "physical fingerprint SHA"),
            (contract["level_evidence"], {key: bindings["level_evidence"][key] for key in contract["level_evidence"]}, "tracked level attestation"),
        )
        for observed, expected, label in contract_checks:
            if observed != expected:
                raise ValueError(f"offline {label}가 post receipt와 다릅니다")

        meter = validate_bootstrap_meter_raw(
            repository_root / supplied["meter"],
            repository_root=repository_root,
            expected_hardware_identity=contract["hardware_identity"],
            require_fresh=False,
        )
        if meter["sha256"] != guards["meter"].sha256:
            raise ValueError("offline generic meter validation SHA가 held raw와 다릅니다")
        if require_current_meter_execution:
            _validate_offline_meter_repository_execution_v6(
                meter["metadata"], repository_root=repository_root
            )
        followup = fullband_v6_meter.validate_fullband_v6_followup(
            meter["metadata"].get("fullband_v6_followup"),
            expected_contract=contract,
            expected_devices=bindings["hardware"]["resolved_devices"],
        )
        post = meter["metadata"].get("fullband_v6_post_capture_revalidation")
        if post != {"passed": True, "error": None}:
            raise ValueError("offline meter post-capture revalidation이 PASS가 아닙니다")

        for guard in guards.values():
            guard.verify()
        file_checks = (
            (guards["plan"].sha256, bindings["signal_plan"]["file_sha256"], "plan"),
            (guards["authority"].sha256, bindings["live_capture_authority"]["file_sha256"], "authority"),
            (guards["meter"].sha256, bindings["meter"]["raw_sha256"], "meter raw"),
            (guards["meter_receipt"].sha256, bindings["meter"]["receipt_sha256"], "meter receipt"),
            (guards["evidence"].sha256, bindings["level_evidence"]["file_sha256"], "level evidence"),
            (guards["hardware"].sha256, bindings["hardware"]["file_sha256"], "hardware"),
        )
        for observed, expected, label in file_checks:
            if observed != expected:
                raise ValueError(f"offline {label} file SHA가 post receipt와 다릅니다")

        completed = meter["completed_at_utc"].isoformat()
        meter_identity = fullband_v6_meter.payload_sha256(
            {
                "schema": fullband_v6_meter.FULLBAND_V6_METER_IDENTITY_SCHEMA,
                "path": supplied["meter"],
                "receipt_path": anticipated_receipt,
                "raw_sha256": guards["meter"].sha256,
                "receipt_sha256": guards["meter_receipt"].sha256,
                "completed_at_utc": completed,
                "followup_contract_sha256": followup["followup_contract_sha256"],
            }
        )
        if meter_identity != bindings["meter"]["identity_sha256"]:
            raise ValueError("offline meter identity가 post receipt와 다릅니다")
    return json.loads(_canonical_json_bytes(bindings).decode("utf-8"))


def _extract_raw_metadata_snapshot(
    *, repository_root: Path, raw_relative_path: str, expected_file_sha256: str
) -> dict[str, Any]:
    snapshot = read_repository_file_nofollow(repository_root, raw_relative_path)
    if snapshot["sha256"] != _require_sha256(
        expected_file_sha256, label="expected live raw file SHA"
    ):
        raise ValueError("live raw file SHA가 publisher receipt와 다릅니다")
    try:
        with np.load(io.BytesIO(snapshot["bytes"]), allow_pickle=False) as archive:
            metadata_member = np.array(archive["metadata_json_utf8"], copy=True)
    except (KeyError, OSError, ValueError) as error:
        raise ValueError("live raw metadata snapshot을 읽을 수 없습니다") from error
    if metadata_member.dtype != np.uint8 or metadata_member.ndim != 1:
        raise ValueError("live raw metadata_json_utf8 dtype/shape이 다릅니다")
    try:
        metadata = json.loads(metadata_member.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("live raw metadata JSON이 잘못됐습니다") from error
    if not isinstance(metadata, dict):
        raise ValueError("live raw metadata가 mapping이 아닙니다")
    return {"snapshot": snapshot, "metadata": metadata}


def _validate_receipt_witness_against_raw_metadata(
    receipt: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    """Self-hashed receipt가 raw session witness를 바꾸지 않았는지 대조한다."""

    if receipt.get("operator_confirmations") != metadata.get("operator_confirmations"):
        raise ValueError("external receipt operator confirmations가 raw와 다릅니다")
    if receipt.get("primitive_post_capture_binding") != metadata.get("post_capture_binding"):
        raise ValueError("external receipt primitive post binding이 raw와 다릅니다")
    devices = _exact_mapping(
        receipt.get("resolved_devices"), {"input", "output"}, label="receipt devices"
    )
    raw_bindings = metadata.get("bindings")
    telemetry = metadata.get("duplex_telemetry_scalars")
    if not isinstance(raw_bindings, Mapping) or not isinstance(telemetry, Mapping):
        raise ValueError("raw hardware/telemetry binding이 필요합니다")
    if devices != raw_bindings.get("hardware", {}).get("resolved_devices"):
        raise ValueError("external receipt devices가 raw hardware binding과 다릅니다")
    if devices != {
        "input": telemetry.get("resolved_input_device"),
        "output": telemetry.get("resolved_output_device"),
    }:
        raise ValueError("external receipt devices가 raw telemetry와 다릅니다")
    lock_receipt = _exact_mapping(
        receipt.get("audio_lock"), _AUDIO_LOCK_RECEIPT_KEYS, label="receipt audio lock"
    )
    lock_identity = {
        key: lock_receipt[key] for key in _AUDIO_LOCK_IDENTITY_KEYS
    }
    if lock_receipt["purpose"] != "fullband_causal_v6_live_capture":
        raise ValueError("external receipt audio lock purpose가 v6가 아닙니다")
    if lock_receipt["exclusive_lock_observed"] is not True:
        raise ValueError("external receipt에 exclusive audio lock witness가 없습니다")
    computed_lock_sha = audio_lock_identity_sha256(lock_identity)
    if lock_receipt["identity_sha256"] != computed_lock_sha:
        raise ValueError("external receipt audio lock identity SHA가 내용과 다릅니다")
    session = metadata.get("session")
    if (
        not isinstance(session, Mapping)
        or session.get("audio_lock_identity_sha256") != computed_lock_sha
    ):
        raise ValueError("external receipt audio lock가 raw session과 다릅니다")


def issue_external_post_capture_receipt_v6(
    *,
    repository_root: str | os.PathLike[str],
    raw_relative_path: str,
    expected_raw_file_sha256: str,
    plan_envelope_path: str,
    live_authority_path: str,
    meter_raw_path: str,
    level_evidence_path: str,
    hardware_path: str,
    audio_lock: Mapping[str, Any],
    expected_resolved_devices: Mapping[str, Any],
) -> dict[str, Any]:
    """raw와 actual external files를 독립 대조해 external receipt를 발행한다."""

    root = _repository_root(repository_root)
    raw_relative = _canonical_relative_path(raw_relative_path, label="live raw path")
    if raw_relative != SEALED_RAW_RELATIVE_PATH:
        raise ValueError("live raw path가 authority sealed path와 다릅니다")
    header = _extract_raw_metadata_snapshot(
        repository_root=root,
        raw_relative_path=raw_relative,
        expected_file_sha256=expected_raw_file_sha256,
    )
    header_metadata = header["metadata"]
    raw_bindings = header_metadata.get("bindings")
    if not isinstance(raw_bindings, Mapping):
        raise ValueError("live raw metadata에 bindings가 필요합니다")
    external_bindings = collect_actual_external_bindings_v6(
        repository_root=root,
        plan_envelope_path=plan_envelope_path,
        live_authority_path=live_authority_path,
        meter_raw_path=meter_raw_path,
        level_evidence_path=level_evidence_path,
        hardware_path=hardware_path,
        preflight_binding=raw_bindings.get("preflight", {}),
        require_meter_fresh=True,
        require_sealed_raw_fresh=False,
    )
    loaded = load_live_raw_v6(
        root / raw_relative,
        repository_root=root,
        expected_bindings=external_bindings,
        expected_raw_file_sha256=header["snapshot"]["sha256"],
        require_analysis_admission=False,
    )
    metadata = loaded["metadata"]
    if metadata["bindings"] != external_bindings:
        raise ValueError("raw self binding이 actual external file 재검증과 다릅니다")
    lock_receipt = validate_held_audio_lock(
        root, audio_lock, expected_purpose="fullband_causal_v6_live_capture"
    )
    devices = _exact_mapping(
        expected_resolved_devices, {"input", "output"}, label="resolved devices"
    )
    if any(type(devices[name]) is not int or devices[name] < 0 for name in devices):
        raise ValueError("resolved devices는 음이 아닌 exact int여야 합니다")
    if devices != external_bindings["hardware"]["resolved_devices"]:
        raise ValueError("post receipt device가 fresh meter/hardware binding과 다릅니다")
    telemetry = metadata["duplex_telemetry_scalars"]
    if devices != {
        "input": telemetry["resolved_input_device"],
        "output": telemetry["resolved_output_device"],
    }:
        raise ValueError("post receipt device가 actual capture telemetry와 다릅니다")
    if lock_receipt["identity_sha256"] != metadata["session"][
        "audio_lock_identity_sha256"
    ]:
        raise ValueError("external post audio lock가 raw session lock와 다릅니다")
    invalid = list(metadata["invalid_reasons"])
    if metadata["status"] != "CAPTURE_PASS" or metadata["valid"] is not True:
        invalid.append("raw_capture_not_pass")
    if metadata["post_capture_binding"]["valid"] is not True:
        invalid.append("primitive_post_capture_binding_invalid")
    invalid = list(dict.fromkeys(invalid))
    valid = not invalid
    raw_receipt = {
        "path": raw_relative,
        "file_sha256": loaded["raw_file_sha256"],
        "metadata_payload_sha256": _payload_sha256(metadata),
        "bindings_payload_sha256": _payload_sha256(metadata["bindings"]),
        "array_sha256": dict(metadata["array_sha256"]),
        "capture_id": metadata["session"]["capture_id"],
        "status": metadata["status"],
        "schema": metadata["schema"],
    }
    core: dict[str, Any] = {
        "schema": EXTERNAL_POST_RECEIPT_SCHEMA,
        "status": "POST_CAPTURE_PASS" if valid else "INVALID",
        "valid": valid,
        "invalid_reasons": invalid,
        "scope": EXTERNAL_POST_RECEIPT_SCOPE,
        "raw": raw_receipt,
        "external_bindings": external_bindings,
        "operator_confirmations": metadata["operator_confirmations"],
        "primitive_post_capture_binding": metadata["post_capture_binding"],
        "audio_lock": lock_receipt,
        "resolved_devices": devices,
        "analysis_admission_eligible": valid,
        "canonical_training_eligible": False,
        "hardware_sample_slip_authority": False,
    }
    receipt = {**core, "receipt_payload_sha256": _payload_sha256(core)}
    target_relative = external_post_receipt_relative_path(raw_relative)
    published = _publish_json_noreplace(root, target_relative, receipt)
    return {**published, "receipt": receipt}


def issue_invalid_external_post_capture_receipt_v6(
    *,
    repository_root: str | os.PathLike[str],
    raw_relative_path: str,
    expected_raw_file_sha256: str,
    audio_lock: Mapping[str, Any],
    expected_resolved_devices: Mapping[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    """post 외부 재검증 실패를 raw prebinding에만 묶은 INVALID receipt로 보존한다.

    이 경로는 external bytes가 정상이라고 주장하지 않는다. raw 자체의 canonical
    container와 내부 prebinding만 검증하고 analysis admission을 영구히 false로 둔다.
    """

    root = _repository_root(repository_root)
    raw_relative = _canonical_relative_path(raw_relative_path, label="live raw path")
    if raw_relative != SEALED_RAW_RELATIVE_PATH:
        raise ValueError("INVALID receipt raw path가 sealed path와 다릅니다")
    if (
        not isinstance(errors, list)
        or not errors
        or any(type(item) is not str or not item for item in errors)
    ):
        raise ValueError("INVALID external receipt에는 nonempty exact error list가 필요합니다")
    header = _extract_raw_metadata_snapshot(
        repository_root=root,
        raw_relative_path=raw_relative,
        expected_file_sha256=expected_raw_file_sha256,
    )
    metadata_header = header["metadata"]
    prebindings = metadata_header.get("bindings")
    if not isinstance(prebindings, Mapping):
        raise ValueError("INVALID receipt raw에 prebindings가 없습니다")
    loaded = load_live_raw_v6(
        root / raw_relative,
        repository_root=root,
        expected_bindings=prebindings,
        expected_raw_file_sha256=header["snapshot"]["sha256"],
        require_analysis_admission=False,
    )
    metadata = loaded["metadata"]
    devices = _exact_mapping(
        expected_resolved_devices, {"input", "output"}, label="resolved devices"
    )
    if any(type(devices[name]) is not int or devices[name] < 0 for name in devices):
        raise ValueError("INVALID receipt resolved device가 exact하지 않습니다")
    try:
        lock_receipt = validate_held_audio_lock(
            root, audio_lock, expected_purpose="fullband_causal_v6_live_capture"
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as lock_error:
        lock = _exact_mapping(
            audio_lock, _AUDIO_LOCK_IDENTITY_KEYS, label="audio lock"
        )
        lock_receipt = {
            "path": _canonical_relative_path(lock["path"], label="audio lock path"),
            "identity_sha256": audio_lock_identity_sha256(lock),
            "pid": lock["pid"],
            "uid": lock["uid"],
            "purpose": lock["purpose"],
            "device": lock["device"],
            "inode": lock["inode"],
            "exclusive_lock_observed": False,
        }
        errors = [
            *errors,
            f"audio_lock_validation_failed:{type(lock_error).__name__}:{lock_error}",
        ]
    errors = list(dict.fromkeys(errors))
    raw_receipt = {
        "path": raw_relative,
        "file_sha256": loaded["raw_file_sha256"],
        "metadata_payload_sha256": _payload_sha256(metadata),
        "bindings_payload_sha256": _payload_sha256(metadata["bindings"]),
        "array_sha256": dict(metadata["array_sha256"]),
        "capture_id": metadata["session"]["capture_id"],
        "status": metadata["status"],
        "schema": metadata["schema"],
    }
    core: dict[str, Any] = {
        "schema": EXTERNAL_POST_RECEIPT_SCHEMA,
        "status": "INVALID",
        "valid": False,
        "invalid_reasons": errors,
        "scope": EXTERNAL_POST_RECEIPT_SCOPE,
        "raw": raw_receipt,
        # 이 mapping은 raw가 캡처 전에 봉인한 값이다. actual post validation이라고
        # 부르지 않으며 status/valid가 어떤 offline admission도 막는다.
        "external_bindings": metadata["bindings"],
        "operator_confirmations": metadata["operator_confirmations"],
        "primitive_post_capture_binding": metadata["post_capture_binding"],
        "audio_lock": lock_receipt,
        "resolved_devices": devices,
        "analysis_admission_eligible": False,
        "canonical_training_eligible": False,
        "hardware_sample_slip_authority": False,
    }
    receipt = {**core, "receipt_payload_sha256": _payload_sha256(core)}
    published = _publish_json_noreplace(
        root, external_post_receipt_relative_path(raw_relative), receipt
    )
    return {**published, "receipt": receipt}


def _load_external_post_capture_receipt_v6(
    *,
    repository_root: str | os.PathLike[str],
    receipt_relative_path: str,
    expected_receipt_file_sha256: str,
    plan_envelope_path: str,
    live_authority_path: str,
    meter_raw_path: str,
    level_evidence_path: str,
    hardware_path: str,
    require_current_meter_execution: bool,
) -> dict[str, Any]:
    """현재 실행 identity 선택을 제외한 공통 receipt/raw 검증 core."""

    if type(require_current_meter_execution) is not bool:
        raise TypeError("require_current_meter_execution은 exact bool이어야 합니다")

    root = _repository_root(repository_root)
    receipt_relative = _canonical_relative_path(
        receipt_relative_path, label="external post receipt path"
    )
    snapshot = read_repository_file_nofollow(root, receipt_relative)
    if snapshot["sha256"] != _require_sha256(
        expected_receipt_file_sha256, label="expected external receipt file SHA"
    ):
        raise ValueError("external post receipt file SHA가 expected 값과 다릅니다")
    receipt = _load_canonical_json_file(snapshot, label="external post receipt")
    _exact_mapping(receipt, _RECEIPT_KEYS, label="external post receipt")
    if receipt["schema"] != EXTERNAL_POST_RECEIPT_SCHEMA:
        raise ValueError("external post receipt schema가 다릅니다")
    declared = _require_sha256(
        receipt["receipt_payload_sha256"], label="external receipt payload SHA"
    )
    core = {
        key: item for key, item in receipt.items() if key != "receipt_payload_sha256"
    }
    if _payload_sha256(core) != declared:
        raise ValueError("external post receipt payload SHA가 내용과 다릅니다")
    raw = _exact_mapping(receipt["raw"], _RAW_RECEIPT_KEYS, label="receipt raw")
    if receipt_relative != external_post_receipt_relative_path(raw["path"]):
        raise ValueError("external receipt path가 raw sibling sealed path가 아닙니다")
    bindings = receipt["external_bindings"]
    if not isinstance(bindings, Mapping) or set(bindings) != set(BINDING_KEYS):
        raise ValueError("external receipt binding key 집합이 exact하지 않습니다")
    current = _collect_offline_external_bindings_without_backend(
        repository_root=root,
        expected_bindings=bindings,
        plan_envelope_path=plan_envelope_path,
        live_authority_path=live_authority_path,
        meter_raw_path=meter_raw_path,
        level_evidence_path=level_evidence_path,
        hardware_path=hardware_path,
        require_current_meter_execution=require_current_meter_execution,
    )
    if current != bindings:
        raise ValueError("offline current external files가 post receipt binding과 다릅니다")
    loaded = load_live_raw_v6(
        root / raw["path"],
        repository_root=root,
        expected_bindings=current,
        expected_raw_file_sha256=raw["file_sha256"],
        require_analysis_admission=False,
    )
    metadata = loaded["metadata"]
    comparisons = {
        "metadata_payload_sha256": _payload_sha256(metadata),
        "bindings_payload_sha256": _payload_sha256(metadata["bindings"]),
        "capture_id": metadata["session"]["capture_id"],
        "status": metadata["status"],
        "schema": metadata["schema"],
    }
    for name, observed in comparisons.items():
        if raw[name] != observed:
            raise ValueError(f"external receipt raw.{name}가 canonical raw와 다릅니다")
    if raw["array_sha256"] != metadata["array_sha256"]:
        raise ValueError("external receipt raw array SHA map이 canonical raw와 다릅니다")
    _validate_receipt_witness_against_raw_metadata(receipt, metadata)
    if (
        receipt["status"] != "POST_CAPTURE_PASS"
        or receipt["valid"] is not True
        or receipt["invalid_reasons"] != []
        or receipt["analysis_admission_eligible"] is not True
        or receipt["canonical_training_eligible"] is not False
        or receipt["hardware_sample_slip_authority"] is not False
        or receipt["scope"] != EXTERNAL_POST_RECEIPT_SCOPE
        or metadata["status"] != "CAPTURE_PASS"
        or metadata["valid"] is not True
    ):
        raise ValueError("INVALID external receipt/raw는 offline analysis admission이 아닙니다")
    return {
        "receipt_path": root.joinpath(*PurePosixPath(receipt_relative).parts),
        "receipt_file_sha256": snapshot["sha256"],
        "receipt": receipt,
        "raw": loaded,
    }


def load_external_post_capture_receipt_v6(
    *,
    repository_root: str | os.PathLike[str],
    receipt_relative_path: str,
    expected_receipt_file_sha256: str,
    plan_envelope_path: str,
    live_authority_path: str,
    meter_raw_path: str,
    level_evidence_path: str,
    hardware_path: str,
) -> dict[str, Any]:
    """offline 분석용: capture 당시와 같은 clean meter 실행 identity까지 요구한다."""

    return _load_external_post_capture_receipt_v6(
        repository_root=repository_root,
        receipt_relative_path=receipt_relative_path,
        expected_receipt_file_sha256=expected_receipt_file_sha256,
        plan_envelope_path=plan_envelope_path,
        live_authority_path=live_authority_path,
        meter_raw_path=meter_raw_path,
        level_evidence_path=level_evidence_path,
        hardware_path=hardware_path,
        require_current_meter_execution=True,
    )


def _load_external_post_capture_receipt_v6_archival_forensics(
    *,
    repository_root: str | os.PathLike[str],
    receipt_relative_path: str,
    expected_receipt_file_sha256: str,
    plan_envelope_path: str,
    live_authority_path: str,
    meter_raw_path: str,
    level_evidence_path: str,
    hardware_path: str,
) -> dict[str, Any]:
    """과거 raw forensic용 loader.

    receipt가 봉인한 plan/authority/meter/evidence/hardware/raw의 canonical bytes와 모든
    SHA/semantic binding은 동일하게 재검증한다. 다만 capture commit과 다른 clean 분석
    commit에서 읽을 수 있도록 *현재* ``set_amp_level.py`` blob 동일성만 요구하지 않는다.
    반환값은 분석/P/S/학습 admission capability가 아니다.
    """

    admitted = _load_external_post_capture_receipt_v6(
        repository_root=repository_root,
        receipt_relative_path=receipt_relative_path,
        expected_receipt_file_sha256=expected_receipt_file_sha256,
        plan_envelope_path=plan_envelope_path,
        live_authority_path=live_authority_path,
        meter_raw_path=meter_raw_path,
        level_evidence_path=level_evidence_path,
        hardware_path=hardware_path,
        require_current_meter_execution=False,
    )
    receipt = admitted["receipt"]
    return {
        "schema": "fullband_causal_v6_archival_forensics_source_v1",
        "scope": "archival_forensics_only_no_analysis_no_plant_no_training_authority",
        "analysis_admission_eligible": False,
        "canonical_training_eligible": False,
        "source_receipt_evidence": {
            "path": receipt_relative_path,
            "file_sha256": admitted["receipt_file_sha256"],
            "schema": receipt["schema"],
            "status": receipt["status"],
            "valid": receipt["valid"],
            "receipt_payload_sha256": receipt["receipt_payload_sha256"],
        },
        "forensic_raw_snapshot": admitted["raw"],
    }


def _current_analysis_execution_identity_v6(
    repository_root: Path,
) -> dict[str, Any]:
    """현재 clean HEAD의 exact v6 adapter blob을 publication authority로 읽는다."""

    repository_identity = _exact_mapping(
        repository_execution_identity(
            repository_root, ANALYSIS_ADAPTER_RELATIVE_PATH
        ),
        {
            "repository_commit",
            "repository_branch",
            "repository_dirty",
            "script_path",
            "script_file_sha256",
        },
        label="current v6 adapter repository execution identity",
    )
    current = {
        "repository_commit": repository_identity["repository_commit"],
        "repository_branch": repository_identity["repository_branch"],
        "repository_dirty": repository_identity["repository_dirty"],
        "adapter_path": repository_identity["script_path"],
        "adapter_file_sha256": repository_identity["script_file_sha256"],
    }
    if (
        type(current["repository_commit"]) is not str
        or len(current["repository_commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in current["repository_commit"]
        )
        or type(current["repository_branch"]) is not str
        or not current["repository_branch"]
        or current["repository_dirty"] is not False
        or current["adapter_path"] != ANALYSIS_ADAPTER_RELATIVE_PATH
    ):
        raise ValueError("current analysis execution identity가 clean exact v6 adapter가 아닙니다")
    _require_sha256(
        current["adapter_file_sha256"], label="current v6 adapter file SHA"
    )
    return current


def _recompute_analysis_operator_from_admitted_raw_v6(
    admitted_raw: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """admitted immutable raw와 committed plan만으로 publisher가 결과를 재계산한다."""

    if not isinstance(admitted_raw, Mapping):
        raise ValueError("publisher recompute에 admitted raw mapping이 필요합니다")
    arrays = admitted_raw.get("arrays")
    metadata = admitted_raw.get("metadata")
    if not isinstance(arrays, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("publisher recompute에 admitted raw arrays/metadata가 필요합니다")
    scalar = metadata.get("duplex_telemetry_scalars")
    if not isinstance(scalar, Mapping):
        raise ValueError("publisher recompute에 duplex telemetry scalar가 필요합니다")
    required_arrays = set(TELEMETRY_ARRAY_FIELDS) | {
        "actual_submitted_pcm",
        "captured_pcm",
        "capture_valid_mask",
        "submitted_valid_mask",
    }
    missing = required_arrays - set(arrays)
    if missing:
        raise ValueError(
            "publisher recompute raw array가 부족합니다: " + ", ".join(sorted(missing))
        )
    telemetry = {
        **dict(scalar),
        **{name: arrays[name] for name in TELEMETRY_ARRAY_FIELDS},
        "actual_submitted_pcm": arrays["actual_submitted_pcm"],
        "capture_valid_mask": arrays["capture_valid_mask"],
        "submitted_valid_mask": arrays["submitted_valid_mask"],
    }
    plan_envelope = committed_plan_envelope_v6()
    if (
        not isinstance(plan_envelope, Mapping)
        or not isinstance(plan_envelope.get("signal_plan"), Mapping)
    ):
        raise ValueError("committed v6 plan envelope에 signal_plan이 없습니다")
    return analyze_committed_v6_live_delay(
        plan=plan_envelope["signal_plan"],
        submitted_pcm=arrays["actual_submitted_pcm"],
        captured_adc_pcm=arrays["captured_pcm"],
        duplex_telemetry=telemetry,
    )


def _assert_caller_matches_recomputed_v6(
    *,
    caller_analysis: Mapping[str, Any],
    caller_operator: Mapping[str, Any],
    recomputed_analysis: Mapping[str, Any],
    recomputed_operator: Mapping[str, Any],
) -> None:
    """caller 결과가 독립 재계산 결과와 JSON/ndarray 단위로 exact인지 검사한다."""

    validate_analysis_operator_v6(caller_analysis, caller_operator)
    validate_analysis_operator_v6(recomputed_analysis, recomputed_operator)
    if _canonical_json_bytes(caller_analysis) != _canonical_json_bytes(
        recomputed_analysis
    ):
        raise ValueError("caller analysis가 admitted raw 독립 재계산과 exact 일치하지 않습니다")
    if (
        not isinstance(caller_operator, Mapping)
        or not isinstance(recomputed_operator, Mapping)
        or set(caller_operator) != OPERATOR_ARRAY_NAMES | {"receipt"}
        or set(recomputed_operator) != OPERATOR_ARRAY_NAMES | {"receipt"}
    ):
        raise ValueError("caller/recomputed operator key 집합이 exact하지 않습니다")
    if _canonical_json_bytes(caller_operator["receipt"]) != _canonical_json_bytes(
        recomputed_operator["receipt"]
    ):
        raise ValueError("caller operator receipt가 admitted raw 독립 재계산과 다릅니다")
    for name in sorted(OPERATOR_ARRAY_NAMES):
        caller_array = caller_operator[name]
        recomputed_array = recomputed_operator[name]
        if type(caller_array) is not np.ndarray or type(recomputed_array) is not np.ndarray:
            raise ValueError(f"operator {name}은 exact ndarray여야 합니다")
        if (
            caller_array.dtype != recomputed_array.dtype
            or caller_array.shape != recomputed_array.shape
            or np.ascontiguousarray(caller_array).tobytes(order="C")
            != np.ascontiguousarray(recomputed_array).tobytes(order="C")
        ):
            raise ValueError(
                f"caller operator {name}이 admitted raw 독립 재계산과 exact 일치하지 않습니다"
            )


def publish_live_delay_analysis_v6(
    *,
    repository_root: str | os.PathLike[str],
    output_directory_relative_path: str,
    external_receipt_relative_path: str,
    external_receipt_file_sha256: str,
    plan_envelope_path: str,
    live_authority_path: str,
    meter_raw_path: str,
    level_evidence_path: str,
    hardware_path: str,
    analysis_execution_identity: Mapping[str, Any],
    analysis: Mapping[str, Any],
    operator: Mapping[str, Any],
) -> dict[str, Any]:
    """검증된 actual receipt/raw 뒤에서만 두 immutable artifact를 발행한다."""

    root = _repository_root(repository_root)
    directory = _canonical_relative_path(
        output_directory_relative_path, label="analysis output directory"
    ).rstrip("/")
    if not directory.startswith("results/fullband_causal_v6/"):
        raise ValueError("v6 live analysis는 results/fullband_causal_v6 아래에만 발행합니다")
    receipt_sha = _require_sha256(
        external_receipt_file_sha256, label="external receipt file SHA"
    )
    # 호출자가 임의 64hex만 넘겨 raw-before-analysis를 우회하지 못하게 publication
    # API 자체가 actual receipt, current external bytes와 canonical raw를 다시 admission한다.
    admitted = load_external_post_capture_receipt_v6(
        repository_root=root,
        receipt_relative_path=_canonical_relative_path(
            external_receipt_relative_path, label="analysis external receipt path"
        ),
        expected_receipt_file_sha256=receipt_sha,
        plan_envelope_path=_canonical_relative_path(
            plan_envelope_path, label="analysis plan path"
        ),
        live_authority_path=_canonical_relative_path(
            live_authority_path, label="analysis authority path"
        ),
        meter_raw_path=_canonical_relative_path(
            meter_raw_path, label="analysis meter path"
        ),
        level_evidence_path=_canonical_relative_path(
            level_evidence_path, label="analysis evidence path"
        ),
        hardware_path=_canonical_relative_path(
            hardware_path, label="analysis hardware path"
        ),
    )
    if admitted["receipt_file_sha256"] != receipt_sha:
        raise ValueError("analysis publisher receipt SHA admission이 다릅니다")
    execution_keys = {
        "repository_commit",
        "repository_branch",
        "repository_dirty",
        "adapter_path",
        "adapter_file_sha256",
    }
    execution = _exact_mapping(
        analysis_execution_identity,
        execution_keys,
        label="analysis execution identity",
    )
    if (
        type(execution["repository_commit"]) is not str
        or len(execution["repository_commit"]) != 40
        or any(c not in "0123456789abcdef" for c in execution["repository_commit"])
        or type(execution["repository_branch"]) is not str
        or not execution["repository_branch"]
        or execution["repository_dirty"] is not False
        or execution["adapter_path"]
        != ANALYSIS_ADAPTER_RELATIVE_PATH
    ):
        raise ValueError("analysis execution identity가 clean exact v6 adapter가 아닙니다")
    _require_sha256(execution["adapter_file_sha256"], label="analysis adapter SHA")
    current_execution = _current_analysis_execution_identity_v6(root)
    if execution != current_execution:
        raise ValueError(
            "analysis execution identity가 current clean exact v6 adapter checkout과 다릅니다"
        )
    raw_session = admitted["raw"]["metadata"].get("session")
    if not isinstance(raw_session, Mapping) or any(
        raw_session.get(key) != value for key, value in execution.items()
    ):
        raise ValueError("analysis execution identity가 admitted raw capture와 다릅니다")
    recomputed_analysis, recomputed_operator = (
        _recompute_analysis_operator_from_admitted_raw_v6(admitted["raw"])
    )
    _assert_caller_matches_recomputed_v6(
        caller_analysis=analysis,
        caller_operator=operator,
        recomputed_analysis=recomputed_analysis,
        recomputed_operator=recomputed_operator,
    )
    analysis_value = json.loads(_canonical_json_bytes(analysis).decode("utf-8"))
    if (
        analysis_value.get("canonical_training_eligible") is not False
        or analysis_value.get("hardware_slip_authority_available") is not False
    ):
        raise ValueError("offline analysis가 canonical/slip 권한을 확대했습니다")
    operator_receipt = operator.get("receipt")
    if not isinstance(operator_receipt, Mapping):
        raise ValueError("offline operator receipt가 필요합니다")
    if (
        operator_receipt.get("canonical_training_eligible") is not False
        or operator_receipt.get("hardware_sample_slip_authority_available") is not False
    ):
        raise ValueError("offline operator가 canonical/slip 권한을 확대했습니다")
    raw_metadata = admitted["raw"]["metadata"]
    raw_arrays = raw_metadata.get("array_sha256")
    captured_binding = analysis_value.get("captured_raw_binding")
    if not isinstance(raw_arrays, Mapping) or not isinstance(captured_binding, Mapping):
        raise ValueError("analysis/raw array SHA binding이 필요합니다")
    if (
        captured_binding.get("captured_adc_pcm_sha256")
        != raw_arrays.get("captured_pcm")
        or captured_binding.get("actual_submitted_pcm_sha256")
        != raw_arrays.get("actual_submitted_pcm")
        or operator_receipt.get("captured_adc_pcm_sha256")
        != raw_arrays.get("captured_pcm")
        or operator_receipt.get("actual_submitted_pcm_sha256")
        != raw_arrays.get("actual_submitted_pcm")
    ):
        raise ValueError("analysis/operator가 admitted v6 raw array SHA와 다릅니다")
    arrays = {
        name: np.array(value, copy=True, order="C")
        for name, value in operator.items()
        if name != "receipt"
    }
    if not arrays or any(name not in operator_receipt["operator_array_sha256"] for name in arrays):
        raise ValueError("operator array/receipt 결속이 불완전합니다")
    for name, value in arrays.items():
        if operator_receipt["operator_array_sha256"][name] != _array_contract_sha256(value):
            raise ValueError(f"operator array {name} SHA가 core receipt와 다릅니다")
    # NPZ 자체는 receipt JSON을 uint8로 포함해 allow_pickle 없이 재독해 가능하다.
    operator_metadata = {
        "schema": OPERATOR_CONTAINER_SCHEMA,
        "external_post_receipt_file_sha256": receipt_sha,
        "analysis_execution_identity": dict(execution),
        "canonical_training_eligible": False,
        "hardware_sample_slip_authority": False,
        "operator_receipt": json.loads(
            _canonical_json_bytes(operator_receipt).decode("utf-8")
        ),
    }
    stream = io.BytesIO()
    np.savez(
        stream,
        **arrays,
        metadata_json_utf8=np.frombuffer(
            _canonical_json_bytes(operator_metadata), dtype=np.uint8
        ).copy(),
    )
    operator_bytes = stream.getvalue()
    analysis_envelope = {
        "schema": ANALYSIS_ENVELOPE_SCHEMA,
        "external_post_receipt_file_sha256": receipt_sha,
        "operator_file_sha256": hashlib.sha256(operator_bytes).hexdigest(),
        "analysis_execution_identity": dict(execution),
        "analysis": analysis_value,
        "canonical_training_eligible": False,
        "hardware_sample_slip_authority": False,
    }
    analysis_bytes = _canonical_json_file_bytes(analysis_envelope)
    return _publish_analysis_directory_noreplace(
        root,
        directory,
        analysis_bytes=analysis_bytes,
        operator_bytes=operator_bytes,
    )


def publish_live_delay_failure_v6(
    *,
    repository_root: str | os.PathLike[str],
    failure_relative_path: str,
    raw_relative_path: str,
    raw_file_sha256: str,
    external_receipt_relative_path: str,
    external_receipt_file_sha256: str,
    failure_stage: str,
    optimizer_started: bool,
    error: str,
    available_snr_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """offline 실패를 raw/receipt bytes에 결속해 단일 immutable JSON으로 보존한다."""

    root = _repository_root(repository_root)
    target = _canonical_relative_path(failure_relative_path, label="failure path")
    if not target.startswith("results/fullband_causal_v6/") or not target.endswith(".json"):
        raise ValueError("v6 failure는 results/fullband_causal_v6 아래 JSON이어야 합니다")
    raw_relative = _canonical_relative_path(raw_relative_path, label="failure raw path")
    if raw_relative != SEALED_RAW_RELATIVE_PATH:
        raise ValueError("v6 failure raw path가 exact sealed path와 다릅니다")
    receipt_relative = _canonical_relative_path(
        external_receipt_relative_path, label="failure receipt path"
    )
    if receipt_relative != external_post_receipt_relative_path(raw_relative):
        raise ValueError("v6 failure receipt가 raw의 canonical sibling이 아닙니다")
    expected_raw_sha = _require_sha256(raw_file_sha256, label="failure raw SHA")
    expected_receipt_sha = _require_sha256(
        external_receipt_file_sha256, label="failure receipt SHA"
    )
    if type(failure_stage) is not str or not failure_stage.strip():
        raise ValueError("failure_stage는 nonempty string이어야 합니다")
    if type(optimizer_started) is not bool:
        raise ValueError("optimizer_started는 exact bool이어야 합니다")
    if type(error) is not str or not error.strip():
        raise ValueError("error는 nonempty string이어야 합니다")
    snr = None
    if available_snr_receipt is not None:
        if not isinstance(available_snr_receipt, Mapping):
            raise ValueError("available SNR receipt는 mapping 또는 null이어야 합니다")
        snr = json.loads(_canonical_json_bytes(available_snr_receipt).decode("utf-8"))

    with ExitStack() as stack:
        raw_guard = stack.enter_context(
            RepositoryFileGuard(root, raw_relative, label="v6 failure raw")
        )
        receipt_guard = stack.enter_context(
            RepositoryFileGuard(root, receipt_relative, label="v6 failure receipt")
        )
        if raw_guard.sha256 != expected_raw_sha:
            raise ValueError("v6 failure raw SHA가 actual bytes와 다릅니다")
        if receipt_guard.sha256 != expected_receipt_sha:
            raise ValueError("v6 failure receipt SHA가 actual bytes와 다릅니다")
        try:
            receipt = json.loads(receipt_guard.bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("v6 failure receipt JSON을 읽을 수 없습니다") from exc
        if not isinstance(receipt, Mapping) or receipt.get("schema") != EXTERNAL_POST_RECEIPT_SCHEMA:
            raise ValueError("v6 failure receipt schema가 exact v6 post receipt가 아닙니다")
        raw_binding = receipt.get("raw")
        if not isinstance(raw_binding, Mapping) or raw_binding.get("path") != raw_relative:
            raise ValueError("v6 failure receipt의 raw path binding이 다릅니다")
        if raw_binding.get("file_sha256") != expected_raw_sha:
            raise ValueError("v6 failure receipt의 raw SHA binding이 다릅니다")
        raw_guard.verify()
        receipt_guard.verify()

    core = {
        "schema": FAILURE_SCHEMA,
        "status": "FAILED",
        "raw": {"path": raw_relative, "file_sha256": expected_raw_sha},
        "external_post_receipt": {
            "path": receipt_relative,
            "file_sha256": expected_receipt_sha,
        },
        "failure_stage": failure_stage,
        "optimizer_started": optimizer_started,
        "error": error,
        "available_snr_receipt": snr,
        "analysis_published": False,
        "operator_published": False,
        "canonical_training_eligible": False,
        "hardware_sample_slip_authority": False,
    }
    value = {**core, "failure_payload_sha256": _payload_sha256(core)}
    return _publish_json_noreplace(root, target, value)


def _array_contract_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _publish_analysis_directory_noreplace(
    repository_root: Path,
    directory_relative_path: str,
    *,
    analysis_bytes: bytes,
    operator_bytes: bytes,
) -> dict[str, Any]:
    """analysis.json+operator.npz를 하나의 renameat2 transaction으로 발행한다."""

    directory = _canonical_relative_path(
        directory_relative_path, label="analysis directory"
    )
    target = PurePosixPath(directory)
    parent_relative = PurePosixPath(*target.parts[:-1]).as_posix()
    target_name = target.name
    _placeholder = f"{parent_relative}/.parent-open-placeholder"
    _filename, chain = _open_parent_chain(
        repository_root, _placeholder, create=True
    )
    parent_fd = chain[-1][1]
    staging = f".{target_name}.staging-{secrets.token_hex(16)}"
    staging_fd = -1
    published = False
    try:
        try:
            os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"analysis directory를 덮어쓸 수 없습니다: {directory}")
        os.mkdir(staging, 0o700, dir_fd=parent_fd)
        staging_fd = os.open(staging, _directory_flags(), dir_fd=parent_fd)
        for filename, payload in (
            ("analysis.json", analysis_bytes),
            ("operator.npz", operator_bytes),
        ):
            descriptor = os.open(
                filename,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=staging_fd,
            )
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                os.close(descriptor)
        os.fsync(staging_fd)
        _verify_chain(chain)
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("atomic analysis publication에 renameat2가 필요합니다")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            parent_fd,
            os.fsencode(staging),
            parent_fd,
            os.fsencode(target_name),
            1,  # RENAME_NOREPLACE
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(
                    f"analysis directory를 덮어쓸 수 없습니다: {directory}"
                )
            raise OSError(error, os.strerror(error), directory)
        published = True
        staging = ""
        os.fsync(parent_fd)
        _verify_chain(chain)
    finally:
        if staging_fd >= 0:
            os.close(staging_fd)
        if staging and not published:
            try:
                child_fd = os.open(staging, _directory_flags(), dir_fd=parent_fd)
                try:
                    for filename in ("analysis.json", "operator.npz"):
                        try:
                            os.unlink(filename, dir_fd=child_fd)
                        except FileNotFoundError:
                            pass
                finally:
                    os.close(child_fd)
                os.rmdir(staging, dir_fd=parent_fd)
            except OSError:
                pass
        _close_chain(chain)
    output = repository_root.joinpath(*target.parts)
    return {
        "analysis": {
            "path": output / "analysis.json",
            "relative_path": f"{directory}/analysis.json",
            "file_sha256": hashlib.sha256(analysis_bytes).hexdigest(),
        },
        "operator": {
            "path": output / "operator.npz",
            "relative_path": f"{directory}/operator.npz",
            "file_sha256": hashlib.sha256(operator_bytes).hexdigest(),
        },
    }


def _publish_bytes_noreplace(
    repository_root: str | os.PathLike[str], relative_path: str, payload: bytes
) -> dict[str, Any]:
    """binary artifact용 dirfd/no-replace/fsync publisher."""

    root = _repository_root(repository_root)
    relative = _canonical_relative_path(relative_path, label="binary target")
    filename, chain = _open_parent_chain(root, relative, create=True)
    parent_fd = chain[-1][1]
    staging = ""
    descriptor = -1
    linked = False
    try:
        try:
            os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"기존 binary target을 덮어쓰지 않습니다: {relative}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        for _ in range(128):
            staging = f".{filename}.staging-{secrets.token_hex(16)}"
            try:
                descriptor = os.open(staging, flags, 0o600, dir_fd=parent_fd)
                break
            except FileExistsError:
                continue
        if descriptor < 0:
            raise RuntimeError("binary staging 이름을 확보하지 못했습니다")
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _verify_chain(chain)
        os.link(staging, filename, src_dir_fd=parent_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
        linked = True
        try:
            os.unlink(staging, dir_fd=parent_fd)
            staging = ""
        finally:
            os.fsync(parent_fd)
        _verify_chain(chain)
    except BaseException:
        if staging and not linked:
            try:
                os.unlink(staging, dir_fd=parent_fd)
            except OSError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        _close_chain(chain)
    return {
        "path": root.joinpath(*PurePosixPath(relative).parts),
        "relative_path": relative,
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }


__all__ = [
    "ANALYSIS_ENVELOPE_SCHEMA",
    "EXTERNAL_POST_RECEIPT_SCHEMA",
    "EXTERNAL_POST_RECEIPT_SUFFIX",
    "FAILURE_SCHEMA",
    "audio_lock_identity_sha256",
    "assert_repository_target_fresh_nofollow",
    "collect_actual_external_bindings_v6",
    "external_post_receipt_relative_path",
    "issue_external_post_capture_receipt_v6",
    "issue_invalid_external_post_capture_receipt_v6",
    "load_external_post_capture_receipt_v6",
    "publish_live_delay_analysis_v6",
    "publish_live_delay_failure_v6",
    "read_repository_file_nofollow",
    "validate_held_audio_lock",
]
