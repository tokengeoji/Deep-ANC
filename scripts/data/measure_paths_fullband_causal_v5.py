#!/usr/bin/env python3
"""v5 near-white causal P/S signal-only 계획을 생성하고 exact 조건수를 검사한다."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import hashlib
import importlib
import tempfile
import math
import io
import secrets
import shlex
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.dsp.fullband_causal_v5 import (  # noqa: E402
    build_plan_v5,
    exact_condition_audit_v5,
)
from deep_anc.data.repository_fd import repository_execution_identity  # noqa: E402


FULLBAND_V5_LIVE_ADAPTER_IMPLEMENTED = True
ADAPTER_REPOSITORY_PATH = "scripts/data/measure_paths_fullband_causal_v5.py"
LIVE_AUDIO_LOCK_PURPOSE = "fullband_causal_v5_live_capture"
LIVE_WATCHDOG_GRACE_SECONDS = 2.0
SPEAKER_DISCONNECT_NOTICE = (
    "[스피커 출력 종료] v5 오디오 스트림 stop/abort/close가 확인됐습니다. "
    "지금 스피커/앰프 연결을 즉시 해제하세요."
)
SPEAKER_STOP_UNCONFIRMED_NOTICE = (
    "[경고: 출력 종료 확인 불가] 지금 스피커/앰프를 물리적으로 즉시 분리하세요. "
    "이 캡처는 INVALID로만 보존합니다."
)
SPEAKER_PREFLIGHT_ABORT_NOTICE = (
    "[출력 시작 전 중단] 오디오 출력을 열지 않았습니다. "
    "스피커/앰프가 연결되어 있으면 지금 물리적으로 분리하세요."
)
_CONFIRMATION_KEYS = {
    "speaker_output",
    "user_present",
    "volume_minimum",
    "routing_and_geometry",
    "same_amplifier_setting",
}


def _repository_execution_identity() -> dict[str, Any]:
    """Return exact clean-checkout identity or fail before backend import."""
    identity = repository_execution_identity(REPO_ROOT, ADAPTER_REPOSITORY_PATH)
    return {
        "repository_commit": identity["repository_commit"],
        "repository_branch": identity["repository_branch"],
        "repository_dirty": identity["repository_dirty"],
        "adapter_path": identity["script_path"],
        "adapter_file_sha256": identity["script_file_sha256"],
    }


def _lexical_repository_target(target: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(REPO_ROOT)))
    lexical = Path(os.path.abspath(os.fspath(target.expanduser())))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise ValueError("signal plan은 저장소 내부에만 저장합니다") from error
    cursor = root
    if cursor.is_symlink():
        raise ValueError("repository root symlink를 거부합니다")
    for part in relative.parts[:-1]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"output parent symlink를 거부합니다: {cursor}")
    if lexical.is_symlink():
        raise ValueError(f"output target symlink를 거부합니다: {lexical}")
    return lexical


def _write_json_no_replace(payload: dict, target: Path) -> Path:
    lexical = _lexical_repository_target(target)
    lexical.parent.mkdir(parents=True, exist_ok=True)
    lexical = _lexical_repository_target(lexical)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(os.fspath(lexical), flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    directory_fd = os.open(
        os.fspath(lexical.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return lexical


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _publish_raw_no_replace(
    *,
    plan: dict,
    submitted_pcm: np.ndarray,
    captured_pcm: np.ndarray,
    callback_frames: np.ndarray,
) -> Path:
    """이미 얻은 배열만 exact plan raw path에 immutable NPZ로 발행한다."""

    target = REPO_ROOT / plan["publisher_contract"]["raw_session_relative_path"]
    lexical = _lexical_repository_target(target)
    lexical.parent.mkdir(parents=True, exist_ok=True)
    lexical = _lexical_repository_target(lexical)
    submitted = np.asarray(submitted_pcm)
    captured = np.asarray(captured_pcm)
    callbacks = np.asarray(callback_frames)
    if submitted.dtype != np.int16 or list(submitted.shape) != plan["actual_submitted_shape"]:
        raise ValueError("submitted_pcm dtype/shape이 plan과 다릅니다")
    if _array_sha256(submitted) != plan["actual_submitted_pcm_sha256"]:
        raise ValueError("submitted_pcm SHA가 plan과 다릅니다")
    if captured.dtype != np.dtype("<i4") or captured.ndim != 2 or captured.shape[0] != submitted.shape[0] or captured.shape[1] != 2:
        raise ValueError("captured_pcm은 submitted와 같은 frame 수의 exact <i4 [frame,2]여야 합니다")
    expected_callbacks = math.ceil(len(captured) / 256)
    if (
        callbacks.dtype != np.dtype("<i8")
        or callbacks.ndim != 1
        or len(callbacks) != expected_callbacks
        or np.any(callbacks != 256)
        or not (0 <= int(np.sum(callbacks)) - len(captured) < 256)
    ):
        raise ValueError("callback_frames는 exact <i8 256-frame accounting이어야 합니다")
    metadata = {
        "schema": "fullband_causal_raw_capture_v5",
        "signal_plan_payload_sha256": plan["canonical_payload_sha256"],
        "submitted_pcm_sha256": _array_sha256(submitted),
        "captured_pcm_sha256": _array_sha256(captured),
        "callback_frames_sha256": _array_sha256(callbacks),
        "live_authority_at_plan_time": None,
        "role": plan["publisher_contract"]["role"],
        "callback_semantics": plan["publisher_contract"]["callback_semantics"],
        "live_xrun_slip_authority": plan["publisher_contract"]["live_xrun_slip_authority"],
    }
    metadata_bytes = np.frombuffer(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    ).copy()
    stream = io.BytesIO()
    np.savez(
        stream,
        submitted_pcm=submitted,
        captured_pcm=captured,
        callback_frames=callbacks,
        metadata_json_utf8=metadata_bytes,
    )
    raw_bytes = stream.getvalue()
    descriptor = -1
    staging: Path | None = None
    try:
        descriptor, staging_name = tempfile.mkstemp(
            prefix=f".{lexical.name}.staging-", dir=lexical.parent
        )
        staging = Path(staging_name)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        # same-directory hard-link publication은 target이 이미 있으면 atomic하게 실패한다.
        os.link(staging, lexical, follow_symlinks=False)
        staging.unlink()
    except Exception:
        if staging is not None:
            staging.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    directory_fd = os.open(
        os.fspath(lexical.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return lexical


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _repository_relative(value: str | Path, *, label: str) -> str:
    root = Path(os.path.abspath(os.fspath(REPO_ROOT)))
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    lexical = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = lexical.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"{label}가 repository 밖입니다") from error
    if not relative or "\\" in relative or any(
        part in {"", ".", ".."} for part in Path(relative).parts
    ):
        raise ValueError(f"{label}가 canonical repository 상대경로가 아닙니다")
    return relative


def _assert_live_capture_authority_enabled_before_execute(
    args: argparse.Namespace,
) -> None:
    """sealed authority가 명시적으로 열기 전에는 live 출력을 거부한다.

    현재 committed v5 authority는 의도적으로 capture-only다. bytes와
    provenance가 내부적으로 일관되더라도 어댑터가 ``_execute_live`` 또는
    audio backend import까지 도달하지 못하게 static contract와 별도로 막는다.
    """

    from deep_anc.dsp.fullband_live_authority_v5 import (
        EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256,
        EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256,
        SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
        load_exact_saved_live_capture_authority_v5,
    )

    authority_relative = _repository_relative(
        args.live_authority, label="live authority"
    )
    if authority_relative != SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH:
        raise ValueError("--live-authority가 sealed v5 authority path와 다릅니다")
    loaded = load_exact_saved_live_capture_authority_v5(
        REPO_ROOT / authority_relative,
        repository_root=REPO_ROOT,
        expected_file_sha256=EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256,
        expected_payload_sha256=EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256,
    )
    authority = loaded.get("authority")
    if not isinstance(authority, Mapping):
        raise RuntimeError("sealed v5 authority payload가 없습니다")
    if authority.get("plan_live_capture_enabled") is not True:
        raise RuntimeError(
            "assets/contracts/fullband_causal_v5_live_capture_authority.json의 "
            "plan_live_capture_enabled=false이므로 --execute-live를 fail-closed "
            "차단합니다"
        )


def _meter_contract_module() -> Any:
    module = importlib.import_module("deep_anc.dsp.fullband_v5_meter")
    required = (
        "validate_fullband_v5_static_contract",
        "validate_fullband_v5_meter_raw_static",
        "validate_fullband_v5_meter_raw",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"fullband_v5_meter public API가 부족합니다: {missing}")
    return module


def _static_contract_before_backend_import(args: argparse.Namespace) -> dict[str, Any]:
    """sounddevice import 전에 pinned file/raw/PCM-clock-CPU 조건을 닫는다."""

    from deep_anc.dsp.fullband_live_authority_v5 import (
        SEALED_RAW_RELATIVE_PATH,
        committed_plan_envelope_v5,
    )
    from deep_anc.dsp.fullband_live_post_v5 import (
        assert_repository_target_fresh_nofollow,
        external_post_receipt_relative_path,
    )
    from deep_anc.dsp.measurement_level import assert_live_pcm_clock_preconditions

    raw_relative = _repository_relative(args.raw_target, label="raw target")
    if raw_relative != SEALED_RAW_RELATIVE_PATH:
        raise ValueError("--raw-target이 capture authority sealed path와 다릅니다")
    receipt_relative = external_post_receipt_relative_path(raw_relative)
    assert_repository_target_fresh_nofollow(REPO_ROOT, raw_relative)
    assert_repository_target_fresh_nofollow(REPO_ROOT, receipt_relative)
    meter_contract = _meter_contract_module()
    static = meter_contract.validate_fullband_v5_static_contract(
        repository_root=REPO_ROOT,
        plan_envelope_path=args.plan_envelope,
        live_authority_path=args.live_authority,
        level_evidence_path=args.level_evidence,
        hardware_path=args.hardware,
        raw_target_path=args.raw_target,
        require_sealed_raw_fresh=True,
    )
    if not isinstance(static, Mapping):
        raise RuntimeError("fullband-v5 static contract 반환이 mapping이 아닙니다")
    # static contract가 actual tracked plan bytes/SHA를 held dirfd 아래 검증했다.
    # 두 번째 pathname open 대신 같은 pinned builder의 in-memory envelope를 쓴다.
    static_meter = meter_contract.validate_fullband_v5_meter_raw_static(
        args.meter_raw,
        repository_root=REPO_ROOT,
        now_utc=None,
        require_fresh=True,
        require_sealed_raw_fresh=True,
    )
    static = {
        **dict(static),
        "exact_plan": {"envelope": committed_plan_envelope_v5()},
        "prevalidated_meter": static_meter,
    }
    hardware = static.get("hardware_audio")
    if not isinstance(hardware, Mapping):
        raise RuntimeError("fullband-v5 static contract에 hardware_audio가 없습니다")
    # 이 함수는 proc/sysfs/CPU만 읽고 PortAudio backend를 import/open하지 않는다.
    assert_live_pcm_clock_preconditions(dict(hardware))
    return static


def _resolve_devices(static: Mapping[str, Any]) -> dict[str, int]:
    from deep_anc.audio_io import resolve_alsa_portaudio_device

    hardware = static["hardware_audio"]
    devices = {
        "input": resolve_alsa_portaudio_device(
            hardware["input"]["card"],
            hardware["input"]["pcm"],
            "input",
            2,
        ),
        "output": resolve_alsa_portaudio_device(
            hardware["output"]["card"],
            hardware["output"]["pcm"],
            "output",
            2,
        ),
    }
    if any(type(devices[name]) is not int or devices[name] < 0 for name in devices):
        raise ValueError("resolved input/output device는 음이 아닌 exact int여야 합니다")
    return devices


def _array_contract_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _preflight_report_and_binding(
    *,
    raw: np.ndarray,
    report: Mapping[str, Any],
    hardware_identity_sha256: str,
    resolved_input_device: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from deep_anc.dsp.fullband_live_raw_v5 import (
        PREFLIGHT_IDENTITY_SCHEMA,
        PREFLIGHT_REPORT_SCHEMA,
    )

    owned = np.array(raw, dtype="<i4", copy=True, order="C")
    if owned.ndim != 2 or owned.shape[0] <= 0 or owned.shape[1] != 2:
        raise ValueError("input-only preflight raw는 nonempty exact <i4 [frames,2]여야 합니다")
    if report.get("passed") is not True:
        raise RuntimeError("input-only ERR/REF preflight가 PASS가 아닙니다")
    if report.get("resolved_input_device") != resolved_input_device:
        raise ValueError("preflight resolved input device가 capture device와 다릅니다")
    if report.get("sample_rate_hz") != 48_000 or report.get("frames") != len(owned):
        raise ValueError("preflight sample rate/frame 계약이 다릅니다")
    channels = report.get("channels")
    if not isinstance(channels, list) or len(channels) != 2:
        raise ValueError("preflight ERR/REF channel summary가 필요합니다")
    identity_core = {
        "schema": PREFLIGHT_IDENTITY_SCHEMA,
        "raw_sha256": _array_contract_sha256(owned),
        "hardware_identity_sha256": hardware_identity_sha256,
        "resolved_input_device": resolved_input_device,
        "sample_rate_hz": 48_000,
        "frames": len(owned),
    }
    identity_sha = _payload_sha256(identity_core)
    canonical_report = {
        "schema": PREFLIGHT_REPORT_SCHEMA,
        "passed": True,
        "identity_sha256": identity_sha,
        "resolved_input_device": resolved_input_device,
        "sample_rate_hz": 48_000,
        "frames": len(owned),
        "channels": channels,
    }
    binding = {
        "schema": PREFLIGHT_REPORT_SCHEMA,
        "raw_sha256": _array_contract_sha256(owned),
        "report_sha256": _payload_sha256(canonical_report),
        "identity_sha256": identity_sha,
        "passed": True,
    }
    return canonical_report, binding


def _confirmations(args: argparse.Namespace) -> dict[str, bool]:
    value = {
        "speaker_output": bool(args.confirm_speaker),
        "user_present": bool(args.confirm_user_present),
        "volume_minimum": bool(args.confirm_volume_minimum),
        "routing_and_geometry": bool(args.confirm_routing_and_geometry),
        "same_amplifier_setting": bool(args.confirm_same_amplifier_setting),
    }
    if set(value) != _CONFIRMATION_KEYS or any(item is not True for item in value.values()):
        raise ValueError("v5 live capture의 다섯 operator confirmation이 모두 필요합니다")
    return value


def _post_capture_binding_from_bindings(
    bindings: Mapping[str, Any],
    *,
    audio_lock_identity_sha256: str,
    valid: bool,
    error: str | None,
    raw_target_fresh: bool,
) -> dict[str, Any]:
    return {
        "schema": "fullband_causal_v5_post_capture_binding_v1",
        "valid": bool(valid),
        "error": error,
        "refreshed_signal_plan_file_sha256": bindings["signal_plan"]["file_sha256"],
        "refreshed_signal_plan_payload_sha256": bindings["signal_plan"]["payload_sha256"],
        "refreshed_signal_pcm_sha256": bindings["signal_plan"]["pcm_sha256"],
        "refreshed_authority_file_sha256": bindings["live_capture_authority"]["file_sha256"],
        "refreshed_authority_payload_sha256": bindings["live_capture_authority"]["payload_sha256"],
        "refreshed_meter_raw_sha256": bindings["meter"]["raw_sha256"],
        "refreshed_meter_receipt_sha256": bindings["meter"]["receipt_sha256"],
        "refreshed_level_evidence_file_sha256": bindings["level_evidence"]["file_sha256"],
        "refreshed_hardware_file_sha256": bindings["hardware"]["file_sha256"],
        "refreshed_hardware_identity_sha256": bindings["hardware"]["identity_sha256"],
        "refreshed_physical_fingerprint_sha256": bindings["hardware"]["physical_fingerprint_sha256"],
        "refreshed_audio_lock_identity_sha256": audio_lock_identity_sha256,
        "resolved_devices": dict(bindings["hardware"]["resolved_devices"]),
        "raw_target_fresh": bool(raw_target_fresh),
    }


def _offline_command(
    *,
    args: argparse.Namespace,
    receipt_relative_path: str,
    receipt_file_sha256: str,
    capture_id: str,
) -> str:
    analysis_output = f"results/fullband_causal_v5/analysis_{capture_id}"
    parts = [
        str(Path(sys.executable).absolute()),
        str(Path(__file__).resolve(strict=True)),
        "--offline-analyze",
        "--plan-envelope",
        _repository_relative(args.plan_envelope, label="plan envelope"),
        "--live-authority",
        _repository_relative(args.live_authority, label="live authority"),
        "--meter-raw",
        _repository_relative(args.meter_raw, label="meter raw"),
        "--level-evidence",
        _repository_relative(args.level_evidence, label="level evidence"),
        "--hardware",
        _repository_relative(args.hardware, label="hardware"),
        "--raw-target",
        _repository_relative(args.raw_target, label="raw target"),
        "--post-receipt",
        receipt_relative_path,
        "--expected-post-receipt-sha256",
        receipt_file_sha256,
        "--analysis-output",
        analysis_output,
    ]
    return " ".join(shlex.quote(value) for value in parts)


def _execute_live(args: argparse.Namespace) -> int:
    """exact 11.605333초 capture만 수행한다; 지연 분석은 절대 호출하지 않는다."""

    # 직접 호출도 audio primitive와 아래의 명시적 sounddevice import보다 먼저 막는다.
    try:
        _assert_live_capture_authority_enabled_before_execute(args)
    except (
        AssertionError,
        FileNotFoundError,
        FileExistsError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"[중단] v5 live authority gate 실패: {error}", file=sys.stderr)
        print(SPEAKER_PREFLIGHT_ABORT_NOTICE, file=sys.stderr, flush=True)
        return 2

    from deep_anc.audio_duplex_v5 import DuplexCaptureFailure, capture_duplex_v5
    from deep_anc.audio_io import capture_measurement_preflight_raw
    from deep_anc.dsp.fullband_live_post_v5 import (
        assert_repository_target_fresh_nofollow,
        audio_lock_identity_sha256,
        collect_actual_external_bindings_v5,
        external_post_receipt_relative_path,
        issue_external_post_capture_receipt_v5,
        issue_invalid_external_post_capture_receipt_v5,
        validate_held_audio_lock,
    )
    from deep_anc.dsp.fullband_live_raw_v5 import publish_live_raw_v5
    from deep_anc.dsp.measurement_level import (
        assert_live_pcm_clock_preconditions,
        collect_alsa_physical_fingerprint,
        repository_audio_lock,
    )

    try:
        confirmations = _confirmations(args)
        execution_identity = _repository_execution_identity()
        static = _static_contract_before_backend_import(args)
        plan, submitted = build_plan_v5()
        if plan != static["exact_plan"]["envelope"]["signal_plan"]:
            raise ValueError("committed builder plan과 sealed envelope가 다릅니다")
        submitted = np.array(submitted, dtype="<i2", copy=True, order="C")
        if len(submitted) != 557_056:
            raise ValueError("v5 exact submitted frame 수가 557056이 아닙니다")
    except (FileNotFoundError, FileExistsError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"[중단] v5 backend import 전 preflight 실패: {error}", file=sys.stderr)
        print(SPEAKER_PREFLIGHT_ABORT_NOTICE, file=sys.stderr, flush=True)
        return 2

    # static bytes/raw/PCM/clock/CPU 검증 뒤에만 PortAudio backend를 import/query한다.
    try:
        sd = importlib.import_module("sounddevice")
        devices = _resolve_devices(static)
        meter_contract = _meter_contract_module()
        meter = meter_contract.validate_fullband_v5_meter_raw(
            args.meter_raw,
            repository_root=REPO_ROOT,
            now_utc=None,
            require_fresh=True,
            require_sealed_raw_fresh=True,
        )
        if meter != static["prevalidated_meter"]:
            raise ValueError("backend import 전후 fresh meter bytes/followup이 변경됐습니다")
        if meter["hardware"]["resolved_devices"] != devices:
            raise ValueError("fresh meter resolved devices가 current PortAudio device와 다릅니다")
    except (FileNotFoundError, FileExistsError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"[중단] v5 device/fresh meter preflight 실패: {error}", file=sys.stderr)
        print(SPEAKER_PREFLIGHT_ABORT_NOTICE, file=sys.stderr, flush=True)
        return 2

    print(
        "[v5 실기 예정] 먼저 입력 전용 1.5초(출력 0초, settle 뒤 1.0초 분석), "
        "그 다음 ch0 소음 스피커와 ch1 상쇄 스피커를 사용하는 exact duplex "
        f"{len(submitted) / 48_000.0:.6f}초를 한 번 실행합니다. "
        f"duplex hard-max는 {len(submitted) / 48_000.0 + LIVE_WATCHDOG_GRACE_SECONDS:.6f}초입니다.\n"
        "두 출력에는 서로 분리된 primary/secondary near-white slot과 저레벨 pilot이 "
        "포함됩니다. 사용자 입회·시작 전 볼륨 최저·같은 앰프 설정 확인이 결속됐습니다.\n"
        f"raw target: {_repository_relative(args.raw_target, label='raw target')}",
        flush=True,
    )

    capture: Any = None
    published: dict[str, Any] | None = None
    post_receipt: dict[str, Any] | None = None
    speaker_notice_printed = False
    raw_relative = _repository_relative(args.raw_target, label="raw target")
    receipt_relative = external_post_receipt_relative_path(raw_relative)
    try:
        with repository_audio_lock(REPO_ROOT, purpose=LIVE_AUDIO_LOCK_PURPOSE) as audio_lock:
            lock_identity = audio_lock_identity_sha256(audio_lock)
            validate_held_audio_lock(
                REPO_ROOT, audio_lock, expected_purpose=LIVE_AUDIO_LOCK_PURPOSE
            )
            preflight_raw, public_report = capture_measurement_preflight_raw(
                sd, static["hardware_audio"], seconds=1.5, settle_seconds=0.5
            )
            preflight_report, preflight_binding = _preflight_report_and_binding(
                raw=preflight_raw,
                report=public_report,
                hardware_identity_sha256=static["hardware"]["identity_sha256"],
                resolved_input_device=devices["input"],
            )
            bindings = collect_actual_external_bindings_v5(
                repository_root=REPO_ROOT,
                plan_envelope_path=_repository_relative(args.plan_envelope, label="plan envelope"),
                live_authority_path=_repository_relative(args.live_authority, label="live authority"),
                meter_raw_path=_repository_relative(args.meter_raw, label="meter raw"),
                level_evidence_path=_repository_relative(args.level_evidence, label="level evidence"),
                hardware_path=_repository_relative(args.hardware, label="hardware"),
                preflight_binding=preflight_binding,
                require_meter_fresh=True,
                require_sealed_raw_fresh=True,
            )
            if bindings["hardware"]["resolved_devices"] != devices:
                raise ValueError("pre-open external binding device가 current device와 다릅니다")

            def pre_open_check() -> None:
                # capture primitive가 모든 large buffer를 만든 뒤, Stream open 직전에 실행된다.
                assert_repository_target_fresh_nofollow(REPO_ROOT, raw_relative)
                assert_repository_target_fresh_nofollow(REPO_ROOT, receipt_relative)
                validate_held_audio_lock(
                    REPO_ROOT, audio_lock, expected_purpose=LIVE_AUDIO_LOCK_PURPOSE
                )
                assert_live_pcm_clock_preconditions(static["hardware_audio"])
                refreshed_fingerprint = collect_alsa_physical_fingerprint(
                    static["hardware_config"]
                )
                if refreshed_fingerprint != static["physical_fingerprint"]:
                    raise RuntimeError("Stream open 직전 ALSA physical fingerprint가 변경됐습니다")
                refreshed_devices = _resolve_devices(static)
                if refreshed_devices != devices:
                    raise RuntimeError("Stream open 직전 resolved devices가 변경됐습니다")
                refreshed = collect_actual_external_bindings_v5(
                    repository_root=REPO_ROOT,
                    plan_envelope_path=bindings["signal_plan"]["path"],
                    live_authority_path=bindings["live_capture_authority"]["path"],
                    meter_raw_path=bindings["meter"]["path"],
                    level_evidence_path=bindings["level_evidence"]["path"],
                    hardware_path=bindings["hardware"]["path"],
                    preflight_binding=preflight_binding,
                    require_meter_fresh=True,
                    require_sealed_raw_fresh=True,
                )
                if refreshed != bindings:
                    raise RuntimeError("Stream open 직전 external bytes/fingerprint/device가 변경됐습니다")

            started_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
            def output_closed_notice(confirmed: bool) -> None:
                nonlocal speaker_notice_printed
                print(
                    SPEAKER_DISCONNECT_NOTICE
                    if confirmed
                    else SPEAKER_STOP_UNCONFIRMED_NOTICE,
                    flush=True,
                )
                speaker_notice_printed = True

            try:
                try:
                    capture = capture_duplex_v5(
                        sd,
                        submitted_pcm=submitted,
                        input_device=devices["input"],
                        output_device=devices["output"],
                        pre_open_check=pre_open_check,
                        watchdog_grace_seconds=LIVE_WATCHDOG_GRACE_SECONDS,
                        on_output_closed=output_closed_notice,
                    )
                except DuplexCaptureFailure as failure:
                    capture = failure
            finally:
                if not speaker_notice_printed:
                    print(SPEAKER_STOP_UNCONFIRMED_NOTICE, flush=True)
                    speaker_notice_printed = True

            # primitive close callback 또는 finally notice보다 앞에 postcheck/저장을 두지 않는다.
            telemetry = capture.telemetry if isinstance(capture, DuplexCaptureFailure) else capture[1]
            completed_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()

            try:
                refreshed = collect_actual_external_bindings_v5(
                    repository_root=REPO_ROOT,
                    plan_envelope_path=bindings["signal_plan"]["path"],
                    live_authority_path=bindings["live_capture_authority"]["path"],
                    meter_raw_path=bindings["meter"]["path"],
                    level_evidence_path=bindings["level_evidence"]["path"],
                    hardware_path=bindings["hardware"]["path"],
                    preflight_binding=preflight_binding,
                    require_meter_fresh=True,
                    require_sealed_raw_fresh=True,
                )
                validate_held_audio_lock(
                    REPO_ROOT, audio_lock, expected_purpose=LIVE_AUDIO_LOCK_PURPOSE
                )
                post_valid = refreshed == bindings and _resolve_devices(static) == devices
                post_error = None if post_valid else "post_capture_binding_changed"
                raw_fresh = True
            except (FileNotFoundError, FileExistsError, KeyError, OSError, RuntimeError, ValueError) as error:
                post_valid = False
                post_error = f"{type(error).__name__}: {error}"
                try:
                    assert_repository_target_fresh_nofollow(REPO_ROOT, raw_relative)
                    raw_fresh = True
                except (FileExistsError, OSError, RuntimeError, ValueError):
                    raw_fresh = False
            primitive_post = _post_capture_binding_from_bindings(
                bindings,
                audio_lock_identity_sha256=lock_identity,
                valid=post_valid,
                error=post_error,
                raw_target_fresh=raw_fresh,
            )
            published = publish_live_raw_v5(
                args.raw_target,
                repository_root=REPO_ROOT,
                planned_submitted_pcm=submitted,
                capture=capture,
                preflight_raw_int32=preflight_raw,
                preflight_report=preflight_report,
                session={
                    "schema": "fullband_causal_v5_live_session_v1",
                    "capture_id": secrets.token_hex(16),
                    "started_at_utc": started_at_utc,
                    "completed_at_utc": completed_at_utc,
                    "audio_lock_identity_sha256": lock_identity,
                    **execution_identity,
                },
                bindings=bindings,
                operator_confirmations=confirmations,
                post_capture_binding=primitive_post,
            )
            try:
                post_receipt = issue_external_post_capture_receipt_v5(
                    repository_root=REPO_ROOT,
                    raw_relative_path=raw_relative,
                    expected_raw_file_sha256=published["raw_file_sha256"],
                    plan_envelope_path=bindings["signal_plan"]["path"],
                    live_authority_path=bindings["live_capture_authority"]["path"],
                    meter_raw_path=bindings["meter"]["path"],
                    level_evidence_path=bindings["level_evidence"]["path"],
                    hardware_path=bindings["hardware"]["path"],
                    audio_lock=audio_lock,
                    expected_resolved_devices=devices,
                )
            except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as post_error:
                # raw는 이미 durable하다. 외부 결속 실패를 숨기거나 재측정하지 않고,
                # raw prebinding+SHA에만 묶인 영구 INVALID receipt를 시도한다.
                post_receipt = issue_invalid_external_post_capture_receipt_v5(
                    repository_root=REPO_ROOT,
                    raw_relative_path=raw_relative,
                    expected_raw_file_sha256=published["raw_file_sha256"],
                    audio_lock=audio_lock,
                    expected_resolved_devices=devices,
                    errors=[
                        "external_post_capture_validation_failed:"
                        f"{type(post_error).__name__}:{post_error}"
                    ],
                )
    except (FileNotFoundError, FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        if not speaker_notice_printed:
            print(SPEAKER_PREFLIGHT_ABORT_NOTICE, file=sys.stderr, flush=True)
        print(f"[실패] v5 live raw/post receipt 보존 실패: {error}", file=sys.stderr)
        if published is not None:
            print(
                f"[보존] immutable raw: {published['path']} | "
                f"SHA256 {published['raw_file_sha256']}",
                file=sys.stderr,
            )
            print(
                "[차단] durable external post receipt가 없습니다. "
                "이 raw는 offline analysis admission이 불가능하며 재측정하지 않습니다.",
                file=sys.stderr,
            )
        return 1

    assert published is not None and post_receipt is not None
    metadata = published["metadata"]
    print(f"[보존] raw {raw_relative} | SHA256 {published['raw_file_sha256']}")
    print(
        f"[보존] external post receipt {post_receipt['relative_path']} | "
        f"SHA256 {post_receipt['file_sha256']}"
    )
    if metadata["status"] != "CAPTURE_PASS" or post_receipt["receipt"]["valid"] is not True:
        print(
            "[INVALID] 캡처/외부 결속 실패 raw는 보존했지만 offline delay analysis를 열지 않습니다.",
            file=sys.stderr,
        )
        return 1
    command = _offline_command(
        args=args,
        receipt_relative_path=post_receipt["relative_path"],
        receipt_file_sha256=post_receipt["file_sha256"],
        capture_id=metadata["session"]["capture_id"],
    )
    print("[CAPTURE_PASS] raw와 external receipt fsync 완료. 분석은 아직 실행하지 않았습니다.")
    print("offline 분석 명령:\n" + command)
    return 0


def _offline_analyze(args: argparse.Namespace) -> int:
    """external receipt admission 뒤 committed live-delay core만 실행한다."""

    from deep_anc.dsp.fullband_live_authority_v5 import (
        committed_plan_envelope_v5,
    )
    from deep_anc.dsp.fullband_live_delay_core import analyze_committed_v5_live_delay
    from deep_anc.dsp.fullband_live_post_v5 import (
        load_external_post_capture_receipt_v5,
        publish_live_delay_analysis_v5,
    )

    admitted = load_external_post_capture_receipt_v5(
        repository_root=REPO_ROOT,
        receipt_relative_path=_repository_relative(args.post_receipt, label="post receipt"),
        expected_receipt_file_sha256=args.expected_post_receipt_sha256,
        plan_envelope_path=_repository_relative(args.plan_envelope, label="plan envelope"),
        live_authority_path=_repository_relative(args.live_authority, label="live authority"),
        meter_raw_path=_repository_relative(args.meter_raw, label="meter raw"),
        level_evidence_path=_repository_relative(args.level_evidence, label="level evidence"),
        hardware_path=_repository_relative(args.hardware, label="hardware"),
    )
    raw = admitted["raw"]
    if _repository_relative(args.raw_target, label="raw target") != raw["metadata"]["bindings"]["signal_plan"]["raw_session_relative_path"]:
        raise ValueError("offline --raw-target이 external receipt sealed raw와 다릅니다")
    # External receipt admission이 actual plan bytes/SHA를 이미 dirfd 기준으로
    # 검증했다. 분석 직전에 pathname을 다시 열지 않고 같은 pinned deterministic
    # builder의 in-memory envelope를 사용해 TOCTOU 재도입을 막는다.
    plan_envelope = committed_plan_envelope_v5()
    arrays = raw["arrays"]
    metadata = raw["metadata"]
    telemetry = {
        **metadata["duplex_telemetry_scalars"],
        **{
            name: arrays[name]
            for name in (
                "callback_sequence",
                "callback_start_frames",
                "callback_frame_counts",
                "input_buffer_adc_time",
                "output_buffer_dac_time",
                "callback_current_time",
                "callback_status_bitmask",
            )
        },
        "actual_submitted_pcm": arrays["actual_submitted_pcm"],
        "capture_valid_mask": arrays["capture_valid_mask"],
        "submitted_valid_mask": arrays["submitted_valid_mask"],
    }
    analysis, operator = analyze_committed_v5_live_delay(
        plan=plan_envelope["signal_plan"],
        submitted_pcm=arrays["actual_submitted_pcm"],
        captured_adc_pcm=arrays["captured_pcm"],
        duplex_telemetry=telemetry,
    )
    published = publish_live_delay_analysis_v5(
        repository_root=REPO_ROOT,
        output_directory_relative_path=_repository_relative(
            args.analysis_output, label="analysis output"
        ),
        external_receipt_file_sha256=admitted["receipt_file_sha256"],
        analysis=analysis,
        operator=operator,
    )
    print(
        "[OFFLINE_MATH_PASS] actual raw/external receipt에 결속된 분석 발행 완료; "
        "canonical_training=false; hardware_sample_slip_authority=false"
    )
    print(
        f"analysis {published['analysis']['relative_path']} | "
        f"SHA256 {published['analysis']['file_sha256']}"
    )
    print(
        f"operator {published['operator']['relative_path']} | "
        f"SHA256 {published['operator']['file_sha256']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-live", action="store_true")
    mode.add_argument("--offline-analyze", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--raw-session-relative-path",
        default="results/fullband_causal_v5/raw_capture.npz",
    )
    parser.add_argument(
        "--plan-envelope",
        default="assets/contracts/fullband_causal_v5_signal_plan.json",
    )
    parser.add_argument(
        "--live-authority",
        default="assets/contracts/fullband_causal_v5_live_capture_authority.json",
    )
    parser.add_argument("--meter-raw")
    parser.add_argument(
        "--level-evidence",
        default="assets/measured/measurement_level_evidence.json",
    )
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument(
        "--raw-target", default="results/fullband_causal_v5/raw_capture.npz"
    )
    parser.add_argument("--post-receipt")
    parser.add_argument("--expected-post-receipt-sha256")
    parser.add_argument("--analysis-output")
    parser.add_argument("--confirm-speaker", action="store_true")
    parser.add_argument("--confirm-user-present", action="store_true")
    parser.add_argument("--confirm-volume-minimum", action="store_true")
    parser.add_argument("--confirm-routing-and-geometry", action="store_true")
    parser.add_argument("--confirm-same-amplifier-setting", action="store_true")
    args = parser.parse_args(argv)

    if args.execute_live:
        if args.meter_raw is None:
            print("[중단] --execute-live에는 fresh --meter-raw가 필요합니다", file=sys.stderr)
            return 2
        try:
            _assert_live_capture_authority_enabled_before_execute(args)
        except (
            AssertionError,
            FileNotFoundError,
            FileExistsError,
            KeyError,
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            print(f"[중단] v5 live authority gate 실패: {error}", file=sys.stderr)
            print(SPEAKER_PREFLIGHT_ABORT_NOTICE, file=sys.stderr, flush=True)
            return 2
        return _execute_live(args)

    if args.offline_analyze:
        missing = [
            name
            for name in (
                "meter_raw",
                "post_receipt",
                "expected_post_receipt_sha256",
                "analysis_output",
            )
            if getattr(args, name) is None
        ]
        if missing:
            print(
                f"[중단] --offline-analyze 필수 인자가 없습니다: {missing}",
                file=sys.stderr,
            )
            return 2
        try:
            return _offline_analyze(args)
        except (FileNotFoundError, FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            print(f"[실패] v5 offline analysis 거부: {error}", file=sys.stderr)
            return 2

    try:
        plan, submitted = build_plan_v5(
            raw_session_relative_path=args.raw_session_relative_path
        )
        condition = exact_condition_audit_v5(plan, submitted)
        if args.output is not None:
            _write_json_no_replace(
                {
                    "schema": "fullband_causal_signal_plan_envelope_v5",
                    "signal_plan": plan,
                    "support_1024_condition_receipt": condition,
                },
                args.output,
            )
    except (FileExistsError, OSError, ValueError) as error:
        print(f"[실패] v5 signal-only 계획 발행 거부: {error}", file=sys.stderr)
        return 2
    print(
        "[PASS] v5 signal-only | "
        f"{plan['duration_seconds']:.3f}s | peak PCM {plan['actual_submitted_peak_pcm']} | "
        f"support1024 condition {condition['joint_fit_condition_number']:.6f}"
    )
    print(f"[SHA] plan {plan['canonical_payload_sha256']}")
    print(f"[SHA] PCM  {plan['actual_submitted_pcm_sha256']}")
    print("[잠금] 실제 출력 0회; live authority=None; canonical training=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
