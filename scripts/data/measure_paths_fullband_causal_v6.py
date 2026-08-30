#!/usr/bin/env python3
"""시간 분리 clock checkpoint v6의 단일 실측·오프라인 분석 어댑터.

실기 모드는 clean exact checkout, sealed v6 plan/authority, fresh 20초 meter,
현재 ALSA fingerprint와 입력 preflight를 모두 통과한 뒤에만 출력 stream을 연다.
캡처가 끝나면 분석보다 먼저 출력 close를 알리고 raw와 외부 receipt를 immutable하게
발행한다. 오프라인 모드는 그 receipt를 다시 검증한 뒤에만 clock/P/S LS를 실행한다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import secrets
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.repository_fd import repository_execution_identity  # noqa: E402
from deep_anc.dsp.fullband_causal_v6 import (  # noqa: E402
    V6ClockAdmissionError,
    build_plan_v6,
    exact_condition_audit_v6,
)
from deep_anc.dsp.fullband_live_authority_v6 import (  # noqa: E402
    DURATION_SECONDS,
    SEALED_HARDWARE_RELATIVE_PATH,
    SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
    SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
    SEALED_RAW_RELATIVE_PATH,
    TOTAL_FRAMES,
    committed_plan_envelope_v6,
)


FULLBAND_V6_LIVE_ADAPTER_IMPLEMENTED = True
ADAPTER_REPOSITORY_PATH = "scripts/data/measure_paths_fullband_causal_v6.py"
LIVE_AUDIO_LOCK_PURPOSE = "fullband_causal_v6_live_capture"
LIVE_WATCHDOG_GRACE_SECONDS = 2.0
DEFAULT_LEVEL_EVIDENCE = "assets/measured/measurement_level_evidence.json"
SPEAKER_DISCONNECT_NOTICE = (
    "[스피커 출력 종료] v6 오디오 stream stop/abort/close가 확인됐습니다. "
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


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _repository_relative(value: str | Path, *, label: str) -> str:
    root = Path(os.path.abspath(os.fspath(REPO_ROOT)))
    supplied = Path(value)
    lexical = Path(
        os.path.abspath(os.fspath(supplied if supplied.is_absolute() else root / supplied))
    )
    try:
        relative = lexical.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"{label}가 repository 밖입니다") from error
    if not relative or "\\" in relative or any(
        part in {"", ".", ".."} for part in Path(relative).parts
    ):
        raise ValueError(f"{label}가 canonical repository 상대경로가 아닙니다")
    return relative


def _repository_execution_identity() -> dict[str, Any]:
    identity = repository_execution_identity(REPO_ROOT, ADAPTER_REPOSITORY_PATH)
    return {
        "repository_commit": identity["repository_commit"],
        "repository_branch": identity["repository_branch"],
        "repository_dirty": identity["repository_dirty"],
        "adapter_path": identity["script_path"],
        "adapter_file_sha256": identity["script_file_sha256"],
    }


def _confirmations(args: argparse.Namespace) -> dict[str, bool]:
    value = {
        "speaker_output": bool(args.confirm_speaker),
        "user_present": bool(args.confirm_user_present),
        "volume_minimum": bool(args.confirm_volume_minimum),
        "routing_and_geometry": bool(args.confirm_routing_and_geometry),
        "same_amplifier_setting": bool(args.confirm_same_amplifier_setting),
    }
    if set(value) != _CONFIRMATION_KEYS or any(item is not True for item in value.values()):
        raise ValueError("v6 live capture의 다섯 operator confirmation이 모두 필요합니다")
    return value


def _preflight_report_and_binding(
    *,
    raw: np.ndarray,
    report: Mapping[str, Any],
    hardware_identity_sha256: str,
    resolved_input_device: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from deep_anc.dsp.fullband_live_raw_v6 import (
        PREFLIGHT_IDENTITY_SCHEMA,
        PREFLIGHT_REPORT_SCHEMA,
    )

    owned = np.array(raw, dtype="<i4", copy=True, order="C")
    if owned.ndim != 2 or owned.shape[0] <= 0 or owned.shape[1] != 2:
        raise ValueError("input-only preflight raw는 nonempty exact <i4 [frames,2]여야 합니다")
    if report.get("passed") is not True:
        raise RuntimeError("input-only ERR/REF preflight가 PASS가 아닙니다")
    if report.get("resolved_input_device") != resolved_input_device:
        raise ValueError("preflight input device가 v6 capture device와 다릅니다")
    if report.get("sample_rate_hz") != 48_000 or report.get("frames") != len(owned):
        raise ValueError("preflight sample rate/frame 계약이 다릅니다")
    channels = report.get("channels")
    if not isinstance(channels, list) or len(channels) != 2:
        raise ValueError("preflight ERR/REF channel summary가 필요합니다")
    identity_core = {
        "schema": PREFLIGHT_IDENTITY_SCHEMA,
        "raw_sha256": _array_sha256(owned),
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
        "raw_sha256": _array_sha256(owned),
        "report_sha256": _payload_sha256(canonical_report),
        "identity_sha256": identity_sha,
        "passed": True,
    }
    return canonical_report, binding


def _static_contract_before_backend_import(args: argparse.Namespace) -> dict[str, Any]:
    """PortAudio import 전에 모든 v6 byte/path/clock/CPU gate를 닫는다."""

    from deep_anc.dsp.fullband_live_post_v6 import (
        assert_repository_target_fresh_nofollow,
        external_post_receipt_relative_path,
    )
    from deep_anc.dsp.fullband_v6_meter import (
        validate_fullband_v6_meter_raw_static,
        validate_fullband_v6_static_contract,
    )
    from deep_anc.dsp.measurement_level import assert_live_pcm_clock_preconditions

    exact_paths = {
        "plan envelope": (args.plan_envelope, SEALED_PLAN_ENVELOPE_RELATIVE_PATH),
        "live authority": (args.live_authority, SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH),
        "level evidence": (args.level_evidence, DEFAULT_LEVEL_EVIDENCE),
        "hardware": (args.hardware, SEALED_HARDWARE_RELATIVE_PATH),
        "raw target": (args.raw_target, SEALED_RAW_RELATIVE_PATH),
    }
    for label, (supplied, expected) in exact_paths.items():
        if _repository_relative(supplied, label=label) != expected:
            raise ValueError(f"{label}가 v6 sealed path와 다릅니다")
    raw_relative = SEALED_RAW_RELATIVE_PATH
    receipt_relative = external_post_receipt_relative_path(raw_relative)
    assert_repository_target_fresh_nofollow(REPO_ROOT, raw_relative)
    assert_repository_target_fresh_nofollow(REPO_ROOT, receipt_relative)
    static = validate_fullband_v6_static_contract(
        repository_root=REPO_ROOT,
        plan_envelope_path=args.plan_envelope,
        live_authority_path=args.live_authority,
        level_evidence_path=args.level_evidence,
        hardware_path=args.hardware,
        raw_target_path=args.raw_target,
        require_sealed_raw_fresh=True,
    )
    meter = validate_fullband_v6_meter_raw_static(
        args.meter_raw,
        repository_root=REPO_ROOT,
        now_utc=None,
        require_fresh=True,
        require_sealed_raw_fresh=True,
    )
    hardware = static.get("hardware_audio")
    if not isinstance(hardware, Mapping):
        raise RuntimeError("v6 static contract에 hardware_audio가 없습니다")
    assert_live_pcm_clock_preconditions(dict(hardware))
    return {
        **dict(static),
        "exact_plan": committed_plan_envelope_v6(),
        "prevalidated_meter": meter,
    }


def _resolve_devices(static: Mapping[str, Any]) -> dict[str, int]:
    from deep_anc.audio_io import resolve_alsa_portaudio_device

    hardware = static["hardware_audio"]
    devices = {
        "input": resolve_alsa_portaudio_device(
            hardware["input"]["card"], hardware["input"]["pcm"], "input", 2
        ),
        "output": resolve_alsa_portaudio_device(
            hardware["output"]["card"], hardware["output"]["pcm"], "output", 2
        ),
    }
    if any(type(value) is not int or value < 0 for value in devices.values()):
        raise ValueError("v6 resolved devices는 음이 아닌 exact int여야 합니다")
    return devices


def _post_capture_binding(
    bindings: Mapping[str, Any],
    *,
    audio_lock_identity_sha256: str,
    valid: bool,
    error: str | None,
    raw_target_fresh: bool,
) -> dict[str, Any]:
    return {
        "schema": "fullband_causal_v6_post_capture_binding_v1",
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
    *, args: argparse.Namespace, receipt_relative_path: str,
    receipt_file_sha256: str, capture_id: str,
) -> str:
    parts = [
        str(Path(sys.executable).absolute()),
        str(Path(__file__).resolve(strict=True)),
        "--offline-analyze",
        "--plan-envelope", _repository_relative(args.plan_envelope, label="plan envelope"),
        "--live-authority", _repository_relative(args.live_authority, label="live authority"),
        "--meter-raw", _repository_relative(args.meter_raw, label="meter raw"),
        "--level-evidence", _repository_relative(args.level_evidence, label="level evidence"),
        "--hardware", _repository_relative(args.hardware, label="hardware"),
        "--raw-target", _repository_relative(args.raw_target, label="raw target"),
        "--post-receipt", receipt_relative_path,
        "--expected-post-receipt-sha256", receipt_file_sha256,
        "--analysis-output", f"results/fullband_causal_v6/analysis_{capture_id}",
        "--failure-output", f"results/fullband_causal_v6/failure_{capture_id}.json",
    ]
    return " ".join(shlex.quote(value) for value in parts)


def _execute_live(args: argparse.Namespace) -> int:
    """입력 preflight 뒤 exact 24.576초 v6 duplex를 한 번만 수행한다."""

    from deep_anc.audio_duplex_v6 import DuplexCaptureFailure, capture_duplex_v6
    from deep_anc.audio_io import capture_measurement_preflight_raw
    from deep_anc.dsp.fullband_live_post_v6 import (
        assert_repository_target_fresh_nofollow,
        audio_lock_identity_sha256,
        collect_actual_external_bindings_v6,
        external_post_receipt_relative_path,
        issue_external_post_capture_receipt_v6,
        issue_invalid_external_post_capture_receipt_v6,
        validate_held_audio_lock,
    )
    from deep_anc.dsp.fullband_live_raw_v6 import publish_live_raw_v6
    from deep_anc.dsp.fullband_v6_meter import validate_fullband_v6_meter_raw
    from deep_anc.dsp.measurement_level import (
        assert_live_pcm_clock_preconditions,
        collect_alsa_physical_fingerprint,
        repository_audio_lock,
    )

    try:
        confirmations = _confirmations(args)
        execution_identity = _repository_execution_identity()
        static = _static_contract_before_backend_import(args)
        plan, submitted = build_plan_v6(raw_session_relative_path=SEALED_RAW_RELATIVE_PATH)
        if plan != static["exact_plan"]["signal_plan"]:
            raise ValueError("v6 committed builder plan과 sealed envelope가 다릅니다")
        submitted = np.array(submitted, dtype="<i2", copy=True, order="C")
        if submitted.shape != (TOTAL_FRAMES, 2) or len(submitted) / 48_000 != DURATION_SECONDS:
            raise ValueError("v6 exact frame/duration 계약이 다릅니다")
    except (FileNotFoundError, FileExistsError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"[중단] v6 backend import 전 preflight 실패: {error}", file=sys.stderr)
        print(SPEAKER_PREFLIGHT_ABORT_NOTICE, file=sys.stderr, flush=True)
        return 2

    try:
        sd = importlib.import_module("sounddevice")
        devices = _resolve_devices(static)
        meter = validate_fullband_v6_meter_raw(
            args.meter_raw,
            repository_root=REPO_ROOT,
            now_utc=None,
            require_fresh=True,
            require_sealed_raw_fresh=True,
        )
        if meter != static["prevalidated_meter"]:
            raise ValueError("backend import 전후 v6 meter bytes/followup이 변경됐습니다")
        if meter["hardware"]["resolved_devices"] != devices:
            raise ValueError("fresh v6 meter device가 current PortAudio device와 다릅니다")
    except (FileNotFoundError, FileExistsError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"[중단] v6 device/fresh meter preflight 실패: {error}", file=sys.stderr)
        print(SPEAKER_PREFLIGHT_ABORT_NOTICE, file=sys.stderr, flush=True)
        return 2

    print(
        "[v6 실기 예정] 입력 전용 1.5초(출력 0초) 뒤 "
        f"ch0 primary/noise와 ch1 secondary/control을 순차 사용하는 duplex {DURATION_SECONDS:.3f}초를 "
        f"한 번 실행합니다. hard-max={DURATION_SECONDS + LIVE_WATCHDOG_GRACE_SECONDS:.3f}초.\n"
        "clock checkpoint와 near-white PE는 시간 분리되어 있고, 두 출력은 동시에 활성화되지 않습니다.\n"
        f"raw target: {SEALED_RAW_RELATIVE_PATH}",
        flush=True,
    )

    capture: Any = None
    published: dict[str, Any] | None = None
    post_receipt: dict[str, Any] | None = None
    speaker_notice_printed = False
    receipt_relative = external_post_receipt_relative_path(SEALED_RAW_RELATIVE_PATH)
    try:
        with repository_audio_lock(REPO_ROOT, purpose=LIVE_AUDIO_LOCK_PURPOSE) as audio_lock:
            lock_identity = audio_lock_identity_sha256(audio_lock)
            validate_held_audio_lock(REPO_ROOT, audio_lock, expected_purpose=LIVE_AUDIO_LOCK_PURPOSE)
            preflight_raw, public_report = capture_measurement_preflight_raw(
                sd, static["hardware_audio"], seconds=1.5, settle_seconds=0.5
            )
            preflight_report, preflight_binding = _preflight_report_and_binding(
                raw=preflight_raw,
                report=public_report,
                hardware_identity_sha256=static["hardware"]["identity_sha256"],
                resolved_input_device=devices["input"],
            )
            bindings = collect_actual_external_bindings_v6(
                repository_root=REPO_ROOT,
                plan_envelope_path=SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
                live_authority_path=SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
                meter_raw_path=_repository_relative(args.meter_raw, label="meter raw"),
                level_evidence_path=DEFAULT_LEVEL_EVIDENCE,
                hardware_path=SEALED_HARDWARE_RELATIVE_PATH,
                preflight_binding=preflight_binding,
                require_meter_fresh=True,
                require_sealed_raw_fresh=True,
            )
            if bindings["hardware"]["resolved_devices"] != devices:
                raise ValueError("v6 pre-open external binding device가 current device와 다릅니다")

            def pre_open_check() -> None:
                assert_repository_target_fresh_nofollow(REPO_ROOT, SEALED_RAW_RELATIVE_PATH)
                assert_repository_target_fresh_nofollow(REPO_ROOT, receipt_relative)
                validate_held_audio_lock(REPO_ROOT, audio_lock, expected_purpose=LIVE_AUDIO_LOCK_PURPOSE)
                assert_live_pcm_clock_preconditions(static["hardware_audio"])
                fingerprint = collect_alsa_physical_fingerprint(static["hardware_config"])
                if fingerprint != static["physical_fingerprint"]:
                    raise RuntimeError("v6 Stream open 직전 ALSA fingerprint가 변경됐습니다")
                if _resolve_devices(static) != devices:
                    raise RuntimeError("v6 Stream open 직전 device mapping이 변경됐습니다")
                refreshed = collect_actual_external_bindings_v6(
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
                    raise RuntimeError("v6 Stream open 직전 external binding이 변경됐습니다")

            def output_closed_notice(confirmed: bool) -> None:
                nonlocal speaker_notice_printed
                print(
                    SPEAKER_DISCONNECT_NOTICE if confirmed else SPEAKER_STOP_UNCONFIRMED_NOTICE,
                    flush=True,
                )
                speaker_notice_printed = True

            started_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
            try:
                try:
                    capture = capture_duplex_v6(
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

            completed_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
            try:
                refreshed = collect_actual_external_bindings_v6(
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
                validate_held_audio_lock(REPO_ROOT, audio_lock, expected_purpose=LIVE_AUDIO_LOCK_PURPOSE)
                post_valid = refreshed == bindings and _resolve_devices(static) == devices
                post_error = None if post_valid else "post_capture_binding_changed"
                raw_fresh = True
            except (FileNotFoundError, FileExistsError, KeyError, OSError, RuntimeError, ValueError) as error:
                post_valid = False
                post_error = f"{type(error).__name__}: {error}"
                try:
                    assert_repository_target_fresh_nofollow(REPO_ROOT, SEALED_RAW_RELATIVE_PATH)
                    raw_fresh = True
                except (FileExistsError, OSError, RuntimeError, ValueError):
                    raw_fresh = False
            primitive_post = _post_capture_binding(
                bindings,
                audio_lock_identity_sha256=lock_identity,
                valid=post_valid,
                error=post_error,
                raw_target_fresh=raw_fresh,
            )
            published = publish_live_raw_v6(
                args.raw_target,
                repository_root=REPO_ROOT,
                planned_submitted_pcm=submitted,
                capture=capture,
                preflight_raw_int32=preflight_raw,
                preflight_report=preflight_report,
                session={
                    "schema": "fullband_causal_v6_live_session_v1",
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
                post_receipt = issue_external_post_capture_receipt_v6(
                    repository_root=REPO_ROOT,
                    raw_relative_path=SEALED_RAW_RELATIVE_PATH,
                    expected_raw_file_sha256=published["raw_file_sha256"],
                    plan_envelope_path=bindings["signal_plan"]["path"],
                    live_authority_path=bindings["live_capture_authority"]["path"],
                    meter_raw_path=bindings["meter"]["path"],
                    level_evidence_path=bindings["level_evidence"]["path"],
                    hardware_path=bindings["hardware"]["path"],
                    audio_lock=audio_lock,
                    expected_resolved_devices=devices,
                )
            except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                post_receipt = issue_invalid_external_post_capture_receipt_v6(
                    repository_root=REPO_ROOT,
                    raw_relative_path=SEALED_RAW_RELATIVE_PATH,
                    expected_raw_file_sha256=published["raw_file_sha256"],
                    audio_lock=audio_lock,
                    expected_resolved_devices=devices,
                    errors=[f"external_post_capture_validation_failed:{type(error).__name__}:{error}"],
                )
    except (FileNotFoundError, FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        if not speaker_notice_printed:
            print(SPEAKER_PREFLIGHT_ABORT_NOTICE, file=sys.stderr, flush=True)
        print(f"[실패] v6 live raw/post receipt 보존 실패: {error}", file=sys.stderr)
        if published is not None:
            print(
                f"[보존] immutable raw: {published['path']} | SHA256 {published['raw_file_sha256']}",
                file=sys.stderr,
            )
        return 1

    assert published is not None and post_receipt is not None
    metadata = published["metadata"]
    print(f"[보존] raw {SEALED_RAW_RELATIVE_PATH} | SHA256 {published['raw_file_sha256']}")
    print(f"[보존] external receipt {post_receipt['relative_path']} | SHA256 {post_receipt['file_sha256']}")
    if metadata["status"] != "CAPTURE_PASS" or post_receipt["receipt"]["valid"] is not True:
        print("[INVALID] raw는 보존했지만 offline delay analysis admission은 닫혔습니다.", file=sys.stderr)
        return 1
    command = _offline_command(
        args=args,
        receipt_relative_path=post_receipt["relative_path"],
        receipt_file_sha256=post_receipt["file_sha256"],
        capture_id=metadata["session"]["capture_id"],
    )
    print("[CAPTURE_PASS] raw와 receipt fsync 완료. 분석은 아직 실행하지 않았습니다.")
    print("offline 분석 명령:\n" + command)
    return 0


def _offline_analyze(args: argparse.Namespace) -> int:
    from deep_anc.dsp.fullband_live_delay_core_v6 import (
        analyze_committed_v6_live_delay,
        validate_committed_v6_plan_and_derive_windows,
        validate_duplex_telemetry_v6,
    )
    from deep_anc.dsp.fullband_live_post_v6 import (
        load_external_post_capture_receipt_v6,
        publish_live_delay_analysis_v6,
        publish_live_delay_failure_v6,
    )

    # Offline math도 live capture와 같은 clean exact adapter blob에서만 허용한다.
    # 이 gate는 receipt/raw를 읽거나 core를 호출하기 전에 실행한다.
    execution_identity = _repository_execution_identity()
    admitted = load_external_post_capture_receipt_v6(
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
    metadata = raw["metadata"]
    session = metadata.get("session")
    expected_execution = {
        "repository_commit": execution_identity["repository_commit"],
        "repository_branch": execution_identity["repository_branch"],
        "repository_dirty": False,
        "adapter_path": execution_identity["adapter_path"],
        "adapter_file_sha256": execution_identity["adapter_file_sha256"],
    }
    if not isinstance(session, Mapping) or any(
        session.get(key) != value for key, value in expected_execution.items()
    ):
        raise ValueError(
            "offline checkout/adapter identity가 capture raw session과 exact 일치하지 않습니다"
        )
    if _repository_relative(args.raw_target, label="raw target") != SEALED_RAW_RELATIVE_PATH:
        raise ValueError("offline raw target이 v6 sealed path와 다릅니다")
    arrays = raw["arrays"]
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
    plan = committed_plan_envelope_v6()["signal_plan"]

    def publish_failure(
        *, stage: str, optimizer_started: bool, error: BaseException,
        available: Mapping[str, Any] | None = None,
    ) -> int:
        failure = publish_live_delay_failure_v6(
            repository_root=REPO_ROOT,
            failure_relative_path=_repository_relative(args.failure_output, label="failure output"),
            raw_relative_path=SEALED_RAW_RELATIVE_PATH,
            raw_file_sha256=raw["raw_file_sha256"],
            external_receipt_relative_path=_repository_relative(args.post_receipt, label="post receipt"),
            external_receipt_file_sha256=admitted["receipt_file_sha256"],
            failure_stage=stage,
            optimizer_started=optimizer_started,
            error=f"{type(error).__name__}: {error}",
            available_snr_receipt=available,
        )
        print(
            f"[OFFLINE_FAIL] immutable failure {failure['path']} | SHA256 {failure['file_sha256']}",
            file=sys.stderr,
        )
        return 1

    try:
        validate_committed_v6_plan_and_derive_windows(
            plan, arrays["actual_submitted_pcm"]
        )
        validate_duplex_telemetry_v6(
            telemetry,
            captured_adc_pcm=arrays["captured_pcm"],
            expected_submitted_pcm=arrays["actual_submitted_pcm"],
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        return publish_failure(
            stage="pre_core_validation",
            optimizer_started=False,
            error=error,
        )
    try:
        analysis, operator = analyze_committed_v6_live_delay(
            plan=plan,
            submitted_pcm=arrays["actual_submitted_pcm"],
            captured_adc_pcm=arrays["captured_pcm"],
            duplex_telemetry=telemetry,
        )
    except V6ClockAdmissionError as error:
        return publish_failure(
            stage=error.stage,
            optimizer_started=error.optimizer_started,
            error=error,
            available=error.available_receipt or None,
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        # 위 prevalidation을 통과했고 common-clock 내부 expected 실패는 모두
        # V6ClockAdmissionError다. 따라서 이 경로는 clock optimizer 이후 P/S,
        # shifted condition, subband 또는 holdout 단계의 실패다.
        return publish_failure(
            stage="post_clock_operator_analysis",
            optimizer_started=True,
            error=error,
        )
    try:
        published = publish_live_delay_analysis_v6(
            repository_root=REPO_ROOT,
            output_directory_relative_path=_repository_relative(args.analysis_output, label="analysis output"),
            external_receipt_relative_path=_repository_relative(args.post_receipt, label="post receipt"),
            external_receipt_file_sha256=admitted["receipt_file_sha256"],
            plan_envelope_path=_repository_relative(args.plan_envelope, label="plan envelope"),
            live_authority_path=_repository_relative(args.live_authority, label="live authority"),
            meter_raw_path=_repository_relative(args.meter_raw, label="meter raw"),
            level_evidence_path=_repository_relative(args.level_evidence, label="level evidence"),
            hardware_path=_repository_relative(args.hardware, label="hardware"),
            analysis_execution_identity=expected_execution,
            analysis=analysis,
            operator=operator,
        )
    except (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        return publish_failure(
            stage="analysis_publication",
            optimizer_started=True,
            error=error,
        )
    print("[OFFLINE_MATH_PASS] v6 raw/receipt 결속 분석 발행 완료; canonical training=false")
    print(f"analysis {published['analysis']['relative_path']} | SHA256 {published['analysis']['file_sha256']}")
    print(f"operator {published['operator']['relative_path']} | SHA256 {published['operator']['file_sha256']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-live", action="store_true")
    mode.add_argument("--offline-analyze", action="store_true")
    parser.add_argument("--plan-envelope", default=SEALED_PLAN_ENVELOPE_RELATIVE_PATH)
    parser.add_argument("--live-authority", default=SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH)
    parser.add_argument("--meter-raw")
    parser.add_argument("--level-evidence", default=DEFAULT_LEVEL_EVIDENCE)
    parser.add_argument("--hardware", default=SEALED_HARDWARE_RELATIVE_PATH)
    parser.add_argument("--raw-target", default=SEALED_RAW_RELATIVE_PATH)
    parser.add_argument("--post-receipt")
    parser.add_argument("--expected-post-receipt-sha256")
    parser.add_argument("--analysis-output")
    parser.add_argument("--failure-output")
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
        return _execute_live(args)
    if args.offline_analyze:
        required = (
            "meter_raw", "post_receipt", "expected_post_receipt_sha256",
            "analysis_output", "failure_output",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            print(f"[중단] --offline-analyze 필수 인자가 없습니다: {missing}", file=sys.stderr)
            return 2
        try:
            return _offline_analyze(args)
        except (FileNotFoundError, FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            print(f"[실패] v6 offline analysis 거부: {error}", file=sys.stderr)
            return 2

    try:
        plan, submitted = build_plan_v6(raw_session_relative_path=SEALED_RAW_RELATIVE_PATH)
        condition = exact_condition_audit_v6(plan, submitted)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[실패] v6 무음 계획 검증 거부: {error}", file=sys.stderr)
        return 2
    print(
        "[PASS] v6 무음 signal-only | "
        f"{plan['duration_seconds']:.3f}s | frames {len(submitted)} | "
        f"peak PCM {plan['actual_submitted_peak_pcm']} | "
        f"support1024 condition {condition['joint_fit_condition_number']:.6f}"
    )
    print(f"[SHA] plan {plan['canonical_payload_sha256']}")
    print(f"[SHA] PCM  {plan['actual_submitted_pcm_sha256']}")
    print("[잠금] 실제 출력 0회; live authority=capture-only; canonical training=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
