#!/usr/bin/env python3
"""Stage-2 2 kHz legacy combined-duplex diagnostic capture.

이 파일의 combined PortAudio callback frame index는 USB DAC와 APE ADC의 hardware
frame identity가 아니다. 보존된 v2 raw는 진단 자료일 뿐 P/S/plant/training authority가
아니다. dry-run만 유지하며 ``--execute-live``는 backend import 전에 닫힌다. USB
output-master 경로도 실제 split-clock failure 뒤 retired forensic-only다. 현행 후보는
RT5640/J511 same-card S32 actual-P/S의 read-only preflight이며, live adapter가 생기기
전에는 그 경로도 출력을 열지 않는다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import io
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from deep_anc.data.repository_fd import (  # noqa: E402
    RepositoryFileGuard,
    assert_repository_target_fresh_nofollow,
    repository_execution_identity,
)
from deep_anc.dsp.measurement_level import (  # noqa: E402
    BOOTSTRAP_METER_MAX_AGE_SECONDS,
    load_measurement_level_evidence,
    measurement_hardware_identity,
    meter_receipt_path,
    validate_bootstrap_meter_raw,
)
from deep_anc.dsp.stage2_2khz_analysis_v2 import (  # noqa: E402
    analyse_stage2_v2_diagnostic_preflight,
)
from deep_anc.dsp.stage2_2khz_live_v2 import (  # noqa: E402
    publish_stage2_v2_partial_raw_no_replace,
    publish_stage2_v2_phase_raw_no_replace,
    seal_and_publish_physical_operating_level_evidence,
    seal_and_publish_diagnostic_authorization,
    snapshot_published_stage2_v2_phase,
    validate_published_diagnostic_authorization,
    validate_published_physical_operating_level_evidence,
)
from deep_anc.dsp.stage2_2khz_measurement_v2 import (  # noqa: E402
    LIVE_SAFE_FALLBACK_STATUS,
    Stage2MeasurementV2Error,
    audit_stage2_v2_live_safe_dpss_gram,
    build_stage2_v2_live_safe_fallback_plan,
)
from deep_anc.train.stage2_2khz_git_authority import (  # noqa: E402
    verify_tracked_head_file,
)


ADAPTER_PATH = "scripts/data/measure_paths_stage2_2khz.py"
DEFAULT_CONFIG = "configs/stage2_2khz_measurement.json"
DEFAULT_HARDWARE = "configs/hardware_jetson.yaml"
LIVE_AUDIO_LOCK_PURPOSE = "stage2_2khz_v2_two_phase_capture"
WATCHDOG_GRACE_SECONDS = 2.0
LEGACY_COMBINED_LIVE_AUTHORITY_DISABLED = True
_REQUIRED_CONFIRMATIONS = {
    "speaker_output",
    "user_present",
    "volume_fixed_after_meter_adjustment",
    "routing_and_geometry",
    "same_amplifier_setting",
}
_CRITICAL_LIVE_FILES = (
    ADAPTER_PATH,
    "src/deep_anc/dsp/stage2_2khz_measurement_v2.py",
    "src/deep_anc/dsp/stage2_2khz_analysis_v2.py",
    "src/deep_anc/dsp/stage2_2khz_live_v2.py",
    "src/deep_anc/dsp/stage2_2khz_level_contract.py",
    "configs/stage2_2khz_measurement.json",
    "configs/stage2_2khz_source_operating_level.json",
    "configs/hardware_jetson.yaml",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--hardware", default=DEFAULT_HARDWARE)
    parser.add_argument("--meter-raw")
    parser.add_argument("--expected-meter-raw-sha256")
    parser.add_argument("--confirm-speaker", action="store_true")
    parser.add_argument("--confirm-user-present", action="store_true")
    parser.add_argument("--confirm-volume-fixed", action="store_true")
    parser.add_argument("--confirm-routing-and-geometry", action="store_true")
    parser.add_argument("--confirm-same-amplifier-setting", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write-plan", action="store_true")
    mode.add_argument("--execute-live", action="store_true")
    return parser


def _confirmations(arguments: argparse.Namespace) -> dict[str, bool]:
    values = {
        "speaker_output": bool(arguments.confirm_speaker),
        "user_present": bool(arguments.confirm_user_present),
        "volume_fixed_after_meter_adjustment": bool(arguments.confirm_volume_fixed),
        "routing_and_geometry": bool(arguments.confirm_routing_and_geometry),
        "same_amplifier_setting": bool(arguments.confirm_same_amplifier_setting),
    }
    if set(values) != _REQUIRED_CONFIRMATIONS or any(value is not True for value in values.values()):
        raise Stage2MeasurementV2Error("Stage-2 live의 다섯 confirmation이 모두 필요합니다")
    return values


def _relative_repository_path(value: str, *, label: str) -> str:
    supplied = Path(value)
    lexical = Path(os.path.abspath(supplied if supplied.is_absolute() else REPOSITORY_ROOT / supplied))
    try:
        relative = lexical.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as exc:
        raise Stage2MeasurementV2Error(f"{label}가 repository 밖입니다") from exc
    if not relative or "\\" in relative or any(part in {"", ".", ".."} for part in Path(relative).parts):
        raise Stage2MeasurementV2Error(f"{label}가 canonical 상대경로가 아닙니다")
    return relative


def _validate_fresh_meter(
    path: str,
    expected_sha256: str,
    *,
    expected_hardware_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in expected_sha256
    ):
        raise Stage2MeasurementV2Error("fresh meter expected SHA-256이 필요합니다")
    relative = _relative_repository_path(path, label="meter raw")
    receipt_relative = _relative_repository_path(
        str(meter_receipt_path(REPOSITORY_ROOT / relative)), label="meter receipt"
    )
    with RepositoryFileGuard(
        REPOSITORY_ROOT, relative, label="Stage-2 fresh meter"
    ) as raw_guard, RepositoryFileGuard(
        REPOSITORY_ROOT, receipt_relative, label="Stage-2 fresh meter receipt"
    ) as receipt_guard:
        if raw_guard.sha256 != expected_sha256:
            raise Stage2MeasurementV2Error("fresh meter file SHA가 expected와 다릅니다")
        verified = validate_bootstrap_meter_raw(
            relative,
            repository_root=REPOSITORY_ROOT,
            expected_hardware_identity=dict(expected_hardware_identity),
            require_fresh=True,
        )
        if verified["sha256"] != raw_guard.sha256:
            raise Stage2MeasurementV2Error("official meter validator/raw guard SHA가 다릅니다")
        raw_guard.verify()
        receipt_guard.verify()
    metadata = verified["metadata"]
    # official validator가 raw recipe/device를 검사한 뒤에도 calibration evidence가 실제
    # tracked evidence bytes와 같은지 별도로 닫는다.
    level_evidence = load_measurement_level_evidence(
        "assets/measured/measurement_level_evidence.json",
        repository_root=REPOSITORY_ROOT,
    )
    evidence = metadata.get("calibration_evidence")
    if (
        not isinstance(evidence, Mapping)
        or Path(str(evidence.get("path", ""))).resolve()
        != Path(str(level_evidence["_evidence_path"])).resolve()
        or evidence.get("sha256") != level_evidence["_evidence_sha256"]
        or level_evidence.get("hardware_identity") != dict(expected_hardware_identity)
    ):
        raise Stage2MeasurementV2Error("meter calibration evidence bytes/hardware binding이 다릅니다")
    completed = verified["completed_at_utc"]
    age = (dt.datetime.now(dt.timezone.utc) - completed).total_seconds()
    return {
        "path": relative,
        "sha256": expected_sha256,
        "receipt_path": receipt_relative,
        "receipt_sha256": receipt_guard.sha256,
        "capture_id": metadata["capture_id"],
        "completed_at_utc": metadata["completed_at_utc"],
        "age_seconds": age,
        "meter_ch0_dbfs": verified["meter_ch0_dbfs"],
        "freshness_max_seconds": BOOTSTRAP_METER_MAX_AGE_SECONDS,
        "resolved_devices": dict(metadata["resolved_devices"]),
        "physical_fingerprint": metadata["hardware_identity"]["physical_fingerprint"],
        "hardware_identity": dict(expected_hardware_identity),
        "calibration_evidence": {
            "path": _relative_repository_path(
                str(level_evidence["_evidence_path"]), label="calibration evidence"
            ),
            "sha256": level_evidence["_evidence_sha256"],
        },
    }


def _load_hardware(path: str) -> tuple[dict[str, Any], str]:
    import yaml

    relative = _relative_repository_path(path, label="hardware config")
    with RepositoryFileGuard(REPOSITORY_ROOT, relative, label="Stage-2 hardware config") as guard:
        value = yaml.safe_load(guard.bytes.decode("utf-8"))
        sha = guard.sha256
        guard.verify()
    if not isinstance(value, dict) or not isinstance(value.get("audio"), dict):
        raise Stage2MeasurementV2Error("hardware YAML audio mapping이 없습니다")
    return value, sha


def _measurement_git_authority() -> dict[str, Any]:
    branch = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if branch != "dev":
        raise Stage2MeasurementV2Error("Stage-2 live는 attached dev branch만 허용합니다")
    files: dict[str, str] = {}
    head: str | None = None
    for relative in _CRITICAL_LIVE_FILES:
        _content, digest, observed_head = verify_tracked_head_file(
            REPOSITORY_ROOT, relative
        )
        if head is not None and observed_head != head:
            raise Stage2MeasurementV2Error("critical live file 검증 중 HEAD가 변경됐습니다")
        head = observed_head
        files[relative] = digest
    assert head is not None
    return {
        "schema": "stage2_2khz_live_origin_dev_exact_bundle_v1",
        "branch": branch,
        "head": head,
        "origin_dev": head,
        "critical_file_sha256": files,
    }


def _resolve_devices(hardware: Mapping[str, Any]) -> dict[str, int]:
    from deep_anc.audio_io import resolve_alsa_portaudio_device

    audio = hardware["audio"]
    devices = {
        "input": resolve_alsa_portaudio_device(
            audio["input"]["card"], audio["input"]["pcm"], "input", 2
        ),
        "output": resolve_alsa_portaudio_device(
            audio["output"]["card"], audio["output"]["pcm"], "output", 2
        ),
    }
    if any(type(value) is not int or value < 0 for value in devices.values()):
        raise Stage2MeasurementV2Error("resolved audio device가 exact non-negative int가 아닙니다")
    return devices


def _run_two_phase_capture(
    *,
    repository_root: str,
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    backend: Any,
    devices: Mapping[str, int],
    capture_metadata: Mapping[str, Any],
    capture_callable: Callable[..., tuple[np.ndarray, Mapping[str, Any]]],
    pre_open_check: Callable[[], None] | None,
) -> dict[str, Any]:
    """테스트 주입 가능한 diagnostic raw-first -> PS state machine."""

    boundary = int(plan["live_phase_contract"]["diagnostic_phase_stop_frame"])
    phase_pcm = {
        "diagnostic": np.asarray(submitted_pcm[:boundary]),
        "ps": np.asarray(submitted_pcm[boundary:]),
    }
    publications: dict[str, Any] = {}

    def capture_one(phase: str) -> tuple[np.ndarray, Mapping[str, Any]]:
        try:
            return capture_callable(
                backend,
                submitted_pcm=phase_pcm[phase],
                input_device=int(devices["input"]),
                output_device=int(devices["output"]),
                pre_open_check=pre_open_check,
                watchdog_grace_seconds=WATCHDOG_GRACE_SECONDS,
                on_output_closed=lambda confirmed: print(
                    f"[출력 종료] {phase} stream close confirmed={confirmed}", flush=True
                ),
            )
        except Exception as error:
            from deep_anc.audio_duplex_stage2 import DuplexCaptureFailure

            if isinstance(error, DuplexCaptureFailure):
                partial = publish_stage2_v2_partial_raw_no_replace(
                    repository_root,
                    plan,
                    submitted_pcm,
                    phase=phase,
                    failure=error,
                    capture_metadata={**dict(capture_metadata), "phase": phase},
                )
                raise Stage2MeasurementV2Error(
                    f"{phase} stream 실패 raw 보존: {partial['path']} {partial['sha256']}"
                ) from error
            raise

    diagnostic_capture, diagnostic_telemetry = capture_one("diagnostic")
    publications["diagnostic"] = publish_stage2_v2_phase_raw_no_replace(
        repository_root,
        plan,
        submitted_pcm,
        phase="diagnostic",
        actual_submitted_pcm=phase_pcm["diagnostic"],
        captured_pcm=diagnostic_capture,
        telemetry=diagnostic_telemetry,
        capture_metadata={**dict(capture_metadata), "phase": "diagnostic"},
    )
    diagnostic_raw = snapshot_published_stage2_v2_phase(
        repository_root,
        publications["diagnostic"],
        plan,
        submitted_pcm,
        phase="diagnostic",
    )
    diagnostic_receipt = analyse_stage2_v2_diagnostic_preflight(
        plan,
        submitted_pcm,
        diagnostic_raw["captured_pcm"],
        transport_counters={"xrun": 0, "clip": 0, "callback_status": 0},
    )
    if diagnostic_receipt["passed"] is not True:
        return {
            "status": "DIAGNOSTIC_BLOCKED_PS_BACKEND_NOT_CALLED",
            "diagnostic_publication": publications["diagnostic"],
            "diagnostic_receipt": diagnostic_receipt,
            "ps_backend_calls_allowed": 0,
        }
    publications["authorization"] = seal_and_publish_diagnostic_authorization(
        repository_root,
        plan,
        diagnostic_analysis_receipt=diagnostic_receipt,
        diagnostic_raw=diagnostic_raw,
    )
    authorization = validate_published_diagnostic_authorization(
        repository_root,
        plan,
        submitted_pcm,
        publications["authorization"],
    )
    auth_ref = {
        "path": publications["authorization"]["path"],
        "sha256": publications["authorization"]["sha256"],
    }
    ps_capture, ps_telemetry = capture_one("ps")
    publications["ps"] = publish_stage2_v2_phase_raw_no_replace(
        repository_root,
        plan,
        submitted_pcm,
        phase="ps",
        actual_submitted_pcm=phase_pcm["ps"],
        captured_pcm=ps_capture,
        telemetry=ps_telemetry,
        capture_metadata={**dict(capture_metadata), "phase": "ps"},
        diagnostic_authorization_ref=auth_ref,
    )
    ps_raw = snapshot_published_stage2_v2_phase(
        repository_root,
        publications["ps"],
        plan,
        submitted_pcm,
        phase="ps",
    )
    return {
        "status": "TWO_PHASE_RAW_CAPTURE_PASS_OFFLINE_PS_ANALYSIS_REQUIRED",
        "diagnostic_publication": publications["diagnostic"],
        "diagnostic_authorization": publications["authorization"],
        "ps_publication": publications["ps"],
        "diagnostic_receipt": diagnostic_receipt,
        "authorization": authorization["authorization"],
        "ps_raw_sha256": ps_raw["raw_npz_sha256"],
        "canonical_training_eligible": False,
    }


def _execute_live(
    *,
    config: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
    submitted_pcm: Any = None,
    hardware_path: str,
    confirmations: dict[str, bool],
    meter_raw_path: str | None = None,
    expected_meter_raw_sha256: str | None = None,
) -> int:
    del config
    if LEGACY_COMBINED_LIVE_AUTHORITY_DISABLED:
        print(
            "[BLOCKED_LEGACY_COMBINED] USB DAC/APE cross-clock combined callback raw는 "
            "P/S authority가 아닙니다. output-master 경로는 retired forensic-only입니다. "
            "RT5640/J511 same-card S32 actual-P/S read-only preflight를 사용하세요. "
            "sounddevice import/open=0; output=0",
            file=sys.stderr,
        )
        return 2
    if set(confirmations) != _REQUIRED_CONFIRMATIONS or any(
        value is not True for value in confirmations.values()
    ):
        print("[중단] Stage-2 live confirmation이 완전하지 않습니다", file=sys.stderr)
        return 2
    if not meter_raw_path or not expected_meter_raw_sha256:
        print("[중단] fresh --meter-raw와 --expected-meter-raw-sha256가 필요합니다", file=sys.stderr)
        return 2
    try:
        identity = repository_execution_identity(REPOSITORY_ROOT, ADAPTER_PATH)
        git_authority = _measurement_git_authority()
        canonical_plan, canonical_pcm = build_stage2_v2_live_safe_fallback_plan()
        if plan is not None and dict(plan) != canonical_plan:
            raise Stage2MeasurementV2Error("caller plan이 canonical Stage-2 v2와 다릅니다")
        if submitted_pcm is not None and not np.array_equal(np.asarray(submitted_pcm), canonical_pcm):
            raise Stage2MeasurementV2Error("caller submitted PCM이 canonical과 다릅니다")
        hardware, hardware_sha = _load_hardware(hardware_path)
        for key in (
            "diagnostic_phase_raw",
            "diagnostic_analysis_receipt",
            "ps_phase_raw",
            "source_operating_level",
        ):
            assert_repository_target_fresh_nofollow(
                REPOSITORY_ROOT, canonical_plan["artifacts"][key], create_parents=True
            )
        from deep_anc.dsp.measurement_level import collect_alsa_physical_fingerprint

        current_fingerprint = collect_alsa_physical_fingerprint(hardware)
        current_hardware_identity = measurement_hardware_identity(
            hardware, physical_fingerprint=current_fingerprint
        )
        meter = _validate_fresh_meter(
            meter_raw_path,
            expected_meter_raw_sha256,
            expected_hardware_identity=current_hardware_identity,
        )
    except (FileNotFoundError, FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"[중단] Stage-2 backend import 전 preflight 실패: {error}", file=sys.stderr)
        return 2

    try:
        sd = importlib.import_module("sounddevice")
        devices = _resolve_devices(hardware)
        if devices != meter["resolved_devices"]:
            raise Stage2MeasurementV2Error("fresh meter 이후 PortAudio device mapping이 변경됐습니다")
        print(
            "[Stage-2 실제 출력] diagnostic 11.605초를 먼저 실행·raw fsync·판정합니다. "
            "PASS인 경우에만 P/S 12.395초를 이어서 실행합니다. 총 signal 출력은 정확히 24.000초, "
            "watchdog 포함 stream별 최대 +2초입니다.",
            flush=True,
        )
        from deep_anc.dsp.measurement_level import (
            assert_live_pcm_clock_preconditions,
            repository_audio_lock,
        )

        with repository_audio_lock(REPOSITORY_ROOT, purpose=LIVE_AUDIO_LOCK_PURPOSE) as audio_lock:
            def pre_open_check() -> None:
                assert_live_pcm_clock_preconditions(hardware["audio"])
                if _measurement_git_authority() != git_authority:
                    raise Stage2MeasurementV2Error("stream open 직전 origin/dev authority가 변경됐습니다")
                if _resolve_devices(hardware) != devices:
                    raise Stage2MeasurementV2Error("stream open 직전 device mapping이 변경됐습니다")
                observed_fingerprint = collect_alsa_physical_fingerprint(hardware)
                if observed_fingerprint != current_fingerprint:
                    raise Stage2MeasurementV2Error(
                        "stream open 직전 ALSA physical fingerprint가 meter 이후 변경됐습니다"
                    )
                if measurement_hardware_identity(
                    hardware, physical_fingerprint=observed_fingerprint
                ) != current_hardware_identity:
                    raise Stage2MeasurementV2Error(
                        "stream open 직전 physical hardware identity가 변경됐습니다"
                    )
                refreshed_meter = _validate_fresh_meter(
                    meter_raw_path,
                    expected_meter_raw_sha256,
                    expected_hardware_identity=current_hardware_identity,
                )
                if refreshed_meter != meter:
                    # age_seconds는 호출 시각에 따라 증가하므로 bytes/identity 권위 필드만
                    # 비교하고 현재 freshness 자체는 official validator가 바로 위에서 검사한다.
                    for key in (
                        "path",
                        "sha256",
                        "receipt_path",
                        "receipt_sha256",
                        "capture_id",
                        "completed_at_utc",
                        "meter_ch0_dbfs",
                        "freshness_max_seconds",
                        "resolved_devices",
                        "physical_fingerprint",
                        "hardware_identity",
                        "calibration_evidence",
                    ):
                        if refreshed_meter[key] != meter[key]:
                            raise Stage2MeasurementV2Error(
                                f"stream open 직전 fresh meter {key} binding이 변경됐습니다"
                            )

            from deep_anc.audio_duplex_stage2 import capture_duplex_stage2

            result = _run_two_phase_capture(
                repository_root=str(REPOSITORY_ROOT),
                plan=canonical_plan,
                submitted_pcm=canonical_pcm,
                backend=sd,
                devices=devices,
                capture_metadata={
                    "capture_id": secrets.token_hex(16),
                    "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "repository_execution_identity": identity,
                    "measurement_git_authority": git_authority,
                    "hardware_config_sha256": hardware_sha,
                    "resolved_devices": devices,
                    "fresh_meter": meter,
                    "operator_confirmations": confirmations,
                    "audio_lock": dict(audio_lock),
                },
                capture_callable=capture_duplex_stage2,
                pre_open_check=pre_open_check,
            )
    except (FileNotFoundError, FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"[실패] Stage-2 two-phase raw capture: {error}", file=sys.stderr)
        return 1
    if result["status"] != "TWO_PHASE_RAW_CAPTURE_PASS_OFFLINE_PS_ANALYSIS_REQUIRED":
        print(
            "[BLOCKED] diagnostic raw는 보존했지만 THD/level/ADC gate 실패로 P/S stream을 열지 않았습니다.",
            file=sys.stderr,
        )
        print(json.dumps(result["diagnostic_receipt"], ensure_ascii=False, indent=2))
        return 1
    try:
        operating_level = seal_and_publish_physical_operating_level_evidence(
            str(REPOSITORY_ROOT),
            canonical_plan,
            canonical_pcm,
            diagnostic_authorization_publication=result["diagnostic_authorization"],
            ps_raw_publication=result["ps_publication"],
        )
        validated_level = validate_published_physical_operating_level_evidence(
            str(REPOSITORY_ROOT),
            canonical_plan,
            canonical_pcm,
            operating_level,
        )
        if validated_level["evidence"] != operating_level["evidence"]:
            raise Stage2MeasurementV2Error("physical operating level reload가 다릅니다")
    except (FileNotFoundError, FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            f"[실패] P/S raw는 보존됐지만 physical operating level sealing 실패: {error}",
            file=sys.stderr,
        )
        return 1
    print("[RAW_CAPTURE_PASS] diagnostic authorization과 P/S raw가 immutable 발행됐습니다.")
    print(f"diagnostic={result['diagnostic_publication']['path']} SHA={result['diagnostic_publication']['sha256']}")
    print(f"authorization={result['diagnostic_authorization']['path']} SHA={result['diagnostic_authorization']['sha256']}")
    print(f"ps={result['ps_publication']['path']} SHA={result['ps_publication']['sha256']}")
    print(f"source_level={operating_level['path']} SHA={operating_level['sha256']}")
    print("offline DPSS/clock/P/S 분석 전이므로 canonical_training_eligible=False")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.execute_live:
        try:
            confirmations = _confirmations(arguments)
        except Stage2MeasurementV2Error as error:
            print(f"[중단] {error}", file=sys.stderr)
            return 2
        return _execute_live(
            hardware_path=arguments.hardware,
            confirmations=confirmations,
            meter_raw_path=arguments.meter_raw,
            expected_meter_raw_sha256=arguments.expected_meter_raw_sha256,
        )
    if arguments.write_plan:
        print("[BLOCK] plan 단독 발행은 physical raw authority가 아니므로 지원하지 않습니다.", file=sys.stderr)
        return 2
    plan, submitted = build_stage2_v2_live_safe_fallback_plan()
    gram = audit_stage2_v2_live_safe_dpss_gram(plan, submitted, zeros_by_path=(1_297, 1_158))
    if gram["numerical_subspace_passed"] is not True:
        print("[실패] live-safe DPSS numerical preflight 실패", file=sys.stderr)
        return 2
    print("Stage-2 2 kHz v2 two-phase 무음 검증 완료")
    print(f"signal={plan['signal_seconds']:.3f}s frames={plan['signal_frames']}")
    print(f"diagnostic={plan['live_phase_contract']['diagnostic_phase_seconds']:.6f}s")
    print(f"ps={plan['live_phase_contract']['ps_phase_seconds']:.6f}s")
    print(f"plan_sha256={plan['canonical_payload_sha256']}")
    print(
        f"projected_gram={gram['gram_dimension']} rank={gram['numeric_rank']} "
        f"condition={gram['projected_normal_matrix_condition_number']:.6f}"
    )
    print(f"live_status={LIVE_SAFE_FALLBACK_STATUS}; sounddevice import/open=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
