#!/usr/bin/env python3
"""RT5640/J511 same-card S32 실제 Stage-2 P/S raw 캡처.

기본 동작은 무음 dry-run이다. ``--execute-live``도 다섯 operator 확인과
read-only RT5640/J511 preflight가 모두 통과한 뒤에만 backend를 import한다. stream은
pre-arm 동안 exact zero만 내보내며, negotiated route/hw_params/J511/PCM 점검이
성공한 뒤에만 24초 sealed S32 자극을 arm한다. raw는 분석 전에 O_EXCL로 한 번만
발행한다. 이 명령은 P/S 분석이나 학습 권한을 자동으로 만들지 않는다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from deep_anc.audio_duplex_s32_disarmed_v10_3 import (  # noqa: E402
    S32DisarmedDuplexCaptureFailure,
    capture_disarmed_planned_s32_duplex,
)
from deep_anc.audio_io import resolve_alsa_portaudio_device  # noqa: E402
from deep_anc.data.repository_fd import repository_execution_identity  # noqa: E402
from deep_anc.dsp.rt5640_stage2_s32_preflight import (  # noqa: E402
    ReadOnlyCommandResult,
    _read_ape_routes,
    _read_j511_samples,
    assert_rt5640_stage2_s32_preflight,
)
from deep_anc.dsp.stage2_2khz_actual_ps_plan import (  # noqa: E402
    RAW_TARGET_RELATIVE_PATH,
    build_stage2_actual_ps_excitation_plan,
    load_stage2_actual_ps_static_config,
)
from deep_anc.dsp.stage2_2khz_actual_ps_live import (  # noqa: E402
    array_sha256,
    publish_actual_ps_raw_no_replace,
)
from deep_anc.dsp.measurement_level import (  # noqa: E402
    assert_live_pcm_clock_preconditions,
    repository_audio_lock,
)
from deep_anc.dsp.stage2_2khz_actual_ps_s32_capture import (  # noqa: E402
    validate_stage2_actual_ps_s32_post_start_receipt,
    validate_stage2_actual_ps_s32_user_live_gate,
)


SPEAKER_DISCONNECT_NOTICE = (
    "[출력 종료] RT5640/J511 S32 stream close 확인. 지금 스피커/앰프 연결을 해제하세요."
)
SPEAKER_STOP_UNCONFIRMED_NOTICE = (
    "[경고: 출력 종료 확인 불가] 스피커/앰프를 즉시 물리적으로 분리하세요. raw는 INVALID로 보존합니다."
)
_CONFIRMATION_KEYS = frozenset(
    {
        "confirm_speaker",
        "confirm_user_present",
        "confirm_volume_minimum",
        "confirm_routing_and_geometry",
        "confirm_same_amplifier_setting",
    }
)
_EXPECTED_ROUTES = {
    "I2S1 Mux": "ADMAIF1",
    "ADMAIF1 Mux": "I2S1",
    "ADMAIF2 Mux": "I2S2",
    "I2S2 Mux": "ADMAIF2",
}


def _read_only_runner(command: tuple[str, ...]) -> ReadOnlyCommandResult:
    completed = subprocess.run(
        list(command), check=False, capture_output=True, text=True, timeout=10.0
    )
    return ReadOnlyCommandResult(
        returncode=int(completed.returncode),
        stdout=str(completed.stdout),
        stderr=str(completed.stderr),
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _confirmations(args: argparse.Namespace) -> dict[str, bool]:
    values = {
        "confirm_speaker": bool(args.confirm_speaker),
        "confirm_user_present": bool(args.confirm_user_present),
        "confirm_volume_minimum": bool(args.confirm_volume_minimum),
        "confirm_routing_and_geometry": bool(args.confirm_routing_and_geometry),
        "confirm_same_amplifier_setting": bool(args.confirm_same_amplifier_setting),
    }
    if set(values) != _CONFIRMATION_KEYS or any(value is not True for value in values.values()):
        raise ValueError("실제 P/S 출력에는 다섯 operator confirmation이 모두 필요합니다")
    return values


def _pcm_owner_pids() -> tuple[set[int], str]:
    nodes = tuple(sorted(Path("/dev/snd").glob("pcm*")))
    if not nodes:
        raise RuntimeError("/dev/snd PCM node가 없습니다")
    result = subprocess.run(
        ["fuser", "-v", *(str(node) for node in nodes)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"stream-open 후 fuser 확인 실패: exit={result.returncode}")
    text = f"{result.stdout}\n{result.stderr}"
    pids: set[int] = set()
    for line in text.splitlines():
        match = re.match(r"^\s*/dev/snd/\S+:\s+\S+\s+(?P<pid>[0-9]+)\b", line)
        if match is not None:
            pids.add(int(match.group("pid")))
    return pids, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_hw_params(text: str) -> dict[str, Any]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    required = {"format": "S32_LE", "channels": "2", "rate": "48000", "period_size": "256"}
    for key, expected in required.items():
        observed = values.get(key, "")
        if key == "rate":
            observed = observed.split()[0] if observed else ""
        if observed != expected:
            raise RuntimeError(f"hw_params {key}={observed!r}, expected={expected!r}")
    return {
        "format": values["format"],
        "channels": int(values["channels"]),
        "rate": int(values["rate"].split()[0]),
        "period_size": int(values["period_size"]),
        "raw_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _read_open_hw_params() -> dict[str, Any]:
    """stream-start 뒤 APE pcm1 capture/pcm0 playback의 실제 hw_params를 읽는다."""

    from deep_anc.audio_io import alsa_card_index

    card = alsa_card_index("APE")
    endpoints = {
        "input": Path(f"/proc/asound/card{card}/pcm1c/sub0"),
        "output": Path(f"/proc/asound/card{card}/pcm0p/sub0"),
    }
    result: dict[str, Any] = {}
    for direction, root in endpoints.items():
        try:
            status = (root / "status").read_text(encoding="utf-8", errors="replace")
            params = (root / "hw_params").read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            raise RuntimeError(f"{direction} stream의 proc hw_params를 읽지 못했습니다: {error}") from error
        if not status.strip() or status.splitlines()[0].strip() == "closed":
            raise RuntimeError(f"stream-start 뒤 {direction} PCM이 closed입니다")
        result[direction] = {
            "status_first_line": status.splitlines()[0].strip(),
            "status_raw_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            "hw_params": _parse_hw_params(params),
        }
    return result


def _build_post_start_receipt(
    *,
    plan: Mapping[str, Any],
    static: Mapping[str, Any],
    devices: Mapping[str, int],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    states = _read_j511_samples(_read_only_runner)
    if len(states) != 3 or len(set(states)) != 1 or states[0] not in {"HP", "HS"}:
        raise RuntimeError(f"stream open 후 J511 상태가 안정적이지 않습니다: {list(states)!r}")
    routes = _read_ape_routes(_read_only_runner)
    route_json = {
        key: {str(item): value for item, value in dict(row).items()}
        for key, row in routes.items()
    }
    for control, expected in _EXPECTED_ROUTES.items():
        observed = route_json.get(control, {}).get("observed")
        if observed != expected:
            raise RuntimeError(f"stream open 후 APE route가 다릅니다: {control}={observed!r}")
    pids, fuser_sha = _pcm_owner_pids()
    own_pid = os.getpid()
    foreign = sorted(pid for pid in pids if pid != own_pid)
    if own_pid not in pids:
        raise RuntimeError("stream open 후 현재 capture PID가 PCM owner로 보이지 않습니다")
    if foreign:
        raise RuntimeError(f"stream open 후 다른 PCM owner가 있습니다: {foreign!r}")
    hardware = static["hardware_audio"]
    hw_params = _read_open_hw_params()
    occupancy = {
        "only_capture_owned_pcm_nodes": True,
        "foreign_owners": [],
        "fuser_owner_pid": own_pid,
        "fuser_output_sha256": fuser_sha,
    }
    snapshot = {
        "negotiated_hw_params": hw_params,
        "j511": {
            "state": states[0],
            "samples": 3,
            "all_samples_equal": True,
            "three_identical_connected_samples": True,
        },
        "routes": route_json,
        "occupancy": occupancy,
    }
    return {
        "schema": "stage2_2khz_actual_ps_s32_post_start_pre_arm_receipt_v1",
        "passed": True,
        "stream_started": True,
        "checked_before_arm": True,
        "pre_arm_output_exact_zero": True,
        "speaker_output_armed": False,
        "raw_written": False,
        "actual_ps_plan_sha256": plan["canonical_payload_sha256"],
        "actual_config_payload_sha256": plan["rt5640_static_config"]["config_payload_sha256"],
        "negotiated_hardware_audio": hardware,
        "negotiated_hw_params": hw_params,
        "resolved_input_device": int(devices["input"]),
        "resolved_output_device": int(devices["output"]),
        "j511": {
            "state": states[0],
            "samples": 3,
            "all_samples_equal": True,
            "three_identical_connected_samples": True,
        },
        "pcm_occupancy": occupancy,
        "ape_routes": route_json,
        "pre_snapshot_sha256": str(preflight["receipt_sha256"]),
        "post_snapshot_sha256": _payload_sha256(snapshot),
        "authority": {
            "physical_ps_authority": False,
            "canonical_training_eligible": False,
            "deployment_eligible": False,
        },
    }


def _hardware_for_clock(static: Mapping[str, Any]) -> dict[str, Any]:
    hardware = static["hardware_audio"]
    return {
        "sample_rate": int(hardware["sample_rate_hz"]),
        "block_size": int(hardware["block_size"]),
        "latency": str(hardware["latency"]),
        "input": {
            "card": str(hardware["input"]["card"]),
            "pcm": int(hardware["input"]["pcm"]),
        },
        "output": {
            "card": str(hardware["output"]["card"]),
            "pcm": int(hardware["output"]["pcm"]),
        },
    }


def _live(args: argparse.Namespace) -> int:
    confirmations = _confirmations(args)
    static = load_stage2_actual_ps_static_config(repository_root=REPO_ROOT)
    plan, planned = build_stage2_actual_ps_excitation_plan()
    preflight = assert_rt5640_stage2_s32_preflight()
    user_live_gate = {
        "schema": "stage2_2khz_actual_ps_s32_explicit_user_live_gate_v1",
        "approved": True,
        **confirmations,
        "one_time_actual_ps_output": True,
        "actual_ps_plan_sha256": plan["canonical_payload_sha256"],
        "actual_config_payload_sha256": plan["rt5640_static_config"]["config_payload_sha256"],
        "expected_output_duration_seconds": plan["duration_seconds"],
        "speaker_disconnect_notice_required": "출력 종료 — 지금 스피커 분리",
    }
    validate_stage2_actual_ps_s32_user_live_gate(user_live_gate, plan, planned)
    identity = repository_execution_identity(REPO_ROOT, "scripts/jetson/measure_stage2_2khz_actual_ps_s32.py")
    print(
        f"[실제 출력 예정] RT5640/J511 same-card S32 full-PE {plan['duration_seconds']:.3f}초, "
        "pre-arm은 zero-only입니다. 끝나면 raw를 먼저 저장하고 분석합니다.",
        flush=True,
    )
    import sounddevice as sd

    devices = {
        "input": resolve_alsa_portaudio_device("APE", 1, "input", 2),
        "output": resolve_alsa_portaudio_device("APE", 0, "output", 2),
    }
    closed_notice = False
    post_start: dict[str, Any] | None = None
    session = {
        "schema": "stage2_2khz_actual_ps_s32_live_session_v1",
        "capture_id": hashlib.sha256(os.urandom(32)).hexdigest()[:32],
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        **identity,
        "resolved_devices": devices,
        "operator_confirmations": confirmations,
    }
    captured: np.ndarray | None = None
    telemetry: Mapping[str, Any] | None = None
    partial = False
    failure_message: str | None = None
    try:
        with repository_audio_lock(REPO_ROOT, purpose="stage2_2khz_actual_ps_s32_live") as audio_lock:
            del audio_lock

            def pre_open_check() -> None:
                assert_live_pcm_clock_preconditions(_hardware_for_clock(static))
                fresh = assert_rt5640_stage2_s32_preflight()
                if fresh["receipt_sha256"] != preflight["receipt_sha256"]:
                    raise RuntimeError("stream open 직전 RT5640 preflight receipt가 변경됐습니다")
                current = {
                    "input": resolve_alsa_portaudio_device("APE", 1, "input", 2),
                    "output": resolve_alsa_portaudio_device("APE", 0, "output", 2),
                }
                if current != devices:
                    raise RuntimeError(f"stream open 직전 PortAudio mapping 변경: {current!r}")

            def post_start_pre_arm_check() -> None:
                nonlocal post_start
                post_start = _build_post_start_receipt(
                    plan=plan,
                    static=static,
                    devices=devices,
                    preflight=preflight,
                )

            def output_closed(confirmed: bool) -> None:
                nonlocal closed_notice
                print(SPEAKER_DISCONNECT_NOTICE if confirmed else SPEAKER_STOP_UNCONFIRMED_NOTICE, flush=True)
                closed_notice = True

            captured, telemetry = capture_disarmed_planned_s32_duplex(
                sd,
                planned_pcm=planned,
                input_device=devices["input"],
                output_device=devices["output"],
                pre_open_check=pre_open_check,
                post_start_pre_arm_check=post_start_pre_arm_check,
                on_output_closed=output_closed,
                watchdog_grace_seconds=2.0,
            )
    except S32DisarmedDuplexCaptureFailure as failure:
        partial = True
        captured = failure.captured_pcm
        telemetry = {
            **failure.telemetry,
            "actual_submitted_pcm": failure.actual_submitted_pcm,
            "capture_valid_mask": failure.capture_valid_mask,
            "submitted_valid_mask": failure.submitted_valid_mask,
        }
        failure_message = str(failure)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        if not closed_notice:
            print(SPEAKER_STOP_UNCONFIRMED_NOTICE, file=sys.stderr, flush=True)
        print(f"[중단] actual P/S stream이 열리지 않았습니다: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    finally:
        if not closed_notice and captured is not None:
            print(SPEAKER_STOP_UNCONFIRMED_NOTICE, file=sys.stderr, flush=True)

    assert captured is not None and telemetry is not None
    session["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        publication = publish_actual_ps_raw_no_replace(
            REPO_ROOT,
            plan=plan,
            planned_s32_pcm=planned,
            captured_pcm=captured,
            telemetry=telemetry,
            preflight_receipt=preflight,
            user_live_gate=user_live_gate,
            post_start_receipt=post_start,
            session=session,
            partial=partial,
            failure_message=failure_message,
        )
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"[실패] raw를 no-replace 발행하지 못했습니다: {error}", file=sys.stderr)
        return 1
    print(f"[보존] {publication['path']} | SHA256 {publication['sha256']}")
    if partial:
        print("[INVALID] partial raw만 보존했습니다. 자동 재측정하지 않습니다.", file=sys.stderr)
        return 1
    print("[CAPTURE_PASS] raw 저장 완료. clock/phase/P/S 분석 전이며 학습 권한은 false입니다.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="계획만 생성하고 backend를 열지 않음")
    mode.add_argument("--execute-live", action="store_true", help="승인·preflight 후 한 번의 실제 캡처")
    for name, help_text in (
        ("speaker", "스피커/앰프가 연결됨"),
        ("user-present", "사용자 입회"),
        ("volume-minimum", "앰프 최소 볼륨"),
        ("routing-and-geometry", "배선·덕트 위치 확인"),
        ("same-amplifier-setting", "meter와 같은 앰프 설정"),
    ):
        parser.add_argument(f"--confirm-{name}", action="store_true", help=help_text)
    args = parser.parse_args(argv)
    if args.dry_run:
        plan, pcm = build_stage2_actual_ps_excitation_plan()
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_NO_AUDIO",
                    "raw_target": RAW_TARGET_RELATIVE_PATH,
                    "plan_sha256": plan["canonical_payload_sha256"],
                    "planned_s32_sha256": array_sha256(pcm),
                    "frames": int(len(pcm)),
                    "duration_seconds": plan["duration_seconds"],
                    "audio_backend_imported": False,
                    "speaker_output": False,
                    "raw_written": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    try:
        return _live(args)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"[중단] {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
