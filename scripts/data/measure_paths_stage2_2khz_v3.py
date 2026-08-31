#!/usr/bin/env python3
"""Retired Stage-2 output-master P/S v3 forensic adapter.

USB AB13X output과 APE input의 actual split-clock failure 뒤 이 경로는 더 이상
physical P/S를 재측정하지 않는다. 기본 실행의 무음 dry-run과 기존 raw의 offline
forensic 검증만 보존한다. ``--execute-live``는 backend import 전에 영구 차단한다.
현행 physical 후보는 RT5640/J511 same-card S32 actual-P/S read-only preflight다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
from pathlib import Path
import secrets
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    # 직접 ``python scripts/data/...``로 실행해도 ``scripts.data``의 tracked
    # diagnostic adapter를 같은 checkout에서만 import하게 한다. 이 경로는
    # sounddevice/PCM을 열기 전에 설정된다.
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from deep_anc.data.repository_fd import (  # noqa: E402
    assert_repository_target_fresh_nofollow,
    repository_execution_identity,
)
from deep_anc.dsp.measurement_level import measurement_hardware_identity  # noqa: E402
from deep_anc.dsp.stage2_2khz_measurement_v2 import (  # noqa: E402
    build_stage2_v2_live_safe_fallback_plan,
)
from deep_anc.dsp.stage2_2khz_output_master_diagnostic import (  # noqa: E402
    output_master_session_targets,
)
from deep_anc.dsp.stage2_2khz_output_master_ps_v3 import (  # noqa: E402
    _assess_stage2_output_master_ps_v3_admission_after_cli_authority,
    _run_stage2_output_master_ps_v3_after_cli_authority,
    assess_stage2_output_master_ps_v3_admission,
    analyse_stage2_output_master_ps_v3_capture,
    build_stage2_output_master_ps_v3_plan,
    output_master_ps_v3_session_targets,
    publish_stage2_output_master_diagnostic_linearity_v3_no_replace,
    publish_stage2_output_master_ps_v3_physical_level_no_replace,
    validate_published_stage2_output_master_diagnostic_linearity_v3,
)
from scripts.data.capture_stage2_output_master_diagnostic import (  # noqa: E402
    _authority_equal_except_meter_age,
    _output_master_git_authority,
)
from scripts.data.measure_paths_stage2_2khz import (  # noqa: E402
    DEFAULT_HARDWARE,
    WATCHDOG_GRACE_SECONDS,
    _confirmations,
    _load_hardware,
    _resolve_devices,
    _validate_fresh_meter,
)


ADAPTER_PATH = "scripts/data/measure_paths_stage2_2khz_v3.py"
OUTPUT_MASTER_SPLIT_CLOCK_RETIRED = True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default=DEFAULT_HARDWARE)
    parser.add_argument("--meter-raw")
    parser.add_argument("--expected-meter-raw-sha256")
    parser.add_argument(
        "--diagnostic-session",
        help=(
            "results/stage2_2khz_output_master_diagnostic 바로 아래 durable session"
        ),
    )
    parser.add_argument(
        "--diagnostic-clock-sha256",
        help="해당 session clock_receipt.json의 expected file SHA-256",
    )
    parser.add_argument(
        "--diagnostic-linearity-sha256",
        help="해당 session linearity_v3.json의 expected file SHA-256",
    )
    parser.add_argument("--confirm-speaker", action="store_true")
    parser.add_argument("--confirm-user-present", action="store_true")
    parser.add_argument("--confirm-volume-fixed", action="store_true")
    parser.add_argument("--confirm-routing-and-geometry", action="store_true")
    parser.add_argument("--confirm-same-amplifier-setting", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--publish-diagnostic-linearity", action="store_true")
    mode.add_argument("--execute-live", action="store_true")
    return parser


def _diagnostic_arguments(arguments: argparse.Namespace):
    supplied = (
        arguments.diagnostic_session is not None,
        arguments.diagnostic_clock_sha256 is not None,
        arguments.diagnostic_linearity_sha256 is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError(
            "--diagnostic-session/--diagnostic-clock-sha256/"
            "--diagnostic-linearity-sha256는 함께 필요합니다"
        )
    if not all(supplied):
        return {}
    digest = str(arguments.diagnostic_clock_sha256)
    linearity_digest = str(arguments.diagnostic_linearity_sha256)
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in (digest, linearity_digest)
    ):
        raise ValueError("diagnostic clock/linearity expected SHA-256 형식이 잘못됐습니다")
    session = str(arguments.diagnostic_session)
    target = output_master_session_targets(session)["clock_receipt"]
    return {
        "diagnostic_session_relative_path": session,
        "diagnostic_clock_publication": {"path": target, "sha256": digest},
        "diagnostic_linearity_publication": {
            "path": f"{session}/linearity_v3.json",
            "sha256": linearity_digest,
        },
    }


def _validate_v3_git_authority(
    identity: Mapping[str, Any], authority: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        identity.get("repository_branch") != "dev"
        or identity.get("repository_dirty") is not False
        or authority.get("branch") != "dev"
        or identity.get("repository_commit") != authority.get("head")
        or authority.get("head") != authority.get("origin_dev")
    ):
        raise ValueError(
            "Stage-2 P/S v3 live는 clean attached dev가 exact origin/dev HEAD인 경우만 허용됩니다"
        )
    return {"identity": dict(identity), "authority": dict(authority)}


def _new_ps_session() -> str:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"results/stage2_2khz_output_master_ps_v3/{now}_{secrets.token_hex(8)}"


def _dry_run(arguments: argparse.Namespace) -> int:
    plan, full = build_stage2_v2_live_safe_fallback_plan()
    v3_plan, _ps = build_stage2_output_master_ps_v3_plan(plan, full)
    try:
        evidence = _diagnostic_arguments(arguments)
        if evidence:
            validate_published_stage2_output_master_diagnostic_linearity_v3(
                str(REPOSITORY_ROOT),
                evidence["diagnostic_session_relative_path"],
                evidence["diagnostic_clock_publication"],
                evidence["diagnostic_linearity_publication"],
                plan,
                full,
            )
        admission = assess_stage2_output_master_ps_v3_admission(plan, full)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"[BLOCKED] diagnostic artifact 검증 실패: {error}", file=sys.stderr)
        return 2
    print("Retired output-master P/S v3 forensic 무음 dry-run")
    print(
        f"ps_output={v3_plan['ps_output_seconds']:.6f}s "
        f"frames={v3_plan['ps_output_frames']}"
    )
    print(
        "input_clock_frames=variable; input/output same-index identity=false; "
        "legacy combined authority=false"
    )
    print(f"plan_sha256={v3_plan['canonical_payload_sha256']}")
    print(f"status={admission['status']}")
    for blocker in admission["blockers"]:
        print(f"blocker={blocker}")
    print("sounddevice import/open=0; P/S output=0; raw write=0")
    return 0


def _execute_live(arguments: argparse.Namespace) -> int:
    if OUTPUT_MASTER_SPLIT_CLOCK_RETIRED:
        print(
            "[BLOCKED_RETIRED_OUTPUT_MASTER_SPLIT_CLOCK] USB AB13X output-master "
            "경로는 실제 global clock failure raw 뒤 forensic-only입니다. "
            "sounddevice import/open=0; P/S output=0; raw write=0",
            file=sys.stderr,
        )
        return 2
    try:
        confirmations = _confirmations(arguments)
        if not arguments.meter_raw or not arguments.expected_meter_raw_sha256:
            raise ValueError("fresh --meter-raw와 --expected-meter-raw-sha256가 필요합니다")
        evidence = _diagnostic_arguments(arguments)
        if not evidence:
            raise ValueError("durable diagnostic clock/linearity publication이 필요합니다")
        identity = repository_execution_identity(REPOSITORY_ROOT, ADAPTER_PATH)
        git_authority = _output_master_git_authority()
        _validate_v3_git_authority(identity, git_authority)
        plan, full = build_stage2_v2_live_safe_fallback_plan()
        durable = validate_published_stage2_output_master_diagnostic_linearity_v3(
            str(REPOSITORY_ROOT),
            evidence["diagnostic_session_relative_path"],
            evidence["diagnostic_clock_publication"],
            evidence["diagnostic_linearity_publication"],
            plan,
            full,
        )
        diagnostic_binding = durable["receipt"]
        del diagnostic_binding
        hardware, hardware_sha = _load_hardware(arguments.hardware)
        from deep_anc.dsp.measurement_level import collect_alsa_physical_fingerprint

        fingerprint = collect_alsa_physical_fingerprint(hardware)
        hardware_identity = measurement_hardware_identity(
            hardware, physical_fingerprint=fingerprint
        )
        meter = _validate_fresh_meter(
            arguments.meter_raw,
            arguments.expected_meter_raw_sha256,
            expected_hardware_identity=hardware_identity,
        )
        devices = _resolve_devices(hardware)
        if devices != meter["resolved_devices"]:
            raise ValueError("fresh meter 이후 PortAudio device mapping이 변경됐습니다")
        # Durable diagnostic raw과 exact continuity는 admission이 다시 검증한다.
        from deep_anc.dsp.stage2_2khz_output_master_ps_v3 import (
            validate_output_master_diagnostic_clock_publication,
        )

        clock_bundle = validate_output_master_diagnostic_clock_publication(
            str(REPOSITORY_ROOT),
            evidence["diagnostic_session_relative_path"],
            evidence["diagnostic_clock_publication"],
            plan,
            full,
        )
        capture_metadata = {
            "capture_id": clock_bundle["physical_capture_binding"]["capture_id"],
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "repository_execution_identity": identity,
            "measurement_git_authority": git_authority,
            "hardware_config_sha256": hardware_sha,
            "resolved_devices": devices,
            "fresh_meter": meter,
            "operator_confirmations": confirmations,
        }
        admission = _assess_stage2_output_master_ps_v3_admission_after_cli_authority(
            plan,
            full,
            repository_root=str(REPOSITORY_ROOT),
            ps_capture_metadata=capture_metadata,
            **evidence,
        )
        if admission["ps_stream_may_open"] is not True:
            raise ValueError(f"P/S v3 admission BLOCKED: {admission['blockers']}")
        ps_session = _new_ps_session()
        for target in output_master_ps_v3_session_targets(ps_session).values():
            assert_repository_target_fresh_nofollow(
                REPOSITORY_ROOT, target, create_parents=True
            )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"[BLOCKED_BEFORE_AUDIO] {error}", file=sys.stderr)
        return 2

    try:
        from deep_anc.audio_duplex_stage2 import capture_output_master_stage2
        from deep_anc.dsp.measurement_level import (
            assert_live_pcm_clock_preconditions,
            collect_alsa_physical_fingerprint,
            repository_audio_lock,
        )

        def pre_open_check() -> None:
            assert_live_pcm_clock_preconditions(hardware["audio"])
            current_authority = _output_master_git_authority()
            if current_authority != git_authority:
                raise ValueError("stream open 직전 origin/dev authority가 변경됐습니다")
            if _resolve_devices(hardware) != devices:
                raise ValueError("stream open 직전 device mapping이 변경됐습니다")
            observed = collect_alsa_physical_fingerprint(hardware)
            if observed != fingerprint:
                raise ValueError("stream open 직전 ALSA fingerprint가 변경됐습니다")
            refreshed = _validate_fresh_meter(
                arguments.meter_raw,
                arguments.expected_meter_raw_sha256,
                expected_hardware_identity=hardware_identity,
            )
            if not _authority_equal_except_meter_age(refreshed, meter):
                raise ValueError("stream open 직전 fresh meter binding이 변경됐습니다")
            for target in output_master_ps_v3_session_targets(ps_session).values():
                assert_repository_target_fresh_nofollow(
                    REPOSITORY_ROOT, target, create_parents=False
                )

        print(
            "[Stage-2 output-master P/S 실제 출력] "
            "12.394667초 P/S signal + input-only pre/post-roll을 1회 실행합니다.",
            flush=True,
        )
        with repository_audio_lock(
            REPOSITORY_ROOT, purpose="stage2_2khz_output_master_ps_v3"
        ):

            def capture_once(*, submitted_pcm):
                backend = importlib.import_module("sounddevice")
                return capture_output_master_stage2(
                    backend,
                    submitted_pcm=submitted_pcm,
                    input_device=devices["input"],
                    output_device=devices["output"],
                    pre_open_check=pre_open_check,
                    watchdog_grace_seconds=WATCHDOG_GRACE_SECONDS,
                    on_output_closed=lambda confirmed: print(
                        "[스피커 출력 종료 — 지금 스피커/앰프를 분리하세요] "
                        f"output close confirmed={confirmed}",
                        flush=True,
                    ),
                )

            result = _run_stage2_output_master_ps_v3_after_cli_authority(
                plan,
                full,
                capture_callable=capture_once,
                repository_root=str(REPOSITORY_ROOT),
                ps_session_relative_path=ps_session,
                ps_capture_metadata=capture_metadata,
                **evidence,
            )
        if result["ps_raw_written"] is not True:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1
        level = publish_stage2_output_master_ps_v3_physical_level_no_replace(
            str(REPOSITORY_ROOT),
            ps_session,
            result["ps_raw_publication"],
            plan,
            full,
            diagnostic_session_relative_path=evidence[
                "diagnostic_session_relative_path"
            ],
            diagnostic_clock_publication=evidence[
                "diagnostic_clock_publication"
            ],
            diagnostic_linearity_publication=evidence[
                "diagnostic_linearity_publication"
            ],
        )
        analysis, _arrays = analyse_stage2_output_master_ps_v3_capture(
            plan,
            full,
            repository_root=str(REPOSITORY_ROOT),
            diagnostic_session_relative_path=evidence[
                "diagnostic_session_relative_path"
            ],
            diagnostic_clock_publication=evidence[
                "diagnostic_clock_publication"
            ],
            diagnostic_linearity_publication=evidence[
                "diagnostic_linearity_publication"
            ],
            ps_session_relative_path=ps_session,
            ps_raw_publication=result["ps_raw_publication"],
            physical_level_publication={
                "path": level["path"],
                "sha256": level["sha256"],
            },
        )
    except (FileNotFoundError, FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"[FAIL_AFTER_ADMISSION] {error}", file=sys.stderr)
        return 1
    print(json.dumps({"execution": result, "analysis": analysis}, ensure_ascii=False, indent=2))
    return 0


def _publish_diagnostic_linearity(arguments: argparse.Namespace) -> int:
    """기존 diagnostic raw/clock만 재개방하는 무음 offline publication."""

    try:
        if (
            not arguments.diagnostic_session
            or not arguments.diagnostic_clock_sha256
            or arguments.diagnostic_linearity_sha256 is not None
        ):
            raise ValueError(
                "offline publication은 --diagnostic-session과 "
                "--diagnostic-clock-sha256만 필요하며 linearity SHA는 아직 없어야 합니다"
            )
        clock_digest = str(arguments.diagnostic_clock_sha256)
        if len(clock_digest) != 64 or any(
            character not in "0123456789abcdef" for character in clock_digest
        ):
            raise ValueError("diagnostic clock SHA-256 형식이 잘못됐습니다")
        identity = repository_execution_identity(REPOSITORY_ROOT, ADAPTER_PATH)
        authority = _output_master_git_authority()
        _validate_v3_git_authority(identity, authority)
        plan, full = build_stage2_v2_live_safe_fallback_plan()
        session = str(arguments.diagnostic_session)
        clock_ref = {
            "path": output_master_session_targets(session)["clock_receipt"],
            "sha256": clock_digest,
        }
        target = f"{session}/linearity_v3.json"
        assert_repository_target_fresh_nofollow(
            REPOSITORY_ROOT, target, create_parents=False
        )
        publication = publish_stage2_output_master_diagnostic_linearity_v3_no_replace(
            str(REPOSITORY_ROOT), session, clock_ref, plan, full
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"[BLOCKED_NO_AUDIO] {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "sounddevice_import_open": 0,
                "linearity_publication": {
                    "path": publication["path"],
                    "sha256": publication["sha256"],
                },
                "receipt": publication["receipt"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.publish_diagnostic_linearity:
        return _publish_diagnostic_linearity(arguments)
    if arguments.execute_live:
        return _execute_live(arguments)
    return _dry_run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
