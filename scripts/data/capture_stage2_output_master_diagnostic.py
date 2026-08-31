#!/usr/bin/env python3
"""Stage-2 2 kHz output-master diagnostic-only A/B capture.

기본 실행은 무음 dry-run이며 ``sounddevice``를 import하지 않는다. ``--execute-live``는
fresh meter와 clean exact ``origin/dev``를 확인한 뒤 canonical diagnostic 11.605초만
출력한다. raw를 고유 session에 no-replace로 저장·재로딩해 global clock을 분석하지만
P/S stream을 자동으로 열거나 plant/training authority를 만들지 않는다.
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

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from deep_anc.data.repository_fd import (  # noqa: E402
    assert_repository_target_fresh_nofollow,
    repository_execution_identity,
)
from deep_anc.dsp.measurement_level import (  # noqa: E402
    measurement_hardware_identity,
)
from deep_anc.dsp.stage2_2khz_measurement_v2 import (  # noqa: E402
    Stage2MeasurementV2Error,
    audit_stage2_v2_live_safe_dpss_gram,
    build_stage2_v2_live_safe_fallback_plan,
)
from deep_anc.dsp.stage2_2khz_output_master_diagnostic import (  # noqa: E402
    OUTPUT_MASTER_SESSION_ROOT,
    POST_ROLL_FRAMES,
    PRE_ROLL_FRAMES,
    OutputMasterDiagnosticCaptureError,
    capture_publish_reload_analyse_output_master_diagnostic,
    output_master_session_targets,
)
from deep_anc.train.stage2_2khz_git_authority import (  # noqa: E402
    verify_tracked_head_file,
)

# 검증된 v2 안전/장치/meter 구현을 그대로 재사용한다. 이 파일은 diagnostic-only라
# v2의 P/S state machine을 호출하지 않는다.
from scripts.data.measure_paths_stage2_2khz import (  # noqa: E402
    DEFAULT_HARDWARE,
    WATCHDOG_GRACE_SECONDS,
    _confirmations,
    _load_hardware,
    _measurement_git_authority,
    _resolve_devices,
    _validate_fresh_meter,
)


ADAPTER_PATH = "scripts/data/capture_stage2_output_master_diagnostic.py"
LIVE_AUDIO_LOCK_PURPOSE = "stage2_2khz_output_master_diagnostic_only"
_EXTRA_CRITICAL_FILES = (
    ADAPTER_PATH,
    "src/deep_anc/audio_duplex_stage2.py",
    "src/deep_anc/dsp/stage2_2khz_diagnostic_clock.py",
    "src/deep_anc/dsp/stage2_2khz_output_master_diagnostic.py",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    mode.add_argument("--execute-live", action="store_true")
    return parser


def _output_master_git_authority() -> dict[str, Any]:
    base = _measurement_git_authority()
    head = str(base["head"])
    hashes = dict(base["critical_file_sha256"])
    for relative in _EXTRA_CRITICAL_FILES:
        _content, digest, observed_head = verify_tracked_head_file(
            REPOSITORY_ROOT, relative
        )
        if observed_head != head:
            raise Stage2MeasurementV2Error(
                "output-master critical file 검증 중 HEAD가 변경됐습니다"
            )
        hashes[relative] = digest
    return {
        "schema": "stage2_2khz_output_master_origin_dev_exact_bundle_v1",
        "branch": base["branch"],
        "head": head,
        "origin_dev": base["origin_dev"],
        "critical_file_sha256": hashes,
    }


def _new_session_relative_path() -> str:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"{OUTPUT_MASTER_SESSION_ROOT}/{now}_{secrets.token_hex(8)}"


def _authority_equal_except_meter_age(
    current: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    keys = (
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
    )
    return all(current[key] == expected[key] for key in keys)


def _execute_live(arguments: argparse.Namespace) -> int:
    try:
        confirmations = _confirmations(arguments)
    except Stage2MeasurementV2Error as error:
        print(f"[중단] {error}", file=sys.stderr)
        return 2
    if not arguments.meter_raw or not arguments.expected_meter_raw_sha256:
        print(
            "[중단] fresh --meter-raw와 --expected-meter-raw-sha256가 필요합니다",
            file=sys.stderr,
        )
        return 2
    try:
        identity = repository_execution_identity(REPOSITORY_ROOT, ADAPTER_PATH)
        git_authority = _output_master_git_authority()
        plan, full_submitted = build_stage2_v2_live_safe_fallback_plan()
        boundary = int(plan["live_phase_contract"]["diagnostic_phase_stop_frame"])
        diagnostic_seconds = boundary / 48_000.0
        nonzero_frames = int(
            np.count_nonzero(np.any(full_submitted[:boundary] != 0, axis=1))
        )
        hardware, hardware_sha = _load_hardware(arguments.hardware)
        session = _new_session_relative_path()
        targets = output_master_session_targets(session)
        for target in targets.values():
            assert_repository_target_fresh_nofollow(
                REPOSITORY_ROOT, target, create_parents=True
            )
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
    except (
        FileNotFoundError,
        FileExistsError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"[중단] output-master backend import 전 preflight 실패: {error}", file=sys.stderr)
        return 2

    try:
        sd = importlib.import_module("sounddevice")
        devices = _resolve_devices(hardware)
        if devices != meter["resolved_devices"]:
            raise Stage2MeasurementV2Error(
                "fresh meter 이후 PortAudio device mapping이 변경됐습니다"
            )
        print(
            "[Stage-2 output-master 실제 출력] diagnostic-only signal "
            f"stream {diagnostic_seconds:.6f}초(실제 nonzero PCM "
            f"{nonzero_frames / 48_000.0:.6f}초)를 한 번 출력합니다. "
            f"input-only pre-roll={PRE_ROLL_FRAMES / 48_000:.6f}초, "
            f"post-roll={POST_ROLL_FRAMES / 48_000:.6f}초이며 두 구간은 무출력입니다. "
            "raw 저장·재로딩·clock 분석 후에도 P/S는 자동 실행하지 않습니다.",
            flush=True,
        )
        from deep_anc.dsp.measurement_level import (
            assert_live_pcm_clock_preconditions,
            collect_alsa_physical_fingerprint,
            repository_audio_lock,
        )

        with repository_audio_lock(
            REPOSITORY_ROOT, purpose=LIVE_AUDIO_LOCK_PURPOSE
        ) as audio_lock:

            def pre_open_check() -> None:
                assert_live_pcm_clock_preconditions(hardware["audio"])
                if _output_master_git_authority() != git_authority:
                    raise Stage2MeasurementV2Error(
                        "stream open 직전 origin/dev authority가 변경됐습니다"
                    )
                if _resolve_devices(hardware) != devices:
                    raise Stage2MeasurementV2Error(
                        "stream open 직전 device mapping이 변경됐습니다"
                    )
                observed = collect_alsa_physical_fingerprint(hardware)
                if observed != fingerprint:
                    raise Stage2MeasurementV2Error(
                        "stream open 직전 ALSA fingerprint가 meter 이후 변경됐습니다"
                    )
                if measurement_hardware_identity(
                    hardware, physical_fingerprint=observed
                ) != hardware_identity:
                    raise Stage2MeasurementV2Error(
                        "stream open 직전 hardware identity가 변경됐습니다"
                    )
                refreshed = _validate_fresh_meter(
                    arguments.meter_raw,
                    arguments.expected_meter_raw_sha256,
                    expected_hardware_identity=hardware_identity,
                )
                if not _authority_equal_except_meter_age(refreshed, meter):
                    raise Stage2MeasurementV2Error(
                        "stream open 직전 fresh meter binding이 변경됐습니다"
                    )
                for target in targets.values():
                    assert_repository_target_fresh_nofollow(
                        REPOSITORY_ROOT, target, create_parents=False
                    )

            from deep_anc.audio_duplex_stage2 import capture_output_master_stage2

            result = capture_publish_reload_analyse_output_master_diagnostic(
                str(REPOSITORY_ROOT),
                session,
                plan,
                full_submitted,
                backend=sd,
                devices=devices,
                capture_metadata={
                    "capture_id": secrets.token_hex(16),
                    "started_at_utc": dt.datetime.now(
                        dt.timezone.utc
                    ).isoformat(),
                    "repository_execution_identity": identity,
                    "measurement_git_authority": git_authority,
                    "hardware_config_sha256": hardware_sha,
                    "resolved_devices": devices,
                    "fresh_meter": meter,
                    "operator_confirmations": confirmations,
                    "audio_lock": dict(audio_lock),
                },
                capture_callable=capture_output_master_stage2,
                pre_open_check=pre_open_check,
                watchdog_grace_seconds=WATCHDOG_GRACE_SECONDS,
                on_output_closed=lambda confirmed: print(
                    "[스피커 출력 종료 — 지금 스피커/앰프를 분리하세요] "
                    f"output close confirmed={confirmed}",
                    flush=True,
                ),
            )
    except OutputMasterDiagnosticCaptureError as error:
        print(f"[실패] {error}", file=sys.stderr)
        print("자동 재측정 금지 — 보존 raw를 먼저 분석합니다.", file=sys.stderr)
        return 1
    except (
        FileNotFoundError,
        FileExistsError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"[실패] output-master diagnostic: {error}", file=sys.stderr)
        return 1
    print(
        f"raw={result['raw_publication']['path']} "
        f"SHA={result['raw_publication']['sha256']}"
    )
    print(
        f"clock={result['clock_publication']['path']} "
        f"SHA={result['clock_publication']['sha256']}"
    )
    print(
        "diagnostic-only: ps_phase_may_start=False, "
        "plant_identification_eligible=False, canonical_training_eligible=False"
    )
    if result["clock_receipt"]["passed"] is not True:
        print("[BLOCKED] global affine clock gate FAIL; 자동 재측정/P/S 실행 없음")
        return 1
    print("[PASS] output-master global affine clock A/B; P/S는 별도 승인 경로에서만 진행")
    return 0


def _dry_run() -> int:
    plan, submitted = build_stage2_v2_live_safe_fallback_plan()
    gram = audit_stage2_v2_live_safe_dpss_gram(
        plan, submitted, zeros_by_path=(1_297, 1_158)
    )
    if gram["numerical_subspace_passed"] is not True:
        print("[실패] output-master signal numerical preflight 실패", file=sys.stderr)
        return 2
    boundary = int(plan["live_phase_contract"]["diagnostic_phase_stop_frame"])
    nonzero_frames = int(
        np.count_nonzero(np.any(submitted[:boundary] != 0, axis=1))
    )
    print("Stage-2 output-master diagnostic-only 무음 dry-run PASS")
    print(f"output_stream={boundary / 48_000.0:.6f}s frames={boundary}")
    print(
        f"nonzero_output={nonzero_frames / 48_000.0:.6f}s "
        f"frames={nonzero_frames} peak_pcm={int(np.max(np.abs(submitted[:boundary])))}"
    )
    print(
        f"input_pre_roll={PRE_ROLL_FRAMES / 48_000.0:.6f}s "
        f"input_post_roll={POST_ROLL_FRAMES / 48_000.0:.6f}s"
    )
    print(f"plan_sha256={plan['canonical_payload_sha256']}")
    print("sounddevice import/open=0; raw write=0; P/S backend calls=0")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.execute_live:
        return _execute_live(arguments)
    return _dry_run()


if __name__ == "__main__":
    raise SystemExit(main())
