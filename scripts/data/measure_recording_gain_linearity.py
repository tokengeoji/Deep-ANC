#!/usr/bin/env python3
"""Source gain v2 bounded ESS/IMD probe를 계획·캡처·분석한다.

``--dry-run``과 ``--analyze-raw``는 sounddevice를 import/open하지 않는다. 실제 NS
speaker 출력은 exact saved plan, clean expected commit, fresh hardware fingerprint와 모든
확인 플래그를 갖춘 ``--execute-live``에서만 열린다. 한 level/slot 뒤 peak를 검사하여
0.40 이상 또는 다음 level 예측 0.45 이상이면 더 높은 출력을 열지 않고 partial raw를
immutable하게 보존한다.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, ContextManager

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.audio_io import (  # noqa: E402
    alsa_card_index,
    analyze_int32_input_probe,
    capture_measurement_preflight_raw,
    resolve_alsa_portaudio_device,
)
from deep_anc.config import load_yaml  # noqa: E402
from deep_anc.data.recording_gain_linearity import (  # noqa: E402
    GAIN_LINEARITY_RAW_SCHEMA,
    INPUT_PREFLIGHT_SECONDS,
    RecordingGainLinearityError,
    build_gain_linearity_plan,
    issue_gain_linearity_receipt,
    load_gain_linearity_plan,
    next_level_stop_decision,
)
from deep_anc.data.repository_fd import (  # noqa: E402
    RepositoryFileGuard,
    repository_execution_identity,
)
from deep_anc.dsp.measurement_level import (  # noqa: E402
    assert_live_pcm_clock_preconditions,
    collect_alsa_physical_fingerprint,
    measurement_hardware_identity,
    repository_audio_lock,
    validate_measurement_hardware_contract,
)
from scripts.data import measure_paths_broadband_interleaved as broadband  # noqa: E402
from scripts.data import measure_paths_interleaved as mpi  # noqa: E402


DEFAULT_HARDWARE = "configs/hardware_jetson.yaml"
DEFAULT_PLAN = "results/data_audit/recording_gain_linearity_v2_plan.json"
DEFAULT_RAW_ROOT = "results/recording_gain_linearity_v2"
SCRIPT_RELATIVE_PATH = "scripts/data/measure_recording_gain_linearity.py"
LIVE_LOCK_PURPOSE = "recording_gain_linearity_v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path, *, results: bool = False) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise RecordingGainLinearityError("경로는 저장소 안이어야 합니다") from exc
    if results and (not relative.parts or relative.parts[0] != "results"):
        raise RecordingGainLinearityError("live raw 경로는 results/ 아래여야 합니다")
    return resolved


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _environment(hardware_path: str | Path) -> dict[str, Any]:
    hardware_file = _repo_path(hardware_path)
    config = load_yaml(hardware_file)
    audio, channel_map = validate_measurement_hardware_contract(config)
    mpi.validate_alsa_pcm_mapping(
        input_card=alsa_card_index(str(audio["input"]["card"])),
        input_pcm=int(audio["input"]["pcm"]),
        output_card=alsa_card_index(str(audio["output"]["card"])),
        output_pcm=int(audio["output"]["pcm"]),
    )
    fingerprint = collect_alsa_physical_fingerprint(config)
    return {
        "hardware_file": hardware_file,
        "config": config,
        "audio": audio,
        "channel_map": channel_map,
        "fingerprint": fingerprint,
        "identity": measurement_hardware_identity(
            config, physical_fingerprint=fingerprint
        ),
    }


def _raw_session_for_plan(plan: dict[str, Any], requested: str | Path | None) -> Path:
    if requested is None:
        requested = Path(DEFAULT_RAW_ROOT) / str(plan["plan_payload_sha256"])[:16]
    result = _repo_path(requested, results=True)
    broadband.validate_fresh_raw_session_target(result)
    return result


def _write_plan(path: Path, payload: dict[str, Any]) -> str:
    data = (
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RecordingGainLinearityError(f"plan은 no-replace입니다: {path}") from exc
    return hashlib.sha256(data).hexdigest()


def _telemetry_invalid(telemetry: dict[str, Any]) -> list[str]:
    reasons = mpi.capture_telemetry_invalid_reasons(telemetry)
    if int(telemetry.get("callback_status_count", 0)) != 0:
        reasons.append("callback_status_nonzero")
    if telemetry.get("stream_abort_error") is not None:
        reasons.append("stream_abort_error")
    if telemetry.get("stream_close_error") is not None:
        reasons.append("stream_close_error")
    if telemetry.get("output_stop_confirmed") is not True:
        reasons.append("output_stop_unconfirmed")
    return list(dict.fromkeys(reasons))


def execute_live_capture(
    *,
    hardware_path: str | Path,
    expected_commit: str,
    plan_path: str | Path,
    expected_plan_sha256: str,
    raw_session_dir: Path,
    confirmations: dict[str, bool],
    sounddevice_module: Any | None = None,
    capture_function: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]] = (
        mpi.capture_measurement_preserving_partial
    ),
    audio_lock_factory: Callable[..., ContextManager[Any]] = repository_audio_lock,
) -> dict[str, Any]:
    required = {
        "speaker_output": True,
        "user_present": True,
        "volume_minimum": True,
        "routing_and_geometry": True,
        "same_amplifier_setting": True,
        "bounded_gain_probe": True,
    }
    if confirmations != required:
        raise RecordingGainLinearityError("v2 live 확인 플래그가 모두 필요합니다")
    expected_commit = str(expected_commit).lower()
    execution = repository_execution_identity(REPO_ROOT, SCRIPT_RELATIVE_PATH)
    if execution["repository_commit"] != expected_commit:
        raise RecordingGainLinearityError("live repository commit이 외부 expected commit과 다릅니다")
    environment = _environment(hardware_path)
    loaded = load_gain_linearity_plan(
        repo_root=REPO_ROOT,
        plan_path=_relative(_repo_path(plan_path)),
        expected_sha256=expected_plan_sha256,
    )
    plan = loaded["payload"]
    pcm = loaded["pcm"]
    rebuilt, rebuilt_pcm = build_gain_linearity_plan(
        repo_root=REPO_ROOT,
        hardware_path=_relative(environment["hardware_file"]),
        source_commit=expected_commit,
        physical_fingerprint=environment["fingerprint"],
    )
    if plan != rebuilt or not np.array_equal(pcm, rebuilt_pcm):
        raise RecordingGainLinearityError("saved plan이 현재 commit/hardware/PCM과 다릅니다")
    raw_session_dir = _repo_path(raw_session_dir, results=True)
    broadband.validate_fresh_raw_session_target(raw_session_dir)
    if sounddevice_module is None:
        sounddevice_module = importlib.import_module("sounddevice")
    sd = sounddevice_module
    audio = environment["audio"]
    channel_map = environment["channel_map"]
    preflight_raw: np.ndarray | None = None
    preflight_report: dict[str, Any] | None = None
    segment_raw: list[np.ndarray] = []
    segment_pcm: list[np.ndarray] = []
    segment_telemetry: list[dict[str, Any]] = []
    invalid_reasons: list[str] = []
    safety_stop: dict[str, Any] | None = None
    created: Path | None = None
    plan_guard = RepositoryFileGuard(
        REPO_ROOT, loaded["file"]["path"], label="gain-linearity saved plan"
    )
    plan_guard.__enter__()
    if plan_guard.sha256 != str(expected_plan_sha256).lower():
        plan_guard.close()
        raise RecordingGainLinearityError("held saved plan SHA가 외부 anchor와 다릅니다")

    def verify_live_authority() -> None:
        plan_guard.verify()
        refreshed_execution = repository_execution_identity(
            REPO_ROOT, SCRIPT_RELATIVE_PATH
        )
        if refreshed_execution != execution:
            raise RecordingGainLinearityError("live 중 repository execution identity 변경")
        refreshed_environment = _environment(hardware_path)
        if (
            refreshed_environment["fingerprint"] != environment["fingerprint"]
            or _sha256_file(refreshed_environment["hardware_file"])
            != plan["hardware"]["sha256"]
        ):
            raise RecordingGainLinearityError("live 중 hardware/fingerprint 변경")
        refreshed = load_gain_linearity_plan(
            repo_root=REPO_ROOT,
            plan_path=loaded["file"]["path"],
            expected_sha256=expected_plan_sha256,
        )
        if refreshed["payload"] != plan or not np.array_equal(refreshed["pcm"], pcm):
            raise RecordingGainLinearityError("live 중 saved plan/PCM 변경")
        if created is not None:
            for name in ("raw_measurement.npz", "metadata.json"):
                if (created / name).exists():
                    raise RecordingGainLinearityError(
                        f"live 중 raw target이 선점됐습니다: {name}"
                    )
        assert_live_pcm_clock_preconditions(audio)

    def capture_all() -> None:
        nonlocal preflight_raw, preflight_report, safety_stop, created
        with audio_lock_factory(REPO_ROOT, purpose=LIVE_LOCK_PURPOSE):
            assert_live_pcm_clock_preconditions(audio)
            preflight_raw, official_preflight = capture_measurement_preflight_raw(
                sd, audio, seconds=INPUT_PREFLIGHT_SECONDS
            )
            recomputed = analyze_int32_input_probe(
                preflight_raw,
                min_rms_dbfs=-80.0,
                max_clip_ratio=0.005,
            )
            preflight_report = {
                **recomputed,
                "device": int(official_preflight["resolved_input_device"]),
                "sample_rate": int(official_preflight["sample_rate_hz"]),
                "settle_seconds": 0.5,
            }
            channels = preflight_report.get("channels", [])
            indices = (channel_map["error_mic"], channel_map["reference_mic"])
            if official_preflight.get("passed") is not True or len(channels) < 2 or not all(
                bool(channels[index].get("valid")) for index in indices
            ):
                raise RecordingGainLinearityError("ERR/REF input-only preflight FAIL")
            in_dev = int(preflight_report["device"])
            out_dev = int(
                resolve_alsa_portaudio_device(
                    audio["output"]["card"], audio["output"]["pcm"], "output", 2
                )
            )
            created = mpi.create_session_directory(
                raw_session_dir.parent, raw_session_dir.name
            )
            for index, group in enumerate(plan["capture_groups"]):
                level = int(group["level_millionths"])
                verify_live_authority()
                start, stop = int(group["start_frame"]), int(group["stop_frame"])
                segment = pcm[start:stop]
                output_float = segment.astype(np.float32) / np.float32(32767.0)
                try:
                    recorded, submitted, telemetry = capture_function(
                        sd,
                        fs=int(audio["sample_rate"]),
                        block_size=int(audio["block_size"]),
                        latency=str(audio["latency"]),
                        in_dev=in_dev,
                        out_dev=out_dev,
                        output_float=output_float,
                        pre_open_check=verify_live_authority,
                        record_callback_time_info=False,
                    )
                except mpi.PartialCaptureError as exc:
                    count = int(exc.telemetry.get("captured_frames", 0))
                    segment_raw.append(np.asarray(exc.recorded_raw[:count], dtype=np.int32))
                    segment_pcm.append(np.asarray(exc.output_pcm[:count], dtype=np.int16))
                    segment_telemetry.append(
                        {
                            **dict(exc.telemetry),
                            "level_millionths": level,
                            "start_frame": start,
                            "stop_frame": stop,
                        }
                    )
                    invalid_reasons.append("partial_capture")
                    try:
                        verify_live_authority()
                    except Exception as binding_exc:
                        invalid_reasons.append(
                            f"post_capture_binding:{type(binding_exc).__name__}"
                        )
                    raise
                if not np.array_equal(submitted, segment):
                    invalid_reasons.append("submitted_pcm_not_exact_segment")
                segment_raw.append(np.asarray(recorded, dtype=np.int32))
                segment_pcm.append(np.asarray(submitted, dtype=np.int16))
                telemetry = {
                    **dict(telemetry),
                    "level_millionths": level,
                    "start_frame": start,
                    "stop_frame": stop,
                }
                segment_telemetry.append(telemetry)
                verify_live_authority()
                telemetry_reasons = _telemetry_invalid(telemetry)
                invalid_reasons.extend(telemetry_reasons)
                current_peak = float(
                    np.max(np.abs(recorded.astype(np.float64) / float(2**31)))
                )
                decision = next_level_stop_decision(
                    observed_peak=current_peak,
                    current_millionths=level,
                    next_millionths=(
                        int(plan["capture_groups"][index + 1]["level_millionths"])
                        if index + 1 < len(plan["capture_groups"])
                        else None
                    ),
                )
                if telemetry_reasons or decision["stop"]:
                    safety_stop = decision
                    if telemetry_reasons:
                        safety_stop = {
                            **decision,
                            "stop": True,
                            "reasons": [*decision["reasons"], *telemetry_reasons],
                        }
                    invalid_reasons.extend(decision["reasons"])
                    break

    capture_error: BaseException | None = None
    try:
        mpi.capture_with_speaker_release_notice(capture_all)
    except BaseException as exc:
        capture_error = exc
        invalid_reasons.append(f"capture_exception:{type(exc).__name__}")
    # input-only preflight 단계의 실패는 speaker output과 raw가 전혀 없으므로 기존
    # 예외를 그대로 전달한다. 첫 output stream을 열어 session이 생긴 뒤의 모든 실패는
    # prior/partial arrays부터 immutable raw로 보존한다.
    if created is None:
        plan_guard.close()
        assert capture_error is not None
        raise capture_error
    if preflight_raw is None or preflight_report is None:
        plan_guard.close()
        raise RecordingGainLinearityError("live preflight raw/report가 없습니다")
    submitted_all = (
        np.concatenate(segment_pcm, axis=0)
        if segment_pcm
        else np.empty((0, 2), dtype=np.int16)
    )
    recorded_all = (
        np.concatenate(segment_raw, axis=0)
        if segment_raw
        else np.empty((0, 2), dtype=np.int32)
    )
    full_exact = bool(
        submitted_all.shape == pcm.shape
        and recorded_all.shape == pcm.shape
        and np.array_equal(submitted_all, pcm)
        and len(segment_telemetry) == len(plan["capture_groups"])
        and not invalid_reasons
    )
    if not full_exact:
        invalid_reasons.append("capture_not_full_exact_plan")
    metadata = {
        "raw_capture_schema": GAIN_LINEARITY_RAW_SCHEMA,
        "status": "RAW_COMPLETE_NOT_ANALYSED" if full_exact else "INVALID_PARTIAL_RAW",
        "source_commit": expected_commit,
        "repository_execution": execution,
        "hardware": plan["hardware"],
        "plan": {
            "path": loaded["file"]["path"],
            "sha256": loaded["file"]["sha256"],
            "payload_sha256": plan["plan_payload_sha256"],
            "pcm_sha256": plan["output"]["pcm_sha256"],
        },
        "operator_confirmations": confirmations,
        "preflight": preflight_report,
        "segment_telemetry": segment_telemetry,
        "safety_stop": safety_stop,
        "invalid_reasons": list(dict.fromkeys(invalid_reasons)),
        "analysis_status": "NOT_RUN_RAW_FIRST",
        "capture_exception": (
            None
            if capture_error is None
            else f"{type(capture_error).__name__}: {capture_error}"
        ),
    }
    try:
        paths = mpi.write_immutable_raw_capture_atomic(
            created,
            metadata=metadata,
            arrays={
                "submitted_output_pcm_int16": submitted_all,
                "input_raw_int32": recorded_all,
                "preflight_raw_int32": preflight_raw,
            },
        )
    finally:
        plan_guard.close()
    return {"paths": paths, "metadata": metadata, "valid_raw": full_exact}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-live", action="store_true")
    mode.add_argument("--analyze-raw", action="store_true")
    parser.add_argument("--hardware", default=DEFAULT_HARDWARE)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", default=DEFAULT_PLAN)
    parser.add_argument("--plan")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--raw-session-dir")
    parser.add_argument("--raw")
    parser.add_argument("--raw-sha256")
    parser.add_argument("--receipt-out")
    parser.add_argument("--confirm-speaker", action="store_true")
    parser.add_argument("--confirm-user-present", action="store_true")
    parser.add_argument("--confirm-volume-minimum", action="store_true")
    parser.add_argument("--confirm-routing-and-geometry", action="store_true")
    parser.add_argument("--confirm-same-amplifier-setting", action="store_true")
    parser.add_argument("--confirm-bounded-gain-probe", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.analyze_raw:
            if not all((args.raw, args.raw_sha256, args.plan, args.plan_sha256, args.receipt_out)):
                parser.error(
                    "--analyze-raw에는 --raw/--raw-sha256/--plan/--plan-sha256/--receipt-out이 필요합니다"
                )
            loaded = load_gain_linearity_plan(
                repo_root=REPO_ROOT,
                plan_path=_relative(_repo_path(args.plan)),
                expected_sha256=args.plan_sha256,
            )
            if loaded["payload"]["source_commit"] != args.expected_commit.lower():
                raise RecordingGainLinearityError("analysis expected commit이 plan과 다릅니다")
            path, digest, payload = issue_gain_linearity_receipt(
                repo_root=REPO_ROOT,
                output_path=_relative(_repo_path(args.receipt_out)),
                raw_path=_relative(_repo_path(args.raw, results=True)),
                expected_raw_sha256=args.raw_sha256,
                plan_path=loaded["file"]["path"],
                expected_plan_sha256=loaded["file"]["sha256"],
            )
            print(
                json.dumps(
                    {
                        "status": payload["status"],
                        "receipt": _relative(path),
                        "receipt_sha256": digest,
                        "failure_reasons": payload["failure_reasons"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0 if payload["status"] == "PASS" else 1

        environment = _environment(args.hardware)
        plan, _pcm = build_gain_linearity_plan(
            repo_root=REPO_ROOT,
            hardware_path=_relative(environment["hardware_file"]),
            source_commit=args.expected_commit,
            physical_fingerprint=environment["fingerprint"],
        )
        raw_session = _raw_session_for_plan(plan, args.raw_session_dir)
        if args.dry_run:
            output = _repo_path(args.output, results=True)
            digest = _write_plan(output, plan)
            print(
                json.dumps(
                    {
                        "status": "PASS_NO_AUDIO",
                        "plan": _relative(output),
                        "plan_sha256": digest,
                        "raw_session": _relative(raw_session),
                        "audible_seconds": plan["duration"]["audible_nonzero_seconds"],
                        "output_open_seconds": plan["duration"]["output_open_seconds"],
                        "connected_upper_seconds": plan["duration"]["connected_upper_seconds"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            print(
                "[PASS] dry-run: sounddevice import/open 및 speaker 출력 0회.",
                file=sys.stderr,
            )
            return 0

        confirmations = {
            "speaker_output": bool(args.confirm_speaker),
            "user_present": bool(args.confirm_user_present),
            "volume_minimum": bool(args.confirm_volume_minimum),
            "routing_and_geometry": bool(args.confirm_routing_and_geometry),
            "same_amplifier_setting": bool(args.confirm_same_amplifier_setting),
            "bounded_gain_probe": bool(args.confirm_bounded_gain_probe),
        }
        if not all(confirmations.values()) or args.plan is None or args.plan_sha256 is None:
            parser.error("live에는 exact plan SHA와 여섯 확인 플래그가 모두 필요합니다")
        print(
            "[LIVE 직전] NS ch0 only, CS ch1 exact zero. "
            f"audible={plan['duration']['audible_nonzero_seconds']:.3f}s, "
            f"output-open={plan['duration']['output_open_seconds']:.3f}s, "
            "0.40 peak 또는 predictive 0.45에서 즉시 상위 level을 중단합니다.",
            flush=True,
        )
        result = execute_live_capture(
            hardware_path=args.hardware,
            expected_commit=args.expected_commit,
            plan_path=args.plan,
            expected_plan_sha256=args.plan_sha256,
            raw_session_dir=raw_session,
            confirmations=confirmations,
        )
        raw_path = result["paths"]["raw"]
        print(
            json.dumps(
                {
                    "status": "RAW_SAVED" if result["valid_raw"] else "PARTIAL_RAW_SAVED",
                    "raw": _relative(raw_path),
                    "raw_sha256": _sha256_file(raw_path),
                    "analysis": "NOT_RUN_RAW_FIRST",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0 if result["valid_raw"] else 1
    except (
        FileExistsError,
        ImportError,
        KeyError,
        OSError,
        RecordingGainLinearityError,
        RuntimeError,
        TimeoutError,
        ValueError,
    ) as exc:
        print(f"[BLOCKED] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
