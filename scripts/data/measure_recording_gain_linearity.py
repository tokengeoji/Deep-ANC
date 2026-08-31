#!/usr/bin/env python3
"""Source gain v3 bounded ESS/IMD/clock probe를 계획·캡처·분석한다.

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
import io
import json
import os
import signal
import sys
import time
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
    build_gain_linearity_capture_publication_payload,
    build_gain_linearity_plan,
    callback_time_info_evidence,
    issue_gain_linearity_receipt,
    load_gain_linearity_plan,
    next_level_stop_decision,
)
from deep_anc.data.repository_fd import (  # noqa: E402
    RepositoryDirectoryGuard,
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
DEFAULT_PLAN = "results/data_audit/recording_gain_linearity_v3_gainprobe006_plan.json"
DEFAULT_RAW_ROOT = "results/recording_gain_linearity_v3_gainprobe006"
SCRIPT_RELATIVE_PATH = "scripts/data/measure_recording_gain_linearity.py"
LIVE_LOCK_PURPOSE = "recording_gain_linearity_v3_gainprobe006"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(
    value: str | Path,
    *,
    results: bool = False,
    allow_live_recovery: bool = False,
) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise RecordingGainLinearityError("경로는 저장소 안이어야 합니다") from exc
    is_live_recovery = bool(
        len(relative.parts) == 1
        and relative.name.startswith(".deep_anc_live_recovery_")
    )
    if results and (
        not relative.parts
        or (
            relative.parts[0] != "results"
            and not (allow_live_recovery and is_live_recovery)
        )
    ):
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


class _LiveRawSignalGuard:
    """Abort live output immediately, then defer signals until raw commit."""

    def __init__(self) -> None:
        self._watched = tuple(
            value
            for value in (
                getattr(signal, "SIGINT", None),
                getattr(signal, "SIGTERM", None),
                getattr(signal, "SIGHUP", None),
            )
            if value is not None
        )
        self._previous: dict[int, Any] = {}
        self._pending: list[int] = []
        self._defer = False
        self._installed = False

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("live raw signal guard가 이미 설치됐습니다")
        self._previous = {
            int(value): signal.getsignal(value) for value in self._watched
        }

        def handle(signum: int, _frame: Any) -> None:
            self._pending.append(int(signum))
            if self._defer:
                return
            raise mpi.LiveAudioTermination(int(signum))

        for value in self._watched:
            signal.signal(value, handle)
        self._installed = True

    def defer_after_output(self) -> None:
        # capture_with_speaker_release_notice calls this before printing the
        # final output-stop notice, closing the notice→raw-commit signal gap.
        self._defer = True

    def resume_before_output(self) -> None:
        """다음 stream 직전에 pending 종료 요청을 처리하고 live mode로 복귀한다."""

        if self.pending_signal is not None:
            raise mpi.LiveAudioTermination(int(self.pending_signal))
        self._defer = False

    def remember(self, signum: Any) -> None:
        if type(signum) is int and signum > 0:
            self._pending.append(int(signum))

    @property
    def pending_signal(self) -> int | None:
        return None if not self._pending else int(self._pending[0])

    def close(self) -> None:
        if not self._installed:
            return
        for value, previous in self._previous.items():
            signal.signal(value, previous)
        self._installed = False


def _commit_raw_capture_held(
    *,
    session_guard: RepositoryDirectoryGuard,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Serialize once and publish raw/sidecar only through the held dirfd.

    SIGINT/SIGTERM/SIGHUP are deferred until the raw and same-inode SHA are
    durable.  A parent recovery hardlink is retained so a renamed session
    directory cannot make the unique capture unreachable.
    """

    if "metadata_json" in arrays:
        raise ValueError("raw arrays에 reserved metadata_json key가 있습니다")
    safe_metadata = mpi.cw._json_safe(metadata)
    canonical_json = json.dumps(
        safe_metadata, ensure_ascii=False, sort_keys=True
    )
    raw_buffer = io.BytesIO()
    deferred_signal: int | None = None
    raw_ref: dict[str, Any] | None = None
    metadata_ref: dict[str, Any] | None = None
    metadata_error: str | None = None
    publication_ref: dict[str, Any] | None = None
    publication_payload: dict[str, Any] | None = None
    publication_error: str | None = None
    try:
        with mpi.defer_termination_signals_during_raw_commit():
            np.savez_compressed(
                raw_buffer,
                metadata_json=np.asarray(canonical_json),
                **arrays,
            )
            raw_ref = session_guard.publish_bytes_noreplace(
                "raw_measurement.npz",
                raw_buffer.getvalue(),
                preserve_parent_recovery_link=True,
                recovery_tag="gainprobe_v3_raw",
            )
            metadata_bytes = (
                json.dumps(
                    safe_metadata,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            try:
                metadata_ref = session_guard.publish_bytes_noreplace(
                    "metadata.json", metadata_bytes
                )
            except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
                # metadata_json is already embedded in the durable raw.  A
                # sidecar race must not destroy the unique capture or suppress
                # its same-inode SHA, but it does make the live result invalid.
                metadata_error = f"{type(exc).__name__}: {exc}"
            if (
                raw_ref.get("final_published") is True
                and metadata_ref is not None
                and metadata_error is None
                and session_guard.binding_valid()
            ):
                try:
                    publication_payload = (
                        build_gain_linearity_capture_publication_payload(
                            canonical_session_path=session_guard.relative_path,
                            raw_ref=raw_ref,
                            metadata_ref=metadata_ref,
                            metadata=safe_metadata,
                        )
                    )
                    publication_bytes = (
                        json.dumps(
                            publication_payload,
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                            allow_nan=False,
                        )
                        + "\n"
                    ).encode("utf-8")
                    publication_ref = session_guard.publish_bytes_noreplace(
                        "capture_publication.json", publication_bytes
                    )
                    session_guard.verify()
                except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
                    publication_error = f"{type(exc).__name__}: {exc}"
            else:
                publication_error = "raw/metadata canonical publication prerequisite 불충족"
    except mpi.DeferredTerminationSignal as exc:
        deferred_signal = int(exc.signum)
    if raw_ref is None:
        raise RecordingGainLinearityError("held raw publication 결과가 없습니다")
    binding_valid = session_guard.binding_valid()
    selected_relative = (
        raw_ref["path"]
        if binding_valid and raw_ref.get("final_published") is True
        else raw_ref.get("recovery_path")
    )
    if not isinstance(selected_relative, str) or not selected_relative:
        raise RecordingGainLinearityError(
            "session path 변경 뒤 raw recovery path가 없습니다"
        )
    return {
        "paths": {
            "raw": REPO_ROOT / selected_relative,
            "metadata": (
                None
                if metadata_ref is None
                else REPO_ROOT / str(metadata_ref["path"])
            ),
            "publication": (
                None
                if publication_ref is None
                else REPO_ROOT / str(publication_ref["path"])
            ),
        },
        "raw_ref": raw_ref,
        "metadata_ref": metadata_ref,
        "publication_ref": publication_ref,
        "publication_payload": publication_payload,
        "session_binding_valid": binding_valid,
        "metadata_error": metadata_error,
        "publication_error": publication_error,
        "deferred_termination_signal": deferred_signal,
    }


def _global_live_deadline_elapsed(
    *,
    campaign_started: float,
    campaign_deadline: float,
    now: float,
    reserve_seconds: float,
    stage: str,
) -> float:
    """전체 live 창의 단일 monotonic deadline을 검사한다.

    개별 stream watchdog만으로는 stream 사이의 authority 재검증/전환 지연이 누적될 수
    있으므로, 다음 출력에 필요한 시간을 예약할 수 없으면 장치를 열기 전에 중단한다.
    """

    values = (campaign_started, campaign_deadline, now, reserve_seconds)
    if (
        not all(np.isfinite(float(value)) for value in values)
        or campaign_deadline < campaign_started
        or reserve_seconds < 0.0
        or now < campaign_started
        or now + reserve_seconds > campaign_deadline
    ):
        raise TimeoutError(
            "gain-linearity global live deadline 초과/부족: "
            f"stage={stage}, reserve={reserve_seconds:.6f}s"
        )
    return float(now - campaign_started)


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
    monotonic_function: Callable[[], float] = time.monotonic,
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
        raise RecordingGainLinearityError("v3 live 확인 플래그가 모두 필요합니다")
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
    callback_raw_arrays: dict[str, np.ndarray] = {}
    invalid_reasons: list[str] = []
    safety_stop: dict[str, Any] | None = None
    session_guard: RepositoryDirectoryGuard | None = None
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
        if session_guard is not None:
            session_guard.verify()
            for name in (
                "raw_measurement.npz",
                "metadata.json",
                "capture_publication.json",
            ):
                session_guard.assert_leaf_fresh(name)
        assert_live_pcm_clock_preconditions(audio)

    def capture_all() -> None:
        nonlocal preflight_raw, preflight_report, safety_stop, session_guard
        with audio_lock_factory(REPO_ROOT, purpose=LIVE_LOCK_PURPOSE):
            campaign_started = float(monotonic_function())
            campaign_deadline = campaign_started + float(
                plan["duration"]["live_campaign_hard_deadline_seconds"]
            )

            def assert_global_deadline(
                stage: str, *, reserve_seconds: float = 0.0
            ) -> float:
                return _global_live_deadline_elapsed(
                    campaign_started=campaign_started,
                    campaign_deadline=campaign_deadline,
                    now=float(monotonic_function()),
                    reserve_seconds=float(reserve_seconds),
                    stage=stage,
                )

            assert_global_deadline(
                "before_input_preflight", reserve_seconds=INPUT_PREFLIGHT_SECONDS
            )
            assert_live_pcm_clock_preconditions(audio)
            preflight_raw, official_preflight = capture_measurement_preflight_raw(
                sd,
                audio,
                seconds=INPUT_PREFLIGHT_SECONDS,
                absolute_deadline_monotonic=campaign_deadline,
                monotonic_function=monotonic_function,
            )
            assert_global_deadline("after_input_preflight")
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
            session_guard = RepositoryDirectoryGuard.create_fresh(
                REPO_ROOT,
                _relative(raw_session_dir),
                label="gain-linearity live raw session",
            )
            for index, group in enumerate(plan["capture_groups"]):
                # 앞 stream cleanup→partial-array handoff 동안 queue된 종료 요청이
                # 있으면 다음 speaker output을 절대 열지 않는다.
                signal_guard.resume_before_output()
                level = int(group["level_millionths"])
                group_seconds = (
                    int(group["stop_frame"]) - int(group["start_frame"])
                ) / float(audio["sample_rate"])
                assert_global_deadline(
                    f"before_group_{index}_authority",
                    reserve_seconds=(
                        group_seconds
                        + float(
                            plan["duration"]["per_stream_watchdog_grace_seconds"]
                        )
                    ),
                )
                verify_live_authority()
                assert_global_deadline(
                    f"after_group_{index}_authority",
                    reserve_seconds=(
                        group_seconds
                        + float(
                            plan["duration"]["per_stream_watchdog_grace_seconds"]
                        )
                    ),
                )
                start, stop = int(group["start_frame"]), int(group["stop_frame"])
                segment = pcm[start:stop]
                output_float = segment.astype(np.float32) / np.float32(32767.0)

                def pre_open_check() -> None:
                    verify_live_authority()
                    assert_global_deadline(
                        f"group_{index}_pre_open",
                        reserve_seconds=(
                            group_seconds
                            + float(
                                plan["duration"][
                                    "per_stream_watchdog_grace_seconds"
                                ]
                            )
                        ),
                    )

                try:
                    recorded, submitted, telemetry = capture_function(
                        sd,
                        fs=int(audio["sample_rate"]),
                        block_size=int(audio["block_size"]),
                        latency=str(audio["latency"]),
                        in_dev=in_dev,
                        out_dev=out_dev,
                        output_float=output_float,
                        pre_open_check=pre_open_check,
                        record_callback_time_info=True,
                        absolute_deadline_monotonic=campaign_deadline,
                        monotonic_function=monotonic_function,
                        on_output_cleanup_complete=(
                            signal_guard.defer_after_output
                        ),
                    )
                except mpi.PartialCaptureError as exc:
                    count = int(exc.telemetry.get("captured_frames", 0))
                    partial_telemetry = dict(exc.telemetry)
                    partial_callback = partial_telemetry.pop(
                        "callback_time_info", None
                    )
                    try:
                        callback_evidence, callback_arrays = (
                            callback_time_info_evidence(
                                partial_callback,
                                group_index=index,
                                expected_frames=int(segment.shape[0]),
                            )
                        )
                    except (KeyError, TypeError, ValueError):
                        callback_evidence, callback_arrays = None, {}
                    callback_raw_arrays.update(callback_arrays)
                    segment_raw.append(np.asarray(exc.recorded_raw[:count], dtype=np.int32))
                    segment_pcm.append(np.asarray(exc.output_pcm[:count], dtype=np.int16))
                    segment_telemetry.append(
                        {
                            **partial_telemetry,
                            "level_millionths": level,
                            "start_frame": start,
                            "stop_frame": stop,
                            "callback_time_info": callback_evidence,
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
                callback_time_info = telemetry.pop("callback_time_info", None)
                group_integrity_reasons: list[str] = []
                try:
                    callback_evidence, callback_arrays = callback_time_info_evidence(
                        callback_time_info,
                        group_index=index,
                        expected_frames=int(segment.shape[0]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    group_integrity_reasons.append(
                        f"callback_time_witness:{type(exc).__name__}"
                    )
                    callback_evidence = None
                    callback_arrays = {}
                callback_raw_arrays.update(callback_arrays)
                if not np.array_equal(submitted, segment):
                    group_integrity_reasons.append(
                        "submitted_pcm_not_exact_segment"
                    )
                segment_raw.append(np.asarray(recorded, dtype=np.int32))
                segment_pcm.append(np.asarray(submitted, dtype=np.int16))
                telemetry = {
                    **dict(telemetry),
                    "level_millionths": level,
                    "start_frame": start,
                    "stop_frame": stop,
                    "callback_time_info": callback_evidence,
                    "live_campaign_elapsed_seconds": assert_global_deadline(
                        f"group_{index}_captured"
                    ),
                }
                segment_telemetry.append(telemetry)
                verify_live_authority()
                assert_global_deadline(f"group_{index}_post_authority")
                telemetry_reasons = _telemetry_invalid(telemetry)
                group_stop_reasons = [
                    *group_integrity_reasons,
                    *telemetry_reasons,
                ]
                invalid_reasons.extend(group_stop_reasons)
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
                if group_stop_reasons or decision["stop"]:
                    safety_stop = decision
                    if group_stop_reasons:
                        safety_stop = {
                            **decision,
                            "stop": True,
                            "reasons": [
                                *decision["reasons"],
                                *group_stop_reasons,
                            ],
                        }
                    invalid_reasons.extend(decision["reasons"])
                    break
            assert_global_deadline("capture_all_complete")

    signal_guard = _LiveRawSignalGuard()
    signal_guard.install()
    try:
        capture_error: BaseException | None = None

        def capture_and_arm_raw_commit() -> None:
            # 성공/실패 모두 disconnect notice를 출력하기 *전에* signal을 queue-only로
            # 바꾼다. capture 중 신호는 inner stream handler가 즉시 abort한다.
            try:
                capture_all()
            finally:
                signal_guard.defer_after_output()

        try:
            mpi.capture_with_speaker_release_notice(capture_and_arm_raw_commit)
        except BaseException as exc:
            capture_error = exc
            invalid_reasons.append(f"capture_exception:{type(exc).__name__}")
            if isinstance(exc, mpi.LiveAudioTermination):
                signal_guard.remember(int(exc.signum))
            if isinstance(exc, mpi.PartialCaptureError):
                signal_guard.remember(exc.telemetry.get("termination_signal"))
        # input-only preflight 단계의 실패는 speaker output과 raw가 전혀 없으므로 기존
        # 예외를 그대로 전달한다. 첫 output stream을 열어 session이 생긴 뒤의 모든 실패는
        # prior/partial arrays부터 immutable raw로 보존한다.
        if session_guard is None:
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
        for leaf in (
            "raw_measurement.npz",
            "metadata.json",
            "capture_publication.json",
        ):
            try:
                session_guard.assert_leaf_fresh(leaf)
            except (FileExistsError, OSError, RuntimeError, ValueError):
                invalid_reasons.append(f"raw_session_leaf_preclaimed:{leaf}")
        session_binding_valid = session_guard.binding_valid()
        if not session_binding_valid:
            invalid_reasons.append("raw_session_binding_changed")
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
            "status": (
                "RAW_COMPLETE_NOT_ANALYSED" if full_exact else "INVALID_PARTIAL_RAW"
            ),
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
            committed = _commit_raw_capture_held(
                session_guard=session_guard,
                metadata=metadata,
                arrays={
                    "submitted_output_pcm_int16": submitted_all,
                    "input_raw_int32": recorded_all,
                    "preflight_raw_int32": preflight_raw,
                    **callback_raw_arrays,
                },
            )
        finally:
            plan_guard.close()
            session_guard.close()
        publication_valid = bool(
            committed["raw_ref"].get("final_published") is True
            and committed["metadata_error"] is None
            and committed["publication_ref"] is not None
            and committed["publication_error"] is None
            and committed["session_binding_valid"] is True
        )
        # Raw/metadata/publication이 모두 durable해진 뒤 먼저 기존 signal handler를
        # 복원한다. pending을 읽은 뒤 close하는 순서는 그 사이 신호를 조용히 삼킬 수 있다.
        signal_guard.close()
        deferred_signal = (
            signal_guard.pending_signal
            if signal_guard.pending_signal is not None
            else committed["deferred_termination_signal"]
        )
        return {
            "paths": committed["paths"],
            "raw_ref": committed["raw_ref"],
            "metadata_ref": committed["metadata_ref"],
            "publication_ref": committed["publication_ref"],
            "publication_payload": committed["publication_payload"],
            "metadata": metadata,
            "valid_raw": bool(full_exact and publication_valid),
            "publication_valid": publication_valid,
            "metadata_error": committed["metadata_error"],
            "publication_error": committed["publication_error"],
            "deferred_termination_signal": deferred_signal,
        }
    finally:
        signal_guard.close()
        # 첫 output 뒤 metadata/array 조립에서 예상 밖 BaseException이 나도 held
        # dirfd와 plan fd가 다음 측정을 가로막지 않게 한다. 정상 raw commit 경로에서
        # 이미 닫혔어도 두 guard의 close는 idempotent다.
        plan_guard.close()
        if session_guard is not None:
            session_guard.close()


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
    parser.add_argument("--publication")
    parser.add_argument("--publication-sha256")
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
            if not all(
                (
                    args.raw,
                    args.raw_sha256,
                    args.publication,
                    args.publication_sha256,
                    args.plan,
                    args.plan_sha256,
                    args.receipt_out,
                )
            ):
                parser.error(
                    "--analyze-raw에는 raw/plan/publication path+SHA와 --receipt-out이 필요합니다"
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
                raw_path=_relative(
                    _repo_path(
                        args.raw,
                        results=True,
                        allow_live_recovery=True,
                    )
                ),
                expected_raw_sha256=args.raw_sha256,
                plan_path=loaded["file"]["path"],
                expected_plan_sha256=loaded["file"]["sha256"],
                publication_path=_relative(
                    _repo_path(args.publication, results=True)
                ),
                expected_publication_sha256=args.publication_sha256,
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

        if args.dry_run:
            execution = repository_execution_identity(
                REPO_ROOT, SCRIPT_RELATIVE_PATH
            )
            if execution["repository_commit"] != args.expected_commit.lower():
                raise RecordingGainLinearityError(
                    "dry-run repository commit이 외부 expected commit과 다릅니다"
                )
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
                        "nominal_active_seconds": plan["duration"][
                            "nominal_active_seconds"
                        ],
                        "exact_nonzero_pcm_seconds": plan["duration"][
                            "exact_nonzero_pcm_seconds"
                        ],
                        "output_open_seconds": plan["duration"]["output_open_seconds"],
                        "live_campaign_hard_deadline_seconds": plan["duration"][
                            "live_campaign_hard_deadline_seconds"
                        ],
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
            f"exact-nonzero={plan['duration']['exact_nonzero_pcm_seconds']:.3f}s, "
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
                    "raw_sha256": result["raw_ref"]["sha256"],
                    "raw_inode": result["raw_ref"]["inode"],
                    "raw_recovery": result["raw_ref"].get("recovery_path"),
                    "publication": (
                        None
                        if result["paths"]["publication"] is None
                        else _relative(result["paths"]["publication"])
                    ),
                    "publication_sha256": (
                        None
                        if result["publication_ref"] is None
                        else result["publication_ref"]["sha256"]
                    ),
                    "metadata_publication_error": result["metadata_error"],
                    "capture_publication_error": result["publication_error"],
                    "analysis": "NOT_RUN_RAW_FIRST",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        if result["deferred_termination_signal"] is not None:
            signum = int(result["deferred_termination_signal"])
            print(
                f"[중단] signal {signum}은 raw fsync/SHA 고정 뒤 처리했습니다.",
                file=sys.stderr,
            )
            return 128 + signum
        return 0 if result["valid_raw"] else 1
    except mpi.LiveAudioTermination as exc:
        print(f"[중단] signal {exc.signum}; speaker output은 즉시 차단했습니다.", file=sys.stderr)
        return int(exc.exit_code)
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
