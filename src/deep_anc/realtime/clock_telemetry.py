"""PortAudio callback clock/queue 증거를 fail-closed로 수집한다.

이 모듈이 볼 수 있는 것은 *호출된* callback뿐이다. ALSA PortAudio가 callback 전에
capture period를 버리면 Python은 그 사건을 직접 볼 수 없다. 따라서 모든 구조 검사가
정상이어도 authority는 ``PASS``가 아니라 ``INCONCLUSIVE``다. 명시적 timestamp,
status, completion, ring 또는 watchdog 결함이 하나라도 있으면 ``BLOCKED``다.

noise/cancel 두 출력은 하나의 PortAudio output stream/device에 함께 제출된다. 별도
APE ADC와 output DAC 사이의 drift를 두 output channel 사이의 상대 위상 drift로
재해석하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CLOCK_TELEMETRY_SCHEMA = "realtime_clock_telemetry_v1"
CLOCK_RECEIPT_BUNDLE_SCHEMA = "realtime_clock_receipt_bundle_v1"
TIME_INFO_FIELDS = (
    "inputBufferAdcTime",
    "outputBufferDacTime",
    "currentTime",
)
_TIME_FIELD_RECEIPT_NAMES = {
    "inputBufferAdcTime": "input_buffer_adc_time",
    "outputBufferDacTime": "output_buffer_dac_time",
    "currentTime": "callback_current_time",
}
_UNOBSERVABLE_SILENT_DROP = (
    "PortAudio가 callback 호출 전에 버린 capture period는 이 telemetry만으로 0건임을 "
    "증명할 수 없습니다"
)


@dataclass(frozen=True)
class RuntimeCounterSnapshot:
    """callback 완료 시점의 누적 runtime counter와 현재 backlog."""

    xrun_count: int = 0
    deadline_miss_count: int = 0
    engine_error_blocks: int = 0
    input_ring_drop_samples: int = 0
    output_ring_drop_samples: int = 0
    input_ring_overrun_blocks: int = 0
    output_ring_overrun_blocks: int = 0
    input_ring_underrun_blocks: int = 0
    output_ring_underrun_blocks: int = 0
    ring_add_samples: int = 0
    input_backlog_samples: int = 0
    output_backlog_samples: int = 0
    fallback_silence_blocks: int = 0
    watchdog_trip_counts: Mapping[str, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        values = {
            "xrun_count": int(self.xrun_count),
            "deadline_miss_count": int(self.deadline_miss_count),
            "engine_error_blocks": int(self.engine_error_blocks),
            "input_ring_drop_samples": int(self.input_ring_drop_samples),
            "output_ring_drop_samples": int(self.output_ring_drop_samples),
            "input_ring_overrun_blocks": int(self.input_ring_overrun_blocks),
            "output_ring_overrun_blocks": int(self.output_ring_overrun_blocks),
            "input_ring_underrun_blocks": int(self.input_ring_underrun_blocks),
            "output_ring_underrun_blocks": int(self.output_ring_underrun_blocks),
            "ring_add_samples": int(self.ring_add_samples),
            "input_backlog_samples": int(self.input_backlog_samples),
            "output_backlog_samples": int(self.output_backlog_samples),
            "fallback_silence_blocks": int(self.fallback_silence_blocks),
            "watchdog_trip_counts": {
                str(key): int(value)
                for key, value in sorted((self.watchdog_trip_counts or {}).items())
            },
        }
        return values


@dataclass(frozen=True)
class CallbackToken:
    """callback 진입과 완료를 exact하게 연결하는 불투명 token."""

    callback_index: int
    callback_start_frame: int
    callback_frame_count: int
    input_buffer_adc_time: float | None
    output_buffer_dac_time: float | None
    callback_current_time: float | None
    input_step_error_samples: float | None
    output_step_error_samples: float | None
    current_step_error_samples: float | None
    portaudio_status_present: bool
    portaudio_status_text: str
    callback_enter_monotonic_ns: int
    record_slot_reserved: bool


def _time_info_value(time_info: Any, field: str) -> tuple[float | None, str | None]:
    try:
        if isinstance(time_info, Mapping):
            raw = time_info[field]
        else:
            raw = getattr(time_info, field)
    except (AttributeError, KeyError, TypeError):
        return None, "missing"
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None, "not_numeric"
    if not math.isfinite(value):
        return None, "nonfinite"
    return value, None


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def bind_recording_to_clock_receipt(
    telemetry_payload: Mapping[str, Any],
    *,
    recording_path: str | Path,
    recording_sha256: str,
) -> dict[str, Any]:
    """runtime NPZ와 exact telemetry payload를 한 sidecar receipt에 결속한다."""

    digest = str(recording_sha256).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("recording_sha256는 lowercase SHA-256이어야 합니다")
    authority = str(telemetry_payload.get("authority_status", ""))
    if authority not in {"INCONCLUSIVE", "BLOCKED"}:
        raise ValueError("clock authority는 INCONCLUSIVE/BLOCKED만 발행할 수 있습니다")
    payload = dict(telemetry_payload)
    return {
        "schema_version": CLOCK_RECEIPT_BUNDLE_SCHEMA,
        "authority_status": authority,
        "runtime_clock_telemetry_sha256": payload_sha256(payload),
        "recording_npz": str(Path(recording_path)),
        "recording_npz_sha256": digest,
        "runtime_clock_telemetry": payload,
    }


def write_clock_receipt_exclusive(
    path: str | Path, payload: Mapping[str, Any]
) -> tuple[Path, str]:
    """canonical JSON receipt를 O_EXCL no-replace로 쓰고 byte SHA를 반환한다."""

    target = Path(path)
    authority = str(payload.get("authority_status", ""))
    if authority not in {"INCONCLUSIVE", "BLOCKED"}:
        raise ValueError("clock receipt는 false PASS를 발행할 수 없습니다")
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(target, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return target, hashlib.sha256(raw).hexdigest()


class ClockTelemetryRecorder:
    """단일 PortAudio callback thread의 clock/queue witness 수집기."""

    def __init__(
        self,
        *,
        sample_rate: int,
        block_size: int,
        input_device: str,
        output_device: str,
        allowed_input_backlog_samples: int,
        allowed_output_backlog_samples: int,
        frame_step_tolerance_samples: float = 0.5,
        max_callback_records: int = 22_500,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.input_device = str(input_device)
        self.output_device = str(output_device)
        self.allowed_input_backlog_samples = int(allowed_input_backlog_samples)
        self.allowed_output_backlog_samples = int(allowed_output_backlog_samples)
        self.frame_step_tolerance_samples = float(frame_step_tolerance_samples)
        self.max_callback_records = int(max_callback_records)
        if self.sample_rate <= 0 or self.block_size <= 0:
            raise ValueError("clock telemetry sample_rate/block_size는 양수여야 합니다")
        if self.allowed_input_backlog_samples < 0 or self.allowed_output_backlog_samples < 0:
            raise ValueError("허용 ring backlog는 0 이상이어야 합니다")
        if not math.isfinite(self.frame_step_tolerance_samples) or not (
            0.0 <= self.frame_step_tolerance_samples <= 1.0
        ):
            raise ValueError("frame-step tolerance는 0–1 sample이어야 합니다")
        if self.max_callback_records <= 0:
            raise ValueError("max_callback_records는 양수여야 합니다")

        self._next_callback_index = 0
        self._next_callback_frame = 0
        self._pending: dict[int, CallbackToken] = {}
        self._records: list[dict[str, Any]] = []
        self._completed_callbacks = 0
        self._incomplete_callbacks = 0
        self._status_callbacks = 0
        self._records_omitted = 0
        self._callback_deadline_miss_count = 0
        self._maximum_callback_host_duration_ns = 0
        self._issue_counts: dict[str, int] = {}
        self._issue_examples: list[str] = []
        self._last_times: dict[str, tuple[float, int]] = {}
        self._time_stats = {
            field: {
                "finite_count": 0,
                "missing_or_nonfinite_count": 0,
                "strict_monotonic_violation_count": 0,
                "frame_step_violation_count": 0,
                "maximum_absolute_frame_step_error_samples": None,
            }
            for field in TIME_INFO_FIELDS
        }
        self._last_snapshot = RuntimeCounterSnapshot()
        self._maximum_input_backlog = 0
        self._maximum_output_backlog = 0

    def _issue(self, code: str, detail: str) -> None:
        self._issue_counts[code] = self._issue_counts.get(code, 0) + 1
        if len(self._issue_examples) < 64:
            self._issue_examples.append(f"{code}: {detail}")

    def _observe_time(
        self, field: str, value: float | None, frame_count: int, callback_index: int
    ) -> float | None:
        stats = self._time_stats[field]
        if value is None:
            stats["missing_or_nonfinite_count"] += 1
            return None
        stats["finite_count"] += 1
        previous = self._last_times.get(field)
        step_error: float | None = None
        if previous is not None:
            previous_time, previous_frames = previous
            delta = value - previous_time
            if delta <= 0.0:
                stats["strict_monotonic_violation_count"] += 1
                self._issue(
                    f"{field}_not_strict_monotonic",
                    f"callback={callback_index}, delta={delta!r}",
                )
            step_error = delta * self.sample_rate - previous_frames
            maximum = stats["maximum_absolute_frame_step_error_samples"]
            absolute = abs(step_error)
            if maximum is None or absolute > maximum:
                stats["maximum_absolute_frame_step_error_samples"] = absolute
            if absolute > self.frame_step_tolerance_samples:
                stats["frame_step_violation_count"] += 1
                self._issue(
                    f"{field}_unexpected_frame_step",
                    f"callback={callback_index}, error_samples={step_error!r}",
                )
        self._last_times[field] = (value, frame_count)
        return step_error

    def begin_callback(self, *, frames: int, time_info: Any, status: Any) -> CallbackToken:
        """callback 첫 줄에서 호출한다. 잘못된 time_info도 기록하고 예외로 숨기지 않는다."""

        callback_index = self._next_callback_index
        self._next_callback_index += 1
        frame_count = int(frames)
        start_frame = self._next_callback_frame
        if frame_count > 0:
            self._next_callback_frame += frame_count
        else:
            self._issue("nonpositive_callback_frames", f"callback={callback_index}, frames={frames!r}")
        if frame_count != self.block_size:
            self._issue(
                "unexpected_callback_frame_count",
                f"callback={callback_index}, expected={self.block_size}, observed={frame_count}",
            )

        values: dict[str, float | None] = {}
        for field in TIME_INFO_FIELDS:
            value, error = _time_info_value(time_info, field)
            values[field] = value
            if error is not None:
                self._time_stats[field]["missing_or_nonfinite_count"] += 1
                self._issue(
                    f"{field}_{error}", f"callback={callback_index}"
                )

        step_errors: dict[str, float | None] = {}
        for field in TIME_INFO_FIELDS:
            value = values[field]
            if value is None:
                step_errors[field] = None
            else:
                step_errors[field] = self._observe_time(
                    field, value, frame_count, callback_index
                )

        status_present = bool(status)
        status_text = str(status).strip() if status_present else ""
        if status_present:
            self._status_callbacks += 1
            self._issue(
                "portaudio_callback_status",
                f"callback={callback_index}, status={status_text or '<true>'}",
            )

        reserve = len(self._records) + len(self._pending) < self.max_callback_records
        if not reserve:
            self._records_omitted += 1
            self._issue(
                "callback_record_capacity_exceeded",
                f"callback={callback_index}, max={self.max_callback_records}",
            )
        token = CallbackToken(
            callback_index=callback_index,
            callback_start_frame=start_frame,
            callback_frame_count=frame_count,
            input_buffer_adc_time=values["inputBufferAdcTime"],
            output_buffer_dac_time=values["outputBufferDacTime"],
            callback_current_time=values["currentTime"],
            input_step_error_samples=step_errors["inputBufferAdcTime"],
            output_step_error_samples=step_errors["outputBufferDacTime"],
            current_step_error_samples=step_errors["currentTime"],
            portaudio_status_present=status_present,
            portaudio_status_text=status_text,
            callback_enter_monotonic_ns=time.monotonic_ns(),
            record_slot_reserved=reserve,
        )
        self._pending[callback_index] = token
        return token

    def _observe_snapshot(self, snapshot: RuntimeCounterSnapshot) -> dict[str, Any]:
        raw = snapshot.as_dict()
        for name, value in raw.items():
            if name == "watchdog_trip_counts":
                for watchdog, count in value.items():
                    if int(count) < 0:
                        self._issue("negative_runtime_counter", f"{watchdog}={count}")
                continue
            if int(value) < 0:
                self._issue("negative_runtime_counter", f"{name}={value}")
        self._last_snapshot = snapshot
        self._maximum_input_backlog = max(
            self._maximum_input_backlog, int(snapshot.input_backlog_samples)
        )
        self._maximum_output_backlog = max(
            self._maximum_output_backlog, int(snapshot.output_backlog_samples)
        )
        return raw

    def finish_callback(
        self,
        token: CallbackToken,
        *,
        snapshot: RuntimeCounterSnapshot,
        entry_snapshot: RuntimeCounterSnapshot | None = None,
    ) -> None:
        """outdata 작성과 모든 ring/safety 처리가 끝난 뒤 호출한다."""

        pending = self._pending.pop(token.callback_index, None)
        if pending != token:
            self._issue(
                "callback_completion_token_mismatch", f"callback={token.callback_index}"
            )
        completed_ns = time.monotonic_ns()
        duration_ns = max(0, completed_ns - token.callback_enter_monotonic_ns)
        self._maximum_callback_host_duration_ns = max(
            self._maximum_callback_host_duration_ns, duration_ns
        )
        deadline_ns = self.block_size * 1_000_000_000 / self.sample_rate
        if duration_ns >= deadline_ns:
            self._callback_deadline_miss_count += 1
            self._issue(
                "callback_host_deadline_miss",
                f"callback={token.callback_index}, duration_ns={duration_ns}, "
                f"deadline_ns={deadline_ns}",
            )
        entry_counters = None
        if entry_snapshot is not None:
            entry_counters = self._observe_snapshot(entry_snapshot)
        counters = self._observe_snapshot(snapshot)
        self._completed_callbacks += 1
        if token.record_slot_reserved:
            self._records.append(
                self._record(
                    token,
                    completed=True,
                    completed_ns=completed_ns,
                    entry_counters=entry_counters,
                    counters=counters,
                )
            )

    def abort_callback(self, token: CallbackToken, *, error: BaseException) -> None:
        """callback 예외를 incomplete record로 남긴다. 예외를 정상 완료로 바꾸지 않는다."""

        self._pending.pop(token.callback_index, None)
        self._incomplete_callbacks += 1
        self._issue(
            "callback_incomplete",
            f"callback={token.callback_index}, error={type(error).__name__}: {error}",
        )
        if token.record_slot_reserved:
            self._records.append(
                self._record(
                    token,
                    completed=False,
                    completed_ns=time.monotonic_ns(),
                    counters=None,
                    error=f"{type(error).__name__}: {error}",
                )
            )

    @staticmethod
    def _record(
        token: CallbackToken,
        *,
        completed: bool,
        completed_ns: int,
        counters: Mapping[str, Any] | None = None,
        entry_counters: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "callback_index": token.callback_index,
            "callback_start_frame": token.callback_start_frame,
            "callback_frame_count": token.callback_frame_count,
            "callback_end_frame_exclusive": (
                token.callback_start_frame + max(0, token.callback_frame_count)
            ),
            "input_buffer_adc_time": token.input_buffer_adc_time,
            "output_buffer_dac_time": token.output_buffer_dac_time,
            "callback_current_time": token.callback_current_time,
            "input_frame_step_error_samples": token.input_step_error_samples,
            "output_frame_step_error_samples": token.output_step_error_samples,
            "current_frame_step_error_samples": token.current_step_error_samples,
            "portaudio_status_present": token.portaudio_status_present,
            "portaudio_status_text": token.portaudio_status_text,
            "callback_enter_monotonic_ns": token.callback_enter_monotonic_ns,
            "callback_complete_monotonic_ns": completed_ns,
            "callback_host_duration_ns": max(
                0, completed_ns - token.callback_enter_monotonic_ns
            ),
            "completed": bool(completed),
            "error": error,
            "runtime_counters_at_callback_entry": (
                dict(entry_counters) if entry_counters is not None else None
            ),
            "runtime_counters_after_callback": dict(counters) if counters is not None else None,
        }

    def _counter_blockers(self, snapshot: RuntimeCounterSnapshot) -> list[str]:
        blockers: list[str] = []
        raw = snapshot.as_dict()
        zero_fields = (
            "xrun_count",
            "deadline_miss_count",
            "engine_error_blocks",
            "input_ring_drop_samples",
            "output_ring_drop_samples",
            "input_ring_overrun_blocks",
            "output_ring_overrun_blocks",
            "input_ring_underrun_blocks",
            "output_ring_underrun_blocks",
            "ring_add_samples",
            "fallback_silence_blocks",
        )
        for field in zero_fields:
            if int(raw[field]) != 0:
                blockers.append(f"{field}={raw[field]}")
        watchdogs = raw["watchdog_trip_counts"]
        nonzero_watchdogs = {
            key: value for key, value in watchdogs.items() if int(value) != 0
        }
        if nonzero_watchdogs:
            blockers.append(f"watchdog_trip_counts={nonzero_watchdogs}")
        if self._maximum_input_backlog > self.allowed_input_backlog_samples:
            blockers.append(
                "maximum_input_backlog_samples="
                f"{self._maximum_input_backlog}>{self.allowed_input_backlog_samples}"
            )
        if self._maximum_output_backlog > self.allowed_output_backlog_samples:
            blockers.append(
                "maximum_output_backlog_samples="
                f"{self._maximum_output_backlog}>{self.allowed_output_backlog_samples}"
            )
        return blockers

    def live_status(self) -> str:
        """UI용 싼 판정. 정상이어도 silent-drop 한계 때문에 INCONCLUSIVE다."""

        if self._issue_counts or self._counter_blockers(self._last_snapshot):
            return "BLOCKED"
        return "INCONCLUSIVE"

    def build_receipt(
        self, *, final_snapshot: RuntimeCounterSnapshot | None = None
    ) -> dict[str, Any]:
        """전체 callback record를 포함하는 JSON-serializable immutable payload를 만든다."""

        if final_snapshot is not None:
            self._observe_snapshot(final_snapshot)
        issue_counts = dict(self._issue_counts)
        issue_examples = list(self._issue_examples)
        if self._pending:
            for index in sorted(self._pending):
                issue_counts["callback_completion_missing"] = (
                    issue_counts.get("callback_completion_missing", 0) + 1
                )
                if len(issue_examples) < 64:
                    issue_examples.append(
                        f"callback_completion_missing: callback={index}"
                    )
        if self._next_callback_index == 0:
            issue_counts["no_callback_observed"] = 1
            if len(issue_examples) < 64:
                issue_examples.append(
                    "no_callback_observed: callback이 한 번도 완료되지 않았습니다"
                )

        counter_blockers = self._counter_blockers(self._last_snapshot)
        structural_blockers = list(issue_examples)
        structural_blockers.extend(counter_blockers)
        structural_status = "BLOCKED" if structural_blockers else "PASS"
        authority_status = "BLOCKED" if structural_blockers else "INCONCLUSIVE"
        authority_reasons = list(structural_blockers)
        authority_reasons.append(_UNOBSERVABLE_SILENT_DROP)
        return {
            "schema_version": CLOCK_TELEMETRY_SCHEMA,
            "authority_status": authority_status,
            "structural_status": structural_status,
            "authority_reasons": authority_reasons,
            "unobservable_limitations": [_UNOBSERVABLE_SILENT_DROP],
            "sample_rate": self.sample_rate,
            "block_size": self.block_size,
            "frame_step_tolerance_samples": self.frame_step_tolerance_samples,
            "input_device": self.input_device,
            "output_device": self.output_device,
            "clock_semantics": {
                "noise_and_cancel_outputs_share_one_output_stream_device_clock": True,
                "adc_dac_drift_is_not_noise_cancel_relative_output_phase": True,
                "callback_frame_counter_is_application_observed_not_physical_adc_proof": True,
            },
            "counter_semantics": {
                "deadline_miss_count": (
                    "engine.step wall-time이 한 callback block deadline 이상인 횟수"
                ),
                "engine_error_blocks": (
                    "engine.step 예외로 상쇄 출력을 무음으로 대체한 블록 수; "
                    "canonical runtime에서는 exact 0이어야 함"
                ),
                "fallback_silence_blocks": (
                    "callback에서 output ring data가 없어 실제 상쇄 출력을 무음으로 "
                    "채운 횟수; startup/queue starvation도 포함하며 engine.step "
                    "deadline miss 또는 실제 DAC deadline miss와 동일하지 않음"
                ),
                "deadline_and_fallback_are_distinct_not_duplicate_aliases": True,
            },
            "callback_summary": {
                "callback_count": self._next_callback_index,
                "completed_callback_count": self._completed_callbacks,
                "incomplete_callback_count": self._incomplete_callbacks,
                "pending_callback_count": len(self._pending),
                "portaudio_status_callback_count": self._status_callbacks,
                "callback_host_deadline_miss_count": (
                    self._callback_deadline_miss_count
                ),
                "maximum_callback_host_duration_ns": (
                    self._maximum_callback_host_duration_ns
                ),
                "application_observed_frames": self._next_callback_frame,
                "application_observed_seconds": (
                    self._next_callback_frame / self.sample_rate
                ),
                "stored_callback_record_count": len(self._records),
                "omitted_callback_record_count": self._records_omitted,
                "maximum_callback_record_capacity": self.max_callback_records,
            },
            "time_domains": {
                _TIME_FIELD_RECEIPT_NAMES[field]: dict(stats)
                for field, stats in self._time_stats.items()
            },
            "runtime_counters_final": self._last_snapshot.as_dict(),
            "maximum_input_backlog_samples": self._maximum_input_backlog,
            "maximum_output_backlog_samples": self._maximum_output_backlog,
            "allowed_input_backlog_samples": self.allowed_input_backlog_samples,
            "allowed_output_backlog_samples": self.allowed_output_backlog_samples,
            "issue_counts": dict(sorted(issue_counts.items())),
            "issue_examples": issue_examples,
            "callbacks": list(self._records),
        }


__all__ = [
    "CLOCK_RECEIPT_BUNDLE_SCHEMA",
    "CLOCK_TELEMETRY_SCHEMA",
    "CallbackToken",
    "ClockTelemetryRecorder",
    "RuntimeCounterSnapshot",
    "TIME_INFO_FIELDS",
    "bind_recording_to_clock_receipt",
    "payload_sha256",
    "sha256_file",
    "write_clock_receipt_exclusive",
]
