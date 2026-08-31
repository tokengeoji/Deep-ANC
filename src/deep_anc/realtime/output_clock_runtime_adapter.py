"""OutputStream callback과 ref-only scheduler를 잇는 장치 독립 어댑터.

이 모듈은 ``sounddevice``를 import하거나 PCM 장치를 열지 않는다. 실제 PortAudio
통합부가 넘겨 주는 callback 인자와 abort hook만 소비한다. 따라서 현재 모듈 자체는
배포 가능성이나 물리 ANC 성능의 증거가 아니다.

시간축 소유권은 다음처럼 고정한다.

* AB13X ``OutputStream`` callback만 output callback/frame cursor를 전진시킨다.
* 모델 추론은 전용 worker thread에서만 실행하고 ``[digital REF, exact-zero ERR]``를
  :class:`~deep_anc.realtime.output_clock_master.OutputClockMasterScheduler`에 다시
  검증받는다.
* APE ``InputStream`` callback은 immutable raw witness를 보존할 뿐, 모델 입력·추론
  wake-up·출력 pacing에 관여하지 않는다. ADC/DAC timestamp도 서로 비교하지 않는다.
* prime 뒤 결과가 늦거나, status/xrun/drop/add/slip/exception/backlog가 하나라도
  생기면 무음 fallback으로 계속하지 않고 영구 BLOCKED 후 stream abort hook을 부른다.

실제 ``sounddevice.OutputStream``을 여는 얇은 진입점은 canonical checkpoint, lead,
experiment contract, export sidecar와 runtime receipt가 생긴 뒤 별도로 추가해야 한다.
현재 admission의 ``sounddevice_integrated=False``와 ``deployment_eligible=False``를
이 모듈이 변경하거나 우회하지 않는다.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .output_clock_master import (
    OutputClockMasterBlocked,
    OutputClockMasterScheduler,
    OutputClockMasterSchedulerReceipt,
    OutputDiscontinuityCounters,
)


_FROZEN = ConfigDict(frozen=True, extra="forbid")


class OutputClockRuntimeAbort(RuntimeError):
    """영구 fail-closed 뒤 실제 stream adapter가 CallbackAbort로 바꿀 예외."""


class RefOnlyRuntimeEngine(Protocol):
    """canonical DL engine의 최소 인터페이스."""

    def step(self, reference: np.ndarray, error_feature: np.ndarray) -> np.ndarray:
        ...

    def reset(self) -> None:
        ...


class OutputClockRuntimeCounters(BaseModel):
    """서로 다른 실패 원인을 alias하지 않는 최종 exact counter snapshot."""

    model_config = _FROZEN

    output_callback_count: int = Field(default=0, ge=0)
    output_performance_callback_count: int = Field(default=0, ge=0)
    input_witness_callback_count: int = Field(default=0, ge=0)
    inference_step_count: int = Field(default=0, ge=0)

    output_xrun_count: int = Field(default=0, ge=0)
    input_xrun_count: int = Field(default=0, ge=0)
    output_callback_status_count: int = Field(default=0, ge=0)
    input_callback_status_count: int = Field(default=0, ge=0)
    output_callback_deadline_miss_count: int = Field(default=0, ge=0)
    input_callback_deadline_miss_count: int = Field(default=0, ge=0)
    inference_deadline_miss_count: int = Field(default=0, ge=0)
    inference_engine_exception_count: int = Field(default=0, ge=0)
    callback_exception_count: int = Field(default=0, ge=0)
    inference_queue_underflow_count: int = Field(default=0, ge=0)
    inference_queue_overflow_count: int = Field(default=0, ge=0)
    input_witness_queue_overflow_count: int = Field(default=0, ge=0)
    fallback_silence_block_count: int = Field(default=0, ge=0)
    dropped_sample_count: int = Field(default=0, ge=0)
    added_sample_count: int = Field(default=0, ge=0)
    sample_slip_count: int = Field(default=0, ge=0)
    stale_or_reused_control_block_count: int = Field(default=0, ge=0)
    nonzero_error_feature_block_count: int = Field(default=0, ge=0)
    scheduler_lock_contention_count: int = Field(default=0, ge=0)
    callback_reentry_count: int = Field(default=0, ge=0)
    timestamp_missing_count: int = Field(default=0, ge=0)
    watchdog_abort_count: int = Field(default=0, ge=0)
    abort_hook_error_count: int = Field(default=0, ge=0)

    allowed_backlog_samples: Literal[256] = 256
    maximum_absolute_backlog_samples: int = Field(default=0, ge=0)
    maximum_excess_backlog_samples: int = Field(default=0, ge=0)

    def violations(self) -> dict[str, int]:
        informational = {
            "output_callback_count",
            "output_performance_callback_count",
            "input_witness_callback_count",
            "inference_step_count",
            "allowed_backlog_samples",
            "maximum_absolute_backlog_samples",
        }
        out = {
            key: int(value)
            for key, value in self.model_dump().items()
            if key not in informational and int(value) != 0
        }
        if self.maximum_absolute_backlog_samples > self.allowed_backlog_samples:
            out["maximum_absolute_backlog_samples"] = int(
                self.maximum_absolute_backlog_samples
            )
        return out


class InputWitnessFrame(BaseModel):
    """APE input의 raw-only immutable witness 한 블록."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    input_callback_index: int = Field(ge=0)
    input_frame_start: int = Field(ge=0)
    input_frame_stop: int = Field(gt=0)
    input_adc_time_seconds: float
    raw_dtype: str
    raw_shape: tuple[int, int]
    raw_payload_sha256: str
    raw: np.ndarray

    @model_validator(mode="after")
    def _validate_raw_identity(self) -> "InputWitnessFrame":
        if self.input_frame_stop - self.input_frame_start != self.raw_shape[0]:
            raise ValueError("input witness frame range와 raw shape가 다릅니다")
        if self.raw.shape != self.raw_shape or str(self.raw.dtype) != self.raw_dtype:
            raise ValueError("input witness raw dtype/shape identity가 다릅니다")
        digest = hashlib.sha256(np.ascontiguousarray(self.raw).tobytes()).hexdigest()
        if digest != self.raw_payload_sha256:
            raise ValueError("input witness raw payload SHA가 다릅니다")
        if self.raw.flags.writeable:
            raise ValueError("input witness raw는 immutable이어야 합니다")
        return self


class OutputClockRuntimeAdapterReceipt(BaseModel):
    """mock/device-independent integration 증거. 물리 배포 authority는 항상 false."""

    model_config = _FROZEN

    schema_version: Literal["output_clock_runtime_adapter_receipt_v1"] = (
        "output_clock_runtime_adapter_receipt_v1"
    )
    authority: Literal[
        "device_independent_adapter_only_not_physical_performance"
    ] = "device_independent_adapter_only_not_physical_performance"
    scheduler_receipt: OutputClockMasterSchedulerReceipt
    counters: OutputClockRuntimeCounters
    output_clock_owner: Literal["ab13x_outputstream_callback"] = (
        "ab13x_outputstream_callback"
    )
    inference_execution_context: Literal[
        "dedicated_worker_not_output_callback"
    ] = "dedicated_worker_not_output_callback"
    input_role: Literal[
        "raw_safety_eval_witness_only_not_model_or_output_pacing"
    ] = "raw_safety_eval_witness_only_not_model_or_output_pacing"
    cross_clock_timestamp_alignment_used: Literal[False] = False
    sounddevice_imported_or_stream_opened: Literal[False] = False
    physical_performance_pass: Literal[False] = False
    deployment_eligible: Literal[False] = False

    @model_validator(mode="after")
    def _validate_structural_only(self) -> "OutputClockRuntimeAdapterReceipt":
        violations = self.counters.violations()
        if violations:
            raise ValueError(f"adapter receipt에 runtime violation이 있습니다: {violations}")
        return self


@dataclass
class _MutableCounters:
    output_callback_count: int = 0
    output_performance_callback_count: int = 0
    input_witness_callback_count: int = 0
    inference_step_count: int = 0
    output_xrun_count: int = 0
    input_xrun_count: int = 0
    output_callback_status_count: int = 0
    input_callback_status_count: int = 0
    output_callback_deadline_miss_count: int = 0
    input_callback_deadline_miss_count: int = 0
    inference_deadline_miss_count: int = 0
    inference_engine_exception_count: int = 0
    callback_exception_count: int = 0
    inference_queue_underflow_count: int = 0
    inference_queue_overflow_count: int = 0
    input_witness_queue_overflow_count: int = 0
    fallback_silence_block_count: int = 0
    dropped_sample_count: int = 0
    added_sample_count: int = 0
    sample_slip_count: int = 0
    stale_or_reused_control_block_count: int = 0
    nonzero_error_feature_block_count: int = 0
    scheduler_lock_contention_count: int = 0
    callback_reentry_count: int = 0
    timestamp_missing_count: int = 0
    watchdog_abort_count: int = 0
    abort_hook_error_count: int = 0
    maximum_absolute_backlog_samples: int = 0
    maximum_excess_backlog_samples: int = 0


class OutputClockMasterRuntimeAdapter:
    """주입된 mock/PortAudio callback을 위한 fail-closed runtime core.

    이 클래스는 stream을 만들지 않는다. 실제 통합부는 callback 예외를
    ``sounddevice.CallbackAbort``로 변환하고 ``abort_stream``이 실제 stream을
    abort/close하도록 결속해야 한다.
    """

    def __init__(
        self,
        *,
        scheduler: OutputClockMasterScheduler,
        engine: RefOnlyRuntimeEngine,
        future_source: Callable[[int, int, int], np.ndarray],
        abort_stream: Callable[[str], None],
        input_channels: int = 2,
        input_witness_capacity_blocks: int = 4096,
        active_anc_gain: float = 1.0,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if abort_stream is None or not callable(abort_stream):
            raise ValueError("실제 stream abort hook이 필수입니다")
        if int(input_channels) <= 0:
            raise ValueError("input_channels는 양수여야 합니다")
        if int(input_witness_capacity_blocks) <= 0:
            raise ValueError("input witness capacity는 양수여야 합니다")
        if not math.isfinite(float(active_anc_gain)) or not 0.0 <= float(
            active_anc_gain
        ) <= 1.0:
            raise ValueError("active_anc_gain은 유한한 [0,1]이어야 합니다")
        admission = scheduler.admission
        if admission.sounddevice_integrated is not False:
            raise ValueError("현재 structural admission은 sounddevice_integrated=false여야 합니다")
        if admission.deployment_eligible is not False:
            raise ValueError("현재 structural admission은 deployment_eligible=false여야 합니다")

        self.scheduler = scheduler
        self.engine = engine
        self.future_source = future_source
        self.abort_stream = abort_stream
        self.input_channels = int(input_channels)
        self.input_witness_capacity_blocks = int(input_witness_capacity_blocks)
        self.active_anc_gain = float(active_anc_gain)
        self.clock_ns = clock_ns
        self.sample_rate = int(admission.sample_rate)
        self.block_size = int(admission.block_size)
        self.block_deadline_ns = int(
            self.block_size * 1_000_000_000 // self.sample_rate
        )

        self._scheduler_lock = threading.Lock()
        self._callback_gate = threading.Lock()
        self._counter_lock = threading.Lock()
        self._block_lock = threading.Lock()
        self._input_witness_lock = threading.Lock()
        self._counters = _MutableCounters()
        self._blocked_reason: str | None = None
        self._abort_hook_called = False
        self._anc_requested = False
        self._prime_pending = True
        self._next_output_callback_index = 0
        self._next_output_frame_start = 0
        self._next_input_callback_index = 0
        self._next_input_frame_start = 0
        self._previous_output_dac_time: float | None = None
        self._previous_input_adc_time: float | None = None
        self._input_witness: deque[InputWitnessFrame] = deque()
        self._job_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._worker_thread_id: int | None = None
        self._output_callback_thread_id: int | None = None

    @property
    def blocked_reason(self) -> str | None:
        with self._block_lock:
            return self._blocked_reason

    @property
    def worker_alive(self) -> bool:
        thread = self._worker_thread
        return bool(thread is not None and thread.is_alive())

    def _increment(self, **values: int) -> None:
        with self._counter_lock:
            for name, delta in values.items():
                current = int(getattr(self._counters, name))
                setattr(self._counters, name, current + int(delta))

    def _set_backlog(self, absolute_samples: int) -> None:
        absolute = max(0, int(absolute_samples))
        allowed = self.block_size
        with self._counter_lock:
            self._counters.maximum_absolute_backlog_samples = max(
                self._counters.maximum_absolute_backlog_samples, absolute
            )
            excess = max(0, absolute - allowed)
            self._counters.maximum_excess_backlog_samples = max(
                self._counters.maximum_excess_backlog_samples, excess
            )

    def counters_snapshot(self) -> OutputClockRuntimeCounters:
        with self._counter_lock:
            payload = vars(self._counters).copy()
        return OutputClockRuntimeCounters(
            **payload,
            allowed_backlog_samples=self.block_size,
        )

    def _raise_if_blocked(self) -> None:
        reason = self.blocked_reason
        if reason is not None:
            raise OutputClockRuntimeAbort(reason)

    def _trip(self, reason: str, **counter_changes: int) -> None:
        first = False
        with self._block_lock:
            if self._blocked_reason is None:
                self._blocked_reason = str(reason)
                first = True
        if first:
            if counter_changes:
                self._increment(**counter_changes)
            self._stop_event.set()
            self._job_event.set()
            try:
                self.abort_stream(str(reason))
            except BaseException:
                self._increment(abort_hook_error_count=1)
            self._abort_hook_called = True
        raise OutputClockRuntimeAbort(str(self._blocked_reason))

    @staticmethod
    def _status_has_xrun(status: Any, *, output: bool) -> bool:
        names = (
            ("output_underflow", "output_overflow")
            if output
            else ("input_underflow", "input_overflow")
        )
        return any(bool(getattr(status, name, False)) for name in names)

    @staticmethod
    def _timestamp(time_info: Any, key: str) -> float | None:
        value: Any
        if isinstance(time_info, Mapping):
            value = time_info.get(key)
        else:
            value = getattr(time_info, key, None)
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    def _check_local_clock_step(
        self,
        *,
        current: float,
        previous: float | None,
        label: str,
    ) -> None:
        if previous is None:
            return
        samples_float = (current - previous) * self.sample_rate
        nearest = int(round(samples_float))
        # PortAudio의 double timestamp 반올림만 허용한다. ADC와 DAC를 서로 비교하지
        # 않고 각 stream 내부의 frame step만 확인한다.
        if nearest != self.block_size or abs(samples_float - nearest) > 0.25:
            delta = nearest - self.block_size
            changes: dict[str, int] = {"sample_slip_count": 1}
            if delta < 0:
                changes["dropped_sample_count"] = abs(delta)
            elif delta > 0:
                changes["added_sample_count"] = delta
            self._trip(
                f"{label} callback timestamp sample slip: "
                f"observed={samples_float:.9f}, expected={self.block_size}",
                **changes,
            )

    def start_worker(self) -> None:
        self._raise_if_blocked()
        if self.worker_alive:
            raise RuntimeError("inference worker는 한 번만 시작할 수 있습니다")
        if self._worker_thread is not None:
            raise RuntimeError("종료된 worker를 재사용할 수 없습니다")
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="anc-output-clock-inference",
            daemon=True,
        )
        self._worker_thread.start()

    def stop_worker(self, *, timeout_seconds: float = 1.0) -> None:
        thread = self._worker_thread
        if thread is None:
            return
        self._stop_event.set()
        self._job_event.set()
        thread.join(timeout=float(timeout_seconds))
        if thread.is_alive():
            self._trip("inference worker stop watchdog timeout", watchdog_abort_count=1)

    def _worker_loop(self) -> None:
        self._worker_thread_id = threading.get_ident()
        while not self._stop_event.is_set():
            self._job_event.wait(timeout=0.1)
            self._job_event.clear()
            if self._stop_event.is_set():
                break
            try:
                self._process_one_inference_job()
            except OutputClockRuntimeAbort:
                break

    def _process_one_inference_job(self) -> bool:
        """오직 dedicated worker context에서 pending job 하나를 처리한다."""

        self._raise_if_blocked()
        if threading.get_ident() != self._worker_thread_id:
            self._trip(
                "engine.step이 dedicated worker thread 밖에서 호출되었습니다",
                callback_exception_count=1,
            )

        with self._scheduler_lock:
            try:
                job = self.scheduler.claim_inference_job()
            except OutputClockMasterBlocked as exc:
                self._trip(str(exc), inference_queue_underflow_count=1)
        if job is None:
            return False

        self._set_backlog(self.block_size)
        if np.any(job.error_feature != np.float32(0.0)):
            self._trip(
                "worker job의 ERR feature가 exact zero가 아닙니다",
                nonzero_error_feature_block_count=1,
            )
        t0 = int(self.clock_ns())
        try:
            control = self.engine.step(job.reference, job.error_feature)
        except BaseException as exc:
            self._trip(
                f"engine.step exception: {type(exc).__name__}: {exc}",
                inference_engine_exception_count=1,
            )
        t1 = int(self.clock_ns())
        elapsed = t1 - t0
        if elapsed < 0:
            self._trip("inference monotonic clock가 역행했습니다", watchdog_abort_count=1)
        if elapsed >= self.block_deadline_ns:
            self._trip(
                f"inference deadline miss: elapsed_ns={elapsed}, "
                f"deadline_ns={self.block_deadline_ns}",
                inference_deadline_miss_count=1,
            )

        try:
            control_array = np.asarray(control)
            with self._scheduler_lock:
                self.scheduler.submit_inference_result(
                    job_id=job.job_id,
                    source_callback_index=job.source_callback_index,
                    target_output_callback_index=job.target_output_callback_index,
                    reference_used=job.reference,
                    error_feature_used=job.error_feature,
                    control=control_array,
                )
        except OutputClockMasterBlocked as exc:
            message = str(exc)
            changes: dict[str, int] = {}
            if "stale/reused/unknown" in message:
                changes["stale_or_reused_control_block_count"] = 1
            elif "exact zero" in message:
                changes["nonzero_error_feature_block_count"] = 1
            else:
                changes["inference_queue_overflow_count"] = 1
            self._trip(message, **changes)
        except BaseException as exc:
            self._trip(
                f"inference result adapter exception: {type(exc).__name__}: {exc}",
                callback_exception_count=1,
            )
        self._increment(inference_step_count=1)
        self._set_backlog(self.block_size)
        return True

    def request_anc_on(self) -> None:
        self._raise_if_blocked()
        if self._anc_requested:
            return
        if not self.worker_alive:
            self._trip("ANC ON 전에 dedicated inference worker가 살아 있지 않습니다", watchdog_abort_count=1)
        with self._scheduler_lock:
            try:
                self.scheduler.request_anc_on()
            except OutputClockMasterBlocked as exc:
                self._trip(str(exc), inference_queue_overflow_count=1)
        self._anc_requested = True
        self._prime_pending = True

    def request_anc_off(self) -> None:
        self._raise_if_blocked()
        if not self._anc_requested:
            return
        with self._scheduler_lock:
            try:
                self.scheduler.request_anc_off()
            except OutputClockMasterBlocked as exc:
                self._trip(str(exc), inference_queue_overflow_count=1)
        self._anc_requested = False
        self._prime_pending = True

    def request_reset(self) -> None:
        self._raise_if_blocked()
        with self._scheduler_lock:
            try:
                self.scheduler.request_reset()
                self.engine.reset()
            except OutputClockMasterBlocked as exc:
                self._trip(str(exc), inference_queue_overflow_count=1)
            except BaseException as exc:
                self._trip(
                    f"engine.reset exception: {type(exc).__name__}: {exc}",
                    inference_engine_exception_count=1,
                )
        self._prime_pending = True

    def output_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """향후 ``sounddevice.OutputStream`` callback에 직접 넘길 장치 독립 본체."""

        callback_start = int(self.clock_ns())
        try:
            out = np.asarray(outdata)
            if out.shape == (self.block_size, 2):
                out.fill(0)
            self._raise_if_blocked()
            if not self._callback_gate.acquire(blocking=False):
                self._trip("output callback reentry", callback_reentry_count=1)
            try:
                callback_thread_id = threading.get_ident()
                if self._output_callback_thread_id is None:
                    self._output_callback_thread_id = callback_thread_id
                elif callback_thread_id != self._output_callback_thread_id:
                    self._trip(
                        "output callback owner thread가 실행 중 변경되었습니다",
                        callback_reentry_count=1,
                    )
                if bool(status):
                    changes = {"output_callback_status_count": 1}
                    if self._status_has_xrun(status, output=True):
                        changes["output_xrun_count"] = 1
                    self._trip(f"output callback status: {status}", **changes)
                if int(frames) != self.block_size:
                    delta = int(frames) - self.block_size
                    changes = {"sample_slip_count": 1}
                    if delta < 0:
                        changes["dropped_sample_count"] = abs(delta)
                    else:
                        changes["added_sample_count"] = delta
                    self._trip(
                        f"output callback frames={frames}; expected={self.block_size}",
                        **changes,
                    )
                if out.shape != (self.block_size, 2) or out.dtype != np.int16:
                    self._trip(
                        f"output buffer contract mismatch: shape={out.shape}, dtype={out.dtype}",
                        callback_exception_count=1,
                    )
                dac_time = self._timestamp(time_info, "outputBufferDacTime")
                if dac_time is None:
                    self._trip(
                        "outputBufferDacTime이 없습니다",
                        timestamp_missing_count=1,
                    )
                assert dac_time is not None
                self._check_local_clock_step(
                    current=dac_time,
                    previous=self._previous_output_dac_time,
                    label="output",
                )

                if not self._scheduler_lock.acquire(blocking=False):
                    self._trip(
                        "output callback 시 scheduler lock contention",
                        scheduler_lock_contention_count=1,
                        inference_queue_underflow_count=1,
                    )
                try:
                    callback_index = self._next_output_callback_index
                    frame_start = self._next_output_frame_start
                    source = self.future_source(
                        callback_index, frame_start, self.block_size
                    )
                    gain = (
                        0.0
                        if (not self._anc_requested or self._prime_pending)
                        else self.active_anc_gain
                    )
                    block = self.scheduler.output_callback(
                        callback_index=callback_index,
                        global_output_frame_start=frame_start,
                        future_source=np.asarray(source),
                        anc_gain=gain,
                    )
                except OutputClockMasterBlocked as exc:
                    message = str(exc)
                    changes: dict[str, int] = {}
                    if "underflow/late" in message:
                        changes["inference_queue_underflow_count"] = 1
                    elif "overflow" in message:
                        changes["inference_queue_overflow_count"] = 1
                    elif "drop/add/reorder" in message or "sample slip" in message:
                        changes["sample_slip_count"] = 1
                    else:
                        changes["callback_exception_count"] = 1
                    self._trip(message, **changes)
                except BaseException as exc:
                    self._trip(
                        f"output adapter exception: {type(exc).__name__}: {exc}",
                        callback_exception_count=1,
                    )
                finally:
                    self._scheduler_lock.release()

                out[:, :] = block.stereo_pcm_s16
                self._previous_output_dac_time = dac_time
                self._next_output_callback_index += 1
                self._next_output_frame_start += self.block_size
                self._increment(
                    output_callback_count=1,
                    output_performance_callback_count=int(
                        block.receipt.performance_window_included
                    ),
                )
                if block.receipt.protocol_prime:
                    self._prime_pending = False
                self._set_backlog(self.block_size if self._anc_requested else 0)
                if self._anc_requested:
                    self._job_event.set()
            finally:
                self._callback_gate.release()
        except OutputClockRuntimeAbort:
            try:
                np.asarray(outdata).fill(0)
            except BaseException:
                pass
            raise
        except BaseException as exc:
            self._trip(
                f"unhandled output callback exception: {type(exc).__name__}: {exc}",
                callback_exception_count=1,
            )

        elapsed = int(self.clock_ns()) - callback_start
        if elapsed < 0:
            self._trip("output callback monotonic clock가 역행했습니다", watchdog_abort_count=1)
        if elapsed >= self.block_deadline_ns:
            np.asarray(outdata).fill(0)
            self._trip(
                f"output callback deadline miss: elapsed_ns={elapsed}, "
                f"deadline_ns={self.block_deadline_ns}",
                output_callback_deadline_miss_count=1,
            )

    def input_witness_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """APE input raw를 보존하되 추론·출력 pacing에는 절대로 사용하지 않는다."""

        callback_start = int(self.clock_ns())
        self._raise_if_blocked()
        if bool(status):
            changes = {"input_callback_status_count": 1}
            if self._status_has_xrun(status, output=False):
                changes["input_xrun_count"] = 1
            self._trip(f"input callback status: {status}", **changes)
        if int(frames) != self.block_size:
            delta = int(frames) - self.block_size
            changes = {"sample_slip_count": 1}
            if delta < 0:
                changes["dropped_sample_count"] = abs(delta)
            else:
                changes["added_sample_count"] = delta
            self._trip(
                f"input callback frames={frames}; expected={self.block_size}",
                **changes,
            )
        raw = np.asarray(indata)
        if raw.ndim != 2 or raw.shape != (self.block_size, self.input_channels):
            self._trip(
                f"input witness shape mismatch: {raw.shape}",
                callback_exception_count=1,
            )
        adc_time = self._timestamp(time_info, "inputBufferAdcTime")
        if adc_time is None:
            self._trip("inputBufferAdcTime이 없습니다", timestamp_missing_count=1)
        assert adc_time is not None
        self._check_local_clock_step(
            current=adc_time,
            previous=self._previous_input_adc_time,
            label="input",
        )
        immutable = np.frombuffer(
            np.ascontiguousarray(raw).tobytes(order="C"), dtype=raw.dtype
        ).reshape(raw.shape)
        overflow = False
        with self._input_witness_lock:
            if len(self._input_witness) >= self.input_witness_capacity_blocks:
                overflow = True
            else:
                frame_start = self._next_input_frame_start
                self._input_witness.append(
                    InputWitnessFrame(
                        input_callback_index=self._next_input_callback_index,
                        input_frame_start=frame_start,
                        input_frame_stop=frame_start + self.block_size,
                        input_adc_time_seconds=adc_time,
                        raw_dtype=str(raw.dtype),
                        raw_shape=(self.block_size, self.input_channels),
                        raw_payload_sha256=hashlib.sha256(
                            immutable.tobytes()
                        ).hexdigest(),
                        raw=immutable,
                    )
                )
        if overflow:
            self._trip(
                "input raw witness queue overflow",
                input_witness_queue_overflow_count=1,
            )
        self._previous_input_adc_time = adc_time
        self._next_input_callback_index += 1
        self._next_input_frame_start += self.block_size
        self._increment(input_witness_callback_count=1)
        elapsed = int(self.clock_ns()) - callback_start
        if elapsed < 0:
            self._trip("input callback monotonic clock가 역행했습니다", watchdog_abort_count=1)
        if elapsed >= self.block_deadline_ns:
            self._trip(
                f"input callback deadline miss: elapsed_ns={elapsed}, "
                f"deadline_ns={self.block_deadline_ns}",
                input_callback_deadline_miss_count=1,
            )

    def drain_input_witness(self) -> tuple[InputWitnessFrame, ...]:
        """raw writer가 호출한다. 출력 callback/worker에는 어떤 신호도 보내지 않는다."""

        with self._input_witness_lock:
            frames = tuple(self._input_witness)
            self._input_witness.clear()
        return frames

    def watchdog_abort(self, reason: str) -> None:
        self._trip(f"runtime watchdog: {reason}", watchdog_abort_count=1)

    def close_evidence_window(self) -> OutputClockRuntimeAdapterReceipt:
        self._raise_if_blocked()
        self.stop_worker()
        counters = self.counters_snapshot()
        violations = counters.violations()
        if violations:
            self._trip(f"runtime counters가 0 계약을 위반했습니다: {violations}")
        scheduler_counters = OutputDiscontinuityCounters(
            output_xrun_count=counters.output_xrun_count + counters.input_xrun_count,
            output_callback_status_count=(
                counters.output_callback_status_count
                + counters.input_callback_status_count
            ),
            deadline_miss_count=(
                counters.output_callback_deadline_miss_count
                + counters.input_callback_deadline_miss_count
                + counters.inference_deadline_miss_count
            ),
            inference_queue_underflow_count=counters.inference_queue_underflow_count,
            inference_queue_overflow_count=(
                counters.inference_queue_overflow_count
                + counters.input_witness_queue_overflow_count
            ),
            fallback_silence_block_count=counters.fallback_silence_block_count,
            dropped_sample_count=counters.dropped_sample_count,
            added_sample_count=counters.added_sample_count,
            sample_slip_count=counters.sample_slip_count,
            stale_or_reused_control_block_count=(
                counters.stale_or_reused_control_block_count
            ),
            nonzero_error_feature_block_count=(
                counters.nonzero_error_feature_block_count
            ),
            maximum_absolute_backlog_samples=(
                counters.maximum_absolute_backlog_samples
            ),
            maximum_excess_backlog_samples=counters.maximum_excess_backlog_samples,
        )
        try:
            with self._scheduler_lock:
                scheduler_receipt = self.scheduler.close_evidence_window(
                    discontinuity_counters=scheduler_counters
                )
        except OutputClockMasterBlocked as exc:
            self._trip(str(exc), callback_exception_count=1)
        return OutputClockRuntimeAdapterReceipt(
            scheduler_receipt=scheduler_receipt,
            counters=counters,
        )


__all__ = [
    "InputWitnessFrame",
    "OutputClockMasterRuntimeAdapter",
    "OutputClockRuntimeAbort",
    "OutputClockRuntimeAdapterReceipt",
    "OutputClockRuntimeCounters",
    "RefOnlyRuntimeEngine",
]
