"""AB13X 출력 클록 단일 시간축을 위한 순수 ref-only 스케줄러 기반.

이 모듈은 ``sounddevice``를 import하거나 장치를 열지 않는다. 실제 출력 콜백은
여기서 만들어진 S16 stereo 블록을 제출하는 얇은 어댑터여야 하고, 모델 추론은
반드시 :meth:`OutputClockMasterScheduler.claim_inference_job`을 소비하는 별도 worker가
수행해야 한다.

시간축 규약은 다음 하나뿐이다.

* 출력 callback ``k``가 future digital source ``U_k``를 생성/등록한다.
* worker는 ``[U_k, exact-zero ERR]``만 모델에 넣는다.
* 그 결과 ``y_k``는 정확히 callback ``k + 1``의 CS 채널에만 제출된다.

callback 0은 이 1-block handoff를 채우는 protocol-defined ANC-OFF prime이다.
성능 구간이나 fallback으로 세지 않는다. reset 또는 ANC OFF→ON 뒤에도 같은 prime을
다시 거쳐야 한다. prime 다음 callback에 결과가 준비되지 않았다면 silence fallback을
내지 않고 영구 BLOCKED가 된다.

주의
----
검증된 admission과 scheduler receipt는 구조적 준비 증거일 뿐이다. 실제 AB13X/APE
통합, ADC↔DAC clock witness, 안전 watchdog 또는 물리 감쇠 PASS를 주장하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..audio_io import float32_to_pcm_int16
from ..dsp.control_band_contract import BroadbandFullOctaveContractV3
from ..dsp.timing import TrainingTimingContract
from .noise_gen import DigitalReferenceBuffer


_FROZEN = ConfigDict(frozen=True, extra="forbid")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BLOCK_SIZE = 256
_SAMPLE_RATE = 48_000
_EQUIVALENCE_TOLERANCE = 1.0e-5


class OutputClockMasterBlocked(RuntimeError):
    """fail-closed 규약 위반으로 scheduler/admission을 사용할 수 없음."""


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if _SHA256_RE.fullmatch(str(value)) is None:
        raise ValueError(f"{name} 는 lowercase SHA-256이어야 합니다")


def _raw_array_sha256(array: np.ndarray) -> str:
    """배열의 실제 contiguous payload byte SHA.

    dtype/shape는 receipt의 별도 필드에 고정한다. 이 함수는 실제 PortAudio에 넘길
    S16 payload와 모델에 넘길 float32 payload byte를 숨은 재양자화 없이 결속한다.
    """

    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    # immutable bytes를 backing store로 써서 caller가 job/output을 바꾸지 못하게 한다.
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _float32_block(name: str, value: np.ndarray, block_size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != (block_size,):
        raise ValueError(
            f"{name} 는 shape=({block_size},), dtype=float32여야 합니다: "
            f"{array.shape}/{array.dtype}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 에 NaN/Inf가 있습니다")
    return np.ascontiguousarray(array)


def _gain_block(value: float | np.ndarray, block_size: int) -> np.ndarray:
    if np.isscalar(value):
        scalar = float(value)
        gain = np.full(block_size, scalar, dtype=np.float32)
    else:
        gain = _float32_block("anc_gain", np.asarray(value), block_size)
    if not np.all(np.isfinite(gain)) or np.any(gain < 0.0) or np.any(gain > 1.0):
        raise ValueError("anc_gain은 모든 sample에서 유한한 [0, 1]이어야 합니다")
    return np.ascontiguousarray(gain, dtype=np.float32)


class CanonicalErrZeroReceipt(BaseModel):
    """dropout 확률 대신 실제 canonical train population 전부가 ERR=0임을 증명."""

    model_config = _FROZEN

    schema_version: Literal["canonical_train_err_exact_zero_receipt_v1"] = (
        "canonical_train_err_exact_zero_receipt_v1"
    )
    canonical_population_sha256: str
    item_receipts_sha256: str
    item_count: int = Field(gt=0)
    total_error_feature_samples: int = Field(gt=0)
    nonzero_error_feature_sample_count: Literal[0] = 0
    maximum_absolute_error_feature: Literal[0.0] = 0.0

    @model_validator(mode="after")
    def _validate_receipt(self) -> "CanonicalErrZeroReceipt":
        _require_sha256(
            "canonical_population_sha256", self.canonical_population_sha256
        )
        _require_sha256("item_receipts_sha256", self.item_receipts_sha256)
        return self

    def digest(self) -> str:
        return _canonical_digest(self.model_dump(mode="json"))


class RefOnlyModelInputContract(BaseModel):
    """APE 입력을 모델 feature에서 제거한 exact 2-channel compatibility 규약."""

    model_config = _FROZEN

    schema_version: Literal["digital_reference_err_zero_input_v1"] = (
        "digital_reference_err_zero_input_v1"
    )
    mode: Literal["digital_reference_only_err_exact_zero"] = (
        "digital_reference_only_err_exact_zero"
    )
    model_channel_order: tuple[
        Literal["digital_reference"], Literal["error_exact_zero"]
    ] = ("digital_reference", "error_exact_zero")
    reference_dropout_probability: float = 0.0
    error_dropout_probability: float
    ape_input_role: Literal["raw_safety_evaluation_witness_only"] = (
        "raw_safety_evaluation_witness_only"
    )
    ape_may_pace_output: Literal[False] = False
    ape_may_supply_model_feature: Literal[False] = False

    @model_validator(mode="after")
    def _validate_mode(self) -> "RefOnlyModelInputContract":
        if float(self.reference_dropout_probability) != 0.0:
            raise ValueError("ref-only admission은 reference_dropout=0만 허용합니다")
        error_dropout = float(self.error_dropout_probability)
        if not math.isfinite(error_dropout) or not 0.0 <= error_dropout <= 1.0:
            raise ValueError("error_dropout_probability는 [0, 1]이어야 합니다")
        return self

    def digest(self) -> str:
        return _canonical_digest(self.model_dump(mode="json"))


class OutputClockMasterAdmission(BaseModel):
    """output-clock-master/ref-only 실행 전 immutable structural admission.

    SHA만 임의 문자열로 적는 것을 막기 위해 v3/timing/model-input payload를 inline으로
    보존하고 각각의 digest를 다시 계산한다. ablation/G0/validation/deployment artifact는
    외부 immutable receipt SHA로 결속하지만, 그 존재/내용 검증은 통합 gate 책임이다.
    """

    model_config = _FROZEN

    schema_version: Literal["output_clock_master_ref_only_admission_v1"] = (
        "output_clock_master_ref_only_admission_v1"
    )
    authority: Literal["structural_admission_only_not_physical_performance"] = (
        "structural_admission_only_not_physical_performance"
    )
    output_clock_owner: Literal["ab13x_outputstream_callback"] = (
        "ab13x_outputstream_callback"
    )
    output_channels: tuple[Literal["NS"], Literal["CS"]] = ("NS", "CS")
    inference_execution_context: Literal["dedicated_worker_not_output_callback"] = (
        "dedicated_worker_not_output_callback"
    )
    sample_rate: Literal[48_000] = 48_000
    block_size: Literal[256] = 256
    handoff_samples: Literal[256] = 256
    digital_reference_lead_samples: int = Field(ge=0)

    control_band_contract: BroadbandFullOctaveContractV3
    control_band_contract_sha256: str
    training_timing_contract: TrainingTimingContract
    training_timing_contract_sha256: str
    experiment_contract_sha256: str
    checkpoint_sha256: str
    deployment_artifact_sha256: str

    model_input_contract: RefOnlyModelInputContract
    model_input_mode_sha256: str
    canonical_train_err_zero_receipt: CanonicalErrZeroReceipt | None = None
    canonical_train_err_zero_receipt_sha256: str | None = None

    ref_only_ablation_receipt_sha256: str
    ref_only_g0_receipt_sha256: str
    ref_only_validation_receipt_sha256: str
    offline_streaming_equivalence_receipt_sha256: str
    offline_streaming_max_abs_error: float = Field(ge=0.0)
    offline_streaming_tolerance: Literal[1.0e-5] = _EQUIVALENCE_TOLERANCE

    reference_dropout_probability: Literal[0.0] = 0.0
    physical_performance_pass: Literal[False] = False
    sounddevice_integrated: Literal[False] = False
    deployment_eligible: Literal[False] = False

    @model_validator(mode="after")
    def _validate_admission(self) -> "OutputClockMasterAdmission":
        canonical_v3 = BroadbandFullOctaveContractV3.canonical()
        if self.control_band_contract.model_dump(mode="json") != canonical_v3.model_dump(
            mode="json"
        ):
            raise ValueError("exact canonical broadband v3 payload가 아닙니다")
        if self.control_band_contract_sha256 != self.control_band_contract.digest():
            raise ValueError("control_band_contract_sha256가 inline v3 payload와 다릅니다")

        if self.training_timing_contract_sha256 != self.training_timing_contract.digest():
            raise ValueError("training_timing_contract_sha256가 inline payload와 다릅니다")
        if int(self.training_timing_contract.sample_rate) != self.sample_rate:
            raise ValueError("training timing sample_rate가 output scheduler와 다릅니다")
        if int(self.training_timing_contract.handoff_samples) != self.handoff_samples:
            raise ValueError("training timing handoff는 정확히 256 samples여야 합니다")
        if (
            int(self.training_timing_contract.digital_reference_lead_samples)
            != self.digital_reference_lead_samples
        ):
            raise ValueError("runtime lead가 TrainingTimingContract lead와 다릅니다")

        if self.model_input_mode_sha256 != self.model_input_contract.digest():
            raise ValueError("model_input_mode_sha256가 inline ref-only payload와 다릅니다")
        if float(self.reference_dropout_probability) != float(
            self.model_input_contract.reference_dropout_probability
        ):
            raise ValueError("admission/model-input reference_dropout이 다릅니다")

        for name in (
            "control_band_contract_sha256",
            "training_timing_contract_sha256",
            "experiment_contract_sha256",
            "checkpoint_sha256",
            "deployment_artifact_sha256",
            "model_input_mode_sha256",
            "ref_only_ablation_receipt_sha256",
            "ref_only_g0_receipt_sha256",
            "ref_only_validation_receipt_sha256",
            "offline_streaming_equivalence_receipt_sha256",
        ):
            _require_sha256(name, str(getattr(self, name)))

        err_dropout = float(self.model_input_contract.error_dropout_probability)
        receipt = self.canonical_train_err_zero_receipt
        receipt_sha = self.canonical_train_err_zero_receipt_sha256
        if err_dropout != 1.0 and receipt is None:
            raise ValueError(
                "error_dropout=1이 아니면 canonical train item 전부 ERR exact-zero receipt가 "
                "필수입니다"
            )
        if (receipt is None) != (receipt_sha is None):
            raise ValueError("canonical ERR-zero receipt payload와 SHA를 함께 제공해야 합니다")
        if receipt is not None:
            _require_sha256("canonical_train_err_zero_receipt_sha256", str(receipt_sha))
            if receipt.digest() != receipt_sha:
                raise ValueError("canonical ERR-zero receipt SHA가 payload와 다릅니다")

        equivalence = float(self.offline_streaming_max_abs_error)
        if not math.isfinite(equivalence) or equivalence > float(
            self.offline_streaming_tolerance
        ):
            raise ValueError("offline-streaming equivalence가 1e-5를 넘습니다")
        return self

    def digest(self) -> str:
        return _canonical_digest(self.model_dump(mode="json"))


class OutputDiscontinuityCounters(BaseModel):
    """실제 어댑터 counter: 불연속/excess는 0, absolute backlog는 256 이하."""

    model_config = _FROZEN

    output_xrun_count: int = Field(default=0, ge=0)
    output_callback_status_count: int = Field(default=0, ge=0)
    deadline_miss_count: int = Field(default=0, ge=0)
    inference_queue_underflow_count: int = Field(default=0, ge=0)
    inference_queue_overflow_count: int = Field(default=0, ge=0)
    fallback_silence_block_count: int = Field(default=0, ge=0)
    dropped_sample_count: int = Field(default=0, ge=0)
    added_sample_count: int = Field(default=0, ge=0)
    sample_slip_count: int = Field(default=0, ge=0)
    stale_or_reused_control_block_count: int = Field(default=0, ge=0)
    nonzero_error_feature_block_count: int = Field(default=0, ge=0)
    allowed_backlog_samples: Literal[256] = 256
    maximum_absolute_backlog_samples: int = Field(default=0, ge=0)
    maximum_excess_backlog_samples: int = Field(default=0, ge=0)

    def violations(self) -> dict[str, int]:
        ignored = {"allowed_backlog_samples", "maximum_absolute_backlog_samples"}
        violations = {
            name: int(value)
            for name, value in self.model_dump().items()
            if name not in ignored and int(value) != 0
        }
        if self.maximum_absolute_backlog_samples > self.allowed_backlog_samples:
            violations["maximum_absolute_backlog_samples"] = int(
                self.maximum_absolute_backlog_samples
            )
        return violations


class InferenceFrameReceipt(BaseModel):
    model_config = _FROZEN

    job_id: str
    epoch: int
    source_callback_index: int
    target_output_callback_index: int
    source_frame_start: int
    source_frame_stop: int
    target_output_frame_start: int
    target_output_frame_stop: int
    reference_float32_sha256: str
    error_feature_float32_sha256: str
    error_feature_max_abs: Literal[0.0] = 0.0
    control_float32_sha256: str
    inference_execution_context: Literal["dedicated_worker_not_output_callback"] = (
        "dedicated_worker_not_output_callback"
    )


class OutputFrameReceipt(BaseModel):
    model_config = _FROZEN

    callback_index: int
    epoch: int
    global_output_frame_start: int
    global_output_frame_stop: int
    state: Literal[
        "anc_off", "startup_prime", "reset_prime", "rearm_prime", "anc_on"
    ]
    startup_prime: bool
    protocol_prime: bool
    prime_reason: Literal["none", "startup", "reset", "anc_on_transition"]
    performance_window_included: bool
    inference_ran_in_output_callback: Literal[False] = False
    output_clock_owner: Literal["ab13x_outputstream_callback"] = (
        "ab13x_outputstream_callback"
    )
    channel_order: tuple[Literal["NS"], Literal["CS"]] = ("NS", "CS")

    generated_source_frame_start: int
    generated_source_frame_stop: int
    generated_source_float32_sha256: str
    generated_source_pcm_s16_sha256: str
    reference_frame_start: int
    reference_frame_stop: int
    reference_float32_sha256: str
    playback_source_frame_start: int
    playback_source_frame_stop: int
    playback_source_float32_sha256: str
    playback_source_pcm_s16_sha256: str

    control_job_id: str | None
    control_source_callback_index: int | None
    control_reference_frame_start: int | None
    control_reference_frame_stop: int | None
    model_control_float32_sha256: str
    anc_gain_frame_start: int
    anc_gain_frame_stop: int
    anc_gain_float32_sha256: str
    anc_gain_min: float
    anc_gain_max: float
    actual_control_float32_sha256: str
    actual_control_pcm_s16_sha256: str
    actual_control_output_frame_start: int
    actual_control_output_frame_stop: int
    submitted_stereo_pcm_s16_sha256: str


class OutputClockMasterSchedulerReceipt(BaseModel):
    model_config = _FROZEN

    schema_version: Literal["output_clock_master_scheduler_receipt_v1"] = (
        "output_clock_master_scheduler_receipt_v1"
    )
    authority: Literal["structural_scheduler_only_not_physical_performance"] = (
        "structural_scheduler_only_not_physical_performance"
    )
    admission_sha256: str
    output_frames: tuple[OutputFrameReceipt, ...]
    inference_frames: tuple[InferenceFrameReceipt, ...]
    discontinuity_counters: OutputDiscontinuityCounters
    reset_count: int
    performance_output_block_count: int
    terminal_tail_target_callback_index: int | None
    physical_performance_pass: Literal[False] = False
    run_realtime_integrated: Literal[False] = False

    def digest(self) -> str:
        return _canonical_digest(self.model_dump(mode="json"))


@dataclass(frozen=True)
class InferenceJob:
    """worker에 넘기는 immutable-bytes backed ref-only job."""

    job_id: str
    epoch: int
    source_callback_index: int
    target_output_callback_index: int
    source_frame_start: int
    source_frame_stop: int
    target_output_frame_start: int
    target_output_frame_stop: int
    reference_float32_sha256: str
    error_feature_float32_sha256: str
    reference: np.ndarray
    error_feature: np.ndarray


@dataclass(frozen=True)
class OutputCallbackBlock:
    """출력 어댑터가 그대로 제출할 immutable S16 stereo 블록."""

    stereo_pcm_s16: np.ndarray
    receipt: OutputFrameReceipt


@dataclass
class _InternalJob:
    public: InferenceJob
    reference: np.ndarray
    error_feature: np.ndarray


@dataclass
class _InferenceResult:
    job: _InternalJob
    control: np.ndarray
    control_sha256: str


class OutputClockMasterScheduler:
    """장치와 모델을 모르는 deterministic one-block output scheduler."""

    def __init__(
        self,
        admission: OutputClockMasterAdmission,
        *,
        initial_global_output_frame: int = 0,
    ) -> None:
        if int(initial_global_output_frame) < 0:
            raise ValueError("initial_global_output_frame은 0 이상이어야 합니다")
        self.admission = admission
        self.block_size = int(admission.block_size)
        self._reference_buffer = DigitalReferenceBuffer(
            lead_samples=int(admission.digital_reference_lead_samples)
        )
        self._next_callback_index = 0
        self._next_global_output_frame = int(initial_global_output_frame)
        self._anc_requested = False
        self._prime_required = True
        self._next_prime_reason: Literal[
            "startup", "reset", "anc_on_transition"
        ] = "startup"
        self._ever_primed = False
        self._epoch = 0
        self._reset_count = 0
        self._pending_job: _InternalJob | None = None
        self._inflight_job: _InternalJob | None = None
        self._results: dict[int, _InferenceResult] = {}
        self._output_receipts: list[OutputFrameReceipt] = []
        self._inference_receipts: list[InferenceFrameReceipt] = []
        self._blocked_reason: str | None = None
        self._closed = False
        self._terminal_tail_target: int | None = None

    @property
    def blocked_reason(self) -> str | None:
        return self._blocked_reason

    def _ensure_live(self) -> None:
        if self._blocked_reason is not None:
            raise OutputClockMasterBlocked(self._blocked_reason)
        if self._closed:
            raise OutputClockMasterBlocked("scheduler evidence window가 이미 닫혔습니다")

    def _block(self, reason: str) -> None:
        self._blocked_reason = str(reason)
        raise OutputClockMasterBlocked(self._blocked_reason)

    def _clear_pipeline_for_protocol_transition(self) -> None:
        # 성능 구간 밖의 명시적 OFF/reset 경계다. worker가 이미 claim한 job이 있으면
        # race를 숨길 수 없으므로 전환 자체를 차단한다.
        if self._inflight_job is not None:
            self._block("ANC OFF/reset 전환 시 inference job이 아직 inflight입니다")
        self._pending_job = None
        self._results.clear()

    def request_anc_on(self) -> None:
        """다음 output callback을 exact-zero prime으로 예약한다."""

        self._ensure_live()
        if self._anc_requested:
            return
        self._clear_pipeline_for_protocol_transition()
        self._anc_requested = True
        self._prime_required = True
        self._next_prime_reason = (
            "anc_on_transition" if self._ever_primed else "startup"
        )

    def request_anc_off(self) -> None:
        """출력을 즉시 protocol OFF로 전환한다(다음 callback gain은 exact zero)."""

        self._ensure_live()
        self._clear_pipeline_for_protocol_transition()
        self._anc_requested = False
        self._prime_required = True
        self._next_prime_reason = "anc_on_transition"
        self._epoch += 1

    def request_reset(self) -> None:
        """모델 reset 경계를 기록하고 ANC ON 유지 시에도 다시 prime한다.

        실제 engine state reset은 통합 어댑터가 같은 경계에서 별도로 수행해야 한다.
        """

        self._ensure_live()
        self._clear_pipeline_for_protocol_transition()
        self._prime_required = True
        self._next_prime_reason = "reset"
        self._epoch += 1
        self._reset_count += 1

    def _schedule_job(
        self,
        *,
        callback_index: int,
        global_output_frame_start: int,
        reference: np.ndarray,
    ) -> None:
        if self._pending_job is not None or self._inflight_job is not None:
            self._block("inference queue overflow: 이전 job이 아직 남아 있습니다")
        if self._results:
            self._block("stale/early control result가 다음 source 등록 전에 남아 있습니다")

        error_feature = np.zeros(self.block_size, dtype=np.float32)
        reference_sha = _raw_array_sha256(reference)
        error_sha = _raw_array_sha256(error_feature)
        target_callback = callback_index + 1
        target_frame = global_output_frame_start + self.block_size
        identity = {
            "admission_sha256": self.admission.digest(),
            "epoch": self._epoch,
            "source_callback_index": callback_index,
            "target_output_callback_index": target_callback,
            "source_frame_start": global_output_frame_start,
            "target_output_frame_start": target_frame,
            "reference_float32_sha256": reference_sha,
            "error_feature_float32_sha256": error_sha,
        }
        job_id = _canonical_digest(identity)
        public = InferenceJob(
            job_id=job_id,
            epoch=self._epoch,
            source_callback_index=callback_index,
            target_output_callback_index=target_callback,
            source_frame_start=global_output_frame_start,
            source_frame_stop=global_output_frame_start + self.block_size,
            target_output_frame_start=target_frame,
            target_output_frame_stop=target_frame + self.block_size,
            reference_float32_sha256=reference_sha,
            error_feature_float32_sha256=error_sha,
            reference=_readonly_copy(reference),
            error_feature=_readonly_copy(error_feature),
        )
        self._pending_job = _InternalJob(
            public=public,
            reference=np.ascontiguousarray(reference).copy(),
            error_feature=error_feature,
        )

    def claim_inference_job(self) -> InferenceJob | None:
        """별도 worker가 호출한다. 빈 polling은 failure가 아니므로 ``None`` 반환."""

        self._ensure_live()
        if self._inflight_job is not None:
            return None
        if self._pending_job is None:
            return None
        self._inflight_job = self._pending_job
        self._pending_job = None
        return self._inflight_job.public

    def submit_inference_result(
        self,
        *,
        job_id: str,
        source_callback_index: int,
        target_output_callback_index: int,
        reference_used: np.ndarray,
        error_feature_used: np.ndarray,
        control: np.ndarray,
    ) -> None:
        """worker 결과를 제출하며 actual model input identity를 다시 검증한다."""

        self._ensure_live()
        job = self._inflight_job
        if job is None or job.public.job_id != job_id:
            self._block("stale/reused/unknown inference job 결과가 제출되었습니다")
        assert job is not None  # type narrowing
        public = job.public
        if int(source_callback_index) != public.source_callback_index:
            self._block("control source callback identity가 다릅니다")
        if int(target_output_callback_index) != public.target_output_callback_index:
            delta = int(target_output_callback_index) - public.target_output_callback_index
            self._block(f"control target이 정확히 k+1이 아닙니다(delta={delta})")

        try:
            used_ref = _float32_block("reference_used", reference_used, self.block_size)
            used_err = _float32_block(
                "error_feature_used", error_feature_used, self.block_size
            )
            control_array = _float32_block("control", control, self.block_size)
        except ValueError as exc:
            self._block(str(exc))
        if _raw_array_sha256(used_ref) != public.reference_float32_sha256:
            self._block("worker가 U_k와 다른 reference를 모델에 사용했습니다")
        if np.any(used_err != np.float32(0.0)):
            self._block("ref-only worker의 ERR feature가 exact zero가 아닙니다")
        if _raw_array_sha256(used_err) != public.error_feature_float32_sha256:
            self._block("worker ERR feature payload SHA가 job과 다릅니다")
        if public.target_output_callback_index in self._results:
            self._block("같은 target callback control이 중복 제출되었습니다")

        control_sha = _raw_array_sha256(control_array)
        self._results[public.target_output_callback_index] = _InferenceResult(
            job=job,
            control=control_array.copy(),
            control_sha256=control_sha,
        )
        self._inference_receipts.append(
            InferenceFrameReceipt(
                job_id=public.job_id,
                epoch=public.epoch,
                source_callback_index=public.source_callback_index,
                target_output_callback_index=public.target_output_callback_index,
                source_frame_start=public.source_frame_start,
                source_frame_stop=public.source_frame_stop,
                target_output_frame_start=public.target_output_frame_start,
                target_output_frame_stop=public.target_output_frame_stop,
                reference_float32_sha256=public.reference_float32_sha256,
                error_feature_float32_sha256=public.error_feature_float32_sha256,
                control_float32_sha256=control_sha,
            )
        )
        self._inflight_job = None

    def output_callback(
        self,
        *,
        callback_index: int,
        global_output_frame_start: int,
        future_source: np.ndarray,
        anc_gain: float | np.ndarray,
    ) -> OutputCallbackBlock:
        """한 output callback의 NS/CS를 조립한다. 모델 추론은 호출하지 않는다."""

        self._ensure_live()
        if int(callback_index) != self._next_callback_index:
            self._block(
                "callback drop/add/reorder: "
                f"expected={self._next_callback_index}, actual={callback_index}"
            )
        if int(global_output_frame_start) != self._next_global_output_frame:
            self._block(
                "global output frame sample slip: "
                f"expected={self._next_global_output_frame}, "
                f"actual={global_output_frame_start}"
            )
        try:
            source = _float32_block("future_source", future_source, self.block_size)
            gain = _gain_block(anc_gain, self.block_size)
        except ValueError as exc:
            self._block(str(exc))
        if np.any(np.abs(source) > 1.0):
            self._block("future_source가 float→S16 무클립 범위 [-1,1]을 넘습니다")

        playback, reference = self._reference_buffer.process(source)
        playback = np.ascontiguousarray(playback, dtype=np.float32)
        reference = np.ascontiguousarray(reference, dtype=np.float32)
        if _raw_array_sha256(reference) != _raw_array_sha256(source):
            self._block("DigitalReferenceBuffer가 future U_k identity를 보존하지 않았습니다")

        result: _InferenceResult | None = None
        state: Literal[
            "anc_off", "startup_prime", "reset_prime", "rearm_prime", "anc_on"
        ]
        protocol_prime = False
        startup_prime = False
        prime_reason: Literal["none", "startup", "reset", "anc_on_transition"] = (
            "none"
        )
        performance = False
        if not self._anc_requested:
            state = "anc_off"
            if np.any(gain != np.float32(0.0)):
                self._block("ANC OFF callback의 anc_gain은 exact zero여야 합니다")
            model_control = np.zeros(self.block_size, dtype=np.float32)
        elif self._prime_required:
            if self._pending_job is not None or self._inflight_job is not None or self._results:
                self._block("prime 시작 전에 stale pipeline state가 남아 있습니다")
            if np.any(gain != np.float32(0.0)):
                self._block("protocol prime callback의 anc_gain은 exact zero여야 합니다")
            startup_prime = not self._ever_primed
            protocol_prime = True
            prime_reason = self._next_prime_reason
            if startup_prime:
                state = "startup_prime"
            elif prime_reason == "reset":
                state = "reset_prime"
            else:
                state = "rearm_prime"
            model_control = np.zeros(self.block_size, dtype=np.float32)
            self._prime_required = False
            self._ever_primed = True
        else:
            result = self._results.pop(int(callback_index), None)
            if result is None:
                pending = self._pending_job is not None
                inflight = self._inflight_job is not None
                self._block(
                    "inference underflow/late result: callback k에 y_(k-1)가 없음 "
                    f"(pending={pending}, inflight={inflight})"
                )
            assert result is not None
            public = result.job.public
            if public.target_output_callback_index != int(callback_index):
                self._block("early/late control target identity 불일치")
            if public.source_callback_index != int(callback_index) - 1:
                self._block("control이 바로 전 callback U_(k-1)에서 생성되지 않았습니다")
            if public.epoch != self._epoch:
                self._block("reset/ON epoch 이전 stale control이 제출되었습니다")
            state = "anc_on"
            performance = True
            model_control = result.control

        actual_control = np.ascontiguousarray(model_control * gain, dtype=np.float32)
        if not np.all(np.isfinite(actual_control)) or np.any(
            np.abs(actual_control) > 1.0
        ):
            self._block("gain 적용 control이 float→S16 무클립 범위를 넘습니다")

        generated_source_pcm = float32_to_pcm_int16(source)
        playback_pcm = float32_to_pcm_int16(playback)
        control_pcm = float32_to_pcm_int16(actual_control)
        stereo_pcm = np.stack((playback_pcm, control_pcm), axis=1).astype(
            np.int16, copy=False
        )

        public_result = None if result is None else result.job.public
        frame_start = int(global_output_frame_start)
        receipt = OutputFrameReceipt(
            callback_index=int(callback_index),
            epoch=self._epoch,
            global_output_frame_start=frame_start,
            global_output_frame_stop=frame_start + self.block_size,
            state=state,
            startup_prime=startup_prime,
            protocol_prime=protocol_prime,
            prime_reason=prime_reason,
            performance_window_included=performance,
            generated_source_frame_start=frame_start,
            generated_source_frame_stop=frame_start + self.block_size,
            generated_source_float32_sha256=_raw_array_sha256(source),
            generated_source_pcm_s16_sha256=_raw_array_sha256(generated_source_pcm),
            reference_frame_start=frame_start,
            reference_frame_stop=frame_start + self.block_size,
            reference_float32_sha256=_raw_array_sha256(reference),
            playback_source_frame_start=frame_start
            - int(self.admission.digital_reference_lead_samples),
            playback_source_frame_stop=frame_start
            + self.block_size
            - int(self.admission.digital_reference_lead_samples),
            playback_source_float32_sha256=_raw_array_sha256(playback),
            playback_source_pcm_s16_sha256=_raw_array_sha256(playback_pcm),
            control_job_id=None if public_result is None else public_result.job_id,
            control_source_callback_index=(
                None if public_result is None else public_result.source_callback_index
            ),
            control_reference_frame_start=(
                None if public_result is None else public_result.source_frame_start
            ),
            control_reference_frame_stop=(
                None if public_result is None else public_result.source_frame_stop
            ),
            model_control_float32_sha256=_raw_array_sha256(model_control),
            anc_gain_frame_start=frame_start,
            anc_gain_frame_stop=frame_start + self.block_size,
            anc_gain_float32_sha256=_raw_array_sha256(gain),
            anc_gain_min=float(np.min(gain)),
            anc_gain_max=float(np.max(gain)),
            actual_control_float32_sha256=_raw_array_sha256(actual_control),
            actual_control_pcm_s16_sha256=_raw_array_sha256(control_pcm),
            actual_control_output_frame_start=frame_start,
            actual_control_output_frame_stop=frame_start + self.block_size,
            submitted_stereo_pcm_s16_sha256=_raw_array_sha256(stereo_pcm),
        )
        self._output_receipts.append(receipt)

        # OFF에서는 APE가 pacing/feature가 아니므로 inference job을 만들지 않는다.
        # prime/ON에서는 U_k를 callback 밖 worker에 넘겨 정확히 k+1을 예약한다.
        if self._anc_requested:
            self._schedule_job(
                callback_index=int(callback_index),
                global_output_frame_start=frame_start,
                reference=reference,
            )

        self._next_callback_index += 1
        self._next_global_output_frame += self.block_size
        return OutputCallbackBlock(
            stereo_pcm_s16=_readonly_copy(stereo_pcm),
            receipt=receipt,
        )

    def close_evidence_window(
        self,
        *,
        discontinuity_counters: OutputDiscontinuityCounters,
    ) -> OutputClockMasterSchedulerReceipt:
        """유한 실험 구간을 닫고 immutable structural receipt를 만든다.

        마지막 callback이 만든 U_k의 target ``k+1``은 관측 창 밖의 protocol tail로
        명시한다. 이것은 중간 callback drop/fallback이 아니다. tail은 최대 하나만
        허용하며, inflight race가 있으면 닫지 않는다.
        """

        self._ensure_live()
        violations = discontinuity_counters.violations()
        if violations:
            self._block(
                "runtime discontinuity/excess-backlog counters가 허용 계약을 위반했습니다: "
                f"{violations}"
            )
        if self._inflight_job is not None:
            self._block("evidence window 종료 시 inference worker job이 inflight입니다")
        if not any(row.protocol_prime for row in self._output_receipts):
            self._block("protocol prime output receipt가 없습니다")
        performance_count = sum(
            int(row.performance_window_included) for row in self._output_receipts
        )
        if performance_count <= 0:
            self._block("completed y 이후 ANC ON performance callback이 없습니다")

        tail_targets: list[int] = []
        if self._pending_job is not None:
            tail_targets.append(self._pending_job.public.target_output_callback_index)
        tail_targets.extend(self._results.keys())
        if len(tail_targets) != 1 or tail_targets[0] != self._next_callback_index:
            self._block(
                "종료 시 정확히 하나의 unobserved k+1 tail만 허용합니다: "
                f"targets={tail_targets}, next={self._next_callback_index}"
            )
        self._terminal_tail_target = tail_targets[0]
        self._closed = True
        return OutputClockMasterSchedulerReceipt(
            admission_sha256=self.admission.digest(),
            output_frames=tuple(self._output_receipts),
            inference_frames=tuple(self._inference_receipts),
            discontinuity_counters=discontinuity_counters,
            reset_count=self._reset_count,
            performance_output_block_count=performance_count,
            terminal_tail_target_callback_index=self._terminal_tail_target,
        )


__all__ = [
    "CanonicalErrZeroReceipt",
    "InferenceJob",
    "OutputCallbackBlock",
    "OutputClockMasterAdmission",
    "OutputClockMasterBlocked",
    "OutputClockMasterScheduler",
    "OutputClockMasterSchedulerReceipt",
    "OutputDiscontinuityCounters",
    "RefOnlyModelInputContract",
]
