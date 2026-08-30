"""Zero-reset surrogate segment의 causal full-octave ``P*n + S*y`` composition.

target만 S(z)에 통과시키면 crop 이전 actuator output의 secondary delay/FIR tail을
잃는다. 또한 digital reference ``x_ref(t)=n(t+K)`` 또는 input-only mic augmentation을
P 입력으로 쓰면 physical disturbance가 K samples 앞서거나 mic noise/dropout까지 P에
들어간다. 이 module은 controller를 256-sample block으로 prefix부터 target까지 한 번만
전진시키고, clean playback timeline ``n``에 P를, controller output에 S를 적용해
target crop의 ``e = P*n + S*y``를 만든다.

v1은 digital-reference surrogate segment에만 한정한다. measured ``d`` tensor나 외부
recurrent state를 받지 않는다. 따라서 arbitrary/off-by-one ``d`` 또는 유한 prefix로
live GLSTM/MHSA state를 재현했다고 주장하는 길이 없다. Trainer, DataLoader, run
directory, audio I/O, GPU 초기화도 만들지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch
from torch import nn

from ..losses.broadband_loss import CausalFIRPath
from .full_octave_causal_plant_binding_v4 import FullOctaveCausalPlantBindingV4


CAUSAL_SECONDARY_PREFIX_ADAPTER_SCHEMA_V1 = "causal_secondary_prefix_adapter_v1"
CAUSAL_PREFIX_STATE_ORIGIN_SCHEMA_V1 = "causal_prefix_state_origin_v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if not _SHA256.fullmatch(text):
        raise ValueError(f"{label}는 lowercase SHA-256이어야 합니다")
    return text


def _as_int_tuple(value: tuple[int, ...], *, expected: int, label: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or len(value) != expected:
        raise ValueError(f"{label} 길이는 batch({expected})와 같아야 합니다")
    parsed: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{label}는 0 이상 bool 아닌 int tuple이어야 합니다")
        parsed.append(int(item))
    return tuple(parsed)


@dataclass(frozen=True)
class CausalPrefixStateOriginV1:
    """v1이 허용하는 유일한 controller state origin receipt."""

    kind: Literal["segment_start_zero_state"]
    binding_sha256: str
    source_sha256: tuple[str, ...]
    schema_version: str = CAUSAL_PREFIX_STATE_ORIGIN_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema_version != CAUSAL_PREFIX_STATE_ORIGIN_SCHEMA_V1:
            raise ValueError("causal prefix state-origin schema가 다릅니다")
        _require_sha256(self.binding_sha256, label="state origin binding SHA")
        if not isinstance(self.source_sha256, tuple) or not self.source_sha256:
            raise ValueError("state origin source SHA tuple이 비었습니다")
        for value in self.source_sha256:
            _require_sha256(value, label="state origin source SHA")
        if self.kind != "segment_start_zero_state":
            raise ValueError("v1은 segment-start zero state만 허용합니다")


@dataclass(frozen=True)
class CausalPrefixBatchV1:
    """이미 timing-v2로 정렬된 digital-reference surrogate segment.

    ``d_target``을 받지 않는다. ``clean_playback_timeline``은 segment time 0부터
    ``prefix + target + digital_reference_lead``까지의 input-only augmentation 이전
    physical playback ``n``이다. 즉 common gain/polarity/EQ는 이미 반영될 수 있지만,
    mic noise/hum/dropout은 포함하지 않는다. adapter는 이 timeline에서 P input과 clean
    preview를 모두 유도하므로 P/S/target crop의 시간축을 하나로 고정한다.
    """

    x_prefix: torch.Tensor
    x_target: torch.Tensor
    source_sha256: tuple[str, ...]
    clean_playback_source_sha256: tuple[str, ...]
    clean_playback_timeline: torch.Tensor
    controller_reference_preaugmentation: torch.Tensor
    training_timing_contract_sha256: str
    segment_prefix_start_samples: tuple[int, ...]
    segment_target_start_samples: tuple[int, ...]
    global_sample_indices: tuple[int, ...]
    state_origin: CausalPrefixStateOriginV1


class CausalStreamingControlModelV1(Protocol):
    """Adapter가 허용하는 최소 model API; ``forward`` 호출은 의도적으로 없다."""

    hop: int

    def init_states(self, batch: int = 1, device: torch.device | str = "cpu") -> Any:
        """zero-reset segment 첫 sample 직전의 state를 반환한다."""

    def streaming_step(
        self, x_block: torch.Tensor, state: Any
    ) -> tuple[torch.Tensor, Any]:
        """현재 block만 읽고 [batch,1,block] output과 다음 state를 반환한다."""


@dataclass(frozen=True)
class CausalSecondaryPrefixResultV1:
    """target 창만 loss/evaluator에 전달하는 FP32 surrogate 결과."""

    y_prefix: torch.Tensor
    y_target: torch.Tensor
    clean_reference_preview: torch.Tensor
    primary_target: torch.Tensor
    secondary_target: torch.Tensor
    error_target: torch.Tensor
    final_state: Any
    binding_sha256: str
    prefix_samples: int
    target_samples: int
    schema_version: str = CAUSAL_SECONDARY_PREFIX_ADAPTER_SCHEMA_V1


class CausalSecondaryPrefixAdapterV1(nn.Module):
    """future raw-bound binding용 causal P/S composition; 현재 fixture는 test-only."""

    def __init__(
        self,
        binding: FullOctaveCausalPlantBindingV4,
        *,
        _allow_test_fixture: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(binding, FullOctaveCausalPlantBindingV4):
            raise TypeError("verified FullOctaveCausalPlantBindingV4만 허용합니다")
        if binding.fixture_only and not _allow_test_fixture:
            raise ValueError(
                "raw-bound production authority가 없는 test fixture는 public adapter에서 "
                "허용되지 않습니다"
            )
        self.binding = binding
        self.primary_path = CausalFIRPath(binding.primary_operator)
        self.secondary_path = CausalFIRPath(binding.secondary_operator)
        self.block_size = int(binding.block_size)
        self.binding_sha256 = binding.digest()

    @classmethod
    def from_verified_full_octave_binding(
        cls, binding: FullOctaveCausalPlantBindingV4
    ) -> "CausalSecondaryPrefixAdapterV1":
        """future raw-bound issuer의 non-fixture binding만 받는 production entrypoint."""

        return cls(binding)

    @classmethod
    def _for_test_fixture(
        cls, binding: FullOctaveCausalPlantBindingV4
    ) -> "CausalSecondaryPrefixAdapterV1":
        return cls(binding, _allow_test_fixture=True)

    @staticmethod
    def _validate_input_tensor(value: torch.Tensor, *, label: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label}는 torch.Tensor여야 합니다")
        if value.ndim != 3 or int(value.shape[0]) < 1 or int(value.shape[1]) < 1:
            raise ValueError(f"{label}는 [batch, channels, samples]이어야 합니다")
        if int(value.shape[-1]) < 1:
            raise ValueError(f"{label} sample 길이가 비었습니다")
        if not value.is_floating_point():
            raise ValueError(f"{label}는 floating point tensor여야 합니다")
        if not torch.isfinite(value).all():
            raise ValueError(f"{label}에 NaN/Inf가 있습니다")
        return value

    @classmethod
    def _validate_output_tensor(cls, value: torch.Tensor, *, label: str) -> torch.Tensor:
        value = cls._validate_input_tensor(value, label=label)
        if int(value.shape[1]) != 1:
            raise ValueError(f"{label}는 controller output [batch, 1, samples]이어야 합니다")
        return value

    def _validate_batch(self, batch: CausalPrefixBatchV1) -> tuple[int, int, int]:
        if not isinstance(batch, CausalPrefixBatchV1):
            raise TypeError("CausalPrefixBatchV1이 필요합니다")
        x_prefix = self._validate_input_tensor(batch.x_prefix, label="x_prefix")
        x_target = self._validate_input_tensor(batch.x_target, label="x_target")
        batch_size = int(x_prefix.shape[0])
        prefix_samples = int(x_prefix.shape[-1])
        target_samples = int(x_target.shape[-1])
        if (
            int(x_target.shape[0]) != batch_size
            or int(x_target.shape[1]) != int(x_prefix.shape[1])
        ):
            raise ValueError("x_prefix와 x_target batch/channel shape가 다릅니다")
        if x_prefix.device != x_target.device:
            raise ValueError("prefix와 target tensor device가 다릅니다")
        if prefix_samples % self.block_size or target_samples % self.block_size:
            raise ValueError("prefix와 target은 256-sample block의 정수배여야 합니다")
        if prefix_samples < int(self.binding.required_prefix_samples):
            raise ValueError("prefix가 P/S delay + handoff + FIR support history보다 짧습니다")
        if int(self.binding.reference_channel_index) >= int(x_prefix.shape[1]):
            raise ValueError("binding reference channel이 input batch channel에 없습니다")
        if batch.training_timing_contract_sha256 != self.binding.training_timing_contract_sha256:
            raise ValueError("batch timing-v2 payload SHA와 causal plant binding이 다릅니다")
        if not isinstance(batch.source_sha256, tuple) or len(batch.source_sha256) != batch_size:
            raise ValueError("source SHA tuple 길이는 batch와 같아야 합니다")
        for value in batch.source_sha256:
            _require_sha256(value, label="source SHA")
        if (
            not isinstance(batch.clean_playback_source_sha256, tuple)
            or len(batch.clean_playback_source_sha256) != batch_size
        ):
            raise ValueError("clean playback source SHA tuple 길이는 batch와 같아야 합니다")
        for value in batch.clean_playback_source_sha256:
            _require_sha256(value, label="clean playback source SHA")
        if batch.clean_playback_source_sha256 != batch.source_sha256:
            raise ValueError("controller source와 clean playback source lineage가 다릅니다")
        clean_timeline = self._validate_output_tensor(
            batch.clean_playback_timeline,
            label="clean_playback_timeline",
        )
        lead = int(self.binding.training_timing_contract.digital_reference_lead_samples)
        expected_timeline_samples = prefix_samples + target_samples + lead
        if (
            int(clean_timeline.shape[0]) != batch_size
            or clean_timeline.device != x_prefix.device
            or int(clean_timeline.shape[-1]) != expected_timeline_samples
        ):
            raise ValueError(
                "clean playback timeline은 batch/device와 prefix + target + derived lead 길이가 "
                "정확히 같아야 합니다"
            )
        preaugmentation_reference = self._validate_output_tensor(
            batch.controller_reference_preaugmentation,
            label="controller_reference_preaugmentation",
        )
        expected_preview = clean_timeline[..., lead : lead + prefix_samples + target_samples]
        if (
            preaugmentation_reference.shape != expected_preview.shape
            or preaugmentation_reference.device != expected_preview.device
            or not torch.equal(preaugmentation_reference.float(), expected_preview.float())
        ):
            raise ValueError(
                "controller pre-augmentation reference가 clean playback의 derived lead preview와 "
                "exact하게 일치하지 않습니다"
            )
        prefix_start = _as_int_tuple(
            batch.segment_prefix_start_samples,
            expected=batch_size,
            label="segment prefix start",
        )
        target_start = _as_int_tuple(
            batch.segment_target_start_samples,
            expected=batch_size,
            label="segment target start",
        )
        _as_int_tuple(
            batch.global_sample_indices,
            expected=batch_size,
            label="global sample index",
        )
        if any(
            target != start + prefix_samples
            for start, target in zip(prefix_start, target_start)
        ):
            raise ValueError("target은 같은 segment의 prefix 직후 sample에서 시작해야 합니다")
        if any(start != 0 for start in prefix_start):
            raise ValueError("zero-reset state는 training segment sample 0에서만 시작해야 합니다")
        origin = batch.state_origin
        if not isinstance(origin, CausalPrefixStateOriginV1):
            raise TypeError("CausalPrefixStateOriginV1이 필요합니다")
        if origin.binding_sha256 != self.binding_sha256:
            raise ValueError("state origin과 causal plant binding SHA가 다릅니다")
        if origin.source_sha256 != batch.source_sha256:
            raise ValueError("state origin과 batch source SHA가 다릅니다")
        return batch_size, prefix_samples, target_samples

    def _stream(
        self,
        model: CausalStreamingControlModelV1,
        value: torch.Tensor,
        *,
        state: Any,
    ) -> tuple[torch.Tensor, Any]:
        hop = getattr(model, "hop", None)
        if isinstance(hop, bool) or not isinstance(hop, int) or int(hop) <= 0:
            raise ValueError("streaming model은 explicit positive integer hop을 가져야 합니다")
        if self.block_size % int(hop):
            raise ValueError(
                "streaming model hop은 256-sample callback block을 정확히 나눠야 합니다"
            )
        context = getattr(model, "context", None)
        if context is not None and (
            isinstance(context, bool)
            or not isinstance(context, int)
            or context < 0
            or context > self.block_size
        ):
            raise ValueError("streaming model context는 0 이상 256 이하 int여야 합니다")
        input_channels = getattr(model, "in_channels", None)
        if input_channels is not None and int(input_channels) != int(value.shape[1]):
            raise ValueError("streaming model input channels와 prefix batch가 다릅니다")
        step = getattr(model, "streaming_step", None)
        if not callable(step):
            raise TypeError("streaming model에 streaming_step API가 없습니다")
        blocks: list[torch.Tensor] = []
        for start in range(0, int(value.shape[-1]), self.block_size):
            x_block = value[..., start : start + self.block_size]
            output, state = step(x_block, state)
            output = self._validate_output_tensor(output, label="streaming_step output")
            if (
                int(output.shape[0]) != int(x_block.shape[0])
                or int(output.shape[-1]) != int(x_block.shape[-1])
                or output.device != x_block.device
            ):
                raise ValueError("streaming_step output batch/time/device가 input block과 다릅니다")
            # P/S convolution과 full-octave loss는 FP32다. .float()는 graph를 끊지 않는다.
            blocks.append(output.float())
        return torch.cat(blocks, dim=-1), state

    def forward(
        self,
        model: CausalStreamingControlModelV1,
        batch: CausalPrefixBatchV1,
    ) -> CausalSecondaryPrefixResultV1:
        """continuous-prefix ``P*n + S*y`` target crop을 FP32로 계산한다."""

        batch_size, prefix_samples, target_samples = self._validate_batch(batch)
        initializer = getattr(model, "init_states", None)
        if not callable(initializer):
            raise TypeError("zero-reset segment에는 init_states API가 필요합니다")
        state = initializer(batch=batch_size, device=batch.x_prefix.device)
        y_prefix, state = self._stream(model, batch.x_prefix, state=state)
        y_target, state = self._stream(model, batch.x_target, state=state)
        y_full = torch.cat((y_prefix, y_target), dim=-1)
        total_samples = prefix_samples + target_samples
        lead = int(self.binding.training_timing_contract.digital_reference_lead_samples)
        clean_playback = batch.clean_playback_timeline[..., :total_samples].float()
        clean_reference_preview = batch.clean_playback_timeline[
            ..., lead : lead + total_samples
        ].float()
        primary_full = self.primary_path(clean_playback)
        secondary_full = self.secondary_path(y_full)
        primary_target = primary_full[..., prefix_samples:].float()
        secondary_target = secondary_full[..., prefix_samples:].float()
        if primary_target.shape != y_target.shape or secondary_target.shape != y_target.shape:
            raise RuntimeError("primary/secondary target crop shape가 causal output과 다릅니다")
        # 측정 FIR의 극성을 보존한다. 어느 branch도 여기서 부호를 더 뒤집지 않는다.
        error_target = primary_target + secondary_target
        return CausalSecondaryPrefixResultV1(
            y_prefix=y_prefix,
            y_target=y_target,
            clean_reference_preview=clean_reference_preview,
            primary_target=primary_target,
            secondary_target=secondary_target,
            error_target=error_target,
            final_state=state,
            binding_sha256=self.binding_sha256,
            prefix_samples=prefix_samples,
            target_samples=target_samples,
        )
