"""Stage-2 pretrain smoke의 family-balanced sampler와 causal tensor adapter.

generic ``Trainer``를 사용하지 않는 별도 경계다. sampler는 네 family quota를
실제 index batch에서 강제하고, adapter는 prefix 전체에 ``P*n``과 ``S*y``를
적용한 뒤 valid target에서 Stage-2 전용 loss를 계산한다.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch
from torch import nn

from ..dsp.stage2_2khz_contract import (
    STAGE2_2KHZ_SOURCE_FAMILIES,
    Stage2TwoKilohertzContract,
)
from ..losses.broadband_loss import CausalFIRPath
from ..losses.stage2_2khz_loss import Stage2TwoKilohertzLoss
from .causal_secondary_prefix_adapter_v1 import (
    CausalPrefixBatchV1,
    CausalPrefixStateOriginV1,
    CausalStreamingControlModelV1,
)
from .stage2_2khz_binding import Stage2TwoKilohertzPlantBinding


STAGE2_2KHZ_SAMPLER_SCHEMA = "stage2_2khz_family_component_sampler_v1"
STAGE2_2KHZ_PREFIX_ADAPTER_SCHEMA = "stage2_2khz_causal_prefix_adapter_v1"
STAGE2_2KHZ_TRAINER_ADAPTER_SCHEMA = "stage2_2khz_trainer_adapter_v1"
STAGE2_MAX_ACTUATOR_PEAK_PCM = 98
STAGE2_MAX_ACTUATOR_ABS = STAGE2_MAX_ACTUATOR_PEAK_PCM / 32768.0

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def require_stage2_actuator_limit(model_config: Mapping[str, Any]) -> float:
    """Stage-2 model output limiter가 measured actuator admission 안인지 검사한다.

    누락 시 HybridANCModel의 legacy default 0.2로 조용히 돌아가는 것을 허용하지 않는다.
    이 gate는 config를 읽은 직후, model parameter/CUDA allocation 전에 호출해야 한다.
    """

    if not isinstance(model_config, Mapping):
        raise ValueError("Stage-2 model config는 mapping이어야 합니다")
    limiter = model_config.get("limiter")
    if not isinstance(limiter, Mapping) or set(limiter) != {"limit"}:
        raise ValueError("Stage-2 model limiter.limit를 명시적으로 하나만 선언해야 합니다")
    raw = limiter.get("limit")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("Stage-2 model limiter.limit는 finite 숫자여야 합니다")
    limit = float(raw)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("Stage-2 model limiter.limit는 finite 양수여야 합니다")
    if not math.isclose(
        limit, STAGE2_MAX_ACTUATOR_ABS, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise ValueError(
            "Stage-2 model actuator limit는 physical binding의 98/32768과 exact해야 합니다"
        )
    return limit


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if not _SHA256.fullmatch(text):
        raise ValueError(f"{label}는 lowercase SHA-256이어야 합니다")
    return text


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class Stage2SamplerRecord:
    """manifest actual row에 결속된 sampler 최소 identity."""

    dataset_index: int
    source_family: str
    component_id: str
    split: str
    source_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.dataset_index, bool) or int(self.dataset_index) < 0:
            raise ValueError("Stage-2 dataset index는 0 이상 bool 아닌 int여야 합니다")
        if self.source_family not in STAGE2_2KHZ_SOURCE_FAMILIES:
            raise ValueError(f"알 수 없는 Stage-2 source family: {self.source_family!r}")
        if not str(self.component_id).strip():
            raise ValueError("Stage-2 component_id가 비었습니다")
        if self.split not in {"train", "val", "test"}:
            raise ValueError("Stage-2 split은 train/val/test 중 하나여야 합니다")
        _require_sha256(self.source_sha256, label="Stage-2 source SHA")


class Stage2FamilyComponentBatchSampler:
    """실제 index를 family→component→item 순으로 균등 선택한다.

    source-mix ratio 태그 확률은 사용하지 않는다. 각 batch에 네 family가
    정확히 같은 개수로 들어가고, component cursor는 global step에서만
    결정되어 중단/재개에서도 같은 index를 낸다.
    """

    def __init__(
        self,
        records: Sequence[Stage2SamplerRecord],
        *,
        batch_size: int,
        seed: int,
        split: Literal["train", "val"] = "train",
        manifest_bundle_sha256: str,
        sampler_receipt_sha256: str,
    ) -> None:
        if isinstance(batch_size, bool) or int(batch_size) < 4 or int(batch_size) % 4:
            raise ValueError("Stage-2 batch_size는 4 이상 4의 배수여야 합니다")
        if isinstance(seed, bool) or int(seed) < 0:
            raise ValueError("Stage-2 sampler seed는 0 이상 int여야 합니다")
        _require_sha256(manifest_bundle_sha256, label="manifest bundle SHA")
        _require_sha256(sampler_receipt_sha256, label="sampler receipt SHA")
        selected = tuple(record for record in records if record.split == split)
        if not selected:
            raise ValueError(f"Stage-2 sampler {split} record가 비었습니다")
        indices = [int(record.dataset_index) for record in selected]
        if len(indices) != len(set(indices)):
            raise ValueError("Stage-2 sampler dataset_index가 중복됩니다")
        by_family_component: dict[str, dict[str, tuple[Stage2SamplerRecord, ...]]] = {}
        for family in STAGE2_2KHZ_SOURCE_FAMILIES:
            components: dict[str, list[Stage2SamplerRecord]] = defaultdict(list)
            for record in selected:
                if record.source_family == family:
                    components[record.component_id].append(record)
            if not components:
                raise ValueError(f"Stage-2 sampler에 {family} family/component가 없습니다")
            by_family_component[family] = {
                key: tuple(sorted(values, key=lambda item: item.dataset_index))
                for key, values in sorted(components.items())
            }
        self.records = selected
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.split = split
        self.manifest_bundle_sha256 = manifest_bundle_sha256
        self.sampler_receipt_sha256 = sampler_receipt_sha256
        self._by_family_component = by_family_component

    @property
    def quota_per_family(self) -> int:
        return self.batch_size // 4

    def _offset(self, label: str, modulo: int) -> int:
        digest = hashlib.sha256(f"{self.seed}:{label}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % int(modulo)

    def batch_records(
        self,
        global_step: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        worker_id: int = 0,
        num_workers: int = 1,
    ) -> tuple[Stage2SamplerRecord, ...]:
        if isinstance(global_step, bool) or int(global_step) < 0:
            raise ValueError("Stage-2 global_step은 0 이상 int여야 합니다")
        for value, label, minimum in (
            (rank, "rank", 0),
            (world_size, "world_size", 1),
            (worker_id, "worker_id", 0),
            (num_workers, "num_workers", 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Stage-2 {label} 값이 잘못됐습니다")
        if rank >= world_size or worker_id >= num_workers:
            raise ValueError("Stage-2 rank/worker가 world_size/num_workers 범위 밖입니다")
        # worker는 이미 정해진 global draw를 decode만 한다. RNG identity에 worker ID를
        # 넣으면 worker 수 변경이나 재개 시 augmentation/source가 달라진다.
        del worker_id, num_workers
        chosen: list[Stage2SamplerRecord] = []
        draw_index = int(global_step) * int(world_size) + int(rank)
        cursor_base = draw_index * self.quota_per_family
        for family in STAGE2_2KHZ_SOURCE_FAMILIES:
            components = self._by_family_component[family]
            component_ids = tuple(components)
            component_offset = self._offset(f"component:{family}", len(component_ids))
            for slot in range(self.quota_per_family):
                cursor = cursor_base + slot
                component_id = component_ids[(component_offset + cursor) % len(component_ids)]
                items = components[component_id]
                item_offset = self._offset(f"item:{family}:{component_id}", len(items))
                cycle = cursor // len(component_ids)
                chosen.append(items[(item_offset + cycle) % len(items)])
        # family별 덩어리를 남기지 않되, global step으로만 순서를 결정한다.
        return tuple(
            sorted(
                chosen,
                key=lambda record: hashlib.sha256(
                    f"{self.seed}:{draw_index}:{record.dataset_index}".encode("utf-8")
                ).digest(),
            )
        )

    def batch_indices(
        self, global_step: int, *, rank: int = 0, world_size: int = 1
    ) -> tuple[int, ...]:
        return tuple(
            record.dataset_index
            for record in self.batch_records(global_step, rank=rank, world_size=world_size)
        )

    def global_sample_indices(
        self, global_step: int, *, rank: int = 0, world_size: int = 1
    ) -> tuple[int, ...]:
        if isinstance(global_step, bool) or int(global_step) < 0:
            raise ValueError("Stage-2 global_step은 0 이상 int여야 합니다")
        if (
            isinstance(rank, bool)
            or isinstance(world_size, bool)
            or not isinstance(rank, int)
            or not isinstance(world_size, int)
            or world_size < 1
            or not 0 <= rank < world_size
        ):
            raise ValueError("Stage-2 DDP rank/world_size가 잘못됐습니다")
        draw_index = int(global_step) * int(world_size) + int(rank)
        start = draw_index * self.batch_size
        return tuple(range(start, start + self.batch_size))

    def augmentation_seeds(
        self, global_step: int, *, rank: int = 0, world_size: int = 1
    ) -> tuple[int, ...]:
        records = self.batch_records(global_step, rank=rank, world_size=world_size)
        globals_ = self.global_sample_indices(global_step, rank=rank, world_size=world_size)
        return tuple(
            int.from_bytes(
                hashlib.sha256(
                    f"{self.seed}:{global_index}:{record.source_sha256}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            for record, global_index in zip(records, globals_, strict=True)
        )

    def expected_receipt_payload(self) -> dict[str, Any]:
        components = {
            family: list(self._by_family_component[family])
            for family in STAGE2_2KHZ_SOURCE_FAMILIES
        }
        component_set_sha = hashlib.sha256(_canonical_json(components)).hexdigest()
        return {
            "schema": STAGE2_2KHZ_SAMPLER_SCHEMA,
            "status": "PASS",
            "canonical_pretrain_eligible": True,
            "control_band_contract_sha256": (
                Stage2TwoKilohertzContract.canonical().digest()
            ),
            "manifest_bundle_sha256": self.manifest_bundle_sha256,
            "algorithm": "family_then_component_global_step_round_robin_v1",
            "global_sample_index_deterministic": True,
            "augmentation_seed_source": "seed_global_sample_index_source_sha256_v1",
            "dataloader_worker_id_affects_draw": False,
            "ddp_draw_identity": "global_step_world_size_rank_v1",
            "source_mix_random_tag_selector_used": False,
            "split": self.split,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "family_counts": {
                family: self.quota_per_family for family in STAGE2_2KHZ_SOURCE_FAMILIES
            },
            "component_sets_sha256": component_set_sha,
            "actual_batch_family_recheck_required": True,
            "actual_target_density_recheck_required": True,
        }


@dataclass(frozen=True)
class Stage2ActualBatchIdentity:
    """dataset index tensor와 같이 전달되는 실제 batch identity."""

    source_families: tuple[str, ...]
    component_ids: tuple[str, ...]
    splits: tuple[str, ...]
    source_sha256: tuple[str, ...]
    dataset_indices: tuple[int, ...]
    global_sample_indices: tuple[int, ...]
    manifest_row_sha256: tuple[str, ...]
    augmentation_seeds: tuple[int, ...]
    manifest_bundle_sha256: str
    sampler_receipt_sha256: str
    global_step: int

    def __post_init__(self) -> None:
        lengths = {
            len(self.source_families),
            len(self.component_ids),
            len(self.splits),
            len(self.source_sha256),
            len(self.dataset_indices),
            len(self.global_sample_indices),
            len(self.manifest_row_sha256),
            len(self.augmentation_seeds),
        }
        if len(lengths) != 1 or not self.source_families:
            raise ValueError("Stage-2 actual batch identity tuple 길이가 다릅니다")
        _require_sha256(self.manifest_bundle_sha256, label="batch manifest SHA")
        _require_sha256(self.sampler_receipt_sha256, label="batch sampler receipt SHA")
        if isinstance(self.global_step, bool) or int(self.global_step) < 0:
            raise ValueError("Stage-2 batch global_step이 잘못됐습니다")
        if any(split != "train" for split in self.splits):
            raise ValueError("Stage-2 train step에 val/test item을 넘길 수 없습니다")
        for value in self.source_sha256:
            _require_sha256(value, label="batch source SHA")
        for value in self.manifest_row_sha256:
            _require_sha256(value, label="batch manifest row SHA")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (*self.dataset_indices, *self.global_sample_indices, *self.augmentation_seeds)
        ):
            raise ValueError("Stage-2 dataset/global/augmentation identity가 잘못됐습니다")

    @classmethod
    def from_sampler(
        cls,
        sampler: Stage2FamilyComponentBatchSampler,
        *,
        global_step: int,
        rank: int = 0,
        world_size: int = 1,
        worker_id: int = 0,
        num_workers: int = 1,
    ) -> "Stage2ActualBatchIdentity":
        records = sampler.batch_records(
            global_step,
            rank=rank,
            world_size=world_size,
            worker_id=worker_id,
            num_workers=num_workers,
        )
        global_indices = sampler.global_sample_indices(
            global_step, rank=rank, world_size=world_size
        )
        augmentation_seeds = sampler.augmentation_seeds(
            global_step, rank=rank, world_size=world_size
        )
        row_sha = tuple(
            hashlib.sha256(
                _canonical_json(
                    {
                        "dataset_index": item.dataset_index,
                        "source_family": item.source_family,
                        "component_id": item.component_id,
                        "split": item.split,
                        "source_sha256": item.source_sha256,
                    }
                )
            ).hexdigest()
            for item in records
        )
        return cls(
            source_families=tuple(item.source_family for item in records),
            component_ids=tuple(item.component_id for item in records),
            splits=tuple(item.split for item in records),
            source_sha256=tuple(item.source_sha256 for item in records),
            dataset_indices=tuple(item.dataset_index for item in records),
            global_sample_indices=global_indices,
            manifest_row_sha256=row_sha,
            augmentation_seeds=augmentation_seeds,
            manifest_bundle_sha256=sampler.manifest_bundle_sha256,
            sampler_receipt_sha256=sampler.sampler_receipt_sha256,
            global_step=int(global_step),
        )


@dataclass(frozen=True)
class Stage2TensorBatch:
    """tensor timeline과 manifest row/global RNG identity를 분리해 보존한다."""

    causal: CausalPrefixBatchV1
    dataset_indices: tuple[int, ...]
    manifest_row_sha256: tuple[str, ...]
    augmentation_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.causal, CausalPrefixBatchV1):
            raise TypeError("Stage2TensorBatch.causal에는 CausalPrefixBatchV1이 필요합니다")
        batch = int(self.causal.x_prefix.shape[0])
        if not (
            len(self.dataset_indices)
            == len(self.manifest_row_sha256)
            == len(self.augmentation_seeds)
            == batch
        ):
            raise ValueError("Stage2TensorBatch identity 길이가 tensor batch와 다릅니다")
        for value in self.manifest_row_sha256:
            _require_sha256(value, label="tensor manifest row SHA")


@dataclass(frozen=True)
class Stage2PrefixResult:
    y_prefix: torch.Tensor
    y_target: torch.Tensor
    primary_target: torch.Tensor
    secondary_target: torch.Tensor
    error_target: torch.Tensor
    final_state: Any
    binding_sha256: str
    prefix_samples: int
    target_samples: int
    schema_version: str = STAGE2_2KHZ_PREFIX_ADAPTER_SCHEMA


class Stage2CausalPrefixAdapter(nn.Module):
    """Stage-2 binding에만 열리는 causal ``P*n + S*y`` adapter."""

    def __init__(
        self,
        binding: Stage2TwoKilohertzPlantBinding,
        *,
        _allow_test_fixture: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(binding, Stage2TwoKilohertzPlantBinding):
            raise TypeError("Stage2TwoKilohertzPlantBinding이 필요합니다")
        if binding.fixture_only and not _allow_test_fixture:
            raise ValueError("test fixture Stage-2 binding은 production adapter에 쓸 수 없습니다")
        self.binding = binding
        self.primary_path = CausalFIRPath(binding.primary_operator)
        self.secondary_path = CausalFIRPath(binding.secondary_operator)
        self.block_size = int(binding.block_size)
        self.binding_sha256 = binding.digest()

    @classmethod
    def from_verified_binding(
        cls, binding: Stage2TwoKilohertzPlantBinding
    ) -> "Stage2CausalPrefixAdapter":
        return cls(binding)

    @classmethod
    def _for_test_fixture(
        cls, binding: Stage2TwoKilohertzPlantBinding
    ) -> "Stage2CausalPrefixAdapter":
        return cls(binding, _allow_test_fixture=True)

    @staticmethod
    def _tensor(value: torch.Tensor, *, label: str, channels: int | None = None) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 3
            or int(value.shape[0]) < 1
            or int(value.shape[1]) < 1
            or int(value.shape[-1]) < 1
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"{label}는 finite floating [B,C,T] tensor여야 합니다")
        if channels is not None and int(value.shape[1]) != int(channels):
            raise ValueError(f"{label} channel 수가 {channels}가 아닙니다")
        return value

    def _validate_batch(self, batch: CausalPrefixBatchV1) -> tuple[int, int, int]:
        if not isinstance(batch, CausalPrefixBatchV1):
            raise TypeError("CausalPrefixBatchV1이 필요합니다")
        prefix = self._tensor(batch.x_prefix, label="x_prefix")
        target = self._tensor(batch.x_target, label="x_target")
        batch_size = int(prefix.shape[0])
        prefix_samples = int(prefix.shape[-1])
        target_samples = int(target.shape[-1])
        if target.shape[:2] != prefix.shape[:2] or target.device != prefix.device:
            raise ValueError("Stage-2 prefix/target batch/channel/device가 다릅니다")
        if prefix_samples % 256 or target_samples % 256:
            raise ValueError("Stage-2 prefix/target은 256-sample block의 정수배여야 합니다")
        if prefix_samples < int(self.binding.required_prefix_samples):
            raise ValueError("Stage-2 prefix가 P/S delay+FIR history보다 짧습니다")
        if batch.training_timing_contract_sha256 != self.binding.training_timing_contract_sha256:
            raise ValueError("Stage-2 batch timing-v2 SHA와 binding이 다릅니다")
        if batch.source_sha256 != batch.clean_playback_source_sha256:
            raise ValueError("Stage-2 controller/clean playback source lineage가 다릅니다")
        if len(batch.source_sha256) != batch_size:
            raise ValueError("Stage-2 source SHA tuple 길이가 batch와 다릅니다")
        for value in batch.source_sha256:
            _require_sha256(value, label="Stage-2 prefix source SHA")
        lead = int(self.binding.training_timing_contract.digital_reference_lead_samples)
        clean = self._tensor(
            batch.clean_playback_timeline,
            label="clean_playback_timeline",
            channels=1,
        )
        if clean.shape != (batch_size, 1, prefix_samples + target_samples + lead):
            raise ValueError("Stage-2 clean timeline 길이가 prefix+target+derived lead가 아닙니다")
        preview = self._tensor(
            batch.controller_reference_preaugmentation,
            label="controller_reference_preaugmentation",
            channels=1,
        )
        expected_preview = clean[..., lead : lead + prefix_samples + target_samples]
        if preview.shape != expected_preview.shape or not torch.equal(
            preview.float(), expected_preview.float()
        ):
            raise ValueError("Stage-2 digital reference preview가 derived lead와 exact하지 않습니다")
        if tuple(batch.segment_prefix_start_samples) != (0,) * batch_size:
            raise ValueError("Stage-2 zero state는 segment sample 0에서만 시작합니다")
        if tuple(batch.segment_target_start_samples) != (prefix_samples,) * batch_size:
            raise ValueError("Stage-2 target은 같은 segment prefix 직후에서 시작해야 합니다")
        if len(batch.global_sample_indices) != batch_size or any(
            isinstance(value, bool) or int(value) < 0 for value in batch.global_sample_indices
        ):
            raise ValueError("Stage-2 global sample index tuple이 잘못됐습니다")
        origin = batch.state_origin
        if not isinstance(origin, CausalPrefixStateOriginV1):
            raise TypeError("Stage-2 prefix에 CausalPrefixStateOriginV1이 필요합니다")
        if origin.binding_sha256 != self.binding_sha256 or origin.source_sha256 != batch.source_sha256:
            raise ValueError("Stage-2 state origin이 binding/source와 다릅니다")
        return batch_size, prefix_samples, target_samples

    def _stream(
        self, model: CausalStreamingControlModelV1, value: torch.Tensor, *, state: Any
    ) -> tuple[torch.Tensor, Any]:
        hop = getattr(model, "hop", None)
        if isinstance(hop, bool) or not isinstance(hop, int) or hop <= 0 or 256 % hop:
            raise ValueError("Stage-2 model hop은 256을 정확히 나누는 양수 int여야 합니다")
        step = getattr(model, "streaming_step", None)
        if not callable(step):
            raise TypeError("Stage-2 model에 streaming_step API가 없습니다")
        outputs: list[torch.Tensor] = []
        for start in range(0, int(value.shape[-1]), 256):
            block = value[..., start : start + 256]
            output, state = step(block, state)
            output = self._tensor(output, label="streaming_step output", channels=1)
            if output.shape != (int(block.shape[0]), 1, int(block.shape[-1])):
                raise ValueError("Stage-2 streaming_step output shape가 input block과 다릅니다")
            outputs.append(output.float())
        return torch.cat(outputs, dim=-1), state

    def forward(
        self, model: CausalStreamingControlModelV1, batch: CausalPrefixBatchV1
    ) -> Stage2PrefixResult:
        batch_size, prefix_samples, target_samples = self._validate_batch(batch)
        initializer = getattr(model, "init_states", None)
        if not callable(initializer):
            raise TypeError("Stage-2 zero-reset model에 init_states API가 필요합니다")
        state = initializer(batch=batch_size, device=batch.x_prefix.device)
        y_prefix, state = self._stream(model, batch.x_prefix, state=state)
        y_target, state = self._stream(model, batch.x_target, state=state)
        y_full = torch.cat((y_prefix, y_target), dim=-1)
        total = prefix_samples + target_samples
        clean = batch.clean_playback_timeline[..., :total].float()
        primary = self.primary_path(clean)[..., prefix_samples:].float()
        secondary = self.secondary_path(y_full)[..., prefix_samples:].float()
        if primary.shape != y_target.shape or secondary.shape != y_target.shape:
            raise RuntimeError("Stage-2 P/S target crop shape가 controller output과 다릅니다")
        error = primary + secondary
        return Stage2PrefixResult(
            y_prefix=y_prefix,
            y_target=y_target,
            primary_target=primary,
            secondary_target=secondary,
            error_target=error,
            final_state=state,
            binding_sha256=self.binding_sha256,
            prefix_samples=prefix_samples,
            target_samples=target_samples,
        )


@dataclass(frozen=True)
class Stage2TrainStepResult:
    loss: torch.Tensor
    metrics: dict[str, float]
    causal_result: Stage2PrefixResult
    plant_binding_sha256: str
    manifest_bundle_sha256: str
    sampler_receipt_sha256: str
    control_band_contract_sha256: str
    schema_version: str = STAGE2_2KHZ_TRAINER_ADAPTER_SCHEMA


class Stage2TwoKilohertzTrainerAdapter:
    """optimizer 직전의 최소 Stage-2 pretrain tensor consumer."""

    def __init__(
        self,
        adapter: Stage2CausalPrefixAdapter,
        criterion: Stage2TwoKilohertzLoss,
        *,
        manifest_bundle_sha256: str,
        sampler_receipt_sha256: str,
        _allow_test_fixture: bool = False,
    ) -> None:
        if not isinstance(adapter, Stage2CausalPrefixAdapter):
            raise TypeError("Stage2CausalPrefixAdapter가 필요합니다")
        if not isinstance(criterion, Stage2TwoKilohertzLoss):
            raise TypeError("Stage2TwoKilohertzLoss가 필요합니다")
        if adapter.binding.fixture_only and not _allow_test_fixture:
            raise ValueError("fixture Stage-2 binding으로 production trainer adapter를 만들 수 없습니다")
        _require_sha256(manifest_bundle_sha256, label="trainer manifest SHA")
        _require_sha256(sampler_receipt_sha256, label="trainer sampler receipt SHA")
        contract = Stage2TwoKilohertzContract.canonical()
        if adapter.binding.control_band_contract != contract:
            raise ValueError("Stage-2 adapter binding contract가 canonical과 다릅니다")
        if criterion.control_band_contract != contract:
            raise ValueError("Stage-2 criterion contract가 canonical과 다릅니다")
        if criterion.loss_config.family_balanced_sampler_receipt_sha256 != sampler_receipt_sha256:
            raise ValueError("Stage-2 criterion/sampler receipt SHA가 다릅니다")
        self.adapter = adapter
        self.criterion = criterion
        self.manifest_bundle_sha256 = manifest_bundle_sha256
        self.sampler_receipt_sha256 = sampler_receipt_sha256
        self.control_band_contract_sha256 = contract.digest()

    @classmethod
    def from_verified_components(
        cls,
        adapter: Stage2CausalPrefixAdapter,
        criterion: Stage2TwoKilohertzLoss,
        *,
        manifest_bundle_sha256: str,
        sampler_receipt_sha256: str,
    ) -> "Stage2TwoKilohertzTrainerAdapter":
        return cls(
            adapter,
            criterion,
            manifest_bundle_sha256=manifest_bundle_sha256,
            sampler_receipt_sha256=sampler_receipt_sha256,
        )

    @classmethod
    def _for_test_fixture(
        cls,
        adapter: Stage2CausalPrefixAdapter,
        criterion: Stage2TwoKilohertzLoss,
        *,
        manifest_bundle_sha256: str,
        sampler_receipt_sha256: str,
    ) -> "Stage2TwoKilohertzTrainerAdapter":
        return cls(
            adapter,
            criterion,
            manifest_bundle_sha256=manifest_bundle_sha256,
            sampler_receipt_sha256=sampler_receipt_sha256,
            _allow_test_fixture=True,
        )

    def compute_loss(
        self,
        model: CausalStreamingControlModelV1,
        batch: Stage2TensorBatch,
        identity: Stage2ActualBatchIdentity,
    ) -> Stage2TrainStepResult:
        if not isinstance(batch, Stage2TensorBatch):
            raise TypeError("Stage2TensorBatch가 필요합니다")
        if identity.manifest_bundle_sha256 != self.manifest_bundle_sha256:
            raise ValueError("Stage-2 actual batch manifest SHA가 trainer와 다릅니다")
        if identity.sampler_receipt_sha256 != self.sampler_receipt_sha256:
            raise ValueError("Stage-2 actual batch sampler receipt SHA가 trainer와 다릅니다")
        if identity.source_sha256 != batch.causal.source_sha256:
            raise ValueError("Stage-2 actual batch identity/source tensor lineage가 다릅니다")
        if identity.dataset_indices != batch.dataset_indices:
            raise ValueError("Stage-2 manifest dataset index가 tensor row identity와 다릅니다")
        if identity.global_sample_indices != batch.causal.global_sample_indices:
            raise ValueError("Stage-2 RNG global sample index가 tensor timeline과 다릅니다")
        if identity.manifest_row_sha256 != batch.manifest_row_sha256:
            raise ValueError("Stage-2 manifest row SHA가 tensor batch와 다릅니다")
        if identity.augmentation_seeds != batch.augmentation_seeds:
            raise ValueError("Stage-2 augmentation seed가 global sample identity와 다릅니다")
        causal = self.adapter(model, batch.causal)
        if not torch.equal(
            causal.error_target, causal.primary_target + causal.secondary_target
        ):
            raise RuntimeError("Stage-2 causal adapter의 e=P*n+S*y exact relation이 깨졌습니다")
        loss, metrics = self.criterion(
            causal.y_target,
            causal.primary_target,
            causal.secondary_target,
            source_families=identity.source_families,
        )
        metrics = {str(key): float(value) for key, value in metrics.items()}
        metrics.update(
            {
                "stage2_consumer_causal_prefix_used": 1.0,
                "stage2_consumer_actual_primary_output_used": 1.0,
                "stage2_consumer_actual_secondary_output_used": 1.0,
                "stage2_consumer_actual_batch_identity_used": 1.0,
                "stage2_consumer_test_selection_used": 0.0,
                "stage2_consumer_fixture_only": float(self.adapter.binding.fixture_only),
            }
        )
        return Stage2TrainStepResult(
            loss=loss,
            metrics=metrics,
            causal_result=causal,
            plant_binding_sha256=self.adapter.binding_sha256,
            manifest_bundle_sha256=self.manifest_bundle_sha256,
            sampler_receipt_sha256=self.sampler_receipt_sha256,
            control_band_contract_sha256=self.control_band_contract_sha256,
        )


__all__ = [
    "STAGE2_2KHZ_PREFIX_ADAPTER_SCHEMA",
    "STAGE2_2KHZ_SAMPLER_SCHEMA",
    "STAGE2_2KHZ_TRAINER_ADAPTER_SCHEMA",
    "STAGE2_MAX_ACTUATOR_ABS",
    "STAGE2_MAX_ACTUATOR_PEAK_PCM",
    "Stage2ActualBatchIdentity",
    "Stage2CausalPrefixAdapter",
    "Stage2FamilyComponentBatchSampler",
    "Stage2PrefixResult",
    "Stage2SamplerRecord",
    "Stage2TensorBatch",
    "Stage2TrainStepResult",
    "Stage2TwoKilohertzTrainerAdapter",
    "require_stage2_actuator_limit",
]
