"""v3 full-octave의 인과 학습·matched FxLMS 소비 경계.

이 모듈은 다음 세 가지를 하나의 텐서 흐름으로 묶는다.

1. :class:`CausalSecondaryPrefixAdapterV1`가 prefix부터 controller를 streaming으로
   전진시켜 ``P*n + S*y``의 valid target crop을 만든다.
2. :class:`BroadbandFullOctaveLossPrimitiveV3`가 그 *같은* ``y``, ``P*n``, ``S*y``로
   125/250/500/1k/2k/4k/8k Hz loss를 계산한다.
3. 같은 reference/P/S/prefix/block 설정으로 FxLMS를 처음부터 target 끝까지 실행해
   surrogate matched evaluator에 넘긴다.

여기서 ``production``은 raw-bound non-fixture binding을 뜻할 뿐, 현재 Jetson의
physical P/S가 canonical이라는 뜻이 아니다. 현 저장소에는 아직 그러한 binding
publisher와 DNH/lineage authority가 없으므로 public constructor는 test fixture를
거부한다. 이 모듈은 GPU, DataLoader, optimizer, audio I/O, run directory를 만들지
않는다. 실제 canonical training launch는 별도 raw-first admission이 계속 차단한다.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..dsp.control_band_contract import BroadbandFullOctaveContractV3
from ..eval.full_octave_v3 import FullOctaveV3MatchedSegment
from ..eval.fxlms_baseline import run_fxlms_offline
from ..losses.broadband_loss import (
    BroadbandFullOctaveLossPrimitiveV3,
)
from .causal_secondary_prefix_adapter_v1 import (
    CausalPrefixBatchV1,
    CausalSecondaryPrefixAdapterV1,
    CausalSecondaryPrefixResultV1,
    CausalStreamingControlModelV1,
)


FULL_OCTAVE_V3_TRAINER_CONSUMER_SCHEMA = "full_octave_v3_trainer_consumer_v1"
FULL_OCTAVE_V3_MATCHED_FXLMS_SCHEMA = "full_octave_v3_matched_fxlms_v1"
FULL_OCTAVE_V3_SURROGATE_ROLE = "surrogate_causal_ps_not_canonical_training"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if not _SHA256.fullmatch(text):
        raise ValueError(f"{label}는 lowercase SHA-256이어야 합니다")
    return text


@dataclass(frozen=True)
class FullOctaveV3TrainStepResult:
    """optimizer 밖에서 계산한 한 causal full-octave loss step의 receipt."""

    loss: torch.Tensor
    metrics: dict[str, float]
    causal_result: CausalSecondaryPrefixResultV1
    causal_plant_binding_sha256: str
    control_band_contract_sha256: str
    role: str = FULL_OCTAVE_V3_SURROGATE_ROLE
    schema_version: str = FULL_OCTAVE_V3_TRAINER_CONSUMER_SCHEMA


@dataclass(frozen=True)
class FullOctaveV3FxLMSConfig:
    """Deep-ANC와 동일 prefix/block P/S를 쓰는 deterministic FxLMS 설정."""

    block_size: int = 256
    control_length: int = 256
    mu: float = 0.05
    leakage: float = 1.0e-6
    schema_version: str = FULL_OCTAVE_V3_MATCHED_FXLMS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != FULL_OCTAVE_V3_MATCHED_FXLMS_SCHEMA:
            raise ValueError("v3 matched FxLMS schema가 다릅니다")
        if int(self.block_size) != 256:
            raise ValueError("v3 matched FxLMS block은 canonical 256 samples여야 합니다")
        if int(self.control_length) < 1:
            raise ValueError("v3 matched FxLMS control_length는 양수여야 합니다")
        if not math.isfinite(float(self.mu)) or float(self.mu) < 0.0:
            raise ValueError("v3 matched FxLMS mu는 유한한 0 이상 값이어야 합니다")
        if not math.isfinite(float(self.leakage)) or not 0.0 <= float(self.leakage) < 1.0:
            raise ValueError("v3 matched FxLMS leakage는 [0,1)이어야 합니다")


@dataclass(frozen=True)
class FullOctaveV3EvaluationIdentity:
    """surrogate matched segment의 family/group provenance.

    identity는 실제 physical session을 대체하지 않는다. 단위 평가에서도 source
    family와 독립 group이 빠지는 것을 막아, pure evaluator가 같은 객체를 그대로
    받을 수 있게 한다.
    """

    session_id: str
    source_family: str
    group_id: str
    error_position_id: str = "surrogate_center"

    def __post_init__(self) -> None:
        for field in ("session_id", "source_family", "group_id", "error_position_id"):
            if not str(getattr(self, field)).strip():
                raise ValueError(f"v3 evaluation identity {field}가 비었습니다")
        if self.source_family not in BroadbandFullOctaveContractV3.canonical().source_families:
            raise ValueError(f"지원하지 않는 v3 source family: {self.source_family!r}")


class FullOctaveV3TrainerConsumer:
    """인과 prefix 결과를 v3 loss에 직접 연결하는 pure training-step consumer.

    ``CausalSecondaryPrefixAdapterV1``의 ``error_target``만 받아 다시 만든 값으로
    손실을 계산하는 우회를 허용하지 않는다. 손실에는 반드시 adapter가 생성한
    ``y_target``, ``primary_target=P*n``, ``secondary_target=S*y``를 각각 전달한다.
    """

    def __init__(
        self,
        adapter: CausalSecondaryPrefixAdapterV1,
        loss: BroadbandFullOctaveLossPrimitiveV3,
        *,
        _allow_test_fixture: bool = False,
    ) -> None:
        if not isinstance(adapter, CausalSecondaryPrefixAdapterV1):
            raise TypeError("CausalSecondaryPrefixAdapterV1이 필요합니다")
        if not isinstance(loss, BroadbandFullOctaveLossPrimitiveV3):
            raise TypeError("BroadbandFullOctaveLossPrimitiveV3가 필요합니다")
        if adapter.binding.fixture_only and not _allow_test_fixture:
            raise ValueError(
                "raw-bound production authority가 없는 test fixture는 v3 trainer consumer에 허용되지 않습니다"
            )
        contract = BroadbandFullOctaveContractV3.canonical()
        if adapter.binding.control_band_contract != contract:
            raise ValueError("causal prefix binding이 canonical full-octave v3 계약이 아닙니다")
        if loss.control_band_contract != contract:
            raise ValueError("v3 loss가 canonical full-octave v3 계약이 아닙니다")
        if adapter.binding.control_band_contract_sha256 != loss.control_band_contract_sha256:
            raise ValueError("causal prefix binding/loss control-band SHA가 다릅니다")
        if int(adapter.binding.block_size) != int(adapter.block_size):
            raise ValueError("causal prefix binding/adapter block size가 다릅니다")
        self.adapter = adapter
        self.loss = loss
        self.control_band_contract_sha256 = contract.digest()

    @classmethod
    def from_verified_components(
        cls,
        adapter: CausalSecondaryPrefixAdapterV1,
        loss: BroadbandFullOctaveLossPrimitiveV3,
    ) -> "FullOctaveV3TrainerConsumer":
        """future raw-bound non-fixture publisher만 쓰는 production entrypoint."""

        return cls(adapter, loss)

    @classmethod
    def _for_test_fixture(
        cls,
        adapter: CausalSecondaryPrefixAdapterV1,
        loss: BroadbandFullOctaveLossPrimitiveV3,
    ) -> "FullOctaveV3TrainerConsumer":
        """CPU regression 전용. canonical training admission에는 절대 쓰지 않는다."""

        return cls(adapter, loss, _allow_test_fixture=True)

    def compute_loss(
        self,
        model: CausalStreamingControlModelV1,
        batch: CausalPrefixBatchV1,
    ) -> FullOctaveV3TrainStepResult:
        """prefix 포함 P*n/S*y를 계산하고 7-octave loss를 반환한다.

        optimizer step을 호출하지 않으므로 caller가 mixed precision, accumulation,
        distributed reduction을 명시적으로 책임진다. 모든 P/S convolution과 loss는
        adapter/loss primitive가 FP32로 고정한다.
        """

        causal = self.adapter(model, batch)
        expected_error = causal.primary_target + causal.secondary_target
        if not torch.equal(causal.error_target, expected_error):
            raise RuntimeError("causal adapter error target이 P*n + S*y와 exact하지 않습니다")
        loss, metrics = self.loss(
            causal.y_target,
            causal.primary_target,
            causal.secondary_target,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("full-octave v3 loss가 finite하지 않습니다")
        metrics = {str(key): float(value) for key, value in metrics.items()}
        metrics.update(
            {
                "v3_consumer_causal_prefix_used": 1.0,
                "v3_consumer_actual_secondary_output_used": 1.0,
                "v3_consumer_equal_octave_count": float(
                    len(self.loss.objective_bands_hz)
                ),
                "v3_consumer_fixture_only": float(self.adapter.binding.fixture_only),
                "v3_consumer_canonical_training_claim": 0.0,
            }
        )
        return FullOctaveV3TrainStepResult(
            loss=loss,
            metrics=metrics,
            causal_result=causal,
            causal_plant_binding_sha256=self.adapter.binding_sha256,
            control_band_contract_sha256=self.control_band_contract_sha256,
        )


class FullOctaveV3MatchedFxLMSEvaluator:
    """Deep-ANC/FxLMS를 동일 causal P/S/prefix로 만들어 주는 surrogate evaluator.

    physical recording의 FxLMS 비교를 대신하지 않는다. 이 클래스가 생성한 segment는
    ``surrogate_matched_causal_ps_not_physical``으로 고정돼 full-octave raw G4와
    자동 결합되지 않는다.
    """

    def __init__(
        self,
        adapter: CausalSecondaryPrefixAdapterV1,
        *,
        fxlms: FullOctaveV3FxLMSConfig | None = None,
        _allow_test_fixture: bool = False,
    ) -> None:
        if not isinstance(adapter, CausalSecondaryPrefixAdapterV1):
            raise TypeError("CausalSecondaryPrefixAdapterV1이 필요합니다")
        if adapter.binding.fixture_only and not _allow_test_fixture:
            raise ValueError(
                "raw-bound production authority가 없는 test fixture는 v3 matched FxLMS evaluator에 허용되지 않습니다"
            )
        cfg = FullOctaveV3FxLMSConfig() if fxlms is None else fxlms
        if not isinstance(cfg, FullOctaveV3FxLMSConfig):
            raise TypeError("FullOctaveV3FxLMSConfig가 필요합니다")
        if int(cfg.block_size) != int(adapter.block_size):
            raise ValueError("FxLMS/Deep-ANC의 callback block size가 다릅니다")
        self.adapter = adapter
        self.fxlms = cfg

    @classmethod
    def from_verified_binding(
        cls,
        adapter: CausalSecondaryPrefixAdapterV1,
        *,
        fxlms: FullOctaveV3FxLMSConfig | None = None,
    ) -> "FullOctaveV3MatchedFxLMSEvaluator":
        return cls(adapter, fxlms=fxlms)

    @classmethod
    def _for_test_fixture(
        cls,
        adapter: CausalSecondaryPrefixAdapterV1,
        *,
        fxlms: FullOctaveV3FxLMSConfig | None = None,
    ) -> "FullOctaveV3MatchedFxLMSEvaluator":
        return cls(adapter, fxlms=fxlms, _allow_test_fixture=True)

    def _full_reference_and_disturbance(
        self,
        batch: CausalPrefixBatchV1,
        *,
        prefix_samples: int,
        target_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total = int(prefix_samples) + int(target_samples)
        reference_index = int(self.adapter.binding.reference_channel_index)
        x_full = torch.cat((batch.x_prefix, batch.x_target), dim=-1)
        if int(x_full.shape[-1]) != total:
            raise RuntimeError("matched FxLMS reference prefix/target 길이가 다릅니다")
        reference = x_full[:, reference_index : reference_index + 1].float()
        clean = batch.clean_playback_timeline[..., :total].float()
        disturbance = self.adapter.primary_path(clean).float()
        if reference.shape != disturbance.shape:
            raise RuntimeError("matched FxLMS reference/P*n shape가 다릅니다")
        return reference, disturbance

    def evaluate(
        self,
        model: CausalStreamingControlModelV1,
        batch: CausalPrefixBatchV1,
        *,
        identity: FullOctaveV3EvaluationIdentity,
    ) -> tuple[CausalSecondaryPrefixResultV1, FullOctaveV3MatchedSegment]:
        """하나의 명시 identity에 대한 matched target crop을 만든다.

        evaluator가 batch 내부 item에 ``-itemN`` suffix를 붙여 session/group을
        만들어서는 안 된다. 그것은 하나의 source lineage를 여러 독립 bootstrap
        group으로 위장할 수 있기 때문이다. per-item identity receipt가 별도로
        구현되기 전에는 batch size 1만 허용한다.
        """

        if (
            isinstance(batch, CausalPrefixBatchV1)
            and isinstance(batch.x_prefix, torch.Tensor)
            and batch.x_prefix.ndim >= 1
            and int(batch.x_prefix.shape[0]) != 1
        ):
            raise ValueError(
                "matched FxLMS evaluator는 독립 session/group을 임의 생성하지 않도록 "
                "batch size=1만 허용합니다"
            )
        was_training = bool(getattr(model, "training", False))
        try:
            model.eval()
            with torch.no_grad():
                causal = self.adapter(model, batch)
                reference, disturbance = self._full_reference_and_disturbance(
                    batch,
                    prefix_samples=causal.prefix_samples,
                    target_samples=causal.target_samples,
                )
        finally:
            model.train(was_training)

        prefix = int(causal.prefix_samples)
        target = int(causal.target_samples)
        if int(reference.shape[0]) != 1:  # pragma: no cover - adapter shape guard의 방어선
            raise RuntimeError("matched FxLMS evaluator batch size 방어선이 우회됐습니다")
        if int(reference.shape[-1]) != prefix + target:
            raise RuntimeError("matched FxLMS full reference 길이가 target crop과 다릅니다")
        if not torch.equal(causal.primary_target, disturbance[..., prefix:]):
            raise RuntimeError("Deep-ANC/FxLMS disturbance P*n target crop이 다릅니다")
        secondary = self.adapter.binding.secondary_operator
        baseline = run_fxlms_offline(
            reference[0, 0].detach().cpu().numpy(),
            disturbance[0, 0].detach().cpu().numpy(),
            secondary.post_onset_fir,
            secondary.base_delay_samples,
            block=int(self.fxlms.block_size),
            control_len=int(self.fxlms.control_length),
            mu=float(self.fxlms.mu),
            leakage=float(self.fxlms.leakage),
        )
        fxlms_error = np.asarray(baseline["e"], dtype=np.float32).reshape(-1)
        if fxlms_error.shape != (prefix + target,) or not np.all(np.isfinite(fxlms_error)):
            raise RuntimeError("matched FxLMS error trajectory가 prefix/target 계약을 지키지 않습니다")
        segment = FullOctaveV3MatchedSegment(
            session_id=identity.session_id,
            source_family=identity.source_family,
            group_id=identity.group_id,
            error_position_id=identity.error_position_id,
            sample_rate=int(self.adapter.binding.training_timing_contract.sample_rate),
            disturbance_off=disturbance[0, 0, prefix:].detach().cpu().numpy(),
            error_deep_anc=causal.error_target[0, 0].detach().cpu().numpy(),
            error_fxlms=fxlms_error[prefix:],
            causal_plant_binding_sha256=self.adapter.binding_sha256,
        )
        return causal, segment


__all__ = [
    "FULL_OCTAVE_V3_MATCHED_FXLMS_SCHEMA",
    "FULL_OCTAVE_V3_SURROGATE_ROLE",
    "FULL_OCTAVE_V3_TRAINER_CONSUMER_SCHEMA",
    "FullOctaveV3EvaluationIdentity",
    "FullOctaveV3FxLMSConfig",
    "FullOctaveV3MatchedFxLMSEvaluator",
    "FullOctaveV3TrainStepResult",
    "FullOctaveV3TrainerConsumer",
]
