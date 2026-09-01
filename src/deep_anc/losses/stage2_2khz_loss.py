"""125 Hz--2 kHz Stage-2 전용 손실과 tensor admission 경계.

기존 Stage-1 연속대역 손실이나 full-octave v3 손실의 숫자만 바꾸지 않는다. 이
primitive는 다섯 octave를 각각 target 에너지로 정규화해 정확히 같은 baseline
가중치를 주고, 최악 octave guard와 1.6 kHz 6 dB hard guard를 별도로 더한다.

입력은 causal adapter가 만든 valid crop의 ``y``, ``d=P*n``, ``S*y``다. 측정 FIR의
극성이 이미 ``S*y``에 있으므로 error는 오직 ``e=d+S*y``로 만든다. 4/8 kHz는
positive attenuation 목표가 아니라 actuator-output do-no-harm 항이다.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from torch import nn

from ..dsp.stage2_2khz_contract import (
    STAGE2_2KHZ_DNH_BANDS_HZ,
    STAGE2_2KHZ_OBJECTIVE_BANDS_HZ,
    STAGE2_2KHZ_SOURCE_FAMILIES,
    Stage2TwoKilohertzContract,
)
from .config import DEFAULT_DNH_MARGIN_DB


STAGE2_2KHZ_LOSS_SCHEMA = "stage2_2khz_dedicated_loss_v1"
STAGE2_2KHZ_DNH_SCHEMA = "stage2_2khz_actuator_output_4k_8k_margin_hinge_v1"
STAGE2_2KHZ_BATCH_TENSOR_ADMISSION_SCHEMA = (
    "stage2_2khz_actual_family_density_batch_v1"
)
STAGE2_2KHZ_NMSE_FLOOR_DB = -80.0
STAGE2_2KHZ_ONE_POINT_SIX_SENTINEL_BAND_HZ = (
    1425.437949,
    1795.939277,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if not _SHA256.fullmatch(text):
        raise ValueError(f"{label}는 lowercase SHA-256이어야 합니다")
    return text


class Stage2TwoKilohertzLossConfig(BaseModel):
    """DNH gradient calibration과 batch authority를 결속한 immutable 설정."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["stage2_2khz_dedicated_loss_v1"] = (
        STAGE2_2KHZ_LOSS_SCHEMA
    )
    control_band_contract: Stage2TwoKilohertzContract
    control_band_contract_sha256: str
    minimum_target_d_density_ratio: Literal[0.25] = 0.25
    minimum_valid_items_per_octave: Literal[4] = 4
    item_cvar_q: Literal[0.25] = 0.25
    item_cvar_alpha: Literal[0.7] = 0.7
    octave_worst_guard_weight: Literal[0.7] = 0.7
    low_mid_positive_guard_weight: Literal[1.0] = 1.0
    one_point_six_khz_positive_guard_weight: Literal[1.0] = 1.0
    one_point_six_khz_minimum_attenuation_db: Literal[6.0] = 6.0
    one_point_six_khz_training_margin_db: Literal[0.1] = 0.1
    # 2 kHz의 과거 3 dB 하드 게이트는 제거했다. 이 weight는 2 kHz 증폭을
    # 막는 보조 hinge에만 적용되며, 실제 하한은 0 dB이다.
    two_khz_positive_guard_weight: Literal[1.0] = 1.0
    two_khz_minimum_attenuation_db: Literal[0.0] = 0.0
    relative_nmse_floor_db: Literal[-80.0] = STAGE2_2KHZ_NMSE_FLOOR_DB
    lambda_dnh: float
    dnh_margin_db: float = DEFAULT_DNH_MARGIN_DB
    dnh_band_floor_db: Literal[-60.0] = -60.0
    dnh_schema_version: Literal[
        "stage2_2khz_actuator_output_4k_8k_margin_hinge_v1"
    ] = STAGE2_2KHZ_DNH_SCHEMA
    dnh_calibration_receipt_sha256: str
    dnh_observed_gradient_share: float
    family_balanced_sampler_receipt_sha256: str
    actual_batch_family_and_density_rechecked: Literal[True] = True
    objective_semantics: Literal[
        "five_exact_one_fifth_baseline_plus_additive_worst_and_gate_guards"
    ] = "five_exact_one_fifth_baseline_plus_additive_worst_and_gate_guards"
    two_khz_positive_is_secondary_diagnostic: Literal[True] = True
    generic_stage1_loss_allowed: Literal[False] = False
    full_octave_v3_loss_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _validate_stage2(self) -> "Stage2TwoKilohertzLossConfig":
        canonical = Stage2TwoKilohertzContract.canonical()
        if self.control_band_contract != canonical:
            raise ValueError("Stage-2 loss는 exact canonical 2 kHz 계약이 필요합니다")
        supplied = _require_sha256(
            self.control_band_contract_sha256,
            label="Stage-2 control-band contract SHA",
        )
        if supplied != canonical.digest():
            raise ValueError("Stage-2 inline contract payload와 SHA가 다릅니다")
        _require_sha256(
            self.dnh_calibration_receipt_sha256,
            label="Stage-2 DNH calibration receipt SHA",
        )
        _require_sha256(
            self.family_balanced_sampler_receipt_sha256,
            label="Stage-2 sampler receipt SHA",
        )
        share = float(self.dnh_observed_gradient_share)
        if not math.isfinite(share) or not 0.2 <= share <= 0.4:
            raise ValueError("Stage-2 output-y DNH gradient share는 [0.2, 0.4]여야 합니다")
        if not math.isfinite(float(self.lambda_dnh)) or float(self.lambda_dnh) <= 0.0:
            raise ValueError("Stage-2 4/8 kHz DNH를 끄거나 non-finite로 둘 수 없습니다")
        if (
            not math.isfinite(float(self.dnh_margin_db))
            or float(self.dnh_margin_db) > float(DEFAULT_DNH_MARGIN_DB) + 1.0e-12
        ):
            raise ValueError("Stage-2 DNH margin을 G4-consistent 한계보다 완화할 수 없습니다")
        return self

    @classmethod
    def parse(
        cls, raw: Mapping[str, Any] | "Stage2TwoKilohertzLossConfig"
    ) -> "Stage2TwoKilohertzLossConfig":
        return raw if isinstance(raw, cls) else cls.model_validate(dict(raw))


class Stage2TwoKilohertzLoss(nn.Module):
    """다섯 octave, 1.6 kHz 6 dB guard, 2 kHz positive guard, 4/8 kHz DNH 소비자."""

    def __init__(
        self,
        loss_cfg: Mapping[str, Any] | Stage2TwoKilohertzLossConfig,
        *,
        sample_rate: int = 48_000,
    ) -> None:
        super().__init__()
        cfg = Stage2TwoKilohertzLossConfig.parse(loss_cfg)
        contract = cfg.control_band_contract
        if int(sample_rate) != int(contract.sample_rate):
            raise ValueError("Stage-2 loss sample rate와 control-band 계약이 다릅니다")
        self.loss_config = cfg
        self.control_band_contract = contract
        self.control_band_contract_sha256 = contract.digest()
        self.sample_rate = int(sample_rate)
        self.objective_bands_hz = tuple(
            tuple(float(value) for value in band)
            for band in contract.octave_objective_bands_hz
        )
        self.dnh_bands_hz = tuple(
            tuple(float(value) for value in band)
            for band in contract.do_no_harm_octave_bands_hz
        )
        if self.objective_bands_hz != STAGE2_2KHZ_OBJECTIVE_BANDS_HZ:
            raise ValueError("Stage-2 objective octave bytes가 canonical constant와 다릅니다")
        if self.dnh_bands_hz != STAGE2_2KHZ_DNH_BANDS_HZ:
            raise ValueError("Stage-2 4/8 kHz DNH bytes가 canonical constant와 다릅니다")
        self.minimum_density_ratio = float(cfg.minimum_target_d_density_ratio)
        self.minimum_valid_items = int(cfg.minimum_valid_items_per_octave)
        self.item_cvar_q = float(cfg.item_cvar_q)
        self.item_cvar_alpha = float(cfg.item_cvar_alpha)
        self.octave_worst_guard_weight = float(cfg.octave_worst_guard_weight)
        self.relative_nmse_floor_ratio = 10.0 ** (
            float(cfg.relative_nmse_floor_db) / 10.0
        )
        self.lambda_dnh = float(cfg.lambda_dnh)
        self.dnh_margin_db = float(cfg.dnh_margin_db)
        self.dnh_band_floor = 10.0 ** (float(cfg.dnh_band_floor_db) / 10.0)

    @staticmethod
    def _as_batch_time(value: torch.Tensor, *, label: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label}는 torch.Tensor여야 합니다")
        if value.ndim == 3 and int(value.shape[1]) == 1:
            value = value[:, 0, :]
        if value.ndim != 2 or int(value.shape[0]) < 1 or int(value.shape[-1]) < 2:
            raise ValueError(f"{label}는 [batch,time] 또는 [batch,1,time]이어야 합니다")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{label}는 finite floating tensor여야 합니다")
        # FFT/loss는 bf16/autocast에서도 FP32로 고정한다.
        return value.float()

    def _band_bins(
        self,
        samples: int,
        band_hz: Sequence[float],
        *,
        include_upper: bool,
    ) -> tuple[int, int]:
        lo, hi = (float(value) for value in band_hz)
        maximum = int(samples) // 2
        lo_bin = max(0, int(math.ceil(lo * int(samples) / self.sample_rate)))
        hi_bin = min(maximum, int(math.floor(hi * int(samples) / self.sample_rate)))
        if not include_upper and hi < self.sample_rate / 2.0:
            hi_bin = min(
                hi_bin,
                int(math.ceil(hi * int(samples) / self.sample_rate)) - 1,
            )
        if lo_bin > hi_bin:
            raise ValueError(f"{samples}샘플 FFT에 [{lo:g},{hi:g})Hz bin이 없습니다")
        return lo_bin, hi_bin

    def _density_ratios(
        self, target_power: torch.Tensor, *, samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        densities: list[torch.Tensor] = []
        union_power = target_power.new_zeros((target_power.shape[0],))
        union_bins = 0
        for index, band in enumerate(self.objective_bands_hz):
            lo, hi = self._band_bins(
                samples,
                band,
                include_upper=index == len(self.objective_bands_hz) - 1,
            )
            selected = target_power[..., lo : hi + 1]
            densities.append(selected.mean(dim=-1))
            union_power = union_power + selected.sum(dim=-1)
            union_bins += int(selected.shape[-1])
        baseline = union_power / float(union_bins)
        tiny = torch.finfo(baseline.dtype).tiny
        ratios = torch.stack(
            tuple(
                torch.where(
                    baseline > tiny,
                    density / baseline.clamp_min(tiny),
                    torch.zeros_like(baseline),
                )
                for density in densities
            ),
            dim=1,
        )
        return ratios, ratios.detach() >= self.minimum_density_ratio

    def _single_band_density_ratio(
        self,
        target_power: torch.Tensor,
        *,
        samples: int,
        band_hz: Sequence[float],
    ) -> torch.Tensor:
        """objective union의 bin-평균 대비 auxiliary band target density."""

        union_sum = target_power.new_zeros((target_power.shape[0],))
        union_bins = 0
        for index, objective_band in enumerate(self.objective_bands_hz):
            lo, hi = self._band_bins(
                samples,
                objective_band,
                include_upper=index == len(self.objective_bands_hz) - 1,
            )
            selected = target_power[..., lo : hi + 1]
            union_sum = union_sum + selected.sum(dim=-1)
            union_bins += int(selected.shape[-1])
        band_lo, band_hi = self._band_bins(samples, band_hz, include_upper=False)
        band_density = target_power[..., band_lo : band_hi + 1].mean(dim=-1)
        baseline = union_sum / float(union_bins)
        tiny = torch.finfo(baseline.dtype).tiny
        return torch.where(
            baseline > tiny,
            band_density / baseline.clamp_min(tiny),
            torch.zeros_like(baseline),
        )

    def _band_nmse_db(
        self,
        error_power: torch.Tensor,
        target_power: torch.Tensor,
        *,
        samples: int,
        band_hz: Sequence[float],
        include_upper: bool,
    ) -> torch.Tensor:
        lo, hi = self._band_bins(samples, band_hz, include_upper=include_upper)
        e_selected = error_power[..., lo : hi + 1]
        d_selected = target_power[..., lo : hi + 1]
        weights = torch.full(
            (hi - lo + 1,), 2.0, dtype=e_selected.dtype, device=e_selected.device
        )
        if lo == 0:
            weights[0] = 1.0
        if samples % 2 == 0 and hi == samples // 2:
            weights[-1] = 1.0
        e_sum = (e_selected * weights).sum(dim=-1)
        d_sum = (d_selected * weights).sum(dim=-1)
        tiny = torch.finfo(e_sum.dtype).tiny
        safe_d = d_sum.clamp_min(tiny)
        relative_floor = (safe_d.detach() * self.relative_nmse_floor_ratio).clamp_min(
            tiny
        )
        safe_e = torch.maximum(e_sum, relative_floor)
        return (10.0 / math.log(10.0)) * (torch.log(safe_e) - torch.log(safe_d))

    def _item_objective(self, values: torch.Tensor) -> torch.Tensor:
        flat = values.reshape(-1)
        if int(flat.numel()) < self.minimum_valid_items:
            raise ValueError("Stage-2 item objective에 valid item이 4개 미만입니다")
        k = max(
            self.minimum_valid_items,
            int(math.ceil(self.item_cvar_q * int(flat.numel()))),
        )
        k = min(k, int(flat.numel()))
        cvar = flat.topk(k).values.mean()
        return (1.0 - self.item_cvar_alpha) * flat.mean() + self.item_cvar_alpha * cvar

    @staticmethod
    def _family_indices(
        source_families: Sequence[str], *, batch_size: int
    ) -> dict[str, tuple[int, ...]]:
        parsed = tuple(str(value) for value in source_families)
        if len(parsed) != int(batch_size):
            raise ValueError("Stage-2 source family 개수가 실제 batch와 다릅니다")
        expected = tuple(STAGE2_2KHZ_SOURCE_FAMILIES)
        counts = Counter(parsed)
        if set(counts) != set(expected) or len(set(counts.values())) != 1:
            raise ValueError(
                "Stage-2 실제 batch의 speech/music/environment/machine 개수가 균등하지 않습니다"
            )
        return {
            family: tuple(index for index, value in enumerate(parsed) if value == family)
            for family in expected
        }

    def _actuator_dnh(
        self, actuator_output: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        samples = int(actuator_output.shape[-1])
        spectrum = torch.fft.rfft(actuator_output, dim=-1, norm="ortho")
        power = spectrum.real.square() + spectrum.imag.square()
        protected_lo = self.objective_bands_hz[0][0]
        protected_hi = self.objective_bands_hz[-1][1]
        lo, hi = self._band_bins(
            samples, (protected_lo, protected_hi), include_upper=True
        )
        protected_power = power[..., lo : hi + 1].sum(dim=-1)
        total_power = power.sum(dim=-1)
        denominator = torch.maximum(
            protected_power.detach(), total_power.detach() * self.dnh_band_floor
        ).clamp_min(torch.finfo(power.dtype).tiny)
        loss = actuator_output.new_zeros(())
        metrics: dict[str, float] = {
            "stage2_dnh_control_gradient_detached": 1.0,
        }
        worst = -math.inf
        for index, band in enumerate(self.dnh_bands_hz):
            band_lo, band_hi = self._band_bins(
                samples,
                band,
                include_upper=index == len(self.dnh_bands_hz) - 1,
            )
            numerator = power[..., band_lo : band_hi + 1].sum(dim=-1)
            ratio_db = 10.0 * torch.log10(numerator / denominator + 1.0e-30)
            hinge = torch.relu(ratio_db - self.dnh_margin_db)
            loss = loss + 0.5 * self._item_objective(hinge)
            observed = float(ratio_db.max().detach())
            metrics[f"stage2_dnh_octave_{index}_max_db"] = observed
            worst = max(worst, observed)
        metrics["stage2_dnh_worst_db"] = worst
        return loss, metrics

    def forward(
        self,
        actuator_output: torch.Tensor,
        target_d: torch.Tensor,
        secondary_output: torch.Tensor,
        *,
        source_families: Sequence[str],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """valid crop만 사용해 Stage-2 전용 loss를 계산한다."""

        y = self._as_batch_time(actuator_output, label="actuator_output")
        d = self._as_batch_time(target_d, label="target_d")
        s_y = self._as_batch_time(secondary_output, label="secondary_output")
        if y.shape != d.shape or s_y.shape != d.shape:
            raise ValueError("Stage-2 y/d/S*y shape가 서로 다릅니다")
        family_indices = self._family_indices(
            source_families, batch_size=int(d.shape[0])
        )

        # 측정 FIR 극성을 이 식 이외에서 반전하지 않는다.
        e = d + s_y
        samples = int(d.shape[-1])
        D = torch.fft.rfft(d, dim=-1, norm="ortho")
        E = torch.fft.rfft(e, dim=-1, norm="ortho")
        d_power = D.real.square() + D.imag.square()
        e_power = E.real.square() + E.imag.square()
        density, valid = self._density_ratios(d_power, samples=samples)

        objectives: list[torch.Tensor] = []
        octave_item_worst: list[torch.Tensor] = []
        octave_family_worst: list[torch.Tensor] = []
        octave_family_cells: list[torch.Tensor] = []
        metrics: dict[str, float] = {}
        for index, band in enumerate(self.objective_bands_hz):
            mask = valid[:, index]
            valid_count = int(mask.sum().detach())
            if valid_count < self.minimum_valid_items:
                raise ValueError(
                    "Stage-2 target-density 자격 부족: "
                    f"octave={index}, valid_items={valid_count}, "
                    f"required={self.minimum_valid_items}, threshold={self.minimum_density_ratio:.2f}"
                )
            # 네 family 중 하나가 모두 density-invalid인 batch를 family-balanced로
            # 위장할 수 없다. 각 octave에 네 family가 실제로 하나 이상 남아야 한다.
            family_objectives: list[torch.Tensor] = []
            minimum_family_items = max(
                1, int(math.ceil(self.minimum_valid_items / len(family_indices)))
            )
            for family, indices in family_indices.items():
                family_mask = torch.zeros_like(mask)
                family_mask[list(indices)] = True
                family_selected_mask = mask & family_mask
                if int(family_selected_mask.sum().detach()) < minimum_family_items:
                    raise ValueError(
                        f"Stage-2 {family} family가 objective octave {index}에서 "
                        "target-density valid item 하한을 제공하지 않았습니다"
                    )
            raw = self._band_nmse_db(
                e_power,
                d_power,
                samples=samples,
                band_hz=band,
                include_upper=index == len(self.objective_bands_hz) - 1,
            )
            selected = raw[mask]
            for indices in family_indices.values():
                family_mask = torch.zeros_like(mask)
                family_mask[list(indices)] = True
                family_objectives.append(raw[mask & family_mask].max())
            family_stack = torch.stack(tuple(family_objectives))
            aggregate = family_stack.mean()
            objectives.append(aggregate)
            octave_family_worst.append(family_stack.max())
            octave_family_cells.append(family_stack)
            octave_item_worst.append(selected.max())
            metrics[f"stage2_octave_{index}_valid_items"] = float(valid_count)
            metrics[f"stage2_octave_{index}_density_min_ratio"] = float(
                density[:, index][mask].min().detach()
            )
            metrics[f"stage2_octave_{index}_objective_nmse_db"] = float(
                aggregate.detach()
            )
            metrics[f"stage2_octave_{index}_worst_nmse_db"] = float(
                selected.max().detach()
            )
            metrics[f"stage2_octave_{index}_family_worst_nmse_db"] = float(
                family_stack.max().detach()
            )

        stacked = torch.stack(tuple(objectives))
        equal = stacked.mean()  # 각 octave baseline 정확히 1/5
        worst = torch.stack(tuple(octave_item_worst)).max()
        low_mid_guard = torch.relu(torch.stack(tuple(octave_family_worst[:4]))).sum()
        # 2 kHz octave 평균이 좋더라도 1.6 kHz 전이부만 near-zero인 결과를 허용하지
        # 않는다. contract bytes를 바꾸지 않는 auxiliary one-third-octave sentinel이며,
        # objective와 동일하게 actual P*n density/family를 다시 검사한다.
        sentinel_density = self._single_band_density_ratio(
            d_power,
            samples=samples,
            band_hz=STAGE2_2KHZ_ONE_POINT_SIX_SENTINEL_BAND_HZ,
        )
        sentinel_valid = sentinel_density.detach() >= self.minimum_density_ratio
        sentinel_count = int(sentinel_valid.sum().detach())
        if sentinel_count < self.minimum_valid_items:
            raise ValueError(
                "Stage-2 1.6 kHz sentinel target-density valid item이 4개 미만입니다"
            )
        sentinel_family_objectives: list[torch.Tensor] = []
        for family, indices in family_indices.items():
            family_mask = torch.zeros_like(sentinel_valid)
            family_mask[list(indices)] = True
            family_valid = sentinel_valid & family_mask
            if int(family_valid.sum().detach()) < minimum_family_items:
                raise ValueError(
                    f"Stage-2 {family} family가 1.6 kHz sentinel density-valid item을 "
                    "하한만큼 제공하지 않았습니다"
                )
        sentinel_raw = self._band_nmse_db(
            e_power,
            d_power,
            samples=samples,
            band_hz=STAGE2_2KHZ_ONE_POINT_SIX_SENTINEL_BAND_HZ,
            include_upper=False,
        )
        for indices in family_indices.values():
            family_mask = torch.zeros_like(sentinel_valid)
            family_mask[list(indices)] = True
            sentinel_family_objectives.append(
                sentinel_raw[sentinel_valid & family_mask].max()
            )
        sentinel_family_stack = torch.stack(tuple(sentinel_family_objectives))
        sentinel_objective = sentinel_family_stack.mean()
        # 평가 comparator는 sentinel attenuation>=6 dB다. 학습 hinge에는
        # 수치적으로 경계에 남지 않도록 0.1 dB의 내부 여유를 둔다.
        sentinel_guard = torch.relu(
            sentinel_family_stack.max()
            + float(self.loss_config.one_point_six_khz_minimum_attenuation_db)
            + float(self.loss_config.one_point_six_khz_training_margin_db)
        )
        worst_with_sentinel = torch.maximum(worst, sentinel_raw[sentinel_valid].max())
        # 2 kHz의 과거 3 dB hard gate는 제거하고, 양의 감쇠(증폭 방지)만
        # 보조 hinge로 유지한다.
        two_khz_guard = torch.relu(
            octave_family_worst[4]
            + float(self.loss_config.two_khz_minimum_attenuation_db)
        )
        family_cells = torch.stack(tuple(octave_family_cells))
        thresholds = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=family_cells.dtype,
            device=family_cells.device,
        ).unsqueeze(1)
        cell_violations = torch.relu(family_cells - thresholds)
        sentinel_threshold = -float(
            self.loss_config.one_point_six_khz_minimum_attenuation_db
        ) - float(self.loss_config.one_point_six_khz_training_margin_db)
        sentinel_violations = torch.relu(
            sentinel_family_stack - sentinel_threshold
        )
        # PASS cell의 연속 개선은 -20 dB에서 bounded한다. 어떤 family×band failure도
        # 다른 cell의 큰 음수 reward로 상쇄되지 않도록 실패 cell마다 불연속 barrier를
        # 더하고, hinge는 threshold 방향 gradient를 제공한다.
        bounded_equal = torch.maximum(
            family_cells,
            torch.full_like(family_cells, -20.0),
        ).mean()
        threshold_failure_count = (cell_violations.detach() > 0.0).sum() + (
            sentinel_violations.detach() > 0.0
        ).sum()
        threshold_failure_barrier = 25.0 * threshold_failure_count.to(
            family_cells.dtype
        )
        dnh, dnh_metrics = self._actuator_dnh(y)
        objective = (
            bounded_equal
            + threshold_failure_barrier
            + 5.0 * (cell_violations.sum() + sentinel_violations.sum())
            + self.octave_worst_guard_weight
            * torch.clamp(worst_with_sentinel, min=-20.0, max=0.0)
            + float(self.loss_config.low_mid_positive_guard_weight) * low_mid_guard
            + float(self.loss_config.one_point_six_khz_positive_guard_weight)
            * sentinel_guard
            + float(self.loss_config.two_khz_positive_guard_weight) * two_khz_guard
            + self.lambda_dnh * dnh
        )
        if not torch.isfinite(objective):
            raise RuntimeError("Stage-2 dedicated loss가 finite하지 않습니다")
        metrics.update(dnh_metrics)
        metrics.update(
            {
                "loss": float(objective.detach()),
                "stage2_octave_equal_nmse_db": float(equal.detach()),
                "stage2_family_cell_bounded_equal_nmse_db": float(
                    bounded_equal.detach()
                ),
                "stage2_family_cell_threshold_failure_count": float(
                    threshold_failure_count.detach()
                ),
                "stage2_family_cell_threshold_failure_barrier": float(
                    threshold_failure_barrier.detach()
                ),
                "stage2_octave_worst_nmse_db": float(worst.detach()),
                "stage2_frequency_worst_with_1p6_sentinel_nmse_db": float(
                    worst_with_sentinel.detach()
                ),
                "stage2_low_mid_positive_guard": float(low_mid_guard.detach()),
                "stage2_one_point_six_khz_sentinel_nmse_db": float(
                    sentinel_objective.detach()
                ),
                "stage2_one_point_six_khz_sentinel_positive_guard": float(
                    sentinel_guard.detach()
                ),
                "stage2_one_point_six_khz_training_margin_db": float(
                    self.loss_config.one_point_six_khz_training_margin_db
                ),
                "stage2_one_point_six_khz_minimum_attenuation_db": float(
                    self.loss_config.one_point_six_khz_minimum_attenuation_db
                ),
                "stage2_one_point_six_khz_sentinel_valid_items": float(
                    sentinel_count
                ),
                "stage2_one_point_six_khz_sentinel_density_min_ratio": float(
                    sentinel_density[sentinel_valid].min().detach()
                ),
                "stage2_two_khz_positive_guard": float(two_khz_guard.detach()),
                "stage2_two_khz_objective_nmse_db": float(stacked[4].detach()),
                "stage2_dnh": float(dnh.detach()),
                "stage2_dnh_lambda": self.lambda_dnh,
                "stage2_dnh_observed_gradient_share": float(
                    self.loss_config.dnh_observed_gradient_share
                ),
                "stage2_exact_equal_octave_weight": 0.2,
                "stage2_actual_family_balance_rechecked": 1.0,
                "stage2_actual_target_density_rechecked": 1.0,
                "stage2_generic_stage1_loss_used": 0.0,
                "stage2_full_octave_v3_loss_used": 0.0,
                "stage2_two_khz_positive_is_secondary_diagnostic": 1.0,
            }
        )
        return objective, metrics


__all__ = [
    "STAGE2_2KHZ_BATCH_TENSOR_ADMISSION_SCHEMA",
    "STAGE2_2KHZ_DNH_SCHEMA",
    "STAGE2_2KHZ_LOSS_SCHEMA",
    "STAGE2_2KHZ_ONE_POINT_SIX_SENTINEL_BAND_HZ",
    "Stage2TwoKilohertzLoss",
    "Stage2TwoKilohertzLossConfig",
]
