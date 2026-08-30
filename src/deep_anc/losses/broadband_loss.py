"""최종 광대역 point-control 역할의 손실 계약.

Stage-1 :class:`~deep_anc.losses.anc_loss.ANCLoss`는 하나의 연속 trusted band를
정규화한다. 그 수식을 11.314 kHz로 단순히 늘리면 폭이 넓고 에너지가 큰 대역이
목적함수를 지배해 저역 성공이 고역 실패를 가릴 수 있다. 이 모듈은
``ControlBandContract.broadband_point_control()``의 일곱 subband를 각각 자기 target
에너지로 정규화한 뒤 **동일 가중**하고, 별도의 worst-subband CVaR guard를 섞는다.

기존 Stage-1 설정과 constructor는 바꾸지 않는다. 광대역 P/S와 recorded-v2 receipt가
통과하기 전에는 이 클래스를 실제 학습에 연결해서는 안 된다.
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from pydantic import BaseModel, ConfigDict, model_validator

from ..data.broadband_population_contract_v3 import MIN_DENSITY_RATIO
from ..dsp.control_band_contract import (
    BROADBAND_POINT_CONTROL_CONTRACT_ID,
    BROADBAND_V3_OCTAVE_OBJECTIVE_BANDS_HZ,
    BROADBAND_V3_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ,
    BROADBAND_V3_STAGE1_LOW_GUARD_SUBBANDS_HZ,
    BroadbandFullOctaveContractV3,
    ControlBandContract,
    resolve_control_band_contract,
)
from ..dsp.nonlinear import RandomNonlinear
from ..dsp.measured_band_path import (
    MEASURED_BAND_INTERPOLATION_SCHEMA,
    MEASURED_BAND_PATH_SCHEMA_VERSION,
    MeasuredBandPath,
)
from ..dsp.filters import fft_filter
from ..dsp.secondary_path import (
    DifferentiableSecondaryPath,
    fft_causal_filter,
    integer_delay,
)
from .anc_loss import ANCLoss
from ..dsp.timing import FrequencyBand
from .config import DEFAULT_DNH_MARGIN_DB, DoNoHarmPlan, LossConfig


BROADBAND_LOSS_SCHEMA_VERSION = "broadband_equal_subband_loss_v3"
BROADBAND_DNH_SCHEMA_VERSION = "actuator_output_union_leakage_v1"
BROADBAND_FULL_OCTAVE_DNH_SCHEMA_VERSION = (
    "actuator_output_union_margin_hinge_v3"
)
BROADBAND_FULL_OCTAVE_NMSE_FLOOR_DB = -80.0
BROADBAND_DNH_DOMAIN = "actuator_output_y_nl"
BROADBAND_LINEAR_SPECTRAL_SCHEMA = "finite_sequence_linear_dtft_no_ifft_v1"
BROADBAND_CAUSAL_PATH_SCHEMA = "fullband_causal_joint_fir_operator_npz_v4"
BROADBAND_CAUSAL_CONVOLUTION_SCHEMA = (
    "full_linear_causal_convolution_continuous_prefix_valid_crop_v1"
)
BROADBAND_CAUSAL_INTERPOLATION_SCHEMA = "not_applicable_frozen_causal_fir_v1"
BROADBAND_DNH_CALIBRATION_PLACEHOLDER = (
    "requires_output_y_gradient_share_0p2_0p4"
)

# v2 ``BroadbandLossConfig``와 schema/id를 공유하지 않는다. 이 primitive는 live v5
# causal authority가 생긴 뒤 별도 criterion wiring을 할 때만 소비할 수 있다.
BROADBAND_FULL_OCTAVE_LOSS_SCHEMA_VERSION = (
    "broadband_full_octave_loss_primitive_v3"
)
BROADBAND_FULL_OCTAVE_TRAINING_BLOCKER = (
    "BLOCKED_INCOMPLETE_BROADBAND_V3_ADMISSION"
)
BROADBAND_FULL_OCTAVE_ADMISSION_BLOCKERS = (
    "MISSING_LIVE_V5_CAUSAL_AUTHORITY_ENVELOPE",
    "MISSING_OUTPUT_Y_GRADIENT_SHARE_0P2_0P4_CALIBRATION",
    "MISSING_ACTUAL_FAMILY_BALANCED_BATCH_RECEIPT_BINDING",
    "MISSING_CAUSAL_PREFIX_OPERATOR_TIMING_BINDING",
)


def _lower_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label}는 lowercase SHA-256이어야 합니다")
    return text


@dataclass(frozen=True)
class CausalFIRPathData:
    """연속 캡처에서 동결한 finite causal P/S 연산자.

    ``coarse_delay_samples``는 FIR 앞의 정수 0이고, 분수 지연은
    ``post_onset_fir``의 위상에 이미 들어 있다. 따라서 forward에서 분수 지연을
    한 번 더 적용하는 필드는 의도적으로 없다. ``handoff_extra_samples``는 S 역할에만
    허용하며 256-sample 런타임 handoff를 정확히 한 번 더한다.
    """

    role: Literal["primary", "secondary"]
    post_onset_fir: np.ndarray
    coarse_delay_samples: int
    fractional_delay_samples: float
    support_samples: int
    sample_rate: int
    handoff_extra_samples: int
    operator_file_sha256: str
    operator_internal_sha256: str
    fir_sha256: str
    authority_sha256: str
    source_path: str
    fractional_delay_encoded_in_post_onset_fir: bool = True

    def __post_init__(self) -> None:
        fir = np.asarray(self.post_onset_fir)
        if fir.dtype != np.dtype("<f8") and fir.dtype != np.dtype("=f8"):
            raise ValueError("causal P/S post-onset FIR은 float64여야 합니다")
        fir = np.ascontiguousarray(fir, dtype=np.float64).reshape(-1)
        if (
            fir.size < 1
            or fir.size != int(self.support_samples)
            or not np.all(np.isfinite(fir))
            or float(np.max(np.abs(fir))) <= 0.0
        ):
            raise ValueError("causal P/S FIR support/finite/nonzero 계약 위반")
        if int(self.coarse_delay_samples) < 0:
            raise ValueError("causal P/S coarse delay는 0 이상이어야 합니다")
        if int(self.sample_rate) <= 0:
            raise ValueError("causal P/S sample rate는 양수여야 합니다")
        fractional = float(self.fractional_delay_samples)
        if not math.isfinite(fractional) or not -0.5 <= fractional < 0.5:
            raise ValueError("causal P/S fractional delay는 [-0.5, 0.5)여야 합니다")
        if not bool(self.fractional_delay_encoded_in_post_onset_fir):
            raise ValueError("분수 지연이 post-onset FIR 위상에 포함되지 않았습니다")
        handoff = int(self.handoff_extra_samples)
        if handoff < 0 or (self.role == "primary" and handoff != 0):
            raise ValueError("handoff는 secondary causal 연산자에만 허용합니다")
        actual_fir_sha = hashlib.sha256(fir.tobytes(order="C")).hexdigest()
        if actual_fir_sha != _lower_sha256(self.fir_sha256, label="causal FIR SHA"):
            raise ValueError("causal FIR bytes가 authority SHA와 다릅니다")
        for label, value in (
            ("operator file SHA", self.operator_file_sha256),
            ("operator internal SHA", self.operator_internal_sha256),
            ("authority SHA", self.authority_sha256),
        ):
            _lower_sha256(value, label=label)
        object.__setattr__(self, "post_onset_fir", fir)

    @property
    def base_delay_samples(self) -> int:
        return int(self.coarse_delay_samples) + int(self.handoff_extra_samples)

    @property
    def history_samples(self) -> int:
        # PlantSettle의 보수적 규약과 동일하게 delay+FIR taps 전체를 버린다.
        return self.base_delay_samples + int(self.support_samples)


class CausalFIRPath(nn.Module):
    """동결된 causal FIR을 torch/numpy에서 같은 순서로 적용한다."""

    def __init__(self, data: CausalFIRPathData) -> None:
        super().__init__()
        self.data = data
        self.register_buffer(
            "fir", torch.from_numpy(data.post_onset_fir.astype(np.float32, copy=True))
        )
        self.base_delay = int(data.base_delay_samples)
        self.sample_rate = int(data.sample_rate)
        self.history_samples = int(data.history_samples)

    @staticmethod
    def _validate_nominal_perturbation(perturb: dict | None) -> None:
        values = dict(perturb or {})
        if int(values.get("jitter", 0)) != 0:
            raise ValueError("canonical causal P/S는 delay jitter를 허용하지 않습니다")
        if float(values.get("gain_db", 0.0)) != 0.0:
            raise ValueError("canonical causal P/S는 gain perturb를 허용하지 않습니다")
        if float(values.get("tilt_db_per_octave", 0.0)) != 0.0:
            raise ValueError("canonical causal P/S는 tilt perturb를 허용하지 않습니다")
        if bool(values.get("allpass", False)):
            raise ValueError("canonical causal P/S는 allpass perturb를 허용하지 않습니다")

    def sample_perturbation(self) -> dict[str, object]:
        return {
            "jitter": 0,
            "gain_db": 0.0,
            "tilt_db_per_octave": 0.0,
            "allpass": False,
        }

    def forward(
        self, value: torch.Tensor, perturb: dict | None = None
    ) -> torch.Tensor:
        self._validate_nominal_perturbation(perturb)
        delayed = integer_delay(value, self.base_delay)
        return fft_causal_filter(delayed, self.fir.to(value.device))

    def filter_numpy(self, value: np.ndarray) -> np.ndarray:
        samples = np.asarray(value, dtype=np.float32).reshape(-1)
        filtered = fft_filter(samples, self.data.post_onset_fir)
        if self.base_delay <= 0:
            return np.asarray(filtered, dtype=np.float32)
        output = np.zeros_like(filtered, dtype=np.float32)
        if self.base_delay < output.size:
            output[self.base_delay :] = filtered[: -self.base_delay]
        return output


class BroadbandLossConfig(LossConfig):
    """광대역 역할 전용 fail-closed 설정.

    ``nmse_cvar_*``는 각 subband 안의 item 최악값을, ``subband_guard_*``는 일곱
    subband 사이의 최악값을 다룬다. 두 축을 분리해야 쉬운 source나 저역이 어려운
    source/고역을 평균으로 숨길 수 없다.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["broadband_equal_subband_loss_v3"] = (
        BROADBAND_LOSS_SCHEMA_VERSION
    )
    control_band_contract_id: Literal[
        "broadband_point_control_150_11314_v2"
    ] = BROADBAND_POINT_CONTROL_CONTRACT_ID

    # 광대역 역할은 연속 fullband 또는 Stage-1 curriculum weight로 되돌릴 수 없다.
    nmse_objective: Literal["trusted_band"] = "trusted_band"
    band_weight: Literal["trusted_only"] = "trusted_only"
    nmse_cvar_min_k: Literal[4] = 4

    # 대역 CVaR. alpha<1을 강제해 equal-subband 항의 그래디언트가 일곱 대역 모두에
    # 남고, alpha>0을 강제해 평균-only 우회를 닫는다.
    subband_guard_q: float = 0.25
    subband_guard_alpha: float = 0.7
    subband_guard_min_k: int = 2

    # recorded-v2 receipt와 같은 hard floor다. Literal로 고정해 학습이 막힌 뒤
    # config 숫자만 낮추는 우회를 허용하지 않는다. 각 subband에서 이 밀도를
    # 통과한 item이 item-CVaR의 최소 표본 수보다 적으면 해당 batch는 학습하지 않는다.
    minimum_target_d_density_ratio: Literal[0.25] = MIN_DENSITY_RATIO

    # 진단용 measured-band DTFT와 canonical continuous-causal FIR을 명시적으로
    # 분리한다. 새 causal 역할을 tone-only measured key로 오표기하지 않는다.
    plant_representation_schema: Literal[
        "measured_band_complex_response_v1",
        "fullband_causal_joint_fir_operator_npz_v4",
    ] = MEASURED_BAND_PATH_SCHEMA_VERSION
    interpolation_schema: Literal[
        "bulk_delay_removed_piecewise_linear_complex_no_extrapolation_v1",
        "not_applicable_frozen_causal_fir_v1",
    ] = MEASURED_BAND_INTERPOLATION_SCHEMA
    linear_spectral_schema: Literal[
        "finite_sequence_linear_dtft_no_ifft_v1",
        "full_linear_causal_convolution_continuous_prefix_valid_crop_v1",
    ] = BROADBAND_LINEAR_SPECTRAL_SCHEMA
    dnh_domain: Literal["actuator_output_y_nl"] = BROADBAND_DNH_DOMAIN
    dnh_schema_version: Literal[
        "actuator_output_union_leakage_v1"
    ] = BROADBAND_DNH_SCHEMA_VERSION
    dnh_calibration_status: Literal[
        "requires_output_y_gradient_share_0p2_0p4"
    ] = BROADBAND_DNH_CALIBRATION_PLACEHOLDER

    # 이 두 항의 현재 구현은 time-domain S*y를 요구한다. measured-band 버전이
    # 별도로 검증되기 전에는 0 외의 값으로 조용히 compact FIR을 재도입할 수 없다.
    lambda_mrstft: Literal[0.0] = 0.0
    lambda_frame: Literal[0.0] = 0.0

    @model_validator(mode="after")
    def _validate_broadband(self) -> "BroadbandLossConfig":
        band_count = len(
            ControlBandContract.broadband_point_control().point_control_subbands_hz
        )
        if not 0.0 < float(self.subband_guard_q) < 1.0:
            raise ValueError(
                "loss.subband_guard_q는 (0,1)이어야 합니다 — 1이면 worst-band "
                "guard가 equal-subband 평균과 같아집니다"
            )
        if not 0.0 < float(self.subband_guard_alpha) < 1.0:
            raise ValueError(
                "loss.subband_guard_alpha는 (0,1)이어야 합니다 — 0이면 worst-band "
                "guard가 사라지고, 1이면 비선택 subband 그래디언트가 사라집니다"
            )
        if not 1 <= int(self.subband_guard_min_k) < band_count:
            raise ValueError(
                "loss.subband_guard_min_k는 1 이상이면서 subband 수보다 작아야 "
                f"합니다: {self.subband_guard_min_k} (subband={band_count})"
            )
        if float(self.lambda_dnh) <= 0.0:
            raise ValueError(
                "광대역 손실은 point-control union 밖 do-no-harm을 끌 수 없습니다"
            )
        if self.do_no_harm_bands is not None:
            raise ValueError(
                "광대역 손실은 do_no_harm_bands 직접 열거를 허용하지 않습니다 — "
                "보호 union의 전체 여집합은 control-band 계약에서 유도해야 합니다"
            )
        causal = self.plant_representation_schema == BROADBAND_CAUSAL_PATH_SCHEMA
        expected_interpolation = (
            BROADBAND_CAUSAL_INTERPOLATION_SCHEMA
            if causal
            else MEASURED_BAND_INTERPOLATION_SCHEMA
        )
        expected_linear = (
            BROADBAND_CAUSAL_CONVOLUTION_SCHEMA
            if causal
            else BROADBAND_LINEAR_SPECTRAL_SCHEMA
        )
        if self.interpolation_schema != expected_interpolation:
            raise ValueError(
                "광대역 plant/interpolation schema 조합이 다릅니다: "
                f"plant={self.plant_representation_schema}, "
                f"interpolation={self.interpolation_schema}"
            )
        if self.linear_spectral_schema != expected_linear:
            raise ValueError(
                "광대역 plant/linear schema 조합이 다릅니다: "
                f"plant={self.plant_representation_schema}, "
                f"linear={self.linear_spectral_schema}"
            )
        return self

    @classmethod
    def parse(cls, raw: dict[str, Any] | None) -> "BroadbandLossConfig":
        """광대역 YAML dict를 검증한다. 모르는/Stage-1 키는 즉시 거부한다."""

        return cls.model_validate(dict(raw or {}))


class BroadbandANCLoss(ANCLoss):
    """일곱 제어 subband를 독립 정규화하는 광대역 ANC 손실.

    실제 plant 적용, 비선형, DNH, frame/MR-STFT, saturation 및 FP32 경계는 부모의
    검증된 구현을 그대로 사용한다. 주 NMSE 집계만 광대역 계약으로 교체한다.
    """

    def __init__(
        self,
        plant: MeasuredBandPath | CausalFIRPath,
        loss_cfg: dict[str, Any] | None,
        sample_rate: int,
        nonlinear: RandomNonlinear | None = None,
        limiter_limit: float | None = None,
        control_band_contract: ControlBandContract | None = None,
    ) -> None:
        if isinstance(plant, DifferentiableSecondaryPath) or not isinstance(
            plant, (MeasuredBandPath, CausalFIRPath)
        ):
            raise TypeError(
                "광대역 loss는 legacy compact FIR/DifferentiableSecondaryPath를 사용할 "
                "수 없고 diagnostic MeasuredBandPath 또는 authority-bound "
                "CausalFIRPath만 허용합니다"
            )
        contract = (
            ControlBandContract.broadband_point_control()
            if control_band_contract is None
            else control_band_contract
        )
        if contract.role != "broadband_point_control":
            raise ValueError("Stage-1 control-band 계약을 광대역 손실에 사용할 수 없습니다")
        if int(sample_rate) != int(contract.sample_rate):
            raise ValueError(
                "광대역 손실 sample rate가 control-band 계약과 다릅니다: "
                f"loss={sample_rate}, contract={contract.sample_rate}"
            )

        cfg = BroadbandLossConfig.parse(loss_cfg)
        if cfg.control_band_contract_id != contract.contract_id:
            raise ValueError(
                "광대역 loss/control-band contract id가 다릅니다: "
                f"{cfg.control_band_contract_id} != {contract.contract_id}"
            )
        expected_causal = cfg.plant_representation_schema == BROADBAND_CAUSAL_PATH_SCHEMA
        if expected_causal != isinstance(plant, CausalFIRPath):
            raise ValueError(
                "광대역 loss plant object와 plant_representation_schema가 다릅니다"
            )

        # 부모는 Stage-1과 공유하는 키만 받는다. 광대역 키를 LossConfig에 추가하면
        # 기존 YAML의 extra=forbid 경계를 약화시키므로 명시적으로 투영한다.
        base_cfg = {
            field: getattr(cfg, field)
            for field in LossConfig.model_fields
        }
        super().__init__(
            plant=plant,
            loss_cfg=base_cfg,
            sample_rate=int(sample_rate),
            nonlinear=nonlinear,
            target_band_hz=tuple(contract.point_control_target_hz),
            trusted_band_hz=tuple(contract.point_control_target_hz),
            limiter_limit=limiter_limit,
        )

        self.broadband_loss_config = cfg
        self.loss_config = cfg
        # 외부 checkpoint/실험 계약이 그대로 봉인할 수 있는 authority.
        self.control_band_contract = contract
        self.control_band_contract_sha256 = contract.digest()
        self.point_control_subbands_hz = tuple(contract.point_control_subbands_hz)
        self.subband_guard_q = float(cfg.subband_guard_q)
        self.subband_guard_alpha = float(cfg.subband_guard_alpha)
        self.subband_guard_min_k = int(cfg.subband_guard_min_k)
        self.minimum_target_d_density_ratio = float(
            cfg.minimum_target_d_density_ratio
        )
        self.plant_representation_schema = cfg.plant_representation_schema
        self.interpolation_schema = cfg.interpolation_schema
        self.linear_spectral_schema = cfg.linear_spectral_schema
        self.dnh_domain = cfg.dnh_domain
        self.dnh_schema_version = cfg.dnh_schema_version
        self.dnh_calibration_status = cfg.dnh_calibration_status

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        result = 1
        while result < int(value):
            result *= 2
        return result

    def _linear_dtft_size(self, samples: int) -> int:
        """IFFT 없는 finite-sequence DTFT는 circular convolution이 아니다.

        ``H(f)Y(f)``는 유한 입력과 물리 LTI의 full linear-convolution DTFT를 선택한
        주파수에서 평가하는 식이다. IFFT/길이 절단을 하지 않으므로 wrap될 시간축이
        없다. 테스트는 known FIR의 full ``T+L-1`` 직접 convolution DTFT와 이 식을
        대조한다. 다만 이는 입력 유한열을 구간 밖 0으로 둔 수학적 formulation일
        뿐, 연속 녹음의 random crop에 필요한 이전 plant/model state를 만들지 않는다.
        따라서 criterion factory는 prefix/state 계약이 생기기 전 학습 admission을
        별도로 BLOCKED한다.
        """

        if int(samples) < 2:
            raise ValueError("measured-band DTFT에는 2샘플 이상이 필요합니다")
        return int(samples)

    def _target_density_by_subband(
        self, d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """item×subband target-``d`` spectral-density ratio와 유효 mask.

        receipt의 정의와 같이 각 subband의 평균 spectral density를 전체 point-control
        union의 평균 density로 나눈다. 합계 에너지가 아니라 bin당 density이므로 넓은
        고역 band가 단지 FFT bin이 많다는 이유로 통과하지 않는다. 입력 segment마다
        다시 계산해야 group-level receipt가 보장하지 않는 local silence를 걸러낼 수 있다.
        """

        if d.ndim != 2 or d.shape[-1] < 2:
            raise ValueError(
                "광대역 target-density 입력은 [batch,time]이고 2샘플 이상이어야 합니다"
            )
        samples = int(d.shape[-1])
        spectrum = torch.fft.rfft(d, dim=-1, norm="ortho")
        power = spectrum.real.square() + spectrum.imag.square()
        densities: list[torch.Tensor] = []
        total_power = power.new_zeros((power.shape[0],))
        total_bins = 0
        for index, (lo, hi) in enumerate(self.point_control_subbands_hz):
            lo_bin, hi_bin = self._band_bins(samples, lo, hi)
            if index != len(self.point_control_subbands_hz) - 1:
                hi_bin = min(
                    hi_bin,
                    int(math.ceil(float(hi) * samples / self.sample_rate)) - 1,
                )
            if lo_bin > hi_bin:
                raise ValueError(
                    f"세그먼트 {samples}샘플 FFT에 target-density subband "
                    f"[{lo:g}, {hi:g}] bin이 없습니다"
                )
            selected = power[..., lo_bin : hi_bin + 1]
            densities.append(selected.mean(dim=-1))
            total_power = total_power + selected.sum(dim=-1)
            total_bins += int(selected.shape[-1])
        if total_bins <= 0:
            raise RuntimeError("광대역 target-density union에 FFT bin이 없습니다")
        flat_density = total_power / float(total_bins)
        tiny = torch.finfo(flat_density.dtype).tiny
        ratios = torch.stack(
            tuple(
                torch.where(
                    flat_density > tiny,
                    band_density / flat_density.clamp_min(tiny),
                    torch.zeros_like(flat_density),
                )
                for band_density in densities
            ),
            dim=1,
        )
        valid = ratios.detach() >= self.minimum_target_d_density_ratio
        return ratios, valid

    def _subband_cvar(self, values: torch.Tensor) -> torch.Tensor:
        """일곱 band scalar의 상위 q(나쁜 값) 평균."""

        if values.ndim != 1 or values.numel() != len(self.point_control_subbands_hz):
            raise ValueError(
                "subband guard 입력은 control-band 계약과 같은 길이의 1-D여야 합니다"
            )
        count = int(values.numel())
        k = min(
            count,
            max(
                self.subband_guard_min_k,
                int(math.ceil(self.subband_guard_q * count)),
            ),
        )
        return values.topk(k).values.mean()

    def _dnh_band_bins(self, samples: int, lo_hz: float, hi_hz: float) -> tuple[int, int]:
        """보호 union 경계 bin을 DNH에서 제외한다.

        연속 구간이 겹치지 않아도 양끝 포함 FFT 산술은 150 Hz와 11.314 kHz 경계
        bin을 양측에 넣을 수 있다. 광대역 역할은 이를 실제 bin 수준에서도 0-overlap
        으로 만든다.
        """

        lo_bin, hi_bin = self._band_bins(samples, lo_hz, hi_hz)
        protected_lo, protected_hi = self.control_band_contract.point_control_target_hz
        # 각 DNH 조각도 [lo, hi)로 두어 octave/edge 경계 bin을 두 번 벌점하지
        # 않는다. Nyquist만 마지막 조각에 포함한다.
        if hi_hz < self.sample_rate / 2.0:
            hi_bin = min(
                hi_bin,
                int(math.ceil(hi_hz * samples / self.sample_rate)) - 1,
            )
        if lo_hz >= protected_hi:
            lo_bin = max(
                lo_bin,
                int(math.floor(protected_hi * samples / self.sample_rate)) + 1,
            )
        return lo_bin, hi_bin

    @staticmethod
    def _metric_band_name(index: int, band: tuple[float, float]) -> str:
        lo, hi = band
        return f"nmse_subband_{index}_{int(round(lo))}_{int(round(hi))}"

    def _main_nmse_objective(
        self,
        e: torch.Tensor,
        d: torch.Tensor,
        nmse_fullband_db: torch.Tensor,
        nmse_trusted_db: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """equal-subband item-CVaR와 worst-subband CVaR를 혼합한다."""

        del nmse_fullband_db, nmse_trusted_db
        raw_per_item = tuple(
            self._band_nmse_db(
                e,
                d,
                band,
                include_upper=index == len(self.point_control_subbands_hz) - 1,
            )
            for index, band in enumerate(self.point_control_subbands_hz)
        )
        density_ratios, density_valid = self._target_density_by_subband(d)
        return self._aggregate_subband_objective(
            raw_per_item, density_ratios, density_valid
        )

    def _aggregate_subband_objective(
        self,
        raw_per_item: tuple[torch.Tensor, ...],
        density_ratios: torch.Tensor,
        density_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """시간/DTFT 표현과 무관한 7대역 equal+worst 집계."""

        if len(raw_per_item) != len(self.point_control_subbands_hz):
            raise ValueError("광대역 subband objective 길이가 control 계약과 다릅니다")
        required_items = int(self.nmse_cvar_min_k)
        per_item: list[torch.Tensor] = []
        for index, (band, values) in enumerate(
            zip(self.point_control_subbands_hz, raw_per_item, strict=True)
        ):
            mask = density_valid[:, index]
            valid_count = int(mask.sum().detach())
            if valid_count < required_items:
                lo, hi = band
                raise ValueError(
                    "광대역 batch의 target-d density가 부족합니다: "
                    f"subband=[{lo:g},{hi:g})Hz, valid_items={valid_count}, "
                    f"required={required_items}, threshold="
                    f"{self.minimum_target_d_density_ratio:.2f}. "
                    "group-level coverage receipt는 sampled segment 자격을 대신하지 "
                    "않습니다"
                )
            per_item.append(values[mask])
        selected_per_item = tuple(per_item)
        # 각 band 안에서 item-CVaR를 먼저 적용한 scalar들. 대역 폭, FFT bin 수,
        # d 에너지와 무관하게 일곱 scalar에 정확히 1/7씩 baseline weight를 준다.
        band_objectives = torch.stack(
            tuple(self._worst_aggregate(values) for values in selected_per_item)
        )
        equal_subband = band_objectives.mean()
        guard = self._subband_cvar(band_objectives)
        objective = (
            (1.0 - self.subband_guard_alpha) * equal_subband
            + self.subband_guard_alpha * guard
        )

        low_items = torch.cat(selected_per_item[:4])
        high_items = torch.cat(selected_per_item[4:])
        metrics: dict[str, float] = {
            "nmse_subband_equal_db": float(equal_subband.detach()),
            "nmse_subband_guard_cvar_db": float(guard.detach()),
            "nmse_subband_worst_db": float(
                torch.stack(tuple(values.max() for values in per_item)).max().detach()
            ),
            "nmse_low_worst_db": float(low_items.max().detach()),
            "nmse_high_worst_db": float(high_items.max().detach()),
            "target_d_density_threshold": self.minimum_target_d_density_ratio,
        }
        for index, (band, values, aggregate) in enumerate(
            zip(
                self.point_control_subbands_hz,
                selected_per_item,
                band_objectives,
                strict=True,
            )
        ):
            prefix = self._metric_band_name(index, band)
            valid_values = density_ratios[:, index][density_valid[:, index]]
            metrics[f"{prefix}_density_valid_items"] = float(valid_values.numel())
            metrics[f"{prefix}_density_min_ratio"] = float(valid_values.min().detach())
            metrics[f"{prefix}_mean_db"] = float(values.mean().detach())
            metrics[f"{prefix}_cvar_db"] = float(self._cvar(values).detach())
            metrics[f"{prefix}_worst_db"] = float(values.max().detach())
            metrics[f"{prefix}_objective_db"] = float(aggregate.detach())
        return objective, metrics

    def _spectral_density_mask(
        self, d_power: torch.Tensor, *, n_fft: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        densities: list[torch.Tensor] = []
        total = d_power.new_zeros((d_power.shape[0],))
        total_bins = 0
        for index, (lo, hi) in enumerate(self.point_control_subbands_hz):
            lo_bin, hi_bin = self._band_bins(n_fft, lo, hi)
            if index != len(self.point_control_subbands_hz) - 1:
                hi_bin = min(
                    hi_bin,
                    int(math.ceil(float(hi) * n_fft / self.sample_rate)) - 1,
                )
            selected = d_power[..., lo_bin : hi_bin + 1]
            if selected.shape[-1] == 0:
                raise ValueError(f"zero-padded DTFT에 {lo:g}-{hi:g}Hz bin이 없습니다")
            densities.append(selected.mean(dim=-1))
            total = total + selected.sum(dim=-1)
            total_bins += int(selected.shape[-1])
        flat = total / float(total_bins)
        tiny = torch.finfo(flat.dtype).tiny
        ratios = torch.stack(
            tuple(
                torch.where(
                    flat > tiny,
                    value / flat.clamp_min(tiny),
                    torch.zeros_like(flat),
                )
                for value in densities
            ),
            dim=1,
        )
        return ratios, ratios.detach() >= self.minimum_target_d_density_ratio

    def _actuator_output_dnh(
        self, y_nl: torch.Tensor, *, n_fft: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """미측정 S를 전혀 사용하지 않는 actuator-output leakage 벌점.

        분모는 detach해 control-union bin에 DNH gradient가 생기지 않는다. 이 항은
        pressure dB 보장이 아니라 output containment prior이며, 실제 pressure 악화는
        물리 G4가 별도로 막는다.
        """

        flat = y_nl.squeeze(1)
        spectrum = torch.fft.rfft(flat, n=n_fft)
        power = spectrum.real.square() + spectrum.imag.square()
        protected_lo, protected_hi = self.control_band_contract.point_control_target_hz
        protected_lo_bin, protected_hi_bin = self._band_bins(
            n_fft, protected_lo, protected_hi
        )
        protected_power = power[
            ..., protected_lo_bin : protected_hi_bin + 1
        ].sum(dim=-1)
        total_power = power.sum(dim=-1)
        relative_floor = total_power.detach() * self.dnh_band_floor
        denominator = torch.maximum(protected_power.detach(), relative_floor)
        denominator = denominator + torch.finfo(power.dtype).tiny

        plan = self.do_no_harm
        assert plan is not None
        loss = y_nl.new_zeros(())
        metrics: dict[str, float] = {
            "y_dnh_control_gradient_detached": 1.0,
        }
        worst_db = -math.inf
        for item in plan.bands:
            lo, hi = item.band.as_tuple()
            lo_bin, hi_bin = self._dnh_band_bins(n_fft, lo, hi)
            if lo_bin > hi_bin:
                continue
            numerator = power[..., lo_bin : hi_bin + 1].sum(dim=-1)
            ratio = numerator / denominator
            loss = loss + float(item.weight) * self._worst_aggregate(ratio)
            ratio_db = 10.0 * torch.log10(ratio.detach() + 1.0e-30)
            observed = float(ratio_db.max())
            metrics[f"y_dnh_{int(lo)}_{int(hi)}_max_db"] = observed
            worst_db = max(worst_db, observed)
        if math.isfinite(worst_db):
            metrics["y_dnh_worst_db"] = worst_db
        return loss, metrics

    def _forward_fp32(
        self,
        y: torch.Tensor,
        d: torch.Tensor,
        loss_start_sample: int = 0,
        perturb: dict | None = None,
        nl_params: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """측정대역 DTFT만 사용하는 광대역 전용 forward.

        IFFT하는 circular convolution은 수행하지 않는다. 유한 입력의 DTFT에서
        ``D + H_measured*Y``만 평가하며 known FIR full linear convolution과 독립
        등가 테스트를 둔다. 이 메서드 자체는 finite zero-extended fixture와 독립
        수치 검증에만 사용한다. 연속 random segment를 ``pre-settled``라고 가정하지
        않으며, prefix/state 없는 Trainer 연결은 criterion factory가 차단한다.
        """

        if isinstance(self.plant, CausalFIRPath):
            return self._forward_causal_fp32(
                y,
                d,
                loss_start_sample=loss_start_sample,
                perturb=perturb,
                nl_params=nl_params,
            )

        if int(loss_start_sample) != 0:
            raise ValueError(
                "measured-band finite-sequence DTFT는 loss_start_sample=0만 "
                "허용합니다; random crop의 missing prefix/state를 settle crop으로 "
                "대체할 수 없습니다"
            )
        if y.ndim != 3 or d.shape != y.shape or y.shape[1] != 1:
            raise ValueError("광대역 measured-band loss 입력은 같은 [B,1,T]여야 합니다")
        batch = int(y.shape[0])
        y_nl = y
        if self.training and self.nonlinear is not None:
            if nl_params is None:
                nl_params = self.nonlinear.sample(batch)
            y_nl = self.nonlinear.apply_torch(y, nl_params)
        perturb = dict(perturb or {})
        if bool(perturb.get("allpass", False)):
            raise ValueError("measured-band response는 미검증 allpass perturb를 허용하지 않습니다")

        n_fft = self._linear_dtft_size(int(y.shape[-1]))
        Y = torch.fft.rfft(y_nl.squeeze(1), n=n_fft)
        D = torch.fft.rfft(d.squeeze(1), n=n_fft)
        d_power = D.real.square() + D.imag.square()
        frequencies = torch.fft.rfftfreq(
            n_fft, d=1.0 / float(self.sample_rate), device=y.device
        )
        raw_per_item: list[torch.Tensor] = []
        for index, (lo, hi) in enumerate(self.point_control_subbands_hz):
            lo_bin, hi_bin = self._band_bins(n_fft, lo, hi)
            if index != len(self.point_control_subbands_hz) - 1:
                hi_bin = min(
                    hi_bin,
                    int(math.ceil(float(hi) * n_fft / self.sample_rate)) - 1,
                )
            query = frequencies[lo_bin : hi_bin + 1]
            response = self.plant.response_at(
                query,
                jitter_samples=int(perturb.get("jitter", 0)),
                gain_db=float(perturb.get("gain_db", 0.0)),
                tilt_db_per_octave=float(perturb.get("tilt_db_per_octave", 0.0)),
            ).to(dtype=Y.dtype)
            estimate = D[..., lo_bin : hi_bin + 1] + response * Y[
                ..., lo_bin : hi_bin + 1
            ]
            estimate_power = estimate.real.square() + estimate.imag.square()
            target_power = d_power[..., lo_bin : hi_bin + 1]
            raw_per_item.append(
                10.0
                * torch.log10(
                    (estimate_power.sum(dim=-1) + 1.0e-10)
                    / (target_power.sum(dim=-1) + 1.0e-10)
                )
            )
        density_ratios, density_valid = self._spectral_density_mask(
            d_power, n_fft=n_fft
        )
        l_nmse, nmse_metrics = self._aggregate_subband_objective(
            tuple(raw_per_item), density_ratios, density_valid
        )
        l_dnh, dnh_metrics = self._actuator_output_dnh(y_nl, n_fft=n_fft)
        l_sat, u_over_limit = self.saturation_penalty(y)
        l_pow = y.pow(2).mean()
        zero = y.new_zeros(())
        self._last_terms = {
            "nmse": (1.0, l_nmse),
            "mrstft": (0.0, zero),
            "dnh": (self.lambda_dnh, l_dnh),
            "frame": (0.0, zero),
            "sat": (self.lambda_sat, l_sat),
            "pow": (self.lambda_pow, l_pow),
        }
        total = sum(
            (weight * term for weight, term in self._last_terms.values()),
            start=zero,
        )
        metrics: dict[str, float] = {
            "loss": float(total.detach()),
            "nmse_db": float(l_nmse.detach()),
            "nmse_control_union_db": float(l_nmse.detach()),
            "dnh": float(l_dnh.detach()),
            "sat": float(l_sat.detach()),
            "sat_u_over_limit_max": float(u_over_limit.abs().max().detach()),
            "out_pow": float(l_pow.detach()),
            "linear_dtft_n_fft": float(n_fft),
            "linear_dtft_uses_ifft": 0.0,
        }
        metrics.update(nmse_metrics)
        metrics.update(dnh_metrics)
        return total, metrics

    def _forward_causal_fp32(
        self,
        y: torch.Tensor,
        d: torch.Tensor,
        *,
        loss_start_sample: int,
        perturb: dict | None,
        nl_params: dict | None,
    ) -> tuple[torch.Tensor, dict]:
        """연속 prefix 전체에 S를 적용하고 오직 valid target crop에서 손실을 낸다.

        segment 왼쪽을 0으로 둔 finite-sequence 근사는 허용하지 않는다. 입력 ``y,d``는
        같은 연속 session/source에서 잘라 온 ``prefix + target``이어야 하며, prefix가
        적어도 S의 coarse delay + handoff + FIR history를 덮어야 한다. 모델 hidden-state
        warm-up은 이 하한보다 더 길 수 있고 Trainer가 exact prefix를 별도로 결속한다.
        """

        assert isinstance(self.plant, CausalFIRPath)
        if y.ndim != 3 or d.shape != y.shape or y.shape[1] != 1:
            raise ValueError("광대역 causal loss 입력은 같은 [B,1,T]여야 합니다")
        skip = int(loss_start_sample)
        if skip < int(self.plant.history_samples):
            raise ValueError(
                "causal S valid crop 전에 연속 prefix가 부족합니다: "
                f"prefix={skip}, required>={self.plant.history_samples}"
            )
        if skip >= int(y.shape[-1]) - 1:
            raise ValueError("causal S valid crop 뒤 target이 2샘플 이상 남아야 합니다")

        batch = int(y.shape[0])
        y_nl = y
        if self.training and self.nonlinear is not None:
            if nl_params is None:
                nl_params = self.nonlinear.sample(batch)
            y_nl = self.nonlinear.apply_torch(y, nl_params)
        if perturb is None:
            perturb = self.plant.sample_perturbation()
        s_y = self.plant(y_nl, perturb)
        # 측정 FIR에 극성이 이미 포함된다. 어디에서도 부호를 한 번 더 뒤집지 않는다.
        e = d + s_y

        e_flat = e.squeeze(1)[..., skip:]
        d_flat = d.squeeze(1)[..., skip:]
        y_valid = y[..., skip:]
        y_nl_valid = y_nl[..., skip:]
        e_pow = e_flat.square().sum(dim=-1)
        d_pow = d_flat.square().sum(dim=-1)
        nmse_fullband_db = 10.0 * torch.log10(
            (e_pow + 1.0e-10) / (d_pow + 1.0e-10)
        )
        l_nmse, nmse_metrics = self._main_nmse_objective(
            e_flat, d_flat, nmse_fullband_db, None
        )

        # 광대역 DNH 정의는 actuator output y_nl의 point-control union 여집합이다.
        # S의 비신뢰 out-of-band 위상을 다시 끌어오지 않는다.
        n_fft = int(e_flat.shape[-1])
        l_dnh, dnh_metrics = self._actuator_output_dnh(
            y_nl_valid, n_fft=n_fft
        )
        l_sat, u_over_limit = self.saturation_penalty(y_valid)
        l_pow = y_valid.square().mean()
        zero = y.new_zeros(())
        self._last_terms = {
            "nmse": (1.0, l_nmse),
            "mrstft": (0.0, zero),
            "dnh": (self.lambda_dnh, l_dnh),
            "frame": (0.0, zero),
            "sat": (self.lambda_sat, l_sat),
            "pow": (self.lambda_pow, l_pow),
        }
        total = sum(
            (weight * term for weight, term in self._last_terms.values()),
            start=zero,
        )
        metrics: dict[str, float] = {
            "loss": float(total.detach()),
            "nmse_db": float(l_nmse.detach()),
            "nmse_control_union_db": float(l_nmse.detach()),
            "nmse_fullband_db": float(nmse_fullband_db.mean().detach()),
            "nmse_fullband_worst_db": float(nmse_fullband_db.max().detach()),
            "dnh": float(l_dnh.detach()),
            "sat": float(l_sat.detach()),
            "sat_u_over_limit_max": float(u_over_limit.abs().max().detach()),
            "out_pow": float(l_pow.detach()),
            "causal_prefix_samples": float(skip),
            "causal_plant_history_samples": float(self.plant.history_samples),
            "causal_handoff_extra_samples": float(
                self.plant.data.handoff_extra_samples
            ),
            "linear_dtft_uses_ifft": 0.0,
        }
        metrics.update(nmse_metrics)
        metrics.update(dnh_metrics)
        return total, metrics


class BroadbandFullOctaveLossConfigV3(BaseModel):
    """v3 full-octave primitive만을 위한 별도 immutable 설정.

    v2의 ``control_band_contract_id``나 ``BroadbandLossConfig``를 받아서 숫자만
    확장하는 경로는 의도적으로 없다. inline v3 payload와 그 payload의 SHA가 모두
    있어야 하며, 이 설정 자체도 live v5 authority가 생기기 전에는 학습 admission을
    열 수 없다고 선언한다.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["broadband_full_octave_loss_primitive_v3"] = (
        BROADBAND_FULL_OCTAVE_LOSS_SCHEMA_VERSION
    )
    control_band_contract: BroadbandFullOctaveContractV3
    control_band_contract_sha256: str
    minimum_target_d_density_ratio: Literal[0.25] = 0.25
    minimum_valid_items_per_band: Literal[4] = 4
    item_cvar_q: Literal[0.25] = 0.25
    item_cvar_alpha: Literal[0.7] = 0.7
    octave_worst_guard_weight: Literal[0.7] = 0.7
    relative_nmse_floor_db: Literal[-80.0] = BROADBAND_FULL_OCTAVE_NMSE_FLOOR_DB
    stage1_positive_guard_weight: Literal[1.0] = 1.0
    low_high_positive_guard_weight: Literal[1.0] = 1.0
    diagnostic_design_hyperparameters_not_campaign_authority: Literal[True] = True
    campaign_hyperparameter_authority_sha256: None = None
    # live operator에서 output-y gradient share 0.2--0.4를 입증하기 전에는 어떤
    # 숫자도 canonical default가 아니다. 호출자가 diagnostic 값을 명시해야 하고,
    # 이 primitive는 calibration receipt를 발행/소비하지 않으므로 admission은 계속 BLOCKED다.
    lambda_dnh: float
    dnh_calibration_receipt_sha256: None = None
    dnh_calibration_status: Literal[
        "BLOCKED_MISSING_OUTPUT_Y_GRADIENT_SHARE_0P2_0P4_RECEIPT"
    ] = "BLOCKED_MISSING_OUTPUT_Y_GRADIENT_SHARE_0P2_0P4_RECEIPT"
    dnh_margin_db: float = DEFAULT_DNH_MARGIN_DB
    dnh_band_floor_db: Literal[-60.0] = -60.0
    dnh_domain: Literal["actuator_output_y_nl"] = BROADBAND_DNH_DOMAIN
    dnh_schema_version: Literal["actuator_output_union_margin_hinge_v3"] = (
        BROADBAND_FULL_OCTAVE_DNH_SCHEMA_VERSION
    )
    objective_semantics: Literal[
        "seven_exact_one_seventh_baseline_plus_additive_worst_guard"
    ] = "seven_exact_one_seventh_baseline_plus_additive_worst_guard"
    stage1_guard_semantics: Literal[
        "four_independent_positive_attenuation_hinges_150_1600"
    ] = "four_independent_positive_attenuation_hinges_150_1600"
    legacy_v2_automatic_promotion_allowed: Literal[False] = False
    legacy_v2_checkpoint_automatic_promotion_allowed: Literal[False] = False
    actual_family_balanced_batch_receipt_consumed: Literal[False] = False
    canonical_training_eligible: Literal[False] = False
    training_admission_status: Literal[
        "BLOCKED_INCOMPLETE_BROADBAND_V3_ADMISSION"
    ] = BROADBAND_FULL_OCTAVE_TRAINING_BLOCKER
    training_admission_blockers: tuple[str, ...] = (
        BROADBAND_FULL_OCTAVE_ADMISSION_BLOCKERS
    )

    @model_validator(mode="after")
    def _validate_full_octave(self) -> "BroadbandFullOctaveLossConfigV3":
        # resolver를 일부러 거친다. schema_version이 v2이면 canonical v3와 모양이
        # 비슷해도 이 지점에서 자동 승격되지 않는다.
        resolved = resolve_control_band_contract(
            self.control_band_contract.model_dump(mode="python")
        )
        if type(resolved) is not BroadbandFullOctaveContractV3:
            raise ValueError("full-octave loss에는 별도 v3 control-band 계약이 필요합니다")
        canonical = BroadbandFullOctaveContractV3.canonical()
        if resolved.model_dump(mode="json") != canonical.model_dump(mode="json"):
            raise ValueError("full-octave loss에는 exact canonical v3 payload가 필요합니다")
        supplied = _lower_sha256(
            self.control_band_contract_sha256,
            label="full-octave control-band contract SHA",
        )
        if supplied != resolved.digest() or supplied != canonical.digest():
            raise ValueError("inline full-octave v3 payload와 SHA가 다릅니다")
        if not math.isfinite(float(self.lambda_dnh)) or float(self.lambda_dnh) <= 0.0:
            raise ValueError("full-octave control union 밖 DNH를 끌 수 없습니다")
        if (
            not math.isfinite(float(self.dnh_margin_db))
            or float(self.dnh_margin_db) > float(DEFAULT_DNH_MARGIN_DB) + 1.0e-12
        ):
            raise ValueError("v3 DNH margin을 G4-consistent 한계보다 완화할 수 없습니다")
        if tuple(self.training_admission_blockers) != (
            BROADBAND_FULL_OCTAVE_ADMISSION_BLOCKERS
        ):
            raise ValueError("v3 training admission blocker 집합을 변경할 수 없습니다")
        return self

    @classmethod
    def parse(
        cls,
        raw: Mapping[str, Any] | "BroadbandFullOctaveLossConfigV3",
    ) -> "BroadbandFullOctaveLossConfigV3":
        if isinstance(raw, cls):
            return raw
        return cls.model_validate(dict(raw))


class BroadbandFullOctaveLossPrimitiveV3(nn.Module):
    """live causal P/S 연결 전 독립 검산 가능한 v3 후단 손실 primitive.

    입력 ``secondary_output``은 연속 prefix/state를 가진 causal S가 이미 계산한
    ``S*y``의 valid crop이어야 한다. 이 클래스는 식별 authority나 prefix를 만들지
    않고 ``e = d + S*y``의 부호, 7 octave 집계, Stage-1 guard, actuator-output DNH만
    담당한다. 따라서 criterion factory/trainer에 연결되어 있지 않으며 config에도
    ``canonical_training_eligible=False``가 봉인된다.
    """

    def __init__(
        self,
        loss_cfg: Mapping[str, Any] | BroadbandFullOctaveLossConfigV3,
        *,
        sample_rate: int = 48_000,
    ) -> None:
        super().__init__()
        cfg = BroadbandFullOctaveLossConfigV3.parse(loss_cfg)
        contract = cfg.control_band_contract
        if int(sample_rate) != int(contract.sample_rate):
            raise ValueError("full-octave loss sample rate와 v3 계약이 다릅니다")
        self.loss_config = cfg
        self.control_band_contract = contract
        self.control_band_contract_sha256 = contract.digest()
        self.sample_rate = int(sample_rate)
        self.objective_bands_hz = tuple(
            tuple(float(value) for value in band)
            for band in contract.equal_weight_octave_objective_bands_hz
        )
        self.physical_bands_hz = tuple(
            tuple(float(value) for value in band)
            for band in contract.physical_identification_subbands_hz
        )
        self.stage1_guard_bands_hz = tuple(
            tuple(float(value) for value in band)
            for band in contract.stage1_low_guard_subbands_hz
        )
        if self.objective_bands_hz != BROADBAND_V3_OCTAVE_OBJECTIVE_BANDS_HZ:
            raise ValueError("v3 octave objective bytes가 canonical constant와 다릅니다")
        if self.physical_bands_hz != BROADBAND_V3_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ:
            raise ValueError("v3 physical identification bands가 canonical과 다릅니다")
        if self.stage1_guard_bands_hz != BROADBAND_V3_STAGE1_LOW_GUARD_SUBBANDS_HZ:
            raise ValueError("v3 Stage-1 guard bands가 canonical과 다릅니다")
        self.minimum_density_ratio = float(cfg.minimum_target_d_density_ratio)
        self.minimum_valid_items = int(cfg.minimum_valid_items_per_band)
        self.item_cvar_q = float(cfg.item_cvar_q)
        self.item_cvar_alpha = float(cfg.item_cvar_alpha)
        self.octave_worst_guard_weight = float(cfg.octave_worst_guard_weight)
        self.relative_nmse_floor_db = float(cfg.relative_nmse_floor_db)
        self.relative_nmse_floor_ratio = 10.0 ** (
            self.relative_nmse_floor_db / 10.0
        )
        self.lambda_dnh = float(cfg.lambda_dnh)
        self.dnh_band_floor = 10.0 ** (float(cfg.dnh_band_floor_db) / 10.0)
        self.dnh_plan = DoNoHarmPlan.derive(
            protected=FrequencyBand(
                lo_hz=self.objective_bands_hz[0][0],
                hi_hz=self.objective_bands_hz[-1][1],
            ),
            nyquist_hz=self.sample_rate / 2.0,
            margin_db=float(cfg.dnh_margin_db),
        )

    @staticmethod
    def _as_batch_time(value: torch.Tensor, *, label: str) -> torch.Tensor:
        if value.ndim == 3 and value.shape[1] == 1:
            value = value[:, 0, :]
        if value.ndim != 2 or int(value.shape[0]) < 1 or int(value.shape[-1]) < 2:
            raise ValueError(f"{label}는 [batch,time] 또는 [batch,1,time]이어야 합니다")
        if not torch.isfinite(value).all():
            raise ValueError(f"{label}에 NaN/Inf가 있습니다")
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
            raise ValueError(
                f"{samples}샘플 FFT에 [{lo:g},{hi:g})Hz bin이 없습니다"
            )
        return lo_bin, hi_bin

    def _density_ratios(
        self,
        target_power: torch.Tensor,
        *,
        samples: int,
        bands_hz: Sequence[Sequence[float]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        densities: list[torch.Tensor] = []
        total_power = target_power.new_zeros((target_power.shape[0],))
        total_bins = 0
        for index, band in enumerate(bands_hz):
            lo_bin, hi_bin = self._band_bins(
                samples,
                band,
                include_upper=index == len(bands_hz) - 1,
            )
            selected = target_power[..., lo_bin : hi_bin + 1]
            densities.append(selected.mean(dim=-1))
            total_power = total_power + selected.sum(dim=-1)
            total_bins += int(selected.shape[-1])
        flat_density = total_power / float(total_bins)
        tiny = torch.finfo(flat_density.dtype).tiny
        ratios = torch.stack(
            tuple(
                torch.where(
                    flat_density > tiny,
                    value / flat_density.clamp_min(tiny),
                    torch.zeros_like(flat_density),
                )
                for value in densities
            ),
            dim=1,
        )
        return ratios, ratios.detach() >= self.minimum_density_ratio

    def _band_nmse_db(
        self,
        error_power: torch.Tensor,
        target_power: torch.Tensor,
        *,
        samples: int,
        band_hz: Sequence[float],
        include_upper: bool,
    ) -> torch.Tensor:
        lo_bin, hi_bin = self._band_bins(
            samples, band_hz, include_upper=include_upper
        )
        e_selected = error_power[..., lo_bin : hi_bin + 1]
        d_selected = target_power[..., lo_bin : hi_bin + 1]
        weights = torch.full(
            (hi_bin - lo_bin + 1,),
            2.0,
            dtype=e_selected.dtype,
            device=e_selected.device,
        )
        if lo_bin == 0:
            weights[0] = 1.0
        if samples % 2 == 0 and hi_bin == samples // 2:
            weights[-1] = 1.0
        e_sum = (e_selected * weights).sum(dim=-1)
        d_sum = (d_selected * weights).sum(dim=-1)
        # 고정 absolute epsilon은 동일 파형의 레벨만 낮췄을 때 NMSE를 0 dB로
        # 끌어올린다. 반대로 (e+tiny)/(d+tiny)는 exact cancellation의 backward에서
        # reciprocal overflow 뒤 0*inf를 만들어 NaN gradient를 낼 수 있다. target
        # energy에 비례한 floor를 먼저 만들고 log-domain 차이를 취해 두 문제를 함께
        # 막는다. floor 아래는 이미 -80 dB이므로 추가 gradient를 주지 않는다.
        tiny = torch.finfo(e_sum.dtype).tiny
        safe_d_sum = d_sum.clamp_min(tiny)
        relative_e_floor = (
            safe_d_sum.detach() * self.relative_nmse_floor_ratio
        ).clamp_min(tiny)
        safe_e_sum = torch.maximum(e_sum, relative_e_floor)
        return (10.0 / math.log(10.0)) * (
            torch.log(safe_e_sum) - torch.log(safe_d_sum)
        )

    def _item_objective(self, values: torch.Tensor) -> torch.Tensor:
        flat = values.reshape(-1)
        if int(flat.numel()) < self.minimum_valid_items:
            raise ValueError("v3 item objective에 valid item이 4개 미만입니다")
        k = max(
            self.minimum_valid_items,
            int(math.ceil(self.item_cvar_q * int(flat.numel()))),
        )
        k = min(k, int(flat.numel()))
        cvar = flat.topk(k).values.mean()
        return (1.0 - self.item_cvar_alpha) * flat.mean() + self.item_cvar_alpha * cvar

    def _qualified_objectives(
        self,
        raw_per_band: Sequence[torch.Tensor],
        density_ratios: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        bands_hz: Sequence[Sequence[float]],
        role: str,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict[str, float]]:
        if (
            len(raw_per_band) != len(bands_hz)
            or tuple(density_ratios.shape) != tuple(valid_mask.shape)
            or int(density_ratios.shape[1]) != len(bands_hz)
        ):
            raise ValueError(f"{role} band/density shape가 계약과 다릅니다")
        aggregates: list[torch.Tensor] = []
        selected: list[torch.Tensor] = []
        metrics: dict[str, float] = {}
        for index, (band, values) in enumerate(
            zip(bands_hz, raw_per_band, strict=True)
        ):
            mask = valid_mask[:, index]
            valid_count = int(mask.sum().detach())
            if valid_count < self.minimum_valid_items:
                lo, hi = (float(value) for value in band)
                raise ValueError(
                    f"v3 {role} target-energy 자격 부족: band=[{lo:g},{hi:g})Hz, "
                    f"valid_items={valid_count}, required={self.minimum_valid_items}, "
                    f"threshold={self.minimum_density_ratio:.2f}"
                )
            qualified = values[mask]
            aggregate = self._item_objective(qualified)
            selected.append(qualified)
            aggregates.append(aggregate)
            valid_ratios = density_ratios[:, index][mask]
            prefix = f"{role}_{index}"
            metrics[f"{prefix}_valid_items"] = float(valid_count)
            metrics[f"{prefix}_density_min_ratio"] = float(
                valid_ratios.min().detach()
            )
            metrics[f"{prefix}_objective_db"] = float(aggregate.detach())
            metrics[f"{prefix}_worst_db"] = float(qualified.max().detach())
        return torch.stack(tuple(aggregates)), tuple(selected), metrics

    def _actuator_output_dnh(
        self, actuator_output: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        samples = int(actuator_output.shape[-1])
        spectrum = torch.fft.rfft(actuator_output, dim=-1, norm="ortho")
        power = spectrum.real.square() + spectrum.imag.square()
        protected_lo = self.objective_bands_hz[0][0]
        protected_hi = self.objective_bands_hz[-1][1]
        lo_bin, hi_bin = self._band_bins(
            samples, (protected_lo, protected_hi), include_upper=True
        )
        protected_power = power[..., lo_bin : hi_bin + 1].sum(dim=-1)
        total_power = power.sum(dim=-1)
        denominator = torch.maximum(
            protected_power.detach(), total_power.detach() * self.dnh_band_floor
        ) + torch.finfo(power.dtype).tiny
        loss = actuator_output.new_zeros(())
        metrics: dict[str, float] = {"v3_dnh_control_gradient_detached": 1.0}
        worst_db = -math.inf
        for index, item in enumerate(self.dnh_plan.bands):
            lo, hi = item.band.as_tuple()
            try:
                outside_lo, outside_hi = self._band_bins(
                    samples,
                    (lo, hi),
                    include_upper=hi >= self.sample_rate / 2.0,
                )
            except ValueError:
                # protected edge와 octave edge의 decimal 차이가 1 FFT bin보다 작은
                # 1e-8 Hz 수치 조각을 만들 수 있다. 표현 가능한 bin이 없으므로 감시
                # 임계 완화가 아니라 empty discrete set 제거다.
                continue
            # protected union의 양 경계 bin은 DNH가 다시 소유하지 않는다.
            if lo >= protected_hi:
                outside_lo = max(
                    outside_lo,
                    int(math.floor(protected_hi * samples / self.sample_rate)) + 1,
                )
            if outside_lo > outside_hi:
                continue
            numerator = power[..., outside_lo : outside_hi + 1].sum(dim=-1)
            ratio_db = 10.0 * torch.log10(numerator / denominator + 1.0e-30)
            hinge = torch.relu(ratio_db - float(item.margin_db))
            loss = loss + float(item.weight) * self._item_objective(hinge)
            observed = float(ratio_db.max().detach())
            metrics[f"v3_dnh_band_{index}_max_db"] = observed
            worst_db = max(worst_db, observed)
        if math.isfinite(worst_db):
            metrics["v3_dnh_worst_db"] = worst_db
        return loss, metrics

    def forward(
        self,
        actuator_output: torch.Tensor,
        target_d: torch.Tensor,
        secondary_output: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """valid crop에서 v3 objective를 계산한다.

        ``secondary_output``은 ``S*y``이고 이 메서드가 오직 ``d + S*y``를 만든다.
        별도 부호 반전 인자는 존재하지 않는다.
        """

        y = self._as_batch_time(actuator_output, label="actuator_output")
        d = self._as_batch_time(target_d, label="target_d")
        s_y = self._as_batch_time(secondary_output, label="secondary_output")
        if y.shape != d.shape or s_y.shape != d.shape:
            raise ValueError("v3 loss의 y/d/S*y shape가 서로 다릅니다")
        # 측정 FIR의 극성은 S*y에 포함되어 있다. 추가 반전 없이 이 식만 허용한다.
        e = d + s_y
        samples = int(d.shape[-1])
        D = torch.fft.rfft(d, dim=-1, norm="ortho")
        E = torch.fft.rfft(e, dim=-1, norm="ortho")
        d_power = D.real.square() + D.imag.square()
        e_power = E.real.square() + E.imag.square()

        octave_raw = tuple(
            self._band_nmse_db(
                e_power,
                d_power,
                samples=samples,
                band_hz=band,
                include_upper=index == len(self.objective_bands_hz) - 1,
            )
            for index, band in enumerate(self.objective_bands_hz)
        )
        octave_density, octave_valid = self._density_ratios(
            d_power, samples=samples, bands_hz=self.objective_bands_hz
        )
        octave_objectives, _, metrics = self._qualified_objectives(
            octave_raw,
            octave_density,
            octave_valid,
            bands_hz=self.objective_bands_hz,
            role="v3_octave",
        )
        octave_equal = octave_objectives.mean()  # 각 octave 정확히 1/7 baseline
        octave_worst = octave_objectives.max()
        # equal baseline은 각 octave에 정확히 1/7을 유지한다. worst 항은 convex
        # replacement가 아니라 별도 additive guard이므로 비-worst octave의 계수를
        # (1-weight)/7로 조용히 줄이지 않는다.
        octave_mixed = (
            octave_equal + self.octave_worst_guard_weight * octave_worst
        )
        low_worst = octave_objectives[:4].max()
        high_worst = octave_objectives[4:].max()
        # 저역의 큰 성공과 고역 실패(또는 반대)가 평균에서 상쇄되지 않는다.
        low_high_positive_guard = torch.relu(low_worst) + torch.relu(high_worst)

        # Stage-1 guard 자격은 150--1600만 다시 정규화하지 않고, 별도 physical
        # identification 8-band population과 같은 denominator를 쓴다.
        physical_density, physical_valid = self._density_ratios(
            d_power, samples=samples, bands_hz=self.physical_bands_hz
        )
        stage1_raw = tuple(
            self._band_nmse_db(
                e_power,
                d_power,
                samples=samples,
                band_hz=band,
                include_upper=False,
            )
            for band in self.stage1_guard_bands_hz
        )
        stage1_columns = tuple(range(1, 5))
        stage1_density = physical_density[:, stage1_columns]
        stage1_valid = physical_valid[:, stage1_columns]
        stage1_objectives, _, stage1_metrics = self._qualified_objectives(
            stage1_raw,
            stage1_density,
            stage1_valid,
            bands_hz=self.stage1_guard_bands_hz,
            role="v3_stage1_guard",
        )
        stage1_positive_guard = torch.relu(stage1_objectives).sum()

        dnh, dnh_metrics = self._actuator_output_dnh(y)
        objective = (
            octave_mixed
            + float(self.loss_config.stage1_positive_guard_weight)
            * stage1_positive_guard
            + float(self.loss_config.low_high_positive_guard_weight)
            * low_high_positive_guard
            + self.lambda_dnh * dnh
        )
        metrics.update(stage1_metrics)
        metrics.update(dnh_metrics)
        metrics.update(
            {
                "loss": float(objective.detach()),
                "nmse_v3_octave_equal_db": float(octave_equal.detach()),
                "nmse_v3_octave_worst_objective_db": float(
                    octave_worst.detach()
                ),
                "nmse_v3_low_worst_objective_db": float(low_worst.detach()),
                "nmse_v3_high_worst_objective_db": float(high_worst.detach()),
                "v3_stage1_positive_guard": float(
                    stage1_positive_guard.detach()
                ),
                "v3_low_high_positive_guard": float(
                    low_high_positive_guard.detach()
                ),
                "v3_dnh": float(dnh.detach()),
                "v3_dnh_lambda_diagnostic": self.lambda_dnh,
                "v3_dnh_gradient_share_calibrated": 0.0,
                "v3_dnh_replaces_physical_err_g4": 0.0,
                "v3_equal_octave_weight": 1.0 / 7.0,
                "v3_relative_nmse_floor_db": self.relative_nmse_floor_db,
                "v3_design_hyperparameters_diagnostic_only": 1.0,
                "v3_campaign_hyperparameter_authority_present": 0.0,
                "v3_stage1_upper_1600_hz_exclusive": 1.0,
                "v3_frequency_1600_objective_octave_index": 4.0,
                "v3_canonical_training_eligible": 0.0,
                "v3_actual_family_balanced_batch_receipt_consumed": 0.0,
            }
        )
        return objective, metrics


__all__ = [
    "BROADBAND_CAUSAL_CONVOLUTION_SCHEMA",
    "BROADBAND_CAUSAL_INTERPOLATION_SCHEMA",
    "BROADBAND_CAUSAL_PATH_SCHEMA",
    "BROADBAND_DNH_CALIBRATION_PLACEHOLDER",
    "BROADBAND_DNH_DOMAIN",
    "BROADBAND_DNH_SCHEMA_VERSION",
    "BROADBAND_FULL_OCTAVE_LOSS_SCHEMA_VERSION",
    "BROADBAND_FULL_OCTAVE_DNH_SCHEMA_VERSION",
    "BROADBAND_FULL_OCTAVE_NMSE_FLOOR_DB",
    "BROADBAND_FULL_OCTAVE_ADMISSION_BLOCKERS",
    "BROADBAND_FULL_OCTAVE_TRAINING_BLOCKER",
    "BROADBAND_LINEAR_SPECTRAL_SCHEMA",
    "BROADBAND_LOSS_SCHEMA_VERSION",
    "BroadbandANCLoss",
    "BroadbandFullOctaveLossConfigV3",
    "BroadbandFullOctaveLossPrimitiveV3",
    "BroadbandLossConfig",
    "CausalFIRPath",
    "CausalFIRPathData",
]
