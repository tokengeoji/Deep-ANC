"""Canonical digital-reference 모델 입력 계약.

Stage-1 배포는 출력 clock이 진행시킨다. 따라서 controller는 미래 digital
reference와 exact-zero 호환 채널만 입력으로 사용한다. 비동기 ERR 마이크는
안전·평가 witness이며 모델 feature가 아니다. 학습과 streaming이 같은 payload와
digest를 결속하도록 dataset/runtime adapter와 독립된 공용 모듈에 둔다.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

if TYPE_CHECKING:
    import torch


_FROZEN = ConfigDict(frozen=True, extra="forbid")


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RefOnlyModelInputContract(BaseModel):
    """Output-clock-master ANC의 2채널 모델 호환 계약."""

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


def canonical_stage1_model_input_contract() -> RefOnlyModelInputContract:
    """Canonical Stage-1 자격을 갖는 유일한 모델 입력 계약을 반환한다."""

    return RefOnlyModelInputContract(error_dropout_probability=1.0)


def canonical_stage1_model_input_payload() -> dict[str, Any]:
    return canonical_stage1_model_input_contract().model_dump(mode="json")


def resolve_stage1_model_input_contract(
    data_cfg: dict[str, Any] | None,
) -> RefOnlyModelInputContract | None:
    """Opt-in data 계약을 해석하고 불완전하거나 상충하는 변형을 거부한다.

    누락은 의도적으로 허용한다. 그래야 legacy/diagnostic dataset이 과거 channel
    dropout 분포를 유지한다. Canonical role admission은 별도로 exact payload를
    강제한다.
    """

    if not isinstance(data_cfg, dict):
        return None
    raw = data_cfg.get("model_input_contract")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("data.model_input_contract는 exact mapping이어야 합니다")
    canonical = canonical_stage1_model_input_payload()
    if set(raw) != set(canonical):
        raise ValueError(
            "data.model_input_contract key 집합이 exact하지 않습니다: "
            f"missing={sorted(set(canonical) - set(raw))}, "
            f"extra={sorted(set(raw) - set(canonical))}"
        )
    contract = RefOnlyModelInputContract.model_validate(raw)
    if contract.model_dump(mode="json") != canonical:
        raise ValueError(
            "Stage-1 canonical model input은 digital reference 유지 + ERR exact-zero "
            "계약과 정확히 같아야 합니다"
        )
    if str(data_cfg.get("reference_mode", "digital")) != "digital":
        raise ValueError("ref-only Stage-1 model input은 reference_mode=digital 전용입니다")

    dropout = data_cfg.get("broadband_channel_dropout")
    if dropout is not None:
        expected = {"reference_probability": 0.0, "error_probability": 1.0}
        if not isinstance(dropout, dict) or set(dropout) != set(expected):
            raise ValueError(
                "model-input contract와 broadband_channel_dropout의 key 집합이 "
                "상충합니다"
            )
        if any(float(dropout[key]) != value for key, value in expected.items()):
            raise ValueError(
                "model-input contract와 broadband_channel_dropout이 상충합니다: "
                "reference=0/error=1이어야 합니다"
            )
    return contract


def apply_stage1_ref_only_numpy(
    reference: np.ndarray,
    error_feature: np.ndarray,
    contract: RefOnlyModelInputContract | None,
) -> tuple[np.ndarray, np.ndarray]:
    """확률적 dropout이나 dtype 변경 없이 ERR exact-zero를 적용한다."""

    if contract is None:
        return reference, error_feature
    ref = np.asarray(reference)
    err = np.asarray(error_feature)
    if ref.shape != err.shape or ref.ndim != 1:
        raise ValueError("ref-only dataset input은 같은 shape의 1-D REF/ERR여야 합니다")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(err)):
        raise ValueError("ref-only dataset input에 NaN/Inf가 있습니다")
    if not np.any(ref != 0.0):
        raise ValueError("ref-only dataset item의 digital reference 전체가 0입니다")
    return ref, np.zeros_like(err)


def validate_stage1_ref_only_tensor(
    value: "torch.Tensor",
    contract: RefOnlyModelInputContract | None,
    *,
    label: str,
) -> None:
    """Canonical train/val batch의 REF/ERR 계약을 채널별로 한 번씩 검사한다.

    경량 config 도구가 PyTorch/CUDA를 eager import하지 않도록 tensor 경계에서만
    torch를 가져온다. ERR ``count_nonzero``는 NaN/Inf도 nonzero로 거부한다. REF의
    per-item max-abs는 finite와 전체-zero 여부를 한 reduction으로 함께 판정한다.
    """

    if contract is None:
        return
    import torch

    if not isinstance(value, torch.Tensor) or value.ndim not in (2, 3):
        raise ValueError(f"{label}는 [2,T] 또는 [B,2,T] tensor여야 합니다")
    channel_axis = 0 if value.ndim == 2 else 1
    if int(value.shape[channel_axis]) != 2 or int(value.shape[-1]) < 1:
        raise ValueError(f"{label} channel/length shape가 ref-only 계약과 다릅니다")
    reference = value[0] if value.ndim == 2 else value[:, 0]
    error = value[1] if value.ndim == 2 else value[:, 1]
    if int(torch.count_nonzero(error).item()) != 0:
        raise ValueError(f"{label} ERR feature는 finite exact zero여야 합니다")
    flattened = reference.reshape(1, -1) if value.ndim == 2 else reference.reshape(
        reference.shape[0], -1
    )
    maximum_absolute = torch.amax(torch.abs(flattened), dim=1)
    if not bool(torch.isfinite(maximum_absolute).all().item()):
        raise ValueError(f"{label}의 x_ref에 NaN/Inf가 있습니다")
    if bool(torch.any(maximum_absolute == 0).item()):
        raise ValueError(f"{label}에 x_ref 전체-zero(dropout) item이 있습니다")


__all__ = [
    "RefOnlyModelInputContract",
    "apply_stage1_ref_only_numpy",
    "canonical_stage1_model_input_contract",
    "canonical_stage1_model_input_payload",
    "resolve_stage1_model_input_contract",
    "validate_stage1_ref_only_tensor",
]
