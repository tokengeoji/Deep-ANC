"""측정 tone 안에서만 정의되는 광대역 P/S 전달함수.

광대역 interleaved 측정은 Nyquist 전체가 아니라 약 100--11.4 kHz의 복소 tone만
관측한다. 이 관측으로 만든 compact FIR을 전대역 물리 모델처럼 사용하면 설계행렬의
미관측 null-space가 학습 그래디언트에 들어간다. 이 모듈은 그 경로를 만들지 않는다.

전달함수는 bulk fractional delay를 제거한 residual을 piecewise-linear로 보간한 뒤
같은 delay를 다시 적용한다. 요청 주파수가 유효 제어대역 또는 측정 tone의 convex
hull 밖이면 0으로 메우지 않고 실패한다. 시간영역 ``forward``도 의도적으로 제공하지
않는다. 소비자는 zero-padded DTFT에서 필요한 제어대역 bin만 요청해야 한다.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


MEASURED_BAND_PATH_SCHEMA_VERSION = "measured_band_complex_response_v1"
MEASURED_BAND_SOURCE_ARTIFACT_SCHEMA = (
    "broadband_measured_band_plant_v2_raw_derived"
)
MEASURED_BAND_INTERPOLATION_SCHEMA = (
    "bulk_delay_removed_piecewise_linear_complex_no_extrapolation_v1"
)
MEASURED_BAND_HOLDOUT_SCHEMA = "every_other_tone_seven_subband_holdout_v1"
MEASURED_BAND_MIN_HOLDOUT_AGREEMENT = 0.995
MEASURED_BAND_MAX_HOLDOUT_RELATIVE_ERROR = 0.10
MEASURED_BAND_MIN_HOLDOUT_TONES = 4


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label}는 lowercase SHA-256이어야 합니다")
    return text


def _npz_scalar(archive: Any, key: str) -> Any:
    if key not in archive:
        raise ValueError(f"measured-band plant에 {key} metadata가 없습니다")
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"measured-band plant의 {key}가 scalar가 아닙니다")
    return value.reshape(-1)[0].item()


def _immutable_vector(value: object, *, dtype: np.dtype[Any], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).reshape(-1).copy()
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{label}이 비었거나 NaN/Inf를 포함합니다")
    result.setflags(write=False)
    return result


def _response_sha256(
    *,
    role: str,
    sample_rate: int,
    frequencies_hz: np.ndarray,
    transfer: np.ndarray,
    bulk_delay_samples: int,
    bulk_delay_fractional_samples: float,
    pre_roll_samples: int,
    effective_delay_samples: int,
    fractional_effective_delay_samples: float,
    delay_semantics: str,
    valid_band_hz: tuple[float, float],
    control_band_contract_sha256: str,
    source_analysis_sha256: str,
    plant_evidence_sha256: str,
) -> str:
    header = {
        "schema_version": MEASURED_BAND_PATH_SCHEMA_VERSION,
        "interpolation_schema": MEASURED_BAND_INTERPOLATION_SCHEMA,
        "role": role,
        "sample_rate": int(sample_rate),
        "bulk_delay_samples": int(bulk_delay_samples),
        "bulk_delay_fractional_samples": float(bulk_delay_fractional_samples),
        "pre_roll_samples": int(pre_roll_samples),
        "effective_delay_samples": int(effective_delay_samples),
        "fractional_effective_delay_samples": float(
            fractional_effective_delay_samples
        ),
        "delay_semantics": str(delay_semantics),
        "valid_band_hz": [float(value) for value in valid_band_hz],
        "control_band_contract_sha256": control_band_contract_sha256,
        "source_analysis_sha256": source_analysis_sha256,
        "plant_evidence_sha256": plant_evidence_sha256,
        "tone_count": int(frequencies_hz.size),
    }
    digest = hashlib.sha256(_canonical_json_bytes(header))
    for array in (
        np.asarray(frequencies_hz, dtype="<f8"),
        np.asarray(transfer.real, dtype="<f8"),
        np.asarray(transfer.imag, dtype="<f8"),
    ):
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _interpolate_numpy(
    frequencies_hz: np.ndarray,
    transfer: np.ndarray,
    query_hz: np.ndarray,
    *,
    bulk_delay_fractional_samples: float,
    sample_rate: int,
) -> np.ndarray:
    """Delay 제거 residual을 선형 보간한다. 호출자가 범위를 먼저 검사한다."""

    phase = np.exp(
        2j
        * np.pi
        * frequencies_hz
        * float(bulk_delay_fractional_samples)
        / float(sample_rate)
    )
    residual = transfer * phase
    real = np.interp(query_hz, frequencies_hz, residual.real)
    imag = np.interp(query_hz, frequencies_hz, residual.imag)
    restored = (real + 1j * imag) * np.exp(
        -2j
        * np.pi
        * query_hz
        * float(bulk_delay_fractional_samples)
        / float(sample_rate)
    )
    return np.asarray(restored, dtype=np.complex128)


def build_every_other_tone_holdout_receipt(
    *,
    frequencies_hz: np.ndarray,
    transfer: np.ndarray,
    bulk_delay_fractional_samples: float,
    sample_rate: int,
    subbands_hz: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """각 subband에서 번갈아 숨긴 tone을 이웃 tone만으로 예측한다."""

    frequency = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    measured = np.asarray(transfer, dtype=np.complex128).reshape(-1)
    train_mask = (np.arange(frequency.size, dtype=np.int64) % 2) == 0
    train_f = frequency[train_mask]
    train_h = measured[train_mask]
    rows: list[dict[str, Any]] = []
    for raw_band in subbands_hz:
        lo, hi = (float(value) for value in raw_band)
        holdout_mask = (
            (~train_mask)
            & (frequency >= lo)
            & (frequency <= hi)
            & (frequency > train_f[0])
            & (frequency < train_f[-1])
        )
        query = frequency[holdout_mask]
        target = measured[holdout_mask]
        # 가장 좁은 150--300 Hz cell은 drive별 16 Hz grid에서 every-other
        # holdout이 4~5개다. 8개를 요구하면 물리 grid 자체가 영원히 통과할 수 없다.
        if query.size < MEASURED_BAND_MIN_HOLDOUT_TONES:
            raise ValueError(
                f"measured-band holdout {lo:g}-{hi:g}Hz tone이 {query.size}개뿐입니다"
            )
        estimate = _interpolate_numpy(
            train_f,
            train_h,
            query,
            bulk_delay_fractional_samples=bulk_delay_fractional_samples,
            sample_rate=sample_rate,
        )
        denominator = float(np.linalg.norm(target) * np.linalg.norm(estimate))
        agreement = (
            float(abs(complex(np.vdot(target, estimate))) / denominator)
            if denominator > 0.0
            else 0.0
        )
        target_norm = float(np.linalg.norm(target))
        error = (
            float(np.linalg.norm(estimate - target) / target_norm)
            if target_norm > 0.0
            else math.inf
        )
        rows.append(
            {
                "band_hz": [lo, hi],
                "holdout_tone_count": int(query.size),
                "complex_agreement": agreement,
                "relative_error": error,
                "passed": bool(
                    agreement >= MEASURED_BAND_MIN_HOLDOUT_AGREEMENT
                    and error <= MEASURED_BAND_MAX_HOLDOUT_RELATIVE_ERROR
                ),
            }
        )
    passed = bool(rows and all(bool(row["passed"]) for row in rows))
    return {
        "schema_version": MEASURED_BAND_HOLDOUT_SCHEMA,
        "interpolation_schema": MEASURED_BAND_INTERPOLATION_SCHEMA,
        "minimum_complex_agreement": MEASURED_BAND_MIN_HOLDOUT_AGREEMENT,
        "maximum_relative_error": MEASURED_BAND_MAX_HOLDOUT_RELATIVE_ERROR,
        "rows": rows,
        "passed": passed,
    }


@dataclass(frozen=True)
class MeasuredBandPathData:
    role: str
    sample_rate: int
    frequencies_hz: np.ndarray
    transfer: np.ndarray
    bulk_delay_samples: int
    bulk_delay_fractional_samples: float
    pre_roll_samples: int
    effective_delay_samples: int
    fractional_effective_delay_samples: float
    delay_semantics: str
    valid_band_hz: tuple[float, float]
    control_band_contract_sha256: str
    source_analysis_sha256: str
    plant_evidence_sha256: str
    response_sha256: str
    holdout_receipt: Mapping[str, Any]
    source_path: str

    @classmethod
    def from_arrays(
        cls,
        *,
        role: str,
        sample_rate: int,
        frequencies_hz: object,
        transfer: object,
        bulk_delay_samples: int,
        bulk_delay_fractional_samples: float,
        pre_roll_samples: int,
        effective_delay_samples: int,
        fractional_effective_delay_samples: float,
        delay_semantics: str,
        valid_band_hz: Sequence[float],
        control_band_contract_sha256: str,
        source_analysis_sha256: str,
        plant_evidence_sha256: str,
        subbands_hz: Sequence[Sequence[float]],
        source_path: str,
    ) -> "MeasuredBandPathData":
        if role not in {"primary", "secondary"}:
            raise ValueError(f"measured-band plant role이 잘못됐습니다: {role!r}")
        rate = int(sample_rate)
        delay = float(bulk_delay_fractional_samples)
        bulk_integer = int(bulk_delay_samples)
        pre_roll = int(pre_roll_samples)
        effective = int(effective_delay_samples)
        fractional_effective = float(fractional_effective_delay_samples)
        semantics = str(delay_semantics)
        band_values = tuple(float(value) for value in valid_band_hz)
        if (
            rate <= 0
            or not math.isfinite(delay)
            or not math.isfinite(fractional_effective)
            or delay < 0.0
            or bulk_integer < 0
            or pre_roll < 0
            or effective < 0
            or len(band_values) != 2
            or not 0.0 < band_values[0] < band_values[1] < rate / 2.0
        ):
            raise ValueError("measured-band sample rate/delay/valid band가 잘못됐습니다")
        if semantics != "effective_zeros_before_compact_fir":
            raise ValueError(
                "measured-band delay_semantics가 canonical compact pre-roll 계약과 "
                f"다릅니다: {semantics!r}"
            )
        if bulk_integer != int(round(delay)):
            raise ValueError(
                "measured-band bulk integer/fractional delay가 round 관계가 아닙니다: "
                f"integer={bulk_integer}, fractional={delay}"
            )
        if effective != bulk_integer - pre_roll:
            raise ValueError(
                "measured-band effective delay는 bulk integer - pre-roll이어야 합니다: "
                f"effective={effective}, bulk={bulk_integer}, pre_roll={pre_roll}"
            )
        expected_fractional_effective = delay - float(pre_roll)
        if not math.isclose(
            fractional_effective,
            expected_fractional_effective,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "measured-band fractional effective delay는 bulk fractional - "
                "pre-roll이어야 합니다: "
                f"effective={fractional_effective}, expected={expected_fractional_effective}"
            )
        frequency = _immutable_vector(
            frequencies_hz, dtype=np.dtype(np.float64), label="measured frequencies"
        )
        measured = _immutable_vector(
            transfer, dtype=np.dtype(np.complex128), label="measured transfer"
        )
        if frequency.size != measured.size or frequency.size < 16:
            raise ValueError("measured frequency/transfer 길이가 다르거나 tone이 부족합니다")
        if np.any(np.diff(frequency) <= 0.0):
            raise ValueError("measured frequencies는 strict 증가해야 합니다")
        if frequency[0] > band_values[0] or frequency[-1] < band_values[1]:
            raise ValueError(
                "valid band가 measured tone convex hull 밖입니다: "
                f"hull=[{frequency[0]:g},{frequency[-1]:g}], band={band_values}"
            )
        control_sha = _require_sha256(
            control_band_contract_sha256, label="control-band contract SHA"
        )
        analysis_sha = _require_sha256(
            source_analysis_sha256, label="source analysis SHA"
        )
        evidence_sha = _require_sha256(
            plant_evidence_sha256, label="plant evidence SHA"
        )
        response_sha = _response_sha256(
            role=role,
            sample_rate=rate,
            frequencies_hz=frequency,
            transfer=measured,
            bulk_delay_samples=bulk_integer,
            bulk_delay_fractional_samples=delay,
            pre_roll_samples=pre_roll,
            effective_delay_samples=effective,
            fractional_effective_delay_samples=fractional_effective,
            delay_semantics=semantics,
            valid_band_hz=(band_values[0], band_values[1]),
            control_band_contract_sha256=control_sha,
            source_analysis_sha256=analysis_sha,
            plant_evidence_sha256=evidence_sha,
        )
        receipt = build_every_other_tone_holdout_receipt(
            frequencies_hz=frequency,
            transfer=measured,
            bulk_delay_fractional_samples=delay,
            sample_rate=rate,
            subbands_hz=subbands_hz,
        )
        if not bool(receipt["passed"]):
            failed = [row for row in receipt["rows"] if not row["passed"]]
            raise ValueError(f"measured-band every-other-tone holdout 실패: {failed}")
        return cls(
            role=role,
            sample_rate=rate,
            frequencies_hz=frequency,
            transfer=measured,
            bulk_delay_samples=bulk_integer,
            bulk_delay_fractional_samples=delay,
            pre_roll_samples=pre_roll,
            effective_delay_samples=effective,
            fractional_effective_delay_samples=fractional_effective,
            delay_semantics=semantics,
            valid_band_hz=(band_values[0], band_values[1]),
            control_band_contract_sha256=control_sha,
            source_analysis_sha256=analysis_sha,
            plant_evidence_sha256=evidence_sha,
            response_sha256=response_sha,
            holdout_receipt=receipt,
            source_path=str(source_path),
        )


def load_measured_band_path(
    path: str | Path,
    *,
    role: str,
    valid_band_hz: Sequence[float],
    subbands_hz: Sequence[Sequence[float]],
) -> MeasuredBandPathData:
    """Publisher NPZ의 measured complex tone을 immutable response로 승격한다.

    compact ``fir``의 존재 여부는 보지 않는다. tone/complex response metadata가 없는
    legacy FIR-only NPZ는 필수 key 검사에서 실패한다.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"measured-band plant가 없습니다: {source}")
    with np.load(source, allow_pickle=False) as archive:
        artifact_schema = str(_npz_scalar(archive, "schema_version"))
        if artifact_schema != MEASURED_BAND_SOURCE_ARTIFACT_SCHEMA:
            raise ValueError(
                "measured-band source artifact schema가 canonical publisher와 "
                f"다릅니다: {artifact_schema!r}"
            )
        artifact_role = str(_npz_scalar(archive, "plant_role"))
        if artifact_role != role:
            raise ValueError(
                f"measured-band plant role이 다릅니다: {artifact_role!r} != {role!r}"
            )
        compact_role = str(_npz_scalar(archive, "compact_role"))
        compact_training_eligible = bool(
            _npz_scalar(archive, "compact_training_eligible")
        )
        if compact_role != "diagnostic_only" or compact_training_eligible:
            raise ValueError(
                "measured-band artifact의 compact FIR은 diagnostic_only이고 "
                "training_eligible=false여야 합니다"
            )
        frequency = np.asarray(archive["measured_frequencies_hz"], dtype=np.float64)
        real = np.asarray(archive["measured_transfer_real"], dtype=np.float64)
        imag = np.asarray(archive["measured_transfer_imag"], dtype=np.float64)
        if not (frequency.shape == real.shape == imag.shape):
            raise ValueError("measured-band tone/complex transfer shape가 다릅니다")
        transfer = np.asarray(real + 1j * imag, dtype=np.complex128)
        declared_transfer_sha = _require_sha256(
            _npz_scalar(archive, "aligned_mean_transfer_sha256"),
            label="measured complex transfer SHA",
        )
        actual_transfer_sha = hashlib.sha256(
            transfer.tobytes(order="C")
        ).hexdigest()
        if actual_transfer_sha != declared_transfer_sha:
            raise ValueError(
                "measured-band complex transfer SHA가 array bytes와 다릅니다"
            )
        return MeasuredBandPathData.from_arrays(
            role=role,
            sample_rate=int(_npz_scalar(archive, "sample_rate")),
            frequencies_hz=frequency,
            transfer=transfer,
            bulk_delay_samples=int(_npz_scalar(archive, "bulk_delay_samples")),
            bulk_delay_fractional_samples=float(
                _npz_scalar(archive, "bulk_delay_fractional_samples")
            ),
            pre_roll_samples=int(_npz_scalar(archive, "pre_roll_samples")),
            effective_delay_samples=int(
                _npz_scalar(archive, "effective_delay_samples")
            ),
            fractional_effective_delay_samples=float(
                _npz_scalar(archive, "fractional_effective_delay_samples")
            ),
            delay_semantics=str(_npz_scalar(archive, "delay_semantics")),
            valid_band_hz=valid_band_hz,
            control_band_contract_sha256=str(
                _npz_scalar(archive, "control_band_contract_sha256")
            ),
            source_analysis_sha256=str(
                _npz_scalar(archive, "source_analysis_npz_sha256")
            ),
            plant_evidence_sha256=str(
                _npz_scalar(archive, "broadband_plant_evidence_sha256")
            ),
            subbands_hz=subbands_hz,
            source_path=str(source),
        )


class MeasuredBandPath(nn.Module):
    """측정대역 response 조회 전용 module. 시간영역 convolution은 제공하지 않는다."""

    def __init__(self, data: MeasuredBandPathData, *, extra_delay_samples: int = 0) -> None:
        super().__init__()
        extra = int(extra_delay_samples)
        if extra < 0:
            raise ValueError("measured-band extra delay는 음수일 수 없습니다")
        residual = data.transfer * np.exp(
            2j
            * np.pi
            * data.frequencies_hz
            * data.bulk_delay_fractional_samples
            / data.sample_rate
        )
        self.register_buffer(
            "frequencies_hz", torch.from_numpy(data.frequencies_hz.copy())
        )
        self.register_buffer(
            "residual_real", torch.from_numpy(residual.real.astype(np.float64))
        )
        self.register_buffer(
            "residual_imag", torch.from_numpy(residual.imag.astype(np.float64))
        )
        self.sample_rate = int(data.sample_rate)
        self.bulk_delay_fractional_samples = float(
            data.bulk_delay_fractional_samples
        )
        self.bulk_delay_samples = int(data.bulk_delay_samples)
        self.pre_roll_samples = int(data.pre_roll_samples)
        self.effective_delay_samples = int(data.effective_delay_samples)
        self.fractional_effective_delay_samples = float(
            data.fractional_effective_delay_samples
        )
        self.delay_semantics = str(data.delay_semantics)
        self.extra_delay_samples = extra
        self.valid_band_hz = tuple(data.valid_band_hz)
        self.response_sha256 = str(data.response_sha256)
        self.role = str(data.role)
        self.holdout_receipt = dict(data.holdout_receipt)

    def response_at(
        self,
        query_hz: torch.Tensor,
        *,
        jitter_samples: int = 0,
        gain_db: float = 0.0,
        tilt_db_per_octave: float = 0.0,
    ) -> torch.Tensor:
        if query_hz.ndim != 1 or query_hz.numel() == 0:
            raise ValueError("measured-band query는 비어 있지 않은 1-D여야 합니다")
        if not torch.isfinite(query_hz).all():
            raise ValueError("measured-band query에 NaN/Inf가 있습니다")
        lo = float(query_hz.min().detach().cpu())
        hi = float(query_hz.max().detach().cpu())
        valid_lo, valid_hi = self.valid_band_hz
        measured_lo = float(self.frequencies_hz[0])
        measured_hi = float(self.frequencies_hz[-1])
        if lo < valid_lo - 1.0e-9 or hi > valid_hi + 1.0e-9:
            raise ValueError(
                f"measured-band query가 valid band 밖입니다: [{lo:g},{hi:g}] vs "
                f"[{valid_lo:g},{valid_hi:g}]"
            )
        if lo < measured_lo or hi > measured_hi:
            raise ValueError("measured-band query를 measured tone hull 밖으로 extrapolate할 수 없습니다")

        frequency = self.frequencies_hz.to(device=query_hz.device, dtype=torch.float64)
        query = query_hz.to(dtype=torch.float64)
        upper = torch.searchsorted(frequency, query, right=False).clamp(1, frequency.numel() - 1)
        lower = upper - 1
        f0 = frequency[lower]
        f1 = frequency[upper]
        weight = (query - f0) / (f1 - f0)
        r0 = torch.complex(
            self.residual_real.to(query.device)[lower],
            self.residual_imag.to(query.device)[lower],
        )
        r1 = torch.complex(
            self.residual_real.to(query.device)[upper],
            self.residual_imag.to(query.device)[upper],
        )
        residual = r0 + weight * (r1 - r0)
        total_delay = (
            self.bulk_delay_fractional_samples
            + self.extra_delay_samples
            + int(jitter_samples)
        )
        phase = torch.exp(
            torch.complex(
                torch.zeros_like(query),
                -2.0 * math.pi * query * total_delay / float(self.sample_rate),
            )
        )
        gain = 10.0 ** (float(gain_db) / 20.0)
        if float(tilt_db_per_octave) != 0.0:
            octaves = torch.log2(torch.clamp(query, min=20.0) / 500.0)
            magnitude = gain * torch.pow(
                torch.full_like(query, 10.0),
                float(tilt_db_per_octave) * octaves / 20.0,
            )
        else:
            magnitude = torch.full_like(query, gain)
        return residual * phase * magnitude

    def forward(self, *_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError(
            "MeasuredBandPath는 시간영역 convolution을 제공하지 않습니다 — "
            "response_at()을 zero-padded linear DTFT에서만 사용하세요"
        )


__all__ = [
    "MEASURED_BAND_HOLDOUT_SCHEMA",
    "MEASURED_BAND_INTERPOLATION_SCHEMA",
    "MEASURED_BAND_MAX_HOLDOUT_RELATIVE_ERROR",
    "MEASURED_BAND_MIN_HOLDOUT_TONES",
    "MEASURED_BAND_MIN_HOLDOUT_AGREEMENT",
    "MEASURED_BAND_PATH_SCHEMA_VERSION",
    "MEASURED_BAND_SOURCE_ARTIFACT_SCHEMA",
    "MeasuredBandPath",
    "MeasuredBandPathData",
    "build_every_other_tone_holdout_receipt",
    "load_measured_band_path",
]
