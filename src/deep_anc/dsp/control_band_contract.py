"""저역과 2/4/8 kHz를 함께 다루는 광대역 제어 계약.

기존 strict P/S는 150--1600 Hz에서 유효하며 그대로 보존한다. 이 모듈은 그 자산의
숫자를 8 kHz로 늘려 쓰는 대신, 최종 목표를 별도 역할로 선언하고 광대역 P/S 증거가
없으면 닫히게 한다. 8 kHz는 octave *중심*이므로 제어/식별 상단은
``8000 * sqrt(2)`` Hz다.

이 모듈은 오디오 장치를 열지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, model_validator

from .timing import PlantDelays


CONTROL_BAND_CONTRACT_SCHEMA = "control_band_contract_v2"
CONTROL_BAND_CONTRACT_V3_SCHEMA = "control_band_contract_v3"
STAGE1_STRICT_CONTRACT_ID = "stage1_strict_150_1600_v1"
BROADBAND_POINT_CONTROL_CONTRACT_ID = "broadband_point_control_150_11314_v2"
BROADBAND_FULL_OCTAVE_CONTRACT_ID = "broadband_full_octave_88_11314_v3"

STAGE1_STRICT_SUBBANDS_HZ: tuple[tuple[float, float], ...] = (
    (150.0, 300.0),
    (300.0, 600.0),
    (600.0, 1000.0),
    (1000.0, 1600.0),
)

OCTAVE_8K_UPPER_HZ = 8000.0 * math.sqrt(2.0)
"""8 kHz 중심 octave의 상단. 8 kHz에서 자극을 끝내면 8 kHz octave 증거가 아니다."""
BROADBAND_GLOBAL_CLOCK_MAX_RESIDUAL_SAMPLES = 0.06755189029558945
BROADBAND_MEASURED_INTERPOLATION_MIN_AGREEMENT = 0.995
BROADBAND_MEASURED_INTERPOLATION_MAX_RELATIVE_ERROR = 0.10

BROADBAND_HIGH_SUBBANDS_HZ: tuple[tuple[float, float], ...] = (
    (1600.0, 2000.0 * math.sqrt(2.0)),
    (2000.0 * math.sqrt(2.0), 4000.0 * math.sqrt(2.0)),
    (4000.0 * math.sqrt(2.0), OCTAVE_8K_UPPER_HZ),
)

BROADBAND_POINT_CONTROL_SUBBANDS_HZ = (
    *STAGE1_STRICT_SUBBANDS_HZ,
    *BROADBAND_HIGH_SUBBANDS_HZ,
)

# v3는 v2의 150 Hz 하단을 숫자만 바꿔 재발행하지 않는다. 아래 세 집합은
# 각각 물리 식별, equal-weight 학습 목적, Stage-1 보존 guard라는 서로 다른
# 의미를 갖고 별도 필드로 직렬화된다. 명시된 decimal grid 자체가 계약이다.
BROADBAND_V3_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ: tuple[
    tuple[float, float], ...
] = (
    (88.3883476483, 150.0),
    (150.0, 300.0),
    (300.0, 600.0),
    (600.0, 1000.0),
    (1000.0, 1600.0),
    (1600.0, 2828.4271247462),
    (2828.4271247462, 5656.8542494924),
    (5656.8542494924, 11313.7084989848),
)

BROADBAND_V3_OCTAVE_OBJECTIVE_CENTERS_HZ: tuple[float, ...] = (
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    8000.0,
)

BROADBAND_V3_OCTAVE_OBJECTIVE_BANDS_HZ: tuple[
    tuple[float, float], ...
] = (
    (88.3883476483, 176.7766952966),
    (176.7766952966, 353.5533905933),
    (353.5533905933, 707.1067811865),
    (707.1067811865, 1414.2135623731),
    (1414.2135623731, 2828.4271247462),
    (2828.4271247462, 5656.8542494924),
    (5656.8542494924, 11313.7084989848),
)

BROADBAND_V3_STAGE1_LOW_GUARD_SUBBANDS_HZ = STAGE1_STRICT_SUBBANDS_HZ
BROADBAND_V3_EXCITATION_LOWER_MAX_HZ = 80.0
BROADBAND_V3_EXCITATION_UPPER_MIN_HZ = 11313.7084989848

BROADBAND_OCTAVE_CENTERS_HZ = (
    125.0,
    250.0,
    500.0,
    1000.0,
    1600.0,
    2000.0,
    4000.0,
    8000.0,
)

# 한 번의 dense 60--8k 자극은 clock witness를 0/64로 만들었다. 겹치는 panel마다
# sparse clock pilot과 공통 phase anchor를 두는 측정기에서 사용할 대역 계획이다.
BROADBAND_MEASUREMENT_PANELS_HZ: tuple[tuple[float, float], ...] = (
    (100.0, 1800.0),
    (1400.0, 3200.0),
    (2800.0, 6000.0),
    (5400.0, 8500.0),
    (7800.0, 11400.0),
)

REQUIRED_SOURCE_FAMILIES = ("speech", "music", "environment", "machine")


def phase_error_degrees(samples: float, frequency_hz: float, sample_rate: float) -> float:
    """주어진 sample 오차가 만드는 위상 오차를 degree로 반환한다."""

    if not all(math.isfinite(float(value)) for value in (samples, frequency_hz, sample_rate)):
        raise ValueError("sample/주파수/sample rate는 유한해야 합니다")
    if float(frequency_hz) < 0.0 or float(sample_rate) <= 0.0:
        raise ValueError("frequency는 0 이상, sample rate는 양수여야 합니다")
    return float(360.0 * float(frequency_hz) * float(samples) / float(sample_rate))


def max_timing_error_samples_for_attenuation(
    attenuation_db: float,
    frequency_hz: float,
    sample_rate: float,
) -> float:
    """동일 gain 두 파형의 위상 오차만 있을 때 허용되는 최대 timing 오차.

    잔차비 ``r = 2*sin(|phi|/2)``와 ``attenuation=-20log10(r)``의 역이다.
    이는 모델 성능 보장이 아니라 측정/런타임의 위상 해상도 예산이다.
    """

    attenuation = float(attenuation_db)
    frequency = float(frequency_hz)
    rate = float(sample_rate)
    if not all(math.isfinite(value) for value in (attenuation, frequency, rate)):
        raise ValueError("감쇠/주파수/sample rate는 유한해야 합니다")
    if attenuation <= 0.0 or frequency <= 0.0 or rate <= 0.0:
        raise ValueError("감쇠/주파수/sample rate는 양수여야 합니다")
    residual = 10.0 ** (-attenuation / 20.0)
    phase_radians = 2.0 * math.asin(min(1.0, residual / 2.0))
    return float(phase_radians * rate / (2.0 * math.pi * frequency))


def _same_bands(
    actual: Sequence[Sequence[float]], expected: Sequence[Sequence[float]]
) -> bool:
    try:
        a = tuple(tuple(float(value) for value in band) for band in actual)
        e = tuple(tuple(float(value) for value in band) for band in expected)
    except (TypeError, ValueError):
        return False
    if len(a) != len(e) or any(len(band) != 2 for band in a):
        return False
    return all(
        math.isclose(av, ev, rel_tol=0.0, abs_tol=1.0e-9)
        for actual_band, expected_band in zip(a, e, strict=True)
        for av, ev in zip(actual_band, expected_band, strict=True)
    )


def _validate_contiguous_subbands(bands: Sequence[Sequence[float]]) -> None:
    parsed = tuple(tuple(float(value) for value in band) for band in bands)
    if not parsed:
        raise ValueError("point-control subband가 비었습니다")
    for index, band in enumerate(parsed):
        if len(band) != 2 or not all(math.isfinite(value) for value in band):
            raise ValueError(f"subband #{index}가 유효한 [lo, hi]가 아닙니다: {band!r}")
        if not 0.0 <= band[0] < band[1]:
            raise ValueError(f"subband #{index} 순서가 잘못됐습니다: {band!r}")
        if index and not math.isclose(
            parsed[index - 1][1], band[0], rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise ValueError(
                "point-control subband에 gap/overlap이 있습니다: "
                f"{parsed[index - 1]!r} -> {band!r}"
            )


def _windows_cover(
    windows: Sequence[Sequence[float]], target: Sequence[float]
) -> bool:
    lo_target, hi_target = (float(value) for value in target)
    cursor = lo_target
    for raw in sorted((tuple(float(value) for value in item) for item in windows)):
        if len(raw) != 2 or raw[0] >= raw[1]:
            return False
        if raw[1] <= cursor:
            continue
        if raw[0] > cursor + 1.0e-9:
            return False
        cursor = max(cursor, raw[1])
        if cursor >= hi_target:
            return True
    return False


def _exact_bands(
    actual: Sequence[Sequence[float]], expected: Sequence[Sequence[float]]
) -> bool:
    """v3 immutable decimal grid의 exact float equality를 검사한다."""

    try:
        parsed = tuple(tuple(float(value) for value in band) for band in actual)
    except (TypeError, ValueError):
        return False
    return parsed == tuple(tuple(float(value) for value in band) for band in expected)


class BroadbandFullOctaveContractV3(BaseModel):
    """125 Hz octave 전체부터 8 kHz octave 전체까지의 별도 최종 계약.

    기존 :class:`ControlBandContract`에 default field를 추가하지 않는다. 그래야
    Stage-1 v1과 broadband v2의 JSON/digest가 byte-for-byte 유지된다.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["control_band_contract_v3"] = (
        CONTROL_BAND_CONTRACT_V3_SCHEMA
    )
    contract_id: Literal["broadband_full_octave_88_11314_v3"] = (
        BROADBAND_FULL_OCTAVE_CONTRACT_ID
    )
    role: Literal["broadband_full_octave"] = "broadband_full_octave"
    sample_rate: int = 48_000
    physical_identification_subbands_hz: tuple[tuple[float, float], ...]
    octave_objective_centers_hz: tuple[float, ...]
    equal_weight_octave_objective_bands_hz: tuple[tuple[float, float], ...]
    stage1_low_guard_subbands_hz: tuple[tuple[float, float], ...]
    required_excitation_lower_hz: float
    required_excitation_upper_hz: float
    source_families: tuple[str, ...] = REQUIRED_SOURCE_FAMILIES
    physical_identification_semantics: Literal[
        "plant_identification_and_coverage_not_loss_weighting"
    ] = "plant_identification_and_coverage_not_loss_weighting"
    octave_objective_semantics: Literal[
        "seven_equal_weight_target_energy_normalized_octaves"
    ] = "seven_equal_weight_target_energy_normalized_octaves"
    stage1_low_guard_semantics: Literal[
        "separate_positive_attenuation_guard_not_octave_objective"
    ] = "separate_positive_attenuation_guard_not_octave_objective"
    legacy_v2_automatic_promotion_allowed: Literal[False] = False
    requires_exact_v3_contract_sha256: Literal[True] = True

    @model_validator(mode="after")
    def _validate_v3(self) -> "BroadbandFullOctaveContractV3":
        if self.sample_rate != 48_000:
            raise ValueError("canonical v3 control-band 계약은 48 kHz여야 합니다")
        if tuple(self.source_families) != REQUIRED_SOURCE_FAMILIES:
            raise ValueError("v3도 speech/music/environment/machine 네 family가 필수입니다")

        _validate_contiguous_subbands(self.physical_identification_subbands_hz)
        _validate_contiguous_subbands(self.equal_weight_octave_objective_bands_hz)
        _validate_contiguous_subbands(self.stage1_low_guard_subbands_hz)
        if not _exact_bands(
            self.physical_identification_subbands_hz,
            BROADBAND_V3_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ,
        ):
            raise ValueError("v3 physical identification 8구간을 변경할 수 없습니다")
        if tuple(self.octave_objective_centers_hz) != (
            BROADBAND_V3_OCTAVE_OBJECTIVE_CENTERS_HZ
        ):
            raise ValueError("v3 equal-weight octave center 7개를 변경할 수 없습니다")
        if not _exact_bands(
            self.equal_weight_octave_objective_bands_hz,
            BROADBAND_V3_OCTAVE_OBJECTIVE_BANDS_HZ,
        ):
            raise ValueError("v3 equal-weight octave objective band를 변경할 수 없습니다")
        if not _exact_bands(
            self.stage1_low_guard_subbands_hz,
            BROADBAND_V3_STAGE1_LOW_GUARD_SUBBANDS_HZ,
        ):
            raise ValueError("v3 Stage-1 low guard 4구간을 변경할 수 없습니다")

        # 목적 octave가 실제 center를 갖는지 독립적으로 다시 계산한다. 식별
        # 8구간이나 Stage-1 guard를 이 필드에 복사하면 위 exact gate와 여기서 막힌다.
        for center, (lower, upper) in zip(
            self.octave_objective_centers_hz,
            self.equal_weight_octave_objective_bands_hz,
            strict=True,
        ):
            if not math.isclose(
                math.sqrt(lower * upper), center, rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise ValueError("v3 objective band의 기하 중심이 octave center와 다릅니다")

        lower = float(self.required_excitation_lower_hz)
        upper = float(self.required_excitation_upper_hz)
        if not math.isfinite(lower) or not 0.0 < lower <= (
            BROADBAND_V3_EXCITATION_LOWER_MAX_HZ
        ):
            raise ValueError("v3 excitation lower는 0 Hz 초과 80 Hz 이하여야 합니다")
        if (
            not math.isfinite(upper)
            or upper < BROADBAND_V3_EXCITATION_UPPER_MIN_HZ
            or upper > self.sample_rate / 2.0
        ):
            raise ValueError(
                "v3 excitation upper는 11,313.7084989848 Hz 이상 Nyquist 이하여야 합니다"
            )
        if lower > self.physical_identification_subbands_hz[0][0]:
            raise ValueError("v3 excitation이 physical identification 하단을 덮지 않습니다")
        if upper < self.physical_identification_subbands_hz[-1][1]:
            raise ValueError("v3 excitation이 physical identification 상단을 덮지 않습니다")
        return self

    @classmethod
    def canonical(cls) -> "BroadbandFullOctaveContractV3":
        return cls(
            physical_identification_subbands_hz=(
                BROADBAND_V3_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ
            ),
            octave_objective_centers_hz=(
                BROADBAND_V3_OCTAVE_OBJECTIVE_CENTERS_HZ
            ),
            equal_weight_octave_objective_bands_hz=(
                BROADBAND_V3_OCTAVE_OBJECTIVE_BANDS_HZ
            ),
            stage1_low_guard_subbands_hz=(
                BROADBAND_V3_STAGE1_LOW_GUARD_SUBBANDS_HZ
            ),
            required_excitation_lower_hz=BROADBAND_V3_EXCITATION_LOWER_MAX_HZ,
            required_excitation_upper_hz=BROADBAND_V3_EXCITATION_UPPER_MIN_HZ,
        )

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class ControlBandContract(BaseModel):
    """Stage-1 strict와 최종 광대역 목표를 섞지 않는 immutable 계약."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["control_band_contract_v2"] = CONTROL_BAND_CONTRACT_SCHEMA
    contract_id: Literal[
        "stage1_strict_150_1600_v1",
        "broadband_point_control_150_11314_v2",
    ]
    role: Literal["stage1_strict", "broadband_point_control"]
    sample_rate: int = 48_000
    point_control_target_hz: tuple[float, float]
    point_control_subbands_hz: tuple[tuple[float, float], ...]
    octave_centers_hz: tuple[float, ...]
    required_excitation_upper_hz: float
    measurement_panels_hz: tuple[tuple[float, float], ...]
    source_families: tuple[str, ...] = REQUIRED_SOURCE_FAMILIES
    low_band_requirement: Literal["positive_attenuation"] = "positive_attenuation"
    high_band_requirement: Literal["do_no_harm_only", "matched_fxlms_superiority"]
    matched_fxlms_required: bool
    spatial_validation_required: bool
    minimum_spatial_error_positions: int
    measurement_resolution_attenuation_db: float = 10.0

    @model_validator(mode="after")
    def _validate_contract(self) -> "ControlBandContract":
        if self.sample_rate != 48_000:
            raise ValueError("canonical control-band 계약은 48 kHz여야 합니다")
        if tuple(self.source_families) != REQUIRED_SOURCE_FAMILIES:
            raise ValueError("speech/music/environment/machine 네 family를 모두 유지해야 합니다")
        _validate_contiguous_subbands(self.point_control_subbands_hz)
        target = tuple(float(value) for value in self.point_control_target_hz)
        first = self.point_control_subbands_hz[0][0]
        last = self.point_control_subbands_hz[-1][1]
        if not _same_bands((target,), ((first, last),)):
            raise ValueError("point-control target과 subband union이 다릅니다")
        if not _windows_cover(self.measurement_panels_hz, target):
            raise ValueError("measurement panel이 point-control target 전체를 덮지 않습니다")
        if self.required_excitation_upper_hz > self.sample_rate / 2.0:
            raise ValueError("required excitation upper가 Nyquist를 넘습니다")
        if not math.isfinite(self.measurement_resolution_attenuation_db) or (
            self.measurement_resolution_attenuation_db <= 0.0
        ):
            raise ValueError("measurement resolution attenuation은 양수여야 합니다")

        if self.role == "stage1_strict":
            if self.contract_id != STAGE1_STRICT_CONTRACT_ID:
                raise ValueError("stage1 role과 contract id가 다릅니다")
            if not _same_bands(self.point_control_subbands_hz, STAGE1_STRICT_SUBBANDS_HZ):
                raise ValueError("stage1 strict subband를 바꿀 수 없습니다")
            if self.matched_fxlms_required or self.spatial_validation_required:
                raise ValueError("stage1 strict를 최종 FxLMS/공간 증거로 승격할 수 없습니다")
            if self.high_band_requirement != "do_no_harm_only":
                raise ValueError("stage1의 1.6kHz 밖은 do-no-harm 진단 역할뿐입니다")
            if self.minimum_spatial_error_positions != 1:
                raise ValueError("stage1 strict는 단일 ERR 위치 계약입니다")
        else:
            if self.contract_id != BROADBAND_POINT_CONTROL_CONTRACT_ID:
                raise ValueError("broadband role과 contract id가 다릅니다")
            if not _same_bands(
                self.point_control_subbands_hz, BROADBAND_POINT_CONTROL_SUBBANDS_HZ
            ):
                raise ValueError("광대역 point-control subband를 축소/변경할 수 없습니다")
            if not math.isclose(
                self.required_excitation_upper_hz,
                OCTAVE_8K_UPPER_HZ,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError("8 kHz octave는 excitation이 11.314 kHz까지 있어야 합니다")
            if not self.matched_fxlms_required:
                raise ValueError("광대역 최종 계약은 matched FxLMS 비교가 필수입니다")
            if self.high_band_requirement != "matched_fxlms_superiority":
                raise ValueError("광대역 고역은 matched FxLMS 우위가 필수입니다")
            if not self.spatial_validation_required or self.minimum_spatial_error_positions < 5:
                raise ValueError("고차모드 quiet-zone은 최소 5개 ERR 위치 검증이 필요합니다")
            if not math.isclose(
                self.measurement_resolution_attenuation_db,
                20.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("광대역 timing/phase 측정 해상도는 20 dB-grade여야 합니다")
            for center in (2000.0, 4000.0, 8000.0):
                if center not in self.octave_centers_hz:
                    raise ValueError(f"광대역 octave center가 빠졌습니다: {center:g}Hz")
        return self

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def stage1_strict(cls) -> "ControlBandContract":
        return cls(
            contract_id=STAGE1_STRICT_CONTRACT_ID,
            role="stage1_strict",
            point_control_target_hz=(150.0, 1600.0),
            point_control_subbands_hz=STAGE1_STRICT_SUBBANDS_HZ,
            octave_centers_hz=(125.0, 250.0, 500.0, 1000.0, 1600.0),
            required_excitation_upper_hz=1600.0,
            measurement_panels_hz=((150.0, 1600.0),),
            high_band_requirement="do_no_harm_only",
            matched_fxlms_required=False,
            spatial_validation_required=False,
            minimum_spatial_error_positions=1,
        )

    @classmethod
    def broadband_point_control(cls) -> "ControlBandContract":
        return cls(
            contract_id=BROADBAND_POINT_CONTROL_CONTRACT_ID,
            role="broadband_point_control",
            point_control_target_hz=(150.0, OCTAVE_8K_UPPER_HZ),
            point_control_subbands_hz=BROADBAND_POINT_CONTROL_SUBBANDS_HZ,
            octave_centers_hz=BROADBAND_OCTAVE_CENTERS_HZ,
            required_excitation_upper_hz=OCTAVE_8K_UPPER_HZ,
            measurement_panels_hz=BROADBAND_MEASUREMENT_PANELS_HZ,
            high_band_requirement="matched_fxlms_superiority",
            matched_fxlms_required=True,
            spatial_validation_required=True,
            minimum_spatial_error_positions=5,
            measurement_resolution_attenuation_db=20.0,
        )


ResolvedControlBandContract = ControlBandContract | BroadbandFullOctaveContractV3


def resolve_control_band_contract(
    payload: Mapping[str, Any] | ResolvedControlBandContract,
) -> ResolvedControlBandContract:
    """schema를 명시적으로 분기하며 v2를 v3로 자동 승격하지 않는다."""

    if isinstance(payload, (ControlBandContract, BroadbandFullOctaveContractV3)):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError("control-band contract payload는 mapping이어야 합니다")
    schema = payload.get("schema_version")
    if schema == CONTROL_BAND_CONTRACT_SCHEMA:
        return ControlBandContract.model_validate(payload)
    if schema == CONTROL_BAND_CONTRACT_V3_SCHEMA:
        return BroadbandFullOctaveContractV3.model_validate(payload)
    raise ValueError(f"지원하지 않는 control-band contract schema입니다: {schema!r}")


class BroadbandPlantEvidence(BaseModel):
    """새 multi-panel P/S 분석기가 발행해야 하는 최소 raw-derived 증거."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["broadband_interleaved_plant_evidence_v4"] = (
        "broadband_interleaved_plant_evidence_v4"
    )
    control_band_contract_sha256: str
    primary_capture_id: str
    secondary_capture_id: str
    primary_raw_sha256: str
    secondary_raw_sha256: str
    primary_analysis_sha256: str
    secondary_analysis_sha256: str
    exact_plan_file_sha256: str
    exact_plan_payload_sha256: str
    exact_plan_pcm_sha256: str
    measurement_level_evidence_sha256: str
    fresh_meter_raw_sha256: str
    fresh_meter_receipt_sha256: str
    timing_marker_pcm_sha256: str
    fixed_clock_pilot_sha256: str
    submitted_pilot_validation_sha256: str
    submitted_pilot_cross_channel_null_sha256: str
    submitted_pilot_cross_channel_max_absolute: float
    submitted_pilot_cross_channel_max_ratio: float
    global_clock_input_domain: str
    global_clock_map_sha256: str
    global_clock_slope_samples_per_sample: float
    global_clock_intercept_samples: float
    global_clock_max_residual_samples: float
    clock_trajectory_agreement_samples: float
    transition_anchor_valid_counts: tuple[int, ...]
    callback_timing_valid: bool
    callback_sample_slip_count: int
    panel_clock_offsets_samples: tuple[float, ...]
    applied_per_drive_phase_repair_samples: tuple[float, ...]
    primary_marker_delay_samples: float
    secondary_marker_delay_samples: float
    primary_marker_branch_width_samples: float
    secondary_marker_branch_width_samples: float
    primary_marker_alias_candidate_count: int
    secondary_marker_alias_candidate_count: int
    primary_bulk_delay_fractional_samples: float
    secondary_bulk_delay_fractional_samples: float
    primary_bulk_delay_samples: int
    secondary_bulk_delay_samples: int
    primary_effective_delay_samples: int
    secondary_effective_delay_samples: int
    pre_roll_samples: int
    handoff_extra_samples: int
    derived_lead_samples: int
    panel_primary_minus_secondary_bulk_delay_samples: tuple[float, ...]
    panel_relative_delay_deviation_samples: tuple[float, ...]
    sample_rate: int
    block_size: int
    latency: str
    observed_submitted_pcm: bool
    excitation_panels_hz: tuple[tuple[float, float], ...]
    verified_subbands_hz: tuple[tuple[float, float], ...]
    primary_consistency: tuple[float, ...]
    secondary_consistency: tuple[float, ...]
    clock_valid_repeats: tuple[int, ...]
    clock_min_adjacent_score_observed: tuple[float, ...]
    relative_phase_jitter_samples: tuple[float, ...]
    separation_crosscheck_agreement: tuple[float, ...]
    separation_crosscheck_relative_error: tuple[float, ...]
    measured_interpolation_agreement: tuple[float, ...]
    measured_interpolation_relative_error: tuple[float, ...]
    primary_compact_role: str
    secondary_compact_role: str
    primary_compact_training_eligible: bool
    secondary_compact_training_eligible: bool
    primary_compact_identifiability_sha256: str
    secondary_compact_identifiability_sha256: str
    compact_roundtrip_agreement: tuple[float, ...]
    compact_roundtrip_relative_error: tuple[float, ...]
    xrun_count: int
    clip_count: int

    @model_validator(mode="after")
    def _validate_evidence_shape(self) -> "BroadbandPlantEvidence":
        for field in (
            "control_band_contract_sha256",
            "primary_raw_sha256",
            "secondary_raw_sha256",
            "primary_analysis_sha256",
            "secondary_analysis_sha256",
            "exact_plan_file_sha256",
            "exact_plan_payload_sha256",
            "exact_plan_pcm_sha256",
            "measurement_level_evidence_sha256",
            "fresh_meter_raw_sha256",
            "fresh_meter_receipt_sha256",
            "timing_marker_pcm_sha256",
            "fixed_clock_pilot_sha256",
            "submitted_pilot_validation_sha256",
            "submitted_pilot_cross_channel_null_sha256",
            "global_clock_map_sha256",
            "primary_compact_identifiability_sha256",
            "secondary_compact_identifiability_sha256",
        ):
            value = str(getattr(self, field))
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{field}는 lowercase SHA-256이어야 합니다")
        if not self.primary_capture_id.strip() or not self.secondary_capture_id.strip():
            raise ValueError("P/S capture id가 비었습니다")
        if self.global_clock_input_domain != (
            "actual_submitted_int16_period_spectrum_not_intended_float"
        ):
            raise ValueError(
                "global clock input domain은 실제 submitted int16 spectrum이어야 합니다"
            )
        delay_scalars = (
            self.primary_bulk_delay_fractional_samples,
            self.secondary_bulk_delay_fractional_samples,
            *self.panel_primary_minus_secondary_bulk_delay_samples,
            *self.panel_relative_delay_deviation_samples,
            self.global_clock_slope_samples_per_sample,
            self.global_clock_intercept_samples,
            self.global_clock_max_residual_samples,
            self.clock_trajectory_agreement_samples,
            self.submitted_pilot_cross_channel_max_absolute,
            self.submitted_pilot_cross_channel_max_ratio,
            *self.panel_clock_offsets_samples,
            *self.applied_per_drive_phase_repair_samples,
            self.primary_marker_delay_samples,
            self.secondary_marker_delay_samples,
            self.primary_marker_branch_width_samples,
            self.secondary_marker_branch_width_samples,
        )
        if not all(math.isfinite(float(value)) for value in delay_scalars):
            raise ValueError("광대역 P/S delay evidence는 유한값이어야 합니다")
        for field in (
            "primary_bulk_delay_samples",
            "secondary_bulk_delay_samples",
            "primary_effective_delay_samples",
            "secondary_effective_delay_samples",
            "pre_roll_samples",
            "handoff_extra_samples",
            "derived_lead_samples",
        ):
            if int(getattr(self, field)) < 0:
                raise ValueError(f"{field}는 0 이상이어야 합니다")
        if any(
            float(value) < 0.0
            for value in self.panel_relative_delay_deviation_samples
        ):
            raise ValueError("panel relative-delay deviation은 0 이상이어야 합니다")
        if self.xrun_count < 0 or self.clip_count < 0:
            raise ValueError("xrun/clip count는 0 이상이어야 합니다")
        if self.callback_sample_slip_count < 0:
            raise ValueError("callback sample slip count는 0 이상이어야 합니다")
        if any(int(value) < 0 for value in self.transition_anchor_valid_counts):
            raise ValueError("transition anchor valid count는 0 이상이어야 합니다")
        if (
            self.primary_marker_alias_candidate_count < 0
            or self.secondary_marker_alias_candidate_count < 0
        ):
            raise ValueError("marker alias candidate count는 0 이상이어야 합니다")
        return self


class BroadbandPlantAudit(BaseModel):
    """광대역 plant 증거 판정. 숫자가 없으면 PASS를 만들지 않는다."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["PASS", "BLOCKED"]
    reasons: tuple[str, ...]
    contract_sha256: str
    max_timing_error_samples_by_subband: tuple[float, ...]

    @property
    def ok(self) -> bool:
        return self.status == "PASS"

    def raise_if_blocked(self) -> "BroadbandPlantAudit":
        if not self.ok:
            raise ValueError("광대역 P/S 증거가 BLOCKED입니다: " + "; ".join(self.reasons))
        return self


def audit_broadband_plant_evidence(
    contract: ControlBandContract,
    evidence: BroadbandPlantEvidence,
) -> BroadbandPlantAudit:
    """모든 highband subband가 독립적으로 통과할 때만 PASS한다."""

    reasons: list[str] = []
    if contract.role != "broadband_point_control":
        reasons.append("stage1 strict 계약은 광대역 성능 증거가 아닙니다")
    if evidence.control_band_contract_sha256 != contract.digest():
        reasons.append("control-band contract SHA가 다릅니다")
    if evidence.primary_capture_id != evidence.secondary_capture_id:
        reasons.append("P/S가 같은 capture가 아닙니다")
    if evidence.primary_raw_sha256 != evidence.secondary_raw_sha256:
        reasons.append("P/S가 같은 immutable raw를 가리키지 않습니다")
    if evidence.primary_analysis_sha256 != evidence.secondary_analysis_sha256:
        reasons.append("P/S가 같은 immutable analysis를 가리키지 않습니다")
    if (evidence.sample_rate, evidence.block_size, evidence.latency) != (48_000, 256, "low"):
        reasons.append("광대역 P/S는 48kHz/256/low여야 합니다")
    if not evidence.observed_submitted_pcm:
        reasons.append("실제 제출 int16 PCM provenance가 없습니다")
    if not 0.0 <= float(evidence.submitted_pilot_cross_channel_max_absolute) <= 1.0e-8:
        reasons.append("submitted pilot 반대 channel absolute null이 깨졌습니다")
    if not 0.0 <= float(evidence.submitted_pilot_cross_channel_max_ratio) <= 1.0e-12:
        reasons.append("submitted pilot 반대 channel 상대 null이 깨졌습니다")
    if not _same_bands(evidence.excitation_panels_hz, contract.measurement_panels_hz):
        reasons.append("excitation panel이 계약과 다릅니다")
    if not _same_bands(evidence.verified_subbands_hz, contract.point_control_subbands_hz):
        reasons.append("검증 subband가 광대역 point-control 계약과 다릅니다")
    if evidence.xrun_count != 0 or evidence.clip_count != 0:
        reasons.append("xrun/clip이 0이 아닙니다")
    if not evidence.callback_timing_valid or evidence.callback_sample_slip_count != 0:
        reasons.append("callback time_info가 유효하지 않거나 sample slip이 있습니다")
    if len(evidence.transition_anchor_valid_counts) != 4 or any(
        int(value) != 8 for value in evidence.transition_anchor_valid_counts
    ):
        reasons.append("transition anchor 4개가 각각 8 adjacent-valid여야 합니다")
    if not 0.0 <= float(evidence.global_clock_max_residual_samples) <= (
        BROADBAND_GLOBAL_CLOCK_MAX_RESIDUAL_SAMPLES
    ):
        reasons.append("global clock residual이 11.314kHz 20dB timing 예산을 넘습니다")
    if not 0.0 <= float(evidence.clock_trajectory_agreement_samples) <= (
        BROADBAND_GLOBAL_CLOCK_MAX_RESIDUAL_SAMPLES
    ):
        reasons.append("P/S clock trajectory agreement가 timing 예산을 넘습니다")
    if len(evidence.panel_clock_offsets_samples) != len(contract.measurement_panels_hz):
        reasons.append("panel global-clock offset vector 길이가 다릅니다")
    if len(evidence.applied_per_drive_phase_repair_samples) != 2 * len(
        contract.measurement_panels_hz
    ) or any(
        float(value) != 0.0
        for value in evidence.applied_per_drive_phase_repair_samples
    ):
        reasons.append("highband 결과 기반 per-drive phase repair는 0이어야 합니다")
    if (
        not 0.0 < float(evidence.primary_marker_branch_width_samples) < 3_000.0
        or not 0.0 < float(evidence.secondary_marker_branch_width_samples) < 3_000.0
        or int(evidence.primary_marker_alias_candidate_count) != 1
        or int(evidence.secondary_marker_alias_candidate_count) != 1
    ):
        reasons.append("timing marker가 3000-sample alias branch 하나를 고르지 못했습니다")

    # P/S artifact에 저장할 effective delay와 runtime handoff를 같은
    # PlantDelays 규약으로 다시 풀어 lead를 검증한다. bulk, compact
    # pre-roll, effective delay를 섞어 쓰면 고역에서 샘플 단위 위상
    # 오차가 생기므로 수동 숫자 계산을 허용하지 않는다.
    if int(round(evidence.primary_bulk_delay_fractional_samples)) != int(
        evidence.primary_bulk_delay_samples
    ) or int(round(evidence.secondary_bulk_delay_fractional_samples)) != int(
        evidence.secondary_bulk_delay_samples
    ):
        reasons.append("P/S fractional bulk delay와 integer bulk delay가 다릅니다")
    if (
        evidence.primary_bulk_delay_samples - evidence.pre_roll_samples
        != evidence.primary_effective_delay_samples
        or evidence.secondary_bulk_delay_samples - evidence.pre_roll_samples
        != evidence.secondary_effective_delay_samples
    ):
        reasons.append("P/S bulk−pre-roll이 effective delay와 다릅니다")
    try:
        plant_delays = PlantDelays(
            primary_delay_samples=evidence.primary_effective_delay_samples,
            secondary_delay_samples=evidence.secondary_effective_delay_samples,
            handoff_samples=evidence.handoff_extra_samples,
            sample_rate=evidence.sample_rate,
        )
        lead = plant_delays.lead()
        if lead.is_clamped:
            reasons.append("광대역 digital-reference lead가 0으로 clamp됩니다")
        if int(lead) != int(evidence.derived_lead_samples):
            reasons.append("PlantDelays.lead()와 저장된 broadband lead가 다릅니다")
    except ValueError as exc:
        reasons.append(f"광대역 PlantDelays evidence가 잘못됐습니다: {exc}")

    n_bands = len(contract.point_control_subbands_hz)
    band_vectors = {
        "primary consistency": evidence.primary_consistency,
        "secondary consistency": evidence.secondary_consistency,
        "relative phase jitter": evidence.relative_phase_jitter_samples,
        "separation agreement": evidence.separation_crosscheck_agreement,
        "separation relative error": evidence.separation_crosscheck_relative_error,
        "measured interpolation agreement": evidence.measured_interpolation_agreement,
        "measured interpolation relative error": (
            evidence.measured_interpolation_relative_error
        ),
        "compact agreement": evidence.compact_roundtrip_agreement,
        "compact relative error": evidence.compact_roundtrip_relative_error,
    }
    for label, values in band_vectors.items():
        if len(values) != n_bands or not all(math.isfinite(float(value)) for value in values):
            reasons.append(f"{label} vector가 {n_bands}개 유한값이 아닙니다")

    n_panels = len(contract.measurement_panels_hz)
    if len(evidence.clock_valid_repeats) != n_panels or any(
        int(value) < 8 for value in evidence.clock_valid_repeats
    ):
        reasons.append("각 panel의 clock valid repeat가 8 이상이어야 합니다")
    if len(evidence.clock_min_adjacent_score_observed) != n_panels or any(
        not math.isfinite(float(value)) or float(value) < 0.995
        for value in evidence.clock_min_adjacent_score_observed
    ):
        reasons.append("각 panel의 clock adjacent score가 0.995 이상이어야 합니다")
    panel_relative = evidence.panel_primary_minus_secondary_bulk_delay_samples
    panel_deviation = evidence.panel_relative_delay_deviation_samples
    if len(panel_relative) != n_panels or not all(
        math.isfinite(float(value)) for value in panel_relative
    ):
        reasons.append(f"panel P−S fractional delay vector가 {n_panels}개 유한값이 아닙니다")
    if len(panel_deviation) != n_panels or not all(
        math.isfinite(float(value)) and float(value) >= 0.0
        for value in panel_deviation
    ):
        reasons.append(f"panel P−S delay deviation vector가 {n_panels}개 유한값이 아닙니다")
    final_fractional_relative = (
        float(evidence.primary_bulk_delay_fractional_samples)
        - float(evidence.secondary_bulk_delay_fractional_samples)
    )
    if len(panel_relative) == n_panels and len(panel_deviation) == n_panels:
        expected_deviation = tuple(
            abs(float(value) - final_fractional_relative)
            for value in panel_relative
        )
        if any(
            not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-9)
            for actual, expected in zip(
                panel_deviation, expected_deviation, strict=True
            )
        ):
            reasons.append("panel P−S delay deviation이 final broadband delay에서 유도되지 않았습니다")
        panel_phase_limits = tuple(
            max_timing_error_samples_for_attenuation(
                contract.measurement_resolution_attenuation_db,
                panel[1],
                contract.sample_rate,
            )
            for panel in contract.measurement_panels_hz
        )
        if any(
            float(actual) > limit
            for actual, limit in zip(
                expected_deviation, panel_phase_limits, strict=True
            )
        ):
            reasons.append("panel간 P−S relative delay가 20dB 측정 해상도 예산을 넘습니다")

    phase_limits = tuple(
        max_timing_error_samples_for_attenuation(
            contract.measurement_resolution_attenuation_db,
            band[1],
            contract.sample_rate,
        )
        for band in contract.point_control_subbands_hz
    )
    if len(evidence.relative_phase_jitter_samples) == n_bands and any(
        float(actual) > limit
        for actual, limit in zip(
            evidence.relative_phase_jitter_samples, phase_limits, strict=True
        )
    ):
        reasons.append("subband별 fractional phase jitter가 20dB 측정 해상도 예산을 넘습니다")
    if len(evidence.primary_consistency) == n_bands and any(
        float(value) < 0.95 for value in evidence.primary_consistency
    ):
        reasons.append("P consistency가 한 subband라도 0.95 미만입니다")
    if len(evidence.secondary_consistency) == n_bands and any(
        float(value) < 0.95 for value in evidence.secondary_consistency
    ):
        reasons.append("S consistency가 한 subband라도 0.95 미만입니다")
    if len(evidence.separation_crosscheck_agreement) == n_bands and any(
        float(value) < 0.999 for value in evidence.separation_crosscheck_agreement
    ):
        reasons.append("joint-LS/cubic agreement가 한 subband라도 0.999 미만입니다")
    if len(evidence.separation_crosscheck_relative_error) == n_bands and any(
        float(value) > 0.01 for value in evidence.separation_crosscheck_relative_error
    ):
        reasons.append("joint-LS/cubic relative error가 한 subband라도 0.01 초과입니다")
    if len(evidence.measured_interpolation_agreement) == n_bands and any(
        float(value) < BROADBAND_MEASURED_INTERPOLATION_MIN_AGREEMENT
        for value in evidence.measured_interpolation_agreement
    ):
        reasons.append("measured-band holdout agreement가 한 subband라도 0.995 미만입니다")
    if len(evidence.measured_interpolation_relative_error) == n_bands and any(
        float(value) > BROADBAND_MEASURED_INTERPOLATION_MAX_RELATIVE_ERROR
        for value in evidence.measured_interpolation_relative_error
    ):
        reasons.append("measured-band holdout relative error가 한 subband라도 0.10 초과입니다")
    if (
        evidence.primary_compact_role != "diagnostic_only"
        or evidence.secondary_compact_role != "diagnostic_only"
        or evidence.primary_compact_training_eligible
        or evidence.secondary_compact_training_eligible
    ):
        reasons.append("compact FIR은 diagnostic_only/training_eligible=false여야 합니다")

    return BroadbandPlantAudit(
        status="PASS" if not reasons else "BLOCKED",
        reasons=tuple(reasons),
        contract_sha256=contract.digest(),
        max_timing_error_samples_by_subband=phase_limits,
    )


__all__ = [
    "BROADBAND_FULL_OCTAVE_CONTRACT_ID",
    "BROADBAND_HIGH_SUBBANDS_HZ",
    "BROADBAND_MEASUREMENT_PANELS_HZ",
    "BROADBAND_OCTAVE_CENTERS_HZ",
    "BROADBAND_POINT_CONTROL_CONTRACT_ID",
    "BROADBAND_POINT_CONTROL_SUBBANDS_HZ",
    "BROADBAND_V3_EXCITATION_LOWER_MAX_HZ",
    "BROADBAND_V3_EXCITATION_UPPER_MIN_HZ",
    "BROADBAND_V3_OCTAVE_OBJECTIVE_BANDS_HZ",
    "BROADBAND_V3_OCTAVE_OBJECTIVE_CENTERS_HZ",
    "BROADBAND_V3_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ",
    "BROADBAND_V3_STAGE1_LOW_GUARD_SUBBANDS_HZ",
    "BroadbandFullOctaveContractV3",
    "BroadbandPlantAudit",
    "BroadbandPlantEvidence",
    "CONTROL_BAND_CONTRACT_SCHEMA",
    "CONTROL_BAND_CONTRACT_V3_SCHEMA",
    "ControlBandContract",
    "OCTAVE_8K_UPPER_HZ",
    "STAGE1_STRICT_CONTRACT_ID",
    "STAGE1_STRICT_SUBBANDS_HZ",
    "audit_broadband_plant_evidence",
    "max_timing_error_samples_for_attenuation",
    "phase_error_degrees",
    "resolve_control_band_contract",
]
