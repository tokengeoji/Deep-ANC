"""125 Hz--2 kHz single-point Stage-2의 immutable 제어 계약.

이 계약은 기존 ``control_band_contract`` v1/v2 및 full-octave v3를 수정하거나
축소해 재사용하지 않는다. Stage-2의 성공은 실제 덕트 한 ERR 위치에서
125/250/500/1000 Hz를 보존하면서 2 kHz *옥타브 전체*의 증폭을 막는 것이다.
6 dB 성능 목표는 별도 1.6 kHz sentinel 평가 계약에서 강제한다.
4/8 kHz는 제어 목표가 아니라 별도의 do-no-harm 관측 대역이며, Stage-2 PASS를
다점 quiet-zone 또는 full-octave v3 PASS로 승격할 수 없다.

이 모듈은 오디오 장치나 artifact 파일을 열지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, model_validator


STAGE2_2KHZ_CONTRACT_SCHEMA = "stage2_2khz_contract_v2"
STAGE2_2KHZ_CONTRACT_ID = "broadband_2khz_octave_88_2828_v2"

STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ: tuple[
    tuple[float, float], ...
] = (
    (88.3883476483, 150.0),
    (150.0, 300.0),
    (300.0, 600.0),
    (600.0, 1000.0),
    (1000.0, 1600.0),
    (1600.0, 2828.4271247462),
)

STAGE2_2KHZ_OBJECTIVE_CENTERS_HZ: tuple[float, ...] = (
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
)

STAGE2_2KHZ_OBJECTIVE_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (88.3883476483, 176.7766952966),
    (176.7766952966, 353.5533905933),
    (353.5533905933, 707.1067811865),
    (707.1067811865, 1414.2135623731),
    (1414.2135623731, 2828.4271247462),
)

STAGE2_2KHZ_DNH_CENTERS_HZ: tuple[float, ...] = (4000.0, 8000.0)
STAGE2_2KHZ_DNH_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (2828.4271247462, 5656.8542494924),
    (5656.8542494924, 11313.7084989848),
)

STAGE2_2KHZ_SOURCE_FAMILIES: tuple[str, ...] = (
    "speech",
    "music",
    "environment",
    "machine",
)

STAGE2_2KHZ_REQUIRED_EXCITATION_LOWER_MAX_HZ = 80.0
STAGE2_2KHZ_REQUIRED_EXCITATION_UPPER_MIN_HZ = 2828.4271247462
STAGE2_2KHZ_MIN_GROUPS_PER_FAMILY_OCTAVE = 4
STAGE2_2KHZ_MIN_SOURCE_DENSITY_RATIO = 0.25
STAGE2_2KHZ_MINIMUM_ATTENUATION_DB = 0.0
STAGE2_2KHZ_DNH_MAX_WORST10_AMPLIFICATION_DB = 1.0


def _exact_rows(
    actual: Sequence[Sequence[float]], expected: Sequence[Sequence[float]]
) -> bool:
    try:
        parsed = tuple(tuple(float(value) for value in row) for row in actual)
    except (TypeError, ValueError):
        return False
    return parsed == tuple(tuple(float(value) for value in row) for row in expected)


def _validate_contiguous(rows: Sequence[Sequence[float]], *, label: str) -> None:
    parsed = tuple(tuple(float(value) for value in row) for row in rows)
    if not parsed:
        raise ValueError(f"{label}가 비었습니다")
    for index, row in enumerate(parsed):
        if len(row) != 2 or not all(math.isfinite(value) for value in row):
            raise ValueError(f"{label} #{index}가 유효한 [lo, hi]가 아닙니다")
        if not 0.0 < row[0] < row[1]:
            raise ValueError(f"{label} #{index}의 순서가 잘못됐습니다")
        if index and row[0] != parsed[index - 1][1]:
            raise ValueError(f"{label}에 gap/overlap이 있습니다")


class Stage2TwoKilohertzContract(BaseModel):
    """2 kHz octave 증폭 방지와 1.6 kHz sentinel 목표를 위한 single-point 계약."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["stage2_2khz_contract_v2"] = (
        STAGE2_2KHZ_CONTRACT_SCHEMA
    )
    contract_id: Literal["broadband_2khz_octave_88_2828_v2"] = (
        STAGE2_2KHZ_CONTRACT_ID
    )
    role: Literal["broadband_2khz_octave_single_point"] = (
        "broadband_2khz_octave_single_point"
    )
    sample_rate: Literal[48000] = 48_000
    physical_identification_subbands_hz: tuple[tuple[float, float], ...] = (
        STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ
    )
    octave_objective_centers_hz: tuple[float, ...] = (
        STAGE2_2KHZ_OBJECTIVE_CENTERS_HZ
    )
    octave_objective_bands_hz: tuple[tuple[float, float], ...] = (
        STAGE2_2KHZ_OBJECTIVE_BANDS_HZ
    )
    do_no_harm_octave_centers_hz: tuple[float, ...] = STAGE2_2KHZ_DNH_CENTERS_HZ
    do_no_harm_octave_bands_hz: tuple[tuple[float, float], ...] = (
        STAGE2_2KHZ_DNH_BANDS_HZ
    )
    required_excitation_lower_hz: float = (
        STAGE2_2KHZ_REQUIRED_EXCITATION_LOWER_MAX_HZ
    )
    required_excitation_upper_hz: float = (
        STAGE2_2KHZ_REQUIRED_EXCITATION_UPPER_MIN_HZ
    )
    source_families: tuple[str, ...] = STAGE2_2KHZ_SOURCE_FAMILIES
    minimum_groups_per_family_octave: int = (
        STAGE2_2KHZ_MIN_GROUPS_PER_FAMILY_OCTAVE
    )
    minimum_source_density_ratio: float = STAGE2_2KHZ_MIN_SOURCE_DENSITY_RATIO
    low_octave_minimum_attenuation_db: float = 0.0
    low_octave_threshold_semantics: Literal["strictly_greater_than"] = (
        "strictly_greater_than"
    )
    two_khz_octave_minimum_attenuation_db: float = (
        STAGE2_2KHZ_MINIMUM_ATTENUATION_DB
    )
    two_khz_threshold_semantics: Literal["strictly_greater_than"] = (
        "strictly_greater_than"
    )
    do_no_harm_max_worst10_amplification_db: float = (
        STAGE2_2KHZ_DNH_MAX_WORST10_AMPLIFICATION_DB
    )
    do_no_harm_threshold_semantics: Literal["strictly_less_than"] = (
        "strictly_less_than"
    )
    physical_identification_semantics: Literal[
        "plant_identification_and_source_coverage_not_octave_loss_weighting"
    ] = "plant_identification_and_source_coverage_not_octave_loss_weighting"
    do_no_harm_semantics: Literal[
        "4k_8k_observation_only_not_positive_attenuation_objective"
    ] = "4k_8k_observation_only_not_positive_attenuation_objective"
    single_point_only: Literal[True] = True
    spatial_quiet_zone_claim_allowed: Literal[False] = False
    legacy_automatic_promotion_allowed: Literal[False] = False
    full_octave_v3_automatic_promotion_allowed: Literal[False] = False
    requires_exact_contract_sha256: Literal[True] = True

    @model_validator(mode="after")
    def _validate_contract(self) -> "Stage2TwoKilohertzContract":
        _validate_contiguous(
            self.physical_identification_subbands_hz,
            label="Stage-2 physical identification subband",
        )
        _validate_contiguous(
            self.octave_objective_bands_hz,
            label="Stage-2 octave objective band",
        )
        _validate_contiguous(
            self.do_no_harm_octave_bands_hz,
            label="Stage-2 do-no-harm octave band",
        )
        if not _exact_rows(
            self.physical_identification_subbands_hz,
            STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ,
        ):
            raise ValueError("Stage-2 physical identification 6구간을 변경할 수 없습니다")
        if tuple(self.octave_objective_centers_hz) != (
            STAGE2_2KHZ_OBJECTIVE_CENTERS_HZ
        ):
            raise ValueError("Stage-2 objective octave center 5개를 변경할 수 없습니다")
        if not _exact_rows(
            self.octave_objective_bands_hz,
            STAGE2_2KHZ_OBJECTIVE_BANDS_HZ,
        ):
            raise ValueError("Stage-2 objective octave band를 변경할 수 없습니다")
        if tuple(self.do_no_harm_octave_centers_hz) != STAGE2_2KHZ_DNH_CENTERS_HZ:
            raise ValueError("Stage-2 do-no-harm center는 4/8 kHz여야 합니다")
        if not _exact_rows(
            self.do_no_harm_octave_bands_hz,
            STAGE2_2KHZ_DNH_BANDS_HZ,
        ):
            raise ValueError("Stage-2 do-no-harm 4/8 kHz band를 변경할 수 없습니다")
        if tuple(self.source_families) != STAGE2_2KHZ_SOURCE_FAMILIES:
            raise ValueError("speech/music/environment/machine 네 family가 모두 필요합니다")

        for center, (lower, upper) in zip(
            self.octave_objective_centers_hz,
            self.octave_objective_bands_hz,
            strict=True,
        ):
            if not math.isclose(
                math.sqrt(lower * upper), center, rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise ValueError("Stage-2 objective band의 기하 중심이 octave center와 다릅니다")
        for center, (lower, upper) in zip(
            self.do_no_harm_octave_centers_hz,
            self.do_no_harm_octave_bands_hz,
            strict=True,
        ):
            if not math.isclose(
                math.sqrt(lower * upper), center, rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise ValueError("Stage-2 do-no-harm band의 기하 중심이 octave center와 다릅니다")

        lower = float(self.required_excitation_lower_hz)
        upper = float(self.required_excitation_upper_hz)
        if (
            not math.isfinite(lower)
            or lower <= 0.0
            or lower > STAGE2_2KHZ_REQUIRED_EXCITATION_LOWER_MAX_HZ
            or lower > self.physical_identification_subbands_hz[0][0]
        ):
            raise ValueError("Stage-2 excitation lower는 0 Hz 초과 80 Hz 이하여야 합니다")
        if (
            not math.isfinite(upper)
            or upper < STAGE2_2KHZ_REQUIRED_EXCITATION_UPPER_MIN_HZ
            or upper < self.physical_identification_subbands_hz[-1][1]
            or upper > self.sample_rate / 2.0
        ):
            raise ValueError("Stage-2 excitation upper는 2,828.4271247462 Hz 이상이어야 합니다")
        if int(self.minimum_groups_per_family_octave) != (
            STAGE2_2KHZ_MIN_GROUPS_PER_FAMILY_OCTAVE
        ):
            raise ValueError("family×octave 독립 group 하한 4를 변경할 수 없습니다")
        if float(self.minimum_source_density_ratio) != (
            STAGE2_2KHZ_MIN_SOURCE_DENSITY_RATIO
        ):
            raise ValueError("source density ratio 하한 0.25를 변경할 수 없습니다")
        if float(self.low_octave_minimum_attenuation_db) != 0.0:
            raise ValueError("125--1kHz octave 감쇠 하한 0 dB를 변경할 수 없습니다")
        if float(self.two_khz_octave_minimum_attenuation_db) != (
            STAGE2_2KHZ_MINIMUM_ATTENUATION_DB
        ):
            raise ValueError("2 kHz octave 증폭 방지 하한 0 dB를 변경할 수 없습니다")
        if float(self.do_no_harm_max_worst10_amplification_db) != (
            STAGE2_2KHZ_DNH_MAX_WORST10_AMPLIFICATION_DB
        ):
            raise ValueError("4/8 kHz worst10 증폭 한도 1 dB를 변경할 수 없습니다")
        return self

    @classmethod
    def canonical(cls) -> "Stage2TwoKilohertzContract":
        return cls()

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


__all__ = [
    "STAGE2_2KHZ_CONTRACT_ID",
    "STAGE2_2KHZ_CONTRACT_SCHEMA",
    "STAGE2_2KHZ_DNH_BANDS_HZ",
    "STAGE2_2KHZ_DNH_CENTERS_HZ",
    "STAGE2_2KHZ_DNH_MAX_WORST10_AMPLIFICATION_DB",
    "STAGE2_2KHZ_MIN_GROUPS_PER_FAMILY_OCTAVE",
    "STAGE2_2KHZ_MIN_SOURCE_DENSITY_RATIO",
    "STAGE2_2KHZ_MINIMUM_ATTENUATION_DB",
    "STAGE2_2KHZ_OBJECTIVE_BANDS_HZ",
    "STAGE2_2KHZ_OBJECTIVE_CENTERS_HZ",
    "STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ",
    "STAGE2_2KHZ_REQUIRED_EXCITATION_LOWER_MAX_HZ",
    "STAGE2_2KHZ_REQUIRED_EXCITATION_UPPER_MIN_HZ",
    "STAGE2_2KHZ_SOURCE_FAMILIES",
    "Stage2TwoKilohertzContract",
]
