"""Stage-2 2 kHz single-point OFF/ON raw segment 평가기.

입력은 동일 모델·동일 Stage-2 P/S binding에서 얻은 한 ERR 위치의 OFF/ON
segment다. 125/250/500/1000/2000 Hz objective는 OFF disturbance의 대역별
source density를 먼저 통과한 segment만 집계한다. 4/8 kHz는 positive attenuation
목표가 아니라 worst-10 증폭 do-no-harm gate다.

2 kHz octave 평균이 1.6 kHz 부근의 near-zero 성능을 가리는 경로를 닫기 위해
1600 Hz one-third-octave sentinel도 같은 source-density/group/평균/worst-10/CI
strict-positive gate로 별도 집계한다. 이 sentinel은 immutable Stage-2 control-band
contract bytes를 바꾸지 않는 평가 admission이다.

이 평가기의 PASS는 single-point 주파수 성능만 뜻한다. 다점 quiet-zone, runtime
deadline/xrun, matched FxLMS 우위 또는 full-octave v3 PASS를 주장하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..dsp.stage2_2khz_contract import Stage2TwoKilohertzContract
from .metrics import band_nmse_db, band_power
from .trusted_subbands import cluster_bootstrap_ci


STAGE2_2KHZ_RAW_EVAL_SCHEMA = "stage2_2khz_single_point_raw_eval_v1"
STAGE2_2KHZ_RAW_EVAL_DOMAIN = "physical_single_point_anc_off_on_raw"
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260831
MIN_RAW_SEGMENT_SAMPLES = 4096
ONE_POINT_SIX_KHZ_SENTINEL_CENTER_HZ = 1600.0
ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ = (1425.437949, 1795.939277)
ONE_POINT_SIX_KHZ_SENTINEL_MINIMUM_ATTENUATION_DB = 0.0

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Stage2TwoKilohertzRawSegment:
    """한 source segment의 matched ANC OFF/ON ERR 파형과 provenance identity."""

    session_id: str
    source_family: str
    group_id: str
    error_position_id: str
    sample_rate: int
    disturbance_off: np.ndarray
    error_on: np.ndarray
    raw_artifact_sha256: str
    model_artifact_sha256: str
    stage2_plant_binding_sha256: str
    control_band_contract_sha256: str
    evaluation_domain: str = STAGE2_2KHZ_RAW_EVAL_DOMAIN


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _serial(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _worst10_mean(values: Sequence[float]) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if array.size == 0 or not np.all(np.isfinite(array)):
        return float("nan")
    count = max(1, int(math.ceil(0.1 * array.size)))
    return float(np.mean(array[:count]))


def _validate_segment(
    segment: Stage2TwoKilohertzRawSegment,
    *,
    contract: Stage2TwoKilohertzContract,
) -> tuple[np.ndarray, np.ndarray]:
    for field in ("session_id", "source_family", "group_id", "error_position_id"):
        if not str(getattr(segment, field)).strip():
            raise ValueError(f"Stage-2 raw segment {field}가 비었습니다")
    if segment.source_family not in contract.source_families:
        raise ValueError(f"지원하지 않는 Stage-2 source family: {segment.source_family!r}")
    if int(segment.sample_rate) != int(contract.sample_rate):
        raise ValueError("Stage-2 raw segment는 48 kHz여야 합니다")
    if segment.evaluation_domain != STAGE2_2KHZ_RAW_EVAL_DOMAIN:
        raise ValueError("legacy/surrogate 결과를 Stage-2 physical OFF/ON raw로 승격할 수 없습니다")
    for field in (
        "raw_artifact_sha256",
        "model_artifact_sha256",
        "stage2_plant_binding_sha256",
        "control_band_contract_sha256",
    ):
        if not _SHA256.fullmatch(str(getattr(segment, field))):
            raise ValueError(f"{field}는 lowercase SHA-256이어야 합니다")
    if segment.control_band_contract_sha256 != contract.digest():
        raise ValueError("raw segment의 Stage-2 contract SHA가 exact canonical SHA와 다릅니다")

    disturbance = np.asarray(segment.disturbance_off, dtype=np.float64).reshape(-1)
    error_on = np.asarray(segment.error_on, dtype=np.float64).reshape(-1)
    if disturbance.size != error_on.size:
        raise ValueError("Stage-2 OFF/ON raw segment 길이가 다릅니다")
    if disturbance.size < MIN_RAW_SEGMENT_SAMPLES:
        raise ValueError(
            f"Stage-2 raw segment는 최소 {MIN_RAW_SEGMENT_SAMPLES} samples여야 합니다"
        )
    if disturbance.size % 256 != 0:
        raise ValueError("Stage-2 raw segment 길이는 256-frame block의 배수여야 합니다")
    if not np.all(np.isfinite(disturbance)) or not np.all(np.isfinite(error_on)):
        raise ValueError("Stage-2 OFF/ON raw segment에 NaN/Inf가 있습니다")
    return disturbance, error_on


def _segment_rows(
    segment: Stage2TwoKilohertzRawSegment,
    *,
    contract: Stage2TwoKilohertzContract,
) -> list[dict[str, Any]]:
    disturbance, error_on = _validate_segment(segment, contract=contract)
    objective_bands = tuple(contract.octave_objective_bands_hz)
    objective_powers = tuple(
        band_power(
            disturbance,
            contract.sample_rate,
            band,
            include_upper=index == len(objective_bands) - 1,
        )
        for index, band in enumerate(objective_bands)
    )
    total_objective_power = float(sum(objective_powers))
    total_objective_width = float(sum(hi - lo for lo, hi in objective_bands))

    rows: list[dict[str, Any]] = []
    for index, (band, power) in enumerate(zip(objective_bands, objective_powers, strict=True)):
        flat_fraction = float(band[1] - band[0]) / total_objective_width
        density = (
            0.0
            if total_objective_power <= np.finfo(np.float64).tiny
            else float((power / total_objective_power) / flat_fraction)
        )
        attenuation = -band_nmse_db(
            disturbance,
            error_on,
            contract.sample_rate,
            band,
            include_upper=index == len(objective_bands) - 1,
        )
        rows.append(
            {
                "octave_role": "attenuation_objective",
                "octave_index": index,
                "octave_center_hz": float(contract.octave_objective_centers_hz[index]),
                "octave_band_hz": [float(band[0]), float(band[1])],
                "off_band_power": float(power),
                "source_density_ratio": density,
                "source_density_gate_required": True,
                "source_density_gate_pass": bool(
                    density >= contract.minimum_source_density_ratio
                ),
                "attenuation_db": float(attenuation),
            }
        )

    sentinel_band = ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ
    sentinel_power = band_power(
        disturbance,
        contract.sample_rate,
        sentinel_band,
        include_upper=False,
    )
    sentinel_flat_fraction = float(sentinel_band[1] - sentinel_band[0]) / (
        total_objective_width
    )
    sentinel_density = (
        0.0
        if total_objective_power <= np.finfo(np.float64).tiny
        else float(
            (sentinel_power / total_objective_power) / sentinel_flat_fraction
        )
    )
    sentinel_attenuation = -band_nmse_db(
        disturbance,
        error_on,
        contract.sample_rate,
        sentinel_band,
        include_upper=False,
    )
    rows.append(
        {
            "octave_role": "one_point_six_khz_sentinel",
            "octave_index": None,
            "octave_center_hz": ONE_POINT_SIX_KHZ_SENTINEL_CENTER_HZ,
            "octave_band_hz": [float(sentinel_band[0]), float(sentinel_band[1])],
            "off_band_power": float(sentinel_power),
            "source_density_ratio": sentinel_density,
            "source_density_gate_required": True,
            "source_density_gate_pass": bool(
                sentinel_density >= contract.minimum_source_density_ratio
            ),
            "attenuation_db": float(sentinel_attenuation),
        }
    )

    dnh_bands = tuple(contract.do_no_harm_octave_bands_hz)
    for index, band in enumerate(dnh_bands):
        off_power = band_power(
            disturbance,
            contract.sample_rate,
            band,
            include_upper=index == len(dnh_bands) - 1,
        )
        attenuation = -band_nmse_db(
            disturbance,
            error_on,
            contract.sample_rate,
            band,
            include_upper=index == len(dnh_bands) - 1,
        )
        rows.append(
            {
                "octave_role": "do_no_harm_observation",
                "octave_index": len(objective_bands) + index,
                "octave_center_hz": float(contract.do_no_harm_octave_centers_hz[index]),
                "octave_band_hz": [float(band[0]), float(band[1])],
                "off_band_power": float(off_power),
                # DNH는 source coverage나 positive attenuation 주장이 아니다. OFF가
                # 조용할 때 ON이 에너지를 주입하는 경우에도 전력비가 보수적으로 실패한다.
                "source_density_ratio": None,
                "source_density_gate_required": False,
                "source_density_gate_pass": True,
                "attenuation_db": float(attenuation),
            }
        )
    return rows


def _aggregate_objective_cell(
    rows: Sequence[dict[str, Any]],
    *,
    center_hz: float,
    contract: Stage2TwoKilohertzContract,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    covered = [row for row in rows if bool(row["source_density_gate_pass"])]
    values = np.asarray([float(row["attenuation_db"]) for row in covered], dtype=np.float64)
    groups = np.asarray([str(row["group_id"]) for row in covered])
    unique_groups = len(set(groups.tolist())) if groups.size else 0
    if values.size:
        mean = float(np.mean(values))
        worst10 = _worst10_mean(values)
        ci_lo, ci_hi, _ = cluster_bootstrap_ci(
            values,
            groups,
            min_groups=contract.minimum_groups_per_family_octave,
            n_resamples=int(n_resamples),
            seed=int(seed),
        )
    else:
        mean = worst10 = ci_lo = ci_hi = float("nan")

    coverage_pass = unique_groups >= contract.minimum_groups_per_family_octave
    is_two_khz = float(center_hz) == 2000.0
    threshold = (
        contract.two_khz_octave_minimum_attenuation_db
        if is_two_khz
        else contract.low_octave_minimum_attenuation_db
    )
    finite = all(math.isfinite(value) for value in (mean, worst10, ci_lo))
    if is_two_khz:
        mean_pass = finite and mean >= threshold
        worst10_pass = finite and worst10 >= threshold
        ci_pass = finite and ci_lo >= threshold
        comparator = ">="
    else:
        mean_pass = finite and mean > threshold
        worst10_pass = finite and worst10 > threshold
        ci_pass = finite and ci_lo > threshold
        comparator = ">"
    passed = bool(coverage_pass and mean_pass and worst10_pass and ci_pass)
    finite_margins = (
        (mean - threshold, worst10 - threshold, ci_lo - threshold)
        if finite
        else ()
    )
    densities = [float(row["source_density_ratio"]) for row in rows]
    return {
        "n_segments_total": len(rows),
        "n_segments_source_covered": len(covered),
        "n_independent_groups_source_covered": unique_groups,
        "minimum_independent_groups_required": (
            contract.minimum_groups_per_family_octave
        ),
        "source_density_ratio_min": _serial(min(densities)) if densities else None,
        "source_density_ratio_mean": _serial(float(np.mean(densities))) if densities else None,
        "source_density_threshold": contract.minimum_source_density_ratio,
        "source_density_and_group_coverage_pass": coverage_pass,
        "attenuation_mean_db": _serial(mean),
        "attenuation_worst10_mean_db": _serial(worst10),
        "attenuation_cluster_ci95_db": [_serial(ci_lo), _serial(ci_hi)],
        "attenuation_threshold_db": float(threshold),
        "attenuation_threshold_comparator": comparator,
        "attenuation_mean_pass": bool(mean_pass),
        "attenuation_worst10_pass": bool(worst10_pass),
        "attenuation_ci_lower_pass": bool(ci_pass),
        "minimum_attenuation_gate_margin_db": (
            _serial(min(finite_margins)) if finite_margins else None
        ),
        "passed": passed,
    }


def _aggregate_dnh_cell(
    rows: Sequence[dict[str, Any]],
    *,
    contract: Stage2TwoKilohertzContract,
) -> dict[str, Any]:
    values = np.asarray([float(row["attenuation_db"]) for row in rows], dtype=np.float64)
    unique_groups = len({str(row["group_id"]) for row in rows})
    group_pass = unique_groups >= contract.minimum_groups_per_family_octave
    mean = float(np.mean(values)) if values.size else float("nan")
    worst10 = _worst10_mean(values)
    worst10_amplification = (
        max(0.0, -worst10) if math.isfinite(worst10) else float("nan")
    )
    dnh_pass = bool(
        math.isfinite(worst10_amplification)
        and worst10_amplification
        < contract.do_no_harm_max_worst10_amplification_db
    )
    dnh_margin = (
        contract.do_no_harm_max_worst10_amplification_db
        - worst10_amplification
        if math.isfinite(worst10_amplification)
        else float("nan")
    )
    return {
        "n_segments_total": len(rows),
        "n_independent_groups": unique_groups,
        "minimum_independent_groups_required": (
            contract.minimum_groups_per_family_octave
        ),
        "group_coverage_pass": group_pass,
        "source_density_gate_required": False,
        "attenuation_mean_db": _serial(mean),
        "attenuation_worst10_mean_db": _serial(worst10),
        "worst10_amplification_db": _serial(worst10_amplification),
        "worst10_amplification_limit_db": (
            contract.do_no_harm_max_worst10_amplification_db
        ),
        "worst10_amplification_comparator": "<",
        "do_no_harm_pass": dnh_pass,
        "minimum_do_no_harm_gate_margin_db": _serial(dnh_margin),
        "passed": bool(group_pass and dnh_pass),
    }


def evaluate_stage2_2khz_raw_segments(
    segments: Sequence[Stage2TwoKilohertzRawSegment],
    *,
    contract: Stage2TwoKilohertzContract,
    minimum_groups: int | None = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """네 family의 single-point raw를 125 Hz--2 kHz와 4/8 kHz DNH로 판정한다."""

    canonical = Stage2TwoKilohertzContract.canonical()
    if contract != canonical:
        raise ValueError("exact canonical Stage-2 2 kHz contract가 필요합니다")
    required_groups = (
        contract.minimum_groups_per_family_octave
        if minimum_groups is None
        else int(minimum_groups)
    )
    if required_groups != contract.minimum_groups_per_family_octave:
        raise ValueError("family×octave 독립 group 하한 4를 변경할 수 없습니다")
    if int(n_resamples) <= 0:
        raise ValueError("bootstrap resample 수는 양수여야 합니다")
    if not segments:
        raise ValueError("평가할 Stage-2 OFF/ON raw segment가 없습니다")

    positions: set[str] = set()
    model_shas: set[str] = set()
    plant_shas: set[str] = set()
    seen_session_position: set[tuple[str, str]] = set()
    raw_sha_to_group: dict[str, str] = {}
    raw_rows: list[dict[str, Any]] = []
    for segment in segments:
        disturbance, _ = _validate_segment(segment, contract=canonical)
        del disturbance
        key = (str(segment.session_id), str(segment.error_position_id))
        if key in seen_session_position:
            raise ValueError("같은 Stage-2 session/ERR position raw segment가 중복됐습니다")
        seen_session_position.add(key)
        positions.add(str(segment.error_position_id))
        model_shas.add(str(segment.model_artifact_sha256))
        plant_shas.add(str(segment.stage2_plant_binding_sha256))
        previous_group = raw_sha_to_group.setdefault(
            str(segment.raw_artifact_sha256), str(segment.group_id)
        )
        if previous_group != str(segment.group_id):
            raise ValueError("같은 raw artifact를 서로 다른 독립 group으로 셀 수 없습니다")
        for row in _segment_rows(segment, contract=canonical):
            raw_rows.append(
                {
                    "session_id": str(segment.session_id),
                    "source_family": str(segment.source_family),
                    "group_id": str(segment.group_id),
                    "error_position_id": str(segment.error_position_id),
                    "raw_artifact_sha256": str(segment.raw_artifact_sha256),
                    **row,
                }
            )

    if len(positions) != 1:
        raise ValueError("Stage-2 evaluator는 ERR 위치 하나의 single-point raw만 받습니다")
    if len(model_shas) != 1:
        raise ValueError("하나의 exact model artifact raw만 함께 평가할 수 있습니다")
    if len(plant_shas) != 1:
        raise ValueError("하나의 exact Stage-2 P/S binding raw만 함께 평가할 수 있습니다")

    position = next(iter(positions))
    objective_cells: list[dict[str, Any]] = []
    sentinel_cells: list[dict[str, Any]] = []
    dnh_cells: list[dict[str, Any]] = []
    reasons: list[str] = []
    for family_index, family in enumerate(canonical.source_families):
        for octave_index, (center, band) in enumerate(
            zip(
                canonical.octave_objective_centers_hz,
                canonical.octave_objective_bands_hz,
                strict=True,
            )
        ):
            selected = [
                row
                for row in raw_rows
                if row["source_family"] == family
                and row["octave_role"] == "attenuation_objective"
                and float(row["octave_center_hz"]) == float(center)
            ]
            aggregate = _aggregate_objective_cell(
                selected,
                center_hz=float(center),
                contract=canonical,
                n_resamples=int(n_resamples),
                seed=int(seed) + 101 * family_index + octave_index,
            )
            cell = {
                "error_position_id": position,
                "source_family": family,
                "octave_role": "attenuation_objective",
                "octave_center_hz": float(center),
                "octave_band_hz": [float(band[0]), float(band[1])],
                **aggregate,
            }
            objective_cells.append(cell)
            if not cell["passed"]:
                reasons.append(
                    f"{position}/{family}/{center:.0f}Hz objective가 "
                    "source-density/group/mean/worst10/CI-lower gate를 통과하지 못했습니다"
                )

        sentinel_selected = [
            row
            for row in raw_rows
            if row["source_family"] == family
            and row["octave_role"] == "one_point_six_khz_sentinel"
        ]
        sentinel_aggregate = _aggregate_objective_cell(
            sentinel_selected,
            center_hz=ONE_POINT_SIX_KHZ_SENTINEL_CENTER_HZ,
            contract=canonical,
            n_resamples=int(n_resamples),
            seed=int(seed) + 101 * family_index + 97,
        )
        sentinel_cell = {
            "error_position_id": position,
            "source_family": family,
            "octave_role": "one_point_six_khz_sentinel",
            "octave_center_hz": ONE_POINT_SIX_KHZ_SENTINEL_CENTER_HZ,
            "octave_band_hz": [
                float(ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ[0]),
                float(ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ[1]),
            ],
            **sentinel_aggregate,
        }
        sentinel_cells.append(sentinel_cell)
        if not sentinel_cell["passed"]:
            reasons.append(
                f"{position}/{family}/1600Hz one-third-octave sentinel이 "
                "source-density/group/strict-positive mean/worst10/CI-lower gate를 "
                "통과하지 못했습니다"
            )

        for dnh_index, (center, band) in enumerate(
            zip(
                canonical.do_no_harm_octave_centers_hz,
                canonical.do_no_harm_octave_bands_hz,
                strict=True,
            )
        ):
            selected = [
                row
                for row in raw_rows
                if row["source_family"] == family
                and row["octave_role"] == "do_no_harm_observation"
                and float(row["octave_center_hz"]) == float(center)
            ]
            aggregate = _aggregate_dnh_cell(selected, contract=canonical)
            cell = {
                "error_position_id": position,
                "source_family": family,
                "octave_role": "do_no_harm_observation",
                "octave_center_hz": float(center),
                "octave_band_hz": [float(band[0]), float(band[1])],
                **aggregate,
            }
            dnh_cells.append(cell)
            if not cell["passed"]:
                reasons.append(
                    f"{position}/{family}/{center:.0f}Hz DNH가 "
                    "독립 group 또는 worst10 증폭 <1 dB를 통과하지 못했습니다"
                )

    performance_margins = [
        float(cell["minimum_attenuation_gate_margin_db"])
        for cell in objective_cells
        if cell["minimum_attenuation_gate_margin_db"] is not None
    ] + [
        float(cell["minimum_attenuation_gate_margin_db"])
        for cell in sentinel_cells
        if cell["minimum_attenuation_gate_margin_db"] is not None
    ] + [
        float(cell["minimum_do_no_harm_gate_margin_db"])
        for cell in dnh_cells
        if cell["minimum_do_no_harm_gate_margin_db"] is not None
    ]
    two_khz_family_means = [
        float(cell["attenuation_mean_db"])
        for cell in objective_cells
        if float(cell["octave_center_hz"]) == 2000.0
        and cell["attenuation_mean_db"] is not None
    ]
    payload: dict[str, Any] = {
        "schema": STAGE2_2KHZ_RAW_EVAL_SCHEMA,
        "role": STAGE2_2KHZ_RAW_EVAL_DOMAIN,
        "status": "PASS" if not reasons else "BLOCKED",
        "single_point_only": True,
        "spatial_quiet_zone_claim": False,
        "full_octave_v3_claim": False,
        "deployment_runtime_claim": False,
        "control_band_contract_id": canonical.contract_id,
        "control_band_contract_sha256": canonical.digest(),
        "model_artifact_sha256": next(iter(model_shas)),
        "stage2_plant_binding_sha256": next(iter(plant_shas)),
        "error_position_id": position,
        "minimum_source_density_ratio": canonical.minimum_source_density_ratio,
        "minimum_groups_per_family_octave": required_groups,
        "bootstrap_resamples": int(n_resamples),
        "bootstrap_seed": int(seed),
        "objective_cells": objective_cells,
        "one_point_six_khz_sentinel_band_hz": [
            float(ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ[0]),
            float(ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ[1]),
        ],
        "one_point_six_khz_sentinel_cells": sentinel_cells,
        "do_no_harm_cells": dnh_cells,
        "all_source_density_and_group_gates_pass": bool(
            objective_cells
            and sentinel_cells
            and all(
                bool(cell["source_density_and_group_coverage_pass"])
                for cell in (*objective_cells, *sentinel_cells)
            )
        ),
        "all_attenuation_objectives_pass": bool(
            objective_cells and all(bool(cell["passed"]) for cell in objective_cells)
        ),
        "all_one_point_six_khz_sentinel_gates_pass": bool(
            sentinel_cells and all(bool(cell["passed"]) for cell in sentinel_cells)
        ),
        "all_do_no_harm_gates_pass": bool(
            dnh_cells and all(bool(cell["passed"]) for cell in dnh_cells)
        ),
        # 모델 선택은 threshold PASS 뒤에도 멈추지 않는다. 단, 이 값은 주파수
        # 성능 gate의 dB margin이며 latency gate는 별도 exact PASS가 먼저 필요하다.
        "minimum_frequency_gate_margin_db": (
            _serial(min(performance_margins)) if performance_margins else None
        ),
        "two_khz_family_equal_mean_attenuation_db": (
            _serial(float(np.mean(two_khz_family_means)))
            if len(two_khz_family_means) == len(canonical.source_families)
            else None
        ),
        "checkpoint_selection_policy": {
            "three_db_is_minimum_not_optimization_target": True,
            "eligibility_requires_all_band_family_density_group_dnh_gates_pass": True,
            "eligibility_requires_one_point_six_khz_sentinel_pass": True,
            "one_point_six_khz_sentinel_runtime_exact_zero_required": True,
            "eligibility_requires_external_physical_runtime_latency_gate_pass": True,
            "eligible_from_this_frequency_report_alone": False,
            "primary_order": "maximize_minimum_frequency_gate_margin_db",
            "secondary_order": "maximize_two_khz_family_equal_mean_attenuation_db",
            "latency_or_discontinuity_failure_can_never_be_traded_for_attenuation": True,
        },
        "reasons": reasons,
        "limitations": [
            "한 ERR 위치의 single-point 결과이며 spatial quiet-zone 증거가 아닙니다",
            "4/8 kHz는 do-no-harm 관측일 뿐 positive attenuation 목표가 아닙니다",
            "runtime deadline/xrun/drop/add/slip과 matched FxLMS는 별도 raw gate입니다",
            "legacy 또는 full-octave v3 결과로 자동 승격할 수 없습니다",
        ],
    }
    payload["evidence_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "MIN_RAW_SEGMENT_SAMPLES",
    "ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ",
    "ONE_POINT_SIX_KHZ_SENTINEL_CENTER_HZ",
    "ONE_POINT_SIX_KHZ_SENTINEL_MINIMUM_ATTENUATION_DB",
    "STAGE2_2KHZ_RAW_EVAL_DOMAIN",
    "STAGE2_2KHZ_RAW_EVAL_SCHEMA",
    "Stage2TwoKilohertzRawSegment",
    "evaluate_stage2_2khz_raw_segments",
]
