"""광대역 point-control과 matched FxLMS의 raw segment 판정.

Stage-1 150--1600 Hz G4를 확장 해석하지 않고 별도 계약으로 계산한다. 모든
split/source lineage 검증이 끝난 test segment를 입력으로 받아, 저역의 양의 감쇠와
고역의 matched FxLMS 우위를 family×subband별로 독립 판정한다. 전체 평균은 어느
한 대역이나 family의 실패를 숨길 수 없다.

이 모듈은 오디오 장치를 열거나 artifact를 저장하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..dsp.control_band_contract import ControlBandContract
from .metrics import band_nmse_db, band_power
from .trusted_subbands import cluster_bootstrap_ci


BROADBAND_POINT_CONTROL_EVAL_SCHEMA = "broadband_point_control_eval_v1"
MIN_TARGET_D_DENSITY_RATIO = 0.25
MIN_GROUPS_PER_FAMILY_BAND = 4
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260828


@dataclass(frozen=True)
class BroadbandControlSegment:
    """동일 physical window의 OFF/DL/FxLMS matched segment."""

    session_id: str
    source_family: str
    group_id: str
    error_position_id: str
    sample_rate: int
    disturbance_off: np.ndarray
    error_deep_anc: np.ndarray
    error_fxlms: np.ndarray


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _worst10_mean(values: Sequence[float]) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if array.size == 0 or not np.all(np.isfinite(array)):
        return float("nan")
    count = max(1, int(math.ceil(array.size * 0.1)))
    return float(np.mean(array[:count]))


def _validate_segment(
    segment: BroadbandControlSegment,
    *,
    contract: ControlBandContract,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for field in ("session_id", "source_family", "group_id", "error_position_id"):
        value = str(getattr(segment, field)).strip()
        if not value:
            raise ValueError(f"broadband segment {field}가 비었습니다")
    if segment.source_family not in contract.source_families:
        raise ValueError(f"지원하지 않는 source family: {segment.source_family!r}")
    if int(segment.sample_rate) != contract.sample_rate:
        raise ValueError("broadband segment는 control contract의 48 kHz여야 합니다")
    values = tuple(
        np.asarray(raw, dtype=np.float64).reshape(-1)
        for raw in (
            segment.disturbance_off,
            segment.error_deep_anc,
            segment.error_fxlms,
        )
    )
    if min(value.size for value in values) < 256:
        raise ValueError("broadband matched segment가 256 samples보다 짧습니다")
    if len({value.size for value in values}) != 1:
        raise ValueError("OFF/DL/FxLMS matched segment 길이가 다릅니다")
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("broadband matched segment에 NaN/Inf가 있습니다")
    return values


def _segment_rows(
    segment: BroadbandControlSegment,
    *,
    contract: ControlBandContract,
) -> list[dict[str, Any]]:
    disturbance, error_dl, error_fxlms = _validate_segment(segment, contract=contract)
    total_power = band_power(
        disturbance,
        contract.sample_rate,
        contract.point_control_target_hz,
    )
    target_width = (
        contract.point_control_target_hz[1] - contract.point_control_target_hz[0]
    )
    rows: list[dict[str, Any]] = []
    for band_index, band in enumerate(contract.point_control_subbands_hz):
        include_upper = band_index == len(contract.point_control_subbands_hz) - 1
        source_power = band_power(
            disturbance,
            contract.sample_rate,
            band,
            include_upper=include_upper,
        )
        flat_fraction = (band[1] - band[0]) / target_width
        density = (
            0.0
            if total_power <= np.finfo(np.float64).tiny
            else float((source_power / total_power) / flat_fraction)
        )
        dl_attenuation = -band_nmse_db(
            disturbance,
            error_dl,
            contract.sample_rate,
            band,
            include_upper=include_upper,
        )
        fxlms_attenuation = -band_nmse_db(
            disturbance,
            error_fxlms,
            contract.sample_rate,
            band,
            include_upper=include_upper,
        )
        rows.append(
            {
                "band_index": band_index,
                "band_hz": [float(band[0]), float(band[1])],
                "target_density_ratio": density,
                "covered": density >= MIN_TARGET_D_DENSITY_RATIO,
                "attenuation_deep_anc_db": dl_attenuation,
                "attenuation_fxlms_db": fxlms_attenuation,
                "paired_delta_deep_anc_minus_fxlms_db": (
                    dl_attenuation - fxlms_attenuation
                ),
            }
        )
    return rows


def _aggregate_cell(
    rows: Sequence[dict[str, Any]],
    *,
    high_band: bool,
    minimum_groups: int,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    covered = [row for row in rows if bool(row["covered"])]
    groups = np.asarray([str(row["group_id"]) for row in covered])
    dl = np.asarray(
        [float(row["attenuation_deep_anc_db"]) for row in covered],
        dtype=np.float64,
    )
    delta = np.asarray(
        [float(row["paired_delta_deep_anc_minus_fxlms_db"]) for row in covered],
        dtype=np.float64,
    )
    unique_groups = len(set(groups.tolist())) if groups.size else 0
    if dl.size:
        dl_mean = float(np.mean(dl))
        dl_worst10 = _worst10_mean(dl)
        dl_ci_lo, dl_ci_hi, _ = cluster_bootstrap_ci(
            dl,
            groups,
            min_groups=minimum_groups,
            n_resamples=n_resamples,
            seed=seed,
        )
        delta_mean = float(np.mean(delta))
        delta_worst10 = _worst10_mean(delta)
        delta_ci_lo, delta_ci_hi, _ = cluster_bootstrap_ci(
            delta,
            groups,
            min_groups=minimum_groups,
            n_resamples=n_resamples,
            seed=seed + 1,
        )
    else:
        dl_mean = dl_worst10 = dl_ci_lo = dl_ci_hi = float("nan")
        delta_mean = delta_worst10 = delta_ci_lo = delta_ci_hi = float("nan")

    coverage_pass = unique_groups >= minimum_groups
    attenuation_pass = bool(
        coverage_pass
        and all(math.isfinite(value) for value in (dl_mean, dl_worst10, dl_ci_lo))
        and dl_mean > 0.0
        and dl_worst10 > 0.0
        and dl_ci_lo > 0.0
    )
    superiority_pass = bool(
        not high_band
        or (
            coverage_pass
            and all(
                math.isfinite(value)
                for value in (delta_mean, delta_worst10, delta_ci_lo)
            )
            and delta_mean > 0.0
            and delta_worst10 > 0.0
            and delta_ci_lo > 0.0
        )
    )

    def serial(value: float) -> float | None:
        return float(value) if math.isfinite(float(value)) else None

    return {
        "n_segments_total": len(rows),
        "n_segments_covered": len(covered),
        "n_independent_groups_covered": unique_groups,
        "coverage_pass": coverage_pass,
        "deep_anc_attenuation_mean_db": serial(dl_mean),
        "deep_anc_attenuation_worst10_mean_db": serial(dl_worst10),
        "deep_anc_attenuation_cluster_ci95_db": [serial(dl_ci_lo), serial(dl_ci_hi)],
        "positive_attenuation_pass": attenuation_pass,
        "paired_delta_mean_db": serial(delta_mean),
        "paired_delta_worst10_mean_db": serial(delta_worst10),
        "paired_delta_cluster_ci95_db": [serial(delta_ci_lo), serial(delta_ci_hi)],
        "matched_fxlms_superiority_required": high_band,
        "matched_fxlms_superiority_pass": superiority_pass,
        "passed": attenuation_pass and superiority_pass,
    }


def evaluate_broadband_point_control_segments(
    segments: Sequence[BroadbandControlSegment],
    *,
    contract: ControlBandContract,
    require_spatial: bool = False,
    minimum_groups: int = MIN_GROUPS_PER_FAMILY_BAND,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """raw matched segments를 family×subband(필요 시 위치별)로 판정한다."""

    if contract.role != "broadband_point_control":
        raise ValueError("광대역 point-control 평가에는 broadband contract가 필요합니다")
    minimum = int(minimum_groups)
    if minimum < MIN_GROUPS_PER_FAMILY_BAND:
        raise ValueError("family×band 독립 group 하한을 4보다 낮출 수 없습니다")
    if int(n_resamples) <= 0:
        raise ValueError("bootstrap resample 수는 양수여야 합니다")
    if not segments:
        raise ValueError("평가할 broadband matched segment가 없습니다")

    seen_session_position: set[tuple[str, str]] = set()
    raw_rows: list[dict[str, Any]] = []
    positions: set[str] = set()
    for segment in segments:
        key = (segment.session_id, segment.error_position_id)
        if key in seen_session_position:
            raise ValueError("같은 session/ERR position segment가 중복됐습니다")
        seen_session_position.add(key)
        positions.add(segment.error_position_id)
        for row in _segment_rows(segment, contract=contract):
            raw_rows.append(
                {
                    "session_id": segment.session_id,
                    "source_family": segment.source_family,
                    "group_id": segment.group_id,
                    "error_position_id": segment.error_position_id,
                    **row,
                }
            )

    if not require_spatial and len(positions) != 1:
        raise ValueError(
            "point-control 평가는 ERR 위치 하나만 받아야 합니다; 다점은 spatial=True로 "
            "각 위치를 독립 판정하세요"
        )
    spatial_count_pass = (
        len(positions) >= contract.minimum_spatial_error_positions
        if require_spatial
        else True
    )
    dimensions = sorted(positions) if require_spatial else [next(iter(positions))]
    cells: list[dict[str, Any]] = []
    reasons: list[str] = []
    for position in dimensions:
        for family in contract.source_families:
            for band_index, band in enumerate(contract.point_control_subbands_hz):
                selected = [
                    row
                    for row in raw_rows
                    if row["error_position_id"] == position
                    and row["source_family"] == family
                    and row["band_index"] == band_index
                ]
                aggregate = _aggregate_cell(
                    selected,
                    high_band=band_index >= 4,
                    minimum_groups=minimum,
                    n_resamples=int(n_resamples),
                    seed=int(seed) + band_index,
                )
                cell = {
                    "error_position_id": position,
                    "source_family": family,
                    "band_hz": [float(band[0]), float(band[1])],
                    **aggregate,
                }
                cells.append(cell)
                if not cell["passed"]:
                    reasons.append(
                        f"{position}/{family}/{band[0]:.0f}-{band[1]:.0f}Hz가 "
                        "coverage/positive attenuation/FxLMS superiority 중 하나를 통과하지 "
                        "못했습니다"
                    )
    if require_spatial and not spatial_count_pass:
        reasons.append(
            f"독립 ERR 위치 {len(positions)} < "
            f"{contract.minimum_spatial_error_positions}"
        )

    payload: dict[str, Any] = {
        "schema": BROADBAND_POINT_CONTROL_EVAL_SCHEMA,
        "role": (
            "spatial_quiet_zone_acoustic_metric_only"
            if require_spatial
            else "single_point_acoustic_metric_only"
        ),
        "status": "PASS" if not reasons else "BLOCKED",
        "control_band_contract_sha256": contract.digest(),
        "minimum_target_d_density_ratio": MIN_TARGET_D_DENSITY_RATIO,
        "minimum_groups_per_family_band": minimum,
        "bootstrap_resamples": int(n_resamples),
        "bootstrap_seed": int(seed),
        "error_positions": sorted(positions),
        "spatial_position_count_pass": spatial_count_pass,
        "cells": cells,
        "reasons": reasons,
        "limitations": [
            "latency/xrun/deadline/runtime stability는 별도 게이트입니다",
            "manifest/test-once/provenance 결속은 이 pure metric 함수 밖에서 검증해야 합니다",
        ],
    }
    payload["evidence_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


__all__ = [
    "BROADBAND_POINT_CONTROL_EVAL_SCHEMA",
    "BroadbandControlSegment",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "MIN_GROUPS_PER_FAMILY_BAND",
    "MIN_TARGET_D_DENSITY_RATIO",
    "evaluate_broadband_point_control_segments",
]
