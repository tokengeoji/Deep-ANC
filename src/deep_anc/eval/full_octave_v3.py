"""125 Hz--8 kHz v3의 matched surrogate 평가기.

이 모듈은 실제 오디오 장치나 파일을 열지 않는다.  이미 같은 causal P/S binding에서
생성된 ``OFF / Deep-ANC / FxLMS`` 세 파형만 받아, 125/250/500/1k/2k/4k/8k Hz
octave를 분리해 판정한다.  따라서 이 결과는 *surrogate matched P/S* 결과이며,
실제 덕트의 G4 또는 배포 성능을 주장하지 않는다.

기존 ``broadband_point_control`` v2는 150 Hz 하단의 7개 physical-control
subband를 사용한다. v3는 125 Hz octave 전체(88.388 Hz부터)를 별도 계약으로
강제하므로, 두 결과를 섞거나 자동 승격하지 않기 위해 별도 evaluator를 둔다.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..dsp.control_band_contract import BroadbandFullOctaveContractV3
from .metrics import band_nmse_db, band_power
from .trusted_subbands import cluster_bootstrap_ci


FULL_OCTAVE_V3_SURROGATE_EVAL_SCHEMA = "full_octave_v3_surrogate_matched_eval_v1"
FULL_OCTAVE_V3_EVAL_DOMAIN = "surrogate_matched_causal_ps_not_physical"
MIN_TARGET_D_DENSITY_RATIO = 0.25
MIN_GROUPS_PER_FAMILY_OCTAVE = 4
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260829

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FullOctaveV3MatchedSegment:
    """같은 zero-reset prefix와 같은 causal P/S에서 얻은 target crop 하나.

    ``error_fxlms``는 반드시 Deep-ANC와 같은 controller reference, P(n), S(y),
    prefix length 및 block size를 사용해 생성돼야 한다. 이 객체는 그 사실을
    증명하는 artifact가 아니라, evaluator가 다른 binding을 섞지 않도록 하는
    in-memory receipt다.
    """

    session_id: str
    source_family: str
    group_id: str
    error_position_id: str
    sample_rate: int
    disturbance_off: np.ndarray
    error_deep_anc: np.ndarray
    error_fxlms: np.ndarray
    causal_plant_binding_sha256: str
    evaluation_domain: str = FULL_OCTAVE_V3_EVAL_DOMAIN


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
    segment: FullOctaveV3MatchedSegment,
    *,
    contract: BroadbandFullOctaveContractV3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for field in ("session_id", "source_family", "group_id", "error_position_id"):
        if not str(getattr(segment, field)).strip():
            raise ValueError(f"full-octave segment {field}가 비었습니다")
    if segment.source_family not in contract.source_families:
        raise ValueError(f"지원하지 않는 source family: {segment.source_family!r}")
    if int(segment.sample_rate) != int(contract.sample_rate):
        raise ValueError("full-octave matched segment는 48 kHz여야 합니다")
    if segment.evaluation_domain != FULL_OCTAVE_V3_EVAL_DOMAIN:
        raise ValueError("v3 evaluator는 physical/legacy 결과를 surrogate matched 결과로 섞지 않습니다")
    if not _SHA256.fullmatch(str(segment.causal_plant_binding_sha256)):
        raise ValueError("causal plant binding SHA는 lowercase SHA-256이어야 합니다")
    values = tuple(
        np.asarray(raw, dtype=np.float64).reshape(-1)
        for raw in (
            segment.disturbance_off,
            segment.error_deep_anc,
            segment.error_fxlms,
        )
    )
    if min(value.size for value in values) < 256:
        raise ValueError("full-octave matched segment가 256 samples보다 짧습니다")
    if len({value.size for value in values}) != 1:
        raise ValueError("OFF/Deep-ANC/FxLMS matched segment 길이가 다릅니다")
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("full-octave matched segment에 NaN/Inf가 있습니다")
    return values


def _segment_rows(
    segment: FullOctaveV3MatchedSegment,
    *,
    contract: BroadbandFullOctaveContractV3,
) -> list[dict[str, Any]]:
    disturbance, error_dl, error_fxlms = _validate_segment(segment, contract=contract)
    bands = tuple(contract.equal_weight_octave_objective_bands_hz)
    total_power = sum(
        band_power(
            disturbance,
            contract.sample_rate,
            band,
            include_upper=index == len(bands) - 1,
        )
        for index, band in enumerate(bands)
    )
    total_width = sum(float(hi - lo) for lo, hi in bands)
    rows: list[dict[str, Any]] = []
    for index, band in enumerate(bands):
        include_upper = index == len(bands) - 1
        power = band_power(
            disturbance,
            contract.sample_rate,
            band,
            include_upper=include_upper,
        )
        flat_fraction = (float(band[1]) - float(band[0])) / total_width
        density = (
            0.0
            if total_power <= np.finfo(np.float64).tiny
            else float((power / total_power) / flat_fraction)
        )
        deep_attenuation = -band_nmse_db(
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
                "octave_index": index,
                "octave_center_hz": float(contract.octave_objective_centers_hz[index]),
                "octave_band_hz": [float(band[0]), float(band[1])],
                "target_density_ratio": density,
                "covered": density >= MIN_TARGET_D_DENSITY_RATIO,
                "attenuation_deep_anc_db": deep_attenuation,
                "attenuation_fxlms_db": fxlms_attenuation,
                "paired_delta_deep_anc_minus_fxlms_db": (
                    deep_attenuation - fxlms_attenuation
                ),
            }
        )
    return rows


def _serial(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _aggregate_cell(
    rows: Sequence[dict[str, Any]],
    *,
    requires_fxlms_superiority: bool,
    minimum_groups: int,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    covered = [row for row in rows if bool(row["covered"])]
    groups = np.asarray([str(row["group_id"]) for row in covered])
    deep = np.asarray(
        [float(row["attenuation_deep_anc_db"]) for row in covered], dtype=np.float64
    )
    delta = np.asarray(
        [float(row["paired_delta_deep_anc_minus_fxlms_db"]) for row in covered],
        dtype=np.float64,
    )
    unique_groups = len(set(groups.tolist())) if groups.size else 0
    if deep.size:
        deep_mean = float(np.mean(deep))
        deep_worst10 = _worst10_mean(deep)
        deep_ci_lo, deep_ci_hi, _ = cluster_bootstrap_ci(
            deep,
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
        deep_mean = deep_worst10 = deep_ci_lo = deep_ci_hi = float("nan")
        delta_mean = delta_worst10 = delta_ci_lo = delta_ci_hi = float("nan")

    coverage_pass = unique_groups >= minimum_groups
    attenuation_pass = bool(
        coverage_pass
        and all(math.isfinite(value) for value in (deep_mean, deep_worst10, deep_ci_lo))
        and deep_mean > 0.0
        and deep_worst10 > 0.0
        and deep_ci_lo > 0.0
    )
    superiority_pass = bool(
        not requires_fxlms_superiority
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
    return {
        "n_segments_total": len(rows),
        "n_segments_covered": len(covered),
        "n_independent_groups_covered": unique_groups,
        "coverage_pass": coverage_pass,
        "deep_anc_attenuation_mean_db": _serial(deep_mean),
        "deep_anc_attenuation_worst10_mean_db": _serial(deep_worst10),
        "deep_anc_attenuation_cluster_ci95_db": [_serial(deep_ci_lo), _serial(deep_ci_hi)],
        "positive_attenuation_pass": attenuation_pass,
        "paired_delta_mean_db": _serial(delta_mean),
        "paired_delta_worst10_mean_db": _serial(delta_worst10),
        "paired_delta_cluster_ci95_db": [_serial(delta_ci_lo), _serial(delta_ci_hi)],
        "matched_fxlms_superiority_required": requires_fxlms_superiority,
        "matched_fxlms_superiority_pass": superiority_pass,
        "passed": attenuation_pass and superiority_pass,
    }


def evaluate_full_octave_v3_matched_segments(
    segments: Sequence[FullOctaveV3MatchedSegment],
    *,
    contract: BroadbandFullOctaveContractV3,
    minimum_groups: int = MIN_GROUPS_PER_FAMILY_OCTAVE,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """동일 causal P/S 및 prefix의 surrogate Deep-ANC/FxLMS를 7 octave로 판정한다.

    2/4/8 kHz octave(index 4--6)는 positive attenuation뿐 아니라 FxLMS보다의
    paired mean/worst-10/cluster-CI 우위가 모두 필요하다. 저역을 평균내 고역 실패를
    숨기거나, FxLMS의 다른 P/S·reference를 가져와 비교하는 길은 허용하지 않는다.
    """

    canonical = BroadbandFullOctaveContractV3.canonical()
    if contract != canonical:
        raise ValueError("exact canonical full-octave v3 contract가 필요합니다")
    minimum = int(minimum_groups)
    if minimum < MIN_GROUPS_PER_FAMILY_OCTAVE:
        raise ValueError("family×octave 독립 group 하한을 4보다 낮출 수 없습니다")
    if int(n_resamples) <= 0:
        raise ValueError("bootstrap resample 수는 양수여야 합니다")
    if not segments:
        raise ValueError("평가할 full-octave matched segment가 없습니다")

    binding_shas = {str(segment.causal_plant_binding_sha256) for segment in segments}
    if len(binding_shas) != 1:
        raise ValueError("Deep-ANC/FxLMS surrogate 비교에는 하나의 exact causal P/S binding이 필요합니다")

    seen_session_position: set[tuple[str, str]] = set()
    raw_rows: list[dict[str, Any]] = []
    positions: set[str] = set()
    for segment in segments:
        key = (segment.session_id, segment.error_position_id)
        if key in seen_session_position:
            raise ValueError("같은 session/ERR position segment가 중복됐습니다")
        seen_session_position.add(key)
        positions.add(segment.error_position_id)
        for row in _segment_rows(segment, contract=canonical):
            raw_rows.append(
                {
                    "session_id": segment.session_id,
                    "source_family": segment.source_family,
                    "group_id": segment.group_id,
                    "error_position_id": segment.error_position_id,
                    **row,
                }
            )

    # 이 consumer는 하나의 surrogate error point만 다룬다. 다점 quiet-zone은 실제
    # simultaneous ERR witness와 별도 physical evaluator가 생길 때까지 승격하지 않는다.
    if len(positions) != 1:
        raise ValueError("v3 surrogate evaluator는 error position 하나만 받아야 합니다")
    position = next(iter(positions))
    cells: list[dict[str, Any]] = []
    reasons: list[str] = []
    for family in canonical.source_families:
        for octave_index, band in enumerate(canonical.equal_weight_octave_objective_bands_hz):
            selected = [
                row
                for row in raw_rows
                if row["source_family"] == family and row["octave_index"] == octave_index
            ]
            aggregate = _aggregate_cell(
                selected,
                requires_fxlms_superiority=octave_index >= 4,
                minimum_groups=minimum,
                n_resamples=int(n_resamples),
                seed=int(seed) + octave_index,
            )
            cell = {
                "error_position_id": position,
                "source_family": family,
                "octave_center_hz": float(canonical.octave_objective_centers_hz[octave_index]),
                "octave_band_hz": [float(band[0]), float(band[1])],
                **aggregate,
            }
            cells.append(cell)
            if not cell["passed"]:
                reasons.append(
                    f"{position}/{family}/{cell['octave_center_hz']:.0f}Hz octave가 "
                    "coverage/positive attenuation/FxLMS superiority 중 하나를 통과하지 못했습니다"
                )

    payload: dict[str, Any] = {
        "schema": FULL_OCTAVE_V3_SURROGATE_EVAL_SCHEMA,
        "role": FULL_OCTAVE_V3_EVAL_DOMAIN,
        "status": "PASS" if not reasons else "BLOCKED",
        "canonical_training_or_physical_g4_claim": False,
        "control_band_contract_sha256": canonical.digest(),
        "causal_plant_binding_sha256": next(iter(binding_shas)),
        "minimum_target_d_density_ratio": MIN_TARGET_D_DENSITY_RATIO,
        "minimum_groups_per_family_octave": minimum,
        "bootstrap_resamples": int(n_resamples),
        "bootstrap_seed": int(seed),
        "error_position_id": position,
        "cells": cells,
        "reasons": reasons,
        "limitations": [
            "동일 causal P/S surrogate의 비교일 뿐 실제 덕트 ON/OFF raw가 아닙니다",
            "latency/xrun/deadline/runtime stability와 physical G4는 별도 raw-first gate입니다",
            "실제 multi-position quiet zone은 simultaneous ERR witness가 있는 physical evaluator가 필요합니다",
        ],
    }
    payload["evidence_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "FULL_OCTAVE_V3_EVAL_DOMAIN",
    "FULL_OCTAVE_V3_SURROGATE_EVAL_SCHEMA",
    "FullOctaveV3MatchedSegment",
    "MIN_GROUPS_PER_FAMILY_OCTAVE",
    "MIN_TARGET_D_DENSITY_RATIO",
    "evaluate_full_octave_v3_matched_segments",
]
