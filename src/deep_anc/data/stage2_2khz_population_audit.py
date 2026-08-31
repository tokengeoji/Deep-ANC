"""Stage-2 2 kHz recorded/public population의 byte-grounded 감사.

기존 Stage-1 coverage/report를 수정하거나 승격하지 않는다. canonical 82-session
holdout의 lineage를 기존 strict validator로 다시 유도한 뒤, 각 session의 실제
``source_aligned.wav``와 ``mics.wav`` bytes에서 다음을 독립 계산한다.

* 125/250/500/1000/2000 Hz octave source-density;
* 실제 ERR target density와 source→ERR coherence의 joint-valid coverage;
* split×family×octave 독립 lineage component 4개 하한;
* 과거 요약의 1600--2828.427 Hz joint-valid component 수;
* 한 신규 component가 여러 부족 octave를 채우는 최소 recording-slot 계획.

public manifest는 현재 경로의 실제 bytes와 lineage 필드 완전성을 inventory한다.
없는 원격 bytes나 lineage 없는 legacy manifest를 추정으로 PASS시키지 않는다.
이 모듈은 오디오 장치, 네트워크 또는 GPU를 사용하지 않는다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import coherence, welch

from ..dsp.control_band_contract import ControlBandContract
from ..dsp.stage2_2khz_contract import Stage2TwoKilohertzContract
from .holdout_contract import validate_holdout_contract


STAGE2_POPULATION_AUDIT_SCHEMA = "stage2_2khz_population_byte_audit_v3"
MIN_COHERENCE = 0.60
ONE_POINT_SIX_KHZ_SENTINEL_CENTER_HZ = 1600.0
ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ = (1425.437949, 1795.939277)
DEFAULT_START_SECONDS = 5.0
DEFAULT_STOP_SECONDS = 65.0
DEFAULT_NPERSEG = 8192
DEFAULT_NOVERLAP = 4096
REQUIRED_SPLITS = ("train", "val", "test")
PUBLIC_MANIFEST_RELATIVE_PATHS = (
    "data/manifests/speech.jsonl",
    "data/manifests/music.jsonl",
    "data/manifests/esc50.jsonl",
    "data/manifests/canonical_v4/demand.jsonl",
)


class Stage2PopulationAuditError(ValueError):
    """입력 bytes/schema가 fail-closed 감사를 수행할 수 없을 때 발생한다."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_snapshot(path: Path, *, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise Stage2PopulationAuditError(f"repository 밖 파일은 감사할 수 없습니다: {resolved}") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise Stage2PopulationAuditError(f"regular non-symlink file이 아닙니다: {resolved}")
    return {
        "path": relative.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage2PopulationAuditError(f"JSON object key가 중복됐습니다: {key!r}")
        result[key] = value
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise Stage2PopulationAuditError(f"manifest를 읽을 수 없습니다: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_json_no_duplicates)
        except json.JSONDecodeError as exc:
            raise Stage2PopulationAuditError(
                f"manifest JSON 오류: {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise Stage2PopulationAuditError(f"manifest row가 object가 아닙니다: {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise Stage2PopulationAuditError(f"manifest가 비었습니다: {path}")
    return rows


def _mono_float64(raw: np.ndarray, *, channel: int | None = None) -> np.ndarray:
    values = np.asarray(raw)
    if values.ndim == 2:
        if channel is None:
            if values.shape[1] == 1:
                values = values[:, 0]
            else:
                values = np.mean(values, axis=1)
        else:
            if not 0 <= int(channel) < values.shape[1]:
                raise Stage2PopulationAuditError(
                    f"요청 channel {channel}이 audio shape {values.shape}에 없습니다"
                )
            values = values[:, int(channel)]
    elif values.ndim != 1:
        raise Stage2PopulationAuditError(f"audio shape은 1D/2D여야 합니다: {values.shape}")
    result = np.asarray(values, dtype=np.float64)
    if np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(values.dtype)
        result = result / float(max(abs(int(info.min)), abs(int(info.max))))
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise Stage2PopulationAuditError("audio가 비었거나 NaN/Inf를 포함합니다")
    return result


def _density_ratios(
    frequency: np.ndarray,
    psd: np.ndarray,
    bands_hz: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    bands = tuple(tuple(float(value) for value in band) for band in bands_hz)
    if not bands:
        raise Stage2PopulationAuditError("density band가 비었습니다")
    total = (frequency >= bands[0][0]) & (frequency < bands[-1][1])
    if not np.any(total):
        raise Stage2PopulationAuditError("density 전체 구간에 Welch bin이 없습니다")
    baseline = float(np.mean(psd[total]))
    if not math.isfinite(baseline):
        raise Stage2PopulationAuditError("density baseline이 finite하지 않습니다")
    values: list[float] = []
    for lower, upper in bands:
        selected = (frequency >= lower) & (frequency < upper)
        if not np.any(selected):
            raise Stage2PopulationAuditError(
                f"density band [{lower:g}, {upper:g})에 Welch bin이 없습니다"
            )
        density = float(np.mean(psd[selected]))
        ratio = (
            0.0
            if baseline <= np.finfo(np.float64).tiny
            else float(density / baseline)
        )
        if not math.isfinite(ratio) or ratio < 0.0:
            raise Stage2PopulationAuditError("density ratio가 finite한 0 이상이 아닙니다")
        values.append(ratio)
    return tuple(values)


def _density_ratio_against_objective_baseline(
    frequency: np.ndarray,
    psd: np.ndarray,
    *,
    band_hz: Sequence[float],
    objective_bands_hz: Sequence[Sequence[float]],
) -> float:
    """한 auxiliary band의 mean PSD를 전체 objective mean PSD로 정규화한다."""

    if len(band_hz) != 2 or not objective_bands_hz:
        raise Stage2PopulationAuditError("sentinel/objective band가 비었습니다")
    lower, upper = (float(value) for value in band_hz)
    baseline_lower = float(objective_bands_hz[0][0])
    baseline_upper = float(objective_bands_hz[-1][1])
    if not baseline_lower <= lower < upper <= baseline_upper:
        raise Stage2PopulationAuditError("sentinel이 objective baseline 밖입니다")
    selected = (frequency >= lower) & (frequency < upper)
    baseline = (frequency >= baseline_lower) & (frequency < baseline_upper)
    if not np.any(selected) or not np.any(baseline):
        raise Stage2PopulationAuditError("sentinel/baseline에 Welch bin이 없습니다")
    baseline_mean = float(np.mean(psd[baseline]))
    band_mean = float(np.mean(psd[selected]))
    value = (
        0.0
        if baseline_mean <= np.finfo(np.float64).tiny
        else float(band_mean / baseline_mean)
    )
    if not math.isfinite(value) or value < 0.0:
        raise Stage2PopulationAuditError("sentinel density ratio가 finite한 0 이상이 아닙니다")
    return value


def measure_stage2_recorded_signals(
    source: np.ndarray,
    target_err: np.ndarray,
    *,
    sample_rate: int,
    contract: Stage2TwoKilohertzContract,
    nperseg: int = DEFAULT_NPERSEG,
    noverlap: int = DEFAULT_NOVERLAP,
) -> dict[str, Any]:
    """한 recorded crop의 source/ERR density와 coherence를 한 FFT 경로에서 계산한다."""

    if contract != Stage2TwoKilohertzContract.canonical():
        raise Stage2PopulationAuditError("exact canonical Stage-2 2 kHz contract가 필요합니다")
    src = _mono_float64(source)
    err = _mono_float64(target_err)
    if src.size != err.size:
        raise Stage2PopulationAuditError("source/ERR crop 길이가 다릅니다")
    if int(sample_rate) != contract.sample_rate:
        raise Stage2PopulationAuditError("recorded crop은 48 kHz여야 합니다")
    nper = int(nperseg)
    overlap = int(noverlap)
    if src.size < nper or nper < 256 or not 0 <= overlap < nper:
        raise Stage2PopulationAuditError("Welch/coherence window 설정이 잘못됐습니다")

    frequency, source_psd = welch(
        src, fs=sample_rate, nperseg=nper, noverlap=overlap, detrend=False
    )
    target_frequency, target_psd = welch(
        err, fs=sample_rate, nperseg=nper, noverlap=overlap, detrend=False
    )
    coherence_frequency, coherence_values = coherence(
        src, err, fs=sample_rate, nperseg=nper, noverlap=overlap, detrend=False
    )
    if not np.array_equal(frequency, target_frequency) or not np.array_equal(
        frequency, coherence_frequency
    ):
        raise Stage2PopulationAuditError("Welch/coherence frequency grid가 다릅니다")
    if not np.all(np.isfinite(source_psd)) or not np.all(np.isfinite(target_psd)):
        raise Stage2PopulationAuditError("Welch PSD에 NaN/Inf가 있습니다")

    objective_bands = contract.octave_objective_bands_hz
    source_density = _density_ratios(frequency, source_psd, objective_bands)
    target_density = _density_ratios(frequency, target_psd, objective_bands)
    objective_coherence: list[float] = []
    for lower, upper in objective_bands:
        selected = (frequency >= lower) & (frequency < upper)
        value = float(np.median(coherence_values[selected]))
        if not math.isfinite(value):
            raise Stage2PopulationAuditError("objective coherence가 finite하지 않습니다")
        objective_coherence.append(value)

    sentinel_band = ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ
    sentinel_selected = (frequency >= sentinel_band[0]) & (
        frequency < sentinel_band[1]
    )
    sentinel_source_density = _density_ratio_against_objective_baseline(
        frequency,
        source_psd,
        band_hz=sentinel_band,
        objective_bands_hz=objective_bands,
    )
    sentinel_target_density = _density_ratio_against_objective_baseline(
        frequency,
        target_psd,
        band_hz=sentinel_band,
        objective_bands_hz=objective_bands,
    )
    sentinel_coherence = float(np.median(coherence_values[sentinel_selected]))
    if not math.isfinite(sentinel_coherence):
        raise Stage2PopulationAuditError("1.6 kHz sentinel coherence가 finite하지 않습니다")

    # 과거 1600--2828 요약은 150--11.314 kHz 전체 PSD baseline을 썼다.
    # Stage-2 objective denominator로 다시 정의하면 '3개'를 독립 재검산한 것이
    # 아니므로, 기존 v2 physical grid도 같은 Welch bytes에서 별도로 계산한다.
    legacy = ControlBandContract.broadband_point_control()
    legacy_density = _density_ratios(
        frequency, target_psd, legacy.point_control_subbands_hz
    )
    legacy_coherence: list[float] = []
    for lower, upper in legacy.point_control_subbands_hz:
        selected = (frequency >= lower) & (frequency < upper)
        value = float(np.median(coherence_values[selected]))
        if not math.isfinite(value):
            raise Stage2PopulationAuditError("legacy panel coherence가 finite하지 않습니다")
        legacy_coherence.append(value)

    threshold = contract.minimum_source_density_ratio
    source_pass = tuple(value >= threshold for value in source_density)
    target_pass = tuple(value >= threshold for value in target_density)
    coherence_pass = tuple(value >= MIN_COHERENCE for value in objective_coherence)
    sentinel_source_pass = sentinel_source_density >= threshold
    sentinel_target_pass = sentinel_target_density >= threshold
    sentinel_coherence_pass = sentinel_coherence >= MIN_COHERENCE
    return {
        "source_density_ratio": source_density,
        "source_density_pass": source_pass,
        "target_err_density_ratio": target_density,
        "target_err_density_pass": target_pass,
        "source_err_coherence": tuple(objective_coherence),
        "source_err_coherence_pass": coherence_pass,
        "target_path_joint_valid": tuple(
            density and coh
            for density, coh in zip(target_pass, coherence_pass, strict=True)
        ),
        "population_joint_valid": tuple(
            source_ok and target_ok and coh_ok
            for source_ok, target_ok, coh_ok in zip(
                source_pass, target_pass, coherence_pass, strict=True
            )
        ),
        "one_point_six_khz_sentinel_source_density_ratio": (
            sentinel_source_density,
        ),
        "one_point_six_khz_sentinel_source_density_pass": (
            sentinel_source_pass,
        ),
        "one_point_six_khz_sentinel_target_err_density_ratio": (
            sentinel_target_density,
        ),
        "one_point_six_khz_sentinel_target_err_density_pass": (
            sentinel_target_pass,
        ),
        "one_point_six_khz_sentinel_source_err_coherence": (
            sentinel_coherence,
        ),
        "one_point_six_khz_sentinel_source_err_coherence_pass": (
            sentinel_coherence_pass,
        ),
        "one_point_six_khz_sentinel_target_path_joint_valid": (
            sentinel_target_pass and sentinel_coherence_pass,
        ),
        "one_point_six_khz_sentinel_population_joint_valid": (
            sentinel_source_pass
            and sentinel_target_pass
            and sentinel_coherence_pass,
        ),
        "legacy_physical_target_density_ratio": legacy_density,
        "legacy_physical_source_err_coherence": tuple(legacy_coherence),
        "legacy_physical_joint_valid": tuple(
            density >= threshold and coh >= MIN_COHERENCE
            for density, coh in zip(legacy_density, legacy_coherence, strict=True)
        ),
    }


def build_stage2_coverage_cells(
    sessions: Sequence[Mapping[str, Any]],
    *,
    contract: Stage2TwoKilohertzContract,
) -> list[dict[str, Any]]:
    """모든 split×family×objective cell을 누락 없이 fail-closed 집계한다."""

    if contract != Stage2TwoKilohertzContract.canonical():
        raise Stage2PopulationAuditError("exact canonical Stage-2 contract가 필요합니다")
    rows = tuple(sessions)
    cells: list[dict[str, Any]] = []
    for split in REQUIRED_SPLITS:
        for family in contract.source_families:
            selected = [
                row
                for row in rows
                if row.get("split") == split and row.get("source_family") == family
            ]
            for index, (center, band) in enumerate(
                zip(
                    contract.octave_objective_centers_hz,
                    contract.octave_objective_bands_hz,
                    strict=True,
                )
            ):
                source_sessions = [
                    str(row["session_id"])
                    for row in selected
                    if bool(row["source_density_pass"][index])
                ]
                source_groups = sorted(
                    {
                        str(row["group_id"])
                        for row in selected
                        if bool(row["source_density_pass"][index])
                    }
                )
                target_joint_sessions = [
                    str(row["session_id"])
                    for row in selected
                    if bool(row["target_path_joint_valid"][index])
                ]
                target_joint_groups = sorted(
                    {
                        str(row["group_id"])
                        for row in selected
                        if bool(row["target_path_joint_valid"][index])
                    }
                )
                population_sessions = [
                    str(row["session_id"])
                    for row in selected
                    if bool(row["population_joint_valid"][index])
                ]
                population_groups = sorted(
                    {
                        str(row["group_id"])
                        for row in selected
                        if bool(row["population_joint_valid"][index])
                    }
                )
                minimum = contract.minimum_groups_per_family_octave
                cells.append(
                    {
                        "split": split,
                        "source_family": family,
                        "octave_index": index,
                        "octave_center_hz": float(center),
                        "octave_band_hz": [float(band[0]), float(band[1])],
                        "total_sessions": len(selected),
                        "source_density_valid_sessions": len(source_sessions),
                        "source_density_independent_groups": len(source_groups),
                        "source_density_group_ids": source_groups,
                        "source_density_pass": len(source_groups) >= minimum,
                        "source_density_component_deficit": max(
                            0, minimum - len(source_groups)
                        ),
                        "target_path_joint_valid_sessions": len(target_joint_sessions),
                        "target_path_joint_independent_groups": len(target_joint_groups),
                        "target_path_joint_group_ids": target_joint_groups,
                        "target_path_joint_pass": len(target_joint_groups) >= minimum,
                        "target_path_joint_component_deficit": max(
                            0, minimum - len(target_joint_groups)
                        ),
                        "population_joint_valid_sessions": len(population_sessions),
                        "population_joint_independent_groups": len(population_groups),
                        "population_joint_group_ids": population_groups,
                        "population_joint_pass": len(population_groups) >= minimum,
                        "population_joint_component_deficit": max(
                            0, minimum - len(population_groups)
                        ),
                        "minimum_independent_groups_required": minimum,
                    }
                )
    return cells


def build_stage2_sentinel_cells(
    sessions: Sequence[Mapping[str, Any]],
    *,
    contract: Stage2TwoKilohertzContract,
) -> list[dict[str, Any]]:
    """split×family 1.6 kHz sentinel source/ERR/coherence joint coverage를 센다."""

    if contract != Stage2TwoKilohertzContract.canonical():
        raise Stage2PopulationAuditError("exact canonical Stage-2 contract가 필요합니다")
    rows = tuple(sessions)
    cells: list[dict[str, Any]] = []
    source_key = "one_point_six_khz_sentinel_source_density_pass"
    target_key = "one_point_six_khz_sentinel_target_path_joint_valid"
    population_key = "one_point_six_khz_sentinel_population_joint_valid"
    for split in REQUIRED_SPLITS:
        for family in contract.source_families:
            selected = [
                row
                for row in rows
                if row.get("split") == split and row.get("source_family") == family
            ]
            for key in (source_key, target_key, population_key):
                if any(
                    not isinstance(row.get(key), (tuple, list))
                    or len(row[key]) != 1
                    for row in selected
                ):
                    raise Stage2PopulationAuditError(
                        f"1.6 kHz sentinel session field가 불완전합니다: {split}/{family}/{key}"
                    )

            def identities(key: str) -> tuple[list[str], list[str]]:
                sessions_valid = [
                    str(row["session_id"])
                    for row in selected
                    if bool(row[key][0])
                ]
                groups_valid = sorted(
                    {
                        str(row["group_id"])
                        for row in selected
                        if bool(row[key][0])
                    }
                )
                return sessions_valid, groups_valid

            source_sessions, source_groups = identities(source_key)
            target_sessions, target_groups = identities(target_key)
            population_sessions, population_groups = identities(population_key)
            minimum = contract.minimum_groups_per_family_octave
            cells.append(
                {
                    "split": split,
                    "source_family": family,
                    "sentinel_center_hz": ONE_POINT_SIX_KHZ_SENTINEL_CENTER_HZ,
                    "sentinel_band_hz": [
                        float(ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ[0]),
                        float(ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ[1]),
                    ],
                    "total_sessions": len(selected),
                    "source_density_valid_sessions": len(source_sessions),
                    "source_density_independent_groups": len(source_groups),
                    "source_density_group_ids": source_groups,
                    "source_density_pass": len(source_groups) >= minimum,
                    "source_density_component_deficit": max(
                        0, minimum - len(source_groups)
                    ),
                    "target_path_joint_valid_sessions": len(target_sessions),
                    "target_path_joint_independent_groups": len(target_groups),
                    "target_path_joint_group_ids": target_groups,
                    "target_path_joint_pass": len(target_groups) >= minimum,
                    "target_path_joint_component_deficit": max(
                        0, minimum - len(target_groups)
                    ),
                    "population_joint_valid_sessions": len(population_sessions),
                    "population_joint_independent_groups": len(population_groups),
                    "population_joint_group_ids": population_groups,
                    "population_joint_pass": len(population_groups) >= minimum,
                    "population_joint_component_deficit": max(
                        0, minimum - len(population_groups)
                    ),
                    "minimum_independent_groups_required": minimum,
                }
            )
    return cells


def plan_minimum_multioctave_additions(
    cells: Sequence[Mapping[str, Any]],
    *,
    contract: Stage2TwoKilohertzContract,
    sentinel_cells: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """한 신규 component가 여러 octave를 채운다는 최선 조건의 최소 slot 계획."""

    plans: list[dict[str, Any]] = []
    total_slots = 0
    for split in REQUIRED_SPLITS:
        for family in contract.source_families:
            selected = [
                row
                for row in cells
                if row.get("split") == split and row.get("source_family") == family
            ]
            if len(selected) != len(contract.octave_objective_centers_hz):
                raise Stage2PopulationAuditError(
                    f"candidate plan cell이 누락됐습니다: {split}/{family}"
                )
            deficits = {
                float(row["octave_center_hz"]): int(
                    row["population_joint_component_deficit"]
                )
                for row in selected
            }
            sentinel_deficit = 0
            if sentinel_cells is not None:
                sentinel_selected = [
                    row
                    for row in sentinel_cells
                    if row.get("split") == split
                    and row.get("source_family") == family
                ]
                if len(sentinel_selected) != 1:
                    raise Stage2PopulationAuditError(
                        f"sentinel candidate plan cell이 누락됐습니다: {split}/{family}"
                    )
                sentinel_deficit = int(
                    sentinel_selected[0]["population_joint_component_deficit"]
                )
                if not 0 <= sentinel_deficit <= contract.minimum_groups_per_family_octave:
                    raise Stage2PopulationAuditError("sentinel component deficit이 유효하지 않습니다")
            slots = max((*deficits.values(), sentinel_deficit), default=0)
            total_slots += slots
            requirements: list[dict[str, Any]] = []
            for slot in range(1, slots + 1):
                centers = sorted(
                    center for center, deficit in deficits.items() if deficit >= slot
                )
                requirements.append(
                    {
                        "slot_id": f"{split}-{family}-new-lineage-{slot:02d}",
                        "required_objective_octaves_hz": centers,
                        "one_point_six_khz_sentinel_required": (
                            sentinel_deficit >= slot
                        ),
                        "candidate_identity": None,
                        "assignment_status": "UNASSIGNED_NEW_LINEAGE_REQUIRED",
                        "must_be_independent_of_existing_and_other_slots": True,
                        "conditioning_allowed": split == "train",
                        "untouched_natural_unseen_required": split == "test",
                        "training_or_model_selection_use_allowed": split != "test",
                    }
                )
            plans.append(
                {
                    "split": split,
                    "source_family": family,
                    "deficit_by_octave_hz": {
                        f"{center:g}": deficits[center] for center in sorted(deficits)
                    },
                    "one_point_six_khz_sentinel_component_deficit": sentinel_deficit,
                    "minimum_new_components_if_each_slot_covers_all_listed_octaves": slots,
                    "slots": requirements,
                }
            )
    return {
        "method": "deterministic_deficit_layering_set_cover_lower_bound_v1",
        "minimum_new_recording_slots_lower_bound": total_slots,
        "attainability_condition": (
            "각 slot의 새 독립 lineage source가 그 slot에 지정된 모든 objective octave와 "
            "요구된 1.6 kHz sentinel에서 "
            "source-density, actual ERR-density, source→ERR coherence를 동시에 통과해야 합니다"
        ),
        "plans": plans,
        "final_unseen_policy": {
            "test_slots_are_untouched_natural_sources": True,
            "conditioned_training_stimulus_may_not_fill_test_slots": True,
            "test_source_training_or_model_selection_use_allowed": False,
            "reservation_status": "BLOCKED_UNASSIGNED",
        },
    }


def audit_stage2_lineage_assignments(
    assignments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """component/composite WAV/original clip의 split crossing을 exact 집계한다."""

    component_splits: dict[str, set[str]] = defaultdict(set)
    composite_sha_splits: dict[str, set[str]] = defaultdict(set)
    composite_path_splits: dict[str, set[str]] = defaultdict(set)
    clip_splits: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(assignments):
        split = str(row.get("split") or "")
        component = str(row.get("group_id") or "")
        wav_sha = str(row.get("source_pool_wav_sha256") or "")
        wav_path = str(row.get("source_pool_wav_path") or "")
        clips = row.get("original_clips")
        if (
            split not in REQUIRED_SPLITS
            or not component
            or len(wav_sha) != 64
            or any(value not in "0123456789abcdef" for value in wav_sha)
            or not wav_path
            or not isinstance(clips, (tuple, list))
            or not clips
            or not all(isinstance(value, str) and value for value in clips)
        ):
            raise Stage2PopulationAuditError(
                f"lineage assignment #{index}가 불완전합니다"
            )
        component_splits[component].add(split)
        composite_sha_splits[wav_sha].add(split)
        composite_path_splits[wav_path].add(split)
        for clip in clips:
            clip_splits[str(clip).casefold()].add(split)

    def crossings(values: Mapping[str, set[str]]) -> list[list[Any]]:
        return [
            [name, sorted(splits)]
            for name, splits in sorted(values.items())
            if len(splits) > 1
        ]

    component_crossings = crossings(component_splits)
    composite_sha_crossings = crossings(composite_sha_splits)
    composite_path_crossings = crossings(composite_path_splits)
    clip_crossings = crossings(clip_splits)
    passed = not (
        component_crossings
        or composite_sha_crossings
        or composite_path_crossings
        or clip_crossings
    )
    return {
        "status": "PASS" if passed else "BLOCKED",
        "component_cross_split": component_crossings,
        "composite_wav_sha_cross_split": composite_sha_crossings,
        "composite_wav_path_cross_split": composite_path_crossings,
        "original_clip_cross_split": clip_crossings,
        "cross_split_counts": {
            "component": len(component_crossings),
            "composite_wav_sha": len(composite_sha_crossings),
            "composite_wav_path": len(composite_path_crossings),
            "original_clip": len(clip_crossings),
        },
    }


def build_stage2_data_readiness(
    *,
    cells: Sequence[Mapping[str, Any]],
    sentinel_cells: Sequence[Mapping[str, Any]] | None = None,
    lineage_audit: Mapping[str, Any],
    public_inventory: Mapping[str, Any],
    contract: Stage2TwoKilohertzContract,
) -> dict[str, Any]:
    """public scratch-pretrain과 recorded fine-tune 모집단 축을 분리한다.

    이 함수가 내는 PASS는 각 데이터 모집단 축의 byte/lineage coverage PASS일
    뿐이다. P/S, latency, 모델, optimizer 등 전체 학습 admission을 의미하지
    않는다. 특히 recorded 부족은 public scratch-pretrain 축을 차단하지 않는다.
    """

    if contract != Stage2TwoKilohertzContract.canonical():
        raise Stage2PopulationAuditError("exact canonical Stage-2 contract가 필요합니다")
    expected_keys = {
        (split, family, float(center))
        for split in REQUIRED_SPLITS
        for family in contract.source_families
        for center in contract.octave_objective_centers_hz
    }
    actual_keys: set[tuple[str, str, float]] = set()
    recorded_blockers: list[str] = []
    for row in cells:
        key = (
            str(row.get("split") or ""),
            str(row.get("source_family") or ""),
            float(row.get("octave_center_hz", math.nan)),
        )
        if key in actual_keys:
            raise Stage2PopulationAuditError(f"coverage cell이 중복됐습니다: {key}")
        actual_keys.add(key)
        deficit = int(row.get("population_joint_component_deficit", -1))
        passed = row.get("population_joint_pass")
        if deficit < 0 or not isinstance(passed, (bool, np.bool_)):
            raise Stage2PopulationAuditError(f"coverage cell 판정이 불완전합니다: {key}")
        if bool(passed) != (deficit == 0):
            raise Stage2PopulationAuditError(f"coverage cell PASS/deficit이 모순입니다: {key}")
        if deficit:
            recorded_blockers.append(
                "RECORDED_POPULATION_DEFICIT:"
                f"{key[0]}:{key[1]}:{key[2]:g}Hz:{deficit}"
            )
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise Stage2PopulationAuditError(
            f"coverage cell 집합이 canonical 60 cells와 다릅니다: "
            f"missing={missing}, unexpected={unexpected}"
        )

    if sentinel_cells is not None:
        expected_sentinel_keys = {
            (split, family)
            for split in REQUIRED_SPLITS
            for family in contract.source_families
        }
        actual_sentinel_keys: set[tuple[str, str]] = set()
        for row in sentinel_cells:
            key = (
                str(row.get("split") or ""),
                str(row.get("source_family") or ""),
            )
            if key in actual_sentinel_keys:
                raise Stage2PopulationAuditError(
                    f"1.6 kHz sentinel coverage cell이 중복됐습니다: {key}"
                )
            actual_sentinel_keys.add(key)
            deficit = int(row.get("population_joint_component_deficit", -1))
            passed = row.get("population_joint_pass")
            if deficit < 0 or not isinstance(passed, (bool, np.bool_)):
                raise Stage2PopulationAuditError(
                    f"1.6 kHz sentinel coverage cell 판정이 불완전합니다: {key}"
                )
            if bool(passed) != (deficit == 0):
                raise Stage2PopulationAuditError(
                    f"1.6 kHz sentinel coverage cell PASS/deficit이 모순입니다: {key}"
                )
            if deficit:
                recorded_blockers.append(
                    "RECORDED_SENTINEL_POPULATION_DEFICIT:"
                    f"{key[0]}:{key[1]}:1600Hz_one_third:{deficit}"
                )
        if actual_sentinel_keys != expected_sentinel_keys:
            raise Stage2PopulationAuditError(
                "1.6 kHz sentinel coverage cell 집합이 canonical 12 cells와 다릅니다"
            )

    lineage_status = str(lineage_audit.get("status") or "")
    if lineage_status not in {"PASS", "BLOCKED"}:
        raise Stage2PopulationAuditError("recorded lineage status가 유효하지 않습니다")
    if lineage_status != "PASS":
        recorded_blockers.append("RECORDED_LINEAGE_CROSS_SPLIT")

    public_status = str(public_inventory.get("status") or "")
    public_blockers_raw = public_inventory.get("blockers")
    if (
        public_status not in {"PASS", "BLOCKED"}
        or not isinstance(public_blockers_raw, list)
        or not all(isinstance(value, str) and value for value in public_blockers_raw)
    ):
        raise Stage2PopulationAuditError("public inventory status/blockers가 유효하지 않습니다")
    public_blockers = sorted(set(public_blockers_raw))
    if (public_status == "PASS") != (not public_blockers):
        raise Stage2PopulationAuditError("public inventory PASS와 blockers가 모순입니다")

    recorded_blockers = sorted(set(recorded_blockers))
    recorded_status = "PASS" if not recorded_blockers else "BLOCKED"
    return {
        "public_synthetic_scratch_pretrain": {
            "status": public_status,
            "blockers": public_blockers,
            "scope": "local_public_source_bytes_and_lineage_population_only",
            "recorded_population_required": False,
            "full_training_admission": False,
        },
        "recorded_measured_finetune": {
            "status": recorded_status,
            "blockers": recorded_blockers,
            "scope": "recorded_source_target_joint_density_and_lineage_population_only",
            "public_pretrain_population_substitutes_for_recorded_deficits": False,
            "full_training_admission": False,
        },
        "separation_invariant": (
            "recorded_measured_finetune BLOCKED must not demote an independently "
            "PASS public_synthetic_scratch_pretrain axis"
        ),
    }


def _parse_source_csvs(root: Path) -> dict[str, tuple[str, tuple[str, ...]]]:
    mapping: dict[str, tuple[str, tuple[str, ...]]] = {}
    for relative in ("data/source_pool/sources.csv", "data/source_pool_v2/sources.csv"):
        path = root / relative
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"source_family", "path", "clips"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise Stage2PopulationAuditError(f"source CSV schema가 다릅니다: {relative}")
            for number, row in enumerate(reader, start=2):
                source_path = str(row["path"]).replace("\\", "/")
                try:
                    clips_raw = json.loads(str(row["clips"]))
                except json.JSONDecodeError as exc:
                    raise Stage2PopulationAuditError(
                        f"source CSV clips JSON 오류: {relative}:{number}"
                    ) from exc
                if (
                    not source_path
                    or source_path in mapping
                    or not isinstance(clips_raw, list)
                    or not clips_raw
                    or not all(isinstance(value, str) and value for value in clips_raw)
                ):
                    raise Stage2PopulationAuditError(
                        f"source CSV row가 유효하지 않습니다: {relative}:{number}"
                    )
                mapping[source_path] = (
                    str(row["source_family"]),
                    tuple(value.casefold() for value in clips_raw),
                )
    return mapping


def _repository_source_path(value: Any, *, root: Path) -> tuple[str, Path]:
    text = str(value or "").replace("\\", "/")
    for prefix in ("data/source_pool/", "data/source_pool_v2/"):
        position = text.find(prefix)
        if position >= 0:
            relative = Path(text[position:])
            return relative.as_posix(), (root / relative).resolve(strict=True)
    raise Stage2PopulationAuditError(f"session program.file이 canonical source pool 밖입니다: {text}")


def _audit_public_manifests(root: Path, *, contract: Stage2TwoKilohertzContract) -> dict[str, Any]:
    inventories: list[dict[str, Any]] = []
    blockers: list[str] = []
    for relative in PUBLIC_MANIFEST_RELATIVE_PATHS:
        manifest = root / relative
        if not manifest.is_file():
            blockers.append(f"MISSING_MANIFEST:{relative}")
            inventories.append({"path": relative, "status": "MISSING"})
            continue
        rows = _read_jsonl(manifest)
        existing_snapshots: list[dict[str, Any]] = []
        missing = outside = decode_failures = metadata_mismatches = 0
        density_pass_counts: dict[str, list[int]] = {
            split: [0] * len(contract.octave_objective_bands_hz)
            for split in REQUIRED_SPLITS
        }
        split_counts: Counter[str] = Counter()
        lineage_complete = True
        for index, row in enumerate(rows):
            split = str(row.get("split") or "")
            split_counts[split] += 1
            if not {
                "content_sha256",
                "content_size",
                "group_id",
                "lineage_keys",
                "lineage_schema",
            }.issubset(row):
                lineage_complete = False
            raw_path = Path(str(row.get("path") or ""))
            candidate = raw_path if raw_path.is_absolute() else root / raw_path
            try:
                resolved = candidate.resolve(strict=True)
            except FileNotFoundError:
                missing += 1
                continue
            try:
                resolved.relative_to(root)
            except ValueError:
                outside += 1
                continue
            if not resolved.is_file() or resolved.is_symlink():
                missing += 1
                continue
            snapshot = _file_snapshot(resolved, root=root)
            declared_sha = row.get("content_sha256")
            declared_size = row.get("content_size")
            if declared_sha is not None and str(declared_sha) != snapshot["sha256"]:
                metadata_mismatches += 1
            if declared_size is not None and int(declared_size) != snapshot["size_bytes"]:
                metadata_mismatches += 1
            try:
                values, rate = sf.read(resolved, dtype="float64", always_2d=True)
                mono = _mono_float64(values)
                if int(rate) != int(row.get("sample_rate", rate)):
                    raise Stage2PopulationAuditError("manifest/audio sample rate 불일치")
                nper = min(DEFAULT_NPERSEG, mono.size)
                if nper < 256:
                    raise Stage2PopulationAuditError("public source가 256 samples보다 짧습니다")
                overlap = min(DEFAULT_NOVERLAP, nper - 1)
                frequency, psd = welch(
                    mono,
                    fs=int(rate),
                    nperseg=nper,
                    noverlap=overlap,
                    detrend=False,
                )
                if float(rate) / 2.0 < contract.required_excitation_upper_hz:
                    raise Stage2PopulationAuditError("public source native Nyquist가 2 kHz octave 상단 미만")
                ratios = _density_ratios(
                    frequency, psd, contract.octave_objective_bands_hz
                )
                if split in density_pass_counts:
                    for band_index, value in enumerate(ratios):
                        if value >= contract.minimum_source_density_ratio:
                            density_pass_counts[split][band_index] += 1
            except (OSError, RuntimeError, ValueError, Stage2PopulationAuditError):
                decode_failures += 1
            existing_snapshots.append(snapshot)
        exact_bytes_digest = _digest(sorted(existing_snapshots, key=lambda item: item["path"]))
        complete = (
            len(existing_snapshots) == len(rows)
            and missing == 0
            and outside == 0
            and decode_failures == 0
            and metadata_mismatches == 0
            and lineage_complete
            and set(split_counts) == set(REQUIRED_SPLITS)
        )
        if not complete:
            blockers.append(f"PUBLIC_MANIFEST_NOT_STAGE2_READY:{relative}")
        inventories.append(
            {
                "path": relative,
                "manifest_size_bytes": manifest.stat().st_size,
                "manifest_sha256": sha256_file(manifest),
                "rows": len(rows),
                "split_counts": dict(sorted(split_counts.items())),
                "existing_repository_files": len(existing_snapshots),
                "missing_files": missing,
                "outside_repository_files": outside,
                "decode_or_density_failures": decode_failures,
                "declared_content_metadata_mismatches": metadata_mismatches,
                "lineage_fields_complete": lineage_complete,
                "existing_actual_bytes_set_sha256": exact_bytes_digest,
                "source_density_pass_items_by_split_octave": {
                    split: values for split, values in density_pass_counts.items()
                },
                "independent_lineage_coverage_computable": bool(lineage_complete),
                "status": "PASS" if complete else "BLOCKED",
            }
        )
    # 현재 bundle에 DNS/MIMII canonical manifests가 없다는 사실도 명시한다.
    blockers.extend(("CANONICAL_DNS_MANIFEST_ABSENT", "CANONICAL_MIMII_MACHINE_MANIFEST_ABSENT"))
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "inventories": inventories,
        "blockers": sorted(set(blockers)),
        "authority": "local_actual_bytes_inventory_not_training_manifest_bundle",
    }


def audit_stage2_2khz_population(
    *,
    repository_root: str | Path,
    recorded_manifest_path: str | Path = "data/manifests/recorded_regrouped.jsonl",
    holdout_path: str | Path = "data/manifests/recorded_holdout.json",
    start_seconds: float = DEFAULT_START_SECONDS,
    stop_seconds: float = DEFAULT_STOP_SECONDS,
    nperseg: int = DEFAULT_NPERSEG,
    noverlap: int = DEFAULT_NOVERLAP,
) -> dict[str, Any]:
    """canonical 82 sessions와 현 source manifest의 Stage-2 준비도를 계산한다."""

    root = Path(repository_root).resolve(strict=True)
    contract = Stage2TwoKilohertzContract.canonical()
    manifest = (root / recorded_manifest_path).resolve(strict=True)
    holdout = (root / holdout_path).resolve(strict=True)
    manifest_snapshot = _file_snapshot(manifest, root=root)
    holdout_summary = validate_holdout_contract(holdout, repo_root=root)
    if int(holdout_summary["active_session_count"]) != 82:
        raise Stage2PopulationAuditError("canonical active holdout은 정확히 82 sessions여야 합니다")
    rows = _read_jsonl(manifest)
    if len(rows) != 82:
        raise Stage2PopulationAuditError("recorded_regrouped manifest는 정확히 82 rows여야 합니다")

    source_csv = _parse_source_csvs(root)
    begin_s = float(start_seconds)
    end_s = float(stop_seconds)
    if not (
        math.isfinite(begin_s)
        and math.isfinite(end_s)
        and 0.0 <= begin_s < end_s
    ):
        raise Stage2PopulationAuditError("recorded population start/stop이 잘못됐습니다")

    seen_sessions: set[str] = set()
    composite_cache: dict[str, dict[str, Any]] = {}
    lineage_assignments: list[dict[str, Any]] = []
    session_results: list[dict[str, Any]] = []
    for row in rows:
        session_id = str(row.get("session_id") or "")
        family = str(row.get("source_family") or "")
        split = str(row.get("split") or "")
        group = str(row.get("group_id") or "")
        if (
            not session_id
            or session_id in seen_sessions
            or family not in contract.source_families
            or split not in REQUIRED_SPLITS
            or not group
        ):
            raise Stage2PopulationAuditError(f"recorded manifest identity가 잘못됐습니다: {session_id!r}")
        seen_sessions.add(session_id)
        base = (manifest.parent / str(row.get("path") or "")).resolve(strict=True)
        try:
            base.relative_to(root / "data/recorded")
        except ValueError as exc:
            raise Stage2PopulationAuditError(f"recorded session path가 active root 밖입니다: {base}") from exc
        session_path = base / "session.json"
        source_path = base / "source_aligned.wav"
        mics_path = base / "mics.wav"
        session_payload = json.loads(
            session_path.read_text(encoding="utf-8"), object_pairs_hook=_json_no_duplicates
        )
        if session_payload.get("session_id") != session_id:
            raise Stage2PopulationAuditError(f"session.json ID가 manifest와 다릅니다: {session_id}")
        program = session_payload.get("program")
        if not isinstance(program, dict) or program.get("type") != "file":
            raise Stage2PopulationAuditError(f"recorded session이 file source가 아닙니다: {session_id}")
        source_relative, composite_path = _repository_source_path(
            program.get("file"), root=root
        )
        if source_relative not in source_csv:
            raise Stage2PopulationAuditError(f"source CSV에 session source가 없습니다: {source_relative}")
        source_family, clips = source_csv[source_relative]
        if source_family != family:
            raise Stage2PopulationAuditError(f"source CSV/manifest family가 다릅니다: {session_id}")
        if source_relative not in composite_cache:
            composite_cache[source_relative] = _file_snapshot(
                composite_path, root=root
            )
        composite_snapshot = composite_cache[source_relative]
        lineage_assignments.append(
            {
                "split": split,
                "group_id": group,
                "source_pool_wav_sha256": composite_snapshot["sha256"],
                "source_pool_wav_path": source_relative,
                "original_clips": clips,
            }
        )

        source_rate, source_raw = wavfile.read(source_path, mmap=True)
        mics_rate, mics_raw = wavfile.read(mics_path, mmap=True)
        if int(source_rate) != contract.sample_rate or int(mics_rate) != contract.sample_rate:
            raise Stage2PopulationAuditError(f"recorded WAV sample rate가 48 kHz가 아닙니다: {session_id}")
        begin = int(round(begin_s * contract.sample_rate))
        end = int(round(end_s * contract.sample_rate))
        if len(source_raw) < end or len(mics_raw) < end:
            raise Stage2PopulationAuditError(f"recorded WAV가 고정 population crop보다 짧습니다: {session_id}")
        source = _mono_float64(source_raw[begin:end])
        target = _mono_float64(mics_raw[begin:end], channel=0)
        measured = measure_stage2_recorded_signals(
            source,
            target,
            sample_rate=contract.sample_rate,
            contract=contract,
            nperseg=int(nperseg),
            noverlap=int(noverlap),
        )
        session_results.append(
            {
                "session_id": session_id,
                "source_family": family,
                "split": split,
                "group_id": group,
                "source_pool_group_id": str(row.get("source_pool_group_id") or ""),
                "source_pool_wav": composite_snapshot,
                "session_json": _file_snapshot(session_path, root=root),
                "source_aligned_wav": _file_snapshot(source_path, root=root),
                "mics_wav": _file_snapshot(mics_path, root=root),
                **{key: list(value) for key, value in measured.items()},
            }
        )

    lineage_result = audit_stage2_lineage_assignments(lineage_assignments)

    cells = build_stage2_coverage_cells(session_results, contract=contract)
    sentinel_cells = build_stage2_sentinel_cells(
        session_results, contract=contract
    )
    additions = plan_minimum_multioctave_additions(
        cells,
        contract=contract,
        sentinel_cells=sentinel_cells,
    )
    legacy_index = 4
    legacy_valid_sessions = [
        row["session_id"]
        for row in session_results
        if bool(row["legacy_physical_joint_valid"][legacy_index])
    ]
    legacy_valid_groups = sorted(
        {
            row["group_id"]
            for row in session_results
            if bool(row["legacy_physical_joint_valid"][legacy_index])
        }
    )
    public = _audit_public_manifests(root, contract=contract)
    readiness = build_stage2_data_readiness(
        cells=cells,
        sentinel_cells=sentinel_cells,
        lineage_audit=lineage_result,
        public_inventory=public,
        contract=contract,
    )
    public_pretrain = readiness["public_synthetic_scratch_pretrain"]
    recorded_finetune = readiness["recorded_measured_finetune"]
    recorded_pass = recorded_finetune["status"] == "PASS"
    overall_pass = recorded_pass and public_pretrain["status"] == "PASS"
    payload: dict[str, Any] = {
        "schema": STAGE2_POPULATION_AUDIT_SCHEMA,
        "role": "local_actual_bytes_data_readiness_not_training_or_performance_authority",
        "status": "PASS" if overall_pass else "BLOCKED",
        "status_scope": "combined_public_and_recorded_data_inventory_only",
        "canonical_training_data_admission": False,
        "control_band_contract": contract.model_dump(mode="json"),
        "control_band_contract_sha256": contract.digest(),
        "inputs": {
            "recorded_manifest": manifest_snapshot,
            "canonical_holdout": {
                "path": Path(holdout_path).as_posix(),
                "sha256": holdout_summary["sha256"],
                "active_session_count": holdout_summary["active_session_count"],
                "provenance_report": holdout_summary["provenance_report"],
                "provenance_report_sha256": holdout_summary[
                    "provenance_report_sha256"
                ],
                "component_membership_sha256": holdout_summary["lineage"][
                    "component_membership_sha256"
                ],
            },
        },
        "population": {
            "start_seconds": begin_s,
            "stop_seconds": end_s,
            "nperseg": int(nperseg),
            "noverlap": int(noverlap),
            "source_density_threshold": contract.minimum_source_density_ratio,
            "source_err_coherence_threshold": MIN_COHERENCE,
            "source_file": "source_aligned.wav",
            "target_file_channel": "mics.wav ch0 ERR",
        },
        "lineage_audit": {
            **lineage_result,
            "canonical_holdout_bytes_and_metadata_rederived": True,
            "dimensions_verified": [
                "same composite WAV path and content SHA",
                "same original clip",
                "FMA artist and album connected component",
                "LibriSpeech speaker and Gutenberg book connected component",
                "ESC-50 source-file component for current environment/machine recordings",
            ],
            "future_mimii_machine_session_lineage_status": (
                "BLOCKED_CANONICAL_MIMII_MANIFEST_ABSENT"
            ),
        },
        "data_readiness_axes": readiness,
        "public_scratch_pretrain_status": public_pretrain["status"],
        "public_scratch_pretrain_blockers": public_pretrain["blockers"],
        "recorded_finetune_status": recorded_finetune["status"],
        "recorded_finetune_blockers": recorded_finetune["blockers"],
        # 이전 field 소비자에게도 의미를 유지하되, public 상태와 결합하지 않는다.
        "recorded_population_status": recorded_finetune["status"],
        "coverage_cells": cells,
        "one_point_six_khz_sentinel_coverage": {
            "center_hz": ONE_POINT_SIX_KHZ_SENTINEL_CENTER_HZ,
            "band_hz": [
                float(ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ[0]),
                float(ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ[1]),
            ],
            "source_density_threshold": contract.minimum_source_density_ratio,
            "source_err_coherence_threshold": MIN_COHERENCE,
            "near_zero_performance_allowed": False,
            "cells": sentinel_cells,
        },
        "legacy_1600_2828_joint_recalculation": {
            "band_hz": [1600.0, 2828.42712474619],
            "definition": (
                "target ERR density>=0.25 relative to legacy 150--11313Hz grid "
                "and source→ERR coherence>=0.60"
            ),
            "joint_valid_sessions": legacy_valid_sessions,
            "joint_valid_session_count": len(legacy_valid_sessions),
            "joint_valid_group_ids": legacy_valid_groups,
            "joint_valid_independent_group_count": len(legacy_valid_groups),
            "historical_summary_value": 3,
            "independent_recalculation_matches_historical_summary": (
                len(legacy_valid_groups) == 3
            ),
        },
        "minimum_addition_plan": additions,
        "public_manifest_actual_bytes_inventory": public,
        "sessions": session_results,
        "limitations": [
            "source-density만으로 actual P/S 또는 ANC attenuation을 주장할 수 없습니다",
            "현재 82-session machine lineage는 ESC-50 source-file 기준이며 MIMII physical machine/session manifest는 없습니다",
            "candidate slot은 최소 개수의 요구조건이며 실제 새 source identity가 배정될 때까지 수집 완료가 아닙니다",
            "untouched natural test slot은 conditioning된 training stimulus로 채울 수 없습니다",
        ],
    }
    payload["evidence_sha256"] = _digest(payload)
    return payload


def write_audit_exclusive(payload: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "DEFAULT_NOVERLAP",
    "DEFAULT_NPERSEG",
    "DEFAULT_START_SECONDS",
    "DEFAULT_STOP_SECONDS",
    "MIN_COHERENCE",
    "PUBLIC_MANIFEST_RELATIVE_PATHS",
    "REQUIRED_SPLITS",
    "STAGE2_POPULATION_AUDIT_SCHEMA",
    "Stage2PopulationAuditError",
    "audit_stage2_2khz_population",
    "audit_stage2_lineage_assignments",
    "build_stage2_data_readiness",
    "build_stage2_coverage_cells",
    "measure_stage2_recorded_signals",
    "plan_minimum_multioctave_additions",
    "sha256_file",
    "write_audit_exclusive",
]
