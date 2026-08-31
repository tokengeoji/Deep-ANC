"""recorded-val 선택과 recorded-test 1회 개봉의 파일 capability 계약."""

from __future__ import annotations

import copy
import hashlib
import ctypes
import errno
import fcntl
import io
import json
import math
import os
import secrets
import stat
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from ..data.manifest import (
    read_manifest_bytes,
    validate_group_id,
    validate_session_id,
    validate_source_family,
)
from ..dsp.do_no_harm import (
    MAX_OUT_OF_BAND_AMPLIFICATION_DB,
    OCTAVE_BAND_CENTERS_HZ,
)
from ..dsp.timing import TrainingTimingContract
from ..eval.trusted_subbands import (
    MIN_GROUPS_PER_FAMILY,
    cluster_bootstrap_ci,
    validate_strict_trusted_subband_metrics,
)
from ..eval.recorded_sampling import (
    CANONICAL_EDGE_TRIM_SECONDS,
    CANONICAL_MAX_SEGMENTS_PER_SESSION,
    CANONICAL_SEGMENT_SECONDS,
    RECORDED_SAMPLING_CONTRACT_SCHEMA,
    canonical_feedback_delay_samples,
    canonical_warmup_samples,
    deterministic_segment_starts,
    effective_segment_samples,
)
from .experiment_contract import validate_embedded_experiment_contract


CAPABILITY_ENV = "DEEP_ANC_RECORDED_TEST_TOKEN"
LEDGER_ROOT = Path("results/recorded_test_ledger")
VAL_BORDERLINE_MARGIN_DB = 0.3
OFFICIAL_FINETUNE_SEEDS = frozenset({20260803, 20260903})
CANONICAL_G4_SOURCE_FAMILIES = (
    "environment",
    "machine",
    "music",
    "speech",
)
_CAMPAIGN_OPERATIONAL_KEYS = {
    "seed",
    "resume",
    "run_until_step",
    "ckpt_dir",
    "resolved_contract_run_dir",
    "experiment_contract",
    "experiment_contract_sha256",
    # seed=20260903 admission proof이다. 일반 experiment contract에는 그대로
    # 결속하지만 두 공식 seed의 학습 의미를 비교하는 projection에서만 제외한다.
    "second_seed_prerequisite",
    "second_seed_prerequisite_sha256",
}
_STRICT_SUBBAND_UNVERIFIED_MARGIN_DB = -1_000_000.0
"""실측 dB가 아니라 fail-closed selection ordering sentinel.

strict 부대역이 검증되지 않은 artifact가 JSON/selection ordering에서 NaN/Inf로
직렬화되어 다른 검증을 흐리게 하지 않되, 어떤 G4 PASS candidate보다도 항상 뒤로
가도록 한다.
"""


_PERSISTED_G4_NUMERIC_ATOL = 1.0e-10
"""저장 float와 raw 재계산값을 대조할 때의 절대 허용오차.

평가기는 같은 NumPy reduction을 쓰므로 이 값은 통계 여유가 아니라 저장/플랫폼의
마지막 비트 차이만 허용한다. 정책 threshold/center는 아래에서 별도로 bit-exact
비교한다.
"""


def _g4_required_scalar(data: np.lib.npyio.NpzFile, key: str) -> object:
    """persisted G4 scalar를 shape까지 fail-closed로 읽는다."""

    if key not in data.files:
        raise ValueError(f"persisted G4 필드가 없습니다: {key}")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"persisted G4 {key}는 scalar여야 합니다")
    return value.reshape(-1)[0].item()


def _g4_required_bool_scalar(data: np.lib.npyio.NpzFile, key: str) -> bool:
    value = np.asarray(data[key]) if key in data.files else None
    if value is None or value.size != 1 or value.dtype.kind != "b":
        raise ValueError(f"persisted G4 {key}는 bool scalar여야 합니다")
    return bool(value.reshape(-1)[0])


def _g4_required_int_scalar(data: np.lib.npyio.NpzFile, key: str) -> int:
    value = np.asarray(data[key]) if key in data.files else None
    if value is None or value.size != 1 or value.dtype.kind not in {"i", "u"}:
        raise ValueError(f"persisted G4 {key}는 integer scalar여야 합니다")
    return int(value.reshape(-1)[0])


def _g4_required_float_scalar(data: np.lib.npyio.NpzFile, key: str) -> float:
    value = np.asarray(data[key]) if key in data.files else None
    if value is None or value.size != 1 or value.dtype.kind != "f":
        raise ValueError(f"persisted G4 {key}는 floating-point scalar여야 합니다")
    return float(value.reshape(-1)[0])


def _g4_required_str_scalar(data: np.lib.npyio.NpzFile, key: str) -> str:
    value = np.asarray(data[key]) if key in data.files else None
    if value is None or value.size != 1 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"persisted G4 {key}는 string scalar여야 합니다")
    return str(value.reshape(-1)[0])


def _g4_dtype_matches(value: np.ndarray, dtype: object) -> bool:
    expected_kind = np.dtype(dtype).kind
    if expected_kind == "U":
        return value.dtype.kind in {"U", "S"}
    if expected_kind in {"i", "u"}:
        return value.dtype.kind in {"i", "u"}
    return value.dtype.kind == expected_kind


def _g4_require_vector(
    data: np.lib.npyio.NpzFile,
    key: str,
    *,
    length: int,
    dtype: object,
) -> np.ndarray:
    if key not in data.files:
        raise ValueError(f"persisted G4 raw/summary 필드가 없습니다: {key}")
    raw = np.asarray(data[key])
    if not _g4_dtype_matches(raw, dtype):
        raise ValueError(
            f"persisted G4 {key} dtype={raw.dtype}; expected kind={np.dtype(dtype).kind}"
        )
    value = raw.astype(dtype, copy=False)
    if value.shape != (length,):
        raise ValueError(
            f"persisted G4 {key} shape={value.shape}; expected=({length},)"
        )
    return value


def _g4_require_matrix(
    data: np.lib.npyio.NpzFile,
    key: str,
    *,
    shape: tuple[int, int],
    dtype: object,
) -> np.ndarray:
    if key not in data.files:
        raise ValueError(f"persisted G4 raw/summary 필드가 없습니다: {key}")
    raw = np.asarray(data[key])
    if not _g4_dtype_matches(raw, dtype):
        raise ValueError(
            f"persisted G4 {key} dtype={raw.dtype}; expected kind={np.dtype(dtype).kind}"
        )
    value = raw.astype(dtype, copy=False)
    if value.shape != shape:
        raise ValueError(
            f"persisted G4 {key} shape={value.shape}; expected={shape}"
        )
    return value


def _g4_same_numeric(actual: np.ndarray | float, expected: np.ndarray | float, *, key: str) -> None:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual_array.shape != expected_array.shape or not np.allclose(
        actual_array,
        expected_array,
        rtol=0.0,
        atol=_PERSISTED_G4_NUMERIC_ATOL,
        equal_nan=True,
    ):
        raise ValueError(f"persisted G4 {key}가 raw segment 재계산값과 다릅니다")


def _g4_distribution(values: np.ndarray, *, worst_is_high: bool) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("persisted G4 raw metric이 비었거나 NaN/Inf를 포함합니다")
    count = max(1, int(math.ceil(array.size * 0.1)))
    ordered = np.sort(array)
    worst = ordered[-count:] if worst_is_high else ordered[:count]
    return {
        "mean_db": float(np.mean(array)),
        "median_db": float(np.median(array)),
        "worst10_mean_db": float(np.mean(worst)),
    }


def _g4_manifest_session_map(
    *,
    manifest_bytes: bytes,
    manifest_path: str | Path,
    split: str,
) -> tuple[dict[str, tuple[str, str]], str]:
    """immutable manifest bytes의 selected split을 raw segment metadata와 묶는다."""

    try:
        entries = read_manifest_bytes(
            manifest_bytes,
            manifest_path=manifest_path,
            split=split,
        )
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"persisted G4 manifest bytes를 해석할 수 없습니다: {exc}") from exc
    if not entries:
        raise ValueError(f"persisted G4 manifest {split} split이 비었습니다")

    sessions: dict[str, tuple[str, str]] = {}
    for index, entry in enumerate(entries):
        try:
            session = validate_session_id(entry.get("session_id"))
            family = validate_source_family(entry.get("source_family"))
            group = validate_group_id(entry.get("group_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "persisted canonical G4 manifest entry에 "
                f"session_id/source_family/group_id가 필요합니다: #{index}: {exc}"
            ) from exc
        if session in sessions:
            raise ValueError(
                f"persisted canonical G4 manifest {split}에 session_id가 중복됩니다: {session!r}"
            )
        sessions[session] = (family, group)
    return sessions, hashlib.sha256(manifest_bytes).hexdigest()


def _g4_validate_manifest_binding(
    *,
    data: np.lib.npyio.NpzFile,
    segment_session: np.ndarray,
    segment_family: np.ndarray,
    segment_group: np.ndarray,
    split: str,
    manifest_bytes: bytes,
    manifest_path: str | Path,
) -> tuple[tuple[str, ...], str]:
    """metrics의 모든 segment와 selected manifest session의 전단사 결속을 강제한다."""

    sessions, manifest_sha = _g4_manifest_session_map(
        manifest_bytes=manifest_bytes,
        manifest_path=manifest_path,
        split=split,
    )
    stored_manifest_sha = _g4_required_str_scalar(data, "manifest_sha256")
    if stored_manifest_sha != manifest_sha:
        raise ValueError("persisted G4 manifest_sha256가 immutable manifest bytes와 다릅니다")

    raw_sessions = set(segment_session.tolist())
    selected_sessions = set(sessions)
    omitted = sorted(selected_sessions.difference(raw_sessions))
    extra = sorted(raw_sessions.difference(selected_sessions))
    if omitted or extra:
        parts: list[str] = []
        if omitted:
            parts.append("누락=" + ", ".join(omitted))
        if extra:
            parts.append("매니페스트 밖=" + ", ".join(extra))
        raise ValueError(
            "persisted G4 raw segment_session_id가 selected manifest session과 정확히 "
            "일치하지 않습니다 (cherry-pick/forgery 금지): " + "; ".join(parts)
        )
    for session, family, group in zip(
        segment_session.tolist(), segment_family.tolist(), segment_group.tolist(), strict=True
    ):
        expected_family, expected_group = sessions[session]
        if family != expected_family or group != expected_group:
            raise ValueError(
                "persisted G4 segment session/family/group가 immutable manifest와 다릅니다: "
                f"session={session!r}, actual=({family!r}, {group!r}), "
                f"expected=({expected_family!r}, {expected_group!r})"
            )
    return tuple(sorted({family for family, _ in sessions.values()})), manifest_sha


def _g4_validate_sampling_binding(
    *,
    data: np.lib.npyio.NpzFile,
    segment_session: np.ndarray,
    sample_rate: int,
    split: str,
    manifest_bytes: bytes,
    manifest_path: str | Path,
    checkpoint_cfg: dict[str, Any],
) -> dict[str, Any]:
    """canonical segment 모집단과 세션별 deterministic start exact set을 재구성한다."""

    schema = _g4_required_str_scalar(data, "recorded_sampling_contract_schema")
    if schema != RECORDED_SAMPLING_CONTRACT_SCHEMA:
        raise ValueError(
            "canonical persisted G4 recorded sampling schema가 현행 계약과 다릅니다"
        )
    if not _g4_required_bool_scalar(data, "recorded_sampling_canonical"):
        raise ValueError("canonical persisted G4가 diagnostic sampling으로 생성됐습니다")
    model_hop = _g4_required_int_scalar(data, "recorded_sampling_model_hop")
    maximum = _g4_required_int_scalar(
        data, "recorded_sampling_max_segments_per_session"
    )
    seconds = _g4_required_float_scalar(data, "recorded_sampling_segment_seconds")
    segment_samples = _g4_required_int_scalar(data, "segment_samples")
    edge_trim_samples = _g4_required_int_scalar(data, "edge_trim_samples")
    plant_settle_samples = _g4_required_int_scalar(
        data, "recorded_sampling_plant_settle_samples"
    )
    warmup_samples = _g4_required_int_scalar(data, "warmup_samples")
    metric_samples = _g4_required_int_scalar(data, "metric_samples_per_segment")
    feedback_delay = _g4_required_int_scalar(data, "feedback_delay_samples")
    if not isinstance(checkpoint_cfg, dict):
        raise ValueError("canonical persisted G4에는 immutable checkpoint cfg가 필요합니다")
    model_cfg = checkpoint_cfg.get("model")
    data_cfg = checkpoint_cfg.get("data")
    if not isinstance(model_cfg, dict) or not isinstance(data_cfg, dict):
        raise ValueError("canonical checkpoint cfg에 model/data 계약이 없습니다")
    checkpoint_hop = model_cfg.get("hop")
    if (
        isinstance(checkpoint_hop, bool)
        or not isinstance(checkpoint_hop, int)
        or int(checkpoint_hop) != model_hop
    ):
        raise ValueError(
            "canonical persisted G4 model_hop이 checkpoint model.hop과 다릅니다"
        )
    checkpoint_rate = data_cfg.get("sample_rate")
    if (
        isinstance(checkpoint_rate, bool)
        or not isinstance(checkpoint_rate, int)
        or int(checkpoint_rate) != sample_rate
    ):
        raise ValueError(
            "canonical persisted G4 sample_rate가 checkpoint data.sample_rate와 다릅니다"
        )
    checkpoint_seconds = data_cfg.get("segment_seconds")
    if (
        isinstance(checkpoint_seconds, bool)
        or not isinstance(checkpoint_seconds, (int, float))
        or not math.isfinite(float(checkpoint_seconds))
        or float(checkpoint_seconds) != seconds
    ):
        raise ValueError(
            "canonical persisted G4 segment_seconds가 checkpoint data와 다릅니다"
        )
    if str(data_cfg.get("reference_mode", "")) != "digital":
        raise ValueError("canonical persisted G4 checkpoint는 digital reference여야 합니다")
    if str(data_cfg.get("recorded_lead_mode", "")) != "timeline":
        raise ValueError("canonical persisted G4 checkpoint는 recorded timeline lead여야 합니다")
    timing = TrainingTimingContract.from_data_config(data_cfg)
    if int(timing.sample_rate) != sample_rate:
        raise ValueError("canonical persisted G4 timing sample_rate가 metrics와 다릅니다")
    timing_sha = timing.digest()
    checkpoint_loss_start = checkpoint_cfg.get("loss_start_sample")
    if (
        isinstance(checkpoint_loss_start, bool)
        or not isinstance(checkpoint_loss_start, int)
        or int(checkpoint_loss_start) != plant_settle_samples
    ):
        raise ValueError(
            "canonical persisted G4 PlantSettle이 checkpoint loss_start_sample과 다릅니다"
        )
    expected_feedback_delay = canonical_feedback_delay_samples(data_cfg)
    if feedback_delay != expected_feedback_delay:
        raise ValueError(
            "canonical persisted G4 feedback delay가 checkpoint 기본 중앙값과 다릅니다"
        )
    expected_warmup = canonical_warmup_samples(
        data_cfg,
        sample_rate=sample_rate,
        plant_settle_samples=plant_settle_samples,
    )
    if warmup_samples != expected_warmup:
        raise ValueError(
            "canonical persisted G4 warmup이 checkpoint 기본값/PlantSettle과 다릅니다"
        )
    if maximum != CANONICAL_MAX_SEGMENTS_PER_SESSION:
        raise ValueError(
            "canonical persisted G4 max_segments_per_session이 64와 다릅니다"
        )
    if seconds != CANONICAL_SEGMENT_SECONDS:
        raise ValueError("canonical persisted G4 segment_seconds가 1.5초와 다릅니다")
    expected_segment_samples = effective_segment_samples(
        sample_rate=sample_rate,
        model_hop=model_hop,
        segment_seconds=seconds,
    )
    if segment_samples != expected_segment_samples:
        raise ValueError(
            "canonical persisted G4 segment_samples가 segment_seconds/hop 유도값과 다릅니다"
        )
    if (
        warmup_samples < 0
        or warmup_samples >= segment_samples
        or metric_samples != segment_samples - warmup_samples
    ):
        raise ValueError(
            "canonical persisted G4 metric_samples_per_segment가 segment-warmup과 다릅니다"
        )
    expected_edge_trim = int(round(CANONICAL_EDGE_TRIM_SECONDS * sample_rate))
    if edge_trim_samples != expected_edge_trim:
        raise ValueError(
            "canonical persisted G4 edge_trim_samples가 0.25초와 다릅니다"
        )

    n_segments = int(segment_session.size)
    starts = _g4_require_vector(
        data, "segment_start_sample", length=n_segments, dtype=np.int64
    )
    leads = _g4_require_vector(
        data, "segment_recorded_lead_samples", length=n_segments, dtype=np.int64
    )
    if np.any(leads < 0):
        raise ValueError("canonical persisted G4 recorded lead에 음수가 있습니다")
    delays = _g4_require_vector(
        data, "segment_recorded_delay_samples", length=n_segments, dtype=np.float64
    )
    timing_shas = _g4_require_vector(
        data, "segment_timing_contract_sha256", length=n_segments, dtype=np.str_
    )
    source_timelines = _g4_require_vector(
        data, "segment_source_timeline", length=n_segments, dtype=np.str_
    )
    if not np.all(np.isfinite(delays)) or np.any(delays < 0.0):
        raise ValueError("canonical persisted G4 recorded delay가 유효하지 않습니다")
    expected_leads = np.asarray(
        [timing.recorded_lead_samples(float(value)) for value in delays],
        dtype=np.int64,
    )
    if not np.array_equal(leads, expected_leads):
        raise ValueError(
            "canonical persisted G4 recorded lead가 checkpoint timing과 session delay에서 "
            "유도한 값과 다릅니다"
        )
    if not np.all(timing_shas == timing_sha):
        raise ValueError(
            "canonical persisted G4 segment timing SHA가 checkpoint timing과 다릅니다"
        )
    if not np.all(source_timelines == "source_aligned.wav"):
        raise ValueError(
            "canonical persisted G4 segment source timeline이 source_aligned.wav가 아닙니다"
        )
    scalar_timing = {
        "digital_reference_lead_samples": int(timing.digital_reference_lead_samples),
        "primary_delay_samples": int(timing.primary_zeros_before_fir_samples),
        "secondary_delay_samples": int(timing.secondary_delay_samples),
        "secondary_handoff_samples": int(timing.handoff_samples),
    }
    for key, expected in scalar_timing.items():
        if _g4_required_int_scalar(data, key) != expected:
            raise ValueError(
                f"canonical persisted G4 {key}가 checkpoint timing과 다릅니다"
            )
    try:
        entries = read_manifest_bytes(
            manifest_bytes, manifest_path=manifest_path, split=split
        )
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"canonical sampling manifest를 해석할 수 없습니다: {exc}") from exc
    entries.sort(
        key=lambda entry: (
            str(entry["group_id"]),
            str(entry["session_id"]),
            str(entry["path"]),
        )
    )
    expected_sessions: list[str] = []
    expected_starts: list[int] = []
    timeline_evidence: list[dict[str, Any]] = []
    for entry in entries:
        session = validate_session_id(entry.get("session_id"))
        entry_rate = entry.get("sample_rate")
        duration = entry.get("duration_s")
        if (
            isinstance(entry_rate, bool)
            or not isinstance(entry_rate, int)
            or int(entry_rate) != sample_rate
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0.0
        ):
            raise ValueError(
                f"canonical sampling manifest session={session!r}의 "
                "sample_rate가 evaluator와 다르거나 duration_s가 유효하지 않습니다"
            )
        declared_frames_float = float(duration) * sample_rate
        declared_frames = int(round(declared_frames_float))
        if not math.isclose(
            declared_frames_float,
            float(declared_frames),
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                f"canonical sampling manifest session={session!r} duration이 "
                "정수 sample 수로 환산되지 않습니다"
            )
        indices = np.flatnonzero(segment_session == session)
        if indices.size == 0:
            raise ValueError(
                f"canonical sampling selected session={session!r} segment가 없습니다"
            )
        session_dir = Path(str(entry.get("path", ""))).expanduser()
        try:
            metadata_snapshot = snapshot_regular_file(session_dir / "session.json")
            metadata = json.loads(metadata_snapshot.content.decode("utf-8"))
        except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"canonical sampling session={session!r} immutable session.json을 "
                f"읽을 수 없습니다: {exc}"
            ) from exc
        timeline = metadata.get("timeline") if isinstance(metadata, dict) else None
        declared_delay = (
            timeline.get("aligned_lag_median_samples")
            if isinstance(timeline, dict)
            else None
        )
        if (
            isinstance(declared_delay, bool)
            or not isinstance(declared_delay, (int, float))
            or not math.isfinite(float(declared_delay))
            or float(declared_delay) < 0.0
        ):
            raise ValueError(
                f"canonical sampling session={session!r} session.json timeline."
                "aligned_lag_median_samples가 유효하지 않습니다"
            )
        if not np.all(delays[indices] == float(declared_delay)):
            raise ValueError(
                f"canonical persisted G4 session={session!r} recorded delay가 immutable "
                "session.json timeline과 다릅니다"
            )
        session_leads = np.unique(leads[indices])
        if session_leads.size != 1:
            raise ValueError(
                f"canonical sampling session={session!r} recorded lead가 일관되지 않습니다"
            )
        usable_samples = declared_frames - int(session_leads[0])
        session_starts = deterministic_segment_starts(
            usable_samples,
            segment_samples,
            maximum,
            edge_trim_samples=edge_trim_samples,
        )
        if not session_starts:
            raise ValueError(
                f"canonical sampling session={session!r}에 평가 가능한 segment가 없습니다"
            )
        expected_sessions.extend([session] * len(session_starts))
        expected_starts.extend(session_starts)
        timeline_evidence.append(
            {
                "session_id": session,
                "session_json_path": str(metadata_snapshot.path),
                "session_json_sha256": metadata_snapshot.sha256,
                "aligned_lag_median_samples": float(declared_delay),
            }
        )

    expected_session_array = np.asarray(expected_sessions, dtype=np.str_)
    expected_start_array = np.asarray(expected_starts, dtype=np.int64)
    if not np.array_equal(segment_session, expected_session_array):
        raise ValueError(
            "canonical persisted G4 segment_session_id 순서/개수가 deterministic "
            "manifest population과 다릅니다"
        )
    if not np.array_equal(starts, expected_start_array):
        raise ValueError(
            "canonical persisted G4 segment_start_sample이 각 selected session의 "
            "expected deterministic start exact set과 다릅니다"
        )
    return {
        "schema": schema,
        "model_hop": model_hop,
        "max_segments_per_session": maximum,
        "segment_seconds": seconds,
        "segment_samples": segment_samples,
        "edge_trim_samples": edge_trim_samples,
        "feedback_delay_samples": feedback_delay,
        "warmup_samples": warmup_samples,
        "metric_samples_per_segment": metric_samples,
        "plant_settle_samples": plant_settle_samples,
        "training_timing_contract_sha256": timing_sha,
        "session_timeline_evidence_sha256": hashlib.sha256(
            json.dumps(
                timeline_evidence,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }


def validate_persisted_g4_metrics(
    data: np.lib.npyio.NpzFile,
    *,
    expected_split: str,
    manifest_bytes: bytes | None = None,
    manifest_path: str | Path | None = None,
    checkpoint_cfg: dict[str, Any] | None = None,
    canonical: bool,
    surrogate_diagnostic: bool = False,
    min_groups: int = MIN_GROUPS_PER_FAMILY,
) -> dict[str, Any]:
    """official recorded G4 NPZ를 raw evidence에서 단일 방식으로 재감사한다.

    Summary scalar/boolean는 모두 편의용 복사본일 뿐 authority가 아니다. canonical
    경로에서는 raw global/family/octave arrays와 immutable selected manifest bytes를
    함께 요구하고 재계산한다. ``surrogate_diagnostic=True``는 20k
    loss-pilot의 같은 population/주파수 계약을 감사하되 surrogate 물리를
    명시적으로 허용하는 campaign 전용 진단 경로다. 이 결과는 실제
    덕트 성능/test PASS 근거가 아니다. ``canonical=False``는 manifest 결속이
    없는 사람용 진단이다.
    """

    if expected_split not in {"val", "test"}:
        raise ValueError(f"persisted G4 expected_split이 지원되지 않습니다: {expected_split!r}")
    minimum = int(min_groups)
    if minimum < 1:
        raise ValueError("persisted G4 min_groups는 1 이상이어야 합니다")
    if surrogate_diagnostic and not canonical:
        raise ValueError(
            "surrogate_diagnostic persisted G4는 canonical population 결속을 "
            "함께 요구합니다"
        )

    required = {
        "split",
        "g4_metric_scope",
        "physics_status",
        "allow_surrogate",
        "sample_rate",
        "manifest_sha256",
        "trusted_band_hz",
        "n_segments",
        "n_sessions",
        "n_groups",
        "segment_session_id",
        "segment_source_family",
        "segment_group_id",
        "per_segment_trusted_db",
        "per_segment_fullband_db",
        "per_segment_gap_db",
        "per_segment_octave_attenuation_db",
        "nmse_trusted_mean_db",
        "nmse_trusted_median_db",
        "nmse_trusted_worst10_mean_db",
        "nmse_fullband_mean_db",
        "nmse_fullband_median_db",
        "nmse_fullband_worst10_mean_db",
        "nmse_gap_trusted_minus_fullband_mean_db",
        "source_family",
        "source_n_segments",
        "source_n_sessions",
        "source_n_groups",
        "source_nmse_trusted_mean_db",
        "source_nmse_trusted_worst10_mean_db",
        "source_nmse_fullband_mean_db",
        "source_nmse_fullband_worst10_mean_db",
        "source_gap_trusted_minus_fullband_mean_db",
        "source_trusted_ci_lo_db",
        "source_trusted_ci_hi_db",
        "octave_center_hz",
        "octave_attenuation_mean_db",
        "octave_attenuation_median_db",
        "octave_attenuation_worst10_mean_db",
        "octave_trusted",
        "g4_trusted_pass",
        "g4_fullband_pass",
        "g4_source_pass",
        "g4_worst_source_trusted_mean_db",
        "g4_worst_source_trusted_worst10_db",
        "g4_worst_source_family",
        "g4_do_no_harm_pass",
        "g4_max_out_of_band_amplification_db",
        "g4_worst_octave_center_hz",
        "g4_worst_octave_worst10_db",
        "g4_power_pass",
        "g4_ci_pass",
        "g4_min_groups_per_family",
        "g4_underpowered_families",
        "g4_pass",
        "g4_verdict",
    }
    if canonical:
        required.update(
            {
                "recorded_sampling_contract_schema",
                "recorded_sampling_canonical",
                "recorded_sampling_model_hop",
                "recorded_sampling_max_segments_per_session",
                "recorded_sampling_segment_seconds",
                "recorded_sampling_plant_settle_samples",
                "segment_samples",
                "metric_samples_per_segment",
                "edge_trim_samples",
                "warmup_samples",
                "feedback_delay_samples",
                "digital_reference_lead_samples",
                "primary_delay_samples",
                "secondary_delay_samples",
                "secondary_handoff_samples",
                "segment_start_sample",
                "segment_recorded_lead_samples",
                "segment_recorded_delay_samples",
                "segment_timing_contract_sha256",
                "segment_source_timeline",
            }
        )
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError("persisted G4 raw 계약 필드 누락: " + ", ".join(missing))

    split = _g4_required_str_scalar(data, "split")
    if split != expected_split:
        raise ValueError(f"persisted G4 split={split!r}; expected={expected_split!r}")
    scope = _g4_required_str_scalar(data, "g4_metric_scope")
    if scope not in {"canonical_recorded_g4", "diagnostic_noncanonical"}:
        raise ValueError("persisted G4 g4_metric_scope가 지원되지 않습니다")
    expected_scope = (
        "diagnostic_noncanonical" if surrogate_diagnostic else "canonical_recorded_g4"
    )
    if canonical and scope != expected_scope:
        raise ValueError(
            "persisted G4 metric scope가 요구된 measured/surrogate 물리와 "
            f"다릅니다: actual={scope!r}, expected={expected_scope!r}"
        )
    physics_status = _g4_required_str_scalar(data, "physics_status")
    allow_surrogate = _g4_required_bool_scalar(data, "allow_surrogate")
    sample_rate = _g4_required_int_scalar(data, "sample_rate")
    if canonical:
        expected_physics = (
            "secondary_surrogate_representation_pretrain"
            if surrogate_diagnostic
            else "measured_primary_path"
        )
        if (
            physics_status != expected_physics
            or allow_surrogate is not surrogate_diagnostic
            or sample_rate != 48_000
        ):
            raise ValueError(
                "canonical-population persisted G4의 physics/surrogate/sample-rate가 "
                "요구 계약과 다릅니다: "
                f"physics={physics_status!r}, allow_surrogate={allow_surrogate!r}, "
                f"sample_rate={sample_rate!r}"
            )
    n_segments = _g4_required_int_scalar(data, "n_segments")
    n_sessions = _g4_required_int_scalar(data, "n_sessions")
    n_groups = _g4_required_int_scalar(data, "n_groups")
    if n_segments <= 0 or n_sessions <= 0 or n_groups <= 0:
        raise ValueError("persisted G4 n_segments/n_sessions/n_groups는 모두 양수여야 합니다")

    segment_session = _g4_require_vector(
        data, "segment_session_id", length=n_segments, dtype=np.str_
    )
    segment_family = _g4_require_vector(
        data, "segment_source_family", length=n_segments, dtype=np.str_
    )
    segment_group = _g4_require_vector(
        data, "segment_group_id", length=n_segments, dtype=np.str_
    )
    for session, family, group in zip(
        segment_session.tolist(), segment_family.tolist(), segment_group.tolist(), strict=True
    ):
        try:
            validate_session_id(session)
            validate_source_family(family)
            validate_group_id(group)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"persisted G4 raw segment metadata가 유효하지 않습니다: {exc}") from exc
    if n_sessions != int(np.unique(segment_session).size):
        raise ValueError("persisted G4 n_sessions가 raw segment_session_id와 다릅니다")
    if n_groups != int(np.unique(segment_group).size):
        raise ValueError("persisted G4 n_groups가 raw segment_group_id와 다릅니다")
    for session in np.unique(segment_session):
        mask = segment_session == session
        if np.unique(segment_family[mask]).size != 1 or np.unique(segment_group[mask]).size != 1:
            raise ValueError("persisted G4 한 session의 family/group가 일관되지 않습니다")
    for group in np.unique(segment_group):
        if np.unique(segment_family[segment_group == group]).size != 1:
            raise ValueError("persisted G4 한 group이 여러 source family에 걸쳐 있습니다")

    manifest_families: tuple[str, ...] | None = None
    manifest_sha = ""
    sampling_contract: dict[str, Any] | None = None
    if canonical:
        if manifest_bytes is None or manifest_path is None:
            raise ValueError("canonical persisted G4에는 immutable manifest bytes/path가 필요합니다")
        if not isinstance(checkpoint_cfg, dict):
            raise ValueError("canonical persisted G4에는 immutable checkpoint cfg가 필요합니다")
        manifest_families, manifest_sha = _g4_validate_manifest_binding(
            data=data,
            segment_session=segment_session,
            segment_family=segment_family,
            segment_group=segment_group,
            split=split,
            manifest_bytes=manifest_bytes,
            manifest_path=manifest_path,
        )
        if manifest_families != CANONICAL_G4_SOURCE_FAMILIES:
            raise ValueError(
                "canonical persisted G4 selected split의 source family가 "
                f"{list(CANONICAL_G4_SOURCE_FAMILIES)}와 정확히 같아야 합니다: "
                f"actual={list(manifest_families)}"
            )
        sampling_contract = _g4_validate_sampling_binding(
            data=data,
            segment_session=segment_session,
            sample_rate=sample_rate,
            split=split,
            manifest_bytes=manifest_bytes,
            manifest_path=manifest_path,
            checkpoint_cfg=checkpoint_cfg,
        )

    trusted = _g4_require_vector(
        data, "per_segment_trusted_db", length=n_segments, dtype=np.float64
    )
    fullband = _g4_require_vector(
        data, "per_segment_fullband_db", length=n_segments, dtype=np.float64
    )
    gap = _g4_require_vector(
        data, "per_segment_gap_db", length=n_segments, dtype=np.float64
    )
    if not np.all(np.isfinite(trusted)) or not np.all(np.isfinite(fullband)):
        raise ValueError("persisted G4 raw trusted/fullband에 NaN/Inf가 있습니다")
    _g4_same_numeric(gap, trusted - fullband, key="per_segment_gap_db")
    trusted_stats = _g4_distribution(trusted, worst_is_high=True)
    fullband_stats = _g4_distribution(fullband, worst_is_high=True)
    _g4_same_numeric(
        _g4_required_float_scalar(data, "nmse_trusted_mean_db"),
        trusted_stats["mean_db"],
        key="nmse_trusted_mean_db",
    )
    _g4_same_numeric(
        _g4_required_float_scalar(data, "nmse_trusted_median_db"),
        trusted_stats["median_db"],
        key="nmse_trusted_median_db",
    )
    _g4_same_numeric(
        _g4_required_float_scalar(data, "nmse_trusted_worst10_mean_db"),
        trusted_stats["worst10_mean_db"],
        key="nmse_trusted_worst10_mean_db",
    )
    _g4_same_numeric(
        _g4_required_float_scalar(data, "nmse_fullband_mean_db"),
        fullband_stats["mean_db"],
        key="nmse_fullband_mean_db",
    )
    _g4_same_numeric(
        _g4_required_float_scalar(data, "nmse_fullband_median_db"),
        fullband_stats["median_db"],
        key="nmse_fullband_median_db",
    )
    _g4_same_numeric(
        _g4_required_float_scalar(data, "nmse_fullband_worst10_mean_db"),
        fullband_stats["worst10_mean_db"],
        key="nmse_fullband_worst10_mean_db",
    )
    gap_mean = float(np.mean(gap))
    _g4_same_numeric(
        _g4_required_float_scalar(data, "nmse_gap_trusted_minus_fullband_mean_db"),
        gap_mean,
        key="nmse_gap_trusted_minus_fullband_mean_db",
    )

    families_raw = np.asarray(data["source_family"])
    if families_raw.ndim != 1 or families_raw.dtype.kind not in {"U", "S"}:
        raise ValueError("persisted G4 source_family는 string 1차원 배열이어야 합니다")
    families = families_raw.astype(str, copy=False)
    if families.size == 0 or any(not family for family in families):
        raise ValueError("persisted G4 source_family는 비지 않은 1차원 배열이어야 합니다")
    if len(set(families.tolist())) != families.size:
        raise ValueError("persisted G4 source_family에 중복이 있습니다")
    expected_families = tuple(sorted(set(segment_family.tolist())))
    if tuple(families.tolist()) != expected_families:
        raise ValueError(
            "persisted G4 source_family가 raw segment_source_family의 정렬된 집합과 다릅니다"
        )
    if manifest_families is not None and expected_families != manifest_families:
        raise ValueError("persisted G4 source_family가 selected manifest family와 다릅니다")
    family_count = int(families.size)
    source_n_segments = _g4_require_vector(
        data, "source_n_segments", length=family_count, dtype=np.int64
    )
    source_n_sessions = _g4_require_vector(
        data, "source_n_sessions", length=family_count, dtype=np.int64
    )
    source_n_groups = _g4_require_vector(
        data, "source_n_groups", length=family_count, dtype=np.int64
    )
    source_trusted_mean = _g4_require_vector(
        data, "source_nmse_trusted_mean_db", length=family_count, dtype=np.float64
    )
    source_trusted_worst10 = _g4_require_vector(
        data,
        "source_nmse_trusted_worst10_mean_db",
        length=family_count,
        dtype=np.float64,
    )
    source_fullband_mean = _g4_require_vector(
        data, "source_nmse_fullband_mean_db", length=family_count, dtype=np.float64
    )
    source_fullband_worst10 = _g4_require_vector(
        data,
        "source_nmse_fullband_worst10_mean_db",
        length=family_count,
        dtype=np.float64,
    )
    source_gap_mean = _g4_require_vector(
        data,
        "source_gap_trusted_minus_fullband_mean_db",
        length=family_count,
        dtype=np.float64,
    )
    source_ci_lo = _g4_require_vector(
        data, "source_trusted_ci_lo_db", length=family_count, dtype=np.float64
    )
    source_ci_hi = _g4_require_vector(
        data, "source_trusted_ci_hi_db", length=family_count, dtype=np.float64
    )

    recomputed_n_segments = np.zeros(family_count, dtype=np.int64)
    recomputed_n_sessions = np.zeros(family_count, dtype=np.int64)
    recomputed_n_groups = np.zeros(family_count, dtype=np.int64)
    recomputed_trusted_mean = np.zeros(family_count, dtype=np.float64)
    recomputed_trusted_worst10 = np.zeros(family_count, dtype=np.float64)
    recomputed_fullband_mean = np.zeros(family_count, dtype=np.float64)
    recomputed_fullband_worst10 = np.zeros(family_count, dtype=np.float64)
    recomputed_gap_mean = np.zeros(family_count, dtype=np.float64)
    recomputed_ci_lo = np.full(family_count, np.nan, dtype=np.float64)
    recomputed_ci_hi = np.full(family_count, np.nan, dtype=np.float64)
    for index, family in enumerate(families.tolist()):
        mask = segment_family == family
        values = trusted[mask]
        full_values = fullband[mask]
        recomputed_n_segments[index] = int(values.size)
        recomputed_n_sessions[index] = int(np.unique(segment_session[mask]).size)
        recomputed_n_groups[index] = int(np.unique(segment_group[mask]).size)
        trusted_distribution = _g4_distribution(values, worst_is_high=True)
        fullband_distribution = _g4_distribution(full_values, worst_is_high=True)
        recomputed_trusted_mean[index] = trusted_distribution["mean_db"]
        recomputed_trusted_worst10[index] = trusted_distribution["worst10_mean_db"]
        recomputed_fullband_mean[index] = fullband_distribution["mean_db"]
        recomputed_fullband_worst10[index] = fullband_distribution["worst10_mean_db"]
        recomputed_gap_mean[index] = float(np.mean(gap[mask]))
        lo, hi, _ = cluster_bootstrap_ci(
            values,
            segment_group[mask],
            min_groups=minimum,
        )
        recomputed_ci_lo[index] = lo
        recomputed_ci_hi[index] = hi
    for key, actual, expected in (
        ("source_n_segments", source_n_segments, recomputed_n_segments),
        ("source_n_sessions", source_n_sessions, recomputed_n_sessions),
        ("source_n_groups", source_n_groups, recomputed_n_groups),
    ):
        if not np.array_equal(actual, expected):
            raise ValueError(f"persisted G4 {key}가 raw segment 재계산값과 다릅니다")
    for key, actual, expected in (
        ("source_nmse_trusted_mean_db", source_trusted_mean, recomputed_trusted_mean),
        (
            "source_nmse_trusted_worst10_mean_db",
            source_trusted_worst10,
            recomputed_trusted_worst10,
        ),
        ("source_nmse_fullband_mean_db", source_fullband_mean, recomputed_fullband_mean),
        (
            "source_nmse_fullband_worst10_mean_db",
            source_fullband_worst10,
            recomputed_fullband_worst10,
        ),
        ("source_gap_trusted_minus_fullband_mean_db", source_gap_mean, recomputed_gap_mean),
        ("source_trusted_ci_lo_db", source_ci_lo, recomputed_ci_lo),
        ("source_trusted_ci_hi_db", source_ci_hi, recomputed_ci_hi),
    ):
        _g4_same_numeric(actual, expected, key=key)

    trusted_band = _g4_require_vector(
        data, "trusted_band_hz", length=2, dtype=np.float64
    )
    if not np.all(np.isfinite(trusted_band)):
        raise ValueError("persisted G4 trusted_band_hz는 유한한 [lo, hi]여야 합니다")
    centers_raw = np.asarray(data["octave_center_hz"])
    if centers_raw.ndim != 1 or centers_raw.dtype.kind != "f":
        raise ValueError("persisted G4 octave_center_hz는 floating-point 1차원 배열이어야 합니다")
    centers = centers_raw.astype(np.float64, copy=False)
    if centers.size == 0 or not np.all(np.isfinite(centers)):
        raise ValueError("persisted G4 octave_center_hz가 유효하지 않습니다")
    if np.unique(centers).size != centers.size:
        raise ValueError("persisted G4 octave_center_hz에 중복이 있습니다")
    canonical_centers = np.asarray(OCTAVE_BAND_CENTERS_HZ, dtype=np.float64)
    if canonical and (centers.shape != canonical_centers.shape or not np.array_equal(centers, canonical_centers)):
        raise ValueError(
            "canonical persisted G4 octave_center_hz가 125/250/500/1000/1600/2000/4000/8000Hz와 정확히 다릅니다"
        )
    if canonical and not np.array_equal(trusted_band, np.asarray((150.0, 1600.0))):
        raise ValueError("canonical persisted G4 trusted_band_hz가 150–1600Hz와 정확히 다릅니다")
    if not canonical and scope == "canonical_recorded_g4" and manifest_bytes is None:
        # 파일 자체가 canonical 형식이어도 manifest population을 모르면 selection
        # authority가 아니다. 아래 반환값은 diagnostic으로 고정된다.
        pass
    octave_count = int(centers.size)
    octave_values = _g4_require_matrix(
        data,
        "per_segment_octave_attenuation_db",
        shape=(n_segments, octave_count),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(octave_values)):
        raise ValueError("persisted G4 raw octave attenuation에 NaN/Inf가 있습니다")
    octave_mean = _g4_require_vector(
        data, "octave_attenuation_mean_db", length=octave_count, dtype=np.float64
    )
    octave_median = _g4_require_vector(
        data, "octave_attenuation_median_db", length=octave_count, dtype=np.float64
    )
    octave_worst10 = _g4_require_vector(
        data,
        "octave_attenuation_worst10_mean_db",
        length=octave_count,
        dtype=np.float64,
    )
    octave_trusted = _g4_require_vector(
        data, "octave_trusted", length=octave_count, dtype=np.bool_
    )
    recomputed_octave_mean = np.zeros(octave_count, dtype=np.float64)
    recomputed_octave_median = np.zeros(octave_count, dtype=np.float64)
    recomputed_octave_worst10 = np.zeros(octave_count, dtype=np.float64)
    for index in range(octave_count):
        distribution = _g4_distribution(octave_values[:, index], worst_is_high=False)
        recomputed_octave_mean[index] = distribution["mean_db"]
        recomputed_octave_median[index] = distribution["median_db"]
        recomputed_octave_worst10[index] = distribution["worst10_mean_db"]
    _g4_same_numeric(octave_mean, recomputed_octave_mean, key="octave_attenuation_mean_db")
    _g4_same_numeric(octave_median, recomputed_octave_median, key="octave_attenuation_median_db")
    _g4_same_numeric(
        octave_worst10,
        recomputed_octave_worst10,
        key="octave_attenuation_worst10_mean_db",
    )
    expected_octave_trusted = (centers >= trusted_band[0]) & (centers <= trusted_band[1])
    if not np.array_equal(octave_trusted, expected_octave_trusted):
        raise ValueError("persisted G4 octave_trusted가 trusted_band_hz 규약과 다릅니다")
    threshold = _g4_required_float_scalar(
        data, "g4_max_out_of_band_amplification_db"
    )
    if not math.isfinite(threshold) or threshold != float(MAX_OUT_OF_BAND_AMPLIFICATION_DB):
        raise ValueError(
            "persisted G4 do-no-harm threshold가 canonical "
            "MAX_OUT_OF_BAND_AMPLIFICATION_DB와 정확히 다릅니다"
        )
    out_of_band_mask = ~expected_octave_trusted
    if not np.any(out_of_band_mask):
        raise ValueError("persisted G4에 do-no-harm을 판정할 대역 밖 octave가 없습니다")
    out_of_band_indices = np.flatnonzero(out_of_band_mask)
    worst_octave_index = int(
        out_of_band_indices[
            np.argmin(recomputed_octave_worst10[out_of_band_mask])
        ]
    )
    expected_worst_octave_hz = float(centers[worst_octave_index])
    expected_worst_octave_db = float(recomputed_octave_worst10[worst_octave_index])
    _g4_same_numeric(
        _g4_required_float_scalar(data, "g4_worst_octave_center_hz"),
        expected_worst_octave_hz,
        key="g4_worst_octave_center_hz",
    )
    _g4_same_numeric(
        _g4_required_float_scalar(data, "g4_worst_octave_worst10_db"),
        expected_worst_octave_db,
        key="g4_worst_octave_worst10_db",
    )
    expected_do_no_harm_pass = bool(
        not np.any(recomputed_octave_worst10[out_of_band_mask] <= -threshold)
    )

    expected_trusted_pass = bool(trusted_stats["mean_db"] < 0.0)
    expected_fullband_pass = bool(fullband_stats["mean_db"] <= 0.0)
    worst_source_index = int(np.argmax(recomputed_trusted_mean))
    expected_worst_source_family = str(families[worst_source_index])
    expected_worst_source_mean = float(recomputed_trusted_mean[worst_source_index])
    expected_worst_source_worst10 = float(np.max(recomputed_trusted_worst10))
    expected_source_pass = bool(
        np.all(recomputed_trusted_mean < 0.0)
        and np.all(recomputed_trusted_worst10 < 0.0)
    )
    expected_power_pass = bool(np.all(recomputed_n_groups >= minimum))
    expected_ci_pass = bool(
        np.all(np.isfinite(recomputed_ci_hi)) and np.all(recomputed_ci_hi < 0.0)
    )
    expected_underpowered = families[recomputed_n_groups < minimum]
    for key, expected in (
        ("g4_trusted_pass", expected_trusted_pass),
        ("g4_fullband_pass", expected_fullband_pass),
        ("g4_source_pass", expected_source_pass),
        ("g4_do_no_harm_pass", expected_do_no_harm_pass),
        ("g4_power_pass", expected_power_pass),
        ("g4_ci_pass", expected_ci_pass),
    ):
        if _g4_required_bool_scalar(data, key) != expected:
            raise ValueError(f"persisted G4 {key}가 raw segment 재계산값과 다릅니다")
    if _g4_required_int_scalar(data, "g4_min_groups_per_family") != minimum:
        raise ValueError("persisted G4 g4_min_groups_per_family가 canonical 정책과 다릅니다")
    stored_underpowered_raw = np.asarray(data["g4_underpowered_families"])
    if stored_underpowered_raw.dtype.kind not in {"U", "S"}:
        raise ValueError("persisted G4 g4_underpowered_families는 string 배열이어야 합니다")
    stored_underpowered = stored_underpowered_raw.astype(str, copy=False)
    if stored_underpowered.shape != expected_underpowered.shape or not np.array_equal(
        stored_underpowered, expected_underpowered
    ):
        raise ValueError("persisted G4 g4_underpowered_families가 raw group count와 다릅니다")
    _g4_same_numeric(
        _g4_required_float_scalar(data, "g4_worst_source_trusted_mean_db"),
        expected_worst_source_mean,
        key="g4_worst_source_trusted_mean_db",
    )
    _g4_same_numeric(
        _g4_required_float_scalar(data, "g4_worst_source_trusted_worst10_db"),
        expected_worst_source_worst10,
        key="g4_worst_source_trusted_worst10_db",
    )
    if _g4_required_str_scalar(data, "g4_worst_source_family") != expected_worst_source_family:
        raise ValueError("persisted G4 g4_worst_source_family가 raw family mean과 다릅니다")

    # strict validator도 raw segment metadata를 읽는다. canonical이면 위 manifest
    # 결속이 먼저 identity를 잠근 상태이고, no-manifest diagnostic이어도 scalar/threshold
    # 위조 자체는 허용하지 않는다. manifest 부재는 아래 classifier가 authority를
    # ``diagnostic``으로 제한하는 이유이지 raw 계약 검사를 생략할 이유가 아니다.
    strict_subband = validate_strict_trusted_subband_metrics(
        data,
        min_groups=minimum,
    )
    strict_flags = strict_subband["flags"]
    strict_mean = np.asarray(strict_subband["mean_db"], dtype=np.float64)
    strict_worst10 = np.asarray(strict_subband["worst10_db"], dtype=np.float64)
    strict_hard_failure = bool(
        np.any(np.isfinite(strict_mean) & (strict_mean >= 0.0))
        or np.any(np.isfinite(strict_worst10) & (strict_worst10 >= 0.0))
    )
    strict_inconclusive = not (
        strict_flags["g4_trusted_subband_coverage_pass"]
        and strict_flags["g4_trusted_subband_power_pass"]
        and strict_flags["g4_trusted_subband_ci_pass"]
    )
    expected_verdict = (
        "FAIL"
        if (
            not expected_trusted_pass
            or not expected_fullband_pass
            or not expected_source_pass
            or not expected_do_no_harm_pass
            or strict_hard_failure
        )
        else "INCONCLUSIVE"
        if (not expected_power_pass or not expected_ci_pass or strict_inconclusive)
        else "PASS"
    )
    expected_g4_pass = expected_verdict == "PASS"
    if _g4_required_str_scalar(data, "g4_verdict") != expected_verdict:
        raise ValueError("persisted G4 g4_verdict가 raw evidence 재계산값과 다릅니다")
    if _g4_required_bool_scalar(data, "g4_pass") != expected_g4_pass:
        raise ValueError("persisted G4 g4_pass가 raw evidence 재계산값과 다릅니다")

    return {
        "canonical": bool(canonical),
        "surrogate_diagnostic": bool(surrogate_diagnostic),
        "performance_authority": bool(canonical and not surrogate_diagnostic),
        "manifest_bound": manifest_families is not None,
        "manifest_sha256": manifest_sha,
        "sampling_contract": sampling_contract,
        "split": split,
        "n_segments": n_segments,
        "n_sessions": n_sessions,
        "n_groups": n_groups,
        "families": tuple(families.tolist()),
        "trusted": trusted_stats,
        "fullband": fullband_stats,
        "gap_mean_db": gap_mean,
        "source_trusted_mean_db": recomputed_trusted_mean,
        "source_trusted_worst10_db": recomputed_trusted_worst10,
        "source_ci_hi_db": recomputed_ci_hi,
        "source_n_groups": recomputed_n_groups,
        "worst_source_family": expected_worst_source_family,
        "worst_source_mean_db": expected_worst_source_mean,
        "worst_source_worst10_db": expected_worst_source_worst10,
        "octave_centers_hz": centers,
        "octave_worst10_db": recomputed_octave_worst10,
        "worst_octave_hz": expected_worst_octave_hz,
        "worst_octave_db": expected_worst_octave_db,
        "threshold_db": threshold,
        "flags": {
            "g4_trusted_pass": expected_trusted_pass,
            "g4_fullband_pass": expected_fullband_pass,
            "g4_source_pass": expected_source_pass,
            "g4_do_no_harm_pass": expected_do_no_harm_pass,
            "g4_power_pass": expected_power_pass,
            "g4_ci_pass": expected_ci_pass,
            "g4_pass": expected_g4_pass,
        },
        "g4_verdict": expected_verdict,
        "strict_subband": strict_subband,
    }


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    content: bytes
    sha256: str


def snapshot_regular_file(path: str | Path) -> FileSnapshot:
    """한 immutable regular-file snapshot으로 bytes/hash/load를 함께 만든다."""

    target = Path(os.path.abspath(Path(path)))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise ValueError(f"regular-file snapshot을 열 수 없습니다: {target}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"regular file만 허용합니다: {target}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in stable):
        raise ValueError(f"regular-file snapshot 도중 파일이 변경됐습니다: {target}")
    if len(content) != int(after.st_size):
        raise ValueError(f"regular-file snapshot byte 수가 size와 다릅니다: {target}")
    try:
        pathname = target.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"regular-file snapshot pathname이 사라졌습니다: {target}") from exc
    if stat.S_ISLNK(pathname.st_mode) or (pathname.st_dev, pathname.st_ino) != (
        after.st_dev,
        after.st_ino,
    ):
        raise ValueError(f"regular-file snapshot pathname이 retarget됐습니다: {target}")
    return FileSnapshot(target, content, hashlib.sha256(content).hexdigest())


def seed_neutral_campaign_sha256(
    cfg: dict[str, Any], *, _visited_init_sha: frozenset[str] = frozenset()
) -> str:
    """seed/출력 위치만 제거하고 init 계보까지 결속한 campaign digest.

    recorded-val selection과 pipeline이 반드시 같은 구현을 사용해야 한다. selection JSON에
    임의의 64자리 값을 넣어 별도 test ledger를 여는 경로를 막기 위해 test capability도
    checkpoint embedded cfg에서 이 값을 다시 계산한다.
    """

    contract = validate_embedded_experiment_contract(cfg)
    semantic = copy.deepcopy(
        {
            key: value
            for key, value in cfg.items()
            if key not in _CAMPAIGN_OPERATIONAL_KEYS
        }
    )
    init_value = cfg.get("init_ckpt")
    if init_value:
        from ..config import REPO_ROOT

        init_path = Path(str(init_value)).expanduser()
        if not init_path.is_absolute():
            init_path = Path(REPO_ROOT) / init_path
        init_snapshot = snapshot_regular_file(init_path)
        if init_snapshot.sha256 in _visited_init_sha:
            raise ValueError("campaign init checkpoint 계보에 순환이 있습니다")
        init_state = torch.load(
            io.BytesIO(init_snapshot.content), map_location="cpu", weights_only=False
        )
        if not isinstance(init_state, dict) or not isinstance(
            init_state.get("cfg"), dict
        ):
            raise ValueError("campaign init checkpoint에 resolved cfg가 없습니다")
        init_cfg = init_state["cfg"]
        semantic["init_ckpt"] = {
            "experiment_role": init_cfg.get("experiment_role"),
            "init_eligible": init_cfg.get("init_eligible"),
            "loss_selection_sha256": init_cfg.get("loss_selection_sha256"),
            "seed_neutral_campaign_sha256": seed_neutral_campaign_sha256(
                init_cfg,
                _visited_init_sha=_visited_init_sha | {init_snapshot.sha256},
            ),
        }
    source = contract.get("source") or {}
    artifact_identity = {
        name: {
            key: entry.get(key)
            for key in ("exists", "size_bytes", "sha256")
            if key in entry
        }
        for name, entry in sorted((contract.get("artifacts") or {}).items())
        if name not in {"init_checkpoint", "second_seed_prerequisite"}
        and isinstance(entry, dict)
    }
    payload = {
        "schema_version": 1,
        "semantic_config": semantic,
        "source": {
            "git_commit": source.get("git_commit"),
            "source_tree_sha256": source.get("source_tree_sha256"),
        },
        "artifacts": artifact_identity,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_checkpoint_recorded_manifest(
    cfg: dict[str, Any],
    contract: dict[str, Any],
    manifest: FileSnapshot,
) -> None:
    """checkpoint contract의 canonical recorded manifest와 selection bytes를 결속한다."""

    if not cfg.get("recorded_manifest"):
        raise ValueError("selection checkpoint cfg에 recorded_manifest가 없습니다")
    artifacts = contract.get("artifacts")
    recorded = artifacts.get("recorded_manifest") if isinstance(artifacts, dict) else None
    if not isinstance(recorded, dict) or not bool(recorded.get("exists")):
        raise ValueError(
            "selection checkpoint experiment contract에 recorded_manifest artifact가 없습니다"
        )
    expected_path = str(Path(str(recorded.get("path", ""))).absolute())
    expected = {
        "path": expected_path,
        "size_bytes": int(recorded.get("size_bytes", -1)),
        "sha256": str(recorded.get("sha256", "")),
    }
    actual = {
        "path": str(manifest.path),
        "size_bytes": len(manifest.content),
        "sha256": manifest.sha256,
    }
    if actual != expected:
        raise ValueError(
            "selection manifest가 checkpoint embedded experiment contract의 "
            f"recorded_manifest와 다릅니다: actual={actual}, expected={expected}"
        )


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def write_json_exclusive(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(os.path.abspath(Path(path)))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"JSON artifact를 덮어쓸 수 없습니다: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        content = _json_bytes(payload)
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, target, follow_symlinks=False)
        directory_fd = os.open(
            target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_json_snapshot(path: str | Path) -> tuple[dict[str, Any], FileSnapshot]:
    snapshot = snapshot_regular_file(path)
    try:
        payload = json.loads(snapshot.content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON artifact가 손상됐습니다: {snapshot.path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact 최상위가 mapping이 아닙니다: {snapshot.path}")
    return payload, snapshot


def _sha256_identity(value: object, *, name: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"test ledger {name}가 64자리 SHA-256이 아닙니다")
    return text


def _npz_scalar(data: np.lib.npyio.NpzFile, key: str) -> object:
    if key not in data.files:
        raise ValueError(f"recorded-val G4 필드가 없습니다: {key}")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"recorded-val G4 {key}는 scalar여야 합니다")
    return value.reshape(-1)[0].item()


def classify_recorded_val_metrics(
    metrics_bytes: bytes,
    *,
    manifest_bytes: bytes | None = None,
    manifest_path: str | Path | None = None,
    checkpoint_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """capability와 pipeline이 공유하는 G4 clear-pass/borderline 분류기.

    ``manifest_bytes`` 없는 호출은 과거/수동 분석을 위한 **diagnostic** 경로다.
    raw summary의 정합성은 확인하지만 selected population과 결속할 수 없으므로
    절대로 ``clear_pass``를 반환하지 않는다. test capability를 여는 호출은 반드시
    immutable manifest snapshot bytes/path를 함께 전달한다.
    """

    if (manifest_bytes is None) != (manifest_path is None):
        raise ValueError("recorded-val canonical 분류에는 manifest bytes와 path를 함께 줘야 합니다")
    canonical = manifest_bytes is not None
    try:
        archive = np.load(io.BytesIO(metrics_bytes), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("recorded-val metrics.npz가 손상됐습니다") from exc
    with archive as data:
        evidence = validate_persisted_g4_metrics(
            data,
            expected_split="val",
            manifest_bytes=manifest_bytes,
            manifest_path=manifest_path,
            checkpoint_cfg=checkpoint_cfg,
            canonical=canonical,
            min_groups=MIN_GROUPS_PER_FAMILY,
        )
    verdict = str(evidence["g4_verdict"])
    threshold = float(evidence["threshold_db"])
    margins = {
        "trusted_mean_db": -float(evidence["trusted"]["mean_db"]),
        "fullband_mean_db": -float(evidence["fullband"]["mean_db"]),
        "worst_source_mean_db": -float(evidence["worst_source_mean_db"]),
        "worst_source_worst10_db": -float(evidence["worst_source_worst10_db"]),
        "do_no_harm_db": float(evidence["worst_octave_db"]) + threshold,
    }
    flags = dict(evidence["flags"])
    strict_subband = evidence["strict_subband"]
    if strict_subband is not None:
        strict_flags = strict_subband["flags"]
        flags.update(strict_flags)
        # coverage/group power가 있으면 PASS 여부와 무관하게 실제 dB margin을
        # 보존한다. 수치가 -0.1 dB로 아슬아슬하게 실패한 경우를 sentinel
        # clear-fail로 바꾸면 second-seed 규약을 잘못 닫는다.
        if (
            strict_flags["g4_trusted_subband_coverage_pass"]
            and strict_flags["g4_trusted_subband_power_pass"]
        ):
            strict_mean = np.asarray(strict_subband["mean_db"], dtype=np.float64)
            strict_worst10 = np.asarray(
                strict_subband["worst10_db"], dtype=np.float64
            )
            strict_ci_hi = np.asarray(strict_subband["ci_hi_db"], dtype=np.float64)
            if not (
                strict_mean.size
                and strict_worst10.size
                and strict_ci_hi.size
                and np.isfinite(strict_mean).all()
                and np.isfinite(strict_worst10).all()
                and np.isfinite(strict_ci_hi).all()
            ):
                raise ValueError("strict trusted subband G4 margin이 finite/nonempty가 아닙니다")
            margins.update(
                {
                    "strict_subband_mean_db": -float(np.max(strict_mean)),
                    "strict_subband_worst10_db": -float(np.max(strict_worst10)),
                    "strict_subband_ci_hi_db": -float(np.max(strict_ci_hi)),
                }
            )
        else:
            # target energy 또는 독립 group이 부족해 수치 자체를 해석할 수 없는
            # 경우에만 sentinel을 쓴다. 이 경로는 아래에서 inconclusive_data로
            # 분류되며 second seed가 아니라 추가 수집을 요구한다.
            margins["strict_subband_unverified"] = (
                _STRICT_SUBBAND_UNVERIFIED_MARGIN_DB
            )
    else:
        margins["manifest_unbound_diagnostic"] = _STRICT_SUBBAND_UNVERIFIED_MARGIN_DB
    ci_hi = np.asarray(evidence["source_ci_hi_db"], dtype=np.float64)
    if ci_hi.size == 0 or not bool(np.isfinite(ci_hi).all()):
        raise ValueError("recorded-val bootstrap CI 상단이 finite/nonempty가 아닙니다")
    margins["worst_source_ci_hi_db"] = -float(np.max(ci_hi))
    if not all(math.isfinite(value) for value in margins.values()):
        raise ValueError("recorded-val G4 margin에 NaN/Inf가 있습니다")
    selection_metric = float(evidence["trusted"]["worst10_mean_db"])
    if not math.isfinite(selection_metric):
        raise ValueError("recorded-val 선택 지표가 non-finite")
    minimum = min(margins.values())
    # 한 지표가 -0.1 dB여도 다른 필수 지표가 -10 dB면 명확한 FAIL이다.
    # second seed는 *최악 gate*까지 경계 안에 있을 때만 허용한다. PASS 쪽도
    # 최소 여유가 0.3 dB 이내일 때 같은 규칙으로 borderline이다.
    near_boundary = abs(minimum) <= VAL_BORDERLINE_MARGIN_DB
    numeric_pass = all(value >= 0.0 for value in margins.values())
    discrete_pass = all(flags.values()) and verdict == "PASS"
    data_inconclusive_reasons: list[str] = []
    if canonical:
        if not bool(flags.get("g4_power_pass", False)):
            data_inconclusive_reasons.append("family_group_power")
        if not bool(flags.get("g4_trusted_subband_coverage_pass", False)):
            data_inconclusive_reasons.append("strict_subband_target_coverage")
        if not bool(flags.get("g4_trusted_subband_power_pass", False)):
            data_inconclusive_reasons.append("strict_subband_group_power")
    if not canonical:
        status = "diagnostic"
    elif verdict == "INCONCLUSIVE" and data_inconclusive_reasons:
        # target energy나 독립 group 부족은 seed variance가 아니다. 같은 데이터로
        # 100k+50k를 한 번 더 돌려도 표본이 생기지 않으므로 targeted recording을
        # 요구하고 second-seed 자격과 분리한다.
        status = "inconclusive_data"
    elif verdict == "INCONCLUSIVE" or near_boundary:
        status = "borderline"
    elif numeric_pass and discrete_pass:
        status = "clear_pass"
    else:
        status = "clear_fail"
    return {
        "status": status,
        "canonical": canonical,
        "manifest_bound": bool(evidence["manifest_bound"]),
        "sampling_contract": evidence["sampling_contract"],
        "boundary_margin_db": VAL_BORDERLINE_MARGIN_DB,
        "minimum_margin_db": minimum,
        "margins_db": margins,
        "g4_verdict": verdict,
        "g4_flags": flags,
        "data_inconclusive_reasons": data_inconclusive_reasons,
        "selection_metric_db": selection_metric,
    }


def _path_inside_repository(
    root: Path, value: object, *, label: str
) -> Path:
    """canonical authority가 가리키는 실제 regular-file 경로를 저장소에 가둔다."""

    candidate = Path(str(value or "")).expanduser()
    target = Path(os.path.abspath(candidate if candidate.is_absolute() else root / candidate))
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label}는 저장소 내부여야 합니다: {target}") from exc
    if root.is_symlink():
        raise ValueError(f"{label} 저장소 root는 심볼릭 링크일 수 없습니다: {root}")
    cursor = root
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise ValueError(f"{label} 경로에 심볼릭 링크가 있습니다: {cursor}")
    try:
        target.resolve(strict=True).relative_to(root.resolve(strict=True))
    except FileNotFoundError as exc:
        raise ValueError(f"{label}가 없습니다: {target}") from exc
    except ValueError as exc:
        raise ValueError(f"{label} resolved path가 저장소 밖입니다: {target}") from exc
    return target


def validate_canonical_finetune_checkpoint_chain(
    saved_cfg: dict[str, Any],
    checkpoint: FileSnapshot,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """선택 후보가 완료된 50k→동일-seed 100k canonical chain인지 검증한다.

    seed-neutral digest는 두 공식 seed의 학습 의미를 비교하기 위한 projection일 뿐
    init seed/secondary admission 권위가 아니다. 따라서 recorded-test capability와
    cross-seed direct CLI 모두 이 validator를 통과해야 한다.
    """

    from ..config import validate_canonical_training_policy
    from .campaign_prerequisite import validate_canonical_pretrain_prerequisites
    from .completion_receipt import validate_completion_receipt

    if repo_root is None:
        from ..config import REPO_ROOT

        root = Path(REPO_ROOT)
    else:
        root = Path(repo_root)
    root = Path(os.path.abspath(root))
    selected_path = _path_inside_repository(
        root, checkpoint.path, label="canonical fine-tune selection checkpoint"
    )
    if selected_path != checkpoint.path or selected_path.name != "best.pt":
        raise ValueError("selection checkpoint는 canonical fine-tune best.pt여야 합니다")

    validate_canonical_training_policy(saved_cfg)
    seed = saved_cfg.get("seed")
    if (
        saved_cfg.get("experiment_role") != "canonical_finetune"
        or saved_cfg.get("init_eligible") is not False
        or seed not in OFFICIAL_FINETUNE_SEEDS
    ):
        raise ValueError(
            "selection checkpoint는 공식 seed의 canonical_finetune이어야 합니다"
        )
    fine_contract = validate_embedded_experiment_contract(saved_cfg)
    fine_completion = validate_completion_receipt(
        selected_path.parent,
        expected_role="canonical_finetune",
        expected_init_eligible=False,
        repo_root=root,
    )
    if (
        fine_completion.get("experiment_contract_sha256")
        != fine_contract.get("sha256")
        or fine_completion.get("best_checkpoint_sha256") != checkpoint.sha256
        or fine_completion.get("schedule_total_steps") != 50_000
        or fine_completion.get("completed_step") != 50_000
    ):
        raise ValueError(
            "selection checkpoint의 exact canonical 50k completion/contract가 다릅니다"
        )

    init_value = saved_cfg.get("init_ckpt")
    if not isinstance(init_value, str) or not init_value.strip():
        raise ValueError("canonical fine-tune selection에 init_ckpt가 없습니다")
    init_path = _path_inside_repository(
        root, init_value, label="canonical pretrain init checkpoint"
    )
    if init_path.name != "best.pt":
        raise ValueError("canonical fine-tune init_ckpt는 canonical pretrain best.pt여야 합니다")
    init_snapshot = snapshot_regular_file(init_path)
    try:
        init_state = torch.load(
            io.BytesIO(init_snapshot.content), map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("canonical pretrain init checkpoint를 읽을 수 없습니다") from exc
    if not isinstance(init_state, dict) or not isinstance(init_state.get("cfg"), dict):
        raise ValueError("canonical pretrain init checkpoint에 resolved cfg가 없습니다")
    init_cfg = init_state["cfg"]
    validate_canonical_training_policy(init_cfg)
    if (
        init_cfg.get("experiment_role") != "canonical_pretrain"
        or init_cfg.get("init_eligible") is not True
        or init_cfg.get("seed") != seed
    ):
        raise ValueError(
            "canonical fine-tune init은 동일 seed의 init-eligible canonical_pretrain이어야 합니다"
        )
    init_contract = validate_embedded_experiment_contract(init_cfg)
    init_completion = validate_completion_receipt(
        init_path.parent,
        expected_role="canonical_pretrain",
        expected_init_eligible=True,
        repo_root=root,
    )
    if (
        init_completion.get("experiment_contract_sha256")
        != init_contract.get("sha256")
        or init_completion.get("best_checkpoint_sha256") != init_snapshot.sha256
        or init_completion.get("schedule_total_steps") != 100_000
        or init_completion.get("completed_step") != 100_000
    ):
        raise ValueError(
            "fine-tune init의 exact canonical 100k completion/contract가 다릅니다"
        )

    # primary는 raw v7 campaign ledger를, secondary는 그 validator가 dispatch하는
    # 별도 fixed-path prerequisite와 primary sealed selection을 전부 다시 연다.
    validate_canonical_pretrain_prerequisites(init_cfg, repo_root=root)
    return {
        "seed": seed,
        "fine_completion": fine_completion,
        "pretrain_checkpoint": init_snapshot,
        "pretrain_cfg": init_cfg,
        "pretrain_completion": init_completion,
    }


def _validate_selection_candidate(
    payload: dict[str, Any], *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    selected = payload.get("selected")
    if payload.get("selection_split") != "val" or not isinstance(selected, dict):
        raise ValueError("recorded-val selection이 고정되지 않았습니다")
    checkpoint = snapshot_regular_file(selected.get("checkpoint", ""))
    manifest = snapshot_regular_file(payload.get("manifest", ""))
    metrics = snapshot_regular_file(
        Path(str(selected.get("evaluation_dir", ""))) / "metrics.npz"
    )
    if checkpoint.sha256 != selected.get("checkpoint_sha256"):
        raise ValueError("selection checkpoint bytes가 바뀌었습니다")
    if manifest.sha256 != payload.get("manifest_sha256"):
        raise ValueError("selection manifest bytes가 바뀌었습니다")
    if metrics.sha256 != selected.get("metrics_sha256"):
        raise ValueError("selection val metrics bytes가 바뀌었습니다")
    try:
        state = torch.load(
            io.BytesIO(checkpoint.content), map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("selection checkpoint를 읽을 수 없습니다") from exc
    if not isinstance(state, dict) or not isinstance(state.get("cfg"), dict):
        raise ValueError("selection checkpoint에 resolved cfg가 없습니다")
    saved_cfg = state["cfg"]
    embedded = validate_embedded_experiment_contract(saved_cfg)
    if embedded.get("sha256") != payload.get("experiment_contract_sha256"):
        raise ValueError("selection checkpoint embedded contract가 selection과 다릅니다")
    validate_checkpoint_recorded_manifest(saved_cfg, embedded, manifest)
    seed = saved_cfg.get("seed")
    if (
        seed not in OFFICIAL_FINETUNE_SEEDS
        or payload.get("seed") != seed
        or selected.get("seed") != seed
    ):
        raise ValueError("selection/checkpoint official fine-tune seed가 다릅니다")
    calculated_campaign = seed_neutral_campaign_sha256(saved_cfg)
    declared_campaign = _sha256_identity(
        payload.get("seed_neutral_campaign_sha256"), name="seed-neutral campaign"
    )
    if calculated_campaign != declared_campaign or selected.get(
        "seed_neutral_campaign_sha256"
    ) != declared_campaign:
        raise ValueError("selection seed-neutral campaign digest가 checkpoint와 다릅니다")
    validate_canonical_finetune_checkpoint_chain(
        saved_cfg, checkpoint, repo_root=repo_root
    )
    with np.load(io.BytesIO(metrics.content), allow_pickle=False) as data:
        provenance = {
            "split": str(_npz_scalar(data, "split")),
            "checkpoint_sha256": str(_npz_scalar(data, "checkpoint_sha256")),
            "manifest_sha256": str(_npz_scalar(data, "manifest_sha256")),
            "experiment_contract_sha256": str(
                _npz_scalar(data, "experiment_contract_sha256")
            ),
        }
    if provenance != {
        "split": "val",
        "checkpoint_sha256": checkpoint.sha256,
        "manifest_sha256": manifest.sha256,
        "experiment_contract_sha256": payload.get("experiment_contract_sha256"),
    }:
        raise ValueError("selection val metrics provenance가 checkpoint/manifest/contract와 다릅니다")
    decision = classify_recorded_val_metrics(
        metrics.content,
        manifest_bytes=manifest.content,
        manifest_path=manifest.path,
        checkpoint_cfg=saved_cfg,
    )
    if selected.get("decision") != decision:
        raise ValueError("selection decision이 val metrics 재분류와 다릅니다")
    return decision


def validate_recorded_val_selection(
    payload: dict[str, Any], *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    """immutable recorded-val bundle을 raw bytes에서 다시 검증한다.

    single-seed terminal, second-seed admission, cross-seed finalizer가 같은
    validator를 공유하도록 공개한 좁은 API다. 반환값은 저장 claim이 아니라 현재
    checkpoint/manifest/metrics bytes에서 다시 계산한 decision이다.
    """

    return _validate_selection_candidate(payload, repo_root=repo_root)


def validate_test_open_selection(
    payload: dict[str, Any], *, repo_root: str | Path | None = None
) -> None:
    """single clear-pass 또는 검증된 2-seed final만 test 개봉을 허용한다."""

    campaign = _sha256_identity(
        payload.get("seed_neutral_campaign_sha256"), name="seed-neutral campaign"
    )
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("selection decision이 없습니다")
    if decision.get("status") != "cross_seed_final":
        current = _validate_selection_candidate(payload, repo_root=repo_root)
        if payload.get("seed") != 20260803 or current.get("status") != "clear_pass":
            raise ValueError("single-seed test는 seed 20260803 val clear-pass만 허용합니다")
        if decision != current:
            raise ValueError("single-seed top-level decision이 val metrics와 다릅니다")
        return

    records = payload.get("seed_selections")
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("cross-seed final에는 두 seed selection snapshot이 필요합니다")
    bundles: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("cross-seed seed selection record가 mapping이 아닙니다")
        bundle, snapshot = read_json_snapshot(record.get("path", ""))
        if snapshot.sha256 != record.get("sha256"):
            raise ValueError("cross-seed seed selection bytes가 final 뒤 바뀌었습니다")
        seed = record.get("seed")
        if seed not in OFFICIAL_FINETUNE_SEEDS or bundle.get("seed") != seed:
            raise ValueError("cross-seed official seed가 다릅니다")
        if bundle.get("seed_neutral_campaign_sha256") != campaign:
            raise ValueError("cross-seed bundle campaign digest가 다릅니다")
        if bundle.get("manifest_sha256") != payload.get("manifest_sha256"):
            raise ValueError("cross-seed bundle recorded manifest가 다릅니다")
        current = _validate_selection_candidate(bundle, repo_root=repo_root)
        if bundle.get("decision") != current:
            raise ValueError("cross-seed bundle top-level decision이 metrics와 다릅니다")
        bundles.append((int(seed), bundle, current))
    if {seed for seed, _, _ in bundles} != set(OFFICIAL_FINETUNE_SEEDS):
        raise ValueError("cross-seed final official seed 집합이 다릅니다")
    first = next(item for item in bundles if item[0] == 20260803)
    if first[2].get("status") != "borderline":
        raise ValueError(
            "첫 seed가 numeric/CI borderline이 아니어서 2-seed 자격이 없습니다; "
            "data INCONCLUSIVE는 추가 녹음 대상입니다"
        )
    eligible = [
        item
        for item in bundles
        if item[2].get("g4_verdict") == "PASS"
        and float(item[2].get("minimum_margin_db", float("-inf"))) >= 0.0
    ]
    if not eligible:
        raise ValueError("cross-seed final에 val G4 PASS winner가 없습니다")
    winner = max(eligible, key=lambda item: (float(item[2]["minimum_margin_db"]), -item[0]))
    if payload.get("seed") != winner[0] or payload.get("selected") != winner[1].get("selected"):
        raise ValueError("cross-seed final winner가 margin-max selection과 다릅니다")


def canonical_test_ledger_paths_from_payload(
    selection: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """동일 selection snapshot payload에서 3-SHA ledger 경로를 유도한다."""

    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("test ledger selection.selected가 없습니다")
    identity = {
        # ledger scope는 개별 seed/checkpoint가 아니라 전체 campaign이다. 첫 seed를
        # 잘못 개봉한 뒤 다른 seed winner로 두 번째 ledger를 만드는 우회를 막는다.
        "seed_neutral_campaign_sha256": _sha256_identity(
            selection.get("seed_neutral_campaign_sha256"),
            name="seed-neutral campaign",
        ),
        "manifest_sha256": _sha256_identity(
            selection.get("manifest_sha256"), name="manifest"
        ),
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    ledger_id = hashlib.sha256(encoded).hexdigest()
    if repo_root is None:
        from ..config import REPO_ROOT

        root = Path(REPO_ROOT)
    else:
        root = Path(repo_root)
    directory = Path(os.path.abspath(root)) / LEDGER_ROOT / ledger_id
    return directory / "capability.json", directory / "consumed.json"


def canonical_test_ledger_event_paths_from_payload(
    selection: dict[str, Any], *, repo_root: str | Path | None = None
) -> dict[str, Path]:
    capability, consumed = canonical_test_ledger_paths_from_payload(
        selection, repo_root=repo_root
    )
    return {
        "issued": capability,
        "running": consumed,
        "completed": capability.parent / "completed.json",
        "failed": capability.parent / "failed.json",
    }


@contextmanager
def _test_terminal_ledger_lock(paths: dict[str, Path]) -> Iterator[None]:
    """completed/failed 최종 상태 전이를 프로세스 간 직렬화한다.

    ``completed.json``과 ``failed.json``은 서로 다른 pathname이므로 각각을
    no-replace로 공개하는 것만으로는 상호 배타가 되지 않는다. 두 프로세스가
    반대 상태가 아직 없음을 동시에 확인한 뒤 두 파일을 모두 만들 수 있기
    때문이다. 동일 ledger 디렉터리의 영속 advisory lock을 잡은 동안 반대 상태를
    다시 확인하고 최종 마커를 공개해야 한다.

    lock 파일은 의도적으로 삭제하지 않는다. 삭제 후 재생성하면 기존 inode를
    잠근 프로세스와 새 inode를 잠근 프로세스가 동시에 임계구역에 들어갈 수 있다.
    stale 파일 자체에는 권한이 없으며, 커널 lock은 descriptor/프로세스 종료 시
    자동 해제된다. 기존 completed/failed JSON 스키마와 경로는 바꾸지 않는다.
    """

    completed = Path(os.path.abspath(paths["completed"]))
    failed = Path(os.path.abspath(paths["failed"]))
    if completed.parent != failed.parent:
        raise ValueError("test terminal ledger marker는 같은 directory에 있어야 합니다")
    directory = completed.parent
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".terminal.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    locked = False
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"test terminal lock은 regular file이어야 합니다: {lock_path}")
        pathname = lock_path.lstat()
        if stat.S_ISLNK(pathname.st_mode) or (
            pathname.st_dev,
            pathname.st_ino,
        ) != (opened.st_dev, opened.st_ino):
            raise ValueError(f"test terminal lock pathname이 retarget됐습니다: {lock_path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        # lock 대기 중 pathname이 바뀐 경우 별도 inode lock으로 갈라지는 것을
        # 허용하지 않는다. 정상 경로에서는 같은 영속 lock 파일을 계속 재사용한다.
        pathname = lock_path.lstat()
        if stat.S_ISLNK(pathname.st_mode) or (
            pathname.st_dev,
            pathname.st_ino,
        ) != (opened.st_dev, opened.st_ino):
            raise ValueError(f"test terminal lock pathname이 대기 중 바뀌었습니다: {lock_path}")
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def canonical_test_ledger_paths(
    selection_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """seed-neutral campaign+manifest의 저장소 내 유일 test ledger 경로."""

    selection, _ = read_json_snapshot(selection_path)
    return canonical_test_ledger_paths_from_payload(
        selection, repo_root=repo_root
    )


def _require_canonical_ledger_path(
    actual: str | Path, expected: Path, *, label: str
) -> Path:
    target = Path(os.path.abspath(Path(actual)))
    canonical = Path(os.path.abspath(expected))
    if target != canonical:
        raise ValueError(
            f"{label}는 campaign canonical ledger 경로만 허용합니다: "
            f"requested={target}, expected={canonical}"
        )
    return target


def publish_directory_noreplace(staging: str | Path, target: str | Path) -> Path:
    """same-filesystem staging dir을 Linux renameat2(NOREPLACE)로 원자 공개."""

    source = Path(os.path.abspath(Path(staging)))
    destination = Path(os.path.abspath(Path(target)))
    if source.parent != destination.parent:
        raise ValueError("atomic directory publication은 sibling staging만 허용합니다")
    source_stat = source.lstat()
    if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISDIR(source_stat.st_mode):
        raise ValueError("atomic directory staging은 실제 directory여야 합니다")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"평가 산출 디렉터리를 덮어쓸 수 없습니다: {destination}")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("canonical no-replace directory publish에 renameat2가 필요합니다")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(
                f"평가 산출 디렉터리를 덮어쓸 수 없습니다: {destination}"
            )
        raise OSError(error, os.strerror(error), str(destination))
    directory_fd = os.open(
        destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return destination


def issue_test_capability(
    *,
    selection_path: str | Path,
    capability_path: str | Path,
    repo_root: str | Path | None = None,
) -> str:
    selection, selection_snapshot = read_json_snapshot(selection_path)
    selected = selection.get("selected")
    if selection.get("selection_split") != "val" or not isinstance(selected, dict):
        raise ValueError("recorded-val selection이 고정되지 않았습니다")
    validate_test_open_selection(selection, repo_root=repo_root)
    expected_capability, expected_consumed = canonical_test_ledger_paths_from_payload(
        selection, repo_root=repo_root
    )
    event_paths = canonical_test_ledger_event_paths_from_payload(
        selection, repo_root=repo_root
    )
    capability_path = _require_canonical_ledger_path(
        capability_path, expected_capability, label="test capability"
    )
    terminal = [
        path for path in event_paths.values() if path.exists() or path.is_symlink()
    ]
    if terminal:
        raise FileExistsError(
            "이 campaign/manifest test ledger는 이미 발급/소비됐습니다: "
            f"{terminal}"
        )
    token = secrets.token_urlsafe(32)
    payload = {
        "schema_version": 1,
        "phase": "issued",
        "seed_neutral_campaign_sha256": selection.get(
            "seed_neutral_campaign_sha256"
        ),
        "selection_sha256": selection_snapshot.sha256,
        "experiment_contract_sha256": selection.get(
            "experiment_contract_sha256"
        ),
        "selected_checkpoint_sha256": selected.get("checkpoint_sha256"),
        "manifest_sha256": selection.get("manifest_sha256"),
        "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        "issued_at_unix_ns": time.time_ns(),
    }
    write_json_exclusive(capability_path, payload)
    return token


def consume_test_capability(
    *,
    selection_path: str | Path,
    capability_path: str | Path,
    consumed_marker_path: str | Path,
    token: str,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    repo_root: str | Path | None = None,
) -> tuple[FileSnapshot, FileSnapshot, dict[str, Any]]:
    """test bytes를 읽기 직전에 capability를 영구 소비하고 snapshot을 반환한다."""

    if not token:
        raise ValueError(f"test capability token이 없습니다 ({CAPABILITY_ENV})")
    selection, selection_snapshot = read_json_snapshot(selection_path)
    validate_test_open_selection(selection, repo_root=repo_root)
    expected_capability, expected_consumed = canonical_test_ledger_paths_from_payload(
        selection, repo_root=repo_root
    )
    event_paths = canonical_test_ledger_event_paths_from_payload(
        selection, repo_root=repo_root
    )
    capability_path = _require_canonical_ledger_path(
        capability_path, expected_capability, label="test capability"
    )
    consumed_marker_path = _require_canonical_ledger_path(
        consumed_marker_path, expected_consumed, label="test consumed marker"
    )
    for phase in ("running", "completed", "failed"):
        path = event_paths[phase]
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"test ledger phase가 이미 존재합니다: {phase}={path}")
    capability, capability_snapshot = read_json_snapshot(capability_path)
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("selection.selected가 없습니다")
    expected = {
        "selection_sha256": selection_snapshot.sha256,
        "seed_neutral_campaign_sha256": selection.get(
            "seed_neutral_campaign_sha256"
        ),
        "experiment_contract_sha256": selection.get("experiment_contract_sha256"),
        "selected_checkpoint_sha256": selected.get("checkpoint_sha256"),
        "manifest_sha256": selection.get("manifest_sha256"),
        "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
    }
    for key, value in expected.items():
        if capability.get(key) != value:
            raise ValueError(f"test capability {key}가 selection과 다릅니다")
    if capability.get("phase") != "issued":
        raise ValueError("test capability phase가 issued가 아닙니다")
    checkpoint = snapshot_regular_file(checkpoint_path)
    manifest = snapshot_regular_file(manifest_path)
    if checkpoint.sha256 != selected.get("checkpoint_sha256"):
        raise ValueError("selection 뒤 checkpoint bytes가 바뀌었습니다")
    if manifest.sha256 != selection.get("manifest_sha256"):
        raise ValueError("selection 뒤 manifest bytes가 바뀌었습니다")
    marker = {
        "schema_version": 1,
        "phase": "running",
        "seed_neutral_campaign_sha256": selection.get(
            "seed_neutral_campaign_sha256"
        ),
        "selection_sha256": selection_snapshot.sha256,
        "capability_sha256": capability_snapshot.sha256,
        "experiment_contract_sha256": selection.get("experiment_contract_sha256"),
        "selected_checkpoint_sha256": checkpoint.sha256,
        "manifest_sha256": manifest.sha256,
        "consumed_at_unix_ns": time.time_ns(),
    }
    write_json_exclusive(consumed_marker_path, marker)
    return checkpoint, manifest, marker


def _active_test_ledger(
    *,
    selection_path: str | Path,
    capability_path: str | Path,
    consumed_marker_path: str | Path,
    repo_root: str | Path | None,
) -> tuple[dict[str, Any], FileSnapshot, FileSnapshot, dict[str, Path]]:
    selection, selection_snapshot = read_json_snapshot(selection_path)
    validate_test_open_selection(selection, repo_root=repo_root)
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("test ledger selection.selected가 없습니다")
    paths = canonical_test_ledger_event_paths_from_payload(
        selection, repo_root=repo_root
    )
    _require_canonical_ledger_path(capability_path, paths["issued"], label="test capability")
    _require_canonical_ledger_path(
        consumed_marker_path, paths["running"], label="test consumed marker"
    )
    capability, capability_snapshot = read_json_snapshot(paths["issued"])
    running, running_snapshot = read_json_snapshot(paths["running"])
    if (
        capability.get("schema_version") != 1
        or running.get("schema_version") != 1
        or capability.get("phase") != "issued"
        or running.get("phase") != "running"
    ):
        raise ValueError("test ledger issued/running phase가 손상됐습니다")
    selection_identity = {
        "selection_sha256": selection_snapshot.sha256,
        "seed_neutral_campaign_sha256": selection.get(
            "seed_neutral_campaign_sha256"
        ),
        "experiment_contract_sha256": selection.get(
            "experiment_contract_sha256"
        ),
        "selected_checkpoint_sha256": selected.get("checkpoint_sha256"),
        "manifest_sha256": selection.get("manifest_sha256"),
    }
    for key, expected in selection_identity.items():
        if capability.get(key) != expected:
            raise ValueError(f"test capability {key}가 immutable selection과 다릅니다")
        if running.get(key) != expected:
            raise ValueError(f"test running marker {key}가 immutable selection과 다릅니다")
    if running.get("capability_sha256") != capability_snapshot.sha256:
        raise ValueError("test running marker capability SHA가 다릅니다")
    return running, running_snapshot, selection_snapshot, paths


def complete_test_evaluation(
    *,
    selection_path: str | Path,
    capability_path: str | Path,
    consumed_marker_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path | None = None,
) -> Path:
    running, running_snapshot, selection_snapshot, paths = _active_test_ledger(
        selection_path=selection_path,
        capability_path=capability_path,
        consumed_marker_path=consumed_marker_path,
        repo_root=repo_root,
    )
    directory = Path(os.path.abspath(Path(output_dir)))
    directory_stat = directory.lstat()
    if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
        raise ValueError("canonical test output은 실제 directory여야 합니다")
    markdown = snapshot_regular_file(directory / "metrics.md")
    metrics = snapshot_regular_file(directory / "metrics.npz")
    # completed marker는 단순히 ``metrics.npz가 있었다``는 영수증이 아니다. 여기서
    # raw G4/immutable manifest 결속까지 다시 확인해야 malformed/forged test output이
    # ledger completed 상태만 먼저 만들고 completion audit을 압박하는 경로가 없다.
    selection, current_selection_snapshot = read_json_snapshot(selection_path)
    if current_selection_snapshot.sha256 != selection_snapshot.sha256:
        raise ValueError("test completion 직전 selection bytes가 바뀌었습니다")
    selected_manifest = snapshot_regular_file(selection.get("manifest", ""))
    if selected_manifest.sha256 != running.get("manifest_sha256"):
        raise ValueError("test completion manifest bytes가 running ledger와 다릅니다")
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("test completion selection.selected가 없습니다")
    selected_checkpoint = snapshot_regular_file(selected.get("checkpoint", ""))
    if selected_checkpoint.sha256 != running.get("selected_checkpoint_sha256"):
        raise ValueError("test completion checkpoint bytes가 running ledger와 다릅니다")
    try:
        selected_state = torch.load(
            io.BytesIO(selected_checkpoint.content),
            map_location="cpu",
            weights_only=False,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("test completion checkpoint를 읽을 수 없습니다") from exc
    if not isinstance(selected_state, dict) or not isinstance(
        selected_state.get("cfg"), dict
    ):
        raise ValueError("test completion checkpoint에 resolved cfg가 없습니다")
    selected_cfg = selected_state["cfg"]
    try:
        archive = np.load(io.BytesIO(metrics.content), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("test completion metrics.npz가 손상됐습니다") from exc
    with archive as data:
        expected_provenance = {
            "checkpoint_sha256": selected.get("checkpoint_sha256"),
            "experiment_contract_sha256": selection.get(
                "experiment_contract_sha256"
            ),
            "selection_sha256": selection_snapshot.sha256,
            "test_capability_sha256": running.get("capability_sha256"),
            "test_consumed_marker_sha256": running_snapshot.sha256,
        }
        for key, expected in expected_provenance.items():
            actual = _g4_required_str_scalar(data, key)
            if actual != expected:
                raise ValueError(
                    f"test completion metrics {key}가 one-shot ledger와 다릅니다"
                )
        g4_evidence = validate_persisted_g4_metrics(
            data,
            expected_split="test",
            manifest_bytes=selected_manifest.content,
            manifest_path=selected_manifest.path,
            checkpoint_cfg=selected_cfg,
            canonical=True,
            min_groups=MIN_GROUPS_PER_FAMILY,
        )
    g4_verdict = str(g4_evidence.get("g4_verdict", ""))
    if g4_verdict not in {"PASS", "FAIL", "INCONCLUSIVE"}:
        raise ValueError(
            f"test completion raw G4 verdict가 승인 enum이 아닙니다: {g4_verdict!r}"
        )
    common_payload = {
        "schema_version": 1,
        "selection_sha256": selection_snapshot.sha256,
        "running_marker_sha256": running_snapshot.sha256,
        "experiment_contract_sha256": running.get("experiment_contract_sha256"),
        "seed_neutral_campaign_sha256": running.get(
            "seed_neutral_campaign_sha256"
        ),
        "selected_checkpoint_sha256": running.get("selected_checkpoint_sha256"),
        "manifest_sha256": running.get("manifest_sha256"),
        "output_dir": str(directory),
        "metrics_markdown_sha256": markdown.sha256,
        "metrics_npz_sha256": metrics.sha256,
        "g4_verdict": g4_verdict,
    }
    if g4_verdict == "PASS":
        terminal_path = paths["completed"]
        payload = {
            **common_payload,
            "phase": "completed",
            "completed_at_unix_ns": time.time_ns(),
        }
    else:
        # 평가 계산 자체는 유효하므로 raw output directory를 삭제하거나 일반 예외로
        # 바꾸지 않는다. 다만 FAIL/INCONCLUSIVE는 deployment/readiness completion이
        # 아니며, immutable raw SHA와 verdict를 failed terminal marker에 봉인한다.
        terminal_path = paths["failed"]
        payload = {
            **common_payload,
            "phase": "failed",
            "error_type": f"G4_{g4_verdict}",
            "failure_class": "valid_g4_terminal_rejection",
            "failed_at_unix_ns": time.time_ns(),
        }
    with _test_terminal_ledger_lock(paths):
        if g4_verdict == "PASS":
            if paths["failed"].exists() or paths["failed"].is_symlink():
                raise FileExistsError("failed test ledger는 completed로 승격할 수 없습니다")
        elif paths["completed"].exists() or paths["completed"].is_symlink():
            raise FileExistsError("completed test ledger를 G4 rejection으로 바꿀 수 없습니다")
        write_json_exclusive(terminal_path, payload)
    return terminal_path


def fail_test_evaluation(
    *,
    selection_path: str | Path,
    capability_path: str | Path,
    consumed_marker_path: str | Path,
    error_type: str,
    repo_root: str | Path | None = None,
) -> Path:
    running, running_snapshot, selection_snapshot, paths = _active_test_ledger(
        selection_path=selection_path,
        capability_path=capability_path,
        consumed_marker_path=consumed_marker_path,
        repo_root=repo_root,
    )
    payload = {
        "schema_version": 1,
        "phase": "failed",
        "selection_sha256": selection_snapshot.sha256,
        "running_marker_sha256": running_snapshot.sha256,
        "experiment_contract_sha256": running.get("experiment_contract_sha256"),
        "seed_neutral_campaign_sha256": running.get(
            "seed_neutral_campaign_sha256"
        ),
        "selected_checkpoint_sha256": running.get("selected_checkpoint_sha256"),
        "manifest_sha256": running.get("manifest_sha256"),
        "error_type": str(error_type),
        "failed_at_unix_ns": time.time_ns(),
    }
    with _test_terminal_ledger_lock(paths):
        if paths["completed"].exists() or paths["completed"].is_symlink():
            raise FileExistsError("completed test ledger를 failed로 바꿀 수 없습니다")
        write_json_exclusive(paths["failed"], payload)
    return paths["failed"]


__all__ = [
    "CAPABILITY_ENV",
    "CANONICAL_G4_SOURCE_FAMILIES",
    "FileSnapshot",
    "consume_test_capability",
    "canonical_test_ledger_paths",
    "canonical_test_ledger_paths_from_payload",
    "canonical_test_ledger_event_paths_from_payload",
    "classify_recorded_val_metrics",
    "complete_test_evaluation",
    "issue_test_capability",
    "fail_test_evaluation",
    "publish_directory_noreplace",
    "read_json_snapshot",
    "seed_neutral_campaign_sha256",
    "snapshot_regular_file",
    "validate_canonical_finetune_checkpoint_chain",
    "validate_checkpoint_recorded_manifest",
    "validate_persisted_g4_metrics",
    "validate_recorded_val_selection",
    "validate_test_open_selection",
    "write_json_exclusive",
]
