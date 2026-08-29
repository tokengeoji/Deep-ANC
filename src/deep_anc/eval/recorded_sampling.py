"""Canonical recorded G4의 결정론적 segment sampling 단일 출처.

성능 지표가 좋아도 세션마다 좋은 구간 몇 개만 고르면 실제 모집단의 성능이 아니다.
이 모듈의 상수와 시작점 함수는 coverage 감사, evaluator, persisted G4 validator가
공유한다. 오디오 장치를 열지 않는다.
"""

from __future__ import annotations

import math

import numpy as np


RECORDED_SAMPLING_CONTRACT_SCHEMA = "recorded-deterministic-nonoverlap/v2"
CANONICAL_MAX_SEGMENTS_PER_SESSION = 64
CANONICAL_EDGE_TRIM_SECONDS = 0.25
CANONICAL_SEGMENT_SECONDS = 1.5


def effective_segment_samples(
    *, sample_rate: int, model_hop: int, segment_seconds: float
) -> int:
    """요청 길이를 evaluator와 같은 hop 정렬 sample 수로 바꾼다."""

    rate = int(sample_rate)
    hop = int(model_hop)
    seconds = float(segment_seconds)
    if rate <= 0 or hop <= 0:
        raise ValueError("recorded sampling sample_rate/model_hop은 양수여야 합니다")
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("recorded sampling segment_seconds는 유한한 양수여야 합니다")
    raw = int(round(seconds * rate))
    aligned = (raw // hop) * hop
    if aligned < hop:
        raise ValueError("recorded sampling segment가 model hop보다 짧습니다")
    return aligned


def canonical_feedback_delay_samples(data_cfg: dict) -> int:
    """checkpoint 학습 범위에서 canonical 평가가 쓰는 유일한 feedback 지연."""

    raw = (data_cfg.get("closed_loop") or {}).get(
        "feedback_delay_samples", [0, 0]
    )
    if isinstance(raw, bool):
        raise ValueError("recorded sampling feedback delay는 bool일 수 없습니다")
    if isinstance(raw, (int, float)):
        lo = hi = int(raw)
    else:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(
                "recorded sampling closed_loop.feedback_delay_samples는 [lo, hi]여야 합니다"
            )
        lo, hi = int(raw[0]), int(raw[1])
    if lo < 0 or hi < lo:
        raise ValueError(
            f"recorded sampling feedback delay 범위가 잘못됐습니다: {raw!r}"
        )
    # eval.recorded.resolve_feedback_delay(requested=None)와 같은 Python round 규약.
    return int(round((lo + hi) / 2.0))


def canonical_warmup_samples(
    data_cfg: dict,
    *,
    sample_rate: int,
    plant_settle_samples: int,
) -> int:
    """checkpoint 기본 warmup과 불가피한 plant 정착 구간의 exact 최댓값."""

    rate = int(sample_rate)
    settle = int(plant_settle_samples)
    seconds = float((data_cfg.get("closed_loop") or {}).get("warmup_seconds", 0.25))
    if rate <= 0 or settle < 0:
        raise ValueError("recorded sampling sample_rate/plant_settle이 유효하지 않습니다")
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("recorded sampling warmup_seconds는 유한한 0 이상이어야 합니다")
    return max(settle, int(round(seconds * rate)))


def deterministic_segment_starts(
    usable_samples: int,
    segment_samples: int,
    max_segments: int,
    edge_trim_samples: int = 0,
) -> list[int]:
    """양끝 trim 뒤 고르게 분포한 비중첩 segment 시작점을 반환한다."""

    usable = int(usable_samples)
    segment = int(segment_samples)
    maximum = int(max_segments)
    trim = int(edge_trim_samples)
    if segment <= 0:
        raise ValueError("segment_samples는 양수여야 합니다")
    if maximum <= 0:
        raise ValueError("max_segments는 양수여야 합니다")
    if trim < 0:
        raise ValueError("edge_trim_samples는 0 이상이어야 합니다")
    trimmed = usable - 2 * trim
    count = trimmed // segment
    if count <= 0:
        return []
    candidates = np.arange(count, dtype=np.int64) * segment + trim
    if count <= maximum:
        return [int(value) for value in candidates]
    indices = np.linspace(0, count - 1, num=maximum, dtype=np.int64)
    return [int(candidates[index]) for index in indices]


__all__ = [
    "CANONICAL_EDGE_TRIM_SECONDS",
    "CANONICAL_MAX_SEGMENTS_PER_SESSION",
    "CANONICAL_SEGMENT_SECONDS",
    "RECORDED_SAMPLING_CONTRACT_SCHEMA",
    "canonical_feedback_delay_samples",
    "canonical_warmup_samples",
    "deterministic_segment_starts",
    "effective_segment_samples",
]
