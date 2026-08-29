"""공식 recorded G4의 strict trusted-band 부대역 계약.

``150–1600 Hz 평균`` 하나만으로는 1000–1600 Hz의 실패를 숨길 수 있다. 이 모듈은
공식 실측 G4가 반드시 함께 보는 네 구간과, 그 구간에 실제 취소 대상 에너지가 있었는지
판정하는 규약의 단일 출처다. 1600 Hz 밖을 신뢰 대역으로 넓히지 않는다.

이 파일은 오디오 장치를 열지 않는다.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


STRICT_TRUSTED_BAND_HZ: tuple[float, float] = (150.0, 1600.0)
"""현재 strict P/S가 보증하는 공식 G4 trusted 대역."""

STRICT_TRUSTED_SUBBANDS_HZ: tuple[tuple[float, float], ...] = (
    (150.0, 300.0),
    (300.0, 600.0),
    (600.0, 1000.0),
    (1000.0, 1600.0),
)
"""공식 trusted-band 분할.

FFT 집계는 :func:`strict_subband_includes_upper_edge`에 따라 앞 세 구간 [lo, hi),
마지막 구간 [lo, hi]로 읽으므로 경계가 겹치지 않고 150–1600 Hz를 완전히 덮는다.
"""

STRICT_TRUSTED_SUBBAND_SCHEMA = "strict_trusted_150_1600_subbands_v1"
"""metrics.npz에 저장하는 fail-closed schema 식별자."""

MIN_SUBBAND_SOURCE_ENERGY_DENSITY_RATIO = 0.25
"""각 부대역 target(=ERR ``d``) coverage의 현재 임시 폭 정규화 PSD 하한.

부대역의 target(=ERR ``d``) 전력을 150–1600 Hz 전체 target 전력으로 나눈 뒤,
그 부대역의 폭 비율로 다시 나눈 값이다. 평탄한 스펙트럼은 1.0이고, 해당 부대역에
실제 신호가 없는 경우는 0에 가깝다. 0.25는 live octave target-energy gate와 같은
고역에 신호가 없는 녹음의 filter/microphone floor를 "고역 ANC"라고 잘못 주장하는
일을 막기 위한 fail-closed provisional 값이다. **82세션의 family×subband density
분포 audit으로 정당화·계약 결속되기 전에는 이 숫자 자체를 성능/데이터 적합성 근거로
주장하지 않는다.** metrics에는 원 density와 이 threshold를 함께 보존한다.
"""

MIN_GROUPS_PER_FAMILY = 4
"""G4 family별 독립 group/strict 부대역 coverage의 공통 통계 하한."""


def is_strict_trusted_band(value: Sequence[float]) -> bool:
    """``value``가 현재 공식 trusted 대역과 정확히 같은지 판정한다.

    float serialization의 미세 오차만 허용하고, 150–600 legacy 또는 1600 Hz 밖 확장은
    공식 네 부대역 G4로 조용히 해석하지 않는다.
    """

    try:
        items = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return False
    if len(items) != 2 or not all(math.isfinite(item) for item in items):
        return False
    return all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-6)
        for actual, expected in zip(items, STRICT_TRUSTED_BAND_HZ, strict=True)
    )


def strict_trusted_subbands_for(
    trusted_band_hz: Sequence[float],
) -> tuple[tuple[float, float], ...] | None:
    """공식 150–1600 Hz일 때만 canonical partition을 반환한다.

    ``None``은 진단/legacy 대역이지, 빈 부대역을 통과로 읽으라는 뜻이 아니다.
    ``write_recorded_metrics``와 completion audit은 이 경우 final G4 PASS를 금지한다.
    """

    if not is_strict_trusted_band(trusted_band_hz):
        return None
    return STRICT_TRUSTED_SUBBANDS_HZ


def strict_subband_includes_upper_edge(band_hz: Sequence[float]) -> bool:
    """strict partition의 FFT 경계 포함 규약을 반환한다.

    네 부대역은 물리적으로 겹치지 않는 half-open 분할이고, 마지막 1000–1600 Hz만
    1600 Hz를 보존하려고 upper edge를 포함한다. 따라서 정확히 1000 Hz인 target
    성분은 1000–1600 Hz에만 속하며 두 인접 부대역을 동시에 채울 수 없다.
    """

    try:
        band = tuple(float(item) for item in band_hz)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"strict trusted 부대역은 [lo, hi]여야 합니다: {band_hz!r}") from exc
    if band not in STRICT_TRUSTED_SUBBANDS_HZ:
        raise ValueError(f"strict trusted 부대역이 아닙니다: {band!r}")
    return band == STRICT_TRUSTED_SUBBANDS_HZ[-1]


def source_energy_density_ratio(
    subband_power: float,
    trusted_power: float,
    band_hz: Sequence[float],
) -> float:
    """target ``d``의 canonical 부대역 energy-density ratio를 계산한다.

    API 이름은 기존 evaluator/metrics 호환성을 위해 유지한다. 여기서 ``source``는
    원본 ``source_aligned.wav``가 아니라 ANC 취소 대상인 ERR target ``d``를 뜻한다.
    """

    try:
        lo, hi = (float(item) for item in band_hz)
    except (TypeError, ValueError) as exc:  # pragma: no cover - 호출자 방어
        raise ValueError(f"부대역은 [lo, hi]여야 합니다: {band_hz!r}") from exc
    if (lo, hi) not in STRICT_TRUSTED_SUBBANDS_HZ:
        raise ValueError(f"strict trusted 부대역이 아닙니다: [{lo:g}, {hi:g}]")
    numerator = float(subband_power)
    denominator = float(trusted_power)
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        raise ValueError("target(=ERR d) energy power는 유한해야 합니다")
    if numerator < 0.0 or denominator < 0.0:
        raise ValueError("target(=ERR d) energy power는 0 이상이어야 합니다")
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0
    flat_fraction = (hi - lo) / (
        STRICT_TRUSTED_BAND_HZ[1] - STRICT_TRUSTED_BAND_HZ[0]
    )
    return float((numerator / denominator) / flat_fraction)


def source_energy_covered(
    subband_power: float,
    trusted_power: float,
    band_hz: Sequence[float],
    *,
    min_density_ratio: float = MIN_SUBBAND_SOURCE_ENERGY_DENSITY_RATIO,
) -> tuple[bool, float]:
    """부대역에 실제 target 에너지가 충분한지와 density ratio를 반환한다."""

    threshold = float(min_density_ratio)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("target(=ERR d) energy density 하한은 0보다 크고 1 이하여야 합니다")
    ratio = source_energy_density_ratio(subband_power, trusted_power, band_hz)
    return bool(ratio >= threshold), ratio


def cluster_bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    min_groups: int,
    n_resamples: int = 10_000,
    seed: int = 20260805,
    alpha: float = 0.05,
) -> tuple[float, float, int]:
    """그룹 단위 평균 bootstrap CI. 그룹 부족 시 수치를 지어내지 않는다."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    if values.size != groups.size:
        raise ValueError(f"값과 그룹 길이가 다릅니다: {values.size} != {groups.size}")
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("nan"), float("nan"), int(np.unique(groups).size)
    minimum = int(min_groups)
    if minimum < 1:
        raise ValueError("min_groups는 1 이상이어야 합니다")
    if int(n_resamples) <= 0:
        raise ValueError("n_resamples는 양수여야 합니다")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha는 0과 1 사이여야 합니다")
    unique = np.unique(groups)
    if unique.size < minimum:
        return float("nan"), float("nan"), int(unique.size)
    by_group = [values[groups == key] for key in unique]
    # 각 bootstrap draw는 뽑힌 group 안의 모든 segment를 붙여 평균낸다. 따라서
    # group별 sum/count를 따로 더하면 기존 concatenate 구현과 수치적으로 같은데,
    # family×4 subband의 16 CI를 평가할 때 Python 160,000회 루프를 피할 수 있다.
    # count를 함께 더해야 group 크기가 다른 경우에도 segment 가중 평균이 보존된다.
    group_sums = np.asarray([float(item.sum()) for item in by_group], dtype=np.float64)
    group_counts = np.asarray([int(item.size) for item in by_group], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    picks = rng.integers(0, unique.size, size=(int(n_resamples), unique.size))
    draws = group_sums[picks].sum(axis=1) / group_counts[picks].sum(axis=1)
    lo = float(np.percentile(draws, 100.0 * float(alpha) / 2.0))
    hi = float(np.percentile(draws, 100.0 * (1.0 - float(alpha) / 2.0)))
    return lo, hi, int(unique.size)


_METRIC_ARRAY_FIELDS = (
    "source_trusted_subband_n_segments",
    "source_trusted_subband_n_groups",
    "source_trusted_subband_coverage_fraction",
    "source_trusted_subband_source_energy_density_ratio_mean",
    "source_trusted_subband_nmse_mean_db",
    "source_trusted_subband_nmse_worst10_mean_db",
    "source_trusted_subband_ci_lo_db",
    "source_trusted_subband_ci_hi_db",
    "source_trusted_subband_coverage_pass",
    "source_trusted_subband_power_pass",
    "source_trusted_subband_mean_pass",
    "source_trusted_subband_worst10_pass",
    "source_trusted_subband_ci_pass",
    "source_trusted_subband_pass",
)
_METRIC_FLAG_FIELDS = (
    "g4_trusted_subband_schema_pass",
    "g4_trusted_subband_coverage_pass",
    "g4_trusted_subband_power_pass",
    "g4_trusted_subband_mean_pass",
    "g4_trusted_subband_worst10_pass",
    "g4_trusted_subband_ci_pass",
    "g4_trusted_subband_pass",
    "g4_upper_trusted_subband_pass",
)


def _require_scalar_kind(data, key: str, kinds: set[str], label: str) -> object:
    if key not in data.files:
        raise ValueError(f"strict trusted subband metrics 필드가 없습니다: {key}")
    value = np.asarray(data[key])
    if value.size != 1 or value.dtype.kind not in kinds:
        raise ValueError(
            f"strict trusted subband {key}는 {label} scalar여야 합니다"
        )
    return value.reshape(-1)[0].item()


def _dtype_matches(value: np.ndarray, dtype: object) -> bool:
    expected_kind = np.dtype(dtype).kind
    if expected_kind == "U":
        return value.dtype.kind in {"U", "S"}
    if expected_kind in {"i", "u"}:
        return value.dtype.kind in {"i", "u"}
    return value.dtype.kind == expected_kind


def _require_shape(data, key: str, shape: tuple[int, ...], dtype) -> np.ndarray:
    if key not in data.files:
        raise ValueError(f"strict trusted subband metrics 필드가 없습니다: {key}")
    raw = np.asarray(data[key])
    if not _dtype_matches(raw, dtype):
        raise ValueError(
            f"strict trusted subband {key} dtype={raw.dtype}; "
            f"expected kind={np.dtype(dtype).kind}"
        )
    value = raw.astype(dtype, copy=False)
    if value.shape != shape:
        raise ValueError(
            f"strict trusted subband {key} shape={value.shape}; expected={shape}"
        )
    return value


def _same_numeric(actual: np.ndarray, expected: np.ndarray, *, key: str) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=1.0e-10, equal_nan=True):
        raise ValueError(f"strict trusted subband {key}가 raw segment 재계산값과 다릅니다")


def validate_strict_trusted_subband_metrics(
    data,
    *,
    min_groups: int,
) -> dict:
    """metrics.npz의 strict 부대역 G4 증거를 raw segment에서 재감사한다.

    Legacy metrics에는 이 schema/배열이 없으므로 예외로 거부한다. 새 파일도 summary
    boolean만 믿지 않고 per-segment coverage/NMSE/group에서 mean, worst10, cluster CI를
    다시 계산한다. 따라서 1000–1600 Hz를 생략하거나 평균 속에 숨긴 artifact는 final
    completion에 도달할 수 없다.
    """

    required = {
        "strict_trusted_subband_schema",
        "strict_trusted_subband_min_source_energy_density_ratio",
        "trusted_subband_hz",
        "source_family",
        "n_segments",
        "n_sessions",
        "segment_session_id",
        "segment_source_family",
        "segment_group_id",
        "per_segment_trusted_subband_nmse_db",
        "per_segment_trusted_subband_coverage",
        "per_segment_trusted_subband_source_energy_density_ratio",
        *_METRIC_ARRAY_FIELDS,
        *_METRIC_FLAG_FIELDS,
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(
            "strict trusted 150–1600Hz 부대역을 판정하지 않는 구버전 metrics입니다: "
            + ", ".join(missing)
        )
    if str(
        _require_scalar_kind(
            data,
            "strict_trusted_subband_schema",
            {"U", "S"},
            "string",
        )
    ) != (
        STRICT_TRUSTED_SUBBAND_SCHEMA
    ):
        raise ValueError("strict trusted subband schema가 canonical v1이 아닙니다")
    density_threshold = float(
        _require_scalar_kind(
            data,
            "strict_trusted_subband_min_source_energy_density_ratio",
            {"f"},
            "floating-point",
        )
    )
    if not math.isfinite(density_threshold) or not 0.0 < density_threshold <= 1.0:
        raise ValueError("strict trusted subband target-energy density threshold가 유효하지 않습니다")
    # metrics.npz 안의 scalar는 evidence이지 정책 authority가 아니다. 이 값을 낮춰
    # coverage mask/summary를 함께 다시 쓰면 더 약한 gate를 통과할 수 있으므로, 현재
    # canonical contract의 단일 출처와 bit-exact하게 같아야 한다. 82-session audit 뒤
    # 정책 값을 바꿀 때도 evaluator·validator가 이 constant를 함께 사용한다.
    if density_threshold != float(MIN_SUBBAND_SOURCE_ENERGY_DENSITY_RATIO):
        raise ValueError(
            "strict trusted subband target-energy density threshold가 canonical 정책과 다릅니다"
        )
    bands = _require_shape(
        data,
        "trusted_subband_hz",
        (len(STRICT_TRUSTED_SUBBANDS_HZ), 2),
        np.float64,
    )
    expected_bands = np.asarray(STRICT_TRUSTED_SUBBANDS_HZ, dtype=np.float64)
    if bands.shape != expected_bands.shape or not np.array_equal(bands, expected_bands):
        raise ValueError("strict trusted subband 경계가 canonical 150–1600Hz 분할과 다릅니다")

    families_raw = np.asarray(data["source_family"])
    if families_raw.ndim != 1 or families_raw.dtype.kind not in {"U", "S"}:
        raise ValueError(
            "strict trusted subband source_family는 string 1차원 배열이어야 합니다"
        )
    families = families_raw.astype(str, copy=False)
    if families.size == 0 or any(not item for item in families) or len(set(families)) != families.size:
        raise ValueError("strict trusted subband source_family가 비었거나 중복됩니다")
    n_segments = int(
        _require_scalar_kind(data, "n_segments", {"i", "u"}, "integer")
    )
    n_sessions = int(
        _require_scalar_kind(data, "n_sessions", {"i", "u"}, "integer")
    )
    segment_session = _require_shape(
        data, "segment_session_id", (n_segments,), np.str_
    ).astype(str, copy=False)
    segment_family = _require_shape(
        data, "segment_source_family", (n_segments,), np.str_
    ).astype(str, copy=False)
    segment_group = _require_shape(
        data, "segment_group_id", (n_segments,), np.str_
    ).astype(str, copy=False)
    expected_per_segment_shape = (n_segments, len(STRICT_TRUSTED_SUBBANDS_HZ))
    values = _require_shape(
        data,
        "per_segment_trusted_subband_nmse_db",
        expected_per_segment_shape,
        np.float64,
    )
    coverage = _require_shape(
        data,
        "per_segment_trusted_subband_coverage",
        expected_per_segment_shape,
        np.bool_,
    )
    density = _require_shape(
        data,
        "per_segment_trusted_subband_source_energy_density_ratio",
        expected_per_segment_shape,
        np.float64,
    )
    if (
        segment_session.size != n_segments
        or segment_family.size != n_segments
        or segment_group.size != n_segments
    ):
        raise ValueError("strict trusted subband segment metadata 길이가 n_segments와 다릅니다")
    if n_sessions <= 0 or n_sessions != int(np.unique(segment_session).size):
        raise ValueError(
            "strict trusted subband n_sessions가 raw segment_session_id와 다릅니다"
        )
    if (
        any(not item for item in segment_session)
        or any(not item for item in segment_family)
        or any(not item for item in segment_group)
    ):
        raise ValueError("strict trusted subband segment session/family/group가 비었습니다")
    segment_families = set(segment_family.tolist())
    declared_families = set(families.tolist())
    if segment_families != declared_families:
        raise ValueError(
            "strict trusted subband source_family와 raw segment_source_family 집합이 다릅니다"
        )
    for session in np.unique(segment_session):
        session_mask = segment_session == session
        if np.unique(segment_family[session_mask]).size != 1:
            raise ValueError(
                "strict trusted subband 한 session이 여러 source family에 걸쳐 있습니다"
            )
        if np.unique(segment_group[session_mask]).size != 1:
            raise ValueError(
                "strict trusted subband 한 session이 여러 lineage group에 걸쳐 있습니다"
            )
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(density)):
        raise ValueError("strict trusted subband raw NMSE/target-energy에 NaN/Inf가 있습니다")
    if np.any(density < 0.0):
        raise ValueError("strict trusted subband target-energy density가 음수입니다")
    expected_coverage = density >= density_threshold
    if not np.array_equal(coverage, expected_coverage):
        raise ValueError(
            "strict trusted subband coverage mask가 target-energy density 규약과 다릅니다"
        )

    shape = (families.size, len(STRICT_TRUSTED_SUBBANDS_HZ))
    n_covered = np.zeros(shape, dtype=np.int64)
    n_groups = np.zeros(shape, dtype=np.int64)
    fraction = np.zeros(shape, dtype=np.float64)
    density_mean = np.full(shape, np.nan, dtype=np.float64)
    mean = np.full(shape, np.nan, dtype=np.float64)
    worst10 = np.full(shape, np.nan, dtype=np.float64)
    ci_lo = np.full(shape, np.nan, dtype=np.float64)
    ci_hi = np.full(shape, np.nan, dtype=np.float64)
    coverage_pass = np.zeros(shape, dtype=np.bool_)
    power_pass = np.zeros(shape, dtype=np.bool_)
    mean_pass = np.zeros(shape, dtype=np.bool_)
    worst10_pass = np.zeros(shape, dtype=np.bool_)
    ci_pass = np.zeros(shape, dtype=np.bool_)
    subband_pass = np.zeros(shape, dtype=np.bool_)
    minimum = int(min_groups)
    if minimum < 1:
        raise ValueError("min_groups는 1 이상이어야 합니다")

    for family_index, family in enumerate(families):
        family_mask = segment_family == family
        if not np.any(family_mask):
            raise ValueError(f"strict trusted subband {family} raw segment가 없습니다")
        for band_index in range(len(STRICT_TRUSTED_SUBBANDS_HZ)):
            mask = coverage[family_mask, band_index]
            selected = values[family_mask, band_index][mask]
            selected_groups = segment_group[family_mask][mask]
            n_covered[family_index, band_index] = int(selected.size)
            n_groups[family_index, band_index] = int(np.unique(selected_groups).size)
            fraction[family_index, band_index] = float(np.mean(mask))
            density_mean[family_index, band_index] = float(
                np.mean(density[family_mask, band_index])
            )
            coverage_pass[family_index, band_index] = bool(selected.size > 0)
            power_pass[family_index, band_index] = bool(
                coverage_pass[family_index, band_index]
                and n_groups[family_index, band_index] >= minimum
            )
            if selected.size:
                mean[family_index, band_index] = float(np.mean(selected))
                count = max(1, int(math.ceil(selected.size * 0.1)))
                worst10[family_index, band_index] = float(
                    np.mean(np.sort(selected)[-count:])
                )
                lo, hi, _ = cluster_bootstrap_ci(
                    selected,
                    selected_groups,
                    min_groups=minimum,
                )
                ci_lo[family_index, band_index] = lo
                ci_hi[family_index, band_index] = hi
                mean_pass[family_index, band_index] = bool(
                    mean[family_index, band_index] < 0.0
                )
                worst10_pass[family_index, band_index] = bool(
                    worst10[family_index, band_index] < 0.0
                )
                ci_pass[family_index, band_index] = bool(
                    power_pass[family_index, band_index]
                    and math.isfinite(hi)
                    and hi < 0.0
                )
            subband_pass[family_index, band_index] = bool(
                coverage_pass[family_index, band_index]
                and power_pass[family_index, band_index]
                and mean_pass[family_index, band_index]
                and worst10_pass[family_index, band_index]
                and ci_pass[family_index, band_index]
            )

    stored = {
        "source_trusted_subband_n_segments": (n_covered, np.int64),
        "source_trusted_subband_n_groups": (n_groups, np.int64),
        "source_trusted_subband_coverage_fraction": (fraction, np.float64),
        "source_trusted_subband_source_energy_density_ratio_mean": (
            density_mean,
            np.float64,
        ),
        "source_trusted_subband_nmse_mean_db": (mean, np.float64),
        "source_trusted_subband_nmse_worst10_mean_db": (worst10, np.float64),
        "source_trusted_subband_ci_lo_db": (ci_lo, np.float64),
        "source_trusted_subband_ci_hi_db": (ci_hi, np.float64),
        "source_trusted_subband_coverage_pass": (coverage_pass, np.bool_),
        "source_trusted_subband_power_pass": (power_pass, np.bool_),
        "source_trusted_subband_mean_pass": (mean_pass, np.bool_),
        "source_trusted_subband_worst10_pass": (worst10_pass, np.bool_),
        "source_trusted_subband_ci_pass": (ci_pass, np.bool_),
        "source_trusted_subband_pass": (subband_pass, np.bool_),
    }
    for key, (expected, dtype) in stored.items():
        actual = _require_shape(data, key, shape, dtype)
        if np.issubdtype(np.dtype(dtype), np.bool_) or np.issubdtype(
            np.dtype(dtype), np.integer
        ):
            if not np.array_equal(actual, expected):
                raise ValueError(
                    f"strict trusted subband {key}가 raw segment 재계산값과 다릅니다"
                )
        else:
            _same_numeric(actual, expected, key=key)

    expected_flags = {
        "g4_trusted_subband_schema_pass": True,
        "g4_trusted_subband_coverage_pass": bool(
            coverage_pass.size and np.all(coverage_pass)
        ),
        "g4_trusted_subband_power_pass": bool(
            power_pass.size and np.all(power_pass)
        ),
        "g4_trusted_subband_mean_pass": bool(mean_pass.size and np.all(mean_pass)),
        "g4_trusted_subband_worst10_pass": bool(
            worst10_pass.size and np.all(worst10_pass)
        ),
        "g4_trusted_subband_ci_pass": bool(ci_pass.size and np.all(ci_pass)),
        "g4_trusted_subband_pass": bool(
            subband_pass.size and np.all(subband_pass)
        ),
        "g4_upper_trusted_subband_pass": bool(
            subband_pass.shape[1] and np.all(subband_pass[:, -1])
        ),
    }
    for key, expected in expected_flags.items():
        if bool(_require_scalar_kind(data, key, {"b"}, "bool")) != expected:
            raise ValueError(
                f"strict trusted subband {key}가 raw segment 재계산값과 다릅니다"
            )
    return {
        "source_families": families.tolist(),
        "subbands_hz": bands.tolist(),
        "source_energy_density_threshold": density_threshold,
        "flags": expected_flags,
        "mean_db": mean,
        "worst10_db": worst10,
        "ci_hi_db": ci_hi,
    }


__all__ = [
    "MIN_SUBBAND_SOURCE_ENERGY_DENSITY_RATIO",
    "MIN_GROUPS_PER_FAMILY",
    "STRICT_TRUSTED_BAND_HZ",
    "STRICT_TRUSTED_SUBBAND_SCHEMA",
    "STRICT_TRUSTED_SUBBANDS_HZ",
    "cluster_bootstrap_ci",
    "is_strict_trusted_band",
    "source_energy_covered",
    "source_energy_density_ratio",
    "strict_subband_includes_upper_edge",
    "strict_trusted_subbands_for",
    "validate_strict_trusted_subband_metrics",
]
