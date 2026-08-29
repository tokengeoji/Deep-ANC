"""실측 target ``d`` strict 부대역 coverage 사전계산 증거 계약.

FFT를 매 readiness 호출마다 다시 수행하지 않는 대신, 보고서는 현재 manifest bytes와
resolved timing/평가 파라미터에 정확히 결속된다. 이 모듈은 보고서의 구조와 집계도
재검산하며, 하나라도 확인할 수 없으면 예외로 닫는다. 오디오 장치는 열지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..dsp.timing import TrainingTimingContract
from ..eval.trusted_subbands import (
    MIN_GROUPS_PER_FAMILY,
    MIN_SUBBAND_SOURCE_ENERGY_DENSITY_RATIO,
    STRICT_TRUSTED_BAND_HZ,
    STRICT_TRUSTED_SUBBANDS_HZ,
)
from ..eval.recorded_sampling import (
    CANONICAL_EDGE_TRIM_SECONDS,
    CANONICAL_MAX_SEGMENTS_PER_SESSION,
    effective_segment_samples,
)
from ..train.evaluation_contract import snapshot_regular_file
from .manifest import read_manifest_bytes


RECORDED_SUBBAND_COVERAGE_SCHEMA_VERSION = 3
RECORDED_SUBBAND_COVERAGE_KIND = "recorded_strict_subband_coverage_audit"
CANONICAL_COVERAGE_SPLITS = ("train", "val", "test")
CANONICAL_COVERAGE_REPORT_DIRECTORY = "results/data_audit/recorded_subband_coverage"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _without_evidence_sha256(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "evidence_sha256"}


def seal_recorded_subband_coverage_report(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """보고서 전체 semantic payload에 자체 무결성 SHA를 붙인다."""

    sealed = dict(payload)
    sealed.pop("evidence_sha256", None)
    sealed["evidence_sha256"] = hashlib.sha256(
        _canonical_json_bytes(sealed)
    ).hexdigest()
    return sealed


def build_recorded_subband_coverage_contract(
    *,
    manifest_path: str | Path,
    manifest_content: bytes,
    data_cfg: dict[str, Any],
    model_hop: int,
    splits: Sequence[str] = CANONICAL_COVERAGE_SPLITS,
    max_segments_per_session: int = CANONICAL_MAX_SEGMENTS_PER_SESSION,
    edge_trim_seconds: float = CANONICAL_EDGE_TRIM_SECONDS,
) -> dict[str, Any]:
    """현재 입력에서 보고서가 반드시 가져야 하는 exact 계약을 만든다."""

    path = Path(manifest_path).expanduser().absolute()
    split_values = tuple(str(value) for value in splits)
    if not split_values or len(split_values) != len(set(split_values)):
        raise ValueError("coverage split은 중복 없는 비어 있지 않은 목록이어야 합니다")
    if any(value not in CANONICAL_COVERAGE_SPLITS for value in split_values):
        raise ValueError(f"지원하지 않는 coverage split: {split_values!r}")
    max_segments = int(max_segments_per_session)
    if max_segments < 1:
        raise ValueError("max_segments_per_session은 1 이상이어야 합니다")
    edge_trim = float(edge_trim_seconds)
    if not math.isfinite(edge_trim) or edge_trim < 0.0:
        raise ValueError("edge_trim_seconds는 유한한 0 이상이어야 합니다")
    hop = int(model_hop)
    if hop <= 0:
        raise ValueError("model_hop은 양수여야 합니다")
    timing = TrainingTimingContract.from_data_config(data_cfg)
    sample_rate = int(data_cfg.get("sample_rate", 0))
    if sample_rate <= 0 or int(timing.sample_rate) != sample_rate:
        raise ValueError("coverage sample_rate와 training timing contract가 다릅니다")
    lead = int(data_cfg.get("digital_reference_lead_samples", -1))
    if lead != int(timing.digital_reference_lead_samples):
        raise ValueError("coverage digital lead와 training timing contract가 다릅니다")
    warmup_seconds = float((data_cfg.get("closed_loop") or {}).get("warmup_seconds", 0.25))
    if not math.isfinite(warmup_seconds) or warmup_seconds < 0.0:
        raise ValueError("coverage warmup_seconds는 유한한 0 이상이어야 합니다")
    warmup_samples = int(round(warmup_seconds * sample_rate))
    segment_seconds = float(data_cfg.get("segment_seconds", 0.0))
    if not math.isfinite(segment_seconds) or segment_seconds <= 0.0:
        raise ValueError("coverage segment_seconds는 유한한 양수여야 합니다")
    segment_samples = effective_segment_samples(
        sample_rate=sample_rate,
        model_hop=hop,
        segment_seconds=segment_seconds,
    )

    contract = {
        "manifest": {
            "path": str(path),
            "size_bytes": len(manifest_content),
            "sha256": hashlib.sha256(manifest_content).hexdigest(),
        },
        "requested_splits": list(split_values),
        "sample_rate": sample_rate,
        "model_hop": hop,
        "segment_seconds": segment_seconds,
        "segment_samples": segment_samples,
        "digital_reference_lead_samples": lead,
        "training_timing_contract_sha256": timing.digest(),
        "warmup_samples": warmup_samples,
        "max_segments_per_session": max_segments,
        "edge_trim_seconds": edge_trim,
        "trusted_band_hz": [float(value) for value in STRICT_TRUSTED_BAND_HZ],
        "strict_subbands_hz": [
            [float(value) for value in band] for band in STRICT_TRUSTED_SUBBANDS_HZ
        ],
        "min_source_energy_density_ratio": float(
            MIN_SUBBAND_SOURCE_ENERGY_DENSITY_RATIO
        ),
        "min_groups_per_family": int(MIN_GROUPS_PER_FAMILY),
    }
    contract["coverage_contract_sha256"] = hashlib.sha256(
        _canonical_json_bytes(contract)
    ).hexdigest()
    return contract


def recorded_subband_coverage_report_path(
    directory: str | Path,
    contract: dict[str, Any],
) -> Path:
    """manifest/timing/threshold를 모두 포함한 generation-keyed report 경로."""

    digest = contract.get("coverage_contract_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("coverage contract SHA-256이 유효하지 않습니다")
    return Path(directory).expanduser().absolute() / f"{digest}.json"


def _load_json_without_duplicates(content: bytes, *, label: str) -> dict[str, Any]:
    def no_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label}에 중복 JSON key가 있습니다: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} JSON을 읽을 수 없습니다: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} 최상위는 JSON object여야 합니다")
    return value


def _require_exact_number(actual: object, expected: float | int, *, label: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ValueError(f"{label}은 숫자여야 합니다")
    value = float(actual)
    if not math.isfinite(value) or value != float(expected):
        raise ValueError(f"{label} 불일치: report={actual!r}, expected={expected!r}")


def _require_nonnegative_int(value: object, *, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}은 정수여야 합니다")
    minimum = 1 if positive else 0
    if value < minimum:
        raise ValueError(f"{label}은 {minimum} 이상이어야 합니다")
    return int(value)


def _entry_inventory(entries: list[dict[str, Any]]) -> dict[str, Any]:
    group_splits: dict[str, str] = {}
    group_families: dict[str, str] = {}
    sessions: set[str] = set()
    paths: set[str] = set()
    by_split_family: dict[tuple[str, str], set[str]] = {}
    split_sessions: dict[str, int] = {split: 0 for split in CANONICAL_COVERAGE_SPLITS}
    families: set[str] = set()
    for index, entry in enumerate(entries):
        required = ("path", "split", "session_id", "group_id", "source_family")
        if any(not str(entry.get(key, "")).strip() for key in required):
            raise ValueError(f"manifest entry #{index}의 coverage 필드가 불완전합니다")
        split = str(entry["split"])
        if split not in CANONICAL_COVERAGE_SPLITS:
            raise ValueError(f"manifest entry #{index} split이 잘못됐습니다: {split!r}")
        family = str(entry["source_family"])
        group = str(entry["group_id"])
        session = str(entry["session_id"])
        path = str(Path(str(entry["path"])).expanduser().absolute())
        if session in sessions or path in paths:
            raise ValueError("manifest session_id/path가 중복됐습니다")
        sessions.add(session)
        paths.add(path)
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"manifest group이 split을 넘나듭니다: {group!r}")
        if group in group_families and group_families[group] != family:
            raise ValueError(f"manifest group이 family를 넘나듭니다: {group!r}")
        group_splits[group] = split
        group_families[group] = family
        by_split_family.setdefault((split, family), set()).add(group)
        split_sessions[split] += 1
        families.add(family)
    return {
        "families": tuple(sorted(families)),
        "groups": by_split_family,
        "split_sessions": split_sessions,
    }


def validate_recorded_subband_coverage_report(
    report_path: str | Path,
    *,
    manifest_path: str | Path,
    data_cfg: dict[str, Any],
    model_hop: int,
    required_families: Sequence[str],
    configured_min_groups_per_family: int,
    splits: Sequence[str] = CANONICAL_COVERAGE_SPLITS,
    max_segments_per_session: int = CANONICAL_MAX_SEGMENTS_PER_SESSION,
    edge_trim_seconds: float = CANONICAL_EDGE_TRIM_SECONDS,
) -> dict[str, Any]:
    """precomputed report를 현재 manifest/config와 exact 재검증한다.

    반환은 검증된 요약이다. coverage 부족은 유효한 증거이므로 구조 오류와 구분해
    ``all_requested_splits_pass=False``로 반환한다.
    """

    report_snapshot = snapshot_regular_file(report_path)
    manifest_snapshot = snapshot_regular_file(manifest_path)
    report = _load_json_without_duplicates(
        report_snapshot.content, label="recorded subband coverage report"
    )
    expected_keys = {
        "schema_version",
        "kind",
        "manifest",
        "requested_splits",
        "sample_rate",
        "model_hop",
        "segment_seconds",
        "segment_samples",
        "digital_reference_lead_samples",
        "training_timing_contract_sha256",
        "warmup_samples",
        "max_segments_per_session",
        "edge_trim_seconds",
        "trusted_band_hz",
        "strict_subbands_hz",
        "min_source_energy_density_ratio",
        "min_groups_per_family",
        "coverage_contract_sha256",
        "all_requested_splits_pass",
        "splits",
        "evidence_sha256",
    }
    if set(report) != expected_keys:
        raise ValueError(
            "recorded subband coverage report 필드 집합이 다릅니다: "
            f"missing={sorted(expected_keys.difference(report))}, "
            f"extra={sorted(set(report).difference(expected_keys))}"
        )
    if report.get("schema_version") != RECORDED_SUBBAND_COVERAGE_SCHEMA_VERSION:
        raise ValueError("recorded subband coverage report schema_version이 다릅니다")
    if report.get("kind") != RECORDED_SUBBAND_COVERAGE_KIND:
        raise ValueError("recorded subband coverage report kind가 다릅니다")
    evidence_sha = report.get("evidence_sha256")
    expected_evidence_sha = hashlib.sha256(
        _canonical_json_bytes(_without_evidence_sha256(report))
    ).hexdigest()
    if evidence_sha != expected_evidence_sha:
        raise ValueError("recorded subband coverage report evidence SHA-256이 다릅니다")

    contract = build_recorded_subband_coverage_contract(
        manifest_path=manifest_snapshot.path,
        manifest_content=manifest_snapshot.content,
        data_cfg=data_cfg,
        model_hop=model_hop,
        splits=splits,
        max_segments_per_session=max_segments_per_session,
        edge_trim_seconds=edge_trim_seconds,
    )
    for key, expected in contract.items():
        actual = report.get(key)
        if isinstance(expected, float):
            _require_exact_number(actual, expected, label=f"coverage.{key}")
        elif actual != expected:
            raise ValueError(
                f"recorded subband coverage {key} 불일치: "
                f"report={actual!r}, expected={expected!r}"
            )
    configured_minimum = int(configured_min_groups_per_family)
    if configured_minimum != MIN_GROUPS_PER_FAMILY:
        raise ValueError(
            "readiness 계열별 group 하한이 strict G4 단일 출처와 다릅니다: "
            f"configured={configured_minimum}, expected={MIN_GROUPS_PER_FAMILY}"
        )

    entries = read_manifest_bytes(
        manifest_snapshot.content, manifest_path=manifest_snapshot.path
    )
    inventory = _entry_inventory(entries)
    families = tuple(sorted(str(value) for value in required_families))
    if not families or families != inventory["families"]:
        raise ValueError(
            "coverage required family와 manifest family가 다릅니다: "
            f"required={families!r}, manifest={inventory['families']!r}"
        )
    split_payloads = report.get("splits")
    if not isinstance(split_payloads, dict) or set(split_payloads) != set(splits):
        raise ValueError("coverage split payload가 requested_splits와 다릅니다")

    all_split_pass = True
    weak: list[dict[str, Any]] = []
    expected_pairs = {
        (family, tuple(float(value) for value in band))
        for family in families
        for band in STRICT_TRUSTED_SUBBANDS_HZ
    }
    for split in splits:
        payload = split_payloads[split]
        if not isinstance(payload, dict) or set(payload) != {
            "n_sessions",
            "n_segments",
            "group_power_pass",
            "rows",
        }:
            raise ValueError(f"coverage split={split} 필드 집합이 다릅니다")
        n_sessions = _require_nonnegative_int(
            payload["n_sessions"], label=f"coverage.{split}.n_sessions", positive=True
        )
        if n_sessions != inventory["split_sessions"][split]:
            raise ValueError(f"coverage split={split} session 수가 manifest와 다릅니다")
        n_segments = _require_nonnegative_int(
            payload["n_segments"], label=f"coverage.{split}.n_segments", positive=True
        )
        rows = payload["rows"]
        if not isinstance(rows, list):
            raise ValueError(f"coverage split={split}.rows는 list여야 합니다")
        seen_pairs: set[tuple[str, tuple[float, float]]] = set()
        family_segments: dict[str, int] = {}
        split_pass = True
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != {
                "source_family",
                "band_hz",
                "n_segments",
                "n_covered_segments",
                "n_covered_groups",
                "covered_group_ids",
                "density_mean",
                "density_median",
                "density_p10",
                "group_power_pass",
            }:
                raise ValueError(
                    f"coverage split={split} row #{row_index} 필드 집합이 다릅니다"
                )
            family = str(row["source_family"])
            try:
                band = tuple(float(value) for value in row["band_hz"])
            except (TypeError, ValueError) as exc:
                raise ValueError("coverage row band_hz를 읽을 수 없습니다") from exc
            pair = (family, band)
            if pair not in expected_pairs or pair in seen_pairs:
                raise ValueError(f"coverage split={split} family×band가 잘못됐습니다: {pair!r}")
            seen_pairs.add(pair)
            row_segments = _require_nonnegative_int(
                row["n_segments"],
                label=f"coverage.{split}.{family}.{band}.n_segments",
                positive=True,
            )
            previous_segments = family_segments.setdefault(family, row_segments)
            if previous_segments != row_segments:
                raise ValueError(f"coverage split={split} family별 segment 수가 band마다 다릅니다")
            covered_segments = _require_nonnegative_int(
                row["n_covered_segments"],
                label=f"coverage.{split}.{family}.{band}.n_covered_segments",
            )
            if covered_segments > row_segments:
                raise ValueError("coverage covered segment 수가 전체보다 큽니다")
            covered_groups = row["covered_group_ids"]
            if (
                not isinstance(covered_groups, list)
                or any(not isinstance(value, str) or not value for value in covered_groups)
                or covered_groups != sorted(set(covered_groups))
            ):
                raise ValueError("coverage covered_group_ids는 정렬된 고유 문자열이어야 합니다")
            declared_group_count = _require_nonnegative_int(
                row["n_covered_groups"],
                label=f"coverage.{split}.{family}.{band}.n_covered_groups",
            )
            if declared_group_count != len(covered_groups):
                raise ValueError("coverage covered group 수와 ID 목록이 다릅니다")
            allowed_groups = inventory["groups"].get((split, family), set())
            if not set(covered_groups).issubset(allowed_groups):
                raise ValueError("coverage report가 manifest에 없는 group을 주장합니다")
            if covered_segments < declared_group_count:
                raise ValueError("coverage covered segment 수가 covered group 수보다 작습니다")
            for metric in ("density_mean", "density_median", "density_p10"):
                value = row[metric]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ValueError(f"coverage {metric}은 유한한 0 이상이어야 합니다")
            expected_pass = declared_group_count >= MIN_GROUPS_PER_FAMILY
            if not isinstance(row["group_power_pass"], bool) or row["group_power_pass"] != expected_pass:
                raise ValueError("coverage row group_power_pass 집계가 위조됐습니다")
            split_pass = split_pass and expected_pass
            if not expected_pass:
                weak.append(
                    {
                        "split": split,
                        "source_family": family,
                        "band_hz": list(band),
                        "n_covered_groups": declared_group_count,
                    }
                )
        if seen_pairs != expected_pairs:
            raise ValueError(f"coverage split={split} family×band 행이 불완전합니다")
        if sum(family_segments.values()) != n_segments:
            raise ValueError(f"coverage split={split} 전체 segment 집계가 다릅니다")
        if not isinstance(payload["group_power_pass"], bool) or payload["group_power_pass"] != split_pass:
            raise ValueError(f"coverage split={split} group_power_pass 집계가 위조됐습니다")
        all_split_pass = all_split_pass and split_pass
    if (
        not isinstance(report["all_requested_splits_pass"], bool)
        or report["all_requested_splits_pass"] != all_split_pass
    ):
        raise ValueError("coverage all_requested_splits_pass 집계가 위조됐습니다")
    return {
        "report_path": str(report_snapshot.path),
        "report_sha256": report_snapshot.sha256,
        "evidence_sha256": evidence_sha,
        "manifest_sha256": manifest_snapshot.sha256,
        "all_requested_splits_pass": all_split_pass,
        "weak": weak,
    }


__all__ = [
    "CANONICAL_COVERAGE_SPLITS",
    "CANONICAL_COVERAGE_REPORT_DIRECTORY",
    "CANONICAL_EDGE_TRIM_SECONDS",
    "CANONICAL_MAX_SEGMENTS_PER_SESSION",
    "RECORDED_SUBBAND_COVERAGE_KIND",
    "RECORDED_SUBBAND_COVERAGE_SCHEMA_VERSION",
    "build_recorded_subband_coverage_contract",
    "seal_recorded_subband_coverage_report",
    "recorded_subband_coverage_report_path",
    "validate_recorded_subband_coverage_report",
]
