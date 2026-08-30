#!/usr/bin/env python3
"""현재 82세션과 source pool에서 광대역 recorded-v2 최소 캠페인을 계산한다.

오디오 장치를 import/open하지 않고 stdout에만 JSON을 출력한다. 기존 source를 신규 source로
승격하거나 missing raw를 추정하지 않는다. 원본 파일이 로컬에 없으면 그대로 ``missing``으로,
가공된 48 kHz pool WAV만 있으면 native-rate provenance 미확인으로 기록한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.broadband_recording_campaign import (  # noqa: E402
    build_missing_source_campaign,
)
from deep_anc.dsp.control_band_contract import ControlBandContract  # noqa: E402


HISTORICAL_DIAGNOSTIC = (
    REPO_ROOT / "results/data_audit/broadband_prerequisite_diagnostic_20260828.json"
)
CURRENT_DIAGNOSTIC = (
    REPO_ROOT
    / "results/data_audit/broadband_prerequisite_diagnostic_20260828_20db.json"
)
MANIFEST = REPO_ROOT / "data/manifests/recorded_regrouped.jsonl"
POOL_CSVS = (
    REPO_ROOT / "data/source_pool/sources.csv",
    REPO_ROOT / "data/source_pool_v2/sources.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    relative = resolved.relative_to(REPO_ROOT).as_posix()
    return {
        "path": relative,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _validated_current_diagnostic(contract: ControlBandContract) -> dict[str, Any]:
    payload = json.loads(CURRENT_DIAGNOSTIC.read_text(encoding="utf-8"))
    if payload.get("schema") != "broadband_prerequisite_audit_v1" or payload.get(
        "role"
    ) != "diagnostic_only_not_readiness_receipt":
        raise ValueError("현행 broadband prerequisite diagnostic 역할/schema가 다릅니다")
    if payload.get("control_band_contract_sha256") != contract.digest():
        raise ValueError("현행 diagnostic control-band SHA가 현재 계약과 다릅니다")
    coverage = payload.get("recorded_coverage")
    if not isinstance(coverage, dict) or coverage.get(
        "role"
    ) != "diagnostic_only_not_campaign_receipt":
        raise ValueError("현행 recorded coverage가 diagnostic 역할이 아닙니다")
    embedded_sha = coverage.get("evidence_sha256")
    unsealed = {key: value for key, value in coverage.items() if key != "evidence_sha256"}
    recomputed = hashlib.sha256(
        json.dumps(
            unsealed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if embedded_sha != recomputed:
        raise ValueError("현행 recorded coverage evidence SHA가 재계산과 다릅니다")
    if coverage.get("control_band_contract_sha256") != contract.digest():
        raise ValueError("현행 recorded coverage contract SHA가 현재 계약과 다릅니다")
    manifest_ref = coverage.get("manifest")
    qa_ref = coverage.get("qa")
    expected_manifest = _reference(MANIFEST)
    expected_qa = _reference(REPO_ROOT / "data/manifests/recorded_qa.json")
    for label, actual, expected in (
        ("manifest", manifest_ref, expected_manifest),
        ("recorded QA", qa_ref, expected_qa),
    ):
        if not isinstance(actual, dict) or any(
            actual.get(key) != expected[key] for key in ("size_bytes", "sha256")
        ):
            raise ValueError(f"현행 coverage {label} bytes가 현재 파일과 다릅니다")
    if len(coverage.get("sessions", [])) != 82:
        raise ValueError("현행 recorded coverage가 82세션이 아닙니다")
    return payload


def _manifest_rows() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 82 or len({str(row["session_id"]) for row in rows}) != len(rows):
        raise ValueError("현재 recorded manifest가 canonical 82개 session이 아닙니다")
    return rows


def _pool_rows() -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    by_path: dict[str, dict[str, str]] = {}
    summaries: list[dict[str, Any]] = []
    for path in POOL_CSVS:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rates = Counter(str(row.get("sample_rate_hz", "")) for row in rows)
        families = Counter(str(row.get("source_family", "")) for row in rows)
        crest_values = sorted(float(row.get("crest_factor_db", "nan")) for row in rows)
        if not crest_values or any(not math.isfinite(value) for value in crest_values):
            raise ValueError(f"source pool crest_factor_db가 유효하지 않습니다: {path}")
        header_counts: Counter[tuple[int, int, str]] = Counter()
        header_mismatches: list[str] = []
        for row in rows:
            key = str(row["path"])
            if key in by_path:
                raise ValueError(f"source pool path가 중복됩니다: {key}")
            by_path[key] = row
            wav = (REPO_ROOT / key).resolve(strict=True)
            wav.relative_to(REPO_ROOT)
            info = sf.info(str(wav))
            header_counts[
                (int(info.samplerate), int(info.channels), str(info.subtype))
            ] += 1
            if int(info.samplerate) != int(row["sample_rate_hz"]):
                header_mismatches.append(key)
        summaries.append(
            {
                "file": _reference(path),
                "row_count": len(rows),
                "processed_wav_sample_rate_counts": dict(sorted(rates.items())),
                "processed_wav_header_counts": [
                    {
                        "sample_rate": rate,
                        "channels": channels,
                        "subtype": subtype,
                        "count": count,
                    }
                    for (rate, channels, subtype), count in sorted(
                        header_counts.items()
                    )
                ],
                "csv_header_sample_rate_mismatches": header_mismatches,
                "processed_wav_crest_factor_db": {
                    "minimum": crest_values[0],
                    "median": statistics.median(crest_values),
                    "maximum": crest_values[-1],
                },
                "family_counts": dict(sorted(families.items())),
                "native_rate_provenance_in_csv": False,
            }
        )
    return by_path, summaries


def _session_audio_headers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    files = ("source.wav", "source_aligned.wav", "mics.wav")
    counters: dict[str, Counter[tuple[int, int, str]]] = {
        name: Counter() for name in files
    }
    frames: dict[str, list[int]] = {name: [] for name in files}
    for row in rows:
        session = (MANIFEST.parent / str(row["path"])).resolve(strict=True)
        session.relative_to(REPO_ROOT)
        for name in files:
            path = (session / name).resolve(strict=True)
            path.relative_to(session)
            info = sf.info(str(path))
            counters[name][(int(info.samplerate), int(info.channels), str(info.subtype))] += 1
            frames[name].append(int(info.frames))
    return {
        name: {
            "count": sum(counters[name].values()),
            "header_counts": [
                {
                    "sample_rate": rate,
                    "channels": channels,
                    "subtype": subtype,
                    "count": count,
                }
                for (rate, channels, subtype), count in sorted(counters[name].items())
            ],
            "minimum_frames": min(frames[name]),
            "maximum_frames": max(frames[name]),
        }
        for name in files
    }


def _raw_basename_index() -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    raw_root = REPO_ROOT / "data/raw"
    for path in raw_root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            index[path.name].append(path)
    return index


def _active_original_audit(
    rows: list[dict[str, Any]], pool_by_path: dict[str, dict[str, str]]
) -> dict[str, Any]:
    index = _raw_basename_index()
    placements: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        session_dir = (MANIFEST.parent / str(row["path"])).resolve(strict=True)
        session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
        source_path = str(session.get("program", {}).get("file", ""))
        source_row = pool_by_path.get(source_path)
        if source_row is None:
            raise ValueError(f"session source가 두 sources.csv에 없습니다: {source_path}")
        clips = json.loads(str(source_row["clips"]))
        if not isinstance(clips, list) or not clips:
            raise ValueError(f"source pool clips가 비었습니다: {source_path}")
        family = str(row["source_family"])
        placements[family].extend(str(clip) for clip in clips)

    result: dict[str, Any] = {}
    for family in ("speech", "music", "environment", "machine"):
        clips = sorted(set(placements[family]))
        rate_counts: Counter[str] = Counter()
        unique_match = 0
        missing = 0
        ambiguous = 0
        for clip in clips:
            matches = index.get(clip, [])
            if not matches:
                missing += 1
                continue
            if len(matches) != 1:
                ambiguous += 1
                continue
            unique_match += 1
            try:
                rate_counts[str(int(sf.info(str(matches[0])).samplerate))] += 1
            except RuntimeError:
                rate_counts["unreadable"] += 1
        result[family] = {
            "active_original_placements": len(placements[family]),
            "active_unique_original_basenames": len(clips),
            "unique_local_basename_matches": unique_match,
            "missing_local_originals": missing,
            "ambiguous_local_basenames": ambiguous,
            "native_header_rate_counts_for_unique_local_matches": dict(
                sorted(rate_counts.items())
            ),
            "match_role": "basename_match_diagnostic_only",
        }
    return result


def _lineage_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for family in ("speech", "music", "environment", "machine"):
        for split in ("train", "val", "test"):
            selected = [
                row
                for row in rows
                if row["source_family"] == family and row["split"] == split
            ]
            counts[family][split] = {
                "sessions": len(selected),
                "independent_groups": len({str(row["group_id"]) for row in selected}),
            }
    for row in rows:
        group_splits[str(row["group_id"])].add(str(row["split"]))
    return {
        "by_family_split": dict(counts),
        "cross_split_group_ids": sorted(
            group for group, splits in group_splits.items() if len(splits) > 1
        ),
    }


def build_report() -> dict[str, Any]:
    historical_diagnostic = json.loads(
        HISTORICAL_DIAGNOSTIC.read_text(encoding="utf-8")
    )
    contract = ControlBandContract.broadband_point_control()
    diagnostic = _validated_current_diagnostic(contract)
    current_coverage = diagnostic["recorded_coverage"]
    rows = _manifest_rows()
    pool_by_path, pool_summaries = _pool_rows()
    campaign = build_missing_source_campaign(diagnostic)
    report: dict[str, Any] = {
        "schema": "broadband_recorded_v2_local_campaign_audit_v2",
        "role": "read_only_diagnostic_and_missing_source_specification",
        "status": "BLOCKED",
        "evidence": {
            "historical_broadband_diagnostic": {
                "file": _reference(HISTORICAL_DIAGNOSTIC),
                "stored_control_band_contract_sha256": historical_diagnostic.get(
                    "control_band_contract_sha256"
                ),
                "current_control_band_contract_sha256": contract.digest(),
                "canonical_reuse_allowed": bool(
                    historical_diagnostic.get("control_band_contract_sha256")
                    == contract.digest()
                ),
            },
            "current_no_replace_recorded_scan": {
                "file": _reference(CURRENT_DIAGNOSTIC),
                "control_band_contract_sha256": contract.digest(),
                "recorded_coverage_evidence_sha256": current_coverage["evidence_sha256"],
                "session_count": len(current_coverage["sessions"]),
            },
            "recorded_manifest": _reference(MANIFEST),
            "source_pool_csvs": pool_summaries,
            "session_audio_headers": _session_audio_headers(rows),
            "active_original_sources": _active_original_audit(rows, pool_by_path),
            "lineage": _lineage_counts(rows),
        },
        "campaign": campaign,
        "blockers": [
            "2.828--11.314kHz의 모든 split×family cell에 qualifying group이 0개",
            "신규 native fs>=22628Hz lossless source 48개와 변환 receipt가 아직 없음",
            "canonical fullband causal P가 없어 predicted-ERR PSD 후보를 확정할 수 없음",
            (
                "recorded-v2 raw-first/absolute DAC-q/7-band 재검산 계약은 구현됐지만 "
                "verified source plan과 fullband causal plant가 없어 dry-run/live authority가 BLOCKED"
            ),
        ],
    }
    unsealed = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    report["evidence_sha256"] = hashlib.sha256(unsealed).hexdigest()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="선택한 저장소 내부 신규 JSON 경로에 exclusive-create로 저장",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_report()
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        try:
            output.relative_to(REPO_ROOT)
        except ValueError as exc:
            print("[FAIL] output은 저장소 내부여야 합니다", file=sys.stderr)
            return 1
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("x", encoding="utf-8") as handle:
                handle.write(text)
        except FileExistsError:
            print(f"[FAIL] 기존 output을 덮어쓰지 않습니다: {output}", file=sys.stderr)
            return 1
        print(f"[saved] {output}", file=sys.stderr)
    print(text, end="")
    return 2 if report["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
