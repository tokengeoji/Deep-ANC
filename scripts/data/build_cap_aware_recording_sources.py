#!/usr/bin/env python3
"""물리 gain cap에서 source 파생물과 exact 무출력 audit를 발행한다.

이 스크립트는 오디오 장치를 import/open하지 않는다. ``--write``가 없으면 source와
strict-P를 읽고 메모리에서만 파생·검증한다. PASS 파생물은 coverage-training 전용이며
natural/unseen 평가 증거가 아니다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _relative(value: str, *, label: str) -> str:
    path = Path(value)
    text = path.as_posix()
    if not text or path.is_absolute() or ".." in path.parts or text != value:
        raise ValueError(
            f"{label}는 canonical 저장소 상대경로여야 합니다"
        )
    return text


def _held_file(relative: str, expected_sha256: str, *, label: str) -> bytes:
    relative = _relative(relative, label=label)
    path = REPO_ROOT / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} regular file이 없습니다")
    raw = path.read_bytes()
    if _sha256(raw) != expected_sha256:
        raise ValueError(f"{label} SHA가 외부 anchor와 다릅니다")
    return raw


def _source_rows(raw: bytes) -> list[dict[str, str]]:
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"source plan CSV decode 실패: {exc}") from exc
    required = {
        "path",
        "start_seconds",
        "source_family",
        "group_id",
        "lineage_key",
        "split",
        "source_file_sha256",
    }
    if not required.issubset(reader.fieldnames or ()) or not rows:
        raise ValueError("source plan 필수 열/행이 없습니다")
    return rows


def _apply_candidate_overrides(
    rows: list[dict[str, str]],
    *,
    raw: bytes,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from deep_anc.data import public_lineage
    from deep_anc.data.recorded_generation import _canonical_source_lineage

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"candidate override JSON decode 실패: {exc}") from exc
    required = {
        "schema",
        "role",
        "canonical_live_eligible",
        "source_plan",
        "strict_primary",
        "amplitude_millionths",
        "lineage_authority_evidence_sha256",
        "overrides",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("candidate override 최상위 필드 집합이 다릅니다")
    if (
        payload["schema"]
        != "recording_source_conditioning_candidate_overrides/v1"
        or payload["role"]
        != "local_exact_cap_candidate_set_not_canonical_generation_or_natural_evaluation_evidence"
        or payload["canonical_live_eligible"] is not False
        or payload["source_plan"]
        != {"path": args.source_plan, "sha256": args.source_plan_sha256}
        or payload["strict_primary"]
        != {"path": args.strict_primary, "sha256": args.strict_primary_sha256}
        or payload["amplitude_millionths"] != args.amplitude_millionths
    ):
        raise ValueError("candidate override base authority/cap/role이 다릅니다")
    overrides = payload["overrides"]
    if not isinstance(overrides, list) or not overrides:
        raise ValueError("candidate override 행이 없습니다")
    authority = _canonical_source_lineage(REPO_ROOT.resolve())
    if (
        payload["lineage_authority_evidence_sha256"]
        != authority["evidence_sha256"]
    ):
        raise ValueError("candidate override lineage authority SHA가 stale입니다")
    override_fields = {
        "source_row_number",
        "expected_origin_path",
        "expected_origin_sha256",
        "path",
        "source_file_sha256",
        "start_seconds",
        "source_family",
        "group_id",
        "lineage_key",
        "split",
        "authority_kind",
    }
    seen_rows: set[int] = set()
    replacement_lineages: set[str] = set()
    used_replacement_authority_tokens: dict[str, str] = {}
    for item in overrides:
        if not isinstance(item, dict) or set(item) != override_fields:
            raise ValueError("candidate override row 필드 집합이 다릅니다")
        row_number = item["source_row_number"]
        if (
            isinstance(row_number, bool)
            or not isinstance(row_number, int)
            or not 2 <= row_number <= len(rows) + 1
            or row_number in seen_rows
        ):
            raise ValueError("candidate override source_row_number가 유효하지 않습니다")
        seen_rows.add(row_number)
        row = rows[row_number - 2]
        if (
            row.get("path") != item["expected_origin_path"]
            or row.get("source_file_sha256") != item["expected_origin_sha256"]
        ):
            raise ValueError(f"candidate override row {row_number} origin이 stale입니다")
        path = _relative(item["path"], label=f"override row {row_number} path")
        source_raw = _held_file(
            path,
            item["source_file_sha256"],
            label=f"override row {row_number} source",
        )
        if not source_raw:
            raise ValueError("candidate override source가 비었습니다")
        lineage = str(item["lineage_key"])
        if lineage in replacement_lineages:
            raise ValueError("candidate override replacement lineage가 중복됩니다")
        replacement_lineages.add(lineage)
        kind = item["authority_kind"]
        if kind == "source_pool_component":
            authority_row = authority["rows"].get(path)
            component = authority["component_by_path"].get(path)
            authority_tokens = set(
                authority["authority_tokens_by_path"].get(path, set())
            )
            if (
                not isinstance(authority_row, dict)
                or component != lineage
                or authority_row.get("source_family") != item["source_family"]
                or authority_row.get("group_id") != item["group_id"]
                or component in authority["active_components"]
                or not authority_tokens
            ):
                raise ValueError(
                    f"candidate override row {row_number} source-pool lineage가 free가 아닙니다"
                )
        elif kind == "librispeech_component":
            identities = public_lineage.librispeech_lineage_keys(
                Path(path).name, authority["librispeech_chapters"]
            )
            components = {
                authority["librispeech_component_by_identity"].get(identity)
                for identity in identities
            }
            if (
                components != {lineage}
                or lineage in authority["active_librispeech_components"]
                or item["source_family"] != "speech"
            ):
                raise ValueError(
                    f"candidate override row {row_number} LibriSpeech lineage가 free가 아닙니다"
                )
            authority_tokens = {f"librispeech_component:{lineage}"}
        else:
            raise ValueError("candidate override authority_kind가 지원되지 않습니다")
        overlap = sorted(authority_tokens & set(used_replacement_authority_tokens))
        if overlap:
            first = used_replacement_authority_tokens[overlap[0]]
            raise ValueError(
                "candidate override authority token이 서로 겹칩됩니다: "
                f"token={overlap[0]}, paths={first},{path}"
            )
        for token in authority_tokens:
            used_replacement_authority_tokens[token] = path
        for field in (
            "path",
            "source_file_sha256",
            "source_family",
            "group_id",
            "lineage_key",
            "split",
        ):
            row[field] = str(item[field])
        row["start_seconds"] = str(float(item["start_seconds"]))
        # Candidate source가 바뀌었는데 stale CSV의 raw-member 계보를
        # 영수증에 남기면 새 WAV와 다른 원본을 봉인하게 된다.
        # 따라서 override는 exact 새 파일을 자신의 raw member로 명시한다.
        row["raw_member_path"] = path
        row["raw_member_sha256"] = str(item["source_file_sha256"])
        row["raw_member_lineage_key"] = lineage

    expected_counts = {
        ("speech", "train"): 2,
        ("speech", "val"): 1,
        ("speech", "test"): 2,
        ("music", "train"): 1,
        ("music", "val"): 2,
        ("music", "test"): 2,
        ("environment", "train"): 1,
        ("environment", "val"): 1,
        ("environment", "test"): 3,
        ("machine", "train"): 1,
        ("machine", "val"): 1,
        ("machine", "test"): 2,
    }
    observed_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["source_family"], row["split"])
        observed_counts[key] = observed_counts.get(key, 0) + 1
    if len(rows) != 19 or observed_counts != expected_counts:
        raise ValueError("candidate override 후 family×split 19행 계약이 다릅니다")
    final_lineages = [row["lineage_key"] for row in rows]
    if len(set(final_lineages)) != len(final_lineages):
        raise ValueError("candidate override 후 lineage_key가 중복됩니다")
    return payload


def _strict_primary(raw: bytes) -> np.ndarray:
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            fir = np.asarray(archive["fir"], dtype=np.float64).reshape(-1)
            sample_rate = int(np.asarray(archive["sample_rate"]).item())
            band = np.asarray(archive["consistency_band_hz"], dtype=np.float64)
            xrun = int(np.asarray(archive["xrun_count"]).item())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError(f"strict primary NPZ decode 실패: {exc}") from exc
    if (
        fir.size < 2
        or not bool(np.isfinite(fir).all())
        or sample_rate != 48_000
        or band.tolist() != [150.0, 1600.0]
        or xrun != 0
    ):
        raise ValueError(
            "strict primary FIR/sample-rate/band/xrun 계약 위반"
        )
    return fir


def build_campaign(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, bytes]]:
    from deep_anc.data.recording_source_conditioning import (
        CONDITIONING_CAMPAIGN_SCHEMA,
        CONDITIONING_ROLE,
        condition_source_at_cap,
    )

    source_plan_raw = _held_file(
        args.source_plan, args.source_plan_sha256, label="source plan"
    )
    primary_raw = _held_file(
        args.strict_primary, args.strict_primary_sha256, label="strict primary"
    )
    rows = _source_rows(source_plan_raw)
    override_ref = None
    if args.candidate_overrides is not None:
        if args.candidate_overrides_sha256 is None:
            raise ValueError("--candidate-overrides에는 SHA-256이 필요합니다")
        override_raw = _held_file(
            args.candidate_overrides,
            args.candidate_overrides_sha256,
            label="candidate overrides",
        )
        _apply_candidate_overrides(rows, raw=override_raw, args=args)
        override_ref = {
            "path": args.candidate_overrides,
            "size": len(override_raw),
            "sha256": args.candidate_overrides_sha256,
        }
    fir = _strict_primary(primary_raw)
    output_root = _relative(args.out_root, label="output root")
    outputs: dict[str, bytes] = {}
    receipts: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        source_path = _relative(row["path"], label=f"row {row_number} source")
        source_raw = _held_file(
            source_path,
            row["source_file_sha256"],
            label=f"row {row_number} source",
        )
        lineage = {
            "source_row_number": row_number,
            "source_family": row["source_family"],
            "group_id": row["group_id"],
            "lineage_key": row["lineage_key"],
            "split": row["split"],
            "raw_member_path": row.get("raw_member_path", ""),
            "raw_member_sha256": row.get("raw_member_sha256", ""),
            "raw_member_lineage_key": row.get("raw_member_lineage_key", ""),
        }
        result = condition_source_at_cap(
            source_bytes=source_raw,
            source_path=source_path,
            start_seconds=float(row["start_seconds"]),
            strict_primary_fir=fir,
            strict_primary_path=args.strict_primary,
            strict_primary_sha256=args.strict_primary_sha256,
            amplitude_millionths=args.amplitude_millionths,
            lineage=lineage,
        )
        receipt = json.loads(json.dumps(result.receipt))
        stem = (
            f"row-{row_number:02d}-{row['source_family']}-"
            f"{_sha256(source_raw)[:12]}"
        )
        wav_path = None
        if result.wav_bytes is not None:
            wav_path = f"{output_root}/{stem}.wav"
            outputs[wav_path] = result.wav_bytes
        receipt_path = f"{output_root}/{stem}.receipt.json"
        # path는 seal 밖의 transport 위치가 아니라 campaign mapping으로 보존한다.
        outputs[receipt_path] = _pretty_json_bytes(receipt)
        receipts.append(
            {
                "source_row_number": row_number,
                "source_path": source_path,
                "source_family": row["source_family"],
                "group_id": row["group_id"],
                "lineage_key": row["lineage_key"],
                "split": row["split"],
                "receipt_path": receipt_path,
                "status": receipt["status"],
                "receipt_sha256": receipt["receipt_sha256"],
                "selected_recipe": receipt["selected_recipe"],
                "derived_wav_path": wav_path,
                "derived_wav": receipt["derived_wav"],
                "blocker_reasons": receipt["blocker_reasons"],
                "exact_cap_audit": receipt["exact_cap_audit"],
            }
        )
    feasible = sum(item["status"] == "PASS_COVERAGE_TRAINING_ONLY" for item in receipts)
    campaign: dict[str, Any] = {
        "schema": CONDITIONING_CAMPAIGN_SCHEMA,
        "role": CONDITIONING_ROLE,
        "source_plan": {
            "path": args.source_plan,
            "size": len(source_plan_raw),
            "sha256": args.source_plan_sha256,
        },
        "strict_primary": {
            "path": args.strict_primary,
            "size": len(primary_raw),
            "sha256": args.strict_primary_sha256,
        },
        "amplitude_millionths": args.amplitude_millionths,
        "candidate_overrides": override_ref,
        "canonical_live_eligible": False,
        "row_count": len(receipts),
        "feasible_row_count": feasible,
        "all_rows_feasible": feasible == len(receipts),
        "rows": receipts,
        "thresholds_relaxed": False,
        "natural_unprocessed_evaluation_eligible": False,
    }
    campaign["campaign_sha256"] = _sha256(_canonical_json_bytes(campaign))
    return campaign, outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--source-plan-sha256", required=True)
    parser.add_argument("--strict-primary", required=True)
    parser.add_argument("--strict-primary-sha256", required=True)
    parser.add_argument("--amplitude-millionths", type=int, required=True)
    parser.add_argument("--candidate-overrides")
    parser.add_argument("--candidate-overrides-sha256")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--campaign-out", required=True)
    parser.add_argument("--write", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        campaign, outputs = build_campaign(args)
        if args.write:
            from deep_anc.data.recording_source_conditioning import publish_no_replace

            for relative, raw in sorted(outputs.items()):
                publish_no_replace(REPO_ROOT / relative, raw)
            publish_no_replace(
                REPO_ROOT / _relative(args.campaign_out, label="campaign out"),
                _pretty_json_bytes(campaign),
            )
    except (OSError, ValueError) as exc:
        print(f"[BLOCKED] cap-aware source build 실패: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": (
                    "PASS" if campaign["all_rows_feasible"] else "BLOCKED"
                ),
                "mode": "write" if args.write else "check-only",
                "row_count": campaign["row_count"],
                "feasible_row_count": campaign["feasible_row_count"],
                "campaign_sha256": campaign["campaign_sha256"],
                "campaign_out": args.campaign_out,
                "blocked_rows": [
                    {
                        "source_row_number": row["source_row_number"],
                        "blocker_reasons": row["blocker_reasons"],
                    }
                    for row in campaign["rows"]
                    if row["status"] != "PASS_COVERAGE_TRAINING_ONLY"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if campaign["all_rows_feasible"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
