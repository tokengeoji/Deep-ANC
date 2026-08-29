#!/usr/bin/env python3
"""기록된 ANC target ``d``의 strict 150–1600 Hz 부대역 coverage를 감사한다.

저장된 WAV만 읽으며 오디오 장치를 열지 않는다. 공식 recorded G4와
같은 segment/edge trim/warmup, target-energy density, family/group 규약을 쓴다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.config import load_train_config  # noqa: E402
from deep_anc.data.manifest import read_manifest_bytes  # noqa: E402
from deep_anc.data.recorded_subband_coverage import (  # noqa: E402
    CANONICAL_COVERAGE_SPLITS,
    CANONICAL_EDGE_TRIM_SECONDS,
    CANONICAL_MAX_SEGMENTS_PER_SESSION,
    RECORDED_SUBBAND_COVERAGE_KIND,
    RECORDED_SUBBAND_COVERAGE_SCHEMA_VERSION,
    build_recorded_subband_coverage_contract,
    seal_recorded_subband_coverage_report,
    recorded_subband_coverage_report_path,
    validate_recorded_subband_coverage_report,
)
from deep_anc.eval.metrics import band_power  # noqa: E402
from deep_anc.eval.recorded import (  # noqa: E402
    iter_recorded_segments,
    load_and_audit_recorded_manifest,
    resolve_warmup_samples,
)
from deep_anc.eval.trusted_subbands import (  # noqa: E402
    MIN_GROUPS_PER_FAMILY,
    MIN_SUBBAND_SOURCE_ENERGY_DENSITY_RATIO,
    STRICT_TRUSTED_BAND_HZ,
    STRICT_TRUSTED_SUBBANDS_HZ,
    source_energy_covered,
    strict_subband_includes_upper_edge,
)
from deep_anc.train.evaluation_contract import (  # noqa: E402
    snapshot_regular_file,
    write_json_exclusive,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/train_pretrain_tiny.yaml"
    )
    parser.add_argument(
        "--manifest", default="data/manifests/recorded_regrouped.jsonl"
    )
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"),
        default=CANONICAL_COVERAGE_SPLITS,
    )
    parser.add_argument(
        "--max-segments-per-session",
        type=int,
        default=CANONICAL_MAX_SEGMENTS_PER_SESSION,
    )
    parser.add_argument(
        "--edge-trim-seconds", type=float, default=CANONICAL_EDGE_TRIM_SECONDS
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--canonical-out-dir",
        default=None,
        help=(
            "manifest/timing/threshold 전체 계약 SHA 이름으로 immutable report를 생성하거나 "
            "같은 generation의 기존 report를 재검증"
        ),
    )
    parser.add_argument(
        "--verify-existing",
        default=None,
        help="WAV를 재분석하지 않고 기존 report를 현재 manifest/config에 exact 대조",
    )
    return parser


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if sum(bool(value) for value in (args.out, args.verify_existing, args.canonical_out_dir)) > 1:
        raise ValueError("--out/--verify-existing/--canonical-out-dir 중 하나만 지정하세요")
    if args.max_segments_per_session < 1:
        raise ValueError("max-segments-per-session은 1 이상이어야 합니다")
    if not math.isfinite(args.edge_trim_seconds) or args.edge_trim_seconds < 0.0:
        raise ValueError("edge-trim-seconds는 유한한 0 이상이어야 합니다")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    manifest_snapshot = snapshot_regular_file(manifest_path)
    # 한 번의 공식 loader 호출이 전체 manifest의 path/session/group split
    # 누수를 먼저 감사한다. train은 해당 loader의 평가 split이 아니므로
    # 검증된 같은 bytes를 read_manifest로 필터한다.
    load_and_audit_recorded_manifest(
        manifest_path,
        "val",
        manifest_bytes=manifest_snapshot.content,
    )
    all_entries = read_manifest_bytes(
        manifest_snapshot.content, manifest_path=manifest_snapshot.path
    )

    cfg = load_train_config(
        args.config,
        [
            "experiment_role=diagnostic_overfit",
            "init_eligible=false",
            "contract_run_dir=false",
            "run_until_step=500",
            "data.digital_primary_path_mode=measured",
        ],
    )
    data_cfg = cfg["data"]
    sample_rate = int(data_cfg["sample_rate"])
    model_hop = int(cfg["model"]["hop"])
    warmup_samples = resolve_warmup_samples(data_cfg, sample_rate)
    contract = build_recorded_subband_coverage_contract(
        manifest_path=manifest_snapshot.path,
        manifest_content=manifest_snapshot.content,
        data_cfg=data_cfg,
        model_hop=model_hop,
        splits=args.splits,
        max_segments_per_session=args.max_segments_per_session,
        edge_trim_seconds=args.edge_trim_seconds,
    )
    if int(contract["warmup_samples"]) != warmup_samples:
        raise RuntimeError("coverage contract와 evaluator warmup 계산이 다릅니다")
    generated_destination: Path | None = None
    verify_value = args.verify_existing
    if args.canonical_out_dir:
        report_directory = Path(args.canonical_out_dir)
        if not report_directory.is_absolute():
            report_directory = REPO_ROOT / report_directory
        generated_destination = recorded_subband_coverage_report_path(
            report_directory, contract
        )
    if verify_value:
        report_path = Path(verify_value)
        if not report_path.is_absolute():
            report_path = REPO_ROOT / report_path
        summary = validate_recorded_subband_coverage_report(
            report_path,
            manifest_path=manifest_snapshot.path,
            data_cfg=data_cfg,
            model_hop=model_hop,
            required_families=sorted(
                {str(entry["source_family"]) for entry in all_entries}
            ),
            configured_min_groups_per_family=MIN_GROUPS_PER_FAMILY,
            splits=args.splits,
            max_segments_per_session=args.max_segments_per_session,
            edge_trim_seconds=args.edge_trim_seconds,
        )
        print(
            json.dumps(
                summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        )
        return 0 if summary["all_requested_splits_pass"] else 1

    split_payloads: dict[str, object] = {}
    overall_pass = True
    for split in args.splits:
        entries = [dict(entry) for entry in all_entries if entry.get("split") == split]
        if not entries:
            raise ValueError(f"recorded manifest에 {split!r} split이 비었습니다")
        entries.sort(
            key=lambda entry: (
                str(entry["group_id"]),
                str(entry["session_id"]),
                str(entry["path"]),
            )
        )
        segments = iter_recorded_segments(
            entries,
            data_cfg,
            model_hop=model_hop,
            max_segments_per_session=args.max_segments_per_session,
            edge_trim_seconds=args.edge_trim_seconds,
        )
        rows: dict[tuple[str, tuple[float, float]], dict[str, object]] = defaultdict(
            lambda: {
                "n_segments": 0,
                "n_covered_segments": 0,
                "covered_groups": set(),
                "density": [],
            }
        )
        n_segments = 0
        for segment in segments:
            n_segments += 1
            target = np.asarray(segment.d[warmup_samples:], dtype=np.float64)
            trusted_power = band_power(
                target, sample_rate, STRICT_TRUSTED_BAND_HZ
            )
            for band in STRICT_TRUSTED_SUBBANDS_HZ:
                subband_power = band_power(
                    target,
                    sample_rate,
                    band,
                    include_upper=strict_subband_includes_upper_edge(band),
                )
                covered, density = source_energy_covered(
                    subband_power, trusted_power, band
                )
                row = rows[(segment.source_family, band)]
                row["n_segments"] = int(row["n_segments"]) + 1
                row["density"].append(float(density))
                if covered:
                    row["n_covered_segments"] = int(row["n_covered_segments"]) + 1
                    row["covered_groups"].add(segment.group_id)

        rendered: list[dict[str, object]] = []
        split_pass = True
        for (family, band), row in sorted(rows.items()):
            densities = list(row["density"])
            covered_groups = sorted(row["covered_groups"])
            group_pass = len(covered_groups) >= MIN_GROUPS_PER_FAMILY
            split_pass = split_pass and group_pass
            rendered.append(
                {
                    "source_family": family,
                    "band_hz": [float(band[0]), float(band[1])],
                    "n_segments": int(row["n_segments"]),
                    "n_covered_segments": int(row["n_covered_segments"]),
                    "n_covered_groups": len(covered_groups),
                    "covered_group_ids": covered_groups,
                    "density_mean": float(np.mean(densities)),
                    "density_median": _percentile(densities, 50.0),
                    "density_p10": _percentile(densities, 10.0),
                    "group_power_pass": bool(group_pass),
                }
            )
        overall_pass = overall_pass and split_pass
        split_payloads[split] = {
            "n_sessions": len(entries),
            "n_segments": n_segments,
            "group_power_pass": bool(split_pass),
            "rows": rendered,
        }

    payload = seal_recorded_subband_coverage_report({
        "schema_version": RECORDED_SUBBAND_COVERAGE_SCHEMA_VERSION,
        "kind": RECORDED_SUBBAND_COVERAGE_KIND,
        **contract,
        "all_requested_splits_pass": bool(overall_pass),
        "splits": split_payloads,
    })
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    destination: Path | None = generated_destination
    if args.out:
        destination = Path(args.out)
        if not destination.is_absolute():
            destination = REPO_ROOT / destination
    if destination is not None:
        if generated_destination is not None and (
            destination.exists() or destination.is_symlink()
        ):
            existing = snapshot_regular_file(destination)
            fresh_bytes = (encoded + "\n").encode("utf-8")
            if existing.content != fresh_bytes:
                raise ValueError(
                    "기존 canonical coverage report가 raw WAV fresh 재계산 bytes와 "
                    f"다릅니다: {destination}"
                )
            print(f"[coverage audit] fresh 재계산 exact 기존 report: {destination}")
        else:
            write_json_exclusive(destination, payload)
            print(f"[coverage audit] 저장: {destination}")
    print(encoded)
    return 0 if overall_pass else 1


def _cli() -> int:
    try:
        return main()
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[coverage audit 오류] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
