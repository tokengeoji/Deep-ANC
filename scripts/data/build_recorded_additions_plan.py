#!/usr/bin/env python3
"""Stage-1 600--1600 Hz coverage 보충용 canonical 19행 plan을 생성·검증한다.

오디오 장치를 열지 않는다. exact environment/music source-pool 9행,
immutable DEMAND environment 1행, external DNS speech receipt 5행,
ESC-50 repeat composite 4행만 허용한다.

이 stage-1 plan은 최종 광대역-v2
데이터가 아니다. 1.6/2/4/8 kHz 또는 8 kHz octave 상단 11.314 kHz coverage를
증명하거나 ``broadband_point_control_150_11314_v2`` receipt에 사용할 수 없다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data import public_lineage  # noqa: E402
from deep_anc.data.holdout_contract import (  # noqa: E402
    HoldoutContractError,
    read_regular_file_snapshot,
    reject_symlink_components,
)
from deep_anc.data.recorded_generation import (  # noqa: E402
    CANONICAL_EXTERNAL_ESC_MACHINE_FILES,
    CANONICAL_EXTERNAL_ESC_OUTPUT_NAMES,
    CANONICAL_EXTERNAL_ESC_SPLITS,
    CANONICAL_SOURCE_POOL_ADDITIONS,
    EXTERNAL_REPEAT_COUNT,
    EXTERNAL_TRANSFORM,
    SOURCE_KIND_EXTERNAL,
    SOURCE_KIND_EXTERNAL_DEMAND_ENVIRONMENT,
    SOURCE_KIND_EXTERNAL_DNS_SPEECH,
    SOURCE_KIND_POOL,
    SOURCE_PLAN_FIELDS,
    SOURCE_PLAN_ROOT,
    SOURCE_SELECTION_STRICT_PRIMARY_PATH,
    SOURCE_SELECTION_STRICT_PRIMARY_SHA256,
    RecordedGenerationError,
    _canonical_external_composite_bytes,
    _canonical_source_selection_evidence,
    _canonical_source_lineage,
    _read_source_plan,
    validate_generation_id,
)
from deep_anc.data.recording_gain_linearity import (  # noqa: E402
    RecordingGainLinearityError,
    validate_gain_linearity_receipt,
)
from deep_anc.data.recording_source_gain import (  # noqa: E402
    PHYSICAL_SELECTOR_MAX_AMPLITUDE_MILLIONTHS,
    RecordingSourceGainError,
    audit_source_plan_at_measured_cap,
)
from deep_anc.data.recorded_demand_selection import (  # noqa: E402
    DEMAND_LINEAGE_KEY,
    DEMAND_PUBLIC_GROUP_ID,
    DEMAND_RECORDED_SPLIT,
    DEMAND_SELECTION_ORIGIN_SOURCE,
    DEMAND_SELECTION_RECEIPT,
    DEMAND_SELECTION_SOURCE,
    DEMAND_TRANSFORM,
    DEMAND_WINDOW_SECONDS,
    DEMAND_WINDOW_START_SECONDS,
    validate_demand_selection_receipt,
)
from deep_anc.data.recorded_dns_selection import (  # noqa: E402
    DNS_REPEAT_COUNT,
    DNS_SELECTION_RECEIPT,
    DNS_TRANSFORM,
    validate_dns_selection_receipt,
)

CANONICAL_GENERATION_ID = "stage1-coverage-v3-gain012"


def _snapshot(relative: str, *, label: str):
    return read_regular_file_snapshot(
        REPO_ROOT / relative,
        root=REPO_ROOT,
        label=label,
        capture_bytes=True,
    )


def _empty_row() -> dict[str, str]:
    return {field: "" for field in SOURCE_PLAN_FIELDS}


def build_rows(
    generation_id: str,
    *,
    dns_selection_receipt_sha256: str,
    demand_selection_receipt_sha256: str,
) -> list[dict[str, str]]:
    generation_id = validate_generation_id(generation_id)
    if generation_id != CANONICAL_GENERATION_ID:
        raise ValueError(
            f"현행 exact source plan generation-id는 {CANONICAL_GENERATION_ID!r}입니다"
        )
    authority = _canonical_source_lineage(REPO_ROOT)
    # source-pool speech/Libri 후보는 rejected evidence로만 보존하고, 외부 DNS
    # receipt가 없거나 SHA/lineage/raw/composite가 다르면 여기서 BLOCKED된다.
    _canonical_source_selection_evidence(REPO_ROOT, authority)
    dns_selection = validate_dns_selection_receipt(
        repo_root=REPO_ROOT,
        receipt_path=DNS_SELECTION_RECEIPT,
        expected_receipt_sha256=dns_selection_receipt_sha256,
        require_source_files=True,
    )
    demand_selection = validate_demand_selection_receipt(
        repo_root=REPO_ROOT,
        receipt_path=DEMAND_SELECTION_RECEIPT,
        expected_receipt_sha256=demand_selection_receipt_sha256,
        require_source_files=True,
    )
    rows: list[dict[str, str]] = []

    for path, (family, start, split) in CANONICAL_SOURCE_POOL_ADDITIONS.items():
        source_row = authority["rows"].get(path)
        component = authority["component_by_path"].get(path)
        if not isinstance(source_row, dict) or not isinstance(component, str):
            raise ValueError(f"canonical source-pool authority row 없음: {path}")
        snapshot = _snapshot(path, label=f"canonical source-pool addition {path}")
        row = _empty_row()
        row.update(
            {
                "source_kind": SOURCE_KIND_POOL,
                "path": path,
                "seconds": "15.0",
                "start_seconds": str(start),
                "source_family": family,
                "group_id": str(source_row["group_id"]),
                "lineage_key": component,
                "split": split,
                "source_file_sha256": snapshot.sha256,
                "transform": "identity",
                "transform_repeat_count": "1",
            }
        )
        rows.append(row)

    demand_selected = demand_selection["selected"]
    demand_source = demand_selected["bundle_source"]
    demand_origin = demand_selected["origin_bundle_source"]
    demand_row = _empty_row()
    demand_row.update(
        {
            "source_kind": SOURCE_KIND_EXTERNAL_DEMAND_ENVIRONMENT,
            "path": DEMAND_SELECTION_SOURCE,
            "seconds": str(DEMAND_WINDOW_SECONDS),
            "start_seconds": str(DEMAND_WINDOW_START_SECONDS),
            "source_family": "environment",
            "group_id": DEMAND_PUBLIC_GROUP_ID,
            "lineage_key": DEMAND_LINEAGE_KEY,
            "split": DEMAND_RECORDED_SPLIT,
            "source_file_sha256": str(demand_source["sha256"]),
            "raw_member_path": DEMAND_SELECTION_ORIGIN_SOURCE,
            "raw_member_sha256": str(demand_origin["sha256"]),
            "raw_member_lineage_key": DEMAND_PUBLIC_GROUP_ID,
            "authority_metadata_sha256": demand_selection[
                "public_manifest_sha256"
            ],
            "inventory_path": demand_selection["receipt_path"],
            "inventory_sha256": demand_selection["receipt_sha256"],
            "transform": DEMAND_TRANSFORM,
            "transform_repeat_count": "1",
        }
    )
    rows.append(demand_row)

    for item in dns_selection["selected"]:
        public_group = str(item["public_group_id"])
        digest12 = hashlib.sha256(public_group.encode("utf-8")).hexdigest()[:12]
        raw_ref = item["raw_output"]
        composite_ref = item["composite_output"]
        row = _empty_row()
        row.update(
            {
                "source_kind": SOURCE_KIND_EXTERNAL_DNS_SPEECH,
                "path": str(composite_ref["path"]),
                "seconds": "15.0",
                "start_seconds": "0.0",
                "source_family": "speech",
                "group_id": public_group,
                "lineage_key": f"speech-dns-lineage-{digest12}",
                "split": str(item["recorded_split"]),
                "source_file_sha256": str(composite_ref["sha256"]),
                "raw_member_path": str(raw_ref["path"]),
                "raw_member_sha256": str(raw_ref["sha256"]),
                "raw_member_lineage_key": public_group,
                "authority_metadata_sha256": dns_selection[
                    "public_manifest_sha256"
                ],
                "inventory_path": dns_selection["receipt_path"],
                "inventory_sha256": dns_selection["receipt_sha256"],
                "transform": DNS_TRANSFORM,
                "transform_repeat_count": str(DNS_REPEAT_COUNT),
            }
        )
        rows.append(row)

    for raw_name, (_source_id, _category) in CANONICAL_EXTERNAL_ESC_MACHINE_FILES.items():
        raw_path = f"data/raw/noise/esc50/ESC-50-master/audio/{raw_name}"
        raw = _snapshot(raw_path, label=f"canonical external ESC-50 {raw_name}")
        assert raw.data is not None
        expected_output = _canonical_external_composite_bytes(
            REPO_ROOT / raw_path, raw_bytes=raw.data
        )
        output_path = (
            f"{SOURCE_PLAN_ROOT}/{generation_id}_sources/"
            f"{CANONICAL_EXTERNAL_ESC_OUTPUT_NAMES[raw_name]}"
        )
        output = _snapshot(output_path, label=f"canonical ESC-50 composite {raw_name}")
        if output.data != expected_output:
            raise ValueError(f"ESC-50 composite bytes가 exact transform과 다릅니다: {output_path}")
        raw_lineage = public_lineage.esc50_lineage_keys(
            raw_name, authority["esc50_metadata"]
        )[0]
        digest12 = hashlib.sha256(raw_lineage.encode("utf-8")).hexdigest()[:12]
        row = _empty_row()
        row.update(
            {
                "source_kind": SOURCE_KIND_EXTERNAL,
                "path": output_path,
                "seconds": "15.0",
                "start_seconds": "0.0",
                "source_family": "machine",
                "group_id": f"machine-esc50-source-{digest12}",
                "lineage_key": f"machine-external-lineage-{digest12}",
                "split": CANONICAL_EXTERNAL_ESC_SPLITS[raw_name],
                "source_file_sha256": output.sha256,
                "raw_member_path": raw_path,
                "raw_member_sha256": raw.sha256,
                "raw_member_lineage_key": raw_lineage,
                "authority_metadata_sha256": authority["esc50_metadata_sha256"],
                "inventory_path": authority["esc50_metadata_path"],
                "inventory_sha256": authority["esc50_metadata_sha256"],
                "transform": EXTERNAL_TRANSFORM,
                "transform_repeat_count": str(EXTERNAL_REPEAT_COUNT),
            }
        )
        rows.append(row)
    return rows


def _render(rows: list[dict[str, str]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=SOURCE_PLAN_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _measured_cap_from_receipt(
    *, receipt_path: str, expected_receipt_sha256: str
) -> int:
    summary = validate_gain_linearity_receipt(
        repo_root=REPO_ROOT,
        receipt_path=receipt_path,
        expected_sha256=expected_receipt_sha256,
    )
    if summary.get("passed") is not True:
        raise ValueError("gain-linearity receipt가 PASS가 아닙니다")
    analysis = summary["payload"].get("analysis")
    cap = analysis.get("supported_max_amplitude_millionths") if isinstance(analysis, dict) else None
    if (
        isinstance(cap, bool)
        or not isinstance(cap, int)
        or not 1 <= cap <= PHYSICAL_SELECTOR_MAX_AMPLITUDE_MILLIONTHS
    ):
        raise ValueError("gain-linearity receipt measured amplitude cap이 유효하지 않습니다")
    return cap


def _require_all_rows_feasible(
    *, relative_plan: str, plan_sha256: str, amplitude_millionths: int
) -> dict[str, object]:
    audit = audit_source_plan_at_measured_cap(
        repo_root=REPO_ROOT,
        source_plan=relative_plan,
        expected_source_plan_sha256=plan_sha256,
        strict_primary=SOURCE_SELECTION_STRICT_PRIMARY_PATH,
        expected_strict_primary_sha256=SOURCE_SELECTION_STRICT_PRIMARY_SHA256,
        amplitude_millionths=amplitude_millionths,
    )
    if audit["row_count"] != 19 or audit["feasible_row_count"] != 19:
        concise = [
            f"row={item['source_row_number']} path={item['path']} "
            f"reasons={','.join(item['reasons'])}"
            for item in audit["blockers"]
        ]
        raise ValueError(
            "physical cap에서 source plan 19/19 feasible이 아닙니다: "
            + "; ".join(concise)
        )
    return audit


def _validate_bytes(
    raw: bytes, *, generation_id: str, amplitude_millionths: int
) -> dict[str, object]:
    staging_parent = REPO_ROOT / "results"
    reject_symlink_components(staging_parent, root=REPO_ROOT)
    with tempfile.TemporaryDirectory(
        prefix="recorded-generation-plan-", dir=staging_parent
    ) as directory:
        candidate = Path(directory) / f"{generation_id}.csv"
        candidate.write_bytes(raw)
        _read_source_plan(
            repo_root=REPO_ROOT,
            relative=candidate.relative_to(REPO_ROOT).as_posix(),
            require_source_files=True,
        )
        return _require_all_rows_feasible(
            relative_plan=candidate.relative_to(REPO_ROOT).as_posix(),
            plan_sha256=hashlib.sha256(raw).hexdigest(),
            amplitude_millionths=amplitude_millionths,
        )


def _publish_no_replace(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(path.parent, root=REPO_ROOT)
    if path.exists() or path.is_symlink():
        snapshot = read_regular_file_snapshot(
            path,
            root=REPO_ROOT,
            label="existing canonical additions source plan",
            capture_bytes=True,
        )
        if snapshot.data != raw:
            raise ValueError(f"기존 source plan bytes가 달라 overwrite하지 않습니다: {path}")
        return
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-id", default=CANONICAL_GENERATION_ID)
    parser.add_argument(
        "--dns-selection-receipt-sha256",
        required=True,
        help="Elice selector stdout에서 별도 전달한 selection receipt 파일 SHA-256",
    )
    parser.add_argument(
        "--demand-selection-receipt-sha256",
        required=True,
        help="Elice DEMAND selector stdout에서 별도 전달한 receipt 파일 SHA-256",
    )
    parser.add_argument(
        "--gain-linearity-receipt",
        required=True,
        help="Jetson bounded physical gain/linearity PASS receipt의 저장소 상대경로",
    )
    parser.add_argument(
        "--gain-linearity-receipt-sha256",
        required=True,
        help="gain-linearity receipt의 외부 전달 SHA-256",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        generation_id = validate_generation_id(args.generation_id)
        dns_receipt_sha = str(args.dns_selection_receipt_sha256).lower()
        demand_receipt_sha = str(args.demand_selection_receipt_sha256).lower()
        for option, value in (
            ("--dns-selection-receipt-sha256", dns_receipt_sha),
            ("--demand-selection-receipt-sha256", demand_receipt_sha),
            (
                "--gain-linearity-receipt-sha256",
                str(args.gain_linearity_receipt_sha256).lower(),
            ),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(
                    f"{option}에는 외부 전달 64자리 SHA-256이 필요합니다"
                )
        gain_receipt_sha = str(args.gain_linearity_receipt_sha256).lower()
        measured_cap = _measured_cap_from_receipt(
            receipt_path=args.gain_linearity_receipt,
            expected_receipt_sha256=gain_receipt_sha,
        )
        destination = REPO_ROOT / SOURCE_PLAN_ROOT / f"{generation_id}.csv"
        if args.verify_existing:
            snapshot, rows, lineage_sha, selection_evidence = _read_source_plan(
                repo_root=REPO_ROOT,
                relative=destination.relative_to(REPO_ROOT).as_posix(),
                require_source_files=True,
            )
            raw_sha = snapshot.sha256
            selection_sha = str(selection_evidence["evidence_sha256"])
            receipt_summary = validate_dns_selection_receipt(
                repo_root=REPO_ROOT,
                receipt_path=DNS_SELECTION_RECEIPT,
                expected_receipt_sha256=dns_receipt_sha,
                require_source_files=True,
            )
            if any(
                row.get("inventory_sha256") != receipt_summary["receipt_sha256"]
                for row in rows
                if row.get("source_kind") == SOURCE_KIND_EXTERNAL_DNS_SPEECH
            ):
                raise ValueError(
                    "기존 source plan DNS 행이 외부 selection receipt SHA와 다릅니다"
                )
            demand_summary = validate_demand_selection_receipt(
                repo_root=REPO_ROOT,
                receipt_path=DEMAND_SELECTION_RECEIPT,
                expected_receipt_sha256=demand_receipt_sha,
                require_source_files=True,
            )
            demand_rows = [
                row
                for row in rows
                if row.get("source_kind")
                == SOURCE_KIND_EXTERNAL_DEMAND_ENVIRONMENT
            ]
            if (
                len(demand_rows) != 1
                or demand_rows[0].get("inventory_sha256")
                != demand_summary["receipt_sha256"]
            ):
                raise ValueError(
                    "기존 source plan DEMAND 행이 외부 selection receipt SHA와 다릅니다"
                )
            cap_audit = _require_all_rows_feasible(
                relative_plan=destination.relative_to(REPO_ROOT).as_posix(),
                plan_sha256=raw_sha,
                amplitude_millionths=measured_cap,
            )
        else:
            rows = build_rows(
                generation_id,
                dns_selection_receipt_sha256=dns_receipt_sha,
                demand_selection_receipt_sha256=demand_receipt_sha,
            )
            raw = _render(rows)
            cap_audit = _validate_bytes(
                raw,
                generation_id=generation_id,
                amplitude_millionths=measured_cap,
            )
            raw_sha = hashlib.sha256(raw).hexdigest()
            authority = _canonical_source_lineage(REPO_ROOT)
            lineage_sha = authority["evidence_sha256"]
            selection_evidence = _canonical_source_selection_evidence(
                REPO_ROOT, authority
            )
            # check-only candidate는 이미 _validate_bytes에서 같은 selection 계약을
            # 통과했다. 실제 evidence SHA는 current authority에서 별도로 재유도한다.
            selection_sha = str(selection_evidence["evidence_sha256"])
            if args.write:
                _publish_no_replace(destination, raw)
                _read_source_plan(
                    repo_root=REPO_ROOT,
                    relative=destination.relative_to(REPO_ROOT).as_posix(),
                    require_source_files=True,
                )
    except (
        OSError,
        RuntimeError,
        ValueError,
        HoldoutContractError,
        RecordedGenerationError,
        RecordingGainLinearityError,
        RecordingSourceGainError,
    ) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    mode = "verified" if args.verify_existing else ("written" if args.write else "checked")
    print(
        f"source plan {mode}: {destination}\n"
        f"rows: {len(rows)}\nsha256: {raw_sha}\nlineage evidence: {lineage_sha}"
        f"\nselection evidence: {selection_sha}"
        f"\nphysical cap amplitude millionths: {measured_cap}"
        f"\n19/19 cap audit: {cap_audit['evidence_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
