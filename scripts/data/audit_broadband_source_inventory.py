#!/usr/bin/env python3
"""광대역 recorded-v2 48-source acquisition inventory를 read-only로 감사."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_anc.data.broadband_source_inventory import (  # noqa: E402
    acquisition_manifest_contract,
    audit_acquisition_manifest,
    collect_local_source_summary,
    collect_rclone_metadata_summary,
    inspect_existing_pipeline_capability,
)


DEFAULT_CAMPAIGN = (
    "results/data_audit/"
    "broadband_recorded_v2_campaign_audit_20260828_20db.json"
)


def _json_no_duplicates(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"JSON duplicate key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 최상위가 object가 아닙니다: {path}")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _repository_file(path_text: str, *, must_exist: bool = True) -> Path:
    path = Path(path_text)
    candidate = path if path.is_absolute() else REPO_ROOT / path
    resolved = candidate.resolve(strict=must_exist)
    resolved.relative_to(REPO_ROOT)
    if must_exist and (resolved.is_symlink() or not resolved.is_file()):
        raise ValueError(f"regular repository file이 아닙니다: {resolved}")
    return resolved


def _file_reference(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_noreplace(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "실제 header/SHA/lineage/11.314kHz evidence가 있는 source만 48-slot에 셉니다. "
            "오디오 장치와 외부 파일 content는 열지 않습니다."
        )
    )
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    parser.add_argument(
        "--acquisition-manifest",
        help="없으면 현재 verified 후보를 0개로 감사합니다",
    )
    parser.add_argument(
        "--drive-remote-root",
        help=(
            "선택: rclone lsjson만 실행할 backup data root "
            "(예: gdrive:DeepANC/.../data)"
        ),
    )
    parser.add_argument("--output", help="지정하면 no-replace JSON으로 발행")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign_path = _repository_file(args.campaign)
    campaign = _json_no_duplicates(campaign_path)
    manifest = None
    manifest_reference = None
    if args.acquisition_manifest:
        manifest_path = _repository_file(args.acquisition_manifest)
        manifest = _json_no_duplicates(manifest_path)
        manifest_reference = _file_reference(manifest_path)

    acquisition = audit_acquisition_manifest(campaign, manifest)
    payload: dict[str, Any] = {
        "schema": "broadband_recorded_v2_source_inventory_report_v1",
        "role": "read_only_local_and_remote_metadata_audit_not_live_plan",
        "status": acquisition["status"],
        "campaign": _file_reference(campaign_path),
        "acquisition_manifest": manifest_reference,
        "acquisition_contract": acquisition_manifest_contract(),
        "acquisition_audit": acquisition,
        "local_inventory": collect_local_source_summary(REPO_ROOT),
        "drive_inventory": (
            collect_rclone_metadata_summary(args.drive_remote_root)
            if args.drive_remote_root
            else {
                "access": "not_requested",
                "external_content_read_or_copied": False,
                "cohorts": {},
            }
        ),
        "existing_pipeline": inspect_existing_pipeline_capability(REPO_ROOT),
        "safety": {
            "audio_devices_opened": False,
            "speaker_output_count": 0,
            "external_files_copied_or_modified": False,
            "remote_operation": "rclone_lsjson_only" if args.drive_remote_root else None,
        },
    }
    payload["evidence_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    if args.output:
        output = _repository_file(args.output, must_exist=False)
        _write_noreplace(output, payload)
        print(output.relative_to(REPO_ROOT))
    else:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
