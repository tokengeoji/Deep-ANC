#!/usr/bin/env python3
"""Jetson 로컬 source/manifests를 population-v3 mapping 관점에서 감사한다."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_anc.data.broadband_population_availability_v3 import (  # noqa: E402
    AvailabilityInputV3,
    audit_population_v3_availability,
)


_INPUT_KINDS = {"public_jsonl", "recorded_jsonl", "source_pool_csv"}
_FAMILIES = {"speech", "music", "environment", "machine"}


def _parse_binding(value: str, *, allowed: set[str], label: str) -> tuple[str, str]:
    left, separator, right = str(value).partition("=")
    if not separator or left not in allowed or not right.strip():
        choices = ", ".join(sorted(allowed))
        raise argparse.ArgumentTypeError(f"{label} 형식은 {{{choices}}}=PATH 입니다")
    return left, right


def _manifest_binding(value: str) -> AvailabilityInputV3:
    kind, path = _parse_binding(value, allowed=_INPUT_KINDS, label="--input")
    return AvailabilityInputV3(kind=kind, path=path)


def _tree_binding(value: str) -> AvailabilityInputV3:
    family, path = _parse_binding(
        value,
        allowed=_FAMILIES,
        label="--scan-unreferenced",
    )
    return AvailabilityInputV3(
        kind="unreferenced_audio_tree",
        path=path,
        source_family=family,
    )


def _repository_output(path_text: str) -> Path:
    path = Path(path_text)
    candidate = path if path.is_absolute() else REPO_ROOT / path
    resolved = Path(os.path.abspath(candidate))
    resolved.relative_to(REPO_ROOT)
    current = REPO_ROOT
    for part in resolved.relative_to(REPO_ROOT).parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError("output 경로에 symlink가 있습니다")
    return resolved


def _write_noreplace(path: Path, payload: dict[str, object]) -> None:
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "legacy manifest와 실제 local audio bytes의 v3 mapping 가능성만 감사합니다. "
            "causal P/density/population manifest authority는 발급하지 않습니다."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        type=_manifest_binding,
        default=[],
        metavar="KIND=PATH",
        help="public_jsonl, recorded_jsonl, source_pool_csv 중 하나 (반복 가능)",
    )
    parser.add_argument(
        "--scan-unreferenced",
        action="append",
        type=_tree_binding,
        default=[],
        metavar="FAMILY=PATH",
        help="manifest가 참조하지 않은 local audio를 별도 mapping 후보로 조사",
    )
    parser.add_argument(
        "--causal-p-authority",
        help="선택: CausalPrimaryOperatorV3 JSON 후보. 유효해도 authority 없이 structural-only",
    )
    parser.add_argument("--output", help="no-replace JSON 보고서 경로")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = tuple(args.input) + tuple(args.scan_unreferenced)
    if not inputs:
        raise SystemExit("--input 또는 --scan-unreferenced가 최소 하나 필요합니다")
    report = audit_population_v3_availability(
        repository_root=REPO_ROOT,
        inputs=inputs,
        causal_p_authority_path=args.causal_p_authority,
    )
    payload = report.model_dump(mode="json")
    if args.output:
        output = _repository_output(args.output)
        _write_noreplace(output, payload)
        output_label = output.relative_to(REPO_ROOT).as_posix()
    else:
        output_label = None
    summary = {
        "schema_version": report.schema_version,
        "status": report.status,
        "authority": report.authority,
        "output": output_label,
        "evidence_sha256": report.evidence_sha256,
        "causal_p_status": report.causal_primary.status,
        "summary": report.summary.model_dump(mode="json"),
        "input_status": [
            {
                "kind": item.kind,
                "path": item.path,
                "status": item.status,
                "entries_seen": item.entries_seen,
                "entries_emitted": item.entries_emitted,
            }
            for item in report.inputs
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
