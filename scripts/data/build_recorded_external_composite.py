#!/usr/bin/env python3
"""Untouched ESC-50 5초 raw를 canonical 15초 추가녹음 composite로 만든다."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data import public_lineage  # noqa: E402
from deep_anc.data.holdout_contract import (  # noqa: E402
    read_regular_file_snapshot,
    reject_symlink_components,
)
from deep_anc.data.recorded_generation import (  # noqa: E402
    CANONICAL_EXTERNAL_ESC_MACHINE_FILES,
    EXTERNAL_REPEAT_COUNT,
    EXTERNAL_TRANSFORM,
    SOURCE_PLAN_ROOT,
    _canonical_external_composite_bytes,
    _canonical_source_lineage,
    validate_generation_id,
)


def _repo_relative(value: str, *, prefix: str) -> tuple[Path, str]:
    candidate = Path(value).expanduser()
    candidate = candidate if candidate.is_absolute() else REPO_ROOT / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("경로는 저장소 내부여야 합니다") from exc
    if not relative.startswith(prefix.rstrip("/") + "/"):
        raise ValueError(f"경로는 {prefix}/ 아래여야 합니다: {relative}")
    return candidate, relative


def _publish_no_replace(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(path.parent, root=REPO_ROOT)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"기존 composite bytes가 달라 overwrite하지 않습니다: {path}")
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--raw-member", required=True)
    parser.add_argument("--family", choices=("machine",), required=True)
    parser.add_argument("--out-name", required=True, help="경로 구분자 없는 .wav 파일명")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        generation = validate_generation_id(args.generation_id)
        if Path(args.out_name).name != args.out_name or not args.out_name.endswith(".wav"):
            raise ValueError("--out-name은 경로 구분자 없는 .wav 파일명이어야 합니다")
        raw_path, raw_relative = _repo_relative(
            args.raw_member,
            prefix="data/raw/noise/esc50/ESC-50-master/audio",
        )
        lineage = _canonical_source_lineage(REPO_ROOT)
        raw_key = public_lineage.esc50_lineage_keys(
            raw_path.name, lineage["esc50_metadata"]
        )[0]
        if (
            CANONICAL_EXTERNAL_ESC_MACHINE_FILES.get(raw_path.name)
            != lineage["esc50_authority"].get(raw_path.name)
        ):
            raise ValueError(
                "raw member가 canonical external ESC-50 machine 4개 inventory 밖입니다"
            )
        if ("esc50", raw_key) in lineage["active_identity_keys"]:
            raise ValueError(f"raw member가 parent82 active identity와 겹칩니다: {raw_key}")
        raw_snapshot = read_regular_file_snapshot(
            raw_path,
            root=REPO_ROOT,
            label="external ESC-50 raw member",
            capture_bytes=True,
        )
        assert raw_snapshot.data is not None
        raw_bytes = raw_snapshot.data
        composite = _canonical_external_composite_bytes(
            raw_path, raw_bytes=raw_bytes
        )
        output_relative = (
            f"{SOURCE_PLAN_ROOT}/{generation}_sources/{args.out_name}"
        )
        output = REPO_ROOT / output_relative
        _publish_no_replace(output, composite)
        token = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12]
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print(
        "source_kind=external_exact_composite\n"
        f"path={output_relative}\n"
        "seconds=15.0\nstart_seconds=0.0\n"
        f"source_family={args.family}\n"
        f"group_id={args.family}-esc50-source-{token}\n"
        f"lineage_key={args.family}-external-lineage-{token}\n"
        f"source_file_sha256={hashlib.sha256(composite).hexdigest()}\n"
        f"raw_member_path={raw_relative}\n"
        f"raw_member_sha256={hashlib.sha256(raw_bytes).hexdigest()}\n"
        f"raw_member_lineage_key={raw_key}\n"
        f"authority_metadata_sha256={lineage['esc50_metadata_sha256']}\n"
        f"inventory_path={lineage['esc50_metadata_path']}\n"
        f"inventory_sha256={lineage['esc50_metadata_sha256']}\n"
        f"transform={EXTERNAL_TRANSFORM}\n"
        f"transform_repeat_count={EXTERNAL_REPEAT_COUNT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
