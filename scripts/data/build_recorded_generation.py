#!/usr/bin/env python3
"""기존 82세션과 별도 추가 19세션을 immutable recorded generation으로 봉인한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.holdout_contract import (  # noqa: E402
    HoldoutContractError,
    read_regular_file_snapshot,
    reject_symlink_components,
)
from deep_anc.data.recorded_generation import (  # noqa: E402
    GENERATION_ROOT,
    RecordedGenerationError,
    build_combined_manifest_bytes,
    build_recorded_generation_payload,
    validate_recorded_generation,
)


def _publish_no_replace(path: Path, data: bytes, *, repo_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(path.parent, root=repo_root)
    if path.exists():
        existing = read_regular_file_snapshot(
            path, root=repo_root, label=f"existing {path.name}", capture_bytes=True
        )
        if existing.data != data:
            raise RecordedGenerationError(
                f"기존 artifact bytes가 다릅니다. 자동 overwrite하지 않습니다: {path}"
            )
        return
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--expected-holdout-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    directory = REPO_ROOT / GENERATION_ROOT / args.generation_id
    combined_path = directory / "recorded.jsonl"
    generation_path = directory / "generation.json"
    try:
        combined = build_combined_manifest_bytes(
            repo_root=REPO_ROOT,
            generation_id=args.generation_id,
            expected_holdout_sha256=args.expected_holdout_sha256,
            require_source_files=True,
        )
        _publish_no_replace(combined_path, combined, repo_root=REPO_ROOT)
        payload = build_recorded_generation_payload(
            repo_root=REPO_ROOT,
            generation_id=args.generation_id,
            expected_holdout_sha256=args.expected_holdout_sha256,
            require_source_files=True,
        )
        raw = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        _publish_no_replace(generation_path, raw, repo_root=REPO_ROOT)
        digest = hashlib.sha256(raw).hexdigest()
        summary = validate_recorded_generation(
            generation_path,
            repo_root=REPO_ROOT,
            expected_sha256=digest,
            require_source_files=True,
        )
    except (OSError, HoldoutContractError, RecordedGenerationError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print(
        f"recorded generation: {generation_path}\n"
        f"sha256: {digest}\n"
        f"sessions: {summary['parent_session_count']} + "
        f"{summary['addition_session_count']} = {summary['recorded_session_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
