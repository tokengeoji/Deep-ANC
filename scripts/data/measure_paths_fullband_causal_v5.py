#!/usr/bin/env python3
"""v5 near-white causal P/S signal-only 계획을 생성하고 exact 조건수를 검사한다."""

from __future__ import annotations

import argparse
import json
import os
import sys
import hashlib
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.dsp.fullband_causal_v5 import (  # noqa: E402
    CANONICAL_BLOCKER,
    LIVE_AUTHORITY,
    build_plan_v5,
    exact_condition_audit_v5,
)


def _lexical_repository_target(target: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(REPO_ROOT)))
    lexical = Path(os.path.abspath(os.fspath(target.expanduser())))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise ValueError("signal plan은 저장소 내부에만 저장합니다") from error
    cursor = root
    if cursor.is_symlink():
        raise ValueError("repository root symlink를 거부합니다")
    for part in relative.parts[:-1]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"output parent symlink를 거부합니다: {cursor}")
    if lexical.is_symlink():
        raise ValueError(f"output target symlink를 거부합니다: {lexical}")
    return lexical


def _write_json_no_replace(payload: dict, target: Path) -> Path:
    lexical = _lexical_repository_target(target)
    lexical.parent.mkdir(parents=True, exist_ok=True)
    lexical = _lexical_repository_target(lexical)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(os.fspath(lexical), flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    directory_fd = os.open(
        os.fspath(lexical.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return lexical


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _publish_raw_no_replace(
    *,
    plan: dict,
    submitted_pcm: np.ndarray,
    captured_pcm: np.ndarray,
    callback_frames: np.ndarray,
) -> Path:
    """이미 얻은 배열만 exact plan raw path에 immutable NPZ로 발행한다."""

    target = REPO_ROOT / plan["publisher_contract"]["raw_session_relative_path"]
    lexical = _lexical_repository_target(target)
    lexical.parent.mkdir(parents=True, exist_ok=True)
    lexical = _lexical_repository_target(lexical)
    submitted = np.asarray(submitted_pcm)
    captured = np.asarray(captured_pcm)
    callbacks = np.asarray(callback_frames)
    if submitted.dtype != np.int16 or list(submitted.shape) != plan["actual_submitted_shape"]:
        raise ValueError("submitted_pcm dtype/shape이 plan과 다릅니다")
    if _array_sha256(submitted) != plan["actual_submitted_pcm_sha256"]:
        raise ValueError("submitted_pcm SHA가 plan과 다릅니다")
    if captured.ndim != 2 or captured.shape[0] != submitted.shape[0] or captured.shape[1] != 2:
        raise ValueError("captured_pcm은 submitted와 같은 frame 수의 [frame,2]여야 합니다")
    if callbacks.ndim != 1 or not np.issubdtype(callbacks.dtype, np.integer):
        raise ValueError("callback_frames는 1-D integer 배열이어야 합니다")
    metadata = {
        "schema": "fullband_causal_raw_capture_v5",
        "signal_plan_payload_sha256": plan["canonical_payload_sha256"],
        "submitted_pcm_sha256": _array_sha256(submitted),
        "captured_pcm_sha256": _array_sha256(captured),
        "callback_frames_sha256": _array_sha256(callbacks),
        "live_authority_at_plan_time": None,
    }
    metadata_bytes = np.frombuffer(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    ).copy()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(os.fspath(lexical), flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            np.savez(
                handle,
                submitted_pcm=submitted,
                captured_pcm=captured,
                callback_frames=callbacks,
                metadata_json_utf8=metadata_bytes,
            )
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    directory_fd = os.open(
        os.fspath(lexical.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return lexical


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-live", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--raw-session-relative-path",
        default="results/fullband_causal_v5/raw_capture.npz",
    )
    args = parser.parse_args(argv)

    # live 분기는 signal 생성이나 어떤 장치 backend 접근보다 먼저 닫힌다.
    if args.execute_live:
        assert LIVE_AUTHORITY is None
        print("[중단] v5 live authority=None; 실제 출력을 허용하지 않습니다", file=sys.stderr)
        print(f"[차단] {CANONICAL_BLOCKER}", file=sys.stderr)
        return 2

    try:
        plan, submitted = build_plan_v5(
            raw_session_relative_path=args.raw_session_relative_path
        )
        condition = exact_condition_audit_v5(plan, submitted)
        if args.output is not None:
            _write_json_no_replace(
                {
                    "schema": "fullband_causal_signal_plan_envelope_v5",
                    "signal_plan": plan,
                    "support_1024_condition_receipt": condition,
                },
                args.output,
            )
    except (FileExistsError, OSError, ValueError) as error:
        print(f"[실패] v5 signal-only 계획 발행 거부: {error}", file=sys.stderr)
        return 2
    print(
        "[PASS] v5 signal-only | "
        f"{plan['duration_seconds']:.3f}s | peak PCM {plan['actual_submitted_peak_pcm']} | "
        f"support1024 condition {condition['joint_fit_condition_number']:.6f}"
    )
    print(f"[SHA] plan {plan['canonical_payload_sha256']}")
    print(f"[SHA] PCM  {plan['actual_submitted_pcm_sha256']}")
    print("[잠금] 실제 출력 0회; live authority=None; canonical training=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
