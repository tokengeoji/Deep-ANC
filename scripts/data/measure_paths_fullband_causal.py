#!/usr/bin/env python3
"""Fullband causal P/S 식별 signal plan과 immutable raw-first 계약.

현재 live authority는 고정되지 않았다. 따라서 ``--execute-live``는 sounddevice를
import하거나 ALSA 장치를 조사하기 전에 항상 실패한다. ``--dry-run``만 plan을 만들며
실제 live 경로는 exact plan/file/PCM SHA 검토 후 별도 변경으로 열어야 한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.config import load_yaml  # noqa: E402
from deep_anc.dsp.fullband_causal_probe import (  # noqa: E402
    BLOCK_SIZE,
    SAMPLE_RATE,
    build_signal_plan,
)
from deep_anc.dsp.measurement_level import (  # noqa: E402
    atomic_publish_noreplace,
    validate_measurement_hardware_contract,
)
from scripts.data import measure_paths_broadband_interleaved as broadband_measure  # noqa: E402
from scripts.data import measure_paths_interleaved as interleaved_measure  # noqa: E402


FULLBAND_CAUSAL_LIVE_AUTHORITY: dict[str, str] | None = None
RAW_CAPTURE_SCHEMA = "fullband_causal_ps_raw_first_v1"
CALLBACK_TIME_INFO_FIELDS = broadband_measure.CALLBACK_TIME_INFO_FIELDS


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_bound_signal_plan(hardware_path: str | Path) -> tuple[dict[str, Any], np.ndarray]:
    hardware_file = Path(hardware_path).expanduser().resolve()
    hardware = load_yaml(hardware_file)
    audio, channels = validate_measurement_hardware_contract(hardware)
    if (int(audio["sample_rate"]), int(audio["block_size"]), str(audio["latency"])) != (
        SAMPLE_RATE,
        BLOCK_SIZE,
        "low",
    ):
        raise ValueError("fullband causal plan은 48kHz/256/low만 허용합니다")
    expected_channels = {
        "error_mic": 0,
        "reference_mic": 1,
        "noise_out": 0,
        "cancel_out": 1,
    }
    if channels != expected_channels:
        raise ValueError(f"fullband causal channel map이 다릅니다: {channels!r}")
    plan, pcm = build_signal_plan()
    plan = {
        **plan,
        "hardware": {
            "path": hardware_file.relative_to(REPO_ROOT.resolve()).as_posix(),
            "sha256": _sha256_file(hardware_file),
            "channels": channels,
        },
        "safety": {
            "authority_fixed": False,
            "immutable_raw_first": True,
            "requires_audio_device_occupancy_check": True,
            "requires_fresh_level_evidence_binding": True,
            "requires_callback_dac_adc_timestamps": True,
            "requires_xrun_clip_zero": True,
            "requires_operator_confirmations": [
                "speaker_output",
                "user_present",
                "volume_minimum",
                "routing_and_geometry",
                "same_amplifier_setting",
            ],
            "analysis_before_raw_publish_forbidden": True,
        },
    }
    return plan, pcm


def write_plan_noreplace(path: str | Path, plan: dict[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    try:
        target.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("signal plan은 저장소 안에만 저장할 수 있습니다") from exc
    if target.exists():
        raise FileExistsError(f"기존 signal plan을 덮어쓰지 않습니다: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.partial"
    try:
        with temporary.open("xb") as handle:
            handle.write(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True).encode())
            handle.flush()
            os.fsync(handle.fileno())
        atomic_publish_noreplace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def validate_live_authority_before_audio_import(
    *, plan_path: str | Path | None, expected_plan: dict[str, Any], pcm: np.ndarray
) -> dict[str, str]:
    """sounddevice/장치 점유 검사보다 먼저 exact authority를 검증한다."""

    if FULLBAND_CAUSAL_LIVE_AUTHORITY is None:
        raise RuntimeError(
            "fullband causal live authority SHA가 고정되지 않아 출력이 잠겨 있습니다"
        )
    if plan_path is None:
        raise ValueError("live에는 저장된 exact --plan이 필요합니다")
    path = Path(plan_path).expanduser().resolve()
    payload = path.read_bytes()
    loaded = json.loads(payload.decode("utf-8"))
    if loaded != expected_plan:
        raise ValueError("저장된 plan이 현재 signal/hardware 계약과 다릅니다")
    authority = dict(FULLBAND_CAUSAL_LIVE_AUTHORITY)
    observed = {
        "path": path.relative_to(REPO_ROOT.resolve()).as_posix(),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_sha256": hashlib.sha256(_canonical_json_bytes(loaded)).hexdigest(),
        "pcm_sha256": hashlib.sha256(np.asarray(pcm).tobytes(order="C")).hexdigest(),
    }
    if observed != authority:
        raise ValueError("fullband causal exact live authority path/SHA가 다릅니다")
    return authority


def publish_raw_capture(
    *,
    session_dir: str | Path,
    plan: dict[str, Any],
    planned_pcm: np.ndarray,
    submitted_pcm: np.ndarray,
    input_raw_int32: np.ndarray,
    preflight_raw_int32: np.ndarray,
    callback_time_info: dict[str, np.ndarray] | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """분석 전에 submitted/raw/callback을 immutable raw로만 발행한다."""

    target = Path(session_dir).expanduser().resolve()
    planned = np.asarray(planned_pcm)
    submitted = np.asarray(submitted_pcm)
    recorded = np.asarray(input_raw_int32)
    preflight = np.asarray(preflight_raw_int32)
    invalid = list(metadata.get("invalid_reasons") or [])
    bound_snapshots: dict[str, tuple[Path, str]] = {}
    for label, field in (("level_evidence", "level_evidence"), ("fresh_meter", "meter")):
        binding = metadata.get(field)
        if not isinstance(binding, dict):
            invalid.append(f"{label}_binding_missing")
            continue
        path = Path(str(binding.get("path", ""))).expanduser().resolve()
        expected = str(binding.get("sha256", ""))
        if (
            not path.is_file()
            or len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)
        ):
            invalid.append(f"{label}_binding_invalid")
            continue
        observed = _sha256_file(path)
        if observed != expected:
            invalid.append(f"{label}_sha_mismatch")
            continue
        bound_snapshots[label] = (path, observed)
    if planned.dtype != np.int16 or submitted.dtype != np.int16:
        invalid.append("submitted_pcm_dtype_mismatch")
    if planned.shape != submitted.shape or not np.array_equal(planned, submitted):
        invalid.append("submitted_pcm_not_exact_plan")
    expected_sha = str(plan.get("output", {}).get("pcm_sha256", ""))
    actual_sha = hashlib.sha256(submitted.tobytes(order="C")).hexdigest()
    if actual_sha != expected_sha:
        invalid.append("submitted_pcm_sha_mismatch")
    if recorded.dtype != np.int32 or recorded.shape != (planned.shape[0], 2):
        invalid.append("input_raw_shape_or_dtype_mismatch")
    if preflight.dtype != np.int32 or preflight.ndim != 2 or preflight.shape[1] != 2:
        invalid.append("preflight_raw_shape_or_dtype_mismatch")
    callback_arrays: dict[str, np.ndarray] = {}
    try:
        callback_summary = broadband_measure.validate_callback_time_info(
            callback_time_info, expected_frames=int(planned.shape[0])
        )
        assert callback_time_info is not None
        callback_arrays = {
            field: np.asarray(callback_time_info[field]).copy()
            for field in CALLBACK_TIME_INFO_FIELDS
        }
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        callback_summary = {"valid": False, "error": str(exc)}
        invalid.append("callback_time_info_invalid")
    confirmations = metadata.get("operator_confirmations")
    required = {
        "speaker_output": True,
        "user_present": True,
        "volume_minimum": True,
        "routing_and_geometry": True,
        "same_amplifier_setting": True,
    }
    if confirmations != required:
        invalid.append("operator_confirmations_invalid")
    if int(metadata.get("xrun_count", -1)) != 0:
        invalid.append("xrun_nonzero")
    clip_count = int(
        np.count_nonzero(np.abs(recorded.astype(np.int64)) >= np.iinfo(np.int32).max)
    )
    if clip_count != 0:
        invalid.append("input_clip_nonzero")
    invalid = list(dict.fromkeys(invalid))
    stored = {
        **metadata,
        "raw_capture_schema": RAW_CAPTURE_SCHEMA,
        "status": "PASS" if not invalid else "INVALID",
        "valid": not invalid,
        "invalid_reasons": invalid,
        "submitted_pcm_sha256": actual_sha,
        "callback_timing": callback_summary,
        "analysis_status": "NOT_RUN_RAW_FIRST",
        "plan_binding": {
            "schema": str(plan.get("schema", "")),
            "payload_sha256": hashlib.sha256(_canonical_json_bytes(plan)).hexdigest(),
            "pcm_sha256": expected_sha,
        },
    }
    target.mkdir(parents=True, exist_ok=False)
    paths = interleaved_measure.write_immutable_raw_capture_atomic(
        target,
        metadata=stored,
        arrays={
            "submitted_output_pcm_int16": submitted,
            "input_raw_int32": recorded,
            "preflight_raw_int32": preflight,
            **callback_arrays,
        },
    )
    # raw 발행 후에도 bound bytes가 같아야 한다. 달라졌다면 raw는 보존하고 승격은 실패한다.
    for label, (path, expected) in bound_snapshots.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError(f"raw publish 중 {label} bytes가 변경됐습니다")
    return {"paths": paths, "metadata": stored, "valid": not invalid}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-live", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute_live:
        # 의도적으로 plan 생성, sounddevice import, ALSA 조회보다 먼저 막는다.
        try:
            validate_live_authority_before_audio_import(
                plan_path=args.plan, expected_plan={}, pcm=np.empty(0, dtype=np.int16)
            )
        except (RuntimeError, ValueError) as exc:
            print(f"[중단] {exc}", file=sys.stderr)
            return 2
        raise RuntimeError("authority 고정 뒤에도 live capture 구현 검토가 필요합니다")
    if not args.dry_run:
        print("[중단] --dry-run만 허용됩니다; live는 잠겨 있습니다", file=sys.stderr)
        return 2
    try:
        plan, pcm = build_bound_signal_plan(args.hardware)
        if args.output is not None:
            saved = write_plan_noreplace(args.output, plan)
            print(f"[PASS] signal-only plan 저장: {saved}")
        print(
            "[PASS] fullband causal signal-only dry-run | "
            f"{plan['output']['duration_seconds']:.3f}s | "
            f"peak PCM {plan['output']['peak_pcm']} | "
            f"PCM SHA {plan['output']['pcm_sha256']}"
        )
        print("[잠금] live authority SHA가 없어 오디오 출력은 불가능합니다")
        return 0
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"[실패] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
