#!/usr/bin/env python3
"""광대역 recorded-v2 source plan/실측 수집 무음 dry-run.

현재 구현은 계약 검증 전용이다. ``RECORDED_V2_LIVE_AUTHORITY=None``이므로
``--execute-live``도 sounddevice를 import하기 전에 실패한다. canonical fullband causal
P/S와 48개 이상의 verified source plan이 생긴 뒤 exact SHA를 새 commit에서 고정해야만
live recorder 구현 검토를 시작할 수 있다.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.recorded_v2_capture import (  # noqa: E402
    RECORDED_V2_LIVE_AUTHORITY,
    RecordedV2Blocked,
    capture_contract,
    validate_source_plan,
)


DEFAULT_SOURCE_PLAN = "data/source_plans/recorded_broadband_v2/canonical_v1.json"
DEFAULT_PLANT = "assets/measured/fullband_causal_physical_plant_evidence_v1.json"
DEFAULT_OUT_ROOT = "data/recorded_broadband_v2/canonical_v1"
INPUT_ONLY_PREFLIGHT_SECONDS_PER_SESSION = 3.5
SILENT_STREAM_SETTLE_SECONDS_PER_SESSION = 1.0


def _assert_read_only_audio_unoccupied() -> dict[str, object]:
    """향후 live 직전 /dev/snd 점유와 kernel PCM status를 read-only로 검사한다."""

    status_rows: list[dict[str, str]] = []
    occupied: list[str] = []
    for raw_path in sorted(glob.glob("/proc/asound/card*/pcm*/sub*/status")):
        path = Path(raw_path)
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        status_rows.append({"path": raw_path, "status": text})
        if text and text.lower() != "closed":
            occupied.append(f"{raw_path}: {text}")
    command = ["fuser", "-v", *sorted(glob.glob("/dev/snd/*"))]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    # fuser exit 1은 사용자 process가 없다는 뜻이다. 출력이 있으면 control node인지까지
    # 사람이 검토해야 하며 live PCM은 자동으로 열지 않는다.
    fuser_text = (result.stdout + result.stderr).strip()
    pcm_owners = [
        line
        for line in fuser_text.splitlines()
        if "pcmC" in line and line.strip()
    ]
    if occupied or pcm_owners:
        detail = "; ".join(occupied + pcm_owners)
        raise RecordedV2Blocked(f"오디오 PCM 장치가 점유 중입니다: {detail}")
    return {
        "status": "PASS_READ_ONLY",
        "kernel_pcm_status": status_rows,
        "fuser_returncode": result.returncode,
        "fuser_text": fuser_text,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", default=DEFAULT_SOURCE_PLAN)
    parser.add_argument("--plant-evidence", default=DEFAULT_PLANT)
    parser.add_argument("--expected-source-plan-sha256")
    parser.add_argument("--expected-plant-sha256")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-live", action="store_true")
    parser.add_argument("--confirm-user-present", action="store_true")
    parser.add_argument("--confirm-volume-minimum", action="store_true")
    parser.add_argument("--confirm-routing-and-geometry", action="store_true")
    parser.add_argument("--confirm-speaker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _relative_or_blocked(value: str, *, label: str) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else REPO_ROOT / path
    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise RecordedV2Blocked(f"{label} 경로가 저장소 밖입니다") from exc
    return resolved


def _offline_validate(args: argparse.Namespace) -> dict[str, object]:
    if not args.expected_source_plan_sha256:
        raise RecordedV2Blocked("--expected-source-plan-sha256 외부 anchor가 없습니다")
    if not args.expected_plant_sha256:
        raise RecordedV2Blocked("--expected-plant-sha256 외부 anchor가 없습니다")
    source_plan = _relative_or_blocked(args.source_plan, label="source plan")
    plant = _relative_or_blocked(args.plant_evidence, label="plant evidence")
    if not source_plan.is_file():
        raise RecordedV2Blocked(f"verified source plan이 없습니다: {source_plan}")
    if not plant.is_file():
        raise RecordedV2Blocked(f"canonical fullband causal plant가 없습니다: {plant}")
    result = validate_source_plan(
        source_plan,
        expected_plan_sha256=args.expected_source_plan_sha256,
        expected_plant_sha256=args.expected_plant_sha256,
        repository_root=REPO_ROOT,
    )
    if result["plant"]["file"]["path"] != plant.relative_to(REPO_ROOT).as_posix():
        raise RecordedV2Blocked(
            "--plant-evidence 경로가 source plan이 봉인한 exact plant 경로와 다릅니다"
        )
    out_root = _relative_or_blocked(args.out_root, label="recorded-v2 out-root")
    if out_root.exists() or out_root.is_symlink():
        raise RecordedV2Blocked(f"no-replace out-root가 이미 존재합니다: {out_root}")
    return {**result, "out_root": out_root.relative_to(REPO_ROOT).as_posix()}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = capture_contract()
    print(
        "recorded-v2 capture contract: "
        f"{contract['contract_sha256']} | source=15.000초/720000 frames | "
        "clock=152--600Hz only | LIVE_AUTHORITY=None"
    )
    if args.execute_live:
        confirmations = {
            "--confirm-user-present": args.confirm_user_present,
            "--confirm-volume-minimum": args.confirm_volume_minimum,
            "--confirm-routing-and-geometry": args.confirm_routing_and_geometry,
        }
        missing = [name for name, passed in confirmations.items() if not passed]
        if missing:
            print("[BLOCKED] 실기 확인 누락: " + ", ".join(missing), file=sys.stderr)
            return 2
    try:
        result = _offline_validate(args)
    except (OSError, RuntimeError, ValueError, RecordedV2Blocked) as exc:
        print(f"[BLOCKED] {exc}", file=sys.stderr)
        print("[오디오] sounddevice import/open 0회, 파일 생성/수정 0개", file=sys.stderr)
        return 2

    count = int(result["session_count"])
    audible = count * 15.0
    output_open = audible + count * SILENT_STREAM_SETTLE_SECONDS_PER_SESSION
    connected_upper = output_open + count * INPUT_ONLY_PREFLIGHT_SECONDS_PER_SESSION
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(
        f"예상 source audible={audible:.1f}초, output-open={output_open:.1f}초, "
        f"input-only 포함 연결 상한={connected_upper:.1f}초 (분석/저장 제외)"
    )
    if args.dry_run:
        # 구조 PASS는 live authority가 아니다. 이 분기를 테스트 fixture로 통과해도 스피커를
        # 열 수 없으며, 실제 live는 아래 외부 authority를 별도로 요구한다.
        print("[DRY-RUN STRUCTURAL PASS] 파일 생성/수정 및 오디오 장치 open 없음")
        print("[LIVE BLOCKED] RECORDED_V2_LIVE_AUTHORITY=None")
        return 2

    if RECORDED_V2_LIVE_AUTHORITY is None:
        print("[BLOCKED] recorded-v2 exact live authority가 None입니다", file=sys.stderr)
        print("[오디오] sounddevice import/open 0회", file=sys.stderr)
        return 2
    expected_authority = {
        "source_plan_sha256": args.expected_source_plan_sha256,
        "plant_evidence_sha256": args.expected_plant_sha256,
        "capture_contract_sha256": contract["contract_sha256"],
    }
    if dict(RECORDED_V2_LIVE_AUTHORITY) != expected_authority:
        print("[BLOCKED] live authority와 exact plan/plant/contract SHA가 다릅니다", file=sys.stderr)
        return 2
    try:
        _assert_read_only_audio_unoccupied()
    except (OSError, RecordedV2Blocked) as exc:
        print(f"[BLOCKED] {exc}", file=sys.stderr)
        return 2
    print(
        "[BLOCKED] live callback/raw-first publisher 결합은 authority 고정 뒤 별도 검토가 "
        "필요합니다. 오디오를 열지 않았습니다.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
