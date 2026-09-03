#!/usr/bin/env python3
"""acoustic pilot(scratch/finetune) 학습 진행률을 loss_log.txt에서 읽어 보고한다.

Trainer는 stdout(step 로그) 외에 ``<ckpt_dir>/loss_log.txt``에도 같은 줄을
매 log_every step마다 즉시 flush로 append한다(trainer.py). stdout은 nohup으로
파일 리다이렉트하면 완전 버퍼링돼 학습을 재시작하지 않고는 실시간으로 볼 수
없지만, loss_log.txt는 그 문제와 무관하게 항상 최신 상태다.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

LINE_RE = re.compile(
    r"step\s+(?P<step>\d+)\s*\|\s*loss\s+(?P<loss>[-\d.]+)\s*\|\s*"
    r"nmse_t\s+(?P<nmse_t>[-\d.]+)\s*dB\s*\|\s*nmse_f\s+(?P<nmse_f>[-\d.]+)\s*dB\s*\|\s*"
    r"lr\s+(?P<lr>[\d.eE+-]+)\s*\|\s*(?P<sps>[\d.]+)\s*it/s"
)


def _read_run_until_step(run_dir: Path, default: int) -> int:
    snapshot = run_dir / "config_snapshot.yaml"
    if not snapshot.exists():
        return default
    for line in snapshot.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("run_until_step:"):
            return int(line.split(":", 1)[1].strip())
    return default


def _report_one(run_dir: Path) -> str:
    log_path = run_dir / "loss_log.txt"
    if not log_path.exists():
        return f"{run_dir.name}: loss_log.txt 없음 (아직 step {0}에 못 미침 또는 시작 전)"
    last_line = ""
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_line = line.strip()
    if not last_line:
        return f"{run_dir.name}: loss_log.txt 비어 있음"
    match = LINE_RE.search(last_line)
    if not match:
        return f"{run_dir.name}: 파싱 실패 — {last_line!r}"
    step = int(match["step"])
    sps = float(match["sps"])
    total = _read_run_until_step(run_dir, default=step)
    remaining = max(0, total - step)
    eta_seconds = remaining / sps if sps > 0 else float("inf")
    eta_minutes = eta_seconds / 60.0
    percent = 100.0 * step / total if total else 0.0
    return (
        f"{run_dir.name}: step {step}/{total} ({percent:5.1f}%) | "
        f"loss {match['loss']} | nmse_t {match['nmse_t']} dB | "
        f"nmse_f {match['nmse_f']} dB | {sps:.2f} it/s | "
        f"남은 ETA ~{eta_minutes:.1f}분 (eval 시간 미포함, 대략치)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        dest="run_dirs",
        default=[],
        help="run 디렉터리 (반복 지정 가능). 기본값: 두 acoustic pilot",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="0보다 크면 그 초 간격으로 반복 출력한다",
    )
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.run_dirs] or [
        Path("runs/acoustic_pilot_scratch"),
        Path("runs/acoustic_pilot_finetune"),
    ]

    while True:
        for run_dir in run_dirs:
            print(_report_one(run_dir))
        if args.watch <= 0:
            break
        print("-" * 60)
        time.sleep(args.watch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
