#!/usr/bin/env python3
"""GPU 작업 큐 감독자 CLI.

    job_queue.py verify --queue configs/elice/queue_gpu1.yaml    # 스키마 검증만
    job_queue.py plan   --queue configs/elice/queue_gpu1.yaml    # 실행 예정 순서
    job_queue.py run --allow-legacy-diagnostic --queue configs/elice/queue_gpu1.yaml
    job_queue.py run    --queue ... --dry-run                    # GPU 무접촉 예행

감독자는 어떤 작업 실패에도 종료하지 않는다. 종료하는 순간 GPU 가 놀기 때문이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.ops.job_queue import (  # noqa: E402
    LockHeldError,
    QueueSpecError,
    Supervisor,
    gpu_is_free,
    gpu_snapshot,
    load_queue,
    lock_owner,
    proc_identity,
    read_status,
)

EXIT_OK = 0
EXIT_SPEC = 2
EXIT_ALREADY_RUNNING = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "plan", "verify", "status"])
    parser.add_argument("--queue", required=False, default=None)
    parser.add_argument("--gpu", type=int, default=None, help="큐의 gpu 와 일치해야 한다")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-legacy-diagnostic",
        action="store_true",
        help="legacy_diagnostic 큐를 실제 실행할 때만 명시한다",
    )
    parser.add_argument(
        "--exit-when-drained",
        action="store_true",
        help="큐가 마르면 종료한다(기본은 살아남아 큐 파일 추가를 계속 확인)",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "status":
        state_dir = REPO_ROOT / "runs" / "queue"
        payload = {
            path.stem: read_status(path) for path in sorted(state_dir.glob("gpu*.json"))
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return EXIT_OK

    if not args.queue:
        print("[중단] --queue 가 필요합니다", file=sys.stderr)
        return EXIT_SPEC

    try:
        spec = load_queue(args.queue)
    except QueueSpecError as exc:
        print(f"[중단] 큐 정의 오류: {exc}", file=sys.stderr)
        return EXIT_SPEC

    if args.gpu is not None and args.gpu != spec.gpu:
        print(f"[중단] --gpu {args.gpu} 가 큐의 gpu {spec.gpu} 와 다릅니다", file=sys.stderr)
        return EXIT_SPEC

    if (
        args.command == "run"
        and spec.execution_class == "legacy_diagnostic"
        and not args.allow_legacy_diagnostic
    ):
        print(
            "[중단] legacy_diagnostic 큐는 canonical 학습을 만들 수 없습니다. "
            "과거 진단을 의도한 경우에만 --allow-legacy-diagnostic를 명시하세요.",
            file=sys.stderr,
        )
        return EXIT_SPEC

    if args.command == "verify":
        print(f"[OK] {spec.source}: class={spec.execution_class}, gpu={spec.gpu}, 작업 {len(spec.jobs)}개, adopt {len(spec.adopt)}개")
        return EXIT_OK

    if args.command == "plan":
        print(f"큐: {spec.source} (GPU{spec.gpu})")
        for entry in spec.entry_gate.get("wait_for_pids") or []:
            identity = proc_identity(int(entry["pid"]))
            alive = "생존" if identity else "없음"
            print(f"  진입 게이트 pid {entry['pid']}: {alive} — {entry.get('note', '')}")
            if identity:
                print(f"      cmdline: {identity['cmdline'][:120]}")
        for path in spec.entry_gate.get("acquire_locks") or []:
            print(f"  진입 게이트 lock: {path} owner={lock_owner(REPO_ROOT / path)}")
        free, reason = gpu_is_free(spec.gpu)
        print(f"  GPU{spec.gpu} 유휴: {free} — {reason}")
        print(f"  GPU{spec.gpu} 스냅샷: {gpu_snapshot(spec.gpu)}")
        for entry in spec.adopt or []:
            print(f"  [adopt] {entry.get('id')} ← {entry.get('run_dir')}")
        for index, job in enumerate(spec.jobs, 1):
            print(f"  {index}. [{job.tier}] {job.id} ({job.kind}) — {job.reason}")
        return EXIT_OK

    supervisor = Supervisor(
        spec, dry_run=args.dry_run, exit_when_drained=args.exit_when_drained
    )
    try:
        supervisor.acquire_own_lock()
    except LockHeldError as exc:
        print(f"[중단] {exc}", file=sys.stderr)
        return EXIT_ALREADY_RUNNING
    try:
        return supervisor.run()
    except KeyboardInterrupt:
        supervisor.log("인터럽트 — 감독자만 종료한다(자식 학습은 그대로 둔다)")
        return EXIT_OK
    finally:
        supervisor.close()


if __name__ == "__main__":
    raise SystemExit(main())
