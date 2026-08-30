#!/usr/bin/env python3
"""실측 fine-tune 준비 또는 완료를 원샷으로 검증한다 (오디오 출력 없음).

준비 검사::

  .venv/bin/python scripts/train/check_finetune.py \
    --config configs/train_finetune.yaml

완료 검사::

  .venv/bin/python scripts/train/check_finetune.py \
    --config configs/train_finetune.yaml \
    --completion-checkpoint runs/finetune_tiny/ckpt/best.pt

완료 검사는 같은 checkpoint로 생성된 ``eval_recorded_{val,test}/metrics.npz``의
SHA-256 provenance와 G4 동시 PASS까지 요구한다. 조건이 없거나 불확실하면 FAIL이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT, load_train_config  # noqa: E402
from deep_anc.train.finetune_readiness import (  # noqa: E402
    audit_finetune_completion,
    audit_finetune_readiness,
    render_audit_markdown,
)
from deep_anc.train.process_lock import autostart_state_dir  # noqa: E402


def _repo_output(value: str | Path) -> Path:
    path = Path(value).expanduser()
    path = path if path.is_absolute() else REPO_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"리포트는 Deep_ANC 내부에만 저장할 수 있습니다: {path}") from exc
    return path


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_finetune.yaml")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="학습 설정 오버라이드(key=value); train.py와 같은 규약",
    )
    parser.add_argument(
        "--completion-checkpoint",
        default=None,
        help="지정하면 준비가 아니라 fine-tuning 완료를 검증",
    )
    parser.add_argument(
        "--val-metrics",
        default=None,
        help="기본: <checkpoint run>/eval_recorded_val/metrics.npz",
    )
    parser.add_argument(
        "--test-metrics",
        default=None,
        help="기본: <checkpoint run>/eval_recorded_test/metrics.npz",
    )
    parser.add_argument("--selection", default=None)
    parser.add_argument("--test-capability", default=None)
    parser.add_argument("--test-consumed-marker", default=None)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="기본: results/finetune_autostart/<run-key>/audit; Deep_ANC 내부만 허용",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_train_config(args.config, args.overrides)
        run_dir = Path(str(cfg["ckpt_dir"]))
        run_dir = run_dir if run_dir.is_absolute() else REPO_ROOT / run_dir
        out_dir = _repo_output(args.out_dir or autostart_state_dir(run_dir) / "audit")

        if args.completion_checkpoint:
            checkpoint = Path(args.completion_checkpoint).expanduser()
            checkpoint = checkpoint if checkpoint.is_absolute() else REPO_ROOT / checkpoint
            # checkpoint.parent=<run>/ckpt, parent.parent=<run>
            candidate_run = checkpoint.resolve().parent.parent
            val_metrics = args.val_metrics or candidate_run / "eval_recorded_val" / "metrics.npz"
            test_metrics = args.test_metrics or candidate_run / "eval_recorded_test" / "metrics.npz"
            report = audit_finetune_completion(
                cfg,
                checkpoint=checkpoint,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                selection=args.selection,
                test_capability=args.test_capability,
                test_consumed_marker=args.test_consumed_marker,
            )
            stem = "completion"
        else:
            report = audit_finetune_readiness(cfg)
            stem = "readiness"
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[오류] fine-tune 검사 설정 실패: {exc}", file=sys.stderr)
        return 2

    json_path = out_dir / f"{stem}.json"
    markdown_path = out_dir / f"{stem}.md"
    _atomic_write(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(markdown_path, render_audit_markdown(report))

    verdict = "PASS" if report["ok"] else "FAIL"
    print(f"[{verdict}] {report['kind']}")
    for item in report["checks"]:
        print(f"  [{'PASS' if item['ok'] else 'FAIL'}] {item['id']}: {item['message']}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
