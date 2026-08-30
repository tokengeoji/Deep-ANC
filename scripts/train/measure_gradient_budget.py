#!/usr/bin/env python3
"""선택된 20k loss-pilot checkpoint의 strict-S gradient budget raw evidence를 만든다.

이 스크립트는 소리를 출력하지 않는다. ``checkpoint.pt`` 자체와 deterministic synthetic
validation batch를 immutable receipt에 보관할 뿐이며, 0.2–0.4 판정은 canonical ledger
검증 시 그 raw bytes에서 다시 계산된다.

예시:
  .venv/bin/python scripts/train/measure_gradient_budget.py \
    --checkpoint runs/<pilot>/ckpt/best.pt \
    --out-dir results/training_prerequisites/evidence/gradient_<pilot-sha>
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.synth_dataset import SynthANCDataset, make_eval_batch  # noqa: E402
from deep_anc.train.campaign_evidence import (  # noqa: E402
    G0_BATCH_SIZE,
    publish_gradient_budget_evidence,
    repo_path,
)
from deep_anc.train.evaluation_contract import snapshot_regular_file  # noqa: E402


def _checkpoint(path: Path) -> dict:
    snapshot = snapshot_regular_file(path)
    try:
        raw = torch.load(
            io.BytesIO(snapshot.content), map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint를 읽을 수 없습니다: {path}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("cfg"), dict):
        raise ValueError("checkpoint에 resolved cfg가 없습니다")
    return raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="selected loss-pilot best.pt")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="no-replace raw gradient evidence directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=G0_BATCH_SIZE,
        help=f"고정 synthetic val batch 수 (campaign 승인값: {G0_BATCH_SIZE})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size != G0_BATCH_SIZE:
        raise ValueError(
            f"campaign gradient evidence batch-size는 {G0_BATCH_SIZE}로 고정됩니다"
        )
    checkpoint = repo_path(REPO_ROOT, args.checkpoint, label="gradient checkpoint")
    if checkpoint.name != "best.pt":
        raise ValueError("gradient evidence는 selected loss-pilot best.pt만 허용합니다")
    raw = _checkpoint(checkpoint)
    cfg = raw["cfg"]
    if (
        str(cfg.get("experiment_role")) != "loss_pilot"
        or cfg.get("init_eligible") is not False
        or int(cfg.get("run_until_step", -1)) != 20_000
    ):
        raise ValueError("gradient evidence checkpoint가 completed loss_pilot 20k가 아닙니다")
    dataset = SynthANCDataset(cfg["data"], cfg["duct"], split="val", seed=1234)
    batch = make_eval_batch(dataset, n_items=G0_BATCH_SIZE, seed=12345)
    receipt = publish_gradient_budget_evidence(
        repo_root=REPO_ROOT,
        output_dir=args.out_dir,
        checkpoint=checkpoint,
        batch=batch,
    )
    print(f"[campaign gradient] raw receipt → {receipt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
