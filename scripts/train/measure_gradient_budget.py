#!/usr/bin/env python3
"""G0 또는 선택된 20k checkpoint의 strict-S output-y gradient evidence를 만든다.

이 스크립트는 소리를 출력하지 않는다. ``checkpoint.pt`` 자체와 deterministic synthetic
validation batch를 immutable receipt에 보관한다. 측정량은 model parameter-gradient가
아니라 ``‖λ·∂L_dnh/∂y‖ / ‖∂L_nmse/∂y‖``다. 0.2–0.4 판정과 범위 밖일 때의
λ 추천은 canonical ledger 검증 시에도 같은 raw bytes에서 다시 계산된다.

공식 alpha별 pre-pilot 예시:
  .venv/bin/python scripts/train/measure_gradient_budget.py \
    --g0-receipt results/training_prerequisites/evidence/g0_<alpha>_<lambda>/receipt.json \
    --out-dir results/training_prerequisites/evidence/gradient_<alpha>_<lambda>

``--checkpoint runs/<pilot>/ckpt/best.pt --authoritative-g0-receipt <receipt>``는
선택 후 교차검산용이다. 새 batch를 만들지 않고 모든 alpha가 공유한 batch SHA 중 winner
G0의 authoritative batch artifact를 그대로 참조한다. schema v7 campaign을 여는 authority는 각 alpha의
approved G0 receipt에 결속된 pre-pilot mode다.
실패 G0는 ``--failed-g0-receipt``로 recommendation-only receipt를 만들 수 있지만,
이 kind는 pilot/init 자격을 절대 열지 않으며 추천 λ의 fresh G0를 처음부터 재실행해야 한다.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.train.campaign_evidence import (  # noqa: E402
    G0_BATCH_SIZE,
    EVIDENCE_SCHEMA_VERSION,
    FAILED_G0_RECEIPT_KIND,
    G0_RECEIPT_KIND,
    calibrate_dnh_output_gradient,
    publish_failed_g0_gradient_recommendation,
    publish_gradient_budget_evidence,
    publish_prepilot_gradient_evidence,
    repo_path,
    snapshot_from_reference,
    validate_failed_g0_gradient_recommendation,
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="post-pilot audit용 selected loss-pilot best.pt")
    source.add_argument(
        "--g0-receipt",
        help="공식 alpha별 pre-pilot calibration용 approved G0 receipt.json",
    )
    source.add_argument(
        "--failed-g0-receipt",
        help=(
            "NMSE<-6 실패 G0의 다음 fresh-run lambda 추천 전용 receipt.json. "
            "campaign/pilot 자격은 열지 않습니다."
        ),
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="no-replace raw gradient evidence directory",
    )
    parser.add_argument(
        "--authoritative-g0-receipt",
        help=(
            "--checkpoint selected-20k audit가 재사용할 winner G0 receipt. "
            "receipt의 exact batch artifact를 복제 없이 참조합니다."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=G0_BATCH_SIZE,
        help=f"고정 synthetic val batch 수 (campaign 승인값: {G0_BATCH_SIZE})",
    )
    return parser


def _g0_source(
    receipt_path: Path,
    *,
    expected_kind: str,
    label: str,
) -> tuple[dict, dict[str, torch.Tensor], Path]:
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} receipt를 읽을 수 없습니다: {receipt_path}") from exc
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema_version",
        "kind",
        "checkpoint",
        "batch",
        "environment",
    }:
        raise ValueError(f"{label} receipt key 집합이 canonical schema와 다릅니다")
    if (
        receipt["schema_version"] != EVIDENCE_SCHEMA_VERSION
        or receipt["kind"] != expected_kind
    ):
        raise ValueError(f"{label} source kind가 다릅니다")
    checkpoint_snapshot = snapshot_from_reference(
        REPO_ROOT, receipt["checkpoint"], label=f"{label} checkpoint"
    )
    batch_snapshot = snapshot_from_reference(
        REPO_ROOT, receipt["batch"], label=f"{label} batch"
    )
    raw = _checkpoint(checkpoint_snapshot.path)
    try:
        batch = torch.load(
            io.BytesIO(batch_snapshot.content), map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} fixed batch를 읽을 수 없습니다") from exc
    if not isinstance(batch, dict) or set(batch) != {"x", "d"}:
        raise ValueError("G0 fixed batch는 정확히 x,d여야 합니다")
    if any(not isinstance(value, torch.Tensor) for value in batch.values()):
        raise ValueError("G0 fixed batch x,d가 tensor가 아닙니다")
    return raw, batch, batch_snapshot.path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size != G0_BATCH_SIZE:
        raise ValueError(
            f"campaign gradient evidence batch-size는 {G0_BATCH_SIZE}로 고정됩니다"
        )
    if args.checkpoint and not args.authoritative_g0_receipt:
        raise ValueError(
            "selected-20k --checkpoint audit에는 --authoritative-g0-receipt가 필요합니다"
        )
    if not args.checkpoint and args.authoritative_g0_receipt:
        raise ValueError(
            "--authoritative-g0-receipt는 selected-20k --checkpoint audit에만 사용합니다"
        )
    failed_source = False
    if args.g0_receipt:
        g0_receipt = repo_path(
            REPO_ROOT, args.g0_receipt, label="pre-pilot G0 receipt"
        )
        raw, batch, _batch_artifact = _g0_source(
            g0_receipt,
            expected_kind=G0_RECEIPT_KIND,
            label="pre-pilot G0",
        )
        cfg = raw["cfg"]
        if (
            str(cfg.get("experiment_role")) != "diagnostic_overfit"
            or cfg.get("init_eligible") is not False
            or int(raw.get("step", -1)) != 500
        ):
            raise ValueError("pre-pilot source가 approved 500-step G0 형식이 아닙니다")
        receipt = publish_prepilot_gradient_evidence(
            repo_root=REPO_ROOT,
            output_dir=args.out_dir,
            g0_receipt=g0_receipt,
        )
        label = "alpha별 pre-pilot G0 DNH calibration"
    elif args.failed_g0_receipt:
        failed_source = True
        g0_receipt = repo_path(
            REPO_ROOT,
            args.failed_g0_receipt,
            label="failed G0 diagnostic receipt",
        )
        raw, batch, _batch_artifact = _g0_source(
            g0_receipt,
            expected_kind=FAILED_G0_RECEIPT_KIND,
            label="failed G0 diagnostic",
        )
        cfg = raw["cfg"]
        if (
            str(cfg.get("experiment_role")) != "diagnostic_overfit"
            or cfg.get("init_eligible") is not False
            or int(raw.get("step", -1)) != 500
        ):
            raise ValueError("failed G0 source가 sealed 500-step diagnostic 형식이 아닙니다")
        receipt = None
        label = "failed G0 DNH recommendation-only calibration"
    else:
        checkpoint = repo_path(
            REPO_ROOT, args.checkpoint, label="gradient checkpoint"
        )
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
        authoritative_g0_receipt = repo_path(
            REPO_ROOT,
            args.authoritative_g0_receipt,
            label="authoritative common G0 receipt",
        )
        _g0_raw, batch, batch_artifact = _g0_source(
            authoritative_g0_receipt,
            expected_kind=G0_RECEIPT_KIND,
            label="authoritative common G0",
        )
        receipt = publish_gradient_budget_evidence(
            repo_root=REPO_ROOT,
            output_dir=args.out_dir,
            checkpoint=checkpoint,
            batch_artifact=batch_artifact,
        )
        label = "selected 20k pilot DNH calibration"
    calibration = calibrate_dnh_output_gradient(
        cfg,
        raw["model"],
        batch,
        repo_root=REPO_ROOT,
        label=label,
    )
    if failed_source:
        receipt = publish_failed_g0_gradient_recommendation(
            repo_root=REPO_ROOT,
            output_dir=args.out_dir,
            failed_g0_receipt=g0_receipt,
            calibration=calibration,
        )
        # receipt claim이 raw 실패 checkpoint/batch에서 재현되고, 실제 G0 NMSE도
        # 여전히 -6 dB 미만이 아님을 같은 실행에서 다시 닫는다.
        validate_failed_g0_gradient_recommendation(
            receipt,
            repo_root=REPO_ROOT,
        )
    assert receipt is not None
    print(f"[campaign gradient] raw receipt → {receipt}", flush=True)
    print(
        "[campaign gradient] domain=model_output_y(global_l2) "
        f"current_lambda={calibration['current_lambda_dnh']:.12g} "
        f"current_share={calibration['current_share']:.9f}",
        flush=True,
    )
    if failed_source:
        if calibration["approved"] is True:
            detail = (
                "현재 lambda_dnh share는 0.2–0.4 안이므로 유지합니다. "
                "G0 실패 원인은 DNH 비율 밖에서 찾아야 합니다."
            )
        else:
            detail = (
                "다음 fresh G0 추천 lambda_dnh="
                f"{calibration['recommended_lambda_dnh']:.12g}; "
                "추천값 실제 재계산 share="
                f"{calibration['recommended_share']:.9f}"
            )
        print(
            "[campaign gradient] DIAGNOSTIC ONLY — "
            f"{detail} 실패 checkpoint는 init/pilot에 사용할 수 없고, 새 contract로 "
            "G0를 처음부터 재실행해야 합니다.",
            flush=True,
        )
        return 2
    if calibration["approved"] is True:
        print(
            "[campaign gradient] PASS 0.2–0.4 — 현재 lambda_dnh를 유지합니다.",
            flush=True,
        )
        return 0
    print(
        "[campaign gradient] FAIL 0.2–0.4 — 현재 campaign은 승인하지 않습니다. "
        f"다음 full G0/pilot/probe 재실행 추천 lambda_dnh="
        f"{calibration['recommended_lambda_dnh']:.12g}; "
        f"추천값 실제 재계산 share={calibration['recommended_share']:.9f}",
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
