#!/usr/bin/env python3
"""raw campaign evidence만으로 canonical-pretrain v7 ledger를 no-replace 발행한다.

이 명령은 사람이 NMSE, gradient share, pilot score, winner, probe PASS를 입력받지
않는다. alpha별 G0/pre-pilot calibration/pilot/probe와 selected-20k gradient,
A100 smoke artifact pathname만 받고, ledger 발행 뒤 같은 raw bytes를 다시
계산하는 validator를 즉시 실행한다.

예시 (모든 artifact는 Elice repository 내부의 immutable 파일이어야 한다):
  .venv/bin/python scripts/train/issue_canonical_pretrain_prerequisite.py \
    --bootstrap-receipt-sha256 "$BOOTSTRAP_SHA" \
    --loss-alpha <WINNER_ALPHA> --loss-lambda-dnh <WINNER_LAMBDA> \
    --g0-receipt results/training_prerequisites/evidence/g0_07/receipt.json \
    --prepilot-gradient-receipt results/training_prerequisites/evidence/gradient_07/receipt.json \
    --g0-receipt results/training_prerequisites/evidence/g0_10/receipt.json \
    --prepilot-gradient-receipt results/training_prerequisites/evidence/gradient_10/receipt.json \
    --gradient-receipt results/training_prerequisites/evidence/gradient_selected20k/receipt.json \
    --pilot-best runs/<pilot07>/ckpt/best.pt --pilot-last runs/<pilot07>/ckpt/last.pt \
    --pilot-metrics runs/<pilot07>/eval_recorded_val/metrics.npz \
    --pilot-best runs/<pilot10>/ckpt/best.pt --pilot-last runs/<pilot10>/ckpt/last.pt \
    --pilot-metrics runs/<pilot10>/eval_recorded_val/metrics.npz \
    --probe-best runs/<probe07>/ckpt/best.pt --probe-last runs/<probe07>/ckpt/last.pt \
    --probe-metrics runs/<probe07>/eval_recorded_val/metrics.npz \
    --probe-init-checkpoint runs/<pilot07>/ckpt/best.pt \
    --probe-best runs/<probe10>/ckpt/best.pt --probe-last runs/<probe10>/ckpt/last.pt \
    --probe-metrics runs/<probe10>/eval_recorded_val/metrics.npz \
    --probe-init-checkpoint runs/<pilot10>/ckpt/best.pt \
    --smoke-receipt results/training_prerequisites/a100_pretrain_smoke/<target>/receipt.json \
    --smoke-environment-receipt results/training_prerequisites/a100_pretrain_smoke/<target>/environment_receipt.json \
    --smoke-telemetry results/training_prerequisites/a100_pretrain_smoke/<target>/telemetry.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.config import (  # noqa: E402
    canonical_recorded_manifest_for_data,
    load_train_config,
)
from deep_anc.train.campaign_evidence import (  # noqa: E402
    PILOT_SELECTION_RULE,
    make_campaign_evidence_reference,
)
from deep_anc.train.campaign_prerequisite import (  # noqa: E402
    CANONICAL_PATH,
    SCHEMA_VERSION,
    validate_canonical_pretrain_ledger_payload,
    validate_canonical_pretrain_prerequisites,
)
from deep_anc.train.evaluation_contract import (  # noqa: E402
    snapshot_regular_file,
    write_json_exclusive,
)
from deep_anc.train.experiment_contract import require_exact_source_trust  # noqa: E402


def _sha(value: str, *, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label}가 64자리 SHA-256이 아닙니다")
    return text


def _reference(path: str, *, label: str) -> dict[str, str]:
    return make_campaign_evidence_reference(REPO_ROOT, path, label=label)


def _canonical_cfg(
    config: str,
    bootstrap_sha: str,
    prerequisite_sha: str,
    *,
    loss_alpha: float,
    loss_lambda_dnh: float,
) -> dict:
    alpha = float(loss_alpha)
    lambda_dnh = float(loss_lambda_dnh)
    if not math.isfinite(alpha):
        raise ValueError("selected alpha가 finite가 아닙니다")
    if not math.isfinite(lambda_dnh) or lambda_dnh <= 0.0:
        raise ValueError("selected lambda_dnh가 finite 양수가 아닙니다")
    alpha_literal = f"{alpha:.1f}" if alpha.is_integer() else f"{alpha:.12g}"
    lambda_literal = repr(lambda_dnh)
    return load_train_config(
        config,
        [
            f"data.bootstrap_receipt_sha256={bootstrap_sha}",
            f"campaign_prerequisite_sha256={prerequisite_sha}",
            # ``1``이 아닌 ``1.0``으로 직렬화해 loss selection digest가 pilot과
            # 달라지는 것을 막는다. winner와 다른 값을 주면 raw validator가 거부한다.
            f"loss.nmse_cvar_alpha={alpha_literal}",
            f"loss.lambda_dnh={lambda_literal}",
        ],
    )


def _ledger_sha256(payload: dict) -> str:
    """``write_json_exclusive``와 byte-for-byte 같은 prospective JSON digest."""

    content = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_pretrain_tiny.yaml")
    parser.add_argument("--bootstrap-receipt-sha256", required=True)
    parser.add_argument(
        "--loss-alpha",
        type=float,
        required=True,
        help=(
            "raw measured-probe selection으로 유도된 canonical alpha. issuer와 "
            "이후 100k "
            "명령은 반드시 같은 float literal을 사용한다."
        ),
    )
    parser.add_argument(
        "--loss-lambda-dnh",
        type=float,
        required=True,
        help="raw winner identity에서 유도된 canonical lambda_dnh",
    )
    parser.add_argument("--g0-receipt", action="append", required=True)
    parser.add_argument(
        "--prepilot-gradient-receipt", action="append", required=True
    )
    parser.add_argument(
        "--gradient-receipt",
        required=True,
        help="selected 20k winner의 post-pilot output-gradient receipt",
    )
    parser.add_argument("--pilot-best", action="append", required=True)
    parser.add_argument("--pilot-last", action="append", required=True)
    parser.add_argument("--pilot-metrics", action="append", required=True)
    parser.add_argument(
        "--pilot-manifest",
        action="append",
        default=None,
        help=(
            "각 pilot recorded-val manifest. 생략 시 resolved canonical data의 "
            "recorded generation에서 권위 manifest를 유도해 모든 candidate에 "
            "사용합니다."
        ),
    )
    parser.add_argument("--probe-best", action="append", required=True)
    parser.add_argument("--probe-last", action="append", required=True)
    parser.add_argument("--probe-metrics", action="append", required=True)
    parser.add_argument(
        "--probe-manifest",
        action="append",
        default=None,
        help=(
            "각 measured probe recorded-val manifest. 생략 시 resolved canonical "
            "data의 recorded generation에서 권위 manifest를 유도해 모든 "
            "candidate에 사용합니다."
        ),
    )
    parser.add_argument("--probe-init-checkpoint", action="append", required=True)
    parser.add_argument("--smoke-receipt", required=True)
    parser.add_argument("--smoke-environment-receipt", required=True)
    parser.add_argument("--smoke-telemetry", required=True)
    parser.add_argument(
        "--out",
        default=CANONICAL_PATH,
        help=f"canonical fixed path only: {CANONICAL_PATH}",
    )
    return parser


def _ordered_candidate_inputs(
    args: argparse.Namespace, *, default_manifest: str
) -> list[tuple[str, ...]]:
    """CLI의 pilot/probe 반복 인자를 동일 index chain으로 고정한다."""

    lengths = {
        len(args.g0_receipt),
        len(args.prepilot_gradient_receipt),
        len(args.pilot_best),
        len(args.pilot_last),
        len(args.pilot_metrics),
        len(args.probe_best),
        len(args.probe_last),
        len(args.probe_metrics),
        len(args.probe_init_checkpoint),
    }
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError(
            "G0/pre-pilot-gradient/pilot/probe best/last/metrics/init은 같은 개수와 순서의 "
            "최소 두 candidate여야 합니다"
        )
    count = len(args.pilot_best)
    if not str(default_manifest).strip():
        raise ValueError("resolved canonical recorded manifest가 비었습니다")
    pilot_manifests = args.pilot_manifest or [default_manifest] * count
    probe_manifests = args.probe_manifest or [default_manifest] * count
    if len(pilot_manifests) != count:
        raise ValueError("--pilot-manifest는 생략하거나 모든 candidate마다 한 번 지정해야 합니다")
    if len(probe_manifests) != count:
        raise ValueError("--probe-manifest는 생략하거나 모든 candidate마다 한 번 지정해야 합니다")
    return list(
        zip(
            args.g0_receipt,
            args.prepilot_gradient_receipt,
            args.pilot_best,
            args.pilot_last,
            args.pilot_metrics,
            pilot_manifests,
            args.probe_best,
            args.probe_last,
            args.probe_metrics,
            probe_manifests,
            args.probe_init_checkpoint,
            strict=True,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.out != CANONICAL_PATH:
        raise ValueError(f"campaign ledger output은 {CANONICAL_PATH}로 고정됩니다")
    bootstrap_sha = _sha(args.bootstrap_receipt_sha256, label="bootstrap receipt SHA")
    loss_alpha = float(args.loss_alpha)
    loss_lambda_dnh = float(args.loss_lambda_dnh)
    destination = REPO_ROOT / CANONICAL_PATH
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"canonical prerequisite ledger를 덮어쓸 수 없습니다: {destination}")

    # 아직 ledger bytes가 없으므로 dummy external anchor로 canonical semantic target을
    # resolve한다. campaign SHA는 source/P/S/loss 선택 자체에는 포함하지 않으며, 실제
    # write 뒤에는 새 SHA로 resolve한 cfg를 validator가 다시 검사한다.
    provisional = _canonical_cfg(
        args.config,
        bootstrap_sha,
        "0" * 64,
        loss_alpha=loss_alpha,
        loss_lambda_dnh=loss_lambda_dnh,
    )
    require_exact_source_trust(
        provisional, repo_root=REPO_ROOT, roles={"canonical_pretrain"}
    )
    canonical_recorded_manifest = canonical_recorded_manifest_for_data(
        provisional.get("data") or {}
    )
    candidate_inputs = _ordered_candidate_inputs(
        args, default_manifest=canonical_recorded_manifest
    )
    contract = provisional["experiment_contract"]
    source = contract["source"]
    artifacts = contract["artifacts"]
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "git_commit": source["git_commit"],
            "source_tree_sha256": source["source_tree_sha256"],
            "bootstrap_receipt_sha256": bootstrap_sha,
            "primary_path_sha256": artifacts["primary_path"]["sha256"],
            "secondary_path_sha256": artifacts["secondary_path"]["sha256"],
        },
        "gradient_budget": {
            "receipt": _reference(args.gradient_receipt, label="gradient budget receipt")
        },
        "loss_pilot_selection": {
            "selection_rule": PILOT_SELECTION_RULE,
            "candidates": [
                {
                    "g0": {
                        "receipt": _reference(
                            g0_receipt, label=f"candidate[{index}] G0 receipt"
                        )
                    },
                    "gradient_calibration": {
                        "receipt": _reference(
                            prepilot_gradient_receipt,
                            label=f"candidate[{index}] pre-pilot gradient receipt",
                        )
                    },
                    "pilot": {
                        "best_checkpoint": _reference(
                            pilot_best, label=f"pilot[{index}] best checkpoint"
                        ),
                        "last_checkpoint": _reference(
                            pilot_last, label=f"pilot[{index}] last checkpoint"
                        ),
                        "metrics": _reference(
                            pilot_metrics, label=f"pilot[{index}] recorded-val metrics"
                        ),
                        "manifest": _reference(
                            pilot_manifest, label=f"pilot[{index}] recorded manifest"
                        ),
                    },
                    "measured_probe": {
                        "best_checkpoint": _reference(
                            probe_best, label=f"probe[{index}] best checkpoint"
                        ),
                        "last_checkpoint": _reference(
                            probe_last, label=f"probe[{index}] last checkpoint"
                        ),
                        "metrics": _reference(
                            probe_metrics, label=f"probe[{index}] recorded-val metrics"
                        ),
                        "manifest": _reference(
                            probe_manifest, label=f"probe[{index}] recorded manifest"
                        ),
                        "init_checkpoint": _reference(
                            probe_init, label=f"probe[{index}] pilot init checkpoint"
                        ),
                    },
                }
                for index, (
                    g0_receipt,
                    prepilot_gradient_receipt,
                    pilot_best,
                    pilot_last,
                    pilot_metrics,
                    pilot_manifest,
                    probe_best,
                    probe_last,
                    probe_metrics,
                    probe_manifest,
                    probe_init,
                ) in enumerate(candidate_inputs)
            ],
        },
        "a100_smoke_resume": {
            "evidence": _reference(args.smoke_receipt, label="A100 smoke receipt"),
            "environment_receipt": _reference(
                args.smoke_environment_receipt, label="A100 smoke environment receipt"
            ),
            "telemetry": _reference(args.smoke_telemetry, label="A100 smoke telemetry"),
        },
    }
    # 이 prospective SHA가 config의 external trust anchor가 된다. raw validator를
    # 먼저 돌려 실패한 evidence가 canonical fixed pathname을 점유하지 않게 한다.
    prospective_sha = _ledger_sha256(ledger)
    prospective_cfg = _canonical_cfg(
        args.config,
        bootstrap_sha,
        prospective_sha,
        loss_alpha=loss_alpha,
        loss_lambda_dnh=loss_lambda_dnh,
    )
    validate_canonical_pretrain_ledger_payload(
        prospective_cfg,
        ledger,
        repo_root=REPO_ROOT,
    )
    write_json_exclusive(destination, ledger)
    ledger_sha = snapshot_regular_file(destination).sha256
    if ledger_sha != prospective_sha:  # pragma: no cover - writer/digest invariant
        raise RuntimeError("canonical prerequisite JSON digest가 prospective SHA와 다릅니다")
    # 원자 공개 뒤, canonical entrypoint와 동일한 path+external SHA에서 raw proof를
    # 재계산한다. 어떤 hand-written score/boolean도 이 단계에서는 읽지 않는다.
    canonical = _canonical_cfg(
        args.config,
        bootstrap_sha,
        ledger_sha,
        loss_alpha=loss_alpha,
        loss_lambda_dnh=loss_lambda_dnh,
    )
    validate_canonical_pretrain_prerequisites(canonical, repo_root=REPO_ROOT)
    print(
        "[campaign prerequisite] PASS — "
        f"ledger={CANONICAL_PATH} sha256={ledger_sha}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
