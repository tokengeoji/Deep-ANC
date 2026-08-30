#!/usr/bin/env python3
"""raw campaign evidence만으로 canonical-pretrain v5 ledger를 no-replace 발행한다.

이 명령은 사람이 NMSE, gradient share, pilot score, winner, probe PASS를 입력받지
않는다. G0 receipt, checkpoint/metrics, measured probe, A100 smoke artifact의 pathname만
받고, ledger 발행 뒤 같은 raw bytes를 다시 계산하는 validator를 즉시 실행한다.

예시 (모든 artifact는 Elice repository 내부의 immutable 파일이어야 한다):
  .venv/bin/python scripts/train/issue_canonical_pretrain_prerequisite.py \
    --bootstrap-receipt-sha256 "$BOOTSTRAP_SHA" \
    --loss-alpha <WINNER_ALPHA> \
    --g0-receipt results/training_prerequisites/evidence/g0_x/receipt.json \
    --gradient-receipt results/training_prerequisites/evidence/gradient_x/receipt.json \
    --pilot-best runs/<pilot07>/ckpt/best.pt --pilot-last runs/<pilot07>/ckpt/last.pt \
    --pilot-metrics runs/<pilot07>/eval_recorded_val/metrics.npz \
    --pilot-best runs/<pilot10>/ckpt/best.pt --pilot-last runs/<pilot10>/ckpt/last.pt \
    --pilot-metrics runs/<pilot10>/eval_recorded_val/metrics.npz \
    --probe-best runs/<probe>/ckpt/best.pt --probe-last runs/<probe>/ckpt/last.pt \
    --probe-metrics runs/<probe>/eval_recorded_val/metrics.npz \
    --probe-init-checkpoint runs/<selected-pilot>/ckpt/best.pt \
    --smoke-receipt results/training_prerequisites/a100_pretrain_smoke/<target>/receipt.json \
    --smoke-environment-receipt results/training_prerequisites/a100_pretrain_smoke/<target>/environment_receipt.json \
    --smoke-telemetry results/training_prerequisites/a100_pretrain_smoke/<target>/telemetry.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.config import load_train_config  # noqa: E402
from deep_anc.train.campaign_evidence import (  # noqa: E402
    CANONICAL_RECORDED_VAL_MANIFEST,
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
) -> dict:
    return load_train_config(
        config,
        [
            f"data.bootstrap_receipt_sha256={bootstrap_sha}",
            f"campaign_prerequisite_sha256={prerequisite_sha}",
            # ``1``이 아닌 ``1.0``으로 직렬화해 loss selection digest가 pilot과
            # 달라지는 것을 막는다. winner와 다른 값을 주면 raw validator가 거부한다.
            f"loss.nmse_cvar_alpha={float(loss_alpha):.1f}" if float(loss_alpha).is_integer()
            else f"loss.nmse_cvar_alpha={float(loss_alpha):.12g}",
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
            "raw pilot selection으로 유도된 canonical alpha. issuer와 이후 100k "
            "명령은 반드시 같은 float literal을 사용한다."
        ),
    )
    parser.add_argument("--g0-receipt", required=True)
    parser.add_argument("--gradient-receipt", required=True)
    parser.add_argument("--pilot-best", action="append", required=True)
    parser.add_argument("--pilot-last", action="append", required=True)
    parser.add_argument("--pilot-metrics", action="append", required=True)
    parser.add_argument(
        "--pilot-manifest",
        action="append",
        default=None,
        help=(
            "각 pilot recorded-val manifest. 생략 시 canonical "
            f"{CANONICAL_RECORDED_VAL_MANIFEST}를 모든 candidate에 사용합니다."
        ),
    )
    parser.add_argument("--probe-best", required=True)
    parser.add_argument("--probe-last", required=True)
    parser.add_argument("--probe-metrics", required=True)
    parser.add_argument(
        "--probe-manifest", default=CANONICAL_RECORDED_VAL_MANIFEST
    )
    parser.add_argument("--probe-init-checkpoint", required=True)
    parser.add_argument("--smoke-receipt", required=True)
    parser.add_argument("--smoke-environment-receipt", required=True)
    parser.add_argument("--smoke-telemetry", required=True)
    parser.add_argument(
        "--out",
        default=CANONICAL_PATH,
        help=f"canonical fixed path only: {CANONICAL_PATH}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.out != CANONICAL_PATH:
        raise ValueError(f"campaign ledger output은 {CANONICAL_PATH}로 고정됩니다")
    bootstrap_sha = _sha(args.bootstrap_receipt_sha256, label="bootstrap receipt SHA")
    loss_alpha = float(args.loss_alpha)
    lengths = {len(args.pilot_best), len(args.pilot_last), len(args.pilot_metrics)}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError("pilot best/last/metrics는 같은 개수의 최소 두 candidate여야 합니다")
    manifests = args.pilot_manifest or [CANONICAL_RECORDED_VAL_MANIFEST] * len(
        args.pilot_best
    )
    if len(manifests) != len(args.pilot_best):
        raise ValueError("--pilot-manifest는 생략하거나 모든 candidate마다 한 번 지정해야 합니다")
    destination = REPO_ROOT / CANONICAL_PATH
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"canonical prerequisite ledger를 덮어쓸 수 없습니다: {destination}")

    # 아직 ledger bytes가 없으므로 dummy external anchor로 canonical semantic target을
    # resolve한다. campaign SHA는 source/P/S/loss 선택 자체에는 포함하지 않으며, 실제
    # write 뒤에는 새 SHA로 resolve한 cfg를 validator가 다시 검사한다.
    provisional = _canonical_cfg(
        args.config, bootstrap_sha, "0" * 64, loss_alpha=loss_alpha
    )
    require_exact_source_trust(
        provisional, repo_root=REPO_ROOT, roles={"canonical_pretrain"}
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
        "g0": {"receipt": _reference(args.g0_receipt, label="campaign G0 receipt")},
        "gradient_budget": {
            "receipt": _reference(args.gradient_receipt, label="gradient budget receipt")
        },
        "loss_pilot_selection": {
            "selection_rule": PILOT_SELECTION_RULE,
            "candidates": [
                {
                    "best_checkpoint": _reference(best, label=f"pilot[{index}] best checkpoint"),
                    "last_checkpoint": _reference(last, label=f"pilot[{index}] last checkpoint"),
                    "metrics": _reference(metrics, label=f"pilot[{index}] recorded-val metrics"),
                    "manifest": _reference(manifest, label=f"pilot[{index}] recorded manifest"),
                }
                for index, (best, last, metrics, manifest) in enumerate(
                    zip(args.pilot_best, args.pilot_last, args.pilot_metrics, manifests)
                )
            ],
        },
        "measured_probe": {
            "best_checkpoint": _reference(args.probe_best, label="measured probe best checkpoint"),
            "last_checkpoint": _reference(args.probe_last, label="measured probe last checkpoint"),
            "metrics": _reference(args.probe_metrics, label="measured probe recorded-val metrics"),
            "manifest": _reference(args.probe_manifest, label="measured probe recorded manifest"),
            "init_checkpoint": _reference(args.probe_init_checkpoint, label="measured probe init checkpoint"),
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
        args.config, bootstrap_sha, prospective_sha, loss_alpha=loss_alpha
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
        args.config, bootstrap_sha, ledger_sha, loss_alpha=loss_alpha
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
