#!/usr/bin/env python3
"""primary borderline + fresh seed-20260903 smoke로 prerequisite를 발행한다.

사람이 판정 boolean, winner, seed-neutral digest 또는 출력 경로를 입력하지 않는다.
primary selection과 smoke raw artifact를 validator가 다시 열어 고정 경로에 no-replace로
발행한 뒤, published bytes와 외부 SHA가 결속된 resolved config로 한 번 더 검증한다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.config import load_train_config  # noqa: E402
from deep_anc.train.evaluation_contract import (  # noqa: E402
    read_json_snapshot,
    snapshot_regular_file,
    write_json_exclusive,
)
from deep_anc.train.experiment_contract import require_exact_source_trust  # noqa: E402
from deep_anc.train.second_seed_prerequisite import (  # noqa: E402
    CONFIG_PATH_KEY,
    CONFIG_SHA256_KEY,
    SECONDARY_SEED,
    build_second_seed_prerequisite_payload,
    prerequisite_sha256,
    second_seed_prerequisite_path,
    validate_second_seed_prerequisite_payload,
    validate_second_seed_prerequisites,
)


def _sha(value: object, *, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label}가 64자리 lowercase SHA-256이 아닙니다")
    return text


def _relative(path: Path) -> str:
    target = path.absolute()
    try:
        return target.relative_to(REPO_ROOT.absolute()).as_posix()
    except ValueError as exc:
        raise ValueError(f"second-seed prerequisite 경로가 저장소 밖입니다: {target}") from exc


def _overrides(
    *,
    bootstrap_sha256: str,
    campaign_prerequisite_sha256: str,
    loss_alpha: float,
    loss_lambda_dnh: float,
    prerequisite_path: Path,
    prerequisite_sha256_value: str,
) -> list[str]:
    alpha = float(loss_alpha)
    lambda_dnh = float(loss_lambda_dnh)
    if not math.isfinite(alpha) or alpha not in {0.7, 0.85, 1.0}:
        raise ValueError("selected alpha는 0.7/0.85/1.0 중 하나여야 합니다")
    if not math.isfinite(lambda_dnh) or lambda_dnh <= 0.0:
        raise ValueError("selected lambda_dnh는 finite 양수여야 합니다")
    return [
        f"seed={SECONDARY_SEED}",
        f"data.bootstrap_receipt_sha256={bootstrap_sha256}",
        f"campaign_prerequisite_sha256={campaign_prerequisite_sha256}",
        f"loss.nmse_cvar_alpha={alpha!r}",
        "loss.lambda_frame=0.0",
        f"loss.lambda_dnh={lambda_dnh!r}",
        f"{CONFIG_PATH_KEY}={json.dumps(_relative(prerequisite_path))}",
        f"{CONFIG_SHA256_KEY}={json.dumps(prerequisite_sha256_value)}",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_pretrain_tiny.yaml")
    parser.add_argument("--bootstrap-receipt-sha256", required=True)
    parser.add_argument("--campaign-prerequisite-sha256", required=True)
    parser.add_argument("--loss-alpha", type=float, required=True)
    parser.add_argument("--loss-lambda-dnh", type=float, required=True)
    parser.add_argument("--primary-selection", required=True)
    parser.add_argument("--secondary-smoke-receipt", required=True)
    parser.add_argument("--secondary-smoke-environment-receipt", required=True)
    parser.add_argument("--secondary-smoke-telemetry", required=True)
    parser.add_argument(
        "--out",
        default=None,
        help="생략 권장. 지정하면 seed-neutral campaign fixed path와 exact해야 합니다.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bootstrap_sha = _sha(args.bootstrap_receipt_sha256, label="bootstrap receipt SHA")
    ledger_sha = _sha(
        args.campaign_prerequisite_sha256,
        label="canonical campaign prerequisite SHA",
    )

    selection, _ = read_json_snapshot(args.primary_selection)
    campaign = _sha(
        selection.get("seed_neutral_campaign_sha256"),
        label="primary selection seed-neutral campaign SHA",
    )
    destination = second_seed_prerequisite_path(campaign, repo_root=REPO_ROOT)
    if args.out is not None:
        requested = Path(args.out).expanduser()
        if not requested.is_absolute():
            requested = REPO_ROOT / requested
        if requested.absolute() != destination.absolute():
            raise ValueError(
                "--out은 seed-neutral campaign fixed path만 허용합니다: "
                f"requested={requested.absolute()}, expected={destination}"
            )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"second-seed prerequisite를 덮어쓸 수 없습니다: {destination}"
        )

    config = Path(args.config).expanduser()
    if not config.is_absolute():
        config = REPO_ROOT / config
    provisional = load_train_config(
        config,
        _overrides(
            bootstrap_sha256=bootstrap_sha,
            campaign_prerequisite_sha256=ledger_sha,
            loss_alpha=args.loss_alpha,
            loss_lambda_dnh=args.loss_lambda_dnh,
            prerequisite_path=destination,
            prerequisite_sha256_value="0" * 64,
        ),
    )
    payload = build_second_seed_prerequisite_payload(
        provisional,
        primary_selection=args.primary_selection,
        smoke_receipt=args.secondary_smoke_receipt,
        smoke_environment_receipt=args.secondary_smoke_environment_receipt,
        smoke_telemetry=args.secondary_smoke_telemetry,
        repo_root=REPO_ROOT,
    )
    prospective_sha = prerequisite_sha256(payload)
    prospective_cfg = load_train_config(
        config,
        _overrides(
            bootstrap_sha256=bootstrap_sha,
            campaign_prerequisite_sha256=ledger_sha,
            loss_alpha=args.loss_alpha,
            loss_lambda_dnh=args.loss_lambda_dnh,
            prerequisite_path=destination,
            prerequisite_sha256_value=prospective_sha,
        ),
    )
    validate_second_seed_prerequisite_payload(
        prospective_cfg, payload, repo_root=REPO_ROOT
    )
    write_json_exclusive(destination, payload)
    published = snapshot_regular_file(destination)
    if published.sha256 != prospective_sha:  # pragma: no cover - writer invariant
        raise RuntimeError("published second-seed prerequisite SHA가 prospective SHA와 다릅니다")

    # prerequisite 자체도 experiment contract 입력이다. 발행 전 resolved config는 이를
    # exists=false로 지문하므로, published bytes가 생긴 뒤 반드시 다시 resolve해야 실제
    # 학습과 같은 source/experiment contract를 검증할 수 있다.
    final_cfg = load_train_config(
        config,
        _overrides(
            bootstrap_sha256=bootstrap_sha,
            campaign_prerequisite_sha256=ledger_sha,
            loss_alpha=args.loss_alpha,
            loss_lambda_dnh=args.loss_lambda_dnh,
            prerequisite_path=destination,
            prerequisite_sha256_value=published.sha256,
        ),
    )
    require_exact_source_trust(
        final_cfg, repo_root=REPO_ROOT, roles={"canonical_pretrain"}
    )
    validate_second_seed_prerequisites(final_cfg, repo_root=REPO_ROOT)
    print(
        "[second-seed prerequisite] PASS — "
        f"path={_relative(destination)} sha256={published.sha256}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
