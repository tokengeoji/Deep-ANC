#!/usr/bin/env python3
"""학습 진입점.

Canonical pretrain/fine-tune은 bootstrap/campaign/init SHA anchor를 요구하므로
설정 파일만 주어서는 실행되지 않는다. 실제 명령은
``docs/05_training_elice.md``와 최신 ``HANDOFF.md``를 따른다.

진단용 예시:
  .venv/bin/python scripts/train/train.py \
    --config configs/train_pretrain_tiny.yaml \
    --set experiment_role=diagnostic_overfit --set init_eligible=false \
    --set contract_run_dir=false --set run_until_step=500
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import load_train_config          # noqa: E402
from deep_anc.train.finetune_readiness import (        # noqa: E402
    require_finetune_readiness,
)
from deep_anc.train.process_lock import (               # noqa: E402
    LockHeldError,
    ProcessLock,
    autostart_state_dir,
    resolve_run_dir,
)
from deep_anc.train.trainer import (                   # noqa: E402
    STRICT_RUN_ROLES,
    Trainer,
    preflight_canonical_resume,
)


def requires_same_run_lock(cfg: dict) -> bool:
    return str(cfg.get("experiment_role", "")) in STRICT_RUN_ROLES or any(
        bool(cfg.get(name, False))
        for name in (
            "require_measured_primary_path",
            "require_init_checkpoint",
            "require_recorded_manifest",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="재개할 체크포인트 경로")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="설정 오버라이드 (예: --set batch_size=4 --set optimizer.lr=1e-4)",
    )
    args = parser.parse_args()

    cfg = load_train_config(args.config, args.overrides)
    if args.resume:
        cfg["resume"] = args.resume
    # canonical resume은 ProcessLock/state-dir조차 만들기 전에 동일 immutable
    # snapshot으로 contract/model/optimizer/scheduler/RNG를 전부 preview한다.
    resume_preflight = preflight_canonical_resume(cfg)

    # measured/recorded fine-tune은 파일 존재만 확인하고 시작하지 않는다. official
    # P/S 품질·대역·동일 디지털 gain, lead, 완료된 init checkpoint, recorded 전수
    # QA/분할/분량을 GPU 초기화 전에 모두 통과해야 한다. pretrain 설정에는 영향 없음.
    is_guarded_finetune = any(
        bool(cfg.get(name, False))
        for name in (
            "require_measured_primary_path",
            "require_init_checkpoint",
            "require_recorded_manifest",
        )
    )
    if is_guarded_finetune:
        report = require_finetune_readiness(cfg)
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[fine-tune readiness] PASS ({len(report['checks'])} gates)",
                flush=True,
            )

    if requires_same_run_lock(cfg):
        run_dir = resolve_run_dir(cfg["ckpt_dir"])
        state_dir = autostart_state_dir(run_dir)
        # torchrun이면 rank0만 launcher/run lock을 소유한다. 모든 rank가 같은
        # flock을 잡으려 하면 정상 DDP 자체를 중복 실행으로 오인한다.
        if int(os.environ.get("RANK", "0")) != 0:
            Trainer(cfg, resume_preflight=resume_preflight).train()
            return 0
        try:
            with ProcessLock(
                state_dir / "train.lock",
                role=f"{cfg.get('experiment_role', 'training')} train",
                metadata={"run_dir": str(run_dir)},
            ):
                Trainer(cfg, resume_preflight=resume_preflight).train()
        except LockHeldError as exc:
            print(f"[중단] {exc}", file=sys.stderr)
            return 3
    else:
        Trainer(cfg, resume_preflight=resume_preflight).train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
