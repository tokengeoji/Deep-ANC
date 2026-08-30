#!/usr/bin/env python3
"""한 개 고정 batch 과적합으로 ANC 학습 gradient가 살아 있는지 진단한다.

실제 장치를 사용하지 않는 GPU/CPU 진단 도구다. 같은 입력과 d(t)를 반복 학습해
nominal plant에서는 손실이 내려가는지, 관측 불가능한 plant 증강에서는 gradient가
상쇄되는지를 분리한다.

예시:
  .venv/bin/python scripts/bench/diagnose_training_overfit.py \
    --model-config configs/model_tiny.yaml --mode nominal \
    --primary-mode secondary_surrogate --lead-samples 109 --require-nmse-db -6
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import load_train_config, load_yaml
from deep_anc.losses.config import LossConfig
from deep_anc.train.trainer import Trainer


_DEAD_KEY = re.compile(r"loss\.([A-Za-z_][A-Za-z0-9_]*)=")


def _warned_keys(loss_cfg: dict) -> tuple[str, ...]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LossConfig.parse(dict(loss_cfg))
    dead: list[str] = []
    for item in caught:
        if issubclass(item.category, DeprecationWarning):
            dead.extend(_DEAD_KEY.findall(str(item.message)))
    return tuple(dict.fromkeys(dead))


def deprecated_lambda_fields() -> frozenset[str]:
    """``LossConfig`` 의 ``lambda_*`` 중 **폐기된** 것을 실행 시점에 알아낸다.

    이름을 여기 적으면 그것이 두 번째 선언이다. 각 필드를 하나씩 넣어 보고 그 모듈이
    DeprecationWarning 을 내는지로 판정한다 — 폐기 사실의 소유자가 답한다.
    """

    dead: set[str] = set()
    for name in LossConfig.model_fields:
        if not name.startswith("lambda_"):
            continue
        try:
            if name in _warned_keys({name: 0.0}):
                dead.add(name)
        except (TypeError, ValueError):  # pragma: no cover - 방어
            continue
    return frozenset(dead)


def deprecated_loss_keys(loss_cfg: dict) -> tuple[str, ...]:
    """이 loss 블록에 남아 있는 **폐기 키**를 ``LossConfig`` 에게 물어서 알아낸다.

    목록을 여기 적지 않는 것이 요점이다. 폐기 여부의 단일 출처는
    ``deep_anc/losses/config.py`` 이고, 이 함수는 그 모듈이 내는 DeprecationWarning 을
    읽을 뿐이다. 목록을 복사해 오면 다음에 항이 하나 더 죽을 때 여기만 옛 사실을
    들고 있게 된다 — 그것이 바로 이 스크립트가 실제로 당한 사고다.
    """

    return _warned_keys(loss_cfg)


def isolate_nmse(loss_cfg: dict) -> dict:
    """``--nmse-only``: NMSE 이외의 모든 항을 끈 loss 블록을 만든다.

    왜 손으로 적지 않는가 (2026-08-06 반증 #17/#20)
    ---------------------------------------------
    이전 판은 ``lambda_mrstft`` / ``lambda_pow`` / ``lambda_clip`` **세 개를 리터럴로**
    0 으로 만들었다. 그 사이에 ``lambda_clip`` 은 폐기 키가 되어 아무 효과가 없어졌고
    (모델이 ``y = L·tanh(u/L)`` 라 구조적으로 죽은 항이었다), 새 항 3개
    (``lambda_dnh`` / ``lambda_frame`` / ``lambda_sat``)가 생겼는데 목록은 그대로였다.
    직접 확인된 결과: "NMSE 만" 모드에서 dnh 0.12 / frame 0.5 / sat 1.0 이 켜진 채
    돌았다 — **격리가 조용히 깨졌다.**

    그래서 이제 끄는 항을 ``LossConfig`` 의 필드에서 유도한다. 손실에 항이 하나
    추가되면(``lambda_*`` 필드가 하나 생기면) 이 함수가 자동으로 그것도 끈다.
    폐기 키가 남아 있으면 0 으로 덮어 조용히 넘어가지 않고 ``ValueError`` 로 죽는다.
    """

    raw = dict(loss_cfg or {})
    dead = deprecated_loss_keys(raw)
    if dead:
        raise ValueError(
            "loss 블록에 폐기 키가 남아 있습니다: "
            + ", ".join(f"loss.{name}" for name in dead)
            + " — 이 키는 아무 효과가 없으므로 0 으로 덮어도 '끈 것'이 아닙니다. "
            "설정에서 지우세요."
        )
    # 유일한 목적함수 항인 NMSE 를 제외한 모든 **살아 있는** 가중치 필드.
    # 폐기 필드를 0 으로 적으면 죽은 키를 새로 심는 셈이라 제외한다.
    retired = deprecated_lambda_fields()
    switches = tuple(
        name
        for name in LossConfig.model_fields
        if name.startswith("lambda_") and name not in retired
    )
    if not switches:  # pragma: no cover - 방어
        raise ValueError("LossConfig 에서 lambda_* 필드를 찾지 못했습니다")
    isolated = {**raw, **{name: 0.0 for name in switches}}
    parsed = LossConfig.parse(isolated)
    still_on = {
        name: float(getattr(parsed, name))
        for name in switches
        if float(getattr(parsed, name) or 0.0) != 0.0
    }
    if still_on:  # pragma: no cover - 방어
        raise ValueError(f"NMSE 격리에 실패했습니다 — 남아 있는 항: {still_on}")
    return isolated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_pretrain.yaml")
    parser.add_argument("--model-config", default="configs/model_tiny.yaml")
    parser.add_argument("--mode", choices=["nominal", "augmented"], default="nominal")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--lead-samples", type=int, default=109)
    parser.add_argument(
        "--primary-mode",
        choices=["rir_surrogate", "secondary_surrogate", "measured"],
        default="secondary_surrogate",
    )
    parser.add_argument(
        "--level-dbfs", type=float, default=-30.0,
        help="고정 batch의 RMS 레벨(기본 -30 dBFS; limiter 내 과적합 게이트)",
    )
    parser.add_argument(
        "--nmse-only", action="store_true",
        help=(
            "NMSE 이외의 모든 손실 항을 끕니다 (끄는 목록은 LossConfig 에서 유도 — "
            "항이 추가돼도 따라옵니다)."
        ),
    )
    parser.add_argument(
        "--require-nmse-db", type=float, default=None,
        help="최종 NMSE가 이 값 이하가 아니면 종료코드 2를 반환합니다.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.log_every < 1:
        parser.error("steps, batch-size, log-every는 1 이상이어야 합니다")

    cfg = load_train_config(args.config)
    cfg["model_config"] = args.model_config
    cfg["model"] = load_yaml(args.model_config)
    cfg["batch_size"] = args.batch_size
    cfg["num_workers"] = 0
    cfg["optimizer"]["lr"] = args.lr
    cfg["optimizer"]["weight_decay"] = 0.0
    # 일반 사전학습의 5k-step warmup을 고정-batch 진단에 적용하면 500-step 동안
    # 유효 LR이 거의 0이 되어 false negative가 난다. 진단은 처음부터 지정 LR을 쓴다.
    cfg["schedule"]["warmup_steps"] = 0
    cfg["schedule"]["total_steps"] = args.steps
    cfg["schedule"]["min_lr"] = args.lr
    cfg["data"]["source_mix_ratio"] = {"synthetic": 1.0}
    cfg["data"]["digital_primary_path_mode"] = args.primary_mode
    cfg["data"]["digital_reference_lead_samples"] = args.lead_samples
    cfg["data"]["level_dbfs"] = [args.level_dbfs, args.level_dbfs]
    if args.mode == "augmented":
        # 실패 원인 재현용: 모델 입력에 주어지지 않는 plant 위상/지연과 강한
        # 비선형을 매 step 랜덤화한다. production 기본 설정에는 사용하지 않는다.
        cfg["data"]["plant_perturbation"] = {
            "delay_jitter_range": [0, 512],
            "gain_tilt_db_per_octave": [-2, 2],
            "gain_db": [-3, 3],
            "allpass_perturb": True,
        }
        cfg["data"]["nonlinear"] = {
            "sef_eta_choices": [0.05, 0.1, 0.2, 10.0],
            "drive_range": [1.0, 4.0],
            "hardclip_prob": 0.05,
        }
    if args.nmse_only:
        cfg["loss"] = isolate_nmse(cfg.get("loss") or {})
    cfg["ckpt_dir"] = tempfile.mkdtemp(prefix=f"deep_anc_overfit_{args.mode}_")

    trainer = Trainer(cfg)
    model = trainer.model
    criterion = trainer.criterion
    optimizer = trainer.optimizer
    batch = next(trainer.train_iter)
    batch = {k: v.clone() for k, v in batch.items()}

    model.train()
    if args.mode == "nominal":
        # eval은 loss의 비선형과 plant perturbation만 끈다. 모델은 계속 train 상태다.
        criterion.eval()
    else:
        criterion.train()

    device = trainer.device
    start = time.monotonic()
    initial_loss = None
    final_metrics: dict[str, float] = {}
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if trainer.amp_dtype is not None and device.type == "cuda":
            with torch.autocast("cuda", dtype=trainer.amp_dtype):
                loss, metrics = trainer._forward_loss(batch)
        else:
            loss, metrics = trainer._forward_loss(batch)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.grad_clip))
        optimizer.step()

        if initial_loss is None:
            initial_loss = float(metrics["loss"])
        final_metrics = metrics
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                x = batch["x"].to(device, non_blocking=True)
                y = model(x)
                y_rms = float(y.float().square().mean().sqrt())
                y_peak = float(y.float().abs().max())
            print(
                f"step {step:5d} | loss {metrics['loss']:9.4f} | "
                f"nmse {metrics['nmse_db']:8.3f} dB | grad {grad_norm:9.3e} | "
                f"y_rms {y_rms:8.5f} | y_peak {y_peak:8.5f}",
                flush=True,
            )

    elapsed = time.monotonic() - start
    assert initial_loss is not None
    print(
        f"결과 mode={args.mode} primary={args.primary_mode} lead={args.lead_samples}: "
        f"loss {initial_loss:.4f} -> {final_metrics['loss']:.4f}, "
        f"NMSE {final_metrics['nmse_db']:.3f} dB, "
        f"{args.steps / max(elapsed, 1e-9):.2f} step/s"
    )

    if trainer.writer is not None:
        trainer.writer.close()
    if trainer.loss_log is not None:
        trainer.loss_log.close()
    if (
        args.require_nmse_db is not None
        and float(final_metrics["nmse_db"]) > args.require_nmse_db
    ):
        print(
            f"[실패] NMSE {final_metrics['nmse_db']:.3f} dB > "
            f"요구 {args.require_nmse_db:.3f} dB",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
