#!/usr/bin/env python3
"""한 개 고정 batch 과적합으로 ANC 학습 gradient가 살아 있는지 진단한다.

실제 장치를 사용하지 않는 GPU/CPU 진단 도구다. 같은 입력과 d(t)를 반복 학습해
nominal plant에서는 손실이 내려가는지, 관측 불가능한 plant 증강에서는 gradient가
상쇄되는지를 분리한다.

예시:
  .venv/bin/python scripts/bench/diagnose_training_overfit.py \
    --model-config configs/model_tiny.yaml --mode nominal \
    --primary-mode secondary_surrogate

기본 lead는 로드한 학습 설정의 ``PlantDelays.lead()`` 결과를 사용하고, G0 합격선
``trusted NMSE <= -6 dB``를 강제한다. ``--lead-samples``와 ``--no-require-nmse``는
의도적인 반증/진단 실행에서만 사용한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import load_train_config
from deep_anc.losses.config import LossConfig
from deep_anc.train.trainer import (
    Trainer,
    validate_finite_gradients,
    validate_finite_output,
    validate_finite_parameters,
    validate_g0_nmse,
)
from deep_anc.train.campaign_evidence import (
    configure_g0_evidence_determinism,
    publish_failed_g0_evidence,
    publish_g0_evidence,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


_DEAD_KEY = re.compile(r"loss\.([A-Za-z_][A-Za-z0-9_]*)=")
_LOSS_TERM_FIELDS = {
    "mrstft": "lambda_mrstft",
    "dnh": "lambda_dnh",
    "frame": "lambda_frame",
    "sat": "lambda_sat",
    "pow": "lambda_pow",
}


def resolved_config_sha256(cfg: dict) -> str:
    """진단 로그가 서로 다른 seed/config를 같은 control로 오인하지 않게 한다."""

    payload = json.dumps(
        cfg,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fixed_batch_sha256(batch: dict[str, torch.Tensor]) -> str:
    """고정-batch 진단의 실제 입력 바이트를 이름·dtype·shape와 함께 결속한다."""

    digest = hashlib.sha256()
    for name in sorted(batch):
        value = batch[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"G0 batch.{name}가 Tensor가 아닙니다: {type(value).__name__}")
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # campaign receipt는 tiny canonical derivative config에서만 유효하다. legacy
    # train_pretrain.yaml 기본값으로 receipt를 만들고 뒤늦게 거부되는 낭비를 막는다.
    parser.add_argument("--config", default="configs/train_pretrain_tiny.yaml")
    parser.add_argument("--model-config", default="configs/model_tiny.yaml")
    parser.add_argument("--mode", choices=["nominal", "augmented"], default="nominal")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--loss-alpha",
        type=float,
        choices=[0.7, 0.85, 1.0],
        default=None,
        help="alpha별 공식 G0/pilot identity의 nmse_cvar_alpha",
    )
    parser.add_argument(
        "--loss-lambda-dnh",
        type=float,
        default=None,
        help="alpha별 공식 G0/pilot identity의 finite 양수 lambda_dnh",
    )
    parser.add_argument(
        "--lead-samples",
        type=int,
        default=None,
        help="진단용 lead override. 기본은 학습 설정에서 유도된 값을 그대로 사용합니다.",
    )
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
        "--disable-loss-term",
        action="append",
        choices=sorted(_LOSS_TERM_FIELDS),
        default=[],
        help=(
            "지정한 보조 손실 항만 0으로 끄는 통제 실험(여러 번 지정 가능). "
            "--nmse-only와 함께 쓸 수 없습니다."
        ),
    )
    parser.add_argument(
        "--require-nmse-db", type=float, default=-6.0,
        help="최종 NMSE가 이 값 미만이 아니면 종료코드 2를 반환합니다(기본 G0: -6 dB).",
    )
    parser.add_argument(
        "--no-require-nmse",
        action="store_const",
        const=None,
        dest="require_nmse_db",
        help="진단 전용: G0 NMSE 합격선을 적용하지 않습니다.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--bootstrap-receipt-sha256",
        default=None,
        help=(
            "campaign G0 receipt를 만들 때 결속할 Elice bootstrap SHA-256. "
            "--evidence-dir와 함께 필수입니다."
        ),
    )
    parser.add_argument(
        "--evidence-dir",
        default=None,
        help=(
            "G0의 immutable raw checkpoint/batch/receipt directory. 합격하면 campaign "
            "G0 kind, 실패하면 lambda 재추천 전용 failed-diagnostic kind로 봉인합니다. "
            "프로세스 시작 전 승인된 CUBLAS_WORKSPACE_CONFIG가 필요하고, "
            "live 결정론 backend를 함께 결속합니다. 기존 directory는 절대 "
            "덮어쓰지 않습니다."
        ),
    )
    return parser


def build_diagnostic_overrides(args: argparse.Namespace, ckpt_dir: str) -> list[str]:
    """canonical 설정을 오염시키지 않는 전용 diagnostic resolved-config 입력."""

    overrides = [
        "experiment_role=diagnostic_overfit",
        "canonical_trust_policy=null",
        "init_eligible=false",
        "contract_run_dir=false",
        "require_init_checkpoint=false",
        "init_ckpt=null",
        "resume=null",
        f"model_config={json.dumps(str(args.model_config))}",
        f"batch_size={int(args.batch_size)}",
        "num_workers=0",
        f"optimizer.lr={float(args.lr)}",
        "optimizer.weight_decay=0.0",
        "schedule.warmup_steps=0",
        f"schedule.total_steps={int(args.steps)}",
        f"schedule.min_lr={float(args.lr)}",
        "data.source_mix_ratio={synthetic: 1.0}",
        f"data.digital_primary_path_mode={args.primary_mode}",
        f"data.level_dbfs=[{float(args.level_dbfs)}, {float(args.level_dbfs)}]",
        f"ckpt_dir={json.dumps(str(ckpt_dir))}",
    ]
    # 일반 diagnostic은 과거 ERR-context 분포를 보존한다. campaign authority를 만드는
    # 공식 G0만 canonical train/pilot과 동일한 REF-only input-contract file을 유지한다.
    if getattr(args, "evidence_dir", None) is None:
        overrides.append("data_model_input_contract_config=null")
    if args.lead_samples is not None:
        overrides.append(
            f"data.digital_reference_lead_samples={int(args.lead_samples)}"
        )
    if args.loss_alpha is not None:
        alpha = float(args.loss_alpha)
        alpha_literal = f"{alpha:.1f}" if alpha.is_integer() else f"{alpha:.12g}"
        overrides.append(f"loss.nmse_cvar_alpha={alpha_literal}")
    if args.loss_lambda_dnh is not None:
        overrides.append(f"loss.lambda_dnh={float(args.loss_lambda_dnh)!r}")
    bootstrap_sha = getattr(args, "bootstrap_receipt_sha256", None)
    if bootstrap_sha is not None:
        overrides.append(f"data.bootstrap_receipt_sha256={str(bootstrap_sha).lower()}")
    if args.mode == "augmented":
        overrides.extend(
            [
                "data.plant_perturbation="
                + json.dumps(
                    {
                        "delay_jitter_range": [0, 512],
                        "gain_tilt_db_per_octave": [-2, 2],
                        "gain_db": [-3, 3],
                        "allpass_perturb": True,
                    }
                ),
                "data.nonlinear="
                + json.dumps(
                    {
                        "sef_eta_choices": [0.05, 0.1, 0.2, 10.0],
                        "drive_range": [1.0, 4.0],
                        "hardclip_prob": 0.05,
                    }
                ),
            ]
        )
    for term in args.disable_loss_term:
        overrides.append(f"loss.{_LOSS_TERM_FIELDS[term]}=0.0")
    return overrides


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.log_every < 1:
        parser.error("steps, batch-size, log-every는 1 이상이어야 합니다")
    if args.nmse_only and args.disable_loss_term:
        parser.error("--nmse-only와 --disable-loss-term은 함께 사용할 수 없습니다")
    if args.loss_lambda_dnh is not None and (
        not math.isfinite(float(args.loss_lambda_dnh))
        or float(args.loss_lambda_dnh) <= 0.0
    ):
        parser.error("--loss-lambda-dnh는 finite 양수여야 합니다")
    if args.evidence_dir is not None:
        bootstrap = str(args.bootstrap_receipt_sha256 or "").lower()
        if len(bootstrap) != 64 or any(char not in "0123456789abcdef" for char in bootstrap):
            parser.error("--evidence-dir에는 64자리 --bootstrap-receipt-sha256이 필요합니다")
        if args.loss_alpha is None or args.loss_lambda_dnh is None:
            parser.error(
                "공식 --evidence-dir에는 alpha별 --loss-alpha와 "
                "--loss-lambda-dnh가 모두 필요합니다"
            )
        if (
            args.mode != "nominal"
            or args.steps != 500
            or args.batch_size != 4
            or args.primary_mode != "secondary_surrogate"
            or args.nmse_only
            or args.disable_loss_term
            or args.lead_samples is not None
            or args.require_nmse_db != -6.0
        ):
            parser.error(
                "공식 G0 evidence는 nominal/500-step/batch4/secondary_surrogate/"
                "full-loss/derived-lead/NMSE<-6 계약만 허용합니다"
            )

    # 일반 diagnostic-only 경로는 기존처럼 backend 상태를 강제하지 않는다. 반면
    # campaign을 여는 G0 evidence는 모델/optimizer/CUDA context를 만들기 전에
    # 결정론 backend를 활성화하고 그 live 시작 상태를 마지막 receipt와 결속한다.
    g0_determinism_environment = (
        configure_g0_evidence_determinism()
        if args.evidence_dir is not None
        else None
    )

    diagnostic_dir = tempfile.mkdtemp(prefix=f"deep_anc_overfit_{args.mode}_")
    overrides = build_diagnostic_overrides(args, diagnostic_dir)
    cfg = load_train_config(args.config, overrides)
    lead_samples = int(
        cfg["data"].get("digital_reference_lead_samples", 0)
        if args.lead_samples is None
        else args.lead_samples
    )
    if lead_samples < 0:
        parser.error("lead-samples는 0 이상이어야 합니다")
    if args.nmse_only:
        isolated = isolate_nmse(cfg.get("loss") or {})
        cfg = load_train_config(
            args.config,
            [*overrides, "loss=" + json.dumps(isolated)],
        )

    trainer = Trainer(cfg)
    model = trainer.model
    criterion = trainer.criterion
    optimizer = trainer.optimizer
    batch = next(trainer.train_iter)
    batch = {k: v.clone() for k, v in batch.items()}
    print(
        "[diagnostic contract] "
        f"config={Path(args.config).as_posix()} "
        f"resolved_config_sha256={resolved_config_sha256(cfg)} "
        f"seed={cfg.get('seed')} "
        f"role={cfg.get('experiment_role')} "
        f"loss_start_sample={trainer.loss_start_sample} "
        f"batch_sha256={fixed_batch_sha256(batch)} "
        f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}",
        flush=True,
    )

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
        validate_finite_gradients(model)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.grad_clip))
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError(f"G0 finite 계약 위반: gradient norm={grad_norm}")
        validate_finite_gradients(model)
        optimizer.step()
        validate_finite_parameters(model)

        if initial_loss is None:
            initial_loss = float(metrics["loss"])
        final_metrics = metrics
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                x = batch["x"].to(device, non_blocking=True)
                y = model(x)
                validate_finite_output(y)
                y_rms = float(y.float().square().mean().sqrt())
                y_peak = float(y.float().abs().max())
            print(
                f"step {step:5d} | loss {metrics['loss']:9.4f} | "
                f"nmse {metrics['nmse_db']:8.3f} dB | grad {grad_norm:9.3e} | "
                f"y_rms {y_rms:8.5f} | y_peak {y_peak:8.5f} | "
                f"mrstft {metrics['mrstft']:8.4f} | dnh {metrics['dnh']:8.4f} | "
                f"frame {metrics['frame']:8.4f} | sat {metrics['sat']:8.4f}",
                flush=True,
            )

    elapsed = time.monotonic() - start
    assert initial_loss is not None
    print(
        f"결과 mode={args.mode} primary={args.primary_mode} lead={lead_samples} "
        f"disabled={','.join(args.disable_loss_term) or 'none'}: "
        f"loss {initial_loss:.4f} -> {final_metrics['loss']:.4f}, "
        f"NMSE {final_metrics['nmse_db']:.3f} dB, "
        f"{args.steps / max(elapsed, 1e-9):.2f} step/s"
    )

    if trainer.writer is not None:
        trainer.writer.close()
    if trainer.loss_log is not None:
        trainer.loss_log.close()
    g0_failure: BaseException | None = None
    if args.require_nmse_db is not None:
        try:
            validate_g0_nmse(
                final_metrics, maximum_exclusive_db=float(args.require_nmse_db)
            )
        except (FloatingPointError, ValueError) as exc:
            print(f"[실패] {exc}", file=sys.stderr)
            g0_failure = exc
    if args.evidence_dir is not None:
        raw_model = model.module if hasattr(model, "module") else model
        publisher = (
            publish_failed_g0_evidence
            if g0_failure is not None
            else publish_g0_evidence
        )
        receipt = publisher(
            repo_root=REPO_ROOT,
            output_dir=args.evidence_dir,
            cfg=cfg,
            model_state=raw_model.state_dict(),
            batch={"x": batch["x"], "d": batch["d"]},
            steps=args.steps,
            mode=args.mode,
            primary_mode=args.primary_mode,
            require_nmse_db=args.require_nmse_db,
            nmse_only=args.nmse_only,
            disable_loss_terms=args.disable_loss_term,
            determinism_environment=g0_determinism_environment,
        )
        if g0_failure is None:
            print(f"[campaign G0] PASS raw receipt → {receipt}", flush=True)
        else:
            print(
                "[campaign G0] FAIL diagnostic raw receipt → "
                f"{receipt}\n"
                "[campaign G0] 이 receipt는 pilot/init 자격이 없고, "
                "DNH lambda 추천 후 fresh G0를 처음부터 재실행하는 데만 씁니다.",
                flush=True,
            )
    return 2 if g0_failure is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
