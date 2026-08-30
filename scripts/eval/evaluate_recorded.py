#!/usr/bin/env python3
"""독립 recorded val/test를 체크포인트의 resolved 물리 설정으로 평가한다.

예시:
  .venv/bin/python scripts/eval/evaluate_recorded.py \
    --ckpt runs/finetune_base/ckpt/best.pt \
    --manifest data/manifests/recorded_train.jsonl --split test

저장된 오디오 파일만 읽으며 오디오 장치를 열거나 실제 소리를 출력하지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.dsp.timing import PlantSettle  # noqa: E402
from deep_anc.eval.recorded import (  # noqa: E402
    evaluate_recorded_segments,
    iter_recorded_segments,
    load_and_audit_recorded_manifest,
    load_recorded_eval_context,
    resolve_feedback_delay,
    resolve_warmup_samples,
    write_recorded_metrics,
)


DEFAULT_OCTAVE_BANDS = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="평가할 PyTorch checkpoint")
    parser.add_argument(
        "--manifest",
        default="data/manifests/recorded_train.jsonl",
        help="train/val/test split과 group_id가 모두 담긴 recorded JSONL",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--out",
        default=None,
        help="산출 디렉터리(기본: checkpoint run/eval_recorded_<split>)",
    )
    parser.add_argument(
        "--allow-surrogate",
        action="store_true",
        help="surrogate checkpoint를 진단용으로만 허용(물리 성능 판정 금지)",
    )
    parser.add_argument("--max-segments-per-session", type=int, default=8)
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=None,
        help="생략 시 checkpoint data.segment_seconds 사용",
    )
    parser.add_argument(
        "--feedback-delay-samples",
        type=int,
        default=None,
        help="생략 시 checkpoint 학습 범위 중앙값 사용",
    )
    parser.add_argument(
        "--edge-trim-seconds",
        type=float,
        default=0.25,
        help="record_duct fade/초기 상태를 피할 세션 양끝 제외 길이",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=None,
        help="S(z) 적용 후 지표에서 제외(기본: checkpoint closed_loop.warmup_seconds)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--octave-bands-hz",
        nargs="+",
        type=float,
        default=list(DEFAULT_OCTAVE_BANDS),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        context = load_recorded_eval_context(
            args.ckpt,
            allow_surrogate=args.allow_surrogate,
            device=args.device,
        )
        entries = load_and_audit_recorded_manifest(args.manifest, args.split)
        model_hop = int(context.cfg["model"]["hop"])
        feedback_delay = resolve_feedback_delay(
            context.cfg["data"], args.feedback_delay_samples
        )
        # 학습이 버리는 정착 구간과 **같은 출처**에서 하한을 받는다. 두 숫자를 각자
        # 유도하면 갈라진다 — 실제로 학습 0 / 평가 12000 으로 갈라져 있었다.
        plant_settle = PlantSettle.derive(
            secondary_delay_samples=int(context.secondary_path.delay_samples),
            handoff_samples=int(context.secondary_handoff_samples),
            fir_taps=int(context.secondary_path.fir.size),
            sample_rate=int(context.sample_rate),
        )
        warmup_samples = resolve_warmup_samples(
            context.cfg["data"],
            context.sample_rate,
            args.warmup_seconds,
            min_samples=plant_settle.samples,
        )
        edge_trim_samples = int(
            round(float(args.edge_trim_seconds) * context.sample_rate)
        )
        segments = iter_recorded_segments(
            entries,
            context.cfg["data"],
            model_hop=model_hop,
            max_segments_per_session=args.max_segments_per_session,
            segment_seconds=args.segment_seconds,
            feedback_delay_samples=feedback_delay,
            edge_trim_seconds=args.edge_trim_seconds,
        )
        result = evaluate_recorded_segments(
            context.model,
            context.plant,
            segments,
            sample_rate=context.sample_rate,
            trusted_band_hz=context.trusted_band_hz,
            octave_bands_hz=args.octave_bands_hz,
            device=context.device,
            batch_size=args.batch_size,
            warmup_samples=warmup_samples,
        )
        checkpoint = Path(args.ckpt)
        out_dir = (
            Path(args.out)
            if args.out
            else checkpoint.parent.parent / f"eval_recorded_{args.split}"
        )
        markdown_path, npz_path = write_recorded_metrics(
            result,
            out_dir,
            checkpoint=checkpoint,
            manifest=args.manifest,
            split=args.split,
            context=context,
            feedback_delay_samples=feedback_delay,
            allow_surrogate=args.allow_surrogate,
            edge_trim_samples=edge_trim_samples,
            warmup_samples=warmup_samples,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 2

    trusted = result["trusted"]
    fullband = result["fullband"]
    print(
        f"recorded {args.split}: {result['n_sessions']} sessions / "
        f"{result['n_groups']} groups / {result['n_segments']} segments"
    )
    print(
        f"trusted {trusted['mean_db']:+.2f} dB "
        f"(worst10 {trusted['worst10_mean_db']:+.2f}) | "
        f"fullband {fullband['mean_db']:+.2f} dB "
        f"(worst10 {fullband['worst10_mean_db']:+.2f}) | "
        f"gap {result['gap_mean_db']:+.2f} dB"
    )
    print(f"산출물: {markdown_path}, {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
