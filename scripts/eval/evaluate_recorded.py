#!/usr/bin/env python3
"""독립 recorded val/test를 체크포인트의 resolved 물리 설정으로 평가한다.

진단용 val 예시(test를 열지 않음):
  .venv/bin/python scripts/eval/evaluate_recorded.py \
    --ckpt runs/finetune_base/ckpt/best.pt \
    --manifest data/manifests/recorded_regrouped.jsonl --split val

공식 test는 고정된 selection과 single-use capability/consumed marker가 모두
필요하다. 제어된 test 명령은 ``run_finetune_pipeline.py evaluate-test``가
출력한 값을 그대로 사용한다.

저장된 오디오 파일만 읽으며 오디오 장치를 열거나 실제 소리를 출력하지 않는다.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.dsp.timing import PlantSettle  # noqa: E402
from deep_anc.eval.recorded_sampling import (  # noqa: E402
    CANONICAL_EDGE_TRIM_SECONDS,
    CANONICAL_MAX_SEGMENTS_PER_SESSION,
    CANONICAL_SEGMENT_SECONDS,
)
from deep_anc.eval.recorded import (  # noqa: E402
    evaluate_recorded_segments,
    iter_recorded_segments,
    load_and_audit_recorded_manifest,
    load_recorded_eval_context,
    resolve_feedback_delay,
    resolve_warmup_samples,
    write_recorded_metrics,
)
from deep_anc.train.evaluation_contract import (  # noqa: E402
    CAPABILITY_ENV,
    complete_test_evaluation,
    consume_test_capability,
    fail_test_evaluation,
    publish_directory_noreplace,
    snapshot_regular_file,
)


# Recorded G4의 중심주파수는 손실/게이트의 단일 출처를 그대로 쓴다. 이곳에
# 별도 tuple을 두면 CLI 기본값만 구형 7-center로 남아 canonical artifact를
# 조용히 만들 수 있다.
from deep_anc.dsp.do_no_harm import OCTAVE_BAND_CENTERS_HZ  # noqa: E402

DEFAULT_OCTAVE_BANDS = OCTAVE_BAND_CENTERS_HZ


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
    parser.add_argument(
        "--max-segments-per-session",
        type=int,
        default=CANONICAL_MAX_SEGMENTS_PER_SESSION,
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=None,
        help="생략 시 checkpoint data.segment_seconds 사용",
    )
    parser.add_argument(
        "--diagnostic-sampling-override",
        action="store_true",
        help=(
            "진단 전용: canonical 64 segment/1.5초/edge 0.25초 모집단 외 설정 허용. "
            "이 결과는 val 선택이나 test capability에 사용할 수 없음"
        ),
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
    parser.add_argument("--selection", default=None)
    parser.add_argument("--test-capability", default=None)
    parser.add_argument("--test-consumed-marker", default=None)
    parser.add_argument(
        "--allow-legacy-recorded-timeline",
        action="store_true",
        help="진단 전용: source.wav+상수 lead 구형 평가 허용(공식 test 금지)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    consumed = None
    staging_dir: Path | None = None
    terminal_rejection: Path | None = None
    try:
        requested_segment_seconds = (
            CANONICAL_SEGMENT_SECONDS
            if args.segment_seconds is None
            else float(args.segment_seconds)
        )
        canonical_edge_trim = float(CANONICAL_EDGE_TRIM_SECONDS)
        sampling_matches_canonical = (
            int(args.max_segments_per_session)
            == CANONICAL_MAX_SEGMENTS_PER_SESSION
            and requested_segment_seconds == CANONICAL_SEGMENT_SECONDS
            and float(args.edge_trim_seconds) == canonical_edge_trim
        )
        if not sampling_matches_canonical and not args.diagnostic_sampling_override:
            raise ValueError(
                "canonical recorded 평가는 max-segments=64, segment=1.5초, "
                "edge-trim=0.25초가 필요합니다. 축소 모집단은 "
                "--diagnostic-sampling-override를 명시해야 하며 성능 판정에 쓸 수 없습니다"
            )
        if (
            not args.diagnostic_sampling_override
            and (
                args.feedback_delay_samples is not None
                or args.warmup_seconds is not None
            )
        ):
            raise ValueError(
                "canonical recorded 평가는 checkpoint 기본 feedback 중앙값과 "
                "warmup/PlantSettle만 사용합니다. --feedback-delay-samples 또는 "
                "--warmup-seconds override는 --diagnostic-sampling-override에서만 "
                "허용되며 canonical val/test 근거가 될 수 없습니다"
            )
        if args.split == "test" and args.diagnostic_sampling_override:
            raise ValueError(
                "recorded test capability에서는 diagnostic sampling override를 허용하지 않습니다"
            )
        checkpoint = Path(args.ckpt)
        out_dir = (
            Path(args.out)
            if args.out
            else checkpoint.parent.parent / f"eval_recorded_{args.split}"
        )
        if out_dir.exists():
            raise FileExistsError(f"평가 산출 디렉터리는 덮어쓸 수 없습니다: {out_dir}")
        if args.split == "test":
            if args.allow_legacy_recorded_timeline:
                raise ValueError(
                    "공식 recorded test에서는 legacy source.wav/상수 lead를 허용하지 않습니다"
                )
            if not all(
                (args.selection, args.test_capability, args.test_consumed_marker)
            ):
                raise ValueError(
                    "recorded test는 selection+single-use capability+consumed marker가 "
                    "모두 필요합니다"
                )
            checkpoint_snapshot, manifest_snapshot, consumed = consume_test_capability(
                selection_path=args.selection,
                capability_path=args.test_capability,
                consumed_marker_path=args.test_consumed_marker,
                token=os.environ.get(CAPABILITY_ENV, ""),
                checkpoint_path=checkpoint,
                manifest_path=args.manifest,
            )
        else:
            checkpoint_snapshot = snapshot_regular_file(checkpoint)
            manifest_snapshot = snapshot_regular_file(args.manifest)
            consumed = None
        context = load_recorded_eval_context(
            args.ckpt,
            allow_surrogate=args.allow_surrogate,
            device=args.device,
            checkpoint_bytes=checkpoint_snapshot.content,
            checkpoint_sha256=checkpoint_snapshot.sha256,
        )
        if consumed is not None and consumed.get(
            "experiment_contract_sha256"
        ) != context.cfg.get("experiment_contract_sha256"):
            raise ValueError("test capability와 checkpoint embedded contract가 다릅니다")
        consumed_marker_sha = (
            snapshot_regular_file(args.test_consumed_marker).sha256
            if consumed is not None
            else ""
        )
        entries = load_and_audit_recorded_manifest(
            args.manifest, args.split, manifest_bytes=manifest_snapshot.content
        )
        model_hop = int(context.cfg["model"]["hop"])
        resolved_segment_seconds = (
            float(context.cfg["data"]["segment_seconds"])
            if args.segment_seconds is None
            else float(args.segment_seconds)
        )
        if (
            not args.diagnostic_sampling_override
            and resolved_segment_seconds != CANONICAL_SEGMENT_SECONDS
        ):
            raise ValueError(
                "canonical checkpoint data.segment_seconds가 1.5초와 다릅니다"
            )
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
            allow_legacy_source_timeline=args.allow_legacy_recorded_timeline,
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
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_name = tempfile.mkdtemp(
            prefix=f".{out_dir.name}.", suffix=".staging", dir=out_dir.parent
        )
        staging_dir = Path(temporary_name)
        staging_dir.rmdir()
        staged_markdown, staged_npz = write_recorded_metrics(
            result,
            staging_dir,
            checkpoint=checkpoint,
            manifest=args.manifest,
            split=args.split,
            context=context,
            feedback_delay_samples=feedback_delay,
            allow_surrogate=args.allow_surrogate,
            model_hop=model_hop,
            max_segments_per_session=args.max_segments_per_session,
            segment_seconds=resolved_segment_seconds,
            canonical_sampling=(
                not args.diagnostic_sampling_override
                and not args.allow_legacy_recorded_timeline
            ),
            edge_trim_samples=edge_trim_samples,
            warmup_samples=warmup_samples,
            checkpoint_sha256=checkpoint_snapshot.sha256,
            manifest_sha256=manifest_snapshot.sha256,
            experiment_contract_sha256=str(
                context.cfg.get("experiment_contract_sha256", "")
            ),
            selection_sha256=(
                str(consumed.get("selection_sha256", "")) if consumed else ""
            ),
            test_capability_sha256=(
                str(consumed.get("capability_sha256", "")) if consumed else ""
            ),
            test_consumed_marker_sha256=consumed_marker_sha,
            exclusive=True,
        )
        publish_directory_noreplace(staging_dir, out_dir)
        staging_dir = None
        markdown_path = out_dir / staged_markdown.name
        npz_path = out_dir / staged_npz.name
        if consumed is not None:
            terminal_marker = complete_test_evaluation(
                selection_path=args.selection,
                capability_path=args.test_capability,
                consumed_marker_path=args.test_consumed_marker,
                output_dir=out_dir,
            )
            if terminal_marker.name != "completed.json":
                terminal_rejection = terminal_marker
    except (FileExistsError, FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        if consumed is not None:
            try:
                fail_test_evaluation(
                    selection_path=args.selection,
                    capability_path=args.test_capability,
                    consumed_marker_path=args.test_consumed_marker,
                    error_type=type(exc).__name__,
                )
            except Exception as ledger_exc:
                print(f"[오류] test failed ledger 기록 실패: {ledger_exc}", file=sys.stderr)
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir)
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
    if terminal_rejection is not None:
        print(
            "[거부] valid recorded test raw는 보존했지만 G4가 PASS가 아니므로 "
            f"completion/deployment를 열지 않았습니다: {terminal_rejection}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
