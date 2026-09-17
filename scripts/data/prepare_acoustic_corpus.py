#!/usr/bin/env python3
"""학습 환경에 명시 stage한 공개 음원을 검증한다(다운로드/오디오 출력 없음).

PC에 원본을 자동 내려받지 않는다. Drive inventory 준비와 로컬 data_ready는 다르다.
source root 아래 esc50/music/speech/demand/machine을 요구하며 --help는 무입출력이다.
machine(MIMII)은 항상 train-only 보조 자료다. source_index_meta.json에
usage_policy=train_only_auxiliary 선언이 필요하며 val/test 또는 독립 평가에 쓰지 않는다.
QA 실패는 inventory.jsonl와 qa.json을 보존하고 exit 1, 성공은 exit 0이다.
성공은 실제 감쇠 성능을 의미하지 않는다. 쓰기 실패 시 부분 결과 폴더가 남을 수 있다.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.data.public_corpus import prepare_public_corpus  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw/acoustic", help="명시 stage된 로컬 source root")
    parser.add_argument("--out", default="data/acoustic/manifests", help="새 결과 폴더(덮어쓰기 금지)")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--reuse", action="store_true", help="기존 결과를 전체 재검증; 수정하지 않음")
    args = parser.parse_args(argv)
    try:
        report = prepare_public_corpus(args.raw_root, args.out, seed=args.seed, reuse=args.reuse)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"준비 실패: {exc}", file=sys.stderr)
        return 1
    if not report["data_ready"]:
        print(f"QA 실패: {len(report['issues'])}개 issue. inventory/qa 보존, 학습 manifest 미발급.", file=sys.stderr)
        return 1
    print(f"로컬 PCM QA 통과: {args.out} (machine=train-only, 독립 평가 금지; 실제 ANC 성능 주장이 아님)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
