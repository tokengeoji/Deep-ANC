#!/usr/bin/env python3
"""노이즈 풀 인덱싱 → JSONL manifest 생성 (파일 단위 train/val/test 분할).

  .venv/bin/python scripts/data/prepare_noise_pool.py
리샘플/정규화는 학습 로더(NoisePool)가 실시간으로 수행하므로 여기서는 인덱스만 만든다.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT                         # noqa: E402
from deep_anc.data.manifest import (                          # noqa: E402
    assign_splits,
    scan_wavs,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/noise")
    parser.add_argument("--out", default="data/manifests")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--holdout",
        default="data/manifests/recorded_holdout.json",
        help=(
            "실측 재생에 쓴 원본 목록(JSON). 여기 있는 클립은 합성 manifest 에서 "
            "**구성 단계에서** 제외한다 (D1 코퍼스 누수). 파일이 없으면 경고만 하고 진행"
        ),
    )
    parser.add_argument(
        "--allow-corpus-leak",
        action="store_true",
        help="held-out 제외를 끈다. 진단 전용이며 학습 manifest 를 만들 때 쓰면 안 된다",
    )
    args = parser.parse_args()

    root = REPO_ROOT / args.root
    out_dir = REPO_ROOT / args.out

    if not root.exists():
        print(f"소스 루트 없음: {root}")
        return 1

    # ---- 실측과 겹치는 원본 제외 (D1) -------------------------------------------
    # 같은 오디오가 두 브랜치에 동시에 들어가면 모델은 같은 입력에 **상충하는 정답**을
    # 받는다. 합성은 이상적 P/S 라 −18 dB 까지 가능하고 실측은 실제 플랜트라 천장이
    # 훨씬 낮다. 실측(2026-08-05): 이 저장소의 data/raw 기준으로 실측 4계열 691 클립이
    # **전부** 합성 태그 디렉터리 안에 있다 (music 60 → raw/music, speech 218 →
    # raw/speech, machine 188 + environment 225 → raw/noise/esc50).
    # 사후 검사(check_corpus_disjoint)만으로는 부족하다 — 구성 단계에서 빼야 한다.
    holdout: set[str] = set()
    if not args.allow_corpus_leak and args.holdout:
        holdout_path = REPO_ROOT / args.holdout
        if holdout_path.is_file():
            payload = json.loads(holdout_path.read_text(encoding="utf-8"))
            for values in (payload.get("families") or {}).values():
                holdout.update(str(item).rsplit("/", 1)[-1].casefold() for item in values)
            print(f"held-out 클립 {len(holdout)}개 제외 ({holdout_path})")
        else:
            print(
                f"[경고] held-out 목록이 없습니다: {holdout_path} — "
                "scripts/data/make_recorded_holdout.py 를 먼저 도세요"
            )

    # data/raw/noise/ 아래의 모든 하위 폴더를 태그로 자동 인식 —
    # 새 데이터셋은 폴더만 추가하면 되고, data_sim.yaml source_mix_ratio 에 같은
    # 이름의 키를 넣으면 학습에 반영된다 (speech/music 등).
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    for src in subdirs:
        tag = src.name
        entries = scan_wavs(src, tag)
        if not entries:
            print(f"[skip] {src} 에 오디오 없음")
            continue
        dropped = 0
        if holdout:
            kept = [
                entry
                for entry in entries
                if str(entry["path"]).replace("\\", "/").rsplit("/", 1)[-1].casefold()
                not in holdout
            ]
            dropped = len(entries) - len(kept)
            entries = kept
            if not entries:
                print(f"[skip] {src}: held-out 제외 후 남은 파일이 없습니다")
                continue
        entries = assign_splits(entries, {"train": 0.9, "val": 0.05}, seed=args.seed)
        out = out_dir / f"{tag}.jsonl"
        write_manifest(entries, out)
        n_train = sum(1 for e in entries if e["split"] == "train")
        total_h = sum(e["duration_s"] for e in entries) / 3600.0
        suffix = f", held-out 제외 {dropped}" if dropped else ""
        print(
            f"{tag}: {len(entries)}개 파일 ({total_h:.1f}h), train {n_train}{suffix} → {out}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
