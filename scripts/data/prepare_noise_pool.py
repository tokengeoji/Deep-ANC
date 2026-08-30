#!/usr/bin/env python3
"""노이즈 풀 인덱싱 → JSONL manifest 생성 (파일 단위 train/val/test 분할).

  .venv/bin/python scripts/data/prepare_noise_pool.py
리샘플/정규화는 학습 로더(NoisePool)가 실시간으로 수행하므로 여기서는 인덱스만 만든다.

2026-08-06 수정 — **선언한 태그를 전부 만들지 못하면 실패한다.**
------------------------------------------------------------
이전 판은 ``--root data/raw/noise`` 하나만 스캔했다. 그 아래에는 ``esc50`` 밖에 없고
``music``(data/raw/music/fma_small) 과 ``speech``(data/raw/speech/LibriSpeech) 는
**다른 루트**에 있다. 그래서 ``data/manifests`` 에는 ``esc50.jsonl`` 하나만 생겼다.

그 상태가 조용한 이유가 문제였다. ``src/deep_anc/data/synth_dataset.py`` 는 manifest 가
없는 태그를 로그 한 줄 없이 **합성원으로 폴백**한다. 즉 ``source_mix_ratio`` 가
speech 0.15 / music 0.10 을 선언해도 실제로는 그 0.25 가 전부 synthetic 으로 돌아가고,
학습 기록에는 선언된 혼합비가 남는다. 선언과 실행이 갈라진 것을 아무도 못 본다.

그래서 이 스크립트는 이제
  1. ``source_mix_ratio`` (+ acoustic 판)의 **태그 목록을 단일 출처로 읽고**,
  2. ``data/raw`` 전체를 그 태그로 매칭해 스캔하며,
  3. 비율 > 0 인데 소재가 없어 manifest 를 만들지 못한 태그가 하나라도 있으면
     **경고가 아니라 종료코드 1** 로 끝난다.

이 저장소에 실제로 있는 원본: ``music/fma_small``, ``noise/esc50``,
``speech/LibriSpeech``. ``dns_fullband`` / ``demand`` / ``machine`` 은 **유실됐다** —
있는 척하지 않는다. 세 태그가 선언에 남아 있는 한 이 스크립트는 실패하는 것이 옳다.
고치는 방법은 둘뿐이다: 원본을 다시 받거나, ``configs/data_sim.yaml`` 의 혼합비에서
그 태그를 지우는 것이다(그러면 게이트도 그 태그를 요구하지 않는다).
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT, load_yaml                # noqa: E402
from deep_anc.data.manifest import (                            # noqa: E402
    assign_splits,
    scan_wavs,
    write_manifest,
)


_FROZEN = ConfigDict(frozen=True, extra="forbid")


class DeclaredPool(BaseModel):
    """혼합비가 선언한 소스 태그 하나. 태그 목록의 단일 출처는 ``data_sim.yaml`` 이다."""

    model_config = _FROZEN

    tag: str
    ratio: float
    """모든 혼합비 표에서 이 태그가 갖는 최대 비율. 0 이면 지금은 쓰이지 않는다."""

    @model_validator(mode="after")
    def _validate(self) -> "DeclaredPool":
        if not self.tag or self.tag != self.tag.strip():
            raise ValueError(f"태그가 비었거나 공백이 있습니다: {self.tag!r}")
        if any(ch in self.tag for ch in ("/", "\\", ".")):
            raise ValueError(f"태그는 파일명 조각이 아니라 이름이어야 합니다: {self.tag!r}")
        if not (0.0 <= self.ratio <= 1.0):
            raise ValueError(f"{self.tag}: 비율은 [0,1] 이어야 합니다: {self.ratio}")
        return self


class PoolPlan(BaseModel):
    """"무엇을 만들어야 하는가" 의 선언. 스캔 결과와 대조되는 기준이다."""

    model_config = _FROZEN

    pools: tuple[DeclaredPool, ...]
    roots: tuple[str, ...]

    @model_validator(mode="after")
    def _validate(self) -> "PoolPlan":
        seen = [item.tag for item in self.pools]
        if len(seen) != len(set(seen)):
            raise ValueError(f"태그가 중복됐습니다: {seen}")
        if not self.roots:
            raise ValueError("스캔할 루트가 없습니다")
        return self

    def required_tags(self) -> tuple[str, ...]:
        return tuple(item.tag for item in self.pools if item.ratio > 0.0)


def declared_pools(data_config: Path) -> tuple[DeclaredPool, ...]:
    """``source_mix_ratio`` (+ acoustic 판)에서 태그를 읽는다 — 유일한 경로.

    태그를 이 스크립트에 리터럴로 적으면 그것이 **두 번째 선언**이 되고, 설정과
    갈라진 순간 아무도 모른다(이 저장소가 반복한 발생기 A).
    """

    cfg = load_yaml(data_config)
    ratios: dict[str, float] = {}
    for key in ("source_mix_ratio", "source_mix_ratio_acoustic"):
        for tag, value in (cfg.get(key) or {}).items():
            if str(tag) == "synthetic":
                continue  # 파일 소재가 없는 즉석 생성원
            ratios[str(tag)] = max(float(value), ratios.get(str(tag), 0.0))
    return tuple(
        DeclaredPool(tag=tag, ratio=ratio) for tag, ratio in sorted(ratios.items())
    )


def discover_tag_dirs(root: Path, tags: frozenset[str]) -> dict[str, list[Path]]:
    """``root`` 아래에서 **선언된 태그 이름과 같은 디렉터리**를 찾는다.

    ``data/raw/music/fma_small`` 은 ``music`` 에서 멈추고(그 아래 전부가 music),
    ``data/raw/noise/esc50`` 은 ``noise`` 가 태그가 아니므로 한 단계 더 내려가
    ``esc50`` 에서 멈춘다. 즉 디렉터리 깊이를 스크립트가 가정하지 않는다 —
    ``--root data/raw/noise`` 하나만 보던 판이 music/speech 를 통째로 놓친 이유가
    바로 그 가정이었다.
    """

    found: dict[str, list[Path]] = {}
    if not root.is_dir():
        return found
    stack = [root]
    while stack:
        current = stack.pop()
        for child in sorted(p for p in current.iterdir() if p.is_dir()):
            if child.name in tags:
                found.setdefault(child.name, []).append(child)
            else:
                stack.append(child)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help=(
            "스캔할 원본 루트(반복 지정 가능). 기본은 data/raw 전체다 — "
            "music/noise/speech 가 서로 다른 루트에 있어서 하나만 보면 조용히 누락된다"
        ),
    )
    parser.add_argument("--out", default="data/manifests")
    parser.add_argument(
        "--data-config",
        default="configs/data_sim.yaml",
        help="source_mix_ratio 를 읽을 설정. 태그 목록의 단일 출처다",
    )
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

    plan = PoolPlan(
        pools=declared_pools(REPO_ROOT / args.data_config),
        roots=tuple(args.root or ["data/raw"]),
    )
    out_dir = REPO_ROOT / args.out
    tags = frozenset(item.tag for item in plan.pools)

    roots = [REPO_ROOT / value for value in plan.roots]
    missing_roots = [str(path) for path in roots if not path.exists()]
    if missing_roots:
        print(f"소스 루트 없음: {', '.join(missing_roots)}", file=sys.stderr)
        return 1

    tag_dirs: dict[str, list[Path]] = {}
    for root in roots:
        for tag, paths in discover_tag_dirs(root, tags).items():
            tag_dirs.setdefault(tag, []).extend(paths)

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

    written: dict[str, int] = {}
    for pool in plan.pools:
        sources = tag_dirs.get(pool.tag, [])
        entries: list[dict] = []
        for src in sources:
            entries.extend(scan_wavs(src, pool.tag))
        if not entries:
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
                print(f"[skip] {pool.tag}: held-out 제외 후 남은 파일이 없습니다")
                continue
        entries = assign_splits(entries, {"train": 0.9, "val": 0.05}, seed=args.seed)
        out = out_dir / f"{pool.tag}.jsonl"
        write_manifest(entries, out)
        written[pool.tag] = len(entries)
        n_train = sum(1 for e in entries if e["split"] == "train")
        total_h = sum(e["duration_s"] for e in entries) / 3600.0
        suffix = f", held-out 제외 {dropped}" if dropped else ""
        where = ", ".join(str(p.relative_to(REPO_ROOT)) for p in sources)
        print(
            f"{pool.tag}: {len(entries)}개 파일 ({total_h:.1f}h), train {n_train}{suffix} "
            f"← {where} → {out}"
        )

    # ---- 선언했는데 못 만든 태그 = 조용한 폴백의 씨앗 -----------------------------
    # 여기서 멈추지 않으면 synth_dataset 이 그 태그를 합성원으로 **로그 없이** 대체하고,
    # 학습은 선언한 혼합비와 다른 데이터로 돈다.
    missing = [tag for tag in plan.required_tags() if tag not in written]
    if missing:
        print("", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        print(
            "[실패] 선언된 소스 태그의 원본을 찾지 못했습니다: " + ", ".join(missing),
            file=sys.stderr,
        )
        print(
            "  스캔한 루트: " + ", ".join(str(p) for p in roots),
            file=sys.stderr,
        )
        print(
            "  이 상태로 학습하면 synth_dataset 이 없는 태그를 **조용히** 합성원으로\n"
            "  폴백하므로, 선언한 source_mix_ratio 와 다른 데이터로 돌게 됩니다.\n"
            f"  둘 중 하나를 하세요: (1) 원본을 {roots[0]} 아래 태그 이름 디렉터리로 받는다,\n"
            f"  (2) {args.data_config} 의 source_mix_ratio 에서 그 태그를 지운다.",
            file=sys.stderr,
        )
        print("=" * 78, file=sys.stderr)
        return 1

    print(f"완료: manifest {len(written)}개 ({', '.join(sorted(written))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
