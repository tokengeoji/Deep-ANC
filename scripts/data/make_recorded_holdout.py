#!/usr/bin/env python3
"""실측 source CSV에서 held-out 후보를 보는 **진단 전용** 도구.

  .venv/bin/python scripts/data/make_recorded_holdout.py --diagnostic-only

학습이 소비하는 canonical ``data/manifests/recorded_holdout.json``은 이 도구로 만들지
않는다. 그 파일에는 historical builder/PCM 재현, active 82세션, source CSV SHA와
provenance report 교차검증이 필요하다. 권위 복구는 다음 명령 하나뿐이다::

  .venv/bin/python scripts/data/repair_source_pool_provenance.py \
      --repair-csv --write-active-holdout --require-downstream-gates

왜 필요한가 (D1 — 코퍼스 누수)
-----------------------------
실측이 재생한 music 원본 60 트랙이 **전부** 합성 학습 풀에도 들어 있고, 그중 55개는
합성 *train* split 이다. 같은 오디오에 **상충하는 정답**이 주어진다 — 합성 브랜치는
이상적 P/S 라 −18 dB 까지 상쇄 가능하고, 실측 브랜치는 정렬 붕괴로 천장이 −0.4 dB
였다. 모델이 같은 음악에서 반대 방향 gradient 를 받는다. 그리고 **music 만 이 조건에
있었고, music 만 개선되지 않았다**(+0.09 dB, 나머지 −0.85 ~ −2.05 dB).

왜 (a) held-out 목록인가, (b) music 태그 재분배가 아니라
-------------------------------------------------------
겹치는 60 트랙은 ``data/raw/music/fma_small`` 8000 트랙의 **0.75%** 다. music 태그는
전체 혼합의 10% 이므로 제외 비용은 학습 데이터의 **0.075%** — 사실상 0 이다.
반면 (b) 태그 재분배는 실제 결함(같은 파일이 두 브랜치에 있음)과 무관하게 학습
분포를 바꾼다. 원인이 아닌 것을 바꿔서 증상을 덮는 쪽이고, 나중에 실측 계열이
늘어나면 또 재분배해야 한다. **정확히 겹치는 것만 빼는 (a)가 원인 수정이다.**

이 진단 출력은 ``results/diagnostics`` 밖으로 쓸 수 없고 학습 도구가 소비하지 않는다.
CSV만 읽어 만든 목록을 canonical provenance로 승격시키는 우회 경로를 없애기 위함이다.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT  # noqa: E402


def clip_key(value: str) -> str:
    """경로 표기가 달라도 같은 원본이면 같은 키.

    합성 매니페스트는 ``data/raw/music/fma_small/033/033012.mp3`` 를, 실측 sources.csv
    는 ``033012.mp3`` 를 들고 있다. 정규화하지 않고 비교하면 교집합이 **영원히 비어
    보인다** — 게이트가 있는데 아무것도 못 잡는 가장 흔한 실패 방식이다.
    """

    return str(value).strip().replace("\\", "/").rsplit("/", 1)[-1].casefold()


def collect(sources_csv: Path) -> dict[str, list[str]]:
    families: dict[str, set[str]] = {}
    with sources_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            family = str(row.get("source_family", "")).strip()
            raw = row.get("clips") or "[]"
            try:
                clips = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{sources_csv}: clips 열을 읽을 수 없습니다: {exc}") from exc
            # 잘린 기록으로 holdout 을 만들면 **빼야 할 클립을 빼지 않는다.**
            # 2026-08-07: used[:12] 절단으로 256 placement 가 누락돼 있었다.
            declared = row.get("clip_count")
            if declared not in (None, "") and int(declared) > len(clips):
                raise ValueError(
                    f"{sources_csv}: 재생 기록이 잘려 있습니다 — "
                    f"{row.get('path', '?')} clip_count={int(declared)} vs "
                    f"clips {len(clips)}개. holdout 이 불완전해집니다"
                )
            families.setdefault(family, set()).update(clip_key(item) for item in clips)
    return {family: sorted(values) for family, values in sorted(families.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="비권위 후보 목록임을 명시한다. 없으면 아무 파일도 쓰지 않고 실패",
    )
    parser.add_argument(
        "--sources",
        action="append",
        default=None,
        help=(
            "실측 재생에 쓴 소스 목록(반복 지정 가능). 기본은 v2. 두 풀이 섞인 "
            "재녹음이라면 **양쪽을 다 지정해야** held-out 이 실제 재생분을 덮는다"
        ),
    )
    parser.add_argument(
        "--out",
        default="results/diagnostics/recorded_holdout.diagnostic.json",
    )
    parser.add_argument(
        "--synthetic-root",
        default="data/raw",
        help="겹침 비율을 실제로 세기 위한 합성 풀 루트 (있으면)",
    )
    args = parser.parse_args()

    if not args.diagnostic_only:
        print(
            "[중단] 이 도구는 canonical holdout을 만들 수 없습니다. 권위 복구는 "
            "repair_source_pool_provenance.py --repair-csv --write-active-holdout을 "
            "사용하세요. 후보만 보려면 --diagnostic-only를 명시하세요.",
            file=sys.stderr,
        )
        return 2
    out_path = (REPO_ROOT / args.out).resolve()
    diagnostic_root = (REPO_ROOT / "results/diagnostics").resolve()
    try:
        out_path.relative_to(diagnostic_root)
    except ValueError:
        print("[중단] 진단 출력은 results/diagnostics 아래만 허용됩니다", file=sys.stderr)
        return 2
    if not out_path.name.endswith(".diagnostic.json"):
        print("[중단] 진단 출력 파일명은 .diagnostic.json으로 끝나야 합니다", file=sys.stderr)
        return 2

    # 세션이 **실제로 재생한** 풀을 단일 출처로 본다. 설정을 v1 로 둔 채 v2 로 녹음하면
    # held-out 이 실제 재생분을 못 덮고, 누수 게이트가 PASS 하면서 100% 누수를 통과시킨다
    # (2026-08-06 재현됨). 인자로 명시하면 그 값이 우선한다.
    if args.sources:
        selected = [str(value) for value in args.sources]
    else:
        from deep_anc.train.finetune_readiness import observed_source_pools

        observed = observed_source_pools(REPO_ROOT / "data" / "recorded")
        selected = sorted(observed) or ["data/source_pool_v2/sources.csv"]
        if observed:
            print(
                "실측 세션이 재생한 풀에서 유도: "
                + ", ".join(f"{k} ({v}세션)" for k, v in sorted(observed.items()))
            )

    csv_paths = [REPO_ROOT / value for value in selected]
    missing = [str(p) for p in csv_paths if not p.is_file()]
    if missing:
        print(f"sources.csv 가 없습니다: {', '.join(missing)}", file=sys.stderr)
        return 1
    families: dict[str, list[str]] = {}
    for path in csv_paths:
        for family, clips in collect(path).items():
            families.setdefault(family, [])
            families[family] = sorted(set(families[family]) | set(clips))
    sources_csv = csv_paths[0]

    overlap: dict[str, dict] = {}
    root = REPO_ROOT / args.synthetic_root
    if root.is_dir():
        available: set[str] = set()
        for path in root.rglob("*"):
            if path.is_file():
                available.add(clip_key(path.name))
        for family, clips in families.items():
            hits = sorted(set(clips) & available)
            overlap[family] = {
                "recorded_clips": len(clips),
                "present_in_synthetic_pool": len(hits),
                "ratio": len(hits) / max(1, len(clips)),
            }

    payload = {
        "authoritative": False,
        "purpose": "CSV 기반 held-out 후보 진단. 학습 manifest 구성에 사용 금지.",
        "sources_csv": [str(p.relative_to(REPO_ROOT)) for p in csv_paths],
        "families": families,
        "total_clips": sum(len(values) for values in families.values()),
        "overlap_with_synthetic_pool": overlap,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"held-out 목록: {out_path}")
    for family, clips in families.items():
        item = overlap.get(family, {})
        print(
            f"  {family:<12} 실측 클립 {len(clips):4d}"
            + (
                f"  합성 풀 교집합 {item['present_in_synthetic_pool']:4d} "
                f"({100.0 * item['ratio']:.0f}%)"
                if item
                else ""
            )
        )
    print(f"  합계 {payload['total_clips']} 클립")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
