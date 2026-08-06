#!/usr/bin/env python3
"""소스 풀을 순서대로 재생하며 recorded 세션을 **연속으로** 수집한다.

⚠ 스피커에서 계속 소리가 난다. 사용자 입회 하에만 실행한다.

왜 배치인가
-----------
파인튜닝 게이트는 80세션 **그리고** 90분을 요구한다. 세션 하나를 손으로 돌리면 80번을
사람이 지켜봐야 하고, 중간에 하나가 실패하면 어디까지 됐는지 알 수 없다. 이 스크립트는

* ``data/source_pool/sources.csv`` 를 순서대로 소비하고,
* 이미 녹음된 소스는 건너뛰며(재개 가능),
* **세션마다 즉시 QA** 를 돌려 클리핑/무신호를 그 자리에서 잡고,
* 진행 상황을 ``batch_progress.csv`` 에 한 행씩 남긴다.

즉시 QA 가 핵심이다. 2시간을 다 돌린 뒤에 "전부 클리핑이었다"를 알면 2시간을 다시 써야 한다.

    .venv/bin/python scripts/data/record_session_batch.py --confirm-speaker
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.eval.artifacts import write_csv  # noqa: E402

# 마이크 입력의 허용 범위. 상한은 클리핑 직전, 하한은 "스피커가 실제로 울렸는가".
MAX_CLIP_RATIO = 0.0
MIN_ERR_RMS_DBFS = -60.0
MAX_ERR_PEAK = 0.95


def analyse_session(session_dir: Path) -> dict:
    import soundfile as sf

    mics, rate = sf.read(str(session_dir / "mics.wav"), dtype="float32", always_2d=True)
    err = mics[:, 0].astype(np.float64)
    ref = mics[:, 1].astype(np.float64) if mics.shape[1] > 1 else np.zeros_like(err)

    def dbfs(values: np.ndarray) -> float:
        power = float(np.mean(values**2))
        return 10.0 * np.log10(power + 1e-30)

    return {
        "sample_rate_hz": int(rate),
        "seconds": mics.shape[0] / float(rate),
        "err_rms_dbfs": dbfs(err),
        "ref_rms_dbfs": dbfs(ref),
        "err_peak": float(np.max(np.abs(err))),
        "ref_peak": float(np.max(np.abs(ref))),
        "clip_ratio": float(np.mean(np.abs(mics) >= 0.999)),
    }


def qa_verdict(stats: dict) -> tuple[bool, str]:
    if stats["clip_ratio"] > MAX_CLIP_RATIO:
        return False, f"입력 클리핑 {stats['clip_ratio']:.4%}"
    if stats["err_peak"] > MAX_ERR_PEAK:
        return False, f"ERR peak {stats['err_peak']:.3f} 가 한계에 근접"
    if stats["err_rms_dbfs"] < MIN_ERR_RMS_DBFS:
        return False, f"ERR RMS {stats['err_rms_dbfs']:.1f} dBFS — 스피커가 울리지 않았을 수 있음"
    return True, "ok"


def already_recorded(out_root: Path) -> set[str]:
    """이미 수집된 세션의 소스 경로 집합. 재개 시 중복 녹음을 막는다."""

    done: set[str] = set()
    for meta_path in sorted(out_root.glob("*/session.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        source = (meta.get("program") or {}).get("file")
        if source:
            done.add(str(source))
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        default="data/source_pool_v2/sources.csv",
        help=(
            "재생할 소스 목록. **v2 가 기본이다** — v1 은 machine 이 8 그룹뿐이라 "
            "분할 하한(계열별 9 = val 4 + test 4 + train 1)을 만족할 수 없고, "
            "그 풀로 녹음하면 make_recorded_manifest 가 EXIT=2 로 거부한다"
        ),
    )
    parser.add_argument("--out-root", default="data/recorded")
    parser.add_argument("--amplitude", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None, help="이번 실행에서 녹음할 최대 세션 수")
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="일시적 xrun 재시도를 끈다(진단용). 기본은 세션당 1회 재시도",
    )
    parser.add_argument(
        "--confirm-speaker",
        action="store_true",
        help="사용자 입회 확인. 없으면 실행하지 않는다",
    )
    args = parser.parse_args(argv)

    if not args.confirm_speaker:
        print(
            "[중단] 스피커에서 장시간 소리가 재생됩니다. --confirm-speaker 로 확인하세요.",
            file=sys.stderr,
        )
        return 2

    sources_path = REPO_ROOT / args.sources
    if not sources_path.exists():
        print(f"[중단] 소스 목록이 없습니다: {sources_path}", file=sys.stderr)
        print("  먼저: .venv/bin/python scripts/data/build_recording_sources.py", file=sys.stderr)
        return 2

    entries = list(csv.DictReader(sources_path.open(encoding="utf-8")))
    if args.families:
        entries = [e for e in entries if e["source_family"] in set(args.families)]
    out_root = REPO_ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    done = already_recorded(out_root)

    pending = [e for e in entries if e["path"] not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        print("모든 소스가 이미 녹음되었습니다.")
        return 0

    planned_seconds = sum(float(e["seconds"]) for e in pending)
    print(f"소스 {len(entries)}개 중 {len(done)}개 완료 · 이번 실행 {len(pending)}개")
    print(f"예상 재생 {planned_seconds / 60.0:.1f}분 (+ 세션당 준비 시간)")
    print(f"소스 진폭 {args.amplitude} · 출력 {out_root}\n")

    progress_path = out_root / "batch_progress.csv"
    rows: list[dict] = []
    if progress_path.exists():
        rows = list(csv.DictReader(progress_path.open(encoding="utf-8")))

    consecutive_failures = 0
    started = time.monotonic()
    attempt_retry = not args.no_retry

    for index, entry in enumerate(pending, start=1):
        before = {p.name for p in out_root.iterdir() if p.is_dir()}
        command = [
            ".venv/bin/python", "scripts/data/record_duct.py",
            "--program", "file",
            "--file", entry["path"],
            "--seconds", str(float(entry["seconds"])),
            "--amplitude", str(args.amplitude),
            "--source-family", entry["source_family"],
            "--group-id", entry["group_id"],
            "--out-root", args.out_root,
        ]
        elapsed = time.monotonic() - started
        print(
            f"[{index}/{len(pending)}] {entry['source_family']:11s} "
            f"{Path(entry['path']).name}  (경과 {elapsed / 60.0:.1f}분)"
        )
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
        created = sorted(
            {p.name for p in out_root.iterdir() if p.is_dir()} - before
        )

        # xrun 은 일시적이다 — 특히 **배치의 첫 세션**은 full-duplex 스트림을 처음 여는
        # 순간이라 거의 항상 한 번 걸린다(실측: music_000 이 배치 선두일 때 4회 연속 실패,
        # 같은 파일이 배치 중간에 있을 때는 통과). 한 번은 다시 시도한다. 두 번 연속
        # 실패하면 일시적 문제가 아니므로 그대로 기록한다.
        if (result.returncode != 0 or not created) and attempt_retry:
            print("    [재시도] 일시적 xrun 으로 보입니다")
            time.sleep(1.0)
            before = {p.name for p in out_root.iterdir() if p.is_dir()}
            result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
            created = sorted({p.name for p in out_root.iterdir() if p.is_dir()} - before)

        row = {
            "source_family": entry["source_family"],
            "group_id": entry["group_id"],
            "source_path": entry["path"],
            "returncode": result.returncode,
            "session_id": created[0] if created else "",
        }
        if result.returncode != 0 or not created:
            row["verdict"] = "record_failed"
            row["detail"] = (result.stderr or result.stdout).strip().splitlines()[-1:] or [""]
            row["detail"] = row["detail"][0][:200]
            consecutive_failures += 1
            print(f"    [실패] {row['detail']}")
        else:
            session_dir = out_root / created[0]
            stats = analyse_session(session_dir)
            ok, reason = qa_verdict(stats)
            row.update(stats)
            row["verdict"] = "ok" if ok else "qa_failed"
            row["detail"] = reason
            if ok:
                consecutive_failures = 0
                print(
                    f"    ERR {stats['err_rms_dbfs']:6.1f} dBFS peak {stats['err_peak']:.3f} · "
                    f"REF {stats['ref_rms_dbfs']:6.1f} dBFS · clip {stats['clip_ratio']:.3%}"
                )
            else:
                consecutive_failures += 1
                print(f"    [QA 실패] {reason}")

        rows.append(row)
        write_csv(progress_path, rows)

        if consecutive_failures >= args.max_consecutive_failures:
            print(
                f"\n[중단] {consecutive_failures}회 연속 실패. 스피커/배선/볼륨을 확인하세요.",
                file=sys.stderr,
            )
            return 1
        time.sleep(args.settle_seconds)

    ok_rows = [r for r in rows if r.get("verdict") == "ok"]
    total_seconds = sum(float(r.get("seconds", 0) or 0) for r in ok_rows)
    print(f"\n완료 세션 {len(ok_rows)}개 · {total_seconds / 60.0:.1f}분")
    print(f"진행 기록: {progress_path}")
    print("다음: .venv/bin/python scripts/data/make_recorded_manifest.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
