#!/usr/bin/env python3
"""시간축이 깨진 실측 세션을 **되돌릴 수 있게** 격리한다.

  # 무엇이 격리되는지 먼저 본다 (파일을 건드리지 않음)
  .venv/bin/python scripts/data/quarantine_recorded_sessions.py --dry-run

  # 실행 (data/recorded → data/recorded_broken 로 이동 + 되돌리기 대장 기록)
  .venv/bin/python scripts/data/quarantine_recorded_sessions.py --reason "결함 2: 시간축 붕괴"

  # 되돌리기
  .venv/bin/python scripts/data/quarantine_recorded_sessions.py --restore

왜 삭제가 아니라 이동인가
------------------------
이 80 세션은 **쓸모없는 데이터가 아니라 시간축 라벨이 틀린 데이터**다. 음향 자체는
무손상이다 — ``coh²(REF→ERR) = 0.977~0.993``, REF→ERR 지연 세션간 산포 0.31 샘플.
틀린 것은 ``source.wav`` 와 ``mics.wav`` 의 인덱스 대응 하나뿐이고, REF 증인 재정렬로
``coh²(source→ERR)`` 가 0.02~0.07 → 0.87~0.96 으로 복구된다(실측 3세션).

즉 이 데이터는 **재녹음 없이 되살릴 수 있는 후보**다. 삭제하면 그 선택지가 사라지고,
재녹음 93분과 스피커 연결 시간을 되돌릴 수 없게 지불해야 한다. 반대로 그냥 두면
manifest 를 통해 학습에 다시 섞여 들어간다 — 실제로 그렇게 −0.07 dB 를 얻었다.

그래서 (a) 디렉터리를 옮겨 학습 경로에서 확실히 빼고, (b) 어디서 왔는지와 왜 옮겼는지를
대장에 남겨 한 명령으로 되돌릴 수 있게 한다. manifest 는 함께 옮겨 두어 "매니페스트는
남았는데 데이터가 없다" 는 어중간한 상태를 만들지 않는다.
"""

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT  # noqa: E402

LEDGER_NAME = "quarantine_ledger.json"


def _load_ledger(target: Path) -> dict:
    path = target / LEDGER_NAME
    if not path.is_file():
        return {"entries": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or "entries" not in value:
        raise ValueError(f"손상된 격리 대장: {path}")
    return value


def _save_ledger(target: Path, ledger: dict) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / LEDGER_NAME).write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/recorded")
    parser.add_argument("--target", default="data/recorded_broken")
    parser.add_argument(
        "--manifest-dir",
        default="data/manifests",
        help="함께 옮길 recorded manifest 가 있는 디렉터리",
    )
    parser.add_argument("--reason", default="결함 2: 재생↔녹음 시간축 붕괴")
    parser.add_argument("--sessions", nargs="*", default=None, help="특정 세션만 (기본 전부)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restore", action="store_true", help="대장을 읽어 원위치로 되돌린다")
    args = parser.parse_args()

    source_root = REPO_ROOT / args.source
    target_root = REPO_ROOT / args.target

    if args.restore:
        ledger = _load_ledger(target_root)
        # --sessions 는 격리할 때뿐 아니라 **되돌릴 때도** 존중해야 한다.
        # 2026-08-06 이전에는 무시돼서 대장의 80개가 전부 돌아왔다. 재정렬에 실패한
        # 33개까지 같이 돌아오면 QA 가 그것을 다시 걸러내야 하고, 무엇보다 "무엇을
        # 되돌렸는가" 를 사람이 통제할 수 없다.
        wanted = set(args.sessions or [])
        restored = 0
        skipped_by_filter = 0
        for entry in list(ledger["entries"]):
            if wanted and entry["session_id"] not in wanted:
                skipped_by_filter += 1
                continue
            src = REPO_ROOT / entry["quarantined_to"]
            dst = REPO_ROOT / entry["original_path"]
            if not src.exists():
                print(f"[skip] 격리본이 없습니다: {src}")
                continue
            if dst.exists():
                print(f"[skip] 원위치에 이미 있습니다: {dst}")
                continue
            if args.dry_run:
                print(f"[dry-run] {src} → {dst}")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                ledger["entries"].remove(entry)
            restored += 1
        if not args.dry_run:
            _save_ledger(target_root, ledger)
        print(f"되돌린 항목: {restored}")
        if skipped_by_filter:
            print(f"--sessions 필터로 남겨 둔 항목: {skipped_by_filter}")
        return 0

    if not source_root.is_dir():
        print(f"원본 디렉터리가 없습니다: {source_root}", file=sys.stderr)
        return 1

    names = args.sessions
    if not names:
        names = sorted(
            p.name for p in source_root.iterdir() if p.is_dir() and (p / "mics.wav").is_file()
        )
    if not names:
        print("격리할 세션이 없습니다", file=sys.stderr)
        return 1

    ledger = _load_ledger(target_root)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    moved = 0
    for name in names:
        src = source_root / name
        dst = target_root / name
        if not src.is_dir():
            print(f"[skip] 없음: {src}")
            continue
        if dst.exists():
            print(f"[skip] 격리본이 이미 있습니다: {dst}")
            continue
        if args.dry_run:
            print(f"[dry-run] {src.relative_to(REPO_ROOT)} → {dst.relative_to(REPO_ROOT)}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            ledger["entries"].append(
                {
                    "session_id": name,
                    "original_path": str(src.relative_to(REPO_ROOT)),
                    "quarantined_to": str(dst.relative_to(REPO_ROOT)),
                    "reason": args.reason,
                    "quarantined_at": stamp,
                }
            )
        moved += 1

    # manifest 도 같이 옮긴다. 데이터만 옮기고 manifest 를 남기면 학습이 "경로 없음" 으로
    # 죽거나, 더 나쁘게는 일부만 읽고 조용히 계속 돈다.
    manifest_dir = REPO_ROOT / args.manifest_dir
    manifest_moved = []
    if manifest_dir.is_dir() and not args.sessions:
        for path in sorted(manifest_dir.glob("recorded_*.jsonl")):
            dst = target_root / "manifests" / path.name
            if args.dry_run:
                print(f"[dry-run] {path.relative_to(REPO_ROOT)} → {dst.relative_to(REPO_ROOT)}")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(dst))
            manifest_moved.append(path.name)

    if not args.dry_run:
        ledger.setdefault("manifests", []).extend(manifest_moved)
        ledger["reason"] = args.reason
        _save_ledger(target_root, ledger)
        readme = target_root / "README.md"
        readme.write_text(
            "# 격리된 실측 세션\n\n"
            f"- 사유: {args.reason}\n"
            f"- 시각: {stamp}\n"
            f"- 세션 수: {moved}\n\n"
            "이 세션들은 **삭제된 것이 아니다.** 음향은 무손상이고"
            "(coh²(REF→ERR)=0.977~0.993, REF→ERR 지연 세션간 산포 0.31 샘플),\n"
            "틀린 것은 source.wav 와 mics.wav 의 인덱스 대응뿐이다.\n\n"
            "복구 시도:\n\n"
            "```bash\n"
            ".venv/bin/python scripts/data/realign_recorded_sessions.py \\\n"
            f"    --root {args.target} --report results/timeline/realign.json\n"
            "```\n\n"
            "되돌리기:\n\n"
            "```bash\n"
            ".venv/bin/python scripts/data/quarantine_recorded_sessions.py --restore\n"
            "```\n",
            encoding="utf-8",
        )

    print(f"격리: {moved}개 세션, manifest {len(manifest_moved)}개 → {target_root}")
    if args.dry_run:
        print("(dry-run — 아무것도 옮기지 않았습니다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
