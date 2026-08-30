#!/usr/bin/env python3
"""실측 세션 → 이식 가능한 group-aware train/val/test manifest.

  .venv/bin/python scripts/data/make_recorded_manifest.py
"""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT                          # noqa: E402
from deep_anc.data.manifest import (                          # noqa: E402
    MANIFEST_PATH_BASE,
    assign_splits,
    manifest_relative_path,
    validate_group_id,
    validate_session_id,
    validate_source_family,
    write_manifest,
)


def _load_session_metadata(session: Path) -> dict:
    metadata_path = session / "session.json"
    if not metadata_path.exists():
        return {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{metadata_path}: session.json을 읽을 수 없습니다: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path}: 최상위 값은 JSON 객체여야 합니다")
    return metadata


def _legacy_source_family(metadata: dict) -> str:
    """구형 session.json의 program.type을 안전한 family 폴백으로 사용한다."""
    program = metadata.get("program", {})
    if isinstance(program, dict):
        program_type = program.get("type")
        if isinstance(program_type, str) and program_type.strip():
            return program_type
    return "legacy"


def build_recorded_entries(
    root: str | Path,
    manifest_path: str | Path,
    *,
    seed: int = 20260803,
    ratios: dict[str, float] | None = None,
    min_groups_per_split: int | None = None,
) -> list[dict]:
    """세션을 스캔하고 group 단위 split을 붙인 매니페스트 항목을 만든다.

    구형 세션은 ``group_id=<세션 디렉터리명>``과
    ``source_family=<program.type 또는 legacy>``로 승격한다.

    ``min_groups_per_split`` 은 계열마다 val·test 가 가져야 할 **독립 그룹 하한**이고,
    기본값은 ``eval.recorded.MIN_GROUPS_PER_FAMILY`` 다 — G4 평가기가 쓰는 값과
    **같은 상수**를 읽는다. 여기서 따로 4 를 적으면 그것이 세 번째 정의가 되고,
    언젠가 한쪽만 바뀐다(이 저장소에서 반복된 발생기 A).
    """
    from deep_anc.eval.recorded import MIN_GROUPS_PER_FAMILY

    floor = int(MIN_GROUPS_PER_FAMILY if min_groups_per_split is None else min_groups_per_split)
    if floor < MIN_GROUPS_PER_FAMILY:
        raise ValueError(
            f"min_groups_per_split 는 G4 하한 {MIN_GROUPS_PER_FAMILY} 아래로 내릴 수 없습니다: "
            f"{floor}. 강화 방향(더 큰 값)만 허용합니다 — 게이트가 요구하는 그룹 수를 "
            "manifest 쪽에서 낮추면 학습이 끝난 뒤 G4 판정 불가로만 드러납니다."
        )
    min_groups_per_split = floor
    root = Path(root)
    manifest_path = Path(manifest_path)
    split_ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    entries: list[dict] = []

    for session in sorted(root.iterdir()) if root.exists() else []:
        if not session.is_dir():
            continue
        mics = session / "mics.wav"
        if not mics.exists():
            continue

        metadata = _load_session_metadata(session)
        inferred_metadata: list[str] = []
        if "session_id" not in metadata:
            inferred_metadata.append("session_id")
        if "group_id" not in metadata:
            inferred_metadata.append("group_id")
        if "source_family" not in metadata:
            inferred_metadata.append("source_family")
        session_id = validate_session_id(metadata.get("session_id", session.name))
        group_id = validate_group_id(metadata.get("group_id", session.name))
        source_family = validate_source_family(
            metadata.get("source_family", _legacy_source_family(metadata))
        )
        try:
            info = sf.info(str(mics))
        except RuntimeError as exc:
            raise ValueError(f"{mics}: 오디오 헤더를 읽을 수 없습니다: {exc}") from exc
        if int(info.channels) < 2:
            raise ValueError(f"{mics}: err/ref 2채널이 필요하지만 {info.channels}채널입니다")
        if int(info.samplerate) <= 0 or int(info.frames) <= 0:
            raise ValueError(f"{mics}: 비어 있거나 잘못된 오디오입니다")

        entries.append(
            {
                "path": manifest_relative_path(session, manifest_path),
                "path_base": MANIFEST_PATH_BASE,
                "duration_s": float(info.frames) / float(info.samplerate),
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "tag": "recorded",
                "session_id": session_id,
                "group_id": group_id,
                "source_family": source_family,
                "metadata_inferred": inferred_metadata,
            }
        )

    return assign_splits(
        entries,
        split_ratios,
        seed=seed,
        group_key="group_id",
        stratify_key="source_family",
        # 비율이 0 인 split 에는 하한을 걸지 않는다. 우회로가 아니다 —
        # val_ratio=0 으로 하한을 피하면 val 그룹이 0 이 되고, 그러면
        # recorded_statistical_power 게이트가 그 자리에서 FAIL 한다.
        min_units_per_split={
            split: min_groups_per_split
            for split in ("val", "test")
            if split_ratios.get(split, 0.0) > 0.0
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/recorded")
    parser.add_argument("--out", default="data/manifests/recorded_train.jsonl")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--min-groups-per-split",
        type=int,
        default=None,
        help=(
            "계열마다 val·test 가 가져야 할 독립 그룹 하한. 기본값은 G4 평가기의 "
            "MIN_GROUPS_PER_FAMILY 를 그대로 읽는다. 강화(더 큰 값)만 허용한다."
        ),
    )
    args = parser.parse_args()

    root_arg = Path(args.root)
    out_arg = Path(args.out)
    root = root_arg if root_arg.is_absolute() else REPO_ROOT / root_arg
    out = out_arg if out_arg.is_absolute() else REPO_ROOT / out_arg
    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    try:
        entries = build_recorded_entries(
            root,
            out,
            seed=args.seed,
            ratios=ratios,
            min_groups_per_split=args.min_groups_per_split,
        )
    except ValueError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 2
    if not entries:
        print(f"세션 없음: {root} — 먼저 record_duct.py 로 수집하세요")
        return 1

    write_manifest(entries, out)
    total_min = sum(e["duration_s"] for e in entries) / 60.0
    groups = {e["group_id"] for e in entries}
    families = sorted({e["source_family"] for e in entries})
    counts = {
        split: sum(1 for entry in entries if entry["split"] == split)
        for split in ("train", "val", "test")
    }
    print(
        f"{len(entries)}개 세션/{len(groups)}개 그룹 ({total_min:.1f}분), "
        f"train/val/test={counts['train']}/{counts['val']}/{counts['test']} → {out}"
    )
    print(f"source_family: {', '.join(families)}")
    # 계열×split 의 **그룹 수**를 그대로 보여준다 — 게이트가 보는 축이 세션 수가 아니라
    # 그룹 수이므로, 세션 수만 출력하면 통과 여부를 화면에서 판단할 수 없다.
    from deep_anc.eval.recorded import MIN_GROUPS_PER_FAMILY

    print(f"{'계열':<14}{'train':>7}{'val':>6}{'test':>6}   (그룹 수, val·test 하한 {MIN_GROUPS_PER_FAMILY})")
    for family in families:
        cells = []
        for split in ("train", "val", "test"):
            cells.append(
                len({
                    entry["group_id"]
                    for entry in entries
                    if entry["source_family"] == family and entry["split"] == split
                })
            )
        print(f"{family:<14}{cells[0]:>7}{cells[1]:>6}{cells[2]:>6}")
    inferred = [entry for entry in entries if entry["metadata_inferred"]]
    if inferred:
        print(
            f"[경고] legacy 메타데이터를 추론한 세션 {len(inferred)}개 — "
            "manifest의 metadata_inferred 필드를 검토하세요",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
