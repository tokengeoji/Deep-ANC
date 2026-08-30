"""JSONL manifest — 데이터 인덱스 (경로/분할/길이/태그).

분할 규칙 (누수 방지): 노이즈 풀은 원본 파일 단위, 실측 데이터는 세션 단위로
train/val/test 를 나눈다. tests/test_dataset.py 가 교차 누수를 검사한다.

새 매니페스트의 이식 가능한 경로는 ``path_base: "manifest"``를 명시한다.
표시가 없는 경로는 기존 매니페스트 호환을 위해 해석하지 않고 그대로 반환한다.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path, PureWindowsPath
from typing import Any

import soundfile as sf

VALID_SPLITS = ("train", "val", "test")
SUPPORTED_AUDIO_EXTENSIONS = frozenset({".wav", ".flac", ".mp3"})
MANIFEST_PATH_BASE = "manifest"


def _validate_metadata_id(field: str, value: Any) -> str:
    """그룹/소스 식별자를 경로가 아닌 짧은 메타데이터 값으로 검증한다."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}는 비어 있지 않은 문자열이어야 합니다: {value!r}")
    if value != value.strip():
        raise ValueError(f"{field} 앞뒤 공백은 허용하지 않습니다: {value!r}")
    if len(value) > 128:
        raise ValueError(f"{field}는 128자를 넘을 수 없습니다: {value!r}")
    if any(ch in value for ch in ("/", "\\", "\0", "\n", "\r")):
        raise ValueError(f"{field}에 경로 구분자/제어문자를 사용할 수 없습니다: {value!r}")
    return value


def validate_group_id(value: Any) -> str:
    """실측 세션의 상관 그룹 ID를 검증하고 문자열로 반환한다."""
    return _validate_metadata_id("group_id", value)


def validate_source_family(value: Any) -> str:
    """speech/music/environment 같은 소스 계열 값을 검증한다."""
    return _validate_metadata_id("source_family", value)


def validate_session_id(value: Any) -> str:
    """세션 디렉터리와 매니페스트를 연결하는 세션 ID를 검증한다."""
    return _validate_metadata_id("session_id", value)


def _is_absolute_on_any_platform(value: str) -> bool:
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _validate_entry(entry: dict, *, index: int) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"manifest entry #{index}는 객체여야 합니다: {entry!r}")

    path = entry.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"manifest entry #{index}의 path가 올바르지 않습니다: {path!r}")

    path_base = entry.get("path_base")
    if path_base is not None:
        if path_base != MANIFEST_PATH_BASE:
            raise ValueError(
                f"manifest entry #{index}: 지원하지 않는 path_base={path_base!r}"
            )
        if _is_absolute_on_any_platform(path):
            raise ValueError(
                f"manifest entry #{index}: path_base='manifest' 경로는 상대 경로여야 합니다: "
                f"{path!r}"
            )

    if "group_id" in entry:
        validate_group_id(entry["group_id"])
    if "source_family" in entry:
        validate_source_family(entry["source_family"])
    if "session_id" in entry:
        validate_session_id(entry["session_id"])


def validate_group_splits(entries: list[dict]) -> None:
    """동일 group_id의 split/source_family가 충돌하면 즉시 거부한다."""
    seen: dict[str, str] = {}
    seen_families: dict[str, str] = {}
    for index, entry in enumerate(entries):
        if "group_id" not in entry:
            continue
        group_id = validate_group_id(entry["group_id"])
        if "source_family" in entry:
            family = validate_source_family(entry["source_family"])
            previous_family = seen_families.setdefault(group_id, family)
            if previous_family != family:
                raise ValueError(
                    f"group_id={group_id!r}의 source_family가 일관되지 않습니다: "
                    f"{previous_family!r}, {family!r}"
                )
        split = entry.get("split")
        if split is None:
            continue
        if split not in VALID_SPLITS:
            raise ValueError(
                f"manifest entry #{index}: split은 {VALID_SPLITS} 중 하나여야 합니다: {split!r}"
            )
        previous = seen.setdefault(group_id, split)
        if previous != split:
            raise ValueError(
                f"group_id={group_id!r}가 여러 split에 걸쳐 있습니다: "
                f"{previous!r}, {split!r}"
            )


def manifest_relative_path(target: str | Path, manifest_path: str | Path) -> str:
    """target을 manifest 파일의 부모 기준 POSIX 상대 경로로 만든다."""
    target_abs = Path(target).resolve()
    manifest_parent = Path(manifest_path).resolve().parent
    return Path(os.path.relpath(target_abs, start=manifest_parent)).as_posix()


def write_manifest(entries: list[dict], path: str | Path) -> None:
    for index, entry in enumerate(entries):
        _validate_entry(entry, index=index)
        if entry.get("split") not in VALID_SPLITS:
            raise ValueError(f"split 은 {VALID_SPLITS} 중 하나여야 합니다: {entry}")
    validate_group_splits(entries)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_manifest(path: str | Path, split: str | None = None) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest 없음: {p}")
    if split is not None and split not in VALID_SPLITS:
        raise ValueError(f"split 은 {VALID_SPLITS} 중 하나여야 합니다: {split!r}")

    raw_entries: list[dict] = []
    with open(p, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            _validate_entry(entry, index=line_number)
            raw_entries.append(entry)

    # split 필터 전에 전체 매니페스트를 검사해야 group 누수를 놓치지 않는다.
    validate_group_splits(raw_entries)

    entries: list[dict] = []
    manifest_parent = p.resolve().parent
    for raw_entry in raw_entries:
        if split is not None and raw_entry.get("split") != split:
            continue
        entry = dict(raw_entry)
        if entry.get("path_base") == MANIFEST_PATH_BASE:
            entry["path"] = str((manifest_parent / entry["path"]).resolve())
        entries.append(entry)
    return entries


def scan_wavs(root: str | Path, tag: str) -> list[dict]:
    """하위 WAV/FLAC/MP3 파일을 재귀 스캔해 manifest 엔트리를 생성한다.

    ``soundfile``로 읽을 수 없는 파일은 건너뛰며, split은 호출자가 배정한다.
    """
    root = Path(root)
    entries = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        try:
            info = sf.info(str(p))
        except RuntimeError:
            continue
        entries.append(
            {
                "path": str(p),
                "duration_s": float(info.frames) / float(info.samplerate),
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "tag": tag,
            }
        )
    return entries


def assign_splits(
    entries: list[dict],
    ratios: dict[str, float],
    seed: int = 20260802,
    group_key: str | None = "group_id",
    stratify_key: str | None = None,
    min_units_per_split: dict[str, int] | None = None,
) -> list[dict]:
    """재현 가능한 분할을 배정한다.

    모든 항목에 ``group_key``가 있으면 그룹을 원자 단위로 섞는다. 일부 항목에만
    키가 있으면 누수 가능성이 있으므로 거부한다. 키가 전혀 없으면 legacy 파일
    단위 분할을 유지한다. ``stratify_key``가 있으면 각 계층 안에서 그룹을 따로
    배정하며, 그룹 수가 양수 비율 split 수 이상이면 모든 split을 최소 1회 덮는다.

    ``min_units_per_split`` — 계층마다 split 이 가져야 할 **단위(그룹) 하한**
    ------------------------------------------------------------------------
    비율 분할만으로는 이 하한을 만족시킬 수 없다. 2026-08-06 실측: 실측 80세션/
    64그룹을 8:1:1 로 나누면 val·test 가 계열당 **1~2 그룹**이 되는데
    ``recorded_statistical_power`` 게이트는 **4 그룹**을 요구한다. 비율로 4를 얻으려면
    계열당 40그룹(=160세션)이 필요하고, 그건 스피커 시간 2배다.

    그래서 하한을 **먼저 확보하고 나머지를 비율로** 나눈다. 하한을 만족시킬 수 없으면
    조용히 적게 주지 않고 **실패한다** — "게이트가 요구하는 것을 만족하지 못하는
    manifest" 가 조용히 만들어지면, 그 사실은 학습이 끝난 뒤 G4 판정 불가로만 드러난다.

    하한의 단일 출처는 ``eval.recorded.MIN_GROUPS_PER_FAMILY`` 이고, 이 함수는 그것을
    **읽지 않는다** — 호출자가 넘긴다. 여기서 다시 유도하면 그것이 두 번째 정의가 된다.
    """
    import numpy as np

    n = len(entries)
    if n == 0:
        return []

    train_ratio = float(ratios.get("train", 0.9))
    val_ratio = float(ratios.get("val", 0.05))
    test_ratio = float(ratios.get("test", 1.0 - train_ratio - val_ratio))
    ratio_values = (train_ratio, val_ratio, test_ratio)
    if any(not np.isfinite(value) or value < 0.0 for value in ratio_values):
        raise ValueError(f"분할 비율은 유한한 0 이상 값이어야 합니다: {ratios}")
    if not np.isclose(sum(ratio_values), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"train/val/test 분할 비율 합은 1이어야 합니다: {ratios}")

    grouped = False
    groups: dict[str, list[int]] = {}
    if group_key is None and any("group_id" in entry for entry in entries):
        raise ValueError("group_id가 있는 항목은 그룹 분할을 비활성화할 수 없습니다")
    if group_key is not None:
        present = [group_key in entry for entry in entries]
        if any(present) and not all(present):
            raise ValueError(
                f"{group_key!r}는 모든 항목에 있거나 모두 없어야 합니다"
            )
        if all(present):
            grouped = True
            for index, entry in enumerate(entries):
                group_id = validate_group_id(entry[group_key])
                groups.setdefault(group_id, []).append(index)

    rng = np.random.default_rng(seed)
    out = [dict(e) for e in entries]

    if grouped:
        # 입력 순서가 바뀌어도 같은 seed/group 집합이면 결과가 같도록 먼저 정렬한다.
        unit_indices = [groups[group_id] for group_id in sorted(groups)]
    else:
        unit_indices = [[index] for index in range(n)]

    strata: dict[str, list[int]] = {"__all__": list(range(len(unit_indices)))}
    if stratify_key is not None:
        strata = {}
        for unit_index, indices in enumerate(unit_indices):
            missing = [index for index in indices if stratify_key not in entries[index]]
            if missing:
                raise ValueError(
                    f"stratify_key={stratify_key!r}가 그룹의 모든 항목에 필요합니다: "
                    f"entry indices={missing}"
                )
            values: set[str] = set()
            for index in indices:
                raw_value = entries[index][stratify_key]
                if stratify_key == "source_family":
                    value = validate_source_family(raw_value)
                else:
                    value = _validate_metadata_id(stratify_key, raw_value)
                values.add(value)
            if len(values) != 1:
                raise ValueError(
                    f"한 그룹 안의 {stratify_key!r} 값은 같아야 합니다: {sorted(values)}"
                )
            value = next(iter(values))
            strata.setdefault(value, []).append(unit_index)

    split_ratios = (train_ratio, val_ratio, test_ratio)

    floors = [0, 0, 0]
    if min_units_per_split:
        unknown = sorted(set(min_units_per_split) - set(VALID_SPLITS))
        if unknown:
            raise ValueError(f"알 수 없는 split 이름: {unknown}")
        for position, name in enumerate(VALID_SPLITS):
            value = int(min_units_per_split.get(name, 0))
            if value < 0:
                raise ValueError(f"{name} 단위 하한은 0 이상이어야 합니다: {value}")
            if value > 0 and split_ratios[position] <= 0.0:
                raise ValueError(
                    f"{name} 비율이 0 인데 단위 하한 {value} 를 요구할 수 없습니다"
                )
            floors[position] = value

    def allocate_counts(n_units: int, stratum: str = "__all__") -> list[int]:
        raw = [n_units * ratio for ratio in split_ratios]
        counts = [math.floor(value) for value in raw]
        remaining = n_units - sum(counts)
        remainder_order = sorted(
            range(len(counts)), key=lambda i: (-(raw[i] - counts[i]), i)
        )
        for index in remainder_order[:remaining]:
            counts[index] += 1

        positive = [i for i, ratio in enumerate(split_ratios) if ratio > 0.0]
        if n_units >= len(positive):
            for missing_index in positive:
                if counts[missing_index] > 0:
                    continue
                donors = [i for i in positive if counts[i] > 1]
                if not donors:  # 방어 코드: n_units >= len(positive)이면 발생하지 않는다.
                    break
                donor = max(
                    donors,
                    key=lambda i: (counts[i] - raw[i], counts[i], split_ratios[i], -i),
                )
                counts[donor] -= 1
                counts[missing_index] += 1

        if any(floors):
            # 하한을 먼저 확보한다. 남는 것이 없으면 조용히 줄이지 않고 실패한다.
            shortfall = [
                f"{VALID_SPLITS[i]} {floors[i]}" for i in range(3) if floors[i] > 0
            ]
            # train 에도 최소 1 단위는 남겨야 학습이 성립한다.
            train_floor = 1 if split_ratios[0] > 0.0 else 0
            needed = sum(floors) + max(0, train_floor - floors[0])
            if n_units < needed:
                where = f"[{stratum}] " if stratum != "__all__" else ""
                raise ValueError(
                    f"{where}단위(그룹)가 {n_units} 개뿐이라 split 하한을 만족할 수 없습니다: "
                    f"요구 {', '.join(shortfall)} + train 최소 {train_floor} = {needed}. "
                    "비율을 바꿔도 해결되지 않습니다 — 서로 다른 그룹(원본)을 더 모아야 합니다."
                )
            for position in range(3):
                if counts[position] >= floors[position]:
                    continue
                deficit = floors[position] - counts[position]
                # 여유가 가장 많은 split 에서 가져온다. train 을 먼저 헐되 1 은 남긴다.
                while deficit > 0:
                    donors = [
                        i
                        for i in range(3)
                        if counts[i] - floors[i] > (train_floor if i == 0 else 0)
                    ]
                    if not donors:
                        donors = [i for i in range(3) if counts[i] > floors[i]]
                    if not donors:  # 방어: 위 needed 검사가 이미 막는다.
                        raise ValueError(
                            f"[{stratum}] split 하한을 만족시킬 단위가 없습니다"
                        )
                    donor = max(donors, key=lambda i: (counts[i] - floors[i], -i))
                    counts[donor] -= 1
                    counts[position] += 1
                    deficit -= 1
        return counts

    for stratum in sorted(strata):
        stratum_units = strata[stratum]
        order = rng.permutation(stratum_units)
        n_train, n_val, _n_test = allocate_counts(len(stratum_units), stratum)
        for rank, unit_index in enumerate(order):
            if rank < n_train:
                assigned = "train"
            elif rank < n_train + n_val:
                assigned = "val"
            else:
                assigned = "test"
            for entry_index in unit_indices[int(unit_index)]:
                out[entry_index]["split"] = assigned

    validate_group_splits(out)
    return out
