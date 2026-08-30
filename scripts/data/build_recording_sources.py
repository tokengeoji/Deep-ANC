#!/usr/bin/env python3
"""덕트 녹음에 재생할 **소스 오디오 풀**을 만든다 (스피커 출력 없음).

recorded 세션은 실제 소리를 덕트에 흘려 넣고 ERR/REF 마이크로 받는다. 그러려면 재생할
소리가 먼저 있어야 하는데, 공개 데이터는 짧은 클립(ESC-50 5초)이거나 포맷/샘플레이트가
제각각이다. 한 클립을 70초 동안 반복 재생하면 세션 안의 다양성이 사라지므로, 서로 다른
클립을 이어 붙여 세션 길이의 파일을 미리 만든다.

    data/source_pool/<family>/<family>_<idx>.wav    48kHz mono float32
    data/source_pool/sources.csv                    어떤 원본이 들어갔는지 (출처 추적)

family 는 파인튜닝 readiness 가 요구하는 4종이다 — speech / music / environment / machine.
같은 원본에서 나온 세션은 같은 ``group_id`` 를 받아야 split 누수를 막을 수 있으므로,
세션 파일 하나는 **한 그룹의 클립들로만** 채운다.

    .venv/bin/python scripts/data/build_recording_sources.py --sessions-per-family 20
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.eval.artifacts import write_csv  # noqa: E402

TARGET_RATE = 48000

# ESC-50 카테고리를 family 로 나눈다. 회전/왕복 기계음과 환경음을 섞으면 기능 2 의
# "소스 종류별 최악값" 평가가 흐려지므로 여기서 분명히 가른다.
ESC50_MACHINE = (
    "engine", "chainsaw", "vacuum_cleaner", "washing_machine", "hand_saw",
    "helicopter", "airplane", "train",
)
ESC50_ENVIRONMENT = (
    "rain", "sea_waves", "wind", "thunderstorm", "crackling_fire", "water_drops",
    "pouring_water", "crickets", "insects", "chirping_birds", "frog", "footsteps",
    "fireworks", "church_bells", "toilet_flush", "clock_tick",
)

CROSSFADE_SECONDS = 0.05

# 재생 경로는 **피크**로 정규화된다(NoiseProgram: mono/peak*amplitude). 그래서 파일의
# 크레스트 팩터가 곧 실제 음향 에너지를 결정한다. ESC-50 클립을 그대로 이어 붙이면
# 크레스트가 27dB까지 올라가고, 진폭 0.15로 재생해도 RMS 는 0.0026 밖에 되지 않아
# 마이크에는 잡음 바닥만 남는다(실측: 구동/무구동 대비 +0.3dB).
# 소프트 클리핑으로 크레스트를 이 값까지 낮춘 뒤 피크 정규화한다.
TARGET_CREST_DB = 10.0


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """wav/flac/mp3 를 mono float32 로 읽는다. mp3 는 libsndfile 지원 여부에 따라 폴백."""

    import soundfile as sf

    try:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        if path.suffix.lower() != ".mp3":
            raise
        import audioread  # type: ignore

        with audioread.audio_open(str(path)) as handle:
            rate = int(handle.samplerate)
            channels = int(handle.channels)
            chunks = [np.frombuffer(buf, dtype="<i2") for buf in handle]
            raw = np.concatenate(chunks).astype(np.float32) / 32768.0
        data = raw.reshape(-1, channels)
    return data.mean(axis=1).astype(np.float32), int(rate)


def to_target_rate(signal: np.ndarray, rate: int) -> np.ndarray:
    if rate == TARGET_RATE:
        return signal
    from math import gcd

    from scipy import signal as sps

    divisor = gcd(int(rate), TARGET_RATE)
    return sps.resample_poly(signal, TARGET_RATE // divisor, int(rate) // divisor).astype(
        np.float32
    )


def concatenate(clips: list[np.ndarray], total_samples: int) -> np.ndarray:
    """클립을 짧은 크로스페이드로 이어 붙여 정확히 ``total_samples`` 를 채운다.

    경계를 그냥 붙이면 광대역 클릭이 생겨 덕트 응답이 아니라 클릭의 응답을 녹음하게 된다.
    """

    fade = int(CROSSFADE_SECONDS * TARGET_RATE)
    output = np.zeros(total_samples, dtype=np.float32)
    position = 0
    index = 0
    while position < total_samples and clips:
        clip = clips[index % len(clips)]
        index += 1
        if clip.size <= 2 * fade:
            continue
        # 클립별 레벨 차이가 크면 한 클립이 세션을 지배한다. RMS 로 맞춘다.
        rms = float(np.sqrt(np.mean(clip.astype(np.float64) ** 2)))
        if rms < 1e-6:
            continue
        clip = (clip / rms * 0.1).astype(np.float32)
        window = np.ones(clip.size, dtype=np.float32)
        window[:fade] = np.linspace(0.0, 1.0, fade)
        window[-fade:] = np.linspace(1.0, 0.0, fade)
        clip = clip * window
        take = min(clip.size, total_samples - position)
        output[position : position + take] += clip[:take]
        position += max(1, take - fade)
    return limit_crest(output)


def crest_db(signal: np.ndarray) -> float:
    values = np.asarray(signal, dtype=np.float64)
    rms = float(np.sqrt(np.mean(values**2)))
    peak = float(np.max(np.abs(values)))
    if rms <= 0.0 or peak <= 0.0:
        return float("inf")
    return 20.0 * np.log10(peak / rms)


def limit_crest(signal: np.ndarray, target_db: float = TARGET_CREST_DB) -> np.ndarray:
    """소프트 클리핑으로 크레스트 팩터를 target 까지 낮추고 피크 정규화한다.

    하드 클리핑은 고조파를 넣어 덕트 응답과 섞이므로 쓰지 않는다. tanh 는 임계를 넘는
    부분만 완만히 눌러 파형의 저역 구조를 보존한다.
    """

    values = np.asarray(signal, dtype=np.float64)
    rms = float(np.sqrt(np.mean(values**2)))
    if rms <= 0.0:
        return values.astype(np.float32)
    # 임계를 낮춰가며 목표 크레스트에 도달하는 가장 약한 압축을 고른다.
    for threshold_db in np.arange(target_db, -2.0, -0.5):
        threshold = rms * (10.0 ** (threshold_db / 20.0))
        compressed = threshold * np.tanh(values / threshold)
        if crest_db(compressed) <= target_db:
            values = compressed
            break
    else:
        values = compressed
    peak = float(np.max(np.abs(values)))
    return (values / peak * 0.99).astype(np.float32) if peak > 0 else values.astype(np.float32)


# 한 그룹에 넣을 서로 다른 원본 녹음(src_file) 수.
#
# 왜 카테고리가 아니라 src_file 인가
# --------------------------------
# 2026-08-06 실측: group_id 를 ESC-50 **카테고리**로 잡았더니 machine 이 8 그룹뿐이었고,
# 그 8 그룹으로는 `min_groups_per_family_per_split=4` (val 4 + test 4 + train 1 = 9)를
# 만족할 수 없어 manifest 생성이 실패한다. 세션을 아무리 늘려도 그룹은 안 는다.
#
# 그런데 `esc50.csv` 에는 `src_file`(원본 Freesound 녹음 ID)이 있고 machine 8 카테고리의
# 고유 src_file 은 **209 개**다 (airplane 32 / chainsaw 22 / engine 34 / hand_saw 22 /
# helicopter 15 / train 28 / vacuum_cleaner 30 / washing_machine 26).
# 같은 카테고리라도 다른 src_file 은 다른 녹음이므로 독립 클러스터로 쓸 수 있다.
#
# src_file 하나당 클립이 1~2 개(5초)뿐이라 70초 세션을 채우지 못한다. 그래서 서로소인
# src_file 을 이만큼씩 묶어 한 그룹으로 만든다 — 그룹 독립성을 지키면서 세션 안 다양성도
# 남긴다. machine 기준 209/4 ≈ 52 그룹이 나온다.
SRC_FILES_PER_GROUP = 4


def collect_esc50(root: Path) -> dict[str, list[tuple[Path, str]]]:
    import csv as csv_module

    meta = root / "ESC-50-master" / "meta" / "esc50.csv"
    audio = root / "ESC-50-master" / "audio"
    if not meta.exists():
        return {}
    # (family, category, src_file) → 파일들. src_file 이 그룹의 원자 단위다.
    by_source: dict[tuple[str, str, str], list[Path]] = {}
    for row in csv_module.DictReader(meta.open(encoding="utf-8")):
        category = row["category"]
        family = (
            "machine" if category in ESC50_MACHINE
            else "environment" if category in ESC50_ENVIRONMENT
            else None
        )
        if family is None:
            continue
        # src_file 이 없는 구형 메타데이터는 카테고리로 폴백한다(그룹이 거칠어질 뿐 안전).
        src_file = str(row.get("src_file") or category)
        by_source.setdefault((family, category, src_file), []).append(
            audio / row["filename"]
        )

    buckets: dict[str, list[tuple[Path, str]]] = {"machine": [], "environment": []}
    # 카테고리 안에서 src_file 을 정렬해 결정적으로 묶는다. 카테고리를 가로지르지
    # 않는 이유: 한 세션이 여러 카테고리를 섞으면 "소스 종류별 최악값"(절대목표 2)
    # 평가에서 그 세션이 어느 계열의 무엇인지 말할 수 없게 된다.
    per_category: dict[tuple[str, str], list[str]] = {}
    for family, category, src_file in by_source:
        per_category.setdefault((family, category), []).append(src_file)
    for (family, category), src_files in sorted(per_category.items()):
        for index, start in enumerate(range(0, len(src_files), SRC_FILES_PER_GROUP)):
            group = f"{category}-{index:02d}"
            for src_file in sorted(src_files)[start : start + SRC_FILES_PER_GROUP]:
                for path in sorted(by_source[(family, category, src_file)]):
                    buckets[family].append((path, group))
    return buckets


def clips_used_by_sessions(csv_paths: list[Path]) -> set[str]:
    """이미 녹음된 세션이 쓴 클립 파일명을 모은다 (``sources.csv`` 의 ``clips`` 열).

    왜 필요한가
    ----------
    2026-08-06: 재정렬로 47세션이 살아남고 33세션만 다시 받으면 되는데, 그 33세션의
    소스를 새 그룹 정의로 다시 만들면 **같은 클립이 옛 그룹과 새 그룹에 동시에 들어갈
    수 있다.** 그러면 group 단위 split 이 무의미해진다 — 같은 오디오가 train 과 test 에
    함께 있게 된다.

    ``clips`` 열은 세션당 앞 12개만 저장하므로 이 집합은 **하한**이다. 그래서 호출부는
    이걸로 클립을 지운 뒤에도 그룹 수가 충분한지 따로 확인해야 한다.
    """

    import ast
    import csv as csv_module

    used: set[str] = set()
    for csv_path in csv_paths:
        path = csv_path if csv_path.is_absolute() else REPO_ROOT / csv_path
        if not path.exists():
            print(f"  [warn] 없는 경로: {path}")
            continue
        for row in csv_module.DictReader(path.open(encoding="utf-8")):
            raw = (row.get("clips") or "").strip()
            if not raw:
                continue
            try:
                names = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                print(f"  [warn] clips 열을 읽지 못했습니다: {raw[:60]}")
                continue
            used.update(str(name) for name in names)
    return used


MIN_DISTINCT_GROUPS = 8


def collect_tree(root: Path, suffixes: tuple[str, ...]) -> list[tuple[Path, str]]:
    """디렉터리 트리에서 파일을 모으고, **그룹 키가 되는 디렉터리 깊이를 자동으로 고른다.**

    LibriSpeech 는 화자 디렉터리, FMA 는 앨범 디렉터리가 그룹이다. 그런데 데이터셋마다
    루트 아래 껍데기 디렉터리 수가 달라서(``LibriSpeech/dev-clean/<화자>/...`` vs
    ``fma_small/<앨범>/...``) 깊이를 상수로 박으면 틀린 것을 그룹으로 쓴다. 실제로 깊이 1을
    쓰다가 화자 대신 ``dev-clean`` 하나만 나와서, 음성 20세션이 전부 같은 그룹이 될 뻔했다.
    그러면 manifest 의 8:1:1 분할에서 speech 가 통째로 한 split 에 몰려 val/test 에서
    source_family 가 누락되고, G2/G4 게이트가 그 자리에서 깨진다.

    그래서 **서로 다른 그룹이 ``MIN_DISTINCT_GROUPS`` 이상 나오는 가장 얕은 깊이**를 쓴다.
    """

    if not root.exists():
        return []
    files = [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    if not files:
        return []
    relative = [path.relative_to(root).parts for path in files]
    max_depth = max(len(parts) - 1 for parts in relative)
    chosen = 0
    for depth in range(max_depth):
        distinct = {parts[depth] for parts in relative if len(parts) > depth + 1}
        chosen = depth
        if len(distinct) >= MIN_DISTINCT_GROUPS:
            break
    return [
        (path, parts[chosen] if len(parts) > chosen + 1 else "root")
        for path, parts in zip(files, relative)
    ]


def build_family(
    name: str,
    candidates: list[tuple[Path, str]],
    *,
    sessions: int,
    seconds: float,
    out_root: Path,
    rng: random.Random,
) -> list[dict]:
    if not candidates:
        print(f"[skip] {name}: 원본이 없습니다")
        return []
    by_group: dict[str, list[Path]] = {}
    for path, group in candidates:
        by_group.setdefault(group, []).append(path)
    groups = sorted(by_group)
    rng.shuffle(groups)

    total_samples = int(seconds * TARGET_RATE)
    directory = out_root / name
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for index in range(sessions):
        # 한 세션 = 한 그룹. 그룹 수가 부족하면 순환하되 group_id 는 그대로 유지한다.
        group = groups[index % len(groups)]
        pool = list(by_group[group])
        rng.shuffle(pool)
        clips: list[np.ndarray] = []
        used: list[str] = []
        collected = 0
        for path in pool:
            try:
                signal, rate = load_audio(path)
            except Exception as exc:  # 한 파일 때문에 풀 전체를 잃지 않는다
                print(f"  [warn] {path.name}: {exc}")
                continue
            signal = to_target_rate(signal, rate)
            if signal.size < int(0.5 * TARGET_RATE):
                continue
            clips.append(signal)
            used.append(path.name)
            collected += signal.size
            if collected >= total_samples * 1.2:
                break
        if not clips:
            print(f"  [warn] {name} #{index}: 사용 가능한 클립이 없습니다 (group={group})")
            continue
        audio = concatenate(clips, total_samples)
        out_path = directory / f"{name}_{index:03d}.wav"
        import soundfile as sf

        sf.write(str(out_path), audio, TARGET_RATE, subtype="FLOAT")
        crest = crest_db(audio)
        rows.append({
            "source_family": name,
            "session_index": index,
            "group_id": f"{name}-{group}".replace("_", "-").lower(),
            "path": str(out_path.relative_to(REPO_ROOT)),
            "seconds": total_samples / TARGET_RATE,
            "sample_rate_hz": TARGET_RATE,
            "clip_count": len(clips),
            "crest_factor_db": crest,
            "rms_at_unit_peak": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
            # ⚠ 2026-08-07 — 여기가 ``used[:12]`` 였다. clip_count 는 17 인데 clips 에는
            # 12개만 적혀, **재생한 원본 256개가 어디에도 기록되지 않았다.** 누수 게이트와
            # holdout 생성기가 이 열을 유일한 근거로 읽으므로, 기록되지 않은 클립은
            # "재생한 적 없는 것" 이 되어 합성 학습셋에 남는다. 실제로 실측 test 세션이
            # 재생한 ``4-117627-A-25.wav`` 가 합성 train 에 split=train 으로 살아 있었다
            # (정규화 상호상관 0.884 @ 59.40 s, 대조군 0.069/0.089).
            # 표시용 절단과 기록용 목록을 같은 열에 두면 안 된다.
            "clips": list(used),
        })
        print(
            f"  {out_path.name}  group={group}  클립 {len(clips)}개  "
            f"crest {crest:.1f} dB  rms {np.sqrt(np.mean(audio.astype(np.float64)**2)):.4f}"
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-per-family", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=70.0)
    parser.add_argument("--out", default="data/source_pool")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--families", nargs="+", default=["environment", "machine", "speech", "music"]
    )
    parser.add_argument(
        "--keep-disjoint-from",
        nargs="+",
        default=[],
        metavar="SOURCES_CSV",
        help=(
            "이미 녹음에 쓰인 sources.csv 들. 거기 등장한 클립은 새 소스에서 제외한다. "
            "일부 세션만 다시 녹음할 때 필수다 — 같은 클립이 옛 그룹과 새 그룹에 "
            "동시에 들어가면 split 을 가로지른다."
        ),
    )
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    out_root = REPO_ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    esc50 = collect_esc50(REPO_ROOT / "data/raw/noise/esc50")
    sources = {
        "environment": esc50.get("environment", []),
        "machine": esc50.get("machine", []),
        "speech": collect_tree(REPO_ROOT / "data/raw/speech", (".flac", ".wav")),
        "music": collect_tree(REPO_ROOT / "data/raw/music", (".mp3", ".wav")),
    }

    if args.keep_disjoint_from:
        used = clips_used_by_sessions(
            [Path(value) for value in args.keep_disjoint_from]
        )
        if not used:
            print(
                "[중단] --keep-disjoint-from 이 클립을 하나도 못 읽었습니다. "
                "경로가 sources.csv 인지 확인하세요 — 빈 목록으로 진행하면 "
                "누수를 막는 척만 하게 됩니다.",
                file=sys.stderr,
            )
            return 2
        before = {family: len(items) for family, items in sources.items()}
        sources = {
            family: [(path, group) for path, group in items if path.name not in used]
            for family, items in sources.items()
        }
        print(f"[누수 차단] 이미 녹음에 쓰인 클립 {len(used)}개 제외")
        for family in sorted(sources):
            removed = before[family] - len(sources[family])
            if removed:
                print(f"  {family}: {before[family]} → {len(sources[family])} (−{removed})")

    rows: list[dict] = []
    for family in args.families:
        print(f"\n[{family}] 후보 {len(sources.get(family, []))}개")
        rows += build_family(
            family, sources.get(family, []),
            sessions=args.sessions_per_family,
            seconds=args.seconds,
            out_root=out_root,
            rng=rng,
        )

    if not rows:
        print("[중단] 만들어진 소스가 없습니다.", file=sys.stderr)
        return 2

    # --families 로 한 계열만 다시 만들어도 목록 전체를 잃지 않는다. 덮어쓰면 디스크에는
    # 파일이 60개 있는데 목록에는 20개만 남아, 녹음 배치가 나머지를 영원히 건너뛴다.
    csv_path = out_root / "sources.csv"
    merged: dict[tuple[str, int], dict] = {}
    if csv_path.exists():
        import csv as csv_module

        for existing in csv_module.DictReader(csv_path.open(encoding="utf-8")):
            path_value = REPO_ROOT / existing["path"]
            if not path_value.exists():
                continue  # 파일이 사라진 항목은 되살리지 않는다
            existing["session_index"] = int(existing["session_index"])
            existing["seconds"] = float(existing["seconds"])
            merged[(existing["source_family"], existing["session_index"])] = existing
    for row in rows:
        merged[(row["source_family"], row["session_index"])] = row
    ordered = [merged[key] for key in sorted(merged)]

    path = write_csv(csv_path, ordered)
    total_minutes = sum(float(r["seconds"]) for r in ordered) / 60.0
    families = sorted({r["source_family"] for r in ordered})
    print(f"\n이번 실행 {len(rows)}개 · 목록 전체 {len(ordered)}개 · 총 {total_minutes:.1f}분")
    print(f"계열: {', '.join(families)}")
    print(f"목록: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
