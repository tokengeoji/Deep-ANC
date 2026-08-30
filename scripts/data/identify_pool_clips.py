"""풀 WAV의 후보 clip을 찾는 **비권위 진단 전용** 지문 실험.

경고
----
이 도구의 ``clip_count`` 등분 가정은 historical builder와 맞지 않는다. 실제 builder는
가변 길이 원본을 50 ms crossfade로 이어 붙이고 ``take - fade``만큼 진행하며, 70초
끝에서는 마지막 clip을 자른다. 따라서 등분 조각은 실제 경계가 아니며 NCC 결과로
``sources.csv``·recorded holdout·split을 복구하거나 누수 없음의 근거를 만들 수 없다.

권위 복구에는 ``repair_source_pool_provenance.py``를 사용한다. 그 도구는 v1/v2 당시 Git
builder·seed·RNG/exclusion을 재현하고 보존 WAV와 PCM exact(정수 PCM만 1 LSB)까지
검증한다. 이 파일은 알고리즘 비교를 위한 ``*.diagnostic.json``만 만들 수 있으며,
명시적인 ``--diagnostic-only`` 확인 없이는 실행을 거부한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

TARGET_RATE = 48_000
NCC_ACCEPT = 0.40
FINGERPRINT_CANDIDATES = 24
MEL_BANDS = 40
AUTHORITATIVE_PROVENANCE = False


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    n = int(round(x.size * dst / src))
    return np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),
        np.linspace(0.0, 1.0, x.size, endpoint=False),
        x,
    ).astype(np.float64)


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(str(path), dtype="float64", always_2d=True)
    return audio.mean(axis=1), int(rate)


def _mel_filterbank(n_fft: int, rate: int, bands: int) -> np.ndarray:
    def to_mel(f: np.ndarray | float) -> np.ndarray | float:
        return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    def to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    edges = to_hz(np.linspace(to_mel(50.0), to_mel(rate / 2.0), bands + 2))
    bins = np.floor((n_fft + 1) * edges / rate).astype(int)
    bank = np.zeros((bands, n_fft // 2 + 1), dtype=np.float64)
    for b in range(bands):
        lo, mid, hi = bins[b], bins[b + 1], bins[b + 2]
        if mid <= lo:
            mid = lo + 1
        if hi <= mid:
            hi = mid + 1
        hi = min(hi, bank.shape[1] - 1)
        if mid >= bank.shape[1]:
            continue
        bank[b, lo:mid] = np.linspace(0.0, 1.0, max(1, mid - lo))
        bank[b, mid:hi] = np.linspace(1.0, 0.0, max(1, hi - mid))
    return bank


def fingerprint(x: np.ndarray, rate: int, bank: np.ndarray, n_fft: int) -> np.ndarray:
    """로그-멜 평균 벡터. 길이·이득에 둔감해야 후보 선별에 쓸 수 있다."""

    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    hop = n_fft // 2
    frames = 1 + (x.size - n_fft) // hop
    window = np.hanning(n_fft)
    acc = np.zeros(bank.shape[0], dtype=np.float64)
    for i in range(frames):
        seg = x[i * hop : i * hop + n_fft] * window
        power = np.abs(np.fft.rfft(seg)) ** 2
        acc += bank @ power
    acc /= max(1, frames)
    vec = np.log10(acc + 1e-12)
    vec -= vec.mean()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """정규화 상호상관의 최대값. 길이가 다르면 짧은 쪽을 미끄러뜨린다."""

    if a.size < b.size:
        a, b = b, a
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    n = 1 << int(np.ceil(np.log2(a.size + b.size)))
    corr = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    return float(np.max(np.abs(corr)) / (na * nb))


def load_corpus(manifests: list[Path]) -> list[tuple[str, Path]]:
    seen: dict[str, Path] = {}
    for manifest in manifests:
        if not manifest.is_file():
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            path = Path(row["path"])
            if path.is_file():
                seen.setdefault(path.name, path)
    return sorted(seen.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="등분 경계가 historical builder와 다르며 provenance에 쓸 수 없음을 확인",
    )
    parser.add_argument("--pool", action="append", required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ncc-accept", type=float, default=NCC_ACCEPT)
    parser.add_argument("--candidates", type=int, default=FINGERPRINT_CANDIDATES)
    args = parser.parse_args(argv)

    if not args.diagnostic_only:
        parser.error(
            "이 도구는 비권위 진단 전용입니다. provenance 복구에는 "
            "repair_source_pool_provenance.py를 사용하세요. 계속하려면 "
            "--diagnostic-only를 명시하세요"
        )
    out = Path(args.out)
    if not out.name.endswith(".diagnostic.json"):
        parser.error(
            "진단 결과는 '*.diagnostic.json'으로만 기록할 수 있습니다. "
            "sources.csv/holdout과 혼동되는 이름은 거부합니다"
        )

    n_fft = 2048
    bank = _mel_filterbank(n_fft, TARGET_RATE, MEL_BANDS)

    corpus = load_corpus([Path(m) for m in args.manifest])
    print(f"코퍼스 {len(corpus)}개 지문 계산 중...", flush=True)
    names: list[str] = []
    prints: list[np.ndarray] = []
    audio_cache: dict[str, np.ndarray] = {}
    unreadable: list[str] = []
    for name, path in corpus:
        try:
            x, rate = _read_mono(path)
        except Exception as exc:  # noqa: BLE001 - 코퍼스 파일 하나가 전수를 막으면 안 된다
            unreadable.append(f"{name}: {exc}")
            continue
        x = _resample(x, rate, TARGET_RATE)
        audio_cache[name] = x
        names.append(name)
        prints.append(fingerprint(x, TARGET_RATE, bank, n_fft))
    matrix = np.stack(prints)
    # 읽지 못한 파일은 **조용히 넘기지 않는다** — 그 파일이 재생됐다면 식별에서
    # 빠지고, 빠진 것을 "재생 안 함" 으로 읽는 것이 애초의 결함이다.
    if unreadable:
        print(f"  ⚠ 읽지 못한 코퍼스 {len(unreadable)}개:", flush=True)
        for line in unreadable[:20]:
            print(f"      {line}", flush=True)
    print(f"  지문 {matrix.shape} (읽기 실패 {len(unreadable)})", flush=True)

    report: dict[str, dict] = {
        "_authority": {
            "authoritative": AUTHORITATIVE_PROVENANCE,
            "purpose": "diagnostic_only",
            "known_invalid_assumption": (
                "clip_count 등분 경계는 variable-length + 50ms crossfade builder와 불일치"
            ),
            "must_not_feed": ["sources.csv", "recorded_holdout.json", "recorded split"],
        }
    }
    for pool_dir in args.pool:
        pool = Path(pool_dir)
        rows = list(csv.DictReader((pool / "sources.csv").open(encoding="utf-8")))
        resolved: list[dict] = []
        for row in rows:
            path = REPO_ROOT / row["path"]
            count = int(row["clip_count"])
            x, rate = _read_mono(path)
            x = _resample(x, rate, TARGET_RATE)
            bounds = np.linspace(0, x.size, count + 1).astype(int)
            found: list[str] = []
            weak: list[dict] = []
            for i in range(count):
                seg = x[bounds[i] : bounds[i + 1]]
                if seg.size < n_fft:
                    weak.append({"index": i, "reason": "too_short"})
                    continue
                fp = fingerprint(seg, TARGET_RATE, bank, n_fft)
                scores = matrix @ fp
                top = np.argsort(scores)[::-1][: args.candidates]
                best_name, best_ncc = None, 0.0
                for j in top:
                    value = ncc(seg, audio_cache[names[j]])
                    if value > best_ncc:
                        best_name, best_ncc = names[j], value
                if best_name is not None and best_ncc >= args.ncc_accept:
                    found.append(best_name)
                else:
                    weak.append(
                        {"index": i, "best": best_name, "ncc": round(best_ncc, 4)}
                    )
            resolved.append(
                {
                    "path": row["path"],
                    "group_id": row["group_id"],
                    "source_family": row["source_family"],
                    "clip_count": count,
                    "declared_clips": json.loads(row.get("clips") or "[]"),
                    "identified": sorted(set(found)),
                    "identified_placements": len(found),
                    "unresolved": weak,
                }
            )
            print(
                f"  {row['path']}  선언 {count}  식별 {len(found)}  "
                f"미해결 {len(weak)}",
                flush=True,
            )
        report[pool_dir] = {"rows": resolved}

    out.parent.mkdir(parents=True, exist_ok=True)
    report["_unreadable_corpus"] = unreadable
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    total_declared = total_found = total_weak = 0
    for pool_dir, payload in (
        (k, v) for k, v in report.items() if not k.startswith("_")
    ):
        d = sum(r["clip_count"] for r in payload["rows"])
        f = sum(r["identified_placements"] for r in payload["rows"])
        w = sum(len(r["unresolved"]) for r in payload["rows"])
        total_declared += d
        total_found += f
        total_weak += w
        print(f"{pool_dir}: 선언 {d}  식별 {f}  미해결 {w}")
    print(f"합계: 선언 {total_declared}  식별 {total_found}  미해결 {total_weak}")
    print(f"→ {out}")
    return 0 if total_weak == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
