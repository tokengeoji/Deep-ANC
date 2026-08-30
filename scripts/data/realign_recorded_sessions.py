#!/usr/bin/env python3
"""기존 실측 세션의 시간축을 REF 증인으로 되감아 ``source_aligned.wav`` 를 만든다.

  .venv/bin/python scripts/data/realign_recorded_sessions.py --self-test
  .venv/bin/python scripts/data/realign_recorded_sessions.py --root data/recorded \
      --report results/timeline/realign.json
  .venv/bin/python scripts/data/realign_recorded_sessions.py --root data/recorded --dry-run

무엇을 하는가
------------
``source.wav`` 는 재생 **배열**이지 방출 **시각**이 아니다. AB13X USB DAC 의 UAC1
ADAPTIVE PLL 헌팅(주기 4~5초, 진폭 259~407 샘플) 때문에 둘의 대응이 시간에 따라
흔들리고, ``record_duct.py`` 는 그 대응을 측정하지 않고 인덱스 동일성으로 **단언**했다.

REF 마이크는 ERR 과 같은 ADC 를 타므로 재생 신호의 시간축 증인이다. 이 스크립트는
``source → REF`` 로 L(t) 를 추정해 ``source_aligned[t] = source(t − L(t))`` 를 만들고,
**추정에 쓰지 않은 ERR 채널로만** 검증한다(홀드아웃).

절대 규칙
--------
* ``source.wav`` 는 **절대 덮어쓰지 않는다.** 원본 provenance 가 사라지면 워프
  알고리즘을 개선했을 때 재생성 여부를 판정할 수 없다.
* 기준 미달 세션은 ``source_aligned.wav`` 를 쓰지 않는다(실패-폐쇄). 리포트에는
  왜 떨어졌는지 숫자로 남긴다 — "검사했는데 통과 못 했다"와 "검사하지 않았다"를
  구분할 수 없으면 그게 다음 사고다.

알고리즘·임계값의 단일 출처는 ``src/deep_anc/data/timeline.py`` 다. 여기서 다시
유도하지 않는다.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT                          # noqa: E402
from deep_anc.dsp.invariants import (                          # noqa: E402
    MIN_STREAM_DELAY_VALID_WINDOW_RATIO,
)
from deep_anc.data.timeline import (                           # noqa: E402
    TIMELINE_METHOD,
    TimelineSettings,
    align_source_to_adc,
    estimate_lag_track,
    warp_by_lag_track,
)

# 게이트 기본값 — CLI 는 강화 방향으로만 받는다.
DEFAULT_MIN_COHERENCE = 0.90
DEFAULT_MIN_VALID_WINDOW_RATIO = 0.90

# 같은 물리량(지연궤적 유효창 비율)에 임계가 두 곳에 선언돼 있다. 두 값이 다른 것은
# 의도된 것이다 — 여기는 **재정렬본을 만들 때** 요구하는 값이고(0.90), invariants 쪽은
# **이미 만들어진 세션을 학습에 들일 때** 요구하는 바닥이다(0.77). 그러나 2026-08-06
# 통합 검증 전까지 둘 사이에 대조가 없어서 바닥 쪽이 0.50 까지 내려가 있어도 아무도
# 몰랐고, 실제로 그 틈으로 프레임 슬립 세션이 통과했다. 순서가 뒤집히면 느슨한 쪽이
# 학습 데이터를 지키게 되므로 import 시점에 못 박는다.
if DEFAULT_MIN_VALID_WINDOW_RATIO < MIN_STREAM_DELAY_VALID_WINDOW_RATIO:
    raise AssertionError(
        "재정렬 게이트가 QA 바닥보다 느슨합니다: "
        f"realign {DEFAULT_MIN_VALID_WINDOW_RATIO} < QA {MIN_STREAM_DELAY_VALID_WINDOW_RATIO}. "
        "재정렬은 학습 진입보다 항상 같거나 엄격해야 합니다"
    )


def self_test(sample_rate: int = 48000, seconds: float = 8.0) -> int:
    """알려진 L(t) 를 주입한 합성 신호로 추정기의 왕복 정확도를 잰다.

    하드웨어 없이 도는 유일한 정직한 검증이다. 실제 세션에서 되찾은 coh² 가 높다는
    것만으로는 "L(t) 를 맞게 추정했다"를 증명하지 못한다 — 정답을 아는 신호로 재야 한다.
    """

    from scipy.signal import butter, lfilter

    fs = int(sample_rate)
    n = int(seconds * fs)
    pad = 4000
    rng = np.random.default_rng(20260805)
    noise = rng.standard_normal(n + 2 * pad)
    b, a = butter(4, [100.0 / (fs / 2), 1600.0 / (fs / 2)], btype="band")
    wide = lfilter(b, a, noise)

    t = np.arange(n, dtype=np.float64)
    lag_true = 1500.0 + 134.0 * np.sin(2.0 * np.pi * t / fs / 4.5)
    pos = t + pad - lag_true
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    witness = (1.0 - frac) * wide[i0] + frac * wide[i0 + 1]
    acoustic_delay = 142
    holdout = np.concatenate([np.zeros(acoustic_delay), witness[:-acoustic_delay]])
    source = wide[pad : pad + n]

    settings = TimelineSettings(sample_rate=fs)
    track = estimate_lag_track(source, witness, settings)
    centres = (np.asarray(track.times_s) * fs).astype(np.int64)
    valid = np.asarray(track.valid, dtype=bool)
    estimated = np.asarray(track.lag_samples, dtype=np.float64)[valid]
    truth = 1500.0 + 134.0 * np.sin(2.0 * np.pi * centres[valid] / fs / 4.5)
    max_error = float(np.max(np.abs(estimated - truth)))

    aligned, report = align_source_to_adc(source, witness, holdout, fs, settings=settings)
    del aligned

    print("[self-test] 주입 L(t) = 1500 + 134·sin(2π t / 4.5s)")
    print(
        f"  추정 최대 오차 {max_error:.2f} 샘플 (유효창 {track.valid_window_ratio:.3f}, "
        f"창 {centres.size}개)"
    )
    print(
        f"  coh²(150-600Hz) {report.coh2_150_600_before:.3f} → "
        f"{report.coh2_150_600_after:.3f}"
    )
    print(
        f"  잔여 지연 중앙 {report.aligned_lag_median_samples:.2f} "
        f"(주입 음향지연 {acoustic_delay}), robust-std "
        f"{report.aligned_lag_robust_std_samples:.2f} 샘플"
    )

    ok = True
    # 창 0.25s 안에서 L(t) 가 최대 47 샘플 움직이는 극단 조건이라 창 평균 오차가 남는다.
    # 실측 세션의 변조 진폭은 이보다 3~5배 작다. 판정은 "워프 후 잔여" 로 한다.
    if max_error > 4.0:
        print(f"  [FAIL] 추정 오차 {max_error:.2f} > 4.0 샘플", file=sys.stderr)
        ok = False
    if report.coh2_150_600_after < 0.95:
        print(f"  [FAIL] 재정렬 후 coh² {report.coh2_150_600_after:.3f} < 0.95", file=sys.stderr)
        ok = False
    if abs(report.aligned_lag_median_samples - acoustic_delay) > 2.0:
        print("  [FAIL] 잔여 지연 중앙값이 주입 음향지연과 2 샘플 넘게 다릅니다", file=sys.stderr)
        ok = False
    if report.aligned_lag_robust_std_samples > 2.0:
        print(
            f"  [FAIL] 잔여 robust-std {report.aligned_lag_robust_std_samples:.2f} > 2.0",
            file=sys.stderr,
        )
        ok = False
    print("[self-test] " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def realign_session(
    session_dir: Path,
    *,
    settings: TimelineSettings,
    min_coherence: float,
    min_valid_window_ratio: float,
    write: bool,
) -> dict:
    mics_path = session_dir / "mics.wav"
    source_path = session_dir / "source.wav"
    result: dict = {"session": session_dir.name, "path": str(session_dir)}
    if not mics_path.is_file() or not source_path.is_file():
        result.update(ok=False, error="mics.wav 또는 source.wav 가 없습니다")
        return result

    mics, sr = sf.read(str(mics_path), dtype="float32", always_2d=True)
    source, source_sr = sf.read(str(source_path), dtype="float32", always_2d=True)
    if sr != settings.sample_rate or source_sr != settings.sample_rate:
        result.update(ok=False, error=f"샘플레이트 불일치: {sr}/{source_sr}")
        return result
    if mics.shape[1] < 2:
        result.update(ok=False, error="mics.wav 가 2채널이 아닙니다")
        return result

    started = time.time()
    aligned, report = align_source_to_adc(
        source[:, 0], mics[:, 1], mics[:, 0], settings.sample_rate, settings=settings
    )
    result["seconds"] = round(time.time() - started, 2)
    result["timeline"] = report.as_metadata()
    passed = (
        report.coh2_150_600_after >= min_coherence
        and report.valid_window_ratio >= min_valid_window_ratio
    )
    result["ok"] = bool(passed)
    if not passed:
        result["error"] = (
            f"coh² {report.coh2_150_600_after:.3f} (하한 {min_coherence}), "
            f"유효창 {report.valid_window_ratio:.3f} (하한 {min_valid_window_ratio})"
        )
        return result

    if write:
        # source.wav 는 건드리지 않는다.
        sf.write(str(session_dir / "source_aligned.wav"), aligned, settings.sample_rate,
                 subtype="FLOAT")
        meta_path = session_dir / "session.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["timeline"] = dict(report.as_metadata())
            meta["timeline"]["usable_for_digital_reference"] = True
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        result["written"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/recorded")
    parser.add_argument("--report", default=None, help="JSON 리포트 저장 경로")
    parser.add_argument("--dry-run", action="store_true", help="판정만 하고 파일을 쓰지 않는다")
    parser.add_argument("--self-test", action="store_true", help="합성 왕복 검증만 수행")
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N개 세션만 (진단용)")
    parser.add_argument("--min-coherence", type=float, default=DEFAULT_MIN_COHERENCE)
    parser.add_argument(
        "--min-valid-window-ratio", type=float, default=DEFAULT_MIN_VALID_WINDOW_RATIO
    )
    args = parser.parse_args()

    if args.min_coherence < DEFAULT_MIN_COHERENCE:
        parser.error("--min-coherence 는 게이트 강화 방향으로만 조정할 수 있습니다")
    if args.min_valid_window_ratio < DEFAULT_MIN_VALID_WINDOW_RATIO:
        parser.error("--min-valid-window-ratio 는 게이트 강화 방향으로만 조정할 수 있습니다")

    if args.self_test:
        return self_test()

    root = Path(args.root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    sessions = sorted(p for p in root.iterdir() if p.is_dir() and (p / "mics.wav").is_file())
    if args.limit:
        sessions = sessions[: args.limit]
    if not sessions:
        print(f"세션이 없습니다: {root}", file=sys.stderr)
        return 1

    settings = TimelineSettings(sample_rate=48000)
    results = []
    for index, session_dir in enumerate(sessions, start=1):
        item = realign_session(
            session_dir,
            settings=settings,
            min_coherence=args.min_coherence,
            min_valid_window_ratio=args.min_valid_window_ratio,
            write=not args.dry_run,
        )
        results.append(item)
        timeline = item.get("timeline", {})
        print(
            f"[{index:3d}/{len(sessions)}] {item['session']:<28} "
            f"{'PASS' if item.get('ok') else 'FAIL'} "
            f"coh² {timeline.get('coh2_150_600_before', float('nan')):.3f}→"
            f"{timeline.get('coh2_150_600_after', float('nan')):.3f} "
            f"유효창 {timeline.get('valid_window_ratio', float('nan')):.3f} "
            f"잔여 {timeline.get('aligned_lag_median_samples', float('nan')):.1f}"
            f"±{timeline.get('aligned_lag_robust_std_samples', float('nan')):.1f}",
            flush=True,
        )

    passed = [item for item in results if item.get("ok")]
    medians = [
        item["timeline"]["aligned_lag_median_samples"]
        for item in passed
        if "timeline" in item
    ]
    summary = {
        "method": TIMELINE_METHOD,
        "root": str(root),
        "sessions": len(results),
        "passed": len(passed),
        "failed": len(results) - len(passed),
        "pass_ratio": len(passed) / max(1, len(results)),
        "session_lag_median_spread_samples": (
            float(max(medians) - min(medians)) if medians else float("nan")
        ),
        "dry_run": bool(args.dry_run),
        "min_coherence": float(args.min_coherence),
        "min_valid_window_ratio": float(args.min_valid_window_ratio),
    }
    print(
        f"\n합계: {summary['passed']}/{summary['sessions']} PASS "
        f"({100.0 * summary['pass_ratio']:.1f}%), 세션간 잔여지연 중앙값 산포 "
        f"{summary['session_lag_median_spread_samples']:.2f} 샘플"
    )
    if args.report:
        report_path = Path(args.report)
        if not report_path.is_absolute():
            report_path = REPO_ROOT / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"summary": summary, "sessions": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"리포트: {report_path}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
