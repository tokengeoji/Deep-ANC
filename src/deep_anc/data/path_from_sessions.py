"""녹음된 세션에서 **음향 경로를 직접 추정**한다 — 스피커를 울리지 않고.

왜 이것이 필요한가
------------------
공식 P/S 아티팩트는 6초짜리 톤 프로브 한 번에서 나온다. 그런데 학습에 쓰는 실측
세션은 **84분치 광대역 신호**이고, 그것이 곧 학습 분포다. 같은 물리를 훨씬 많은
데이터로 다시 볼 수 있는데 아무도 보지 않았다.

2026-08-06 에 이것이 급해졌다. 앰프 배선을 바꾼 뒤 재측정한 P/S 가 공식 아티팩트와
최적 필터 −P/S 기준 **36.6% 다르게** 나왔는데(600–1000Hz 복소일치 0.78), 그것이
플랜트의 진짜 변화인지 측정 잡음인지 가릴 방법이 없었다. 재측정은 스피커 시간을 쓰고,
한 번 더 재도 어느 쪽이 맞는지 확정되지 않는다.

세션 녹음이 답을 갖고 있다: **8/4 에 받은 47세션과 8/6 배선 후 받은 25세션의 음향
경로를 비교하면 된다.** 둘이 같으면 플랜트는 안 바뀐 것이고, 다르면 바뀐 것이다.

무엇을 잴 수 있고 무엇을 못 재는가 (정직하게)
--------------------------------------------
녹음 중 **소음 스피커만** 울린다. 상쇄 스피커는 무음이다. 따라서

* ✅ **REF→ERR** 음향 경로 = ``P_err / P_ref``. 두 마이크가 같은 음장을 서로 다른
  지점에서 들으므로 이 비는 순수 음향이고, 재생 타임베이스가 **분자·분모에서 상쇄된다**.
* ⚠ ``P_err`` 자체는 절대 재생 지연을 알아야 하는데 그것은 재현되지 않는다
  (low-latency 1565~1659 / high 2858~2888, 드리프트 364~729 ppm). ``source_aligned.wav``
  는 REF 를 증인으로 삼아 워프한 것이라 source→REF 지연이 0 으로 맞춰져 있다.
* ❌ **S(z)(상쇄→ERR)는 못 잰다.** 그 스피커가 울린 적이 없다. S 는 인터리브
  프로브로만 얻는다.
* ❌ 상쇄→REF 되먹임 경로도 같은 이유로 못 잰다.

즉 이 모듈은 P/S 측정을 **대체하지 않는다.** 덕트 음향이 세션 사이에 변했는지를
독립적으로 판정하는 도구다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

__all__ = ["SessionPath", "estimate_ref_to_err", "compare_session_groups"]


@dataclass(frozen=True)
class SessionPath:
    """한 세션에서 뽑은 REF→ERR 전달함수."""

    session_id: str
    source_family: str
    freqs_hz: np.ndarray
    transfer: np.ndarray
    """복소 전달함수 ``H(f) = P_err(f)/P_ref(f)``."""

    coherence: np.ndarray
    """``|Sxy|²/(Sxx·Syy)``. 이 값이 낮은 빈은 평균에서 신뢰할 수 없다."""

    frames: int


def estimate_ref_to_err(
    session_dir: Path,
    *,
    nperseg: int = 8192,
    max_seconds: float = 60.0,
) -> SessionPath:
    """``mics.wav`` 의 REF(ch1) → ERR(ch0) 전달함수를 Welch 교차스펙트럼으로 뽑는다.

    소스 파일을 쓰지 않는다 — 두 마이크만 쓰면 재생 타임베이스가 아예 개입하지 않는다.
    이것이 이 추정을 워프·드리프트로부터 자유롭게 만든다.
    """

    import json

    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    with sf.SoundFile(session_dir / "mics.wav") as handle:
        rate = int(handle.samplerate)
        frames = min(handle.frames, int(max_seconds * rate))
        mics = handle.read(frames=frames, dtype="float64", always_2d=True)
    err, ref = mics[:, 0], mics[:, 1]

    freqs, pxx = signal.welch(ref, fs=rate, nperseg=nperseg, noverlap=nperseg // 2)
    _, pyy = signal.welch(err, fs=rate, nperseg=nperseg, noverlap=nperseg // 2)
    _, pxy = signal.csd(ref, err, fs=rate, nperseg=nperseg, noverlap=nperseg // 2)

    transfer = pxy / (pxx + 1e-30)
    coherence = np.abs(pxy) ** 2 / ((pxx * pyy) + 1e-30)
    return SessionPath(
        session_id=str(meta.get("session_id", session_dir.name)),
        source_family=str(meta.get("source_family", "?")),
        freqs_hz=freqs,
        transfer=transfer,
        coherence=np.clip(coherence, 0.0, 1.0),
        frames=int(frames),
    )


def _weighted_average(paths: list[SessionPath], min_coherence: float) -> tuple[np.ndarray, np.ndarray, int]:
    """코히런스로 가중한 복소 평균. 낮은 코히런스 빈은 스스로 빠진다."""

    if not paths:
        raise ValueError("세션이 없습니다")
    weights = np.stack([np.where(p.coherence >= min_coherence, p.coherence, 0.0) for p in paths])
    values = np.stack([p.transfer for p in paths])
    total = weights.sum(axis=0)
    mean = (weights * values).sum(axis=0) / (total + 1e-30)
    mean_coherence = np.stack([p.coherence for p in paths]).mean(axis=0)
    return mean, mean_coherence, len(paths)


def compare_session_groups(
    group_a: list[SessionPath],
    group_b: list[SessionPath],
    *,
    bands_hz: tuple[tuple[float, float], ...] = (
        (150.0, 300.0),
        (300.0, 600.0),
        (600.0, 1000.0),
        (1000.0, 1600.0),
    ),
    min_coherence: float = 0.5,
) -> list[dict]:
    """두 세션 집단의 REF→ERR 경로를 대역별로 비교한다.

    반환 항목마다 ``complex_agreement`` 와 ``gain_db`` 를 준다. 전자가 1 에 가까우면
    **형상이 같다**는 뜻이고, 후자는 공통 배율이다. 형상이 같은데 배율만 다르면 그것은
    앰프 노브이지 음향 변화가 아니다.
    """

    mean_a, coh_a, n_a = _weighted_average(group_a, min_coherence)
    mean_b, coh_b, n_b = _weighted_average(group_b, min_coherence)
    freqs = group_a[0].freqs_hz

    rows: list[dict] = []
    for lo, hi in bands_hz:
        mask = (freqs >= lo) & (freqs < hi)
        a, b = mean_a[mask], mean_b[mask]
        denom = np.sqrt((np.abs(a) ** 2).sum() * (np.abs(b) ** 2).sum()) + 1e-30
        agreement = float(np.abs((a * np.conj(b)).sum()) / denom)
        gain = (a * np.conj(b)).sum() / ((np.abs(a) ** 2).sum() + 1e-30)
        rows.append(
            {
                "band_hz": (float(lo), float(hi)),
                "complex_agreement": agreement,
                "gain_db": float(20.0 * np.log10(np.abs(gain) + 1e-30)),
                "relative_error": float(
                    np.abs(b - a).sum() / (np.abs(a).sum() + 1e-30)
                ),
                "coherence_a": float(coh_a[mask].mean()),
                "coherence_b": float(coh_b[mask].mean()),
                "sessions_a": n_a,
                "sessions_b": n_b,
            }
        )
    return rows
