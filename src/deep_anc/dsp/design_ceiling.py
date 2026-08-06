"""최적 인과 FIR 이 낼 수 있는 상쇄량의 **상한**을 실측 P/S 에서 직접 푼다.

왜 이 모듈이 있는가
------------------
2026-08-06 감사가 재현한 fail-open: ``readiness.measured_design_ceiling_db`` 는 사람이
한 번 계산해 **설정에 적어 둔 숫자**였고, 게이트는 그것을 그대로 믿었다. 실제로
``--set readiness.measured_design_ceiling_db=30.0`` 으로 날조해도 게이트가 PASS 한다
(직접 확인: 구속 상한이 "플랜트 일관성 27.73 dB" 로 폴백하며 통과). 100.0 으로 해도 같다.

즉 **모델이 물리적으로 도달할 수 없는 목표를 세워도 진입 게이트가 막지 못했다.**
배선을 고쳐 P/S 를 다시 측정하면 아티팩트 sha 는 바뀌지만 설정의 4.58 은 그대로 남아
계속 통과한다 — 게이트가 자기 자신을 증명하는 구조였다.

이제 게이트는 이 모듈을 불러 **아티팩트에서 다시 푼다.** 선언값은 대조 대상이다.

무엇을 푸는가
------------
digital-reference 구성에서 ``x_ref[t] = n[t+K]`` (lead K), ``d = P ⊛ n`` 이므로
반노이즈 ``y = w ⊛ x_ref`` 를 지나온 오차는

    e = (p + advance_K(s ⊛ w)) ⊛ n

이다. 양쪽을 K 만큼 밀면 목표는 ``s ⊛ w ≈ −delay_K(p)`` 인 표준 최소제곱이 되고,
``w`` 를 길이 M 의 **인과** FIR 로 제한한 해가 곧 설계 상한이다.

조건수에 대하여 (숨기지 않는다)
------------------------------
대역을 벽돌담(brick-wall)으로 자르면 정규방정식이 특이해진다 — 실측 유효 랭크가
2048 중 약 124 이고, 정규화 없이 풀면 ``|w|max ≈ 6.7e+141`` 이 나온다. 그 해는 수치
쓰레기이지 물리적 상한이 아니다. 그래서 여기서는

1. 대역 제한을 **Butterworth 대역통과**로 걸어 전이대역을 남기고,
2. Tikhonov 정규화 ``λ‖w‖²`` 를 걸며,
3. λ 를 바꿔 가며 **상한이 안정한 구간**을 보고한다.

정규화는 상한을 **낮추는** 방향으로만 작용하므로(제약을 더 거는 것이므로) 여기서 나온
값은 진짜 상한의 하계다. 게이트 판정에는 그것이 안전한 방향이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import linalg, signal

__all__ = [
    "DesignCeiling",
    "cached_design_ceiling_db",
    "constrained_design_ceiling_db",
    "design_ceiling_db",
    "worst_octave_ceiling_db",
]


@dataclass(frozen=True)
class DesignCeiling:
    """설계 상한 계산 결과. 수치와 **그것을 만든 조건**을 함께 들고 다닌다."""

    ceiling_db: float
    """대역 내 달성 가능 상쇄량 (양수 = 개선)."""

    band_hz: tuple[float, float]
    taps: int
    lead_samples: int
    regularisation: float
    condition_number: float
    max_tap: float
    """해의 최대 탭 크기. 1e3 을 넘으면 정규화가 부족하다는 신호다."""

    stable_over_regularisation: bool
    """수치적으로 믿을 만한 해인가.

    두 가지를 **동시에** 요구한다: (a) λ 를 10배 위아래로 흔들어도 상한이 0.2 dB 안에서
    유지되고, (b) 최대 탭이 1e3 미만이다. (b) 가 없으면 놓친다 — S 를 1e-9 배로 줄인
    장난감 입력에서 이 함수는 ``|w|max = 1.4e9`` 로 31.6 dB 를 "달성" 했고 (a) 만으로는
    안정하다고 말했다. 그 해는 리미터에 닿는 순간 없는 것이 된다."""


def _bandlimit(x: np.ndarray, band_hz: tuple[float, float], sample_rate: float) -> np.ndarray:
    lo, hi = float(band_hz[0]), float(band_hz[1])
    nyquist = float(sample_rate) / 2.0
    hi = min(hi, nyquist * 0.99)
    sos = signal.butter(4, [lo / nyquist, hi / nyquist], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def _solve(target: np.ndarray, s: np.ndarray, taps: int, ridge: float):
    """``s ⊛ w ≈ target`` 을 길이 ``taps`` 인과 FIR 로 푼다."""

    n = target.size
    columns = np.zeros((n, taps), dtype=np.float64)
    for k in range(taps):
        columns[k:, k] = s[: n - k]
    gram = columns.T @ columns
    scale = float(np.trace(gram)) / max(1, taps)
    gram_reg = gram + ridge * scale * np.eye(taps)
    rhs = columns.T @ target
    w = linalg.solve(gram_reg, rhs, assume_a="pos")
    residual = target - columns @ w
    cond = float(np.linalg.cond(gram_reg))
    return w, residual, cond


def design_ceiling_db(
    primary_fir: np.ndarray,
    secondary_fir: np.ndarray,
    *,
    lead_samples: int,
    band_hz: tuple[float, float],
    sample_rate: float,
    taps: int = 2048,
    regularisation: float = 1.0e-6,
) -> DesignCeiling:
    """실측 P/S 에서 최적 인과 FIR 상한을 푼다.

    반환값 ``ceiling_db`` 는 **양수일수록 좋다** (대역 내 잔차/교란 에너지비의 −10log10).
    """

    p = np.asarray(primary_fir, dtype=np.float64).reshape(-1)
    s = np.asarray(secondary_fir, dtype=np.float64).reshape(-1)
    if p.size == 0 or s.size == 0:
        raise ValueError("P/S FIR 이 비어 있습니다")
    if taps < 16:
        raise ValueError(f"taps 는 16 이상이어야 합니다: {taps}")

    length = int(2 ** math.ceil(math.log2(p.size + s.size + taps + int(lead_samples) + 16)))
    p_pad = np.zeros(length)
    p_pad[: p.size] = p
    s_pad = np.zeros(length)
    s_pad[: s.size] = s

    # lead 만큼 P 를 뒤로 민다 — x_ref 가 미래를 보는 만큼 w 가 여유를 얻는다.
    lead = int(lead_samples)
    if lead > 0:
        p_pad = np.concatenate([np.zeros(lead), p_pad])[:length]

    p_band = _bandlimit(p_pad, band_hz, sample_rate)
    s_band = _bandlimit(s_pad, band_hz, sample_rate)

    target = -p_band
    disturbance = float(np.sum(p_band**2))
    if disturbance <= 0.0:
        raise ValueError("대역 내 교란 에너지가 0 입니다 — 상한을 정의할 수 없습니다")

    results = {}
    for ridge in (regularisation / 10.0, regularisation, regularisation * 10.0):
        w, residual, cond = _solve(target, s_band, taps, ridge)
        value = -10.0 * math.log10(float(np.sum(residual**2)) / disturbance)
        results[ridge] = (value, float(np.abs(w).max()), cond)

    ceiling, max_tap, cond = results[regularisation]
    spread = max(v[0] for v in results.values()) - min(v[0] for v in results.values())
    numerically_sane = max_tap < 1.0e3
    return DesignCeiling(
        ceiling_db=float(ceiling),
        band_hz=(float(band_hz[0]), float(band_hz[1])),
        taps=int(taps),
        lead_samples=lead,
        regularisation=float(regularisation),
        condition_number=cond,
        max_tap=max_tap,
        stable_over_regularisation=bool(spread <= 0.2 and numerically_sane),
    )


def constrained_design_ceiling_db(
    primary_fir: np.ndarray,
    secondary_fir: np.ndarray,
    *,
    lead_samples: int,
    band_hz: tuple[float, float],
    sample_rate: float,
    max_out_of_band_amplification_db: float,
    taps: int = 2048,
    regularisation: float = 1.0e-6,
) -> tuple[float, float]:
    """**대역 밖을 해치지 않는다는 제약 아래** 달성 가능한 대역 내 상한.

    이것이 게이트가 실제로 비교해야 할 값이다. 무제약 상한은 "대역 밖을 얼마든지
    증폭해도 좋다면" 의 값이고, G4 는 그것을 금지한다. 2026-08-06 감사가 이 차이를
    지적했고 세 에이전트가 독립적으로 재현했다.

    ``min ‖e_in‖² + μ‖e_out‖²`` 를 μ 를 훑으며 풀고, 대역 밖 증폭이 허용치에 닿는
    지점의 대역 내 값을 돌려준다. 반환: ``(대역내 상한 dB, 그때의 대역밖 증폭 dB)``.
    """

    p = np.asarray(primary_fir, dtype=np.float64).reshape(-1)
    s = np.asarray(secondary_fir, dtype=np.float64).reshape(-1)
    length = int(2 ** math.ceil(math.log2(p.size + s.size + taps + int(lead_samples) + 16)))
    p_pad = np.zeros(length)
    p_pad[: p.size] = p
    s_pad = np.zeros(length)
    s_pad[: s.size] = s
    lead = int(lead_samples)
    if lead > 0:
        p_pad = np.concatenate([np.zeros(lead), p_pad])[:length]

    nyquist = float(sample_rate) / 2.0
    lo, hi = float(band_hz[0]), min(float(band_hz[1]), nyquist * 0.99)
    sos_in = signal.butter(4, [lo / nyquist, hi / nyquist], btype="bandpass", output="sos")

    def split(x):
        inside = signal.sosfiltfilt(sos_in, x)
        return inside, x - inside

    p_in, p_out = split(p_pad)
    s_in, s_out = split(s_pad)

    def columns(kernel):
        n = kernel.size
        out = np.zeros((n, taps), dtype=np.float64)
        for k in range(taps):
            out[k:, k] = kernel[: n - k]
        return out

    a_in, a_out = columns(s_in), columns(s_out)
    g_in, g_out = a_in.T @ a_in, a_out.T @ a_out
    b_in, b_out = a_in.T @ (-p_in), a_out.T @ (-p_out)
    scale = float(np.trace(g_in)) / max(1, taps)
    e_in0 = float(np.sum(p_in**2))
    e_out0 = float(np.sum(p_out**2))

    def evaluate(mu: float) -> tuple[float, float]:
        gram = g_in + mu * g_out + regularisation * scale * np.eye(taps)
        w = linalg.solve(gram, b_in + mu * b_out, assume_a="pos")
        r_in = p_in + a_in @ w
        r_out = p_out + a_out @ w
        return (
            -10.0 * math.log10(float(np.sum(r_in**2)) / e_in0),
            10.0 * math.log10(float(np.sum(r_out**2)) / e_out0),
        )

    # μ 가 커질수록 대역 밖 증폭이 줄고 대역 내 상한도 줄어든다 — 단조이므로 이분한다.
    # 전수 훑기(41점)는 2048×2048 풀이가 41번이라 2분을 넘겼다.
    limit = float(max_out_of_band_amplification_db)
    lo_mu, hi_mu = 1.0e-4, 1.0e6
    if evaluate(hi_mu)[1] > limit:
        raise ValueError(
            "대역 밖 제약을 만족하는 해를 찾지 못했습니다 — μ 훑기 범위를 넓히세요"
        )
    best = evaluate(hi_mu)
    for _ in range(12):
        mid = math.sqrt(lo_mu * hi_mu)
        value, gain_out = evaluate(mid)
        if gain_out <= limit:
            best = (value, gain_out)
            hi_mu = mid
        else:
            lo_mu = mid
    return best


def cached_design_ceiling_db(
    primary_path: "Path",
    secondary_path: "Path",
    *,
    lead_samples: int,
    band_hz: tuple[float, float],
    sample_rate: float,
    taps: int = 2048,
    cache_path: "Path | None" = None,
) -> DesignCeiling:
    """아티팩트 sha256 을 키로 캐시한다. **P/S 를 다시 재면 자동으로 무효화된다.**

    2048 탭 정규방정식은 게이트마다 풀기에는 비싸다(수십 초). 그렇다고 값을 설정에
    적어 두면 그것이 바로 이번에 고친 fail-open 이다. 캐시 키에 아티팩트 sha 가 들어가
    있으므로, 배선을 고쳐 P/S 를 다시 재는 순간 캐시가 빗나가고 다시 풀린다.
    """

    import hashlib
    import json
    from pathlib import Path

    from ..dsp.secondary_path import load_secondary_path

    primary_path = Path(primary_path)
    secondary_path = Path(secondary_path)

    def sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    key = "|".join(
        [
            sha(primary_path),
            sha(secondary_path),
            str(int(lead_samples)),
            f"{float(band_hz[0]):.3f}-{float(band_hz[1]):.3f}",
            str(int(taps)),
            f"{float(sample_rate):.1f}",
        ]
    )
    cache_path = Path(cache_path) if cache_path is not None else (
        primary_path.parent / ".design_ceiling_cache.json"
    )
    store: dict = {}
    if cache_path.is_file():
        try:
            store = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            store = {}
    hit = store.get(key)
    if isinstance(hit, dict) and "ceiling_db" in hit:
        return DesignCeiling(
            ceiling_db=float(hit["ceiling_db"]),
            band_hz=(float(band_hz[0]), float(band_hz[1])),
            taps=int(taps),
            lead_samples=int(lead_samples),
            regularisation=float(hit.get("regularisation", 1.0e-6)),
            condition_number=float(hit.get("condition_number", float("nan"))),
            max_tap=float(hit.get("max_tap", float("nan"))),
            stable_over_regularisation=bool(hit.get("stable", True)),
        )

    result = design_ceiling_db(
        load_secondary_path(primary_path).fir,
        load_secondary_path(secondary_path).fir,
        lead_samples=int(lead_samples),
        band_hz=band_hz,
        sample_rate=sample_rate,
        taps=taps,
    )
    store[key] = {
        "ceiling_db": result.ceiling_db,
        "regularisation": result.regularisation,
        "condition_number": result.condition_number,
        "max_tap": result.max_tap,
        "stable": result.stable_over_regularisation,
        "primary": str(primary_path),
        "secondary": str(secondary_path),
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(cache_path)
    except OSError:
        pass
    return result


def worst_octave_ceiling_db(
    primary_path: "Path",
    secondary_path: "Path",
    *,
    lead_samples: int,
    band_hz: tuple[float, float],
    sample_rate: float,
    taps: int = 2048,
    cache_path: "Path | None" = None,
) -> tuple[float, float]:
    """제어 대역 안 **각 옥타브에서** 푼 상한 중 최악값. 반환 ``(dB, 중심주파수)``.

    왜 대역평균이 아니라 옥타브인가
    ------------------------------
    절대목표 1(저역과 고역을 **모두** 제거)의 평가 게이트 G4 는 **옥타브별**로 판정한다.
    그런데 진입 게이트는 대역평균 상한(에너지가중)만 보고 있었다. 두 값이 크게 다르다 —
    실측(official P1602/S1462, lead 116)::

        전대역 [150, 1600]          +4.827 dB
        옥타브  125 [150,  176.8]  +19.891
        옥타브  250 [176.8, 353.6] +19.648
        옥타브  500 [353.6, 707.1]  **+2.159**   ← 최악
        옥타브 1000 [707.1,1414.2]  +5.223
        옥타브 2000 [1414.2,1600]   +7.384

    대역평균 4.83 은 저역의 큰 여유가 중역의 병목을 가린 값이다. 평균으로 판정하면
    "목표 1.0 + 여유 3.0 = 4.0 을 만족한다"가 되지만, 실제로 옥타브 500 에서는 2.159 뿐이라
    여유가 1.159 밖에 없다. **평균이 최악값을 가리는 것**이 이 저장소가 반복해서 겪은
    실패 형태이고, 여기서도 같은 형태였다.

    옥타브 500 이 유독 낮은 이유는 널 하나가 깊어서가 아니다. 그 옥타브를 반으로 쪼개면
    353.6–500 은 5.47, 500–707.1 은 5.97 dB 다 — **한 옥타브 안에서 인과 FIR 이 서로
    모순되는 등화를 요구받는다.** FIR 길이와 정규화 전 범위에서 안정하므로 학습이나
    정규화로 풀리지 않는다.
    """

    from pathlib import Path as _Path

    from .do_no_harm import OCTAVE_BAND_CENTERS_HZ, octave_band_edges_hz

    lo_band, hi_band = float(band_hz[0]), float(band_hz[1])
    worst: tuple[float, float] | None = None
    for center in OCTAVE_BAND_CENTERS_HZ:
        lo, hi = octave_band_edges_hz(float(center))
        lo, hi = max(lo, lo_band), min(hi, hi_band)
        if hi - lo < 20.0:
            continue
        solved = cached_design_ceiling_db(
            _Path(primary_path),
            _Path(secondary_path),
            lead_samples=lead_samples,
            band_hz=(lo, hi),
            sample_rate=sample_rate,
            taps=taps,
            cache_path=cache_path,
        )
        if worst is None or solved.ceiling_db < worst[0]:
            worst = (float(solved.ceiling_db), float(center))
    if worst is None:
        raise ValueError(
            f"제어 대역 [{lo_band:g}, {hi_band:g}] 안에 옥타브가 없습니다 — "
            "상한을 옥타브별로 판정할 수 없습니다"
        )
    return worst
