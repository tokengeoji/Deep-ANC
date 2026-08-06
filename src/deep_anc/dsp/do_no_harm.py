"""대역 밖 '악화 금지' 예산의 **단일 출처** — 손실 힌지와 G4 게이트를 잇는다.

왜 이 모듈이 있는가
------------------
2026-08-06 실측: **손실을 정확히 만족한 모델이 게이트를 8.5 dB 차이로 FAIL 한다.**

``configs/train_finetune.yaml`` 은 ``dnh_margin_db: 6.0`` 으로 "대역 밖에서 반노이즈가
교란보다 6 dB 커도 봐준다" 고 말하는데, ``eval/recorded.py`` 의 G4 게이트는
"옥타브 감쇠가 −1.0 dB 보다 나쁘면 FAIL" 이라고 말한다. 두 값은 서로를 모른 채
각자 적혀 있었고, 대조하는 코드도 테스트도 없었다 — [[pareto-defect-clustering]] 의
발생기 A(같은 물리량을 두 곳에서 따로 유도)가 손실과 게이트 사이에서 재현된 것이다.

직접 실행으로 확인한 두 개의 불일치 (``d`` 는 20 Hz–20 kHz 백색, N=131072, fs=48 kHz):

======================================  =========================================
사례                                    게이트가 본 최악 옥타브
======================================  =========================================
힌지를 **정확히** 만족 (6.0 dB, 동상)   4000 Hz **−9.53 dB**  (한계 −1.0)
힌지 만족 + 넓은 dnh 대역 안 에너지 집중  2000 Hz **−12.63 dB**
이 모듈이 유도한 마진 (−18.27 dB)       4000 Hz **−1.00 dB**  ← 정확히 포화
======================================  =========================================

즉 결함은 하나가 아니라 둘이다.

1. **임계 불일치** — 힌지는 ``bandpower(S·y)/bandpower(d)`` 를 보고 게이트는
   ``bandpower(e)/bandpower(d)`` 를 본다. ``e = d + S·y`` 이므로 같은 물리량이 아니다.
2. **대역 분할 불일치** — 힌지 대역 ``[1633, 6000]`` 은 게이트 옥타브 2000(1414–2828)과
   4000(2828–5657)을 **가로지른다**. 대역 전체 비율을 만족시키면서 에너지를 한 옥타브에
   몰아넣을 수 있고, 그러면 그 옥타브만 3.1 dB 더 나빠진다.

보장 (정리)
----------
``ρ = 10^(margin/10)`` 이라 하자. 어떤 dnh 대역 ``B`` 에서도 ``P_sy(B) ≤ ρ·P_d(B)`` 이고,
**모든 dnh 대역이 옥타브 경계를 가로지르지 않으면**, 옥타브 ``O`` 안에서

    P_e(O) = Σ_i P_e(B_i) ≤ Σ_i (√P_d(B_i) + √P_sy(B_i))²   (대역별 Cauchy–Schwarz)
                          ≤ (1+√ρ)² Σ_i P_d(B_i) = (1+√ρ)² P_d(O)

이므로 감쇠 ≥ ``−20·log10(1+√ρ)`` dB 다. 이것을 게이트 임계 ``G`` 와 같게 놓으면

    margin_db = 20·log10(10^(G/20) − 1)

가 나온다. **이 값이 게이트를 보장하는 가장 느슨한 마진**이고, 이보다 크면 보장이 없다.
분할 조건이 왜 필요한지가 유도에 그대로 보인다 — ``Σ_i P_e(B_i) = P_e(O)`` 는 ``B_i`` 가
``O`` 를 **분할**할 때만 성립한다.

남는 가정 (숨기지 않는다)
------------------------
옥타브 125(88–177 Hz)와 2000(1414–2828 Hz)은 보호 대역 경계를 걸친다. 그 옥타브의
보호 대역 쪽 조각에는 힌지가 걸리지 않으므로(단측 힌지를 개선 요구 대역에 겹치면 두 항이
서로 상쇄한다 — ``losses/config.py DoNoHarmPlan`` 참조), 위 보장은 **"보호 대역 조각에서
증폭하지 않는다"** 를 전제한다. 그 조각을 통째로 포함하는 옥타브 250·500·1000 은 G4 가
직접 판정하므로 사각지대는 걸친 두 옥타브의 슬라이버뿐이다.

왜 마진이 음수인 것이 이상하지 않은가
----------------------------------
"+6 dB" 는 "시뮬레이터의 |S| 오차를 봐준다" 는 뜻이었다. 그런데 **|S| 를 과소평가한 경우
실제 증폭은 손실이 아는 것보다 나쁘다.** 즉 플랜트 불확실성은 마진을 **좁히는** 방향으로
작용해야 한다. 완화 방향으로 쓴 것이 부호 오류였다. 그래서 이 모듈의
``plant_uncertainty_db`` 는 유도값에서 **빼기만** 한다.

핫패스 금지 — 경계에서만 부른다.
"""

from __future__ import annotations

import math
from typing import Sequence

__all__ = [
    "MAX_OUT_OF_BAND_AMPLIFICATION_DB",
    "OCTAVE_BAND_CENTERS_HZ",
    "gate_consistent_margin_db",
    "octave_band_edges_hz",
    "octave_boundary_edges_hz",
    "worst_case_amplification_db",
]


MAX_OUT_OF_BAND_AMPLIFICATION_DB = 1.0
"""옥타브 밴드 감쇠가 이보다 더 음수(=증폭)면 실패다. **절대목표 1의 게이트다.**

왜 fullband 평균으로는 안 되는가
--------------------------------
``fullband NMSE ≤ 0`` 은 대역 밖 증폭을 **원리적으로** 잡지 못한다. NMSE 는 ``d`` 의
에너지로 정규화되는데, ``d`` 에 에너지가 거의 없는 대역에서는 ``e`` 가 몇십 dB 커져도
전체 비율이 거의 안 변하기 때문이다. 실측 반증(results/session_20260804_0939)::

    tone300:  trusted +6.26 dB / fullband **+5.95 dB**   ← 둘 다 판정 기준을 만족
              band_1000 −16.84 / band_2000 −15.42 / band_4000 −18.03 / band_8000 **−21.56**

즉 8 kHz 를 21 dB 증폭하면서 G4 를 통과했다. 옥타브 감쇠는 ``octave_rows`` 로 이미
계산해 npz 에 **저장까지 하고 있었는데** 판정에는 한 번도 쓰이지 않았다.

임계 1.0 dB 의 뜻: "개선을 요구하지 않는다. 다만 해치지 마라." 신뢰 대역 밖은 상쇄
대상이 아니므로 0 dB 근처면 충분하고, 측정 잡음 여유로 1 dB 를 준다. 실제 결함은
15~22 dB 라 이 허용치의 15~22배다.

⚠ 이 상수는 **손실 힌지 마진의 유도원**이기도 하다(:func:`gate_consistent_margin_db`).
여기를 건드리면 손실이 따라 움직인다 — 그것이 의도다. 두 값이 따로 놀던 것이 결함이었다.
"""


OCTAVE_BAND_CENTERS_HZ: tuple[float, ...] = (
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    8000.0,
)
"""G4 가 판정하는 옥타브 중심주파수. ``configs/eval*.yaml octave_bands_hz`` 의 단일 출처.

설정 파일에 같은 목록이 세 벌 적혀 있었다(``eval.yaml`` / ``eval_demo.yaml`` /
``eval_live_demo.yaml``). 손실 대역을 이 경계에 정렬해야 하므로 이제는 **코드가 원본**이고,
설정이 어긋나면 ``tests/test_do_no_harm_contract.py`` 가 실패한다.
"""


def octave_band_edges_hz(center_hz: float) -> tuple[float, float]:
    """옥타브 밴드 경계 ``(fc/√2, fc·√2)``.

    ``eval/metrics.py`` 안에만 있던 한 줄을 끌어냈다. 손실 대역을 이 경계에 정렬하려면
    손실 쪽에서도 같은 식이 필요한데, 복붙하면 그것이 여섯 번째 복제가 된다.
    """

    if not math.isfinite(center_hz) or center_hz <= 0.0:
        raise ValueError(f"옥타브 중심주파수는 유한한 양수여야 합니다: {center_hz}")
    root2 = math.sqrt(2.0)
    return (center_hz / root2, center_hz * root2)


def octave_boundary_edges_hz(
    centers_hz: Sequence[float] = OCTAVE_BAND_CENTERS_HZ,
    *,
    nyquist_hz: float,
) -> tuple[float, ...]:
    """옥타브 경계를 정렬·중복제거해 돌려준다 (Nyquist 안쪽만).

    do-no-harm 대역을 자를 때 이 값들을 **반드시 절단점에 포함**해야 한다. 그래야 어떤
    do-no-harm 대역도 옥타브를 가로지르지 않고, 대역별 상한이 옥타브 상한으로 합쳐진다.
    """

    if not math.isfinite(nyquist_hz) or nyquist_hz <= 0.0:
        raise ValueError(f"Nyquist 주파수는 유한한 양수여야 합니다: {nyquist_hz}")
    edges: set[float] = set()
    for center in centers_hz:
        lo, hi = octave_band_edges_hz(float(center))
        for value in (lo, hi):
            if 0.0 < value < nyquist_hz:
                edges.add(round(value, 6))
    return tuple(sorted(edges))


def worst_case_amplification_db(margin_db: float) -> float:
    """마진 ``margin_db`` 인 힌지를 정확히 만족했을 때 **게이트가 볼 수 있는 최악 증폭**.

    ``20·log10(1 + 10^(margin/20))``. 위상이 동상으로 정렬된 경우(Cauchy–Schwarz 등호)다.
    실측으로 이 상한이 도달 가능함을 확인했다 — margin 6.0 → 4000 Hz 옥타브 −9.53 dB,
    이 식의 값도 9.53 dB.
    """

    if not math.isfinite(margin_db):
        raise ValueError(f"margin_db 는 유한해야 합니다: {margin_db}")
    return 20.0 * math.log10(1.0 + 10.0 ** (margin_db / 20.0))


def gate_consistent_margin_db(
    gate_db: float = MAX_OUT_OF_BAND_AMPLIFICATION_DB,
    *,
    plant_uncertainty_db: float = 0.0,
) -> float:
    """게이트 임계에서 **유도한** 힌지 마진 — 이것이 유일한 유도 경로다.

    ``20·log10(10^(gate/20) − 1) − plant_uncertainty_db``.

    ``gate_db = 1.0`` → **−18.27 dB**. 즉 대역 밖 반노이즈는 교란보다 18 dB 아래여야
    "옥타브를 1 dB 이상 키우지 않는다"가 **보장**된다. 이보다 느슨한 값은 보장이 아니라
    희망이고, 2026-08-06 이전의 +6.0 dB 는 희망이었다(실측 −9.53 dB).

    ``plant_uncertainty_db`` 는 |S| 추정 오차를 **좁히는 방향으로만** 반영한다 — 이유는
    모듈 docstring 참조. 기본 0.0 은 "손실이 쓰는 |S| 를 그대로 믿는다" 는 뜻이다.
    """

    if not math.isfinite(gate_db) or gate_db <= 0.0:
        raise ValueError(
            f"게이트 임계는 유한한 양수여야 합니다: {gate_db} — 0 이하면 어떤 대역 밖 "
            "출력도 허용되지 않아 마진이 −∞ 가 됩니다"
        )
    if not math.isfinite(plant_uncertainty_db) or plant_uncertainty_db < 0.0:
        raise ValueError(
            f"plant_uncertainty_db 는 유한한 0 이상이어야 합니다: {plant_uncertainty_db}"
        )
    amplitude = 10.0 ** (gate_db / 20.0) - 1.0
    return 20.0 * math.log10(amplitude) - float(plant_uncertainty_db)
