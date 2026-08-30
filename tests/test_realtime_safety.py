"""실시간 워치독 — **발동한다**와 **정상 신호에서 발동하지 않는다**를 둘 다 증명한다.

왜 이 파일이 있는가
------------------
2026-08-06 감사에서 확정된 6건은 증상이 달라도 발생기가 둘뿐이었다.

* **발생기 B (실패해본 적 없는 게이트)** — 워치독이 "감시 중"이라고 주장하는데
  그 주장이 반증된 적이 없다. 클립 워치독(S1)은 모델의 tanh 리미터와 **같은 값**을
  재고 있어 clip_fraction 이 구조적으로 항상 0 이었다. 발산 워치독(S3)은 베이스라인이
  0 이면 조용히 비활성이었다. NaN(S6)은 메시지 없이 삼켜졌다.
* **발생기 A (두 도메인 간 시간 정렬 부기)** — 같은 물리량을 두 곳에서 따로 유도하고
  대조하지 않는다. 출력 백로그는 1 hop 인데 입력 백로그는 8 hop 이었고, 어느 쪽도
  학습 플랜트의 ``handoff_extra_samples`` 와 대조되지 않았다.

그 1차 대응은 **발동만** 증명했고, 그래서 다음 감사에서 두 개가 정상 운용을 깼다.

* **R1** 정숙 상태에서 잡은 베이스라인 + 외부 소음 ON = 123 ms 만에 자동 OFF.
* **R2** DC 워치독이 80 Hz 영평균 상쇄음을 DC 로 오인 (목표 대역 안이다).
  당시 오기각 방지 테스트는 375 Hz(블록당 정확히 2주기) **하나**만 썼다 —
  이 실패를 구조적으로 피해 가는 유일한 주파수다.

그래서 이 파일의 계약은 **두 개**다:

1. :class:`WatchdogId` 에 이름이 있으면 그것을 **발동시키는** 시나리오가 있어야 한다
   (``test_every_watchdog_can_be_made_to_fire``).
2. 같은 이름에 대해 **정상 신호에서 발동하지 않는다**는 시나리오도 있어야 하고,
   그 시나리오는 **목표 대역 80–1600 Hz 전체**를 리미터가 허용하는 진폭까지 몰아야
   한다 (``test_every_watchdog_has_a_band_wide_false_positive_scenario``).
   단일 주파수 오기각 테스트는 회피 가능하므로 이 파일에서 금지다.

실기에서 오디오를 내지 않는다 — 전부 콜백/감시자 직접 호출이다.
"""

from __future__ import annotations

import math
import re
import types

import numpy as np
import pytest
from scipy import signal as sp_signal

from deep_anc.config import REPO_ROOT
from deep_anc.dsp.filters import DCBlocker
from deep_anc.ops.gate_registry import GATES, gate
from deep_anc.realtime.noise_gen import DigitalReferenceBuffer, NoiseProgram
from deep_anc.realtime.ring_buffer import SPSCRing
from deep_anc.realtime.run_realtime import RealtimeANC
from deep_anc.realtime.safety import (
    HARDWARE_PROTECTION_WATCHDOGS,
    BaselineTracker,
    BlockObservation,
    BlockVerdict,
    FadeGate,
    OnePoleLowPass,
    PipelineHandoffBudget,
    PowerEMA,
    RateWindow,
    SafetyLimits,
    SafetySupervisor,
    WatchdogId,
    _BUDGET_TOKEN,
)
from deep_anc.realtime.ui import RuntimeState

FS = 48_000
BLOCK = 256
DUCT = {"secondary_path": {"handoff_extra_samples": BLOCK}}

# duct.yaml realistic_target_band_hz = [80, 1600]. 이 대역 안의 어떤 상쇄음도
# 워치독을 건드리면 안 된다 — 저역 상쇄를 키우는 것이 절대 목표 1 이기 때문이다.
TARGET_BAND_HZ = (80.0, 1600.0)


# ======================================================================================
# 도우미
# ======================================================================================
def _supervisor(**safety_cfg) -> SafetySupervisor:
    return SafetySupervisor(safety_cfg, FS, BLOCK)


def _isolated_dc_supervisor(**safety_cfg) -> SafetySupervisor:
    """DC 워치독만 살려 둔 감시자.

    다른 워치독이 발동하면 ``check_block`` 이 창을 비우므로 DC 상태까지 지워진다 —
    그러면 "DC 가 오발동하지 않았다" 가 다른 워치독 덕분인지 알 수 없다. 비율 임계를
    1.0 으로 두면 ``rate > 1.0`` 이 성립할 수 없어 그 워치독만 조용해진다
    (**DC 게이트 자체는 손대지 않는다**).
    """

    isolation = {
        "saturation_rate_mute": 1.0,
        "rms_rate_mute": 1.0,
        "deadline_miss_rate_mute": 1.0,
        "divergence_rate_mute": 1.0,
        "handoff_drop_rate_mute": 1.0,
    }
    isolation.update(safety_cfg)
    return _supervisor(**isolation)


def _tanh_limited(raw: np.ndarray, limit: float = 0.2) -> np.ndarray:
    """모델 출력단과 **같은 연산** (models/hybrid_anc.py: ``limit·tanh(y/limit)``)."""

    return (limit * np.tanh(np.asarray(raw, dtype=np.float64) / limit)).astype(np.float32)


def _sine(amplitude: float, freq_hz: float = 375.0, blocks: int = 1, phase: float = 0.0):
    """정현파 ``blocks`` 블록. 기본 375 Hz = 블록당 정확히 2주기.

    ⚠ 375 Hz 는 **오기각 테스트에 쓰면 안 되는 주파수**다. 같은 블록을 반복해도 진짜
    DC 가 생기지 않아 DC 워치독 오발동(R2)을 구조적으로 피해 간다. 오기각은 반드시
    :func:`_band_sweep_cases` 로 대역 전체를 몰아서 본다.
    """

    t = np.arange(blocks * BLOCK, dtype=np.float64) / FS
    return (amplitude * np.sin(2 * np.pi * freq_hz * t + phase)).astype(np.float32)


def _chirp(amplitude: float, blocks: int, phase: float = 0.0) -> np.ndarray:
    """목표 대역 80→1600 Hz 로그 스윕."""

    duration = blocks * BLOCK / FS
    t = np.arange(blocks * BLOCK, dtype=np.float64) / FS
    wave = sp_signal.chirp(
        t, TARGET_BAND_HZ[0], duration, TARGET_BAND_HZ[1], method="logarithmic",
        phi=math.degrees(phase),
    )
    return (amplitude * wave).astype(np.float32)


def _band_noise(amplitude: float, blocks: int, seed: int = 0) -> np.ndarray:
    """목표 대역으로 제한된 잡음 — 정현파가 못 만드는 파형을 덮는다."""

    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(blocks * BLOCK)
    sos = sp_signal.butter(
        4, [TARGET_BAND_HZ[0] / (FS / 2), TARGET_BAND_HZ[1] / (FS / 2)], btype="band",
        output="sos",
    )
    filtered = sp_signal.sosfilt(sos, raw)
    peak = float(np.max(np.abs(filtered)))
    return (filtered / peak * amplitude).astype(np.float32)


def _multitone(amplitude: float, blocks: int, phase: float = 0.0) -> np.ndarray:
    """대역 안 5개 톤 합 — 피크 대비 rms 가 정현파와 다른 파형."""

    t = np.arange(blocks * BLOCK, dtype=np.float64) / FS
    freqs = (80.0, 137.0, 300.0, 723.0, 1600.0)
    wave = np.zeros_like(t)
    for index, freq in enumerate(freqs):
        wave += np.sin(2 * np.pi * freq * t + phase + index * 0.7)
    wave /= float(np.max(np.abs(wave)))
    return (amplitude * wave).astype(np.float32)


def _log_tones(count: int = 12) -> tuple[float, ...]:
    """목표 대역을 로그 등간격으로 덮는 시험 주파수."""

    lo, hi = TARGET_BAND_HZ
    return tuple(float(f) for f in np.geomspace(lo, hi, count))


def _band_sweep_cases(
    amplitude: float, blocks: int, *, phases: int
) -> list[tuple[str, np.ndarray]]:
    """오기각 반증용 신호 묶음 — **대역 전체 × 여러 위상**.

    단일 주파수 하나로는 워치독의 오발동을 반증할 수 없다. R2 가 정확히 그래서
    살아남았다(375 Hz 만 봤다).
    """

    cases: list[tuple[str, np.ndarray]] = []
    phase_values = [2.0 * math.pi * k / phases for k in range(phases)]
    for freq in _log_tones():
        for index, phase in enumerate(phase_values):
            cases.append((f"tone {freq:.0f}Hz φ{index}", _sine(amplitude, freq, blocks, phase)))
    for index, phase in enumerate(phase_values):
        cases.append((f"chirp 80→1600Hz φ{index}", _chirp(amplitude, blocks, phase)))
        cases.append((f"multitone φ{index}", _multitone(amplitude, blocks, phase)))
    for seed in range(2):
        cases.append((f"band noise seed{seed}", _band_noise(amplitude, blocks, seed)))
    return cases


def _drive(
    supervisor: SafetySupervisor,
    *,
    blocks: int,
    signal_for,
    had_output_data=lambda i: True,
    error_power=1.0e-4,
    anc_output_enabled: bool = True,
    stale_input_samples: int = 0,
    anc_on: bool = True,
    observer=None,
) -> tuple[BlockVerdict | None, int]:
    """감시자를 N 블록 돌리고 **처음 mute 한 판정**과 그 블록 번호를 돌려준다.

    런타임과 같은 규약으로 돈다: 상쇄 출력이 실제로 나가는지(``anc_output_active``)는
    운용자 스위치 **그리고** "발산 프로브 중이 아님" 이다. 프로브가 시작되면
    ``run_realtime`` 이 게이트를 닫듯이 여기서도 닫힌다 — 그래야 프로브 시나리오가
    실기와 같은 것을 시험한다.

    ``error_power`` 는 상수 또는 ``f(index, anc_output_active) -> float`` 다.
    후자가 물리를 표현한다: **발산이면 출력을 닫는 순간 에러가 떨어지고, 외부
    소음이면 떨어지지 않는다.**

    ``stale_input_samples`` 도 상수 또는 ``f(index) -> int`` 다. 상수를 주면 **매
    블록** 입력을 버린 것이 되어 드롭율이 100% 다 — "한계의 90%" 같은 산발 드롭을
    표현하려면 반드시 콜러블을 써야 한다.
    """

    callable_power = callable(error_power)
    callable_stale = callable(stale_input_samples)
    for index in range(blocks):
        active = bool(anc_output_enabled) and not supervisor.probe_active
        power = float(error_power(index, active)) if callable_power else float(error_power)
        stale = int(stale_input_samples(index)) if callable_stale else int(stale_input_samples)
        report = supervisor.limit_output(signal_for(index))
        verdict = supervisor.check_block(
            BlockObservation(
                anc_on=anc_on,
                output=report,
                error_power=power,
                anc_output_active=active,
                had_output_data=bool(had_output_data(index)),
                stale_input_samples=stale,
            )
        )
        if observer is not None:
            observer(index, report, verdict)
        if verdict.mute:
            return verdict, index
    return None, blocks


def _prime_baseline(supervisor: SafetySupervisor, power: float, blocks: int = 8) -> None:
    """ANC OFF 구간을 돌려 유효 베이스라인을 잡아 둔다 (실기의 시작 절차)."""

    _drive(
        supervisor,
        blocks=blocks,
        signal_for=lambda i: np.zeros(BLOCK, dtype=np.float32),
        error_power=power,
        anc_output_enabled=False,
        anc_on=False,
    )
    assert supervisor.baseline.valid


# ======================================================================================
# S1 — 리미터와 같은 값을 재던 죽은 워치독을 대체한다
# ======================================================================================
def test_the_old_clip_counter_is_structurally_dead_on_the_dl_path():
    """모델 출력에는 ``|y| > limit`` 가 **존재할 수 없다**는 것을 못박아 둔다.

    이것이 S1 의 본질이다. 임계값을 낮추는 것으로는 고쳐지지 않는다 — 보호 장치와
    같은 값을 재는 감시자는 언제나 0 을 본다.
    """

    supervisor = _supervisor()
    hot = _tanh_limited(_sine(50.0))  # 모델이 아무리 세게 밀어도
    report = supervisor.limit_output(hot)

    # float32 에서 tanh 는 리미터 값으로 반올림되므로 "엄격히 작다"가 아니라
    # "리미터를 넘지 않는다" 가 정확한 서술이다. 결론은 같다 — 하드 리미터 카운터는
    # 이 신호에서 영원히 0 이다.
    assert float(np.max(np.abs(hot))) <= 0.2 + 1.0e-7
    assert report.clipped_fraction == 0.0, "tanh 출력이 하드 리미터에 걸릴 수는 없다"
    assert report.saturated_fraction > 0.5, "그러나 리미터 직전 영역은 가득 차 있다"


def test_saturation_watchdog_detects_output_the_old_clip_counter_declared_clean():
    """옛 워치독이 '정상'이라고 보고하던 바로 그 출력에서 mute 가 걸린다."""

    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-4)
    hot = _tanh_limited(_sine(50.0))
    verdict, index = _drive(supervisor, blocks=400, signal_for=lambda i: hot)

    assert verdict is not None, "포화 워치독이 발동하지 않았습니다 — 다시 죽은 코드입니다"
    assert WatchdogId.OUTPUT_SATURATION in verdict.fired
    assert index < 100, f"1초 안에 발동해야 합니다 (실제 {index} 블록)"
    assert any("리미터 근처" in text for text in verdict.messages)


def test_output_rms_watchdog_detects_sustained_over_power():
    """피크는 리미터에 안 닿아도 **에너지**가 상한을 넘으면 잡는다.

    포화 워치독과 잡는 파형이 다르다는 것을 보인다 (진폭 0.185 정현파는 포화 비율이
    0 이지만 rms 0.131 로 상한 0.12 를 넘는다).
    """

    supervisor = _supervisor()
    loud = _sine(0.185)
    report = supervisor.limit_output(loud)
    assert report.saturated_fraction == 0.0
    assert report.rms > supervisor.limits.rms_threshold

    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-4)
    verdict, _ = _drive(supervisor, blocks=400, signal_for=lambda i: loud)
    assert verdict is not None
    assert WatchdogId.OUTPUT_RMS in verdict.fired
    assert WatchdogId.OUTPUT_SATURATION not in verdict.fired


# ======================================================================================
# S2 — 스트릭이 못 잡던 교대 미스
# ======================================================================================
def test_deadline_watchdog_detects_the_alternating_miss_the_streak_never_caught():
    """블록 하나 걸러 무음이 나가는 상태가 무한 지속되지 않는다.

    옛 구현은 ``elif had_data: streak = 0`` 이라 1,0,1,0 에서 스트릭이 1 을 넘지
    못했다. 상쇄음이 절반만 나가는 이 상태가 음향적으로 가장 나쁘다.
    """

    # 옛 로직 재현 — 교대 미스에서 스트릭은 절대 임계값(3)에 닿지 못한다.
    streak, peak = 0, 0
    for index in range(2000):
        had_data = index % 2 == 0
        streak = 0 if had_data else streak + 1
        peak = max(peak, streak)
    assert peak == 1, "옛 스트릭 워치독은 교대 미스에서 1 을 넘지 못한다"

    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-4)
    verdict, index = _drive(
        supervisor,
        blocks=800,
        signal_for=lambda i: _sine(0.05),
        had_output_data=lambda i: i % 2 == 0,
    )
    assert verdict is not None, "교대 미스를 여전히 못 잡습니다"
    assert WatchdogId.DEADLINE_MISS_RATE in verdict.fired
    assert index < 200, f"1초 안에 발동해야 합니다 (실제 {index} 블록)"


def test_deadline_watchdog_tolerates_a_rare_isolated_miss():
    """오기각 방지 — 드문 언더런 하나로 ANC 를 끄지는 않는다."""

    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-4)
    verdict, _ = _drive(
        supervisor,
        blocks=1000,
        signal_for=lambda i: _sine(0.05),
        error_power=2.0e-5,
        had_output_data=lambda i: i % 50 != 0,  # 미스율 2% < 20%
    )
    assert verdict is None


# ======================================================================================
# S3 — 발산: 감시 불가는 fail-closed, 판정 불가는 **다시 잰다**
# ======================================================================================
def test_divergence_watchdog_detects_the_missing_baseline_instead_of_going_quiet():
    """베이스라인을 한 번도 못 잡으면 옛 워치독은 조용히 비활성이었다 — 이제 fail-closed.

    상쇄 출력이 처음부터 열려 있으면 (= ANC OFF 구간이 없으면) 베이스라인 수집
    자체가 불가능하다. 그 상태로 상쇄음을 무한정 내보내지 않는다.
    """

    supervisor = _supervisor()
    assert not supervisor.baseline_is_valid(0.0, initialized=True)
    assert not supervisor.baseline_is_valid(1.0e-4, initialized=False)
    assert not supervisor.baseline.valid

    verdict, index = _drive(
        supervisor,
        blocks=2000,
        signal_for=lambda i: _sine(0.05),
        error_power=1.0e-4,
        anc_output_enabled=True,
    )
    assert verdict is not None, "베이스라인 없이 상쇄음을 무한정 내보내고 있습니다"
    assert WatchdogId.DIVERGENCE in verdict.fired
    expected = supervisor.windows.baseline_grace_blocks
    assert index == expected - 1, f"유예 {expected} 블록 후 즉시 꺼져야 합니다"
    assert any("베이스라인이 없습니다" in text for text in verdict.messages)


def test_divergence_watchdog_detects_error_power_above_the_baseline():
    """정상 경로 — 상쇄음 자신이 에러를 키우면 프로브가 그것을 **측정으로** 확정한다.

    발산의 물리적 서명은 하나다: **상쇄 출력을 닫으면 에러가 떨어진다.**
    """

    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-5)

    def error_power(_index: int, anc_output_active: bool) -> float:
        return 1.0e-2 if anc_output_active else 1.0e-5

    verdict, index = _drive(
        supervisor, blocks=600, signal_for=lambda i: _sine(0.05), error_power=error_power
    )
    assert verdict is not None
    assert WatchdogId.DIVERGENCE in verdict.fired
    assert any("발산 확정" in text for text in verdict.messages)
    assert supervisor.divergence_probes == 1
    assert supervisor.divergence_rebaselines == 0
    # 의심(≤ divergence 창) → 프로브(1s) → 확정. 그 사이 출력은 이미 닫혀 있었다.
    assert index <= supervisor.windows.divergence_blocks + supervisor.windows.divergence_probe_blocks


def test_divergence_probe_closes_the_output_before_it_decides():
    """프로브는 **보호를 미루지 않는다** — 의심 즉시 상쇄 출력이 0 이 된다.

    이것이 "fail-open 이 아니다" 의 근거다. mute 와 음향적으로 동일한 동작을 먼저
    취하고, 미루는 것은 진단뿐이다.
    """

    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-5)
    active_log: list[bool] = []

    def error_power(_index: int, anc_output_active: bool) -> float:
        active_log.append(anc_output_active)
        return 1.0e-2 if anc_output_active else 1.0e-5

    verdict, index = _drive(
        supervisor, blocks=600, signal_for=lambda i: _sine(0.05), error_power=error_power
    )
    assert verdict is not None and verdict.mute
    closed = [i for i, active in enumerate(active_log) if not active]
    assert closed, "프로브가 상쇄 출력을 닫지 않았습니다"
    # 첫 폐쇄 시점이 mute 시점보다 divergence_probe_blocks 만큼 앞선다.
    assert index - closed[0] == supervisor.windows.divergence_probe_blocks - 1


def test_a_quiet_baseline_plus_external_noise_does_not_shut_anc_down():
    """**R1 회귀 방지.** 정숙 상태에서 잡은 베이스라인 + 외부 소음 ON.

    실측 반증: 정숙 5s 로 baseline 2.223e-06 이 유효해진 뒤 외부 소음을 켜면
    23블록(123 ms) 만에 "+18.8dB → 자동 OFF" 였다. 이제는 프로브가 "출력을 닫아도
    에러가 그대로다"를 재고 베이스라인을 갱신한다.
    """

    supervisor = _supervisor()
    _prime_baseline(supervisor, 2.223e-06)  # 조용한 방의 마이크 플로어
    external = 2.223e-06 * 10 ** (18.8 / 10.0)

    verdict, _ = _drive(
        supervisor,
        blocks=2000,
        signal_for=lambda i: _sine(0.05),
        error_power=lambda _i, _active: external,  # 외부 소음원 — 우리 출력과 무관
    )
    assert verdict is None, "외부 소음원을 켜자 ANC 가 꺼졌습니다 (R1 재발)"
    assert supervisor.divergence_probes == 1, "의심은 했어야 한다 (감시를 끄면 안 된다)"
    assert supervisor.divergence_rebaselines == 1, "측정으로 베이스라인을 갱신했어야 한다"
    assert supervisor.baseline.power == pytest.approx(external, rel=1.0e-6)
    assert supervisor.trip_counts[WatchdogId.DIVERGENCE] == 0


def test_repeated_inconclusive_probes_end_in_a_fail_closed_mute():
    """진단이 계속 결론에 이르지 못하면 **끈다** — 무한 재개는 fail-open 이다."""

    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-6)
    limit = supervisor.limits.divergence_probe_limit
    step = {"level": 1.0e-6}

    def error_power(_index: int, _active: bool) -> float:
        # 프로브가 끝날 때마다 장면이 또 10배 시끄러워진다 — 원인 특정 불가.
        return step["level"] * 10.0 ** supervisor.divergence_rebaselines * 100.0

    verdict, _ = _drive(
        supervisor, blocks=6000, signal_for=lambda i: _sine(0.05), error_power=error_power
    )
    assert verdict is not None and verdict.mute
    assert WatchdogId.DIVERGENCE in verdict.fired
    assert supervisor.divergence_rebaselines == limit
    assert any("fail-closed" in text for text in verdict.messages)


def test_divergence_watchdog_survives_alternating_divergence():
    """발산도 스트릭이 아니라 비율이다 — 한 블록 걸러 정상이어도 잡힌다."""

    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-5)
    powers = [1.0e-2, 1.0e-6]

    def error_power(index: int, anc_output_active: bool) -> float:
        return powers[index % 2] if anc_output_active else 1.0e-5

    verdict, _ = _drive(
        supervisor, blocks=800, signal_for=lambda i: _sine(0.05), error_power=error_power
    )
    assert verdict is not None, "교대 발산을 놓쳤습니다 (스트릭 회귀)"
    assert WatchdogId.DIVERGENCE in verdict.fired


def test_baseline_is_only_collected_while_the_output_gate_is_closed():
    """베이스라인 수집 규칙의 단일 출처를 직접 시험한다.

    수집 조건은 "소음이 재생 중" 이 **아니라** "상쇄 출력이 나가지 않는다" 다.
    전자를 쓰면 외부 소음원 운용에서 감시가 통째로 사라진다(옛 fail-open).
    """

    tracker = BaselineTracker(SafetyLimits.from_config({}), FS, BLOCK)
    assert not tracker.valid

    tracker.observe(error_power=1.0e-4, anc_output_active=False)
    assert tracker.valid and tracker.power == pytest.approx(1.0e-4)

    for _ in range(500):  # 출력이 열려 있는 동안에는 얼어 있어야 한다
        tracker.observe(error_power=1.0, anc_output_active=True)
    assert tracker.power == pytest.approx(1.0e-4)
    assert tracker.blocks_since_update == 500

    tracker.force(2.0e-4)
    assert tracker.power == pytest.approx(2.0e-4) and tracker.blocks_since_update == 0

    # 무음(하한 이하)은 "잰 것이 없다" 로 취급한다 — fail-closed.
    tracker.reset()
    tracker.observe(error_power=1.0e-12, anc_output_active=False)
    assert tracker.initialized and not tracker.valid


# ======================================================================================
# S6 — NaN 을 조용히 삼키지 않는다
# ======================================================================================
def test_nonfinite_output_watchdog_detects_nan_instead_of_swallowing_it():
    """NaN 한 블록이면 즉시 보고하고 mute. 무한 무음을 정상으로 오인하지 않는다."""

    supervisor = _supervisor()
    broken = np.full(BLOCK, np.nan, dtype=np.float32)
    broken[:10] = np.inf

    report = supervisor.limit_output(broken)
    assert report.nonfinite_fraction == 1.0
    assert np.all(np.isfinite(report.signal)), "출력 자체는 유한해야 한다 (스피커 보호)"

    verdict = supervisor.check_block(
        BlockObservation(
            anc_on=True,
            output=report,
            error_power=1.0e-4,
            anc_output_active=True,
            had_output_data=True,
        )
    )
    assert verdict.mute
    assert WatchdogId.NONFINITE_OUTPUT in verdict.fired
    assert any("NaN/Inf" in text for text in verdict.messages), "메시지 없이 삼키면 안 된다"


# ======================================================================================
# S7 — 출력 DC (하드웨어 보호) + R2 오기각
# ======================================================================================
def test_output_dc_blocker_removes_the_offset_before_it_reaches_the_amplifier():
    """모델이 DC 0.19 를 내도 스피커로 나가는 신호에서는 DC 가 사라진다."""

    supervisor = _supervisor()
    offset = np.full(BLOCK, 0.19, dtype=np.float32)
    tail = None
    for _ in range(80):  # ≈ 0.43 s
        tail = supervisor.limit_output(offset)
    assert abs(tail.dc_in - 0.19) < 1.0e-6
    assert abs(tail.dc_out) < 0.01, f"출력 DC 가 남아 있습니다: {tail.dc_out}"


def test_output_dc_watchdog_detects_a_sustained_offset():
    """차단기와 별개로 **모델이 DC 를 내고 있다는 사실 자체**를 잡아 ANC 를 끈다."""

    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-4)
    offset = np.full(BLOCK, 0.19, dtype=np.float32)
    verdict, index = _drive(supervisor, blocks=400, signal_for=lambda i: offset)

    assert verdict is not None, "출력 DC 워치독이 발동하지 않았습니다 (하드웨어 손상 경로)"
    assert WatchdogId.OUTPUT_DC in verdict.fired
    assert index < supervisor.windows.dc_blocks
    assert any("보이스코일" in text for text in verdict.messages)


def test_output_dc_watchdog_still_catches_a_small_pure_offset():
    """한계를 겨우 넘는 순수 DC 도 잡는다 — 우세도 조건이 뚜껑이 되면 안 된다."""

    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-4)
    threshold = supervisor.limits.dc_threshold
    offset = np.full(BLOCK, threshold * 1.5, dtype=np.float32)
    verdict, _ = _drive(supervisor, blocks=800, signal_for=lambda i: offset)
    assert verdict is not None and WatchdogId.OUTPUT_DC in verdict.fired


def test_a_moving_average_cannot_separate_80hz_from_dc():
    """**R2 의 근본 원인**을 수치로 못박는다 — 창 길이만으로는 안 된다.

    길이 T 이동평균이 주파수 f·진폭 A 의 영평균 정현파에 남기는 겉보기 평균의
    상계는 ``A/(π f T)`` 다. 옛 구현의 T(=53 ms, MeanWindow.min_blocks)에서 목표
    대역 하단 80 Hz·리미터 진폭 0.2 를 넣으면 한계 0.01 을 넘는다.
    """

    limits = SafetyLimits.from_config({})
    amplitude = limits.control_limit
    old_window_s = 10 * BLOCK / FS  # ceil(38/4)=10 블록 = 53 ms

    def moving_average_leak(freq: float, window_s: float) -> float:
        """실제 최악 위상 잔여 평균 (해석 상계가 아니라 직접 계산)."""

        samples = int(round(window_s * FS))
        t = np.arange(samples) / FS
        worst = 0.0
        for phase in np.linspace(0.0, 2 * np.pi, 64, endpoint=False):
            worst = max(worst, abs(float(np.mean(amplitude * np.sin(2 * np.pi * freq * t + phase)))))
        return worst

    leak_80 = moving_average_leak(80.0, old_window_s)
    assert leak_80 > limits.dc_threshold, (
        f"옛 53 ms 이동평균의 80 Hz 누설 {leak_80:.4f} 이 한계 "
        f"{limits.dc_threshold:.4f} 이하라면 R2 재현이 성립하지 않는다"
    )

    # 판정: **창 길이만으로도 80 Hz 는 넘길 수 있다** (A/(πfT) ≤ 한계 → T ≥ 80 ms).
    # 그러나 그 창은 여유가 1배뿐이고, 여유 2배를 요구하면 창이 3배로 길어진다 =
    # 진짜 DC 검출이 그만큼 느려진다. 게다가 필요한 창은 1/f 로 커지는데 출력 DC
    # 차단기의 코너는 3.8 Hz 라 그 아래까지는 모델이 무엇이든 낼 수 있다.
    tuned_window_s = 2.0 * amplitude / (math.pi * 80.0 * limits.dc_threshold)
    assert tuned_window_s > 2.5 * old_window_s, "여유 2배를 주면 창이 3배로 길어진다"
    assert moving_average_leak(80.0, tuned_window_s) < 0.5 * limits.dc_threshold
    assert moving_average_leak(20.0, tuned_window_s) > limits.dc_threshold, (
        "80 Hz 기준으로 늘린 창도 그 아래 주파수는 여전히 DC 로 오인한다 — "
        "출력 DC 차단기의 코너는 3.8 Hz 라 그 위는 모델이 무엇이든 낼 수 있다"
    )

    # 실제로 쓰는 판정기(저역통과)는 창을 늘리지 않고 같은 신호를 배제한다.
    supervisor = _supervisor()
    lowpass = supervisor._dc_lowpass
    t = np.arange(600 * BLOCK) / FS
    wave = amplitude * np.sin(2 * np.pi * 80.0 * t)
    trace = []
    for index in range(600):
        block = wave[index * BLOCK : (index + 1) * BLOCK]
        trace.append(abs(lowpass.update(float(np.mean(block)))))
    peak, steady = max(trace), max(trace[-200:])
    assert steady < 0.5 * limits.dc_threshold, (
        f"저역통과 DC 추정값이 80 Hz 리미터 진폭에서 {steady:.5f} — 한계 "
        f"{limits.dc_threshold:.4f} 대비 정상상태 여유가 2배 미만이다"
    )
    # 0 에서 출발하는 1차 IIR 은 과도구간에서 정상상태의 최대 2배까지 간다.
    # 그래도 한계를 넘지 않아야 하고, 실제 판정에는 AC 우세도가 더해져
    # (test_output_dc_watchdog_ignores_a_zero_mean_tone 이 재는) 여유가 5배가 된다.
    assert peak < limits.dc_threshold, f"과도구간 최대 {peak:.5f}"


def test_output_dc_watchdog_ignores_a_zero_mean_tone():
    """오기각 방지 — 정상 상쇄음(영평균)은 DC 워치독을 건드리지 않는다.

    ⚠ 예전에는 이 테스트가 375 Hz **하나**만 썼다(블록당 정확히 2주기 = R2 를 구조적으로
    피해 가는 유일한 주파수). 이제 목표 대역 전체를 **리미터 진폭까지** 몬다.
    """

    limits = SafetyLimits.from_config({})
    amplitude = limits.control_limit  # 더 크게 낼 수 없다 — 리미터가 여기서 자른다
    failures: list[str] = []
    worst_ratio, worst_name = 0.0, ""
    for name, wave in _band_sweep_cases(amplitude, blocks=250, phases=8):
        supervisor = _isolated_dc_supervisor()
        _prime_baseline(supervisor, 1.0e-4)
        peak = {"ratio": 0.0}

        def observe(_i, _report, _verdict, supervisor=supervisor, peak=peak):
            # 이 블록이 DC 로 판정되기까지 남은 여유 (1.0 이면 발동 직전).
            need = max(
                supervisor._dc_threshold,
                min(
                    supervisor._dc_dominance * supervisor._ac_lowpass.value,
                    supervisor._dc_hard_threshold,
                ),
            )
            peak["ratio"] = max(peak["ratio"], abs(supervisor._dc_lowpass.value) / need)

        _drive(
            supervisor,
            blocks=250,
            signal_for=lambda i, w=wave: w[i * BLOCK : (i + 1) * BLOCK],
            error_power=2.0e-5,
            observer=observe,
        )
        if peak["ratio"] > worst_ratio:
            worst_ratio, worst_name = peak["ratio"], name
        if supervisor.trip_counts[WatchdogId.OUTPUT_DC]:
            failures.append(f"{name}: DC={supervisor._dc_lowpass.value:+.5f}")
    assert failures == [], (
        "영평균 목표 대역 상쇄음이 DC 워치독을 발동시켰습니다 (R2 재발) — "
        + "; ".join(failures[:8])
    )
    # 발동하지 않았다는 것만으로는 부족하다 — 실측 반증에서 여유가 1.74배뿐이었고
    # 저역 상쇄를 조금만 키우면 걸렸다. 최소 2배 여유를 요구한다.
    assert worst_ratio < 0.5, (
        f"DC 워치독 여유가 {1.0 / max(worst_ratio, 1e-9):.2f}배뿐입니다 "
        f"(최악 {worst_name}) — 저역 상쇄를 키우면 걸립니다"
    )


# ======================================================================================
# 백로그 비대칭 (발생기 A)
# ======================================================================================
def test_handoff_budget_derives_one_hop_from_the_duct_config():
    """예산은 duct.yaml 의 handoff(**단일 출처**)에서만 나온다."""

    budget = PipelineHandoffBudget.derive(duct_cfg=DUCT, hop=BLOCK)
    assert budget.handoff_samples == BLOCK
    assert budget.input_keep_backlog_samples == budget.output_keep_backlog_samples == BLOCK
    assert budget.effective_handoff_samples == BLOCK


def test_handoff_budget_rejects_the_input_output_asymmetry():
    """감사에서 발견된 값(입력 8 hop / 출력 1 hop)은 **만들어지지 않는다**."""

    with pytest.raises(ValueError, match="비대칭"):
        PipelineHandoffBudget(
            hop_samples=BLOCK,
            handoff_samples=BLOCK,
            input_keep_backlog_samples=BLOCK * 8,   # 실기에 있던 값
            output_keep_backlog_samples=BLOCK,
            token=_BUDGET_TOKEN,                    # 검증기 자체를 시험하기 위한 내부 토큰
        )

    # 대칭이어도 1 hop 을 넘으면 실효 핸드오프가 학습 가정에서 벗어난다.
    with pytest.raises(ValueError, match="실효 핸드오프"):
        PipelineHandoffBudget(
            hop_samples=BLOCK,
            handoff_samples=BLOCK,
            input_keep_backlog_samples=BLOCK * 2,
            output_keep_backlog_samples=BLOCK * 2,
            token=_BUDGET_TOKEN,
        )

    # 손으로는 아예 못 만든다 (dsp/timing.Lead 와 같은 규약).
    with pytest.raises(TypeError, match="derive"):
        PipelineHandoffBudget(
            hop_samples=BLOCK,
            handoff_samples=BLOCK,
            input_keep_backlog_samples=BLOCK,
            output_keep_backlog_samples=BLOCK,
        )


def test_handoff_budget_rejects_a_duct_handoff_that_disagrees_with_the_hop():
    """학습 플랜트 핸드오프와 런타임 hop 이 어긋나면 시작 자체를 막는다."""

    with pytest.raises(ValueError, match="1 hop"):
        PipelineHandoffBudget.derive(
            duct_cfg={"secondary_path": {"handoff_extra_samples": 512}}, hop=BLOCK
        )


def test_build_engine_takes_the_handoff_from_the_budget_not_from_the_duct_config():
    """FxLMS 엔진의 핸드오프가 **두 번째 유도**가 되지 않는다 (발생기 A).

    ``engines.build_engine`` 은 예전에 duct cfg 를 직접 다시 읽었다. 이제
    ``PipelineHandoffBudget`` 하나에서만 나오므로 소스에 그 재유도가 남아 있으면
    여기서 실패한다.
    """

    source = (REPO_ROOT / "src/deep_anc/realtime/engines.py").read_text(encoding="utf-8")
    assert "PipelineHandoffBudget" in source
    assert "handoff_extra_samples\"" not in source.replace("handoff_extra_samples=budget", ""), (
        "build_engine 이 duct cfg 에서 handoff 를 다시 읽고 있습니다"
    )
    assert re.search(r"handoff_extra_samples=budget\.handoff_samples", source), source[-800:]


def test_handoff_backlog_watchdog_detects_dropped_stale_input():
    """오래된 입력을 버리는 상태가 지속되면 ANC 를 끈다.

    입력을 버렸다는 것은 추론이 뒤처져 실효 핸드오프가 학습 가정에서 이탈했다는
    뜻이고, 그 지연의 안티노이즈는 상쇄가 아니라 증폭이 될 수 있다.
    """

    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-4)
    verdict, index = _drive(
        supervisor,
        blocks=800,
        signal_for=lambda i: _sine(0.05),
        stale_input_samples=BLOCK,
    )
    assert verdict is not None
    assert WatchdogId.HANDOFF_BACKLOG in verdict.fired
    assert index < 200


def test_ring_backlog_of_one_hop_keeps_the_newest_block():
    """예산 값이 실제로 '지연 0 추가'를 뜻하는지 링버퍼로 확인한다."""

    ring = SPSCRing(1, BLOCK * 64)
    for value in range(8):
        ring.push(np.full((1, BLOCK), float(value), dtype=np.float32))

    block, ok = ring.pop_latest(BLOCK, keep_backlog=BLOCK)
    assert ok and float(block[0, 0]) == 7.0, "1 hop 예산은 최신 블록을 준다"

    ring = SPSCRing(1, BLOCK * 64)
    for value in range(8):
        ring.push(np.full((1, BLOCK), float(value), dtype=np.float32))
    block, ok = ring.pop_latest(BLOCK, keep_backlog=BLOCK * 8)
    assert ok and float(block[0, 0]) == 0.0, "8 hop 예산은 7 hop 낡은 블록을 준다 (옛 동작)"


def test_run_realtime_reads_the_backlog_allowance_only_from_the_budget():
    """백로그 산술이 다시 두 곳으로 갈라지지 않게 소스를 검사한다 (발생기 A).

    ``keep_backlog=`` 의 값은 전부 ``handoff_budget`` 에서 와야 한다. 리터럴이
    하나라도 돌아오면 여기서 실패한다.
    """

    source = (REPO_ROOT / "src/deep_anc/realtime/run_realtime.py").read_text(
        encoding="utf-8"
    )
    uses = re.findall(r"keep_backlog=([^\n,)]+)", source)
    assert uses, "keep_backlog 사용처를 찾지 못했습니다 (테스트가 낡았습니다)"
    assert all("handoff_budget" in item for item in uses), uses


# ======================================================================================
# 오기각 방지 — **대역 전체**를 리미터가 허용하는 진폭까지 몬다
# ======================================================================================
def test_clean_anti_noise_trips_no_watchdog():
    """정상 운용 3초에서는 어떤 워치독도 발동하지 않는다.

    워치독을 세게 걸수록 오기각 위험이 커진다. 발동 증명과 **짝**으로 이 테스트가
    있어야 "발동은 하지만 쓸 수 없는" 워치독이 되지 않는다.
    """

    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-4)
    blocks = int(3.0 * FS / BLOCK)
    tone = _sine(0.05, blocks=blocks)
    verdict, _ = _drive(
        supervisor,
        blocks=blocks,
        signal_for=lambda i: tone[i * BLOCK : (i + 1) * BLOCK],
        error_power=2.0e-5,
    )
    assert verdict is None
    assert all(count == 0 for count in supervisor.trip_counts.values())


def test_no_watchdog_fires_across_the_target_band_at_the_legitimate_output_ceiling():
    """**이 파일의 두 번째 계약.** 80–1600 Hz 를 허용 진폭 천장까지 몰아도 조용하다.

    "허용 진폭 천장" 은 테스트가 고른 숫자가 아니라 RMS 워치독이 정의한 값이다
    (``rms_threshold × √2``). 여기서 임의로 낮은 진폭을 쓰면 게이트를 우회할 수
    있으므로, 스윕 도중 실제 블록 RMS 가 상한에 **거의 닿았다**는 것도 함께 단언한다.
    """

    limits = SafetyLimits.from_config({})
    # 블록 RMS 는 저역에서 부분 주기 때문에 A/√2 보다 커진다(80 Hz 에서 0.763A).
    # 상한에 닿되 넘지는 않는 최대 진폭.
    amplitude = limits.legitimate_tone_amplitude * 0.9
    assert amplitude > 0.7 * limits.control_limit

    failures: list[str] = []
    peak_rms = 0.0
    peak_saturation = 0.0
    for name, wave in _band_sweep_cases(amplitude, blocks=300, phases=4):
        supervisor = _supervisor()
        _prime_baseline(supervisor, 1.0e-4)
        stats = {"rms": 0.0, "sat": 0.0}

        def observe(_i, report, _v, stats=stats):
            stats["rms"] = max(stats["rms"], report.rms)
            stats["sat"] = max(stats["sat"], report.saturated_fraction)

        verdict, index = _drive(
            supervisor,
            blocks=300,
            signal_for=lambda i, w=wave: w[i * BLOCK : (i + 1) * BLOCK],
            error_power=2.0e-5,
            had_output_data=lambda i: True,
            observer=observe,
        )
        peak_rms = max(peak_rms, stats["rms"])
        peak_saturation = max(peak_saturation, stats["sat"])
        fired = {k.value: v for k, v in supervisor.trip_counts.items() if v}
        if verdict is not None or fired:
            failures.append(f"{name}@{index}: {fired}")

    assert failures == [], (
        "목표 대역 정상 상쇄음이 워치독을 발동시켰습니다 — 워치독이 절대 목표의 "
        "적이 됩니다: " + "; ".join(failures[:8])
    )
    assert peak_rms <= limits.rms_threshold, (
        f"스윕 진폭이 RMS 상한을 넘었다 ({peak_rms:.4f} > {limits.rms_threshold:.4f}) — "
        "그 진폭은 '정상 상쇄음'이 아니므로 이 테스트가 무의미해진다"
    )
    assert peak_rms > 0.9 * limits.rms_threshold, (
        f"스윕이 상한 근처까지 가지 않았다 ({peak_rms:.4f}) — 낮은 진폭으로 "
        "게이트를 우회한 셈이다"
    )
    assert peak_saturation < limits.saturation_fraction


def test_the_rms_watchdog_still_fires_just_above_that_ceiling():
    """천장이 **실제 경계**임을 보인다 — 조금만 넘으면 잡힌다 (게이트가 살아 있다)."""

    limits = SafetyLimits.from_config({})
    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-4)
    loud = _sine(limits.legitimate_tone_amplitude * 1.1, freq_hz=800.0, blocks=400)
    verdict, _ = _drive(
        supervisor,
        blocks=400,
        signal_for=lambda i: loud[i * BLOCK : (i + 1) * BLOCK],
        error_power=2.0e-5,
    )
    assert verdict is not None and WatchdogId.OUTPUT_RMS in verdict.fired


def test_watchdogs_are_idle_while_anc_is_off():
    """ANC OFF 구간의 링버퍼 언더런으로는 아무 일도 일어나지 않는다."""

    supervisor = _supervisor()
    verdict, _ = _drive(
        supervisor,
        blocks=1000,
        signal_for=lambda i: np.full(BLOCK, np.nan, dtype=np.float32),
        had_output_data=lambda i: False,
        anc_output_enabled=False,
        anc_on=False,
    )
    assert verdict is None


# ======================================================================================
# 측정 모드 — 하드웨어 보호는 내려가지 않는다
# ======================================================================================
def test_measurement_mode_downgrades_performance_watchdogs_but_not_hardware_ones():
    """캘리브레이션은 의도적으로 큰 출력을 낸다. 그렇다고 DC 를 흘려도 되는 것은 아니다."""

    supervisor = _supervisor(measurement_mode=True)
    _prime_baseline(supervisor, 1.0e-4)
    hot = _tanh_limited(_sine(50.0))
    verdict, _ = _drive(supervisor, blocks=400, signal_for=lambda i: hot)
    assert verdict is None, "측정 모드에서 성능 워치독이 mute 하면 캘리브레이션이 깨진다"
    assert supervisor.trip_counts[WatchdogId.OUTPUT_SATURATION] > 0, "자문 보고는 남아야 한다"

    supervisor = _supervisor(measurement_mode=True)
    _prime_baseline(supervisor, 1.0e-4)
    offset = np.full(BLOCK, 0.19, dtype=np.float32)
    verdict, _ = _drive(supervisor, blocks=400, signal_for=lambda i: offset)
    assert verdict is not None and verdict.mute
    assert set(verdict.fired) & HARDWARE_PROTECTION_WATCHDOGS


def test_measurement_mode_never_closes_the_output_for_a_probe():
    """프로브는 출력을 닫는 **동작**이다 — 캘리브레이션 중에는 자문만 남긴다."""

    supervisor = _supervisor(measurement_mode=True)
    _prime_baseline(supervisor, 1.0e-5)
    verdict, _ = _drive(
        supervisor,
        blocks=400,
        signal_for=lambda i: _sine(0.05),
        error_power=lambda _i, active: 1.0e-2 if active else 1.0e-5,
    )
    assert verdict is None
    assert supervisor.divergence_probes == 0
    assert supervisor.trip_counts[WatchdogId.DIVERGENCE] > 0


# ======================================================================================
# 설정 경계
# ======================================================================================
def test_safety_config_rejects_unknown_keys_and_reports_legacy_ones():
    """오타는 조용히 기본값으로 떨어지지 않고, 폐기된 키는 안내를 남긴다."""

    with pytest.raises(Exception):
        SafetyLimits.from_config({"controll_limit": 0.1})

    limits = SafetyLimits.from_config(
        {"control_limit": 0.2, "clip_streak_mute": 20, "deadline_miss_mute": 3}
    )
    assert limits.control_limit == 0.2
    assert len(limits.legacy_notes) == 2
    assert any("교대 미스" in note for note in limits.legacy_notes)


def test_shipped_runtime_configs_use_the_current_safety_keys():
    """출하 설정이 폐기 키를 쓰지 않고, 현장에서 조정할 값이 실제로 설정에 있다.

    폐기 키만 적혀 있으면 런타임은 코드 기본값으로 돌고 현장 조정이 불가능하다 —
    설정이 권위 있어 보이면서 아무 효과가 없는 상태다.
    """

    from deep_anc.config import load_runtime_config

    tunables = {
        "saturation_rate_mute",
        "rms_rate_mute",
        "deadline_miss_rate_mute",
        "divergence_rate_mute",
        "divergence_probe_s",
        "output_dc_lowpass_hz",
        "output_dc_ac_dominance",
        "handoff_drop_rate_mute",
        "baseline_grace_s",
    }
    for name in ("configs/runtime.yaml", "configs/runtime_tiny.yaml"):
        raw = dict(load_runtime_config(name).get("safety", {}) or {})
        limits = SafetyLimits.from_config(raw)
        assert limits.legacy_notes == (), f"{name}: 폐기 키가 남아 있습니다 {limits.legacy_notes}"
        missing = sorted(tunables - set(raw))
        assert missing == [], f"{name}: 현장 조정 키가 빠졌습니다 {missing}"


def test_safety_config_refuses_to_recreate_the_dead_watchdog():
    """``saturation_ratio = 1.0`` 은 리미터와 같은 값을 재는 것 = S1 의 재현이다."""

    with pytest.raises(ValueError, match="죽은 워치독"):
        SafetyLimits.from_config({"saturation_ratio": 1.0})
    with pytest.raises(ValueError):
        SafetyLimits.from_config({"divergence_ratio": 0.5})
    with pytest.raises(ValueError):
        SafetyLimits.from_config({"deadline_miss_rate_mute": 0.0})
    # 확인 임계가 발산 임계보다 크면 어떤 진짜 발산도 확정되지 않는다.
    with pytest.raises(ValueError, match="divergence_probe_confirm_ratio"):
        SafetyLimits.from_config({"divergence_ratio": 4.0, "divergence_probe_confirm_ratio": 5.0})


def test_safety_limits_are_frozen():
    """유도 임계값이 나중에 조용히 바뀌면 단일 출처가 무의미하다."""

    limits = SafetyLimits.from_config({})
    with pytest.raises(Exception):
        limits.control_limit = 0.9


def test_block_observation_rejects_impossible_values():
    """음수/비유한 파워는 관측 자체가 만들어지지 않는다."""

    supervisor = _supervisor()
    report = supervisor.limit_output(_sine(0.05))
    for bad in ({"error_power": -1.0}, {"error_power": float("nan")},
                {"stale_input_samples": -1}):
        kwargs = {
            "anc_on": True,
            "output": report,
            "error_power": 1.0e-4,
            "anc_output_active": True,
            "had_output_data": True,
        }
        kwargs.update(bad)
        with pytest.raises(ValueError):
            BlockObservation(**kwargs)


def test_one_pole_lowpass_rejects_a_cutoff_above_the_block_nyquist():
    """블록률 나이키스트 위의 차단주파수는 저역통과가 아니다 — 생성 시점에 막는다."""

    with pytest.raises(ValueError):
        OnePoleLowPass(200.0, FS / BLOCK)
    with pytest.raises(ValueError):
        OnePoleLowPass(0.0, FS / BLOCK)


def test_safety_stays_inside_the_callback_deadline_budget():
    """핫패스 회귀 방지 — 감시자가 콜백 마감을 잡아먹지 않는지 실측한다.

    마감은 5.33 ms 이고 콜백 본체가 이미 0.46 ms 를 쓴다. 워치독을 늘리다 보면
    검증을 샘플 단위로 넣기 쉬운데, 그러면 xrun 이 나고 **워치독이 사고의 원인**이
    된다. 부하 스파이크에 흔들리지 않도록 여러 배치의 최소값으로 판정한다.
    """

    import time

    supervisor = _supervisor()
    tone = _sine(0.05)
    best = float("inf")
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(200):
            report = supervisor.limit_output(tone)
            supervisor.check_block(
                BlockObservation(
                    anc_on=True,
                    output=report,
                    error_power=1.0e-4,
                    anc_output_active=True,
                    had_output_data=True,
                )
            )
        best = min(best, (time.perf_counter() - start) / 200.0)
    assert best < 1.5e-3, f"블록당 {best*1e6:.0f} µs — 5.33 ms 마감에서 너무 비쌉니다"


def test_rate_window_is_immune_to_alternation():
    """대체 원시자료: 스트릭이 아니라 비율이라는 것을 직접 확인한다."""

    window = RateWindow(100)
    for index in range(100):
        window.update(index % 2 == 0)
    assert window.rate == pytest.approx(0.5)
    assert window.exceeds(0.2)
    assert not window.exceeds(0.9)
    window.reset()
    assert window.count == 0 and window.rate == 0.0


# ======================================================================================
# 메타 — 열거된 워치독은 발동 증명 **과** 대역 전체 오기각 반증을 둘 다 갖는다
# ======================================================================================
def _trip_nonfinite() -> BlockVerdict:
    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-4)
    verdict, _ = _drive(
        supervisor, blocks=4, signal_for=lambda i: np.full(BLOCK, np.nan, dtype=np.float32)
    )
    return verdict


def _trip_dc() -> BlockVerdict:
    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-4)
    verdict, _ = _drive(
        supervisor, blocks=400, signal_for=lambda i: np.full(BLOCK, 0.19, dtype=np.float32)
    )
    return verdict


def _trip_saturation() -> BlockVerdict:
    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-4)
    verdict, _ = _drive(
        supervisor, blocks=400, signal_for=lambda i: _tanh_limited(_sine(50.0))
    )
    return verdict


def _trip_rms() -> BlockVerdict:
    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-4)
    verdict, _ = _drive(supervisor, blocks=400, signal_for=lambda i: _sine(0.185))
    return verdict


def _trip_deadline() -> BlockVerdict:
    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-4)
    verdict, _ = _drive(
        supervisor,
        blocks=800,
        signal_for=lambda i: _sine(0.05),
        had_output_data=lambda i: i % 2 == 0,
    )
    return verdict


def _trip_divergence() -> BlockVerdict:
    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-5)
    verdict, _ = _drive(
        supervisor,
        blocks=600,
        signal_for=lambda i: _sine(0.05),
        error_power=lambda _i, active: 1.0e-2 if active else 1.0e-5,
    )
    return verdict


def _trip_handoff() -> BlockVerdict:
    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-4)
    verdict, _ = _drive(
        supervisor,
        blocks=800,
        signal_for=lambda i: _sine(0.05),
        stale_input_samples=BLOCK,
    )
    return verdict


TRIP_SCENARIOS = {
    WatchdogId.NONFINITE_OUTPUT: _trip_nonfinite,
    WatchdogId.OUTPUT_DC: _trip_dc,
    WatchdogId.OUTPUT_SATURATION: _trip_saturation,
    WatchdogId.OUTPUT_RMS: _trip_rms,
    WatchdogId.DEADLINE_MISS_RATE: _trip_deadline,
    WatchdogId.DIVERGENCE: _trip_divergence,
    WatchdogId.HANDOFF_BACKLOG: _trip_handoff,
}


def _band_signals(amplitude: float, blocks: int, phases: int = 2):
    """오기각 시나리오가 공통으로 쓰는 대역 신호 묶음."""

    return _band_sweep_cases(amplitude, blocks, phases=phases)


def _sweep_without_firing(
    watchdog: WatchdogId,
    *,
    amplitude: float,
    blocks: int = 260,
    phases: int = 2,
    isolate_dc: bool = False,
    **drive_kwargs,
) -> list[str]:
    """대역 전체를 몰면서 ``watchdog`` 이 한 번이라도 발동하면 그 사례를 돌려준다."""

    drive_kwargs.setdefault("error_power", 2.0e-5)
    failures: list[str] = []
    for name, wave in _band_signals(amplitude, blocks, phases):
        supervisor = _isolated_dc_supervisor() if isolate_dc else _supervisor()
        _prime_baseline(supervisor, 1.0e-4)
        _drive(
            supervisor,
            blocks=blocks,
            signal_for=lambda i, w=wave: w[i * BLOCK : (i + 1) * BLOCK],
            **drive_kwargs,
        )
        if supervisor.trip_counts[watchdog]:
            failures.append(name)
    return failures


_LEGIT = SafetyLimits.from_config({})


def _no_false_fire_nonfinite() -> list[str]:
    return _sweep_without_firing(
        WatchdogId.NONFINITE_OUTPUT, amplitude=_LEGIT.control_limit
    )


def _no_false_fire_dc() -> list[str]:
    # DC 는 진폭과 직교한다 — **리미터 한계까지** 몰아도 울리면 안 된다.
    return _sweep_without_firing(
        WatchdogId.OUTPUT_DC, amplitude=_LEGIT.control_limit, phases=8, isolate_dc=True
    )


def _no_false_fire_saturation() -> list[str]:
    return _sweep_without_firing(
        WatchdogId.OUTPUT_SATURATION, amplitude=_LEGIT.legitimate_tone_amplitude * 0.9
    )


def _no_false_fire_rms() -> list[str]:
    return _sweep_without_firing(
        WatchdogId.OUTPUT_RMS, amplitude=_LEGIT.legitimate_tone_amplitude * 0.9
    )


def _no_false_fire_deadline() -> list[str]:
    # 한계의 90% 로 민다: 미스율 18% < 20%. 여유 있는 값으로 통과시키는 것은
    # 오기각 방지의 증거가 아니다.
    every = max(2, int(round(1.0 / (0.9 * _LEGIT.deadline_miss_rate_mute))))
    return _sweep_without_firing(
        WatchdogId.DEADLINE_MISS_RATE,
        amplitude=_LEGIT.legitimate_tone_amplitude * 0.9,
        had_output_data=lambda i: i % every != 0,
    )


def _no_false_fire_handoff() -> list[str]:
    every = max(2, int(round(1.0 / (0.9 * _LEGIT.handoff_drop_rate_mute))))
    return _sweep_without_firing(
        WatchdogId.HANDOFF_BACKLOG,
        amplitude=_LEGIT.legitimate_tone_amplitude * 0.9,
        stale_input_samples=lambda i: BLOCK if i % every == 0 else 0,
    )


def _no_false_fire_divergence() -> list[str]:
    """R1 그 자체 — 정숙 베이스라인 + 외부 소음이 대역 전체에서 ANC 를 끄지 않는다."""

    failures: list[str] = []
    external = 2.223e-06 * 10 ** (18.8 / 10.0)
    amplitude = _LEGIT.legitimate_tone_amplitude * 0.9
    for name, wave in _band_signals(amplitude, 400, phases=2):
        supervisor = _supervisor()
        _prime_baseline(supervisor, 2.223e-06)
        verdict, _ = _drive(
            supervisor,
            blocks=400,
            signal_for=lambda i, w=wave: w[i * BLOCK : (i + 1) * BLOCK],
            error_power=lambda _i, _active: external,
        )
        if supervisor.trip_counts[WatchdogId.DIVERGENCE] or verdict is not None:
            failures.append(name)
    return failures


FALSE_POSITIVE_SCENARIOS = {
    WatchdogId.NONFINITE_OUTPUT: _no_false_fire_nonfinite,
    WatchdogId.OUTPUT_DC: _no_false_fire_dc,
    WatchdogId.OUTPUT_SATURATION: _no_false_fire_saturation,
    WatchdogId.OUTPUT_RMS: _no_false_fire_rms,
    WatchdogId.DEADLINE_MISS_RATE: _no_false_fire_deadline,
    WatchdogId.DIVERGENCE: _no_false_fire_divergence,
    WatchdogId.HANDOFF_BACKLOG: _no_false_fire_handoff,
}


@pytest.mark.parametrize("watchdog", list(WatchdogId), ids=lambda w: w.value)
def test_every_watchdog_can_be_made_to_fire(watchdog: WatchdogId):
    """**계약 1.** 워치독을 추가하면 발동 시나리오도 함께 와야 한다.

    S1 이 죽어 있었던 이유는 단순하다 — 아무도 발동시켜 본 적이 없다.
    """

    assert watchdog in TRIP_SCENARIOS, (
        f"{watchdog.value} 를 발동시키는 시나리오가 없습니다 — 발동시킬 수 없는 "
        "워치독은 죽은 코드입니다 (S1 이 그랬습니다)"
    )
    verdict = TRIP_SCENARIOS[watchdog]()
    assert verdict is not None and verdict.mute, f"{watchdog.value} 가 mute 하지 않았습니다"
    assert watchdog in verdict.fired
    assert verdict.messages and all(text.strip() for text in verdict.messages)


@pytest.mark.parametrize("watchdog", list(WatchdogId), ids=lambda w: w.value)
def test_every_watchdog_has_a_band_wide_false_positive_scenario(watchdog: WatchdogId):
    """**계약 2.** 목표 대역 전체·여러 위상·리미터가 허용하는 진폭에서 조용해야 한다.

    발동 증명만 강제하면 워치독이 절대 목표(저역 상쇄를 키우는 것)의 적이 된다.
    실제로 그렇게 됐다 — R2 는 80 Hz 상쇄음을 DC 로 오인했고, 당시 오기각 테스트는
    375 Hz **하나**뿐이었다. 단일 주파수 테스트는 회피 가능하므로 금지다.
    """

    assert watchdog in FALSE_POSITIVE_SCENARIOS, (
        f"{watchdog.value} 의 오발동 반증 시나리오가 없습니다 — 발동만 증명된 "
        "워치독은 운용을 깨뜨립니다 (R1/R2 가 그랬습니다)"
    )
    failures = FALSE_POSITIVE_SCENARIOS[watchdog]()
    assert failures == [], (
        f"{watchdog.value} 가 정상 목표 대역 신호에서 발동했습니다: "
        + "; ".join(failures[:8])
    )


def test_the_false_positive_sweep_actually_covers_the_target_band():
    """스윕이 대역을 실제로 덮는지 — 시나리오가 조용해진 이유가 '안 몰아서'면 안 된다."""

    cases = _band_sweep_cases(0.1, blocks=4, phases=2)
    names = [name for name, _ in cases]
    assert any("80Hz" in name for name in names)
    assert any("1600Hz" in name for name in names)
    assert any("chirp" in name for name in names)
    assert any("noise" in name for name in names)
    tones = _log_tones()
    assert tones[0] == pytest.approx(TARGET_BAND_HZ[0])
    assert tones[-1] == pytest.approx(TARGET_BAND_HZ[1])
    assert len(tones) >= 8


def test_every_watchdog_is_declared_as_a_gate_with_a_negative_fixture():
    """워치독 ↔ 게이트 선언 1:1. 선언에는 FAIL 시키는 fixture 가 필수다."""

    for watchdog in WatchdogId:
        declaration = gate(watchdog.value)  # 선언이 없으면 KeyError
        assert declaration.owner == "src/deep_anc/realtime/safety.py"
        assert declaration.negative_fixture.startswith("tests/test_realtime_safety.py::")

    declared = {item.gate_id for item in GATES}
    assert "runtime_pipeline_handoff_budget" in declared


# ======================================================================================
# 콜백 직접 호출 (스트림 없음 — 실기에서 소리를 내지 않는다)
# ======================================================================================
def _bare_runtime(safety_cfg: dict | None = None) -> RealtimeANC:
    """오디오 장치 없이 ``_callback`` 만 돌릴 수 있는 최소 런타임."""

    anc = object.__new__(RealtimeANC)
    anc.sd = types.SimpleNamespace(CallbackAbort=RuntimeError)
    anc.cfg = {}
    anc.fs, anc.block, anc.hop = FS, BLOCK, BLOCK
    anc.ch_err, anc.ch_ref = 0, 1
    anc.ch_noise, anc.ch_cancel = 0, 1
    anc.reference = "digital"
    anc.digital_reference_lead = 0
    anc.state = RuntimeState(start_on=False)
    anc.err_dc, anc.ref_dc = DCBlocker(0.995), DCBlocker(0.995)
    anc.safety = SafetySupervisor(safety_cfg or {}, FS, BLOCK)
    anc._fade_samples = 0
    anc.anc_gate = FadeGate(0, initial=0.0)
    anc.noise_gate = FadeGate(0, initial=1.0)
    anc.program = NoiseProgram({"type": "silence"}, FS)
    anc.digital_reference_buffer = DigitalReferenceBuffer(0)
    anc.in_ring = SPSCRing(4, BLOCK * 64)
    anc.out_ring = SPSCRing(1, BLOCK * 64)
    anc.handoff_budget = PipelineHandoffBudget.derive(duct_cfg=DUCT, hop=BLOCK)
    anc.err_meter, anc.ctrl_meter = PowerEMA(FS, 0.4), PowerEMA(FS, 0.4)
    anc._last_input_drops = 0
    anc.step_times_ms = []
    anc.xruns = 0
    anc._last_anc = False
    anc._adaptation_hold_samples = 0
    anc.engine = types.SimpleNamespace(secondary_total_length=0)
    anc.record_len, anc.rec_pos, anc.rec = 0, 0, None
    return anc


_INT32_SCALE = 2.0 ** 31


def _pump(anc: RealtimeANC, y: np.ndarray, err_amplitude: float = 0.0, phase: float = 0.0):
    """엔진 출력 한 블록을 넣고 콜백을 한 번 돌린 뒤 상쇄 채널을 돌려준다.

    ``err_amplitude`` 는 에러 마이크에 들어오는 **외부** 음압(300 Hz 톤)이다.
    """

    anc.out_ring.push(np.asarray(y, dtype=np.float32).reshape(1, -1))
    indata = np.zeros((BLOCK, 2), dtype=np.int32)
    if err_amplitude > 0.0:
        t = np.arange(BLOCK, dtype=np.float64) / FS + phase
        wave = err_amplitude * np.sin(2 * np.pi * 300.0 * t)
        indata[:, anc.ch_err] = (wave * _INT32_SCALE * 0.5).astype(np.int32)
    outdata = np.zeros((BLOCK, 2), dtype=np.int16)
    anc._callback(indata, outdata, BLOCK, None, 0)
    return outdata[:, anc.ch_cancel].astype(np.float64) / 32767.0


def test_callback_mutes_anc_when_the_engine_emits_nan():
    """스트림 없이 콜백만 돌려도 mute 가 실제로 걸리는지 확인한다 (S6 종단)."""

    anc = _bare_runtime()
    anc.state.anc_enabled = True
    emitted = _pump(anc, np.full(BLOCK, np.nan, dtype=np.float32))

    assert anc.state.anc_enabled is False, "NaN 을 내는 엔진으로 ANC 가 켜진 채 남았습니다"
    assert np.all(np.isfinite(emitted)) and np.all(emitted == 0.0)
    assert not anc.state.messages.empty()


def test_callback_strips_output_dc_and_then_mutes():
    """S7 종단: 앰프로 나가는 신호에서 DC 가 사라지고, 이어서 ANC 가 꺼진다."""

    anc = _bare_runtime()
    anc.state.anc_enabled = True
    offset = np.full(BLOCK, 0.19, dtype=np.float32)

    last = None
    muted_at = None
    for index in range(200):
        last = _pump(anc, offset, err_amplitude=0.01)
        if not anc.state.anc_enabled:
            muted_at = index
            break

    assert muted_at is not None, "DC 출력이 무한정 앰프로 나갔습니다"
    assert abs(float(np.mean(last))) < 0.5 * 0.19, "DC 차단기가 동작하지 않았습니다"
    assert anc.safety.trip_counts[WatchdogId.OUTPUT_DC] > 0


def test_callback_uses_the_symmetric_backlog_budget():
    """콜백이 예산 값을 실제로 쓰는지 — 출력 링에 쌓여도 최신 블록만 나간다."""

    anc = _bare_runtime()
    anc.state.anc_enabled = True
    for _ in range(3):  # 낡은 블록 3개가 밀려 있어도
        anc.out_ring.push(np.zeros((1, BLOCK), dtype=np.float32))
    emitted = _pump(anc, _sine(0.05))
    rms = float(np.sqrt(np.mean(emitted ** 2)))
    assert rms == pytest.approx(0.05 / np.sqrt(2.0), rel=0.05), (
        "낡은 블록이 나갔습니다 — 백로그 재동기가 동작하지 않습니다"
    )


# --------------------------------------------------------------------------------------
# 종단 운용 시나리오 — 여기가 R1 이 나온 자리다 (예전에는 테스트가 하나도 없었다)
# --------------------------------------------------------------------------------------
def _run_scenario(anc: RealtimeANC, seconds: float, err_amplitude, y_amplitude: float = 0.05):
    """콜백을 ``seconds`` 동안 돌린다. ``err_amplitude`` 는 f(block, anc) -> 진폭."""

    blocks = int(seconds * FS / BLOCK)
    tone = _sine(y_amplitude, freq_hz=300.0, blocks=blocks)
    for index in range(blocks):
        level = err_amplitude(index, anc)
        _pump(
            anc,
            tone[index * BLOCK : (index + 1) * BLOCK],
            err_amplitude=level,
            phase=index * BLOCK / FS,
        )


def test_callback_keeps_anc_on_when_an_external_noise_source_starts(capsys):
    """**R1 종단 재현.** 정숙 5s → ANC ON → 외부 소음 ON 에서 ANC 가 살아 있어야 한다.

    실측 반증에서는 이 경로가 23블록(123 ms) 만에 mute 했고, 이 경로를 도는 테스트는
    저장소에 하나도 없었다.
    """

    anc = _bare_runtime()
    quiet, loud = 0.0015, 0.02

    # [1] ANC OFF · 정숙 5s → 조용한 방의 마이크 플로어가 베이스라인이 된다
    _run_scenario(anc, 5.0, lambda i, a: quiet)
    baseline_quiet = anc.safety.baseline.power
    assert anc.safety.baseline.valid
    assert baseline_quiet < 1.0e-5

    # [2] 운용자가 ANC ON  [3] 외부 소음원 ON
    anc.state.anc_enabled = True
    _run_scenario(anc, 8.0, lambda i, a: loud)

    assert anc.state.anc_enabled is True, (
        "외부 소음원을 켜자 ANC 가 자동으로 꺼졌습니다 (R1 재발) — "
        f"trip={anc.safety.trip_counts}"
    )
    assert anc.safety.divergence_probes == 1, "감시는 했어야 한다"
    assert anc.safety.divergence_rebaselines == 1, "측정으로 베이스라인을 갱신했어야 한다"
    assert anc.safety.baseline.power > 50 * baseline_quiet
    assert anc.safety.trip_counts[WatchdogId.DIVERGENCE] == 0


def test_callback_mutes_when_the_anti_noise_itself_raises_the_error():
    """같은 경로에서 **진짜 발산**은 확실히 잡는다 (프로브가 fail-open 이 아니라는 증거).

    여기서는 에러 마이크 레벨이 우리 출력에 실제로 반응한다: 상쇄음이 나가면 20 dB
    나빠지고, 게이트가 닫히면 원래대로 돌아온다.
    """

    anc = _bare_runtime()
    quiet, worse = 0.002, 0.02

    _run_scenario(anc, 5.0, lambda i, a: quiet)
    anc.state.anc_enabled = True

    def error_level(_index: int, runtime: RealtimeANC) -> float:
        return worse if runtime.anc_gate.current > 0.001 else quiet

    _run_scenario(anc, 8.0, error_level)

    assert anc.state.anc_enabled is False, "발산인데 ANC 가 켜진 채 남았습니다"
    assert anc.safety.trip_counts[WatchdogId.DIVERGENCE] > 0
    assert anc.safety.divergence_probes == 1
    assert anc.safety.divergence_rebaselines == 0


def test_callback_closes_the_output_during_a_divergence_probe():
    """프로브 중 상쇄 채널이 실제로 무음인지 — 콜백이 게이트를 닫는지 확인한다."""

    anc = _bare_runtime()
    _run_scenario(anc, 5.0, lambda i, a: 0.002)
    anc.state.anc_enabled = True

    tone = _sine(0.05, freq_hz=300.0, blocks=1)
    probe_seen = False
    silent_while_probing = True
    for index in range(int(6.0 * FS / BLOCK)):
        emitted = _pump(anc, tone, err_amplitude=0.02, phase=index * BLOCK / FS)
        if anc.safety.probe_active:
            probe_seen = True
            if index > 0 and float(np.max(np.abs(emitted))) > 1.0e-3:
                # 페이드 직후 한 블록은 잔여가 있을 수 있으므로 두 블록 뒤부터 본다
                if anc.anc_gate.current == 0.0:
                    silent_while_probing = False
        if not anc.state.anc_enabled:
            break

    assert probe_seen, "프로브가 일어나지 않았습니다"
    assert silent_while_probing, "프로브 중에 상쇄음이 계속 나갔습니다 (보호가 미뤄졌다)"
