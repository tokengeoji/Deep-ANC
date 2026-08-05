"""실시간 워치독 — **각각이 실제로 발동하는 것을 증명한다** (2026-08-06 실시간 감사).

왜 이 파일이 있는가
------------------
감사에서 확정된 6건은 증상이 달라도 발생기가 둘뿐이었다.

* **발생기 B (실패해본 적 없는 게이트)** — 워치독이 "감시 중"이라고 주장하는데
  그 주장이 반증된 적이 없다. 클립 워치독(S1)은 모델의 tanh 리미터와 **같은 값**을
  재고 있어 clip_fraction 이 구조적으로 항상 0 이었다. 발산 워치독(S3)은 베이스라인이
  0 이면 조용히 비활성이었다. NaN(S6)은 메시지 없이 삼켜졌다.
* **발생기 A (두 도메인 간 시간 정렬 부기)** — 같은 물리량을 두 곳에서 따로 유도하고
  대조하지 않는다. 출력 백로그는 1 hop 인데 입력 백로그는 8 hop 이었고, 어느 쪽도
  학습 플랜트의 ``handoff_extra_samples`` 와 대조되지 않았다. 데드라인 스트릭(S2)은
  교대 미스에 매 블록 리셋돼 영원히 발동하지 않았다.

그래서 이 파일의 규칙은 하나다: **:class:`WatchdogId` 에 이름이 있으면 그것을
발동시키는 시나리오가 여기 있어야 한다.** ``test_every_watchdog_can_be_made_to_fire``
가 열거를 순회하므로 시나리오를 빠뜨린 워치독은 즉시 실패한다.

실기에서 오디오를 내지 않는다 — 전부 콜백/감시자 직접 호출이다.
"""

from __future__ import annotations

import re
import types

import numpy as np
import pytest

from deep_anc.config import REPO_ROOT
from deep_anc.dsp.filters import DCBlocker
from deep_anc.ops.gate_registry import GATES, gate
from deep_anc.realtime.noise_gen import DigitalReferenceBuffer, NoiseProgram
from deep_anc.realtime.ring_buffer import SPSCRing
from deep_anc.realtime.run_realtime import RealtimeANC
from deep_anc.realtime.safety import (
    HARDWARE_PROTECTION_WATCHDOGS,
    BlockObservation,
    BlockVerdict,
    FadeGate,
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


# ======================================================================================
# 도우미
# ======================================================================================
def _supervisor(**safety_cfg) -> SafetySupervisor:
    return SafetySupervisor(safety_cfg, FS, BLOCK)


def _tanh_limited(raw: np.ndarray, limit: float = 0.2) -> np.ndarray:
    """모델 출력단과 **같은 연산** (models/hybrid_anc.py: ``limit·tanh(y/limit)``)."""

    return (limit * np.tanh(np.asarray(raw, dtype=np.float64) / limit)).astype(np.float32)


def _sine(amplitude: float, freq_hz: float = 375.0, blocks: int = 1) -> np.ndarray:
    """375 Hz = 블록당 정확히 2주기.

    같은 블록을 반복해 넣어도 **진짜 DC 가 생기지 않는다.** 예컨대 300 Hz(1.6주기)로
    같은 블록을 반복하면 그 신호의 실제 평균이 0.03 이 되어 DC 워치독이 옳게 발동한다
    — 시험 신호가 만든 DC 였다.
    """

    t = np.arange(blocks * BLOCK, dtype=np.float64) / FS
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _drive(
    supervisor: SafetySupervisor,
    *,
    blocks: int,
    signal_for,
    had_output_data=lambda i: True,
    error_power: float = 1.0e-4,
    baseline_power: float = 1.0e-4,
    baseline_valid: bool = True,
    stale_input_samples: int = 0,
    anc_on: bool = True,
) -> tuple[BlockVerdict | None, int]:
    """감시자를 N 블록 돌리고 **처음 mute 한 판정**과 그 블록 번호를 돌려준다."""

    for index in range(blocks):
        report = supervisor.limit_output(signal_for(index))
        verdict = supervisor.check_block(
            BlockObservation(
                anc_on=anc_on,
                output=report,
                error_power=error_power,
                baseline_power=baseline_power,
                baseline_valid=baseline_valid,
                had_output_data=bool(had_output_data(index)),
                stale_input_samples=stale_input_samples,
            )
        )
        if verdict.mute:
            return verdict, index
    return None, blocks


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
    verdict, _ = _drive(
        supervisor,
        blocks=1000,
        signal_for=lambda i: _sine(0.05),
        had_output_data=lambda i: i % 50 != 0,  # 미스율 2% < 20%
    )
    assert verdict is None


# ======================================================================================
# S3 — 감시 불가 상태를 정상으로 취급하지 않는다
# ======================================================================================
def test_divergence_watchdog_detects_the_missing_baseline_instead_of_going_quiet():
    """``baseline_power == 0`` 이면 옛 워치독은 조용히 비활성이었다 — 이제 fail-closed."""

    supervisor = _supervisor()
    assert not supervisor.baseline_is_valid(0.0, initialized=True)
    assert not supervisor.baseline_is_valid(1.0e-4, initialized=False)

    verdict, index = _drive(
        supervisor,
        blocks=2000,
        signal_for=lambda i: _sine(0.05),
        error_power=1.0e-4,
        baseline_power=0.0,
        baseline_valid=False,
    )
    assert verdict is not None, "베이스라인 없이 상쇄음을 무한정 내보내고 있습니다"
    assert WatchdogId.DIVERGENCE in verdict.fired
    expected = supervisor.windows.baseline_grace_blocks
    assert index == expected - 1, f"유예 {expected} 블록 후 즉시 꺼져야 합니다"
    assert any("베이스라인이 없습니다" in text for text in verdict.messages)


def test_divergence_watchdog_detects_error_power_above_the_baseline():
    """정상 경로 — 유효 베이스라인이 있을 때 발산은 여전히 잡힌다."""

    supervisor = _supervisor()
    verdict, index = _drive(
        supervisor,
        blocks=400,
        signal_for=lambda i: _sine(0.05),
        error_power=1.0e-2,
        baseline_power=1.0e-5,
        baseline_valid=True,
    )
    assert verdict is not None
    assert WatchdogId.DIVERGENCE in verdict.fired
    assert index < supervisor.windows.divergence_blocks


def test_divergence_watchdog_survives_alternating_divergence():
    """발산도 스트릭이 아니라 비율이다 — 한 블록 걸러 정상이어도 잡힌다."""

    supervisor = _supervisor()
    powers = [1.0e-2, 1.0e-6]
    fired = None
    for index in range(400):
        report = supervisor.limit_output(_sine(0.05))
        verdict = supervisor.check_block(
            BlockObservation(
                anc_on=True,
                output=report,
                error_power=powers[index % 2],
                baseline_power=1.0e-5,
                baseline_valid=True,
                had_output_data=True,
            )
        )
        if verdict.mute:
            fired = verdict
            break
    assert fired is not None, "교대 발산을 놓쳤습니다 (스트릭 회귀)"
    assert WatchdogId.DIVERGENCE in fired.fired


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
            baseline_power=1.0e-4,
            baseline_valid=True,
            had_output_data=True,
        )
    )
    assert verdict.mute
    assert WatchdogId.NONFINITE_OUTPUT in verdict.fired
    assert any("NaN/Inf" in text for text in verdict.messages), "메시지 없이 삼키면 안 된다"


# ======================================================================================
# S7 — 출력 DC (하드웨어 보호)
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
    offset = np.full(BLOCK, 0.19, dtype=np.float32)
    verdict, index = _drive(supervisor, blocks=400, signal_for=lambda i: offset)

    assert verdict is not None, "출력 DC 워치독이 발동하지 않았습니다 (하드웨어 손상 경로)"
    assert WatchdogId.OUTPUT_DC in verdict.fired
    assert index < supervisor.windows.dc_blocks
    assert any("보이스코일" in text for text in verdict.messages)


def test_output_dc_watchdog_ignores_a_zero_mean_tone():
    """오기각 방지 — 정상 상쇄음(영평균)은 DC 워치독을 건드리지 않는다."""

    supervisor = _supervisor()
    verdict, _ = _drive(supervisor, blocks=600, signal_for=lambda i: _sine(0.15)[:BLOCK])
    assert verdict is None or WatchdogId.OUTPUT_DC not in verdict.fired


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


def test_handoff_backlog_watchdog_detects_dropped_stale_input():
    """오래된 입력을 버리는 상태가 지속되면 ANC 를 끈다.

    입력을 버렸다는 것은 추론이 뒤처져 실효 핸드오프가 학습 가정에서 이탈했다는
    뜻이고, 그 지연의 안티노이즈는 상쇄가 아니라 증폭이 될 수 있다.
    """

    supervisor = _supervisor()
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


# ======================================================================================
# 오기각 방지
# ======================================================================================
def test_clean_anti_noise_trips_no_watchdog():
    """정상 운용 3초에서는 어떤 워치독도 발동하지 않는다.

    워치독을 세게 걸수록 오기각 위험이 커진다. 발동 증명과 **짝**으로 이 테스트가
    있어야 "발동은 하지만 쓸 수 없는" 워치독이 되지 않는다.
    """

    supervisor = _supervisor()
    blocks = int(3.0 * FS / BLOCK)
    tone = _sine(0.05, blocks=blocks)
    verdict, _ = _drive(
        supervisor,
        blocks=blocks,
        signal_for=lambda i: tone[i * BLOCK : (i + 1) * BLOCK],
        error_power=2.0e-5,
        baseline_power=1.0e-4,
        baseline_valid=True,
    )
    assert verdict is None
    assert all(count == 0 for count in supervisor.trip_counts.values())


def test_watchdogs_are_idle_while_anc_is_off():
    """ANC OFF 구간의 링버퍼 언더런으로는 아무 일도 일어나지 않는다."""

    supervisor = _supervisor()
    verdict, _ = _drive(
        supervisor,
        blocks=1000,
        signal_for=lambda i: np.full(BLOCK, np.nan, dtype=np.float32),
        had_output_data=lambda i: False,
        baseline_valid=False,
        anc_on=False,
    )
    assert verdict is None


# ======================================================================================
# 측정 모드 — 하드웨어 보호는 내려가지 않는다
# ======================================================================================
def test_measurement_mode_downgrades_performance_watchdogs_but_not_hardware_ones():
    """캘리브레이션은 의도적으로 큰 출력을 낸다. 그렇다고 DC 를 흘려도 되는 것은 아니다."""

    supervisor = _supervisor(measurement_mode=True)
    hot = _tanh_limited(_sine(50.0))
    verdict, _ = _drive(supervisor, blocks=400, signal_for=lambda i: hot)
    assert verdict is None, "측정 모드에서 성능 워치독이 mute 하면 캘리브레이션이 깨진다"
    assert supervisor.trip_counts[WatchdogId.OUTPUT_SATURATION] > 0, "자문 보고는 남아야 한다"

    supervisor = _supervisor(measurement_mode=True)
    offset = np.full(BLOCK, 0.19, dtype=np.float32)
    verdict, _ = _drive(supervisor, blocks=400, signal_for=lambda i: offset)
    assert verdict is not None and verdict.mute
    assert set(verdict.fired) & HARDWARE_PROTECTION_WATCHDOGS


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


def test_safety_config_refuses_to_recreate_the_dead_watchdog():
    """``saturation_ratio = 1.0`` 은 리미터와 같은 값을 재는 것 = S1 의 재현이다."""

    with pytest.raises(ValueError, match="죽은 워치독"):
        SafetyLimits.from_config({"saturation_ratio": 1.0})
    with pytest.raises(ValueError):
        SafetyLimits.from_config({"divergence_ratio": 0.5})
    with pytest.raises(ValueError):
        SafetyLimits.from_config({"deadline_miss_rate_mute": 0.0})


def test_safety_limits_are_frozen():
    """유도 임계값이 나중에 조용히 바뀌면 단일 출처가 무의미하다."""

    limits = SafetyLimits.from_config({})
    with pytest.raises(Exception):
        limits.control_limit = 0.9


def test_block_observation_rejects_impossible_values():
    """음수/비유한 파워는 관측 자체가 만들어지지 않는다."""

    supervisor = _supervisor()
    report = supervisor.limit_output(_sine(0.05))
    for bad in ({"error_power": -1.0}, {"baseline_power": float("nan")},
                {"stale_input_samples": -1}):
        kwargs = {
            "anc_on": True,
            "output": report,
            "error_power": 1.0e-4,
            "baseline_power": 1.0e-4,
            "baseline_valid": True,
            "had_output_data": True,
        }
        kwargs.update(bad)
        with pytest.raises(ValueError):
            BlockObservation(**kwargs)


def test_safety_stays_inside_the_callback_deadline_budget():
    """핫패스 회귀 방지 — 감시자가 콜백 마감을 잡아먹지 않는지 실측한다.

    마감은 5.33 ms 이고 콜백 본체가 이미 0.46 ms 를 쓴다. 워치독을 늘리다 보면
    검증을 샘플 단위로 넣기 쉬운데, 그러면 xrun 이 나고 **워치독이 사고의 원인**이
    된다. 실측 ≈ 0.31 ms/블록. 부하 스파이크에 흔들리지 않도록 여러 배치의
    최소값으로 판정한다.
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
                    baseline_power=1.0e-4,
                    baseline_valid=True,
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
# 메타 — 열거된 워치독은 전부 발동 증명과 게이트 선언을 갖는다
# ======================================================================================
def _trip_nonfinite() -> BlockVerdict:
    supervisor = _supervisor()
    verdict, _ = _drive(
        supervisor, blocks=4, signal_for=lambda i: np.full(BLOCK, np.nan, dtype=np.float32)
    )
    return verdict


def _trip_dc() -> BlockVerdict:
    supervisor = _supervisor()
    verdict, _ = _drive(
        supervisor, blocks=400, signal_for=lambda i: np.full(BLOCK, 0.19, dtype=np.float32)
    )
    return verdict


def _trip_saturation() -> BlockVerdict:
    supervisor = _supervisor()
    verdict, _ = _drive(
        supervisor, blocks=400, signal_for=lambda i: _tanh_limited(_sine(50.0))
    )
    return verdict


def _trip_rms() -> BlockVerdict:
    supervisor = _supervisor()
    verdict, _ = _drive(supervisor, blocks=400, signal_for=lambda i: _sine(0.185))
    return verdict


def _trip_deadline() -> BlockVerdict:
    supervisor = _supervisor()
    verdict, _ = _drive(
        supervisor,
        blocks=800,
        signal_for=lambda i: _sine(0.05),
        had_output_data=lambda i: i % 2 == 0,
    )
    return verdict


def _trip_divergence() -> BlockVerdict:
    supervisor = _supervisor()
    verdict, _ = _drive(
        supervisor,
        blocks=400,
        signal_for=lambda i: _sine(0.05),
        error_power=1.0e-2,
        baseline_power=1.0e-5,
    )
    return verdict


def _trip_handoff() -> BlockVerdict:
    supervisor = _supervisor()
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


@pytest.mark.parametrize("watchdog", list(WatchdogId), ids=lambda w: w.value)
def test_every_watchdog_can_be_made_to_fire(watchdog: WatchdogId):
    """**이 파일의 계약.** 워치독을 추가하면 발동 시나리오도 함께 와야 한다.

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
    anc.baseline_power, anc.baseline_init = 0.0, False
    anc._last_input_drops = 0
    anc.step_times_ms = []
    anc.xruns = 0
    anc._last_anc = False
    anc._adaptation_hold_samples = 0
    anc.engine = types.SimpleNamespace(secondary_total_length=0)
    anc.record_len, anc.rec_pos, anc.rec = 0, 0, None
    return anc


def _pump(anc: RealtimeANC, y: np.ndarray) -> np.ndarray:
    """엔진 출력 한 블록을 넣고 콜백을 한 번 돌린 뒤 상쇄 채널을 돌려준다."""

    anc.out_ring.push(np.asarray(y, dtype=np.float32).reshape(1, -1))
    indata = np.zeros((BLOCK, 2), dtype=np.int32)
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
    for index in range(120):
        last = _pump(anc, offset)
        if not anc.state.anc_enabled:
            muted_at = index
            break

    assert muted_at is not None, "DC 출력이 무한정 앰프로 나갔습니다"
    # mute 시점(≈50 ms)에 이미 차단기가 DC 의 절반 이상을 걷어냈다. 완전 제거는
    # test_output_dc_blocker_removes_the_offset_before_it_reaches_the_amplifier 가 본다.
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
