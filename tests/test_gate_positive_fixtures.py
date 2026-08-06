"""오발동 반증 — **정상 산출물을 운용 범위 끝까지 몰아도 게이트가 울리지 않는가.**

왜 이 파일이 있는가 (2026-08-06 반증 #13, 군집 B 의 나머지 절반)
--------------------------------------------------------------
``tests/test_gate_registry.py`` 는 "게이트를 FAIL 시키는 fixture 가 있는가" 만
강제했다. 그 절반은 옳았지만 나머지 절반이 비어 있었다: **"정상 입력에서 발동하지
않는가" 를 실제 운용 범위 전체로 몰아본 게이트가 하나도 없었다.**

이 저장소에서 게이트의 반응은 전부 차단이다.
  · readiness/QA 게이트 → 학습을 시작하지 못한다.
  · 런타임 워치독      → mute, 즉 상쇄 0 dB = 절대목표 2 의 최악값.
그래서 오발동 한 건의 대가는 결함 한 건의 대가와 같다. 그런데 실측으로 확인된 오발동이
이미 있었다: 제대로 재정렬된 세션 9개 중 4개(44%)를 QA 지연 게이트가 떨어뜨리고 있었고,
그 사실은 "정상 데이터로 전수를 돌려본 사람" 이 나올 때까지 아무도 몰랐다.

이 파일의 규칙: **여유를 주지 않는다.** 정상값을 한계의 90% 지점, 목표 대역 최저·최고
주파수, 최소 세션 수, 허용 spread 최대값처럼 **경계에 붙여** 넣고 PASS 를 요구한다.
여유 있는 값으로 한 번 통과시켜 보는 것은 오기각 방지의 증거가 아니다.

실기에서 소리를 내지 않는다 — 전부 순수 함수·감시자 직접 호출이다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from deep_anc.config import REPO_ROOT

# 기존 테스트 모듈의 픽스처를 **재사용**한다. 같은 세션/신호 생성기를 두 벌 만드는 것이
# 이 저장소가 반복한 발생기 A 이므로, 정상 짝도 negative 짝과 같은 소재로 만든다.
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ======================================================================================
# 런타임 워치독 7종 — 목표 대역 전체를 한계의 90% 로 3초 몰아본다
# ======================================================================================
from test_realtime_safety import BLOCK, FS, _drive, _prime_baseline, _supervisor  # noqa: E402

from deep_anc.realtime.safety import BlockObservation, WatchdogId  # noqa: E402

# duct.yaml realistic_target_band_hz = [80, 1600] 의 양 끝과 S 신뢰대역 [150, 1600] 의
# 양 끝을 전부 포함한다. 80Hz 는 블록(5.33ms)당 0.43주기라 DC 워치독이 가장 헷갈리는
# 주파수이고, 1600Hz 는 덕트 평면파 컷오프 직전이다.
OPERATING_BAND_HZ = (80.0, 150.0, 300.0, 600.0, 1000.0, 1600.0)


def test_every_watchdog_stays_silent_across_the_operating_band_at_90_percent_of_its_limit():
    """정상 운용을 **한계의 90%** 로 3초씩, 목표 대역 전 주파수에서 돌린다.

    동시에 미는 축 (전부 각 한계의 90%):
      · 출력 RMS      = 0.9 × 0.12 (control_limit 0.2 × output_rms_ratio 0.6)
      · 데드라인 미스율 = 0.9 × 0.2 = 18% (교대가 아니라 실제로 미스한다)
      · 입력 백로그 드롭율 = 0.9 × 0.2 = 18%
      · 에러/베이스라인 = 0.9 × 4.0 = 3.6 배 (외부 소음 — 출력을 닫아도 안 내려간다)
      · 주파수        = 80 / 150 / 300 / 600 / 1000 / 1600 Hz, 위상 8종

    마지막 축이 반증 #11 이 지적한 시나리오다: 조용한 방에서 잡힌 베이스라인 위로
    외부 소음원이 켜지는 것. 그때 mute 하면 상쇄는 0 dB 가 된다 — 결함과 같은 값이다.
    """

    limits = _supervisor().limits
    amplitude = 0.9 * limits.rms_threshold * np.sqrt(2.0)     # rms = 90% of 0.12
    miss_rate = 0.9 * limits.deadline_miss_rate_mute          # 0.18
    drop_rate = 0.9 * limits.handoff_drop_rate_mute           # 0.18
    ratio = 0.9 * limits.divergence_ratio                     # 3.6
    baseline = 1.0e-4
    blocks = int(3.0 * FS / BLOCK)
    miss_every = max(2, int(round(1.0 / miss_rate)))
    drop_every = max(2, int(round(1.0 / drop_rate)))

    fired: list[str] = []
    for freq in OPERATING_BAND_HZ:
        for phase in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            t = np.arange(blocks * BLOCK, dtype=np.float64) / FS
            wave = (amplitude * np.sin(2.0 * np.pi * freq * t + phase)).astype(np.float32)
            supervisor = _supervisor()
            _prime_baseline(supervisor, baseline)
            verdict, index = None, blocks
            for i in range(blocks):
                report = supervisor.limit_output(wave[i * BLOCK : (i + 1) * BLOCK])
                block = supervisor.check_block(
                    BlockObservation(
                        anc_on=True,
                        output=report,
                        # 외부 소음: 상쇄 출력을 닫아도 에러 파워가 그대로다.
                        error_power=baseline * ratio,
                        anc_output_active=not supervisor.probe_active,
                        had_output_data=(i % miss_every != 0),
                        # 드롭율 18% — **블록의 18%** 에서만 오래된 입력을 버린다.
                        stale_input_samples=(BLOCK if i % drop_every == 0 else 0),
                    )
                )
                if block.mute:
                    verdict, index = block, i
                    break
            if verdict is not None:
                fired.append(
                    f"{freq:.0f}Hz phase={phase:.2f} block={index} "
                    f"{[item.value for item in verdict.fired]}"
                )
    assert fired == [], "정상 운용에서 워치독이 발동했습니다: " + "; ".join(fired[:6])


def test_the_same_battery_one_notch_past_the_limit_does_fire():
    """오기각 방지 짝이 '게이트가 꺼져 있어서 통과' 가 아님을 못박는다.

    같은 신호를 한계 위(RMS 1.1배)로 올리면 같은 감시자가 mute 한다.
    """

    limits = _supervisor().limits
    amplitude = 1.1 * limits.rms_threshold * np.sqrt(2.0)
    blocks = int(3.0 * FS / BLOCK)
    t = np.arange(blocks * BLOCK, dtype=np.float64) / FS
    wave = (amplitude * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)
    supervisor = _supervisor()
    _prime_baseline(supervisor, 1.0e-4)
    verdict, _ = _drive(
        supervisor,
        blocks=blocks,
        signal_for=lambda i: wave[i * BLOCK : (i + 1) * BLOCK],
        error_power=1.0e-4,
    )
    assert verdict is not None
    assert WatchdogId.OUTPUT_RMS in verdict.fired


# ======================================================================================
# 런타임 시작 게이트 — 출하 설정 그대로, 경계값에서
# ======================================================================================
def test_runtime_lead_gate_accepts_the_measured_lead_116_exactly():
    """런타임 lead 게이트: 확정 플랜트 lead **116 정확 일치**에서 통과한다 (여유 0).

    lead = S 1462 + handoff 256 − P 1602 = 116. 한 샘플만 어긋나도 거부돼야 한다
    (실측: δ=16 샘플이면 600Hz 에서 부호가 뒤집혀 +1.40 dB 증폭이다).
    """

    from deep_anc.realtime.run_realtime import validate_digital_reference_lead

    assert validate_digital_reference_lead("digital", 116, 116) == 116
    with pytest.raises(ValueError):
        validate_digital_reference_lead("digital", 116, 115)


def test_runtime_input_preflight_accepts_a_quiet_room_at_90_percent_of_its_limits(
    monkeypatch,
):
    """입력 프리플라이트: 유효 판정 하한의 90% 지점에서도 정상으로 본다.

    · RMS −72 dBFS (하한 −80 dBFS 의 90% 지점)
    · 클립 비율 0.0045 (한계 0.005 의 90%)
    · 고유 코드 8개 = 최소값 정확히
    죽은 채널 검출은 "고정값" 을 잡는 것이지 "조용한 방" 을 잡는 것이 아니다.
    """

    from deep_anc.audio_io import analyze_int32_input_probe
    from deep_anc.realtime import run_realtime

    rng = np.random.default_rng(20260806)
    frames = 48_000
    scale = 10.0 ** (-72.0 / 20.0) * (2.0 ** 31)
    raw = np.round(scale * rng.standard_normal((frames, 2)) * np.sqrt(1.0)).astype(np.int32)
    clip_count = int(round(0.0045 * frames))
    raw[rng.choice(frames, size=clip_count, replace=False), 0] = 2 ** 31 - 1
    report = analyze_int32_input_probe(raw)
    assert [item["valid"] for item in report["channels"]] == [True, True], report

    monkeypatch.setattr(run_realtime, "capture_input_probe", lambda *a, **k: report)
    cfg = {"reference": "digital", "hardware": {"audio": {}}}
    assert run_realtime.input_preflight(cfg, seconds=0.1) is True


def test_fxlms_adaptation_gate_accepts_the_one_fully_safe_combination():
    """FxLMS 적응 게이트: 8개 조건이 **전부** 만족된 1가지 조합에서는 허용한다.

    fail-closed 는 "아무 때도 안 켜진다" 가 아니다. 8축 중 하나라도 어긋나면 막고,
    전부 맞으면 켜져야 한다 — 그러지 않으면 적응이 영원히 죽은 코드다.
    """

    from deep_anc.realtime.run_realtime import fxlms_adaptation_allowed

    ready = {
        "requested": True,
        "full_anc_gain": True,
        "full_noise_gain": True,
        "hold_samples": 0,
        "output_clip_fraction": 0.0,
        "input_clip_fraction": 0.0,
        # 게이트 하한 1.0e-12 의 90% 위 — 경계 바로 위의 정상 레퍼런스 파워.
        "reference_power": 1.0e-12 / 0.9,
        "stream_ok": True,
    }
    assert fxlms_adaptation_allowed(**ready)
    assert len(ready) == 8
    assert not fxlms_adaptation_allowed(**{**ready, "reference_power": 1.0e-12})


def test_handoff_budget_accepts_the_shipped_256_sample_hop():
    """핸드오프 예산: 출하 duct.yaml 의 handoff 256 에서 **정확히 1 hop** 은 통과한다.

    negative 짝은 입력 8 hop / 출력 1 hop 비대칭을 만들지 못하게 하는 것이고,
    이쪽은 정상 구성(대칭 1 hop)이 거부되지 않는지 본다.
    """

    from deep_anc.config import load_yaml
    from deep_anc.realtime.safety import PipelineHandoffBudget

    duct = load_yaml(REPO_ROOT / "configs/duct.yaml")
    budget = PipelineHandoffBudget.derive(duct_cfg=duct, hop=256)
    assert budget.handoff_samples == 256
    assert budget.input_keep_backlog_samples == budget.output_keep_backlog_samples == 256
    assert budget.effective_handoff_samples == 256


# ======================================================================================
# 실측 세션 QA — 최소 길이/최소 레벨/최대 클립 비율의 경계에서
# ======================================================================================
from test_recorded_qa import _band_noise, _write_session  # noqa: E402

from deep_anc.data.recorded_qa import (  # noqa: E402
    RecordedQASettings,
    validate_recorded_sessions,
)

QA_SEGMENT_SAMPLES = 48_000        # 1 초
QA_LEAD_SAMPLES = 116              # 확정 플랜트 lead


def _qa_settings(**overrides) -> RecordedQASettings:
    values = dict(
        sample_rate=FS,
        segment_samples=QA_SEGMENT_SAMPLES,
        digital_reference_lead_samples=QA_LEAD_SAMPLES,
        block_frames=8192,
        required_splits=("train", "val", "test"),
        # 목표 대역 최저(150) ~ 최고(1600). 좁은 대역에서만 보면 고역 붕괴를 놓친다.
        alignment_band_hz=(150.0, 1600.0),
    )
    values.update(overrides)
    return RecordedQASettings(**values)


def _healthy_session(
    session: Path,
    *,
    session_id: str,
    group_id: str,
    family: str,
    split: str,
    frames: int,
    scale: float = 1.0,
    clip_fraction: float = 0.0,
    seed: int = 20260806,
) -> dict:
    """정상 세션 하나. ``scale`` 로 레벨을, ``clip_fraction`` 으로 클립 비율을 민다."""

    source = _band_noise(frames, seed)
    rng = np.random.default_rng(seed + 7)
    err = 0.8 * np.roll(source, 120) + 0.01 * rng.standard_normal(frames)
    ref = 0.9 * np.roll(source, 60) + 0.01 * rng.standard_normal(frames)
    mics = scale * np.stack([err, ref], axis=1)
    if clip_fraction > 0.0:
        count = int(round(clip_fraction * frames))
        index = rng.choice(frames, size=count, replace=False)
        mics[index, 0] = 0.995
    entry = _write_session(
        session,
        session_id=session_id,
        group_id=group_id,
        family=family,
        frames=frames,
        mics=mics.astype(np.float32),
        source=(scale * source).astype(np.float32),
    )
    entry["split"] = split
    return entry


def _qa_manifest(root: Path, **kwargs) -> list[dict]:
    """계열 4종 × split 3종 = 필수 커버리지를 **정확히** 채운 최소 manifest."""

    entries = []
    for family in ("speech", "music", "environment", "machine"):
        for split in ("train", "val", "test"):
            name = f"{family}-{split}"
            entries.append(
                _healthy_session(
                    root / name,
                    session_id=name,
                    group_id=f"group-{name}",
                    family=family,
                    split=split,
                    seed=20260806 + 17 * len(entries),
                    **kwargs,
                )
            )
    return entries


def test_recorded_qa_gates_pass_at_the_minimum_length_and_full_coverage(tmp_path):
    """QA 전 게이트가 **정확히 최소 길이**의 정상 세션 12개에서 PASS 한다.

    몰아본 경계:
      · 길이 = segment 48000 + lead 116 + 1 = 48117 샘플 (한 샘플 더 짧으면 FAIL)
      · 계열 4종 × split 3종 = 필수 커버리지를 정확히 채움 (여유 0)
      · group_id 가 split 을 넘지 않음 (누수 0건)
      · 정렬 대역 150–1600Hz — 목표 대역 최저·최고 주파수 전체
    """

    settings = _qa_settings()
    frames = settings.minimum_frames
    assert frames == QA_SEGMENT_SAMPLES + QA_LEAD_SAMPLES + 1
    entries = _qa_manifest(tmp_path / "sessions", frames=frames)

    report = validate_recorded_sessions(entries, settings)

    failures = [
        (session["session_id"], session["errors"])
        for session in report["sessions"]
        if session["errors"]
    ]
    assert failures == [], failures
    assert report["errors"] == [], report["errors"]
    assert report["ok"], report


def test_recorded_qa_level_and_clip_gates_pass_at_90_percent_of_their_limits(tmp_path):
    """레벨·클리핑 게이트를 각각 한계의 90% 로 민다.

    · 클립 비율 = 0.9 × 0.005 = 0.0045 (한계 바로 아래)
    · 마이크 RMS = 하한 −80 dBFS 의 90% 지점인 −72 dBFS 부근 (그 아래면 FAIL)
    """

    settings = _qa_settings()
    frames = settings.minimum_frames

    clipped = _qa_manifest(tmp_path / "clip", frames=frames, clip_fraction=0.0045)
    report = validate_recorded_sessions(clipped, settings)
    assert [s["errors"] for s in report["sessions"] if s["errors"]] == []
    assert report["ok"], report

    # 레벨 하한은 **두 곳**에 따로 있다 (발생기 A — 이 테스트가 그 사실을 고정한다):
    #   · recorded_qa.min_mic_rms_dbfs      = −80 dBFS
    #   · timeline.TimelineSettings.min_window_rms = 2.0e−4  (= −74 dBFS)
    # 둘 중 **더 센 쪽**이 실제 하한이므로, 정상 경계는 −74 dBFS 쪽에서 잡아야 한다.
    # 여기서는 그 값의 1.1배(여유 10%)까지 낮춘다.
    from deep_anc.data.timeline import TimelineSettings

    floor_rms = float(TimelineSettings(sample_rate=FS).min_window_rms)
    base_rms = float(np.sqrt(np.mean(np.square(_band_noise(frames, 20260806)))))
    quiet = _qa_manifest(
        tmp_path / "quiet", frames=frames, scale=1.1 * floor_rms / base_rms
    )
    report = validate_recorded_sessions(quiet, settings)
    assert [s["errors"] for s in report["sessions"] if s["errors"]] == []
    assert report["ok"], report
    measured = report["sessions"][0]["audio"]["source"]["rms_dbfs"][0]
    assert -80.0 < measured < -70.0, measured


def _jitter_sessions(root: Path, sigma: float, frames: int) -> list[dict]:
    """창마다 지연이 ``sigma`` 규모로 흔들리는 정상 세션 12개."""

    entries: list[dict] = []
    for index, (family, split) in enumerate(
        [(f, s) for f in ("speech", "music", "environment", "machine")
         for s in ("train", "val", "test")]
    ):
        seed = 20260806 + 31 * index
        source = _band_noise(frames, seed)
        rng = np.random.default_rng(seed + 3)
        # 창(1초)마다 지연을 sigma 규모로 흔든다. 정수 샘플 롤이라 신호 내용은
        # 그대로다 — 흔들리는 것은 시간축뿐이다.
        err = np.zeros(frames)
        step = FS
        jitter = rng.normal(0.0, sigma, size=frames // step + 1)
        for start in range(0, frames, step):
            stop = min(frames, start + step)
            lag = 120 + int(round(jitter[start // step]))
            err[start:stop] = 0.8 * np.roll(source, lag)[start:stop]
        err += 0.01 * rng.standard_normal(frames)
        ref = 0.9 * np.roll(source, 60) + 0.01 * rng.standard_normal(frames)
        name = f"{family}-{split}"
        entries.append(
            _write_session(
                root / name,
                session_id=name,
                group_id=f"group-{name}",
                family=family,
                frames=frames,
                mics=np.stack([err, ref], axis=1).astype(np.float32),
                source=source.astype(np.float32),
            )
        )
        entries[-1]["split"] = split
    return entries


def test_recorded_qa_delay_gates_pass_at_90_percent_of_the_robust_limits(tmp_path):
    """지연 안정성 게이트를 **실제로 도달 가능한 경계**까지 민다 — 오기각 44% 사고의 짝.

    2026-08-06 이전: 선언된 한계가 robust-std **8.0** 이었는데 같은 데이터에 걸린
    코히런스 하한 0.60 이 **먼저** 묶었다. 12 초 세션에 창별 지연 흔들림 σ 를 키우며 실측::

        σ=1.0 coh 0.974 rstd 1.24 | σ=3.0 coh 0.880 rstd 2.57
        σ=5.0 coh 0.731 rstd 3.49 | σ=6.0 coh 0.623 rstd 4.92
        σ=7.2 coh 0.501 rstd 6.00 → **코히런스 게이트가 FAIL** (지연 게이트는 여전히 통과)

    즉 robust-std 의 90% 지점(7.2)은 물리적으로 도달할 수 없었다. 같은 물리량(시간축
    안정성)에 두 임계가 따로 걸려 있고 서로 대조되지 않는 발생기 A 였다.

    **지금은 둘이 하나의 선언에서 유도된다** — 지터 상한은
    ``σ_max = (fs/2πf_top)·√(−ln coh_min)`` 로 신뢰대역 상단 1600 Hz 와 코히런스 하한
    0.60 에서 나온다(= 3.41 샘플). 그래서 이 픽스처는 지연 게이트를 그 한계의 94% 까지
    밀고, 코히런스는 하한 위에 남는다. 새 실측(σ 를 키우며)::

        σ=2.0 rstd 2.47 저역coh 0.930 고역coh 0.896 | σ=2.5 rstd 3.21 저역 0.895 고역 0.843
        σ=3.0 rstd 3.78 → 지연 게이트 FAIL (12개 중 3개)

    실측 정상 세션의 robust-std 는 1.22~2.99 였으므로 이 픽스처(3.21)는 실기 최악보다
    나쁜 정상값이다.
    """

    from deep_anc.dsp.invariants import (
        MAX_STREAM_DELAY_P95_P5_SAMPLES,
        MAX_STREAM_DELAY_ROBUST_STD_SAMPLES,
    )

    settings = _qa_settings()
    frames = 12 * FS                      # 12 초 = 창 180개 이상
    # 유도된 상한 3.41 샘플의 94% 지점에 붙게 고른 값이다.
    entries = _jitter_sessions(tmp_path / "jitter", 2.5, frames)

    report = validate_recorded_sessions(entries, settings)

    failures = [
        (session["session_id"], session["errors"])
        for session in report["sessions"]
        if session["errors"]
    ]
    assert failures == [], failures
    coherence = [
        float(session["alignment"]["source_err_coherence"])
        for session in report["sessions"]
    ]
    # 코히런스는 하한 위에 남아야 한다 — 이제 먼저 묶는 것은 지연 게이트다.
    assert min(coherence) > 0.60, coherence
    high = [
        float(session["alignment"]["source_err_coherence_high"])
        for session in report["sessions"]
    ]
    # 고역도 함께 판정된다 (2026-08-06 이전에는 이 값을 보는 게이트가 0개였다).
    assert min(high) > 0.60, high
    measured = [
        float(session["alignment"]["source_err_delay_robust_std_samples"])
        for session in report["sessions"]
    ]
    # 실제로 한계까지 밀었는가 — 90% 를 넘어야 "경계에서 통과" 라고 말할 수 있다.
    assert max(measured) <= MAX_STREAM_DELAY_ROBUST_STD_SAMPLES
    assert max(measured) >= 0.90 * MAX_STREAM_DELAY_ROBUST_STD_SAMPLES, measured
    spreads = [
        float(session["alignment"]["source_err_delay_p95_p5_samples"])
        for session in report["sessions"]
    ]
    assert max(spreads) <= MAX_STREAM_DELAY_P95_P5_SAMPLES


def test_the_delay_gate_now_implies_the_coherence_gate(tmp_path):
    """두 게이트가 **함께 묶이는지** 를 측정으로 강제한다 (2026-08-06 에 계약이 바뀌었다).

    옛 상태: 한계가 robust-std **8.0** 이라 σ=7.2 에서 코히런스만 FAIL 하고 지연은
    통과했다. 같은 물리량(시간축 안정성)에 두 임계가 따로 걸려 있고 대조되지 않는
    발생기 A 였고, 그 틈으로 **1600 Hz coh² 0.06 인 세션이 지연 게이트를 통과**할 수
    있었다 — 절대목표 1의 고역이 학습 데이터에서 사라져도 아무도 몰랐다는 뜻이다.

    지금 지터 상한은 코히런스 하한에서 유도된다
    (``σ_max = (fs/2πf_top)·√(−ln coh_min)``). 그러면 다음이 **정리**가 된다:

        지연 게이트를 통과했다  ⇒  대역 상단에서도 코히런스 하한을 만족한다

    이 테스트는 지터를 넓게 훑어 그 함의가 한 번도 깨지지 않는 것을 확인한다.
    반례가 하나라도 나오면 유도가 틀린 것이다.
    """

    from deep_anc.dsp.invariants import MAX_STREAM_DELAY_ROBUST_STD_SAMPLES

    settings = _qa_settings()
    checked = 0
    for sigma in (1.0, 2.0, 2.5, 3.0, 4.0, 6.0):
        entries = _jitter_sessions(tmp_path / f"sweep{sigma}", sigma, 12 * FS)[:2]
        report = validate_recorded_sessions(entries, settings)
        for session in report["sessions"]:
            alignment = session["alignment"]
            rstd = float(alignment["source_err_delay_robust_std_samples"])
            if rstd > MAX_STREAM_DELAY_ROBUST_STD_SAMPLES:
                continue          # 지연 게이트가 거부한 세션 — 함의의 전제가 아니다
            checked += 1
            assert float(alignment["source_err_coherence"]) >= 0.60, (
                f"σ={sigma}: 지연 게이트를 통과(rstd {rstd:.2f})했는데 코히런스가 "
                f"{alignment['source_err_coherence']:.3f} 입니다 — 유도가 깨졌습니다"
            )
            assert float(alignment["source_err_coherence_high"]) >= 0.60, (
                f"σ={sigma}: 지연 게이트를 통과(rstd {rstd:.2f})했는데 **고역** "
                f"코히런스가 {alignment['source_err_coherence_high']:.3f} 입니다"
            )
    assert checked >= 4, f"함의를 확인한 세션이 {checked}개뿐입니다 — 훑기가 좁습니다"


def test_the_old_limit_would_have_admitted_a_dead_high_band():
    """옛 상한 8.0 이 무엇을 통과시켰는지 **숫자로** 남긴다 (음성 대조).

    이것이 없으면 위 테스트는 "유도가 맞다" 를 보일 뿐, **유도가 필요했다** 는 것을
    보이지 못한다.
    """

    from deep_anc.dsp.invariants import (
        CONTROLLED_BAND_TOP_HZ,
        MAX_STREAM_DELAY_ROBUST_STD_SAMPLES,
        MIN_STREAM_COHERENCE,
        coherence_from_delay_jitter,
    )

    dead = coherence_from_delay_jitter(CONTROLLED_BAND_TOP_HZ, 8.0, 48_000.0)
    assert dead < 0.07, dead
    alive = coherence_from_delay_jitter(
        CONTROLLED_BAND_TOP_HZ, MAX_STREAM_DELAY_ROBUST_STD_SAMPLES, 48_000.0
    )
    assert alive == pytest.approx(MIN_STREAM_COHERENCE, abs=1e-9)


# ======================================================================================
# 시간축 재정렬 — 유효창 비율 하한 바로 위
# ======================================================================================
def test_timeline_valid_window_ratio_passes_just_above_the_floor():
    """유효창 비율이 하한 0.90 바로 위(0.9 이상)면 추적을 인정한다.

    negative 짝(무음 소스, 비율 0.6 미만)과 같은 소재를 쓰되, 무음 구간을 전체의 5%
    로만 두어 **하한 근처의 정상**을 만든다.
    """

    from test_recorded_timeline import _make_session

    from deep_anc.data.timeline import TimelineSettings, estimate_lag_track

    source, witness, _holdout, _lag = _make_session(seconds=8.0)
    nearly = source.copy()
    nearly[int(7.6 * FS) :] = 0.0        # 마지막 5% 만 무음
    settings = TimelineSettings(sample_rate=FS)
    track = estimate_lag_track(nearly, witness, settings)
    assert track.valid_window_ratio >= 0.9, track.valid_window_ratio


# ======================================================================================
# 측정·재분석 게이트
# ======================================================================================
def test_timebase_drift_accepts_the_steady_state_repeats_at_90_percent_of_the_tolerance():
    """드리프트 게이트: 허용 2.0 의 90%(1.8 샘플)까지 흔들려도 정상으로 본다.

    negative 짝은 워밍업 반복(편차 10.44)을 기각하는 것이고, 이쪽은 **정상상태 구간이
    기각되지 않는 것**을 본다. 실측 정상 구간 편차는 1.0 미만이다.
    """

    from test_interleaved_probe import MEASURED_CANCEL_TAU, MEASURED_NOISE_TAU

    from deep_anc.dsp.interleaved_probe import timebase_drift

    common = 0.5 * (MEASURED_NOISE_TAU + MEASURED_CANCEL_TAU)
    drift, median = timebase_drift(common[2:10])          # 실측 정상상태 구간
    assert float(np.max(np.abs(drift - median))) < 1.8

    # 그리고 한계의 90% 로 인위적으로 흔든 궤적도 통과 범위 안이다.
    steady = median + 1.8 * np.array([0.0, 1.0, -1.0, 0.5, -0.5, 1.0, -1.0, 0.0])
    shaken, shaken_median = timebase_drift(np.cumsum(steady))
    assert np.all(np.abs(shaken - shaken_median) <= 2.0 + 1.0e-9)


def test_reanalysis_accepts_metadata_that_matches_the_npz_copy(tmp_path):
    """위조 검출 게이트: metadata.json 과 NPZ 사본이 **한 바이트도 다르지 않으면** 통과.

    negative 짝은 위조를 잡는 것이고, 이쪽은 정상 캡처를 못 읽는 일이 없는지 본다.
    """

    from test_reanalyse_paths import _write_capture

    import scripts.data.reanalyse_paths_interleaved as ra

    session = _write_capture(tmp_path / "clean")
    capture = ra.load_capture(session)
    assert int(capture["meta"]["telemetry"]["xrun_count"]) == 0
    assert len(capture["sha256"]) == 64


# ======================================================================================
# G4 평가 게이트
# ======================================================================================
def test_g4_statistical_power_passes_at_exactly_the_group_floor():
    """검정력 게이트: 계열당 그룹이 **정확히 하한(4)** 일 때 판정 가능해야 한다.

    하한이 "이 값이면 된다" 인지 "이 값이면 안 된다" 인지 갈라지면 진입 게이트와
    완료 게이트가 서로 다른 대답을 한다 — 이 저장소가 반복한 사고 모양이다.
    """

    from deep_anc.eval.recorded import MIN_GROUPS_PER_FAMILY, cluster_bootstrap_ci

    rng = np.random.default_rng(20260806)
    groups = np.repeat(np.arange(MIN_GROUPS_PER_FAMILY), 5)
    values = -6.0 + 0.2 * rng.standard_normal(groups.size)
    low, high, n_groups = cluster_bootstrap_ci(values, groups)
    assert n_groups == MIN_GROUPS_PER_FAMILY == 4
    assert np.isfinite(low) and np.isfinite(high)
    assert high < 0.0

    # 하나 모자라면 CI 를 지어내지 않는다 (negative 짝과 같은 경계).
    fewer = np.repeat(np.arange(MIN_GROUPS_PER_FAMILY - 1), 5)
    low, high, n_groups = cluster_bootstrap_ci(values[: fewer.size], fewer)
    assert n_groups == 3
    assert not np.isfinite(low)


def test_metrics_comparison_accepts_two_runs_from_the_same_plant():
    """플랜트 동일성 게이트: 지문이 **완전히 같으면** 비교를 허용한다.

    확정 플랜트 값(P 1602 / S 1462 / handoff 256 / lead 116)을 그대로 쓴다.
    """

    from deep_anc.dsp.invariants import check_plant_fingerprint_match
    from deep_anc.dsp.timing import PlantFingerprint

    payload = dict(
        primary_delay_samples=1602,
        secondary_delay_samples=1462,
        handoff_samples=256,
        lead_samples=116,
        sample_rate=48_000,
        physics_status="measured_primary_path",
        optimize_band_hz=[150.0, 1600.0],
        secondary_sha256=None,
        primary_sha256=None,
        capture_id="225546_f7b0fecd",
        configured_lead_samples=116,
    )
    left = PlantFingerprint(**payload)
    right = PlantFingerprint(**json.loads(json.dumps(payload)))
    assert check_plant_fingerprint_match(left, right).ok
    assert left.digest() == right.digest()

    # 한 샘플만 달라도 비교가 거부된다.
    drifted = PlantFingerprint(**{**payload, "secondary_delay_samples": 1463})
    assert not check_plant_fingerprint_match(left, drifted).ok


# ======================================================================================
# 수집(record_duct) CLI — 강화 방향은 통과한다
# ======================================================================================
def test_recording_gate_cli_accepts_the_default_and_tightened_values(monkeypatch, capsys):
    """게이트 인자를 **기본값(0.90) 그대로** 또는 더 세게 줘도 거부되지 않는다.

    실기에서 소리를 내지 않기 위해, 게이트 검사 **직후**에 있는 source_family 검증에
    걸리도록 잘못된 계열명을 함께 준다. 하드웨어에 닿기 전에 멈추면서도 게이트 인자가
    거부되지 않았다는 것을 보인다.
    """

    from test_record_duct_gates import RECORD_DUCT

    for coherence, ratio in ((0.90, 0.90), (0.99, 0.99)):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "record_duct.py",
                "--min-timeline-coherence", str(coherence),
                "--min-valid-window-ratio", str(ratio),
                "--source-family", "bad/family",
            ],
        )
        with pytest.raises(SystemExit) as excinfo:
            RECORD_DUCT.main()
        message = capsys.readouterr().err
        assert excinfo.value.code == 2
        assert "게이트는 강화만 합니다" not in message, message
        assert "source_family" in message


# ======================================================================================
# 메타 — 이 파일이 조용히 비어 가지 않게
# ======================================================================================
def test_this_file_is_referenced_by_the_registry():
    """레지스트리가 실제로 이 파일의 fixture 를 positive 짝으로 쓰고 있는가."""

    from deep_anc.ops.gate_registry import GATES

    used = {
        item.positive_fixture.split("::", 1)[0]
        for item in GATES
        if item.positive_fixture
    }
    assert "tests/test_gate_positive_fixtures.py" in used
    covered = [
        item.gate_id
        for item in GATES
        if item.positive_fixture.startswith("tests/test_gate_positive_fixtures.py")
    ]
    assert len(covered) >= 15, covered
