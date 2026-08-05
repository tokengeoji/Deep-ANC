"""교차 도메인 불변식 검사기 — 정상 입력과 **오염 입력** 양쪽을 넣는다 (발생기 A'/B).

각 검사에 대해 반드시 두 개가 있다:
  · 오염 입력에서 FAIL 하는가 — 이게 없으면 그 검사가 작동한다는 증거가 없다.
  · 정상 입력에서 PASS 하는가 — 이게 없으면 무조건 거부하는 검사와 구별되지 않는다.

오염 입력은 만들어낸 것이 아니라 **2026-08-04 에 실제로 일어난 값**이다.
"""

from __future__ import annotations

import numpy as np
import pytest

from deep_anc.dsp.invariants import (
    InvariantViolation,
    check_corpus_disjoint,
    check_lead_agreement,
    check_measured_delay_agreement,
    check_plant_fingerprint_match,
    check_relative_tau_constancy,
    check_stream_coherence,
    check_stream_delay_stability,
    derive_playback_to_error_delay_samples,
)
from deep_anc.dsp.timing import BandPlan, PlantDelays, PlantFingerprint


FS = 48_000

# 2026-08-04 출하 아티팩트의 실측 P−S 상대 τ. 반복 11 에서 1.4 → 32 샘플 점프.
SHIPPED_RELATIVE_TAU = [
    0.0, 1.20, 1.13, 1.09, 1.09, 1.29, 1.41, 1.47, 1.14, 1.48, 1.36,
    32.11, 32.18, 31.75, 30.26, 29.06,
]


# ----------------------------------------------------------------------------------
# P−S 상대 τ 상수성
# ----------------------------------------------------------------------------------
def test_relative_tau_constancy_fails_on_the_measured_frame_slip():
    rel = np.asarray(SHIPPED_RELATIVE_TAU)
    result = check_relative_tau_constancy(rel, np.zeros_like(rel), tolerance_samples=3.0)

    assert not result.ok
    assert result.measured["outlier_indices"] == [11, 12, 13, 14, 15]
    # 게이트가 실제로 본 값은 range 32.18 하나였고, 아티팩트가 스스로 써넣은 허용치
    # 48 과 비교해 통과시켰다. 계단 자체를 보면 30.75 샘플이다.
    assert result.measured["max_consecutive_step_samples"] == pytest.approx(30.75, abs=0.01)
    with pytest.raises(InvariantViolation, match="프레임 슬립"):
        result.raise_if_failed()


def test_relative_tau_constancy_passes_on_the_clean_subset():
    """같은 캡처의 오염 전 11 반복은 통과해야 한다 — 무조건 거부가 아님을 증명."""

    rel = np.asarray(SHIPPED_RELATIVE_TAU[:11])
    result = check_relative_tau_constancy(rel, np.zeros_like(rel), tolerance_samples=3.0)

    assert result.ok
    assert result.measured["max_deviation_samples"] < 3.0
    result.raise_if_failed()


def test_relative_tau_constancy_survives_a_contamination_majority():
    """오염이 과반이어도 잡아야 한다 — MAD 스케일 임계였다면 여기서 무력화된다."""

    rel = np.asarray([0.0, 1.2, 1.1, 1.3] + [32.0] * 12)
    result = check_relative_tau_constancy(rel, np.zeros_like(rel), tolerance_samples=3.0)

    assert not result.ok
    assert result.measured["outlier_indices"]


# ----------------------------------------------------------------------------------
# 재생→캡처 결맞음
# ----------------------------------------------------------------------------------
def _band_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    from scipy import signal

    sos = signal.butter(6, [120.0, 900.0], btype="bandpass", fs=FS, output="sos")
    return signal.sosfilt(sos, rng.standard_normal(n)).astype(np.float64)


def test_stream_coherence_passes_on_a_delayed_copy():
    """정상 세션 = 캡처가 재생의 지연·감쇠 사본 + 잡음. 실측 coh² 0.96~0.99."""

    rng = np.random.default_rng(20260805)
    n = FS * 4
    source = _band_noise(rng, n)
    capture = np.roll(source, 120) * 0.7 + 0.05 * rng.standard_normal(n)

    result = check_stream_coherence(
        source, capture, sample_rate=FS, band_hz=(150.0, 600.0), min_coherence=0.60
    )
    assert result.ok
    assert result.measured["coherence"] > 0.9
    result.raise_if_failed()


def test_stream_coherence_fails_on_a_collapsed_timebase():
    """실측 결함 2 재현: 음향(REF→ERR)은 멀쩡한데 source→ERR 만 붕괴한 세션.

    붕괴본 실측 coh²(source→ERR) 0.021~0.126 / coh²(REF→ERR) 0.959~0.991.
    파일별 RMS·clip·길이는 전부 정상이라 기존 QA 는 80/80 을 PASS 시켰다.
    """

    rng = np.random.default_rng(4)
    n = FS * 4
    source = _band_noise(rng, n)
    unrelated = _band_noise(np.random.default_rng(5), n)
    capture = np.roll(unrelated, 120) * 0.7 + 0.05 * rng.standard_normal(n)
    reference = np.roll(unrelated, 60) * 0.8 + 0.05 * rng.standard_normal(n)

    result = check_stream_coherence(
        source,
        capture,
        sample_rate=FS,
        band_hz=(150.0, 600.0),
        min_coherence=0.60,
        control=reference,
    )

    assert not result.ok
    assert result.measured["coherence"] < 0.3
    # 대조군이 살아 있으면 진단까지 나와야 한다.
    assert result.measured["control_coherence"] > 0.6
    assert "타임베이스" in result.detail
    with pytest.raises(InvariantViolation):
        result.raise_if_failed()


# ----------------------------------------------------------------------------------
# 플랜트 지문
# ----------------------------------------------------------------------------------
def _fingerprint(*, primary: int, secondary: int, physics_status: str) -> PlantFingerprint:
    delays = PlantDelays(
        primary_delay_samples=primary,
        secondary_delay_samples=secondary,
        handoff_samples=256,
        sample_rate=FS,
    )
    bands = BandPlan.resolve(
        plant_trusted_band_hz=(150.0, 1600.0),
        duct_cfg={"acoustics": {"realistic_target_band_hz": [80.0, 800.0]}},
        sample_rate=FS,
    )
    return PlantFingerprint.build(
        delays=delays,
        lead=delays.lead(),
        physics_status=physics_status,
        bands=bands,
    )


def test_plant_fingerprint_match_fails_across_the_20260804_plants():
    before = _fingerprint(
        primary=1489,
        secondary=1342,
        physics_status="secondary_surrogate_representation_pretrain",
    )
    after = _fingerprint(
        primary=1608, secondary=1465, physics_status="measured_primary_path"
    )

    result = check_plant_fingerprint_match(before, after)
    assert not result.ok
    assert any("secondary_delay_samples" in item for item in result.measured["differences"])
    with pytest.raises(InvariantViolation, match="서로 다른 플랜트"):
        result.raise_if_failed()


def test_plant_fingerprint_match_passes_for_the_same_plant():
    one = _fingerprint(primary=1602, secondary=1462, physics_status="measured_primary_path")
    two = _fingerprint(primary=1602, secondary=1462, physics_status="measured_primary_path")

    result = check_plant_fingerprint_match(one, two)
    assert result.ok
    result.raise_if_failed()


# ----------------------------------------------------------------------------------
# lead 유도값 일치
# ----------------------------------------------------------------------------------
def _measured_delays() -> PlantDelays:
    return PlantDelays(
        primary_delay_samples=1602,
        secondary_delay_samples=1462,
        handoff_samples=256,
        sample_rate=FS,
    )


def test_lead_agreement_passes_on_the_measured_plant():
    result = check_lead_agreement(116, _measured_delays())
    assert result.ok
    assert result.measured["derived_lead_samples"] == 116
    assert result.measured["relative_delay_samples"] == 140
    result.raise_if_failed()


def test_lead_agreement_fails_on_the_aaeef41_mismatch():
    """설정이 109 인데 실측 지연이 요구하는 값은 116 이었다 — 양쪽 다 "통과" 였다."""

    result = check_lead_agreement(109, _measured_delays())

    assert not result.ok
    assert result.measured["mismatch_samples"] == 7
    with pytest.raises(InvariantViolation, match="유도되는 값과 다릅니다"):
        result.raise_if_failed()


def test_lead_agreement_tolerance_is_explicit_and_only_widens_when_asked():
    delays = _measured_delays()
    assert not check_lead_agreement(109, delays).ok
    assert check_lead_agreement(109, delays, tolerance_samples=16).ok


# ======================================================================================
# 재생→캡처 지연 안정성 (결함 2 의 두 번째 축)
# ======================================================================================
def _seeded_band_noise(frames: int, seed: int) -> np.ndarray:
    from scipy.signal import butter, lfilter

    rng = np.random.default_rng(seed)
    b, a = butter(4, [100.0 / (FS / 2), 2_000.0 / (FS / 2)], btype="band")
    return lfilter(b, a, rng.standard_normal(frames))


def test_stream_delay_stability_passes_on_a_constant_delay():
    """지연이 상수면 통과한다 — 무조건 거부하는 검사가 아님을 증명."""

    frames = FS * 5
    source = _seeded_band_noise(frames, 1)
    capture = np.roll(source, 120)
    result = check_stream_delay_stability(source, capture, sample_rate=FS)
    assert result.ok
    assert abs(result.measured["delay_median_samples"] - 120.0) <= 2.0
    assert result.measured["delay_range_samples"] <= 256.0


def test_stream_delay_stability_fails_on_a_drifting_timebase():
    """지연이 창마다 크게 움직이면 FAIL 한다.

    실측(2026-08-04, 1초창): 붕괴 세션 τ std 1019~2216 / range 8869~13532 인데
    음향 대조군은 std 17.7~20.1 / range 106~215 였다. 두 무리가 50배 벌어져 있다.
    """

    frames = FS * 6
    source = _seeded_band_noise(frames, 2)
    capture = np.zeros(frames)
    for index, start in enumerate(range(0, frames, FS)):
        stop = min(frames, start + FS)
        capture[start:stop] = np.roll(source, 120 + index * 1_200)[start:stop]
    result = check_stream_delay_stability(source, capture, sample_rate=FS)
    assert not result.ok
    assert "떠다닙니다" in result.detail
    assert result.measured["delay_range_samples"] > 256.0
    with pytest.raises(InvariantViolation):
        result.raise_if_failed()


# ======================================================================================
# D2 — 같은 지연을 두 방법으로 잰 값의 대조
# ======================================================================================
def test_measured_delay_agreement_passes_within_tolerance():
    result = check_measured_delay_agreement(1_672.0, 1_690.0, observation_count=80)
    assert result.ok
    assert result.measured["mismatch_samples"] == 18.0


def test_measured_delay_agreement_fails_on_the_d2_gap():
    """2026-08-05 감사가 찾은 실제 불일치를 그대로 넣는다.

    관측(독립 3방법 일치): 포락선 상관 80세션 중앙값 1672 / 반송파 상관 1663±73 /
    제어기 시작지연 스윕 ~1670.
    유도: P 벌크지연 1602 + argmax tap 247 = 1849.
    """

    result = check_measured_delay_agreement(1_672.0, 1_849.0, observation_count=80)
    assert not result.ok
    assert result.measured["mismatch_samples"] == 177.0
    assert "두 방법으로 잰 값이 다릅니다" in result.detail


def test_derived_delay_uses_the_same_estimator_as_the_observation():
    """유도 쪽은 무게중심이 아니라 **최대점**을 써야 한다.

    상호상관 최대점은 임펄스 응답의 최대점에 대응한다. 무게중심을 쓰면 잔향 꼬리가
    값을 뒤로 끌어당겨 가짜 불일치가 만들어진다 — 실측 P(z) 에서 argmax 247 vs
    무게중심 363.7 로 117 샘플 차이다.
    """

    fir = np.zeros(512)
    fir[247] = 1.0
    fir[400:] = 0.2  # 잔향 꼬리: 무게중심을 뒤로 끌어당긴다
    assert derive_playback_to_error_delay_samples(1_602, fir) == 1_849.0

    with pytest.raises(ValueError):
        derive_playback_to_error_delay_samples(-1, fir)
    with pytest.raises(ValueError):
        derive_playback_to_error_delay_samples(0, np.asarray([np.nan, 1.0]))


# ======================================================================================
# D1 — 코퍼스 누수
# ======================================================================================
def test_corpus_disjoint_passes_on_separate_corpora():
    result = check_corpus_disjoint(
        {"music": ["a.mp3", "b.mp3"], "speech": ["s1.flac"]},
        {"music": ["data/raw/music/000/c.mp3"], "speech": ["data/raw/speech/x.flac"]},
    )
    assert result.ok
    assert result.measured["total_overlap_clips"] == 0


def test_corpus_disjoint_fails_on_the_music_leak():
    """2026-08-05 감사가 찾은 상태 — music 만 100% 겹친다.

    실측: music 60/60 겹침, 그중 55개가 합성 train. 나머지 계열은 0.
    같은 곡에서 두 브랜치가 반대 방향 gradient 를 주고, **music 만 개선되지 않았다**.
    """

    recorded = {
        "music": [f"{index:06d}.mp3" for index in range(60)],
        "speech": ["spk-1.flac", "spk-2.flac"],
    }
    synthetic = {
        # 합성 풀은 저장소 경로를 들고 실측 csv 는 basename 만 든다 —
        # 정규화가 없으면 교집합이 영원히 비어 보인다.
        "music": [f"data/raw/music/fma_small/000/{index:06d}.mp3" for index in range(60)],
        "speech": ["data/raw/speech/LibriSpeech/other.flac"],
    }
    splits = {path: ("train" if index < 55 else "val")
              for index, path in enumerate(synthetic["music"])}

    result = check_corpus_disjoint(recorded, synthetic, synthetic_splits=splits)

    assert not result.ok
    assert "music" in result.detail
    families = result.measured["families"]
    assert families["music"]["overlap_clips"] == 60
    assert families["music"]["overlap_ratio"] == 1.0
    assert families["music"]["overlap_by_synthetic_split"] == {"train": 55, "val": 5}
    assert families["speech"]["overlap_clips"] == 0


def test_corpus_disjoint_normalises_paths_before_comparing():
    """경로 형태가 달라도 같은 원본이면 잡아야 한다.

    이 정규화가 없으면 게이트가 **있는데도 아무것도 잡지 못한다** — 누수 게이트의
    가장 흔한 실패 방식이다.
    """

    result = check_corpus_disjoint(
        {"music": ["Track01.MP3"]},
        {"music": ["data\\raw\\music\\000\\track01.mp3"]},
    )
    assert not result.ok
    assert result.measured["families"]["music"]["overlap_clips"] == 1
