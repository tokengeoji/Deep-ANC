"""실기 세션 평가의 source-energy fail-closed 규약.

이 테스트는 오디오 장치를 열지 않는다. 실제 출력 source와 OFF/ON ERR 파형을
메모리에서 만들어, 고역 floor를 감쇠/증폭이라고 해석하지 않는지만 검증한다.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.demo import evaluate_session


FS = 48_000


def _tone(hz: float, *, seconds: float = 2.0) -> np.ndarray:
    time = np.arange(int(FS * seconds), dtype=np.float64) / FS
    return np.sin(2.0 * np.pi * hz * time)


def _rows_for(source: np.ndarray, *, threshold: float = 0.25) -> dict[float, dict]:
    # ERR OFF/ON은 충분히 큰 같은 톤/잡음이어도, source가 해당 octave에 없으면
    # 공개 attenuation은 NaN이어야 한다.
    rows = evaluate_session.qualify_octave_attenuation_by_off_source(
        source,
        0.5 * source,
        source,
        FS,
        [125.0, 250.0, 1_000.0, 2_000.0, 8_000.0],
        (150.0, 1_600.0),
        min_source_energy_density_ratio=threshold,
    )
    return {float(row["center_hz"]): row for row in rows}


def test_octave_gate_qualifies_only_the_played_tone_band():
    rows = _rows_for(_tone(300.0))

    assert rows[250.0]["source_energy_valid"]
    assert rows[250.0]["attenuation_db"] == pytest.approx(6.0206, abs=0.1)
    assert rows[250.0]["source_energy_fraction"] > 0.99

    # 1 kHz/2 kHz/8 kHz는 ERROR 신호의 filter floor가 존재해도 source를 재생하지
    # 않았다. raw 진단값은 남지만, 성능으로 읽는 필드는 fail-closed NaN이다.
    for center in (1_000.0, 2_000.0, 8_000.0):
        assert not rows[center]["source_energy_valid"]
        assert rows[center]["source_energy_density_ratio"] < 0.25
        assert np.isnan(rows[center]["attenuation_db"])
        assert np.isfinite(rows[center]["unqualified_attenuation_db"])


def test_octave_gate_rejects_powered_err_when_source_track_is_silent():
    rng = np.random.default_rng(20260828)
    err_off = rng.normal(size=FS * 2)
    rows = evaluate_session.qualify_octave_attenuation_by_off_source(
        err_off,
        0.5 * err_off,
        np.zeros_like(err_off),
        FS,
        [250.0, 1_000.0, 2_000.0],
        (150.0, 1_600.0),
        min_source_energy_density_ratio=0.25,
    )

    assert all(not row["source_energy_valid"] for row in rows)
    assert all(row["source_energy_fraction"] == 0.0 for row in rows)
    assert all(np.isnan(row["attenuation_db"]) for row in rows)
    assert all(np.isfinite(row["unqualified_attenuation_db"]) for row in rows)


def test_octave_gate_keeps_125hz_in_a_broadband_source():
    """PSD 25% default는 48 kHz broadband source의 좁은 125 Hz octave를 버리지 않는다."""

    rng = np.random.default_rng(20260828)
    source = rng.normal(size=FS * 4)
    rows = _rows_for(source)

    assert rows[125.0]["source_energy_density_ratio"] > 0.25
    assert rows[125.0]["source_energy_valid"]
    assert np.isfinite(rows[125.0]["attenuation_db"])


@pytest.mark.parametrize("invalid", [0.0, -0.01, 1.01, float("nan")])
def test_octave_source_energy_density_threshold_fails_before_live_evaluation(invalid):
    with pytest.raises(ValueError, match="0보다 크고 1 이하"):
        evaluate_session.validate_octave_source_energy_density_ratio(invalid)


def test_live_protocol_keeps_noise_source_on_while_only_anc_toggles(monkeypatch):
    """OFF/ON은 source가 아니라 ANC control의 상태만 바꿔야 한다.

    Fake runtime과 Fake sleep을 써서 실제 ALSA/스피커를 전혀 열지 않는다. 각 sleep은
    분석 구간을 뜻하므로 세 구간 모두 source ON이고 ANC만 OFF→ON→OFF여야 한다.
    """

    class FakeANC:
        instance: "FakeANC | None" = None

        def __init__(self, _cfg, *, record_seconds):
            self.fs = 10
            self.record_seconds = record_seconds
            self.state = SimpleNamespace(
                anc_enabled=False,
                noise_enabled=False,
                latest_stats={"underruns": 0, "xruns": 0},
                fatal_error=None,
            )
            self.started = False
            self.stopped = False
            FakeANC.instance = self

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def session_data(self):
            # base=4/on=5/tail=2이면 최대 slice 끝은 105 sample이다.
            samples = 140
            return {
                "err": np.ones(samples, dtype=np.float32),
                "source": np.ones(samples, dtype=np.float32),
                "anc_gain": np.ones(samples, dtype=np.float32),
            }

    observed_states: list[tuple[bool, bool]] = []

    def fake_sleep(_seconds):
        assert FakeANC.instance is not None
        observed_states.append(
            (
                bool(FakeANC.instance.state.anc_enabled),
                bool(FakeANC.instance.state.noise_enabled),
            )
        )

    monkeypatch.setattr(evaluate_session, "RealtimeANC", FakeANC)
    monkeypatch.setattr(evaluate_session.time, "sleep", fake_sleep)

    result = evaluate_session.run_scenario(
        {}, {"baseline_seconds": 4.0, "on_seconds": 5.0, "tail_seconds": 2.0}
    )

    assert observed_states == [(False, True), (True, True), (False, True)]
    assert result["protocol_state"] == {
        "noise_enabled_requested": True,
        "anc_enabled_baseline_requested": False,
        "anc_enabled_on_requested": True,
        "anc_enabled_tail_requested": False,
    }
    assert result["off_source"].size == 20
    assert FakeANC.instance is not None and FakeANC.instance.stopped
