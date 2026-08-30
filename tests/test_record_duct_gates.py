"""``record_duct.py`` 의 두 게이트에 대한 **실패 증명**.

게이트를 만들면서 "정상 데이터에서 통과하는 것"만 보고 끝낸 것이 결함 군집 B 의
발생기다. 여기 두 테스트는 각각 게이트가 **거부하는 것**을 본다.
"""

from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from deep_anc.config import REPO_ROOT
from deep_anc.data.recorded_qa import (
    CAPTURE_MIN_LOW_BAND_COHERENCE,
    CAPTURE_MIN_RAW_VALID_WINDOW_RATIO,
    MIN_REF_ERR_COHERENCE,
)
from deep_anc.data.timeline import TimelineReport
from deep_anc.dsp.invariants import (
    MAX_STREAM_DELAY_P95_P5_SAMPLES,
    MAX_STREAM_DELAY_ROBUST_STD_SAMPLES,
    MIN_STREAM_COHERENCE,
    MIN_STREAM_DELAY_VALID_WINDOW_RATIO,
)


def _load_record_duct():
    """스크립트를 모듈로 적재한다 (sounddevice 는 main() 안에서만 import 된다)."""

    path = REPO_ROOT / "scripts" / "data" / "record_duct.py"
    spec = importlib.util.spec_from_file_location("record_duct_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RECORD_DUCT = _load_record_duct()


def _report(**overrides) -> TimelineReport:
    payload = {
        "track_band_hz": (150.0, 700.0),
        "track_window_s": 0.25,
        "track_hop_s": 0.0625,
        "valid_window_ratio": 0.95,
        "raw_lag_median_samples": 1539.5,
        "raw_lag_ptp_samples": 271.0,
        "raw_lag_std_samples": 80.0,
        "aligned_lag_median_samples": 142.5,
        "aligned_lag_std_samples": 2.2,
        "aligned_lag_robust_std_samples": 1.1,
        "aligned_lag_p95_p5_samples": 7.1,
        "aligned_valid_window_ratio": 0.96,
        "coh2_150_600_before": 0.067,
        "coh2_150_600_after": 0.958,
        "coh2_600_1600_before": 0.011,
        "coh2_600_1600_after": 0.817,
        "coh2_ref_err_150_600": 0.993,
    }
    payload.update(overrides)
    return TimelineReport(**payload)


# ------------------------------------------------------- 입력 레일 게이트
def test_input_rail_gate_rejects_the_measured_railing_microphones():
    """negative fixture — 2026-08-05 실측 상태를 재현한 입력은 거부돼야 한다.

    실측: 입력단 무전원/미연결에서 6~23 Hz 초저역이 int32 풀스케일을 때렸다
    (레일 비율 ch0 0.0475~0.0899 / ch1 0.0768~0.1033). 예전 자가진단은 **하한만**
    봤기 때문에 이 상태를 "아주 살아 있는 마이크" 로 통과시켰다.
    """

    fs = 48000
    t = np.arange(2 * fs) / fs
    railing = np.clip(6.0 * np.sin(2.0 * np.pi * 6.0 * t), -1.0, 1.0)
    probe = np.stack([railing, railing * 1.1], axis=1)
    ok, ratios = RECORD_DUCT.input_rail_gate(probe)
    assert not ok
    assert min(ratios) > 0.04


def test_input_rail_gate_passes_a_healthy_quiet_room():
    """positive fixture — 정상 세션 수준(peak 0.006)은 통과해야 한다 (오기각 방지)."""

    rng = np.random.default_rng(0)
    probe = rng.normal(0.0, 0.001, size=(96000, 2))
    ok, ratios = RECORD_DUCT.input_rail_gate(probe)
    assert ok
    assert max(ratios) == 0.0


def test_input_rail_gate_also_rejects_a_single_bad_channel():
    """한 채널만 죽어도 세션은 못 쓴다 — max 로 판정하는지 확인."""

    rng = np.random.default_rng(1)
    good = rng.normal(0.0, 0.001, size=96000)
    bad = np.sign(rng.normal(size=96000))
    ok, _ = RECORD_DUCT.input_rail_gate(np.stack([good, bad], axis=1))
    assert not ok


# ------------------------------------------------------- 시간축 실패-폐쇄 게이트
def test_timeline_gate_rejects_the_shipped_corpus_numbers():
    """negative fixture — 출하된 80 세션의 실제 수치는 전부 거부돼야 한다.

    실측 QA(2026-08-06, 80/80): ``coh²(source→ERR,150-600Hz)`` 최소 0.025 / 중앙
    0.072 / 최대 0.198. 음향 대조군 ``coh²(REF→ERR)`` 는 0.848~0.994 로 멀쩡했다.
    """

    for measured in (0.025, 0.072, 0.198):
        assert not RECORD_DUCT.timeline_gate(
            _report(coh2_150_600_after=measured, coh2_ref_err_150_600=0.989),
            min_coherence=0.90,
            min_valid_window_ratio=0.90,
        )


def test_timeline_gate_rejects_low_valid_window_ratio_even_with_high_coherence():
    """coh² 만 보면 '추정이 대부분 실패했는데 남은 창은 잘 맞았다' 를 통과시킨다."""

    assert not RECORD_DUCT.timeline_gate(
        _report(coh2_150_600_after=0.95, valid_window_ratio=0.55),
        min_coherence=0.90,
        min_valid_window_ratio=0.90,
    )


def test_timeline_gate_accepts_a_recovered_session():
    """positive fixture — 실측 재정렬 결과(0.952 / 유효창 0.981)는 통과해야 한다."""

    assert RECORD_DUCT.timeline_gate(
        _report(coh2_150_600_after=0.952, valid_window_ratio=0.981),
        min_coherence=0.90,
        min_valid_window_ratio=0.90,
    )


@pytest.mark.parametrize(
    ("overrides", "failed_label"),
    [
        (
            {"coh2_150_600_after": CAPTURE_MIN_LOW_BAND_COHERENCE - 0.001},
            "source_aligned→ERR,150-600Hz",
        ),
        (
            {"coh2_600_1600_after": MIN_STREAM_COHERENCE - 0.001},
            "source_aligned→ERR,600-1600Hz",
        ),
        (
            {"coh2_ref_err_150_600": MIN_REF_ERR_COHERENCE - 0.001},
            "REF→ERR,150-600Hz",
        ),
        (
            {"valid_window_ratio": CAPTURE_MIN_RAW_VALID_WINDOW_RATIO - 0.001},
            "raw valid-window ratio",
        ),
        (
            {
                "aligned_valid_window_ratio": (
                    MIN_STREAM_DELAY_VALID_WINDOW_RATIO - 0.001
                )
            },
            "source_aligned→ERR valid-window ratio",
        ),
        (
            {
                "aligned_lag_robust_std_samples": (
                    MAX_STREAM_DELAY_ROBUST_STD_SAMPLES + 0.001
                )
            },
            "residual robust-std",
        ),
        (
            {
                "aligned_lag_p95_p5_samples": (
                    MAX_STREAM_DELAY_P95_P5_SAMPLES + 0.001
                )
            },
            "residual p95-p5",
        ),
    ],
)
def test_capture_gate_rejects_each_downstream_condition_before_publish(
    overrides, failed_label
):
    """수집기는 downstream QA에서 떨어질 세션을 active tree에 먼저 발행하지 않는다."""

    report = _report(**overrides)
    metadata, result = RECORD_DUCT._timeline_metadata_with_capture_gate(
        report,
        min_coherence=CAPTURE_MIN_LOW_BAND_COHERENCE,
        min_valid_window_ratio=CAPTURE_MIN_RAW_VALID_WINDOW_RATIO,
    )

    assert not result.ok
    assert len(result.failed_conditions) == 1
    assert failed_label in result.failure_text
    assert metadata["usable_for_digital_reference"] is False
    assert metadata["capture_gate"]["failed_conditions"] == list(
        result.failed_conditions
    )


def test_capture_gate_error_lists_only_conditions_that_failed():
    """통과한 저역/원시 추적률을 ``<`` 오류로 출력하던 과거 오진을 막는다."""

    report = _report(
        coh2_600_1600_after=MIN_STREAM_COHERENCE - 0.01,
        aligned_lag_p95_p5_samples=MAX_STREAM_DELAY_P95_P5_SAMPLES + 1.0,
    )
    result = RECORD_DUCT.timeline_gate_result(
        report,
        min_coherence=CAPTURE_MIN_LOW_BAND_COHERENCE,
        min_valid_window_ratio=CAPTURE_MIN_RAW_VALID_WINDOW_RATIO,
    )

    assert len(result.failed_conditions) == 2
    assert "600-1600Hz" in result.failure_text
    assert "residual p95-p5" in result.failure_text
    assert "150-600Hz" not in result.failure_text
    assert "raw valid-window" not in result.failure_text
    assert "REF→ERR" not in result.failure_text


def test_capture_gate_exact_boundaries_pass_and_mark_timeline_usable():
    """각 비교는 계약대로 하한 포함(≥), 상한 포함(≤)이다."""

    report = _report(
        coh2_150_600_after=CAPTURE_MIN_LOW_BAND_COHERENCE,
        coh2_600_1600_after=MIN_STREAM_COHERENCE,
        coh2_ref_err_150_600=MIN_REF_ERR_COHERENCE,
        valid_window_ratio=CAPTURE_MIN_RAW_VALID_WINDOW_RATIO,
        aligned_valid_window_ratio=MIN_STREAM_DELAY_VALID_WINDOW_RATIO,
        aligned_lag_robust_std_samples=MAX_STREAM_DELAY_ROBUST_STD_SAMPLES,
        aligned_lag_p95_p5_samples=MAX_STREAM_DELAY_P95_P5_SAMPLES,
    )
    metadata, result = RECORD_DUCT._timeline_metadata_with_capture_gate(
        report,
        min_coherence=CAPTURE_MIN_LOW_BAND_COHERENCE,
        min_valid_window_ratio=CAPTURE_MIN_RAW_VALID_WINDOW_RATIO,
    )

    assert result.ok
    assert result.failed_conditions == ()
    assert result.failure_text == ""
    assert metadata["usable_for_digital_reference"] is True


@pytest.mark.parametrize(
    "field",
    ["aligned_lag_robust_std_samples", "aligned_lag_p95_p5_samples"],
)
def test_capture_gate_nonfinite_residual_is_measurement_failure(field):
    """NaN 비교가 False라서 조용히 통과하는 fail-open 경로를 닫는다."""

    metadata, result = RECORD_DUCT._timeline_metadata_with_capture_gate(
        _report(**{field: float("nan")}),
        min_coherence=CAPTURE_MIN_LOW_BAND_COHERENCE,
        min_valid_window_ratio=CAPTURE_MIN_RAW_VALID_WINDOW_RATIO,
    )

    assert not result.ok
    assert result.failed_conditions == (
        (
            "source_aligned→ERR residual robust-std(samples) 측정 불가(non-finite)"
            if field == "aligned_lag_robust_std_samples"
            else "source_aligned→ERR residual p95-p5(samples) 측정 불가(non-finite)"
        ),
    )
    assert metadata["usable_for_digital_reference"] is False


_INVALID_GATE_ARGUMENTS = tuple(
    (option, value)
    for option, floor in (
        ("--min-timeline-coherence", CAPTURE_MIN_LOW_BAND_COHERENCE),
        ("--min-valid-window-ratio", CAPTURE_MIN_RAW_VALID_WINDOW_RATIO),
    )
    for value in ("nan", "inf", str(floor - 0.01), "1.000001")
)


# ------------------------------------------------------- 게이트 완화 금지
@pytest.mark.parametrize(
    ("option", "value"),
    _INVALID_GATE_ARGUMENTS,
)
def test_cli_rejects_nonfinite_loose_or_above_one_gates_in_dry_run(
    option, value, capsys
):
    """dry-run도 유효하지 않은 계약을 PASS로 표시하면 안 된다."""

    with pytest.raises(SystemExit) as excinfo:
        RECORD_DUCT.main([option, value, "--dry-run"])
    assert excinfo.value.code == 2
    output = capsys.readouterr()
    assert "게이트는 강화만 합니다" in output.err
    assert "[DRY-RUN PASS]" not in output.out


@pytest.mark.parametrize(("option", "value"), _INVALID_GATE_ARGUMENTS)
def test_cli_rejects_invalid_gates_before_live_audio_import(
    option, value, monkeypatch, capsys
):
    """live 호출에서도 sounddevice import 전에 같은 계약을 거부한다."""

    original_import = builtins.__import__
    audio_imports: list[str] = []

    def guarded_import(name, *args, **kwargs):
        if name == "sounddevice":
            audio_imports.append(name)
            raise AssertionError("유효하지 않은 게이트가 오디오 import에 도달함")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(SystemExit) as excinfo:
        RECORD_DUCT.main(
            [
                option,
                value,
                "--confirm-user-present",
                "--confirm-volume-minimum",
                "--confirm-routing-and-geometry",
            ]
        )
    assert excinfo.value.code == 2
    assert audio_imports == []
    assert "게이트는 강화만 합니다" in capsys.readouterr().err


def test_io_timestamp_summary_is_marked_as_provenance_only():
    """콜백 타임스탬프는 진단용이지 수정 수단이 아니다 — 그 사실이 산출물에 남아야 한다.

    실측: ``dac − adc`` 가 0.010/0.020 s 두 값 사이를 16 샘플 단위로만 튀어 실제
    ±130 샘플 변조를 전혀 보여주지 않는다. 이 요약을 보고 정렬을 고치려 들면 안 된다.
    """

    fs = 48000
    frames = 256
    count = 400
    adc = np.arange(count) * frames / fs
    dac = adc + np.where(np.arange(count) % 2 == 0, 0.010, 0.020)
    stamps = np.stack([adc, dac, np.full(count, float(frames))], axis=1)
    summary = RECORD_DUCT._summarise_io_timestamps(stamps, fs)
    assert summary["callbacks"] == count
    assert summary["frames_total"] == count * frames
    assert summary["unique_frames"] == [frames]
    assert summary["dac_minus_adc_s"]["unique_values"] == 2
    assert "정렬 복원에 쓰지 말 것" in summary["note"]
