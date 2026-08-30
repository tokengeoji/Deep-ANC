"""저장된 캡처만으로 official P/S 를 다시 푸는 오프라인 경로를 검증한다.

이 파일이 지키는 것은 두 가지다.

1. **오염된 캡처가 통과하지 못한다.** 2026-08-05 결함 1 은 출력 버퍼 프레임 슬립
   반복 5개와 정상상태 미도달 반복 1개가 그대로 평균에 들어가 official 이 된
   사건이었다. 여기서는 그 슬립을 합성 캡처에 **직접 주입**하고, 새 게이트가
   정확히 그 반복만 버리는지 확인한다. 게이트를 끄면 이 테스트가 깨진다.
2. **도구가 게이트를 우회하는 수단이 되지 않는다.** 오프라인 재분석은 본질적으로
   "파라미터를 바꿔 좋아 보이는 결과를 고르는" 유혹을 만든다. 완화 방향 인자 거부와
   캡처 위조 탐지가 없으면 게이트 전체가 무의미해진다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from deep_anc.dsp.interleaved_probe import build_interleaved_probe

from scripts.data import measure_paths_interleaved as mpi
from scripts.data import reanalyse_paths_interleaved as ra


FS = 8_000
PERIOD_SECONDS = 0.25
BAND_HZ = (100.0, 1000.0)
AMPLITUDE = 0.02
WARMUP = 2
REPEATS = 16
P_DELAY = 100
S_DELAY = 60
FIT_BAND = (150.0, 800.0)
BAND_KWARGS = dict(
    fit_band_hz=FIT_BAND,
    consistency_band_hz=(150.0, 800.0),
    required_band_hz=(150.0, 800.0),
)
INT32_MAX = 2**31 - 1


def _probe():
    return build_interleaved_probe(
        sample_rate=FS, period_seconds=PERIOD_SECONDS, band_hz=BAND_HZ,
        amplitude=AMPLITUDE, tone_spacing_hz=None,
    )


# 두 경로에 **같은 커널**을 쓴다. 벌크 지연 추정은 대역제한 정합필터라 커널 모양에
# 따라 몇 샘플 치우치는데, 커널이 같으면 그 치우침이 두 채널에 공통으로 실려
# P−S 가 정확히 보존된다 — 이 측정이 실제로 주장하는 물리량이 바로 그것이다.
_KERNEL = np.exp(-np.arange(24) / 6.0) * np.cos(np.arange(24) * 0.7)


def _path_ir(delay: int) -> np.ndarray:
    ir = np.zeros(delay + _KERNEL.size)
    ir[delay:] = _KERNEL
    return ir


def _slip(signal: np.ndarray, *, start: int, samples: int) -> np.ndarray:
    """``start`` 이후를 ``samples`` 만큼 미끄러뜨린다 — 출력 버퍼 프레임 슬립 모형."""

    out = signal.copy()
    out[start:] = signal[max(0, start - samples) : signal.size - samples]
    return out


def _write_capture(
    session: Path,
    *,
    slip_start_period: int | None = None,
    slip_samples: int = 32,
    metadata_override: dict | None = None,
    tamper_metadata_json: bool = False,
) -> Path:
    """합성 캡처 디렉터리를 만든다. 실제 측정 스크립트와 같은 구조로 쓴다."""

    probe = _probe()
    lead_in = FS // 2
    total = WARMUP + REPEATS
    period = probe.period_samples
    noise_play = np.tile(probe.noise_signal.astype(np.float64), total)
    cancel_play = np.tile(probe.cancel_signal.astype(np.float64), total)
    if slip_start_period is not None:
        cancel_play = _slip(
            cancel_play,
            start=(WARMUP + slip_start_period) * period,
            samples=slip_samples,
        )

    playback = np.zeros((lead_in + total * period, 2))
    playback[lead_in:, 0] = np.tile(probe.noise_signal.astype(np.float64), total)
    playback[lead_in:, 1] = np.tile(probe.cancel_signal.astype(np.float64), total)

    driven = np.zeros(playback.shape[0])
    driven[lead_in:] = noise_play
    err = np.convolve(driven, _path_ir(P_DELAY))[: playback.shape[0]]
    driven = np.zeros(playback.shape[0])
    driven[lead_in:] = cancel_play
    err = err + np.convolve(driven, _path_ir(S_DELAY))[: playback.shape[0]]

    recorded = np.zeros((err.size, 2))
    recorded[:, 0] = err
    recorded[:, 1] = err * 0.5
    recorded_raw = np.rint(np.clip(recorded * 40.0, -1.0, 1.0) * INT32_MAX)

    rng = np.random.default_rng(11)
    preflight = rng.normal(scale=1e-5, size=(FS, 2))
    preflight_raw = np.rint(preflight * INT32_MAX)

    resolution = FS / period
    crest_noise, crest_cancel = probe.crest_db()
    metadata = {
        "capture_id": "cap-synthetic",
        "method": mpi.METHOD,
        "sample_rate": FS,
        "block_size": 256,
        "latency": "low",
        "amplitude": AMPLITUDE,
        "design_band_hz": [BAND_HZ[0], BAND_HZ[1]],
        "required_band_hz": [150.0, 800.0],
        "channel_band_hz": {
            drive: [
                float(probe.bins_for(drive)[0] * resolution),
                float(probe.bins_for(drive)[-1] * resolution),
            ]
            for drive in ("noise", "cancel")
        },
        "period_seconds": PERIOD_SECONDS,
        "warmup_periods": WARMUP,
        "repeats": REPEATS,
        "lead_in_samples": lead_in,
        "guard_bins": probe.guard_bins(),
        "crest_db": {"noise": crest_noise, "cancel": crest_cancel},
        "warp": {"applied": False},
        "telemetry": {"xrun_count": 0, "completed": True},
        "invalid_reasons": [],
        "valid": True,
    }
    metadata.update(metadata_override or {})

    session.mkdir(parents=True, exist_ok=True)
    stored = dict(metadata)
    if tamper_metadata_json:
        stored = dict(metadata, amplitude=AMPLITUDE * 2)
    np.savez_compressed(
        session / "raw_measurement.npz",
        output=playback.astype(np.float32),
        input_raw_int32=recorded_raw.astype(np.int32),
        preflight_raw_int32=preflight_raw.astype(np.int32),
        metadata_json=np.asarray(json.dumps(stored, ensure_ascii=False, sort_keys=True)),
    )
    (session / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return session


def _analyse(capture, **overrides):
    kwargs = dict(
        fir_length=256, pre_roll=32, max_delay_samples=800,
        min_alignment_score=mpi.DEFAULT_MIN_ALIGNMENT_SCORE,
        min_kept_repeats=8,
        max_relative_tau_samples=mpi.DEFAULT_MAX_RELATIVE_TAU_SAMPLES,
        max_drift_deviation_samples=mpi.DEFAULT_MAX_DRIFT_DEVIATION_SAMPLES,
        max_delay_jitter_samples=3,
        **BAND_KWARGS,
    )
    kwargs.update(overrides)
    return mpi.analyse_capture(
        err=capture["err"], probe=capture["probe"],
        period_starts=capture["period_starts"],
        snr_spectra=capture["snr_spectra"], **kwargs,
    )


def test_clean_capture_round_trips_to_the_injected_plant(tmp_path):
    capture = ra.load_capture(_write_capture(tmp_path / "clean"))
    results, report = _analyse(capture)

    assert int(report["keep"].sum()) == REPEATS      # 깨끗하면 하나도 안 버린다
    assert report["relative_delay_spread_samples"] == 0
    p_delay = results["noise"]["model"]["delay_samples"]
    s_delay = results["cancel"]["model"]["delay_samples"]
    # 절대 지연은 대역제한 정합필터라 커널 모양만큼 치우친다(설계상 재현 안 되는 양).
    assert abs(p_delay - P_DELAY) <= 5 and abs(s_delay - S_DELAY) <= 5
    # 재현되어야 하는 것은 **P−S** 다. 여기서는 정확히 맞아야 한다.
    assert p_delay - s_delay == P_DELAY - S_DELAY
    for item in results.values():
        assert item["reasons"] == [], item["reasons"]
        assert item["model"]["consistency"] > 0.999


def test_injected_frame_slip_is_rejected_exactly(tmp_path):
    """오염 반복만 정확히 버려야 한다 — 이것이 결함 1 의 회귀 테스트다."""

    slip_at = 10
    session = _write_capture(
        tmp_path / "slip", slip_start_period=slip_at, slip_samples=32
    )
    capture = ra.load_capture(session)
    clean = ra.load_capture(_write_capture(tmp_path / "ref"))
    clean_results, _ = _analyse(clean)
    results, report = _analyse(capture)

    dropped = set(np.flatnonzero(~report["keep"]).tolist())
    # 슬립된 반복 전부와, 그 직전 반복 하나(드리프트 게이트가 전이를 양쪽에서 본다).
    assert dropped == set(range(slip_at - 1, REPEATS)), dropped
    assert set(report["relative_tau_rejected"].tolist()) == set(range(slip_at, REPEATS))
    assert int(report["keep"].sum()) >= 8

    # 살아남은 반복만으로 만든 플랜트는 오염 없는 캡처와 **같은 답**을 낸다.
    for drive in ("noise", "cancel"):
        assert (
            results[drive]["model"]["delay_samples"]
            == clean_results[drive]["model"]["delay_samples"]
        )
        assert results[drive]["model"]["consistency"] > 0.999
        assert results[drive]["reasons"] == []
    assert report["relative_tau_max_abs"] < mpi.MAX_KEPT_RELATIVE_TAU_ABS_SAMPLES


def test_frame_slip_passes_the_old_gate_and_fails_the_new_one(tmp_path):
    """옛 게이트(허용 48 샘플 + 정렬점수 0.5)는 이 슬립을 통과시킨다.

    통과시킨다는 사실 자체가 이 변경의 근거다. 출하본 아티팩트는 정확히 이 상태로
    delay_spread 32 를 허용치 48 과 비교해 통과했다.
    """

    session = _write_capture(tmp_path / "slip2", slip_start_period=10, slip_samples=32)
    capture = ra.load_capture(session)

    # 옛 규약: 상대 τ 게이트 없음(허용치를 슬립보다 크게), 점수 하한 0.5.
    _, old_report = _analyse(
        capture, min_alignment_score=0.5,
        max_relative_tau_samples=1_000.0,
        max_drift_deviation_samples=1_000.0,
        max_delay_jitter_samples=48, min_kept_repeats=3,
    )
    assert int(old_report["keep"].sum()) == REPEATS, "옛 게이트는 오염을 통과시켰다"
    assert old_report["relative_delay_spread_samples"] >= 30

    # 새 규약: 오염 반복이 버려지고 상대 τ spread 가 허용치 안으로 들어온다.
    _, new_report = _analyse(capture)
    assert int(new_report["keep"].sum()) < REPEATS
    assert new_report["relative_delay_spread_samples"] <= 3


def test_majority_frame_slip_fails_closed(tmp_path):
    """슬립이 과반이면 **어느 무리가 옳은지 가릴 수 없다** — 조용히 고르면 안 된다.

    중앙값 편차 게이트는 본질적으로 다수 무리를 남긴다. 슬립이 과반이면 남는 쪽이
    슬립 이후 무리이고, 그대로 쓰면 P−S 가 슬립만큼 통째로 틀린다(= lead 가 틀린다).
    스트림의 첫 분석 주기와 프레임 정렬이 다르다는 사실로 이것을 잡아 실패 폐쇄한다.
    """

    session = _write_capture(tmp_path / "slip3", slip_start_period=3, slip_samples=32)
    capture = ra.load_capture(session)
    with pytest.raises(ValueError, match="프레임 슬립이 과반"):
        _analyse(capture)


def test_sub_band_consistency_is_recorded_per_band(tmp_path):
    """총계는 에너지 가중이라 약한 대역을 숨긴다 — 부대역별로 남겨야 게이트가 본다."""

    capture = ra.load_capture(_write_capture(tmp_path / "band"))
    results, _ = _analyse(capture)
    for item in results.values():
        model = item["model"]
        edges = model["band_consistency_hz"]
        assert edges.shape == (len(mpi.CONSISTENCY_SUB_BANDS_HZ), 2)
        inside = [
            value for (lo, hi), value in zip(edges, model["band_consistency"])
            if lo >= BAND_KWARGS["required_band_hz"][0]
            and hi <= BAND_KWARGS["required_band_hz"][1]
        ]
        assert inside, "필수 대역 안에 판정 가능한 부대역이 하나도 없다"
        assert min(inside) > mpi.MIN_BAND_CONSISTENCY
        # 자극 대역(100-1000Hz) 밖 부대역은 톤이 없어 판정 불가로 남는다.
        assert np.isnan(model["band_consistency"][-1])


def test_low_alignment_score_repeat_is_dropped(tmp_path):
    """정렬 신뢰도 하한 상향이 실제로 작동하는지 — 손상 반복 하나를 주입해 확인한다."""

    session = _write_capture(tmp_path / "score")
    capture = ra.load_capture(session)
    probe = capture["probe"]
    corrupt = capture["period_starts"][4]
    rng = np.random.default_rng(7)
    err = capture["err"].copy()
    err[corrupt : corrupt + probe.period_samples] = rng.normal(
        scale=float(np.std(err)), size=probe.period_samples
    )
    capture["err"] = err

    results, report = _analyse(capture)
    assert 4 in set(np.flatnonzero(~report["keep"]).tolist())
    for item in results.values():
        assert item["model"]["alignment_scores"][4] < mpi.DEFAULT_MIN_ALIGNMENT_SCORE


def test_metadata_forgery_is_detected(tmp_path):
    session = _write_capture(tmp_path / "forged", tamper_metadata_json=True)
    with pytest.raises(ValueError, match="metadata.json"):
        ra.load_capture(session)


def test_probe_reconstruction_mismatch_is_detected(tmp_path):
    """자극을 재구성하지 못하면 다른 신호를 분석하는 것이다 — 조용히 진행하면 안 된다."""

    session = _write_capture(
        tmp_path / "probe", metadata_override={"guard_bins": 3}
    )
    with pytest.raises(ValueError, match="프로브 재구성 실패"):
        ra.load_capture(session)


def test_defective_captures_are_refused(tmp_path):
    xrun = _write_capture(
        tmp_path / "xrun",
        metadata_override={"telemetry": {"xrun_count": 1, "completed": True}},
    )
    with pytest.raises(ValueError, match="xrun"):
        ra.load_capture(xrun)

    broken = _write_capture(
        tmp_path / "broken", metadata_override={"invalid_reasons": ["input_clip_ch0"]}
    )
    with pytest.raises(ValueError, match="캡처 자체가 결함"):
        ra.load_capture(broken)


@pytest.mark.parametrize(
    "override, expected",
    [
        ({"min_alignment_score": 0.80}, "min-alignment-score"),
        ({"max_relative_tau_samples": 48.0}, "max-relative-tau-samples"),
        ({"max_drift_deviation_samples": 25.0}, "max-drift-deviation-samples"),
        ({"min_kept_repeats": 3}, "min-kept-repeats"),
        ({"max_delay_jitter_ms": 1.0}, "max-delay-jitter-ms"),
        ({"consistency_band": [150.0, 600.0]}, "consistency-band"),
    ],
)
def test_loosening_arguments_are_refused(override, expected):
    args = ra.build_parser().parse_args(["results/calibration_interleaved/x"])
    for key, value in override.items():
        setattr(args, key, value)
    with pytest.raises(ValueError, match=expected):
        ra.reject_loosening(args)


def test_tightening_arguments_are_allowed():
    args = ra.build_parser().parse_args(["results/calibration_interleaved/x"])
    args.min_alignment_score = 0.99
    args.max_relative_tau_samples = 1.0
    args.max_drift_deviation_samples = 0.5
    args.min_kept_repeats = 16
    args.max_delay_jitter_ms = 0.02
    ra.reject_loosening(args)     # 예외가 없어야 한다


def test_defaults_come_from_the_measurement_script():
    """허용 범위의 단일 출처는 측정 스크립트다 — 재분석 도구가 따로 정하면 갈라진다."""

    args = ra.build_parser().parse_args(["results/calibration_interleaved/x"])
    assert args.min_alignment_score == mpi.DEFAULT_MIN_ALIGNMENT_SCORE
    assert args.max_relative_tau_samples == mpi.DEFAULT_MAX_RELATIVE_TAU_SAMPLES
    assert args.max_drift_deviation_samples == mpi.DEFAULT_MAX_DRIFT_DEVIATION_SAMPLES
    assert args.consistency_band == list(mpi.DEFAULT_CONSISTENCY_BAND_HZ)


def test_official_write_backs_up_the_previous_artifact(tmp_path):
    capture = ra.load_capture(_write_capture(tmp_path / "write"))
    results, report = _analyse(capture)
    arrays = mpi._official_arrays(
        model=results["cancel"]["model"],
        relative_delay_spread=int(report["relative_delay_spread_samples"]),
        max_delay_jitter_samples=3, fs=FS,
        consistency=float(results["cancel"]["model"]["consistency"]),
        band_hz=(150.0, 800.0), amplitude=AMPLITUDE, block_size=256, latency="low",
        output_channel="cancel", repeats=int(report["keep"].sum()), xrun_count=0,
        capture_id="cap-synthetic", probe=capture["probe"], drive="cancel",
        snr_db=results["cancel"]["snr_db"], period_seconds=PERIOD_SECONDS,
        drift_samples_per_period=float(report["drift_samples_per_period"]),
        relative_tau_max_abs=float(report["relative_tau_max_abs"]),
    )
    target = tmp_path / "out" / "secondary.npz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old-artifact")

    with pytest.raises(FileExistsError):
        ra._backup_and_replace(target, arrays, overwrite=False)

    ra._backup_and_replace(target, arrays, overwrite=True)
    backup = target.with_suffix(target.suffix + ".orig")
    assert backup.read_bytes() == b"old-artifact"
    with np.load(target, allow_pickle=False) as data:
        assert abs(int(data["delay_samples"]) - S_DELAY) <= 5
        assert data["band_consistency"].size == len(mpi.CONSISTENCY_SUB_BANDS_HZ)
        # 앵커 규약이 아티팩트에 박혀 있어야 절대 지연을 재현할 수 있다.
        assert int(data["anchor_repeat"]) == int(report["anchor"])
        assert data["kept_repeat_indices"].size == int(report["keep"].sum())
