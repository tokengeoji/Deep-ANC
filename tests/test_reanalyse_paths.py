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

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from deep_anc.dsp.interleaved_probe import build_interleaved_probe
from deep_anc.train import finetune_readiness as readiness

from scripts.data import measure_paths_interleaved as mpi
from scripts.data import reanalyse_paths_interleaved as ra


FS = 48_000
PERIOD_SECONDS = 0.125
BAND_HZ = (60.0, 1650.0)
AMPLITUDE = 0.003
WARMUP = 2
REPEATS = 16
P_DELAY = 600
S_DELAY = 360
FIT_BAND = (150.0, 1600.0)
BAND_KWARGS = dict(
    fit_band_hz=FIT_BAND,
    consistency_band_hz=(150.0, 1600.0),
    required_band_hz=(150.0, 1600.0),
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
    omit_measurement: bool = False,
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
    # 합성 plant를 선형 범위 안에서 int32로 옮긴다. 이전 40배는 peak 4.45로 심하게
    # clip되어, 실제 측정 preflight가 거부할 비선형 응답을 "clean" fixture로 만들었다.
    recorded_raw = np.rint(np.clip(recorded * 8.0, -1.0, 1.0) * INT32_MAX)

    rng = np.random.default_rng(11)
    preflight = rng.normal(scale=2e-4, size=(FS, 2))
    preflight_raw = np.rint(preflight * INT32_MAX)

    resolution = FS / period
    crest_noise, crest_cancel = probe.crest_db()
    metadata = {
        "capture_id": "cap-synthetic",
        "method": mpi.METHOD,
        "raw_capture_schema": mpi.RAW_CAPTURE_SCHEMA,
        "sample_rate": FS,
        "block_size": 256,
        "latency": "low",
        "channel_map": dict(mpi.OFFICIAL_CHANNEL_MAP),
        "operator_confirmations": {
            "user_present": True,
            "volume_minimum": True,
            "routing_and_geometry": True,
        },
        "amplitude": AMPLITUDE,
        "design_band_hz": [BAND_HZ[0], BAND_HZ[1]],
        "required_band_hz": [150.0, 1600.0],
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
        "telemetry": {
            "xrun_count": 0,
            "unexpected_status_count": 0,
            "callback_error": None,
            "captured_frames": int(recorded_raw.shape[0]),
            "completed": True,
        },
        "preflight": {
            **mpi.cw.analyze_int32_input_probe(preflight_raw.astype(np.int32)),
            "sample_rate": FS,
        },
        "measurement": mpi.cw.analyze_int32_input_probe(
            recorded_raw.astype(np.int32)
        ),
        "invalid_reasons": [],
        "valid": True,
        "analysis_contract": {
            "fit_band_hz": list(FIT_BAND),
            "consistency_band_hz": [150.0, 1600.0],
            "required_band_hz": [150.0, 1600.0],
            "fir_length": 2048,
            "pre_roll_samples": 256,
            "max_delay_samples": 1200,
            "min_alignment_score": mpi.DEFAULT_MIN_ALIGNMENT_SCORE,
            "min_kept_repeats": 8,
            "max_relative_tau_samples": mpi.DEFAULT_MAX_RELATIVE_TAU_SAMPLES,
            "max_drift_deviation_samples": (
                mpi.DEFAULT_MAX_DRIFT_DEVIATION_SAMPLES
            ),
            "max_delay_jitter_samples": 3,
            "clock_band_hz": list(mpi.CLOCK_BAND_HZ),
            "clock_min_adjacent_score": mpi.CLOCK_MIN_ADJACENT_SCORE,
            "clock_max_err_ref_delta_samples": (
                mpi.CLOCK_MAX_ERR_REF_DELTA_SAMPLES
            ),
            "clock_max_subwindow_spread_samples": (
                mpi.CLOCK_MAX_SUBWINDOW_SPREAD_SAMPLES
            ),
            "clock_max_adjacent_change_samples": (
                mpi.CLOCK_MAX_ADJACENT_CHANGE_SAMPLES
            ),
            "clock_max_abs_period_delta_samples": (
                mpi.CLOCK_MAX_ABS_PERIOD_DELTA_SAMPLES
            ),
            "separation_algorithm": mpi.SEPARATION_ALGORITHM,
            "separation_algorithm_version": mpi.SEPARATION_ALGORITHM_VERSION,
        },
    }
    metadata.update(metadata_override or {})
    if omit_measurement:
        metadata.pop("measurement", None)

    session.mkdir(parents=True, exist_ok=True)
    stored = dict(metadata)
    if tamper_metadata_json:
        stored = dict(metadata, amplitude=AMPLITUDE * 2)
    np.savez_compressed(
        session / "raw_measurement.npz",
        output=playback.astype(np.float32),
        output_pcm_int16=mpi.cw.float32_to_pcm_int16(playback.astype(np.float32)),
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
        fir_length=2048, pre_roll=256, max_delay_samples=1200,
        min_alignment_score=mpi.DEFAULT_MIN_ALIGNMENT_SCORE,
        min_kept_repeats=8,
        max_relative_tau_samples=mpi.DEFAULT_MAX_RELATIVE_TAU_SAMPLES,
        max_drift_deviation_samples=mpi.DEFAULT_MAX_DRIFT_DEVIATION_SAMPLES,
        max_delay_jitter_samples=3,
        **BAND_KWARGS,
    )
    kwargs.update(overrides)
    return mpi.analyse_capture(
        err=capture["err"], ref=capture["ref"],
        output_pcm_int16=capture["output_pcm_int16"], probe=capture["probe"],
        period_starts=capture["period_starts"],
        snr_spectra=capture["snr_spectra"], **kwargs,
    )


def test_clean_capture_round_trips_to_the_injected_plant(tmp_path):
    capture = ra.load_capture(_write_capture(tmp_path / "clean"))
    results, report = _analyse(capture)

    # 마지막 nominal segment에는 다음 cycle이 없어 독립 q/cubic witness를 만들 수 없다.
    assert int(report["keep"].sum()) == REPEATS - 1
    assert report["relative_delay_spread_samples"] == 0
    p_model = results["noise"]["model"]
    s_model = results["cancel"]["model"]
    p_delay = p_model["delay_samples"]
    s_delay = s_model["delay_samples"]
    # 절대 지연은 대역제한 정합필터라 커널 모양만큼 치우친다(설계상 재현 안 되는 양).
    assert abs(p_model["bulk_delay_samples"] - P_DELAY) <= 5
    assert abs(s_model["bulk_delay_samples"] - S_DELAY) <= 5
    assert p_delay == p_model["bulk_delay_samples"] - 256
    assert s_delay == s_model["bulk_delay_samples"] - 256
    # 재현되어야 하는 것은 **P−S** 다. 여기서는 정확히 맞아야 한다.
    assert p_delay - s_delay == P_DELAY - S_DELAY
    for item in results.values():
        assert item["reasons"] == [], item["reasons"]
        assert item["model"]["consistency"] > 0.999
        check = item["model"]["compact_transfer_round_trip"]
        assert check["passed"] is True
        assert check["complex_agreement"] >= mpi.MIN_COMPACT_TRANSFER_AGREEMENT
        assert check["relative_error"] <= mpi.MAX_COMPACT_TRANSFER_RELATIVE_ERROR


def test_postconversion_gate_rejects_odd_bin_periodic_preroll(tmp_path, monkeypatch):
    """반복 consistency가 완벽해도 odd S FIR 후처리가 틀리면 저장 전에 거부한다."""

    capture = ra.load_capture(_write_capture(tmp_path / "bad-postprocess"))
    corrected = mpi.channel_impulse_response

    def old_periodic_roll(probe, transfer, *, drive, pre_roll=0):
        unrolled = corrected(probe, transfer, drive=drive, pre_roll=0)
        return np.roll(unrolled, int(pre_roll))

    monkeypatch.setattr(mpi, "channel_impulse_response", old_periodic_roll)
    odd_drive = next(
        drive for drive in ("noise", "cancel")
        if int(capture["probe"].bins_for(drive)[0]) % 2 == 1
    )
    with pytest.raises(
        ValueError, match=rf"{odd_drive} compact FIR 복소 전달 round-trip 실패"
    ):
        _analyse(capture)


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
    clock_rejected = set(np.flatnonzero(~report["separation"]["valid"]).tolist())
    relative_rejected = set(report["relative_tau_rejected"].tolist())
    assert clock_rejected | relative_rejected == dropped
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


def test_frame_slip_gate_cannot_be_weakened_through_analysis_arguments(tmp_path):
    """직접 API 호출도 옛 완화값으로 q/상대지연 hard gate를 우회하지 못한다."""

    session = _write_capture(tmp_path / "slip2", slip_start_period=10, slip_samples=32)
    capture = ra.load_capture(session)

    with pytest.raises(ValueError, match="hard"):
        _analyse(
            capture, min_alignment_score=0.5,
            max_relative_tau_samples=1_000.0,
            max_drift_deviation_samples=1_000.0,
            max_delay_jitter_samples=48, min_kept_repeats=3,
        )

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
        # official compact gate가 요구하는 마지막 1000-1600Hz도 독립 판정된다.
        assert model["band_consistency"][-1] > mpi.MIN_BAND_CONSISTENCY


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
        np.testing.assert_array_equal(item["model"]["kept_mask"], report["keep"])
        assert item["model"]["aligned_transfers"].shape[0] == int(report["keep"].sum())


def test_metadata_forgery_is_detected(tmp_path):
    session = _write_capture(tmp_path / "forged", tamper_metadata_json=True)
    with pytest.raises(ValueError, match="metadata.json"):
        ra.load_capture(session)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("output", "stored ideal output"),
        ("output_pcm_int16", "stored actual output_pcm_int16"),
    ],
)
def test_playback_provenance_tampering_is_detected(tmp_path, field, message):
    session = _write_capture(tmp_path / f"tampered-{field}")
    raw = session / "raw_measurement.npz"
    with np.load(raw, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]).copy() for name in data.files}
    if field == "output":
        arrays[field][FS // 2, 0] += np.float32(1e-4)
    else:
        arrays[field][FS // 2, 0] += np.int16(1)
    np.savez_compressed(raw, **arrays)

    with pytest.raises(ValueError, match=message):
        ra.load_capture(session)


def test_reanalyse_loads_only_the_immutable_raw_pair(tmp_path):
    session = _write_capture(tmp_path / "immutable")
    raw = session / "raw_measurement.npz"
    raw_metadata = session / "metadata.json"
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (raw, raw_metadata)
    }
    mpi.write_analysis_outputs_atomic(
        session,
        metadata={"capture_id": "analysis-must-not-replace-raw", "valid": False},
        arrays={"untrusted_analysis": np.asarray([999.0])},
    )

    capture = ra.load_capture(session)

    assert capture["meta"]["capture_id"] == "cap-synthetic"
    assert capture["sha256"] == before["raw_measurement.npz"]
    assert capture["metadata_sidecar_recovered"] is False
    assert capture["output_pcm_provenance"] == mpi.OUTPUT_PCM_PROVENANCE_OBSERVED
    for path in (raw, raw_metadata):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before[path.name]


def test_reanalyse_hashes_and_loads_the_same_byte_snapshot(tmp_path, monkeypatch):
    session = _write_capture(tmp_path / "snapshot")
    raw = session / "raw_measurement.npz"
    original_bytes = raw.read_bytes()
    original_sha256 = hashlib.sha256(original_bytes).hexdigest()
    original_read_bytes = Path.read_bytes
    swapped = False

    def swap_after_snapshot(path):
        nonlocal swapped
        snapshot = original_read_bytes(path)
        if Path(path) == raw and not swapped:
            swapped = True
            raw.write_bytes(b"attacker replacement after snapshot")
        return snapshot

    monkeypatch.setattr(Path, "read_bytes", swap_after_snapshot)
    capture = ra.load_capture(session)

    assert swapped is True
    assert capture["sha256"] == original_sha256
    assert raw.read_bytes() == b"attacker replacement after snapshot"


def test_reanalyse_recovers_from_embedded_metadata_when_sidecar_is_missing(tmp_path):
    """sidecar 승격 실패 뒤에도 canonical raw 하나만으로 재분석할 수 있어야 한다."""

    session = _write_capture(tmp_path / "embedded-recovery")
    raw = session / "raw_measurement.npz"
    raw_sha256 = hashlib.sha256(raw.read_bytes()).hexdigest()
    (session / "metadata.json").unlink()

    capture = ra.load_capture(session)

    assert capture["metadata_sidecar_recovered"] is True
    assert capture["meta"]["capture_id"] == "cap-synthetic"
    assert capture["sha256"] == raw_sha256


def test_legacy_capture_derives_pcm_for_diagnostics_but_cannot_be_official(tmp_path):
    session = _write_capture(tmp_path / "legacy-no-observed-pcm")
    raw = session / "raw_measurement.npz"
    with np.load(raw, allow_pickle=False) as data:
        arrays = {
            name: np.asarray(data[name]).copy()
            for name in data.files
            if name != "output_pcm_int16"
        }
    np.savez_compressed(raw, **arrays)

    capture = ra.load_capture(session)

    assert capture["output_pcm_provenance"] == mpi.OUTPUT_PCM_PROVENANCE_DERIVED
    with pytest.raises(ValueError, match="관측한 output_pcm_int16"):
        ra.require_observed_output_pcm_for_official(capture)


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

    unexpected = _write_capture(
        tmp_path / "unexpected-status",
        metadata_override={
            "telemetry": {
                "xrun_count": 0,
                "unexpected_status_count": 1,
                "completed": True,
            }
        },
    )
    with pytest.raises(ValueError, match="unexpected callback status"):
        ra.load_capture(unexpected)

    incomplete = _write_capture(
        tmp_path / "incomplete",
        metadata_override={
            "telemetry": {
                "xrun_count": 0,
                "unexpected_status_count": 0,
                "completed": False,
            }
        },
    )
    with pytest.raises(ValueError, match="완료되지 않은"):
        ra.load_capture(incomplete)

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
        band_hz=(150.0, 1600.0), amplitude=AMPLITUDE, block_size=256, latency="low",
        channel_map=dict(mpi.OFFICIAL_CHANNEL_MAP),
        operator_confirmations={
            "user_present": True,
            "volume_minimum": True,
            "routing_and_geometry": True,
        },
        output_channel="cancel", repeats=int(report["keep"].sum()), xrun_count=0,
        capture_id="cap-synthetic", probe=capture["probe"], drive="cancel",
        snr_db=results["cancel"]["snr_db"], period_seconds=PERIOD_SECONDS,
        drift_samples_per_period=float(report["drift_samples_per_period"]),
        max_drift_deviation_samples=mpi.DEFAULT_MAX_DRIFT_DEVIATION_SAMPLES,
        relative_tau_max_abs=float(report["relative_tau_max_abs"]),
        source_raw_npz_path="results/raw_measurement.npz",
        source_raw_npz_sha256="a" * 64,
        source_analysis_npz_path="results/analysis_results.npz",
        source_analysis_npz_sha256="b" * 64,
        output_pcm_provenance=mpi.OUTPUT_PCM_PROVENANCE_OBSERVED,
        separation=report["separation"],
        separation_crosscheck=report["separation_crosscheck"],
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
        assert abs(int(data["bulk_delay_samples"]) - S_DELAY) <= 5
        assert int(data["delay_samples"]) == int(data["bulk_delay_samples"]) - 256
        assert int(data["pre_roll_samples"]) == 256
        assert str(data["delay_semantics"]) == mpi.DELAY_SEMANTICS
        assert float(data["compact_transfer_complex_agreement"]) >= (
            mpi.MIN_COMPACT_TRANSFER_AGREEMENT
        )
        assert float(data["compact_transfer_relative_error"]) <= (
            mpi.MAX_COMPACT_TRANSFER_RELATIVE_ERROR
        )
        assert data["aligned_mean_transfer_real"].shape == data[
            "tone_frequencies_hz"
        ].shape
        assert data["aligned_mean_transfer_imag"].shape == data[
            "tone_frequencies_hz"
        ].shape
        aligned = data["aligned_mean_transfer_real"] + 1j * data[
            "aligned_mean_transfer_imag"
        ]
        assert str(data["aligned_mean_transfer_sha256"]) == (
            mpi.aligned_transfer_sha256(data["tone_frequencies_hz"], aligned)
        )
        np.testing.assert_array_equal(
            data["compact_transfer_subband_hz"],
            np.asarray(mpi.COMPACT_TRANSFER_SUB_BANDS_HZ),
        )
        assert np.all(
            data["compact_transfer_subband_complex_agreement"]
            >= mpi.MIN_COMPACT_TRANSFER_AGREEMENT
        )
        assert np.all(
            data["compact_transfer_subband_relative_error"]
            <= mpi.MAX_COMPACT_TRANSFER_RELATIVE_ERROR
        )
        assert data["band_consistency"].size == len(mpi.CONSISTENCY_SUB_BANDS_HZ)
        # 앵커 규약이 아티팩트에 박혀 있어야 절대 지연을 재현할 수 있다.
        assert int(data["anchor_repeat"]) == int(report["anchor"])
        assert data["kept_repeat_indices"].size == int(report["keep"].sum())


def _run_reanalysis_write(
    tmp_path: Path, monkeypatch, *, fail_official: bool = False
) -> tuple[int, Path, Path, Path]:
    session = _write_capture(
        tmp_path / "results" / "capture", omit_measurement=True
    )
    primary = tmp_path / "assets" / "primary.npz"
    secondary = tmp_path / "assets" / "secondary.npz"
    monkeypatch.setattr(ra, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        ra.cw,
        "_repo_path",
        lambda value, require_results=False: Path(value).resolve(),
    )
    if fail_official:
        monkeypatch.setattr(
            ra.mpi,
            "write_official_pair_atomic",
            lambda *_a, **_k: (_ for _ in ()).throw(
                OSError("injected official pair failure")
            ),
        )
    result = ra.main(
        [
            str(session),
            "--write",
            "--primary-out", str(primary),
            "--secondary-out", str(secondary),
            "--fit-band", "150", "1600",
            "--fir-length", "2048",
            "--pre-roll", "256",
            "--max-delay-ms", "25",
        ]
    )
    return result, session, primary, secondary


def test_reanalysis_write_promotes_versioned_analysis_and_official_pair_atomically(
    tmp_path, monkeypatch, capsys
):
    result, session, primary, secondary = _run_reanalysis_write(
        tmp_path, monkeypatch
    )
    output = capsys.readouterr().out

    assert result == 0
    success = output[output.rfind("[성공]") :].strip()
    assert success == mpi.official_pair_success_message(
        primary,
        secondary,
        repository_root=tmp_path,
    ).strip()
    assert "d_noise_delay_samples" not in success
    assert "digital_reference_lead_samples" not in success
    assert "duct.yaml" not in success
    assert primary.is_file() and secondary.is_file()
    analysis = list(session.glob("analysis_results.reanalysis_*.npz"))
    metadata = list(session.glob("analysis_metadata.reanalysis_*.json"))
    assert len(analysis) == len(metadata) == 1
    with np.load(primary, allow_pickle=False) as p_data, np.load(
        secondary, allow_pickle=False
    ) as s_data:
        assert str(p_data["source_analysis_npz_path"]) == str(
            analysis[0].relative_to(tmp_path)
        )
        assert str(p_data["source_analysis_npz_path"]) == str(
            s_data["source_analysis_npz_path"]
        )
        assert str(p_data["source_analysis_npz_sha256"]) == hashlib.sha256(
            analysis[0].read_bytes()
        ).hexdigest()
    monkeypatch.setattr(readiness, "REPO_ROOT", tmp_path)
    primary_audit = readiness.audit_official_path_model(
        primary,
        expected_output_channel="noise",
        sample_rate=FS,
        required_band_hz=(150.0, 1600.0),
    )
    secondary_audit = readiness.audit_official_path_model(
        secondary,
        expected_output_channel="cancel",
        sample_rate=FS,
        required_band_hz=(150.0, 1600.0),
    )
    assert primary_audit["interleaved"]["separation"]["source_analysis"][
        "sha256"
    ] == secondary_audit["interleaved"]["separation"]["source_analysis"][
        "sha256"
    ]


def test_reanalysis_official_failure_keeps_raw_and_atomic_versioned_analysis(
    tmp_path, monkeypatch
):
    result, session, primary, secondary = _run_reanalysis_write(
        tmp_path, monkeypatch, fail_official=True
    )

    assert result == 2
    assert (session / "raw_measurement.npz").is_file()
    assert len(list(session.glob("analysis_results.reanalysis_*.npz"))) == 1
    assert len(list(session.glob("analysis_metadata.reanalysis_*.json"))) == 1
    assert not primary.exists() and not secondary.exists()
    assert not list(session.glob("*.partial"))
