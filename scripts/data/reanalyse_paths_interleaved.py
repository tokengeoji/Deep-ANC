#!/usr/bin/env python3
"""저장된 캡처를 재분석해 official P/S NPZ 를 만든다 — 스피커를 울리지 않는다.

왜 필요한가
----------
측정 후처리 결함은 immutable 원시 캡처가 남아 있으면 재생 없이 진단할 수 있다. 다만
옛 캡처처럼 실제 제출 PCM 관측값이나 q+joint-LS provenance가 없는 자료는
``derived_not_observed`` 진단 결과일 뿐 official 승격이나 training-ready 근거가 될 수 없다.
새 캡처만 actual int16 DAC 명령, time-domain clock witness, joint-LS 및 cubic crosscheck를
모두 보존하며, ``--write``도 그 엄격한 schema를 만족할 때만 official pair를 만든다.

무엇을 하지 않는가
----------------
파라미터를 바꿔가며 "좋아 보이는" 결과를 고르는 도구가 아니다. 기본값은 측정
스크립트와 **같은 상수**를 import 하고, 분석 자체도 ``measure_paths_interleaved``
의 ``analyse_capture`` 를 그대로 호출한다(온라인 경로와 코드가 갈라지면 재현성이
깨진다). 쓰기는 ``--write`` 를 명시해야만 일어나며, 사용한 모든 파라미터가
아티팩트와 리포트에 박힌다. 게이트를 **약화하는** 방향의 값은 거부한다 —
허용 범위의 단일 출처는 ``measure_paths_interleaved`` 의 ``DEFAULT_*`` 다.

사용법::

  .venv/bin/python scripts/data/reanalyse_paths_interleaved.py \\
      results/calibration_interleaved/20260804_235822_03f4c088 --dry-run

  .venv/bin/python scripts/data/reanalyse_paths_interleaved.py \\
      results/calibration_interleaved/20260804_235822_03f4c088 --write \\
      --primary-out assets/measured/primary_path_il_strict_<capture-id>.npz \\
      --secondary-out assets/measured/secondary_path_il_strict_<capture-id>.npz
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import calibrate_wideband as cw  # noqa: E402
import measure_paths_interleaved as mpi  # noqa: E402

from deep_anc.audio_io import pcm_int32_to_float32  # noqa: E402
from deep_anc.config import DEFAULT_HANDOFF_SAMPLES, REPO_ROOT  # noqa: E402
from deep_anc.dsp.timing import PlantDelays  # noqa: E402 — lead 유도의 단일 출처
from deep_anc.dsp.interleaved_probe import (  # noqa: E402
    build_interleaved_probe,
    dewarp_recording,
    track_warp,
)

MIN_KEPT_REPEATS = 8
MAX_DELAY_JITTER_MS = 0.0625      # 48kHz 에서 3 샘플


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", help="results/calibration_interleaved/<세션>")
    parser.add_argument("--primary-out", default=None)
    parser.add_argument("--secondary-out", default=None)
    parser.add_argument(
        "--write", action="store_true",
        help="지정하지 않으면 수치만 출력하고 아무것도 쓰지 않는다",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="--write 의 반대. 명시해도 되고 생략해도 기본이 dry-run 이다",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="기존 official NPZ 를 교체한다. 원본은 <이름>.orig 로 백업된다",
    )
    parser.add_argument("--fit-band", type=float, nargs=2,
                        default=list(mpi.DEFAULT_FIT_BAND_HZ))
    parser.add_argument("--consistency-band", type=float, nargs=2,
                        default=list(mpi.DEFAULT_CONSISTENCY_BAND_HZ))
    parser.add_argument("--required-band", type=float, nargs=2, default=[150.0, 1600.0])
    parser.add_argument("--min-alignment-score", type=float,
                        default=mpi.DEFAULT_MIN_ALIGNMENT_SCORE)
    parser.add_argument("--max-relative-tau-samples", type=float,
                        default=mpi.DEFAULT_MAX_RELATIVE_TAU_SAMPLES)
    parser.add_argument("--max-drift-deviation-samples", type=float,
                        default=mpi.DEFAULT_MAX_DRIFT_DEVIATION_SAMPLES)
    parser.add_argument("--min-kept-repeats", type=int, default=MIN_KEPT_REPEATS)
    parser.add_argument("--max-delay-jitter-ms", type=float, default=MAX_DELAY_JITTER_MS)
    parser.add_argument("--fir-length", type=int, default=2048)
    parser.add_argument("--pre-roll", type=int, default=256)
    parser.add_argument("--max-delay-ms", type=float, default=100.0)
    return parser


def reject_loosening(args: argparse.Namespace) -> None:
    """게이트를 **약화하는** 방향의 값은 거부한다. 강화는 허용한다.

    오프라인 재분석 도구는 본질적으로 "파라미터를 바꿔 결과를 고르는" 유혹을 만든다.
    이 함수가 없으면 게이트 전체가 무의미해진다.
    """

    numeric = {
        "fit-band": args.fit_band,
        "consistency-band": args.consistency_band,
        "required-band": args.required_band,
        "min-alignment-score": [args.min_alignment_score],
        "max-relative-tau-samples": [args.max_relative_tau_samples],
        "max-drift-deviation-samples": [args.max_drift_deviation_samples],
        "max-delay-jitter-ms": [args.max_delay_jitter_ms],
        "max-delay-ms": [args.max_delay_ms],
    }
    for name, values in numeric.items():
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name}에 NaN/Inf를 쓸 수 없습니다")
    for name, values in (
        ("fit-band", args.fit_band),
        ("consistency-band", args.consistency_band),
        ("required-band", args.required_band),
    ):
        if len(values) != 2 or not 0.0 < float(values[0]) < float(values[1]):
            raise ValueError(f"{name}은 증가하는 양의 [lo, hi]여야 합니다")
    if int(args.fir_length) <= 0 or int(args.pre_roll) < 0:
        raise ValueError("fir-length는 양수이고 pre-roll은 음수가 아니어야 합니다")
    if int(args.pre_roll) >= int(args.fir_length):
        raise ValueError("pre-roll은 fir-length보다 작아야 합니다")
    if float(args.max_delay_ms) <= 0.0:
        raise ValueError("max-delay-ms는 양수여야 합니다")

    checks = [
        (args.min_alignment_score < mpi.DEFAULT_MIN_ALIGNMENT_SCORE,
         "min-alignment-score"),
        (args.max_relative_tau_samples > mpi.DEFAULT_MAX_RELATIVE_TAU_SAMPLES,
         "max-relative-tau-samples"),
        (args.max_drift_deviation_samples > mpi.DEFAULT_MAX_DRIFT_DEVIATION_SAMPLES,
         "max-drift-deviation-samples"),
        (args.min_kept_repeats < MIN_KEPT_REPEATS, "min-kept-repeats"),
        (args.max_delay_jitter_ms > MAX_DELAY_JITTER_MS, "max-delay-jitter-ms"),
        (args.consistency_band[0] > args.required_band[0]
         or args.consistency_band[1] < args.required_band[1], "consistency-band"),
    ]
    bad = [name for condition, name in checks if condition]
    if bad:
        raise ValueError("게이트를 약화하는 인자는 쓸 수 없습니다: " + ", ".join(bad))


def load_capture(session: Path) -> dict[str, Any]:
    """캡처를 읽고 **프로브 재구성이 원본과 같은지 증명한다**.

    다르면 원본과 다른 자극을 분석하는 것이므로, 결과가 그럴듯해 보여도 틀린다.
    """

    npz_path = session / "raw_measurement.npz"
    # 한 번 읽은 immutable byte snapshot으로 SHA와 np.load를 모두 수행한다. path를
    # 두 번 열면 그 사이 rename/swap으로 "검사한 bytes != 분석한 bytes"가 될 수 있다.
    raw_bytes = npz_path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        recorded_raw = np.asarray(data["input_raw_int32"])
        preflight_raw = np.asarray(data["preflight_raw_int32"])
        stored_playback = np.asarray(data["output"])
        stored_err = np.asarray(data["err"]) if "err" in data.files else None
        stored_ref = np.asarray(data["ref"]) if "ref" in data.files else None
        stored_output_pcm = (
            np.asarray(data["output_pcm_int16"])
            if "output_pcm_int16" in data.files
            else None
        )

    # NPZ 내부 metadata_json이 canonical recovery source다. raw NPZ 승격 뒤 sidecar
    # rename만 실패할 수 있으므로 sidecar 부재는 embedded metadata로 읽기 전용 복구한다.
    # sidecar가 존재한다면 둘 중 하나만 손댄 위조는 계속 실패-폐쇄한다.
    metadata_path = session / "metadata.json"
    metadata_sidecar_recovered = not metadata_path.is_file()
    if not metadata_sidecar_recovered:
        on_disk = json.loads(metadata_path.read_text(encoding="utf-8"))
        if json.dumps(on_disk, sort_keys=True) != json.dumps(meta, sort_keys=True):
            raise ValueError("metadata.json 과 NPZ 내부 metadata_json 이 다릅니다")
    if meta.get("invalid_reasons"):
        raise ValueError(f"캡처 자체가 결함입니다: {meta['invalid_reasons']}")
    if int(meta.get("telemetry", {}).get("xrun_count", 0)) != 0:
        raise ValueError("xrun 이 있는 캡처는 재분석해도 official 이 될 수 없습니다")
    if int(meta.get("telemetry", {}).get("unexpected_status_count", 0)) != 0:
        raise ValueError(
            "unexpected callback status가 있는 캡처는 재분석해도 official 이 될 수 없습니다"
        )
    if meta.get("telemetry", {}).get("callback_error"):
        raise ValueError(
            "callback error가 있는 캡처는 재분석해도 official 이 될 수 없습니다"
        )
    if not bool(meta.get("telemetry", {}).get("completed", False)):
        raise ValueError("완료되지 않은 캡처는 재분석해도 official 이 될 수 없습니다")

    fs = int(meta["sample_rate"])
    channel_map = dict(meta.get("channel_map", mpi.OFFICIAL_CHANNEL_MAP))
    if meta.get("raw_capture_schema") == mpi.RAW_CAPTURE_SCHEMA:
        if fs != mpi.OFFICIAL_SAMPLE_RATE:
            raise ValueError(
                f"strict raw sample_rate={fs}; {mpi.OFFICIAL_SAMPLE_RATE}여야 합니다"
            )
        if meta.get("channel_map") != mpi.OFFICIAL_CHANNEL_MAP:
            raise ValueError("strict raw channel_map이 official 0/1 계약과 다릅니다")
        if meta.get("operator_confirmations") != {
            "user_present": True,
            "volume_minimum": True,
            "routing_and_geometry": True,
        }:
            raise ValueError("strict raw operator confirmations가 없습니다")
    probe = build_interleaved_probe(
        sample_rate=fs, period_seconds=float(meta["period_seconds"]),
        band_hz=tuple(meta["design_band_hz"]), amplitude=float(meta["amplitude"]),
        tone_spacing_hz=None,
    )
    if probe.guard_bins() != int(meta["guard_bins"]):
        raise ValueError("프로브 재구성 실패: guard_bins 불일치")
    crest = probe.crest_db()
    for got, want in zip(crest, (meta["crest_db"]["noise"], meta["crest_db"]["cancel"])):
        if abs(float(got) - float(want)) > 1e-6:
            raise ValueError(f"프로브 재구성 실패: crest {got} != {want}")
    recon = {
        key: [
            float(v) for v in probe.bins_for(key)[[0, -1]] * fs / probe.period_samples
        ]
        for key in ("noise", "cancel")
    }
    stored = {k: [float(x) for x in v] for k, v in meta["channel_band_hz"].items()}
    if recon != stored:
        raise ValueError(f"프로브 재구성 실패: channel_band_hz {recon} != {stored}")

    # raw가 주장하는 ideal probe와 실제 callback에 제출한 int16 command를 모두
    # metadata만으로 재구성한다. 어느 하나라도 없거나 한 code라도 다르면 다른 자극의
    # 응답을 분석할 수 있으므로 fail-closed한다.
    lead_in = int(meta.get("lead_in_samples", fs // 2))
    total_periods = int(meta["warmup_periods"]) + int(meta["repeats"])
    expected_playback = np.zeros(
        (lead_in + total_periods * probe.period_samples, 2), dtype=np.float32
    )
    expected_playback[lead_in:, channel_map["noise_out"]] = np.tile(
        probe.noise_signal, total_periods
    )
    expected_playback[lead_in:, channel_map["cancel_out"]] = np.tile(
        probe.cancel_signal, total_periods
    )
    expected_frames = int(expected_playback.shape[0])
    if recorded_raw.dtype != np.int32 or recorded_raw.shape != (expected_frames, 2):
        raise ValueError(
            f"input_raw_int32 dtype/shape 계약 위반: "
            f"{recorded_raw.dtype} {recorded_raw.shape} != int32 {(expected_frames, 2)}"
        )
    if preflight_raw.dtype != np.int32 or preflight_raw.ndim != 2 or preflight_raw.shape[1] != 2:
        raise ValueError("preflight_raw_int32은 [frames,2] int32여야 합니다")
    telemetry = meta.get("telemetry", {})
    if int(telemetry.get("captured_frames", expected_frames)) != expected_frames:
        raise ValueError("telemetry.captured_frames가 expected playback frames와 다릅니다")
    if stored_playback.dtype != np.float32 or not np.array_equal(
        stored_playback, expected_playback
    ):
        raise ValueError("stored ideal output이 metadata로 재구성한 probe와 다릅니다")
    expected_output_pcm = cw.float32_to_pcm_int16(expected_playback)
    if stored_output_pcm is None:
        # 구 캡처는 callback에 제출한 PCM 자체를 저장하지 않았다. ideal float가 exact인
        # 경우에만 분석용으로 결정론적으로 복원하되, 관측 provenance로 승격하지 않는다.
        output_pcm_provenance = mpi.OUTPUT_PCM_PROVENANCE_DERIVED
    else:
        if stored_output_pcm.dtype != np.int16 or not np.array_equal(
            stored_output_pcm, expected_output_pcm
        ):
            raise ValueError("stored actual output_pcm_int16이 ideal probe 양자화와 다릅니다")
        output_pcm_provenance = mpi.OUTPUT_PCM_PROVENANCE_OBSERVED
    playback = stored_playback.astype(np.float64)

    recorded = pcm_int32_to_float32(recorded_raw)
    expected_err = recorded[:, channel_map["error_mic"]].astype(np.float32)
    expected_ref = recorded[:, channel_map["reference_mic"]].astype(np.float32)
    if stored_err is not None and (
        stored_err.dtype != np.float32 or not np.array_equal(stored_err, expected_err)
    ):
        raise ValueError("stored err float32가 input_raw_int32 변환과 다릅니다")
    if stored_ref is not None and (
        stored_ref.dtype != np.float32 or not np.array_equal(stored_ref, expected_ref)
    ):
        raise ValueError("stored ref float32가 input_raw_int32 변환과 다릅니다")

    recomputed_measurement = cw.analyze_int32_input_probe(recorded_raw)
    if "measurement" in meta and json.dumps(
        cw._json_safe(recomputed_measurement), sort_keys=True
    ) != json.dumps(cw._json_safe(meta["measurement"]), sort_keys=True):
        raise ValueError("measurement report가 input_raw_int32 재계산과 다릅니다")
    channels = recomputed_measurement.get("channels", [])
    if len(channels) < 2 or not all(bool(item.get("valid")) for item in channels[:2]):
        raise ValueError("input_raw_int32 ERR/REF channel이 유효하지 않습니다")
    recomputed_preflight = cw.analyze_int32_input_probe(preflight_raw)
    stored_preflight = meta.get("preflight", {})
    for key in ("frames", "channels"):
        if json.dumps(
            cw._json_safe(recomputed_preflight.get(key)), sort_keys=True
        ) != json.dumps(cw._json_safe(stored_preflight.get(key)), sort_keys=True):
            raise ValueError(f"preflight report {key}가 raw 재계산과 다릅니다")
    if int(stored_preflight.get("sample_rate", fs)) != fs:
        raise ValueError("preflight sample_rate가 capture sample_rate와 다릅니다")

    err = recorded[:, channel_map["error_mic"]].astype(np.float64)
    ref = recorded[:, channel_map["reference_mic"]].astype(np.float64)
    if meta.get("warp", {}).get("applied"):
        mono = playback[:, 0] + playback[:, 1]
        centres, delays, peaks = track_warp(
            mono, err, window=int(meta["warp"]["window"])
        )
        err = dewarp_recording(err, centres, delays, peaks, min_peak=0.2)

    # lead_in 은 2026-08-05 이전 캡처에 기록돼 있지 않다. 그 시절 코드가 쓴 값이
    # fs//2 하나뿐이므로 그것을 기본값으로 쓰되, 기록이 있으면 기록을 따른다.
    starts = [
        lead_in + (int(meta["warmup_periods"]) + k) * probe.period_samples
        for k in range(int(meta["repeats"]))
    ]
    if starts[-1] + probe.period_samples > err.size:
        raise ValueError(
            f"녹음이 짧아 마지막 주기를 자를 수 없습니다: {err.size} 샘플"
        )

    preflight_err = pcm_int32_to_float32(preflight_raw)[
        :, channel_map["error_mic"]
    ].astype(np.float64)
    if preflight_err.size < probe.period_samples:
        preflight_err = np.pad(
            preflight_err, (0, probe.period_samples - preflight_err.size)
        )
    noise_spectrum = np.fft.rfft(preflight_err[-probe.period_samples:])
    signal_spectrum = np.fft.rfft(err[starts[0]: starts[0] + probe.period_samples])

    return {
        "meta": meta,
        "sha256": digest,
        "metadata_sidecar_recovered": metadata_sidecar_recovered,
        "output_pcm_provenance": output_pcm_provenance,
        "fs": fs,
        "probe": probe,
        "err": err,
        "ref": ref,
        "output_pcm_int16": (
            expected_output_pcm
            if stored_output_pcm is None
            else stored_output_pcm.astype(np.int16, copy=False)
        ),
        "period_starts": starts,
        "lead_in": lead_in,
        "snr_spectra": (signal_spectrum, noise_spectrum),
    }


def require_observed_output_pcm_for_official(capture: dict[str, Any]) -> None:
    """legacy 파생 PCM은 진단 분석만 허용하고 official 승격은 막는다."""

    provenance = capture.get("output_pcm_provenance")
    if provenance != mpi.OUTPUT_PCM_PROVENANCE_OBSERVED:
        raise ValueError(
            "official 저장에는 캡처 당시 관측한 output_pcm_int16이 필요합니다; "
            f"provenance={provenance!r}"
        )
    meta = capture.get("meta", {})
    if meta.get("raw_capture_schema") != mpi.RAW_CAPTURE_SCHEMA:
        raise ValueError(
            f"official 저장에는 strict raw schema {mpi.RAW_CAPTURE_SCHEMA!r}가 필요합니다"
        )
    if int(meta.get("sample_rate", -1)) != mpi.OFFICIAL_SAMPLE_RATE:
        raise ValueError(
            f"official 저장에는 sample_rate={mpi.OFFICIAL_SAMPLE_RATE}가 필요합니다"
        )
    if meta.get("channel_map") != mpi.OFFICIAL_CHANNEL_MAP:
        raise ValueError(
            f"official 저장에는 exact channel map {mpi.OFFICIAL_CHANNEL_MAP}가 필요합니다"
        )
    if meta.get("operator_confirmations") != {
        "user_present": True,
        "volume_minimum": True,
        "routing_and_geometry": True,
    }:
        raise ValueError("official 저장에는 세 operator confirmation이 모두 필요합니다")
    if bool(meta.get("warp", {}).get("applied", False)):
        raise ValueError("legacy dewarp/warp 적용 캡처는 diagnostic-only이며 official이 될 수 없습니다")


def require_official_analysis_contract(
    capture: dict[str, Any], args: argparse.Namespace
) -> None:
    """official 재분석은 immutable raw에 캡처 전에 박힌 분석 계약과 exact해야 한다."""

    meta = capture.get("meta", {})
    contract = meta.get("analysis_contract")
    if not isinstance(contract, dict):
        raise ValueError("strict raw analysis_contract가 없어 official 재분석할 수 없습니다")
    fs = int(capture["fs"])
    requested = {
        "fit_band_hz": [float(v) for v in args.fit_band],
        "consistency_band_hz": [float(v) for v in args.consistency_band],
        "required_band_hz": [float(v) for v in args.required_band],
        "fir_length": int(args.fir_length),
        "pre_roll_samples": int(args.pre_roll),
        "max_delay_samples": int(round(float(args.max_delay_ms) / 1000.0 * fs)),
        "min_alignment_score": float(args.min_alignment_score),
        "min_kept_repeats": int(args.min_kept_repeats),
        "max_relative_tau_samples": float(args.max_relative_tau_samples),
        "max_drift_deviation_samples": float(args.max_drift_deviation_samples),
        "max_delay_jitter_samples": int(
            round(float(args.max_delay_jitter_ms) / 1000.0 * fs)
        ),
    }
    for key, value in requested.items():
        stored = contract.get(key)
        if isinstance(value, list):
            equal = (
                isinstance(stored, list)
                and len(stored) == len(value)
                and np.array_equal(
                    np.asarray(stored, dtype=np.float64),
                    np.asarray(value, dtype=np.float64),
                )
            )
        elif isinstance(value, float):
            equal = np.isfinite(value) and float(stored) == value
        else:
            equal = stored == value
        if not equal:
            raise ValueError(
                f"official reanalysis parameter {key}={value!r}가 "
                f"raw canonical {stored!r}와 다릅니다"
            )
    fixed = {
        "clock_band_hz": list(mpi.CLOCK_BAND_HZ),
        "clock_min_adjacent_score": mpi.CLOCK_MIN_ADJACENT_SCORE,
        "clock_max_err_ref_delta_samples": mpi.CLOCK_MAX_ERR_REF_DELTA_SAMPLES,
        "clock_max_subwindow_spread_samples": (
            mpi.CLOCK_MAX_SUBWINDOW_SPREAD_SAMPLES
        ),
        "clock_max_adjacent_change_samples": mpi.CLOCK_MAX_ADJACENT_CHANGE_SAMPLES,
        "clock_max_abs_period_delta_samples": (
            mpi.CLOCK_MAX_ABS_PERIOD_DELTA_SAMPLES
        ),
        "separation_algorithm": mpi.SEPARATION_ALGORITHM,
        "separation_algorithm_version": mpi.SEPARATION_ALGORITHM_VERSION,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"raw analysis_contract {key}={contract.get(key)!r}; "
                f"required={expected!r}"
            )
    unambiguous = int(capture["probe"].period_samples) // int(
        capture["probe"].bin_step("noise")
    )
    if not (
        0 <= requested["pre_roll_samples"] < requested["fir_length"] <= unambiguous
        and requested["pre_roll_samples"] < requested["max_delay_samples"]
    ):
        raise ValueError("raw canonical FIR/pre-roll/max-delay sparse alias 계약 위반")


def _backup_and_replace(path: Path, arrays: dict[str, Any], *, overwrite: bool) -> str:
    """official NPZ 를 쓴다. 기존 파일은 ``.orig`` 로 백업하고 나서 교체한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    note = "생성"
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"기존 정식 모델은 덮어쓰지 않습니다: {path}")
        backup = path.with_suffix(path.suffix + ".orig")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
            note = f"교체 (백업 {backup.name})"
        else:
            note = f"교체 (백업 {backup.name} 유지)"
        path.unlink()
    cw.save_official_model(path, valid=True, arrays=arrays)
    return note


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    write = bool(args.write) and not bool(args.dry_run)
    if args.write and args.dry_run:
        print("[중단] --write 와 --dry-run 을 함께 쓸 수 없습니다", file=sys.stderr)
        return 2

    try:
        reject_loosening(args)
        session = cw._repo_path(args.capture_dir, require_results=True)
        capture = load_capture(session)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[중단] {exc}", file=sys.stderr)
        return 2

    meta, fs, probe = capture["meta"], capture["fs"], capture["probe"]
    if write:
        try:
            require_observed_output_pcm_for_official(capture)
            require_official_analysis_contract(capture, args)
        except ValueError as exc:
            print(f"[중단] {exc}", file=sys.stderr)
            return 2
    fit_band = (float(args.fit_band[0]), float(args.fit_band[1]))
    consistency_band = (
        float(args.consistency_band[0]), float(args.consistency_band[1])
    )
    required_band = (float(args.required_band[0]), float(args.required_band[1]))
    max_jitter = int(round(float(args.max_delay_jitter_ms) / 1000.0 * fs))

    try:
        results, report = mpi.analyse_capture(
            err=capture["err"],
            ref=capture["ref"],
            output_pcm_int16=capture["output_pcm_int16"],
            probe=probe,
            period_starts=capture["period_starts"],
            snr_spectra=capture["snr_spectra"],
            fir_length=int(args.fir_length), pre_roll=int(args.pre_roll),
            max_delay_samples=int(round(args.max_delay_ms / 1000.0 * fs)),
            fit_band_hz=fit_band, consistency_band_hz=consistency_band,
            required_band_hz=required_band,
            min_alignment_score=float(args.min_alignment_score),
            min_kept_repeats=int(args.min_kept_repeats),
            max_relative_tau_samples=float(args.max_relative_tau_samples),
            max_drift_deviation_samples=float(args.max_drift_deviation_samples),
            max_delay_jitter_samples=max_jitter,
        )
    except ValueError as exc:
        print(f"[기각] {session.name}: {exc}", file=sys.stderr)
        return 1

    keep, anchor = report["keep"], int(report["anchor"])
    p_delay = int(results["noise"]["model"]["delay_samples"])
    s_delay = int(results["cancel"]["model"]["delay_samples"])
    # handoff 상수와 lead 관계식을 손으로 다시 쓰지 않는다 (발생기 A).
    handoff = DEFAULT_HANDOFF_SAMPLES
    lead = int(
        PlantDelays(
            primary_delay_samples=p_delay,
            secondary_delay_samples=s_delay,
            handoff_samples=handoff,
            sample_rate=int(fs),
        )
        .lead()
        .samples
    )
    dropped = [int(v) for v in np.flatnonzero(~keep)]

    print(
        f"=== {session.name} ===\n"
        f"  kept={int(keep.sum())}/{keep.size}  버린 반복 {dropped}  anchor={anchor}\n"
        f"  drift={report['drift_samples_per_period']:.2f} 샘플/주기 "
        f"({report['drift_ppm']:.0f} ppm) · "
        f"유지 |rel|max={report['relative_tau_max_abs']:.3f} "
        f"(spread {report['relative_delay_spread_samples']}, 허용 {max_jitter})\n"
        f"  P={p_delay}  S={s_delay}  P-S={p_delay - s_delay}  lead={lead}"
    )
    for drive, label in (("noise", "P"), ("cancel", "S")):
        model = results[drive]["model"]
        bands = "  ".join(
            f"{lo:.0f}-{hi:.0f}:{value:.4f}"
            for (lo, hi), value in zip(
                model["band_consistency_hz"], model["band_consistency"]
            )
        )
        print(
            f"  {label} {consistency_band[0]:.0f}-{consistency_band[1]:.0f} "
            f"{model['consistency']:.4f} (전대역 {model['fullband_consistency']:.4f}) "
            f"| {bands}"
        )
        if results[drive]["reasons"]:
            print(f"    [미달] {', '.join(results[drive]['reasons'])}")

    valid = not results["noise"]["reasons"] and not results["cancel"]["reasons"]
    params = {
        "min_alignment_score": float(args.min_alignment_score),
        "max_relative_tau_samples": float(args.max_relative_tau_samples),
        "max_drift_deviation_samples": float(args.max_drift_deviation_samples),
        "min_kept_repeats": int(args.min_kept_repeats),
        "max_delay_jitter_samples": int(max_jitter),
        "fit_band_hz": list(fit_band),
        "consistency_band_hz": list(consistency_band),
        "required_band_hz": list(required_band),
        "fir_length": int(args.fir_length),
        "pre_roll": int(args.pre_roll),
    }
    summary = {
        "reanalysed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_capture_dir": str(session.relative_to(REPO_ROOT)),
        "source_capture_id": str(meta["capture_id"]),
        "source_npz_sha256": capture["sha256"],
        "output_pcm_provenance": capture["output_pcm_provenance"],
        "params": params,
        "valid": bool(valid),
        "anchor_repeat": anchor,
        "kept_repeat_indices": [int(v) for v in np.flatnonzero(keep)],
        "dropped_repeat_indices": dropped,
        "drift_rejected_repeats": [int(v) for v in report["drift_rejected"]],
        "relative_tau_rejected_repeats": [
            int(v) for v in report["relative_tau_rejected"]
        ],
        "drift_samples_per_period": float(report["drift_samples_per_period"]),
        "drift_ppm": float(report["drift_ppm"]),
        "relative_tau_centre_samples": float(report["relative_tau_centre"]),
        "relative_tau_max_abs_samples": float(report["relative_tau_max_abs"]),
        "relative_delay_spread_samples": int(report["relative_delay_spread_samples"]),
        "primary_delay_samples": p_delay,
        "secondary_delay_samples": s_delay,
        "relative_delay_samples": p_delay - s_delay,
        "digital_reference_lead_samples": lead,
        "channels": {
            drive: {
                "consistency": float(item["model"]["consistency"]),
                "fullband_consistency": float(item["model"]["fullband_consistency"]),
                "band_consistency": [
                    float(v) for v in item["model"]["band_consistency"]
                ],
                "band_consistency_hz": [
                    [float(lo), float(hi)]
                    for lo, hi in item["model"]["band_consistency_hz"]
                ],
                "delay_samples": int(item["model"]["delay_samples"]),
                "tone_snr_median_db": float(np.median(item["snr_db"])),
                "reasons": item["reasons"],
            }
            for drive, item in results.items()
        },
    }

    if not write:
        print("  [dry-run] 아무것도 쓰지 않았습니다 (--write 로 저장)")
        return 0 if valid else 1
    if not valid:
        print("\n[실패] 게이트 미달 — official 을 쓰지 않습니다", file=sys.stderr)
        return 1
    if not args.primary_out or not args.secondary_out:
        print("[중단] --write 에는 --primary-out/--secondary-out 이 필요합니다",
              file=sys.stderr)
        return 2

    if args.overwrite:
        print(
            "[중단] pair-atomic official은 기존 파일을 in-place 교체하지 않습니다; "
            "새 P/S 경로를 지정하세요",
            file=sys.stderr,
        )
        return 2
    channel_band = {k: tuple(map(float, v)) for k, v in meta["channel_band_hz"].items()}
    try:
        primary_out = cw._repo_path(args.primary_out)
        secondary_out = cw._repo_path(args.secondary_out)
        if primary_out == secondary_out:
            raise ValueError("P 와 S 는 다른 파일이어야 합니다")
        analysis_suffix = (
            ".reanalysis_"
            + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_")
            + uuid.uuid4().hex[:8]
        )
        summary["intended_primary_out"] = str(primary_out)
        summary["intended_secondary_out"] = str(secondary_out)
        analysis_paths = mpi.write_analysis_outputs_atomic(
            session,
            metadata=summary,
            arrays=mpi.analysis_provenance_arrays(results, report),
            suffix=analysis_suffix,
        )
        source_raw_path = str((session / "raw_measurement.npz").relative_to(REPO_ROOT))
        source_analysis_path = str(
            analysis_paths["results"].relative_to(REPO_ROOT)
        )
        source_analysis_sha256 = hashlib.sha256(
            analysis_paths["results"].read_bytes()
        ).hexdigest()
        official: dict[str, dict[str, Any]] = {}
        for drive in ("noise", "cancel"):
            item = results[drive]
            arrays = mpi._official_arrays(
                model=item["model"],
                relative_delay_spread=int(report["relative_delay_spread_samples"]),
                max_delay_jitter_samples=max_jitter,
                fs=fs, consistency=float(item["model"]["consistency"]),
                band_hz=channel_band[drive],
                amplitude=float(meta["amplitude"]),
                block_size=int(meta["block_size"]),
                latency=str(meta["latency"]),
                channel_map=dict(meta["channel_map"]),
                operator_confirmations=dict(meta["operator_confirmations"]),
                output_channel=item["output_channel"],
                repeats=int(keep.sum()),
                xrun_count=int(meta["telemetry"].get("xrun_count", 0)),
                # capture_id 는 **원본과 동일하게** 유지한다 — finetune_readiness 의
                # P/S capture_id 일치 검사가 계속 성립해야 한다.
                capture_id=str(meta["capture_id"]),
                probe=probe, drive=drive, snr_db=item["snr_db"],
                period_seconds=float(meta["period_seconds"]),
                drift_samples_per_period=float(report["drift_samples_per_period"]),
                max_drift_deviation_samples=float(
                    args.max_drift_deviation_samples
                ),
                relative_tau_max_abs=float(report["relative_tau_max_abs"]),
                source_raw_npz_path=source_raw_path,
                source_raw_npz_sha256=capture["sha256"],
                source_analysis_npz_path=source_analysis_path,
                source_analysis_npz_sha256=source_analysis_sha256,
                output_pcm_provenance=capture["output_pcm_provenance"],
                separation=report["separation"],
                separation_crosscheck=report["separation_crosscheck"],
            )
            arrays["consistency_band_hz"] = np.asarray(consistency_band, np.float64)
            arrays["reanalysed"] = np.bool_(True)
            arrays["source_capture_id"] = np.str_(str(meta["capture_id"]))
            arrays["source_npz_sha256"] = np.str_(capture["sha256"])
            arrays["reanalysis_params_json"] = np.str_(
                json.dumps(params, sort_keys=True)
            )
            official[drive] = arrays
        mpi.write_official_pair_atomic(
            primary_out,
            official["noise"],
            secondary_out,
            official["cancel"],
        )
    except (OSError, ValueError, FileExistsError) as exc:
        print(
            f"[중단] 저장 실패: {exc}. immutable raw 및 완성된 versioned analysis는 "
            "삭제하지 않았습니다",
            file=sys.stderr,
        )
        return 2

    print(
        mpi.official_pair_success_message(
            primary_out,
            secondary_out,
            repository_root=REPO_ROOT,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
