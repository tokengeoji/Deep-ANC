#!/usr/bin/env python3
"""저장된 캡처를 재분석해 official P/S NPZ 를 만든다 — 스피커를 울리지 않는다.

왜 필요한가
----------
측정 후처리 결함(2026-08-05 결함 1)은 원시 캡처가 남아 있으면 재생 없이 고칠 수
있다. 실측: ``20260804_235822_03f4c088`` 를 재분석하면 150-1600Hz 반복 일관성이
P 0.9797→0.9994, S 0.9689→0.9991 로 올라가고 lead 가 113→116 으로 바뀐다.

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
      --primary-out assets/measured/primary_path_il.npz \\
      --secondary-out assets/measured/secondary_path_il.npz
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
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
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    with np.load(npz_path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        recorded_raw = np.asarray(data["input_raw_int32"], dtype=np.int32)
        preflight_raw = np.asarray(data["preflight_raw_int32"], dtype=np.int32)
        playback = np.asarray(data["output"], dtype=np.float64)

    # metadata.json 과 NPZ 안 사본이 일치해야 한다(둘 중 하나만 손댄 캡처 차단).
    on_disk = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
    if json.dumps(on_disk, sort_keys=True) != json.dumps(meta, sort_keys=True):
        raise ValueError("metadata.json 과 NPZ 내부 metadata_json 이 다릅니다")
    if meta.get("invalid_reasons"):
        raise ValueError(f"캡처 자체가 결함입니다: {meta['invalid_reasons']}")
    if int(meta.get("telemetry", {}).get("xrun_count", 0)) != 0:
        raise ValueError("xrun 이 있는 캡처는 재분석해도 official 이 될 수 없습니다")

    fs = int(meta["sample_rate"])
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

    err = pcm_int32_to_float32(recorded_raw)[:, 0].astype(np.float64)
    if meta.get("warp", {}).get("applied"):
        mono = playback[:, 0] + playback[:, 1]
        centres, delays, peaks = track_warp(
            mono, err, window=int(meta["warp"]["window"])
        )
        err = dewarp_recording(err, centres, delays, peaks, min_peak=0.2)

    # lead_in 은 2026-08-05 이전 캡처에 기록돼 있지 않다. 그 시절 코드가 쓴 값이
    # fs//2 하나뿐이므로 그것을 기본값으로 쓰되, 기록이 있으면 기록을 따른다.
    lead_in = int(meta.get("lead_in_samples", fs // 2))
    starts = [
        lead_in + (int(meta["warmup_periods"]) + k) * probe.period_samples
        for k in range(int(meta["repeats"]))
    ]
    if starts[-1] + probe.period_samples > err.size:
        raise ValueError(
            f"녹음이 짧아 마지막 주기를 자를 수 없습니다: {err.size} 샘플"
        )

    preflight_err = pcm_int32_to_float32(preflight_raw)[:, 0].astype(np.float64)
    if preflight_err.size < probe.period_samples:
        preflight_err = np.pad(
            preflight_err, (0, probe.period_samples - preflight_err.size)
        )
    noise_spectrum = np.fft.rfft(preflight_err[-probe.period_samples:])
    signal_spectrum = np.fft.rfft(err[starts[0]: starts[0] + probe.period_samples])

    return {
        "meta": meta,
        "sha256": digest,
        "fs": fs,
        "probe": probe,
        "err": err,
        "period_starts": starts,
        "lead_in": lead_in,
        "snr_spectra": (signal_spectrum, noise_spectrum),
    }


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
    fit_band = (float(args.fit_band[0]), float(args.fit_band[1]))
    consistency_band = (
        float(args.consistency_band[0]), float(args.consistency_band[1])
    )
    required_band = (float(args.required_band[0]), float(args.required_band[1]))
    max_jitter = int(round(float(args.max_delay_jitter_ms) / 1000.0 * fs))

    try:
        results, report = mpi.analyse_capture(
            err=capture["err"], probe=probe,
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
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
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

    out_dir = session / f"reanalysis_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    channel_band = {k: tuple(map(float, v)) for k, v in meta["channel_band_hz"].items()}
    try:
        primary_out = cw._repo_path(args.primary_out)
        secondary_out = cw._repo_path(args.secondary_out)
        if primary_out == secondary_out:
            raise ValueError("P 와 S 는 다른 파일이어야 합니다")
        notes = {}
        for drive, out_path in (("noise", primary_out), ("cancel", secondary_out)):
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
                output_channel=item["output_channel"],
                repeats=int(keep.sum()),
                xrun_count=int(meta["telemetry"].get("xrun_count", 0)),
                # capture_id 는 **원본과 동일하게** 유지한다 — finetune_readiness 의
                # P/S capture_id 일치 검사가 계속 성립해야 한다.
                capture_id=str(meta["capture_id"]),
                probe=probe, drive=drive, snr_db=item["snr_db"],
                period_seconds=float(meta["period_seconds"]),
                drift_samples_per_period=float(report["drift_samples_per_period"]),
                relative_tau_max_abs=float(report["relative_tau_max_abs"]),
            )
            arrays["consistency_band_hz"] = np.asarray(consistency_band, np.float64)
            arrays["reanalysed"] = np.bool_(True)
            arrays["source_capture_id"] = np.str_(str(meta["capture_id"]))
            arrays["source_npz_sha256"] = np.str_(capture["sha256"])
            arrays["reanalysis_params_json"] = np.str_(
                json.dumps(params, sort_keys=True)
            )
            notes[drive] = _backup_and_replace(
                out_path, arrays, overwrite=bool(args.overwrite)
            )
    except (OSError, ValueError, FileExistsError) as exc:
        print(f"[중단] 저장 실패: {exc}", file=sys.stderr)
        (out_dir / "report.json").write_text(
            json.dumps(cw._json_safe(summary), ensure_ascii=False, indent=2,
                       sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 2

    summary["primary_out"] = str(primary_out.relative_to(REPO_ROOT))
    summary["secondary_out"] = str(secondary_out.relative_to(REPO_ROOT))
    (out_dir / "report.json").write_text(
        json.dumps(cw._json_safe(summary), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"\n[성공] P {primary_out.relative_to(REPO_ROOT)} ({notes['noise']})\n"
        f"       S {secondary_out.relative_to(REPO_ROOT)} ({notes['cancel']})\n"
        f"       리포트 {(out_dir / 'report.json').relative_to(REPO_ROOT)}\n\n"
        f"duct.yaml: d_noise_delay_samples: {p_delay}\n"
        f"data_sim.yaml: digital_reference_lead_samples: {lead} "
        f"(= S {s_delay} + handoff {handoff} − P {p_delay})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
