"""acoustic 녹음의 순수 오프라인 진단 — 실제 잔류 ERR에 S를 다시 적용하지 않는다.

OFF/ON은 서로 다른 시각이다. 관측된 ERR 감소량을 ANC의 인과적 효과로 인증하지
않으며, ON의 REF에는 F 누설이 있을 수 있어 REF 정규화도 하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .acoustic_readiness import MIN_PATH_CONSISTENCY, build_acoustic_readiness_report

SIGNALS = ("err", "ref", "source", "control", "anc_gain")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(data, key: str):
    value = np.asarray(data[key])
    if value.ndim != 0:
        raise ValueError(f"{key}는 scalar여야 합니다")
    return value.item()


def _positive(value, name: str, *, allow_zero=False) -> float:
    if isinstance(value, (bool, str)):
        raise ValueError(f"{name}는 유한한 숫자여야 합니다")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}는 유한한 숫자여야 합니다") from exc
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        raise ValueError(f"{name} 범위가 잘못됐습니다")
    return result


def _load_recording(path: Path) -> tuple[int, dict, dict | None]:
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("녹음은 NPZ여야 합니다")
    with archive as data:
        if not {"fs", *SIGNALS}.issubset(data.files):
            raise ValueError("fs/err/ref/source/control/anc_gain이 모두 필요합니다")
        fs_value = _positive(_scalar(data, "fs"), "fs")
        if not fs_value.is_integer():
            raise ValueError("fs는 양의 정수여야 합니다")
        values = {}
        for key in SIGNALS:
            arr = np.asarray(data[key])
            if (arr.ndim != 1 or arr.size == 0 or arr.dtype.kind not in "fiu"
                    or not np.all(np.isfinite(arr))):
                raise ValueError(f"{key}: 비어 있지 않은 유한 실수 1차원 배열이 필요합니다")
            values[key] = arr.astype(np.float64)
            if np.max(np.abs(values[key])) > np.sqrt(np.finfo(np.float64).max / arr.size) / 2:
                raise ValueError(f"{key}: 전력 계산의 수치 범위를 벗어난 입력입니다")
        if len({arr.size for arr in values.values()}) != 1:
            raise ValueError("녹음 배열 길이가 다릅니다")
        if np.any((values["anc_gain"] < 0) | (values["anc_gain"] > 1)):
            raise ValueError("anc_gain은 0~1 범위여야 합니다")
        metadata_keys = {"recording_schema_version", "recording_meta_json"}
        present = metadata_keys.intersection(data.files)
        metadata = None
        if present:
            if present != metadata_keys:
                raise ValueError("지원하지 않거나 불완전한 recording schema입니다")
            version = _scalar(data, "recording_schema_version")
            if isinstance(version, bool) or not isinstance(version, int) or version != 1:
                raise ValueError("지원하지 않거나 불완전한 recording schema입니다")
            try:
                metadata = json.loads(_scalar(data, "recording_meta_json"))
            except (TypeError, ValueError) as exc:
                raise ValueError("recording_meta_json이 잘못됐습니다") from exc
            if not isinstance(metadata, dict):
                raise ValueError("recording_meta_json은 object여야 합니다")
            json.dumps(metadata, allow_nan=False)
    return int(fs_value), values, metadata


def _validate_binding(meta: dict, cfg: dict, secondary: dict, digest: str, n: int) -> None:
    audio = cfg["hardware"]["audio"]
    expected = {
        "reference": "mic", "controller": cfg.get("controller", "dl"),
        "sample_rate": audio["sample_rate"], "block_size": audio["block_size"],
        "hop": cfg["hop"], "latency": audio["latency"],
        "digital_reference_lead_samples": 0,
        "handoff_extra_samples": cfg["duct"]["secondary_path"]["handoff_extra_samples"],
        "secondary_sha256": digest, "secondary_delay_samples": secondary["delay_samples"],
        "secondary_fir_length": secondary["fir_length"], "recorded_samples": n,
        "control_limit": cfg.get("safety", {}).get("control_limit", 0.2),
        "dc_blocker_r": cfg["hardware"].get("dc_blocker_r", 0.995),
        "channels": cfg["hardware"]["channels"],
    }
    for key, value in expected.items():
        if key not in meta or meta[key] != value:
            raise ValueError(f"녹음 메타와 분석 설정/S가 다릅니다: {key}")
    requested = _positive(meta.get("record_requested_samples"), "record_requested_samples")
    if not requested.is_integer() or requested < n:
        raise ValueError("record_requested_samples가 기록 길이보다 작거나 정수가 아닙니다")
    health = meta.get("runtime_health")
    if not isinstance(health, dict) or health.get("scope") != "whole_runtime_including_startup_stop":
        raise ValueError("runtime_health의 누적 범위가 없거나 잘못됐습니다")
    for key in ("xrun_count", "input_ring_drops", "output_ring_drops", "output_ring_underruns"):
        value = health.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"runtime_health {key}는 0 이상의 정수여야 합니다")
    if not isinstance(health.get("fatal_error"), bool):
        raise ValueError("runtime_health fatal_error는 bool이어야 합니다")


def _runs(gain: np.ndarray) -> list[dict]:
    labels = np.where(gain <= 0.001, 0, np.where(gain >= 0.999, 1, -1))
    boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, gain.size]
    return [
        {"start": int(start), "stop": int(stop), "state": {0: "off", 1: "on", -1: "transition"}[int(labels[start])]}
        for start, stop in zip(boundaries[:-1], boundaries[1:])
    ]


def _ratio_db(numerator: float, denominator: float, floor: float) -> float | None:
    # 수치 바닥 이하를 0 dB나 무한대 감쇠로 변환하지 않는다.
    if numerator <= floor or denominator <= floor:
        return None
    return float(10 * (np.log10(numerator) - np.log10(denominator)))


def _frequency_bands(fs: int, secondary: dict) -> list[dict]:
    nyquist = fs / 2
    if nyquist <= 1000:
        raise ValueError("1 kHz 이상 진단에는 sample_rate > 2000이 필요합니다")
    trusted = secondary["consistency_band_hz"]
    quality = secondary["repeat_consistency"]
    validated = trusted is not None and quality is not None and quality >= MIN_PATH_CONSISTENCY
    bands = [("full", 0, nyquist), ("low_0_1000", 0, 1000), ("high_1000_nyquist", 1000, nyquist)]
    if validated:
        bands.append(("trusted", *trusted))
    for center in (125, 250, 500, 1000, 2000, 4000, 8000):
        low, high = center / np.sqrt(2), center * np.sqrt(2)
        if high < nyquist:
            bands.append((f"octave_{center}", low, high))
    return [
        {"band": name, "low_hz": float(low), "high_hz": float(high),
         "trusted": bool(validated and trusted[0] <= low and high <= trusted[1])}
        for name, low, high in bands
    ]


def _window_rows(values: dict, span: dict, fs: int, size: int, bands: list, cycle: int, phase: str) -> list:
    frequency = np.fft.rfftfreq(size, 1 / fs)
    weights = np.full(frequency.size, 2.0)
    weights[0] = 1.0
    if size % 2 == 0:
        weights[-1] = 1.0
    masks = [
        (frequency >= b["low_hz"]) & ((frequency < b["high_hz"]) if b["high_hz"] < fs / 2 else (frequency <= b["high_hz"]))
        for b in bands
    ]
    out = []
    for start in range(span["start"], span["stop"] - size + 1, size):
        powers = {
            key: np.abs(np.fft.rfft(values[key][start:start + size], norm="ortho")) ** 2 * weights / size
            for key in ("err", "ref")
        }
        for band, mask in zip(bands, masks):
            out.append({
                "cycle": cycle, "phase": phase, "start": start, "stop": start + size,
                "band": band["band"], "err_power": float(np.sum(powers["err"][mask])),
                "ref_power": float(np.sum(powers["ref"][mask])), "fft_bins": int(mask.sum()),
            })
    return out


def _metric_row(cycle: int, band: dict, windows: list, floor: float) -> dict:
    by_phase = {phase: [w for w in windows if w["phase"] == phase and w["band"] == band["band"]]
                for phase in ("off_pre", "on", "off_post")}
    err = {phase: float(np.mean([w["err_power"] for w in rows])) for phase, rows in by_phase.items()}
    ref = {phase: float(np.mean([w["ref_power"] for w in rows])) for phase, rows in by_phase.items()}
    baseline = min(err["off_pre"], err["off_post"])
    observations = [_ratio_db(baseline, w["err_power"], floor) for w in by_phase["on"]]
    all_valid = all(v is not None for v in observations)
    observed = np.array(observations, dtype=float) if all_valid else None
    count = len(observations)
    worst_count = math.ceil(count * 0.1)
    return {
        "cycle": cycle, **band,
        "off_pre_power": err["off_pre"], "off_post_power": err["off_post"], "on_power": err["on"],
        "observed_err_reduction_db": _ratio_db(baseline, err["on"], floor),
        "observed_vs_pre_db": _ratio_db(err["off_pre"], err["on"], floor),
        "observed_vs_post_db": _ratio_db(err["off_post"], err["on"], floor),
        "off_drift_db": _ratio_db(err["off_post"], err["off_pre"], floor),
        "median_observed_db": float(np.median(observed)) if all_valid else None,
        "p10_observed_db": float(np.percentile(observed, 10)) if all_valid else None,
        "worst_observed_db": float(np.min(observed)) if all_valid else None,
        "worst10_mean_db": float(np.mean(np.sort(observed)[:worst_count])) if all_valid else None,
        "n_on_windows": count, "worst10_window_count": worst_count,
        "n_power_invalid_on_windows": sum(v is None for v in observations),
        "emergent_on_energy": bool(baseline <= floor and any(w["err_power"] > floor for w in by_phase["on"])),
        "on_minus_baseline_power": err["on"] - baseline,
        "ref_pre_power": ref["off_pre"], "ref_on_power": ref["on"], "ref_post_power": ref["off_post"],
        "ref_on_vs_pre_db": _ratio_db(ref["on"], ref["off_pre"], floor),
        "ref_on_vs_post_db": _ratio_db(ref["on"], ref["off_post"], floor),
    }


def analyze_acoustic_session(
    npz_path: str | Path, cfg: dict, source_family: str, *, window_seconds: float = 1.0,
    startup_guard_seconds: float = 1.0, on_warmup_seconds: float = 2.0,
    edge_guard_seconds: float = 0.5, power_floor: float = 1e-12,
) -> dict:
    """새 파일을 쓰거나 장치를 열지 않는 진단. 구형 녹음의 모드는 사용자 제공 설정이다."""
    if not isinstance(source_family, str) or not source_family.strip():
        raise ValueError("source_family를 명시해야 합니다")
    readiness = build_acoustic_readiness_report(cfg)
    secondary = readiness["secondary_path"]
    path = Path(npz_path).expanduser().resolve()
    recording_hash = sha256_file(path)
    fs, values, metadata = _load_recording(path)
    if fs != readiness["sample_rate"]:
        raise ValueError("녹음 fs와 설정/S sample_rate가 다릅니다")
    secondary_hash = sha256_file(secondary["path"])
    n = len(values["err"])
    if metadata is not None:
        _validate_binding(metadata, cfg, secondary, secondary_hash, n)
    size = math.ceil(_positive(window_seconds, "window_seconds") * fs)
    if size < 2:
        raise ValueError("분석 창에는 2샘플 이상이 필요합니다")
    startup = math.ceil(_positive(startup_guard_seconds, "startup_guard_seconds", allow_zero=True) * fs)
    warmup = math.ceil(_positive(on_warmup_seconds, "on_warmup_seconds", allow_zero=True) * fs)
    edge = math.ceil(_positive(edge_guard_seconds, "edge_guard_seconds", allow_zero=True) * fs)
    floor = _positive(power_floor, "power_floor")
    # control/anc_gain은 핸드오프 이후 출력 callback의 인덱스다. handoff를 중복 가산하지 않는다.
    tail = secondary["delay_samples"] + secondary["fir_length"] - 1
    bands = _frequency_bands(fs, secondary)
    runs = _runs(values["anc_gain"])
    plateaus = [r for r in runs if r["state"] != "transition"]
    cycles, windows, metrics = [], [], []
    for index, run in enumerate(plateaus):
        if run["state"] != "on":
            continue
        cycle = {"cycle": len(cycles), "usable": False, "reason": None, "intervals": {"on_raw": dict(run)}}
        cycles.append(cycle)
        if index == 0 or plateaus[index - 1]["state"] != "off":
            cycle["reason"] = "missing_off_pre"
            continue
        if index + 1 == len(plateaus) or plateaus[index + 1]["state"] != "off":
            cycle["reason"] = "missing_off_post"
            continue
        spans = {}
        for phase, raw in (("off_pre", plateaus[index - 1]), ("on", run), ("off_post", plateaus[index + 1])):
            leading = max(edge, tail, warmup if phase == "on" else (startup if raw["start"] == 0 else 0))
            start, stop = min(raw["stop"], raw["start"] + leading), max(raw["start"], raw["stop"] - edge)
            stop = max(start, stop)
            length = stop - start
            spans[phase] = {"start": start, "stop": stop, "raw_start": raw["start"], "raw_stop": raw["stop"],
                            "window_count": length // size, "discarded_remainder_samples": length % size}
        cycle["intervals"].update(spans)
        if any(span["window_count"] == 0 for span in spans.values()):
            cycle["reason"] = "insufficient_guarded_duration"
            continue
        cycle["usable"] = True
        selected = [row for phase, span in spans.items()
                    for row in _window_rows(values, span, fs, size, bands, cycle["cycle"], phase)]
        windows.extend(selected)
        metrics.extend(_metric_row(cycle["cycle"], band, selected, floor) for band in bands)
    quality = {
        key: {"rms": float(np.sqrt(np.mean(values[key] ** 2))), "peak": float(np.max(np.abs(values[key]))),
              "recorded_clip_proxy_fraction": float(np.mean(np.abs(values[key]) >= 0.98))}
        for key in ("err", "ref", "source", "control")
    }
    limit = _positive(cfg.get("safety", {}).get("control_limit", 0.2), "control_limit")
    quality["control"]["limit_contact_fraction"] = float(np.mean(np.abs(values["control"]) >= limit * (1 - 1e-5)))
    issues = ["source_stationarity_unknown", "ref_feedback_path_not_compensated", "samplewise_runtime_health_unknown"]
    if metadata is None:
        issues.extend(["legacy_recording_context_unverified", "runtime_health_unknown"])
    elif any(metadata["runtime_health"][key] for key in (
        "xrun_count", "input_ring_drops", "output_ring_drops", "output_ring_underruns", "fatal_error",
    )):
        issues.append("runtime_faults_reported")
    if any(quality[key]["recorded_clip_proxy_fraction"] > 0 for key in ("err", "ref")):
        issues.append("recorded_input_clipping_proxy")
    if quality["source"]["peak"] > math.sqrt(floor):
        issues.append("internal_source_present")
    if quality["control"]["limit_contact_fraction"] > 0:
        issues.append("recorded_control_limit_contact")
    if quality["control"]["peak"] <= math.sqrt(floor):
        issues.append("control_inactive")
    if not cycles:
        issues.append("no_full_on_interval")
    if any(not c["usable"] for c in cycles):
        issues.append("incomplete_or_too_short_cycles")
    if sha256_file(path) != recording_hash or sha256_file(secondary["path"]) != secondary_hash:
        raise ValueError("분석 중 입력 녹음/S 파일이 변경됐습니다")
    return {
        "schema": "deep_anc.acoustic_session_report.v1", "source_family": source_family.strip(),
        "diagnostic_only": True, "performance_claim_allowed": False,
        "comparison_available": any(r["band"] == "full" and r["observed_err_reduction_db"] is not None for r in metrics),
        "all_cycles_complete": bool(cycles) and all(c["usable"] for c in cycles),
        "sample_rate": fs, "recorded_samples": n,
        "provenance": {"recording_path": str(path), "recording_sha256": recording_hash,
                       "secondary_path": secondary["path"], "secondary_sha256": secondary_hash,
                       "recording_context_verified": metadata is not None, "recording_metadata": metadata},
        "selection": {"window_samples": size, "startup_guard_samples": startup,
                      "on_warmup_samples": warmup, "edge_guard_samples": edge, "secondary_tail_samples": tail,
                      "handoff_added_to_recorded_output": False, "power_floor": floor},
        "method": {"power": "one_sided_parseval_per_fixed_window",
                   "frequency_edges": "[low, high), Nyquist 포함; low에는 DC 포함",
                   "baseline": "min(mean(pre_OFF_window_power), mean(post_OFF_window_power))",
                   "worst10": "작은 관측 감소량 ceil(N*0.1)개 평균; p10과 별도",
                   "ref_normalization": False, "secondary_reapplied_to_error": False},
        "quality": quality, "runtime_health": metadata.get("runtime_health") if metadata else None,
        "issues": issues, "path_readiness": readiness, "intervals": runs,
        "cycles": cycles, "metrics": metrics, "windows": windows,
    }
