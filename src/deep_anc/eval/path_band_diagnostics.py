"""저장 ESS 반복의 대역별 진단. S 모델·신뢰대역을 생성하거나 승격하지 않는다.

정렬 전후는 동일한 compact IR을 사용한다. fixed_origin은 compact FFT에 저장된
반복 지연의 위상만 복원하므로 두 결과의 차이를 추출 창 차이와 혼동하지 않는다.
반복 일관성은 입력 SNR·실시간 클록 안정성·ANC 감쇠의 증명이 아니다.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


NUMERICAL_POWER_FLOOR = 1e-30
REFERENCE_CONSISTENCY = 0.9


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name}: 정수가 필요합니다")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number < minimum:
        raise ValueError(f"{name}: {minimum} 이상의 유한 정수가 필요합니다")
    return int(number)


def _band(values, fs: int, name: str) -> tuple[float, float]:
    raw = np.asarray(values)
    if raw.shape != (2,) or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError(f"{name}: 유한한 두 주파수가 필요합니다")
    lo, hi = map(float, raw)
    if not 0 <= lo < hi <= fs / 2:
        raise ValueError(f"{name}: 0 <= low < high <= Nyquist여야 합니다")
    return lo, hi


def _raw_quality(data, metadata: dict) -> dict:
    """있는 PCM/telemetry만 검사한다. 없는 자료를 무클립·무고장으로 바꾸지 않는다."""
    arrays, quality = {}, {}
    for key, kind in (("input_raw_int32", "i"), ("preflight_raw_int32", "i"),
                      ("output_pcm_int16", "i"), ("output", "f")):
        if key not in data.files:
            quality[key] = None
            continue
        values = np.asarray(data[key])
        expected_dtype = {"input_raw_int32": np.int32, "preflight_raw_int32": np.int32,
                          "output_pcm_int16": np.int16}.get(key)
        if (values.ndim != 2 or values.shape[1] != 2 or values.dtype.kind != kind
                or not np.isfinite(values).all()
                or (expected_dtype is not None and values.dtype != expected_dtype)):
            raise ValueError(f"{key}: 원시 측정 dtype/[samples,2] 형식이 잘못됐습니다")
        arrays[key] = values
        normalized = values.astype(np.float64)
        if key.endswith("int32"):
            normalized /= 2 ** 31
            threshold = 0.99
        elif key.endswith("int16"):
            normalized /= 32767
            threshold = 1.0
        else:
            threshold = 1.0
        quality[key] = {
            "samples": int(values.shape[0]), "clip_threshold_abs": threshold,
            "channels": [{"channel": i,
                          "peak": float(np.max(np.abs(normalized[:, i]))) if len(values) else None,
                          "clip_count": int(np.count_nonzero(np.abs(normalized[:, i]) >= threshold)),
                          "clip_fraction": float(np.mean(np.abs(normalized[:, i]) >= threshold)) if len(values) else None}
                         for i in range(2)],
        }
    full_lengths = {len(values) for key, values in arrays.items() if key != "preflight_raw_int32"}
    for key in ("err", "ref"):
        if key in data.files:
            values = np.asarray(data[key])
            if values.ndim != 1 or values.dtype.kind != "f" or not np.isfinite(values).all():
                raise ValueError(f"{key}: 유한한 실수 1차원 배열이 필요합니다")
            full_lengths.add(len(values))
            if "input_raw_int32" in arrays:
                channel = 0 if key == "err" else 1
                expected = arrays["input_raw_int32"][:, channel].astype(np.float32) * np.float32(1 / 2 ** 31)
                if not np.array_equal(values, expected):
                    raise ValueError(f"{key}와 input_raw_int32가 일치하지 않습니다")
    if len(full_lengths) > 1:
        raise ValueError("원시 출력/입력 배열 길이가 다릅니다")
    if "output" in arrays and "output_pcm_int16" in arrays:
        expected = np.rint(np.clip(arrays["output"].astype(np.float32), -1, 1) * 32767).astype(np.int16)
        if not np.array_equal(expected, arrays["output_pcm_int16"]):
            raise ValueError("output과 output_pcm_int16이 일치하지 않습니다")
    telemetry = metadata.get("telemetry")
    if telemetry is not None and not isinstance(telemetry, dict):
        raise ValueError("telemetry는 object여야 합니다")
    counts = ("xrun_count", "unexpected_status_count", "callback_count", "captured_frames")
    health = {key: None for key in (*counts, "completed", "callback_error")}
    if telemetry is not None:
        for key in counts:
            if key in telemetry:
                health[key] = _integer(telemetry[key], f"telemetry.{key}")
        if "completed" in telemetry:
            if not isinstance(telemetry["completed"], bool):
                raise ValueError("telemetry.completed는 bool이어야 합니다")
            health["completed"] = telemetry["completed"]
        if "callback_error" in telemetry:
            error = telemetry["callback_error"]
            if error is not None and not isinstance(error, str):
                raise ValueError("telemetry.callback_error는 문자열 또는 null이어야 합니다")
            health["callback_error"] = error
    quality["telemetry"] = health
    captured = health["captured_frames"]
    quality["captured_frames_matches_input"] = (
        captured == len(arrays["input_raw_int32"])
        if captured is not None and "input_raw_int32" in arrays else None
    )
    # cancel/ch1 메타와 대조하는 것은 저장된 샘플뿐이다. 실제 배선·음향 경로는 추정하지 않는다.
    quality["inactive_output_channel_nonzero_samples"] = {
        key: int(np.count_nonzero(arrays[key][:, 0])) if key in arrays else None
        for key in ("output", "output_pcm_int16")
    }
    quality["input_snr"] = None
    quality["clock_stability"] = None
    quality["preflight_claim"] = metadata.get("preflight")
    quality["original_result_claim"] = metadata.get("result")
    return quality


def _band_rows(transfer: np.ndarray, frequency: np.ndarray, bands: list, excitation: tuple,
               mode: str) -> tuple[list, list, dict]:
    powers = np.abs(transfer) ** 2
    numeric = np.any(powers > NUMERICAL_POWER_FLOOR, axis=0)
    consistency = np.full(frequency.shape, np.nan)
    consistency[numeric] = np.clip(
        np.abs(np.mean(transfer[:, numeric], axis=0)) ** 2 / np.mean(powers[:, numeric], axis=0), 0, 1
    )
    excited = (frequency >= excitation[0]) & (frequency <= excitation[1])
    rows, pairs = [], []
    for name, low, high, inclusive_high in bands:
        mask = (frequency >= low) & ((frequency <= high) if inclusive_high else (frequency < high))
        count = int(mask.sum())
        covered = excitation[0] <= low and high <= excitation[1]
        available = bool(covered and count > 0)
        complete = bool(available and np.all(numeric[mask]))
        scores, gain_deltas = [], []
        for left in range(transfer.shape[0]):
            for right in range(left + 1, transfer.shape[0]):
                a, b = transfer[left, mask], transfer[right, mask]
                norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
                supported = available and norm_a ** 2 > NUMERICAL_POWER_FLOOR and norm_b ** 2 > NUMERICAL_POWER_FLOOR
                score = float(np.clip(np.vdot(a / norm_a, b / norm_b).real, -1, 1)) if supported else None
                gain = float(20 * (np.log10(norm_b) - np.log10(norm_a))) if supported else None
                scores.append(score)
                gain_deltas.append(gain)
                pairs.append({"mode": mode, "band": name, "left_repeat": left, "right_repeat": right,
                              "phase_sensitive_cosine": score, "right_over_left_magnitude_db": gain})
        all_pairs = all(v is not None for v in scores)
        rows.append({
            "mode": mode, "band": name, "low_hz": low, "high_hz": high,
            "high_edge_inclusive": inclusive_high, "fft_bin_count": count,
            "excitation_covers_band": covered,
            "support_status": "outside_excitation" if not covered else "empty_fft_band" if count == 0 else "within_excitation",
            "power_floor_bin_count": int(np.count_nonzero(~numeric[mask])) if available else None,
            "coherence_min": float(np.min(consistency[mask])) if complete else None,
            "coherence_p10": float(np.percentile(consistency[mask], 10)) if complete else None,
            "coherence_median": float(np.median(consistency[mask])) if complete else None,
            "fraction_at_reference_0_9": float(np.mean(consistency[mask] >= REFERENCE_CONSISTENCY)) if complete else None,
            "worst_pair_phase_sensitive_cosine": min(scores) if available and all_pairs else None,
            "max_pair_magnitude_difference_db": max(abs(v) for v in gain_deltas) if available and all_pairs else None,
        })
    spectra = {"frequencies_hz": frequency.tolist(),
               "coherence": [float(v) if valid else None for v, valid in zip(consistency, excited & numeric)],
               "within_excitation": excited.tolist()}
    return rows, pairs, spectra


def analyze_path_bands(raw_npz: str | Path, band_hz=(800.0, 1600.0)) -> dict:
    """ESS raw_measurement.npz를 읽기만 한다. 누락 delay는 추정해서 채우지 않는다."""
    path = Path(raw_npz).expanduser().resolve()
    digest = _hash(path)
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("raw-npz는 NPZ여야 합니다")
    with archive as data:
        if not {"repeat_irs", "metadata_json"} <= set(data.files):
            raise ValueError("repeat_irs와 metadata_json이 필요합니다")
        text = np.asarray(data["metadata_json"])
        if text.ndim != 0 or text.dtype.kind != "U":
            raise ValueError("metadata_json은 문자열 scalar여야 합니다")
        metadata = json.loads(str(text))
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json은 object여야 합니다")
        json.dumps(metadata, allow_nan=False)
        if (not isinstance(metadata.get("schema_version"), int)
                or isinstance(metadata.get("schema_version"), bool) or metadata["schema_version"] != 1):
            raise ValueError("지원하는 raw schema_version은 1입니다")
        if metadata.get("measurement_kind") != "wideband_path_raw_diagnostic":
            raise ValueError("ESS wideband_path_raw_diagnostic 자료만 지원합니다")
        cfg, result = metadata.get("configuration"), metadata.get("result")
        if not isinstance(cfg, dict) or not isinstance(result, dict):
            raise ValueError("configuration과 result object가 필요합니다")
        if cfg.get("output_channel") != "cancel" or _integer(cfg.get("output_channel_index"), "output_channel_index") != 1:
            raise ValueError("S 진단에는 cancel/ch1 ESS 자료가 필요합니다")
        fs = _integer(cfg.get("sample_rate"), "sample_rate", 1)
        repeats = _integer(cfg.get("repeats"), "repeats", 3)
        length = _integer(cfg.get("fir_length"), "fir_length", 2)
        target = _band(band_hz, fs, "band")
        excitation = _band(cfg.get("band_hz"), fs, "configuration.band_hz")
        irs = np.asarray(data["repeat_irs"])
        if irs.ndim != 2 or irs.shape[0] != repeats or irs.dtype.kind not in "fiu" or not np.isfinite(irs).all():
            raise ValueError("repeat_irs는 설정의 모든 반복과 일치하는 유한 [repeats,samples] 배열이어야 합니다")
        raw_delays = result.get("repeat_delay_samples")
        if not isinstance(raw_delays, list) or len(raw_delays) != repeats:
            raise ValueError("정렬 불가: 모든 반복의 repeat_delay_samples가 필요합니다")
        delays = [_integer(v, "repeat_delay_samples") for v in raw_delays]
        if any(delay + length > irs.shape[1] for delay in delays):
            raise ValueError("정렬 compact FIR을 추출할 길이가 부족합니다")
        compact = np.stack([irs[i, delay:delay + length] for i, delay in enumerate(delays)]).astype(np.float64)
        if np.max(np.abs(compact)) > np.sqrt(np.finfo(np.float64).max) / (4 * length * repeats):
            raise ValueError("repeat_irs가 주파수 파워 계산의 수치 범위를 벗어납니다")
        quality = _raw_quality(data, metadata)
    nfft = 1 << (length - 1).bit_length()
    frequency = np.fft.rfftfreq(nfft, 1 / fs)
    aligned = np.fft.rfft(compact, n=nfft, axis=1)
    fixed = aligned * np.exp(-2j * np.pi * np.asarray(delays)[:, None] * frequency[None, :] / fs)
    bands = [("target", *target, True)]
    for name, low, high in (("target_below_1khz", target[0], min(target[1], 1000.0)),
                            ("target_at_or_above_1khz", max(target[0], 1000.0), target[1])):
        if low < high:
            bands.append((name, low, high, high != 1000.0 or low >= 1000.0))
    rows, pairs, spectra = [], [], {}
    for mode, transfer in (("fixed_origin", fixed), ("delay_aligned_compact", aligned)):
        mode_rows, mode_pairs, spectrum = _band_rows(transfer, frequency, bands, excitation, mode)
        rows.extend(mode_rows)
        pairs.extend(mode_pairs)
        spectra[mode] = spectrum
    issues = ["input_snr_unknown", "clock_stability_unknown", "repeat_ir_derivation_not_recomputed", "no_secondary_promotion"]
    for key in ("input_raw_int32", "preflight_raw_int32", "output", "output_pcm_int16"):
        item = quality[key]
        if item is None or item["samples"] == 0:
            issues.append(f"{key}_unknown")
        elif any(channel["clip_count"] for channel in item["channels"]):
            issues.append(f"{key}_clipping_observed")
    health = quality["telemetry"]
    if any(health[key] is None for key in ("xrun_count", "unexpected_status_count", "completed")):
        issues.append("runtime_health_incomplete")
    if health["xrun_count"] or health["unexpected_status_count"] or health["callback_error"] or health["completed"] is False:
        issues.append("runtime_faults_reported")
    if quality["captured_frames_matches_input"] is False:
        issues.append("captured_length_mismatch")
    if any(value for value in quality["inactive_output_channel_nonzero_samples"].values()):
        issues.append("unexpected_output_channel_samples")
    if any(row["support_status"] != "within_excitation" for row in rows):
        issues.append("requested_band_not_fully_supported")
    if _hash(path) != digest:
        raise ValueError("분석 중 원시 측정 파일이 변경됐습니다")
    return {
        "schema": "deep_anc.path_band_diagnostics.v1", "diagnostic_only": True,
        "performance_claim_allowed": False, "promote_secondary_allowed": False,
        "source": {"path": str(path), "sha256": digest},
        "configuration": cfg, "target_band_hz": list(target), "excitation_band_hz": list(excitation),
        "repeats_used": list(range(repeats)), "repeats_dropped": [],
        "repeat_delay_samples": delays, "delay_spread_samples": max(delays) - min(delays),
        "delay_spread_ms": 1000 * (max(delays) - min(delays)) / fs,
        "method": {
            "fft_size": nfft, "compact_fir_length": length, "fft_bin_spacing_hz": fs / nfft,
            "fixed_origin": "FFT(same_compact_IR) * exp(-j*2*pi*f*stored_repeat_delay/fs)",
            "aligned": "FFT(same_compact_IR); 저장 지연만 제거, 추가 정렬/warp/반복 선별 없음",
            "coherence": "abs(mean(H_repeat))^2 / mean(abs(H_repeat)^2)",
            "pair_similarity": "real(vdot(H_left,H_right))/(norm(H_left)*norm(H_right)); 극성 민감",
            "pair_magnitude_db": "20*log10(norm(H_right)/norm(H_left))",
            "numerical_power_floor": NUMERICAL_POWER_FLOOR,
            "reference_consistency": REFERENCE_CONSISTENCY,
            "interpretation": "0.9는 참고선이며 성능/신뢰대역 게이트가 아니다. FFT zero-padding은 실제 분해능을 늘리지 않는다.",
        },
        "quality": quality, "issues": issues, "metrics": rows, "pairs": pairs, "spectra": spectra,
    }
