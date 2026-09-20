"""16 kHz 공통 고역 지표. 미래 정렬·파형 정규화·플랜트 재적용을 하지 않는다."""

from __future__ import annotations

import hashlib
from numbers import Real

import numpy as np


BANDS = ("full", "low", "high", "priority", "tail")
PHASES = ("onset", "transition", "steady")


def _real(value, name, *, minimum=0.0, strict=True):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or (value <= minimum if strict else value < minimum)):
        raise ValueError(f"{name}: 유한 실수 범위를 확인하십시오")
    return float(value)


def _int(value, name, minimum=0, maximum=1_048_576):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not minimum <= value <= maximum):
        raise ValueError(f"{name}: 정수 범위를 확인하십시오")
    return int(value)


def _wave(value, name):
    array = np.asarray(value)
    if (array.ndim != 1 or not 1 <= array.size <= 1_048_576 or array.dtype.kind not in "fiu"
            or not np.isfinite(array).all() or np.max(np.abs(array.astype(np.float64))) > 1e6):
        raise ValueError(f"{name}: 유한 실수 1차원 파형이 필요합니다")
    return np.asarray(array, dtype=np.float64)


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: 비어 있지 않은 문자열이 필요합니다")
    return value


def _rate(sample_rate):
    if _int(sample_rate, "sample_rate", 1) != 16000:
        raise ValueError("이 비교의 sample_rate는 16000이어야 합니다; 리샘플링하지 않습니다")


def waveform_sha256(value):
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def band_powers(values, *, sample_rate):
    """직사각 창/단측 FFT Parseval 파워. 1 kHz는 high에, 1.6 kHz는 tail에 속한다."""
    _rate(sample_rate)
    values = _wave(values, "values")
    spectrum = np.abs(np.fft.rfft(values)) ** 2 / values.size ** 2
    weights = np.full(spectrum.size, 2.)
    weights[0] = 1.
    if values.size % 2 == 0:
        weights[-1] = 1.
    spectrum *= weights
    frequency = np.fft.rfftfreq(values.size, 1. / sample_rate)
    return {
        "full": float(np.mean(values * values)),
        "low": float(spectrum[frequency < 1000].sum()),
        "high": float(spectrum[frequency >= 1000].sum()),
        "priority": float(spectrum[(frequency >= 1000) & (frequency < 1600)].sum()),
        "tail": float(spectrum[frequency >= 1600].sum()),
    }


def validate_regions(regions, samples):
    """구간은 [start,stop), phase 간에도 겹침 금지. 빈 phase는 미평가로 남긴다."""
    if not isinstance(regions, dict) or set(regions) != set(PHASES):
        raise ValueError("regions에는 onset/transition/steady 세 키가 필요합니다")
    normalized, occupied = {}, []
    for name in PHASES:
        intervals = regions[name]
        if not isinstance(intervals, (list, tuple)):
            raise ValueError("각 phase는 [start,stop] 구간 목록이어야 합니다")
        normalized[name] = []
        for interval in intervals:
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise ValueError("각 구간은 start/stop 두 정수여야 합니다")
            start = _int(interval[0], "start", 0, samples)
            stop = _int(interval[1], "stop", 1, samples)
            if start >= stop:
                raise ValueError("구간은 start < stop이어야 합니다")
            normalized[name].append([start, stop])
            occupied.append((start, stop))
        normalized[name].sort()
    ordered = sorted(occupied)
    if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("onset/transition/steady 구간은 서로 겹칠 수 없습니다")
    return normalized


def _attenuation(baseline, residual, floor):
    if baseline is None or residual is None or min(baseline, residual) <= floor:
        return None
    return float(10. * (np.log10(baseline) - np.log10(residual)))


def evaluate_trace(
    disturbance, residual, *, control, raw_control, regions, session_id, source_id,
    source_kind, method, sample_rate, control_limit, power_floor,
    comparison_context_sha256=None,
):
    """공통 구간의 파워·감쇠·실패를 반환한다. 여러 구간을 붙여 FFT하지 않는다.

    각 interval을 먼저 FFT하고 phase 안에서 샘플 수로 가중한 파워를 합친다.
    full은 원래 전체 파형의 별도 FFT다. floor는 수치 기준이며 물리 noise floor가 아니다.
    실제 제어가 한도를 넘더라도 숨겨 clip하거나 삭제하지 않고 별도 횟수로 남긴다.
    """
    _rate(sample_rate)
    if comparison_context_sha256 is not None and (
        not isinstance(comparison_context_sha256, str) or len(comparison_context_sha256) != 64
        or any(v not in "0123456789abcdef" for v in comparison_context_sha256)
    ):
        raise ValueError("comparison_context_sha256는 소문자 SHA256 또는 None이어야 합니다")
    limit, floor = _real(control_limit, "control_limit"), _real(power_floor, "power_floor")
    ids = {name: _identifier(value, name) for name, value in (
        ("session_id", session_id), ("source_id", source_id), ("source_kind", source_kind), ("method", method))}
    d, e, u, raw = (_wave(value, name) for value, name in (
        (disturbance, "disturbance"), (residual, "residual"), (control, "control"), (raw_control, "raw_control")))
    if not d.shape == e.shape == u.shape == raw.shape:
        raise ValueError("d/e/control/raw_control 길이는 같아야 합니다")
    regions = validate_regions(regions, d.size)
    rows, interval_rows = [], []
    for region, intervals in {"full": [[0, d.size]], **regions}.items():
        samples = sum(stop - start for start, stop in intervals)
        baseline_energy, residual_energy = dict.fromkeys(BANDS, 0.), dict.fromkeys(BANDS, 0.)
        limited = post_limit = 0
        for start, stop in intervals:
            baseline = band_powers(d[start:stop], sample_rate=sample_rate)
            errors = band_powers(e[start:stop], sample_rate=sample_rate)
            count = stop - start
            limited += int(np.count_nonzero(np.abs(raw[start:stop]) > limit))
            post_limit += int(np.count_nonzero(np.abs(u[start:stop]) > limit))
            for band in BANDS:
                baseline_energy[band] += count * baseline[band]
                residual_energy[band] += count * errors[band]
                interval_rows.append({"region": region, "band": band, "start": start, "stop": stop,
                                      "baseline_power": baseline[band], "residual_power": errors[band],
                                      "attenuation_unavailable_reason": "energy_floor_limited"
                                      if min(baseline[band], errors[band]) <= floor else None,
                                      "attenuation_db": _attenuation(baseline[band], errors[band], floor)})
        for band in BANDS:
            baseline = baseline_energy[band] / samples if samples else None
            error = residual_energy[band] / samples if samples else None
            rows.append({"region": region, "band": band, "samples": samples,
                         "baseline_power": baseline, "residual_power": error,
                         "attenuation_db": _attenuation(baseline, error, floor),
                         "attenuation_unavailable_reason": ("region_not_evaluated" if not samples else
                             "energy_floor_limited" if min(baseline, error) <= floor else None),
                         "emergent_error_energy": bool(samples and baseline <= floor < error),
                         "amplification": bool(samples and error > max(baseline, floor)),
                         "below_floor": bool(samples and min(baseline, error) <= floor),
                         "limited_samples": limited, "post_limit_exceeded_samples": post_limit,
                         "limited_fraction": limited / samples if samples else None})
    return {"schema": "high_frequency_trace.v1", **ids, "sample_rate": int(sample_rate),
            "control_limit": limit, "power_floor": floor, "samples": int(d.size), "regions": regions,
            "disturbance_sha256_f64le": waveform_sha256(d), "residual_sha256_f64le": waveform_sha256(e),
            "control_sha256_f64le": waveform_sha256(u), "raw_control_sha256_f64le": waveform_sha256(raw),
            "comparison_context_sha256": comparison_context_sha256,
            "fairness_unverified": comparison_context_sha256 is None,
            "metrics": rows, "interval_metrics": interval_rows,
            "claims": {"physical_performance_claim_allowed": False, "superiority_proven": False},
            "convention": "rectangular one-sided FFT Parseval; floor is numerical, not measured noise floor"}


def metric_row(record, region, band):
    if region not in ("full", *PHASES) or band not in BANDS:
        raise ValueError("알 수 없는 region/band")
    found = [row for row in record["metrics"] if row["region"] == region and row["band"] == band]
    if len(found) != 1:
        raise ValueError("region/band 지표가 없거나 중복입니다")
    return found[0]


def paired_session_comparison(
    records, *, candidate_method, baseline_method, region, band,
    confidence, bootstrap_replicates, seed,
):
    """동일 d/구간의 대응 결과를 세션 에너지로 합친 뒤 세션 단위 bootstrap한다.

    세션 내부 crop은 독립 표본으로 세지 않는다. missing pair/미평가/수치 floor 세션도
    남기며, 하나라도 불완전하면 전체 평균/CI는 null이다. 표본 세션 1개의 CI도 null이다.
    CI가 양수여도 실측·독립성·실용차 기준을 자동 인증하거나 우위를 선언하지 않는다.
    """
    _identifier(candidate_method, "candidate_method")
    _identifier(baseline_method, "baseline_method")
    if candidate_method == baseline_method:
        raise ValueError("서로 다른 두 방법이 필요합니다")
    if region not in ("full", *PHASES) or band not in BANDS:
        raise ValueError("알 수 없는 region/band")
    confidence = _real(confidence, "confidence")
    if confidence >= 1:
        raise ValueError("confidence는 1 미만이어야 합니다")
    replicates = _int(bootstrap_replicates, "bootstrap_replicates", 1, 100000)
    seed = _int(seed, "seed", 0, 2**32 - 1)
    selected = {}
    for record in records:
        if record["method"] not in (candidate_method, baseline_method):
            continue
        key = (_identifier(record["session_id"], "session_id"), _identifier(record["source_id"], "source_id"))
        methods = selected.setdefault(key, {})
        if record["method"] in methods:
            raise ValueError("동일 세션/원천/방법의 중복 결과")
        methods[record["method"]] = record
    if not selected:
        raise ValueError("비교할 결과가 없습니다")
    sessions, pairs = {}, []
    common_floor = None
    fairness_unverified = False
    for (session, source), methods in sorted(selected.items()):
        aggregate = sessions.setdefault(session, {"session_id": session, "samples": 0,
            "baseline_energy": 0., "candidate_error_energy": 0., "baseline_error_energy": 0.,
            "source_count": 0, "complete": True, "candidate_limited_samples": 0,
            "baseline_limited_samples": 0})
        aggregate["source_count"] += 1
        if len(methods) != 2:
            aggregate["complete"] = False
            pairs.append({"session_id": session, "source_id": source, "status": "missing_method"})
            continue
        candidate, baseline = methods[candidate_method], methods[baseline_method]
        if candidate.get("comparison_context_sha256") != baseline.get("comparison_context_sha256"):
            raise ValueError("대응 비교의 P/S/지연/runtime context가 다릅니다")
        fairness_unverified |= candidate.get("comparison_context_sha256") is None
        for name in ("sample_rate", "control_limit", "power_floor", "samples", "regions",
                     "disturbance_sha256_f64le", "source_kind"):
            if candidate[name] != baseline[name]:
                raise ValueError(f"대응 비교의 {name} 계약이 다릅니다")
        if common_floor is not None and common_floor != candidate["power_floor"]:
            raise ValueError("비교 세션들의 power_floor가 다릅니다")
        common_floor = candidate["power_floor"]
        a, b = metric_row(candidate, region, band), metric_row(baseline, region, band)
        if a["samples"] != b["samples"] or a["baseline_power"] != b["baseline_power"]:
            raise ValueError("대응 비교의 disturbance/구간 파워가 다릅니다")
        n = a["samples"]
        pairs.append({"session_id": session, "source_id": source,
                      "status": "paired" if n else "region_not_evaluated",
                      "candidate_attenuation_db": a["attenuation_db"],
                      "baseline_attenuation_db": b["attenuation_db"]})
        if not n:
            aggregate["complete"] = False
            continue
        aggregate["samples"] += n
        aggregate["baseline_energy"] += a["baseline_power"] * n
        aggregate["candidate_error_energy"] += a["residual_power"] * n
        aggregate["baseline_error_energy"] += b["residual_power"] * n
        aggregate["candidate_limited_samples"] += a["limited_samples"]
        aggregate["baseline_limited_samples"] += b["limited_samples"]
    values = []
    for session in sessions.values():
        n = session["samples"]
        a = b = delta = None
        if n:
            d = session["baseline_energy"] / n
            a = _attenuation(d, session["candidate_error_energy"] / n, common_floor)
            b = _attenuation(d, session["baseline_error_energy"] / n, common_floor)
            if a is not None and b is not None and session["complete"]:
                delta = a - b
                values.append(delta)
        session.update(candidate_attenuation_db=a, baseline_attenuation_db=b, delta_db=delta)
        session["complete"] = bool(session["complete"] and delta is not None)
    complete = all(row["complete"] for row in sessions.values())
    mean = float(np.mean(values)) if complete else None
    interval = None
    if complete and len(values) >= 2:
        rng = np.random.default_rng(seed)
        values = np.asarray(values)
        boot = np.empty(replicates)
        for start in range(0, replicates, 256):
            stop = min(replicates, start + 256)
            boot[start:stop] = values[rng.integers(0, values.size, (stop - start, values.size))].mean(axis=1)
        alpha = (1. - confidence) / 2.
        interval = [float(v) for v in np.quantile(boot, [alpha, 1. - alpha])]
    return {"schema": "high_frequency_paired_sessions.v1", "candidate_method": candidate_method,
            "baseline_method": baseline_method, "region": region, "band": band,
            "paired_complete": complete, "independent_unit": "caller_declared_session_not_crop",
            "session_count": len(sessions), "mean_delta_db": mean, "bootstrap_ci_db": interval,
            "confidence": confidence, "bootstrap_replicates": replicates, "seed": seed,
            "sessions": list(sessions.values()), "pairs": pairs,
            "fairness_unverified": bool(fairness_unverified or not complete),
            "fairness_scope": "declared d/regions/limit/context equality only; physical provenance not certified",
            "physical_performance_claim_allowed": False, "superiority_proven": False}
