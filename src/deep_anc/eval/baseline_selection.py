"""명시 후보만 validation에서 비교하고 선택을 동결하는 비딥러닝 기준선 도구.

import만으로 실행하지 않는다. 실제 탐색은 별도 실행 지시가 필요하다. train/test 파형을
읽지 않으며 split 메타를 먼저 모두 검사한다. 결과 실패 행과 제한 기록을 삭제하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .classical_anc import run_classical_anc
from .high_frequency_metrics import (
    _attenuation, _identifier, _int, _rate, _real, _wave, evaluate_trace,
    metric_row, validate_regions, waveform_sha256,
)


_CANDIDATE_KEYS = {"id", "algorithm", "mu", "control_length", "block_samples",
                   "normalization_epsilon", "weight_norm_limit"}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _new_output_path(out):
    path = Path(out).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {path}")
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError("결과 경로의 부모 symlink는 허용하지 않습니다")
    return path


def _candidate(value):
    if not isinstance(value, dict) or set(value) != _CANDIDATE_KEYS:
        raise ValueError(f"각 candidate에는 정확히 {sorted(_CANDIDATE_KEYS)}가 필요합니다")
    result = {"id": _identifier(value["id"], "candidate id"), "algorithm": value["algorithm"]}
    if result["algorithm"] not in ("fxlms", "fxnlms"):
        raise ValueError("candidate algorithm은 fxlms/fxnlms여야 합니다")
    for name, maximum in (("control_length", 4096), ("block_samples", 1048576)):
        result[name] = _int(value[name], name, 1, maximum)
    for name in ("mu", "normalization_epsilon", "weight_norm_limit"):
        result[name] = _real(value[name], name)
        if result[name] > 1e6:
            raise ValueError(f"{name} 수치 범위 초과")
    return result


def _score(records, *, power_floor):
    sessions = {}
    for record in sorted(records, key=lambda r: (r["session_id"], r["source_id"])):
        row = metric_row(record, "steady", "high")
        entry = sessions.setdefault(record["session_id"], {"session_id": record["session_id"],
            "samples": 0, "disturbance_energy": 0., "residual_energy": 0., "complete": True})
        if not row["samples"]:
            entry["complete"] = False
            continue
        entry["samples"] += row["samples"]
        entry["disturbance_energy"] += row["baseline_power"] * row["samples"]
        entry["residual_energy"] += row["residual_power"] * row["samples"]
    for row in sessions.values():
        count = row["samples"]
        row["attenuation_db"] = (_attenuation(row["disturbance_energy"] / count,
            row["residual_energy"] / count, power_floor) if count and row["complete"] else None)
    complete = bool(sessions) and all(row["attenuation_db"] is not None for row in sessions.values())
    score = float(np.mean([row["attenuation_db"] for row in sessions.values()])) if complete else None
    return score, list(sessions.values())


def select_validation_baselines(
    cases, candidates, *, secondary, secondary_estimate, additional_delay_samples,
    estimate_delay_samples, sample_rate, control_limit, power_floor,
    max_limited_fraction, max_low_amplification_db, max_case_runs, out,
    runtime_conditions, runner=None,
):
    """온라인 적응 기준선의 validation 탐색을 수행하고 신규 out에 전 행을 보존한다.

    후보의 id/algorithm/mu/control_length/block_samples/normalization_epsilon/
    weight_norm_limit은 모두 필수다. 공유 실제/추정 S·지연·출력 한도도 필수이며 추측하지
    않는다. runtime_conditions는 호출자가 선언한 공통 timing/입력 정보의 JSON dict다.

    case 필수 필드: split='validation', session_id, source_id, source_kind,
    reference, disturbance, regions. 어떤 배열도 보기 전에 모든 split을 검사한다.
    후보×case 예산을 넘으면 전부 거부하며 일부 자료로 승자를 고르지 않는다.

    목적함수는 steady high[1k,Nyquist]의 세션 에너지 감쇠를 세션 간 동일 가중 평균한다.
    전체 제한 전 초과 비율과 각 명시 interval의 저역 악화를 먼저 검사하고, 실제 출력
    한도 위반/새 저역 에너지/실패/계산 불가가 있으면 후보를 제외한다. 동점은 정확한
    float 점수 동률일 때 candidate id 사전순이다. algorithm별로 하나씩 선택한다.
    후보·실패·세션 결과는 보존하며 test 최적화/최종 우위 검증은 수행하지 않는다.
    """
    if not isinstance(cases, (list, tuple)) or not cases:
        raise ValueError("비어 있지 않은 validation case 목록이 필요합니다")
    # metadata pass: 아래 검사가 끝나기 전에 reference/disturbance에 접근하지 않는다.
    for case in cases:
        if not isinstance(case, dict) or case.get("split") != "validation":
            raise ValueError("탐색은 validation만 허용합니다; train/test 접근 금지")
    destination = _new_output_path(out)
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError("명시적인 비어 있지 않은 candidate 목록이 필요합니다")
    grid = [_candidate(item) for item in candidates]
    if len({row["id"] for row in grid}) != len(grid):
        raise ValueError("candidate id 중복")
    budget = _int(max_case_runs, "max_case_runs", 1, 1000000)
    if len(cases) * len(grid) > budget:
        raise ValueError("후보×case 실행 예산 초과; 일부 자료로 탐색하지 않습니다")
    _rate(sample_rate)
    delay = _int(additional_delay_samples, "additional_delay_samples")
    estimate_delay = _int(estimate_delay_samples, "estimate_delay_samples")
    limit, floor = _real(control_limit, "control_limit"), _real(power_floor, "power_floor")
    clip_threshold = _real(max_limited_fraction, "max_limited_fraction", strict=False)
    low_threshold = _real(max_low_amplification_db, "max_low_amplification_db", strict=False)
    if clip_threshold > 1 or limit > 1e6:
        raise ValueError("출력/clip 제한 범위 오류")
    if not isinstance(runtime_conditions, dict) or not runtime_conditions:
        raise ValueError("공통 runtime_conditions를 명시해야 합니다")
    runtime_conditions = json.loads(_canonical(runtime_conditions))
    s, estimate = _wave(secondary, "secondary"), _wave(secondary_estimate, "secondary_estimate")
    if max(s.size, estimate.size) > 8192 or not np.any(s) or not np.any(estimate):
        raise ValueError("S/S_hat 길이 또는 zero model 오류")
    prepared, identifiers = [], set()
    for case in cases:
        meta = {name: _identifier(case[name], name) for name in ("session_id", "source_id", "source_kind")}
        key = (meta["session_id"], meta["source_id"])
        if key in identifiers:
            raise ValueError("세션/원천 case 중복")
        identifiers.add(key)
        x, d = _wave(case["reference"], "reference"), _wave(case["disturbance"], "disturbance")
        if x.shape != d.shape:
            raise ValueError("REF와 disturbance의 길이가 다릅니다")
        regions = validate_regions(case["regions"], x.size)
        prepared.append({**meta, "x": x, "d": d, "regions": regions})
    prepared.sort(key=lambda row: (row["session_id"], row["source_id"]))
    evidence = [{**{k: row[k] for k in ("session_id", "source_id", "source_kind", "regions")},
        "split": "validation", "samples": int(row["x"].size),
        "reference_sha256_f64le": waveform_sha256(row["x"]),
        "disturbance_sha256_f64le": waveform_sha256(row["d"])} for row in prepared]
    context = {"sample_rate": int(sample_rate), "secondary_sha256_f64le": waveform_sha256(s),
        "secondary_taps": int(s.size), "estimate_sha256_f64le": waveform_sha256(estimate),
        "estimate_taps": int(estimate.size), "additional_delay_samples": delay,
        "estimate_delay_samples": estimate_delay, "control_limit": limit,
        "runtime_conditions": runtime_conditions, "primary_representation": "provided_disturbance_per_case"}
    protocol = {"context": context, "power_floor": floor, "max_limited_fraction": clip_threshold,
        "max_low_amplification_db": low_threshold, "max_case_runs": budget,
        "objective": "mean_session_steady_high_attenuation_db",
        "tie_break": "exact_score_tie_then_lexicographic_candidate_id",
        "validation_fingerprint": _digest(evidence), "candidate_grid_sha256": _digest(grid)}
    _new_output_path(destination)  # 입력 검증 중 다른 writer가 만든 경로도 거부한다.
    destination.mkdir(parents=True, exist_ok=False)
    execute = run_classical_anc if runner is None else runner
    case_rows, summaries = [], []
    with (destination / "case_results.jsonl").open("x", encoding="utf-8") as journal:
        for candidate in grid:
            records, failures, violations = [], [], []
            kwargs = {key: value for key, value in candidate.items() if key != "id"}
            for case in prepared:
                row = {"candidate_id": candidate["id"], "session_id": case["session_id"],
                       "source_id": case["source_id"], "status": "ok"}
                try:
                    result = execute(case["x"], case["d"], s, **kwargs,
                        secondary_estimate=estimate, additional_delay_samples=delay,
                        estimate_delay_samples=estimate_delay, control_limit=limit)
                    record = evaluate_trace(case["d"], result["residual"], control=result["control"],
                        raw_control=result["raw_control"], regions=case["regions"],
                        session_id=case["session_id"], source_id=case["source_id"], source_kind=case["source_kind"],
                        method=candidate["id"], sample_rate=sample_rate, control_limit=limit, power_floor=floor,
                        comparison_context_sha256=_digest(context))
                    full = metric_row(record, "full", "full")
                    if full["limited_fraction"] > clip_threshold:
                        violations.append({"source_id": case["source_id"], "reason": "clipping_fraction"})
                    if full["post_limit_exceeded_samples"]:
                        violations.append({"source_id": case["source_id"], "reason": "actual_output_limit"})
                    for low in record["interval_metrics"]:
                        if low["band"] != "low":
                            continue
                        emergent = low["baseline_power"] <= floor < low["residual_power"]
                        if emergent or (low["attenuation_db"] is not None and low["attenuation_db"] < -low_threshold):
                            violations.append({"source_id": case["source_id"], "reason": "low_band_amplification",
                                               "region": low["region"], "start": low["start"], "stop": low["stop"]})
                    row["trace"] = record
                    row["runner_settings"] = result.get("settings", {})
                    row["diagnostics"] = {key: result.get(key) for key in (
                        "limited_samples", "adapted_blocks", "clipped_guard_blocks", "weight_limited_blocks", "weight_norm")}
                    _canonical(row)  # 비유한 진단/직렬화 불가 runner 결과도 실패 행으로 보존.
                    records.append(record)
                except Exception as exc:
                    row = {"candidate_id": candidate["id"], "session_id": case["session_id"],
                           "source_id": case["source_id"], "status": "failed",
                           "error_type": type(exc).__name__, "error": str(exc)}
                    failures.append({"source_id": case["source_id"], "error_type": type(exc).__name__})
                case_rows.append(row)
                journal.write(_canonical(row).decode() + "\n")
                journal.flush()
            score, sessions = _score(records, power_floor=floor)
            eligible = not failures and not violations and score is not None and len(records) == len(prepared)
            summaries.append({"candidate": candidate, "eligible": eligible, "score": score,
                "sessions": sessions, "failures": failures, "violations": violations})
    selected = {}
    for algorithm in sorted({item["algorithm"] for item in grid}):
        eligible = [row for row in summaries if row["eligible"] and row["candidate"]["algorithm"] == algorithm]
        chosen = min(eligible, key=lambda row: (-row["score"], row["candidate"]["id"])) if eligible else None
        selected[algorithm] = chosen["candidate"] if chosen else None
    payload = {"schema": "frozen_validation_baseline.v1", "protocol": protocol,
        "selected_by_algorithm": selected, "selection_complete": all(v is not None for v in selected.values()),
        "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__), Path(__file__).with_name("classical_anc.py"),
            Path(__file__).with_name("high_frequency_metrics.py"), Path(__file__).with_name("sfanc_fxnlms.py"))},
        "split_verification": "explicit_case_metadata; undeclared_source_provenance_not_certified",
        "test_accessed": False, "neural_training_executed": False,
        "physical_performance_claim_allowed": False, "superiority_proven": False}
    frozen = {"payload": payload, "sha256": _digest(payload)}
    report = {"schema": "validation_baseline_search.v1", "protocol": protocol, "cases": evidence,
              "candidate_results": summaries, "case_results": case_rows, "frozen_selection": frozen,
              "baseline_validation_tuning_executed": True, "neural_training_executed": False,
              "test_accessed": False, "physical_performance_claim_allowed": False}
    for name, value in (("selection.json", frozen), ("report.json", report)):
        with (destination / name).open("x", encoding="utf-8") as stream:
            stream.write(_canonical(value).decode() + "\n")
    return report


def load_frozen_selection(path):
    """JSON 해시·schema 검증만 한다. pickle/모델/음원/적응을 실행하지 않는다."""
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(artifact, dict) or set(artifact) != {"payload", "sha256"}
            or not isinstance(artifact["payload"], dict)
            or _digest(artifact["payload"]) != artifact["sha256"]
            or artifact["payload"].get("schema") != "frozen_validation_baseline.v1"):
        raise ValueError("동결 selection schema/SHA 불일치")
    return artifact
