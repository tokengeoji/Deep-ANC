#!/usr/bin/env python3
"""Fullband causal P/S raw의 offline-only FIR/support/tail 분석기.

오디오 모듈을 import하지 않는다. 입력 response는 callback timestamp와 clock witness를
통과해 DAC sample q에 이미 resample된 것이어야 한다. raw ADC index를 그대로 넣거나
compact partial-band FIR을 넣으면 승격하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.dsp.control_band_contract import (  # noqa: E402
    BROADBAND_MEASURED_INTERPOLATION_MAX_RELATIVE_ERROR,
    BROADBAND_MEASURED_INTERPOLATION_MIN_AGREEMENT,
    BROADBAND_POINT_CONTROL_SUBBANDS_HZ,
)
from deep_anc.dsp.fullband_causal_probe import (  # noqa: E402
    CANDIDATE_SUPPORT_SAMPLES,
    FIT_ANALYSIS_REPEATS,
    FIT_PERIOD_SAMPLES,
    HOLDOUT_ANALYSIS_REPEATS,
    HOLDOUT_PERIOD_SAMPLES,
    MAXIMUM_DELAY_SAMPLES,
    MAX_DESIGN_CONDITION,
    SAMPLE_RATE,
    TAIL_SCAN_SAMPLES,
    build_signal_plan,
    off_grid_holdout_bins,
)
from deep_anc.dsp.measurement_level import atomic_publish_noreplace  # noqa: E402


ANALYSIS_SCHEMA = "fullband_causal_ps_analysis_v1"
MIN_VALID_FIT_REPEATS = 12
MIN_VALID_HOLDOUT_REPEATS = 6
MIN_ADJACENT_SCORE = 0.995
MAX_CLOCK_RESIDUAL_SAMPLES_20DB = 0.06755189029558946
MAX_TAIL_L1_RATIO = 0.03
TAIL_BLOCK_SAMPLES = 1_024
MAX_TAIL_DECAY_RATIO = 0.80
TAIL_BOOTSTRAP_RESAMPLES = 2_000
TAIL_BOOTSTRAP_SEED = 20_260_828


def _complex_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    left = np.asarray(reference, dtype=np.complex128).reshape(-1)
    right = np.asarray(estimate, dtype=np.complex128).reshape(-1)
    if left.size == 0 or left.size != right.size:
        raise ValueError("complex metric vector가 비었거나 길이가 다릅니다")
    denominator = float(np.linalg.norm(left))
    estimate_norm = float(np.linalg.norm(right))
    relative = float(np.linalg.norm(right - left) / max(denominator, 1.0e-30))
    agreement = float(
        abs(complex(np.vdot(left, right)))
        / max(denominator * estimate_norm, 1.0e-30)
    )
    return {"relative_error": relative, "complex_agreement": agreement}


def _period_rows(plan: Mapping[str, Any], *, path: str, role: str) -> tuple[int, int, int]:
    kind = f"{path}_{role}"
    rows = [row for row in plan["layout"] if row.get("kind") == kind]
    if len(rows) != 1:
        raise ValueError(f"plan에 {kind} row가 exact 1개가 아닙니다")
    row = rows[0]
    period = int(row["period_samples"])
    warmup = int(row["warmup_repeats"])
    repeats = int(row["analysis_repeats"])
    return int(row["start_frame"]) + warmup * period, period, repeats


def _repeat_stack(
    response: np.ndarray, *, start: int, period: int, repeats: int
) -> np.ndarray:
    signal = np.asarray(response, dtype=np.float64).reshape(-1)
    stop = int(start) + int(period) * int(repeats)
    if start < 0 or stop > signal.size:
        raise ValueError("response에 분석 repeat window가 없습니다")
    stack = signal[start:stop].reshape(int(repeats), int(period))
    if not np.all(np.isfinite(stack)):
        raise ValueError("response repeat에 NaN/Inf가 있습니다")
    return stack


def _repeat_impulses(input_pcm: np.ndarray, response_stack: np.ndarray) -> np.ndarray:
    x = np.asarray(input_pcm, dtype=np.float64).reshape(-1) / 32_767.0
    if response_stack.ndim != 2 or response_stack.shape[1] != x.size:
        raise ValueError("input period/response repeat shape가 다릅니다")
    spectrum = np.fft.rfft(x)
    magnitude = np.abs(spectrum)
    condition = float(np.max(magnitude) / np.min(magnitude))
    if np.any(magnitude <= 0.0) or condition > MAX_DESIGN_CONDITION:
        raise ValueError(f"actual int16 period가 full-rank/condition gate 실패: {condition}")
    return np.stack(
        [np.fft.irfft(np.fft.rfft(row) / spectrum, n=x.size) for row in response_stack]
    )


def _bulk_delay(repeat_impulses: np.ndarray) -> int:
    stack = np.asarray(repeat_impulses, dtype=np.float64)
    if stack.ndim != 2 or stack.shape[0] < MIN_VALID_FIT_REPEATS:
        raise ValueError("bulk onset에는 valid repeat impulse가 12개 이상 필요합니다")
    vector = np.median(stack, axis=0)
    search = np.abs(vector[: MAXIMUM_DELAY_SAMPLES + 1])
    peak_index = int(np.argmax(search))
    peak = float(search[peak_index])
    if peak <= 0.0 or not math.isfinite(peak):
        raise ValueError("0..4800 delay branch에 유효한 impulse peak가 없습니다")
    per_tap_mad = 1.4826 * np.median(
        np.abs(stack[:, : MAXIMUM_DELAY_SAMPLES + 1] - vector[: MAXIMUM_DELAY_SAMPLES + 1]),
        axis=0,
    )
    noise = float(np.median(per_tap_mad))
    threshold = max(8.0 * noise, peak * 1.0e-4, np.finfo(np.float64).eps)
    candidates = np.flatnonzero(search >= threshold)
    if candidates.size == 0:
        raise ValueError("repeat-noise 상한을 넘는 causal onset이 없습니다")
    onset = int(candidates[0])
    if onset > peak_index:
        raise ValueError("causal onset이 bulk peak 뒤에 있습니다")
    return onset


def _tail_receipt(
    repeat_impulses: np.ndarray,
    *,
    delay: int,
    support: int,
    input_peak: float,
    heldout_output_rms: float,
) -> dict[str, Any]:
    stack = np.asarray(repeat_impulses, dtype=np.float64)
    start = int(delay) + int(support)
    stop = int(delay) + TAIL_SCAN_SAMPLES
    if stop > stack.shape[1] or start >= stop:
        raise ValueError("tail scan window가 period 밖이거나 비었습니다")
    # coefficient를 iid로 취급하지 않는다. 한 repeat의 impulse vector 전체가 하나의
    # cluster이며, cluster를 복원추출한 뒤 median operator를 다시 만들어 L1 분포를 얻는다.
    repeat_tail = np.sum(np.abs(stack[:, start:stop]), axis=1)
    repeat_retained = np.sum(np.abs(stack[:, delay:start]), axis=1)
    rng = np.random.default_rng(TAIL_BOOTSTRAP_SEED + int(support))
    bootstrap_tail = np.empty(TAIL_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    bootstrap_retained = np.empty(TAIL_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for draw in range(TAIL_BOOTSTRAP_RESAMPLES):
        selected = rng.integers(0, stack.shape[0], size=stack.shape[0])
        # whole-repeat L1을 resample한다. coefficient별 iid bootstrap보다 보수적이며
        # 한 repeat 내부의 모든 tap 상관을 그대로 cluster 안에 보존한다.
        bootstrap_tail[draw] = float(np.mean(repeat_tail[selected]))
        bootstrap_retained[draw] = float(np.mean(repeat_retained[selected]))
    tail_upper = float(np.quantile(bootstrap_tail, 0.95, method="higher"))
    retained_lower = float(np.quantile(bootstrap_retained, 0.05, method="lower"))
    median = np.median(stack, axis=0)
    floor = max(float(np.sum(np.abs(median[delay:start]))) * 1.0e-10, 1.0e-15)
    block_sums = []
    for block_start in range(stop - 4 * TAIL_BLOCK_SAMPLES, stop, TAIL_BLOCK_SAMPLES):
        block_sums.append(
            float(np.sum(np.abs(median[block_start : block_start + TAIL_BLOCK_SAMPLES])))
        )
    if max(block_sums) <= floor:
        decay_ratio = 0.0
        remainder = 0.0
        decay_valid = True
    else:
        ratios = [
            right / max(left, floor)
            for left, right in zip(block_sums[:-1], block_sums[1:], strict=True)
        ]
        decay_ratio = float(max(ratios))
        decay_valid = bool(
            all(right <= left for left, right in zip(block_sums[:-1], block_sums[1:], strict=True))
            and decay_ratio < MAX_TAIL_DECAY_RATIO
        )
        remainder = (
            float(block_sums[-1] * decay_ratio / (1.0 - decay_ratio))
            if decay_valid
            else math.inf
        )
    upper_with_remainder = tail_upper + remainder
    ratio = upper_with_remainder / max(retained_lower, 1.0e-30)
    induced_ratio = (
        upper_with_remainder * float(input_peak)
        / max(float(heldout_output_rms), 1.0e-30)
    )
    passed = bool(
        decay_valid
        and math.isfinite(ratio)
        and ratio <= MAX_TAIL_L1_RATIO
        and math.isfinite(induced_ratio)
        and induced_ratio <= MAX_TAIL_L1_RATIO
    )
    return {
        "support_samples": int(support),
        "scan_stop_samples_after_bulk": TAIL_SCAN_SAMPLES,
        "tail_l1_upper_95": tail_upper,
        "geometric_remainder_upper": remainder if math.isfinite(remainder) else None,
        "tail_to_retained_l1_ratio_upper": ratio if math.isfinite(ratio) else None,
        "heldout_induced_output_ratio_upper": (
            induced_ratio if math.isfinite(induced_ratio) else None
        ),
        "last_block_l1": block_sums,
        "decay_ratio_upper": decay_ratio,
        "decay_valid": decay_valid,
        "maximum_ratio": MAX_TAIL_L1_RATIO,
        "bootstrap": {
            "unit": "whole_repeat_impulse_vector",
            "resamples": TAIL_BOOTSTRAP_RESAMPLES,
            "seed": TAIL_BOOTSTRAP_SEED + int(support),
            "one_sided_quantile": 0.95,
        },
        "passed": passed,
    }


def _response_for_kernel(input_pcm: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    x = np.asarray(input_pcm, dtype=np.float64).reshape(-1) / 32_767.0
    h = np.asarray(kernel, dtype=np.float64).reshape(-1)
    padded = np.zeros(x.size, dtype=np.float64)
    padded[: h.size] = h
    return np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(padded), n=x.size)


def _band_metrics(
    *, reference: np.ndarray, estimate: np.ndarray, period_samples: int
) -> list[dict[str, Any]]:
    ref = np.fft.rfft(np.asarray(reference, dtype=np.float64))
    est = np.fft.rfft(np.asarray(estimate, dtype=np.float64))
    frequency = np.fft.rfftfreq(int(period_samples), 1.0 / SAMPLE_RATE)
    reports = []
    for index, (low, high) in enumerate(BROADBAND_POINT_CONTROL_SUBBANDS_HZ):
        mask = (frequency >= low) & (
            frequency <= high if index == len(BROADBAND_POINT_CONTROL_SUBBANDS_HZ) - 1 else frequency < high
        )
        metric = _complex_metrics(ref[mask], est[mask])
        reports.append(
            {
                "band_hz": [float(low), float(high)],
                **metric,
                "passed": bool(
                    metric["relative_error"]
                    <= BROADBAND_MEASURED_INTERPOLATION_MAX_RELATIVE_ERROR
                    and metric["complex_agreement"]
                    >= BROADBAND_MEASURED_INTERPOLATION_MIN_AGREEMENT
                ),
            }
        )
    return reports


def _panel_cross_binding(
    *, kernel: np.ndarray, panel: Mapping[str, Any], synthetic_fixture: bool
) -> list[dict[str, Any]]:
    if not synthetic_fixture:
        if panel.get("schema") != "broadband_panel_raw_cross_binding_v1":
            raise ValueError("production panel raw cross-binding schema가 필요합니다")
        for field in ("raw_sha256", "analysis_sha256", "capture_id"):
            value = str(panel.get(field, ""))
            if field.endswith("sha256") and (
                len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"panel {field}가 canonical SHA-256이 아닙니다")
            if field == "capture_id" and not value:
                raise ValueError("panel capture_id가 비었습니다")
        if float(panel.get("minimum_repeat_consistency", 0.0)) < 0.95:
            raise ValueError("panel raw repeat consistency가 0.95 미만입니다")
    frequencies = np.asarray(panel.get("frequencies_hz"), dtype=np.float64).reshape(-1)
    reference = np.asarray(panel.get("transfer"), dtype=np.complex128).reshape(-1)
    if frequencies.size == 0 or frequencies.size != reference.size:
        raise ValueError("7-band panel frequency/transfer binding이 필요합니다")
    index = np.arange(np.asarray(kernel).size, dtype=np.float64)
    estimated = np.exp(
        -2j * np.pi * np.outer(frequencies, index) / SAMPLE_RATE
    ) @ np.asarray(kernel, dtype=np.float64)
    reports = []
    for band_index, (low, high) in enumerate(BROADBAND_POINT_CONTROL_SUBBANDS_HZ):
        mask = (frequencies >= low) & (
            frequencies <= high
            if band_index == len(BROADBAND_POINT_CONTROL_SUBBANDS_HZ) - 1
            else frequencies < high
        )
        if int(mask.sum()) < 4:
            raise ValueError(f"panel band {low:g}–{high:g} Hz tone이 4개 미만입니다")
        metric = _complex_metrics(reference[mask], estimated[mask])
        reports.append(
            {
                "band_hz": [float(low), float(high)],
                "tone_count": int(mask.sum()),
                **metric,
                "passed": bool(
                    metric["relative_error"]
                    <= BROADBAND_MEASURED_INTERPOLATION_MAX_RELATIVE_ERROR
                    and metric["complex_agreement"]
                    >= BROADBAND_MEASURED_INTERPOLATION_MIN_AGREEMENT
                ),
            }
        )
    return reports


def analyse_resampled_capture(
    *,
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    responses_by_path: Mapping[str, np.ndarray],
    clock_receipt: Mapping[str, Any],
    panel_bindings: Mapping[str, Mapping[str, Any]],
    final_noise_floor_receipts: Mapping[str, Mapping[str, Any]],
    synthetic_fixture: bool = False,
) -> dict[str, Any]:
    """DAC-q resampled response에서 P/S causal operator를 fail-closed로 고른다."""

    expected_plan, expected_pcm = build_signal_plan()
    # hardware binding 같은 wrapper field는 허용하되 signal-only core는 exact여야 한다.
    for key in expected_plan:
        if plan.get(key) != expected_plan[key]:
            raise ValueError(f"signal plan core field가 현재 계약과 다릅니다: {key}")
    pcm = np.asarray(submitted_pcm)
    if pcm.dtype != np.int16 or not np.array_equal(pcm, expected_pcm):
        raise ValueError("actual submitted int16 PCM이 exact plan과 다릅니다")
    if clock_receipt.get("schema") != "absolute_dac_q_timewarp_v1":
        raise ValueError("absolute DAC-q clock/time-warp receipt가 필요합니다")
    if not synthetic_fixture:
        for field in ("source_raw_sha256", "timewarp_payload_sha256"):
            value = str(clock_receipt.get(field, ""))
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"production clock receipt {field}가 필요합니다")
    clock_valid = bool(
        int(clock_receipt.get("valid_fit_repeats", -1)) >= MIN_VALID_FIT_REPEATS
        and int(clock_receipt.get("valid_holdout_repeats", -1)) >= MIN_VALID_HOLDOUT_REPEATS
        and float(clock_receipt.get("minimum_adjacent_score", 0.0)) >= MIN_ADJACENT_SCORE
        and float(clock_receipt.get("maximum_residual_samples", math.inf))
        <= MAX_CLOCK_RESIDUAL_SAMPLES_20DB
        and int(clock_receipt.get("sample_slip_count", -1)) == 0
    )
    if not clock_valid:
        raise ValueError("clock/repeat/20dB timing gate가 실패했습니다")

    fit_period = np.asarray(expected_pcm[
        next(row["start_frame"] for row in expected_plan["layout"] if row["kind"] == "primary_fit"):
        next(row["start_frame"] for row in expected_plan["layout"] if row["kind"] == "primary_fit") + FIT_PERIOD_SAMPLES,
        0,
    ])
    holdout_period = np.asarray(expected_pcm[
        next(row["start_frame"] for row in expected_plan["layout"] if row["kind"] == "primary_holdout"):
        next(row["start_frame"] for row in expected_plan["layout"] if row["kind"] == "primary_holdout") + HOLDOUT_PERIOD_SAMPLES,
        0,
    ])
    # row 첫 period는 warmup이지만 입력 period 자체는 동일하다.
    path_receipts: dict[str, Any] = {}
    all_passed = True
    for path in ("primary", "secondary"):
        if (
            path not in responses_by_path
            or path not in panel_bindings
            or path not in final_noise_floor_receipts
        ):
            raise ValueError(f"{path} response/panel/final-noise binding이 필요합니다")
        noise_floor = dict(final_noise_floor_receipts[path])
        noise_floor_passed = bool(
            noise_floor.get("schema") == "final_tail_input_noise_floor_v1"
            and noise_floor.get("valid") is True
            and abs(float(noise_floor.get("last_tail_vs_input_noise_db", math.inf))) <= 1.0
        )
        if not synthetic_fixture:
            source_sha = str(noise_floor.get("source_raw_sha256", ""))
            noise_floor_passed = bool(
                noise_floor_passed
                and len(source_sha) == 64
                and all(char in "0123456789abcdef" for char in source_sha)
                and source_sha == str(clock_receipt.get("source_raw_sha256", ""))
            )
        if not noise_floor_passed:
            raise ValueError(f"{path} final 0.1s tail이 input-only noise floor ±1 dB가 아닙니다")
        response = np.asarray(responses_by_path[path], dtype=np.float64).reshape(-1)
        if response.size != expected_pcm.shape[0]:
            raise ValueError(f"{path} response 길이가 plan과 다릅니다")
        fit_start, fit_size, fit_repeats = _period_rows(expected_plan, path=path, role="fit")
        hold_start, hold_size, hold_repeats = _period_rows(expected_plan, path=path, role="holdout")
        fit_stack = _repeat_stack(response, start=fit_start, period=fit_size, repeats=fit_repeats)
        hold_stack = _repeat_stack(response, start=hold_start, period=hold_size, repeats=hold_repeats)
        impulses = _repeat_impulses(fit_period, fit_stack)
        median_impulse = np.median(impulses, axis=0)
        delay = _bulk_delay(impulses)
        observed_holdout = np.median(hold_stack, axis=0)
        candidates = []
        selected_kernel: np.ndarray | None = None
        selected_support: int | None = None
        for support in CANDIDATE_SUPPORT_SAMPLES:
            kernel = np.zeros(delay + support, dtype=np.float64)
            kernel[delay:] = median_impulse[delay : delay + support]
            prediction = _response_for_kernel(holdout_period, kernel)
            bands = _band_metrics(
                reference=observed_holdout,
                estimate=prediction,
                period_samples=HOLDOUT_PERIOD_SAMPLES,
            )
            off_bins = off_grid_holdout_bins()
            ref_fft = np.fft.rfft(observed_holdout)
            est_fft = np.fft.rfft(prediction)
            off_grid = _complex_metrics(ref_fft[off_bins], est_fft[off_bins])
            off_passed = bool(
                off_grid["relative_error"]
                <= BROADBAND_MEASURED_INTERPOLATION_MAX_RELATIVE_ERROR
                and off_grid["complex_agreement"]
                >= BROADBAND_MEASURED_INTERPOLATION_MIN_AGREEMENT
            )
            tail = _tail_receipt(
                impulses,
                delay=delay,
                support=support,
                input_peak=float(np.max(np.abs(holdout_period.astype(np.float64) / 32_767.0))),
                heldout_output_rms=float(np.sqrt(np.mean(observed_holdout * observed_holdout))),
            )
            panel = _panel_cross_binding(
                kernel=kernel,
                panel=panel_bindings[path],
                synthetic_fixture=synthetic_fixture,
            )
            passed = bool(
                all(item["passed"] for item in bands)
                and off_passed
                and tail["passed"]
                and all(item["passed"] for item in panel)
            )
            candidates.append(
                {
                    "support_samples": support,
                    "holdout_subbands": bands,
                    "off_grid_holdout": {
                        "tone_count": int(off_bins.size),
                        **off_grid,
                        "passed": off_passed,
                    },
                    "tail": tail,
                    "panel_cross_binding": panel,
                    "passed": passed,
                }
            )
            if passed and selected_kernel is None:
                selected_kernel = kernel
                selected_support = support
        path_passed = selected_kernel is not None
        all_passed &= path_passed
        path_receipts[path] = {
            "delay_samples": delay,
            "selected_support_samples": selected_support,
            "selected_kernel": selected_kernel,
            "selected_kernel_semantics": (
                "explicit_integer_bulk_delay_then_post_onset_fir; fractional_phase_in_taps; "
                "secondary_handoff_256_not_baked"
            ),
            "final_noise_floor": noise_floor,
            "candidates": candidates,
            "passed": path_passed,
        }
    return {
        "schema": ANALYSIS_SCHEMA,
        "status": "PASS" if all_passed else "BLOCKED",
        # v1 periodic prototype은 finite support/tail을 증명하지 못한다. production
        # receipt를 넣어도 canonical plant로 승격할 수 없다.
        "canonical_training_eligible": False,
        "invalidated_for_canonical_reason": (
            "periodic_deconvolution_cannot_prove_finite_causal_support_or_tail"
        ),
        "synthetic_fixture_only": bool(synthetic_fixture),
        "operator_role": "fullband_causal_history_prefix",
        "compact_partial_band_promotion_forbidden": True,
        "clock_receipt": dict(clock_receipt),
        "paths": path_receipts,
    }


def receipt_json_safe(receipt: Mapping[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value
    return convert(dict(receipt))


def publish_receipt_noreplace(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"기존 causal analysis를 덮어쓰지 않습니다: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = receipt_json_safe(receipt)
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.partial"
    try:
        with temporary.open("xb") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode())
            handle.flush()
            os.fsync(handle.fileno())
        atomic_publish_noreplace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-session", type=Path, required=True)
    parser.parse_args(argv)
    print(
        "[BLOCKED] production raw의 absolute DAC-q time-warp/panel binding publisher는 "
        "live authority 고정 전 열리지 않습니다. synthetic core tests만 허용됩니다.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
