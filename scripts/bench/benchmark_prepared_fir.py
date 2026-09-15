#!/usr/bin/env python3
"""합성 사전 FIR 1개의 워밍업 효과를 비교하는 무출력 CPU 벤치마크.

실측 재현/선택기/CNN 학습/실시간 성능 시험이 아니다. exit 0은 새 보고서 생성만
뜻한다. 기존 결과는 덮어쓰지 않으며 장치, WAV, 외부 학습 자원을 사용하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.baselines.fxlms_core import FxLMSController, SampleDelay  # noqa: E402
from deep_anc.baselines.prepared_fir import (  # noqa: E402
    FilterProposal,
    PreparedFxNLMSController,
)


FAMILIES = ("low_300", "target_900", "target_1300", "mixed", "white")
VARIANTS = ("zero_control", "cold_fxnlms", "fixed_fir", "prepared_fxnlms")
POWER_FLOOR = 1e-16


def _excitation(family: str, samples: int, fs: int, seed: int) -> np.ndarray:
    """REF만 생성한다. 평가 ERR/미래 샘플로 사전 후보를 고르지 않는다."""
    rng = np.random.default_rng(seed)
    if family == "white":
        return np.clip(rng.normal(0.0, 0.04, samples), -0.16, 0.16).astype(np.float32)
    frequencies = {
        "low_300": (300.0,), "target_900": (900.0,),
        "target_1300": (1300.0,), "mixed": (300.0, 900.0, 1300.0),
    }[family]
    time = np.arange(samples, dtype=np.float64) / fs
    value = np.zeros(samples, dtype=np.float64)
    for frequency in frequencies:
        value += 0.04 * np.sin(2.0 * np.pi * frequency * time + rng.uniform(-np.pi, np.pi))
    return value.astype(np.float32)


class _Plant:
    """이전 출력 블록을 사용해 e=d+S*y를 인과적으로 계산한다.

    previous_y에 이미 1 hop 지연이 있으므로 delay 큐에는 나머지만 넣는다.
    controller의 filtered-x에는 전체 S delay + handoff를 전달한다.
    """

    def __init__(self, s_hat, effective_delay, hop, saturation_level=None):
        self.delay = SampleDelay(effective_delay - hop)
        self.s_hat = s_hat
        self.state = np.zeros(len(s_hat) - 1, dtype=np.float32)
        self.saturation_level = saturation_level

    def step(self, previous_y):
        drive = previous_y
        if self.saturation_level is not None:
            drive = self.saturation_level * np.tanh(drive / self.saturation_level)
        delayed = self.delay.process(drive)
        output, self.state = signal.lfilter(self.s_hat, [1.0], delayed, zi=self.state)
        return np.asarray(output, dtype=np.float32)


def _rollout(reference, disturbance, controller, *, hop, s_hat, effective_delay,
             adapt, control_limit, saturation_level=None, training=False):
    plant = _Plant(s_hat, effective_delay, hop, saturation_level)
    previous_y = np.zeros(hop, dtype=np.float32)
    errors = np.zeros_like(reference)
    controls = np.zeros_like(reference)
    updates = 0
    train_clips = 0
    for begin in range(0, reference.size, hop):
        end = begin + hop
        error = disturbance[begin:end] + plant.step(previous_y)
        if controller is None:
            output = np.zeros(hop, dtype=np.float32)
        else:
            output = controller.generate_block(reference[begin:end])
            clipped = False
            if training:
                # 기존 FxLMS 코어에는 limiter가 없다. 학습만 여기서 정확히 한 번 적용.
                clipped = bool(np.any(np.abs(output) > control_limit))
                train_clips += int(np.count_nonzero(np.abs(output) > control_limit))
                if clipped:
                    # 지연된 clipping ERR를 정상 선형 gradient로 재사용하지 않는다.
                    # 다른 seed로 성공 후보를 골라 재시도하지 않고 학습 자체를 무효화한다.
                    raise ValueError("사전 학습 출력 clipping: 후보 무효, 자동 재학습/선별 없음")
                output = np.clip(output, -control_limit, control_limit)
            result = controller.adapt_block(error, enabled=adapt and not clipped)
            updates += int(result.adapted)
        errors[begin:end] = error
        controls[begin:end] = output
        previous_y = output
    return {
        "error": errors, "control": controls, "adapted_blocks": updates,
        "clip_samples": train_clips if training else int(getattr(controller, "clip_samples", 0)),
    }


def _power(values: np.ndarray, low: float | None = None, high: float | None = None,
           *, fs: int) -> float:
    if low is None:
        return float(np.mean(np.asarray(values, dtype=np.float64) ** 2))
    spectrum = np.fft.rfft(np.asarray(values, dtype=np.float64))
    power = np.abs(spectrum) ** 2 / values.size ** 2
    weights = np.full(power.shape, 2.0)
    weights[0] = 1.0
    if values.size % 2 == 0:
        weights[-1] = 1.0
    frequencies = np.fft.rfftfreq(values.size, 1.0 / fs)
    selected = (frequencies >= low) & (frequencies < high)
    return float(np.sum(power[selected] * weights[selected]))


def _metrics(reference, disturbance, rollout, periods, *, fs, scenario, family, variant):
    rows = []
    bands = (("fullband", None, None), ("below_800", 0.0, 800.0),
             ("target_800_1600", 800.0, 1600.0), ("above_1600", 1600.0, fs / 2.0 + 1))
    for period, (begin, end) in periods.items():
        for band, low, high in bands:
            baseline = _power(disturbance[begin:end], low, high, fs=fs)
            residual = _power(rollout["error"][begin:end], low, high, fs=fs)
            available = baseline > POWER_FLOOR and residual > POWER_FLOOR
            rows.append({
                "scenario": scenario, "source_family": family, "variant": variant,
                "period": period, "start_sample": begin, "end_sample_exclusive": end,
                "band": band, "baseline_power": baseline, "error_power": residual,
                "ref_power": _power(reference[begin:end], low, high, fs=fs),
                "reduction_db": float(10.0 * (np.log10(baseline) - np.log10(residual))) if available else None,
                "comparison_available": available,
                "emergent_error_energy": baseline <= POWER_FLOOR < residual,
                "control_peak": float(np.max(np.abs(rollout["control"][begin:end]))),
                "synthetic_only": True, "diagnostic_only": True,
                "performance_claim_allowed": False,
            })
    return rows


def run_benchmark(*, sample_rate=48000, hop=256, secondary_delay_samples=1465,
                  stress_preview_samples=140, control_length=16, train_blocks=400,
                  eval_blocks=160, train_seed=1729, heldout_seed=2718,
                  saturation_level=None) -> dict:
    """독립 합성 train의 고정 후보를 전달한다. 실측/음성/음악 평가는 아니다."""
    counts = (sample_rate, hop, control_length, train_blocks, eval_blocks)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in counts):
        raise ValueError("sample rate/hop/taps/blocks는 양의 정수여야 합니다")
    if sample_rate <= 3200:
        raise ValueError("800–1600 Hz 평가에는 sample_rate > 3200이 필요합니다")
    if secondary_delay_samples < 0 or stress_preview_samples < 0:
        raise ValueError("지연/REF preview는 음수가 될 수 없습니다")
    effective_delay = secondary_delay_samples + hop
    if stress_preview_samples >= effective_delay:
        raise ValueError("긴 지연 stress의 preview는 S delay + handoff보다 작아야 합니다")
    if eval_blocks * hop < 8 * (effective_delay + hop) or train_blocks * hop <= effective_delay:
        raise ValueError("초기/조건 변경/후기 구간을 분리하기에 합성 블록 수가 부족합니다")
    if train_seed < 0 or heldout_seed < 0 or train_seed in range(heldout_seed, heldout_seed + len(FAMILIES)):
        raise ValueError("train과 모든 heldout family seed는 겹치지 않는 비음수 정수여야 합니다")
    if saturation_level is not None and (not np.isfinite(saturation_level) or saturation_level <= 0):
        raise ValueError("saturation_level은 유한한 양수여야 합니다")

    # 실측 S가 아닌 명시적인 짧은 선형 FIR. 지연 숫자만 현재 설정에 맞춘 기본값이다.
    s_hat = np.array([-0.5, -0.08], dtype=np.float32)
    control_limit = 0.1
    mu = 0.25
    train_ref = _excitation("white", train_blocks * hop, sample_rate, train_seed)
    train_d = SampleDelay(effective_delay).process(0.15 * train_ref)
    trainer = FxLMSController(s_hat, secondary_delay_samples=effective_delay,
                             control_len=control_length, mu=mu, leakage=0.0,
                             weight_norm_limit=2.0)
    train_result = _rollout(train_ref, train_d, trainer, hop=hop, s_hat=s_hat,
                            effective_delay=effective_delay, adapt=True,
                            control_limit=control_limit, training=True)
    coefficients = tuple(float(value) for value in trainer.w)
    kwargs = dict(sample_rate=sample_rate, hop=hop,
                  secondary_delay_samples=secondary_delay_samples,
                  handoff_extra_samples=hop, control_length=control_length,
                  reference_peak_limit=0.2, control_limit=control_limit,
                  prepared_l1_limit=2.0, residual_norm_limit=2.0,
                  mu=mu, leakage=0.0, transition_samples=hop,
                  max_proposal_age_samples=hop)
    count = eval_blocks * hop
    gain_change = (eval_blocks // 2) * hop
    span = count // 8
    metrics = []
    runs = []
    for scenario, preview in (("causal_toy", effective_delay),
                              ("long_delay_stress", stress_preview_samples)):
        change_at_err = gain_change + preview
        periods = {
            "early": (effective_delay, effective_delay + span),
            "before_gain_change": (change_at_err - span, change_at_err),
            "after_gain_change": (change_at_err, change_at_err + span),
            "steady": (count - span, count),
        }
        for index, family in enumerate(FAMILIES):
            # 같은 heldout REF/외란을 모든 대조군에 그대로 재생한다. train 참조 없음.
            reference = _excitation(family, count, sample_rate, heldout_seed + index)
            gain = np.full(count, 0.15, dtype=np.float32)
            gain[gain_change:] = 0.20
            disturbance = SampleDelay(preview).process(gain * reference)
            for variant in VARIANTS:
                controller = None if variant == "zero_control" else PreparedFxNLMSController(s_hat, **kwargs)
                if variant in ("fixed_fir", "prepared_fxnlms"):
                    decision = controller.submit(FilterProposal(
                        context_id=controller.context_id, generation=controller.generation,
                        revision=1, observed_until_sample=0, coefficients=coefficients,
                        source_id=f"independent_synthetic_train_seed_{train_seed}",
                    ))
                    if not decision.accepted:
                        raise ValueError(f"사전 후보 거부: {decision.reason}")
                result = _rollout(reference, disturbance, controller, hop=hop, s_hat=s_hat,
                                  effective_delay=effective_delay,
                                  adapt=variant in ("cold_fxnlms", "prepared_fxnlms"),
                                  control_limit=control_limit, saturation_level=saturation_level)
                metrics.extend(_metrics(reference, disturbance, result, periods, fs=sample_rate,
                                        scenario=scenario, family=family, variant=variant))
                runs.append({"scenario": scenario, "source_family": family, "variant": variant,
                             "clip_samples": result["clip_samples"],
                             "adapted_blocks": result["adapted_blocks"],
                             "residual_norm": float(np.linalg.norm(controller.residual_weights)) if controller is not None else 0.0})

    return {
        "schema_version": 1, "synthetic_only": True, "diagnostic_only": True,
        "performance_claim_allowed": False, "real_time_claim_allowed": False,
        "reference_mode": "acoustic_simulation", "selector_implemented": False,
        "cnn_trained": False, "candidate_count": 1,
        "controller_configuration": kwargs.copy(),
        "context_id": PreparedFxNLMSController(s_hat, **kwargs).context_id,
        "training": {"scenario": "causal_toy", "seed": train_seed,
                     "source_family": "white", "blocks": train_blocks,
                     "reference_sha256": hashlib.sha256(train_ref.tobytes()).hexdigest(),
                     "coefficients": list(coefficients), "clip_samples": train_result["clip_samples"],
                     "primary_preview_samples": effective_delay, "primary_gain": 0.15,
                     "method": "causal FxNLMS, independent train stream; no heldout/oracle"},
        "scenarios": {
            "causal_toy": {"primary_preview_samples": effective_delay},
            "long_delay_stress": {
                "primary_preview_samples": stress_preview_samples,
                "training_primary_preview_samples": effective_delay,
                "uses_same_candidate_without_retraining": True,
            },
        },
        "heldout": {"seed": heldout_seed, "family_seeds": {family: heldout_seed + index for index, family in enumerate(FAMILIES)},
                    "blocks": eval_blocks, "source_gain_change_sample": gain_change,
                    "primary_gains": [0.15, 0.20], "source_families": list(FAMILIES)},
        "plant": {"sample_rate": sample_rate, "hop": hop, "synthetic_secondary_fir": s_hat.tolist(),
                  "secondary_delay_samples": secondary_delay_samples, "handoff_samples": hop,
                  "effective_control_delay_samples": effective_delay,
                  "stress_reference_preview_samples": stress_preview_samples,
                  "control_length": control_length, "control_limit": control_limit,
                  "transition_samples": hop, "saturation_level": saturation_level,
                  "polarity": "e=d+S*y", "measured_secondary_path_used": False},
        "warnings": [
            "전체 결과는 합성 진단이며 실제 Jetson/USB DAC/덕트의 800–1600 Hz 감쇠 증거가 아닙니다.",
            "causal_toy는 REF preview를 S+handoff와 같게 둔 별도 조건이며 현재 하드웨어 조건이 아닙니다.",
            "long_delay_stress는 지연 숫자만 참고합니다. 실측 S/F, 음성/음악, USB drift, 공간 성능은 재현하지 않습니다.",
            "긴 지연에서는 사전 후보의 primary preview 조건도 불일치합니다. 같은 후보를 재학습 없이 옮긴 스트레스이며 해당 조건에 맞춰 준비된 필터의 최적성/일반적 성능으로 해석할 수 없습니다.",
            "주기 신호 감쇠를 독립 white-noise 또는 모든 소리에 대한 예측 능력으로 일반화하지 마세요.",
            "한 정적 후보의 워밍업만 비교합니다. CNN/방향 분류/느린 온라인 선택/계수 생성은 미구현입니다.",
            "tanh 옵션은 비선형 스트레스일 뿐 실측 비선형 모델이나 선형 FIR의 비선형 해결 증명이 아닙니다.",
            "FFT 직사각 창 대역 파워이며 유한 창 누설이 있습니다. 무신호/수치 바닥 이하 감쇠는 null입니다.",
            "exit 0은 보고서 생성 성공이며 특정 방법의 우수성/안전성/실시간성 통과가 아닙니다.",
        ],
        "runs": runs, "metrics": metrics,
    }


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(1, f"{self.prog}: 인자 오류: {message}\n")


def _new_output_path(value: str) -> Path:
    out = Path(value).expanduser()
    if not out.is_absolute():
        out = Path.cwd() / out
    for part in (out, *out.parents):
        if part.is_symlink():
            raise ValueError(f"출력 경로에 심볼릭 링크를 사용할 수 없습니다: {part}")
    if out.exists():
        raise FileExistsError(f"기존 출력 경로는 덮어쓰지 않습니다: {out}")
    return out


def main(argv=None) -> int:
    parser = _Parser(description=__doc__)
    parser.add_argument("--out", required=True, help="아직 존재하지 않는 결과 디렉터리")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--secondary-delay-samples", type=int, default=1465)
    parser.add_argument("--stress-preview-samples", type=int, default=140)
    parser.add_argument("--control-length", type=int, default=16)
    parser.add_argument("--train-blocks", type=int, default=400)
    parser.add_argument("--eval-blocks", type=int, default=160)
    parser.add_argument("--train-seed", type=int, default=1729)
    parser.add_argument("--heldout-seed", type=int, default=2718)
    parser.add_argument("--saturation-level", type=float, help="선택적 tanh 출력 경로 스트레스 크기(실측 아님)")
    args = parser.parse_args(argv)
    try:
        options = vars(args).copy()
        out = _new_output_path(options.pop("out"))
        report = run_benchmark(**options)
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(report["metrics"][0]))
        writer.writeheader()
        for row in report["metrics"]:
            writer.writerow({key: "null" if value is None else value for key, value in row.items()})
        summary = "# Prepared FIR 합성 진단\n\n"
        summary += "`synthetic_only=true`, `diagnostic_only=true`, `performance_claim_allowed=false`.\n\n"
        summary += "한 독립 train 후보 → zero/cold/fixed/prepared+FxNLMS, 같은 지연·출력 제한으로 비교합니다.\n\n"
        summary += "숫자의 단일 출처는 report.json의 metrics이며 metrics.csv가 같은 행을 보존합니다.\n\n"
        summary += "\n".join(f"- {warning}" for warning in report["warnings"]) + "\n"
        _new_output_path(str(out))
        out.mkdir(parents=True, exist_ok=False)
        for name, content in (("report.json", encoded), ("metrics.csv", buffer.getvalue()),
                              ("summary.md", summary)):
            with (out / name).open("x", encoding="utf-8", newline="") as handle:
                handle.write(content)
    except (OSError, ValueError, TypeError, RuntimeError, FloatingPointError) as exc:
        print(f"[FAIL] 합성 벤치마크: {exc}", file=sys.stderr)
        return 1
    print(f"[진단 보고서 생성] {out} — synthetic_only, 실제 감쇠/성능 통과 아님")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
