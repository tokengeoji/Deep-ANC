"""48 kHz 광대역 제어 파형의 모델 출력 표현력 감사.

이 모듈은 음향 성능을 추정하지 않는다. ``ConvTranspose1d`` 출력 격자가
100--11.314 kHz 파형을 샘플 단위로 *표현할 수 있는지*와, 저장된 weight의
출력 합성부가 수치적으로 얼마나 취약한지만 검사한다.

중요한 경계
-----------
``hop=128``은 출력 sample rate를 375 Hz로 낮추지 않는다. 한 frame의 latent가
128개 polyphase sample을 함께 만들기 때문에 합성 polyphase 행렬이 full row rank이면
Nyquist 아래 파형을 만들 수 있다. 반대로 shape가 맞는다는 사실만으로 그 rank나
BF16/FP16 위상 정밀도가 보장되지는 않는다.

여기서 검사하는 ``positive_branch``는 ``head -> PReLU -> decoder``의 PReLU 양의
선형 구간이다. 이는 필요한 구조적 조건이자 유용한 수치 진단이지만 전체 비선형 모델의
충분조건은 아니다. 실제 학습 admission에는 :func:`broadband_g0_gate_spec`의 plant/limiter/
prefix/streaming G0가 별도로 필요하다.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..dsp.control_band_contract import (
    BROADBAND_GLOBAL_CLOCK_MAX_RESIDUAL_SAMPLES,
    BROADBAND_OCTAVE_CENTERS_HZ,
    BROADBAND_POINT_CONTROL_SUBBANDS_HZ,
    OCTAVE_8K_UPPER_HZ,
    REQUIRED_SOURCE_FAMILIES,
)


BROADBAND_REPRESENTABILITY_SCHEMA = "broadband_model_representability_v1"
BROADBAND_G0_GATE_SCHEMA = "broadband_g0_representability_gate_v1"

SAMPLE_RATE = 48_000
RUNTIME_BLOCK_SAMPLES = 256
BROADBAND_LOWER_HZ = 100.0
BROADBAND_UPPER_HZ = OCTAVE_8K_UPPER_HZ
TWENTY_DB_RELATIVE_COMPLEX_ERROR = 0.1

# octave 중심뿐 아니라 최종 octave 상단과 측정 하단을 포함한다. hop frame-rate
# alias가 같은 주파수도 sample-phase vector는 다르므로 빼면 안 된다.
DEFAULT_PROBE_FREQUENCIES_HZ = (
    100.0,
    125.0,
    250.0,
    500.0,
    1_000.0,
    1_600.0,
    2_000.0,
    4_000.0,
    8_000.0,
    BROADBAND_UPPER_HZ,
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _finite_weight(
    state: Mapping[str, torch.Tensor], name: str, *, ndim: int
) -> torch.Tensor:
    value = state.get(name)
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"checkpoint {name} weight shape가 유효하지 않습니다")
    value = value.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"checkpoint {name} weight에 NaN/Inf가 있습니다")
    return value


def output_lattice_contract(
    model_cfg: Mapping[str, Any],
    *,
    sample_rate: int = SAMPLE_RATE,
    runtime_block_samples: int = RUNTIME_BLOCK_SAMPLES,
) -> dict[str, Any]:
    """모델 shape, receptive field와 실제 block handoff의 구조 계약을 계산한다."""

    rate = int(sample_rate)
    block = int(runtime_block_samples)
    hop = int(model_cfg.get("hop", 0))
    win = int(model_cfg.get("win", 0))
    encoder = model_cfg.get("encoder")
    tcn = model_cfg.get("tcn")
    glstm = model_cfg.get("glstm") or {}
    if not isinstance(encoder, Mapping) or not isinstance(tcn, Mapping):
        raise ValueError("model encoder/tcn 설정이 없습니다")
    core_channels = int(encoder.get("channels", 0))
    head_channels = 2 * core_channels
    dilations = tuple(int(value) for value in tcn.get("dilations", ()))
    repeats = int(tcn.get("repeats", 0))
    kernel = int(tcn.get("kernel", 0))
    if (
        rate <= 0
        or block <= 0
        or hop <= 0
        or win <= hop
        or core_channels <= 0
        or repeats <= 0
        or kernel <= 0
        or not dilations
        or any(value <= 0 for value in dilations)
    ):
        raise ValueError("model 출력 격자 설정이 유효하지 않습니다")

    tcn_history_frames = repeats * sum((kernel - 1) * d for d in dilations)
    tcn_receptive_frames = 1 + tcn_history_frames
    decoder_frame_span = int(math.ceil(win / hop))
    # GLSTM 이전의 유한 경로만 센 값이다. GLSTM이 있으면 전체 과거 수용영역은 무한이다.
    finite_branch_span_samples = (
        (tcn_history_frames + decoder_frame_span - 1) * hop + win
    )
    maximum_intra_hop_future_dependency = hop - 1
    nyquist = rate / 2.0
    structural_reasons: list[str] = []
    if block % hop != 0:
        structural_reasons.append("runtime block이 model hop의 배수가 아닙니다")
    if win % hop != 0:
        structural_reasons.append("decoder win이 hop polyphase로 정확히 나뉘지 않습니다")
    if core_channels < hop:
        structural_reasons.append(
            "frame core channel이 hop보다 작아 positive-branch HxC full row rank가 불가능합니다"
        )
    if head_channels < hop:
        structural_reasons.append("decoder 입력 channel이 hop보다 작습니다")
    if nyquist < BROADBAND_UPPER_HZ:
        structural_reasons.append("sample-rate Nyquist가 8 kHz octave 상단보다 낮습니다")
    if block <= maximum_intra_hop_future_dependency:
        structural_reasons.append("1-block handoff가 encoder intra-hop dependency보다 짧습니다")

    return {
        "schema_version": BROADBAND_REPRESENTABILITY_SCHEMA,
        "sample_rate": rate,
        "runtime_block_samples": block,
        "runtime_block_seconds": block / float(rate),
        "hop_samples": hop,
        "win_samples": win,
        "encoder_context_samples": win - hop,
        "frames_per_runtime_block": block // hop if block % hop == 0 else None,
        "decoder_polyphase_phases": hop,
        "decoder_frame_span": decoder_frame_span,
        "core_channels": core_channels,
        "head_channels": head_channels,
        "nyquist_hz": nyquist,
        "broadband_upper_hz": BROADBAND_UPPER_HZ,
        "sample_rate_covers_broadband": nyquist >= BROADBAND_UPPER_HZ,
        "tcn_history_frames": tcn_history_frames,
        "tcn_receptive_frames": tcn_receptive_frames,
        "finite_branch_input_span_samples": finite_branch_span_samples,
        "finite_branch_input_span_seconds": finite_branch_span_samples / float(rate),
        "glstm_present": bool(glstm),
        "past_receptive_field": "unbounded_state" if glstm else "finite",
        # frame-boundary causality와 sample causality를 혼동하지 않도록 반드시 노출한다.
        "maximum_intra_hop_future_dependency_samples": (
            maximum_intra_hop_future_dependency
        ),
        "sample_causal_without_runtime_handoff": maximum_intra_hop_future_dependency == 0,
        "runtime_handoff_makes_dependency_implementable": (
            block > maximum_intra_hop_future_dependency
        ),
        "structural_passed": not structural_reasons,
        "reasons": structural_reasons,
    }


def _quantized_numpy(value: torch.Tensor, precision: str) -> np.ndarray:
    if precision == "fp64_reference":
        return value.double().numpy()
    if precision == "fp16_weights":
        return value.half().float().double().numpy()
    if precision == "bf16_weights":
        return value.bfloat16().float().double().numpy()
    raise ValueError(f"알 수 없는 precision: {precision}")


def positive_branch_polyphase_matrices(
    model_state: Mapping[str, torch.Tensor],
    *,
    hop: int,
    win: int,
    precision: str = "fp64_reference",
) -> np.ndarray:
    """``head->PReLU(positive)->decoder``의 ``[Q,H,C]`` polyphase 행렬."""

    hop_i = int(hop)
    win_i = int(win)
    if hop_i <= 0 or win_i <= 0 or win_i % hop_i != 0:
        raise ValueError("hop/win polyphase shape가 유효하지 않습니다")
    decoder = _finite_weight(model_state, "decoder.weight", ndim=3)
    head = _finite_weight(model_state, "head.weight", ndim=3)
    if decoder.shape[1] != 1 or decoder.shape[2] != win_i:
        raise ValueError(
            f"decoder weight가 [C,1,{win_i}]이 아닙니다: {tuple(decoder.shape)}"
        )
    if head.shape[0] != decoder.shape[0] or head.shape[2] != 1:
        raise ValueError("head/decoder channel 또는 1x1 kernel shape가 다릅니다")
    decoder_np = _quantized_numpy(decoder[:, 0, :], precision)
    head_np = _quantized_numpy(head[:, :, 0], precision)
    phases = []
    for frame_offset in range(win_i // hop_i):
        polyphase = decoder_np[
            :, frame_offset * hop_i : (frame_offset + 1) * hop_i
        ].T
        phases.append(polyphase @ head_np)
    return np.stack(phases, axis=0)


def _matrix_at_sample_frequency(
    matrices: np.ndarray, *, frequency_hz: float, sample_rate: int, hop: int
) -> np.ndarray:
    frame_omega = 2.0 * math.pi * float(frequency_hz) * int(hop) / int(sample_rate)
    phase = np.exp(-1j * frame_omega * np.arange(matrices.shape[0]))
    return np.tensordot(phase, matrices, axes=(0, 0))


def checkpoint_polyphase_report(
    model_state: Mapping[str, torch.Tensor],
    model_cfg: Mapping[str, Any],
    *,
    frequencies_hz: Sequence[float] = DEFAULT_PROBE_FREQUENCIES_HZ,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, Any]:
    """저장 weight의 tone steering rank와 FP16/BF16 민감도를 재계산한다.

    20 dB cancellation이면 복소 반대파형의 상대 오차 진폭이 0.1 이하여야 한다.
    FP16/BF16 행렬에는 FP64에서 구한 *같은* latent steering을 적용한다. 양자화 뒤 다시
    최적화하면 실제 deployment가 갖지 못하는 보정 자유도를 주므로 금지한다.
    """

    lattice = output_lattice_contract(model_cfg, sample_rate=sample_rate)
    hop = int(lattice["hop_samples"])
    win = int(lattice["win_samples"])
    frequency = np.asarray(tuple(float(v) for v in frequencies_hz), dtype=np.float64)
    if (
        frequency.ndim != 1
        or frequency.size == 0
        or not np.all(np.isfinite(frequency))
        or np.any(frequency <= 0.0)
        or np.any(frequency >= sample_rate / 2.0)
    ):
        raise ValueError("probe frequency가 (0, Nyquist) 유한 범위가 아닙니다")

    matrices = {
        precision: positive_branch_polyphase_matrices(
            model_state, hop=hop, win=win, precision=precision
        )
        for precision in ("fp64_reference", "fp16_weights", "bf16_weights")
    }
    rows: list[dict[str, Any]] = []
    for value in frequency:
        response = {
            precision: _matrix_at_sample_frequency(
                matrix,
                frequency_hz=float(value),
                sample_rate=int(sample_rate),
                hop=hop,
            )
            for precision, matrix in matrices.items()
        }
        reference = response["fp64_reference"]
        target = np.exp(
            2j * math.pi * float(value) * np.arange(hop) / float(sample_rate)
        )
        latent, _, _, singular = np.linalg.lstsq(reference, target, rcond=None)
        reconstructed = reference @ latent
        target_norm = float(np.linalg.norm(target))
        relative_error = float(np.linalg.norm(reconstructed - target) / target_norm)
        algebraic_tolerance = float(
            max(reference.shape)
            * np.finfo(np.float64).eps
            * float(singular[0])
        )
        float32_tolerance = float(
            max(reference.shape)
            * np.finfo(np.float32).eps
            * float(singular[0])
        )
        algebraic_rank = int(np.count_nonzero(singular > algebraic_tolerance))
        float32_numeric_rank = int(np.count_nonzero(singular > float32_tolerance))
        minimum = float(singular[-1])
        condition = float(singular[0] / minimum) if minimum > 0.0 else math.inf
        quantization_errors = {
            precision: float(
                np.linalg.norm(response[precision] @ latent - target) / target_norm
            )
            for precision in ("fp16_weights", "bf16_weights")
        }
        rows.append(
            {
                "frequency_hz": float(value),
                "frame_frequency_radians": float(
                    math.remainder(
                        2.0 * math.pi * float(value) * hop / float(sample_rate),
                        2.0 * math.pi,
                    )
                ),
                "algebraic_rank": algebraic_rank,
                "float32_numeric_rank": float32_numeric_rank,
                "required_row_rank": hop,
                "minimum_singular_value": minimum,
                "maximum_singular_value": float(singular[0]),
                "condition_number": condition if math.isfinite(condition) else None,
                "fp64_steering_relative_error": relative_error,
                "latent_rms_per_unit_output_rms": float(
                    np.sqrt(np.mean(np.abs(latent) ** 2))
                ),
                "fp16_weight_steering_relative_error": quantization_errors[
                    "fp16_weights"
                ],
                "bf16_weight_steering_relative_error": quantization_errors[
                    "bf16_weights"
                ],
                "fp16_meets_20db_complex_error": (
                    quantization_errors["fp16_weights"]
                    <= TWENTY_DB_RELATIVE_COMPLEX_ERROR
                ),
                "bf16_meets_20db_complex_error": (
                    quantization_errors["bf16_weights"]
                    <= TWENTY_DB_RELATIVE_COMPLEX_ERROR
                ),
            }
        )

    required_rank = hop
    algebraic_passed = all(
        row["algebraic_rank"] == required_rank
        and row["fp64_steering_relative_error"] <= 1.0e-10
        for row in rows
    )
    fp16_passed = all(row["fp16_meets_20db_complex_error"] for row in rows)
    bf16_passed = all(row["bf16_meets_20db_complex_error"] for row in rows)
    prelu = _finite_weight(model_state, "head_act.weight", ndim=1)
    return {
        "schema_version": BROADBAND_REPRESENTABILITY_SCHEMA,
        "linearization": "head_prelu_positive_branch_then_convtranspose_polyphase",
        "linearization_is_full_model_sufficiency_proof": False,
        "sample_rate": int(sample_rate),
        "hop_samples": hop,
        "win_samples": win,
        "prelu_negative_slope": [float(value) for value in prelu.tolist()],
        "twenty_db_relative_complex_error_limit": (
            TWENTY_DB_RELATIVE_COMPLEX_ERROR
        ),
        "frequencies": rows,
        "minimum_algebraic_rank": min(row["algebraic_rank"] for row in rows),
        "minimum_float32_numeric_rank": min(
            row["float32_numeric_rank"] for row in rows
        ),
        "maximum_condition_number": max(
            float(row["condition_number"] or math.inf) for row in rows
        ),
        "maximum_latent_rms_per_unit_output_rms": max(
            float(row["latent_rms_per_unit_output_rms"]) for row in rows
        ),
        "maximum_fp16_weight_steering_relative_error": max(
            float(row["fp16_weight_steering_relative_error"]) for row in rows
        ),
        "maximum_bf16_weight_steering_relative_error": max(
            float(row["bf16_weight_steering_relative_error"]) for row in rows
        ),
        "algebraic_probe_passed": algebraic_passed,
        "fp16_weight_probe_passed": fp16_passed,
        "bf16_weight_probe_passed": bf16_passed,
        "canonical_training_admitted": False,
        "blocked_reason": "requires_fullband_causal_plant_and_broadband_g0_receipt",
    }


def tone_limiter_feasibility_report(
    primary_response: Sequence[complex] | np.ndarray,
    secondary_response: Sequence[complex] | np.ndarray,
    *,
    source_peak: float,
    limiter_limit: float,
    operational_fraction: float = 0.9,
) -> dict[str, Any]:
    """동일 tone bin의 ``|P/S|``로 limiter 필요 peak를 진단한다.

    이 결과는 동시 다중대역/시간영역 inverse peak를 보장하지 않으므로 G0 admission으로
    사용할 수 없다. fullband G0는 실제 causal operator로 만든 oracle waveform을 검사한다.
    """

    primary = np.asarray(primary_response, dtype=np.complex128).reshape(-1)
    secondary = np.asarray(secondary_response, dtype=np.complex128).reshape(-1)
    peak = float(source_peak)
    limit = float(limiter_limit)
    fraction = float(operational_fraction)
    if (
        primary.size == 0
        or primary.shape != secondary.shape
        or not np.all(np.isfinite(primary))
        or not np.all(np.isfinite(secondary))
        or not math.isfinite(peak)
        or peak <= 0.0
        or not math.isfinite(limit)
        or limit <= 0.0
        or not 0.0 < fraction < 1.0
    ):
        raise ValueError("tone limiter feasibility 입력이 유효하지 않습니다")
    magnitude = np.abs(secondary)
    if np.any(magnitude <= np.finfo(np.float64).tiny):
        raise ValueError("secondary response에 inverse 불가능한 0 bin이 있습니다")
    ratio = np.abs(primary) / magnitude
    required = peak * ratio
    operating_limit = fraction * limit
    worst = float(np.max(required))
    return {
        "kind": "tone_only_diagnostic_not_multiband_g0",
        "source_peak": peak,
        "limiter_limit": limit,
        "operational_fraction": fraction,
        "operational_limit": operating_limit,
        "required_control_peak_max": worst,
        "required_control_peak_p95": float(np.quantile(required, 0.95)),
        "margin_db_at_worst": float(20.0 * math.log10(operating_limit / worst)),
        "passed": worst <= operating_limit,
        "canonical_training_admitted": False,
    }


def broadband_g0_gate_spec() -> dict[str, Any]:
    """결과를 보기 전에 고정하는 실제 광대역 G0 표현력 gate 명세."""

    payload: dict[str, Any] = {
        "schema_version": BROADBAND_G0_GATE_SCHEMA,
        "role": "pretraining_model_representability_only",
        "init_eligible": False,
        "required_control_band_hz": [150.0, BROADBAND_UPPER_HZ],
        "required_source_families": list(REQUIRED_SOURCE_FAMILIES),
        "required_subbands_hz": [
            [float(lo), float(hi)] for lo, hi in BROADBAND_POINT_CONTROL_SUBBANDS_HZ
        ],
        "artifact_requirements": {
            "fullband_causal_primary_operator": True,
            "fullband_causal_secondary_operator": True,
            "measured_panel_cross_binding": True,
            "exact_operator_prefix_or_state": True,
            "operator_and_control_band_sha_binding": True,
            "fixed_batch_source_and_lineage_sha_binding": True,
        },
        "oracle_feasibility": {
            "construction": "constrained_causal_P_over_S_inverse_on_exact_prefix",
            "maximum_abs_control": 0.18,
            "limiter_limit": 0.2,
            "minimum_nmse_db_every_subband": -20.0,
            "reason": "20dB residual amplitude requires relative complex error <=0.1",
        },
        "sample_phase_coverage": {
            "impulse_residues_mod_hop": list(range(128)),
            "tone_centers_hz": list(DEFAULT_PROBE_FREQUENCIES_HZ),
            "required_octave_centers_hz": list(BROADBAND_OCTAVE_CENTERS_HZ),
            "simultaneous_low_high_required": True,
        },
        "model_overfit_gate": {
            "maximum_steps": 2_000,
            "every_subband_nmse_db_strictly_below": -6.0,
            "every_item_nmse_db_strictly_below": 0.0,
            "zero_output_is_failure": True,
            "limiter_abs_peak_at_most": 0.18,
            "limiter_saturation_fraction": 0.0,
        },
        "precision_and_runtime_gate": {
            "offline_streaming_max_abs_error": 1.0e-5,
            "bf16_and_fp16_induced_complex_relative_error_at_most": 0.1,
            "upper_band_timing_residual_samples_at_most": (
                BROADBAND_GLOBAL_CLOCK_MAX_RESIDUAL_SAMPLES
            ),
            "runtime_block_samples": RUNTIME_BLOCK_SAMPLES,
            "deadline_miss_xrun_slip": 0,
        },
        "boundary_contract": {
            "zero_history_fixture_may_use_exact_zero_initial_state": True,
            "random_recorded_crop_requires_real_prefix_or_serialized_state": True,
            "settle_crop_does_not_replace_missing_prefix": True,
            "resume_must_reproduce_prefix_and_model_state": True,
        },
    }
    payload["sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def load_checkpoint_snapshot(path: str | Path) -> tuple[dict[str, Any], str]:
    """한 FD에서 checkpoint bytes와 SHA를 읽어 TOCTOU 없이 decode한다."""

    import io

    checkpoint = Path(path)
    with checkpoint.open("rb") as handle:
        raw = handle.read()
    digest = hashlib.sha256(raw).hexdigest()
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not isinstance(state.get("model"), Mapping):
        raise ValueError(f"checkpoint model/cfg schema가 없습니다: {checkpoint}")
    if not isinstance(state.get("cfg"), Mapping):
        raise ValueError(f"checkpoint cfg schema가 없습니다: {checkpoint}")
    return state, digest


__all__ = [
    "BROADBAND_G0_GATE_SCHEMA",
    "BROADBAND_REPRESENTABILITY_SCHEMA",
    "BROADBAND_UPPER_HZ",
    "DEFAULT_PROBE_FREQUENCIES_HZ",
    "TWENTY_DB_RELATIVE_COMPLEX_ERROR",
    "broadband_g0_gate_spec",
    "checkpoint_polyphase_report",
    "load_checkpoint_snapshot",
    "output_lattice_contract",
    "positive_branch_polyphase_matrices",
    "tone_limiter_feasibility_report",
]
