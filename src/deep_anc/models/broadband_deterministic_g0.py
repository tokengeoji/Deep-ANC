"""실측 오디오 없이 실행하는 Tiny 광대역 구조 G0.

이 시험은 *실제 덕트 성능*이 아니다. 같은 캡처의 P/S bulk delay에서 공통 지연을
제거한 unity-gain delay-only fixture를 사용해 다음 질문만 답한다.

* 실제 :class:`HybridANCNet` forward가 저역·고역 tone과 동시 혼합을 학습하는가?
* 128개 ``sample_index % hop`` impulse가 모두 양의 감쇠를 갖는가?
* 같은 weight/state의 256-sample streaming 출력이 offline과 같은가?

실제 fullband causal P/S, 정확한 prefix, 네 source family가 없으므로 이 결과는 canonical
학습이나 배포를 절대 열지 않는다. 그 admission은 ``broadband_g0_gate_spec``가 맡는다.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..dsp.control_band_contract import (
    BROADBAND_OCTAVE_CENTERS_HZ,
    BROADBAND_POINT_CONTROL_SUBBANDS_HZ,
    OCTAVE_8K_UPPER_HZ,
)
from ..dsp.timing import PlantDelays
from .broadband_representability import (
    broadband_g0_gate_spec,
    checkpoint_polyphase_report,
    output_lattice_contract,
)
from .hybrid_anc import HybridANCNet, parameter_count


BROADBAND_DETERMINISTIC_G0_SCHEMA = "broadband_deterministic_delay_g0_v1"
DETERMINISTIC_G0_SEED = 20_260_828
DETERMINISTIC_G0_STEPS = 500
DETERMINISTIC_G0_AMPLITUDE = 0.003
DETERMINISTIC_G0_LENGTH = 1_024
DETERMINISTIC_G0_EVALUATION_START = 512
DETERMINISTIC_G0_IMPULSE_ANCHOR = 384


def _delay(value: torch.Tensor, samples: int) -> torch.Tensor:
    amount = int(samples)
    if amount < 0:
        raise ValueError("causal delay는 0 이상이어야 합니다")
    if amount == 0:
        return value
    return F.pad(value, (amount, 0))[..., : value.shape[-1]]


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(array.dtype), "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _unique_sorted(values: Sequence[float]) -> tuple[float, ...]:
    result: list[float] = []
    for value in sorted(float(item) for item in values):
        if not result or not math.isclose(value, result[-1], rel_tol=0.0, abs_tol=1e-9):
            result.append(value)
    return tuple(result)


def _normalised_timing_fixture(
    *,
    primary_delay_samples: int,
    secondary_delay_samples: int,
    handoff_samples: int,
    sample_rate: int,
) -> tuple[PlantDelays, PlantDelays, int]:
    """같은 P/S 공통 bulk만 제거하고 :meth:`PlantDelays.lead`를 보존한다."""

    measured = PlantDelays(
        primary_delay_samples=int(primary_delay_samples),
        secondary_delay_samples=int(secondary_delay_samples),
        handoff_samples=int(handoff_samples),
        sample_rate=int(sample_rate),
    )
    common = min(
        int(measured.primary_delay_samples), int(measured.secondary_delay_samples)
    )
    normalised = PlantDelays(
        primary_delay_samples=int(measured.primary_delay_samples) - common,
        secondary_delay_samples=int(measured.secondary_delay_samples) - common,
        handoff_samples=int(measured.handoff_samples),
        sample_rate=int(measured.sample_rate),
    )
    measured_lead = measured.lead()
    normalised_lead = normalised.lead()
    if (
        int(measured_lead) != int(normalised_lead)
        or int(measured_lead.raw_samples) != int(normalised_lead.raw_samples)
    ):
        raise AssertionError("공통 delay 정규화가 lead를 바꿨습니다")
    return measured, normalised, common


def build_deterministic_g0_fixture(
    model_cfg: Mapping[str, Any],
    *,
    primary_delay_samples: int,
    secondary_delay_samples: int,
    handoff_samples: int,
    sample_rate: int = 48_000,
    length: int = DETERMINISTIC_G0_LENGTH,
    amplitude: float = DETERMINISTIC_G0_AMPLITUDE,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """tone·혼합·128 phase impulse의 고정 입력과 primary target을 만든다."""

    lattice = output_lattice_contract(
        model_cfg,
        sample_rate=int(sample_rate),
        runtime_block_samples=int(handoff_samples),
    )
    hop = int(lattice["hop_samples"])
    block = int(lattice["runtime_block_samples"])
    samples = int(length)
    level = float(amplitude)
    if samples <= 0 or samples % hop or samples % block:
        raise ValueError("G0 length는 hop과 runtime block의 양의 공배수여야 합니다")
    if not math.isfinite(level) or not 0.0 < level < 0.2:
        raise ValueError("G0 amplitude가 limiter 안의 유한 양수여야 합니다")

    measured, normalised, common = _normalised_timing_fixture(
        primary_delay_samples=primary_delay_samples,
        secondary_delay_samples=secondary_delay_samples,
        handoff_samples=handoff_samples,
        sample_rate=sample_rate,
    )
    lead = int(normalised.lead())
    secondary_total = int(normalised.secondary_total_delay_samples)
    latest_impulse_error = (
        DETERMINISTIC_G0_IMPULSE_ANCHOR + hop - 1 + secondary_total
    )
    if latest_impulse_error >= samples:
        raise ValueError(
            "G0 impulse tail이 segment 밖입니다: "
            f"latest={latest_impulse_error}, length={samples}"
        )

    requested = (100.0, *BROADBAND_OCTAVE_CENTERS_HZ, OCTAVE_8K_UPPER_HZ)
    band_probes = tuple(
        math.sqrt(float(lo) * float(hi))
        for lo, hi in BROADBAND_POINT_CONTROL_SUBBANDS_HZ
    )
    tone_frequencies = _unique_sorted((*requested, *band_probes))
    target_device = torch.device(device)
    time_axis = torch.arange(
        samples + lead, dtype=torch.float32, device=target_device
    )
    tone_sources = [
        level
        * torch.sin(
            2.0 * math.pi * frequency * time_axis / float(sample_rate)
            + 0.37 * index
        )
        for index, frequency in enumerate(tone_frequencies)
    ]
    mixture = sum(
        torch.sin(
            2.0 * math.pi * frequency * time_axis / float(sample_rate)
            + 0.19 * index
        )
        for index, frequency in enumerate(tone_frequencies)
    )
    mixture = level * mixture / mixture.abs().max().clamp_min(1e-12)
    rows = [*tone_sources, mixture]
    mixture_index = len(rows) - 1
    impulse_start = len(rows)
    for residue in range(hop):
        impulse = torch.zeros_like(time_axis)
        # x_ref[anchor+residue]에 impulse가 오도록 source time에 lead를 더한다.
        impulse[DETERMINISTIC_G0_IMPULSE_ANCHOR + residue + lead] = level
        rows.append(impulse)
    source = torch.stack(rows)
    x_ref = source[:, lead : lead + samples]
    primary = _delay(source[:, :samples], int(normalised.primary_delay_samples))
    model_input = torch.stack([x_ref, torch.zeros_like(x_ref)], dim=1)
    metadata = {
        "schema_version": BROADBAND_DETERMINISTIC_G0_SCHEMA,
        "role": "STRUCTURAL_DIAGNOSTIC_NOT_PHYSICAL_PLANT_PERFORMANCE",
        "sample_rate": int(sample_rate),
        "length_samples": samples,
        "amplitude": level,
        "tone_frequencies_hz": list(tone_frequencies),
        "required_octave_centers_hz": list(BROADBAND_OCTAVE_CENTERS_HZ),
        "subband_probe_frequencies_hz": list(band_probes),
        "mixture_index": mixture_index,
        "impulse_index_start": impulse_start,
        "impulse_count": hop,
        "impulse_residues_mod_hop": list(range(hop)),
        "impulse_anchor_samples": DETERMINISTIC_G0_IMPULSE_ANCHOR,
        "evaluation_start_samples": DETERMINISTIC_G0_EVALUATION_START,
        "measured_delays": measured.model_dump(),
        "removed_common_delay_samples": common,
        "normalised_delay_fixture": normalised.model_dump(),
        "derived_lead_samples": lead,
        "fixture_note": (
            "same-capture P/S bulk의 공통 delay만 제거한 unity-gain delay-only fixture; "
            "P/S FIR 또는 고역 덕트 응답 증거가 아님"
        ),
        "model_input_sha256": _tensor_sha256(model_input),
        "primary_target_sha256": _tensor_sha256(primary),
    }
    return model_input, primary, metadata


def _attenuation_db(residual: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
    numerator = primary.square().sum(dim=-1).clamp_min(1e-20)
    denominator = residual.square().sum(dim=-1).clamp_min(1e-20)
    return 10.0 * torch.log10(numerator / denominator)


def _subband_attenuation(
    primary: np.ndarray, residual: np.ndarray, *, sample_rate: int
) -> list[dict[str, float]]:
    if primary.ndim != 1 or primary.shape != residual.shape or primary.size < 4:
        raise ValueError("subband 입력 shape가 유효하지 않습니다")
    window = np.hanning(primary.size)
    frequency = np.fft.rfftfreq(primary.size, d=1.0 / float(sample_rate))
    p_power = np.abs(np.fft.rfft(primary * window)) ** 2
    e_power = np.abs(np.fft.rfft(residual * window)) ** 2
    rows = []
    for index, (lo, hi) in enumerate(BROADBAND_POINT_CONTROL_SUBBANDS_HZ):
        if index + 1 == len(BROADBAND_POINT_CONTROL_SUBBANDS_HZ):
            mask = (frequency >= float(lo)) & (frequency <= float(hi))
        else:
            mask = (frequency >= float(lo)) & (frequency < float(hi))
        if not np.any(mask):
            raise ValueError(f"FFT grid에 subband [{lo}, {hi}] bin이 없습니다")
        off = float(np.sum(p_power[mask]))
        on = float(np.sum(e_power[mask]))
        attenuation = 10.0 * math.log10(max(off, 1e-30) / max(on, 1e-30))
        rows.append(
            {
                "lower_hz": float(lo),
                "upper_hz": float(hi),
                "off_power": off,
                "on_power": on,
                "attenuation_db": attenuation,
            }
        )
    return rows


def _clone_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    raise TypeError(f"알 수 없는 streaming state: {type(value)!r}")


def _stream_suffix(
    model: HybridANCNet,
    model_input: torch.Tensor,
    states: list[Any],
    *,
    start: int,
    block_samples: int,
) -> tuple[torch.Tensor, list[Any]]:
    outputs = []
    current = states
    for offset in range(int(start), model_input.shape[-1], int(block_samples)):
        output, current = model.streaming_step(
            model_input[..., offset : offset + block_samples], current
        )
        outputs.append(output)
    return torch.cat(outputs, dim=-1), current


def run_deterministic_broadband_g0(
    model_cfg: Mapping[str, Any],
    *,
    primary_delay_samples: int,
    secondary_delay_samples: int,
    handoff_samples: int,
    sample_rate: int = 48_000,
    steps: int = DETERMINISTIC_G0_STEPS,
    seed: int = DETERMINISTIC_G0_SEED,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """실제 Tiny forward/streaming을 짧게 overfit하고 fail-closed receipt를 반환한다."""

    step_count = int(steps)
    if not 1 <= step_count <= 2_000:
        raise ValueError("deterministic G0 steps는 1..2000이어야 합니다")
    seed_value = int(seed)
    target_device = torch.device(device)
    torch.manual_seed(seed_value)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed_value)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model_input, primary, fixture = build_deterministic_g0_fixture(
        model_cfg,
        primary_delay_samples=primary_delay_samples,
        secondary_delay_samples=secondary_delay_samples,
        handoff_samples=handoff_samples,
        sample_rate=sample_rate,
        device=target_device,
    )
    model = HybridANCNet(dict(model_cfg)).to(target_device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.0e-3, weight_decay=0.0
    )
    evaluation_start = int(fixture["evaluation_start_samples"])
    secondary_total = int(
        fixture["normalised_delay_fixture"]["secondary_delay_samples"]
    ) + int(handoff_samples)
    tone_end = int(fixture["mixture_index"])
    mixture_index = tone_end
    impulse_start = int(fixture["impulse_index_start"])
    denominator = (
        primary[:, evaluation_start:].square().sum(dim=-1).clamp_min(1e-12)
    )

    started = time.monotonic()
    final_training_loss = math.inf
    for _ in range(step_count):
        optimizer.zero_grad(set_to_none=True)
        control = model(model_input)[:, 0]
        residual = primary + _delay(control, secondary_total)
        ratios = (
            residual[:, evaluation_start:].square().sum(dim=-1) / denominator
        )
        # tone, simultaneous mixture, 128 impulse phase를 서로 같은 그룹 가중치로 둔다.
        loss = (
            ratios[:tone_end].mean()
            + ratios[mixture_index]
            + ratios[impulse_start:].mean()
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("deterministic G0 loss가 NaN/Inf입니다")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final_training_loss = float(loss.detach())
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    elapsed = time.monotonic() - started

    model.eval()
    with torch.no_grad():
        offline = model(model_input)[:, 0]
        offline_residual = primary + _delay(offline, secondary_total)
        crop_primary = primary[:, evaluation_start:]
        crop_residual = offline_residual[:, evaluation_start:]
        attenuation = _attenuation_db(crop_residual, crop_primary)

        states = model.init_states(batch=model_input.shape[0], device=target_device)
        streaming, _ = _stream_suffix(
            model,
            model_input,
            states,
            start=0,
            block_samples=int(handoff_samples),
        )
        streaming = streaming[:, 0]
        streaming_error = float(torch.max(torch.abs(streaming - offline)))

        # Prefix 2 blocks 뒤 state를 복제하고 같은 suffix를 두 번 재생한다. settle crop으로
        # state 누락을 숨기지 않고 encoder/TCN/LSTM/decoder-tail 전부를 이어받는다.
        prefix_samples = 2 * int(handoff_samples)
        prefix_states = model.init_states(
            batch=model_input.shape[0], device=target_device
        )
        _, prefix_states = _stream_suffix(
            model,
            model_input[..., :prefix_samples],
            prefix_states,
            start=0,
            block_samples=int(handoff_samples),
        )
        suffix_input = model_input[..., prefix_samples:]
        suffix_a, _ = _stream_suffix(
            model,
            suffix_input,
            _clone_state(prefix_states),
            start=0,
            block_samples=int(handoff_samples),
        )
        suffix_b, _ = _stream_suffix(
            model,
            suffix_input,
            _clone_state(prefix_states),
            start=0,
            block_samples=int(handoff_samples),
        )
        # 같은 state를 두 번 복제한 결과끼리만 비교하면 state가 잘못됐어도 항상 0이
        # 된다. 처음부터 중단 없이 계산한 streaming suffix와 prefix 뒤 복원한 state의
        # suffix를 직접 비교해야 실제 resume 경계를 검증할 수 있다. 두 번째 replay는
        # 복제 state 자체의 결정론을 별도 기록한다.
        uninterrupted_suffix = streaming[:, prefix_samples:].unsqueeze(1)
        prefix_resume_error = float(
            torch.max(torch.abs(suffix_a - uninterrupted_suffix))
        )
        cloned_state_replay_error = float(torch.max(torch.abs(suffix_a - suffix_b)))

    attenuation_cpu = attenuation.detach().cpu().numpy()
    tone_frequency = list(fixture["tone_frequencies_hz"])
    tone_rows = [
        {
            "frequency_hz": float(frequency),
            "attenuation_db": float(attenuation_cpu[index]),
        }
        for index, frequency in enumerate(tone_frequency)
    ]
    impulse_values = attenuation_cpu[impulse_start:]
    mixture_primary = crop_primary[mixture_index].detach().cpu().numpy()
    mixture_residual = crop_residual[mixture_index].detach().cpu().numpy()
    mixture_bands = _subband_attenuation(
        mixture_primary, mixture_residual, sample_rate=int(sample_rate)
    )
    polyphase = checkpoint_polyphase_report(
        model.state_dict(), model_cfg, sample_rate=int(sample_rate)
    )

    minimum_tone = min(row["attenuation_db"] for row in tone_rows)
    minimum_mix_band = min(row["attenuation_db"] for row in mixture_bands)
    minimum_impulse = float(np.min(impulse_values))
    output_peak = float(offline.detach().abs().max().cpu())
    saturation_fraction = float(
        (offline.detach().abs() >= float(model.limit) * 0.999).float().mean().cpu()
    )
    checks = {
        "all_tones_at_least_6db": minimum_tone >= 6.0,
        "all_seven_mixture_subbands_at_least_6db": minimum_mix_band >= 6.0,
        "all_128_impulse_residues_positive_attenuation": minimum_impulse > 0.0,
        "limiter_peak_at_most_0_18": output_peak <= 0.18,
        "limiter_saturation_fraction_zero": saturation_fraction == 0.0,
        "offline_streaming_max_abs_error_at_most_1e_5": streaming_error <= 1e-5,
        "prefix_state_matches_uninterrupted_at_most_1e_5": (
            prefix_resume_error <= 1e-5
        ),
        "cloned_state_replay_exact": cloned_state_replay_error == 0.0,
        "positive_branch_polyphase_algebraic_rank_full": bool(
            polyphase["algebraic_probe_passed"]
        ),
    }
    passed = all(checks.values())
    lattice = output_lattice_contract(
        model_cfg,
        sample_rate=int(sample_rate),
        runtime_block_samples=int(handoff_samples),
    )
    return {
        "schema_version": BROADBAND_DETERMINISTIC_G0_SCHEMA,
        "status": "PASS_STRUCTURAL_DIAGNOSTIC" if passed else "FAIL_STRUCTURAL_DIAGNOSTIC",
        "canonical_training_admitted": False,
        "canonical_block_reason": (
            "delay-only unity fixture는 fullband measured causal P/S, exact physical prefix, "
            "네 source family evidence를 대신하지 않습니다"
        ),
        "seed": seed_value,
        "device": str(target_device),
        "training_steps": step_count,
        "training_elapsed_seconds": elapsed,
        "final_training_loss": final_training_loss,
        "determinism": {
            "torch_deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        },
        "parameter_count": parameter_count(model),
        "fixture": fixture,
        "timing_interpretation": {
            "maximum_intra_hop_future_dependency_samples": lattice[
                "maximum_intra_hop_future_dependency_samples"
            ],
            "runtime_block_samples_available_before_output": int(handoff_samples),
            "availability_margin_samples": int(handoff_samples)
            - int(lattice["maximum_intra_hop_future_dependency_samples"]),
            "sample_causal_without_block_handoff": lattice[
                "sample_causal_without_runtime_handoff"
            ],
        },
        "tone_attenuation": tone_rows,
        "minimum_tone_attenuation_db": minimum_tone,
        "simultaneous_mixture_subband_attenuation": mixture_bands,
        "minimum_simultaneous_mixture_subband_attenuation_db": minimum_mix_band,
        "impulse_residue_attenuation_db": [float(value) for value in impulse_values],
        "minimum_impulse_residue_attenuation_db": minimum_impulse,
        "control_abs_peak": output_peak,
        "limiter_limit": float(model.limit),
        "limiter_saturation_fraction": saturation_fraction,
        "offline_streaming_max_abs_error": streaming_error,
        "prefix_state_replay_max_abs_error": prefix_resume_error,
        "cloned_state_replay_max_abs_error": cloned_state_replay_error,
        "polyphase_weight_only_diagnostic": polyphase,
        "checks": checks,
        "required_physical_broadband_g0": broadband_g0_gate_spec(),
    }


__all__ = [
    "BROADBAND_DETERMINISTIC_G0_SCHEMA",
    "DETERMINISTIC_G0_AMPLITUDE",
    "DETERMINISTIC_G0_LENGTH",
    "DETERMINISTIC_G0_SEED",
    "DETERMINISTIC_G0_STEPS",
    "build_deterministic_g0_fixture",
    "run_deterministic_broadband_g0",
]
